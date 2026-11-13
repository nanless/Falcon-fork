import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools.streaming.exclude_realssa import (
    IndexedSamples,
    batch_exclusions,
    load_exclusions,
    require_retained_global,
    training_mean,
)


@pytest.fixture
def manifest(tmp_path):
    wav = b"candidate audio bytes"
    tar = tmp_path / "sample.tar"
    tar.write_bytes(b"header" + wav + b"padding")
    shard = {"path": str(tar), "size": tar.stat().st_size, "mtime_ns": tar.stat().st_mtime_ns}
    sample = {"id": "rail", "shard": 0, "fields": {"mix": [6, len(wav)]}}

    class Samples:
        samples = [sample]
        shards = [shard]

        def __len__(self):
            return len(self.samples)

    payload = {
        "index_sha256": "expected", "count": 1,
        "predicate": "raw 16000 Hz PCM16 ch0 equals -32768 for every sample",
        "members": [{"index": 0, "id": "rail", "shard": shard, "mix_location": [6, len(wav)],
                     "mix_wav_sha256": hashlib.sha256(wav).hexdigest()}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return SimpleNamespace(dataset=Samples(), path=path, tar=tar, payload=payload)


def test_manifest_binds_index_row_and_raw_audio(manifest):
    excluded, digest = load_exclusions(manifest.path, "expected", manifest.dataset)
    assert excluded == {0: "rail"}
    assert digest == hashlib.sha256(manifest.path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="index SHA"):
        load_exclusions(manifest.path, "changed index", manifest.dataset)
    manifest.payload["members"][0]["id"] = "other"
    manifest.path.write_text(json.dumps(manifest.payload))
    with pytest.raises(ValueError, match="row differs"):
        load_exclusions(manifest.path, "expected", manifest.dataset)
    manifest.payload["members"][0]["id"] = "rail"
    manifest.path.write_text(json.dumps(manifest.payload))
    manifest.tar.write_bytes(manifest.tar.read_bytes().replace(b"audio", b"other"))
    with pytest.raises(ValueError, match="WAV SHA differs"):
        load_exclusions(manifest.path, "expected", manifest.dataset)


def test_indices_and_original_loss_denominator():
    originals = [(torch.tensor([1]), torch.tensor([2]), torch.tensor([3])),
                 (torch.tensor([4]), torch.tensor([5]), torch.tensor([6]))]
    indexed = IndexedSamples(originals)
    assert len(indexed) == len(originals)
    assert indexed[1] == (*originals[1], 1)
    values = torch.tensor([2., 10.], requires_grad=True)
    assert torch.equal(training_mean(values), values.mean())
    excluded, hits = batch_exclusions(torch.tensor([1, 0]), {0: "rail"}, "cpu")
    assert hits == {0: "rail"}
    require_retained_global(excluded)
    reduced = training_mean(values, excluded)
    assert reduced.item() == 1.
    reduced.backward()
    assert values.grad.tolist() == [0.5, 0.0]


def test_all_excluded_fail_closed(monkeypatch):
    excluded = torch.ones(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="no unexcluded samples"):
        require_retained_global(excluded)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda total: total.add_(1))
    require_retained_global(excluded)
