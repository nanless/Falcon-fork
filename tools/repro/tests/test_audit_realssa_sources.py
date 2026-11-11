"""Check source-overlap auditing and privacy with synthetic tar shards."""
import builtins
import io
import json
import tarfile

import pytest

from tools.streaming.audit_realssa_sources import audit
from tools.streaming.realssa_tar import build_index


def shard(path, samples):
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w") as archive:
        for sample, target in samples:
            for suffix, blob in (("mix.wav", b"x"), ("target.wav", b"y"), ("mouths.pt", b"z"),
                                 ("json", json.dumps({"target": target}).encode())):
                info = tarfile.TarInfo(sample + "." + suffix)
                info.size = len(blob)
                archive.addfile(info, io.BytesIO(blob))


def test_full_audit_anonymous_and_resumable(tmp_path, monkeypatch):
    src = tmp_path / "raw"
    shard(src / "tr" / "a.tar", [("a", "/private/AAAAAAAAAAA/one.wav"), ("b", "/private/BBBBBBBBBBB/two.wav")])
    shard(src / "cv" / "a.tar", [("c", "/private/AAAAAAAAAAA/one.wav")])
    shard(src / "tt" / "a.tar", [("d", "/private/BBBBBBBBBBB/three.wav"), ("e", None)])
    indices = {}
    for split in ("tr", "cv", "tt"):
        indices[split] = tmp_path / (split + ".json")
        build_index(src, split, indices[split])
    cache = tmp_path / "cache"
    output = tmp_path / "first"
    report = audit(indices, output, workers=2, cache_dir=cache)
    assert report["pairwise"]["tr_cv"] == {"target_path_intersection": 1, "tr_rows_with_shared_target": 1, "cv_rows_with_shared_target": 1, "video_directory_intersection": 1}
    assert report["pairwise"]["tr_tt"] == {"target_path_intersection": 0, "tr_rows_with_shared_target": 0, "tt_rows_with_shared_target": 0, "video_directory_intersection": 1}
    assert report["splits"]["tt"]["errors"] == {"missing_target": 1}
    assert report["splits"]["tt"]["checked"] == 2
    for file in [output / "report.json", *cache.glob("*.json")]:
        assert "/private/" not in file.read_text() and "AAAAAAAAAAA" not in file.read_text()
    original = builtins.open
    def reject_shards(path, *args, **kwargs):
        if str(path).endswith(".tar"):
            raise AssertionError("resume must use cached per-shard summaries")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", reject_shards)
    resumed = audit(indices, tmp_path / "second", workers=2, cache_dir=cache)
    assert resumed["pairwise"] == report["pairwise"]
    assert resumed["splits"] == report["splits"]
    with pytest.raises(FileExistsError):
        audit(indices, output, workers=2, cache_dir=cache)


def test_changed_shard_records_coverage_failure_anonymously(tmp_path):
    src = tmp_path / "raw"
    for split in ("tr", "cv", "tt"):
        shard(src / split / "a.tar", [(split, "/private/AAAAAAAAAAA/one.wav")])
    indices = {}
    for split in ("tr", "cv", "tt"):
        indices[split] = tmp_path / (split + ".json")
        build_index(src, split, indices[split])
    with (src / "tr" / "a.tar").open("ab") as stream:
        stream.write(b"changed")
    out = tmp_path / "results"
    report = audit(indices, out, workers=2)
    assert report["splits"]["tr"]["checked"] == 1
    assert report["splits"]["tr"]["valid_target"] == 0
    assert report["splits"]["tr"]["errors"] == {"shard_ValueError": 1}
    assert report["failed_shards"][0]["affected_samples"] == 1
    assert "/private/" not in (out / "report.json").read_text()


def test_shard_changed_during_audit_is_reported_and_not_cached(tmp_path, monkeypatch):
    src = tmp_path / "raw"
    for split in ("tr", "cv", "tt"):
        shard(src / split / "a.tar", [(split, "/private/AAAAAAAAAAA/one.wav")])
    indices = {}
    for split in ("tr", "cv", "tt"):
        indices[split] = tmp_path / (split + ".json")
        build_index(src, split, indices[split])
    target = src / "tr" / "a.tar"
    original = builtins.open
    class MutatingReader:
        def __init__(self, path, *args, **kwargs):
            self.file = original(path, *args, **kwargs)
            self.mutated = False
        def __enter__(self): return self
        def __exit__(self, *args): self.file.close()
        def seek(self, *args): return self.file.seek(*args)
        def read(self, *args):
            if not self.mutated:
                with original(target, "ab") as stream:
                    stream.write(b"changed")
                self.mutated = True
            return self.file.read(*args)
    def open_mutating(path, *args, **kwargs):
        if str(path) == str(target):
            return MutatingReader(path, *args, **kwargs)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_mutating)
    out = tmp_path / "result"
    result = audit(indices, out, workers=1)
    assert result["splits"]["tr"]["valid_target"] == 0
    assert result["splits"]["tr"]["errors"] == {"shard_ValueError": 1}
    assert len(list((out / "cache").glob("*.json"))) == 2
