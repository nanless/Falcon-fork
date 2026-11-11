import json

import pytest
import torch

from tools.streaming import cache_teacher


class _FakeDataset:
    def __init__(self, path, *, seed, crop, augment):
        self.samples = json.loads(open(path, encoding="utf-8").read())["samples"]
        self.shards = []
        self.split = "tr"
        self.incomplete = True
        self.seed, self.crop, self.augment = seed, crop, augment
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        value = float(self.samples[index] + self.epoch)
        mixture = torch.full((cache_teacher.SEGMENT_SAMPLES,), value)
        return mixture, mixture, torch.full((1, 1, 1), value)

    def set_epoch(self, epoch):
        self.epoch = epoch


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_teacher, "RealSSATarDataset", _FakeDataset)
    monkeypatch.setattr(cache_teacher, "load_teacher", lambda checkpoint, device: object())
    monkeypatch.setattr(cache_teacher, "predict_teacher",
                        lambda teacher, mixture, video, microbatch_size: mixture * 2)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"samples": [10, 20, 30, 40, 50]}))
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    root = tmp_path / "cache"
    cache_teacher.prepare(root, index, teacher, seed=17, world_size=2, microbatch_size=2)
    return root, index, teacher


def test_cache_requires_all_shards_and_returns_fixed_epoch_teacher(plan):
    root, index, teacher = plan
    cache_teacher.generate_shard(root, 1, device="cpu", workers=0)
    with pytest.raises(FileNotFoundError):
        cache_teacher.finalize(root)
    with pytest.raises(FileExistsError):
        cache_teacher.generate_shard(root, 1, device="cpu", workers=0)
    cache_teacher.generate_shard(root, 0, device="cpu", workers=0)
    manifest = cache_teacher.finalize(root)
    assert [(s["start"], s["stop"]) for s in manifest["shards"]] == [(0, 2), (2, 5)]
    dataset = _FakeDataset(index, seed=17, crop="random", augment=True)
    cached = cache_teacher.CachedTeacherDataset(dataset, root, index_path=index, teacher_checkpoint=teacher)
    cached.set_epoch(92)
    assert dataset.epoch == 0
    for i in (4, 2, 0, 1, 3):
        mixture, _, _, teacher_output = cached[i]
        torch.testing.assert_close(teacher_output, mixture * 2)
    cache_teacher._read_final(root, full_hashes=True)
    with pytest.raises(FileExistsError):
        cache_teacher.finalize(root)
    with pytest.raises(ValueError, match="seed"):
        cache_teacher.CachedTeacherDataset(
            _FakeDataset(index, seed=2, crop="random", augment=True), root,
            index_path=index, teacher_checkpoint=teacher,
        )


def test_cache_refuses_changed_index_and_checks_full_shard_checksum(plan):
    root, index, teacher = plan
    for rank in (0, 1):
        cache_teacher.generate_shard(root, rank, device="cpu", workers=0)
    cache_teacher.finalize(root)
    wrong_index = index.with_name("wrong.json")
    wrong_index.write_text(json.dumps({"samples": [20, 10, 30, 40, 50]}))
    with pytest.raises(ValueError, match="index SHA256"):
        cache_teacher._read_final(root, index_path=wrong_index)
    path = root / "shard-0000" / "teacher.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    # Match the actual filesystem timestamp so the size/time checks pass
    # and the byte-level check is the one that rejects this payload.
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["file_mtime_ns"] = path.stat().st_mtime_ns
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shard SHA256"):
        cache_teacher._read_final(root, full_hashes=True)
