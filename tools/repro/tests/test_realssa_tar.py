"""Offline tar-index regression tests; use synthetic data only."""
import io
import json
import os
import tarfile

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from tools.streaming.realssa_tar import RealSSATarDataset, build_index


def add(archive, name, blob):
    info = tarfile.TarInfo(name)
    info.size = len(blob)
    archive.addfile(info, io.BytesIO(blob))


def make_shard(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for key, n, omit in samples:
            ramp = np.arange(n * 640, dtype=np.float32) / 100000
            wav = np.stack((ramp, ramp * 2, ramp * 3, ramp * 4), axis=1)
            buffer = io.BytesIO()
            sf.write(buffer, wav, 16000, format="WAV", subtype="FLOAT")
            sound = buffer.getvalue()
            mouths = torch.arange(n, dtype=torch.uint8)[:, None, None, None].expand(n, 96, 96, 3).clone()
            buffer = io.BytesIO()
            torch.save(mouths, buffer)
            fields = {"mix.wav": sound, "target.wav": sound, "mouths.pt": buffer.getvalue(), "json": b"{}"}
            for kind, content in fields.items():
                if kind != omit:
                    add(archive, f"{key}.{kind}", content)


def test_index_crop_channel_and_incomplete_reporting(tmp_path):
    root = tmp_path / "src"
    make_shard(root / "tr" / "tr-000000.tar", [("a", 60, None), ("missing", 55, "json")])
    make_shard(root / "tr" / "tr-000001.tar", [("b", 20, None)])
    path = tmp_path / "indices" / "train.json"
    info = build_index(root, "tr", path)
    assert info["sample_count"] == 2 and info["incomplete_count"] == 1 and info["shard_count"] == 2
    index = json.loads(path.read_text())
    assert index["incomplete"] == [{"id": "missing", "shard": 0, "missing": ["meta"]}]
    ds = RealSSATarDataset(path, crop="center")
    mix, target, video = ds[0]
    assert ds.split == "tr" and mix.shape == target.shape == (32000,) and video.shape == (50, 88, 88)
    torch.testing.assert_close(mix, target)
    assert mix[0].item() == pytest.approx(5 * 640 / 100000)
    assert video[0, 0, 0].item() == pytest.approx((4 / 255 - .421) / .165, abs=2e-5)
    short, _, short_video = ds[1]
    assert torch.count_nonzero(short[20*640:]) == 0
    torch.testing.assert_close(short_video[-1], torch.full_like(short_video[-1], -.421/.165))
    with pytest.raises(FileExistsError):
        build_index(root, "tr", path)


def test_random_epoch_and_worker_determinism(tmp_path):
    root = tmp_path / "src"
    make_shard(root / "cv" / "cv-000000.tar", [(str(i), 100, None) for i in range(4)])
    path = tmp_path / "cv.json"
    build_index(root, "cv", path)
    ds = RealSSATarDataset(path, seed=42, crop="random")
    ds.set_epoch(7)
    first = [float(ds[i][0][0]) for i in range(len(ds))]
    workers = [float(batch[0][0, 0]) for batch in DataLoader(ds, batch_size=1, num_workers=2)]
    assert workers == first
    ds.set_epoch(8)
    second = [float(ds[i][0][0]) for i in range(len(ds))]
    assert first != second
    ds.set_epoch(7)
    assert [float(ds[i][0][0]) for i in range(len(ds))] == first


def test_duplicate_or_changed_shard_is_rejected(tmp_path):
    root = tmp_path / "src"
    first = root / "tr" / "tr-000000.tar"
    second = root / "tr" / "tr-000001.tar"
    make_shard(first, [("same", 50, None)])
    make_shard(second, [("same", 50, None)])
    with pytest.raises(ValueError, match="duplicate sample ID"):
        build_index(root, "tr", tmp_path / "bad.json")
    second.unlink()
    index = tmp_path / "valid.json"
    build_index(root, "tr", index)
    with first.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="indexed tar changed"):
        RealSSATarDataset(index)


def test_visual_preprocessing_matches_original_train_and_val(tmp_path, monkeypatch):
    import random
    from torchvision.transforms.functional import rgb_to_grayscale
    from look2hear.datas.transform import get_preprocessing_pipelines

    root = tmp_path / "src"
    shard = root / "tr" / "tr-000000.tar"
    shard.parent.mkdir(parents=True)
    frame = torch.arange(96 * 96 * 3, dtype=torch.int32).reshape(96, 96, 3).remainder(256).to(torch.uint8)
    video = frame.unsqueeze(0).expand(50, -1, -1, -1).clone()
    buffer = io.BytesIO(); torch.save(video, buffer); vid = buffer.getvalue()
    buffer = io.BytesIO(); sf.write(buffer, np.zeros((32000, 4), np.float32), 16000, format="WAV", subtype="FLOAT"); wav = buffer.getvalue()
    with tarfile.open(shard, "w") as archive:
        for suffix, data in (("mix.wav", wav), ("target.wav", wav), ("mouths.pt", vid), ("json", b"{}")):
            add(archive, "x." + suffix, data)
    path = tmp_path / "index.json"; build_index(root, "tr", path)
    gray = rgb_to_grayscale(video.permute(0, 3, 1, 2)).squeeze(1).numpy()
    pipeline = get_preprocessing_pipelines()
    val = RealSSATarDataset(path, seed=42, crop="start", augment=False)[0][2]
    torch.testing.assert_close(val, torch.from_numpy(pipeline["val"](gray)).float(), rtol=1e-6, atol=1e-6)
    seeded = random.Random(42 << 64)
    left, top, flip = seeded.randint(0, 8), seeded.randint(0, 8), seeded.random()
    values = iter((left, top))
    monkeypatch.setattr("look2hear.datas.transform.random.randint", lambda a, b: next(values))
    monkeypatch.setattr("look2hear.datas.transform.random.random", lambda: flip)
    ref = torch.from_numpy(pipeline["train"](gray)).float()
    actual = RealSSATarDataset(path, seed=42, crop="start", augment=True)[0][2]
    torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-6)


def test_parallel_shard_scan_cache_resume_and_stale_source(tmp_path, monkeypatch):
    root = tmp_path / "src"
    for i in range(3):
        make_shard(root / "tr" / f"tr-{i:06d}.tar", [(f"sample{i}", 51, None)])
    cache = tmp_path / "cache"
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    info = build_index(root, "tr", first, workers=2, cache_dir=cache)
    assert info["sample_count"] == 3 and len(list(cache.glob("*.json"))) == 3
    original = tarfile.open
    monkeypatch.setattr(tarfile, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse cached shards")))
    again = build_index(root, "tr", second, workers=2, cache_dir=cache)
    assert again == info
    assert json.loads(first.read_text()) == json.loads(second.read_text())
    monkeypatch.setattr(tarfile, "open", original)
    # Any changed source identity is scanned again, never accepted from old cache.
    shard = root / "tr" / "tr-000000.tar"
    with shard.open("ab") as stream:
        stream.write(b"changed")
    third = tmp_path / "third.json"
    build_index(root, "tr", third, workers=2, cache_dir=cache)
    assert len(list(cache.glob("*.json"))) == 4


def test_failed_index_write_never_publishes_partial_file(tmp_path, monkeypatch):
    root = tmp_path / "src"
    make_shard(root / "tr" / "tr-000000.tar", [("sample", 50, None)])
    dest = tmp_path / "out" / "index.json"
    def fail_dump(*args, **kwargs):
        args[1].write("partial")
        raise OSError("disk full")
    monkeypatch.setattr("tools.streaming.realssa_tar.json.dump", fail_dump)
    with pytest.raises(OSError, match="disk full"):
        build_index(root, "tr", dest)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []
