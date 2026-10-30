"""Tests for tools/repro/lrs2_2mix_to_falcon.py using a tiny fake LRS2-2Mix tree.

No real LRS2 data is required: the fixture reproduces the documented layout
(`audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}`, `mouths/<id>.npz`, `raw_audio/...`)
with 2 s 16 kHz mono PCM16 wavs and uint8 `[50,96,96]` mouth npz files.

Run with the project environment::

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_lrs2_convert.py -q
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
CONVERTER = REPO / "tools" / "repro" / "lrs2_2mix_to_falcon.py"
SR = 16000
FRAME = SR // 25  # 640
N = 2 * SR  # 2.000 s
MOUTH_SHAPE = (50, 96, 96)


# ---------------------------------------------------------------------------
# fake source tree
# ---------------------------------------------------------------------------
def _wav(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), SR, subtype="PCM_16")


def _tone(n: int, freq: float, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t) + 0.001 * rng.randn(n)).astype(np.float32)


def _mouth_npz(path: Path, seed: int) -> None:
    rng = np.random.RandomState(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=rng.randint(0, 256, size=MOUTH_SHAPE, dtype=np.uint8))


def _make_fake_src(root: Path) -> Path:
    """Create 2 samples in tr/cv/tt with mix == s1 + s2 and matching mouths."""
    specs = {
        "tr": [("1000_00001", "2000_00002"), ("1000_00003", "2000_00004")],
        "cv": [("1000_00005", "2000_00006")],
        "tt": [("1000_00007", "2000_00008")],
    }
    for split, pairs in specs.items():
        base = root / "audio" / "wav16k" / "min" / split
        for idx, (s1, s2) in enumerate(pairs):
            stem = f"{s1}_1.5_{s2}_-1.5"
            a = _tone(N, 180 + 20 * idx, seed=10 * idx + 1)
            b = _tone(N, 300 + 30 * idx, amp=0.08, seed=10 * idx + 2)
            _wav(base / "s1" / f"{stem}.wav", a)
            _wav(base / "s2" / f"{stem}.wav", b)
            _wav(base / "mix" / f"{stem}.wav", a + b)
            _mouth_npz(root / "mouths" / f"{s1}.npz", seed=idx + 1)
            _mouth_npz(root / "mouths" / f"{s2}.npz", seed=idx + 101)
    # raw audio: 2 s clips from different speakers (approximate negatives)
    for raw_split, ids in {"train": ["9000_00001", "9000_00002"],
                           "val": ["9000_00003"], "test": ["9000_00004"]}.items():
        for j, clip_id in enumerate(ids):
            _wav(root / "raw_audio" / raw_split / f"{clip_id}.wav",
                 _tone(N, 400 + 10 * j, amp=0.07, seed=200 + j))
    return root


@pytest.fixture(scope="module")
def fake_src(tmp_path_factory) -> Path:
    return _make_fake_src(tmp_path_factory.mktemp("lrs2_rebuild"))


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, str(CONVERTER), *args],
                          cwd=str(REPO), capture_output=True, text=True, timeout=timeout, env=env)


def _load_json(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_fallback_conversion_layout_and_contract(fake_src: Path, tmp_path: Path):
    out = tmp_path / "falcon"
    res = _run(["--src", str(fake_src), "--out", str(out), "--jobs", "2",
                "--validate-samples", "10", "--limit", "2"])
    assert res.returncode == 0, res.stderr
    assert "INDEX READY" in res.stdout
    assert (out / "conversion_stats.json").is_file()

    for split, n in {"tr": 2, "cv": 1, "tt": 1}.items():
        sdir = out / split
        assert len(list((sdir / "mix").glob("*.wav"))) == n
        assert len(list((sdir / "target").glob("*.wav"))) == n
        mix_json = _load_json(sdir / "mix.json")
        target_json = _load_json(sdir / "target.json")
        near_json = _load_json(sdir / "near.json")
        assert len(mix_json) == len(target_json) == len(near_json) == n
        assert all(len(e) == 2 for e in mix_json)
        assert all(len(e) == 3 for e in target_json)
        # fallback near = [[mix_path, n_samples]]
        assert near_json == [[e] for e in mix_json]
        for (mix_path, nsamp), (tgt_path, mouth_path, tn) in zip(mix_json, target_json):
            assert nsamp == tn == N
            assert Path(mix_path).is_file() and Path(tgt_path).is_file()
            assert Path(mouth_path).name.endswith(".pt")
            mouths = torch.load(mouth_path, map_location="cpu", weights_only=True)
            assert mouths.dtype == torch.uint8
            assert tuple(mouths.shape) == (50, 96, 96, 3)
            # all three channels are copies of the npz grayscale frames
            assert torch.equal(mouths[..., 0], mouths[..., 1]) and torch.equal(mouths[..., 0], mouths[..., 2])
            # target audio is a byte-identical copy of s1
            stem = Path(mix_path).stem
            s1 = fake_src / "audio/wav16k/min" / split / "s1" / f"{stem}.wav"
            assert Path(tgt_path).read_bytes() == s1.read_bytes()
            # mix == s1 + s2 within PCM16 rounding
            mix, _ = sf.read(mix_path, dtype="float32")
            a, _ = sf.read(str(s1), dtype="float32")
            b, _ = sf.read(str(fake_src / "audio/wav16k/min" / split / "s2" / f"{stem}.wav"),
                           dtype="float32")
            assert np.max(np.abs(mix - (a + b))) <= 1e-3
    stats = _load_json(out / "conversion_stats.json")
    assert stats["target"] == "s1" and stats["near_mode"] == "fallback"
    for split in ("tr", "cv", "tt"):
        assert stats["validation"][split]["problems"] == []
        assert stats["validation"][split]["checked"] > 0


def test_target_s2_uses_s2_audio_and_mouths(fake_src: Path, tmp_path: Path):
    out = tmp_path / "falcon_s2"
    res = _run(["--src", str(fake_src), "--out", str(out), "--target", "s2",
                "--jobs", "2", "--validate-samples", "5"])
    assert res.returncode == 0, res.stderr
    target_json = _load_json(out / "tr" / "target.json")
    assert target_json
    for tgt_path, mouth_path, n in target_json:
        stem = Path(tgt_path).stem
        s2 = fake_src / "audio/wav16k/min/tr/s2" / f"{stem}.wav"
        assert Path(tgt_path).read_bytes() == s2.read_bytes()
        # mouth file named after s2 utterance id (third field of the stem recipe)
        import re
        m = re.match(r"^(\d+_\d+)_([-+]?\d+(?:\.\d+)?)_(\d+_\d+)_([-+]?\d+(?:\.\d+)?)$", stem)
        assert m is not None
        assert Path(mouth_path).stem == m.group(3)


def test_near_mode_raw_audio_uses_other_speakers(fake_src: Path, tmp_path: Path):
    out = tmp_path / "falcon_raw"
    res = _run(["--src", str(fake_src), "--out", str(out), "--near-mode", "raw_audio",
                "--near-per-sample", "1", "--jobs", "2", "--validate-samples", "5"])
    assert res.returncode == 0, res.stderr
    for split, raw_split in {"tr": "train", "cv": "val", "tt": "test"}.items():
        mix_json = _load_json(out / split / "mix.json")
        near_json = _load_json(out / split / "near.json")
        assert len(near_json) == len(mix_json)
        for mix_entry, near_entry in zip(mix_json, near_json):
            assert len(near_entry) == 1
            near_path, near_n = near_entry[0]
            assert raw_split in near_path and near_path.endswith(".wav")
            assert near_n == N
            info = sf.info(near_path)
            assert (info.samplerate, info.channels, info.frames) == (SR, 1, N)
            stem = Path(mix_entry[0]).stem
            parts = stem.split("_")
            speakers = {parts[0], parts[2]}
            assert Path(near_path).stem.split("_")[0] not in speakers
    stats = _load_json(out / "conversion_stats.json")
    assert stats["near_mode"] == "raw_audio"
    assert all(not v["problems"] for v in stats["validation"].values())


def test_dry_run_writes_nothing(fake_src: Path, tmp_path: Path):
    out = tmp_path / "should_not_exist"
    res = _run(["--src", str(fake_src), "--out", str(out), "--dry-run"])
    assert res.returncode == 0, res.stderr
    assert "dry-run: nothing written" in res.stdout
    assert not out.exists()


def test_reentrant_second_run_skips_existing(fake_src: Path, tmp_path: Path):
    out = tmp_path / "falcon_reentrant"
    first = _run(["--src", str(fake_src), "--out", str(out), "--jobs", "2", "--validate-samples", "0"])
    assert first.returncode == 0, first.stderr
    before = {p: p.read_bytes() for p in sorted((out / "tr").glob("*.json"))}
    second = _run(["--src", str(fake_src), "--out", str(out), "--jobs", "2", "--validate-samples", "0"])
    assert second.returncode == 0, second.stderr
    stats = _load_json(out / "conversion_stats.json")
    assert stats["conversions"]["tr"]["skipped_existing"] == 2
    assert {p: p.read_bytes() for p in sorted((out / "tr").glob("*.json"))} == before


def test_limit_and_missing_mouth_are_counted(fake_src: Path, tmp_path: Path):
    # copy fixture so we can add a sample whose target mouth npz is missing
    import shutil
    src = tmp_path / "src_bad"
    shutil.copytree(fake_src, src)
    bad_stem = "1000_00009_1.5_2000_00010_-1.5"
    a = _tone(N, 220, seed=7)
    b = _tone(N, 260, amp=0.08, seed=8)
    base = src / "audio/wav16k/min/tr"
    _wav(base / "s1" / f"{bad_stem}.wav", a)
    _wav(base / "s2" / f"{bad_stem}.wav", b)
    _wav(base / "mix" / f"{bad_stem}.wav", a + b)
    # no mouths/1000_00009.npz on purpose
    out = tmp_path / "falcon_bad"
    res = _run(["--src", str(src), "--out", str(out), "--jobs", "2",
                "--splits", "tr", "--validate-samples", "0"])
    assert res.returncode == 0, res.stderr
    stats = _load_json(out / "conversion_stats.json")
    assert stats["conversions"]["tr"]["missing_mouth_npz"] == 1
    assert len(_load_json(out / "tr" / "mix.json")) == 2  # the 2 valid samples


def test_mix_sum_mismatch_is_reported(fake_src: Path, tmp_path: Path):
    import shutil
    src = tmp_path / "src_mismatch"
    shutil.copytree(fake_src, src)
    base = src / "audio/wav16k/min/tr"
    stem = "1000_00001_1.5_2000_00002_-1.5"
    mix, _ = sf.read(str(base / "mix" / f"{stem}.wav"), dtype="float32")
    _wav(base / "mix" / f"{stem}.wav", mix + 0.05)  # corrupt the recipe
    out = tmp_path / "falcon_mismatch"
    res = _run(["--src", str(src), "--out", str(out), "--splits", "tr", "--jobs", "2",
                "--validate-samples", "10"])
    assert res.returncode == 0, res.stderr
    stats = _load_json(out / "conversion_stats.json")
    problems = stats["validation"]["tr"]["problems"]
    assert any("mix-(s1+s2)" in p for p in problems)


# ---------------------------------------------------------------------------
# --near-mode other-speaker (K=1, the other source waveform of the same mixture)
# ---------------------------------------------------------------------------
def test_near_mode_other_speaker_target_s1(fake_src: Path, tmp_path: Path):
    """target=s1 -> near = the same mixture's s2 wav (K=1, deterministic)."""
    out = tmp_path / "falcon_other_s1"
    res = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tr", "--limit", "2",
                "--near-mode", "other-speaker", "--jobs", "2", "--validate-samples", "10"])
    assert res.returncode == 0, res.stderr
    mix_json = _load_json(out / "tr" / "mix.json")
    near_json = _load_json(out / "tr" / "near.json")
    assert len(mix_json) == len(near_json) == 2
    for (mix_path, _n), near_entry in zip(mix_json, near_json):
        # schema must match the on-disk dataset: [[abs_other_speaker_wav, n_samples]]
        assert isinstance(near_entry, list) and len(near_entry) == 1
        near_path, near_n = near_entry[0]
        assert isinstance(near_path, str) and Path(near_path).is_absolute()
        assert near_n == N
        stem = Path(mix_path).stem
        expect = (fake_src / "audio/wav16k/min/tr/s2" / f"{stem}.wav").resolve()
        assert Path(near_path) == expect
        assert Path(near_path) != Path(mix_path)
        info = sf.info(near_path)
        assert (info.samplerate, info.channels, info.frames) == (SR, 1, N)
        # the near waveform is exactly s2 (not the mixture)
        near, _ = sf.read(near_path, dtype="float32")
        s2, _ = sf.read(str(expect), dtype="float32")
        mix, _ = sf.read(mix_path, dtype="float32")
        assert np.max(np.abs(near - s2)) == 0.0
        assert np.max(np.abs(near - mix)) > 1e-3
    stats = _load_json(out / "conversion_stats.json")
    assert stats["near_mode"] == "other-speaker" and stats["target"] == "s1"
    assert stats["validation"]["tr"]["problems"] == []
    assert stats["validation"]["tr"]["near_ok"] == 2


def test_near_mode_other_speaker_target_s2(fake_src: Path, tmp_path: Path):
    """target=s2 -> near = the same mixture's s1 wav; target audio/mouth follow s2."""
    out = tmp_path / "falcon_other_s2"
    res = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tr", "--limit", "2",
                "--target", "s2", "--near-mode", "other-speaker", "--jobs", "2",
                "--validate-samples", "10"])
    assert res.returncode == 0, res.stderr
    target_json = _load_json(out / "tr" / "target.json")
    near_json = _load_json(out / "tr" / "near.json")
    assert len(target_json) == len(near_json) == 2
    for (tgt_path, mouth_path, _n), near_entry in zip(target_json, near_json):
        stem = Path(tgt_path).stem
        s1 = (fake_src / "audio/wav16k/min/tr/s1" / f"{stem}.wav").resolve()
        s2 = (fake_src / "audio/wav16k/min/tr/s2" / f"{stem}.wav").resolve()
        assert Path(tgt_path).read_bytes() == s2.read_bytes()
        parts = stem.split("_")
        assert Path(mouth_path).stem == "_".join(parts[3:5])  # mouth belongs to the s2 utterance
        assert len(near_entry) == 1
        assert Path(near_entry[0][0]) == s1
        assert near_entry[0][1] == N
    stats = _load_json(out / "conversion_stats.json")
    assert stats["near_mode"] == "other-speaker" and stats["target"] == "s2"
    assert all(not v["problems"] for v in stats["validation"].values())


def test_near_mode_other_speaker_dry_run_reports_counts(fake_src: Path, tmp_path: Path):
    out = tmp_path / "other_speaker_should_not_exist"
    res = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tr",
                "--near-mode", "other-speaker", "--dry-run"])
    assert res.returncode == 0, res.stderr
    assert "near-mode=other-speaker" in res.stdout
    assert "planned near entries" in res.stdout
    assert "target=s1 -> near=s2" in res.stdout
    assert "dry-run: nothing written" in res.stdout
    assert not out.exists()
