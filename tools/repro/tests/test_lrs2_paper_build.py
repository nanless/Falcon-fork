"""Tests for tools/repro/phase2/build_lrs2_paper_data.py (review A5-1 fix + build gate).

Synthetic LRS2-2Mix tree only (2 s 16 kHz mono, mouths uint8 [50,96,96]); no real
26 GB data required and no full build is run here.

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_lrs2_paper_build.py -q
"""

from __future__ import annotations

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
BUILDER = REPO / "tools" / "repro" / "phase2" / "build_lrs2_paper_data.py"
SR = 16000
FRAME = SR // 25  # 640
N = 2 * SR  # 32000 = 50 frames
MOUTH_SHAPE = (50, 96, 96)


# ---------------------------------------------------------------------------
# synthetic source tree
# ---------------------------------------------------------------------------
def _tone(n: int, freq: float, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t) + 0.001 * rng.randn(n)).astype(np.float32)


def _wav(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), SR, subtype="PCM_16")


def _mouth_npz(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    np.savez(path, data=rng.randint(0, 256, size=MOUTH_SHAPE, dtype=np.uint8))


def _make_fake_src(root: Path) -> Path:
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
    return root


@pytest.fixture()
def fake_src(tmp_path: Path) -> Path:
    return _make_fake_src(tmp_path / "lrs2_rebuild")


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, str(BUILDER), *args],
                          cwd=str(REPO), capture_output=True, text=True, timeout=timeout, env=env)


def _load(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# --copy-wavs: target/near must never collapse onto the same file (A5-1)
# ---------------------------------------------------------------------------
def test_copy_wavs_layout_distinct_target_near_and_gate(fake_src: Path, tmp_path: Path):
    out = tmp_path / "paper_copy"
    res = _run(["--src", str(fake_src), "--out", str(out), "--copy-wavs",
                "--limit", "2", "--jobs", "2"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "== gate: expected entries" in res.stdout
    stats = _load(out / "conversion_stats.json")
    assert stats["gate"]["ok"] is True, stats["gate"]["problems"]
    assert stats["gate"]["problems"] == []
    assert stats["copy_wavs"] is True

    n_tr = len(_load(out / "tr" / "mix.json"))
    assert n_tr == 4  # 2 mixtures x bidirectional
    for split in ("tr", "cv", "tt"):
        mix_json = _load(out / split / "mix.json")
        tgt_json = _load(out / split / "target.json")
        near_json = _load(out / split / "near.json")
        assert len(mix_json) == len(tgt_json) == len(near_json)
        for i, (mix_entry, tgt_entry, near_entry) in enumerate(zip(mix_json, tgt_json, near_json)):
            tgt_path, mouth_path, _n = tgt_entry
            near_path, near_n = near_entry[0]
            # A5-1: distinct suffixed target names, never the old <stem>.wav collapse
            suffix = ".s1.wav" if i % 2 == 0 else ".s2.wav"
            assert Path(tgt_path).name.endswith(suffix), tgt_path
            assert Path(near_path) != Path(tgt_path)
            assert Path(tgt_path).is_file() and Path(near_path).is_file()
            assert Path(near_path).parent.name == "near"
            # mouth speaker must match the target speaker encoded in the file name
            parts = Path(mix_entry[0]).stem.split("_")
            expect_id = "_".join(parts[0:2]) if i % 2 == 0 else "_".join(parts[3:5])
            assert Path(mouth_path).stem == expect_id
            assert near_n == N
            # waveform semantics: target = own speaker, near = the other speaker
            src_split = fake_src / "audio/wav16k/min" / split
            stem = Path(mix_entry[0]).stem
            own = src_split / ("s1" if i % 2 == 0 else "s2") / f"{stem}.wav"
            other = src_split / ("s2" if i % 2 == 0 else "s1") / f"{stem}.wav"
            assert Path(tgt_path).read_bytes() == own.read_bytes()
            assert Path(near_path).read_bytes() == other.read_bytes()


# ---------------------------------------------------------------------------
# default (no --copy-wavs): JSON references the raw sources, gate still runs
# ---------------------------------------------------------------------------
def test_default_layout_references_source_and_gate(fake_src: Path, tmp_path: Path):
    out = tmp_path / "paper_ref"
    res = _run(["--src", str(fake_src), "--out", str(out), "--limit", "2", "--jobs", "2"])
    assert res.returncode == 0, res.stdout + res.stderr
    stats = _load(out / "conversion_stats.json")
    assert stats["copy_wavs"] is False
    assert stats["gate"]["ok"] is True, stats["gate"]["problems"]
    assert not (out / "tr" / "mix").exists()  # nothing copied

    tgt_json = _load(out / "tr" / "target.json")
    near_json = _load(out / "tr" / "near.json")
    for i, (tgt_entry, near_entry) in enumerate(zip(tgt_json, near_json)):
        assert Path(tgt_entry[0]).parent.name == ("s1" if i % 2 == 0 else "s2")
        assert Path(near_entry[0][0]).parent.name == ("s2" if i % 2 == 0 else "s1")
        assert Path(tgt_entry[0]) != Path(near_entry[0][0])


def test_gate_fails_on_wrong_expected_count(fake_src: Path, tmp_path: Path):
    out = tmp_path / "paper_bad_count"
    res = _run(["--src", str(fake_src), "--out", str(out), "--limit", "2",
                "--jobs", "2", "--expect-entries", "tr=99"])
    assert res.returncode == 2, res.stdout + res.stderr
    assert "gate FAILED" in res.stdout
    assert "!= expected 99" in res.stdout
    stats = _load(out / "conversion_stats.json")
    assert stats["gate"]["ok"] is False
    assert any("entry count" in p for p in stats["gate"]["problems"])


def test_gate_detects_self_referencing_near(fake_src: Path, tmp_path: Path):
    """覆盖闸门的 near==target 检查：手工把某条 near 改成 target 自身。"""
    out = tmp_path / "paper_self_ref"
    res = _run(["--src", str(fake_src), "--out", str(out), "--limit", "1", "--jobs", "2"])
    assert res.returncode == 0, res.stdout + res.stderr
    near_path = out / "tr" / "near.json"
    tgt_path = out / "tr" / "target.json"
    target_json = _load(tgt_path)
    near_json = _load(near_path)
    near_json[0] = [[target_json[0][0], target_json[0][2]]]  # near -> target itself
    near_path.write_text(json.dumps(near_json, indent=0))

    # call the public gate directly on the corrupted JSON (no rebuild, no 26 GB data)
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_lrs2_paper_data", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problems = mod.validate_build(out, ["tr"], {"tr": 2}, copy_wavs=False)
    assert any("near == target" in p for p in problems), problems
