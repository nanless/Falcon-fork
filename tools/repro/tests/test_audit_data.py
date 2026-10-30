"""Tests for tools/repro/audit_data.py and the fetch_data.py checksum helpers.

All fixtures are tiny synthetic Falcon folder-format datasets built in tmp_path; the
real /root/data trees are never touched.

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audit_data.py -q
"""

from __future__ import annotations

import importlib.util
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
AUDIT = REPO / "tools" / "repro" / "audit_data.py"
FETCH = REPO / "tools" / "repro" / "fetch_data.py"
SR = 16000
FRAME = SR // 25  # 640
N = 2 * SR  # 32000 = 50 frames


def _tone(n: int, freq: float, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64) / SR
    return (0.1 * np.sin(2 * np.pi * freq * t) + 0.001 * rng.randn(n)).astype(np.float32)


def _wav(path: Path, data: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sr, subtype="PCM_16")


def _write_split(root: Path, split: str, n: int = 2, fallback: bool = False,
                 bad_mouth: bool = False, sr: int = SR) -> None:
    d = root / split
    mix_json, tgt_json, near_json = [], [], []
    for i in range(n):
        stem = f"{1000 + i:04d}_{i:05d}"
        mix_path = d / "mix" / f"{stem}.wav"
        tgt_path = d / "target" / f"{stem}.wav"
        mouth_path = d / "mouths" / f"{stem}.pt"
        near_path = d / "near" / f"{stem}.near0.wav"
        _wav(mix_path, _tone(N, 180 + 10 * i, seed=i), sr=sr)
        _wav(tgt_path, _tone(N, 220 + 10 * i, seed=100 + i), sr=sr)
        _wav(near_path, _tone(N, 300 + 10 * i, seed=200 + i), sr=sr)
        mouth_path.parent.mkdir(parents=True, exist_ok=True)
        if bad_mouth:
            torch.save(torch.zeros((50, 96, 96), dtype=torch.uint8), mouth_path)  # 3-D
        else:
            torch.save(torch.randint(0, 256, (50, 96, 96, 3), dtype=torch.uint8), mouth_path)
        mix_json.append([str(mix_path.resolve()), N])
        tgt_json.append([str(tgt_path.resolve()), str(mouth_path.resolve()), N])
        if fallback:
            near_json.append([[str(mix_path.resolve()), N]])  # self-reference sentinel
        else:
            near_json.append([[str(near_path.resolve()), N]])
    for name, obj in (("mix.json", mix_json), ("target.json", tgt_json), ("near.json", near_json)):
        (d / name).write_text(json.dumps(obj, indent=0))


def _make_dataset(root: Path, **kwargs) -> Path:
    for split in ("tr", "cv", "tt"):
        _write_split(root, split, **kwargs)
    return root


def _run(script: Path, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, str(script), *args],
                          cwd=str(REPO), capture_output=True, text=True, timeout=timeout, env=env)


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_data_under_test", FETCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# audit_data.py: good dataset
# ---------------------------------------------------------------------------
def test_audit_good_dataset_exit_0(tmp_path: Path):
    root = _make_dataset(tmp_path / "data")
    res = _run(AUDIT, [str(root), "--sample", "2"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "K=" in res.stdout and "coverage=100.00%" in res.stdout
    assert "status=OK" in res.stdout
    assert "total_fallback=0" in res.stdout


def test_audit_full_mode_and_json(tmp_path: Path):
    root = _make_dataset(tmp_path / "data")
    out_json = tmp_path / "audit.json"
    res = _run(AUDIT, [str(root), "--full", "--json", str(out_json)])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "(full)" in res.stdout
    summary = json.loads(out_json.read_text())
    assert summary["ok"] is True and summary["total_fallback"] == 0
    assert summary["total_entries"] == 6
    assert set(summary["splits"]) == {"tr", "cv", "tt"}


# ---------------------------------------------------------------------------
# audit_data.py: failure modes (must exit 1)
# ---------------------------------------------------------------------------
def test_audit_detects_fallback_sentinel_exit_1(tmp_path: Path):
    root = _make_dataset(tmp_path / "data_fb", fallback=True)
    res = _run(AUDIT, [str(root), "--sample", "2"])
    assert res.returncode == 1, res.stdout + res.stderr
    assert "fallback" in res.stdout.lower()
    assert "status=FAILED" in res.stdout


def test_audit_detects_missing_json_exit_1(tmp_path: Path):
    root = _make_dataset(tmp_path / "data_missing")
    (root / "cv" / "near.json").unlink()
    res = _run(AUDIT, [str(root), "--sample", "2"])
    assert res.returncode == 1, res.stdout + res.stderr
    assert "missing JSON" in res.stdout


def test_audit_detects_bad_mouth_shape_exit_1(tmp_path: Path):
    root = _make_dataset(tmp_path / "data_bad_mouth", bad_mouth=True)
    res = _run(AUDIT, [str(root), "--sample", "2"])
    assert res.returncode == 1, res.stdout + res.stderr
    assert "mouths shape" in res.stdout


def test_audit_detects_wrong_sample_rate_exit_1(tmp_path: Path):
    root = _make_dataset(tmp_path / "data_8k", sr=8000)
    res = _run(AUDIT, [str(root), "--sample", "1"])
    assert res.returncode == 1, res.stdout + res.stderr
    assert "samplerate" in res.stdout


def test_audit_missing_root_exit_1(tmp_path: Path):
    res = _run(AUDIT, [str(tmp_path / "does_not_exist")])
    assert res.returncode == 1
    assert "not found" in res.stdout


# ---------------------------------------------------------------------------
# fetch_data.py: --sha256 / --bytes helpers (A5-3); no network involved
# ---------------------------------------------------------------------------
def test_fetch_verify_download_bytes_and_sha256_ok(tmp_path: Path, capsys):
    mod = _load_fetch_module()
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"falcon-data-" + bytes(range(256)) * 3)
    size = p.stat().st_size
    digest = mod.file_sha256(str(p))
    assert mod.verify_download(str(p), size, digest) == 0
    out = capsys.readouterr().out
    assert "[verify] OK sha256=" in out and digest in out


def test_fetch_verify_download_wrong_bytes_or_sha_exit_2(tmp_path: Path, capsys):
    mod = _load_fetch_module()
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"x" * 1000)
    assert mod.verify_download(str(p), 999, None) == 2
    assert "FAILED bytes" in capsys.readouterr().out
    assert mod.verify_download(str(p), None, "0" * 64) == 2
    assert "FAILED sha256" in capsys.readouterr().out


def test_fetch_verify_download_missing_file_exit_2(tmp_path: Path, capsys):
    mod = _load_fetch_module()
    assert mod.verify_download(str(tmp_path / "nope.bin"), 1, None) == 2
    assert "file not found" in capsys.readouterr().out


def test_fetch_verify_download_noop_when_unrequested(tmp_path: Path):
    mod = _load_fetch_module()
    assert mod.verify_download(str(tmp_path / "nope.bin"), None, None) == 0


def test_fetch_cli_rejects_bad_sha256_hex():
    """argparse 层校验：非法 sha 在联网前即被拒绝（exit 2）。"""
    res = _run(FETCH, ["--dataset", "lrs2-2mix", "--dry-run", "--sha256", "not-a-sha"])
    assert res.returncode == 2
    assert "64 位十六进制" in res.stderr
