"""End-to-end tests for tools/repro/realssa_to_falcon.py using *fake* RealSSA-Sim tar shards.

No gated RealSSA-Sim data is downloaded or required: the tests build local shards with
the published member layout (``<id>.mix.wav/.target.wav/.json/.mouths.pt``) and run the
converter as a user would.

Run with the project environment::

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_realssa_convert.py -q
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
CONVERTER = REPO / "tools" / "repro" / "realssa_to_falcon.py"
SR = 16000
FRAME = SR // 25  # 640


# ---------------------------------------------------------------------------
# fake shard helpers
# ---------------------------------------------------------------------------
def _wav_bytes(samples: int, channels: int = 4, sample_rate: int = SR) -> bytes:
    """Deterministic multi-channel wav: channel c is a distinct tone."""
    t = np.arange(samples, dtype=np.float64) / sample_rate
    chans = []
    for c in range(channels):
        freq = 180.0 + 40.0 * c
        chans.append(0.2 * np.sin(2 * np.pi * freq * t + 0.3 * c))
    audio = np.stack(chans, axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _mouths_bytes(frames: int, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    mouths = rng.randint(0, 256, size=(frames, 96, 96, 3), dtype=np.uint8)
    buf = io.BytesIO()
    torch.save(torch.from_numpy(mouths), buf)
    return buf.getvalue()


def _json_bytes(sample_id: str) -> bytes:
    meta = {
        "scene": "small_meeting",
        "room": {"dims": [5.0, 4.0, 3.0], "mic_center": [2.5, 2.0, 1.5], "t60": 0.45},
        "components": [
            {"type": "near", "source": "librispeech", "sir": -3.0,
             "position": [1.0, 1.0, 1.2], "path": f"/generation/{sample_id}/near0.wav"},
            {"type": "far", "source": "librispeech", "sir": -6.0,
             "position": [4.0, 3.0, 1.2], "path": f"/generation/{sample_id}/far0.wav"},
            {"type": "env", "source": "wham", "sir": None,
             "position": None, "path": f"/generation/{sample_id}/env.wav"},
        ],
        "target_position": [0.5, 0.5, 1.2],
        "transcript": "synthetic transcript",
    }
    return json.dumps(meta).encode("utf-8")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_shard(path: Path, specs: list[tuple[str, int, int]],
                include_incomplete: bool = True) -> Path:
    """Create one fake shard; ``specs`` items are (sample_id, audio_samples, mouth_frames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for seed, (sample_id, samples, frames) in enumerate(specs):
            _add(tar, f"{sample_id}.mix.wav", _wav_bytes(samples))
            _add(tar, f"{sample_id}.target.wav", _wav_bytes(samples))
            _add(tar, f"{sample_id}.json", _json_bytes(sample_id))
            _add(tar, f"{sample_id}.mouths.pt", _mouths_bytes(frames, seed=seed))
        if include_incomplete:
            # emulate the four incomplete training examples: no json / no mouth crops
            _add(tar, "00099_incomplete.mix.wav", _wav_bytes(32000))
            _add(tar, "00099_incomplete.target.wav", _wav_bytes(32000))
    return path


SPECS = [
    ("00001_01_87828", 3 * SR, 75),   # exact 3 s: 75 frames * 640 = 48000 samples
    ("00002_05_59973", 3 * SR, 60),   # mouth shorter than audio -> trim to 60 frames
    ("00003_00_12345", 2 * SR, 50),   # exact 2 s
]


@pytest.fixture(scope="module")
def fake_src(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp("realssa_src")
    _make_shard(src / "tt" / "tt-000000.tar", SPECS)
    return src


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, str(CONVERTER), *args],
        cwd=str(REPO), capture_output=True, text=True, timeout=timeout, env=env,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_end_to_end_channel0(fake_src: Path, tmp_path: Path):
    out = tmp_path / "falcon_data"
    result = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tt",
                   "--channel", "0", "--limit-shards", "1"])
    assert result.returncode == 0, result.stderr
    assert "converted 3 sample(s)" in result.stdout, result.stdout
    assert "skipped incomplete=1" in result.stdout

    for sample_id, samples, mouth_frames in SPECS:
        frames = min(mouth_frames, samples // FRAME)
        mix_path = out / "tt" / "mix" / f"{sample_id}.wav"
        tgt_path = out / "tt" / "target" / f"{sample_id}.wav"
        mouth_path = out / "tt" / "mouths" / f"{sample_id}.pt"
        meta_path = out / "tt" / "mix" / f"{sample_id}_meta.json"

        info = sf.info(str(mix_path))
        assert (info.samplerate, info.channels) == (SR, 1)
        assert info.frames == frames * FRAME, (sample_id, info.frames, frames * FRAME)
        tinfo = sf.info(str(tgt_path))
        assert (tinfo.samplerate, tinfo.channels, tinfo.frames) == (SR, 1, frames * FRAME)

        mouths = torch.load(str(mouth_path), map_location="cpu", weights_only=True)
        assert mouths.dtype == torch.uint8
        assert tuple(mouths.shape) == (frames, 96, 96, 3)

        meta = json.loads(meta_path.read_text())
        assert meta["sample_id"] == sample_id
        assert meta["provenance"]["channel_policy"] == "0"
        assert meta["provenance"]["aligned_frames"] == frames
        types = [c["type"] for c in meta["components"]]
        assert {"near", "far", "env"} <= set(types)
        assert all(isinstance(c.get("path"), str) for c in meta["components"])

        # channel 0 of the original 4ch signal must be preserved
        expected = _wav_bytes(samples)  # regenerate identical bytes
        orig = sf.read(io.BytesIO(expected), dtype="float32", always_2d=True)[0][:, 0]
        got = sf.read(str(mix_path), dtype="float32")[0]
        assert np.allclose(got, orig[: frames * FRAME], atol=2e-4)

    mix_json = json.loads((out / "tt" / "mix.json").read_text())
    target_json = json.loads((out / "tt" / "target.json").read_text())
    near_json = json.loads((out / "tt" / "near.json").read_text())
    assert len(mix_json) == len(target_json) == len(near_json) == len(SPECS)
    assert len(mix_json[0]) == 2 and len(target_json[0]) == 3
    # no near wavs were shipped -> every near.json entry is the mixture fallback sentinel
    for entry, mix_entry in zip(near_json, mix_json):
        assert entry == [[mix_entry[0], mix_entry[1]]]
    assert "no near-speaker waveforms" in result.stdout


def test_channel_mean_matches_manual_average(fake_src: Path, tmp_path: Path):
    out = tmp_path / "mean_data"
    result = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tt",
                   "--channel", "mean", "--limit-samples", "1"])
    assert result.returncode == 0, result.stderr
    sample_id, samples, _ = SPECS[0]
    orig = sf.read(io.BytesIO(_wav_bytes(samples)), dtype="float32", always_2d=True)[0]
    expected = orig.mean(axis=1)
    got = sf.read(str(out / "tt" / "mix" / f"{sample_id}.wav"), dtype="float32")[0]
    n = min(len(got), len(expected))
    assert np.allclose(got[:n], expected[:n], atol=2e-4)
    meta = json.loads((out / "tt" / "mix" / f"{sample_id}_meta.json").read_text())
    assert meta["provenance"]["channel_policy"] == "mean"


def test_dry_run_writes_nothing(fake_src: Path, tmp_path: Path):
    out = tmp_path / "dry_data"  # must stay absent
    result = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tt", "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "[dry-run] no files were written" in result.stdout
    assert not out.exists()


def test_missing_src_has_clear_error(tmp_path: Path):
    result = _run(["--src", str(tmp_path / "does_not_exist"), "--out", str(tmp_path / "o")])
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)


def test_external_near_dir_is_wired_into_meta_and_near_json(fake_src: Path, tmp_path: Path):
    near_dir = tmp_path / "near"
    near_dir.mkdir()
    sample_id = SPECS[0][0]
    near_wav = near_dir / f"{sample_id}.near0.wav"
    t = np.arange(3 * SR, dtype=np.float64) / SR
    sf.write(str(near_wav), 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32), SR,
             subtype="PCM_16")

    out = tmp_path / "near_data"
    result = _run(["--src", str(fake_src), "--out", str(out), "--splits", "tt",
                   "--channel", "0", "--near-dir", str(near_dir), "--limit-samples", "1"])
    assert result.returncode == 0, result.stderr

    meta = json.loads((out / "tt" / "mix" / f"{sample_id}_meta.json").read_text())
    external = [c for c in meta["components"] if c.get("source") == "external-near"]
    assert len(external) == 1 and external[0]["type"] == "near"
    assert Path(external[0]["path"]).resolve() == near_wav.resolve()

    near_json = json.loads((out / "tt" / "near.json").read_text())
    mix_json = json.loads((out / "tt" / "mix.json").read_text())
    assert len(near_json) == len(mix_json) == 1
    assert near_json[0][0][0] == str(near_wav.resolve())
    assert near_json[0][0][1] == 3 * SR
