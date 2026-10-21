"""Tests for tools/repro/make_synthetic_data.py + the dataset mono guard (F5).

Covers:
* the synthetic generator CLI and the folder/JSON layout it writes;
* ``AVSpeechTrainDatasetWithNear`` and ``AVSpeechEvalDataset`` reading that data;
* F5 regression: mono wavs must stay 1-D in the eval dataset (previously a bare
  ``mean(-1)`` collapsed them to 0-dim and ``torch.stft`` crashed);
* the same mono guard in ``check_avspeech_dataset.py`` (stale copy);
* multi-channel behaviour is unchanged (channel mean).

Run with the project environment::

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_synthetic_data.py -q
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
GENERATOR = REPO / "tools" / "repro" / "make_synthetic_data.py"
SR = 16000
FRAME = SR // 25  # 640

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic") / "data"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(out), "--n", "1",
         "--seconds", "3", "--seed", "0"],
        cwd=str(REPO), capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "wrote 3 split(s)" in result.stdout
    return out


# ---------------------------------------------------------------------------
# layout / JSON
# ---------------------------------------------------------------------------
def test_synthetic_layout_and_indexes(synth_root: Path):
    for split, (mix_n, target_n, near_n) in {
        "tr": (1, 1, 1), "cv": (1, 1, 1), "tt": (1, 1, 1),
    }.items():
        split_dir = synth_root / split
        for sub in ("mix", "target", "mouths", "near"):
            assert (split_dir / sub).is_dir(), (split, sub)
        mix_json = json.loads((split_dir / "mix.json").read_text())
        target_json = json.loads((split_dir / "target.json").read_text())
        near_json = json.loads((split_dir / "near.json").read_text())
        assert len(mix_json) == mix_n and len(target_json) == target_n and len(near_json) == near_n
        assert len(mix_json[0]) == 2 and len(target_json[0]) == 3
        assert Path(mix_json[0][0]).is_absolute() and Path(target_json[0][1]).is_absolute()

        info = sf.info(mix_json[0][0])
        assert (info.samplerate, info.channels) == (SR, 1)
        assert info.frames == 3 * SR

        mouths = torch.load(target_json[0][1], map_location="cpu", weights_only=True)
        assert mouths.dtype == torch.uint8 and tuple(mouths.shape) == (75, 96, 96, 3)

        meta = json.loads((split_dir / "mix" / "tr_00000_meta.json").read_text()) \
            if split == "tr" else None
        if meta is not None:
            types = [c["type"] for c in meta["components"]]
            assert "near" in types and "far" in types and "env" in types
            near_component = next(c for c in meta["components"] if c["type"] == "near")
            assert Path(near_component["path"]).is_file()


# ---------------------------------------------------------------------------
# datasets on the synthetic data
# ---------------------------------------------------------------------------
def test_train_dataset_item(synth_root: Path):
    from look2hear.datas.avspeech_dataset_near import AVSpeechTrainDatasetWithNear

    ds = AVSpeechTrainDatasetWithNear(
        json_dir=str(synth_root / "tr"), sample_rate=SR, segment=2.0,
        normalize_audio=False, samples=1, use_near=True,
    )
    mixed, target, mouth, key, near_audios = ds[0]
    assert tuple(mixed.shape) == (2 * SR,) and mixed.dtype == torch.float32
    assert tuple(target.shape) == (2 * SR,) and target.dtype == torch.float32
    assert tuple(mouth.shape) == (50, 88, 88)  # 2 s @ 25 fps, center/random cropped to 88
    assert isinstance(key, str) and key.endswith(".wav")
    assert len(near_audios) == 1, "synthetic near waves should give K=1 (NASS smoke path)"
    assert tuple(near_audios[0].shape) == (2 * SR,)


def test_eval_dataset_mono_guard(synth_root: Path):
    """F5 regression: mono eval audio must be 1-D with full length (was 0-dim before fix)."""
    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset

    ds = AVSpeechEvalDataset(json_dir=str(synth_root / "tt"), sample_rate=SR, normalize_audio=False)
    mix, source, mouth, key, n_near = ds[0]
    expected_samples = 3 * SR
    assert mix.ndim == 1, f"F5 regression: mono mixture collapsed to shape {tuple(mix.shape)}"
    assert source.ndim == 1
    assert tuple(mix.shape) == (expected_samples,)
    assert tuple(source.shape) == (expected_samples,)
    assert tuple(mouth.shape) == (75, 88, 88)
    assert isinstance(n_near, int)
    # what audio_visual_test.py feeds the model:
    assert tuple(mix[None].shape) == (1, expected_samples)
    assert tuple(torch.from_numpy(mouth[None, None]).shape) == (1, 1, 75, 88, 88)
    # .pt stores 75 frames = 3 s at 25 fps, matching the 48000-sample audio
    assert mouth.shape[0] == expected_samples // FRAME


def test_eval_dataset_stereo_still_uses_channel_mean(tmp_path: Path):
    """Multi-channel wavs keep the original channel-mean behaviour."""
    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset

    d = tmp_path / "stereo_tt"
    for sub in ("mix", "target", "mouths"):
        (d / sub).mkdir(parents=True)
    n = 2 * SR
    t = np.arange(n, dtype=np.float32) / SR
    left = 0.2 * np.sin(2 * np.pi * 200 * t)
    right = left + 0.1
    mix = np.stack([left, right], axis=1).astype(np.float32)
    target = mix.copy()
    sf.write(str(d / "mix" / "s0.wav"), mix, SR, subtype="PCM_16")
    sf.write(str(d / "target" / "s0.wav"), target, SR, subtype="PCM_16")
    torch.save(torch.zeros(50, 96, 96, 3, dtype=torch.uint8), d / "mouths" / "s0.pt")
    json.dump([[str(d / "mix" / "s0.wav"), n]], open(d / "mix.json", "w"))
    json.dump([[str(d / "target" / "s0.wav"), str(d / "mouths" / "s0.pt"), n]],
              open(d / "target.json", "w"))

    ds = AVSpeechEvalDataset(json_dir=str(d), sample_rate=SR, normalize_audio=False)
    mixture = ds[0][0]
    assert mixture.ndim == 1 and tuple(mixture.shape) == (n,)
    # compare against the channel mean of the *read-back* stereo file (PCM16 quantised)
    readback, _ = sf.read(str(d / "mix" / "s0.wav"), dtype="float32", always_2d=True)
    assert np.allclose(mixture.numpy(), readback.mean(axis=1), atol=1e-6)


def test_check_copy_mono_guard(synth_root: Path):
    """The stale diagnostic copy in check_avspeech_dataset.py got the same guard."""
    from look2hear.datas.check_avspeech_dataset import AVSpeechEvalDataset as CheckEvalDataset

    ds = CheckEvalDataset(json_dir=str(synth_root / "tt"), sample_rate=SR, normalize_audio=False)
    mix, source, *_ = ds[0]
    assert mix.ndim == 1 and tuple(mix.shape) == (3 * SR,)
    assert source.ndim == 1
