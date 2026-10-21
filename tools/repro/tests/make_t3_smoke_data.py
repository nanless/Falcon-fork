"""Create a minimal synthetic evaluation index for the Phase1/T3 smoke run.

This is intentionally independent from ``tools/repro/make_synthetic_data.py``
(T4's training-oriented generator): it only builds an existing-format test split
that ``AVSpeechEvalDataset`` / ``audio_visual_test.py`` can consume without the
train/valid splits.

Output layout (``--out``, default ``/tmp/t3_smoke/data``)::

    <out>/tt/mix.json                # [[mix_wav_path, n_samples], ...]
    <out>/tt/target.json             # [[target_wav_path, mouth_pt_path, n_samples], ...]
    <out>/tt/mix/u0000.wav           # stereo 16 kHz (identical channels)
    <out>/tt/target/u0000.wav        # stereo 16 kHz (identical channels)
    <out>/tt/mouths/u0000.pt         # uint8 RGB [F, 96, 96, 3] (channel-last, as the dataset expects)
    <out>/tt/mix/u0000_meta.json     # {"components": [...]} (eval discards n_near, but this exercises the read path)

Stereo is used on purpose: the eval dataset averages channels before the mono
guard from T4 lands, so this fixture works on both the old and the fixed dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def make_like_face_mouths(frames: int, rng: np.random.Generator, height: int = 96, width: int = 96) -> np.ndarray:
    """Return uint8 RGB ``[F, H, W, 3]`` mouth-like crops (smooth random field)."""
    base = rng.random((frames, height, width, 1))
    # smooth slightly so consecutive frames are not pure noise
    smooth = base.copy()
    smooth[:, 1:-1, 1:-1, :] = (
        base[:, :-2, 1:-1, :] + base[:, 2:, 1:-1, :] + base[:, 1:-1, :-2, :] + base[:, 1:-1, 2:, :]
    ) / 4.0
    rgb = np.repeat(smooth, 3, axis=-1)
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="/tmp/t3_smoke/data", help="output data root")
    parser.add_argument("--seconds", type=float, default=2.0, help="utterance duration (s)")
    parser.add_argument("--n", type=int, default=1, help="number of utterances")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sr = 16000
    seconds = float(args.seconds)
    n_samples = int(round(seconds * sr))
    n_frames = int(round(seconds * 25))
    out = Path(args.out).resolve()
    split = out / "tt"
    for sub in ("mix", "target", "mouths"):
        (split / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    time_axis = np.arange(n_samples) / sr
    mix_infos, target_infos = [], []

    for i in range(args.n):
        # deterministic "speech-like" target (harmonics + slow envelope) and interferer
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * time_axis + 0.4 * i)
        target = envelope * (
            0.12 * np.sin(2 * np.pi * 220.0 * time_axis)
            + 0.06 * np.sin(2 * np.pi * 440.0 * time_axis + 0.3)
            + 0.03 * np.sin(2 * np.pi * 660.0 * time_axis + 0.7)
        )
        target += 0.005 * rng.standard_normal(n_samples)
        interferer = 0.10 * np.sin(2 * np.pi * 310.0 * time_axis + 0.9) + 0.05 * rng.standard_normal(n_samples)
        mix = target + interferer  # roughly 0 dB target-to-interferer for the synthetic sample

        # stereo with identical channels (eval dataset takes .mean(-1) -> 1-D)
        stem = f"u{i:04d}"
        mix_wav = split / "mix" / f"{stem}.wav"
        target_wav = split / "target" / f"{stem}.wav"
        sf.write(mix_wav, np.stack([mix, mix], axis=1).astype(np.float32), sr)
        sf.write(target_wav, np.stack([target, target], axis=1).astype(np.float32), sr)

        mouth = make_like_face_mouths(n_frames, rng)
        mouth_path = split / "mouths" / f"{stem}.pt"
        torch.save(torch.from_numpy(mouth), mouth_path)  # uint8 [F, 96, 96, 3] channel-last

        meta_path = split / "mix" / f"{stem}_meta.json"
        meta_path.write_text(json.dumps({
            "components": [
                {"type": "target", "path": str(target_wav)},
                {"type": "near", "path": str(target_wav), "sir_db": 0.0},
            ]
        }, indent=2))

        mix_infos.append([str(mix_wav), n_samples])
        target_infos.append([str(target_wav), str(mouth_path), n_samples])

    (split / "mix.json").write_text(json.dumps(mix_infos, indent=2))
    (split / "target.json").write_text(json.dumps(target_infos, indent=2))
    print(f"[T3] synthetic eval index written to {out}")
    print(f"[T3]   utterances={len(mix_infos)}  samples/utt={n_samples}  mouth_frames={n_frames}  sr={sr}")
    print(f"[T3]   test index: {split}  (mix.json + target.json + mix/ + target/ + mouths/)")


if __name__ == "__main__":
    main()
