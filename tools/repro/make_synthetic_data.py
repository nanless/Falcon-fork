#!/usr/bin/env python
"""Generate a small synthetic AVTSE dataset in the Falcon folder layout.

Purpose
-------
Phase-1 smoke tests (T1/T3): provide a tiny dataset that the *unmodified*
``look2hear.datas`` DataModule can read directly, so training/evaluation can be
exercised end-to-end without the paper datasets.

Layout written under ``--out`` (one sub-directory per split)::

    <out>/<split>/
        mix/<stem>.wav              16 kHz mono mixture        (PCM_16)
        mix/<stem>_meta.json        RealSSA-style meta: components[near/far/env]
        target/<stem>.wav           16 kHz mono target speech (PCM_16)
        mouths/<stem>.pt            uint8 RGB mouth crops, torch [T, 96, 96, 3]
        near/<stem>_near<k>.wav     16 kHz mono synthetic near-field interferers
        mix.json target.json near.json    written by the *real* DataPreProcess code

The JSON indexes are produced by importing the repository's
``DataPreProcess/process_lrs23.py`` and ``DataPreProcess/generate_near_json.py``
rather than re-implementing them, so the generated dataset follows the same
contract the training loader expects (absolute paths, near.json aligned to
mix.json, etc.).

Notes
-----
* Audio is mono 16 kHz on purpose: this exercises the F5 mono guard
  (``docs/REPRODUCTION_PLAN.md`` §6.1). Before the fix, the eval dataset
  collapsed mono waveforms to 0-dim tensors.
* ``mouths.pt`` is ``uint8 RGB [T,96,96,3]`` — the RealSSA-Sim format; the
  dataset's Grayscale + crop pipeline turns it into ``[F,88,88]``.
* ``T`` is derived from ``--seconds`` at 25 fps and audio length is
  ``T * 640`` samples, so audio and mouth frames are exactly aligned.
* This data is for pipeline smoke tests only — never for quality metrics.

Usage
-----
    python tools/repro/make_synthetic_data.py \
        --out /tmp/falcon_smoke/data --n 2 --seconds 3 --seed 0

Then point the DataModule at it::

    train_dir: /tmp/falcon_smoke/data/tr
    valid_dir: /tmp/falcon_smoke/data/cv
    test_dir:  /tmp/falcon_smoke/data/tt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640, matches look2hear/datas/*.py
MOUTH_H = MOUTH_W = 96
MOUTH_CHANNELS = 3
SCENES = ("small_meeting", "medium_indoor", "large_public", "outdoor")


# ---------------------------------------------------------------------------
# DataPreProcess reuse
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _datapreprocess_module():
    root = _repo_root() / "DataPreProcess"
    return _load_module("dp_process_lrs23", root / "process_lrs23.py")


def build_indexes(data_root: str | Path, splits=("tr", "cv", "tt")) -> dict[str, dict]:
    """Build mix.json / target.json / near.json for a folder-layout dataset.

    Reuses ``DataPreProcess/process_lrs23.py`` (mix/target) and its embedded
    ``generate_near_json`` logic. Returns ``{split: {"mix": path, "target": path,
    "near": path, "num": n}}``.

    Empty splits are tolerated: the upstream ``generate_near_json`` crashes on an
    empty split because of ``min(near_counts)``; we write an empty ``near.json``
    ourselves in that case instead of crashing.
    """
    process_lrs23 = _datapreprocess_module()
    data_root = Path(data_root).resolve()
    written: dict[str, dict] = {}
    for split in splits:
        split_dir = data_root / split
        (split_dir / "mix").mkdir(parents=True, exist_ok=True)
        (split_dir / "target").mkdir(parents=True, exist_ok=True)
        for source in ("mix", "target"):
            process_lrs23.preprocess_one_dir(
                str(split_dir / source), str(split_dir), source, split
            )
        mix_json = split_dir / "mix.json"
        target_json = split_dir / "target.json"
        near_json = split_dir / "near.json"
        with open(mix_json) as f:
            mix_infos = json.load(f)
        if not mix_infos:
            with open(near_json, "w") as f:
                json.dump([], f, indent=2)
        else:
            # process_lrs23.py embeds the same near.json generation logic
            # (its module-level `generate_near_json` function).
            process_lrs23.generate_near_json(
                str(mix_json), str(split_dir / "mix"), str(near_json)
            )
        written[split] = {
            "mix": str(mix_json),
            "target": str(target_json),
            "near": str(near_json),
            "num": len(mix_infos),
        }
    return written


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------
def _speechlike(n: int, rng: np.random.RandomState, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A modulated harmonic tone that looks like a voiced utterance to a model."""
    t = np.arange(n, dtype=np.float64) / sr
    f0 = rng.uniform(95.0, 240.0)
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 6):
        amp = 1.0 / k
        sig += amp * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    amp_env = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t + rng.uniform(0, 2 * np.pi)))
    sig = sig * amp_env + rng.normal(0.0, 0.02, n)
    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    return (0.25 * sig).astype(np.float32)


def _environmental_noise(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Low-passed noise (moving average) as a crude environmental bed."""
    raw = rng.normal(0.0, 1.0, n)
    kernel = np.ones(17, dtype=np.float64) / 17.0
    smooth = np.convolve(raw, kernel, mode="same")
    smooth = smooth / (np.max(np.abs(smooth)) + 1e-8)
    return (0.05 * smooth).astype(np.float32)


def _mouth_tensor(frames: int, rng: np.random.RandomState) -> torch.Tensor:
    """uint8 RGB mouth crops [T,96,96,3] (RealSSA-Sim layout), temporally smooth-ish."""
    base = rng.normal(0.0, 1.0, (MOUTH_H, MOUTH_W, MOUTH_CHANNELS))
    kernel_h = np.ones((5, 1, 1)) / 5.0
    mouth = np.empty((frames, MOUTH_H, MOUTH_W, MOUTH_CHANNELS), dtype=np.float32)
    for f in range(frames):
        frame = base + rng.normal(0.0, 0.35, base.shape)
        mouth[f] = frame
    # cheap spatial smoothing via a 3x3 box filter over H/W
    padded = np.pad(mouth, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    mouth = (
        padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:]
        + padded[:, 1:-1, :-2] + padded[:, 1:-1, 1:-1] + padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2] + padded[:, 2:, 1:-1] + padded[:, 2:, 2:]
    ) / 9.0
    mouth = mouth - mouth.min()
    mouth = mouth / (mouth.max() + 1e-8) * 255.0
    return torch.from_numpy(mouth.astype(np.uint8))


def _write_mono(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def make_sample(
    split_dir: Path,
    stem: str,
    frames: int,
    rng: np.random.RandomState,
    near_per_sample: int = 1,
) -> dict:
    """Write one synthetic utterance (audio, mouths, near sources, meta)."""
    n = frames * SAMPLES_PER_FRAME
    target = _speechlike(n, rng)
    interference = _speechlike(n, rng)
    env = _environmental_noise(n, rng)
    mix = 0.70 * target + 0.25 * interference + 0.05 * env
    mix = np.clip(mix, -0.98, 0.98).astype(np.float32)

    _write_mono(split_dir / "mix" / f"{stem}.wav", mix)
    _write_mono(split_dir / "target" / f"{stem}.wav", target)

    mouth_path = split_dir / "mouths" / f"{stem}.pt"
    mouth_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_mouth_tensor(frames, rng), mouth_path)

    components = []
    for k in range(int(near_per_sample)):
        near = _speechlike(n, rng)
        near_path = split_dir / "near" / f"{stem}_near{k}.wav"
        _write_mono(near_path, near)
        components.append(
            {
                "type": "near",
                "path": str(near_path),
                "source": "synthetic",
                "sir_db": round(float(rng.uniform(-5.0, 10.0)), 3),
                "position": [round(float(x), 3) for x in rng.uniform(-2.0, 2.0, 3)],
            }
        )
    components.append(
        {
            "type": "far",
            "path": "synthetic://far/provenance-only",
            "source": "synthetic",
            "sir_db": round(float(rng.uniform(-5.0, 10.0)), 3),
            "position": [round(float(x), 3) for x in rng.uniform(-6.0, 6.0, 3)],
        }
    )
    components.append({"type": "env", "path": "synthetic://env/provenance-only", "source": "synthetic"})

    meta = {
        "sample_id": stem,
        "scene": SCENES[int(rng.randint(len(SCENES)))],
        "room": {"dims": [round(float(x), 3) for x in rng.uniform(3.0, 12.0, 3)],
                 "t60": round(float(rng.uniform(0.2, 1.2)), 3)},
        "components": components,
        "target_position": [round(float(x), 3) for x in rng.uniform(-2.0, 2.0, 3)],
        "transcript": "synthetic utterance for pipeline smoke tests",
        "provenance": {"generator": "tools/repro/make_synthetic_data.py",
                       "audio": "16 kHz mono PCM16", "mouths": "uint8 RGB [T,96,96,3]"},
    }
    meta_path = split_dir / "mix" / f"{stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {
        "stem": stem,
        "samples": n,
        "frames": frames,
        "mix": str(split_dir / "mix" / f"{stem}.wav"),
        "target": str(split_dir / "target" / f"{stem}.wav"),
        "mouths": str(mouth_path),
        "num_near": int(near_per_sample),
    }


def make_synthetic_dataset(
    out: str | Path,
    splits=("tr", "cv", "tt"),
    n_per_split: int = 2,
    seconds: float = 3.0,
    seed: int = 0,
    near_per_sample: int = 1,
) -> dict:
    """Create the synthetic dataset and return a summary dict (also used by tests)."""
    out = Path(out)
    frames = int(round(seconds * FPS))
    if frames < 50:
        raise SystemExit(
            "--seconds must be >= 2.0: the training loader draws 2 s (32000 samples / 50 frames) "
            f"windows, got {seconds}s ({frames} frames)"
        )
    if n_per_split < 1:
        raise SystemExit("--n must be >= 1")
    summary = {"root": str(out.resolve()), "splits": {}, "frames": frames,
               "samples": frames * SAMPLES_PER_FRAME, "seconds": frames / FPS}
    for split_idx, split in enumerate(splits):
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(seed * 1009 + split_idx)
        samples = []
        for i in range(n_per_split):
            stem = f"{split}_{i:05d}"
            samples.append(make_sample(split_dir, stem, frames, rng, near_per_sample))
        summary["splits"][split] = samples
    indexes = build_indexes(out, splits)
    summary["indexes"] = indexes
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a tiny synthetic Falcon-format dataset for Phase-1 smoke tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Output dataset root (contains tr/cv/tt).")
    parser.add_argument("--splits", default="tr,cv,tt", help="Comma-separated split names.")
    parser.add_argument("--n", type=int, default=2, help="Utterances per split.")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="Utterance duration; must be >= 2.0 (training window is 2 s).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--near", type=int, default=1,
                        help="Synthetic near-field interferers per sample (for NASS contrastive smoke).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete an existing non-empty --out directory first.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not splits:
        raise SystemExit("--splits must not be empty")
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"output directory {out} exists and is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    summary = make_synthetic_dataset(
        out, splits=splits, n_per_split=args.n, seconds=args.seconds,
        seed=args.seed, near_per_sample=args.near,
    )

    print(f"[make_synthetic_data] wrote {len(splits)} split(s) under {out}")
    print(f"[make_synthetic_data] duration = {summary['seconds']:.2f}s "
          f"({summary['samples']} samples @ {SAMPLE_RATE} Hz, {summary['frames']} mouth frames @ {FPS} fps)")
    for split in splits:
        idx = summary["indexes"][split]
        n = len(summary["splits"][split])
        print(f"  {split}: {n} utterance(s) | mix.json={idx['mix']} target.json={idx['target']} "
              f"near.json={idx['near']}")
    print("\nDataModule config snippet (configs/Falcon.yml -> datamodule.data_config):")
    print("  train_dir: %s" % (out / "tr"))
    print("  valid_dir: %s" % (out / "cv"))
    print("  test_dir:  %s" % (out / "tt"))
    print("\nReminder: synthetic data is for pipeline smoke tests only (not for metrics).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
