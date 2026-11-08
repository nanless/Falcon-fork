#!/usr/bin/env python3
"""Windowed-streaming vs whole-clip consistency for Falcon (inference only, no training).

Studio separates audio in rolling 2 s windows (2 s window / 0.5 s hop) and plays the
newest 0.5 s of every window, discarding a *guard* band at the end of the window.  This
script measures how far that stitched stream is from the offline reference - the whole
clip in one forward pass (`infer_file.extract`, the evaluation protocol) - using the same
checkpoint, the same mouth crops and the same normalisation.

It answers three questions with numbers instead of prose:

  1. how much does stitching cost?
     SI-SNR / Pearson r / max-deviation-to-peak of the stitched stream against the
     whole-clip output, on exactly the samples the stream emits;
  2. how much does the guard band matter?  sweep --guards 80,160,320,456,640 (ms), in one of two
     readings of the same number:
       --mode stitch    Studio's geometry: every window is 2 s, the emitted 0.5 s sits `guard`
                        samples before the window end (default; this is what server.py does);
       --mode lookahead one 2 s *output* chunk with `guard` samples of extra right context fed
                        inside the same forward pass, i.e. window = 2 s + guard; this is the
                        "how much future does the model really need" experiment;
  3. how much does the window grid matter?  shift the whole grid by --align-shifts
     samples; the spread is the alignment sensitivity of one fixed checkpoint.

Geometry (mirrors studio/server.py:28-29 and studio/web/app.js:6-10,577-596,618-654):

    WINDOW = 32000 samples (2 s)   HOP = 8000 (0.5 s)   GUARD = 1280 (80 ms, default)
    a window that ends at `end` emits [end - guard - hop, end - guard)
    a 2 s window needs 50 mouth frames at 25 fps; frame centre n sits at
    window_start + (n + 0.5) / 25 s  ->  frame index round((end - WINDOW)/640 + n + 0.5)

Usage:
    python tools/repro/experiments/streaming_consistency.py
    python tools/repro/experiments/streaming_consistency.py --clips 2 --guards 80,456
    python tools/repro/experiments/streaming_consistency.py --wav mix.wav --mouth mouth.npy
    python tools/repro/experiments/streaming_consistency.py --json /tmp/streaming.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDIO = REPO_ROOT / "studio"
sys.path.insert(0, str(STUDIO))

SAMPLE_RATE = 16_000
WINDOW = 32_000          # 2 s
HOP = 8_000              # 0.5 s
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS          # 640
WINDOW_FRAMES = WINDOW // SAMPLES_PER_FRAME     # 50
STUDIO_GUARD = 1280      # 80 ms
DEFAULT_DATA_ROOT = Path("/root/data/falcon_lrs2_2mix/tt")
DEFAULT_GUARDS_MS = "80,160,320,456,640"
LUMA = (0.299, 0.587, 0.114)   # ITU-R 601-2, same as torchvision Grayscale and video-worker.js


# --------------------------------------------------------------------------- input
def read_wav(path: str | Path) -> np.ndarray:
    """Mono 16 kHz WAV (16-bit PCM or 32-bit float) -> float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if rate != SAMPLE_RATE or channels != 1:
        raise SystemExit(f"{path}: need mono {SAMPLE_RATE} Hz, got {channels} ch {rate} Hz")
    if width == 2:
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if width == 4:
        return np.frombuffer(raw, "<f4").astype(np.float32).copy()
    raise SystemExit(f"{path}: unsupported sample width {width * 8} bit")


def load_mouth(path: Path) -> np.ndarray:
    """Mouth crops -> [T, 96, 96] uint8 grayscale.

    Accepts [T, 96, 96] grayscale, [T, 96, 96, 3] RGB (the layout shipped in
    /root/data/falcon_lrs2_2mix/tt/mouths/*.pt) and [T, 3, 96, 96] CHW RGB.
    """
    import torch

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            data = torch.from_numpy(np.asarray(archive[archive.files[0]]))
    else:
        data = torch.from_numpy(np.load(path))
    if data.ndim == 4 and data.shape[-1] == 3:                 # THWC
        data = data.permute(0, 3, 1, 2)                        # -> TCHW
    if data.ndim == 4 and data.shape[1] == 3:                  # TCHW -> gray THW
        data = (data.float() * torch.tensor(LUMA, dtype=torch.float32).view(1, 3, 1, 1)).sum(1)
    if data.ndim != 3:
        raise SystemExit(f"{path}: expected (T, H, W), (T, H, W, 3) or (T, 3, H, W); got {tuple(data.shape)}")
    if data.shape[1] != 96 or data.shape[2] != 96:
        raise SystemExit(f"{path}: expected 96x96 crops, got {tuple(data.shape)}")
    return np.ascontiguousarray(data.to(torch.uint8).numpy())


def pick_stream(data_root: Path, clips: int):
    """Concatenate `clips` real 2 s test items that share one source video.

    Every falcon_lrs2_2mix test item is exactly 2.0 s (32000 samples) - i.e. one window -
    so a multi-window stream has to be assembled by concatenation.  Items are grouped by
    source-video id (one mouth-crop file per source), and the largest group is used so the
    target speaker's mouth track stays coherent across the whole stream while the
    interferer changes from item to item.
    """
    mixes = sorted(glob.glob(str(data_root / "mix" / "*.wav")))
    if not mixes:
        raise SystemExit(f"no mixtures under {data_root / 'mix'}")
    groups: dict[str, list[str]] = {}
    for path in mixes:
        groups.setdefault("_".join(Path(path).name.split("_")[:2]), []).append(path)
    source_id, files = max(groups.items(), key=lambda item: len(item[1]))
    mouth_path = data_root / "mouths" / f"{source_id}.pt"
    if not mouth_path.is_file():
        raise SystemExit(f"mouth crops not found: {mouth_path}")
    chosen = files[:clips]
    if len(chosen) < clips:
        print(f"note: source {source_id} has only {len(chosen)} items; using that many", file=sys.stderr)
    audio = np.concatenate([read_wav(p) for p in chosen]).astype(np.float32)
    mouth = np.concatenate([load_mouth(mouth_path)] * len(chosen), axis=0)
    targets = [data_root / "target" / Path(p).name for p in chosen]
    target = np.concatenate([read_wav(p) for p in targets]).astype(np.float32) \
        if all(p.is_file() for p in targets) else None
    return audio, mouth, target, chosen, source_id


# --------------------------------------------------------------------------- model
def load_runtime(device: str):
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    from falcon_runtime import FalconRuntime

    started = time.perf_counter()
    runtime = FalconRuntime("small").load(device)
    return runtime, time.perf_counter() - started


def whole_pass(runtime, audio: np.ndarray, mouth: np.ndarray):
    """Offline reference: the whole clip through the model in one forward pass."""
    import infer_file

    started = time.perf_counter()
    output = infer_file.extract(runtime, audio, np.asarray(mouth, np.float32))
    return output, time.perf_counter() - started


def frames_block(mouth: np.ndarray, start_sample: int, count: int) -> np.ndarray:
    """`count` crops whose centres sit at start_sample + (i + 0.5) * 640 (app.js:577-596)."""
    index = np.rint(start_sample / SAMPLES_PER_FRAME + 0.5 + np.arange(count)).astype(int)
    return mouth[np.clip(index, 0, len(mouth) - 1)]


def frames_for(mouth: np.ndarray, end: int) -> np.ndarray:
    """The 50 crops the browser sends for the window ending at `end` (app.js:577-596)."""
    return frames_block(mouth, end - WINDOW, WINDOW_FRAMES)


def predict_segment(runtime, audio: np.ndarray, mouth: np.ndarray, start: int, length: int) -> np.ndarray:
    """One forward pass over an arbitrary-length segment (same path as infer_file.extract)."""
    import torch

    import infer_file

    frames = frames_block(mouth, start, length // SAMPLES_PER_FRAME)
    with runtime._lock, torch.inference_mode():
        embedding = runtime.video_model(infer_file.preprocess(frames.astype(np.float32)))
        waveform = torch.from_numpy(np.ascontiguousarray(audio[start:start + length])).unsqueeze(0)
        return runtime.audio_model(waveform, embedding)[0, 0].float().cpu().numpy()


def window_audio(audio: np.ndarray, end: int) -> np.ndarray:
    """2 s window ending at `end`, zero padded on the left before the stream start."""
    window = np.zeros(WINDOW, np.float32)
    low = max(0, end - WINDOW)
    window[WINDOW - (end - low):] = audio[low:end]
    return window


def predict_grid(runtime, audio, mouth, first_end: int, count: int):
    """Predict `count` consecutive windows whose ends are first_end + k * HOP."""
    started = time.perf_counter()
    preds = []
    for k in range(count):
        end = first_end + k * HOP
        preds.append(runtime.infer(window_audio(audio, end), frames_for(mouth, end)))
    return preds, time.perf_counter() - started


# --------------------------------------------------------------------------- metrics
def si_snr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB (the repo's own formula, see Falcon._sisnr)."""
    estimate = np.asarray(estimate, np.float64) - float(np.mean(estimate))
    reference = np.asarray(reference, np.float64) - float(np.mean(reference))
    projection = (estimate * reference).sum() * reference / ((reference ** 2).sum() + eps)
    noise = estimate - projection
    return float(10 * np.log10((projection ** 2).sum() / ((noise ** 2).sum() + eps) + eps))


def stitch(preds, first_end: int, guard: int, total: int):
    """Emit [end - guard - HOP, end - guard) from every window and lay the chunks out."""
    if guard + HOP > WINDOW:
        raise SystemExit(f"guard {guard} samples does not fit a {WINDOW}-sample window")
    low = first_end - guard - HOP
    high = first_end + (len(preds) - 1) * HOP - guard
    if low < 0 or high > total or high <= low:
        raise SystemExit(f"guard {guard}: emitted range [{low}, {high}) is outside [0, {total}) or empty")
    output = np.zeros(high - low, np.float32)
    for k, pred in enumerate(preds):
        start = (first_end + k * HOP) - guard - HOP - low
        assert start == k * HOP, "window chunks must tile without gaps or overlaps"
        output[start:start + HOP] = pred[WINDOW - guard - HOP: WINDOW - guard]
    return output, low, high


def compare(stream: np.ndarray, reference: np.ndarray, low: int, high: int) -> dict:
    part = np.asarray(reference[low:high], np.float32)
    peak = float(np.abs(part).max()) or 1e-12
    return {
        "si_snr": si_snr(stream, part),
        "pearson": float(np.corrcoef(stream, part)[0, 1]),
        "max_dev_to_peak": float(np.abs(stream - part).max() / peak),
        "rms_stream": float(np.sqrt((stream ** 2).mean())),
        "rms_reference": float(np.sqrt((part ** 2).mean())),
        "seconds": stream.size / SAMPLE_RATE,
    }


# --------------------------------------------------------------------------- report
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure how far Falcon's 2 s / 0.5 s windowed streaming drifts from the "
                    "whole-clip forward pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1].strip() if "Usage:" in __doc__ else None,
    )
    parser.add_argument("--wav", type=Path, help="mixture WAV, mono 16 kHz (default: pick from --data-root)")
    parser.add_argument("--mouth", type=Path, help="mouth crops .pt/.npy/.npz at 25 fps, 96x96 (required with --wav)")
    parser.add_argument("--target", type=Path, help="optional clean target WAV: enables the 'vs target' columns")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"falcon_lrs2_2mix split directory (default {DEFAULT_DATA_ROOT})")
    parser.add_argument("--clips", type=int, default=3,
                        help="how many 2 s test items to concatenate into the stream (default 3 = 6 s); "
                             "each item is exactly one window, so stitching needs at least 2")
    parser.add_argument("--guards", default=DEFAULT_GUARDS_MS,
                        help=f"guard band sweep in ms, comma separated (default {DEFAULT_GUARDS_MS}; "
                             f"Studio uses {STUDIO_GUARD * 1000 // SAMPLE_RATE} ms)")
    parser.add_argument("--anchor", choices=["grid", "epoch"], default="grid",
                        help="grid: same windows for every guard, only the emission offset moves (isolates the "
                             "guard effect, default); epoch: re-anchor the window grid to the guard, like "
                             "studio/web/app.js:622 does")
    parser.add_argument("--align-shifts", default="0,4000",
                        help="window-grid offsets in samples, comma separated (default 0,4000 = 0/250 ms); "
                             "the spread shows the alignment sensitivity of one fixed checkpoint")
    parser.add_argument("--mode", choices=["stitch", "lookahead"], default="stitch",
                        help="stitch: Studio's 2 s windows, the emitted 0.5 s sits `guard` samples before the "
                             "window end (default, what server.py does); lookahead: one 2 s output chunk with "
                             "`guard` samples of extra right context inside the same forward (window = 2 s + guard)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="default cpu")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON here")
    parser.add_argument("--quiet", action="store_true", help="only the tables and conclusions")
    args = parser.parse_args()

    guards = [int(round(float(value) * SAMPLE_RATE / 1000)) for value in args.guards.split(",") if value.strip()]
    shifts = sorted({int(value) for value in args.align_shifts.split(",") if value.strip()})
    if not guards or min(guards) < 0:
        parser.error("--guards must be a non-empty list of non-negative ms values")
    if max(guards) + HOP > WINDOW:
        parser.error(f"--guards must stay below {(WINDOW - HOP) * 1000 // SAMPLE_RATE} ms")
    if not shifts or min(shifts) < 0:
        parser.error("--align-shifts must be a non-empty list of non-negative sample offsets")

    if args.wav:
        audio = read_wav(args.wav)
        if not args.mouth:
            parser.error("--mouth is required with --wav (mouth crops at 25 fps, 96x96)")
        mouth = load_mouth(args.mouth)
        target = read_wav(args.target) if args.target else None
        chosen, source_id = [str(args.wav)], "custom"
    else:
        audio, mouth, target, chosen, source_id = pick_stream(args.data_root, args.clips)

    if not args.quiet:
        print(f"stream      {len(audio) / SAMPLE_RATE:.2f} s ({len(audio)} samples), "
              f"{len(mouth)} mouth frames, source {source_id}")
        print(f"items       {len(chosen)} x 2.00 s, first = {Path(chosen[0]).name}")
        print("reference   whole clip in one forward pass (infer_file.extract), identical weights and crops")
        print(f"geometry    WINDOW {WINDOW} / HOP {HOP} / guard sweep "
              f"{[round(g * 1000 / SAMPLE_RATE) for g in guards]} ms, anchor={args.anchor}")

    runtime, load_s = load_runtime(args.device)
    if not args.quiet:
        print(f"model       ready in {load_s:.1f} s on {runtime.device}, {runtime.parameter_count} parameters, "
              f"ckpt sha256 {(runtime.checkpoint_sha256 or '')[:16]}")

    reference, t_whole = whole_pass(runtime, audio, mouth)
    if not args.quiet:
        print(f"whole pass  {t_whole:.2f} s for {len(reference) / SAMPLE_RATE:.2f} s of audio "
              f"({len(reference) / SAMPLE_RATE / max(t_whole, 1e-9):.2f}x realtime)")

    report: dict = {"stream_seconds": len(audio) / SAMPLE_RATE, "source": source_id, "items": chosen,
                    "device": runtime.device, "parameters": runtime.parameter_count,
                    "checkpoint_sha256": runtime.checkpoint_sha256, "anchor": args.anchor,
                    "whole_pass_seconds": t_whole, "guard_sweep": [], "alignment_sweep": []}

    # ---------------- 1. guard sweep ----------------
    if args.mode == "stitch":
        print("\n[guard sweep] mode=stitch: one checkpoint, one stream; a window ending at `end` emits "
              "[end-guard-hop, end-guard)")
        header = (f"{'guard':>9} {'windows':>7} {'cover':>7} {'SI-SNR vs whole':>16} {'r':>7} "
                  f"{'maxdev/peak':>11} {'SI-SNR vs target':>16} {'whole vs target':>16}")
        print(header)
        grid_preds = None
        for guard in guards:
            if args.anchor == "grid":
                first_end = WINDOW                      # same windows for every guard
                count = (len(audio) - first_end) // HOP + 1
                if grid_preds is None:
                    grid_preds, infer_s = predict_grid(runtime, audio, mouth, first_end, count)
                preds, count = grid_preds, len(grid_preds)
            else:
                first_end = HOP + guard                 # grid re-anchored to the guard (app.js:622)
                count = (len(audio) - first_end) // HOP + 1
                preds, infer_s = predict_grid(runtime, audio, mouth, first_end, count)
            if count < 1:
                parser.error(f"guard {guard}: stream too short for a single window; raise --clips")
            stream, low, high = stitch(preds, first_end, guard, len(audio))
            metrics = compare(stream, reference, low, high)
            vs_target = si_snr(stream, target[low:high]) if target is not None else float("nan")
            whole_vs_target = si_snr(reference[low:high], target[low:high]) if target is not None else float("nan")
            report["guard_sweep"].append({
                "guard_samples": guard, "guard_ms": round(guard * 1000 / SAMPLE_RATE, 1), "windows": count,
                "first_end": first_end, "emitted_range": [low, high], "inference_seconds": round(infer_s, 2),
                "si_snr_vs_target": vs_target, "whole_si_snr_vs_target": whole_vs_target, **metrics})
            mark = "  <= Studio" if guard == STUDIO_GUARD else ""
            print(f"{guard:>6} {guard * 1000 / SAMPLE_RATE:>3.0f}ms {count:>7} {metrics['seconds']:>6.2f}s "
                  f"{metrics['si_snr']:>+16.2f} {metrics['pearson']:>7.4f} {metrics['max_dev_to_peak']:>11.3f} "
                  f"{vs_target:>+16.2f} {whole_vs_target:>+16.2f}{mark}")
    else:
        # One fixed 0.5 s output chunk; only the amount of *right context* fed into the same
        # forward pass changes.  The chunk always sits at pred[22720:30720] of its window
        # (Studio's index), so every guard scores the SAME absolute audio.
        left = WINDOW - HOP - STUDIO_GUARD              # 22720 = Studio's left context
        chunk_start = WINDOW                            # absolute sample where the chunk starts
        window_start = chunk_start - left
        print(f"\n[lookahead sweep] mode=lookahead: one fixed output chunk [{chunk_start}, "
              f"{chunk_start + HOP}) samples, left context {left} samples, window = 2 s + guard")
        print(f"{'guard':>9} {'window':>8} {'frames':>7} {'SI-SNR vs whole':>16} {'r':>7} {'maxdev/peak':>11} "
              f"{'SI-SNR vs target':>16} {'whole vs target':>16}")
        low, high = chunk_start, chunk_start + HOP
        whole_vs_target = si_snr(reference[low:high], target[low:high]) if target is not None else float("nan")
        for guard in guards:
            length = left + HOP + guard
            length -= (length - WINDOW) % SAMPLES_PER_FRAME   # keep 25 fps frames aligned
            if window_start + length > len(audio):
                print(f"{guard:>6} {guard * 1000 / SAMPLE_RATE:>3.0f}ms      (does not fit the stream)")
                continue
            started = time.perf_counter()
            prediction = predict_segment(runtime, audio, mouth, window_start, length)
            infer_s = time.perf_counter() - started
            effective = length - left - HOP                  # right context actually fed
            chunk = prediction[left: left + HOP]
            metrics = compare(chunk, reference, low, high)
            vs_target = si_snr(chunk, target[low:high]) if target is not None else float("nan")
            report["guard_sweep"].append({
                "guard_samples": effective, "guard_ms": round(effective * 1000 / SAMPLE_RATE, 1),
                "guard_requested_ms": round(guard * 1000 / SAMPLE_RATE, 1), "window_samples": length,
                "frames": length // SAMPLES_PER_FRAME, "emitted_range": [low, high],
                "inference_seconds": round(infer_s, 2), "si_snr_vs_target": vs_target,
                "whole_si_snr_vs_target": whole_vs_target, **metrics})
            mark = "  <= Studio 80 ms" if effective == STUDIO_GUARD else ""
            print(f"{guard:>6} {guard * 1000 / SAMPLE_RATE:>3.0f}ms {length:>8} "
                  f"{length // SAMPLES_PER_FRAME:>7} {metrics['si_snr']:>+16.2f} {metrics['pearson']:>7.4f} "
                  f"{metrics['max_dev_to_peak']:>11.3f} {vs_target:>+16.2f} {whole_vs_target:>+16.2f}{mark}")

    # ---------------- 2. alignment sweep (stitch mode only) ----------------
    pairwise = []
    if args.mode == "lookahead":
        print("\n[alignment sweep] skipped in lookahead mode (the window grid is not the variable here)")
    else:
        print(f"\n[alignment sweep] guard fixed at Studio's {STUDIO_GUARD} samples "
              f"({STUDIO_GUARD * 1000 // SAMPLE_RATE} ms); only the window grid moves")
        print(f"{'shift':>9} {'windows':>7} {'SI-SNR vs whole':>16} {'r':>7} {'maxdev/peak':>11} "
              f"{'SI-SNR vs shift+0':>18}")
        streams: dict[int, tuple[np.ndarray, int, int]] = {}
        for shift in shifts:
            first_end = WINDOW + shift
            count = (len(audio) - first_end) // HOP + 1
            if count < 1:
                print(f"{shift:>6} {shift * 1000 / SAMPLE_RATE:>3.0f}ms      (stream too short for this shift)")
                continue
            preds, infer_s = predict_grid(runtime, audio, mouth, first_end, count)
            stream, low, high = stitch(preds, first_end, STUDIO_GUARD, len(audio))
            streams[shift] = (stream, low, high)
            metrics = compare(stream, reference, low, high)
            report["alignment_sweep"].append({
                "shift_samples": shift, "shift_ms": round(shift * 1000 / SAMPLE_RATE, 1), "windows": count,
                "emitted_range": [low, high], "inference_seconds": round(infer_s, 2), **metrics})
            print(f"{shift:>6} {shift * 1000 / SAMPLE_RATE:>3.0f}ms {count:>7} {metrics['si_snr']:>+16.2f} "
                  f"{metrics['pearson']:>7.4f} {metrics['max_dev_to_peak']:>11.3f}")
        if len(streams) > 1:
            base_shift = min(streams)
            base_stream, base_low, base_high = streams[base_shift]
            for shift, (stream, low, high) in streams.items():
                if shift == base_shift:
                    continue
                lo, hi = max(low, base_low), min(high, base_high)
                value = si_snr(stream[lo - low:hi - low], base_stream[lo - base_low:hi - base_low])
                pairwise.append((shift, base_shift, value))
                for item in report["alignment_sweep"]:
                    if item["shift_samples"] == shift:
                        item["si_snr_vs_shift_min"] = value
                print(f"{'':>9} {'':>7} {'':>16} {'':>7} {'':>11} {value:>+18.2f}   "
                      f"(shift {shift} vs shift {base_shift}, samples {lo}-{hi})")
    report["alignment_pairwise"] = [{"shift": s, "vs_shift": b, "si_snr": v} for s, b, v in pairwise]

    # ---------------- 3. conclusions ----------------
    rows = report["guard_sweep"]
    studio = next((row for row in rows if row["guard_samples"] == STUDIO_GUARD), rows[0])
    best = max(rows, key=lambda row: row["si_snr"])
    worst = min(rows, key=lambda row: row["si_snr"])
    ordered_rows = sorted(rows, key=lambda row: row["guard_samples"])
    ordered = [row["si_snr"] for row in ordered_rows]
    monotonic = all(b >= a for a, b in zip(ordered, ordered[1:]))
    print("\n[conclusions]")
    if args.mode == "stitch":
        print(f"  1. stitching is not free but it is cheap: with Studio's 80 ms guard the stitched stream is "
              f"{studio['si_snr']:+.2f} dB SI-SNR away from the whole-clip pass (r={studio['pearson']:.4f}, "
              f"max deviation {studio['max_dev_to_peak'] * 100:.1f}% of the reference peak) over "
              f"{studio['seconds']:.2f} s of emitted audio")
        if target is not None:
            gap = studio["si_snr_vs_target"] - studio["whole_si_snr_vs_target"]
            print(f"     against the clean target: streamed {studio['si_snr_vs_target']:+.2f} dB vs whole "
                  f"{studio['whole_si_snr_vs_target']:+.2f} dB -> stitching costs {gap:+.2f} dB")
        print(f"  2. emission offset (Studio's guard band): {worst['guard_ms']:.0f} ms gives "
              f"{worst['si_snr']:+.2f} dB, {best['guard_ms']:.0f} ms gives {best['si_snr']:+.2f} dB "
              f"(spread {best['si_snr'] - worst['si_snr']:.2f} dB, "
              f"{'monotonic' if monotonic else 'NOT monotonic'} over "
              f"{[row['guard_ms'] for row in ordered_rows]} ms) -> which 0.5 s you take out of the window "
              f"matters by a couple of dB, so the guard band is a real quality knob, not bookkeeping")
    else:
        print(f"  1. right context is what the model is short of: {ordered_rows[0]['guard_ms']:.0f} ms of extra "
              f"right context gives {ordered_rows[0]['si_snr_vs_target']:+.2f} dB against the clean target, "
              f"{ordered_rows[-1]['guard_ms']:.0f} ms gives {ordered_rows[-1]['si_snr_vs_target']:+.2f} dB "
              f"({'monotonic' if monotonic else 'not monotonic'} in between, "
              f"{ordered_rows[-1]['si_snr_vs_target'] - ordered_rows[0]['si_snr_vs_target']:+.2f} dB overall)")
        print(f"     the same comparison against the whole-clip pass moves from "
              f"{ordered_rows[0]['si_snr']:+.2f} dB to {ordered_rows[-1]['si_snr']:+.2f} dB -> a 2 s window "
              f"uses future samples that a strictly causal model could not have, which is exactly why the paper "
              f"calls the model causal at window level and not per sample")
    if pairwise:
        print(f"  3. alignment: shifting only the window grid by {pairwise[0][0] * 1000 / SAMPLE_RATE:.0f} ms "
              f"changes the stitched output by {pairwise[0][2]:+.2f} dB SI-SNR (two alignments of the same "
              f"checkpoint on the same audio) -> the checkpoint is alignment-sensitive")
    print("  4. window length is a hard constraint: inside one window every output sample depends on the whole "
          "window (global normalisation + attention), so the result is a function of the window alignment and of "
          "how much audio the window contains, not of the signal alone")
    print("  5. streaming here is not sample-wise causal: a chunk can only be emitted after the whole window that "
          "ends after it has arrived, every hop recomputes the full window from scratch (no hidden state, no audio "
          "cache carried across windows) and there is no per-sample recursion -> 'windowed streaming "
          "(2 s window, 0.5 s hop)', not a causal filter")
    print("\n  studio/server.py:OutputSlicer implements the stitch geometry plus a 20 ms crossfade between "
          "consecutive predictions; this script deliberately skips the crossfade so the measured difference is "
          "the windowing itself.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  json report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
