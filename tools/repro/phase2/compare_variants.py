#!/usr/bin/env python
"""Phase-2 B: Falcon-small vs Falcon-big, whole-clip vs Studio-windowed, and visual ablations.

Evaluates released checkpoints on a JSON-index test split (e.g. LRS2-2Mix tt converted to the
repository layout) with the Phase-1 corrected metrics (:class:`look2hear.metrics.ALLMetricsTracker`).

Three axes
----------
* variant  : ``small`` (``falcon.pt``, ``FalconSpeakerContrastive``) vs ``big`` (``falcon-big.pt``,
  ``SpeakerContrastiveV2`` from ``studio/falcon_model``);
* protocol : ``whole`` (clip fed in one pass — what ``audio_visual_test.py`` does) vs ``windowed``
  (Studio streaming simulation: 2 s window / 0.5 s hop / 80 ms guard / 320-sample crossfade);
* visual   : ``real`` mouth crops vs ``zero`` (all-zero crops) vs ``shuffle`` (crops rolled by
  ``--shuffle-ms`` ms) — visual-branch contribution.

Usage
-----
::

    PY=/root/miniforge3/envs/falcon/bin/python
    PYTHONPATH=$PWD $PY tools/repro/phase2/compare_variants.py \
        --variants small,big --protocols whole,windowed --visuals real \
        --data-dir /root/tmp/lrs2_eval_tt_full --limit 300 --device cuda \
        --out-dir docs/verification/phase2

    # visual ablation (300 utts, whole-clip, small model)
    PYTHONPATH=$PWD $PY tools/repro/phase2/compare_variants.py \
        --variants small --protocols whole --visuals real,zero,shuffle \
        --data-dir /root/tmp/lrs2_eval_tt_full --limit 300 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[3]
CKPT_DIR = Path("/root/.cache/falcon_weights")
VARIANTS = {
    "small": {
        "config": REPO / "configs" / "Falcon.yml",
        "ckpt": CKPT_DIR / "falcon.pt",
    },
    "big": {
        "config": REPO / "studio" / "configs" / "FalconBig.yml",
        "ckpt": CKPT_DIR / "falcon-big.pt",
    },
}

# Studio streaming constants (studio/server.py: SR, WINDOW, HOP, GUARD, FADE)
SR = 16_000
WINDOW = 32_000          # 2.0 s
HOP = 8_000              # 0.5 s
GUARD = 1_280            # 80 ms
FADE = 320               # 20 ms crossfade
LEFT_CONTEXT = WINDOW - GUARD - HOP          # 22,720 samples = 1.42 s (first window's left padding)
FIRST_ENDPOINT = LEFT_CONTEXT + HOP          # 30,720? see note below
CHUNK_LEN = HOP


def _resolve_tt_dir(path: Path) -> Path:
    for cand in (path / "tt", path):
        if (cand / "mix.json").is_file() and (cand / "target.json").is_file():
            return cand
    raise FileNotFoundError(f"no tt/mix.json+target.json under {path}")


# --------------------------------------------------------------------------- models
def load_models(variant: str, device: str):
    """Build (audio_model, video_model) for a variant and strict-load released weights."""
    sys.path.insert(0, str(REPO))
    info = VARIANTS[variant]
    conf = yaml.safe_load(info["config"].read_text())
    ckpt = torch.load(info["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    audio_state = {k[len("audio_model."):]: v for k, v in state.items() if k.startswith("audio_model.")}
    video_state = {k[len("video_model."):]: v for k, v in state.items() if k.startswith("video_model.")}
    audio_name = conf["audionet"]["audionet_name"]

    if variant == "small":
        import look2hear.models as models

        model_cls = getattr(models, audio_name)  # FalconSpeakerContrastive
    else:
        # big model only exists in the vendored Studio copy
        from studio.falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2

        model_cls = SpeakerContrastiveV2
        assert audio_name == "SpeakerContrastiveV2", audio_name

    audio_model = model_cls(sample_rate=SR, **copy.deepcopy(conf["audionet"]["audionet_config"]))
    audio_model.load_state_dict(audio_state, strict=True)

    from look2hear.videomodels import FRCNNVideoModel

    vcfg = copy.deepcopy(conf["videonet"]["videonet_config"])
    vcfg.pop("pretrain", None)
    video_model = FRCNNVideoModel(**vcfg)
    video_model.load_state_dict(video_state, strict=True)

    audio_model.to(device).eval()
    video_model.to(device).eval()
    n = sum(p.numel() for p in audio_model.parameters()) + sum(p.numel() for p in video_model.parameters())
    print(f"[{variant}] audio={sum(p.numel() for p in audio_model.parameters()):,} "
          f"video={sum(p.numel() for p in video_model.parameters()):,} total={n:,} "
          f"| ckpt={info['ckpt'].name} epoch={ckpt.get('epoch')}")
    return audio_model, video_model


# --------------------------------------------------------------------------- mouth
def mouth_to_tensor(mouth: np.ndarray, device: str) -> torch.Tensor:
    """``[F,H,W]`` numpy (already lip-reading preprocessed) -> ``[1,1,F,H,W]`` float32."""
    t = torch.from_numpy(np.asarray(mouth)).float()
    if t.dim() != 3:
        raise ValueError(f"expected [F,H,W] mouth crops, got {tuple(t.shape)}")
    return t.unsqueeze(0).unsqueeze(0).to(device)


def window_mouth(frames: torch.Tensor, window_start_sample: int, n_frames: int = 50) -> torch.Tensor:
    """Select/replicate 25-fps crops for a 2 s window starting at ``window_start_sample``.

    Dataset crops are indexed by absolute frame (frame ``f`` covers ``[f*640, (f+1)*640)`` samples).
    Out-of-range frames (silence padding at the clip edges) are edge-replicated, which mirrors the
    front-end behaviour of keeping the nearest available crop.
    """
    f0 = int(round(window_start_sample / (SR // 25)))
    idx = torch.arange(f0, f0 + n_frames, device=frames.device)
    idx = idx.clamp(0, frames.shape[0] - 1)
    return frames[idx]


# --------------------------------------------------------------------------- inference
@torch.inference_mode()
def infer_whole(model, video_model, audio: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    mouth_emb = video_model(mouth_to_tensor(frames.cpu().numpy(), audio.device))
    return model(audio.unsqueeze(0), mouth_emb)[0, 0]


@torch.inference_mode()
def infer_windowed(model, video_model, audio: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Simulate the Studio 2 s / 0.5 s streaming protocol and stitch the emitted chunks."""
    length = audio.numel()
    padded = torch.cat([audio.new_zeros(LEFT_CONTEXT), audio, audio.new_zeros(WINDOW)])
    chunks, future = [], None
    k = 0
    while True:
        endpoint = FIRST_ENDPOINT + HOP * k
        chunk_start = endpoint - (WINDOW - GUARD)          # emitted chunk start in stream coords
        if chunk_start >= length:
            break
        win = padded[chunk_start:chunk_start + WINDOW]
        if win.numel() < WINDOW:
            win = F.pad(win, (0, WINDOW - win.numel()))
        window_start_stream = chunk_start - LEFT_CONTEXT
        wframes = window_mouth(frames, window_start_stream)
        mouth_emb = video_model(mouth_to_tensor(wframes.cpu().numpy(), audio.device))
        pred = model(win.unsqueeze(0), mouth_emb)[0, 0]     # [WINDOW]
        chunk = pred[WINDOW - GUARD - HOP: WINDOW - GUARD].clone()   # [HOP]
        if future is not None:                              # crossfade with previous tail
            ramp = torch.linspace(0, 1, FADE, device=chunk.device)
            chunk[:FADE] = future * (1 - ramp) + chunk[:FADE] * ramp
        future = pred[WINDOW - GUARD: WINDOW - GUARD + FADE].clone()
        n = min(HOP, length - chunk_start)
        chunks.append(chunk[:n])
        k += 1
    out = torch.cat(chunks)
    if out.numel() < length:
        out = F.pad(out, (0, length - out.numel()))
    return out[:length]


# --------------------------------------------------------------------------- evaluation
def run_one(dataset, indices, variant, protocol, visual, device, out_dir: Path, shuffle_ms: float,
            tag: str):
    audio_model, video_model = load_models(variant, device)
    from look2hear.metrics import ALLMetricsTracker

    csv_path = out_dir / f"P2B-{tag}-{variant}-{protocol}-{visual}.csv"
    tracker = ALLMetricsTracker(save_file=str(csv_path), sample_rate=SR)
    shift = int(round(shuffle_ms / 1000 * SR // (SR // 25))) if visual == "shuffle" else 0

    t0 = time.time()
    for n, idx in enumerate(indices):
        mix, clean, mouth, key, _ = dataset[idx]
        audio = torch.as_tensor(mix, dtype=torch.float32, device=device)
        frames = torch.from_numpy(np.asarray(mouth)).float()
        if visual == "zero":
            frames = torch.zeros_like(frames)
        elif visual == "shuffle":
            frames = torch.roll(frames, shifts=shift, dims=0)
        if protocol == "whole":
            est = infer_whole(audio_model, video_model, audio, frames)
        else:
            est = infer_windowed(audio_model, video_model, audio, frames)
        tracker(mix=mix, clean=clean.unsqueeze(0), estimate=est.unsqueeze(0).cpu(), key=key)
        if n % 100 == 99 or n == len(indices) - 1:
            mean = tracker.get_mean()
            elapsed = time.time() - t0
            eta = elapsed / (n + 1) * (len(indices) - n - 1)
            print(f"  [{variant}/{protocol}/{visual}] {n + 1}/{len(indices)} "
                  f"SI-SNRi={mean['si-snr_i']:.3f} SDRi={mean['sdr_i']:.3f} "
                  f"({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)
    mean, std = tracker.get_mean(), tracker.get_std()
    tracker.final()
    wall = time.time() - t0
    row = {
        "variant": variant, "protocol": protocol, "visual": visual, "n": len(indices),
        "si_snr": mean["si-snr"], "si_snr_i": mean["si-snr_i"],
        "sdr": mean["sdr"], "sdr_i": mean["sdr_i"],
        "si_snr_i_std": std["si-snr_i"], "sdr_i_std": std["sdr_i"],
        "pesq": mean["pesq"], "stoi": mean["stoi"],
        "wall_s": round(wall, 1), "csv": csv_path.name,
    }
    print(f"[RESULT] {json.dumps(row, ensure_ascii=False)}")
    del audio_model, video_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/root/tmp/lrs2_eval_tt_full"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all utterances")
    ap.add_argument("--variants", default="small", help="comma list: small,big")
    ap.add_argument("--protocols", default="whole", help="comma list: whole,windowed")
    ap.add_argument("--visuals", default="real", help="comma list: real,zero,shuffle")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--shuffle-ms", type=float, default=1000.0)
    ap.add_argument("--out-dir", type=Path, default=REPO / "docs" / "verification" / "phase2")
    ap.add_argument("--tag", default="full", help="prefix for CSV/summary files")
    args = ap.parse_args(argv)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    sys.path.insert(0, str(REPO))
    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = _resolve_tt_dir(args.data_dir)
    dataset = AVSpeechEvalDataset(json_dir=str(test_dir), sample_rate=SR, normalize_audio=False)
    indices = list(range(len(dataset)))
    if args.limit > 0:
        indices = indices[: args.limit]
    print(f"[setup] device={device} test_dir={test_dir} utterances={len(indices)}")

    rows = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for protocol in [p.strip() for p in args.protocols.split(",") if p.strip()]:
            for visual in [v.strip() for v in args.visuals.split(",") if v.strip()]:
                print(f"\n=== run: variant={variant} protocol={protocol} visual={visual} "
                      f"n={len(indices)} ===", flush=True)
                rows.append(run_one(dataset, indices, variant, protocol, visual, device,
                                    args.out_dir, args.shuffle_ms, args.tag))

    vtag = args.variants.replace(",", "+")
    summary = args.out_dir / f"P2B-summary-{args.tag}-{vtag}.json"
    summary.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[summary] wrote {summary}")
    print("| variant | protocol | visual | n | SI-SNRi | SDRi | PESQ | STOI | wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['protocol']} | {r['visual']} | {r['n']} | "
              f"{r['si_snr_i']:.3f} | {r['sdr_i']:.3f} | {r['pesq']:.3f} | {r['stoi']:.3f} | "
              f"{r['wall_s']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
