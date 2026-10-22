#!/usr/bin/env python
"""Phase-2 B diagnostic: sanity-check the windowed (Studio) inference simulation.

Compares, on a handful of utterances, the **signal** produced by

* ``whole``   : the 2 s clip fed in one pass (what ``audio_visual_test.py`` does), and
* ``windowed``: the Studio 2 s/0.5 s/80 ms guard protocol simulation from
  ``compare_variants.py``.

Reports the SNR between the two estimates, their correlation, and the size of the
crossfade discontinuities at chunk boundaries (should be small relative to the signal),
so that the reported metric difference can be attributed to the protocol rather than a bug.

Usage::

    PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/phase2/windowed_diag.py \
        --data-dir /root/tmp/lrs2_eval_tt_full --limit 5 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "repro" / "phase2"))

from compare_variants import (  # noqa: E402
    HOP, SR, WINDOW, _resolve_tt_dir, infer_whole, infer_windowed, load_models,
)


def snr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = (est * ref).sum() / (ref.pow(2).sum() + 1e-12)
    proj = alpha * ref
    noise = est - proj
    return float(10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + 1e-12) + 1e-12))


def boundary_ratio(est: torch.Tensor) -> float:
    """max |jump| at 0.5 s chunk boundaries / median |sample-to-sample diff|."""
    e = est.detach().cpu().numpy()
    jumps = [abs(e[b] - e[b - 1]) for b in range(HOP, len(e), HOP) if b < len(e)]
    if not jumps:
        return float("nan")
    med = float(np.median(np.abs(np.diff(e)))) + 1e-12
    return max(jumps) / med


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/root/tmp/lrs2_eval_tt_full"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args(argv)

    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    test_dir = _resolve_tt_dir(args.data_dir)
    ds = AVSpeechEvalDataset(json_dir=str(test_dir), sample_rate=SR, normalize_audio=False)
    model, vm = load_models("small", device)

    print(f"# windowed vs whole-clip signal diagnosis (n={args.limit}, device={device})")
    print("| idx | SNR(est_win, est_whole) dB | corr | max|jump|/median|diff| (win) | "
          "max|jump|/median|diff| (whole) |")
    print("|---|---|---|---|---|")
    rows = []
    for idx in range(args.limit):
        mix, clean, mouth, key, _ = ds[idx]
        audio = torch.as_tensor(mix, dtype=torch.float32, device=device)
        frames = torch.from_numpy(np.asarray(mouth)).float()
        est_whole = infer_whole(model, vm, audio, frames).float().cpu()
        est_win = infer_windowed(model, vm, audio, frames).float().cpu()
        corr = float(np.corrcoef(est_whole.numpy(), est_win.numpy())[0, 1])
        snr = snr_db(est_whole, est_win)
        bw, bm = boundary_ratio(est_win), boundary_ratio(est_whole)
        rows.append((snr, corr, bw))
        print(f"| {idx} | {snr:.2f} | {corr:.4f} | {bw:.2f} | {bm:.2f} |")
        print(f"  key={key}")
    if rows:
        arr = np.asarray(rows, dtype=np.float64)
        print(f"\nmean SNR={arr[:, 0].mean():.2f} dB | mean corr={arr[:, 1].mean():.4f} | "
              f"mean boundary ratio={arr[:, 2].mean():.2f}")
        print("Interpretation: a high SNR/corr means the windowed simulation tracks the whole-clip "
              "output; metric differences then come from protocol edge/padding effects, not a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
