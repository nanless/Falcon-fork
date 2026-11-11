"""Time the complete visual-plus-audio model on a continuous CPU stream."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from look2hear.models.FalconPaperLookahead import FalconPaperLookahead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visual-init", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--history-frames", type=int, default=125)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--blocks", type=int, default=500)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = FalconPaperLookahead(args.visual_init, width=args.width, rnn_hidden=args.hidden,
                                  history_frames=args.history_frames).eval()
    # Fixed stream is random input for computational measurement, not a quality test.
    audio = torch.randn(1, 3200)
    video = torch.randn(1, 5, 88, 88)
    timings = []
    state = None
    with torch.inference_mode():
        for i in range(args.warmup + args.blocks):
            before = time.perf_counter_ns()
            output, state = model(audio, video, state)
            elapsed = (time.perf_counter_ns() - before) / 1e6
            if i >= args.warmup:
                timings.append(elapsed)
    data = {"configuration": vars(args), "platform": platform.platform(),
            "torch_version": str(torch.__version__), "cpu_count": os.cpu_count(),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)), "max_ms": max(timings),
            "count_at_or_above_60ms": sum(t >= 60 for t in timings), "all_ms": timings}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in data.items() if k != "all_ms"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
