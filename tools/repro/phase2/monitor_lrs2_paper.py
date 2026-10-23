#!/usr/bin/env python3
"""监控 LRS2-2Mix 论文配方训练：读取实验目录下 metrics.csv，输出最新 epoch/val 指标/吞吐/ETA。"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

DEFAULT_ROOTS = [
    "/root/tmp/p2d_paper_train/Experiments/checkpoint",
    "Experiments/checkpoint",
]


def find_metrics(exp_name: str) -> Path | None:
    pats = [f"/root/tmp/*/Experiments/checkpoint/{exp_name}/version_0/metrics.csv",
            f"Experiments/checkpoint/{exp_name}/version_0/metrics.csv",
            f"/root/**/{exp_name}/version_0/metrics.csv"]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive="**" in pat))
        if hits:
            return Path(hits[-1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="LRS2_2Mix_Paper_8gpu", help="exp_name")
    ap.add_argument("--watch", type=int, default=0, help="每 N 秒刷新（0=只打印一次）")
    args = ap.parse_args()
    while True:
        path = find_metrics(args.exp)
        if not path:
            print(f"[{time.strftime('%F %T')}] metrics.csv not found for exp={args.exp}")
        else:
            rows = list(csv.DictReader(open(path)))
            vals = [(int(r["epoch"]), float(r["val_pit_sisnr"]), r.get("step")) for r in rows
                    if r.get("val_pit_sisnr") not in (None, "")]
            steps = [int(r["step"]) for r in rows if r.get("step")]
            last_step = max(steps) if steps else 0
            print(f"[{time.strftime('%F %T')}] exp={args.exp} file={path}")
            if vals:
                best = max(vals, key=lambda x: x[1])
                print(f"  epochs={len(vals)} last=ep{vals[-1][0]} val_pit_sisnr={vals[-1][1]:.4f} "
                      f"best=ep{best[0]} {best[1]:.4f} last_step={last_step}")
            else:
                print(f"  no val row yet; last_step={last_step}")
        if not args.watch:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
