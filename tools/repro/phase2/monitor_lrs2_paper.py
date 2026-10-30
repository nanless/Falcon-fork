#!/usr/bin/env python3
"""监控 LRS2-2Mix 论文配方训练：读取实验目录下 metrics.csv，输出最新 epoch/val 指标/吞吐/ETA。

续训（``audio_train.py --ckpt_path ...``）会让 CSVLogger 新建 ``version_1``、``version_2`` …
目录，因此这里用 glob ``version_*`` 找**版本号最大**（同版本按 mtime 最新）的 metrics.csv，
而不是硬编码 ``version_0``（旧实现会一直读续训前的旧文件）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

DEFAULT_ROOTS = [
    "/root/tmp/*/Experiments/checkpoint",
    "Experiments/checkpoint",
]

# 兼容旧名字（历史文档/脚本可能 import 它）
DEFAULT_EXP = "LRS2_2Mix_Paper_v2"
LEGACY_EXP = "LRS2_2Mix_Paper_8gpu"


def _version_key(path: Path) -> tuple[int, float]:
    """Sort key for ``.../version_<N>/metrics.csv``: numeric version, then mtime."""
    version = -1
    parent = path.parent.name  # version_<N>
    if parent.startswith("version_"):
        try:
            version = int(parent.split("_", 1)[1])
        except ValueError:
            version = -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (version, mtime)


def find_metrics(exp_name: str, roots: list[str] | None = None) -> Path | None:
    """Return the newest ``metrics.csv`` for ``exp_name`` across known roots.

    Each root is a ``.../Experiments/checkpoint``-style directory (may contain
    ``*``); we look at ``<root>/<exp_name>/version_*/metrics.csv`` and pick the
    highest version number (ties broken by newest mtime).  This is resume-safe:
    a resumed run writes ``version_1`` while the old data stays in ``version_0``.
    """
    bases = [os.path.join(root, exp_name) for root in (roots or DEFAULT_ROOTS)]
    hits: list[Path] = []
    for base in bases:
        for exp_dir in glob.glob(base):
            for version_dir in glob.glob(os.path.join(exp_dir, "version_*")):
                metrics = os.path.join(version_dir, "metrics.csv")
                if os.path.isfile(metrics):
                    hits.append(Path(metrics))
    if not hits:
        return None
    return max(hits, key=_version_key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=DEFAULT_EXP,
                    help=f"exp_name（默认 {DEFAULT_EXP}；旧实验用 {LEGACY_EXP}）")
    ap.add_argument("--watch", type=int, default=0, help="每 N 秒刷新（0=只打印一次）")
    ap.add_argument("--root", action="append", default=None,
                    help="额外的 checkpoint 根目录（可重复；默认 /root/tmp/*/Experiments/checkpoint 与 ./Experiments/checkpoint）")
    args = ap.parse_args()
    while True:
        path = find_metrics(args.exp, args.root)
        if not path:
            print(f"[{time.strftime('%F %T')}] metrics.csv not found for exp={args.exp}")
        else:
            rows = list(csv.DictReader(open(path)))
            vals = [(int(r["epoch"]), float(r["val_pit_sisnr"]), r.get("step")) for r in rows
                    if r.get("val_pit_sisnr") not in (None, "")]
            steps = [int(r["step"]) for r in rows if r.get("step")]
            last_step = max(steps) if steps else 0
            print(f"[{time.strftime('%F %T')}] exp={args.exp} version={path.parent.name} file={path}")
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
