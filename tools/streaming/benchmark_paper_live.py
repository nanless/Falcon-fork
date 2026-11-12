"""Measure single-stream CPU processing of the checkpoint-backed online API.

The capture clock and detection/preprocessing are outside this benchmark.
Measures both an output-triggering push and the sum of all pushes since the
previous output. Random preprocessed frames are repeated deterministically.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tools.streaming.infer_paper_lookahead import digest
from tools.streaming.paper_live_inference import PaperLiveInference


def distribution(values):
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
        "count_ge_60ms": sum(x >= 60 for x in values),
        "all_ms": values,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="new JSON path under voiceprint/experiments")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=931)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.warmup < 0 or args.blocks < 1:
        parser.error("threads and blocks must be positive; warmup must be nonnegative")
    root = Path("/root/group-shared/voiceprint/experiments").resolve()
    out = Path(args.output).resolve()
    if not out.is_relative_to(root) or out == root:
        parser.error("output must be under voiceprint/experiments")
    if out.exists():
        raise FileExistsError(out)
    source_root = Path(__file__).resolve().parents[2]
    code_files = [
        "tools/streaming/benchmark_paper_live.py",
        "tools/streaming/paper_live_inference.py",
        "tools/streaming/lookahead_session.py",
        "tools/streaming/infer_paper_lookahead.py",
        "look2hear/models/FalconPaperStreamPrototype.py",
        "look2hear/models/FalconLookaheadVisual.py",
        "look2hear/videomodels/frcnn_videomodel.py",
    ]
    code_hashes = {name: digest(source_root / name) for name in code_files}
    live = PaperLiveInference(args.checkpoint, device="cpu", precision=args.precision, threads=args.threads)
    rng = np.random.default_rng(args.seed)
    tick = rng.normal(size=640).astype(np.float32)
    mouth = rng.normal(size=(88, 88)).astype(np.float32)
    emitting, aggregate = [], []
    accumulated = 0.0
    emitted = 0
    for sequence in range(3 * (args.warmup + args.blocks) + 2):
        started = time.perf_counter_ns()
        chunks = live.push(tick, mouth, sequence=sequence)
        duration = (time.perf_counter_ns() - started) / 1e6
        accumulated += duration
        if chunks:
            if len(chunks) != 1 or chunks[0].shape != (1920,) or not np.isfinite(chunks[0]).all():
                raise RuntimeError("unexpected output during benchmark")
            emitted += 1
            if emitted > args.warmup:
                emitting.append(duration)
                aggregate.append(accumulated)
            accumulated = 0.0
    if len(emitting) != args.blocks:
        raise RuntimeError("benchmark emitted an unexpected number of blocks")
    cpu_name = ""
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        cpu_name = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                         if line.startswith("model name")), "")
    try:
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                         text=True, timeout=5).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": live.checkpoint_path,
        "checkpoint_sha256": live.checkpoint_sha256,
        "model_config": {key: live.config.get(key) for key in ("width", "rnn_hidden", "heads", "history_frames", "mask_limit")},
        "precision": args.precision,
        "seed": args.seed,
        "input": "deterministic repeated random mono tick and already normalized 88x88 mouth frame",
        "platform": platform.platform(),
        "cpu_model": cpu_name,
        "cpu_count": os.cpu_count(),
        "threads": args.threads,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "git_head": commit,
        "source_sha256": code_hashes,
        "warmup_blocks": args.warmup,
        "measured_blocks": args.blocks,
        "scope": "CPU API processing time including input conversion, buffering, model and output conversion; excludes capture, face detection, network, and wall-clock input collection",
        "emitting_push": distribution(emitting),
        "sum_pushes_per_output": distribution(aggregate),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("emitting_push", "sum_pushes_per_output")}, ensure_ascii=False))
    print(json.dumps({name: {k: v for k, v in report[name].items() if k != "all_ms"}
                      for name in ("emitting_push", "sum_pushes_per_output")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
