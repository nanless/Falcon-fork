"""Measure the stateful API on continuous synthetic input; excludes capture and playback."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .session import StreamingSession
from .common import SAMPLES, FRAMES, SAMPLES_PER_FRAME, SR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--chunks', type=int, default=500)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.chunks < 1:
        parser.error('warmup must be nonnegative and chunks positive')
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(314159)
    session = StreamingSession(args.checkpoint, device=args.device)
    samples = rng.normal(0, .1, size=(SAMPLES,)).astype(np.float32)
    mouth = rng.integers(0, 256, size=(FRAMES, 96, 96), dtype=np.uint8)
    times = []
    memory = []
    for i in range(args.warmup + args.chunks):
        start_sample = i * SAMPLES
        frame_samples = [start_sample + j * SAMPLES_PER_FRAME for j in range(FRAMES)]
        start = time.perf_counter_ns()
        out = session.push(samples, mouth, sequence=i, start_sample=start_sample,
                           video_samples=frame_samples)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        if out.shape != (SAMPLES,) or not np.isfinite(out).all():
            raise RuntimeError('bad output')
        if i >= args.warmup:
            times.append(elapsed)
            if session.device.type == 'cuda':
                memory.append(torch.cuda.memory_allocated(session.device))
    result = {
        'scope': 'session.push including CPU crop/normalization, transfer, model, and output copy; synthetic already-captured inputs; no tracking/network/capture/playback',
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'training_status': session.training_status,
        'device': str(session.device),
        'gpu_name': torch.cuda.get_device_name(session.device) if session.device.type == 'cuda' else None,
        'torch': torch.__version__, 'python': platform.python_version(),
        'warmup_chunks': args.warmup, 'chunks': args.chunks,
        'sample_rate': SR, 'chunk_samples': SAMPLES, 'chunk_video_frames': FRAMES,
        'chunk_ms': 1000 * SAMPLES / SR,
        'simulated_audio_seconds': args.chunks * SAMPLES / SR,
        'latency_ms': {'p50': float(np.percentile(times, 50)), 'p90': float(np.percentile(times, 90)),
                       'p95': float(np.percentile(times, 95)), 'p99': float(np.percentile(times, 99)),
                       'max': float(max(times))},
        'memory_allocated_bytes': {'first': memory[0], 'last': memory[-1], 'max': max(memory)} if memory else None,
        'output_samples': args.chunks * SAMPLES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
