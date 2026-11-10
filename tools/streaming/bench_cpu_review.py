"""Measure CPU chunk processing on repeated real LRS2 input, optionally paced."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from tools.streaming.common import load_mono
from tools.streaming.session import StreamingSession


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {key: float(np.percentile(a, p)) for key, p in [('p50',50),('p90',90),('p95',95),('p99',99),('max',100)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--index-dir', required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--chunks', type=int, default=200)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--paced', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.threads < 1 or a.chunks < 1 or a.warmup < 0:
        p.error('positive threads/chunks and nonnegative warmup are required')
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    model_hash_before = digest('look2hear/models/FalconStreaming.py')
    checkpoint_hash_before = digest(a.checkpoint)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    root = Path(a.index_dir)
    mixture = json.loads((root/'mix.json').read_text())[0]
    target = json.loads((root/'target.json').read_text())[0]
    if Path(mixture[0]).name != Path(target[0]).name: raise ValueError('mixed index pairing')
    wave = load_mono(mixture[0]).numpy()
    frames = torch.load(target[1], map_location='cpu', weights_only=True).numpy()
    available = min(len(wave)//2560, len(frames)//4)
    if available < 1: raise ValueError('not enough aligned audio/video')
    session = StreamingSession(a.checkpoint, device='cpu')
    def push(i, seq):
        j = i % available
        base = seq * 2560
        return session.push(wave[j*2560:(j+1)*2560],frames[j*4:(j+1)*4],sequence=seq,
                            start_sample=base,video_samples=[base+k*640 for k in range(4)])
    for i in range(a.warmup): push(i,i)
    session.reset()
    duration, delay, finish = [], [], []
    started = time.perf_counter()
    peak_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for i in range(a.chunks):
        release = started + i * .160
        if a.paced:
            remaining = release - time.perf_counter()
            if remaining > 0: time.sleep(remaining)
        else:
            release = time.perf_counter()
        beg = time.perf_counter()
        y = push(i,i)
        end = time.perf_counter()
        if y.shape != (2560,) or not np.isfinite(y).all(): raise RuntimeError('invalid output')
        duration.append((end-beg)*1000)
        delay.append(max(0.,(beg-release)*1000))
        finish.append((end-release)*1000)
    ended = time.perf_counter()
    model_hash_after = digest('look2hear/models/FalconStreaming.py')
    checkpoint_hash_after = digest(a.checkpoint)
    if model_hash_before != model_hash_after or checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError('model or checkpoint changed during measurement')
    cpu_name = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_name = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    result = {
        'date': '2026-11-11', 'scope':'CPU session.push including crop/normalize, audio+video model and output copy; excludes capture, tracking, network, playback',
        'inputs':'repeats complete 160ms chunks from one real LRS2 test utterance; repeated boundaries are synthetic',
        'cpu':cpu_name, 'threads':a.threads, 'torch_threads':torch.get_num_threads(),
        'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'checkpoint':str(Path(a.checkpoint).resolve()), 'checkpoint_sha256':checkpoint_hash_before,
        'model_sha256':model_hash_before,
        'audio':mixture[0], 'audio_sha256':digest(mixture[0]), 'video':target[1], 'video_sha256':digest(target[1]),
        'torch_version':torch.__version__, 'warmup':a.warmup, 'chunks':a.chunks, 'paced':a.paced,
        'unique_source_chunks':available, 'duration_ms':stats(duration),
        'release_to_start_lateness_ms':stats(delay) if a.paced else None,
        'release_to_finish_ms':stats(finish) if a.paced else None,
        'processing_over_140ms':sum(x > 140 for x in duration),
        'processing_over_160ms':sum(x > 160 for x in duration),
        'paced_release_to_finish_over_140ms':sum(x > 140 for x in finish) if a.paced else None,
        'paced_release_to_finish_over_160ms':sum(x > 160 for x in finish) if a.paced else None,
        'wall_seconds':ended-started,
        'peak_rss_kib_before':peak_rss_before,
        'peak_rss_kib_after':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'measurements_ms':{'processing':duration,'release_to_start':delay if a.paced else None,'release_to_finish':finish if a.paced else None},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cpu_affinity','measurements_ms')},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
