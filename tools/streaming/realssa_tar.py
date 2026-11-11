"""Indexed, process-safe access to RealSSA-Sim tar shards without extraction.

Index the tar member byte offsets once; no audio or image payload is copied.
The four published fields must occur exactly once for every retained sample.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
import tempfile
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import rgb_to_grayscale

from tools.streaming.common import SR, SAMPLES_PER_FRAME

FORMAT = "realssa-tar-index-v1"
SUFFIXES = {".mix.wav": "mix", ".target.wav": "target", ".mouths.pt": "mouth", ".json": "meta"}
REQUIRED = frozenset(SUFFIXES.values())
SEGMENT_FRAMES = 50
SEGMENT_SAMPLES = SEGMENT_FRAMES * SAMPLES_PER_FRAME


def _member(name):
    name = Path(name).name
    for suffix, kind in SUFFIXES.items():
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)], kind
    return None


def _scan_one(shard, cache_dir):
    stat = shard.stat()
    identity = {"path": str(shard), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache_file = None
    if cache_dir is not None:
        token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_file = cache_dir / (token + ".json")
        if cache_file.exists():
            with cache_file.open(encoding="utf8") as handle:
                cached = json.load(handle)
            if cached.get("format") != FORMAT or cached.get("shard") != identity:
                raise ValueError(f"invalid index cache {cache_file}")
            return identity, cached["samples"]
    samples = {}
    with tarfile.open(shard, "r:") as archive:
        for info in archive:
            if not info.isfile():
                continue
            parsed = _member(info.name)
            if parsed is None:
                continue
            key, kind = parsed
            item = samples.setdefault(key, {})
            if kind in item:
                raise ValueError(f"duplicate {kind} member for {key} in {shard}")
            item[kind] = [info.offset_data, info.size]
    after = shard.stat()
    if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
        raise ValueError(f"tar changed during scan: {shard}")
    if cache_file is not None:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=cache_dir,
                                         prefix=".realssa-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump({"format": FORMAT, "shard": identity, "samples": samples},
                          handle, ensure_ascii=False, separators=(",", ":"))
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            try:
                os.link(temporary, cache_file)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    return identity, samples


def build_index(src_root, split, index_path, limit_shards=None, *, workers=1, cache_dir=None):
    """Build shard/member offsets with optional bounded, resumable scans.

    Incomplete samples are listed, not counted as complete. Cache identity uses
    source path, size, mtime; it does not replace a payload checksum.
    Duplicate sample IDs or members are errors. Output is never overwritten.
    """
    if split not in {"tr", "cv", "tt"}:
        raise ValueError("split must be tr, cv or tt")
    if limit_shards is not None and limit_shards < 1:
        raise ValueError("limit_shards must be positive")
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    output = Path(index_path)
    if output.exists():
        raise FileExistsError(output)
    directory = Path(src_root).expanduser().resolve() / split
    shards = sorted(directory.glob("*.tar"))
    if limit_shards is not None:
        shards = shards[:limit_shards]
    if not shards:
        raise ValueError(f"no uncompressed tar shards in {directory}")
    cache = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    results = [None] * len(shards)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one, shard, cache): i for i, shard in enumerate(shards)}
        for completed, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            results[i] = future.result()
            if completed % 10 == 0 or completed == len(shards):
                print(f"indexed shards {completed}/{len(shards)} ({split})", file=sys.stderr, flush=True)
    records, omitted, manifest, seen = [], [], [], set()
    for sid, (identity, samples) in enumerate(results):
        manifest.append(identity)
        for key, fields in sorted(samples.items()):
            if key in seen:
                raise ValueError(f"duplicate sample ID across shards: {key}")
            seen.add(key)
            missing = sorted(REQUIRED - fields.keys())
            if missing:
                omitted.append({"id": key, "shard": sid, "missing": missing})
            else:
                records.append({"id": key, "shard": sid, "fields": fields})
    index = {"format": FORMAT, "split": split, "sample_count": len(records),
             "incomplete_count": len(omitted), "incomplete": omitted,
             "shards": manifest, "samples": records}
    if not records:
        raise ValueError(f"no complete samples in {directory}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write completely and sync before publishing; linking refuses overwrite.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=output.parent,
                                     prefix=".realssa-index-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(index, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {key: index[key] for key in ("format", "split", "sample_count", "incomplete_count")} | {"shard_count": len(shards)}


class RealSSATarDataset(Dataset):
    """Aligned 2 s mono channel-zero waveform and 50 normalized visual frames.

    Random training crop is keyed by seed, epoch, and sample index. Call
    set_epoch before constructing the DataLoader iterator; worker processes must
    be restarted per epoch (persistent_workers=False). Center/start crops are
    deterministic and suited to evaluation. Sequences shorter than 2 s are
    zero-padded in audio and raw uint8 video before normalization. Set augment=True
    for the original training path: one random crop and optional flip per clip.
    """
    def __init__(self, index_path, *, seed=0, crop="random", augment=False):
        if crop not in {"random", "center", "start"}:
            raise ValueError("crop must be random, center or start")
        with open(index_path, encoding="utf8") as handle:
            index = json.load(handle)
        if index.get("format") != FORMAT or index.get("sample_count") != len(index.get("samples", [])):
            raise ValueError("invalid RealSSA tar index")
        self.shards, self.samples = index["shards"], index["samples"]
        self.split = index["split"]
        self.incomplete = index["incomplete"]
        self.seed, self.epoch, self.crop, self.augment = int(seed), 0, crop, bool(augment)
        for entry in self.shards:
            stat = Path(entry["path"]).stat()
            if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
                raise ValueError("indexed tar changed: " + entry["path"])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        sample_id = record["id"]
        shard = self.shards[record["shard"]]["path"]
        payload = {}
        # The descriptor is opened in the worker that consumes the sample;
        # sharing offsets between threads/processes is impossible.
        with open(shard, "rb") as stream:
            for kind in ("mix", "target", "mouth"):
                offset, size = record["fields"][kind]
                stream.seek(offset)
                payload[kind] = stream.read(size)
                if len(payload[kind]) != size:
                    raise ValueError(f"truncated {kind} for {sample_id} in {shard}")
        audio = []
        for kind in ("mix", "target"):
            wav, sr = sf.read(io.BytesIO(payload[kind]), dtype="float32", always_2d=True)
            if sr != SR or wav.shape[1] != 4 or not np.isfinite(wav).all():
                raise ValueError(f"invalid {kind} audio for {sample_id}: {sr} Hz, {wav.shape}")
            audio.append(wav[:, 0].copy())
        frames = torch.load(io.BytesIO(payload["mouth"]), map_location="cpu", weights_only=True)
        if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8 or frames.ndim not in (3, 4):
            raise ValueError(f"invalid mouth tensor for {sample_id}")
        available = min(len(audio[0]) // SAMPLES_PER_FRAME,
                        len(audio[1]) // SAMPLES_PER_FRAME, len(frames))
        if available < 1:
            raise ValueError(f"no aligned 40 ms frame for {sample_id}")
        last = max(0, available - SEGMENT_FRAMES)
        rng = random.Random((self.seed << 64) ^ (self.epoch << 32) ^ index)
        if self.crop == "start":
            start = 0
        elif self.crop == "center":
            start = last // 2
        else:
            start = rng.randint(0, last)
        count = min(SEGMENT_FRAMES, available - start)
        begin, end = start * SAMPLES_PER_FRAME, (start + count) * SAMPLES_PER_FRAME
        waveforms = []
        for wav in audio:
            out = torch.zeros(SEGMENT_SAMPLES, dtype=torch.float32)
            out[:end-begin] = torch.from_numpy(wav[begin:end])
            waveforms.append(out)
        view = frames[start:start+count]
        padding = torch.zeros((SEGMENT_FRAMES-count, *frames.shape[1:]), dtype=torch.uint8)
        raw = torch.cat((view, padding), dim=0)
        # Match the original training path: torchvision grayscale on uint8,
        # then one shared crop/flip across the entire temporal sequence.
        if raw.ndim == 4 and raw.shape[1:] == (96, 96, 3):
            gray = rgb_to_grayscale(raw.permute(0, 3, 1, 2)).squeeze(1)
        elif raw.ndim == 3 and raw.shape[1:] == (96, 96):
            gray = raw
        else:
            raise ValueError(f"invalid mouth dimensions for {sample_id}: {tuple(raw.shape)}")
        if self.augment:
            left, top = rng.randint(0, 8), rng.randint(0, 8)
            gray = gray[:, top:top+88, left:left+88]
            if rng.random() < .5:
                gray = gray.flip(-1)
        else:
            gray = gray[:, 4:92, 4:92]
        video = (gray.float() / 255.0 - .421) / .165
        return waveforms[0], waveforms[1], video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--split", required=True, choices=("tr", "cv", "tt"))
    parser.add_argument("--index", required=True)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-dir", help="optional resumable per-shard scan cache")
    args = parser.parse_args()
    print(json.dumps(build_index(args.src_root, args.split, args.index, args.limit_shards, workers=args.workers, cache_dir=args.cache_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
