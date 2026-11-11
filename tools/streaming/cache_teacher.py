"""Generate and load immutable fixed-crop teacher targets for RealSSA training."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tools.streaming.distillation import load_teacher, predict_teacher
from tools.streaming.evaluate_realssa import sha
from tools.streaming.realssa_tar import RealSSATarDataset, SEGMENT_SAMPLES

_ROOT = Path(__file__).resolve().parents[2]
_PLAN = "plan.json"
_MANIFEST = "manifest.json"


def _write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def teacher_source_sha256():
    """Hash the released teacher loading and waveform inference source files."""
    paths = [_ROOT / "configs/Falcon.yml"]
    for filename in ("distillation.py", "evaluate_realssa.py"):
        paths.append(Path(__file__).with_name(filename))
    paths.extend(_ROOT / "look2hear/models" / name for name in (
        "__init__.py", "Falcon.py", "Falcon_Contrastive.py", "base_model.py",
    ))
    for name in ("videomodels", "layers", "utils"):
        paths.extend(sorted((_ROOT / "look2hear" / name).rglob("*.py")))
    return {str(path.relative_to(_ROOT)): sha(path) for path in paths}


def _code_hashes():
    paths = (Path(__file__), Path(__file__).with_name("realssa_tar.py"))
    return {**teacher_source_sha256(),
            **{str(path.relative_to(_ROOT)): sha(path) for path in paths}}


def _range(count, world_size, rank):
    if not 0 <= rank < world_size <= count:
        raise ValueError("invalid teacher shard rank or world size")
    return count * rank // world_size, count * (rank + 1) // world_size


def prepare(cache_dir, index_path, teacher_checkpoint, *, seed=42, world_size=2, microbatch_size=16):
    cache_dir = Path(cache_dir).resolve()
    index_path = Path(index_path).resolve(strict=True)
    teacher_checkpoint = Path(teacher_checkpoint).resolve(strict=True)
    dataset = RealSSATarDataset(index_path, seed=seed, crop="random", augment=True)
    if dataset.split != "tr":
        raise ValueError("teacher cache requires a training index")
    _range(len(dataset), world_size, 0)
    if microbatch_size < 1:
        raise ValueError("teacher cache microbatch must be positive")
    plan = {
        "format": "realssa-teacher-cache-plan-v1",
        "index_path": str(index_path),
        "index_sha256": sha(index_path),
        "checkpoint_path": str(teacher_checkpoint),
        "checkpoint_sha256": sha(teacher_checkpoint),
        "count": len(dataset),
        "seed": seed,
        "crop": "random",
        "augment": True,
        "crop_epoch": 0,
        "world_size": world_size,
        "microbatch_size": microbatch_size,
        "waveform_samples": SEGMENT_SAMPLES,
        "dtype": "float32",
        "code_sha256": _code_hashes(),
    }
    cache_dir.mkdir()
    _write_json(cache_dir / _PLAN, plan)
    return plan


def _check_plan(cache_dir, *, index_path=None, teacher_checkpoint=None):
    plan = _read_json(cache_dir / _PLAN)
    if (plan["format"] != "realssa-teacher-cache-plan-v1" or plan["crop_epoch"] != 0
            or plan["crop"] != "random" or plan["augment"] is not True
            or plan["dtype"] != "float32" or plan["waveform_samples"] != SEGMENT_SAMPLES):
        raise ValueError("teacher cache plan has an unsupported format or crop protocol")
    if sha(index_path or plan["index_path"]) != plan["index_sha256"]:
        raise ValueError("teacher cache index SHA256 mismatch")
    if sha(teacher_checkpoint or plan["checkpoint_path"]) != plan["checkpoint_sha256"]:
        raise ValueError("teacher cache checkpoint SHA256 mismatch")
    if _code_hashes() != plan["code_sha256"]:
        raise ValueError("teacher cache source SHA256 mismatch")
    return plan


def _shard_name(rank):
    return f"shard-{rank:04d}"


def generate_shard(cache_dir, rank, *, device, workers=4):
    cache_dir = Path(cache_dir).resolve(strict=True)
    plan = _check_plan(cache_dir)
    start, stop = _range(plan["count"], plan["world_size"], rank)
    destination = cache_dir / _shard_name(rank)
    if destination.exists() or (cache_dir / _MANIFEST).exists():
        raise FileExistsError(destination)
    dataset = RealSSATarDataset(plan["index_path"], seed=plan["seed"], crop="random", augment=True)
    if len(dataset) != plan["count"] or dataset.split != "tr":
        raise ValueError("teacher cache dataset length or split changed")
    device = torch.device(device)
    loader = DataLoader(
        Subset(dataset, range(start, stop)),
        batch_size=plan["microbatch_size"],
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    teacher = load_teacher(plan["checkpoint_path"], device)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=cache_dir))
    try:
        path = temporary / "teacher.npy"
        waveforms = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(stop - start, SEGMENT_SAMPLES),
        )
        cursor = start
        for mixture, _, video in loader:
            prediction = predict_teacher(
                teacher, mixture.to(device), video.to(device),
                microbatch_size=plan["microbatch_size"],
            )
            waveforms[cursor - start:cursor + len(prediction) - start] = prediction.cpu().numpy()
            cursor += len(prediction)
            if (cursor - start) % (plan["microbatch_size"] * 128) == 0:
                print(f"rank={rank} examples={cursor - start}/{stop - start}", flush=True)
        if cursor != stop:
            raise RuntimeError("teacher shard generation stopped before covering its range")
        waveforms.flush()
        del waveforms
        if _check_plan(cache_dir) != plan:
            raise RuntimeError("teacher cache plan changed during generation")
        stat = path.stat()
        metadata = {
            "rank": rank, "start": start, "stop": stop,
            "plan_sha256": sha(cache_dir / _PLAN),
            "file_size": stat.st_size, "file_mtime_ns": stat.st_mtime_ns,
            "file_sha256": sha(path),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        _write_json(temporary / "metadata.json", metadata)
        temporary.rename(destination)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _checked_shards(cache_dir, plan, *, full_hashes):
    shards = []
    for rank in range(plan["world_size"]):
        folder = cache_dir / _shard_name(rank)
        meta = _read_json(folder / "metadata.json")
        expected = _range(plan["count"], plan["world_size"], rank)
        if ((meta["start"], meta["stop"]) != expected or meta["rank"] != rank
                or meta["plan_sha256"] != sha(cache_dir / _PLAN)):
            raise ValueError("teacher cache shard range, rank or plan identity mismatch")
        path = folder / "teacher.npy"
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (meta["file_size"], meta["file_mtime_ns"]):
            raise ValueError("teacher cache shard file changed")
        if full_hashes and sha(path) != meta["file_sha256"]:
            raise ValueError("teacher cache shard SHA256 mismatch")
        waveforms = np.load(path, mmap_mode="r", allow_pickle=False)
        if waveforms.shape != (expected[1] - expected[0], SEGMENT_SAMPLES) or waveforms.dtype != np.float32:
            raise ValueError("teacher cache shard waveform shape or dtype mismatch")
        del waveforms
        shards.append(meta)
    return shards


def finalize(cache_dir):
    cache_dir = Path(cache_dir).resolve(strict=True)
    destination = cache_dir / _MANIFEST
    if destination.exists():
        raise FileExistsError(destination)
    plan = _check_plan(cache_dir)
    result = {
        "format": "realssa-teacher-cache-v1",
        "plan_sha256": sha(cache_dir / _PLAN),
        "shards": _checked_shards(cache_dir, plan, full_hashes=True),
    }
    with tempfile.TemporaryDirectory(prefix=".manifest-", dir=cache_dir) as temporary:
        candidate = Path(temporary) / _MANIFEST
        _write_json(candidate, result)
        os.link(candidate, destination)
    return result


def _read_final(cache_dir, *, index_path=None, teacher_checkpoint=None, full_hashes=False):
    plan = _check_plan(cache_dir, index_path=index_path, teacher_checkpoint=teacher_checkpoint)
    manifest = _read_json(cache_dir / _MANIFEST)
    if (manifest["format"] != "realssa-teacher-cache-v1"
            or manifest["plan_sha256"] != sha(cache_dir / _PLAN)):
        raise ValueError("teacher cache manifest plan identity mismatch")
    shards = _checked_shards(cache_dir, plan, full_hashes=full_hashes)
    if shards != manifest["shards"]:
        raise ValueError("teacher cache finalized shard metadata mismatch")
    return plan, shards


class FixedCropDataset(Dataset):
    """Use the epoch-zero crop/augmentation while preserving sampler reshuffling."""

    def __init__(self, dataset):
        if dataset.crop != "random" or not dataset.augment:
            raise ValueError("fixed crop expects RealSSA random crop and augmentation")
        self.dataset = dataset
        self.dataset.set_epoch(0)
        self.samples = dataset.samples
        self.shards = dataset.shards
        self.split = dataset.split
        self.incomplete = dataset.incomplete

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def set_epoch(self, epoch):
        self.dataset.set_epoch(0)


class CachedTeacherDataset(FixedCropDataset):
    def __init__(self, dataset, cache_dir, *, index_path, teacher_checkpoint):
        super().__init__(dataset)
        self.cache_dir = Path(cache_dir).resolve(strict=True)
        plan, self.teacher_shards = _read_final(
            self.cache_dir, index_path=index_path, teacher_checkpoint=teacher_checkpoint,
        )
        if len(dataset) != plan["count"] or dataset.seed != plan["seed"] or dataset.split != "tr":
            raise ValueError("teacher cache differs from training dataset seed, split or length")
        self.teacher_cache_sha256 = sha(self.cache_dir / _MANIFEST)
        self._stops = [meta["stop"] for meta in self.teacher_shards]
        self._mmap = {}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rank = bisect.bisect_right(self._stops, index)
        if rank not in self._mmap:
            path = self.cache_dir / _shard_name(rank) / "teacher.npy"
            self._mmap[rank] = np.load(path, mmap_mode="r", allow_pickle=False)
        offset = index - self.teacher_shards[rank]["start"]
        teacher = torch.from_numpy(self._mmap[rank][offset].copy())
        return (*self.dataset[index], teacher)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--index", required=True)
    prepare_parser.add_argument("--teacher-checkpoint", required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--world-size", type=int, default=2)
    prepare_parser.add_argument("--microbatch-size", type=int, default=16)
    shard_parser = commands.add_parser("shard")
    shard_parser.add_argument("--rank", type=int, required=True)
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--workers", type=int, default=4)
    for sub in (prepare_parser, shard_parser, commands.add_parser("finalize"), commands.add_parser("verify")):
        sub.add_argument("--cache-dir", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "prepare":
        result = prepare(args.cache_dir, args.index, args.teacher_checkpoint,
                         seed=args.seed, world_size=args.world_size, microbatch_size=args.microbatch_size)
    elif args.command == "shard":
        result = generate_shard(args.cache_dir, args.rank, device=args.device, workers=args.workers)
    elif args.command == "finalize":
        result = finalize(args.cache_dir)
    else:
        plan, shards = _read_final(Path(args.cache_dir), full_hashes=True)
        result = {"verified_count": plan["count"], "verified_shards": len(shards)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
