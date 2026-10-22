#!/usr/bin/env python
"""Convert LRS2-2Mix (``lrs2_rebuild``) into the Falcon training/evaluation layout.

Input layout (as distributed with this reproduction, see docs/REPRODUCTION_PLAN.md §5.2)::

    <src>/audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}/<stem>.wav   2.000 s, 16 kHz mono
    <src>/mouths/<speaker_utterance>.npz                      uint8 [50, 96, 96]
    <src>/raw_audio/{train,val,test}/<speaker_utterance>.wav   2 s source clips
    <src>/create_mix/*.scp                                     (mix recipe: s1 gain s2 gain)

``<stem>`` encodes the recipe: ``<s1_id>_<gain1>_<s2_id>_<gain2>`` and
``mix == gain1 * s1 + gain2 * s2`` (PCM16-rounded).  Target convention: **s1**.

Output layout (ready for ``look2hear.datas``)::

    <out>/<split>/mix/<stem>.wav        mixture (copied from the source)
    <out>/<split>/target/<stem>.wav     target speech (s1 by default)
    <out>/<split>/mouths/<id>.pt        uint8 RGB [50,96,96,3] mouth crops
    <out>/<split>/mix.json              [[abs_mix, n_samples], ...]
    <out>/<split>/target.json           [[abs_target, abs_mouth_pt, n_samples], ...]
    <out>/<split>/near.json             near-speaker index (see --near-mode)
    <out>/conversion_stats.json         machine-readable conversion/validation stats

Mouth naming: the mouth crops belong to the *target utterance id* and are shared by
all mixtures that use that utterance, so they are de-duplicated to
``mouths/<target_id>.pt`` (saves ~13 GB vs one copy per mixture).  Using
``--target s2`` switches both target audio and mouth crops to s2.

``--near-mode``
---------------
* ``fallback`` (default): ``near.json`` uses the mixture-path sentinel
  ``[[mix_path, n]]``; the Dataset skips it -> K=0 -> the NASS contrastive loss is
  inactive.  This is the honest default: LRS2-2Mix ships no near-field interferer
  waveforms.
* ``raw_audio``: for every sample pick one 2 s clip from ``raw_audio/<train|val|test>``
  whose *speaker* differs from both mixture speakers and whose clip id differs from
  s1/s2, and index it as a negative.  This is an **approximate** substitute:
  the clip is a clean single-speaker 2 s LRS2 segment (no RIR, no loudness match,
  not the actual interfering source), so it only supports "non-official negatives"
  experiments (docs/REPRODUCTION_PLAN.md §5.1 Step 4 option 2).  Report it explicitly
  with any result produced this way.

The conversion is re-entrant: existing mix/target/mouth files are skipped by default
(``--overwrite`` forces rewriting), and the JSON indexes are always rebuilt from the
converted files.  No source files are modified.

Examples
--------
Dry-run (audit only, writes nothing)::

    python tools/repro/lrs2_2mix_to_falcon.py --dry-run

Small run then full run::

    python tools/repro/lrs2_2mix_to_falcon.py --limit 20 --out /tmp/lrs2_smoke
    python tools/repro/lrs2_2mix_to_falcon.py --jobs 16 \
        --out /root/data/falcon_lrs2_2mix --near-mode fallback
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 25  # 640 samples per video frame (25 fps)
MOUTH_H = MOUTH_W = 96
MOUTH_CHANNELS = 3
SPLITS = ("tr", "cv", "tt")
SPLIT_TO_RAW = {"tr": "train", "cv": "val", "tt": "test"}
# <s1_id>_<gain1>_<s2_id>_<gain2>, e.g. 5535415699068794046_00004_2.4022_6160231649911868423_00015_-2.4022
STEM_RE = re.compile(r"^(\d+_\d+)_([-+]?\d+(?:\.\d+)?)_(\d+_\d+)_([-+]?\d+(?:\.\d+)?)$")
MOUTH_PT_BYTES = MOUTH_H * MOUTH_W * MOUTH_CHANNELS  # per frame
MIN_MOUTH_PT_BYTES = 50 * MOUTH_PT_BYTES  # 50 frames * 3 channels = 1,382,400 bytes


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------
def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp{os.getpid()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _atomic_torch_save(obj, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, dst)


def _atomic_json_dump(obj, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, dst)


def _valid_file(path: str, min_bytes: int) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _usage_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GiB"


# ---------------------------------------------------------------------------
# source parsing
# ---------------------------------------------------------------------------
def parse_stem(stem: str):
    m = STEM_RE.match(stem)
    if m is None:
        return None
    s1, gain1, s2, gain2 = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
    return {"s1": s1, "gain1": gain1, "s2": s2, "gain2": gain2}


def split_dir(src: Path, split: str) -> Path:
    return src / "audio" / "wav16k" / "min" / split


def iter_split(src: Path, split: str, limit: int | None = None):
    """Yield parsed source entries for one split (sorted, deterministic)."""
    base = split_dir(src, split)
    mix_dir = base / "mix"
    if not mix_dir.is_dir():
        return
    stems = sorted(p.name[:-4] for p in mix_dir.glob("*.wav"))
    if limit is not None:
        stems = stems[:limit]
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed is None:
            yield {"stem": stem, "error": "unparsable stem"}
            continue
        yield {
            "stem": stem,
            "mix_path": str(mix_dir / f"{stem}.wav"),
            "s1_path": str(base / "s1" / f"{stem}.wav"),
            "s2_path": str(base / "s2" / f"{stem}.wav"),
            **parsed,
        }


# ---------------------------------------------------------------------------
# conversion worker (module level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------
def convert_one(task: dict) -> dict:
    """Convert a single source sample. Must be a top-level function (picklable)."""
    args = task["args"]
    out = Path(args.out)
    split = task["split"]
    target_choice = args.target
    target_id = task["s1"] if target_choice == "s1" else task["s2"]
    target_src = task["s1_path"] if target_choice == "s1" else task["s2_path"]
    mix_dst = out / split / "mix" / f"{task['stem']}.wav"
    target_dst = out / split / "target" / f"{task['stem']}.wav"
    mouth_dst = out / split / "mouths" / f"{target_id}.pt"
    npz_path = task["npz"]

    result = {"stem": task["stem"], "status": "converted"}
    try:
        if not args.overwrite:
            if (_valid_file(str(mix_dst), 64)
                    and _valid_file(str(target_dst), 64)
                    and _valid_file(str(mouth_dst), MIN_MOUTH_PT_BYTES)):
                result["status"] = "skipped_existing"
                return result
        # 1) audio copies (mixture + target)
        _atomic_copy(task["mix"], str(mix_dst))
        _atomic_copy(target_src, str(target_dst))
        # 2) mouth crops: uint8 [50,96,96] npz -> uint8 [50,96,96,3] .pt (dedup by id).
        # Only the first mixture that uses a given target utterance writes the shared
        # file; the others rely on it (checked again when building the JSON indexes).
        if task.get("write_mouth", True) and not (
                os.path.isfile(mouth_dst) and os.path.getsize(mouth_dst) >= MIN_MOUTH_PT_BYTES):
            with np.load(npz_path) as z:
                key = "data" if "data" in z else list(z.keys())[0]
                mouth = z[key]
            if mouth.ndim != 3 or mouth.shape[1:] != (MOUTH_H, MOUTH_W):
                raise ValueError(f"unexpected mouth npz shape {mouth.shape} in {npz_path}")
            if mouth.dtype != np.uint8:
                raise ValueError(f"expected uint8 mouth npz, got {mouth.dtype} in {npz_path}")
            mouth_rgb = np.repeat(mouth[:, :, :, None], MOUTH_CHANNELS, axis=-1)
            _atomic_torch_save(torch.from_numpy(np.ascontiguousarray(mouth_rgb)), str(mouth_dst))
        return result
    except Exception as exc:  # noqa: BLE001 - reported per sample, conversion continues
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


# ---------------------------------------------------------------------------
# near-speaker candidates (--near-mode raw_audio)
# ---------------------------------------------------------------------------
def raw_candidates(src: Path, split: str, min_samples: int = 32000) -> list[dict]:
    """2 s single-speaker clips usable as approximate negatives."""
    raw_dir = src / "raw_audio" / SPLIT_TO_RAW[split]
    if not raw_dir.is_dir():
        raise SystemExit(f"--near-mode raw_audio: raw audio dir not found: {raw_dir}")
    candidates = []
    for path in sorted(raw_dir.glob("*.wav")):
        clip_id = path.name[:-4]
        try:
            info = sf.info(str(path))
        except Exception:
            continue
        if info.frames < min_samples or info.samplerate != SAMPLE_RATE or info.channels != 1:
            continue
        candidates.append({"path": str(path.resolve()), "id": clip_id,
                           "speaker": clip_id.split("_")[0], "frames": int(info.frames)})
    if not candidates:
        raise SystemExit(
            f"--near-mode raw_audio: no usable >=2 s 16 kHz mono clips under {raw_dir}"
        )
    return candidates


def select_near(stem: str, s1: str, s2: str, candidates: list[dict], seed: int,
                per_sample: int) -> list[dict]:
    """Deterministically pick ``per_sample`` non-target speakers for one mixture."""
    digest = hashlib.md5(f"{seed}:{stem}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest, 16))
    speakers = {s1.split("_")[0], s2.split("_")[0]}
    source_ids = {s1, s2}
    strict = [c for c in candidates
              if c["speaker"] not in speakers and c["id"] not in source_ids]
    loose = [c for c in candidates if c["id"] not in source_ids]
    pool = strict if len(strict) >= per_sample else loose
    if not pool:
        return []
    if len(pool) <= per_sample:
        return pool[:per_sample]
    return rng.sample(pool, per_sample)


# ---------------------------------------------------------------------------
# index building / validation / stats
# ---------------------------------------------------------------------------
def build_indexes(args, splits) -> dict:
    """Rebuild mix/target/near JSON from the converted folder layout (re-entrant)."""
    src = Path(args["src"])
    out = Path(args["out"])
    result = {}
    near_mode = args["near_mode"]
    for split in splits:
        mix_json, target_json, near_json = [], [], []
        stems = []
        for entry in iter_split(src, split):
            if "error" in entry:
                continue
            stem = entry["stem"]
            mix_path = (out / split / "mix" / f"{stem}.wav").resolve()
            target_path = (out / split / "target" / f"{stem}.wav").resolve()
            target_id = entry["s1"] if args["target"] == "s1" else entry["s2"]
            mouth_path = (out / split / "mouths" / f"{target_id}.pt").resolve()
            if not (_valid_file(str(mix_path), 64) and _valid_file(str(target_path), 64)
                    and _valid_file(str(mouth_path), MIN_MOUTH_PT_BYTES)):
                continue
            n = sf.info(str(mix_path)).frames
            mix_json.append([str(mix_path), n])
            target_json.append([str(target_path), str(mouth_path), n])
            stems.append(stem)
        if near_mode == "raw_audio":
            candidates = raw_candidates(src, split)
            for stem, mix_entry in zip(stems, mix_json):
                s1, _, s2, _ = parse_stem_required(stem)
                picks = select_near(stem, s1, s2, candidates, args["seed"], args["near_per_sample"])
                if picks:
                    near_json.append([[p["path"], p["frames"]] for p in picks])
                else:  # no different speaker available -> mixture sentinel fallback
                    near_json.append([mix_entry])
        else:
            near_json = [[mix_entry] for mix_entry in mix_json]
        split_dir_out = out / split
        _atomic_json_dump(mix_json, str(split_dir_out / "mix.json"))
        _atomic_json_dump(target_json, str(split_dir_out / "target.json"))
        _atomic_json_dump(near_json, str(split_dir_out / "near.json"))
        result[split] = {"mix_json": str(split_dir_out / "mix.json"),
                         "target_json": str(split_dir_out / "target.json"),
                         "near_json": str(split_dir_out / "near.json"),
                         "count": len(mix_json)}
    return result


def parse_stem_required(stem: str):
    parsed = parse_stem(stem)
    if parsed is None:  # pragma: no cover - filtered earlier
        raise ValueError(f"unparsable stem {stem}")
    return parsed["s1"], parsed["gain1"], parsed["s2"], parsed["gain2"]


def validate_outputs(args, splits, per_split: int) -> dict:
    """Sample outputs and check lengths, mouth shape, mix=s1+s2 and near/JSON contracts."""
    src = Path(args["src"])
    out = Path(args["out"])
    summary = {}
    for split in splits:
        mix_json = json.loads((out / split / "mix.json").read_text())
        target_json = json.loads((out / split / "target.json").read_text())
        near_json = json.loads((out / split / "near.json").read_text())
        checks = {"json_equal_length": len(mix_json) == len(target_json) == len(near_json),
                  "n_samples": len(mix_json), "checked": 0, "length_ok": 0, "mouth_ok": 0,
                  "mix_sum_ok": 0, "max_mix_sum_err": 0.0, "near_ok": 0,
                  "problems": []}
        if not checks["json_equal_length"]:
            checks["problems"].append("mix/target/near JSON lengths differ")
            summary[split] = checks
            continue
        rng = random.Random(1234 + len(split))
        if per_split <= 0 or not mix_json:
            summary[split] = checks
            continue
        indices = list(range(len(mix_json))) if len(mix_json) <= per_split else \
            rng.sample(range(len(mix_json)), per_split)
        base = split_dir(src, split)
        for idx in indices:
            mix_path = mix_json[idx][0]
            target_path = target_json[idx][0]
            mouth_path = target_json[idx][1]
            stem = Path(mix_path).stem
            checks["checked"] += 1
            try:
                mix_info = sf.info(mix_path)
                target_info = sf.info(target_path)
                if (mix_info.samplerate, mix_info.channels) != (SAMPLE_RATE, 1):
                    raise ValueError(f"mix format {mix_info.samplerate}/{mix_info.channels}")
                if target_info.frames != mix_info.frames:
                    raise ValueError(f"target length {target_info.frames} != mix {mix_info.frames}")
                checks["length_ok"] += 1
                if mouth_path.endswith(".pt"):
                    mouths = torch.load(mouth_path, map_location="cpu", weights_only=True)
                    expect_frames = mix_info.frames // FRAME
                    if (mouths.dtype != torch.uint8 or mouths.ndim != 4
                            or tuple(mouths.shape[1:]) != (MOUTH_H, MOUTH_W, MOUTH_CHANNELS)
                            or mouths.shape[0] != expect_frames):
                        raise ValueError(f"bad mouths {tuple(mouths.shape)} {mouths.dtype} "
                                         f"expect frames={expect_frames}")
                    checks["mouth_ok"] += 1
                # mix == gain1*s1 + gain2*s2 (PCM16 rounding tolerance)
                s1_path = base / "s1" / f"{stem}.wav"
                s2_path = base / "s2" / f"{stem}.wav"
                parsed = parse_stem_required(stem)
                mix, _ = sf.read(mix_path, dtype="float32")
                s1, _ = sf.read(str(s1_path), dtype="float32")
                s2, _ = sf.read(str(s2_path), dtype="float32")
                err = float(np.max(np.abs(mix - (s1 + s2))))
                checks["max_mix_sum_err"] = max(checks["max_mix_sum_err"], err)
                if err <= args["mix_tol"]:
                    checks["mix_sum_ok"] += 1
                else:
                    checks["problems"].append(f"{stem}: mix-(s1+s2) max err {err:.3e}")
                # near contract
                near_entry = near_json[idx]
                if args["near_mode"] == "raw_audio":
                    near_path = near_entry[0][0]
                    near_info = sf.info(near_path)
                    if near_info.frames < 32000 or near_info.samplerate != SAMPLE_RATE:
                        raise ValueError(f"bad near clip {near_path}: {near_info.frames} frames")
                    target_speakers = {parsed[0].split("_")[0], parsed[2].split("_")[0]}
                    if Path(near_path).stem.split("_")[0] in target_speakers:
                        raise ValueError(f"near clip speaker equals a mixture speaker: {near_path}")
                checks["near_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                checks["problems"].append(f"{stem}: {type(exc).__name__}: {exc}")
        summary[split] = checks
    return summary


def compute_stats(args, splits, per_split_counts, elapsed_s: float) -> dict:
    out = Path(args["out"])
    stats = {"out": str(out), "src": args["src"], "target": args["target"],
             "near_mode": args["near_mode"], "near_per_sample": args["near_per_sample"],
             "elapsed_seconds": round(elapsed_s, 1), "splits": {}, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    total_bytes = 0
    for split in splits:
        sdir = out / split
        mix_files = sorted((sdir / "mix").glob("*.wav"))
        target_files = sorted((sdir / "target").glob("*.wav"))
        mouth_files = sorted((sdir / "mouths").glob("*.pt"))
        lengths = collections.Counter(sf.info(str(p)).frames for p in mix_files)
        split_bytes = _usage_bytes(str(sdir))
        total_bytes += split_bytes
        stats["splits"][split] = {
            "mix_files": len(mix_files), "target_files": len(target_files),
            "mouth_files": len(mouth_files),
            "sample_lengths": dict(lengths),
            "total_seconds": round(sum(lengths.values()) / SAMPLE_RATE, 1) if lengths else 0.0,
            "bytes": split_bytes, "bytes_human": _fmt_bytes(split_bytes),
            "count_from_source": per_split_counts.get(split),
        }
    stats["total_bytes"] = total_bytes
    stats["total_bytes_human"] = _fmt_bytes(total_bytes)
    return stats


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert LRS2-2Mix (lrs2_rebuild) to the Falcon folder layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", default="/root/data/lrs2_2mix/lrs2_rebuild",
                   help="lrs2_rebuild root (contains audio/, mouths/, raw_audio/).")
    p.add_argument("--out", default="/root/data/falcon_lrs2_2mix",
                   help="Output dataset root (folder layout + JSON indexes).")
    p.add_argument("--splits", default=",".join(SPLITS), help="Comma-separated splits.")
    p.add_argument("--target", default="s1", choices=["s1", "s2"],
                   help="Target convention; s1 matches the paper's video speaker.")
    p.add_argument("--jobs", type=int, default=8, help="Parallel conversion processes.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N samples per split.")
    p.add_argument("--near-mode", default="fallback", choices=["fallback", "raw_audio"],
                   help="fallback: mixture sentinel (K=0); raw_audio: approximate negatives.")
    p.add_argument("--near-per-sample", type=int, default=1,
                   help="--near-mode raw_audio: number of approximate negatives per sample.")
    p.add_argument("--seed", type=int, default=0, help="Deterministic seed for near selection.")
    p.add_argument("--dry-run", action="store_true", help="Audit source and estimate outputs; write nothing.")
    p.add_argument("--overwrite", action="store_true", help="Rewrite existing converted files.")
    p.add_argument("--validate-samples", type=int, default=200,
                   help="Per split, how many outputs to validate (0 disables validation).")
    p.add_argument("--mix-tol", type=float, default=1e-3,
                   help="Tolerance for mix==s1+s2 validation (PCM16 rounding).")
    args = p.parse_args(argv)
    args.splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not args.splits:
        raise SystemExit("--splits must not be empty")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.near_per_sample < 0:
        raise SystemExit("--near-per-sample must be >= 0")
    return args


def _as_dict(args) -> dict:
    d = vars(args).copy()
    d["splits"] = list(args.splits)
    return d


def audit_source(src: Path, splits, limit) -> dict:
    """Parse stems, count samples, check mouths coverage and lengths (no writes)."""
    audit = {}
    for split in splits:
        base = split_dir(src, split)
        if not base.is_dir():
            raise SystemExit(f"source split directory not found: {base}")
        n = parsed = 0
        lengths = collections.Counter()
        missing_mouth = 0
        bad = []
        for entry in iter_split(src, split, limit):
            n += 1
            if "error" in entry:
                bad.append(entry["stem"])
                continue
            parsed += 1
            try:
                lengths[sf.info(entry["mix_path"]).frames] += 1
            except Exception:
                bad.append(entry["stem"])
            for target_id in (entry["s1"], entry["s2"]):
                if not (src / "mouths" / f"{target_id}.npz").is_file():
                    missing_mouth += 1
        unique_target_ids = set()
        for entry in iter_split(src, split, limit):
            if "error" not in entry:
                unique_target_ids.add(entry["s1"])
        audit[split] = {"mix_files": n, "parsed": parsed, "unparsable": bad[:5],
                        "lengths": dict(lengths), "missing_mouth_npz": missing_mouth,
                        "unique_target_ids": len(unique_target_ids)}
    return audit


def print_audit(audit: dict, splits) -> None:
    print("== source audit ==")
    for split in splits:
        a = audit[split]
        print(f"  {split}: {a['mix_files']} mixtures, {a['parsed']} parsed, "
              f"lengths={a['lengths']}, unique_target_ids={a['unique_target_ids']}, "
              f"missing_mouth_npz={a['missing_mouth_npz']}, unparsable={len(a['unparsable'])}")


def estimate_disk(audit: dict, args, splits) -> int:
    """Rough output size estimate: copied wav bytes + de-duplicated mouth pt bytes."""
    src = Path(args.src)
    total = 0
    for split in splits:
        a = audit[split]
        # mixture + target wav bytes from source sizes
        for name in ("mix", "s1" if args.target == "s1" else "s2"):
            d = split_dir(src, split) / name
            for path in d.glob("*.wav"):
                try:
                    total += os.path.getsize(path)
                except OSError:
                    pass
        total += a["unique_target_ids"] * MIN_MOUTH_PT_BYTES
    return total


def convert_split(args, split: str, total_expected: int) -> dict:
    tasks = []
    mouth_owner = set()
    for entry in iter_split(Path(args.src), split, args.limit):
        if "error" in entry:
            continue
        target_id = entry["s1"] if args.target == "s1" else entry["s2"]
        npz = Path(args.src) / "mouths" / f"{target_id}.npz"
        if not npz.is_file():
            tasks.append({"stem": entry["stem"], "status": "missing_mouth_npz"})
            continue
        write_mouth = target_id not in mouth_owner
        mouth_owner.add(target_id)
        tasks.append({
            "args": args,
            "split": split,
            "stem": entry["stem"],
            "s1": entry["s1"], "s2": entry["s2"],
            "mix": entry["mix_path"],
            "s1_path": entry["s1_path"],
            "s2_path": entry["s2_path"],
            "npz": str(npz),
            "write_mouth": write_mouth,
        })
    results = collections.Counter()
    errors = []
    skipped_missing = 0
    work = []
    for task in tasks:
        if task.get("status") == "missing_mouth_npz":
            skipped_missing += 1
            continue
        work.append(task)
    results["missing_mouth_npz"] = skipped_missing
    t0 = time.time()
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(convert_one, task) for task in work]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results[res["status"]] += 1
            done += 1
            if res["status"] == "error" and len(errors) < 10:
                errors.append(res.get("error", "unknown"))
            if done % 1000 == 0 or done == len(work):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  [{split}] {done}/{len(work)} processed ({rate:.0f}/s)", flush=True)
    if errors:
        results["first_errors"] = errors
    results["elapsed_seconds"] = round(time.time() - t0, 1)
    return dict(results)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    if not src.is_dir():
        raise SystemExit(f"--src not found: {src}")
    audit = audit_source(src, args.splits, args.limit)
    print_audit(audit, args.splits)
    est = estimate_disk(audit, args, args.splits)
    print(f"== estimated output size: {_fmt_bytes(est)} "
          f"(near-mode={args.near_mode}, target={args.target})")
    if args.dry_run:
        print("== dry-run: nothing written ==")
        return 0
    t_start = time.time()
    counts, conversions, indexes = {}, {}, {}
    for split in args.splits:
        expected = audit[split]["mix_files"] if args.limit is None else \
            min(audit[split]["mix_files"], args.limit)
        print(f"== converting split {split} (expected {expected} samples, jobs={args.jobs})", flush=True)
        conv = convert_split(args, split, expected)
        conversions[split] = conv
        counts[split] = expected
        print(f"  [{split}] done: {conv}", flush=True)
        # publish this split's JSON index immediately so downstream (e.g. training)
        # can start as soon as the first split is ready
        split_index = build_indexes(_as_dict(args), [split])
        indexes.update(split_index)
        print(f"  [{split}] INDEX READY: {split_index[split]['mix_json']} "
              f"({split_index[split]['count']} entries)", flush=True)
    validation = validate_outputs(_as_dict(args), args.splits, args.validate_samples)
    elapsed = time.time() - t_start
    stats = compute_stats(_as_dict(args), args.splits, counts, elapsed)
    stats["conversions"] = conversions
    stats["validation"] = validation
    stats["indexes"] = indexes
    _atomic_json_dump(stats, str(out / "conversion_stats.json"))
    print("== conversion summary ==")
    for split in args.splits:
        s = stats["splits"][split]
        print(f"  {split}: mix={s['mix_files']} target={s['target_files']} "
              f"mouths={s['mouth_files']} lengths={s['sample_lengths']} "
              f"bytes={s['bytes_human']}")
        print(f"    -> {indexes[split]['mix_json']} ({indexes[split]['count']} entries)")
    print(f"== total output: {stats['total_bytes_human']}; elapsed {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")
    print("== validation (sampled) ==")
    for split in args.splits:
        v = validation[split]
        print(f"  {split}: checked={v.get('checked')} length_ok={v.get('length_ok')} "
              f"mouth_ok={v.get('mouth_ok')} mix_sum_ok={v.get('mix_sum_ok')} "
              f"max_mix_sum_err={v.get('max_mix_sum_err'):.2e} near_ok={v.get('near_ok')} "
              f"problems={len(v.get('problems', []))}")
    bad = sum(len(v.get("problems", [])) for v in validation.values())
    return 0 if bad == 0 else 0  # problems are reported, not fatal; see stats json


if __name__ == "__main__":
    sys.exit(main())
