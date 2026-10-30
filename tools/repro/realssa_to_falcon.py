#!/usr/bin/env python
"""Convert RealSSA-Sim WebDataset TAR shards into the Falcon folder layout.

RealSSA-Sim (Hugging Face ``JusperLee/RealSSA-Sim``, gated) ships 240 WebDataset
TAR shards.  Every example ``<id>`` is four consecutive members::

    <id>.mix.wav      4-channel PCM16 mixture, 16 kHz
    <id>.target.wav   4-channel PCM16 target speech, 16 kHz
    <id>.json         scene / room / components(type, source, SIR, position) / ...
    <id>.mouths.pt    uint8 RGB mouth crops, torch tensor [T, 96, 96, 3]

This script converts a split into the folder layout expected by
``look2hear.datas.avspeech_dataset*`` and then builds the JSON indexes with the
repository's own ``DataPreProcess`` code::

    <out>/<split>/mix/<id>.wav              mono 16 kHz PCM16 (channel policy below)
    <out>/<split>/target/<id>.wav           mono 16 kHz PCM16, same channel
    <out>/<split>/mouths/<id>.pt            uint8 RGB [T,96,96,3] (unchanged)
    <out>/<split>/mix/<id>_meta.json        compact RealSSA meta with components[].path
    <out>/<split>/mix.json target.json near.json

Channel policy (``--channel``)
-----------------------------
RealSSA-Sim is 4-channel and the paper's protocol is not documented; the dataset
card suggests reporting the channel and using the *same* channel for mixture and
target.  Choices: ``mean`` (average the 4 channels), or ``0``/``1``/``2``/``3``.

Frame alignment (default on)
----------------------------
Following the dataset card, ``frames = min(mouth_frames, samples_mix//640,
samples_target//640)`` and audio/target/mouths are trimmed to ``frames`` so that
mouth frame ``k`` covers samples ``[640k, 640(k+1))`` — the same 640 samples/frame
assumption as the training loader.  ``--no-align`` keeps raw lengths (not
recommended: the loader may then slice past the mouth array).

Near-field negatives (important)
--------------------------------
RealSSA-Sim ships mixtures and targets only; the ``components[].path`` values are
generation-time provenance paths and are **not** downloadable source waveforms.
Without near waveforms, ``near.json`` falls back to mixture sentinels, the dataset
skips them and the NASS contrastive loss stays inactive (K=0).  This is a known
limitation of the released data (see docs/REPRODUCTION_PLAN.md §5.1 Step 4).
If you have a compatible directory of near wavs, pass ``--near-dir`` with files
named ``<id>.near0.wav``, ``<id>.near1.wav``... and they will be wired into the
meta / near.json.

Examples
--------
Convert one test shard, channel 0, building the JSON indexes::

    python tools/repro/realssa_to_falcon.py \
        --src RealSSA-Sim/tt/tt-000000.tar \
        --out /data/realssa_falcon --splits tt --channel 0

Convert a downloaded split tree (``RealSSA-Sim/{tr,cv,tt}/*.tar``)::

    python tools/repro/realssa_to_falcon.py \
        --src RealSSA-Sim --out /data/realssa_falcon --channel mean

Dry-run (inspect pairing / counts without writing anything)::

    python tools/repro/realssa_to_falcon.py --src RealSSA-Sim --out /tmp/x --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import stat
import uuid
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# make sibling helper importable when the script is executed directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic_data import build_indexes  # noqa: E402

SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // 25  # 640
MOUTH_SHAPE_TAIL = (96, 96, 3)

# WebDataset member suffix -> logical field
SUFFIX_TO_KIND = {
    ".mix.wav": "mix",
    ".target.wav": "target",
    ".json": "json",
    ".mouths.pt": "mouths",
}
REQUIRED_KINDS = ("mix", "target", "json", "mouths")


# ---------------------------------------------------------------------------
# TAR scanning / reading
# ---------------------------------------------------------------------------
def _split_member_name(name: str) -> tuple[str, str] | None:
    """Return ``(sample_id, kind)`` for a recognised WebDataset member name."""
    base = os.path.basename(name)
    for suffix, kind in SUFFIX_TO_KIND.items():
        if base.endswith(suffix):
            sample_id = base[: -len(suffix)]
            if not sample_id:
                return None
            return sample_id, kind
    return None


def scan_shard(shard: Path) -> tuple[dict[str, dict[str, str]], int, int]:
    """List complete/uncomplete samples in a shard without reading payloads.

    Returns ``(samples, num_members, num_incomplete)`` where ``samples`` maps
    ``sample_id -> {kind: member_name}`` for entries that have all four fields.
    """
    per_id: dict[str, dict[str, str]] = {}
    num_members = 0
    with tarfile.open(shard, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parsed = _split_member_name(member.name)
            if parsed is None:
                continue
            num_members += 1
            sample_id, kind = parsed
            per_id.setdefault(sample_id, {})[kind] = member.name
    complete = {sid: fields for sid, fields in per_id.items()
                if all(k in fields for k in REQUIRED_KINDS)}
    return complete, num_members, len(per_id) - len(complete)


def shard_member_index(shard: Path) -> dict[str, tarfile.TarInfo]:
    """Return ``{member_name: TarInfo}`` for file members (one tar scan per shard)."""
    with tarfile.open(shard, "r:*") as tf:
        return {m.name: m for m in tf.getmembers() if m.isfile()}


def read_payloads(tf: tarfile.TarFile, index: dict[str, tarfile.TarInfo],
                  fields: dict[str, str]) -> dict[str, bytes]:
    """Extract the four member payloads of one example using an open tar + index.

    ``TarInfo``-based extraction seeks directly to the member offset, so this is
    O(1) per member instead of re-scanning the (potentially huge) shard.
    """
    payloads: dict[str, bytes] = {}
    for kind, name in fields.items():
        info = index.get(name)
        if info is None:
            raise FileNotFoundError(f"member not found: {name}")
        stream = tf.extractfile(info)
        if stream is None:
            raise FileNotFoundError(f"member not readable: {name}")
        payloads[kind] = stream.read()
    return payloads


# ---------------------------------------------------------------------------
# Audio / mouths / metadata
# ---------------------------------------------------------------------------
def _read_multichannel(wav_bytes: bytes, what: str) -> tuple[np.ndarray, int]:
    """Decode wav bytes to ``(samples, channels) float32`` and validate 16 kHz."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{what}: expected {SAMPLE_RATE} Hz (RealSSA-Sim is 16 kHz), got {sr} Hz; "
            "this converter does not resample"
        )
    if data.size == 0:
        raise ValueError(f"{what}: empty audio")
    return data, sr


def select_channel(data: np.ndarray, channel: str, what: str) -> np.ndarray:
    """Apply the ``--channel`` policy to ``(samples, channels)`` audio."""
    channels = data.shape[1]
    if channel == "mean":
        return data.mean(axis=1).astype(np.float32)
    if channel not in {"0", "1", "2", "3"}:
        raise ValueError(f"--channel must be mean|0|1|2|3, got {channel!r}")
    index = int(channel)
    if channels <= index:
        raise ValueError(
            f"{what}: --channel {index} requested but the wav has only {channels} channel(s)"
        )
    return data[:, index].astype(np.float32)


def load_mouths(mouths_bytes: bytes, sample_id: str) -> torch.Tensor:
    """Load the RealSSA uint8 RGB mouth tensor ``[T,96,96,3]``."""
    mouths = torch.load(io.BytesIO(mouths_bytes), map_location="cpu", weights_only=True)
    if not torch.is_tensor(mouths):
        raise ValueError(f"{sample_id}: .mouths.pt is not a torch tensor ({type(mouths)})")
    if mouths.dtype != torch.uint8:
        raise ValueError(f"{sample_id}: expected uint8 mouth crops, got {mouths.dtype}")
    if mouths.ndim != 4 or tuple(mouths.shape[1:]) != MOUTH_SHAPE_TAIL:
        raise ValueError(
            f"{sample_id}: expected mouth crops [T,96,96,3], got {tuple(mouths.shape)}"
        )
    return mouths


def _component_type(component: dict) -> str:
    for key in ("type", "kind", "component_type"):
        value = component.get(key)
        if isinstance(value, str) and value:
            return value.lower()
    return "unknown"


def _component_path(component: dict, sample_id: str, index: int) -> str:
    """Provenance-only path.

    RealSSA ``path`` fields point at generation-time files that are not part of the
    release.  We keep a string (never an existing local path) so downstream
    ``generate_near_json`` can safely skip it, and never treat it as downloaded
    data.
    """
    value = component.get("path")
    if isinstance(value, str) and value:
        return value
    return f"realssa-sim://{sample_id}/component{index}"


def find_near_files(near_dir: Path | None, sample_id: str) -> list[str]:
    """Optional external near waveforms: ``<near_dir>/<id>.near*.wav``."""
    if near_dir is None:
        return []
    matches = sorted(near_dir.glob(f"{sample_id}.near*.wav")) + sorted(
        near_dir.glob(f"{sample_id}_near*.wav")
    )
    return [str(p.resolve()) for p in matches if p.is_file()]


def build_meta(
    sample_id: str,
    source_meta: dict,
    channel: str,
    aligned_frames: int,
    near_paths: list[str],
) -> dict:
    """Create the compact metadata written next to the mixture."""
    raw_components = source_meta.get("components") or []
    components: list[dict] = []
    if isinstance(raw_components, list):
        for i, comp in enumerate(raw_components):
            if not isinstance(comp, dict):
                continue
            entry = {
                "type": _component_type(comp),
                "path": _component_path(comp, sample_id, i),
                "source": comp.get("source"),
                "sir_db": comp.get("sir_db", comp.get("sir", comp.get("SIR"))),
                "position": comp.get("position"),
            }
            components.append(entry)
    # optional external near waveforms (not shipped by RealSSA-Sim)
    for near_path in near_paths:
        try:
            n = sf.info(near_path).frames
        except Exception:  # pragma: no cover - unreadable file
            n = None
        components.append({"type": "near", "path": near_path, "source": "external-near",
                           "sir_db": None, "position": None, "n_samples": n})
    meta = {
        "sample_id": sample_id,
        "scene": source_meta.get("scene"),
        "room": source_meta.get("room"),
        "target_position": source_meta.get("target_position", source_meta.get("target_pos")),
        "transcript": source_meta.get("transcript"),
        "components": components,
        "provenance": {
            "dataset": "RealSSA-Sim",
            "channel_policy": channel,
            "audio": "16 kHz mono PCM16",
            "mouths": "uint8 RGB [T,96,96,3]",
            "aligned_frames": int(aligned_frames),
            "samples_per_frame": SAMPLES_PER_FRAME,
            "source_json_keys": sorted(source_meta.keys()),
            "note": "components[].path values are generation-time provenance only; "
                    "RealSSA-Sim does not ship near-speaker waveforms",
        },
    }
    return meta


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _atomic_save_mouths(mouths: torch.Tensor, destination: Path) -> None:
    """Publish a complete tensor without overwriting a shared destination inode."""
    destination = Path(destination)
    try:
        target_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        target_mode = None
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    # O_EXCL makes the unpredictable name exclusive. Capture the caller's normal
    # creation mode (including umask), then make the still-empty file private
    # before serializing any tensor bytes.
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as stream:
            creation_mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode)
            os.fchmod(stream.fileno(), 0o600)
            torch.save(mouths, stream)
            stream.flush()
            os.fchmod(stream.fileno(), creation_mode if target_mode is None else target_mode)
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class Converter:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).expanduser().resolve()
        self.near_dir = Path(args.near_dir).expanduser().resolve() if args.near_dir else None
        self.stats: dict[str, dict] = {}

    # -- output helpers ---------------------------------------------------
    def _prepare_out(self) -> None:
        if self.args.dry_run:
            return
        self.out.mkdir(parents=True, exist_ok=True)

    def _write_sample(self, split: str, sample_id: str, payloads: dict[str, bytes]) -> dict:
        args = self.args
        mix_bytes = payloads["mix"]
        target_bytes = payloads["target"]
        json_bytes = payloads["json"]
        mouths_bytes = payloads["mouths"]

        # write audio first (cheap); the heavy JSON/mouth parsing happens after a
        # successful decode so broken samples are reported with a clear message
        mix_mc, _ = _read_multichannel(mix_bytes, f"{sample_id}.mix.wav")
        target_mc, _ = _read_multichannel(target_bytes, f"{sample_id}.target.wav")
        mix_mono = select_channel(mix_mc, args.channel, f"{sample_id}.mix.wav")
        target_mono = select_channel(target_mc, args.channel, f"{sample_id}.target.wav")
        mouths = load_mouths(mouths_bytes, sample_id)
        source_meta = json.loads(json_bytes.decode("utf-8"))

        mouth_frames = int(mouths.shape[0])
        if args.align:
            frames = min(mouth_frames, len(mix_mono) // SAMPLES_PER_FRAME,
                         len(target_mono) // SAMPLES_PER_FRAME)
        else:
            frames = mouth_frames
        if frames <= 0:
            raise ValueError(f"{sample_id}: no aligned frames (mouth={mouth_frames}, "
                             f"mix={len(mix_mono)}, target={len(target_mono)})")
        if args.align:
            mix_mono = mix_mono[: frames * SAMPLES_PER_FRAME]
            target_mono = target_mono[: frames * SAMPLES_PER_FRAME]
            mouths = mouths[:frames]

        split_dir = self.out / split
        mix_path = split_dir / "mix" / f"{sample_id}.wav"
        target_path = split_dir / "target" / f"{sample_id}.wav"
        mouths_path = split_dir / "mouths" / f"{sample_id}.pt"
        meta_path = split_dir / "mix" / f"{sample_id}_meta.json"

        mix_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        mouths_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(mix_path), mix_mono, SAMPLE_RATE, subtype="PCM_16")
        sf.write(str(target_path), target_mono, SAMPLE_RATE, subtype="PCM_16")
        _atomic_save_mouths(mouths.contiguous(), mouths_path)

        near_paths = find_near_files(self.near_dir, sample_id)
        meta = build_meta(sample_id, source_meta, args.channel, frames, near_paths)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return {
            "sample_id": sample_id,
            "frames": int(frames),
            "samples": int(frames * SAMPLES_PER_FRAME),
            "mouth_frames_original": mouth_frames,
            "num_near": len(near_paths),
            "mix": str(mix_path),
            "target": str(target_path),
            "mouths": str(mouths_path),
            "meta": str(meta_path),
        }

    def convert_split(self, split: str, shards: list[Path]) -> dict:
        args = self.args
        stats = {"split": split, "shards": len(shards), "samples": 0, "skipped_incomplete": 0,
                 "skipped_error": 0, "samples_with_external_near": 0, "examples": []}
        remaining = args.limit_samples
        for shard in shards:
            complete, num_members, num_incomplete = scan_shard(shard)
            stats["skipped_incomplete"] += num_incomplete
            sample_ids = sorted(complete)
            if args.dry_run:
                take = len(sample_ids) if remaining is None else min(remaining, len(sample_ids))
                stats["samples"] += take
                stats["examples"].extend(sample_ids[:take])
                if remaining is not None:
                    remaining -= take
                    if remaining <= 0:
                        break
                continue
            # one tar open per shard; members are addressed via the TarInfo index
            with tarfile.open(shard, "r:*") as tf:
                index = {m.name: m for m in tf.getmembers() if m.isfile()}
                for sample_id in sample_ids:
                    if remaining is not None and remaining <= 0:
                        break
                    try:
                        payloads = read_payloads(tf, index, complete[sample_id])
                        info = self._write_sample(split, sample_id, payloads)
                    except Exception as exc:  # noqa: BLE001 - keep converting, report
                        stats["skipped_error"] += 1
                        print(f"  [warn] {split}/{sample_id} from {shard.name}: {exc}")
                        continue
                    stats["samples"] += 1
                    stats["samples_with_external_near"] += int(info["num_near"] > 0)
                    stats["examples"].append(sample_id)
                    if remaining is not None:
                        remaining -= 1
                    if stats["samples"] % 50 == 0:
                        print(f"  {split}: converted {stats['samples']} samples ...")
            if remaining is not None and remaining <= 0:
                break
        return stats

    def run(self) -> dict:
        args = self.args
        src = Path(args.src).expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"--src does not exist: {src}")

        shards_by_split: dict[str, list[Path]] = {}
        if src.is_file():
            if len(args.splits) != 1:
                raise SystemExit(
                    f"--src is a single tar file, so --splits must name exactly one split "
                    f"(got {args.splits}); e.g. --splits tt"
                )
            shards_by_split[args.splits[0]] = [src]
        else:
            for split in args.splits:
                split_dir = src / split
                if split_dir.is_dir():
                    shards = sorted(list(split_dir.glob("*.tar")) + list(split_dir.glob("*.tar.gz")))
                else:
                    shards = []
                if not shards and split == args.splits[0] and any(src.glob("*.tar")):
                    # single-split download directory (e.g. --src RealSSA-Sim/tt)
                    shards = sorted(list(src.glob("*.tar")) + list(src.glob("*.tar.gz")))
                if not shards:
                    print(f"  [warn] no *.tar shards found for split {split!r} under {src}")
                    continue
                if args.limit_shards is not None:
                    shards = shards[: args.limit_shards]
                shards_by_split[split] = shards
        if not shards_by_split:
            raise SystemExit(
                f"no shards to convert: looked for {src}/{{{'/'.join(args.splits)}}}/*.tar "
                f"(and {src}/*.tar for the first split). Download a shard from the gated "
                f"RealSSA-Sim dataset first, e.g. tt/tt-000000.tar."
            )

        total_shards = sum(len(v) for v in shards_by_split.values())
        print(f"[realssa_to_falcon] src={src} out={self.out} channel={args.channel} "
              f"align={args.align} dry_run={args.dry_run} shards={total_shards}")
        self._prepare_out()
        results = {}
        for split, shards in shards_by_split.items():
            print(f"[realssa_to_falcon] split {split}: {len(shards)} shard(s)")
            stats = self.convert_split(split, shards)
            results[split] = stats
            mode = "would convert" if args.dry_run else "converted"
            print(f"  {split}: {mode} {stats['samples']} sample(s); "
                  f"skipped incomplete={stats['skipped_incomplete']}, errors={stats['skipped_error']}")

        summary = {"src": str(src), "out": str(self.out), "channel": args.channel,
                   "align": args.align, "dry_run": args.dry_run,
                   "splits": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}}
        if args.dry_run:
            print("\n[dry-run] no files were written. First samples per split:")
            for split, stats in results.items():
                print(f"  {split}: {stats['examples'][:5]}{' ...' if len(stats['examples']) > 5 else ''}")
            return summary

        # Build DataPreProcess JSON indexes for the converted splits.
        if args.build_index:
            converted = [s for s, st in results.items() if st["samples"] > 0]
            if converted:
                print(f"[realssa_to_falcon] building mix/target/near JSON via DataPreProcess for {converted}")
                indexes = build_indexes(self.out, converted)
                summary["indexes"] = indexes
                for split, idx in indexes.items():
                    print(f"  {split}: mix.json={idx['mix']} target.json={idx['target']} "
                          f"near.json={idx['near']} ({idx['num']} samples)")
            else:
                print("[realssa_to_falcon] no samples converted; skipped JSON index build")
        else:
            print("[realssa_to_falcon] --no-index: JSON indexes were not built. Run "
                  "DataPreProcess/process_lrs23.py <out> manually if needed.")

        no_external_near = any(st["samples"] > 0 and st["samples_with_external_near"] == 0
                               for st in results.values())
        if no_external_near:
            print("\n[realssa_to_falcon] NOTE: RealSSA-Sim ships no near-speaker waveforms and no "
                  "external --near-dir was supplied.\n  near.json entries fall back to mixture "
                  "sentinels -> the dataset skips them -> K=0 -> the NASS contrastive loss stays "
                  "inactive.\n  This matches the released-data limitation (docs/REPRODUCTION_PLAN.md "
                  "§5.1 Step 4); pass --near-dir to use compatible external negatives.")
        print("\n[realssa_to_falcon] done. Point the DataModule at: "
              f"train_dir={self.out}/tr valid_dir={self.out}/cv test_dir={self.out}/tt")
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert RealSSA-Sim WebDataset TAR shards to the Falcon folder layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", required=True,
                        help="RealSSA-Sim root containing {tr,cv,tt}/*.tar, or a single .tar file.")
    parser.add_argument("--out", required=True, help="Output dataset root (folder layout).")
    parser.add_argument("--splits", default="tr,cv,tt", help="Comma-separated splits to convert.")
    parser.add_argument("--channel", default="mean", choices=["mean", "0", "1", "2", "3"],
                        help="4-channel -> mono policy (use the same for mixture and target).")
    parser.add_argument("--limit-shards", type=int, default=None,
                        help="Only the first N shards per split (sorted by name).")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Maximum samples to convert per split.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan shards and report counts/pairing without writing files.")
    parser.add_argument("--near-dir", default=None,
                        help="Optional dir with <id>.near*.wav external near waveforms.")
    parser.add_argument("--no-index", dest="build_index", action="store_false",
                        help="Do not build mix/target/near JSON indexes after conversion.")
    parser.add_argument("--no-align", dest="align", action="store_false",
                        help="Keep raw audio/mouth lengths instead of trimming to min(frames) alignment.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete an existing non-empty --out before converting.")
    parser.set_defaults(build_index=True, align=True)
    args = parser.parse_args(argv)
    args.splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not args.splits:
        raise SystemExit("--splits must not be empty")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    if not args.dry_run and out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output directory {out} exists and is not empty; pass --overwrite")
        shutil.rmtree(out)
    try:
        Converter(args).run()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI error surface
        raise SystemExit(f"realssa_to_falcon failed: {type(exc).__name__}: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
