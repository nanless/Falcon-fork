#!/usr/bin/env python
"""Export the frozen FRCNN lip-reading backbone from a released Falcon checkpoint.

Background
----------
The released Falcon checkpoints (``falcon.pt`` / ``falcon-big.pt``) are dictionaries::

    {"state_dict": {...}, "architecture": "FalconSpeakerContrastive", "epoch": 714}

The visual encoder (a frozen ``FRCNNVideoModel``, 11,186,688 params) is stored under the
``video_model.*`` prefix (137 tensors). The training code needs it as a **separate**
pretrained-backbone file in the format read by ``FRCNNVideoModel.init_from``::

    torch.load(path)["model_state_dict"]      # keys without the "video_model." prefix
    # look2hear/videomodels/frcnn_videomodel.py:86-88 -> update_frcnn_parameter(model, ...)

``configs/Falcon.yml`` expects that file at ``pretrain_zoo/frcnn_128_512.backbone.pth.tar``.
The upstream README only has a TODO for this file, so this script rebuilds it from the
published weights.  No large file is written into the repository by default: the default
destination lives in the Hugging-Face cache directory, next to the source checkpoints.

Usage
-----
::

    # small model, default paths
    python tools/repro/extract_video_backbone.py

    # big model
    python tools/repro/extract_video_backbone.py --variant big

    # explicit paths / overwrite / inspect only
    python tools/repro/extract_video_backbone.py --src /path/falcon.pt --dst /path/out.pth.tar --force
    python tools/repro/extract_video_backbone.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CACHE_DIR = Path("/root/.cache/falcon_weights")
VARIANTS = {
    "small": {
        "src": CACHE_DIR / "falcon.pt",
        "dst": CACHE_DIR / "frcnn_128_512.backbone.pth.tar",
        "keys": 137,
    },
    "big": {
        "src": CACHE_DIR / "falcon-big.pt",
        # distinct default name: falcon.pt and falcon-big.pt ship *different* visual-backbone
        # weights (same architecture/key set), so the big export must not clobber the small one.
        "dst": CACHE_DIR / "frcnn_128_512.backbone.big.pth.tar",
        "keys": 137,
    },
}
PREFIX = "video_model."


def load_checkpoint(path: Path) -> dict:
    """Load a released Falcon checkpoint, tolerating both dict and bare state_dict files."""
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "Download it first (see AGENTS.md / docs/REPRODUCTION_PLAN.md) or pass --src."
        )
    try:
        # new checkpoints contain only tensors + str/int metadata -> weights_only is safe
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # pragma: no cover - fall back for older/pickled checkpoints
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_video_state_dict(ckpt: dict) -> dict:
    """Return ``video_model.*`` tensors with the prefix stripped."""
    sd = ckpt.get("state_dict", ckpt)
    if not isinstance(sd, dict):
        raise TypeError(f"unexpected checkpoint layout: state_dict is {type(sd).__name__}")
    video = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    if not video:
        raise KeyError(
            f"no tensors with prefix {PREFIX!r} in the checkpoint; "
            f"found prefixes: {sorted({k.split('.')[0] for k in sd})}"
        )
    return video


def build_output(video_sd: dict, src: Path, ckpt: dict) -> dict:
    """Build the file payload expected by ``FRCNNVideoModel.init_from``."""
    meta = {
        "source_checkpoint": src.name,
        "source_architecture": ckpt.get("architecture"),
        "source_epoch": ckpt.get("epoch"),
        "num_keys": len(video_sd),
        "note": "Extracted from the released audio-visual checkpoint; visual encoder is frozen "
                "during training (look2hear/videomodels/frcnn_videomodel.py: update_frcnn_parameter).",
    }
    return {"model_state_dict": video_sd, "meta": meta}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Export the frozen FRCNN visual backbone from a released Falcon checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="small",
                    help="which released checkpoint to read (default: small)")
    ap.add_argument("--src", type=Path, default=None,
                    help="source Falcon checkpoint (default: cache dir falcon.pt / falcon-big.pt)")
    ap.add_argument("--dst", type=Path, default=None,
                    help="output .pth.tar path (default: cache dir frcnn_128_512.backbone.pth.tar; "
                         "the repository is never written by default)")
    ap.add_argument("--expect-keys", type=int, default=None,
                    help="expected number of video tensors (default: 137)")
    ap.add_argument("--force", action="store_true", help="overwrite --dst if it exists")
    ap.add_argument("--dry-run", action="store_true", help="inspect only, do not write")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    variant = VARIANTS[args.variant]
    src: Path = args.src or variant["src"]
    dst: Path = args.dst or variant["dst"]
    expect_keys = args.expect_keys if args.expect_keys is not None else variant["keys"]

    say = (lambda *a: None) if args.quiet else print

    say(f"[extract_video_backbone] variant={args.variant}")
    say(f"  src = {src}")
    say(f"  dst = {dst}")

    try:
        ckpt = load_checkpoint(src)
    except FileNotFoundError as exc:
        say(f"  ERROR: {exc}")
        return 2
    say(f"  checkpoint: architecture={ckpt.get('architecture')!r} epoch={ckpt.get('epoch')!r} "
        f"state_dict_entries={len(ckpt.get('state_dict', ckpt))}")

    try:
        video_sd = extract_video_state_dict(ckpt)
    except (KeyError, TypeError) as exc:
        say(f"  ERROR: {exc}")
        return 2
    n_params = sum(v.numel() for v in video_sd.values())
    dtypes = sorted({str(v.dtype) for v in video_sd.values()})
    say(f"  video tensors: {len(video_sd)} keys, {n_params:,} elements, dtypes={dtypes}")
    say(f"  sample keys: {list(video_sd)[:3]}{' ...' if len(video_sd) > 3 else ''}")

    if expect_keys is not None and len(video_sd) != expect_keys:
        say(f"  WARNING: expected {expect_keys} keys, got {len(video_sd)} "
            "(start/endswith filters can differ between releases)")
    if dst.exists() and not args.force and not args.dry_run:
        say(f"  ERROR: {dst} already exists; pass --force to overwrite")
        return 2
    if args.dry_run:
        say("  [dry-run] nothing written")
        return 0

    payload = build_output(video_sd, src, ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    say(f"  wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    say(f"  meta: {json.dumps(payload['meta'], ensure_ascii=False)}")
    say("  load back check:")
    back = torch.load(dst, map_location="cpu", weights_only=False)
    say(f"    torch.load(dst)['model_state_dict'] -> {len(back['model_state_dict'])} keys, "
        f"meta present={('meta' in back)}")
    ok = set(back["model_state_dict"]) == set(video_sd)
    say(f"    key set identical to source: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
