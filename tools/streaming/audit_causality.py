"""Audit chunk boundaries on a real aligned utterance with a streaming checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools.streaming.common import FRAMES, SAMPLES, load_mono, mouth_tensor, read_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    if args.threads < 1:
        parser.error("--threads must be positive")
    model_hash = sha("look2hear/models/FalconStreaming.py")
    checkpoint_hash = sha(args.checkpoint)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = Path(args.index_dir)
    mix = json.loads((root / "mix.json").read_text())[0]
    target = json.loads((root / "target.json").read_text())[0]
    if Path(mix[0]).name != Path(target[0]).name:
        raise ValueError("index identity mismatch")
    count = 3
    x = load_mono(mix[0], 0, count * SAMPLES).unsqueeze(0)
    raw_video = torch.load(target[1], map_location="cpu", weights_only=True)
    y = mouth_tensor(raw_video[:count * FRAMES]).unsqueeze(0)
    if x.shape != (1, count * SAMPLES) or y.shape != (1, count * FRAMES, 88, 88):
        raise ValueError("three complete aligned chunks required")
    checkpoint = read_checkpoint(args.checkpoint, "cpu")
    model = FalconStreaming(**checkpoint["model_config"]).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    generator = torch.Generator().manual_seed(20261111)
    rows = []
    with torch.inference_mode():
        baseline = model(x, y)
        # Future chunks may change, but completed chunks must be invariant.
        for chunk in range(1, count):
            for kind in ("audio", "video"):
                altered_x, altered_y = x.clone(), y.clone()
                if kind == "audio":
                    view = altered_x[:, chunk * SAMPLES:]
                    view.add_(0.4 * torch.randn(view.shape, generator=generator))
                else:
                    view = altered_y[:, chunk * FRAMES:]
                    view.add_(2 * torch.randn(view.shape, generator=generator))
                out = model(altered_x, altered_y)
                prior = (out[:, :chunk * SAMPLES] - baseline[:, :chunk * SAMPLES]).abs().max().item()
                if prior > 2e-6:
                    raise AssertionError(f"{kind}: future chunk {chunk} changed earlier output: {prior}")
                rows.append({"kind": kind, "changed_from_chunk": chunk,
                             "prior_max_abs_delta": prior,
                             "changed_chunk_max_abs_delta": (out[:, chunk*SAMPLES:] - baseline[:, chunk*SAMPLES:]).abs().max().item()})
        # Report (rather than require for arbitrary weights) whether within-chunk
        # later information influences earlier output. Contract tests use crafted
        # weights/inputs to establish this capability deterministically.
        for kind in ("audio", "video"):
            altered_x, altered_y = x.clone(), y.clone()
            if kind == "audio":
                view = altered_x[:, SAMPLES + SAMPLES // 2:2 * SAMPLES]
                view.add_(0.4 * torch.randn(view.shape, generator=generator))
            else:
                view = altered_y[:, FRAMES + 2:2 * FRAMES]
                view.add_(2 * torch.randn(view.shape, generator=generator))
            out = model(altered_x, altered_y)
            rows.append({"kind": kind + "_within_chunk",
                         "earlier_same_chunk_max_abs_delta": (out[:, SAMPLES:SAMPLES + SAMPLES//2] - baseline[:, SAMPLES:SAMPLES + SAMPLES//2]).abs().max().item(),
                         "previous_chunk_max_abs_delta": (out[:, :SAMPLES] - baseline[:, :SAMPLES]).abs().max().item()})
    if model_hash != sha("look2hear/models/FalconStreaming.py") or checkpoint_hash != sha(args.checkpoint):
        raise RuntimeError("model or checkpoint changed during audit")
    result = {"scope": "CPU aligned real utterance, three 120ms chunks; prior chunks invariant under later chunk perturbations",
              "checkpoint_format": checkpoint["format"], "training_status": checkpoint.get("training_status"),
              "seed": 20261111, "threads": args.threads,
              "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checkpoint_hash,
              "model_sha256": model_hash, "audio": mix[0], "audio_sha256": sha(mix[0]),
              "mouth": target[1], "mouth_sha256": sha(target[1]), "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"boundary_cases": 4, "within_chunk_diagnostics": 2,
                      "max_prior_delta": max(row["prior_max_abs_delta"] for row in rows if "prior_max_abs_delta" in row),
                      "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
