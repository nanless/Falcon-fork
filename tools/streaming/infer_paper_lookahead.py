"""Run a saved paper-style lookahead training checkpoint on official TAR data.

Example (run on the server, use a new output directory):
  python -m tools.streaming.infer_paper_lookahead \\
    --checkpoint /root/group-shared/voiceprint/experiments/RUN/epoch-0000-step-00000001.pt \\
    --index /root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/index/full-S1xPSK/tt.json \\
    --output-dir /root/group-shared/voiceprint/experiments/EVAL --count 2
Each utterance is read once from the official TAR index, receives the same
2.12 s input / 2 s output crop as training, and is processed in streaming
120 ms steps with 80 ms lookahead. CSV/JSON/PCM WAV files belong to the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from look2hear.models.FalconPaperStreamPrototype import PaperStreamPrototype
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel
from tools.streaming.lookahead_training import run_lookahead_sequence
from tools.streaming.realssa_lookahead import RealSSALookaheadDataset
from tools.streaming.train_ddp import pairwise_neg_sisdr


class SavedPaperModel(nn.Module):
    """Names match training checkpoint; no separate visual weight file needed."""
    def __init__(self, config):
        super().__init__()
        self.visual = FalconLookaheadVisual(FRCNNVideoModel(backbone_type="resnet", relu_type="prelu", pretrain=None))
        self.core = PaperStreamPrototype(width=config["width"], rnn_hidden=config["rnn_hidden"],
                                         heads=config["heads"], history_frames=config["history_frames"])

    def forward(self, audio, video, output_samples):
        def step(audio_window, video_window, state):
            core_state, video_state = (None, None) if state is None else state
            visual, video_state = self.visual(video_window, video_state)
            output, core_state = self.core(audio_window, visual.transpose(1, 2).contiguous(), core_state)
            return output, (core_state, video_state)
        return run_lookahead_sequence(audio, video, step, output_samples=output_samples)[0]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def si_sdr_values(estimate, target):
    return -pairwise_neg_sisdr(estimate.unsqueeze(1), target.unsqueeze(1)).reshape(-1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True, help="official RealSSA-Sim tr/cv/tt index JSON")
    p.add_argument("--output-dir", required=True, help="new directory under voiceprint/experiments")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--include-reference", action="store_true")
    a = p.parse_args(argv)
    if a.start < 0 or a.count < 1 or a.batch_size < 1 or a.threads < 1:
        p.error("start must be >= 0 and count, batch-size, threads must be positive")
    root = Path("/root/group-shared/voiceprint/experiments").resolve()
    out_dir = Path(a.output_dir).resolve()
    if not out_dir.is_relative_to(root) or out_dir == root:
        p.error("output-dir must be a new subdirectory of voiceprint/experiments")
    device = torch.device(a.device)
    if device.type not in ("cpu", "cuda"):
        p.error("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    if a.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("CUDA BF16 is unsupported")
    torch.set_num_threads(a.threads)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict) or "model" not in checkpoint:
        raise ValueError("not a paper lookahead training checkpoint")
    model = SavedPaperModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    dataset = RealSSALookaheadDataset(a.index, seed=checkpoint["config"]["seed"], crop="center", augment=False)
    stop = min(a.start + a.count, len(dataset))
    if a.start >= stop:
        raise ValueError("start is past the end of the data index")
    loader = DataLoader(Subset(dataset, range(a.start, stop)), batch_size=a.batch_size, shuffle=False, num_workers=0)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, (mix, target, video) in enumerate(loader):
            mix, target, video = mix.to(device), target.to(device), video.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                prediction = model(mix, video, target.shape[1])
            prediction = prediction.float()
            if not torch.isfinite(prediction).all():
                raise RuntimeError("model produced non-finite output")
            scores = si_sdr_values(prediction, target.float())
            baseline = si_sdr_values(mix[:, :target.shape[1]].float(), target.float())
            for j in range(prediction.size(0)):
                index = a.start + batch_number*a.batch_size + j
                target_rms = target[j].float().sub(target[j].float().mean()).square().mean().sqrt().item()
                # PCM float WAV preserves unclipped model output for exact offline evaluation.
                stem = f"sample-{index:06d}"
                sf.write(out_dir / f"{stem}-estimate.wav", prediction[j].cpu().numpy(), 16000, subtype="FLOAT")
                if a.include_reference:
                    sf.write(out_dir / f"{stem}-mixture.wav", mix[j, :target.shape[1]].float().cpu().numpy(), 16000, subtype="FLOAT")
                    sf.write(out_dir / f"{stem}-target.wav", target[j].float().cpu().numpy(), 16000, subtype="FLOAT")
                score, base = float(scores[j]), float(baseline[j])
                valid = target_rms > 1e-8 and math.isfinite(score) and math.isfinite(base)
                rows.append({"index": index, "sample_id": dataset.samples[index]["id"],
                             "estimate_wav": f"{stem}-estimate.wav", "target_rms": target_rms,
                             "si_sdri_db": score-base if valid else None,
                             "si_sdr_db": score if valid else None, "mixture_si_sdr_db": base if valid else None})
    with open(out_dir / "samples.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finite = [r["si_sdri_db"] for r in rows if r["si_sdri_db"] is not None]
    report = {"checkpoint": str(Path(a.checkpoint).resolve()), "checkpoint_sha256": digest(a.checkpoint),
              "checkpoint_updates": checkpoint.get("updates"), "checkpoint_identity": checkpoint.get("identity"),
              "index": str(Path(a.index).resolve()), "index_sha256": digest(a.index),
              "crop": "center", "device": str(device), "precision": a.precision, "threads": a.threads,
              "output_samples": 32000, "chunk_samples": 1920, "lookahead_samples": 1280,
              "input_samples": 33920, "count": len(rows), "valid_si_sdri": len(finite),
              "mean_si_sdri_db": sum(finite)/len(finite) if finite else None,
              "elapsed_seconds": time.monotonic()-started, "samples": rows}
    with open(out_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("count", "valid_si_sdri", "mean_si_sdri_db", "elapsed_seconds")}, ensure_ascii=False))
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
