"""Compare released Falcon and v3 on identical RealSSA 2s crops."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
import yaml
from torch.nn import functional as F
from look2hear.models import FalconSpeakerContrastive
from look2hear.models.FalconStreaming import FalconStreaming
from look2hear.videomodels import FRCNNVideoModel
from look2hear.metrics.core import si_sdr_improvement
from tools.streaming.common import read_checkpoint
from tools.streaming.realssa_tar import RealSSATarDataset


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_baseline(path, device):
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/Falcon.yml").read_text())
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("architecture") != "FalconSpeakerContrastive":
        raise ValueError("expected released FalconSpeakerContrastive checkpoint")
    state = ckpt["state_dict"]
    audio = FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(config["audionet"]["audionet_config"]))
    vcfg = copy.deepcopy(config["videonet"]["videonet_config"])
    vcfg.pop("pretrain", None)
    visual = FRCNNVideoModel(**vcfg)
    audio.load_state_dict({k.removeprefix("audio_model."): v for k, v in state.items() if k.startswith("audio_model.")}, strict=True)
    visual.load_state_dict({k.removeprefix("video_model."): v for k, v in state.items() if k.startswith("video_model.")}, strict=True)
    return audio.to(device).eval(), visual.to(device).eval(), ckpt.get("epoch")


def validate_reference(target):
    if target.numel() == 0 or not torch.isfinite(target).all():
        raise ValueError("empty or nonfinite target")
    if torch.count_nonzero(target - target.reshape(-1)[0]) == 0:
        raise ValueError("degenerate constant target: zero centered energy")


@torch.inference_mode()
def predict(model, visual, stream, mix, mouth, device):
    x, v = mix[None].to(device), mouth[None].to(device)
    baseline = model(x, visual(v.unsqueeze(1)))[0, 0]
    padded = torch.cat((v, v[:, -1:].expand(-1, 2, -1, -1)), dim=1)
    streaming = stream(F.pad(x, (0, 1280)), padded)[0, :32000]
    if baseline.shape != mix.shape or streaming.shape != mix.shape:
        raise ValueError("unexpected output shape")
    if not torch.isfinite(baseline).all() or not torch.isfinite(streaming).all():
        raise ValueError("nonfinite output")
    return baseline.cpu(), streaming.cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tar-index", required=True)
    p.add_argument("--streaming-checkpoint", required=True)
    p.add_argument("--baseline-checkpoint", default="/root/.cache/falcon_weights/falcon.pt")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0", "cuda:1"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=16, help="number of examples; 0 means all")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.limit < 0 or a.threads < 1:
        p.error("invalid limit or thread count")
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(a.threads)
    root = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__).resolve(), root / "tools/streaming/realssa_tar.py", root / "tools/streaming/common.py", root / "configs/Falcon.yml"]
    for directory in ("models", "videomodels", "layers", "losses", "metrics", "utils"):
        code_paths.extend(sorted((root / "look2hear" / directory).rglob("*.py")))
    code_hashes = {str(path.relative_to(root)): sha(path) for path in code_paths}
    inputs = (a.tar_index, a.baseline_checkpoint, a.streaming_checkpoint)
    input_hashes = [sha(path) for path in inputs]
    data = RealSSATarDataset(a.tar_index, seed=42, crop="center", augment=False)
    if data.split not in ("cv", "tt"):
        raise ValueError("evaluation requires cv or tt")
    model, visual, epoch = load_baseline(a.baseline_checkpoint, a.device)
    ckpt = read_checkpoint(a.streaming_checkpoint, "cpu")
    stream = FalconStreaming().to(a.device)
    stream.load_state_dict(ckpt["state_dict"], strict=True)
    stream.eval()
    rows = []
    for i in range(len(data) if a.limit == 0 else min(a.limit, len(data))):
        row = {"id": data.samples[i]["id"], "index": i}
        try:
            mix, target, mouth = data[i]
            validate_reference(target)
            old, new = predict(model, visual, stream, mix, mouth, a.device)
            for name, estimate in (("falcon", old), ("streaming", new)):
                value = si_sdr_improvement(estimate, target, mix)
                row[name + "_si_sdri_db"] = float(value) if math.isfinite(value) else None
            row["status"] = "ok" if all(row[k] is not None for k in ("falcon_si_sdri_db", "streaming_si_sdri_db")) else "invalid_metric"
        except (OSError, ValueError, RuntimeError) as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"{i+1} {row}", flush=True)
    good = [r for r in rows if r["status"] == "ok"]
    summary = {"attempted": len(rows), "valid_pairs": len(good), "failed": len(rows) - len(good)}
    if good:
        for name in ("falcon", "streaming"):
            key = name + "_si_sdri_db"
            summary[key + "_mean"] = sum(r[key] for r in good) / len(good)
        summary["streaming_minus_falcon_db"] = summary["streaming_si_sdri_db_mean"] - summary["falcon_si_sdri_db_mean"]
    if any(sha(path) != digest for path, digest in zip(inputs, input_hashes)) or any(sha(root / path) != digest for path, digest in code_hashes.items()):
        raise RuntimeError("evaluation input or code changed during run")
    result = {
        "scope": "same center-cropped 2s RealSSA, ch0; baseline full 2s vs v3 160ms chunks; not paper protocol",
        "preprocessing": "uint8 grayscale truncation, center 88x88, normalized; identical inputs to both",
        "split": data.split, "index": str(Path(a.tar_index).resolve()), "index_sha256": input_hashes[0],
        "baseline_checkpoint": str(Path(a.baseline_checkpoint).resolve()), "baseline_sha256": input_hashes[1], "baseline_epoch": epoch,
        "streaming_checkpoint": str(Path(a.streaming_checkpoint).resolve()), "streaming_sha256": input_hashes[2],
        "streaming_steps": ckpt.get("steps"), "streaming_training_status": ckpt.get("training_status"),
        "metric": "zero-mean SI-SDR improvement in dB, project si_sdr_improvement",
        "short_input_padding": "samples shorter than 2s are zero-padded by the dataset; SI-SDRi uses the full 2s including padding",
        "code_hash_scope": "all Python files under look2hear models/videomodels/layers/losses/metrics/utils plus evaluator, dataset, common, Falcon config; excludes installed dependencies and non-Python resources",
        "code_sha256": code_hashes,
        "torch_version": str(torch.__version__), "device": a.device, "threads": a.threads, "summary": summary, "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
