"""Full aligned-utterance RealSSA-Sim comparison of released and block-causal Falcon."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch
from torchvision.transforms.functional import rgb_to_grayscale

from look2hear.metrics.core import SDR_FILTER_LENGTH, sdr, si_sdr
from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.common import FRAMES, SAMPLES, SAMPLES_PER_FRAME, SR, read_checkpoint
from tools.streaming.evaluate_realssa import index_provenance, load_baseline, predict, sha
from tools.streaming.realssa_tar import RealSSATarDataset


FORMAT = "realssa-utterance-paired-v1"
METRICS = ("si_snri_db", "sdri_db")


def decode_aligned(mix_bytes, target_bytes, mouth_bytes):
    """Read first-channel audio and the longest common prefix on the 25 fps grid."""
    audio = []
    for payload in (mix_bytes, target_bytes):
        wave, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        if rate != SR or wave.shape[1] != 4 or not np.isfinite(wave).all():
            raise ValueError(f"expected finite 4-channel {SR} Hz audio, got {rate} Hz, {wave.shape}")
        audio.append(wave[:, 0].copy())

    frames = torch.load(io.BytesIO(mouth_bytes), map_location="cpu", weights_only=True)
    if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8:
        raise ValueError("mouth payload must be a uint8 tensor")
    if frames.ndim == 4 and frames.shape[1:] == (96, 96, 3):
        gray = rgb_to_grayscale(frames.permute(0, 3, 1, 2)).squeeze(1)
    elif frames.ndim == 3 and frames.shape[1:] == (96, 96):
        gray = frames
    else:
        raise ValueError(f"invalid mouth shape {tuple(frames.shape)}")

    usable_frames = min(len(gray), *(len(wave) // SAMPLES_PER_FRAME for wave in audio))
    if usable_frames == 0:
        raise ValueError("no common aligned 40 ms interval")
    usable_samples = usable_frames * SAMPLES_PER_FRAME
    video = (gray[:usable_frames, 4:92, 4:92].float() / 255.0 - 0.421) / 0.165
    geometry = {
        "aligned_samples": usable_samples,
        "aligned_frames": usable_frames,
        "mixture_trimmed_samples": len(audio[0]) - usable_samples,
        "target_trimmed_samples": len(audio[1]) - usable_samples,
        "video_trimmed_frames": len(gray) - usable_frames,
        "streaming_pad_samples": (-usable_samples) % SAMPLES,
        "streaming_pad_frames": (-usable_frames) % FRAMES,
    }
    return torch.from_numpy(audio[0][:usable_samples]), torch.from_numpy(audio[1][:usable_samples]), video, geometry


class RealSSAUtteranceDataset(RealSSATarDataset):
    """Reuse indexed TAR identity checks; load full aligned utterances without training crops."""

    def __getitem__(self, index):
        record = self.samples[index]
        shard = self.shards[record["shard"]]["path"]
        payloads = []
        with open(shard, "rb") as stream:
            for kind in ("mix", "target", "mouth"):
                offset, size = record["fields"][kind]
                stream.seek(offset)
                payload = stream.read(size)
                if len(payload) != size:
                    raise ValueError(f"truncated {kind} for {record['id']} in {shard}")
                payloads.append(payload)
        return decode_aligned(*payloads)


def predict_utterance(released, visual, student, mixture, mouth, device):
    """Run one baseline pass and student chunks on one integer-sample video clock."""
    if (mixture.ndim != 1 or mouth.ndim != 3 or len(mixture) == 0
            or len(mixture) != len(mouth) * SAMPLES_PER_FRAME):
        raise ValueError("audio and video must have aligned full 40 ms intervals")
    # Frame k covers samples [640*k, 640*(k+1)); chunk j sees only frames 3*j to 3*j+2.
    # The shared predictor pads the trailing chunk and trims its output before scoring.
    return predict(released, visual, student, mixture, mouth, device)


def finite_difference(estimate, mixture):
    value = estimate - mixture
    return float(value) if math.isfinite(value) else None


def summarize(rows):
    summary = {
        "attempted": len(rows),
        "constant_targets": sum(r["status"] == "constant_target" for r in rows),
        "errors": sum(r["status"] == "error" for r in rows),
        "invalid_metrics": sum(r["status"] == "invalid_metric" for r in rows),
    }
    for metric in METRICS:
        baseline_key, student_key = f"released_{metric}", f"streaming_{metric}"
        paired = [r for r in rows if r.get(baseline_key) is not None and r.get(student_key) is not None]
        cell = {"valid_pairs": len(paired)}
        if paired:
            baseline = np.array([r[baseline_key] for r in paired])
            student = np.array([r[student_key] for r in paired])
            cell.update(released_mean=float(baseline.mean()), streaming_mean=float(student.mean()),
                        paired_delta_mean=float((student - baseline).mean()),
                        paired_delta_median=float(np.median(student - baseline)))
        summary[metric] = cell
    return summary


def _publish_json(path, result):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def evaluate(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.threads < 1 or args.start < 0 or args.limit < 0:
        raise ValueError("invalid threads, start, or limit")
    torch.set_num_threads(args.threads)
    paths = {key: Path(getattr(args, key)).resolve() for key in
             ("tar_index", "baseline_checkpoint", "streaming_checkpoint")}
    hashes = {key: sha(path) for key, path in paths.items()}
    data = RealSSAUtteranceDataset(paths["tar_index"], crop="center", augment=False)
    index = json.loads(paths["tar_index"].read_text())
    if data.split not in ("cv", "tt") or args.start >= len(data):
        raise ValueError("evaluation requires a valid start in a cv or tt index")
    if len({sample["id"] for sample in data.samples}) != len(data):
        raise ValueError("evaluation index contains duplicate sample IDs")
    end = min(args.start + args.limit, len(data)) if args.limit else len(data)

    root = Path(__file__).resolve().parents[2]
    code = [Path(__file__).resolve(), root / "configs/Falcon.yml"]
    code.extend(root / "tools/streaming" / f"{name}.py" for name in
                ("evaluate_realssa", "realssa_tar", "common"))
    for module in ("models", "videomodels", "layers", "losses", "metrics", "utils"):
        code.extend(sorted((root / "look2hear" / module).rglob("*.py")))
    code_hashes = {str(path.relative_to(root)): sha(path) for path in code}
    baseline, visual, epoch = load_baseline(paths["baseline_checkpoint"], args.device)
    checkpoint = read_checkpoint(paths["streaming_checkpoint"], "cpu")
    student = FalconStreaming(**checkpoint["model_config"]).to(args.device).eval()
    student.load_state_dict(checkpoint["state_dict"], strict=True)

    rows = []
    for position in range(args.start, end):
        row = {"index": position, "id": data.samples[position]["id"]}
        try:
            mixture, target, mouth, geometry = data[position]
            row.update(geometry)
            if bool((target == target[0]).all()):
                row.update(status="constant_target", error="zero centered target energy")
            else:
                old, new = predict_utterance(baseline, visual, student, mixture, mouth, args.device)
                for metric, score in (("si_snri_db", si_sdr), ("sdri_db", sdr)):
                    reference = score(mixture, target)
                    row[f"released_{metric}"] = finite_difference(score(old, target), reference)
                    row[f"streaming_{metric}"] = finite_difference(score(new, target), reference)
                row["status"] = "ok" if all(
                    row[f"{model}_{metric}"] is not None
                    for model in ("released", "streaming") for metric in METRICS
                ) else "invalid_metric"
        except (OSError, ValueError, RuntimeError) as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if (any(sha(path) != hashes[key] for key, path in paths.items())
            or any(sha(root / path) != digest for path, digest in code_hashes.items())):
        raise RuntimeError("evaluation input or source code changed during the run")
    result = {
        "format": FORMAT,
        "protocol": {
            "scope": "full aligned RealSSA-Sim utterances, first microphone channel",
            "sample_rate": SR, "samples_per_frame": SAMPLES_PER_FRAME,
            "chunk_samples": SAMPLES, "chunk_frames": FRAMES,
            "baseline": "released falcon.pt, batch size 1, entire aligned utterance in one pass",
            "student": "one continuous chunk-wise causal state per utterance",
            "alignment": "common 25 fps prefix; frame k is assigned sample 640*k",
            "mouth_preprocessing": "uint8 grayscale, center 88x88, (x/255-0.421)/0.165",
            "tail": "zero-pad final audio block, repeat last observed mouth, crop output to aligned sample count",
            "si_snri_db": "zero-mean SI-SDR improvement over input mixture",
            "sdri_db": f"fast_bss_eval SDR improvement over input mixture, {SDR_FILTER_LENGTH}-tap filter",
            "constant_targets": "excluded from both paired metrics, reported separately",
            "device": args.device, "threads": args.threads,
            "versions": {package: importlib.metadata.version(package)
                         for package in ("torch", "torchvision", "fast-bss-eval", "soundfile")},
        },
        "identity": {
            **{key: {"path": str(path), "sha256": hashes[key]} for key, path in paths.items()},
            "index_count": len(data), "index_incomplete_count": index.get("incomplete_count"),
            "split": data.split, **index_provenance(index),
            "baseline_epoch": epoch, "streaming_model_config": checkpoint["model_config"],
            "streaming_steps": checkpoint.get("steps"), "code_sha256": code_hashes,
        },
        "start": args.start, "end_exclusive": end, "summary": summarize(rows), "rows": rows,
    }
    _publish_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return int(bool(result["summary"]["errors"] or result["summary"]["invalid_metrics"]))


def merge_reports(index, digest, shards):
    """Require complete, non-overlapping index coverage and a single evaluation identity."""
    samples = index["samples"]
    if not shards or not samples or len({row["id"] for row in samples}) != len(samples):
        raise ValueError("empty or non-unique evaluation index or shards")
    first = shards[0]
    identity = first["identity"]
    if identity["tar_index"]["sha256"] != digest or identity["index_count"] != len(samples):
        raise ValueError("evaluation and merge index differ")
    rows = []
    for shard in sorted(shards, key=lambda row: row["start"]):
        if (shard["format"] != FORMAT or shard["identity"] != identity
                or shard["protocol"] != first["protocol"]):
            raise ValueError("shards have different identities or protocols")
        if (shard["start"] != len(rows) or not shard["rows"]
                or shard["end_exclusive"] != shard["start"] + len(shard["rows"])):
            raise ValueError("shards must cover contiguous, non-overlapping index ranges")
        for offset, row in enumerate(shard["rows"], shard["start"]):
            if offset >= len(samples) or row["index"] != offset or row["id"] != samples[offset]["id"]:
                raise ValueError("row and index identity differ")
        rows.extend(shard["rows"])
    if len(rows) != len(samples):
        raise ValueError("incomplete index coverage")
    return {"format": FORMAT, "identity": identity, "protocol": first["protocol"],
            "start": 0, "end_exclusive": len(rows), "summary": summarize(rows), "rows": rows}


def merge(args):
    paths = [Path(path).resolve() for path in args.shards]
    result = merge_reports(json.loads(Path(args.tar_index).read_text()), sha(args.tar_index),
                           [json.loads(path.read_text()) for path in paths])
    result["input_shards"] = [{"path": str(path), "sha256": sha(path)} for path in paths]
    _publish_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return int(bool(result["summary"]["errors"] or result["summary"]["invalid_metrics"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score a contiguous range")
    combine = commands.add_parser("merge", help="audit and merge a complete set of ranges")
    for command in (run, combine):
        command.add_argument("--tar-index", required=True)
        command.add_argument("--output", required=True)
    run.add_argument("--streaming-checkpoint", required=True)
    run.add_argument("--baseline-checkpoint", default="/root/.cache/falcon_weights/falcon.pt")
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--limit", type=int, default=0, help="0 means all remaining rows")
    combine.add_argument("shards", nargs="+")
    args = parser.parse_args()
    raise SystemExit(evaluate(args) if args.command == "run" else merge(args))


if __name__ == "__main__":
    main()
