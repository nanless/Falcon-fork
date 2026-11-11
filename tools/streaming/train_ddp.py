"""Two-GPU streaming trial with reconstruction and optional positive contrastive loss.

Run with torchrun; checkpoint format follows the selected visual mode.
The final incomplete chunk is padded and excluded from the loss.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.training_config import yaml_arguments
from look2hear.losses import pairwise_neg_sisdr
from look2hear.losses.nass_contrastive_loss import NASSContrastiveLoss
from look2hear.models.Falcon_Contrastive import AttentionSpeakerPool
from tools.streaming.common import (
    CHECKPOINT_FORMAT, checkpoint_format_for_visual_mode, FRAMES, SAMPLES, SAMPLES_PER_FRAME, IndexedAudioVisualDataset, read_checkpoint, load_mono, mouth_tensor,
)

SEGMENT_SAMPLES = 32000
SEGMENT_FRAMES = 50
CHUNKS = (SEGMENT_SAMPLES + SAMPLES - 1) // SAMPLES


class TwoSecondDataset(IndexedAudioVisualDataset):
    """Aligned 2-second segments on the same 40 ms grid as streaming data."""
    def __init__(self, directory, *, validation=False):
        super().__init__(directory, SEGMENT_FRAMES // FRAMES)
        self.validation = validation
        self.samples = SEGMENT_SAMPLES
        self.frame_count = SEGMENT_FRAMES
        self.items = [(m, t) for m, t in self.items
                      if m[1] >= self.samples and t[2] >= self.samples]
        if not self.items:
            raise ValueError("no index entries long enough for 2-second segment")

    def __getitem__(self, index):
        if not self.validation:
            return super().__getitem__(index)
        mix, target = self.items[index]
        raw = torch.load(target[1], map_location="cpu", weights_only=True)
        if len(raw) < self.frame_count:
            raise ValueError(f"mouth video is too short for the selected chunk count: {target[1]}")
        max_frame = min((min(int(mix[1]), int(target[2])) - self.samples) // SAMPLES_PER_FRAME,
                        len(raw) - self.frame_count)
        frame_start = max_frame // 2
        start = frame_start * SAMPLES_PER_FRAME
        mixture = load_mono(mix[0], start, self.samples)
        source = load_mono(target[0], start, self.samples)
        video = mouth_tensor(raw[frame_start:frame_start + self.frame_count])
        if len(mixture) != self.samples or len(source) != self.samples or len(video) != self.frame_count:
            raise ValueError(f"incomplete aligned segment for {mix[0]}")
        return mixture, source, video


class SyntheticTwoSeconds(Dataset):
    """Test input, never a speech quality evaluation."""
    def __init__(self, size=16):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(index + 902)
        target = torch.randn(SEGMENT_SAMPLES, generator=gen) * .1
        mixture = target + torch.randn(SEGMENT_SAMPLES, generator=gen) * .05
        video = torch.randn(SEGMENT_FRAMES, 88, 88, generator=gen)
        return mixture, target, video


def pad_two_seconds(mixture, video):
    if mixture.ndim != 2 or mixture.shape[1] != SEGMENT_SAMPLES:
        raise ValueError("expected 2 seconds of mono audio")
    if video.ndim != 4 or video.shape[1:] != (SEGMENT_FRAMES, 88, 88):
        raise ValueError("expected 50 aligned 88x88 video frames")
    padded_audio = F.pad(mixture, (0, CHUNKS * SAMPLES - SEGMENT_SAMPLES))
    padded_video = torch.cat((video, video[:, -1:].expand(-1, CHUNKS * FRAMES - SEGMENT_FRAMES, -1, -1)), dim=1)
    return padded_audio, padded_video




def target_degeneracy_masks(target, silence_rms_threshold):
    """Return SI-SDR-degenerate and near-zero masks for each target.

    Exact constant detection uses min/max equality, independent of float32
    mean-reduction roundoff. A positive threshold additionally classifies
    centered-RMS values below threshold as degenerate.
    """
    if target.ndim != 2 or target.numel() == 0:
        raise ValueError("target must have nonempty [batch,time] shape")
    threshold_sq = float(silence_rms_threshold) ** 2
    centered = target.detach() - target.detach().mean(dim=1, keepdim=True)
    exact_constant = target.detach().amax(dim=1) == target.detach().amin(dim=1)
    degenerate = exact_constant | (centered.square().mean(dim=1) <= threshold_sq)
    silent = target.detach().square().mean(dim=1) <= threshold_sq
    return degenerate, silent

def masked_positive_cosine_loss(query, positive, valid=None, alpha=1.0):
    """Mean over globally valid examples under DDP, retaining an all-zero graph.

    DDP averages gradients across ranks, so rank-local sums are multiplied by
    world size and divided by the detached all-reduced global valid count.
    Every rank using a validity mask participates in exactly one collective.
    """
    losses = float(alpha) * (1.0 - F.cosine_similarity(query, positive, dim=1))
    if valid is None:
        return losses.mean()
    if valid.shape != losses.shape:
        raise ValueError("positive validity mask shape mismatch")
    mask = valid.to(dtype=losses.dtype)
    total = mask.sum().detach().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return (losses * mask).sum() * world / total.clamp_min(1)


def reduce_training_loss_metrics(contrastive, variance, device):
    """Log global valid positive mean and rank-average variance, without gradients."""
    values = torch.stack((
        contrastive.detach() if contrastive is not None else torch.zeros((), device=device),
        variance.detach() if variance is not None else torch.zeros((), device=device),
    ))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return (values[0].item() if contrastive is not None else None,
            values[1].item() if variance is not None else None)

class TrainingWrapper(torch.nn.Module):
    # The auxiliary head is used for training only; checkpoint state is separate.
    def __init__(self, model, positive_contrastive, variance_weight=0.0, variance_floor=0.05, silence_rms_threshold=0.0, exclude_degenerate_contrastive=False):
        super().__init__()
        self.model = model
        self.positive_contrastive = positive_contrastive
        self.variance_weight = float(variance_weight)
        self.variance_floor = float(variance_floor)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.exclude_degenerate_contrastive = bool(exclude_degenerate_contrastive)
        self.speaker_pool = AttentionSpeakerPool(model.channels, 64) if positive_contrastive else None
        self.contrastive = NASSContrastiveLoss(margin=.5) if positive_contrastive else None

    def encode_waveform(self, waveform):
        if waveform.ndim != 2 or waveform.shape[1] != SEGMENT_SAMPLES:
            raise ValueError("speaker encoder requires exactly the valid 2-second waveform")
        framed = F.pad(waveform, (128, 0)).unfold(-1, 256, 128)
        spectrum = torch.fft.rfft(framed.float(), n=256, dim=-1)
        features = torch.cat((spectrum.real, spectrum.imag), dim=-1).transpose(1, 2)
        return self.model.audio_proj(features).unsqueeze(-1)

    def positive_loss(self, output, target):
        query = self.speaker_pool(self.encode_waveform(output))
        with torch.no_grad():
            positive = self.speaker_pool(self.encode_waveform(target))
        # This training objective explicitly has no negatives; preserve alpha/epsilon
        # and detached-reference semantics of NASSContrastiveLoss, without its
        # per-sample Python loop. The general loss still handles negative inputs.
        valid = None
        if self.exclude_degenerate_contrastive:
            degenerate, _ = target_degeneracy_masks(target, self.silence_rms_threshold)
            valid = ~degenerate
        return masked_positive_cosine_loss(query, positive, valid, self.contrastive.alpha_pos)

    def variance_penalty(self, target):
        # Rank-local regularizer: only this rank's active examples are used.
        # Only the auxiliary pool receives gradients; encoder features are detached.
        features = self.encode_waveform(target).detach()
        embeddings = F.normalize(self.speaker_pool(features), dim=1)
        degenerate, _ = target_degeneracy_masks(target, self.silence_rms_threshold)
        valid = ~degenerate
        valid_embeddings = embeddings[valid]
        if valid_embeddings.shape[0] < 2:
            return embeddings.sum() * 0.0
        # Population variance (correction=0); no batch identity assumptions.
        std = (valid_embeddings.var(dim=0, unbiased=False) + 1e-8).sqrt()
        return F.relu(self.variance_floor - std).mean()

    def forward(self, padded_audio, video, target):
        output = self.model(padded_audio, video)
        positive = self.positive_loss(output[:, :SEGMENT_SAMPLES], target) if self.positive_contrastive else None
        variance = self.variance_penalty(target) if self.positive_contrastive and self.variance_weight > 0 else None
        return output, positive, variance


def reconstruction_components(output, target, mixture, silence_rms_threshold=0.0,
                              silence_loss_weight=1.0, speech_ratio_floor=0.0):
    """Per-sample hybrid: original negative SI-SDR for speech, normalized MSE for degenerate targets.

    The scale-free target MSE trains zero and near-constant targets, where SI-SDR
    is undefined. Threshold is centered target RMS; exact/near-zero RMS is also
    exposed separately for logging. Normalized MSE denominator is floored.
    """
    if output.ndim != 2 or output.shape != target.shape or output.shape != mixture.shape or output.numel() == 0:
        raise ValueError("output, target and mixture must have identical nonempty [batch,time] shape")
    degenerate, silent = target_degeneracy_masks(target, silence_rms_threshold)
    sisdr = pairwise_neg_sisdr(output.unsqueeze(1), target.unsqueeze(1)).reshape(-1)
    normalized_mse = (output - target).square().mean(dim=1) / mixture.square().mean(dim=1).clamp_min(1e-8)
    speech = sisdr
    if speech_ratio_floor:
        # Apply the floor in power-ratio space. logaddexp avoids exponentiating
        # a large ratio and smoothly reduces gradients near zero projection.
        db_per_neper = 10.0 / math.log(10.0)
        speech = -db_per_neper * torch.logaddexp(
            -sisdr / db_per_neper, sisdr.new_tensor(math.log(speech_ratio_floor)))
    combined = torch.where(degenerate, float(silence_loss_weight) * normalized_mse, speech)
    return combined, sisdr, normalized_mse, degenerate, silent


def neg_si_snr(output, target):
    if output.shape != target.shape:
        raise ValueError("output and target must have identical shape")
    return pairwise_neg_sisdr(output.unsqueeze(1), target.unsqueeze(1)).mean()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collision_target_evidence(dataset, item):
    # Index offsets select only the matching target and metadata tar members.
    path = dataset.shards[item["shard"]]["path"]
    fields = item["fields"]
    sample_id = item["id"]
    with open(path, "rb") as stream:
        offset, size = fields["meta"]
        stream.seek(offset)
        raw_meta = stream.read(size)
        if len(raw_meta) != size:
            raise ValueError(f"truncated metadata for {sample_id}")
        metadata = json.loads(raw_meta)
        source = metadata.get("target") if isinstance(metadata, dict) else None
        if not isinstance(source, str) or not source:
            raise ValueError(f"missing metadata target path for {sample_id}")
        offset, size = fields["target"]
        stream.seek(offset)
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError(f"truncated target for {sample_id}")
            digest.update(block)
            remaining -= len(block)
    return source, digest.hexdigest()


def validate_tar_splits(train, validation):
    if train is not None and train.split != "tr":
        raise ValueError("training tar index must have split=tr")
    if validation is not None and validation.split != "cv":
        raise ValueError("validation tar index must have split=cv")
    result = {"matching_bare_ids": 0, "same_target_source": 0, "same_target_wav": 0,
              "scope": "same_bare_id_only"}
    if train is not None and validation is not None:
        tr = {item["id"]: item for item in train.samples}
        cv = {item["id"]: item for item in validation.samples}
        overlap = sorted(tr.keys() & cv.keys())
        result["matching_bare_ids"] = len(overlap)
        for key in overlap:
            left_source, left_digest = _collision_target_evidence(train, tr[key])
            right_source, right_digest = _collision_target_evidence(validation, cv[key])
            result["same_target_source"] += int(left_source == right_source)
            result["same_target_wav"] += int(left_digest == right_digest)
        if result["same_target_source"] or result["same_target_wav"]:
            raise ValueError(f"train/validation same-ID target conflict: {result}")
    return result

def index_fingerprint(tar_path, directory):
    if tar_path:
        return {"kind": "tar", "path": str(Path(tar_path).resolve()), "sha256": sha256(tar_path)}
    if directory:
        root = Path(directory).resolve()
        return {"kind": "directory", "path": str(root),
                "mix_sha256": sha256(root / "mix.json"), "target_sha256": sha256(root / "target.json")}
    return None


def validate_resume_protocol(saved, current):
    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint lacks strict protocol metadata; warm-start is not supported")
    mismatches = [key for key, value in current.items() if saved.get(key) != value]
    if mismatches:
        raise ValueError("resume protocol mismatch: " + ", ".join(sorted(mismatches)))


def advance_validation(scheduler, loss, best, stale, partial, min_delta=0.0):
    # Step-limited partial epochs report validation but do not affect epoch policy.
    if partial:
        return best, stale, False
    scheduler.step(loss)
    improved = loss < best - min_delta
    return loss if improved else best, 0 if improved else stale + 1, improved


def advance_validation_dual(scheduler, loss, numeric_best, significant_best, stale, partial, min_delta=0.0, policy_suppressed=False):
    """Track actual lowest separately from significant-improvement early stopping."""
    if partial:
        return numeric_best, significant_best, stale, False
    numeric_improved = loss < numeric_best
    if policy_suppressed:
        return min(numeric_best, loss), significant_best, stale, numeric_improved
    scheduler.step(loss)
    significant_improved = loss < significant_best - min_delta
    return (min(numeric_best, loss), loss if significant_improved else significant_best,
            0 if significant_improved else stale + 1, numeric_improved)



def validation_tracking_from_checkpoint(checkpoint):
    """Restore and validate independent numeric-best and patience reference."""
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in
            ("best_validation", "significant_best_validation", "stale_validations")):
        raise ValueError("resume checkpoint lacks dual validation tracking state")
    numeric = checkpoint["best_validation"]
    significant = checkpoint["significant_best_validation"]
    stale = checkpoint["stale_validations"]
    if (isinstance(numeric, bool) or not isinstance(numeric, (int, float))
            or isinstance(significant, bool) or not isinstance(significant, (int, float))
            or math.isnan(numeric) or math.isnan(significant)
            or numeric == -math.inf or significant == -math.inf
            or numeric > significant or type(stale) is not int or stale < 0
            or (significant == math.inf and stale != 0)):
        raise ValueError("resume checkpoint has invalid dual validation tracking state")
    return float(numeric), float(significant), stale

def warmup_learning_rate(optimizer, target_lr, update, warmup_steps):
    """Set the next update rate; the saved update counter makes resume exact."""
    if warmup_steps and update < warmup_steps:
        rate = target_lr * (update + 1) / warmup_steps
        for group in optimizer.param_groups:
            group["lr"] = rate


def atomic_checkpoint(path, payload):
    # Publish without replacing an existing file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.link(temp_name, path)
    finally:
        os.unlink(temp_name)



def load_warm_start(path, model, training_model, visual_sha, *, random_visual, require_speaker_pool):
    """Strict model initialization; explicitly does not load optimizer, scheduler, or steps."""
    before = sha256(path)
    source = read_checkpoint(path, "cpu")
    source_config = source["model_config"]
    if source_config.get("channels") != model.channels or tuple(source_config.get("dilations", ())) != model.dilations:
        raise ValueError("warm-start model channels or dilations differ")
    if source.get("visual_initialization_sha256") != visual_sha:
        raise ValueError("warm-start visual initialization differs from checkpoint")
    if bool(source.get("visual_initialization") == "random_frozen") != bool(random_visual):
        raise ValueError("warm-start visual initialization mode differs from checkpoint")
    source_state = dict(source["state_dict"])
    source_mode = source_config.get("synthesis_mode", "nonnegative")
    destination_mode = model.synthesis_mode
    destination_state = model.state_dict()
    if source_mode != destination_mode:
        if (source_mode, destination_mode) != ("nonnegative", "signed_residual"):
            raise ValueError("unsupported synthesis warm-start migration")
        for name in ("synthesis_residual.weight", "synthesis_residual.bias"):
            if name in source_state:
                raise ValueError("legacy warm-start contains a signed residual tensor")
            source_state[name] = torch.zeros_like(destination_state[name])
    source_has_history = bool(source_config.get("attention_history_chunks", 0))
    if source_has_history and not model.attention_history_chunks:
        raise ValueError("cannot discard history attention in warm-start")
    if model.attention_history_chunks and not source_has_history:
        if "history_attention_gate" in source_state:
            raise ValueError("legacy warm-start contains a history attention gate")
        source_state["history_attention_gate"] = torch.zeros_like(destination_state["history_attention_gate"])
    source_subband = source_config.get("subband_refiner", False)
    if source_subband and not model.subband_refiner:
        raise ValueError("cannot discard the subband refiner in warm-start")
    if model.subband_refiner and not source_subband:
        subband = {name: value for name, value in destination_state.items() if name.startswith("subband.")}
        if not subband or source_state.keys() & subband.keys():
            raise ValueError("invalid legacy warm-start subband state")
        source_state.update(subband)
    source_state["_streaming_format_version"] = destination_state["_streaming_format_version"].detach().cpu().clone()
    model.load_state_dict(source_state, strict=True)
    pool_sha = None
    if require_speaker_pool:
        pool = source.get("speaker_pool_state")
        if pool is None:
            raise ValueError("warm-start source lacks required speaker pool")
        training_model.speaker_pool.load_state_dict(pool, strict=True)
        digest = hashlib.sha256()
        for name, value in sorted(pool.items()):
            if not isinstance(value, torch.Tensor):
                raise ValueError("warm-start speaker pool contains non-tensor value")
            digest.update(name.encode("utf8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        pool_sha = digest.hexdigest()
    if sha256(path) != before:
        raise RuntimeError("warm-start checkpoint changed during loading")
    return {"path": str(Path(path).resolve()), "sha256": before,
            "source_format": source["format"], "source_model_config": source_config,
            "source_steps": source.get("steps"), "speaker_pool_loaded": require_speaker_pool,
            "speaker_pool_sha256": pool_sha, "optimizer_scheduler_and_steps_reset": True}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", help="YAML configuration; explicit CLI options override it")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--index-dir")
    data.add_argument("--tar-index", help="prebuilt RealSSA tar index JSON")
    data.add_argument("--synthetic", action="store_true")
    visual = parser.add_mutually_exclusive_group(required=True)
    visual.add_argument("--visual-init", help="strict FRCNN model_state_dict file; source recorded, then frozen")
    visual.add_argument("--random-visual", action="store_true", help="explicitly freeze random visual backbone for interface testing only")
    objective = parser.add_mutually_exclusive_group(required=True)
    objective.add_argument("--reconstruction-only", action="store_true", help="only reconstruction loss")
    objective.add_argument("--positive-contrastive", action="store_true", help="NASS positive term only; no near-field negative waveforms")
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--exclude-degenerate-contrastive", action="store_true",
                        help="exclude zero/constant targets from positive speaker loss")
    parser.add_argument("--variance-weight", type=float, default=0.0)
    parser.add_argument("--variance-floor", type=float, default=0.05)
    parser.add_argument("--silence-rms-threshold", type=float, default=0.0)
    parser.add_argument("--silence-loss-weight", type=float, default=1.0)
    parser.add_argument("--visual-mode", choices=("causal", "chunk_centered"), default="causal")
    parser.add_argument("--synthesis-mode", choices=("nonnegative", "signed_residual"), default="nonnegative")
    parser.add_argument("--subband-refiner", action="store_true")
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--speech-ratio-floor", type=float, default=0.0,
                        help="training-only soft floor added to the linear SI-SDR power ratio")
    parser.add_argument("--attention-history-chunks", type=int, default=0)
    parser.add_argument("--teacher-checkpoint", help="released Falcon checkpoint for training-only distillation")
    parser.add_argument("--teacher-cache", help="complete verified teacher cache for the fixed training view")
    parser.add_argument("--fixed-crop", action="store_true", help="reuse the deterministic epoch-zero crop and flip")
    parser.add_argument("--teacher-microbatch", type=int, default=16)
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--teacher-min-improvement-db", type=float, default=0.0)
    parser.add_argument("--teacher-min-sisdr-db", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", help="new directory for immutable per-epoch checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="strictly resume model, optimizer and scheduler")
    initialization.add_argument("--warm-start", help="initialize model and optional speaker pool only; optimizer and steps restart")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--valid-index-dir")
    validation.add_argument("--valid-tar-index")
    parser.add_argument("--validation-limit", type=int, default=0, help="0 evaluates the full validation index; positive values limit to its first N entries")
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--model-channels", type=int, default=320)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4, help="per-rank batch")
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0, help="optimizer updates; 0 means no limit")
    parser.add_argument("--lr", type=float, default=None, help="new runs default to 0.001; resumes retain saved optimizer rate unless overridden")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    try:
        tokens, config_path, config_sha256 = yaml_arguments(argv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args = parser.parse_args(tokens)
    args.config = config_path
    args.config_sha256 = config_sha256
    if min(args.batch_size, args.accum, args.epochs, args.save_every, args.validation_batch_size, args.model_channels, args.early_stop_patience) < 1 or min(args.max_steps, args.workers, args.validation_limit, args.warmup_steps, args.lr_patience) < 0:
        parser.error("invalid batch/accum/epochs/steps/workers")
    if (args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0)) or not np.isfinite(args.weight_decay) or args.weight_decay < 0 or not np.isfinite(args.contrastive_weight) or args.contrastive_weight < 0:
        parser.error("invalid optimizer parameters")
    if not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("min-delta must be finite and nonnegative")
    for name in ("variance_weight", "variance_floor", "silence_rms_threshold", "silence_loss_weight"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.variance_weight > 0 and not args.positive_contrastive:
        parser.error("variance regularizer requires positive-contrastive")
    try:
        checkpoint_format_for_visual_mode(args.visual_mode, args.synthesis_mode,
                                          args.attention_history_chunks, args.subband_refiner)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.teacher_microbatch < 1
            or not all(np.isfinite(getattr(args, name)) for name in
                       ("distillation_weight", "teacher_min_improvement_db", "teacher_min_sisdr_db"))
            or args.distillation_weight < 0):
        parser.error("invalid teacher microbatch, weight, or quality threshold")
    if bool(args.teacher_checkpoint) != (args.distillation_weight > 0):
        parser.error("teacher checkpoint and positive distillation weight must be supplied together")
    if (args.teacher_cache or args.fixed_crop) and not args.tar_index:
        parser.error("teacher cache and fixed crop require a training TAR index")
    if args.teacher_cache and not args.teacher_checkpoint:
        parser.error("teacher cache requires the teacher checkpoint for identity verification")
    if args.teacher_cache and not args.fixed_crop:
        parser.error("teacher cache requires --fixed-crop so cached targets and student views match")
    if not np.isfinite(args.spectral_loss_weight) or args.spectral_loss_weight < 0:
        parser.error("spectral loss weight must be finite and nonnegative")
    if not 0 <= args.speech_ratio_floor < 1:
        parser.error("speech ratio floor must be in [0, 1)")
    if args.teacher_checkpoint and args.synthetic:
        parser.error("distillation requires real audiovisual data")
    if not 0 <= args.seed <= 2**32 - 2:
        parser.error("seed must be between 0 and 2**32-2 for two ranks")
    return args


def main():
    args = parse_args()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise ValueError("launch with torchrun; RANK/WORLD_SIZE are required")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise ValueError("this trial requires exactly two ranks")
    if Path(args.output).exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=args.backend)
    try:
        # Check all ranks received identical effective settings before output.
        effective_arguments = vars(args).copy()
        arg_signature = hashlib.sha256(json.dumps(effective_arguments, sort_keys=True).encode()).hexdigest()
        signatures = [None] * world
        dist.all_gather_object(signatures, arg_signature)
        if any(signature != arg_signature for signature in signatures):
            raise ValueError("training configuration differs across ranks")
        if args.checkpoint_dir and rank == 0:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=False)
        dist.barrier()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        if args.synthetic:
            dataset = SyntheticTwoSeconds()
        elif args.tar_index:
            from tools.streaming.realssa_tar import RealSSATarDataset
            dataset = RealSSATarDataset(args.tar_index, seed=args.seed, crop="random", augment=True)
        else:
            dataset = TwoSecondDataset(args.index_dir)
        cache_sha = None
        cache_code_sha = None
        if args.fixed_crop:
            from tools.streaming.cache_teacher import CachedTeacherDataset, FixedCropDataset
            cache_code_sha = sha256(Path(__file__).with_name("cache_teacher.py"))
            if args.teacher_cache:
                dataset = CachedTeacherDataset(dataset, args.teacher_cache, index_path=args.tar_index,
                                               teacher_checkpoint=args.teacher_checkpoint)
                cache_sha = dataset.teacher_cache_sha256
            else:
                dataset = FixedCropDataset(dataset)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, drop_last=True,
                            pin_memory=device.type == "cuda", prefetch_factor=1 if args.workers else None)
        if len(loader) < args.accum:
            raise ValueError("dataset too small for batch, ranks and accumulation")
        model = FalconStreaming(channels=args.model_channels, visual_mode=args.visual_mode,
                                synthesis_mode=args.synthesis_mode,
                                attention_history_chunks=args.attention_history_chunks,
                                subband_refiner=args.subband_refiner)
        model_config = {"channels": model.channels, "dilations": list(model.dilations)}
        if args.visual_mode != "causal":
            model_config["visual_mode"] = args.visual_mode
        if args.synthesis_mode != "nonnegative":
            model_config["synthesis_mode"] = args.synthesis_mode
        if args.attention_history_chunks:
            model_config["attention_history_chunks"] = args.attention_history_chunks
        if args.subband_refiner:
            model_config["subband_refiner"] = True
        model_code_sha = sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconStreaming.py")
        trainer_code_sha = sha256(__file__)
        data_code_sha = sha256(Path(__file__).with_name("realssa_tar.py")) if (args.tar_index or args.valid_tar_index) else sha256(Path(__file__).with_name("common.py"))
        train_index_sha = sha256(args.tar_index) if args.tar_index else None
        validation_index_sha = sha256(args.valid_tar_index) if args.valid_tar_index else None
        visual_sha = None
        visual_meta = None
        if args.visual_init:
            payload = torch.load(args.visual_init, map_location="cpu", weights_only=True)
            visual_meta = payload.get("meta") if isinstance(payload, dict) else None
            del payload
            model.video_model.init_from(args.visual_init)
            visual_sha = sha256(args.visual_init)
        for parameter in model.video_model.parameters():
            parameter.requires_grad_(False)
        model = model.to(device)
        training_model = TrainingWrapper(model, args.positive_contrastive, args.variance_weight,
                                         args.variance_floor, args.silence_rms_threshold,
                                         args.exclude_degenerate_contrastive).to(device)
        objective_name = "negative_si_snr_plus_positive_contrastive_no_negatives" if args.positive_contrastive else "negative_si_snr_reconstruction_only"
        if args.speech_ratio_floor:
            objective_name += "_with_soft_sisdr_floor"
        spectral_loss = None
        spectral_protocol = None
        if args.spectral_loss_weight:
            from tools.streaming.spectral_loss import MultiResolutionSpectralLoss

            spectral_loss = MultiResolutionSpectralLoss().to(device)
            spectral_protocol = {"weight": args.spectral_loss_weight,
                                 "source_sha256": sha256(Path(__file__).with_name("spectral_loss.py"))}
            objective_name += "_with_multires_spectral"
        teacher = None
        distillation_protocol = None
        if args.teacher_checkpoint:
            from tools.streaming.cache_teacher import teacher_source_sha256
            from tools.streaming.distillation import load_teacher, predict_teacher, distillation_components
            teacher_sha = sha256(args.teacher_checkpoint)
            if not args.teacher_cache:
                teacher = load_teacher(args.teacher_checkpoint, device)
            if sha256(args.teacher_checkpoint) != teacher_sha:
                raise RuntimeError("teacher checkpoint changed while loading")
            distillation_protocol = {
                "checkpoint_sha256": teacher_sha,
                "source_sha256": teacher_source_sha256(),
                "weight": args.distillation_weight,
                "microbatch": args.teacher_microbatch if not args.teacher_cache else None,
                "cache_manifest_sha256": cache_sha,
                "min_improvement_db": args.teacher_min_improvement_db,
                "min_sisdr_db": args.teacher_min_sisdr_db,
            }
            objective_name += "_with_released_teacher"
        checkpoint = read_checkpoint(args.resume, device) if args.resume else None
        if checkpoint is not None:
            if checkpoint.get("training_objective") != objective_name:
                raise ValueError("resume checkpoint lacks matching training objective")
            if checkpoint.get("visual_initialization_sha256") != visual_sha:
                raise ValueError("resume visual initialization differs from checkpoint")
            if bool(checkpoint.get("visual_initialization") == "random_frozen") != bool(args.random_visual):
                raise ValueError("resume visual initialization mode differs from checkpoint")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if args.positive_contrastive:
                training_model.speaker_pool.load_state_dict(checkpoint["speaker_pool_state"], strict=True)
        warm_start_metadata = checkpoint.get("warm_start_metadata") if checkpoint else None
        if args.warm_start:
            warm_start_metadata = load_warm_start(
                args.warm_start, model, training_model, visual_sha,
                random_visual=args.random_visual, require_speaker_pool=args.positive_contrastive)
        train_fingerprint = {"kind": "synthetic"} if args.synthetic else index_fingerprint(args.tar_index, args.index_dir)
        valid_fingerprint = index_fingerprint(args.valid_tar_index, args.valid_index_dir)
        resume_protocol = {"train": train_fingerprint, "validation": valid_fingerprint,
                           "world_size": world, "batch_per_rank": args.batch_size, "accumulation": args.accum,
                           "seed": args.seed, "workers": args.workers, "validation_limit": args.validation_limit, "weight_decay": args.weight_decay, "visual_sha256": visual_sha,
                           "model_code_sha256": model_code_sha, "trainer_code_sha256": trainer_code_sha,
                           "config_parser_sha256": sha256(Path(__file__).with_name("training_config.py")),
                           "data_code_sha256": data_code_sha, "torch_version": str(torch.__version__), "objective": objective_name,
                           "contrastive_weight": args.contrastive_weight if args.positive_contrastive else 0.0,
                           "optimizer": "AdamW", "segment_samples": SEGMENT_SAMPLES,
                           "model_channels": args.model_channels, "warmup_steps": args.warmup_steps,
                           "lr_patience": args.lr_patience, "early_stop_patience": args.early_stop_patience,
                           "min_delta": args.min_delta, "validation_batch_size": args.validation_batch_size,
                           "visual_mode": args.visual_mode, "variance_weight": args.variance_weight,
                           "variance_scope": "rank_local",
                           "exclude_degenerate_contrastive": args.exclude_degenerate_contrastive,
                           "contrastive_reduction": "global_valid_mean_v1",
                           "variance_floor": args.variance_floor,
                           "silence_rms_threshold": args.silence_rms_threshold,
                           "silence_loss_weight": args.silence_loss_weight,
                           "speech_ratio_floor": args.speech_ratio_floor,
                           "reconstruction_protocol": "centered_rms_sisdr_or_mixture_normalized_mse_v1",
                           "validation_tracking_version": 2,
                           "synthesis_mode": args.synthesis_mode,
                           "attention_history_chunks": args.attention_history_chunks,
                           "subband_refiner": args.subband_refiner,
                           "spectral": spectral_protocol,
                           "distillation": distillation_protocol,
                           "fixed_crop": args.fixed_crop,
                           "cache_code_sha256": cache_code_sha}
        if checkpoint is not None:
            validate_resume_protocol(checkpoint.get("resume_protocol"), resume_protocol)
            if args.lr is not None and args.lr != checkpoint.get("target_learning_rate"):
                raise ValueError("resume protocol mismatch: target_learning_rate")
        ddp = DistributedDataParallel(training_model, device_ids=[local_rank] if device.type == "cuda" else None)
        target_lr = args.lr if args.lr is not None else (checkpoint.get("target_learning_rate", .001) if checkpoint else .001)
        optimizer = torch.optim.AdamW((p for p in training_model.parameters() if p.requires_grad), lr=target_lr, weight_decay=args.weight_decay)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.valid_tar_index:
            from tools.streaming.realssa_tar import RealSSATarDataset
            valid = RealSSATarDataset(args.valid_tar_index, seed=args.seed, crop="center", augment=False)
        else:
            valid = TwoSecondDataset(args.valid_index_dir, validation=True) if args.valid_index_dir else None
        split_identity_audit = validate_tar_splits(dataset if args.tar_index else None, valid if args.valid_tar_index else None)
        if rank == 0:
            print(f"split_identity_audit={split_identity_audit}", flush=True)
        if args.index_dir and args.valid_index_dir and Path(args.index_dir).resolve() == Path(args.valid_index_dir).resolve():
            raise ValueError("training and validation index directories must differ")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=args.lr_patience, threshold=args.min_delta, threshold_mode="abs", min_lr=1e-6) if valid else None
        if checkpoint is not None and scheduler is not None:
            if checkpoint.get("scheduler") is None:
                raise ValueError("resume checkpoint lacks validation scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler"])
        updates = int(checkpoint.get("steps", 0)) if checkpoint else 0
        start_updates = updates
        start_epoch = int(checkpoint.get("next_epoch", 0)) if checkpoint else 0
        if checkpoint:
            best, significant_best, stale = validation_tracking_from_checkpoint(checkpoint)
            if scheduler is not None and scheduler.best != significant_best:
                raise ValueError("resume scheduler and significant-best validation state disagree")
        else:
            best, significant_best, stale = float("inf"), float("inf"), 0
        end_reason = "epochs"
        best_checkpoint = checkpoint.get("best_checkpoint") if checkpoint else None
        validation_mean = None
        def checkpoint_payload(current_epoch, partial):
            return {
                "format": checkpoint_format_for_visual_mode(
                    args.visual_mode, args.synthesis_mode, args.attention_history_chunks, args.subband_refiner),
                "state_dict": model.state_dict(), "model_config": model_config,
                "distillation": distillation_protocol, "spectral": spectral_protocol,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                "resume_protocol": resume_protocol, "split_identity_audit": split_identity_audit,
                "configuration_path": args.config, "configuration_sha256": args.config_sha256,
                "effective_arguments": effective_arguments,
                "steps": updates, "next_epoch": current_epoch if partial else current_epoch + 1,
                "resume_restarts_epoch": partial, "best_validation": best,
                "significant_best_validation": significant_best, "stale_validations": stale,
                "warm_start_metadata": warm_start_metadata,
                "validation_loss": validation_mean, "validation_limit": args.validation_limit,
                "validation_samples": (min(len(valid), args.validation_limit) if args.validation_limit else len(valid)) if valid is not None else 0,
                "validation_total": len(valid) if valid is not None else 0, "best_checkpoint": best_checkpoint,
                "training_status": "synthetic_smoke" if args.synthetic else "real_data_optimization",
                "training_objective": objective_name,
                "speaker_pool_state": training_model.speaker_pool.state_dict() if args.positive_contrastive else None,
                "positive_contrastive_weight": args.contrastive_weight if args.positive_contrastive else 0.0,
                "positive_contrastive_definition": "output_and_target_valid_2s_reencoded_by_current_audio_proj" if args.positive_contrastive else None,
                "variance_weight": args.variance_weight, "variance_floor": args.variance_floor,
                "variance_scope": "rank_local",
                "exclude_degenerate_contrastive": args.exclude_degenerate_contrastive,
                "contrastive_reduction": "global_valid_mean_v1",
                "silence_rms_threshold": args.silence_rms_threshold, "silence_loss_weight": args.silence_loss_weight,
                "speech_ratio_floor": args.speech_ratio_floor,
                "speaker_contrastive_loss": args.positive_contrastive, "near_negatives": False,
                "visual_initialization": "random_frozen" if args.random_visual else str(Path(args.visual_init).resolve()),
                "visual_initialization_sha256": visual_sha, "visual_initialization_metadata": visual_meta, "visual_frozen": True,
                "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
                "train_index": "synthetic" if args.synthetic else str(Path(args.tar_index or args.index_dir).resolve()),
                "validation_index": str(Path(args.valid_tar_index or args.valid_index_dir).resolve()) if (args.valid_tar_index or args.valid_index_dir) else None,
                "train_tar_index_sha256": train_index_sha, "validation_tar_index_sha256": validation_index_sha,
                "model_code_sha256": model_code_sha, "torch_version": str(torch.__version__),
                "segment_samples": SEGMENT_SAMPLES, "segment_video_frames": SEGMENT_FRAMES,
                "world_size": world, "batch_per_rank": args.batch_size, "gradient_accumulation": args.accum,
                "learning_rate": optimizer.param_groups[0]["lr"], "target_learning_rate": target_lr, "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "seed": args.seed, "end_reason": end_reason,
            }
        if rank == 0:
            visual_label = "random-frozen" if args.random_visual else args.visual_init
            print(f"config={args.config} config_sha256={args.config_sha256} effective_arguments={json.dumps(effective_arguments, sort_keys=True)}", flush=True)
            print(f"trial: backend={args.backend} world={world} batch_per_rank={args.batch_size} accumulation={args.accum} global_batch={world*args.batch_size*args.accum} segment_samples={SEGMENT_SAMPLES} visual={visual_label} objective={objective_name} near_negatives=NOT_USED", flush=True)
        training_model.train()
        for epoch in range(start_epoch, start_epoch + args.epochs):
            epoch_started = time.perf_counter()
            sampler.set_epoch(epoch)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            # Process full accumulation groups only, so each update has the same scale.
            usable_batches = len(loader) // args.accum * args.accum
            for index, batch in enumerate(loader):
                if index >= usable_batches:
                    break
                mixture, target, video = (item.to(device) for item in batch[:3])
                if args.teacher_cache:
                    teacher_output = batch[3].to(device)
                elif teacher is not None:
                    teacher_output = predict_teacher(teacher, mixture, video, microbatch_size=args.teacher_microbatch)
                else:
                    teacher_output = None
                padded, frames = pad_two_seconds(mixture, video)
                output, contrastive, variance = ddp(padded, frames, target)
                output = output[:, :SEGMENT_SAMPLES]
                combined, _, _, degenerate, silent = reconstruction_components(
                    output, target, mixture, args.silence_rms_threshold, args.silence_loss_weight,
                    speech_ratio_floor=args.speech_ratio_floor)
                loss = combined.mean()
                spectral_value = None
                if spectral_loss is not None:
                    spectral_value = spectral_loss(output, target, mixture, ~degenerate).mean()
                    loss = loss + args.spectral_loss_weight * spectral_value
                distill_metrics = None
                if teacher_output is not None:
                    auxiliary, selected, teacher_gain = distillation_components(
                        output, teacher_output, target, mixture,
                        min_improvement_db=args.teacher_min_improvement_db,
                        min_teacher_sisdr_db=args.teacher_min_sisdr_db,
                        silence_rms_threshold=args.silence_rms_threshold)
                    loss = loss + args.distillation_weight * auxiliary.mean()
                    distill_metrics = torch.stack((auxiliary.detach().sum(), selected.float().sum(),
                                                   torch.where(selected, teacher_gain, 0).sum()))
                if contrastive is not None:
                    loss = loss + args.contrastive_weight * contrastive
                if variance is not None:
                    loss = loss + args.variance_weight * variance
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                (loss / args.accum).backward()
                if (index + 1) % args.accum == 0:
                    norm = torch.nn.utils.clip_grad_norm_(training_model.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("non-finite gradient")
                    # Every rank participates; contrastive is locally scaled so
                    # its rank average equals the global valid-example mean.
                    contrastive_mean, variance_rank_mean = reduce_training_loss_metrics(contrastive, variance, device)
                    if distill_metrics is not None:
                        dist.all_reduce(distill_metrics, op=dist.ReduceOp.SUM)
                        summed_loss, selected_count, summed_gain = distill_metrics.tolist()
                        teacher_log = (f" teacher_loss={summed_loss / (world * args.batch_size):.6g}"
                                       f" teacher_selected={int(selected_count)}/{world * args.batch_size}"
                                       f" teacher_gain_db={summed_gain / max(1, selected_count):.5g}")
                    else:
                        teacher_log = ""
                    spectral_log = ""
                    if spectral_value is not None:
                        global_spectral = spectral_value.detach()
                        dist.all_reduce(global_spectral, op=dist.ReduceOp.SUM)
                        spectral_log = f" spectral_loss_last_microbatch={global_spectral.item() / world:.6g}"
                    warmup_learning_rate(optimizer, target_lr, updates, args.warmup_steps)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    if rank == 0:
                        print(f"epoch={epoch} step={updates} loss_last_microbatch={loss.item():.6f} grad_norm={norm.item():.6f} lr={optimizer.param_groups[0]['lr']:.8g} rank0_speech={int((~degenerate).sum())} rank0_silent={int(silent.sum())} rank0_constant={int((degenerate & ~silent).sum())} contrastive_global_last_microbatch={contrastive_mean} variance_rank_mean_last_microbatch={variance_rank_mean}{teacher_log}{spectral_log}", flush=True)
                    if args.max_steps and updates - start_updates >= args.max_steps:
                        end_reason = "max_steps"
                        break
            partial = end_reason == "max_steps" and (index + 1) < usable_batches
            improved = False
            validation_mean = None
            if valid is not None:
                # Each sample is evaluated exactly once across ranks, without sampler padding.
                training_model.eval()
                # combined sum/count, valid SI-SDR sum/count, silent NMSE sum/count,
                # and non-silent but centered-degenerate count.
                total = torch.zeros(7, device=device, dtype=torch.float64)
                with torch.inference_mode():
                    validation_count = min(len(valid), args.validation_limit) if args.validation_limit else len(valid)
                    validation_set = Subset(valid, range(rank, validation_count, world))
                    validation_loader = DataLoader(validation_set, batch_size=args.validation_batch_size,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        prefetch_factor=1 if args.workers else None)
                    for mixture, target, video in validation_loader:
                        mixture, target, video = (x.to(device, non_blocking=True) for x in (mixture, target, video))
                        padded, frames = pad_two_seconds(mixture, video)
                        prediction = model(padded, frames)[:, :SEGMENT_SAMPLES]
                        combined, sisdr, nmse, degenerate, silent = reconstruction_components(
                            prediction, target, mixture, args.silence_rms_threshold, args.silence_loss_weight)
                        if not torch.isfinite(combined).all():
                            raise RuntimeError("non-finite validation loss")
                        total += torch.stack((combined.sum(), combined.new_tensor(combined.numel()),
                                              sisdr[~degenerate].sum(), (~degenerate).sum(),
                                              nmse[silent].sum(), silent.sum(),
                                              (degenerate & ~silent).sum())).to(dtype=torch.float64)
                dist.all_reduce(total)
                mean = (total[0] / total[1]).item()
                validation_mean = mean
                best, significant_best, stale, improved = advance_validation_dual(
                    scheduler, mean, best, significant_best, stale, partial, args.min_delta,
                    policy_suppressed=updates < args.warmup_steps)
                if rank == 0:
                    learning_rate = optimizer.param_groups[0]["lr"]
                    speech_loss = (total[2] / total[3]).item() if total[3].item() else float("nan")
                    silent_nmse = (total[4] / total[5]).item() if total[5].item() else float("nan")
                    print(f"epoch={epoch} validation_samples={int(total[1])} validation_total={len(valid)} validation_limit={args.validation_limit} loss={mean:.6f} lr={learning_rate:.8g} partial={partial} speech_neg_sisdr={speech_loss:.6f} speech_count={int(total[3])} silent_nmse={silent_nmse:.6f} silent_count={int(total[5])} constant_count={int(total[6])}", flush=True)
                training_model.train()
                if stale >= args.early_stop_patience:
                    end_reason = "early_stopping"
            should_save = bool(args.checkpoint_dir and ((epoch + 1) % args.save_every == 0 or improved or end_reason in ("max_steps", "early_stopping")))
            if should_save:
                checkpoint_path = Path(args.checkpoint_dir) / f"epoch-{epoch:04d}-step-{updates:08d}.pt"
                if improved:
                    best_checkpoint = str(checkpoint_path.resolve())
                if rank == 0:
                    atomic_checkpoint(checkpoint_path, checkpoint_payload(epoch, partial))
                    print(f"checkpoint={checkpoint_path} best={best_checkpoint}", flush=True)
                dist.barrier()
            if rank == 0:
                print(f"epoch={epoch} elapsed_seconds={time.perf_counter() - epoch_started:.3f} updates={updates}", flush=True)
            if end_reason in ("max_steps", "early_stopping"):
                break
        dist.barrier()
        if rank == 0:
            atomic_checkpoint(args.output, checkpoint_payload(epoch, partial))
            print(f"saved {args.output}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
