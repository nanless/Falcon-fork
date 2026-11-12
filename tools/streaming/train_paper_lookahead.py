"""DDP training for the experimental paper-like 120ms/80ms model.

Example smoke (use a fresh run name, never an existing experiment directory):
  torchrun --standalone --nproc_per_node=2 -m tools.streaming.train_paper_lookahead \
    --config configs/RealSSAPaperLookahead.yml --run-name smoke --batch-size 2 \
    --max-steps 1 --validation-limit 4
A completed-epoch checkpoint can be resumed into a fresh run directory. Partial
smoke checkpoints are not exact iterator checkpoints and cannot be resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml

from look2hear.models.FalconPaperStreamPrototype import PaperStreamPrototype
from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel
from tools.streaming.lookahead_training import run_lookahead_sequence
from tools.streaming.robust_schedule import RobustWarmupPlateau, global_gradient_norm
from tools.streaming.realssa_lookahead import RealSSALookaheadDataset
from tools.streaming.scale_sdr_loss import scale_sdr_loss
from tools.streaming import stability_metrics
from tools.streaming.training_config import UniqueKeyLoader


class PaperLookaheadTrainingModel(nn.Module):
    """Use original frozen FRCNN and differentiable, non-detached core state."""
    def __init__(self, config):
        super().__init__()
        backbone = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu", pretrain=None)
        backbone.init_from(config["visual_weights"])
        self.visual = FalconLookaheadVisual(backbone)
        self.core = PaperStreamPrototype(width=config["width"], rnn_hidden=config["rnn_hidden"],
            heads=config["heads"], history_frames=config["history_frames"], mask_limit=config["mask_limit"])
        self.checkpoint_chunks = config["checkpoint_chunks"]
        self.activation_batch_chunk = config.get("activation_batch_chunk", 0)

    def _forward_sequence(self, audio, video, output_samples=None):
        def step(window_audio, window_video, state):
            core_state, video_state = (None, None) if state is None else state
            with torch.no_grad():
                features, video_state = self.visual(window_video, video_state)
                features = features.transpose(1, 2).contiguous()
            if self.training and self.checkpoint_chunks:
                output, core_state = checkpoint(self.core, window_audio, features, core_state,
                                                use_reentrant=False)
            else:
                output, core_state = self.core(window_audio, features, core_state)
            return output, (core_state, video_state)
        result, _ = run_lookahead_sequence(audio, video, step, output_samples=output_samples)
        return result

    def forward(self, audio, video, output_samples=None):
        # One DDP forward and one full-batch loss/backward/optimizer update.
        # Slice only to checkpoint the activation graph, never the optimizer or loss.
        chunk = self.activation_batch_chunk
        if not self.training or chunk <= 0 or audio.size(0) <= chunk:
            return self._forward_sequence(audio, video, output_samples)
        outputs = []
        for start in range(0, audio.size(0), chunk):
            a, v = audio[start:start+chunk], video[start:start+chunk]
            outputs.append(checkpoint(lambda x, y: self._forward_sequence(x, y, output_samples),
                                      a, v, use_reentrant=False))
        return torch.cat(outputs, dim=0)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def training_loss(output, target, mix):
    return scale_sdr_loss(output, target, mix)


def load_training_config(path):
    """Load a mapping while rejecting duplicate keys at every level."""
    try:
        with open(path, encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid training configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("training configuration must be a YAML mapping")
    return config


def validate_training_config(config):
    """Reject invalid config types and numeric values before initializing DDP."""
    required = {"train_index", "valid_index", "visual_weights", "experiment_root", "batch_size",
                "validation_batch_size", "workers", "lr", "warmup_steps", "weight_decay", "epochs",
                "lr_patience", "early_stop_patience", "min_delta", "seed", "width", "rnn_hidden",
                "heads", "history_frames", "checkpoint_chunks", "precision",
                "grad_norm_limit", "max_consecutive_bad_steps", "validation_severe_delta",
                "mask_limit", "activation_batch_chunk"}
    if set(config) != required:
        raise ValueError("config must have exactly the documented keys")
    strings = ("train_index", "valid_index", "visual_weights", "experiment_root", "precision")
    integers = ("batch_size", "validation_batch_size", "workers", "warmup_steps", "epochs",
                "lr_patience", "early_stop_patience", "seed", "width", "rnn_hidden", "heads",
                "history_frames", "max_consecutive_bad_steps", "activation_batch_chunk")
    numbers = ("lr", "weight_decay", "min_delta", "grad_norm_limit", "validation_severe_delta", "mask_limit")
    for key in strings:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"config {key} must be a nonempty string")
    for key in integers:
        if type(config[key]) is not int:
            raise ValueError(f"config {key} must be an integer")
    for key in numbers:
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"config {key} must be a finite number")
    if type(config["checkpoint_chunks"]) is not bool:
        raise ValueError("config checkpoint_chunks must be a boolean")
    if config["precision"] not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    positive_ints = ("batch_size", "validation_batch_size", "epochs", "early_stop_patience",
                     "width", "rnn_hidden", "heads", "max_consecutive_bad_steps", "activation_batch_chunk")
    nonnegative_ints = ("workers", "warmup_steps", "lr_patience", "history_frames")
    if any(config[k] < 1 for k in positive_ints) or any(config[k] < 0 for k in nonnegative_ints):
        raise ValueError("invalid integer configuration range")
    # rank 1 seeds numpy with seed + 1; numpy requires values below 2**32.
    if not 0 <= config["seed"] < 2**32 - 1:
        raise ValueError("seed must allow both rank seeds in [0, 2**32)")
    if config["width"] % config["heads"]:
        raise ValueError("width must be divisible by heads")
    if (config["lr"] <= 0 or config["weight_decay"] < 0 or config["min_delta"] < 0
            or config["grad_norm_limit"] <= 0 or config["validation_severe_delta"] <= 0
            or config["mask_limit"] <= 0):
        raise ValueError("invalid optimizer or numeric configuration range")


def all_finite(value, device):
    ok = torch.tensor(int(torch.isfinite(value).all().item()), device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0, help="bounded smoke only; 0 uses the whole split")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single new directory name")
    if args.validation_limit < 0 or args.train_limit < 0 or args.max_steps < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("invalid test override")
    config = load_training_config(args.config)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    try:
        validate_training_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("this recipe requires exactly two processes and two GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        run_dir = Path(config["experiment_root"]) / args.run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"] + rank)
        random.seed(config["seed"] + rank)
        train = RealSSALookaheadDataset(config["train_index"], seed=config["seed"], crop="random", augment=True)
        train_base = train
        if args.train_limit:
            train = Subset(train, range(min(len(train), args.train_limit)))
        valid = RealSSALookaheadDataset(config["valid_index"], seed=config["seed"], crop="center", augment=False)
        if train_base.split != "tr" or valid.split != "cv":
            raise ValueError("official training requires tr index and cv validation index")
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                     seed=config["seed"], drop_last=True)
        loader = DataLoader(train, sampler=sampler, batch_size=config["batch_size"], drop_last=True,
            num_workers=config["workers"], pin_memory=True,
            prefetch_factor=1 if config["workers"] else None)
        if not len(loader):
            raise ValueError("training split has no full per-rank batch")
        model = PaperLookaheadTrainingModel(config).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = RobustWarmupPlateau(config["lr"], config["warmup_steps"], factor=0.5,
            patience=config["lr_patience"], min_delta=config["min_delta"],
            severe_delta=config["validation_severe_delta"])
        identity = {"visual_sha256": sha256(config["visual_weights"]),
                    "train_index_sha256": sha256(config["train_index"]),
                    "valid_index_sha256": sha256(config["valid_index"]),
                    "config_sha256": sha256(args.config),
                    "trainer_sha256": sha256(__file__),
                    "core_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconPaperStreamPrototype.py"),
                    "visual_code_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconLookaheadVisual.py"),
                    "window_code_sha256": sha256(Path(__file__).with_name("lookahead_training.py")),
                    "data_code_sha256": sha256(Path(__file__).with_name("realssa_lookahead.py")),
                    "schedule_code_sha256": sha256(Path(__file__).with_name("robust_schedule.py")),
                    "config_loader_sha256": sha256(Path(__file__).with_name("training_config.py")),
                    "loss_code_sha256": sha256(Path(__file__).with_name("scale_sdr_loss.py")),
                    "metrics_code_sha256": sha256(Path(__file__).with_name("stability_metrics.py")),
                    "legacy_data_code_sha256": sha256(Path(__file__).with_name("realssa_tar.py")),
                    "common_code_sha256": sha256(Path(__file__).with_name("common.py")),
                    "mask_source_sha256": sha256(Path(__file__).with_name("train_ddp.py")),
                    "backbone_code_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/videomodels/frcnn_videomodel.py")}
        protocol = {"train_limit": args.train_limit, "validation_limit": args.validation_limit}
        start_epoch = updates = stale = 0
        consecutive_bad_steps = skipped_steps = 0
        best = float("inf")
        best_checkpoint = None
        if args.resume:
            saved = torch.load(args.resume, map_location=device, weights_only=False)
            if saved["identity"] != identity or saved["config"] != config or saved["world_size"] != world or saved["protocol"] != protocol:
                raise ValueError("resume identity, configuration or device count differs")
            if saved["partial"]:
                raise ValueError("cannot resume a partial-epoch checkpoint")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_epoch, updates, stale, best = saved["next_epoch"], saved["updates"], saved["stale"], saved["best"]
            consecutive_bad_steps, skipped_steps = saved["consecutive_bad_steps"], saved["skipped_steps"]
            best_checkpoint = saved["best_checkpoint"]
            if best != scheduler.best:
                raise ValueError("resume best validation and scheduler state differ")
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        if rank == 0:
            print(json.dumps({"config": config, "identity": identity, "train_samples": len(train),
                "valid_samples": len(valid), "batches_per_epoch": len(loader), "world_size": world,
                "global_batch": world * config["batch_size"], "gradient_accumulation": 1,
                "objective": "scale-sensitive negative waveform SDR plus log silence NMSE; no near negatives"}, ensure_ascii=False), flush=True)
        for epoch in range(start_epoch, config["epochs"]):
            begin = time.monotonic()
            model.train()
            sampler.set_epoch(epoch)
            train_base.set_epoch(epoch)
            partial = False
            for index, batch in enumerate(loader):
                mix, target, video = (x.to(device, non_blocking=True) for x in batch[:3])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                    output = ddp(mix, video, target.shape[1])
                loss_values, masks = training_loss(output.float(), target.float(), mix[:, :target.shape[1]].float())
                loss = loss_values.mean()
                if not all_finite(loss, device):
                    raise RuntimeError("non-finite loss on at least one rank")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                # DDP gradients are synchronized, and MAX makes rejection
                # identical even if local norms differ or a rank sees NaN.
                max_norm = global_gradient_norm(norm)
                if not torch.isfinite(max_norm) or max_norm.item() > config["grad_norm_limit"]:
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_bad_steps += 1
                    skipped_steps += 1
                    scheduler.reduce_after_bad_step()
                    if rank == 0:
                        print(f"epoch={epoch} step={updates} rejected_gradient_norm={max_norm.item():.6g} lr_scale={scheduler.scale:.8g} consecutive_rejections={consecutive_bad_steps} skipped_total={skipped_steps}", flush=True)
                    if consecutive_bad_steps >= config["max_consecutive_bad_steps"]:
                        raise RuntimeError("too many consecutive abnormal gradient steps")
                    continue
                consecutive_bad_steps = 0
                for group in optimizer.param_groups:
                    group["lr"] = scheduler.learning_rate(updates)
                optimizer.step()
                updates += 1
                metrics_local = stability_metrics.empty(device)
                for metric_start in range(0, output.shape[0], 256):
                    part = slice(metric_start, metric_start + 256)
                    metrics_local = stability_metrics.merge(metrics_local, stability_metrics.collect(loss_values[part], {k: v[part] for k, v in masks.items()}, output[part], mix[part, :target.shape[1]]))
                metrics_global = stability_metrics.summarize(stability_metrics.distributed_reduce(metrics_local))
                peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.int64)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(f"epoch={epoch} step={updates} stability={json.dumps(metrics_global, sort_keys=True)}", flush=True)
                    print(f"epoch={epoch} step={updates} loss_rank0={loss.item():.6f} loss_max_rank0={loss_values.detach().max().item():.6f} silent_rank0={masks['silent'].sum().item()} grad_norm={norm.item():.6f} lr={optimizer.param_groups[0]['lr']:.8g} peak_allocated_gib={peak[0].item()/2**30:.3f} peak_reserved_gib={peak[1].item()/2**30:.3f}", flush=True)
                if args.max_steps and updates >= args.max_steps:
                    partial = index + 1 < len(loader)
                    break
            model.eval()
            limit = min(len(valid), args.validation_limit) if args.validation_limit else len(valid)
            valid_loader = DataLoader(Subset(valid, range(rank, limit, world)),
                batch_size=config["validation_batch_size"], num_workers=config["workers"],
                pin_memory=True, prefetch_factor=1 if config["workers"] else None)
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            validation_metrics = stability_metrics.empty(device)
            with torch.inference_mode():
                for batch in valid_loader:
                    mix, target, video = (x.to(device, non_blocking=True) for x in batch[:3])
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                        prediction = model(mix, video, target.shape[1])
                    values, masks = training_loss(prediction.float(), target.float(), mix[:, :target.shape[1]].float())
                    validation_metrics = stability_metrics.merge(validation_metrics, stability_metrics.collect(values, masks, prediction.float(), mix[:, :target.shape[1]].float()))
                    totals += torch.stack((values.double().sum(), values.new_tensor(values.numel(), dtype=torch.float64), masks["silent"].double().sum(), values[masks["speech"]].double().sum(), masks["speech"].double().sum()))
            dist.all_reduce(totals)
            validation_metrics = stability_metrics.summarize(stability_metrics.distributed_reduce(validation_metrics))
            if rank == 0:
                print(f"epoch={epoch} validation_stability={json.dumps(validation_metrics, sort_keys=True)}", flush=True)
            if not torch.isfinite(totals).all() or totals[1] == 0:
                raise RuntimeError("invalid validation totals")
            validation = (totals[0] / totals[1]).item()
            improved = False
            if not partial and not args.validation_limit:
                previous_significant_best = scheduler.plateau_best
                event = scheduler.observe(validation)
                best = scheduler.best
                improved = event["improved"]
                if validation < previous_significant_best - config["min_delta"]:
                    stale = 0
                else:
                    stale += 1
                if rank == 0:
                    print(f"epoch={epoch} validation_best={best:.6f} lr_scale={scheduler.scale:.8g} lr_reduced={event['reduced']} severe={event['severe']}", flush=True)
            if rank == 0:
                final = run_dir / f"epoch-{epoch:04d}-step-{updates:08d}.pt"
                if improved:
                    best_checkpoint = str(final)
                payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": config, "identity": identity,
                    "world_size": world, "protocol": protocol, "next_epoch": epoch + 1, "updates": updates,
                    "stale": stale, "best": best, "best_checkpoint": best_checkpoint,
                    "consecutive_bad_steps": consecutive_bad_steps, "skipped_steps": skipped_steps,
                    "partial": partial, "validation": validation, "objective": "scale_sensitive_waveform_sdr_plus_log_silence"}
                tmp = run_dir / f".epoch-{epoch:04d}.tmp"
                torch.save(payload, tmp)
                os.replace(tmp, final)
                print(f"epoch={epoch} validation={validation:.6f} samples={int(totals[1])} silent={int(totals[2])} speech_validation={(totals[3]/totals[4]).item() if totals[4] else float('nan'):.6f} elapsed={time.monotonic()-begin:.1f} checkpoint={final}", flush=True)
            dist.barrier()
            if (args.max_steps and updates >= args.max_steps) or (updates >= config["warmup_steps"] and stale >= config["early_stop_patience"]):
                break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
