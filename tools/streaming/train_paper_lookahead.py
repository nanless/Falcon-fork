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
from tools.streaming.exclude_realssa import (
    IndexedSamples, batch_exclusions, load_exclusions, replace_excluded_inputs,
    require_retained_global, training_mean,
)


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


def training_loss(output, target, mix, valid_samples=None):
    return scale_sdr_loss(output, target, mix, valid_samples=valid_samples)


def all_finite(value, device):
    ok = torch.tensor(int(torch.isfinite(value).all().item()), device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--train-exclusion-manifest")
    parser.add_argument("--validation-exclusion-manifest", help="report a parallel validation sensitivity")
    parser.add_argument("--diagnostic-checkpoint", help="initialize a bounded LR arm from historical model and AdamW")
    parser.add_argument("--diagnostic-checkpoint-sha256")
    parser.add_argument("--diagnostic-output-root", type=Path)
    parser.add_argument("--diagnostic-validation-manifest", type=Path,
        help="fixed CV subset, optionally extended with the audited exclusion sentinels")
    parser.add_argument("--diagnostic-lr", type=float)
    parser.add_argument("--diagnostic-attempts", type=int)
    parser.add_argument("--diagnostic-replay-batch", type=int, help="DDP forward/backward only at historical next epoch")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0, help="bounded smoke only; 0 uses the whole split")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single new directory name")
    if args.validation_limit < 0 or args.train_limit < 0 or args.max_steps < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("invalid test override")
    replay_batch = args.diagnostic_replay_batch
    diagnostic = any(value is not None for value in (
        args.diagnostic_checkpoint, args.diagnostic_checkpoint_sha256, args.diagnostic_lr,
        args.diagnostic_attempts, replay_batch, args.diagnostic_output_root, args.diagnostic_validation_manifest))
    if diagnostic and (args.resume or args.max_steps or args.train_limit or args.batch_size is not None
            or not args.diagnostic_checkpoint or not args.diagnostic_checkpoint_sha256
            or not args.diagnostic_output_root or (replay_batch is not None and args.diagnostic_validation_manifest)):
        parser.error("diagnostic needs checkpoint and SHA, output root, original batch and full training index")
    if replay_batch is not None and (replay_batch < 0 or args.diagnostic_lr is not None or args.diagnostic_attempts is not None):
        parser.error("diagnostic replay requires a nonnegative batch number and no LR or step override")
    if diagnostic and replay_batch is None and (not args.train_exclusion_manifest or not args.validation_limit
            or args.diagnostic_lr is None or not 0 < args.diagnostic_lr < float("inf")
            or args.diagnostic_attempts is None or not 1 <= args.diagnostic_attempts <= 8):
        parser.error("diagnostic needs checkpoint and SHA, finite positive LR, 1–8 attempts, exclusion manifest and validation limit")
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = {"train_index", "valid_index", "visual_weights", "experiment_root", "batch_size",
                "validation_batch_size", "workers", "lr", "warmup_steps", "weight_decay", "epochs",
                "lr_patience", "early_stop_patience", "min_delta", "seed", "width", "rnn_hidden",
                "heads", "history_frames", "checkpoint_chunks", "precision",
                "grad_norm_limit", "max_consecutive_bad_steps", "validation_severe_delta",
                "mask_limit", "activation_batch_chunk"}
    if not isinstance(config, dict) or set(config) != required:
        parser.error("config must have exactly the documented keys")
    if config["precision"] not in ("bf16", "fp32"):
        parser.error("precision must be bf16 or fp32")
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if min(config["batch_size"], config["validation_batch_size"], config["epochs"], config["early_stop_patience"]) < 1:
        parser.error("batch sizes, epochs and early-stop patience must be positive")
    if (config["lr"] <= 0 or config["warmup_steps"] < 0 or config["workers"] < 0
            or config["grad_norm_limit"] <= 0 or config["max_consecutive_bad_steps"] < 1
            or config["validation_severe_delta"] <= 0 or config["mask_limit"] <= 0
            or config["activation_batch_chunk"] < 1):
        parser.error("invalid optimizer or worker configuration")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("this recipe requires exactly two processes and two GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        run_dir = (args.diagnostic_output_root or Path(config["experiment_root"])) / args.run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"] + rank)
        random.seed(config["seed"] + rank)
        train = RealSSALookaheadDataset(config["train_index"], seed=config["seed"], crop="random",
                                         augment=True, return_valid_samples=True)
        train_base = train
        valid = RealSSALookaheadDataset(config["valid_index"], seed=config["seed"], crop="center",
                                         augment=False, return_valid_samples=True)
        if train_base.split != "tr" or valid.split != "cv":
            raise ValueError("official training requires tr index and cv validation index")
        train_sha, valid_sha = sha256(config["train_index"]), sha256(config["valid_index"])
        train_excluded = valid_excluded = None
        validation_rows = None
        if args.diagnostic_validation_manifest:
            selection = json.loads(args.diagnostic_validation_manifest.read_bytes())
            if selection["index_sha256"] != valid_sha:
                raise ValueError("diagnostic CV selection index SHA differs")
            validation_rows = [row["index"] for row in selection["rows"]]
            if (not validation_rows or len(validation_rows) != len(set(validation_rows))
                    or any(not 0 <= i < len(valid) or valid.samples[i]["id"] != row["id"]
                           for i, row in zip(validation_rows, selection["rows"], strict=True))):
                raise ValueError("invalid diagnostic CV rows")
        exclusion_identity = {}
        if args.train_exclusion_manifest:
            train_excluded, exclusion_identity["train_exclusion_sha256"] = load_exclusions(
                args.train_exclusion_manifest, train_sha, train_base)
            train = IndexedSamples(train)
        if args.validation_exclusion_manifest:
            valid_excluded, exclusion_identity["validation_exclusion_sha256"] = load_exclusions(
                args.validation_exclusion_manifest, valid_sha, valid)
            valid = IndexedSamples(valid)
            if validation_rows is not None:
                validation_rows.extend(i for i in valid_excluded if i not in validation_rows)
        if args.train_limit:
            train = Subset(train, range(min(len(train), args.train_limit)))
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                     seed=config["seed"], drop_last=True)
        loader = DataLoader(train, sampler=sampler, batch_size=config["batch_size"], drop_last=True,
            num_workers=config["workers"], pin_memory=True,
            prefetch_factor=1 if config["workers"] else None)
        if not len(loader):
            raise ValueError("training split has no full per-rank batch")
        model = PaperLookaheadTrainingModel(config).to(device)
        optimizer = None if replay_batch is not None else torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = RobustWarmupPlateau(config["lr"], config["warmup_steps"], factor=0.5,
            patience=config["lr_patience"], min_delta=config["min_delta"],
            severe_delta=config["validation_severe_delta"])
        identity = {"visual_sha256": sha256(config["visual_weights"]),
                    "train_index_sha256": train_sha,
                    "valid_index_sha256": valid_sha,
                    "config_sha256": sha256(args.config),
                    "trainer_sha256": sha256(__file__),
                    "core_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconPaperStreamPrototype.py"),
                    "visual_code_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconLookaheadVisual.py"),
                    "window_code_sha256": sha256(Path(__file__).with_name("lookahead_training.py")),
                    "data_code_sha256": sha256(Path(__file__).with_name("realssa_lookahead.py")),
                    "schedule_code_sha256": sha256(Path(__file__).with_name("robust_schedule.py")),
                    "loss_code_sha256": sha256(Path(__file__).with_name("scale_sdr_loss.py")),
                    "metrics_code_sha256": sha256(Path(__file__).with_name("stability_metrics.py")),
                    "legacy_data_code_sha256": sha256(Path(__file__).with_name("realssa_tar.py")),
                    "common_code_sha256": sha256(Path(__file__).with_name("common.py")),
                    "mask_source_sha256": sha256(Path(__file__).with_name("train_ddp.py")),
                    "backbone_code_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/videomodels/frcnn_videomodel.py")}
        if exclusion_identity:
            identity.update(exclusion_identity)
            identity["exclusion_code_sha256"] = sha256(Path(__file__).with_name("exclude_realssa.py"))
        protocol = {"train_limit": args.train_limit, "validation_limit": args.validation_limit}
        if diagnostic:
            protocol["diagnostic"] = {
                "checkpoint": args.diagnostic_checkpoint,
                "checkpoint_sha256": args.diagnostic_checkpoint_sha256,
                "effective_lr": args.diagnostic_lr,
                "maximum_attempts": args.diagnostic_attempts,
                "replay_batch": replay_batch,
                "validation_manifest_sha256": sha256(args.diagnostic_validation_manifest) if validation_rows else None,
                "validation_rows": validation_rows,
                "scheduler": "none" if replay_batch is not None else "fixed LR; historical AdamW state retained; validation is observational",
            }
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
        if diagnostic:
            if sha256(args.diagnostic_checkpoint) != args.diagnostic_checkpoint_sha256:
                raise ValueError("diagnostic checkpoint SHA differs")
            saved = torch.load(args.diagnostic_checkpoint, map_location="cpu", weights_only=True)
            # The trainer hash changes with this experiment. Data, model, and all
            # remaining historical config fields must still be identical.
            historical_identity = dict(saved["identity"])
            historical_identity.pop("trainer_sha256")
            # Explicit migration from the two known pre-fix implementations.
            previous = {
                "data_code_sha256": "a1aa92647f67de2583ab240388e52bbbfc8e79a31aba32e1b5ea2e70fd1124b1",
                "loss_code_sha256": "16519c78d9c4411d2b53e546101f96cf8b03989e1c875b409867de7c85993e99",
            }
            for key, digest in previous.items():
                if historical_identity.get(key) == digest:
                    historical_identity[key] = identity[key]
            if (saved["partial"] or saved["config"] != config or saved["world_size"] != world or
                    any(identity.get(k) != v for k,v in historical_identity.items())):
                raise ValueError("diagnostic requires a compatible completed historical checkpoint")
            model.load_state_dict(saved["model"], strict=True)
            if replay_batch is None:
                optimizer.load_state_dict(saved["optimizer"])
            start_epoch, updates = saved["next_epoch"], saved["updates"]
            if not 0 <= start_epoch < config["epochs"]:
                raise ValueError("historical next epoch outside configured training range")
            del saved
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        exclusion_log = run_dir / f"training-exclusions-rank{rank}.jsonl"
        attempts = 0
        if rank == 0:
            print(json.dumps({"config": config, "identity": identity, "train_samples": len(train),
                "valid_samples": len(valid), "batches_per_epoch": len(loader), "world_size": world,
                "global_batch": world * config["batch_size"], "gradient_accumulation": 1,
                "exclusion_counts": {"training": len(train_excluded or {}), "validation_sensitivity": len(valid_excluded or {})},
                "protocol": protocol,
                "objective": "scale-sensitive negative waveform SDR plus log silence NMSE; no near negatives"}, ensure_ascii=False), flush=True)
        for epoch in range(start_epoch, config["epochs"]):
            begin = time.monotonic()
            model.train()
            sampler.set_epoch(epoch)
            train_base.set_epoch(epoch)
            partial = False
            epoch_loader = loader
            batch_start = 0
            if replay_batch is not None:
                batch_start = replay_batch
                first = batch_start * config["batch_size"]
                indices = list(sampler)[first:first + config["batch_size"]]
                if len(indices) != config["batch_size"]:
                    raise ValueError("diagnostic replay batch is incomplete or out of range")
                epoch_loader = DataLoader(Subset(train, indices), batch_size=config["batch_size"],
                    num_workers=config["workers"], pin_memory=True,
                    prefetch_factor=1 if config["workers"] else None)
            for index, batch in enumerate(epoch_loader, start=batch_start):
                attempts += 1
                mix, target, video = (x.to(device, non_blocking=True) for x in batch[:3])
                excluded_mask = None
                if train_excluded is not None:
                    excluded_mask, hits = batch_exclusions(batch[4], train_excluded, device)
                    with exclusion_log.open("a", encoding="utf-8") as stream:
                        print(json.dumps({"epoch": epoch, "batch": index, "updates_before": updates,
                                          "batch_size": len(batch[4]), "retained": len(batch[4]) - len(hits),
                                          "excluded": len(hits), "exclusions": hits}), file=stream)
                    require_retained_global(excluded_mask)
                    if hits:
                        mix, target, video = replace_excluded_inputs(excluded_mask, mix, target, video)
                if replay_batch is not None:
                    model.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                    output = ddp(mix, video, target.shape[1])
                lengths = batch[3].to(device, non_blocking=True)
                loss_values, masks = training_loss(output.float(), target.float(),
                                                  mix[:, :target.shape[1]].float(), lengths)
                loss = training_mean(loss_values, excluded_mask)
                if not all_finite(loss, device):
                    raise RuntimeError("non-finite loss on at least one rank")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                # DDP gradients are synchronized, and MAX makes rejection
                # identical even if local norms differ or a rank sees NaN.
                max_norm = global_gradient_norm(norm)
                if replay_batch is not None:
                    record = {"epoch": epoch, "batch": index, "rank": rank, "updates_loaded": updates,
                              "optimizer_steps": 0, "gradient_norm": max_norm.item(), "loss": loss.item(),
                              "shape": [list(x.shape) for x in batch[:3]],
                              "input_sha256": [hashlib.sha256(memoryview(x.contiguous().numpy()).cast("B")).hexdigest() for x in batch[:3]],
                              "prediction_sha256": hashlib.sha256(memoryview(output.detach().cpu().contiguous().numpy()).cast("B")).hexdigest()}
                    with (run_dir / f"replay-rank{rank}.json").open("x") as stream:
                        json.dump(record, stream, indent=2)
                    print(json.dumps(record), flush=True)
                    return
                if diagnostic and rank == 0:
                    print(json.dumps({"diagnostic_attempt": attempts, "epoch": epoch, "batch": index,
                        "updates_before": updates, "gradient_norm": max_norm.item(), "loss_rank0": loss.item(),
                        "fixed_lr": args.diagnostic_lr, "accepted": bool(torch.isfinite(max_norm) and max_norm.item() <= config["grad_norm_limit"])}), flush=True)
                if not torch.isfinite(max_norm) or max_norm.item() > config["grad_norm_limit"]:
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_bad_steps += 1
                    skipped_steps += 1
                    if not diagnostic:
                        scheduler.reduce_after_bad_step()
                    if rank == 0:
                        print(f"epoch={epoch} step={updates} rejected_gradient_norm={max_norm.item():.6g} lr_scale={scheduler.scale:.8g} consecutive_rejections={consecutive_bad_steps} skipped_total={skipped_steps}", flush=True)
                    if diagnostic and attempts >= args.diagnostic_attempts:
                        partial = True
                        break
                    if consecutive_bad_steps >= config["max_consecutive_bad_steps"]:
                        raise RuntimeError("too many consecutive abnormal gradient steps")
                    continue
                consecutive_bad_steps = 0
                for group in optimizer.param_groups:
                    group["lr"] = args.diagnostic_lr if diagnostic else scheduler.learning_rate(updates)
                optimizer.step()
                updates += 1
                metrics_local = stability_metrics.empty(device)
                for metric_start in range(0, output.shape[0], 256):
                    part = slice(metric_start, metric_start + 256)
                    active = slice(None) if excluded_mask is None else ~excluded_mask[part]
                    values = loss_values[part][active]
                    if values.numel():
                        metrics_local = stability_metrics.merge(metrics_local, stability_metrics.collect(
                            values, {k: v[part][active] for k, v in masks.items()}, output[part][active],
                            mix[part, :target.shape[1]][active]))
                metrics_global = stability_metrics.summarize(stability_metrics.distributed_reduce(metrics_local))
                peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.int64)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(f"epoch={epoch} step={updates} stability={json.dumps(metrics_global, sort_keys=True)}", flush=True)
                    retained = slice(None) if excluded_mask is None else ~excluded_mask
                    retained_values = loss_values.detach()[retained]
                    maximum = retained_values.max().item() if retained_values.numel() else float("nan")
                    print(f"epoch={epoch} step={updates} loss_rank0={loss.item():.6f} loss_max_rank0={maximum:.6f} silent_rank0={masks['silent'][retained].sum().item()} grad_norm={norm.item():.6f} lr={optimizer.param_groups[0]['lr']:.8g} peak_allocated_gib={peak[0].item()/2**30:.3f} peak_reserved_gib={peak[1].item()/2**30:.3f}", flush=True)
                if diagnostic and attempts >= args.diagnostic_attempts:
                    partial = True
                    break
                if args.max_steps and updates >= args.max_steps:
                    partial = index + 1 < len(loader)
                    break
            model.eval()
            limit = min(len(valid), args.validation_limit) if args.validation_limit else len(valid)
            valid_indices = validation_rows if validation_rows is not None else range(limit)
            valid_loader = DataLoader(Subset(valid, valid_indices[rank::world]),
                batch_size=config["validation_batch_size"], num_workers=config["workers"],
                pin_memory=True, prefetch_factor=1 if config["workers"] else None)
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            validation_metrics = stability_metrics.empty(device)
            sensitivity = torch.zeros(3, dtype=torch.float64, device=device)
            with torch.inference_mode():
                for batch in valid_loader:
                    mix, target, video = (x.to(device, non_blocking=True) for x in batch[:3])
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                        prediction = model(mix, video, target.shape[1])
                    values, masks = training_loss(prediction.float(), target.float(),
                                                  mix[:, :target.shape[1]].float(),
                                                  batch[3].to(device, non_blocking=True))
                    validation_metrics = stability_metrics.merge(validation_metrics, stability_metrics.collect(values, masks, prediction.float(), mix[:, :target.shape[1]].float()))
                    totals += torch.stack((values.double().sum(), values.new_tensor(values.numel(), dtype=torch.float64), masks["silent"].double().sum(), values[masks["speech"]].double().sum(), masks["speech"].double().sum()))
                    if valid_excluded is not None:
                        removed, _ = batch_exclusions(batch[4], valid_excluded, device)
                        sensitivity += torch.stack((values[~removed].double().sum(), (~removed).sum(), removed.sum()))
            dist.all_reduce(totals)
            validation_metrics = stability_metrics.summarize(stability_metrics.distributed_reduce(validation_metrics))
            if valid_excluded is not None:
                dist.all_reduce(sensitivity)
                if rank == 0:
                    retained, excluded = int(sensitivity[1]), int(sensitivity[2])
                    loss = (sensitivity[0] / sensitivity[1]).item() if retained else None
                    print(json.dumps({"epoch": epoch, "validation_sensitivity": {
                        "loss": loss, "retained": retained, "excluded": excluded}}), flush=True)
            if rank == 0:
                print(f"epoch={epoch} validation_stability={json.dumps(validation_metrics, sort_keys=True)}", flush=True)
            if not torch.isfinite(totals).all() or totals[1] == 0:
                raise RuntimeError("invalid validation totals")
            validation = (totals[0] / totals[1]).item()
            improved = False
            if not diagnostic and not partial and not args.validation_limit:
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
                if diagnostic:
                    payload["diagnostic_attempts"] = attempts
                tmp = run_dir / f".epoch-{epoch:04d}.tmp"
                torch.save(payload, tmp)
                os.replace(tmp, final)
                print(f"epoch={epoch} validation={validation:.6f} samples={int(totals[1])} silent={int(totals[2])} speech_validation={(totals[3]/totals[4]).item() if totals[4] else float('nan'):.6f} elapsed={time.monotonic()-begin:.1f} checkpoint={final}", flush=True)
            dist.barrier()
            if diagnostic or (args.max_steps and updates >= args.max_steps) or (updates >= config["warmup_steps"] and stale >= config["early_stop_patience"]):
                break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
