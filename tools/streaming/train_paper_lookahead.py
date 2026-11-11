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
from tools.streaming.realssa_lookahead import RealSSALookaheadDataset
from tools.streaming.train_ddp import reconstruction_components


class PaperLookaheadTrainingModel(nn.Module):
    """Use original frozen FRCNN and differentiable, non-detached core state."""
    def __init__(self, config):
        super().__init__()
        backbone = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu", pretrain=None)
        backbone.init_from(config["visual_weights"])
        self.visual = FalconLookaheadVisual(backbone)
        self.core = PaperStreamPrototype(width=config["width"], rnn_hidden=config["rnn_hidden"],
            heads=config["heads"], history_frames=config["history_frames"])
        self.checkpoint_chunks = config["checkpoint_chunks"]

    def forward(self, audio, video, output_samples=None):
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def training_loss(output, target, mix):
    # SI-SDR for non-degenerate speech; silence uses scale-normalized MSE.
    # The TAR does not supply the individual near-field negative waveforms.
    return reconstruction_components(output, target, mix)[0].mean()


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
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = {"train_index", "valid_index", "visual_weights", "experiment_root", "batch_size",
                "validation_batch_size", "workers", "lr", "warmup_steps", "weight_decay", "epochs",
                "lr_patience", "early_stop_patience", "min_delta", "seed", "width", "rnn_hidden",
                "heads", "history_frames", "checkpoint_chunks", "precision"}
    if not isinstance(config, dict) or set(config) != required:
        parser.error("config must have exactly the documented keys")
    if config["precision"] not in ("bf16", "fp32"):
        parser.error("precision must be bf16 or fp32")
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if min(config["batch_size"], config["validation_batch_size"], config["epochs"], config["early_stop_patience"]) < 1:
        parser.error("batch sizes, epochs and early-stop patience must be positive")
    if config["lr"] <= 0 or config["warmup_steps"] < 0 or config["workers"] < 0:
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
            patience=config["lr_patience"], threshold=config["min_delta"], threshold_mode="abs", min_lr=1e-6)
        identity = {"visual_sha256": sha256(config["visual_weights"]),
                    "train_index_sha256": sha256(config["train_index"]),
                    "valid_index_sha256": sha256(config["valid_index"]),
                    "config_sha256": sha256(args.config),
                    "trainer_sha256": sha256(__file__),
                    "core_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconPaperStreamPrototype.py"),
                    "visual_code_sha256": sha256(Path(__file__).resolve().parents[2] / "look2hear/models/FalconLookaheadVisual.py"),
                    "window_code_sha256": sha256(Path(__file__).with_name("lookahead_training.py")),
                    "data_code_sha256": sha256(Path(__file__).with_name("realssa_lookahead.py"))}
        protocol = {"train_limit": args.train_limit, "validation_limit": args.validation_limit}
        start_epoch = updates = stale = 0
        best = float("inf")
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
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        if rank == 0:
            print(json.dumps({"config": config, "identity": identity, "train_samples": len(train),
                "valid_samples": len(valid), "batches_per_epoch": len(loader), "world_size": world,
                "global_batch": world * config["batch_size"], "gradient_accumulation": 1,
                "objective": "negative SI-SDR plus silence NMSE; no near negatives"}, ensure_ascii=False), flush=True)
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
                loss = training_loss(output.float(), target.float(), mix[:, :target.shape[1]].float())
                if not all_finite(loss, device):
                    raise RuntimeError("non-finite loss on at least one rank")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not all_finite(norm, device):
                    raise RuntimeError("non-finite gradient on at least one rank")
                if updates < config["warmup_steps"]:
                    for group in optimizer.param_groups:
                        group["lr"] = config["lr"] * (updates + 1) / config["warmup_steps"]
                optimizer.step()
                updates += 1
                peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.int64)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(f"epoch={epoch} step={updates} loss_rank0={loss.item():.6f} grad_norm={norm.item():.6f} lr={optimizer.param_groups[0]['lr']:.8g} peak_allocated_gib={peak[0].item()/2**30:.3f} peak_reserved_gib={peak[1].item()/2**30:.3f}", flush=True)
                if args.max_steps and updates >= args.max_steps:
                    partial = index + 1 < len(loader)
                    break
            model.eval()
            limit = min(len(valid), args.validation_limit) if args.validation_limit else len(valid)
            valid_loader = DataLoader(Subset(valid, range(rank, limit, world)),
                batch_size=config["validation_batch_size"], num_workers=config["workers"],
                pin_memory=True, prefetch_factor=1 if config["workers"] else None)
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            with torch.inference_mode():
                for batch in valid_loader:
                    mix, target, video = (x.to(device, non_blocking=True) for x in batch[:3])
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                        prediction = model(mix, video, target.shape[1])
                    values = reconstruction_components(prediction.float(), target.float(), mix[:, :target.shape[1]].float())[0]
                    totals += torch.stack((values.double().sum(), values.new_tensor(values.numel(), dtype=torch.float64)))
            dist.all_reduce(totals)
            if not torch.isfinite(totals).all() or totals[1] == 0:
                raise RuntimeError("invalid validation totals")
            validation = (totals[0] / totals[1]).item()
            if not partial and not args.validation_limit and updates >= config["warmup_steps"]:
                scheduler.step(validation)
                if validation < best - config["min_delta"]:
                    best, stale = validation, 0
                else:
                    stale += 1
            if rank == 0:
                payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": config, "identity": identity,
                    "world_size": world, "protocol": protocol, "next_epoch": epoch + 1, "updates": updates,
                    "stale": stale, "best": best, "partial": partial, "validation": validation}
                tmp = run_dir / f".epoch-{epoch:04d}.tmp"
                final = run_dir / f"epoch-{epoch:04d}-step-{updates:08d}.pt"
                torch.save(payload, tmp)
                os.replace(tmp, final)
                print(f"epoch={epoch} validation={validation:.6f} samples={int(totals[1])} elapsed={time.monotonic()-begin:.1f} checkpoint={final}", flush=True)
            dist.barrier()
            if args.max_steps and updates >= args.max_steps or stale >= config["early_stop_patience"]:
                break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
