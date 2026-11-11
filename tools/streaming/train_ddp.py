"""Two-GPU streaming trial with reconstruction and optional positive contrastive loss.

Run with torchrun; output is a v3 checkpoint loadable by streaming inference.
The final 80 ms in the 13th block is padded and excluded from the loss.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from look2hear.models.FalconStreaming import FalconStreaming
from look2hear.losses import pairwise_neg_sisdr
from look2hear.losses.nass_contrastive_loss import NASSContrastiveLoss
from look2hear.models.Falcon_Contrastive import AttentionSpeakerPool
from tools.streaming.common import (
    CHECKPOINT_FORMAT, FRAMES, SAMPLES, SAMPLES_PER_FRAME, IndexedAudioVisualDataset, read_checkpoint, load_mono, mouth_tensor,
)

SEGMENT_SAMPLES = 32000
SEGMENT_FRAMES = 50
CHUNKS = 13


class TwoSecondDataset(IndexedAudioVisualDataset):
    """Aligned 2-second segments on the same 40 ms grid as streaming data."""
    def __init__(self, directory, *, validation=False):
        super().__init__(directory, 12)
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


class TrainingWrapper(torch.nn.Module):
    # The auxiliary head is used for training only; checkpoint state is separate.
    def __init__(self, model, positive_contrastive):
        super().__init__()
        self.model = model
        self.positive_contrastive = positive_contrastive
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
        return self.contrastive(query, positive, [None] * output.size(0), beta=0.0).squeeze()

    def forward(self, padded_audio, video, target):
        output = self.model(padded_audio, video)
        positive = self.positive_loss(output[:, :SEGMENT_SAMPLES], target) if self.positive_contrastive else None
        return output, positive


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


def advance_validation(scheduler, loss, best, stale, partial):
    # Step-limited partial epochs report validation but do not affect epoch policy.
    if partial:
        return best, stale, False
    scheduler.step(loss)
    improved = loss < best
    return min(best, loss), 0 if improved else stale + 1, improved


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", help="new directory for immutable per-epoch checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", help="resume model/optimizer/scheduler; data position restarts at the beginning of the saved epoch")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--valid-index-dir")
    validation.add_argument("--valid-tar-index")
    parser.add_argument("--validation-limit", type=int, default=0, help="0 evaluates the full validation index; positive values limit to its first N entries")
    parser.add_argument("--batch-size", type=int, default=4, help="per-rank batch")
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0, help="optimizer updates; 0 means no limit")
    parser.add_argument("--lr", type=float, default=None, help="new runs default to 0.001; resumes retain saved optimizer rate unless overridden")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    args = parser.parse_args()
    if min(args.batch_size, args.accum, args.epochs, args.save_every) < 1 or args.max_steps < 0 or args.workers < 0 or args.validation_limit < 0:
        parser.error("invalid batch/accum/epochs/steps/workers")
    if (args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0)) or not np.isfinite(args.weight_decay) or args.weight_decay < 0 or not np.isfinite(args.contrastive_weight) or args.contrastive_weight < 0:
        parser.error("invalid optimizer parameters")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        parser.error("launch with torchrun; RANK/WORLD_SIZE are required")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 2:
        parser.error("this trial requires exactly two ranks")
    if Path(args.output).exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=args.backend)
    try:
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
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, drop_last=True)
        if len(loader) < args.accum:
            raise ValueError("dataset too small for batch, ranks and accumulation")
        model = FalconStreaming()
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
        training_model = TrainingWrapper(model, args.positive_contrastive).to(device)
        objective_name = "negative_si_snr_plus_positive_contrastive_no_negatives" if args.positive_contrastive else "negative_si_snr_reconstruction_only"
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
        train_fingerprint = {"kind": "synthetic"} if args.synthetic else index_fingerprint(args.tar_index, args.index_dir)
        valid_fingerprint = index_fingerprint(args.valid_tar_index, args.valid_index_dir)
        resume_protocol = {"train": train_fingerprint, "validation": valid_fingerprint,
                           "world_size": world, "batch_per_rank": args.batch_size, "accumulation": args.accum,
                           "seed": args.seed, "workers": args.workers, "validation_limit": args.validation_limit, "weight_decay": args.weight_decay, "visual_sha256": visual_sha,
                           "model_code_sha256": model_code_sha, "trainer_code_sha256": trainer_code_sha,
                           "data_code_sha256": data_code_sha, "torch_version": str(torch.__version__), "objective": objective_name,
                           "contrastive_weight": args.contrastive_weight if args.positive_contrastive else 0.0,
                           "optimizer": "AdamW", "segment_samples": SEGMENT_SAMPLES}
        if checkpoint is not None:
            validate_resume_protocol(checkpoint.get("resume_protocol"), resume_protocol)
            if args.lr is not None and args.lr != checkpoint.get("learning_rate"):
                raise ValueError("resume protocol mismatch: learning_rate")
        ddp = DistributedDataParallel(training_model, device_ids=[local_rank] if device.type == "cuda" else None)
        optimizer = torch.optim.AdamW((p for p in training_model.parameters() if p.requires_grad), lr=args.lr if args.lr is not None else .001, weight_decay=args.weight_decay)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if args.lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=15) if valid else None
        if checkpoint is not None and scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        updates = int(checkpoint.get("steps", 0)) if checkpoint else 0
        start_updates = updates
        start_epoch = int(checkpoint.get("next_epoch", 0)) if checkpoint else 0
        best = float(checkpoint.get("best_validation", float("inf"))) if checkpoint else float("inf")
        stale = int(checkpoint.get("stale_validations", 0)) if checkpoint else 0
        end_reason = "epochs"
        best_checkpoint = checkpoint.get("best_checkpoint") if checkpoint else None
        validation_mean = None
        def checkpoint_payload(current_epoch, partial):
            return {
                "format": CHECKPOINT_FORMAT, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                "resume_protocol": resume_protocol, "split_identity_audit": split_identity_audit,
                "steps": updates, "next_epoch": current_epoch if partial else current_epoch + 1,
                "resume_restarts_epoch": partial, "best_validation": best, "stale_validations": stale,
                "validation_loss": validation_mean, "validation_limit": args.validation_limit,
                "validation_samples": (min(len(valid), args.validation_limit) if args.validation_limit else len(valid)) if valid is not None else 0,
                "validation_total": len(valid) if valid is not None else 0, "best_checkpoint": best_checkpoint,
                "training_status": "synthetic_smoke" if args.synthetic else "real_data_optimization",
                "training_objective": objective_name,
                "speaker_pool_state": training_model.speaker_pool.state_dict() if args.positive_contrastive else None,
                "positive_contrastive_weight": args.contrastive_weight if args.positive_contrastive else 0.0,
                "positive_contrastive_definition": "output_and_target_valid_2s_reencoded_by_v3_audio_proj" if args.positive_contrastive else None,
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
                "learning_rate": optimizer.param_groups[0]["lr"], "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "seed": args.seed, "end_reason": end_reason,
            }
        if rank == 0:
            visual_label = "random-frozen" if args.random_visual else args.visual_init
            print(f"trial: backend={args.backend} world={world} batch_per_rank={args.batch_size} accumulation={args.accum} global_batch={world*args.batch_size*args.accum} segment_samples={SEGMENT_SAMPLES} visual={visual_label} objective={objective_name} near_negatives=NOT_USED", flush=True)
        training_model.train()
        for epoch in range(start_epoch, start_epoch + args.epochs):
            sampler.set_epoch(epoch)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            # Process full accumulation groups only, so each update has the same scale.
            usable_batches = len(loader) // args.accum * args.accum
            for index, (mixture, target, video) in enumerate(loader):
                if index >= usable_batches:
                    break
                mixture, target, video = (item.to(device) for item in (mixture, target, video))
                padded, frames = pad_two_seconds(mixture, video)
                output, contrastive = ddp(padded, frames, target)
                output = output[:, :SEGMENT_SAMPLES]
                loss = neg_si_snr(output, target)
                if contrastive is not None:
                    loss = loss + args.contrastive_weight * contrastive
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                (loss / args.accum).backward()
                if (index + 1) % args.accum == 0:
                    norm = torch.nn.utils.clip_grad_norm_(training_model.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("non-finite gradient")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    if rank == 0:
                        print(f"epoch={epoch} step={updates} loss_last_microbatch={loss.item():.6f} grad_norm={norm.item():.6f}", flush=True)
                    if args.max_steps and updates - start_updates >= args.max_steps:
                        end_reason = "max_steps"
                        break
            partial = end_reason == "max_steps" and (index + 1) < usable_batches
            improved = False
            validation_mean = None
            if valid is not None:
                # Each sample is evaluated exactly once across ranks, without sampler padding.
                training_model.eval()
                total = torch.zeros(2, device=device, dtype=torch.float64)
                with torch.inference_mode():
                    validation_count = min(len(valid), args.validation_limit) if args.validation_limit else len(valid)
                    for i in range(rank, validation_count, world):
                        mixture, target, video = valid[i]
                        mixture, target, video = (x.unsqueeze(0).to(device) for x in (mixture, target, video))
                        padded, frames = pad_two_seconds(mixture, video)
                        val_loss = neg_si_snr(model(padded, frames)[:, :SEGMENT_SAMPLES], target)
                        if not torch.isfinite(val_loss):
                            raise RuntimeError("non-finite validation loss")
                        total += torch.tensor([val_loss.item(), 1], device=device, dtype=torch.float64)
                dist.all_reduce(total)
                mean = (total[0] / total[1]).item()
                validation_mean = mean
                best, stale, improved = advance_validation(scheduler, mean, best, stale, partial)
                if rank == 0:
                    learning_rate = optimizer.param_groups[0]["lr"]
                    print(f"epoch={epoch} validation_samples={int(total[1])} validation_total={len(valid)} validation_limit={args.validation_limit} loss={mean:.6f} lr={learning_rate:.8g} partial={partial}", flush=True)
                training_model.train()
                if stale >= 30:
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
