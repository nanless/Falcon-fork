"""Fixed-budget RMS augmentation x labeled visible-silence supervision."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import pad

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from tools.streaming import rolling_factorial_ab as factorial, rolling_init_ab as init
from tools.streaming.common import mouth_tensor
from tools.streaming.evaluate_realssa import load_baseline, sha
from tools.streaming.rolling_model_checkpoint import RollingExperimentLiveInference, pack_rolling_checkpoint
from tools.streaming.scale_sdr_loss import scale_sdr_loss

ARMS = {f"level_{level}_silence_{silent}": (level == "on", silent == "on")
        for silent in ("off", "on") for level in ("off", "on")}
REAL_INTERVAL = 4


def load_episodes(manifest):
    payload = init.read_json(manifest)
    records = payload.get("episodes", payload.get("rows", []))
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("episodes must have unique IDs")
    episodes = []
    for row in records:
        path = Path(row["npz"])
        if not path.is_absolute():
            path = manifest.parent / path
        if sha(path) != row["sha256"]:
            raise ValueError("real episode checksum differs from frozen manifest")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in ("audio", "target", "mouth", "silent", "active", "mouth_valid")}
        ticks = len(arrays["mouth"])
        if any(arrays[k].shape != (ticks * init.TICK_SAMPLES,) or arrays[k].dtype != np.float32
               or not np.isfinite(arrays[k]).all() for k in ("audio", "target")):
            raise ValueError("real audio must be finite float32 on the 16 kHz / 25 fps grid")
        if any(arrays[k].shape != (ticks,) or arrays[k].dtype != np.bool_
               for k in ("silent", "active", "mouth_valid")):
            raise ValueError("real episode masks must be aligned boolean arrays")
        if (arrays["silent"] & arrays["active"]).any():
            raise ValueError("target cannot be simultaneously labeled silent and active")
        valid = arrays["mouth_valid"]
        last_start = (ticks - 5) // 3
        masks = {}
        for kind in ("silent", "active"):
            available = [block for block in range(init.HISTORY, last_start + 1)
                         if valid[3*(block-init.HISTORY)-2 if block > init.HISTORY else 0:3*block+5].all()
                         and arrays[kind][3*block:3*block+3].all()]
            requested = row.get(kind + "_blocks", available)
            if not requested or len(requested) != len(set(requested)) or set(requested) - set(available):
                raise ValueError(f"unusable real {kind} blocks for {row['id']}")
            masks[kind] = sorted(requested)
        video = mouth_tensor(torch.from_numpy(arrays["mouth"]))
        episode = {"id": row["id"], "recording_id": row["recording_id"], "speaker_id": row["speaker_id"],
                   "room_id": row.get("room_id"), "sha256": row["sha256"],
                   "mix": torch.from_numpy(arrays["audio"]), "target": torch.from_numpy(arrays["target"]),
                   "video": video, **masks}
        episodes.append(episode)
    return episodes


@torch.no_grad()
def cache_episodes(episodes, visual, device):
    for ep in episodes:
        video = ep.pop("video")[None].to(device)
        ticks = video.shape[1]
        frames = pad(video, (0, 0, 0, 0, 0, (-ticks) % 3 + 2))
        features, state = [], None
        for start in range(0, ticks, 3):
            feature, state = visual(frames[:, start:start+5], state)
            features.append(feature)
        ep["features"] = torch.cat(features, dim=-1)


def make_schedule(data, episodes, steps):
    by_recording = defaultdict(list)
    for i, episode in enumerate(episodes):
        by_recording[episode["recording_id"]].append(i)
    episode_order = [i for group in zip_longest(*by_recording.values())
                     for i in group if i is not None]
    schedule, real_slots = [], []
    for index, batch in enumerate(data["train_schedule"][:steps]):
        slots = []
        for slot, (a, b) in enumerate(zip(batch[::2], batch[1::2], strict=True)):
            if a[0] != b[0] or a[1] != 0 or b[1] != 1 or a[-1] != b[-1]:
                raise ValueError("expected frozen paired directions")
            kind = "counterlevel" if slot in (0, 1) else "natural"
            level = factorial.LEVELS[(index*2 + slot) % len(factorial.LEVELS)] if kind == "counterlevel" else 0
            slots.append({"row": a[0], "block": a[-1], "kind": kind, "s0_minus_s1_db": level})
        real = None
        if (index+1) % REAL_INTERVAL == 0:
            slots.pop()
            number = len(real_slots)
            j = episode_order[number % len(episode_order)]
            episode = episodes[j]
            pass_number = number // len(episode_order)
            transitions = [(quiet, spoken) for quiet in episode["silent"] for spoken in episode["active"]
                           if spoken > quiet]
            if transitions:
                quiet, spoken = min(transitions, key=lambda pair: (pair[1]-pair[0], pair[0]))
            else:
                quiet = episode["silent"][pass_number % len(episode["silent"])]
                spoken = min(episode["active"], key=lambda block: abs(block-quiet))
            real = {"episode": j, "silent_block": quiet, "active_block": spoken}
            real_slots.append(real)
        schedule.append({"lrs2": slots, "real": real})
    return schedule


def episode_batch(ep, slot, device):
    items = []
    for key in ("silent_block", "active_block"):
        block = slot[key]
        audio, lips = init.window(ep, ep["mix"], 0, block)
        start = block * init.BLOCK_SAMPLES
        reference = ep["target"][start:start+init.BLOCK_SAMPLES]
        items.append((audio, lips, reference, audio[-(init.BLOCK_SAMPLES+init.PREVIEW_SAMPLES):-init.PREVIEW_SAMPLES]))
    return tuple(torch.stack(tensors).to(device) for tensors in zip(*items, strict=True))


def real_quiet_loss(prediction, mixture):
    """Penalize relative output energy in the manually labeled silent block."""
    mixture_power = mixture.float().square().mean().clamp_min(1e-8)
    residual_power = prediction.float().square().mean()
    return 10 / math.log(10) * torch.log1p(residual_power / mixture_power)


def compute_losses(model, lrs2, real, silence_on):
    lrs2_count = len(lrs2[0])
    if real is None:
        estimate = init.forward(model, *lrs2[:2])
        values, _ = scale_sdr_loss(estimate, lrs2[2], lrs2[3])
        return values.mean(), values.mean(), None, None
    audio, lips, reference, mixture = [torch.cat((a,b), dim=0) for a,b in zip(lrs2, real, strict=True)]
    estimate = init.forward(model, audio, lips)
    lrs2_values, _ = scale_sdr_loss(estimate[:lrs2_count], reference[:lrs2_count], mixture[:lrs2_count])
    active, _ = scale_sdr_loss(estimate[-1:], reference[-1:], mixture[-1:])
    quiet = real_quiet_loss(estimate[-2], mixture[-2])
    contribution = quiet if silence_on else quiet.detach().new_zeros(())
    return (lrs2_values.sum()+contribution+active[0])/(lrs2_count+2), lrs2_values.mean(), quiet, active[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-lrs2", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data = init.read_json(args.frozen_lrs2)
    if (not 1 <= args.steps <= len(data["train_schedule"]) or not math.isfinite(args.lr)
        or args.lr <= 0 or (args.smoke and args.steps != REAL_INTERVAL)):
        parser.error("invalid short-run budget, or smoke must include one shared real batch")
    if sha(args.checkpoint) != data["release_sha256"]:
        raise ValueError("published initialization differs from frozen LRS2 comparison")
    torch.set_num_threads(2)
    args.output.mkdir(exist_ok=False)
    episodes = load_episodes(args.episodes)
    schedule = make_schedule(data, episodes, args.steps)
    counts = Counter(row["kind"] for step in schedule for row in step["lrs2"])
    counts["real"] = sum(step["real"] is not None for step in schedule)
    identity = {"source_sha256": sha(__file__), "lrs2_sha256": sha(args.frozen_lrs2),
                "episodes_manifest_sha256": sha(args.episodes), "initial_weights_sha256": sha(args.checkpoint),
                "counts_mixture_slots": dict(counts), "steps": args.steps, "lr": args.lr,
                "history_blocks": init.HISTORY, "episode_count": len(episodes),
                "schedule": schedule,
                "supervision": {"silence_off": "same real quiet input, zero gradient and unchanged denominator",
                                "silence_on": "zero target only for labeled silent scored block",
                                "quiet_loss": "both arms log mixture-relative output, applied only when ON",
                                "active_target": "same unmodified aligned individual target track"}}
    init.save(args.output/"identity.json", identity)
    if args.prepare_only:
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose only the dedicated experiment GPU")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    base, backbone, epoch = load_baseline(args.checkpoint, "cpu")
    base.eval().spk_pool_proj.requires_grad_(False)
    models = {name: copy.deepcopy(base).to(device) for name in ARMS}
    visual = FalconLookaheadVisual(backbone).eval().to(device)
    train, holdout = map(init.load_pairs, (data["train_records"], data["holdout_records"]))
    check_pair = dict(holdout[0], video=holdout[0]["video"].clone())
    init.cache_features(train+holdout, visual, device)
    check_pair["features"] = holdout[0]["features"]
    cache_episodes(episodes, visual, device)
    fixed_visual_hash = init.module_hash(visual)
    model_hashes = {name: init.module_hash(model) for name, model in models.items()}
    if len(set(model_hashes.values())) != 1:
        raise RuntimeError("initial waveform models differ")
    shared_real = next((step["real"] for step in schedule if step["real"] is not None), None)
    if shared_real is None:
        raise ValueError("short experiment must include labeled supervision")
    first_real = episode_batch(episodes[shared_real["episode"]], shared_real, device)
    real_records = []
    for step in schedule:
        slot = step["real"]
        if slot is not None:
            episode = episodes[slot["episode"]]
            _, _, target, mixture = episode_batch(episode, slot, device)
            relative_db = [init.score_ratio(a.square().mean(), b.square().mean())
                           for a,b in zip(target, mixture, strict=True)]
            real_records.append({"episode": episode["id"], "recording_id": episode["recording_id"],
                                 "speaker_id": episode["speaker_id"], **slot,
                                 "track_relative_mixture_db": relative_db})
    init.save(args.output/"initial_models.json", {"published_epoch": epoch, "visual_sha256": fixed_visual_hash,
              "audio_sha256": model_hashes, "first_real_batch_sha256": init.tensor_hash(first_real),
              "real_rows": real_records})
    if not args.smoke:
        baseline = factorial.evaluate(models[next(iter(ARMS))], holdout, data["holdout_third_source"], device, 16)
        init.save(args.output/"baseline.json", baseline)
    for model in models.values():
        for layer in model.modules():
            if isinstance(layer, torch.nn.RNNBase):
                if layer.dropout:
                    raise ValueError("expected dropout-free RNN")
                layer.train()
    optimizers = {name: torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=args.lr)
                  for name,m in models.items()}
    with (args.output/"updates.jsonl").open("x") as log:
        for step, slot in enumerate(schedule, 1):
            lrs2 = {level: factorial.pack(train, slot["lrs2"], level, device) for level in (False,True)}
            real = slot["real"]
            shared = None if real is None else episode_batch(episodes[real["episode"]], real, device)
            for name, model in models.items():
                level, silence = ARMS[name]
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss, mix_loss, quiet_loss, active_loss = compute_losses(model, lrs2[level], shared, silence)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{name}: nonfinite loss at update {step}")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
                optimizer.step()
                log.write(json.dumps({"arm": name, "step": step, "loss": float(loss.detach()),
                                     "lrs2_loss": float(mix_loss.detach()), "gradient_norm": norm,
                                     "clipped": norm > 5,
                                     "real_quiet_loss": None if quiet_loss is None else float(quiet_loss.detach()),
                                     "real_active_loss": None if active_loss is None else float(active_loss.detach())})+"\n")
            log.flush()
            if step % 8 == 0 or step == args.steps:
                print("completed four arms", step, flush=True)
    if init.module_hash(visual) != fixed_visual_hash:
        raise RuntimeError("shared frozen visual model changed")
    for name, model in models.items():
        model.eval()
        torch.save(pack_rolling_checkpoint(model, visual, history_blocks=init.HISTORY,
                                           arm=name, steps=args.steps), args.output/f"{name}.pt")
    init.save(args.output/"checkpoints.json", {name: sha(args.output/f"{name}.pt") for name in ARMS})
    print("four checkpoints saved", flush=True)
    parity = {name: factorial.verify_live(model, check_pair, device, RollingExperimentLiveInference,
                                         args.output/f"{name}.pt") for name,model in models.items()}
    init.save(args.output/"live_parity.json", parity)
    if not args.smoke:
        for name, model in models.items():
            result = factorial.evaluate(model, holdout, data["holdout_third_source"], device, 16)
            init.save(args.output/f"{name}.json", result)
            print(name, json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
