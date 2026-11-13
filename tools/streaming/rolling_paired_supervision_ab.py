"""Compare same-mixture co-batching with the same requests split across updates."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import random
from pathlib import Path
import subprocess

import torch

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from tools.streaming import rolling_init_ab as init
from tools.streaming.evaluate_realssa import load_baseline, sha
from tools.streaming.rolling_factorial_ab import evaluate, verify_live
from tools.streaming.rolling_model_checkpoint import RollingExperimentLiveInference, pack_rolling_checkpoint
from tools.streaming.scale_sdr_loss import scale_sdr_loss


ARMS = ("paired_off", "paired_on")
SEED = init.SEED + 41
REQUESTS_PER_UPDATE = 8


def overlap_blocks(pair):
    power = pair["targets"].reshape(2, -1, init.TICK_SAMPLES).square().mean(-1)
    # Require both clean references to have speech energy in at least two
    # identical ticks. Thresholds are relative to each speaker and absolute.
    threshold = (power.amax(1, keepdim=True) * .01).clamp_min(1e-5)
    joint = (power >= threshold).all(0)
    return [block for block in init.BLOCKS if int(joint[block*3:block*3+3].sum()) >= 2]


def split_updates(rows, blocks, rng):
    # Pick two leads per direction independently in each four-mixture half.
    first, second = [], []
    for i in range(0, REQUESTS_PER_UPDATE, REQUESTS_PER_UPDATE // 2):
        leads = [0, 0, 1, 1]
        rng.shuffle(leads)
        for row, block, lead in zip(rows[i:i+4], blocks[i:i+4], leads, strict=True):
            first.append((row, lead, "natural", 0, block))
            second.append((row, 1-lead, "natural", 0, block))
    on = [[row for pair in zip(first[i:i+4], second[i:i+4], strict=True) for row in pair] for i in (0, 4)]
    off = [first[:4]+second[4:], first[4:]+second[:4]]
    return {"paired_off": [sorted(update) for update in off],
            "paired_on": [sorted(update) for update in on]}


def make_schedule(train, steps):
    rng = random.Random(SEED)
    eligible = {i: blocks for i, pair in enumerate(train) if (blocks := overlap_blocks(pair))}
    if len(eligible) < REQUESTS_PER_UPDATE:
        raise ValueError("insufficient mixtures containing simultaneous speech from both references")
    pending = []
    schedule = []
    for _ in range(0, steps, 2):
        if len(pending) < REQUESTS_PER_UPDATE:
            refilled = [i for i in eligible if i not in pending]
            rng.shuffle(refilled)
            pending.extend(refilled)
        rows = pending[:REQUESTS_PER_UPDATE]
        del pending[:REQUESTS_PER_UPDATE]
        blocks = [rng.choice(eligible[i]) for i in rows]
        schedule.append(split_updates(rows, blocks, rng))
    for item in schedule:
        validate_updates(item)
    return schedule, eligible


def validate_updates(item):
    marginal = {}
    for arm in ARMS:
        updates = item[arm]
        if len(updates) != 2:
            raise ValueError("a matched schedule block must contain two updates")
        marginal[arm] = Counter()
        for update in updates:
            if len(update) != REQUESTS_PER_UPDATE or Counter(r[1] for r in update) != {0: 4, 1: 4}:
                raise ValueError("each update must contain eight balanced directions")
            expected = 2 if arm == "paired_on" else 1
            if set(Counter(r[0] for r in update).values()) != {expected}:
                raise ValueError(f"{arm}: invalid number of requests per mixture")
            for row in update:
                if row[2:4] != ("natural", 0):
                    raise ValueError("training requires unchanged original mixture audio")
            marginal[arm].update(update)
    shared = marginal["paired_on"]
    groups = Counter((row, block) for row, _, _, _, block in shared)
    if (shared != marginal["paired_off"] or set(shared.values()) != {1}
            or set(groups.values()) != {2} or len({row for row, _ in groups}) != REQUESTS_PER_UPDATE):
        raise ValueError("arms must reuse the same two requests for each of eight disjoint mixture windows")


def update_model(model, optimizer, batch, arm, step):
    audio, mouth, target, mixed = batch
    optimizer.zero_grad(set_to_none=True)
    prediction = init.forward(model, audio, mouth)
    values, masks = scale_sdr_loss(prediction, target, mixed)
    loss = values.mean()
    if not bool(torch.isfinite(loss)) or not bool(masks["speech"].all()):
        raise RuntimeError(f"{arm}: invalid speech loss at update {step}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return {"arm": arm, "step": step, "loss": float(loss.detach()),
            "gradient_norm": float(norm), "gradient_clipped": bool(norm > 5.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True, help="data_identity.json from the original initialization comparison")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.steps <= 96 or args.steps % 2 or (args.smoke and args.steps != 2):
        parser.error("use 2 to 96 even updates per arm and exactly two for smoke")
    torch.set_num_threads(2)
    data = init.read_json(args.frozen)
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != data["release_sha256"]:
        raise ValueError("release checkpoint differs from the frozen experiment")
    train, holdout = map(init.load_pairs, (data["train_records"], data["holdout_records"]))
    schedule, eligible = make_schedule(train, args.steps)
    args.output.mkdir(exist_ok=False)
    code_paths = [Path(__file__), Path(init.__file__), Path(__file__).with_name("rolling_falcon_live.py")]
    identity = {
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {path.name: sha(path) for path in code_paths},
        "frozen_identity_sha256": sha(args.frozen),
        "checkpoint_sha256": checkpoint_sha,
        "steps_per_arm": args.steps,
        "requests_per_update": REQUESTS_PER_UPDATE,
        "requests_per_arm": args.steps * REQUESTS_PER_UPDATE,
        "seen_train_mixtures": len({row[0] for block in schedule for update in block["paired_on"] for row in update}),
        "history_ms": init.HISTORY * 120,
        "output_ms": 120,
        "preview_ms": 80,
        "learning_rate": 3e-4,
        "gradient_clip_norm": 5.,
        "seed": SEED,
        "eligible_train_blocks": eligible,
        "source_train_mixtures": len(train),
        "schedule": schedule,
    }
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
    models = {arm: copy.deepcopy(base).to(device) for arm in ARMS}
    visual = FalconLookaheadVisual(backbone).eval().to(device)
    visual_hash = init.module_hash(visual)
    buffer_hash = init.tensor_hash(list(base.buffers()))
    check_pair = dict(holdout[0])
    init.cache_features(train+holdout, visual, device)
    check_pair["features"] = holdout[0]["features"]
    model_hash = init.module_hash(base)
    if any(init.module_hash(model) != model_hash for model in models.values()):
        raise RuntimeError("arms do not share the original model state")
    # Hash actual per-request tensors for the first two updates, after feature caching.
    replay, predictions = {}, {}
    for arm in ARMS:
        replay[arm], predictions[arm] = {}, {}
        for update in schedule[0][arm]:
            batch = init.pack(train, None, update, device)
            replay[arm].update({row: init.tensor_hash([x[i] for x in batch]) for i, row in enumerate(update)})
            with torch.no_grad():
                predictions[arm].update(zip(update, init.forward(models[arm], *batch[:2]), strict=True))
            if arm == "paired_on" and (not torch.equal(batch[0][::2], batch[0][1::2])
                                        or not torch.equal(batch[3][::2], batch[3][1::2])):
                raise RuntimeError("paired requests did not receive the same mixture")
    if replay[ARMS[0]] != replay[ARMS[1]]:
        raise RuntimeError("the two-update matched requests are not byte-identical")
    prediction_max_difference = 0.
    for row, waveform in predictions[ARMS[0]].items():
        paired = predictions[ARMS[1]][row]
        torch.testing.assert_close(waveform, paired, atol=2e-5, rtol=2e-5)
        prediction_max_difference = max(prediction_max_difference, float((waveform-paired).abs().max()))
    init.save(args.output/"model_identity.json", {
        "checkpoint_epoch": epoch, "both_arms_initial_model_sha256": model_hash,
        "shared_frozen_visual_sha256": visual_hash,
        "trainable_parameters": sum(p.numel() for p in base.parameters() if p.requires_grad),
        "first_two_updates_shared_request_sha256": {str(row): hashes for row, hashes in replay[ARMS[0]].items()},
        "first_two_updates_initial_waveform_max_abs_difference": prediction_max_difference,
        "torch": str(torch.__version__), "device": torch.cuda.get_device_name(), "tf32": False,
    })
    if not args.smoke:
        initial = evaluate(models[ARMS[0]], holdout, data["holdout_third_source"], device, 16)
        init.save(args.output/"baseline.json", initial)
        print("initial LRS2 evaluation saved", flush=True)
    for model in models.values():
        for layer in model.modules():
            if isinstance(layer, torch.nn.RNNBase):
                if layer.dropout:
                    raise ValueError("expected dropout-free RNN")
                layer.train()
    optimizers = {arm: torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
                  for arm, model in models.items()}
    clipping_counts = Counter({arm: 0 for arm in ARMS})
    with (args.output/"updates.jsonl").open("x") as log:
        for i, item in enumerate(schedule):
            for j in range(2):
                step = 2*i + j + 1
                # Alternate GPU execution order, keeping both arms in lockstep.
                for arm in (ARMS if j == 0 else reversed(ARMS)):
                    batch = init.pack(train, None, item[arm][j], device)
                    result = update_model(models[arm], optimizers[arm], batch, arm, step)
                    clipping_counts[arm] += result["gradient_clipped"]
                    log.write(json.dumps(result, allow_nan=False) + "\n")
                log.flush()
                if step % 8 == 0 or step == args.steps:
                    print("completed both arms", step, flush=True)
    init.save(args.output/"gradient_clipping_counts.json", dict(clipping_counts))
    if init.module_hash(visual) != visual_hash:
        raise RuntimeError("common frozen visual encoder changed")
    checkpoints = {}
    for arm, model in models.items():
        model.eval()
        if init.tensor_hash(list(model.buffers())) != buffer_hash:
            raise RuntimeError("a fixed waveform buffer changed")
        path = args.output/f"{arm}.pt"
        torch.save(pack_rolling_checkpoint(model, visual, history_blocks=init.HISTORY,
                                           arm=arm, steps=args.steps), path)
        checkpoints[arm] = sha(path)
    init.save(args.output/"checkpoints.json", checkpoints)
    print("both checkpoints saved", flush=True)
    parity = {arm: verify_live(model, check_pair, device, RollingExperimentLiveInference, args.output/f"{arm}.pt")
              for arm, model in models.items()}
    init.save(args.output/"live_parity.json", parity)
    if not args.smoke:
        for arm, model in models.items():
            result = evaluate(model, holdout, data["holdout_third_source"], device, 16)
            init.save(args.output/f"{arm}.json", result)
            print(arm, "LRS2 evaluation saved", flush=True)


if __name__ == "__main__":
    main()
