"""Matched-budget 2x2 rolling Falcon structure and training-data experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path

import torch

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from tools.streaming import rolling_init_ab as init
from tools.streaming.evaluate_realssa import load_baseline, sha
from tools.streaming.scale_sdr_loss import scale_sdr_loss

ARMS = ("base_natural", "base_augmented", "conditioned_natural", "conditioned_augmented")
LEVELS = (0, -6, 6, -12, 12)
SILENCE_RULE = {"target_relative_peak_db": -30, "target_relative_other_db": -25,
                "other_power_min": 1e-6, "mouth_motion_quantile": .55}
FLIPS = {"flip_s0_plus6": 6, "flip_s0_minus6": -6,
         "flip_s0_plus12": 12, "flip_s0_minus12": -12}


def silent_slots(pairs):
    """Return visible target-quiet candidates with aligned acoustic and motion checks."""
    records = []
    rule = SILENCE_RULE
    for row, pair in enumerate(pairs):
        for direction in (0, 1):
            target = pair["targets"][direction]
            other = pair["targets"][1-direction]
            frames = pair["video"][direction]
            motion = (frames[1:] - frames[:-1]).square().mean((1, 2)).sqrt()
            motion_limit = float(motion.quantile(rule["mouth_motion_quantile"]))
            peak = float(target.reshape(-1, init.TICK_SAMPLES).square().mean(-1).max())
            for block in init.BLOCKS:
                lo, hi = block*init.BLOCK_SAMPLES, (block+1)*init.BLOCK_SAMPLES
                a, b = float(target[lo:hi].square().mean()), float(other[lo:hi].square().mean())
                movement = float(motion[3*block-1:3*block+2].median())
                if (b > rule["other_power_min"] and
                    a <= peak * 10 ** (rule["target_relative_peak_db"] / 10) and
                    a <= b * 10 ** (rule["target_relative_other_db"] / 10) and
                    movement <= motion_limit):
                    records.append({"row": row, "block": block, "direction": direction,
                                    "target_power": a, "other_power": b, "target_peak": peak,
                                    "mouth_motion": movement, "mouth_motion_limit": motion_limit})
    return records


def schedule_from_frozen(data, silent, steps):
    output = []
    if not silent:
        raise ValueError("no conservatively identified visible target-quiet training block")
    for step, batch in enumerate(data["train_schedule"][:steps]):
        rows = []
        for slot, (first, second) in enumerate(zip(batch[::2], batch[1::2], strict=True)):
            if first[:2] != [second[0], 0] or second[1] != 1 or first[-1] != second[-1]:
                raise ValueError("expected fixed paired target requests")
            position = step * (len(batch)//2) + slot
            row, block = first[0], first[-1]
            kind, level, quiet_direction = "natural", 0, None
            if position % 16 == 15:
                chosen = silent[(position//16) % len(silent)]
                row, block, quiet_direction = chosen["row"], chosen["block"], chosen["direction"]
                kind = "visible_quiet"
            elif position % 4 in (0, 1):
                kind = "counterlevel"
                level = LEVELS[(position//4*2+position%4) % len(LEVELS)]
            rows.append({"row": row, "block": block, "kind": kind, "s0_minus_s1_db": level,
                         "quiet_direction": quiet_direction})
        output.append(rows)
    return output


def shared_level(pair, level):
    """RMS-control both sources once so both target requests see the identical mixture."""
    targets = pair["targets"]
    rms = targets.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    # Keep the geometric mean level fixed before one common mixture-level correction.
    gains = rms.prod().sqrt() / rms * targets.new_tensor([[10 ** (level/40)], [10 ** (-level/40)]])
    targets = targets * gains
    mix = targets.sum(0)
    common = min(float(pair["mix"].square().mean().sqrt() / mix.square().mean().sqrt().clamp_min(1e-8)),
                 .98 / float(mix.abs().amax().clamp_min(1e-8)))
    return mix * common, targets * common


def paired_training_audio(pair, slot, augmented):
    if not augmented or slot["kind"] == "natural":
        return pair["mix"], pair["targets"]
    if slot["kind"] == "counterlevel":
        return shared_level(pair, slot["s0_minus_s1_db"])
    if slot["kind"] != "visible_quiet":
        raise ValueError("unknown paired fixture")
    targets = pair["targets"].clone()
    start = slot["block"] * init.BLOCK_SAMPLES
    stop = start + init.BLOCK_SAMPLES
    targets[slot["quiet_direction"], start:stop] = 0
    mix = pair["mix"].clone()
    mix[start:stop] = targets[:, start:stop].sum(0)
    return mix, targets


def pack(pairs, slots, augmented, device):
    audios, features, targets = [], [], []
    for slot in slots:
        pair = pairs[slot["row"]]
        mixture, references = paired_training_audio(pair, slot, augmented)
        block = slot["block"]
        start = block * init.BLOCK_SAMPLES
        for direction in (0, 1):
            audio, mouth = init.window(pair, mixture, direction, block)
            audios.append(audio)
            features.append(mouth)
            targets.append(references[direction, start:start+init.BLOCK_SAMPLES])
    audio = torch.stack(audios).to(device)
    mouth = torch.stack(features)
    target = torch.stack(targets).to(device)
    if not torch.equal(audio[::2], audio[1::2]):
        raise RuntimeError("the two requested targets must receive the same mixture")
    return audio, mouth, target, audio[:, -(init.BLOCK_SAMPLES+init.PREVIEW_SAMPLES):-init.PREVIEW_SAMPLES]


def summary_rows(scores):
    summary = {}
    for kind, rows in scores.items():
        if kind.startswith("flip_both"):
            summary[kind] = {"pairs": len(rows), "both_polarities_correct": sum(
                row["both_polarities_correct"] for row in rows)}
            continue
        fields = set(rows[0]) - {"row", "direction", "start_block"}
        mean = {k: sum(row[k] for row in rows)/len(rows) for k in sorted(fields)}
        if "target_preferred" in fields:
            mean["correct_directions"] = sum(row["target_preferred"] for row in rows)
            mean["both_directions_correct"] = sum(a["target_preferred"] and b["target_preferred"]
                                                  for a,b in zip(rows[::2], rows[1::2], strict=True))
        summary[kind] = {"directions": len(rows), **mean}
    return summary


@torch.no_grad()
def evaluate(model, holdout, third, device, batch_windows):
    cases = []
    for kind in (*init.KINDS, *FLIPS):
        for row, pair in enumerate(holdout):
            paired = shared_level(pair, FLIPS[kind]) if kind in FLIPS else None
            for direction in (0, 1):
                if paired is None:
                    gap = {"weak6": -6, "weak12": -12}.get(kind, 0)
                    mixture, target, near = init.fixture(holdout, third, row, direction, kind, gap)
                else:
                    mixture, refs = paired
                    target, near = refs[direction], [refs[1-direction]]
                cases.append((kind, row, direction, mixture, target, near, pair["span"]))
    scores = defaultdict(list)
    for offset in range(0, len(cases), 12):
        batch = cases[offset:offset+12]
        windows = [init.window(holdout[row], mixture, d, block)
                   for _,row,d,mixture,_,_,start in batch for block in range(start,start+init.SCORED_BLOCKS)]
        audio, video = map(torch.stack, zip(*windows))
        output = torch.cat([init.forward(model, audio[i:i+batch_windows].to(device), video[i:i+batch_windows]).cpu()
                            for i in range(0, len(audio), batch_windows)]).reshape(len(batch), -1)
        for wave, (kind,row,d,mixture,target,near,start) in zip(output,batch,strict=True):
            span = slice(start*init.BLOCK_SAMPLES,(start+init.SCORED_BLOCKS)*init.BLOCK_SAMPLES)
            scores[kind].append({"row": row, "direction": d, "start_block": start,
                                 **init.score_case(wave, mixture[span], target[span], [x[span] for x in near])})
    for gap in (6, 12):
        plus, minus = scores[f"flip_s0_plus{gap}"], scores[f"flip_s0_minus{gap}"]
        for i in range(0, len(plus), 2):
            four = (*plus[i:i+2], *minus[i:i+2])
            if len({r["row"] for r in four}) != 1:
                raise RuntimeError("different mixtures in flip polarity pairing")
            scores[f"flip_both{gap}"].append({"row": four[0]["row"], "both_polarities_correct":
                                              all(r["target_preferred"] for r in four)})
    return {"summary": summary_rows(scores), "rows": scores}


def verify_live(model, pair, device, live_class, checkpoint, solo_direction=0):
    import numpy as np
    engine = live_class(checkpoint, device=str(device), threads=2)
    pieces = []
    mix = pair["mix"].numpy()
    mouth = pair["video"][solo_direction].numpy()
    for i, frame in enumerate(mouth):
        pieces.extend(engine.push(mix[i*init.TICK_SAMPLES:(i+1)*init.TICK_SAMPLES], frame, sequence=i))
    pieces.extend(engine.finish())
    actual = torch.from_numpy(np.concatenate(pieces))
    block = pair["span"]
    a, f = init.window(pair, pair["mix"], solo_direction, block)
    with torch.no_grad():
        expected = init.forward(model, a[None].to(device), f[None])[0].cpu()
    start = block*init.BLOCK_SAMPLES
    cut = actual[start:start+init.BLOCK_SAMPLES]
    torch.testing.assert_close(cut, expected, atol=2e-5, rtol=2e-5)
    return float((cut-expected).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True, help="previous data_identity.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batch-windows", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="validate four first updates and checkpoint/live parity")
    args = parser.parse_args()
    data = init.read_json(args.frozen)
    if not 1 <= args.steps <= len(data["train_schedule"]) or not 1 <= args.eval_batch_windows <= 32 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("step count or evaluation batch is outside the bounded protocol")
    if sha(args.checkpoint) != data["release_sha256"]:
        raise ValueError("release checkpoint differs from the frozen experiment")
    if args.smoke and args.steps != 1:
        parser.error("smoke requires exactly one optimizer update per arm")
    torch.set_num_threads(2)
    args.output.mkdir(exist_ok=False)
    train, holdout = map(init.load_pairs, (data["train_records"], data["holdout_records"]))
    silent = silent_slots(train)
    schedule = schedule_from_frozen(data, silent, args.steps)
    identity = {"source_sha256": sha(__file__), "frozen_identity_sha256": sha(args.frozen),
                "checkpoint_sha256": sha(args.checkpoint), "recipe": {
                    "history_blocks": init.HISTORY, "steps": args.steps, "lr": args.lr,
                    "train_pairs": len(train), "heldout_pairs": len(holdout),
                    "slots_per_update": len(schedule[0]), "quiet_rule": SILENCE_RULE,
                    "augmented_pair_slots": dict(Counter(r["kind"] for b in schedule for r in b)),
                    "holdout_real_used": False},
                "quiet_candidates": silent, "schedule": schedule}
    init.save(args.output/"identity.json", identity)
    if args.prepare_only:
        return
    from tools.streaming.rolling_model_checkpoint import (
        RollingExperimentLiveInference, make_frequency_conditioned, pack_rolling_checkpoint,
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose only the dedicated experiment GPU")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    base, backbone, _ = load_baseline(args.checkpoint, "cpu")
    base.eval().spk_pool_proj.requires_grad_(False)
    conditioned = make_frequency_conditioned(base, seed=init.SEED)
    models = {name: copy.deepcopy(conditioned if name.startswith("conditioned") else base).to(device)
              for name in ARMS}
    visual = FalconLookaheadVisual(backbone).eval().to(device)
    check_pair = dict(holdout[0], video=holdout[0]["video"].clone())
    init.cache_features(train+holdout, visual, device)
    check_pair["features"] = holdout[0]["features"]
    batches = {r: pack(train, schedule[0], r == "augmented", device)
               for r in ("natural", "augmented")}
    with torch.no_grad():
        for batch in batches.values():
            reference = init.forward(models["base_natural"], *batch[:2])
            candidate = init.forward(models["conditioned_natural"], *batch[:2])
            torch.testing.assert_close(candidate, reference, atol=2e-6, rtol=2e-6)
    init.save(args.output/"initial_models.json", {
        "model_sha256": {name: init.module_hash(model) for name,model in models.items()},
        "visual_sha256": init.module_hash(visual),
        "trainable_parameters": {name: sum(p.numel() for p in model.parameters() if p.requires_grad)
                                 for name,model in models.items()},
        "first_batch_sha256": {r: init.tensor_hash(batch) for r,batch in batches.items()}})
    if not args.smoke:
        baseline = evaluate(models["base_natural"], holdout, data["holdout_third_source"],
                            device, args.eval_batch_windows)
        init.save(args.output/"baseline.json", baseline)
        print("shared starting baseline", json.dumps(baseline["summary"]), flush=True)
    for model in models.values():
        for layer in model.modules():
            if isinstance(layer, torch.nn.RNNBase):
                if layer.dropout:
                    raise ValueError("expected dropout-free RNN")
                layer.train()
    optimizers = {name: torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
                  for name,model in models.items()}
    with (args.output/"updates.jsonl").open("x") as log:
        for step, slots in enumerate(schedule, 1):
            batches = {r: pack(train, slots, r == "augmented", device)
                       for r in ("natural", "augmented")}
            for name, model in models.items():
                audio, mouth, target, mixed = batches[name.rsplit("_", 1)[1]]
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                prediction = init.forward(model, audio, mouth)
                values, masks = scale_sdr_loss(prediction, target, mixed)
                loss = values.mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"{name}: nonfinite loss at update {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps({"arm":name,"step":step,"loss":float(loss.detach()),
                                      "gradient_norm":float(norm),
                                      "silent_rows":int(masks["silent"].sum())})+"\n")
            log.flush()
            if step % 8 == 0 or step == args.steps:
                print("completed four arms", step, flush=True)
    for name, model in models.items():
        model.eval()
        checkpoint = args.output/f"{name}.pt"
        torch.save(pack_rolling_checkpoint(model.cpu(), visual, history_blocks=init.HISTORY,
                                           arm=name, steps=args.steps), checkpoint)
        model.to(device)
    init.save(args.output/"checkpoints.json", {name: sha(args.output/f"{name}.pt") for name in models})
    print("four checkpoints saved", flush=True)
    parity = {name: verify_live(model, check_pair, device, RollingExperimentLiveInference,
                                args.output/f"{name}.pt") for name,model in models.items()}
    init.save(args.output/"live_parity.json", parity)
    if not args.smoke:
        for name, model in models.items():
            result = evaluate(model, holdout, data["holdout_third_source"], device, args.eval_batch_windows)
            init.save(args.output/f"{name}.json", result)
            print(name, json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
