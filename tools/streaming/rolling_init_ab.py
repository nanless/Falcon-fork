"""Fixed-budget Falcon Small rolling initialization comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import torch
from torch.nn.functional import pad

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from tools.streaming.common import load_mono, mouth_tensor
from tools.streaming.evaluate_realssa import load_baseline, sha
from tools.streaming.rolling_falcon_live import BLOCK_SAMPLES, PREVIEW_SAMPLES, TICK_SAMPLES
from tools.streaming.scale_sdr_loss import scale_sdr_loss

HISTORY = 5
TOTAL = 32000
SCORED_BLOCKS = 5
BLOCKS = range(HISTORY, TOTAL // BLOCK_SAMPLES)
SEED = 20261113
LEVELS = (0, -6, 6, -12, 12)
KINDS = ("natural", "balanced", "weak6", "weak12", "single", "two", "mute_target")


def save(path, content):
    with path.open("x") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def record_from_paths(paths):
    return {key: [sha(path) for path in values] if isinstance(values, list) else sha(values)
            for key, values in paths.items()}


def train_records(directory, count, excluded_parents):
    mixes, targets = (read_json(directory / (kind + ".json")) for kind in ("mix", "target"))
    grouped = defaultdict(dict)
    for (mix, n), (target, mouth, m) in zip(mixes, targets, strict=True):
        if n != TOTAL or m != TOTAL:
            continue
        path = Path(target)
        if path.name != Path(mix).name or path.parent.name not in ("s1", "s2"):
            raise ValueError("invalid LRS2 directional index")
        direction = path.parent.name
        if direction in grouped[mix]:
            raise ValueError("duplicate mixture direction")
        grouped[mix][direction] = (target, mouth)
    candidates = sorted(grouped)
    random.Random(SEED).shuffle(candidates)
    selected = []
    for mix in candidates:
        group = grouped[mix]
        if set(group) != {"s1", "s2"}:
            continue
        targets, mouths = map(list, zip(group["s1"], group["s2"]))
        parents = [Path(mouth).stem.rsplit("_", 1)[0] for mouth in mouths]
        if len(set(parents)) != 2 or not set(parents).isdisjoint(excluded_parents):
            continue
        paths = {"mix": mix, "targets": targets, "mouths": mouths}
        selected.append({"id": Path(mix).stem, "paths": paths, "parent_video_ids": parents,
                         "sha256": record_from_paths(paths)})
        if len(selected) == count:
            return selected
    raise ValueError("insufficient training mixtures with disjoint video parents")


def third_sources(records):
    rng = random.Random(SEED + 2)
    result = []
    for row in records:
        candidates = [j for j, other in enumerate(records)
                      if set(row["parent_video_ids"]).isdisjoint(other["parent_video_ids"])]
        if not candidates:
            raise ValueError("no independent third-source parent in this split")
        result.append([rng.choice(candidates), rng.randrange(2)])
    return result


def choose_span(targets):
    power = targets.reshape(2, -1, TICK_SAMPLES).square().mean(-1)
    active = power > power.amax(1, keepdim=True) * .01
    return max(range(HISTORY, BLOCKS.stop - SCORED_BLOCKS + 1),
               key=lambda b: int(active[:, 3*b:3*(b+SCORED_BLOCKS)].all(0).sum()))


def load_pairs(records):
    pairs = []
    for row in records:
        paths = row["paths"]
        if record_from_paths(paths) != row["sha256"]:
            raise ValueError("mixture, reference or mouth hash changed")
        mixture = load_mono(paths["mix"])
        targets = torch.stack([load_mono(p) for p in paths["targets"]])
        if mixture.shape != (TOTAL,) or targets.shape != (2, TOTAL):
            raise ValueError("unexpected LRS2 length")
        if (mixture - targets.sum(0)).abs().max() > 3 / 32768:
            raise ValueError("reference sources do not sum to the mixture")
        video = torch.stack([mouth_tensor(torch.load(p, weights_only=True, map_location="cpu"))
                             for p in paths["mouths"]])
        if video.shape != (2, 50, 88, 88):
            raise ValueError("unexpected mouth shape")
        pairs.append({"mix": mixture, "targets": targets, "video": video,
                      "span": choose_span(targets)})
    return pairs


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def cache_features(pairs, visual, device):
    for pair in pairs:
        frames = pad(pair.pop("video"), (0, 0, 0, 0, 0, 3)).to(device)
        context = None
        pieces = []
        for start in range(0, 50, 3):
            feature, context = visual(frames[:, start:start+5], context)
            pieces.append(feature)
        pair["features"] = torch.cat(pieces, dim=-1)


def fixture(pairs, third, row, direction, kind, gap=0):
    pair = pairs[row]
    target, other = pair["targets"][direction], pair["targets"][1-direction]
    references = []
    if kind == "natural":
        return pair["mix"], target, [other]
    if kind in ("balanced", "weak6", "weak12", "scaled"):
        target_power, other_power = [x.square().mean().sqrt().clamp_min(1e-8) for x in (target, other)]
        other = other * (target_power / other_power) * 10 ** (-gap / 20)
        references = [other]
    elif kind == "two":
        j, d = third[row]
        extra = pairs[j]["targets"][d]
        extra = extra * (other.square().mean() / extra.square().mean().clamp_min(1e-12)).sqrt()
        references = [other, extra]
    elif kind == "mute_target":
        references = [other]
        target = torch.zeros_like(target)
    elif kind != "single":
        raise ValueError(f"unknown fixture {kind}")
    mixture = target + sum(references)
    gain = min(1., .98 / float(mixture.abs().amax().clamp_min(1e-8)))
    return mixture * gain, target * gain, [x * gain for x in references]


def window(pair, mixture, direction, block):
    begin = (block - HISTORY) * BLOCK_SAMPLES
    end = (block + 1) * BLOCK_SAMPLES
    frames = pair["features"][direction, :, begin//TICK_SAMPLES:end//TICK_SAMPLES]
    held = frames[..., -1:].expand(-1, 2)
    return mixture[begin:end+PREVIEW_SAMPLES], torch.cat((frames, held), dim=-1)


def forward(model, audio, visual):
    estimate = model(audio, visual)
    if estimate.shape != (audio.shape[0], 1, audio.shape[-1]):
        raise ValueError("expected one output waveform per target request")
    return estimate[:, 0, -(BLOCK_SAMPLES+PREVIEW_SAMPLES):-PREVIEW_SAMPLES]


def init_models(checkpoint, device):
    released, backbone, epoch = load_baseline(checkpoint, "cpu")
    kwargs = released.get_model_args()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        scratch = type(released)(**kwargs)
    if released.get_model_args() != scratch.get_model_args():
        raise ValueError("model constructors differ")
    if list(released.state_dict()) != list(scratch.state_dict()):
        raise ValueError("model state keys differ")
    if kwargs["n_src"] != 1:
        raise ValueError("short experiment requires one target output")
    rbuf, sbuf = dict(released.named_buffers()), dict(scratch.named_buffers())
    if set(rbuf) != set(sbuf) or any(rbuf[k].shape != sbuf[k].shape for k in rbuf):
        raise ValueError("initialization changed fixed buffer geometry")
    for key, buffer in sbuf.items():
        buffer.copy_(rbuf[key])
    for model in (released, scratch):
        model.spk_pool_proj.requires_grad_(False)
        model.eval().to(device)
    pairs = list(zip(released.named_parameters(), scratch.named_parameters(), strict=True))
    if any((name, a.shape, a.requires_grad) != (other, b.shape, b.requires_grad)
           for (name, a), (other, b) in pairs):
        raise ValueError("trainable parameter signature differs")
    if not any(not torch.equal(a,b) for (_, a), (_, b) in pairs if a.requires_grad):
        raise ValueError("initialization arms have identical trainable parameters")
    shared_buffers = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in rbuf.items()}
    visual = FalconLookaheadVisual(backbone).eval().to(device)
    identity = {"epoch": epoch, "config": kwargs, "same_fixed_buffer_sha256": shared_buffers,
                "shared_frozen_frcnn_sha256": module_hash(visual),
                "initial_model_sha256": {"released": module_hash(released), "scratch": module_hash(scratch)},
                "waveform_trainable_parameters": sum(p.numel() for p in released.parameters() if p.requires_grad)}
    return {"released": released, "scratch": scratch}, visual, identity


def training_schedule(train, steps, batch_pairs):
    rng = random.Random(SEED + 5)
    schedule = []
    for step in range(steps):
        rows = []
        for slot, idx in enumerate(rng.sample(range(len(train)), batch_pairs)):
            kind = ("scaled", "scaled", "two", "single", "mute_target", "natural")[(step+slot) % 6]
            block = train[idx]["span"] + rng.randrange(SCORED_BLOCKS)
            for d in (0, 1):
                gap = LEVELS[(step * batch_pairs+slot) % len(LEVELS)] * (1 if d == 0 else -1)
                rows.append((idx, d, kind, gap, block))
        schedule.append(rows)
    return schedule


def pack(pairs, third, rows, device):
    audios, videos, targets, mixtures = [], [], [], []
    for row, direction, kind, gap, block in rows:
        mixture, target, _ = fixture(pairs, third, row, direction, kind, gap)
        audio, visual = window(pairs[row], mixture, direction, block)
        audios.append(audio)
        videos.append(visual)
        start = block * BLOCK_SAMPLES
        targets.append(target[start:start+BLOCK_SAMPLES])
        mixtures.append(mixture[start:start+BLOCK_SAMPLES])
    return (torch.stack(audios).to(device), torch.stack(videos), torch.stack(targets).to(device),
            torch.stack(mixtures).to(device))


def tensor_hash(items):
    return [hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for x in items]


def score_ratio(numerator, denominator):
    return float(10 * (numerator.double().clamp_min(1e-12) / denominator.double().clamp_min(1e-12)).log10())


def score_si(estimate, reference):
    est = estimate.double() - estimate.double().mean()
    ref = reference.double() - reference.double().mean()
    projection = ref * (est.dot(ref) / ref.square().sum().clamp_min(1e-12))
    return score_ratio(projection.square().mean(), (est - projection).square().mean())


def score_case(estimate, mix, target, near):
    if not torch.isfinite(estimate).all():
        raise ValueError("nonfinite evaluation output")
    if not target.any():
        return {"leakage_db": score_ratio(estimate.square().mean(), mix.square().mean())}
    direct = score_ratio(target.square().mean(), (estimate - target).square().mean())
    if not near:
        return {"snr_db": direct}
    target_score = score_si(estimate, target)
    margin = target_score - max(score_si(estimate, x) for x in near)
    return {"snr_db": direct, "si_sdri_db": target_score - score_si(mix, target),
            "selection_margin_db": margin, "target_preferred": margin > 0}


@torch.no_grad()
def evaluate(model, holdout, third, device):
    cases = []
    for kind in KINDS:
        for row, pair in enumerate(holdout):
            for direction in (0, 1):
                gap = {"weak6": -6, "weak12": -12}.get(kind, 0)
                mixture, target, near = fixture(holdout, third, row, direction, kind, gap)
                cases.append((kind, row, direction, mixture, target, near, pair["span"]))
    scores = defaultdict(list)
    for offset in range(0, len(cases), 16):
        batch = cases[offset:offset+16]
        windows = [window(holdout[row], mixture, d, block)
                   for _,row,d,mixture,_,_,start in batch for block in range(start,start+SCORED_BLOCKS)]
        audio, video = map(torch.stack, zip(*windows))
        output = torch.cat([forward(model, audio[i:i+16].to(device), video[i:i+16]).cpu()
                            for i in range(0, len(audio), 16)]).reshape(len(batch), -1)
        for wave, (kind,row,d,mixture,target,near,start) in zip(output,batch,strict=True):
            span = slice(start*BLOCK_SAMPLES,(start+SCORED_BLOCKS)*BLOCK_SAMPLES)
            scores[kind].append({"row": row, "direction": d, "start_block": start,
                                 **score_case(wave,mixture[span],target[span],[x[span] for x in near])})
    summary = {}
    for kind, rows in scores.items():
        fields = set.intersection(*(set(r) for r in rows)) - {"row","direction","start_block"}
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in sorted(fields)}
        if "target_preferred" in fields:
            mean["correct_directions"] = sum(r["target_preferred"] for r in rows)
            mean["both_directions_correct"] = sum(a["target_preferred"] and b["target_preferred"]
                                                  for a,b in zip(rows[::2],rows[1::2],strict=True))
        summary[kind] = {"directions": len(rows), **mean}
    return {"summary": summary, "rows": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=120)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.steps <= 256 and 2 <= args.batch_pairs <= args.train_pairs <= 512
            and math.isfinite(args.lr) and args.lr > 0):
        parser.error("invalid bounded experiment budget")
    torch.set_num_threads(2)
    args.output.mkdir(exist_ok=False)
    holdout_rows = read_json(args.holdout)["records"]
    train_rows = train_records(args.train_index, args.train_pairs,
                               {p for r in holdout_rows for p in r["parent_video_ids"]})
    if {r["id"] for r in train_rows} & {r["id"] for r in holdout_rows}:
        raise ValueError("training and holdout mixtures overlap")
    train, holdout = map(load_pairs, (train_rows, holdout_rows))
    train_third, holdout_third = map(third_sources, (train_rows, holdout_rows))
    schedule = training_schedule(train, args.steps, args.batch_pairs)
    identity = {"repo_head": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "source_sha256": sha(__file__), "train_index_sha256":
                {name: sha(args.train_index / name) for name in ("mix.json","target.json")},
                "holdout_manifest_sha256": sha(args.holdout), "release_sha256": sha(args.checkpoint),
                "train_records": train_rows, "holdout_records": holdout_rows,
                "train_third_source": train_third, "holdout_third_source": holdout_third,
                "train_schedule": schedule, "holdout_starts": [r["span"] for r in holdout],
                "history_blocks": HISTORY, "scored_blocks": SCORED_BLOCKS, "seed": SEED,
                "steps": args.steps, "lr": args.lr, "batch_pairs": args.batch_pairs,
                "torch": str(torch.__version__)}
    save(args.output/"data_identity.json", identity)
    if args.prepare_only:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose only the dedicated A800 for this experiment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    models, visual, model_identity = init_models(args.checkpoint, device)
    model_identity["device"] = torch.cuda.get_device_name()
    cache_features(train+holdout, visual, device)
    save(args.output/"model_identity.json", model_identity)
    first = pack(train,train_third,schedule[0],device)
    first_identity = tensor_hash(first)
    optimizers = {name: torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
                  for name,model in models.items()}
    with torch.no_grad():
        for name,model in models.items():
            replay = pack(train,train_third,schedule[0],device)
            if tensor_hash(replay) != first_identity:
                raise ValueError("same frozen batch replay changed")
            prediction = forward(model,*replay[:2])
            repeated = forward(model,*first[:2])
            torch.testing.assert_close(prediction,repeated,atol=3e-6,rtol=3e-6)
            isolated = forward(model,first[0][:1],first[1][:1])
            torch.testing.assert_close(prediction[:1],isolated,atol=2e-5,rtol=2e-5)
    save(args.output/"replay.json", {"batch_tensor_sha256_both_arms":first_identity,
                                     "same_inputs_repeat_and_first_direction_isolated":True})
    for name,model in models.items():
        baseline = evaluate(model,holdout,holdout_third,device)
        save(args.output/f"{name}_before.json",baseline)
        print(name,"before",json.dumps(baseline["summary"]),flush=True)
    # cuDNN recurrent backward needs training mode. Only enable it on the
    # dropout-free recurrent layers; every other layer keeps eval behavior.
    for model in models.values():
        for layer in model.modules():
            if isinstance(layer, torch.nn.RNNBase):
                if layer.dropout:
                    raise ValueError("comparison expects dropout-free recurrent layers")
                layer.train()
    with (args.output/"updates.jsonl").open("x") as updates:
        for step, rows in enumerate(schedule, 1):
            batch = pack(train,train_third,rows,device)
            for name,model in models.items():
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(model,*batch[:2])
                values,_ = scale_sdr_loss(prediction,batch[2],batch[3])
                loss = values.mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"{name}: nonfinite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                updates.write(json.dumps({"arm":name,"step":step,"loss":float(loss.detach()),
                                          "gradient_norm":float(norm)})+"\n")
            updates.flush()
            if step % 8 == 0 or step == args.steps:
                print("completed both arms",step,flush=True)
    if module_hash(visual) != model_identity["shared_frozen_frcnn_sha256"]:
        raise RuntimeError("the common frozen visual encoder changed")
    for name,model in models.items():
        model.eval()
        if {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                for k,v in model.named_buffers()} != model_identity["same_fixed_buffer_sha256"]:
            raise RuntimeError("the shared fixed waveform buffers changed")
        final = evaluate(model,holdout,holdout_third,device)
        save(args.output/f"{name}_after.json",final)
        print(name,"after",json.dumps(final["summary"]),flush=True)
        state = {f"audio_model.{k}":v.cpu() for k,v in model.state_dict().items()}
        state.update({f"video_model.{k}":v.cpu() for k,v in visual.backbone.state_dict().items()})
        torch.save({"architecture": "FalconSpeakerContrastive", "state_dict": state, "steps":args.steps},
                   args.output/f"{name}_after.pt")


if __name__ == "__main__":
    main()
