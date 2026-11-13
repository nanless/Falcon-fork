"""Freeze native RoomReader pairs and score two separate single-target requests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess

import cv2
import numpy as np
import soundfile as sf
import torch

from tools.streaming import evaluate_roomreader as room
from tools.streaming.rolling_model_checkpoint import RollingExperimentLiveInference
from tools.streaming.select_roomreader_intervals import SOURCE, labels, ranges, sha, wav

# Earlier RoomReader training and diagnostic sessions are excluded.
SESSIONS = ("S01", "S02", "S03", "S04", "S07", "S09", "S12", "S13",
            "S14", "S18", "S19", "S21", "S22", "S24", "S25", "S26", "S28")
GROUPS = ("weak_two", "weak_three_plus", "other_three_plus")
HISTORY_BLOCKS = 5


def candidates(root, session):
    annotation = root / "annotations/elan_sessions" / f"{session}.eaf"
    people = labels(annotation)
    names = list(people)
    path = root / "audio/individual_participants_audio" / session
    audio = {name: room.one(path, f"{session}_{name}_*.wav") for name in names}
    mix_path = root / "audio/all_participants_audio" / f"{session}.wav"
    max_ticks = min(sf.info(path).frames for path in [mix_path, *audio.values()]) // 1280
    speech = np.stack([people[name][0][:max_ticks] for name in names])
    encoded = (speech * (2 ** np.arange(len(names))[:, None])).sum(0)
    results = []
    for mask in sorted(set(encoded)):
        active = [name for index, name in enumerate(names) if mask & (1 << index)]
        students = [name for name in active if name.startswith("P")]
        if len(students) < 2:
            continue
        for start, end in sorted(ranges(encoded == mask), key=lambda x: (x[0] - x[1], x[0]))[:30]:
            if end - start < (4 if len(active) >= 3 else 8) or start < 16 or end + 16 > max_ticks:
                continue
            try:
                mixture = wav(mix_path, start, end)
                power = {name: float(np.mean(wav(audio[name], start, end) ** 2)) for name in active}
            except sf.LibsndfileError as exc:
                print(session, start, end, "unreadable audio", str(exc), flush=True)
                continue
            strong, weak = sorted(students, key=lambda name: (-power[name], name))[:2]
            weak_db = room.db_ratio(power[weak], power[strong])
            if room.db_ratio(np.mean(mixture**2), 1) < -47:
                continue
            if min(room.db_ratio(value, 1) for value in power.values()) < -50:
                continue
            if -14 <= weak_db <= -6:
                group = "weak_three_plus" if len(active) >= 3 else "weak_two"
            elif len(active) >= 3 and weak_db >= -14:
                group = "other_three_plus"
            else:
                continue
            results.append({"session": session, "start_ms": start * room.TICK_MS,
                            "end_ms": end * room.TICK_MS, "group": group,
                            "targets": [weak, strong], "active": active,
                            "weak_to_paired_reference_db": weak_db})
    return sorted(results, key=lambda row: (row["start_ms"] - row["end_ms"],
                                            abs(row["weak_to_paired_reference_db"] + 9)))


def freeze_candidate(row, index, root, detector):
    # First request loads the native mix and reference stems once. The alternate
    # request shares those exact arrays and independently loads the other video.
    weak, strong = row["targets"]
    session = row["session"]
    observed, mix, stems, mouth_a, present_a = room.read_clip(
        row | {"target": weak, "group": "triple"}, index, root, detector)
    start, end = observed["start_context_ms"], observed["end_context_ms"]
    visual = root / "video/individual_participants/individual_participants_audio_all" / session
    video_b = room.one(visual, f"{session}_{strong}_*_all.mp4")
    mouth_b, present_b, clock_b = room.media_pipe_mouth(video_b, detector, start, end)
    a, b = row["start_ms"], row["end_ms"]
    first = max(0, (a - start) // room.TICK_MS - 2)
    last = min(len(present_a), (b - start) // room.TICK_MS + 2)
    if not present_a[first:last].all() or not present_b[first:last].all():
        return "mouth_not_continuous", None
    score = slice((a - start) * room.SAMPLES_PER_MS, (b - start) * room.SAMPLES_PER_MS)
    x, reference = mix[score], stems[score]
    gains, powers, residual, rank = room.fit(reference, x)
    residual_db = room.db_ratio(residual, np.var(x))
    ids = [observed["speaker_order"].index(person) for person in row["active"]]
    if (rank != reference.shape[1] or residual_db > -10 or np.any(gains[ids] <= 0)
            or np.any(powers[ids] < 1e-6)):
        return "native_mix_reference_fit", None
    weak_index, strong_index = [observed["speaker_order"].index(person) for person in row["targets"]]
    weak_db = room.db_ratio(powers[weak_index], powers[strong_index])
    if row["group"] != "other_three_plus" and not -14 <= weak_db <= -6:
        return "native_mix_weak_ratio", None
    mouths = np.stack((mouth_a, mouth_b))
    present = np.stack((present_a, present_b))
    identity = row | {"id": index, "start_context_ms": start, "end_context_ms": end,
                      "speaker_order": observed["speaker_order"], "native_mix_fit_residual_db": residual_db,
                      "native_weak_to_paired_db": weak_db,
                      "mixture_sha256": room.array_sha256(mix), "stems_sha256": room.array_sha256(stems),
                      "mouth_sha256": [room.array_sha256(mouth) for mouth in mouths],
                      "present_sha256": [room.array_sha256(flags) for flags in present],
                      "mouth_context_valid_fraction": present.mean(axis=1).tolist(),
                      "mouth_clock_max_abs_ms": [observed["mouth_clock_max_abs_ms"], clock_b]}
    if identity["mouth_sha256"][0] == identity["mouth_sha256"][1]:
        raise ValueError("the two requested speakers have identical mouth-track bytes")
    return None, (identity, mix, stems, mouths, present)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    detector, version = room.build_detector(args.face_model)
    selected = []
    decisions = []
    excluded_sessions = {}
    try:
        for session in args.sessions:
            try:
                clips = candidates(args.source, session)
            except (ValueError, OSError, sf.LibsndfileError) as exc:
                excluded_sessions[session] = str(exc)
                print(session, "session_assets_unavailable", str(exc), flush=True)
                continue
            counts = dict.fromkeys(GROUPS, 0)
            for row in clips:
                group = row["group"]
                if counts[group] >= args.per_group_per_session:
                    continue
                if any(r["session"] == session and row["start_ms"] < r["end_ms"] + 1280
                       and r["start_ms"] < row["end_ms"] + 1280 for r in selected):
                    continue
                try:
                    failure, result = freeze_candidate(row, len(selected), args.source, detector)
                except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
                    failure, result = f"asset_error: {exc}", None
                decisions.append(row | {"rejection": failure})
                if result is None:
                    continue
                identity, mix, stems, mouths, present = result
                np.savez_compressed(args.output / f"{identity['id']:03d}.npz",
                                    mix=mix, stems=stems, mouths=mouths, present=present)
                selected.append(identity)
                counts[group] += 1
            print(session, counts, flush=True)
    finally:
        detector.close()
    if not selected:
        raise ValueError("no native pairs satisfied the frozen selection rules")
    write_json(args.output / "manifest.json", {
        "rows": selected, "candidate_decisions": decisions, "excluded_sessions": excluded_sessions,
        "annotation_sha256": {session: sha(args.source / "annotations/elan_sessions" / f"{session}.eaf")
                               for session in args.sessions
                               if (args.source / "annotations/elan_sessions" / f"{session}.eaf").is_file()},
        "face_model_sha256": room.sha256(args.face_model), "mediapipe_version": version,
        "code_sha256": room.sha256(Path(__file__)), "source": str(args.source),
    })


def check_arrays(row, mix, stems, mouths, present):
    count = (row["end_context_ms"] - row["start_context_ms"]) * room.SAMPLES_PER_MS
    ticks = count // 640
    if (mix.shape != (count,) or stems.shape != (count, len(row["speaker_order"]))
            or mouths.shape != (2, ticks, 88, 88) or present.shape != (2, ticks)
            or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (mix, stems, mouths))
            or present.dtype != np.bool_):
        raise ValueError(f"cached array geometry or values differ: {row['id']}")
    if (room.array_sha256(mix) != row["mixture_sha256"]
            or room.array_sha256(stems) != row["stems_sha256"]
            or [room.array_sha256(a) for a in mouths] != row["mouth_sha256"]
            or [room.array_sha256(a) for a in present] != row["present_sha256"]):
        raise ValueError(f"cached array hashes differ: {row['id']}")


def score_request(output, mixture, stems, names, active, target):
    target_idx = names.index(target)
    competitors = [name for name in active if name != target]
    scores = {name: room.si_sdr(output, stems[:, names.index(name)]) for name in active}
    margin = scores[target] - max(scores[name] for name in competitors)
    gain, powers, residual, _ = room.fit(stems, output)
    native_gain, native_powers, _, _ = room.fit(stems, mixture)
    fit_margin = room.db_ratio(powers[target_idx],
                               max(powers[names.index(name)] for name in competitors))
    fit_residual = room.db_ratio(residual, np.var(output))
    ranking = sorted(scores, key=lambda name: (-scores[name], name))
    preferred = ranking[0] if scores[ranking[0]] > scores[ranking[1]] else None
    return {"target": target, "preferred_active_reference": preferred, "target_si_sdr_db": scores[target],
            "target_sisdri_db": scores[target] - room.si_sdr(mixture, stems[:, target_idx]),
            "per_active_reference_si_sdr_db": scores, "si_sdr_margin_db": margin,
            "target_preferred": margin > 0, "fit_target_margin_db": fit_margin,
            "fit_target_preferred": fit_margin > 0 and gain[target_idx] * native_gain[target_idx] > 0,
            "fit_residual_db": fit_residual,
            "target_projection_delta_db": room.db_ratio(powers[target_idx], native_powers[target_idx]),
            "output_to_mixture_db": room.db_ratio(np.mean(output**2), np.mean(mixture**2))}


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        if row["group"].startswith("weak_"):
            grouped["weak_all"].append(row)
        if len(row["active"]) >= 3:
            grouped["three_plus_all"].append(row)
        grouped[row["group"]].append(row)
    summary = {}
    for group, entries in grouped.items():
        models = {}
        for name in entries[0]["metrics"]:
            requests = [entry["metrics"][name] for entry in entries]
            models[name] = {
                "both_targets_preferred": sum(all(r["target_preferred"] for r in pair) for pair in requests),
                "neither_target_preferred": sum(not any(r["target_preferred"] for r in pair) for pair in requests),
                "both_outputs_prefer_same": sum(pair[0]["preferred_active_reference"] is not None
                                                and pair[0]["preferred_active_reference"] == pair[1]["preferred_active_reference"]
                                                for pair in requests),
                "first_target_preferred": sum(pair[0]["target_preferred"] for pair in requests),
                "second_target_preferred": sum(pair[1]["target_preferred"] for pair in requests),
                "both_fit_targets_preferred": sum(all(r["fit_target_preferred"] for r in pair) for pair in requests),
                "first_target_mean_si_sdri_db": statistics.mean(pair[0]["target_sisdri_db"] for pair in requests),
                "first_target_median_margin_db": statistics.median(pair[0]["si_sdr_margin_db"] for pair in requests),
                "first_target_median_projection_db": statistics.median(pair[0]["target_projection_delta_db"]
                                                                      for pair in requests),
            }
        targets = {person for entry in entries for person in entry["targets"]}
        summary[group] = {"pairs": len(entries), "sessions": len({e["session"] for e in entries}),
                          "unique_target_ids": len(targets),
                          "scored_ms": sum(e["end_ms"] - e["start_ms"] for e in entries),
                          "models": models}
        if {"paired_off", "paired_on"} <= models.keys():
            summary[group]["paired_on_minus_off"] = paired_contrasts(entries)
    return summary


def paired_contrasts(entries):
    """Compare results per frozen pair, resampling complete sessions."""
    sessions = sorted({entry["session"] for entry in entries})
    model_pairs = [(entry["metrics"]["paired_off"], entry["metrics"]["paired_on"]) for entry in entries]
    metrics = {
        "both_targets_preferred": lambda pair: int(all(r["target_preferred"] for r in pair)),
        "first_target_preferred": lambda pair: int(pair[0]["target_preferred"]),
        "first_target_si_sdri_db": lambda pair: pair[0]["target_sisdri_db"],
    }
    results = {}
    for name, value in metrics.items():
        delta = np.array([value(on) - value(off) for off, on in model_pairs])
        result = {"wins": int(np.sum(delta > 0)), "losses": int(np.sum(delta < 0)),
                  "ties": int(np.sum(delta == 0)), "mean_delta": float(delta.mean())}
        if len(sessions) >= 5:
            groups = [[d for d, entry in zip(delta, entries, strict=True) if entry["session"] == session]
                      for session in sessions]
            sums = np.array([sum(g) for g in groups])
            counts = np.array([len(g) for g in groups])
            rng = np.random.default_rng(20261113)
            samples = rng.integers(len(groups), size=(10000, len(groups)))
            simulated = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
            result["session_bootstrap_95_percent"] = np.quantile(simulated, [.025, .975]).tolist()
        results[name] = result
    return results


def evaluate(args):
    manifest = args.cache / "manifest.json"
    rows = json.loads(manifest.read_text())["rows"]
    if not rows:
        raise ValueError("input cache contains no paired rows")
    paths = {path.stem: path for path in args.arm}
    if len(paths) != len(args.arm) or paths.keys() & {"mixture", "released"}:
        raise ValueError("arm checkpoint names must be distinct and not reserved")
    paths = {"released": args.release} | paths
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    models = {name: RollingExperimentLiveInference(path, device=args.device)
              for name, path in paths.items()}
    if any(model.history_blocks != HISTORY_BLOCKS for model in models.values()):
        raise ValueError("all checkpoints must use 600 ms history")
    write_json(args.output / "identity.json", {
        "manifest_sha256": room.sha256(manifest), "code_sha256": room.sha256(Path(__file__)),
        "weights": {name: room.sha256(path) for name, path in paths.items()}, "device": args.device,
        "torch": torch.__version__, "history_ms": 600, "block_ms": 120, "lookahead_ms": 80,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    })
    results = []
    with (args.output / "rows.jsonl").open("x") as stream, torch.inference_mode():
        for row in rows:
            with np.load(args.cache / f"{row['id']:03d}.npz", allow_pickle=False) as clip:
                mix, stems, mouths, present = (clip[key] for key in ("mix", "stems", "mouths", "present"))
            check_arrays(row, mix, stems, mouths, present)
            begin = row["start_context_ms"]
            score = slice((row["start_ms"] - begin) * room.SAMPLES_PER_MS,
                          (row["end_ms"] - begin) * room.SAMPLES_PER_MS)
            outputs = {"mixture": [mix, mix]}
            outputs.update({name: [room.streaming(model, mix, lip, flags)
                                   for lip, flags in zip(mouths, present, strict=True)]
                            for name, model in models.items()})
            metrics = {name: [score_request(y[score], mix[score], stems[score], row["speaker_order"],
                                            row["active"], target)
                              for y, target in zip(pair, row["targets"], strict=True)]
                       for name, pair in outputs.items()}
            result = row | {"metrics": metrics}
            print(json.dumps(result, allow_nan=False), file=stream, flush=True)
            results.append(result)
            print(row["id"], row["session"], row["group"], flush=True)
    write_json(args.output / "summary.json", summarize(results))


def write_json(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("output", type=Path)
    prep.add_argument("--source", type=Path, default=SOURCE)
    prep.add_argument("--face-model", type=Path, required=True)
    prep.add_argument("--sessions", nargs="+", default=SESSIONS, choices=SESSIONS)
    prep.add_argument("--per-group-per-session", type=int, default=2)
    prep.set_defaults(run=prepare)
    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("cache", type=Path)
    evaluate_cmd.add_argument("output", type=Path)
    evaluate_cmd.add_argument("--release", type=Path, required=True)
    evaluate_cmd.add_argument("--arm", type=Path, action="append", default=[])
    evaluate_cmd.add_argument("--device", default="cuda:0")
    evaluate_cmd.set_defaults(run=evaluate)
    args = parser.parse_args()
    if args.run is prepare and (args.per_group_per_session < 1
                                or len(set(args.sessions)) != len(args.sessions)):
        parser.error("prepare requires positive cap and unique sessions")
    args.run(args)


if __name__ == "__main__":
    main()
