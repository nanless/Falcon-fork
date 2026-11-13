"""Cache native RoomReader speech/silence transitions for matched streaming training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from tools.streaming.evaluate_roomreader import (
    build_detector, db_ratio, fit, media_pipe_mouth, read_audio,
)
from tools.streaming.select_roomreader_intervals import labels

TICK_MS = 40
TICK_SAMPLES = 640
BLOCK_TICKS = 3
HISTORY_BLOCKS = 5
PREVIEW_TICKS = 2
BLIND_SESSIONS = ("S05", "S06", "S11", "S15", "S16")
TRAIN_SESSIONS = ("S08", "S10", "S17", "S20", "S23", "S27", "S29", "S30")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def participant_paths(root, session):
    root = root / "audio/individual_participants_audio" / session
    return {path.stem.split("_", 2)[1]: path for path in sorted(root.glob("*.wav"))}


def block_numbers(ticks):
    """Only full, mature blocks with two real preview ticks."""
    return [
        block for block in range(HISTORY_BLOCKS, (len(ticks) - PREVIEW_TICKS) // BLOCK_TICKS)
        if ticks[block * BLOCK_TICKS:(block + 1) * BLOCK_TICKS].all()
    ]


def dilated(occupied, guard=2):
    result = occupied.copy()
    for i in range(1, guard + 1):
        result[i:] |= occupied[:-i]
        result[:-i] |= occupied[i:]
    return result


def transition_type(silent, active):
    silent_blocks, active_blocks = block_numbers(silent), block_numbers(active)
    bracket = any(min(active_blocks) < block < max(active_blocks) for block in silent_blocks)
    if bracket:
        return "stop_reopen"
    return "reopen" if max(active_blocks) > max(silent_blocks) else "stop"


def candidate_windows(annotations, target, shared, limit):
    target_speech, target_occupied = annotations[target]
    others_speech = np.logical_or.reduce([
        pair[0] for name, pair in annotations.items() if name not in shared and name != target
    ])
    shared_occupied = np.logical_or.reduce([dilated(annotations[name][1]) for name in shared])
    silent = ~dilated(target_occupied) & others_speech
    width = 72
    candidates = []
    for start in range(0, len(silent) - width, 6):
        sl = slice(start, start + width)
        if shared_occupied[sl].any():
            continue
        muted = block_numbers(silent[sl])
        active = block_numbers(target_speech[sl])
        if len(muted) < 3 or len(active) < 2:
            continue
        score = (transition_type(silent[sl], target_speech[sl]) == "stop_reopen", min(len(muted), 8) + min(len(active), 8), -start)
        candidates.append((score, start, silent[sl], target_speech[sl]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    retained = []
    for _, start, silent, active in candidates:
        if all(abs(start - a[0]) >= width for a in retained):
            retained.append((start, silent, active))
        if len(retained) == limit:
            break
    return retained


def audio_supervision(root, session, paths, target, shared, start, silent, active):
    end = start + len(silent)
    milliseconds = (TICK_MS * start, TICK_MS * end)
    mix = read_audio(root / "audio/all_participants_audio" / f"{session}.wav", *milliseconds)
    names = sorted(paths)
    stems = np.column_stack([read_audio(paths[name], *milliseconds) for name in names])
    mix_ticks = mix.reshape(-1, TICK_SAMPLES).astype(np.float64)
    mix_power = np.mean(mix_ticks**2)
    shared_power = max(np.mean(stems[:, names.index(name)].astype(np.float64)**2) for name in shared)
    shared_db = db_ratio(shared_power, mix_power, floor=1e-15)
    if shared_db > -30:
        return None
    gains, _, residual, rank = fit(stems, mix)
    gain = float(gains[names.index(target)])
    if rank < len(names) - 1 or not 0.05 <= gain <= 10 or residual > .08 * mix_power:
        return None
    target_audio = (stems[:, names.index(target)] * gain).astype(np.float32)
    y = target_audio.reshape(-1, TICK_SAMPLES).astype(np.float64)
    safe_silent = silent.copy()
    for block in block_numbers(silent):
        sl = slice(BLOCK_TICKS * block, BLOCK_TICKS * (block + 1))
        target_power, input_power = np.mean(y[sl]**2), np.mean(mix_ticks[sl]**2)
        if db_ratio(target_power, input_power, floor=1e-15) > -24 or input_power < 10**(-4.7):
            safe_silent[sl] = False
    safe_active = active & (np.mean(y**2, axis=1) > 10**(-5.2))
    if len(block_numbers(safe_silent)) < 3 or len(block_numbers(safe_active)) < 2:
        return None
    return mix, target_audio, safe_silent, safe_active, shared_db, gain, float(residual / mix_power)


def write_record(args, rows, session, target, start, silent, active, shared, paths, detector):
    try:
        result = audio_supervision(args.source, session, paths, target, shared, start, silent, active)
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        print("skip_audio", session, target, start, type(exc).__name__, str(exc), flush=True)
        return False
    if result is None:
        return False
    mixture, reference, silent, active, shared_db, gain, residual = result
    transition = transition_type(silent, active)
    video_dir = args.source / "video/individual_participants/individual_participants_audio_all" / session
    matches = sorted(video_dir.glob(f"{session}_{target}_*_all.mp4"))
    if len(matches) != 1:
        return False
    video, = matches
    try:
        mouths, present, clock_error = media_pipe_mouth(video, detector, TICK_MS*start, TICK_MS*(start+len(silent)))
    except (RuntimeError, ValueError, cv2.error) as exc:
        print("skip_video", session, target, start, type(exc).__name__, str(exc), flush=True)
        return False
    if not present.all():
        return False
    # The evaluation extractor returns normalized center crops; restore their
    # uint8 values exactly and preserve their 88x88 center for the training loader.
    pixels = np.rint(np.clip((mouths * .165 + .421) * 255, 0, 255)).astype(np.uint8)
    if not np.allclose((pixels.astype(np.float32) / 255 - .421) / .165, mouths, atol=1e-6):
        raise RuntimeError("center mouth pixels cannot be recovered exactly")
    mouth = np.pad(pixels, ((0, 0), (4, 4), (4, 4)), mode="edge")
    record_id = f"roomreader-{len(rows):04d}"
    path = args.output / f"{record_id}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, audio=mixture, target=reference, mouth=mouth,
                            silent=silent, active=active, mouth_valid=present)
    row = {
        "id": record_id, "npz": str(path.resolve()), "sha256": digest(path),
        "recording_id": session, "speaker_id": target, "source_start_ms": TICK_MS * start,
        "silent_blocks": block_numbers(silent), "active_blocks": block_numbers(active),
        "transition": transition, "mouth_clock_max_abs_ms": clock_error,
        "shared_tutor_ids": sorted(shared), "shared_tutor_to_native_mix_db": shared_db,
        "native_target_track_gain": gain, "native_mix_fit_residual_fraction": residual,
    }
    with (args.output / "accepted.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    rows.append(row)
    print(record_id, session, transition, len(row["silent_blocks"]), len(row["active_blocks"]), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=TRAIN_SESSIONS)
    parser.add_argument("--blind-sessions", nargs="+", default=BLIND_SESSIONS)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--per-session", type=int, default=6)
    args = parser.parse_args()
    if min(args.count, args.per_session) < 1 or set(args.sessions) & set(args.blind_sessions):
        parser.error("training must exclude blind sessions and count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    blind = set().union(*(set(participant_paths(args.source, s)) for s in args.blind_sessions))
    if not blind:
        raise ValueError("empty blind identity inventory")
    detector, version = build_detector(args.face_model)
    rows = []
    excluded = []
    with detector:
        for session in args.sessions:
            before_session = len(rows)
            paths = participant_paths(args.source, session)
            annotation_path = args.source / "annotations/elan_sessions" / f"{session}.eaf"
            annotations = labels(annotation_path)
            shared = set(paths) & blind
            if not shared or any(not s.startswith("T") for s in shared) or not shared <= annotations.keys():
                excluded.append(session)
                continue
            if len(set(paths) & annotations.keys() - shared) < 2:
                continue
            for target in sorted((set(paths) & annotations.keys()) - shared):
                accepted = 0
                for start, silent, active in candidate_windows(annotations, target, shared, 16):
                    if write_record(args, rows, session, target, start, silent, active, shared, paths, detector):
                        accepted += 1
                        if accepted == 2 or len(rows) == args.count or len(rows) - before_session == args.per_session:
                            break
                if len(rows) == args.count or len(rows) - before_session == args.per_session:
                    break
            if len(rows) == args.count:
                break
    if not rows:
        raise RuntimeError("no valid native recordings passed supervision and mouth checks")
    report = {
        "format": "roomreader-native-silence-factorial-v1", "source": str(args.source),
        "source_license": "RoomReader non-commercial research license", "face_model_sha256": digest(args.face_model),
        "mediapipe_version": version, "blind_sessions": args.blind_sessions, "blind_ids": sorted(blind),
        "shared_tutor_policy": "No annotation and at most -30 dB personal track versus mix in full windows",
        "source_annotation_sha256": {s: digest(args.source / "annotations/elan_sessions" / f"{s}.eaf")
                                    for s in sorted({r["recording_id"] for r in rows})},
        "excluded_sessions": excluded, "episodes": rows,
        "source_sha256": digest(Path(__file__)), "max_episodes_per_recording": args.per_session,
    }
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
