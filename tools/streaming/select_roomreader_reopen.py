"""Freeze RoomReader speech onsets after a visible silent interval."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from tools.streaming.evaluate_roomreader import db_ratio, fit, read_audio, sha256


def intervals(path):
    root = ET.parse(path)
    slots = {item.attrib["TIME_SLOT_ID"]: int(item.attrib["TIME_VALUE"])
             for item in root.findall(".//TIME_SLOT") if "TIME_VALUE" in item.attrib}
    people = defaultdict(dict)
    for tier in root.findall("TIER"):
        person = tier.get("PARTICIPANT")
        kind = tier.attrib["TIER_ID"].rsplit("_", 1)[-1]
        if person is None or kind not in ("Check", "tok", "Para"):
            continue
        if kind in people[person]:
            raise ValueError(f"duplicate annotation tier: {path} {person} {kind}")
        rows = []
        for item in tier.findall(".//ALIGNABLE_ANNOTATION"):
            begin = slots.get(item.get("TIME_SLOT_REF1"))
            end = slots.get(item.get("TIME_SLOT_REF2"))
            value = (item.findtext("ANNOTATION_VALUE") or "").strip()
            if begin is not None and end is not None and begin < end and any(c.isalnum() for c in value):
                rows.append((begin, end))
        people[person][kind] = sorted(rows)
    return {person: data for person, data in people.items() if data.keys() >= {"Check", "tok", "Para"}}


def intersection(interval, start, stop):
    return interval[0] < stop and interval[1] > start


def power(values):
    return float(np.mean(np.square(values, dtype=np.float64)))


def locate_onset(target, origin_ms, annotated_ms):
    """Return the first of three energetic 10 ms bins near the transcript."""
    energy = np.mean(target.reshape(-1, 160).astype(np.float64) ** 2, axis=1)
    centers = origin_ms + np.arange(len(energy)) * 10
    prior = energy[(centers >= annotated_ms - 380) & (centers < annotated_ms - 150)]
    later = energy[(centers >= annotated_ms + 80) & (centers < annotated_ms + 280)]
    if len(prior) < 15 or len(later) < 15:
        return None
    threshold = max(1e-6, 10 * float(np.median(prior)), float(np.quantile(later, .8)) * .035)
    starts = [i for i in np.flatnonzero((centers >= annotated_ms - 110) & (centers <= annotated_ms + 110))
              if np.all(energy[i:i+3] >= threshold)]
    return int(centers[starts[0]]) if starts else None


def read_candidate(root, session, speaker, anno, names, paths, people, gaps):
    annotation_start = anno[0]
    if annotation_start < 900 or anno[1] - annotation_start < 360:
        return None
    own = people[speaker]
    if any(intersection((a, b), annotation_start - 460, annotation_start - 40)
           for a, b in own["Check"] + own["tok"] + own["Para"]):
        return None
    if any(intersection((a, b), annotation_start - 40, annotation_start + 310) for a, b in own["Para"]):
        return None
    words = [(a, b) for a, b in own["tok"] if annotation_start - 70 <= a <= annotation_start + 120]
    if not words or any(abs(annotation_start - existing) < 2000 for existing in gaps):
        return None
    onset_token = words[0][0]
    others = [who for who in names if who != speaker and
              any(intersection(row, annotation_start - 300, annotation_start + 240) for row in people[who]["Check"])
              and any(intersection(row, annotation_start - 300, annotation_start + 240) for row in people[who]["tok"])]
    if not others:
        return None
    start = (onset_token // 10) * 10 - 420
    stop = (onset_token // 10) * 10 + 440
    audio = root / "audio"
    mix_path = audio / "all_participants_audio" / f"{session}.wav"
    stems = np.column_stack([read_audio(paths[name], start, stop) for name in names])
    target_index = names.index(speaker)
    reference = stems[:, target_index]
    onset = locate_onset(reference, start, onset_token)
    if onset is None:
        return None
    before = slice((onset - start - 240) * 16, (onset - start - 40) * 16)
    after120 = slice((onset - start) * 16, (onset - start + 120) * 16)
    after240 = slice((onset - start) * 16, (onset - start + 240) * 16)
    others_indices = [names.index(who) for who in others]
    before_target, after_target = power(reference[before]), power(reference[after120])
    before_other = sum(power(stems[before, index]) for index in others_indices)
    after_other = sum(power(stems[after240, index]) for index in others_indices)
    if (before_target > 10**(-4.5) or after_target < 10**(-4.5)
            or before_other < 10**(-4.2) or after_other < 10**(-4.4)
            or db_ratio(after_target, before_target) < 15
            or db_ratio(before_target, before_other) > -15):
        return None
    mixture = read_audio(mix_path, start, stop)
    _, projection, residual, rank = fit(stems[after120], mixture[after120])
    residual_db = db_ratio(residual, float(np.var(mixture[after120])))
    if rank != len(names) or residual_db > -10 or projection[target_index] < 1e-6:
        return None
    video = list((root / "video/individual_participants/individual_participants_audio_all" / session)
                 .glob(f"{session}_{speaker}_*_all.mp4"))
    if len(video) != 1:
        return None
    return {"session": session, "target": speaker, "group": "reopen",
            "start_ms": onset - 240, "end_ms": onset + 240, "onset_ms": onset,
            "official_checked_start_ms": annotation_start, "official_token_start_ms": onset_token,
            "acoustic_minus_token_ms": onset - onset_token, "other_annotated": others,
            "mix_fit_residual_120_db": residual_db,
            "target_before_dbfs": db_ratio(before_target, 1),
            "target_after120_dbfs": db_ratio(after_target, 1),
            "other_before_dbfs": db_ratio(before_other, 1),
            "other_after240_dbfs": db_ratio(after_other, 1),
            "after240_sir_db": db_ratio(power(reference[after240]), after_other)}


def select(root, frozen, per_session):
    old = json.loads(frozen.read_text())["rows"]
    sessions = sorted({row["session"] for row in old})
    selected, source_hashes = [], {}
    for session in sessions:
        elan = root / "annotations/elan_sessions" / f"{session}.eaf"
        source_hashes[session] = sha256(elan)
        people = intervals(elan)
        names = sorted(people)
        paths = {}
        for name in names:
            candidates = list((root / "audio/individual_participants_audio" / session)
                              .glob(f"{session}_{name}_*.wav"))
            if len(candidates) != 1:
                raise ValueError(f"missing unique reference audio: {session} {name}")
            paths[name] = candidates[0]
        room_rows = []
        old_times = [row["start_ms"] for row in old if row["session"] == session]
        holdout_targets = sorted({row["target"] for row in old if row["session"] == session})
        for target in holdout_targets:
            n_target = 0
            for anno in people[target]["Check"]:
                row = read_candidate(root, session, target, anno, names, paths, people,
                                     old_times + [r["onset_ms"] for r in room_rows])
                if row:
                    room_rows.append(row)
                    n_target += 1
                    if n_target >= 4:
                        break
        room_rows.sort(key=lambda row: (abs(row["acoustic_minus_token_ms"]), row["mix_fit_residual_120_db"]))
        per_person = defaultdict(int)
        chosen = []
        for row in room_rows:
            if per_person[row["target"]] >= 2:
                continue
            chosen.append(row)
            per_person[row["target"]] += 1
            if len(chosen) == per_session:
                break
        selected.extend(chosen)
        print(session, len(room_rows), len(chosen), flush=True)
    return {"rows": selected, "official_elan_sha256": source_hashes,
            "holdout_sessions": sessions, "original_frozen_manifest_sha256": sha256(frozen),
            "code_sha256": sha256(Path(__file__)),
            "protocol": "Official checked utterance and word onset, verified with native separate microphone; original mixed track."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-session", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.per_session < 1:
        parser.error("output must be new and per-session must be positive")
    result = select(args.dataset_root, args.frozen, args.per_session)
    if not result["rows"]:
        raise ValueError("no verified onsets")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
