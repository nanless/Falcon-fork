"""Measure a local ASR first-word recall proxy on exported RoomReader reopening clips."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokens(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized)


def official_words(path):
    root = ET.parse(path)
    times = {slot.attrib["TIME_SLOT_ID"]: int(slot.attrib["TIME_VALUE"])
             for slot in root.findall(".//TIME_SLOT") if "TIME_VALUE" in slot.attrib}
    speakers = {}
    for tier in root.findall("TIER"):
        if not tier.attrib["TIER_ID"].endswith("_tok") or not tier.get("PARTICIPANT"):
            continue
        speaker = tier.attrib["PARTICIPANT"]
        if speaker in speakers:
            raise ValueError(f"duplicate word tier: {path} {speaker}")
        words = defaultdict(list)
        for row in tier.findall(".//ALIGNABLE_ANNOTATION"):
            text = row.findtext("ANNOTATION_VALUE") or ""
            start, end = (times.get(row.attrib[key]) for key in ("TIME_SLOT_REF1", "TIME_SLOT_REF2"))
            if start is not None and end is not None and end > start and tokens(text):
                words[start].append({"text": text, "start_ms": start, "end_ms": end})
        speakers[speaker] = words
    return speakers


def matched_words(expected, decoded, clip_start_ms):
    start = expected["start_ms"] - clip_start_ms
    end = expected["end_ms"] - clip_start_ms
    # Whisper timestamp alignment is approximate on short speech. Compare
    # time-overlapping words with a fixed 200 ms allowance at each edge.
    overlapping = [word for word in decoded
                   if word["end"] * 1000 >= start - 200 and word["start"] * 1000 <= end + 200]
    recognized = tokens(" ".join(word["word"] for word in overlapping))
    wanted = tokens(expected["text"])
    return any(recognized[i:i + len(wanted)] == wanted
               for i in range(len(recognized) - len(wanted) + 1))


def decode(model, path):
    wave, rate = sf.read(path, dtype="float32")
    if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError(f"expected finite 16 kHz mono waveform: {path}")
    result = model.transcribe(wave, language="en", task="transcribe", word_timestamps=True,
                              temperature=0.0, condition_on_previous_text=False, fp16=False,
                              verbose=None)
    words = [word for segment in result["segments"] for word in segment["words"]]
    return result["text"].strip(), words


def validate_manifest(dataset_root, manifest):
    annotations = {}
    for session, expected_sha256 in manifest["official_elan_sha256"].items():
        path = dataset_root / "annotations" / "elan_sessions" / f"{session}.eaf"
        if digest(path) != expected_sha256:
            raise ValueError(f"RoomReader annotation changed: {path}")
        annotations[session] = official_words(path)
    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--whisper-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    labels = ("target_reference", *args.arm)
    if (len(set(labels)) != len(labels) or args.threads < 1
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels)):
        parser.error("labels must be unique filename stems and --threads must be positive")

    import torch
    import whisper

    torch.set_num_threads(args.threads)
    manifest = json.loads(args.manifest.read_text())
    identity_path = args.result / "identity.json"
    identity = json.loads(identity_path.read_text())
    if identity["manifest_sha256"] != digest(args.manifest):
        raise ValueError("result and official reopening manifest differ")
    annotations = validate_manifest(args.dataset_root, manifest)
    scoring = []
    metrics_path = args.result / "rows.jsonl"
    with metrics_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["group"] == "reopen_120":
                scoring.append(row)
    if len(scoring) != len(manifest["rows"]) or [r["id"] for r in scoring] != list(range(len(scoring))):
        raise ValueError("expected one reopening record per frozen case")
    model = whisper.load_model(str(args.whisper_checkpoint), device=args.device)
    rows = []
    for row in scoring:
        index = row["id"]
        row_output = {"id": index, "session": row["session"], "target": row["target"],
                      "onset_ms": row["onset_ms"], "start_context_ms": row["start_context_ms"]}
        if "metrics" not in row:
            rows.append(row_output | {"excluded": "acoustic reopening result missing"})
            continue
        official = annotations[row["session"]][row["target"]].get(row["official_token_start_ms"], [])
        if len(official) != 1 or official[0]["end_ms"] > row["end_context_ms"]:
            rows.append(row_output | {"excluded": "official first word ambiguous or not fully covered"})
            continue
        clip = args.result / "waveforms" / f"{index:03d}"
        comparisons = {}
        for label in labels:
            path = clip / f"{label}.wav"
            hypothesis, words = decode(model, path)
            comparisons[label] = {
                "file": str(path), "sha256": digest(path), "hypothesis": hypothesis,
                "words": words, "word_recalled": matched_words(official[0], words, row["start_context_ms"]),
            }
        print(index, {label: item["word_recalled"] for label, item in comparisons.items()}, flush=True)
        rows.append(row_output | {"official_first_word": official[0], "comparisons": comparisons})
    eligible = [r for r in rows if r.get("comparisons", {}).get("target_reference", {}).get("word_recalled")]
    if not eligible:
        raise RuntimeError("local ASR recalled no official first word on the reference tracks")
    summary = {
        "attempted": len(rows), "decoded": sum("comparisons" in row for row in rows),
        "reference_recall_eligible": len(eligible),
        "first_word_recalled": {label: sum(row["comparisons"][label]["word_recalled"] for row in eligible)
                                for label in args.arm},
        "paired_arm_transitions": {
            f"{a}_to_{b}": {
                "newly_recalled": sum(not row["comparisons"][a]["word_recalled"]
                                      and row["comparisons"][b]["word_recalled"] for row in eligible),
                "newly_missed": sum(row["comparisons"][a]["word_recalled"]
                                    and not row["comparisons"][b]["word_recalled"] for row in eligible),
            } for a, b in zip(args.arm, args.arm[1:])
        },
    }
    result = {"manifest_sha256": digest(args.manifest), "rows_sha256": digest(metrics_path),
              "identity_sha256": digest(identity_path), "code_sha256": digest(Path(__file__)),
              "whisper_weights_sha256": digest(args.whisper_checkpoint), "whisper_version": whisper.__version__,
              "device": args.device, "summary": summary, "rows": rows}
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
