"""Score a single lip-guided output on selected RoomReader intervals."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import soundfile as sf
import torch

from tools.streaming.evaluate_realssa import load_baseline
from tools.streaming.paper_live_inference import PaperLiveInference
from tools.streaming.rolling_falcon_live import RollingFalconLiveInference

SAMPLES_PER_MS = 16
TICK_MS = 40
CONTEXT_MS = 640


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def one(parent, pattern):
    paths = sorted(parent.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"{parent}/{pattern}: expected one asset, got {len(paths)}")
    return paths[0]


def read_audio(path, start, end):
    with sf.SoundFile(path) as stream:
        if stream.samplerate != 32000 or stream.channels != 1:
            raise ValueError(f"expected RoomReader 32k mono: {path}")
        stream.seek(32 * start)
        audio = stream.read(32 * (end - start), dtype="float32")
    if audio.size != 32 * (end - start) or not np.isfinite(audio).all():
        raise ValueError(f"audio length or values differ: {path}")
    return resample_poly(audio, 1, 2).astype(np.float32)


def video_frames(video, start, end, fps, width, height):
    # Decode a sequential segment in native video order. The frame clock was
    # already checked; OpenFace row index is one-based and tracks this clock.
    first = round(start * fps / 1000)
    last = round((end - TICK_MS) * fps / 1000)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
        cap.release()
        raise RuntimeError(f"cannot seek {video} to frame {first}")
    wanted = set(np.rint(np.arange(start, end, TICK_MS) * fps / 1000).astype(int))
    frames = {}
    try:
        for index in range(first, last + 1):
            okay, frame = cap.read()
            if not okay or frame.shape[:2] != (height, width):
                raise RuntimeError(f"cannot read {video} frame {index}")
            if index in wanted:
                frames[index] = frame
    finally:
        cap.release()
    return frames


def read_mouth(video, landmarks, start, end):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=r_frame_rate,width,height,start_time", "-of", "json", str(video)]))
    stream, = probe["streams"]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den
    if fps < 25 or abs(float(stream["start_time"])) > .001:
        raise ValueError(f"unexpected video clock {stream}")
    wanted = ["frame", "timestamp", "success", "confidence"] + [
        f"{axis}_{i}" for axis in ("x", "y") for i in range(48, 68)]
    raw = pd.read_csv(landmarks, usecols=lambda col: col.strip() in wanted)
    raw.columns = raw.columns.str.strip()
    raw = raw.set_index("frame")
    if not raw.index.is_unique:
        raise ValueError(f"duplicate frame indexes in {landmarks}")
    width, height = stream["width"], stream["height"]
    frames = video_frames(video, start, end, fps, width, height)
    ticks = np.arange(start, end, TICK_MS)
    indices = np.rint(ticks * fps / 1000).astype(int)
    selected = raw.loc[indices + 1]
    clock_error = np.max(np.abs(selected["timestamp"].to_numpy() - ticks / 1000))
    if clock_error > .5 / fps + .001:
        raise ValueError(f"OpenFace clock does not match video within one frame: {clock_error}")
    result, present = [], []
    previous = None
    for i, values in zip(indices, selected[wanted[1:]].to_numpy(), strict=True):
        _, success, confidence, *xy = values
        xs, ys = np.array(xy).reshape(2, 20)
        mouth_width = np.linalg.norm([xs[6] - xs[0], ys[6] - ys[0]])
        size = round(2.13 * mouth_width)
        cx, cy = float(np.mean(xs[:12])), float(np.mean(ys[:12]))
        valid = bool(success and confidence >= .80 and 16 <= size <= min(height, width)
                     and size / 2 <= cx <= width - size / 2 and size / 2 <= cy <= height - size / 2)
        if valid:
            # Use the corpus-supplied track, temporally smooth crop geometry,
            # and then apply the published crop and normalisation.
            geometry = np.array([cx, cy, size])
            previous = geometry if previous is None else .5 * (previous + geometry)
            cx, cy, size = previous
            left, top = round(cx - size / 2), round(cy - size / 2)
            right, bottom = round(cx + size / 2), round(cy + size / 2)
            patch = cv2.cvtColor(frames[int(i)][top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)[4:92, 4:92]
            frame = (patch.astype(np.float32) / 255 - .421) / .165
        else:
            frame = np.zeros((88, 88), dtype=np.float32)
        result.append(frame)
        present.append(valid)
    return np.stack(result), np.array(present), float(clock_error * 1000)


MEDIAPIPE_LIPS = (61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185)


def media_pipe_mouth(video, detector, start, end):
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps < 25:
        raise ValueError(f"expected native FPS >= 25: {video}")
    frames = video_frames(video, start, end, fps, width, height)
    times = np.arange(start, end, TICK_MS)
    indices = np.rint(times * fps / 1000).astype(int)
    result, present = [], []
    previous = None
    for index in indices:
        image = frames[int(index)]
        resized = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        image_input = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        detections = detector.detect(image_input).face_landmarks
        crop = None
        if len(detections) == 1:
            points = np.array([(detections[0][i].x*width, detections[0][i].y*height) for i in MEDIAPIPE_LIPS])
            cx, cy = points.mean(0)
            size = round(2.13*np.linalg.norm(points[0]-points[10]))
            if 16 <= size <= min(height,width) and size/2 <= cx <= width-size/2 and size/2 <= cy <= height-size/2:
                geometry = np.array([cx, cy, size])
                previous = geometry if previous is None else .5 * (previous + geometry)
                cx, cy, size = previous
                patch = cv2.getRectSubPix(image, (round(size), round(size)), (cx, cy))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                patch = cv2.resize(patch, (96,96), interpolation=cv2.INTER_AREA)[4:92,4:92]
                crop = (patch.astype(np.float32)/255 - .421)/.165
        present.append(crop is not None)
        result.append(crop if crop is not None else np.zeros((88,88), dtype=np.float32))
    return np.stack(result), np.array(present), float(np.max(np.abs(indices/fps*1000-times)))



def streaming(model, mixture, frames, present=None):
    model.reset()
    outputs = []
    if mixture.shape != (len(frames) * 640,):
        raise ValueError("audio duration must match the 25 fps mouth track")
    for index, mouth in enumerate(frames):
        audio = mixture[index*640:(index+1)*640]
        if present is not None and not present[index]:
            mouth = None
        outputs.extend(model.push(audio, mouth, sequence=index))
    outputs.extend(model.finish())
    result = np.concatenate(outputs)
    if result.shape != mixture.shape or not np.isfinite(result).all():
        raise RuntimeError("stream output does not preserve finite single-channel audio")
    return result


def db_ratio(first, second, floor=1e-12):
    return float(10 * np.log10(max(first, floor) / max(second, floor)))


def si_sdr(output, reference):
    output, reference = [v.astype(np.float64) - v.mean(dtype=np.float64)
                         for v in (output, reference)]
    projected = (output @ reference) / max(reference @ reference, 1e-12) * reference
    return db_ratio(np.mean(projected**2), np.mean((output-projected)**2))


def fit(stems, signal):
    matrix = stems.astype(np.float64)
    matrix -= matrix.mean(axis=0)
    centered = signal.astype(np.float64) - signal.mean(dtype=np.float64)
    gains, _, rank, _ = np.linalg.lstsq(matrix, centered, rcond=1e-6)
    reconstructed = matrix @ gains
    components = np.mean((matrix * gains)**2, axis=0)
    residual = np.mean((centered - reconstructed)**2)
    return gains, components, residual, int(rank)


def array_sha256(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def read_clip(row, i, dataset_root, detector):
    session, target = row["session"], row["target"]
    if row["group"] not in ("triple", "visible_silent", "weak", "reopen"):
        raise ValueError(f"unknown group {row['group']}")
    a, b = row["start_ms"], row["end_ms"]
    if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b <= a:
        raise ValueError("invalid scoring interval")
    start = max(0, a//TICK_MS*TICK_MS - CONTEXT_MS)
    end = (b+TICK_MS-1)//TICK_MS*TICK_MS + CONTEXT_MS
    root = dataset_root
    audio = root/"audio"
    mix_path = audio/"all_participants_audio"/f"{session}.wav"
    participant_paths = sorted((audio/"individual_participants_audio"/session).glob("*.wav"))
    names = [p.stem.split("_",2)[1] for p in participant_paths]
    if target not in names:
        raise ValueError(f"target reference missing for {session} {target}")
    visual = root/"video/individual_participants/individual_participants_audio_all"/session
    video = one(visual,f"{session}_{target}_*_all.mp4")
    landmarks = None if detector else one(root/"features/OpenFace_Features",f"{session}_{target}_*_all.csv")
    mixture = read_audio(mix_path,start,end)
    stems = np.column_stack([read_audio(p,start,end) for p in participant_paths])
    if detector:
        mouths, present, clock = media_pipe_mouth(video, detector, start, end)
    else:
        mouths, present, clock = read_mouth(video, landmarks, start, end)
    tick_start = (a-start)//TICK_MS
    tick_end = (b-start+TICK_MS-1)//TICK_MS
    valid = bool(present[tick_start:tick_end].all())
    observed = row | {"id":i,"start_context_ms":start,"end_context_ms":end,
        "mouth_valid_all_score_ticks":valid,"mouth_valid_context_fraction":float(present.mean()),
        "mouth_clock_max_abs_ms":clock,"speaker_order":names,
        "mouth_track_sha256": array_sha256(mouths),
        "mixture_crop_sha256": array_sha256(mixture),
        "asset_paths":{"mix":str(mix_path),"video":str(video),"landmarks":str(landmarks),
                        "stems":[str(p) for p in participant_paths]}}
    observed["stems_sha256"] = array_sha256(stems)
    observed["present_sha256"] = array_sha256(present)
    return observed, mixture, stems, mouths, present


def check_clip(observed, mixture, stems, mouths, present):
    samples = (observed["end_context_ms"] - observed["start_context_ms"]) * SAMPLES_PER_MS
    n_ticks = samples // 640
    if (mixture.shape != (samples,) or stems.shape != (samples, len(observed["speaker_order"]))
            or mouths.shape != (n_ticks, 88, 88) or present.shape != (n_ticks,)
            or mixture.dtype != np.float32 or stems.dtype != np.float32
            or mouths.dtype != np.float32 or present.dtype != np.bool_):
        raise ValueError(f"cached row geometry differs: {observed['id']}")
    for key, array in (("mixture_crop_sha256", mixture), ("stems_sha256", stems),
                       ("mouth_track_sha256", mouths), ("present_sha256", present)):
        if array_sha256(array) != observed[key]:
            raise ValueError(f"cached row hash differs: {observed['id']} {key}")


def build_detector(face_model):
    import mediapipe as mp

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model)),
        num_faces=2, min_face_detection_confidence=.5)
    return mp.tasks.vision.FaceLandmarker.create_from_options(options), mp.__version__


def write_cache(args, rows, detector, mp_version):
    cached = []
    for i, row in enumerate(rows):
        observed, mixture, stems, mouths, present = read_clip(row, i, args.dataset_root, detector)
        check_clip(observed, mixture, stems, mouths, present)
        np.savez_compressed(args.output / f"{i:03d}.npz", mixture=mixture, stems=stems,
                            mouths=mouths, present=present)
        cached.append(observed)
        print(i, row["session"], row["target"], row["group"], observed["mouth_valid_all_score_ticks"],
              flush=True)
    with (args.output / "cache.json").open("x") as stream:
        json.dump({"manifest_sha256": sha256(args.manifest), "code_sha256": sha256(Path(__file__)),
                   "landmark_model_sha256": sha256(args.face_model) if args.face_model else None,
                   "media_pipe_version": mp_version, "rows": cached}, stream, indent=2, allow_nan=False)



def reopen_windows(observed):
    onset = observed["onset_ms"]
    if observed["start_ms"] != onset - 240 or observed["end_ms"] != onset + 240:
        raise ValueError("reopen case must have 240 ms around the acoustic onset")
    context = observed["start_context_ms"]
    return [(group, start, stop, slice((start - context) * SAMPLES_PER_MS,
                                      (stop - context) * SAMPLES_PER_MS))
            for group, start, stop in (("reopen_pre_silent", onset - 240, onset - 40),
                                       ("reopen_120", onset, onset + 120),
                                       ("reopen_240", onset, onset + 240))]


def measure_reopen(observed, outputs, stems, target_index):
    mixture = outputs["mixture"]
    for group, start, stop, score in reopen_windows(observed):
        row = observed | {"group": group, "start_ms": start, "end_ms": stop}
        if not observed["mouth_valid_all_score_ticks"]:
            yield row
            continue
        x = mixture[score]
        reference = stems[score, target_index]
        gains, powers, residual, rank = fit(stems[score], x)
        native_target_power = powers[target_index]
        native_target_gain = gains[target_index]
        native_residual_db = db_ratio(residual, np.var(x))
        if group != "reopen_pre_silent" and (native_target_power < 1e-6
                or native_target_gain <= 0 or native_residual_db > -10 or rank != stems.shape[1]):
            yield row | {"score_failure": "native target component not independently identifiable",
                         "native_fit_residual_db": native_residual_db}
            continue
        baseline = si_sdr(x, reference) if group != "reopen_pre_silent" else None
        results = {}
        for name, wave in outputs.items():
            y = wave[score]
            if y.shape != x.shape or not np.isfinite(y).all():
                raise ValueError("invalid reopen output")
            metrics = {"output_to_mixture_db": db_ratio(np.mean(y**2), np.mean(x**2))}
            if group != "reopen_pre_silent":
                gains_y, powers_y, residual_y, _ = fit(stems[score], y)
                projection_db = db_ratio(powers_y[target_index], native_target_power)
                others = np.delete(powers_y, target_index)
                target_sisdr = si_sdr(y, reference)
                metrics.update(target_si_sdr_db=target_sisdr, target_sisdri_db=target_sisdr - baseline,
                               target_to_loudest_interferer_db=db_ratio(powers_y[target_index], others.max()),
                               target_projection_delta_db=projection_db,
                               target_projection_below_minus6db=projection_db < -6,
                               target_projection_sign_reversed=bool(gains_y[target_index] * native_target_gain < 0),
                               reference_fit_residual_to_output_db=db_ratio(residual_y, np.var(y)))
            results[name] = metrics
        yield row | {"metrics": results, "native_fit_residual_db": native_residual_db}


def export_reopen_waveforms(directory, observed, outputs, target):
    """Save native-level aligned audio for offline review on the source host."""
    clip = directory / f"{observed['id']:03d}"
    clip.mkdir()
    for label, wave in {"target_reference": target, **outputs}.items():
        sf.write(clip / f"{label}.wav", wave, 16000, subtype="FLOAT")


def args_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--old", type=Path)
    parser.add_argument("--arm", type=Path, action="append", default=[])
    parser.add_argument("--experiment-arm", type=Path, action="append", default=[],
                        help="Add a versioned rolling experiment checkpoint.")
    parser.add_argument("--face-model", type=Path,
                        help="Use one MediaPipe face landmarker for all clips instead of the released OpenFace CSV.")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--export-reopen-waveforms", action="store_true",
                        help="Write aligned reopening WAVs to the local result directory for listening.")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--prepare-cache", action="store_true", help="Freeze input crops, mouth tracks and references without model inference.")
    cache_group.add_argument("--input-cache", type=Path, help="Read input crops previously frozen by --prepare-cache.")
    args = parser.parse_args()
    if args.input_cache is None and args.dataset_root is None:
        parser.error("--dataset-root is required when input crops are not cached")
    if args.prepare_cache and args.export_reopen_waveforms:
        parser.error("--export-reopen-waveforms needs inference")
    if args.input_cache and args.face_model:
        parser.error("--face-model is only used when cropping uncached input")
    if not args.prepare_cache and args.release is None:
        parser.error("--release is required for inference")
    return args


def main():
    args = args_main()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads(args.manifest.read_text())["rows"]
    if not rows:
        raise ValueError("manifest contains no rows")
    if args.export_reopen_waveforms:
        if any(row["group"] != "reopen" for row in rows):
            raise ValueError("waveform export requires a reopening-only manifest")
        (args.output / "waveforms").mkdir(mode=0o700)
    detector, mp_version = build_detector(args.face_model) if args.face_model else (None, None)
    if args.prepare_cache:
        try:
            write_cache(args, rows, detector, mp_version)
        finally:
            if detector is not None:
                detector.close()
        return
    cache = None
    if args.input_cache is not None:
        cache = json.loads((args.input_cache / "cache.json").read_text())
        if cache["manifest_sha256"] != sha256(args.manifest) or len(cache["rows"]) != len(rows):
            raise ValueError("input cache and manifest differ")
    rolling = RollingFalconLiveInference(args.release, history_blocks=5, device=args.device)
    old = PaperLiveInference(args.old, device=args.device) if args.old else None
    arm_paths = args.arm + args.experiment_arm
    arm_labels = {path.stem for path in arm_paths}
    reserved = {"mixture", "rolling", "old_streaming", "released_whole_window", "target_reference"}
    if len(arm_labels) != len(arm_paths) or arm_labels & reserved:
        raise ValueError("arm paths must have unique, non-reserved file names")
    arms = {path.stem: RollingFalconLiveInference(path, device=args.device) for path in args.arm}
    if args.experiment_arm:
        from tools.streaming.rolling_model_checkpoint import RollingExperimentLiveInference

        arms.update({path.stem: RollingExperimentLiveInference(path, device=args.device)
                     for path in args.experiment_arm})
    audio_model, visual_model, _ = load_baseline(args.release, args.device)
    identity = {"manifest_sha256": sha256(args.manifest), "code_sha256": sha256(Path(__file__)),
                "input_cache_sha256": sha256(args.input_cache / "cache.json") if cache else None,
                "weights": {"release":sha256(args.release), "old":sha256(args.old) if args.old else None,
                            **{p.stem:sha256(p) for p in arm_paths}},
                "landmark_model_sha256": cache["landmark_model_sha256"] if cache else (
                    sha256(args.face_model) if args.face_model else None),
                "torch": torch.__version__, "device": args.device,
                "media_pipe_version": cache["media_pipe_version"] if cache else mp_version,
                "git": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    with (args.output / "identity.json").open("x") as stream:
        json.dump(identity, stream, indent=2)
    with (args.output / "rows.jsonl").open("x") as stream, torch.inference_mode():
        for i, row in enumerate(rows):
            if cache is None:
                observed, mixture, stems, mouths, present = read_clip(row, i, args.dataset_root, detector)
            else:
                observed = cache["rows"][i]
                if {key: observed[key] for key in row} != row:
                    raise ValueError(f"cached row identity differs: {i}")
                with np.load(args.input_cache / f"{i:03d}.npz", allow_pickle=False) as clip:
                    mixture, stems, mouths, present = (clip[key] for key in ("mixture", "stems", "mouths", "present"))
                check_clip(observed, mixture, stems, mouths, present)
            start = observed["start_context_ms"]
            a, b = observed["start_ms"], observed["end_ms"]
            valid = observed["mouth_valid_all_score_ticks"]
            session, target = observed["session"], observed["target"]
            target_index = observed["speaker_order"].index(target)
            score = slice((a - start) * SAMPLES_PER_MS, (b - start) * SAMPLES_PER_MS)
            if not valid and row["group"] == "reopen":
                for part in measure_reopen(observed, {"mixture": mixture}, stems, target_index):
                    print(json.dumps(part, allow_nan=False), file=stream, flush=True)
                print(i, session, target, a, b, row["group"], "missing score mouth", flush=True)
                continue
            if valid:
                begin = time.perf_counter()
                outputs = {"mixture":mixture, "rolling":streaming(rolling,mixture,mouths,present)}
                outputs.update({name:streaming(model,mixture,mouths,present) for name, model in arms.items()})
                if old:
                    outputs["old_streaming"] = streaming(old,mixture,mouths)
                m = torch.from_numpy(mixture)[None].to(args.device)
                v = torch.from_numpy(mouths)[None,None].to(args.device)
                outputs["released_whole_window"] = audio_model(m,visual_model(v))[0,0].cpu().numpy()
                if row["group"] == "reopen":
                    if args.export_reopen_waveforms:
                        export_reopen_waveforms(args.output / "waveforms", observed, outputs, stems[:, target_index])
                    for part in measure_reopen(observed, outputs, stems, target_index):
                        print(json.dumps(part, allow_nan=False), file=stream, flush=True)
                    print(i, session, target, a, b, row["group"], "scored", flush=True)
                    continue
                x = mixture[score]
                reference = stems[score,target_index]
                measured = {}
                for name, output in outputs.items():
                    y = output[score]
                    if y.shape != x.shape or not np.isfinite(y).all():
                        raise RuntimeError("invalid scored waveform")
                    gains, powers, residual, rank = fit(stems[score],y)
                    others = np.delete(powers,target_index)
                    measured[name] = {
                        "output_to_mixture_db":db_ratio(float(np.mean(y**2)),float(np.mean(x**2))),
                        "target_si_sdr_db":si_sdr(y,reference) if row["group"] != "visible_silent" else None,
                        "target_fit_rms_dbfs":db_ratio(powers[target_index], 1),
                        "interferer_fit_to_mixture_db":db_ratio(others.sum(),float(np.mean(x**2))),
                        "reference_fit_residual_to_output_db":db_ratio(residual,float(np.var(y))),
                        "fit_rank":rank,"source_gains":gains.tolist(),"source_powers":powers.tolist(),
                        "target_to_interferers_db":db_ratio(powers[target_index],others.sum()),
                        "target_to_loudest_interferer_db":db_ratio(powers[target_index],others.max())}
                observed["scored_stem_rms_dbfs"] = [db_ratio(p,1) for p in np.mean(stems[score]**2,axis=0)]
                observed["metrics"] = measured
                observed["wall_seconds"] = time.perf_counter()-begin
                print(i,session,target,a,b,row["group"],"scored",flush=True)
            else:
                print(i,session,target,a,b,row["group"],"missing score mouth",flush=True)
            print(json.dumps(observed,allow_nan=False),file=stream,flush=True)
    if detector:
        detector.close()
    grouped = defaultdict(list)
    with (args.output / "rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row["group"]].append(row)
    with (args.output / "summary.json").open("x") as stream:
        json.dump({group: summarize_group(group, subset) for group, subset in grouped.items()},
                  stream, indent=2, allow_nan=False)


def summarize_group(group, subset):
    scored = [row for row in subset if "metrics" in row]
    duration_ms = sum(row["end_ms"] - row["start_ms"] for row in scored)
    metrics = {}
    for model in scored[0]["metrics"] if scored else ():
        records = [row["metrics"][model] for row in scored]
        silence = group in ("visible_silent", "reopen_pre_silent")
        primary = "output_to_mixture_db" if silence else "target_si_sdr_db"
        values = [record[primary] for record in records]
        result = {f"{primary}_mean": statistics.mean(values),
                  f"{primary}_median": statistics.median(values),
                  f"{primary}_minimum": min(values), f"{primary}_maximum": max(values)}
        if not silence:
            result["target_preferred"] = sum(record["target_to_loudest_interferer_db"] > 0 for record in records)
            result["better_than_mixture"] = sum(
                record[primary] > row["metrics"]["mixture"][primary]
                for record, row in zip(records, scored, strict=True))
        else:
            result["more_than_minus_10_db_mixture"] = sum(value > -10 for value in values)
        if group in ("reopen_120", "reopen_240"):
            for name in ("target_sisdri_db", "target_projection_delta_db"):
                result[name + "_mean"] = statistics.mean(record[name] for record in records)
                result[name + "_median"] = statistics.median(record[name] for record in records)
            for name in ("target_projection_below_minus6db", "target_projection_sign_reversed"):
                result[name] = sum(record[name] for record in records)
        metrics[model] = result
    return {"attempted":len(subset), "scored":len(scored), "duration_ms":duration_ms, "metrics":metrics}


if __name__ == "__main__":
    main()
