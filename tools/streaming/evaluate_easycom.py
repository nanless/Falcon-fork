"""Prepare and score an independently captured EasyCom acoustic overlap holdout."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from tools.streaming.rolling_falcon_live import RollingFalconLiveInference

RATIO = 800
TICK = 640
LANDMARKS = (61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def audio(path, start, stop, channel=0):
    with sf.SoundFile(path) as source:
        assert source.samplerate == 48000
        source.seek(start * 2400)
        y = source.read((stop-start)*2400,dtype="float32",always_2d=True)[:,channel]
    assert len(y) == (stop-start)*2400
    return resample_poly(y,1,3).astype("float32")


def source(root, group, session, filename):
    return root/group/session/filename


def frames(detector, video, face, participant, first, last):
    lookup = {row["Frame_Number"]-1:next((p for p in row["Participants"] if p["Participant_ID"]==participant),None)
              for row in face}
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    fps = capture.get(cv2.CAP_PROP_FPS)
    assert abs(fps-20)<1e-3
    capture.set(cv2.CAP_PROP_POS_FRAMES,first)
    outputs=[]
    try:
        for i in range(first,last):
            ok, frame = capture.read()
            box = lookup.get(i)
            patch = None
            if ok and box:
                x1,y1,x2,y2=(int(box[k]) for k in ("x1","y1","x2","y2"))
                w,h=x2-x1,y2-y1
                x1,y1=max(0,x1-w//6),max(0,y1-h//6)
                x2,y2=min(frame.shape[1],x2+w//6),min(frame.shape[0],y2+h//6)
                crop = frame[y1:y2,x1:x2]
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
                if result.face_landmarks:
                    lm=result.face_landmarks[0]
                    xy=np.array([(lm[j].x*(x2-x1)+x1,lm[j].y*(y2-y1)+y1) for j in LANDMARKS])
                    center=xy.mean(0)
                    width=float(np.linalg.norm(xy[0]-xy[10]))
                    span=width*2.5
                    if 15 < span < 180 and 0 < center[0] < frame.shape[1] and 0 < center[1] < frame.shape[0]:
                        patch=cv2.getRectSubPix(frame,(round(span),round(span)),tuple(center))
                        patch=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
                        patch=cv2.resize(patch,(96,96),interpolation=cv2.INTER_AREA)[4:92,4:92]
                        patch=(patch.astype(np.float32)/255-.421)/.165
            outputs.append(patch)
    finally:
        capture.release()
    # 20 Hz source frames are held on the 25 Hz clock. Include the final
    # partial audio tick when the requested interval is off the common grid.
    ticks = ((last - first) * RATIO + TICK - 1) // TICK
    return [outputs[(i * 4) // 5] for i in range(ticks)]


def power(y):
    return float(np.mean(np.square(y,dtype=np.float64)))


def ratio(y,x):
    return 10*np.log10(max(power(y),1e-12)/max(power(x),1e-12))


def sisdr(y,x):
    x=x.astype(np.float64)-float(np.mean(x,dtype=np.float64))
    y=y.astype(np.float64)-float(np.mean(y,dtype=np.float64))
    fit=(y@x)/max(x@x,1e-12)*x
    return float(ratio(fit,y-fit))


def infer(model, mix, mouths):
    model.reset()
    pieces=[]
    for i, mouth in enumerate(mouths):
        pieces.extend(model.push(mix[i*TICK:(i+1)*TICK],mouth,sequence=i))
    pieces.extend(model.finish())
    output=np.concatenate(pieces)
    if output.shape != mix.shape or not np.isfinite(output).all():
        raise ValueError("expected one finite output waveform on the original audio clock")
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate_windows",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-root", type=Path, required=True, help="EasyCom Main directory")
    ap.add_argument("--face-model", type=Path, required=True)
    ap.add_argument("--released", type=Path, required=True)
    ap.add_argument("--warm", type=Path)
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--experiment-arm", action="append", nargs=2, metavar=("LABEL", "CHECKPOINT"), default=[])
    ap.add_argument("--per-group", type=int, default=5)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cv2.setNumThreads(1)
    if args.per_group < 1:
        ap.error("--per-group must be positive")
    if bool(args.warm) != bool(args.scratch):
        ap.error("--warm and --scratch must be supplied together")
    weights = {"published": args.released}
    if args.warm:
        weights.update(warm96=args.warm, scratch96=args.scratch)
    experiment_arms = {label: Path(path) for label, path in args.experiment_arm}
    if set(weights) & set(experiment_arms) or len(experiment_arms) != len(args.experiment_arm):
        ap.error("duplicate model labels")
    streams = {name: RollingFalconLiveInference(path, device=args.device, threads=2)
               for name, path in weights.items()}
    if experiment_arms:
        from tools.streaming.rolling_model_checkpoint import RollingExperimentLiveInference

        streams.update({name: RollingExperimentLiveInference(path, device=args.device)
                        for name, path in experiment_arms.items()})
        weights.update(experiment_arms)
    windows=json.loads(args.candidate_windows.read_text())
    opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.face_model)),
            num_faces=1,min_face_detection_confidence=.55)
    n_goal = dict.fromkeys(("overlap3", "visible_silent"), args.per_group)
    used=set()
    counts={k:0 for k in n_goal}
    rows=[]
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as detector:
        for row in windows:
            group=row["type"]
            if counts[group]>=n_goal[group]: continue
            session, key, who=row["session"],row["file"],row["target"]
            if (session,key) in used or (group=="visible_silent" and row["n_frames"]<24): continue
            head_dir=args.dataset_root/"Close_Microphone_Audio"/session
            headsets={int(p.stem.rsplit("_",1)[1]):p for p in head_dir.glob(key+"_Participant_ID_*.wav")}
            if who not in headsets: continue
            a,b=row["start_frame"]-1,row["end_frame_excl"]-1
            b=min(b,a+60)
            b=a+(b-a)//4*4
            if b-a < 12: continue
            first=max(0,a-12)
            stop=b+4
            video=source(args.dataset_root,"Video_Compressed",session,key+".mp4")
            wave=source(args.dataset_root,"Glasses_Microphone_Array_Audio",session,key+".wav")
            face=Path(row["face_boxes"])
            boxes=json.loads(face.read_text())
            if not video.is_file() or not wave.is_file() or stop>len(boxes): continue
            m=frames(detector,video,boxes,who,first,stop)
            score=slice((a-first)*RATIO,(b-first)*RATIO)
            frame_score=slice((a-first)*5//4,(b-first)*5//4)
            valid=sum(x is not None for x in m[frame_score])/len(m[frame_score])
            if valid<.95:
                print("reject_landmark",group,session,key,who,round(valid,3),flush=True)
                continue
            mix=audio(wave,first,stop,channel=4)
            refs={i:audio(path,first,stop)[score] for i,path in headsets.items()
                  if i==who or i in row["active_others"]}
            x=mix[score]
            result={"group":group,"session":session,"file":key,"target":who,"start_s":a/20,
                    "end_s":b/20,"annotated_other_ids":row["active_others"],
                    "scored_headset_ids":sorted(refs),"visible_fraction":valid, "models":{}}
            for name,model in streams.items():
                y=infer(model,mix,m)[score]
                score_row={"residual_power_db":float(ratio(y,x))}
                if group=="overlap3":
                    score_row["target_sisdr_in_db"]=sisdr(x,refs[who])
                    score_row["target_sisdr_out_db"]=sisdr(y,refs[who])
                    score_row["target_sisdri_db"]=score_row["target_sisdr_out_db"]-score_row["target_sisdr_in_db"]
                    score_row["other_headset_sisdr_out_db"]={i:sisdr(y,r) for i,r in refs.items() if i!=who}
                result["models"][name]=score_row
            result["input_sha256"]={k:sha(v) for k,v in (("annotation",Path(row["annotation"])),("face_boxes",face))}
            result["audio_slice_sha256"] = hashlib.sha256(mix.tobytes()).hexdigest()
            result["mouth_slice_sha256"] = hashlib.sha256(b"".join(
                np.zeros((88,88), dtype=np.float32).tobytes() if patch is None else patch.tobytes() for patch in m)).hexdigest()
            result["reference_slice_sha256"]={i:hashlib.sha256(r.tobytes()).hexdigest() for i,r in refs.items()}
            rows.append(result)
            counts[group]+=1
            used.add((session,key))
            print(group,session,key,who,{k:round(v.get("target_sisdri_db",v["residual_power_db"]),2) for k,v in result["models"].items()},flush=True)
            if counts==n_goal:break
    identity={"candidate_sha256":sha(args.candidate_windows),"code_sha256":sha(Path(__file__)),"landmark_sha256":sha(args.face_model),
              "models_sha256":{k:sha(path) for k,path in weights.items()},"mic_channel_one_based":5,"counts":counts,"rows":rows}
    if counts != n_goal:
        raise RuntimeError(f"accepted only {counts}, expected {n_goal}")
    with (args.output/"result.json").open("x") as f:
        json.dump(identity,f,indent=2,allow_nan=False)

if __name__=="__main__":
    main()
