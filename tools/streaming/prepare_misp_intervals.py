"""Prepare original far-mic recordings and specified mouth tracks for MISP probes."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

SESSION = "R71_S412413414415416417_C04_I1"
SAMPLES_PER_TICK = 640
CONTEXT_BEFORE_TICKS = 20
CONTEXT_AFTER_TICKS = 5

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def crop(image, coordinates):
    left, top, right, bottom = coordinates
    size = max(96,right-left,bottom-top)
    x = (left+right-size)//2
    y = (top+bottom-size)//2
    roi=image[max(0,y):min(image.shape[0],y+size),max(0,x):min(image.shape[1],x+size)]
    if roi.size==0:
        raise ValueError("empty mouth crop")
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    if roi.shape!=(96,96):
        roi=cv2.resize(roi,(96,96),interpolation=cv2.INTER_AREA)
    return roi

def choose(shortlist, detections, session, count):
    rows = [r for r in shortlist if r["session"] == session]
    chosen=[]
    for group in ("triple","visible_silent"):
        current=[]
        for row in rows:
            if row["group"]!=group:
                continue
            if group=="triple" and len(row["active_throughout"])<3:
                continue
            a,b=(row[k]//40 for k in ("start_ms","end_ms"))
            if any(abs(a-x["start_ms"]//40)<80 for x in current):
                continue
            other = next((p for p in row["active_throughout"] if p != row["target"]
                          and all(p in detections.get(f,{}) for f in range(a,b))),None)
            if group=="triple" and other is None:
                continue
            row=dict(row,alternate_mouth=other)
            current.append(row)
            if len(current) == count:
                break
        chosen+=current
    return chosen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--source-media", type=Path, help="Directory containing SESSION.mp4, SESSION.wav and SESSION.json.")
    p.add_argument("--shortlist", type=Path, help="Output of misp_select.py.")
    p.add_argument("--session", default=SESSION)
    p.add_argument("--count", type=int, default=6, help="Maximum clips per group.")
    a = p.parse_args()
    if a.count < 1:
        p.error("--count must be positive")
    cv2.setNumThreads(1)
    root = a.source_media or a.base / "misp_media"
    mouth_json=root/f"{a.session}.json"
    video=root/f"{a.session}.mp4"
    audio=root/f"{a.session}.wav"
    detections={int(f):{str(d["id"]):d["lip"] for d in values if len(d["lip"])==4
                       and d["lip"][2]>d["lip"][0] and d["lip"][3]>d["lip"][1]}
                for f,values in json.loads(mouth_json.read_text()).items()}
    shortlist = a.shortlist or a.base / "misp_shortlist.json"
    rows=choose(json.loads(shortlist.read_text())["rows"], detections, a.session, a.count)
    if not rows:
        raise ValueError("No matching clips for the requested session")
    a.out.mkdir(exist_ok=False)
    stream=cv2.VideoCapture(str(video))
    if stream.get(cv2.CAP_PROP_FPS)!=25:
        raise ValueError("unexpected fps")
    nframes=int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    reports=[]
    with sf.SoundFile(audio) as reader:
        if reader.samplerate!=16000 or reader.channels!=1:
            raise ValueError("unexpected audio clock")
        for row in rows:
            a_tick,b_tick=(row[k]//40 for k in ("start_ms","end_ms"))
            start=max(0,a_tick-CONTEXT_BEFORE_TICKS)
            stop=min(b_tick+CONTEXT_AFTER_TICKS,nframes)
            if b_tick>stop:
                raise ValueError("score extends beyond video")
            if not stream.set(cv2.CAP_PROP_POS_FRAMES,start):
                raise ValueError("seek failed")
            primary=row["target"]
            alternate=row["alternate_mouth"]
            people=[primary]+([alternate] if alternate else [])
            mouth={p:np.zeros((stop-start,96,96),np.uint8) for p in people}
            valid={p:np.zeros(stop-start,bool) for p in people}
            for n in range(start,stop):
                if int(stream.get(cv2.CAP_PROP_POS_FRAMES))!=n:
                    raise ValueError("wrong source frame")
                ok,im=stream.read()
                if not ok: raise ValueError("frame not decoded")
                for p in people:
                    if p in detections.get(n,{}):
                        mouth[p][n-start]=crop(im,detections[n][p])
                        valid[p][n-start]=True
            score=slice(a_tick-start,b_tick-start)
            if not all(valid[p][score].all() for p in people):
                raise ValueError("missing designated mouth in score interval")
            reader.seek(start*SAMPLES_PER_TICK)
            signal=reader.read((stop-start)*SAMPLES_PER_TICK,dtype="float32")
            if len(signal)!=(stop-start)*SAMPLES_PER_TICK:
                raise ValueError("truncated audio")
            out=a.out/(row["group"]+"_"+row["target"]+"_"+str(row["start_ms"])+".npz")
            with out.open("xb") as f:
                np.savez_compressed(f,audio=signal,mouth=mouth[primary],mouth_valid=valid[primary],
                      alternate=mouth[alternate] if alternate else np.empty(0,dtype=np.uint8),
                      alternate_valid=valid[alternate] if alternate else np.empty(0,dtype=bool),
                      score_start=(a_tick-start)*SAMPLES_PER_TICK,score_stop=(b_tick-start)*SAMPLES_PER_TICK)
            reports.append(dict(row,path=str(out),sha256=sha(out),clip_start_ms=start*40,
                                clip_stop_ms=stop*40,mouth_context_missing_ticks=int((~valid[primary]).sum())))
            print(row["group"],primary,alternate,row["start_ms"],row["end_ms"],str(out),flush=True)
    stream.release()
    with (a.out/"manifest.json").open("x") as f:
        json.dump({"source":"MISP2022 AVSD evaluation original Far_0 and official lip bounding boxes",
                   "source_video":str(video),"source_audio":str(audio),"source_mouth":str(mouth_json),
                   "shortlist_sha256":sha(shortlist),"source_audio_sha256":sha(audio),
                   "source_mouth_sha256":sha(mouth_json),"rows":reports},f,indent=2)
if __name__=="__main__": main()
