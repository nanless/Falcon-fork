"""Find natural in-room overlap and visible silence from MISP2022 AVSD labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SOURCE = Path("/aispeech-cfs/voiceprint/data/speaker_diarization_catalog/misp_home_tv_2021_2022/payload/files/avsd")
TICK = .04

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def ranges(mask):
    edges=np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a),int(b)) for a,b in zip(edges[::2], edges[1::2], strict=True)]

def select(source, split):
    with ZipFile(source/f"{split}_far_RTTM.zip") as labels, ZipFile(source/f"{split}_far_detection_result.zip") as visuals:
        results=[]
        for member in sorted(labels.namelist()):
            if not member.endswith(".rttm"):continue
            scene=Path(member).stem
            raw_labels=labels.read(member)
            v_member=f"{split}_far_detection_result/{scene}_Far.json"
            raw_visual=visuals.read(v_member)
            visual=json.loads(raw_visual)
            times=defaultdict(list)
            for line in raw_labels.decode().splitlines():
                fields=line.split()
                if len(fields)!=10 or fields[0]!="SPEAKER" or fields[1]!=scene:
                    raise ValueError(f"bad RTTM line {scene}")
                a, length=map(float,fields[3:5])
                times[fields[7]].append((a,a+length))
            ids=sorted(times)
            n=min(max(map(int,visual))+1,int(max(b for entries in times.values() for _,b in entries)/TICK)+1)
            active=np.zeros((len(ids),n),bool)
            occupied=active.copy()
            visible=active.copy()
            for j,person in enumerate(ids):
                for start,end in times[person]:
                    active[j,int(np.ceil(start/TICK)):int(end/TICK)]=True
                    occupied[j,max(0,int(start/TICK)-2):int(np.ceil(end/TICK))+2]=True
            lookup={p:j for j,p in enumerate(ids)}
            for key,detections in visual.items():
                frame=int(key)
                if frame>=n:continue
                for d in detections:
                    who=lookup.get(str(d["id"]))
                    lip=d.get("lip")
                    if who is not None and len(lip)==4 and lip[2]>lip[0] and lip[3]>lip[1]:
                        visible[who,frame]=True
            for j,person in enumerate(ids):
                candidates={"triple":active[j] & (active.sum(0)>=3) & visible[j],
                            "visible_silent":~occupied[j] & (active.sum(0)>=1) & visible[j]}
                for group,mask in candidates.items():
                    for a,b in sorted(ranges(mask),key=lambda pair:(pair[0]-pair[1],pair[0]))[:4]:
                        if b-a<(6 if group=="triple" else 20):continue
                        if group=="visible_silent":
                            b=min(b,a+35)
                        results.append({"session":scene,"split":split,"target":person,"group":group,
                                        "start_ms":40*a,"end_ms":40*b,
                                        "minimum_active_count":int(active[:,a:b].sum(0).min()),
                                        "active": [p for k,p in enumerate(ids) if active[k,a:b].any()],
                                        "active_throughout":[p for k,p in enumerate(ids) if active[k,a:b].all()],
                                        "rttm_member":member,"rttm_sha256":sha(raw_labels),
                                        "mouth_member":v_member,"mouth_sha256":sha(raw_visual),
                                        "mix_member":f"{split}_far_audio/{scene}_Far_0.wav",
                                        "video_member":f"{split}_far_video/{scene}_Far.mp4"})
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out",type=Path)
    ap.add_argument("--source", type=Path, default=SOURCE, help="Directory containing the AVSD ZIP files.")
    ap.add_argument("--splits", nargs="+", default=("dev", "eval"), choices=("dev", "eval"))
    args=ap.parse_args()
    rows = [row for split in args.splits for row in select(args.source, split)]
    rows.sort(key=lambda row:(row["group"],row["start_ms"]-row["end_ms"],row["session"],row["target"]))
    with args.out.open("x") as f:
        json.dump({"source":str(args.source),"source_url":"https://challenge.xfyun.cn/misp_dataset",
                   "protocol":"25 Hz candidates, RTTM whole tick and official detected target lip box on every score tick. Verify native video fps and 0/1-index shift when preparing inputs.",
                   "rows":rows},f,indent=2,allow_nan=False)
    print({k:sum(r["group"]==k for r in rows) for k in ("triple","visible_silent")},flush=True)
if __name__=="__main__": main()
