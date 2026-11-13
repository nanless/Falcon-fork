"""RoomReader independent shortlist on the published synchronized clock."""
import argparse
from functools import lru_cache
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

SOURCE = Path("/aispeech-cfs/voiceprint/data/multimodal/roomreader/raw")
SESSIONS = ("S05", "S06", "S11", "S15", "S16")
TICK_MS = 40

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def ranges(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a,b in zip(edges[::2], edges[1::2], strict=True)]

def labels(path):
    root = ET.parse(path)
    slots = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE"))
             for t in root.findall(".//TIME_SLOT") if t.get("TIME_VALUE")}
    people = defaultdict(dict)
    for tier in root.findall("TIER"):
        person = tier.get("PARTICIPANT")
        kind = tier.get("TIER_ID").rsplit("_", 1)[-1]
        if not person or kind not in {"Check","tok","Para"}:
            continue
        length = max(slots.values())//TICK_MS + 1
        mask = np.zeros(length, dtype=bool)
        for a in tier.findall(".//ALIGNABLE_ANNOTATION"):
            value = (a.findtext("ANNOTATION_VALUE") or "").strip()
            begin = slots.get(a.get("TIME_SLOT_REF1"))
            end = slots.get(a.get("TIME_SLOT_REF2"))
            if begin is None or end is None or not any(c.isalnum() for c in value):
                continue
            if kind == "Para" or kind=="Check":
                mask[begin//TICK_MS:(end+TICK_MS-1)//TICK_MS] = True
            else:
                mask[(begin+TICK_MS-1)//TICK_MS:end//TICK_MS] = True
        if kind in people[person]:
            raise ValueError("duplicate annotation tier")
        people[person][kind] = mask
    people = {name: mask for name,mask in sorted(people.items()) if {"Check","tok","Para"} <= mask.keys()}
    return {name: (m["Check"] & m["tok"] & ~m["Para"], m["Check"] | m["tok"] | m["Para"]) for name,m in people.items()}

def wav(path,start,end):
    with sf.SoundFile(path) as reader:
        if reader.samplerate != 32000 or reader.channels != 1:
            raise ValueError(f"unexpected WAV geometry {path}")
        reader.seek(start*1280)
        values = reader.read((end-start)*1280,dtype="float64")
    if len(values) != (end-start)*1280:
        raise ValueError(f"truncated WAV {path}")
    return values

def db(power):
    return float(10*np.log10(max(power,1e-15)))

@lru_cache(maxsize=None)
def openface_end_ms(path):
    with path.open("rb") as f:
        f.seek(max(0,path.stat().st_size-65536))
        times = []
        for line in f.read().splitlines()[1:]:
            v=line.split(b",")
            try:
                if len(v)>200 and int(v[0])>0 and int(v[1])==0 and 0<=float(v[3])<=1 and int(v[4]) in (0,1):
                    times.append(float(v[2]))
            except ValueError:
                continue
    return int(1000*max(times,default=-1))


def build(session, source, require_openface):
    annotation = source/"annotations/elan_sessions"/f"{session}.eaf"
    people = labels(annotation)
    names = list(people)
    if len(names) < 3:
        return []
    audio = {p: next((source/"audio/individual_participants_audio"/session).glob(f"{session}_{p}_*.wav")) for p in names}
    mix = source/"audio/all_participants_audio"/f"{session}.wav"
    speech = np.stack([people[p][0] for p in names])
    occupied = np.stack([people[p][1] for p in names])
    scores = []
    for i,target in enumerate(names):
        # Keep a fixed simultaneous set throughout each speech window.
        encoded = (speech * (2 ** np.arange(len(names))[:,None])).sum(0)
        for code in sorted(set(encoded)):
            keep = [j for j in range(len(names)) if code & (1<<j)]
            if i not in keep or len(keep)<2:
                continue
            group = "triple" if len(keep)>=3 else "weak"
            for a,b in sorted(ranges(encoded==code),key=lambda ab: (ab[0]-ab[1],ab[0]))[:24]:
                if b-a < (4 if group=="triple" else 8):
                    break
                scores.append((group,a,b,i,keep))
        others_active = speech.sum(0)-speech[i]
        silent = ~(occupied[i] | np.roll(occupied[i],1) | np.roll(occupied[i],-1))
        for a,b in sorted(ranges(silent & (others_active>=1)),key=lambda ab:(ab[0]-ab[1],ab[0]))[:9]:
            if b-a < 12:
                continue
            b=min(b,a+35)
            keep=[j for j in range(len(names)) if speech[j,a:b].any()]
            scores.append(("visible_silent",a,b,i,keep))
    result = []
    for group,a,b,i,keep in scores:
        target = names[i]
        active = [names[j] for j in keep]
        target_audio = wav(audio[target],a,b)
        individual = {p: wav(audio[p],a,b) for p in active if p != target}
        mixture = wav(mix,a,b)
        p_target = float(np.mean(target_audio**2))
        p_others = {p:float(np.mean(x**2)) for p,x in individual.items()}
        sir = db(p_target)-db(sum(p_others.values()))
        if not p_others or db(np.mean(mixture**2)) < -47:
            continue
        if group != "visible_silent" and (db(p_target)<-52 or min(map(db,p_others.values()))<-52):
            continue
        if group=="weak" and not -14 <= sir <= -6:
            continue
        if group=="visible_silent" and (sir>-20 or db(p_target)>-45):
            continue
        vdir=source/"video/individual_participants/individual_participants_audio_all"/session
        vids=list(vdir.glob(f"{session}_{target}_*_all.mp4"))
        if len(vids)!=1:
            continue
        vid=vids[0]
        feature=source/"features/OpenFace_Features"/(vid.stem+".csv")
        if not feature.is_file() or (require_openface and b*TICK_MS + 640 > openface_end_ms(feature)):
            continue
        result.append({"session":session,"target":target,"start_ms":a*40,"end_ms":b*40,
                       "group":group,"active":active,"native_sir_db":sir,
                       "target_rms_dbfs":db(p_target),"mix_rms_dbfs":db(np.mean(mixture**2)),
                       "source_rms_dbfs":{p:db(v) for p,v in p_others.items()},
                       "mix":str(mix),"target_audio":str(audio[target]),
                       "reference_audio":{p:str(audio[p]) for p in active},
                       "target_video":str(vid),"openface":str(feature)})
    print(session,len(scores),len(result),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out",type=Path)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--sessions", nargs="+", default=SESSIONS)
    ap.add_argument("--require-openface", action="store_true")
    args=ap.parse_args()
    source = args.source
    all_rows=[]
    for session in args.sessions:
        all_rows.extend(build(session, source, args.require_openface))
    rows=[]
    for group in ("triple","visible_silent","weak"):
        subset=[x for x in all_rows if x["group"]==group]
        subset.sort(key=lambda row:(row["start_ms"]-row["end_ms"],abs(row["native_sir_db"]+9)))
        selected=[]
        for row in subset:
            if sum(r["session"]==row["session"] for r in selected)>=3:
                continue
            if any(r["session"]==row["session"] and abs(r["start_ms"]-row["start_ms"])<2000 for r in selected):
                continue
            selected.append(row)
            if len(selected)==12:
                break
        rows+=selected
    with args.out.open("x") as f:
        json.dump({"source":str(source),"mix_protocol":"native published conversation mix, never resynthesized",
                   "source_url":"https://sigmedia.tv/datasets/room_reader/",
                   "license":"RoomReader Non-Commercial Licence Agreement; internal research use",
                                      "require_openface_timestamp_coverage": args.require_openface,
                   "visible_silent": "Continuous mouth visibility must be verified in decoded video.",
                   "weak":"individual stream power ratio prior to verifying exact mix mapping",
                   "annotation_sha256":{s:sha(source/"annotations/elan_sessions"/f"{s}.eaf") for s in args.sessions},
                   "rows":rows},f,indent=2,allow_nan=False)
if __name__=="__main__":
    main()
