"""Measure single-track extraction on native MISP Far_0 and supplied lip tracks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from tools.streaming.evaluate_realssa import load_baseline
from tools.streaming.paper_live_inference import PaperLiveInference
from tools.streaming.rolling_falcon_live import RollingFalconLiveInference, TICK_SAMPLES

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def power(wave):
    return float(np.mean(np.square(wave, dtype=np.float64)))

def ratio_db(numer,denom):
    return float(10*np.log10(max(numer,1e-12)/max(denom,1e-12)))

def normalized(images):
    return (images[:,4:92,4:92].astype(np.float32)/255-.421)/.165

def live(stream,audio,mouth,valid,mask_missing):
    stream.reset()
    blocks=[]
    for i,frame in enumerate(mouth):
        x=audio[i*TICK_SAMPLES:(i+1)*TICK_SAMPLES]
        available = bool(valid[i])
        blocks.extend(stream.push(x,frame if available else (None if mask_missing else np.zeros_like(frame)),sequence=i))
    blocks.extend(stream.finish())
    return np.concatenate(blocks)

def evaluate(label,path,device,manifest,mode):
    if mode=="full":
        network, visual, _ = load_baseline(path,device)
        def predict(audio,mouth,valid):
            video=np.where(valid[:,None,None],mouth,0.)
            a=torch.from_numpy(audio).to(device)[None]
            v=torch.from_numpy(video).to(device)[None,None]
            with torch.inference_mode():
                result=network(a,visual(v))
            if result.shape!=(1,1,len(audio)):
                raise ValueError("released Falcon did not return a single aligned target")
            return result[0,0].cpu().numpy()
    else:
        if mode == "experiment":
            from tools.streaming.rolling_model_checkpoint import RollingExperimentLiveInference

            stream = RollingExperimentLiveInference(path, device=device)
        elif mode == "paper":
            stream = PaperLiveInference(path, device=device)
        else:
            stream = RollingFalconLiveInference(path, device=device)
        def predict(audio,mouth,valid):
            return live(stream,audio,mouth,valid,mode!="paper")
    rows=[]
    for entry in manifest["rows"]:
        sample=Path(entry["path"])
        if sha(sample)!=entry["sha256"]:
            raise ValueError(f"input hash differs: {sample}")
        with np.load(sample,allow_pickle=False) as payload:
            audio=payload["audio"]
            score=slice(int(payload["score_start"]),int(payload["score_stop"]))
            va=payload["mouth_valid"]
            mouth=normalized(payload["mouth"])
            y=predict(audio,mouth,va)
            if y.shape!=audio.shape or not np.isfinite(y).all():
                raise ValueError("model output geometry or numeric error")
            record={"group":entry["group"],"target":entry["target"],
                    "score_duration_ms":entry["end_ms"]-entry["start_ms"],
                    "start_ms":entry["start_ms"],"mix_power":power(audio[score]),
                    "output_power":power(y[score]),"score_visible_ticks":int(va[score.start//TICK_SAMPLES:score.stop//TICK_SAMPLES].sum())}
            record["output_to_mix_db"]=ratio_db(record["output_power"],record["mix_power"])
            if payload["alternate"].size:
                alternate=predict(audio,normalized(payload["alternate"]),payload["alternate_valid"])
                if alternate.shape!=audio.shape or not np.isfinite(alternate).all():
                    raise ValueError("alternate output geometry or numeric error")
                record.update(alternate_target=entry["alternate_mouth"],
                              difference_to_mix_db=ratio_db(power((y-alternate)[score]),record["mix_power"]),
                              difference_to_output_db=ratio_db(power((y-alternate)[score]),
                                                               .5*(power(y[score])+power(alternate[score]))))
        rows.append(record)
        print(label,record["group"],record["start_ms"],flush=True)
    groups={}
    for group,metric in (("triple","difference_to_mix_db"),("visible_silent","output_to_mix_db")):
        subset=[r for r in rows if r["group"]==group]
        values=[r[metric] for r in subset]
        groups[group]={"count":len(subset),"total_score_seconds":sum(r["score_duration_ms"] for r in subset)/1000,
                       "metric":metric,"mean_db":statistics.mean(values),"median_db":statistics.median(values),
                       "min_db":min(values),"max_db":max(values)}
    triple=[r for r in rows if r["group"]=="triple"]
    groups["triple"]["mean_output_to_mix_db"]=statistics.mean(r["output_to_mix_db"] for r in triple)
    groups["triple"]["mean_difference_to_output_db"]=statistics.mean(r["difference_to_output_db"] for r in triple)
    return {"weights_sha256":sha(path),"mode":mode,"summary":groups,"rows":rows}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",default=2,type=int)
    parser.add_argument("--model",action="append",nargs=3,metavar=("LABEL","MODE","CHECKPOINT"),required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads(args.manifest.read_text())
    models={}
    for label,mode,path in args.model:
        if mode not in ("full","paper","rolling","experiment") or label in models:
            raise ValueError(f"invalid model descriptor: {label} {mode}")
        models[label]=evaluate(label,Path(path),args.device,manifest,mode)
        print(label,models[label]["summary"],flush=True)
    with args.output.open("x") as f:
        json.dump({"source_manifest_sha256":sha(args.manifest),"source_sha256":sha(__file__),
                   "torch":torch.__version__,"device":args.device,"models":models},f,indent=2,allow_nan=False)

if __name__=="__main__": main()
