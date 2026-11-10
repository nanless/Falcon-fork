"""Perturb real audio/video after time boundaries and verify prior CPU output."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tools.streaming.common import load_mono, mouth_tensor, read_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--index-dir',required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--output',required=True)
    a=p.parse_args(); path=Path(a.output)
    if path.exists(): raise FileExistsError(path)
    model_hash_before=sha('look2hear/models/FalconStreaming.py'); checkpoint_hash_before=sha(a.checkpoint)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    root=Path(a.index_dir);mix=json.loads((root/'mix.json').read_text())[0]; target=json.loads((root/'target.json').read_text())[0]
    if Path(mix[0]).name!=Path(target[0]).name: raise ValueError('index identity mismatch')
    x=load_mono(mix[0],0,5120).unsqueeze(0)
    y=mouth_tensor(torch.load(target[1],map_location='cpu',weights_only=True)[:8]).unsqueeze(0)
    if x.shape!=(1,5120) or y.shape!=(1,8,88,88):raise ValueError('two complete aligned chunks required')
    model=FalconStreaming().eval(); ckpt=read_checkpoint(a.checkpoint,'cpu');model.load_state_dict(ckpt['state_dict'],strict=True)
    gen=torch.Generator().manual_seed(20261111)
    rows=[]
    with torch.inference_mode():
        baseline=model(x,y)
        for cut in (1,127,128,129,639,640,641,1280,2432,2559,2560,2561,3840,4992):
            altered=x.clone(); altered[:,cut:] += .4*torch.randn(altered[:,cut:].shape,generator=gen)
            out=model(altered,y); safe=(cut//128)*128
            prior=(out[:,:safe]-baseline[:,:safe]).abs().max().item() if safe else None
            after=(out[:,cut:]-baseline[:,cut:]).abs().max().item()
            if prior is not None and prior>2e-6:raise AssertionError(f'audio cut {cut}: leakage {prior}')
            if after<1e-7:raise AssertionError(f'audio cut {cut}: perturbation had no observed effect')
            rows.append({'kind':'audio','cut_samples':cut,'prior_full_hop_samples':safe,'prior_max_abs_delta':prior,'after_max_abs_delta':after})
        for frame in (1,2,3,4,5,7):
            altered=y.clone();altered[:,frame:]+=2*torch.randn(altered[:,frame:].shape,generator=gen)
            out=model(x,altered);cut=frame*640
            prior=(out[:,:cut]-baseline[:,:cut]).abs().max().item()
            after=(out[:,cut:]-baseline[:,cut:]).abs().max().item()
            if prior>2e-6:raise AssertionError(f'video frame {frame}: leakage {prior}')
            if after<1e-9:raise AssertionError(f'video frame {frame}: perturbation had no observed effect')
            rows.append({'kind':'video','cut_frame':frame,'cut_samples':cut,'prior_max_abs_delta':prior,'after_max_abs_delta':after})
    if model_hash_before!=sha('look2hear/models/FalconStreaming.py') or checkpoint_hash_before!=sha(a.checkpoint):
        raise RuntimeError('model or checkpoint changed during audit')
    result={'date':'2026-11-11','scope':'CPU real aligned LRS2 utterance, two chunks, random checkpoint; only complete 8ms hops before audio perturbation are asserted',
      'seed':20261111,'threads':a.threads,'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':checkpoint_hash_before,
      'model_sha256':model_hash_before,'audio':mix[0],'audio_sha256':sha(mix[0]),'mouth':target[1],'mouth_sha256':sha(target[1]),
      'rows':rows}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'audio_cases':14,'video_cases':6,'max_prior_delta':max(row['prior_max_abs_delta'] or 0 for row in rows),'output':str(path)},ensure_ascii=False))

if __name__=='__main__':main()
