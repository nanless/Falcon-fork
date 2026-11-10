"""Reproduce frame alignment A/B with existing audio, CPU and frozen checkpoint.
Run at Falcon-fork repo root with the falcon conda Python, see JSON command field.
This writes only /tmp/falcon_frame_alignment_probe_20261110.json.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch

repo=Path("/root/code/github_repos/Falcon-fork")
ckpt=Path("/root/.cache/falcon_weights/falcon.pt")
out=Path("/tmp/falcon_frame_alignment_probe_20261110.json")
def sha256(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("scprobe",repo/"tools/repro/experiments/streaming_consistency.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
from falcon_runtime import FalconRuntime
torch.set_num_threads(4)
rt=FalconRuntime("small",checkpoint_path=ckpt).load("cpu")
audio,mouth,target,chosen,source=s.pick_stream(s.DEFAULT_DATA_ROOT,3)
reference,_=s.whole_pass(rt,audio,mouth)
first_end=s.WINDOW
count=(len(audio)-first_end)//s.HOP+1
old,_=s.predict_grid(rt,audio,mouth,first_end,count)
def corrected(mouth,start,count):
 f0=int(np.ceil(start/s.SAMPLES_PER_FRAME-0.5))
 return mouth[np.clip(f0+np.arange(count),0,len(mouth)-1)]
s.frames_block=corrected
new,_=s.predict_grid(rt,audio,mouth,first_end,count)
yold,lo,hi=s.stitch(old,first_end,s.STUDIO_GUARD,len(audio))
ynew,_,_=s.stitch(new,first_end,s.STUDIO_GUARD,len(audio))
files=[Path(x) for x in chosen]
files += [s.DEFAULT_DATA_ROOT/"target"/Path(x).name for x in chosen]
files += [s.DEFAULT_DATA_ROOT/"mouths"/(source+".pt")]
files += [repo/"tools/repro/experiments/streaming_consistency.py",repo/"studio/falcon_runtime.py",repo/"configs/Falcon.yml",repo/"studio/configs/Falcon.yml"]
result={"repo_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
 "torch":torch.__version__,"cpu_threads":torch.get_num_threads(),"seed":"no random input; existing dataset",
 "checkpoint":{"path":str(ckpt),"sha256":sha256(ckpt)},
 "input_and_code_sha256":{str(f):sha256(f) for f in files},
 "script_sha256":sha256(Path(__file__)),
 "source_id":source,"windows":count,"range_samples":[lo,hi],"guard_ms":80,"anchor":"grid",
 "assumption":"Each source crop n has timestamp (n+0.5)*640; ties select earlier crop; clipping repeats boundary frames.",
 "old_vs_whole":s.compare(yold,reference,lo,hi),"corrected_vs_whole":s.compare(ynew,reference,lo,hi),
 "old_vs_corrected":s.compare(yold,ynew,0,len(ynew)),"old_corrected_max_abs":float(np.max(np.abs(yold-ynew))),
 "command":"cd /root/code/github_repos/Falcon-fork && PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python /tmp/falcon_frame_alignment_probe_20261110.py"}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
