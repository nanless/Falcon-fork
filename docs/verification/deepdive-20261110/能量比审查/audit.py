import csv, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf
out=Path(sys.argv[1]); base=Path('/root/data/falcon_lrs2_2mix_paper/tr')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
index_paths={k:base/(k+'.json') for k in ('mix','target','near')}
idx={k:json.loads(p.read_text()) for k,p in index_paths.items()}
rows=[]
for i in range(0,len(idx['mix']),40):
 mix,tgt,near=(idx[k][i] for k in ('mix','target','near'))
 stem=Path(mix[0]).stem; parts=stem.split('_'); assert len(parts)==6
 ap,bp=Path(tgt[0]),Path(near[0][0]); assert ap.parent.name=='s1' and bp.parent.name=='s2' and ap.stem==stem and bp.stem==stem
 a,sra=sf.read(ap,dtype='float32'); b,srb=sf.read(bp,dtype='float32'); assert sra==srb==16000 and a.shape==b.shape==(32000,)
 a=a.astype('float64'); b=b.astype('float64'); actual=10*np.log10(np.sum(a*a)/np.sum(b*b)); filename=float(parts[2])-float(parts[5])
 rows.append(dict(index=i,mix_stem=stem,target_path=str(ap),near_path=str(bp),target_sha256=sha(ap),near_sha256=sha(bp),sample_rate=sra,n_samples=len(a),filename_gain_difference_db=filename,waveform_energy_ratio_db=float(actual),difference_db=float(actual-filename)))
with (out/'samples.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
x=np.array([r['waveform_energy_ratio_db'] for r in rows]);err=np.array([r['difference_db'] for r in rows]); summary={'repo_head':subprocess.check_output(['git','-C','/root/code/github_repos/Falcon-fork','rev-parse','HEAD'],text=True).strip(),'input_indexes':{k:{'path':str(v),'sha256':sha(v),'entries':len(idx[k])} for k,v in index_paths.items()},'selection':'zero-based index range(0, len(mix.json), 40); alternating paper directions make these s1-target entries; no random sampling','waveform_measurement':'10*log10(sum(s1_pcm_float64**2)/sum(s2_pcm_float64**2)), no mean removal; files decoded PCM as float32 then promoted float64','count':len(rows),'min_db':float(x.min()),'max_db':float(x.max()),'outside_0_10':int(((x<0)|(x>10)).sum()),'s1_energy_less_than_s2':int((x<0).sum()),'max_abs_difference_db':float(np.max(np.abs(err))),'samples_csv_sha256':sha(out/'samples.csv'),'script_sha256':sha(__file__)}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2));print('ARTIFACT_DIR',out)
