"""Read-only path, timing, pairing and split audit of resident Falcon indexes."""
import argparse, collections, hashlib, json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='docs/data_catalog/07-project-resident-audit-20261111.json',help='new JSON path; never overwrites an existing file')
args=parser.parse_args()
out=Path(args.output)
if out.exists(): raise FileExistsError(f'output already exists: {out}')
torch.set_num_threads(1)
roots={'lrs2_single':Path('/root/data/falcon_lrs2_2mix'), 'lrs2_paper':Path('/root/data/falcon_lrs2_2mix_paper'), 'realssa_mean':Path('/root/data/falcon_realssa_sim'), 'realssa_ch0':Path('/root/data/falcon_realssa_sim_ch0')}
pattern=re.compile(r'^(\d+)_(\d+)_(-?\d+(?:\.\d+)?)_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def audio(path):
 v,sr=sf.read(path,dtype='float64',always_2d=True)
 return v,sr

groups={}; paths=set(); sets={}; meta={}; start=time.time()
for name,root in roots.items():
 for split in ('tr','cv','tt'):
  p=root/split
  if not p.is_dir():continue
  inputs=[p/'mix.json',p/'target.json',p/'near.json']
  if not all(f.exists() for f in inputs):continue
  mix,target,near=[json.loads(f.read_text()) for f in inputs]
  if not (len(mix)==len(target)==len(near)):raise ValueError(f'row mismatch: {p}')
  paths_group={'mix':set(),'target':set(),'mouth':set(),'near':set()}
  invalid=collections.Counter(); lengths=collections.Counter(); identities=set(); videos=set(); parsed=0; self_near=0
  for m,t,n in zip(mix,target,near):
   if len(m)<2 or len(t)<3:invalid['malformed']+=1;continue
   paths_group['mix'].add(m[0]);paths_group['target'].add(t[0]);paths_group['mouth'].add(t[1]);lengths[int(m[1])]+=1
   if int(m[1])!=int(t[2]):invalid['mix_target_lengths']+=1
   if Path(m[0]).name!=Path(t[0]).name:invalid['mix_target_basenames']+=1
   if len(n)!=1:invalid['near_entry_count_not_one']+=1
   for pair in n:
    if len(pair)>=2:
     paths_group['near'].add(pair[0]);self_near+=pair[0]==m[0]
     if int(pair[1])!=int(m[1]):invalid['near_length']+=1
   match=pattern.fullmatch(Path(m[0]).stem)
   if match:
    parsed+=1;g=match.groups();identities.update(((g[0],g[1]),(g[3],g[4])));videos.update((g[0],g[3]))
  for z in paths_group.values():paths.update(z)
  index_samples=sorted(set([0,len(mix)//4,len(mix)//2,3*len(mix)//4,len(mix)-1]))
  samples=[]
  for i in index_samples:
   m,t,n=mix[i],target[i],near[i]
   try:
    a,ar=audio(m[0]);b,br=audio(t[0]);npath=n[0][0] if len(n) else None
    rec={'index':i,'mix':m[0],'target':t[0],'mouth':t[1],'near':npath,'mix_shape':list(a.shape),'target_shape':list(b.shape),'mix_rate':ar,'target_rate':br,'mix_finite':bool(np.isfinite(a).all()),'target_finite':bool(np.isfinite(b).all())}
    if npath and npath!=m[0]:
     d,dr=audio(npath);rec['near_shape']=list(d.shape);rec['near_rate']=dr
     if a.shape==b.shape==d.shape and ar==br==dr:rec['mix_minus_target_near_max_abs']=float(np.max(np.abs(a-b-d)))
    if i==0:
     v=torch.load(t[1],map_location='cpu',weights_only=True);rec['mouth_dtype']=str(v.dtype);rec['mouth_shape']=list(v.shape)
    samples.append(rec)
   except Exception as e:samples.append({'index':i,'error':f'{type(e).__name__}: {e}'})
  key=f'{name}/{split}'
  groups[key]={'root':str(p),'index_sha256':{f.name:sha(f) for f in inputs},'rows':len(mix),'unique_paths':{k:len(v) for k,v in paths_group.items()},'samples_by_length':dict(sorted(lengths.items())),'hours_assuming_16khz':sum(k*v for k,v in lengths.items())/16000/3600,'invalid_rows':dict(invalid),'parsed_lrs2_mixtures':parsed,'unique_filename_first_id':len(videos),'unique_filename_id_plus_segment':len(identities),'self_near':self_near,'decoded_samples':samples}
  sets[key]=paths_group;meta[key]={'utterances':identities,'videos':videos}
  print('indexed',key,len(mix),'unique paths accumulated',len(paths),flush=True)
print('checking',len(paths),'unique paths',flush=True)
exists={};inodes={};failures=collections.Counter()
for i,path in enumerate(paths,1):
 try:
  st=os.stat(path);exists[path]=True;inodes[path]=(st.st_dev,st.st_ino)
 except OSError as e:exists[path]=False;failures[type(e).__name__]+=1
 if i%25000==0:print('stat progress',i,'/',len(paths),flush=True)
for key,group in groups.items():
 group['path_checks']={}
 for kind,values in sets[key].items():
  missing=[p for p in values if not exists[p]]
  group['path_checks'][kind]={'distinct_paths_checked':len(values),'missing_count':len(missing),'missing_examples':sorted(missing)[:10],'unique_device_inode':len({inodes[x] for x in values if x in inodes})}
cross={}
for root in ('lrs2_single','lrs2_paper'):
 for a,b in (('tr','cv'),('tr','tt'),('cv','tt')):
  k1,k2=f'{root}/{a}',f'{root}/{b}'
  if k1 in meta and k2 in meta:
   cross[f'{root}:{a}-{b}']={'filename_id_plus_segment':len(meta[k1]['utterances']&meta[k2]['utterances']), 'filename_first_id':len(meta[k1]['videos']&meta[k2]['videos'])}
result={'started_unix':start,'ended_unix':time.time(),'scope':'All rows of listed index JSON; stat all unique referenced paths; decode 5 spread samples per group, load first mouth only. Filename numeric prefixes and prefix+clip-number keys do not establish person or original video identity.','groups':groups,'cross_split_filename_ids':cross,'all_unique_paths':len(paths),'stat_errors':dict(failures)}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('WROTE',out,flush=True)
