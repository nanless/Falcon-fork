"""Inspect label tar first members and first sample headers of AVCocktail video shards."""
import argparse, collections, json, re, tarfile
from pathlib import Path
root=Path('/aispeech-cfs/voiceprint/data/multimodal/avcocktail/raw')
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
labels=sorted((root/'labels').glob('video_*/*.tar'));videos=sorted((root/'videos').glob('video_*/*.tar'))
label_rows=[];video_rows=[]
for path in labels:
 with tarfile.open(path,'r:') as t:
  member=t.next()
  if member is None:raise ValueError(f'empty label tar: {path}')
  data=t.extractfile(member).read()
  text=data.decode('utf-8')
  label_rows.append({'archive':str(path),'member':member.name,'bytes':len(data),'webvtt':text.startswith('WEBVTT'),'cue_count':len(re.findall(r'--> ',text))})
for path in videos:
 with tarfile.open(path,'r:') as t:
  entries=[];key=None
  for member in t:
   suffix=member.name.rsplit('.',1)[-1]; k=member.name.rsplit('.',1)[0]
   if key is None:key=k
   if k!=key:break
   item={'member':member.name,'bytes':member.size}
   if suffix in ('sample_id','start_time','end_time') and member.size<256:
    item['value']=t.extractfile(member).read().decode('utf8')
   entries.append(item)
   if suffix=='video':break
  video_rows.append({'archive':str(path),'sample_key':key,'first_sample_entries':entries})
res={'date':'2026-11-11','scope':'All label tar first members and first sample headers of each video tar; no all-sample traversal or media decoding.','label_tar_count':len(labels),'video_tar_count':len(videos),'label_vtt_count':sum(r['webvtt'] for r in label_rows),'video_tar_kinds':dict(collections.Counter(x.name.split('-chunk-')[0] for x in videos)),'label_cues_total':sum(r['cue_count'] for r in label_rows),'label_cue_count_min_max':[min(r['cue_count'] for r in label_rows),max(r['cue_count'] for r in label_rows)],'labels':label_rows,'video_first_samples':video_rows}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(res,f,ensure_ascii=False,indent=2);f.write('\n')
print('saved',out,'labels',len(labels),'videos',len(videos),'all webvtt',res['label_vtt_count'])
