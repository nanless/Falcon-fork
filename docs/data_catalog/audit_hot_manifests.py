"""Read complete existing TSV indexes and inspect bounded media samples."""
import argparse, collections, hashlib, json, os, subprocess
from datetime import datetime,timezone,timedelta
from pathlib import Path
BASE=Path('/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/derived/avsr_reproduction/manifests')
SETS=['mms_llama_433h_nocap_srlength_v6_20261004','mms_llama_1759h_v6_datarepair_20261004']

def probe(path):
 p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels','-of','json',path],capture_output=True,text=True,timeout=30)
 return {'exit':p.returncode,'media':json.loads(p.stdout) if p.returncode==0 else None,'error':p.stderr[:500] if p.returncode else None}

def audit(directory,split):
 p=BASE/directory/(split+'.tsv'); h=hashlib.sha256(); groups=collections.Counter(); subgroups=collections.Counter(); duration_gap=collections.Counter(); first={}; rows=0; bad=0; samples=0; frame_count=0; checks=[]
 with p.open('rb') as f:
  head=next(f);h.update(head)
  for i,raw in enumerate(f):
   h.update(raw); a=raw.decode('utf-8').rstrip('\r\n').split('\t');rows+=1
   if len(a)!=6:bad+=1;continue
   source=a[0].split('/')[0];groups[source]+=1
   if source=='vox2':
    part='dev' if 'preprocessed_dev_seg16' in a[1] else 'test' if 'preprocessed_test_seg16' in a[1] else 'other'
    subgroups[part]+=1
   try:
    frames=int(a[3]); audio_samples=int(a[4]); frame_count+=frames;samples+=audio_samples
    delta=abs(audio_samples/16000-frames/25)
    for threshold in (0.04,0.08,0.1,1.0):
     if delta>threshold+1e-9:duration_gap[str(threshold)]+=1
   except ValueError:bad+=1
   if source not in first:first[source]={'row':i,'source':a[0],'video':a[1],'audio':a[2],'frames':a[3],'samples':a[4],'column6':a[5]}
   interval=20000 if split=='train' else 100
   if i%interval==0:checks.append({'row':i,'source':source,'video':a[1],'audio':a[2],'video_exists':os.path.isfile(a[1]),'audio_exists':os.path.isfile(a[2])})
 for entry in first.values():
  for k in ('video','audio'):entry[k+'_ffprobe']=probe(entry[k])
  q=subprocess.run(['ffmpeg','-v','error','-i',entry['video'],'-frames:v','1','-f','null','-'],capture_output=True,text=True,timeout=45)
  entry['video_first_frame_decode']={'exit':q.returncode,'error':q.stderr[:500] if q.returncode else None}
 wrd=p.with_suffix('.wrd')
 with wrd.open('rb') as f: text_rows=sum(1 for _ in f)
 return {'path':str(p),'sha256':h.hexdigest(),'header':head.decode('utf-8').strip(),'rows':rows,'field_or_numeric_errors':bad,'source_groups':dict(groups),'vox2_source_partition_from_path':dict(subgroups),'absolute_audio_video_duration_difference_over_seconds':dict(duration_gap),'audio_samples_sum':samples,'audio_hours_at_16k':samples/16000/3600,'video_frame_count_sum':frame_count,'text_file':str(wrd),'text_rows':text_rows,'bounded_path_checks':checks,'first_per_group':first}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result={'timestamp':datetime.now(timezone(timedelta(hours=8))).isoformat(),'scope':'Complete TSV text/index rows and aligned WRD row counts; sample file existence every 20000 train or 100 test rows; first item per source group ffprobe and video first-frame decode. No media-wide validation.','sets':{name:{sp:audit(name,sp) for sp in ('train','test')} for name in SETS}}
 with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 for name, splits in result['sets'].items():
  for split,d in splits.items():print(name,split,'rows',d['rows'],'groups',d['source_groups'],'sampled',len(d['bounded_path_checks']),'missing',sum(not (x['video_exists'] and x['audio_exists']) for x in d['bounded_path_checks']),'hours',round(d['audio_hours_at_16k'],3))
if __name__=='__main__':main()
