"""Probe selected archived audiovisual pairs; extracts only small members to temp."""
import argparse, hashlib, io, json, re, subprocess, tarfile, tempfile, zipfile
from pathlib import Path
import numpy as np
ROOT=Path('/aispeech-cfs/voiceprint/data/multimodal')
ITEMS=[
 ('aspire_audio','aspire/raw/ASPIRE.zip','ASPIRE/Clean/S1/Audio/129_001_lbiv2s.wav'),
 ('aspire_video','aspire/raw/ASPIRE.zip','ASPIRE/Clean/S1/Video/129_001_lbiv2s.mov'),
 ('chinese_test_audio','chinese_lips/raw/processed_test.zip','processed_test/130_42_M_TY_064.wav'),
 ('chinese_test_video','chinese_lips/raw/processed_test.zip','processed_test/130_42_M_TY_064.mp4'),
 ('mld_60_audio','mld_vc/raw/audio.zip','audio/Online/speaker_21/TXHY/9/60db.wav'),
 ('mld_60_video','mld_vc/raw/video.zip','video/Online/speaker_21/TXHY/9/60db.mp4'),
 ('mld_clean_audio','mld_vc/raw/audio.zip','audio/Online/speaker_21/TXHY/9/clean.wav'),
 ('mld_clean_video','mld_vc/raw/video.zip','video/Online/speaker_21/TXHY/9/clean.mp4'),
]

def probe(data,suffix,tmp):
 if len(data)>10_000_000:raise ValueError('sample exceeds bounded size')
 path=tmp/('sample'+suffix);path.write_bytes(data)
 p=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,timeout=30,check=True)
 d=json.loads(p.stdout); streams=[]
 for x in d['streams']:
  streams.append({k:x.get(k) for k in ('codec_type','codec_name','sample_rate','channels','width','height','r_frame_rate','duration','nb_frames') if k in x})
 out={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'container_duration':d.get('format',{}).get('duration'),'streams':streams}
 if any(s.get('codec_type')=='video' for s in streams):
  p=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-frames:v','1','-f','null','-'],capture_output=True,timeout=30)
  out['video_first_frame_decode_exit']=p.returncode
 if any(s.get('codec_type')=='audio' for s in streams):
  p=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-t','1','-ac','1','-ar','16000','-f','f32le','pipe:1'],capture_output=True,timeout=30)
  samples=np.frombuffer(p.stdout,dtype='<f4')
  out['audio_one_second']={'decode_exit':p.returncode,'sample_count':int(samples.size),'rms':float(np.sqrt(np.mean(samples.astype('float64')**2))) if samples.size else None}
 return out

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
result={'date':'2026-11-11','scope':'Selected ZIP/tar members, no full archive extraction or sync proof; decoded first video frame and first second audio when present','samples':[]}
with tempfile.TemporaryDirectory(prefix='falcon-av-probe-',dir='/root/tmp') as dirname:
 tmp=Path(dirname)
 for name,archive,member in ITEMS:
  with zipfile.ZipFile(ROOT/archive) as z:
   data=z.read(member)
  result['samples'].append({'name':name,'archive':str(ROOT/archive),'member':member,**probe(data,Path(member).suffix,tmp)})
 base=ROOT/'avcocktail/raw';vp=base/'videos/video_0/asd-chunk-eval-0.tar';lp=base/'labels/video_0/label-eval-0.tar'
 with tarfile.open(vp,'r:') as tar:
  members={}
  for member in tar:
   if not member.name.startswith('0000000000.'):break
   members[member.name.split('.')[-1]]=tar.extractfile(member).read()
  av={'archive':str(vp),'sample_key':'0000000000', 'metadata':{k:v.decode('utf8') for k,v in members.items() if k!='video'},'video':probe(members['video'],'.mp4',tmp)}
 with tarfile.open(lp,'r:') as tar:
  label=tar.extractfile('0000000000.label').read().decode('utf8')
 av['label']={'archive':str(lp),'member':'0000000000.label','format':'WebVTT' if label.startswith('WEBVTT') else 'other','cue_count':len(re.findall(r'--> ',label)),'byte_count':len(label.encode('utf8'))}
 result['avcocktail']=av
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('saved',out,'samples',len(result['samples']),'avcocktail',result['avcocktail']['label']['format'])
