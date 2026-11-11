"""Read only the first matching test pair from the first audio/video tar pieces."""
import argparse,hashlib,json,subprocess,tarfile,tempfile
import numpy as np
from scipy.signal import correlate,correlation_lags
from pathlib import Path
root=Path('/aispeech-cfs/voiceprint/data/multimodal/voxceleb2_mix/raw')
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
items=[('audio_clean_part_aa','./audio_clean/test/id00017/01dfn2spqyE/00001.wav'),('orig_part_aa','orig/test/id00017/01dfn2spqyE/00001.mp4')]
result={'date':'2026-11-11','scope':'First matching test pair only, read from first tar parts; no full-volume traversal or mouth/audio sync validation.','samples':[]}
with tempfile.TemporaryDirectory(prefix='falcon-vox-probe-',dir='/root/tmp') as td:
 for archive,name in items:
  with tarfile.open(root/archive,'r|') as t:
   for m in t:
    if m.name==name:
     if m.size>10_000_000:raise ValueError('sample too large')
     data=t.extractfile(m).read();break
   else:raise ValueError(f'missing sample {name}')
  path=Path(td)/('sample'+Path(name).suffix);path.write_bytes(data)
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
  streams=[{k:s.get(k) for k in ('codec_type','codec_name','sample_rate','channels','width','height','r_frame_rate','duration','nb_frames') if k in s} for s in info['streams']]
  typ='video' if name.endswith('.mp4') else 'audio'
  cmd=['ffmpeg','-v','error','-i',str(path),'-map',f'0:{"v" if typ=="video" else "a"}:0']
  cmd+=['-frames:v','1','-f','null','-'] if typ=='video' else ['-t','1','-f','null','-']
  dec=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=30)
  result['samples'].append({'archive':str(root/archive),'member':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'streams':streams,'container_duration':info.get('format',{}).get('duration'),'decode_exit':dec.returncode})
 # Decode both audio streams at 16 kHz and search for an audio/audio lag within 200 ms.
 arrays=[]
 for suffix in ('.wav','.mp4'):
  audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(Path(td)/('sample'+suffix)),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1'])
  arrays.append(np.frombuffer(audio,dtype='<f4').astype('float64'))
 x,y=arrays; corr=correlate(x,y,mode='full',method='fft'); lags=correlation_lags(len(x),len(y),mode='full')
 selection=np.abs(lags)<=3200; lag=int(lags[selection][np.argmax(np.abs(corr[selection]))])
 i=max(0,lag);j=max(0,-lag);n=min(len(x)-i,len(y)-j)
 result['audio_audio_probe']={'sample_rate':16000,'lengths':[len(x),len(y)],'searched_abs_lag_samples':3200,'best_lag_samples':lag,'overlap_pearson':float(np.corrcoef(x[i:i+n],y[j:j+n])[0,1]),'overlap_samples':n}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('saved',out)
