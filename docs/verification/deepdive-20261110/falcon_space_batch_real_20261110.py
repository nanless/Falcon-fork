import sys,json,torch,soundfile as sf,numpy as np,time,hashlib,os
from pathlib import Path
repo=Path("/root/code/github_repos/Falcon-fork")
sys.path.insert(0,str(repo/"studio"));sys.path.insert(0,str(repo/"space"))
from falcon_runtime import FalconRuntime
from clip import separate
idx=0;d=Path("/root/data/falcon_lrs2_2mix/tt")
mix=json.loads((d/"mix.json").read_text());target=json.loads((d/"target.json").read_text())
a,sr=sf.read(mix[idx][0],dtype="float32");ref,sr_ref=sf.read(target[idx][0],dtype="float32")
c=torch.load(target[idx][1],map_location="cpu",weights_only=True).numpy()[:,:,:,0]; times=np.arange(len(c),dtype=np.int64)*640+320
ckpt="/root/.cache/falcon_weights/falcon.pt";runtime=FalconRuntime("small",ckpt).load("cpu")
start=time.monotonic();y1,n1,k1=separate(runtime.audio_model,runtime.video_model,a,c,times,batch=1);y4,n4,k4=separate(runtime.audio_model,runtime.video_model,a,c,times,batch=4)
def score(x,y):
 x=x.astype(np.float64);y=y.astype(np.float64);x-=x.mean();y-=y.mean();p=(x@y)/(y@y+1e-12)*y;return float(10*np.log10((p@p+1e-12)/((x-p)@(x-p)+1e-12)))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
 return h.hexdigest()
delta=y4-y1
res={"sample_index":idx,"mix":mix[idx][0],"mix_sha256":sha(mix[idx][0]),"target":target[idx][0],"target_sha256":sha(target[idx][0]),"mouth":target[idx][1],"mouth_sha256":sha(target[idx][1]),"checkpoint":ckpt,"checkpoint_sha256":runtime.checkpoint_sha256,"sample_rate":sr,"ref_sample_rate":sr_ref,"audio_samples":len(a),"mouth_shape":c.shape,"mouth_dtype":str(c.dtype),"times":"center timestamps 320+640*i samples","device":"cpu","torch_threads":torch.get_num_threads(),"torch":torch.__version__,"window_count":[n1,n4],"clip_count":[k1,k4],"batch1_rms":float(np.sqrt(np.mean(y1*y1))),"batch4_rms":float(np.sqrt(np.mean(y4*y4))),"delta_rms":float(np.sqrt(np.mean(delta*delta))),"delta_max":float(np.max(np.abs(delta))),"delta_pct":float(100*np.sqrt(np.mean(delta*delta))/np.sqrt(np.mean(y1*y1))),"batch1_sisdr":score(y1,ref),"batch4_sisdr":score(y4,ref),"duration_s":time.monotonic()-start}
print(json.dumps(res,indent=2,ensure_ascii=False))
