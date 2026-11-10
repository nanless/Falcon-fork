import sys,json,torch,soundfile as sf,numpy as np,hashlib
from pathlib import Path
repo=Path("/root/code/github_repos/Falcon-fork");sys.path.insert(0,str(repo/"studio"))
from falcon_runtime import FalconRuntime,preprocess_mouth
D=Path("/root/data/falcon_lrs2_2mix/tt");M=json.loads((D/"mix.json").read_text());T=json.loads((D/"target.json").read_text())
idxs=[0,1,2,3];aud=[];refs=[];v=[]
for i in idxs:
 a,sr=sf.read(M[i][0],dtype="float32"); r,_=sf.read(T[i][0],dtype="float32");frames=torch.load(T[i][1],map_location="cpu",weights_only=True).numpy()[:,:,:,0];aud.append(a);refs.append(r);v.append(preprocess_mouth(frames))
runtime=FalconRuntime("small","/root/.cache/falcon_weights/falcon.pt").load("cpu")
x=torch.tensor(np.stack(aud));mouth=torch.cat(v)
with torch.inference_mode():
 emb=runtime.video_model(mouth); pred4=runtime.audio_model(x,emb)[:,0].numpy();pred1=np.stack([runtime.audio_model(x[i:i+1],emb[i:i+1])[0,0].numpy() for i in range(4)])
def score(x,y):
 x=x.astype(np.float64);y=y.astype(np.float64);x-=x.mean();y-=y.mean();p=(x@y)/(y@y+1e-12)*y;return float(10*np.log10((p@p+1e-12)/((x-p)@(x-p)+1e-12)))
res=[]
for i in range(4):
 d=pred4[i]-pred1[i];res.append({"index":idxs[i],"mix":M[idxs[i]][0],"mouth":T[idxs[i]][1],"rms":float(np.sqrt(np.mean(pred1[i]**2))),"delta_rms":float(np.sqrt(np.mean(d**2))),"delta_max":float(np.max(np.abs(d))),"sisdr_batch1":score(pred1[i],refs[i]),"sisdr_batch4":score(pred4[i],refs[i])})
print(json.dumps({"checkpoint_sha256":runtime.checkpoint_sha256,"torch_threads":torch.get_num_threads(),"device":"cpu","description":"whole 2 s: same visual embeddings and inputs; only audio model batch differs; first four LRS2 tt items","samples":res},indent=2))
