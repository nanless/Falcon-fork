import json,sys,torch,soundfile as sf,numpy as np
from pathlib import Path
repo=Path("/root/code/github_repos/Falcon-fork");sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/"studio"))
from falcon_runtime import FalconRuntime,preprocess_mouth
from tools.repro.experiments.streaming_consistency import frames_block
D=Path("/root/data/falcon_lrs2_2mix/tt");M=json.load(open(D/"mix.json"));T=json.load(open(D/"target.json"));i=0
a,sr=sf.read(M[i][0],dtype="float32");r,_=sf.read(T[i][0],dtype="float32");frames=torch.load(T[i][1],map_location="cpu",weights_only=True).numpy()[:,:,:,0]
old=frames_block(frames,0,50);correct=frames[np.arange(50)]
rt=FalconRuntime("small","/root/.cache/falcon_weights/falcon.pt").load("cpu");yold=rt.infer(a,old);ycorrect=rt.infer(a,correct)
def score(x,y):
 x=x.astype(np.float64);y=y.astype(np.float64);x-=x.mean();y-=y.mean();p=(x@y)/(y@y+1e-12)*y;return float(10*np.log10((p@p+1e-12)/((x-p)@(x-p)+1e-12)))
d=yold-ycorrect
print(json.dumps({"index":i,"mix":M[i][0],"mouth":T[i][1],"ckpt_sha256":rt.checkpoint_sha256,"device":"cpu","threads":torch.get_num_threads(),"indices_old_first_10":frames_block(np.arange(50),0,10).tolist(),"indices_correct_first_10":list(range(10)),"delta_rms":float(np.sqrt(np.mean(d*d))),"delta_max":float(np.abs(d).max()),"sisdr_old":score(yold,r),"sisdr_correct":score(ycorrect,r)},indent=2))
