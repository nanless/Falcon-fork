import sys,json,torch,soundfile as sf,numpy as np
from pathlib import Path
repo=Path("/root/code/github_repos/Falcon-fork");sys.path.insert(0,str(repo/"studio"))
from falcon_runtime import FalconRuntime,preprocess_mouth
D=Path("/root/data/falcon_lrs2_2mix/tt");M=json.loads((D/"mix.json").read_text());T=json.loads((D/"target.json").read_text())
indices=[7,15,43];aud=[];frames=[];refs=[]
for i in indices:
 a,sr=sf.read(M[i][0],dtype="float32");r,_=sf.read(T[i][0],dtype="float32");im=torch.load(T[i][1],map_location="cpu",weights_only=True).numpy()[:,:,:,0];aud.append(a);refs.append(r);frames.append(preprocess_mouth(im))
runtime=FalconRuntime("small","/root/.cache/falcon_weights/falcon.pt").load("cpu");x=torch.tensor(np.stack(aud));m=torch.cat(frames)
with torch.inference_mode():
 emb=runtime.video_model(m); target_x=x[2:3];both_x=torch.cat([x[0:1],target_x]); a=runtime.audio_model(both_x,torch.cat([emb[0:1],emb[2:3]]))[1,0].numpy();b=runtime.audio_model(both_x,torch.cat([emb[1:2],emb[2:3]]))[1,0].numpy()
d=b-a
print(json.dumps({"indices_in_current_tt_index":indices,"before_video_1":T[indices[0]][1],"before_video_2":T[indices[1]][1],"target_audio":M[indices[2]][0],"target_video":T[indices[2]][1],"checkpoint_sha256":runtime.checkpoint_sha256,"device":"cpu","threads":torch.get_num_threads(),"output_rms":float(np.sqrt(np.mean(a*a))),"diff_rms":float(np.sqrt(np.mean(d*d))),"diff_max":float(np.max(np.abs(d))),"diff_rms_pct":float(100*np.sqrt(np.mean(d*d))/np.sqrt(np.mean(a*a)))},indent=2))
