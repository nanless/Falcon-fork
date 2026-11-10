import argparse, asyncio, base64, json, os, socket, subprocess, time, urllib.request
from pathlib import Path
import numpy as np
import websockets
parser=argparse.ArgumentParser(description="Probe two simultaneous local WebSocket streams using an existing random checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--device", default="cuda")
args=parser.parse_args()
root=str(Path(__file__).resolve().parents[3])
socket0=socket.socket(); socket0.bind(("127.0.0.1",0));port=socket0.getsockname()[1];socket0.close()
env=os.environ.copy();env["PYTHONPATH"]=root;env["OMP_NUM_THREADS"]="2"
cmd=[__import__("sys").executable,"-m","tools.streaming.server","--checkpoint",args.checkpoint,"--device",args.device,"--port",str(port)]
p=subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
try:
 for i in range(120):
  try:
   with urllib.request.urlopen(f"http://127.0.0.1:{port}/status",timeout=.2) as r: status=json.load(r)
   break
  except Exception:
   if p.poll() is not None: raise RuntimeError(p.stderr.read())
   time.sleep(.25)
 else: raise RuntimeError("server startup timed out")
 print("status",status,flush=True)
 def payload(seq,amp=1):
  n=seq*2560;rng=np.random.default_rng(77);v=rng.integers(0,256,(4,96,96),dtype=np.uint8)
  a=(amp*np.sin(np.arange(2560,dtype=np.float32)*.03)).astype("<f4")
  enc=lambda x:base64.b64encode(x.tobytes()).decode("ascii")
  return {"type":"chunk","sequence":seq,"start_sample":n,"video_samples":[n+k*640 for k in range(4)],"audio_f32le":enc(a),"mouth_gray_u8":enc(v)}
 async def run():
  url=f"ws://127.0.0.1:{port}/ws"
  async with websockets.connect(url,max_size=200000) as a,websockets.connect(url,max_size=200000) as b:
   ready=await asyncio.gather(a.recv(),b.recv()); print("ready",[json.loads(v)["type"] for v in ready],flush=True)
   for i in range(8):
    msg=json.dumps(payload(i))
    t=time.monotonic();await asyncio.gather(a.send(msg),b.send(msg)); replies=await asyncio.gather(a.recv(),b.recv()); dt=(time.monotonic()-t)*1000
    x,y=map(json.loads,replies);assert x["type"]=="result" and y["type"]=="result" and x["audio_f32le"]==y["audio_f32le"]
    print("iteration",i,"both_client_ms",round(dt,2),"server_queues_ms",round(x["timing_ms"]["queue"],2),round(y["timing_ms"]["queue"],2),flush=True)
   await a.send(json.dumps({"type":"reset"}));assert json.loads(await a.recv())["type"]=="reset"
   await a.send(json.dumps(payload(0)));r=json.loads(await a.recv());assert r["sequence"]==0
   await b.send(json.dumps(payload(8)));r=json.loads(await b.recv());assert r["sequence"]==8
  print("TWO_SIMULTANEOUS_STREAMS_OK",flush=True)
 asyncio.run(run())
finally:
 p.terminate()
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:p.kill();p.wait()
