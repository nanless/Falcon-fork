"""Paced loopback WebSocket load probe, with synthetic audio and mouth images.

Server must already be running on the same machine. Measures each chunk from
start of client payload encoding through full result receipt and JSON parsing.
It does not measure camera/microphone capture, real tracking, network to another
machine, audio playback, or extraction quality. Random checkpoints are allowed.
"""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
import numpy as np
import websockets
from tools.streaming.common import SAMPLES, FRAMES, SR, SAMPLES_PER_FRAME


def stats(v):
    return {"p50":float(np.percentile(v,50)), "p90":float(np.percentile(v,90)),
            "p95":float(np.percentile(v,95)), "p99":float(np.percentile(v,99)),
            "min":float(np.min(v)), "max":float(np.max(v))}

async def run(a):
    rng = np.random.default_rng(a.seed)
    audio = (rng.standard_normal(SAMPLES) * .1).astype("<f4")
    mouth = rng.integers(0,256,(FRAMES,96,96), dtype=np.uint8)
    measurements = []
    async with websockets.connect(a.url, max_size=100000) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready": raise RuntimeError(str(ready))
        # Absolute deadlines prevent a slow iteration from silently lowering the rate.
        deadline = time.perf_counter()
        for i in range(a.warmup + a.chunks):
            deadline += SAMPLES / SR
            await asyncio.sleep(max(0,deadline - time.perf_counter()))
            start = time.perf_counter()
            start_sample = i*SAMPLES
            request = {"type":"chunk", "sequence":i, "start_sample":start_sample,
                       "video_samples":[start_sample+j*SAMPLES_PER_FRAME for j in range(FRAMES)],
                       "audio_f32le":base64.b64encode(audio.tobytes()).decode("ascii"),
                       "mouth_gray_u8":base64.b64encode(mouth.tobytes()).decode("ascii")}
            encoded = json.dumps(request,separators=(",", ":"))
            await ws.send(encoded)
            result = json.loads(await ws.recv())
            elapsed = (time.perf_counter()-start)*1000
            if result.get("type") != "result" or result.get("sequence") != i:
                raise RuntimeError(f"unexpected response: {result}")
            if len(base64.b64decode(result["audio_f32le"],validate=True)) != SAMPLES*4:
                raise RuntimeError("wrong output length")
            measurements.append({"sequence":i,"client_elapsed_ms":elapsed,
                                 "schedule_late_ms":max(0,(start-deadline)*1000),
                                 "server":result["timing_ms"]})
    scored = measurements[a.warmup:]
    metric_names = sorted(scored[0]["server"])
    return {"url":a.url, "seed":a.seed, "chunk_ms": SAMPLES * 1000 / SR, "warmup_chunks":a.warmup,
            "measured_chunks":a.chunks, "training_status":ready.get("training_status"),
            "client_encode_send_receive_parse_ms":stats([m["client_elapsed_ms"] for m in scored]),
            "client_schedule_lateness_ms":stats([m["schedule_late_ms"] for m in scored]),
            "server_ms":{name:stats([m["server"][name] for m in scored]) for name in metric_names},
            "over_chunk_ms_count":sum(m["client_elapsed_ms"] > SAMPLES * 1000 / SR for m in scored),
            "scope":"synthetic audio and mouth frames; same-machine WebSocket; excludes actual capture, face tracking, real network, speaker playback and separation quality",
            "samples":scored}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="ws://127.0.0.1:8765/ws")
    p.add_argument("--warmup",type=int,default=4)
    p.add_argument("--chunks",type=int,default=30)
    p.add_argument("--seed",type=int,default=20261110)
    p.add_argument("--output")
    args=p.parse_args()
    if args.warmup<0 or args.chunks<1: p.error("invalid chunk counts")
    result=asyncio.run(run(args))
    rendered=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        with Path(args.output).open("x") as f: f.write(rendered+"\n")
    summary={k:v for k,v in result.items() if k!="samples"}
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
