"""Offline replay through the exact stateful 160 ms live-model API."""
import argparse
import math
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
if __package__:
    from .common import SR, SAMPLES, FRAMES, SAMPLES_PER_FRAME, load_mono, mouth_tensor, read_checkpoint
else:
    from common import SR, SAMPLES, FRAMES, SAMPLES_PER_FRAME, load_mono, mouth_tensor, read_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--audio", required=True, help="16kHz mono WAV")
p.add_argument("--mouth", required=True, help="uint8 [frames,96,96,3] or [frames,96,96] tensor, 25fps aligned with WAV")
p.add_argument("--output", required=True, help="new WAV path; input-length output with final padding trimmed")
p.add_argument("--device", default="cpu")
a = p.parse_args()
if Path(a.output).exists(): raise FileExistsError(a.output)
audio = load_mono(a.audio)
if not len(audio): raise ValueError("empty audio")
raw = torch.load(a.mouth, map_location="cpu", weights_only=True)
required_frames = math.ceil(len(audio) / SAMPLES_PER_FRAME)
if len(raw) < required_frames:
    raise ValueError(f"need at least {required_frames} time-aligned mouth frames, got {len(raw)}")
video = mouth_tensor(raw[:required_frames])
chunks = math.ceil(len(audio) / SAMPLES)
end_padding = chunks * SAMPLES - len(audio)
audio = torch.nn.functional.pad(audio, (0, end_padding))
video_pad = chunks * FRAMES - len(video)
if video_pad:
    video = torch.cat((video, video[-1:].expand(video_pad, -1, -1)))
ckpt = read_checkpoint(a.checkpoint, a.device)
model = FalconStreaming().to(a.device)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
outputs = []
state = None
elapsed = []
with torch.inference_mode():
    for i in range(chunks):
        x = audio[i*SAMPLES:(i+1)*SAMPLES].unsqueeze(0).to(a.device)
        v = video[i*FRAMES:(i+1)*FRAMES].unsqueeze(0).to(a.device)
        if str(a.device).startswith("cuda"): torch.cuda.synchronize()
        start = time.perf_counter()
        y, state = model.forward_chunk(x, v, state)
        if str(a.device).startswith("cuda"): torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        outputs.append(y.squeeze(0).cpu())
out = torch.cat(outputs)[:len(audio)-end_padding].numpy()
if not np.isfinite(out).all(): raise RuntimeError("non-finite model output")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
with open(a.output, "xb") as stream:
    sf.write(stream, out, SR, format="WAV", subtype="FLOAT")
print(f"saved {a.output}: samples={len(out)} chunks={chunks} padded_samples={end_padding} "
      f"training_status={ckpt.get('training_status', 'legacy_checkpoint')} forward_p50_ms={np.median(elapsed):.3f} "
      f"forward_p90_ms={np.percentile(elapsed,90):.3f}; first chunk includes warmup, "
      "timing excludes input transfer, decoding, tracking, buffering and playback")
