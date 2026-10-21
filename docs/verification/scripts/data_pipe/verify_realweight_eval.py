"""R3: real-weight verification of the evaluation data path (falcon.pt, local, no network).

Checks:
 1) falcon.pt top-level format (supports PROVENANCE.md claim: state_dict + architecture + epoch);
 2) build models from configs/Falcon.yml and load ckpt with strict=True;
 3) replicate audio_visual_test.py:229-231 end-to-end on AVSpeechEvalDataset with stereo,
    odd-length wavs and synthetic uint8 mouth .pt: per-sample output length == T; audio-frame vs
    video-frame alignment via NeuroSCNetFusion nearest interpolation.
Run:
  PYTHONPATH=/tmp/dp-verify/stubs:$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_realweight_eval.py
"""
import copy
import json
import os
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

REPO = "/root/code/github_repos/Falcon-fork"
CKPT = "/root/.cache/falcon_weights/falcon.pt"
ROOT = "/tmp/dp-verify/realweight"
sys.path.insert(0, REPO)
_DP_STUB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "docs", "verification", "stubs", "data_pipe"))
sys.path.insert(0, _DP_STUB)
sys.path.insert(0, "/tmp/dp-verify/stubs")  # optional fallback

import look2hear.models as models  # noqa: E402
import look2hear.videomodels as vmodels  # noqa: E402
from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset  # noqa: E402
from look2hear.models.Falcon import NeuroSCNetFusion  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "Falcon.yml")))

print("=" * 78)
print("[1] checkpoint format")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
print("  torch.load type :", type(ckpt).__name__)
if isinstance(ckpt, dict):
    print("  top-level keys  :", list(ckpt.keys()))
    print("  architecture    :", ckpt.get("architecture"))
    print("  epoch           :", ckpt.get("epoch"))
    sd = ckpt["state_dict"]
    audio_state = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    video_state = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
    print("  state_dict size :", len(sd), "| audio_model.*:", len(audio_state), "| video_model.*:", len(video_state))
    other = [k for k in sd if not (k.startswith("audio_model.") or k.startswith("video_model."))]
    print("  other keys      :", other[:5])
print("  => audio_visual_test.py:119-120 的 ckpt['state_dict'] 兼容:", "state_dict" in (ckpt if isinstance(ckpt, dict) else {}))

print("=" * 78)
print("[2] build models from configs/Falcon.yml + strict load")
audio_cfg = copy.deepcopy(cfg["audionet"]["audionet_config"])  # deepcopy: Separator pops keys
model = getattr(models, cfg["audionet"]["audionet_name"])(
    sample_rate=cfg["datamodule"]["data_config"]["sample_rate"], **audio_cfg)
video_cfg = copy.deepcopy(cfg["videonet"]["videonet_config"])
video_cfg.pop("pretrain", None)
video = vmodels.FRCNNVideoModel(**video_cfg, pretrain=None)
missing, unexpected = model.load_state_dict(audio_state, strict=True)
print("  audio load strict=True: OK")
video.load_state_dict(video_state, strict=True)
print("  video load strict=True: OK")
model.eval()
video.eval()
print("  audio params:", sum(p.numel() for p in model.parameters()),
      "| video params:", sum(p.numel() for p in video.parameters()))

# ---------------------------------------------------------------- fixture
shutil.rmtree(ROOT, ignore_errors=True)
for sub in ["mix", "target", "mouths"]:
    os.makedirs(os.path.join(ROOT, sub))
rs = np.random.RandomState(0)
specs = [("seg2s", 32000, 50), ("odd", 48513, 75)]
mix_json, tgt_json = [], []
for i, (stem, T, F) in enumerate(specs):
    x = (rs.randn(T) * 0.3).astype(np.float32)
    y = (rs.randn(T) * 0.1).astype(np.float32)
    mixp = os.path.join(ROOT, "mix", stem + ".wav")
    tgtp = os.path.join(ROOT, "target", stem + ".wav")
    mop = os.path.join(ROOT, "mouths", stem + ".pt")
    sf.write(mixp, np.stack([x, x], axis=1), 16000)          # 2-channel (evade mono P0)
    sf.write(tgtp, np.stack([y, y], axis=1), 16000)
    frames = (rs.rand(F, 96, 96, 1) * 255).astype(np.uint8)  # uint8 storage, like inference format
    torch.save(torch.from_numpy(frames), mop)
    n = sf.info(mixp).frames
    mix_json.append([mixp, n])
    tgt_json.append([tgtp, mop, n])
json.dump(mix_json, open(os.path.join(ROOT, "mix.json"), "w"))
json.dump(tgt_json, open(os.path.join(ROOT, "target.json"), "w"))
ds = AVSpeechEvalDataset(json_dir=ROOT, sample_rate=16000, normalize_audio=False)

print("=" * 78)
print("[3] audio_visual_test.py:229-231 replication with real weights")
seen = []
orig_fwd = NeuroSCNetFusion.forward


def probe(self, audio, video_feat):
    seen.append((tuple(audio.shape), tuple(video_feat.shape)))
    return orig_fwd(self, audio, video_feat)


NeuroSCNetFusion.forward = probe
try:
    for idx, (stem, T, F) in enumerate(specs):
        mix, src, mouth, key, n_near = ds[idx]
        print(f"\n  sample[{idx}] {stem}: T={T}, mouth_frames={F}")
        print(f"    dataset item : mix{tuple(mix.shape)} {mix.dtype}; mouth{tuple(mouth.shape)} {mouth.dtype}; key={key}; n_near={n_near}")
        t0 = time.time()
        seen.clear()
        with torch.no_grad():
            mouth_in = torch.from_numpy(mouth[None, None]).float()
            mouth_emb = video(mouth_in)
            est = model(mix[None], mouth_emb)
        dt = time.time() - t0
        print(f"    mouth_in     : {tuple(mouth_in.shape)} -> mouth_emb {tuple(mouth_emb.shape)}")
        print(f"    fusion calls : audio{seen[0][0]} <- video{seen[0][1]}" if seen else "    no fusion captured")
        print(f"    est_sources  : {tuple(est.shape)} {est.dtype} | len==T: {est.shape[-1] == T}"
              f" | finite: {bool(torch.isfinite(est).all())} | CPU {dt:.2f}s")
finally:
    NeuroSCNetFusion.forward = orig_fwd
print("\n== DONE")
