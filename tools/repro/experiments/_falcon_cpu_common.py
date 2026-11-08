"""Falcon(small) CPU loader — 随仓库携带的 W2-12 共享加载器副本。

来源：2026-11-08 深挖轮的 `/tmp/w212/w212_common.py`（58 行，sha256 71e7a1b1d8…），
供 `streaming_canonical.py` 使用。与原文件的唯一差别：仓库根目录不再硬编码，改为按
本文件位置推导（`tools/repro/experiments/` → parents[3]），checkpoint / config 可用环境
变量覆盖。装载逻辑（strict=True、fp32、eval、no_grad）逐行未改。
"""
import os, sys, copy, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch, yaml
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "studio"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4") or 4))

from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive
from falcon_model.videomodels.frcnn_videomodel import FRCNNVideoModel

CKPT = os.environ.get("FALCON_CKPT", "/root/.cache/falcon_weights/falcon.pt")
CFG  = os.environ.get("FALCON_CFG", os.path.join(REPO, "studio/configs/Falcon.yml"))

def build(with_video=True, ckpt=CKPT, cfg=CFG, verbose=True):
    config = yaml.safe_load(open(cfg))
    if "train_conf" in config: config = config["train_conf"]
    acfg = copy.deepcopy(config["audionet"]["audionet_config"])
    vcfg = copy.deepcopy(config["videonet"]["videonet_config"])
    vcfg["pretrain"] = None
    audio = FalconSpeakerContrastive(sample_rate=16000, **acfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    st = ck.get("state_dict", ck)
    astate = {k[len("audio_model."):]: v for k, v in st.items() if k.startswith("audio_model.")}
    audio.load_state_dict(astate, strict=True)
    audio.eval(); audio.requires_grad_(False)
    video = None
    if with_video:
        video = FRCNNVideoModel(**vcfg)
        vstate = {k[len("video_model."):]: v for k, v in st.items() if k.startswith("video_model.")}
        video.load_state_dict(vstate, strict=True)
        video.eval(); video.requires_grad_(False)
    if verbose:
        n = sum(p.numel() for p in audio.parameters())
        print(f"[build] audio params={n:,} video={'on' if video else 'off'} ckpt={os.path.basename(ckpt)}")
    return audio, video, config

def mouth_frames(n=50, seed=0):
    g = np.random.RandomState(seed)
    return (g.rand(n, 96, 96) * 255).astype(np.uint8)

def preprocess_mouth(frames):
    x = frames[:, 4:92, 4:92].astype(np.float32) / 255.0
    x = (x - 0.421) / 0.165
    return torch.from_numpy(np.ascontiguousarray(x))[None, None]

def video_embed(video, frames=None, seed=0):
    f = mouth_frames(50, seed) if frames is None else frames
    with torch.no_grad():
        e = video(preprocess_mouth(f))
    return e

def run(audio, mouth_emb, wav):
    """wav: np (32000,) float32 -> out (32000,)"""
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None]
        y = audio(x, mouth_emb)
    return y[0, 0].numpy()
