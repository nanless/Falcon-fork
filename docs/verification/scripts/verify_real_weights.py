"""Lead 独立验证脚本 3：真实发布权重（falcon.pt / falcon-big.pt）格式、strict 加载、双代码库一致性。

背景：沙箱默认无权重；Lead 已通过 hf-mirror 下载：
    /root/.cache/falcon_weights/falcon.pt      (47,156,343 B, JusperLee/Falcon)
    /root/.cache/falcon_weights/falcon-big.pt  (49,645,087 B, JusperLee/Falcon-Big)
本脚本回答：
  A. 发布文件顶层结构（是否 {"state_dict": ...}）、key 前缀、数量、是否只有 audio_model.*/video_model.*
  B. 训练侧 look2hear 模型 strict 加载是否 0 missing/unexpected（audio 525 / video 137 之说）
  C. Studio 侧 falcon_model 模型 strict 加载，且与训练侧模型 **同权重同输入 CPU 前向是否 bit-identical**
  D. FalconRuntime(variant=small, checkpoint_path=本地文件) 真实加载 + infer 形状，参数数
  E. Falcon-Big（SpeakerContrastiveV2）strict 加载与参数量
  F. audio_visual_test 的 ckpt["state_dict"] 假设是否成立

运行（在仓库根）：
    PYTHONPATH=<repo>/docs/verification/stubs:<repo> \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_real_weights.py
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
WEIGHTS = Path("/root/.cache/falcon_weights")
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "studio"))

torch.set_num_threads(4)
SECTION = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)  # noqa: E731


def inspect_ckpt(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"file={path.name} size={path.stat().st_size:,}B type={type(ck).__name__}")
    if isinstance(ck, dict):
        print("top-level keys:", list(ck.keys()))
        state = ck.get("state_dict", ck)
        print("state_dict? ", "state_dict" in ck, "| entries:", len(state))
        pre = {}
        others = []
        for k in state:
            head = k.split(".")[0]
            pre[head] = pre.get(head, 0) + 1
            if head not in ("audio_model", "video_model"):
                others.append(k)
        print("prefix counts:", pre)
        print("non-audio/video keys:", len(others), others[:5])
        dtypes = sorted({str(v.dtype) for v in state.values() if torch.is_tensor(v)})
        print("tensor dtypes:", dtypes)
        for field in ("architecture", "epoch", "train_conf", "hyper_parameters", "optimizer_states"):
            if field in ck:
                v = ck[field]
                print(f"  meta[{field}]:", type(v).__name__,
                      (list(v.keys())[:6] if isinstance(v, dict) else str(v)[:80]))
        return ck, state
    raise SystemExit(f"unexpected checkpoint type: {type(ck)}")


# ---------------------------------------------------------------- A falcon.pt
SECTION("A. falcon.pt 顶层结构与 key 前缀")
ck_small, state_small = inspect_ckpt(WEIGHTS / "falcon.pt")

audio_state = {k[len("audio_model."):]: v for k, v in state_small.items() if k.startswith("audio_model.")}
video_state = {k[len("video_model."):]: v for k, v in state_small.items() if k.startswith("video_model.")}
print(f"audio_model.* = {len(audio_state)} keys, video_model.* = {len(video_state)} keys")
print("has optimizer state?", "optimizer_states" in ck_small, "| has epoch?", "epoch" in ck_small)

# ---------------------------------------------------------------- B training side
SECTION("B. 训练侧 look2hear：strict 加载真实权重")
import look2hear.models as l2h_models  # noqa: E402
import look2hear.videomodels as l2h_video  # noqa: E402

cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
train_audio = getattr(l2h_models, cfg["audionet"]["audionet_name"])(
    sample_rate=16000, **copy.deepcopy(cfg["audionet"]["audionet_config"])
)
missing, unexpected = train_audio.load_state_dict(audio_state, strict=True)
print("training audio_model strict load: OK (missing/unexpected returned empty)")
print("  params =", f"{sum(p.numel() for p in train_audio.parameters()):,}")

vconf = copy.deepcopy(cfg["videonet"]["videonet_config"])
vconf["pretrain"] = None
train_video = l2h_video.FRCNNVideoModel(**vconf)
train_video.load_state_dict(video_state, strict=True)
print("training video_model strict load: OK")
print("  params =", f"{sum(p.numel() for p in train_video.parameters()):,}")

# ---------------------------------------------------------------- C studio side
SECTION("C. Studio 侧 falcon_model：strict 加载 + 与训练侧 bit-identical 前向")
from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive as StudioSmall  # noqa: E402

scfg = yaml.safe_load((REPO / "studio" / "configs" / "Falcon.yml").read_text())
studio_audio = StudioSmall(sample_rate=16000, **copy.deepcopy(scfg["audionet"]["audionet_config"]))
studio_audio.load_state_dict(audio_state, strict=True)
print("studio audio_model strict load: OK, params =",
      f"{sum(p.numel() for p in studio_audio.parameters()):,}")

torch.manual_seed(0)
audio = torch.randn(1, 32000)
mouth_emb = torch.randn(1, 512, 50)
train_audio.eval()
studio_audio.eval()
with torch.no_grad():
    y_train = train_audio(audio, mouth_emb)
    y_studio = studio_audio(audio, mouth_emb)
d = (y_train - y_studio).abs().max().item()
print(f"eval-mode forward: train {tuple(y_train.shape)} vs studio {tuple(y_studio.shape)}, max|Δ| = {d:.3e}")
print("bit-identical:", d == 0.0)

train_audio.train()
studio_audio.train()
with torch.no_grad():
    t_train = train_audio(audio, mouth_emb)
    t_studio = studio_audio(audio, mouth_emb)
print("train-mode tuple shapes:", tuple(t_train[0].shape), tuple(t_train[1].shape))
print("train-mode est max|Δ| =", f"{(t_train[0] - t_studio[0]).abs().max().item():.3e}",
      "| sep_emb max|Δ| =", f"{(t_train[1] - t_studio[1]).abs().max().item():.3e}")
train_audio.eval()
studio_audio.eval()

# ---------------------------------------------------------------- D FalconRuntime
SECTION("D. FalconRuntime 真实加载 + 2s 窗口 infer")
from falcon_runtime import FalconRuntime, preprocess_mouth  # noqa: E402

rt = FalconRuntime(variant="small", checkpoint_path=str(WEIGHTS / "falcon.pt")).load("cpu")
st = rt.status()
print("status:", {k: st[k] for k in ("ready", "variant", "architecture", "device", "checkpoint",
                                     "sample_rate", "video_fps", "window_seconds",
                                     "audio_weight_keys", "video_weight_keys", "parameters",
                                     "warmup_ms", "mode")})
rng = np.random.default_rng(0)
audio_np = np.zeros(32000, dtype=np.float32)
mouth_np = np.full((50, 96, 96), 107, np.uint8)
out = rt.infer(audio_np, mouth_np)
print("infer output:", out.shape, out.dtype, "| finite:", np.isfinite(out).all())
frames_t = preprocess_mouth(mouth_np)
print("preprocess_mouth:", tuple(frames_t.shape), frames_t.dtype,
      "min/max =", f"{frames_t.min():.4f}/{frames_t.max():.4f}")
with torch.no_grad():
    emb = rt.video_model(frames_t)
    manual = rt.audio_model(torch.zeros(1, 32000), emb)
d2 = (manual[0, 0].numpy() - out).max() if manual.ndim == 3 else None
print("runtime window output shape check:", tuple(manual.shape), "| runtime vs manual max|Δ| =", d2)

# ---------------------------------------------------------------- E Falcon-Big
SECTION("E. falcon-big.pt + SpeakerContrastiveV2")
ck_big, state_big = inspect_ckpt(WEIGHTS / "falcon-big.pt")
big_audio_state = {k[len("audio_model."):]: v for k, v in state_big.items() if k.startswith("audio_model.")}
big_video_state = {k[len("video_model."):]: v for k, v in state_big.items() if k.startswith("video_model.")}
print(f"big audio keys={len(big_audio_state)}, video keys={len(big_video_state)}")

from falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2  # noqa: E402

bcfg = yaml.safe_load((REPO / "studio" / "configs" / "FalconBig.yml").read_text())
big_audio = SpeakerContrastiveV2(sample_rate=16000, **copy.deepcopy(bcfg["audionet"]["audionet_config"]))
big_audio.load_state_dict(big_audio_state, strict=True)
big_video = l2h_video.FRCNNVideoModel(**vconf)
big_video.load_state_dict(big_video_state, strict=True)
big_params = sum(p.numel() for p in big_audio.parameters()) + sum(p.numel() for p in big_video.parameters())
print(f"big strict load OK: audio={sum(p.numel() for p in big_audio.parameters()):,} "
      f"video={sum(p.numel() for p in big_video.parameters()):,} total={big_params:,} (README 12.35M)")

# ---------------------------------------------------------------- F audio_visual_test
SECTION("F. audio_visual_test 的 ckpt['state_dict'] 假设")
print("falcon.pt 顶层是否有 state_dict:", isinstance(ck_small, dict) and "state_dict" in ck_small)
print("audio_visual_test._load_models_from_ckpt 只做 ckpt['state_dict'] ->",
      "可行" if isinstance(ck_small, dict) and "state_dict" in ck_small else "会 KeyError")
print("state_dict 中 audio_model./video_model. 前缀 -> 该函数 strip 前缀后 strict=False 加载 -> 可工作（前提 conf.yml audionet_name 正确）")

SECTION("汇总")
print("small: audio keys", len(audio_state), "video keys", len(video_state),
      "| params", f"{rt.parameter_count:,}")
print("big  : audio keys", len(big_audio_state), "video keys", len(big_video_state),
      "| params", f"{big_params:,}")
print("DONE")
