"""R2 复核（layers-dsp → 05-data-pipeline）：执行级验证 A60/A64/A65/A68 与关键契约。

运行（仓库根）：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/review_05_data_pipeline_checks.py

只读：不修改源码、不联网；权重 /root/.cache/falcon_weights/falcon.pt（只读）。
"""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

SEC = lambda t: print("\n" + "=" * 78 + f"\n== {t}\n" + "=" * 78)

# ============================================================ V1 检查点格式
SEC("[V1] A68：falcon.pt 格式 / sha256 / strict 加载")
h = hashlib.sha256()
with open(CKPT, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print("  size =", CKPT.stat().st_size, "B")
print("  sha256 =", h.hexdigest())
print("  （05 文档称 47,156,343 B / a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c）")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
audio_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
video_sd = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
print("  top keys =", list(ck.keys()), "| architecture =", ck["architecture"], "| epoch =", ck["epoch"])
print(f"  state_dict keys = {len(sd)} (audio {len(audio_sd)} + video {len(video_sd)})")

import look2hear.models as models                                    # noqa: E402
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel  # noqa: E402
cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
acfg = cfg["audionet"]["audionet_config"]
audio_model = models.FalconSpeakerContrastive(sample_rate=16000, **acfg)
audio_model.load_state_dict(audio_sd, strict=True)
audio_model.eval()
video_model = FRCNNVideoModel(pretrain=None)
video_model.load_state_dict(video_sd, strict=True)
video_model.train(False)
print(f"  strict load OK；audio params={sum(p.numel() for p in audio_model.parameters()):,} "
      f"video params={sum(p.numel() for p in video_model.parameters()):,} "
      f"合计={sum(p.numel() for p in audio_model.parameters()) + sum(p.numel() for p in video_model.parameters()):,}")

# ============================================================ V2 整句推理契约
SEC("[V2] A68：真实权重整句评测契约（T=32000/F=50 与 T=48513/F=75）")
for T, F in ((32000, 50), (48513, 75)):
    torch.manual_seed(0)
    wav = torch.randn(1, T)
    mouth = torch.randn(1, 1, F, 88, 88)
    with torch.no_grad():
        emb = video_model(mouth)
        y = audio_model(wav, emb)
    tp = 1 + T // 128
    enc_shape = (1, 112, tp, 129)
    print(f"  T={T:6d} F={F:3d} -> video {tuple(emb.shape)}; audio T'=1+T//128={tp} "
          f"(05 称 {enc_shape}); est={tuple(y.shape)} len==T:{y.shape[-1] == T} finite:{torch.isfinite(y).all().item()}")

# ============================================================ V3 dtype 条件化
SEC("[V3] §5.4 dtype 条件化：uint8 .pt → float64?")
print("  numpy =", np.__version__)
u8 = np.arange(12, dtype=np.uint8).reshape(1, 3, 4)
print("  (uint8 - 0.0)/255.0 dtype =", ((u8 - 0.0) / 255.0).dtype)
f32 = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
print("  (float32 - 0.0)/255.0 dtype =", ((f32 - 0.0) / 255.0).dtype)

# 训练 transform 管线：transform.py 顶层 import cv2/torchvision，用空 module stub 注入（不改源码）
for name in ("cv2", "torchvision"):
    sys.modules.setdefault(name, types.ModuleType(name))
spec = importlib.util.spec_from_file_location("dp_transform", REPO / "look2hear/datas/transform.py")
T_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T_mod)
val_pipe = T_mod.get_preprocessing_pipelines()["val"]
frames_u8 = (np.random.RandomState(0).randint(0, 256, size=(50, 96, 96))).astype(np.uint8)
out_train = val_pipe(frames_u8)
print(f"  训练 val 管线输出: shape={out_train.shape} dtype={out_train.dtype} "
      f"min={out_train.min():.4f} max={out_train.max():.4f}")

# ============================================================ V4 studio vs 训练管线
SEC("[V4] A60：Studio preprocess_mouth vs 训练 val 管线（逐元素）")
sys.path.insert(0, str(REPO / "studio"))
import falcon_runtime as FR  # noqa: E402
studio_out = FR.preprocess_mouth(frames_u8)
print(f"  studio preprocess_mouth: shape={tuple(studio_out.shape)} dtype={studio_out.dtype}")
s_np = studio_out[0, 0].numpy()
print(f"  原始比较（float64 vs float32）max|Δ| = {np.max(np.abs(out_train - s_np)):.3e}")
print(f"  cast 训练输出到 float32 后 max|Δ| = "
      f"{np.max(np.abs(out_train.astype(np.float32) - s_np)):.3e}  -> 精确相等 = "
      f"{np.array_equal(out_train.astype(np.float32), s_np)}")
# 梯度帧（唯一值）便于判断是否为浮点误差
grad = np.tile(np.arange(96, dtype=np.uint8)[None, None, :], (50, 96, 1))
o1 = val_pipe(grad)
o2 = FR.preprocess_mouth(grad)[0, 0].numpy()
print(f"  梯度帧：未 cast max|Δ|={np.max(np.abs(o1 - o2)):.3e} | cast-f32 max|Δ|="
      f"{np.max(np.abs(o1.astype(np.float32) - o2)):.3e}")
# 公式手算对照
manual = (grad[:, 4:92, 4:92].astype(np.float64) / 255.0 - 0.421) / 0.165
print(f"  训练管线 vs 手算 float64: max|Δ|={np.max(np.abs(o1 - manual)):.3e}")

# ============================================================ V5 其它契约
SEC("[V5] A64/A65：val 循环与 _meta 链的关键源码断言")
lit = (REPO / "look2hear/system/av_litmodule.py").read_text().splitlines()
print("  av_litmodule.py:214-216 ->", [lit[i - 1].strip() for i in range(214, 217)])
print("  av_litmodule.py:125 ->", lit[124].strip())
print("  av_litmodule.py:174 ->", lit[173].strip())
train_py = (REPO / "audio_train.py").read_text().splitlines()
print("  audio_train.py:51 ->", train_py[50].strip())
print("  'trainer.test(' 在 audio_train.py:", any("trainer.test(" in x for x in train_py))
near_src = (REPO / "look2hear/datas/avspeech_dataset_near.py").read_text().splitlines()
for lo, hi in ((188, 191), (193, 193)):
    print(f"  near.py:{lo}-{hi}:", " || ".join(near_src[i - 1].strip() for i in range(lo, hi + 1)))
nass = (REPO / "look2hear/losses/nass_contrastive_loss.py").read_text().splitlines()
print("  nass_contrastive_loss.py:70-76:", " || ".join(nass[i - 1].strip() for i in range(70, 77)))
proc = (REPO / "DataPreProcess/process_lrs23.py").read_text().splitlines()
print("  process_lrs23.py:67-72:", " || ".join(proc[i - 1].strip() for i in range(67, 73)))
gen = (REPO / "DataPreProcess/generate_near_json.py").read_text().splitlines()
print("  generate_near_json.py:60-66:", " || ".join(gen[i - 1].strip() for i in range(60, 67)))
print("\nDONE")
