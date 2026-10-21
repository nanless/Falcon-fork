"""R3 复核：真实权重下 Studio 副本与训练副本的 state_dict 对齐 + CPU 数值一致性。

用途：为 02 文档 A44/A52/A53 提供 [验证] 级证据（PROVENANCE.md:33 声称"CPU 上 bit-identical"）。
权重：/root/.cache/falcon_weights/falcon.pt（Lead 已下载；只读，不联网、不训练）。

运行（仓库根）：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r3_studio_weights.py
"""
import copy
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
WEIGHTS = Path("/root/.cache/falcon_weights/falcon.pt")
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

SEC = lambda t: print("\n" + "=" * 78 + f"\n== {t}\n" + "=" * 78)

# ---------- 1) 先只 import studio 副本（检查依赖面） ----------
SEC("[W1] Studio 副本：import 面与 strict=True 加载")
sys.path.insert(0, str(REPO / "studio"))
import falcon_model.models.Falcon_Contrastive as sFC   # noqa: E402
import falcon_model.models.Falcon as sF                # noqa: E402
print("  studio classes:", sFC.FalconSpeakerContrastive.__name__, "/", sF.Falcon.__name__)
print("  librosa loaded:", "librosa" in sys.modules,
      "| torch_complex loaded:", "torch_complex" in sys.modules,
      "| look2hear loaded:", any(m.startswith("look2hear") for m in sys.modules))

ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
print("  ckpt keys:", list(ck.keys()), "| architecture:", ck["architecture"], "| epoch:", ck["epoch"])
sd = ck["state_dict"]
audio_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
video_sd = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
print(f"  audio_model.* keys={len(audio_sd)}  video_model.* keys={len(video_sd)}  total={len(sd)}")

studio_cfg = yaml.safe_load((REPO / "studio/configs/Falcon.yml").read_text())
s_ac = copy.deepcopy(studio_cfg["audionet"]["audionet_config"])
m_studio = sFC.FalconSpeakerContrastive(sample_rate=16000, **s_ac)
m_studio.eval()
print("  studio strict=True load:", m_studio.load_state_dict(audio_sd, strict=True))
print("  studio params:", f"{sum(p.numel() for p in m_studio.parameters()):,}",
      "| state_dict keys:", len(m_studio.state_dict()))

torch.manual_seed(0)
audio = torch.randn(1, 32000)
mouth = torch.randn(1, 512, 50)
with torch.no_grad():
    y_studio = m_studio(audio, mouth)
print("  studio forward:", tuple(y_studio.shape), "| stats:",
      round(y_studio.min().item(), 5), round(y_studio.max().item(), 5))

# ---------- 2) 再 import 训练副本，比较 ----------
SEC("[W2] 训练副本 vs Studio 副本（真实权重、CPU、eval）")
sys.path.insert(0, str(REPO))
import look2hear.models as models                      # noqa: E402
print("  librosa loaded now:", "librosa" in sys.modules,
      "| pytorch_lightning:", "pytorch_lightning" in sys.modules,
      "| torch_complex:", "torch_complex" in sys.modules)
train_cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
t_ac = copy.deepcopy(train_cfg["audionet"]["audionet_config"])
m_train = models.FalconSpeakerContrastive(sample_rate=16000, **t_ac)
m_train.eval()
print("  training strict=True load:", m_train.load_state_dict(audio_sd, strict=True))
print("  training params:", f"{sum(p.numel() for p in m_train.parameters()):,}",
      "| state_dict keys:", len(m_train.state_dict()))
with torch.no_grad():
    y_train = m_train(audio, mouth)
diff = (y_train - y_studio).abs().max().item()
print(f"  train vs studio 输出 max|Δ| = {diff:.3e} -> bit-identical: {diff == 0.0}")
print("  state_dict 两副本 key 完全一致:",
      list(m_train.state_dict().keys()) == list(m_studio.state_dict().keys()))

# ---------- 3) Studio 侧的 MPS bridge 静态核对 ----------
SEC("[W3] Studio 的 CPU FFT bridge 行号与内容（训练侧无）")
s_src = (REPO / "studio/falcon_model/models/Falcon.py").read_text().splitlines()
for lo, hi, tag in ((1107, 1124, "STFTEncoder.forward"), (1176, 1200, "STFTDecoder.forward")):
    print(f"  -- {tag} (studio {lo}-{hi}) --")
    for i in range(lo, hi + 1):
        line = s_src[i - 1]
        if any(k in line for k in ("compute_device", "stft_x", "output_device", ".cpu()",
                                    "spec.device", "spec.to(", "output.to(", "torch.stft", "torch.istft")):
            print(f"    {i}: {line.strip()}")
print("\nDONE")
