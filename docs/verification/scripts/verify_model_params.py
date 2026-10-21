"""Lead 独立验证脚本 1：模型结构 / 参数量 / 形状 / 因果性 / 死代码。

运行（在仓库根）：
    PYTHONPATH=<repo>/docs/verification/stubs:<repo> \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_model_params.py

环境说明：
- 系统 python 是 3.8 且无 torch；本脚本使用 /root/.venvs/falcon (py3.10 + torch 2.8 + pytorch_lightning 2.6)。
- `timm` 用 docs/verification/stubs 下的 DropPath stub 替代（真实 timm 依赖 torchvision，沙箱未装）。
- `typeguard.check_argument_types` 同样用 stub（真实 typeguard>=3 已移除该 API）。
- 不加载真实权重（沙箱无 falcon.pt / pretrain_zoo），只验证结构、参数量、形状、因果性与接口。
"""
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
aconf = cfg["audionet"]["audionet_config"]

import look2hear.models as models  # noqa: E402
import look2hear.videomodels as videomodels  # noqa: E402

print("=" * 78)
print("== audionet_name:", cfg["audionet"]["audionet_name"])
print("=" * 78)


def count(mod):
    total = sum(p.numel() for p in mod.parameters())
    train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return total, train


model = getattr(models, cfg["audionet"]["audionet_name"])(
    sample_rate=cfg["datamodule"]["data_config"]["sample_rate"], **aconf
)
model.eval()

print("\n-- 音频网络各子模块参数量 --")
for name, child in model.named_children():
    t, tr = count(child)
    print(f"{name:20s} total={t:>10,} trainable={tr:>10,}")

tot, train = count(model)
print(f"\nAUDIO_MODEL total={tot:,} trainable={train:,}")

print("\n-- 冻结视觉编码器 FRCNNVideoModel --")
vcfg = dict(cfg["videonet"]["videonet_config"])
vcfg["pretrain"] = None
frcnn = videomodels.FRCNNVideoModel(**vcfg)
f_tot, f_train = count(frcnn)
print(f"FRCNN total={f_tot:,} trainable={f_train:,} (未加载 pretrain 时 requires_grad=True)")
print(f"AUDIO+FRCNN total={tot + f_tot:,}  trainable(audio only)={train:,}")

frames = torch.randn(1, 1, 50, 96, 96)
with torch.no_grad():
    vout = frcnn(frames)
print("FRCNN forward [1,1,50,96,96] ->", tuple(vout.shape), vout.dtype)

print("\n-- 前向形状随输入长度变化 (mouth_emb 时间步 50, train 模式返回 (est, emb)) --")
print(f"{'audio L':>8} | {'est':>18} | {'sep_emb':>28} | {'frames=1+L/hop?':>16}")
for L in (16000, 32000, 48000):
    a = torch.randn(1, L)
    m = torch.randn(1, aconf["pretrained_vout_chan"], 50)
    model.train()
    with torch.no_grad():
        est, emb = model(a, m)
    expected = 1 + L // aconf["enc_dec_params"]["hop_length"]
    print(f"{L:>8} | {str(tuple(est.shape)):>18} | {str(tuple(emb.shape)):>28} | {expected:>16}")
model.eval()

print("\n-- 因果性探测：只改后半段音频 (sample>=16000)，观察不同区域输出变化 --")
torch.manual_seed(0)
B, T = 1, 32000
mouth = torch.randn(B, aconf["pretrained_vout_chan"], 50)
a1 = torch.randn(B, T)
a2 = a1.clone()
a2[..., T // 2:] = torch.randn(B, T // 2)


def region_diff(x1, x2, name, split):
    d = (x1 - x2).abs()
    early = d[..., :split]
    late = d[..., split:]
    print(f"  {name:18s} early(0:{split}) max|d|={early.max().item():.3e}   "
          f"late({split}:) max|d|={late.max().item():.3e}   shape={tuple(x1.shape)}")


with torch.no_grad():
    e1, e2 = model.encoder(a1), model.encoder(a2)
    # STFT center=True: 帧 i 依赖采样 [i*hop-win/2, i*hop+win/2)，故前半段帧应完全不变
    region_diff(e1, e2, "encoder(frames)", 120)
    b1, b2 = model.audio_bottleneck(e1), model.audio_bottleneck(e2)
    # audio_bottleneck 的 pre_norm 是 cLNhw：GlobalLayerNorm 在全窗口统计 -> 泄漏
    region_diff(b1, b2, "audio_bottleneck", 120)
    v = model.video_bottleneck(mouth)
    print("  video_bottleneck   shape:", tuple(v.shape),
          " params:", sum(p.numel() for p in model.video_bottleneck.parameters()))
    s1, s2 = model.separator(b1, v), model.separator(b2, v)
    region_diff(s1, s2, "separator", 120)
    m1, m2 = model.mask_generator(s1, e1), model.mask_generator(s2, e2)
    region_diff(m1, m2, "mask_generator", 120)
    y1 = model.decoder(m1, a1.shape)
    y2 = model.decoder(m2, a2.shape)
    L = y1.shape[-1]
    for lo, hi in [(0, 8000), (8000, 16000), (16000, 24000), (24000, 32000)]:
        print(f"  est.samples[{lo:>5}:{hi:>5}] max|d| = "
              f"{(y1[..., lo:hi] - y2[..., lo:hi]).abs().max().item():.3e}")

print("\n-- 死代码验证：Falcon.forward 引用了不存在的 _mdcl_loss --")
base_aconf = {k: v for k, v in aconf.items() if k != "spk_embed_dim"}
base = models.Falcon(sample_rate=16000, **base_aconf)
print("  Falcon._mdcl_loss_hard:", hasattr(base, "_mdcl_loss_hard"),
      "| Falcon._mdcl_loss:", hasattr(base, "_mdcl_loss"))
base.train()
try:
    base(torch.randn(1, 32000), torch.randn(1, 512, 50), near_audio=torch.randn(1, 32000))
    print("  Falcon(train, near_audio) -> 未报错（与源码预期不符）")
except AttributeError as e:
    print("  Falcon(train, near_audio) -> AttributeError:", e)

print("\n-- 关键常量 --")
print("  encoder out chan:", model.encoder.get_out_chan())
print("  mdcl_window:", tuple(model.mdcl_window.shape),
      "| persistent:", "mdcl_window" in dict(model.named_buffers()).keys())
print("  spk_embed_dim:", getattr(model, "spk_embed_dim", None))
print("  n_src:", model.n_src, "| causal:", model.causal)
