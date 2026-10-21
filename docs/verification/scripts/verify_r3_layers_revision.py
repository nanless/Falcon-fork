"""R3 修订用复核：A36'/§4.1 采样级/ A19 协议表 / A49 逐级 / A57 行号 / §2.1 计数 / PL 依赖链。

运行（仓库根）：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r3_layers_revision.py
"""
import ast
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

from look2hear.layers import normalizations as N              # noqa: E402
import look2hear.models as models                              # noqa: E402
from look2hear.models.Falcon import AudioBlock                 # noqa: E402

torch.manual_seed(0)
SEC = lambda t: print("\n" + "=" * 78 + f"\n== {t}\n" + "=" * 78)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
WIN, HOP = ACONF["enc_dec_params"]["win"], ACONF["enc_dec_params"]["hop_length"]

model = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(ACONF))
model.eval()


def first_changed_1d(d, thresh=0.0):
    nz = torch.nonzero(d > thresh).flatten()
    return nz.min().item() if len(nz) else None


# ============================================================ A36'
SEC("[V1] A36'：STFTEncoder 帧级 lookahead（s=16000 / 20000）")
base = torch.randn(1, 32000)
for s0 in (16000, 16001, 16255, 16256, 20000, 21000):
    pert = base.clone()
    torch.manual_seed(1000 + s0)
    pert[..., s0:] = torch.randn(1, 32000 - s0)
    with torch.no_grad():
        e1, e2 = model.encoder(base), model.encoder(pert)
    d = (e1 - e2).abs().amax(dim=(1, 3))[0]
    first = first_changed_1d(d)
    print(f"  扰动 s>={s0:5d} -> 首变帧={first}  (帧 {first - 1 if first else '-'} delta="
          f"{d[first - 1].item() if first else float('nan'):.3e}, 帧 {first} delta={d[first].item():.3e})"
          if first else f"  扰动 s>={s0} -> 无变化")

# ============================================================ 采样级串联
SEC("[V2] encoder→decoder 串联的采样级过去泄漏（解析上界 ≤127）")
print("  解析：输入采样 p 影响 encoder 帧 n∈[ceil((p-127)/128), floor((p+127)/128)]；")
print("        decoder 合成帧 m 依赖 encoder 帧 m-3..m-1；iSTFT 采样 s 由帧 ⌊s/128⌋、⌊s/128⌋+1 叠加")
print("        => 给定 p=128q+r，最早受影响合成帧 m=q+1，最早受影响采样 s_min=128q=p-r")
print("        => 采样级过去泄漏 = r ≤ 127（再减去 hann 窗首点 0 的 1 个采样）")
base = torch.randn(1, 32000)
for s0 in (16000, 16001, 20000, 20032, 20127, 24000):
    pert = base.clone()
    torch.manual_seed(2000 + s0)
    pert[..., s0:] = torch.randn(1, 32000 - s0)
    with torch.no_grad():
        e1, e2 = model.encoder(base), model.encoder(pert)
        y1 = model.decoder(e1, base.shape)
        y2 = model.decoder(e2, base.shape)
    d = (y1 - y2).abs()[0, 0]
    first = first_changed_1d(d)
    r = s0 % HOP
    print(f"  p={s0:5d} (r={r:3d}) -> 首变输出采样={first}；泄漏 p-first={s0 - first:3d}"
          f"（解析上界 r={r}，hann 首点 0 使实测 r-1={r - 1}）")

# ============================================================ A19 协议表
SEC("[V3] A19 扰动协议表 + common-mode 抵消 + GN-only vs cLNhw")
c_ln_hw = N.LayerNormalizationHW(64, eps=1e-5).eval()
t8 = torch.randn(1, 64, 8, 65)
spike = t8.clone(); spike[:, :, 6, :] += 20.0
with torch.no_grad():
    o1, o2 = c_ln_hw(t8), c_ln_hw(spike)
print(f"  (a) cLNhw T=8 单帧尖峰+20@t=6 -> 前缀 t<5 max|Δ| = "
      f"{(o1 - o2).abs()[..., :5, :].max().item():.3e}")
b_frame_all = t8.clone(); b_frame_all[:, :, 6, :] += 20.0        # 单帧全通道 +20
b_frame_one = t8.clone(); b_frame_one[:, 0, 6, :] += 20.0        # 单帧单通道 +20
b_all_all = t8 + 5.0                                              # 整段全通道同量 +5（真 common-mode）
b_all_one = t8.clone(); b_all_one[:, 0, :, :] += 5.0              # 整段单通道 +5
with torch.no_grad():
    base_o = c_ln_hw(t8)
    for tag, xx in (("单帧+全通道 +20", b_frame_all), ("单帧+单通道 +20", b_frame_one),
                    ("整段+全通道 +5 (common-mode)", b_all_all), ("整段+单通道 +5", b_all_one)):
        dd = (base_o - c_ln_hw(xx)).abs()
        print(f"  (b) {tag:30s} 前缀 t<5 max|Δ| = {dd[..., :5, :].max().item():.3e} | "
              f"全段 max|Δ| = {dd.max().item():.3e}")

a1 = torch.randn(1, 32000)
a_x10 = a1.clone(); a_x10[..., 16000:] = 10.0 * a1[..., 16000:]
a_rep = a1.clone(); torch.manual_seed(7); a_rep[..., 16000:] = torch.randn(1, 16000)
pre = model.audio_bottleneck.pre_norm
with torch.no_grad():
    e_base = model.encoder(a1)
    for tag, aw in (("波形×10", a_x10), ("同分布替换", a_rep)):
        ew = model.encoder(aw)
        p1, p2 = pre(e_base), pre(ew)
        b2 = model.audio_bottleneck(ew)
        b1 = model.audio_bottleneck(e_base)
        print(f"  (c) {tag:8s}: pre_norm 左段(T<125) = {(p1 - p2).abs()[..., :125, :].max().item():.3e} | "
              f"audio_bottleneck 左段 = {(b1 - b2).abs()[..., :125, :].max().item():.3e}")

# GN-only vs cLNhw：同一 ×10 探针
gln = N.GlobalLayerNorm(64, eps=1e-5).eval()
c4 = N.CumulateLN4D(64, eps=1e-5).eval()
x1 = torch.randn(1, 64, 251, 65)
x2 = x1.clone(); x2[..., 125:, :] = 10.0 * x1[..., 125:, :]
with torch.no_grad():
    g1, g2 = gln(x1), gln(x2)
    h1, h2 = c_ln_hw(x1), c_ln_hw(x2)
print(f"  (d) GN-only  ×10 探针左段 = {(g1 - g2).abs()[..., :125, :].max().item():.3e}")
print(f"      cLNhw    ×10 探针左段 = {(h1 - h2).abs()[..., :125, :].max().item():.3e}"
      f"  （GN 的纯尺度泄漏被 CumulateLN4D 抵消）")

# ============================================================ A49 逐级 lookahead
SEC("[V4] A49：逐级对称卷积 lookahead + 单遍 AudioBlock 57 帧")
print("  解析（每级 downsample conv）：L0 k=3,p=1,s=1 -> ±1 L0 帧；")
print("    L1/L2/L3 k=5,p=2,s=2 -> 本级 ±2 帧，换到 L0 帧 = ±2·2^level -> ±4 / ±8 / ±16")
ab = copy.deepcopy(model.separator.audio_net).eval()


def strip_clnhw(module):
    for name, child in list(module.named_children()):
        if isinstance(child, N.LayerNormalizationHW):
            setattr(module, name, nn.Identity())
        else:
            strip_clnhw(child)


strip_clnhw(ab)
shapes = {}
hs = []
for i, layer in enumerate(ab.downsample_layers):
    def mk(i):
        def hook(mod, inp, out):
            shapes[i] = tuple(out.shape)
        return hook
    hs.append(layer.register_forward_hook(mk(i)))
z1 = torch.randn(1, 112, 251, 129)
z2 = z1.clone(); torch.manual_seed(11); z2[..., 125:, :] = torch.randn(1, 112, 126, 129)
with torch.no_grad():
    o1, o2 = ab(z1), ab(z2)
for h in hs:
    h.remove()
print("  各级下采样输出形状:", [shapes[i][-2:] for i in range(4)])
d = (o1 - o2).abs().amax(dim=(1, 3))[0]
first_out = first_changed_1d(d)
print(f"  AudioBlock 输出首变帧={first_out}（扰动帧>=125）-> 单遍 lookahead = {125 - first_out} 帧 "
      f"= {(125 - first_out) * HOP} 采样 = {(125 - first_out) * HOP / 16000 * 1000:.0f} ms")
print("  说明：Falcon.py:818-824 的 residual_conv 是 k=1（无时间感受野），不能作为非因果证据；")
print("        非因果结构证据应为 :765（downsample k=3/5）、:801（fusion k=5）、:814（concat k=5）。")

# ============================================================ V4b/V4c 视频分支
SEC("[V4b] VideoBlock：eval vs train 前缀因果（BatchNorm1d 批内统计）")
vb = copy.deepcopy(model.separator.video_net).eval()
v1 = torch.randn(1, 512, 50)
v2 = v1.clone(); torch.manual_seed(21); v2[..., 30:] = torch.randn(1, 512, 20)
for mode, tag in ((False, "eval"), (True, "train")):
    vb.train(mode)
    with torch.no_grad():
        oo1, oo2 = vb(v1), vb(v2)
    print(f"  VideoBlock({tag:5s}) 左段 t<30 max|Δ| = {(oo1 - oo2).abs()[..., :30].max().item():.3e}")

SEC("[V4c] FRCNN frontend3D 时间感受野（改帧 >=30，最左受影响输出帧）")
sys.path.insert(0, str(REPO))
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel  # noqa: E402
fr = FRCNNVideoModel(pretrain=None)
fr.train(False)   # 注意：FRCNNVideoModel.train() 重写后无 return self，所以不能用 fr.eval()
with torch.no_grad():
    f1 = torch.randn(1, 1, 50, 96, 96)
    f2 = f1.clone(); torch.manual_seed(31); f2[:, :, 30:] = torch.randn(1, 1, 20, 96, 96)
    o1, o2 = fr(f1), fr(f2)
print("  FRCNN 输出形状:", tuple(o1.shape))
dd = (o1 - o2).abs().amax(dim=1)[0]
nz = torch.nonzero(dd > 0).flatten()
print("  最早受影响输出帧 =", nz.min().item() if len(nz) else None,
      "（解析：Conv3d k=5,p=2 ⇒ ±2 帧 = ±80 ms）")

# ============================================================ A57 行号
SEC("[V5] A57 证据行号：causal=False 分支的 Identity")
src = (REPO / "look2hear/models/Falcon.py").read_text().splitlines()
for ln in (30, 66, 67, 68, 84, 85, 86):
    print(f"  Falcon.py:{ln:3d}: {src[ln - 1]}")

# ============================================================ §2.1 名字计数
SEC("[V6] layers/__init__.py 导入名字数（AST）")
tree = ast.parse((REPO / "look2hear/layers/__init__.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        names = [a.name for a in node.names]
        if node.module in ("cnnlayers", "rnnlayers"):
            print(f"  from .{node.module} import -> {len(names)} 个: {names}")

# ============================================================ PL 依赖链
SEC("[V7] import look2hear.layers 的依赖链（PL/librosa/torch_complex）")
print("  utils/__init__.py 相关行：", repr(
    (REPO / "look2hear/utils/__init__.py").read_text().splitlines()[11]))
print("  lightning_utils.py 顶层 import:",
      repr((REPO / "look2hear/utils/lightning_utils.py").read_text().splitlines()[2]))
print("  当前进程 sys.modules（此进程早已 import look2hear.models）:")
for m in ("pytorch_lightning", "librosa", "torch_complex", "typeguard", "espnet.nets.pytorch_backend.nets_utils"):
    print(f"    {m}: {m in sys.modules}")
# cLNhw 实例数
mods = list(model.modules())
print("\n  LayerNormalizationHW 实例数 =", sum(1 for x in mods if isinstance(x, N.LayerNormalizationHW)),
      "| GlobalLayerNorm =", sum(1 for x in mods if isinstance(x, N.GlobalLayerNorm)),
      "| CumulateLN4D =", sum(1 for x in mods if isinstance(x, N.CumulateLN4D)),
      "| LayerNormalization4D =", sum(1 for x in mods if isinstance(x, N.LayerNormalization4D)))
print("\nDONE")
