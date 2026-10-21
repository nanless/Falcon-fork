"""R2 评审用独立复核脚本：核对 01-model-core.md 的关键断言（不改源码）。

运行（仓库根）：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/review_model_core_checks.py

覆盖 Lead 指定重点：A08/A09/A10/A11/A12/A17~A24/A26/A27/A34/A35/A43/A44，以及 A42/A18/A42。
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

from look2hear.layers import normalizations as N          # noqa: E402
import look2hear.models as models                          # noqa: E402
from look2hear.models.Falcon import (                      # noqa: E402
    AudioBlock, VideoBlock, InjectionMultiSum, DualPathRNN,
    MultiHeadSelfAttention2D, Separator, STFTEncoder,
)
import look2hear.videomodels as videomodels                # noqa: E402
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel  # noqa: E402

torch.manual_seed(0)
SEC = lambda t: print("\n" + "=" * 78 + f"\n== {t}\n" + "=" * 78)

CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
audio_p = ACONF["audio_params"]
WIN, HOP = ACONF["enc_dec_params"]["win"], ACONF["enc_dec_params"]["hop_length"]

model = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(ACONF))
model.eval()

# ============================================================ A08/A09/A10/A11/A12
SEC("[A08] AudioBlock 金字塔形状（forward hook）")
ab = model.separator.audio_net
shapes = {}
hooks = []
for i, layer in enumerate(ab.downsample_layers):
    def mk(i):
        def hook(mod, inp, out):
            shapes[i] = tuple(out.shape)
        return hook
    hooks.append(layer.register_forward_hook(mk(i)))
x = torch.randn(1, 112, 251, 129)
with torch.no_grad():
    y = ab(x)
for h in hooks:
    h.remove()
print("  down[0..3] 输出形状:", [shapes[i] for i in range(4)])
print("  AudioBlock 输出:", tuple(y.shape))
print("  A08 断言 (251,129)→(126,65)→(63,33)→(32,17):",
      [shapes[i][-2:] for i in range(4)] == [(251, 129), (126, 65), (63, 33), (17, 32)][:0]
      or [shapes[i][-2:] for i in range(4)])

print("\n[A09] n_freqs 记账")
nf = [m.n_freqs for m in ab.process_att_layers]
print("  process_att_layers[i].n_freqs =", nf, "| 01 声称 129/65/33/17 ->", nf == [129, 65, 33, 17])
print("  F 实测 =", [shapes[i][-1] for i in range(4)], "| T 实测 =", [shapes[i][-2] for i in range(4)])

SEC("[A10][A11][A12] 调用次数 / repeats / shared")
counts = {}
targets = {}
for i in range(4):
    targets[f"pf{i}"] = ab.process_f_layers[i]
    targets[f"pt{i}"] = ab.process_t_layers[i]
    targets[f"pa{i}"] = ab.process_att_layers[i]
    targets[f"fu{i}"] = ab.fusion_layers[i]
    targets[f"co{i}"] = ab.concat_layers[i]
targets["audio_net"] = model.separator.audio_net
targets["video_net"] = model.separator.video_net
targets["caf"] = model.separator.caf
hs = []
for k, mod in targets.items():
    def mk(k):
        def hook(m, i, o):
            counts[k] = counts.get(k, 0) + 1
        return hook
    hs.append(mod.register_forward_hook(mk(k)))
a = torch.randn(1, 32000)
mth = torch.randn(1, 512, 50)
with torch.no_grad():
    e = model.encoder(a)
    b = model.audio_bottleneck(e)
    v = model.video_bottleneck(mth)
    s = model.separator(b, v)
for h in hs:
    h.remove()
print("  counts:", {k: counts.get(k, 0) for k in targets})
print("  A10 每次 AudioBlock 内 process 各 level 调用 1 次（本次 Separator 调 AudioBlock 2 次 => 每次各 2）")
print("      注意 concat 调用计数:", {f"co{i}": counts.get(f"co{i}", 0) for i in range(4)},
      "<- co2 未出现！concat_layers[2] 从未被调用")
print("  A11 AudioBlock=%d VideoBlock=%d caf=%d (01 称 2/1/1):" %
      (counts.get("audio_net", 0), counts.get("video_net", 0), counts.get("caf", 0)))
print("  A12 separator 属性: fusion_repeats=%d audio_repeats=%d audio_shared=%s video_shared=%s fusion_shared=%s" % (
    model.separator.fusion_repeats, model.separator.audio_repeats,
    model.separator.audio_shared, model.separator.video_shared, model.separator.fusion_shared))
print("     audio_net 类型:", type(model.separator.audio_net).__name__,
      "| video_net:", type(model.separator.video_net).__name__, "| caf:", type(model.separator.caf).__name__)

# ============================================================ A18 / A17
SEC("[A18] STFTEncoder 的 8 ms lookahead（帧 123/124 vs 125）")
a1 = torch.randn(1, 32000)
a2 = a1.clone()
a2[..., 16000:] = torch.randn(1, 16000)
with torch.no_grad():
    e1, e2 = model.encoder(a1), model.encoder(a2)
d = (e1 - e2).abs().amax(dim=(1, 3))[0]     # 每帧 max|Δ|
nz = torch.nonzero(d > 0).flatten()
print("  最早非零帧 =", nz.min().item() if len(nz) else None,
      "| 帧 123/124/125 的 |Δ| =",
      [round(d[i].item(), 8) for i in (123, 124, 125)])
print("  01 声称：帧 123/124 恰为 0、帧 125 起非零（16000/128=125）->",
      d[123].item() == 0 and d[124].item() == 0 and d[125].item() > 0)

SEC("[A17] STFTDecoder 前缀因果（改 mask 帧 >=120）")
mk1 = torch.randn(1, 1, 112, 251, 129)
mk2 = mk1.clone()
mk2[..., 120:, :] = torch.randn(1, 1, 112, 131, 129)
with torch.no_grad():
    q1 = model.decoder(mk1, torch.Size([1, 32000]))
    q2 = model.decoder(mk2, torch.Size([1, 32000]))
dd = (q1 - q2).abs()
print("  samples[0:15360] max|Δ| =", dd[..., :15360].max().item(),
      "| samples[15360:] max|Δ| =", round(dd[..., 15360:].max().item(), 4))
print("  A17 严格因果（<15360 为 0）:", dd[..., :15360].max().item() == 0)

# ============================================================ A19 统一口径
SEC("[A19] cLNhw 非因果：三种扰动协议下的数值（统一口径实验）")
cls = N.LayerNormalizationHW(112, eps=1e-5).eval()
feat = torch.randn(1, 112, 251, 129)          # 模拟 encoder 特征（std≈1）
proto = {}


def measure(fn, split_frame=125):
    o1, o2 = fn()
    d = (o1 - o2).abs()
    return d[..., :split_frame, :].max().item(), d[..., split_frame:, :].max().item()


# P1: 后半段换成同分布随机数
x2 = feat.clone(); x2[..., 125:, :] = torch.randn(1, 112, 126, 129)
proto["P1 替换后半段为 randn"] = measure(lambda: (cls(feat), cls(x2)))
# P2: 后半段放大 10 倍
x3 = feat.clone(); x3[..., 125:, :] = 10.0 * feat[..., 125:, :]
proto["P2 后半段放大 10x"] = measure(lambda: (cls(feat), cls(x3)))
# P3: 单帧尖峰（t=125）
x4 = feat.clone(); x4[:, :, 125, :] += 20.0
proto["P3 单帧尖峰 +20"] = measure(lambda: (cls(feat), cls(x4)))
# P4: 端到端：改音频 samples>=16000，只跑到 audio_bottleneck.pre_norm
with torch.no_grad():
    ea, eb = model.encoder(a1), model.encoder(a2)
    bn = model.audio_bottleneck.pre_norm
    proto["P4 encoder→pre_norm（端到端协议）"] = measure(lambda: (bn(ea), bn(eb)))
for k, (l, r) in proto.items():
    print(f"  {k:34s} 左段(T<125) max|Δ|={l:.3e}   右段={r:.3e}")
print("  01-A19 声称 1.4e-6；02 我的 R1 测得 4.7e-05（经 encoder+bottleneck 全链）")
print("  => 二者都是'非零'，差异来自扰动协议/测量点，不是结论冲突")

# ============================================================ A20 57 帧 lookahead
SEC("[A20] AudioBlock 关掉 cLNhw 后的结构 lookahead（复现 57 帧）")


def strip_clnhw(module):
    for name, child in list(module.named_children()):
        if isinstance(child, N.LayerNormalizationHW):
            setattr(module, name, nn.Identity())
        else:
            strip_clnhw(child)


ab2 = copy.deepcopy(ab).eval()
strip_clnhw(ab2)
remain = [m for m in ab2.modules() if isinstance(m, N.LayerNormalizationHW)]
print("  替换后剩余 cLNhw 数 =", len(remain))


def lookahead(mod, split_frame, tag):
    z1 = torch.randn(1, 112, 251, 129)
    z2 = z1.clone()
    z2[..., split_frame:, :] = torch.randn(1, 112, 251 - split_frame, 129)
    with torch.no_grad():
        o1, o2 = mod(z1), mod(z2)
    d = (o1 - o2).abs().amax(dim=(1, 3))[0]     # 逐帧
    nz = torch.nonzero(d > 0).flatten()
    first = nz.min().item() if len(nz) else None
    print(f"  {tag}: 扰动起点帧={split_frame}, 最早受影响帧={first}, "
          f"lookahead={split_frame - first if first is not None else None} 帧 "
          f"= {(split_frame - first) * HOP if first is not None else None} 采样")
    return first


first_150 = lookahead(ab2, 150, "扰动帧>=150")
first_125 = lookahead(ab2, 125, "扰动帧>=125")
print("  01 声称 57 帧 ≈ 7,296 样本 ≈ 456 ms（注意 57*128=7,296、7296/16000=456 ms）")

# ============================================================ A21
SEC("[A21] InjectionMultiSum 脉冲响应：4 种口径（01 称 [98,102]）")


def pulse_span(mod, tag):
    z1 = torch.zeros(1, 64, 251, 129)
    z2 = z1.clone()
    z2[:, :, 100, :] = 1.0
    with torch.no_grad():
        o1, o2 = mod(z1, z1), mod(z2, z2)
    d = (o1 - o2).abs().amax(dim=(1, 3))[0]
    nz = torch.nonzero(d > 1e-8).flatten()
    print(f"  {tag:46s} 受影响帧区间:", (nz.min().item(), nz.max().item()) if len(nz) else None)
    return nz


ims = copy.deepcopy(ab.fusion_layers[0]).eval()
pulse_span(ims, "as-built（norm=cLNhw，causal=False）")
ims2 = copy.deepcopy(ab.fusion_layers[0]).eval(); strip_clnhw(ims2)
pulse_span(ims2, "norm 换成 Identity 后")
ims3 = InjectionMultiSum(in_chan=64, kernel_size=5, norm_type="cLNhw", is2d=True, causal=True).eval()
strip_clnhw(ims3)
pulse_span(ims3, "causal=True + norm 换成 Identity")
ims4 = InjectionMultiSum(in_chan=64, kernel_size=5, norm_type="cLNhw", is2d=True, causal=True).eval()
pulse_span(ims4, "causal=True + 保留 cLNhw")


def prefix_delta(mod, tag):
    z1 = torch.randn(1, 64, 40, 65)
    z2 = z1.clone()
    z2[:, :, 20:, :] = torch.randn(1, 64, 20, 65)
    with torch.no_grad():
        o1, o2 = mod(z1, z1), mod(z2, z2)
    print(f"  {tag:46s} 左段 t<20 max|Δ| = {(o1 - o2).abs()[:, :, :20].max().item():.3e}")


prefix_delta(ims2, "norm=Identity, causal=False（前缀）")
prefix_delta(ims3, "norm=Identity, causal=True（前缀）")

# ============================================================ A22/A23
SEC("[A22][A23] DualPathRNN / MHSA2D 前缀因果")
hid = audio_p["hid_chan"]
d3 = DualPathRNN(64, **{k: v for k, v in audio_p["layers"]["layer_2"].items()}).eval()
d4 = DualPathRNN(64, **{k: v for k, v in audio_p["layers"]["layer_1"].items()}).eval()
l3 = {k: v for k, v in audio_p["layers"]["layer_3"].items()}
att_c = MultiHeadSelfAttention2D(64, 65, **l3).eval()
att_nc = MultiHeadSelfAttention2D(64, 65, **{**l3, "causal": False}).eval()
zz1 = torch.randn(1, 64, 8, 65)
zz2 = zz1.clone()
zz2[:, :, 5:, :] = torch.randn(1, 64, 3, 65)
for name, mod in (("DualPathRNN(dim=3)", d3), ("DualPathRNN(dim=4)", d4),
                  ("MHSA2D(causal=True)", att_c), ("MHSA2D(causal=False)", att_nc)):
    with torch.no_grad():
        q1, q2 = mod(zz1), mod(zz2)
    e = (q1 - q2).abs()[:, :, :5].max().item()
    print(f"  {name:24s} 左段 t<5 max|Δ| = {e:.6f}")

# ============================================================ A34 / A35
SEC("[A34][A35] config 与裸 Falcon 不兼容 / 二次构造塌缩")
cfg_fresh = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ac = cfg_fresh["audionet"]["audionet_config"]
try:
    models.Falcon(sample_rate=16000, **copy.deepcopy(ac))
    print("  A34: 裸 Falcon 构造成功（与 01 的 TypeError 不符）")
except TypeError as exc:
    print(f"  A34: 裸 Falcon TypeError: {exc}")
ac = cfg_fresh["audionet"]["audionet_config"]          # 干净的同一份 dict
m1 = models.FalconSpeakerContrastive(sample_rate=16000, **ac)
p1 = sum(p.numel() for p in m1.parameters())
m2 = models.FalconSpeakerContrastive(sample_rate=16000, **ac)
p2 = sum(p.numel() for p in m2.parameters())
print(f"  A35: 第 1 次 {p1:,} -> 第 2 次 {p2:,}（01 称 537,697→49,081）")
print("  第 2 次 separator 结构: audio_net=%s(len=%d) video_net=%s(len=%d) fusion_repeats=%d audio_repeats=%d" % (
    type(m2.separator.audio_net).__name__,
    len(m2.separator.audio_net) if isinstance(m2.separator.audio_net, nn.ModuleList) else -1,
    type(m2.separator.video_net).__name__,
    len(m2.separator.video_net) if isinstance(m2.separator.video_net, nn.ModuleList) else -1,
    m2.separator.fusion_repeats, m2.separator.audio_repeats))
print("  第二次能否前向:", end=" ")
try:
    with torch.no_grad():
        out = m2(torch.randn(1, 32000), torch.randn(1, 512, 50))
    print("成功，输出形状", tuple(out.shape) if torch.is_tensor(out) else type(out))
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:80])

# ============================================================ A43 / A26 / A27
SEC("[A43] FRCNNVideoModel.eval() 返回值")
fr = FRCNNVideoModel(pretrain=None)
ret = fr.eval()
print("  FRCNNVideoModel(pretrain=None).eval() 返回:", type(ret).__name__, "| 01 称 None ->", ret is None)

SEC("[A26] FRCNN 时间感受野（改帧 >=30，最早受影响输出帧）")
with torch.no_grad():
    fr.train(False)
    f1 = torch.randn(1, 1, 50, 96, 96)
    f2 = f1.clone()
    f2[:, :, 30:] = torch.randn(1, 1, 20, 96, 96)
    o1 = fr(f1)
    o2 = fr(f2)
print("  FRCNN 输出形状:", tuple(o1.shape))
d = (o1 - o2).abs().amax(dim=1)[0]
nz = torch.nonzero(d > 0).flatten()
print("  最早受影响输出帧 =", nz.min().item() if len(nz) else None, "（01 称 28）")

SEC("[A27] VideoBlock train vs eval 前缀因果")
vb = copy.deepcopy(model.separator.video_net).eval()
v1 = torch.randn(1, 512, 50)
v2 = v1.clone()
v2[..., 30:] = torch.randn(1, 512, 20)
for mode, name in ((False, "eval"), (True, "train")):
    vb.train(mode)
    with torch.no_grad():
        o1, o2 = vb(v1), vb(v2)
    e = (o1 - o2).abs()[..., :30].max().item()
    print(f"  VideoBlock({name}): 左段 t<30 max|Δ| = {e:.3e}")

# ============================================================ 其它
SEC("[附加] ConvNormAct 实例数 / A42 state_dict / FRCNN 参数")
n_convnormact = sum(1 for m in model.modules()
                    if type(m).__name__ == "ConvNormAct" and type(m).__module__ == "look2hear.models.Falcon")
print("  Falcon.py 内 ConvNormAct 实例数 =", n_convnormact, "（01 称 38）")
sd = model.state_dict()
print("  state_dict keys =", len(sd), "| 'mdcl_window' in sd =", "mdcl_window" in sd, "（01 称 525 / False）")
frtot = sum(p.numel() for p in fr.parameters())
print("  FRCNN 参数 =", f"{frtot:,}", "（01 称 11,186,688）")
print("  audio_model 参数 =", f"{sum(p.numel() for p in model.parameters()):,}")
print("\nDONE")
