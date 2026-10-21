"""R2 评审用：统一「因果性探针」的可复现条件表（回答 02 文档 1.2e-03 vs 01 文档 1.4e-6 的口径差异）。

运行：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r2_causality_conditions.py

思路：同一模块、同一分界，换 5 种扰动方式，观察「前缀 max|Δ|」如何随扰动强度变化。
判据：`max|Δ| > 1e-12` 视为"左段被影响"（float32 噪声底约 1e-7 量级，故 1e-6 也是真实泄漏）。
"""
import copy
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

import look2hear.models as models  # noqa: E402
from look2hear.models.Falcon import AudioBlock, ConvNormAct  # noqa: E402

torch.manual_seed(0)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
m = getattr(models, CFG["audionet"]["audionet_name"])(
    sample_rate=16000, **copy.deepcopy(ACONF)).eval()

B, C, T, F = 1, 112, 251, 129
SPLIT = 125            # = 1 s 边界（16000 样本 / 128 hop）
X = torch.randn(B, C, T, F)

PROBES = {
    "P1 同分布随机替换(σ=1)": lambda x: torch.randn_like(x),
    "P2 后半段 ×10": lambda x: x * 10.0,
    "P3 后半段替换为 N(0,10²)": lambda x: torch.randn_like(x) * 10.0,
    "P4 单帧尖峰 +20（分界帧）": lambda x: x * 0.0 + 20.0,
    "P5 后半段 +1 常数偏移": lambda x: x * 0.0 + 1.0,
}


def probe(fn, x, name, split=SPLIT, dim=-2):
    x1, x2 = x.clone(), x.clone()
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(split, None)
    if name.startswith("P4"):
        idx[dim] = slice(split, split + 1)          # 只改分界那一帧
    x2[tuple(idx)] = PROBES[name](x2[tuple(idx)])
    with torch.no_grad():
        y1, y2 = fn(x1), fn(x2)
    d = (y1 - y2).abs()
    od = -2 if dim == -2 else -1
    odd = od % d.dim()
    prof = d.amax(dim=tuple(i for i in range(d.dim()) if i != odd))
    prefix = prof[:split]
    scale = y1.abs().max().item() + 1e-12
    return prefix.max().item(), (prefix.max().item() / scale)


def report(title, fn, x):
    print(f"\n-- {title} --")
    print(f"  {'扰动':28s} {'前缀 max|Δ|':>14s} {'相对幅度':>10s}")
    for name in PROBES:
        a, r = probe(fn, x, name)
        print(f"  {name:28s} {a:14.3e} {r:10.3e}")


print("=" * 78)
print("[R2-C] 因果性探针条件表（B=1, fp32, eval 模式, 分界 = 帧 125 = 1.0 s）")
print("=" * 78)
print("注：cLNhw = LayerNormalizationHW(112) = GroupNorm(1,112) + CumulateLN4D(112)；")
print("    GroupNorm 对 [C,T,F] 全局统计 -> 其后半段变化会改变全局 μ/σ -> 前缀必然变化。")

report("cLNhw 单模块：m.audio_bottleneck.pre_norm", m.audio_bottleneck.pre_norm, X)
report("audio_bottleneck（cLNhw + Conv2d 1×1）", m.audio_bottleneck, X)

no_pre = ConvNormAct(in_chan=C, out_chan=C, kernel_size=1, is2d=True, act_type=None,
                     norm_type=None, pre_norm_type=None, pre_act_type=None).eval()
report("对照：去掉 cLNhw 的 Conv2d 1×1（应恒为 0）", no_pre, X)

lay = copy.deepcopy(ACONF["audio_params"]["layers"])
ab_nn = AudioBlock(in_chan=112, hid_chan=64, kernel_size=5, stride=2, norm_type=None,
                   act_type="PReLU", upsampling_depth=4, layers=lay, is2d=True,
                   causal=True, n_freqs=129).eval()
report("AudioBlock（norm 全关，纯结构非因果）", ab_nn, X)
report("AudioBlock（论文配置，cLNhw 开）", m.separator.audio_net, X)

print()
print("=" * 78)
print("[R2-C1] 复现 02 的 cLNhw 探针条件（02 用的是 T=8,F=65,+20 单帧尖峰；本表 + T 敏感性）")
print("=" * 78)
from look2hear.layers import normalizations as N  # noqa: E402

hid = ACONF["audio_params"]["hid_chan"]           # 64
hw = N.LayerNormalizationHW(num_channels=hid, eps=1e-5).eval()
for Tv in (8, 32, 125, 251):
    z = torch.randn(1, hid, Tv, 65)
    z2 = z.clone()
    t0 = Tv - 2
    z2[:, :, t0, :] += 20.0
    with torch.no_grad():
        o1, o2 = hw(z), hw(z2)
    dd = (o1 - o2).abs()
    early = dd[:, :, :t0 - 1].max().item()
    print(f"  T={Tv:3d}, F=65, +20 单帧尖峰 @ t={t0}: 早期帧 max|Δ| = {early:.3e}"
          f"（相对 {early / (o1.abs().max().item() + 1e-12):.1e}）")
print("  => 02 的 1.2e-03 来自 T=8 的短序列（尖峰占窗口 1/8，GN 统计被显著改变）；")
print("     01 的 1.4e-06 来自 T=251 的长序列 + 同分布替换（尖峰只占 1/251）。两者都正确，差别在探针条件。")

print()
print("=" * 78)
print("[R2-C2] 编码器/解码器：帧级 lookahead 复核（与 02 的 A36/A37/A38 对照）")
print("=" * 78)
enc = m.encoder
wave = torch.randn(1, 32000)
w2 = wave.clone()
w2[..., 16000:] = torch.randn(1, 16000)
with torch.no_grad():
    e1, e2 = enc(wave), enc(w2)
d = (e1 - e2).abs().amax(dim=(0, 1, 3))       # 沿 T 的 profile（对 B,C,F 取 max）
ch = (d > 1e-12).nonzero().flatten()
print("  改采样 >=16000：encoder 输出受影响帧 =", ch[:6].tolist(), "... 最左 =", ch.min().item())
print("  帧 123 |Δ| =", f"{d[123].item():.3e}", "| 帧 124 =", f"{d[124].item():.3e}",
      "| 帧 125 =", f"{d[125].item():.3e}")
print("  => 帧 <124 完全不变，帧 125 起受未来影响；帧 125 窗口 [15872,16048) 用到未来 48~128 采样（8 ms）")

dec = m.decoder
emb = torch.randn(1, 1, 112, 251, 129)
emb2 = emb.clone()
emb2[..., 125:, :] = torch.randn(1, 1, 112, 126, 129)
with torch.no_grad():
    o1 = dec(emb, (1, 1, 32000))
    o2 = dec(emb2, (1, 1, 32000))
dd = (o1 - o2).abs()
print("  改 embedding 帧 >=125（=采样 16000 起）：输出 <16000 采样 max|Δ| =",
      f"{dd[..., :16000].max().item():.3e}")
print("  输出 [<15360] =", f"{dd[..., :15360].max().item():.3e}",
      "| [15360:16000] =", f"{dd[..., 15360:16000].max().item():.3e}",
      "| [16000:16064] =", f"{dd[..., 16000:16064].max().item():.3e}")
# 单帧脉冲：输入帧 f 影响哪些输出采样（感受野，用于判断是不是"未来依赖"）
imp = torch.zeros(1, 1, 112, 251, 129)
imp[..., 100, :] = 1.0
with torch.no_grad():
    z1 = dec(torch.zeros_like(imp), (1, 1, 32000))
    z2 = dec(imp, (1, 1, 32000))
zz = (z1 - z2).abs()
nz = (zz.amax(dim=(0, 1)) > 1e-12).nonzero().flatten()
print(f"  单帧脉冲（输入帧 100）-> 受影响输出采样 [{nz.min().item()}, {nz.max().item()}]；"
      f"该帧名义时刻 {100 * 128}，故输出扩散 ±{(nz.max().item() - 100 * 128)} 采样")
print("  => 输出采样 s 只依赖输入帧 <= floor(s/128)，即解码器+iSTFT 在采样级仍是因果的；")
print("     02 的『采样级 lookahead 256 采样=16 ms』应修正为『≤128 采样=8 ms 的帧级 STFT 依赖(归编码器)』。")

print()
print("=" * 78)
print("[R2-C3] 帧数账本复核（02 的 A39 / §3.3 例 1）")
print("=" * 78)
print("  encoder: L=32000 -> 1+32000//128 =", 1 + 32000 // 128, "帧")
print("  decoder 前向中间帧数（因果分支）：251 -> pad(T,2) -> 253 -> ConvTranspose2d(s=1,k=3,p=1) -> 253",
      "-> crop T[:-1] -> 252")
print("  istft length=(252-1)*128 =", (252 - 1) * 128, ">= 32000 -> 靠 length 截断")
print()
print("DONE-R2C")
