"""model-core 补充验证 4：定量 lookahead（严格/非严格因果的界）、视频分支因果性。

运行：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify4.py
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
from look2hear.models.Falcon import AudioBlock  # noqa: E402
from look2hear.videomodels import FRCNNVideoModel  # noqa: E402

torch.manual_seed(0)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
m = getattr(models, CFG["audionet"]["audionet_name"])(
    sample_rate=16000, **copy.deepcopy(ACONF)).eval()
sep, ab = m.separator, m.separator.audio_net
HOP = ACONF["enc_dec_params"]["hop_length"]

print("=" * 78)
print("[O] 严格因果 vs 有界 lookahead（按输入分界，给出受影响的最左输出索引）")
print("=" * 78)


def prefix_profile(fn, x, in_dim, out_dim):
    """只改 x[in_dim] >= k 的后缀，返回 (输出 profile, k, 前缀中最靠左的受影响索引)。"""
    n = x.size(in_dim)
    k = n // 2
    idx = [slice(None)] * x.dim()
    idx[in_dim] = slice(k, None)
    x1, x2 = x.clone(), x.clone()
    x2[tuple(idx)] = torch.randn_like(x2[tuple(idx)])
    with torch.no_grad():
        y1, y2 = fn(x1), fn(x2)
    d = (y1 - y2).abs()
    od = out_dim % d.dim()
    prof = d.amax(dim=tuple(i for i in range(d.dim()) if i != od))
    ch = (prof > 1e-12).nonzero().flatten()
    first = ch.min().item() if ch.numel() else None
    return n, k, prof, first


# --- 1) STFTEncoder：样本分隔 -> 帧
n, k, prof, first = prefix_profile(lambda t: m.encoder(t), torch.randn(1, 32000), -1, -2)
print(f"  STFTEncoder: 输入 {n} 样本, 分界 {k} (帧 {k // HOP}), 输出 {prof.shape[0]} 帧")
print(f"    最左受影响帧 = {first}; 逐帧 |Δ| 在分界附近:",
      [round(prof[i].item(), 3) for i in range(k // HOP - 2, k // HOP + 3)])
print("    结论: 帧 f 的窗口 = [f*hop-win/2, f*hop+win/2)，帧 125 需要样本 16048-1 -> "
      "lookahead = win/2 = 128 样本 = 8 ms")

# --- 2) STFTDecoder：帧分隔 -> 样本
n, k, prof, first = prefix_profile(
    lambda t: m.decoder(t, (1, 1, 32000)), torch.randn(1, 1, 112, 251, 129), -2, -1)
print(f"  STFTDecoder: 输入 {n} 帧, 分界 {k} (={k * HOP} 样本), 输出 {prof.shape[0]} 样本")
print(f"    前缀(样本<{k * HOP})最大 |Δ| = {prof[:k * HOP].max().item():.3e} -> "
      f"{'严格因果' if prof[:k * HOP].max().item() == 0 else '有泄漏'}")
print(f"    最左受影响输出样本索引 = {first} (应 >= {k * HOP})")

# --- 3) AudioBlock（关掉 cLNhw 归一化后，纯结构因果）
lay = copy.deepcopy(ACONF["audio_params"]["layers"])
ab_nn = AudioBlock(in_chan=112, hid_chan=64, kernel_size=5, stride=2, norm_type=None,
                   act_type="PReLU", upsampling_depth=4, layers=lay, is2d=True,
                   causal=True, n_freqs=129).eval()
x_ab = torch.randn(1, 112, 251, 129)
for tag, mod in (("AudioBlock(norm 全关)", ab_nn), ("AudioBlock(实际 cLNhw)", ab),
                 ("audio_bottleneck(cLNhw)", m.audio_bottleneck)):
    n, k, prof, first = prefix_profile(lambda t, mo=mod: mo(t), x_ab, -2, -2)
    print(f"  {tag:26s}: 分界 {k} 帧, 前缀最左受影响帧 = {first}, "
          f"lookahead = {k - first if first is not None else 0} 帧 = "
          f"{(k - first) * HOP if first is not None else 0} 样本")
print("    理论: 每级对称 k=5 卷积 ±2 帧；level0..3 换算到 level0 为 ±2,±4,±8,±16 -> 合计 ~30 帧 ≈ 240 ms")

# --- 4) 端到端：改未来音频，前 1s 输出逐 0.25s 段的变化（Lead 的 2.97e-4 复核）
audio = torch.randn(1, 32000)
mouth = torch.randn(1, ACONF["pretrained_vout_chan"], 50)
a2 = audio.clone()
a2[..., 16000:] = torch.randn(1, 16000)
with torch.no_grad():
    y1, y2 = m(audio, mouth), m(a2, mouth)
d = (y1 - y2).abs()
print("  端到端 2s 窗口：改 1.0-2.0s 音频后，前 1s 输出的 |Δ|:")
for lo in range(0, 16000, 2000):
    print(f"    [{lo:5d}:{lo + 2000:5d}] ({lo / 16000:.2f}-{(lo + 2000) / 16000:.2f}s) "
          f"max|Δ| = {d[..., lo:lo + 2000].max().item():.3e}")

print()
print("=" * 78)
print("[P] 视频分支因果性")
print("=" * 78)
frcnn = FRCNNVideoModel(pretrain=None)
frcnn.eval()          # 注意: FRCNNVideoModel.train() 不 return self，所以不能链式 .eval()
vid = torch.randn(1, 1, 50, 96, 96)
vid2 = vid.clone()
vid2[:, :, 30:] = torch.randn(1, 1, 20, 96, 96)
with torch.no_grad():
    f1, f2 = frcnn(vid), frcnn(vid2)
print("  FRCNN 输出:", tuple(f1.shape), "= (B, 512, T_v), T_v=50 (2s@25fps)")
d = (f1 - f2).abs().amax(dim=(0, 1))
nz = (d > 1e-12).nonzero().flatten()
print("  改视频帧>=30 -> 输出受影响帧:", nz.tolist()[:8], "最左 =", nz.min().item())
print("  前端 Conv3d kernel=(5,7,7) padding=(2,3,3) -> 时间感受野 ±2 帧 = ±80 ms")
vb = sep.video_net
v_in = torch.randn(1, 512, 50)
for mode in ("eval", "train"):
    getattr(vb, mode)()
    v2 = v_in.clone()
    v2[:, :, 25:] = torch.randn(1, 512, 25)
    with torch.no_grad():
        o1, o2 = vb(v_in), vb(v2)
    dd = (o1 - o2).abs().amax(dim=(0, 1))
    ch = (dd[:25] > 1e-12).nonzero().flatten()
    print(f"  VideoBlock({mode}) 改帧>=25 -> 前 25 帧受影响 max|Δ| = "
          f"{dd[:25].max().item():.3e}, 最左受影响帧 = "
          f"{ch.min().item() if ch.numel() else 'none'}")
print("  说明: eval 下 BN1d 用 running stats -> 只由单向 LSTM 决定因果; "
      "train 下 BN1d 用批内(含未来帧)统计 -> 非因果")

print()
print("=" * 78)
print("[Q] Falcon(causal=True/False) 是否真的改变行为？")
print("=" * 78)
base_kw = {k: v for k, v in copy.deepcopy(ACONF).items() if k != "spk_embed_dim"}
mt = getattr(models, CFG["audionet"]["audionet_name"])(
    sample_rate=16000, **{**base_kw, "causal": True})
mf = getattr(models, CFG["audionet"]["audionet_name"])(
    sample_rate=16000, **{**base_kw, "causal": False})
print("  param 数:", sum(p.numel() for p in mt.parameters()),
      "vs", sum(p.numel() for p in mf.parameters()))
print("  state_dict key 完全一致:",
      list(mt.state_dict().keys()) == list(mf.state_dict().keys()))
# 权重相同 -> 输出是否一致（eval）
mf.load_state_dict(mt.state_dict())
mt.eval(); mf.eval()
with torch.no_grad():
    o1 = mt(torch.randn(1, 32000), torch.randn(1, 512, 50))
    o2 = mf(torch.randn(1, 32000), torch.randn(1, 512, 50))
print("  仅顶层 causal 不同（其余 config 相同）时输出是否 bit-identical:",
      torch.equal(o1, o2) if o1.shape == o2.shape else "shape mismatch")
print("  解释: Falcon.causal 只写入 audio_params/video_params；")
print("        AudioBlock.causal 仅传给 DualPathRNN(causal=...) 而 DualPathRNN 从不读取 self.causal；")
print("        VideoBlock.causal 从不读取；ConvNormAct 的 causal 在 AudioBlock 内硬编码 False；")
print("        真正生效的是 enc_dec_params.causal=True 与 layer_3.causal=True。")
print()
print("DONE4")
