"""layers-dsp 验证脚本：STFT/iSTFT 等价性、归一化语义、因果性、层形状/参数。

运行（在仓库根）：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_layers_dsp.py

只做数值/形状/因果性验证，不加载权重、不改源码。
"""
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

torch.manual_seed(0)
torch.set_printoptions(precision=6)

SEC = lambda t: print("\n" + "=" * 78 + f"\n== {t}\n" + "=" * 78)


def maxdiff(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------- imports
SEC("[0] import look2hear.layers（顺带验证 __init__ 全量导入链）")
import look2hear.layers as L                      # noqa: E402
from look2hear.layers import stft as lay_stft     # noqa: E402
from look2hear.layers import normalizations as N  # noqa: E402
from look2hear.layers import activations as A     # noqa: E402
import look2hear.models as models                 # noqa: E402

print("look2hear.layers.stft file :", lay_stft.__file__)
mods = [m for m in ("look2hear.layers.stft", "look2hear.layers.stft_tfgn",
                    "look2hear.utils.stft", "look2hear.utils.complex_utils",
                    "look2hear.layers.enc_dec", "look2hear.layers.cnnlayers",
                    "look2hear.layers.rnnlayers", "librosa", "librosa.filters",
                    "torch_complex", "torch_complex.tensor", "typeguard",
                    "timm", "timm.models.layers") if m in sys.modules]
print("import look2hear.models 后已加载的模块:")
for m in mods:
    f = Path(sys.modules[m].__file__)
    try:
        tag = str(f.relative_to(REPO))
    except ValueError:
        tag = f"<第三方> {f.parent.name}/"
    print("   ", m, "->", tag)

cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
aconf = cfg["audionet"]["audionet_config"]
enc_params = aconf["enc_dec_params"]
WIN, HOP = enc_params["win"], enc_params["hop_length"]
print(f"config enc_dec_params: win={WIN} hop={HOP} out_chan={enc_params['out_chan']} "
      f"kernel={enc_params['kernel_size']} causal={enc_params['causal']}")

# ------------------------------------------------------- STFT equivalence
SEC("[1] forward_stft(librosa 卷积实现) vs torch.stft")
wav = torch.randn(2, 32000)
hann = torch.hann_window(WIN, periodic=True)
ref = torch.stft(wav, WIN, hop_length=HOP, window=hann, center=True,
                 return_complex=True)
ref_ri = torch.stack([ref.real, ref.imag], -1)          # [B,F,T,2]

for center in (True, False):
    got = lay_stft.forward_stft(wav, WIN, HOP, window="hann", mode="librosa",
                                onesided=True, center=center)
    c = torch.stft(wav, WIN, hop_length=HOP, window=hann, center=center,
                   return_complex=True)
    r = torch.stack([c.real, c.imag], -1)          # [B, F, T, 2]
    print(f"center={center!s:5s} shape got={tuple(got.shape)} ref={tuple(r.shape)} "
          f"max|diff|={maxdiff(got, r):.3e} dtype={got.dtype}")

# mode="torch" 路径应逐位一致
got_t = lay_stft.forward_stft(wav, WIN, HOP, window="hann", mode="torch",
                              onesided=True, center=True)
print(f"mode='torch' vs torch.stft: max|diff|={maxdiff(got_t, ref_ri):.3e}")

# FFT 点数：round_pow_of_two 对非 2 次幂 frame_len 的影响
for fl in (200, 256, 400):
    out = lay_stft.forward_stft(wav, fl, HOP, window="hann", mode="librosa",
                                round_pow_of_two=True, center=True)
    nfft = 2 ** math.ceil(math.log2(fl))
    print(f"frame_len={fl:4d} -> fft_size={nfft:4d}, bins={out.shape[1]}, "
          f"frames={out.shape[2]} (expect 1+32000//{HOP}={1 + 32000 // HOP})")

SEC("[2] inverse_stft(librosa 卷积实现) vs torch.istft + 重建误差")
x = torch.randn(2, 32000)
spec = torch.stft(x, WIN, hop_length=HOP, window=hann, center=True,
                  return_complex=True)
spec_ri = torch.stack([spec.real, spec.imag], -1)        # [B,F,T,2]
rec_lib = lay_stft.inverse_stft(spec_ri, WIN, HOP, window="hann",
                                mode="librosa", center=True)
rec_torch = torch.istft(spec, WIN, hop_length=HOP, window=hann, center=True,
                        length=32000)
print(f"librosa-iSTFT shape={tuple(rec_lib.shape)}  torch.istft shape={tuple(rec_torch.shape)}")
print(f"iSTFT 两条路径 max|diff| = {maxdiff(rec_lib, rec_torch):.3e}")
print(f"round-trip(librosa) max|diff| vs 原信号 = {maxdiff(rec_lib, x):.3e}")
print(f"round-trip(torch)   max|diff| vs 原信号 = {maxdiff(rec_torch, x):.3e}")

# center=True 的帧数/长度约定
print("num_frames 公式：1+floor(L/hop) =", 1 + 32000 // HOP,
      "| 频点数 win//2+1 =", WIN // 2 + 1)
stft_base = lay_stft.STFT(frame_len=WIN, frame_hop=HOP, window="hann",
                          center=True, mode="librosa")
print("STFTBase.num_frames(32000) =", stft_base.num_frames(torch.tensor([32000])).tolist(),
      "| num_bins =", stft_base.num_bins, "| win_length =", stft_base.win_length,
      "| K.shape =", tuple(stft_base.K.shape))

# ------------------------------------------------- 中心填充/因果性 lookahead
SEC("[3] torch.stft(center=True) 的 lookahead（编码器帧级因果性边界）")
B = 32000
a1 = torch.zeros(1, B)
a2 = a1.clone()
p = 20000
a2[0, p] = 1.0                       # 单个采样脉冲
f1 = torch.stft(a1, WIN, hop_length=HOP, window=hann, center=True, return_complex=True).abs()
f2 = torch.stft(a2, WIN, hop_length=HOP, window=hann, center=True, return_complex=True).abs()
d = (f1 - f2).abs().amax(dim=1)[0]   # 每帧能量变化
idx = torch.nonzero(d > 1e-8).flatten()
print(f"脉冲位置 p={p}: 受影响帧区间 [{idx.min().item()}, {idx.max().item()}]")
print(f"理论：帧 t 覆盖采样 [t*{HOP}-{WIN//2}, t*{HOP}+{WIN//2}) -> "
      f"t_min=ceil((p-{WIN//2-1})/{HOP})={math.ceil((p - (WIN // 2 - 1)) / HOP)}, "
      f"t_max=floor((p+{WIN//2})/{HOP})={math.floor((p + WIN // 2) / HOP)}")
print(f"=> 帧 t 依赖未来采样最多 win/2={WIN//2} 个（{WIN//2/16000*1000:.1f} ms 的帧内 lookahead）")

# --------------------------------------------------- normalizations 语义
SEC("[4] 归一化精确语义")
C, T, F = 4, 6, 3
x4 = torch.randn(1, C, T, F)


def show_norm(name, mod, shape_check=True):
    y = mod(x4)
    print(f"{name:34s} -> {tuple(y.shape)}  out_mean={y.mean().item():+.3e} "
          f"out_std={y.std(unbiased=False).item():.4f}")


print("-- 别名解析（normalizations.get）--")
for ident in ("gLN", "cLN", "LN", "LN4d", "cLNhw", "BatchNorm1d", "BatchNorm",
              "LayerNormalization4D", "BatchNorm2d", None):
    cls = N.get(ident)
    print(f"  get({ident!r:24s}) -> {cls if isinstance(cls, str) else cls.__name__}")

print("\n-- GroupNorm/GlobalLayerNorm/cLNhw 的统计维度 --")
gln = N.GlobalLayerNorm(num_channels=C, eps=1e-5)
y = gln(x4)
manual = torch.nn.GroupNorm(1, C, eps=1e-5)(x4)
print("GlobalLayerNorm 是 nn.GroupNorm(1,C):", isinstance(gln.norm, nn.GroupNorm),
      "| 与手算 GroupNorm 一致:", maxdiff(y, manual) < 1e-6)
print("  => 统计维度 = (C,H,W) =", (C, T, F), "即跨时间全窗口，pytorch GroupNorm over all non-batch dims")
m = x4.mean(dim=(1, 2, 3), keepdim=True)
v = x4.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
print("  手算 (mean over C,T,F) max|diff| =", maxdiff(y, (x4 - m) / torch.sqrt(v + 1e-5)))

cln4 = N.CumulateLN4D(dimension=C, eps=1e-8)
y = cln4(x4)
# 手算 H 维累积（H=T）
step_sum = x4.sum(1, keepdim=True)
cum = torch.cumsum(step_sum, dim=2)
cnt = torch.arange(C, C * (T + 1), C, dtype=x4.dtype).view(1, 1, -1, 1)
cm = cum / cnt
cp = torch.cumsum(x4.pow(2).sum(1, keepdim=True), dim=2)
cv = (cp - 2 * cm * cum) / cnt + cm.pow(2)
manual = (x4 - cm.expand(-1, C, -1, -1)) / torch.sqrt(cv + 1e-8).expand(-1, C, -1, -1)
print("CumulateLN4D 与手算(统计=跨 C 与 t'<=t，逐频点 f) 一致:", maxdiff(y, manual) < 1e-6)
print("  H=0 步统计只含第 0 帧；H>0 步包含 <=H 的全部帧 => 时间上是因果累积")
print("  注意：统计量按 (C, t'<=t) 计算但对每个频点 f 独立（cum_sum 保留 W 维）")

ln4d_c1 = N.LayerNormalization4D((C, 1))
mu1 = x4.mean(dim=1, keepdim=True)
sd1 = torch.sqrt(x4.var(dim=1, unbiased=False, keepdim=True) + 1e-5)
manual1 = ((x4 - mu1) / sd1) * ln4d_c1.gamma + ln4d_c1.beta
print("LayerNormalization4D((C,1)) -> dim =", ln4d_c1.dim,
      "gamma", tuple(ln4d_c1.gamma.shape),
      "| 等价 per-channel LN(dim=1):", maxdiff(ln4d_c1(x4), manual1) < 1e-6)
ln4d_cf = N.LayerNormalization4D((C, F))
mu2 = x4.mean(dim=(1, 3), keepdim=True)
sd2 = torch.sqrt(x4.var(dim=(1, 3), unbiased=False, keepdim=True) + 1e-5)
manual2 = ((x4 - mu2) / sd2) * ln4d_cf.gamma + ln4d_cf.beta
print("LayerNormalization4D((C,F)) -> dim =", ln4d_cf.dim,
      "gamma", tuple(ln4d_cf.gamma.shape),
      "| 等价 LN(dim=(1,3)):", maxdiff(ln4d_cf(x4), manual2) < 1e-6,
      "（跨 C 与 F 统计，逐帧独立）")

cln = N.cLN(C)   # CumulateLN, 输入 (B,C,T)
x3 = torch.randn(2, C, T)
y = cln(x3)
step = torch.cumsum(x3.sum(1), 1)
cnt3 = torch.arange(C, C * (T + 1), C, dtype=x3.dtype).view(1, -1)
cm3 = (step / cnt3).unsqueeze(1)
cp3 = torch.cumsum(x3.pow(2).sum(1), 1)
cv3 = ((cp3 - 2 * cm3.squeeze(1) * step) / cnt3 + cm3.squeeze(1).pow(2)).unsqueeze(1)
print("cLN=CumulateLN(输入 (B,C,T)) 与手算一致:",
      maxdiff(y, (x3 - cm3) / torch.sqrt(cv3 + 1e-8)) < 1e-6,
      "| 增益/偏置参数:", tuple(cln.gain.shape), tuple(cln.bias.shape))

print("\n-- LayerNormalizationHW(cLNhw) 复合语义 --")
hw = N.LayerNormalizationHW(num_channels=C, eps=1e-5)
y = hw(x4)
y_gn = nn.GroupNorm(1, C, eps=1e-5)(x4)
y_manual = cln4.__class__(dimension=C, eps=1e-5)(y_gn)
print("cLNhw == CumulateLN4D(GroupNorm(1,C)(x)):",
      maxdiff(y, y_manual) < 1e-6, "（先全窗口 GN，再沿时间累积 LN）")

print("\n-- BatchNorm 解析 --")
print("get('BatchNorm1d') is nn.BatchNorm1d:", N.get("BatchNorm1d") is nn.BatchNorm1d,
      "| get('BatchNorm') is 本文件包装类:", N.get("BatchNorm") is N.BatchNorm,
      "| get('BatchNorm2d') is nn.BatchNorm2d:", N.get("BatchNorm2d") is nn.BatchNorm2d)
print("=> 配置里 norm_type: BatchNorm1d 实际用 torch 原生 BatchNorm1d（非本项目包装类）")

SEC("[5] 激活解析")
for ident in (None, "PReLU", "ReLU", "Tanh", "Sigmoid", "GELU", "prelu", "relu", "tanh"):
    cls = A.get(ident)
    name = getattr(cls, "__name__", cls)
    mod = cls() if ident is not None else cls()
    print(f"  activations.get({ident!r:10s}) -> {name:12s} 实例: {mod.__class__.__name__}")
print("=> config 用 'PReLU'/'ReLU'/'Tanh'/'Sigmoid'（大小写敏感，走 hasattr(nn,...)）；"
      "小写别名 prelu/relu 走 globals()")

# ---------------------------------------------------------- 模型形状/因果性
SEC("[6] Falcon 前向形状（configs/Falcon.yml）")
aconf_full = dict(aconf)
model = getattr(models, cfg["audionet"]["audionet_name"])(
    sample_rate=16000, **aconf_full)
model.eval()
emb_params = sum(p.numel() for p in model.encoder.parameters())
print(f"STFTEncoder 参数 = {emb_params:,} | bottleneck =",
      sum(p.numel() for p in model.audio_bottleneck.parameters()),
      "| separator =", sum(p.numel() for p in model.separator.parameters()),
      "| mask_gen =", sum(p.numel() for p in model.mask_generator.parameters()),
      "| decoder =", sum(p.numel() for p in model.decoder.parameters()))
print(f"video_bottleneck 参数 = {sum(p.numel() for p in model.video_bottleneck.parameters())} "
      f"(kernel_size<0 -> conv=Identity, out_chan=in_chan)")
print("video_bottleneck.module:", model.video_bottleneck.conv.__class__.__name__)
tot_audio = sum(sum(p.numel() for p in m.parameters()) for m in
                (model.encoder, model.audio_bottleneck, model.separator,
                 model.mask_generator, model.decoder))
print(f"audio 子模块参数合计 = {tot_audio:,}")

mouth = torch.randn(1, aconf["pretrained_vout_chan"], 50)
for L_ in (32000,):
    a = torch.randn(1, L_)
    with torch.no_grad():
        e = model.encoder(a)
        b = model.audio_bottleneck(e)
        v = model.video_bottleneck(mouth)
        s = model.separator(b, v)
        msk = model.mask_generator(s, e)
        y = model.decoder(msk, a.shape)
    print(f"  input [1,{L_}] ->")
    print(f"    STFTEncoder            {tuple(e.shape)}   (T=1+L/hop={1 + L_ // HOP}, F=win/2+1={WIN // 2 + 1})")
    print(f"    audio_bottleneck(cLNhw){tuple(b.shape)}")
    print(f"    video_bottleneck       {tuple(v.shape)}")
    print(f"    Separator(AudioBlock)  {tuple(s.shape)}")
    print(f"    MaskGenerator          {tuple(msk.shape)}")
    print(f"    STFTDecoder            {tuple(y.shape)}")

SEC("[6b] STFTDecoder 内部帧数记账（T 方向）")
dec = model.decoder
with torch.no_grad():
    emb = torch.randn(1, 1, dec.in_chan, 251, 129)
    x = emb.view(1, dec.in_chan, 251, 129)
    if dec.causal:
        x = torch.nn.functional.pad(x, (0, 0, dec.kernel_size - 1, 0))
    print(f"  mask 帧数=251 -> causal pad 后 T={x.shape[-2]}")
    y = dec.decoder(x)
    print(f"  ConvTranspose2d(k={dec.kernel_size},s={dec.stride},p={dec.padding}) -> "
          f"T={y.shape[-2]}, F={y.shape[-1]}; crop[:-{dec.padding}] -> T={y.shape[-2] - dec.padding}")
    y_crop = y[:, :, :-dec.padding, :]
    print(f"  crop 后 T={y_crop.shape[-2]}（输入 mask 251 帧）")
    spec = torch.complex(y_crop[:, 0], y_crop[:, 1])
    print(f"  torch.istft 收到 {spec.shape[-2]} 帧 -> (T-1)*hop = "
          f"{(spec.shape[-2] - 1) * dec.hop_length} 采样 >= 32000；length= 截断到 32000")

SEC("[7] 因果性探测 1：扰动后半段音频，看各层输出哪些部分变化")
B, T = 1, 32000
torch.manual_seed(1)
a1 = torch.randn(B, T)
a2 = a1.clone()
a2[..., T // 2:] = 10.0 * a1[..., T // 2:]      # 后半段整体放大 10 倍：显著改变全窗口统计量
split_t = 120   # 帧索引：采样 16000 对应帧 16000/128=125，取 120 保守


def probe(name, f1, f2, split):
    """4D 张量 [B,C,T,F] 按 T 切片；3D/2D 按最后一维切片。"""
    d = (f1 - f2).abs()
    if d.dim() == 4:
        early, late = d[..., :split, :].max().item(), d[..., split:, :].max().item()
        tag = f"T<{split} / T>={split}"
    else:
        early, late = d[..., :split].max().item(), d[..., split:].max().item()
        tag = f"[...,:{split}] / [...,{split}:]"
    print(f"  {name:24s} {tag:22s} early max|d|={early:.3e}  late max|d|={late:.3e}")


with torch.no_grad():
    e1, e2 = model.encoder(a1), model.encoder(a2)
    probe("STFTEncoder", e1, e2, split_t)
    b1, b2 = model.audio_bottleneck(e1), model.audio_bottleneck(e2)
    probe("audio_bottleneck", b1, b2, split_t)
    v = model.video_bottleneck(mouth)
    s1, s2 = model.separator(b1, v), model.separator(b2, v)
    probe("Separator", s1, s2, split_t)
    m1, m2 = model.mask_generator(s1, e1), model.mask_generator(s2, e2)
    probe("MaskGenerator", m1, m2, split_t)
    y1, y2 = model.decoder(m1, a1.shape), model.decoder(m2, a2.shape)
    print("  decoder 输出按采样段:")
    for lo, hi in ((0, 8000), (8000, 16000), (16000, 24000), (24000, 32000)):
        print(f"     samples[{lo:>5}:{hi:>5}] max|d|={(y1[..., lo:hi] - y2[..., lo:hi]).abs().max().item():.3e}")

SEC("[7b] 因果性探测 1b：STFTDecoder 自身的时间因果性")
with torch.no_grad():
    mk1 = torch.randn(1, 1, 112, 251, 129)
    mk2 = mk1.clone()
    mk2[..., 120:, :] = torch.randn(1, 1, 112, 131, 129)
    q1 = model.decoder(mk1, torch.Size([1, 32000]))
    q2 = model.decoder(mk2, torch.Size([1, 32000]))
for lo, hi in ((0, 8000), (8000, 15360), (15360, 16000), (16000, 24000)):
    print(f"  decoder samples[{lo:>5}:{hi:>5}] max|d|="
          f"{(q1[..., lo:hi] - q2[..., lo:hi]).abs().max().item():.3e}")
print("  => center=True 的 iSTFT 使帧级扰动扩散 ±win/2 采样；解码器卷积本身为 causal 裁剪")

SEC("[8] 因果性探测 2：逐层子模块（DualPathRNN / MHSA2D / cLNhw）")
from look2hear.models.Falcon import DualPathRNN, MultiHeadSelfAttention2D  # noqa: E402
hid = aconf["audio_params"]["hid_chan"]
l1cfgw = dict(aconf["audio_params"]["layers"]["layer_1"])
l2cfgw = dict(aconf["audio_params"]["layers"]["layer_2"])
l3cfgw = dict(aconf["audio_params"]["layers"]["layer_3"])
l3cfgw["n_freqs"] = 65
dprnn_f = DualPathRNN(in_chan=hid, **l1cfgw)   # dim=4 频轴 双向
dprnn_t = DualPathRNN(in_chan=hid, causal=True, **l2cfgw)  # dim=3 时间轴 单向
mhsa = MultiHeadSelfAttention2D(in_chan=hid, **l3cfgw)

z1 = torch.randn(1, hid, 8, 65)     # [B,C,T,F]
z2 = z1.clone()
z2[:, :, 5:, :] = torch.randn(1, hid, 3, 65)   # 扰动 t>=5

for name, mod in (("DualPathRNN dim=4", dprnn_f), ("DualPathRNN dim=3", dprnn_t),
                  ("MHSA2D causal=True", mhsa)):
    with torch.no_grad():
        o1, o2 = mod(z1), mod(z2)
    d = (o1 - o2).abs()
    print(f"  {name:20s} out{tuple(o1.shape)} early t<5 max|d|={d[:, :, :5].max().item():.3e} "
          f"late t>=5 max|d|={d[:, :, 5:].max().item():.3e}")
mhsa_nc = MultiHeadSelfAttention2D(in_chan=hid, **{**l3cfgw, "causal": False})
with torch.no_grad():
    o1, o2 = mhsa_nc(z1), mhsa_nc(z2)
d = (o1 - o2).abs()
print(f"  {'MHSA2D causal=False':20s} out{tuple(o1.shape)} early t<5 max|d|={d[:, :, :5].max().item():.3e} "
      f"late t>=5 max|d|={d[:, :, 5:].max().item():.3e}")

# cLNhw 的时间因果性：只改单帧（t=6，尖峰），看 t<5 是否变化
hw = N.LayerNormalizationHW(num_channels=hid, eps=1e-5)
z3 = z1.clone()
z3[:, :, 6, :] += 20.0          # 单帧尖峰：显著改变 GroupNorm(1,C) 的全窗口统计量
with torch.no_grad():
    o1, o2 = hw(z1), hw(z3)
d = (o1 - o2).abs()
print(f"  {'cLNhw(LayerNormHW)':20s} out{tuple(o1.shape)} early t<5 max|d|={d[:, :, :5].max().item():.3e} "
      f"late t>=5 max|d|={d[:, :, 5:].max().item():.3e}")
print("  => GroupNorm(1,C) 跨 T 统计 => t<5 的输出也变化：cLNhw 不是时间因果的（但其后的 CumulateLN4D 是）")

SEC("[9] DualPathRNN 参数化/out_proj/Comba 默认值")
print("dim=4 config:", l1cfgw, "| out_proj:", dprnn_f.out_proj)
print("dim=3 config:", l2cfgw, "| out_proj:", dprnn_t.out_proj)
print("  rnn:", dprnn_f.rnn.input_size, dprnn_f.rnn.hidden_size, dprnn_f.rnn.bidirectional,
      "| num_direction:", dprnn_f.num_direction)
try:
    DualPathRNN(in_chan=8, hid_chan=8, dim=3)     # 默认 rnn_type="Comba"
    print("  默认 rnn_type='Comba' 构造成功")
except Exception as exc:
    print(f"  默认 rnn_type='Comba' 构造失败: {type(exc).__name__}: {exc}")

SEC("[10] Separator 会 pop 配置字典（studio deep-copy 的原因）")
import copy as _copy
_cfg2 = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())["audionet"]["audionet_config"]
d_audio = _copy.deepcopy(_cfg2["audio_params"]); d_audio.pop("causal", None)
d_video = _copy.deepcopy(_cfg2["video_params"]); d_video.pop("causal", None)
print("  [fresh yaml] audio_params 原始 keys:", sorted(d_audio))
print("  [fresh yaml] video_params 原始 keys:", sorted(d_video))
before = (dict(d_audio), dict(d_video))
from look2hear.models.Falcon import Separator  # noqa: E402
sep = Separator(audio_params=d_audio, video_params=d_video,
                audio_bn_chan=112, video_bn_chan=512, fusion_params={})
print("  audio_params keys before:", sorted(before[0]))
print("  audio_params keys after :", sorted(d_audio))
print("  video_params keys after :", sorted(d_video))
print("  => pop('repeats'/'shared') 破坏调用方配置；studio/falcon_runtime.py 用 deep-copy 规避")

SEC("[11] layers/enc_dec.py（Asteroid filterbank）是否被 Falcon 使用")
print("  models.Falcon 的 encoder 类型:", type(model.encoder).__name__, "| module:",
      type(model.encoder).__module__)
print("  decoder:", type(model.decoder).__name__, "|", type(model.decoder).__module__)
print("  => Falcon 用的是 models/Falcon.py 内部定义的 STFTEncoder/STFTDecoder（torch.stft），")
print("     不是 layers/enc_dec.py 的 Encoder/Decoder/FreeFB（后者全仓无调用方）")

print("\nDONE")
