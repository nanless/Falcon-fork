"""model-core 独立验证脚本：构造链 / 参数量拆解 / 逐级形状 / 逐组件因果性 / MDCL 数学 / 死代码。

运行（仓库根）：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify.py

只读项目代码、不改源码；全部输入为随机张量，不加载权重。
输出：stdout（同时重定向到 docs/verification/runs/model-core-verify.log）
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
import look2hear.models.Falcon as _falcon_mod  # noqa: E402  (实际被同名类遮蔽，见 [F])
from look2hear.models.Falcon import AudioBlock, ConvNormAct, InjectionMultiSum, MultiHeadSelfAttention2D  # noqa: E402

torch.manual_seed(0)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
NAME = CFG["audionet"]["audionet_name"]


def nparam(m):
    return sum(p.numel() for p in m.parameters())


print("=" * 78)
print("[A] 构造链与配置键消耗")
print("=" * 78)
raw = copy.deepcopy(ACONF)
m = getattr(models, NAME)(sample_rate=16000, **copy.deepcopy(ACONF))
m.eval()
print("audionet_name =", NAME)
print("config keys(depth1) =", list(raw.keys()))
print("构造后 Falcon.audio_params keys =", sorted(m.audio_params.keys()))
print("  -> 'repeats' still in model.audio_params? ", "repeats" in m.audio_params)
print("  -> 'shared'  still in model.audio_params? ", "shared" in m.audio_params)
print("  -> 'causal' in model.audio_params?       ", m.audio_params.get("causal"))
print("  -> 'repeats' still in model.video_params?", "repeats" in m.video_params)
print("separator.fusion_repeats / audio_repeats =",
      m.separator.fusion_repeats, m.separator.audio_repeats)
print("separator.audio_shared/video_shared/fusion_shared =",
      m.separator.audio_shared, m.separator.video_shared, m.separator.fusion_shared)
print("audio_net type =", type(m.separator.audio_net).__name__,
      "| video_net type =", type(m.separator.video_net).__name__,
      "| caf type =", type(m.separator.caf).__name__)
print("mdcl_margin/mdcl_alpha =", m.mdcl_margin, m.mdcl_alpha)
print("spk_embed_dim =", m.spk_embed_dim, "| n_src =", m.n_src, "| causal =", m.causal)
print("传入的 config dict 是否被构造函数 pop 改写:", "repeats" not in raw["audio_params"],
      "(True=被 pop 掉，同一 dict 不能复用)")

print()
print("=" * 78)
print("[B] 参数总账与逐层拆解")
print("=" * 78)
print(f"audio_model total={nparam(m):,}  trainable="
      f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
for name, ch in m.named_children():
    print(f"  {name:18s} {nparam(ch):>10,}")
print("  --- separator 内部 ---")
sep = m.separator
for name, ch in sep.named_children():
    print(f"    separator.{name:14s} {nparam(ch):>9,}")
ab = sep.audio_net
print("  --- AudioBlock 内部（单实例，权重被复用两次） ---")
for name, ch in ab.named_children():
    print(f"    audio_net.{name:18s} {nparam(ch):>9,}")
for i in range(ab.upsampling_depth):
    print(f"    audio_net.downsample_layers[{i}]      {nparam(ab.downsample_layers[i]):>9,}")
    print(f"    audio_net.process_f_layers[{i}]       {nparam(ab.process_f_layers[i]):>9,}"
          f"  (DualPathRNN dim={ab.process_f_layers[i].dim},"
          f" bidir={ab.process_f_layers[i].bidirectional})")
    print(f"    audio_net.process_t_layers[{i}]       {nparam(ab.process_t_layers[i]):>9,}"
          f"  (DualPathRNN dim={ab.process_t_layers[i].dim},"
          f" bidir={ab.process_t_layers[i].bidirectional})")
    print(f"    audio_net.process_att_layers[{i}]     {nparam(ab.process_att_layers[i]):>9,}"
          f"  (n_freqs={ab.process_att_layers[i].n_freqs},"
          f" n_head={ab.process_att_layers[i].n_head}, causal={ab.process_att_layers[i].causal})")
    print(f"    audio_net.fusion_layers[{i}]          {nparam(ab.fusion_layers[i]):>9,}")
    print(f"    audio_net.concat_layers[{i}]          {nparam(ab.concat_layers[i]):>9,}")
vb = sep.video_net
print("  --- VideoBlock 内部 ---")
for name, ch in vb.named_children():
    print(f"    video_net.{name:18s} {nparam(ch):>9,}")
print("  --- caf ---")
for name, ch in sep.caf.named_children():
    print(f"    caf.{name:20s} {nparam(ch):>9,}")
print("  --- mask_generator / decoder / spk_pool_proj / encoder ---")
for name, ch in m.mask_generator.named_children():
    print(f"    mask_generator.{name:16s} {nparam(ch):>9,}")
for name, ch in m.decoder.named_children():
    print(f"    decoder.{name:20s} {nparam(ch):>9,}")
for name, ch in m.spk_pool_proj.named_children():
    print(f"    spk_pool_proj.{name:15s} {nparam(ch):>9,}")
for name, ch in m.encoder.named_children():
    print(f"    encoder.{name:20s} {nparam(ch):>9,}")

from collections import Counter
cnt = Counter()
for _, mod in m.named_modules():
    n = nparam(mod) if len(list(mod.children())) == 0 else 0
    if n:
        cnt[type(mod).__name__] += n
print("  --- 按叶子模块类统计参数量 ---")
for k, v in cnt.most_common():
    print(f"    {k:26s} {v:>9,}")
print("  --- Falcon.py 中定义但未出现在模型里的类（死代码） ---")
mod = sys.modules["look2hear.models.Falcon"]
defined = [n for n, o in vars(mod).items()
           if isinstance(o, type) and issubclass(o, torch.nn.Module)]
used = {type(x).__name__ for _, x in m.named_modules()}
print("    defined:", sorted(set(defined)))
print("    unused :", sorted(set(defined) - used))

print()
print("=" * 78)
print("[C] 输入/输出形状与单位（L=32000=2s@16k, mouth_emb=[B,512,50]）")
print("=" * 78)
B = 1
audio = torch.randn(B, 32000)
mouth = torch.randn(B, ACONF["pretrained_vout_chan"], 50)
m.train()
with torch.no_grad():
    est, emb = m(audio, mouth)
print("forward(train) ->", tuple(est.shape), tuple(emb.shape))
m.eval()
with torch.no_grad():
    out = m(audio, mouth)
print("forward(eval)  ->", type(out).__name__, tuple(out.shape))
print("T' = 1 + floor(L/hop) =", 1 + 32000 // 128, "| F = win/2+1 =", 256 // 2 + 1)

shapes, seen = [], set()
hooks = []
def mk(tag):
    def f(modu, inp, outp):
        if tag not in seen:
            seen.add(tag)
            shapes.append((tag, tuple(inp[0].shape), tuple(outp.shape)))
    return f
with torch.no_grad():
    e = m.encoder(audio)
    a_bn = m.audio_bottleneck(e)
    v_bn = m.video_bottleneck(mouth)
for i in range(ab.upsampling_depth):
    for tag, modu in ((f"down[{i}]", ab.downsample_layers[i]),
                      (f"proc_f[{i}]", ab.process_f_layers[i]),
                      (f"proc_t[{i}]", ab.process_t_layers[i]),
                      (f"proc_att[{i}]", ab.process_att_layers[i]),
                      (f"fusion[{i}]", ab.fusion_layers[i]),
                      (f"concat[{i}]", ab.concat_layers[i])):
        hooks.append(modu.register_forward_hook(mk(tag)))
with torch.no_grad():
    s = m.separator(a_bn, v_bn)
for h in hooks:
    h.remove()
with torch.no_grad():
    se = m.mask_generator(s, e)
    y = m.decoder(se, audio.shape)
print("encoder          ->", tuple(e.shape))
print("audio_bottleneck ->", tuple(a_bn.shape))
print("video_bottleneck ->", tuple(v_bn.shape), "params=", nparam(m.video_bottleneck))
for tag, i_shape, o_shape in shapes:
    print(f"  {tag:12s} in={i_shape} out={o_shape}")
print("separator out  ->", tuple(s.shape))
print("mask_gen out   ->", tuple(se.shape))
print("decoder out    ->", tuple(y.shape))
print("_encode_audio  ->", tuple(m._encode_audio(audio).shape))
print("AudioBlock 输入 T'=", e.shape[-2], "video T=", v_bn.shape[-1],
      "-> NeuroSCNetFusion nearest 插值比 =", e.shape[-2] / v_bn.shape[-1])

print()
print("=" * 78)
print("[D] 因果性：前缀扰动探测（改后半段输入 -> 左段输出是否变化）")
print("=" * 78)


def probe(fn, x, in_dim, out_dim, split_frac=0.5, label=""):
    """把 x 沿 in_dim 的后半段替换为噪声；分别看输出左段/右段变化。
    lookahead = 输出左段中仍被改动的最大越界量（输出单位：帧或采样点）。"""
    n = x.size(in_dim)
    k = int(n * split_frac)
    idx = [slice(None)] * x.dim()
    idx[in_dim] = slice(k, None)
    x1, x2 = x.clone(), x.clone()
    x2[tuple(idx)] = torch.randn_like(x2[tuple(idx)])
    with torch.no_grad():
        y1, y2 = fn(x1), fn(x2)
    d = (y1 - y2).abs()
    od = out_dim % d.dim()
    prof = d.amax(dim=tuple(i for i in range(d.dim()) if i != od))
    b = max(1, int(round(k * prof.shape[0] / n)))
    pre = prof[:b].max().item()
    suf = prof[b:].max().item() if prof.shape[0] > b else float("nan")
    ch = (prof[:b] > 1e-12).nonzero().flatten()
    look = (b - 1 - ch.max().item()) if ch.numel() else 0
    print(f"  {label:32s} 左段[0:{b}]={pre:.3e} 右段={suf:.3e} | 左段越界={look} 单位")


print("-- 波形/帧边界（前 1s 输入 vs 后 1s 输入）--")
probe(lambda t: m.encoder(t), torch.randn(B, 32000), -1, -2, label="STFTEncoder(样本->帧)")
probe(lambda t: m.decoder(t, (B, 1, 32000)), torch.randn(B, 1, 112, 251, 129), -2, -1,
      label="STFTDecoder(帧->样本)")
print("-- cLNhw 归一化是否为泄漏源 --")
probe(lambda t: m.audio_bottleneck(t), a_bn, -2, -2, label="audio_bottleneck(cLNhw)")
probe(lambda t: m.audio_bottleneck.pre_norm(t), a_bn, -2, -2, label="仅 pre_norm=cLNhw")
conv_only = ConvNormAct(in_chan=112, out_chan=112, kernel_size=1, is2d=True,
                        act_type=None, norm_type=None, pre_norm_type=None,
                        pre_act_type=None).eval()
probe(lambda t: conv_only(t), a_bn, -2, -2, label="去掉 cLNhw 的 1x1 conv")
print("-- 其余组件 --")
probe(lambda t: ab(t), a_bn, -2, -2, label="AudioBlock(完整)")
lay = copy.deepcopy(ACONF["audio_params"]["layers"])
ab_nonorm = AudioBlock(in_chan=112, hid_chan=64, kernel_size=5, stride=2,
                       norm_type=None, act_type="PReLU", upsampling_depth=4,
                       layers=lay, is2d=True, causal=True, n_freqs=129).eval()
probe(lambda t: ab_nonorm(t), a_bn, -2, -2, label="AudioBlock(norm 全关)")
probe(lambda t: ab.process_f_layers[3](t), torch.randn(B, 64, 32, 17), -2, -2,
      label="DualPathRNN dim=4 频域路径")
probe(lambda t: ab.process_t_layers[3](t), torch.randn(B, 64, 32, 17), -2, -2,
      label="DualPathRNN dim=3 时域路径(单向)")
probe(lambda t: ab.process_att_layers[3](t), torch.randn(B, 64, 32, 17), -2, -2,
      label="MHSA2D(causal=True)")
att_nc = MultiHeadSelfAttention2D(in_chan=64, n_freqs=17, n_head=4, hid_chan=4,
                                  act_type="PReLU", norm_type="LayerNormalization4D",
                                  dim=3, causal=False).eval()
probe(lambda t: att_nc(t), torch.randn(B, 64, 32, 17), -2, -2, label="MHSA2D(causal=False)")
ims_nc = InjectionMultiSum(in_chan=64, kernel_size=5, norm_type=None, is2d=True,
                           causal=False).eval()
ims_c = InjectionMultiSum(in_chan=64, kernel_size=5, norm_type=None, is2d=True,
                          causal=True).eval()
x5 = torch.randn(B, 64, 251, 129)
probe(lambda t: ims_nc(t, t), x5, -2, -2, label="InjectionMultiSum(causal=False)")
probe(lambda t: ims_c(t, t), x5, -2, -2, label="InjectionMultiSum(causal=True)")
probe(lambda t: m.mask_generator(t, e), torch.randn(B, 112, 251, 129), -2, -2,
      label="MaskGenerator")
probe(lambda t: sep.caf(t, v_bn), a_bn, -2, -2, label="NeuroSCNetFusion")

print()
print("-- 脉冲响应：输入单帧/单点非零，输出受影响范围（感受野） --")
def impulse(fn, shape, t0, out_dim, dim=-2, tag=""):
    x1, x2 = torch.zeros(shape), torch.zeros(shape)
    idx = [slice(None)] * len(shape)
    idx[dim] = t0
    x2[tuple(idx)] = 1.0
    with torch.no_grad():
        y1, y2 = fn(x1), fn(x2)
    d = (y1 - y2).abs()
    od = out_dim % d.dim()
    prof = d.amax(dim=tuple(i for i in range(d.dim()) if i != od))
    nz = (prof > 1e-12).nonzero().flatten()
    if nz.numel() == 0:
        print(f"  {tag:32s} 无响应")
        return
    print(f"  {tag:32s} t0={t0} -> 输出[{nz.min().item()},{nz.max().item()}], "
          f"未来+{nz.max().item() - t0}, 过去-{t0 - nz.min().item()}")

impulse(lambda t: m.encoder(t), (1, 32000), 8000, out_dim=-2, dim=-1, tag="STFTEncoder(样本)")
impulse(lambda t: m.decoder(t, (1, 1, 32000)), (1, 1, 112, 251, 129), 100, out_dim=-1,
        dim=-2, tag="STFTDecoder(帧)")
impulse(lambda t: m.audio_bottleneck.pre_norm(t), (1, 112, 251, 129), 100, out_dim=-2,
        tag="cLNhw pre_norm")
impulse(lambda t: conv_only(t), (1, 112, 251, 129), 100, out_dim=-2, tag="1x1 conv")
impulse(lambda t: ab.downsample_layers[0](t), (1, 64, 251, 129), 100, out_dim=-2,
        tag="AudioBlock.down[0](stride1)")
impulse(lambda t: ab.downsample_layers[3](t), (1, 64, 63, 33), 30, out_dim=-2,
        tag="AudioBlock.down[3](stride2)")
impulse(lambda t: ims_nc(t, t), (1, 64, 251, 129), 100, out_dim=-2,
        tag="InjectionMultiSum(causal=False)")
impulse(lambda t: ims_c(t, t), (1, 64, 251, 129), 100, out_dim=-2,
        tag="InjectionMultiSum(causal=True)")
impulse(lambda t: ab.process_t_layers[3](t), (1, 64, 32, 17), 16, out_dim=-2,
        tag="DualPathRNN 时域(单向)")
impulse(lambda t: ab.process_att_layers[3](t), (1, 64, 32, 17), 16, out_dim=-2,
        tag="MHSA2D(causal=True)")
print("  参考: hop=128 样本/帧, win=256；2s 窗口 251 帧；深层 T=251->126->63->32")

print()
print("=" * 78)
print("[E] MDCL 数学验证 (_sisnr/_mag_sim/_mdcl_loss_hard)")
print("=" * 78)
r = torch.randn(2, 16000)
n0 = torch.randn(2, 16000)
n0 = n0 - (n0 * r).sum(-1, keepdim=True) / (r * r).sum(-1, keepdim=True) * r
n0 = n0 / n0.norm(dim=-1, keepdim=True) * (0.1 * r.norm(dim=-1, keepdim=True))
print("  _sisnr(est=ref)      =", [round(v, 3) for v in m._sisnr(r, r).tolist()])
print("  _sisnr(SNR=20dB)     =", [round(v, 3) for v in m._sisnr(r + n0, r).tolist()], "(期望 ≈20)")
print("  _mag_sim(est=ref)    =", [round(v, 3) for v in m._mag_sim(r, r).tolist()])
print("  _mag_sim(SNR=20dB)   =", [round(v, 3) for v in m._mag_sim(r + n0, r).tolist()])
est = torch.randn(2, 16000, requires_grad=True)
K0 = m._mdcl_loss_hard(est, [])
print("  K=0 ->", tuple(K0.shape), K0.item())
K1 = m._mdcl_loss_hard(est, [r], beta=1.0)
manual1 = torch.relu(m._mag_sim(est, r) - m.mdcl_margin).mean()
print("  K=1 ->", f"{K1.item():.6f}", "| 手算 =", f"{manual1.item():.6f}",
      "| allclose:", torch.allclose(K1, manual1))
near = [r, torch.randn(2, 16000), torch.randn(2, 16000)]
for beta in (0.0, 1.0, 20.0):
    L = m._mdcl_loss_hard(est, near, beta=beta)
    sims = torch.stack([m._mag_sim(est, it) for it in near], 0)
    w = torch.softmax(beta * sims, 0)
    manual = (w * torch.relu(sims - m.mdcl_margin)).sum(0).mean()
    print(f"  K=3 beta={beta:<5} L={L.item():.6f} manual={manual.item():.6f} "
          f"allclose={torch.allclose(L, manual)} weights[:,0]={[round(v, 4) for v in w[:, 0].tolist()]}")
L = m._mdcl_loss_hard(est, near, beta=1.0)
L.backward()
print("  grad 存在:", est.grad is not None, "| |grad| =", f"{est.grad.abs().sum().item():.4e}")
print("  mdcl_margin(默认, config 未给) =", m.mdcl_margin, "| mdcl_alpha =", m.mdcl_alpha)

print()
print("=" * 78)
print("[F] 死代码 / 不一致 / 抗误用")
print("=" * 78)
print("  Falcon._mdcl_loss_hard 存在:", hasattr(m, "_mdcl_loss_hard"),
      "| Falcon._mdcl_loss 存在:", hasattr(m, "_mdcl_loss"))
base = models.Falcon(sample_rate=16000,
                     **{k: v for k, v in copy.deepcopy(ACONF).items() if k != "spk_embed_dim"})
base.train()
try:
    base(torch.randn(1, 32000), torch.randn(1, 512, 50), near_audio=torch.randn(1, 32000))
    print("  Falcon(train, near_audio) -> 未抛错（与源码推断不符！）")
except AttributeError as ex:
    print("  Falcon(train, near_audio) -> AttributeError:", ex)
print("  FalconSpeakerContrastive.train() 返回 tuple:", isinstance(est, tuple))
try:
    models.Falcon(sample_rate=16000, **copy.deepcopy(ACONF))
except TypeError as ex:
    print("  configs/Falcon.yml 的 audionet_config 直接喂 Falcon -> TypeError:", ex)
from look2hear.models import get as mget
for key in ("Falcon", "falconspeakercontrastive", "AVTIGER_causal"):
    try:
        print(f"  models.get({key!r}) ->", mget(key).__name__)
    except ValueError as ex:
        print(f"  models.get({key!r}) -> ValueError:", ex)
try:
    exec("from look2hear.models import *", {})
    print("  from look2hear.models import * -> OK")
except Exception as ex:
    print("  from look2hear.models import * ->", type(ex).__name__, ex)
print("  look2hear.models.Falcon 属性指向:", _falcon_mod)

print()
print("-- n_src=2 抗误用（STFTDecoder.view 假设 in_chan=n_src*out_chan）--")
try:
    m2 = getattr(models, NAME)(sample_rate=16000, **{**copy.deepcopy(ACONF), "n_src": 2})
    m2.eval()
    with torch.no_grad():
        o2 = m2(audio, mouth)
    print("  n_src=2 forward ->", type(o2))
except Exception as ex:
    print("  n_src=2 forward ->", type(ex).__name__, str(ex)[:220])

print()
print("-- 未被使用的配置键（video_params.layers / fusion_params 其余键）--")
print("  video_params keys =", sorted(m.video_params.keys()),
      "| layer_1 =", m.video_params.get("layers", {}).get("layer_1"))
print("  fusion_params =", {k: v for k, v in CFG['audionet']['audionet_config']['fusion_params'].items()},
      "| Separator 只用 fusion_shared（Falcon.py:997）")
print()
print("-- 权重共享/复用证据：一次 Separator.forward 内的模块调用次数 --")
cc = {"audio": 0, "video": 0, "caf": 0}
hh = [ab.register_forward_hook(lambda *a: cc.__setitem__("audio", cc["audio"] + 1)),
      vb.register_forward_hook(lambda *a: cc.__setitem__("video", cc["video"] + 1)),
      sep.caf.register_forward_hook(lambda *a: cc.__setitem__("caf", cc["caf"] + 1))]
with torch.no_grad():
    m.separator(a_bn, v_bn)
for h in hh:
    h.remove()
print("  调用次数:", cc)
levels = {i: 0 for i in range(ab.upsampling_depth)}
hk = [ab.process_t_layers[i].register_forward_hook(
    (lambda i: lambda *a: levels.__setitem__(i, levels[i] + 1))(i))
    for i in range(ab.upsampling_depth)]
with torch.no_grad():
    m.separator(a_bn, v_bn)
for h in hk:
    h.remove()
print("  process_t_layers 各 level 调用次数:", levels)

print()
print("-- 归一化/因果开关的实际取值 --")
print("  audio_bottleneck.pre_norm =", type(m.audio_bottleneck.pre_norm).__name__,
      "| .norm =", type(m.audio_bottleneck.norm).__name__,
      "| .conv =", type(m.audio_bottleneck.conv).__name__)
print("  ab.gateway.conv / residual_conv.conv =",
      type(ab.gateway.conv).__name__, "/", type(ab.residual_conv.conv).__name__)
print("  ab.downsample_layers[0].causal =", ab.downsample_layers[0].causal,
      "| conv.padding =", ab.downsample_layers[0].conv.padding)
print("  ab.downsample_layers[1].causal =", ab.downsample_layers[1].causal,
      "| conv.padding =", ab.downsample_layers[1].conv.padding,
      "| conv.stride =", ab.downsample_layers[1].conv.stride)
print("  ab.fusion_layers[0].local_embedding.causal =",
      ab.fusion_layers[0].local_embedding.causal,
      "| conv.padding =", ab.fusion_layers[0].local_embedding.conv.padding)
print("  encoder.conv.causal / conv.padding =", m.encoder.conv.causal,
      m.encoder.conv.conv.padding)
print("  decoder.causal/padding =", m.decoder.causal, m.decoder.padding,
      "| decoder op =", type(m.decoder.decoder).__name__,
      "| win/hop =", m.decoder.win, m.decoder.hop_length)
print("  encoder/decoder 的 torch.stft/istft 未传 center/pad_mode -> center=True, reflect")
mg = m.mask_generator
print("  mask_generator.RI_split/output_gate/dw_gate/direct =",
      mg.RI_split, mg.output_gate, mg.dw_gate, mg.direct)
print("  mask_generator.mask_generator[1].conv =",
      type(mg.mask_generator[1].conv).__name__,
      "| padding =", mg.mask_generator[1].conv.padding)
print()
print("-- mdcl_window buffer --")
print("  mdcl_window:", tuple(m.mdcl_window.shape), "| 在 state_dict 中:",
      "mdcl_window" in m.state_dict(), "| state_dict key 数:", len(m.state_dict()))
print("  state_dict 首/尾 key:", list(m.state_dict())[0], "|", list(m.state_dict())[-1])
print()
print("DONE")
