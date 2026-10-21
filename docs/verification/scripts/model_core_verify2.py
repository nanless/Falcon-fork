"""model-core 补充验证：因果性定量（感受野/新增延迟）、视频分支因果性、说话人头与 NASS 损失。

运行：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify2.py
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
from look2hear.losses.nass_contrastive_loss import NASSContrastiveLoss  # noqa: E402
from look2hear.models.Falcon import AudioBlock  # noqa: E402
from look2hear.videomodels import FRCNNVideoModel  # noqa: E402

torch.manual_seed(0)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
NAME = CFG["audionet"]["audionet_name"]
m = getattr(models, NAME)(sample_rate=16000, **copy.deepcopy(ACONF))
m.eval()
sep = m.separator
ab = sep.audio_net
HOP = ACONF["enc_dec_params"]["hop_length"]   # 128
WIN = ACONF["enc_dec_params"]["win"]          # 256

print("=" * 78)
print("[G] 全模型前缀扰动（改后 1s，逐 1/8 段看左段变化）")
print("=" * 78)
B, L = 1, 32000
audio = torch.randn(B, L)
mouth = torch.randn(B, ACONF["pretrained_vout_chan"], 50)
a2 = audio.clone()
a2[..., L // 2:] = torch.randn(B, L // 2)
with torch.no_grad():
    y1 = m(audio, mouth)
    y2 = m(a2, mouth)
d = (y1 - y2).abs()
seg = L // 8
for i in range(8):
    lo, hi = i * seg, (i + 1) * seg
    print(f"  samples[{lo:5d}:{hi:5d}]  t={(lo / 16000):.2f}-{(hi / 16000):.2f}s  "
          f"max|Δ|={d[..., lo:hi].max().item():.3e}")
print("  -> 前 0.5s 内已有 1e-4 量级差异；1.0s 后迅速增大到 O(1)")

print()
print("=" * 78)
print("[H] 因果性定量：各组件「未来依赖」帧数/样本数")
print("=" * 78)


def lookahead(fn, x, in_dim, out_dim, split, label, unit=""):
    """只把 split 之后的输入换成噪声，检查输出在 split 之前是否有变化，
    并找出输出左段中「最靠右的受影响索引」相对分界的越界量。"""
    idx = [slice(None)] * x.dim()
    idx[in_dim] = slice(split, None)
    x1, x2 = x.clone(), x.clone()
    x2[tuple(idx)] = torch.randn_like(x2[tuple(idx)])
    with torch.no_grad():
        y1, y2 = fn(x1), fn(x2)
    d = (y1 - y2).abs()
    od = out_dim % d.dim()
    prof = d.amax(dim=tuple(i for i in range(d.dim()) if i != od))
    ch = (prof > 1e-12).nonzero().flatten()
    if ch.numel() == 0:
        print(f"  {label:34s} 输出完全不依赖未来 -> 严格因果")
        return
    tot = prof.shape[0]
    print(f"  {label:34s} 输出长度={tot}, 受影响索引 max={ch.max().item()}, "
          f"分界={split} -> 越界 {ch.max().item() - split} 个{unit or '单位'}")


# 编码器：样本 -> 帧
lookahead(lambda t: m.encoder(t), torch.randn(1, 32000), -1, -2, 16000,
          "STFTEncoder(stdft center=True)", unit="帧")
# 解码器：帧 -> 样本
lookahead(lambda t: m.decoder(t, (1, 1, 32000)), torch.randn(1, 1, 112, 251, 129),
          -2, -1, 125, "STFTDecoder(causal=True)", unit="样本")
# AudioBlock：关闭所有 cLNhw（仅剩卷积/RNN/attn 的结构性因果）
lay = copy.deepcopy(ACONF["audio_params"]["layers"])
ab_nonorm = AudioBlock(in_chan=112, hid_chan=64, kernel_size=5, stride=2,
                       norm_type=None, act_type="PReLU", upsampling_depth=4,
                       layers=lay, is2d=True, causal=True, n_freqs=129).eval()
x_ab = torch.randn(1, 112, 251, 129)
lookahead(lambda t: ab_nonorm(t), x_ab, -2, -2, 125, "AudioBlock(norm 全关)", unit="帧")
lookahead(lambda t: ab(t), x_ab, -2, -2, 125, "AudioBlock(实际 cLNhw 开)", unit="帧")
lookahead(lambda t: m.audio_bottleneck(t), x_ab, -2, -2, 125, "audio_bottleneck(cLNhw)", unit="帧")
# 深层分辨率换算：level-3 的 1 帧 = 8 个 level-0 帧
print("  换算: level0 1 帧 = 128 样本 = 8 ms; level3 1 帧 = 8 帧 level0 = 1024 样本 = 64 ms")

print()
print("=" * 78)
print("[I] 视频分支因果性（冻结 FRCNN + VideoBlock）")
print("=" * 78)
frcnn = FRCNNVideoModel(pretrain=None).eval()
vid = torch.randn(1, 1, 50, 96, 96)
vid2 = vid.clone()
vid2[:, :, 30:] = torch.randn(1, 1, 20, 96, 96)
with torch.no_grad():
    f1 = frcnn(vid)
    f2 = frcnn(vid2)
print("  FRCNN 输出形状:", tuple(f1.shape), "(B, 512, T_v)，T_v=50=2s@25fps")
d = (f1 - f2).abs().amax(dim=(0, 1))
print("  改视频帧>=30，输出帧受影响范围:", (d > 1e-12).nonzero().flatten().tolist()[:6], "...",
      "max =", (d > 1e-12).nonzero().flatten().max().item())
print("  -> 前端 Conv3d(kernel=(5,7,7), padding=(2,3,3)) 时间感受野 ±2 帧 = ±80 ms")
vb = sep.video_net
v_in = torch.randn(1, 512, 50)
for mode in ("eval", "train"):
    getattr(vb, mode)()
    v2 = v_in.clone()
    v2[:, :, 25:] = torch.randn(1, 512, 25)
    with torch.no_grad():
        o1, o2 = vb(v_in), vb(v2)
    dd = (o1 - o2).abs().amax(dim=(0, 1))
    ch = (dd > 1e-12).nonzero().flatten()
    print(f"  VideoBlock({mode}) 改帧>=25 -> 左段受影响 max index = "
          f"{ch.max().item() if ch.numel() else 'none'} (分界 24)")
print("  -> eval 下 BatchNorm1d 用 running stats，因果由单向 LSTM 决定；train 下 BN 统计含未来帧")

print()
print("=" * 78)
print("[J] FalconSpeakerContrastive 说话人头 + NASS 对比损失")
print("=" * 78)
e = m.encoder(audio)
emb_map = m.mask_generator(sep(m.audio_bottleneck(e), m.video_bottleneck(mouth)), e)
pool = m.spk_pool_proj(emb_map[:, 0])
norm = pool.norm(dim=-1)
z = m.pool_spk_embed(emb_map[:, 0])
print("  sep_emb[:,0] ->", tuple(emb_map[:, 0].shape), "| pool_spk_embed ->",
      tuple(z.shape), "| L2 norm =", [round(v, 6) for v in z.norm(dim=-1).tolist()])
print("  encode_stft(target wav) ->", tuple(m.encode_stft(audio).shape))
loss_fn = NASSContrastiveLoss(margin=0.5, alpha_pos=1.0, alpha_neg=1.0)
q = z.detach().clone().requires_grad_(True)
p = m.pool_spk_embed(m.encode_stft(torch.randn(1, 32000)))
n1 = m.pool_spk_embed(m.encode_stft(torch.randn(1, 32000)))
n2 = m.pool_spk_embed(m.encode_stft(torch.randn(1, 32000)))
for negs, beta in (([None], 0.0), ([n1], 0.0), ([n1], 5.0), ([n1, n2], 20.0)):
    L = loss_fn(q, p, negs, beta)
    print(f"  NASSLoss(K={0 if negs[0] is None else len(negs)}, beta={beta}) = {L.item():.6f}")
L = loss_fn(q, p, [n1, n2], 20.0)
L.backward()
print("  query 有梯度:", q.grad is not None, "| pos/neg 是否 detached（litmodule 用 no_grad 算）: 见 av_litmodule_nass_v2.py:106-131")
print("  cos(q,p) =", round(torch.nn.functional.cosine_similarity(q, p).item(), 4),
      "| cos(q,n1) =", round(torch.nn.functional.cosine_similarity(q, n1).item(), 4))

print()
print("=" * 78)
print("[K] 复算：参数量手算 vs 实测")
print("=" * 78)
def n(mo):
    return sum(p_.numel() for p_ in mo.parameters())
rows = [
    ("encoder.conv = 3*112*3*3 (bias=False)", 3 * 112 * 3 * 3, n(m.encoder.conv)),
    ("audio_bottleneck = cLNhw(112)+conv112->112 k1", 224 + 224 + 112 * 112 + 112,
     n(m.audio_bottleneck)),
    ("ab.gateway = dwconv k1 + PReLU", 112 + 112 + 1, n(ab.gateway)),
    ("ab.projection = 112*64+64 + cLNhw(64) + PReLU", 112 * 64 + 64 + 256 + 1,
     n(ab.projection)),
    ("ab.down[0] = dwconv k3 + cLNhw(64)", 64 * 9 + 64 + 256, n(ab.downsample_layers[0])),
    ("ab.down[1] = dwconv k5 + cLNhw(64)", 64 * 25 + 64 + 256, n(ab.downsample_layers[1])),
    ("proc_f[0] = biLSTM(64->24)*2 + out_proj(48->64) + LN4D(64,1)",
     2 * (4 * 24 * 64 + 4 * 24 * 24 + 2 * 4 * 24) + 48 * 64 + 128,
     n(ab.process_f_layers[0])),
    ("proc_t[0] = LSTM(64->72) + out_proj(72->64) + LN4D(64,1)",
     4 * 72 * 64 + 4 * 72 * 72 + 2 * 4 * 72 + 72 * 64 + 128,
     n(ab.process_t_layers[0])),
    ("proc_att[0] = 4*(2*1293+5169)+20673", 4 * (2 * 1293 + 5169) + 20673,
     n(ab.process_att_layers[0])),
    ("fusion[0] = 3*(64*25 + cLNhw(64))", 3 * (64 * 25 + 256), n(ab.fusion_layers[0])),
    ("ab.residual_conv = 64*112+112", 64 * 112 + 112, n(ab.residual_conv)),
    ("video_net.gateway = dwconv512 k1 + PReLU", 512 + 512 + 1, n(vb.gateway)),
    ("video_net.projection = 512*32+32 + BN1d(32) + PReLU", 512 * 32 + 32 + 64 + 1,
     n(vb.projection)),
    ("video_net.rnn = LSTM(32->32)", 4 * 32 * 32 + 4 * 32 * 32 + 2 * 128, n(vb.rnn)),
    ("video_net.residual_conv = 32*512+512", 32 * 512 + 512, n(vb.residual_conv)),
    ("caf.v_proj = Conv1d(512->112,k1,groups=16)", 512 * 112 // 16 + 112, n(sep.caf.v_proj)),
    ("mask_generator = PReLU + conv112->112 k1", 1 + 112 * 112 + 112, n(m.mask_generator)),
    ("decoder = ConvTranspose2d(112->2,k3)", 112 * 2 * 9, n(m.decoder)),
    ("spk_pool_proj = attn(112->28->1) + proj(112->64->64)", 112 * 28 + 28 + 28 +
     112 * 64 + 64 + 64 * 64 + 64, n(m.spk_pool_proj)),
]
ok = True
for name, manual, actual in rows:
    flag = "OK" if manual == actual else "MISMATCH"
    ok &= manual == actual
    print(f"  {flag:9s} {name:52s} 手算={manual:>7,} 实测={actual:>7,}")
tot = n(m)
print(f"  合计 audio_model 手算={sum(r[1] for r in rows):,} 实测={tot:,} "
      f"| 全部一致={ok and sum(r[1] for r in rows) == tot}")
print(f"  可训练/冻结: audio={tot:,}; FRCNN={n(frcnn):,}; 合计={tot + n(frcnn):,}")
print()
print("DONE2")
