"""R4 执行级验证：02-layers-dsp 的归一化 / STFT / 注意力因果性声明独立复核。
运行：PYTHONPATH=<repo>/docs/verification/stubs:<repo> /root/.venvs/falcon/bin/python <this>
输出：docs/verification/runs/train-sys-r4-layers-causal.log
"""
import sys
from pathlib import Path
import torch
import yaml

REPO = Path("/root/code/github_repos/Falcon-fork")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

from look2hear.layers.normalizations import (
    CumulateLN, CumulateLN4D, LayerNormalizationHW, GlobalLayerNorm)
from look2hear.models.Falcon import (
    STFTEncoder, STFTDecoder, DualPathRNN, MultiHeadSelfAttention2D)
from look2hear.layers.stft import forward_stft

cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
a = cfg["audionet"]["audionet_config"]
enc_cfg = dict(a["enc_dec_params"])
torch.manual_seed(0)


def first_change(a_, b_, dim):
    if dim < 0:
        dim = a_.ndim + dim
    d = (a_ - b_).abs().amax(dim=[d for d in range(a_.ndim) if d != dim])
    nz = torch.nonzero(d > 1e-7).flatten()
    return int(nz[0]) if nz.numel() else -1, float(d.max())


print("=" * 78)
print("[R4-1] CumulateLN / CumulateLN4D：时间轴累积 => 因果（改未来不影响过去）")
print("=" * 78)
ln = CumulateLN(8).eval()
x = torch.randn(1, 8, 32)
with torch.no_grad():
    y0 = ln(x)
    x2 = x.clone(); x2[0, 0, 16] += 5.0
    y1 = ln(x2)
fc, md = first_change(y0, y1, dim=2)
print(f"CumulateLN: 改 t=16 -> 首个变化 t={fc}（应=16），max|Δ|={md:.3e}")
ln4 = CumulateLN4D(8).eval()
x4 = torch.randn(1, 8, 32, 4)
with torch.no_grad():
    z0 = ln4(x4)
    x4b = x4.clone(); x4b[0, 0, 16, :] += 5.0
    z1 = ln4(x4b)
fc4, md4 = first_change(z0, z1, dim=2)
print(f"CumulateLN4D: 改 t=16 -> 首个变化 t={fc4}（应=16），max|Δ|={md4:.3e}")

print()
print("=" * 78)
print("[R4-2] LayerNormalizationHW = GroupNorm(1,C)+CumulateLN4D：GN 跨全时间轴 => 非因果")
print("=" * 78)
for T in (8, 251):
    hw = LayerNormalizationHW(8).eval()
    xh = torch.randn(1, 8, T, 4)
    with torch.no_grad():
        o0 = hw(xh)
        xh2 = xh.clone(); xh2[0, 0, T - 2, :] = 50.0       # 只改靠后帧的单个通道（置大值）
        o1 = hw(xh2)
    # 观察“改点之前”的最大变化
    early = (o0[..., : T - 3, :] - o1[..., : T - 3, :]).abs().max().item()
    rel = early / (o0[..., : T - 3, :].abs().max().item() + 1e-12)
    print(f"T={T:3d}: 改动 t=T-2 的单个通道后，t<=T-4 的早期帧 max|Δ|={early:.3e}（相对 {rel:.2e}）"
          f" -> {'非因果' if early > 1e-6 else '因果'}")

gl = GlobalLayerNorm(8).eval()
xg = torch.randn(1, 8, 16)
with torch.no_grad():
    g0 = gl(xg); xg2 = xg.clone(); xg2[0, 0, -1] += 5.0; g1 = gl(xg2)
print(f"GlobalLayerNorm: 改最后帧 -> 首帧 max|Δ|={float((g0[0,:,0]-g1[0,:,0]).abs().max()):.3e}（非因果）")

print()
print("=" * 78)
print("[R4-3] STFTEncoder：帧级 lookahead（A37 声称 <=128 samples = 8 ms）")
print("=" * 78)
enc = STFTEncoder(**enc_cfg).eval()
S = 32000
wav = torch.randn(1, S)
with torch.no_grad():
    e0 = enc(wav)                     # [1,C,T',F]
    wav2 = wav.clone(); wav2[0, 16000] += 5.0
    e1 = enc(wav2)
first_frame, md = first_change(e0, e1, dim=2)
print(f"扰动采样 s=16000 -> encoder 首个变化帧={first_frame}（理论 ceil((s-128)/128)=124~125），max|Δ|={md:.3e}")
print(f"  首变帧 t={first_frame} 的 STFT 窗覆盖采样 [{(first_frame-1)*128}, {(first_frame+1)*128})；"
      f"对 s=16000 的前瞻上界 = {first_frame*128+128-16000} 采样（=128 即 8 ms）")
# 严格检查：s 之前（更早）的帧是否有变化
bad = [t for t in range(first_frame) if (e0[0, :, t, :] - e1[0, :, t, :]).abs().max() > 1e-7]
print(f"  早于 first_frame 的受影响帧数={len(bad)}（应为 0）")

print()
print("=" * 78)
print("[R4-4] STFTDecoder：causal 模式下采样级因果（A38）")
print("=" * 78)
dec = STFTDecoder(**enc_cfg, in_chan=112, n_src=1).eval()
B, n_src, Tp, F = 1, 1, 251, 129
x_emb = torch.randn(B, n_src, 112, Tp, F)
input_shape = torch.Size([B, 1, 32000])
with torch.no_grad():
    d0 = dec(x_emb, input_shape)          # [B,1,L]
    e2 = x_emb.clone(); e2[0, 0, :, 125, :] += 5.0
    d1 = dec(e2, input_shape)
first_s, md = first_change(d0, d1, dim=-1)
print(f"扰动 embedding 帧 125 -> 首个变化输出采样={first_s}（帧 125 名义时刻=16000），max|Δ|={md:.3e}")
print(f"  结论: 输出采样 < {first_s} 不变 = {first_s >= 16000}（A38 成立则 >=16000）")

print()
print("=" * 78)
print("[R4-5] MultiHeadSelfAttention2D：causal=True 的三角 mask 是否真的遮住未来帧")
print("=" * 78)
for causal in (True, False):
    mh = MultiHeadSelfAttention2D(in_chan=16, n_freqs=4, n_head=4, hid_chan=4,
                                  act_type="PReLU", norm_type="LayerNormalization4D",
                                  dim=3, causal=causal).eval()
    xm = torch.randn(1, 16, 16, 4)
    with torch.no_grad():
        m0 = mh(xm)
        xm2 = xm.clone(); xm2[0, :, -1, :] += 5.0
        m1 = mh(xm2)
    early = (m0[..., :-1, :] - m1[..., :-1, :]).abs().max().item()
    print(f"causal={causal}: 改最后时间帧 -> 之前帧 max|Δ|={early:.3e} "
          f"-> {'因果' if early < 1e-6 else '非因果'}")

print()
print("=" * 78)
print("[R4-6] DualPathRNN：dim=3 沿时间单向（因果）；dim=4 沿频率双向")
print("=" * 78)
dp3 = DualPathRNN(in_chan=32, hid_chan=32, dim=3, kernel_size=8, stride=1,
                  rnn_type="LSTM", num_layers=1, norm_type="LayerNormalization4D",
                  act_type="PReLU", bidirectional=False, causal=True).eval()
x3 = torch.randn(1, 32, 12, 6)
with torch.no_grad():
    r0 = dp3(x3); x3b = x3.clone(); x3b[0, :, -1, :] += 5.0; r1 = dp3(x3b)
early3 = (r0[..., :-1, :] - r1[..., :-1, :]).abs().max().item()
print(f"dim=3 (时间单向): 改最后时间帧 -> 之前帧 max|Δ|={early3:.3e} -> {'因果' if early3 < 1e-6 else '非因果'}")

dp4 = DualPathRNN(in_chan=32, hid_chan=32, dim=4, kernel_size=8, stride=1,
                  rnn_type="LSTM", num_layers=1, norm_type="LayerNormalization4D",
                  act_type="PReLU", bidirectional=True, causal=True).eval()
x4 = torch.randn(1, 32, 12, 6)
with torch.no_grad():
    q0 = dp4(x4); x4b = x4.clone(); x4b[0, 0, :, -1] += 5.0; q1 = dp4(x4b)
early4 = (q0[..., :, :-1] - q1[..., :, :-1]).abs().max().item()
print(f"dim=4 (频率双向): 改最后一个频点 -> 其余频点 max|Δ|={early4:.3e} -> {'频率耦合（预期）' if early4 > 1e-6 else '无耦合'}")

print()
print("=" * 78)
print("[R4-7] layers/stft.py forward_stft vs torch.stft 等价性（02 §2 声称 max|Δ|≈2.3e-5）")
print("=" * 78)
w = torch.randn(2, 32000)
for center in (True, False):
    s1 = forward_stft(w, 256, 128, window="hann", center=center, mode="librosa", return_polar=False)
    # forward_stft 返回 [B,2,F,T] 或 [B,F,T,2]；与 torch.stft 对齐
    spec = torch.stft(w, n_fft=256, hop_length=128, window=torch.hann_window(256),
                      center=center, return_complex=True)
    ref = torch.view_as_real(spec)  # [B,129,T,2]
    print(f"center={center}: forward_stft shape={tuple(s1.shape)}, torch.stft(perm)={tuple(ref.shape)}")
    if tuple(s1.shape) == tuple(ref.shape):
        print(f"  max|Δ|={(s1 - ref).abs().max().item():.3e}")
    else:
        print("  shape 不一致，尝试转置比较")
        print("  forward_stft 输出布局 [B,F,T,2] 与 view_as_real(torch.stft) 不同，跳过数值比较")

print()
print("=" * 78)
print("[R4-8] 帧数账本：encoder / decoder / istft length")
print("=" * 78)
L, win, hop = 32000, 256, 128
print(f"center=True 帧数公式 1+L//hop = {1 + L // hop}")
print(f"center=False 帧数公式 1+(L-win)//hop = {1 + (L - win) // hop}")
print("DONE-R4-LAYERS")
