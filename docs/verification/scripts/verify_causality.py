"""Lead 独立验证脚本 2：因果性 / 有效前瞻（lookahead）精确定位。

思路：把输入音频的第 s 个采样点改动 delta，观察输出波形（以及各中间 stage）从哪个采样点开始
发生变化。若严格因果，则输出索引 < s 的采样点应完全不变。

同时定位“泄漏”发生的第一层：encoder / audio_bottleneck / separator / mask_generator / decoder，
分别在时间轴（-2 维）上找 first-changed / last-changed 索引。

运行（在仓库根）：
    PYTHONPATH=<repo>/docs/verification/stubs:<repo> \
      /root/.venvs/falcon/bin/python docs/verification/scripts/verify_causality.py
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

torch.manual_seed(0)
model = models.FalconSpeakerContrastive(sample_rate=16000, **aconf)
model.eval()

B, T = 1, 32000
hop = aconf["enc_dec_params"]["hop_length"]      # 128
win = aconf["enc_dec_params"]["win"]             # 256
mouth = torch.randn(B, aconf["pretrained_vout_chan"], 50)

a1 = torch.randn(B, T)
s = 20000  # 被改动的采样点位置
a2 = a1.clone()
a2[..., s] += 1.0

TOL = 1e-6


def first_last_changed(x1, x2, time_dim):
    d = (x1 - x2).abs()
    # 逐时间步取最大差异（time_dim 支持负数）
    td = time_dim % d.ndim
    dims = [i for i in range(d.ndim) if i != td]
    per_t = d.amax(dim=dims)
    idx = (per_t > TOL).nonzero().flatten()
    if idx.numel() == 0:
        return None, None, 0.0
    return int(idx[0]), int(idx[-1]), float(per_t.max())


def frames_to_samples(i):
    return i * hop


print("=" * 78)
print(f"单采样点扰动: a2[{s}] += 1.0  (win={win}, hop={hop})")
print("严格因果 => 输出中所有 index < s 的采样点不变")
print("=" * 78)

with torch.no_grad():
    y1 = model(a1, mouth)  # eval 模式返回 tensor
    y2 = model(a2, mouth)
f, l, mx = first_last_changed(y1, y2, time_dim=2)
print(f"\n[全模型 est] first_changed_sample={f} last_changed_sample={l} max|d|={mx:.3e}")
print(f"  改动点 s={s}: 若 f >= s 则严格逐采样因果；实际 f-s = {f - s} samples ({(f - s) / 16000 * 1000:.2f} ms)")
print(f"  影响区间: [{f}, {l}]  (相对改动点: {f - s} .. {l - s} samples)")

# ---- 逐 stage 定位 ----
print("\n-- 逐 stage 的 first/last changed（时间维） --")
with torch.no_grad():
    e1 = model.encoder(a1)
    e2 = model.encoder(a2)
    f, l, mx = first_last_changed(e1, e2, time_dim=2)
    print(f"encoder          : frames [{f}, {l}]  (samples [{frames_to_samples(f)}, "
          f"{frames_to_samples(l)}])  max|d|={mx:.3e}")

    b1 = model.audio_bottleneck(e1)
    b2 = model.audio_bottleneck(e2)
    f, l, mx = first_last_changed(b1, b2, time_dim=2)
    print(f"audio_bottleneck : frames [{f}, {l}]  (samples [{frames_to_samples(f)}, "
          f"{frames_to_samples(l)}])  max|d|={mx:.3e}")

    v = model.video_bottleneck(mouth)
    sep1 = model.separator(b1, v)
    sep2 = model.separator(b2, v)
    f, l, mx = first_last_changed(sep1, sep2, time_dim=2)
    print(f"separator        : frames [{f}, {l}]  (samples [{frames_to_samples(f)}, "
          f"{frames_to_samples(l)}])  max|d|={mx:.3e}")

    m1 = model.mask_generator(sep1, e1)
    m2 = model.mask_generator(sep2, e2)
    f, l, mx = first_last_changed(m1, m2, time_dim=3)
    print(f"mask_generator   : frames [{f}, {l}]  (samples [{frames_to_samples(f)}, "
          f"{frames_to_samples(l)}])  max|d|={mx:.3e}")

# ---- 定位泄漏到“过去”的第一层：用更小的扰动并检查 first_changed < s 的程度 ----
print("\n-- 泄漏来源定位（改动点 s=20000，STFT 对称窗 ±8ms 本身的自然前瞻约 128 samples）--")
print("说明：torch.stft(center=True) 使每帧使用 [t-128, t+128]，因此编码器输出的“首变帧”")
print("      合法位置为 ceil((s-128)/hop) 附近；若 bottleneck 之后首变位置明显提前，即为归一化泄漏。")
with torch.no_grad():
    for name, (x1, x2), td in [
        ("encoder", (e1, e2), 2),
        ("audio_bottleneck", (b1, b2), 2),
        ("separator", (sep1, sep2), 2),
    ]:
        d = (x1 - x2).abs().amax(dim=[0, 1, 3])
        dur = (d > TOL).nonzero().flatten()
        print(f"{name:16s}: first_frame={int(dur[0]) if dur.numel() else None} "
              f"(samples~{frames_to_samples(int(dur[0])) if dur.numel() else None})")

print("\n-- 关键结构常量 --")
print("torch.stft 默认 center:", torch.stft(torch.zeros(256), n_fft=256, hop_length=128,
                                          window=torch.hann_window(256), return_complex=True).shape)
print("encoder conv causal padding:", model.encoder.conv.causal_padding,
      "| decoder convT kernel:", model.decoder.kernel_size,
      "| decoder causal pad:", model.decoder.causal)
print("sep.audio_net type:", type(model.separator.audio_net).__name__)
flags = sorted({(type(m).__name__, getattr(m, "causal", None))
                for _, m in model.separator.named_modules()
                if hasattr(m, "causal")})
print("modules with .causal attr (cls, value):", flags)
print("normalizations in model:")
for n, m in model.named_modules():
    if m.__class__.__name__ in ("LayerNormalizationHW", "GlobalLayerNorm", "CumulateLN4D",
                                "LayerNormalization4D", "CumulateLN", "CumulativeLayerNorm"):
        print(f"  {n:60s} <= {m.__class__.__name__}")
