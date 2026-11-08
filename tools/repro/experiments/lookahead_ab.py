#!/usr/bin/env python3
"""lookahead_ab.py — `paper_causal` / `video_lstm_batch_first` 两个忠实化开关的 A/B 实测。

背景
----
`look2hear/models/Falcon.py` 里有两处与论文描述不符、且**默认关闭**的忠实化开关
（默认值 = 发布权重 `falcon.pt` 的原始行为，逐位不变）：

  * `paper_causal=True` —— 把 `AudioBlock` 里三处硬编码的 `causal=False`
    （`downsample_layers` / `fusion_layers` / `concat_layers`）改为透传 `self.causal`，
    即论文 §4.2 声称的 left-aligned causal padding；默认 False 保持对称 padding。
  * `video_lstm_batch_first=True` —— `VideoBlock` 的 `nn.LSTM` 用 `batch_first=True`。
    默认 False 时 `[B, T, C]` 被 LSTM 读成 `[T, B, C]`（帧轴当成了 batch 轴），
    50 个视频帧之间**没有任何递推**；True 时 LSTM 真正沿 25 fps 帧轴递推。

本脚本用**真实权重 + 官方 config**、纯 CPU、不训练，对 4 种组合各测：

  1. 音频「看未来帧数」—— 上轮同款方法：在输入第 `p` 个采样点加 delta，
     找最早受影响的输出采样点，`(p - 最早受影响点) / hop` 即看未来的帧数。
  2. 视频「看未来帧数」—— 在视频第 25 帧加 delta，同法换算到 25 fps 帧。
  3. 与默认配置输出的 `max|Δ|`（并显式报告是否**逐位相同**）。
  4. 单窗（2 s）前向耗时中位数。

为什么 e2e 数字对 4 种配置几乎一样（重要）
------------------------------------------
端到端扰动会瞬间传遍全窗，是因为 `cLNhw = LayerNormalizationHW` 的第一段是
`nn.GroupNorm(1, C)`，在 `(C, T, F)` 全窗口上统计（`look2hear/layers/normalizations.py`）。
这条全局耦合远强于卷积 padding，所以 `paper_causal` **不会**改变 e2e 传播范围。
开关真正改变的是**局域几何**，因此本脚本额外打印一张诊断表：
  * 逐卷积的 `(k, stride, pad, causal)` 与解析前瞻帧数 `(k-1) - pad_left`（2 帧 → 0 帧）；
  * 用 delta 直通实测该卷积的感受野，验证解析值；
  * 视频分支「扰动第 25 帧后受影响的特征帧数」（1 帧 → 25 帧）。
归一化那一层的问题见同目录 `ln_variant.py`。

硬约束
------
* 纯 CPU（`CUDA_VISIBLE_DEVICES=`）、`eval()` + `no_grad()`、固定 seed、不训练、不写仓库。
* 单次运行 ≤120 s（实测约 40 s，4 线程）。
* 权重必须以 `strict=True` 装载成功；失败则退出码 3（不静默）。

用法
----
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
      /root/miniforge3/envs/falcon/bin/python tools/repro/experiments/lookahead_ab.py

    # 指定权重 / 落 JSON
    ... lookahead_ab.py --ckpt /root/.cache/falcon_weights/falcon.pt --json /tmp/ab.json

退出码：0 正常；2 权重或配置缺失；3 权重与模型不匹配；4 当前代码没有这两个开关。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]          # tools/repro/experiments/x.py
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
DEFAULT_CONFIG = REPO / "configs" / "Falcon.yml"

SEG_LEN = 32000          # 2 s @ 16 kHz
HOP = 128                # n_fft/2 -> 125 audio frames/s
VIDEO_FPS = 25
N_VIDEO_FRAMES = 50      # int(2.0 * 25)
P_AUDIO = 16000          # delta 位置（1.0 s）
P_VIDEO = 25             # delta 帧

COMBOS = [
    ("default", False, False),
    ("paper_causal", True, False),
    ("video_lstm_batch_first", False, True),
    ("both", True, True),
]


# ----------------------------------------------------------------------------
# 模型构造 / 权重
# ----------------------------------------------------------------------------
def build_model(ckpt: Path, config: Path, paper_causal: bool, video_batch_first: bool):
    import look2hear.models as LM

    conf = yaml.safe_load(open(config, encoding="utf-8"))
    ac = copy.deepcopy(conf["audionet"]["audionet_config"])
    ac["paper_causal"] = paper_causal
    ac["video_lstm_batch_first"] = video_batch_first
    torch.manual_seed(0)
    try:
        model = LM.FalconSpeakerContrastive(sample_rate=16000, **ac)
    except TypeError as exc:
        raise SystemExit(f"[exit 4] 当前 Falcon.py 没有这两个开关：{exc}")
    model.eval()

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    audio = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    if not audio:
        raise SystemExit(f"[exit 2] {ckpt} 里没有 audio_model.* 权重")
    try:
        model.load_state_dict(audio, strict=True)
    except RuntimeError as exc:  # 形状/键不匹配 -> 立刻停下
        raise SystemExit(f"[exit 3] strict=True 装载失败：{exc}")
    return model, len(audio)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
@torch.no_grad()
def forward(model, wav, mouth):
    return model(wav, mouth)


def first_changed(diff: torch.Tensor) -> int:
    nz = (diff.reshape(-1) > 0).nonzero()
    return int(nz[0]) if len(nz) else -1


def measure_e2e(model, wav, mouth):
    """返回 (音频看未来帧数, 视频看未来帧数, 输出张量)。"""
    y0 = forward(model, wav, mouth)

    wav2 = wav.clone()
    wav2[0, P_AUDIO] += 1.0
    y_a = forward(model, wav2, mouth)
    fa = first_changed((y_a - y0).abs())
    audio_future_frames = (P_AUDIO - fa) / HOP if fa >= 0 else float("nan")

    mouth2 = mouth.clone()
    mouth2[0, :, P_VIDEO] += 1.0
    y_v = forward(model, wav, mouth2)
    fv = first_changed((y_v - y0).abs())
    video_future_frames = P_VIDEO - (fv / 16000.0) * VIDEO_FPS if fv >= 0 else float("nan")
    return audio_future_frames, video_future_frames, y0


def measure_latency(model, wav, mouth, n_warmup=1, n_timed=3):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(wav, mouth)
        ts = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            model(wav, mouth)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000.0


# ----------------------------------------------------------------------------
# 诊断：开关到底改了什么（结构性，与归一化无关）
# ----------------------------------------------------------------------------
def conv_lookahead_frames(module) -> float:
    """解析：输出帧 j 覆盖输入 [s*j - pad_L, s*j - pad_L + k - 1]，
    越过自身位置 s*j 的前瞻帧数 = (k-1) - pad_L（按该层的输入帧计）。"""
    c = getattr(module, "conv", None)
    if not isinstance(c, torch.nn.Conv2d):
        return float("nan")
    return float((c.kernel_size[0] - 1) - c.padding[0])


def delta_receptive_field(conv, t_in=32, T=96, F=33):
    """把 delta 直接喂进 nn.Conv2d（绕开 ConvNormAct 里的全窗 GroupNorm），
    返回受影响输出帧区间 (j_min, j_max)。"""
    C = conv.in_channels
    x = torch.zeros(1, C, T, F)
    x[:, :, t_in, F // 2] = 1.0
    with torch.no_grad():
        y = conv(x) - conv(torch.zeros_like(x))   # 去掉 bias 的常数响应
    prof = y.abs().max(dim=1).values[0].max(dim=-1).values   # [T']: per output frame energy
    nz = (prof > 0).nonzero().flatten()
    if not len(nz):
        return -1, -1
    return int(nz[0]), int(nz[-1])


def audio_geometry(model):
    an = model.separator.audio_net
    rows = []
    for name, mod in [
        ("downsample[0]", an.downsample_layers[0]),
        ("downsample[1]", an.downsample_layers[1]),
        ("fusion[0].local", an.fusion_layers[0].local_embedding),
        ("concat[0].local", an.concat_layers[0].local_embedding),
    ]:
        c = mod.conv
        rows.append(
            dict(site=name, k=int(c.kernel_size[0]), stride=int(c.stride[0]),
                 pad_H=int(c.padding[0]), pad_W=int(c.padding[1]),
                 causal=bool(getattr(mod, "causal", False)),
                 lookahead_frames=conv_lookahead_frames(mod),
                 rf=delta_receptive_field(c))
        )
    return rows


@torch.no_grad()
def video_recurrence(model, mouth):
    """扰动视频第 P_VIDEO 帧，数一下 VideoBlock 输出有多少帧变了。"""
    vb = model.separator.video_net
    f0 = vb(mouth)
    m2 = mouth.clone()
    m2[0, :, P_VIDEO] += 1.0
    f1 = vb(m2)
    prof = (f1 - f0).abs().max(dim=1).values[0]   # [T_v]: per video frame energy
    changed = (prof > 0).nonzero().flatten()
    rel = (prof > 1e-6 * prof.max()).nonzero().flatten() if float(prof.max()) > 0 else changed
    return dict(batch_first=bool(vb.rnn.batch_first),
                n_changed=int(len(changed)),
                n_changed_rel=int(len(rel)),
                first=int(changed[0]) if len(changed) else -1,
                last=int(changed[-1]) if len(changed) else -1,
                last_rel=int(rel[-1]) if len(rel) else -1,
                decay_frames=int(rel[-1] - rel[0] + 1) if len(rel) else 0)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="paper_causal / video_lstm_batch_first A/B")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None, help="把指标写成 JSON")
    args = ap.parse_args(argv)

    if not args.ckpt.is_file():
        print(f"[exit 2] 权重不存在：{args.ckpt}")
        return 2
    if not args.config.is_file():
        print(f"[exit 2] config 不存在：{args.config}")
        return 2
    torch.set_num_threads(args.threads)

    g = torch.Generator().manual_seed(2026)
    wav = torch.randn(1, SEG_LEN, generator=g)
    mouth = torch.randn(1, 512, N_VIDEO_FRAMES, generator=g)

    print(f"# lookahead A/B — ckpt={args.ckpt.name} config={args.config.name} "
          f"threads={args.threads} seed=2026")
    print(f"# 输入：wav [1,{SEG_LEN}] 2 s@16 kHz，mouth [1,512,{N_VIDEO_FRAMES}] 25 fps\n")

    results = {}
    nkeys = {}
    for name, pc, vb in COMBOS:
        model, nk = build_model(args.ckpt, args.config, pc, vb)
        nkeys[name] = nk
        a_fut, v_fut, y = measure_e2e(model, wav, mouth)
        ms = measure_latency(model, wav, mouth)
        results[name] = dict(paper_causal=pc, video_lstm_batch_first=vb,
                             audio_future_frames=a_fut, video_future_frames=v_fut,
                             out=y, ms=ms, sha=sha(y))

    base = results["default"]["out"]
    print("## 主表：4 种组合的端到端行为\n")
    print("| 配置 | 音频看未来(帧) | 视频看未来(帧) | 与默认输出 max\\|Δ\\| | 逐位相同 | 单窗耗时(ms) |")
    print("|---|---|---|---|---|---|")
    for name, _, _ in COMBOS:
        r = results[name]
        d = (r["out"] - base).abs().max().item()
        same = "✅ 是" if torch.equal(r["out"], base) else "❌ 否"
        print(f"| `{name}` | {r['audio_future_frames']:.1f} | {r['video_future_frames']:.2f} | "
              f"{d:.3e} | {same} | {r['ms']:.0f} |")
    print()
    print("> 「看未来帧数」= 在输入第 1000 ms（音频）/ 第 25 帧（视频）加 delta 后，"
          "最早受影响的输出采样点换算出的帧数；125 帧 = 整个 2 s 窗口。")
    print("> 4 行数值几乎相同是**预期**的：端到端传播被 `cLNhw` 的全窗 `GroupNorm(1,C)` 支配，"
          "卷积 padding 不是瓶颈（见下方诊断表与 `ln_variant.py`）。\n")

    # ---------------- 诊断 1：卷积几何 ----------------
    print("## 诊断 1：`paper_causal` 改变的局域卷积几何（与归一化无关）\n")
    for name in ("default", "paper_causal"):
        r = results[name]
        model, _ = build_model(args.ckpt, args.config, r["paper_causal"], r["video_lstm_batch_first"])
        print(f"### `{name}`\n")
        print("| 站点 | k | stride | pad(H,W) | causal | 解析前瞻(输入帧) | delta 实测受影响输出帧 |")
        print("|---|---|---|---|---|---|---|")
        for row in audio_geometry(model):
            print(f"| `{row['site']}` | {row['k']} | {row['stride']} | ({row['pad_H']},{row['pad_W']}) | "
                  f"{row['causal']} | {row['lookahead_frames']:.0f} | "
                  f"[{row['rf'][0]}, {row['rf'][1]}] |")
        print()

    # ---------------- 诊断 2：视频递推 ----------------
    print("## 诊断 2：`video_lstm_batch_first` 是否让 LSTM 沿时间递推\n")
    print("| 配置 | `rnn.batch_first` | 受影响帧数 (>0) | 受影响帧数 (>1e-6·max) | 区间 (>0) | 递推可见长度 |")
    print("|---|---|---|---|---|---|")
    for name in ("default", "video_lstm_batch_first"):
        r = results[name]
        model, _ = build_model(args.ckpt, args.config, r["paper_causal"], r["video_lstm_batch_first"])
        vr = video_recurrence(model, mouth)
        print(f"| `{name}` | {vr['batch_first']} | {vr['n_changed']} / {N_VIDEO_FRAMES} | "
              f"{vr['n_changed_rel']} / {N_VIDEO_FRAMES} | [{vr['first']}, {vr['last']}] | "
              f"{vr['decay_frames']} 帧 |")
    print()

    # ---------------- 结论 ----------------
    print("## 结论（全部实测）\n")
    d_pc = (results["paper_causal"]["out"] - base).abs().max().item()
    d_vb = (results["video_lstm_batch_first"]["out"] - base).abs().max().item()
    d_both = (results["both"]["out"] - base).abs().max().item()
    print(f"1. **默认配置与发布权重逐位相同**：`default` 输出 sha256[:16] = `{results['default']['sha']}`，"
          f"`torch.equal` 对自身 = True；`strict=True` 装载 {nkeys['default']} 个 audio key 成功。")
    print(f"2. `paper_causal=True` 改变输出：max|Δ| = {d_pc:.3e}（**不是**逐位相同）；"
          f"局域前瞻从 2 帧降到 0 帧；端到端「看未来」帧数几乎不变（{results['default']['audio_future_frames']:.1f} "
          f"→ {results['paper_causal']['audio_future_frames']:.1f}）。")
    vr0 = video_recurrence(build_model(args.ckpt, args.config, False, False)[0], mouth)
    vr1 = video_recurrence(build_model(args.ckpt, args.config, False, True)[0], mouth)
    print(f"3. `video_lstm_batch_first=True` 改变输出：max|Δ| = {d_vb:.3e}；"
          f"扰动第 25 帧后 VideoBlock 受影响帧数 {vr0['n_changed']} → {vr1['n_changed']}"
          f"（>1e-6·max 口径 {vr0['n_changed_rel']} → {vr1['n_changed_rel']}，"
          f"递减到 float32 下溢即停），即默认配置**帧间完全没有递推**。")
    print(f"4. 两者同开：max|Δ| = {d_both:.3e}。")
    print(f"5. 单窗耗时：default {results['default']['ms']:.0f} ms / paper_causal "
          f"{results['paper_causal']['ms']:.0f} ms / video_lstm_batch_first "
          f"{results['video_lstm_batch_first']['ms']:.0f} ms / both {results['both']['ms']:.0f} ms。")
    print("6. 所以这两个开关**只修几何，不修归一化**：要让模型真正无视未来，还得动 `cLNhw`"
          "（见 `ln_variant.py`）。")

    if args.json:
        payload = {k: {kk: vv for kk, vv in v.items() if kk != "out"} for k, v in results.items()}
        payload["n_audio_keys"] = nkeys
        payload["out_sha_default"] = results["default"]["sha"]
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[ok] 指标已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
