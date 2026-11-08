#!/usr/bin/env python3
"""ln_variant.py — `cLNhw` vs 逐 (t,f) 通道 LayerNorm 的因果对照实验。

目的
----
`look2hear/layers/normalizations.py` 的 `cLNhw`（`LayerNormalizationHW`）是

    GlobalLayerNorm(C,T,F)  ->  CumulateLN4D(沿 T 累积、统计跨 C)

第一阶段的 `nn.GroupNorm(1,C)` 把**整个窗口**的 (C,T,F) 压成逐样本标量统计，
于是窗内任意一帧的输出都依赖窗内所有帧：时间因果在窗内被完全破坏。本脚本用
**不重训**的权重替换（只在本进程内 monkey-patch，不改仓库前向代码）来量化：

  metric 1 尾部扰动传播：扰动输入尾部 32 帧，看特征图每帧的相对响应 (dB)。
           因果模型在扰动区之前应当是数值零；as_shipped 应当全窗响应。
  metric 2 窗长敏感性：同一段音频跑 2 s 窗 vs 跑 4 s 窗取前 2 s，比 SI-SNR。
           依赖全窗统计的模型，多给未来上下文就会改变前 2 s 的结果。
  metric 3 输出电平：不重训换归一化的增益失配（rms 比）。

口径（重要）
------------
逐帧 dB 一律以**全局输出能量**为基准（不是逐帧自身能量）。逐帧基准会让近静音
帧把 1e-12 级的数值残差放大成 "-40 dB 的假响应"，从而虚报感受野。

硬约束
------
* 纯 CPU、不训练、不写日志、不改 `layers/` 逻辑；替换只发生在本进程的模块树上。
* 确定性：固定 seed；`eval()` + `no_grad()`；输入固定（真实 wav 或合成白噪）。

用法
----
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
      /root/miniforge3/envs/falcon/bin/python tools/repro/experiments/ln_variant.py

    ... ln_variant.py --variant ln_c --json /tmp/ln_c.json
    ... ln_variant.py --wav /path/to.wav --allow-dirty

退出码：0 正常；2 权重/配置缺失；3 权重与模型结构不匹配。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------------
# 路径 / 常量
# ----------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]          # tools/repro/experiments/x.py
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
DEFAULT_CONFIG = REPO / "configs" / "Falcon.yml"

# 输入优先级：in-domain LRS2-2Mix > RealSSA 实录 > 合成白噪。
# 说明：本 ckpt 在 video=0 时掩码趋零（输出比输入低约 50 dB），绝对音质不可用作
# 结论；本实验全部指标都是**同输入下的相对量**（变体间 / 窗长间），故仍成立。
LRS2_GLOB = "/root/data/falcon_lrs2_2mix/tt/mix/*.wav"   # 每段 2.000 s / 16 kHz
WAV_CANDIDATES = (
    Path("/root/data/RealSSA-Real/audio/inference/mid_indoor/internet_cafe/1.wav"),
)

SR = 16000
WIN = 256            # enc_dec_params.win
HOP = 128            # enc_dec_params.hop_length
DUR_2S = 2 * SR
DUR_4S = 4 * SR
TAIL_FRAMES = 32     # 被扰动的尾部帧数
DB_FLOOR = -240.0    # 相对响应下限（1e-24，即数值零）
RESP_EPS = 1e-24
RESP_THRESH_DB = -100.0
EARLY_FRAMES = 121   # "帧 0..120" 这一段

VARIANTS = ("as_shipped", "ln_c")


# ----------------------------------------------------------------------------
# 出处（provenance）
# ----------------------------------------------------------------------------

def git_provenance() -> dict:
    """记录被测量的代码树状态：并发改动下数字必须能归因到具体树。"""
    import subprocess

    def run(*a):
        try:
            return subprocess.run(a, cwd=str(REPO), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    dirty = [ln for ln in run("git", "status", "--porcelain").splitlines() if ln]
    model_dirty = [ln for ln in dirty if "look2hear/" in ln]
    return {
        "tree": str(REPO),
        "head": run("git", "rev-parse", "--short", "HEAD") or "unknown",
        "dirty_all": len(dirty),
        "dirty_model": model_dirty,
        "model_dirty": bool(model_dirty),
    }


# ----------------------------------------------------------------------------
# 模型构建
# ----------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, ckpt_path: Path):
    """按官方 config 建 `FalconSpeakerContrastive` 并灌入真实 `falcon.pt`。

    `falcon.pt` 结构：``{"state_dict": {audio_model.*, video_model.*}, "architecture",
    "epoch"}``；音频侧前缀 ``audio_model.`` 长 12 字符，剥掉后才能对上 Falcon 本体。
    这里做**严格覆盖断言**（525/525 key 必须一一对应），否则退出——绝不允许
    "部分参数悄悄保持随机初始化"污染测量。
    """
    import look2hear.models as M

    aconf = cfg["audionet"]
    model = getattr(M, aconf["audionet_name"])(**aconf["audionet_config"])

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        sys.exit(f"[FATAL] {ckpt_path} 里没有 'state_dict'：{list(ck)[:6]}")
    audio = {k[12:]: v for k, v in ck["state_dict"].items()
             if k.startswith("audio_model.")}
    if not audio:
        sys.exit(f"[FATAL] {ckpt_path} 里没有 audio_model.* 权重")

    mk, ak = set(model.state_dict()), set(audio)
    if mk != ak:
        sys.exit(
            "[FATAL] 权重与模型结构不匹配（拒绝部分加载）\n"
            f"  仅权重有 {len(ak - mk)}：{sorted(ak - mk)[:5]}\n"
            f"  仅模型有 {len(mk - ak)}：{sorted(mk - ak)[:5]}"
        )
    model.load_state_dict(audio, strict=True)
    model.eval()
    return model, ck.get("epoch"), sum(v.numel() for v in audio.values())


# ----------------------------------------------------------------------------
# 变体注入（进程内替换，不动仓库）
# ----------------------------------------------------------------------------

def swap_clnhw_to_ln_c(model):
    """把每个 `LayerNormalizationHW` 换成逐 (t,f) 的通道 LayerNorm。

    * 逐 (t,f)：`LayerNormalization4D((C,1))` → ``dim=(1,)``，归一化只跨 C，
      帧 t 的统计不含任何其它帧 —— 这正是要恢复的"因果归一化"。
    * C 不变，故所有下游形状完全一致，可直接前向。
    * 新模块 gamma=1/beta=0，**不**拷贝旧 affine —— 本实验测的就是"不重训直接换"。
    """
    from look2hear.layers.normalizations import (
        LayerNormalization4D,
        LayerNormalizationHW,
    )

    targets = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, LayerNormalizationHW):
                targets.append((parent, child_name, child))

    for parent, child_name, old in targets:
        channels = old.global_layer_norm.num_channels
        new = LayerNormalization4D((channels, 1))
        new.gamma.data.fill_(1.0)
        new.beta.data.fill_(0.0)
        new.requires_grad_(False)
        setattr(parent, child_name, new)

    left = sum(1 for m in model.modules() if isinstance(m, LayerNormalizationHW))
    if left:
        sys.exit(f"[FATAL] 替换后仍有 {left} 个 LayerNormalizationHW")
    n_ln4d = sum(1 for m in model.modules() if isinstance(m, LayerNormalization4D))
    return len(targets), n_ln4d


# ----------------------------------------------------------------------------
# 输入
# ----------------------------------------------------------------------------

def make_inputs(wav_path, seed: int):
    """返回 (x4s, 来源说明)。优先 in-domain LRS2-2Mix（拼两段 2 s），退化到实录/白噪。"""
    import glob

    import soundfile as sf

    if wav_path:
        cand = Path(wav_path)
        if not cand.exists():
            sys.exit(f"[FATAL] --wav 不存在：{cand}")
        segs = [cand]
    else:
        segs = [Path(p) for p in sorted(glob.glob(LRS2_GLOB))[:2]]
        if not segs:
            segs = [c for c in WAV_CANDIDATES if c.exists()][:1]

    if segs:
        xs, names = [], []
        for p in segs:
            data, sr = sf.read(str(p), frames=DUR_2S, dtype="float32", always_2d=True)
            if sr != SR:
                sys.exit(f"[FATAL] {p} 采样率 {sr} != {SR}")
            v = torch.from_numpy(data.mean(axis=1))[:DUR_2S]
            if v.numel() < DUR_2S:
                v = torch.cat([v, v.new_zeros(DUR_2S - v.numel())])
            xs.append(v)
            names.append(p.name)
        x = torch.cat(xs)[:DUR_4S]
        if x.numel() < DUR_4S:
            x = torch.cat([x, x.new_zeros(DUR_4S - x.numel())])
        note = "in-domain `" + "` + `".join(names) + "`（各取前 2 s 拼接成 4 s）"
        return x.view(1, 1, DUR_4S).contiguous(), note

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, DUR_4S, generator=g) * 0.1
    return x.contiguous(), "**合成白噪**（未找到真实 wav；电平/rms 比仅供参考）"


# ----------------------------------------------------------------------------
# 前向与度量
# ----------------------------------------------------------------------------

@torch.no_grad()
def forward_parts(model, wav: torch.Tensor) -> dict:
    emb = model.encoder(wav)                        # (B,N,T,F)
    aud = model.audio_bottleneck(emb)               # (B,C,T,F)
    vid = model.video_bottleneck(
        wav.new_zeros(wav.size(0), model.pretrained_vout_chan, 8)
    )
    sep = model.separator(aud, vid)                 # (B,C,T,F) <- cLNhw 在这里面
    masks = model.mask_generator(sep, emb)
    out = model.decoder(masks, wav.shape)           # (B,n_src,L)
    return {"sep": sep, "audio": out}


def frame_db_global(y_ref: torch.Tensor, y_pert: torch.Tensor):
    """逐帧相对响应 (dB)，基准是**全局**输出能量：

        db[t] = 10*log10( mean_{B,C,F}(Δy_t²) / mean_all(y_ref²) )

    这样近静音帧不会把数值残差放大成假响应。
    """
    num = (y_pert - y_ref).pow(2).mean(dim=(0, 1, 3))
    den = y_ref.pow(2).mean().clamp_min(1e-30)
    return 10.0 * torch.log10((num / den).clamp_min(RESP_EPS))


def sisnr_db(est: torch.Tensor, ref: torch.Tensor) -> float:
    """SI-SNR(dB)，与 `Falcon._sisnr` 同式（去均值→投影→10log10），独立实现。"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    proj = dot * ref / ref.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    noise = est - proj
    val = 10.0 * torch.log10(
        proj.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(1e-8) + 1e-8
    )
    return float(val.mean())


def rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def measure(model, variant: str, x4s, x2s, x2s_pert, p_start: int) -> dict:
    n_swapped, n_ln4d = (0, None)
    if variant == "ln_c":
        n_swapped, n_ln4d = swap_clnhw_to_ln_c(model)

    # ---- metric 1: 尾部扰动传播（separator 特征图，逐帧 dB）----
    ref = forward_parts(model, x2s)["sep"]
    pert = forward_parts(model, x2s_pert)["sep"]
    db = frame_db_global(ref, pert)
    early = db[:EARLY_FRAMES]
    resp = torch.nonzero(db > RESP_THRESH_DB).flatten()
    first_resp = int(resp[0]) if resp.numel() else -1
    last_resp = int(resp[-1]) if resp.numel() else -1

    # ---- metric 2: 窗长敏感性（4 s 窗的前 2 s  vs  直接 2 s 窗）----
    out4 = forward_parts(model, x4s)["audio"]
    out2 = forward_parts(model, x2s)["audio"]
    sisnr_2s_vs_4s = sisnr_db(out2[:, 0, :], out4[..., :DUR_2S][:, 0, :])

    return {
        "variant": variant,
        "n_swapped": n_swapped,
        "n_ln4d": n_ln4d,
        "n_frames": int(db.numel()),
        "p_start_frame": p_start,
        "early_min_db": float(early.min()),
        "early_max_db": float(early.max()),
        "all_min_db": float(db.min()),
        "first_resp_frame": first_resp,
        "last_resp_frame": last_resp,
        "lookback_frames": p_start - first_resp if first_resp >= 0 else -1,
        "sisnr_2s_vs_4s_db": sisnr_2s_vs_4s,
        "out2_rms": rms(out2),
        "out4_rms": rms(out4),
        "sep_rms": rms(ref),
    }


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------

def render(rows, meta) -> str:
    L = ["## ln_variant: `cLNhw` vs 逐 (t,f) 通道 LayerNorm（不重训替换）\n"]
    info = meta.get("provenance") or {}
    if info:
        flag = ("**dirty**（数字属于当前工作树，不属于 HEAD）"
                if info.get("model_dirty") else "clean")
        extra = (f"；`look2hear/` 改动 {len(info['dirty_model'])} 个文件"
                 if info.get("model_dirty") else "")
        L.append(f"代码树 `{info.get('tree')}` @ `{info.get('head')}` — {flag}{extra}\n")
    L.append(
        f"ckpt `{meta['ckpt']}`（epoch {meta['epoch']}，{meta['params']:,} 参数，"
        f"{meta['n_swapped_total']} 个 `cLNhw` 实例）  \n"
        f"输入：{meta['input_note']}，rms {meta['x_rms']:.4f}  \n"
        f"视频输入：**全零**（无真实口型；两变体完全相同，故只做相对比较）  \n"
        f"扰动：尾部 {TAIL_FRAMES} 帧（≈{TAIL_FRAMES * HOP / SR * 1000:.0f} ms，"
        f"幅度 {meta['pert_amp']:.4f}，首个受影响帧 ≈{meta['p_start']}）\n"
    )
    L.append(
        "| 变体 | 帧 0..120 最小响应 (dB) | 首响应帧 | 回看帧数 | "
        "2s-vs-4s SI-SNR (dB) | 输出 rms | rms 比 |"
    )
    L.append("|---|---|---|---|---|---|---|")
    base_rms = next((r["out2_rms"] for r in rows if r["variant"] == "as_shipped"), None)
    for r in rows:
        rr = f"{r['out2_rms'] / base_rms:.3f}×" if base_rms else "—"
        L.append(
            f"| `{r['variant']}` | {r['early_min_db']:.1f} | {r['first_resp_frame']} "
            f"| {r['lookback_frames']} | {r['sisnr_2s_vs_4s_db']:.2f} "
            f"| {r['out2_rms']:.5f} | {rr} |"
        )
    L.append("")
    return "\n".join(L)


def conclusion(rows) -> str:
    """一行结论：因果是否恢复、lookahead 是否回到几何界、代价多少。"""
    by = {r["variant"]: r for r in rows}
    if len(by) < 2:
        r = rows[0]
        return (f"[单变体] `{r['variant']}`：帧 0..120 最小响应 {r['early_min_db']:.1f} dB，"
                f"首响应帧 {r['first_resp_frame']}（回看 {r['lookback_frames']} 帧），"
                f"2s-vs-4s SI-SNR {r['sisnr_2s_vs_4s_db']:.2f} dB。")
    a, b = by["as_shipped"], by["ln_c"]
    causal = b["early_min_db"] <= DB_FLOOR + 1 and a["early_min_db"] > -80.0
    gain = b["sisnr_2s_vs_4s_db"] - a["sisnr_2s_vs_4s_db"]
    ratio = b["out2_rms"] / a["out2_rms"] if a["out2_rms"] else float("nan")
    level_db = 20.0 * math.log10(ratio) if ratio and ratio > 0 else float("nan")
    return (
        f"`ln_c` **{'能' if causal else '不能'}把窗内因果恢复**：扰动区之前（帧 0..120）"
        f"的响应从 {a['early_min_db']:.1f} dB（as_shipped，处处响应）"
        f"变成 {b['early_min_db']:.1f} dB（-240 = 数值零）；"
        f"但回看仍有 {b['lookback_frames']} 帧（as_shipped {a['lookback_frames']}），"
        f"**没有**回到几何界（单帧扰动 p=125 测得 57 帧、尾部区块扰动 110 帧，回看量随扰动位置变化，机制尚未定位到具体算子）；"
        f"不重训的代价：输出电平 {level_db:+.1f} dB（{ratio:.1f}×），"
        f"2s-vs-4s SI-SNR {a['sisnr_2s_vs_4s_db']:.2f} → {b['sisnr_2s_vs_4s_db']:.2f} dB"
        f"（{gain:+.2f} dB，{'有改善' if gain > 0.5 else '未改善'}）。"
    )


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="cLNhw vs 逐 (t,f) 通道 LayerNorm 的因果/窗长对照（CPU，不训练）")
    ap.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--wav", type=Path, default=None, help="4 s 输入音频（缺省自动找 RealSSA）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="被测树有未提交改动时不告警（默认告警：数字无法归因到 HEAD）")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    prov = git_provenance()
    if prov["model_dirty"] and not args.allow_dirty:
        print(f"[WARN] 被测代码树不干净（look2hear/ 有未提交改动）：{prov['dirty_model']}\n"
              "       数字属于当前工作树，不能归因到 HEAD。要复跑 HEAD 基线请用 "
              "git archive 导出干净树再跑，或加 --allow-dirty 消音。", file=sys.stderr)

    for p in (args.ckpt, args.config):
        if not p.exists():
            print(f"[FATAL] 找不到 {p}", file=sys.stderr)
            return 2

    cfg = load_config(args.config)
    x4s, input_note = make_inputs(args.wav, args.seed)
    x2s = x4s[..., :DUR_2S].contiguous()

    # 尾部扰动：只加在最后 TAIL_FRAMES 帧覆盖的采样上
    tail = TAIL_FRAMES * HOP
    gg = torch.Generator().manual_seed(args.seed + 1)
    pert_amp = 0.5 * rms(x2s)                       # 相对输入电平，避免越界失真
    x2s_pert = x2s.clone()
    x2s_pert[..., -tail:] += torch.randn(1, 1, tail, generator=gg) * pert_amp
    first_sample = DUR_2S - tail
    p_start = math.ceil((first_sample - (WIN // 2 - 1)) / HOP)

    rows, meta_extra = [], {}
    picked = VARIANTS if args.variant == "both" else (args.variant,)
    for variant in picked:
        model, epoch, n_params = build_model(cfg, args.ckpt)
        if not meta_extra:
            meta_extra = {"ckpt": str(args.ckpt), "epoch": epoch, "params": n_params}
        rows.append(measure(model, variant, x4s, x2s, x2s_pert, p_start))
        del model

    meta = {**meta_extra, "seed": args.seed, "x_rms": rms(x4s), "pert_amp": pert_amp,
            "p_start": p_start, "input_note": input_note,
            "n_swapped_total": next((r["n_swapped"] for r in rows
                                     if r["variant"] == "ln_c"), 0),
            "provenance": prov}

    print(render(rows, meta))
    print("**结论**：" + conclusion(rows))
    if meta["n_swapped_total"]:
        print(f"\n替换明细：{meta['n_swapped_total']} 个 `LayerNormalizationHW` → "
              f"`LayerNormalization4D((C,1))`；模型内 `LayerNormalization4D` 现为 "
              f"{next(r['n_ln4d'] for r in rows if r['variant'] == 'ln_c')} 个。"
              "未改任何仓库前向代码（只在本进程内 setattr）。")

    if args.json:
        args.json.write_text(json.dumps({"meta": meta, "rows": rows},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
