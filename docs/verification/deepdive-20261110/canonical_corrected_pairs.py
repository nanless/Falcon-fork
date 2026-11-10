#!/usr/bin/env python3
"""streaming_canonical.py — Studio 拼接流式 vs 整段前向的仲裁脚本（Falcon small, falcon.pt, CPU）。

来源
----
本文件逐行复制自 2026-11-08 深挖轮的仲裁脚本
`falcon-deepdive-20261108/_scratch/R6/canonical.py`（149 行，sha256 31d0877ebb59...）。
**计算逻辑未改动**，只做了三处非数值改动：
  1. 补 shebang 与这段来源 / 用途说明；
  2. 把原来的 `sys.path.insert(0, "/tmp/w212"); import w212_common as C`
     换成同目录随仓库携带的 `_falcon_cpu_common.py`（内容等同原 `w212_common.py`，
     只把仓库路径改为按 `__file__` 推导），使脚本不依赖 `/tmp` 草稿目录；
  3. 加 `--json` / `--seconds` 参数；默认仍写 `/tmp/w213/out_<N>s.json`。

用途
----
回答「Studio 的拼接流式比整段一次前向差多少」这个口径问题。Studio 把长音频切成 2 s 窗、
每窗只播最新 0.5 s、并在窗尾留 1.28 s guard；本脚本在同一段流上给六组量：
  * `studio`          —— 从 t=0 起播（含冷启动首窗，首窗内前 1.42 s 是静音）
  * `steady_metrics_excl_cold_start` —— 剔除第 1 个 hop
  * `hard_splice`     —— 与 10 ms join 交叉淡化对照
  * `ref_tiling`      —— 2 s 窗 / 2 s hop 的参考平铺
  * `guard_stitch`    —— guard 80 / 160 / 320 / 456 / 640 ms 扫描
  * `align` / `align_pairwise` —— 换窗对齐 ±250 ms 的整段一致性与两两差
指标为 SI-SNR、Pearson r、峰值 / RMS 相对偏差、rms 比，外加 per-chunk SI-SNR 分布。

用法
----
    python tools/repro/experiments/streaming_canonical.py                    # 6 s 流
    python tools/repro/experiments/streaming_canonical.py --seconds 4        # 4 s 流
    python tools/repro/experiments/streaming_canonical.py --json out.json    # 显式落盘

约束
----
纯 CPU（`torch.set_num_threads(4)`）、`eval()` + `no_grad()`、不训练、不写仓库（除非
`--json` 显式指到仓库内）。6 s 流实测约 89 s；依赖 `/root/data/falcon_lrs2_2mix/tt` 与
`/root/.cache/falcon_weights/falcon.pt`（sha256 a1888b5a…）。
"""
import os, sys, json, glob, time, argparse
from pathlib import Path
import numpy as np, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _falcon_cpu_common as C  # 见同目录该文件：原 /tmp/w212/w212_common.py 的随仓库副本
torch.set_num_threads(4)
SR, HOP, WIN, GUARD, FADE, LEFT = 16000, 8000, 32000, 1280, 320, 22720
ROOT, OUT = "/root/data/falcon_lrs2_2mix/tt", "/tmp/w213"
def load_stream(n, fade=160):
    mixes = sorted(glob.glob(f"{ROOT}/mix/*.wav"))
    groups = {}
    for p in mixes:
        groups.setdefault("_".join(os.path.basename(p).split("_")[:2]), []).append(p)
    sid, files = max(groups.items(), key=lambda kv: len(kv[1]))
    files = files[:n]
    parts = [sf.read(p)[0].astype(np.float32) for p in files]
    x, joins = parts[0].copy(), []
    for c in parts[1:]:
        r = np.linspace(0, 1, fade + 2, dtype=np.float32)[1:-1]
        x = np.concatenate([x[:-fade], x[-fade:] * (1 - r) + c[:fade] * r, c[fade:]])
        joins.append(len(x) - (len(c) - fade))
    pad = (-len(x)) % HOP
    x = np.concatenate([x, np.zeros(pad, np.float32)])
    mt = torch.load(f"{ROOT}/mouths/{sid}.pt", map_location="cpu", weights_only=True).float()
    if mt.ndim == 4:
        mt = (mt * torch.tensor([.299, .587, .114]).view(1, 1, 1, 3)).sum(-1)
    track = mt.numpy().astype(np.uint8)
    track = np.concatenate([track] * n) if len(track) < 50 * n else track[:50 * n]
    return x, track, sid, files, joins, pad
_ap = argparse.ArgumentParser(
    description="Studio 拼接流式 vs 整段前向（CPU 仲裁）：SI-SNR / r / 偏差 / guard 与对齐扫描")
_ap.add_argument("seconds", nargs="?", type=int, default=6,
                 help="流长度，秒（深挖轮用 6 与 4；片段数 = seconds//2，默认 6）")
_ap.add_argument("--seconds", dest="seconds_opt", type=int, default=None,
                 help="同 positional，显式写法")
_ap.add_argument("--json", type=Path, default=None,
                 help="结果 JSON 路径；缺省 /tmp/w213/out_<N>s.json")
_args = _ap.parse_args()
a, v, cfg = C.build(verbose=False)
N = _args.seconds_opt if _args.seconds_opt is not None else _args.seconds
x, track, sid, files, joins, pad = load_stream(N // 2)
def f0_of(start):
    """index of the first 25 fps crop whose centre (n+0.5)*640 lies inside [start, start+2 s)."""
    return int(np.ceil(start / 640 - 0.5))
EC = {}
def emb_at(f0, count):
    key = (f0, count)
    if key not in EC:
        EC[key] = C.video_embed(v, track[np.clip(f0 + np.arange(count), 0, len(track) - 1)])
    return EC[key]
PC = {}
def pad_w(start, length):
    out = np.zeros(length, np.float32)
    lo, hi = max(0, start), min(len(x), start + length)
    if hi > lo:
        out[lo - start:hi - start] = x[lo:hi]
    return out
def win_pred(start, length):
    key = (start, length)
    if key not in PC:
        PC[key] = C.run(a, emb_at(f0_of(start), length // 640), pad_w(start, length))
    return PC[key]
def sisnr(e, r, eps=1e-8):
    e = np.asarray(e, np.float64) - np.mean(e)
    r = np.asarray(r, np.float64) - np.mean(r)
    al = (e * r).sum() / ((r * r).sum() + eps)
    t, nz = al * r, e - al * r
    return float(10 * np.log10(((t * t).sum() + eps) / ((nz * nz).sum() + eps)))
def mets(e, r):
    pk, rm = float(np.abs(r).max()) + 1e-12, float(np.sqrt((r * r).mean())) + 1e-12
    return {"si_snr": round(sisnr(e, r), 2), "r": round(float(np.corrcoef(e, r)[0, 1]), 4),
            "maxdev_peak": round(float(np.abs(e - r).max() / pk), 3),
            "maxdev_rms": round(float(np.abs(e - r).max() / rm), 2),
            "rms_ratio": round(float(np.sqrt((e * e).mean()) / rm), 3)}
def stitch(starts, guard, fade):
    out, prev_p, prev_s = {}, None, None
    for s in starts:
        end = s + HOP + guard
        p = win_pred(end - WIN, WIN)
        chunk = p[WIN - guard - HOP:WIN - guard].copy()
        if fade and guard == GUARD and prev_p is not None and s == prev_s + HOP:
            r = np.linspace(0, 1, FADE, dtype=np.float32)
            chunk[:FADE] = prev_p[WIN - guard:WIN - guard + FADE] * (1 - r) + chunk[:FADE] * r
        prev_p, prev_s = p, s
        out[s] = chunk
    return out
def assemble(chunks, lo, hi, clen=HOP):
    y, cov = np.zeros(hi - lo, np.float32), np.zeros(hi - lo, bool)
    for s, c in chunks.items():
        a0, b0 = max(s, lo), min(s + clen, hi)
        if b0 > a0:
            y[a0 - lo:b0 - lo] = c[a0 - s:b0 - s]
            cov[a0 - lo:b0 - lo] = True
    return y, bool(cov.all())
whole = C.run(a, emb_at(0, len(track)), x)
res = {"stream_seconds": round(len(x) / SR, 3), "samples": len(x), "source": sid,
       "clips": [os.path.basename(f) for f in files], "crossfade_ms_at_join": 10,
       "joins": joins, "zero_tail": pad}
t0 = time.time()
starts = [k * HOP for k in range(0, (len(x) - GUARD) // HOP)]      # first end = 9280 = Studio
st = stitch(starts, GUARD, FADE)
lo, hi = 0, starts[-1] + HOP
ys, full = assemble(st, lo, hi)
pc = [sisnr(st[s], whole[s:s + HOP]) for s in starts]
st_steady = [s for s in starts if s >= HOP]
yss, _ = assemble({s: st[s] for s in st_steady}, HOP, hi)
res["studio"] = {"range": [lo, hi], "chunks": len(starts), "covered": full,
                 "metrics": mets(ys, whole[lo:hi]),
                 "steady_metrics_excl_cold_start": mets(yss, whole[HOP:hi]),
                 "per_chunk_si_snr": [round(v, 1) for v in pc],
                 "per_chunk_mean": round(float(np.mean(pc)), 2),
                 "per_chunk_median": round(float(np.median(pc)), 2)}
yhard, _ = assemble(stitch(starts, GUARD, 0), lo, hi)
res["studio"]["hard_splice"] = mets(yhard, whole[lo:hi])
print("[studio]", res["studio"]["metrics"], res["studio"]["per_chunk_si_snr"], flush=True)
refs = list(range(0, len(x) - WIN + 1, WIN))
ref = {s: win_pred(s, WIN) for s in refs}
rng = min(hi, refs[-1] + WIN)
yref, _ = assemble(ref, 0, rng, WIN)
res["ref_tiling"] = {"range": [0, rng], "windows": len(refs),
                     "vs_whole": mets(yref, whole[:rng]), "stitched_vs_ref": mets(ys[:rng], yref)}
print("[tiling]", res["ref_tiling"], flush=True)
gs = [1280, 2560, 5120, 7296, 10240]
common = min(((len(x) - g) // HOP) * HOP for g in gs)
res["guard_stitch"] = []
for g in gs:
    kk = [k * HOP for k in range(0, (len(x) - g) // HOP)]
    yg, ok = assemble(stitch(kk, g, 0), 0, common)
    res["guard_stitch"].append({"guard_ms": round(g / 16, 1), "chunks": common // HOP,
                                "covered": ok, **mets(yg, whole[:common])})
    print("[guard-stitch]", res["guard_stitch"][-1], flush=True)
res["guard_lookahead"] = []
for g in gs:
    e, r = [], []
    for s in (40000, 48000):
        p = C.run(a, emb_at(f0_of(s - LEFT), (LEFT + HOP + g) // 640), pad_w(s - LEFT, LEFT + HOP + g))
        e.append(p[LEFT:LEFT + HOP]); r.append(whole[s:s + HOP])
    res["guard_lookahead"].append({"guard_ms": round(g / 16, 1), "window_samples": LEFT + HOP + g,
                                   **mets(np.concatenate(e), np.concatenate(r))})
    print("[guard-lookahead]", res["guard_lookahead"][-1], flush=True)
shifts = [-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000]
alo, ahi = 16000, 64000
streams, res["align"] = {}, []
for d in shifts:
    kk = [s for s in (d + k * HOP for k in range(-2, 12)) if s + HOP > alo and s < ahi]
    yd, ok = assemble(stitch(kk, GUARD, 0), alo, ahi)
    streams[d] = yd
    res["align"].append({"shift_ms": round(d * 1000 / SR, 1), "covered": ok, **mets(yd, whole[alo:ahi])})
    print("[align]", res["align"][-1], flush=True)
pairs = [(sisnr(streams[i], streams[j]), i, j) for i in shifts for j in shifts if i < j]
distinct = [p for p in pairs if (p[1] - p[2]) % HOP != 0]
res["align_pairwise"] = {"min": round(min(distinct)[0], 2), "min_pair_ms": [min(distinct)[1] * 1000 // SR, min(distinct)[2] * 1000 // SR],
                         "max": round(max(distinct)[0], 2), "max_pair_ms": [max(distinct)[1] * 1000 // SR, max(distinct)[2] * 1000 // SR],
                         "note": "pairs whose shifts differ by an integer multiple of 0.5 s are the same grid phase and excluded"}
res["elapsed_s"] = round(time.time() - t0, 1)
_out = Path(_args.json) if _args.json else Path(OUT) / f"out_{N}s.json"
_out.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(_out, "w"), indent=1)
print(f"[json] {_out}", flush=True)
print("[done]", res["align_pairwise"], res["elapsed_s"], flush=True)
