#!/usr/bin/env python3
"""DNSMOS P.835 (SIG / BAK / OVRL) evaluation for Falcon reproduction.

实现选择：PyPI ``speechmos==0.0.1.1``（MIT License, Microsoft AEC/DNS/PLC Challenge organizers）。
该 wheel **自带** DNSMOS ONNX 模型（``speechmos/dnsmos_models/{sig_bak_ovr,model_v8}.onnx``），
推理时不需要网络 → 可离线运行；依赖 ``onnxruntime``（MIT）+ ``librosa``（ISC）+ ``numpy``。
- 模型：DNSMOS P.835（``sig_bak_ovr.onnx``）+ P.808（``model_v8.onnx``），9.01 s / 1 s hop 分段取均值（speechmos 内置逻辑）。
- 参考：DNSMOS P.835 paper <https://arxiv.org/abs/2110.01763>；模型来自 microsoft/DNS-Challenge。
- 本脚本只做“音频 → SIG/BAK/OVRL/P808”，不做分离；RealSSA-Real 管线见 ``eval_realssa_real.py``。

用法：
  # 合成音频自检（不读数据，验证 onnxruntime/模型可离线运行）
  python tools/repro/phase2/eval_dnsmos.py --synthetic

  # 单个/多个 wav（16 kHz；非 16k 用 --resample）
  python tools/repro/phase2/eval_dnsmos.py --input est.wav mix.wav --out dnsmos.csv
  python tools/repro/phase2/eval_dnsmos.py --input /root/data/est_dir --recursive --out dnsmos.csv

输出 CSV 列：file, duration_s, sig_mos, bak_mos, ovrl_mos, p808_mos, error
"""
from __future__ import annotations

import argparse
import csv
import glob as globlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TARGET_SR = 16000
DEFAULT_MODEL_TYPE = "dnsmos"  # speechmos: "dnsmos"（标准 P.835）或 "dnsmos_personalized"


# ---------------------------------------------------------------------------
# speechmos / onnxruntime 加载（延迟 import，便于给出清晰安装提示）
# ---------------------------------------------------------------------------
def _speechmos():
    try:
        from speechmos import dnsmos  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "缺少 offline DNSMOS 依赖（speechmos/onnxruntime）。安装：\n"
            "  /root/miniforge3/envs/falcon/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \\\n"
            "      'onnxruntime==1.31.0' --no-deps 'speechmos==0.0.1.1'\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc
    return dnsmos


def model_info() -> dict:
    """返回 speechmos 自带 DNSMOS 模型文件、大小与 sha256（用于报告可复现性）。"""
    dnsmos = _speechmos()
    mdir = Path(dnsmos.__file__).resolve().parent / "dnsmos_models"
    info = {"model_dir": str(mdir), "files": {}}
    for p in sorted(mdir.glob("*.onnx")):
        info["files"][p.name] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    return info


def score_array(audio: np.ndarray, sr: int = TARGET_SR, model_type: str = DEFAULT_MODEL_TYPE) -> dict:
    """对 1-D float32 音频做 DNSMOS；返回 {sig_mos,bak_mos,ovrl_mos,p808_mos}。"""
    dnsmos = _speechmos()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"score_array 需要 1-D 单声道音频，得到 shape={arr.shape}")
    if sr != TARGET_SR:
        raise ValueError(f"DNSMOS 模型要求 {TARGET_SR} Hz，得到 {sr}")
    if not np.isfinite(arr).all():
        raise ValueError("音频含 NaN/Inf")
    if arr.size == 0:
        raise ValueError("空音频")
    # speechmos 要求 float 值在 [-1, 1]；超出时按峰值归一化（并返回提示由调用方记录）
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        arr = arr / peak
    result = dnsmos.run(arr, sr, model_type=model_type)
    return {
        "sig_mos": float(result["sig_mos"]),
        "bak_mos": float(result["bak_mos"]),
        "ovrl_mos": float(result["ovrl_mos"]),
        "p808_mos": float(result["p808_mos"]),
    }


def load_audio(path: str, resample: bool = False) -> tuple[np.ndarray, int, float]:
    """读 wav → (mono float32, sr, duration)。stereo 取均值；非 16k 时按需重采样。"""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if sr != TARGET_SR:
        if not resample:
            raise ValueError(f"{path}: sr={sr} != {TARGET_SR}（可加 --resample）")
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        sr = TARGET_SR
    return data, sr, len(data) / TARGET_SR


def score_file(path: str, model_type: str = DEFAULT_MODEL_TYPE, resample: bool = False) -> dict:
    audio, sr, dur = load_audio(path, resample=resample)
    started = time.time()
    scores = score_array(audio, sr, model_type=model_type)
    scores.update({"file": path, "duration_s": round(dur, 3), "elapsed_s": round(time.time() - started, 2)})
    return scores


# ---------------------------------------------------------------------------
# 合成音频自检：不依赖数据，验证离线 ONNX 推理链路
# ---------------------------------------------------------------------------
def synthetic_cases(sr: int = TARGET_SR, seconds: float = 2.0):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(0)
    # 1) “类语音”谐波 + 幅度包络（不是真语音，只用于验证 API 输出有限且在 [1,5]）
    env = 0.5 * (1 + np.sin(2 * np.pi * 1.7 * t)) + 0.1
    tone = 0.25 * env * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t))
    # 2) 谐波 + 白噪声（低 BAK 预期）
    noisy = tone + 0.05 * rng.standard_normal(len(t))
    # 3) 静音
    silence = np.zeros(len(t), dtype=np.float32)
    # 4) 纯噪声
    noise = (0.1 * rng.standard_normal(len(t))).astype(np.float32)
    return [("synthetic_harmonic", tone.astype(np.float32)),
            ("synthetic_harmonic_noise", noisy.astype(np.float32)),
            ("synthetic_silence", silence),
            ("synthetic_noise", noise)]


def run_synthetic(model_type: str, log_path: str | None = None) -> int:
    print(f"[dnsmos] offline backend = speechmos {_version('speechmos')} / onnxruntime {_version('onnxruntime')}")
    print("[dnsmos] 注意：以下为合成信号，只验证离线 ONNX 链路；不代表任何数据集/论文数值。")
    rows, ok = [], True
    for name, audio in synthetic_cases():
        t0 = time.time()
        scores = score_array(audio, TARGET_SR, model_type=model_type)
        rows.append({"case": name, **{k: round(v, 4) for k, v in scores.items()}, "elapsed_s": round(time.time() - t0, 2)})
        finite = all(np.isfinite(scores[k]) and 1.0 <= scores[k] <= 5.0 for k in scores)
        ok = ok and finite
        print(f"  {name:24s} SIG={scores['sig_mos']:.3f} BAK={scores['bak_mos']:.3f} "
              f"OVRL={scores['ovrl_mos']:.3f} P808={scores['p808_mos']:.3f} {'OK' if finite else 'OUT-OF-RANGE'}")
    info = {"backend": "speechmos", "speechmos_version": _version("speechmos"),
            "onnxruntime_version": _version("onnxruntime"), "model_type": model_type,
            "models": model_info(), "cases": rows, "all_finite_in_range": bool(ok)}
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[dnsmos] JSON evidence -> {log_path}")
    print(f"[dnsmos] synthetic smoke: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _version(pkg: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(pkg)
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def expand_inputs(items: list[str], recursive: bool) -> list[str]:
    files: list[str] = []
    for item in items:
        p = Path(item).expanduser()
        if p.is_dir():
            pattern = "**/*.wav" if recursive else "*.wav"
            files.extend(sorted(str(x) for x in p.glob(pattern)))
        elif p.is_file():
            files.append(str(p))
        else:
            matched = sorted(globlib.glob(item, recursive=recursive))
            if not matched:
                raise SystemExit(f"输入不存在或无 wav 匹配：{item}")
            files.extend(matched)
    # 去重保序
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DNSMOS P.835 offline evaluation (speechmos/ONNX)")
    parser.add_argument("--input", nargs="+", default=[], help="wav 文件/目录/glob（可多个）")
    parser.add_argument("--recursive", action="store_true", help="目录输入时递归 *.wav")
    parser.add_argument("--out", default="", help="逐文件 CSV 输出路径（默认打印到 stdout）")
    parser.add_argument("--summary-json", default="", help="汇总 JSON（版本/模型 hash/均值）")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE, choices=["dnsmos", "dnsmos_personalized"],
                        help="speechmos 模型类型；论文 RealSSA 无 personalized 信息，默认标准 dnsmos")
    parser.add_argument("--resample", action="store_true", help="非 16 kHz 时用 librosa 重采样（默认报错提示）")
    parser.add_argument("--synthetic", action="store_true", help="运行合成音频自检并退出")
    parser.add_argument("--synthetic-log", default="", help="--synthetic 的 JSON 证据输出路径")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 个文件（0=全部）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.synthetic:
        return run_synthetic(args.model_type, args.synthetic_log or None)

    if not args.input:
        parser.error("需要 --input（或 --synthetic）")
    files = expand_inputs(args.input, args.recursive)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("[dnsmos] 没有可处理的 wav", file=sys.stderr)
        return 2

    print(f"[dnsmos] backend=speechmos {_version('speechmos')} / onnxruntime {_version('onnxruntime')} "
          f"| files={len(files)} | model_type={args.model_type}")
    # 预热一次 ONNX session（首次调用较慢）
    try:
        score_array(np.zeros(TARGET_SR, dtype=np.float32), TARGET_SR, args.model_type)
    except Exception as exc:
        print(f"[dnsmos] 预热失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows, errors = [], 0
    for i, path in enumerate(files, 1):
        try:
            row = score_file(path, args.model_type, resample=args.resample)
            rows.append(row)
            if not args.quiet:
                print(f"  [{i}/{len(files)}] {Path(path).name}  {row['duration_s']:.2f}s  "
                      f"SIG={row['sig_mos']:.3f} BAK={row['bak_mos']:.3f} OVRL={row['ovrl_mos']:.3f} "
                      f"P808={row['p808_mos']:.3f}")
        except Exception as exc:
            errors += 1
            rows.append({"file": path, "duration_s": "", "sig_mos": "", "bak_mos": "", "ovrl_mos": "",
                         "p808_mos": "", "elapsed_s": "", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i}/{len(files)}] {path}  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    cols = ["file", "duration_s", "sig_mos", "bak_mos", "ovrl_mos", "p808_mos", "elapsed_s", "error"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        print(f"[dnsmos] CSV -> {args.out}")

    valid = [r for r in rows if not r.get("error")]
    summary = {
        "backend": "speechmos", "speechmos_version": _version("speechmos"),
        "onnxruntime_version": _version("onnxruntime"), "model_type": args.model_type,
        "num_files": len(rows), "num_ok": len(valid), "num_errors": errors,
        "mean": {k: (round(float(np.mean([r[k] for r in valid])), 4) if valid else None)
                 for k in ["sig_mos", "bak_mos", "ovrl_mos", "p808_mos"]},
        "std": {k: (round(float(np.std([r[k] for r in valid])), 4) if valid else None)
                for k in ["sig_mos", "bak_mos", "ovrl_mos", "p808_mos"]},
        "models": model_info(),
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[dnsmos] summary JSON -> {args.summary_json}")
    if valid:
        m = summary["mean"]
        print(f"[dnsmos] N={len(valid)} mean SIG={m['sig_mos']} BAK={m['bak_mos']} OVRL={m['ovrl_mos']} P808={m['p808_mos']}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
