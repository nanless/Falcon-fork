#!/usr/bin/env python3
"""RealSSA-Real evaluation pipeline for Falcon (reference-free: DNSMOS P.835).

Dataset: `JusperLee/RealSSA-Real`（gated/manual；本机申请已提交、待批，数据文件当前 403）。
按官方数据卡约定读取 `inference` view 的 54 个 segment：
  - 音频：`audio/inference/**/*.wav`，native **stereo PCM16 16 kHz**；`channel='mean'|'0'|'1'`；
  - 视觉：`mouth/**/*.npz`，`data: uint8 [T,96,96]`（2 s 窗口外，整段 25 fps）；
  - 对齐：`align_tail=True` → `T=min(mouth_frames, floor(audio_samples/640), [landmark/face frames])`，
    只保留前 T 帧视觉与 T*640 音频样本（最多丢 ≤70 ms）；`align_tail=False` 保留原生长度（本脚本为模型输入仍取 min，
    并在 CSV 的 discarded_samples 记录差异）；
  - 参考：**没有干净 target → 不能算 SI-SDR**；本脚本只输出 DNSMOS（SIG/BAK/OVRL/P808），可选 mixture 基线。
  - 官方 loader 可用时（`<data>/scripts/load_inference.py`）优先用其 `load_sample/read_recordings`；
    无数据/无 loader 时用本脚本的 fallback loader（metadata jsonl / 目录扫描），并在报告中标注。

分离：`studio/falcon_runtime.FalconRuntime.from_pretrained(...).load(device)` + `studio/infer_file.extract`（整段推理）。
DNSMOS：`tools/repro/phase2/eval_dnsmos.py` 的 `score_array`（speechmos/ONNX，离线）。

用法：
  # 无数据时：清晰提示 + 用 LRS2-2Mix 做接口冒烟（不伪造 RealSSA 结果）
  python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real --dry-run
  python tools/repro/phase2/eval_realssa_real.py --smoke-lrs2 /root/data/lrs2_2mix/lrs2_rebuild --limit 2

  # 审批通过、数据下载后：
  python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real \
      --channel mean --device cuda --out-dir /root/data/falcon_eval/realssa_real_mean
  python tools/repro/phase2/eval_realssa_real.py ... --channel 0 --limit 5 --dry-run   # 先看清单

安全/合规：不写任何人脸/音频到仓库；默认不保存分离音频（--save-estimates 才保存到 out-dir，且应在获批范围内使用）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STUDIO = REPO / "studio"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(STUDIO) not in sys.path:
    sys.path.insert(0, str(STUDIO))

from eval_dnsmos import score_array, _version  # noqa: E402

SAMPLE_RATE = 16000
FRAME_HOP = 640  # 16000 / 25 fps
CANDIDATE_DATA = Path("/root/data/realssa_real")
GUIDE = "docs/DATA_ACCESS_GUIDE.md"


# ---------------------------------------------------------------------------
# 段发现：官方 loader 优先，其次 metadata jsonl，最后目录扫描
# ---------------------------------------------------------------------------
def _load_official_module(root: Path):
    script = root / "scripts" / "load_inference.py"
    if not script.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("realssa_load_inference", str(script))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 官方 load_inference.py 导入失败（将用 fallback loader）：{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


def _segments_from_official(module, root: Path) -> list[dict]:
    """用官方 read_recordings()/segment_ids 构建 segment 清单（不加载数据）。"""
    segments: list[dict] = []
    rec = module.read_recordings(str(root))
    if not isinstance(rec, dict):
        raise ValueError("read_recordings() 未返回 dict")
    for key, record in rec.items():
        ids = list(record.get("segment_ids") or []) if isinstance(record, dict) else []
        if not ids:
            continue
        # part_order 是“顺序索引”而不是 segment id 本身；按它重排 ids（官方 card 要求按 part_order 拼接）
        order = record.get("part_order") if isinstance(record, dict) else None
        if order and len(order) == len(ids):
            try:
                ids = [ids[int(k)] for k in order]
            except (ValueError, TypeError, IndexError):
                pass
        for sid in ids:
            segments.append({
                "inference_id": str(sid),
                "recording_id": str(key),
                "reference_text": record.get("reference_text") if isinstance(record, dict) else None,
                "scene": str(key).split("/")[0] if "/" in str(key) else "",
                "_record_key": str(key),
            })
    return segments


def _resolve_path(root: Path, value: str, subdirs: tuple[str, ...]) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    candidates = [root / value]
    for sub in subdirs:
        candidates.append(root / sub / value)
        candidates.append(root / sub / Path(value).name)
    for c in candidates:
        if c.exists():
            return c
    return None


def _segments_from_metadata(root: Path) -> list[dict]:
    """fallback：解析 metadata/inference_segments.jsonl（AudioFolder metadata）。"""
    candidates = [root / "metadata" / "inference_segments.jsonl",
                  root / "metadata.jsonl",
                  root / "metadata" / "metadata.jsonl"]
    manifest = next((c for c in candidates if c.exists()), None)
    if manifest is None:
        return []
    records: list[dict] = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            sid = obj.get("inference_id") or obj.get("segment_id") or obj.get("id")
            audio = obj.get("audio") or obj.get("audio_path") or obj.get("wav") or obj.get("file")
            if sid is None or not audio:
                continue
            mouth = obj.get("mouth") or obj.get("mouth_path") or obj.get("mouth_npz")
            audio_path = _resolve_path(root, str(audio), ("audio/inference", "audio", ""))
            if audio_path is None:
                continue
            if mouth:
                mouth_path = _resolve_path(root, str(mouth), ("mouth", "audio/inference", ""))
            else:
                rel = Path(str(audio))
                parts = rel.parts
                if "inference" in parts:
                    idx = parts.index("inference")
                    rel_after = Path(*parts[idx + 1:]).with_suffix(".npz")
                    mouth_path = _resolve_path(root, str(rel_after), ("mouth",))
                else:
                    mouth_path = _resolve_path(root, rel.with_suffix(".npz").name, ("mouth",))
            records.append({
                "inference_id": str(sid),
                "recording_id": str(obj.get("recording_id") or obj.get("recording") or ""),
                "scene": str(obj.get("scene") or ""),
                "audio_path": str(audio_path),
                "mouth_path": str(mouth_path) if mouth_path else "",
                "face_path": str(obj.get("face") or obj.get("face_path") or ""),
                "landmark_path": str(obj.get("landmark") or obj.get("landmark_path") or ""),
                "metadata": obj,
            })
    return records


def _segments_from_scan(root: Path) -> list[dict]:
    """无 metadata 时按目录扫描 audio/inference/**/*.wav，并镜像 mouth/ 路径。"""
    audio_root = root / "audio" / "inference"
    if not audio_root.exists():
        return []
    mouth_root = root / "mouth"
    records = []
    for wav in sorted(audio_root.rglob("*.wav")):
        rel = wav.relative_to(audio_root)
        mouth = mouth_root / rel.with_suffix(".npz")
        if not mouth.exists():
            hits = list((mouth_root.rglob(f"{wav.stem}.npz")))[:1]
            mouth = hits[0] if hits else mouth
        records.append({
            "inference_id": wav.stem,
            "recording_id": "",
            "scene": rel.parts[0] if len(rel.parts) > 1 else "",
            "audio_path": str(wav),
            "mouth_path": str(mouth) if mouth.exists() else "",
            "face_path": "",
            "landmark_path": "",
            "metadata": {"source": "scan"},
        })
    return records


def discover_segments(root: Path) -> tuple[list[dict], str]:
    """返回 (segments, loader_kind)。"""
    if not root.exists():
        return [], "missing"
    module = _load_official_module(root)
    if module is not None and hasattr(module, "read_recordings"):
        try:
            segs = _segments_from_official(module, root)
            if segs:
                return segs, "official:read_recordings"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 官方 read_recordings 失败，回退：{type(exc).__name__}: {exc}", file=sys.stderr)
    segs = _segments_from_metadata(root)
    if segs:
        return segs, "fallback:metadata-jsonl"
    segs = _segments_from_scan(root)
    if segs:
        return segs, "fallback:dir-scan"
    return [], "empty"


# ---------------------------------------------------------------------------
# 数据读取与对齐
# ---------------------------------------------------------------------------
def _mono_channel(audio: np.ndarray, channel: str, where: str) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"{where}: 音频 ndim={audio.ndim}，不支持")
    if channel == "mean":
        return audio.mean(axis=1)
    if channel in ("0", "1"):
        idx = int(channel)
        if audio.shape[1] <= idx:
            raise ValueError(f"{where}: 请求 channel {idx}，但音频只有 {audio.shape[1]} 声道")
        return audio[:, idx]
    raise ValueError(f"channel 必须 mean|0|1，得到 {channel!r}")


def load_audio_and_mouth(segment: dict, channel: str, align_tail: bool) -> dict:
    import soundfile as sf

    audio_path = segment.get("audio_path")
    mouth_path = segment.get("mouth_path")
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"音频不存在：{audio_path}")
    if not mouth_path or not Path(mouth_path).exists():
        raise FileNotFoundError(f"mouth NPZ 不存在：{mouth_path}")
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{audio_path}: sr={sr} != {SAMPLE_RATE}")
    audio = _mono_channel(audio, channel, str(audio_path)).astype(np.float32)
    native_samples = len(audio)

    npz = np.load(mouth_path, allow_pickle=True)
    keys = list(npz.files)
    mouth = npz["data"] if "data" in keys else npz[keys[0]]
    mouth = np.asarray(mouth)
    if mouth.ndim == 4 and mouth.shape[-1] == 3:  # RGB mouth -> luma
        mouth = (0.299 * mouth[..., 0] + 0.587 * mouth[..., 1] + 0.114 * mouth[..., 2]).round()
    if mouth.ndim != 3 or mouth.shape[1:] != (96, 96):
        raise ValueError(f"{mouth_path}: 期望 uint8 [T,96,96]，得到 shape={mouth.shape} "
                         f"（若为 landmarks 请检查路径）")
    if mouth.dtype != np.uint8:
        mouth = np.clip(mouth, 0, 255).astype(np.uint8)
    mouth_frames = mouth.shape[0]

    # 对齐上限还要考虑 landmark/face 帧数（若 metadata 提供文件；只读 shape 不读大视频）
    frame_limit = mouth_frames
    landmark_path = segment.get("landmark_path")
    if landmark_path and Path(landmark_path).exists():
        try:
            lm = np.load(landmark_path, allow_pickle=True)
            arr = lm["data"] if "data" in lm.files else lm[lm.files[0]]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                frame_limit = min(frame_limit, int(arr.shape[0]))
        except Exception:
            pass

    T = min(frame_limit, native_samples // FRAME_HOP)
    if T < 1:
        raise ValueError(f"{audio_path}: 对齐后 T<1（audio={native_samples}, mouth={mouth_frames}）")
    used_samples = T * FRAME_HOP
    used_audio = audio[:used_samples]
    used_mouth = mouth[:T]
    discarded = native_samples - used_samples
    if discarded > SAMPLE_RATE * 0.07 + FRAME_HOP and align_tail:
        print(f"[warn] {Path(audio_path).name}: align_tail 丢弃 {discarded} 样本 (>70 ms)；"
              f"native={native_samples}, T={T}", file=sys.stderr)
    return {
        "audio": used_audio,
        "mouth": used_mouth,
        "native_samples": native_samples,
        "native_seconds": native_samples / SAMPLE_RATE,
        "mouth_frames": mouth_frames,
        "used_frames": T,
        "used_seconds": used_samples / SAMPLE_RATE,
        "discarded_samples": discarded,
        "align_tail": align_tail,
    }


def _official_load_sample(module, root: Path, segment: dict, channel: str,
                          align_tail: bool, index: int) -> dict:
    """按官方 load_inference.py 的约定调用 load_sample(...)，并归一化返回。

    官方 API（dataset card）:
      load_sample(root, index=0, align_tail=True, channel='mean')
      load_sample(root, inference_id=segment_id, align_tail=True)
    返回通常是 dict（keys 至少含 audio, mouth）；本函数兼容 dict / tuple。
    """
    fn = getattr(module, "load_sample", None)
    if fn is None:
        raise AttributeError("load_inference.py 没有 load_sample")
    params = inspect.signature(fn).parameters
    kwargs = {"align_tail": align_tail}
    if "channel" in params:
        kwargs["channel"] = channel
    if "inference_id" in params:
        kwargs["inference_id"] = segment["inference_id"]
        try:
            result = fn(str(root), **kwargs)
        except TypeError:
            result = fn(str(root), index=index, **kwargs)
    elif "index" in params:
        result = fn(str(root), index=index, **kwargs)
    else:
        result = fn(str(root), **kwargs)

    if isinstance(result, dict):
        audio = result.get("audio", result.get("mix", result.get("mixture")))
        mouth = result.get("mouth", result.get("mouths"))
        extra = {k: result[k] for k in ("speaker_reference", "landmark", "face", "metadata")
                 if k in result}
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        audio, mouth = result[0], result[1]
        extra = {}
    else:
        raise TypeError(f"load_sample 返回类型不支持：{type(result)}")
    if audio is None or mouth is None:
        raise ValueError("load_sample 返回缺少 audio/mouth")

    audio = np.asarray(audio)
    if audio.ndim == 2:  # 官方实现应已按 channel 处理；若仍为 2D 再应用一次
        audio = _mono_channel(audio, channel, "official load_sample")
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    mouth = np.asarray(mouth)
    if mouth.ndim == 4 and mouth.shape[-1] == 3:
        mouth = (0.299 * mouth[..., 0] + 0.587 * mouth[..., 1] + 0.114 * mouth[..., 2]).round()
    if mouth.ndim != 3 or mouth.shape[1:] != (96, 96):
        raise ValueError(f"official load_sample: mouth 期望 [T,96,96]，得到 {mouth.shape}")
    mouth = np.clip(mouth, 0, 255).astype(np.uint8)
    T = min(len(mouth), len(audio) // FRAME_HOP)
    if T < 1:
        raise ValueError("official load_sample: 对齐后 T<1")
    used_samples = T * FRAME_HOP
    return {
        "audio": audio[:used_samples], "mouth": mouth[:T],
        "native_samples": len(audio), "native_seconds": len(audio) / SAMPLE_RATE,
        "mouth_frames": len(mouth), "used_frames": T,
        "used_seconds": used_samples / SAMPLE_RATE,
        "discarded_samples": len(audio) - used_samples,
        "align_tail": align_tail, "loader": "official:load_sample", "extra": extra,
    }


# ---------------------------------------------------------------------------
# Falcon 分离
# ---------------------------------------------------------------------------
def load_runtime(model: str, device: str, checkpoint: str | None):
    from falcon_runtime import FalconRuntime, variant_for

    variant = variant_for(model)
    runtime = FalconRuntime(variant, checkpoint).load(device)
    return runtime


def _extract_windowed(runtime, audio: np.ndarray, mouth: np.ndarray) -> np.ndarray:
    """窗口化推理（2 s 窗 / 0.5 s hop / 80 ms guard），内存有界；等价于 Studio 推理协议。

    长录音（RealSSA-Real 单段可达数分钟）整段前向会产生 O(T) 大特征图导致 OOM；这里按 Studio 调度：
    每个窗口取 pred[LEFT:LEFT+HOP]（绝对 [endpoint-9280, endpoint-1280)），首窗左补零、末窗右补零。
    边界不做 20 ms crossfade（评测指标对拼接不敏感；报告已声明，Phase 3 可加）。
    """
    from infer_file import preprocess                # noqa: E402
    try:
        from infer_file import VIDEO_FPS as _VIDEO_FPS
        _V = int(_VIDEO_FPS)
    except Exception:
        _V = 25
    globals().setdefault("VIDEO_FPS", _V)
    WINDOW, HOP, GUARD = 32000, 8000, 1280
    LEFT = WINDOW - HOP - GUARD                       # 22720
    FRAMES = WINDOW // 640                            # 50
    frames = min(len(mouth), int(len(audio) * VIDEO_FPS // SAMPLE_RATE))
    if frames < VIDEO_FPS:
        raise SystemExit("Need at least 1 s of synchronized audio and mouth frames.")
    samples = frames * SAMPLE_RATE // VIDEO_FPS
    audio = np.asarray(audio[:samples], dtype=np.float32)
    mouth = np.asarray(mouth[:frames])
    out = np.zeros(samples, dtype=np.float32)
    endpoint = LEFT + HOP                              # 首块绝对起点 = 0
    with runtime._lock, torch.inference_mode():
        while endpoint - (WINDOW - LEFT) < samples:
            wstart = endpoint - WINDOW
            win = np.zeros(WINDOW, dtype=np.float32)
            lo, hi = max(0, wstart), min(samples, endpoint)
            if hi > lo:
                win[lo - wstart:hi - wstart] = audio[lo:hi]
            mstart = wstart // 640
            mframes = np.zeros((FRAMES,) + mouth.shape[1:], dtype=np.uint8)
            tlo, thi = max(0, mstart), min(frames, mstart + FRAMES)
            if thi > tlo:
                mframes[tlo - mstart:thi - mstart] = mouth[tlo:thi]
            if thi - tlo < FRAMES:
                last = mouth[min(frames - 1, max(0, thi - 1))]
                mframes[max(0, thi - mstart):] = last
            wav = torch.from_numpy(np.ascontiguousarray(win)).unsqueeze(0).to(runtime.device)
            emb = runtime.video_model(preprocess(mframes).to(runtime.device))
            pred = runtime.audio_model(wav, emb)[0, 0].float().cpu().numpy()
            abs_start = endpoint - (WINDOW - LEFT)
            chunk = pred[LEFT:LEFT + HOP]
            lo2, hi2 = max(0, abs_start), min(samples, abs_start + HOP)
            if hi2 > lo2:
                out[lo2:hi2] = chunk[lo2 - abs_start:hi2 - abs_start]
            endpoint += HOP
    if not np.isfinite(out).all():
        raise SystemExit("Falcon produced non-finite output.")
    return out


def separate(runtime, audio: np.ndarray, mouth: np.ndarray) -> np.ndarray:
    """窗口化推理（长录音整段前向 OOM；与 Studio 的 2 s 窗/0.5 s hop 一致）。"""
    return _extract_windowed(runtime, audio, mouth)


# ---------------------------------------------------------------------------
# Smoke：用 LRS2-2Mix / 合成数据构造 RealSSA-Real 风格目录，仅验证接口
# ---------------------------------------------------------------------------
def build_smoke_dataset(mode: str, limit: int, lrs2_root: str | None) -> Path:
    import soundfile as sf

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = Path("/root/data/falcon_eval") / f"smoke_realssa_like_{mode}_{stamp}"
    (root / "audio" / "inference" / "smoke" / "scene").mkdir(parents=True, exist_ok=True)
    (root / "mouth" / "smoke" / "scene").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    rows = []
    if mode == "lrs2":
        src = Path(lrs2_root or "/root/data/lrs2_2mix/lrs2_rebuild")
        mixes = sorted((src / "audio" / "wav16k" / "min" / "tt" / "mix").glob("*.wav"))[:limit]
        if not mixes:
            raise SystemExit(f"LRS2-2Mix 冒烟需要 {src}/audio/wav16k/min/tt/mix/*.wav")
        import re
        pat = re.compile(r"^(\d+_\d+)_(-?\d+(?:\.\d+)?)_(\d+_\d+)_(-?\d+(?:\.\d+)?)$")
        for i, wav in enumerate(mixes):
            mid = wav.stem
            m = pat.match(mid)
            if not m:
                continue
            id1 = m.group(1)
            audio, sr = sf.read(wav, dtype="float32")
            stereo = np.stack([audio, audio], axis=1)  # upmix 成 RealSSA-Real 的 stereo 形态
            out_audio = root / "audio" / "inference" / "smoke" / "scene" / f"{i:03d}.wav"
            sf.write(out_audio, stereo, sr, subtype="PCM_16")
            mouth_src = src / "mouths" / f"{id1}.npz"
            out_mouth = root / "mouth" / "smoke" / "scene" / f"{i:03d}.npz"
            np.savez(out_mouth, data=np.load(mouth_src)["data"])
            rows.append({"inference_id": f"smoke-{i:03d}", "audio": str(out_audio.relative_to(root)),
                         "mouth": str(out_mouth.relative_to(root)), "recording_id": "smoke",
                         "scene": "smoke", "part_order": i})
    elif mode == "synthetic":
        rng = np.random.default_rng(0)
        for i in range(limit):
            t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
            audio = (0.2 * np.sin(2 * np.pi * (180 + 40 * i) * t) + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
            stereo = np.stack([audio, audio], axis=1)
            out_audio = root / "audio" / "inference" / "smoke" / "scene" / f"{i:03d}.wav"
            sf.write(out_audio, stereo, SAMPLE_RATE, subtype="PCM_16")
            mouth = rng.integers(0, 256, (50, 96, 96), dtype=np.uint8)
            out_mouth = root / "mouth" / "smoke" / "scene" / f"{i:03d}.npz"
            np.savez(out_mouth, data=mouth)
            rows.append({"inference_id": f"synth-{i:03d}", "audio": str(out_audio.relative_to(root)),
                         "mouth": str(out_mouth.relative_to(root)), "recording_id": "synth",
                         "scene": "synth", "part_order": i})
    else:
        raise SystemExit(f"未知 smoke mode {mode!r}")
    with open(root / "metadata" / "inference_segments.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[smoke] 构造 {mode} 风格 mock RealSSA-Real 目录：{root}（{len(rows)} segments；仅接口验证，不代表 RealSSA 结果）")
    return root


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = ["inference_id", "recording_id", "scene", "channel", "align_tail", "audio_s",
                 "mouth_frames", "used_frames", "used_s", "discarded_samples", "inference_ms", "rtf",
                 "sig_mos", "bak_mos", "ovrl_mos", "p808_mos",
                 "mix_sig_mos", "mix_bak_mos", "mix_ovrl_mos", "mix_p808_mos", "error"]
    cols = preferred + [k for k in rows[0] if k not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error") and r.get("ovrl_mos") not in (None, "")]
    def stats(key: str):
        vals = [float(r[key]) for r in ok if r.get(key) not in (None, "")]
        return {"n": len(vals), "mean": round(statistics.fmean(vals), 4) if vals else None,
                "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0 if vals else None}
    metrics = {k: stats(k) for k in ["sig_mos", "bak_mos", "ovrl_mos", "p808_mos"]}
    if any("mix_ovrl_mos" in r for r in ok):
        metrics.update({f"mix_{k}": stats(f"mix_{k}") for k in ["sig_mos", "bak_mos", "ovrl_mos", "p808_mos"]})
    by_scene: dict[str, list[dict]] = {}
    for r in ok:
        by_scene.setdefault(str(r.get("scene") or "(unknown)"), []).append(r)
    scene_stats = {}
    for scene, items in sorted(by_scene.items()):
        scene_stats[scene] = {"n": len(items),
                              **{k: round(statistics.fmean([float(x[k]) for x in items]), 4)
                                 for k in ["sig_mos", "bak_mos", "ovrl_mos", "p808_mos"]}}
    return {"segments_total": len(rows), "segments_ok": len(ok), "segments_error": len(rows) - len(ok),
            "metrics": metrics, "by_scene": scene_stats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RealSSA-Real Falcon+DNSMOS evaluation (reference-free)")
    ap.add_argument("--data", default=os.environ.get("REALSSA_REAL_DIR", str(CANDIDATE_DATA)),
                    help="RealSSA-Real 下载根目录（含 audio/inference、mouth、metadata）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 /root/data/falcon_eval/realssa_real_<UTC>）")
    ap.add_argument("--channel", default="mean", choices=["mean", "0", "1"], help="stereo → mono 策略")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--model", default="JusperLee/Falcon", help="JusperLee/Falcon 或 JusperLee/Falcon-Big")
    ap.add_argument("--checkpoint", default=None, help="可选本地权重（默认用 HF 缓存/短链）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 段（0=全部）")
    ap.add_argument("--align-tail", dest="align_tail", action="store_true", default=True,
                    help="按官方约定截断到 min(mouth,audio)（默认开）")
    ap.add_argument("--no-align-tail", dest="align_tail", action="store_false")
    ap.add_argument("--baseline", action="store_true", help="同时算 mixture 的 DNSMOS 基线")
    ap.add_argument("--skip-model", action="store_true",
                    help="仅接口冒烟：跳过 Falcon 分离，DNSMOS 评 mixture（不能当分离结果）")
    ap.add_argument("--save-estimates", action="store_true", help="保存分离音频到 out-dir/estimates（注意隐私/许可）")
    ap.add_argument("--smoke-lrs2", nargs="?", const="/root/data/lrs2_2mix/lrs2_rebuild", default=None,
                    help="用 LRS2-2Mix 构造 mock RealSSA-Real 目录做接口冒烟（可选路径）")
    ap.add_argument("--smoke-synthetic", action="store_true", help="用合成音频构造 mock 目录做接口冒烟")
    ap.add_argument("--dry-run", action="store_true", help="只列 segment 清单，不加载模型/DNSMOS")
    ap.add_argument("--fail-fast", action="store_true", help="首段失败即退出")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    smoke = args.smoke_lrs2 is not None or args.smoke_synthetic
    if args.smoke_synthetic and args.smoke_lrs2 is None:
        data_root = build_smoke_dataset("synthetic", max(args.limit, 2), None)
        is_smoke, smoke_source = True, "synthetic"
    elif args.smoke_lrs2 is not None:
        data_root = build_smoke_dataset("lrs2", max(args.limit, 2), args.smoke_lrs2)
        is_smoke, smoke_source = True, "LRS2-2Mix tt (mock RealSSA-Real layout)"
    else:
        data_root = Path(args.data)
        is_smoke, smoke_source = False, ""

    out_dir = Path(args.out_dir) if args.out_dir else Path("/root/data/falcon_eval") / f"realssa_real_{stamp}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[realssa-real] data={data_root}  smoke={is_smoke} ({smoke_source})  out={out_dir}")
    print(f"[env] speechmos={_version('speechmos')} onnxruntime={_version('onnxruntime')} "
          f"channel={args.channel} align_tail={args.align_tail} device={args.device} model={args.model}")

    segments, loader_kind = discover_segments(data_root)
    official_module = _load_official_module(data_root) if loader_kind.startswith("official") else None
    if not segments:
        print("\n[WAIT] 未找到 RealSSA-Real 数据（或目录为空）。", file=sys.stderr)
        print(f"  - RealSSA-Real 是 gated(manual) 数据集；本机申请已提交、待人工审批；当前数据文件 403。", file=sys.stderr)
        print(f"  - 审批/下载/字段说明见 {GUIDE} §5 与 docs/verification/phase1/dataset-cards/RealSSA-Real-README.md", file=sys.stderr)
        print("  - 无数据时可用接口冒烟：", file=sys.stderr)
        print("      python tools/repro/phase2/eval_realssa_real.py --smoke-lrs2 /root/data/lrs2_2mix/lrs2_rebuild --limit 2",
              file=sys.stderr)
        print("      python tools/repro/phase2/eval_realssa_real.py --smoke-synthetic --limit 2", file=sys.stderr)
        print("  - 注意：本脚本只算 DNSMOS；RealSSA-Real 没有干净 target，不能算 SI-SDR。", file=sys.stderr)
        return 2
    if args.limit:
        segments = segments[: args.limit]
    print(f"[realssa-real] loader={loader_kind}  segments={len(segments)}")
    for s in segments:
        audio_name = Path(str(s["audio_path"])).name if s.get("audio_path") else "(official loader)"
        print(f"  - {s['inference_id']:24s} scene={s.get('scene', ''):20s} audio={audio_name}")

    if args.dry_run:
        print("[dry-run] 未加载模型、未计算 DNSMOS、未写 CSV。")
        return 0

    # 加载模型（LRS2 smoke 默认真实分离；合成 smoke 可 --skip-model）
    runtime = None
    if not args.skip_model:
        t0 = time.time()
        runtime = load_runtime(args.model, args.device, args.checkpoint)
        print(f"[model] FalconRuntime loaded in {time.time() - t0:.1f}s device={runtime.device}")
    else:
        print("[model] --skip-model：不跑分离，estimate=mixture（仅接口冒烟，不能作为分离结果）", file=sys.stderr)

    # DNSMOS 预热
    score_array(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    rows: list[dict] = []
    est_dir = out_dir / "estimates"
    if args.save_estimates and not args.skip_model:
        est_dir.mkdir(parents=True, exist_ok=True)
    for i, seg in enumerate(segments, 1):
        row = {"inference_id": seg.get("inference_id"), "recording_id": seg.get("recording_id", ""),
               "scene": seg.get("scene", ""), "channel": args.channel, "align_tail": args.align_tail}
        try:
            if official_module is not None:
                loaded = _official_load_sample(official_module, data_root, seg, args.channel,
                                               args.align_tail, index=i - 1)
            else:
                loaded = load_audio_and_mouth(seg, args.channel, args.align_tail)
            row.update({"audio_s": round(loaded["native_seconds"], 3),
                        "mouth_frames": loaded["mouth_frames"],
                        "used_frames": loaded["used_frames"],
                        "used_s": round(loaded["used_seconds"], 3),
                        "discarded_samples": loaded["discarded_samples"]})
            if args.skip_model:
                estimate = loaded["audio"]
                row["inference_ms"] = ""
                row["rtf"] = ""
            else:
                t0 = time.time()
                estimate = separate(runtime, loaded["audio"], loaded["mouth"])
                dt = time.time() - t0
                row["inference_ms"] = round(dt * 1000, 1)
                row["rtf"] = round(dt / max(loaded["used_seconds"], 1e-9), 3)
            scores = score_array(estimate, SAMPLE_RATE)
            row.update({k: round(v, 4) for k, v in scores.items()})
            if args.baseline:
                mix_scores = score_array(loaded["audio"], SAMPLE_RATE)
                row.update({f"mix_{k}": round(v, 4) for k, v in mix_scores.items()})
            if args.save_estimates and not args.skip_model:
                import soundfile as sf
                sf.write(est_dir / f"{seg.get('inference_id', i)}.wav", estimate, SAMPLE_RATE, subtype="PCM_16")
            if not args.quiet:
                print(f"  [{i}/{len(segments)}] {seg.get('inference_id')}  used={loaded['used_seconds']:.2f}s  "
                      f"SIG={row['sig_mos']:.3f} BAK={row['bak_mos']:.3f} OVRL={row['ovrl_mos']:.3f} "
                      f"P808={row['p808_mos']:.3f}")
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [{i}/{len(segments)}] {seg.get('inference_id')} ERROR: {row['error']}", file=sys.stderr)
            if args.fail_fast:
                rows.append(row)
                break
        rows.append(row)

    suffix = "SMOKE_" if is_smoke else ""
    csv_path = out_dir / f"{suffix}realssa_real_dnsmos_per_segment.csv"
    _write_csv(csv_path, rows)
    summary = _summarize(rows)
    meta = {
        "tool": "tools/repro/phase2/eval_realssa_real.py",
        "timestamp_utc": stamp, "data_root": str(data_root), "loader_kind": loader_kind,
        "is_smoke": is_smoke, "smoke_source": smoke_source,
        "channel": args.channel, "align_tail": args.align_tail, "model": args.model,
        "checkpoint": args.checkpoint, "device": args.device, "skip_model": args.skip_model,
        "limit": args.limit, "baseline": args.baseline,
        "speechmos_version": _version("speechmos"), "onnxruntime_version": _version("onnxruntime"),
        "note": "RealSSA-Real has no clean target: DNSMOS only, SI-SDR is not computed.",
        "segments_total": len(segments),
        "summary": summary,
    }
    (out_dir / f"{suffix}run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{suffix}summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[realssa-real] CSV -> {csv_path}")
    print(f"[realssa-real] summary -> {out_dir / (suffix + 'summary.json')}")
    if summary["segments_ok"]:
        m = summary["metrics"]
        print(f"[realssa-real] N={summary['segments_ok']}  "
              f"SIG={m['sig_mos']['mean']} BAK={m['bak_mos']['mean']} "
              f"OVRL={m['ovrl_mos']['mean']} P808={m['p808_mos']['mean']}")
        if is_smoke:
            print("[realssa-real] ⚠️ 这是接口冒烟（非 RealSSA-Real 数据），结果不代表论文数值。")
    return 0 if summary["segments_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
