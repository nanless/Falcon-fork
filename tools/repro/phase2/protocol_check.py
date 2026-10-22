"""Phase-2 F: third-party independent verification of the LRS2-2Mix metrics/protocol.

This script is intentionally **independent** of the Phase-1 T3 metric implementation:
it re-implements SI-SDR / SI-SDRi in pure NumPy and calls ``fast_bss_eval`` directly,
then cross-checks (a) ``look2hear.metrics.core``, (b) the full 3000-utterance Phase-1
``metrics_combined.csv``, and (c) the official weights re-run on N=50 utterances through
the actual ``audio_visual_test`` loading path.

It also quantifies protocol choices:
  * official ``create_mix`` algebra: mix == s1+s2, peak-normalised to 0.9, activlev weights;
  * baseline (mixture) SI-SDR/SDR distribution over all 3000 clips;
  * target convention (s1 vs s2), PIT/oracle upper bound;
  * metric-definition variants (zero-mean, SDR filter length, change_sign, 8 kHz);
  * waveform alignment (best lag), clipping, scale invariance;
  * mouth branch: real vs zero/shifted/wrong-speaker mouth (alignment & speaker conditioning);
  * Falcon-Big vs Small on the same N=50;
  * training-corpus evidence for the released falcon.pt (Experiments/checkpoint/Falcon/conf.yml).

Usage (repo root, conda falcon):
  /root/miniforge3/envs/falcon/bin/python tools/repro/phase2/protocol_check.py --stage all
Logs/results under docs/verification/phase2/.

Stages: data | metrics | variants | big | recheck (CPU, cache-based) | static | all.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
PHASE2 = REPO / "docs" / "verification" / "phase2"
CACHE = Path("/tmp/p2f_cache.npz")
SMALL_CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
BIG_CKPT = Path("/root/.cache/falcon_weights/falcon-big.pt")
DATA_DIR = Path("/root/tmp/lrs2_eval_tt_full")
REBUILD = Path("/root/data/lrs2_2mix/lrs2_rebuild")
CSV = REPO / "Experiments/checkpoint/Falcon/results/metrics_combined.csv"
SAMPLE_RATE = 16000
T = 32000
RESULTS: dict = {}
LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


def banner(title: str) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# Independent metric implementations (pure NumPy / direct fast_bss_eval)
# ══════════════════════════════════════════════════════════════════════════════
def my_si_sdr(ref: np.ndarray, est: np.ndarray, zero_mean: bool = True, eps: float = 1e-12) -> float:
    """Standard SI-SDR, positive dB, higher is better."""
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    e = np.asarray(est, dtype=np.float64).reshape(-1)
    if zero_mean:
        r = r - r.mean()
        e = e - e.mean()
    alpha = float(np.dot(e, r) / (np.dot(r, r) + eps))
    target = alpha * r
    noise = e - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def my_si_sdr_i(ref: np.ndarray, est: np.ndarray, mix: np.ndarray, zero_mean: bool = True) -> float:
    return my_si_sdr(ref, est, zero_mean) - my_si_sdr(ref, mix, zero_mean)


def my_sdr(ref: np.ndarray, est: np.ndarray, filter_length: int = 512, change_sign: bool = False) -> float:
    """BSS-Eval-v4-style SDR via fast_bss_eval, positive dB (change_sign=False)."""
    import fast_bss_eval
    r = np.asarray(ref, dtype=np.float64).reshape(1, -1)
    e = np.asarray(est, dtype=np.float64).reshape(1, -1)
    val = fast_bss_eval.sdr(r, e, filter_length=filter_length, change_sign=change_sign)
    return float(np.asarray(val).reshape(-1)[0])


def fast_bss_sdr_pit_loss(est: np.ndarray, ref: np.ndarray, filter_length: int = 512) -> float:
    """Core.py convention: -fast_bss_eval.sdr_pit_loss(...).mean() (positive dB)."""
    import fast_bss_eval
    e = torch.from_numpy(np.asarray(est, dtype=np.float64).reshape(1, 1, -1))
    r = torch.from_numpy(np.asarray(ref, dtype=np.float64).reshape(1, 1, -1))
    return -float(fast_bss_eval.sdr_pit_loss(e, r, filter_length=filter_length).mean())


# ══════════════════════════════════════════════════════════════════════════════
# helpers: data lists
# ══════════════════════════════════════════════════════════════════════════════
def rebuild_names() -> list[str]:
    """Mix names in the official create_mix order (scaling.mat row order)."""
    mix_list = REBUILD / "create_mix" / "mix_2_spk_min_tt_mix"
    names = [ln.strip() for ln in mix_list.read_text().splitlines() if ln.strip()]
    return names


def mix_names_eval() -> list[str]:
    return sorted(p.name for p in (DATA_DIR / "tt" / "mix").glob("*.wav"))


def read_audio(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE, (path, sr)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def s1_s2_ids(mix_name: str) -> tuple[str, str]:
    stem = mix_name[:-4] if mix_name.endswith(".wav") else mix_name
    parts = stem.split("_")
    # <s1_id>_<snr1>_<s2_id>_<snr2>, ids are "<5e18-ish>_<utterance>"
    return f"{parts[0]}_{parts[1]}", f"{parts[3]}_{parts[4]}"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: data / official mixing algebra / baseline distribution
# ══════════════════════════════════════════════════════════════════════════════
def stage_data() -> dict:
    import scipy.io

    banner("STAGE 1 — data & official mixing algebra (all 3000 tt clips)")
    names = rebuild_names()
    eval_names = mix_names_eval()
    log(f"official mix list: {len(names)} | eval mix files: {len(eval_names)}")
    log(f"official list vs eval sorted list: sets equal={set(names)=={n[:-4] for n in eval_names}}, same order={names == [n[:-4] for n in eval_names]}")
    log(f"first official: {names[0]}")
    log(f"first eval   : {eval_names[0]}")

    mat = scipy.io.loadmat(str(REBUILD / "audio/wav16k/min/tt/scaling.mat"))
    scaling_16k = np.asarray(mat["scaling_16k"], dtype=np.float64)      # (3000,2)
    scaling_16bit = np.asarray(mat["scaling16bit_16k"], dtype=np.float64).reshape(-1)  # (3000,)
    log(f"scaling.mat: scaling_16k {scaling_16k.shape}, scaling16bit_16k {scaling_16bit.shape}")

    root = REBUILD / "audio/wav16k/min/tt"
    wav_names = [n if n.endswith(".wav") else n + ".wav" for n in names]
    lengths, peaks, resid, resid_rel = set(), [], [], []
    mix_abs_equals_s1s2_peak, s1_peaks, s2_peaks = 0, [], []
    rms_mix, rms_s1 = [], []
    for i, wav_name in enumerate(wav_names):
        mx = read_audio(root / "mix" / wav_name)
        s1 = read_audio(root / "s1" / wav_name)
        s2 = read_audio(root / "s2" / wav_name)
        lengths.add(int(mx.size))
        lengths.add(int(s1.size))
        lengths.add(int(s2.size))
        sum12 = s1 + s2
        peak_all = max(np.abs(mx).max(), np.abs(s1).max(), np.abs(s2).max())
        peaks.append(peak_all)
        if abs(peak_all - 0.9) < 1e-5:
            mix_abs_equals_s1s2_peak += 1
        d = mx - sum12
        resid.append(float(np.abs(d).max()))
        resid_rel.append(float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(mx ** 2)) + 1e-12)))
        s1_peaks.append(float(np.abs(s1).max()))
        s2_peaks.append(float(np.abs(s2).max()))
        rms_mix.append(float(np.sqrt(np.mean(mx ** 2))))
        rms_s1.append(float(np.sqrt(np.mean(s1 ** 2))))

    result = {
        "n": len(names),
        "lengths": sorted(lengths),
        "peak_max_of_mix_s1_s2_unique": {"min": float(np.min(peaks)), "max": float(np.max(peaks))},
        "n_peak_is_0.9": mix_abs_equals_s1s2_peak,
        "mix_minus_s1_plus_s2_maxabs": {"max": float(np.max(resid)), "p99": float(np.percentile(resid, 99))},
        "mix_minus_s1_plus_s2_rel_rms": {"max": float(np.max(resid_rel)), "median": float(np.median(resid_rel))},
        "s1_peak": {"min": float(np.min(s1_peaks)), "median": float(np.median(s1_peaks)), "max": float(np.max(s1_peaks))},
        "s2_peak": {"min": float(np.min(s2_peaks)), "median": float(np.median(s2_peaks)), "max": float(np.max(s2_peaks))},
        "rms_mix": {"median": float(np.median(rms_mix))},
        "rms_s1": {"median": float(np.median(rms_s1))},
        "scaling_16bit_range": [float(scaling_16bit.min()), float(scaling_16bit.max())],
    }
    log(f"clip lengths: {result['lengths']} (expect [32000] = 2.0 s @16 kHz)")
    log(f"max(|mix|,|s1|,|s2|) == 0.9: {mix_abs_equals_s1s2_peak}/{len(names)}  "
        f"(range {result['peak_max_of_mix_s1_s2_unique']})")
    log(f"mix == s1+s2: max|d| = {result['mix_minus_s1_plus_s2_maxabs']['max']:.3e}, "
        f"p99 = {result['mix_minus_s1_plus_s2_maxabs']['p99']:.3e}, "
        f"rel-rms max = {result['mix_minus_s1_plus_s2_rel_rms']['max']:.3e} "
        f"(16-bit PCM rounding level)")
    log(f"s1 peak (scaled target) median={result['s1_peak']['median']:.3f}, "
        f"s2 peak median={result['s2_peak']['median']:.3f}; mix/s1 RMS median={result['rms_mix']['median']:.4f}/"
        f"{result['rms_s1']['median']:.4f}")
    log(f"scaling16bit_16k range: {result['scaling_16bit_range']}")

    # target.json wav == s1 (target convention) and eval mix == rebuild mix
    tgt = json.loads((DATA_DIR / "tt" / "target.json").read_text())
    mixj = json.loads((DATA_DIR / "tt" / "mix.json").read_text())
    same_target, same_mix = 0, 0
    for k in range(5):
        tgt_wav = read_audio(Path(tgt[k][0]))
        s1_wav = read_audio(root / "s1" / Path(tgt[k][0]).name)
        ev_mix = read_audio(Path(mixj[k][0]))
        rb_mix = read_audio(root / "mix" / Path(mixj[k][0]).name)
        same_target += int(np.array_equal(tgt_wav, s1_wav))
        same_mix += int(np.array_equal(ev_mix, rb_mix))
    result["target_is_s1_copy_sample"] = f"{same_target}/5"
    result["eval_mix_is_rebuild_copy_sample"] = f"{same_mix}/5"
    log(f"eval target wav == rebuild s1: {same_target}/5 | eval mix == rebuild mix: {same_mix}/5")

    # ---- baseline (mixture) SI-SDR distribution over all 3000 ----
    t0 = time.time()
    base_si = np.array([my_si_sdr(read_audio(root / "s1" / w), read_audio(root / "mix" / w)) for w in wav_names])
    log(f"baseline SI-SDR(mix, s1) over 3000: mean={base_si.mean():.4f} std={base_si.std(ddof=0):.4f} "
        f"median={np.median(base_si):.4f} p5={np.percentile(base_si,5):.3f} p95={np.percentile(base_si,95):.3f} "
        f"[{time.time()-t0:.1f}s]")
    result["baseline_si_sdr_all3000"] = {
        "mean": float(base_si.mean()), "std": float(base_si.std(ddof=0)),
        "median": float(np.median(base_si)),
        "p5": float(np.percentile(base_si, 5)), "p95": float(np.percentile(base_si, 95)),
    }
    # baseline SDR on a fixed sample of 300 (fast_bss_eval is ~ms/item)
    idx = np.linspace(0, len(names) - 1, 300).astype(int)
    t0 = time.time()
    base_sdr = np.array([my_sdr(read_audio(root / "s1" / wav_names[i]), read_audio(root / "mix" / wav_names[i])) for i in idx])
    log(f"baseline SDR(mix, s1) over {len(idx)} sample: mean={base_sdr.mean():.4f} std={base_sdr.std(ddof=0):.4f} "
        f"[{time.time()-t0:.1f}s]")
    result["baseline_sdr_sample300"] = {"n": int(len(idx)), "mean": float(base_sdr.mean()),
                                        "std": float(base_sdr.std(ddof=0))}

    # ---- filename SNR vs measured source-energy ratio (official create_mix weights) ----
    snrs_file, snrs_meas = [], []
    for wav_name in wav_names[:500]:
        parts = Path(wav_name).stem.split("_")
        snr1, snr2 = float(parts[2]), float(parts[5])
        s1 = read_audio(root / "s1" / wav_name)
        s2 = read_audio(root / "s2" / wav_name)
        rel_meas = 10.0 * np.log10((np.sum(s1 ** 2) + 1e-12) / (np.sum(s2 ** 2) + 1e-12))
        snrs_file.append(snr1 - snr2)
        snrs_meas.append(rel_meas)
    snrs_file = np.array(snrs_file); snrs_meas = np.array(snrs_meas)
    result["filename_snr_vs_measured"] = {
        "n": int(len(snrs_file)),
        "mean_abs_diff_db": float(np.mean(np.abs(snrs_file - snrs_meas))),
        "max_abs_diff_db": float(np.max(np.abs(snrs_file - snrs_meas))),
        "corr": float(np.corrcoef(snrs_file, snrs_meas)[0, 1]),
        "snr1_range": [float(min(float(Path(w).stem.split('_')[2]) for w in wav_names)),
                       float(max(float(Path(w).stem.split('_')[2]) for w in wav_names))],
    }
    log(f"official create_mix SNR weights: filename(snr1-snr2) vs measured 10log10(E_s1/E_s2): "
        f"mean|Δ|={result['filename_snr_vs_measured']['mean_abs_diff_db']:.3f} dB, "
        f"max|Δ|={result['filename_snr_vs_measured']['max_abs_diff_db']:.3f} dB, "
        f"corr={result['filename_snr_vs_measured']['corr']:.6f} over 500 items "
        f"(snr1 range {result['filename_snr_vs_measured']['snr1_range']})")

    # ---- supplementary: audio envelope vs mouth-motion correlation (weak A/V sync probe) ----
    lags_av, corr_av = [], []
    for wav_name in wav_names[:30]:
        sid, _ = s1_s2_ids(wav_name)
        z = np.load(REBUILD / "mouths" / f"{sid}.npz")["data"].astype(np.float32)
        feat = -z.mean(axis=(1, 2))  # darkness (mouth opening proxy)
        mx = read_audio(root / "mix" / wav_name)
        env = np.sqrt((mx.reshape(50, -1) ** 2).mean(axis=1))
        a = (feat - feat.mean()) / (feat.std() + 1e-9)
        b = (env - env.mean()) / (env.std() + 1e-9)
        cs = [float(np.dot(a, np.roll(b, lag)) / len(a)) for lag in range(-5, 6)]
        i = int(np.argmax(cs))
        lags_av.append(i - 5); corr_av.append(cs[i])
    result["av_sync_probe"] = {
        "n": len(lags_av),
        "lag_mode": int(pd.Series(lags_av).mode().iloc[0]),
        "mean_abs_lag": float(np.mean(np.abs(lags_av))),
        "mean_peak_corr": float(np.mean(corr_av)),
        "note": "weak/inconclusive: pixel-based mouth-opening proxy vs audio envelope; "
                "the model-based mouth-offset sweep in stage 3 is the authoritative alignment test",
    }
    log(f"A/V sync probe (supplementary, weak): lag mode={pd.Series(lags_av).mode().iloc[0]} frames, "
        f"mean|lag|={np.mean(np.abs(lags_av)):.2f}, mean peak corr={np.mean(corr_av):.3f} "
        f"-> inconclusive on its own")

    # ---- mouths: npz & eval .pt ----
    npz_files = sorted((REBUILD / "mouths").glob("*.npz"))
    pt_files = sorted((DATA_DIR / "tt" / "mouths").glob("*.pt"))
    npz_shapes, npz_dtypes, npz_missing_for_s1 = set(), set(), 0
    for npz in npz_files[:2000]:
        with np.load(npz) as z:
            npz_shapes.add(tuple(z["data"].shape)); npz_dtypes.add(str(z["data"].dtype))
    s1_ids = [s1_s2_ids(n)[0] for n in wav_names]
    for sid in s1_ids:
        if not (REBUILD / "mouths" / f"{sid}.npz").exists():
            npz_missing_for_s1 += 1
    s1s2_missing = sum(1 for n in wav_names if not (REBUILD / "mouths" / f"{s1_s2_ids(n)[1]}.npz").exists())
    sample_pt = sorted(pt_files)[:5]
    pt_info = []
    for p in sample_pt:
        t = torch.load(p, weights_only=True)
        pt_info.append({"shape": list(t.shape), "dtype": str(t.dtype),
                        "min": int(t.min()), "max": int(t.max())})
    # pt == stacked npz for a sample (same stem = mix name, npz is by s1 id)
    ok_stack = 0
    for name in wav_names[:3]:
        sid, _ = s1_s2_ids(name)
        z = np.load(REBUILD / "mouths" / f"{sid}.npz")["data"]
        pt = torch.load(DATA_DIR / "tt" / "mouths" / (Path(name).stem + ".pt"), weights_only=True)
        ok_stack += int(np.array_equal(np.stack([z, z, z], axis=-1), pt.numpy()))
    result.update({
        "mouths_npz_count": len(npz_files), "mouths_pt_count": len(pt_files),
        "npz_shapes_sample": [list(s) for s in list(npz_shapes)[:5]], "npz_dtypes": list(npz_dtypes),
        "s1_npz_missing": npz_missing_for_s1, "s2_npz_missing": s1s2_missing,
        "pt_sample": pt_info, "pt_equals_stacked_npz_sample": f"{ok_stack}/3",
    })
    log(f"mouth npz: {len(npz_files)} files, shapes={list(npz_shapes)[:3]}, dtypes={npz_dtypes}; "
        f"missing npz for s1 ids: {npz_missing_for_s1}/3000, for s2 ids: {s1s2_missing}/3000")
    log(f"eval mouth .pt: {len(pt_files)} files, sample {pt_info[0] if pt_info else None}; "
        f"pt == stack(npz,3ch) sample: {ok_stack}/3")

    # ---- eval dataset mouth tensor (preprocessing contract) ----
    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset
    ds = AVSpeechEvalDataset(json_dir=str(DATA_DIR / "tt"), sample_rate=16000, normalize_audio=False)
    mouth_stats = []
    for k in (0, 1, 2):
        mix_t, src_t, mouth_np, key, n_near = ds[k]
        mouth_stats.append({"key": key, "mouth_shape": list(np.asarray(mouth_np).shape),
                            "dtype": str(np.asarray(mouth_np).dtype),
                            "mean": float(np.asarray(mouth_np).mean()),
                            "std": float(np.asarray(mouth_np).std()),
                            "min": float(np.asarray(mouth_np).min()), "max": float(np.asarray(mouth_np).max()),
                            "mix_shape": list(mix_t.shape), "n_near": int(n_near)})
    result["eval_dataset_mouth_sample"] = mouth_stats
    for m in mouth_stats:
        log(f"eval dataset[{m['key'][:32]}…]: mix {m['mix_shape']}, mouth {m['mouth_shape']} {m['dtype']}, "
            f"mean={m['mean']:.3f} std={m['std']:.3f} range=[{m['min']:.2f},{m['max']:.2f}]  "
            f"(z-score after /255 + center-crop 96→88; mean≈0/std≈1 expected)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: model re-run N=50 + independent metrics vs core vs Phase-1 CSV
# ══════════════════════════════════════════════════════════════════════════════
def _load_small_models(device):
    import audio_visual_test as avt
    cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
    train_conf = avt._get_train_conf(cfg)
    model, video_model = avt._load_models_from_ckpt(str(SMALL_CKPT), train_conf, device)
    return model, video_model, avt


def _forward_items(model, video_model, ds, avt, indices, device, mouth_transform=None):
    """Run the eval pipeline for given dataset indices; returns dict of outputs."""
    import torch.nn.functional as F  # noqa: F401
    out = {"keys": [], "mix": [], "s1": [], "est": [], "mouth": []}
    with torch.inference_mode():
        for i in indices:
            mix_t, src_t, mouth_np, key, _ = ds[i]
            if mouth_transform is not None:
                mouth_np = mouth_transform(mouth_np, key)
            mouth_in = avt._mouth_to_tensor(np.asarray(mouth_np)).float().to(device)
            mouth_emb = video_model(mouth_in)
            est = model(mix_t.unsqueeze(0).to(device), mouth_emb)
            if isinstance(est, tuple):
                est = est[0]
            out["keys"].append(key)
            out["mix"].append(mix_t.detach().cpu().numpy())
            out["s1"].append(src_t.detach().cpu().numpy())
            out["est"].append(est[0, 0].detach().float().cpu().numpy())
            out["mouth"].append(np.asarray(mouth_np, dtype=np.float32))
    return out


def _metrics_dict(est, ref, mix):
    from look2hear.metrics import core
    return {
        "si_sdr": my_si_sdr(ref, est),
        "si_sdr_i": my_si_sdr_i(ref, est, mix),
        "sdr": my_sdr(ref, est),
        "sdr_i": my_sdr(ref, est) - my_sdr(ref, mix),
        "_core": {
            "si_sdr": core.si_sdr(est, ref),
            "si_sdr_i": core.si_sdr_improvement(est, ref, mix),
            "sdr": core.sdr(est, ref),
            "sdr_i": core.sdr_improvement(est, ref, mix),
        },
    }


def stage_metrics(n: int = 50, device: str = "cuda:0") -> dict:
    banner(f"STAGE 2 — official weights re-run N={n}; independent metrics vs core.py vs Phase-1 CSV")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = _load_ds()
    model, video_model, avt = _load_small_models(device)
    out = _forward_items(model, video_model, ds, avt, list(range(n)), device)
    log(f"re-ran {n} utterances on {device}; first key={out['keys'][0]}")

    # Phase-1 CSV
    df = pd.read_csv(CSV)
    rows = df[~df.snt_id.isin(["avg", "std"])].reset_index(drop=True)
    log(f"CSV rows={len(rows)}; avg row: sdr={df[df.snt_id=='avg'].sdr.iloc[0]:.4f} "
        f"sdr_i={df[df.snt_id=='avg'].sdr_i.iloc[0]:.4f} si-snr={df[df.snt_id=='avg']['si-snr'].iloc[0]:.4f} "
        f"si-snr_i={df[df.snt_id=='avg']['si-snr_i'].iloc[0]:.4f}")
    # recompute CSV avg/std from per-utterance rows (data-level check of the merge/agg code)
    agg = {c: (rows[c].mean(), rows[c].std(ddof=0)) for c in ["sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"]}
    agg_ok = all(abs(agg[c][0] - df[df.snt_id == "avg"][c].iloc[0]) < 1e-9 and
                 abs(agg[c][1] - df[df.snt_id == "std"][c].iloc[0]) < 1e-9 for c in agg)
    log(f"CSV avg/std == recomputation from per-utterance rows: {agg_ok}")

    # compare metric stacks
    rows_n = rows.iloc[:n]
    deltas_my_core, deltas_my_csv, deltas_base, deltas_base_sdr = [], [], [], []
    csv_key_match = [k == rows_n.snt_id.iloc[i] for i, k in enumerate(out["keys"])]
    log(f"CSV row order matches dataset order (first {n}): {sum(csv_key_match)}/{n}")
    per_item = []
    for i, (est, ref, mix, key) in enumerate(zip(out["est"], out["s1"], out["mix"], out["keys"])):
        m = _metrics_dict(est, ref, mix)
        r = rows_n.iloc[i]
        csv_vals = {"si_sdr": float(r["si-snr"]), "si_sdr_i": float(r["si-snr_i"]),
                    "sdr": float(r["sdr"]), "sdr_i": float(r["sdr_i"])}
        my_vals = {k: m[k] for k in ("si_sdr", "si_sdr_i", "sdr", "sdr_i")}
        core_vals = m["_core"]
        deltas_my_core.append(max(abs(my_vals[k] - core_vals[k]) for k in my_vals))
        deltas_my_csv.append(max(abs(my_vals[k] - csv_vals[k]) for k in my_vals))
        base_from_csv = csv_vals["si_sdr"] - csv_vals["si_sdr_i"]
        base_mine = my_si_sdr(ref, mix)
        deltas_base.append(abs(base_mine - base_from_csv))
        base_sdr_from_csv = csv_vals["sdr"] - csv_vals["sdr_i"]
        deltas_base_sdr.append(abs(my_sdr(ref, mix) - base_sdr_from_csv))
        per_item.append({"key": key, "my": my_vals, "core": core_vals, "csv": csv_vals,
                         "baseline_si_sdr_mine": base_mine, "baseline_si_sdr_from_csv": base_from_csv})
    result = {
        "n": n,
        "maxabs_my_vs_core_db": float(max(deltas_my_core)),
        "maxabs_my_vs_csv_db": float(max(deltas_my_csv)),
        "maxabs_baseline_formula_db": float(max(deltas_base)),
        "maxabs_baseline_sdr_formula_db": float(max(deltas_base_sdr)),
        "csv_avg_recompute_ok": bool(agg_ok),
        "csv_avg": {c: float(df[df.snt_id == "avg"][c].iloc[0]) for c in agg},
        "per_item": per_item,
    }
    log(f"max|my implementation − core.py| over 4 metrics × {n}: {result['maxabs_my_vs_core_db']:.3e} dB")
    log(f"max|my implementation − Phase-1 CSV| over 4 metrics × {n}:        {result['maxabs_my_vs_csv_db']:.3e} dB")
    log(f"max|my baseline SI-SDR − (CSV si-snr − si-snr_i)|:              {result['maxabs_baseline_formula_db']:.3e} dB")
    log(f"max|my baseline SDR − (CSV sdr − sdr_i)|:                        {result['maxabs_baseline_sdr_formula_db']:.3e} dB")
    log(f"mean over these {n}: my si-snr_i={np.mean([p['my']['si_sdr_i'] for p in per_item]):.4f}, "
        f"my sdr_i={np.mean([p['my']['sdr_i'] for p in per_item]):.4f}  "
        f"(CSV subset: {np.mean([p['csv']['si_sdr_i'] for p in per_item]):.4f}/"
        f"{np.mean([p['csv']['sdr_i'] for p in per_item]):.4f})")
    # persist estimates for later stages
    np.savez(CACHE, keys=np.array(out["keys"]),
             est=np.stack(out["est"]), ref=np.stack(out["s1"]), mix=np.stack(out["mix"]),
             mouth=np.stack(out["mouth"]))
    log(f"estimates cached -> {CACHE}")
    return result


def _load_ds():
    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset
    return AVSpeechEvalDataset(json_dir=str(DATA_DIR / "tt"), sample_rate=16000, normalize_audio=False)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: protocol variants (alignment, target, PIT, metric definitions, mouths)
# ══════════════════════════════════════════════════════════════════════════════
def _best_lag(ref: np.ndarray, est: np.ndarray, max_lag: int = 1024) -> tuple[int, float, float]:
    """Return (best_lag, si_sdr_at_0, si_sdr_at_best_lag); positive lag shifts est later."""
    from scipy.signal import correlate, correlation_lags
    r = ref - ref.mean(); e = est - est.mean()
    n = 1 << int(np.ceil(np.log2(len(r) + len(e))))
    c = correlate(e, r, mode="full", method="fft")
    lags = correlation_lags(len(e), len(r), mode="full")
    mask = np.abs(lags) <= max_lag
    best = int(lags[mask][np.argmax(np.abs(c[mask]))])
    def align(x, lag):
        if lag == 0:
            return x
        if lag > 0:
            return np.concatenate([np.zeros(lag), x[:-lag]])[:len(x)]
        return np.concatenate([x[-lag:], np.zeros(-lag)])[:len(x)]
    return best, my_si_sdr(ref, est), my_si_sdr(ref, align(est, best))


def stage_variants(n: int = 50, mouth_n: int = 20, device: str = "cuda:0") -> dict:
    banner(f"STAGE 3 — protocol variants (alignment / target / PIT / metric defs / mouth); N={n}")
    cache = np.load(CACHE)
    keys, est_all, ref_all, mix_all, mouth_all = (cache["keys"], cache["est"], cache["ref"],
                                                  cache["mix"], cache["mouth"])
    n = min(n, len(keys))
    res: dict = {}

    # -- 3.1 target convention s1 vs s2 & PIT/oracle --
    root = REBUILD / "audio/wav16k/min/tt"
    s2_refs, s1_target_ok = [], []
    for i in range(n):
        _, s2_id = s1_s2_ids(str(keys[i]))
        s2_refs.append(read_audio(root / "s2" / str(keys[i])))
    si_s1 = np.array([my_si_sdr(ref_all[i], est_all[i]) for i in range(n)])
    si_s2 = np.array([my_si_sdr(s2_refs[i], est_all[i]) for i in range(n)])
    base_s1 = np.array([my_si_sdr(ref_all[i], mix_all[i]) for i in range(n)])
    base_s2 = np.array([my_si_sdr(s2_refs[i], mix_all[i]) for i in range(n)])
    imp_s1, imp_s2 = si_s1 - base_s1, si_s2 - base_s2
    # oracle "PIT over targets" upper bound for a 1-output model (not a valid protocol, upper bound only)
    oracle_imp = np.maximum(imp_s1, imp_s2)
    res["target_convention"] = {
        "n": n,
        "si_sdr_i_target_s1_mean": float(imp_s1.mean()),
        "si_sdr_i_target_s2_mean": float(imp_s2.mean()),
        "delta_s1_minus_s2": float(imp_s1.mean() - imp_s2.mean()),
        "oracle_target_pit_mean": float(oracle_imp.mean()),
        "oracle_minus_s1": float(oracle_imp.mean() - imp_s1.mean()),
        "n_items_where_s2_better": int((imp_s2 > imp_s1).sum()),
    }
    log(f"target convention: SI-SDRi(s1)={imp_s1.mean():.3f} dB vs SI-SDRi(s2)={imp_s2.mean():.3f} dB "
        f"(Δ={imp_s1.mean()-imp_s2.mean():+.3f}); s2 better on {(imp_s2>imp_s1).sum()}/{n} items")
    log(f"oracle 2-target 'PIT' upper bound (NOT a valid protocol): {oracle_imp.mean():.3f} dB "
        f"(+{oracle_imp.mean()-imp_s1.mean():.3f} vs s1)")

    # -- 3.2 metric definition variants --
    si_zero_mean = np.array([my_si_sdr(ref_all[i], est_all[i], zero_mean=True) for i in range(n)])
    si_raw = np.array([my_si_sdr(ref_all[i], est_all[i], zero_mean=False) for i in range(n)])
    sdr_512 = np.array([my_sdr(ref_all[i], est_all[i], 512) for i in range(n)])
    sdr_2048 = np.array([my_sdr(ref_all[i], est_all[i], 2048) for i in range(n)])
    sdr_cs_true = np.array([my_sdr(ref_all[i], est_all[i], 512, change_sign=True) for i in range(n)])
    core_sdr = np.array([fast_bss_sdr_pit_loss(est_all[i], ref_all[i], 512) for i in range(n)])
    # 8 kHz metric convention (downsample both est & target; 8k wavs also exist officially)
    from scipy.signal import resample_poly
    def at8k(x):
        return resample_poly(np.asarray(x, dtype=np.float64), up=1, down=2)
    si8k = np.array([my_si_sdr(at8k(ref_all[i]), at8k(est_all[i])) for i in range(n)])
    sdr8k = np.array([my_sdr(at8k(ref_all[i]), at8k(est_all[i]), 512) for i in range(n)])
    res["metric_variants"] = {
        "n": n,
        "si_sdr_zero_mean_mean": float(si_zero_mean.mean()),
        "si_sdr_no_zero_mean_mean": float(si_raw.mean()),
        "delta_no_zero_mean": float(si_raw.mean() - si_zero_mean.mean()),
        "sdr_512_mean": float(sdr_512.mean()),
        "sdr_2048_mean": float(sdr_2048.mean()),
        "delta_sdr_2048": float(sdr_2048.mean() - sdr_512.mean()),
        "sdr_change_sign_true_mean": float(sdr_cs_true.mean()),
        "sdr_change_sign_relation": float((sdr_cs_true + sdr_512).mean()),
        "core_sdr_pit_loss_mean": float(core_sdr.mean()),
        "maxabs_core_vs_my_sdr": float(np.max(np.abs(core_sdr - sdr_512))),
        "si_sdr_8k_mean": float(si8k.mean()),
        "delta_si_sdr_8k": float(si8k.mean() - si_zero_mean.mean()),
        "sdr_8k_mean": float(sdr8k.mean()),
        "delta_sdr_8k": float(sdr8k.mean() - sdr_512.mean()),
    }
    log(f"metric defs: SI-SDR zero-mean={si_zero_mean.mean():.3f} vs raw={si_raw.mean():.3f} "
        f"(Δ={si_raw.mean()-si_zero_mean.mean():+.3f}); SDR filter512={sdr_512.mean():.3f} "
        f"vs 2048={sdr_2048.mean():.3f} (Δ={sdr_2048.mean()-sdr_512.mean():+.3f}); "
        f"change_sign=True = -default (sum≈0: {(sdr_cs_true+sdr_512).mean():.2e})")
    log(f"           core.py SDR (=-sdr_pit_loss) vs my fast_bss_eval.sdr: max|Δ|={np.max(np.abs(core_sdr-sdr_512)):.2e}")
    log(f"           8 kHz metric convention: ΔSI-SDR={si8k.mean()-si_zero_mean.mean():+.3f}, "
        f"ΔSDR={sdr8k.mean()-sdr_512.mean():+.3f}")

    # -- 3.3 alignment / lag / clipping / scale invariance --
    lows, si0, sibest, lags = [], [], [], []
    imp0, impbest = [], []
    clip_frac, clip_delta = [], []
    for i in range(n):
        lag, s0, sb = _best_lag(ref_all[i], est_all[i])
        lags.append(lag); si0.append(s0); sibest.append(sb)
        imp0.append(s0 - base_s1[i])
        impbest.append(sb - base_s1[i])
        peak = np.abs(est_all[i]).max()
        frac = float((np.abs(est_all[i]) > 1.0).mean())
        clip_frac.append(frac)
        clipped = np.clip(est_all[i], -1.0, 1.0)
        clip_delta.append(my_si_sdr(ref_all[i], clipped) - s0)
    lead_ms = np.array(lags) / SAMPLE_RATE * 1000.0
    res["alignment"] = {
        "n": n,
        "lag_samples_mode": int(pd.Series(lags).mode().iloc[0]),
        "lag_samples_mean": float(np.mean(lags)),
        "lag_ms_mean": float(lead_ms.mean()),
        "abs_lag_gt_16_count": int((np.abs(lags) > 16).sum()),
        "si_sdr_i_no_shift_mean": float(np.mean(imp0)),
        "si_sdr_i_best_shift_mean": float(np.mean(impbest)),
        "alignment_gain_mean": float(np.mean(impbest) - np.mean(imp0)),
        "max_single_item_gain": float(np.max(np.array(impbest) - np.array(imp0))),
        "est_peak_median": float(np.median([np.abs(est_all[i]).max() for i in range(n)])),
        "clipped_fraction_median": float(np.median(clip_frac)),
        "clipping_metric_delta_mean": float(np.mean(clip_delta)),
        "scale_invariance_0p1_db": float(my_si_sdr(ref_all[0], est_all[0] * 0.1) - my_si_sdr(ref_all[0], est_all[0])),
        "scale_invariance_10_db": float(my_si_sdr(ref_all[0], est_all[0] * 10) - my_si_sdr(ref_all[0], est_all[0])),
    }
    log(f"alignment: lag mode={pd.Series(lags).mode().iloc[0]} samples, mean={np.mean(lags):+.1f} "
        f"({lead_ms.mean():+.3f} ms), |lag|>16 on {(np.abs(lags)>16).sum()}/{n}")
    log(f"           SI-SDRi @0={np.mean(imp0):.3f} dB, @best-shift={np.mean(impbest):.3f} dB, "
        f"alignment gain={np.mean(impbest)-np.mean(imp0):+.3f} dB (max item {np.max(np.array(impbest)-np.array(imp0)):+.3f})")
    log(f"clipping: est peak median={np.median([np.abs(est_all[i]).max() for i in range(n)]):.3f}, "
        f"clipped fraction median={np.median(clip_frac):.4f}, metric Δ if clipped={np.mean(clip_delta):+.4f} dB")
    log(f"scale invariance: SI-SDR(0.1·est)−SI-SDR(est)={res['alignment']['scale_invariance_0p1_db']:+.2e}, "
        f"×10 → {res['alignment']['scale_invariance_10_db']:+.2e}")

    # -- 3.4 mouth branch ablation / shift / wrong speaker --
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = _load_ds()
    model, video_model, avt = _load_small_models(device)
    from look2hear.datas.transform import get_preprocessing_pipelines
    from torchvision import transforms
    prep = get_preprocessing_pipelines()["val"]
    gray = transforms.Grayscale(num_output_channels=1)

    def prep_npz_mouth(sid: str) -> np.ndarray:
        z = np.load(REBUILD / "mouths" / f"{sid}.npz")["data"]  # (50,96,96) uint8
        t = torch.from_numpy(z).unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)  # (50,3,96,96)
        return prep(gray(t).squeeze(1).numpy())

    idxs = list(range(min(mouth_n, len(keys))))
    base = {"si_sdr_i": [], "si_sdr": []}
    zero = {"si_sdr_i": [], "si_sdr": []}
    shift_p2 = {"si_sdr_i": [], "si_sdr": []}
    shift_m2 = {"si_sdr_i": [], "si_sdr": []}
    wrong = {"si_sdr_i": [], "si_sdr": []}
    refs = {i: ref_all[i] for i in idxs}
    mixes = {i: mix_all[i] for i in idxs}

    def _run_mouth(transform):
        out = _forward_items(model, video_model, ds, avt, idxs, device, mouth_transform=transform)
        vals = {"si_sdr": [], "si_sdr_i": []}
        for i, est in enumerate(out["est"]):
            r, m = refs[idxs[i]], mixes[idxs[i]]
            vals["si_sdr"].append(my_si_sdr(r, est)); vals["si_sdr_i"].append(my_si_sdr_i(r, est, m))
        return vals

    res["mouth"] = {"n": len(idxs)}
    res["mouth"]["baseline"] = {k: float(np.mean(v)) for k, v in _run_mouth(None).items()}
    mb = res["mouth"]["baseline"]["si_sdr_i"]
    for off in (-2, -1, 1, 2):
        key = f"shift_{off:+d}_frames"
        vals = _run_mouth(lambda mouth, key, off=off: np.roll(mouth, off, axis=0))
        res["mouth"][key] = {k: float(np.mean(v)) for k, v in vals.items()}
        log(f"mouth offset {off:+d} frame(s) ({off*40:+d} ms): SI-SDRi={res['mouth'][key]['si_sdr_i']:.3f} dB "
            f"(Δ vs real={res['mouth'][key]['si_sdr_i']-mb:+.3f} dB)")
    vals = _run_mouth(lambda mouth, key: np.zeros_like(mouth, dtype=np.float32))
    res["mouth"]["zero_mouth"] = {k: float(np.mean(v)) for k, v in vals.items()}
    log(f"mouth variant zero_mouth: SI-SDRi={res['mouth']['zero_mouth']['si_sdr_i']:.3f} dB "
        f"(Δ vs real={res['mouth']['zero_mouth']['si_sdr_i']-mb:+.3f} dB)")
    vals = _run_mouth(lambda mouth, key: prep_npz_mouth(s1_s2_ids(str(key))[1]))
    res["mouth"]["wrong_speaker_mouth"] = {k: float(np.mean(v)) for k, v in vals.items()}
    log(f"mouth variant wrong_speaker (s2): SI-SDRi={res['mouth']['wrong_speaker_mouth']['si_sdr_i']:.3f} dB "
        f"(Δ vs real={res['mouth']['wrong_speaker_mouth']['si_sdr_i']-mb:+.3f} dB)")
    best_off = max([0] + [-2, -1, 1, 2], key=lambda o: res["mouth"].get(f"shift_{o:+d}_frames", {"si_sdr_i": -1e9})["si_sdr_i"] if o != 0 else mb)
    res["mouth"]["best_offset_frames"] = int(best_off)
    log(f"mouth offset sweep: best offset = {best_off:+d} frame(s); real (0) SI-SDRi={mb:.3f} dB")

    # -- 3.5 segmentation / per-clip protocol --
    lengths = res.setdefault("segmentation", {})
    lengths["all_clip_lengths"] = sorted({int(read_audio(root / "mix" / str(k)).size) for k in keys})
    lengths["n_eval_clips"] = int(len(pd.read_csv(CSV)) - 2)
    log(f"segmentation: dataset clips are {lengths['all_clip_lengths']} samples = 2.0 s each; "
        f"evaluation is per-clip (paper LRS2-2Mix protocol), no full-utterance aggregation exists in this package")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: Falcon-Big vs Small + training-corpus evidence
# ══════════════════════════════════════════════════════════════════════════════
def _load_big_models(device):
    sys.path.insert(0, str(REPO / "studio"))
    from falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2
    from falcon_model.videomodels.frcnn_videomodel import FRCNNVideoModel
    cfg = yaml.safe_load((REPO / "studio/configs/FalconBig.yml").read_text())
    acfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
    vcfg = copy.deepcopy(cfg["videonet"]["videonet_config"])
    vcfg["pretrain"] = None
    ck = torch.load(str(BIG_CKPT), map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    audio_sd = {k[len("audio_model."):]: v for k, v in state.items() if k.startswith("audio_model.")}
    video_sd = {k[len("video_model."):]: v for k, v in state.items() if k.startswith("video_model.")}
    model = SpeakerContrastiveV2(sample_rate=16000, **acfg)
    model.load_state_dict(audio_sd, strict=True)
    video = FRCNNVideoModel(**vcfg)
    video.load_state_dict(video_sd, strict=True)
    model.to(device).eval(); video.to(device).eval()
    return model, video


def stage_big(n: int = 8, device: str = "cuda:0") -> dict:
    banner(f"STAGE 4 — Falcon-Big vs Small on the same N={n}; training-corpus evidence")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    cache = np.load(CACHE)
    keys, ref_all, mix_all, mouth_all = cache["keys"], cache["ref"], cache["mix"], cache["mouth"]
    n = min(n, len(keys))
    small_si = np.array([my_si_sdr(ref_all[i], cache["est"][i]) for i in range(n)])
    small_sii = np.array([my_si_sdr_i(ref_all[i], cache["est"][i], mix_all[i]) for i in range(n)])
    model, video = _load_big_models(device)
    ds = _load_ds()
    from look2hear.datas.transform import get_preprocessing_pipelines  # noqa: F401
    avt = __import__("audio_visual_test")
    big_sii, big_si = [], []
    with torch.inference_mode():
        for i in range(n):
            mix_t, src_t, mouth_np, key, _ = ds[i]
            mouth_in = avt._mouth_to_tensor(np.asarray(mouth_np)).float().to(device)
            emb = video(mouth_in)
            est = model(mix_t.unsqueeze(0).to(device), emb)
            if isinstance(est, tuple):
                est = est[0]
            est = est[0, 0].detach().float().cpu().numpy()
            r, m = ref_all[i], mix_all[i]
            big_si.append(my_si_sdr(r, est)); big_sii.append(my_si_sdr_i(r, est, m))
    res = {
        "n": n,
        "small_si_sdr_mean": float(small_si.mean()), "small_si_sdr_i_mean": float(small_sii.mean()),
        "big_si_sdr_mean": float(np.mean(big_si)), "big_si_sdr_i_mean": float(np.mean(big_sii)),
        "big_minus_small_si_sdr_i": float(np.mean(big_sii) - small_sii.mean()),
    }
    log(f"Falcon-Small (falcon.pt):     SI-SDR={small_si.mean():.3f}, SI-SDRi={small_sii.mean():.3f}")
    log(f"Falcon-Big (falcon-big.pt):   SI-SDR={np.mean(big_si):.3f}, SI-SDRi={np.mean(big_sii):.3f} "
        f"(Δ vs small={np.mean(big_sii)-small_sii.mean():+.3f} dB)")

    # training-corpus evidence of the released falcon.pt
    conf = yaml.safe_load((REPO / "Experiments/checkpoint/Falcon/conf.yml").read_text())
    dc = conf["datamodule"]["data_config"]
    best = json.loads((REPO / "Experiments/checkpoint/Falcon/best_k_models.json").read_text())
    ck = torch.load(str(SMALL_CKPT), map_location="cpu", weights_only=False)
    res["small_checkpoint"] = {
        "architecture": ck.get("architecture"), "epoch": ck.get("epoch"),
        "best_k_models": {Path(k).name: v for k, v in best.items()},
    }
    res["training_corpus_evidence"] = {
        "conf_snapshot": "Experiments/checkpoint/Falcon/conf.yml",
        "train_dir": dc.get("train_dir"), "valid_dir": dc.get("valid_dir"), "test_dir": dc.get("test_dir"),
        "segment": dc.get("segment"), "sample_rate": dc.get("sample_rate"),
    }
    log(f"released falcon.pt: arch={ck.get('architecture')} epoch={ck.get('epoch')}")
    log(f"training-run conf snapshot datamodule dirs = {dc.get('train_dir')}")
    log("=> falcon.pt was trained on LRS3_complex_nonmoving-derived mixtures, NOT LRS2-2Mix "
        "(cross-corpus evaluation; paper's LRS2 column is under the paper's per-benchmark training protocol).")
    return res



# ══════════════════════════════════════════════════════════════════════════════
# Stage 2b/3b: cache-based recheck (no model forward; CPU-only)
# ══════════════════════════════════════════════════════════════════════════════
def _cache():
    z = np.load(CACHE)
    return z["keys"], z["est"], z["ref"], z["mix"], z["mouth"]


def stage_recheck(n: int = 50) -> dict:
    """Recompute all metric/protocol cross-checks from the cached N=50 estimates.

    The estimates in CACHE were produced by `--stage metrics` with the official weights
    (GPU); this stage is pure CPU and does not need a GPU, so it can be re-run at any time.
    """
    banner(f"STAGE 2b/3b — cache-based recheck (N={n}); no model forward, CPU only")
    keys, est_all, ref_all, mix_all, _ = _cache()
    n = min(n, len(keys))
    df = pd.read_csv(CSV)
    rows = df[~df.snt_id.isin(["avg", "std"])].reset_index(drop=True)
    rows_n = rows.iloc[:n]
    order_ok = sum(str(keys[i]) == rows_n.snt_id.iloc[i] for i in range(n))
    log(f"CSV rows={len(rows)}; dataset-order match={order_ok}/{n}; "
        f"CSV avg si-snr_i={df[df.snt_id=='avg']['si-snr_i'].iloc[0]:.6f} "
        f"sdr_i={df[df.snt_id=='avg'].sdr_i.iloc[0]:.6f}")
    agg_ok = all(abs(rows[c].mean() - df[df.snt_id == "avg"][c].iloc[0]) < 1e-9 and
                 abs(rows[c].std(ddof=0) - df[df.snt_id == "std"][c].iloc[0]) < 1e-9
                 for c in ["sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"])

    per_item, d_core, d_csv, d_base, d_base_sdr = [], [], [], [], []
    for i in range(n):
        est, ref, mix = est_all[i], ref_all[i], mix_all[i]
        my = {"si_sdr": my_si_sdr(ref, est), "si_sdr_i": my_si_sdr_i(ref, est, mix),
              "sdr": my_sdr(ref, est), "sdr_i": my_sdr(ref, est) - my_sdr(ref, mix)}
        from look2hear.metrics import core
        corev = {"si_sdr": core.si_sdr(est, ref), "si_sdr_i": core.si_sdr_improvement(est, ref, mix),
                 "sdr": core.sdr(est, ref), "sdr_i": core.sdr_improvement(est, ref, mix)}
        r = rows_n.iloc[i]
        csvv = {"si_sdr": float(r["si-snr"]), "si_sdr_i": float(r["si-snr_i"]),
                "sdr": float(r["sdr"]), "sdr_i": float(r["sdr_i"])}
        d_core.append(max(abs(my[k] - corev[k]) for k in my))
        d_csv.append(max(abs(my[k] - csvv[k]) for k in my))
        d_base.append(abs(my_si_sdr(ref, mix) - (csvv["si_sdr"] - csvv["si_sdr_i"])))
        d_base_sdr.append(abs(my_sdr(ref, mix) - (csvv["sdr"] - csvv["sdr_i"])))
        per_item.append({"key": str(keys[i]), "my": my, "core": corev, "csv": csvv,
                         "baseline_si_sdr": my_si_sdr(ref, mix),
                         "baseline_sdr": my_sdr(ref, mix)})
    res_metrics = {
        "n": n, "csv_avg_recompute_ok": bool(agg_ok), "dataset_order_match": f"{order_ok}/{n}",
        "maxabs_my_vs_core_db": float(max(d_core)),
        "maxabs_my_vs_csv_db": float(max(d_csv)),
        "maxabs_baseline_formula_db": float(max(d_base)),
        "maxabs_baseline_sdr_formula_db": float(max(d_base_sdr)),
        "csv_avg": {c: float(df[df.snt_id == "avg"][c].iloc[0]) for c in
                    ["sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"]},
        "subset_mean": {"si_sdr_i": float(np.mean([p["my"]["si_sdr_i"] for p in per_item])),
                        "sdr_i": float(np.mean([p["my"]["sdr_i"] for p in per_item]))},
        "per_item": per_item,
    }
    log(f"max|my − core.py| = {res_metrics['maxabs_my_vs_core_db']:.3e} dB ; "
        f"max|my − Phase-1 CSV| = {res_metrics['maxabs_my_vs_csv_db']:.3e} dB ; "
        f"max|baseline SI-SDR − (CSV si-snr−si-snr_i)| = {res_metrics['maxabs_baseline_formula_db']:.3e} dB ; "
        f"max|baseline SDR − (CSV sdr−sdr_i)| = {res_metrics['maxabs_baseline_sdr_formula_db']:.3e} dB")
    log(f"CSV avg/std == per-row recomputation: {agg_ok}; N={n} subset means: "
        f"SI-SDRi={res_metrics['subset_mean']['si_sdr_i']:.4f}, SDRi={res_metrics['subset_mean']['sdr_i']:.4f}")

    # ---- variants that need no model forward ----
    root = REBUILD / "audio/wav16k/min/tt"
    s2_refs = [read_audio(root / "s2" / str(keys[i])) for i in range(n)]
    si_s1 = np.array([my_si_sdr(ref_all[i], est_all[i]) for i in range(n)])
    si_s2 = np.array([my_si_sdr(s2_refs[i], est_all[i]) for i in range(n)])
    base_s1 = np.array([my_si_sdr(ref_all[i], mix_all[i]) for i in range(n)])
    base_s2 = np.array([my_si_sdr(s2_refs[i], mix_all[i]) for i in range(n)])
    imp_s1, imp_s2 = si_s1 - base_s1, si_s2 - base_s2
    oracle_imp = np.maximum(imp_s1, imp_s2)
    si_zm0 = np.array([my_si_sdr(ref_all[i], est_all[i], zero_mean=False) for i in range(n)])
    sdr_512 = np.array([my_sdr(ref_all[i], est_all[i], 512) for i in range(n)])
    sdr_2048 = np.array([my_sdr(ref_all[i], est_all[i], 2048) for i in range(n)])
    sdr_cs = np.array([my_sdr(ref_all[i], est_all[i], 512, change_sign=True) for i in range(n)])
    from scipy.signal import resample_poly
    at8k = lambda x: resample_poly(np.asarray(x, dtype=np.float64), up=1, down=2)  # noqa: E731
    si8k = np.array([my_si_sdr(at8k(ref_all[i]), at8k(est_all[i])) for i in range(n)])
    sdr8k = np.array([my_sdr(at8k(ref_all[i]), at8k(est_all[i]), 512) for i in range(n)])
    lags, impbest = [], []
    for i in range(n):
        lag, s0, sb = _best_lag(ref_all[i], est_all[i])
        lags.append(lag); impbest.append(sb - base_s1[i])
    clip_frac = [float((np.abs(est_all[i]) > 1.0).mean()) for i in range(n)]
    clip_delta = [my_si_sdr(ref_all[i], np.clip(est_all[i], -1, 1)) - si_s1[i] for i in range(n)]
    prev_path = PHASE2 / "P2F-results.json"
    prev_mouth = None
    if prev_path.exists():
        try:
            prev_mouth = json.loads(prev_path.read_text()).get("variants", {}).get("mouth")
        except Exception:  # noqa: BLE001
            prev_mouth = None
    res_variants = {
        "n": n,
        "target_convention": {
            "si_sdr_i_target_s1_mean": float(imp_s1.mean()),
            "si_sdr_i_target_s2_mean": float(imp_s2.mean()),
            "delta_s1_minus_s2": float(imp_s1.mean() - imp_s2.mean()),
            "oracle_target_pit_mean": float(oracle_imp.mean()),
            "oracle_minus_s1": float(oracle_imp.mean() - imp_s1.mean()),
            "n_items_where_s2_better": int((imp_s2 > imp_s1).sum()),
        },
        "metric_variants": {
            "si_sdr_zero_mean_mean": float(si_s1.mean()),
            "si_sdr_no_zero_mean_mean": float(si_zm0.mean()),
            "delta_no_zero_mean": float(si_zm0.mean() - si_s1.mean()),
            "sdr_512_mean": float(sdr_512.mean()),
            "sdr_2048_mean": float(sdr_2048.mean()),
            "delta_sdr_2048": float(sdr_2048.mean() - sdr_512.mean()),
            "sdr_change_sign_true_mean": float(sdr_cs.mean()),
            "change_sign_is_exact_negation": float(np.max(np.abs(sdr_cs + sdr_512))),
            "si_sdr_8k_mean": float(si8k.mean()),
            "delta_si_sdr_8k": float(si8k.mean() - si_s1.mean()),
            "sdr_8k_mean": float(sdr8k.mean()),
            "delta_sdr_8k": float(sdr8k.mean() - sdr_512.mean()),
        },
        "alignment": {
            "lag_mode": int(pd.Series(lags).mode().iloc[0]),
            "lag_mean_samples": float(np.mean(lags)),
            "abs_lag_gt_16_count": int((np.abs(lags) > 16).sum()),
            "si_sdr_i_no_shift_mean": float(imp_s1.mean()),
            "si_sdr_i_best_shift_mean": float(np.mean(impbest)),
            "alignment_gain_mean": float(np.mean(impbest) - imp_s1.mean()),
            "max_single_item_gain": float(np.max(np.array(impbest) - imp_s1)),
            "est_peak_median": float(np.median([np.abs(est_all[i]).max() for i in range(n)])),
            "clipped_fraction_median": float(np.median(clip_frac)),
            "clipping_metric_delta_mean": float(np.mean(clip_delta)),
            "scale_invariance_0p1_db": float(my_si_sdr(ref_all[0], est_all[0] * 0.1) - my_si_sdr(ref_all[0], est_all[0])),
            "scale_invariance_10_db": float(my_si_sdr(ref_all[0], est_all[0] * 10) - my_si_sdr(ref_all[0], est_all[0])),
        },
        "segmentation": {"all_clip_lengths": sorted({int(read_audio(root / "mix" / str(keys[i])).size) for i in range(n)}),
                         "n_eval_clips": int(len(rows))},
        "mouth": prev_mouth if (isinstance(prev_mouth, dict) and "baseline" in prev_mouth) else {
            "note": "mouth sensitivity measured on GPU in the first `--stage variants` run (N=20 first-20 subset, "
                    "official weights) before GPU handoff; re-run `--stage variants --device cuda:1` to reproduce. "
                    "Deltas are relative to this N=20 baseline 7.9106 dB (Phase-1 20-utt anchor), not the N=50 "
                    "target baseline 7.5347 dB.",
            "n": 20,
            "baseline_si_sdr_i": 7.9106, "zero_mouth_delta_db": -14.399,
            "shift_plus2_frames_delta_db": -6.353, "shift_minus2_frames_delta_db": -5.079,
            "wrong_speaker_delta_db": -23.297,
        },
    }
    tv = res_variants["target_convention"]; mv = res_variants["metric_variants"]; al = res_variants["alignment"]
    log(f"target convention: SI-SDRi(s1)={tv['si_sdr_i_target_s1_mean']:.3f} vs (s2)={tv['si_sdr_i_target_s2_mean']:.3f} dB "
        f"(Δ={tv['delta_s1_minus_s2']:+.3f}); PIT/oracle upper bound={tv['oracle_target_pit_mean']:.3f} "
        f"({tv['oracle_minus_s1']:+.3f})")
    log(f"metric defs: zero-mean Δ={mv['delta_no_zero_mean']:+.4f}; SDR filter 512→2048 Δ={mv['delta_sdr_2048']:+.3f}; "
        f"change_sign exact negation max|sum|={mv['change_sign_is_exact_negation']:.1e}; "
        f"8 kHz ΔSI-SDR={mv['delta_si_sdr_8k']:+.3f}, ΔSDR={mv['delta_sdr_8k']:+.3f}")
    log(f"alignment: lag mode={al['lag_mode']}, |lag|>16 on {al['abs_lag_gt_16_count']}/{n}, "
        f"best-shift gain={al['alignment_gain_mean']:+.4f} dB; clipping Δ={al['clipping_metric_delta_mean']:+.4f} dB; "
        f"scale invariance: {al['scale_invariance_0p1_db']:+.1e}/{al['scale_invariance_10_db']:+.1e} dB")
    return {"metrics": res_metrics, "variants": res_variants}


def stage_static() -> dict:
    """Training-corpus / checkpoint evidence for the released falcon.pt (no model forward)."""
    banner("STAGE 4b — released falcon.pt training-corpus evidence (static)")
    conf = yaml.safe_load((REPO / "Experiments/checkpoint/Falcon/conf.yml").read_text())
    dc = conf["datamodule"]["data_config"]
    best = json.loads((REPO / "Experiments/checkpoint/Falcon/best_k_models.json").read_text())
    ck = torch.load(str(SMALL_CKPT), map_location="cpu", weights_only=False)
    res = {
        "checkpoint": {"architecture": ck.get("architecture"), "epoch": ck.get("epoch"),
                       "best_k_models": {Path(k).name: v for k, v in best.items()}},
        "conf_snapshot": {"path": "Experiments/checkpoint/Falcon/conf.yml",
                          "exp_name": conf.get("exp", {}).get("exp_name"),
                          "train_dir": dc.get("train_dir"), "valid_dir": dc.get("valid_dir"),
                          "test_dir": dc.get("test_dir"), "segment": dc.get("segment"),
                          "sample_rate": dc.get("sample_rate"),
                          "normalize_audio": dc.get("normalize_audio")},
        "readme_claim": {"LRS2_2Mix_SI_SNRi": 14.1, "LRS2_2Mix_SDRi": 14.7,
                         "line": "README.md:211 (paper table); README.md:228 'under our training protocol'"},
    }
    log(f"falcon.pt: architecture={res['checkpoint']['architecture']} epoch={res['checkpoint']['epoch']}; "
        f"best_k_models epoch={[Path(k).name for k in best][0] if best else None}")
    log(f"training-run conf snapshot datamodule: train_dir={dc.get('train_dir')}")
    log("=> released falcon.pt was trained on LRS3_complex_nonmoving mixtures; LRS2-2Mix eval is cross-corpus.")
    return res

# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["data", "metrics", "variants", "big", "recheck", "static", "all"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--mouth-n", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    PHASE2.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.stage in ("data", "all"):
        RESULTS["data"] = stage_data()
    if args.stage in ("metrics", "all"):
        RESULTS["metrics"] = stage_metrics(args.n, args.device)
    if args.stage == "variants":
        RESULTS["variants"] = stage_variants(args.n, args.mouth_n, args.device)
    if args.stage in ("recheck", "all"):
        rc = stage_recheck(args.n)
        RESULTS["metrics"] = rc["metrics"]
        RESULTS["variants"] = rc["variants"]
    if args.stage in ("static", "all"):
        RESULTS["big"] = stage_static()
    if args.stage == "big":
        RESULTS["big"] = stage_big(min(args.n, 8), args.device)
    RESULTS["elapsed_s"] = round(time.time() - t0, 1)
    out = PHASE2 / "P2F-results.json"
    if out.exists():  # merge with previous stage runs instead of overwriting sections
        try:
            prev = json.loads(out.read_text())
            prev.update(RESULTS)
            RESULTS = prev
        except Exception:  # noqa: BLE001
            pass
    out.write_text(json.dumps(RESULTS, indent=2))
    (PHASE2 / "P2F-protocol-check.log").write_text("\n".join(LINES) + "\n")
    log(f"\nresults -> {out}")
    log(f"log     -> {PHASE2 / 'P2F-protocol-check.log'}")
    log(f"elapsed {RESULTS['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
