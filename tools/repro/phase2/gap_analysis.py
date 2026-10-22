"""Phase2-A: LRS2-2Mix ~7 dB gap analysis for the released Falcon weights.

Purpose (docs/REPRODUCTION_PLAN.md Phase 2-A / task-18): the released
``falcon.pt`` gives SI-SNRi 7.03 / SDRi 7.71 on our 3000-utterance LRS2-2Mix
test build, while the paper reports 14.1 / 14.7.  This script quantifies how much
of that gap can be explained by *evaluation-protocol choices* (metric
definition, target scaling, mouth input, audio-length convention) versus the
model/checkpoint itself.

What it runs (fixed subset, deterministic):
  1. data self-checks: mix == s1+s2, target == s1, mouth == s1's npz, mono/16 kHz/2 s,
     filename SIR vs measured mixture baseline;
  2. mouth-input variants  (s1 mouth / s2 mouth / zeros / 96-no-crop / temporal shifts /
     1-channel vs 3-channel-gray / no mean-std normalisation);
  3. audio-input variants  (full 2 s / first 1 s / zero-padded 3 s / peak-normalised / rms-normalised);
  4. metric variants on cached predictions (zero-mean on/off, scale-invariant SI-SDR vs plain SNR,
     fast_bss_eval SDR with filter_length 512/256/128/64, whole-utterance vs 2/4-segment averages,
     target scaled to mix energy/RMS, target=s2, best-of-s1/s2 = PIT upper bound);
  5. optional Falcon-Big comparison (``--models small,big``).

The pipeline is intentionally self-contained (it re-implements the validation
mouth transform of ``look2hear/datas``): no Phase-1 source file is modified.

Usage examples::

  # metric-pipeline anchor sanity (Phase-1 reference: SI-SNRi 7.911 / SDRi 8.547)
  python tools/repro/phase2/gap_analysis.py --limit 20 --subset first --models small --variants baseline

  # main run: 200 random utterances, small + big
  python tools/repro/phase2/gap_analysis.py --limit 200 --subset random --seed 0 --models small,big

  # only list variants
  python tools/repro/phase2/gap_analysis.py --list-variants
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]  # <repo>/tools/repro/phase2/<file>
for _p in (str(REPO), str(REPO / "studio")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torchvision import transforms  # noqa: E402

DEFAULT_EVAL_ROOT = "/root/tmp/lrs2_eval_tt_full"
DEFAULT_REBUILD_ROOT = "/root/data/lrs2_2mix/lrs2_rebuild"
DEFAULT_WEIGHTS = "/root/.cache/falcon_weights"
SR = 16000
CLIP_SAMPLES = 32000
N_FRAMES = 50

_GRAY = transforms.Grayscale(num_output_channels=1)

# ---------------------------------------------------------------------------
# Variant specs
# ---------------------------------------------------------------------------
BASELINE = "mouth_s1_88"

MOUTH_VARIANTS = [
    # name, source, crop, shift, normalize, three_ch
    (BASELINE, "s1", 88, 0, True, True),
    ("mouth_s2_88", "s2", 88, 0, True, True),
    ("mouth_zero_88", "zero", 88, 0, True, True),
    ("mouth_s1_96_nocrop", "s1", None, 0, True, True),
    ("mouth_s1_88_shift+5", "s1", 88, 5, True, True),
    ("mouth_s1_88_shift-5", "s1", 88, -5, True, True),
    ("mouth_s1_88_shift+12", "s1", 88, 12, True, True),
    ("mouth_s1_88_shift-12", "s1", 88, -12, True, True),
    ("mouth_s1_88_1ch", "s1", 88, 0, True, False),
    ("mouth_s1_88_nonorm", "s1", 88, 0, False, True),
]
AUDIO_VARIANTS = [
    # name, audio_mode
    ("audio_crop_1s", "crop1s"),
    ("audio_pad_3s", "pad3s"),
    ("audio_norm_peak", "norm_peak"),
    ("audio_norm_rms", "norm_rms"),
]
ALL_VARIANTS = [v[0] for v in MOUTH_VARIANTS] + [v[0] for v in AUDIO_VARIANTS]
LITE_VARIANTS = [BASELINE, "mouth_s2_88", "mouth_zero_88", "mouth_s1_96_nocrop", "audio_norm_peak"]

METRICS = [
    "si_snr", "si_snr_i",
    "si_snr_zm0", "si_snr_i_zm0",
    "snr_i_plain",
    "sdr512", "sdr512_i",
    "sdr256_i", "sdr128_i", "sdr64_i",
    "si_snr_i_seg2", "si_snr_i_seg4",
    "si_snr_i_tgt_s2", "si_snr_est_s2", "si_snr_i_best_target",
    "si_snr_i_tgt_rms_scale", "si_snr_i_tgt_energy_scale",
    "mix_si_snr", "mix_sdr512",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def parse_mixture_name(name: str) -> dict:
    """``<s1spk>_<s1seg>_<snr1>_<s2spk>_<s2seg>_<snr2>.wav`` -> ids/snrs."""
    stem = os.path.splitext(os.path.basename(name))[0]
    parts = stem.split("_")
    if len(parts) != 6:
        raise ValueError(f"unexpected mixture filename: {name!r}")
    return {
        "s1_id": f"{parts[0]}_{parts[1]}",
        "s2_id": f"{parts[3]}_{parts[4]}",
        "s1_snr": float(parts[2]),
        "s2_snr": float(parts[5]),
        "sir": float(parts[2]) - float(parts[5]),
    }


def load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float64")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr} for {path}")
    return np.ascontiguousarray(wav)


def load_gray_mouth_npz(path: str) -> np.ndarray:
    data = np.load(path)
    arr = data[data.files[0]] if len(data.files) == 1 else data
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(-1).round().astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"unexpected mouth npz shape {arr.shape} at {path}")
    return arr.astype(np.uint8)


def mouth_from_pt(path: str) -> np.ndarray:
    """Load the eval-set ``[F,96,96,3]`` uint8 mouth tensor and reduce to grayscale.

    Mirrors the validation preprocessing of ``look2hear/datas``:
    ``Grayscale(num_output_channels=1)`` on ``[F,3,96,96]``.
    """
    t = torch.load(path, map_location="cpu", weights_only=True)
    t = t.permute(0, 3, 1, 2)  # F,3,96,96
    gray = _GRAY(t).squeeze(1)  # F,96,96 uint8
    return gray.numpy()


def mouth_to_model_input(gray: np.ndarray, crop=88, shift=0, normalize=True, three_ch=True) -> torch.Tensor:
    """Build the ``[1,1,F,H,W]`` float32 model input.

    ``three_ch=True`` replicates the train/val pipeline (stack 3 gray channels ->
    torchvision Grayscale -> crop -> Normalize(0,255) -> Normalize(.421,.165)).
    ``three_ch=False`` passes the single-channel gray directly (numerically identical
    for replicated channels; kept as an explicit variant).
    """
    if shift:
        gray = np.roll(gray, int(shift), axis=0)
    if three_ch:
        stacked = np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)
        t = _GRAY(torch.from_numpy(stacked).permute(0, 3, 1, 2)).squeeze(1)
        g = t.float().numpy()
    else:
        g = gray.astype(np.float32)
    if crop:
        s = (g.shape[-1] - int(crop)) // 2
        g = g[:, s:s + int(crop), s:s + int(crop)]
    g = g / 255.0
    if normalize:
        g = (g - 0.421) / 0.165
    return torch.from_numpy(np.ascontiguousarray(g)).float()[None, None]


# ---------------------------------------------------------------------------
# Metrics (all positive = higher-is-better)
# ---------------------------------------------------------------------------
def _zm(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def si_sdr_pair(est: np.ndarray, ref: np.ndarray, zero_mean: bool = True,
                scale_invariant: bool = True) -> float:
    """SI-SDR (scale-invariant) or scale-dependent SNR, in dB, positive convention."""
    e = np.asarray(est, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if zero_mean:
        e, r = _zm(e), _zm(r)
    denom = float((r * r).sum()) + 1e-12
    if scale_invariant:
        alpha = float((e * r).sum()) / denom
        proj = alpha * r
    else:
        proj = r
    noise = e - proj
    ratio = float((proj * proj).sum()) / (float((noise * noise).sum()) + 1e-12)
    return 10.0 * math.log10(ratio + 1e-12)


def si_sdr_improvement(est, mix, ref, zero_mean=True, scale_invariant=True) -> float:
    return (si_sdr_pair(est, ref, zero_mean, scale_invariant)
            - si_sdr_pair(mix, ref, zero_mean, scale_invariant))


def si_sdr_improvement_segments(est, mix, ref, n_seg: int) -> float:
    """Mean of per-segment SI-SDRi over ``n_seg`` equal segments (zero-mean, scale-invariant)."""
    length = len(ref)
    bounds = [(i * length) // n_seg for i in range(n_seg + 1)]
    values = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 32:  # skip degenerate segments
            continue
        values.append(si_sdr_improvement(est[a:b], mix[a:b], ref[a:b]))
    return float(np.mean(values)) if values else float("nan")


def sdr_value(est: np.ndarray, ref: np.ndarray, filter_length: int, device: torch.device) -> float:
    """Positive SDR in dB via fast_bss_eval (BSS-Eval-v4 style), CPU or GPU."""
    import fast_bss_eval
    e = torch.from_numpy(np.ascontiguousarray(est.astype(np.float32)))[None, None].to(device)
    r = torch.from_numpy(np.ascontiguousarray(ref.astype(np.float32)))[None, None].to(device)
    loss = fast_bss_eval.sdr_pit_loss(e, r, filter_length=filter_length)
    return -float(loss.mean())


def sdr_improvement(est, mix, ref, filter_length: int, device) -> float:
    return (sdr_value(est, ref, filter_length, device)
            - sdr_value(mix, ref, filter_length, device))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _device(arg: str) -> torch.device:
    if arg == "cpu" or not torch.cuda.is_available():
        if arg != "cpu":
            print("[gap] CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    idx = int(str(arg).split(":")[-1]) if ":" in str(arg) else 0
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def load_small_model(ckpt_path: str, device: torch.device):
    import look2hear.models
    import look2hear.videomodels

    conf = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
    audio_cfg = conf["audionet"]["audionet_config"]
    video_cfg = dict(conf["videonet"]["videonet_config"])
    video_cfg.pop("pretrain", None)
    audio = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
        sample_rate=SR, **audio_cfg).to(device).eval()
    video = getattr(look2hear.videomodels, conf["videonet"]["videonet_name"])(
        **video_cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    a = {k[len("audio_model."):]: v for k, v in state.items() if k.startswith("audio_model.")}
    v = {k[len("video_model."):]: t for k, t in state.items() if k.startswith("video_model.")}
    audio.load_state_dict(a, strict=True)
    video.load_state_dict(v, strict=True)
    return audio, video, {"audio_keys": len(a), "video_keys": len(v),
                          "audio_params": sum(p.numel() for p in audio.parameters()),
                          "video_params": sum(p.numel() for p in video.parameters())}


def load_big_model(ckpt_path: str, device: torch.device):
    from falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2
    from falcon_model.videomodels.frcnn_videomodel import FRCNNVideoModel

    conf = yaml.safe_load((REPO / "studio" / "configs" / "FalconBig.yml").read_text())
    audio_cfg = conf["audionet"]["audionet_config"]
    video_cfg = dict(conf["videonet"]["videonet_config"])
    video_cfg.pop("pretrain", None)
    audio = SpeakerContrastiveV2(sample_rate=SR, **audio_cfg).to(device).eval()
    video = FRCNNVideoModel(**video_cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    a = {k[len("audio_model."):]: v for k, v in state.items() if k.startswith("audio_model.")}
    v = {k[len("video_model."):]: t for k, t in state.items() if k.startswith("video_model.")}
    audio.load_state_dict(a, strict=True)
    video.load_state_dict(v, strict=True)
    return audio, video, {"audio_keys": len(a), "video_keys": len(v),
                          "audio_params": sum(p.numel() for p in audio.parameters()),
                          "video_params": sum(p.numel() for p in video.parameters())}


@torch.no_grad()
def predict(audio_model, video_model, mix: np.ndarray, mouth_input: torch.Tensor,
            device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(mix.astype(np.float32)))[None].to(device)
    emb = video_model(mouth_input.to(device))
    out = audio_model(x, emb)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.squeeze(0)                      # drop batch
    if out.dim() == 2 and out.shape[0] == 1:  # n_src == 1 -> [T]
        out = out[0]
    return out.float().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def select_indices(n_total: int, limit: int, subset: str, seed: int) -> np.ndarray:
    if subset == "first":
        return np.arange(min(limit or n_total, n_total))
    if subset == "even":
        idx = np.linspace(0, n_total - 1, num=min(limit or n_total, n_total)).round().astype(int)
        return np.unique(idx)
    rng = np.random.default_rng(seed)
    n = min(limit or n_total, n_total)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def variant_specs(names):
    specs = []
    for name, src, crop, shift, norm, three_ch in MOUTH_VARIANTS:
        if name in names:
            specs.append({"name": name, "kind": "mouth", "source": src, "crop": crop,
                          "shift": shift, "normalize": norm, "three_ch": three_ch})
    for name, mode in AUDIO_VARIANTS:
        if name in names:
            specs.append({"name": name, "kind": "audio", "mode": mode})
    return specs


def build_mouth(spec, gray_s1: np.ndarray, gray_s2: np.ndarray) -> torch.Tensor:
    """Build the model mouth input from cached grayscale mouth arrays (s1 = target speaker)."""
    source = spec["source"]
    if source == "zero":
        gray = np.zeros((N_FRAMES, 96, 96), dtype=np.uint8)
    elif source in ("s1", "s1_npz"):
        gray = gray_s1
    elif source == "s2":
        gray = gray_s2
    else:
        raise ValueError(source)
    return mouth_to_model_input(gray, crop=spec["crop"], shift=spec["shift"],
                                normalize=spec["normalize"], three_ch=spec["three_ch"])


def build_audio(spec, mix: np.ndarray):
    """Return (audio_input, eval_slice, mouth_frame_slice, label)."""
    mode = spec["mode"]
    if mode == "full":
        return mix, slice(None), slice(None)
    if mode == "crop1s":
        return mix[:16000], slice(0, 16000), slice(0, 25)
    if mode == "pad3s":
        return np.pad(mix, (0, CLIP_SAMPLES // 2)), slice(0, CLIP_SAMPLES), slice(None)
    if mode == "norm_peak":
        peak = float(np.abs(mix).max()) + 1e-12
        return mix * (0.9 / peak), slice(None), slice(None)
    if mode == "norm_rms":
        std = float(mix.std()) + 1e-12
        return mix / std, slice(None), slice(None)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--rebuild_root", default=DEFAULT_REBUILD_ROOT)
    ap.add_argument("--weights_dir", default=DEFAULT_WEIGHTS)
    ap.add_argument("--models", default="small", help="comma list: small,big")
    ap.add_argument("--limit", type=int, default=200, help="number of utterances (0 = all)")
    ap.add_argument("--subset", choices=["random", "first", "even"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="all", help="'all', 'lite', 'baseline', or comma list")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="utterances per forward (batched GPU inference; pad3s is capped at 2)")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--tag", default="", help="tag added to the default output filename")
    ap.add_argument("--list-variants", action="store_true")
    args = ap.parse_args()

    if args.list_variants:
        print("ALL :", ", ".join(ALL_VARIANTS))
        print("LITE:", ", ".join(LITE_VARIANTS))
        return

    if args.variants == "all":
        variant_names = ALL_VARIANTS
    elif args.variants == "lite":
        variant_names = LITE_VARIANTS
    elif args.variants == "baseline":
        variant_names = [BASELINE]
    else:
        variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in variant_names:
        if name not in ALL_VARIANTS:
            raise ValueError(f"unknown variant {name!r}; use --list-variants")
    specs = variant_specs(variant_names)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ("small", "big"):
            raise ValueError(f"unknown model {m!r}")

    device = _device(args.device)
    eval_dir = Path(args.eval_root) / "tt"
    mix_infos = json.loads((eval_dir / "mix.json").read_text())
    target_infos = json.loads((eval_dir / "target.json").read_text())
    if len(mix_infos) != len(target_infos):
        raise ValueError("mix.json/target.json length mismatch")
    indices = select_indices(len(mix_infos), args.limit, args.subset, args.seed)
    print(f"[gap] subset={args.subset} seed={args.seed} n={len(indices)} / {len(mix_infos)} | device={device}")
    print(f"[gap] variants={variant_names} | models={models}")

    rebuild = Path(args.rebuild_root)
    tt_audio = rebuild / "audio" / "wav16k" / "min" / "tt"
    results = {m: {v: {k: [] for k in METRICS} for v in variant_names} for m in models}
    data_checks = {k: [] for k in ("mix_residual_max", "mix_residual_rel", "mix_r2", "sir_filename",
                                   "mix_si_snr", "mix_sdr512", "eval_mix_equals_rebuild",
                                   "target_equals_s1", "mouth_equals_s1_npz",
                                   "len", "sr", "channels")}
    anchors_first20 = None

    models_loaded = {}
    loaders = {"small": load_small_model, "big": load_big_model}
    for m in models:
        ckpt = Path(args.weights_dir) / ("falcon.pt" if m == "small" else "falcon-big.pt")
        t0 = time.time()
        models_loaded[m] = loaders[m](str(ckpt), device)
        print(f"[gap] loaded {m} model in {time.time() - t0:.1f}s: {models_loaded[m][2]}")
        # warmup
        dummy = torch.zeros(1, CLIP_SAMPLES, device=device)
        dummy_mouth = torch.zeros(1, 1, N_FRAMES, 88, 88, device=device)
        with torch.no_grad():
            _ = models_loaded[m][0](dummy, models_loaded[m][1](dummy_mouth))
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- preload all utterances once (IO + data self-checks) ------------------
    records = []
    t_start = time.time()
    for count, idx in enumerate(indices):
        mix_path, _n = mix_infos[idx]
        target_path, mouth_path, _n2 = target_infos[idx]
        name = os.path.basename(mix_path)
        info = parse_mixture_name(name)
        mix = load_wav(mix_path)
        s1 = load_wav(target_path)
        s2 = load_wav(str(tt_audio / "s2" / name))

        # ---- data self-checks -------------------------------------------------
        residual = mix - (s1 + s2)
        alpha = float((mix @ (s1 + s2)) / ((s1 + s2) @ (s1 + s2) + 1e-12))
        r2 = 1.0 - float(((mix - alpha * (s1 + s2)) ** 2).sum()) / (
            float(((mix - mix.mean()) ** 2).sum()) + 1e-12)
        data_checks["mix_residual_max"].append(float(np.abs(residual).max()))
        data_checks["mix_residual_rel"].append(float(np.abs(residual).max() / (np.abs(mix).max() + 1e-12)))
        data_checks["mix_r2"].append(r2)
        data_checks["sir_filename"].append(info["sir"])
        data_checks["mix_si_snr"].append(si_sdr_pair(mix, s1))
        data_checks["mix_sdr512"].append(sdr_value(mix, s1, 512, device))
        if count < 20:  # file-identity checks on a 20-utterance sample (IO-heavy)
            data_checks["eval_mix_equals_rebuild"].append(
                float(np.abs(mix - load_wav(str(tt_audio / "mix" / name))).max()))
            data_checks["target_equals_s1"].append(
                float(np.abs(s1 - load_wav(str(tt_audio / "s1" / name))).max()))

        s1_npz = str(rebuild / "mouths" / f"{info['s1_id']}.npz")
        s2_npz = str(rebuild / "mouths" / f"{info['s2_id']}.npz")
        gray_s1 = mouth_from_pt(mouth_path)                 # cached per utterance
        gray_s2 = load_gray_mouth_npz(s2_npz)               # cached per utterance
        if count < 20:
            data_checks["mouth_equals_s1_npz"].append(
                float(np.abs(gray_s1.astype(int) - load_gray_mouth_npz(s1_npz).astype(int)).max()))
        data_checks["len"].append(len(mix))
        data_checks["sr"].append(SR)
        data_checks["channels"].append(1)

        # reference target scalings (metric-only variants)
        rms_scale = math.sqrt(float((mix ** 2).mean()) / (float((s1 ** 2).mean()) + 1e-12))
        energy_scale = math.sqrt(float(mix @ mix) / (float(s1 @ s1) + 1e-12))
        records.append({
            "mix": mix, "s1": s1, "s2": s2,
            "gray_s1": gray_s1, "gray_s2": gray_s2,
            "s1_rms_scaled": s1 * rms_scale,
            "s1_energy_scaled": s1 * energy_scale,
            "sir": info["sir"],
        })
        if count % 50 == 0 or count == len(indices) - 1:
            print(f"[gap] preload {count + 1}/{len(indices)} | elapsed {time.time() - t_start:.1f}s", flush=True)
    print(f"[gap] preloaded {len(records)} utterances in {time.time() - t_start:.1f}s")

    BASE_MOUTH_SPEC = {"source": "s1", "crop": 88, "shift": 0, "normalize": True, "three_ch": True}

    def _build_variant_batch(spec):
        """Return (audios, mouth_tensors, eval_slices) for all records."""
        audios, mouths, eval_slices = [], [], []
        if spec["kind"] == "mouth":
            for r in records:
                audios.append(r["mix"])
                mouths.append(build_mouth(spec, r["gray_s1"], r["gray_s2"]))
                eval_slices.append(slice(None))
        else:
            for r in records:
                audio_in, eval_slice, frame_slice = build_audio(spec, r["mix"])
                m = build_mouth(BASE_MOUTH_SPEC, r["gray_s1"], r["gray_s2"])
                if frame_slice != slice(None):
                    m = m[:, :, frame_slice, :, :]
                audios.append(audio_in)
                mouths.append(m)
                eval_slices.append(eval_slice)
        return audios, mouths, eval_slices

    def _run_batched(model_name, audios, mouths, batch_size):
        audio_model, video_model, _meta = models_loaded[model_name]
        bs = max(1, int(batch_size))
        ests = []
        for start in range(0, len(records), bs):
            a = audios[start:start + bs]
            m = torch.cat(mouths[start:start + bs], dim=0).to(device)
            x = torch.stack([torch.from_numpy(np.ascontiguousarray(v.astype(np.float32))) for v in a]).to(device)
            with torch.no_grad():
                emb = video_model(m)
                out = audio_model(x, emb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 3:      # [B, n_src, T]
                out = out[:, 0, :]
            ests.extend(out[i].float().cpu().numpy().astype(np.float64) for i in range(out.shape[0]))
        return ests

    # ---- batched inference per variant + metric computation -------------------
    for model_name in models:
        for spec in specs:
            vname = spec["name"]
            audios, mouths, eval_slices = _build_variant_batch(spec)
            bs = min(args.batch_size, 2) if spec.get("mode") == "pad3s" else args.batch_size
            t_v = time.time()
            ests = _run_batched(model_name, audios, mouths, bs)
            r = results[model_name][vname]
            for rec, est_full, eval_slice in zip(records, ests, eval_slices):
                est = est_full[eval_slice]
                ref = rec["s1"][eval_slice]
                mix_eval = rec["mix"][eval_slice]
                s2_eval = rec["s2"][eval_slice]
                r["si_snr"].append(si_sdr_pair(est, ref, True, True))
                r["si_snr_i"].append(si_sdr_improvement(est, mix_eval, ref, True, True))
                r["si_snr_zm0"].append(si_sdr_pair(est, ref, False, True))
                r["si_snr_i_zm0"].append(si_sdr_improvement(est, mix_eval, ref, False, True))
                r["snr_i_plain"].append(si_sdr_improvement(est, mix_eval, ref, True, False))
                r["sdr512"].append(sdr_value(est, ref, 512, device))
                r["sdr512_i"].append(sdr_value(est, ref, 512, device) - sdr_value(mix_eval, ref, 512, device))
                for L, key in ((256, "sdr256_i"), (128, "sdr128_i"), (64, "sdr64_i")):
                    r[key].append(sdr_value(est, ref, L, device) - sdr_value(mix_eval, ref, L, device))
                r["si_snr_i_seg2"].append(si_sdr_improvement_segments(est, mix_eval, ref, 2))
                r["si_snr_i_seg4"].append(si_sdr_improvement_segments(est, mix_eval, ref, 4))
                r["si_snr_i_tgt_s2"].append(si_sdr_improvement(est, mix_eval, s2_eval))
                r["si_snr_est_s2"].append(si_sdr_pair(est, s2_eval))
                r["si_snr_i_best_target"].append(max(r["si_snr_i"][-1], r["si_snr_i_tgt_s2"][-1]))
                r["si_snr_i_tgt_rms_scale"].append(
                    si_sdr_improvement(est, mix_eval, rec["s1_rms_scaled"][eval_slice]))
                r["si_snr_i_tgt_energy_scale"].append(
                    si_sdr_improvement(est, mix_eval, rec["s1_energy_scaled"][eval_slice]))
                r["mix_si_snr"].append(si_sdr_pair(mix_eval, ref))
                r["mix_sdr512"].append(sdr_value(mix_eval, ref, 512, device))
            print(f"[gap] {model_name}:{vname} done in {time.time() - t_v:.1f}s "
                  f"({len(records)} utts, batch={bs})", flush=True)

    # ---- anchor check: first-20 subset reproduces Phase-1 metrics? -------------
    if args.subset == "first" and len(indices) == 20 and "small" in results:
        anchors_first20 = {
            "si_snr_i": float(np.mean(results["small"][BASELINE]["si_snr_i"])),
            "sdr512_i": float(np.mean(results["small"][BASELINE]["sdr512_i"])),
            "phase1_reference": {"si_snr_i": 7.911, "sdr512_i": 8.547},
        }

    # ---- aggregate + report ---------------------------------------------------
    summary = {"meta": {
        "script": "tools/repro/phase2/gap_analysis.py",
        "eval_root": str(eval_dir), "rebuild_root": str(rebuild),
        "subset": args.subset, "seed": args.seed, "n_utterances": len(indices),
        "models": models, "variants": variant_names, "device": str(device),
        "weights_dir": args.weights_dir,
    }, "data_checks": {}, "models": {}, "anchors_first20": anchors_first20}

    for key, values in data_checks.items():
        arr = np.asarray(values)
        finite = np.isfinite(arr)
        summary["data_checks"][key] = {
            "mean": float(arr[finite].mean()) if finite.any() else float("nan"),
            "std": float(arr[finite].std()) if finite.any() else float("nan"),
            "min": float(arr[finite].min()) if finite.any() else float("nan"),
            "max": float(arr[finite].max()) if finite.any() else float("nan"),
        }

    for model_name in models:
        summary["models"][model_name] = {"loader_meta": models_loaded[model_name][2], "variants": {}}
        for vname in variant_names:
            summary["models"][model_name]["variants"][vname] = {}
            for metric in METRICS:
                arr = np.asarray(results[model_name][vname][metric], dtype=np.float64)
                finite = np.isfinite(arr)
                summary["models"][model_name]["variants"][vname][metric] = {
                    "mean": float(arr[finite].mean()) if finite.any() else float("nan"),
                    "std": float(arr[finite].std()) if finite.any() else float("nan"),
                    "n": int(finite.sum()),
                }

    # ---- print comparison tables ---------------------------------------------
    print("\n=== data self-checks (n={}) ===".format(len(indices)))
    for key, stats in summary["data_checks"].items():
        print(f"  {key:26s} mean={stats['mean']:.6g} min={stats['min']:.6g} max={stats['max']:.6g}")

    print("\n=== mixture baseline consistency ===")
    sir = np.asarray(data_checks["sir_filename"])
    mix_si = np.asarray(data_checks["mix_si_snr"])
    mix_sdr = np.asarray(data_checks["mix_sdr512"])
    print(f"  filename SIR mean={sir.mean():.3f} dB | measured SI-SNR(mix,s1) mean={mix_si.mean():.3f} "
          f"| corr={np.corrcoef(sir, mix_si)[0, 1]:.3f}")
    print(f"  measured SDR512(mix,s1) mean={mix_sdr.mean():.3f} dB")

    for model_name in models:
        print(f"\n=== {model_name} model: variant comparison (n={len(indices)}) ===")
        header = f"{'variant':28s} {'SI-SNR':>8s} {'SI-SNRi':>8s} {'dSIi':>7s} {'SDR512':>8s} {'SDR512i':>8s} {'dSDRi':>7s} {'seg2_i':>7s} {'tgtS2_i':>8s}"
        print(header)
        base = summary["models"][model_name]["variants"][BASELINE]
        for vname in variant_names:
            v = summary["models"][model_name]["variants"][vname]
            d_si = v["si_snr_i"]["mean"] - base["si_snr_i"]["mean"]
            d_sdr = v["sdr512_i"]["mean"] - base["sdr512_i"]["mean"]
            print(f"{vname:28s} {v['si_snr']['mean']:8.2f} {v['si_snr_i']['mean']:8.2f} {d_si:+7.2f} "
                  f"{v['sdr512']['mean']:8.2f} {v['sdr512_i']['mean']:8.2f} {d_sdr:+7.2f} "
                  f"{v['si_snr_i_seg2']['mean']:7.2f} {v['si_snr_i_tgt_s2']['mean']:8.2f}")
        print("  metric variants on baseline predictions (mean dB):")
        for metric in ("si_snr_i", "si_snr_i_zm0", "snr_i_plain", "sdr512_i", "sdr256_i",
                       "sdr128_i", "sdr64_i", "si_snr_i_seg2", "si_snr_i_seg4",
                       "si_snr_i_tgt_rms_scale", "si_snr_i_tgt_energy_scale",
                       "si_snr_i_tgt_s2", "si_snr_i_best_target"):
            print(f"    {metric:30s} {base[metric]['mean']:8.3f} ± {base[metric]['std']:.3f}")

    if anchors_first20:
        print(f"\n=== anchor (first-20 small baseline): {anchors_first20} ===")

    out_json = args.out_json
    if not out_json:
        tag = ("_" + args.tag) if args.tag else ""
        out_json = str(REPO / "docs" / "verification" / "phase2" /
                       f"P2A-results{tag}.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=2))
    print(f"\n[gap] results JSON: {out_json}")


if __name__ == "__main__":
    main()
