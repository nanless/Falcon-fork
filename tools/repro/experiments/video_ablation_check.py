#!/usr/bin/env python
"""Re-runnable "how much does vision actually contribute?" ablation for Falcon.

Why this file exists
--------------------
The only mouth-ablation evidence in this repo is ``docs/verification/phase2/P2F-results.json``
(stage 3 of ``tools/repro/phase2/protocol_check.py``): **N=20** clips, with the author's own note
"re-run ``--stage variants --device cuda:1`` to reproduce". N=20 is too small to quote, and — as
this script shows — the *effective* number of independent visual samples is smaller still,
because Falcon stores one mouth stream per **target utterance id**, not per mixture
(``/root/data/falcon_lrs2_2mix`` has 14,773 mouth files for 20,000 ``tr`` mixtures; the first 40
``tt`` mixtures share only **4** mouth files). Anything that averages over clips as if they were
independent visual samples therefore overstates precision.

What it measures (paired: the audio is identical, only the mouth stream changes)
-------------------------------------------------------------------------------
1. ``real``          – the target speaker's own lip crops (the model's normal input)
2. ``zero_feat``     – post-preprocessing zeros = a flat mid-grey face (raw grey ~107.4).
                       This is what "switch the visual branch off at the feature level" means.
3. ``black_pix``     – raw pixel 0 with the shape kept, i.e. z-score (0/255-0.421)/0.165 = -2.552.
                       Kept separate from ``zero_feat`` because the two are different tensors; the
                       gap between them is an input-distribution effect, not visual information.
4. ``wrong_speaker`` – another speaker's lips. Three selectors (``--wrong-mode``):
                       * ``other`` (default) – a different target speaker drawn deterministically
                         from the split's mouth pool. This is the valid control.
                       * ``next``            – literally the next clip's mouth. **Degenerate in this
                         dataset** (consecutive mixtures share one mouth file); kept only to show
                         why, and it refuses to report a "wrong" result when the stream is
                         identical to the real one.
                       * ``s2``              – the actual interferer's mouth from ``lrs2_rebuild``
                         (the strictest, and what ``protocol_check.py`` used).

Metric: SI-SDRi in dB (SI-SDR(est,ref) − SI-SDR(mix,ref)), formula identical to
``tools/repro/phase2/protocol_check.py:70-84``. Paired deltas are computed per clip, and
additionally per **distinct mouth file** (cluster unit) so the CI does not pretend the clips are
independent visual samples.

Cost / budget
-------------
Written for ``CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4`` (pure CPU). Measured on the 2x A800 host
with 4 threads: video frontend 135 ms/clip, full audio path 511 ms/clip. Three savings keep
``--n 40`` inside the 120 s budget without changing what is computed:
  * audio encoder + bottleneck (40 ms/clip) are computed once per clip and reused by all four
    conditions — they cannot see the mouth stream;
  * the two constant-mouth conditions need **one** video forward each for the whole run;
  * each distinct wrong-speaker mouth file is loaded and encoded once.
The fast path is verified numerically against ``model.forward`` on the first clip
(``--verify-forward``, default on: max|Δ| printed and asserted below 1e-5).

Usage
-----
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=$PWD \\
      /root/miniforge3/envs/falcon/bin/python tools/repro/experiments/video_ablation_check.py \\
        --stage variants --n 40                       # literal "first 40", for comparability
        --stage variants --n 40 --select spread       # 40 clips spread over the split (recommended)
        --stage variants --n 40 --wrong-mode s2       # strictest wrong mouth
        --help

Outputs (``--out``, default ``/tmp/falcon_video_ablation``)
  * ``video_ablation_per_sample.csv`` – one row per clip, SI-SNR / SI-SNRi for every condition
  * ``video_ablation_summary.json``   – aggregates, cluster stats, provenance, caveats
  * ``video_ablation_summary.csv``    – the printed table, machine readable
  * ``video_ablation_summary.txt``    – the printed report (table + conclusion + limits)

Read-only apart from this file: nothing else in the checkout is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

# tools/repro/experiments/video_ablation_check.py -> parents[3] == repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/root/data/falcon_lrs2_2mix")
DEFAULT_REBUILD = Path("/root/data/lrs2_2mix/lrs2_rebuild")  # raw npz mouths + per-source audio
DEFAULT_CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
DEFAULT_CONFIG = REPO / "configs/Falcon.yml"
DEFAULT_OUT = Path("/tmp/falcon_video_ablation")

SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 2.0   # dataset clips are exactly 32000 samples (data/conversion_stats.json)
CONDITIONS = ("real", "zero_feat", "black_pix", "wrong_speaker")
LABELS = {
    "real": "target speaker's own lips",
    "zero_feat": "zeros after preprocessing (= flat mid-grey face)",
    "black_pix": "raw pixel 0, shape kept (= z-score -2.552)",
    "wrong_speaker": "another speaker's lips",
}
ZERO_MEAN = True        # same convention as protocol_check.my_si_sdr
EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def die(msg: str, hints: list[str] | None = None, code: int = 2):
    """Readable failure instead of a stack trace."""
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    for h in hints or []:
        print(f"       - {h}", file=sys.stderr)
    raise SystemExit(code)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def si_sdr(ref, est, zero_mean: bool = ZERO_MEAN, eps: float = EPS) -> float:
    """SI-SDR in dB (positive = better). Same formula as protocol_check.py:70-80."""
    import numpy as np

    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    e = np.asarray(est, dtype=np.float64).reshape(-1)
    if r.size != e.size:
        raise ValueError(f"length mismatch: ref {r.size} vs est {e.size}")
    if zero_mean:
        r = r - r.mean()
        e = e - e.mean()
    alpha = float(np.dot(e, r) / (np.dot(r, r) + eps))
    target = alpha * r
    noise = e - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def si_sdr_i(ref, est, mix, zero_mean: bool = ZERO_MEAN) -> float:
    return si_sdr(ref, est, zero_mean) - si_sdr(ref, mix, zero_mean)


def speaker_of(path) -> str:
    """``<speaker>_<utterance>[.ext]`` -> ``<speaker>``."""
    stem = Path(str(path)).name
    stem = stem.split(".")[0]
    return stem.split("_")[0] if "_" in stem else stem


def s1_s2_ids(mix_name: str) -> tuple[str, str]:
    """``<s1_id>_<snr1>_<s2_id>_<snr2>.wav`` -> (s1_id, s2_id). Mirrors protocol_check.py:124-128."""
    stem = Path(str(mix_name)).name
    for suf in (".wav", ".flac", ".mp3"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"unparsable mixture name: {mix_name!r}")
    return f"{parts[0]}_{parts[1]}", f"{parts[3]}_{parts[4]}"


def select_indices(total: int, start: int, n: int, mode: str) -> list[int]:
    """Clip indices to evaluate.

    ``first``  – the literal first N (comparable with the earlier N=20 P2F run).
    ``spread`` – stride across the whole split, which is the only way to get ~N distinct
                 target utterances (``first`` clusters hard: mouth files are deduplicated
                 per target utterance, so consecutive mixtures repeat one mouth).
    """
    if mode == "first":
        return list(range(start, min(total, start + n)))
    span = total - start
    if span <= 0:
        return []
    stride = max(1, span // max(1, n))
    idx = [start + k * stride for k in range(n)]
    return [i for i in idx if i < total]


# ──────────────────────────────────────────────────────────────────────────────
# args
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_ablation_check.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Paired lip-stream ablation for Falcon (real / zero-feature / black-pixel / "
                    "wrong-speaker). CPU-runnable, bounded runtime, writes per-sample CSV+JSON.",
        epilog="""examples:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=$PWD \\
    python tools/repro/experiments/video_ablation_check.py --stage variants --n 40
  ... --n 40 --select spread --out /tmp/falcon_video_ablation/spread40
  ... --n 40 --wrong-mode s2
""",
    )
    p.add_argument("--stage", default="variants", choices=["variants"],
                   help="only 'variants' (the four mouth conditions) for now; default variants")
    p.add_argument("--n", type=int, default=40,
                   help="number of clips; default 40 (~95 s on 4 CPU threads)")
    p.add_argument("--split", default="tt", choices=["tr", "cv", "tt"],
                   help="split under --data-root; default tt (3000 clips, 267 mouth files)")
    p.add_argument("--start", type=int, default=0, help="first clip index; for sharding, default 0")
    p.add_argument("--select", default="first", choices=["first", "spread"],
                   help="'first' = literal first N (default, matches the earlier P2F sample); "
                        "'spread' = strided over the whole split, giving ~N distinct target "
                        "utterances and therefore a trustworthy CI")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help=f"falcon-format data root; default {DEFAULT_DATA_ROOT}")
    p.add_argument("--rebuild-root", type=Path, default=DEFAULT_REBUILD,
                   help=f"lrs2_rebuild root, only for --wrong-mode s2; default {DEFAULT_REBUILD}")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help=f"checkpoint; default {DEFAULT_CKPT}")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"training YAML used to rebuild the modules; default {DEFAULT_CONFIG}")
    p.add_argument("--device", default="cpu", help="torch device string; default cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output directory, created if missing; default {DEFAULT_OUT}")
    p.add_argument("--wrong-mode", default="other", choices=["other", "next", "s2"],
                   help="'other' (default) = another target speaker from the split's mouth pool; "
                        "'next' = literally the next clip (degenerate here, kept for evidence); "
                        "'s2' = the actual interferer's mouth (strictest)")
    p.add_argument("--batch", type=int, default=1,
                   help="clips per forward; 1 is fastest on cpu (measured), raise on GPU")
    p.add_argument("--verify-forward", dest="verify_forward", action="store_true", default=True,
                   help="check the cached-audio fast path against model.forward (default on)")
    p.add_argument("--no-verify-forward", dest="verify_forward", action="store_false",
                   help="skip that equivalence check")
    p.add_argument("--seed", type=int, default=0, help="reserved; selection is deterministic")
    return p


def check_paths(args) -> None:
    """Fail with a readable list of every missing prerequisite."""
    missing: list[str] = []
    hints: list[str] = []
    split_dir = args.data_root / args.split
    for name in ("mix.json", "target.json"):
        if not (split_dir / name).is_file():
            missing.append(str(split_dir / name))
    if not (split_dir / "mouths").is_dir():
        missing.append(str(split_dir / "mouths") + " (mouth .pt files)")
    if not Path(args.ckpt).is_file():
        missing.append(str(args.ckpt))
        hints.append("checkpoints live in /root/.cache/falcon_weights/ (symlinked to group-shared)")
    if not Path(args.config).is_file():
        missing.append(str(args.config))
        hints.append("pass --config <repo>/configs/Falcon.yml")
    if args.wrong_mode == "s2" and not (args.rebuild_root / "mouths").is_dir():
        missing.append(str(args.rebuild_root / "mouths") + " (needed by --wrong-mode s2)")
        hints.append("use --wrong-mode other (default) to take mouths from the falcon split")
    if missing:
        die("missing prerequisite(s):\n         " + "\n         ".join(missing), hints)
    if args.batch < 1:
        die(f"--batch must be >= 1, got {args.batch}")


# ──────────────────────────────────────────────────────────────────────────────
# model / data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_models(ckpt: Path, config: Path, device):
    """Rebuild audio+video modules from the training YAML and load both from the checkpoint.

    Uses the evaluation harness helper (``audio_visual_test._load_models_from_ckpt``) so the
    weights, the strict loading and the frozen state match production inference.
    """
    import torch
    import yaml

    sys.path.insert(0, str(REPO))
    try:
        import audio_visual_test as avt
    except Exception as exc:  # noqa: BLE001
        die(f"cannot import audio_visual_test from {REPO}: {exc}",
            ["run with PYTHONPATH=$PWD from the repo root"])
    try:
        cfg = yaml.safe_load(Path(config).read_text())
        train_conf = avt._get_train_conf(cfg)
        model, video_model = avt._load_models_from_ckpt(str(ckpt), train_conf, device)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        die(f"failed to rebuild/load models: {type(exc).__name__}: {exc}",
            [f"config: {config}", f"checkpoint: {ckpt}"])
    model.eval()
    video_model.eval()
    for p in video_model.parameters():
        p.requires_grad = False
    log(f"  audio_model : {sum(p.numel() for p in model.parameters())/1e6:.4f} M params "
        f"({sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable)")
    log(f"  video_model : {sum(p.numel() for p in video_model.parameters())/1e6:.4f} M params "
        f"({sum(p.numel() for p in video_model.parameters() if p.requires_grad)} trainable, frozen)")
    return model, video_model, avt


def load_dataset(data_root: Path, split: str):
    sys.path.insert(0, str(REPO))
    from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset

    try:
        return AVSpeechEvalDataset(json_dir=str(Path(data_root) / split), sample_rate=SAMPLE_RATE,
                                   normalize_audio=False)
    except Exception as exc:  # noqa: BLE001
        die(f"failed to build AVSpeechEvalDataset for {Path(data_root)/split}: "
            f"{type(exc).__name__}: {exc}",
            ["expected <split>/{mix,target}.json and <split>/{mix,target,mouths}/"])


def build_mouth_loader(torch, prep, gray):
    """Return ``path -> (F,88,88) float32`` matching ``AVSpeechEvalDataset`` exactly.

    ``avspeech_dataset.py:194-196`` does
        Grayscale(torch.load(pt).permute(0,3,1,2)).squeeze(1).numpy() -> val pipeline
    This function is that chain, with an LRU cache because mouth files are shared by many clips.
    """
    import numpy as np

    cache: dict[str, "np.ndarray"] = {}

    def load(path: str | Path):
        key = str(path)
        if key in cache:
            return cache[key]
        t = torch.load(key, weights_only=True)
        if t.dim() != 4:
            raise ValueError(f"{key}: expected (F,H,W,C) mouth tensor, got {tuple(t.shape)}")
        arr = prep(gray(t.permute(0, 3, 1, 2).float()).squeeze(1).numpy()).astype(np.float32)
        cache[key] = arr
        return arr

    return load


def prep_npz_mouth(npz_path: Path, torch, prep, gray):
    """uint8 (50,96,96) npz -> validation-pipeline output (50,88,88) float32."""
    import numpy as np

    with np.load(npz_path) as z:
        arr = z["data"] if "data" in z else z[list(z.keys())[0]]
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
    return prep(gray(t).squeeze(1).numpy()).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# forward paths
# ──────────────────────────────────────────────────────────────────────────────
def video_embed(video_model, mouth_batch, device):
    import torch

    with torch.inference_mode():
        return video_model(mouth_batch.to(device))


def audio_cache(model, mix, device):
    """The part of the audio path that cannot see the mouth stream: encoder + bottleneck."""
    import torch

    with torch.inference_mode():
        enc = model.encoder(mix)
        ab = model.audio_bottleneck(enc)
    return enc, ab


def forward_from_cache(model, enc, ab, vemb, mix_shape):
    """separator -> mask_generator -> decoder, in ``Falcon.forward``'s exact order.

    ``look2hear/models/Falcon.py:1480-1499`` is
        enc = encoder(mix); ab = audio_bottleneck(enc); v = video_bottleneck(mouth_emb)
        rf = separator(ab, v); me = mask_generator(rf, enc); out = decoder(me, mix.shape)
    so caching ``enc``/``ab`` across conditions is exact (verified, see --verify-forward).
    """
    import torch

    with torch.inference_mode():
        rf = model.separator(ab, vemb)
        me = model.mask_generator(rf, enc)
        out = model.decoder(me, mix_shape)
    if isinstance(out, tuple):
        out = out[0]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# main experiment
# ──────────────────────────────────────────────────────────────────────────────
def stage_variants(args) -> dict:
    import numpy as np
    import torch
    from torchvision import transforms

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device(args.device)
    split_dir = Path(args.data_root) / args.split

    banner(f"STAGE variants — paired lip ablation (split={args.split}, N={args.n}, "
           f"select={args.select}, device={device})")
    log(f"  repo      : {REPO}")
    log(f"  data root : {split_dir}")
    log(f"  checkpoint: {args.ckpt}")
    log(f"  wrong-mode: {args.wrong_mode}")

    model, video_model, avt = load_models(Path(args.ckpt), Path(args.config), device)
    ds = load_dataset(Path(args.data_root), args.split)

    from look2hear.datas.transform import get_preprocessing_pipelines

    prep = get_preprocessing_pipelines()["val"]
    gray = transforms.Grayscale(num_output_channels=1)
    load_mouth = build_mouth_loader(torch, prep, gray)

    # target.json is index-aligned with mix.json for AVSpeechEvalDataset (one source list)
    mix_index = json.loads((split_dir / "mix.json").read_text())
    tgt_index = json.loads((split_dir / "target.json").read_text())
    if len(mix_index) != len(tgt_index):
        die(f"{split_dir}: mix.json has {len(mix_index)} entries but target.json has "
            f"{len(tgt_index)}; the index assumed here is not valid")

    n = int(args.n)
    if n < 4:
        die(f"--n must be >= 4 for a paired comparison, got {n}")
    indices = select_indices(len(ds), max(0, int(args.start)), n, args.select)
    if len(indices) < 4:
        die(f"only {len(indices)} clip(s) selected; lower --n or --start")

    # ---- 1. materialise clips through the official eval preprocessing --------------------
    t0 = time.perf_counter()
    clips, skipped = [], []
    for i in indices:
        mix_t, ref_t, mouth_np, key, _ = ds[i]
        expect = Path(mix_index[i][0]).name
        if Path(key).name != expect:
            die(f"index misalignment at {i}: dataset returned {key!r}, mix.json has {expect!r}",
                ["the split's json/wav pairing changed; this script assumes 1:1 order"])
        mouth_path = str(tgt_index[i][1])
        mouth_np = np.asarray(mouth_np, dtype=np.float32)
        if mouth_np.ndim != 3:
            skipped.append((i, key, f"bad mouth shape {mouth_np.shape}"))
            continue
        if mix_t.numel() != int(SEGMENT_SECONDS * SAMPLE_RATE):
            skipped.append((i, key, f"clip length {mix_t.numel()} != "
                                    f"{int(SEGMENT_SECONDS * SAMPLE_RATE)}"))
            continue
        clips.append({
            "index": i, "key": key, "mix": mix_t,
            "ref": ref_t.detach().cpu().numpy(),
            "mouth": mouth_np,
            "mouth_file": mouth_path,
            "spk": speaker_of(mouth_path),
        })
    if not clips:
        die("no usable clips after validation",
            ["check that mouths/*.pt exist and clips are 32000 samples",
             f"first skipped: {skipped[:3]}"])

    distinct_mouths = len({c["mouth_file"] for c in clips})
    distinct_spks = len({c["spk"] for c in clips})
    log(f"  clips      : {len(clips)} usable, {len(skipped)} skipped "
        f"({time.perf_counter()-t0:.1f} s)")
    log(f"  effective N: {distinct_mouths} distinct mouth file(s), {distinct_spks} distinct "
        f"target speaker(s)")
    if distinct_mouths < len(clips) * 0.5:
        log(f"  [warn] {len(clips)} clips but only {distinct_mouths} independent visual samples — "
            f"per-clip CIs are optimistic; use the cluster rows below, or --select spread")

    # ---- 2. the four mouth streams -------------------------------------------------------
    ref_shape = clips[0]["mouth"].shape
    const_streams = {
        # post-preprocessing zeros = "visual branch off" at the model boundary
        "zero_feat": np.zeros_like(clips[0]["mouth"]),
        # raw pixel 0 through the real pipeline: (0/255 - 0.421)/0.165 = -2.5515
        "black_pix": prep(np.zeros((ref_shape[0], 96, 96), dtype=np.uint8)),
    }
    for k, v in const_streams.items():
        if v.shape != clips[0]["mouth"].shape:
            die(f"constant stream {k} has shape {v.shape}, expected {clips[0]['mouth'].shape}")
    log(f"  const mouths: zero_feat all {const_streams['zero_feat'].min():.3f}; "
        f"black_pix all {const_streams['black_pix'].min():.3f} (raw 0 -> z-score)")

    wrong = build_wrong_streams(args, clips, split_dir, torch, prep, gray, load_mouth,
                               prep_npz_mouth, np)
    n_same = sum(1 for w in wrong if w["same_spk_as_target"])
    log(f"  wrong mouth: mode={args.wrong_mode}, {len(set(w['file'] for w in wrong))} distinct "
        f"file(s), {len(set(w['spk'] for w in wrong))} distinct speaker(s), "
        f"{n_same} share the target speaker")
    for pos, (c, w) in enumerate(zip(clips, wrong)):
        if np.array_equal(w["stream"], c["mouth"]):
            die(f"clip {c['index']}: the 'wrong' mouth is identical to the real mouth "
                f"({Path(w['file']).name})",
                ["this dataset deduplicates mouths per target utterance, so neighbouring "
                 "mixtures often share one mouth stream",
                 "use --wrong-mode other (default) or --wrong-mode s2"])

    # ---- 3. cached audio features + constant video embeddings ----------------------------
    t0 = time.perf_counter()
    mixes = torch.stack([c["mix"] for c in clips])
    cached = []
    step = max(1, args.batch)
    for s in range(0, len(clips), step):
        enc, ab = audio_cache(model, mixes[s:s + step].to(device), device)
        for j in range(enc.shape[0]):
            cached.append((enc[j:j + 1].clone(), ab[j:j + 1].clone()))
    log(f"  audio cache: {len(cached)} clips in {time.perf_counter()-t0:.1f} s "
        f"(encoder+bottleneck, shared by all conditions)")

    const_emb = {name: video_embed(video_model, torch.from_numpy(st)[None, None].float(), device)
                 for name, st in const_streams.items()}
    log(f"  const video embeddings: "
        + ", ".join(f"{k}{tuple(v.shape)}" for k, v in const_emb.items()))

    # ---- 4. the paired run ----------------------------------------------------------------
    rows = []
    verify_err = None
    t_start = time.perf_counter()
    for pos, c in enumerate(clips):
        enc, ab = cached[pos]
        mix_shape = (1, c["mix"].numel())
        row = {"index": c["index"], "key": c["key"], "spk_id": c["spk"],
               "mouth_file": c["mouth_file"],
               "wrong_mouth_file": wrong[pos]["file"],
               "wrong_spk_id": wrong[pos]["spk"],
               "wrong_same_spk": wrong[pos]["same_spk_as_target"],
               "n_frames": int(c["mouth"].shape[0]),
               "mix_si_sdr": si_sdr(c["ref"], c["mix"].numpy())}
        for cond in CONDITIONS:
            if cond == "real":
                stream = c["mouth"]
            elif cond in const_streams:
                stream = const_streams[cond]
            else:
                stream = wrong[pos]["stream"]
            vemb = const_emb[cond] if cond in const_emb else video_embed(
                video_model, torch.from_numpy(stream)[None, None].float(), device)
            est = forward_from_cache(model, enc, ab, vemb, mix_shape)
            est_np = est[0, 0].detach().float().cpu().numpy()
            row[f"{cond}_si_sdr"] = si_sdr(c["ref"], est_np)
            row[f"{cond}_si_sdr_i"] = row[f"{cond}_si_sdr"] - row["mix_si_sdr"]
            if cond == "real" and pos == 0 and args.verify_forward:
                with torch.inference_mode():
                    ref_out = model(c["mix"].unsqueeze(0).to(device), vemb)
                if isinstance(ref_out, tuple):
                    ref_out = ref_out[0]
                verify_err = float((ref_out - est).abs().max())
        rows.append(row)
        done = pos + 1
        if done % 5 == 0 or done == len(clips):
            el = time.perf_counter() - t_start
            eta = el / done * (len(clips) - done)
            log(f"  [{done:3d}/{len(clips)}] {row['key'][:38]:38s} "
                f"real={row['real_si_sdr_i']:+6.2f} zero={row['zero_feat_si_sdr_i']:+7.2f} "
                f"black={row['black_pix_si_sdr_i']:+7.2f} wrong={row['wrong_speaker_si_sdr_i']:+7.2f}"
                f"  ({el:.0f}s, eta {eta:.0f}s)")

    if args.verify_forward and verify_err is not None:
        log(f"\n  fast-path equivalence (clip 1, real mouth): max|Δ| = {verify_err:.3e}")
        if verify_err > 1e-5:
            die(f"cached-audio fast path diverges from model.forward (max|Δ|={verify_err:.3e})",
                ["this would invalidate the ablation"])

    return summarize(args, rows, time.perf_counter() - t_start, verify_err,
                     n_same, skipped, distinct_mouths, distinct_spks)


def build_wrong_streams(args, clips, split_dir, torch, prep, gray, load_mouth, prep_npz,
                        np) -> list[dict]:
    """One wrong-speaker stream per clip, per ``--wrong-mode``."""
    wrong: list[dict] = []

    if args.wrong_mode == "s2":
        for c in clips:
            _, s2 = s1_s2_ids(c["key"])
            npz = Path(args.rebuild_root) / "mouths" / f"{s2}.npz"
            if not npz.is_file():
                die(f"interferer mouth not found for {c['key']}: {npz}",
                    ["use --wrong-mode other, or point --rebuild-root at lrs2_rebuild"])
            wrong.append({"stream": prep_npz(npz, torch, prep, gray), "file": str(npz),
                          "spk": speaker_of(npz), "same_spk_as_target": speaker_of(npz) == c["spk"]})
        return wrong

    if args.wrong_mode == "next":
        # Literal "next clip's mouth". Kept as evidence, but this dataset deduplicates mouths
        # per target utterance, so this often yields the *same* mouth; the caller aborts then.
        for pos, c in enumerate(clips):
            src = clips[(pos + 1) % len(clips)]
            wrong.append({"stream": src["mouth"].copy(), "file": src["mouth_file"],
                          "spk": src["spk"], "same_spk_as_target": src["spk"] == c["spk"]})
        return wrong

    # default: a different target speaker from the split's own mouth pool
    pool: dict[str, list[Path]] = {}
    for f in sorted((split_dir / "mouths").glob("*.pt")):
        pool.setdefault(speaker_of(f), []).append(f)
    speakers = sorted(pool)
    if len(speakers) < 2:
        die(f"split {split_dir.name} has only {len(speakers)} target speaker(s); "
            f"a wrong-speaker control is impossible",
            ["use --wrong-mode s2 (interferer mouth), which does not need a second target speaker"])
    for pos, c in enumerate(clips):
        cands = [s for s in speakers if s != c["spk"]]
        spk = cands[pos % len(cands)]
        files = pool[spk]
        f = files[pos % len(files)]
        wrong.append({"stream": load_mouth(f), "file": str(f), "spk": spk,
                      "same_spk_as_target": False})
    return wrong


def _agg(vals):
    """mean / median / std / sem for a list, tolerant of n==1."""
    import numpy as np

    v = np.asarray(vals, dtype=np.float64)
    n = int(v.size)
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    return {"n": n, "mean": float(v.mean()), "median": float(np.median(v)), "std": std,
            "sem": sem, "min": float(v.min()), "max": float(v.max()),
            "ci95": [float(v.mean() - 1.96 * sem), float(v.mean() + 1.96 * sem)]}


def summarize(args, rows, elapsed, verify_err, n_same, skipped,
              distinct_mouths, distinct_spks) -> dict:
    import numpy as np

    n = len(rows)
    real = np.array([r["real_si_sdr_i"] for r in rows], dtype=np.float64)
    stats = {}
    for cond in CONDITIONS:
        vals = np.array([r[f"{cond}_si_sdr_i"] for r in rows], dtype=np.float64)
        diff = vals - real
        a = _agg(diff)
        stats[cond] = {
            "n": n,
            "mean_si_sdr_i": float(vals.mean()),
            "median_si_sdr_i": float(np.median(vals)),
            "std_si_sdr_i": float(vals.std(ddof=1)) if n > 1 else 0.0,
            "min_si_sdr_i": float(vals.min()),
            "max_si_sdr_i": float(vals.max()),
            "mean_paired_delta_vs_real": a["mean"],
            "median_paired_delta_vs_real": a["median"],
            "sem_paired_delta": a["sem"],
            "ci95_paired_delta": a["ci95"],
            "n_better_than_real": int((diff > 0).sum()),
            "n_worse_than_real": int((diff < 0).sum()),
        }

    # cluster-level stats: unit = distinct mouth file (the real independent visual sample)
    cluster = {}
    by_mouth: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_mouth.setdefault(r["mouth_file"], []).append(i)
    for cond in CONDITIONS:
        if cond == "real":
            continue
        per_mouth = []
        for _, idxs in by_mouth.items():
            d = [rows[i][f"{cond}_si_sdr_i"] - rows[i]["real_si_sdr_i"] for i in idxs]
            per_mouth.append(float(np.mean(d)))
        a = _agg(per_mouth)
        cluster[cond] = {"unit": "distinct mouth file", "n_clusters": a["n"],
                         "mean_delta": a["mean"], "median_delta": a["median"],
                         "sem": a["sem"], "ci95": a["ci95"]}

    mix_vals = np.array([r["mix_si_sdr"] for r in rows], dtype=np.float64)

    # ── printed table ──────────────────────────────────────────────────────────
    header = (f"{'condition':<14}{'N':>4}{'mean Si-SNRi':>14}{'median':>10}"
              f"{'Δ vs real':>11}{'95% CI (clips)':>22}{'better/worse':>14}")
    lines = [header, "-" * len(header)]
    for cond in CONDITIONS:
        s = stats[cond]
        if cond == "real":
            delta, ci, bw = "—", "—", "—"
        else:
            delta = f"{s['mean_paired_delta_vs_real']:+.3f}"
            ci = f"[{s['ci95_paired_delta'][0]:+.2f},{s['ci95_paired_delta'][1]:+.2f}]"
            bw = f"{s['n_better_than_real']}/{s['n_worse_than_real']}"
        lines.append(f"{cond:<14}{s['n']:>4}{s['mean_si_sdr_i']:>14.3f}"
                     f"{s['median_si_sdr_i']:>10.3f}{delta:>11}{ci:>22}{bw:>14}")
    lines.append(f"{'mixture (input)':<14}{n:>4}{mix_vals.mean():>14.3f}"
                 f"{np.median(mix_vals):>10.3f}")
    table = "\n".join(lines)

    cluster_lines = [f"\n{'cluster check (unit = distinct mouth file)':<52}",
                     f"{'condition':<16}{'clusters':>9}{'mean Δ':>10}{'95% CI':>20}"]
    for cond, c in cluster.items():
        cluster_lines.append(f"{cond:<16}{c['n_clusters']:>9}{c['mean_delta']:>+10.3f}"
                             + f"[{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}]".rjust(20))
    cluster_table = "\n".join(cluster_lines)

    d = {c: stats[c]["mean_paired_delta_vs_real"] for c in CONDITIONS}
    dz = cluster.get("zero_feat", {}).get("mean_delta", float("nan"))
    verdict = [
        "",
        "CONCLUSION — the visual contribution, measured (not inferred)",
        f"  * N={n} clips (split={args.split}, select={args.select}), {distinct_mouths} distinct "
        f"mouth file(s) / {distinct_spks} target speaker(s), official falcon.pt, device={args.device}.",
        f"  * Removing the face (zeros after preprocessing = mid-grey) costs "
        f"{d['zero_feat']:+.2f} dB mean / {stats['zero_feat']['median_paired_delta_vs_real']:+.2f} dB "
        f"median; cluster mean over {cluster.get('zero_feat',{}).get('n_clusters','?')} mouth "
        f"file(s) = {dz:+.2f} dB.",
        f"  * A black frame (raw 0, same shape) costs {d['black_pix']:+.2f} dB; the "
        f"{abs(d['zero_feat'] - d['black_pix']):.2f} dB gap to the grey case is an "
        f"input-distribution effect (both carry zero lip content), not visual information.",
        f"  * Another speaker's lips costs {d['wrong_speaker']:+.2f} dB "
        f"({stats['wrong_speaker']['n_better_than_real']}/{n} clips got better with the wrong "
        f"face, mode={args.wrong_mode}).",
        "",
        "INFERENCE (argument, not measurement)",
        "  * real - zero_feat bounds how much the model uses lip motion at all; it does not",
        "    decompose into 'speaker identity' vs 'articulation'.",
        "  * if wrong_speaker is close to zero_feat the visual branch is decorative; clearly above",
        "    it means the encoder carries speaker-specific information that survives the wrong face.",
        "",
        "LIMITS — do not quote without these",
        f"  * one checkpoint, one split, one metric (SI-SDRi, 16 kHz, 2 s clips, window-level); "
        "not the paper's evaluation set.",
        (f"  * effective visual sample size is {distinct_mouths}, not {n}: mouths are deduplicated "
         "per target utterance, so per-clip CIs are optimistic (see cluster rows)."
         if distinct_mouths < n else
         f"  * all {n} clips use distinct mouth files, so clip-level and cluster-level CIs agree."),
        f"  * wrong-mouth mode '{args.wrong_mode}': "
        + {"other": "a different target speaker from the same split (valid control, but not the "
                   "specific interferer of each mixture).",
           "next": "literally the next clip; degenerate here, kept as evidence only.",
           "s2": "the actual interferer s2 of each mixture (strictest)."}[args.wrong_mode],
        "  * the FRCNN is frozen but still runs: 'vision off' is a neutral embedding, not zero cost.",
    ]
    if skipped:
        verdict.append(f"  * {len(skipped)} clip(s) skipped during validation (see summary json).")

    head = ["", f"elapsed: {elapsed:.1f} s for {n} clips x 4 conditions "
                f"({elapsed / max(n, 1) * 1000:.0f} ms/clip)"]
    report = "\n".join(head) + "\n" + table + cluster_table + "\n" + "\n".join(verdict)

    # ── outputs ────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = (["index", "key", "spk_id", "mouth_file", "wrong_mouth_file", "wrong_spk_id",
             "wrong_same_spk", "n_frames", "mix_si_sdr"]
            + [f"{c}_si_sdr" for c in CONDITIONS]
            + [f"{c}_si_sdr_i" for c in CONDITIONS])
    per_sample = out / "video_ablation_per_sample.csv"
    with per_sample.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "stage": "variants",
        "code": "tools/repro/experiments/video_ablation_check.py",
        "repo": str(REPO),
        "git_head": _git_head(),
        "repo_dirty_tracked_files": _repo_dirty(),
        "ckpt": str(args.ckpt),
        "ckpt_sha256_16": _sha16(Path(args.ckpt)),
        "config": str(args.config),
        "split": args.split,
        "data_root": str(args.data_root),
        "n_clips": n,
        "select": args.select,
        "start": args.start,
        "n_distinct_mouths": distinct_mouths,
        "n_distinct_target_speakers": distinct_spks,
        "wrong_mode": args.wrong_mode,
        "wrong_same_speaker_count": n_same,
        "device": args.device,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "torch": _torch_version(),
        "metric": {"name": "SI-SDRi", "sr": SAMPLE_RATE, "zero_mean": ZERO_MEAN,
                   "definition": "si_sdr(ref,est) - si_sdr(ref,mix), dB",
                   "reference_impl": "tools/repro/phase2/protocol_check.py:70-84"},
        "conditions": {c: LABELS[c] for c in CONDITIONS},
        "stats": stats,
        "mixture_baseline": {"mean_si_sdr": float(mix_vals.mean()),
                             "median_si_sdr": float(np.median(mix_vals))},
        "cluster_stats": cluster,
        "elapsed_s": round(elapsed, 2),
        "forward_verify_max_abs_err": verify_err,
        "skipped": [{"index": i, "key": k, "reason": why} for i, k, why in skipped],
        "paired_deltas_vs_real": d,
        "caveats": [
            f"single split ({args.split}), {args.select}-selection, one checkpoint",
            f"effective visual sample size {distinct_mouths} (mouths deduplicated per utterance)",
            "2 s clips, window-level inference; not the paper's full evaluation set",
            f"wrong-mouth mode={args.wrong_mode}",
            "frozen FRCNN: 'vision off' is a neutral embedding, not zero cost",
        ],
    }
    (out / "video_ablation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out / "video_ablation_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n", "mean_si_sdr_i", "median_si_sdr_i", "std",
                    "mean_paired_delta_vs_real", "median_paired_delta_vs_real",
                    "ci95_low", "ci95_high", "n_better", "n_worse", "cluster_n",
                    "cluster_mean_delta", "cluster_ci95_low", "cluster_ci95_high"])
        for cond in CONDITIONS:
            s = stats[cond]
            cl = cluster.get(cond, {})
            w.writerow([cond, s["n"], f"{s['mean_si_sdr_i']:.4f}", f"{s['median_si_sdr_i']:.4f}",
                        f"{s['std_si_sdr_i']:.4f}", f"{s['mean_paired_delta_vs_real']:+.4f}",
                        f"{s['median_paired_delta_vs_real']:+.4f}",
                        f"{s['ci95_paired_delta'][0]:+.4f}", f"{s['ci95_paired_delta'][1]:+.4f}",
                        s["n_better_than_real"], s["n_worse_than_real"],
                        cl.get("n_clusters", ""), f"{cl.get('mean_delta', 0):+.4f}" if cl else "",
                        f"{cl['ci95'][0]:+.4f}" if cl else "", f"{cl['ci95'][1]:+.4f}" if cl else ""])
    (out / "video_ablation_summary.txt").write_text(report + "\n")

    log("\n" + table)
    log(cluster_table)
    log("\n".join(verdict))
    log(f"\nwrote: {per_sample}\n       {out/'video_ablation_summary.json'}"
        f"\n       {out/'video_ablation_summary.csv'}\n       {out/'video_ablation_summary.txt'}")
    return summary


def _git_head() -> str:
    """Commit this run used; ``FALCON_GIT_HEAD`` wins so a HEAD snapshot stays attributable."""
    import subprocess

    env = os.environ.get("FALCON_GIT_HEAD", "").strip()
    if env:
        return env
    try:
        done = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _repo_dirty() -> list[str]:
    """Tracked files modified in the checkout this run imported from (concurrency guard)."""
    import subprocess

    try:
        done = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                               "--untracked-files=no"], capture_output=True, text=True, timeout=15)
        if done.returncode != 0:
            return ["<not a git checkout (HEAD snapshot)>"]
        return [line[3:] for line in done.stdout.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return ["unavailable"]


def _sha16(path: Path) -> str:
    import hashlib

    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unavailable"


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)      # --help works before any heavy import
    if args.n < 1:
        die(f"--n must be >= 1, got {args.n}")
    check_paths(args)
    if not (REPO / "look2hear").is_dir():
        die(f"{REPO} does not look like the Falcon checkout",
            ["the script must live at <repo>/tools/repro/experiments/"])
    t0 = time.perf_counter()
    summary = stage_variants(args)
    log(f"\ntotal wall time: {time.perf_counter()-t0:.1f} s")
    log("paired Δ (mean, dB): real=0.000  "
        + "  ".join(f"{c}={summary['paired_deltas_vs_real'][c]:+.3f}"
                    for c in CONDITIONS if c != "real"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
