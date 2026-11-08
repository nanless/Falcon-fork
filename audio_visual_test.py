"""Offline evaluation entry point for Falcon: SI-SDR / SDR / PESQ / STOI.

Phase1/T3 rewrite (docs/REPRODUCTION_PLAN.md §6.2 F7/F8):

* metrics are now sign-consistent and correctly defined (all positive,
  higher-is-better): SI-SDR, SI-SDRi, SDR, SDRi (fast_bss_eval,
  ``filter_length=512``), PESQ (``pesq(clean, estimate, fs)``) and STOI;
* ``--conf_dir`` accepts any training config containing
  ``audionet`` / ``videonet`` / ``datamodule`` (e.g. ``configs/Falcon.yml``),
  optionally combined with ``--data_dir`` pointing at a JSON index directory
  (``<data_dir>/tt/mix.json`` ... or a directory that directly contains
  ``mix.json``/``target.json``);
* ``${VAR}`` placeholders in the config are expanded with ``os.path.expandvars``;
  unresolved data paths raise an actionable error;
* ``--ckpt_path`` accepts a Lightning ``.ckpt``, a released ``falcon.pt``
  (``{"state_dict": ..., "architecture": ..., "epoch": ...}``), a bare
  state-dict file, or ``hf://<org>/<repo>/<file>``;
* ``_find_best_ckpt`` falls back to ``epoch=*.ckpt`` with **numeric** epoch
  sorting (``epoch=1000`` > ``epoch=742``);
* ``--limit N`` evaluates only the first N test utterances (smoke runs). A run with
  ``N < len(dataset)`` is a **subset run**: ``metrics_counts.json.coverage`` records
  ``limit`` / ``dataset_len`` / ``is_subset`` and forces ``verified=false`` with
  ``reason: "subset run: --limit N of M"``, because N-of-M numbers are not full-set
  coverage even when every planned utterance produced a row;
* ``--allow-partial`` downgrades only the *planned-vs-produced row-count* check to a
  warning (deliberately truncated debugging runs). It is a different failure mode
  from a subset run and it cannot make ``--limit`` output full-set coverage;
* an eval worker does **not** build dataloaders and does not require the
  train/valid splits; it instantiates :class:`AVSpeechEvalDataset` on the
  resolved test index directly (whole-utterance inference, batch size 1).

Three facts about this protocol that are easy to get wrong when quoting numbers
(they are the most common source of mis-attribution around this repo):

1. **Whole-clip single forward, no sliding window.** ``est = model(mix[None],
   video_model(mouth[None, None]))`` runs once per utterance and the decoder
   emits ``istft(..., length=T_in)``, so the estimate is sample-aligned with the
   input: no 2 s window, no hop, no extra padding, no chunk stitching. The
   2 s / 0.5 s hop / 80 ms guard protocol belongs to ``studio/server.py`` and to
   ``eval_realssa_real.py`` (long RealSSA-Real recordings), **not** to the
   LRS2-2Mix / RealSSA-Sim numbers this script produces.
2. **``n_src=1``: single-output target-speaker extraction, not 2-source
   separation.** The reference side is one ``target`` waveform (``target.json``),
   so PIT over sources degenerates to the identity permutation and there is no
   best-of-2 output-channel selection anywhere in the metric path.
3. **``si-snr_i = si-snr - si-snr_mix``, per utterance**: ``si-snr_mix`` is the
   SI-SDR of *that* unprocessed mixture against the same target (not a
   corpus-level average mixture). See ``look2hear/metrics/allwrapper.py``.

Usage::

    # Released weights + local JSON index directory
    python audio_visual_test.py \
        --conf_dir configs/Falcon.yml \
        --data_dir /path/to/index \
        --ckpt_path /root/.cache/falcon_weights/falcon.pt \
        --gpus 0 --limit 1

    # A training config snapshot with real paths
    python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml --gpus 0

    # Hugging Face weights
    python audio_visual_test.py --conf_dir configs/Falcon.yml --data_dir /path/to/index \
        --ckpt_path hf://JusperLee/Falcon/falcon.pt --gpus 0

Results are written to ``<cwd>/Experiments/checkpoint/<exp_name>/results/``
(per-rank shards merged into ``metrics_combined.csv``; per-utterance rows plus
one global ``avg``/``std`` row).
"""

import argparse
import copy
import json
import os
import glob
import re
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import yaml

import torch
import torch.multiprocessing as mp

import look2hear.models
import look2hear.videomodels
from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset
from look2hear.metrics import ALLMetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

warnings.filterwarnings("ignore")

#: Rank-0 coverage report, written next to the per-rank metric shards (W1-07).
COVERAGE_FILENAME = "coverage_rank0.json"

# ==========================================
# 1. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--conf_dir",
                    default="configs/Falcon.yml",
                    help="Training config YAML (needs audionet/videonet/datamodule; e.g. configs/Falcon.yml)")
parser.add_argument("--data_dir",
                    default="",
                    help="Optional JSON index directory (contains tt/mix.json+target.json, "
                         "or mix.json+target.json directly); overrides datamodule.data_config.test_dir")
parser.add_argument("--ckpt_path",
                    default="",
                    help="Checkpoint to evaluate: Lightning .ckpt, falcon.pt, bare state_dict, "
                         "or hf://<org>/<repo>/<file>; default: best in exp_dir")
parser.add_argument("--gpus",
                    type=str,
                    default="0",
                    help="Comma-separated GPU ids (e.g. '0' or '0,1'); '' or 'cpu' forces CPU")
parser.add_argument("--limit",
                    type=int,
                    default=0,
                    help="Evaluate only the first N test utterances (0 = all). Any N < the "
                         "dataset length makes this a SUBSET run: the numbers stay valid for "
                         "those N utterances only, and metrics_counts.json.coverage records "
                         "is_subset=true with verified=false")
parser.add_argument("--allow-partial",
                    action="store_true",
                    help="Downgrade only the planned-vs-produced row-count check to a warning "
                         "(deliberately truncated debugging runs). It is unrelated to --limit: "
                         "a subset run already reports verified=false and this flag cannot "
                         "turn it into full-set coverage")


# ==========================================
# 2. Config / checkpoint helpers
# ==========================================
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _expand_env_vars(value):
    """Recursively expand ``${VAR}`` / ``$VAR`` in all strings of a nested object."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value


def _list_unresolved_vars(value, found=None):
    """Collect remaining ``${VAR}``-looking placeholders (for clear error messages)."""
    found = [] if found is None else found
    if isinstance(value, str):
        for m in _ENV_PATTERN.finditer(value):
            name = m.group(1) or m.group(2)
            if name not in found:
                found.append(name)
    elif isinstance(value, list):
        for v in value:
            _list_unresolved_vars(v, found)
    elif isinstance(value, dict):
        for v in value.values():
            _list_unresolved_vars(v, found)
    return found


def _get_train_conf(config):
    """Return the training config dict, accepting ``{"train_conf": {...}}`` or the raw YAML dict."""
    if isinstance(config, dict) and isinstance(config.get("train_conf"), dict):
        return config["train_conf"]
    if isinstance(config, dict) and "audionet" in config and "videonet" in config:
        return config
    raise ValueError(
        "config must be a training YAML dict containing 'audionet', 'videonet' and 'datamodule'; "
        f"got keys: {sorted(config.keys()) if isinstance(config, dict) else type(config).__name__}"
    )


def _resolve_index_dir(path, split="tt"):
    """Resolve a JSON index directory from ``path`` or ``path/<split>``."""
    if not path:
        raise ValueError("empty index directory path")
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    candidates = [os.path.join(expanded, split), expanded]
    for candidate in candidates:
        mix_json = os.path.join(candidate, "mix.json")
        target_json = os.path.join(candidate, "target.json")
        if os.path.isfile(mix_json) and os.path.isfile(target_json):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"No JSON index found for split '{split}' under {path!r}: expected one of "
        f"{[os.path.join(c, 'mix.json') for c in candidates]} to exist. "
        "Pass --data_dir pointing at the directory that contains mix.json/target.json, "
        "or set the DATA_ROOT environment variable and edit datamodule.data_config."
    )


def _prepare_config(raw_conf, data_dir=""):
    """Expand env vars, resolve the test index directory, return ``(train_conf, test_dir)``."""
    train_conf = copy.deepcopy(raw_conf)
    if isinstance(train_conf.get("train_conf"), dict):  # tolerate wrapped configs
        train_conf = copy.deepcopy(train_conf["train_conf"])
    for key in ("audionet", "videonet", "datamodule"):
        if key not in train_conf:
            raise ValueError(f"config is missing required section '{key}'")
    if "data_config" not in train_conf["datamodule"]:
        raise ValueError("config is missing datamodule.data_config")

    train_conf = _expand_env_vars(train_conf)
    unresolved = _list_unresolved_vars(train_conf)
    if unresolved:
        print(f"[eval] warning: unresolved config placeholders remain: {unresolved} "
              f"(pass --data_dir or export these variables)")

    data_cfg = train_conf["datamodule"]["data_config"]
    if data_dir:
        test_dir = _resolve_index_dir(data_dir, split="tt")
    else:
        configured = data_cfg.get("test_dir", "")
        if not configured or configured in ("--", "-"):
            raise ValueError(
                "datamodule.data_config.test_dir is not set. Pass --data_dir <index_dir> "
                "(containing tt/mix.json+target.json) or edit the config's test_dir."
            )
        if "${" in configured or _ENV_PATTERN.search(configured):
            raise ValueError(
                f"test_dir still contains unresolved placeholders: {configured!r}. "
                "Export the referenced env vars (e.g. DATA_ROOT) or pass --data_dir."
            )
        test_dir = _resolve_index_dir(configured, split="tt")
    return train_conf, test_dir


def _expand_ckpt_candidates(path, exp_dir=None):
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    candidates = [expanded]
    if not os.path.isabs(expanded) and exp_dir:
        candidates.append(os.path.join(exp_dir, expanded))
    return candidates


def _find_best_ckpt(exp_dir):
    """Find the best checkpoint in an experiment directory.

    Priority: ``best_k_models.json`` (lowest val_loss) -> ``last.ckpt`` ->
    ``epoch=*.ckpt`` (numeric epoch order, highest epoch last).
    """
    best_k_path = os.path.join(exp_dir, "best_k_models.json")
    if os.path.isfile(best_k_path):
        with open(best_k_path) as f:
            best_k = json.load(f)
        best_k = {k: v for k, v in best_k.items() if isinstance(v, (int, float))}
        if best_k:
            # pick the lowest-loss entry that actually exists locally
            for path, loss in sorted(best_k.items(), key=lambda kv: kv[1]):
                for cand in _expand_ckpt_candidates(path, exp_dir):
                    if os.path.isfile(cand):
                        print(f"   Using best ckpt: {os.path.basename(cand)} (val_loss={loss:.4f})")
                        return cand
            print("   [eval] best_k_models.json found but none of its checkpoints exist locally; falling back")

    last_path = os.path.join(exp_dir, "last.ckpt")
    if os.path.isfile(last_path):
        print("   Using last.ckpt")
        return last_path

    epoch_ckpts = []
    for path in glob.glob(os.path.join(exp_dir, "epoch=*.ckpt")):
        m = re.search(r"epoch=(\d+)\.ckpt$", os.path.basename(path))
        if m:
            epoch_ckpts.append((int(m.group(1)), path))
    if epoch_ckpts:
        epoch, path = max(epoch_ckpts, key=lambda item: item[0])
        print(f"   Using latest ckpt: {os.path.basename(path)} (epoch={epoch}, numeric order)")
        return path

    raise FileNotFoundError(
        f"No checkpoint found in {exp_dir}. Pass --ckpt_path explicitly "
        "(e.g. --ckpt_path /root/.cache/falcon_weights/falcon.pt) or train first."
    )


def _extract_state_dict(obj, ckpt_path=""):
    """Extract the ``audio_model.*``/``video_model.*`` state dict from a checkpoint object."""
    if not isinstance(obj, dict):
        raise ValueError(f"{ckpt_path}: expected a dict checkpoint, got {type(obj).__name__}")
    state = obj.get("state_dict", obj)
    if not isinstance(state, dict):
        raise ValueError(f"{ckpt_path}: 'state_dict' is not a dict")
    if not any(isinstance(k, str) and k.startswith(("audio_model.", "video_model.")) for k in state):
        raise ValueError(
            f"{ckpt_path}: state dict has no 'audio_model.*'/'video_model.*' keys. "
            "Expected a full Lightning checkpoint or a released falcon.pt. "
            "A bare audio-only 'best_model.pth' (serialize() format) is not supported here."
        )
    return state


def _torch_load_checkpoint(path):
    """Load a checkpoint, preferring ``weights_only=True`` with a clear fallback."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on checkpoint contents
        print(f"   [eval] weights_only=True load failed ({type(exc).__name__}: {exc}); retrying weights_only=False")
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_models_from_ckpt(ckpt_path, config, device):
    """Build audio/video models from the config and load both from ``ckpt_path``."""
    train_conf = _get_train_conf(config)
    ckpt = _torch_load_checkpoint(ckpt_path)
    system_state = _extract_state_dict(ckpt, ckpt_path)

    # --- Audio model (config is deep-copied: Separator.__init__ mutates its dicts) ---
    audionet_cfg = copy.deepcopy(train_conf["audionet"]["audionet_config"])
    audionet_name = train_conf["audionet"]["audionet_name"]
    if not hasattr(look2hear.models, audionet_name):
        raise AttributeError(
            f"look2hear.models has no attribute {audionet_name!r} (config's audionet_name). "
            "The released Falcon weights use 'FalconSpeakerContrastive'; the shipped "
            "Experiments/checkpoint/Falcon/conf.yml still says 'SpeakerContrastiveV2' (stale snapshot)."
        )
    sample_rate = int(train_conf["datamodule"]["data_config"].get("sample_rate", 16000))
    model = getattr(look2hear.models, audionet_name)(sample_rate=sample_rate, **audionet_cfg)

    audio_state = {k[len("audio_model."):]: v for k, v in system_state.items() if k.startswith("audio_model.")}
    if not audio_state:
        raise ValueError(f"{ckpt_path}: no 'audio_model.*' weights found")
    try:
        model.load_state_dict(audio_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{ckpt_path}: audio_model strict load failed ({exc}). "
            "This usually means the config does not match the checkpoint architecture "
            "(e.g. a conf.yml saved after Separator popped repeats/shared, or a different model variant)."
        ) from exc
    print(f"   ✓ audio_model loaded: {len(audio_state)} keys from {os.path.basename(ckpt_path)}")

    # --- Video model (frozen FRCNN; never load 'pretrain' over trained weights) ---
    videonet_cfg = copy.deepcopy(train_conf["videonet"]["videonet_config"])
    pretrain_path = videonet_cfg.pop("pretrain", None)
    videonet_name = train_conf["videonet"]["videonet_name"]
    video_model = getattr(look2hear.videomodels, videonet_name)(**videonet_cfg, pretrain=None)

    video_state = {k[len("video_model."):]: v for k, v in system_state.items() if k.startswith("video_model.")}
    if video_state:
        try:
            video_model.load_state_dict(video_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"{ckpt_path}: video_model strict load failed ({exc})") from exc
        print(f"   ✓ video_model loaded: {len(video_state)} keys from {os.path.basename(ckpt_path)}")
    elif pretrain_path and os.path.isfile(pretrain_path):
        print(f"   ✓ video_model loaded from pretrain: {pretrain_path}")
        video_model = getattr(look2hear.videomodels, videonet_name)(
            **train_conf["videonet"]["videonet_config"])
    else:
        raise ValueError(
            f"{ckpt_path}: no 'video_model.*' weights found and pretrain path "
            f"{pretrain_path!r} does not exist"
        )

    model.to(device).eval()
    video_model.to(device).eval()
    return model, video_model


# ==========================================
# 3. Multi-process worker
# ==========================================
def _device_for_rank(rank, config):
    gpus = str(config.get("gpus", "0")).strip().lower()
    if gpus in ("", "cpu", "none") or not torch.cuda.is_available():
        if gpus not in ("", "cpu", "none"):
            print(f"[eval] CUDA not available; falling back to CPU (requested gpus={config.get('gpus')!r})")
        return torch.device("cpu")
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _mouth_to_tensor(mouth):
    """Convert an ``[F, H, W]`` numpy/torch mouth sequence to a float ``[1, 1, F, H, W]`` tensor."""
    if isinstance(mouth, np.ndarray):
        tensor = torch.from_numpy(mouth)
    elif isinstance(mouth, torch.Tensor):
        tensor = mouth.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(mouth))
    if tensor.dim() != 3:
        raise ValueError(f"expected mouth crops of shape [F, H, W], got {tuple(tensor.shape)}")
    return tensor.float().unsqueeze(0).unsqueeze(0)


def process_worker(rank, world_size, config):
    device = _device_for_rank(rank, config)
    train_conf = _get_train_conf(config)
    data_cfg = train_conf["datamodule"]["data_config"]
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    normalize_audio = bool(data_cfg.get("normalize_audio", False))
    test_dir = config["test_dir"]
    allow_partial = bool(config.get("allow_partial", False))
    n_done = 0  # utterances that actually produced a metrics row (coverage check input)

    dataset = AVSpeechEvalDataset(
        json_dir=test_dir,
        sample_rate=sample_rate,
        normalize_audio=normalize_audio,
    )
    indices = list(range(len(dataset)))
    limit = int(config.get("limit") or 0)
    if limit > 0:
        indices = indices[:limit]
    # Coverage invariant (W1-07): ``--limit`` is applied *before* sharding, so the
    # planned set is exactly ``min(limit, len(dataset))`` and ``indices[rank::world_size]``
    # partitions it across ranks. The explicit equality locks that against refactors
    # (e.g. applying ``--limit`` after the stride would silently shrink the plan).
    planned_total = len(indices)
    if planned_total != (min(limit, len(dataset)) if limit > 0 else len(dataset)):
        raise RuntimeError(
            f"[eval] internal error: planned_total={planned_total} but "
            f"min(limit={limit or 'all'}, dataset={len(dataset)}) disagrees"
        )
    is_subset_run = limit > 0 and limit < len(dataset)
    local_indices = indices[rank::world_size]
    expected_local = len(local_indices)
    if rank == 0:
        planned_per_rank = [len(range(planned_total)[r::world_size]) for r in range(world_size)]
        print(f"[eval] test index: {test_dir} | utterances={len(indices)} "
              f"| this rank={len(local_indices)} | device={device}")
        print(f"[eval] coverage plan: planned={planned_total} "
              f"(min(limit={limit or 'all'}, dataset={len(dataset)})) "
              f"| world_size={world_size} | planned per rank={planned_per_rank}")
        # A --limit run that truncates the set must announce itself: its means are
        # valid for these N utterances but are NOT full-set coverage.
        if is_subset_run:
            print(f"[eval] SUBSET run: --limit {limit} of {len(dataset)}; "
                  f"results are NOT full-set coverage")

    exp_dir = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", train_conf.get("exp", {}).get("exp_name", "eval")
    )
    ckpt_path = config.get("ckpt_path") or _find_best_ckpt(exp_dir)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    model, video_model = _load_models_from_ckpt(ckpt_path, config, device)

    ex_save_dir = os.path.join(exp_dir, "results")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics_file = os.path.join(ex_save_dir, f"metrics_rank{rank}.csv")
    metrics = ALLMetricsTracker(save_file=metrics_file, sample_rate=sample_rate)

    if rank == 0:
        metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
        progress = Progress(
            TextColumn("[bold blue]Testing (Rank 0)", justify="right"),
            BarColumn(bar_width=None),
            "•",
            BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            metricscolumn,
        )
        with progress:
            with torch.no_grad():
                for idx_count, idx in enumerate(progress.track(local_indices)):
                    mix, sources, mouth, key, _ = tensors_to_device(dataset[idx], device=device)
                    mouth_in = _mouth_to_tensor(mouth).float().to(device)
                    mouth_emb = video_model(mouth_in)
                    est_sources = model(mix.unsqueeze(0), mouth_emb)
                    metrics(mix=mix,
                            clean=sources.unsqueeze(0),
                            estimate=est_sources.squeeze(0),
                            key=key)
                    n_done += 1
                    if idx_count % 50 == 0:
                        metricscolumn.update(metrics.get_mean())
    else:
        with torch.no_grad():
            for idx in local_indices:
                mix, sources, mouth, key, _ = tensors_to_device(dataset[idx], device=device)
                mouth_in = _mouth_to_tensor(mouth).float().to(device)
                mouth_emb = video_model(mouth_in)
                est_sources = model(mix.unsqueeze(0), mouth_emb)
                metrics(mix=mix,
                        clean=sources.unsqueeze(0),
                        estimate=est_sources.squeeze(0),
                        key=key)
                n_done += 1

    metrics.final()

    # ---- coverage self-check (W1-07) ---------------------------------------
    # ``n_done`` counts utterances that produced a metrics row; ``expected_local``
    # is what this rank was asked to evaluate. A mismatch means a sample was
    # silently skipped (a future ``continue`` on failure, a filter, ...), which
    # would turn the reported mean into an unannounced subset: refuse, never warn.
    if rank == 0:
        print(f"[eval] coverage: evaluated {n_done}/{expected_local} on rank0 "
              f"(planned={planned_total}, world_size={world_size}, limit={limit or 'all'})")
        print(f"[eval] per-utterance results: {metrics_file}")
    if n_done != expected_local:
        coverage_msg = (
            f"[eval] coverage mismatch on rank {rank}: evaluated {n_done} of "
            f"{expected_local} planned utterances (planned_total={planned_total}, "
            f"world_size={world_size}, limit={limit or 'all'}); the reported mean would be a subset"
        )
        if allow_partial:
            print(f"[eval] WARNING (--allow-partial): {coverage_msg}")
        else:
            raise RuntimeError(
                coverage_msg + "; pass --allow-partial to downgrade this to a warning"
            )
    if rank == 0:
        coverage_path = os.path.join(ex_save_dir, COVERAGE_FILENAME)
        with open(coverage_path, "w", encoding="utf-8") as coverage_file:
            json.dump(
                {
                    "schema_version": 1,
                    "producer": "audio_visual_test.process_worker",
                    "rank": 0,
                    "world_size": world_size,
                    "limit": limit,
                    "dataset_len": len(dataset),
                    "planned_total": planned_total,
                    "planned_per_rank": [len(range(planned_total)[r::world_size]) for r in range(world_size)],
                    "rank0_evaluated": n_done,
                    "allow_partial": allow_partial,
                    "test_dir": test_dir,
                },
                coverage_file,
                indent=2,
                ensure_ascii=False,
            )
            coverage_file.write("\n")


# ==========================================
# 4. Main function and result merging
# ==========================================
def _run_workers(config):
    gpus = str(config.get("gpus", "0")).strip().lower()
    use_cuda = gpus not in ("", "cpu", "none") and torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpus"))
        world_size = len([x for x in str(config.get("gpus")).split(",") if x.strip()])
    else:
        if gpus not in ("", "cpu", "none"):
            print("[eval] CUDA unavailable; running on CPU")
        world_size = 1
    config = dict(config)
    config["gpus"] = "cpu" if not use_cuda else config.get("gpus")
    print(f"🚀 [eval] {world_size} worker(s) | gpus={config['gpus']} | limit={config.get('limit') or 'all'}")
    if world_size > 1:
        mp.spawn(process_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        process_worker(0, 1, config)
    return world_size


def _check_coverage(ex_save_dir, world_size, rows_per_rank, n_rows, allow_partial=False):
    """Verify that every planned utterance produced exactly one metrics row (W1-07).

    Two independent verdicts are reported, and they must not be conflated:

    * **row accounting**: did every *planned* utterance produce exactly one row on
      every rank? A mismatch raises ``RuntimeError`` (the CLI then exits non-zero)
      unless ``allow_partial`` is set, in which case it only warns.
    * **coverage scope**: was the run the *whole* test set? A ``--limit N`` run with
      ``N < dataset_len`` is a **subset run**. Its numbers stay valid for those N
      utterances, but they are not full-set coverage, so ``verified`` is forced to
      ``False`` with ``reason: "subset run: --limit N of M"`` even when the row
      accounting is perfect. Downstream consumers read ``limit`` / ``dataset_len`` /
      ``is_subset`` here (mirrored into ``metrics_counts.json.coverage``) and must
      never quote a subset as an M-utterance result.

    ``rows_per_rank`` are the per-shard per-utterance row counts, ``n_rows`` the
    merged row count. The plan comes from ``coverage_rank0.json``, written by rank 0
    after its inference loop. Returns a JSON-serializable report describing both
    verdicts.
    """
    path = os.path.join(ex_save_dir, COVERAGE_FILENAME)
    if not os.path.isfile(path):
        message = (
            f"[eval] no coverage report at {path}: no rank recorded how many utterances it "
            "planned to evaluate (worker died, or the run predates the coverage check)"
        )
        if allow_partial:
            print(f"[eval] WARNING (--allow-partial): {message}")
            return {
                "verified": False,
                "limit": None,
                "dataset_len": None,
                "is_subset": None,  # unknown: no plan to derive the scope from
                "reason": "missing coverage report",
            }
        raise RuntimeError(message + "; pass --allow-partial to downgrade this to a warning")

    with open(path, encoding="utf-8") as coverage_file:
        plan = json.load(coverage_file)

    # ---- declared coverage scope (W1-07b) ----------------------------------
    # ``limit`` is written by the worker as the raw CLI value (0 = "all"); normalise
    # 0/absent to None so ``is_subset`` is exactly ``limit is not None and
    # limit < dataset_len``. An unknown ``dataset_len`` cannot certify full-set
    # coverage, so it is a problem rather than an assumption of completeness.
    try:
        raw_limit = int(plan.get("limit") or 0)
    except (TypeError, ValueError):
        raw_limit = 0
    limit = raw_limit if raw_limit > 0 else None
    try:
        dataset_len = int(plan["dataset_len"])
    except (KeyError, TypeError, ValueError):
        dataset_len = None
    is_subset = limit is not None and dataset_len is not None and limit < dataset_len

    problems = []
    planned_total = int(plan.get("planned_total", -1))
    planned_per_rank = [int(value) for value in plan.get("planned_per_rank", [])]
    if dataset_len is None:
        problems.append("coverage report has no dataset_len (cannot certify full-set scope)")
    if planned_total < 0:
        problems.append("coverage report has no planned_total")
    if len(planned_per_rank) != world_size:
        problems.append(
            f"coverage report lists {len(planned_per_rank)} ranks but world_size={world_size}"
        )
    if planned_total != sum(planned_per_rank):
        problems.append(
            f"coverage report is self-inconsistent: planned_total={planned_total} but "
            f"planned_per_rank sums to {sum(planned_per_rank)}"
        )
    if dataset_len is not None and planned_total >= 0:
        expected_total = min(limit, dataset_len) if limit is not None else dataset_len
        planned_label = limit if limit is not None else "all"
        if planned_total != expected_total:
            problems.append(
                f"coverage report planned {planned_total} utterances but "
                f"min(limit={planned_label}, dataset_len={dataset_len})={expected_total}"
            )
    for rank, (planned, produced) in enumerate(zip(planned_per_rank, rows_per_rank)):
        if planned != produced:
            problems.append(f"rank {rank}: planned {planned} utterances, produced {produced} rows")
    if sum(rows_per_rank) != planned_total:
        problems.append(
            f"all shards produced {sum(rows_per_rank)} rows in total but {planned_total} were planned"
        )
    if n_rows != planned_total:
        problems.append(
            f"merged CSV has {n_rows} per-utterance rows but {planned_total} were planned"
        )

    # ``verified`` means "this CSV is the full test set, row for row": a subset run can
    # never be verified, no matter how clean its row accounting is.
    report = {
        "verified": False if is_subset else not problems,
        "limit": limit,
        "dataset_len": dataset_len,
        "is_subset": is_subset,
        "planned_total": planned_total,
        "planned_per_rank": planned_per_rank,
        "rows_per_rank": rows_per_rank,
        "rows_merged": n_rows,
        "allow_partial": bool(allow_partial),
    }
    if problems:
        report["problems"] = list(problems)
        if not is_subset:
            report["reason"] = "[eval] coverage check failed: " + "; ".join(problems)
    if is_subset:
        report["reason"] = f"subset run: --limit {limit} of {dataset_len}"

    if problems:
        message = "[eval] coverage check failed: " + "; ".join(problems)
        if allow_partial:
            print(f"[eval] WARNING (--allow-partial): {message}")
            return report
        raise RuntimeError(message + "; pass --allow-partial to downgrade this to a warning")

    if is_subset:
        print(
            f"[eval] coverage rows OK ({planned_total}/{planned_total}) but this is a SUBSET run: "
            f"--limit {limit} of {dataset_len}; verified=false, NOT full-set coverage"
        )
    else:
        print(
            f"[eval] coverage OK: {planned_total}/{planned_total} planned utterances produced rows "
            f"(per rank {rows_per_rank}, world_size={world_size})"
        )
    return report

def _merge_results(exp_dir, world_size, enforce_coverage=False, allow_partial=False):
    ex_save_dir = os.path.join(exp_dir, "results")
    frames = []
    rows_per_rank = []
    for rank in range(world_size):
        shard = os.path.join(ex_save_dir, f"metrics_rank{rank}.csv")
        if os.path.isfile(shard):
            frame = pd.read_csv(shard)
            frames.append(frame)
            # every shard carries its own trailing ``avg``/``std`` rows (ALLMetricsTracker.final)
            rows_per_rank.append(
                int((~frame["snt_id"].astype(str).isin(["avg", "std"])).sum())
            )
        else:
            rows_per_rank.append(0)
    if not frames:
        raise RuntimeError(f"no per-rank metrics CSV found in {ex_save_dir}")

    combined = pd.concat(frames, ignore_index=True)
    snt_ids = combined["snt_id"].astype(str)
    utterances = combined[~snt_ids.isin(["avg", "std"])].copy()
    if utterances.empty:
        raise RuntimeError("no per-utterance metric rows were produced (empty test split?)")

    # ---- coverage check (W1-07): planned vs produced, per rank and in total ----
    coverage = None
    if enforce_coverage:
        coverage = _check_coverage(
            ex_save_dir,
            world_size,
            rows_per_rank,
            n_rows=len(utterances),
            allow_partial=allow_partial,
        )

    # A4-4: mixture baselines (`si-snr_mix`/`sdr_mix`) are now per-utterance columns, so
    # `si-snr - si-snr_mix == si-snr_i` can be re-derived offline from metrics_combined.csv.
    # A4-3: aggregate rows must record how many finite values contributed (never a silent
    # nanmean); per-utterance rows leave the `n_*` columns empty (they are aggregate metadata).
    value_columns = [c for c in utterances.columns if c != "snt_id" and not c.startswith("n_")]
    count_columns = [c for c in utterances.columns if c.startswith("n_")]
    numeric = utterances[value_columns].apply(pd.to_numeric, errors="coerce")
    counts = {c: int(numeric[c].notna().sum()) for c in value_columns}
    nan_counts = {c: int(numeric[c].isna().sum()) for c in value_columns}

    avg = {"snt_id": "avg", **numeric.mean(numeric_only=True).to_dict()}
    std = {"snt_id": "std", **numeric.std(ddof=0, numeric_only=True).to_dict()}
    for count_column in count_columns:
        source = count_column[len("n_"):]
        avg[count_column] = counts.get(source, 0)
        std[count_column] = counts.get(source, 0)

    final = pd.concat([utterances, pd.DataFrame([avg, std])], ignore_index=True)
    final = final[list(utterances.columns)]  # keep the shard schema/order stable
    final_csv_path = os.path.join(ex_save_dir, "metrics_combined.csv")
    final.to_csv(final_csv_path, index=False)

    # Offline validity report for A4-3: count of finite values per metric column.
    identity = None
    if {"si-snr", "si-snr_i", "si-snr_mix"} <= set(value_columns):
        residual = (numeric["si-snr"] - numeric["si-snr_i"] - numeric["si-snr_mix"]).abs()
        identity = {
            "rule": "|si-snr - si-snr_i - si-snr_mix|",
            "max": float(residual.max()) if residual.notna().any() else None,
            "finite_pairs": int(residual.notna().sum()),
        }
    counts_payload = {
        "schema_version": 1,
        "producer": "audio_visual_test._merge_results",
        "workers": int(world_size),
        "n_utterances": int(len(utterances)),
        "columns": value_columns,
        "counts": counts,
        "nan_counts": nan_counts,
        "count_columns": count_columns,
        "aggregation": {
            "avg": "mean over utterance rows (NaN skipped)",
            "std": "population std, ddof=0 (NaN skipped)",
            "n_<col>": "number of finite (non-NaN) utterance values; identical on avg and std rows",
        },
        "si_snr_identity_check": identity,
        "coverage": coverage,
        "combined_csv": final_csv_path,
    }
    counts_json_path = os.path.join(ex_save_dir, "metrics_counts.json")
    with open(counts_json_path, "w", encoding="utf-8") as f:
        json.dump(counts_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n✅ [Success] Combined metrics saved to: {final_csv_path}")
    print(f"✅ [Success] Valid counts saved to: {counts_json_path}")
    print("-" * 40)
    print("Final metrics (per-utterance, all workers):")
    print(numeric.mean(numeric_only=True).to_string())
    print("-" * 40)
    print("Valid counts (finite values per column):")
    print(", ".join(f"{c}={counts[c]}/{len(utterances)}" for c in value_columns))
    print("-" * 40)

    for rank in range(world_size):
        shard = os.path.join(ex_save_dir, f"metrics_rank{rank}.csv")
        if os.path.isfile(shard):
            os.remove(shard)
    coverage_path = os.path.join(ex_save_dir, COVERAGE_FILENAME)
    if os.path.isfile(coverage_path):
        os.remove(coverage_path)  # the verified snapshot lives in metrics_counts.json
    return final_csv_path


def main(config):
    train_conf, test_dir = _prepare_config(_get_train_conf(config), config.get("data_dir", ""))
    config = dict(config)
    config["train_conf"] = train_conf
    config["test_dir"] = test_dir

    exp_dir = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", train_conf.get("exp", {}).get("exp_name", "eval")
    )
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    print(f"[eval] config      : {config.get('conf_dir', '<in-memory>')}")
    print(f"[eval] test index  : {test_dir}")
    print(f"[eval] ckpt_path   : {config.get('ckpt_path') or f'<auto in {exp_dir}>'}")
    print(f"[eval] results dir : {os.path.join(exp_dir, 'results')}")

    world_size = _run_workers(config)
    return _merge_results(
        exp_dir,
        world_size,
        enforce_coverage=True,
        allow_partial=bool(config.get("allow_partial", False)),
    )


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.conf_dir) as f:
        raw_conf = yaml.safe_load(f)
    if not isinstance(raw_conf, dict):
        raise ValueError(f"--conf_dir {args.conf_dir!r} does not contain a YAML mapping")

    ckpt_path = args.ckpt_path
    if ckpt_path.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModuleNotFoundError(
                "huggingface_hub is required for --ckpt_path hf://...; "
                "install it or pass a local checkpoint path"
            ) from exc
        repo_id, _, filename = ckpt_path[len("hf://"):].rpartition("/")
        if not repo_id or not filename:
            raise ValueError(f"expected hf://<org>/<repo>/<file>, got {ckpt_path!r}")
        print(f"[eval] downloading {filename} from HF repo {repo_id} (cached in ~/.cache/huggingface)")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        try:  # The Hub counts downloads by config.yaml, not by the weights file; best effort.
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            get_hf_file_metadata(hf_hub_url(repo_id, "config.yaml"))
        except Exception:
            pass

    arg_dic = {
        "conf_dir": args.conf_dir,
        "data_dir": args.data_dir,
        "ckpt_path": ckpt_path,
        "gpus": args.gpus,
        "limit": args.limit,
        "allow_partial": args.allow_partial,
        "train_conf": raw_conf,
    }
    try:
        main(arg_dic)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
