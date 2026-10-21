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
* ``--limit N`` evaluates only the first N test utterances (smoke runs);
* an eval worker does **not** build dataloaders and does not require the
  train/valid splits; it instantiates :class:`AVSpeechEvalDataset` on the
  resolved test index directly (whole-utterance inference, batch size 1).

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
                    help="Evaluate only the first N test utterances (0 = all)")


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

    dataset = AVSpeechEvalDataset(
        json_dir=test_dir,
        sample_rate=sample_rate,
        normalize_audio=normalize_audio,
    )
    indices = list(range(len(dataset)))
    limit = int(config.get("limit") or 0)
    if limit > 0:
        indices = indices[:limit]
    local_indices = indices[rank::world_size]
    if rank == 0:
        print(f"[eval] test index: {test_dir} | utterances={len(indices)} "
              f"| this rank={len(local_indices)} | device={device}")

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

    metrics.final()
    if rank == 0:
        print(f"[eval] per-utterance results: {metrics_file}")


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


def _merge_results(exp_dir, world_size):
    ex_save_dir = os.path.join(exp_dir, "results")
    frames = []
    for rank in range(world_size):
        shard = os.path.join(ex_save_dir, f"metrics_rank{rank}.csv")
        if os.path.isfile(shard):
            frames.append(pd.read_csv(shard))
    if not frames:
        raise RuntimeError(f"no per-rank metrics CSV found in {ex_save_dir}")

    combined = pd.concat(frames, ignore_index=True)
    snt_ids = combined["snt_id"].astype(str)
    utterances = combined[~snt_ids.isin(["avg", "std"])].copy()
    if utterances.empty:
        raise RuntimeError("no per-utterance metric rows were produced (empty test split?)")

    numeric = utterances.select_dtypes(include=[np.number])
    avg = {"snt_id": "avg", **numeric.mean(numeric_only=True).to_dict()}
    std = {"snt_id": "std", **numeric.std(ddof=0).to_dict()}
    final = pd.concat([utterances, pd.DataFrame([avg, std])], ignore_index=True)
    final_csv_path = os.path.join(ex_save_dir, "metrics_combined.csv")
    final.to_csv(final_csv_path, index=False)

    print(f"\n✅ [Success] Combined metrics saved to: {final_csv_path}")
    print("-" * 40)
    print("Final metrics (per-utterance, all workers):")
    print(numeric.mean(numeric_only=True).to_string())
    print("-" * 40)

    for rank in range(world_size):
        shard = os.path.join(ex_save_dir, f"metrics_rank{rank}.csv")
        if os.path.isfile(shard):
            os.remove(shard)
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
    return _merge_results(exp_dir, world_size)


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

    arg_dic = {
        "conf_dir": args.conf_dir,
        "data_dir": args.data_dir,
        "ckpt_path": ckpt_path,
        "gpus": args.gpus,
        "limit": args.limit,
        "train_conf": raw_conf,
    }
    try:
        main(arg_dic)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
