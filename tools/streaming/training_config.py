"""Validated flat YAML configuration for the two-GPU trainer.

Keys use long CLI option names with underscores. CLI values override YAML;
choosing a member of an exclusive CLI group replaces the complete group.
Relative paths in the file resolve against the configuration directory.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
PATH_KEYS = {"index_dir", "tar_index", "valid_index_dir", "valid_tar_index",
             "visual_init", "output", "checkpoint_dir", "resume", "warm_start", "teacher_checkpoint", "teacher_cache"}
BOOL_KEYS = {"synthetic", "random_visual", "reconstruction_only", "positive_contrastive", "exclude_degenerate_contrastive", "fixed_crop", "subband_refiner"}
INT_KEYS = {"validation_limit", "batch_size", "accum", "epochs", "max_steps",
            "save_every", "workers", "seed", "validation_batch_size", "model_channels",
            "warmup_steps", "lr_patience", "early_stop_patience", "teacher_microbatch", "attention_history_chunks"}
FLOAT_KEYS = {"contrastive_weight", "lr", "weight_decay", "min_delta", "variance_weight", "variance_floor", "silence_rms_threshold", "silence_loss_weight",
              "distillation_weight", "teacher_min_improvement_db", "teacher_min_sisdr_db", "spectral_loss_weight", "speech_ratio_floor"}
STR_KEYS = {"backend", "visual_mode", "synthesis_mode"}
KEYS = PATH_KEYS | BOOL_KEYS | INT_KEYS | FLOAT_KEYS | STR_KEYS
GROUPS = ({"index_dir", "tar_index", "synthetic"},
          {"valid_index_dir", "valid_tar_index"},
          {"visual_init", "random_visual"},
          {"reconstruction_only", "positive_contrastive"},
          {"resume", "warm_start"})


def yaml_arguments(argv=None):
    """Return merged argparse tokens and configuration path and SHA-256."""
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config")
    parsed, _ = pre.parse_known_args(argv)
    if not parsed.config:
        return argv, None, None
    config_path = Path(parsed.config).expanduser().resolve()
    raw = config_path.read_bytes()
    try:
        data = yaml.load(raw, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid training config {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("training configuration must be a YAML mapping")
    unknown = set(data) - KEYS
    if unknown:
        raise ValueError("unknown training configuration keys: " + ", ".join(sorted(unknown)))
    explicit = {a.split("=", 1)[0][2:].replace("-", "_") for a in argv if a.startswith("--")}
    overridden = set(explicit)
    for group in GROUPS:
        if group & explicit:
            overridden.update(group)
    tokens = []
    for key, value in data.items():
        if value is None:
            continue
        if key in BOOL_KEYS:
            valid = type(value) is bool
        elif key in INT_KEYS:
            valid = type(value) is int
        elif key in FLOAT_KEYS:
            valid = type(value) in (int, float)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(f"invalid type or empty value for config key {key}")
        if key in overridden:
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOL_KEYS:
            if value:
                tokens.append(flag)
        else:
            if key in PATH_KEYS:
                path = Path(value).expanduser()
                value = str((config_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
            tokens.extend((flag, str(value)))
    return tokens + argv, str(config_path), hashlib.sha256(raw).hexdigest()
