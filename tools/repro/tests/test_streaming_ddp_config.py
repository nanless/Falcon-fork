"""Behavioral contract for YAML configuration of dual-GPU streaming training."""
from pathlib import Path

import pytest
import yaml


def write_config(tmp_path, text):
    directory = tmp_path / "experiment"
    directory.mkdir(exist_ok=True)
    config = directory / "train.yml"
    config.write_text(text, encoding="utf-8")
    return config


def parser():
    from tools.streaming.train_ddp import parse_args
    return parse_args


def test_config_sets_real_parameters_and_resolves_paths_against_yaml(tmp_path, monkeypatch):
    config = write_config(tmp_path, """
tar_index: data/train.json
valid_tar_index: data/valid.json
visual_init: weights/frcnn.pt
positive_contrastive: true
contrastive_weight: 0.75
output: run/final.pt
checkpoint_dir: run/epochs
batch_size: 3
accum: 2
epochs: 9
max_steps: 5
validation_limit: 16
lr: 0.0007
weight_decay: 0.02
workers: 2
seed: 17
backend: gloo
""")
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    args = parser()(["--config", str(config)])
    root = config.parent
    assert Path(args.tar_index) == root / "data/train.json"
    assert Path(args.valid_tar_index) == root / "data/valid.json"
    assert Path(args.visual_init) == root / "weights/frcnn.pt"
    assert Path(args.output) == root / "run/final.pt"
    assert Path(args.checkpoint_dir) == root / "run/epochs"
    assert args.positive_contrastive is True and not args.reconstruction_only
    assert (args.batch_size, args.accum, args.epochs, args.max_steps) == (3, 2, 9, 5)
    assert (args.validation_limit, args.workers, args.seed, args.backend) == (16, 2, 17, "gloo")
    assert (args.lr, args.weight_decay, args.contrastive_weight) == pytest.approx((.0007, .02, .75))


def test_explicit_cli_overrides_config_including_mutually_exclusive_groups(tmp_path):
    config = write_config(tmp_path, """
tar_index: input/train.json
valid_tar_index: input/cv.json
visual_init: weights.pt
positive_contrastive: true
output: out/final.pt
batch_size: 4
seed: 42
""")
    result = parser()(["--config", str(config), "--synthetic", "--random-visual",
                       "--reconstruction-only", "--batch-size", "1", "--seed", "0"])
    assert result.synthetic is True and result.tar_index is None
    assert result.random_visual is True and result.visual_init is None
    assert result.reconstruction_only is True and not result.positive_contrastive
    assert result.batch_size == 1 and result.seed == 0


@pytest.mark.parametrize("extra", [
    "unknown_parameter: 5\n",
    "batch_size: true\n",
    "batch_size: 1.5\n",
    "batch_size: -1\n",
    "lr: .nan\n",
    "synthetic: true\n",
    "positive_contrastive: false\n",
])
def test_invalid_configuration_is_rejected(tmp_path, extra):
    base = "tar_index: input/train.json\nvisual_init: weights.pt\npositive_contrastive: true\noutput: out/final.pt\n"
    config = write_config(tmp_path, base + extra)
    with pytest.raises((ValueError, SystemExit)):
        parser()(["--config", str(config)])


def test_duplicate_yaml_key_is_rejected_instead_of_silently_overriding(tmp_path):
    config = write_config(tmp_path, """
synthetic: true
random_visual: true
reconstruction_only: true
batch_size: 4
batch_size: 2
output: result.pt
""")
    with pytest.raises((ValueError, SystemExit, yaml.YAMLError)):
        parser()(["--config", str(config)])


def test_legacy_command_line_remains_usable_without_yaml(tmp_path):
    out = str(tmp_path / "checkpoint.pt")
    args = parser()(["--synthetic", "--random-visual", "--reconstruction-only", "--output", out,
                     "--batch-size", "1", "--accum", "1", "--epochs", "1", "--max-steps", "1"])
    assert args.synthetic and args.random_visual and args.reconstruction_only
    assert args.output == out and args.batch_size == args.accum == args.epochs == args.max_steps == 1


def test_resume_config_identity_and_cli_path_precedence(tmp_path, monkeypatch):
    import hashlib

    config = write_config(tmp_path, """
synthetic: true
random_visual: true
reconstruction_only: true
resume: checkpoint/last.pt
output: result/final.pt
""")
    monkeypatch.chdir(tmp_path)
    args = parser()(["--config", str(config)])
    assert Path(args.resume) == config.parent / "checkpoint/last.pt"
    assert args.config == str(config.resolve())
    assert args.config_sha256 == hashlib.sha256(config.read_bytes()).hexdigest()
    args = parser()(["--config", str(config), "--resume", "cli/last.pt", "--output", "cli/out.pt"])
    assert args.resume == "cli/last.pt"
    assert args.output == "cli/out.pt"


def test_configuration_is_identical_on_both_rank_environments(tmp_path, monkeypatch):
    config = write_config(tmp_path, """
synthetic: true
random_visual: true
reconstruction_only: true
output: run/final.pt
checkpoint_dir: run/checkpoints
batch_size: 2
seed: 12
""")
    results = []
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        results.append(vars(parser()(["--config", str(config)])))
    assert results[0] == results[1]



def test_null_learning_rate_and_explicit_override(tmp_path):
    config = write_config(tmp_path, """
synthetic: true
random_visual: true
reconstruction_only: true
output: result.pt
lr: null
""")
    assert parser()(["--config", str(config)]).lr is None
    assert parser()(["--config", str(config), "--lr", "0.0005"]).lr == pytest.approx(0.0005)


def test_seed_bounds_cover_both_distributed_ranks(tmp_path):
    import numpy as np

    args = ["--synthetic", "--random-visual", "--reconstruction-only",
            "--output", str(tmp_path / "out.pt"), "--seed"]
    for seed in (0, 2**32 - 2):
        parsed = parser()(args + [str(seed)])
        for rank in (0, 1):
            np.random.seed(parsed.seed + rank)
    for seed in (-1, 2**32 - 1):
        with pytest.raises(SystemExit):
            parser()(args + [str(seed)])


def test_only_complete_long_options_are_accepted_with_yaml(tmp_path):
    config = write_config(tmp_path, """
synthetic: true
visual_init: weights.pt
positive_contrastive: true
output: result.pt
""")
    args = parser()(["--config", str(config), "--random-visual"])
    assert args.random_visual and args.visual_init is None
    with pytest.raises(SystemExit):
        parser()(["--config", str(config), "--random-v"])
    with pytest.raises(SystemExit):
        parser()(["--conf", str(config), "--random-visual"])
