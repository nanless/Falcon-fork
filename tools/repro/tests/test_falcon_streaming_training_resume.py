"""Actual CLI training and optimizer-resume contract for FalconStreaming."""
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def run_train(*args, check=True):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run([sys.executable, "-m", "tools.streaming.train", *map(str, args)],
                          cwd=REPO, env=env, text=True, capture_output=True,
                          check=check, timeout=180)


def lr_and_step(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt["optimizer"]["param_groups"][0]["lr"], ckpt["steps"]


def test_resume_keeps_or_explicitly_overrides_saved_learning_rate(tmp_path):
    base, kept, override = [tmp_path / x for x in ("base.pt", "kept.pt", "override.pt")]
    common = ("--synthetic", "--chunks", "1", "--steps", "1", "--batch-size", "1", "--device", "cpu")
    run_train(*common, "--lr", "0.0004", "--output", base)
    run_train(*common, "--resume", base, "--output", kept)
    run_train(*common, "--resume", base, "--lr", "0.0001", "--output", override)
    assert lr_and_step(base) == pytest.approx((0.0004, 1))
    assert lr_and_step(kept) == pytest.approx((0.0004, 2))
    assert lr_and_step(override) == pytest.approx((0.0001, 2))
    # Existing output is detected before attempting to read nonexistent data.
    rejected = run_train("--index-dir", tmp_path / "missing", "--output", base, check=False)
    assert rejected.returncode != 0 and "output already exists" in rejected.stderr

@pytest.mark.parametrize("old_format", ["falcon-streaming-v1", "falcon-streaming-v2", "falcon-streaming-v3"])
def test_old_checkpoint_cannot_be_silently_interpreted_as_new_model(tmp_path, old_format):
    from tools.streaming.common import CHECKPOINT_FORMAT, read_checkpoint
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save({"format": old_format, "state_dict": {}}, old)
    torch.save({"format": CHECKPOINT_FORMAT, "state_dict": {"_streaming_format_version": torch.tensor(4, dtype=torch.int64)}, "model_config": {"channels": 192, "dilations": [1, 2, 4, 8, 16, 32]}}, new)
    with pytest.raises(ValueError, match="expected falcon-streaming-v4"):
        read_checkpoint(old, "cpu")
    assert read_checkpoint(new, "cpu")["format"] == "falcon-streaming-v4"


@pytest.mark.parametrize("version", [None, torch.tensor(1), torch.tensor(2), torch.tensor(3.0), torch.tensor([3])])
def test_v4_requires_valid_model_state_version(tmp_path, version):
    from tools.streaming.common import CHECKPOINT_FORMAT, read_checkpoint
    state = {} if version is None else {"_streaming_format_version": version}
    path = tmp_path / "bad.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "state_dict": state}, path)
    with pytest.raises(ValueError, match="expected scalar int64 _streaming_format_version=4"):
        read_checkpoint(path, "cpu")
