"""Phase1/T3 tests for the runnable evaluation entry point helpers.

No model / GPU / real data is needed: only the config, checkpoint-discovery and
result-merging helpers of ``audio_visual_test.py`` are exercised.

Run with::

    cd /root/code/github_repos/Falcon-fork
    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_eval_entry.py -q
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo>/tools/repro/tests/<file>
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import audio_visual_test as avt  # noqa: E402


# ---------------------------------------------------------------------------
# _find_best_ckpt
# ---------------------------------------------------------------------------
def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_best_ckpt_uses_numeric_epoch_order(tmp_path):
    for name in ("epoch=742.ckpt", "epoch=1000.ckpt", "epoch=999.ckpt"):
        _touch(tmp_path / name)
    found = avt._find_best_ckpt(str(tmp_path))
    assert os.path.basename(found) == "epoch=1000.ckpt"  # numeric, not lexical ("999" > "1000" lexically is false)


def test_find_best_ckpt_prefers_lowest_existing_best_k_entry(tmp_path):
    for name in ("a.ckpt", "b.ckpt"):
        _touch(tmp_path / name)
    missing = str(tmp_path / "missing.ckpt")
    best_k = {str(tmp_path / "a.ckpt"): -9.90,
              str(tmp_path / "b.ckpt"): -10.20,
              missing: -11.00}
    (tmp_path / "best_k_models.json").write_text(json.dumps(best_k))
    found = avt._find_best_ckpt(str(tmp_path))
    assert os.path.basename(found) == "b.ckpt"  # lowest value that exists


def test_find_best_ckpt_last_ckpt_priority(tmp_path):
    _touch(tmp_path / "last.ckpt")
    _touch(tmp_path / "epoch=1000.ckpt")
    assert os.path.basename(avt._find_best_ckpt(str(tmp_path))) == "last.ckpt"


def test_find_best_ckpt_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        avt._find_best_ckpt(str(tmp_path))


# ---------------------------------------------------------------------------
# _prepare_config / --data_dir / ${VAR} expansion
# ---------------------------------------------------------------------------
def _falcon_config():
    return yaml.safe_load((REPO_ROOT / "configs" / "Falcon.yml").read_text())


def _make_index(root: Path, split: str = "tt") -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    (split_dir / "mix.json").write_text("[]")
    (split_dir / "target.json").write_text("[]")
    return split_dir


def test_prepare_config_expands_env_placeholders(tmp_path, monkeypatch):
    index_root = tmp_path / "index"
    split_dir = _make_index(index_root)
    monkeypatch.setenv("T3_TEST_ROOT", str(index_root))
    raw = _falcon_config()
    raw["datamodule"]["data_config"]["test_dir"] = "${T3_TEST_ROOT}/tt"
    train_conf, test_dir = avt._prepare_config(raw)
    assert os.path.abspath(test_dir) == os.path.abspath(split_dir)


def test_prepare_config_data_dir_override(tmp_path):
    index_root = tmp_path / "index"
    split_dir = _make_index(index_root)
    raw = _falcon_config()
    raw["datamodule"]["data_config"]["test_dir"] = "--"
    train_conf, test_dir = avt._prepare_config(raw, data_dir=str(index_root))
    assert os.path.abspath(test_dir) == os.path.abspath(split_dir)


def test_prepare_config_unresolved_placeholder_raises(monkeypatch):
    monkeypatch.delenv("T3_UNSET_ROOT", raising=False)
    raw = _falcon_config()
    raw["datamodule"]["data_config"]["test_dir"] = "${T3_UNSET_ROOT}/tt"
    with pytest.raises(ValueError, match="unresolved|test_dir"):
        avt._prepare_config(raw)


def test_prepare_config_rejects_missing_sections():
    with pytest.raises(ValueError, match="missing required section"):
        avt._prepare_config({"audionet": {}, "videonet": {}})


# ---------------------------------------------------------------------------
# checkpoint state-dict extraction
# ---------------------------------------------------------------------------
def test_extract_state_dict_wrapped_and_bare():
    import torch

    wrapped = {"state_dict": {"audio_model.a": torch.zeros(1), "video_model.b": torch.zeros(2)},
               "architecture": "FalconSpeakerContrastive", "epoch": 714}
    for obj in (wrapped, wrapped["state_dict"]):
        state = avt._extract_state_dict(obj, "fake.pt")
        assert state["audio_model.a"].shape == (1,)
        assert state["video_model.b"].shape == (2,)


def test_extract_state_dict_rejects_audio_only_serialize_format():
    import torch

    with pytest.raises(ValueError, match="audio_model"):
        avt._extract_state_dict({"state_dict": {"w": torch.zeros(1), "b": torch.zeros(1)}})


# ---------------------------------------------------------------------------
# result merging: per-utterance rows only, single global avg/std
# ---------------------------------------------------------------------------
def test_merge_results_ignores_per_rank_avg_std_rows(tmp_path):
    exp_dir = tmp_path / "Experiments" / "checkpoint" / "Falcon"
    results = exp_dir / "results"
    results.mkdir(parents=True)
    shard = results / "metrics_rank0.csv"
    pd.DataFrame([
        {"snt_id": "u1", "sdr": 10.0, "sdr_i": 9.0, "si-snr": 11.0, "si-snr_i": 10.0, "pesq": 3.0, "stoi": 0.9},
        {"snt_id": "u2", "sdr": 20.0, "sdr_i": 19.0, "si-snr": 21.0, "si-snr_i": 20.0, "pesq": 4.0, "stoi": 0.95},
        {"snt_id": "avg", "sdr": 15.0, "sdr_i": 14.0, "si-snr": 16.0, "si-snr_i": 15.0, "pesq": 3.5, "stoi": 0.925},
        {"snt_id": "std", "sdr": 5.0, "sdr_i": 5.0, "si-snr": 5.0, "si-snr_i": 5.0, "pesq": 0.5, "stoi": 0.025},
    ]).to_csv(shard, index=False)

    final_path = avt._merge_results(str(exp_dir), world_size=1)
    combined = pd.read_csv(final_path)
    assert list(combined["snt_id"]) == ["u1", "u2", "avg", "std"]
    for column in ("sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"):
        assert combined.loc[combined["snt_id"] == "avg", column].iloc[0] == pytest.approx(
            (combined.loc[combined["snt_id"] == "u1", column].iloc[0]
             + combined.loc[combined["snt_id"] == "u2", column].iloc[0]) / 2.0
        )
    assert not (results / "metrics_rank0.csv").exists()  # shard cleaned up


def test_device_for_rank_cpu_fallback(monkeypatch):
    monkeypatch.setattr(avt.torch.cuda, "is_available", lambda: False)
    device = avt._device_for_rank(0, {"gpus": "0"})
    assert device.type == "cpu"
    device = avt._device_for_rank(0, {"gpus": "cpu"})
    assert device.type == "cpu"
