"""Static + cheap dynamic tests for the Phase-2 LRS2 launcher / v2 config / resume guard.

Covers review items A3-01 (Table 16 config), A3-03 (launcher hardening) and
A3-04 (resume guard + monitor version glob).  No GPU, no training, no network:

* launcher / config / monitor checks are static text or YAML reads;
* ``audio_train`` is imported only for the small pure helpers
  (``resolve_resume_checkpoint`` / ``set_seed``);
* the loss check runs one tiny CPU forward through ``PITLossWrapper``.

Run:
    PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest \
        tools/repro/tests/test_phase2_launcher.py -q
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LAUNCHER = REPO / "tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh"
MONITOR = REPO / "tools/repro/phase2/monitor_lrs2_paper.py"
CONF_V2 = REPO / "configs/LRS2_2Mix_Paper_v2.yml"
CONF_LEGACY = REPO / "configs/LRS2_2Mix_Paper_8gpu.yml"
TRAIN_PY = REPO / "audio_train.py"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing file: {path}"
    return path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 1) Launcher: pinned interpreter, no bare python3, rendered conf, GPU guards
# ──────────────────────────────────────────────────────────────────────────────
def test_launcher_uses_repo_python_and_no_bare_python3():
    text = _read(LAUNCHER)
    assert 'PYTHON_BIN="${PYTHON_BIN:' in text, "PYTHON_BIN must be overridable"
    assert '"$PYTHON_BIN"' in text, "inline python must go through $PYTHON_BIN"
    assert not re.search(r"(?<![\w$])python3(?![\w])", text), \
        "bare python3 found (system Python is forbidden by AGENTS.md §1)"
    # every mention of 'python' must be the pinned binary or a comment
    for i, line in enumerate(text.splitlines(), 1):
        if "python" not in line:
            continue
        ok = ("$PYTHON_BIN" in line or "PYTHON_BIN=" in line
              or "miniforge" in line or line.lstrip().startswith("#"))
        assert ok, f"line {i}: unqualified python invocation: {line!r}"


def test_launcher_checks_all_three_index_json_files():
    text = _read(LAUNCHER)
    for name in ("mix.json", "target.json", "near.json"):
        assert name in text, f"launcher must self-check {name}"
    assert '"tr", "cv", "tt"' in text or "('tr', 'cv', 'tt')" in text


def test_launcher_renders_data_root_into_a_temp_conf():
    text = _read(LAUNCHER)
    assert "RENDERED_CONF" in text
    assert "$TMPDIR" in text and "conf_$$.yml" in text, "temp conf path expected"
    # DATA_ROOT must be written into the three data dirs of a rendered YAML
    assert "DATA_ROOT" in text
    for key in ("train_dir", "valid_dir", "test_dir"):
        assert key in text, f"rendering must rewrite data_config.{key}"
    assert "yaml.safe_load" in text and "yaml.safe_dump" in text, \
        "conf rendering should be done by PyYAML (not blind sed)"
    assert "effective" in text.lower(), "must print the effective data paths"
    # training must point at the rendered conf, not the source conf
    assert '--conf_dir" "$RENDERED_CONF"' in text or '"$RENDERED_CONF"' in text


def test_launcher_enforces_nproc_and_prints_global_batch():
    text = _read(LAUNCHER)
    assert "torch.cuda.device_count()" in text
    assert "NPROC" in text
    assert "CUDA_VISIBLE_DEVICES=" in text, "must export CUDA_VISIBLE_DEVICES=0..N-1"
    assert "global batch" in text
    assert "DRY_RUN" in text and "dry-run" in text


# ──────────────────────────────────────────────────────────────────────────────
# 2) v2 config = paper Table 16; legacy config kept and marked
# ──────────────────────────────────────────────────────────────────────────────
def test_v2_config_matches_table16():
    conf = yaml.safe_load(_read(CONF_V2))
    assert conf["loss"]["train"]["sdr_type"] == "pairwise_neg_sisdr"
    assert conf["loss"]["train"]["config"]["threshold_byloss"] is False
    assert conf["spk_contrast"]["beta_anneal_epochs"] == 1000
    assert conf["datamodule"]["data_config"]["batch_size"] == 4
    assert conf["training"]["gpus"] == -1
    assert conf["training"]["seed"] == 42
    assert conf["training"]["epochs"] == 1000
    assert conf["training"]["early_stop"]["patience"] == 30
    assert conf["exp"]["exp_name"] == "LRS2_2Mix_Paper_v2"
    # unchanged from the paper recipe / old config
    assert conf["optimizer"] == {"optim_name": "adamw", "lr": 0.001, "weight_decay": 0.01}
    assert conf["scheduler"]["sche_config"]["patience"] == 15
    assert conf["scheduler"]["sche_config"]["factor"] == 0.5


def test_legacy_config_is_kept_and_marked():
    text = _read(CONF_LEGACY)
    first_lines = [ln for ln in text.splitlines() if ln.strip()][:3]
    assert first_lines and first_lines[0].startswith("# LEGACY"), "legacy marker missing"
    conf = yaml.safe_load(text)
    # legacy numbers must be untouched (only the header comment was added)
    assert conf["loss"]["train"]["sdr_type"] == "pairwise_neg_snr"
    assert conf["spk_contrast"]["beta_anneal_epochs"] == 200
    assert conf["datamodule"]["data_config"]["batch_size"] == 5
    assert conf["exp"]["exp_name"] == "LRS2_2Mix_Paper_8gpu"


def test_pairwise_neg_sisdr_loss_exists_and_runs():
    import torch

    from look2hear.losses import PITLossWrapper, pairwise_neg_sisdr

    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    est = torch.randn(2, 2, 1600)
    tgt = torch.randn(2, 2, 1600)
    value = loss(est, tgt)
    assert value.ndim == 0 and torch.isfinite(value), f"unexpected loss value: {value}"


# ──────────────────────────────────────────────────────────────────────────────
# 3) A3-04: resume guard + seed helper in audio_train.py
# ──────────────────────────────────────────────────────────────────────────────
def test_audio_train_exposes_resume_guard_and_seed():
    text = _read(TRAIN_PY)
    assert "def resolve_resume_checkpoint(" in text
    assert "def set_seed(" in text
    assert "seed = set_seed(" in text, "main() must apply training.seed"
    assert "Resuming from checkpoint" in text
    assert "WARNING: ckpt_path not found" not in text, \
        "old silent fallback (train from scratch) must be gone"


def test_resume_guard_raises_system_exit_on_missing_path(tmp_path):
    import audio_train

    assert audio_train.resolve_resume_checkpoint("") is None
    assert audio_train.resolve_resume_checkpoint(None) is None

    missing = tmp_path / "does_not_exist.ckpt"
    with pytest.raises(SystemExit):
        audio_train.resolve_resume_checkpoint(str(missing))

    real = tmp_path / "last.ckpt"
    real.write_bytes(b"not-a-real-checkpoint")
    assert audio_train.resolve_resume_checkpoint(str(real)) == str(real)


def test_seed_helper_is_optional_and_reports_seed():
    import audio_train

    assert audio_train.set_seed(None) is None
    assert audio_train.set_seed(42) == 42


# ──────────────────────────────────────────────────────────────────────────────
# 4) Monitor: glob version_* and pick the newest version (resume-safe)
# ──────────────────────────────────────────────────────────────────────────────
def _load_monitor():
    spec = importlib.util.spec_from_file_location("monitor_lrs2_paper", MONITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_uses_version_glob_not_hardcoded_version0():
    text = _read(MONITOR)
    assert "version_*" in text, "monitor must glob version_* (resume creates version_1, ...)"
    # the *code* must not hardcode a version_0 metrics path (docstrings may mention it)
    assert "version_0/metrics.csv" not in text
    assert "recursive=True" not in text, "avoid /root/** recursive scans (slow on big trees)"


def test_monitor_picks_highest_version_dir(tmp_path):
    monitor = _load_monitor()
    exp = tmp_path / "Experiments/checkpoint/LRS2_2Mix_Paper_v2"
    (exp / "version_2").mkdir(parents=True)
    (exp / "version_10").mkdir(parents=True)
    older = exp / "version_2" / "metrics.csv"
    newer = exp / "version_10" / "metrics.csv"
    older.write_text("epoch,val_pit_sisnr\n0,1.0\n", encoding="utf-8")
    newer.write_text("epoch,val_pit_sisnr\n0,1.0\n", encoding="utf-8")
    # numeric ordering: version_10 must win over version_2 (not lexicographic)
    assert monitor._version_key(newer) > monitor._version_key(older)


def test_monitor_finds_metrics_via_cwd_relative_root(tmp_path, monkeypatch):
    monitor = _load_monitor()
    exp_name = "LRS2_2Mix_Paper_v2"
    for version, value in (("version_0", 1.0), ("version_1", 2.0)):
        d = tmp_path / "Experiments/checkpoint" / exp_name / version
        d.mkdir(parents=True)
        (d / "metrics.csv").write_text(
            f"epoch,val_pit_sisnr,step\n0,{value},10\n", encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    found = monitor.find_metrics(exp_name)
    assert found is not None
    assert found.parent.name == "version_1", f"expected newest version, got {found}"
