"""Phase-1 T1: regression tests for `audio_train.py` (training entry point).

These tests never launch a real training run:
  * `pl.Trainer` is monkeypatched by `FakeTrainer`, which records the constructor
    kwargs and raises `_StopAfterTrainer` from `fit()`;
  * datamodule / video model are monkeypatched (no LRS data, no FRCNN backbone);
  * the *real* audio model (`FalconSpeakerContrastive`) is built from
    `configs/Falcon.yml` so that the config-pollution ordering (F9) is exercised
    for real: `Separator.__init__` pops `repeats`/`shared` from the runtime dict.

Run:
    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audio_train_entry.py -q

Scope: this file only; no source files are modified by the tests.
"""
import ast
import copy
import os
import symtable
import sys

import pytest
import torch
import yaml

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import audio_train  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger  # noqa: E402
from pytorch_lightning.strategies.ddp import DDPStrategy  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / fakes
# ──────────────────────────────────────────────────────────────────────────────
class _StopAfterTrainer(Exception):
    """Raised by FakeTrainer.fit to stop main() right after Trainer construction."""


class FakeTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeTrainer.last_kwargs = kwargs

    def fit(self, *args, **kwargs):  # pragma: no cover - always raises
        raise _StopAfterTrainer()


class FakeDataModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def setup(self):
        return None

    @property
    def make_loader(self):
        return (None, None, None)


class FakeVideoModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):  # pragma: no cover - not used by these tests
        return x


def _load_config():
    with open(os.path.join(REPO, "configs", "Falcon.yml")) as f:
        return yaml.safe_load(f)


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audio_train, "AVSpeechDataModuleWithNear", FakeDataModule)
    import look2hear.videomodels as video_models

    monkeypatch.setattr(video_models, "FRCNNVideoModel", FakeVideoModel, raising=False)
    monkeypatch.setattr(audio_train.pl, "Trainer", FakeTrainer)
    FakeTrainer.last_kwargs = None
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────────
# Static checks
# ──────────────────────────────────────────────────────────────────────────────
def test_entry_parses():
    path = os.path.join(REPO, "audio_train.py")
    with open(path) as f:
        ast.parse(f.read())


def test_no_undefined_globals():
    """F1 regression: `comet_logger` (and any other name) must be defined."""
    import builtins

    path = os.path.join(REPO, "audio_train.py")
    with open(path) as f:
        top = symtable.symtable(f.read(), path, "exec")
    module_names = set(top.get_identifiers()) | set(dir(builtins))
    undefined = []

    def walk(table, scope):
        for sym in table.get_symbols():
            if sym.is_global() and not sym.is_assigned() and sym.get_name() not in module_names:
                undefined.append(f"{scope}:{sym.get_name()}")
        for child in table.get_children():
            walk(child, scope + "." + child.get_name())

    walk(top, "<module>")
    assert undefined == [], f"undefined globals: {undefined}"


def test_cli_keeps_conf_and_ckpt():
    option_strings = set()
    defaults = {}
    for action in audio_train.parser._actions:
        option_strings.update(action.option_strings)
        defaults.update({opt: action.default for opt in action.option_strings})
    assert "--conf_dir" in option_strings
    assert "--ckpt_path" in option_strings
    # P3-4: the documented (README) invocation target is now the default
    assert defaults["--conf_dir"] == "configs/Falcon.yml"


def test_normalize_gpus_spec_int_vs_list():
    # a list is ALWAYS explicit GPU indices; a scalar is ALWAYS a GPU count
    assert audio_train._normalize_gpus_spec([1]) == ("indices", [1])
    assert audio_train._normalize_gpus_spec([7]) == ("indices", [7])
    assert audio_train._normalize_gpus_spec((0, 1)) == ("indices", [0, 1])
    assert audio_train._normalize_gpus_spec("0,1") == ("indices", [0, 1])
    assert audio_train._normalize_gpus_spec(8) == ("count", 8)
    assert audio_train._normalize_gpus_spec(None) == ("none", None)
    assert audio_train._normalize_gpus_spec("") == ("none", None)
    assert audio_train._normalize_gpus_spec(0) == ("cpu", None)
    assert audio_train._normalize_gpus_spec(-1) == ("all", None)
    with pytest.raises(ValueError):
        audio_train._normalize_gpus_spec(True)
    # only -1 means "all GPUs"; other negatives are rejected
    with pytest.raises(ValueError):
        audio_train._normalize_gpus_spec(-2)
    with pytest.raises(ValueError):
        audio_train._normalize_gpus_spec(1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Logger factory (F1)
# ──────────────────────────────────────────────────────────────────────────────
def test_logger_default_is_csv(tmp_path):
    cfg = _load_config()
    exp_dir = tmp_path / "Falcon"
    logger = audio_train.build_logger(cfg, str(exp_dir))
    assert isinstance(logger, CSVLogger)
    # P3-3: no duplicated <exp_name>/<exp_name> in the log path
    assert os.path.dirname(str(exp_dir)) in str(logger.save_dir)
    assert logger.log_dir.startswith(str(exp_dir))
    assert os.path.relpath(logger.log_dir, str(exp_dir)).count("Falcon") == 0


def test_logger_none(tmp_path):
    cfg = _load_config()
    cfg["training"]["logger"] = "none"
    assert audio_train.build_logger(cfg, str(tmp_path)) is None


def test_logger_unknown_raises(tmp_path):
    cfg = _load_config()
    cfg["training"]["logger"] = "wandb"
    with pytest.raises(ValueError, match="Unsupported training.logger"):
        audio_train.build_logger(cfg, str(tmp_path))


def test_logger_comet_missing_dependency_message(tmp_path):
    pytest.importorskip  # noqa: B018  (documentation only)
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        cfg = _load_config()
        cfg["training"]["logger"] = "comet"
        with pytest.raises(RuntimeError, match="comet"):
            audio_train.build_logger(cfg, str(tmp_path))
    else:  # pragma: no cover - depends on env
        pytest.skip("comet-ml is installed in this environment")


# ──────────────────────────────────────────────────────────────────────────────
# PL 2.x accelerator/device resolution (F6)
# ──────────────────────────────────────────────────────────────────────────────
def test_resolve_devices_gpu_multi_clamped(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    accelerator, devices, num_devices, strategy = audio_train.resolve_accelerator_and_devices(
        {"gpus": list(range(10))}
    )
    assert accelerator == "gpu"
    assert devices == [0, 1]
    assert num_devices == 2
    assert isinstance(strategy, DDPStrategy)


def test_resolve_devices_gpu_single(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    accelerator, devices, num_devices, strategy = audio_train.resolve_accelerator_and_devices(
        {"gpus": [0]}
    )
    assert (accelerator, devices, num_devices, strategy) == ("gpu", [0], 1, "auto")


def test_resolve_devices_single_index_list_is_index(monkeypatch):
    """P2-1: `gpus: [1]` must select GPU 1, not '1 GPU' (which would use GPU 0)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert audio_train.resolve_accelerator_and_devices({"gpus": [1]}) == (
        "gpu", [1], 1, "auto")


def test_resolve_devices_out_of_range_index_raises(monkeypatch):
    """P2-2: out-of-range explicit indices fail fast with a clear ValueError."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="does not match any visible GPU"):
        audio_train.resolve_accelerator_and_devices({"gpus": [7]})


def test_resolve_devices_empty_list_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="empty list"):
        audio_train.resolve_accelerator_and_devices({"gpus": []})


def test_resolve_devices_duplicate_indices_raise(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="duplicate"):
        audio_train.resolve_accelerator_and_devices({"gpus": [0, 0]})


def test_resolve_devices_str_indices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    acc, devices, n, strategy = audio_train.resolve_accelerator_and_devices({"gpus": "0,1"})
    assert (acc, devices, n) == ("gpu", [0, 1], 2)
    assert isinstance(strategy, DDPStrategy)


def test_resolve_devices_legacy_scalars(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # legacy PL1 `gpus: 0` == CPU
    assert audio_train.resolve_accelerator_and_devices({"gpus": 0}) == ("cpu", 1, 1, "auto")
    # legacy PL1 `gpus: -1` == all visible GPUs
    acc, devices, n, strategy = audio_train.resolve_accelerator_and_devices({"gpus": -1})
    assert (acc, devices, n) == ("gpu", [0, 1], 2)
    assert isinstance(strategy, DDPStrategy)


def test_resolve_devices_gpu_int(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    accelerator, devices, num_devices, strategy = audio_train.resolve_accelerator_and_devices(
        {"gpus": 8}
    )
    assert accelerator == "gpu"
    assert devices == 2 and num_devices == 2
    assert isinstance(strategy, DDPStrategy)


def test_resolve_devices_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    accelerator, devices, num_devices, strategy = audio_train.resolve_accelerator_and_devices(
        {"gpus": list(range(10))}
    )
    assert (accelerator, devices, num_devices, strategy) == ("cpu", 1, 1, "auto")


# ──────────────────────────────────────────────────────────────────────────────
# main() end-to-end up to Trainer construction (F1/F6/F9)
# ──────────────────────────────────────────────────────────────────────────────
def test_main_constructs_trainer_and_dumps_safe_config(patched):
    tmp_path = patched
    cfg = _load_config()
    runtime_cfg = copy.deepcopy(cfg)

    with pytest.raises(_StopAfterTrainer):
        audio_train.main(runtime_cfg)

    kwargs = FakeTrainer.last_kwargs
    assert kwargs is not None
    # F1: logger is constructed (CSVLogger by default), not `comet_logger`
    assert isinstance(kwargs["logger"], CSVLogger)
    # F6: PL 2.x API
    assert kwargs["accelerator"] in ("gpu", "cpu")
    devices = kwargs["devices"]
    if isinstance(devices, list):
        assert devices and all(isinstance(d, int) and d >= 0 for d in devices)
    else:
        assert isinstance(devices, int) and devices >= 1
    assert kwargs["gradient_clip_val"] == 5.0
    assert kwargs["limit_train_batches"] == 1.0
    assert kwargs["sync_batchnorm"] is True
    assert kwargs["num_sanity_val_steps"] == 0
    assert kwargs["max_epochs"] == cfg["training"]["epochs"]
    assert kwargs["default_root_dir"].endswith(os.path.join("Experiments", "checkpoint", "Falcon"))
    assert any(isinstance(cb, ModelCheckpoint) for cb in kwargs["callbacks"])
    assert any(isinstance(cb, EarlyStopping) for cb in kwargs["callbacks"])
    csv_logger = kwargs["logger"]
    # P3-3: CSV path is <exp_dir>/version_N (no doubled "Falcon/Falcon")
    assert csv_logger.log_dir.startswith(kwargs["default_root_dir"])
    assert os.path.relpath(csv_logger.log_dir, kwargs["default_root_dir"]).count("Falcon") == 0

    # F9: dumped conf.yml must be the *pre-model* deepcopy (repeats/shared intact)
    conf_path = os.path.join(
        tmp_path, "Experiments", "checkpoint", "Falcon", "conf.yml"
    )
    assert os.path.exists(conf_path)
    with open(conf_path) as f:
        dumped = yaml.safe_load(f)
    dumped_audio = dumped["audionet"]["audionet_config"]["audio_params"]
    dumped_video = dumped["audionet"]["audionet_config"]["video_params"]
    for key in ("repeats", "shared"):
        assert key in dumped_audio, f"safe conf lost audio_params.{key}"
        assert key in dumped_video, f"safe conf lost video_params.{key}"
    assert "causal" not in dumped_audio, "safe conf must be dumped before Falcon injects `causal`"

    # A1-1 (2026-10-31): model construction must NOT mutate the caller's config anymore.
    # (Before the fix, Separator popped repeats/shared and injected causal into this dict; the
    # polluted dict was stored in ckpt hparams and rebuilt a 49k-parameter model.)
    runtime_audio = runtime_cfg["audionet"]["audionet_config"]["audio_params"]
    for key in ("repeats", "shared"):
        assert key in runtime_audio, f"model construction polluted caller config: lost audio_params.{key}"
    assert runtime_audio["repeats"] == 2 and runtime_audio["shared"] is True
    assert "causal" not in runtime_audio, "model construction must not inject `causal` into caller config"


def test_main_without_logger(patched):
    cfg = _load_config()
    cfg["training"]["logger"] = "none"
    with pytest.raises(_StopAfterTrainer):
        audio_train.main(cfg)
    # PL2: logger=False disables logging (logger=None would create a default logger)
    assert FakeTrainer.last_kwargs["logger"] is False
