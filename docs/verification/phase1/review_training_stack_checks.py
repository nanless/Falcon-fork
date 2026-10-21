"""Phase-1 independent reviewer checks for T1 (`audio_train.py`) + T2 (`look2hear/system/`).

Owner: studio-backend (task-16, reviewer role; implementation not written by me).
This script does NOT modify source; it only inspects the working tree (`repro/phase1`) and runs
throw-away smoke fits in temporary directories.

Coverage
  A. static: undefined-name scan (symtable), diff scope, no Comet-only API left,
     Trainer kwargs / no `gpus=`, unchanged model/loss/system construction blocks,
     unchanged training_step/validation_step sources, T2 changed lines inside
     `on_validation_epoch_end` only.
  B. unit: `build_logger` branch matrix (csv/tb/comet/none/None/False/unknown),
     PL2 logger=None vs False semantics, `resolve_accelerator_and_devices` matrix
     (cpu / single / multi / clamp / str / int / None) + edge cases ([] / 0 / -1).
  C. assembly: run `audio_train.main()` with fake datamodule/video/Trainer, verify
     conf snapshot is written BEFORE model construction, keeps repeats/shared and has no
     injected `causal`, and that the runtime config is polluted afterwards (F9); check
     Trainer kwargs; check `logger: none` reaches Trainer as `False`.
  D. smoke: real 2-step NASSv2 training via `audio_train.main()` (real 32000-sample audio
     model + fake video frontend + fake batches) with CSVLogger; verify artifacts
     (conf.yml / ckpt / best_k_models.json / best_model.pth / metrics.csv), CSV keys and
     values (learning_rate, val_pit_sisnr=-val_loss, bucket, test_pit_sisnr), global_step=2.
  E. no-logger smoke: same smoke with `training.logger: none` must complete and write no CSV.
  F. numerical invariance: eval/train forwards vs `baseline-fingerprint.json`, video forward
     vs `T5-post-fix-fingerprint.json` protocol (T5 files unchanged pre/post fix).
  G. multi-GPU: real-host resolver + a 2-GPU toy DDP fit using the resolved strategy/kwargs
     (~20 s, subprocess `_ddp_smoke_toy.py`).

Run (repo root, conda falcon env):
  /root/miniforge3/envs/falcon/bin/python docs/verification/phase1/review_training_stack_checks.py
Log: docs/verification/phase1/REVIEW-training-stack-checks.log
"""
from __future__ import annotations

import ast
import builtins
import copy
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
PHASE1 = REPO / "docs" / "verification" / "phase1"
LOG_PATH = PHASE1 / "REVIEW-training-stack-checks.log"
sys.path.insert(0, str(REPO))

RESULTS: list[tuple[str, str, str]] = []   # (status, name, detail)


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    return ok


def run(cmd: list[str], timeout: int = 300, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd or REPO), capture_output=True, text=True, timeout=timeout)


# ══════════════════════════════════════════════════════════════════════════════
# A. static
# ══════════════════════════════════════════════════════════════════════════════
def walk_undefined(path: Path) -> list[str]:
    """Independent symtable undefined-global scan (module + nested scopes)."""
    import symtable

    src = path.read_text()
    top = symtable.symtable(src, str(path), "exec")
    module_names = set(top.get_identifiers()) | set(dir(builtins))
    undefined: list[str] = []

    def walk(table, scope):
        for sym in table.get_symbols():
            if sym.is_global() and not sym.is_assigned() and sym.get_name() not in module_names:
                undefined.append(f"{scope}:{sym.get_name()}")
        for child in table.get_children():
            walk(child, f"{scope}.{child.get_name()}")

    walk(top, "<module>")
    return undefined


def class_func_unparse(src: str, class_name: str, func_name: str):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == func_name:
                    return ast.unparse(item)
    return None


def main_func_assign_unparse(src: str, target: str):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for item in ast.walk(node):
                if isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == target:
                            return ast.unparse(item)
    return None


def changed_lines(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse `git diff -U0` output -> {file: [(new_start, new_count), ...]}."""
    out: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            out.setdefault(cur, [])
        elif line.startswith("@@") and cur:
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2) or 1)
                out[cur].append((start, count))
    return out


def section_a() -> None:
    print("\n=== A. static checks ===")
    files = [
        REPO / "audio_train.py",
        REPO / "look2hear/system/av_litmodule.py",
        REPO / "look2hear/system/av_litmodule_nass_v2.py",
        REPO / "look2hear/system/av_litmodule_mdcl.py",
    ]
    for f in files:
        und = walk_undefined(f)
        check(f"static: no undefined globals in {f.name}", not und, str(und))

    at = (REPO / "audio_train.py").read_text()
    at_tree = ast.parse(at)
    names = {n.id for n in ast.walk(at_tree) if isinstance(n, ast.Name)}
    check("static: `comet_logger` undefined name removed", "comet_logger" not in names)

    check("static: `_normalize_gpus_spec` helper present", "def _normalize_gpus_spec" in at, "")
    check("static: old `_as_int_list` helper removed", "def _as_int_list" not in at, "")
    m = re.search(r'"--conf_dir",\s*default="([^"]+)"', at)
    check("static: --conf_dir default is configs/Falcon.yml", bool(m) and m.group(1) == "configs/Falcon.yml",
          m.group(1) if m else "not found")
    trainer_calls = [n for n in ast.walk(at_tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "Trainer"]
    check("static: exactly one pl.Trainer(...) call", len(trainer_calls) == 1, f"n={len(trainer_calls)}")
    if trainer_calls:
        kw = {k.arg for k in trainer_calls[0].keywords if k.arg}
        expected = {"max_epochs", "callbacks", "default_root_dir", "devices", "accelerator",
                    "strategy", "limit_train_batches", "gradient_clip_val", "logger",
                    "sync_batchnorm", "num_sanity_val_steps"}
        check("static: Trainer kwargs are PL2 API (no `gpus=`)", "gpus" not in kw and expected <= kw,
              f"got={sorted(kw)}")

    system_dir = REPO / "look2hear/system"
    comet_hits = []
    for f in system_dir.glob("*.py"):
        src = f.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "experiment":
                comet_hits.append(f"{f.name}:{node.lineno}")
    check("static: no executable `logger.experiment` (Comet-only API) in look2hear/system",
          not comet_hits, str(comet_hits))

    # T2 changed lines must be inside `on_validation_epoch_end` only.
    diff = run(["git", "diff", "-U0", "main", "--", "look2hear/system/"]).stdout
    hunks = changed_lines(diff).get("look2hear/system/av_litmodule.py", [])
    cur_src = (system_dir / "av_litmodule.py").read_text()
    span = None
    for node in ast.parse(cur_src).body:
        pass
    tree = ast.parse(cur_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "on_validation_epoch_end":
            span = (node.lineno, node.end_lineno)
    inside = bool(span) and all(span[0] <= s and s + c - 1 <= span[1] for s, c in hunks)
    check("static: T2 diff hunks confined to on_validation_epoch_end", inside,
          f"hunks={hunks} span={span}")

    # unchanged model/loss/system construction blocks between main and current
    base_at = run(["git", "show", "main:audio_train.py"]).stdout
    for target in ("model", "video_model", "optimizer", "loss_func", "system", "checkpoint"):
        a = main_func_assign_unparse(base_at, target)
        b = main_func_assign_unparse(at, target)
        check(f"static: audio_train `{target} =` construction unchanged", a == b and a is not None,
              "" if a == b else f"baseline={str(a)[:80]} current={str(b)[:80]}")

    # unchanged training_step / validation_step sources
    for mod, cls in (
        ("look2hear/system/av_litmodule.py", "AudioVisualLightningModule"),
        ("look2hear/system/av_litmodule_nass_v2.py", "AudioVisualLightningModuleNASSv2"),
        ("look2hear/system/av_litmodule_mdcl.py", "AudioVisualLightningModuleWithMDCL"),
    ):
        base = run(["git", "show", f"main:{mod}"]).stdout
        cur = (REPO / mod).read_text()
        for fn in ("training_step", "validation_step"):
            a = class_func_unparse(base, cls, fn)
            b = class_func_unparse(cur, cls, fn)
            if a is None and b is None:
                continue
            check(f"static: {cls}.{fn} unchanged", a == b,
                  "" if a == b else "function source differs")

    # T2: log_dict call keyword semantics
    t2_src = (system_dir / "av_litmodule.py").read_text()
    call_found = None
    for node in ast.walk(ast.parse(t2_src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "log_dict":
            call_found = node
    if call_found:
        kw = {k.arg: ast.unparse(k.value) for k in call_found.keywords}
        check("static: T2 log_dict(on_step=False, on_epoch=True, logger=True)",
              kw.get("on_step") == "False" and kw.get("on_epoch") == "True" and kw.get("logger") == "True",
              str(kw))
        metrics_keys = set()
        for node in ast.walk(ast.parse(t2_src)):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "metrics" and isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant):
                                metrics_keys.add(k.value)
                    if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "metrics" and isinstance(tgt.slice, ast.Constant)):
                        metrics_keys.add(tgt.slice.value)
        check("static: T2 preserves baseline metric keys (learning_rate/val_pit_sisnr/test_pit_sisnr)",
              {"learning_rate", "val_pit_sisnr", "test_pit_sisnr"} <= metrics_keys, str(sorted(metrics_keys)))
    else:
        check("static: T2 log_dict call found", False, "no log_dict call")


# ══════════════════════════════════════════════════════════════════════════════
# B. unit-level: logger + device resolution
# ══════════════════════════════════════════════════════════════════════════════
def section_b() -> None:
    print("\n=== B. unit: build_logger + _normalize_gpus_spec/resolve_accelerator_and_devices ===")
    import audio_train
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    cfg0 = yaml_load(REPO / "configs/Falcon.yml")

    def lg(kind, exp_dir="/tmp/phase1-review-logger/checkpoint/Falcon"):
        cfg = copy.deepcopy(cfg0)
        cfg["training"]["logger"] = kind
        return audio_train.build_logger(cfg, exp_dir)

    l = lg("csv")
    check("build_logger: default/explicit csv -> CSVLogger", isinstance(l, CSVLogger))
    if isinstance(l, CSVLogger):
        # P3-3 fix: CSV log dir must be <exp_dir>/version_N (single level, no duplicated exp name)
        check("build_logger: CSV log dir is single-level <exp_dir>/version_N",
              Path(l.log_dir) == Path("/tmp/phase1-review-logger/checkpoint/Falcon") / "version_0",
              f"log_dir={l.log_dir}")
    try:
        lg("tensorboard")
        check("build_logger: tensorboard without tensorboard pkg -> clear RuntimeError", False, "no exception")
    except RuntimeError as exc:
        check("build_logger: tensorboard without tensorboard pkg -> clear RuntimeError",
              "tensorboard" in str(exc).lower(), str(exc)[:100])
    try:
        lg("comet")
        check("build_logger: comet without comet-ml -> clear RuntimeError", False, "no exception")
    except RuntimeError as exc:
        check("build_logger: comet without comet-ml -> clear RuntimeError", "comet" in str(exc).lower(), str(exc)[:100])
    for kind in ("none", "null", "false", None, False):
        check(f"build_logger: {kind!r} -> None", lg(kind) is None)
    for kind in ("wandb", True, 3):
        try:
            lg(kind)
            check(f"build_logger: invalid {kind!r} -> ValueError", False, "no exception")
        except ValueError:
            check(f"build_logger: invalid {kind!r} -> ValueError", True)

    # PL semantics: logger=None creates a default CSVLogger, logger=False disables it.
    t_none = pl.Trainer(accelerator="cpu", devices=1, strategy="auto", max_epochs=1, logger=None,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    t_false = pl.Trainer(accelerator="cpu", devices=1, strategy="auto", max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    check("PL2: Trainer(logger=None) still creates a default logger (so T1 must pass False)",
          len(t_none.loggers) == 1 and t_false.logger is None, f"none={t_none.loggers} false={t_false.logger}")
    del t_none, t_false

    # _normalize_gpus_spec semantic contract (P2-1/P2-2 fix)
    spec_cases = [
        ([1], ("indices", [1])),
        ([7], ("indices", [7])),
        ([0, 1], ("indices", [0, 1])),
        ((0, 1), ("indices", [0, 1])),
        (range(2), ("indices", [0, 1])),
        ("0,1", ("indices", [0, 1])),
        ("1", ("indices", [1])),
        (8, ("count", 8)),
        (None, ("none", None)),
        ("", ("none", None)),
        (0, ("cpu", None)),
        (-1, ("all", None)),
    ]
    for value, expected in spec_cases:
        try:
            got = audio_train._normalize_gpus_spec(value)
            check(f"normalize_gpus_spec: {value!r} -> {expected}", got == expected, f"got={got}")
        except Exception as exc:  # noqa: BLE001
            check(f"normalize_gpus_spec: {value!r} -> {expected}", False, f"{type(exc).__name__}: {exc}")
    for bad in (True, 1.5, "a,b", {"gpu": 0}):
        try:
            audio_train._normalize_gpus_spec(bad)
            check(f"normalize_gpus_spec: invalid {bad!r} -> ValueError", False, "no exception")
        except ValueError:
            check(f"normalize_gpus_spec: invalid {bad!r} -> ValueError", True)

    # resolver matrix (monkeypatch cuda availability/count)
    real_avail, real_count = torch.cuda.is_available, torch.cuda.device_count
    cases = [
        (False, 0, {"gpus": list(range(10))}, ("cpu", 1, 1, "auto")),
        (True, 2, {"gpus": list(range(10))}, ("gpu", [0, 1], 2, "DDP")),
        (True, 1, {"gpus": [0]}, ("gpu", [0], 1, "auto")),
        (True, 2, {"gpus": [0]}, ("gpu", [0], 1, "auto")),
        (True, 2, {"gpus": [1]}, ("gpu", [1], 1, "auto")),          # P2-1 fixed: index not count
        (True, 2, {"gpus": [0, 7]}, ("gpu", [0], 1, "auto")),        # partial clamp + warning
        (True, 2, {"gpus": 8}, ("gpu", 2, 2, "DDP")),
        (True, 2, {"gpus": 1}, ("gpu", 1, 1, "auto")),
        (True, 2, {"gpus": "0,1"}, ("gpu", [0, 1], 2, "DDP")),
        (True, 2, {"gpus": "1"}, ("gpu", [1], 1, "auto")),
        (True, 2, {"gpus": None}, ("gpu", [0], 1, "auto")),
        (True, 2, {"gpus": 0}, ("cpu", 1, 1, "auto")),               # legacy PL1 0 -> CPU
        (True, 2, {"gpus": -1}, ("gpu", [0, 1], 2, "DDP")),          # legacy PL1 -1 -> all
    ]
    for avail, count, tc, expected in cases:
        torch.cuda.is_available = lambda a=avail: a
        torch.cuda.device_count = lambda c=count: c
        try:
            acc, dev, num, strat = audio_train.resolve_accelerator_and_devices(tc)
            got = (acc, dev, num, "DDP" if not isinstance(strat, str) else strat)
            check(f"resolve: cuda={avail} n={count} gpus={tc['gpus']!r} -> {expected}", got == expected, f"got={got}")
        except Exception as exc:  # noqa: BLE001
            check(f"resolve: cuda={avail} n={count} gpus={tc['gpus']!r} raised", False, f"{type(exc).__name__}: {exc}")
        finally:
            torch.cuda.is_available, torch.cuda.device_count = real_avail, real_count

    # P2-2 fixed: invalid specs raise clear ValueError here (not delayed from PL Trainer)
    error_cases = [[], [0, 0], [7], "a,b"]
    for gpus in error_cases:
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 2
        try:
            audio_train.resolve_accelerator_and_devices({"gpus": gpus})
            check(f"resolve: invalid gpus={gpus!r} -> ValueError", False, "no exception")
        except ValueError as exc:
            check(f"resolve: invalid gpus={gpus!r} -> ValueError", True, str(exc)[:80])
        finally:
            torch.cuda.is_available, torch.cuda.device_count = real_avail, real_count

    acc, dev, num, strat = audio_train.resolve_accelerator_and_devices(copy.deepcopy(cfg0["training"]))
    check("resolve: real host (2×A800, Falcon.yml gpus=[0..9]) -> gpu/[0,1]/DDP",
          (acc, num) == ("gpu", 2) and dev == [0, 1] and not isinstance(strat, str),
          f"got=({acc},{dev},{num},{type(strat).__name__})")
    # residual documentation gap (WARN only): `_normalize_gpus_spec` docstring claims the
    # contract is "documented in configs/Falcon.yml / Phase-1 README", but neither has it.
    cfg_lower = (REPO / "configs/Falcon.yml").read_text().lower()
    readme_lower = (PHASE1 / "README.md").read_text().lower()
    documented = ("training.logger" in cfg_lower) or ("training.logger" in readme_lower) or \
                 ("_normalize_gpus_spec" in readme_lower) or ("gpus 语义" in readme_lower) or \
                 ("gpus" in readme_lower and "索引" in readme_lower)
    check("docs: gpus/logger semantic contract documented in config or phase1 README (docstring claim) [WARN-only]",
          documented, "configs/Falcon.yml and phase1/README.md contain no logger/gpus-semantics docs",
          warn_only=True)


# ══════════════════════════════════════════════════════════════════════════════
# C. main() assembly with fakes (ordering + dump + Trainer kwargs)
# ══════════════════════════════════════════════════════════════════════════════
class _Stop(Exception):
    pass


class FakeTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeTrainer.last_kwargs = kwargs

    def fit(self, *a, **k):
        raise _Stop()


class FakeDM:
    def __init__(self, **kwargs):
        pass

    def setup(self):
        pass

    @property
    def make_loader(self):
        return None, None, None


class FakeVideo(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def yaml_load(path: Path):
    import yaml
    return yaml.safe_load(path.read_text())


def _run_main_with_fakes(cfg: dict) -> dict:
    import audio_train
    import look2hear.models
    import look2hear.videomodels

    tmp = Path(tempfile.mkdtemp(prefix="phase1-review-main-"))
    old_cwd = os.getcwd()
    old_t = audio_train.pl.Trainer
    old_dm = audio_train.AVSpeechDataModuleWithNear
    old_video = look2hear.videomodels.FRCNNVideoModel
    old_model = look2hear.models.FalconSpeakerContrastive
    info = {"tmp": str(tmp), "conf_exists_at_model_build": None, "model_seen": False}
    try:
        class WrappedModel(old_model):
            def __init__(self, *a, **k):
                conf = Path(tmp) / "Experiments/checkpoint" / cfg["exp"]["exp_name"] / "conf.yml"
                info["conf_exists_at_model_build"] = conf.exists()
                info["model_seen"] = True
                super().__init__(*a, **k)

        audio_train.pl.Trainer = FakeTrainer
        audio_train.AVSpeechDataModuleWithNear = FakeDM
        look2hear.videomodels.FRCNNVideoModel = FakeVideo
        look2hear.models.FalconSpeakerContrastive = WrappedModel
        os.chdir(tmp)
        runtime_cfg = copy.deepcopy(cfg)
        try:
            audio_train.main(runtime_cfg)
        except _Stop:
            pass
        info["kwargs"] = FakeTrainer.last_kwargs
        info["runtime_cfg"] = runtime_cfg
        info["dumped"] = yaml_load(tmp / "Experiments/checkpoint" / cfg["exp"]["exp_name"] / "conf.yml")
        return info
    finally:
        os.chdir(old_cwd)
        audio_train.pl.Trainer = old_t
        audio_train.AVSpeechDataModuleWithNear = old_dm
        look2hear.videomodels.FRCNNVideoModel = old_video
        look2hear.models.FalconSpeakerContrastive = old_model


def section_c() -> None:
    print("\n=== C. main() assembly (fakes): F9 ordering + dump + Trainer kwargs ===")
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies.ddp import DDPStrategy

    cfg = yaml_load(REPO / "configs/Falcon.yml")
    cfg["training"] = copy.deepcopy(cfg["training"])
    info = _run_main_with_fakes(cfg)
    check("assembly: model built after conf.yml snapshot exists", info["conf_exists_at_model_build"] is True,
          f"conf_exists={info['conf_exists_at_model_build']}")
    dumped = info["dumped"]
    ap = dumped["audionet"]["audionet_config"]["audio_params"]
    vp = dumped["audionet"]["audionet_config"]["video_params"]
    check("F9: dumped conf keeps audio_params.repeats/shared", {"repeats", "shared"} <= set(ap), str(sorted(ap)))
    check("F9: dumped conf keeps video_params.repeats/shared", {"repeats", "shared"} <= set(vp), str(sorted(vp)))
    check("F9: dumped audio_params has no injected `causal`", "causal" not in ap)
    check("F9: dumped conf has main_args.exp_dir", dumped.get("main_args", {}).get("exp_dir", "").endswith(
        os.path.join("Experiments", "checkpoint", cfg["exp"]["exp_name"])), str(dumped.get("main_args")))
    runtime_ap = info["runtime_cfg"]["audionet"]["audionet_config"]["audio_params"]
    check("F9: runtime config IS polluted after real model build (repeats popped, causal injected)",
          "repeats" not in runtime_ap and "causal" in runtime_ap, str(sorted(runtime_ap)))

    kw = info["kwargs"]
    check("assembly: logger is CSVLogger", isinstance(kw.get("logger"), CSVLogger), type(kw.get("logger")).__name__)
    check("assembly: PL2 accelerator/devices/strategy",
          kw.get("accelerator") in ("gpu", "cpu") and kw.get("strategy") is not None,
          f"acc={kw.get('accelerator')} devices={kw.get('devices')} strategy={type(kw.get('strategy')).__name__}")
    check("assembly: no `gpus` kw", "gpus" not in kw)
    check("assembly: training hyper-params unchanged",
          kw.get("limit_train_batches") == 1.0 and kw.get("gradient_clip_val") == 5.0
          and kw.get("sync_batchnorm") is True and kw.get("num_sanity_val_steps") == 0,
          f"limit={kw.get('limit_train_batches')} clip={kw.get('gradient_clip_val')} "
          f"sync_bn={kw.get('sync_batchnorm')} sanity={kw.get('num_sanity_val_steps')}")

    # logger: none -> Trainer(logger=False), never None
    cfg_none = yaml_load(REPO / "configs/Falcon.yml")
    cfg_none["training"]["logger"] = "none"
    info2 = _run_main_with_fakes(cfg_none)
    check("F1: `training.logger: none` reaches Trainer as False (not None)",
          info2["kwargs"].get("logger") is False, repr(info2["kwargs"].get("logger")))


# ══════════════════════════════════════════════════════════════════════════════
# D/E. real smoke fits (T1+T2 integration)
# ══════════════════════════════════════════════════════════════════════════════
T = 32000


class SmokeFakeVideo(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        B, _, Tt = x.shape[:3]
        return x.new_zeros(B, 512, Tt)


class _SmokeTrainDS(torch.utils.data.Dataset):
    def __init__(self, n=2):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        return (torch.randn(T, generator=g) * 0.05, torch.randn(T, generator=g) * 0.05,
                torch.rand(T, 88, 88, generator=g), f"f{i}.wav", torch.randn(T, generator=g) * 0.05)


class _SmokeValDS(torch.utils.data.Dataset):
    def __init__(self, n=1):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(2000 + i)
        return (torch.randn(T, generator=g) * 0.05, torch.randn(T, generator=g) * 0.05,
                torch.rand(T, 88, 88, generator=g), f"v{i}.wav", 1)


def _collate_train(batch):
    mix, tgt, mouth, names, near = zip(*batch)
    return torch.stack(mix), torch.stack(tgt), torch.stack(mouth), list(names), [[n] for n in near]


def _collate_val(batch):
    mix, tgt, mouth, names, nn = zip(*batch)
    return torch.stack(mix), torch.stack(tgt), torch.stack(mouth), list(names), list(nn)


class SmokeDM:
    def __init__(self, **kwargs):
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(_SmokeTrainDS(), batch_size=1, collate_fn=_collate_train, num_workers=0)
        self.val_loader = DataLoader(_SmokeValDS(), batch_size=1, collate_fn=_collate_val, num_workers=0)
        self.test_loader = DataLoader(_SmokeValDS(), batch_size=1, collate_fn=_collate_val, num_workers=0)

    def setup(self):
        pass

    @property
    def make_loader(self):
        return self.train_loader, self.val_loader, self.test_loader


class _RecordingTrainer:
    """Delegate to the real PL Trainer, but record resolved kwargs and final global_step."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._trainer = _RecordingTrainer.real_trainer(**kwargs)
        _RecordingTrainer.instances.append(self)

    def fit(self, *a, **k):
        self._trainer.fit(*a, **k)
        self.global_step_after = self._trainer.global_step

    def __getattr__(self, name):
        return getattr(self._trainer, name)


def _smoke_run(cfg: dict) -> dict:
    import audio_train
    import look2hear.videomodels
    import pytorch_lightning as pl

    tmp = Path(tempfile.mkdtemp(prefix="phase1-review-smoke-"))
    old_cwd = os.getcwd()
    old_t = audio_train.pl.Trainer
    old_dm, old_video = audio_train.AVSpeechDataModuleWithNear, look2hear.videomodels.FRCNNVideoModel
    _RecordingTrainer.real_trainer = pl.Trainer
    _RecordingTrainer.instances = []
    info = {"tmp": str(tmp)}
    try:
        audio_train.pl.Trainer = _RecordingTrainer
        audio_train.AVSpeechDataModuleWithNear = SmokeDM
        look2hear.videomodels.FRCNNVideoModel = SmokeFakeVideo
        os.chdir(tmp)
        runtime_cfg = copy.deepcopy(cfg)
        audio_train.main(runtime_cfg)
        info["runtime_cfg"] = runtime_cfg
        info["trainer"] = _RecordingTrainer.instances[-1]
        exp = Path(tmp) / "Experiments/checkpoint" / cfg["exp"]["exp_name"]
        info["exp_dir"] = str(exp)
        info["files"] = sorted(p.name for p in exp.iterdir()) if exp.exists() else []
        info["csv"] = glob.glob(str(exp / "**" / "metrics.csv"), recursive=True)
        info["conf"] = yaml_load(exp / "conf.yml")
        info["best_k"] = json.loads((exp / "best_k_models.json").read_text()) if (exp / "best_k_models.json").exists() else None
        info["ckpt_exists"] = bool(list(exp.glob("*.ckpt")))
        info["best_model_exists"] = (exp / "best_model.pth").exists()
        return info
    finally:
        os.chdir(old_cwd)
        audio_train.pl.Trainer = old_t
        audio_train.AVSpeechDataModuleWithNear = old_dm
        look2hear.videomodels.FRCNNVideoModel = old_video
        del _RecordingTrainer.real_trainer


def _read_csv_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames or [], list(rd)


def section_de() -> None:
    print("\n=== D. real 2-step smoke training (T1+T2, CSVLogger) ===")
    cfg = yaml_load(REPO / "configs/Falcon.yml")
    cfg["training"]["epochs"] = 1
    cfg["training"]["gpus"] = [0]
    cfg["exp"]["exp_name"] = "phase1_review_smoke"
    info = _smoke_run(cfg)
    print(f"  smoke dir: {info['tmp']}")
    check("smoke: training completed without exception", True)
    check("smoke: global_step == 2 (2 train batches)", getattr(info["trainer"], "global_step_after", None) == 2,
          f"global_step={getattr(info['trainer'], 'global_step_after', None)}")
    check("smoke: conf.yml + best_k_models.json + best_model.pth + *.ckpt written",
          {"conf.yml", "best_k_models.json", "best_model.pth"} <= set(info["files"]) and info["ckpt_exists"],
          str(info["files"]))
    check("smoke: CSVLogger produced metrics.csv", bool(info["csv"]), str(info["csv"]))
    if info["csv"]:
        exp = Path(info["exp_dir"])
        expected_csv = exp / "version_0" / "metrics.csv"
        check("smoke: metrics.csv is single-level <exp_dir>/version_0/metrics.csv",
              Path(info["csv"][0]) == expected_csv, f"got={info['csv'][0]} expected={expected_csv}")
    if info["csv"]:
        cols, rows = _read_csv_rows(info["csv"][0])
        expected_cols = {"learning_rate", "val_pit_sisnr", "test_pit_sisnr",
                         "val_loss/dataloader_idx_0", "train_loss_epoch", "val_pit_sisnr_1near"}
        check("smoke: metrics.csv preserves baseline keys (+bucket, +train keys)",
              expected_cols <= set(cols), f"missing={sorted(expected_cols - set(cols))}")
        lr_vals = [float(r["learning_rate"]) for r in rows if r.get("learning_rate")]
        check("smoke: learning_rate value == optimizer lr (1e-3)", all(abs(v - 1e-3) < 1e-7 for v in lr_vals) and len(lr_vals) >= 1,
              str(lr_vals[:3]))
        val_pairs = [(float(r["val_loss/dataloader_idx_0"]), float(r["val_pit_sisnr"])) for r in rows
                     if r.get("val_pit_sisnr") and r.get("val_loss/dataloader_idx_0")]
        check("smoke: val_pit_sisnr == -val_loss (semantics preserved)",
              len(val_pairs) >= 1 and all(abs(vp + vl) < 1e-4 for vl, vp in val_pairs), str(val_pairs[:2]))
        check("smoke: no NaN/Inf in logged numeric cells",
              all(not (isinstance(v, str) and v.lower() in ("nan", "inf", "-inf"))
                  for r in rows for v in r.values()), "")
        # on_epoch semantics: these appear at epoch-level rows only (the row that also carries epoch)
        epoch_rows = [r for r in rows if r.get("learning_rate")]
        check("smoke: epoch-end metrics logged once (on_epoch=True semantics)",
              len(epoch_rows) == 1, f"n_epoch_rows={len(epoch_rows)}")
    check("F9: smoke dumped conf keeps repeats/shared, no causal",
          {"repeats", "shared"} <= set(info["conf"]["audionet"]["audionet_config"]["audio_params"]) and
          "causal" not in info["conf"]["audionet"]["audionet_config"]["audio_params"])
    check("F9: smoke runtime config polluted after training",
          "repeats" not in info["runtime_cfg"]["audionet"]["audionet_config"]["audio_params"])
    check("smoke: best_k_models.json values are floats and point to existing ckpt",
          isinstance(info["best_k"], dict) and info["best_k"] and
          all(isinstance(v, float) for v in info["best_k"].values()) and
          all(os.path.exists(k) for k in info["best_k"]), str(info["best_k"])[:160])

    print("\n=== E. no-logger smoke (`training.logger: none`) ===")
    cfg2 = yaml_load(REPO / "configs/Falcon.yml")
    cfg2["training"]["epochs"] = 1
    cfg2["training"]["gpus"] = [0]
    cfg2["training"]["logger"] = "none"
    cfg2["exp"]["exp_name"] = "phase1_review_smoke_nolog"
    info2 = _smoke_run(cfg2)
    check("no-logger: training completed (F2 no-logger safe)", True)
    check("no-logger: Trainer received logger=False", info2["trainer"].kwargs.get("logger") is False)
    check("no-logger: no metrics.csv written", not info2["csv"], str(info2["csv"]))
    check("no-logger: artifacts still written (conf/ckpt/best_model)",
          {"conf.yml", "best_model.pth"} <= set(info2["files"]) and info2["ckpt_exists"], str(info2["files"]))


# ══════════════════════════════════════════════════════════════════════════════
# F. numerical invariance
# ══════════════════════════════════════════════════════════════════════════════
def section_f() -> None:
    print("\n=== F. numerical invariance vs Lead/T5 fingerprints ===")
    import look2hear.models as models
    from look2hear.videomodels import FRCNNVideoModel

    base = json.loads((PHASE1 / "baseline-fingerprint.json").read_text())
    t5 = json.loads((PHASE1 / "T5-post-fix-fingerprint.json").read_text())
    cfg = yaml_load(REPO / "configs/Falcon.yml")
    acfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
    vcfg = copy.deepcopy(cfg["videonet"]["videonet_config"])
    vcfg.pop("pretrain", None)
    ck = torch.load("/root/.cache/falcon_weights/falcon.pt", map_location="cpu", weights_only=False)["state_dict"]
    audio = models.FalconSpeakerContrastive(sample_rate=16000, **acfg)
    video = FRCNNVideoModel(**vcfg)
    audio.load_state_dict({k[len("audio_model."):]: v for k, v in ck.items() if k.startswith("audio_model.")}, strict=True)
    video.load_state_dict({k[len("video_model."):]: v for k, v in ck.items() if k.startswith("video_model.")}, strict=True)

    def sha(t):
        return hashlib.sha256(t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()

    audio.eval()
    torch.manual_seed(0)
    a = torch.randn(1, 32000)
    m = torch.randn(1, 512, 50)
    with torch.inference_mode():
        out = audio(a, m)
    check("fingerprint: eval forward sha256 == baseline", sha(out) == base["eval_forward"]["sha256"],
          f"got={sha(out)[:16]} base={base['eval_forward']['sha256'][:16]}")
    check("fingerprint: eval forward mean == baseline", abs(float(out.mean()) - base["eval_forward"]["mean"]) < 1e-12,
          f"got={float(out.mean())} base={base['eval_forward']['mean']}")
    audio.train()
    with torch.inference_mode():
        est, sep = audio(a, m)
    check("fingerprint: train est/ sep sha256 == baseline",
          sha(est) == base["train_forward"]["est_sha256"] and sha(sep) == base["train_forward"]["sep_sha256"])
    video.eval()
    torch.manual_seed(0)
    frames = torch.randn(1, 1, 50, 88, 88)
    with torch.inference_mode():
        vout = video(frames)
    check("fingerprint: video forward sha256 == T5 protocol fingerprint",
          sha(vout) == t5["video_forward"]["sha256"], f"got={sha(vout)[:16]} t5={t5['video_forward']['sha256'][:16]}")


# ══════════════════════════════════════════════════════════════════════════════
# G. multi-GPU DDP toy fit
# ══════════════════════════════════════════════════════════════════════════════
def section_g() -> None:
    print("\n=== G. multi-GPU semantics (toy DDP fit, 2×A800) ===")
    script = PHASE1 / "_ddp_smoke_toy.py"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        check("ddp: 2-GPU toy fit", False, "CUDA/2 GPUs unavailable", warn_only=True)
        return
    proc = run([sys.executable, str(script)], timeout=240)
    ok = proc.returncode == 0 and "DDP TOY FIT OK" in proc.stdout
    check("ddp: 2-GPU toy fit with resolved DDPStrategy/sync_batchnorm/CSVLogger", ok,
          (proc.stdout + proc.stderr).strip().splitlines()[-1] if (proc.stdout or proc.stderr) else "")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    _ddp = PHASE1 / "_ddp_smoke_toy.py"
    if not _ddp.exists():
        print(f"WARNING: {_ddp} missing; DDP check will be skipped", file=sys.stderr)
    sections = [section_a, section_b, section_c, section_de, section_f, section_g]
    for fn in sections:
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            check(f"section {fn.__name__} crashed", False, traceback.format_exc().splitlines()[-1])
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print("\n" + "=" * 70)
    print(f"RESULT: {len(RESULTS)} checks | PASS={len(RESULTS)-len(fails)-len(warns)} WARN={len(warns)} FAIL={len(fails)}")
    for st, name, detail in RESULTS:
        if st != "PASS":
            print(f"  [{st}] {name} :: {detail}")
    lines = [f"[{st}] {name} :: {detail}" for st, name, detail in RESULTS]
    lines.append(f"SUMMARY: total={len(RESULTS)} fail={len(fails)} warn={len(warns)}")
    LOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"log -> {LOG_PATH}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
