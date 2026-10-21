"""Phase-1 独立 QA 回归测试（task-15，owner: layers-dsp）。

目标：验证 Phase1（T1–T5）修复**不改变**官方权重下的模型数值/既有结论，且入口、Studio、
指标实现维持可运行。全部证据可从 `docs/verification/phase1/baseline-fingerprint.json`
（Lead 在未改动 main worktree 上生成）与 `/root/wt-main` 复现。

运行（在仓库根）：

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_phase1_regression.py -q

注意：
- 仅使用 conda `falcon` 环境（torch 2.8.0+cu128，CPU 足够）；不需要 GPU。
- 数值指纹的**算子顺序必须与基线一致**：`torch.manual_seed(0)` → 生成 audio/mouth 输入 →
  再构建模型（构建会消耗 RNG）→ eval 前向 → train 前向；视频指纹在 seed(1) 后先生成输入再建模，
  且**不调用 eval()**（基线在默认 training=True、BatchNorm 使用批内统计下生成，见 §QA-report）。
- 任一 T* 修复尚未落地时，对应“修复后不变量”测试会 skip（而不是 fail），便于分阶段 QA；
  全部落地后应 0 skip。
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
WEIGHTS = Path("/root/.cache/falcon_weights/falcon.pt")
BASELINE_PATH = REPO / "docs/verification/phase1/baseline-fingerprint.json"
MAIN_WT = Path("/root/wt-main")

EXPECTED_AUDIO_PARAMS = 537_697
EXPECTED_VIDEO_PARAMS = 11_186_688
EXPECTED_TOTAL_PARAMS = 11_724_385


# --------------------------------------------------------------------------- helpers
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha(t) -> str:
    """与基线生成脚本一致的指纹：float32 CPU contiguous 原始字节 sha256。"""
    t = t.detach().to("cpu").contiguous()
    return _sha256_bytes(t.numpy().tobytes())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, cwd=REPO, env=None, timeout=900):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=str(cwd), env=e, capture_output=True, text=True, timeout=timeout)


def _git_diff_names() -> list[str]:
    r = _run(["git", "diff", "--name-only", "main"])
    return [x for x in r.stdout.splitlines() if x.strip()]


def _git_untracked_names() -> list[str]:
    r = _run(["git", "status", "--porcelain", "-uall"])
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("?? "):
            out.append(line[3:].strip())
    return out


def _landed(path_prefix: str) -> bool:
    """该文件相对 main 是否已发生改动（含新文件），用于分阶段 skip。"""
    diffs = _git_diff_names()
    if any(d == path_prefix or d.startswith(path_prefix.rstrip("/") + "/") for d in diffs):
        return True
    return any(u == path_prefix or u.startswith(path_prefix.rstrip("/") + "/")
               for u in _git_untracked_names())


# ------------------------------------------------------------------- session fixtures
@pytest.fixture(scope="session")
def baseline() -> dict:
    assert BASELINE_PATH.is_file(), f"缺少基线文件 {BASELINE_PATH}"
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def fingerprints() -> dict:
    """按基线的**精确算子顺序**重算官方权重数值指纹（只算一次）。"""
    import yaml

    sys.path.insert(0, str(REPO))
    import look2hear.models as models
    from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel

    cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
    acfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    audio_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    video_sd = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}

    # --- 与 baseline-fingerprint.json 相同的顺序：seed → 输入 → 建模 ---
    torch.manual_seed(0)
    audio = torch.randn(1, 32000)
    mouth_emb = torch.randn(1, 512, 50)

    audio_model = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg))
    audio_model.load_state_dict(audio_sd, strict=True)
    audio_model.eval()
    with torch.no_grad():
        eval_out = audio_model(audio, mouth_emb)

    audio_model.train()
    with torch.no_grad():
        train_est, sep_emb = audio_model(audio, mouth_emb)

    torch.manual_seed(1)
    video_in = torch.randn(1, 1, 50, 88, 88)
    video_model = FRCNNVideoModel(pretrain=None)
    video_model.load_state_dict(video_sd, strict=True)
    # 基线**未**显式调用 eval()/train(False)：FRCNNVideoModel 默认 training=True，
    # BatchNorm 使用批内统计；调用 eval() 会改变指纹（sum 5955.7 vs 7832.3）。T5 的 return self
    # 修复不影响本路径（本文件不调用 eval）。
    with torch.no_grad():
        video_out = video_model(video_in)

    return {
        "audio_params": sum(p.numel() for p in audio_model.parameters()),
        "video_params": sum(p.numel() for p in video_model.parameters()),
        "audio_keys": len(audio_sd),
        "video_keys": len(video_sd),
        "eval_shape": tuple(eval_out.shape),
        "eval_sha": _tensor_sha(eval_out),
        "eval_sum": float(eval_out.sum()),
        "train_est_shape": tuple(train_est.shape),
        "train_est_sha": _tensor_sha(train_est),
        "sep_shape": tuple(sep_emb.shape),
        "sep_sha": _tensor_sha(sep_emb),
        "video_shape": tuple(video_out.shape),
        "video_sha": _tensor_sha(video_out),
        "video_sum": float(video_out.sum()),
    }


# =============================================================== 1) 环境/权重基线
def test_environment_and_weights(baseline):
    """conda falcon 环境 + 官方权重与基线一致（sha/size/keys 数）。"""
    import torch as _torch

    assert sys.version_info[:2] == (3, 11), f"必须用 conda falcon (py3.11)，当前 {sys.version_info}"
    assert _torch.__version__.startswith("2.8.0"), f"torch 版本漂移: {_torch.__version__}"
    assert WEIGHTS.is_file(), f"缺少官方权重 {WEIGHTS}"
    assert WEIGHTS.stat().st_size == baseline["weights"]["size"]
    assert _file_sha256(WEIGHTS) == (
        "a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c"
    ), "falcon.pt sha256 与 Phase-0/基线不一致"


# =============================================================== 2) 参数量不变量
def test_parameter_counts_match_baseline(fingerprints, baseline):
    """官方权重严格加载后：audio 537,697 / video 11,186,688 / 合计 11,724,385。"""
    assert fingerprints["audio_keys"] == baseline["weights"]["audio_keys"] == 525
    assert fingerprints["video_keys"] == baseline["weights"]["video_keys"] == 137
    assert fingerprints["audio_params"] == EXPECTED_AUDIO_PARAMS == baseline["params"]["audio"]
    assert fingerprints["video_params"] == EXPECTED_VIDEO_PARAMS == baseline["params"]["video"]
    assert (fingerprints["audio_params"] + fingerprints["video_params"]) == EXPECTED_TOTAL_PARAMS


# =============================================================== 3) 数值指纹不变量
def test_eval_forward_fingerprint_unchanged(fingerprints, baseline):
    """eval 前向：形状 [1,1,32000]、sha256 与基线完全一致（max|Δ|=0 的等价证据）。"""
    b = baseline["eval_forward"]
    assert list(fingerprints["eval_shape"]) == b["shape"] == [1, 1, 32000]
    assert fingerprints["eval_sha"] == b["sha256"], (
        "eval 输出指纹变化！若 T1–T5 修复未触碰前向，说明引入了数值变化"
    )
    assert abs(fingerprints["eval_sum"] - b["sum"]) < 1e-5


def test_train_forward_fingerprint_unchanged(fingerprints, baseline):
    """train 模式 (est, sep_emb)：est sha256 + sep_emb [1,1,112,251,129] sha256 与基线一致。"""
    b = baseline["train_forward"]
    assert list(fingerprints["train_est_shape"]) == b["est_shape"] == [1, 1, 32000]
    assert fingerprints["train_est_sha"] == b["est_sha256"]
    assert list(fingerprints["sep_shape"]) == b["sep_shape"] == [1, 1, 112, 251, 129]
    assert fingerprints["sep_sha"] == b["sep_sha256"]


def test_video_forward_fingerprint_unchanged(fingerprints, baseline):
    """视频前端（默认 training 模式）输出 [1,512,50]，sha256 与基线一致。"""
    b = baseline["video_forward"]
    assert list(fingerprints["video_shape"]) == b["shape"] == [1, 512, 50]
    assert fingerprints["video_sha"] == b["sha256"], (
        "视频指纹变化：注意基线是在**未调用 eval()**（BN 批内统计）下生成的；"
        "若有人改了 FRCNNVideoModel 的默认训练/推理切换语义，这里会命中"
    )


# =============================================================== 4) 与 main worktree 对比
_MAIN_WT_SCRIPT = r'''
import copy, hashlib, json, sys
import torch, yaml

REPO = {repo!r}
WEIGHTS = {weights!r}
sys.path.insert(0, REPO)
import look2hear.models as models
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel

def sha(t):
    t = t.detach().to("cpu").contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

cfg = yaml.safe_load(open(REPO + "/configs/Falcon.yml").read())
acfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
a_sd = {{k[12:]: v for k, v in sd.items() if k.startswith("audio_model.")}}
v_sd = {{k[12:]: v for k, v in sd.items() if k.startswith("video_model.")}}

torch.manual_seed(0)
audio = torch.randn(1, 32000)
mouth = torch.randn(1, 512, 50)
m = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg))
m.load_state_dict(a_sd, strict=True)
m.eval()
with torch.no_grad():
    y = m(audio, mouth)
m.train()
with torch.no_grad():
    est, sep = m(audio, mouth)

torch.manual_seed(1)
vin = torch.randn(1, 1, 50, 88, 88)
v = FRCNNVideoModel(pretrain=None)
v.load_state_dict(v_sd, strict=True)
with torch.no_grad():
    vy = v(vin)

print(json.dumps({{
    "audio_params": sum(p.numel() for p in m.parameters()),
    "video_params": sum(p.numel() for p in v.parameters()),
    "eval_sha": sha(y),
    "train_est_sha": sha(est),
    "sep_sha": sha(sep),
    "video_sha": sha(vy),
    "video_sum": float(vy.sum()),
}}))
'''


def test_numeric_invariance_vs_main_worktree(fingerprints, baseline, tmp_path):
    """在干净 main worktree（/root/wt-main）复算指纹，验证 T1–T5 修复与 main 数值逐位一致。"""
    if not (MAIN_WT / "look2hear/models/Falcon.py").is_file():
        pytest.skip("干净 main worktree /root/wt-main 不存在，跳过跨分支数值对比")
    script = tmp_path / "fp_main.py"
    script.write_text(_MAIN_WT_SCRIPT.format(repo=str(MAIN_WT), weights=str(WEIGHTS)))
    env = {"PYTHONPATH": str(MAIN_WT)}
    r = _run([PY, str(script)], cwd=MAIN_WT, env=env, timeout=900)
    assert r.returncode == 0, f"main worktree 指纹脚本失败:\nSTDOUT={r.stdout[-3000:]}\nSTDERR={r.stderr[-3000:]}"
    got = json.loads(r.stdout.strip().splitlines()[-1])
    # main 侧自身应与基线一致（验证基线 JSON 可复现）
    assert got["eval_sha"] == baseline["eval_forward"]["sha256"]
    assert got["train_est_sha"] == baseline["train_forward"]["est_sha256"]
    assert got["sep_sha"] == baseline["train_forward"]["sep_sha256"]
    assert got["video_sha"] == baseline["video_forward"]["sha256"]
    # 当前分支应与 main 逐位一致（正式相位下：无任何数值变化）
    assert got["eval_sha"] == fingerprints["eval_sha"], "eval 输出与 main 不一致"
    assert got["train_est_sha"] == fingerprints["train_est_sha"], "train est 与 main 不一致"
    assert got["sep_sha"] == fingerprints["sep_sha"], "sep_emb 与 main 不一致"
    assert got["video_sha"] == fingerprints["video_sha"], "视频前端输出与 main 不一致"
    assert got["audio_params"] == fingerprints["audio_params"]
    assert got["video_params"] == fingerprints["video_params"]


# =============================================================== 5) import 完整性
IMPORT_MODULES = [
    "look2hear.models",
    "look2hear.models.Falcon",
    "look2hear.models.Falcon_Contrastive",
    "look2hear.videomodels",
    "look2hear.system",
    "look2hear.metrics",
    "look2hear.losses",
    "look2hear.datas",
    "look2hear.layers",
    "look2hear.utils",
    "audio_train",
    "audio_visual_test",
]


def test_import_completeness():
    """全部包 + 训练/评测入口在 falcon 环境可导入（子进程隔离，避免污染本进程）。"""
    code = "import importlib,sys\n" + "\n".join(
        f"importlib.import_module({m!r})" for m in IMPORT_MODULES
    ) + "\nprint('IMPORTS_OK')"
    r = _run([PY, "-c", code], cwd=REPO, env={"PYTHONPATH": str(REPO)}, timeout=600)
    assert r.returncode == 0, (
        f"导入失败（Phase1 修复可能破坏 import）:\nSTDOUT={r.stdout[-3000:]}\nSTDERR={r.stderr[-4000:]}"
    )
    assert "IMPORTS_OK" in r.stdout


# =============================================================== 6) Studio 回归
def test_studio_pytest_regression():
    """Studio Python 测试维持基线：≥35 passed / 2 skipped / 0 failed。"""
    r = _run([PY, "-m", "pytest", "studio/tests", "-q"], cwd=REPO, timeout=900)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        pytest.fail(f"studio/tests 回归失败（rc={r.returncode}）:\n{out[-4000:]}")
    m_pass = re.search(r"(\d+) passed", out)
    m_skip = re.search(r"(\d+) skipped", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    skipped = int(m_skip.group(1)) if m_skip else 0
    assert passed >= 35, f"Studio 通过数下降：{passed} (<35)\n{out[-2000:]}"
    assert skipped == 2, f"Studio skip 数变化：{skipped} (期望 2)\n{out[-2000:]}"


# =============================================================== 7) 指标第二意见
def _independent_si_sdr(est, ref) -> float:
    """独立实现（numpy float64）的零均值 SI-SDR / dB，单源。"""
    import numpy as np

    e = np.asarray(est, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    e = e - e.mean()
    r = r - r.mean()
    alpha = float((e @ r) / (r @ r + 1e-12))
    proj = alpha * r
    noise = e - proj
    return float(10.0 * np.log10((proj @ proj + 1e-12) / (noise @ noise + 1e-12)))


def _metrics_fixture_signals():
    """固定种子的合成信号：target s1、干扰 s2、mixture、estimate=s1+0.3*s2+噪声。"""
    import torch as _t

    g = _t.Generator().manual_seed(20261021)
    n = 16000
    t = _t.arange(n, dtype=_t.float64) / 16000.0
    s1 = _t.sin(2 * _t.pi * 440 * t) + 0.3 * _t.sin(2 * _t.pi * 880 * t)
    s2 = _t.sin(2 * _t.pi * 1234 * t) * 0.8
    noise = 0.05 * _t.randn(n, generator=g, dtype=_t.float64)
    mix = s1 + s2
    est = s1 + 0.3 * s2 + noise
    return s1, s2, mix, est


def test_metrics_second_opinion_fast_bss_eval():
    """T3 指标实现 vs 独立 fast_bss_eval/解析式（SDR、SI-SDR、SDRi、SI-SDRi）。"""
    core = pytest.importorskip(
        "look2hear.metrics.core", reason="T3 尚未落地 look2hear/metrics/core.py"
    )
    import fast_bss_eval
    import torch as _t

    s1, _s2, mix, est = _metrics_fixture_signals()
    one = lambda x: x.reshape(1, 1, -1)  # noqa: E731

    # 独立参考：SDR 用 fast_bss_eval（与 T3 相同后端但本测试自行调用）；SI-SDR 用解析式
    sdr_ref = float(-fast_bss_eval.sdr_pit_loss(one(est), one(s1), filter_length=512).mean())
    sdr_mix_ref = float(-fast_bss_eval.sdr_pit_loss(one(mix), one(s1), filter_length=512).mean())
    sisnr_ref = _independent_si_sdr(est.numpy(), s1.numpy())
    sisnr_mix_ref = _independent_si_sdr(mix.numpy(), s1.numpy())

    assert core.sdr(est, s1) == pytest.approx(sdr_ref, abs=1e-4), "T3 SDR 与 fast_bss_eval 不一致"
    assert core.sdr_improvement(est, s1, mix) == pytest.approx(
        sdr_ref - sdr_mix_ref, abs=1e-4
    ), "T3 SDRi 与独立计算不一致"
    assert core.si_sdr(est, s1) == pytest.approx(sisnr_ref, abs=5e-3), "T3 SI-SDR 与解析式不一致"
    assert core.si_sdr_improvement(est, s1, mix) == pytest.approx(
        sisnr_ref - sisnr_mix_ref, abs=5e-3
    ), "T3 SI-SDRi 与独立计算不一致"

    # 符号约定：更优估计 → 更高分；估计=mixture 时提升量≈0
    assert core.si_sdr(s1 + 0.05 * _t.randn(16000, dtype=_t.float64), s1) > core.si_sdr(est, s1)
    assert core.si_sdr_improvement(mix, s1, mix) == pytest.approx(0.0, abs=1e-2)
    assert core.sdr_improvement(mix, s1, mix) == pytest.approx(0.0, abs=1e-2)


def test_metrics_tracker_csv_sign_and_final_rows(tmp_path):
    """T3 ALLMetricsTracker：CSV 逐句行/avg/std 行符号一致（越高越好），且与 core 一致。"""
    try:
        from look2hear.metrics.allwrapper import ALLMetricsTracker
    except Exception as exc:  # T3 未落地或仍在写入
        pytest.skip(f"ALLMetricsTracker 不可用（T3 进行中）: {exc}")
    import csv

    core = pytest.importorskip("look2hear.metrics.core")
    s1, _s2, mix, est = _metrics_fixture_signals()
    csv_path = tmp_path / "metrics.csv"
    tracker = ALLMetricsTracker(save_file=str(csv_path))
    row = tracker(mix, s1, est, "utt-0")
    tracker.final()

    assert row["sdr"] == pytest.approx(core.sdr(est, s1), abs=1e-6)
    assert row["si-snr"] == pytest.approx(core.si_sdr(est, s1), abs=1e-6)
    assert row["sdr"] > 0 and row["si-snr"] > 0
    assert row["sdr_i"] == pytest.approx(core.sdr_improvement(est, s1, mix), abs=1e-6)
    assert row["si-snr_i"] == pytest.approx(core.si_sdr_improvement(est, s1, mix), abs=1e-6)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["snt_id"] == "utt-0"
    avg, std = [r for r in rows if r["snt_id"] == "avg"][0], [r for r in rows if r["snt_id"] == "std"][0]
    for col in ("sdr", "sdr_i", "si-snr", "si-snr_i"):
        assert float(avg[col]) == pytest.approx(row[col], abs=1e-6), f"{col}: avg 行与逐句行符号/数值不一致"
        assert float(std[col]) == pytest.approx(0.0, abs=1e-6), f"{col}: 单句 std 应为 0"


# =============================================================== 8) 改动范围/语法守卫
ALLOWED_TRACKED = [
    "audio_train.py",
    "audio_visual_test.py",
    # T1 (authorized by Lead): comment-only semantics docs for logger/gpus.
    # Non-comment changes are rejected by test_config_tracked_changes_are_comment_only().
    "configs/Falcon.yml",
    "look2hear/system/*",
    "look2hear/metrics/*",
    "look2hear/datas/*",
    "look2hear/models/Falcon.py",
    "look2hear/models/__init__.py",
    "look2hear/videomodels/frcnn_videomodel.py",
]
ALLOWED_UNTRACKED = [
    "look2hear/metrics/core.py",
    "tools/repro/*",
    "docs/*",
    "AGENTS.md",
    "agents.md",
    "space/*",
]


def test_diff_scope_only_allowed_files():
    """Phase1 只允许改 T1–T5 约定文件；任何越界源码改动 = P0/P1。"""
    violations = []
    for p in _git_diff_names():
        if not any(fnmatch.fnmatch(p, pat) for pat in ALLOWED_TRACKED):
            violations.append(f"tracked modified: {p}")
    for p in _git_untracked_names():
        if not any(fnmatch.fnmatch(p, pat) for pat in ALLOWED_UNTRACKED):
            violations.append(f"untracked new: {p}")
    assert not violations, "发现越界改动（不在 task-9..15 写作用域）:\n" + "\n".join(violations)


def test_changed_python_files_compile():
    """所有被改动/新增的 .py 均可编译（捕获未定义语法/半写入文件）。"""
    files = [p for p in _git_diff_names() if p.endswith(".py")]
    files += [p for p in _git_untracked_names() if p.endswith(".py") and (REPO / p).is_file()]
    if not files:
        pytest.skip("当前无被改动的 Python 文件（Phase1 尚未落地）")
    r = _run([PY, "-m", "py_compile", *files], cwd=REPO, timeout=300)
    assert r.returncode == 0, f"py_compile 失败:\n{r.stderr[-4000:]}"


# =============================================================== 9) T1/T3/T5 验收不变量
def test_t1_audio_train_no_undefined_comet_logger():
    """T1 验收：audio_train.py 不得再以未定义名 `comet_logger` 作为 logger 实参。"""
    if not _landed("audio_train.py"):
        pytest.skip("T1 尚未落地 audio_train.py 改动")
    src = (REPO / "audio_train.py").read_text()
    assert "logger=comet_logger" not in src, "audio_train.py 仍引用未定义的 comet_logger"
    assert ("CSVLogger" in src) or ("TensorBoardLogger" in src), "T1 应提供 CSV/TB logger 构建"


def test_t3_metrics_core_available():
    """T3 验收：core 指标模块存在且导出 4 个正号指标函数；SDR/SI-SDR 均有限值。"""
    if not _landed("look2hear/metrics"):
        pytest.skip("T3 尚未落地 look2hear/metrics 改动")
    core = pytest.importorskip("look2hear.metrics.core")
    for name in ("si_sdr", "si_sdr_improvement", "sdr", "sdr_improvement"):
        assert callable(getattr(core, name, None)), f"core.{name} 缺失"
    s1, _s2, mix, est = _metrics_fixture_signals()
    for fn, args in ((core.si_sdr, (est, s1)), (core.sdr, (est, s1)),
                     (core.si_sdr_improvement, (est, s1, mix)),
                     (core.sdr_improvement, (est, s1, mix))):
        v = fn(*args)
        assert isinstance(v, float) and v == v and abs(v) < 200, f"{fn.__name__} 返回值异常: {v}"


def test_t5_frcnn_eval_returns_self_and_mdcl_alias():
    """T5 验收：FRCNNVideoModel.eval()/train() 返回 self；Falcon 具备 _mdcl_loss 别名。"""
    changed_video = _landed("look2hear/videomodels/frcnn_videomodel.py")
    changed_model = _landed("look2hear/models/Falcon.py")
    if not changed_video and not changed_model:
        pytest.skip("T5 尚未落地模型/视频骨干改动")
    sys.path.insert(0, str(REPO))
    if changed_video:
        from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel

        m = FRCNNVideoModel(pretrain=None)
        assert m.eval() is m, "FRCNNVideoModel.eval() 仍返回 None（train() 未 return self）"
        assert m.train(True) is not None
    if changed_model:
        from look2hear.models.Falcon import Falcon

        assert getattr(Falcon, "_mdcl_loss", None) is not None, "Falcon 仍缺 _mdcl_loss 别名"


def test_t5_models_dunder_all_consistent():
    """T5 验收：look2hear.models.__all__ 中每个名字都真实存在（修复 AVTIGER_causal 悬空）。"""
    if not _landed("look2hear/models/__init__.py"):
        pytest.skip("T5 尚未落地 models/__init__.py 改动")
    sys.path.insert(0, str(REPO))
    import look2hear.models as models

    missing = [n for n in getattr(models, "__all__", []) if not hasattr(models, n)]
    assert not missing, f"__all__ 含不存在的名字: {missing}"

# =============================================================== 10) T1: config 变更仅注释
def test_config_tracked_changes_are_comment_only():
    """`configs/Falcon.yml` 若有改动，只允许注释（键值必须与 main 完全一致）。"""
    cfg = "configs/Falcon.yml"
    if cfg not in _git_diff_names():
        pytest.skip("configs/Falcon.yml 未被改动")
    import yaml

    main_text = _run(["git", "show", f"main:{cfg}"]).stdout
    cur_text = (REPO / cfg).read_text()
    assert yaml.safe_load(main_text) == yaml.safe_load(cur_text), (
        "configs/Falcon.yml 出现键值变化：Phase1 只允许注释级改动（logger/gpus 语义说明）"
    )
    diff = _run(["git", "diff", "main", "--", cfg]).stdout
    bad = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if line[:1] in "+-":
            body = line[1:].strip()
            if body and not body.startswith("#"):
                bad.append(line)
    assert not bad, f"configs/Falcon.yml 存在非注释改动: {bad[:5]}"


# =============================================================== 11) T1: GPU spec 语义
def test_t1_gpu_spec_semantics(tmp_path):
    """T1 独立复核：`training.gpus` 语义（list=索引 / int=数量 / 0=cpu / -1=all / 其余负数报错）。"""
    if not _landed("audio_train.py"):
        pytest.skip("audio_train.py 未改动")
    sys.path.insert(0, str(REPO))
    import audio_train as at

    cases = [
        (None, ("none", None)), ("", ("none", None)), ("0,1", ("indices", [0, 1])),
        ([7], ("indices", [7])), ((3,), ("indices", [3])), ("2", ("indices", [2])),
        (2, ("count", 2)), (0, ("cpu", None)), (-1, ("all", None)),
    ]
    for value, expected in cases:
        assert at._normalize_gpus_spec(value) == expected, f"gpus={value!r} 语义不符"
    for bad in (-2, -5):
        with pytest.raises(ValueError):
            at._normalize_gpus_spec(bad)
    for bad in ([], [0, 0], [99]):
        with pytest.raises(ValueError):
            at.resolve_accelerator_and_devices({"gpus": bad})

    # configs/Falcon.yml 的 10 卡列表在 2 卡机上必须收敛到可见卡且仍用 DDP
    import yaml

    cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
    acc, devices, n, strategy = at.resolve_accelerator_and_devices(cfg["training"])
    import torch as _t

    if _t.cuda.is_available() and _t.cuda.device_count() >= 1:
        assert acc == "gpu"
        assert n <= _t.cuda.device_count()
        if n > 1:
            assert type(strategy).__name__ == "DDPStrategy"
    # CSVLogger 单层目录：<…>/checkpoint/<exp>/version_0（避免 <exp>/<exp>）
    logger = at.build_logger(cfg, str(tmp_path / "Experiments/checkpoint/Falcon"))
    assert logger.log_dir.endswith("Falcon/version_0"), logger.log_dir
    assert logger.log_dir.count("Falcon") == 1, logger.log_dir


# =============================================================== 12) LRS2-2Mix 真实数据 sanity
LRS2_S1 = Path("/root/tmp/lrs2_eval_s1/tt")
LRS2_S2_TARGET = Path("/root/tmp/lrs2_eval_s2/tt/target")
LRS2_LOGS = (Path("/root/tmp/lrs2_eval_s1.log"), Path("/root/tmp/lrs2_eval_s2.log"))


def _lrs2_available():
    return (LRS2_S1 / "mix.json").is_file() and (LRS2_S1 / "target").is_dir()


def _parse_final_metrics(text: str) -> dict:
    """解析 audio_visual_test.py 最终指标块（取最后一次出现，数值越高越好/正号）。"""
    keys = {"sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"}
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in keys:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def test_lrs2_mix_equals_s1_plus_s2():
    """LRS2-2Mix 真实样本：mix == target(s1) + target(s2)（独立验证 target=s1 约定）。"""
    if not _lrs2_available() or not LRS2_S2_TARGET.is_dir():
        pytest.skip("LRS2-2Mix 重建数据不在 /root/tmp，跳过")
    import json

    import numpy as np
    import soundfile as sf

    s1_dir = LRS2_S1 / "target"
    names = [Path(t[0]).name for t in json.loads((LRS2_S1 / "target.json").read_text())[:3]]
    checked = 0
    for name in names:
        mix, _ = sf.read(str(LRS2_S1 / "mix" / name))
        s1, _ = sf.read(str(s1_dir / name))
        s2, _ = sf.read(str(LRS2_S2_TARGET / name))
        n = min(len(mix), len(s1), len(s2))
        mix, s1, s2 = mix[:n], s1[:n], s2[:n]
        assert n > 0
        assert not np.array_equal(s1, s2), f"{name}: s1 与 s2 target 相同，不符合混音对约定"
        # 重建：mix = s1 + s2（float32/PCM16 量化误差量级）
        err = float(np.max(np.abs(mix - (s1 + s2))))
        assert err < 1e-3, f"{name}: mix != s1+s2, max|Δ|={err:.3e}"
        checked += 1
    assert checked == len(names) and checked > 0


def test_lrs2_eval_aggregate_matches_lead_evidence():
    """Lead 的 20 条 LRS2-2Mix 评测证据：target=s1 得 SI-SNRi 7.91/SDRi 8.55；target=s2 对照 4.60。"""
    if not all(p.is_file() for p in LRS2_LOGS):
        pytest.skip("Lead LRS2 评测日志不在 /root/tmp，跳过")
    s1 = _parse_final_metrics(LRS2_LOGS[0].read_text())
    s2 = _parse_final_metrics(LRS2_LOGS[1].read_text())
    for logs, tag, exp in ((s1, "target=s1", {"sdr": 14.309, "sdr_i": 8.547, "si-snr": 13.601,
                                              "si-snr_i": 7.911, "pesq": 2.312, "stoi": 0.923}),
                           (s2, "target=s2", {"si-snr_i": 4.603})):
        assert logs, f"{tag} 日志缺少最终指标块"
        for k, v in exp.items():
            assert k in logs, f"{tag} 缺少 {k}"
            assert abs(logs[k] - v) <= 5e-2, f"{tag} {k}={logs[k]} 期望≈{v}"
    assert s1["si-snr_i"] - s2["si-snr_i"] > 2.0, "target=s1 应显著优于 target=s2 对照（目标约定确认）"


def test_lrs2_eval_end_to_end_smoke(tmp_path):
    """端到端 sanity：官方权重 + LRS2-2Mix 真实数据，`audio_visual_test.py --limit 2`（CPU）出 CSV。"""
    if not _lrs2_available():
        pytest.skip("LRS2-2Mix 重建数据不在 /root/tmp，跳过")
    if not WEIGHTS.is_file():
        pytest.skip("官方权重缺失")
    import csv

    cmd = [
        PY, str(REPO / "audio_visual_test.py"),
        "--conf_dir", str(REPO / "configs/Falcon.yml"),
        "--data_dir", "/root/tmp/lrs2_eval_s1",
        "--ckpt_path", str(WEIGHTS),
        "--gpus", "cpu",
        "--limit", "2",
    ]
    r = _run(cmd, cwd=tmp_path, env={"PYTHONPATH": str(REPO)}, timeout=900)
    assert r.returncode == 0, f"LRS2 端到端评测失败:\nSTDOUT={r.stdout[-2500:]}\nSTDERR={r.stderr[-2500:]}"
    csv_path = tmp_path / "Experiments/checkpoint/Falcon/results/metrics_combined.csv"
    assert csv_path.is_file(), f"未生成 metrics_combined.csv: {r.stdout[-1500:]}"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    utt = [r_ for r_ in rows if r_["snt_id"] not in ("avg", "std")]
    assert len(utt) == 2, f"期望 2 条逐句结果，得到 {len(utt)}"
    assert any(r_["snt_id"] == "avg" for r_ in rows)
    import math

    for row in utt:
        for col in ("sdr", "sdr_i", "si-snr", "si-snr_i"):
            v = float(row[col])
            assert math.isfinite(v), f"{col} 非有限值: {v}"
        # target=s1 约定下，官方权重应有正的 SDR/SI-SNR 与提升
        assert float(row["si-snr"]) > 0 and float(row["sdr"]) > 3.0
        assert float(row["si-snr_i"]) > 0 and float(row["sdr_i"]) > 0
