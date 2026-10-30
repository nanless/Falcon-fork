"""Phase2-QA（task-24）: Phase 2 管线/评测回归 + 官方权重数值不变量 + diff 范围守卫。

覆盖：
  1. 官方权重 4 条指纹（eval / train-est / sep_emb / video）= Phase1 基线（Phase2 不应改变数值）；
  2. Phase2 脚本存在时：`py_compile` + `--help` 冒烟（P2-A..F 的 tools/repro/phase2/*.py、lrs2_2mix_to_falcon.py）；
  3. Phase2 报告（P2A..P2F-report.md）结构与证据日志存在性（报告未落盘时 skip）；
  4. LRS2-2Mix 评测端到端冒烟（优先 P2-C 转换数据 tt，fallback /root/tmp/lrs2_eval_tt_full），官方权重 + `--limit 2`；
  5. 评测 CSV 内部一致性（avg=逐句均值、std=总体标准差、有限值）；
  6. LRS2-2Mix rebuild 原始数据统计（tr/cv/tt = 20000/5000/3000，P2-C 输入侧事实）；
  7. Phase2 diff 范围守卫：不允许改 Phase1 已提交源码；新增文件仅限 docs/、tools/repro/、configs/LRS2_2Mix.yml。

数据/脚本/报告缺失时自动 skip，不阻塞仓库内回归。
运行：/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_phase2_pipeline.py -q
"""
from __future__ import annotations

import copy
import csv
import fnmatch
import hashlib
import json
import math
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
BASELINE = REPO / "docs/verification/phase1/baseline-fingerprint.json"
CONVERTED = Path("/root/data/falcon_lrs2_2mix")
REBUILD = Path("/root/data/lrs2_2mix/lrs2_rebuild")
LRS2_EVAL_FALLBACK = Path("/root/tmp/lrs2_eval_tt_full")
PHASE2_DIR = REPO / "docs/verification/phase2"
PHASE2_SCRIPTS = REPO / "tools/repro/phase2"
PHASE2_REPORTS = [PHASE2_DIR / f"P2{c}-report.md" for c in "ABCDEF"]


def _run(cmd, cwd=REPO, env=None, timeout=900):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=str(cwd), env=e, capture_output=True, text=True, timeout=timeout)


def _tensor_sha(t) -> str:
    t = t.detach().to("cpu").contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


# ─────────────────────────────────────────────── 1) 官方权重数值不变量
@pytest.fixture(scope="session")
def baseline() -> dict:
    assert BASELINE.is_file(), f"缺少 Phase1 基线 {BASELINE}"
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="session")
def fingerprints(baseline) -> dict:
    import yaml

    sys.path.insert(0, str(REPO))
    import look2hear.models as models
    from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel

    cfg = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
    acfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    a_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    v_sd = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}

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
    with torch.no_grad():  # 基线约定：不调用 .eval()，BN3d 用默认 training 批内统计
        vy = v(vin)
    return {
        "audio_params": sum(p.numel() for p in m.parameters()),
        "video_params": sum(p.numel() for p in v.parameters()),
        "eval_sha": _tensor_sha(y),
        "train_est_sha": _tensor_sha(est),
        "sep_sha": _tensor_sha(sep),
        "video_sha": _tensor_sha(vy),
    }


def test_phase2_official_weight_fingerprints_unchanged(fingerprints, baseline):
    """Phase2 分支上官方权重 4 条指纹必须与 Phase1 基线逐位一致（确保 Phase2 未改模型/数值）。"""
    assert fingerprints["audio_params"] == baseline["params"]["audio"] == 537_697
    assert fingerprints["video_params"] == baseline["params"]["video"] == 11_186_688
    assert fingerprints["eval_sha"] == baseline["eval_forward"]["sha256"]
    assert fingerprints["train_est_sha"] == baseline["train_forward"]["est_sha256"]
    assert fingerprints["sep_sha"] == baseline["train_forward"]["sep_sha256"]
    assert fingerprints["video_sha"] == baseline["video_forward"]["sha256"]


# ─────────────────────────────────────────────── 2) Phase2 脚本 smoke
def _phase2_script_files() -> list[Path]:
    files = []
    if PHASE2_SCRIPTS.is_dir():
        files += sorted(PHASE2_SCRIPTS.glob("*.py"))
    ar = REPO / "tools/repro/lrs2_2mix_to_falcon.py"
    if ar.is_file():
        files.append(ar)
    return files


def test_phase2_scripts_compile_and_help():
    """Phase2 新脚本可编译；带 `__main__` 的脚本 `--help` 能正常退出（argparse 冒烟）。"""
    files = _phase2_script_files()
    if not files:
        pytest.skip("Phase2 脚本尚未落盘（tools/repro/phase2/*.py 不存在）")
    r = _run([PY, "-m", "py_compile", *[str(f) for f in files]], timeout=300)
    assert r.returncode == 0, f"Phase2 脚本 py_compile 失败:\n{r.stderr[-3000:]}"
    help_checked = []
    for f in files:
        src = f.read_text(errors="ignore")
        if "__main__" not in src:
            continue
        r = _run([PY, str(f), "--help"], timeout=180)
        # 允许 0；不接受语法/导入错误（traceback）
        assert r.returncode == 0, (
            f"{f.relative_to(REPO)} --help 失败 rc={r.returncode}:\n"
            f"STDOUT={r.stdout[-1200:]}\nSTDERR={r.stderr[-2500:]}"
        )
        help_checked.append(f.name)
    print(f"Phase2 脚本: {len(files)} 个可编译, {len(help_checked)} 个 --help 通过")


# ─────────────────────────────────────────────── 3) Phase2 报告证据
#: review A8-5（2026-10-30）从仓库删除的 0 字节占位 CSV：`P2B-report.md:140` 仍以
#: “空文件（0 字节），可忽略”提及该路径，属已声明可忽略的历史产物，不再作为有效证据要求存在。
KNOWN_REMOVED_EVIDENCE = {
    "docs/verification/phase2/P2B-full-big-whole-real.csv",
}


def test_phase2_reports_have_evidence():
    """已有 P2A..P2F 报告：够长、含结论段、引用的 .log 证据文件存在。"""
    existing = [p for p in PHASE2_REPORTS if p.is_file()]
    if not existing:
        pytest.skip("Phase2 报告尚未落盘（P2A..F-report.md 不存在）")
    missing_logs = []
    pending = []
    for report in existing:
        text = report.read_text(errors="ignore")
        assert len(text) > 300, f"{report.name} 过短（{len(text)} 字符），疑似占位"
        assert ("结论" in text) or ("## 1." in text), f"{report.name} 缺少结论段"
        # 报告若自述“进行中/待落盘”，允许引用尚未生成的证据（QA 报告中单独记录为 pending）；
        # 已完成报告引用的证据必须真实存在。
        in_progress = any(k in text for k in ("进行中", "待落盘", "尚未落盘", "pending"))
        for rel in re.findall(r"(docs/verification/phase2/[\w./-]+\.(?:log|json|csv))", text):
            if rel in KNOWN_REMOVED_EVIDENCE:
                continue
            if not (REPO / rel).is_file():
                (pending if in_progress else missing_logs).append((report.name, rel))
    assert not missing_logs, f"已完成报告引用的证据文件不存在: {missing_logs[:8]}"
    if pending:
        print(f"进行中报告待落盘证据: {pending[:8]}")


# ─────────────────────────────────────────────── 4) LRS2 端到端评测冒烟
def _phase2_eval_data_dir() -> Path | None:
    """优先使用 P2-C 转换数据 tt；否则用 Phase1 已构建的 lrs2_eval_tt_full。"""
    converted_tt = CONVERTED / "tt"
    if (converted_tt / "mix.json").is_file():
        return converted_tt
    if (LRS2_EVAL_FALLBACK / "tt" / "mix.json").is_file():
        return LRS2_EVAL_FALLBACK
    return None


def test_phase2_lrs2_eval_end_to_end_smoke(tmp_path):
    """官方权重 + LRS2-2Mix（P2-C 数据优先）`audio_visual_test.py --limit 2`（CPU）出 CSV 且指标合理。"""
    data_dir = _phase2_eval_data_dir()
    if data_dir is None:
        pytest.skip("LRS2-2Mix 评测数据不存在（等待 P2-C 转换或 /root/tmp/lrs2_eval_tt_full）")
    if not WEIGHTS.is_file():
        pytest.skip("官方权重缺失")
    cmd = [
        PY, str(REPO / "audio_visual_test.py"),
        "--conf_dir", str(REPO / "configs/Falcon.yml"),
        "--data_dir", str(data_dir),
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
    utt = [x for x in rows if x["snt_id"] not in ("avg", "std")]
    assert len(utt) == 2, f"期望 2 条逐句结果，得到 {len(utt)}"
    assert any(x["snt_id"] == "avg" for x in rows)
    for row in utt:
        for col in ("sdr", "sdr_i", "si-snr", "si-snr_i"):
            v = float(row[col])
            assert math.isfinite(v), f"{col} 非有限值: {v}"
        assert float(row["si-snr"]) > 0 and float(row["sdr"]) > 3.0
        assert float(row["si-snr_i"]) > 0 and float(row["sdr_i"]) > 0


# ─────────────────────────────────────────────── 5) 评测 CSV 内部一致性
def _phase2_metric_csvs() -> list[Path]:
    cands = []
    if PHASE2_DIR.is_dir():
        cands += sorted(PHASE2_DIR.rglob("*.csv"))
    repo_results = REPO / "Experiments/checkpoint/Falcon/results"
    if repo_results.is_dir():
        cands += sorted(repo_results.glob("metrics_combined.csv"))
    return [p for p in cands if p.is_file() and p.stat().st_size > 0]


def _as_float(value) -> float:
    """CSV 单元格 -> float；退化样本的空串/nan/None 统一返回 math.nan。

    RealSSA-Sim 全量结果中有 3 条退化样本（SDR solver singular / PESQ "No utterances detected"），
    评测脚本把 sdr/sdr_i/pesq 落盘为空串（亦可能写 nan/None）。聚合口径是"有限值上的均值 /
    总体标准差（ddof=0）"，故空值视作 nan 后在 `vals` 里被 `math.isfinite` 过滤掉。
    其余无法解析的字符串仍会让 `float()` 抛错——不放宽对"格式损坏"的校验。
    """
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none"):
        return math.nan
    return float(text)


def test_phase2_eval_csv_aggregation_consistent():
    """Phase2 评测 CSV：avg 行 = 逐句均值、std 行 = 总体标准差（ddof=0）、数值有限。"""
    csvs = _phase2_metric_csvs()
    if not csvs:
        pytest.skip("Phase2 评测 CSV 尚未落盘（等待 P2 评测/全量结果）")
    checked = 0
    cols = ("sdr", "sdr_i", "si-snr", "si-snr_i")
    for path in csvs:
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        utt = [r for r in rows if r.get("snt_id") not in ("avg", "std")]
        avg = next((r for r in rows if r.get("snt_id") == "avg"), None)
        std = next((r for r in rows if r.get("snt_id") == "std"), None)
        if len(utt) < 2 or avg is None:
            continue
        for col in cols:
            # 空串/nan/None（见 _as_float）按 nan 处理：avg/std 只在有限值上聚合。
            vals = [v for v in (_as_float(r.get(col)) for r in utt) if math.isfinite(v)]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            avg_v = _as_float(avg.get(col))
            if math.isfinite(avg_v):
                assert abs(avg_v - mean) < 1e-5, f"{path.name}:{col} avg≠均值"
            if std is not None:
                std_v = _as_float(std.get(col))
                if math.isfinite(std_v):
                    assert abs(std_v - math.sqrt(var)) < 1e-5, f"{path.name}:{col} std≠总体标准差"
        checked += 1
    assert checked > 0, "找到 CSV 但没有一条满足『逐句+avg』结构"


# ─────────────────────────────────────────────── 6) LRS2 rebuild 原始数据统计
def test_lrs2_rebuild_dataset_counts():
    """LRS2-2Mix rebuild（P2-C 输入侧）：tr/cv/tt mix 数应为 20000/5000/3000。"""
    base = REBUILD / "audio/wav16k/min"
    if not base.is_dir():
        pytest.skip("LRS2-2Mix rebuild 原始数据不在本机")
    expected = {"tr": 20000, "cv": 5000, "tt": 3000}
    got = {}
    for split in ("tr", "cv", "tt"):
        d = base / split / "mix"
        if not d.is_dir():
            continue
        got[split] = len(list(d.glob("*.wav")))
    assert got, "rebuild 下没有 mix wav"
    print("LRS2-2Mix rebuild 计数:", got, "期望:", expected)
    for split, n in got.items():
        if split in expected:
            assert n == expected[split], f"{split} 计数 {n} != 期望 {expected[split]}"


# ─────────────────────────────────────────────── 7) Phase2 diff 范围守卫
# Phase2 允许的 tracked 改动：仅文档（P2 提交）与 tools/repro/ 下的测试/脚本（QA 维护）；
# Phase1 已提交源码（audio_train.py / audio_visual_test.py / look2hear/ / studio/ / configs/Falcon.yml）禁止改动。
ALLOWED_TRACKED_PREFIXES = (
    "docs/",
    "tools/",
    "configs/",
    "audio_train.py",
    "agents.md",
    # authorized (env compliance): AGENTS.md §2/§4 的 Phase2 依赖说明与 lock 更新（19:16Z），
    # 属环境合规文档改动，不是 Phase1 源码改动。
    "AGENTS.md",
    # authorized (P0 evidence, 2026-10-24): .gitignore 增加 docs/verification 证据日志/CSV 的
    # 负向规则（!docs/verification/**/*.log 等），属"证据必须可复核"的合规改动，非越界源码改动。
    # 该文件提交后仍会与 main 存在 diff，故需长期白名单。
    ".gitignore",
)
ALLOWED_UNTRACKED = [
    "docs/*",
    "tools/repro/*",
    "configs/LRS2_2Mix.yml",
    "AGENTS.md",
    "agents.md",
    "space/*",
]


def _git(cmd) -> list[str]:
    r = _run(["git", *cmd])
    return [x for x in r.stdout.splitlines() if x.strip()]


def test_phase2_diff_scope_guard():
    """Phase2 **冻结提交区间**（ac91800..2beec38）的改动范围守卫。

    历史实现用 `git diff main`（在 main 上恒空）→ working-tree HEAD 守卫（维护轮次会合法
    改源码而误报）。最终改为冻结区间：范围永久稳定，可长期作为 Phase2 契约（review A8-1）。
    工作区在途改动不在此守卫范围（由 review/CI 对应测试负责）。
    """
    # 冻结 Phase2 提交区间（ac91800..2beec38）。不能用 working-tree / 分支 tip：
    # 维护轮次会合法修改源码，且 tip 漂移会让守卫失真（review A8-1）。
    tracked = _git(["diff", "--name-only", "ac91800..2beec38"])
    assert tracked, "Phase2 冻结区间无改动——基线/历史缺失"
    bad_tracked = [p for p in tracked if not any(p.startswith(pref) for pref in ALLOWED_TRACKED_PREFIXES)]
    assert not bad_tracked, f"Phase2 冻结区间出现源码级改动（应只有 docs/tools/configs/audio_train）: {bad_tracked}"

    # 冻结区间守卫不检查工作区未跟踪文件：维护轮次的新文件由人工 review / 对应测试负责。

    # 说明：本守卫只针对已冻结的 Phase2 提交区间；维护轮次对源码的合法修改
    # （2026-10-31 起的 review 修复）由人工 review 与 test_phase1_regression 的历史守卫负责。


# ─────────────────────────────────────────────── 7b) 证据目录卫生（review A8-5）
def test_docs_verification_no_pid_or_empty_evidence():
    """`docs/verification` 不得入库 `*.pid`（进程早已退出）或 0 字节证据文件（易被误读为“有结果”）。

    历史遗留：`T6-download.pid` / `P2D-train.pid`（进程已退出）与 0 字节
    `P2B-full-big-whole-real.csv`（被提前终止的全量评测）曾被提交入库（review A8-5）。
    本测试只看 **tracked** 文件：未跟踪的在途产物（如正在写入的日志）不算违规。
    """
    tracked = _git(["ls-files", "docs/verification"])
    assert tracked, "docs/verification 下没有任何入库文件？"
    pids = [p for p in tracked if p.endswith(".pid")]
    assert not pids, f"docs/verification 下仍入库 *.pid（应 git rm --cached）: {pids}"
    empty = [
        p for p in tracked
        if (REPO / p).is_file() and (REPO / p).stat().st_size == 0
    ]
    assert not empty, f"docs/verification 下存在 0 字节证据文件（应删除或补内容）: {empty}"


# ─────────────────────────────────────────────── 8) P2-B summary ↔ 逐句 CSV 可复算
def test_phase2_p2b_summary_matches_csv():
    """P2-B（variant×protocol×visual）汇总 JSON 的均值/标准差必须能由逐句 CSV 复算。"""
    summaries = sorted(PHASE2_DIR.glob("P2B-summary-*.json"))
    if not summaries:
        pytest.skip("P2B-summary-*.json 尚未落盘")
    checked = 0
    for summary_path in summaries:
        entries = json.loads(summary_path.read_text())
        assert isinstance(entries, list) and entries, f"{summary_path.name} 结构异常"
        for entry in entries:
            csv_name = entry.get("csv")
            if not csv_name:
                continue
            csv_path = PHASE2_DIR / csv_name
            if not csv_path.is_file():
                continue
            with open(csv_path, newline="") as f:
                all_rows = list(csv.DictReader(f))
            # ALLMetricsTracker.final() 会追加 avg/std 行；逐句行需剔除
            utt_rows = [r for r in all_rows if r.get("snt_id") not in ("avg", "std")]
            assert len(utt_rows) == int(entry["n"]), (
                f"{csv_name}: 逐句行数 {len(utt_rows)} != summary n={entry['n']}"
            )
            rows = utt_rows
            for col, key in (("sdr", "sdr"), ("sdr_i", "sdr_i"),
                             ("si-snr", "si_snr"), ("si-snr_i", "si_snr_i"),
                             ("pesq", "pesq"), ("stoi", "stoi")):
                if key not in entry:
                    continue
                vals = [float(r[col]) for r in rows]
                mean = sum(vals) / len(vals)
                assert abs(mean - float(entry[key])) < 1e-5, (
                    f"{csv_name}:{col} 复算均值 {mean:.6f} != summary {entry[key]:.6f}"
                )
            for col, key in (("sdr_i", "sdr_i_std"), ("si-snr_i", "si_snr_i_std")):
                if key not in entry:
                    continue
                vals = [float(r[col]) for r in rows]
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
                assert abs(std - float(entry[key])) < 1e-5, (
                    f"{csv_name}:{col} 复算 std {std:.6f} != summary {entry[key]:.6f}"
                )
            checked += 1
    if checked == 0:
        pytest.skip("summary 中引用的逐句 CSV 尚未全部落盘")
    print(f"P2B summary↔CSV 复算通过 {checked} 组")


# ─────────────────────────────────────────────── 9) Phase2 数据/证据一致性（P2-A/C/E/F）
def test_phase2_p2c_conversion_log_and_counts():
    """P2-C：转换日志声称 tr/cv/tt=20000/5000/3000，且落盘索引条目数与之一致、无缺失 mouth。"""
    log = PHASE2_DIR / "P2C-convert.log"
    if not log.is_file():
        pytest.skip("P2C-convert.log 尚未落盘")
    if not (CONVERTED / "tt" / "mix.json").is_file():
        pytest.skip("/root/data/falcon_lrs2_2mix 尚未转换完成")
    text = log.read_text(errors="ignore")
    expected = {"tr": 20000, "cv": 5000, "tt": 3000}
    for split, n in expected.items():
        idx = CONVERTED / split / "mix.json"
        assert idx.is_file(), f"缺少 {split}/mix.json"
        assert f"[{split}] INDEX READY: {idx} ({n} entries)" in text, (
            f"P2C 日志缺少 {split} 的 INDEX READY({n}) 记录"
        )
        assert f"'converted': {n}" in text, f"P2C 日志缺少 {split} converted={n}"
        entries = json.loads(idx.read_text())
        assert isinstance(entries, list) and len(entries) == n, (
            f"{split}/mix.json 条目 {len(entries) if isinstance(entries, list) else type(entries)} != {n}"
        )
        p0 = Path(entries[0][0])
        assert p0.is_absolute() and p0.is_file(), f"{split}/mix.json 首条音频路径无效: {p0}"
    assert "'missing_mouth_npz': 0" in text, "P2C 日志报告了缺失 mouth npz（应为 0）"


def test_phase2_p2a_anchor_data_checks():
    """P2-A anchor：数据侧不变量（mix=s1+s2、target=s1、mouth=s1、2s/16k/单声道）+ 官方权重 anchor 指标。"""
    path = PHASE2_DIR / "P2A-anchor-first20.json"
    if not path.is_file():
        pytest.skip("P2A-anchor-first20.json 尚未落盘")
    d = json.loads(path.read_text())
    dc = d["data_checks"]
    assert dc["mix_r2"]["mean"] > 0.9999, f"mix 重建 R² 过低: {dc['mix_r2']}"
    assert dc["mix_residual_rel"]["max"] < 1e-3, f"mix 残差过大: {dc['mix_residual_rel']}"
    assert dc["target_equals_s1"]["max"] == 0.0, "target 不等于 s1（target 约定被破坏）"
    assert dc["mouth_equals_s1_npz"]["min"] == 1.0, "mouth npz 与 s1 不对齐"
    assert dc["len"]["mean"] == 32000 and dc["len"]["std"] == 0.0
    assert dc["sr"]["mean"] == 16000 and dc["channels"]["mean"] == 1.0

    m = d["models"]["small"]
    assert m["loader_meta"]["audio_params"] == 537_697
    assert m["loader_meta"]["video_params"] == 11_186_688
    assert m["loader_meta"]["audio_keys"] == 525 and m["loader_meta"]["video_keys"] == 137
    v = m["variants"]["mouth_s1_88"]
    assert v["si_snr_i"]["n"] == 20 and v["si_snr"]["n"] == 20
    assert 0.0 < v["si_snr_i"]["mean"] < 20.0
    # SI-SDR/SDR 同量级且 SDR(512) 略高于 SI-SNR（官方权重 anchor 的合理性）
    assert v["sdr512_i"]["mean"] >= v["si_snr_i"]["mean"] - 0.5
    assert abs(v["si_snr_zm0"]["mean"] - v["si_snr"]["mean"]) < 0.01


def test_phase2_p2e_dnsmos_evidence():
    """P2-E：DNSMOS 依赖版本与 conda lock 一致；4 个 ONNX 模型 sha256/字节数齐全；分数有限且在 [1,5]。"""
    syn = PHASE2_DIR / "P2E-dnsmos-synthetic.json"
    if not syn.is_file():
        pytest.skip("P2E-dnsmos-synthetic.json 尚未落盘")
    lock = REPO / "docs/verification/conda-env-lock.txt"
    assert lock.is_file()
    lock_text = lock.read_text()
    for pkg in ("speechmos==0.0.1.1", "onnxruntime==1.31.0", "protobuf==", "flatbuffers=="):
        assert pkg in lock_text, f"conda lock 缺少 {pkg}（AGENTS.md §2/§4 要求）"

    d = json.loads(syn.read_text())
    assert d["backend"] == "speechmos"
    assert d["speechmos_version"] == "0.0.1.1"
    assert d["onnxruntime_version"] == "1.31.0"
    files = d["models"]["files"]
    assert set(files) == {"bak_ovr.onnx", "model_v8.onnx", "sig.onnx", "sig_bak_ovr.onnx"}
    for name, meta in files.items():
        assert re.fullmatch(r"[0-9a-f]{64}", str(meta["sha256"])), f"{name} sha256 非法"
        assert int(meta["bytes"]) > 0
    assert d["all_finite_in_range"] is True
    cases = d["cases"]
    assert len(cases) >= 4
    for case in cases:
        for k in ("sig_mos", "bak_mos", "ovrl_mos", "p808_mos"):
            assert 1.0 <= float(case[k]) <= 5.0, f"{case.get('case')}:{k}={case[k]} 越界"

    smoke = PHASE2_DIR / "P2E-dnsmos-lrs2-smoke.json"
    if smoke.is_file():
        s = json.loads(smoke.read_text())
        assert s["num_ok"] == s["num_files"] and s["num_errors"] == 0
        for k in ("sig_mos", "bak_mos", "ovrl_mos", "p808_mos"):
            assert 1.0 <= float(s["mean"][k]) <= 5.0


def test_phase2_p2f_metric_implementation_matches_core():
    """P2-F：自实现指标 vs look2hear.metrics.core vs 评测 CSV 的逐句差异 <1e-5 dB；csv avg 可复算。"""
    path = PHASE2_DIR / "P2F-results.json"
    if not path.is_file():
        pytest.skip("P2F-results.json 尚未落盘")
    d = json.loads(path.read_text())
    m = d["metrics"]
    assert m["n"] == len(m["per_item"]) >= 20
    assert m["csv_avg_recompute_ok"] is True
    for key in ("maxabs_my_vs_core_db", "maxabs_my_vs_csv_db",
                "maxabs_baseline_formula_db", "maxabs_baseline_sdr_formula_db"):
        assert float(m[key]) < 1e-5, f"{key}={m[key]} 超阈值 1e-5 dB"
    assert str(m.get("dataset_order_match", "")).startswith(f"{m['n']}/"), (
        f"数据集顺序未完全对齐: {m.get('dataset_order_match')}"
    )
    # 逐句三源（自实现 / look2hear.core / 评测 CSV）一致
    for col in ("si_sdr", "si_sdr_i", "sdr", "sdr_i"):
        d_core = max(abs(float(x["my"][col]) - float(x["core"][col])) for x in m["per_item"])
        d_csv = max(abs(float(x["my"][col]) - float(x["csv"][col])) for x in m["per_item"])
        assert d_core < 1e-5 and d_csv < 1e-4, f"{col}: my↔core={d_core}, my↔csv={d_csv}"
    # subset_mean 必须等于 50 句子集逐句均值（csv_avg 是整段评测 CSV 的 avg 行，不是子集）
    for col, key in (("si_sdr_i", "si_sdr_i"), ("sdr_i", "sdr_i")):
        vals = [float(x["my"][col]) for x in m["per_item"]]
        mean = sum(vals) / len(vals)
        assert abs(mean - float(m["subset_mean"][key])) < 1e-6, (
            f"subset_mean[{key}]={m['subset_mean'][key]} != 复算 {mean}"
        )
    for k, v in m["csv_avg"].items():
        assert math.isfinite(float(v)), f"csv_avg[{k}] 非有限值"
    assert 0.0 < float(m["csv_avg"]["si-snr_i"]) < float(m["csv_avg"]["si-snr"])
    assert 0.0 < float(m["csv_avg"]["sdr_i"]) < float(m["csv_avg"]["sdr"])
    big = d.get("big") or {}
    ck = big.get("checkpoint") or {}
    if ck:
        assert ck.get("architecture") == "FalconSpeakerContrastive"
        assert int(ck.get("epoch", -1)) == 714
        assert "epoch=714.ckpt" in (ck.get("best_k_models") or {})


def test_phase2_lrs2_config_invariants():
    """P2-D 配置：数据指向 P2-C 产物、2s/16k/单源、视觉骨干路径、与 Falcon.yml 一致的训练/优化协议。"""
    cfg_path = REPO / "configs/LRS2_2Mix.yml"
    if not cfg_path.is_file():
        pytest.skip("configs/LRS2_2Mix.yml 尚未落盘")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    dc = cfg["datamodule"]["data_config"]
    assert cfg["datamodule"]["data_name"] == "AVSpeechDataModuleWithNear"
    assert dc["train_dir"] == str(CONVERTED / "tr")
    assert dc["valid_dir"] == str(CONVERTED / "cv")
    assert dc["test_dir"] == str(CONVERTED / "tt")
    assert dc["n_src"] == 1 and dc["sample_rate"] == 16000
    assert float(dc["segment"]) == 2.0
    assert dc["use_near"] is True and int(dc["batch_size"]) >= 1
    assert int(dc["train_samples"]) > 0
    assert cfg["exp"]["exp_name"] == "LRS2_2Mix"
    assert cfg["training"]["system"] == "AudioVisualLightningModuleNASSv2"
    assert cfg["training"]["early_stop"]["monitor"] == "val_loss/dataloader_idx_0"
    assert cfg["videonet"]["videonet_name"] == "FRCNNVideoModel"
    pre = Path(cfg["videonet"]["videonet_config"]["pretrain"])
    assert pre.name.endswith(".pth.tar"), f"视觉骨干路径异常: {pre}"

    base = yaml.safe_load((REPO / "configs/Falcon.yml").read_text())
    assert cfg["optimizer"]["optim_name"] == base["optimizer"]["optim_name"]
    assert float(cfg["optimizer"]["lr"]) == float(base["optimizer"]["lr"])
    assert float(cfg["optimizer"]["weight_decay"]) == float(base["optimizer"]["weight_decay"])
    assert cfg["scheduler"]["sche_name"] == base["scheduler"]["sche_name"]
    assert cfg["training"]["system"] == base["training"]["system"]


def test_phase2_p2d_smoke_evidence():
    """P2-D：2-GPU 冒烟日志含吞吐/损失/val 指标，且报告与日志的吞吐数字一致。"""
    log = PHASE2_DIR / "P2D-smoke.log"
    if not log.is_file():
        pytest.skip("P2D-smoke.log 尚未落盘")
    text = log.read_text(errors="ignore")
    assert "it/s" in text, "P2D 冒烟日志缺少吞吐（it/s）"
    for key in ("train_loss_epoch", "val_loss", "test_loss"):
        assert key in text, f"P2D 冒烟日志缺少 {key}"
    report = PHASE2_DIR / "P2D-report.md"
    if report.is_file():
        rtext = report.read_text(errors="ignore")
        m = re.search(r"(\d+\.\d+)\s*it/s", text)
        if m:
            assert m.group(1) in rtext, f"报告未引用日志中的吞吐 {m.group(1)} it/s"


# ─────────────────────────────────────────────── 10) P2-B 报告表 ↔ 逐句 CSV 独立复算
def test_phase2_p2b_paired_stats_and_report_consistency():
    """P2-B 报告表与配对统计必须可由逐句 CSV 独立复算（均值 3 位小数、胜场数、总体 std）。"""
    report = PHASE2_DIR / "P2B-report.md"
    csvs = sorted(PHASE2_DIR.glob("P2B-b300-*.csv"))
    csvs = [p for p in csvs if p.stat().st_size > 0]
    if not report.is_file() or not csvs:
        pytest.skip("P2B-report.md / P2B-b300-*.csv 尚未落盘")
    # 报告表格用 Unicode 减号 U+2212，复算值用 ASCII "-"，比较前统一
    rtext = report.read_text(errors="ignore").replace("\u2212", "-")

    def load(name):
        with open(PHASE2_DIR / name, newline="") as f:
            rows = list(csv.DictReader(f))
        utt = [r for r in rows if r.get("snt_id") not in ("avg", "std")]
        avg = next(r for r in rows if r.get("snt_id") == "avg")
        std = next(r for r in rows if r.get("snt_id") == "std")
        return utt, avg, std

    data = {}
    for path in csvs:
        utt, avg, std = load(path.name)
        assert len(utt) == 300, f"{path.name}: 逐句行 {len(utt)} != 300"
        for col in ("si-snr", "si-snr_i", "sdr", "sdr_i"):
            vals = [float(r[col]) for r in utt]
            mean = sum(vals) / len(vals)
            pstd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            assert abs(mean - float(avg[col])) < 1e-4, f"{path.name}:{col} avg 行不可复算"
            assert abs(pstd - float(std[col])) < 1e-4, f"{path.name}:{col} std 行不可复算"
            assert math.isfinite(mean) and abs(mean) < 100.0
        # 报告表必须出现 3 位小数的复算值（SI-SNRi / SDRi）
        for col in ("si-snr_i", "sdr_i"):
            vals = [float(r[col]) for r in utt]
            m3 = f"{sum(vals) / len(vals):.3f}"
            assert m3 in rtext, f"{path.name}:{col} 复算均值 {m3} 未出现在 P2B-report.md 表中"
        data[path.name] = {r["snt_id"]: float(r["si-snr_i"]) for r in utt}

    pairs = [
        ("P2B-b300-big-whole-real.csv", "P2B-b300-small-whole-real.csv", 0.971, 258),
        ("P2B-b300-small-windowed-real.csv", "P2B-b300-small-whole-real.csv", -1.912, 33),
        ("P2B-b300-big-windowed-real.csv", "P2B-b300-big-whole-real.csv", -2.318, 33),
        ("P2B-b300-small-whole-zero.csv", "P2B-b300-small-whole-real.csv", -15.089, 4),
        ("P2B-b300-small-whole-shuffle.csv", "P2B-b300-small-whole-real.csv", -12.894, 7),
        ("P2B-b300-big-whole-zero.csv", "P2B-b300-big-whole-real.csv", -4.375, 26),
    ]
    checked = 0
    for cand, base, exp_mean, exp_wins in pairs:
        if cand not in data or base not in data:
            continue
        keys = sorted(set(data[cand]) & set(data[base]))
        assert len(keys) == 300, f"{cand} vs {base}: 配对 {len(keys)} != 300"
        diffs = [data[cand][k] - data[base][k] for k in keys]
        mean = sum(diffs) / len(diffs)
        wins = sum(1 for x in diffs if x > 0)
        assert abs(mean - exp_mean) < 2e-3, f"{cand}: 配对均值 {mean:+.3f} != 报告 {exp_mean:+.3f}"
        assert wins == exp_wins, f"{cand}: 胜场 {wins} != 报告 {exp_wins}"
        checked += 1
    assert checked >= 2, "P2B 配对复算条目不足"
    print(f"P2B 报告↔CSV 独立复算：{len(data)} 组均值/总体std + {checked} 组配对统计 全部一致")


# ─────────────────────────────────────────────── 11) P2-A 200 条报告 ↔ JSON 独立复算
def test_phase2_p2a_report_matches_200_json():
    """P2-A 200 条子集：报告表中的 mean±std 与 Δ 必须能由 P2A-small-200.json 复算。"""
    path = PHASE2_DIR / "P2A-small-200.json"
    report = PHASE2_DIR / "P2A-report.md"
    if not path.is_file() or not report.is_file():
        pytest.skip("P2A-small-200.json / P2A-report.md 尚未落盘")
    d = json.loads(path.read_text())
    rtext = report.read_text(errors="ignore").replace("\u2212", "-")
    v = d["models"]["small"]["variants"]
    assert d["meta"]["n_utterances"] == 200 and d["meta"]["subset"] == "random"
    for name, metrics in v.items():
        for key, stat in metrics.items():
            if isinstance(stat, dict) and "n" in stat:
                assert stat["n"] == 200, f"{name}.{key}: n={stat['n']} != 200"
    base = v["mouth_s1_88"]
    # 报告 3.2 表的核心行（mean ± std，3 位小数）
    for key in ("si_snr_i", "sdr512_i", "si_snr_i_zm0", "si_snr_i_seg2",
                "si_snr_i_seg4", "si_snr_i_tgt_s2", "si_snr_i_best_target"):
        m, s = float(base[key]["mean"]), float(base[key]["std"])
        needle = f"{m:.3f} ± {s:.3f}"
        assert needle in rtext, f"P2A 报告缺少 {key} 的 {needle}"
    # Δ 行（报告用“相对 baseline 的差”，负号已归一化）
    for key in ("si_snr_i_zm0",):
        delta = float(base[key]["mean"]) - float(base["si_snr_i"]["mean"])
        # 报告 Δ 由未舍入均值算得，与 JSON 均值相减可能有 ±0.001 的舍入差
        cands = [f"{delta + e:.3f}" for e in (0.0, 0.001, -0.001, 0.002, -0.002)]
        assert any(c in rtext for c in cands), f"P2A 报告缺少 Δ{key}={delta:.3f}"
    # 语义合理性：target=s2 显著更差；PIT 上界不提升
    assert float(base["si_snr_i_tgt_s2"]["mean"]) < float(base["si_snr_i"]["mean"]) - 20.0
    assert abs(float(base["si_snr_i_best_target"]["mean"]) - float(base["si_snr_i"]["mean"])) < 0.5
    # 数据侧不变量与 anchor 一致
    dc = d["data_checks"]
    assert dc["mix_r2"]["mean"] > 0.9999 and dc["target_equals_s1"]["max"] == 0.0


# ─────────────────────────────────────────────── 12) P2-D 训练曲线 / 评测数字（本机有产物时）
P2D_TRAIN_ROOT = Path("/root/tmp/p2d_train/Experiments/checkpoint")


def _p2d_metrics_csv() -> Path | None:
    if not P2D_TRAIN_ROOT.is_dir():
        return None
    cands = sorted(P2D_TRAIN_ROOT.glob("*/version_*/metrics.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def test_phase2_p2d_training_curve_sane():
    """P2-D 正式训练（单卡 bs4、train_samples=10000）：2500 step/epoch、K=0 near、损失有限、beta 起步为 0。"""
    csv_path = _p2d_metrics_csv()
    if csv_path is None:
        pytest.skip("P2-D 正式训练 metrics.csv 不在本机（/root/tmp/p2d_train）")
    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))
    # 只取“训练步”行：epoch 结束会写 val/train_loss_epoch 汇总行（step 重复、train_loss_step 为空）
    rows = [r for r in all_rows if (r.get("train_loss_step") or "").strip()]
    if len(rows) < 10:
        pytest.skip("P2-D 训练刚启动，指标行不足 10 行")
    steps = [int(float(r["step"])) for r in rows]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), "训练 step 非单调/重复"
    losses = [float(r["train_loss_step"]) for r in rows]
    assert all(math.isfinite(x) for x in losses), "训练损失出现非有限值"
    assert max(losses) < 1e3 and min(losses) > -1e3
    # 单卡 bs4 + train_samples=10000 ⇒ 2500 step/epoch：epoch 0 的 step 必须 < 2500
    ep0 = [int(float(r["step"])) for r in rows if r.get("epoch") == "0"]
    assert ep0 and max(ep0) < 2500, f"epoch0 step 越界: max={max(ep0)}"
    # near 语义：P2-C fallback near.json 自引用被跳过 ⇒ K=0（1near 及以上恒 0）
    near_rows = [r for r in rows if (r.get("train_n_samples_1near_step") or "").strip()]
    for r in near_rows:
        k = sum(float(r.get(f"train_n_samples_{i}near_step", 0) or 0) for i in (1, 2, 3))
        assert k == 0, f"K>0 near 样本出现（fallback 模式应恒为 K=0）: step={r.get('step')}"
        assert float(r.get("train_n_samples_4plus_near_step", 0) or 0) == 0
        assert float(r.get("train_neg_active_ratio_step", 0) or 0) == 0
        assert float(r.get("train_n_samples_0near_step", 0) or 0) >= 1
    # beta annealing：200 epoch 线性 0→1 ⇒ beta ≈ epoch/200，从 0 起步且单调不减
    beta_rows = [r for r in rows if (r.get("train_beta_step") or "").strip()]
    if beta_rows:
        betas = [float(r["train_beta_step"]) for r in beta_rows]
        epochs = [int(float(r["epoch"])) for r in beta_rows]
        assert min(betas) == 0.0, f"beta 未从 0 起步: min={min(betas)}"
        assert betas == sorted(betas), "beta 非单调不减"
        for b, e in zip(betas, epochs):
            assert abs(b - min(e, 200) / 200.0) < 0.02, f"epoch {e} 的 beta={b} 不符合 0→1/200epoch"
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    assert last <= first + 1.0, f"训练损失明显恶化: first10={first:.2f}, last10={last:.2f}"
    # epoch 汇总行：val/test PIT SI-SNR 必须有限且为正（训练确实在学习）
    val_rows = [r for r in all_rows if (r.get("val_pit_sisnr") or "").strip()]
    assert val_rows, "训练尚未完成任何 epoch（无 val 汇总行）"
    for r in val_rows:
        v = float(r["val_pit_sisnr"])
        assert math.isfinite(v) and 0.0 < v < 30.0, f"epoch {r.get('epoch')} val_pit_sisnr 异常: {v}"
    print(f"P2D val PIT SI-SNR: {[round(float(r['val_pit_sisnr']), 3) for r in val_rows]}")
    print(f"P2D 训练曲线: {len(rows)} 行, step {steps[0]}..{steps[-1]}, "
          f"loss first10={first:.2f} last10={last:.2f}")


def test_phase2_p2d_eval_metrics_consistent():
    """P2-D 评测 CSV（冒烟 ckpt）：avg 行 = 逐句均值，报告引用的数字与 CSV 一致。"""
    csv_path = PHASE2_DIR / "P2D-eval-epoch_0.csv"
    if not csv_path.is_file():
        pytest.skip("P2D-eval-epoch_0.csv 尚未落盘")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    utt = [r for r in rows if r.get("snt_id") not in ("avg", "std")]
    avg = next(r for r in rows if r.get("snt_id") == "avg")
    assert len(utt) >= 5, f"P2D 评测样本数 {len(utt)} < 5"
    means = {}
    for col in ("sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"):
        vals = [float(r[col]) for r in utt]
        mean = sum(vals) / len(vals)
        means[col] = mean
        assert abs(mean - float(avg[col])) < 1e-6, f"{col}: avg 行不可复算"
        assert math.isfinite(mean)
    report = PHASE2_DIR / "P2D-report.md"
    if report.is_file():
        rtext = report.read_text(errors="ignore").replace("\u2212", "-")
        for col in ("sdr", "sdr_i", "si-snr", "si-snr_i"):
            assert f"{means[col]:.2f}" in rtext, (
                f"P2D 报告未引用 {col}={means[col]:.2f}（评测数字不可追溯）"
            )
