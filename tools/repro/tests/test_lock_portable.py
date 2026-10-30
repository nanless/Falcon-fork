"""A8-4：`docs/verification/conda-env-lock.txt` 可移植性回归。

背景（review A8-4，2026-10-30）：
  原 lock 由 `pip freeze` 直接生成，其中 18 个包被写成
  `pkg @ file:///root/wheelhouse/....whl#sha256=...` —— 绝对路径只在本机有效，
  新机器 `pip install -r` 必然失败（torch/torchaudio/torchvision/pytorch-lightning +
  14 个 nvidia-* CUDA 库）。2026-10-24 已改写为 `pkg==version`，并在文件顶部加入
  “先 cu128 索引装 torch，再 pip install -r”的两步重建说明。

本测试守住这三条不变量（只读；不联网、不安装）：
  1. 无 `file://`、无 `/root/` 等本机绝对路径、无 `pkg @ url` 形式的直接引用；
  2. 每行都是合法 `name==version`，包名规范化后唯一（无重复、无漏版本）；
  3. 包集合与版本 == 当前 conda `falcon` 环境的 `pip freeze`（排除 pip/setuptools/wheel
     这类工具链包：它们随 Python/conda 提供，不应由应用锁约束）。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LOCK = REPO / "docs/verification/conda-env-lock.txt"

#: 工具链包：随解释器/conda 提供，不进入应用依赖锁（`pip freeze` 在 conda 环境里通常也不列）。
TOOLCHAIN = {"pip", "setuptools", "wheel", "distribute"}

_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


def _canonical(name: str) -> str:
    try:
        from packaging.utils import canonicalize_name
    except ImportError:  # pragma: no cover - packaging 是 pip 的硬依赖
        return name.lower().replace("_", "-").replace(".", "-")
    return canonicalize_name(name)


def _lock_entries() -> dict[str, str]:
    """解析 lock -> {canonical_name: version}；格式非法直接 fail（而不是静默跳过）。"""
    assert LOCK.is_file(), f"缺少 lock 文件 {LOCK}"
    entries: dict[str, str] = {}
    for lineno, raw in enumerate(LOCK.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        assert m, f"lock:{lineno} 不是 `name==version` 形式: {line!r}"
        name, version = m.group(1), m.group(2)
        key = _canonical(name)
        assert key not in entries, f"lock:{lineno} 重复包 {name}（{key}）"
        entries[key] = version
    assert entries, "lock 文件没有任何依赖条目"
    return entries


def _dependency_lines() -> list[tuple[int, str]]:
    """lock 中的依赖行（跳过注释/空行）：[(行号, 原始行)]。"""
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(LOCK.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append((lineno, line))
    return out


def _pip_freeze() -> dict[str, str]:
    """当前解释器环境的 `pip freeze` -> {canonical_name: version}（本地命令，不联网）。

    本环境的 torch/PL/nvidia-* 是用 `file:///root/wheelhouse/...` 轮子直接安装的，
    `pip freeze` 会把它们打印成 `pkg @ file://...`（direct reference）。这里用
    `importlib.metadata.version()` 归一化回真实版本号，才能与 lock 的 `pkg==version` 比较。
    """
    r = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, f"pip freeze 失败: {r.stderr[-2000:]}"
    from importlib.metadata import version as dist_version

    out: dict[str, str] = {}
    for raw in r.stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:  # 本地 wheel/VCS/editable 安装
            name = _canonical(line.split(" @ ", 1)[0])
            try:
                out[name] = dist_version(line.split(" @ ", 1)[0])
            except Exception:  # pragma: no cover - 元数据缺失时保留标记
                out[name] = "<direct-reference>"
            continue
        name, _, version = line.partition("==")
        if not version:
            continue
        out[_canonical(name)] = version
    return out


def test_lock_has_no_local_absolute_paths():
    """依赖行不得含本机绝对路径（`file://`、`/root/...`）或 `pkg @ url` 直接引用。

    只看依赖行（跳过 `#` 注释）：顶部重建说明里会合法地提到 `file://` 字样。
    """
    bad: list[str] = []
    for lineno, line in _dependency_lines():
        if "file://" in line:
            bad.append(f"lock:{lineno} 含 file:// -> {line}")
        if "/root/" in line:
            bad.append(f"lock:{lineno} 含 /root/ 绝对路径 -> {line}")
        if " @ " in line:
            bad.append(f"lock:{lineno} 含 `pkg @ url` 直接引用 -> {line}")
    assert not bad, "lock 存在不可移植的依赖行:\n" + "\n".join(bad)


def test_lock_parses_with_versions_and_no_duplicates():
    """每行必须可解析为 `name==version`，规范化包名唯一。"""
    entries = _lock_entries()
    assert len(entries) >= 100, f"lock 条目数异常偏少: {len(entries)}"
    # torch 三件套必须保留本地版本标记 `+cu128`（AGENTS.md §4 明确不得替换）
    for pkg in ("torch", "torchaudio", "torchvision"):
        assert pkg in entries, f"lock 缺少 {pkg}"
        assert entries[pkg].endswith("+cu128"), f"{pkg} 版本丢失 +cu128: {entries[pkg]}"


def test_lock_matches_current_env_pip_freeze():
    """lock 的包集合与版本 == 当前 falcon 环境的 `pip freeze`（排除 pip/setuptools/wheel）。"""
    lock = _lock_entries()
    freeze = {n: v for n, v in _pip_freeze().items() if n not in TOOLCHAIN}

    missing = sorted(set(freeze) - set(lock))
    extra = sorted(set(lock) - set(freeze))
    assert not missing, f"当前环境已安装但 lock 未锁定（pip freeze 有、lock 无）: {missing}"
    assert not extra, f"lock 锁定但当前环境未安装（lock 有、pip freeze 无）: {extra}"

    mismatched = {
        name: (lock[name], freeze[name])
        for name in lock
        if freeze.get(name) != lock[name]
    }
    assert not mismatched, f"lock 版本与当前环境不一致（lock vs pip freeze）: {mismatched}"
