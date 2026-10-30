#!/bin/bash
# Prepare the Studio Python environment.
# stdout contains only the selected Python path; all messages go to stderr.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

YES=0
usage() {
    cat <<'EOF'
用法：bootstrap.sh [--yes]

准备 Falcon Studio 的 Python 环境，并在 stdout 输出最终解释器路径（消息走 stderr）。

  --yes       确认安装/变更依赖；缺少 --yes 时只打印将要变更的包并退出 1
  -h, --help  显示本帮助

环境变量：
  FALCON_MANAGED_ENV=1  受管环境：仅当解释器已能导入 torch/numpy/Pillow 时直接使用，
                        否则报错退出，绝不创建环境或安装包
  FALCON_PYTHON=PATH    指定解释器；若它已具备依赖则直接使用，否则作为创建 venv 的源解释器
  FALCON_VENV=PATH      指定环境目录；若其中解释器已具备依赖则直接使用
EOF
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes|-y) YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'bootstrap.sh: 未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

valid_python() {
    [ -x "$1" ] && "$1" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))' >/dev/null 2>&1
}

has_runtime_deps() {
    [ -x "$1" ] && "$1" -c 'import torch, numpy, PIL' >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Safety guard 1: a managed environment is only ever used as-is. It is never
# created, upgraded or installed into by this script.
# ---------------------------------------------------------------------------
if [ -n "${FALCON_MANAGED_ENV:-}" ]; then
    MANAGED_PYTHON="${FALCON_PYTHON:-}"
    if [ -z "$MANAGED_PYTHON" ] && [ -n "${FALCON_VENV:-}" ]; then
        MANAGED_PYTHON="$FALCON_VENV/bin/python"
    fi
    if [ -z "$MANAGED_PYTHON" ]; then
        MANAGED_PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    fi
    if [ -z "$MANAGED_PYTHON" ] || ! valid_python "$MANAGED_PYTHON"; then
        printf 'FALCON_MANAGED_ENV=1，但没有找到可用的 Python 3.10–3.12：%s\n拒绝创建环境或安装依赖。\n' "${MANAGED_PYTHON:-<none>}" >&2
        exit 1
    fi
    if ! has_runtime_deps "$MANAGED_PYTHON"; then
        printf 'FALCON_MANAGED_ENV=1，但 %s 不能导入 torch/numpy/Pillow；拒绝自动安装依赖。\n' "$MANAGED_PYTHON" >&2
        exit 1
    fi
    printf '受管环境已就绪（跳过创建/安装）：%s\n' "$MANAGED_PYTHON" >&2
    printf '%s\n' "$MANAGED_PYTHON"
    exit 0
fi

# ---------------------------------------------------------------------------
# Safety guard 2: an explicitly supplied interpreter that already imports the
# runtime dependencies is reused as-is instead of provoking an install into it.
# ---------------------------------------------------------------------------
if [ -n "${FALCON_PYTHON:-}" ] || [ -n "${FALCON_VENV:-}" ]; then
    EXPLICIT_PYTHON="${FALCON_PYTHON:-}"
    if [ -z "$EXPLICIT_PYTHON" ]; then
        EXPLICIT_PYTHON="$FALCON_VENV/bin/python"
    fi
    if valid_python "$EXPLICIT_PYTHON" && has_runtime_deps "$EXPLICIT_PYTHON"; then
        printf '复用已有环境（跳过创建/安装）：%s\n' "$EXPLICIT_PYTHON" >&2
        printf '%s\n' "$EXPLICIT_PYTHON"
        exit 0
    fi
fi

if [ -n "${FALCON_VENV:-}" ]; then
    ENV_DIR="$FALCON_VENV"
elif valid_python "$APP_DIR/.venv/bin/python"; then
    ENV_DIR="$APP_DIR/.venv"
else
    ENV_DIR="$HOME/Library/Caches/FalconStudio/venv"
fi
case "$ENV_DIR" in
    /*) ;;
    *) ENV_DIR="$PWD/$ENV_DIR" ;;
esac

if valid_python "$ENV_DIR/bin/python" && "$ENV_DIR/bin/python" "$SCRIPT_DIR/check_environment.py" --quiet; then
    printf '%s\n' "$ENV_DIR/bin/python"
    exit 0
fi

SOURCE_PYTHON=""
if ! valid_python "$ENV_DIR/bin/python"; then
    if [ -e "$ENV_DIR" ]; then
        printf '已有环境无法运行或 Python 版本不受支持：%s\n请设置 FALCON_VENV 指向新的环境目录，再重试。\n' "$ENV_DIR" >&2
        exit 1
    fi
    SOURCE_PYTHON="${FALCON_PYTHON:-}"
    if [ -n "$SOURCE_PYTHON" ] && ! valid_python "$SOURCE_PYTHON"; then
        printf 'FALCON_PYTHON 需要指向可运行的 Python 3.10–3.12：%s\n' "$SOURCE_PYTHON" >&2
        exit 1
    fi
    if [ -z "$SOURCE_PYTHON" ]; then
        for name in python3.11 python3.12 python3.10 python3; do
            candidate="$(command -v "$name" 2>/dev/null || true)"
            if [ -n "$candidate" ] && valid_python "$candidate"; then
                SOURCE_PYTHON="$candidate"
                break
            fi
        done
    fi
    if [ -z "$SOURCE_PYTHON" ]; then
        # Finder does not inherit a terminal's PATH; check common installations.
        for candidate in \
            /opt/homebrew/bin/python3.11 \
            /usr/local/bin/python3.11 \
            /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
            "$HOME"/.local/share/uv/python/cpython-3.11.*/bin/python3.11; do
            if valid_python "$candidate"; then
                SOURCE_PYTHON="$candidate"
                break
            fi
        done
    fi
    if [ -z "$SOURCE_PYTHON" ]; then
        printf '\n没有找到 Python 3.10–3.12。请先安装 Python 3.11，或设置：\n  export FALCON_PYTHON="/完整路径/python3.11"\n然后重新启动。详见 README.md。\n' >&2
        exit 1
    fi
    printf '\n正在创建运行环境：%s\n' "$ENV_DIR" >&2
    mkdir -p "$(dirname -- "$ENV_DIR")"
    "$SOURCE_PYTHON" -m venv "$ENV_DIR" >&2
fi

if [ ! -f "$APP_DIR/requirements-lock.txt" ] && [ ! -f "$APP_DIR/requirements.txt" ]; then
    printf '找不到 %s/requirements-lock.txt 或 requirements.txt，无法安装依赖。\n' "$APP_DIR" >&2
    exit 1
fi
if [ -f "$APP_DIR/requirements-lock.txt" ]; then
    REQUIREMENTS_FILE="$APP_DIR/requirements-lock.txt"
else
    REQUIREMENTS_FILE="$APP_DIR/requirements.txt"
fi

# Print the packages that would be installed or changed, using only the stdlib.
# Exit code: 0 = nothing to do, 3 = changes required, other = cannot compute.
print_install_plan() {
    "$1" - "$REQUIREMENTS_FILE" <<'PY' >&2
import importlib.metadata as metadata
import re
import sys
from pathlib import Path

requirements = Path(sys.argv[1])
specs = []
for raw in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*(.*)$", line)
    if match:
        specs.append((match.group(1), line))


def parts(version):
    match = re.match(r"^(\d+(?:\.\d+)*)", version or "")
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


changes = []
for name, line in specs:
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        changes.append((name, None, line))
        continue
    if "==" in line:
        target = line.split("==", 1)[1].split(",", 1)[0].strip()
        if parts(installed) != parts(target):
            changes.append((name, installed, line))
    else:
        lower = re.search(r">=\s*([0-9][0-9.]*)", line)
        upper = re.search(r"<\s*([0-9][0-9.]*)", line)
        satisfied = not (
            (lower and parts(installed) < parts(lower.group(1)))
            or (upper and parts(installed) >= parts(upper.group(1)))
        )
        if not satisfied:
            changes.append((name, installed, line))

print(f"将安装/变更 {len(changes)}/{len(specs)} 个包（来源：{requirements}）：")
for name, installed, line in changes[:40]:
    state = "未安装" if installed is None else f"已安装 {installed}"
    print(f"  - {name}: {state} -> {line}")
if len(changes) > 40:
    print(f"  …其余 {len(changes) - 40} 个")
sys.exit(3 if changes else 0)
PY
}

PLAN_PYTHON="$ENV_DIR/bin/python"
if ! valid_python "$PLAN_PYTHON"; then
    PLAN_PYTHON="${SOURCE_PYTHON:-}"
fi
plan_status=3
if [ -n "$PLAN_PYTHON" ] && [ -x "$PLAN_PYTHON" ]; then
    set +e
    print_install_plan "$PLAN_PYTHON"
    plan_status=$?
    set -e
fi
if [ "$plan_status" -eq 0 ]; then
    printf '依赖已满足，无需安装。\n' >&2
    printf '%s\n' "$ENV_DIR/bin/python"
    exit 0
fi
if [ "$plan_status" -ne 3 ]; then
    printf '无法预计算依赖差异，将按以下清单安装/变更：\n' >&2
    grep -E '^[A-Za-z0-9_.-]+' "$REQUIREMENTS_FILE" | head -n 40 | sed 's/^/  - /' >&2 || true
fi

if [ "$YES" -ne 1 ]; then
    printf '\n未加 --yes：已停止，尚未安装或变更任何包。\n确认清单无误后重新运行：\n  ./scripts/bootstrap.sh --yes\n' >&2
    exit 1
fi

printf '\n正在安装 Falcon 依赖，首次运行需要联网，请稍候…\n' >&2
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$ENV_DIR/bin/python" -r "$REQUIREMENTS_FILE" >&2
else
    if ! "$ENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        "$ENV_DIR/bin/python" -m ensurepip >&2
    fi
    "$ENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE" >&2
fi
# Version drift is a warning after an explicit install; import failures still fail.
"$ENV_DIR/bin/python" "$SCRIPT_DIR/check_environment.py" --check-only >&2
printf '%s\n' "$ENV_DIR/bin/python"
