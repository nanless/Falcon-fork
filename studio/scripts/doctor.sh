#!/bin/bash
# Diagnose dependencies, PyTorch/CUDA/Metal and model downloads.
# With --check-only, use the supplied interpreter directly: never run bootstrap or install anything.
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

CHECK_ONLY=0
BOOTSTRAP_ARGS=()
usage() {
    cat <<'EOF'
用法：doctor.sh [--check-only] [--yes]

检查依赖、PyTorch/CUDA/Metal 与模型下载状态。

  --check-only  直接使用 FALCON_PYTHON / FALCON_VENV（或 PATH 上的 python3），
                跳过 bootstrap、创建环境与依赖安装，继续跑诊断
  --yes         非 --check-only 模式下，允许 bootstrap 安装/变更依赖
  -h, --help    显示本帮助
EOF
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --yes|-y) BOOTSTRAP_ARGS+=("--yes"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'doctor.sh: 未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$CHECK_ONLY" -eq 1 ]; then
    if [ -n "${FALCON_PYTHON:-}" ]; then
        PYTHON_BIN="$FALCON_PYTHON"
    elif [ -n "${FALCON_VENV:-}" ]; then
        PYTHON_BIN="$FALCON_VENV/bin/python"
    else
        PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    fi
    if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
        printf 'doctor.sh --check-only 需要可执行的解释器：请设置 FALCON_PYTHON 或 FALCON_VENV（当前：%s）。\n' "${PYTHON_BIN:-<none>}" >&2
        exit 1
    fi
    printf '使用指定解释器（跳过 bootstrap 与依赖安装）：%s\n' "$PYTHON_BIN" >&2
    "$PYTHON_BIN" "$SCRIPT_DIR/check_environment.py" --check-only
else
    if [ "${#BOOTSTRAP_ARGS[@]}" -gt 0 ]; then
        PYTHON_BIN="$("$SCRIPT_DIR/bootstrap.sh" "${BOOTSTRAP_ARGS[@]}")"
    else
        PYTHON_BIN="$("$SCRIPT_DIR/bootstrap.sh")"
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/check_environment.py"
fi

"$PYTHON_BIN" - "$APP_DIR" <<'PY'
from pathlib import Path
import platform
import sys
import torch

app = Path(sys.argv[1])
sys.path.insert(0, str(app))
from falcon_runtime import VARIANTS, weights_source
print(f"系统：{platform.platform()} / {platform.machine()}")
print(f"PyTorch：{torch.__version__}")
print(f"MPS 已构建：{torch.backends.mps.is_built()}")
print(f"MPS 当前可用：{torch.backends.mps.is_available()}")
print(f"CUDA 可用：{torch.cuda.is_available()}" + (f"（{torch.cuda.get_device_name(0)}）" if torch.cuda.is_available() else ""))
for variant, info in VARIANTS.items():
    source = weights_source(variant)
    if source == "cached":
        print(f"模型 {info['repo']}：已下载（Hugging Face 缓存）")
    elif source == "huggingface":
        print(f"模型 {info['repo']}：首次加载时从 Hugging Face 下载")
    else:
        print(f"模型 {info['repo']}：未安装 huggingface_hub，无法下载", file=sys.stderr)
        if variant == "small":
            raise SystemExit(1)
print("这里只检查环境和模型文件；实际加载与分离结果请在应用中确认。")
PY
