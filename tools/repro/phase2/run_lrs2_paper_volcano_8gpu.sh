#!/bin/bash
# LRS2-2Mix 论文配方训练 —— 8 卡（火山引擎 / 任意 8×A800 节点）
#
# 配置：configs/LRS2_2Mix_Paper_8gpu.yml
#   - 8 GPUs × bs 5 = global batch 40（与论文 10×4=40 对齐）
#   - lr 1e-3 AdamW/0.01, ReduceLROnPlateau(15,0.5), grad clip 5
#   - epochs 1000 / early stop 30，train_samples 10000/epoch，β 0→1 over 200 epochs
#   - 数据：LRS2-2Mix “论文配方”双向 target（s1/s2 对调）+ near=另一说话人
#
# 用法（在火山引擎任务里直接执行）：
#   bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
# 可用环境变量覆盖：
#   PYTHON_BIN=/path/to/python   CONF=configs/LRS2_2Mix_Paper_8gpu.yml
#   DATA_ROOT=/root/data/falcon_lrs2_2mix_paper   LOG_DIR=/path/to/logs   NPROC=8
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/falcon/bin/python}"
CONF="${CONF:-configs/LRS2_2Mix_Paper_8gpu.yml}"
DATA_ROOT="${DATA_ROOT:-/root/data/falcon_lrs2_2mix_paper}"
LOG_DIR="${LOG_DIR:-$REPO/docs/verification/phase2}"
NPROC="${NPROC:-8}"
EXP_LOG="$LOG_DIR/LRS2-paper-$(date -u +%Y%m%dT%H%M%SZ).log"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$LOG_DIR" "$(dirname "$EXP_LOG")"

echo "== LRS2-2Mix paper recipe (8 GPU) =="
echo "repo      = $REPO"
echo "python    = $PYTHON_BIN"
echo "conf      = $CONF"
echo "data_root = $DATA_ROOT"
echo "log       = $EXP_LOG"

# 0) 环境/数据/权重自检
"$PYTHON_BIN" - <<'PY'
import sys, torch, yaml
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
assert torch.cuda.device_count() >= 8, f"need 8 GPUs, got {torch.cuda.device_count()}"
PY
for f in "$DATA_ROOT/tr/mix.json" "$DATA_ROOT/cv/mix.json" "$DATA_ROOT/tt/mix.json"; do
  [ -f "$f" ] && echo "data ok: $f ($(python3 -c "import json;print(len(json.load(open('$f'))))") entries)" || { echo "MISSING $f"; exit 2; }
done
BACKBONE="${BACKBONE:-/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar}"
if [ ! -f "$BACKBONE" ]; then
  echo "backbone not found: $BACKBONE"
  echo "生成：python tools/repro/extract_video_backbone.py --ckpt /root/.cache/falcon_weights/falcon.pt --out $BACKBONE"
  exit 2
fi

# 1) 启动（PL 2.x 在 8 卡上自动 DDP；gpus: -1 = 全部可见卡）
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
setsid nohup "$PYTHON_BIN" audio_train.py --conf_dir "$CONF" > "$EXP_LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/LRS2-paper.pid"
echo "[launch] pid=$PID conf=$CONF"
echo "[launch] tail -f $EXP_LOG"
echo "[launch] monitor: python3 tools/repro/phase2/monitor_lrs2_paper.py"
echo "$EXP_LOG" > "$LOG_DIR/LRS2-paper.logpath"
