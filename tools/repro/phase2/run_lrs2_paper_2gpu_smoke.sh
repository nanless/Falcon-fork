#!/bin/bash
# 本地 2 卡验证论文配方（2×bs4 × grad_accum 5 = 有效 batch 40，与论文对齐）
# 用途：在 8 卡大规模训练前，验证数据/配置/损失/吞吐是否正常。
# 用法：bash tools/repro/phase2/run_lrs2_paper_2gpu_smoke.sh [smoke|long]
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/falcon/bin/python}"
MODE="${1:-smoke}"
if [ "$MODE" = "long" ]; then CONF=configs/LRS2_2Mix_Paper_2gpu.yml; LOG=docs/verification/phase2/LRS2-paper-2gpu.log; else CONF=configs/LRS2_2Mix_Paper_2gpu_smoke.yml; LOG=docs/verification/phase2/LRS2-paper-2gpu-smoke.log; fi
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
echo "[launch] conf=$CONF log=$LOG"
setsid nohup "$PYTHON_BIN" audio_train.py --conf_dir "$CONF" > "$LOG" 2>&1 &
echo $! | tee docs/verification/phase2/LRS2-paper-2gpu.pid
echo "[launch] tail -f $LOG"
