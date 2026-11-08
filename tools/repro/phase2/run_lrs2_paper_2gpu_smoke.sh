#!/bin/bash
# 本地 2 卡验证（2 卡 × bs4 × grad_accum 5 = 有效 batch 40）
# ⚠ 本脚本的 global batch 是 40，**不是**论文 Table 16 的 32（8 × bs4 × accum1）：
#   40 是沿用 LEGACY 8 卡口径（8 × bs5 × accum1 = 40）；本机只有 2 卡，故用 accum 5 凑回 40。
#   要在 2 卡上对齐论文的 32，需要把 configs/LRS2_2Mix_Paper_2gpu.yml 的
#   accumulate_grad_batches 从 5 改成 4（4 × 2 × 4 = 32）——2026-11-08 注解时只改注释、不动键值。
#   另：默认 smoke 配置 LRS2_2Mix_Paper_2gpu_smoke.yml 是 **legacy 配方**
#   （train=pairwise_neg_snr、β/200、无 seed），同样不是 Table 16；Table 16 配方见
#   configs/LRS2_2Mix_Paper_v2.yml，8 卡启动见 run_lrs2_paper_volcano_8gpu.sh。
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
