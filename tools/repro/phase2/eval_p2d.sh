#!/usr/bin/env bash
# Phase2-D: evaluate a trained LRS2-2Mix checkpoint (SI-SNRi / SDRi) with audio_visual_test.py.
#
# Usage:
#   bash tools/repro/phase2/eval_p2d.sh <ckpt_path> [data_dir] [limit] [gpus]
# Defaults:
#   data_dir = /root/tmp/lrs2_eval_tt_full   (3000 utts, target=s1, P2-A/B verified)
#   limit    = "" (all) -- use e.g. 300 for a quick pass
#   gpus     = 0
#
# Results are written to <workdir>/Experiments/checkpoint/LRS2_2Mix/results/ and
# copied to docs/verification/phase2/P2D-eval-<ckpt stem>.csv
set -euo pipefail

REPO="/root/code/github_repos/Falcon-fork"
ENV="/root/miniforge3/envs/falcon/bin/python"
CONF="$REPO/configs/LRS2_2Mix.yml"
CKPT="${1:?usage: eval_p2d.sh <ckpt_path> [data_dir] [limit] [gpus]}"
DATA="${2:-/root/tmp/lrs2_eval_tt_full}"
LIMIT="${3:-}"
GPUS="${4:-0}"
WORK="/root/tmp/p2d_eval"
TAG="$(basename "$CKPT" .ckpt | tr '=' '_')"
mkdir -p "$WORK" "$REPO/docs/verification/phase2"
cd "$WORK"

EXTRA=()
if [ -n "$LIMIT" ]; then EXTRA+=(--limit "$LIMIT"); fi

echo "[p2d-eval] ckpt=$CKPT data=$DATA limit=${LIMIT:-all} gpus=$GPUS"
PYTHONPATH="$REPO" "$ENV" "$REPO/audio_visual_test.py" \
  --conf_dir "$CONF" \
  --ckpt_path "$CKPT" \
  --data_dir "$DATA" \
  --gpus "$GPUS" \
  "${EXTRA[@]}" 2>&1 | tee "$REPO/docs/verification/phase2/P2D-eval-$TAG.log"

COMBINED="$WORK/Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv"
if [ -f "$COMBINED" ]; then
  cp "$COMBINED" "$REPO/docs/verification/phase2/P2D-eval-$TAG.csv"
  echo "[p2d-eval] combined metrics -> docs/verification/phase2/P2D-eval-$TAG.csv"
  cat "$COMBINED"
else
  echo "[p2d-eval] WARNING: combined metrics not found at $COMBINED" >&2
fi
