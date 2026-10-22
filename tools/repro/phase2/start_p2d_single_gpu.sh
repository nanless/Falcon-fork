#!/usr/bin/env bash
# Phase2-D: start the formal LRS2-2Mix training on GPU0 only (Lead-approved plan, 2026-10-21).
#
# Why single GPU: the two A800s are shared with a non-team process (pid 2514085, ~127 GB
# combined), leaving ~26 GB free on GPU0 and ~9 GB on GPU1. bs4 fits on GPU0 (verified by
# the 1-GPU smoke run); bs2 is the documented fallback if OOM occurs.
#
# Usage:
#   bash tools/repro/phase2/start_p2d_single_gpu.sh [batch_size]
# Defaults: batch_size=4 (per GPU). Work dir: /root/tmp/p2d_train
set -euo pipefail
REPO="/root/code/github_repos/Falcon-fork"
CONF="$REPO/tools/repro/phase2/LRS2_2Mix_1gpu.yml"
BS="${1:-4}"
OUT="$REPO/docs/verification/phase2/P2D-gpu-snapshot.txt"

echo "[p2d] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPU snapshot before launch:" | tee "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$OUT"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$OUT"

# Optional batch override via a temp copy (keeps the committed config untouched)
if [ "$BS" != "4" ]; then
  TMP_CONF="/root/tmp/p2d_train_bs${BS}.yml"
  /root/miniforge3/envs/falcon/bin/python - "$CONF" "$TMP_CONF" "$BS" <<'PY'
import sys, yaml
src, dst, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src))
cfg["datamodule"]["data_config"]["batch_size"] = bs
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False, allow_unicode=True)
print(f"[p2d] batch_size={bs} -> {dst}")
PY
  CONF="$TMP_CONF"
fi

echo "[p2d] launching: CUDA_VISIBLE_DEVICES=0 config=$CONF batch=$BS"
CUDA_VISIBLE_DEVICES=0 bash "$REPO/tools/repro/phase2/run_p2d_train.sh" "$CONF" /root/tmp/p2d_train
echo "[p2d] launched. PID file: $REPO/docs/verification/phase2/P2D-train.pid"
echo "[p2d] log: $REPO/docs/verification/phase2/P2D-train.log"
