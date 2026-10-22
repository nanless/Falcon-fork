#!/usr/bin/env bash
# Phase2-D: launch the formal LRS2-2Mix training in the background.
#
# Usage:
#   bash tools/repro/phase2/run_p2d_train.sh [config] [workdir]
# Defaults:
#   config  = configs/LRS2_2Mix.yml
#   workdir = /root/tmp/p2d_train   (keeps Experiments/ + ckpt outside the repo)
#
# Outputs:
#   docs/verification/phase2/P2D-train.log   (append)
#   docs/verification/phase2/P2D-train.pid   (PID of the detached trainer)
#
# Precondition: P2-C conversion finished -> {tr,cv,tt}/{mix.json,target.json} exist.
set -euo pipefail

REPO="/root/code/github_repos/Falcon-fork"
ENV="/root/miniforge3/envs/falcon/bin/python"
CONF="${1:-$REPO/configs/LRS2_2Mix.yml}"
WORK="${2:-/root/tmp/p2d_train}"
LOG="$REPO/docs/verification/phase2/P2D-train.log"
PIDF="$REPO/docs/verification/phase2/P2D-train.pid"

DATA_ROOT="$($ENV - "$CONF" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["datamodule"]["data_config"]["train_dir"].rsplit("/", 1)[0])
PY
)"

echo "[p2d] config  : $CONF"
echo "[p2d] workdir : $WORK"
echo "[p2d] data    : $DATA_ROOT"

# --- P2-C data readiness gate -------------------------------------------------
for split in tr cv tt; do
  for f in mix.json target.json; do
    if [ ! -s "$DATA_ROOT/$split/$f" ]; then
      echo "[p2d] ABORT: missing/empty $DATA_ROOT/$split/$f (P2-C conversion not finished?)" >&2
      exit 1
    fi
  done
done
echo "[p2d] data readiness: OK ($(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1) total)"

mkdir -p "$WORK"
cd "$WORK"
: > "$LOG"
setsid nohup env PYTHONPATH="$REPO" "$ENV" "$REPO/audio_train.py" --conf_dir "$CONF" \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDF"
sleep 5
if kill -0 "$PID" 2>/dev/null; then
  echo "[p2d] training started: PID=$PID"
  echo "[p2d] log: $LOG"
  echo "[p2d] pid file: $PIDF"
  echo "[p2d] tail: $(tail -n 2 "$LOG" | tr '\n' ' ')"
else
  echo "[p2d] ERROR: trainer exited within 5s; check $LOG" >&2
  tail -n 30 "$LOG" >&2 || true
  exit 1
fi
