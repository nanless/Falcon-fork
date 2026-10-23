#!/bin/bash
# 等网络恢复后自动续传 RealSSA 下载（当前因代理/上游带宽骤降暂停）
# 逻辑：每 5 分钟测一次 HF 下载速度（2MB Range，经 mihomo 代理）；
#       速度 >= 1.0 MB/s 时 relaunch download_realssa_shared.py（自动断点续传 + 完成后 rsync 归档）。
set -u
D=/root/group-shared/voiceprint/data/multimodal/_download
REPO=/root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
LOG=$D/resume_supervisor.log
echo "[supervisor $(date -Is)] waiting for network recovery (threshold 0.35 MB/s)" >> "$LOG"
while true; do
  SPEED=$(timeout 25 curl -sL -x http://127.0.0.1:17897 -H "Authorization: Bearer $TOKEN" \
    -r 0-2000000 -o /dev/null -w "%{speed_download}" \
    "https://huggingface.co/datasets/JusperLee/RealSSA-Sim/resolve/main/cv/cv-000001.tar?download=true" 2>/dev/null || echo 0)
  SPEED=${SPEED%.*}
  [ -z "$SPEED" ] && SPEED=0
  echo "[supervisor $(date -Is)] speed=${SPEED} B/s" >> "$LOG"
  if [ "$SPEED" -ge 350000 ] 2>/dev/null; then
    echo "[supervisor $(date -Is)] network recovered -> launching download" >> "$LOG"
    setsid nohup "$PY" "$REPO/tools/repro/phase2/download_realssa_shared.py" --stage all --workers 2 >> "$D/realssa_console.log" 2>&1 &
    echo "pid=$!" > "$D/download.pid"
    echo "[supervisor $(date -Is)] launched pid=$(cat $D/download.pid)" >> "$LOG"
    exit 0
  fi
  sleep 300
done
