#!/bin/bash
# Phase1 T6: LRS2-2Mix 单文件 lrs2.tar.gz 断点续传下载（后台运行）
# 依赖：curl；本机 HF 直连需 mihomo 代理（127.0.0.1:17897），HF 镜像 hf-mirror.com 当前 TLS 不可达。
set -u
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY=http://127.0.0.1:17897
export no_proxy=127.0.0.1,localhost,::1
export NO_PROXY=127.0.0.1,localhost,::1
URL="https://huggingface.co/datasets/JusperLee/LRS2-2Mix/resolve/main/lrs2.tar.gz"
OUT="/root/data/lrs2_2mix/lrs2.tar.gz"
LOG="/root/code/github_repos/Falcon-fork/docs/verification/phase1/T6-data-fetch.log"
TOTAL=16470508590
mkdir -p "$(dirname "$OUT")"
{
  echo "=================================================================="
  echo "[start] $(date -Is)  URL=$URL"
  echo "[target] $OUT  expected_bytes=$TOTAL (16.47 GB)"
  echo "[note] HF_ENDPOINT=https://hf-mirror.com 在本机 TLS 不可达；改用 huggingface.co + 本机 mihomo 代理。"
  echo "=================================================================="
} >> "$LOG"

attempt=0
while [ "$attempt" -lt 100 ]; do
  attempt=$((attempt+1))
  before=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  echo "[attempt $attempt] $(date -Is) resume_from=${before}B" >> "$LOG"
  curl -L --fail --retry 5 --retry-delay 5 \
       -C - --connect-timeout 30 --speed-time 90 --speed-limit 65536 \
       --no-progress-meter -o "$OUT" "$URL" >> "$LOG" 2>&1
  rc=$?
  after=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  echo "[attempt $attempt] rc=$rc bytes=${after}B elapsed=$(date -Is)" >> "$LOG"
  if [ "$rc" -eq 0 ] && [ "$after" -ge "$TOTAL" ]; then
    echo "[download-complete] $(date -Is) bytes=$after" >> "$LOG"
    break
  fi
  sleep 15
done

size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
echo "[verify] size=$size expected=$TOTAL" >> "$LOG"
if [ "$size" -ge "$TOTAL" ]; then
  echo "[sha256] $(sha256sum "$OUT" | cut -d' ' -f1)" >> "$LOG"
  echo "COMPLETE $size" > /root/data/lrs2_2mix/.download_status
else
  echo "INCOMPLETE $size $TOTAL" > /root/data/lrs2_2mix/.download_status
fi
echo "[end] $(date -Is)" >> "$LOG"
