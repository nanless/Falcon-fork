#!/bin/bash
# 等当前 LRS2 无负样本训练（PID 4002894）结束后，用 s2 干扰说话人作 near 负样本续训 30 epoch。
set -u
REPO=/root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
OLD_PID=${OLD_PID:-4002894}
LOG=$REPO/docs/verification/phase2/P2D-near-train.log
CKPT=/root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/last.ckpt
echo "[supervisor] $(date -Is) waiting for PID $OLD_PID" >> "$LOG"
while kill -0 "$OLD_PID" 2>/dev/null; do sleep 60; done
echo "[supervisor] $(date -Is) old training finished; resume with ckpt=$CKPT" >> "$LOG"
if [ ! -f "$CKPT" ]; then echo "[supervisor] missing ckpt, abort" >> "$LOG"; exit 1; fi
mkdir -p /root/tmp/p2d_near_train && cd /root/tmp/p2d_near_train
PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=0 timeout 86400 "$ENV" "$REPO/audio_train.py" \
  --conf_dir "$REPO/tools/repro/phase2/LRS2_2Mix_near_s2.yml" --ckpt_path "$CKPT" >> "$LOG" 2>&1
echo "[supervisor] $(date -Is) contrastive fine-tune exit=$?" >> "$LOG"
