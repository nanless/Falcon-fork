#!/bin/bash
# LRS2-2Mix 论文 Table 16 配方训练 —— 8 卡（火山引擎 / 任意 8×A800/3090 节点）
#
# 默认配置：configs/LRS2_2Mix_Paper_v2.yml（论文 arXiv:2610.32749 App D Table 16）
#   - 重建损失 -SI-SNR；对比温度 tau(beta) 0->1 线性退火，前 1000 epoch
#   - batch 4/GPU × 8 = global batch 32；epochs 1000；EarlyStopping(patience 30)；seed 42
#   - lr 1e-3 AdamW / wd 0.01、ReduceLROnPlateau(15, 0.5)、grad clip 5
#   - 数据：LRS2-2Mix “论文配方”双向 target（s1/s2 对调）+ near=另一说话人
# 旧配置 configs/LRS2_2Mix_Paper_8gpu.yml 标记为 LEGACY（bs5 / -SNR / beta-200），保留不删。
#
# DATA_ROOT 会被**真正渲染进临时 conf**（$TMPDIR/conf_<pid>.yml）：本脚本把
# data_config.{train_dir,valid_dir,test_dir} 改写为 $DATA_ROOT/{tr,cv,tt}，训练指向该临时
# conf。这样 DATA_ROOT 与训练实际读取的路径不会再错位（旧版只用 DATA_ROOT 做自检）。
#
# 用法（在火山引擎任务里直接执行）：
#   bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
# 可用环境变量覆盖：
#   PYTHON_BIN=/path/to/python  CONF=configs/LRS2_2Mix_Paper_v2.yml
#   DATA_ROOT=/data/falcon_lrs2_2mix_paper  LOG_DIR=/path/to/logs  NPROC=8
#   CKPT_PATH=/path/to/last.ckpt  # 可选：续训（路径必须存在，否则 audio_train 直接 SystemExit）
#   DRY_RUN=1                     # 只做自检 + 渲染 conf + 打印启动命令，不启动训练
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/falcon/bin/python}"
CONF="${CONF:-configs/LRS2_2Mix_Paper_v2.yml}"
DATA_ROOT="${DATA_ROOT:-/root/data/falcon_lrs2_2mix_paper}"
LOG_DIR="${LOG_DIR:-$REPO/docs/verification/phase2}"
NPROC="${NPROC:-8}"
TMPDIR="${TMPDIR:-/root/tmp}"
DRY_RUN="${DRY_RUN:-0}"
CKPT_PATH="${CKPT_PATH:-}"
EXP_LOG="$LOG_DIR/LRS2-paper-$(date -u +%Y%m%dT%H%M%SZ).log"
RENDERED_CONF="$TMPDIR/conf_$$.yml"

mkdir -p "$LOG_DIR" "$TMPDIR"

echo "== LRS2-2Mix paper Table 16 recipe (8 GPU) =="
echo "repo        = $REPO"
echo "python      = $PYTHON_BIN"
echo "conf(src)   = $CONF"
echo "data_root   = $DATA_ROOT"
echo "nproc       = $NPROC"
echo "log         = $EXP_LOG"
echo "rendered    = $RENDERED_CONF"

# 0) NPROC 参数检查，然后按 NPROC 限定可见卡，再断言 device_count == NPROC
if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NPROC must be a positive integer, got '$NPROC'" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PYTHON_BIN" - "$NPROC" <<'PY'
import sys
import torch

want = int(sys.argv[1])
got = torch.cuda.device_count()
print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()} "
      f"| visible gpus: {got} | expected (NPROC): {want}")
if got != want:
    raise SystemExit(
        f"GPU count mismatch: NPROC={want} but torch.cuda.device_count()={got}. "
        "Set NPROC to the number of visible GPUs on this node."
    )
PY

# 1) 渲染临时 conf：把 DATA_ROOT 真正写进三处 data dir（训练读这份）
"$PYTHON_BIN" - "$CONF" "$DATA_ROOT" "$RENDERED_CONF" <<'PY'
import os
import sys

import yaml

src, data_root, out = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
with open(src) as f:
    conf = yaml.safe_load(f)
dc = conf["datamodule"]["data_config"]
for split, key in (("tr", "train_dir"), ("cv", "valid_dir"), ("tt", "test_dir")):
    dc[key] = os.path.join(data_root, split)
with open(out, "w") as f:
    yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)
print(f"[render] source conf : {src}")
print(f"[render] rendered    : {out}")
for key in ("train_dir", "valid_dir", "test_dir"):
    print(f"[render] effective {key} = {dc[key]}")
print(f"[render] batch_size = {dc.get('batch_size')} | "
      f"seed = {(conf.get('training') or {}).get('seed')} | "
      f"loss.train.sdr_type = {conf['loss']['train'].get('sdr_type')} | "
      f"beta_anneal_epochs = {(conf.get('spk_contrast') or {}).get('beta_anneal_epochs')}")
PY

# 打印 global batch = batch_size × NPROC（以及每 epoch 的 optimizer step 估算）
"$PYTHON_BIN" - "$RENDERED_CONF" "$NPROC" <<'PY'
import sys

import yaml

conf = yaml.safe_load(open(sys.argv[1]))
nproc = int(sys.argv[2])
dc = conf["datamodule"]["data_config"]
bs = int(dc["batch_size"])
samples = int(dc.get("train_samples", 10000))
accum = int((conf.get("training") or {}).get("accumulate_grad_batches", 1) or 1)
global_batch = bs * nproc
print(f"global batch = batch_size x NPROC = {bs} x {nproc} = {global_batch} "
      f"(x accum {accum} = {global_batch * accum} effective)")
print(f"optimizer steps/epoch ~= train_samples / global_batch = {samples} / {global_batch} "
      f"= {samples // max(global_batch, 1)}")
PY

# 2) 数据自检（只用 $PYTHON_BIN，不用系统 Python）：mix/target/near 三个索引都要在
"$PYTHON_BIN" - "$DATA_ROOT" <<'PY'
import json
import os
import sys

root = sys.argv[1]
missing = []
for split in ("tr", "cv", "tt"):
    for name in ("mix.json", "target.json", "near.json"):
        path = os.path.join(root, split, name)
        if not os.path.isfile(path):
            missing.append(path)
            continue
        with open(path) as f:
            print(f"data ok: {path} ({len(json.load(f))} entries)")
if missing:
    raise SystemExit("MISSING data index files: " + ", ".join(missing))
PY

# 3) 视觉骨干权重自检（路径从渲染后的 conf 读取）
BACKBONE="$("$PYTHON_BIN" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['videonet']['videonet_config']['pretrain'])" "$RENDERED_CONF")"
if [ ! -f "$BACKBONE" ]; then
  echo "backbone not found: $BACKBONE" >&2
  echo "生成：$PYTHON_BIN tools/repro/extract_video_backbone.py --ckpt /root/.cache/falcon_weights/falcon.pt --out $BACKBONE" >&2
  exit 2
fi
echo "backbone ok: $BACKBONE"

EXP_NAME="$("$PYTHON_BIN" -c "import sys, yaml; print((yaml.safe_load(open(sys.argv[1])).get('exp') or {}).get('exp_name', 'exp'))" "$RENDERED_CONF")"
echo "exp_name = $EXP_NAME"

# 4) 启动（PL 2.x 用 `gpus: -1` + CUDA_VISIBLE_DEVICES=N 自动 DDP；训练读渲染后的 conf）
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

LAUNCH_ARGS=(audio_train.py --conf_dir "$RENDERED_CONF")
if [ -n "$CKPT_PATH" ]; then
  LAUNCH_ARGS+=(--ckpt_path "$CKPT_PATH")
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] would launch: $PYTHON_BIN ${LAUNCH_ARGS[*]}"
  echo "[dry-run] rendered conf kept at: $RENDERED_CONF"
  exit 0
fi

setsid nohup "$PYTHON_BIN" "${LAUNCH_ARGS[@]}" > "$EXP_LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/LRS2-paper.pid"
echo "[launch] pid=$PID conf(rendered)=$RENDERED_CONF"
echo "[launch] tail -f $EXP_LOG"
echo "[launch] monitor: $PYTHON_BIN tools/repro/phase2/monitor_lrs2_paper.py --exp $EXP_NAME --watch 300"
echo "$EXP_LOG" > "$LOG_DIR/LRS2-paper.logpath"
echo "$RENDERED_CONF" > "$LOG_DIR/LRS2-paper.confpath"
