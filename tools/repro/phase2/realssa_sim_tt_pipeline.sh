#!/bin/bash
# RealSSA-Sim tt 转换 + 评测流水线（Phase 2-E；默认 dry-run，审批通过后用 --execute）
#
# 前置：RealSSA-Sim 是 gated(manual) 数据集，本机申请已提交、待人工审批（数据文件 403）。
#      获批后：hf auth login（或 HF_TOKEN），并保证网络/代理可用（见 docs/DATA_ACCESS_GUIDE.md §1）。
#
# 用法：
#   bash tools/repro/phase2/realssa_sim_tt_pipeline.sh                 # dry-run：只检查授权 + 打印将执行的命令
#   bash tools/repro/phase2/realssa_sim_tt_pipeline.sh --execute       # 真正执行（1 个 tt shard → 转换 → 小样本评测）
#   SHARD=tt-000005 CHANNEL=0 LIMIT=5 bash .../realssa_sim_tt_pipeline.sh --execute
#   FULL=1 bash .../realssa_sim_tt_pipeline.sh --execute               # tt 全量 19 shards（约 40.1 GB）
#
# 环境变量：SHARD(默认 tt-000000) / CHANNEL(mean|0|1|2|3，默认 mean) / LIMIT(默认 20) / FULL(0|1)
#           RAW_DIR(默认 /root/data/realssa_sim_raw) / FALCON_DIR(默认 /root/data/realssa_falcon)
#           PROXY(默认 http://127.0.0.1:17897) / LOG(默认 docs/verification/phase2/P2E-sim-pipeline.log)
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$REPO"
ENV=/root/miniforge3/envs/falcon/bin/python
SHARD="${SHARD:-tt-000000}"
CHANNEL="${CHANNEL:-mean}"
LIMIT="${LIMIT:-20}"
FULL="${FULL:-0}"
RAW_DIR="${RAW_DIR:-/root/data/realssa_sim_raw}"
FALCON_DIR="${FALCON_DIR:-/root/data/realssa_falcon}"
PROXY="${PROXY:-http://127.0.0.1:17897}"
LOG="${LOG:-docs/verification/phase2/P2E-sim-pipeline.log}"
CKPT="${CKPT:-/root/.cache/falcon_weights/falcon.pt}"
MODE="dry-run"
[ "${1:-}" = "--execute" ] && MODE="execute"

say() { printf '\n=== %s ===\n' "$*"; }
run() {
  if [ "$MODE" = "execute" ]; then
    echo "+ $*"
    "$@"
  else
    echo "# would run: $*"
  fi
}

mkdir -p "$(dirname "$LOG")"
{
  echo "=================================================================="
  echo "[p2e-sim] $(date -Is) mode=$MODE shard=$SHARD channel=$CHANNEL limit=$LIMIT full=$FULL"
  echo "[p2e-sim] raw=$RAW_DIR falcon=$FALCON_DIR ckpt=$CKPT"
} | tee -a "$LOG"

say "0) 只读检查 gated 授权状态（不下载、不申请；dry-run 也会执行这一步）"
"$ENV" tools/repro/fetch_data.py --dataset realssa-sim --shard "$SHARD" --dry-run --proxy "$PROXY" 2>&1 | tee -a "$LOG"
if [ "$MODE" = "execute" ]; then
  if ! "$ENV" - "$PROXY" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys
os.environ.setdefault("http_proxy", sys.argv[1]); os.environ.setdefault("https_proxy", sys.argv[1])
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
try:
    hf_hub_download(repo_id="JusperLee/RealSSA-Sim", filename="tt/tt-000000.tar",
                    repo_type="dataset", dry_run=True)
    print("ACCESS OK: RealSSA-Sim 已获批（dry-run HEAD 通过）")
except GatedRepoError as exc:
    print("GATED: 仍未获批（403）——等待人工审批，先不要执行下载。")
    raise SystemExit(3)
PY
  then
    echo "[p2e-sim] 未获批，停止（这是预期状态，见 docs/DATA_ACCESS_GUIDE.md §4）。" | tee -a "$LOG"
    exit 0
  fi
fi

say "1) 下载 1 个 tt shard（HF_ENDPOINT 可用镜像；--shard $SHARD）"
run "$ENV" tools/repro/fetch_data.py --dataset realssa-sim --shard "$SHARD" --out "$RAW_DIR" --proxy "$PROXY" 2>&1 | tee -a "$LOG"

say "2) 转换 shard → Falcon folder 布局（--channel $CHANNEL；含 mix/target/mouths + JSON 索引）"
SRC="$RAW_DIR/tt/$SHARD.tar"
if [ "$MODE" = "execute" ]; then
  run "$ENV" tools/repro/realssa_to_falcon.py --src "$SRC" --out "$FALCON_DIR" --splits tt --channel "$CHANNEL" 2>&1 | tee -a "$LOG"
else
  echo "# would run: $ENV tools/repro/realssa_to_falcon.py --src $SRC --out $FALCON_DIR --splits tt --channel $CHANNEL" | tee -a "$LOG"
  echo "# 可先加 --dry-run 只检查 shard 配对/计数：" | tee -a "$LOG"
  echo "#   $ENV tools/repro/realssa_to_falcon.py --src $SRC --out /tmp/realssa_dry --splits tt --channel $CHANNEL --dry-run" | tee -a "$LOG"
fi

say "3) 小样本评测（官方权重；--limit $LIMIT；SI-SNRi/SDRi 按同通道 target）"
run env PYTHONPATH="$REPO" "$ENV" audio_visual_test.py --conf_dir configs/Falcon.yml \
  --data_dir "$FALCON_DIR/tt" --ckpt_path "$CKPT" --gpus 0 --limit "$LIMIT" 2>&1 | tee -a "$LOG"

if [ "$FULL" = "1" ]; then
  say "4) tt 全量（19 shards ≈40.1 GB）"
  run "$ENV" tools/repro/fetch_data.py --dataset realssa-sim --pattern 'tt/*.tar' --out "$RAW_DIR" --proxy "$PROXY" 2>&1 | tee -a "$LOG"
  if [ "$MODE" = "execute" ]; then
    run "$ENV" tools/repro/realssa_to_falcon.py --src "$RAW_DIR" --out "$FALCON_DIR" --splits tt --channel "$CHANNEL" 2>&1 | tee -a "$LOG"
  else
    echo "# would run: $ENV tools/repro/realssa_to_falcon.py --src $RAW_DIR --out $FALCON_DIR --splits tt --channel $CHANNEL" | tee -a "$LOG"
  fi
  run env PYTHONPATH="$REPO" "$ENV" audio_visual_test.py --conf_dir configs/Falcon.yml \
    --data_dir "$FALCON_DIR/tt" --ckpt_path "$CKPT" --gpus 0 2>&1 | tee -a "$LOG"
fi

say "完成（mode=$MODE）"
echo "[p2e-sim] $(date -Is) done; log=$LOG" | tee -a "$LOG"
[ "$MODE" = "dry-run" ] && echo "[p2e-sim] 这是 dry-run：未下载/未转换/未评测。审批通过后加 --execute。" | tee -a "$LOG"
exit 0
