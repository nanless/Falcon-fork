# P2D 报告：LRS2-2Mix 训练复现（Phase 2-D，task-21）

> 分支：`repro/phase2`（未 commit）。环境：`/root/miniforge3/envs/falcon/bin/python`
> （Python 3.11.17 / torch 2.8.0+cu128 / pytorch-lightning 2.6.6 / 2×A800 共享机）。
> 写作用域：`configs/LRS2_2Mix.yml`、`tools/repro/phase2/`、`docs/verification/phase2/`。
> **状态：进行中** —— 配置 + 2-GPU 冒烟已完成；正式训练等待 P2-C 全量数据（`/root/data/falcon_lrs2_2mix`）
> 与 GPU 可用窗口；评测脚本已就绪。

## 1. 结论（阶段）

- `configs/LRS2_2Mix.yml` 已建立并通过 2-GPU 100-step 冒烟：**2.97 it/s（global batch 8 → ≈23.8 clips/s）**，
  train/val/test 指标可记录、ckpt/conf/best_model 正常落盘、F9 修复有效（conf 保留 `repeats/shared`）。
- 100 step 时 **val SI-SDR ≈ 1.95 dB / test ≈ 1.91 dB**（从零初始化，仅 35 s 训练；曲线见 §3）。
- 正式训练（10000 样本/epoch × 100 epoch，2×A800 batch 4/GPU）预计 **≈7 min/epoch、满 100 epoch ≈11.7 h**；
  1–2 h 可覆盖约 8–17 个 epoch（early stop patience=30 大概率不会在 1–2 h 内触发）。
- 阻塞项：① P2-C 数据尚未落盘；② 机器 2×A800 当前被其它 Phase2 评测任务占用（见 §5），
  2 卡同时训练可能 OOM —— 需要 Lead 协调 GPU 时间窗（或先用单卡跑曲线）。

## 2. 配置（`configs/LRS2_2Mix.yml`，基于 `configs/Falcon.yml`）

| 项 | 值 | 说明 |
|---|---|---|
| `datamodule.data_config.{train,valid,test}_dir` | `/root/data/falcon_lrs2_2mix/{tr,cv,tt}` | P2-C 目标格式（P2-C 未就绪前不可用） |
| `batch_size` | 4 | 每 rank/GPU；2 卡 global batch=8 |
| `train_samples` | 10000 | 1 epoch = 10000 随机 2 s 片段（2500 step 单卡 / 1250 step 双卡） |
| `num_workers` / `persistent_workers` / `pin_memory` | 8 / true / true | 与 Falcon.yml 一致 |
| `use_near` | true | P2-C 默认 near.json=fallback（K=0），无真实近场负样本 |
| `videonet.pretrain` | `/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar` | T5 从官方 `falcon.pt` 导出的视觉骨干（11,186,688 参数，requires_grad 全 False；实测 forward `[1,1,50,88,88]→[1,512,50]`） |
| `training.gpus / epochs / early_stop` | `[0,1]` / 100 / monitor=val_loss/dataloader_idx_0, patience=30 | 与 README 训练协议一致 |
| `training.logger` | `csv` | 日志写入 `<cwd>/Experiments/checkpoint/LRS2_2Mix/version_0/` |
| `exp.exp_name` | `LRS2_2Mix` | 结果目录名 |
| optimizer / scheduler | AdamW lr=1e-3 wd=0.01；ReduceLROnPlateau(patience=15, factor=0.5) | 与 Falcon.yml 一致 |
| loss / spk_contrast | PIT(neg-snr/neg-sisdr)、β 0→1.0 over 200 epoch | 与 Falcon.yml 一致 |

**target 约定**：P2-A/B 已确证 LRS2-2Mix 发布数据 `mix = s1 + s2`，且 target=s1 明显更优；本复现固定
**target = s1**（`configs/LRS2_2Mix.yml` 的数据由 P2-C 按该约定生成）。

**near 语义（重要）**：LRS2-2Mix 无“近场干扰人”源；P2-C 的 near.json 是 fallback `[[mix,n]]`，
Dataset 会跳过自引用 ⇒ **训练期 K=0**：NASS 对比损失只剩正项 `α_pos(1−cos(q,p))`，负项恒 0；
β annealing 对负项无作用。这与论文 LRS2 训练配方可能不一致，属数据可用性限制，须在结果解读中说明。

## 3. 冒烟结果（2×A800，`/root/data/lrs2_2mix/index_smoke` 3/3/3 条真实 LRS2-2Mix 数据）

命令（`/root/tmp/p2d_smoke` 为独立 cwd，不污染仓库）：

```bash
cd /root/tmp/p2d_smoke
PYTHONPATH=/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python /root/code/github_repos/Falcon-fork/audio_train.py \
  --conf_dir /root/code/github_repos/Falcon-fork/tools/repro/phase2/smoke_lrs2.yml
```

| 项 | 结果 |
|---|---|
| 训练步数 | 100（= train_samples 800 / batch 4 / 2 GPU） |
| 纯训练耗时 / 吞吐 | **≈35 s → 2.97 it/s**（global batch 8 ⇒ ≈23.8 clips/s） |
| 全程 wall time | 73 s（含数据 init、模型/骨干构建、val/test、ckpt 保存） |
| train_loss_epoch | -8.295（= −SNR ≈ 8.3 dB，符合从零初始化早期） |
| train_spk_contrast_loss | ≈0.0014（K=0 → 仅正项，量级极小） |
| val_pit_sisnr（= −val_loss） | **1.946 dB** |
| test_pit_sisnr | 1.911 dB |
| 产物 | `conf.yml`（保留 repeats/shared、无注入 causal，F9 正常）、`epoch=0.ckpt`、`last.ckpt`、`best_k_models.json`、`best_model.pth`、`version_0/metrics.csv` |
| 日志 | `docs/verification/phase2/P2D-smoke.log` |

吞吐外推（正式配置，双卡）：1250 step/epoch ÷ 2.97 ≈ **7.0 min/epoch**；100 epoch ≈ 11.7 h
（dataloader 与冻结 FRCNN 视频前向为主要开销；实际值以训练日志为准）。

**单卡吞吐补充测量（GPU0，2026-10-21 19:08–19:13 UTC，共享机）**：`smoke_lrs2_1gpu.yml`（bs4、100 step）
实测 **0.58 it/s（≈2.3 clips/s，纯训练 195 s）**，明显低于双卡 2.97 it/s——原因是同时段其它 Phase2 评测
进程占用 CPU/GPU（util 0–10% 但持有 ~55–73 GB 显存），并非单卡固有性能；仅作为“先出曲线”的兜底方案。
单卡正式训练外推：2500 step/epoch ÷ 0.58 ≈ **72 min/epoch**（若资源恢复会显著加快，实际以日志为准）。

**batch=8/GPU 的 A/B 冒烟失败说明**：`P2D-smoke-bs8.log` 显示 `torch.OutOfMemoryError`，但当时
GPU0/GPU1 已被其它 Phase2 评测进程占用（分别约 57/73 GB in use，仅剩 ~25/9 GB），**属共享机资源竞争，
不能据此判断 batch 8 在该模型上不可行**；正式配置仍按任务要求用 batch 4/GPU。

## 4. 正式训练（已启动）

- **启动时间**：2026-10-21T19:26:41Z（UTC）
- **命令**：`bash tools/repro/phase2/start_p2d_single_gpu.sh 4` → 内部
  `CUDA_VISIBLE_DEVICES=0 python audio_train.py --conf_dir tools/repro/phase2/LRS2_2Mix_1gpu.yml`
- **PID**：4002894（`docs/verification/phase2/P2D-train.pid`）
- **日志**：`docs/verification/phase2/P2D-train.log`；工作目录 `/root/tmp/p2d_train`
  （产物在 `/root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/`）
- **GPU/显存**：GPU0（外部进程 pid 2514085 占 55.6 GB），本训练增量约 **7.8 GB**（合计 63.5/81.9 GB），无 OOM
- **配置摘要**：单卡 bs4（global 4）、train_samples=10000 ⇒ **2500 step/epoch**；epochs=100、early stop patience=30；
  AdamW lr=1e-3/wd=0.01、ReduceLROnPlateau(15, 0.5)、grad clip 5、sync_bn；CSVLogger；exp=`LRS2_2Mix_1gpu`
- **吞吐/进度（19:30 UTC）**：已到 step ≈750，估算 **≈5–6 it/s**（首 epoch 预计 8–13 min，含 cv 5000 + tt 3000
  的整句验证遍历；实际以 epoch 完成日志为准）
- 训练启动即验证：`accelerator=gpu devices=[0] num_devices=1 strategy=auto`、audio 537K trainable / video 11.2M frozen

## 4.1 正式训练（原计划，保留）

- 启动脚本：`tools/repro/phase2/run_p2d_train.sh`（setsid + PID + 日志；前置检查 P2-C 的
  `{tr,cv,tt}/{mix,target}.json` 非空，缺数据直接 ABORT）。
- 预期产物：`docs/verification/phase2/P2D-train.log`、`P2D-train.pid`、`/root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix/`。
- 启动条件：**P2-C 全量数据落盘** + **GPU 可用**（见 §5）。启动后立即向 Lead 汇报 PID/配置/吞吐。
- **Lead 裁决（19:13 / 19:18 UTC 更新）**：等 P2-C 数据（预计 10–20 min），然后**固定单卡 GPU0 + bs4** 启动：
  命令 = `CUDA_VISIBLE_DEVICES=0` + `tools/repro/phase2/LRS2_2Mix_1gpu.yml`（gpus=[0]，exp=`LRS2_2Mix_1gpu`，
  其余超参与 `configs/LRS2_2Mix.yml` 相同）；预计 **≈14 min/epoch**，先跑 1–2 h 看曲线。
  若 OOM（GPU0 仅剩 ~26 GB）→ 降 bs2（配置无 AMP 开关）；启动后立即汇报 PID/日志/吞吐/显存，每 ~30 min 更新。
  GPU 现状：占显存的是**外部进程 pid 2514085（两卡共 ~127 GB，非团队任务）**，无法释放；GPU1 仅剩 ~9 GB，
  故不双卡。数据就绪且 GPU1 释放后可考虑重启双卡，否则单卡跑到底。
- **评测脚本已端到端验证**（19:17 UTC）：用 100-step 冒烟 ckpt 在 `/root/tmp/lrs2_eval_tt_full` 上 `--limit 5`
  跑通 `eval_p2d.sh` → `docs/verification/phase2/P2D-eval-epoch_0.csv`（5 条：sdr 3.58 / sdr_i -1.78 /
  si-snr 2.81 / si-snr_i -2.49；未训练基线，仅证明管线可用，不作为结果）。

## 4.2 训练进度与首轮评测（进行中）

| 指标（epoch 0，2500 step） | 值 |
|---|---|
| train_loss_epoch（最后记录） | ≈ −8.0（= −SNR ≈ 8 dB） |
| train_pit_loss / spk_contrast | ≈ −8.0 / ≈ 1e-4（K=0，仅正项） |
| **val_pit_sisnr**（= −val_loss，绝对 SI-SDR） | **6.7325 dB** |
| test_pit_sisnr（每 10 epoch 的 tt 记录） | 6.8147 dB |
| epoch 时间（首 epoch，含模型构建/验证） | ≈ 17 min（train ≈9–10 min + cv5000+tt3000 验证遍历 ≈6–7 min） |
| 稳定训练吞吐（epoch 1 实测） | ≈4.0–4.6 it/s ⇒ 2500 step ≈ 9–10 min/epoch（+ 验证） |

**首轮 ckpt 评测（`epoch=0.ckpt`，300 条 tt 子集，GPU1；`P2D-eval-epoch_0.csv`）vs 官方权重同子集**
（`P2D-eval-falcon.pt.csv`，同协议/同 300 条）：

| 指标 | 训练 1 epoch | 官方 falcon.pt | 差 |
|---|---|---|---|
| SDR | 6.87 | 12.53 | −5.66 |
| SDRi | 1.83 | 7.49 | −5.66 |
| SI-SNR | 6.46 | 11.86 | −5.40 |
| SI-SNRi | 1.52 | 6.92 | −5.40 |
| PESQ | 1.43 | 2.28 | −0.85 |
| STOI | 0.804 | 0.908 | −0.10 |

结论（阶段性）：1 epoch/2500 step 后 val SI-SDR 已到 6.73 dB，但相对官方权重仍有 ~5.4 dB 差距（符合“远未训完”的预期；
官方权重是在 LRS2-2Mix 上用完整配方训练的结果）。**1–2 h 只能覆盖 4–7 epoch，远达不到 100 epoch 配方**；
是否需要长跑（单卡约 28 h/100 epoch）由 Lead/用户决定。全量 3000 条评测将在训练 1–2 h 后做（保证与 P2-A/B
的官方基线同协议同子集可比）。

## 5. 阻塞与风险

1. **P2-C 数据未就绪**（task-20 进行中）：`/root/data/falcon_lrs2_2mix` 尚不存在；正式训练无法开始。
2. **GPU 资源竞争**：本机 2×A800 被其它 Phase 2 评测任务占用（`nvidia-smi` 实测 GPU0 57 GB、
   GPU1 73 GB 已用），2 卡训练可能 OOM（bs8 冒烟即由此失败）。需要 Lead 协调时间窗，或：
   - 先用单卡 GPU0（当前约 25 GB 空闲）跑正式训练曲线（吞吐减半，≈14 min/epoch）；
   - 或等其它任务结束后再双卡启动。
3. **LRS2 无 near 负样本**（§2）：K=0，对比损失负项失效；若要与论文配方对齐，需要 P2-C 的
   `--near-mode raw_audio` 或另做负样本挖掘（另立实验）。
4. **评测协议 7 dB 差距**：官方权重在 `/root/tmp/lrs2_eval_tt_full` 上实测 SI-SNRi 7.03 / SDRi 7.71，
   与论文 14.1/14.7 有差距，P2-A/B/F 正在排查（指标定义/目标缩放/口型输入/整句 vs 窗口等）。
   本任务用**同一评测脚本**对比训练 ckpt 与官方权重，差距会同时体现在两者上，报告会标注该协议基线。

## 6. 后续评测计划（脚本已就绪）

`tools/repro/phase2/eval_p2d.sh <ckpt> [data_dir=/root/tmp/lrs2_eval_tt_full] [limit] [gpus=0]`：
- 用 `audio_visual_test.py --conf_dir configs/LRS2_2Mix.yml --ckpt_path <训练 ckpt> --data_dir /root/tmp/lrs2_eval_tt_full`；
- 输出 `docs/verification/phase2/P2D-eval-*.csv`；
- 对照：官方权重（7.03/7.71，P2-A/B 基线）、论文（14.1/14.7）、以及“未训练但用官方视觉骨干”的基线。

## 7. 复跑命令汇总

```bash
# 冒烟（100 step）
cd /root/tmp/p2d_smoke && PYTHONPATH=/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python /root/code/github_repos/Falcon-fork/audio_train.py \
  --conf_dir /root/code/github_repos/Falcon-fork/tools/repro/phase2/smoke_lrs2.yml

# 正式训练（单卡 GPU0，Lead 批准方案；含数据门禁 + GPU 快照 + PID）
bash /root/code/github_repos/Falcon-fork/tools/repro/phase2/start_p2d_single_gpu.sh 4
# （OOM 时降 bs2：bash .../start_p2d_single_gpu.sh 2）
# 双卡（若 GPU1 释放后）：CUDA_VISIBLE_DEVICES=0,1 bash .../run_p2d_train.sh configs/LRS2_2Mix.yml

# 评测
bash /root/code/github_repos/Falcon-fork/tools/repro/phase2/eval_p2d.sh <ckpt> /root/tmp/lrs2_eval_tt_full 0
```

---
*本报告将在正式训练启动/1–2 h 曲线/评测完成后继续更新（含 P2D-train.pid、CSV 曲线、SI-SNRi/SDRi 对比）。*

---

## 8. 2026-10-24 终态校正

> **本节取代 §1「结论（阶段）」、§4「正式训练（已启动）」、§4.2「训练进度与首轮评测（进行中）」中的
> "进行中 / 等待 P2-C 数据 / 等待 GPU 窗口" 等描述**；其余章节作为历史记录保留。本节数字全部来自
> 已落盘、可复核的证据：`P2D-train-metrics-1gpu.csv`、`P2D-near-train-metrics.csv`、
> `P2D-eval-{falcon-full,epoch86,epoch99,epoch124-contrastive}.{csv,log}`。

### 8.1 训练终态（已完成，不再是"进行中"）

| 项 | 值 | 证据 |
|---|---|---|
| 正式训练（无 near） | **100/100 epoch，2026-10-22 19:54 UTC 结束** | `epoch=99.ckpt`/训练目录 mtime 19:54；`P2D-train-metrics-1gpu.csv` 最后一行 epoch=99、step=249999 |
| 规模 | 单卡 bs4，2500 step/epoch，共 249,999 step | 同上 |
| **best val_pit_sisnr** | **ep99 = 13.2014 dB**（§4.2 中"最佳 val 13.118 / ep86"只是当时的最佳，已被 ep99 超过） | `best_k_models.json`：`epoch=99.ckpt: -13.201412`；metrics ep99 `13.201422` |
| top-5 ckpt（val_pit_sisnr） | ep86 13.1182 / ep96 13.1383 / ep97 13.1149 / ep98 13.0379 / **ep99 13.2014** | `best_k_models.json` |
| test_pit_sisnr（tt，每 10 epoch） | ep0 6.8147 / ep10 10.4094 / ep20 11.5897 / ep30 12.1568 / ep40 12.5586 / ep50 12.8444 / ep60 13.1192 / ep70 13.1067 / ep80 13.3705 / ep90 13.3303 | `P2D-train-metrics-1gpu.csv` |
| contrastive 微调（s2 作 near，K=1） | epoch 100→129（从 ep99 ckpt 续训），**best val ep124 = 13.3909**；test ep120 = 13.7618 | `P2D-near-train-metrics.csv`（`/root/tmp/p2d_near_train/.../LRS2_2Mix_near_s2`，30 个 epoch 行） |

### 8.2 LRS2-2Mix 全量评测终态（tt 3000 条，target=s1，整句推理）

协议：`audio_visual_test.py --conf_dir configs/LRS2_2Mix.yml --data_dir /root/data/falcon_lrs2_2mix --gpus <0|1>`；
单卡、整句（不切窗）、逐句计算后取均值；PESQ/STOI 用同一脚本实现。全部 4 行**同协议、同 3000 条、target=s1**，
可直接横向比较。

| 权重 | SI-SNR | **SI-SNRi** | SDR | **SDRi** | PESQ | STOI | 证据（逐句 CSV / 日志） |
|---|---:|---:|---:|---:|---:|---:|---|
| 官方 `falcon.pt`（跨语料：LRS3 训练） | 12.023 | **7.029** | 12.813 | **7.713** | 2.196 | 0.916 | `P2D-eval-falcon-full.csv/.log`（3000 句，本次补跑） |
| 自训 ep86（无 near） | 13.612 | **8.618** | 14.142 | **9.042** | 2.547 | 0.941 | `P2D-eval-epoch86.log`（历史日志；逐句 CSV 于 2026-10-30 重跑恢复为 `P2D-eval-epoch86.csv`，3,003 行） |
| 自训 **ep99（best val，本次补测）** | 13.720 | **8.726** | 14.222 | **9.122** | 2.578 | 0.944 | `P2D-eval-epoch99.csv/.log`（3000 句） |
| 自训 ep124（contrastive 微调） | 13.852 | **8.858** | 14.370 | **9.270** | 2.576 | 0.943 | `P2D-eval-epoch124-contrastive.csv/.log`（3000 句） |
| 论文 LRS2-2Mix（README/论文） | — | 14.1 | — | 14.7 | — | — | README |

跨句标准差（本次两份新 CSV 的 `std` 行）：官方 SI-SNRi 2.932 / SDRi 2.728；ep99 SI-SNRi 2.452 / SDRi 2.466。
两份新 CSV 均为 3002 行（3000 逐句 + `avg` + `std`）。

精确值（逐句 avg 行；顺序 SDR / SDRi / SI-SNR / SI-SNRi / PESQ / STOI）：
- 官方 `falcon.pt`：`12.813387 / 7.712925 / 12.022821 / 7.029101 / 2.195634 / 0.916350`
- 自训 ep99：`14.222311 / 9.121848 / 13.719868 / 8.726148 / 2.577971 / 0.943988`
- 自训 ep86（2026-10-30 重跑，逐句 CSV 已恢复）：`14.141969 / 9.041506 / 13.611639 / 8.617919 / 2.546623 / 0.940860`（与历史日志完全一致）
- 自训 ep124（contrastive 微调）：`14.370037 / 9.269574 / 13.851912 / 8.858192 / 2.576086 / 0.942796`

### 8.3 结论（更新版）

1. **自训最佳 ckpt（ep99）首次全量评测：SI-SNRi 8.726 / SDRi 9.122**，比 ep86（8.618/9.042）高
   **+0.108 / +0.080 dB**，与 val 曲线的 best（ep99）方向一致；ep124 contrastive 微调再提升
   **+0.132 / +0.148 dB** → **8.858 / 9.270**（这是"近似 s2 负样本"的增益，非论文原始 near 模拟）。
2. **官方权重跨语料基线复跑确认：7.029 / 7.713**，与 Phase-1 的 LRS2 全量评测数字一致（同一脚本、同一 3000 条），
   说明该基线可复现，不是一次性波动。
3. 自训 ckpt 比官方跨语料权重高 **+1.697 SI-SNRi / +1.409 SDRi**（同数据同协议），符合"已在 LRS2 上训练"的预期；
   但距论文 14.1 / 14.7 仍差 **−5.374 / −5.578 dB**，归因不变：单卡 bs4（论文 10 卡、全局 batch 40）、
   100（+30）epoch、LRS2 无 near 导致负项失效或只能近似，以及论文训练配方（数据混合 / lr schedule / 轮数）未公开。
4. 证据完整性：ep86 已于 2026-10-30 补跑（`P2D-eval-epoch86.csv`），**官方 / ep86 / ep99 / ep124 四份均有 3000 句逐句 CSV**；
   两条训练曲线已入库（`P2D-train-metrics-1gpu.csv` 5200 行 / `P2D-near-train-metrics.csv` 1560 行）。

### 8.4 本次评测执行记录（可复核）

- 2026-10-24 13:52 首轮（两卡并行、默认线程）因**线程超订**（每进程 300+ 线程）长时间未产出首行 CSV，
  被终止，**无结果 CSV 被引用**；
- 2026-10-24 14:17 由 Lead 在干净环境重启：`python -u`、`OMP_NUM_THREADS=MKL_NUM_THREADS=4`、
  各自独立 cwd（`/root/tmp/p2d_eval_official`、`/root/tmp/p2d_eval_ep99`）与独立 GPU（0/1），
  **两份均于 14:26Z 完成（exit 0）**，各约 9 min；
- 结果复制：`/root/tmp/p2d_eval_{official,ep99}/Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv`
  → `docs/verification/phase2/P2D-eval-{falcon-full,epoch99}.csv`（Lead 复制；md5 与源文件逐一相同，
  `ff9b9d74…` / `5a7e307f…`，未改动 CSV 内容）。

复跑命令（与本次一致）：

```bash
# 官方权重全量（独立 cwd）
mkdir -p /root/tmp/p2d_eval_official && cd /root/tmp/p2d_eval_official
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python -u /root/code/github_repos/Falcon-fork/audio_visual_test.py \
  --conf_dir /root/code/github_repos/Falcon-fork/configs/LRS2_2Mix.yml \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt \
  --data_dir /root/data/falcon_lrs2_2mix --gpus 1

# ep99（best val）全量
mkdir -p /root/tmp/p2d_eval_ep99 && cd /root/tmp/p2d_eval_ep99
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python -u /root/code/github_repos/Falcon-fork/audio_visual_test.py \
  --conf_dir /root/code/github_repos/Falcon-fork/configs/LRS2_2Mix.yml \
  --ckpt_path /root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/epoch=99.ckpt \
  --data_dir /root/data/falcon_lrs2_2mix --gpus 1
```

### 8.5 与 §6「后续评测计划」的关系

§6 的评测脚本 `tools/repro/phase2/eval_p2d.sh` 仍可用；但**本次全量评测用的是 `audio_visual_test.py`
直调 + 独立 cwd**（避免同名 `Experiments/checkpoint/LRS2_2Mix/results/` 互相覆盖，这正是此前 CSV 被覆盖的原因）。
后续若再用 `eval_p2d.sh`，请先确认它的 cwd 与已有证据不冲突。

---
*[2026-10-24] 本节为终态校正；`P2D-*` 证据文件与两条训练曲线已落盘可复核。*
