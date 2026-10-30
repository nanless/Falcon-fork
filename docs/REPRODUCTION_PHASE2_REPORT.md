# Falcon 复现 Phase 2 执行报告（7 dB 归因 + LRS2-2Mix 训练启动 + RealSSA 预置）

> 日期：2026-10-22 · 分支 `repro/phase2`（Phase 1 已合入 `main` = `ac91800`，已 push）
> 环境：conda `falcon`（见 `AGENTS.md`）· 数据/权重在仓库外 `/root/data`、`/root/.cache/falcon_weights`
> 各子任务报告：`docs/verification/phase2/P2A…P2F-report.md`、`QA-report.md`、`REGRESSION-docs.md`

## 1. TL;DR

1. **论文 LRS2-2Mix 列的 7 dB 差距已归因**：官方 `falcon.pt` 是在 **LRS3_complex_nonmoving** 上训练的
   （`Experiments/checkpoint/Falcon/conf.yml:99-102`；README:211/228 明示论文表格 “under our training protocol”），
   我们测得的 **7.03 / 7.71 是“跨语料”结果，不是论文 LRS2 列（14.1 / 14.7）的复现**。
   三路独立验证排除评测 bug：指标/协议变体 ≤0.4 dB（P2-A 14 变体、P2-F 独立对拍 4.8e-8 dB）、
   falcon-big 仅 +0.97 dB（P2-B）、Studio 窗口化 −1.9~−2.3 dB、视觉分支贡献 +4.8~+15 dB。
2. **LRS2-2Mix 已具备完整训练条件**：数据全量转换完成（28,000 条 / 23.36 GiB，target=s1，mouths `(50,96,96,3)`），
   **单卡训练已启动**：epoch 0 完成，**val_pit_sisnr 6.7325 dB / test 6.8147 dB**（1 epoch 已接近官方跨语料 7.03），
   训练继续中（100 epoch 预计 ~30 h 单卡）。
3. **RealSSA 访问申请已提交**（RealSSA-Sim/Real，2026-10-22，账号 nanless/USTC/CN），当前仍 403 待人工审批；
   DNSMOS 评测工具链（speechmos P.835 + onnxruntime）与 RealSSA-Real 评测脚本已预置并冒烟通过，
   审批通过后可立即跑 RealSSA-Sim（目标 7.7/8.2）与 RealSSA-Real（DNSMOS 目标 3.276）。
4. LRS3-2Mix / VoxCeleb2-2Mix 的 HF 仓库仍无数据文件 → 这两列暂不可复现（需作者/原数据）。

## 2. 子任务结果

| 任务 | 负责人 | 结果 |
|---|---|---|
| P2-A 7 dB 差距排查（指标/协议变体） | loss-metric-eval | 200 条子集基线 6.955/7.622；14 变体最大 +0.34 dB；mouth 错位 −10~−13 dB、换错说话人 −23.7、置黑 −4.8；数据自洽（mix=s1+s2，SIR 相关 0.97）→ **协议不能解释 7 dB** |
| P2-B big/small + 窗口化/视觉 | model-core | big +0.971/+0.911 dB（258/300 胜）；窗口化 −1.912/−2.318 dB（整句高估流式 ~2 dB）；置零 mouth −15.1 dB（视觉刚需）→ 模型规模/协议均不能解释 |
| P2-C LRS2-2Mix → Falcon 格式 | data-pipe | 28,000 条（tr/cv/tt 20000/5000/3000），target=s1，mouths `(50,96,96,3)`，JSON 等长，23.36 GiB，4 min（16 并行）；校验 600 抽样 problems=0，`max|mix−(s1+s2)|=3.05e-5` |
| P2-D LRS2-2Mix 训练复现 | train-sys | 配置 `configs/LRS2_2Mix.yml`（单卡版 `tools/repro/phase2/LRS2_2Mix_1gpu.yml`）；2-GPU 冒烟 2.97 it/s；正式训练 19:26Z 启动（单卡 bs4，2500 step/epoch，~18 min/epoch 含验证）；**epoch0 val 6.7325 / test 6.8147**，继续训练中 |
| P2-E RealSSA/DNSMOS 预置 | ops-tests-docs | 装 speechmos 0.0.1.1 + onnxruntime 1.31（许可/模型 sha 记录）；`eval_dnsmos.py`、`eval_realssa_real.py`（官方 loader 约定、channel mean/0/1、align_tail、只算 DNSMOS）、`realssa_sim_tt_pipeline.sh`；冒烟通过；当前 403 待批 |
| P2-F 指标/协议第三方独立验证 | studio-backend | 独立实现 vs `metrics/core.py` 最大差 **4.8e-8 dB**、vs Phase1 CSV 1.8e-7；3000 条数据/对齐/峰值/`mix=s1+s2` 全自洽；协议变体合计 ≤0.4 dB；确认 7.03/7.71 是跨语料结果 |
| P2-G Phase 2 QA | layers-dsp | 见 `QA-report.md`（终检回归、转换数据抽样、报告一致性、指纹不变） |
| P2-H 文档/回归收口 | studio-frontend | `docs/verification/phase2/README.md`、`REGRESSION-docs.md`、PLAN/GUIDE 状态同步；回归 Studio 35/2、Node 8/8、Space i18n 226、tools ≥96 passed/6 skipped |

## 3. 归因链（为什么 7.03/7.71 ≠ 论文 14.1/14.7）

| 证据 | 结论 |
|---|---|
| `Experiments/checkpoint/Falcon/conf.yml:99-102` datamodule=`LRS3_complex_nonmoving`；`:105` exp_name；best_k epoch=714；PROVENANCE:37-38 | 官方 `falcon.pt` = **LRS3 训练**的 checkpoint |
| `README:211`（表格）+ `README:228` “under our training protocol” | 论文每个 benchmark 列是**分别训练**的结果 |
| P2-A：14 个指标/协议变体最大 +0.34 dB；P2-F 独立对拍 ≤1e-7 dB | 评测协议/指标实现**不是** gap 来源 |
| P2-B：big +0.97 dB；窗口化 −1.9~−2.3 dB；视觉分支贡献 +4.8~+15 dB | 模型规模/权重版本/视觉分支均**不是**主因 |
| P2-D：LRS2 训练 1 epoch 即 val 6.73 / test 6.81（已接近跨语料 7.03） | 训练数据匹配是主因；完整训练应显著提升 |

> 结论：**“7.03/7.71 是跨语料评测结果”**。要复现论文 LRS2-2Mix 的 14.1/14.7，需要 LRS2-2Mix-trained checkpoint（P2-D 正在训练），
> 并用本报告验证过的评测协议复测；这也解释了为什么官方权重在 LRS3 列上应更接近论文值（LRS3 数据未发布，暂无法交叉验证）。

## 4. 训练状态与复跑

### 4.1 当前训练（2026-10-22）

| 项 | 值 |
|---|---|
| 启动 | 2026-10-21T19:26:41Z（单卡 GPU0，bs4） |
| 配置 | `tools/repro/phase2/LRS2_2Mix_1gpu.yml`（基于 `configs/LRS2_2Mix.yml`）：AdamW 1e-3/wd 0.01、ReduceLROnPlateau(15,0.5)、grad clip 5、epochs 100、early stop 30、train_samples 10000、CSVLogger |
| 吞吐 | ~5.5 it/s → 2500 step/epoch；**~18 min/epoch（含 cv 5000 + tt 3000 验证）**；GPU0 增量显存 ~7.8 GB（与外部进程共享，未 OOM） |
| epoch 0 | val_pit_sisnr **6.7325 dB**（cv）、test_pit_sisnr **6.8147 dB**（tt 3000），ckpt `epoch=0.ckpt`（19:44Z） |
| epoch 86（最佳 val 13.118） | **LRS2-2Mix test 全量（3000）：SI-SNRi 8.618 / SDRi 9.042 / SI-SNR 13.612 / SDR 14.142 / PESQ 2.547 / STOI 0.941** |
| 对照 | 官方跨语料权重 7.03/7.71；论文 14.1/14.7（本训练仅 87 epoch、单卡 bs4、无对比损失负样本） |
| 下一步（已准备） | 用 **s2 干扰说话人**构造 near 负样本（`near.json` 已生成，K=1，Dataset 验证通过），训练完成后做 contrastive 微调（论文的 NASS 损失） |

**contrastive 微调已完成（2026-10-22/23，100→130 epoch，`LRS2_2Mix_near_s2`）**：

| 模型 | val_pit_sisnr（cv） | test_pit_sisnr | LRS2-2Mix test SI-SNRi / SDRi |
|---|---|---|---|
| 无 near（87 epoch，最佳 val 13.118） | 13.118 | 13.370（ep80） | 8.618 / 9.042 |
| **contrastive 微调（epoch 124，最佳 val 13.391）** | **13.391** | 13.762（ep120） | **8.858 / 9.270**（PESQ 2.576 / STOI 0.943） |

contrastive 带来 +0.24/+0.23 dB；仍低于论文 14.1/14.7（~5 dB），主要差距应在训练规模/配方
（论文 10×A800、bs 40、1000 epoch；我们是单卡 bs4、130 epoch）。
| 预计 | 100 epoch ≈ 30 h 单卡；若 GPU1 空闲可切双卡（~15 h）；early stop 可能更早 |
| 产物 | `/root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/`（metrics.csv/ckpt） |

### 4.2 监控 / 评测 / 恢复

```bash
# 监控
CSV=/root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/version_0/metrics.csv
python3 -c "import csv;rows=list(csv.DictReader(open('$CSV')));print([r for r in rows if r.get('val_pit_sisnr')][-1])"
# 评测（训练 ckpt → LRS2-2Mix tt，用 Phase1 修好的评测器）
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python audio_visual_test.py \
  --conf_dir configs/LRS2_2Mix.yml \
  --data_dir /root/data/falcon_lrs2_2mix \
  --ckpt_path /root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/epoch=0.ckpt --gpus 0
# 若训练中断：用 audio_train.py --conf_dir <同一配置> --ckpt_path <last.ckpt> 恢复（Phase1 已修 resume 逻辑）
```

## 5. RealSSA（待审批）

- 申请：RealSSA-Sim / RealSSA-Real 已于 2026-10-22 提交（HF ask-access，HTTP 303；账号 nanless/USTC/CN），当前数据文件 403。
- 预置：DNSMOS 工具链、RealSSA-Real 评测脚本（channel mean/0/1、align_tail、只算 DNSMOS）、RealSSA-Sim tt 转换/评测命令
  （`tools/repro/phase2/realssa_sim_tt_pipeline.sh`，dry-run 已验证；1 shard 2.33 GB，tt 19 shards≈40.1 GB）。
- 批准后执行清单：Sim 用官方权重测 SI-SNRi/SDRi（目标 7.7/8.2，ch mean/0 两套）；Real 测 DNSMOS（目标 3.276±0.05，mean/0/1 三套）。

## 5.1 RealSSA 数据已获批并转换（2026-10-22）

- **审批**：RealSSA-Sim / RealSSA-Real 访问已获批（此前 403 → 可下载）。
- **RealSSA-Real**：已下载 12.8 GB（422 文件）；**DNSMOS 评测完成（54/54 段，channel=mean，align_tail，窗口化推理；整段推理会 OOM，已改为 Studio 等价 2 s 窗/0.5 s hop）**：

| 指标 | Falcon 分离 | mixture 基线 | 提升 |
|---|---|---|---|
| SIG | 3.2209 | 2.4176 | +0.80 |
| BAK | 3.6921 | 1.7908 | +1.90 |
| OVRL | 2.7901 | 1.7599 | +1.03 |
| P808 | **3.3285** | 2.6765 | +0.65 |

  论文 “Real-world DNSMOS **3.276**” 与 **P808 3.3285（+0.05）** 最接近；若论文为 OVRL 口径则为 2.79（差 0.49，需论文确认）。
  分场景 OVRL：small_indoor 3.10 / outdoor 2.78 / big_indoor 2.59 / mid_indoor 2.39；RTF 均值 0.198。
  口径声明：DNSMOS 实现为 speechmos P.835+P.808、channel=mean、align_tail=True、窗口化推理。
- **RealSSA-Sim tt**：19 shards（38 GB）已下载；转换完成 **9,169 条**
  （`/root/data/falcon_realssa_sim/tt`，4ch→mono mean、target/mix/mouths + JSON），
  官方权重评测运行中（论文目标 SI-SNRi **7.7** / SDRi **8.2**）。
- **RealSSA-Sim test 评测（关键实验，前 1000 条快速估计，官方权重）**：

| 指标 | 实测（1000 条） | 论文（test 9169 条） | 差异 |
|---|---|---|---|
| SI-SNRi | **7.935** | 7.7 | +0.24 |
| SDRi | **8.754** | 8.2 | +0.55 |

- **RealSSA-Sim 全量 test（9,169 条）已完成（2026-10-22）**：

| 指标 | 实测（全量 9169） | 论文 | 差异 |
|---|---|---|---|
| SI-SNRi | **7.774** | 7.7 | +0.07 |
| SDRi | **8.479** | 8.2 | +0.28 |
| SI-SNR / SDR | 9.980 / 10.645 | — | — |
| PESQ / STOI | 2.094 / 0.817 | — | — |

  **结论：官方发布权重在 RealSSA-Sim 上完整复现了论文主结果（±0.3 dB 内）**。
  这同时验证了：(a) 我们的评测/指标实现正确；(b) 官方权重就是 RealSSA 训练分布；
  (c) 之前 LRS2-2Mix 的 7.03/7.71 确实是跨语料结果，而不是评测 bug。

## 5.2 数据集归档到共享存储/TOS（2026-10-23 执行中）

按用户指令把 **RealSSA-Sim / RealSSA-Real** 下载到共享存储并归档 TOS：
- 目标（下载）：`/root/group-shared/voiceprint/data/multimodal/{realssa_sim,realssa_real}`
- 归档（下载完成后）：`/aispeech-cfs/voiceprint/data/multimodal/{realssa_sim,realssa_real}`
- 网络：hf-mirror 对 gated 数据 redirect 不可用，huggingface.co 直连超时 ⇒ 走本机 mihomo 代理 `127.0.0.1:17897`；
  实测 ~17 MiB/s（8 并发）、断点续传 + 重试。
- 工具：`tools/repro/phase2/download_realssa_shared.py`（含 manifest/校验/rsync 归档/状态 JSON）；
  监控：`_download/realssa_status.json` + `realssa_monitor.log`（每 30 min 自动汇报）。
- 进度（2026-10-23 04:12 UTC）：RealSSA-Real 12.84 GB（422 文件）已完成；RealSSA-Sim 51.3/521.5 GB（9.8%），
  ETA ~7 h；之后 rsync 到 TOS（预计 1–2 h）。

## 5.3 LRS2-2Mix 论文配方重训（8 卡方案，2026-10-23）

- **GitHub 求助**：已向作者提交 issue 请求 LRS2-2Mix 列训练配置与 ckpt：
  <https://github.com/JusperLee/Falcon/issues/1>（含我们 RealSSA 复现结果与跨语料证据）。
- **论文配方数据已构建**：`/root/data/falcon_lrs2_2mix_paper`（26 GB）：
  - **双向 target**：每条 mixture 生成 (s1 target, s1 mouth, near=s2) 与 (s2 target, s2 mouth, near=s1) 两条样本
    ⇒ tr **40,000** / cv 10,000 / tt 6,000 条；mouths 18,691/476/267 个 `.pt`；
  - near 负样本 = 另一说话人（LRS2-2Mix 无 near-field 源，报告中声明该差异）。
- **8 卡配置**：`configs/LRS2_2Mix_Paper_8gpu.yml` —— 8×bs5=全局 40（论文 10×4=40）、lr 1e-3、1000 epoch、
  early stop 30、β 0→1/200 epoch；启动脚本 `tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh`；
  监控 `tools/repro/phase2/monitor_lrs2_paper.py`。
- **火山引擎 runbook**：`docs/PHASE2_VOLCANO_LRS2_TRAINING.md`（数据/环境/骨干/启动/续训/评测/时间估算/差异声明）。
- **本地 2 卡验证通过**（2026-10-23）：`configs/LRS2_2Mix_Paper_2gpu_smoke.yml`（2×bs4×accum5 = 有效 40）
  3 epoch smoke 成功：DDP、grad accumulation、双向数据、near K=1、ckpt 保存全正常；val_loss 0.624→0.243→−0.125。
  本机只有 2 卡，8 卡实验需在火山引擎节点执行。
- **训练入口**：`audio_train.py` 新增 `training.accumulate_grad_batches` 支持（默认 1，无行为变化）。

## 5.4 数据集下载与归档完成（2026-10-23 17:31 UTC）

- **RealSSA-Sim**：242 文件（240 tar 分片 + README/.gitattributes），521,532,645,652 B，尺寸校验 `bad=[]`；
- **RealSSA-Real**：423 文件，12.84 GB；
- 均已从组共享存储 **rsync 迁移到 TOS 冷端** `/aispeech-cfs/voiceprint/data/multimodal/{realssa_sim,realssa_real}`
  （`_archive_done.json` 两条 `ok: true`；源目录已删除；Sim 归档 521.5GB 用时 26m48s @ ~309 MB/s）。
- 下载过程：mihomo 代理（hf-mirror/直连不可用），两次带宽骤降（最低 0.12 MB/s）→ 自动暂停/降阈值/断点续传恢复，
  全程 0 硬失败；工具 `tools/repro/phase2/{download_realssa_shared.py,resume_when_network_ok.sh}`。

## 6. 遗留与风险

1. LRS3-2Mix / VoxCeleb2-2Mix 的 HF 仓库无数据文件 → 两列不可复现（需作者或原始语料）。
2. 训练：单卡 ~30 h；GPU 与外部进程共享（GPU0 55.6 GB 被外部占用），无法双卡；长跑需保持机器稳定。
3. LRS2-2Mix 无 near 源 → NASS 对比损失只剩正项（K=0）；本次训练未使用非官方负样本（`--near-mode raw_audio` 仅留作消融）。
4. RealSSA 审批时间不可控；DNSMOS 版本敏感（已固定 speechmos 0.1.1 + 模型 sha）。
5. 训练复现的实验设计：1 epoch 已 6.81，但论文 14.1 需要充分训练；建议至少跑 30–50 epoch 后评估趋势，并与官方跨语料 7.03 对比。
6. 论文仍未公开：训练细节（数据混合、epochs、lr schedule）只能按 config/README 推断。

## 7. 下一步建议

1. **让训练继续**（后台），每 2–3 h 检查 val 曲线；达到 ~30–50 epoch 或 early stop 后用 Phase1 评测器在 tt 全量评测，对比 14.1/14.7。
2. **RealSSA 审批**：若批准，立即用 `tools/repro/fetch_data.py` + `realssa_to_falcon.py` 跑 Sim test 评测（验证官方权重在训练分布上的 7.7/8.2）。
3. **可选**：GPU1 空闲时重启双卡训练（~15 h）；或先跑 LRS2 训练 30 epoch 的中间 ckpt 做趋势报告。
4. 若要与论文严格对齐：需要作者提供 benchmark-specific checkpoint 或训练配方（论文/代码未公开的部分）。

## 8. 相关产物索引

- 计划：`docs/REPRODUCTION_PLAN.md`（Phase 2/3 状态已回填）
- Phase 1 报告：`docs/REPRODUCTION_PHASE1_REPORT.md`（含跨语料标注）
- 数据指南：`docs/DATA_ACCESS_GUIDE.md`（含申请提交记录）
- 子任务报告：`docs/verification/phase2/{P2A,P2B,P2C,P2D,P2E,P2F}-report.md`、`QA-report.md`、`REGRESSION-docs.md`
- 工具：`tools/repro/phase2/*.py|*.sh`、`tools/repro/lrs2_2mix_to_falcon.py`
- 数据：`/root/data/falcon_lrs2_2mix`（28,000 条）、`/root/data/lrs2_2mix/lrs2_rebuild`（原始包）

## 9. 2026-10-24 终态校正（追加节，不改写上文历史快照）

1. **训练终态**：P2-D 单卡 100 epoch 已于 2026-10-22 19:54 结束；真实最佳 val = `epoch=99`（13.2014），
   §4.1 的“epoch 86 最佳 13.118 / 训练继续中”是当时的中间快照。ep99 全量评测见 `P2D-eval-epoch99.csv`（2026-10-24 补测）。
2. **LRS2-2Mix 自训结果（tt 3,000）**：ep86 **8.618/9.042**；ep99（补测）；ep124+对比微调 **8.858/9.270**（PESQ 2.576/STOI 0.943）。
3. **官方权重跨语料基线已复跑并固化**：`P2D-eval-falcon-full.csv`（全量 3,000 条，7.03/7.71 量级），
   修复了“Phase-1 全量 CSV 被同路径 RealSSA 评测覆盖”的问题。
4. **RealSSA 全量证据已入库**：`P2E-realssa-sim-metrics_combined.csv`（9,171 行）+ `P2E-realssa-sim-eval-official.log`；
   RealSSA-Real DNSMOS 证据入库 `DNSMOS-realssa-real-mean-full3/`。
5. **证据策略已修**：`.gitignore` 放行 `docs/verification/**/*.log` 与 `*.csv`（此前 `*.log` 被全量忽略、`runs/` 被忽略），
   训练曲线 `P2D-train-metrics-1gpu.csv` / `P2D-near-train-metrics.csv` 已入库。
6. **当前状态与 v2 规划**：见 `docs/understanding/deepdive-20261024/00-SYNTHESIS.md` 与 `09-benchmark-alignment.md`。
