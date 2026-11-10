# CHANGELOG（仓库级变更记录）

> 历史提交清单以 `5371df7` 的 38 个提交为基线；下方补充 2026-11-09 与 2026-11-10 的项目文档更新。
> 来源：`git log --date=short` 的提交说明 + 每个提交的 `--name-only` 文件面归类。**只写事实，不评价**；数字是提交说明里的原话。
> 影响面标签：**代码** = `look2hear/`、`audio_*.py`、`evaluated_mac_params.py`、`studio/` 运行代码；**测试** = `tools/repro/tests`、`studio/tests`；**文档** = `*.md`、`docs/`、`project_page/`；**CI** = `.github/workflows/`；**配置·脚本** = `configs/`、`tools/repro/` 下非测试脚本。
> 历史清单中标 `上游` 的 8 个提交来自 `JusperLee/Falcon`；其余为本 fork 的复现工作。

## 2026-11-10

- 新增数据集说明，写明五套开源数据的样本、本机目录、索引条数，以及训练和测试实际使用的划分；中英文入口、状态页和数据访问说明改为指向该说明。
- 重写中英文 README、复现状态、数据训练指南和 Studio / Space 说明；新增模型、训练和评测三个专题及完整文档索引。
- 整理逐文件审阅、模型批次与分窗探针、历史数据复算、15 张流程图渲染和图片审查记录；历史原始日志与结果保留。
- 逐篇修订历史报告中的过时结论，明确权重身份、数据通道、有效样本数、因果探针条件与硬链接关系。
- 同步上游 `5c5d4ff` 的论文徽章与图片资源；上游新增的三个提交均为文档或图片更新。

## 2026-11-09

| 提交 | 一句话 | 影响面 |
| --- | --- | --- |
| `6e5c1b5` | 全库审计后的文档更新、新建与历史归档整理 | 文档 |

## 2026-11-08（本轮 7 个提交）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `5371df7` | CI：repro 证据套件不再作为 CI 门禁（它依赖本机数据/权重/环境锁） | CI（1） |
| `8008a7c` | 测试：收集期的 `is_file` 守卫改成容错版（CI 上 `/root/.cache` 不可穿越） | 测试（2） |
| `8d6ca40` | CI：补 pytorch-lightning 与完整 git 历史，机器本地用例在 CI 跳过，Pages 未启用时优雅跳过 | CI + 测试（3） |
| `8b9d903` | 文档：按 deepdive-20261108 全面校正口径、判定与证据链（含新增 22 篇深挖） | 文档（40） |
| `bdbbbfc` | 复现工具：新增 5 个 CPU 实验脚本与可复核证据 | 配置·脚本 + 文档（16） |
| `e684fd4` | 测试基础设施：CPU 可全绿、不再往仓库落 ckpt、CI 跳过机器相关指纹 | 测试（5） |
| `9a920ce` | 代码修复与忠实化开关：默认数值逐位不变 | 代码（20） |

这一批的落点：`tools/repro/experiments/` 5 个脚本 + `docs/verification/deepdive-20261108/` 9 个证据文件（`bdbbbfc`）；`tools/repro/tests/conftest.py` 新增、线程数与 ckpt 落盘隔离（`e684fd4`、`8008a7c`）；`.github/workflows/tests.yml` 三段改动 + 新增 `pages.yml`（`8d6ca40`、`5371df7`）；`STATUS.md`、`AGENTS.md`、`README*`、deepdive 22 篇统一口径（`8b9d903`）；`look2hear/` 与 `studio/falcon_model/` 两份实现同步加 `paper_causal` / `video_lstm_batch_first` 开关（`9a920ce`）。

## 2026-10-30（11 个）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `6c1d5b7` | STATUS：回填 HEAD=2062528 与 freeze tag 说明 | 文档（1） |
| `2062528` | 复现维护轮：三轮 review Day-1/Week-1 修复（数据 P0 / DNSMOS 重跑 / 模型自描述 / Table16 配置 / Studio 与 CI / 证据收口） | 代码 + 测试 + 文档 + CI（77） |
| `0fdad70` | 三轮详细 review 落盘（10 subagent/轮 ×3 + 10 压缩复核）：150 候选 / 124 open / 17 done | 文档（2） |
| `a3eb0d7` | P1 完成：RealSSA-Sim ch0（论文 channel-1）全量复评 7.977 / 8.452 | 文档（7） |
| `c57aecd` | 证据：重跑恢复 ep86 逐句 CSV（8.618 / 9.042，与历史日志一致） | 文档（6） |
| `16c9541` | 文档：记录 main 已推送（origin/main=d076be9）与论文校正要点 | 文档（3） |
| `d076be9` | 论文对齐（arXiv:2610.32749）：修正 DNSMOS / channel / 训练损失等结论 | 文档（6） |
| `3d36b78` | 文档：记录 2026-10-30 上游同步、arXiv 公开与仓库收口 | 文档（4） |
| `d9ec1e8` | Merge upstream/main 进 repro/phase2 | 代码（1） |
| `714e1a5` | RealSSA 转换：mouths.pt 原子写（临时文件 + O_EXCL + fsync + os.replace） | 配置·脚本（1） |
| `2582f61` | 复现 P0 收口：证据入库 + ep99/官方全量补测 + near 负样本可复现 + 回归/环境/文档修复 | 测试 + 文档 + 配置·脚本（141） |

## 2026-10-29 / 10-27（各 1 个，上游）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `439cbe9` | 上游：在 README 与项目页链接 arXiv 论文 | 文档（7） |
| `ea1fa58` | 上游：统计 Hugging Face Hub 的模型下载数 | 代码 + 测试 + 文档（5） |

## 2026-10-23（3 个）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `2beec38` | RealSSA 数据集下载 + 归档完成（Sim 521.5 GB / Real 12.84 GB → TOS 冷端） | 配置·脚本（2） |
| `29e19c7` | LRS2-2Mix 论文配方重训（8 卡 / 火山引擎）方案落地 | 代码 + 配置·脚本 + 文档（11） |
| `9121598` | RealSSA-Sim 全量复现 7.774 / 8.479 + LRS2 contrastive 微调 8.86 / 9.27 + 数据集下载归档脚本 | 文档 + 配置·脚本（2） |

## 2026-10-22（7 个）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `141c751` | 补 LRS2 contrastive 微调配置（epochs 130 从 100 ckpt 续训，use_near=true，s2 负样本） | 配置·脚本（1） |
| `6d2df43` | LRS2 contrastive 微调配置与自动续训（s2 负样本，等当前训练结束后启动） | 配置·脚本（1） |
| `77a4aa2` | RealSSA-Sim 主结果复现（SI-SNRi 7.94 / SDRi 8.75 vs 论文 7.7 / 8.2） | 文档（2） |
| `82c380b` | 补记 RealSSA-Real DNSMOS（P808 3.329 / OVRL 2.790 vs mixture 基线） | 文档（1） |
| `1dc0fa5` | RealSSA 数据获批并转换（Sim tt 9,169 条 / Real 12.8 GB），窗口化 DNSMOS 评测 | 文档 + 配置·脚本（2） |
| `162c4f1` | LRS2 训练最佳 ckpt 全量评测（8.62 / 9.04）+ s2 负样本 near.json | 文档（1） |
| `9df619b` | 归因 7 dB 差距为跨语料，LRS2-2Mix 全量转换并训练，RealSSA 预置 | 文档 + 配置·脚本 + 测试（63） |

## 2026-10-21（6 个）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `af74cca` | 复现 Phase 2 启动：提交 RealSSA 访问申请并更新数据状态 | 文档（3） |
| `ac91800` | 复现 Phase 1：修复训练/评测入口与指标，重建数据工具链与验证资产 | 代码 + 测试 + 文档 + 配置·脚本（154） |
| `689555c` | 上游：同步中文 README 头与已发布 RealSSA 数据集指南 | 文档（1） |
| `8188b91` | 上游：简化 README 头并链接已发布 RealSSA 数据集 | 文档（1） |
| `a27b1e6` | 上游：新增 Hugging Face ZeroGPU Space 并在 README 链接 | 文档（17） |
| `607352c` | 上游：限制设备启动并暴露采集卡顿诊断 | 代码 + 测试 + 文档（5） |

## 2026-10-20（2 个，上游）

| 提交 | 一句话 | 影响面（文件数） |
|---|---|---|
| `7576854` | 上游：新增项目页（`project_page/`，发布到 cslikai.cn/Falcon） | 文档 + CI（8） |
| `a7af472` | 上游：Falcon 首版（训练、评测与 Falcon Studio） | 代码 + 测试 + 文档 + 配置·脚本（150） |

## 相关入口

- 当前状态与口径：[`STATUS.md`](STATUS.md)（代码、权重、数据与评测记录的当前汇总）
- 2026-11-08 那轮改了什么、验到什么程度：[`14-UPDATE-20261108.md`](understanding/deepdive-20261108/14-UPDATE-20261108.md)
- 证据代际：[`verification/README.md`](verification/README.md)
- 文档总索引：[`README.md`](README.md)
