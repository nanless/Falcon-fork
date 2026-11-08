# docs/ 文档索引与事实源路由

> 最后更新 2026-11-09 · 锚定 HEAD `5371df7` · 链接按 Markdown 语义**相对本文件所在目录 `docs/`** 解析（指仓库根用 `../`）。
> 本文件只回答"想了解 X 该读哪份"，不复述任何数字与结论。

## 0 一句话

HEAD `5371df7` 的 112 篇已跟踪 Markdown 里只有 **1 份事实源**：[`STATUS.md`](STATUS.md)。其余按用途分三层——数字出处（`understanding/deepdive-20261108/`）、机器证据（`verification/`）、历史快照（`deepdive-20261024/`、`reviews/`、phase1/2 旧报告）。

## 1 事实源层级（冲突时自上而下）

| 层 | 文件 | 管什么 | 怎么引用 |
|---|---|---|---|
| 1 事实源 | [`STATUS.md`](STATUS.md) | 复现到哪一层、每个数字的口径、待办。与它冲突的旧文按它为准 | 用节号（`§1`/`§4.3`），不用行号：每轮重写都漂 |
| 2 数字出处 | [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) | STATUS 的数字来自这 13 份报告；每份带脚本路径与实测命令 | `00/01–12/14` 编号引用 |
| 3 运行约定 | [`../AGENTS.md`](../AGENTS.md) | 解释器、环境、常用命令；与旧命令冲突按它执行 | 按节引用 |
| 4 对外叙述 | [`../README.md`](../README.md)、[`../README_zh.md`](../README_zh.md) | 给仓库外读者的复现状态摘要 | 锚点 `#reproduction-status` |
| 5 历史/证据 | 见 §3、§4 | 只作过程记录，结论已被 1–4 层取代 | 带日期锚点引用 |

## 2 分类计数（A3 审计口径，2026-11-09）

| 类别 | 篇数 | 判断依据 | 主要位置 |
|---|---|---|---|
| 现用源 | **13** | 自述为事实源/入口/运行约定，最后更新 2026-11-08 | `STATUS`、`AGENTS`、根 `README*`、deepdive 的 `00`/`14`、`phase2/README` 等 |
| 证据 | **47** | 带实测命令、日志、CSV | `verification/` 报告 + deepdive `01–12` 与 `R1–R7` |
| 历史快照 | **40** | 2026-10-21 / 10-24 / 10-30 三代，带日期锚点 | `reviews/`、`understanding/` 平铺 8 篇、`deepdive-20261024/` |
| 派生 | **12** | 从别处生成：HF 数据卡 5、studio/space README 4、`PROVENANCE`、`NOTICE`、fixture 说明 | 见 §4 |
| 合计 | **112** | HEAD `5371df7` 的 `git ls-files "*.md"`（本文件与另两份新索引尚未提交，不计入） | 13 + 47 + 40 + 12 = 112 |

口径来源：A3 文档盘点（2026-11-09，仓库外审计，§1 汇总表）。这是审计时点的计数，新增文件不回溯改写本表。

## 3 问题 → 入口

| 问题 | 首选入口 | 备选 / 历史 |
|---|---|---|
| 现在复现到哪一层，L1–L4 各是什么 | [`STATUS.md`](STATUS.md) §1 | [`../README.md`](../README.md#reproduction-status)（对外摘要）、[`REPRODUCTION_PHASE2_REPORT.md`](REPRODUCTION_PHASE2_REPORT.md)（10-30 快照） |
| 某个数字从哪来、什么口径 | [`STATUS.md`](STATUS.md) §3–§4 | [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) §0 硬数字表 |
| 参数量 537,697 与死参数怎么算的 | [`01-model-core.md`](understanding/deepdive-20261108/01-model-core.md) | [`verify_model_params.py`](verification/scripts/verify_model_params.py)、[`model-core.md`](verification/model-core.md)（10-21 记录） |
| MACs 两把尺子（10.611 G / 5.382 G）差在哪 | [`STATUS.md`](STATUS.md) §4.3 | [`05-losses-metrics.md`](understanding/deepdive-20261108/05-losses-metrics.md)、[`02-layers-dsp.md`](understanding/deepdive-20261108/02-layers-dsp.md) |
| lookahead / 因果性 / 904 ms 怎么来的 | [`12-causality-latency.md`](understanding/deepdive-20261108/12-causality-latency.md) | [`STATUS.md`](STATUS.md) §4.1、[`lookahead-ab.json`](verification/deepdive-20261108/lookahead-ab.json) |
| 与论文差多少、哪些项对不上 | [`11-paper-alignment.md`](understanding/deepdive-20261108/11-paper-alignment.md) | [`STATUS.md`](STATUS.md) §3.2、[`PAPER_ALIGNMENT_20261030.md`](understanding/PAPER_ALIGNMENT_20261030.md) |
| 数据在哪、多大、怎么转成仓库格式 | [`DATA_ACCESS_GUIDE.md`](DATA_ACCESS_GUIDE.md) | [`06-data-pipeline.md`](understanding/deepdive-20261108/06-data-pipeline.md)、[`T6-report.md`](verification/phase1/T6-report.md) |
| 评测指标怎么算、分母为什么是 9,166 | [`STATUS.md`](STATUS.md) §4.5 | [`07-eval-benchmark.md`](understanding/deepdive-20261108/07-eval-benchmark.md)、[`P2E-report.md`](verification/phase2/P2E-report.md) |
| 8 卡论文配方怎么跑 | [`PHASE2_VOLCANO_LRS2_TRAINING.md`](PHASE2_VOLCANO_LRS2_TRAINING.md) | [`STATUS.md`](STATUS.md) §7、[`04-training-system.md`](understanding/deepdive-20261108/04-training-system.md) |
| 本机用什么解释器、依赖锁是什么 | [`../AGENTS.md`](../AGENTS.md) | [`conda-env-lock.txt`](verification/conda-env-lock.txt)、[`VERIFY-ENV.md`](understanding/VERIFY-ENV.md)（历史） |
| 测试到底多少条、CI 跑不跑 | [`STATUS.md`](STATUS.md) §5 | [`../tools/repro/tests`](../tools/repro/tests)、[`../.github/workflows/tests.yml`](../.github/workflows/tests.yml) |
| 复现实验脚本怎么跑（纯 CPU） | [`../tools/repro/experiments/README.md`](../tools/repro/experiments/README.md) | [`verification/deepdive-20261108/README.md`](verification/deepdive-20261108/README.md) |
| Studio 怎么起、分块几何是什么 | [`08-studio.md`](understanding/deepdive-20261108/08-studio.md) | [`../studio/README.md`](../studio/README.md) |
| Space 与项目页怎么构建 | [`09-space-project-page.md`](understanding/deepdive-20261108/09-space-project-page.md) | [`../space/README.md`](../space/README.md) |
| 历次评审、三轮 review 的结论 | [`reviews/round1/`](reviews/round1/) | [`THREE_ROUND_REVIEW_20261030.md`](understanding/THREE_ROUND_REVIEW_20261030.md)、[`reviews/round2/`](reviews/round2/) |
| 这份文档是哪一轮写的、改了什么 | [`CHANGELOG.md`](CHANGELOG.md) | [`STATUS.md`](STATUS.md) §9、[`14-UPDATE-20261108.md`](understanding/deepdive-20261108/14-UPDATE-20261108.md) |

## 4 目录一览

- **`verification/`** — 34 篇 md 证据的存放地，四代并存。**先看** [`verification/README.md`](verification/README.md) 的代际表，再点进具体代（[`phase1/`](verification/phase1/README.md) / [`phase2/`](verification/phase2/README.md) / [`deepdive-20261108/`](verification/deepdive-20261108/README.md)）。
- **`understanding/`** — 三代深挖：[`deepdive-20261108/`](understanding/deepdive-20261108/00-SYNTHESIS.md)（现用）≻ [`deepdive-20261024/`](understanding/deepdive-20261024/00-SYNTHESIS.md)（10-24 快照）≻ 平铺 8 篇 + [`FINAL_PROJECT_UNDERSTANDING.md`](understanding/FINAL_PROJECT_UNDERSTANDING.md)（10-21 理解轮）。
- **`reviews/`** — [`round1/`](reviews/round1/) 8 篇 + [`round2/`](reviews/round2/) 6 篇，全部锚定 2026-10-21 基线与 sha256，只作历史。
- **顶层 6 篇** — [`STATUS.md`](STATUS.md)（事实源）、[`DATA_ACCESS_GUIDE.md`](DATA_ACCESS_GUIDE.md)、[`PHASE2_VOLCANO_LRS2_TRAINING.md`](PHASE2_VOLCANO_LRS2_TRAINING.md)、[`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md)（历史计划）、[`REPRODUCTION_PHASE1_REPORT.md`](REPRODUCTION_PHASE1_REPORT.md)、[`REPRODUCTION_PHASE2_REPORT.md`](REPRODUCTION_PHASE2_REPORT.md)（10-22 / 10-30 执行快照）。

## 5 派生件（12 篇，不当事实源）

`verification/phase1/dataset-cards/` 5 篇（HF 数据集卡，2026-10-21 抓取）、`studio/README*.md` 与 `space/README*.md` 4 篇、`studio/falcon_model/PROVENANCE.md`、`studio/web/vendor/mediapipe/NOTICE.md`、`verification/scripts/data_pipe/README.md`。体积、条数、license 一律回到 [`STATUS.md`](STATUS.md) §2.5 与 [`DATA_ACCESS_GUIDE.md`](DATA_ACCESS_GUIDE.md)。

## 6 本轮 2026-11-08/09 新增入口

- [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) — 本轮深挖总纲：10 个 Wave-1 域 + 2 个 Wave-2 攻坚的结论与硬数字表；STATUS 的数字出处。
- [`14-UPDATE-20261108.md`](understanding/deepdive-20261108/14-UPDATE-20261108.md) — 按深挖结论改仓库的那一轮记录：改了哪些文件、依据、验证到哪一步。

## 7 约定

1. **新文档写在这里**（`docs/` 下按用途入 `verification/`、`understanding/deepdive-<日期>/`），并在本文件 §3 加一行路由；不新开第二个索引。
2. **旧文档不再改**：`deepdive-20261024/`、`reviews/`、`REPRODUCTION_PLAN.md`、phase1/2 旧报告保留原样作证据链，口径冲突以 §1 的 1–4 层为准。
3. **本文只做路由**：数字、结论、命令一律指向源头文件，避免与 [`STATUS.md`](STATUS.md) 形成第二份口径。
