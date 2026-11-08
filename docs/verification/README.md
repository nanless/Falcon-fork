# verification 证据目录 · 代际表

> 最后更新 2026-11-09 · 锚定 HEAD `5371df7` · 链接按 Markdown 语义**相对本文件所在目录 `docs/verification/`** 解析（回 `docs/` 层用 `../`，回仓库根用 `../../`）。
> 本文件只做索引与"能否复跑"判断，不复述证据内容；单代细节看各代自己的 README。

## 0 结论

四代证据并存，5 个目录。**只有第 4 代能按文档里的命令原样复跑**：纯 CPU、解释器 conda `falcon` 在、权重 sha256 已锁。第 1 代要本机数据与 HF 凭据，第 2 代要 9,169 条推理 + GPU，第 3 代的 30 个脚本把解释器写成了已删除的 `/root/.venvs/falcon`。

## 1 代际表

| 代 | 目录 | 时间 | 覆盖什么 | 能否复跑 |
|---|---|---|---|---|
| 1 Phase1 | [`phase1/`](phase1/README.md) | 2026-10-21 → 10-22 | T1–T6 任务级证据 + QA/REVIEW/REGRESSION：训练入口、日志、指标语义、数据层、视觉骨干、数据获取 | **部分**：解释器在，T1/T2/T3/T5 用合成数据可跑；T4/T6 要 `/root/data` 与 HF 凭据 |
| 2 Phase2 | [`phase2/`](phase2/README.md) | 2026-10-22 → 10-31 | P0 修复 + P2A–P2F：9,169 条 RealSSA-Sim 逐句 CSV、DNSMOS、LRS2 训练与评测、协议对拍 | **否**：要 `/root/data`、`falcon.pt`、GPU；全量推理不可迁移 |
| 3 三轮 review | [`runs/`](runs/)、[`scripts/`](scripts/)、根 5 篇 md | 2026-10-21 → 10-30 | 理解轮 R1–R4 与三轮 review 的脚本与日志：参数量、MACs、因果性、训练契约、Studio 后端 | **否**：30 个脚本的解释器路径已删；换 conda `falcon` 多数逻辑成立，未逐脚本重跑 |
| 4 deepdive-20261108 | [`deepdive-20261108/`](deepdive-20261108/README.md) | 2026-11-08 | 5 个 CPU 实验脚本的产物 JSON：lookahead A/B、LN 变体、流式一致性、视频消融、仲裁口径 | **是**：命令、耗时、权重 sha256 都在该目录 README §1 |

## 2 逐代：入口与复跑命令

### 2.1 第 1 代 Phase1（`phase1/`，43 个文件）

入口 [`phase1/README.md`](phase1/README.md)（任务分工表 + Gate G1）与 [`phase1/QA-report.md`](phase1/QA-report.md)（独立回归）。数值基线是 `baseline-fingerprint.json`：改代码后 sha256 不变即"数值未变"。

复跑（仓库根，conda `falcon`；`ENV=/root/miniforge3/envs/falcon/bin/python`）：

```bash
CUDA_VISIBLE_DEVICES= $ENV -m pytest tools/repro/tests -q      # 现收集 168 条，见 ../STATUS.md §5
```

### 2.2 第 2 代 Phase2（`phase2/`，97 个文件）

入口 [`phase2/README.md`](phase2/README.md)（P2-A…H 索引，顶部有三层校正横幅：2026-10-31、11-08）。逐句数据在 `P2E-realssa-sim-metrics_combined.csv`，DNSMOS 在 `DNSMOS-realssa-real-mean-full4-20261031/`。

不可原样复跑：需要 `/root/data` 下的 LRS2-2Mix 28,000 条与 RealSSA、`/root/.cache/falcon_weights/falcon.pt`、以及 9,169 条推理的 GPU 时间。可复跑的是 CPU 冒烟（`P2D-smoke*.log` 对应的入口）。

### 2.3 第 3 代 三轮 review（`runs/` 51 个日志、`scripts/` 47 个文件、根 5 篇 md）

根 5 篇是理解轮 Lead 的记录：[`model-core.md`](model-core.md)、[`layers-dsp.md`](layers-dsp.md)、[`lead-verification.md`](lead-verification.md)、[`data-pipe-supplement.md`](data-pipe-supplement.md)、[`r3-layers-dsp-anchor.md`](r3-layers-dsp-anchor.md)（外置 sha256 锚点）。`runs/` 是这些脚本当时的 stdout；`scripts/` 的 47 个文件里还含 `data_pipe/` 下的校验脚本与 `pil_stub/` 桩。

**不能原样复跑**：30 个脚本的 docstring 与运行命令写 `/root/.venvs/falcon/bin/python`（py3.10 环境，已删除），直接照抄报 `No such file or directory`。逻辑可迁移：解释器换 `/root/miniforge3/envs/falcon/bin/python`，`PYTHONPATH` 保留 `docs/verification/stubs`（否则 timm/typeguard 行为不同，stub 里的假 `timm` 只实现 `DropPath`）。**未逐脚本重跑**，迁移后是否全绿未验证；这些日志只能标"证据来自 stub 环境"。

### 2.4 第 4 代 deepdive-20261108（`deepdive-20261108/`，9 个文件）

入口该目录的 [`README.md`](deepdive-20261108/README.md)。纯 CPU、`OMP_NUM_THREADS=4`、权重固定 `falcon.pt`（sha256 `a1888b5a…ce928c`，epoch 714）。

```bash
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
/root/miniforge3/envs/falcon/bin/python tools/repro/experiments/lookahead_ab.py --json /tmp/ev/lookahead-ab.json
```

5 个脚本各自 10–100 s，说明见 [`../../tools/repro/experiments/README.md`](../../tools/repro/experiments/README.md)。`canonical.runtime` 类的 `elapsed_s` 随机器变，其余字段已做逐位复跑校验。

## 3 已作废的口径（不要引用）

| 作废项 | 位置 | 现在用哪个 |
|---|---|---|
| full3 DNSMOS（OVRL 2.7901 / P808 3.3285） | `phase2/DNSMOS-realssa-real-mean-full3/` | full4：**OVRL 2.7941 / P808 3.3365**（窗口起点 bug 已修，见 [`P2E-report.md`](phase2/P2E-report.md) §10） |
| "Space 构建 57 文件 / 39,869,169 B" | 多篇 2026-11-08 早期稿 | **54 文件、约 39.87 MB**（57 是把 3 个目录当文件数了） |
| "166 passed / 2 failed（无 GPU）" | 修前测试基线 | **168 passed / 0 failed**（有/无 GPU 都是），见 [`../STATUS.md`](../STATUS.md) §5 |
| "111 passed / 1 skipped"、"96 passed / 6 skipped" | `phase2/README.md` §1 的 10-21 快照 | 同上 168；那两行是历史快照 |
| `STATUS.md:NNN` 行号锚点（约 60 处） | deepdive 各篇 | 改引 `../STATUS.md` 的**节号**（§2.3/§4.3…），行号每轮漂 |
| `/root/.venvs/falcon` 作为可执行路径 | `scripts/*.py`、根 5 篇 md 的命令块 | conda `falcon`（`/root/miniforge3/envs/falcon/bin/python`） |

## 4 本机专有（换机器不能复现的部分）

- **数据**：`/root/data/` 下 7 个数据集目录（LRS2-2Mix 28,000 条、RealSSA-Sim/Real 等），冷端归档在 TOS。
- **权重**：`/root/.cache/falcon_weights/falcon.pt`（软链到 `/root/group-shared/…`），sha256 已在第 4 代 README 里锁。
- **硬件**：2×A800-80G 与主机名；A800 延迟类数字在 [`../STATUS.md`](../STATUS.md) §4.6 标为未验证。
- **日志里的绝对路径**：`runs/`、`phase1/*.log`、`phase2/*.log` 含 `/root/...` 与旧环境路径，属记录事实，不作可执行命令。

## 5 跨代导航

代际链：第 1 代 → 第 2 代 → 第 3 代 → 第 4 代（本目录当前）。

- 上一代：[`phase2/README.md`](phase2/README.md)（2026-10-22 → 10-31 的 Phase 2 索引）
- 下一代：无（第 4 代是当前最新；新证据按日期新建 `deepdive-YYYYMMDD/`）
- 回到文档总索引：[`../README.md`](../README.md)
