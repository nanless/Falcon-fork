# Phase 2 工作区索引（task-25 维护）

> 分支：`repro/phase2`（HEAD `af74cca`，基于 main `ac91800`）· 开始：2026-10-22 · 维护：`studio-frontend`（task-25）
> 目标：收口 Phase 2 并行任务（P2-A…G）的产出索引/状态/复跑命令，并做文档一致性 + Studio/Space/tools 回归。
> 约束：conda `falcon`（Python 3.11.17 / torch 2.8.0+cu128）；不改源码、不 commit；本目录为本任务写作用域。
> 回归与一致性明细：`REGRESSION-docs.md`（v0 2026-10-21T19:10Z）；状态快照见 §1 表。

> **状态更新（2026-10-31）**：本索引的表格是 Phase 2 期间快照；**当前结论/证据/待办以 `docs/STATUS.md` 为唯一事实源**（2026-10-30 三轮 review 详见 `docs/understanding/THREE_ROUND_REVIEW_20261030.md`）。
> 历史快照（2026-10-24，HEAD `2beec38`）：：本索引的“RealSSA 403 / 训练进行中”已过期——RealSSA 已获批、下载并归档
> （Sim 521.5 GB、Real 12.84 GB），RealSSA-Sim test 9,169 条评测 **7.774/8.479**；P2-D 训练已于 2026-10-22 19:54 结束
> （100 epoch，最佳 val ep99 13.201），contrastive 微调 ep124 **8.858/9.270**。新增入库证据：
> `P2E-realssa-sim-eval-official.log`、`P2E-realssa-sim-metrics_combined.csv`、
> `P2D-eval-{epoch86,epoch99,epoch124-contrastive,falcon-full}.{log,csv}`（**ep99 8.726/9.122**、官方全量 **7.029/7.713**）、`P2D-train-metrics-1gpu.csv`、
> `P2D-near-train-metrics.csv`、`DNSMOS-realssa-real-mean-full3/`。完整现状见 `docs/understanding/deepdive-20261024/`（08/09）。

## 1. 任务索引与状态

| 任务 | 内容 | Owner | 期望产出（本目录） | 状态（快照 2026-10-21T19:10Z） |
|---|---|---|---|---|
| **P2-A** | LRS2-2Mix 与论文差 ~7 dB：指标/协议变体排查 | loss-metric-eval | `P2A-report.md`、`P2A-*.log`、`tools/repro/phase2/gap_analysis.py` | ✅ 报告落盘（[P2A-report.md](P2A-report.md)）：200 条×14 变体，基线 6.95/7.62；**所有合理变体收益 ≤0.4 dB**；mouth=s2 −16.7 / 置黑 +2.16 / 错位 ±5/12 帧 −3.4~−5.9 / 96 不裁 −0.04 / 1ch ±0.00 / 不归一化 −0.45 / 补零 3 s −5.13；结论：**指标/协议不能解释 ~7 dB**，指向训练数据/配方（与 P2-B/P2-F 一致） |
| **P2-B** | Falcon-Big vs Small、整句 vs 窗口化推理、mouth 分支贡献 | model-core | `P2B-report.md`、`tools/repro/phase2/compare_variants.py`、日志 | ✅ 报告落盘（[P2B-report.md](P2B-report.md)）：small 6.916/7.492、small windowed 5.004/5.762、big 7.887/8.403、big windowed 5.569/6.270；配对 big−small **+0.971**、窗口化 −1.912/−2.318；视觉消融 −15.089/−12.894/−4.375；**3000 基线行列序 P3 已修**（SI-SNR 12.023 / SDR 12.813） |
| **P2-C** | LRS2-2Mix 全量 → `/root/data/falcon_lrs2_2mix` + JSON/near | data-pipe | `P2C-report.md`、`P2C-convert.log`、`tools/repro/lrs2_2mix_to_falcon.py`（QA 侧测试为 `tools/repro/tests/test_phase2_data.py`） | ✅ 报告落盘（[P2C-report.md](P2C-report.md)）：**全量 28000 条（tr 20000 / cv 5000 / tt 3000）转换完成，23.36 GiB**；三表等长、mouth `[50,96,96,3]` uint8、脚本/测试 sha256 已核、`test_lrs2_convert.py` **7 passed**；near=fallback→**K=0**（训练结果须声明无对比负项） |
| **P2-D** | LRS2-2Mix 训练复现（配置/冒烟/训练/评测） | train-sys | `P2D-report.md`、`P2D-*.csv/log`、`configs/LRS2_2Mix*.yml` | ✅ **终态**：单卡 100 epoch 于 2026-10-22 19:54 结束（best val ep99 13.2014），contrastive ep124 13.3909；tt 3000：官方跨语料 7.029/7.713、ep86 8.618/9.042、ep99 8.726/9.122、ep124 8.858/9.270（论文 14.1/14.7 未复现，需 8 卡论文配方）。详见 [P2D-report.md](P2D-report.md) §8 |
| **P2-E** | RealSSA 评测（DNSMOS + Sim + Real 管线） | ops-tests-docs | `P2E-report.md`、`P2E-*.{log,json,csv}` | ✅ **终态**：RealSSA-Sim test 9,169 条 **7.774/8.479**（mean）、ch0 **7.977/8.452**；RealSSA-Real 54 段 DNSMOS 旧 run（P.835 OVRL 2.7901/P808 3.3285，窗口 bug 修复后重跑见 §10）；论文 3.276 属 RealSSA-trained ckpt、不可直接比 |
| **P2-F** | LRS2-2Mix 指标/协议第三方独立验证 | studio-backend | `P2F-report.md`、`tools/repro/phase2/protocol_check.py`、日志 | ✅ 报告落盘（[P2F-report.md](P2F-report.md)）：独立对拍 **4.8e-8/1.8e-7 dB**、baseline 4.994、target=s1 Δ+24.5、PIT 0、lag 0、无削波；**关键归因：发布 `falcon.pt` 训练集 = LRS3_complex_nonmoving（`conf.yml:99-102`）→ 7.03/7.71 属跨语料结果，不是论文 LRS2 列（14.1/14.7）的复现**；P5 基线 P3 已修（7.911） |
| **P2-G** | Phase 2 独立 QA + 回归 | layers-dsp | `QA-report.md`、`QA-regression.log` | ✅ 历史值 111/1（2026-10-21 collector）；**当前基线见 `docs/STATUS.md` §5**（Phase1 skip 守卫已修） |
| **P2-H** | 本索引 + 文档一致性 | studio-frontend | `README.md`、`REGRESSION-docs.md` | ✅ 历史快照（19:45Z 回归 96/6）；**当前状态以 `docs/STATUS.md` 为准** |

> 说明：本表由 task-25 在终检时更新为 ✅/⚠️ 并补报告链接；其余任务以各自报告的“结论”节为准。

### 1.1 Phase 2 结论摘要（终检 19:45Z）

1. **7 dB 差距已定位（P2-A/B/F 三方一致）**：评测协议/指标/对齐/target/PIT/缩放全部排除（合计 ≤0.4 dB；
   big 仅 +0.97 dB；窗口化 −1.9~−2.3 dB 反而更差）；**主因是发布 `falcon.pt` 为 LRS3 派生数据训练**
   （`Experiments/checkpoint/Falcon/conf.yml:99-102` + `PROVENANCE.md:37-38`）→ 7.03/7.71 是**跨语料**结果。
   要复现论文 LRS2-2Mix 14.1/14.7，需 **LRS2-trained ckpt**（P2-D 训练中）或作者提供 benchmark-specific ckpt。
2. **数据/训练链路已就绪**：LRS2-2Mix 全量 28000 条转换完成（23.36 GiB，JSON/near 齐全）；P2-D 正式训练
   19:26:41Z 已启动（单卡 bs4，PID 4002894）；评测脚本（`audio_visual_test.py` / `eval_p2d.sh`）可用。
3. **RealSSA 仍待审批**（Sim/Real 403）：工具链（DNSMOS/RealSSA-Real/Sim 1-shard）已预置并通过接口冒烟，
   **本 Phase 不产出任何 RealSSA 指标结论**。
4. **回归与一致性全绿**：`tools/repro/tests` 96 passed/6 skipped（102 collected）、Studio 35/2、Node 8/8、
   i18n 226、imports 10/10；HF 5 数据集事实与 GUIDE 一致；GUIDE RealSSA 状态已统一；2 处报告 P3 已修。
5. **已知残留**：P2-D 训练结果待跑完；QA 终检独占重跑待回填；PLAN:138 路径 P3；FINAL/PHASE1_REPORT 的跨语料标注
   建议（见 `REGRESSION-docs.md` §6）。

## 2. 已知基线（Phase 1 交付，Phase 2 的起点）

- **代码/分支**：`main` = `ac91800`（Phase 1 已 commit+push）；`repro/phase2` = `af74cca`。
- **环境**：conda `falcon`：Python 3.11.17、torch 2.8.0+cu128（2×A800）、PL 2.6.6、fastapi/Pillow/huggingface_hub 等齐全。
- **权重**：`/root/.cache/falcon_weights/falcon.pt`（47,156,343 B）、`falcon-big.pt`（49,645,087 B）、
  导出的视觉骨干 `frcnn_128_512.backbone{,.big}.pth.tar`。
- **数据**：
  - LRS2-2Mix：`/root/data/lrs2_2mix/lrs2.tar.gz`（16,470,508,590 B，sha256 `ebdfdf31…767e` = HF LFS）+
    解压 `/root/data/lrs2_2mix/lrs2_rebuild/`（17.39 GB；完整 2 s AV：`audio/wav{16,8}k/min/{tr,cv,tt}/{mix,s1,s2}`、
    `mouths` npz uint8 (50,96,96)、`faces` mp4、`landmarks`、`create_mix` 脚本/清单；tr 20000/cv 5000/tt 3000 组；raw 20,392）。
  - LRS2 评测集：`/root/tmp/lrs2_eval_tt_full`（Phase 1 构建，tt 3000 条，target=s1，mouths→(50,96,96,3).pt）。
  - RealSSA-Sim / RealSSA-Real：**gated（manual）**，申请已提交（`docs/REPRODUCTION_PLAN.md` Phase 2 状态），
    数据文件当前 403；大小/许可见 §5。
- **Phase 1 实测 LRS2-2Mix（官方 `falcon.pt`）**：全量 tt 3000 条 **SI-SNRi 7.03 / SDRi 7.71**（20 条子集 7.91/8.55；
  target=s1 已确证）；论文 LRS2-2Mix 14.1/14.7 → Phase 2 的 ~7 dB 差距是核心待查项。
  证据：`docs/REPRODUCTION_PHASE1_REPORT.md` §3.5；Phase 1 报告与回归见 `docs/verification/phase1/`。

## 3. 如何复跑 Phase 2（命令模板；以各 P2 报告最终命令为准）

> 所有命令在仓库根执行；`ENV=/root/miniforge3/envs/falcon/bin/python`；数据在 `/root/...`，不写仓库。

### 3.1 数据获取 / 转换（P2-C、T6）

```bash
# LRS2-2Mix 若未下载（已下载可跳过；默认断点续传）
$ENV tools/repro/fetch_data.py --dataset lrs2-2mix --proxy http://127.0.0.1:17897
# 全量转换到 Falcon folder 格式（工具 P2-C 落盘后；target 固定 s1）
$ENV tools/repro/lrs2_2mix_to_falcon.py --src /root/data/lrs2_2mix/lrs2_rebuild \
      --out /root/data/falcon_lrs2_2mix --target s1 --near-mode fallback --jobs 8
# 校验
python - <<'PY'
import json; d=json.load(open('/root/data/falcon_lrs2_2mix/tt/mix.json')); print('tt mix entries', len(d))
PY
```

### 3.2 LRS2-2Mix 评测（官方权重；Phase 1 已验证）

```bash
# small（论文模型；已完成全量 3000 条）
PYTHONPATH=$PWD $ENV audio_visual_test.py --conf_dir configs/Falcon.yml \
  --data_dir /root/tmp/lrs2_eval_tt_full --ckpt_path /root/.cache/falcon_weights/falcon.pt --gpus 0
# 子集快速冒烟
PYTHONPATH=$PWD $ENV audio_visual_test.py --conf_dir configs/Falcon.yml \
  --data_dir /root/tmp/lrs2_eval_tt_full --ckpt_path /root/.cache/falcon_weights/falcon.pt --limit 20 --gpus 0
# big（P2-B 对照）
PYTHONPATH=$PWD $ENV audio_visual_test.py --conf_dir studio/configs/FalconBig.yml \
  --data_dir /root/tmp/lrs2_eval_tt_full --ckpt_path /root/.cache/falcon_weights/falcon-big.pt --limit 300 --gpus 0
```

### 3.3 训练复现（P2-D；配置落盘后）

```bash
# 冒烟（1 GPU、小步数；以 P2D-report 为准）
$ENV audio_train.py --conf_dir configs/LRS2_2Mix.yml
# 正式/后台训练命令与日志：docs/verification/phase2/P2D-train.log（P2-D 提供）
```

### 3.4 RealSSA / DNSMOS（P2-E；待审批）

```bash
# 只读探测（当前应显示 GatedRepoError 403）
$ENV tools/repro/fetch_data.py --dataset realssa-sim --dry-run --proxy http://127.0.0.1:17897
$ENV tools/repro/fetch_data.py --dataset realssa-real --dry-run --proxy http://127.0.0.1:17897
# 审批通过后的命令模板（P2-E 报告最终版为准）
# $ENV tools/repro/fetch_data.py --dataset realssa-sim --shard tt-000000 --proxy ...
# $ENV tools/repro/phase2/eval_dnsmos.py --audio est.wav --sr 16000
# $ENV tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real --channel mean --limit 5
```

### 3.5 回归（P2-G / P2-H）

```bash
$ENV -m pytest studio/tests -q                       # 期望 35 passed, 2 skipped
$ENV -m pytest tools/repro/tests -q                  # 19:29Z 实测 96 passed, 6 skipped（102 collected；P2-G 新增 test_phase2_*）
for f in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$f"; done   # 8/8
$ENV space/build.py --out /root/tmp/space-verify && node space/tests/check_i18n.mjs /root/tmp/space-verify   # i18n: 226
PYTHONPATH=$PWD $ENV -c "import look2hear.models, look2hear.layers, look2hear.system, look2hear.metrics, look2hear.losses, look2hear.datas, look2hear.videomodels, look2hear.utils, audio_train, audio_visual_test; print('imports OK')"
```

## 4. 文档一致性核对（v1 终检）

| 检查项 | 方法 | 结果（v1 2026-10-21T19:45Z） |
|---|---|---|
| P2-A…G 报告 vs `DATA_ACCESS_GUIDE.md` / `REPRODUCTION_PHASE1_REPORT.md` / `REPRODUCTION_PLAN.md` 数字与结论 | 逐项比对（LRS2 统计、7.03/7.71、14.1/14.7、target=s1、数据大小/许可） | ✅ 6 份报告逐项核对（`REGRESSION-docs.md` §2.4）：数字 ↔ CSV/JSON/log 一致；7.03/7.71 与 Phase 1 一致；P2-F 跨语料归因与 conf/README/PROVENANCE 证据链一致 |
| 数据集大小/许可/格式 vs HF 事实 | `docs/verification/phase1/dataset-cards/` + HF API（phase1 evidence JSON 已固化） | ✅ 2026-10-21T19:09Z 经代理复核 5 个数据集：LRS2 16,470,508,590 B / Sim 521,532,631,040 B·242 files / Real 12,842,268,608 B·422 files / LRS3·Vox 0 B，与 GUIDE 全部一致 |
| docs 相对链接 0 404 | 脚本扫描 `docs/**/*.md` + 反引号路径 | ✅ Markdown **62 文件/38 条相对链接 0 真 404**（4 条 `[...](...)` 字面量占位）；反引号 910 条 61 unresolved 已分类（55 示意/复合、1 P3 PLAN:138、2 未来产物、1 已解决、2 已处理）；P2 报告新增 128 条 19 unresolved 均复合/相对简写；外链 LRS3 深链 404 已改入口页 |
| Phase 2 新增脚本/报告链接 | 本 README §1 表格逐项核对存在性 | ✅ P2-A…F 报告 + QA 报告全部存在（§1 表格链接可点）；`tools/repro/phase2/` 脚本全部存在且 `--help` 通过；`P2D-train.log`/`P2D-train.pid`/`P2D-eval-epoch_0.csv` 已核对 |

## 5. 数据集事实速查（与 HF 一致，Phase 1 已核）

| 数据集 | gated | 规模 | 许可 | 备注 |
|---|---|---|---|---|
| LRS2-2Mix | 公开（非 gated） | 16.47 GB tar / 17.39 GB 解压；tr 20,000 / cv 5,000 / tt 3,000 组 2 s | HF 标签 apache-2.0；内容源自 LRS2/BBC，受 LRS2/TED 条款约束，**不可再分发** | 完整 AV；mouths (50,96,96)、faces、landmarks；target=s1 |
| RealSSA-Sim | `manual` | 242 files / 240 shards / **521.53 GB**（tr 441.42 / cv 39.97 / tt 40.14） | CC BY-NC-ND 4.0；非商业学术、人工审批、不可转发 | 4ch 16k PCM16 mix/target、mouths.pt [T,96,96,3]、**无 near 源波形** |
| RealSSA-Real | `manual` | 422 files / **12.84 GB**（54 seg / 40 rec / 10 spk / 8 scene / 83.99 min） | 无通用 SPDX license；研究用途 + 参与者授权，人工审批 | stereo 16k、face 224×224/25fps、mouth [T,96,96]、landmarks [T,68,2]、**无 SI-SDR 参考** |
| LRS3-2Mix | 公开但仓库无数据文件 | 仅 README（622 B，usedStorage 0） | HF 标签 apache-2.0（源语料另需申请） | 不可复现该列 |
| VoxCeleb2-2Mix | 公开但仓库无数据文件 | 仅 README（756 B，usedStorage 0） | 同上 | 不可复现该列 |

> 来源：Phase 1 的 HF API/tree 复核（`docs/verification/phase1/REGRESSION-docs-evidence.json`）与 dataset cards；
> Phase 2 若申请获批，以届时 HF 页面/数据集所有者条款为准。

## 6. 链接

- Phase 1 总报告：`../REPRODUCTION_PHASE1_REPORT.md`（仓库根 `docs/`）
- Phase 1 验证资产：`../verification/phase1/`（T1–T6 报告、QA、REGRESSION、dataset-cards）
- 复现计划：`../REPRODUCTION_PLAN.md`（Phase 2/3 状态小节由本任务维护）
- 数据访问指南：`../DATA_ACCESS_GUIDE.md`
- 理解文档：`../understanding/FINAL_PROJECT_UNDERSTANDING.md`

## 7. 维护记录

| 时间（UTC） | 动作 |
|---|---|
| 2026-10-21T18:48Z | task-25 初始化：建立索引/状态/复跑命令框架；Phase 2 各报告尚未落盘（§1 标 ⏳）。 |
| 2026-10-21T18:50Z | `REPRODUCTION_PLAN.md` 增加 Phase 2/3 状态小节（仅状态+链接）；本 README §3 复跑命令定稿。 |
| 2026-10-21T19:10Z | 回归全绿（82 passed/12 skipped；Studio 35/2、Node 8/8、i18n 226、imports 10/10）；`REGRESSION-docs.md` v0 落盘；GUIDE 9 处 RealSSA 状态同步 + LRS3 链接/§11 措辞修复；§1/§4 回填中间证据。 |
| 2026-10-21T19:45Z | **v1 终检**：P2-A…F 报告落盘并逐项核对；回归 96/6；跨语料归因（P2-F）复核；2 处报告 P3 由 owners 修复并复核；§1/§1.1/§4 回填；PLAN Phase 2/3 状态升 19:45Z；`REGRESSION-docs.md` 升 v1。 |
