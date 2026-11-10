# 09 深挖 · 论文/基准对齐与复现规划 v2（README 表 × 实测 × 缺口 × 下一步）

> **2026-11-10 校正：** 本文记录当时的 benchmark 分析。§2/§8 把 `P2B-b300-small-whole-real.csv` 错指为 LRS2 3,000 条、7.029/7.713 的证据；该文件实际为 300 条、6.916/7.492，全量结果见 [P2D CSV](../../verification/phase2/P2D-eval-falcon-full.csv)。发布权重训练谱系尚待确认，RealSSA-Sim 数值接近论文，评测协议存在差异；full3 DNSMOS 已作废。当前结论见[复现状态](../../STATUS.md)与[综合报告](../../REPRODUCTION_REPORT.md)。

> 编写：Lead（2026-10-24）· 仓库 `repro/phase2` HEAD `2beec38` · 环境 conda `falcon`（Py3.11 / torch 2.8.0+cu128 / 2×A800）
> 输入：`README.md`（结果表/效率表）、`Experiments/checkpoint/Falcon/{conf.yml,best_k_models.json}`、
> `docs/REPRODUCTION_PHASE{1,2}_REPORT.md`、`docs/PHASE2_VOLCANO_LRS2_TRAINING.md`、
> `docs/verification/phase2/P2{A..F}-report.md`、本目录 01–08 子系统深挖、以及 2026-10-24 的磁盘/日志抽验。
> 证据标记：`[代码]` `[验证]` `[推断]` `[文档]`。**本文所有“实测”数字都在 §8 给出可核对的日志/CSV 路径**。

---

## 0. 基准结果与评测条件

截至这份报告的 2026-10-24 基线，仓库已记录权重加载、模型前向、LRS2-2Mix 单卡训练和多组评测。八卡论文配方及 TFRB/GCI、NASS 对照仍需相应训练记录；[当前复现状态](../../STATUS.md)汇总了后续验证。

RealSSA-Sim test 的历史全量结果为 **SI-SNRi 7.774 dB、SDRi 8.479 dB**，有效分母分别为 9,169 和 9,166；论文表格为 7.7 / 8.2 dB。当前发布权重的完整训练来源尚待确认；本地 mean 与 ch0 的通道处理各有独立结果。

LRS2-2Mix 的发布权重结果为 **7.029 / 7.713 dB**，自训 epoch 124 的结果为 **8.858 / 9.270 dB**，论文表格为 14.1 / 14.7 dB。比较时需同时列出数据、目标方向、权重及评测协议。

后续工作包括在具备条件的节点执行 8 卡训练，并按实验记录核对数据、模型和指标。以下表格保留当时的分析，其历史判断与当前状态的差异见页首更正和[综合报告](../../REPRODUCTION_REPORT.md)。

---

## 1. 基准对齐矩阵（论文值 × 实测值 × 口径 × 结论）

| # | Benchmark / 指标 | 论文值 | 我们的实测 | 数据（split/条数） | 权重 | 协议 | 差异 | 归因 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **RealSSA-Sim SI-SNRi** | 7.7 | **7.774** | test 全量 9,169 | 官方 `falcon.pt`（small，0.538 M） | 整句、4ch→mono（论文 **channel-1=ch0**；mean 与 ch0 均已复评）、16 kHz | **+0.07** | 训练来源待确认；mean 与 ch0 为不同通道口径 | 数值接近论文；分别引用 mean 7.774/8.479、ch0 7.977/8.452 |
| 2 | **RealSSA-Sim SDRi** | 8.2 | **8.479** | 同上 | 同上 | SDR = fast_bss_eval 512-tap | **+0.28** | SDR 使用 512-tap 滤波器 | 数值与论文接近，分母为 9,166 |
| 3 | RealSSA-Sim（1000 条快速估计） | 7.7/8.2 | 7.935 / 8.754 | test 前 1000 | 官方 small | 同上 | +0.24/+0.55 | 子集偏易 | 参考（非最终） |
| 4 | **RealSSA-Real DNSMOS** | 3.276 | **full4**：P808 **3.3365**<br>OVRL **2.7941**<br>SIG 3.2226 / BAK 3.6981（full3 作废） | 54/54 段（84 min、4 场景） | 官方 small | channel=mean、`align_tail=True`、**窗口化推理**（2 s 窗/0.5 s hop）后整段拼接打分 | 论文采用 P.835；比较 full4 OVRL 与论文 3.276 | 论文结果使用 RealSSA-Sim 训练模型；本地发布权重身份待确认 | 分别记录论文 3.276 与 full4 OVRL 2.7941；窗口及通道见本行协议 |
| 5 | RealSSA-Real mixture 基线 | — | OVRL 1.7599 / P808 2.6765 | 同上 | 无（mix 直接打分） | 同上 | 分离提升 **+1.03 OVRL / +0.65 P808** | — | 参考 |
| 6 | **LRS2-2Mix SI-SNRi** | 14.1 | 官方权重 **7.029**（训练语料待确认）<br>自训 ep86 **8.618**<br>自训+对比 ep124 **8.858** | tt 全量 3,000（2 s clip） | `falcon.pt` / `epoch=86.ckpt` / `epoch=124.ckpt` | target=s1、整句 | −7.07 / −5.48 / −5.24 | 权重训练来源待确认；本地训练为单卡 bs4、无 near 真源、130 epoch，差异涉及训练与评测条件 | ❌ **未复现** |
| 7 | **LRS2-2Mix SDRi** | 14.7 | 7.713 / 9.042 / 9.270 | 同上 | 同上 | 同上 | −6.99 / −5.66 / −5.43 | 同上 | ❌ **未复现** |
| 8 | LRS2-2Mix 自训 val SI-SDR（仅内部指标） | — | 13.118（ep86）/ 13.391（ep124） | cv 5,000 | 自训 | `val_pit_sisnr = −val_loss = SI-SDR(est)` | — | **不是 SI-SNRi，不要与 14.1 比** | 参考 |
| 9 | LRS3-2Mix SI-SNRi / SDRi | 16.4 / 16.7 | 无 | HF 仓库仅 README、无数据文件 | — | — | — | 数据未发布 | ❌ **不可复现**（需作者） |
| 10 | VoxCeleb2-2Mix SI-SNRi / SDRi | 10.5 / 11.8 | 无 | 同上 | — | — | — | 数据未发布 | ❌ **不可复现**（需作者） |
| 11 | 可训练参数 | 0.5 M | **537,697（0.538 M）**<br>生效 532,129；扣对比头 523,113；+冻结 FRCNN 共 11,724,385 | — | — | 统计 audio model；`concat[2]` 5,568 闲置 | 与 README 0.5/11.72 M 一致 | — | ✅ **精确复现** |
| 12 | MACs | 10.6 G | **10.611 G** | 2 s/16 kHz/50 帧 | — | ptflops，audio 网络，视觉输入 64×64、**不含 FRCNN** | 一致 | README 同口径 | ✅ **复现（注意不含视觉编码器）** |
| 13 | CPU 延迟 | 1361.2 ms | 未测 | — | — | 论文为 Intel Xeon / RTX 3090 | — | 本机无 3090；A800 测了也不可直比 | ⚠️ 未复现（可测但口径不同） |
| 14 | GPU 延迟 | 51.3 ms（3090） | 未测 | — | — | batch 1 | — | 需 3090 或声明硬件差异 | ⚠️ 未复现 |
| 15 | RAM / GPU 显存 | 125.3 MB / 364.7 MB | 未测 | — | — | batch 1 | — | 同上 | ⚠️ 未复现 |
| 16 | 10 个 baseline 对照 | README 表 | 未重跑 | — | — | 除 Dolphin/AV-Mossformer2 外引 Swift-Net | — | README 明示“follow values reported by Swift-Net” | 引用即可，不必重跑 |

> **通道敏感性（2026-10-30 完成）**：论文 RealSSA-Sim 用 **channel-1**（代码 `--channel 0`）。我们用同一权重、同一 9,169 条、
> 同一指标分别评测 4ch-**mean** 与 **ch0**：

| 指标 | mean | ch0（论文通道） | ch0 − mean | 论文 | ch0 对论文 |
|---|---|---|---|---|---|
| SI-SNRi | 7.774 | **7.977** | **+0.202** | 7.7 | **+0.28** |
| SDRi | 8.479 | **8.452** | **−0.027** | 8.2 | **+0.25** |
| SI-SNR | 9.980 | 9.320 | −0.660 | — | — |
| SDR | 10.645 | 10.111 | −0.534 | — | — |
| PESQ / STOI | 2.094 / 0.817 | 2.004 / 0.815 | −0.090 / −0.002 | — | — |

> ⇒ 所测两种通道的提升量差异小于 0.3 dB；论文通道为 ch0。数值接近论文，完整权重和协议仍需核对。证据：`P2E-realssa-sim-ch0-{eval-20261030.log,metrics_combined-20261030.csv}`。

> 结论：第 1–2 行数值接近论文，通道与权重条件需分别说明；第 11–12 行参数及 MACs 的统计值一致；第 4、6–7 行存在协议或数值差异，第 9–10 行缺数据，第 13–15 行缺论文硬件实测。对外报告可按这四类分别列出数据、权重、协议与结果。

### 1.1 口径差异清单（同一个数字不同口径会差多少）

| 口径轴 | 变化 | 影响 | 出处 |
|---|---|---|---|
| 跨语料 vs 同语料 | 发布权重评 LRS2-2Mix：7.03，自训权重 8.86，论文 14.1 | 权重来源、训练预算及协议需分别核对 | P2-A/B/F；§8 |
| 整句 vs 窗口化 | RealSSA-Real 只能窗口化（整段 OOM）；LRS2 上窗口化比整句低 **−1.9~−2.3 dB** | DNSMOS/SI-SNRi 都受影响 | P2-B §3.2 |
| 模型 small vs big | big 比 small 高 **+0.97/+0.91 dB**（300 条配对，86% 胜） | 发布 big 不是论文 0.5 M 主模型 | P2-B |
| 通道（4ch） | mean vs ch0 | mean 7.774/8.479；论文通道 ch0 为 7.977/8.452 | 05 §3；论文及 ch0 后续评测 |
| SDR 实现 | fast_bss_eval 512-tap（当前）vs 64/2048 tap | 512→64 差 −0.35；512→2048 差 +0.26 | P2-A |
| PESQ 参考方向 | 旧实现传反（2.73）→ 现 ref=clean（1.03 在该测试信号上） | 历史 bug，已修 | 04 §2.4；F7/F8 |
| DNSMOS 长度 | <9.01 s 循环拼接（2 s→16 s、7 窗）；长段单窗 | 2 s 与 9 s 差 −0.118 OVRL；2 s LRS2 的 OVRL 1.52 不可与 2.79 比 | 04 §2.6/§5.4 |
| near 负样本 | K=0（RealSSA）/ K=1（LRS2 s2 当 near） | K=1 时 **β 退火是 no-op**，对比增益不可归因于 β | 04 §2.2/§5.3；05 §3.5 |
| 训练 vs 评测指标名 | 训练 PIT=负 SNR；`val_pit_sisnr`=SI-SDR(est)，不是 SI-SNRi | 14.1 不能与 val 13.39 比较 | 04 §2.1/§5.2 |
| 参数口径 | 537,697 / 532,129 / 523,113 / 517,545 / 11,724,385 | 引用必须带口径 | 01 §2.1/§5.4 |

---

## 2. 数字出处（可核对）

| 数字 | 出处（文件:行 / 日志） | 证据类型 |
|---|---|---|
| RealSSA-Sim 7.774 / 8.479 / PESQ 2.094 / STOI 0.817 | `/root/tmp/realssa_sim_eval_official.log`（`Final metrics` 段）；`docs/REPRODUCTION_PHASE2_REPORT.md:120-131` | 本机实测（日志在 /root/tmp，**未入仓库** ⚠️） |
| 9,169 条转换 | `docs/verification/phase2/P2E-realssa-sim-convert.log:186-192` | 本机实测 |
| RealSSA-Real DNSMOS 4 项 + 分场景 | `/root/data/falcon_eval/realssa_real_mean_full3/{summary.json,run_metadata.json,*.csv}`；报告 §5.1 表 | 本机实测（JSON 在 /root/data ⚠️） |
| LRS2 官方权重 7.029/7.713 | `docs/verification/phase2/P2D-eval-falcon-full.csv` avg 行（3000 条）；Phase1 §3.5 | 本机实测（CSV 在库） |
| 自训 ep86 8.6179/9.0415 | `/root/tmp/p2d_eval_epoch86_full.log`（`Final metrics`） | 本机实测（日志未入仓库 ⚠️） |
| 自训+对比 ep124 8.8582/9.2696 | `/root/tmp/p2d_eval_contrastive_epoch124.log`；报告 §4.1 表 | 本机实测（未入仓库 ⚠️） |
| 8 卡配方已就绪 | `configs/LRS2_2Mix_Paper_8gpu.yml`；`tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh`；`docs/PHASE2_VOLCANO_LRS2_TRAINING.md` | 代码+文档 |
| 2 卡 smoke 3 epoch 通过（3.82 it/s） | `docs/verification/phase2/LRS2-paper-2gpu-smoke.log` | 本机实测（日志在库） |
| 官方 ckpt provenance（LRS3_complex_nonmoving、ep714） | `Experiments/checkpoint/Falcon/conf.yml:99-105`；`best_k_models.json`；`studio/falcon_model/PROVENANCE.md:37-38` | 仓库快照 |
| 论文 “under our training protocol” | `README.md:228` | 仓库文档 |
| 参数/MACs | 01 §2.1/§2.2；`docs/verification/scripts/verify_{model_params,macs_04}.py` | 本机实测 |
| 0.5 M / 10.6 G / 51.3 ms | `README.md:216-232` | 论文表 |

> ⚠️ 标记是**本轮审计发现的证据缺口**：3 个关键 eval（RealSSA-Sim 全量、LRS2 自训 ep86/ep124、RealSSA-Real DNSMOS）
> 的原始日志/JSON 在 `/root/tmp`、`/root/data` 下，既没进 `docs/verification/` 也没进 git；`/root/tmp` 是临时目录，
> 一旦清理，**“已复现”就只剩报告转述**。P0 收口必须先做（§6-P0）。

---

## 3. 截至今日的复现状态（按 L1–L4）

### L1 推理/结构复现 —— ✅ 完成（强证据）

- 参数四口径 + 冻结 FRCNN 共 11,724,385；MACs 10.611 G（01 §2.1–2.2，`[验证]`）。
- 官方 `falcon.pt` audio 525 / video 137 keys，`strict=True` 零 missing/unexpected，`epoch=714`；
  `falcon-big.pt` 517/137、`SpeakerContrastiveV2`、`epoch=715`（01 §2.5）。
- 训练侧 vs Studio 侧同权重同输入 **max|Δ|=0**（01 §2.6；Phase1 指纹）。
- Studio 可跑：`pytest studio/tests` 35 passed/2 skipped（06 复验，见其文档）；窗口协议 32000/8000/1280/22720/320 与前端逐项一致（07）。
- 结构表述已定量：“窗口级因果 + 无跨窗状态 + 非逐采样流式”；lookahead 57 帧≈456 ms、端到端首变样本 9,612≈−649 ms（01 §2.7）。

### L2 评测复现 —— ⚠️ 部分

| 列 | 状态 | 说明 |
|---|---|---|
| RealSSA-Sim | 数值接近 | mean 7.774/8.479、ch0 7.977/8.452 vs 7.7/8.2；训练权重身份与完整协议仍需核对 |
| RealSSA-Real | ❌ 口径不成立 | 论文用 P.835 OVRL、3.276 属 RealSSA-trained ckpt；full4 OVRL 2.7941；发布权重训练身份待确认（2026-10-30 校正）；窗口化推理另有偏差 |
| LRS2-2Mix | ❌ | 发布权重训练语料待确认（7.03/7.71）；自训 130 ep 8.86/9.27；论文 14.1 需结合训练和评测协议核对 |
| LRS3-2Mix / VoxCeleb2-2Mix | ❌ 阻塞 | HF 仓库无数据文件，只能向作者要 |

### L3 训练复现 —— ⚠️ 进行到“单机小配方”，论文级未跑

- **LRS2-2Mix**：单卡 bs4、100 epoch + 30 epoch 对比微调（s2 当 near）= **8.858/9.270**；
  8 卡论文配方（8×bs5=40、1000 ep、early stop 30、双向 target、near=另一说话人）**已写配置+脚本+runbook，
  本机 2 卡 smoke 3 epoch 通过，尚未在火山引擎 8 卡执行**。
- **RealSSA-Sim**：只有 test（9,169 条）与官方权重评测；`tr/cv` 只在 TOS（521.5 GB），本地无转换产物；
  **没有 RealSSA 训练配置**；near 负样本官方未发布（`near.json` 全 fallback→K=0）。
- **RealSSA-Real**：无干净参考，不能训练/不能算 SI-SDR（只能 DNSMOS/WER/相似度）。

### L4 消融/效率 —— ⚠️ 少量

已做：视觉分支（置零/换人/shuffle）、整句 vs 窗口化、small vs big、mouth 偏移、协议/指标变体、
参数/MACs 复算。
未做：TFRB/GCI 组件消融（需要改模型，且会破坏官方权重 bit-identical 语义）、NASS 有/无与 β 的真实消融
（K=1 时 β no-op，需构造 K≥2 负样本）、延迟/RAM/显存（论文硬件 3090 不在手）。

---

## 4. 复现规划 v2（按优先级/依赖排序，含验收与算力）

> 与旧 `docs/REPRODUCTION_PLAN.md` 的关系：旧计划 Phase 0/1 已完成，Phase 2 的“RealSSA 待审批”已被现实超越
> （已获批/下载/归档/评测），Phase 3 的 LRS2 目标“用官方权重复现 14.1”已被证伪（需核对训练权重及评测协议）。
> 本 v2 记录 2026-10-24 的待办；后续结果见 §11 与现行状态。

### P0 —— 证据收口与文档纠偏（0.5–1 天，无 GPU；**先做**）

1. **把 3 个关键 eval 原始产物复制进仓库**：`/root/tmp/realssa_sim_eval_official.log`、
   `/root/tmp/p2d_eval_epoch86_full.log`、`/root/tmp/p2d_eval_contrastive_epoch124.log`、
   `/root/data/falcon_eval/realssa_real_mean_full3/{summary.json,run_metadata.json,*.csv}` →
   `docs/verification/phase2/`（大 CSV 可只留哈希与摘要）。
2. **修文档-磁盘漂移**：`docs/DATA_ACCESS_GUIDE.md` 仍写 RealSSA 403 待批（多处）；
   `falcon_lrs2_2mix/*/near.json` 已被一次性脚本改成 s2 负样本，但 `conversion_stats.json` / P2-C / QA 仍写 fallback(K=0)；
   仓库无生成该形态的脚本 → **补 `lrs2_2mix_to_falcon.py --near-mode other-speaker` 或写清 provenance**（05 §3.5）。
3. **修 1 个回归失败**：`pytest tools/repro/tests` 现为 104 passed / 7 skipped / **1 failed**
   （`test_phase2_pipeline` 对空串 `float('')`），先修测试或数据再冻结基线（03 §7）。
4. **历史工作区**：当时 `docs/REPRODUCTION_PHASE2_REPORT.md` 有未提交修改；证据索引与结果应随现行工作区状态更新。
- **验收**：`git status` 干净；`docs/verification/phase2/README.md` 索引含全部新证据；`pytest tools/repro/tests` 全绿；
  报告中每个“已复现”数字都能从仓库内文件复算。

### P1 —— RealSSA 评测补强与 DNSMOS 口径澄清（1–2 天，1 GPU）

1. **通道敏感性**：官方权重 × RealSSA-Sim test 9,169：`mean`（已有 7.774/8.479）vs `ch0`，各跑一次；
   目标：给出“通道选择 ≤0.? dB”的实测界，写入报告（论文未声明）。
2. **整句 vs 窗口化**：RealSSA-Sim 用 Studio 等价窗口化（2 s/0.5 s/80 ms guard）再跑一次；
   预期 −1.5~−2.5 dB；用于解释 RealSSA-Real 的窗口化偏差，并与 LRS2 的 −1.9~−2.3 dB 交叉验证。
3. **big 模型对照**：`falcon-big.pt` 在 RealSSA-Sim test 跑 300–1000 条；若 big ≈ 8.7，则说明
   “big 不是论文 0.5 M 主模型”的叙事与 LRS2 一致。
4. **DNSMOS 口径澄清**：① 用更长窗（如 10 s/5 s hop，若显存允许）或“窗口化分离→整段拼接→打分”重跑 RealSSA-Real；
   ② 同时报 SIG/BAK/**OVRL**/**P808** + `n_windows` + channel；③ 对照 mixture 基线；
   ④ ~~给作者确认 3.276 口径~~ → 2026-10-30 论文已确认是 P.835；改为：用 `--channel 0`（论文 channel-1）+ P.835 OVRL 复评，并与论文 RealSSA-trained ckpt 区分。
- **验收**：RealSSA 部分产出“通道×协议×模型×DNSMOS 口径”四张小表；论文 3.276 若仍无法确认，
  报告写“P808 +0.05 / OVRL −0.49，口径未定”。
- **估算**：每次 9,169 条整句评测 ≈ 20–40 min（A800，含 PESQ/STOI/SDR）；窗口化 ×3；DNSMOS 54 段 ≈ 10 min。

### P2 —— LRS2-2Mix 8 卡论文配方（2–3 天含排队；**候选训练路线**）

1. **数据迁移**：`/root/data/falcon_lrs2_2mix_paper`（26 GB，JSON 引用 `lrs2_rebuild` 音频）→ 训练节点；
   或 `build_lrs2_paper_data.py --copy-wavs` 生成自包含版（+~7 GB）。
2. **前置检查**（03 §5.3 已列 9 项）：8 卡可见（脚本断言 ≥8）、全局 batch 40、visual backbone、
   `PYTHON_BIN` 覆盖、`DATA_ROOT` 与 config 写死路径一致、监控读最新 `version_*`、续训 CSVLogger 版本坑、F15 smoke。
3. **启动**：`bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh`（`configs/LRS2_2Mix_Paper_8gpu.yml`）。
4. **评测**：最佳 ckpt → `audio_visual_test.py --conf_dir configs/LRS2_2Mix.yml --data_dir /root/data/falcon_lrs2_2mix`
   （tt 3,000，target=s1，与 benchmark 约定一致；**不要**用 `_paper/tt` 的 6,000 双向）。
5. **验收**：SI-SNRi ≥ **13.6**、SDRi ≥ **14.2**（论文 14.1/14.7，容差 ±0.5，允许训练随机性）；
   同时报 val 曲线、最佳 epoch、seed、与论文配方的全部差异（near=另一说话人、K=1 使 β no-op、双向 target、
   per-rank bs5 vs 4、8 vs 10 卡）。
- **风险**：① 4 卡节点会让 global batch 变 20 而不报错；② F15 per-bucket all_gather 在 8 卡上无人跑过（先 smoke）；
  ③ 数据 `_paper` 不自包含（搬节点必须带 `lrs2_rebuild`）；④ 若只到 10–12 dB，需要继续排查
  （数据混合/负样本/更长训练），**不要再归因于指标口径**。
- **算力**：8×A800 约 2–3 min/epoch（含验证）→ 1000 ep 上限 33–50 h，early stop 通常 5–15 h。

### P3 —— RealSSA-Sim 从零训练（5–7 天；可选但决定“L3 是否完整”）

1. **数据恢复**：TOS `/aispeech-cfs/voiceprint/data/multimodal/realssa_sim/{tr,cv}`（412 GB + 38 GB tar）→
   本地/节点；`realssa_to_falcon.py` 逐 shard 流式转换（tr 100,826 / cv 9,133 条；磁盘预算 ≥600 GB；
   `/root` 只剩 14 TB，够但要规划）。
2. **训练配置**：新建 `configs/RealSSA_Sim_8gpu.yml`（超参只能按 `configs/Falcon.yml` 推断：AdamW 1e-3、
   1000 ep、early stop 30、β 0→1/200、train_samples 10,000；论文 RealSSA 实际轮数未知）。
3. **near 负样本决策**（三选一并在报告声明）：A) 接受 K=0（NASS 只剩正项）；B) 用 meta 里
   `components[type=near].path` 映射 LibriSpeech 语句近似重建（无时间偏移，只能随机裁剪，非原波形）；
   C) batch/同场景其它说话人当负样本（非官方）。
4. **验收**：val SI-SNRi 随训练上升；test 目标 7.7±0.5（同分布）。若显著低于官方权重，说明配方/负样本差异，
   按数据、权重和协议分别说明差异。
- **风险**：合规（cc-by-nc-nd + gated，派生数据不得入库/转发）；8 卡时长；F15；转换 450 GB 的 IO；
  数据卡明确“fallback 不是等价复现”。

### P4 —— 消融与效率（3–5 天，1 GPU + 少量代码分支）

1. **NASS 有/无**：LRS2 上已有“无 near 8.618 → s2-near 8.858”（+0.24/+0.23）；补 PIT-only 对照与
   多 seed（当前只有 1 个 seed）。
2. **β 消融的前置条件**：K=1 时 β 是 no-op（04 §2.2），要做 β 消融必须先构造 K≥2 负样本。
3. **TFRB/GCI**：需要改模型（删/换组件）→ **必须另开分支**，不能动 `look2hear/models/**` 主线，
   否则官方权重 bit-identical 语义被破坏（01 §5.6）；只做“架构意图”对照，不参与主表。
4. **效率**：在 A800 上测 CPU/GPU 延迟、RAM、显存（batch 1，2 s/50 帧）；能借到 3090 就补论文硬件数字，
   否则报告写“硬件不同，仅作相对比较”。MACs 口径沿用 10.611 G（不含视觉编码器）。
5. **窗口化/流式**：复用 P2-B 的窗口化模拟与 Studio 协议，给出“整句→流式”的指标-延迟折中曲线。
- **验收**：一张消融表（每行 = 配置、SI-SNRi/SDRi、Δ、备注“非论文配方/单 seed”）。

### P5 —— 最终报告与差异声明（1–2 天）

产出 `docs/REPRODUCTION_REPORT.md`，必须包含：
① 对齐矩阵（§1，四类状态分别标注）；② 每列的数据/权重/协议/指标版本；③ 与论文的全部差异（论文未公开、
near 负样本、8 vs 10 卡、通道、窗口化、DNSMOS 口径）；④ 随机种子/硬件/耗时；⑤ 复跑命令（P0 后的仓库内路径）；
⑥ 证据索引（每条结论 → `docs/verification/...` 文件）。
**不要**把“跨语料 7.03/7.71”写成“LRS2 列复现”；**不要**把 val SI-SDR 13.39 与论文 14.1 比。

### 4.1 时间线（可并行）

| 周 | 主线 | 并行 |
|---|---|---|
| W1 | P0 收口 + P1 RealSSA 补强 | 作者沟通（issue #1 / DNSMOS 口径 / LRS2 ckpt） |
| W1–W2 | P2 LRS2 8 卡训练（等节点） | P4 效率（无需集群） |
| W2 | P2 评测 + 差异分析 | P4 NASS/seed |
| W3 | P3 RealSSA 训练（若资源允许） | P5 报告框架 |
| W3–W4 | P5 定稿 | — |

---

## 5. 与旧计划（`docs/REPRODUCTION_PLAN.md`，2026-10-21）的差异

| 旧计划条目 | 旧状态 | 今日状态 | 判定 |
|---|---|---|---|
| L2 RealSSA-Sim：需申请+下载+转换后评测 | 🟡 gated 待批 | ✅ 已批/已下载/已转换/已评测 7.774/8.479 | **已完成（旧计划过时）** |
| L2 LRS2-2Mix：用官方权重复现 14.1/14.7 | 计划用 `falcon.pt` | 发布权重训练来源待确认，实测 7.03/7.71；还需核对训练及评测协议 | **旧假设被证伪** |
| L3 训练 “缺 RealSSA-Sim 访问权限 + 格式转换 + P0 修复” | ❌ 不可做 | 访问已解决、格式转换工具已落地、训练入口 P0 已修；仍缺 near 负样本/8 卡/RealSSA 训练配置 | **部分解除** |
| L3 训练 LRS2-2Mix | 未列入（原以为不必） | 已单卡自训 8.858/9.270，8 卡配方待跑 | **新增主线** |
| Phase 0/1（F1–F12 修复、Gate G1） | 待做 | ✅ 已完成（F1–F9/F12/F14 一半/F17；F13 有意保留；F15 未修） | 已完成 |
| 风险 R3（RealSSA gated 审批） | 高 | 已解除 | 关闭 |
| 风险 R5（LRS3/Vox 无数据） | 高 | 仍无数据 | 保持 |
| 风险 R8（指标 bug） | 高 | 已修并以 RealSSA-Sim 闭合验证 | 关闭 |
| 风险 R4（4ch 协议未知） | 中 | 仍未知；已用 mean 并记录 | 保持（P1 补 ch0） |
| R12（参数口径） | 中 | 已给四口径 | 关闭（口径写清即可） |
| `DATA_ACCESS_GUIDE.md` 的 403/待批状态 | — | 与磁盘/事实不符（已获批/归档） | **需修（P0）** |
| `docs/REPRODUCTION_PHASE2_REPORT.md` 的 §5.4 | — | 有未提交的“下载归档完成”新增节 | 需提交（P0） |

---

## 6. 待补实验清单（按“信息增量/成本”排序）

| 优先 | 实验 | 目的 | 成本 | 依赖 |
|---|---|---|---|---|
| ★★★ | LRS2-2Mix 8 卡论文配方训练 + tt 评测 | 唯一可能把 14.1 从“未复现”变“复现/接近”的动作 | 5–15 h×8 卡 | 8 卡节点、数据迁移 |
| ★★★ | 证据收口（P0） | 让“已复现”可被第三方复核 | 0.5–1 人日 | 无 |
| ★★☆ | RealSSA-Sim ch0 vs mean、整句 vs 窗口化、big 对照 | 补全评测口径边界 | 1–2 天×1 卡 | 已有数据 |
| ★★☆ | RealSSA-Real DNSMOS 口径（长窗/整段拼接 + P808/OVRL 双报） | 判定 3.276 口径、缩小 −0.49 | 0.5–1 天 | 已有数据 |
| ★★☆ | 作者沟通：LRS2 ckpt/配置、DNSMOS 口径、论文 | 可能直接消掉 LRS2 与 DNSMOS 两个缺口 | 1 封邮件/issue | — |
| ★☆☆ | RealSSA-Sim 从零训练 | 补 L3 完整性 | 5–7 天×8 卡 | TOS 数据、near 方案 |
| ★☆☆ | NASS 有/无、多 seed、K≥2 β 消融 | L4 消融 | 2–3 天×1–8 卡 | LRS2 训练管线 |
| ★☆☆ | 延迟/RAM/显存（A800；若能借 3090 更佳） | 效率表 | 0.5 天 | — |
| ☆☆☆ | TFRB/GCI 消融 | 架构贡献 | 3–5 天 + 代码分支 | 作者确认（会破坏 bit-identical） |
| ☆☆☆ | LRS3-2Mix / VoxCeleb2-2Mix | 两列不可复现的补全 | 不可估 | 作者给数据 |

---

## 7. 未解问题

1. **~~论文未公开~~**（2026-10-23 已公开 arXiv:2610.32749）：论文给出 −SI-SNR、τ 0→1/1000 epoch、8×3090/bs4/seed42 等；
   benchmark 是否分别训练 ckpt 仍未完全声明。详见 `docs/understanding/PAPER_ALIGNMENT_20261030.md`；
   README 明示 “Paper link will be added when available”。
2. **官方权重 provenance**：`conf.yml` 包含 `LRS3_complex_nonmoving` 和 epoch 714 线索；数值相近无法确认权重的训练数据或协议。
3. **论文 RealSSA 列是否为同一 ckpt**：若论文各 benchmark 列分别训练，则 RealSSA 列也应有专属 ckpt；
   我们只有 1 个 `falcon.pt`，无法验证。
4. **LRS2 列 14.1 的训练配方未知**：8 卡配方是“最接近的合理复现”，不是论文配方；即使跑到 13.6，也只是“接近”。
5. **DNSMOS 3.276**：论文采用 P.835；full4 的 OVRL 为 2.7941，权重和窗口协议仍需对应。
6. **LRS3-2Mix / VoxCeleb2-2Mix 数据不存在**：HF 仓库仅 README；作者 issue #1 未见回复（截至 2026-10-24）。
7. **F15 DDP per-bucket all_gather 未在 8 卡验证**：小 val 子集/分桶不均可能挂死；8 卡开跑前先 smoke。
8. **`concat_layers[2]` 闲置 5,568 参数**：是否为上游漏调用未知；修它会破坏官方权重 bit-identical，
   只能由作者确认后另开分支。
9. **效率表硬件**：论文使用 RTX 3090 / Intel Xeon，本机使用 A800；比较时列出设备和测量协议。
10. **证据时效性**：3 个关键 eval 日志在 `/root/tmp`，随时可能被清理（P0 必须固化）。

---

## 8. 验证记录（本文数字的核对路径）

| 验证 | 命令/文件 | 结果 |
|---|---|---|
| RealSSA-Sim 全量 9,169 官方权重 | `/root/tmp/realssa_sim_eval_official.log`（2026-10-22T18:41） | `si-snr_i 7.774307 / sdr_i 8.479395 / pesq 2.093770 / stoi 0.817290`，exit=0 |
| RealSSA-Real DNSMOS | `/root/data/falcon_eval/realssa_real_mean_full3/summary.json`（54/54 OK） | P808 3.3285、OVRL 2.7901、SIG 3.2209、BAK 3.6921；mix 基线 2.6765/1.7599 |
| LRS2 官方权重 3,000 条 | `docs/verification/phase2/P2D-eval-falcon-full.csv` + Phase1 §3.5 | 7.029 / 7.713 |
| LRS2 自训 ep86 | `/root/tmp/p2d_eval_epoch86_full.log` | 8.617919 / 9.041506 |
| LRS2 自训+对比 ep124 | `/root/tmp/p2d_eval_contrastive_epoch124.log` | 8.858192 / 9.269574 |
| 8 卡配方静态检查 | `configs/LRS2_2Mix_Paper_8gpu.yml`（bs5、gpus:-1、1000 ep、β 0→1/200、双向数据） | 历史配置 global batch 40；论文为 8×bs4 = 32 |
| 2 卡 smoke | `docs/verification/phase2/LRS2-paper-2gpu-smoke.log` | 3 epoch 完成，3.82 it/s，ckpt/best 正常 |
| 官方权重 provenance | `Experiments/checkpoint/Falcon/conf.yml:99-105`；`README.md:228` | `LRS3_complex_nonmoving`；配置线索与论文训练协议需分别核对 |
| 参数/MACs | `verify_model_params.py` / `verify_macs_04.py`（01 §2.1–2.2） | 537,697 / 10.611 G |
| 训练进程现状 | `ps aux \| grep audio_train`；`nvidia-smi` | 无训练进程；2×A800 空闲（0 MiB） |

---

## 9. 证据审计补遗（来自 08 深挖；2026-10-24）

> 08 对 `docs/verification/**` 做了全量证据链审计（437 行）。结论：**数字本身都能在 CSV/JSON/日志对上，但“已在仓库内可复核”的证据只占一部分**。
> 分级定义：L1 = 硬证据且已入库；L2 = 本机实测但证据未入库/已丢失；L3 = 只有报告转述；L4 = 估计/未执行。

| 结论 | 级别 | 关键问题 |
|---|---|---|
| 权重 sha256 / 参数量 / 4 条指纹 | **L1** | 脚本 + `baseline-fingerprint.json` 入库 |
| P2-F 指标对拍 4.8e-8 dB | **L1（对 core.py）/ L2（对 Phase-1 CSV）** | Phase-1 CSV 已被覆盖 |
| P2-A 14 变体 ≤0.34 dB、P2-B 变体矩阵 | **L1** | JSON/CSV 入库；需 LRS2 eval 数据 |
| RealSSA-Sim 7.774/8.479 | **L2** | 逐句 CSV 在 `Experiments/`（被 gitignore），日志在 `/root/tmp` |
| RealSSA-Real DNSMOS | **L2** | 日志未入库，summary.json 在 `/root/data` |
| 官方权重 LRS2 7.03/7.71 | **L2（偏弱）** | 逐句 CSV 已被 RealSSA 评测**覆盖** |
| 自训 8.618/9.042（ep86） | **L2（偏弱）** | CSV 被 ep124 覆盖；只有 `/root/tmp` 日志 |
| 自训+对比 8.858/9.270（ep124） | **L2** | CSV 在磁盘但被 gitignore |
| 训练 100 epoch 完成 | **L2** | `P2D-train.log` 未入库；报告“训练中/最佳 13.118”已过期 |
| LRS2 8 卡 / 火山引擎 | **L3** | 纯 runbook，零执行日志、零结果 |
| big 全量 ≈8.0/8.5 | **L4** | P2-B 自己标注“仅估计，未跑全量” |

**必须先修的证据问题（P0 的具体化）**：

1. **`*.log` 被根 `.gitignore:11` 全量忽略**：`docs/verification` 磁盘 226 文件、git 只跟踪 129；
   `runs/` 45 个日志 **0 入库**；phase2 目录 64 文件里 **29 个日志未入库**（含 `P2D-train.log`、
   `P2E-realssa-real-full3.log`、`P2F-protocol-check.log`、`LRS2-paper-2gpu-smoke.log`）。
   → 报告里“见日志”的引用，第三方 clone 后不存在。
2. **同名输出路径覆盖**：`Experiments/checkpoint/Falcon/results/metrics_combined.csv` 先存 LRS2 全量评测、
   后又被 RealSSA-Sim 评测覆盖；`LRS2_2Mix/results/metrics_combined.csv` 先 ep86、后 ep124 →
   `protocol_check.py:49` 的“vs Phase 1 CSV 1.8e-7 dB”已**不可重跑**。
3. **`P2D-eval-falcon.pt.csv` 实际是 300 条子集（6.916/7.492）**；全量结果为 7.03/7.71。
4. **索引/状态过期**：`docs/verification/phase2/README.md`（HEAD `af74cca`）、`REPRODUCTION_PLAN.md`、
   `DATA_ACCESS_GUIDE.md`、`P2E-report.md` 仍写 “RealSSA 403 / 训练进行中”；实际数据已归档、
   **训练已于 2026-10-22 19:54 结束（100 epoch）**。
5. **环境锁不可移植**：`conda-env-lock.txt` 有 **18 个包 `@ file:///root/wheelhouse`** 绝对路径；
   `pypesq` shim 不在锁文件；`AGENTS.md` 写 lock 111 行、实际 115 行。
6. **回归现状**：`pytest tools/repro/tests` = **104 passed / 1 failed / 7 skipped**（QA 记录“111/1”已过期）：
   1 failed 是 RealSSA 全量 CSV 的 3 个空 SDR 单元格触发 `test_phase2_pipeline.py:245`；
   7 skip 是 Phase-1 diff 守卫因改动已 commit 而失效。
7. **git 同步**：`repro/phase2` = `2beec38`；`origin/repro/phase2` = `9121598`（落后 2 个提交，但这两个提交已在
   `origin/main`）；工作区仅 `docs/REPRODUCTION_PHASE2_REPORT.md` 的 §5.4 +9 行未提交。
8. **协议口径风险（科学层面）**：RealSSA-Sim 评测是**整段 4.6–5.8 s clip** 推理（`AVSpeechEvalDataset` 不裁剪），
   不是 2 s 流式；与论文 7.7/8.2 的“同协议”**未被证明**（P2-B 显示整句 vs 窗口相差 ~1.9–2.3 dB）。
   → 报告必须写“整段推理、channel=mean、PESQ/STOI 同口径”，并说明论文协议未知。

**对 §1 矩阵的两处更新**：

- **LRS2 自训的“最佳”并非 ep86**：100-epoch 单卡 run 的 `metrics.csv` 里真实最佳 val 是 **epoch=99（13.2014）**，
  ep86 为 13.1182；而全量评测用的是 **ep86**（8.618/9.042）。**ep99 后续已评测，结果为 8.726/9.122** → P0 应补一次
  （补测记录见 §11）。
- 官方权重 LRS2 全量 7.03/7.71 的逐句 CSV 已被覆盖 → 证据降级为 L2；可用 `P2B-b300-small-whole-real.csv`
  （300 条，7.492）或重跑全量恢复。

---

## 10. 一句话对外口径（给作者/导师/汇报）

> 参数统计为 0.538 M、MACs 为 10.611 G；同一权重和输入下，训练侧与 Studio 的 CPU 前向最大差为 0。RealSSA-Sim 的 mean 结果为 7.774/8.479、ch0 为 7.977/8.452，论文为 7.7/8.2；完整训练权重身份仍需核对。LRS2 官方权重为 7.029/7.713，自训 ep124 为 8.858/9.270，论文为 14.1/14.7。RealSSA-Real 论文采用 P.835，full4 OVRL 为 2.7941。

## 11. 执行记录（2026-10-24 P0 收口）

按 §4 P0 执行（用户授权），完整清单见 `docs/verification/phase2/P0-FIX-report.md`：

1. **证据固化**：RealSSA-Sim 全量日志/逐句 CSV、RealSSA-Real DNSMOS 三件套、LRS2 ep86/ep124 日志与 ep124 CSV、
   官方权重评测日志、两条训练曲线全部复制到 `docs/verification/phase2/`；`.gitignore` 放行
   `docs/verification/**/*.log|csv` 与 `runs/`（109 文件/3.2 MB 可入库）。
2. **补测（已完成）**：ep99（真实最佳 val 13.2014）**8.726/9.122**；官方权重全量 **7.029/7.713**（与 Phase-1 完全一致）
   —— `P2D-eval-epoch99.{log,csv}`、`P2D-eval-falcon-full.{log,csv}`（各 3,003 行）。
3. **可复现性修复**：`--near-mode other-speaker`（K=1 s2 负样本，与磁盘 near.json 5/5 逐条一致 + 3 单测）；
   CSV 空串回归（`_as_float`）；diff-scope 守卫白名单；conda lock 去 18 个绝对路径。
4. **状态校正**：GUIDE / PLAN / phase2 README / P2E-report / PHASE2_REPORT / FINAL / AGENTS。
5. **回归**：`tools/repro/tests` 109 passed / 6 skipped / 0 failed；`studio/tests` 35/2。
6. **P1 部分**：RealSSA-Sim ch0 转换完成（9,169 条/30 GB），评测随后跑（`P2E-realssa-sim-ch0-*`），
   用于回答"通道协议是否必须声明"。

仍未执行：LRS2 8 卡论文配方（候选训练路线）、RealSSA-Sim 从零训练、3090 延迟。

### 11.1 2026-10-30：上游同步、论文公开、仓库收口

- **上游同步**：`git fetch upstream` 后 `689555c → 439cbe9`（2 个提交：HF 模型下载计数、README/项目页加 arXiv 链接），
  已合并进 `main`；唯一冲突 `audio_visual_test.py` 手工解决——保留 Phase-1 重写版入口，并移植上游的
  “best-effort 触发 HF 下载计数”代码块。
- **论文公开**：arXiv **2610.32749**（2026-10-23，28 页）。摘要确认 RealSSA-Sim 含 **1–5 个动态近场干扰**、
  远场场景另有 **10–15 个远场干扰**，Falcon 用 **encoder-space speaker contrastive loss** 抑制近场泄漏；
  因此本文 §2/§7 关于“论文未公开、协议只能推断”的结论作废，协议细节见 `PAPER_ALIGNMENT_20261030.md`。
- **仓库收口**：分支统一为 **main**（删除本地/远端 `repro/phase1`、`repro/phase2`），移除多余 worktree `wt-main`；
  新增提交：`714e1a5`（RealSSA mouths.pt 原子写）、`d9ec1e8`（上游合并）、`3d36b78`/`d076be9`（同步与论文校正文档）。
  已 push 到 **`origin/main`**（当前值见 `docs/STATUS.md`）；远端只剩 `main`；tag `repro-phase2-freeze-20261024` 为 10-24 历史快照，最新冻结 tag 见 STATUS。
- **论文校正要点**（详见 `PAPER_ALIGNMENT_20261030.md`）：DNSMOS 3.276 = **P.835** 且属 RealSSA-trained ckpt
  （同 ckpt OVRL 2.79 不可比）；RealSSA-Sim 论文用 **channel-1**（`--channel 0`），我们此前用 mean；
  论文训练配方 = **−SI-SNR、τ 0→1/1000 epoch、8×3090/bs4/seed42**（仓库 `pairwise_neg_snr` 与 β/200 属偏离）；
  near 负样本可按 App A.2 由 LibriSpeech+RIR 重建。
- **2026-10-30 补跑**：ep86 逐句 CSV 已重跑恢复（8.618/9.042，与历史日志一致）；RealSSA-Sim **ch0 全量评测运行中**
  （9,169 条）→ **已完成：ch0 SI-SNRi 7.977 / SDRi 8.452，对论文 +0.28/+0.25；与 mean 差 +0.20/−0.03 dB（通道影响 <0.3）**。
