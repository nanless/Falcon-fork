# Phase-2 B 报告：Falcon-Big vs Small、整句 vs 窗口化推理、mouth 分支贡献

> 分支 `repro/phase2`（HEAD `af74cca`，未 commit）· owner `model-core`（task-19）· 2026-10-21
> 环境：conda `falcon`（Python 3.11.17 / torch 2.8.0+cu128，2×A800；本任务用 GPU 0/1 各一进程，单卡 batch=1）
> 数据：`/root/tmp/lrs2_eval_tt_full`（LRS2-2Mix tt 3000 条，target=s1，2 s/16 kHz，mouth 3ch `.pt` → 预处理后 `[50,88,88]`）
> 权重：`/root/.cache/falcon_weights/falcon.pt`（small，epoch 714）、`falcon-big.pt`（big，epoch 715）
> 指标：Phase-1 修正后的 `look2hear.metrics.ALLMetricsTracker`（SI-SDR / SDR 定义见 `look2hear/metrics/core.py`；**越高越好**）

## 1. 结论（TL;DR）

1. **big 比 small 好，但只多 ~1 dB**：300 条同句配对，**big − small = +0.971 dB SI-SNRi / +0.911 dB SDRi**
   （258/300 句获胜，86%）。绝对：big **7.887/8.403** vs small **6.916/7.492**（同 300 条子集；small 全量 3000 条基线为 7.029/7.713）。
   ⇒ **论文 LRS2-2Mix 14.1/14.7 与本地 ~7 dB 的差距不是模型规模/权重版本造成的**（换 big 只能补 ~1 dB）。
2. **Studio 窗口化推理协议比整句推理低 ~2 dB**：同句配对 **small −1.912 dB SI-SNRi / −1.730 dB SDRi**、
   **big −2.318 / −2.133 dB**（均 300 条，窗口化仅 11% 句获胜）。窗口化后 small 5.004/5.762、big 5.569/6.270。
   信号级诊断：窗口化输出与整句输出 **SNR=16.06 dB、corr=0.980**，块边界跳变比与整句同量级（5.16 vs ~4.8）⇒
   **模拟实现正确，差距来自协议本身**（首块只有 0.58 s 真实上下文 + 边缘零填充 + 分块上下文），不是 bug。
   ⇒ 用整句协议评估 Studio 的流式能力会**高估约 2 dB**。
3. **视觉（mouth）分支是刚需**：把 mouth 裁片置零后 small 从 6.916 崩到 **−8.173**（Δ**−15.089**，300 条中仅 4 句变好）；
   把裁片在时间轴上滚动 1 s（保留真实视觉统计但破坏对齐）后为 **−5.977**（Δ**−12.894**）。
   big 对"零 mouth"更鲁棒（Δ−4.375），但仍显著下降。⇒ **音视频融合确实在工作**，"视觉分支没生效"不成立。
4. **三轴合计不能解释 7 dB 差距**：模型版本最多 +1 dB；窗口化协议只影响推理模式（整句协议本身仍是 7.03）；
   视觉分支正常。**剩余 ~6 dB 必须来自数据/混合协议/指标定义**（LRS2-2Mix 的混音构造、target 选择、SDR 定义或论文测试集口径），
   属 P2-A（`gap_analysis.py`）/P2-F（`protocol_check.py`）的结论范围；本报告给出的是"已排除项"。
5. **GPU 已释放**：本任务所有评测进程已结束（`nvidia-smi` 无本任务进程），big 全量 3000 条评测按 Lead 要求**未继续**（见 §6 局限与排期）。

## 2. 方法

### 2.1 三条对照轴

| 轴 | 取值 | 说明 |
|---|---|---|
| variant | `small` / `big` | small：`FalconSpeakerContrastive`（audio 537,697 + video 11,186,688 = 11,724,385）；big：Studio 版 `SpeakerContrastiveV2`（audio **1,158,617** + video 11,186,688 = **12,345,305**），各自 `strict=True` 加载对应 ckpt 的 `audio_model.*`/`video_model.*`（两版视觉骨干权重**不同**，T5 已证 60/137 张量不同） |
| protocol | `whole` / `windowed` | `whole` = 整段 2 s 一次前向（`audio_visual_test.py` 的做法）；`windowed` = 复刻 Studio：2 s 窗 / 0.5 s hop / 80 ms guard / 320 样本 crossfade，按 `studio/server.py::OutputSlicer` 语义逐块取 `pred[22720:30720]` 拼接 |
| visual | `real` / `zero` / `shuffle` | `real`=数据集裁片；`zero`=全零裁片（预处理后恒值）；`shuffle`=时间轴 roll 1 s（保留真实视觉特征但破坏音画对齐） |

### 2.2 评测协议（与 Phase 1 一致）

`AVSpeechEvalDataset`（mono 守卫已修）+ `ALLMetricsTracker`：SI-SDR / SI-SDRi / SDR（fast_bss_eval，filter_length=512）/ SDRi / PESQ / STOI；
SI-SDRi = SI-SDR(est) − SI-SDR(mix)，**正值 = 比混合信号好**。逐句 CSV + avg/std 落盘。

### 2.3 复跑命令

```bash
cd /root/code/github_repos/Falcon-fork
export PYTHONPATH=$PWD
ENV=/root/miniforge3/envs/falcon/bin/python

# ① 300 条：small/big × whole/windowed（真实 mouth）
CUDA_VISIBLE_DEVICES=0 $ENV tools/repro/phase2/compare_variants.py \
  --variants small --protocols whole,windowed --visuals real --limit 300 --tag b300
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/compare_variants.py \
  --variants big   --protocols whole,windowed --visuals real --limit 300 --tag b300

# ② 视觉消融（300 条，整句）
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/compare_variants.py \
  --variants small --protocols whole --visuals zero,shuffle --limit 300 --tag b300
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/compare_variants.py \
  --variants big   --protocols whole --visuals zero --limit 300 --tag b300

# ③ 窗口化模拟的信号级诊断（CPU 即可）
$ENV tools/repro/phase2/windowed_diag.py --limit 5 --device cpu

# ④ 汇总/配对统计
$ENV tools/repro/phase2/analyze_p2b.py --dir docs/verification/phase2 | tee docs/verification/phase2/P2B-analysis.log
```

## 3. 结果

### 3.1 指标汇总（300 条子集，除非注明）

| variant | protocol | visual | n | SI-SNR | SI-SNRi | SDR | SDRi | PESQ | STOI |
|---|---|---|---|---|---|---|---|---|---|
| small | whole | real | 300 | 11.856 | **6.916** | 12.533 | **7.492** | 2.276 | 0.908 |
| small | windowed | real | 300 | 9.944 | **5.004** | 10.804 | **5.762** | 1.949 | 0.884 |
| small | whole | zero | 300 | −3.233 | **−8.173** | −0.062 | **−5.103** | 1.355 | 0.466 |
| small | whole | shuffle | 300 | −1.038 | **−5.977** | 1.685 | **−3.357** | 1.363 | 0.540 |
| big | whole | real | 300 | 12.827 | **7.887** | 13.445 | **8.403** | 2.503 | 0.921 |
| big | windowed | real | 300 | 10.508 | **5.569** | 11.311 | **6.270** | 2.093 | 0.893 |
| big | whole | zero | 300 | 8.452 | **3.512** | 9.309 | **4.267** | 2.000 | 0.813 |
| *small* | *whole* | *real* | **3000** | 12.023 | **7.029** | 12.813 | **7.713** | 2.196 | 0.916 |

> 注：`metrics_combined.csv` 的列序是 `snt_id,sdr,sdr_i,si-snr,si-snr_i,pesq,stoi`（**SDR 在 SI-SNR 之前**），本表已按表头（SI-SNR / SI-SNRi / SDR / SDRi）重排。
> 最后一行是 Phase-1 全量基线（`Experiments/checkpoint/Falcon/results/metrics_combined.csv`），用于说明 300 条子集与全量相差约 0.1 dB
> （子集略难），因此 big 300 条 7.887/8.403 的可比全量估计约为 **8.0/8.5**（仅估计，未跑全量）。

### 3.2 同句配对（配对差 = candidate − baseline，dB）

| 对照 | n | ΔSI-SNRi 均值 | std | 获胜 | 胜率 |
|---|---|---|---|---|---|
| **big vs small**（whole, real） | 300 | **+0.971** | 1.697 | 258/300 | **86.0%** |
| small windowed vs whole | 300 | **−1.912** | 2.513 | 33/300 | 11.0% |
| big windowed vs whole | 300 | **−2.318** | 2.908 | 33/300 | 11.0% |
| small zero-mouth vs real（whole） | 300 | **−15.089** | 10.860 | 4/300 | 1.3% |
| small shuffled-mouth vs real（whole） | 300 | **−12.894** | 9.890 | 7/300 | 2.3% |
| big zero-mouth vs real（whole） | 300 | **−4.375** | 5.661 | 26/300 | 8.7% |

### 3.3 窗口化模拟的信号级诊断（`windowed_diag.py`，5 条，CPU）

| idx | SNR(est_win, est_whole) dB | corr | 窗口化边界跳变比 | 整句边界跳变比 |
|---|---|---|---|---|
| 0–4 | 13.59 / 15.38 / 22.92 / 8.98 / 19.46 | 0.979–0.998 | 3.1–6.9 | 2.9–5.6 |
| **均值** | **16.06 dB** | **0.9797** | 5.16 | ~4.8 |

⇒ 窗口化输出与整句输出高度一致（16 dB SNR），分块拼接没有引入异常跳变；~2 dB 的指标差来自协议上下文差异（首块 0.58 s 有效音频、
窗内零填充、每块只看 2 s），属**协议效应**而非实现错误。

## 4. 分析

1. **模型版本轴**：big（12.35 M，audio 1.16 M）稳定优于 small（+0.97 dB，86% 句获胜），但提升有限且两版视觉骨干不同。
   这与"论文 0.5 M 参数即达到 14.1 dB"的叙事并不矛盾（big 是更宽变体，不是论文主模型），但**不能把 7 dB 差距归因于权重/规模**。
2. **推理协议轴**：整句 → 窗口化损失 ~1.9–2.3 dB。原因分解（[推断]，与 `06-studio-backend.md` §4.1/§5.5 一致）：
   首块左上下文 1.42 s 为静音零填充（只有 0.58 s 真实音频）；每块独立归一化/上下文仅 2 s；guard 80 ms 丢弃末端；
   320 样本 crossfade 平滑但引入少量低通。**Studio 的 2 s/0.5 s 流式设计本身就会比离线整句低约 2 dB**，这是产品侧应记录的预期。
3. **视觉分支轴**：small 在 mouth 置零时崩溃到 −8.17 dB（比混合信号还差 8 dB），shuffle 也 −5.98 dB；big 置零 −4.38 dB。
   说明**音视频融合是有效且被重度使用的**（尤其 small，可能因为其音频容量小、更依赖视觉线索）；"视觉分支失效"假设被否定。
4. **对 7 dB gap 的定量界定**：small 全量 7.03/7.71 vs 论文 14.1/14.7 → 差 **−7.07/−6.99 dB**；
   big 全量估计 8.0/8.5 → 仍差 **−6.1/−6.2 dB**；窗口化只会更低。⇒ 剩余差距不能由本任务的三个轴解释，
   需 P2-A/P2-F 从数据混合协议（SNR 范围、说话人对、target 选择）、指标定义（SDR/SI-SDR 实现差异）、
   论文测试集口径（是否同 3000 条、是否整段、是否 4ch→mono 方式）继续排查。

## 5. 产物清单

| 文件 | 内容 |
|---|---|
| `tools/repro/phase2/compare_variants.py` | 评测主脚本（variant × protocol × visual，逐句 CSV + summary JSON） |
| `tools/repro/phase2/analyze_p2b.py` | 汇总 + 同句配对统计（CSV 驱动，容忍并行覆盖） |
| `tools/repro/phase2/windowed_diag.py` | 窗口化 vs 整句信号级诊断 |
| `docs/verification/phase2/P2B-b300-{small,big}-{whole,windowed}-real.csv` | 逐句指标（各 300 行 + avg/std） |
| `docs/verification/phase2/P2B-b300-small-whole-{zero,shuffle}.csv`、`P2B-b300-big-whole-zero.csv` | 视觉消融逐句指标 |
| `docs/verification/phase2/P2B-analysis.log` | 汇总/配对表（本报告 §3.1/§3.2 的来源） |
| `docs/verification/phase2/logs/P2B-*.log` | 运行日志（含每 100 句进度、每 run 的 [RESULT] 行、wall time） |

## 6. 局限与后续排期

1. **big 全量 3000 条未跑**：按 Lead 的 GPU 协调要求（train-sys 即将启动 LRS2-2Mix 正式训练），big 全量在跑到 **100/3000**
   时主动终止（当时 100 条 SI-SNRi=7.910，与 300 条 7.887 一致）。单独排期时预计 ~37 min（300 条耗时 225 s，含 PESQ/STOI/SDR 计算）。
   复跑命令：`CUDA_VISIBLE_DEVICES=<空闲卡> $ENV tools/repro/phase2/compare_variants.py --variants big --protocols whole --visuals real --limit 0 --tag full`。
2. 窗口化协议只在 **2 s 片段**上模拟：LRS2-2Mix 每条恰为 2 s，因此窗口化=首块 0.58 s 有效音频 + 后续块整窗；更长句子的窗口化损失可能不同。
   > 说明：`docs/verification/phase2/P2B-full-big-whole-real.csv` 是 big 全量被主动终止时留下的**空文件（0 字节）**，可忽略；
   > `P2B-summary-b300.json` 是并行运行时的旧汇总（仅含 small 两行，已被带 variant 名的新 summary 取代），分析脚本以逐句 CSV 为准。
3. 窗口化 mouth 对齐取"窗口起点最近帧 + 越界边缘复制"（±1 帧 ≈ 40 ms 不确定性）；Studio 前端真实帧窗对齐见 `07-studio-frontend.md`。
4. `shuffle` 只做时间 roll，未做跨说话人替换；如需更强控制可加"用另一条样本的 mouth"（脚本预留 visual 扩展位）。
5. **GPU 已释放**：本任务无残留进程（`nvidia-smi` 中 GPU 占用来自其它任务）；后续变体实验按 Lead 要求优先单卡小 batch/CPU。
