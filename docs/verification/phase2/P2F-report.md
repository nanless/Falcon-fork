# Phase-2 F 报告：LRS2-2Mix 指标/协议第三方独立验证（P2-F）

> 分支 `repro/phase2`（HEAD `af74cca`，未 commit）· owner `studio-backend`（task-23）· 2026-10-21
> 环境：conda `falcon`（Python 3.11.17 / torch 2.8.0+cu128 / fast-bss-eval 0.1.4 / PL 2.6.6，2×A800）
> 数据：`/root/tmp/lrs2_eval_tt_full/tt`（3000 条 LRS2-2Mix tt：mix/target=s1/mouth `.pt`，2 s/16 kHz）、
> 原始包 `/root/data/lrs2_2mix/lrs2_rebuild`（audio/wav16k/min/tt/{mix,s1,s2}、mouths npz、create_mix 脚本、scaling.mat）
> 权重：`/root/.cache/falcon_weights/falcon.pt`（small，epoch 714）、`falcon-big.pt`（big，epoch 715）
> 产物：`tools/repro/phase2/protocol_check.py`、`docs/verification/phase2/P2F-results.json`、
> `docs/verification/phase2/P2F-protocol-check.log`、本报告。
> 独立性声明：本任务的 SI-SDR/SI-SDRi/SDR 为**自写独立实现**（纯 NumPy + 直接调用 `fast_bss_eval`），
> 不依赖 T3 的 `look2hear/metrics/core.py`；`core.py` 与 Phase-1 CSV 只作为**被验证对象**参与对拍。
> GPU：本任务 GPU 实验（N=50 推理、N=20 mouth 消融）在 Lead 通知 GPU 协调前已完成并结束；
> 之后全部改为 CPU 复算（`data/recheck/static` 阶段），当前无本任务 GPU 进程/显存占用（GPU 已释放）。

## 1. 结论（TL;DR）

1. **指标实现可信，误差远小于验收阈值。** 用我自己的解析式 SI-SDR/SI-SDRi + 直接 `fast_bss_eval` SDR，
   对同一批 N=50 官方权重输出与 `look2hear/metrics/core.py` 对拍：**最大差 4.8e-8 dB**；
   与 Phase-1 全量 `metrics_combined.csv` 对拍：**最大差 1.8e-7 dB**（要求 ≤0.05 dB）；
   `CSV avg/std == 逐句行重算` 精确成立；由 CSV 反推的 mixture baseline（`si-snr − si-snr_i`）
   与我独立从数据算的 baseline 差 **4.5e-8 dB**（SDR 基线差 8.9e-14 dB）。**T3 指标实现与 eval 脚本无问题。**
2. **评测协议自洽。** 官方混音脚本代数在数据上精确成立：`mix == s1+s2`（最大残差 3.05e-5 = 16-bit PCM 1 LSB，
   p99 相同；相对 RMS 1.9e-4）、`max(|mix|,|s1|,|s2|) == 0.9` 对 **3000/3000** 条精确成立（峰值归一化）；
   全部 clip = **32000 样本/16 kHz = 2.0 s**；`target.json` = s1 拷贝；mouths npz `(50,96,96) uint8`
   对 s1/s2 的 3000 个 id 全部存在，eval `.pt` = 3 通道堆叠，eval dataset 预处理（Grayscale → /255 →
   CenterCrop 88 → mean .421/std .165）与训练 val 管线一致；50 帧 ↔ 2 s@25 fps 帧数对齐。
   **baseline 分布**（全 3000）：SI-SDR(mix,s1) mean **4.994** dB、std 3.003、median 5.011、p5 0.361、p95 9.738；
   与 CSV 反推值精确一致。
3. **指标/协议变体无法解释 ~7 dB 差距。** 在 N=50 上逐项量化：target=s1 正确（对 s2 的 SI-SDRi 为 −17.0，
   Δ=+24.5 dB）；单源不需要 PIT（2-target oracle 上界 +0.000 dB）；输出/目标对齐 lag 恒 0（最佳位移增益 +0.000）；
   est 无削波（峰值中位 0.80，clipped fraction 0）；尺度不变（1e-8 dB）；zero-mean 口径 Δ−0.0005 dB；
   SDR filter 512→2048 Δ+0.257 dB；`change_sign=True` 恰为相反数；8 kHz 口径 ΔSI-SDR +0.111/ΔSDR +0.211 dB；
   2 s 整句协议（无整句/分段歧义，分段变体 ≤0.5 dB）。**合理协议/指标变体合计最多回收 ~0.4 dB。**
4. **7.03/7.71 的定性：协议上可信的“跨语料（cross-corpus）”结果，但不是论文 LRS2-2Mix 列的协议复现。**
   证据链：发布 `falcon.pt` 的训练 run 快照 `Experiments/checkpoint/Falcon/conf.yml` 的 datamodule 指向
   **`LRS3_complex_nonmoving`**（`conf.yml:99-102`；epoch 714 与 `best_k_models.json`/`PROVENANCE.md:37-38` 一致）
   ⇒ 官方权重是在 **LRS3 数据**上训练的；README 的 LRS2-2Mix 数字 14.1/14.7（`README.md:211`）标注为
   “under our training protocol”（`README.md:228`），即论文按 benchmark 训练/评测。把 LRS3 模型直接拿到
   LRS2-2Mix 上评测，7 dB 左右的跨语料退化与“协议/指标/对齐全部正确”并不矛盾。
   量化排除项：指标/协议 ≤0.4 dB；模型规模（big−small）仅 **+0.97/+0.91 dB**（P2-B，300 条配对）；
   窗口化协议 small −1.91(SI-SNRi)/−1.73(SDRi)、big −2.32/−2.13 dB（Phase-1 全量评测用的是整句协议，未用窗口化）；
   视觉分支正常（换错说话人 −23.3 dB、置零 −14.4 dB ⇒ mouth 确实生效且目标正确）。
   ⇒ **剩余 ~6 dB 主要来自训练数据/训练配方与论文 LRS2 列不匹配，而不是评测协议或指标实现。**
5. **要达到论文 14.1/14.7，缺的是“LRS2-2Mix 训练出的 checkpoint”，不是评测修复。**
   需要：P2-D 用 LRS2-2Mix tr（11 h）按论文配方训练（或向作者索取 LRS2 列对应 ckpt），再用本报告验证过的
   同一评测协议复测；可用 LRS3-2Mix 数据做交叉验证预测（发布权重应接近 16.4/16.7，但该数据仓库未提供，无法验证）。
   若无法训练/取得 ckpt，FINAL 应把 7.03/7.71 明确标注为“**LRS3-trained 官方权重在 LRS2-2Mix 上的跨语料泛化结果**”，
   不得当作论文 LRS2 列的复现。

## 2. 方法：独立实现与对拍路径

| 项 | 我的实现 | 被验证对象 |
|---|---|---|
| SI-SDR | 纯 NumPy：去均值 → 最小二乘投影 `α=<e,s>/‖s‖²` → `10log10(‖αs‖²/‖e−αs‖²)`（float64） | `core.si_sdr`（`core.py:152-156`，PIT 包装 `PIT_SI_SDR`，`:75`） |
| SI-SDRi | `SI-SDR(est,s) − SI-SDR(mix,s)`（mixture 与 target 同源同尺度） | `core.si_sdr_improvement`（`:158-166`） |
| SDR | 直接 `fast_bss_eval.sdr(ref[1,T], est[1,T], filter_length=512, change_sign=False)`（正 dB） | `core.sdr` = `−sdr_pit_loss(...).mean()`（`:168-178`）；另验证 `change_sign=True` 语义 |
| SDRi | `SDR(est) − SDR(mix)`，512-tap | `core.sdr_improvement`（`:180-186`） |
| 模型重跑 | 走真实 eval 路径：`audio_visual_test._load_models_from_ckpt` + `AVSpeechEvalDataset` + `_mouth_to_tensor` + `model(mix[None], mouth_emb)`（`audio_visual_test.py:279,351-362,372,417-420`） | Phase-1 全量 CSV（3000 条）前 50 条逐句值 |
| 数据 | 原始 `lrs2_rebuild` wav / mouths npz / `scaling.mat` / `create_mix` 脚本独立解析 | eval 集 `/root/tmp/lrs2_eval_tt_full` |

- N=50 子集 = CSV/dataset 前 50 条（顺序匹配 **50/50**）；子集均值 SI-SDRi **7.535**/SDRi **7.962**，
  与全量 7.029/7.713 的差异在抽样误差内（全量 std≈2.93，N=50 SE≈0.41 dB）。
- 验收阈值 ≤0.05 dB：实测 **4.8e-8 / 1.8e-7 dB**，通过。

## 3. 指标对拍与口径量化（N=50，官方权重）

### 3.1 对拍结果

| 对拍项 | 最大差 | 结论 |
|---|---|---|
| 我的 4 指标 vs `core.py`（SI-SDR/SI-SDRi/SDR/SDRi） | **4.82e-8 dB** | ✅ |
| 我的 4 指标 vs Phase-1 CSV 逐句值 | **1.84e-7 dB** | ✅ |
| baseline SI-SDR 公式：my(mix,s1) vs CSV(`si-snr − si-snr_i`) | **4.47e-8 dB** | ✅ |
| baseline SDR 公式：my(mix,s1) vs CSV(`sdr − sdr_i`) | **8.88e-14 dB** | ✅ |
| CSV avg/std vs 逐句行重算（全 3000） | 精确相等（<1e-9） | ✅ |
| 我的 SDR vs `core.sdr`（同一 est） | 2.96e-12 dB（上轮 run 记录） | ✅ |

### 3.2 指标口径变体（同一 est，N=50）

| 变体 | 均值（dB） | 相对标准口径 Δ |
|---|---|---|
| SI-SDR（zero-mean=True，标准） | 13.197 | — |
| SI-SDR（zero_mean=False） | 13.196 | **−0.0005** |
| SDR（filter_length=512，标准） | 13.723 | — |
| SDR（filter_length=2048） | 13.980 | **+0.257** |
| SDR（`change_sign=True`） | −13.723 | **恰为相反数**（max\|sum\|=0） |
| SI-SDR @8 kHz | 13.308 | **+0.111** |
| SDR @8 kHz（512-tap） | 13.935 | **+0.211** |

P2-A 在 200 条上独立测得：plain-SNR +0.34、4 段平均 +0.28、2 段平均 +0.15、SDR 256/128/64 = −0.09/−0.21/−0.35、
zero-mean −0.02、target 能量/RMS 缩放 ±0.00、best-of-s1/s2 +0.02 —— 与本报告同向且同量级（≤0.4 dB）。

## 4. 协议检查清单（每项给量化影响）

| # | 协议项 | 检查结果 | 量化影响 | 判定 |
|---|---|---|---|---|
| P1 | **s1/s2 混音代数** | 3000/3000：`mix == s1+s2` 至 16-bit PCM LSB（max 3.05e-5，p99 相同）；`max(|mix|,|s1|,|s2|)==0.9` 精确；官方脚本 `create_wav_2speakers.m:89-93,134,147`：8k `activlev` 归一化 → 权重 `10^(snr/20)` → 求和 → 峰值归一到 0.9；SNR 对恒为 (+s,−s)，**s1 恒为更响源**（snr1∈[0.00092,4.999]，均值 2.49 dB） | 0（protocol 定义，训练/评测同源） | ✅ 自洽 |
| P2 | **nominal SNR vs 实测能量比** | filename(`snr1−snr2`) vs 实测 `10log10(E_s1/E_s2)`（500 条）：mean\|Δ\| **0.597** dB、max 3.95、corr 0.960（源于 8k activlev 增益被套用到 16k 文件） | 仅影响“名义 SNR”解释，不影响 SI-SDRi（对同一 mix/target 计算） | ✅ 记录 |
| P3 | **target 约定** | `target.json` wav == rebuild `s1`（5/5 逐样本相等）；N=50：target=s1 SI-SDRi **7.535** vs target=s2 **−17.012**（Δ**+24.55**）；s2 更好的样本 0/50 | 选错 target 会毁掉指标；当前 s1 正确 | ✅ |
| P4 | **PIT** | 单源（n_src=1）→ 不需要 PIT；2-target oracle 上界 = 7.535（+**0.000**） | 0 | ✅ 不适用 |
| P5 | **mouth/音频对齐** | 帧数 50 ↔ 2 s@25 fps；npz `(50,96,96)` uint8、eval `.pt`=(50,96,96,3) 且==stack(npz)；dataset 输出 `(50,88,88)` z-score（与训练 val 管线一致）；**模型 mouth 偏移消融（N=20，前 20 条；基线 SI-SDRi 7.911 dB，与 P3/P4 的 N=50 基线 7.535 是不同子集）**：0 偏移 7.911 dB，±2 帧 −5.1/−6.4 dB，置零 −14.4，换 s2 说话人 −23.3；P2-A（N=200）±5/±12 帧 −10~−13 dB | 0 偏移为最优；若错位 1–2 帧将损失数 dB | ✅ 当前对齐正确（且 mouth 是刚需） |
| P6 | **baseline 分布** | 全 3000 SI-SDR(mix,s1)：mean **4.994**/std 3.003/median 5.011/p5 0.361/p95 9.738；SDR(512) 300 条抽样 mean 5.036（CSV 反推全量 5.1005） | 用于 SI-SDRi/SDRi 基线；与 CSV 精确一致 | ✅ |
| P7 | **整句 vs 分段** | 3000/3000 clip = 32000 样本 = 2.0 s，逐 clip 评测（官方 LRS2-2Mix 2 s 协议；数据卡：11 h/3 h/1.5 h 固定 split）；P2-A 分段变体（1 s 截断 −0.48、2 段 +0.15、4 段 +0.28） | ≤0.5 dB | ✅ 无歧义 |
| P8 | **缩放/削波** | SI-SDR 尺度不变（×0.1/×10 → 1e-8 dB）；est 峰值中位 0.80、无削波（Δ=0）；P2-A 峰值/RMS 归一化 ±0.00 | 0 | ✅ |
| P9 | **波形时间对齐** | est vs s1 互相关：最优 lag 恒 **0** 样本（50/50，\|lag\|>16 为 0），最佳位移增益 **+0.000** dB | 0 | ✅ |
| P10 | **eval 管线一致性** | 用官方 eval 路径重跑 N=50 → 与 Phase-1 CSV 差 1.8e-7 dB；CSV 行序=dataset 序；avg/std 精确 | 0 | ✅ |

## 5. 为什么是 7.03/7.71 而不是 14.1/14.7：归因与“还差什么”

### 5.1 证据链

1. **发布权重训练集 = LRS3 派生数据**：`Experiments/checkpoint/Falcon/conf.yml:99-102` 的 datamodule
   train/valid/test 均为 `${DATA_ROOT}/small/DataPreProcess/**LRS3_complex_nonmoving**`；`:105` `exp_name: Falcon`；
   `best_k_models.json` 最优 ckpt = `epoch=714`，与 `PROVENANCE.md:37-38` 的“falcon.pt 由 epoch 714 导出”一致。
2. **论文数字是“按 benchmark 训练协议”下的结果**：`README.md:228`：“Dolphin and AV-Mossformer2 were
   re-implemented as causal variants **under our training protocol**; the other baselines follow the values
   reported by Swift-Net.”；`README.md:211` 的 LRS2-2Mix 14.1/14.7 与 RealSSA/LRS3/VoxCeleb2 列并列，
   属同一“我们的训练协议”（即每个 benchmark 用自己的训练集训练/调参），而非一个跨语料模型。
3. **本地评测用的是 LRS3-trained 权重在 LRS2-2Mix 上的跨语料评测**：本报告 P1–P10 已排除指标/协议/对齐/
   target/PIT/缩放/分段等所有可量化因素（合计 ≤0.4 dB）；P2-B 独立测得 big 只 +0.97 dB、窗口化 −1.9 dB；
   P2-A 独立测得协议/指标变体 ≤0.4 dB、mouth 对齐敏感且当前已最优。⇒ 剩余 ~6 dB 指向训练数据/配方。

### 5.2 候选原因 × 量化界定

| 候选原因 | 实测/证据 | 对 7 dB 的解释力 | 判定 |
|---|---|---|---|
| 指标定义/实现错误 | 与独立实现对拍 ≤1.8e-7 dB；口径变体 ≤0.4 dB | ≤0.4 dB | ❌ 排除 |
| target 约定/PIT 错误 | s1 正确（Δ24.5 dB），PIT 0 | 0 | ❌ 排除 |
| mouth/音频错位 | 0 偏移最优；±2 帧 −5~−6 dB，±5/±12 帧 −10~−13 | 0（当前对齐正确） | ❌ 排除（但说明 mouth 很敏感） |
| 缩放/削波/8 kHz | 0 / 0 / ≤0.2 dB | ≤0.2 dB | ❌ 排除 |
| 整句 vs 分段 | 全部 2 s clip；分段变体 ≤0.5 dB | ≤0.5 dB | ❌ 排除 |
| 模型规模（small vs big） | big−small = +0.97/+0.91 dB（P2-B，300 条配对） | +1 dB | 🟡 次要 |
| 窗口化推理 | small −1.91/−1.73、big −2.32/−2.13 dB（P2-B） | Phase-1 全量用整句，未用窗口化 | ❌ 不适用 |
| 视觉分支未生效 | 置零 −14.4 dB、换说话人 −23.3 dB | 0（工作正常） | ❌ 排除 |
| **训练语料/配方与论文 LRS2 列不匹配** | 发布 ckpt 训练集=LRS3_complex_nonmoving（conf 快照）；论文按 benchmark 训练 | **~6 dB 主因** | ✅ 主因（[推断] 强证据） |
| 论文 LRS2 列的其他训练细节（负样本、数据混合、epoch、调参）未公开 | 仓库无 LRS2 训练配置/日志 | 未知残余 | ❓ 需 P2-D/作者 |
| LRS2-2Mix test 版本差异 | 数据卡：固定 1.5 h test split，3000 条 2 s clip；与本地一致 | 0 | ❌ 排除 |

### 5.3 要复现 14.1/14.7 还缺什么

1. **一个在 LRS2-2Mix tr 上训练的 checkpoint**（P2-D）：按论文训练协议（2 s、16 kHz、AV、NASS 对比损失、
   论文 batch/lr/epoch 等；见 `docs/REPRODUCTION_PLAN.md` Phase 4）训练，再用**本报告验证过的评测协议**
   （target=s1、整句 2 s、SI-SDR 标准定义、fast_bss_eval 512-tap SDR）复测。
2. 或向作者索取/确认 **paper LRS2-2Mix 列对应的 benchmark-specific checkpoint**（发布 `falcon.pt` 显然不是它）。
3. 交叉验证预测：若发布 `falcon.pt` 确为 LRS3 训练，则其在 **LRS3-2Mix** 上应接近 16.4/16.7；该数据仓库未提供
   （`docs/REPRODUCTION_PLAN.md` §9 R5），无法本地验证。
4. 数据侧无需再修：本报告 P1–P10 已证明评测数据/协议与实现自洽；剩余不确定性是**训练侧配方**。

## 6. 局限（诚实声明）

- N=50 只用于**指标/协议对拍**；全量数字（7.03/7.71）取自 Phase-1 3000 条 CSV（其 avg/std 已独立重算一致）。
- mouth 偏移/换说话人消融在 GPU 交接前以 N=20 完成（数值见 §4-P5）；`protocol_check.py --stage variants`
  可在单卡 GPU 上复跑（CPU 太慢）。A/V 包络相关探针（mouth 开合代理 vs 音频包络）结果弱/不可判
  （mean peak corr 0.21，lag 不稳定），本报告只把它当补充，不作为对齐结论；对齐结论以模型 mouth-offset 消融为准。
- 论文未公开（README: “Paper (TODO)”），无法直接核对论文的训练/评测细节；“发布 ckpt 为 LRS3 训练”由仓库内
  conf 快照推断（[推断] 强证据），非作者确认。
- RealSSA-Sim/Real、LRS3-2Mix、VoxCeleb2-2Mix 数据不可得，无法做跨 benchmark 交叉验证（RealSSA 申请待批，见 P2-E）。

## 7. 复跑命令

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
export PYTHONPATH=$PWD

# ① 纯 CPU（无需 GPU）：数据/混音/协议检查（全 3000 条）+ 缓存指标对拍 + 训练语料证据
CUDA_VISIBLE_DEVICES="" $ENV tools/repro/phase2/protocol_check.py --stage data      # ~1 min
CUDA_VISIBLE_DEVICES="" $ENV tools/repro/phase2/protocol_check.py --stage recheck   # 需先有 metrics 阶段的 /tmp/p2f_cache.npz
CUDA_VISIBLE_DEVICES="" $ENV tools/repro/phase2/protocol_check.py --stage static

# ② 单卡 GPU（小 batch，跑完即释放）：官方权重 N=50 重跑 + 全指标对拍 + mouth 消融
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/protocol_check.py --stage metrics --n 50 --device cuda:0
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/protocol_check.py --stage variants --n 50 --mouth-n 20 --device cuda:0

# ③ 一次跑完（CPU-only；data→metrics→recheck→static，慢，约 30+ min）
CUDA_VISIBLE_DEVICES="" $ENV tools/repro/phase2/protocol_check.py --stage all --device cpu

# 产物
#   docs/verification/phase2/P2F-results.json
#   docs/verification/phase2/P2F-protocol-check.log
```

## 8. 与其它 Phase-2 任务的关系（交叉验证）

| 任务 | 结论 | 与本报告关系 |
|---|---|---|
| P2-A（`loss-metric-eval`，指标/协议变体排查，200 条） | 指标/协议变体 ≤0.4 dB；mouth 对齐敏感且当前最优；数据自洽；剩余 gap 归因训练数据/配方 | **独立复现同向结论**（不同实现、不同子集：P2-A 200 条 vs 本报告 N=50 对拍 + 全 3000 数据检查） |
| P2-B（`model-core`，big vs small / 整句 vs 窗口） | big−small +0.97/+0.91 dB；窗口化 −1.9/−2.3 dB；视觉分支刚需 | 本报告引用其数字，用于排除“模型版本/窗口协议”作为 7 dB 主因 |
| P2-D（训练复现） | 待进行 | 本报告结论：**要复现 14.1/14.7 需 LRS2-trained ckpt（P2-D）** |
| P2-E（RealSSA/DNSMOS） | RealSSA gated 待批 | 本报告不涉及；RealSSA-Sim 将是另一个 benchmark-specific 训练/评测问题 |

## 9. 变更日志

| 日期/轮次 | 内容 |
|---|---|
| 2026-10-21 / P2-F | 初版（P3 勘误：§4-P5 的 mouth 消融基线由误写的 N=50 值 7.535 更正为实际 N=20 前 20 条基线 **7.911**；Δ 值不变）：独立 SI-SDR/SI-SDRi/SDR 实现与 core.py/Phase-1 CSV 对拍（≤1.8e-7 dB）；全 3000 条混音代数/峰值/长度/baseline/mouths 检查；N=50 target/PIT/对齐/缩放/削波/8k/指标口径量化；N=20 mouth 偏移/换说话人消融；发布权重训练语料证据（LRS3_complex_nonmoving）与 gap 归因；结论：协议可信但为跨语料结果，需 LRS2-trained ckpt 才能对齐 14.1/14.7 |
