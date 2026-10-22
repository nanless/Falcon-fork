# Phase-2 A 报告：LRS2-2Mix ~7 dB 差距排查（指标 / 协议变体）

> 分支 `repro/phase2`（HEAD `af74cca`，未 commit）· owner `loss-metric-eval`（task-18）· 2026-10-21
> 环境：conda `falcon`（Python 3.11.17 / torch 2.8.0+cu128 / fast-bss-eval 0.1.4；GPU 用 `CUDA_VISIBLE_DEVICES=1` 单卡批处理，已释放）
> 数据：`/root/tmp/lrs2_eval_tt_full/tt`（3000 条 LRS2-2Mix tt，target=s1，2 s/16 kHz，mouth `(50,96,96,3)` uint8 .pt）
> 权重：`/root/.cache/falcon_weights/falcon.pt`（官方 small，epoch 714；audio 525 + video 137 keys strict 加载）
> 产物：`tools/repro/phase2/gap_analysis.py`、`docs/verification/phase2/P2A-small-200.{log,json}`、
> `P2A-anchor-first20.{log,json}`；本报告。

## 1. 结论（TL;DR）

1. **指标/协议变体无法解释 7 dB 差距。** 在固定 200 条随机子集（seed 0）上，官方 small 权重
   **SI-SNRi 6.95 / SDRi 7.62 dB**（Phase-1 全量 3000 条为 7.03/7.71；子集有代表性）。
   所有“合理”的指标/协议变体相对基线的变化都在 **±0.35 dB 内**：plain-SNR +0.34、4 段平均 +0.28、
   2 段平均 +0.15、SDR 滤波长度 256/128/64 = −0.09/−0.21/−0.35、zero-mean −0.02、
   target 按 mix 能量/RMS 缩放 ±0.00、best-of-s1/s2（PIT 上界）+0.02。
   ⇒ 即使把口径全部换到最有利的变体，最多回收 **~0.3–0.4 dB**，剩余 **~6.6 dB** 仍无法解释。
2. **我们当前的评测协议在全部变体中已是最优/正确**：mouth 用 target(s1) 的裁片、88 中心裁剪、
   标准 mean/std 归一化、2 s 整句、精确帧对齐、mono 16 kHz——任何偏离都变差或不变：
   mouth 换 s2 为 −16.7 dB、置黑为 +2.16（相对 6.95 降 4.8）、96 不裁 −0.04、时间平移 ±5/±12 帧
   **−10~−13 dB**（灾难性）、单通道 vs 3ch 灰 ±0.00、不做归一化 −0.45；音频截 1 s −0.48、
   补零到 3 s −5.13、峰值/RMS 归一化 ±0.00。
3. **数据/混音/指标基线自洽**：`mix == s1+s2`（残差 3.05e-5 = PCM16 LSB，R²=1）、`target == s1`、
   mouth == s1 的 npz（逐元素相等）、mono/16k/32000 全成立；filename SIR 均值 5.01 dB，
   实测 mixture baseline SI-SNR(mix,s1) 均值 4.97 dB（corr 0.97），`SI-SNRi = SI-SNR(est) − SI-SNR(mix)`
   精确成立（Phase-1 全量 12.023 − 7.029 = 4.994 亦与 SIR 均值 4.974 吻合）。
4. **视觉分支是刚需且对齐敏感**：目标说话人 mouth 正确时 6.95 dB；换错说话人（s2 mouth）−16.7 dB；
   置黑（raw gray=0，预处理后恒值 −2.55）仍 +2.16 dB；时间轴错位 ±5 帧（0.2 s）即掉到 −4.5 dB。
   ⇒ 我们使用的 s1 mouth + 0 偏移已是“最佳对齐”，不是 gap 来源。
5. **剩余 gap 的归因（与 P2-B 一致）**：模型版本轴最多 +1 dB（P2-B：big whole 300 条 7.887/8.403，
   比 small +0.971/+0.911）；窗口化协议反而 −1.9 dB。因此 ~6 dB 差距更可能来自
   **发布 `falcon.pt` 的训练数据/配方与论文 LRS2 列不匹配**（论文可能按 benchmark 分别训练；发布权重
   训练集未知），而非评测协议。需 P2-D 训练复现 / 向作者确认 benchmark-specific checkpoint。
6. **GPU 已释放**：本任务所有 GPU 实验已结束（P2-A 无残留进程/显存），未启动新的全量长评测。

## 2. 方法

### 2.1 数据与子集

- 评测集 `/root/tmp/lrs2_eval_tt_full/tt`（Phase-1 构建）：3000 条，JSON 顺序即 Phase-1 `--limit` 顺序。
- 固定子集：`--subset random --seed 0 --limit 200`（对 3000 条无放回随机、排序后取 200）；
  子集 filename SIR 均值 5.010 dB vs 全量 3000 条 4.974 dB，代表性好。
- anchor 子集：`--subset first --limit 20` 用于复现 Phase-1 20 条基线（7.911/8.547）。

### 2.2 模型与推理

- 官方 `falcon.pt`：`FalconSpeakerContrastive`（audio 537,697）+ `FRCNNVideoModel`（11,186,688），
  strict 加载 `audio_model.*`(525 keys)/`video_model.*`(137 keys)；`eval()`。
- 推理：`est = audio_model(mix[None], video_model(mouth[None,None]))`，整句 32000 采样一次前向；
  batch=4（`audio_pad_3s` 因显存限制 batch=2），`torch.inference_mode`。
- mouth 预处理（复刻 `look2hear/datas` 验证管线）：`.pt [50,96,96,3] uint8 → Grayscale → [50,96,96] →
  /255 → CenterCrop(88) → (x−0.421)/0.165 → [1,1,50,88,88]`。

### 2.3 变体矩阵（全部在 200 条同一子集上）

| 类别 | 变体 | 实现 |
|---|---|---|
| 指标 | SI-SDRi（zero-mean on/off） | 标准投影 SI-SDR；基线=同一变体下 mix |
| 指标 | plain SNR improvement | 不做投影的尺度相关 SNR（`10log10(‖s‖²/‖e−s‖²)`） |
| 指标 | SDRi 滤波长度 | `fast_bss_eval.sdr_pit_loss(..., filter_length=512/256/128/64)` |
| 指标 | 整句 vs 分段 | 1 s×2 段 / 0.5 s×4 段，段内 SI-SDRi 再平均 |
| 指标 | target 缩放 | `s1×‖mix‖/‖s1‖`（能量）/ `s1×rms(mix)/rms(s1)`（RMS） |
| 指标 | target=s2 / best-of-s1,s2 | 同一 est 对 s2 的 SI-SDRi；PIT（2 目标）上界 |
| mouth | s1（baseline）/ s2 / 置黑（raw 0） | 分别取 target/干扰说话人 npz；置黑=全 0 灰度 |
| mouth | 96 不裁 / 88 裁 | CenterCrop 88 vs 直接用 96 |
| mouth | 时间平移 | 帧轴 roll ±5 / ±12 帧（0.2/0.48 s） |
| mouth | 3ch 灰 vs 单通道 | Grayscale(3ch 复制) vs 直接单通道（应逐位相同） |
| mouth | 归一化 | 标准 mean/std vs 只 /255（不做均值方差归一化） |
| 音频 | 2 s 整句 / 截 1 s / 补零 3 s | 截断时 mouth 同步取前 25 帧；补零 3 s 用 50 帧，指标取前 2 s |
| 音频 | 峰值归一化 / RMS 归一化 | `mix×0.9/max|mix|`、`mix/std(mix)` |

### 2.4 复跑命令

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
# anchor（20 条，复现 Phase-1 7.911/8.547）
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/gap_analysis.py \
  --limit 20 --subset first --models small --variants baseline --device cuda:0 \
  --batch_size 4 --out_json docs/verification/phase2/P2A-anchor-first20.json
# 主实验（200 条随机，14 变体）
CUDA_VISIBLE_DEVICES=1 $ENV tools/repro/phase2/gap_analysis.py \
  --limit 200 --subset random --seed 0 --models small --variants all --device cuda:0 \
  --batch_size 4 --out_json docs/verification/phase2/P2A-small-200.json
# 仅列出变体： --list-variants
```

## 3. 结果

### 3.1 数据自检（200 条；文件等价性检查取前 20 条）

| 检查 | 结果 |
|---|---|
| `mix − (s1+s2)` 最大绝对残差 | **3.05176e-05**（=1 LSB of PCM16；相对 3.4e-5） |
| `mix ≈ α(s1+s2)` 的 R² | **1.000000** |
| `target` wav vs rebuild `s1` | 逐样本相等（max diff 0） |
| eval `mix` vs rebuild `mix` | 逐样本相等（max diff 0） |
| eval mouth `.pt` vs `mouths/<s1_id>.npz`（3ch 复制） | **逐元素相等**（max diff 0） |
| 采样率 / 通道 / 长度 | 16000 Hz / mono / 32000（全部 200 条） |
| filename SIR vs 实测 SI-SNR(mix,s1) | 5.010 vs **4.970 dB**，corr **0.970**（基线 ≈ SIR，符合定义） |
| 实测 SDR512(mix,s1) | 5.075 dB |

### 3.2 指标变体（baseline 预测，200 条，单位 dB）

| 指标变体 | mean ± std | Δ vs baseline |
|---|---|---|
| **SI-SNRi（baseline：zero-mean SI-SDR，512-tap SDR）** | **6.955 ± 3.134** | — |
| SI-SNRi, zero_mean=False | 6.939 ± 3.141 | −0.016 |
| plain-SNR improvement（尺度相关） | 7.295 ± 2.872 | **+0.340** |
| SDRi, filter_length=512（baseline） | 7.622 ± 2.800 | — |
| SDRi, filter_length=256 | 7.537 ± 2.805 | −0.085 |
| SDRi, filter_length=128 | 7.410 ± 2.816 | −0.212 |
| SDRi, filter_length=64 | 7.270 ± 2.855 | −0.352 |
| SI-SNRi, 2×1 s 段平均 | 7.101 ± 3.081 | +0.146 |
| SI-SNRi, 4×0.5 s 段平均 | 7.237 ± 3.028 | +0.282 |
| target 能量缩放 s1 | 6.955 ± 3.134 | ±0.000 |
| target RMS 缩放 s1 | 6.955 ± 3.134 | ±0.000 |
| target=s2（错误目标） | −16.998 ± 5.113 | −23.95 |
| best-of-s1/s2（PIT 上界） | 6.976 ± 3.013 | +0.021 |

> 结论：**没有任何指标变体把 6.95 提升超过 0.35 dB**；SI-SDR 对 target 缩放严格不变（数学上可证），
> 分段平均只带来 +0.15~0.28 dB；把 SDR 滤波变短反而降低 SDRi。因此“指标定义不同”不是 7 dB 差距主因。

### 3.3 mouth 输入变体（200 条，单位 dB）

| mouth 变体 | SI-SNR | SI-SNRi | ΔSI-SNRi | SDR | SDRi | ΔSDRi |
|---|---|---|---|---|---|---|
| **s1（正确 target，88 crop，标准归一化）** | **11.92** | **6.95** | **0** | **12.70** | **7.62** | **0** |
| s2（错误说话人） | −11.74 | −16.71 | −23.67 | −8.08 | −13.15 | −20.77 |
| 置黑（raw gray=0，预处理后恒值） | 7.13 | 2.16 | −4.80 | 8.33 | 3.26 | −4.37 |
| 96 不裁（直接 96×96） | 11.88 | 6.91 | −0.04 | 12.66 | 7.58 | −0.04 |
| 时间 roll +5 帧（+0.2 s） | 0.42 | −4.55 | −11.50 | 3.06 | −2.02 | −9.64 |
| 时间 roll −5 帧（−0.2 s） | −0.91 | −5.88 | −12.83 | 1.79 | −3.28 | −10.90 |
| 时间 roll +12 帧（+0.48 s） | 0.63 | −4.34 | −11.30 | 2.94 | −2.13 | −9.76 |
| 时间 roll −12 帧（−0.48 s） | 1.52 | −3.45 | −10.41 | 3.71 | −1.36 | −8.98 |
| 单通道直传（不 stack 3ch 再 Grayscale） | 11.92 | 6.95 | −0.00 | 12.70 | 7.62 | −0.00 |
| 不做 mean/std 归一化（只 /255） | 11.47 | 6.50 | −0.45 | 12.23 | 7.16 | −0.47 |

> 关键：**视觉身份必须正确**（s2 mouth 崩 23.7 dB）；**时间对齐必须精确**（±0.2 s 即掉 11~13 dB），
> 我们的 0 偏移已是全变体最优 ⇒ 嘴部对齐/裁剪/通道约定不是 gap 来源。置黑只掉 4.8 dB（模型仍能用音频
> 做部分提取），说明视觉贡献约 4.8 dB、模型本身音频能力有限。

### 3.4 音频输入变体（200 条，单位 dB）

| 音频变体 | SI-SNR | SI-SNRi | ΔSI-SNRi | SDR | SDRi | ΔSDRi |
|---|---|---|---|---|---|---|
| **2 s 整句（baseline）** | **11.92** | **6.95** | **0** | **12.70** | **7.62** | **0** |
| 截断 1 s（mouth 取前 25 帧） | 11.41 | 6.48 | −0.48 | 12.65 | 7.51 | −0.12 |
| 补零到 3 s（mouth 50 帧，指标取前 2 s） | −0.16 | −5.13 | −12.08 | 2.39 | −2.68 | −10.30 |
| 峰值归一化（×0.9/max） | 11.92 | 6.95 | −0.00 | 12.70 | 7.62 | −0.00 |
| RMS 归一化（/std） | 11.92 | 6.95 | −0.00 | 12.70 | 7.62 | −0.00 |

> 2 s 整句是训练/评测的正确约定；补零到 3 s 会让模型崩（−12 dB），说明“长度/补零约定”如果不同只会更差。
> 全局幅度缩放不影响输出质量（模型近似尺度等变，SI-SDR 亦尺度不变）。

### 3.5 mixture baseline 自洽性与 anchor

- 200 条：mean SI-SNR(est) 11.92，mean SI-SNR(mix,s1) 4.97，`i` = 6.95 = 11.92 − 4.97（精确到浮点误差）。
- Phase-1 全量 3000 条：SI-SNR 12.023 / SI-SNRi 7.029 ⇒ baseline = 4.994，与 3000 条 filename SIR 均值
  4.974 相差 0.02 dB ⇒ 全量评测的 baseline 与数据定义一致。
- anchor（first-20，small baseline）：**SI-SNRi 7.911 / SDRi 8.546**，与 Phase-1 记录的 7.911/8.547 完全一致
  （批处理 float32 推理差 <0.001 dB）⇒ 本脚本的模型/预处理/指标管线与 Phase-1 等价。

### 3.6 与 P2-B（model-core）的交叉核对

- 同 300 条子集：P2-B small whole **6.916/7.492**；本报告 200 随机子集 **6.955/7.622**（子集不同，量级一致）。
- P2-B big whole **7.887/8.403**（比 small +0.97/+0.91 dB）；窗口化 −1.9/−2.3 dB；
  ⇒ 模型版本/推理协议合计最多 +1 dB 或更差，不能解释 7 dB。
- 视觉消融：P2-B 的 `zero` 是**模型输入全 0**（post-normalization），本报告 `置黑` 是 raw gray=0
  （预处理后恒值 −2.55）；两者语义不同，故绝对值不同（−8.17 vs +2.16 dB），但结论一致：
  **视觉分支被重度使用、不可缺失**。两报告应并列引用并注明 zero 的定义。
- 两报告均指向同一结论：**剩余 ~6 dB 不能由“评测协议/指标/视觉分支/模型版本”解释**。

## 4. 分析：为什么 7 dB 不是协议问题

1. **可加性估计**：paper LRS2-2Mix SI-SNRi 14.1；本评测 7.03。若 paper 的 baseline 与我们相同（~5 dB，
   由 LRS2-2Mix 的 SNR 构造决定：s1 SNR∈[0,5] dB、s2 SNR=−s1 ⇒ SIR∈[0,10]、均值 ~5 dB），
   则 paper 的 est SI-SNR ≈ 19.1 dB，而我们是 12.0 dB → **需要 +7 dB 的估计质量**。
2. 我们穷举的协议变体最大收益 +0.34 dB（plain SNR），且多数变体变差；数据定义自洽、target/通道/对齐正确；
   视觉分支正常且对齐敏感。因此“协议/指标”不成立。
3. **与模型规模无关**：big 只 +1 dB（P2-B）。**与窗口化无关**：窗口化更低。
4. 最可能解释：**发布 `falcon.pt` 不是论文 LRS2-2Mix 列所用的 checkpoint**（论文很可能按 benchmark 分别
   训练；README/配置快照未给出训练数据组合；`PROVENANCE.md` 只说明导出 epoch 714）。另一个次级假设：
   论文测试集口径不同（不同混合列表/说话人对/对齐），但本报告证明**我们的数据构造与指标实现是自洽的**，
   无法用“实现错误”解释。
5. 若 paper 的 14.1 来自 **benchmark-specific checkpoint**，则复现路径是 P2-D（用官方配方在 LRS2-2Mix
   训练）或向作者索取对应权重；若来自 RealSSA-Sim 训练后 zero-shot 迁移，则需等 RealSSA-Sim 审批后
   先在 RealSSA-Sim 上复现 7.7/8.2（论文主结果），再判断跨数据集泛化差距。

## 5. 局限与风险

- 只用 **small** 官方权重做全变体矩阵（big 由 P2-B 负责，避免重复/GPU 争用）；结论“协议不能解释 7 dB”
  对 big 同样成立（P2-B 数据）。
- 子集 200 条（seed 0）而非全量 3000；子集与全量基线差 0.08 dB（6.95 vs 7.03），不影响结论。
- SDR 只测了 filter_length 64–512；更大 filter（1024/2048）未测（趋势是更大可能略高，但数量级不可能 +7 dB）。
- `fast_bss_eval.sdr_pit_loss` 的 `filter_length=None`/`0` 语义未采用（`None` 结果异常、`0` 报错），
  未作为正式变体。
- mouth 变体是**诊断性**的，不代表论文真实协议；置黑/时间平移用于量化视觉分支与对齐敏感性。
- `change_sign`：`fast_bss_eval.sdr_pit_loss` 返回负 SDR loss，本报告统一取负为正值指标；单独取反只会整体变号、不改变结论，故未作为独立变体。
- target 缩放的 **SDR** 变体未单列：`fast_bss_eval` 内部对信号归一化，预期同样尺度不变；本报告以 SI-SDR 变体验证（±0.00）。
- 真实“论文协议”仍未知（论文未给评测脚本）；本报告只能证明“我们的协议是最优且自洽的”，不能证明
  论文用了别的协议。

## 6. 下一步建议（交接）

1. **P2-D**：用 LRS2-2Mix 训练复现（configs/LRS2_2Mix.yml）→ 若训练后能到 14 dB，则坐实“发布权重不是
   LRS2 checkpoint”；这是最直接的验证。
2. **联系作者**：确认 `falcon.pt` 的训练数据组合、论文 LRS2-2Mix 列所用 checkpoint 与评测脚本/指标实现。
3. **P2-E**：RealSSA-Sim 审批后先复现论文主结果 7.7/8.2（同分布，若也差很多则是权重/评测实现问题；
   若接近则支持“发布权重为 RealSSA 训练、LRS2 为跨域”假设）。
4. **P2-F**：studio-backend 独立协议验证应复现本报告的“协议变体 ≤0.35 dB”结论；若出现分歧，以双方
   脚本对同一 200 条子集的逐句输出 diff 定位。

## 7. 复跑与证据清单

```bash
ENV=/root/miniforge3/envs/falcon/bin/python
cd /root/code/github_repos/Falcon-fork

# 结果 JSON/日志
#   docs/verification/phase2/P2A-small-200.json   （meta/data_checks/models[small].variants[变体][指标]{mean,std,n}）
#   docs/verification/phase2/P2A-small-200.log    （本文所有表的原始 stdout）
#   docs/verification/phase2/P2A-anchor-first20.{json,log}
# 复跑：见 §2.4（anchor 约 30 s；主实验 200 条×14 变体约 5 min，GPU1 batch=4）
```

- 本任务只写 `tools/repro/phase2/` 与 `docs/verification/phase2/`；未修改 Phase-1 源码，未 commit。
- GPU：实验已结束，无残留进程/显存（`nvidia-smi` 无本任务进程），已按要求释放。
- 备注：并发执行时 GPU/CPU 争用会显著变慢（本任务最初在 GPU0 与 P2-B 争用，已中断后改 GPU1 批处理完成）。
