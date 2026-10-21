# 04 损失函数 / 指标 / 视觉编码器 / 评测入口

> 负责人：`loss-metric-eval`（R1 初稿；**R3 已按 data-pipe 的 R2 评审逐条修订，响应见 §9.1**）
> 代码基线：`main` commit `7576854`（`Add the project page ...`）
> 证据标记：`[代码]` 直接读源码；`[推断]` 由代码/文档合理推断；`[验证]` 实际运行确认（脚本与日志见第 8 节）。
> 本文件只覆盖分工范围：`look2hear/losses/**`、`look2hear/metrics/**`、`look2hear/videomodels/**`、
> `audio_visual_test.py`、`evaluated_mac_params.py`；与训练系统/模型核心相交的结论只写到“接口契约”深度。

---

## 1. 子系统职责与总体位置

**一句话结论**：本域负责 Falcon 训练目标（负 SDR 系 PIT 损失 + NASS 说话人对比损失）、评测指标与
结果 CSV、冻结的唇读视觉编码器（FRCNN，11.19 M 参数）以及离线评测/算力统计两个入口脚本。

### 1.1 四个部件与上下游

| 部件 | 文件 | 职责 | 上游使用者 | 下游依赖 |
|---|---|---|---|---|
| 损失函数 | `look2hear/losses/matrix.py`、`pit_wrapper.py`、`mixit.py`、`nass_contrastive_loss.py`、`sisnri.py` | 负 SI-SDR/SNR/SD-SDR、PIT 归约、MixIT、NASS 对比损失 | `audio_train.py:91-99` 构建；`look2hear/system/av_litmodule*.py` 调用 | 纯 torch（`pit_wrapper` 另用 scipy 匈牙利算法 `pit_wrapper.py:4`） |
| 指标 | `look2hear/metrics/wrapper.py`、`allwrapper.py`、`splitwrapper.py` | 逐句 SI-SDR/SDR（以及 SNR/PESQ/STOI）计算、CSV 写入、平均值/标准差 | 只有 `ALLMetricsTracker` 被 `audio_visual_test.py:47,208` 使用；另外两个是死代码 | `torch_mir_eval`、`fast_bss_eval`、`pypesq`、`pystoi`（`requirements.txt:25-29`） |
| 视觉编码器 | `look2hear/videomodels/**`（8 个文件） | 冻结的 lip-reading FRCNN：每帧 96×96 灰度嘴部 → 512 维特征 | `audio_train.py:61-63`、`audio_visual_test.py:150-152`、`evaluated_mac_params.py:51-53` | 纯 torch；预训练权重 `pretrain_zoo/frcnn_128_512.backbone.pth.tar`（仓库未提供，README:107,158） |
| 评测/算力入口 | `audio_visual_test.py`、`evaluated_mac_params.py` | 离线测试集 SI-SDR/SDR 评测（单/多卡、HF 权重）；参数量与 MACs 统计 | 用户 CLI（README:163-171） | 上述三者 + `look2hear/datas` + `look2hear/utils` + `pytorch_lightning`/`ptflops`/`pandas` |

### 1.2 关键数字（本域可复算，均 `[验证]`）

| 数字 | 源码/README 出处 | 本次复算 | 结论 |
|---|---|---|---|
| 音频网络可训练参数 0.5 M | README:34,103 | `FalconSpeakerContrastive`（`configs/Falcon.yml` 配置）= **537,697** | 0.538 M，与论文“0.5 M”口径一致 |
| 视觉编码器参数 11.2 M | README:103 | `FRCNNVideoModel(backbone_type=resnet, relu_type=prelu, width_mult=1.0)` = **11,186,688** | 11.187 M |
| 总参数 11.72 M | README:100 | 537,697 + 11,186,688 = **11,724,385** | 11.72 M，精确吻合 |
| Falcon-Big 总参数 12.35 M | README:101 | 1,158,617（Big audio）+ 11,186,688 = **12,345,305** | 12.35 M，精确吻合（R3 真实权重 [验证]） |
| 发布权重顶层结构 | `studio/falcon_model/PROVENANCE.md:36-40` | `falcon.pt` = `{architecture:'FalconSpeakerContrastive', epoch:714, state_dict}`（525 audio + 137 video keys）；`falcon-big.pt` = `{architecture:'SpeakerContrastiveV2', epoch:715, state_dict}`（517+137 keys） | 与本地/Studio 模型 key+shape 完全一致、strict=True 可加载（R3 [验证]，`runs/r3-weights.log`） |
| MACs 10.6 G | README:206 | `ptflops` 复刻 `evaluated_mac_params.py` 的调用 = **10.611 G**（只含音频网络） | 与 README 一位小数一致 |
| 视觉编码器 MACs（未计入 10.6 G） | 无 | 50 帧：64×64 → 7.77 G；**88×88 → 15.90 G**；96×96 → 17.48 G | 10.6 G 不含视觉编码器（见 §2.5） |

### 1.3 与其它域的边界

- **模型核心（01）**：`Falcon`/`FalconSpeakerContrastive`、`Separator`、`VideoBlock`、融合层。本域只回答
  “视觉特征以什么形状/帧率进入模型”以及“损失/指标在训练/评测中的用法”。
- **训练系统（03）**：Lightning module、optimizer/scheduler、DDP、`conf.yml` 生成。本域在
  §2.5/§7 指出它与本域相关的两个契约问题（config 被 `dict.pop` 污染、system 中 NASS 的 β 调度）。
- **数据管线（05）**：`avspeech_dataset*.py` 的音频/嘴部读取、96→88 裁剪发生在数据集预处理而非 FRCNN
  内（见 §2.3.2/§4.5）。本域在 §7 向 data-pipe 提 3 个待确认问题（单声道音频、88 裁剪一致性、near 负样本）。
- **Studio 后端（06）**：`studio/falcon_model/` 是推理副本；`studio/falcon_runtime.py:95-104` 复刻了
  同样的 96→88 裁剪与归一化（本域用作交叉证据）。

---

## 2. 文件清单与逐文件解析

### 2.1 `look2hear/losses/`

#### 2.1.1 `__init__.py`（44 行）

导出 9 个负 SDR 函数/类、`PITLossWrapper`、`new_PITLossWrapper`、`MixITLossWrapper`、`SISNRi`、
`NASSContrastiveLoss` 以及若干 multistage/freq 损失（`losses/__init__.py:1-21`）。

- `[代码]` `NASSContrastiveLoss` 在第 21 行被 import，但**没有**写进 `__all__`（第 23-44 行）；
  训练侧是直接 `from ..losses.nass_contrastive_loss import NASSContrastiveLoss`
  （`system/av_litmodule_nass_v2.py:17`），所以不影响运行，但 `from look2hear.losses import *` 拿不到它。
- `[代码]` `pairwise_neg_sisdr` 等 9 个名字是**模块级已实例化对象**（`matrix.py:371-380`），
  不是类；`audio_train.py:92-98` 用 `getattr(look2hear.losses, name)` 取的正是这些实例。

#### 2.1.2 `matrix.py`（383 行）——负 SDR 的数学与代码逐行对应

**A. `PairwiseNegSDR`（`matrix.py:188-232`，本域最核心）**

构造：`PairwiseNegSDR(sdr_type, zero_mean=True, take_log=True, EPS=1e-8)`，`sdr_type ∈ {"snr","sisdr","sdsdr"}`
（`matrix.py:189-195`）。输入必须 `[batch, n_src, time]` 且 est/target 同形状（`matrix.py:198-202`），
输出 **`[batch, n_src(est), n_src(target)]` 的负 SDR 矩阵**。

数学（`e`=估计、`s`=目标，均按需去均值后）：

```
零均值（zero_mean=True，matrix.py:204-208）：
    s ← s − mean_t(s);        e ← e − mean_t(e)

外积/广播（matrix.py:210-211）：
    S = s[None, :]        # [B,1,n_src,T]  —— 作为 target 维
    E = e[:, None]        # [B,n_src,1,T]  —— 作为 estimate 维

sisdr/sdsdr（matrix.py:212-218）：
    D[b,i,j]   = Σ_t E[b,i,t]·S[b,j,t]                       (pair_wise_dot)
    En[b,j]    = Σ_t S[b,j,t]² + EPS                         (s_target_energy)
    P[b,i,j,t] = D[b,i,j]·S[b,j,t] / En[b,j]                 (pair_wise_proj = α·s)

snr（matrix.py:219-221）：
    P[b,i,j,t] = S[b,j,t]                                    (不做投影，直接广播 target)

噪声项（matrix.py:222-225）：
    sisdr            : e_noise = E − P      （P = α·s，即 SI-SDR 的 e_noise）
    sdsdr / snr      : e_noise = E − S      （与原始 target 之差）

SDR 与损失（matrix.py:226-232）：
    sdr[b,i,j] = Σ_t P² / (Σ_t e_noise² + EPS)
    loss       = −10·log10(sdr + EPS)        （take_log=True）
```

- `[代码]` `EPS` 出现在三处：`s_target_energy + EPS`（分母）、`Σ e_noise² + EPS`、`log10(... + EPS)`。
- `[代码]` `zero_mean=False` 时跳过第 204-208 行，直流分量会进入 SDR（这不常见，但 `metrics/wrapper.py:30`
  的 `MetricsTracker` 就是这么用的，见 §2.2.1）。
- `[代码]` `take_log=False` 时返回线性域负 SDR（默认 True；仓库内无调用者传 False）。
- `[验证]` 三种 `sdr_type` 与手算公式逐元素对比 `max|diff| = 0`（`runs/04-losses.log` §[1]）；
  `zero_mean` 对直流偏置的不变性/敏感性见同日志 §[2]。
- `[验证]` **SNR 与 SDR 的关键差别**：`est → 2·est` 时 `snr` 变化 −4.015 dB，而 `sisdr` 变化 0
  （尺度不变）——这直接证明 `metrics/allwrapper.py` 里名为 “sdr” 的列其实是 **SNR**（见 §2.2.2）。

**B. `SingleSrcNegSDR`（`matrix.py:235-281`）**

- `[代码]` 输入 `[batch, time]`（`matrix.py:248-252`），输出 `[batch]`；`reduction ∈ {"none","mean"}`，
  `reduction="sum"` 被 `assert` 拒绝（`matrix.py:239`）。
- 公式与 `PairwiseNegSDR` 单对完全相同：`α = Σ e·s /(Σ s²+EPS)`，`loss = −10log10(Σ(αs)²/(Σ(e−αs)²+EPS))`
  （sisdr），snr 时 `α=1`（`matrix.py:260-279`）。
- `[代码]` **小 bug**：构造参数 `EPS` 被忽略，`self.EPS = 1e-8` 写死（`matrix.py:246`）。
- `[验证]` `reduction="none"` 与手算 `max|diff|=0`；`"mean"` 等于 `none` 的均值；`"sum"` 抛
  `AssertionError(NotImplementedError)`（`runs/04-losses.log` §[4]）。

**C. `MultiSrcNegSDR`（`matrix.py:284-326`）**

- `[代码]` 输入 `[B,n_src,T]`，对**同一索引**的 est/target 逐源算 SDR 后取源平均，
  **不做任何排列匹配**（`matrix.py:306-326`）：`loss[b] = −(1/n_src)Σ_i SDR(e_i,s_i)`。
- `[代码]` 同样有 `self.EPS = 1e-8` 忽略入参的问题（`matrix.py:292`）。
- `[验证]` 与手算 `max|diff|=0`（`runs/04-losses.log` §[5]）。它是 `perm_avg`/`pw_mtx_broadcast` 的
  预期损失类型（§2.1.3），但 Falcon 配置不使用。

**D. 未被任何训练/评测路径使用的多阶段/频域损失（死代码，但对理解仓库历史有用）**

| 类/对象 | 行号 | 说明 | 状态 |
|---|---|---|---|
| `STFTBase` / `STFT` | `matrix.py:13-103` | 用 conv1d 实现的 (i)STFT 层，`K = rfft(eye(N))` 预计算核 | 只被 `PIT_SISNR_mag_loss` 引用 |
| `PIT_SISNR_mag_loss` | `matrix.py:105-130` | 频域 SI-SNR；`mel_opt=True` 时调用 `self.mel_fb`，但从未定义 → AttributeError | 死代码 + 潜在 bug |
| `PIT_SISNR_time_loss` | `matrix.py:133-149` | 时域 SI-SNR，`clamp(min=-30)`（`matrix.py:147`） | 只被 MultistageLoss 用 |
| `MultistageLoss` | `matrix.py:152-171` | 多阶段加权：`alpha=0.4·0.8^(1+(epoch-81)//5) if epoch>80 else 0.4`（`matrix.py:161`） | 未被 config/entry 引用 |
| `MultistageLoss_val` | `matrix.py:173-186` | 计算了 mag 损失却只返回 `cur_loss_s.mean()`（`matrix.py:186`），β 加权丢失 | 死代码 + 逻辑 bug |
| `freq_MAE_WavL1Loss` / `freq_MSE_Loss` | `matrix.py:328-369` | STFT 频域 L1/MSE（win=2048/640, hop=512/160） | 仅作为 `pairwise_neg_sisdr_freq_mse` 元组的第 2 项 |
| `multisrc_*`、`freq_mae_wavl1loss`、`pairwise_neg_*_multidecoder` | `matrix.py:372-383` | 别名/元组 | 训练配置未用 |

**E. 别名（`matrix.py:371-383`）**

`[代码]` 9 个函数名是 **全局单例实例**：`pairwise_neg_sisdr = PairwiseNegSDR("sisdr")` 等
（`matrix.py:372-380`）；`pairwise_neg_sisdr_freq_mse = (PairwiseNegSDR("sisdr"), freq_MSE_Loss())`、
`pairwise_neg_snr_multidecoder = (PairwiseNegSDR("snr"), MultiSrcNegSDR("snr"))`（`matrix.py:382-383`）
是给 `pit_from="pw_mtx_broadcast"/"pw_mtx_multidecoder_*"` 用的二元组。

#### 2.1.3 `pit_wrapper.py`（325 行）——PIT 归约的完整语义

文件里有两个 wrapper：`new_PITLossWrapper`（`pit_wrapper.py:6-150`，实验性、未被使用）与
`PITLossWrapper`（`pit_wrapper.py:152-312`，训练实际使用）。

**A. `PITLossWrapper` 的 `pit_from` 六种模式**（允许值见 `pit_wrapper.py:162`）

| `pit_from` | 代码行为 | 对 `loss_func` 的契约 | 是否被 Falcon 配置使用 |
|---|---|---|---|
| `pw_mtx`（默认） | `pw_loss = loss_func(ests, targets)` → `[B,n_src,n_src]`，再找最佳排列、按行取均值 | pairwise 矩阵损失（`PairwiseNegSDR`） | ✅ train 与 val（`configs/Falcon.yml:116-126`） |
| `pw_pt` | 双重循环逐对调用 `loss_func(est_src, target_src)` 组成矩阵（`pit_wrapper.py:182-183,239-247`） | 单源损失（`SingleSrcNegSDR`），输入 `[B,T]` | ❌ |
| `perm_avg` | 对每个排列调用 `loss_func(ests[:,perm], targets)`，取 min 后对 batch 求均值（`pit_wrapper.py:184-193,249-258`） | 已做源维归约的损失（`MultiSrcNegSDR`） | ❌ |
| `pw_mtx_broadcast` | 先用元组第 0 项算矩阵并 PIT 重排，最后再 `mean_loss += 0.5·loss_func[1](reordered, targets).mean()`（`pit_wrapper.py:172-173,233-234`） | `(PairwiseNegSDR, MultiSrcNegSDR)` | ❌（对象别名存在但 config 不用） |
| `pw_mtx_multidecoder_keepmtx` | 用 ests 列表**最后一个 block** 算矩阵选排列；再对每个 block 重排后加第 2 项损失；`equidistant_weight` 只在此分支生效（`pit_wrapper.py:174-176,208-225`） | `(PairwiseNegSDR, MultiSrcNegSDR)`，`ests` 为 list | ❌ |
| `pw_mtx_multidecoder_batchmin` | 把多 block 在 batch 维拼接、targets 重复，再做普通 PIT（`pit_wrapper.py:177-181`） | `PairwiseNegSDR` | ❌ |

**B. 前向主干（`pit_wrapper.py:168-237`）逐行**

1. `[代码]` 按 `pit_from` 得到 `pw_loss`；`pw_mtx` 分支只有一行 `self.loss_func(ests, targets)`
   （`pit_wrapper.py:170-171`）。
2. `[代码]` 断言 `pw_loss.ndim == 3` 且 batch 维与 targets 相同（`pit_wrapper.py:197-202`）。
3. `[代码]` `min_loss, batch_indices = self.find_best_perm(pw_loss, perm_reduce=self.perm_reduce)`
   （`pit_wrapper.py:204-207`）。
4. `[代码]` `threshold_byloss`：**若存在 `min_loss > -30` 的样本，则只保留这些样本**，再求均值
   （`pit_wrapper.py:227-230`，多解码器分支在 215-217 对每个 block 单独做）。
5. `[代码]` `reordered = self.reordered_sources(ests, batch_indices)`（`pit_wrapper.py:231,260-264`），
   `return_ests=True` 时一并返回。
6. `[代码]` `pw_mtx_broadcast` 额外加 0.5×第二损失（`pit_wrapper.py:233-234`）。

**C. 最佳排列搜索（`pit_wrapper.py:266-312`）**

- `[代码]` `n_src ≤ 3` 或提供了 `perm_reduce` → 全排列穷举（`find_best_perm_factorial`，
  `pit_wrapper.py:266-274,276-301`）；`n_src > 3` → scipy 匈牙利算法（`pit_wrapper.py:303-312`）。
  Falcon 是 `n_src=1`，实际永远走 1 个排列的穷举。
- `[代码]` 穷举实现：`pwl = pair_wise_losses.transpose(-1,-2)`（变成 `[B,target,est]`），
  `perms_one_hot` 做 `einsum("bij,pij->bp")` 得到每个排列的**源平均损失**（`pit_wrapper.py:279-289`），
  取 `torch.min(dim=1)` 得最佳排列索引 `batch_indices`（`pit_wrapper.py:297-300`）。
- `[代码]` `reordered_sources` 用 `torch.index_select(s, 0, b)` 把 estimate 源维按排列重排
  （`pit_wrapper.py:260-264`），与 target 对齐。

**D. `threshold_byloss` 的真实语义（重要，容易误解）**

代码不是“丢掉坏样本”，而是 **“只要存在 `min_loss > -30`（即最佳排列 SDR < 30 dB）的样本，
就只保留这些样本”**，即**丢弃 SDR > 30 dB 的“过好”样本**；若所有样本都 > 30 dB，过滤结果为空，
代码退回全体均值（`pit_wrapper.py:227-230`，因为 `if ... .nelement() > 0`）。

`[验证]` 构造 3 个样本（2 个普通 + 1 个近乎完美，逐样本最佳排列损失 ≈ `[-6.316, -6.073, -98.286]`）：

| 配置 | 返回值 | 解释 |
|---|---|---|
| `threshold_byloss=True` | **-6.194741** | 只对两个 `> -30` 的样本求均值（丢弃 −98.3 的完美样本） |
| `threshold_byloss=False` | **-36.891914** | 三个样本的平均 |
| 全部样本都 < −30（完美） | **-98.417** | 过滤为空 → 退回全体均值，不报 nan |

（`runs/04-losses.log` §[7]。）`configs/Falcon.yml:120` 训练用 `true`、`:126` 验证用 `false`；
在正常 SNR/SDR 范围内（< 30 dB）阈值几乎不触发，因此训练/验证损失口径差异很小，但**冒烟测试/近完美
样本上会出现“训练损失突然跳变”**。

**E. `new_PITLossWrapper`（`pit_wrapper.py:6-150`，未完成）**

- `[代码]` 构造参数与 `PITLossWrapper` 相同（`pit_wrapper.py:7-20`），但 `forward` 签名是
  `forward(ests, ests_bn, targets, epoch=None, ...)`（`pit_wrapper.py:22`），`pw_mtx` 分支调用
  `self.loss_func(ests, ests_bn, targets, epoch)`（`pit_wrapper.py:25-26`），**只兼容
  `MultistageLoss` 那种 4 参签名**，随后直接 `return pw_loss`（`pit_wrapper.py:75`），**不做 PIT 归约**。
- `[验证]` 与 `PairwiseNegSDR` 组合 → `TypeError: forward() takes 3 positional arguments but 5 were given`；
  换成 4 参 dummy 损失则原样返回 `[B,S,S]` 矩阵（`runs/04-losses.log` §[8]）→ 可判定为**未完成/实验性代码**。

#### 2.1.4 `mixit.py`（309 行）——MixIT 包装器（仓库未使用）

- `[代码]` 第 1-94 行是**整段注释掉**的旧实现（`MixIT`/`parts_mixgen*`）；有效代码从第 96 行开始。
  类 `MixITLossWrapper(loss_func, generalized=True)`（`mixit.py:102-128`）。
- `[代码]` `forward(est_targets[B,nsrc,T], targets[B,nmix,T])`：`generalized=True` → 两路混合、
  生成**所有非平凡二划分**（`parts_mixit_gen`，`mixit.py:250-260`）；`generalized=False` → 要求
  `nsrc % nmix == 0`，分成 `nmix` 个等大小组（`mixit.py:189-218`）。每个划分把估计源按组求和得到
  “估计混合”，调用 `loss_func`（预期 `MultiSrcNegSDR`），取各 batch 的最小损失再对 batch 平均
  （`mixit.py:213-218,272-284`）。
- `[验证]` 4 估计/2 混合：`generalized=True` 得到 14 个有序划分（7 个无序划分 ×2），损失
  **17.777866** 与手算完全一致；`generalized=False` 得 19.077686；`return_est` 输出 `[2,2,120]`
  （`runs/04-losses.log` §[9]）。
- `[代码]` `[推断]` 整个仓库（含 configs/studio）**没有任何代码 import 或实例化它** → 死代码；
  Falcon 是单源 AVTSE，本来也不需要 MixIT。

#### 2.1.5 `nass_contrastive_loss.py`（81 行）——NASS 同空间说话人对比损失

**公式（文件头注释 `nass_contrastive_loss.py:1-11` 与实现逐行对应）：**

```
输入：q ∈ R^{B×D}（query，可训练、来自分离后音频嵌入，L2 归一化）
      p ∈ R^{B×D}（positive，目标说话人嵌入，调用方 no_grad）
      N_b（negative 集合，K_b 个 near 说话人嵌入；可为 None/空）
      β（hard-negative 温度，随 epoch 退火）

对每个 b（nass_contrastive_loss.py:52-77）：
    sim_pos = cos(q_b, p_b)                                  (57)
    L_pos   = α_pos · (1 − sim_pos)                          (58)
    若 K_b > 0：
        sim_neg = cos(q_b, n_k), k=1..K_b                    (64-66)
        若 K_b == 1:  L_neg = α_neg · relu(sim_neg − margin) (68-69)   ← 不做 softmax
        否则: w = softmax(β · sim_neg)                        (71)
              L_neg = α_neg · Σ_k w_k · relu(sim_neg_k − margin)  (72)
    否则: L_neg = 0                                          (73-74)
    loss_total += L_pos + L_neg; n_valid += 1                 (76-77)
返回 loss_total / n_valid                                     (79-81)
```

**关键语义（易错点）**

- `[代码]` 负样本 **K=1 时省去 softmax 计算**（直接 relu），K>1 时才显式做 `softmax(β·cos)`；
  K=1 时 softmax 恒为 1，与公式等价（`nass_contrastive_loss.py:68-72`），不是语义差异。
- `[代码]` `n_valid` 在循环内**无条件 +1**（`nass_contrastive_loss.py:77`），所以“平均”分母恒为 B，
  **没有负样本的样本也计入分母（只贡献 L_pos）**；`n_valid == 0` 分支只有 B=0 才可能触发
  （`nass_contrastive_loss.py:79-81`）。这与 system 里“按有效样本平均”的直觉不同。
- `[代码]` 损失内部**不 detach** `p`/`N`；调用方必须自己 `torch.no_grad()`
  （`system/av_litmodule_nass_v2.py:106-133` 确实包了 no_grad）。若调用方忘记，梯度会流进对齐目标。
- `[验证]` 与独立手算逐项对拍（3 个样本：K=1、无负样本、K=4 且含 hard negative）：
  `β=0 → 0.959351`、`β=1 → 0.967698`、`β=5 → 1.003706`，与手算差 ≤ 6e-8；`margin=1.5`（大于所有
  cos）时负样本项被 relu 截断为 0，只剩正样本项（`runs/04-losses.log` §[10]）。β 越大越强调
  最相似（最难）的负样本，符合设计意图。

**与 `configs/Falcon.yml` 的映射与 β 调度（交叉核对 `system/av_litmodule_nass_v2.py`）**

| `configs/Falcon.yml` 键 | 值 | 流经 | 最终用于 |
|---|---|---|---|
| `spk_contrast.margin` | 0.5 | `audio_train.py:104,119-135` → `AudioVisualLightningModuleNASSv2(margin=...)` → `NASSContrastiveLoss(margin,...)` | 负样本 relu 阈值 |
| `spk_contrast.alpha_pos` | 1.0 | 同上 → `NASSContrastiveLoss` | 正样本项权重 |
| `spk_contrast.alpha_neg` | 1.0 | 同上 | 负样本项权重 |
| `spk_contrast.beta_init` / `beta_final` | 0.0 / 1.0 | 存到 system 属性（`av_litmodule_nass_v2.py:54-56`） | `_get_beta()` |
| `spk_contrast.beta_anneal_epochs` | 200 | 同上 | `_get_beta()` |
| `spk_contrast.loss_weight` | 1.0 | `audio_train.py:104` → `contrastive_weight` | `total_loss = pit_loss + weight·cl_loss`（`av_litmodule_nass_v2.py:137-138`） |

`[代码]` β 调度（`av_litmodule_nass_v2.py:66-70`）：

```
_get_beta():
    if beta_anneal_epochs <= 0:  return beta_final
    progress = min(current_epoch / beta_anneal_epochs, 1.0)
    return beta_init + (beta_final − beta_init) · progress
```

⇒ 用 `configs/Falcon.yml` 的值：`β(t) = min(t, 200)/200`，epoch 0 → 0（均匀权重），epoch 100 → 0.5，
epoch ≥200 → 1.0（硬负样本加权）。注意 `Experiments/checkpoint/Falcon/conf.yml` 的
`spk_contrast` 也是 `beta_init 0.0 / beta_final 1.0 / beta_anneal_epochs 200`，两份配置一致。
`[代码]` system 的 `__init__` 默认值是 `beta_final=1.5, beta_anneal_epochs=100`
（`av_litmodule_nass_v2.py:47-48`），但 `audio_train.py` 总会显式传 config 值，默认值不会生效
——容易误导读者。

**训练中的完整调用链（`system/av_litmodule_nass_v2.py:73-140`）**

1. `[代码]` video 冻结前向：`with torch.no_grad(): mouth_emb = self.video_model(mouth_in)`（86-87）。
2. `[代码]` 主前向：`est_sources, sep_audio_embs = self.audio_model(mixtures, mouth_emb)`（90），
   形状 `[B,n_src,T]` / `[B,n_src,N,T',F]`（91-92）。
3. `[代码]` PIT 损失：`pit_loss = self.loss_func["train"](est_sources, targets)`（95）→
   `PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx", threshold_byloss=True)`。
4. `[代码]` query：`sep_audio_embs[:,0]` → `bool pool_spk_embed` → `[B,D]` **带梯度**（102-103）。
5. `[代码]` positive：`targets[:,0]` → `encode_stft` → `pool_spk_embed`，**no_grad**（106-109）。
6. `[代码]` negatives：每个样本的 `near_audios[b]` 截/补到 `seg_len` 后 `encode_stft` + `pool_spk_embed`
   （112-132）；`n_near==0` 时 append `None`（120-122）；同时按 0/1/2/3/4+ 桶计数并 log（113-118,182-183）。
7. `[代码]` `cl_loss = contrastive_loss(embed_query, embed_pos, embed_negs_list, beta)`（134）；
   `total_loss = pit_loss + contrastive_weight·cl_loss`（137-138）；并记录
   `train_beta/train_cos_sim_pos/train_cos_sim_neg_avg/train_cos_sim_neg_max/train_neg_active_ratio`
   （172-180）。
8. `[推断]` `train_spk_contrast_loss` 记为 `weighted_cl`（即已乘 loss_weight），而 `train_pit_loss`
   是未加权的 pit_loss；两者相加才等于 `train_loss`。

#### 2.1.6 `sisnri.py`（42 行）——`SISNRi` 对象（有致命 bug、未被使用）

- `[代码]` `SISNRi.__init__` 第 6 行 `super(Loss, self).__init__()` 引用了**未定义名字 `Loss`**
  （本文件没有 `from ... import Loss`，类也继承 `object`）。`[验证]` `SISNRi()` 抛
  `NameError: name 'Loss' is not defined`（`runs/04-losses.log` §[11]）→ 该类**不可实例化**。
- `[代码]` 数学（若绕过 `__init__`）：`sisnr(mix, est, ref)` 返回
  `SI-SNR(est, ref) − SI-SNR(mix, ref)`，即 **SI-SNR improvement（SI-SNRi，dB）**（`sisnri.py:8-24`），
  与定义一致；`compute_loss(mix, ests, refs)` 对源排列穷举取 max，返回 `−mean(max)`（`sisnri.py:26-42`）。
- `[验证]` 用 `SISNRi.__new__(SISNRi)` 绕过构造后与手算 `max|diff|=3.8e-6`（float32）→ 数学正确，
  仅构造坏。`[代码]` 仓库内无任何调用（只有 `losses/__init__.py:20` 的 export）→ 死代码。

### 2.2 `look2hear/metrics/`

三个 tracker 都只写 CSV，不做训练期日志；接口都是 `__call__(mix, clean, estimate, key)` +
`final()`（+ `allwrapper` 的 `get_mean/get_std`）。**只有 `ALLMetricsTracker` 被评测入口使用**。

#### 2.2.1 `wrapper.py`（84 行）`MetricsTracker`——唯一使用真 SDR 的实现（但无人调用）

- `[代码]` 构造（`wrapper.py:19-34`）：CSV 列 `["snt_id","sdr","sdr_i","si-snr","si-snr_i"]`；
  `pit_sisnr = PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=False), pit_from="pw_mtx")`（29-31），
  `pit_snr` 同理（32-34）。**注意 `zero_mean=False`**：SI-SDR 不去均值（非标准写法）。
- `[代码]` 逐句计算（`wrapper.py:36-60`）：
  - `sisnr = pit_sisnr(estimate[None], clean[None])`（负 SI-SDR 损失）；
  - 基线：`mix` 广播到 `n_src` 后同样计算（`wrapper.py:39-41`）；
  - `sisnr_i = sisnr − sisnr_baseline`；
  - **SDR 走 fast_bss_eval**：`sdr = -fast_bss_eval.sdr_pit_loss(estimate, clean).mean()`（44），
    基线同理（45）。`[验证]`（外部包源码核对，见 §8.3）`fast_bss_eval.sdr_pit_loss(est, ref)` 内部
    调用 `sdr(ref, est, change_sign=True)`，返回**负 SDR**，且默认 `filter_length=512` 做卷积滤波
    → 这是 BSS-Eval-v4 风格的 filtered SDR，**不是**简单的投影 SDR。
  - CSV 写入时 `"sdr": sdr.item()`、`"si-snr": -sisnr.item()`（`wrapper.py:48-54`）——两者都转成正 dB。
- `[代码]` `final()` 追加 `avg`/`std` 两行（`wrapper.py:67-83`）。
- `[代码]` 死依赖：第 6 行 `from torch_mir_eval.separation import bss_eval_sources` **未使用**，但使
  `torch-mir-eval` 成为硬依赖（`requirements.txt:26`）；第 12 行 `singlesrc_neg_sisdr` 也未使用。
- `[代码]` 仓库内无任何入口 import `MetricsTracker`（只有 `metrics/__init__.py:1`）。

#### 2.2.2 `allwrapper.py`（120 行）`ALLMetricsTracker`——**audio_visual_test.py 实际使用的类**

- `[代码]` CSV 列 `["snt_id","sdr","sdr_i","si-snr","si-snr_i","pesq","stoi"]`（`allwrapper.py:23-28`）。
- `[代码]` 两个 PIT wrapper 直接用模块级实例：`pit_snr = PITLossWrapper(pairwise_neg_snr, ...)`、
  `pit_sisnr = PITLossWrapper(pairwise_neg_sisdr, ...)`（`allwrapper.py:27-28`）——即
  `PairwiseNegSDR("snr"/"sisdr")`，`zero_mean` 用默认 **True**。
- `[代码]` **“sdr” 列其实是负 SNR**：`sdr = self.pit_snr(estimate[None], clean[None])`（38-41），
  而 `pairwise_neg_snr = PairwiseNegSDR("snr")`（`matrix.py:374`）→ 这是**尺度相关**的 SNR，
  不是 SDR（§2.1.2-A 已用数值证明）；`sdr_i` 因此是 SNR 改善量（SNRi），不是 SDRi。
- `[代码]` **符号 bug**：`row["sdr"] = sdr.item()`（`allwrapper.py:62`）直接写入 PIT 损失的负值，
  而累加时 `self.all_sdrs.append(-sdr.item())`（72）取反；`si-snr` 则在行内取反（64）与累加一致（74）。
  结果：**逐句行的 `sdr`/`sdr_i` 与 `avg`/`std` 行符号相反**。
  `[验证]`（`runs/04-metrics.log` §[1]）单个样本输出：

  | 行 | sdr | sdr_i | si-snr | si-snr_i | pesq | stoi |
  |---|---|---|---|---|---|---|
  | 逐句 | +5.481 | −10.021 | −5.727 | +21.644 | 3.14 | 0.9 |
  | avg | −5.481 | +10.021 | −5.727 | +21.644 | 3.14 | 0.9 |
  | std | 0 | 0 | 0 | 0 | 0 | 0 |

  其中逐句 `sdr = +5.481` 正是 `PairwiseNegSDR('snr')` 的损失（= −SNR，且该样本 SNR=−5.481 dB），
  `avg` 行再取负 ⇒ +SNR。**同一列语义/符号不一致**，下游若直接读逐句行会得到反号的“SDR”。
- `[代码]` **PESQ 实参顺序传反**：`pesq(estimate.squeeze(0), clean.squeeze(0), 16000)`
  （`allwrapper.py:50`），而 `pypesq.pesq(ref, deg, fs)` 的合同是 **ref=干净、deg=估计**
  （外部源码 `pypesq-1.2.4/pypesq/__init__.py:7`，见 §8.3）；PESQ 是**非对称**指标，传反会有偏差。
  `[验证]` 记录到的第 1 个实参确实是 estimate（`runs/04-metrics.log` §[1]）。
- `[代码]` STOI 顺序正确：`stoi(clean.squeeze(0), estimate.squeeze(0), 16000, extended=False)`
  （`allwrapper.py:53-58`），pystoi 合同 `stoi(x=clean, y=deg, fs)`。
- `[代码]` PESQ/STOI 传的是 `squeeze(0)` 后的一维张量，只有 `n_src=1` 成立（`n_src>1` 时是二维，
  pypesq 第 27 行会抛维度错误）。
- `[代码]` `final()` 写 `avg`/`std`（`allwrapper.py:99-120`）；`get_mean/get_std` 返回 dict
  （79-97），供进度条显示。
- `[代码]` 死依赖：`pypesq`/`pystoi` 只在 import 期需要；本环境未安装 → 本次用 stub（§8）。

#### 2.2.3 `splitwrapper.py`（115 行）`SPlitMetricsTracker`——3 源拆分指标（死代码）

- `[代码]` 目标场景：`n_src=3` 的 LRS3-2Mix 类评测，把前 2 个源算 “two”、第 3 个源算 “one”
  （`splitwrapper.py:44-74`），CSV 9 列 `one_snr, one_snr_i, one_si-snr, one_si-snr_i, two_snr, ...`
  （27-37）。
- `[代码]` 先用 `pit_snr(..., return_ests=True)` 得到按 SNR 最佳排列重排后的估计（45-47），再拆分；
  `final()` 只写 `avg` 行（98-115，无 std）。
- `[验证]` 用 3 源合成数据跑通形状逻辑（`runs/04-metrics.log` §[4]）；`[代码]` 仓库内无入口调用。

#### 2.2.4 `__init__.py`（5 行）

`[代码]` 导出 `MetricsTracker`、`SPlitMetricsTracker`、`ALLMetricsTracker`。因 `metrics/__init__.py:1`
会先 import `wrapper.py`，所以**即使只用 `ALLMetricsTracker`，也必须有 `torch_mir_eval` 和
`fast_bss_eval` 才能 import**（对评测环境是隐藏硬依赖）。

### 2.3 `look2hear/videomodels/`（8 个文件）

#### 2.3.1 `__init__.py`（60 行）

- `[代码]` 导入 7 个类 + 3 个 `update_*_parameter` 函数并写 `__all__`（`videomodels/__init__.py:3-25`）；
  提供 `register_model`（28-42）与 `get(identifier)`（45-60，按小写类名查全局）。
- `[代码]` 训练/评测入口都是 `getattr(look2hear.videomodels, config[...]["videonet_name"])`
  （`audio_train.py:61-63`、`audio_visual_test.py:150-152`、`evaluated_mac_params.py:51-53`），
  **没有用 `videomodels.get()`**；`register_model` 也无人调用（死代码）。

#### 2.3.2 `frcnn_videomodel.py`（122 行）——论文实际使用的视觉编码器

**结构（`[代码]` + `[验证]`，输入 `x: [B,1,T,H,W]`，训练/评测实际 `H=W=88`，灰度单通道）：**

```
frontend3D (frcnn_videomodel.py:58-70)
  Conv3d(1→64, k=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)   # 时间同长、空间 /2
  BatchNorm3d(64)
  PReLU(64) 或 ReLU（由 relu_type 决定，53-57）
  MaxPool3d(k=(1,3,3), stride=(1,2,2), padding=(0,1,1))                  # 时间同长、空间 /2
        → [B,64,T,22,22]   （输入 88×88：88→44→22）

threeD_to_2D_tensor (frcnn_videomodel.py:16-19)
        → [B*T,64,22,22]   （B 与 T 维合并，逐帧做 2D CNN）

trunk = ResNet(BasicBlock, [2,2,2,2], relu_type) (frcnn_videomodel.py:42；resnet.py:73-140)
  layer1: 2×BasicBlock 64   → 22×22
  layer2: stride 2, 128     → 11×11
  layer3: stride 2, 256     → 6×6（floor((11+2−3)/2)+1 = 6）
  layer4: stride 2, 512     → 3×3
  avgpool = AdaptiveAvgPool2d(1) → [B*T,512]（resnet.py:95,133-140）

reshape → [B,T,512] → transpose → 输出 [B,512,T]（frcnn_videomodel.py:83-84）
```

- `[验证]` 参数量 **11,186,688**（=11.1867 M）；输入 `[1,1,50,96,96]`/`[1,1,50,88,88]` 均输出
  `(1,512,50)`；`T=25 → 25`、`T=1 → 1`——**时间维完全不降采样**（`runs/04-video.log` §[1]）。
- `[代码]` 时间维无关的层：Conv3d 的 `kernel=(5,7,7)` 第一维是时间核 **5**，配合 `stride=(1,...)`、
  `padding=(2,...)` → 时间长度不变；MaxPool3d 时间 `kernel=1, stride=1, padding=0` → 时间长度不变
  （`frcnn_videomodel.py:59-69`）。**因此“25 fps → 12.5 fps 下采样”在本编码器内不存在**（见 §4.5）。
- `[代码]` 输入尺寸：模块本身**不做 96→88 裁剪**，也不校验尺寸；96/88/64 都能前向
  （`[验证]` `runs/04-video.log` §[4]）。88×88 的裁剪发生在数据集预处理（§2.3.2 末）或 Studio。
- `[代码]` 预训练加载（`frcnn_videomodel.py:71-73,86-88,106-118`）：
  `pretrain` 非空 → `torch.load(path)["model_state_dict"]` → `update_frcnn_parameter`：
  跳过 key 含 `"tcn"` 的权重（110-111），把其余权重 `model_dict.update(...)` 后
  `load_state_dict(model_dict)`（默认 strict=True，114-115），最后**把所有参数的 `requires_grad`
  置 False**（116-117）。
  - `[验证]` 用模型自身 state_dict + 一条多余 `tcn.dummy` 键调用，加载成功且全部参数冻结
    （`runs/04-video.log` §[2]）；`pretrain` 文件不存在 → `FileNotFoundError`
    （仓库无 `pretrain_zoo/`）。
  - `[推断]` 第 114-115 行要求预训练权重与当前模型 key 集合**恰好一致**（除 `tcn.*`），否则
    `load_state_dict` 会因 unexpected keys 报错；因此这份 `frcnn_128_512.backbone.pth.tar`
    必须来自同构 FRCNN 实现。
- `[代码]` `train(mode=True)` 覆写（`frcnn_videomodel.py:90-95`）：先 `super().train(mode)`，若
  `mode=True` 则把所有 `_BatchNorm` 置为 `eval()` → 训练时用冻结的 BN running stats。
  **没有 `return self`** → `[验证]` `FRCNNVideoModel(...).eval()` 的返回值是 **None**
  （`nn.Module.eval()` 返回 `self.train(False)` 的返回值）；`audio_visual_test.py:169-170` 正是**以语句
  形式**调用（未链式赋值），规避了该陷阱（Studio `falcon_runtime.py` 注释也提到）。
  **不要写 `video_model = video_model.eval()`**。
- `[代码]` `check_parameters`（`frcnn_videomodel.py:98-103`）返回 M 为单位的参数量，但**未被使用**
  （`evaluated_mac_params.py:26-31` 里又复制了一份，也未使用）。
- `[代码]` `backbone_type="resnet"` 时 `frontend_nout=64, backend_out=512`（39-42）；
  `"shufflenet"` 时（43-51）用 `ShuffleNetV2(input_size=96, width_mult=...)` 的
  `features+conv_last+globalpool` 作为 trunk（跳过 `conv1/maxpool`），`frontend_nout=24`
  ——因为 `frontend3D` 输出 24 通道恰好是 `features` 的输入通道；`backend_out=1024`（w≠2.0）
  或 2048（w=2.0）。
- `[代码]` `width_mult` 对 `backbone_type="resnet"` **完全无效**（resnet 分支不使用它，
  `frcnn_videomodel.py:39-42`），只在 shufflenet 分支生效；`configs/Falcon.yml:7` 写
  `width_mult: 1.0` 但对实际 resnet 无影响。
- `[代码]` 输出维度 512 与音频网络 `pretrained_vout_chan: 512`（`configs/Falcon.yml:13`）匹配；
  `Falcon.video_bottleneck` 用 `video_bn_params.kernel_size=-1` 构造为 0 参数 Identity 语义
  （Lead 已验证，`docs/verification/runs/lead-model-params.log`）。

**96×96 → 88×88 中心裁剪到底在哪做。**

| 场景 | 位置 | 细节 |
|---|---|---|
| 训练 | `look2hear/datas/transform.py:147-160`：`get_preprocessing_pipelines()["train"]` = `Normalize(0,255) → RandomCrop((88,88)) → HorizontalFlip(0.5) → Normalize(0.421,0.165)`；数据集在 `avspeech_dataset_near.py:167-174`（管线调用与切片在 169-174）对整段 `[T,96,96]` 应用 | 训练是**随机裁剪** 88×88 |
| 验证/测试 | `transform.py:161-168`：`Normalize(0,255) → CenterCrop((88,88)) → Normalize(0.421,0.165)`；`AVSpeechEvalDataset` 在 `avspeech_dataset.py:138,189-192` 应用 | 评测是**中心裁剪** 88×88（左右/上下各去 4 像素） |
| Studio 推理 | `studio/falcon_runtime.py:95-104`：`frames[:, 4:92, 4:92].astype(np.float32)/255.0`，再 `(x−0.421)/0.165` | 显式 `[4:92]`，与 CenterCrop 等价 |
| FRCNN 内部 | **没有** | `frcnn_videomodel.py:75-84` 无 crop/无尺寸断言 |

`[验证]` 用伪造的 1 s 数据（25 帧 96×96，`.pt` 存 **uint8**）跑 `AVSpeechEvalDataset.__getitem__`，
返回 `mouth.shape=(25,88,88)`，与手工“`Normalize(0,255)→[4:92]→Normalize(0.421,0.165)`”的最大差为
**2.57e-7**（`runs/04-eval-flow.log` §[8]）→ 裁剪确实在 dataset 预处理里，且顺序与 `transform.py` 一致。
**dtype 不是常量**：取决于 `.pt` 存储类型——uint8 时 numpy 减法升位为 float64，float32 时保持
float32（data-pipe 独立实测，05 文档 A5/D3）；真实 `.pt` 的存储 dtype 未知（Q13）。下游会统一转
float32：评测 `torch.from_numpy(...).float()`（`audio_visual_test.py:230`）、训练
`mouth_in.type_as(mixtures)`（`av_litmodule_nass_v2.py:87`），因此不影响 FRCNN 输入。

#### 2.3.3 其余视觉模型（均未被任何 config 选中，但不能删——`__init__.py` 会 import）

| 文件 | 类 | 参数量 `[验证]` | 输出 `[验证]` | 说明 |
|---|---|---|---|---|
| `resnet.py`（140 行） | `ResNet` / `BasicBlock` | 被 FRCNN 复用（trunk，含在 11.19 M 内） | — | 18 层式 `[2,2,2,2]`，无第一个 7×7 conv、无 fc；`relu_type=prelu/relu`；`gamma_zero` 可将 bn2 权重置 0（`resnet.py:108-111`） |
| `resnet1D.py`（139 行） | `ResNet1D` / `BasicBlock1D` | 未单独测 | 未调用 | 1D 音频 ResNet；首个 conv `kernel 80, stride 4, padding 38`（`resnet1D.py:80-82`），`avgpool=AvgPool1d(kernel=21, padding=1)`（94）——注释说用于 LRW 降采样到 25 fps；**无 flatten**，输出 `[B,512,T']` |
| `shufflenetv2.py`（173 行） | `ShuffleNetV2` | standalone 2.28 M（默认 n_class=1000）；FRCNN 内 trunk 1.26 M | 见下 | `stage_repeats=[4,8,4]`、width_mult ∈ {0.5,1.0,1.5,2.0}（110-131）；`conv1=conv_bn(3,24,2)` 期望 **3 通道**，但 FRCNN 只取 `features`（输入 24 通道），不会触发 |
| `light_videomodel.py`（124 行） | `LightVideomodel` | **1.259 M** | `(1,1024,50)` | 与 FRCNN(shufflenet) 几乎是同一结构的复写；`pretrain` → `update_light_parameter` 冻结；未被 config 选中 |
| `wujian_videomodel.py`（182 行） | `WujianVideoModel` | **11.318 M** | `(1,256,50)` | `front3D`（Conv3d+PReLU+MaxPool3d）+ 自带 `ResNet`（输出 `class_dim=256`，含 fc+bnfc，`wujian_videomodel.py:52-53,99-139`）；`update_wujian_parameter` 做 `front3D.*→front3d.conv3d.*`/`resnet.*→front3d.resnet.*` 的 key 重映射（167-182）；未被 config 选中 |
| `resnet_videomodel.py`（125 行） | `ResNetVideoModel` | **11.187 M** | `(1,512,50)` | FRCNN-resnet 的“干净复写”：同样 frontend3D+ResNet，另有 `_initialize_weights_randomly`（70-105），`pretrain` → `update_resnet_parameter`；未被 config 选中 |
| `frcnn_videomodel.py`（122 行） | `FRCNNVideoModel` | **11,186,688** | `(1,512,T)` | **唯一被 config 选中的**（`configs/Falcon.yml:2-8`、`studio/configs/Falcon.yml:3`、`studio/configs/FalconBig.yml:166`、`Experiments/checkpoint/Falcon/conf.yml:168`） |

`[代码]` 所有 `update_*_parameter` 的共同语义：跳过 `tcn` 权重 → 合并 state_dict → load → 全部
`requires_grad=False`（`frcnn:106-118`、`resnet_videomodel:112-125`、`light:106-118`、`wujian:167-182`）。
`[代码]` `configs/*.yml` 里的 `width_mult` 只对 shufflenet 分支有意义，而四个 config 全是
`backbone_type: resnet` → `width_mult` 是**误导性配置项**。
`[验证]` shufflenet 分支对 64×64 输入会崩：`globalpool = AvgPool2d(int(96/32)) = AvgPool2d(3)`，
64 输入时特征图为 2×2 → `RuntimeError: Calculated output size: (1024x0x0)`；
88/96 输入时特征图为 3×3 正常（`runs/04-video.log` §[3]）。`[推断]` 若切换到 shufflenet，
输入必须保证空间下采样后为 3×3（88/96 都可以）。

### 2.4 `audio_visual_test.py`（329 行）——离线评测入口

**文件级结论**：这是“从 Lightning `.ckpt` 直接抽 audio/video 权重、单/多卡跑测试集、逐句算
SI-SDR/SNR/PESQ/STOI、合并 CSV”的脚本；与 README 描述大体一致，但存在 **P0-1**（README 评测命令
不可开箱复现：B2/B3/B4）、**P0-2**（`sdr/sdr_i` 实为 −SNR/SNRi 且符号不一致）与**条件性 P0-3**
（单声道数据下评测/训练验证在 `torch.stft` 崩溃），详见 §3.4/§4.6/§7。

**A. `_find_best_ckpt(exp_dir)` 顺序（`audio_visual_test.py:78-108`）**

1. `[代码]` `best_k_models.json` 存在且非空 → `min(best_k, key=best_k.get)`，取**值最小**的路径
   （84-93）；注释说“val_loss is negative; lower is better”。
   - `[代码]` PL 的 `ModelCheckpoint.best_k_models` 存的是**原始监控值**（`pytorch_lightning 2.6.6`
     `callbacks/model_checkpoint.py:929,965`: `current = monitor_candidates.get(self.monitor)`；
     `self.best_k_models[filepath] = current`），而 `monitor="val_loss/dataloader_idx_0"`、
     `mode="min"`（`audio_train.py:140-146`）→ 值就是 val_loss；Falcon 的 val_loss = −SI-SDR，
     为负 → 取 min 正确。
   - `[代码]` 只有 `os.path.exists(best_path)` 通过才返回（91），否则**静默**继续下一优先级。
   - `[验证]` 仓库自带的 `Experiments/checkpoint/Falcon/best_k_models.json` 里 5 条路径都是
     `${ROOT}/Falcon/epoch=*.ckpt`（字面占位符，代码未做 env 展开）→ 全部不存在 → 落到 last.ckpt
     → 也不存在 → 落到 `epoch=*.ckpt` → 也不存在（仓库不提供权重）→ **`FileNotFoundError`**。
     `[验证]` 直接调用 `_find_best_ckpt("Experiments/checkpoint/Falcon")` 抛
     `FileNotFoundError: No checkpoint found in ...`（`runs/04-eval-flow.log` §[1a]）。
2. `[代码]` `last.ckpt`（95-99）。
3. `[代码]` `sorted(glob("epoch=*.ckpt"))[-1]`（101-106）。
   - `[代码]` **字符串排序 bug**：`epoch=1000.ckpt` 会排在 `epoch=742.ckpt` 之前
     （`"epoch=1" < "epoch=7"`），配置 `epochs: 1000` 且早停未触发时可能选错“最新”。
     `[验证]` 临时目录中同时放 `epoch=742.ckpt`、`epoch=1000.ckpt`，返回的是 742
     （`runs/04-eval-flow.log` §[1d]）。
4. 都没有 → `raise FileNotFoundError`（108）。

**B. `_load_models_from_ckpt(ckpt_path, config, device)`（`audio_visual_test.py:111-172`）**

1. `[代码]` `torch.load(ckpt_path, map_location="cpu", weights_only=False)`（119）→ 取
   `ckpt["state_dict"]`（120）。**硬要求 ckpt 是 Lightning 风格**（顶层有 `state_dict` 键）。
   `[验证]` 用一个裸 `state_dict`（无 `state_dict` 键）保存的文件调用 → `KeyError: 'state_dict'`；
   用 `{"state_dict": ...}` 包装则成功（`runs/04-eval-flow.log` §[4]）。
   `[验证]`（R3，Lead 已下载真实权重）发布文件顶层**确实是** `{architecture, epoch, state_dict}`：
   `falcon.pt` = `architecture='FalconSpeakerContrastive'`、`epoch=714`、state_dict 含 525 个
   `audio_model.*` + 137 个 `video_model.*`；`falcon-big.pt` = `architecture='SpeakerContrastiveV2'`、
   `epoch=715`、517+137 个 key（`studio/falcon_model/PROVENANCE.md:36-40` 有同样声明）。实际调用本
   函数加载 falcon.pt 成功（`✓ audio_model loaded: 525 keys` / `✓ video_model loaded: 137 keys`，
   参数量 537,697/11,186,688）→ **`ckpt["state_dict"]` 不是风险点，Q1 关闭**。README:168 的真实
   断点是 conf.yml 类名（B2）与 `${DATA_ROOT}`（B3）。详见 `runs/r3-weights.log`。
2. `[代码]` 音频模型：`copy.deepcopy(config["train_conf"]["audionet"]["audionet_config"])` 后
   `getattr(look2hear.models, audionet_name)(sample_rate=..., **cfg)`（128-132）；
   从 ckpt 抽 `audio_model.*` 去前缀（135-137），`load_state_dict(strict=False)`（139），
   missing/unexpected 只打印数量（140-144）。
3. `[代码]` 视频模型：deepcopy `videonet_config`，**弹掉 `pretrain` 并传 `pretrain=None`**
   （147-149,150-152）——避免用预训练权重覆盖 ckpt 中已训练的视频权重（视频在训练中是被冻结的，
   实际等于同一份权重）；抽 `video_model.*` 后 **`strict=True`** 加载（159-160）；若 ckpt 无视频权重
   则回退到 `pretrain` 路径（162-167，且把含 `pretrain` 的原 cfg 再传一遍）。
4. `[代码]` `model.to(device).eval(); video_model.to(device).eval()`（169-170）。
   `[验证]` 用伪造的 `{"state_dict": {audio_model.*: ..., video_model.*: ...}}` ckpt 做函数级回环：
   audio 525 个 key、video 137 个 key，加载成功且参数与源模型一致（`runs/04-eval-flow.log` §[4]）。

**C. `process_worker(rank, world_size, config)`（`audio_visual_test.py:178-265`）**

1. `[代码]` `torch.cuda.set_device(rank)` + `device = cuda:{rank}`（179-180）→ **强制要求 CUDA**，
   无 CPU 回退（机器学习环境无 GPU 时脚本直接崩）。
2. `[代码]` `data_cfg` 从 `config["train_conf"]["datamodule"]["data_config"]` 过滤掉
   `use_near`/`train_samples`（183-186），再 `AVSpeechDataModuleWithNear(**data_cfg, use_near=False)`
   （187）→ 评测**不用 near.json**（对比学习只在训练用）；`datamodule.setup()`（188）会**连训练集
   一起构造**（`avspeech_dataset_near.py:277-295`：train/val/test 全建），因此评测也要求
   `train_dir` 可读（只是不迭代）。
3. `[代码]` `_, _, test_set = datamodule.make_sets`（189），按 `rank::world_size` 切片索引
   （191-193）→ 多卡是**样本级轮转切分**，不保证负载严格均衡。
4. `[代码]` `exp_dir = cwd/Experiments/checkpoint/<exp_name>`（196-198）；ckpt 优先 CLI
   `config["ckpt_path"]`，否则 `_find_best_ckpt`（201）。
5. `[代码]` 结果目录 `exp_dir/results/`，每 rank 写 `metrics_rank{rank}.csv`（205-208）。
6. `[代码]` 数据迭代**没有 DataLoader**：`test_set[idx]` 直接取样本（228-229、249-250），
   batch 恒为 1；`tensors_to_device(test_set[idx], device)` 对 tuple 返回**长度 5 的 list**
   （`look2hear/utils/torch_utils.py:35-36`），再解包为 `(mix, sources, mouth, key, _)`（`:19-44`）。
7. `[代码]` 前向：`mouth_emb = video_model(torch.from_numpy(mouth[None,None]).float())`（230-231）、
   `est_sources = model(mix[None], mouth_emb)`（231-232）——即 `[1,1,T,88,88]` 视觉输入、
   `[1,T]` 音频；`model.eval()` 下 `FalconSpeakerContrastive.forward` 只返回分离音频
   （`models/Falcon_Contrastive.py:94-122`）。
8. `[代码]` 指标：`metrics(mix=mix, clean=sources.unsqueeze(0), estimate=est_sources.squeeze(0), key=key)`
   （234-237）；异常捕获只忽略消息含 "all zeros"/"nan" 的 `ValueError`（238-242）。
9. `[代码]` rank0 用 rich 进度条每 50 句更新一次 `metricscolumn.update(metrics.get_mean())`
   （211-245）；其他 rank 静默（246-263）。`metrics.final()` 写 avg/std 并关闭文件（265）。

**补充（R3 评审后补全的评测数据契约）**
- `[代码]` **整句、无 2 s 截断/补齐**：`AVSpeechEvalDataset` 用 `start=0, stop=None` 读整段
  （`avspeech_dataset.py:179-191`），因此 T 可变、嘴部 F 帧可变；模型靠 `istft(length=T)` 精确保长
  （`Falcon.py:1166,1182-1188`，data-pipe 实测 T=48513,F=75→out (1,1,48513)）。
- `[代码]` 每样本仍会尝试读 `<mix>_meta.json` 统计 `n_near`（`avspeech_dataset.py:201-210`），评测中
  该值被丢弃（`audio_visual_test.py:229` 的 `_`）；meta 缺失时静默 0 → 评测目录不带 meta 也能跑，
  但每样本多一次文件 IO。
- `[代码]` **评测不过滤 <2 s 短音频**：训练集过滤 `<32000` 样本（`avspeech_dataset_near.py:111-122`），
  `AVSpeechEvalDataset` 全量加载（`:161-171`）→ 训练/评测样本集合不同。
- `[代码]` `val_dataloader/test_dataloader` 在评测路径**从不创建/调用**；`setup()` 必须成功构建
  train/val/test 三个 dataset，因此即使只评测也要求 `train_dir`/`valid_dir`/`test_dir` 三处的
  `mix.json`+`target.json` 存在（`avspeech_dataset_near.py:277-295`）；near.json 不参与评测。

**D. `main(config)` 与结果合并（`audio_visual_test.py:270-313`）**

1. `[代码]` `os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]`；`world_size = len(gpus.split(","))`
   （272-274）。
2. `[代码]` `world_size > 1` → `mp.spawn(process_worker, args=(world_size, config), nprocs=world_size)`
   （279-280）；否则 `process_worker(0, 1, config)`（281-282）。
3. `[代码]` 合并：逐个读 `metrics_rank{i}.csv`（291-294）→ `pd.concat`（297）→
   `exp_dir/results/metrics_combined.csv`（298-299）→ 打印 `numeric_cols.mean()`（302-307）→
   删除 per-rank 文件（309-313）。
   - `[代码]` **注意**：每个 rank 的 CSV 末尾有 `avg`/`std` 行，合并后它们会作为普通样本行进入
     `metrics_combined.csv`；`numeric_cols.mean()` 也会把这些 avg/std 行一起平均 → 最终打印的
     “Final Average Metrics (All GPUs)” 与真实逐句均值有偏差（多卡时尤其明显）。P2。
4. `[代码]` `hf://` 处理在 `__main__`，**spawn 之前**下载一次（315-327）：`rpartition("/")` 拆
   `repo_id/filename`，`hf_hub_download` 缓存到 `~/.cache/huggingface`，把本地路径写入
   `arg_dic["ckpt_path"]`。`[验证]` `"hf://JusperLee/Falcon/falcon.pt"` → repo_id
   `JusperLee/Falcon`、filename `falcon.pt`（`runs/04-eval-flow.log` §[7]）。
5. `[代码]` `compute_metrics = ["si_sdr","sdr"]`（75）**定义了但从未使用**；`--conf_dir` 默认值
   `Experiments/checkpoint/NASSv2_lrs3_complex_nonmoving_img89/conf.yml`（65）在仓库中不存在。

**E. 与 README 的一致性核对**

| README 描述（行号） | 代码实际 | 判定 |
|---|---|---|
| `python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml --gpus 0`（165） | 该 conf.yml 的 `audionet_name: SpeakerContrastiveV2`，而 `look2hear.models` 只有 `Falcon`/`FalconSpeakerContrastive` → `getattr` 抛 `AttributeError`；且数据路径是未展开的 `${DATA_ROOT}`，无 ckpt 文件 | ❌ **不可直接复现**（P0-1；主因 B2+B3+B4，**不是** ckpt 格式；详见 §3.4/§7 Q2/Q3） |
| `--ckpt_path hf://JusperLee/Falcon/falcon.pt`（168） | hf 下载逻辑正确；ckpt 格式**已证可行**（R3：顶层 `{architecture, epoch, state_dict}`，525+137 keys，真实 `_load_models_from_ckpt` 成功）；但命令仍受 B2/B3 阻断 | ✅ ckpt 格式；❌ 仍因 B2/B3 失败（P0-1） |
| 按 `best_k_models.json`（最低项）→ `last.ckpt` → `epoch=*.ckpt`（174） | 与 `_find_best_ckpt` 顺序一致；但自带 JSON 是失效占位路径、权重未随仓库发布 | 逻辑 ✅；数据 ❌（仓库无 ckpt，见 B4） |
| 结果写到 `Experiments/checkpoint/Falcon/results/metrics_combined.csv`（174） | `main()` 的 exp_dir/CSV 路径一致 | ✅ |
| 评测 SI-SDR / SDR（162-174 与结果表 178-191） | 实际 `ALLMetricsTracker` 的 `sdr/sdr_i` = **负 SNR/SNRi**，不是 SDR/SDRi；只有死代码 `MetricsTracker` 用 `fast_bss_eval` 真 SDR | ❌ **口径不符**（P0-2，见 §4.6） |
| 多卡 `--gpus 0,1,2,3`（`audio_visual_test.py:21-24` 的用法示例 + `:270-282` 的 spawn 逻辑） | `mp.spawn` + `CUDA_VISIBLE_DEVICES` + 样本切片，逻辑可用 | ✅ |

### 2.5 `evaluated_mac_params.py`（151 行）——参数量/MACs 统计

**结论**：这是**手写的 ptflops 脚本**（不是 thop、不是 fvcore，也没有自定义 hook），只对**音频网络**
调 `ptflops.get_model_complexity_info`；README 的 0.5 M/10.6 G 是音频网络口径，11.72 M 是
“音频+冻结 FRCNN”的总参数口径。

- `[代码]` 依赖与配置：`from ptflops import get_model_complexity_info`（21）；
  `with open("configs/Falcon.yml")`（39）→ `prepare_parser_from_dict`（`look2hear/utils/parser_utils.py`）
  把 YAML 变成 `arg_dic`，因此超参不在脚本里硬编码（但配置文件名硬编码）。
- `[代码]` 模型构建（44-53）：音频 `getattr(look2hear.models, audionet_name)(**audionet_config)`
  （**不传 `sample_rate`**，用 `Falcon.__init__` 默认 16000）；视频
  `getattr(look2hear.videomodels, videonet_name)(**videonet_config)`——**保留 `pretrain` 路径**，
  所以脚本运行前必须有 `pretrain_zoo/frcnn_128_512.backbone.pth.tar`，否则 `FileNotFoundError`
  （`frcnn_videomodel.py:86-88`）。
- `[代码]` `audiomodel.cuda()`（48）→ **要求 CUDA**。
- `[代码]` 输入构造（77-82）：`with torch.cuda.device(0)`；`video_model.cuda().eval()`；
  `a = randn(1,32000)`（2 s 音频）；`v = randn(1,1,50,64,64)`（**50 帧 64×64**，
  不是评测/训练实际使用的 88×88）；`v = video_model(v)` → `[1,512,50]`。
- `[代码]` 统计（86-96）：`model = audiomodel.cuda().eval()`；
  `get_model_complexity_info(model, (a, v), input_constructor=input_func, as_strings=False,
  print_per_layer_stat=False, verbose=False)`，其中
  `input_func(input)` 返回 `{"audio_mixture": input[0], "mouth_embedding": input[1]}`（75-76）——
  与 `FalconSpeakerContrastive.forward(audio_mixture, mouth_embedding, **kwargs)` 的签名匹配。
- `[代码]` 输出（108-123）：`total_macs/total_params` 初始 0，只累加**音频网络**的结果，
  最后 `print("MACs: ", total_macs/1e9)`、`print("Params: ", total_params/1e6)`。
  **视频编码器的 MACs/参数完全没有被统计**；`video_model` 只用于生成 mouth_embedding。
- `[代码]` 未使用的代码：`check_parameters`（26-31）、大量被注释掉的 CPU 延迟测试/额外层统计
  （43-121）、`import pytorch_lightning/ModelCheckpoint/EarlyStopping/tqdm/DataLoader` 等（12-14）
  都未使用。
- `[验证]` 复刻该 ptflops 调用（CPU，`runs/04-macs.log`）：
  - 音频网络 `MACs = 10.611 G`、`params = 0.538 M` → README:206 的 `10.6 G` 与 README:34/103 的
    `0.5 M` 精确对应（README 取一位/整数）；
  - `FalconSpeakerContrastive` 537,697 + `FRCNNVideoModel` 11,186,688 = **11,724,385** →
    README:100 的 `11.72 M`；
  - 视觉编码器单独 ptflops：50 帧 64×64 = **7.77 G**、88×88 = **15.90 G**、96×96 = **17.48 G**
    → 若把视觉编码器计入，“10.6 G”会变成 ~26.5 G（88×88 口径），因此 README 的 MACs 必须是
    音频-only 口径。
  - `[验证]` R3 复跑 `verify_macs_04.py`（`runs/r3-macs.log`）再次得到 **10.611 G / 0.538 M** 与
    README 的 10.6 G/0.5 M 一致；这是本域对论文效率数字的关键对账证据。
- `[验证]` 数量级复核：`audio_bottleneck` 是 112→112 的 1×1 Conv2d，在 `[1,112,251,129]` 上约
  **0.41 G MACs**；全模型 10.6 G 由 separator 的多组 3×5/3×3 卷积（112 通道、251×129 特征图）
  与 4 次 LSTM 累积而成，量级合理（ptflops 逐层统计可用 `print_per_layer_stat=True` 复核）。
- `[代码]` `[推断]` “MACs” 的 ptflops 口径 = 乘加次数（multiply-accumulate），不含 BN/激活/上采样
  的额外运算；不同框架（thop/fvcore）默认口径略有差异，README 的数字应理解为 ptflops 口径。

---

## 3. 关键调用链与数据流

### 3.1 训练损失链（`[代码]`）

```
configs/Falcon.yml
 ├─ loss.train: {loss_func: PITLossWrapper, sdr_type: pairwise_neg_snr,
 │               config: {pit_from: pw_mtx, threshold_byloss: true}}   (configs/Falcon.yml:114-120)
 ├─ loss.val:   {..., sdr_type: pairwise_neg_sisdr, threshold_byloss: false}  (:121-126)
 └─ spk_contrast: {margin: .5, alpha_pos: 1, alpha_neg: 1, beta 0→1 over 200, weight: 1} (:103-111)
        │
audio_train.py:91-99   loss_func = {"train": PITLossWrapper(getattr(losses, "pairwise_neg_snr"),
        │                                **{"pit_from":"pw_mtx","threshold_byloss":True}), "val": ...}
audio_train.py:103-135 构造 AudioVisualLightningModuleNASSv2(..., margin/alpha/beta/weight)
        │
system/av_litmodule_nass_v2.py:95   pit_loss = self.loss_func["train"](est_sources[B,n_src,T], targets)
        │        └─ PITLossWrapper.forward → PairwiseNegSDR → [B,n_src,n_src] → find_best_perm
        │           → threshold(> -30 过滤) → mean → 标量负 SDR
        │
system/av_litmodule_nass_v2.py:134  cl_loss = NASSContrastiveLoss(q, p, negs, β(epoch))
        │
system/av_litmodule_nass_v2.py:137-138 total = pit_loss + 1.0 * cl_loss → train_loss
```

`audio_train.py:88-89` 的 `val` 配置用 `pairwise_neg_sisdr` + `threshold_byloss=false`，与
`av_litmodule.py:73,95,110,130` 的 validation_step 对应；`AudioVisualLightningModuleNASSv2` 复用基类
`validation_step`（`av_litmodule_nass_v2.py:36,73` 只覆写 training_step）。

### 3.2 视觉特征链（`[代码]` + `[验证]`）

```
mix.json/target.json (DataPreProcess/process_lrs23.py 生成; target.json 第 2 项是 mouth .pt 路径)
   ↓  AVSpeechEvalDataset.__getitem__  (avspeech_dataset.py:125-212)
      sf.read 整段音频（无 2 s 截断）→ [T]
      torch.load(mouth.pt)[T,96,96,1] → permute(0,3,1,2) → Grayscale → squeeze(1) → [T,96,96]
      （[推断] 4-D channel-last 格式由 permute 反推；仓库无生成脚本/资产，见 Q13）
      → get_preprocessing_pipelines()["val"] : Normalize(0,255) → CenterCrop(88,88) → Normalize(.421,.165)
      → mouth [T,88,88] float64
   ↓  audio_visual_test.py:230  torch.from_numpy(mouth[None,None]).float() → [1,1,T,88,88]
   ↓  FRCNNVideoModel.forward  (frcnn_videomodel.py:75-84)  → mouth_emb [1,512,T]  （T 帧 = 25 fps 帧数）
   ↓  FalconSpeakerContrastive.forward (models/Falcon_Contrastive.py:94-122)
      audio: STFTEncoder(2 s @16k) → [1,112,251,129]（2 s；1 s 时 126）
      fusion: NeuroSCNetFusion 把 video [1,512,T] 用 F.interpolate(size=T_a=251, mode="nearest")
              上采样/对齐到音频帧数（models/Falcon.py:239-264，具体 251-258）→ [1,512,251]
      → separator/mask/decoder → est_sources [1,n_src,T]
```

`[验证]` 用伪造数据跑通该链：`mouth_emb (1,512,25)`、`est_sources (1,1,16000)`，CPU 1.12 s
（`runs/04-eval-flow.log` §[9]）。
**“25 fps → 12.5 fps 时间下采样”不存在**：FRCNN 输出 T_out=T_in（§2.3.2）；融合层反而是
**上采样视频特征到音频帧率**（2 s：50 帧 → 251 帧，最近邻 ≈ ×5.02），见 §4.5。配置中的
`video_params.stride: 2`（`configs/Falcon.yml:77`）传给 `VideoBlock`（`models/Falcon.py:904,923`），
但 `VideoBlock.forward`（`models/Falcon.py:964-976`）**从不使用 `self.stride`** → 该键是死配置。

`[代码]` **评测是逐条整句**：`T` 可变（无 2 s 截断/补齐），`STFTEncoder` 帧数 `T'=1+floor(T/128)`，
融合层把视觉 F 帧 nearest 插值到 `T'` 个音频步（≈5.02 步/帧，比例随 T 变化）；`STFTDecoder` 用
`istft(length=input_shape[-1])` 精确还原，输出长度 == 输入 T（`Falcon.py:1166,1182-1188`）。
训练则固定 2 s/50 帧 → 评测/训练的长度分布不同（指标口径影响见 §4.6-8）。

### 3.3 评测链（`[代码]`）

```
__main__ (audio_visual_test.py:315-329)
  ├─ parse args: --conf_dir / --ckpt_path / --gpus (63-73)
  ├─ yaml.safe_load(conf_dir) → arg_dic["train_conf"] (319-322)
  └─ 若 --ckpt_path 以 hf:// 开头：hf_hub_download 一次 → 本地路径 (323-327)
main(arg_dic) (270-313)
  ├─ CUDA_VISIBLE_DEVICES = args.gpus; world_size = len(gpus.split(","))
  ├─ world_size>1 ? mp.spawn(process_worker, args=(world_size, config), nprocs=N) : process_worker(0,1)
  └─ 合并 results/metrics_rank*.csv → results/metrics_combined.csv → 打印 mean → 删 shard
process_worker(rank, world_size, config) (178-265)
  ├─ cuda.set_device(rank); device=cuda:rank
  ├─ AVSpeechDataModuleWithNear(**data_cfg, use_near=False); setup(); _, _, test_set = make_sets
  ├─ local_indices = range(len(test_set))[rank::world_size]
  ├─ exp_dir = cwd/Experiments/checkpoint/<exp_name>
  ├─ ckpt = config["ckpt_path"] or _find_best_ckpt(exp_dir)
  ├─ model, video_model = _load_models_from_ckpt(ckpt, config, device)   (111-172)
  ├─ metrics = ALLMetricsTracker(result_dir/metrics_rank{rank}.csv)
  └─ for idx in local_indices:
        mix, sources, mouth, key, _ = test_set[idx]           # 无 DataLoader
        mouth_emb = video_model(mouth[None,None])             # [1,512,T]
        est = model(mix[None], mouth_emb)                     # [1,1,T]
        metrics(mix, clean=sources[None], estimate=est[0], key)  # 写 CSV 一行
     metrics.final()                                          # 写 avg/std 行
```

`[代码]` 上图即完整评测链：没有 DataLoader/collate/batch；`AVSpeechEvalDataset` 的整句 item 直接
送模型；`AVSpeechDataModuleWithNear.setup()` 仍会构建 train/val/test 三个 dataset（`:277-295`）。

### 3.4 评测链上的已知断点（按“从点命令到出 CSV”顺序）

| # | 断点 | 证据 | 影响 |
|---|---|---|---|
| B1 | 默认 `--conf_dir` 指向不存在的实验目录 | `audio_visual_test.py:64-66` | 不传参必失败 |
| B2 | 自带 `conf.yml` 的 `audionet_name: SpeakerContrastiveV2` 不在 `look2hear.models` | `Experiments/checkpoint/Falcon/conf.yml:88` vs `look2hear/models/__init__.py:1-2` | `AttributeError`（P0） |
| B3 | `conf.yml` 的 `train/valid/test_dir` 是 `${DATA_ROOT}` 占位符；README:158 明确说明“脚本不展开环境变量”、要求用户手改 | `Experiments/checkpoint/Falcon/conf.yml:99-102`；README:158 | 开箱即用会打开 JSON 失败（属文档已说明的配置动作，非代码 bug） |
| B4 | 仓库无任何 `.ckpt`/`falcon.pt`；`best_k_models.json` 路径失效 | `Experiments/checkpoint/Falcon/` 只有 2 个文件；`best_k_models.json:2-6` | 不传 `--ckpt_path` 必 `FileNotFoundError`（P0） |
| B5 | ~~`_load_models_from_ckpt` 要求 `ckpt["state_dict"]`~~ **已关闭**：发布文件确实是 `{architecture, epoch, state_dict}`（R3 实测 falcon.pt/falcon-big.pt，525+137/517+137 键，真实函数加载成功） | `audio_visual_test.py:119-120`；`studio/falcon_model/PROVENANCE.md:36-40`；`runs/r3-weights.log` | 不是断点（仅对“裸 state_dict”自制文件不兼容，`04-eval-flow.log` §[4]） |
| B6 | 无 CUDA 时 `torch.cuda.set_device` 失败 | `audio_visual_test.py:179-180` | 纯 CPU 机器不能评测 |
| B7 | 评测**与训练期验证**都用 `AVSpeechEvalDataset`：单声道 wav 无条件 `.mean(-1)` → 0 维张量 → `mix[None]`=`(1,)` → `torch.stft` 抛 `RuntimeError`（不在 try 内，不捕获） | `look2hear/datas/avspeech_dataset.py:193-194`；`av_litmodule.py:86-119`；`docs/verification/data-pipe-supplement.md` §2；`runs/r3-mono-eval-repro.log` | **条件性 P0-3**：若数据为 mono，README:165/168 评测与训练第一次验证都在首个样本崩溃；若训练成功则反证数据为多声道（真实声道数 Q4） |
| B8 | “sdr/sdr_i” 列实为 −SNR/SNRi 且逐句/avg 符号相反 | `allwrapper.py:27,38-46,62,72` | 结果口径错误（P0，§4.6） |
| B9 | PESQ 实参顺序反 | `allwrapper.py:50` | PESQ 数值偏差 |
| B10 | 合并 CSV/打印均值把每 rank 的 avg/std 行也当样本平均 | `audio_visual_test.py:291-307` | 汇总均值偏差（P2，多卡时） |
| B11 | 训练落盘的 `conf.yml` 可能是**被模型构造污染后**的配置（`Separator` pop 掉 `repeats/shared` 后才 `yaml.safe_dump`）→ 评测重建模型退化为 49,081 参数且 `strict=False` 静默丢权重 | `audio_train.py:54-58`（建模）→ `:82-84`（dump）；`models/Falcon.py:991-996`；`runs/04-eval-flow.log` §[3] | 若用“训练产出的 conf.yml”评测 → 指标静默错误（P1，Q3） |

---

## 4. 关键实现细节与设计动机

### 4.1 为什么是“pairwise 负 SDR 矩阵 + PIT”

- `[代码]` `PairwiseNegSDR` 直接产出 `[B,est,target]` 矩阵，使 PIT 归约只依赖矩阵操作
  （`pit_wrapper.py:279-300`），对 `n_src<=3` 全排列、`n_src>3` 匈牙利，复杂度可控；
  这与 Asteroid 的经典设计一致（本文件是其裁剪版，`pipeline` 见 §8.3）。
- `[代码]` Falcon 配置 `n_src=1`（`configs/Falcon.yml:12`），PIT 实际退化为“取矩阵唯一元素”，
  但评测/损失代码保留了通用性（`metrics/splitwrapper.py` 甚至假设 3 源）。
- `[代码]` 训练用 **SNR**（`pairwise_neg_snr`）而验证用 **SI-SDR**（`pairwise_neg_sisdr`）：
  SNR 对估计尺度敏感，能给“幅度尚未校准”的早期模型更强梯度；SI-SDR 尺度不变，更适合当
  验证/早停指标。这是训练技巧而非论文指标定义；副作用是 `train_loss` 与 `val_loss` 数值不可直接比较
  （前者是 −SNR，后者是 −SI-SDR）。

### 4.2 `zero_mean` / `take_log` / `EPS` 的取值影响

- `[代码]` 训练/评测主路径都用 `zero_mean=True`（默认），即去掉每个信号的直流分量；对 16 kHz
  语音这通常无影响（语音本无强直流），但会改变静音/近零样本的 SDR。
- `[代码]` 只有死代码 `MetricsTracker` 用 `zero_mean=False`（`wrapper.py:30,33`），
  且它同时用 fast_bss_eval 的 filtered SDR（默认 `zero_mean=False`）→ 该 track 的口径与
  `ALLMetricsTracker` **不可比**。`[验证]` 对带 +5 直流偏置的同一输入：
  `ALLMetricsTracker`（zero_mean=True）si-snr=14.294 dB，`MetricsTracker`（zero_mean=False）
  si-snr=28.163 dB（`runs/04-metrics.log` §[3]）→ 差异巨大。
- `[代码]` `take_log=True` 使损失单位为 dB；`EPS` 只影响近零信号（`+1e-8`）。

### 4.3 `threshold_byloss` 的设计动机与真实行为

- 设计意图（从代码反推）：过滤“损失异常”的 batch 样本，避免极端值主导均值。但实现是**保留
  `min_loss > -30` 的样本**，也就是**丢掉 SDR > 30 dB 的样本**，只在存在“正常”样本时才过滤；
  全部都是“完美样本”时又退回全体均值。
- `[推断]` 对 Falcon（训练 SNR、验证 SI-SDR，数据集为混合语音，SDR 通常 0-20 dB）该阈值
  **几乎不会触发**；它的实际风险是**近完美样本/调试数据上训练损失会突变**（见 §2.1.3-D 的验证数字）。
  这是“复制来的模板代码”，与 Falcon 论文设计无关（P2）。

### 4.4 NASS 对比损失的设计动机（与论文/配置对应）

- **同空间（same-space）**：positive/negative 都由**同一个音频 encoder**（`encode_stft`）+
  **同一个 speaker pooling**（`pool_spk_embed`）得到（`Falcon_Contrastive.py:72-91`），
  避免“分离分支 embedding 与外部 speaker encoder 空间不对齐”。
- **query 带梯度、目标 detach**：只训练分离出的音频嵌入去靠近目标说话人、远离 near 说话人
  （`av_litmodule_nass_v2.py:102-133`）；目标侧走 `no_grad`，不会把梯度灌进目标重建分支。
- **hard-negative 加权 + β 退火**：`softmax(β·cos)` 在 β 增大时逐渐聚焦最相似的负样本；
  β 从 0（均匀）平滑到 1（`configs/Falcon.yml:108-110` + `_get_beta`）。
- **margin=0.5**：负样本 cos < 0.5 时 relu 截断，避免“已经足够不同”的负样本继续贡献梯度。
- **loss_weight=1.0**：与 PIT 损失等权相加（`av_litmodule_nass_v2.py:137-138`）；训练日志中
  `train_spk_contrast_loss` 是已乘权重的值（`weighted_cl`）。
- **负样本来源**：数据集的 `near_audios`（`avspeech_dataset_near.py:185-211`，由
  `DataPreProcess/generate_near_json.py` 生成的 `near.json` 索引）；每样本 K 可变，K=0 时为 None。
- `[推断]` 若某 batch 全部样本都没有负样本，损失只剩正样本项，此时对比损失仍会把 query 拉向目标
  说话人（这是“同空间”约束的一部分），但不再有“远离干扰”的约束。

### 4.5 FRCNN 设计：为什么“冻结 + 25 fps 不降采样”

- **冻结**：视觉编码器是 lip-reading 预训练模型（`configs/Falcon.yml:8` 的
  `pretrain_zoo/frcnn_128_512.backbone.pth.tar`），训练时冻结（`update_frcnn_parameter`
  第 116-117 行置 `requires_grad=False`），只训练 0.5 M 音频网络；`train()` 里再把 BN 置 eval
  （`frcnn_videomodel.py:90-95`）→ 用预训练 running stats，避免小 batch/域偏移破坏统计量。
- **25 fps 保留**：唇部视频是 25 fps，每帧 88×88 灰度；FRCNN 用 `(5,7,7)` 的 Conv3d 在时间维
  做 5 帧局部融合但 **stride=1**，MaxPool3d 时间维 **kernel=1/stride=1**，因此每个输出帧对应
  同一时刻的输入帧（`frcnn_videomodel.py:58-70`）——时间信息靠卷积核感受野而非下采样。
- **与音频对齐**：音频 STFT 帧率 = 16000/128 ≈ 125 fps（2 s → 251 帧），视觉 25 fps → 融合层
  `F.interpolate(..., mode="nearest")` 把 50 帧最近邻扩展到 251 帧（`models/Falcon.py:251-258`）。
  `[验证]` 50 帧输入 → 50 帧输出；2 s 音频的融合前音频帧为 251（Lead 日志）。
- **`video_params.stride: 2` 是死配置**：`VideoBlock` 保存 `self.stride`（`Falcon.py:923`）但
  `forward`（964-976）只由 gateway/projection/LSTM/residual_conv 组成，没有 stride 参数 → 该值
  对形状和计算量无影响。**“25fps→12.5fps”是误解**，不是本项目实现。
- **输入 88 而非 96**：96×96 是数据/接口层的嘴部 crop 尺寸；模型实际吃 88×88（训练随机裁剪、
  评测中心裁剪）。FRCNN 本身对尺寸无约束（adaptive pool），因为数据集先裁好了，所以论文/README
  的“96×96 mouth crops”（README:132）指的是**原始数据格式**，不是网络输入。

### 4.6 指标口径问题（本域最重要的风险）

**P0 编号约定（供跨文档引用）**：**P0-1** = README 评测命令不可开箱复现（B2 `audionet_name`、
B3 `${DATA_ROOT}`、B4 无 ckpt）；**P0-2** = `sdr/sdr_i` 实为 −SNR/SNRi 且逐句/avg 符号不一致；
**P0-3（条件性）** = 若真实数据为 mono，评测与训练期验证都在 `torch.stft` 崩溃。
**编号重排说明**：R2 讨论中的 “P0-3 = `ckpt["state_dict"]` 格式风险” 已经 R3 真实权重验证关闭
（§2.4/§8.2，Q1/B5 关闭）；本文件自 R3 起把 **P0-3 重新指代“条件性 mono 风险”**，清单为
P0-1（命令不可复现）/ P0-2（指标口径）/ P0-3（mono 条件性）。

1. **评测入口用的是 `ALLMetricsTracker`，不是 `MetricsTracker`**（`audio_visual_test.py:47,208`）。
2. `ALLMetricsTracker` 的 `sdr/sdr_i` 来自 `PairwiseNegSDR("snr")` → 是**尺度相关的 SNR**，
   不是 SDR/SDRi（§2.1.2-A 数值证据：把 estimate 放大 2 倍，“sdr”变化 +6.3 dB）。
3. `MetricsTracker`（`fast_bss_eval.sdr_pit_loss`，filtered SDR）才是真正的 SDR 口径，但**没有任何
   入口调用它**；`splitwrapper` 也无人调用。
4. 逐句行 `sdr/sdr_i` 是损失值（负号），`avg/std` 行取反 → **同一 CSV 列符号不一致**。
5. PESQ 实参顺序反（§2.2.2），STOI 正确。
6. `si-snr/si-snr_i` 列是正确的 SI-SDR/SI-SDRi（`PairwiseNegSDR("sisdr")`，行内取负）。
7. `[推断]` 论文表中的 SDRi 若由别的（未发布的）脚本/版本产生，则本仓库的评测入口**不能复现**
   该列；若论文表就是本入口产出，则 “SDRi”标签实际是 SNRi。这一点必须由 Lead/作者向仓库 owner
   确认（§7 Q5）。
8. **整句 vs 2 s 窗口口径**（R3 新增）：评测/验证数据集**整句读取**、无 2 s 截断/补齐（`avspeech_dataset.py:179-191`），
   输出长度严格 == 输入 T（`Falcon.py:1166,1182-1188`）；因此本入口的 SI-SDR/SDR 是**整句口径**，
   与论文/Studio 的“2 s 窗口”评估**不可直接横向比较**（05 §8.6；data-pipe supplement §3.2）。即使把
   P0-2 的口径修正为真 SDR，也需先统一窗长口径才能复现论文数字。
9. **训练分布 vs 评测分布**（`[推断]`）：训练只看 2 s 随机段（`segment=2.0`），评测整句长度分布不同；
   长句上模型的长度外推/边界行为未被训练协议覆盖，属于“能训练但未必能复现整句指标”的风险。

### 4.7 MACs 口径（README 数字如何产生）

- 10.6 G / 0.5 M = **音频网络**的一次 2 s 前向（50 帧视觉特征输入），ptflops 口径，**不含**冻结
  FRCNN（11.19 M / 15.9 G@88×88）。
- 11.72 M = 音频（537,697）+ FRCNN（11,186,688）；“Total params*” 的脚注（README:103）说明
  “Including the frozen 11.2 M-parameter FRCNN … paper counts trainable only (0.5 M)”，与代码一致。
- `[推断]` README 的“MACs 10.6 G”脚注写的是“trainable parameters only”，而 MACs 统计脚本实际
  只统计了 audio model → 两者口径一致（冻结 FRCNN 本来就不参与可训练 MACs 讨论）。
- 若要把视觉编码器算进效率，至少是 10.6+15.9≈26.5 G（88×88）/ 7.77 G（若按脚本的 64×64 输入），
  且延迟要重测；这是 README 表格里最容易误读的数字之一。

---

## 5. 对外接口契约

### 5.1 损失

| 组件 | 输入 | 输出 | dtype/device | 备注 |
|---|---|---|---|---|
| `PairwiseNegSDR` | `ests,targets: [B,S,T]` 同形状 | `[B,S_est,S_tgt]` 负 dB | float32/同 device | `sdr_type ∈ {snr,sisdr,sdsdr}`；`zero_mean`/`take_log`/`EPS` 可配 |
| `SingleSrcNegSDR` | `[B,T]` | `[B]`（或标量） | 同上 | `reduction ∈ {none,mean}`，`sum` 被禁止 |
| `MultiSrcNegSDR` | `[B,S,T]` | `[B]` | 同上 | 源维平均，无非排列 |
| `PITLossWrapper` | `ests [B,S,T]`（或 list）、`targets [B,S,T]` | 标量（`return_ests=True` 时 `(标量,[B,S,T])`） | 同上 | 6 种 `pit_from`，见 §2.1.3 |
| `new_PITLossWrapper` | `ests, ests_bn, targets, epoch` | 原始损失输出（不做 PIT） | 同上 | 实验性，不要用 |
| `MixITLossWrapper` | `ests [B,nsrc,T]`、`mix [B,nmix,T]` | 标量（或 `(标量,[B,nmix,T])`） | 同上 | 仓库未使用 |
| `NASSContrastiveLoss` | `q,p: [B,D]`（L2 归一化）、`negs: list[B] of [K_b,D]/None`、`beta: float` | 标量 | 同 q | 内部不 detach；K=1 无 softmax；平均分母恒为 B |
| `SISNRi` | `mix/est/ref [B,T]` | `[B]`（`sisnr`）/标量（`compute_loss`） | — | **类不可实例化**（NameError），死代码 |

### 5.2 指标与 CSV

| 组件 | `__call__(mix, clean, estimate, key)` 形状约定 | CSV 列 | 最终行 |
|---|---|---|---|
| `MetricsTracker` | `mix [T]`、`clean/estimate [S,T]` | `snt_id,sdr,sdr_i,si-snr,si-snr_i` | avg、std |
| `ALLMetricsTracker` | 同上（`clean` 传入 `[1,S,T]` 亦可） | 上述 + `pesq,stoi` | avg、std |
| `SPlitMetricsTracker` | 要求 `S=3` | `one_snr,one_snr_i,one_si-snr,one_si-snr_i,two_snr,...,two_si-snr_i` | avg |

- `[代码]` `mix` 会按 `clean.shape[0]`（源数）复制成 `[S,T]` 再当基线（`wrapper.py:39`、
  `allwrapper.py:33`、`splitwrapper.py:53`）。
- `[代码]` 单位：dB；`si-snr_i` = 估计 SI-SDR − 混合 SI-SDR；`sdr_i` 同理（但 allwrapper 的实为 SNRi）。
- `[代码]` `key` 通常为 `xxx.wav` 文件名（`avspeech_dataset.py:212`），CSV `snt_id` 列直接用它。
- `[代码]` **口径提示**：评测/验证是整句推理（`avspeech_dataset.py:179-191`），输出长度==输入 T；
  SI-SDR/SDR 是整句口径，与论文/Studio 的 2 s 窗口评估不可直接比较（§4.6-8）。
- `[代码]` `ALLMetricsTracker` 的 PESQ/STOI 实参见 §2.2.2；只有 `n_src=1` 时可用。

### 5.3 视觉编码器

| 项 | 契约 |
|---|---|
| 输入 | `[B,1,T,H,W]`，灰度已归一化（0-255→/255→−0.421/0.165），`H=W=88`（评测/训练实际），`T` 为 25 fps 帧数 |
| 输出 | `[B,512,T]`，与输入帧一一对应（时间不降采样） |
| 参数 | 11,186,688（resnet 分支）；shufflenet 分支 1.26 M、输出 1024 维 |
| 预训练 | `torch.load(path)["model_state_dict"]`，加载后全部冻结；`tcn.*` 键被跳过；缺文件直接 `FileNotFoundError` |
| BN | `train()` 时强制 eval 模式；`eval()` 返回值是 `None`（勿链式赋值） |
| 尺寸 | FRCNN 内部不裁剪；96/88 均可；shufflenet 分支要求空间下采样后恰好 3×3（88/96 可，64 会崩） |

### 5.4 评测入口 CLI 与产物

| 项 | 值 |
|---|---|
| CLI | `python audio_visual_test.py --conf_dir <yml> [--ckpt_path <path 或 hf://org/repo/file>] [--gpus "0,1"]` |
| 必需配置键 | `train_conf.audionet.{audionet_name,audionet_config}`、`train_conf.videonet.{videonet_name,videonet_config}`、`train_conf.datamodule.data_config.*`、`train_conf.exp.exp_name` |
| 依赖目录 | `Experiments/checkpoint/<exp_name>/`（ckpt）、`<exp_name>/results/`（输出）；`setup()` 要求 train/valid/test 三目录均有 `mix.json`+`target.json`（`avspeech_dataset_near.py:277-295`） |
| 输出 | `results/metrics_rank{rank}.csv`（临时）→ `results/metrics_combined.csv`（最终，含每 rank 的 avg/std 行） |
| 硬件 | CUDA 必须（`torch.cuda.set_device`）；HF 下载在 spawn 前完成 |
| 入口脚本 | `evaluated_mac_params.py` 无 CLI（`--exp_dir` 定义了但未用），硬编码读 `configs/Falcon.yml`；需要 `pretrain_zoo/*.pth.tar` 与 CUDA |

---

## 6. 可验证断言表

> 说明：`[验证]` 列给出脚本/日志（§8）。置信度：高/中/低。

| # | 断言 | 证据 path:line | 验证方式 | 置信度 |
|---|---|---|---|---|
| A01 | `pairwise_neg_{sisdr,sdsdr,snr}` 是 `PairwiseNegSDR` 的模块级单例实例 | `look2hear/losses/matrix.py:372-380` | 读码 + import 测试 | 高 |
| A02 | `PairwiseNegSDR` 输出 `[B,est,tgt]` 的 `−10log10(Σproj²/(Σnoise²+EPS))`，三种 sdr_type 分别对应 SI-SDR/SD-SDR/SNR | `matrix.py:188-232` | `[验证]` 手算对拍 max diff=0 | 高 |
| A03 | `zero_mean=True` 使 SDR 对直流偏置不变；False 时显著变化 | `matrix.py:204-208` | `[验证]` 加 DC 后 True 不变、False 差 47.8 dB | 高 |
| A04 | SNR 不尺度不变、SI-SDR 尺度不变 | `matrix.py:212-232` | `[验证]` est×2：SNR −4.0 dB，SI-SDR 0 dB | 高 |
| A05 | `SingleSrcNegSDR` 输入 `[B,T]`、支持 none/mean、拒绝 sum | `matrix.py:235-281` | `[验证]` | 高 |
| A06 | `SingleSrcNegSDR`/`MultiSrcNegSDR` 的 `EPS` 构造参数被忽略（写死 1e-8） | `matrix.py:246,292` | 读码 | 高 |
| A07 | `MultiSrcNegSDR` 是同索引逐源平均，不做排列匹配 | `matrix.py:294-326` | `[验证]` 手算 diff=0 | 高 |
| A08 | `PITLossWrapper` 支持 6 种 `pit_from`；`pw_mtx` 即 `loss_func(ests,targets)` 后 PIT | `pit_wrapper.py:162-195` | 读码 + `[验证]` | 高 |
| A09 | `pw_mtx`/`pw_pt`/`perm_avg` 三种模式在同一输入上结果一致（对匹配的 loss 类型） | `pit_wrapper.py:170-193` | `[验证]` 22.393810 三路一致 | 高 |
| A10 | `pw_mtx_broadcast` = PIT + 0.5×第二损失 | `pit_wrapper.py:233-234` | `[验证]` 33.590714 手算一致 | 高 |
| A11 | `threshold_byloss=True` 只保留 `min_loss > -30` 的样本（丢弃 SDR>30dB 的样本）；全被过滤时退回全体 | `pit_wrapper.py:227-230` | `[验证]` -6.194741 vs -36.891914 | 高 |
| A12 | `n_src<=3` 穷举全排列、`n_src>3` 用 scipy 匈牙利 | `pit_wrapper.py:266-274,303-312` | 读码 | 高 |
| A13 | `return_ests=True` 返回按最佳排列重排后的估计 | `pit_wrapper.py:231,260-264` | `[验证]` 与手算排列一致 | 高 |
| A14 | `new_PITLossWrapper` 与 `PairwiseNegSDR` 签名不兼容、且 pw_mtx 分支不做 PIT | `pit_wrapper.py:6-75` | `[验证]` TypeError / 原样返回矩阵 | 高 |
| A15 | `MixITLossWrapper` 对 4 估计/2 混合生成 14 个有序二划分并取最小损失 | `mixit.py:220-284` | `[验证]` 17.777866 手算一致 | 高 |
| A16 | `MixITLossWrapper` 仓库内无调用者（死代码） | 全仓库 grep；`mixit.py:96-309` | grep | 高 |
| A17 | NASS 公式 = `α_pos(1−cos(q,p)) + α_neg Σ softmax(β·cos(q,n))·relu(cos−m)`（K>1）；K=1 时无 softmax | `nass_contrastive_loss.py:52-77` | `[验证]` β=0/1/5 手算对拍 | 高 |
| A18 | NASS 平均分母恒为 B（无负样本样本也计数） | `nass_contrastive_loss.py:48-50,77,79-81` | 读码 + `[验证]` | 高 |
| A19 | NASS 内部不 detach；目标侧 no_grad 由 system 负责 | `nass_contrastive_loss.py:31-81`；`av_litmodule_nass_v2.py:106-133` | `[验证]` 梯度探测 | 高 |
| A20 | β 调度 = `init+(final−init)·min(epoch/anneal,1)`；Falcon 配置 0→1/200 epochs | `av_litmodule_nass_v2.py:66-70`；`configs/Falcon.yml:108-110` | 读码 + 算术 | 高 |
| A21 | 训练损失 = `pit_loss + loss_weight·cl_loss`，`train_spk_contrast_loss` 已乘权重 | `av_litmodule_nass_v2.py:95,134-138,175` | 读码 | 高 |
| A22 | `SISNRi()` 因 `super(Loss, self)` 抛 `NameError`，类不可用（死代码） | `sisnri.py:6` | `[验证]` | 高 |
| A23 | `audio_visual_test.py` 用 `ALLMetricsTracker`，其 `sdr/sdr_i` 来自 `PairwiseNegSDR("snr")` | `audio_visual_test.py:47,208`；`allwrapper.py:27,38-46` | `[验证]` 数值与 loss 值相同 | 高 |
| A24 | `ALLMetricsTracker` 逐句 `sdr/sdr_i` 与 `avg/std` 行符号相反 | `allwrapper.py:62,72` | `[验证]` 表格 | 高 |
| A25 | `ALLMetricsTracker` 的 PESQ 实参顺序与 pypesq 合同相反 | `allwrapper.py:50`；pypesq 源码 `pesq(ref,deg,fs)` | `[验证]` 调用记录 + 外部源码 | 高 |
| A26 | STOI 实参顺序正确 | `allwrapper.py:53-58` | `[验证]` | 高 |
| A27 | `MetricsTracker` 的 SDR 用 `fast_bss_eval.sdr_pit_loss(est, ref)`（返回负 filtered SDR），且 `zero_mean=False` | `wrapper.py:30,33,44-45` | 外部源码核对（§8.3） | 高 |
| A28 | `MetricsTracker`/`SPlitMetricsTracker` 无任何入口调用（死代码） | 全仓库 grep | grep | 高 |
| A29 | `FRCNNVideoModel` resnet 分支参数 11,186,688；输出 `[B,512,T]`；时间步不变 | `frcnn_videomodel.py:28-84` | `[验证]` | 高 |
| A30 | `FRCNNVideoModel.train()` 不返回 self → `.eval()` 返回 None | `frcnn_videomodel.py:90-95` | `[验证]` | 高 |
| A31 | 加载 pretrain 后所有参数 `requires_grad=False`；`tcn.*` 键被跳过 | `frcnn_videomodel.py:106-118` | `[验证]` | 高 |
| A32 | 96→88 裁剪发生在数据集预处理（train 随机/val 中心）与 Studio `[4:92]`，不在 FRCNN 内 | `transform.py:147-168`；`avspeech_dataset.py:138,189`；`falcon_runtime.py:95-104` | `[验证]` mouth 形状 + 手工对拍 | 高 |
| A33 | 25 fps → 12.5 fps 下采样**不存在**；融合层把 50 帧上采样到 251 帧 | `frcnn_videomodel.py:58-70`；`Falcon.py:239-264,964-976` | `[验证]` T_out=T_in + Lead 日志 | 高 |
| A34 | 只有 `FRCNNVideoModel` 被任一 config 选中；其余变体均死代码 | `configs/Falcon.yml:2-8`、`studio/configs/*.yml:3/166`；grep | grep + 读码 | 高 |
| A35 | shufflenet 分支 64×64 输入会崩（globalpool AvgPool2d(3)），88/96 正常 | `shufflenetv2.py:160`；`frcnn_videomodel.py:43-51` | `[验证]` R3 复跑 `runs/r3-video.log`（评审人未复算，本轮复现同一 RuntimeError） | 高 |
| A36 | `_find_best_ckpt` 顺序 = best_k(json,min) → last.ckpt → sorted(epoch)[-1] | `audio_visual_test.py:78-108` | `[验证]` 假目录 | 高 |
| A37 | 自带 `best_k_models.json` 路径为 `${ROOT}` 占位符、仓库无 ckpt → 默认评测抛 FileNotFoundError | `best_k_models.json:2-6`；`audio_visual_test.py:78-108` | `[验证]` | 高 |
| A38 | `epoch=1000.ckpt` 会因字符串排序排在 `epoch=742.ckpt` 之前被选为“最新” | `audio_visual_test.py:103-106` | `[验证]` | 高 |
| A39 | `_load_models_from_ckpt` 要求顶层 `state_dict` 键；音频 strict=False、视频 strict=True；视频 pretrain 被置 None | `audio_visual_test.py:119-120,139,147-160` | `[验证]` 伪造 ckpt 回环 | 高 |
| A40 | 自带 conf.yml 的 `audionet_name=SpeakerContrastiveV2` 在 `look2hear.models` 不存在 → AttributeError | `Experiments/checkpoint/Falcon/conf.yml:88`；`look2hear/models/__init__.py:1-2` | `[验证]` | 高 |
| A41 | `Separator.__init__` 的 `dict.pop("repeats"/"shared")` 会污染传入 config；污染后重建模型参数从 537,697 降到 49,081 | `models/Falcon.py:991-996`；`audio_train.py:54-58` | `[验证]` | 高 |
| A42 | `process_worker` 不用 DataLoader，直接按 `rank::world_size` 切片索引测试集，batch=1 | `audio_visual_test.py:189-193,228-229,249-250` | 读码 + `[验证]` 复刻循环 | 高 |
| A43 | 评测数据读取整段音频（不截 2 s），嘴部 25 fps | `avspeech_dataset.py:179,189-194` | `[验证]` 1 s 样本返回 16000 点/25 帧 | 高 |
| A44 | 单声道 wav → 评测集与**训练期验证**都退化为 0 维张量 → `torch.stft` 抛 `RuntimeError`（评测入口不捕获）；训练集有 ndim 保护、评测集没有 → 条件性 P0-3 | `avspeech_dataset.py:193-194` vs `avspeech_dataset_near.py:176-177`；`Falcon.py:1095-1097,1110-1116`；`data-pipe-supplement.md` §2 | `[验证]` R3 独立重跑 `runs/r3-mono-eval-repro.log`（mono→RuntimeError、stereo→(1,1,40000)） | 高（真实数据声道数待 Q4） |
| A45 | 结果路径 = `cwd/Experiments/checkpoint/<exp_name>/results/metrics_combined.csv`；每 rank 先写 `metrics_rank{r}.csv` 再合并删除 | `audio_visual_test.py:196-208,284-313` | 读码 + `[验证]` 路径拼接 | 高 |
| A46 | 合并 CSV/打印均值会把每 rank 的 avg/std 行当普通样本再次平均 | `audio_visual_test.py:291-307` | 读码 | 高 |
| A47 | `evaluated_mac_params.py` 用 ptflops 只统计 audio model，MACs=**10.611 G**、params=**0.538 M** | `evaluated_mac_params.py:21,86-96,122-123` | `[验证]` R3 复跑 `runs/r3-macs.log`（10.611 G 与 README 10.6 G 对齐） | 高 |
| A48 | README 11.72 M = 537,697+11,186,688；0.5 M/10.6 G 为 audio-only 口径 | `README.md:100,103,206` | `[验证]` | 高 |
| A49 | 视觉编码器 MACs 未计入 10.6 G（实测：64²=7.77 G、88²=15.90 G、96²=17.48 G） | `evaluated_mac_params.py:77-96` | `[验证]` R3 复跑 `runs/r3-macs.log` | 高 |
| A50 | `evaluated_mac_params.py` 保留 config 中的 `pretrain` 路径 → 无 `pretrain_zoo` 时无法运行；且要求 CUDA | `evaluated_mac_params.py:48,51-53` | 读码 + `[验证]` FileNotFoundError | 高 |
| A51 | 发布 `falcon.pt` 顶层 = dict `{architecture:'FalconSpeakerContrastive', epoch:714, state_dict}`，含 525 个 `audio_model.*` + 137 个 `video_model.*`；key 集合/形状与本地模型完全一致，strict=True 可加载；真实 `audio_visual_test._load_models_from_ckpt` 路径成功 → Q1/B5 关闭 | `runs/r3-weights.log`；`studio/falcon_model/PROVENANCE.md:36-40`；`audio_visual_test.py:111-172` | `[验证]`（R3 真实权重 torch.load + 函数级回环） | 高 |
| A52 | 发布 `falcon-big.pt` 顶层 = dict `{architecture:'SpeakerContrastiveV2', epoch:715, state_dict}`，含 517+137 keys；与 `studio` 的 `SpeakerContrastiveV2` + `FRCNNVideoModel` key/形状完全一致，strict 加载成功；Big audio 参数 1,158,617，总 12,345,305≈12.35 M（README:101） | `runs/r3-weights.log`；`studio/configs/FalconBig.yml`；`studio/falcon_model/models/Speaker_Contrastive_v2.py` | `[验证]`（R3，构建 studio 模型 + 回环加载） | 高 |
| A53 | mono wav → `AVSpeechEvalDataset` item 退化为 0 维 → `mix[None]=(1,)` → `torch.stft` 抛 `RuntimeError`（不在 try 内）；训练期 val 用同一 dataset → 同样在首个验证 batch 崩溃 → 条件性 P0-3 | `avspeech_dataset.py:193-194`；`Falcon.py:1095-1097,1110-1116`；`audio_visual_test.py:231-242`；`av_litmodule.py:86-119`；`data-pipe-supplement.md` §2 | `[验证]` R3 独立重跑 data-pipe 脚本（`runs/r3-mono-eval-repro.log`） | 高 |
| A54 | 评测/验证整句读取、无 2 s 截断/补齐；输出长度严格 == 输入 T（`istft(length=input_shape[-1])`）→ SI-SDR/SDR 为整句口径，与论文/Studio 的 2 s 窗口评估不可直接比较 | `avspeech_dataset.py:179-191`；`Falcon.py:1166,1182-1188`；05 §8.6 | `[验证]`（data-pipe：T=48513,F=75→out 48513；本域 §4.6-8/9 讨论） | 高 |
| A55 | mouth 输出 dtype 取决于 `.pt` 存储 dtype：uint8→float64、float32→float32；下游评测 `.float()`、训练 `type_as` 统一转 float32，不影响 FRCNN | `transform.py:147-168`；`avspeech_dataset.py:189-192`；`audio_visual_test.py:230`；`av_litmodule_nass_v2.py:87` | `[验证]`（本域 04-eval-flow §[8] 的 uint8 fixture + data-pipe 05 A5/D3 的 float32 实测） | 高 |

（共 55 条；≥30 条验收要求满足。）

---

## 7. 未解疑问 / 需其它子系统确认的问题

| Q | 问题 | 证据/现状 | 需谁确认 |
|---|---|---|---|
| Q1（**已关闭**） | 发布权重格式：R3 用真实权重实测 `falcon.pt`/`falcon-big.pt` 顶层均为 `{architecture, epoch, state_dict}`；falcon.pt 525+137 keys、architecture=FalconSpeakerContrastive/epoch=714，与本地模型 key+shape 完全一致，`strict=True` 加载 + 真实 `_load_models_from_ckpt` 均成功（`PROVENANCE.md:36-40` 有同样声明）。**结论：ckpt 格式不是 README:168 的阻断点；残留断点 = B2（conf.yml 类名）+ B3（${DATA_ROOT}）** | `runs/r3-weights.log`；`studio/falcon_model/PROVENANCE.md:36-40`；`audio_visual_test.py:119-120` | 已由 loss-metric-eval R3 关闭（Lead 下载的真实权重） |
| Q2 | `Experiments/checkpoint/Falcon/conf.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 是笔误还是来自另一条代码线？`look2hear.models` 只有 `Falcon`/`FalconSpeakerContrastive`，README 命令会 `AttributeError` | `conf.yml:88` vs `look2hear/models/__init__.py:1-2`；`[验证]` | ops-tests-docs / train-sys（是否应改成 `FalconSpeakerContrastive`） |
| Q3 | 真实训练产生的 `conf.yml` 是否也缺 `audio_params.repeats/shared`（`Separator` 的 `dict.pop` 污染）？建模后 `audio_train.py:82-84` 的 `yaml.safe_dump` 才会落盘 → 真实 conf 可能是污染版。若缺键，`_load_models_from_ckpt` 重建 49,081 参数的空 Separator + `strict=False` 静默丢权重（B11） | `[验证]` 模型参数 537,697→49,081；`models/Falcon.py:991-996`；`audio_train.py:54-58`（建模）→ `:82-84`（dump）；`runs/04-eval-flow.log` §[3] | train-sys（R2 评审 03 时已交叉；日志 `runs/train-sys-config-mutation.log`） |
| Q4 | **真实数据 wav 的通道数**（后果已坐实，仅剩数据侧确认）：若 mono → 评测与训练期 val 都在 `torch.stft` 崩溃（条件性 P0-3，`data-pipe-supplement.md` §2 + `runs/r3-mono-eval-repro.log`）；若论文模型确实训练成功，则反证真实训练/评测数据应为多声道（README:67 的 “mono 16 kHz” 表述需核实） | `avspeech_dataset.py:193-194`；`avspeech_dataset_near.py:176-177`；`data-pipe-supplement.md` §2；05 §8.6 | data-pipe（数据格式确认；05 Q1 同问） |
| Q5 | 论文表中 SDRi 是哪条路径算的？当前评测入口的 `sdr/sdr_i` 是 −SNR/SNRi；只有死代码 `MetricsTracker` 用 `fast_bss_eval` 真 SDR。若论文数字来自本入口，则结果表的“SDRi”标签错误 | `allwrapper.py:27,38-46`；`wrapper.py:29-34,44-45`；README:174,178-191 | Lead / model-core / 作者；R2 可与 train-sys 的 03 文档交叉 |
| Q6 | `fast_bss_eval` 的具体版本与 `sdr_pit_loss` 的 `filter_length` 默认值是否就是 512？本仓库 requirements 未 pin 版本（`requirements.txt:3` 注释说未记录版本） | `requirements.txt:1-5,24`；外部 0.1.4 源码 | ops-tests-docs（若作者环境有 freeze） |
| Q7 | `pretrain_zoo/frcnn_128_512.backbone.pth.tar` 的 key 集合是否与当前 `FRCNNVideoModel` 完全一致（否则 `load_state_dict` strict 会失败）？仓库未提供该文件 | `frcnn_videomodel.py:106-118`；README:107,158 | model-core / ops（有网时核对下载页/文件） |
| Q8 | `evaluated_mac_params.py` 用 64×64 视觉输入（第 81 行），与真实 88×88 不同；作者是否有意为之（如只为生成 mouth_emb，不参与 MACs）？ | `evaluated_mac_params.py:80-82`；`[验证]` 64²/88² 视频 MACs 差 2× | ops-tests-docs（复算 README 数字时需口径说明） |
| Q9 | `mask_generator`/`separator` 中的 `video_params.stride=2` 是否有历史意图（早期 VideoBlock 可能做过时间下采样）？当前实现完全未用；若未来实现 stride，融合层最近邻上采样比例会变化 | `models/Falcon.py:904,923,964-976`；`configs/Falcon.yml:77` | model-core（R2 评审 01 时确认） |
| Q10 | `ALLMetricsTracker` 的 PESQ/STOI 是否在论文/README 结果中被使用？README 只报告 DNSMOS（外部工具）与 SI-SNRi/SDRi，PESQ/STOI 列可能是额外产物 | `allwrapper.py:50-58`；README:178-191 | Lead / 作者 |
| Q11 | `_find_best_ckpt` 的 `min(best_k, key=best_k.get)` 依赖 PL 把原始 val_loss 写入 `best_k_models`；不同 PL 版本是否一致？（本环境 PL 2.6.6 已验证；训练时 PL 版本未 pin） | `audio_visual_test.py:84-93`；`audio_train.py:184`；PL 2.6.6 源码 | train-sys（R2/R3 与 03 文档交叉） |
| Q12 | `audio_visual_test.py` 的 `--gpus` 多卡路径在 `mp.spawn` 下 config dict 含 numpy/yaml 对象是否可 pickle？（从代码看是纯 dict/str/float/list，`[推断]` 可） | `audio_visual_test.py:280`；config 来自 yaml.safe_load | ops-tests-docs（有 GPU 环境跑 `--gpus 0,1` 冒烟） |
| Q13 | 真实 mouth `.pt` 的**张量形状与存储 dtype**：数据管线按 `permute(0,3,1,2)` 反推为 `[T,96,96,1]`（channel-last），但仓库内无生成脚本/资产；dtype 可为 uint8/float32。两者决定 dataset 输出形状与 dtype（A55） | `avspeech_dataset.py:189-192`；`transform.py:147-168`；05 Q2 | data-pipe（与 05 Q2 同问） |

---

## 8. 本轮验证活动记录

### 8.1 环境与命令

- 解释器：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128），按 `docs/understanding/VERIFY-ENV.md`
  注入 stub：`PYTHONPATH=$PWD/docs/verification/stubs:$PWD`。
- 全部脚本在 `docs/verification/scripts/`，日志在 `docs/verification/runs/`（共享目录，可复核）：

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_losses_04.py    > docs/verification/runs/04-losses.log
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_metrics_04.py   > docs/verification/runs/04-metrics.log
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_video_04.py     > docs/verification/runs/04-video.log
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_eval_flow_04.py > docs/verification/runs/04-eval-flow.log
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_macs_04.py      > docs/verification/runs/04-macs.log
```

**R3 新增（真实权重 + R2 评审复跑）**：

```bash
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_r3_weights_04.py > docs/verification/runs/r3-weights.log   # Q1/B5/A51/A52
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_video_04.py      > docs/verification/runs/r3-video.log     # A35 复跑
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  docs/verification/scripts/verify_macs_04.py       > docs/verification/runs/r3-macs.log      # A47/A49 复跑
PYTHONPATH=/tmp/dp-verify/stubs:$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python \
  /tmp/dp-verify/verify_p0_eval.py > docs/verification/runs/r3-mono-eval-repro.log            # mono P0 独立复跑
```

### 8.2 结果摘要（逐项）

1. **损失数值对拍**（`04-losses.log`）：三种 `sdr_type` 与手算 `max|diff|=0`；`zero_mean` 行为；
   `PITLossWrapper` 五条分支与手算一致（22.393810 / 33.590714 等）；`threshold_byloss` 三组对照；
   `MixITLossWrapper` 14 划分手算一致（17.777866）；`NASSContrastiveLoss` β∈{0,1,5} 手算差 ≤6e-8；
   `SISNRi()` → `NameError`（数学部分绕过构造后对拍 3.8e-6）；`new_PITLossWrapper` 签名不兼容
   + 不归约；按 `audio_train.py:91-99` 的方式从 `configs/Falcon.yml` 实例化成功
   （train: `PITLossWrapper/PairwiseNegSDR(snr)/threshold=True`；val: `sisdr/threshold=False`）。
2. **指标逻辑**（`04-metrics.log`）：用 stub 替换未安装的 `fast_bss_eval/torch_mir_eval/pypesq/pystoi`
   （stub 签名按真实包源码，见 §8.3），验证 CSV 列、avg/std 行、逐句 `sdr` = `PairwiseNegSDR('snr')`
   损失值、逐句/avg 符号相反、`est→2·est` 时“sdr”变化而 SI-SDR 不变、PESQ 第 1 实参是 estimate、
   STOI 第 1 实参是 clean、`zero_mean` 口径差异（14.294 vs 28.163 dB）。
3. **视觉模型**（`04-video.log`）：FRCNN resnet 11,186,688 参数；5 组输入形状全部输出 `[B,512,T]`
   且 `T_out=T_in`；`train()` 无返回值 → `.eval()` 返回 None；冻结逻辑；pretrain 缺失报错；
   变体参数/输出：shufflenet-FRCNN 1.259 M/(1,1024,50)、ResNetVideoModel 11.187 M、LightVideomodel
   1.259 M、WujianVideoModel 11.318 M/(1,256,50)；FRCNN 内部不裁剪（96/88 均可）；shufflenet 64×64 崩。
4. **评测流程**（`04-eval-flow.log`）：自带目录 `FileNotFoundError`（mock `_find_best_ckpt`）；
   best_k json 选 min、last.ckpt 回退、epoch 字符串排序 bug；conf.yml 类名 `SpeakerContrastiveV2`
   不存在；模型构建污染 config（537,697→49,081）；伪造 Lightning ckpt 回环（525 audio keys + 137
   video keys，strict 加载一致）；裸 state_dict → `KeyError`；shipped conf.yml 触发 `AttributeError`；
   data_cfg 键与 `AVSpeechDataModuleWithNear` 签名匹配、但 `${DATA_ROOT}` 未展开；hf:// 解析；
   结果路径；`AVSpeechEvalDataset` 伪造样本（mouth (25,88,88)，中心裁剪对拍 2.57e-7，整段音频 1 s）；
   单声道探针（mix 0 维）；复刻 process_worker 单样本前向：`mouth_emb (1,512,25)`、
   `est_sources (1,1,16000)`、CPU 1.12 s、metrics CSV 列正确。
5. **参数量/MACs**（`04-macs.log`）：audio 0.538 M / 10.611 G；FRCNN 11.187 M（64²=7.77 G、
   88²=15.90 G、96²=17.48 G）；总参数 11,724,385；与 README 三处数字对账通过。R3 复跑 `runs/r3-macs.log`
   再次得到 **10.611 G / 0.538 M**（评审 A47/A49 的 ❓ 由此关闭）。
6. **真实权重（R3，`runs/r3-weights.log`）**：`falcon.pt` = `{architecture:'FalconSpeakerContrastive',
   epoch:714, state_dict}`，525 audio + 137 video keys，key/shape 与本地模型**逐项一致**、strict=True
   加载成功；`audio_visual_test._load_models_from_ckpt` 真实函数路径成功（537,697/11,186,688 参数）。
   `falcon-big.pt` = `{architecture:'SpeakerContrastiveV2', epoch:715, state_dict}`，517+137 keys，与
   studio `SpeakerContrastiveV2`+FRCNN 完全一致、strict 加载成功；Big audio=1,158,617，总计 12,345,305
   ≈12.35 M（README:101）。→ **Q1/B5 关闭**。
7. **mono P0 独立复跑（R3，`runs/r3-mono-eval-repro.log`）**：mono fixture → `mixture.shape=()`、
   `mix[None]=(1,)` → 真实模型 `torch.stft` `RuntimeError`（stereo 正常 `(1,1,40000)`）；训练期 val
   batch（`default_collate`）同样 RuntimeError → 条件性 P0-3 成立（评审 P1-2；引用 supplement §2/05 §8.6）。

### 8.3 外部依赖核对（来源：PyPI sdist/wheel，未安装）

本环境未安装 `fast_bss_eval`/`pypesq`/`pystoi`/`torch_mir_eval`，为确认调用方向/参数顺序，
下载了以下**小体积源码包**（仅读源码，未安装、未下载模型权重）：

| 包 | 版本 | 用途 | 结论 |
|---|---|---|---|
| `fast-bss-eval` | 0.1.4（33 KB sdist） | `sdr_pit_loss(est, ref)` 的签名与符号 | 内部 `return sdr(ref, est, change_sign=True)` → **返回负 SDR**（loss）；默认 `filter_length=512`（卷积滤波 SDR） |
| `pypesq` | 1.2.4（31 KB sdist） | `pesq` 参数顺序 | `def pesq(ref, deg, fs=16000, normalize=False)` → **第 1 个实参应为 clean**；`allwrapper.py:50` 传反 |
| `asteroid` | 0.6.0（wheel） | PIT wrapper 对照（本仓库是其裁剪版） | 原版 `PITLossWrapper` **没有** `threshold_byloss`/multidecoder 分支 → 这些是本仓库自行加的；`patched` 行为以本仓库代码 + 本次数值验证为准 |

> 注：外部源码 URL 见 `pypi.org/pypi/{fast-bss-eval/0.1.4,pypesq/1.2.4,asteroid/0.6.0}/json`；
> raw.githubusercontent/ jsdelivr 在本沙箱超时，故用 PyPI 文件（`files.pythonhosted.org`）核对。

**仓库内补充证据（R3 采纳 data-pipe 评审）**：
- `studio/falcon_model/PROVENANCE.md:36-40`：发布 `falcon.pt`/`falcon-big.pt` “hold `state_dict`,
  the architecture name and the epoch”，并给出参数 11,724,385、与训练代码 CPU 输出 bit-identical —— 与
  R3 真实权重检查（A51/A52）一致。
- `docs/verification/data-pipe-supplement.md` §2：mono-eval P0 的真实模型复现（本域 R3 独立重跑，日志
  `runs/r3-mono-eval-repro.log`）；§3 固化了评测入口“无 DataLoader/整句/无 collate”契约。
- `docs/understanding/05-data-pipeline.md` §8.6：同一 P0 与整句口径的数据域结论（04/05 现已一致）。

### 8.4 失败/限制

- **真实权重已具备（R3）**：`/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`（Lead 从 hf-mirror
  下载）→ 发布格式/加载路径已 `[验证]`（`runs/r3-weights.log`），Q1 关闭；但**无 `pretrain_zoo/`、无
  LRS2/LRS3 数据** → 端到端训练/评测仍未做；真实 conf.yml 生成方式（Q3）与真实数据声道数（Q4）待确认。
- `fast_bss_eval`/`pypesq`/`pystoi`/`torch_mir_eval` 未安装 → 指标数值用 stub，只验证**逻辑/符号/
  参数顺序**，不验证 PESQ/SDR 的具体数值。
- 未使用 GPU（评测脚本强制 CUDA；本域复刻在 CPU 上完成前向），未测多卡 `mp.spawn`（Q12）。
- 未修改任何源码；新增文件仅为 `docs/verification/scripts/{verify_losses_04.py,verify_metrics_04.py,
  verify_video_04.py,verify_eval_flow_04.py,verify_macs_04.py,verify_r3_weights_04.py}` 与本文件；
  R3 新增日志：`runs/r3-weights.log`、`runs/r3-video.log`、`runs/r3-macs.log`、`runs/r3-mono-eval-repro.log`。

---

## 9. 评审响应与变更日志

### 9.1 对 R2 评审（`docs/reviews/round1/04-losses-metrics-video-review-by-data-pipe.md`）的逐条响应

R2 评审结论：P0=0 / P1=6 / P2=8，另 3 条 ❓（A35/A47/A49 本域外未复算）。**本文件 R3 全部采纳**，
逐条响应如下（“位置”指本文件 R3 版）。

| 评审项 | 结论 | 本文件处理 | 位置/证据 |
|---|---|---|---|
| P1-1（Q1/B5，`falcon.pt` 格式） | 采纳 | Q1 关闭为“已由仓库文档+真实权重解答”；B5 标注已关闭；§2.4-E 行改为“ckpt 格式 ✅ / 仍因 B2/B3 失败”；新增 A51/A52 | §2.4-B、§2.4-E、§3.4-B5、§7-Q1、§8.2-6、A51/A52；`runs/r3-weights.log`；`PROVENANCE.md:36-40` |
| P1-2（mono 后果低估） | 采纳并升级 | B7 升级为**条件性 P0-3**；Q4 改为“仅剩数据侧确认通道数”；A44/A53 重写；补“训练期 val 同 dataset 同样崩” | §3.4-B7、§4.6-P0 编号、§7-Q4、A44/A53、§8.2-7；`data-pipe-supplement.md` §2、05 §8.6、`runs/r3-mono-eval-repro.log` |
| P1-3（dtype=float64 缺前提） | 采纳 | 改为“dtype 取决于 `.pt` 存储类型（uint8→float64、float32→float32），下游统一 float32”；新增 A55 | §2.3.2、A55、§7-Q13 |
| P1-4（行号 166-172 错） | 采纳 | 改为 `avspeech_dataset_near.py:167-174`（管线调用与切片 169-174） | §2.3.2 裁剪表 |
| P1-5（`tensors_to_device` 返回类型） | 采纳 | 改为“对 tuple 返回**长度 5 的 list**（`:35-36`），再解包”；引用范围改 `:19-44` | §2.4-C:6、§5/P2-8 |
| P1-6（遗漏整句 vs 2 s 口径） | 采纳 | §4.6 新增第 8/9 条（整句口径、训练/评测长度分布不同）；§3.2/§5.2 补说明；新增 A54 | §3.2、§4.6-8/9、§5.2、A54 |
| P2-1（`:170` 不能“体现返回 None”） | 采纳 | 改为“`:169-170` 以语句形式调用（未链式赋值），规避了陷阱” | §2.3.2 |
| P2-2（K=1 softmax 措辞） | 采纳 | 改为“K=1 省去 softmax 计算；softmax 恒为 1，与公式等价，不是语义差异” | §2.1.5 |
| P2-3（`make_sets[2]`） | 采纳 | 改为 `_, _, test_set = datamodule.make_sets`（189） | §2.4-C:3 |
| P2-4（setup 行号 277-293） | 采纳 | 改为 `:277-295` | §2.4-C:2、§5.4 |
| P2-5（mouth `.pt` 4-D 格式） | 采纳 | §3.2 标 `[推断]`；新增 Q13（与 05 Q2 对齐） | §3.2、§7-Q13 |
| P2-6（表格混用 ✅/❌） | 采纳 | 拆为“逻辑 ✅；数据 ❌（仓库无 ckpt）” | §2.4-E 行 174 |
| P2-7（P0 未编号） | 采纳 | §2.4 结论与 §4.6 头部加入 **P0-1/P0-2/P0-3** 编号约定与重排说明 | §2.4、§4.6 |
| P2-8（`torch_utils.py:19-42`） | 采纳 | 改为 `:19-44` | §2.4-C:6 |
| ❓ A35（shufflenet 64 崩） | 关闭 | R3 复跑 `verify_video_04.py` 同一 RuntimeError，回填验证列 | A35；`runs/r3-video.log` |
| ❓ A47/A49（MACs） | 关闭 | R3 复跑 `verify_macs_04.py`：**10.611 G / 0.538 M**、视觉 7.77/15.90/17.48 G，回填验证列 | A47/A49；`runs/r3-macs.log` |
| 遗漏-1 训练期 val 也受 mono 影响 | 采纳 | 写入 B7/Q4/A53 与 §4.6 | §3.4-B7、§7-Q4、A53 |
| 遗漏-2 评测读相邻 `_meta.json` | 采纳 | §2.4-C 补充段：每样本读 meta 统计 n_near（评测丢弃、缺失静默 0、多一次 IO） | §2.4-C 补充 |
| 遗漏-3 评测不过滤 <2 s | 采纳 | §2.4-C 补充段：训练过滤 <32000、评测全量 → 样本集合不同 | §2.4-C 补充 |
| 遗漏-4 无截断/补齐、输出长度==T | 采纳 | §3.2 补 `T'=1+floor(T/128)`、`istft(length=)`；§4.6-8 | §3.2、§4.6-8、A54 |
| 遗漏-5 `use_near=False` 完整行为 | 采纳 | §5.4 依赖目录行：setup 需 train/valid/test 三目录的 mix+target json；near.json 不参与评测 | §5.4、§2.4-C 补充 |
| 遗漏-6 评测不创建 val/test dataloader | 采纳 | §3.3 代码块后补注：全程无 DataLoader/collate；val/test dataloader 不被调用 | §3.3 |
| 跨模块-1（04↔05） | 采纳 | mouth 88、dtype 条件化、行号 169-174 已统一；引 05 §8.6/§5.4 | §2.3.2、§4.6、Q4/Q13 |
| 跨模块-2（04↔06） | 采纳 | 引 `PROVENANCE.md:36-40` 关闭 Q1；Studio 权重格式与本入口一致 | §2.4-B、§7-Q1 |
| 跨模块-3（04↔03 config 污染） | 采纳 | B11/Q3 补上 `audio_train.py:54-58`（建模）→ `:82-84`（dump）完整链路 | §3.4-B11、§7-Q3 |
| 跨模块-4/5（04↔01/02） | 采纳 | stride 死配置、88 裁剪归属维持原结论并引源码；未新增矛盾 | §2.3.2、§3.2、§4.5 |
| 跨模块-6（04↔README 原因归属） | 采纳 | §2.4-E 行 165/168 明确主因是 B2+B3（+B4），不是 ckpt 格式 | §2.4-E |
| 跨模块-7（04↔05 DataPreProcess） | 部分采纳 | 数据准备细节归 05 §2.6/§5.9，本文件不展开，仅在 §5.4 注明数据目录契约 | §5.4 |
| 问题清单 1（Q1/B5） | 采纳 | 见 P1-1 行 | §7-Q1 等 |
| 问题清单 2（mono） | 采纳 | 见 P1-2 行 | §3.4-B7 等 |
| 问题清单 3（dtype） | 采纳 | 见 P1-3 行 | §2.3.2、A55 |
| 问题清单 4（整句口径） | 采纳 | 见 P1-6 行 | §4.6-8/9、A54 |
| 问题清单 5（行号/类型） | 采纳 | 见 P1-4/P1-5/P2-3/P2-4/P2-8 行 | 各位置 |
| 问题清单 6（mouth `.pt`） | 采纳 | 见 P2-5 行 | §3.2、Q13 |

### 9.2 变更日志（Changelog）

| 日期 | 轮次 | 改动 | 采纳的评审意见 |
|---|---|---|---|
| 2026-10-21 | R1 初稿 | 新建：8 节模板 + 50 条断言 + Q1-Q12 + 5 个验证脚本/日志；覆盖 losses/metrics/videomodels/audio_visual_test/evaluated_mac_params 全部范围 | — |
| 2026-10-21 | R3 修订 | ① Q1/B5 关闭：新增真实权重 [验证]（falcon.pt 525+137 keys/strict 加载/`_load_models_from_ckpt` 实跑；falcon-big.pt 517+137 keys 与 studio 模型一致；Big 总参数 12.35 M）；② mono 升级为条件性 P0-3 并补“训练期 val 同 dataset”与独立复跑；③ dtype 条件化（uint8→float64/float32→float32）；④ 补“评测整句 vs 训练 2 s”指标口径（§4.6-8/9、§3.2、§5.2）与训练/评测样本集合差异；⑤ 修正行号/类型：`near.py:167-174`、`torch_utils.py:19-44` 且返回 list、`make_sets` 解包、setup `:277-295`；⑥ P2-1..P2-8 全部修正；⑦ A35/A47/A49 用 R3 复跑日志回填、新增 A51-A55（共 55 条）；⑧ §7 Q1 关闭、Q3/Q4 改写、新增 Q13；⑨ §8 增补 R3 验证活动与仓库内证据（PROVENANCE/data-pipe-supplement/05 §8.6）；⑩ 明确 P0-1/P0-2/P0-3 编号并说明原 P0-3 已关闭重排 | 采纳 data-pipe R2 评审全部意见（P0=0/P1=6/P2=8；3 条 ❓ 用 R3 复跑关闭） |
