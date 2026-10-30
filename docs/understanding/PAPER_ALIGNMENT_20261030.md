# 论文公开后对齐：arXiv 2610.32749 协议提取与复现差异校正（2026-10-30）

> 任务：共享任务 `task-14`（owner `losses-metrics`）。**只写本文件**；未改源码/他人文档，未 git commit。
> 论文：Wendi Sang, Kai Li, Yifan Li, Jianqiang Huang, Xiaolin Hu,
> *"Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes"*,
> **arXiv:2610.32749v1 [cs.SD]，2026-10-23 提交，28 页**（NeurIPS 2026）。
> 阅读来源：`https://arxiv.org/html/2610.32749v1`（HTTP 200；全文 99,124 B 已落盘
> `/tmp/dsh-spill-b7byK2/session-113276a90dfe/0f0263a00387-web_fetch.txt`，636 行）。
> 仓库基线：`repro/phase2`（已 merge upstream `689555c→439cbe9`）HEAD `d9ec1e8`。
> 证据标记：`[论文]` 论文正文/表格/附录（给章节/表号）；`[代码]` 仓库 file:line；`[我们]` 既有复现文档/结果；
> `[推断]` 由论文+代码推断；`[待PDF]` HTML 抽取有 MathML 重影、需 PDF 核对。
> 上游同步内容（本次 diff `689555c..439cbe9`）：README/项目页加 arXiv 链接、HF 下载计数、`audio_visual_test.py +5`
> 行（仅下载计数）、`studio/falcon_runtime.py` 计数线程——**不改变任何评测数值/协议**。

---

## 0 结论速览（论文对 6 个旧问题的回答）

| # | 旧问题 | 论文答案 | 对我们旧结论的判定 |
|---|---|---|---|
| 1 | 是否每个 benchmark 分别训练？ | **部分明确**：baseline 在 RealSSA-Sim 与 VoxCeleb2-2Mix 上**分别重训**（§6.1）；Table 6 明确存在“RealSSA-Sim-trained Falcon”与“VoxCeleb2-trained Falcon”两版。LRS2/LRS3 baseline 数字直接引用 Swift-Net。**Falcon 在 LRS2/LRS3 列的训练集/ckpt 论文未明说** | 🟡 部分确认（“按数据集训练”成立；但不是论文逐字声明的 per-benchmark 全矩阵） |
| 2 | 训练损失/配方 | `L_recon = −SI-SNR`、λ_spk=1.0、m=0.5、**τ 0→1 线性退火（Table 16：前 1000 epoch）**；AdamW lr 1e-3/wd 0.01；ReduceLROnPlateau(15, 0.5)；early stop 30；**8×RTX3090、bs 4/GPU、1000 epoch**；2 s/50 帧/25 fps；seed 42（§5.2, App D Table 16） | ❌ **推翻**我们“训练用 SNR 是论文技巧”的推断：论文白纸黑字 **−SI-SNR**；仓库 `configs/Falcon.yml:117` 的 `pairwise_neg_snr` 与 τ=200 都**不是论文最终配方** |
| 3 | 对比损失负样本怎么来（RealSSA 无 near 波形） | 负样本 = **每个 near-field 干扰源的 clean per-source 波形**，与 target 同 RIR、同 `E_a` 编码、stop-grad；**far-field 被排除**（§4.3 Eq.8-10）；RealSSA-Sim 的 near 波形由 LibriSpeech 干扰源+RIR 仿真生成，**发布包不含 near 波形，但论文承诺发布生成脚本与 metadata（含 RIR hash/位置/SIR）可重生成**（App A.2）；Limitations 也承认该损失依赖仿真 near 源 | 🟡 **确认并填上**：负样本语义与我们的 NASS 公式一致；我们“RealSSA 无 near → K=0”仍是**数据发布现状**，但**可用论文 pipeline 重生成**（需 LibriSpeech+LRS3+WHAM!+Hive） |
| 4 | DNSMOS 3.276 是 P.835 OVRL 还是 P.808？ | 论文引用 **[35] = Reddy et al., “DNSMOS P.835” (ICASSP 2022)**，不是 P.808；Table 3 单列一个 DNSMOS 数值 3.276（§5.2, App E）；**具体子分数（SIG/BAK/OVRL）未写明**，按惯例应为 **OVRL** `[推断]` | ❌ **推翻**我们 P2-E“P808 3.3285≈3.276”的猜法：P.808 不是论文指标；我们同 ckpt 的 OVRL 2.7901 与论文 3.276 **不可比**（论文 3.276 是 **RealSSA-Sim-trained** ckpt，见 §6.1/Table 6） |
| 5 | RealSSA-Sim 生成参数 | 四场景（small meeting / medium indoor / large public / outdoor），RT60 0.3-0.6 / 0.5-0.9 / 1.0-2.0 / 0.05-0.1 s；near 干扰 1-5（各场景 1-5/1-3/1-4/1-4），far 10-15；near SIR 2-5/2-8/2-8/2-15 dB；far SIR 20-35 dB；env SIR 5-15/5-10/2-8/−2-5 dB；**split 100,826 / 9,133 / 9,169，129.2/11.7/11.7 h，avg 4.61 s、avg 13.2 speakers、near SIR 8.5 dB、far SIR 27.5 dB**；**评测只用 channel-1**（§3, Table 2/11, App A.1-A.3） | ✅ **确认** test=9,169 与我们的数据一致；❗**发布数据卡写 4-channel，论文写“released waveforms are channel-1 streams”**——见 §5 的口径冲突；我们取 4ch **mean** 是**偏离论文**的 |
| 6 | 消融 / MDCL / 窗口化 | TFRB 是架构块（Eq.5），**GCI 是架构块（Eq.6）但没有独立消融**；消融只有：stage 数（1/2/3→7.1/7.4/7.7）、递归单元（LSTM vs RNN/GRU/SRU）、对比损失（+0.4/+0.2/+0.3 dB）、τ 调度、输出参数化、overlap、missing visual。**全文无 MDCL、无窗口化/流式评测数字** | ✅ **确认**“MDCL 不在论文”；❌ **无法**用论文解释 Studio 窗口化 −1.9~−2.3 dB（论文只有 2 s clip 评测协议） |

**一句话**：论文把“训练配方 / DNSMOS 口径 / RealSSA 生成参数 / 消融”四块空缺全部填上了；但它**没有**说明
Falcon 在 LRS2/LRS3 各列的训练集、也**没有**说明发布 `falcon.pt` 是哪一个 ckpt——这两点仍只能靠仓库证据（`conf.yml`
快照 = LRS3_complex_nonmoving）推断。我们此前“14.1/14.7 缺 LRS2-trained ckpt”的判断被论文结构**支持**
（Table 6 证明训练数据会显著改变 RealSSA-Real 指标），但“论文按 benchmark 分别训练”只被**部分**证实。

---

## 1 阅读范围与可信度

| 章节 | 是否读到 | 备注 |
|---|---|---|
| Abstract / §1 Intro / §2 Related / §3 RealSSA | ✅ 全文 | §3 的样本量/小时数在 HTML 中 MathML 重影，已用 App A.3 Table 13 校正 |
| §4 Method（4.1 管线 / 4.2 Separator+TFRB+GCI / 4.3 对比损失） | ✅ 全文 | 公式 Eq.1-10 已提取；图 2/3/4 仅文字/图注，未解析像素 |
| §5 Experiments（5.1 数据集 / 5.2 配置与指标） | ✅ 全文 | 关键数字有重影，已用 App C/D 表校正 |
| §6 Results（6.1 主表 / 6.2 效率 / 6.3 泛化 / 6.4 消融） | ✅ 全文 | 表 3/4/5/6/7/8/9 数值已读 |
| §7 Conclusions / Limitations | ✅ 全文 | 明确“对比损失依赖仿真 near 源” |
| App A（数据集构造）/ B（预处理）/ C（宽度）/ D（训练）/ E（指标与延迟） | ✅ 全文 | 本文最重要的协议来源 |
| App F（baseline 因果化）/ G（分场景）/ H（定性）/ I（递归单元）/ J（embedding 可视化）/ K（输出参数化）/ L（overlap）/ M（缺失视觉）/ N（LLM 声明） | ✅ 文本；❌ 图片内容 | H/J 的图只看图注 |
| Algorithm 1（mixing 伪代码） | ⚠️ 仅正文 A.2 文字版 | HTML 中伪代码为图，正文已给等价描述 |
| 数字重影清单 `[待PDF]` | — | RealSSA-Real 参与者人数（“8484 participants (1010 male and 55 female)”）、Table 16 τ 行措辞、§6.2 叙述中的百分比顺序；Table 16/5/6 的表值本身可交叉校正 |

> 抽取方法：`web_fetch https://arxiv.org/html/2610.32749v1`（HTTP 200），全文落盘后按 `grep`/分段读取；
> arXiv HTML 的 MathML 会导致“数字+alt 文本”重复（如 `9,1699{,}169`=9,169、`10001000`=1000），本文引用时已去重，
> 凡不能唯一确定的标 `[待PDF]`。

---

## 2 训练协议（§5.2 + App D Table 16）与仓库对照

### 2.1 论文配方（Table 16 原文）

| 项 | 论文值 | 仓库对照 `[代码]` | 判定 |
|---|---|---|---|
| 优化器 | AdamW，lr 1e-3，wd 0.01 | `configs/Falcon.yml:143-147`（adamw/1e-3/0.01） | ✅ 一致 |
| scheduler | ReduceLROnPlateau，factor 0.5，patience 15 | `configs/Falcon.yml:149-152` | ✅ |
| early stop | patience 30，monitor val loss | `configs/Falcon.yml:136-141`（patience 30） | ✅ |
| max epochs | 1000 | `configs/Falcon.yml:135` | ✅ |
| batch | **4 / GPU，8×RTX3090**（global 32） | `configs/Falcon.yml:166` batch_size 4；但 `configs/LRS2_2Mix_Paper_8gpu.yml:146` 是 **5**（8×5=40） | ⚠️ 原始配置一致；**我们自己的 8 卡配方偏离论文** |
| 段长/帧 | 2 s @16 kHz；25 fps；T_v=50；mouth 96×96 灰度 | `segment: 2.0`、`sample_rate: 16000`、videonet config | ✅ |
| STFT | window **256**、hop **128**（16 ms/8 ms），Hann；输入 real+imag+**magnitude** 3 通道 | `configs/Falcon.yml:28-29`（win 256/hop 128）；`look2hear/models/Falcon.py:1118` `torch.stack([spec.real, spec.imag, this_spec], 1)` | ✅ 一致（含 3 通道） |
| 视觉骨干 | FRCNN（同 CTCNet），lipreading 预训练、全程冻结 | `videonet: FRCNNVideoModel`、`pretrain_zoo/...`；`update_frcnn_parameter` 冻结 | ✅ |
| 重建损失 | **`L_recon = −SI-SNR(ŝ, s)`** | **`loss.train.sdr_type: pairwise_neg_snr`**（`configs/Falcon.yml:117`）→ 负 **SNR** | ❌ **不一致（关键）** |
| 对比权重 | λ_spk = 1.0 | `spk_contrast.loss_weight: 1.0`（`:111`） | ✅ |
| margin | m = 0.5 | `spk_contrast.margin: 0.5`（`:105`） | ✅ |
| **τ（β）调度** | **0→1 线性退火，Table 16：前 1000 epoch，之后固定 1** | `beta_anneal_epochs: 200`（`configs/Falcon.yml:110`；发布 run 快照 `Experiments/checkpoint/Falcon/conf.yml:137` 也是 200） | ❌ **不一致（5× 差异）** |
| seed | 42 | 仓库配置未显式记录 | ⚠️ |

`[论文]` App D 原文（去重后）：“The contrastive temperature was linearly annealed from τ=0 to 1 over the first
**1000** training epochs and held at 1 thereafter... The reconstruction loss uses the **negative SI-SNR** formulation...
λ_spk=1.0.”

**影响**：仓库 `configs/Falcon.yml` 与发布权重的 `conf.yml` 都是 **SNR + β/200**；论文最终模型是 **SI-SNR + τ/1000**。
所以发布 `falcon.pt` 的训练目标**不是论文 Table 16 的目标**（至少损失与退火不同），这与我们既有推断
“发布 ckpt 是旧配方”一致，但此前我们以为论文未公开、无法确证——现在可以确证。

### 2.2 是否每个 benchmark 分别训练？

- `[论文]` §6.1：“We retrained each baseline separately on **RealSSA-Sim and VoxCeleb2-2Mix**. Within each dataset,
  all models used the same splits/preprocessing/optimizer/schedule... **Baseline results on LRS2-2Mix and LRS3-2Mix
  are from Swift-Net [17]**.”
- `[论文]` §6.1 下一段：“We then evaluated the models trained on RealSSA-Sim on RealSSA-Real.” → Table 3 的
  RealSSA-Real DNSMOS、RealSSA-Sim SI-SNRi/SDRi 来自 **RealSSA-Sim-trained Falcon**。
- `[论文]` Table 6：“model trained on VoxCeleb2” vs “trained on RealSSA-Sim” 在同一 RealSSA-Real 上对比
  （Falcon DNSMOS 2.878 vs 3.276）→ 至少存在 **两个 Falcon ckpt**。
- `[论文]` **没有**一句话说明 Falcon 的 LRS2-2Mix / LRS3-2Mix / VoxCeleb2-2Mix 列分别用哪个训练集/ckpt。
- `[推断]` 强：Table 3 中 Falcon 在 LRS2-2Mix 14.1/14.7、LRS3-2Mix 16.4/16.7 只能来自**各自训练集**训练的 ckpt
  （跨语料不可能同时 SOTA 且比 Swift-Net 高）；VoxCeleb2 列对应 Table 6 的 VoxCeleb2-trained 模型。
  但这仍**不是论文显式声明**，建议向作者确认。

`[我们]` 旧结论“README:228 'under our training protocol' ⇒ 论文按 benchmark 分别训练”现在可以改写为：
论文明确重训 baseline 的只有 RealSSA-Sim/VoxCeleb2；Falcon 的 per-benchmark 训练是**结构上必需但文字未声明**。
`[我们]` “发布 `falcon.pt` = LRS3_complex_nonmoving 训练”依旧只有仓库侧证据（`Experiments/checkpoint/Falcon/conf.yml:99-102`），
**论文全文未提发布权重及其训练集**（只给 code+demo 链接）。

### 2.3 对比损失与负样本（§4.3 Eq.8-10）

`[论文]` 公式与仓库实现逐项一致：

```
query  q = P_spk(Â)                         # Â：masked encoder-space embedding（带梯度）
pos    p = P_spk(sg(E_a(s)))                # clean target，同 RIR，stop-grad     (Eq.8)
neg    e_k = P_spk(sg(E_a(b_k))), k=1..K    # 每个 near-field 干扰源，同 RIR       (Eq.8)
L_spk = (1 − cos(q,p)) + Σ_k softmax_k(τ·s_k) · ReLU(s_k − m),  s_k = cos(q, e_k)   (Eq.10)
m = 0.5，τ: 0→1 线性退火（Table 16），λ_spk = 1.0
```

- `[论文]` **far-field 干扰被排除**：原文“Far-field interferers are excluded from the negative set: both distance
  attenuation and ... late reverberation drive their residual below the SI-SNR error floor”。
- `[论文]` K = 该样本的 near-field 干扰源个数 → RealSSA-Sim 平均 2.5 个（§3），最大 5；**只有 K≥2 时才用到 τ 的软最大**。
- `[代码]` 仓库 `look2hear/losses/nass_contrastive_loss.py:52-77` 与 Eq.10 同构（正项 `:57-58`；K=1 直接 relu `:68-69`；
  K≥2 `softmax(beta·sim)` `:71-72`；K=0 置零 `:73-74`；分母恒为 B `:77`）。
- `[代码]` 训练侧 `av_litmodule_nass_v2.py:102-134`：query = `sep_audio_embs[:,0]` 经 `pool_spk_embed`；positive =
  `targets[:,0]` 经 `encode_stft`+`pool_spk_embed`（no_grad）；negatives = `near_audios[b]` 逐个编码（no_grad）——
  **与论文“同 encoder、目标侧 stop-grad”一致**。
- `[代码]` 差异：仓库 NASS 在 K=1 时跳过 softmax（论文 Eq.10 在 K=1 时 softmax 也恒为 1，等价）；仓库 β 名
  `beta` 即论文 τ；仓库 `alpha_pos/alpha_neg` 固定 1.0（论文无这两个权重）。
- `[推断]` K=1（LRS2-2Mix 用 s2 当 near）时 **τ 退火失效**这一我们此前的结论仍然成立——论文的 τ 消融
  （Table 9）必须在 **K≥2** 的 RealSSA-Sim 上做才有意义，正好解释为什么论文能做 τ 消融而我们 LRS2 复现做不了。

**RealSSA 的 near 波形从哪来（旧问题核心）**：
- `[论文]` App A.1/A.2：target 来自 LRS3，near/far 干扰来自 **LibriSpeech**；每个样本随机房间几何+RT60，
  gpuRIR 生成 RIR，target 归一化到 −23 LUFS，near 按场景 SIR 叠加；**metadata 记录房间几何、RT60、源位置、
  RIR hash、SIR 值**；“We will release the generation scripts and the metadata together with the dataset”。
- `[论文]` §7 Limitations：“The contrastive loss currently relies on clean per-source waveforms of near-field
  interferers available in simulation”。
- `[我们]` 已下载的 RealSSA-Sim 包只有 mix/target/mouth（4ch wav），`near.json` 只能生成 mix 哨兵
  （`tools/repro/realssa_to_falcon.py:482-489`），所以我们的 RealSSA 训练若直接开跑会 **K=0**（只剩正项）。
- `[推断]` **可在本地重生成的路径**：拿到 released metadata + 生成脚本 + LibriSpeech/LRS3/WHAM!/Hive 后，
  按记录的 RIR/位置/SIR 重放 mixing，即可得到每个 near 源的 clean 波形（对齐到 segment 后作为负样本）。
  这是论文公开后**新解锁**的 RealSSA 训练复现路径（此前我们判定“无法忠实复现负样本”）。

---

## 3 评测协议（§5.2 + App E）

| 项 | 论文口径 | 我们的对照 | 判定 |
|---|---|---|---|
| 分离指标 | **SI-SNRi、SDRi**，均相对 unprocessed mixture（§6, App E；引用 [34] Vincent 2006 = BSS-Eval） | `look2hear/metrics/core.py`：SI-SDR(去均值+PIT)/SDR=`-fast_bss_eval.sdr_pit_loss(filter_length=512)` | ✅ 同族；⚠️ 论文未写 SDR 的 filter length/实现库，**无法逐位对齐** |
| SI-SNR 定义 | 引用 [18] Le Roux et al. “SDR half-baked or well done?”（SI-SDR 去均值、尺度不变） | `core.si_sdr`（zero_mean=True） | ✅ |
| 评测片段 | 公开 benchmark：**固定 2 s clip**（短补零、长随机裁；训练）；评测用官方 test split（App B） | LRS2/RealSSA-Sim 用自己的转换协议（整段/对齐后长度） | ⚠️ RealSSA-Sim test 平均 4.61 s，论文**未写**评测是整段还是 2 s 窗口 `[待确认]` |
| RealSSA-Sim 通道 | **只用 channel-1**（App A.2：“all experiments use the first-channel signal only... released waveforms are channel-1 streams”） | `realssa_to_falcon.py` 默认 **mean**（4ch 平均），我们全部 RealSSA-Sim 结果都是 mean | ❌ **口径偏离**（应改 `--channel 0` 复评） |
| RealSSA-Real 通道 | 论文未写（iPhone 17 Pro 录制，A.4） | 我们用 `channel=mean` | `[待确认]` |
| DNSMOS | **[35] = DNSMOS P.835**（ICASSP 2022）；App E 只写“DNSMOS [35]” | 我们 speechmos P.835+P.808，报 SIG/BAK/OVRL/P808 | ✅ 同模型家族；❌ 论文**未写子分数**（按惯例 OVRL `[推断]`） |
| NISQA | [36] NISQA（Interspeech 2021） | 未做 | ➕ 可补 |
| WER | **Mega-ASR [37]**（arXiv 2605.19833），RealSSA-Sim 全部 + RealSSA-Real 54 段（App E） | 未做 | ➕ 可补（模型可得性待查） |
| target-speaker similarity | **WeSpeaker ResNet34-LM**（HF: Wespeaker/wespeaker-voxceleb-resnet34-LM），与独立 enrollment utterance 的 cosine×100 | 未做 | ➕ 可补（checkpoint 公开） |
| 评测对象 | RealSSA-Real 的指标用 **RealSSA-Sim-trained** 模型（§6.1） | 我们的 RealSSA-Real DNSMOS 用的发布 `falcon.pt`（仓库快照=LRS3 训练） | ❌ **不同 ckpt，不可直接比** |

**DNSMOS 口径的最终答案（旧问题 4）**：
- `[论文]` 引用 [35] “DNSMOS p. 835”（Reddy/Gopal/Cutler, ICASSP 2022），**不是 P.808**；Table 3 单值 3.276。
- `[论文]` Table 6：RealSSA-Sim-trained Falcon **3.276**，VoxCeleb2-trained Falcon **2.878** —— 训练数据差 0.4 MOS。
- `[我们]` 发布 `falcon.pt`（非 RealSSA-trained）在 RealSSA-Real（channel=mean、窗口化推理）上：
  **SIG 3.2209 / BAK 3.6921 / OVRL 2.7901 / P808 3.3285**（`docs/REPRODUCTION_PHASE2_REPORT.md:98-107`）。
- `[推断]` 论文 3.276 最可能是 **P.835 OVRL**（单值惯例）；我们的 OVRL 2.79 与论文 3.276 差 0.49，
  与 Table 6 的“非 RealSSA 训练版本 2.878”同量级（−0.09）→ **差距主要来自 ckpt 训练数据，而非 DNSMOS 实现**。
  我们此前用 P808 3.3285 对齐 3.276 是**巧合**（P.808 不是论文指标）→ **旧结论推翻**。
- `[待确认]` 论文未写 DNSMOS 实现/版本（speechmos？DNS-Challenge 官方脚本？）与子分数；我们只能报告
  “speechmos 0.0.1.1 P.835 OVRL + 模型 sha256”并声明实现不同。

---

## 4 RealSSA-Sim 生成参数（§3 + Table 2/11 + App A.1-A.3）

### 4.1 场景与混合参数（合并 Table 2 与 Table 11）

| 场景 | 房间尺寸 | RT60 | near 数 / 距离 | far 数 / 距离 | near SIR | env SIR | far SIR | 突发噪声 |
|---|---|---|---|---|---|---|---|---|
| Small meeting | 4-6×3-5×2.5-3 m | 0.3–0.6 s | 1–5 / 0.5–1.5 m | 无 | 2–5 dB | 5–15 dB | – | 30%，1 次 |
| Medium indoor | 8-15×8-12×3-4 m | 0.5–0.9 s | 1–3 / 1.0–2.5 m | 10–15 / 3–15 m | 2–8 dB | 5–10 dB | 20–35 dB | 40%，1–2 次 |
| Large public | 20-40×15-30×4-8 m | 1.0–2.0 s | 1–4 / 1.0–3.0 m | 10–15 / 5–20 m | 2–8 dB | 2–8 dB | 20–35 dB | 50%，1–3 次 |
| Outdoor | 50×50×20 m（墙吸收 0.99） | 0.05–0.1 s | 1–4 / 1.0–4.0 m | 10–15 / 5–25 m | 2–15 dB | −2–5 dB | 20–35 dB | 40%，1–3 次 |

`[论文]` 近场阈值 **3 m**（临界距离近似）；四场景均匀采样；平均 near 干扰 **2.5** 个。

### 4.2 数据来源与三条流水线（App A.1/A.2）

- target：**LRS3**（唯一带对齐嘴部视频的源，~438 h / ~9,545 speakers）；
- near/far 干扰：**LibriSpeech**（~1000 h / ~2,484 speakers）；
- 连续环境噪声：**WHAM!**（~80 h）；语义突发/环境事件：**Hive**（~544 h，44.1k→16k）；
- 三步：① gpuRIR 按 RT60 生成 RIR（Sabine 估计、cutoff 15/60 dB；4 元线阵 2.8 cm 间距，阵元高 1.2–1.8 m；
  说话人高 1.0–1.9 m）；② 目标放 near 范围、干扰按场景距离范围放置；③ target 归一化 **−23 LUFS**（EBU R128），
  near 按 near SIR、far 按 20–35 dB、env 噪声（WHAM+Hive 各衰减 3 dB 后求和），突发事件同 SIR 分布，
  最终 **peak-limited −1 dBFS**，per-source gain clip ±40 dB。
- 可复现性：**固定 master seed 42，train/val/test 独立 seed offset，可 bit-for-bit 重生成**；metadata 含
  room/RT60/位置/RIR hash/SIR/Hive label；单卡 3090 全量训练集 **~48 GPU-hours** 生成；发布 generation scripts+metadata。
- `[我们]` 下载的 RealSSA-Sim：242 files/240 shards、521.5 GB（`docs/REPRODUCTION_PHASE2_REPORT.md:164-167`），
  test 9,169 条与论文一致 ✅。

### 4.3 Split 统计（Table 13）

| split | samples | hours | avg dur | avg speakers | avg near SIR | avg far SIR |
|---|---|---|---|---|---|---|
| Train | 100,826 | 129.2 | 4.61 s | 13.21 | 8.51 dB | 27.49 dB |
| Val | 9,133 | 11.7 | 4.61 s | 13.29 | 8.51 dB | 27.47 dB |
| Test | **9,169** | 11.7 | 4.61 s | 13.23 | 8.49 dB | 27.50 dB |

`[我们]` 我们只转换/评测了 test（9,169），**训练集 100,826 条未用于训练**（Phase-2 只做了 LRS2 训练）。
论文公开后，RealSSA-Sim 训练复现的最大缺口从“无 near 负样本”变为“**需要重生成 near 源 + 8×3090 算力**”。

### 4.4 RealSSA-Real（App A.4 / Table 14）

- **54 sessions，总 5,035 s（83:55），8 个环境**（study room、classroom、internet cafe、billiard hall、shopping mall、
  supermarket、street-day、street-night）；iPhone 17 Pro 单机录制，视频 25 fps、音频 16 kHz；不做去噪/去混响/响度归一化；
  参与者人数在 HTML 中重影（`8484 participants (1010 male and 55 female)`）`[待PDF]`，每场 2–5 人 `[待PDF]`；
  人脸打码；只用于评测。
- `[我们]` Phase-2 的 RealSSA-Real 为 54 段、83.99 min ≈ 5,039 s（`P2E-report.md:213`），**与论文 83:55 吻合** → 数据完整。
- `[论文]` App E：RealSSA-Real 无 isolated target → 只报 DNSMOS/NISQA/WER/speaker-sim，**不报 SI-SDR**（与我们 P2-E 一致 ✅）。

---

## 5 论文表格逐项核对（与我们已有数字）

| 论文项 | 论文值 | 位置 | 我们的复现值 | 判定/解释 |
|---|---|---|---|---|
| RealSSA-Sim SI-SNRi / SDRi | **7.7 / 8.2** | Table 3 | 7.774 / 8.479（发布 ckpt；channel=mean） | 数值接近（+0.07/+0.28）；但论文是 RealSSA-trained + channel-1，**我们 ckpt/通道都不同**，不能称“复现 ✅”；先按 `--channel 0` 复评再判断 |
| LRS2-2Mix SI-SNRi / SDRi | **14.1 / 14.7** | Table 3 | 7.029 / 7.713（发布 ckpt 跨语料） | 论文为（未声明的）LRS2 训练 ckpt；我们 P2-D 自训 130 ep 只到 8.858/9.270；差距=训练数据+算力（8×3090/bs4/1000ep vs 我们单卡/130 ep） |
| LRS3-2Mix SI-SNRi / SDRi | **16.4 / 16.7** | Table 3 | 无法评测（LRS3-2Mix 数据未发布） | ❌ 仍不可复现 |
| VoxCeleb2-2Mix SI-SNRi / SDRi | **10.5 / 11.8** | Table 3 | 无法评测（数据未发布） | ❌ 仍不可复现；但 Table 6 证明论文训练过 VoxCeleb2 版 Falcon |
| RealSSA-Real DNSMOS | **3.276** | Table 3 | OVRL 2.7901 / P808 3.3285（发布 ckpt，channel=mean，窗口化） | 论文 = P.835 [35]，**非 P808**；3.276 = RealSSA-trained ckpt；我们 ckpt 不同 → 当前 2.79 不可直接比（见 §3） |
| RealSSA-Real WER / similarity | Falcon **32.74% / 42.23** | Table 4 | 未做 | ➕ 可补（Mega-ASR + WeSpeaker） |
| RealSSA-Sim WER / similarity | Falcon **25.47% / 43.05**；GT **11.83% / 64.69** | Table 4 | 未做 | ➕ 可补 |
| Params | **0.5 M**（trainable） | Table 5 / README | 537,697 = 0.538 M（audio only） | ✅ 口径一致（论文“0.5M”为 trainable/音频网络；11.72M 含冻结 FRCNN 见 README 脚注） |
| MACs | **10.6 G** | Table 5 | 复算 10.611 G（`docs/understanding/04-*.md` A47） | ✅ |
| GPU latency | **51.3 ms**（2 s clip, bs1, RTX3090, CUDA events, median of 25） | Table 5 / App E | 我们未按同协议测（A800） | ⚠️ 需按论文协议复测才能对账 |
| CPU latency | **1361.2 ms**（单线程 Xeon Gold 6226R） | Table 5 / App E | 未测 | ➕ |
| RAM / GPU mem | **125.3 MB / 364.7 MB** | Table 5 | 未测 | ➕ |
| 相对 AV-TF-GridNet | params −91.9%、compute −95.0%（20.2×）、GPU latency −91.0%（11.1×） | Abstract/§6.2 | — | 与 Table 5 商算一致（6.2→0.5、213.8→10.6、567.8→51.3） |

---

## 6 消融与架构（§6.4 + App I/K/L/M/G）

### 6.1 架构块与消融的对应

- `[论文]` **TFRB** = 频率 bi-LSTM → 时间 uni-LSTM → 2D block-causal self-attention（Eq.5，Fig.4）；
  **GCI** = 用 bottleneck 全局特征 gate/注入每个 skip（Eq.6），编码器/解码器各一套。
- `[论文]` 消融**没有**单独拆 GCI；与 TFRB/深度相关的只有 **downsampling stage 数**（Table 7）与**递归单元**（Table 18）。
- `[代码]` 仓库 `look2hear/models/Falcon.py:749-888`：`upsampling_depth=4`、每层 `DualPathRNN`+attention、
  `fusion_layers`（GCI 语义）、`Separator` 的 `audio_repeats = repeats(2) − fusion_repeats(1) = 1`
  （`Falcon.py:991-993`）→ 等价于论文“AudioBlock 1 次（预融合）+ Separator 1 次（后融合）”，共享权重（`shared: true`）；
  `layer_1 hid_chan=24/双向`、`layer_2 hid_chan=72/单向`、`layer_3 n_head=4` 与论文 App C Table 15 **完全一致** ✅。
- **MDCL：论文全文零次出现**；Table 8 的“Loss”只有 reconstruction + encoder-space contrastive。→ 我们
  “MDCL 是死代码、不在论文最终模型”的结论 **确认**（`look2hear/models/Falcon.py:1464-1467` 的 alias 与
  `look2hear/system/av_litmodule_mdcl.py` 仍属历史代码）。

### 6.2 消融数字（论文 Table 7/8/9/18/19/20 + App M）

| 消融 | 设置 | SI-SNRi / SDRi |
|---|---|---|
| Separator 深度（Table 7） | 1 / 2 / 3 stages | 7.1/7.7 → 7.4/7.9 → **7.7/8.2**（+0.6 dB） |
| 对比损失（Table 8） | Falcon w/o → w/ | 7.3/7.8 → **7.7/8.2**（+0.4）；SwiftNet-12 7.3/7.8→7.5/8.0（+0.2）；Dolphin 6.7/7.3→7.0/7.6（+0.3） |
| τ 调度（Table 9） | fixed 0 / fixed 1 / anneal→2 / **anneal→1** | 7.5/8.1、7.4/7.9、7.5/8.0、**7.7/8.2** |
| 递归单元（Table 18） | RNN / SRU / GRU / **LSTM** | 6.2/7.1、7.4/7.9、7.3/7.9、**7.7/8.2**；LSTM 0.25 M / 5.69 G |
| 输出参数化（Table 19） | complex mask / real mask / direct RI | **7.7/8.2** / 7.3/7.9 / 7.1/7.2 |
| overlap（Table 20） | 5 组 0.11–1.0 | SI-SNRi 7.71–7.95、SDRi 8.22–8.39 |
| 缺失视觉（App M） | p=0 / 0.25 / 0.50 / 0.75 / 1.0（random） | 7.48/7.88 → 7.28/7.72 → 6.75/7.30 → 5.97/6.73 → −9.91/0.13 |
| 分场景/分 near 数（App G Table 17） | 16 格 | Falcon 11 格最优、5 格次优；small meeting 最难（6.7–7.1），outdoor 最易（9.0–9.4） |

**对我们的意义**：Table 8 的 +0.4 dB 与我们在 LRS2 上实测“s2-near 微调 +0.24/+0.23 dB”同向同量级 ✅；
但论文的 τ 消融（Table 9）在 **K≥2** 的 RealSSA-Sim 上做，我们的 LRS2 K=1 实验**不能**验证 τ（见 §3）→
复现 τ 消融必须走 RealSSA-Sim 重生成 near 源路径。

---

## 7 与我们的差距：已被论文填上的空格 vs 仍不可复现

| 空格 | 论文是否填上 | 现在我们该做什么 |
|---|---|---|
| DNSMOS 口径（P.835 vs P.808） | ✅ **填上**：P.835 [35]（子分数未写，推断 OVRL） | 报告改称“P.835 OVRL”；**不再声称 P808 3.3285≈3.276**；要复现 3.276 需 **RealSSA-Sim-trained ckpt** |
| RealSSA 训练配方（loss/λ/m/τ/bs/lr/epoch/硬件） | ✅ **填上**（Table 16） | 用 `pairwise_neg_sisdr`（不是 SNR）+ τ/1000 + 8×3090 bs4 重训；仓库 `configs/LRS2_2Mix_Paper_8gpu.yml` 的 bs5、β200 需改 |
| RealSSA near 负样本 | 🟡 **半填**：语义/来源/pipeline 明确；发布包不含 near 波形，需重生成 | 等 generation scripts+metadata 发布，或按 App A.2 用 LibriSpeech+LRS3+WHAM!+Hive 重放 mixing；此前说“无法忠实复现负样本”应改为“可重生成，成本高” |
| LRS2 14.1 配方 | 🟡 **半填**：通用训练配方给了；**LRS2 专用配方/混合脚本仍引用 prior work [6,17]**；Falcon LRS2 ckpt 未声明 | 用 Table 16 配方 + 我们的 LRS2-2Mix 数据（s1 target / s2 near）训练 1000 ep/8×3090；这是唯一有希望从 8.86→14.1 的路径 |
| LRS3 16.4/16.7、Vox 10.5/11.8 | ❌ 数据未发布（LRS3-2Mix/VoxCeleb2-2Mix 仓库无数据） | 仍不可复现；只能引用论文 + 用 Vox 训练 RealSSA-Real（Table 6 路径）做部分交叉验证 |
| RealSSA-Real 的 WER/NISQA/speaker-sim | ✅ 协议填上（Mega-ASR / WeSpeaker ResNet34-LM / NISQA） | 补测：WeSpeaker checkpoint 公开可下；Mega-ASR 可得性待查；NISQA 可 pip |
| RealSSA-Sim 通道口径 | ✅ **填上**：channel-1（我们用了 4ch mean） | `realssa_to_falcon.py --channel 0` 重转 test（或直接读原 wav 第 1 通道）再评；预计改动 ±0.1~0.3 dB |
| 窗口化/流式评测（Studio −1.9~−2.3 dB） | ❌ **未填**：论文只报 2 s clip、8 ms hop、lookahead 120 ms，无长录音拼接/流式评测协议 | Studio 的窗口化数字仍无论文对照；不能声称“与论文流式协议一致” |
| 发布 `falcon.pt` 的训练集 | ❌ **未填**：论文全文未提发布权重 | 仍只能引用仓库 `conf.yml` 快照（LRS3_complex_nonmoving）+ PROVENANCE；建议向作者确认 |
| LRS2/LRS3/Vox 的 Falcon 训练 ckpt | ❌ **未填** | 建议向作者确认；若确有 ckpt 可省 1000 GPU-hours |

---

## 8 对既有文档/结论的修正清单

| 旧结论（出处） | 判定 | 依据 |
|---|---|---|
| “论文未公开，DNSMOS 3.276 口径未知” → 猜测 P808（`P2E-report.md:107`、`REPRODUCTION_PHASE2_REPORT.md:105-107`） | ❌ **推翻** | 论文引用 [35] DNSMOS **P.835**；P808 非论文指标；我们 3.3285 的“+0.05 对齐”是巧合 |
| “P2-F：7.03/7.71 是 LRS3-trained 官方权重跨语料结果；论文按 benchmark 分别训练”（`P2F-report.md` §5） | 🟡 **部分确认** | 论文明确 baseline 重训 RealSSA-Sim/Vox；Table 6 两版 Falcon；但 Falcon LRS2/LRS3 训练集**未明说** |
| “训练用 SNR 是论文技巧”（`docs/understanding/04-*.md` §4.1） | ❌ **推翻** | 论文 App D：`L_recon = −SI-SNR`；仓库 config 的 SNR 是**旧配方/偏离**，不是论文设计 |
| “NASS K=1 时 β 失效”（深挖 04 §2.2） | ✅ **确认并强化** | 论文 τ 消融必须在 K≥2 上做（Table 9）；K=1 时 softmax 恒 1 |
| “MDCL 不在论文最终模型” | ✅ **确认** | 论文全文无 MDCL；§4.3/Table 8 只有 reconstruction+contrastive |
| “RealSSA 无 near 波形 → 无法忠实复现负样本” | 🟡 **修正** | 发布包确实没有；但论文承诺发布 metadata+脚本，可按 App A.2 重生成 → 从“不可复现”改为“高成本可复现” |
| “RealSSA-Sim 通道：论文未记录”（`realssa_to_falcon.py:24` 注释） | ❌ **推翻** | 论文 App A.2 明确 first-channel/channel-1；我们默认 mean 需重评 |
| “RealSSA-Real 我们 P808 3.3285 最接近论文 3.276，可称复现”（Phase-2 报告 §5.1） | ❌ **推翻** | 论文 3.276 是 **RealSSA-Sim-trained** 模型的 P.835 指标；我们的 ckpt（发布权重）不是该模型；OVRL 2.79 只与 Table 6 的 Vox-trained 2.878 同量级 |
| “14.1/14.7 需要 LRS2-trained ckpt”（Phase-2/深挖结论） | ✅ **确认（结构性）** | 论文 Table 6 直接量化“训练数据改变 RealSSA-Real 指标 0.4 MOS”；LRS2 14.1 不可能来自跨语料模型 |
| “发布 falcon.pt 训练集 = LRS3_complex_nonmoving” | 🟡 **未见论文确认** | 仍只有仓库 conf 快照证据；论文未提发布权重 |
| “窗口化推理 −1.9~−2.3 dB” | 🟡 **与论文无冲突也无支持** | 论文无窗口化/流式评测数字；只能作为 Studio 实现差异记录 |

---

## 9 未解问题 / 需 PDF 或作者确认

1. `[待PDF]` RealSSA-Real 参与者人数与性别构成（HTML 重影 `8484 participants (1010 male and 55 female)`；
   按上下文疑为 10 participants/5 男 5 女，但无法确认）。
2. **Falcon 在 LRS2-2Mix / LRS3-2Mix 列的训练集与 ckpt**：论文只明确 baseline 的重训范围与 Table 6 的两个
   Falcon 版本；LRS2 14.1 到底由哪个 ckpt 产生，文中未写 → 建议作者澄清。
3. **发布 `falcon.pt` 是哪一个训练**：论文全文未提及发布权重（只给 code+demo）；仓库侧证据仍为
   `Experiments/checkpoint/Falcon/conf.yml:99-102` = LRS3_complex_nonmoving。
4. **DNSMOS 子分数**：论文只写“DNSMOS [35]”，未写 SIG/BAK/OVRL 哪一个进入 Table 3；按单值惯例推断 OVRL。
5. **SDR 实现细节**：论文引用 Vincent 2006 [34]，但未写 filter length、是否 `change_sign`、用哪个库；
   我们 512-tap fast_bss_eval 只能算同族近似（P2-A 显示 512→64 tap −0.35 dB）。
6. **RealSSA-Sim 评测片段**：论文未写是整段（avg 4.61 s）还是 2 s 窗；我们按转换后整段评。
7. **RealSSA-Real 通道**：iPhone 录制是单声道还是多通道、评测用哪一通道，未写明。
8. **RealSSA-Sim 通道矛盾**：论文说 released waveforms 是 channel-1 streams，但 HF 数据卡写
   “4-channel mixture/target”（`docs/verification/phase1/dataset-cards/RealSSA-Sim-README.md:42,108-109`）——
   需确认发布包是否改版/论文笔误。
9. **生成脚本 release 时间**：论文“We will release the generation scripts and the metadata”；目前仓库/数据包
   是否已含 metadata 需核对（决定 near 负样本重生成成本）。
10. **Mega-ASR [37] 可用性**：arXiv 2605.19833，需确认权重/代码是否公开（WER 复现依赖）。

---

## 10 阅读/验证记录

```bash
# 论文抓取（web_fetch，HTTP 200）
https://arxiv.org/html/2610.32749v1          # 全文 99,124 B，落盘 636 行
https://ar5iv.labs.arxiv.org/html/2610.32749 # 仅 abstract 页（ar5iv 未生成 HTML）
https://arxiv.org/pdf/2610.32749v1           # 未下载（受“不下载大文件”约束）；标 [待PDF] 的项需它

# 仓库侧对照（只读）
git log --oneline -2                          # d9ec1e8 merge upstream/main
git diff --stat 689555c 439cbe9               # README 论文链接/下载计数，无协议变化
sed -n '1,60p' configs/Falcon.yml             # 宽度/STFT/损失/β 配置
sed -n '40,100p' configs/Falcon.yml           # layers: 24/72/4-head；video_params
grep -n "STFTEncoder\|spec.real" look2hear/models/Falcon.py   # 3 通道堆叠 (Falcon.py:1118)
grep -n "repeats\|audio_repeats\|fusion_repeats" look2hear/models/Falcon.py  # 双 B_a 语义
```

- 论文表格引用均按 HTML 的 Table 编号；App 表格编号（Table 10-21）在 HTML 中可读，但 App M 的总表**表号在抽取中丢失**，
  仅引“App M missing visual”。
- 未读内容：图 1/2/3/4/5/6/7/8/9 的像素内容（只读图注/正文描述）；Algorithm 1 的逐行伪代码（正文 A.2 已覆盖语义）；
  App N（LLM 声明）与致谢仅略读。
- 本文件未改动任何代码/配置/他人文档；所有“应改”建议均以文字给出，供 Lead 统一整合与后续执行。

## 11. 2026-10-30 ch0 复评结果（论文 channel-1 验证）

论文 RealSSA-Sim 只用 channel-1（代码 `--channel 0`）。我们此前主结果用 4ch **mean**；2026-10-30 已用同一官方 `falcon.pt`、
同一 9,169 条、同一指标完成 ch0 全量评测：

| 指标 | mean（此前） | ch0（论文通道） | ch0−mean | 论文 | ch0 对论文 |
|---|---|---|---|---|---|
| SI-SNRi | 7.774 | **7.977** | +0.202 | 7.7 | **+0.28** |
| SDRi | 8.479 | **8.452** | −0.027 | 8.2 | **+0.25** |
| SI-SNR / SDR | 9.980 / 10.645 | 9.320 / 10.111 | −0.660 / −0.534 | — | — |
| PESQ / STOI | 2.094 / 0.817 | 2.004 / 0.815 | −0.090 / −0.002 | — | — |

结论：**通道选择影响 <0.3 dB，两通道都在论文 ±0.3 容差内**；论文 channel-1 不解释残余偏差，也不改变“RealSSA-Sim 已复现”。
真正不可比的是 **RealSSA-Real DNSMOS**：论文 3.276 属 RealSSA-trained ckpt 且为 P.835；我们官方权重同口径 **full4 = OVRL 2.7941 / P808 3.3365**（窗口起点 bug 修复后重跑，full3 作废）。
证据：`docs/verification/phase2/P2E-realssa-sim-ch0-{eval-20261030.log,metrics_combined-20261030.csv,report.md}`。

> 说明：当前结论/证据/待办以 `docs/STATUS.md` 为准；本文保留为 2026-10-30 论文协议提取记录。
