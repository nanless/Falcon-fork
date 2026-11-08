# 01 模型主干：`look2hear/models/Falcon.py` 逐层深挖

> 子代理 W1-01｜范围：`look2hear/models/{Falcon.py, Falcon_Contrastive.py, base_model.py, __init__.py}`
> 远端仓库 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork` @ `6c1d5b7`（只读）
> 环境 `/root/miniforge3/envs/falcon/bin/python`（torch 2.8.0+cu128），**全部实测在 CPU（`CUDA_VISIBLE_DEVICES=`）**
> 符号：`T`=音频帧数（hop=128→125 fps）、`F`=频点（n_fft=256→129）、`T_v`=视频帧数（25 fps）。2 s @16 kHz ⇒ `T=251, F=129, T_v=50, L=32000`。
> 论文 arXiv:2610.32749 已于 2026-10-23 公开；本文件只写代码事实，逐条对齐见 11，因果性定论见 12。

---

## 0. 结论速览

1. **架构不是「cross-attention 视听融合」，而是「加性偏置注入」**：视频只经 `Conv1d(512→112,g=16)` + `nearest` 上采样成 `[B,112,T_a]`，`unsqueeze(-1)` 后**逐帧加**到音频特征上（`Falcon.py:241-267`）。全模型**没有任何跨模态注意力**；唯一的注意力在音频主干内部（`MultiHeadSelfAttention2D`，自注意力，只在最深一层）。
2. **模型不是 2 源分离，是目标说话人提取（TSE）**：所有 6 份 config 都是 `n_src: 1`；数据集只有 `target.json`（`look2hear/datas/avspeech_dataset.py:47-49`）。而且 **`n_src≥2` 在代码上是坏的**：实测 `n_src=2` 前向直接 `RuntimeError: shape '[2, 224, 251, 129]' is invalid`（`Falcon.py:1345` 的 `in_chan=encoder_out_chan*n_src` 与 `MaskGenerator` 输出的每源通道数不一致）。所以不是"支持但没训"，是"结构上跑不通"。
3. **参数量 537,697（全部 requires_grad）+ 65 个持久化 BN buffer = state_dict 537,762**；其中 **5,568（1.04%）是死参数**：`concat_layers[2]` 从未被调用——实测「同权重下把它清零，输出 **bit-identical**」；对照组 `concat_layers[1]`/`fusion_layers[2]` 清零分别改变输出 0.441/0.223。
4. **音频 10.6109 GMAC，与论文 10.6 G 同尺**：ptflops 返回值，与 README/STATUS 的 10.611 G 逐位一致，本轮独立复算。
   它**含 LSTM、不含 FRCNN**。整窗（含 88×88×50 帧视觉前端 15.897 GMAC）是 **26.51 GMAC、视觉占 60.0%**。
   同模型换 `torch.utils.flop_counter` 只有 **5.382 GMAC**（不记 `aten::lstm`），那是另一把尺子。
   2026-11-08 Lead 复核更正：此前"FRCNN 7.7693 G + 音频 10.6109 G = 18.380 G"是 **FRCNN @64×64** 的老口径，
   不是本管线 88×88 输入的整窗成本，两数不可并列。
5. **因果性是「窗口级」，且比文档说的更弱**：名义上 `causal=true` 做了 causal conv + 单向 LSTM + tril mask，但 **`cLNhw` 归一化 = `nn.GroupNorm(1, C)`，在 (C,T,F) 全窗口 3.6 M 个元素上统计**（`layers/normalizations.py:134-146`）。实测：把 `input[q:]` 换成噪声（q=16000），输出**从第 1 个采样点就开始变**；单点脉冲 p=16000 影响输出 **[13, 31999]**。即"窗口内完全非因果，连帧级都不是"。
   几何 lookahead 要分三档：单遍 **57 帧 = 456 ms**（关掉 `cLNhw` 后一遍 AudioBlock）、整链 **113 帧 = 904 ms**
   （另一路 110 帧 / 880 ms，差约 3%）、as-shipped **窗尾 ≤2000 ms**。`Falcon.py:772/808/821` 三处硬编码
   `causal=False` 是整链 lookahead 的卷积来源，改回 `causal=True` 约 6 行（11 §5、12 §6）。
6. **解码器带一个 16 ms 固定延迟**：`STFTDecoder` 在 `causal=True` 下前向 pad `k-1=2` 帧、只在末尾裁 1 帧 ⇒ 净 +2 帧 = **+256 采样**。用 delta 权重做纯直通实测：`causal=True` 与理想 istft 的最佳 lag = **+256 采样，corr=1.0000**；`causal=False` 时 lag=0。**发布权重的端到端 lag 实测为 0**（掩码把这个相位/时移吸收掉了），但这意味着"换解码器/重训"会引入 16 ms 系统错位。
7. **`cLNhw` 是全局耦合元凶，也是"窗口长度敏感"的来源**：同一段音频放进 2 s 窗口 vs 放进 4 s 窗口的前 2 s，输出最大差 0.2249（相对最大幅度 **32.3%**），最大偏差在 0.37 s 处。
8. **`_encode_audio` 的 dummy video 不是"无视频"**：它是 `zeros(B,512,1)`（`video_bottleneck` 是恒等映射，0 参数），经 VideoBlock 后被 **bias 抬成一个非零常数向量（norm=9.72，逐帧相同）**，再作为**频率无关的加性偏置**加到音频特征上。数值上与 `zeros(512,50)` 等价到 3e-6，所以不算 bug，但语义上"视觉分支仍在贡献一个学习到的常数"。
9. **Studio 副本与训练侧"功能等价、非逐字节一致"**：`sha256` 不同（`0d613d2b…` vs `917e2bfc…`），diff 只有 3 类功能差异（`timm.DropPath`→本地 `drop_path`、STFT 的 MPS 分支、中英注释）；**同权重同输入实测输出 bit-identical（max|Δ|=0）**。
10. **`falcon.pt` 与 `falcon-big.pt` 是两套结构**：small 525 audio keys / 537,697 参数 / `arch=FalconSpeakerContrastive`；big 517 audio keys / 1,158,617 参数 / `arch=SpeakerContrastiveV2`，且 **big 少 8 个 key，正是 `process_f/t_layers.*.out_proj.weight`**——因为 big 的 `hid_chan*方向数 == in_chan`（48×2=96、96=96）使 `out_proj` 退化成 `Identity`（`Falcon.py:379-383`）。
11. 单窗 2 s 延迟（CPU，B=1、2 s 窗）：**汇总约 0.6 s（4 线程实测 592–694 ms，随负载波动；1 线程约 1.0 s）**；线程数影响很大——**8 线程中位 357 ms（RTF 0.179）**、**1 线程 962 ms（RTF 0.481）**；其中 `separator` 占 **92%**（985.6/1067.2 ms，1 线程）。

---

## 1. 事实基线

### 1.1 文件与类清单（`grep -n "^class "`）

| 行 | 类 | 角色 | 在 Falcon 图里被实例化？ |
|---|---|---|---|
| `Falcon.py:11` | `ConvNormAct` | conv(pre_norm→pre_act→conv→causal trim→norm→act) | ✅ 大量 |
| `Falcon.py:111` | `ConvActNorm` | conv→act→norm（顺序不同，供 attention 的 Q/K/V 用） | ✅ |
| `Falcon.py:205-239` | ~~`FiLMBlock`~~ | 被整段注释掉的 FiLM 融合（**死代码**） | ❌ |
| `Falcon.py:241` | `NeuroSCNetFusion` | **视听融合（加性）** | ✅ 1 个（shared） |
| `Falcon.py:269` | `InjectionMultiSum` | 局部嵌入 × 全局 gate + 全局嵌入（NSNet 式 injection） | ✅ 4+4 个 |
| `Falcon.py:333` | `DualPathRNN` | 单轴 LSTM + 残差 + 可选出投影 | ✅ 8 个（F 轴 4 + T 轴 4） |
| `Falcon.py:419` | `MultiHeadSelfAttention2D` | 沿 T 轴的自注意力（头内 1×1 卷积） | ✅ 4 个 |
| `Falcon.py:547` | `PositionalEncoding` | 正弦位置编码（buffer） | ❌ 只在下面那条死链里 |
| `Falcon.py:565` | `MultiHeadSelfAttention` | `nn.MultiheadAttention` 包装 | ❌ 未被使用 |
| `Falcon.py:620` | `FeedForwardNetwork` | ConvNormAct 三层 | ❌ 未被使用 |
| `Falcon.py:664` | `GlobalAttention` | MHSA+FFN | ❌ 未被使用（整条 `GlobalAttention` 链是死代码） |
| `Falcon.py:693` | `AudioBlock` | **音频主干**（4 级 T/F 双下采样 U 形） | ✅ 1 个（shared，跑 2 次） |
| `Falcon.py:900` | `VideoBlock` | 视频主干（1×1 + LSTM） | ✅ 1 个（shared，跑 1 次） |
| `Falcon.py:978` | `Separator` | 2 次 audio pass + 1 次 video pass + 1 次融合 | ✅ |
| `Falcon.py:1061` | `STFTEncoder` | 复数 STFT → [real,imag,mag] → Conv2d 3→C | ✅ |
| `Falcon.py:1130` | `STFTDecoder` | ConvTranspose2d C→2 → istft | ✅ |
| `Falcon.py:1201` | `MaskGenerator` | 掩码生成 + 复数域相乘 | ✅ |
| `Falcon.py:1284` | `Falcon` | 顶层（`BaseModel` 子类），含 MDCL 损失 | ✅ 基类 |
| `Falcon_Contrastive.py:8` | `AttentionSpeakerPool` | 时间注意力池化 → 说话人嵌入 | ✅（对比头） |
| `Falcon_Contrastive.py:60` | `FalconSpeakerContrastive` | `Falcon` + `spk_pool_proj` | ✅ 发布权重用的就是它 |
| `base_model.py:32` | `BaseModel` | `sample_rate/from_pretrain/serialize/get_model_args` | ✅ |
| `models/__init__.py` | `get/register_model` | 按类名反射构造 | ✅ |

**死代码（类级，从未实例化）**：`FiLMBlock`(注释)、`PositionalEncoding`、`MultiHeadSelfAttention`、`FeedForwardNetwork`、`GlobalAttention`。它们占 0 参数，但会让"读代码的人以为有全局注意力"。证据：`grep -n "GlobalAttention\|FeedForwardNetwork\|PositionalEncoding\|MultiHeadSelfAttention(" look2hear/models/Falcon.py` 只命中定义处，无构造处。

### 1.2 生效配置（`configs/Falcon.yml`，实测构造所用）

| 键 | 值 | 影响 |
|---|---|---|
| `n_src` | **1** | 目标说话人提取 |
| `pretrained_vout_chan` | 512 | 视频特征通道（FRCNN backend_out） |
| `causal` | **true** | 写进 audio_params/video_params |
| `spk_embed_dim` | 64 | 对比头宽度 |
| `enc_dec_params` | `win=256, hop=128, out_chan=112, kernel_size=3, stride=1, bias=false, act_type=null, norm_type=null, causal=true` | 编解码器 |
| `audio_bn_params` | `pre_norm=cLNhw, pre_act=ReLU, out_chan=112, kernel_size=1, is2d=true` | 瓶颈 |
| `audio_params` | `n_freqs=129, hid_chan=64, kernel_size=5, stride=2, norm=cLNhw, act=PReLU, upsampling_depth=4, repeats=2, shared=true, is2d=true` | 主干 |
| `audio_params.layers.layer_1` | `hid=24, dim=4, k=8, rnn=LSTM, bidir=true` | F 轴 |
| `audio_params.layers.layer_2` | `hid=72, dim=3, k=8, rnn=LSTM, bidir=false` | T 轴（因果） |
| `audio_params.layers.layer_3` | `dim=3, n_head=4, hid_chan=4, norm=LayerNormalization4D, causal=true` | T 轴注意力 |
| `video_params` | `hid_chan=32, kernel_size=3, stride=2, norm=BatchNorm1d, act=PReLU, repeats=1, shared=true, is2d=false` | 视频主干 |
| `fusion_params` | `fusion_type=NeuroSCNetFusion, fusion_shared=true, kernel_size=4, is2d=true` | 融合（后 3 个键**无人读取**） |
| `mask_generation_params` | `mask_generator_type=MaskGenerator, mask_act=ReLU, RI_split=true, is2d=true` | 掩码（`mask_generator_type` 落进 `**kwargs` 被丢弃） |

### 1.3 一句话公式

```
spec   = stft(x)                                        # [B,129,251] complex
E      = Conv2d(3→112,k3,causal)([Re,Im,|spec|])        # [B,112,251,129]
A0     = Conv2d(1×1)(cLNhw(ReLU(E)))                    # [B,112,251,129]
V      = Identity(FRCNN(x_video))                       # [B,512,50]   (0 参数)
A1     = AudioBlock(A0)                                 # 第 1 遍（融合用）
V1     = VideoBlock(V)                                  # [B,512,50]
A2     = A0 + W_v(V1)↑251                               # ← 视听融合=逐帧加性偏置
A3     = AudioBlock(A2 + A0)                            # 第 2 遍（RTFS，不再看视频）
M      = ReLU(Conv2d(1×1)(refined))                     # [B,112,251,129]
Z      = complex_mul(M, E)                              # 每源 [B,112,251,129]
y      = istft(ConvT2d(112→2,k3)(Z))                    # [B,1,32000]
```

---

## 2. 机制拆解

### 2.1 `STFTEncoder`（`Falcon.py:1061-1128`）

- `torch.stft(x, n_fft=256, hop_length=128, window=hann)`（`:1117-1124`），**未显式传 `center`，默认 `center=True`** ⇒ `T = 1 + 32000/128 = 251`，并带 **128 采样（8 ms）前瞻 + 两端 reflect pad**（完整 STFT 参数表见 02 §2.5，此处不重复）。
- `stack([real, imag, sqrt(re²+im²+eps)], dim=1).transpose(2,3)` → `[B,3,251,129]`（`:1125-1126`）。
- `ConvNormAct(3→112, k=(3,3), padding=(2,1), causal=True)`：**实测 conv 出 `[1,112,253,129]`，再 `[:, :, :-2, :]` 裁成 `[1,112,251,129]`**。
  - 关键：`padding=(causal_padding, padding)=(2,1)` 在 `nn.Conv2d` 里是 `(pad_H, pad_W)`，`H=时间T`、`W=频率F` ⇒ **时间维两侧各 pad 2（左侧 pad 提供因果性，右侧的 2 帧随后被裁掉）**，频率维对称 pad 1 保持 `F` 不变。这是本仓库唯一真正"因果"的卷积写法，且**只在 `is2d=True` 时成立**。
- `act_type/norm_type = null` ⇒ `activations.get(None)/normalizations.get(None)` 返回 `nn.Identity`（`normalizations.py:231-233`）⇒ 编码器是**纯线性 3→112 投影**（3,024 参数）。

### 2.2 `audio_bottleneck`（`Falcon.py:1322`）

`ConvNormAct(112→112, k=1, pre_norm=cLNhw, pre_act=ReLU)`。前归一化 = `LayerNormalizationHW(112)`：
`GlobalLayerNorm = nn.GroupNorm(num_groups=1, num_channels=112)`（`normalizations.py:134-146`）+ `CumulateLN4D`（沿 H/时间累加，`normalizations.py:167-197`）。
**这就是全模型非因果的根源**：GroupNorm(1,112) 在 `112×251×129 = 3,626,448` 个元素上求均值方差，任何一个时间帧扰动都会改变所有帧。

### 2.3 `AudioBlock`（`Falcon.py:693-898`）——4 级、T 与 F **同时**减半

`upsampling_depth=4`，`i=0` 用 `k=3,stride=1`，`i≥1` 用 `k=5,stride=2`，`groups=hid_chan`（深度可分离）。

| 级 | 输入 shape | `downsample_layers[i]` spec | 输出 shape | 说明 |
|---|---|---|---|---|
| (in) | `[1,112,251,129]` | gateway 1×1 dw `[1,112,251,129]`；projection 1×1 `112→64` + cLNhw + PReLU | `[1,64,251,129]` | 入口 |
| 0 | `[1,64,251,129]` | k3 s1 p1 dw | `[1,64,251,129]` | 只在 decoder 阶段处理 |
| 1 | `[1,64,251,129]` | k5 s2 p2 dw | `[1,64,126,65]` | T、F 同时 /2 |
| 2 | `[1,64,126,65]` | k5 s2 p2 dw | `[1,64,63,33]` | |
| 3 | `[1,64,63,33]` | k5 s2 p2 dw | `[1,64,32,17]` | 最深，**唯一在 encoder 阶段做 RNN/Attn 的级** |

**反直觉 1**：这不是"频率轴金字塔"，而是 **(T,F) 同时 8× 下采样的二维金字塔**。2 s 窗口在 level 3 只剩 `32 帧 × 17 频点`，T 上的帧率从 125 fps 降到 15.6 fps（帧步长 64 ms）。
**反直觉 2**：`downsample_layers` 全部 `causal=False`（`Falcon.py:772` `causal=False` 硬编码），padding `(2,2)` 对称 ⇒ **时间维有 ±2 帧前瞻**（level 1 上 = ±256 采样；累积到 level 3 ≈ ±1792 采样 ≈ 112 ms）。

level 3 的三段处理（实测 shape）：

| 模块 | 输入 | 内部 | 输出 |
|---|---|---|---|
| `process_f_layers[3]` `DualPathRNN(dim=4,hid=24,bidir)` | `[1,64,32,17]` | `transpose→[1,64,17,32]`，`view(17,64,32)`，LSTM 沿 **F=17** 双向 → `(17,32,48)`，`out_proj Linear(48→64)` | `[1,64,32,17]` |
| `process_t_layers[3]` `DualPathRNN(dim=3,hid=72,uni)` | `[1,64,32,17]` | `view(32,17,64)`，LSTM 沿 **T=32** 单向 → `(32,17,72)`，`out_proj Linear(72→64)` | `[1,64,32,17]` |
| `process_att_layers[3]` `MultiHeadSelfAttention2D(n_head=4,hid_chan=4)` | `[1,64,32,17]` | 4 头 Q/K 各 `Conv2d(64→4,1×1)`、V 各 `Conv2d(64→16,1×1)`；`Q,K→[4B,T,E*F=4*17=68]`；`attn=[4B,32,32]`；`V=[4B,32,16*17]`；`attn_concat_proj Conv2d(64→64)` | `[1,64,32,17]` |

**注意力细节（回答"谁跨谁"）**：`MultiHeadSelfAttention2D` 在 `T` 轴上做**自**注意力，`softmax` 的 key 维度 = `T`（32），`emb_dim = hid_chan × F = 68`；`causal=True` 时加 `tril` 掩码（`Falcon.py:518-524`）。**它只在 level 3 存在（每级一个实例，但只有 level 3 在 encoder 阶段被调用；level 2/1/0 的实例在 decoder 阶段被调用）**。所以"全局"上下文靠的是把 level 3 的输出 `F.interpolate(..., mode="nearest")` 放大到各级，而不是跨级注意力。

**Fusion / Concat（NSNet 式，`Falcon.py:316-331`）**：
```
x_fused[i] = local_emb(down[i]) * sigmoid(global_gate(global))↑ + global_emb(global)↑
expanded   = concat[-1](x_fused[2], x_fused[3]) + down[2]     # :870
for i in 1, 0: expanded = concat[i](x_fused[i], expanded) + down[i]  # :881-884
```
`InjectionMultiSum.forward` 里 `F.interpolate(..., size=new_shape, mode="nearest")`（`:322-325`）——**8× 最近邻上采样**，无参数、无平滑。

**实测调用次数（一次完整 forward，含 audio_net 的 2 次调用）**：

| 索引 | `downsample` | `process_f` | `process_t` | `process_att` | `fusion` | `concat` |
|---|---|---|---|---|---|---|
| 0 | 2 | 2 | 2 | 2 | 2 | **2** |
| 1 | 2 | 2 | 2 | 2 | 2 | **2** |
| 2 | 2 | 2 | 2 | 2 | 2 | **0 ← 死** |
| 3 | 2 | 2 | 2 | 2 | 2 | **2** |
| 合计 | 8 | 8 | 8 | 8 | 8 | 6 |

（hook 计数；单次 `AudioBlock` 调用即表中数字 /2。`concat_layers[2]` 为 0 是**结构性**的：`upsampling_depth=4` 时 `:870` 用 `[-1]`，`:881` 的 `range(4-3,-1,-1)` 只产生 `[1,0]`。）

### 2.4 `VideoBlock`（`Falcon.py:900-976`）——比 config 看起来简单得多

`video_bottleneck = ConvNormAct(kernel_size=-1)` ⇒ **`conv = nn.Identity`，`out_chan = in_chan = 512`，0 参数**（`Falcon.py:1323`；实测 `video_bottleneck` 参数 = 0，前向 trace 全是 Identity）。

`VideoBlock` 实际只用了 4 个算子（config 里其它键全被忽略）：

| 层 | spec | 实测 shape |
|---|---|---|
| gateway | `Conv1d(512→512, k=1, groups=512)` + PReLU | `[1,512,50]` |
| projection | `Conv1d(512→32,k=1)` + **BatchNorm1d(32)** + PReLU | `[1,32,50]` |
| rnn | `nn.LSTM(32→32, uni)` | `(1,50,32)` |
| residual_conv | `Conv1d(32→512,k=1)` | `[1,512,50]` |

**被忽略的 `video_params` 键**：`kernel_size=3`（三处 conv 全是 k=1）、`stride`、`upsampling_depth`、`layers`、`num_heads`、`head_dim`、`expand_v`、`use_output_gate/use_short_conv/use_output_correction/use_inner_decay`（`Falcon.py:901-919` 收下但 `__init__` 里一个都不用）。⇒ 视频主干是"两层 1×1 + 一层 LSTM"，**没有卷积形时间建模，也没有 config 里显式写出的 k=3**。

**fps 对齐**：数据集 `fps_len = int(segment*25)`、`mouth_start = int((rand_start/16000)*25)`（`avspeech_dataset.py:43,108`）⇒ 2 s 给 **50 帧**。视频前端 `FRCNNVideoModel` 的 `frontend3D` 是 `Conv3d(k=(5,7,7), stride=(1,2,2), padding=(2,3,3))` + `MaxPool3d(k=(1,3,3))`（`frcnn_videomodel.py:55-72`），**时间 stride=1、时间 padding 对称 2** ⇒ `Tnew = T = 50`，且**视频侧自带 ±2 帧（@25fps = ±80 ms）前瞻**。

### 2.5 视听融合：`NeuroSCNetFusion`（`Falcon.py:241-267`）

```python
v_feat = self.v_proj(video)              # Conv1d(512→112, k=1, groups=gcd(512,112)=16)
v_feat = F.interpolate(v_feat, size=T_a, mode="nearest")   # 50 → 251  (×5.02)
Z_sel  = audio + v_feat.unsqueeze(-1)    # [B,112,251,129] + [B,112,251,1]
```

- **Q/K/V 来源：不存在。** 没有 Query/Key/Value 三元组，没有 softmax 权重，没有跨模态注意。
- 融合是 **rank-1、频率无关的加性偏置**：同一个 `[B,112,T_a]` 值被广播到 129 个频点。
- 维度：`512 →(g=16 深度可分离 1×1)→ 112`，实测参数 **3,696**。
- 帧率对齐：**`nearest` 上采样 50→251**（不是线性/重复帧）；若 `T_v == T_a` 则直接跳过插值（`:258`）。
- 视频的唯一作用：给音频特征加一个"逐帧、逐通道、频率平坦"的偏置项。信息量上限是 `112×251` 个实数/窗。

**这条与"cross-attention 融合"的说法直接冲突**（见 §6-1）。

### 2.6 `MaskGenerator`（`Falcon.py:1201-1282`）——复数掩码，但施加在**学习到的复数嵌入**上

```
refined [B,112,251,129] → PReLU → Conv2d(112→112,k=1) → ReLU → masks [B,112,251,129]
RI_split=True:
  masks.view(B, n_src, 2, 56, T, F)                 # real/imag 各 56
  emb  .view(B, 2, 56, T, F)
  re = emb_re*mask_re - emb_im*mask_im
  im = emb_re*mask_im + emb_im*mask_re              # ← 真正的复数乘法
  cat([re,im], 2) → [B, n_src, 112, T, F]
```

- **不是实数 mask 直接乘**：`RI_split=true` 时是 2×2 复数乘法（`Falcon.py:1263-1267`）；`RI_split=false` 分支才是逐元素实数乘（`:1269-1271`）。发布权重用 `RI_split=true`。
- **关键陷阱**：`in_chan=112` 被**硬切成 0:56 / 56:112 当作实部/虚部**，但编码器输出是 `Conv2d(3→112)` 的自由混合通道，**没有物理约束保证 0:56 是实部**。也就是说这是一个"在任意 56 维子空间上定义复数代数"的结构——网络必须自己学着把前 56 维当实部用。解码器最后 `ConvTranspose2d(112→2)` 才把实/虚读回物理量。所以"复数域掩码"在本实现里是**学出来的复数结构，不是 STFT 的自然复数结构**。
- 掩码是 `ReLU` 输出 ⇒ 非负，只能衰减/保持相位方向（不能翻转）。
- 实测（真实权重 + 白噪声输入）：掩码 **98.9% 元素为 0**，`mean=0.00161`、`p99=0.0151`、`max=1.740`；`|Z|/|E| = 0.0388`。

### 2.7 `STFTDecoder`（`Falcon.py:1130-1199`）

```
x.view(B*n_src, 112, T, F)                         # 112 = encoder_out_chan(1 源)
pad (0,0,k-1,0) → T: 251→253                       # :1178-1179
ConvTranspose2d(112→2, k=3, s=1, p=1) → [1,2,253,129]
trim [:, :, :-padding, :] → [1,2,252,129]          # :1184  只裁 1 帧
complex(re,im) → [1,129,252] → istft(center=True, length=32000)
```
净效果：**+2 帧（256 采样）延迟**（实测见 §4-E10），并且末尾 256 采样被 `length` 裁掉。

### 2.8 `Separator`（`Falcon.py:978-1059`）——为什么"跑两遍"

```
fusion_repeats = video_params.pop("repeats",0) = 1      # :1000  ← 融合遍数取自 **video** 配置
audio_repeats  = audio_params.pop("repeats",0) - fusion_repeats = 2-1 = 1   # :1001
shared=true ⇒ 只有一个 AudioBlock/VideoBlock/caf 实例，被复用
pass1: audio = AudioBlock(audio); video = VideoBlock(video); audio = caf(audio, video)  # :1037-1049
pass2 (RTFS): audio = AudioBlock(audio + audio_residual)   # :1052-1057，audio_residual = **进 Separator 前**的 A0
```
- **视频只参与第 1 遍**；第 2 遍是纯音频精修，且输入是 `第1遍输出 + 原始 A0`。
- `i>0` 才加残差 ⇒ 第 1 遍不加、第 2 遍加（不对称，容易读错）。
- 因此 `AudioBlock` 的 445,766 参数被**用两次**，MACs 翻倍（4.798 G × 2 = 9.595 G）。

### 2.9 对比头 `AttentionSpeakerPool`（`Falcon_Contrastive.py:8-57`）

`[B,112,251,129]` → `mean(dim=-1)` 折频率 → `[B,112,251]` → permute `[B,251,112]` → `Linear(112→28)+Tanh+Linear(28→1)` → `softmax(dim=1)` 时间注意力 → 加权和 `[B,112]` → `Linear(112→64)+ReLU+Linear(64→64)` → `F.normalize`。参数 **14,584**。
**只用于 `encode_stft/pool_spk_embed`（训练时的同一空间说话人对比损失），完全不在分离前向路径上。**

### 2.10 完整 shape 表（输入→逐层→输出；B=1，2 s@16 kHz，`T_v=50`）

| # | 张量 / 模块 | shape | 备注 |
|---|---|---|---|
| 0 | `audio_mixture` | `[1,32000]` | 2 s |
| 1 | `torch.stft` | `[1,129,251]` complex | `center=True` ⇒ 251 帧 |
| 2 | 3 通道堆叠 | `[1,3,251,129]` | `[Re,Im,Mag]` |
| 3 | `encoder.conv` Conv2d k3 p(2,1) | `[1,112,253,129]` | |
| 4 | causal trim `[:-2]` | `[1,112,251,129]` | |
| 5 | `audio_bottleneck`（cLNhw+ReLU+1×1） | `[1,112,251,129]` | 424.29 MMac |
| 6 | `mouth_embedding` | `[1,512,50]` | FRCNN 输出 |
| 7 | `video_bottleneck` | `[1,512,50]` | **Identity，0 参数** |
| 8 | `audio_net` L0 downsample | `[1,64,251,129]` | |
| 9 | L1 | `[1,64,126,65]` | s2 |
| 10 | L2 | `[1,64,63,33]` | s2 |
| 11 | L3 | `[1,64,32,17]` | s2 |
| 12 | L3 f-RNN / t-RNN / attn | `[1,64,32,17]` ×3 | |
| 13 | `fusion_layers[i]` | 各级 shape | 全局上下文来自 `[1,64,32,17]` |
| 14 | `concat_layers[3]` → `[1,64,63,33]` +down[2] → process idx2 | `[1,64,63,33]` | |
| 15 | `concat_layers[1]` → `[1,64,126,65]` +down[1] → process idx1 | `[1,64,126,65]` | |
| 16 | `concat_layers[0]` → `[1,64,251,129]` +down[0] → process idx0 | `[1,64,251,129]` | `concat[2]` 缺席 |
| 17 | `interpolate→(251,129)`，`residual_conv 64→112`，`+residual` | `[1,112,251,129]` | `AudioBlock` 输出 |
| 18 | `video_net` | `[1,512,50]` | 42,850 参数 |
| 19 | `caf.v_proj` + `nearest(50→251)` + `unsqueeze(-1)` + 加 | `[1,112,251,129]` | 第 1 遍结束 |
| 20 | pass2 `audio_net(audio + A0)` | `[1,112,251,129]` | |
| 21 | `mask_generator` `[PReLU, Conv2d 112→112, ReLU]` | `[1,112,251,129]` | 424.29 MMac |
| 22 | `__apply_masks`（复数乘） | `[1,1,112,251,129]` | |
| 23 | `decoder` view+pad | `[1,112,253,129]` | |
| 24 | `ConvTranspose2d(112→2,k3)` | `[1,2,253,129]` | 65.80 MMac |
| 25 | trim 1 帧 | `[1,2,252,129]` | **+2 帧错位** |
| 26 | `complex` + transpose | `[1,129,252]` | |
| 27 | `torch.istft(center=True, length=32000)` | `[1,32000]` | |
| 28 | `view(B,n_src,L)` | `[1,1,32000]` | 输出 |

---

## 3. 关键数据/参数表

### 3.1 参数量拆解（实测 `sum(p.numel())`）

| 模块 | 参数 | 占比 | 是否在分离前向路径上 |
|---|---|---|---|
| `separator.audio_net`（AudioBlock，shared×2 次调用） | **445,766** | 82.9% | ✅ |
| `separator.video_net`（VideoBlock） | 42,850 | 7.97% | ✅ |
| `spk_pool_proj`（对比头） | 14,584 | 2.71% | ❌ 只用于对比损失 |
| `audio_bottleneck` | 13,104 | 2.44% | ✅ |
| `mask_generator` | 12,657 | 2.35% | ✅ |
| `separator.caf`（`v_proj`） | 3,696 | 0.69% | ✅ |
| `encoder`（Conv2d 3→112） | 3,024 | 0.56% | ✅ |
| `decoder`（ConvTranspose2d） | 2,016 | 0.37% | ✅ |
| `video_bottleneck` | **0** | 0 | ✅（恒等） |
| **合计** | **537,697** | 100% | |
| 持久化 buffer（BatchNorm1d(32) 的 running_mean/var/n） | 65 | — | state_dict 求和 537,762 |
| 非持久化 buffer（`mdcl_window`/`encoder.window`/`decoder.window`，各 256） | 768 | — | 不进 state_dict |
| 冻结的 FRCNN 视频前端（`requires_grad=False`） | **11,186,688** | — | ✅（每帧特征提取） |

`audio_net` 内部：`gateway 225 / projection 7,489 / downsample 6,656 / process_f 81,920 / process_t 177,920 / process_att 119,732 / fusion 22,272 / concat 22,272 / residual 7,280`。
**参数重心在 `process_t_layers`（177,920，33%）**——即"沿 T 的 LSTM"，因为它要迭代 `T` 步。

### 3.2 死参数（实测证明，不是推断）

| 项 | 参数 | 证据 |
|---|---|---|
| `concat_layers[2]` | **5,568** | ① 前向 hook 计数 = 0（§2.3 表）；② 同权重清零后输出 **bit-identical**（`max|dOut|=0.000000`）；③ 结构原因：`:870` 用 `[-1]`，`:881` 的 range 只到 `[1,0]` |
| 对照组 `concat_layers[1]` 清零 | — | `max|dOut|=0.440951`（活的） |
| 对照组 `fusion_layers[2]`/`[3]` 清零 | — | `0.223430` / `0.263806`（活的） |
| `FiLMBlock` / `PositionalEncoding` / `MultiHeadSelfAttention` / `FeedForwardNetwork` / `GlobalAttention` | 0 | 类级死代码，从未实例化 |
| `video_params.{kernel_size,stride,upsampling_depth,layers,num_heads,head_dim,expand_v,use_*}` | 0 | VideoBlock 收下不用；`kernel_size=3` 与三处 k=1 conv 不符 |
| `fusion_params.{fusion_type,kernel_size,is2d}` | 0 | Separator 只读 `fusion_shared` |
| `mask_generation_params.mask_generator_type` | 0 | 落进 `MaskGenerator(**kwargs)` 被丢弃 |

⇒ **有效参数 = 537,697 − 5,568 = 532,129**（与 `docs/STATUS.md:33` 的 532,129 一致），再扣对比头 14,584 = **517,545**（分离前向真正用到的）。

### 3.3 MACs 表（ptflops，B=1，2 s@16 kHz，T_v=50，CPU）

| 模块 | MACs | 占比 |
|---|---|---|
| `separator`（含 2×audio_net + caf） | **9,597.85 M** | 90.5% |
| ├ `audio_net` 单次调用 | 4,797.71 M | — |
| └ ×2 次 | 9,595.42 M | |
| `audio_bottleneck` | 424.29 M | 4.0% |
| `mask_generator` | 424.29 M | 4.0% |
| `encoder`（含 STFT） | 98.69 M | 0.93% |
| `decoder` | 65.80 M | 0.62% |
| **音频主干合计（ptflops 返回值）** | **10,610.93 M = 10.6109 G** | 100% |
| 冻结 FRCNN 视频前端（@88×88、50 帧，另算） | **15,896.7 M = 15.897 G** | — |
| **整窗合计（音频 + 视觉，同尺）** | **26,507.7 M = 26.51 G**（视觉占 60.0%） | |

> 口径说明：`10.6109 G` 是 `get_model_complexity_info(model, (wav, mouth), input_constructor=...)` 的**返回值**，与 `docs/STATUS.md:33`、`FINAL_PROJECT_UNDERSTANDING.md:272` 的 10.611 G 逐位一致。
> 2026-11-08 Lead 复核更正：本表原写"FRCNN 7,769.3 M / 端到端 18.380 G"，7,769.3 M 是 FRCNN 在 **64×64** 输入下的
> 历史测量（05 §3.2 保留该行作对照），本管线真实输入是 88×88×50 帧，对应 15.897 G；18.380 G 是两把尺子混算的结果，
> 不再使用。同模型换 `torch.utils.flop_counter` 得 5.382 GMAC（不记 `aten::lstm`），与上表不可并列。
> **ptflops 的 `print_per_layer_stat=True` 打印树不可用于求和**：树里 `audio_net` 显示 8.05 GMac、根节点显示 9.06 GMac，而返回值是 10.61 GMac（共享模块被调用两次时打印树会失真）。要引用就引用返回值，或像我这样逐子模块单独测（本表即逐子模块测，合计 10.6109 G，与整体返回值完全吻合）。

### 3.4 延迟（CPU，单窗 2 s，B=1）

| 线程 | median | min | p90 | RTF |
|---|---|---|---|---|
| 8 | **357.3 ms** | 339.6 ms | 423.9 ms | 0.179 |
| 1 | **962.2 ms** | 953.3 ms | 970.2 ms | 0.481 |

逐段（1 线程，B=1）：`encoder 5.0 ms / audio_bottleneck 41.7 ms / separator 985.6 ms（92%）/ mask_generator 26.5 ms / decoder 18.9 ms / 合计 1067.2 ms`；视频前端（2 s=50 帧）35.8 ms。
⇒ 单窗算力瓶颈 **完全在 separator 的两次 AudioBlock**，与 MACs 结论一致。

### 3.5 真实权重对照

| 项 | `falcon.pt` | `falcon-big.pt` |
|---|---|---|
| 文件大小 | 47,156,343 B | 49,645,087 B |
| 顶层键 | `state_dict / architecture / epoch` | 同 |
| `architecture` | `FalconSpeakerContrastive` | `SpeakerContrastiveV2` |
| `epoch` | 714 | 715 |
| 总 key 数 | 662（audio 525 + video 137） | 654（audio 517 + video 137） |
| audio numel | 537,762 | — |
| video numel | 11,196,308（= 参数 11,186,688 + FRCNN 内 BatchNorm 统计量 9,620） | — |
| 构造配置 | `configs/Falcon.yml` | `studio/configs/FalconBig.yml` + studio `SpeakerContrastiveV2` |
| **`strict=True`** | ✅ OK | ✅ OK |
| 参数量 | 537,697 | **1,158,617** |

**small vs big 结构差异（逐 key 实测）**：
1. 宽度：`encoder/audio_bottleneck/mask/decoder` 112→256；`hid_chan` 64→96；`layer_1.hid` 24→48；`layer_2.hid` 72→96；`video hid` 32→64；`spk_embed_dim` 64→128。
2. **唯一"少 key"的差异**：big **没有** `process_f_layers.*.out_proj.weight` / `process_t_layers.*.out_proj.weight` 共 **8 个** key。原因：`DualPathRNN` 只在 `rnn_out_chan != in_chan` 时建 `Linear`（`Falcon.py:379-383`）。big 里 `48×2 = 96 = in_chan`、`96 = 96` ⇒ `out_proj = Identity`。small 里 `24×2=48≠64`、`72≠64` ⇒ 需要投影。
3. `video_model.*`（137 key）**两者完全同名同形**，且**权重值逐张量 bit-equal**（实测 `all(torch.equal(...))`=True）⇒ 视频前端是同一个冻结 FRCNN。
4. 交叉装载**都失败**（big ckpt→small 模型 / small ckpt→big 模型），报 shape mismatch，符合预期。

---

## 4. 实测验证

所有脚本在 `dev_a800_2gpus:/root/tmp/w1_model/*.py`（**仓库外**，未写入仓库），日志 `*.log`。统一前缀：
```bash
ssh dev_a800_2gpus 'cd /root/tmp/w1_model && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=/root/code/github_repos/Falcon-fork /root/miniforge3/envs/falcon/bin/python <script>'
```

**E1 逐层 shape**（`w1_shapes.py` → `trace.txt`）：给所有叶子模块 + 关键容器挂 forward hook，`torch.no_grad()` 跑一次 `model(torch.randn(1,32000), torch.randn(1,512,50))`。
```text
OUT (1, 1, 32000)
encoder.conv.conv     Conv2d  (1,3,251,129)   → (1,112,253,129)
encoder.conv.norm     Identity (1,112,251,129) ← 已 trim 2 帧
separator.audio_net.downsample_layers.1.conv  (1,64,251,129) → (1,64,126,65)
separator.audio_net.process_f_layers.3.rnn    LSTM  (17,32,64) → (17,32,48)
separator.audio_net.process_t_layers.3.rnn    LSTM  (32,17,64) → (32,17,72)
separator.audio_net.process_att_layers.3.Queries.0.conv (1,64,32,17) → (1,4,32,17)
separator.audio_net.process_att_layers.3.Values.0.conv  (1,64,32,17) → (1,16,32,17)
separator.video_net.projection.conv  Conv1d (1,512,50) → (1,32,50)
separator.video_net.rnn              LSTM   (1,50,32) → (1,50,32)
separator.caf.v_proj                 Conv1d (1,512,50) → (1,112,50)
mask_generator.mask_generator.1.conv Conv2d (1,112,251,129) → (1,112,251,129)
mask_generator                        (1,112,251,129) → (1,1,112,251,129)
decoder.decoder  ConvTranspose2d      (1,112,253,129) → (1,2,253,129)
decoder                               (1,1,112,251,129) → (1,1,32000)
```

**E2 参数/死参数**（`w1_params.py`, `w1_dead.py`）：见 §3.1/§3.2。死参数关键 stdout：
```text
zero concat_layers[2]   max|dOut|=0.000000  bit_equal=True     ← 死
zero concat_layers[1]   max|dOut|=0.440951  bit_equal=False
zero fusion_layers[2]   max|dOut|=0.223430  bit_equal=False
zero fusion_layers[3]   max|dOut|=0.263806  bit_equal=False
```

**E3 三条 forward 路径**（`w1_params.py`）：
```text
mouth=None     -> TypeError: conv1d() received an invalid combination of arguments - got (NoneType, ...)
mouth omitted  -> TypeError: 同上
```
⇒ **不存在 audio-only 前向**。`Falcon.forward` 没有 `if mouth_embedding is None` 分支；`video_bottleneck` 是 `nn.Identity`，直接把 `None` 透传给 `VideoBlock.gateway` 的 `Conv1d`。唯一的"缺视频"兜底是 `_encode_audio` 里的 dummy（见 §5-6）。**video-only 路径根本不存在**（没有视频单独的分离输出）。
另外 `mouth=zeros(1 帧)` 与 `zeros(50 帧)` 输出 `max|Δ| = 0.000000`；`zeros(50)` 与真实视频输出差 `max 0.00973 / rms 0.00185`（白噪声输入，仅作量级参考）。

**E4 MACs**（`w1_macs2.py`）：
```text
encoder (STFT+conv)   MACs=     98.69 M
audio_bottleneck      MACs=    424.29 M
separator (2x audio_net) MACs= 9597.85 M
mask_generator        MACs=    424.29 M
decoder               MACs=     65.80 M
SUM of parts          MACs=   10.6109 G
WHOLE MODEL (ptflops return) MACs= 10.6109 G  params=537,697
audio_net (single call) MACs= 4797.71 M  -> 2 x 4.7977 G = 9.5954 G
FRCNN video MACs = 7.7693 G
```
> 口径：最后一行是 FRCNN 在 **64×64** 输入下的测量（沿用 `evaluated_mac_params.py` 习惯）；真实管线是
> 88×88×50 帧，对应 **15.897 G**，见 §3.3 与 03 §4-J。两数不可混用。

**E5 因果性**（`w1_causal.py`, `w1_causal2.py`）：
```text
# 输入 q 之前相同、q 之后换成 3× 噪声
q=16000: first differing output sample = 1 (0.1 ms); #diff samples=31912; max|d|=2.083e+00
q=24000: first differing output sample = 1 (0.1 ms); #diff samples=31610; max|d|=2.140e+00
# 单点在 p=16000 加 +1.0
affected output range: [13, 31999]  (p-15987 .. p+15999)  max|d|=7.748e-02
# cLNhw 微观证明
perturb frames 200:251 -> 第 0 帧输出变化 max|diff frame0| = 7.152557373046875e-07
GroupNorm spec: num_groups=1 num_channels=112 -> 归一化 112*251*129 = 3,626,448 个元素
# 窗口长度敏感
2s 窗口 vs 4s 窗口的前 2s: max|d|=0.2249  rel=0.323  argmax@5931 (0.37 s)
# 无跨窗状态
同一输入两次（中间插入别的输入）: bit_equal = True
```
结论见 §5-1。

**E6 真实权重**（`w1_weights.py`, `w1_bigkeys.py`）：见 §3.5。

**E7 Studio 一致性**（`w1_parity.py`）：
```text
state_dict keys equal: True | params: 537,697 vs 537,697
max|Δ| trained-vs-studio = 0.0 | bit_equal = True
```

**E8 解码器对齐**（`w1_align.py`, `w1_align2.py`, `w1_align3.py`）：
```text
# 隔离校准：torch.istft 自身对 DC 单帧的峰位 = frame*hop - 47（固定约定，对所有帧相同）
raw torch.istft: DC spike @frame 102 -> peak 13009 (frame*hop=13056, diff=-47)
# delta 权重（w[0,0,1,1]=w[1,1,1,1]=1）+ 把 stft(spec) 直接灌进 decoder：
causal=True : best lag of out vs ideal-istft = +256 samples (+16.0 ms), corr=1.0000
causal=False: best lag of out vs ideal-istft = +0 samples (+0.0 ms),   corr=1.0000
# 真实权重端到端（白噪声）：
real weights: lag=+0 samples (+0.0 ms), corr=0.3929
```

**E9 `n_src=2` 实测崩溃**（`w1_params.py`）：
```text
n_src=2: params: 552,369 | mask conv out chan: 224 | decoder in_chan: 224
RuntimeError: shape '[2, 224, 251, 129]' is invalid for input of size 7252896
  at Falcon.py:1175  x = x.view(batch_size*self.n_src, self.in_chan, *x.shape[-2:])
```
原因：`__apply_masks` 的 `RI_split` 分支把 `n_src*in_chan` 通道解释成"每源 2×56"，输出**每源仍是 112 通道**；而 `decoder.in_chan` 被写成 `encoder_out_chan * n_src = 224`（`Falcon.py:1345`）。n_src=1 时两者数值巧合相等（112=112），所以只有 n_src=1 能跑。

**E10 真实权重前向 sanity**（`w1_final.py`）：
```text
in rms=0.0997  out rms=0.0019  out/in=0.019        ← 白噪声输入，模型近乎静音输出（掩码 98.9% 为 0）
mask zero fraction = 0.9888 | mean=0.00161 | p99=0.0151 | max=1.740
refined: |real video - zero video| max=3.52067 |zero1帧 - zero50帧| max=2.980e-06
encoder 5.0 ms / audio_bottleneck 41.7 / separator 985.6 / mask 26.5 / decoder 18.9 / FULL 1067.2 (1 线程)
```
> 白噪声不是语音，`out/in=0.019` **不能**当作质量/增益结论——干净语音上的量级需要真实数据（已入 GAP）。

---

## 5. 坑与反直觉点

1. **`causal=true` 是"名义因果"，实际窗口内全局耦合**（`Falcon.py:693-898` + `layers/normalizations.py:134-146`）。
   `causal` 只被用于：① `encoder.conv` 的 causal padding（`Falcon.py:61`）、② `STFTDecoder` 的前 pad/尾裁（`:1178-1184`）、③ `DualPathRNN`（T 轴 LSTM 本来就单向）、④ `MultiHeadSelfAttention2D` 的 tril mask（`:518-524`）。
   但 **`norm_type=cLNhw` 出现在 `audio_bottleneck.pre_norm`、`projection.norm`、`downsample_layers[*].norm`、`fusion_layers[*].*.norm`、`concat_layers[*].*.norm`**，每一个都是 `nn.GroupNorm(1,C)` 全窗口统计 ⇒ 实测"扰动后段、第 0 帧就变"。**结论：帧级因果不成立，只有"窗口级因果"（窗与窗之间无状态、无跨窗注意力）成立**，且单窗内需要整窗数据才能算。
2. **`downsample_layers` 全部非因果**（`Falcon.py:772` 硬编码 `causal=False`，k5/s2/p2 对称）⇒ 各级时间前瞻 ±2 帧，level 3 累积 ≈ ±1792 采样（112 ms），与整链编码器阶梯 15 帧 / 120 ms 同量级（三档口径见 12 §3）。
3. **`fusion_layers`/`concat_layers` 也硬编码 `causal=False`**（`:808`、`:821`），k=5 对称 pad ⇒ 又一层 ±2 帧前瞻。
4. **`AudioBlock` 把 T 和 F 一起下采样**（k5/s2 同时作用于 H、W）。想只压频率的人会得到完全不同的感受野；`n_freqs=129`（`Falcon.py:706`）只是为了给 attention 的 `LayerNormalization4D` 算参数形状（`(1,C,1,F)`，`normalizations.py:150-152`），**换 win/hop 得同时手改 `n_freqs`，否则 norm 参数形状与真实 F 不符**。
5. **`concat_layers[2]` 是死的**（5,568 参数 = 1.04%）。不是"少用"，是"从未调用"（bit-equal 实测）。
6. **`_encode_audio` 的 dummy video 不是"零视觉"**（`Falcon.py:1356-1371`）：
   `video_bottleneck(zeros(B,512,1))` = `zeros(B,512,1)`（Identity），但 `VideoBlock` 里有 bias/PReLU/BatchNorm/LSTM ⇒ 输出 **norm=9.72、逐帧完全相同**的非零向量；`caf` 把它广播成**频率无关的常数偏置**加到音频特征。
   实测它与 `zeros(512,50)` 在特征上差 2.98e-6、在输出上差 0.000000 ⇒ **数值上无害**，但语义上"视觉分支仍在贡献一个学习到的常数"，且**LSTM 只跑 1 帧**（隐状态为 0 初值、无时序上下文）。如果将来把视频分支训练得更有表达力，这个常数的分布偏移会被放大——**这是 MDCL/说话人对比路径上的隐患**（`_encode_audio` 被用来给 near/target 音频算说话人嵌入）。
7. **融合不是 cross-attention，也不带任何视觉掩码**。文档/口头描述里的 "cross-attention"、"视觉引导的 mask" 在本代码里**不存在**（§6-1）。
8. **`Separator` 的遍数取自 video 配置**：`fusion_repeats = video_params["repeats"]`（`:1000`），`audio_repeats = audio_params["repeats"] - fusion_repeats`。把 `video_params.repeats` 当成"视频分支重复次数"会直接改变音频主干的运行遍数。
9. **`shared=true` 让"2 次 pass"共享同一份权重**，所以参数量不变但 MACs 翻倍；`docs` 里"0.5 M 参数/10.6 G MACs"的失衡感正来自此（参数 0.54 M，算力 10.6 G，比率 19.7 kMAC/param）。
10. **`n_src≥2` 结构上不可用**（§4-E9）。`MaskGenerator` 的通道账（`n_src*in_chan` 被复数拆成 2×56）与 `decoder.in_chan=encoder_out_chan*n_src` 不自洽。
11. **`STFTDecoder` 有 +2 帧（+256 采样 / 16 ms）系统延迟**（§4-E8）。`causal` 分支 pad `k-1=2` 帧、只裁 `(k-1)//2=1` 帧，净 +1 帧长度误差 + 2 帧位移。发布权重端到端 lag=0，说明掩码把这个位移学掉了——**一旦重训/换解码器/换 hop，这个 16 ms 会以"看得见的错位"形式回来**。
12. **`VideoBlock` 的 config 有 8 个键不生效**，`kernel_size: 3` 与实现（全 k=1）矛盾（`Falcon.py:900-976`）。
13. **`mdcl_window` 是 `persistent=False`**（`Falcon.py:1350-1354`）⇒ 不进 state_dict，`from_pretrain` 重建时依赖 `enc_dec_params["win"]` 重新生成；`enc_dec_params` 缺失就崩。
14. **`Falcon.forward` 有一个未使用的参数 `target_audio`**（`:1481`）——签名里留着，函数体从不读。
15. **`_mdcl_loss = _mdcl_loss_hard` 是类属性别名**（`:1478`），且只在 `self.training and near_audio is not None` 时返回 `(audio, loss)` 二元组（`:1493-1497`）——**推理时返回张量、训练时返回元组**，调用方要区分（`Falcon_Contrastive.forward` 又是另一套：训练返回 `(audio, emb)`，eval 返回 `audio`）。

---

## 6. 与既有文档的冲突 / 更正

| # | 旧说法 | 实际（本轮实测/行号） | 处置 |
|---|---|---|---|
| 1 | （任务书/口头）"cross-attention 视听融合，Q/K/V 来自音频与视频" | **不存在**。融合是 `audio + v_proj(video)↑unsqueeze(-1)` 的加性偏置（`Falcon.py:250-267`）。全模型只有音频内部的**自**注意力（`MultiHeadSelfAttention2D`），且只在 level 3 | 更正；以后描述为"additive visual bias injection" |
| 2 | `docs/STATUS.md:33` "音频 537,697（生效 532,129；扣对比头 523,113）" | 537,697 ✅、532,129 ✅（= 537,697 − 5,568 死参数）。但 **523,113 的基准错了**：`523,113 = 537,697 − 14,584`（从**总参数**扣对比头），而不是 `532,129 − 14,584 = 517,545`（从**生效参数**扣）。三个数各自都有来源，但"生效→扣对比头"这一步的基准不一致 | 澄清：生效 532,129 → 扣对比头应为 **517,545**；523,113 是"总参数 − 对比头" |
| 3 | `docs/STATUS.md:33` "MACs 10.611 G" | ✅ 逐位复现（10,610,934,096），口径是 ptflops、含 LSTM、仅音频主干。含 88×88×50 帧视觉的整窗为 **26.51 GMAC**（视觉 15.897 G） | 已补口径；18.380 G 作废（64×64 老口径混算） |
| 4 | `docs/reviews/round1/01:83-93`、`round2/01:53-63` "`concat_layers[2]` 从未被调用，5,568 参数闲置" | ✅ **成立**，本轮独立复现并升级为"同权重清零 → 输出 bit-identical"的强证明 | 确认 |
| 5 | （既有 deepdive 的因果性表述）"窗口级因果 / 因果模型" | **成立但需要限定**：帧级/采样级因果**不成立**（实测输出第 1 个采样点就依赖后段输入，因 `cLNhw=GroupNorm(1,C)` 全窗统计）。只有"窗间无状态"这一层因果成立 | 更正为"window-causal, frame-noncausal" |
| 6 | `docs/reviews/round2/01:170` "falcon-big: 654 = audio 517 + video 137；params 1,158,617" | ✅ 全部复现。补充：**517 vs 525 的差 = 8 个 `out_proj.weight` 消失**，原因是 big 的 `hid*directions == in_chan` 使投影退化为 Identity（`Falcon.py:379-383`） | 补充机理 |
| 7 | `docs/REPRODUCTION_PHASE2_REPORT.md:16` "视觉分支贡献 +4.8~+15 dB" | 本轮**未复现该数字**（需要真实 LRS2 数据 + 评测脚本，属 W2/W5 范围）。本轮只测到：白噪声输入下真实视频 vs 零视频的输出绝对差 `max 0.00973 / rms 0.00185` | 标记为未复现（GAP-4） |
| 8 | `evaluated_mac_params.py` 的用法（`input_constructor` 传预计算 `v`） | ✅ 正确，且我用同样的 `input_constructor` 复现了 10.6109 G。注意该脚本 `audiomodel.cuda()` 强制 GPU，**CPU 复算需自己写**（本轮已写） | 补充 |

---

## 7. GAP LIST（可被下一轮直接认领）

| ID | 问题 | 建议做法 |
|---|---|---|
| GAP-1 | **权重噪声输入的 `out/in=0.019` 是否是"模型近乎静音"？** 需要真实 LRS2 语音（1 条即可）算输入/输出 rms、SI-SDR(input)、SI-SDR(output) | 用 `/root/data/lrs2_2mix` 取 4 条，跑 `audio_visual_test.py` 的模型加载路径，只做单条前向 |
| GAP-2 | **`+256 采样解码器延迟`在真实权重下是否真的被掩码补偿到 0？** 本轮 lag=0 的互相关只有 0.39，置信度不足 | 用真实语音 + 干净目标，测 `SI-SDR(y, s)` vs `SI-SDR(y, shift(s,±256))`，看哪一个更高 |
| GAP-3 | **`RI_split` 的实/虚子空间是否真的学到了复数结构？** 掩码 98.9% 为 0，且 0:56/56:112 的划分是任意的 | 真实语音上统计掩码两半的相位分布；做一次 `RI_split=false` 的消融训练（小步数）对比 |
| GAP-4 | **视觉分支贡献 +4.8~+15 dB 的口径与复现** | 交接 W2/W5：固定 ckpt，同一批样本 real-video vs zero-video 两次评测 |
| GAP-5 | **`cLNhw` 全局统计是否是训练刻意的？** 论文若声称 streaming 因果，这是硬伤 | 查论文 §3 与 upstream `JusperLee/Falcon` 原始实现（`upstream` remote 已有 `439cbe9`）是否同款 normalizations |
| GAP-6 | **`n_src≥2` 是否曾经能跑**（上游是否有不同写法） | diff `git show 439cbe9:look2hear/models/Falcon.py` 的 `MaskGenerator/STFTDecoder` |
| GAP-7 | **`VideoBlock` 是否本该是 DualPathRNN**：`video_params.layers` 里写了 `layer_1: {kernel_size:3, n_head:8, dropout:0.1, causal:true}`，像是曾经打算接注意力 | 看 upstream 历史版本 / 论文补充材料 |
| GAP-8 | **`process_f_layers` 用双向 LSTM 跨频率是否合理**（F 轴双向无因果问题，但 17 个频点跨 8 个倍频程，双向平滑可能抹掉共振峰） | 消融：F 轴改单向/去掉，看 SI-SDR |
| GAP-9 | **`audio_bottleneck` 的 `pre_norm=cLNhw` 在 1×1 conv 前做全窗 GroupNorm，等价于"用整窗能量归一化"**，这会让模型对窗内能量突变（爆音）敏感 | 真实数据上测不同 SNR/削波下的稳定性 |
| GAP-10 | **单窗延迟只有 CPU 数**（357 ms@8 线程）。GPU（A800/L4）上的单窗延迟与吞吐未测（本轮禁 GPU） | 交给允许 GPU 的轮次，`torch.cuda.Event` + batch 1/8 |

---

## 8. 证据清单

**仓库文件（只读）**
- `look2hear/models/Falcon.py`：11/111/241/269/333/419/547/565/620/664/693/900/978/1061/1130/1201/1284 行各类；`:61,104,159,200` causal padding；`:772,808,821,830` 硬编码 `causal=False`；`:870,881,884` concat 调用；`:1032-1059` Separator；`:1117-1126` STFT；`:1178-1184` 解码器 pad/trim；`:1251-1271` 复数掩码；`:1345` decoder in_chan；`:1356-1371` `_encode_audio`；`:1480-1499` forward。
- `look2hear/models/Falcon_Contrastive.py:8-57`（AttentionSpeakerPool）、`:60-129`。
- `look2hear/models/base_model.py:32-108`；`look2hear/models/__init__.py:20-44`。
- `look2hear/layers/normalizations.py:134-146`（GlobalLayerNorm=GroupNorm(1,C)）、`:150-166`（LayerNormalization4D）、`:167-197`（CumulateLN4D）、`:209-222`（cLNhw 别名）、`:231-233`（`get(None)→Identity`）。
- `look2hear/datas/avspeech_dataset.py:42-49,96-127`（25 fps、只有 `target.json`）。
- `look2hear/videomodels/frcnn_videomodel.py:55-85`（Conv3d 时间 pad=2、stride=1）。
- `look2hear/system/av_litmodule_nass_v2.py:74-90`（`mouth_emb` 必传）。
- `configs/Falcon.yml`、`configs/LRS2_2Mix*.yml`（`n_src: 1`）、`studio/configs/FalconBig.yml`。
- `studio/falcon_model/models/{Falcon.py,Falcon_Contrastive.py,SISNRRepel.py,Speaker_Contrastive_v2.py}`、`studio/falcon_runtime.py:21,36-38`。
- `evaluated_mac_params.py:76-100`（官方 MACs 口径）。

**实测脚本与日志（仓库外，`/root/tmp/w1_model/`）**
| 脚本 | 日志 | 内容 |
|---|---|---|
| `w1_build.py` | — | 构造 + 顶层参数 |
| `w1_spec.py` | — | 全部 conv/LSTM/Linear 的 k/s/p/g/d 与 causal 标记 |
| `w1_shapes.py` | `trace.txt` | 逐层 shape（hook） |
| `w1_params.py` | `params.log` | 参数分组、buffer、路径、n_src=2 |
| `w1_dead.py` | `dead.log` | 死参数清零 bit-equal 证明 + 调用计数 |
| `w1_causal.py` / `w1_causal2.py` | `causal.log` / `causal2.log` | 因果性 4 组实验 + conv pad 清单 |
| `w1_perf.py` / `w1_macs2.py` | `perf.log` / `macs2.log` | MACs、延迟、逐段计时 |
| `w1_weights.py` / `w1_bigkeys.py` | 直接输出 | 真实权重 strict 装载、key 差异 |
| `w1_parity.py` | `parity.log` | Studio vs 训练侧 bit-equal |
| `w1_align.py` / `w1_align2.py` / `w1_align3.py` | `align*.log` | 解码器 +256 采样延迟（delta 权重直通，corr=1.0） |
| `w1_final.py` | `final.log` | 真实权重 sanity、掩码统计、逐段计时 |
| `sha256sum` / `cmp` / `diff` | 直接输出 | `0d613d2b…`(train) vs `917e2bfc…`(studio)，3 类功能差异（import / MPS 分支 / 注释语言） |

**复现三条最关键数字的最小命令**
```bash
# 1) 10.6109 G MACs
cd /root/tmp/w1_model && CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python w1_macs2.py
# 2) concat_layers[2] 是死的（bit-equal）
CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python w1_dead.py
# 3) 解码器 +256 采样 / cLNhw 全窗耦合
CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python w1_align3.py
CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python w1_causal2.py
```
