# 01 核心模型架构（Falcon 音视频目标说话人提取网络）

> 状态：R1 初版 2026-10-21 → **R3 修订版 2026-10-21**（逐条响应 R2 `layers-dsp` + R4 `studio-backend` 两份评审，见 §9.1）。
> owner：`model-core` · 证据基线：commit `7576854` · 真实权重：`/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`（sha256 见 §8.4）。
> 证据标注：`[代码]` 直接读源码；`[验证]` 本机动态运行（脚本/日志见 §8）；`[推断]` 合理推断但未被代码直接证明。
> 行号均为当前 `main` 上的实际行号；跨文件证据给多条。
> **因果性统一措辞：窗口级因果（window-causal）**（定义见 §4.6-⑦，与 02 §4.2、06 §4.1/§5.5 一致）；R3 版 sha256 因文档自指不可内嵌，记录于 `docs/verification/model-core.md` 与汇报。

---

## 1. 子系统职责与总体位置

**一句话结论**：`look2hear/models/` 的 4 个文件（主体是 1487 行的 `Falcon.py`）定义了 Falcon 的**全部可训练参数**——
一个"因果"单次 encoder–decoder 的 AVTSE 网络。实测可训练参数 **537,697（0.538 M）**，加上冻结的 FRCNN 视觉编码器
**11,186,688** 后合计 **11,724,385（11.72 M）**，与 README 的 0.5 M / 11.72 M 口径一致（核算见 §4.7）。
**R3 发现**：其中 **5,568（1.04%）参数从不参与前向**（`concat_layers[2]`，疑似代码 bug），实际生效 532,129（§4.7/§4.9-㉑）。

**两条产品线如何共用这份模型定义**：

| 用途 | 入口 | 关键行 |
|---|---|---|
| 训练 | `python audio_train.py --conf_dir configs/Falcon.yml` | `audio_train.py:55-58` 用 `getattr(look2hear.models, "FalconSpeakerContrastive")(sample_rate=16000, **audionet_config)` |
| 评测 | `audio_visual_test.py` | `audio_visual_test.py:128-140` 用 `deepcopy(config)` 重建同一模型并 `load_state_dict(strict=False)` |
| Studio 推理 | `studio/falcon_runtime.py` | `studio/falcon_runtime.py:149-172` 用 `deepcopy` 重建 + `load_state_dict(strict=True)` |

**一句话数据流**（全部为 [代码]+[验证]，形状见 §3.2）：

```
波形 [B,L] 16 kHz
  └─ STFTEncoder：torch.stft(win=256,hop=128) → 3 通道 (real, imag, magnitude) → Conv2d(3→112,k=3)
       → [B,112,251,129]
  └─ audio_bottleneck：cLNhw + Conv2d(112→112,k=1) → [B,112,251,129]
  └─ Separator
       ├─ AudioBlock#1（4 级时频金字塔 + DualPathRNN(频/时) + MHSA2D）→ [B,112,251,129]
       ├─ VideoBlock（512→32→512 单向 LSTM，输入 [B,512,50]）→ [B,512,50]
       ├─ NeuroSCNetFusion：视频 512→112（groups=16）→ 时间 50→251 nearest → 加性注入
       └─ AudioBlock#2（与 #1 共享权重，输入 = 融合输出 + 音频瓶颈残差）→ [B,112,251,129]
  └─ MaskGenerator：PReLU + Conv2d(112→112,k=1) → 复数掩码（RI_split，56 实 + 56 虚）
       与编码器特征做复数乘法 → sep_audio_emb [B,1,112,251,129]
  └─ STFTDecoder：ConvTranspose2d(112→2,k=3) → 复谱 → istft(length=L) → est [B,1,L]
```

**边界**：本子系统**不含**损失、指标、数据管线、视觉编码器权重（`look2hear/videomodels/`，04 域）与层原语
（`look2hear/layers/`，02 域）。但 Falcon 的"因果性"结论必须把这三者一起考虑（§4.6）。

---

## 2. 文件清单与逐文件解析

### 2.0 文件清单

| 文件 | 行数 | 职责 | 关键点 |
|---|---|---|---|
| `look2hear/models/Falcon.py` | 1487 | 架构定义：17 个 `nn.Module` 类（13 在用 + 4 死代码）+ `Falcon` 主模型 + 3 个 MDCL 工具方法 | 单文件承载全部可训练参数 |
| `look2hear/models/Falcon_Contrastive.py` | 122 | `FalconSpeakerContrastive(Falcon)` + `AttentionSpeakerPool`：同空间说话人对比头 | 训练期多返回一个 `sep_audio_emb` |
| `look2hear/models/base_model.py` | 102 | `BaseModel`：`sample_rate`、序列化、`from_pretrain`、`load_state_dict_in_audio` | 不含任何可训练参数 |
| `look2hear/models/__init__.py` | 41 | 导出 `Falcon`/`FalconSpeakerContrastive` + `get()` 按名取类 | `__all__` 内容与实际导出不一致 |

`look2hear/` 无 `__init__.py`（PEP 420 namespace package），`look2hear.models.Falcon` 作为**属性**被同名类遮蔽（§4.9-⑨）。

### 2.1 `Falcon.py` 的构造原语

#### 2.1.1 `ConvNormAct`（`Falcon.py:9-107`，1D/2D 通用卷积块，**模型里共 38 个实例**）

- `__init__(in_chan, out_chan, kernel_size=-1, stride=1, groups=1, dilation=1, padding=None,
  pre_norm_type=None, pre_act_type=None, norm_type=None, act_type=None, xavier_init=False, bias=True, is2d=False, causal=False)`
- **`kernel_size <= 0` 时退化为 `nn.Identity`**，且 `out_chan := in_chan`（`Falcon.py:30,67,85`）。这一条是 `video_bn_params.kernel_size: -1`
  能"什么都不做、0 参数"的原因（§3.1）。
- 前向顺序固定为：`pre_norm → pre_act → conv → (因果裁剪) → norm → act`（`Falcon.py:97-107`）。注意 pre 分支在 conv **之前**、主分支在 **之后**。
- **因果实现**（`Falcon.py:45-63,100-104`）：`causal=True` 时 `causal_padding = dilation*(k-1)`，作为**左侧** padding 用到 Conv 上，
  forward 后再裁掉**右侧** `causal_padding` 列/行。`is2d=True` 时 padding 元组是 `(causal_padding, padding)`——
  即**时间维（H）左填充 + 频率维（W）对称填充**（`Falcon.py:59`）。这一点是判断"哪个轴是时间"的关键：4D 张量约定为 `[B,C,T,F]`。
- 无任何参数共享/LoRA 逻辑；`groups=in_chan` 即 depthwise（模型里大量使用）。

#### 2.1.2 `ConvActNorm`（`Falcon.py:109-203`，仅被 `MultiHeadSelfAttention2D` 使用）

与 `ConvNormAct` 的差别只有两点：顺序为 `conv → (因果裁剪) → act → norm`（`Falcon.py:195-203`），且 `norm_type="LayerNormalization4D"`
时会把 `(out_chan, n_freqs)` 传给归一化层（`Falcon.py:186-188`）——因此**该模块必须知道当前层的频率维大小**，这是 `AudioBlock` 里
`att_config["n_freqs"] = current_f` 的由来（`Falcon.py:785-786`）。

#### 2.1.3 `NeuroSCNetFusion`（`Falcon.py:239-265`，音视频融合，配置键 `fusion_params`）

- 职责：把视频特征在**通道维**投影到音频通道数，再在**频率维广播加性注入**音频特征。
- 结构：`ain_chan != vin_chan` 时 `nn.Conv1d(vin_chan→ain_chan, k=1, groups=gcd(vin,ain))`，否则 `Identity`
  （本配置 112 vs 512，`gcd=16`，`Falcon.py:242-246`）。
- 形状：`audio [B,C_a,T_a,F]`, `video [B,C_v,T_v]` → `v_proj` → `[B,C_a,T_v]` → 若 `T_v≠T_a` 用 `F.interpolate(mode="nearest")` 拉伸到 `T_a`
  → `unsqueeze(-1)` 成 `[B,C_a,T_a,1]` → 与 audio 相加广播到 `F`（`Falcon.py:254-265`）。
- 参数：3,696（= 512·112/16 + 112）。**因果性**：nearest 上采样只取过去/当前视频帧 + 逐通道 1×1 卷积，**时间上严格因果**（[验证] 前缀扰动 Δ=0，§8-[D]）。
- 命名与实际不符：类名叫 "NeuroSCNetFusion"，实际就是**加性 FiLM-free 注入**；真正的 FiLM 版本 `FiLMBlock` 被整段注释掉（`Falcon.py:205-237`）。

#### 2.1.4 `InjectionMultiSum`（`Falcon.py:267-329`，跨层级 skip 融合；**实例化 8 个（fusion 4 + concat 4），前向实际调用 7 个**）

- 职责：把"全局（深层低分辨率）特征"注入到某层的 local 特征上，做 `local_emb * gate + global_emb`（`Falcon.py:327`）。
- 结构：3 个 `ConvNormAct`，全部 `groups=in_chan`（depthwise）、`bias=False`、`kernel_size=kernel_size`（配置 5）：
  `local_embedding`、`global_embedding`、`global_gate`（`act_type="Sigmoid"`）（`Falcon.py:283-313`）。参数 5,568 = 3×(64·25 + cLNhw(64))。
- 形状：`local [B,C,T_l,F_l]`、`global [B,C,T_g,F_g]`；`new_shape/old_shape` 取末 2 维（`Falcon.py:315-316`）；全局分辨率更小时
  先把 global 特征 `interpolate` 到 local 分辨率再卷积+插值（`Falcon.py:319-325`），最后返回与 local 同形状。
- **因果性（R3 修正：必须带 cLNhw 口径，见 A21a-c）**：`causal=False`（`Falcon.py:801-815` 构造时硬编码），`kernel_size=5` 对称 padding。
  **as-built（含 3 个 cLNhw）脉冲响应是全局 (0,250)**——cLNhw 的 `GlobalLayerNorm` 把单点扰动摊到整窗；把 cLNhw 换成 `Identity` 后才是
  **±2 帧 `[98,102]`**；`causal=True`+去掉 norm 则严格因果（前缀 Δ=0，脉冲 `[100,104]`）。这是模型里最容易被忽视的非因果点之一（§4.6）。

#### 2.1.5 `DualPathRNN`（`Falcon.py:331-415`，频/时双路径 RNN，被用 8 次）

- 职责：在"频率轴"或"时间轴"上做 RNN + 残差。`dim` 决定轴向：`dim=4` 表示沿第 4 维（F）处理，`dim=3` 表示沿第 3 维（T）处理（`Falcon.py:386-392`）。
- 机制（`dim=4`，频率路径）：`x [B,C,T,F]` → `transpose(-2,-1)` → `[B,C,F,T]` → 把 `T` 当 batch、序列轴是 `F`：`view(B*T, C, F)`。
  所以**时间维变成 batch 维，双向 RNN 也不会引入时间上的未来依赖**（[验证] 前缀扰动 Δ=0）。配置 `layer_1: dim=4, bidirectional=true`。
- 机制（`dim=3`，时间路径）：`[B,C,T,F]` → `permute(0,3,1,2)` → `[B,F,C,T]` → `view(B*F, C, T)`，序列轴就是时间；
  配置 `layer_2: dim=3, bidirectional=false` ⇒ **单向 LSTM**（[验证] 前缀扰动 Δ=0、脉冲响应单帧）。
- 归一化：`LayerNormalization4D((in_chan,1))`（`Falcon.py:370`）——`input_dimension[-1]==1` 使 `self.dim=(1,)`，即**只在通道维归一化、逐 (T,F) 位置**，因此是因果的
  （`look2hear/layers/normalizations.py:146-163`）。
- 输出投影：`rnn_out_chan = hid_chan*num_direction`；当 `≠ in_chan` 时用 `nn.Linear` 投影回 `in_chan`（`Falcon.py:367-383`）。
  配置下：频率路径 64→(24×2=48)→线性回 64；时间路径 64→72→线性回 64。
- **死参数**：`kernel_size=8`、`stride=1`、`num_heads=4`、`head_dim=12`、`expand_v`、`use_output_gate`、`use_short_conv`、
  `use_output_correction`、`use_inner_decay` 全部只赋给 `self.*`、从不参与建图（`Falcon.py:336-353,355-365`）。代码里没有任何短卷积/门控实现。
- **死分支**：`if "Comba" in self.rnn_type:`（`Falcon.py:394-397`）不可达——因为构造时 `getattr(nn, self.rnn_type)`（`Falcon.py:372`）要求 `rnn_type` 必须是
  `torch.nn` 里的 RNN 类名，"Comba" 不是（配置里是 `LSTM`）。该分支想用的自定义模块根本不存在。
- **`self.causal` 从不被读取**（`Falcon.py:365` 只赋值）——见 §4.9-⑥。

#### 2.1.6 `MultiHeadSelfAttention2D`（`Falcon.py:417-543`，逐层 2D 自注意力，被用 4 次）

- 职责：在 `[B,C,T,F]` 上做多头自注意力，注意力维度是**时间**（softmax 沿 T），每头用 1×1 深度可分离卷积产生 Q/K/V。
- 结构（`Falcon.py:441-492`）：`n_head=4` 个 `ConvActNorm` 分别产生 Q（C→hid_chan=4）、K（C→4）、V（C→C/n_head=16），
  外加输出投影 `attn_concat_proj`（C→C）；全部 `is2d=True, kernel_size=1, norm_type=LayerNormalization4D`。
- 形状推演（`Falcon.py:494-543`）：Q/K/V 各自 `[B,4,T,F]`/`[B,16,T,F]` → 沿 **batch 维 `cat`** 成 `[B*n_head,...]`
  → `Q: [B',T,4F]`、`K: [B',T,4F]`、`V: [B',T,16F]` → `attn = QKᵀ/√(4F)` → `softmax(dim=-1)` → `[B',T,T]` → `V' = attn@V` → 重组回 `[B,C,T,F]` + residual。
  注意：**embedding 维度把频率维也算进去了**（`emb_dim = hid_chan*F`，config 下 4×129=516），所以分布随 F 变化。
- **因果性**：`causal=True`（配置 `layer_3.causal: true`）时施加 `torch.tril` mask（`Falcon.py:520-524`），[验证] 前缀扰动 Δ=0；
  `causal=False` 时 Δ=4.68（§8-[D]）。但注意力在**整段 T 上一次性计算**（O(T²)、无 KV cache），因此**数学因果但非流式**（§4.6-⑥）。
- 注意 `Queries/Keys/Values` 是用 `nn.ModuleList` 拼的**每头独立卷积**（不是一次 3C 投影后 split），参数因此比标准 MHA 少很多但也更碎。

#### 2.1.7 `PositionalEncoding` / `MultiHeadSelfAttention` / `FeedForwardNetwork` / `GlobalAttention`（`Falcon.py:545-689`，**整组死代码**）

[验证] 构造出的模型 `named_modules()` 里不含这四个类（§8-[B] "Falcon.py 中定义但未出现在模型里的类"）。
它们只被彼此引用（`GlobalAttention` 组合另外三个，`Falcon.py:683-684`），而 `GlobalAttention` 从未被实例化 ⇒ **整条链不可达**。
连带后果：`from timm.models.layers import DropPath`（`Falcon.py:6`）在本仓库里**只服务于这两个死类**（`Falcon.py:590,651`），
是 `timm` 成为硬依赖的唯一原因；Studio 副本因此能把它替换为本地 `drop_path.py`（`studio/falcon_model/PROVENANCE.md`）。

### 2.2 `AudioBlock`（`Falcon.py:691-896`）——参数与算力的主体（445,766 / 537,697 = 82.9%）

**职责**：在 `[B,112,T,F]` 上做 4 级时频金字塔的下采样 → 深层处理 → 逐级上采样融合，输出同形状特征。它是 Falcon 的"分离器主干"。

**构造（`Falcon.py:718-824`）**：

| 子模块 | 定义 | 参数 | 说明 |
|---|---|---|---|
| `gateway` | `ConvNormAct(112→112,k=1,groups=112,act=PReLU)` | 225 | 深度可分离门（其实无门控，仅 1×1 + PReLU） |
| `projection` | `ConvNormAct(112→64,k=1,norm=cLNhw,act=PReLU)` | 7,489 | 通道压缩 112→64 |
| `downsample_layers[i]` ×4 | `ConvNormAct(64→64,k=3(i=0)/5,stride=1(i=0)/2,groups=64,norm=cLNhw,causal=False)` | 896+3×1920 | **stride 是 int，Conv2d 会同时下采样 T 和 F** |
| `process_f_layers[i]` ×4 | `DualPathRNN(dim=4, hid=24, biLSTM)` | 20,480×4 | 频率轴 RNN |
| `process_t_layers[i]` ×4 | `DualPathRNN(dim=3, hid=72, 单向LSTM)` | 44,480×4 | 时间轴 RNN |
| `process_att_layers[i]` ×4 | `MultiHeadSelfAttention2D(n_head=4, hid_chan=4, causal=True)` | 51,693/31,213/20,973/15,853 | 逐层 2D 注意力 |
| `fusion_layers[i]` ×4 | `InjectionMultiSum(64,k=5,causal=False)` | 5,568×4 | 把最深层"全局特征"注入各层 skip |
| `concat_layers[i]` ×4 | `InjectionMultiSum(64,k=5,causal=False)` | 5,568×4（**其中 `concat[2]` 从不被调用，5,568 闲置**） | 解码器逐级融合；调用索引为 `[3],[1],[0]`，见 §4.9-㉑ |
| `residual_conv` | `ConvNormAct(64→112,k=1)` | 7,280 | 输出通道还原 + 残差 |

**形状推演（[验证] §8-[C]，B=1, L=32000）**：

| 阶段 | T | F | 备注 |
|---|---|---|---|
| 输入 | 251 | 129 | `T=1+⌊L/hop⌋`, `F=win/2+1` |
| down[0] (k3,s1) | 251 | 129 | 输出通道 64 |
| down[1] (k5,s2) | 126 | 65 | `⌊(T+4−5)/2⌋+1` |
| down[2] (k5,s2) | 63 | 33 | |
| down[3] (k5,s2) | 32 | 17 | 瓶颈；**只有这一层在编码路径上被 process_f/t/att 处理**（`Falcon.py:842-845`） |
| 解码 path | 63→126→251 | 33→65→129 | `concat[-1]`+process@level2 → level1 → level0 |

`n_freqs` 记录（`Falcon.py:747,770-774`）算出的 F 序列 `129,65,33,17` 与实测**完全一致**（[验证] 每个 `process_att_layers[i].n_freqs`），
所以 `LayerNormalization4D` 的 gamma/beta 形状不会错。（注意该记账**只跟踪 F，不跟踪 T**；之所以没错，是因为同一个 `stride` 同时把 T 也减半。）

**forward（`Falcon.py:826-896`）**：`residual = gateway(x)` → `x_enc = projection(residual)` → 逐级下采样（只在最深级处理）→
`global_features = downsampled_outputs[-1]` → `x_fused[i] = fusion_layers[i](downsampled_outputs[i], global_features)`（`Falcon.py:854-858`）→
`expanded = concat_layers[-1](x_fused[-2], x_fused[-1]) + downsampled_outputs[-2]`（`Falcon.py:868`）→ process@2 → 依次 level1、level0 →
`F.interpolate` 回输入 (T,F) → `residual_conv(expanded) + residual`（`Falcon.py:893-894`）。
[验证] `process_*` 在每个 level **各被调用 1 次**（3 级在解码、0 级最后），即"编码只用瓶颈，解码逐级处理"；
`fusion[0..3]` 各 1 次；**`concat` 只用 `[3]`（`:868`）、`[1]`、`[0]`（`:879` 循环），`concat_layers[2]` 调用 0 次**——其 5,568 参数全程闲置（§4.9-㉑，hook 计数 `{0:2,1:2,2:0,3:2}`）。

**因果性**：`AudioBlock.causal=True` 只传给 `process_t_layers`（`Falcon.py:781`），而 `DualPathRNN` 不读 `causal`（§4.9-⑥）；
`downsample_layers`/`fusion_layers`/`concat_layers`（k>1 的三个）与 `gateway`/`projection`/`residual_conv`（k=1，无时间跨度）全部**硬编码 `causal=False`**
（`Falcon.py:725,734,765,801,814,823`）。R3 修正：A21 的"±2 帧"只在去掉 cLNhw 后成立，as-built 是全局响应（见 §2.1.4、A21a-c）。

### 2.3 `VideoBlock`（`Falcon.py:898-974`，42,850 参数）

**职责**：把冻结 FRCNN 的帧级 512 维特征做时间建模（单向 LSTM）并压到 32 维再还原 512 维，供融合层使用。
**结构**：`gateway`（dwconv512 k1 + PReLU，1,025）→ `projection`（512→32 k1 + BatchNorm1d(32) + PReLU，16,481）→
`nn.LSTM(32,32,bidirectional=False)`（8,448）→ `residual_conv`（32→512 k1，16,896）→ `+ residual`（`Falcon.py:931-972`）。
**形状**：输入 `mouth_embedding [B,512,50]`（`frameworks: studio/falcon_runtime.py:24-27,157`；[验证] FRCNN 输出 `(1,512,50)`），
输出 `[B,512,50]`；LSTM 前 `transpose(1,2)` 成 `[B,50,32]`（`Falcon.py:968-970`）。
**因果性**：单向 LSTM ⇒ eval 下时间严格因果（[验证] 前缀 Δ=0）；但 `BatchNorm1d` 在 **train 模式**用批内（含未来帧）统计 ⇒
训练期非因果（[验证] train 前缀 Δ=4.4e-2）。`layers`/`num_heads`/`head_dim`/`expand_v`/`use_*` 形参**全部未使用**（`Falcon.py:907-917,926-929`），
配置里的 `video_params.layers.layer_1`（n_head=8, norm=cLN, dropout=0.1）是**死配置**。

### 2.4 `Separator`（`Falcon.py:976-1052`）—— 把音频/视频/融合串成"2 次音频块 + 1 次视频块"

**构造**（`Falcon.py:985-1023`）——注意**它用 `dict.pop` 消费配置**：

```python
self.fusion_repeats = self.video_params.pop("repeats", 0)          # =1
self.audio_repeats  = self.audio_params.pop("repeats", 0) - self.fusion_repeats   # =2-1=1
self.audio_shared   = self.audio_params.pop("shared", False)       # =True
self.video_shared   = self.video_params.pop("shared", False)       # =True
self.fusion_shared  = fusion_params.get("fusion_shared", False)    # =True（唯一被读的 fusion 键）
```

- `shared=True` ⇒ 只创建一个 `AudioBlock` / `VideoBlock` 实例，在多轮中**复用同一份权重**（`Falcon.py:1000-1014`）；
  否则创建 `nn.ModuleList`，长度分别为 `fusion_repeats+audio_repeats` 与 `video_repeat`。
- `caf` = `NeuroSCNetFusion(112, 512)`（`fusion_shared=True` 时单实例）（`Falcon.py:1017-1023`）。

**forward（`Falcon.py:1025-1052`）**：
```
audio_residual, video_residual = audio, video
for i in range(fusion_repeats=1):           # 融合轮
    audio = audio_net(audio)                # 第 1 次
    video = video_net(video)                # 唯一 1 次
    audio = caf(audio, video)
for j in range(audio_repeats=1):            # 纯音频轮
    audio = audio_net(audio + audio_residual)   # 第 2 次（共享权重 + 残差）
```
[验证] 一次 `Separator.forward` 内 `AudioBlock` 被调用 **2 次**、`VideoBlock` 1 次、`caf` 1 次；因此 `repeats: 2` 的真实语义是
"**音频块总共跑 2 遍（其中第 1 遍伴随视频融合）**"，而不是"堆 2 个独立音频块"。

### 2.5 `STFTEncoder` / `STFTDecoder`（`Falcon.py:1054-1192`，3,024 + 2,016 参数）

**`STFTEncoder`**：
- `torch.stft(x, n_fft=256, hop_length=128, window=hann, return_complex=True)`（`Falcon.py:1110-1116`）——**未传 `center`/`pad_mode`，
  即 PyTorch 默认 `center=True, pad_mode='reflect'`**（§4.6-①，这是全局因果性的第一个破点）。
- 3 通道堆叠：`[spec.real, spec.imag, sqrt(|spec|²+eps)]` 再 `transpose(2,3)` → `[B,3,T,F]`（`Falcon.py:1117-1118`）。
- `Conv2d(3→112, k=3, stride=1, bias=False, xavier_init=True, is2d=True, causal=True)`（`Falcon.py:1080-1091`）：
  时间维左 pad 2、频率维对称 pad 1，forward 裁掉右侧 2（`Falcon.py:100-104`）⇒ **时间上严格因果**（[验证] 见 §8-[O]）。
- `get_out_chan()` 返回 112（`Falcon.py:1104-1105`）；`window` 注册为非持久 buffer（`Falcon.py:1093`）。

**`STFTDecoder`**：
- `ConvTranspose2d(112→2, k=3, stride=1, padding=1, bias=False, xavier_init)`（`Falcon.py:1148-1157`）+ hann buffer。
- forward（`Falcon.py:1163-1192`）：`x.view(B*n_src, in_chan, T, F)` → **若 `causal` 则时间维前端 pad `k-1=2`** →
  ConvTranspose2d（stride=1、k=3、p=1，本身保持帧数）→ 输入帧数变成 `T+2`、卷积后仍 `T+2` → **裁掉末尾 `padding=1` 帧** ⇒
  实际输出 **`T+1` 帧**（多出的末帧随后被 iSTFT 的 `length` 截断）→ 取实/虚两通道组成复数谱 → `transpose(1,2)` →
  `torch.istft(..., length=input_shape[-1])` → `[B*n_src, L]` → `view(B,n_src,L)`。
- **因果性**：[验证] 累计 0 泄漏（§8-[O]：改动 ≥16000 样本的输入帧后，输出 `<16000` 样本的差恒为 0）。实现上依靠"前端 pad + 末尾裁剪"，
  使输出帧 n 只依赖输入帧 n−1..n−3（因果 ConvTranspose 的固有时延，由 iSTFT 的 `length` 截断吸收）；
  等价于**严格因果 + 3 帧（24 ms）内部 lookback**，不是延迟。
- 注意 `in_chan` 由 `Falcon` 传入 `encoder_out_chan * n_src`（`Falcon.py:1334`），在 `n_src=1` 时与 `MaskGenerator` 输出通道一致；
  `n_src>1` 时 `view` 会形状不符（[验证] `RuntimeError: shape '[2,224,251,129]' is invalid`）⇒ **架构只支持 n_src=1**。
- 同样 `act_type/norm_type/out_chan` 这些 `enc_dec_params` 键被 `*args/**kwargs` 静默吞掉（`Falcon.py:1134-1136`）。

### 2.6 `MaskGenerator`（`Falcon.py:1194-1275`，12,657 参数）

- `mask_generator = nn.Sequential(nn.PReLU(), ConvNormAct(112→112, k=1, act=ReLU, is2d=True, causal=causal))`（`Falcon.py:1227-1237`）；
  `bottleneck_chan == audio_emb_dim == 112`，`mask_output_chan = n_src*in_chan = 112`。
- `RI_split=True`（配置）时（`Falcon.py:1247-1259`）：把 112 通道视为"2×56"的复数——`mask_real=mask[:,:,0]`、`mask_imag=mask[:,:,1]`，
  编码器特征同样拆成 `emb_real/emb_imag`，做**复数乘法**：
  `est_real = e_r*m_r − e_i*m_i`，`est_imag = e_r*m_i + e_i*m_r`，再拼回 112 通道。
  重要理解：这里的"实/虚"是**学到的特征通道分组**，不是 STFT 的真虚部/实部；所谓复数乘法是对特征做双线性交互。
- `output_gate/dw_gate/direct` 默认 False（`Falcon.py:1203-1205,1224,1239-1242,1267-1268`）：本配置下**不启用**额外门控分支。
- **因果性**：全部 1×1 卷积 + 逐元素乘，[验证] 前缀 Δ=0（严格因果）。

### 2.7 `Falcon` 主类（`Falcon.py:1277-1487`）

**构造（`Falcon.py:1278-1343`）**：
```python
self.encoder          = STFTEncoder(**enc_dec_params)                       # out_chan=112
self.audio_bottleneck = ConvNormAct(**audio_bn_params, in_chan=112)
self.video_bottleneck = ConvNormAct(**video_bn_params, in_chan=512)        # kernel=-1 => Identity, 0 参数
self.audio_params['causal'] = self.causal; self.video_params['causal'] = self.causal   # 原地改 dict
self.separator        = Separator(audio_params, video_params, audio_bn_chan=112, video_bn_chan=512, fusion_params)
self.mask_generator   = MaskGenerator(**mask_generation_params, n_src, audio_emb_dim=112, bottleneck_chan=112)
self.decoder          = STFTDecoder(**enc_dec_params, in_chan=112*n_src, n_src=n_src)
self.register_buffer("mdcl_window", hann(256), persistent=False)          # 仅 MDCL 用，不进 state_dict
```
其它属性：`sample_rate=16000`（`BaseModel`）、`mdcl_margin=0.0`、`mdcl_alpha=0.15`（`Falcon.py:1291-1292`，**config 未提供 → 用默认值**）。

**forward（`Falcon.py:1464-1483`）**：
```python
audio_mixture_embedding = self.encoder(audio_mixture)      # [B,112,T,F]
audio = self.audio_bottleneck(audio_mixture_embedding)     # [B,112,T,F]
video = self.video_bottleneck(mouth_embedding)             # [B,512,Tv]（Identity）
refined_features = self.separator(audio, video)            # [B,112,T,F]
sep_emb = self.mask_generator(refined_features, audio_mixture_embedding)   # [B,n_src,112,T,F]
est     = self.decoder(sep_emb, audio_mixture.shape)       # [B,n_src,L]
if self.training and near_audio is not None:               # ← 调用了不存在的 self._mdcl_loss
    return separated_audios, mdcl_loss
return separated_audios
```
（`self._mdcl_loss` 不存在 ⇒ 该分支必定 `AttributeError`，见 §4.9-①。）

**MDCL 三个方法**（训练期对比/排斥损失，`Falcon.py:1345-1462`）：

| 方法 | 行号 | 数学 | 用途 |
|---|---|---|---|
| `_sisnr(est, ref)` | 1361-1376 | 去均值；`α=⟨est,ref⟩/‖ref‖²`；`proj=α·ref`；`noise=est−proj`；`10·log10(‖proj‖²/(‖noise‖²+eps)+eps)` | 时域 SI-SNR [B] dB |
| `_mag_sim(est, ref)` | 1378-1408 | 对两者做 `torch.stft(win,hop,hann)` 取 `abs().flatten(1)`，再套用同一个 `_sisnr` | 相位无关的幅度谱相似度 [B] dB |
| `_mdcl_loss_hard(est, near, beta)` | 1410-1462 | `sim_k=_mag_sim(est, near_k)`；`K=0→0`；`K=1→mean relu(sim−margin)`；`K≥2→mean_B Σ_k softmax_k(β·sim)·relu(sim−margin)` | 硬负样本排斥 |

[验证] `_sisnr` 在已知 20 dB SNR 输入上返回 20.000 dB；`_mag_sim` 返回 16.33 dB（幅度域更严格）；
`_mdcl_loss_hard` 的 K=0/1/3、β=0/1/20 三种情形与手算公式**完全一致**（`allclose=True`），梯度可回传（|grad|≈101）；
`margin=500 → 0`、`margin=-500 → 加权均值`（§8-[E][M]）。
**触发条件**：`Falcon.forward` 中 `self.training and near_audio is not None`（`Falcon.py:1477`）；但见 §4.9-①，该路径当前不可用。
**`_encode_audio`（`Falcon.py:1345-1359`）**：把任意波形编码成 `refined_features`，内部用 `video_bottleneck(zeros(B,512,1))` 伪造零视频特征
（`Falcon.py:1355-1357`）——[验证] 输出 `[1,112,251,129]`。**全仓库无调用者**（死方法；NASS 用的是 `encode_stft`）。

### 2.8 `Falcon_Contrastive.py`（122 行）

#### 2.8.1 `AttentionSpeakerPool`（`Falcon_Contrastive.py:8-57`，14,584 参数）

- 职责：把 `[B,N,T',F]` 特征图**池化成一个 L2 归一化说话人嵌入** `[B,64]`（`spk_embed_dim=64`）。
- 5 步（`Falcon_Contrastive.py:41-57`）：① `mean(dim=-1)` 折叠频率 → `[B,N,T']`；② `permute(0,2,1)` → `[B,T',N]`；
  ③ 时域注意力打分 MLP `N→N//4→1`（`Linear(112,28)+Tanh+Linear(28,1,bias=False)`）；④ `softmax(dim=1)` 得时间权重；
  ⑤ 加权求和 → `[B,N]`；⑥ 投影 `Linear(112,64)+ReLU+Linear(64,64)` → `[B,64]`。
- **注意**：池化时**频率维被简单取均值**，信息有损；注意力是"逐帧标量权重"，不是 self-attention。

#### 2.8.2 `FalconSpeakerContrastive(Falcon)`（`Falcon_Contrastive.py:60-122`）

- `__init__`（63-69）：`super().__init__(*args, **kwargs)` 后额外建 `AttentionSpeakerPool(encoder.get_out_chan()=112, spk_embed_dim=64)`。
  **`spk_embed_dim` 只接受关键字参数**，所以 `configs/Falcon.yml:15` 的 `spk_embed_dim: 64` 只能配这个子类（[验证] 直接喂 `Falcon` 报 `TypeError`）。
- 新增两个训练期接口：`encode_stft(waveform) -> [B,112,T,F]`（只跑 `self.encoder`，72-80）；`pool_spk_embed(feat_map) -> [B,64]`（池化 + L2 归一化，82-91）。
- **重写 `forward`**（94-118）：与 `Falcon.forward` 相同的 5 步前向，但**返回值随 `self.training` 变化**：
  - `training=True` → `(separated_audios [B,n_src,L], separated_audio_embeddings [B,n_src,112,T,F])`；
  - `training=False`（eval）→ 只返回 `separated_audios`。
  [验证] train → `tuple[(1,1,32000),(1,1,112,251,129)]`；eval → `Tensor(1,1,32000)`（§8-[L]）。
- **它吞掉了 `**kwargs`**（`Falcon_Contrastive.py:95`）：传 `near_audio=...` 不会报错，也不会起任何作用（§4.9-②）。

**训练期对比损失的真实触发路径**（不在本文件里，而在 `look2hear/system/av_litmodule_nass_v2.py`，由 `configs/Falcon.yml:130` 选择）：
`query = pool_spk_embed(sep_embs[:,0])`（**有梯度**）→ `pos = pool_spk_embed(encode_stft(target_wav))`（`no_grad`）→
`neg_k = pool_spk_embed(encode_stft(near_wav_k))`（`no_grad`）→ `NASSContrastiveLoss(margin=0.5, α_pos=1, α_neg=1)`，
β 从 `beta_init=0` 线性退火到 `beta_final=1`（200 epoch），`total = PIT + loss_weight(1.0) * cl`（`av_litmodule_nass_v2.py:99-138`，
`nass_contrastive_loss.py:57-72`）。这与 §2.7 的 `_mdcl_loss_hard` 是**两套互不相同的对比/排斥机制**，本配置只用了 NASS 这一套。

### 2.9 `base_model.py` 与 `models/__init__.py`

- `BaseModel`（`base_model.py:30-34`）：只存 `_sample_rate=16000`、`_in_chan=1`；`sample_rate()` 只读访问器（39-40）。
  `load_state_dict_in_audio`（42-51）按 `"audio_model."` 前缀 **硬编码 12 字符**截断；`from_pretrain`（53-78）要求 ckpt 里
  有 `model_name` 且 `strict=False` 载入后**只对 missing 抛错**、unexpected 仅告警；`serialize()`（80-94）虽然 `import pytorch_lightning`
  但只取 `__version__`（注释自称 "Not used in torch.hub"）；`get_model_args`（100-102）基类抛 `NotImplementedError`。
  `Falcon.get_model_args` 只返回 `{"n_src": n_src}`（`Falcon.py:1485-1487`）⇒ **`serialize()`/`from_pretrain` 无法重建本模型**
  （缺全部 config），是遗留的通用模板代码。
- `models/__init__.py`：导入两个类（1-2 行），但 `__all__ = ["AVTIGER_causal"]`（4-6 行）**指向不存在的名字**；
  `get()`（26-41）用 `globals()` 小写映射，`get("Falcon")`/`get("falconspeakercontrastive")` 正常，`get("AVTIGER_causal")` 抛 `ValueError`，
  `from look2hear.models import *` 抛 `AttributeError`（[验证] §8-[F]）。

---

## 3. 关键调用链与数据流（带 file:line）

### 3.1 构造链：`configs/Falcon.yml` 的 `audionet_config` 逐键 → 构造参数 → 子模块

入口：`audio_train.py:54-58`（训练）/ `audio_visual_test.py:128-140`（评测）/ `studio/falcon_runtime.py:157`（推理）三处一致：
`getattr(look2hear.models, "FalconSpeakerContrastive")(sample_rate=16000, **audionet_config)`。

| 配置键（`configs/Falcon.yml`） | 值 | 去向（file:line） | 最终子模块 / 效果 |
|---|---|---|---|
| `n_src` | 1 | `Falcon.__init__` → `Falcon.py:1295,1327,1334,1335` | `MaskGenerator.n_src`、`STFTDecoder.n_src`、`decoder.in_chan=112*n_src` |
| `pretrained_vout_chan` | 512 | `Falcon.py:1296,1312,1321` | `video_bottleneck.in_chan`、`Separator.video_bn_chan`、`VideoBlock(in_chan=512)`、`NeuroSCNetFusion(vin_chan=512)` |
| `causal` | true | `Falcon.py:1304,1314-1315` | 写入 `audio_params['causal']`/`video_params['causal']` → `AudioBlock.causal`/`VideoBlock.causal`。**[验证] 该开关对网络图与输出完全无影响**（§4.6-⑧） |
| `spk_embed_dim` | 64 | `Falcon_Contrastive.py:63-69` | `AttentionSpeakerPool(112, 64)`；**类签名默认 128**，若配置漏写则参数 +19,584（[验证] §8-[S] 旁证） |
| `video_bn_params.kernel_size` | -1 | `Falcon.py:1312` → `ConvNormAct` | `kernel_size<=0` ⇒ `conv=Identity`、`out_chan=in_chan`、norm/act 全 None ⇒ **0 参数透传**（`Falcon.py:30,84-85`；[验证] 0 参数） |
| `audio_bn_params` | `pre_norm=cLNhw, pre_act=ReLU, out_chan=112, k=1, is2d=true` | `Falcon.py:1311,1320` | `ConvNormAct(in=112→out=112)` = cLNhw(112) + ReLU + Conv2d 112→112 k1；13,104 参数；其 `out_chan` 同时作为 `Separator.audio_bn_chan`（112） |
| `enc_dec_params` | `win=256, hop=128, out_chan=112, k=3, stride=1, bias=false, act_type=None, norm_type=None, causal=true` | `Falcon.py:1308`（encoder）、`1332-1336`（decoder） | `STFTEncoder(3→112,k3,causal)` / `STFTDecoder(in=112·n_src→2,k3,causal)`；`act_type/norm_type/out_chan` 传给 decoder 时被 `*args/**kwargs` 吞掉（`Falcon.py:1134-1136`） |
| `audio_params.hid_chan` | 64 | `Falcon.py:1317-1323` → `AudioBlock.hid_chan` | 金字塔内部通道数；`projection` 112→64、`residual_conv` 64→112 |
| `audio_params.kernel_size` / `stride` | 5 / 2 | `AudioBlock`（`Falcon.py:709-710,754,751`） | 第 2-4 级下采样卷积 k=5,s=2；**同一个 int stride 同时把 T 和 F 减半** |
| `audio_params.upsampling_depth` | 4 | `Falcon.py:713` | 4 级金字塔（F=129,65,33,17；T=251,126,63,32） |
| `audio_params.n_freqs` | 129 | `Falcon.py:704,747` | **仅用于**给每层 `MultiHeadSelfAttention2D` 传 `n_freqs` 以定 `LayerNormalization4D` 形状（`Falcon.py:785-786`）；若填错会 broadcast 报错 |
| `audio_params.repeats` | 2 | `Separator` **pop**（`Falcon.py:992`） | `audio_repeats = 2 − fusion_repeats(1) = 1` ⇒ 音频块总共跑 2 遍 |
| `audio_params.shared` | true | `Separator` **pop**（`Falcon.py:995`） | `True` ⇒ **单实例权重复用**（`Falcon.py:1000-1001`），而非 2 份权重 |
| `audio_params.is2d` | true | `AudioBlock` → 各 `ConvNormAct`/`MaskGenerator` | 决定用 Conv2d 与 `(H,W)` 双维 padding/stride |
| `audio_params.norm_type/act_type` | cLNhw / PReLU | `Falcon.py:731-732` 等 | 除 RNN/attention 外几乎所有块的归一化/激活 |
| `audio_params.layers.layer_1` | `hid=24, dim=4, k=8, stride=1, rnn_type=LSTM, num_layers=1, bidirectional=true` | `Falcon.py:777-779` | `process_f_layers[i] = DualPathRNN(in_chan=64, **layer_1)` ⇒ **频率轴双向 LSTM**（24×2=48→线性回 64） |
| `audio_params.layers.layer_2` | `hid=72, dim=3, k=8, stride=1, rnn_type=LSTM, num_layers=1, bidirectional=false` | `Falcon.py:780-782` | `process_t_layers[i] = DualPathRNN(in_chan=64, causal=?, **layer_2)` ⇒ **时间轴单向 LSTM**（72→线性回 64） |
| `audio_params.layers.layer_3` | `dim=3, n_head=4, hid_chan=4, act_type=PReLU, norm_type=LayerNormalization4D, causal=true` | `Falcon.py:785-790` | `process_att_layers[i] = MultiHeadSelfAttention2D(in_chan=64, n_freqs=当前F, **layer_3)`；**这里的 `causal=true` 才真正开启注意力三角 mask** |
| `video_params.hid_chan/kernel_size/stride/norm_type/act_type` | 32 / 3 / 2 / BatchNorm1d / PReLU | `Falcon.py:1009,1012` | `VideoBlock(512→32→LSTM→512)`；`kernel_size/stride` 实际未被 VideoBlock 使用（只有 1×1 conv + LSTM） |
| `video_params.repeats` | 1 | `Separator` pop（`Falcon.py:991`） | `fusion_repeats = 1`（= 视频块跑 1 遍、音频块第 1 遍伴随融合） |
| `video_params.shared` | true | pop（`Falcon.py:996`） | 单实例 VideoBlock |
| `video_params.layers.layer_1` | `kernel_size=3, n_head=8, dropout=0.1, norm_type=cLN, causal=true` | — | **完全未被使用**（VideoBlock 不读 `layers`，`Falcon.py:926-929`）⇒ 死配置 |
| `fusion_params.fusion_shared` | true | `Falcon.py:997,1017-1023` | 单实例 `NeuroSCNetFusion(112,512)` |
| `fusion_params.fusion_type/kernel_size/is2d` | NeuroSCNetFusion/4/true | — | **未被读取**（类名硬编码在 `Falcon.py:1018`）⇒ 死配置 |
| `mask_generation_params.mask_act/RI_split/is2d` | ReLU/true/true | `Falcon.py:1325-1330` → `MaskGenerator` | 掩码生成：PReLU + Conv2d(112→112,k1) + 复数掩码乘法 |
| `mask_generation_params.mask_generator_type` | MaskGenerator | — | 被 `MaskGenerator(**kwargs)` 吞掉（`Falcon.py:1208-1209`）⇒ 死配置 |

**构造副作用（重要）**：`Separator.__init__` 用 `dict.pop` 直接修改传入的 `audio_params`/`video_params`（`Falcon.py:991-996`），
且 `Falcon.__init__` 把同一个 dict 对象既存为属性又传给 Separator（`Falcon.py:1300-1301,1317-1323`）。
后果：[验证] **对同一个 config dict 连续构造两个模型，第二个模型参数量从 537,697 塌缩到 49,081**（`audio_net`/`video_net` 变成长度 0 的
`ModuleList`）且**前向不报错、静默输出错误结果**（§8-[T]）。所有调用方都必须先 `deepcopy`：`audio_visual_test.py:128`、
`studio/falcon_runtime.py:149` 均如此，`PROVENANCE.md` 也专门提到这一点。

### 3.2 forward 数据流（B=1，L=32000=2 s@16 kHz，全部 [验证] §8-[C]）

| # | 阶段 | 代码 | 输入 | 输出 | 说明 |
|---|---|---|---|---|---|
| 0 | 输入 | — | `audio_mixture [1,32000]` float32；`mouth_embedding [1,512,50]` | — | 16 kHz 单声道；25 fps × 2 s = 50 帧 |
| 1 | STFT 编码 | `Falcon.py:1466`,`1107-1121` | `[1,32000]` | `[1,112,251,129]` | `T'=1+⌊32000/128⌋=251`，`F=256/2+1=129`；3 通道 real/imag/mag |
| 2 | 音频瓶颈 | `Falcon.py:1468` | `[1,112,251,129]` | `[1,112,251,129]` | cLNhw + Conv2d(112→112,k1) |
| 3 | 视频瓶颈 | `Falcon.py:1469` | `[1,512,50]` | `[1,512,50]` | Identity（0 参数） |
| 4 | AudioBlock#1 | `Separator` `Falcon.py:1032` | `[1,112,251,129]` | `[1,112,251,129]` | 内部：`[1,64,251,129]`→(126,65)→(63,33)→(32,17)→回到 251×129 |
| 5 | VideoBlock | `Falcon.py:1037` | `[1,512,50]` | `[1,512,50]` | 512→32→单向LSTM→512 |
| 6 | 融合 | `Falcon.py:1042` | `[1,112,251,129]` + `[1,512,50]` | `[1,112,251,129]` | v_proj 512→112（groups=16）；T 50→251 nearest；加性注入 |
| 7 | AudioBlock#2 | `Falcon.py:1048` | `[1,112,251,129]`（+ 第 2 步残差） | `[1,112,251,129]` | **与 #4 共享权重** |
| 8 | 掩码生成 | `Falcon.py:1473`,`1266-1275` | `[1,112,251,129]` + 编码器特征 | `[1,1,112,251,129]` | RI_split 复数乘法（56 实 + 56 虚） |
| 9 | 解码 | `Falcon.py:1474`,`1163-1192` | `[1,1,112,251,129]` + `audio_mixture.shape` | `[1,1,32000]` | ConvTranspose2d(112→2) → 复谱 → iSTFT(length=32000) |
| 10 | 返回 | `Falcon.py:1483` / `Falcon_Contrastive.py:116-118` | — | 见 §5.2 | train 多返回 `sep_audio_emb` |

`AudioBlock` 内部逐级形状（[验证]）：`down[0] 251×129 → down[1] 126×65 → down[2] 63×33 → down[3] 32×17`；
`proc_f/proc_t/proc_att` **只在 level 3 于编码路径被调用**（`Falcon.py:842-845`），解码路径依次调用 level 2、1、0（`Falcon.py:872-888`）；
**R3 修正**：`fusion[0..3]` 每级各 1 次；`concat` **只用 `[3]`、`[1]`、`[0]`，`concat_layers[2]` 调用 0 次**（`Falcon.py:868` 用 `concat[-1]`，
`:879` 的 `range(upsampling_depth-3, -1, -1)` 只遍历 1、0）⇒ 5,568 参数闲置（占 537,697 的 1.04%，§4.9-㉑、A51）。

### 3.3 训练期调用链（与下游的耦合点）

```
AVSpeechDataModuleWithNear（2.0 s 段、16 kHz、use_near=true；configs/Falcon.yml:154-168）
 → batch = (mixtures, targets, mouth, _, near_audios)
 → AudioVisualLightningModuleNASSv2.training_step（configs/Falcon.yml:130 → av_litmodule_nass_v2.py:73）
     ├─ video_model(mouth)  ← torch.no_grad()（av_litmodule_nass_v2.py:86-87）
     ├─ (est, sep_embs) = audio_model(mixtures, mouth_emb)   # 依赖 train 模式返回 tuple（av_litmodule_nass_v2.py:90）
     ├─ pit_loss = PITLossWrapper(pairwise_neg_snr)(est, targets)   # av_litmodule_nass_v2.py:95
     ├─ query/pos/neg 三路 speaker embedding + NASSContrastiveLoss（:99-134）
     └─ total = pit + 1.0*cl（:137-138）
```
`optimizer` 只包含 `model.parameters()`（`audio_train.py:64-67`），视频模型由 `update_frcnn_parameter` 置 `requires_grad=False`
（`look2hear/videomodels/frcnn_videomodel.py:106-119`）且在 forward 时 `no_grad`。

### 3.4 推理期调用链（Studio）

`studio/falcon_runtime.py:24-27`：`SAMPLE_RATE=16000`、`VIDEO_FPS=25`、`WINDOW_SAMPLES=32000`、`WINDOW_FRAMES=50`；
`_forward`（235-244）先 `preprocess_mouth(frames)`（`frames[:,4:92,4:92]/255` + 均值 0.421/方差 0.165，见 101-102 行），
再 `audio_model(waveform, mouth_embedding)` —— **eval 模式**，因此拿到的就是 `[1,1,32000]` 张量（不是 tuple），`separated[0,0]` 取波形。
`PROVENANCE.md` 明确写："The model exposes no streaming state, so this is windowed inference, not sample-by-sample streaming."
——与本轮的因果性结论一致（§4.6）。Studio 副本在 **Apple MPS** 上把 STFT/iSTFT 搬到 CPU（`studio/falcon_model/models/Falcon.py:1111-1122`、
`1184-1198`；逐行核验见 06 文档 §4.8/A45），CPU/CUDA 下与训练侧数值等价（PROVENANCE:33 声称 max abs diff=0）。

---

## 4. 关键实现细节与设计动机

### 4.1 为什么 STFT 输入是 3 通道（real / imag / magnitude）

`STFTEncoder` 不用 `layers/stft.py` 的 STFT 层，而是 `torch.stft` 后手工堆叠 `[real, imag, |X|]`（`Falcon.py:1110-1118`），
再交给一个 3→112 的 2D 卷积。`[推断]` 动机：① 让网络同时拿到相位敏感（实/虚）与相位不敏感（幅度）的表示，训练更稳；
② 3 通道 k=3 卷积只花 3,024 参数，比"先算复数谱再复数网络"便宜；③ 幅度通道显式给出一条与 `_mag_sim` 一致的"谱能量"通路。

### 4.2 为什么只有 0.5 M 可训练参数

| 设计手段 | 证据 | 省下的量级 |
|---|---|---|
| 几乎全部卷积是 depthwise（`groups=in_chan`） | `Falcon.py:762,795-813,1229-1236` | 融合/下采样块从 O(C²k²) 降到 O(Ck²) |
| `AudioBlock` **单实例共享**（`shared: true`）跨 2 轮复用 | `Falcon.py:995,1000-1001`；[验证] 一次 forward 调 2 次 | 少一份 445,766 |
| 通道瓶颈 64（音频）/ 32（视频） | `Falcon.py:709,928` | 主干在 64 通道上跑，而非 112/512 |
| 每层 `process_*` 只建一套、且每级只跑 1 次 | `Falcon.py:749-790,842-845` | 没有"每级 N 个 block"的堆叠 |
| 注意力每头只 4 通道（Q/K） | `configs/Falcon.yml:69` + `Falcon.py:441-481` | MHA 从 O(C²) 降到 O(C·hid) |
| 视频侧只用单向 LSTM(32) | `Falcon.py:950-954` | 视频时间建模 8,448 参数 |

对比：解码器只有 2,016 参数、mask 头 12,657、融合投影 3,696、视频分支 42,850；**82.9% 的参数集中在 `AudioBlock`**（445,766/537,697）。

### 4.3 4 级时频金字塔 + `upsampling_depth` / `repeats` / `shared` 的真实语义

- `upsampling_depth=4` 决定**金字塔级数**：每级 `downsample_layers[i]` 用 `stride=1(i=0)`/`2(i>0)`，且 `kernel_size=3(i=0)`/`5(i>0)`
  （`Falcon.py:751-767`）。因为 `stride` 是 int、`Conv2d` 会同时作用于 H/W（`(2,2)`），F 与 T 同时减半：
  `(251,129)→(126,65)→(63,33)→(32,17)`。
- **编码只在最深级处理**：`if i == upsampling_depth-1: process_f/t/att`（`Falcon.py:842-845`）——这是一种"极简 encoder"设计：
  浅层只做下采样 + 保存 skip，所有时序建模都发生在 32×17 的瓶颈上。计算量因此大幅下降（这与 README 的 MACs 10.6G 相符，待 04 域验证）。
- `repeats`：视频的 `repeats=1` 被当作 `fusion_repeats`（融合轮数），音频的 `repeats=2` 减 1 得 `audio_repeats=1`
  （`Falcon.py:991-993`）⇒ 音频块跑 2 遍、视频块跑 1 遍。**`repeats` 不是"堆叠层数"**，而是"联合轮 + 音频独轮"的计数。
- `shared`：语义是"同一模块实例复用"，因此 `audio_params.shared=true` 时 `repeats` 只影响**调用次数**、不影响参数量；
  若改成 `false`，会创建 2 个独立 AudioBlock，参数量直接翻倍（+445,766）。

### 4.4 `RI_split` 复数掩码乘法的实际含义

`MaskGenerator.__apply_masks`（`Falcon.py:1244-1264`）把 112 通道的 mask 拆成 `[2, 56]`，把编码器 112 通道特征也拆成 `[2, 56]`，
按复数乘法组合，再拼回 112 通道。**这不是对真实 STFT 的掩码**：编码器输出已经是 112 维抽象特征，"实/虚"只是通道分组。
`[推断]` 动机：复数乘法是**双线性交互**（乘法而非相加），比逐通道实掩码表达力强，且零额外参数；
同时与解码器"输出 2 通道复谱"的最终形态对齐，形成一个"抽象复域"到"真复谱"的映射。
风险：`RI_split` 要求 `in_chan` 为偶数（112/2=56 ✓），代码里**没有断言**保护（`Falcon.py:1248-1249`）。

### 4.5 双向/单向与注意力：谁在哪个轴上"看"

| 模块 | 轴向 | 方向 | 证据 |
|---|---|---|---|
| `process_f_layers`（DualPathRNN dim=4） | 频率 F | **双向**（bidirectional=true） | `configs/Falcon.yml:57`；`Falcon.py:386-392` 把 T 变 batch，故双向不引入时间未来依赖 |
| `process_t_layers`（DualPathRNN dim=3） | 时间 T | **单向** | `configs/Falcon.yml:65`；`Falcon.py:398-401` |
| `process_att_layers`（MHSA2D） | 时间 T | **因果 mask**（整段一次算） | `configs/Falcon.yml:72`；`Falcon.py:520-524` |
| `VideoBlock.rnn` | 时间 T_v | 单向 LSTM | `Falcon.py:950-954` |
| `NeuroSCNetFusion` 的时间插值 | T_v→T_a | nearest（只用过去/当前） | `Falcon.py:257-258` |

设计取向很清楚：**允许在频率上做双向（频率不是因果方向），在时间上只允许单向/因果**。
`[推断]` 这是"因果 AVTSE"论文主张的架构级实现方式：把非因果全部推给频率轴与卷积核，时间轴保持单向。

### 4.6 因果性的实现机制与严格性分析（**本子系统最关键的结论**）

**结论（R3 统一措辞：窗口级因果 window-causal）**：Falcon 的 "causal" 是**结构层面**（时间轴的卷积 padding 方向、RNN 方向、注意力 mask）的因果，
**不是严格的逐采样因果**；更准确的表述是**窗口级因果**——在 2 s 窗口内允许存在未来依赖（STFT 帧内 ±128 样本/8 ms；对称卷积累积
57 帧≈456 ms；`cLNhw` 全局统计带来的**无界**依赖），但**跨窗口无状态**，因此它不是逐采样流式模型（定义与交叉引用见 ⑦）。
严格因果被三处打破：① `torch.stft` 的 `center=True`（8 ms；`istft` 的 center 被解码器 3 帧因果位移吸收，不额外增加未来依赖）；
② 部分卷积硬编码 `causal=False`（对称 padding）；③ `cLNhw` 的 `GlobalLayerNorm`（`nn.GroupNorm(1)` 对**整窗**统计）；此外顶层 `causal` 开关本身**完全惰性**（⑧）。

① **STFT 分帧**：`torch.stft(..., return_complex=True)` 未传 `center`/`pad_mode`（`Falcon.py:1110-1116`），
默认 `center=True, pad_mode='reflect'` ⇒ 帧 f 的窗口是 `[f·hop−win/2, f·hop+win/2)`，即需要 **128 样本（8 ms）未来音频**。
[验证] 把 `samples≥16000` 改成噪声后：`encoder` 输出帧 ≤124 的 |Δ| 恰为 0，帧 125 起非零（§8-[O]；R3 复跑 §8.4-[W8]）。
**首变帧公式（R3 新增，供复现）**：`f_first = ⌊(s − win/2)/hop⌋ + 1`——s=16000→125、s=20000→156、s=16128→126（三点全部命中）；
帧 124 的窗 `[15744,16000)` 不含未来样本，故精确说法是"**帧 ≤124 全 0**"。
`istft` 同样是 `center=True`（`Falcon.py:1182-1188`）；它与解码器因果 ConvTranspose 组合后，输出采样 s 只依赖输入嵌入帧 ≤ ⌊s/hop⌋，
因此**不额外增加未来依赖**（[验证] 改 embedding 帧≥125 后输出 <16000 样本 Δ=0；推导见 ⑨ 注）。

② **卷积 padding 方向**：`ConvNormAct` 的因果实现是"左 pad `dilation·(k−1)` + 右侧裁剪"（`Falcon.py:48-63,100-104`）。
配置下**真正开启**因果的只有：`STFTEncoder.conv`（`causal=true`，k=3，`Falcon.py:1090`）与 `STFTDecoder`（`Falcon.py:1170-1177`）。
R3 更正：`MultiHeadSelfAttention2D` 内部的 Q/K/V/concat_proj 四个 `ConvActNorm` 实际都是**显式 `causal=False`**（`Falcon.py:455,467,479,491`），
只是 `kernel_size=1` 无时间跨度故无害；**注意力的因果性来自 `torch.tril` mask（`Falcon.py:520-524`）**，不是卷积。
而 `AudioBlock` 里所有 k>1 的卷积都**硬编码 `causal=False`**：`downsample_layers`（k=3(i=0)/5(i>0)，`Falcon.py:765`）、
`fusion_layers`/`concat_layers`（`InjectionMultiSum`，k=5，`Falcon.py:801,814`）。换算到 level-0 帧单位（T 每级减半）：
**L0 ±1、L1 ±4、L2 ±8、L3 ±16 帧**，单遍累积约 29 帧；
[验证] 把 cLNhw 全部替换为 Identity 后，`AudioBlock` 的**前缀 lookahead = 57 帧 = 7,296 样本 ≈ 456 ms**（§8-[O]）；
R3 复跑分界敏感性：split=100/125/150 → **56/57/58 帧**（±1 帧是 nearest 插值的对齐量化，§8.4-[W7]）——
这就是"纯结构"的非因果量级；若某层卷积改 `causal=True`，[验证] 该层即严格因果（`InjectionMultiSum` 前缀 Δ=0）。

③ **归一化**：`cLNhw = LayerNormalizationHW` = `GlobalLayerNorm(nn.GroupNorm(1))` + `CumulateLN4D`
（`look2hear/layers/normalizations.py:204-221,224-229,134-144`）。`CumulateLN4D` 是沿时间累积的因果归一化，但 `GlobalLayerNorm`
对 `[B,C,T,F]` 的 **C/T/F 全部**求均值方差 ⇒ **依赖整段 2 s 窗口**。它出现在 `audio_bottleneck.pre_norm` 与 `AudioBlock` 的
`projection/downsample/fusion/concat` 里（模型内共 **30 个** `LayerNormalizationHW` 实例，[验证] `named_modules` 计数）。
**[验证]**：在 **P1（同分布 randn 替换后半段）** 协议下，只跑 `audio_bottleneck.pre_norm` 左段仍有 **1.4e-6** 差异；
去掉 cLNhw 的 1×1 conv 同一测试**严格为 0**（§8-[D]）——泄漏源确认为归一化。
**R3 新机理（关键）**：`cLNhw` 的第一段 GN 单独在 P2（×10）协议下左段 Δ=**4.40**，但经第二段 `CumulateLN4D`（沿时间累积、逐频点归一化）后
只剩 **1.9e-6**——**第二段会把"纯尺度型"泄漏抵消掉**；因此"GN 全窗口统计"与"实测只有 1e-6"并不矛盾，模型级非因果主要来自
下游对称卷积（`AudioBlock` 在 P2 下 Δ=7.14）。**引用泄漏量级必须同时给扰动协议**，见 ⑨ 的三协议表。

④ **RNN / 注意力**：[验证] `DualPathRNN(dim=3)` 时间单向 LSTM 前缀 Δ=0；`DualPathRNN(dim=4)` 因 T 是 batch 维，前缀 Δ=0；
`MHSA2D(causal=True)` 前缀 Δ=0，而 `causal=False` 时 Δ=4.68（§8-[D]）。所以这三者是**结构因果**的。
但 `MHSA2D` 在整段 T 上做 O(T²) 注意力、且没有 KV cache（`Falcon.py:494-543`）⇒ **数学因果但不可流式**，流式需要缓存历史 K/V。

⑤ **视频分支**：冻结的 FRCNN 前端 `Conv3d(kernel=(5,7,7), padding=(2,3,3))`（`look2hear/videomodels/frcnn_videomodel.py:58-70`）
在时间上 **±2 帧 = ±80 ms** 对称。[验证] 改视频帧 ≥30 后，输出最左受影响帧 = 28（§8-[P]）。
`VideoBlock` 的 LSTM 单向（eval 下前缀 Δ=0）；但 `BatchNorm1d` 在 **train 模式**用批内统计（含未来帧），[验证] train 前缀 Δ=4.4e-2，
eval 为 0（§8-[P]）。Studio 的 `OutputSlicer` 保留 **80 ms 右上下文**（`GUARD=1280` 样本，`server.py:29`；06 文档 §5.5 给出精确切片语义），
与这里的 ±80 ms **量级吻合**——06 域已完成实现级验证（06 §1.3/§5.5），不再是纯 `[推断]`。

⑥ **端到端量化**（[验证] §8-[O]，B=1、L=32000、只改 1.0–2.0 s 的音频）：
| 输出区间 | max\|Δ\| |
|---|---|
| 0.00–0.12 s | 5.1e-7 |
| 0.12–0.25 s | 3.0e-6 |
| 0.25–0.38 s | 2.3e-5 |
| 0.38–0.50 s | 1.1e-4 |
| 0.50–0.62 s | 8.3e-4 |
| 0.62–0.75 s | 3.2e-3 |
| 0.75–0.88 s | 1.5e-2 |
| 0.88–1.00 s | 4.2e-2 |
即"越靠近分界泄漏越大"，且**前 0.5 s 已有 1e-4 量级**（Lead 独立测得 2.97e-4，同量级），与全局归一化 + 对称卷积的机制一致。

⑦ **统一措辞：窗口级因果（window-causal）**（与 02 §4.2、06 §4.1/§5.5 采用同一句）：

> **窗口级因果**：在 2 s 窗口内允许存在未来依赖（STFT 帧内 ±128 样本/8 ms；对称卷积累积 57 帧≈456 ms；`cLNhw` 全局统计带来的无界依赖），
> 但模型**不携带跨窗口状态**，因此它不是逐采样因果的流式模型。

工程后果：Studio 采用"2 s 窗口 + 0.5 s hop + 80 ms 右上下文"的**分块重叠推理**，`PROVENANCE.md` 也明确 "no streaming state"。
若要真正做逐采样流式，必须改：`cLNhw→因果归一化`、`InjectionMultiSum/downsample 改 causal=True`、`stft 改 center=False`、
`MHSA2D 改 KV-cache`；其中**前两项会改变参数/统计语义 ⇒ checkpoint 不兼容、需重训**，后两项可保持权重不变但会改变边界数值。

⑧ **顶层 `causal` 开关是惰性的（反直觉，[验证]）**：`Falcon(causal=True)` 与 `Falcon(causal=False)`（其余配置相同）
参数量相同（537,697）、`state_dict` key 相同、装载同一权重后**输出 bit-identical**（§8-[S]）。原因链：
`Falcon.causal` 只写入 `audio_params['causal']`/`video_params['causal']`（`Falcon.py:1314-1315`）→
`AudioBlock.causal` 只传给 `DualPathRNN(causal=...)`（`Falcon.py:781`），而 `DualPathRNN` **从不读取 `self.causal`**（`Falcon.py:365` 只赋值）；
`VideoBlock.causal` 从不读取（`Falcon.py:929`）；`MaskGenerator` 的 `causal` 只作用于 k=1 卷积（无时间跨度）；
`MultiHeadSelfAttention2D.causal` 来自 `layer_3.causal`，与顶层开关无关。**真正起作用的因果开关是
`enc_dec_params.causal` 与 `audio_params.layers.layer_3.causal` 两个内层键**。

⑨ **因果性探针协议表（R3 新增，解决 01 与 02 的泄漏数值口径分歧）**：同一"非因果"结论在不同扰动协议下可相差 2–4 个数量级，
引用"泄漏量级"必须同时给协议。统一约定：fp32、`model.eval()`、`torch.manual_seed(0)`、判据 `max|Δ| > 1e-12` 视为"左段被影响"、
分界 = 帧 125（=1.0 s），测"改后半段后左段 max|Δ|"。

| 探针 | 扰动方式 | 测点 | 左段 max\|Δ\| | 说明 |
|---|---|---|---|---|
| P1 同分布替换 | 后半段替换为 `randn`（σ=1） | `audio_bottleneck.pre_norm`（cLNhw） | **9.5e-7 ~ 1.4e-6** | 本文 R1 的 A19 值；GN 的 μ/σ 几乎不变 |
| P2 放大 10× | 后半段 ×10 | 同上 | 1.9e-6 | GN 对纯尺度不敏感 + CumulateLN4D 再归一化 |
| P3 单帧尖峰 | 最末帧 +20（**T=8 人工张量**） | 独立 `LayerNormalizationHW(64)` | **1.2e-3**（02 A19 值；我复算 9.9e-4） | 短序列下尖峰占窗口 1/8，统计被显著改变 |
| P4 波形 ×10 | 音频后半段 ×10、split 帧 120 | `audio_bottleneck`（cLNhw+1×1 conv） | **3.2e-5**（02 §[7] 报 4.7e-5，同量级） | 经 encoder 后进入瓶颈；与 02 的探针条件一致 |
| P5 全链 ×10 | 同 P4 | 最终波形 | **9.7e-4**（0–0.5 s）/ O(1)（>1 s） | 02 §[7] 的波形值；下游逐层放大 |

**跨文档裁定**：02 的 1.2e-3 / 4.7e-5 与本文的 1.4e-6 **都正确、只是协议不同**（前者 T=8 尖峰 / 波形×10，后者 T=251 同分布替换），
不可跨协议比较；FINAL 引用时应至少给两个协议。关键机理：`cLNhw` 第一段 GN 单独在 P2 下 Δ=**4.40**（强泄漏），经第二段
`CumulateLN4D` 后只剩 **1.9e-6**（纯尺度泄漏被抵消）；因此**单层 GN 的"无界非因果"在实测上会被显著削弱，而模型级非因果的
主要来源是下游对称卷积与多层归一化串联**（`AudioBlock` P2 下 Δ=7.14）。反向提醒：P1/P2 会**低估** cLNhw 风险，P3/P4 更敏感。


### 4.7 参数量核算（0.5 M 可训练 / 11.72 M 含冻结视觉）

**逐项可复算表**（每行手算公式与实测值全部一致，[验证] §8-[B][R]）：

| 模块 | 公式 | 参数 |
|---|---|---|
| `encoder.conv` | 3·112·3·3（bias=False） | 3,024 |
| `audio_bottleneck` | cLNhw(112)（GroupNorm 2·112 + CumulateLN 2·112 = 448）+ conv 112·112+112 | 13,104 |
| `ab.gateway` | 112·1·1 + 112 + PReLU(1) | 225 |
| `ab.projection` | 112·64+64 + cLNhw(64)=256 + PReLU(1) | 7,489 |
| `downsample_layers` | i=0: 64·9+64+256=896；i=1..3: 3×(64·25+64+256)=5,760 | 6,656 |
| `process_f_layers` | ×4：biLSTM(64→24) 2·(4·24·64+4·24·24+2·96)=17,280 + Linear(48→64,bias=F)=3,072 + LN4D(64,1)=128 | 81,920 |
| `process_t_layers` | ×4：LSTM(64→72) 4·72·64+4·72·72+2·288=39,744 + Linear(72→64)=4,608 + LN=128 | 177,920 |
| `process_att_layers` | 由 `n_freqs` 决定：51,693+31,213+20,973+15,853（F=129/65/33/17） | 119,732 |
| `fusion_layers` | ×4：3×(64·25+cLNhw(64)=256)=5,568 | 22,272 |
| `concat_layers` | 实例化 ×4（22,272），**前向调用 3 个**（`[3],[1],[0]`） | 22,272（其中 **5,568 闲置**） |
| `ab.residual_conv` | 64·112+112 | 7,280 |
| **`separator.audio_net` 合计** | | **445,766** |
| `video_net` | 1,025 + 16,481 + LSTM(32→32)=8,448 + 16,896 | 42,850 |
| `caf.v_proj` | 512·112/gcd(16) + 112 | 3,696 |
| **`separator` 合计** | 445,766+42,850+3,696 | **492,312** |
| `mask_generator` | PReLU(1) + 112·112+112 | 12,657 |
| `decoder` | ConvTranspose2d 112·2·3·3（bias=False） | 2,016 |
| `spk_pool_proj` | (112·28+28+28)+(112·64+64)+(64·64+64) | 14,584 |
| **audio_model 合计（含闲置）** | | **537,697** |
| — 其中全程闲置参数 | `concat_layers[2]`（§4.9-㉑） | **−5,568** |
| **实际参与前向的参数** | 537,697 − 5,568 | **532,129** |
| FRCNNVideoModel（冻结） | ResNet18 trunk + frontend3D | 11,186,688 |
| **总计（参数总量）** | 537,697 + 11,186,688 | **11,724,385** |

**与论文/README 口径的差异（诚实说明）**：
- README `README.md:34`/`README.md:206` 写 "**0.5 M** trainable"，实测 **537,697 = 0.538 M**；四舍五入到 1 位有效数字即 0.5，属陈述性取整 `[推断]`。
  若把**仅训练期使用**的对比头 `spk_pool_proj`（14,584）排除，则为 **523,113 = 0.523 M**（更接近 0.5）。
- README `README.md:100` 写 "Total params 11.72 M"，实测 **11,724,385 = 11.72 M** ✓（10,000 份一致）。
- **R3 追加（闲置参数口径）**：537,697 中有 **5,568（1.04%）的 `concat_layers[2]` 从不参与前向**（§4.9-㉑）；
  因此"推理时真正生效"的可训练参数为 **532,129（0.532 M）**；若再扣掉训练期专用对比头 `spk_pool_proj`（14,584），
  则为 **517,545（0.518 M）**。FINAL 引用 0.5 M 时必须说明用的是哪个口径。
- "冻结"的依据：`look2hear/videomodels/frcnn_videomodel.py:117` 在 `update_frcnn_parameter` 里 `p.requires_grad=False`
  （由 `pretrain` 路径触发的 `init_from` 调用，`frcnn_videomodel.py:86-88`）；训练时视频模型另在 `torch.no_grad()` 下前向
  （`av_litmodule_nass_v2.py:86-87`）。注意：**`video_bottleneck`/`VideoBlock` 是音频模型的一部分，是可训练的**（`Falcon` 的子模块），
  "冻结的 11.2 M"仅指 FRCNN。
- 未验证项：MACs 10.6 G / 延迟数字属于 `evaluated_mac_params.py` 与 04 域范围（§7-Q9）。

### 4.8 `mdcl_window` 与 state_dict 契约

`mdcl_window` 用 `persistent=False` 注册（`Falcon.py:1339-1343`）⇒ **不在 `state_dict` 里**（[验证] `mdcl_window` 不在
`state_dict()`，共 525 个 key）。因此 MDCL 的窗函数不会污染 checkpoint，`strict=True` 加载也不会因它报错；
反过来，如果未来把 MDCL 权重/窗参数化，就会破坏 Studio 的 strict 加载契约（§5.4）。

**真实权重实测（R3 新增，`[验证]` §8.4-[W1~W3]）**：`falcon.pt` 顶层是 `dict(state_dict=…, architecture='FalconSpeakerContrastive', epoch=714)`，
`state_dict` 共 **662** 条 = `audio_model.*` **525** 条 + `video_model.*` **137** 条；`mdcl_window` **不在其中**（与本地构建的 525 key 完全一致）。
`audio_model` 权重按 key/shape 与 `FalconSpeakerContrastive(configs/Falcon.yml)` **全等（0 missing / 0 unexpected / 0 shape 不一致）**，
`strict=True` 可加载；`video_model` 权重同样与 `FRCNNVideoModel` 全等。
**注意**：`state_dict` 张量 numel ≠ 参数数（audio 537,762 vs 参数 537,697，多 65 = `BatchNorm1d(32)` 的 running stats 等 buffer；
video 11,196,308 vs 11,186,688），做参数量核算必须用 `parameters()` 而不是累加 `state_dict`。

### 4.9 代码坏味道 / 死代码 / 命名与实现不一致 / TODO 清单

| # | 类型 | 问题 | 证据 |
|---|---|---|---|
| ① | **P0 级死路径** | `Falcon.forward` 在 `training and near_audio is not None` 时调用 `self._mdcl_loss(...)`，但类中只有 `_mdcl_loss_hard` | `Falcon.py:1480` vs `Falcon.py:1410`；[验证] 裸 `Falcon` 触发 `AttributeError: 'Falcon' object has no attribute '_mdcl_loss'` |
| ② | P1 契约 | `FalconSpeakerContrastive.forward(..., **kwargs)` 把 `near_audio` 吞进 `**kwargs`：不报错、不生效 ⇒ MDCL 在论文配置下**静默失效** | `Falcon_Contrastive.py:94-95`；[验证] 传 `near_audio` 未抛错（§8-[L]） |
| ③ | P1 契约冲突 | `AudioVisualLightningModuleWithMDCL.training_step` 期望 `est = audio_model(...)` 是张量，但 `FalconSpeakerContrastive` 在 train 模式返回 tuple ⇒ 该 litmodule 与论文模型**不兼容**（只兼容裸 `Falcon`） | `av_litmodule_mdcl.py:91` vs `Falcon_Contrastive.py:116-118` |
| ④ | P1 配置 | `audionet_config` 含 `spk_embed_dim`（仅子类接受）⇒ **该 config 无法构建裸 `Falcon`** | `configs/Falcon.yml:15`；[验证] `TypeError: unexpected keyword argument 'spk_embed_dim'` |
| ⑤ | P1 副作用 | `Separator` 用 `dict.pop` 消费 config（`repeats`/`shared`），破坏调用方 dict；同一 config 二次构建得到**静默残缺**模型（49,081 参数） | `Falcon.py:991-996`；[验证] §8-[T] |
| ⑥ | P1 命名/语义 | 顶层 `causal` 开关**惰性**（对参数量、图结构、输出均无影响）；"causal" 实际由 `enc_dec_params.causal` 与 `layer_3.causal` 决定 | `Falcon.py:781,365,929`；[验证] §8-[S] bit-identical |
| ⑦ | P1 命名 | `MaskGenerator.RI_split` 的"实/虚"只是通道分组，不是真 STFT 实虚部；`NeuroSCNetFusion` 实为加性注入（FiLM 版被注释掉） | `Falcon.py:1247-1259`；`Falcon.py:239-265` vs `205-237` |
| ⑧ | P1 死代码 | `PositionalEncoding`/`MultiHeadSelfAttention`/`FeedForwardNetwork`/`GlobalAttention` 整组不可达；`timm.DropPath` 只为它们而存在 | `Falcon.py:545-689,6,590,651`；[验证] 未出现在 `named_modules()` |
| ⑨ | P2 导出 | `models/__init__.py` 的 `__all__=["AVTIGER_causal"]` 指向不存在的名字；`from look2hear.models import *` 抛 `AttributeError`；`get("AVTIGER_causal")` 抛 `ValueError` | `models/__init__.py:4-6,26-41`；[验证] §8-[F] |
| ⑩ | P2 遮蔽 | `import look2hear.models.Falcon as X` 得到的是**类**而不是模块（包属性被 `from .Falcon import Falcon` 覆盖） | `models/__init__.py:1`；[验证] `X = <class '...Falcon.Falcon'>` |
| ⑪ | P2 死配置 | `video_params.layers.layer_1`、`fusion_params.fusion_type/kernel_size/is2d`、`mask_generation_params.mask_generator_type` 均未被读取 | `Falcon.py:926-929,997,1018,1208-1209` |
| ⑫ | P2 死参数 | `DualPathRNN` 的 `kernel_size/stride/num_heads/head_dim/expand_v/use_*`；`VideoBlock` 的 `layers/num_heads/head_dim/expand_v/use_*`；`STFTDecoder` 的 `act_type/norm_type/out_chan`（被 `*args/**kwargs` 吞） | `Falcon.py:336-353,355-365,907-917,926-929,1134-1136` |
| ⑬ | P2 死分支 | `DualPathRNN` 的 `if "Comba" in rnn_type` 分支不可达（`getattr(nn, "Comba")` 会先抛错） | `Falcon.py:372,394-397` |
| ⑭ | P2 死方法 | `Falcon._encode_audio` 全仓库无调用者（NASS 用 `encode_stft`） | `Falcon.py:1345-1359`；grep 无调用 |
| ⑮ | P2 隐含约束 | 架构只支持 `n_src=1`：`STFTDecoder` 的 `view(B*n_src, in_chan=112*n_src, ...)` 与 MaskGenerator 输出通道不符 | `Falcon.py:1168`；[验证] `n_src=2` → `RuntimeError: shape '[2,224,251,129]' is invalid` |
| ⑯ | P2 效率 | `_mdcl_loss_hard` 在 litmodule 里**逐样本 for 循环**调用（每个样本再对 K 个 near 各做一次 STFT），B×K 次 STFT、无批量化 | `av_litmodule_mdcl.py:109-127` |
| ⑰ | P2 一致性 | 训练入口用 `FalconSpeakerContrastive`（NASS），而 MDCL litmodule 与 `_mdcl_loss_hard` 属另一条**未启用**的训练路线；config 里 `spk_contrast` 与模型 `mdcl_*` 是两套互斥机制，但两者默认值不一致（`margin=0.5` vs `mdcl_margin=0.0`） | `configs/Falcon.yml:104-111` vs `Falcon.py:1291-1292` |
| ⑱ | P2 边界 | `STFTEncoder` 的 `unsqueeze_to_2D` 用海象运算符 `:=`（Python≥3.8 ok），但 `assert s[1]==1` 对 `[B,1,L]` 才成立；`[B,2,L]` 会断言失败 | `Falcon.py:1095-1102` |
| ⑲ | 注释 | `Falcon.py:315-316` 用 `-(len(shape)//2)` 取末 2 维，隐含"只支持 4D"；3D 输入会只取末 1 维 | `Falcon.py:315-316,893` |
| ⑳ | 版本 | 未发现 `TODO/FIXME/XXX` 注释（grep 全文件 0 命中）；"未完成"体现在被注释掉的 `FiLMBlock` 与死类上 | `Falcon.py:205-237` |
| ㉑ | **P0 级死参数（R3 新增）** | `AudioBlock.concat_layers[2]` **从未被调用**：`Falcon.py:868` 用 `concat[-1]`（index 3），`:879` 的循环只遍历 `1,0` ⇒ **5,568 参数（占 537,697 的 1.04%）全程闲置**；疑似"level-2 融合缺失"的代码 bug——注释 `:864-867` 说 "Fuse the two deepest levels (Level N-1 and Level N-2)"，但初始融合只用 `concat[-1]` + `downsampled_outputs[-2]` 直接相加，`x_fused[2]` 未再进解码器。**待作者确认是否补调用（本轮禁改源码）** | `Falcon.py:864-888`；[验证] hook 计数 `{0:2,1:2,2:0,3:2}`、15 个 state_dict key、§8.4-[W6]；R2 `layers-dsp` 与 R4 `studio-backend` 均独立复现 |

### 4.10 README 术语 TFRB / GCI 与代码的映射（R3 新增，回应 Lead 要求）

README 图注提到 **TFRB**（Time-Frequency Refinement Block）与 **GCI**，代码里**没有同名类**；映射如下（`[推断]`，但 README 图注与结构可互证）：

| README 术语 | 代码实现 | 证据 |
|---|---|---|
| **TFRB** | `DualPathRNN(dim=4, bidirectional=True)`（频轴双向 LSTM）+ `DualPathRNN(dim=3, bidirectional=False)`（时轴单向 LSTM）+ `MultiHeadSelfAttention2D(causal=True)`（时间轴 masked self-attn）三件套；在 `AudioBlock` 中每级各实例化一份 | README `README.md:61-62` 图注："a bidirectional LSTM along frequency, a unidirectional LSTM along time, then masked 2D self-attention"；代码 `Falcon.py:777-790`（§2.1.5/§2.1.6/§4.5） |
| **GCI** | `InjectionMultiSum`：`local_emb * sigmoid_gate(global) + global_emb` 的门控注入式融合，在 `fusion_layers` 与 `concat_layers` 中用于"每个尺度" | README `README.md:56` 图注："GCI fusion at every scale"；代码 `Falcon.py:267-329,858-888`（§2.1.4） |

补充：README 图注同时说 "a TFRB at the coarsest scale"——与代码"**编码路径只在最深级跑 TFRB**（`Falcon.py:842-845`）"一致；
"decoder of GCI and TFRB blocks" 与"解码路径在 level 2/1/0 各跑一次 TFRB（`:872-888`）+ 每级 GCI"一致。
**注意 TFRB 的第三件套在编码路径上只在瓶颈 T=32 上做全段注意力**（非流式，见 §4.6-④），且 `x_fused[2]` 未进解码器（§4.9-㉑）。


---

## 5. 对外接口契约

### 5.1 构造契约

```python
model = look2hear.models.FalconSpeakerContrastive(   # 或 .Falcon（但见下）
    sample_rate=16000,                                # 必填
    **copy.deepcopy(config["audionet"]["audionet_config"]),   # 必须 deepcopy！
)
```
- **必须 `deepcopy`**：`Separator.__init__` 会 `pop` 掉 `repeats`/`shared`（§3.1 副作用、§4.9-⑤）。
- `audionet_config` 的键集合与子类强耦合：`spk_embed_dim` 只有 `FalconSpeakerContrastive` 接受（§4.9-④）。
- 隐含约束：`n_src=1`（§4.9-⑮）；`pretrained_vout_chan` 必须等于视觉编码器输出通道（512）；`win/hop_length` 只决定音频帧率，
  与视频 25 fps 无关，二者通过 `NeuroSCNetFusion` 的 nearest 插值对齐（50→251，比 5.02:1）。
- 未在 config 中暴露但可用的构造参数：`mdcl_margin=0.0`、`mdcl_alpha=0.15`（`Falcon.py:1291-1292`）；子类还有 `spk_embed_dim=128` 默认。

### 5.2 forward 契约（**返回值随"类 × 模式"三态变化，最常见的误用来源**）

| 调用方 | 模型 | `model.training` | 调用 | 期望返回 |
|---|---|---|---|---|
| `av_litmodule_nass_v2.training_step` | `FalconSpeakerContrastive` | True | `audio_model(mixtures, mouth_emb)` | `tuple(est [B,1,L], sep_emb [B,1,112,T',129])`（`av_litmodule_nass_v2.py:90`） |
| `av_litmodule.validation_step`（基类） | 同上 | **False**（PL 自动 eval） | 同上 | `est [B,1,L]` 张量（`av_litmodule.py:90`） |
| `audio_visual_test.py` | 同上 | False（`audio_visual_test.py:170`） | `model(mix[None], mouth_emb)` | 张量，随后 `.squeeze(0)`（`audio_visual_test.py:231,236`） |
| `studio/falcon_runtime.py` | 同上 | False | `self.audio_model(waveform, mouth_embedding)` | 张量，取 `separated[0,0]`（`falcon_runtime.py:242-243`） |
| `av_litmodule_mdcl.training_step` | 期望裸 `Falcon` | True | `audio_model(mixtures, mouth_emb)` | **tuple ≠ 张量 ⇒ 与子类不兼容**（`av_litmodule_mdcl.py:91,122`） |
| 任意 | 裸 `Falcon` | True | `..., near_audio=...` | **`AttributeError`**（`Falcon.py:1480`） |

**张量契约**：`audio_mixture` float32 `[B,L]`、单声道 16 kHz、**不做峰值/RMS 归一化**（`configs/Falcon.yml:163 normalize_audio: false`）；
`mouth_embedding` float32 `[B,512,T_v]`（2 s ↔ `T_v=50`）；输出 float32 `[B,n_src,L]`，与输入等长（由 `istft(length=L)` 保证）。
`T' = 1+⌊L/hop⌋`、`F = win/2+1 = 129`（L=32000 ⇒ T'=251）。

### 5.3 依赖方向

- **它依赖谁**：`timm.models.layers.DropPath`（`Falcon.py:6`，只被两个死类用）、`look2hear.layers.normalizations`
  （`cLNhw/gLN/GroupNorm/LayerNormalization4D`）与 `activations`（`Falcon.py:7`）、`base_model.BaseModel`（`Falcon.py:5`）。
- **R3 新增：`import look2hear.models` 的隐式硬依赖链**：`Falcon.py:7 from ..layers import normalizations, activations` 会先执行
  `look2hear/layers/__init__.py`（`:1-20` 全量 import），其中 `layers/stft.py:7,10` 需要 **librosa + distutils**、
  `layers/stft_tfgn.py:7-8` 需要 **torch_complex + typeguard**，并经 `..utils` 触发 `look2hear/utils/__init__.py`——`:1` 加载 `utils/stft.py`、
  `:12` 的 `lightning_utils` 顶层 `from pytorch_lightning.utilities import rank_zero_only`（`utils/lightning_utils.py:3-4`）
  ⇒ **`import look2hear.models` 连带加载 librosa / torch_complex / typeguard / pytorch_lightning / rich**（[验证] `sys.modules` 实测；
  交叉引用 02 文档 A03-A06）。Studio 副本正是靠**裁剪 `layers/stft*.py` + 本地化 DropPath**（`studio/falcon_model/layers/` 仅
  `normalizations.py`+`activations.py`）消除这些依赖（06 §4.8）。
- **谁依赖它**：`look2hear/system/av_litmodule*.py`、`audio_train.py`、`audio_visual_test.py`、`evaluated_mac_params.py`、
  `studio/falcon_model/models/*`（推理副本）；`studio/falcon_runtime.py:37` 按架构名做 `MODEL_CLASSES` 分派，并校验
  `config["audionet"]["audionet_name"]` 必须等于**该 variant 配置的 `architecture`**（small=`FalconSpeakerContrastive`、
  big=`SpeakerContrastiveV2`；`falcon_runtime.py:31-37,152-154`）——R3 按 R4 评审 P2-1 精确化。
- **02 域（layers/DSP）**：本模型对 02 的**关键依赖是归一化的统计范围**——`cLNhw` 里 `GlobalLayerNorm` 是否全局统计，
  直接决定 §4.6-③ 的因果性结论；`DualPathRNN` 只用 `nn.LSTM`，不依赖 `layers/rnnlayers.py`。
- **04 域（视觉）**：`FRCNNVideoModel` 输出 `[B,512,T_v]`（`look2hear/videomodels/frcnn_videomodel.py:83-84`），
  **必须**是 `(B,512,T)` 而不是 `(B,T,512)`，否则 `VideoBlock` 的 Conv1d 会把它当 `(B,C=T)`。另：`FRCNNVideoModel.train(self, mode=True)`
  重写后**没有 `return self`**（`frcnn_videomodel.py:90-96`），导致 `FRCNNVideoModel(...).eval()` **返回 None**
  （`nn.Module.eval` 实现是 `return self.train(False)`）。这是 04 域的一个真实 bug，任何链式调用都会炸（[验证] §8-[P]）。

### 5.4 checkpoint / state_dict 契约

- `audio_model.state_dict()` 共 **525** 个 key（[验证]）；`mdcl_window` 是 `persistent=False` **不在其中**（`Falcon.py:1339-1343`）。
- PL checkpoint 里以 `audio_model.` / `video_model.` 前缀存放；`BaseModel.load_state_dict_in_audio` 硬编码剥 12 字符（`base_model.py:47-48`）。
- 三处加载策略**不一致**（风险点）：`audio_visual_test.py:139` 用 `strict=False` 仅打印 missing/unexpected（可能静默加载残缺权重）；
  `studio/falcon_runtime.py:168-169` 用 `strict=True`（不匹配即失败）；`BaseModel.from_pretrain` 对 missing **抛错**、unexpected 只 warn（`base_model.py:65-77`）。
- 任何重构（改名/改形状/增删 buffer）都会破坏 Studio 的 `strict=True` 加载；`mdcl_window` 若改成 persistent 也会（多一个 key）。
- **真实权重实测（R3，[验证] §8.4）**：`falcon.pt` = `dict(state_dict, architecture='FalconSpeakerContrastive', epoch=714)`，
  662 条（audio 525 + video 137），与本地模型 `strict=True` **全等**；`falcon-big.pt` = `dict(state_dict, architecture='SpeakerContrastiveV2', epoch=715)`，
  audio 517 条 / video 137 条（big 属 06 域，此处仅结构核对）。权重 sha256 见 §8.4。

### 5.5 与 Studio 常量的对齐

`SAMPLE_RATE=16000`、`WINDOW_SAMPLES=32000`（2 s）、`VIDEO_FPS=25`、`WINDOW_FRAMES=50`（`studio/falcon_runtime.py:24-27`）——
与本文档的形状推演完全一致（251 帧 ↔ 32000 样本；50 视频帧 ↔ 2 s @25fps）。模型**内部无任何跨窗口状态**
（`PROVENANCE.md`），因此 Studio 只能用"2 s 窗口 + 0.5 s hop + 80 ms 右上下文"的重叠推理——即 §4.6-⑦ 定义的
**窗口级因果（window-causal）**；06 文档 §4.1/§5.5 已给出 `GUARD=1280`（80 ms）的切片语义与实现级验证。

---

## 6. 可验证断言表

置信度：高=直接可复现；中=依赖配置/环境；低=推断。`[代码]`=读码；`[验证]`=本机运行（脚本见 §8）。

| 编号 | 断言 | 证据 path:line | 验证方式 | 置信度 | 条件/协议（R3 新增列） |
|---|---|---|---|---|---|
| A01 | `audionet_config` 共 11 个顶层键，逐键映射见 §3.1 | `configs/Falcon.yml:11-101` | [代码]+[验证] 构造成功 | 高 | —（结构断言） |
| A02 | 模型可训练参数 537,697；含冻结 FRCNN 合计 11,724,385 | `README.md:34,100,206` | [验证] §8-[B][R] | 高 | 随机初始化；参数=parameters() 计数 |
| A03 | `audio_model` = encoder 3,024 + audio_bottleneck 13,104 + video_bottleneck 0 + separator 492,312 + mask_generator 12,657 + decoder 2,016 + spk_pool_proj 14,584 | `Falcon.py:1308-1343` | [验证] 逐模块 `nparam` | 高 | 同上 |
| A04 | 逐项手算公式与实测逐一相等（20 项，见 §4.7） | 本文 §4.7 表 | [验证] §8-[R] 全 OK | 高 | 同上（§8-[R] 20 项） |
| A05 | `video_bottleneck` 因 `kernel_size=-1` 退化为 Identity、0 参数 | `Falcon.py:30,84-85,1312` | [验证] 0 参数、输出=输入 | 高 | 构造 + 属性检查 |
| A06 | 前向 waveform 输出 `[B,1,L]` 且 `L` 等长；`T'=1+⌊L/128⌋`、`F=129` | `Falcon.py:1110-1119,1182-1190` | [验证] L=8000/16000/32000/48000 均吻合 | 高 | B=1；L∈{8000,16000,32000,48000}；eval |
| A07 | train 模式返回 `(est, sep_emb[B,1,112,T',129])`；eval 只回 `est` | `Falcon_Contrastive.py:116-118` | [验证] tuple / Tensor | 高 | B=1,L=32000；mouth [1,512,50]；train/eval |
| A08 | `AudioBlock` 金字塔形状 `(251,129)→(126,65)→(63,33)→(32,17)` | `Falcon.py:749-774,837-848` | [验证] 逐层 hook | 高 | B=1,L=32000；forward hook |
| A09 | `n_freqs` 记账值 `129/65/33/17` 与实测 F 完全一致 | `Falcon.py:747,770-774,785-786` | [验证] 各层 `n_freqs` 属性 | 高 | B=1,L=32000；读各层属性 |
| A10 | 编码路径只在最深级调用 `process_f/t/att`；解码路径调用 level 2→1→0；**例外**：`concat` 只用 `[3],[1],[0]`（`concat[2]` 0 次，见 A51） | `Falcon.py:842-845,868,879-888` | [验证] 每级 process 调用 1 次；concat hook `{0:2,1:2,2:0,3:2}` | 高 | hook；concat 例外见 A51 |
| A11 | 一次 `Separator.forward` 中 AudioBlock 被调 2 次、VideoBlock 1 次、caf 1 次（权重共享） | `Falcon.py:991-996,1030-1051` | [验证] forward hook 计数 | 高 | hook，一次 Separator.forward |
| A12 | `repeats:2`+`shared:true` ⇒ `fusion_repeats=1, audio_repeats=1`，音频块 2 遍同一实例 | `Falcon.py:991-1006` | [验证] 属性+调用计数 | 高 | 属性 + hook 计数 |
| A13 | 视频特征 `[B,512,50]` 经 `v_proj(groups=16)`→`[B,112,50]`→nearest→`[B,112,251]`→广播加 | `Falcon.py:242-265` | [验证] 形状 hook | 高 | hook；B=1；T=50→251 |
| A14 | `RI_split` 复数乘法：`112 = 2×56`，`est_real=e_r m_r − e_i m_i`，`est_imag=e_r m_i + e_i m_r` | `Falcon.py:1247-1259` | [代码] 逐行 | 高 | —（逐行推导） |
| A15 | `MaskGenerator` 输出 `[B,n_src,112,T,F]`，decoder 输入 `in_chan=112·n_src` | `Falcon.py:1325-1336,1168` | [验证] `[1,1,112,251,129]`→`[1,1,32000]` | 高 | B=1,L=32000 |
| A16 | 架构只支持 `n_src=1`；`n_src=2` 前向 RuntimeError | `Falcon.py:1168` | [验证] §8-[F] | 高 | B=1,L=32000；n_src=2 |
| A17 | `STFTDecoder` 时间严格因果（前缀扰动 Δ 恒 0） | `Falcon.py:1170-1177` | [验证] §8-[O] | 高 | 改 embedding 帧≥125；fp32；阈值 1e-12；分界=128 整数倍 |
| A18 | `STFTEncoder` 因 `torch.stft(center=True)` 有 128 样本（8 ms）未来依赖；精确说法：**帧 ≤124 全 0、帧 125 起非零**，`f_first=⌊(s−128)/128⌋+1` | `Falcon.py:1110-1116` | [验证] 帧 ≤124=0、帧 125 起非零；公式三点命中（s=16000/20000/16128） | 高 | 改音频采样≥16000（同分布 randn）；f_first=⌊(s−128)/128⌋+1 |
| A19 | `cLNhw` 的 `GlobalLayerNorm(nn.GroupNorm(1))` 在 `[B,C,T,F]` 全局统计 ⇒ 非因果；经 `CumulateLN4D` 后纯尺度泄漏被大幅抵消（GN-only ×10 探针 Δ=4.40 → cLNhw 整体 1.9e-6） | `layers/normalizations.py:134-144,204-221`；`Falcon.py:1311` | [验证] P1 同分布替换：pre_norm 左段 Δ=1.4e-6；纯 1×1 conv Δ=0；三协议表见 §4.6-⑨ | 高 | P1 同分布替换（单层 pre_norm）；fp32/eval/seed=0；三协议见 §4.6-⑨ |
| A20 | `AudioBlock` 关掉 cLNhw 后仍有 57 帧≈7,296 样本≈456 ms 的 lookahead（对称 k=5 卷积；L0±1/L1±4/L2±8/L3±16 帧）；分界改变时 56/57/58 帧（±1 量化） | `Falcon.py:765,801,814` | [验证] §8-[O]；R3 split=100/125/150 复跑 | 高 | 关掉全部 cLNhw；split=125 → 57 帧；split=100/150 → 56/58 |
| A21a | `InjectionMultiSum` **as-built（3 个 cLNhw 全开）** 脉冲响应是**全局 (0,250)**（cLNhw 的 GN 把单点摊到全窗） | `Falcon.py:283-329,801,814` | [验证] 脉冲 @t=100；R2 `layers-dsp` 与 R4 `studio-backend` 均复现 | 高 | as-built（含 cLNhw）；脉冲 @t=100 → (0,250) |
| A21b | 去掉 cLNhw（norm=Identity）后，`causal=False` 的脉冲响应为 **±2 帧 [98,102]** | `Falcon.py:283-329` | [验证] 脉冲 @t=100 | 高 | 去 cLNhw（norm=Identity）；脉冲 → [98,102] |
| A21c | `causal=True` + 去 cLNhw 时严格因果（前缀 Δ=0，脉冲 [100,104]） | `Falcon.py:283-329` | [验证] 脉冲/前缀 | 高 | causal=True + 去 norm；前缀 Δ=0；脉冲 → [100,104] |
| A22 | `DualPathRNN(dim=3)` 时间单向；`(dim=4)` 频率双向但时间维是 batch ⇒ 两者时间前缀 Δ=0 | `Falcon.py:386-401`；`configs/Falcon.yml:55,65` | [验证] §8-[D] | 高 | 随机输入；前缀扰动；脉冲响应 |
| A23 | `MHSA2D(causal=True)` 前缀 Δ=0；`causal=False` 时**非零（随机输入下量级 ~5，实测 4.68/4.97，非精确值）** | `Falcon.py:520-524`；`configs/Falcon.yml:72` | [验证] §8-[D] | 高 | causal=False 对照：随机输入下量级 ~5（非精确值） |
| A24 | 端到端改未来 1 s，前 0.5 s 输出仍差 ~1e-4（0–0.12 s: 5.1e-7；0.88–1.0 s: 4.2e-2） | 本文 §4.6-⑥ | [验证] §8-[O]；**两次前向共用同一 `mouth_embedding`**、fp32/eval/seed=0；Lead 独立 2.97e-4 | 高 | 改 1.0–2.0 s；两次前向共用同一 mouth_embedding；fp32/eval；Lead 独立 2.97e-4 |
| A25 | 顶层 `causal` 开关惰性：参数量/state_dict/输出 bit-identical | `Falcon.py:781,365,929,1314-1315` | [验证] §8-[S] | 高 | 同权重同输入；比较 state_dict + 输出 |
| A26 | FRCNN 视觉前端时间感受野 ±2 帧 = ±80 ms | `frcnn_videomodel.py:58-70` | [验证] 改帧≥30 → 最左受影响帧 28 | 高 | 改视频帧≥30；B=1,T_v=50 |
| A27 | `VideoBlock` eval 下因果、train 下因 BatchNorm1d 非因果 | `Falcon.py:940-948` | [验证] §8-[P] | 高 | VideoBlock eval/train 两模式 |
| A28 | `mdcl_margin=0.0`/`mdcl_alpha=0.15` 为默认值，config 未提供 | `Falcon.py:1291-1292`；`configs/Falcon.yml:11-101` | [验证] 属性值 | 高 | 读属性；config 无该键 |
| A29 | `_sisnr` 在 20 dB 已知 SNR 上返回 20.000 dB | `Falcon.py:1361-1376` | [验证] §8-[E] | 高 | 去均值；构造 SNR=20 dB 信号 |
| A30 | `_mag_sim` = 对幅度谱展平后套同一 SI-SNR；20 dB 波形 → 16.33 dB | `Falcon.py:1378-1408` | [验证] §8-[E] | 高 | 同上；STFT(win=256,hop=128,hann) |
| A31 | `_mdcl_loss_hard` 的 K=0/1/≥2 与 β 加权公式与手算完全一致、梯度可回传 | `Falcon.py:1410-1462` | [验证] §8-[E][M] `allclose=True` | 高 | K=0/1/3；β=0/1/20；margin=0/±500 |
| A32 | `Falcon.forward(near_audio≠None)` 调不存在的 `_mdcl_loss` ⇒ AttributeError | `Falcon.py:1480` | [验证] §8-[F] | 高 | 裸 Falcon、train()、near_audio |
| A33 | `FalconSpeakerContrastive` 用 `**kwargs` 静默吞掉 `near_audio` | `Falcon_Contrastive.py:94-95` | [验证] 未抛错 | 高 | 子类 train() + near_audio kwarg |
| A34 | `audionet_config` 无法构建裸 `Falcon`（`spk_embed_dim` 未识别） | `configs/Falcon.yml:15`；`Falcon.py:1278-1293` | [验证] TypeError | 高 | config 原样喂裸 Falcon |
| A35 | 同一 config dict 二次构造 ⇒ 参数量 537,697→49,081，且前向不报错 | `Falcon.py:991-996` | [验证] §8-[T] | 高 | 同一 dict 连续构造两次 |
| A36 | `models/__init__.py` 的 `__all__` 指不存在名字，`import *` 失败 | `models/__init__.py:4-6` | [验证] AttributeError | 高 | — |
| A37 | `PositionalEncoding/MHSA/FFN/GlobalAttention` 不在模型图中（死代码） | `Falcon.py:545-689` | [验证] `named_modules` 无此类 | 高 | named_modules 类名检测 |
| A38 | `timm.DropPath` 只被两个死类使用 | `Falcon.py:6,590,651` | [代码] grep | 高 | grep |
| A39 | `video_params.layers`/`fusion_type`/`mask_generator_type` 死配置 | `Falcon.py:926-929,997,1018,1208-1209` | [代码]+[验证] 属性未建图 | 高 | 属性/建图检查 |
| A40 | `AttentionSpeakerPool` 把 `[B,112,T',F]` 池化为 `[B,64]` 且 L2 归一化 | `Falcon_Contrastive.py:41-57,82-91` | [验证] 形状 + ‖z‖₂=1 | 高 | B=1；‖z‖₂ 检查 |
| A41 | NASS 对比损失 = `α_pos(1−cos⁺) + α_neg Σ softmax(β·cos)·relu(cos−margin)`，query 有梯度、pos/neg detached | `nass_contrastive_loss.py:56-81`；`av_litmodule_nass_v2.py:99-134` | [验证] 数值 + 梯度 | 高 | B=1；β 扫描；梯度检查 |
| A42 | `state_dict()` 525 key，`mdcl_window` 不在其中 | `Falcon.py:1339-1343` | [验证] `'mdcl_window' in sd == False` | 高 | state_dict 检查 |
| A43 | `FRCNNVideoModel(...).eval()` 返回 None（`train()` 重写未 return self） | `frcnn_videomodel.py:90-96`；torch `Module.eval` 实现 | [验证] 直接复现 | 高 | 直接复现 |
| A44 | Studio 副本 = 注释中文化 + `DropPath` 本地化 + **MPS CPU FFT bridge**（编码器 `:1111-1122`、解码器 `:1184-1198`；studio 侧净 **+9 行**）；CPU/CUDA 数值等价，MPS 时设备搬运 | `studio/falcon_model/models/Falcon.py:1111-1122,1184-1198`；`PROVENANCE.md:15-21`；06 §4.8/A45 | [代码] 结构化 diff（新增 56/删除 65，净 -9 ↔ studio +9） | 高 | studio↔look2hear 结构化 diff |
| A45 | README "0.5 M" 与实测 0.538 M 属 1 位有效数字取整；不含对比头时为 0.523 M | `README.md:34,206` | [验证]+[推断] | 中 | README 数字 vs 实测 |
| A46 | **`falcon.pt` 真实权重结构**：`dict(state_dict, architecture='FalconSpeakerContrastive', epoch=714)`；`state_dict` 662 条 = `audio_model.*` 525 + `video_model.*` 137；`mdcl_window` 不在其中 | `/root/.cache/falcon_weights/falcon.pt`；`Falcon.py:1339-1343` | [验证] `torch.load(weights_only=False)`（§8.4-[W1]） | 高 | torch.load(weights_only=False)；CPU |
| A47 | **audio 权重与模型全等**：525/525 key 集合相等、0 shape 不一致、`strict=True` 载入成功（0 missing / 0 unexpected） | `configs/Falcon.yml` + `Falcon_Contrastive.py:63-69` | [验证] §8.4-[W2] | 高 | strict=True 载入 falcon.pt（audio） |
| A48 | **video 权重与 `FRCNNVideoModel` 全等**：137/137 key、0 shape 不一致、`strict=True` OK | `look2hear/videomodels/frcnn_videomodel.py` | [验证] §8.4-[W3] | 高 | strict=True 载入 falcon.pt（video） |
| A49 | 由权重重建后参数：audio **537,697** + video **11,186,688** = **11,724,385** | 权重 + 模型类 | [验证] §8.4-[W2][W3] | 高 | parameters() 求和 |
| A50 | `state_dict` 张量 numel **≠** 参数量：audio 537,762（+65 buffer）、video 11,196,308（+9,620 buffer） | `BatchNorm1d(32)`/ResNet BN buffers | [验证] §8.4-[W1]；参数量须用 `parameters()` | 高 | state_dict numel 求和 |
| A51 | `concat_layers[2]` **从未被调用**：hook 计数 `{0:2,1:2,2:0,3:2}`；5,568 参数（537,697 的 **1.04%**）闲置；有效参数 **532,129**；15 个 state_dict key 仍被保存/加载 | `Falcon.py:864-888` | [验证] §8.4-[W6]（R2/R4 评审独立复现） | 高 | hook + 参数统计；B=1,L=32000 |
| A52 | `AudioBlock`（关 cLNhw）lookahead 随分界偏移 **±1 帧**：split=100/125/150 → 56/57/58 帧 | `Falcon.py:765,801,814` | [验证] §8.4-[W7] | 高 | split=100/125/150 |
| A53 | STFT encoder 首变帧公式 `f_first=⌊(s−win/2)/hop⌋+1`：s=16000→125、20000→156、16128→126（三点命中；注意 ≠ `ceil((s−128)/hop)`） | `Falcon.py:1110-1116` | [验证] §8.4-[W8] | 高 | s=16000/20000/16128 |
| A54 | README 术语映射：**TFRB** = 频轴双向 LSTM + 时轴单向 LSTM + causally-masked MHSA2D；**GCI** = `InjectionMultiSum` | `README.md:56,61-62`；`Falcon.py:777-790,267-329`；02 §1/§3 | [代码] 图注 ↔ 结构 | 中（`[推断]` 命名映射） | README 图注 ↔ 代码结构 |
| A55 | `import look2hear.models` 的隐式硬依赖：librosa / torch_complex / typeguard / **pytorch_lightning** / rich 均被连带加载（`Falcon.py:7`→`layers/__init__.py:19-20`→`utils/__init__.py:1,12`） | `layers/stft.py:7,10`；`layers/stft_tfgn.py:7-8`；`utils/lightning_utils.py:3-4` | [验证] `sys.modules` 实测（§8.4） | 高 | import 后 sys.modules |
| A56 | `falcon-big.pt` 结构：`dict(state_dict, architecture='SpeakerContrastiveV2', epoch=715)`；audio 517 条 / video 137 条（big 属 06 域，仅结构核对） | `/root/.cache/falcon_weights/falcon-big.pt` | [验证] §8.4-[W4] | 高 | torch.load(falcon-big.pt) |

---

## 7. 未解疑问 / 需其它子系统确认的问题

| 编号 | 问题 | 需谁确认 |
|---|---|---|
| Q1 | **论文的 0.5 M 口径**到底是"音频模型全部可训练参数（537,697）"、"扣掉训练期专用对比头（523,113）"，还是"另有裁剪（如层共享后按推理图计）"？关系到 README 表与 checkpoint 的一致性 | `loss-metric-eval` / Lead 对照论文 |
| Q2 | **"causal" 的论文主张**：论文是否明确承认 `cLNhw` 的全局统计与对称卷积破坏严格因果？Studio 的 80 ms 右上下文是否就是为 FRCNN ±2 帧（80 ms）设的？ | `studio-backend`（OutputSlicer/80 ms guard 的语义）/ Lead |
| Q3 | `layers/normalizations.py` 的 `cLNhw`（`LayerNormalizationHW`）是否**有意**设计成"先全局后累积"？02 域是否有 `cLN`（纯因果）可选？改成 `cLN` 会掉多少性能（无权重/数据，无法验证） | `layers-dsp` |
| Q4 | `MultiHeadSelfAttention2D` 在 2 s 窗口上 O(T²)（瓶颈 T=32 时很小，但 level-0 若被调用则是 251²）——实际只有瓶颈调用，故开销小；但**能否**安全退化到"只在瓶颈用注意力"这一设计约束是否被文档化？ | `layers-dsp` / Lead |
| Q5 | `av_litmodule_mdcl.py` 与 `FalconSpeakerContrastive` 的 tuple/张量契约冲突：MDCL 路线是否**已废弃**？仓库里是否有更早的 `Falcon`（无对比头）训练配置（如 `FalconBig.yml` 用 `SpeakerContrastiveV2`）？ | `train-sys` |
| Q6 | `configs/Falcon.yml` 的 `spk_contrast.margin=0.5` 与模型 `mdcl_margin=0.0` 是两套机制；论文实际训练用的是 NASS（`system: AudioVisualLightningModuleNASSv2`）还是 MDCL？`Experiments/checkpoint/Falcon/conf.yml` 快照里是否有覆盖？ | `train-sys` |
| Q7 | `FRCNNVideoModel.train()` 缺 `return self`（`eval()` 返回 None）是否已在评测/Studio 侧踩过坑？`audio_visual_test.py:170` 是语句式调用故未暴露 | `loss-metric-eval` / `studio-backend` |
| Q8 | Studio 的 80 ms 右上下文与本文 §4.6-⑤ 的 FRCNN ±80 ms 是否**同一设计意图**（若是，应在 06 文档中把常量与感受野对应起来） | `studio-backend` |
| Q9 | MACs 10.6 G / RTX3090 51.3 ms / CPU 1361.2 ms（`README.md:206`）无法在无权重环境下复核；`evaluated_mac_params.py` 用 `ptflops` 对 Audiomodel 单独统计（`evaluated_mac_params.py:88-101,122`），是否需要 04 域用 CPU 复算一次 MACs？ | `loss-metric-eval` |
| Q10 | `Falcon.forward` 的 `_mdcl_loss`/`_mdcl_loss_hard` 命名不一致是**重构遗留**还是另有子类覆盖（`studio/falcon_model/models/SISNRRepel.py` 里同样存在）？是否需要修复（本轮禁止改源码）？ | Lead（决定是否作为 P0 记录进 FINAL） |
| Q11 | 训练时 `vis` 分支 `BatchNorm1d` 用批内统计是否会与"因果"宣称冲突（推理 eval 用 running stats 无问题） | `train-sys` |
| Q12 | 是否需要 02 域确认：`layers/stft.py`（785 行）与 `utils/stft.py`（785 行）都**没有**被 `Falcon.py` 使用（本模型直接用 `torch.stft/istft`）——重复实现的清理是否影响其它模型 | `layers-dsp` |
| Q13 | **`concat_layers[2]` 是有意跳过还是 `AudioBlock.forward` 漏调用？** R3 已确认它从未被调用、5,568 参数闲置（疑似 bug）；若属代码 bug，FINAL 需在"0.5 M 口径"处注明"其中 1.04% 不参与前向"，并建议补一行 `concat[2](x_fused[2], expanded)`（本轮禁改源码） | Lead / 作者确认；`train-sys` 评估重训成本 |
| Q14 | **"窗口级因果（window-causal）"是否为 FINAL 的统一措辞？** 我已按此改写 §4.6-⑦ 并交叉引用 02/06；若 Lead 采用，请 02/06 在 R3 同步同一句 | Lead（FINAL 术语）+ `layers-dsp`/`studio-backend` |
| Q15 | **真实权重的 epoch=714（small）/715（big）** 与论文表格（RealSSA/LRS2/LRS3/VoxCeleb2）的对应关系？权重仅含 `state_dict/architecture/epoch`，不含训练超参 | Lead（对照论文/README）+ `loss-metric-eval` |
| Q16 | **cLNhw 泄漏量级在 FINAL 用哪个协议？** 建议同时给"P1 同分布替换（1.4e-6）+ P4 波形×10（3.2e-5）"两条，避免单协议误导（§4.6-⑨） | Lead（FINAL 数字口径） |

---

## 8. 本轮验证活动记录

**环境**：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128，CPU）；必须注入 stub：
`PYTHONPATH=$PWD/docs/verification/stubs:$PWD`（`timm.models.layers.DropPath` + `typeguard` stub，不改源码）。
R1/R2 阶段无权重 ⇒ 随机张量结构/数值验证；**R3 起 Lead 提供真实权重** `/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`，
已用于 §8.4 的"key/shape/strict 加载/参数口径"验证（仍无数据集，**不做训练、不算精度/MACs**）。

### 8.1 我跑的脚本（均已留在仓库，可复跑）

```bash
cd /root/code/github_repos/Falcon-fork
PY=$PWD/docs/verification/stubs:$PWD
/root/.venvs/falcon/bin/python -c 'pass'   # 环境自检
PYTHONPATH=$PY /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify.py  > docs/verification/runs/model-core-verify.log  2>&1
PYTHONPATH=$PY /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify2.py > docs/verification/runs/model-core-verify2.log 2>&1
PYTHONPATH=$PY /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify3.py > docs/verification/runs/model-core-verify3.log 2>&1
PYTHONPATH=$PY /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify4.py > docs/verification/runs/model-core-verify4.log 2>&1
PYTHONPATH=$PY /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify5.py > docs/verification/runs/model-core-r3-verify.log 2>&1   # R3 合并日志（verify5 输出已并入）
```

| 脚本 | 覆盖 | 结果摘要 |
|---|---|---|
| `model_core_verify.py` | [A] 构造/配置键消耗 [B] 参数总账+逐层拆解+死类检测 [C] 形状链 [D] 逐组件因果性+脉冲响应 [E] MDCL 数值 [F] 死代码/抗误用/归一化开关 | 参数 537,697；分隔符 492,312；形状链与 §3.2 完全一致；`_mdcl_loss` 缺失 AttributeError 复现；`n_src=2` RuntimeError；525 keys |
| `model_core_verify2.py` | [G] 端到端前缀扰动 [H] lookahead 定量（**该版实现有 bug，见 8.3**） [I] 视频因果性（**因 8.3-① 崩溃**） | [G] 八段 Δ 已用于 §4.6-⑥；[H]/[I] 作废，改用 verify4 |
| `model_core_verify3.py` | [L] 返回类型契约 [M] MDCL 非零损失/梯度/β/margin [N] 与 litmodule 契约（静态） | train=`tuple[(1,1,32000),(1,1,112,251,129)]`、eval=Tensor；`_mdcl_loss_hard` 与手算 allclose；margin 截断生效 |
| `model_core_verify4.py` | [O] 严格因果/lookahead 定量 [P] 视频分支因果（含 FRCNN） [Q] causal 开关（**该版实现有 dict 污染，见 8.3**） | encoder 8 ms；decoder 0；AudioBlock(norm off) 57 帧；AudioBlock 实际含全局归一化 → 无界；FRCNN ±2 帧；[Q] 由 verify5-[S] 重做 |
| `model_core_verify5.py` | [R] 20 项参数量手算复核 [S] causal 开关重做（每次全新 deepcopy） [T] config 二次构建塌缩复现 | 20/20 手算=实测；causal 两模型 **bit-identical**；二次构建 537,697→49,081 且前向不报错 |

### 8.2 与 Lead 独立验证的交叉核对（[验证] 级，可引用）

| 结论 | Lead 值（`docs/verification/runs/lead-model-params.log`） | 我的复算 | 一致？ |
|---|---|---|---|
| audio_model 参数 | 537,697 | 537,697 | ✓ |
| FRCNN 参数 | 11,186,688 | 11,186,688 | ✓ |
| 合计 | 11,724,385 | 11,724,385 | ✓ |
| 逐模块拆分（encoder/audio_bn/video_bn/separator/mask/decoder/spk） | 3,024 / 13,104 / 0 / 492,312 / 12,657 / 2,016 / 14,584 | 同 | ✓ |
| `video_bottleneck` 0 参数、Identity 语义 | ✓ | ✓（属性 `conv=Identity`、`norm=Identity`） | ✓ |
| L=32000 → T'=251、F=129；train 返回 tuple | ✓ | ✓ | ✓ |
| 改后半段音频，前缀仍有 ~1e-4~1e-3 泄漏（cLNhw 全局统计） | 2.97e-4 | 1.13e-4（0–0.5 s 段；0–0.12 s 仅 5.1e-7） | 同量级 ✓ |
| `_mdcl_loss` 缺失 → AttributeError | ✓ | ✓（裸 `Falcon`；子类被 `**kwargs` 吞掉，见 §4.9-②） | ✓ 且补充了子类行为 |

### 8.3 失败/作废记录（诚实标注）

1. **`model_core_verify2.py` 在 [I] 崩溃**：`FRCNNVideoModel(pretrain=None).eval()` 返回 `None`——
   因为 `frcnn_videomodel.py:90-96` 的 `train(self, mode=True)` 重写没有 `return self`，而 `nn.Module.eval()` 实现是
   `return self.train(False)`。这不是我的脚本 bug，而是**真实代码缺陷**（04 域）；改成分语句调用后验证通过。
2. **`model_core_verify2.py` 的 [H] 作废**：`lookahead()` 把"输出索引"与"输入分界"放在不同单位空间相减（如 16000 样本 vs 251 帧），
   得到无意义的负数；已在 `verify4-[O]` 用统一单位重做。
3. **`model_core_verify4.py` 的 [Q] 作废**：两次构造共享同一份嵌套 config dict，第一次 `Separator` 的 `pop` 污染了第二次，
   得到 557,281 vs 68,665 的假差异（真实差异其实来自 `spk_embed_dim` 被剔除后回落到默认 128）。已在 `verify5-[S]`
   用"每次 `deepcopy` 全新构造"重做，结论：causal 开关 bit-identical。
4. **Lead 脚本 `verify_model_params.py` 曾报** `TypeError: Falcon.__init__() got an unexpected keyword argument 'spk_embed_dim'`——
   这不是脚本错误，而是**配置与裸 `Falcon` 不兼容**的证据（§4.9-④），脚本后续改走 `FalconSpeakerContrastive`。
5. **未做**：MACs/延迟复算（ptflops 对自定义输入构造需额外处理，且属 04 域）；真实权重端到端（无权重，遵守环境限制）；
   训练收敛/精度（无数据）。这些列入 §7-Q1/Q9。

### 8.4 R3 验证：真实权重 + `concat[2]` 闲置 + Studio 副本 diff + 分界敏感性（`[验证]`）

脚本：`docs/verification/scripts/model_core_r3_verify.py`；日志：`docs/verification/runs/model-core-r3-verify.log`。

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_r3_verify.py \
  > docs/verification/runs/model-core-r3-verify.log 2>&1
sha256sum /root/.cache/falcon_weights/falcon.pt /root/.cache/falcon_weights/falcon-big.pt
# a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c  falcon.pt
# ce2a92a3188e8069e979f23aaec21cb6d439170aaf4002b4eaa234f0486e037e  falcon-big.pt
```

| 小节 | 内容 | 结果摘要 |
|---|---|---|
| [W1] | `falcon.pt` 顶层结构 | `dict`，keys=`state_dict/architecture/epoch`；`architecture='FalconSpeakerContrastive'`、`epoch=714`；662 条 = audio **525** + video **137**；`mdcl_window` 不在其中 |
| [W2] | audio 权重 ↔ `FalconSpeakerContrastive` | key 集合相等（525/525）、0 shape 不一致、`strict=False` missing/unexpected=0、**`strict=True` 载入 OK**；载入后参数 **537,697** |
| [W3] | video 权重 ↔ `FRCNNVideoModel` | 137/137 key、0 shape 不一致、`strict=True` OK；参数 **11,186,688**；audio+video=**11,724,385** |
| [W4] | `falcon-big.pt` | `architecture='SpeakerContrastiveV2'`、`epoch=715`；audio **517** 条（state_dict numel 1,158,746）/ video 137 条（11,196,308）（big 属 06 域，仅结构核对） |
| [W5] | Studio ↔ 训练侧 `Falcon.py` diff | 新增 56 / 删除 65（净 **-9**，即 studio 多 9 行）；去注释/空白后新增 8、删除 15，语义差异 = DropPath 来源 + MPS bridge（4+4 行）+ device 参数 |
| [W6] | `concat_layers[2]` 闲置 | hook（一次 model forward，AudioBlock×2）：`{0:2,1:2,2:0,3:2}`；`fusion={0:2,1:2,2:2,3:2}`；concat[2]=**5,568** 参数（1.04%），有效参数 **532,129**，15 个 state_dict key |
| [W7] | A20 分界敏感性 | split=100/125/150 → lookahead **56/57/58 帧**（±1 帧量化） |
| [W8] | encoder 首变帧公式 | `f_first=⌊(s−128)/128⌋+1`：s=16000→**125**、20000→**156**、16128→**126**（三点全部命中） |
| 附加 | `import look2hear.models` 隐式依赖 | `sys.modules` 实测含 `librosa`、`torch_complex`、`typeguard`、**`pytorch_lightning`**、`timm`、`look2hear.layers.stft`、`look2hear.utils.stft`（链路见 §5.3；`pytorch_lightning` 来自 `utils/lightning_utils.py:3-4`，**不是** `BaseModel.serialize`） |

**与其它域的交叉核对**：R2 `layers-dsp` 评审与 R4 `studio-backend` 复核独立复现了 [W6] 的 `concat[2]=0` 与 57 帧 lookahead；
`06-studio-backend.md` §4.8 给出同一组 Studio diff 行号与 +4/+4 行数；`02-layers-dsp.md` A03-A06 给出 import 链路证据。四方一致。


---

## 9. 变更日志（Changelog）

| 日期 | 轮次 | 改了什么 | 采纳的 review 意见 |
|---|---|---|---|
| 2026-10-21 | R1 初版 | 建立 8 节模板全文：逐类解析（17 个类）、构造链逐键映射、完整 forward 形状链、因果性分析、参数量 20 项可复算拆解、5 类接口契约、45 条断言、12 条未解疑问、5 个验证脚本与失败记录 | —（R1 无 review） |
| 2026-10-21 | **R3 修订** | ① 修 P0：`concat_layers[2]` 从未调用（§2.1.4/§2.2/§3.2/§4.7/§4.9-㉑/A10/A51）；② A21 拆三口径（as-built 全局/去 cLNhw ±2/causal=True 因果）；③ A44 补 MPS CPU FFT bridge；④ 新增 §4.6-⑨ **因果性探针协议表**（对齐 01/02 的 1.2e-3 / 4.7e-5 / 1.4e-6，并给 CumulateLN4D 抵消机理）；⑤ 统一"窗口级因果（window-causal）"措辞（§4.6-⑦，交叉引 02/06）；⑥ 新增 §4.10 TFRB/GCI 映射、§5.3 import 硬依赖链、§8.4 真实权重验证；⑦ 断言表加"条件/协议"列并新增 A46-A56；⑧ 修正 §4.6-② MHSA2D Q/K/V 措辞、A20 分界敏感性、A23 量级、A24 mouth 固定；⑨ §7 新增 Q13-Q16 | R2 `layers-dsp`（P0×1、P1×3、P2×6）+ R4 `studio-backend`（P0×1、P1×3、P2×4）+ Lead R3 指令 1–4，逐条响应见 §9.1 |

### 9.1 对 R2/R4 评审意见的逐条响应（R3）

**图例**：✅ 采纳；🟡 部分采纳（说明原因）；❌ 不采纳（说明理由）。

| 来源 | 编号 | 意见 | 响应 | 落地位置 |
|---|---|---|---|---|
| R2 layers-dsp | P0-1 | `concat_layers[2]` 从未调用、5,568 参数闲置、§4.9 未登记 | ✅ 采纳：修正 §2.1.4"实例化 8/调用 7"、§3.2 调用计数、§4.7 参数表加"闲置参数"行、§4.9 新增 ㉑、A10 加例外、新增 A51 | §2.1.4、§2.2、§3.2、§4.7、§4.9-㉑、A10、A51 |
| R2 layers-dsp | P1-1 | A21"±2 帧"缺"去 cLNhw"口径（as-built 为全局响应） | ✅ 采纳：拆 A21a（as-built (0,250)）/A21b（去 cLNhw [98,102]）/A21c（causal=True 去 norm [100,104]）；§2.1.4 同步 | §2.1.4、A21a-c |
| R2 layers-dsp | P1-2 | A44"仅注释/DropPath 不同"不完整，漏 MPS CPU FFT bridge | ✅ 采纳：A44 改为三项并列（注释 + DropPath + MPS bridge，studio `:1111-1122`/`:1184-1198`，净 +9 行）；§3.4、§5.3 交叉引用 06 §4.8 | A44、§3.4、§8.4-[W5] |
| R2 layers-dsp | P1-3 | A19/A24 数值缺扰动协议（与 02 的 4.7e-5 不可比） | ✅ 采纳：新增 §4.6-⑨ 五协议表；A19/A24 加"条件/协议"列；裁定 1.2e-3/4.7e-5/1.4e-6 **三者都正确、协议不同**；补 CumulateLN4D 抵消机理 | §4.6-⑨、A19、A24、A46-A50、A55 |
| R2 layers-dsp | P2-1 | §4.6-② 把 MHSA2D Q/K/V 列为"开启因果"是笔误（实为 `causal=False`） | ✅ 采纳：改为"Q/K/V/concat_proj 均显式 causal=False（k=1 无害）；因果来自 tril mask" | §4.6-② |
| R2 layers-dsp | P2-2 | A24 复现必须固定 `mouth_embedding` | ✅ 采纳：A24 条件列写明"两次前向共用同一 mouth_embedding"（我 R1 脚本即如此，之前未在断言写明） | A24 条件列、§8-[O] |
| R2 layers-dsp | P2-3 | A20 的 57 帧在 split=150 时为 58，需注明 ±1 帧 | ✅ 采纳：A20 加"56/57/58 帧（±1 量化）"，新增 A52 与 §8.4-[W7] | A20、A52、§4.6-② |
| R2 layers-dsp | P2-4 | 补"解码器 3 帧 lookback 吸收 iSTFT 扩散"的说明 | ✅ 采纳：§4.6-① 末句写明"输出采样 s 只依赖嵌入帧 ≤⌊s/hop⌋，不额外增加未来依赖" | §4.6-① |
| R2 layers-dsp | P2-5 | A22/A23 的 Δ=4.68 是随机量，应标量级 | ✅ 采纳：A23 改为"随机输入下量级 ~5（实测 4.68/4.97，非精确值）" | A23 |
| R2 layers-dsp | P2-6 | A10 证据行不含 concat | ✅ 采纳：A10 重写为"process 每级 1 次 + concat 例外见 A51" | A10 |
| R4 studio-backend | P0-1 | 同 R2 P0-1（R3 前未修，判阻塞） | ✅ 采纳：见上（R3 已修） | 同上 |
| R4 studio-backend | P1-1 | A21 口径 | ✅ 采纳（同 R2 P1-1） | A21a-c |
| R4 studio-backend | P1-2 | A44 补 MPS bridge（本域给出精确行号） | ✅ 采纳：采用其行号 `:1110-1122`/`:1183-1198` 与本轮实测 `:1111-1122`/`:1184-1198`（差 1 行=注释行，已在 A44 注明"含注释"） | A44 |
| R4 studio-backend | P1-3 | A19 加扰动协议 | ✅ 采纳：§4.6-⑨ 三/五协议表 | §4.6-⑨ |
| R4 studio-backend | P2-1 | §5.3 的 `audionet_name` 校验只写 small，实际按 variant 的 architecture 比较 | ✅ 采纳：改为"必须等于该 variant 的 `architecture`（small/big 不同）" | §5.3 |
| R4 studio-backend | P2-2 | §4.7 `concat_layers ×4=22,272` 未区分实例化/调用 | ✅ 采纳：改为"实例化 ×4（22,272），调用 3 个，其中 5,568 闲置" | §4.7 |
| R4 studio-backend | P2-3 | 建议 A18 写首变帧公式 `ceil((s−128)/hop)` | 🟡 部分采纳：**公式本身需修正**——`ceil((16000−128)/128)=124` 与实测（帧 125 首变）不符；精确公式为 `f_first=⌊(s−win/2)/hop⌋+1`（三点验证，A53）。已在 §4.6-①/A18/A53 写清 | §4.6-①、A18、A53 |
| R4 studio-backend | P2-4 | §4.6-③ 的 1.4e-6（单层）与 §4.6-⑥ 全模型 5.1e-7 需说明口径 | ✅ 采纳：§4.6-③ 标注"P1 协议、单层 pre_norm"，A19/A24 条件列区分"单层/全模型" | §4.6-③、§4.6-⑨ |
| R4 studio-backend | §4-1 | `concat[2]` 是"参数/算力账"实质缺口，建议 §4.7 表尾加"闲置参数"行 | ✅ 采纳：§4.7 新增"闲置参数 −5,568 / 有效 532,129"两行，并在 0.5 M 口径讨论中给出 532,129（再扣对比头 517,545） | §4.7 |
| R4 studio-backend | §4-2 | 裸 `Falcon` + near_audio 的崩溃应并入 §5.2 契约表 | ✅ 采纳（R1 已在 §5.2 末行，本轮在 §4.9-① 加"裸类误用会崩"标注并互链） | §5.2、A32 |
| R4 studio-backend | §4-3 | 建议补"若要流式，改动点→参数/数值→需重训？"表 | 🟡 部分采纳：§4.6-⑦ 已给出四类改动点并标注"前两项会改变参数/统计语义 ⇒ 需重训，后两项可保权重"；完整影响表建议放 FINAL（跨 02/03 域） | §4.6-⑦ |
| R4 studio-backend | Q5 | `concat[2]` 是"有意跳过"还是代码 bug？FINAL 必须给结论 | 🟡 部分采纳：R3 只能给出**证据与倾向性判断**（疑似 bug：注释说 fuse 两层但 `x_fused[2]` 未再进解码器），**无法确证作者意图**；已登记 §4.9-㉑ 并升级为 Q13（属 Lead/作者决定，本轮禁改源码） | §4.9-㉑、Q13 |
| Lead R3 指令 | 1 | P0 concat 修正 | ✅ 采纳（同上） | §2.1.4/§3.2/§4.7/§4.9 |
| Lead R3 指令 | 2 | P1 三条（A21/A44/A19） | ✅ 采纳（同上） | §2.1.4/A21/A44/§4.6-⑨ |
| Lead R3 指令 | 3 | 保留强化 A18/A43；补 TFRB/GCI；§5.3 import 链；统一窗口级因果 | ✅ 采纳：A18 加公式、A43 保留并交叉引 06；新增 §4.10（TFRB/GCI）、§5.3 import 链（新增 A55）、§4.6-⑦ 统一措辞。**说明**：Lead 写的"A18/A36"按语义判定为 **A18（encoder lookahead）** 与 **A43（eval() 返回 None）**；A36（`__all__` 导出问题）内容正确但与本条无关，已保留 | A18、A43、§4.10、§5.3、A55、§4.6-⑦ |
| Lead R3 指令 | 4 | 更新 sha256、Changelog 加 R3、断言表加"条件/协议"列 | ✅ 采纳：Changelog 本行；断言表已加列；sha256 见 §9.2（自指不可内嵌） | §9、§6 |
| R3 自我更正 | — | R2 评审 02 时我把"import 会拉入 pytorch_lightning"判为错误（依据其日志未列 PL）；R3 实测 **PL 确实被加载**（经 `utils/lightning_utils.py:3-4`），02 的结论正确、只是归因（BaseModel.serialize）不对 | 🟡 已更正：§5.3 按实测写；`docs/reviews/round1/02-layers-dsp-review-by-model-core.md` 末尾追加更正说明；建议 02 作者把归因改为 `utils/lightning_utils` | §5.3、A55、R2 评审文件更正 |

### 9.2 版本与 hash 记录（R3）

- R1 版 sha256：`cd58a08cd7775b8ec8fb91ce13b5f9e053977afb981f0b3bd2a6c8f4cd11b30b`（807 行，R2/R4 评审所据版本）。
- **R3 版 sha256：因"文档内记录自身 hash"会改变 hash（自指），记录于 `docs/verification/model-core.md` §"R3 版本锚点"与给 Lead 的汇报中**。
- 改动前备份对照：R1 版 807 行 → R3 版行数以最终文件为准（新增 §4.6-⑨、§4.10、§8.4、§9.1、§9.2 与 A46-A56）。



