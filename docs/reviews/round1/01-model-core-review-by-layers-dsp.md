# 评审：`01-model-core.md` —— 评审人 `layers-dsp`

> 被评审文档：`docs/understanding/01-model-core.md`（807 行，R1 初版）
> 评审时 sha256（评审开始）：`cd58a08cd7775b8ec8fb91ce13b5f9e053977afb981f0b3bd2a6c8f4cd11b30b`
> 评审时 sha256（评审结束）：`cd58a08cd7775b8ec8fb91ce13b5f9e053977afb981f0b3bd2a6c8f4cd11b30b`（**未变化**，本评审基于此版本）
> 证据基线 commit `7576854`；我的独立复核脚本 `docs/verification/scripts/review_model_core_checks.py`，
> 日志 `docs/verification/runs/review-model-core-checks.log`；另有 R1 文档 `docs/understanding/02-layers-dsp.md`
> 与 `docs/verification/runs/layers-dsp-verify.log` 可直接复用的事实。
> 环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0，CPU），stub `PYTHONPATH=$PWD/docs/verification/stubs:$PWD`。

## 1. 结论

**需修订（Minor revision 偏中等）**。文档整体质量很高：8 节模板完整、45 条断言、结论先行、证据密度高，且**大部分核心断言经我独立复算/动态验证成立**
（参数 537,697/11,724,385、金字塔形状、n_freqs 记账、调用次数、A17~A24 因果性、A26/A27、A34/A35、A43 等全部精确复现）。
但存在 **1 个 P0 事实错误、3 个 P1 重要遗漏/误导、6 个 P2 细节**，其中：

- **P0-1**：`§3.2` 称 "`fusion[i]` 与 `concat[i]` 在每一级各调用 1 次" —— **`concat_layers[2]` 从未被调用**，
  其 **5,568 参数（占 0.538 M 的 1.04%）在整个 forward 中完全闲置**；`§2.1.4` 的"被用 8 次"应为"实例化 8 / 调用 7"，
  且 `§4.9` 死代码清单漏登该参数。这可能不是单纯文档问题，而是 `AudioBlock.forward` 少了一次 level-2 融合的**代码 bug**。
- **P1**：A21 的"InjectionMultiSum 脉冲响应 ±2 帧"**只在把 cLNhw 换成 Identity 后成立**；as-built 响应是全局（0,250）；
- **P1**：A44 "Studio 副本仅注释/DropPath 来源不同，逻辑等价"不完整——还有 MPS CPU FFT bridge（功能性设备搬运，与 PROVENANCE 一致）；
- **P1**：A19 的 1.4e-6 与 02 文档 R1 的 4.7e-05 未给扰动协议，读者会误以为结论冲突（实际同结论、不同口径）。

**问题计数：P0 = 1；P1 = 3；P2 = 6。**

## 2. 断言逐条核对表

✅=正确；❌=错误；❓=无法确认/需作者说明。编号沿用被评审文档。证据列给"我的复核方式 + 结果"。

### 2.1 参数/结构类

| 编号 | 结论 | 我的复核（独立于 01 的脚本） | 备注 |
|---|---|---|---|
| A02 | ✅ | `FalconSpeakerContrastive(cfg)` 参数 **537,697**；FRCNN **11,186,688**；合计 **11,724,385** | 与 Lead 日志、02 文档一致 |
| A03 | ✅ | 逐模块：encoder 3,024 / audio_bottleneck 13,104 / video_bottleneck 0 / separator 492,312 / mask 12,657 / decoder 2,016 / spk_pool_proj 14,584 | 全部精确命中 |
| A04 | ✅ | `§4.7` 表逐项复算：audio_net **445,766**、video_net **42,850**、caf **3,696**；downsample 每级 `[896,1920,1920,1920]`、att 每级 `[51,693,31,213,20,973,15,853]`、fusion/concat 每级 5,568、process_f 20,480、process_t 44,480 —— 与文档逐项一致 | 公式正确（含 `2·96` 的 bias 项） |
| A05 | ✅ | `video_bottleneck.conv is nn.Identity`，0 参数 | — |
| A08 | ✅ | forward hook 实测 `down[0..3] = [(251,129),(126,65),(63,33),(32,17)]`；AudioBlock 输出 `(1,112,251,129)` | 精确 |
| A09 | ✅ | `process_att_layers[i].n_freqs == [129,65,33,17]`，与实测 F 一致 | 精确 |
| A10 | ✅（但见 P0-1） | 每次 AudioBlock 内 `process_{f,t,att}` 各 level 恰好 1 次；一次 Separator.forward 调 AudioBlock 2 次 ⇒ hook 计数各 2 | process 部分正确，**concat 部分错误** |
| A11 | ✅ | hook 计数：AudioBlock=2、VideoBlock=1、caf=1 | 精确 |
| A12 | ✅ | `fusion_repeats=1, audio_repeats=1, audio_shared=True, video_shared=True, fusion_shared=True`；`audio_net`/`video_net` 为单实例（非 ModuleList） | 精确 |
| A42 | ✅ | `len(state_dict)=525`，`'mdcl_window' in sd == False` | 精确 |
| 38 个 ConvNormAct | ✅ | `type(m).__name__=='ConvNormAct'` 且 `__module__=='look2hear.models.Falcon'` 的实例数 = **38** | 文档 §2.1.1 正确 |

### 2.2 因果性类（Lead 指定 A17~A24）

| 编号 | 结论 | 我的复核 | 备注 |
|---|---|---|---|
| A17 | ✅ | 改 mask 帧 ≥120：输出 `samples<15360` max\|Δ\| = **0.0**（严格）；`≥15360` 非零 | 与"3 帧内部 lookback"推演一致：输出帧 n 依赖输入帧 n−3..n−1 |
| A18 | ✅ | 改音频 samples≥16000：encoder 帧 123/124 \|Δ\|=0，帧 125 起非零（实测 4.019） | 与 8 ms 完全吻合 |
| A19 | ✅（数值需加权说明，见 P1-3） | 复现其协议（`encoder→audio_bottleneck.pre_norm`，后半段替换为 randn）：左段 **9.5e-07~1.4e-06**；"去掉 cLNhw 的 1×1 conv"确实为 0（encoder 帧 <125 严格不变 + k=1） | **与 02 文档 4.7e-05 的差异已定位为协议不同**，见 §3-P1-3 |
| A20 | ✅ | 把 AudioBlock 内全部 `LayerNormalizationHW` 换成 `Identity` 后：扰动帧≥125 → 最早受影响输出帧 **68** ⇒ **lookahead=57 帧=7,296 采样=456 ms**（精确复现） | 但 split=150 时测得 58 帧（P2-3） |
| A21 | ❌（口径缺失，P1-1） | **as-built（含 cLNhw）脉冲响应 = (0,250) 全局**；把 cLNhw 换 Identity 后 = **(98,102)** ✅；`causal=True`+去 norm = (100,104)；`causal=True`+保留 cLNhw = (0,250) 全局。前缀：去 norm+causal=False → 2.434；去 norm+causal=True → 0 | 结论"±2 帧/改 causal 即因果"只在**去 norm**口径成立 |
| A22 | ✅ | `DualPathRNN(dim=3)` 与 `(dim=4)` 左段 t<5 max\|Δ\| 均为 **0.0** | 精确 |
| A23 | ✅ | `MHSA2D(causal=True)` 左段 = **0.0**；`causal=False` 左段 = **4.971**（01 写 4.68，随机输入不同但同量级） | 见 P2-5 |
| A24 | ✅ | 复现其协议（共用同一 mouth、改 1.0–2.0 s 为 randn，8×2000 采样）：`3.4e-07 / 2.95e-06 / 4.2e-05 / 1.07e-04 / 5.2e-04 / 3.5e-03 / 1.37e-02 / 4.3e-02` —— 与 01 表 `5.1e-7/3.0e-6/2.3e-5/1.1e-4/8.3e-4/3.2e-3/1.5e-2/4.2e-2` **同量级且单调趋势一致** | **必须固定 mouth**，否则假泄漏 ~1e-1（P2-2） |

### 2.3 视频分支 / 配置副作用 / Studio 副本

| 编号 | 结论 | 我的复核 | 备注 |
|---|---|---|---|
| A26 | ✅ | FRCNN 改帧≥30 → 输出最早受影响帧 = **28**（精确复现）；`frontend3D` 结构 = Conv3d(k=(5,7,7), pad=(2,3,3)) + BN3d + PReLU + MaxPool3d(k=(1,3,3)) ⇒ 时间 ±2 帧 = ±80 ms | 精确 |
| A27 | ✅ | `VideoBlock(eval)` 左段 t<30 = **0.0**；`VideoBlock(train)` = **4.825e-02**（01 写 4.4e-2，随机输入差异） | 结论正确 |
| A34 | ✅ | 裸 `Falcon(sample_rate=16000, **audionet_config)` → `TypeError: Falcon.__init__() got an unexpected keyword argument 'spk_embed_dim'` | 精确 |
| A35 | ✅ | 同一 dict 二次构造：**537,697 → 49,081**；第二次 `audio_net=ModuleList(len=0)`、`video_net=ModuleList(len=0)`、`fusion_repeats=0`、`audio_repeats=0`；**forward 成功返回 tuple**（静默错误） | 精确；`49,081 = 3,024+13,104+3,696+12,657+2,016+14,584` 可复算 |
| A43 | ✅ | `FRCNNVideoModel(pretrain=None).eval()` 返回 **None**（`train()` 重写无 `return self`，`frcnn_videomodel.py:90-96`） | 精确 |
| A44 | ❌（不完整，P1-2） | `Falcon_Contrastive.py` diff=**0** ✅；但 `Falcon.py` diff=**121 行**（65 增/56 删），其中**非注释新增 15 行 / 非注释删除 8 行**，含 **MPS CPU FFT bridge**（encoder `studio/.../Falcon.py:1110-1123`、decoder `:1183-1198`）——这是功能性差异，不是"仅注释/DropPath" | PROVENANCE.md:20-21 自己写了 MPS 改动；02 文档 §4.6 已给出精确 diff |

### 2.4 其它抽查

| 编号 | 结论 | 复核 |
|---|---|---|
| A01 | ✅ | `audionet_config` 顶层键 11 个（n_src / pretrained_vout_chan / causal / spk_embed_dim / video_bn_params / audio_bn_params / enc_dec_params / audio_params / video_params / fusion_params / mask_generation_params） |
| A06 | ✅ | 我 R1 实测 L=32000 → T'=251、F=129；`decoder` 输出等长（`istft(length=L)`） |
| A07 | ✅ | train 返回 tuple、eval 返回 Tensor（01 §8-[L]，与我 R1 一致） |
| A15 | ✅ | MaskGenerator 输出 `[1,1,112,251,129]`、decoder `[1,1,32000]`（我 R1 日志 §[6]） |
| A36/A37/A45 | ✅ | 静态 + 02 文档交叉一致（`__all__`、死类、0.5 M 取整） |
| A13/A14/A16/A25/A28~A33/A38~A41 | ❓ 未独立复核 | 属 01 自述验证（§8 脚本），我未重跑；无理由怀疑，但建议作者在 R3 把 A25/A41 的脚本输出贴进日志以便复核 |

## 3. 错误清单

### P0-1（事实错误 + 漏登死参数）：`concat_layers[2]` 从未被调用

- **文档断言**：`01-model-core.md:394`（§3.2）"`fusion[i]` 与 `concat[i]` 在每一级各调用 1 次"；
  `01-model-core.md:88`（§2.1.4）"`InjectionMultiSum`（…，跨层级 skip 融合，**被用 8 次**）"。
- **源码事实**（`look2hear/models/Falcon.py:826-896`）：
  - `:868` `expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1])` → 用 **concat[3]**；
  - `:879` `for i in range(self.upsampling_depth - 3, -1, -1)` → `i ∈ {1, 0}` → 用 **concat[1]、concat[0]**；
  - **`concat_layers[2]` 无任何调用点**（全文件 `grep concat_layers` 只有 `:868`、`:882` 两处调用）。
- **动态证据**（我的 forward-hook）：一次 Separator.forward（AudioBlock 被调 2 次）计数
  `co0=2, co1=2, co2=0, co3=2` —— `co2` 从未出现。
- **影响**：`concat_layers[2]` 有 **5,568 参数**（= `InjectionMultiSum(64,k=5,cLNhw)`，占 537,697 的 **1.04%**），
  永远不参与前向；`§4.9` 的 20 条坏味道清单**未登记**。
- **可能的代码 bug**：注释（`Falcon.py:864-867`）写"[Step 1: Initial fusion] Fuse the two deepest levels (Level N-1 and Level N-2)"，
  但初始融合用的是 `concat[-1]`+level N-2 直接相加，跳过了 `concat[2]`；若设计意图是"level 2 也应过 concat"，
  则 `concat_layers[2]` 是**漏调用**而非死参数。
- **修正建议**：① `§3.2` 改为"`fusion[i]` 4 级各 1 次；`concat` 只用 `[3],[1],[0]`，`concat_layers[2]` 从不调用"；
  ② `§2.1.4` 改为"实例化 8 次、前向调用 7 次"；③ `§4.9` 新增一条登记 5,568 闲置参数；④ 建议作者在 R3 向 Lead 提出
  "是否应在 forward 中使用 `concat_layers[2]`"的代码问题（本轮禁止改源码）。

### P1-1（误导）：A21 的"InjectionMultiSum 脉冲响应 ±2 帧"缺少"去掉 cLNhw"口径

- **文档**：`01-model-core.md:695`（A21）"`InjectionMultiSum(causal=False)` 脉冲响应 ±2 帧；改 `causal=True` 后前缀 Δ=0"；
  相关表述见 `:95-96`（§2.1.4）。
- **我的实测**（4 种口径，输入 `[1,64,251,129]`，脉冲 @t=100）：
  | 口径 | 受影响帧区间 |
  |---|---|
  | as-built（norm=cLNhw，causal=False） | **(0, 250) 全局** |
  | norm 换 Identity 后 | **(98,102)**（= 01 的结果） |
  | causal=True + norm 换 Identity | (100,104) |
  | causal=True + 保留 cLNhw | **(0, 250) 全局** |
  前缀测试：norm=Identity+causal=False → 左段 Δ=2.434；norm=Identity+causal=True → **0.0**。
- **原因**：`InjectionMultiSum` 的 3 个 `ConvNormAct` 都带 `norm_type=cLNhw`（`Falcon.py:283-313`），
  cLNhw 的 `GlobalLayerNorm` 全局统计会把单脉冲摊到所有帧。01 的 `§4.6-②` 在讲 57 帧时写了"把 cLNhw 全部替换为 Identity"，
  但 A21 断言行**没有带上这个前提**，独立读者无法复现。
- **修正建议**：A21 拆成两条：`A21a`"InjectionMultiSum 卷积核对称 padding ⇒ 去掉 cLNhw 后脉冲响应 ±2 帧（[98,102]）"；
  `A21b`"as-built（含 cLNhw）脉冲响应全局 (0,250)——因为 cLNhw 本身非因果"。§2.1.4 的因果性行也照此加限定。

### P1-2（重要遗漏）：A44 的"Studio 副本逻辑等价"不完整，漏 MPS CPU FFT bridge

- **文档**：`01-model-core.md:718`（A44）"Studio 副本与训练副本仅注释/`DropPath` 来源不同，逻辑等价"。
- **事实**：`look2hear/models/Falcon.py` ↔ `studio/falcon_model/models/Falcon.py`：`diff` 121 行（65 增 / 56 删），
  其中**非注释、非空白的实质差异**：
  - `from timm.models.layers import DropPath` → `from ..drop_path import DropPath`（等价，PROVENANCE:15）；
  - **MPS CPU FFT bridge**：`STFTEncoder.forward` 新增 `compute_device/stft_x=x.cpu() if mps/window.to(stft_x.device)/spec.to(compute_device)`
    （studio `:1110-1123`）；`STFTDecoder.forward` 新增 `output_device/x.cpu()/window.to(spec.device)/output.to(output_device)`
    （studio `:1183-1198`）。这是**功能性设备搬运**，PROVENANCE.md:20-21 与 02 文档 `§4.6` 都明确记录了它。
- **影响**：A44 被 06/02 两域引用时会传播错误印象（"Studio 只改了 import"）；且它低估了 Studio 侧的 device-placement 风险
  （MPS↔CPU 拷贝是真实的工程差异，PROVENANCE:33 只承诺 CPU 上 bit-identical）。
- **修正建议**：A44 改为三项并列：注释/日志中文化 + DropPath 本地化 + **MPS CPU FFT bridge（数值等价、设备不同）**；
  并引用 `studio/falcon_model/models/Falcon.py:1110-1123,1183-1198`；`§2.5`/`§4.6` 可加一句"Studio 版本在 MPS 上把 STFT/iSTFT 搬到 CPU"。

### P1-3（跨文档口径，非事实错误）：A19/A24 的数值必须带扰动协议

- **文档**：`01-model-core.md:693`（A19）"pre_norm 左段 Δ=1.4e-6"；02 文档 R1 写"audio_bottleneck 早期 4.7e-05"。
- **我的统一口径实验**（同一模型、同一权重，测"左段 T<125"）：

  | 扰动协议（改 1.0–2.0 s） | 仅 `pre_norm` | 整个 `audio_bottleneck`（pre_norm+1×1 conv） |
  |---|---|---|
  | 替换为同分布 randn | **9.5e-07 ~ 1.4e-06** | ~6.0e-07 ~ 1.4e-06 |
  | 后半段放大 10× | **1.6e-04** | **7.1e-05** |
  | 单采样尖峰 +5 | 1.4e-06 | 7.2e-07 |

  ⇒ 01 的 1.4e-6 与 02 的 4.7e-05 **都对**：前者=randn 替换协议下的 pre_norm/瓶颈；后者=10× 放大协议、经 encoder+bottleneck 全链。
  非零结论一致（cLNhw 非因果），但数字不可跨协议比较。
- **修正建议**：A19/A24 各加一列"扰动协议"（替换 randn / 放大 k× / 单帧尖峰），并注明测点（pre_norm vs bottleneck vs waveform）；
  02 文档 R3 修订时同步把 4.7e-05 标注为"10× 协议"。**并统一表述**（见 §5-1）。

### P2 级（细节）

- **P2-1**：`§4.6-②`（`:488-489`）"真正开启因果的只有：…`MultiHeadSelfAttention2D` 内部 Q/K/V（k=1，无时间跨度)"——
  这 4 个 `ConvActNorm` 实际是**显式 `causal=False`**（`Falcon.py:455,467,479,491`），只是 k=1 无时间跨度；MHSA2D 的因果性来自
  `tril` mask（`:520-524`）。建议改为"k=1 卷积无时间跨度（causal 标志虽为 False 但无害）；因果性由 `:520-524` 的 tril mask 提供"。
- **P2-2**：A24 的复现**必须固定 `mouth_embedding`**。我先错误地给两次 forward 各采一个随机 mouth，得到全段 ~1e-1 的"假泄漏"
  （`1.39e-01 / 1.52e-01 / …`）；固定 mouth 后才得到 01 表里的单调梯度。建议在 A24 与 §8 注明"两次前向共用同一视频特征"。
- **P2-3**：A20 的"57 帧"依赖扰动对齐：`split=125` 复现 57，`split=150` 得 58（nearest 插值的量化效应）。
  建议注明"在 7,296 样本分界（1.0 s）处测得 57 帧；±1 帧为插值对齐效应"。
- **P2-4**：`§2.5`（`:231-233`）"等价于严格因果 + 3 帧（24 ms）内部 lookback"正确，但可补一句：
  "该 lookback 恰好吸收 iSTFT 合成的 ±win/2 扩散，因此端到端 lookahead 仍 ≤ 128 采样"——否则读者会疑惑
  `§4.6-①`（encoder 8 ms）与解码器 lookback 的关系。
- **P2-5**：A22/A23 的 "Δ=4.68" 是随机输入相关量（我复现 4.971），建议只标"非零（同量级）"或固定 seed 后给值。
- **P2-6**：A10 的证据行（`:684`）只覆盖 `process_*`，不含 `concat`；修 P0-1 时一并补齐，避免"证据行与断言范围不符"。

## 4. 遗漏的重要行为（代码里有、01 没写）

1. **`concat_layers[2]` 闲置参数 5,568**（P0-1）——最实质的遗漏。
2. **`import look2hear.models` 的隐式依赖**：`Falcon.py:7 from ..layers import normalizations, activations` 会先执行
   `look2hear/layers/__init__.py`，从而加载 `layers/stft.py`（`import librosa`）、`layers/stft_tfgn.py`
   （`torch_complex`/`typeguard`/`espnet`）。因此 `§5.3` 的"它依赖谁"应补充：**librosa / torch_complex / typeguard 也是
   `import look2hear.models` 的硬依赖**（`[验证]`：`sys.modules` 出现这些模块；详见 02 文档 §2.1、§4.7）。
   这对部署/裁剪（Studio 已剥离 `layers/stft*.py`）是真实约束。
3. **`look2hear/layers/cnnlayers.py` 的同名 `ConvNormAct` 与 `Falcon.py:9` 语义不同**（1D vs 2D、无 `is2d/causal`）。
   01 在 §2.1.1 讲 `ConvNormAct` 时未声明"这是 Falcon.py 局部定义、与 layers 库同名类无关"，跨模块读者易混。
4. **`states_dict` 里 `audio_model.*` 前缀与 Studio `strict=True` 的耦合**（01 §5.4 已覆盖）——无遗漏，此处仅确认。
5. **n_src>1 失败的具体位置**：01 A16 只给 `Falcon.py:1168`，可补 "`MaskGenerator` 输出 `[B,n_src,112,T,F]` 而
   `decoder.in_chan=112*n_src`，`view(B*n_src, 224, …)` 与 112 通道不符"（02 文档 §5.1 已写）。

## 5. 跨模块一致性问题

1. **因果性表述（01 vs 02，P1 级建议统一）**：两文结论一致（"结构因果、非严格逐采样因果；三处破点：center=True 8 ms、
   对称卷积 57 帧、cLNhw 全局"），但**用词不统一**：01 用"结构层面因果/严格因果被三处打破"，02 用"窗口级近似因果"。
   建议统一为：**"窗口级因果（window-causal）：在 2 s 窗口内允许 ±8 ms（STFT 帧内）与 ±456 ms（对称卷积）及全局归一化的
   未来依赖；跨窗口无状态，因此不是逐采样因果流式模型。"** 两文都在 R3 采用同一句。
2. **A19 数字口径**：见 P1-3；02 文档 R3 应把 4.7e-05 标注协议，01 亦然。
3. **A44 与 02 文档 §4.6、PROVENANCE.md:20-21 的冲突**：见 P1-2；三处应一致写成"注释 + DropPath + MPS bridge"。
4. **`layers/` 死代码范围**：01 的 Q12 说两份 `stft.py` "没有被 Falcon.py 使用"——正确（Falcon 用 `torch.stft`）；
   补充：它们**在 import 时仍被加载**，且 `utils/stft.py` 通过 `layers/stft_tfgn.py → utils/__init__` 的链路被连带导入
   （02 文档 A03~A06 有完整证据链）。01 无需展开，但依赖章节应引用 02 的结论。
5. **`Separator` pop 副作用**：01 §3.1/§4.9-⑤ 与 02 §2.4 完全一致（我两轮都验证 537,697→49,081），无冲突。

## 6. 给作者的问题清单（R3 请回答/补充验证）

| # | 问题 | 期望动作 |
|---|---|---|
| Q1 | `concat_layers[2]` 是有意闲置还是 `AudioBlock.forward` 漏调用？设计意图（注释说 "Fuse the two deepest levels"）支持后者。 | 在 R3 写明结论；若属代码 bug，报 Lead 记录（不改源码），并在 §4.9 增补 5,568 闲置参数条目 |
| Q2 | A21 的脉冲测量是否在"去掉 cLNhw"后做的？as-built 实测是全局响应 (0,250)，与断言不符。 | 拆分 A21a/A21b 并给两种口径的数值 |
| Q3 | A44 是否看到 studio 的 MPS CPU FFT bridge（4+4 行）？ | 修正 A44 与 §2.5/§5.3 的等价性表述 |
| Q4 | A19/A24 的扰动协议、随机种子、mouth 是否固定？我的复现表明这些条件会改变 2~4 个数量级。 | 断言表加"扰动协议"列；§8 写清"共用 mouth、替换 randn、seed=X" |
| Q5 | A20 的 57 帧在 split=150 时为 58，是否接受"±1 帧量化"的写法？ | 注明分界与对齐 |
| Q6 | A22/A23 的 4.68 是否固定 seed 下的值？ | 标"随机输入，量级 1e0 非零"或给 seed |
| Q7 | `§4.6-②` 把 MHSA2D 内部 Q/K/V 列为"开启因果"是否笔误（它们是 `causal=False`）？ | 修正措辞，因果来源指向 tril mask |
| Q8 | 是否同意在 R3 采用 §5-1 的统一"窗口级因果"表述，并与 02 文档同步？ | 回复同意/提出替代措辞 |
