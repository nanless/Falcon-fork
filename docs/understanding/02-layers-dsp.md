# 02 · 层与 DSP 实现（look2hear/layers、utils 数值工具、STFT 重复实现核对）

> 作者：`layers-dsp`；**R3 修订版**——R1 版 757 行/sha256 `ae01042f…`（被两份评审锚定），**R3 版共 895 行**，
> R3 版 sha256 记录于 `docs/verification/r3-layers-dsp-anchor.md`（文件自哈希无法内嵌于自身）。R3 基于两份评审修订（见 §9.1）。
> 证据基准 commit `7576854`。
> 置信度标注：`[代码]` 直接读源码确认；`[推断]` 由源码/配置合理推断；`[验证]` 实际运行脚本确认。
> 验证脚本：R1 `docs/verification/scripts/verify_layers_dsp.py`（日志 `runs/layers-dsp-verify.log`）；
> R3 `docs/verification/scripts/verify_r3_layers_revision.py`（日志 `runs/r3-layers-revision.log`）、
> `docs/verification/scripts/verify_r3_studio_weights.py`（日志 `runs/r3-studio-weights.log`，使用 Lead 提供的真实权重
> `/root/.cache/falcon_weights/falcon.pt`，只读）。
> 环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0），stub 注入见 `docs/understanding/VERIFY-ENV.md`。
> **R3 主要修订**：① P0 —— `STFTEncoder` 不再表述为"严格因果"（改为 conv 子层严格因果 + `torch.stft(center=True)`
> 的 ≤128 采样/8 ms 帧级 lookahead）；② §4.1 采样级 lookahead 由"≤256 采样/16 ms"改为"帧级 8 ms；encoder→decoder
> 串联采样级过去泄漏 ≤127 采样（实测依赖对齐，p=20000 时 31）"；③ A19/A49/A57 补协议与行号；④ 统一"窗口级因果
> （window-causal）"表述（与 `01-model-core.md` 一致）。

## 1. 子系统职责与总体位置

**一句话结论：`look2hear/layers/` 是一套"通用层库"（Conv-TasNet/Asteroid/ESPnet/Conformer 谱系的移植），
但 Falcon 模型本身只从它里面用了 `normalizations` 与 `activations` 两个子模块；真正定义 Falcon 网络
所有层的是 `look2hear/models/Falcon.py`（自带 `ConvNormAct`/`STFTEncoder`/`DualPathRNN`/`MHSA2D`/…）。
`layers/` 其余 5 个文件（`stft.py`、`stft_tfgn.py`、`cnnlayers.py`、`rnnlayers.py`、`enc_dec.py`）在
全仓范围内没有任何调用方，只是被 `layers/__init__.py` 全量 import，因此在"导入模型"时被连坐加载。**

证据：

- `look2hear/models/Falcon.py:7`：`from ..layers import normalizations, activations`——Falcon 对 `layers` 的**唯一**依赖。
- `look2hear/layers/__init__.py:1-20`：全量 import `cnnlayers`（**13** 个名字）、`rnnlayers`（**5** 个）、`enc_dec`（2 个）、
  `stft`（2 个函数）、`stft_tfgn`（1 个类）。（R3 修正：R1 版误写 15/6；AST 计数见 §8-[V6]。）
- 全仓 grep：`TAC|DPRNN|DPRNNBlock|DPRNNLinear|LSTMBlockTF|TransformerBlockTF|FRCNNBlock|Bottomup|Video1DConv|Concat|Conv1DBlock|ConvNorm|NormAct`
  在 `look2hear/layers/` 之外**零命中**（`look2hear/layers/*.py` 内部互引除外）`[代码]`。
- `make_enc_dec`/`FreeFB`/`Encoder`/`Decoder`（`enc_dec.py`）同样零外部调用；Falcon 的编码器是
  `models/Falcon.py:1054 STFTEncoder`（`[验证]`：`type(model.encoder).__module__ == 'look2hear.models.Falcon'`）。

因此本子系统文档的重点分配为：

| 重要性 | 内容 | 位置 |
|---|---|---|
| **P0（Falcon 真实路径）** | `normalizations`（尤其 `cLNhw`）、`activations` | `layers/normalizations.py`、`layers/activations.py` |
| **P0（Falcon 真实路径，定义在 model 层）** | `ConvNormAct`/`ConvActNorm`/`STFTEncoder`/`STFTDecoder`/`DualPathRNN`/`MHSA2D`/`InjectionMultiSum` | `models/Falcon.py`（交叉引用，归属 model-core） |
| **P1（重复实现核对）** | `layers/stft.py` ↔ `utils/stft.py` 同源复制 | 两份 785 行文件 |
| **P2（死代码，但影响导入/依赖）** | `cnnlayers`/`rnnlayers`/`enc_dec`/`stft_tfgn`/`utils.stft`/`utils.complex_utils`/`utils.attn` | 见 §2 |

**与 README 术语的映射（重要纠偏）**：README:56 与 README:61-62 提到的 **TFRB**（Time-Frequency Refinement Block）
与 **GCI** 在代码中**没有同名类**，它们对应 `models/Falcon.py` 的内部模块：

- **TFRB** = `DualPathRNN(dim=4, bidirectional=True)`（沿频率轴双向 LSTM）+ `DualPathRNN(dim=3, bidirectional=False)`
  （沿时间轴单向 LSTM）+ `MultiHeadSelfAttention2D(causal=True)`（时间轴 masked self-attention）三件套，
  在 `AudioBlock` 中按尺度各实例化一份（`models/Falcon.py:777-790`）。`[推断]`（命名映射由 README 图注 + 结构对照得出）
- **GCI** = `InjectionMultiSum`（`models/Falcon.py:267-329`）的"局部嵌入 × 全局门 + 全局嵌入"注入式融合 `[推断]`。

## 2. 文件清单与逐文件解析

### 2.1 `look2hear/layers/__init__.py`（50 行）

导出面 = 5 个文件的所有公共类（`__all__` 见 `:22-50`）。注意两点：

1. `:19 from .stft import forward_stft, inverse_stft` —— 这是 `layers/stft.py` **唯一的 import 点**；
   该模块顶层 `import librosa.filters`（`layers/stft.py:7`）与 `from distutils.version import LooseVersion`（`:10`），
   因此**只要 import 了 `look2hear.layers` 包，librosa 与 distutils 就是硬依赖**（`[验证]`：`import look2hear.models`
   后 `sys.modules` 中出现 `librosa`、`librosa.filters`）。
2. `:20 from .stft_tfgn import Stft` 会连带 import `torch_complex`、`typeguard`、`espnet` 的 `nets_utils`
   （`layers/stft_tfgn.py:3-12`），尽管 Falcon 从不用 `Stft`。
3. `__all__` 里 `"Bottomup"` 重复出现两次（`:37-38`），无害但说明是复制粘贴维护。

### 2.2 `look2hear/layers/stft.py`（785 行）= `look2hear/utils/stft.py`（785 行）

这是同一份代码的两处拷贝（diff 结论见 §2.12）。实现是"用 `conv1d` 做 DFT"的 librosa 兼容 STFT/iSTFT 库，
谱系来自 SpeechBrain/PyTorch-STFT 一类实现（`mode ∈ {librosa, kaldi, torch}`）。

**数学定义**（`init_kernel`，`:59-111`）：

- 帧长 `frame_len = W`，hop `H`，`round_pow_of_two=True` 时 `FFT 点数 B = 2^⌈log2 W⌉`；若 `B ≠ W` 且 `mode="librosa"`，
  窗被居中零填充到 B（`:87-89`）。
- DFT 核矩阵：`K = fft(eye(B) / S, dim=-1)`（`S=1` 未归一化；`normalized=True` 时 `S=√B`），
  实部/虚部堆叠 → 转置 → reshape 成 `[2B, 1, B]` 的 conv1d 权重（`:96-110`）。`inverse=True` 时再除 `B`（`:104-106`），
  即 iDFT 核 `(1/B)·Kᴴ`。
- 前向（`:221-286`）：`wav [N,(C),S]` → `view(-1,1,S)`；`center=True` 时两端 **reflect** 填充 `B/2`（`:253-256`）；
  `kernel = K * window`（窗乘进核里，`:258`）；`conv1d(stride=H)` → `[N·C, 2B, T]`；
  `chunk(2)` 得 real/imag；`onesided` 时取前 `B/2+1` 个 bin；返回 `[N,(C),B/2+1,T,2]`（`return_polar=True` 则 `[mag, phase]`，
  `mag = sqrt(real²+imag²+eps)`，`:281-284`）。
- 逆变换（`:289-354`）：`onesided` 时用共轭对称补齐 `real,B-1..1` / `-imag`（`:327-330`）；`conv_transpose1d(stride=H)`
  做 overlap-add；再除以窗平方的 OLA 能量 `denorm`（`:336-352`），`center=True` 时裁掉两端 `B/2`。
- 帧数公式（`STFTBase.num_frames`，`:673-682`）：先 `center=True` 时把 `L` 加 `win_length`，再
  `T = trunc((L' − win_length)/H) + 1`。由于 `win_length = K.shape[2] = B`（FFT 点数），故 `center=True` 时
  `T = 1 + ⌊L/H⌋`、`center=False` 时 `T = 1 + ⌊(L−B)/H⌋`（`[验证]`：L=32000,H=128 → center=True 251、false 249）。
  注意 `:677` 断言 `L > win_length`。
- **`center` 默认值是 `False`**（`forward_stft` `:484`、`STFTBase` `:636`），与 `torch.stft` 的默认 `center=True` **相反**，
  这是最容易踩的坑之一。

**关键参数与根因校验（`[验证]`，日志 §[1][2]）**：

| 检查 | 结果 |
|---|---|
| `forward_stft(mode="librosa", window="hann", center=True)` vs `torch.stft(center=True)` | shape 均 `[2,129,251,2]`，`max|Δ| = 2.289e-05`（float32 数值误差级） |
| 同上 `center=False` | 均 `[2,129,249,2]`，`max|Δ| = 2.289e-05` |
| `forward_stft(mode="torch")` vs `torch.stft` | `max|Δ| = 0.000e+00`（就是薄包装，`:357-413`） |
| `inverse_stft(mode="librosa")` vs `torch.istft` | `max|Δ| = 2.146e-06` |
| round-trip（librosa 实现）最大误差 | `2.384e-06`（32k 随机信号） |
| `round_pow_of_two` 行为 | `frame_len=200→B=256→bins=129`；`frame_len=400→B=512→bins=257` |

结论：**`layers/stft.py` 的卷积 DFT 与 `torch.stft` 数值等价（float32 精度内），`mode="torch"` 时逐位等价**；
但它是"教科书式"实现（显式 DFT 矩阵 + conv1d），不用 FFT，且 `import librosa`，在 Falcon 中**从未被调用**。

**其它函数**（均无 Falcon 调用方）：

- `export_jit`（`:21-26`）：`[m for m in transform if m.exportable()]` —— 对 `nn.Module` 迭代/调用 `exportable()`
  都会 `TypeError`，是死代码 + bug `[代码]`。
- `init_window`（`:29-56`）：`wnd ∈ {bartlett,hann,hamm,blackman,rect,sqrthann}`；默认 `sqrthann = hann**0.5`，
  且 `periodic=True`（除 rect）。`sqrthann` + `N=2^k` + `H=N/2` 是"完美重构"组合。
- `mel_filter`（`:114-158`）：`librosa.filters.mel(htk=True)`，返回 `[num_mels, N/2+1]`。
- `speed_perturb_filter`（`:161-185`）：多相重采样滤波器，`⌈num_zeros/zeros_per_block⌉` 阶 sinc×cos 窗。
- `splice_feature`（`:188-218`）：沿倒数第 2 维（时间）做 ±lctx/rctx 的边界 clamp 拼接。
- `STFTBase`（`:610-696`）：把 `K`、`w` 注册成 `requires_grad=False` 的 `nn.Parameter`（`:650-651`）；
  `num_bins = K.shape[0]//4 + 1`（`:652`）、`win_length = K.shape[2] = B`（`:654`）。
- `STFT`（`:699-741`）/`iSTFT`（`:744-785`）：`nn.Module` 包装，输入 `[N,(C),S]` / `[N,(C),F,T,2]`。

### 2.3 `look2hear/layers/stft_tfgn.py`（244 行）—— ESPnet 的 `Stft`

- `class Stft(nn.Module, InversibleInterface)`（`:23`）：`center=True` 默认（`:30`），`forward` 返回 `(output, olens)`，
  `output` 形状 `[B, T, F, 2]`（注意是 **帧在前**，与 `layers/stft.py` 的 `[B,F,T,2]` 相反）。
- 实现：torch ≥1.10 或有 CUDA/MKL 时直接 `torch.stft`（`:99-111`），否则退化为逐样本 librosa（`:112-156`）。
- `inverse`（`:188-244`）：`torch.istft` 包装，输出 `(wavs, ilens)`。
- 依赖：`torch_complex`、`typeguard.check_argument_types`（typeguard≥3 已移除该 API，必须 stub）、`espnet` 的 `make_pad_mask`。
- **全仓无调用方**（只被 `layers/__init__.py:20` 导出）。

### 2.4 `look2hear/utils/complex_utils.py`（183 行）

ESPnet beamformer 的复数张量兼容层：在 `torch_complex.tensor.ComplexTensor`（旧式 `[..., 2]` 表示）与
torch 原生 complex（1.9+）之间做统一分发。函数：`new_complex_like`(`:14`)、`is_torch_complex_tensor`(`:28`)、
`is_complex`(`:34`)、`to_double/to_float`(`:38/45`)、`cat/einsum/matmul/trace/reverse/solve/stack`。
`EPS = finfo(double).eps`（`:9`）。**全仓无调用方**（除 `utils/__init__.py:13` 再导出与 `stft_tfgn.py:10` 的 `is_complex`）。
`torch_complex` 是 `import look2hear.utils` 的硬依赖。

### 2.5 `look2hear/utils/attn.py`（28 行）

`CrossAttention(dim, num_heads=8)`：标准 cross-attention，`x1 → Q`、`x2 → K,V`，返回 `(out, attn)`；
`scale = head_dim^-0.5`（`:8`）。**全仓无调用方**（纯死代码）。

### 2.6 `look2hear/utils/inversible_interface.py`（12 行）

`InversibleInterface(ABC)`，抽象方法 `inverse(input, input_lengths)`（`:7-13`）。被 `stft_tfgn.Stft` 继承，
属于 ESPnet 接口约定的一部分。

### 2.7 `look2hear/layers/normalizations.py`（245 行）—— **Falcon 实际使用的归一化**

**这是本子系统最关键的"活代码"文件**。所有归一化通过 `normalizations.get(identifier)`（`:231-245`）解析：
若 `identifier is None` → 返回 `nn.Identity` **类**（调用方会再 `(...)(chan)` 实例化）；若是字符串且 `hasattr(nn, s)` →
返回 **torch 原生类**；否则查本模块 `globals()`（即自定义类/别名）。因此 `get("BatchNorm1d")` 得到的是
**`torch.nn.BatchNorm1d`，不是本文件 `:64` 的包装类**（`[验证]`）。

别名表（`:224-229`）：

| 别名 | 实际类 | 语义 |
|---|---|---|
| `gLN` | `GlobalLayerNorm`(`:134`) | 内部是 `nn.GroupNorm(num_groups=1, num_channels=C)` |
| `cLN` | `CumulateLN`(`:91`) | 3D 输入 `(B,C,T)` 的累积 LN |
| `LN` | `CumulativeLayerNorm`(`:74`) | **名不副实**：只是对 channel 维做 `nn.LayerNorm`，无累积 |
| `LN4d` | `LayerNormalization4D`(`:146`) | 4D 输入、按 `(C,F)` 参数化 |
| `cLNhw` | `LayerNormalizationHW`(`:204`) | `GroupNorm(1,C)` + `CumulateLN4D` 复合 |
| `bN` | `BatchNorm`(`:64`) | `_BatchNorm` 包装（仅重写 `_check_input_dim`）；`get("BatchNorm1d")` 不会用它 |

**精确语义与广播维度**（全部 `[验证]`，日志 §[4]；输入示例 `[B,C,T,F]=[1,4,6,3]`）：

1. `GlobalLayerNorm(C)`：`nn.GroupNorm(1, C, eps)`（`:140`）。对每个样本在 **所有非 batch 维 (C,T,F)** 上联合求
   `μ,σ`（PyTorch GroupNorm 把 C 之后所有维度当"spatial"），再按 channel 做可学习仿射（`γ,β` 各 C 个）。
   `[验证]` 与手算 `x.mean(dim=(1,2,3))` 一致（`max|Δ|=2.4e-07`）。**⇒ 跨整个时间轴统计 → 非因果**。
   代码注释称"Normalizes over C and W dimensions"（`:216`）不准确。
2. `CumulateLN4D(C)`（`:165-202`）：输入 `(B,C,H,W)`，沿 **H（时间）累积**：
   `μ_{t,f} = Σ_{c}Σ_{t'≤t} x[c,t',f] / (C·(t+1))`，`σ²` 同理；`gain/bias` 形状 `[1,C,1,1]`（`:171-172`）。
   **统计跨 (C, t'≤t)，但对每个频点 f 独立**（`cum_sum` 保留 W 维，`:182-190`）。逐步累积 ⇒ **时间因果**。
3. `LayerNormalizationHW(C) = cLNhw`（`:204-221`）：先 `GroupNorm(1,C)` 再 `CumulateLN4D(C)`。
   `[验证]` 与"GN 后接 CumulateLN4D"逐元素一致。**⇒ 第一段是全窗口非因果统计，第二段才是因果累积。**
4. `LayerNormalization4D(input_dimension=(C,F))`（`:146-163`）：`param_size=[1,C,1,F]`；
   `self.dim = (1,3) if F>1 else (1,)`（`:152`）。两种用法：
   - `(C,1)` → `dim=(1,)`，即**逐 (b,t,f) 在 channel 上**归一化，`γ,β` 形状 `[1,C,1,1]`（Falcon `DualPathRNN` 用这种，
     `models/Falcon.py:370`）；
   - `(C,F)`（F>1）→ `dim=(1,3)`，即**逐帧在 (C,F) 上**归一化，`γ,β` 形状 `[1,C,1,F]`（Falcon `MHSA2D` 的 `ConvActNorm` 用这种，
     `models/Falcon.py:186-188`）。**这要求传入的 F 必须等于运行时频率维，否则广播报错**（见 §4.3）。
   - 注：`eps=EPS=1e-5`（`:9`），而 `CumulateLN4D` 默认 `eps=1e-8`。
5. `cLN = CumulateLN(nn.Module)`（`:91-132`）：输入 `(B,C,T)`，`μ_t,σ_t` 为 **t'≤t 的所有帧 × 所有通道** 的统计
   （与 4D 版区别是没有 f 维），`gain/bias` 形状 `[1,C,1]`。**时间因果**。`[验证]` 与手算一致。
   注意文件里 `class CumulateLN` 定义了**两次**（`:51` 继承 `MLayerNorm`、`:91` 继承 `nn.Module`），后者覆盖前者，
   `:51` 那个成为不可达死代码（且它 `cumsum(..., dim=1)` 写错维度，是 bug）。
6. `MLayerNorm`(`:23`)/`GlobalLN`(`:38`)/`ChannelLN`(`:44`)：早期实现，`gamma/beta` 初始化为 **1 和 1**（`:27-28`，beta 应为 0，
   疑似 bug），全仓无人使用。
7. `BatchNorm(_BatchNorm)`（`:64-71`）：仅放宽 `_check_input_dim` 支持 2~4D；Falcon 配置里的 `norm_type: BatchNorm1d`
   实际落到 `nn.BatchNorm1d`（见上）。**注意**：`BatchNorm1d` 用在 `[B,C,T]` 视频特征上，其统计跨 batch 与时间，
   且 **eval 时用 running stats、train 时用 batch stats**，与 streaming 的确定性推理有微妙关系。
8. `CumulativeLayerNorm`（`:74-88`）：`nn.LayerNorm(C)` 作用在转置后的最后一维，**不是**累积统计；
   命名与 `cLN` 极易混淆。

### 2.8 `look2hear/layers/activations.py`（67 行）

`get(identifier)`（`:48-62`）与 `normalizations.get` 同构：`None → nn.Identity`（类）、`hasattr(nn, s) → torch 原生类`、
否则查 `globals()`。本地别名：`linear/relu/prelu/leaky_relu/sigmoid/softmax/tanh/gelu`（`:5-34`），
其中 `softmax(dim=None)` 需要传 dim。`register_activation`（`:37-45`）允许注册自定义激活。
Falcon 配置使用 **大写** 名（`PReLU`/`ReLU`/`Tanh`/`Sigmoid`，`configs/Falcon.yml` 多处），走 `hasattr(nn,...)` 分支；
小写名（`prelu`）走 `globals()` 分支，返回的是**函数**（调用后才是 `nn.PReLU`）——两者最终实例相同（`[验证]` 日志 §[5]）。

### 2.9 `look2hear/layers/enc_dec.py`（461 行）—— Asteroid 滤波器组编解码器

- `make_enc_dec(fb_name, n_filters, kernel_size, stride=None, ...)`（`:8-71`）：成对创建 Encoder/Decoder，
  支持 `who_is_pinv ∈ {enc, decoder}` 做伪逆绑定；不带 pinv 时两个 filterbank 各自独立初始化（`:61-70`）。
- `Filterbank`（`:114-175`）：`stride` 缺省 `kernel_size//2`（`:132`）；预/后处理钩子 `pre_analysis/post_analysis/
  pre_synthesis/post_synthesis`（`:142-156`）。
- `Encoder`（`:223-281`）：`conv1d`，`(batch,1,time) → (batch,freq,conv_time)`；`multishape_conv1d`（`:284-326`）
  支持 1D~4D 输入的"多通道扁平化卷积"。
- `Decoder`（`:341-392`）：`conv_transpose1d`（等价 overlap-add），支持 `length` 截断（`:389-391`）。
- `FreeFB`（`:434-458`）：无约束可学习滤波器 `nn.Parameter[ones(n_filters,1,kernel_size)]`，xavier_normal 初始化（`:453-455`）；
  `free = FreeFB`（`:461`）。
- 因果性：**完全没有 causal 概念**——`padding` 是对称 `conv1d/conv_transpose1d` padding，用 `padding=kernel_size//2`
  即为非因果"same"卷积。仅供离线分离模型使用。
- **全仓无调用方**，但 `make_enc_dec`/`FreeFB` 被 `layers/__init__.py:17` 导出。

### 2.10 `look2hear/layers/cnnlayers.py`（884 行）—— 视觉/音频卷积块库

全部依赖 `normalizations.get`/`activations.get`（`:4`），**全仓无外部调用方**。逐类要点：

| 类（行号） | 作用 / 形状 | 因果性 |
|---|---|---|
| `_Chop1d`(`:7-15`) | `x[..., :-chop]`，用于把 "same" 卷积裁成因果 | 因果辅助 |
| `Conv1DBlock`(`:18-63`) | Conv-TasNet 式 1×1→PReLU→norm→depthwise→PReLU→norm，+res/skip；`(B,F,T)` | `causal=True` 时 `depth_conv1d` 后接 `_Chop1d(padding)`（`:42-43`）⇒ 严格因果 |
| `ConvNormAct`(`:66-101`) | **1D** 版（`nn.Conv1d`，`padding=0` 默认，`act_type="prelu"` 小写默认）；与 `models/Falcon.py:9` 同名类语义不同（见 §4.8） | 无 causal 参数 |
| `ConvNorm`(`:104-131`) | 卷积+norm，无激活 | 无 |
| `NormAct`(`:134-152`) | norm+激活 | 无 |
| `Video1DConv`(`:155-220`) | 视频 depthwise conv，`padding=(dilation*(k-1))//2`（`:189`）⇒ 非因果；BN→ReLU→dconv→1×1 分支 | 非因果 |
| `Concat`(`:223-239`) | 视频特征 `interpolate` 到音频长度后 concat + 1×1 conv | 非因果（nearest 插值） |
| `FRCNNBlock`(`:242-389`) | 4 尺度空间金字塔：`pixel-unshuffle` 式 stride-2 深度可分离卷积（`:278-290`）+ 跨尺度 fuse（`:369-381`）+ 上采样 concat | 非因果 |
| `Bottomup`(`:392-451`) | 只做下采样金字塔，返回 `(residual, 最深特征, 各层列表)` | 非因果 |
| `BottomupTCN`(`:454-495`) | 同上但用 `Video1DConv` 做层间变换 | 非因果 |
| `Bottomup_Concat_Topdown`(`:498-595`) | 金字塔 + top-down 注入 + 逐层 fusion（Vim 风格） | 非因果 |
| `Bottomup_Concat_Topdown_TCN`(`:598-680`) | 同上、无 stride 融合（fusion 层全是 `None`，`:618`） | 非因果 |
| `FRCNNBlockTCN`(`:683-795`) | `FRCNNBlock` 的 TCN 变体 | 非因果 |
| `TAC`(`:798-884`) | Transform-Average-Concatenate 跨麦克风置换不变块，5D 输入 `(B,M,F,K,L)` | 无时间维概念 |

### 2.11 `look2hear/layers/rnnlayers.py`（908 行）—— RNN/Conformer/DPRNN 库

**重点澄清：Falcon 的 `DualPathRNN` 不在这里**。`rnnlayers.py` 里叫 `DPRNN*` 的是 Asteroid/SepFormer 式
dual-path RNN（chunk 内 + chunk 间），与 Falcon 的 `DualPathRNN`（频率轴/时间轴）**只是名字相似，实现不同**。
`rnnlayers.py` 全部类在 Falcon 路径中**无调用方**。

| 类（行号） | 定义 / 形状 |
|---|---|
| `has_arg`(`:15-31`) | 检查 callable 是否接受某关键字参数（用于 `softmax(dim=)` 分支） |
| `SingleRNN`(`:34-73`) | `getattr(nn, rnn_type)`，`batch_first=True`，输入 `[B,T,C]`，输出 `[B,T,hid*dirs]`；`:70` 调 `flatten_parameters()` |
| `LSTMBlockTF`(`:76-105`) | `(B,F,T)` → 转置→SingleRNN→Linear→转置→norm→`+x` 残差 |
| `Linear`(`:109-123`) | 带可选 bias 的 `nn.Linear` 包装 |
| `Swish/Transpose/GLU/FeedForwardModule`(`:126-180`) | Conformer 基础件（GLU 沿 dim=-1 对半切） |
| `PositionalEncoding`(`:183-206`) | 正弦位置编码，`forward(length) → [1,len,d]` |
| `RelativeMultiHeadAttention`(`:209-307`) | Transformer-XL 相对位置注意力，含 `u_bias/v_bias`、`_relative_shift`(`:297-307`) |
| `MultiHeadedSelfAttentionModule`(`:310-359`) | pre-LN + 相对位置 MHSA；`is_casual=True` 时构造 `triu` mask（`:344-352`，拼写 typo） |
| `ResidualConnectionModule`(`:362-377`) | `module(x)*a + x*b` |
| `DepthwiseConv1d/PointwiseConv1d/ConformerConvModule`(`:380-522`) | Conformer 卷积模块（GLU + depthwise + BN + Swish） |
| `TransformerLayer/TransformerBlockTF`(`:525-583`) | Conformer/Transformer 堆叠 |
| `DPRNNBlock`(`:586-637`) | intra-chunk（沿 chunk 内 K 维，**固定 bidirectional=True**，`:598-605`）+ inter-chunk（沿 chunk 数 L 维，可配置方向） |
| `DPRNN`(`:640-771`) | `unfold` 切块（`chunk_size`、`hop_size=chunk//2`，`:720-727`）→ `n_repeats` 个 block → 掩码 → `fold` OLA → 每源 mask；输入 `[B,F,T]`，输出 `[B,n_src,out_chan,T]` |
| `DPRNNLinear`(`:774-908`) | `DPRNN` 的 Linear 变体；`:882` `self.net_out(output.transpose(1, 1))` 其中 **`transpose(1,1)` 是 no-op**，且 `nn.Linear` 用在 `[B,C,T]` 上只改最后一维 → 与 `net_gate` 的输出形状不匹配，**该 forward 无法正常工作**（纯死代码） `[代码]` |

### 2.12 必答：两份 `stft.py` 是否同源复制？—— **是，逐字节同源（仅 2 处纯空白差异）**

**diff 证据**（命令与完整输出见 `docs/verification/runs/layers-dsp-verify.log` 旁的终端记录，可直接复现）：

```bash
diff -u look2hear/layers/stft.py look2hear/utils/stft.py     # 15 行输出 = 2 个 hunk
diff look2hear/layers/stft.py look2hear/utils/stft.py | grep -c '^<'   # 1
diff look2hear/layers/stft.py look2hear/utils/stft.py | grep -c '^>'   # 1
```

两个 hunk 是：

1. `utils/stft.py` **第 1 行多一个空行**（文件以空行开头），导致其后所有行号 +1；
2. `layers/stft.py:164` 有一行**只含 4 个空格**的空白行，在 `utils/stft.py` 中被删除。

归一化证据（去空白后哈希相同）：

```bash
for f in look2hear/layers/stft.py look2hear/utils/stft.py; do tr -d '[:space:]' < $f | md5sum; done
# 3f7053135a33126c8d2a078b98fc2927  (两个文件完全一致)
md5sum look2hear/layers/stft.py look2hear/utils/stft.py
# 533c481a... (layers)  vs  796ceb06... (utils) —— 仅因上述空白差异而不同
```

**结论**：`diff` 差异 = **2 行（1 增 1 删，均为空白）**，`diff -B`（忽略空行）后只剩 1 处 `164d164    `，
`unified diff` 共 15 行。二者是**同一份代码的复制粘贴**，不是两份独立实现。

**各自被谁 import（grep 证据）**：

| 文件 | 唯一 import 点（**限产品代码 `look2hear/`、`studio/`、仓库根 `*.py`**；`docs/verification/scripts/` 的自检脚本除外） | 实际是否被调用（同范围） |
|---|---|---|
| `layers/stft.py` | `look2hear/layers/__init__.py:19`：`from .stft import forward_stft, inverse_stft` | **否**：只导出了 `forward_stft`/`inverse_stft` 两个函数名，产品代码中无任何调用 `forward_stft(`/`inverse_stft(` 的地方（仅 R1 验证脚本 `verify_layers_dsp.py` 调用）（`look2hear/losses/matrix.py` 的是**另一套**自带 STFT）；`STFT`/`iSTFT` 类更未导出 |
| `utils/stft.py` | `look2hear/utils/__init__.py:1`：`from .stft import STFT` | **否**：产品代码中无使用 `utils.STFT` 的地方；`utils/__init__.py:19` 只是把它放进 `__all__` |

**为什么"没被调用却仍会被加载"**：`models/Falcon.py:7 from ..layers import normalizations, activations` 会先执行
`look2hear/layers/__init__.py`（Python 包语义），于是 `layers/stft.py`（需要 librosa）被 import；
而 `layers/stft_tfgn.py:10-12 from ..utils.complex_utils/... ` 又触发 `look2hear/utils/__init__.py`，于是
`utils/stft.py` 也被 import。`[验证]`：`import look2hear.models` 后 `sys.modules` 同时含
`look2hear.layers.stft`、`look2hear.utils.stft`、`librosa`、`torch_complex`（日志 §[0]）。

**第三/第四份 STFT**（避免"以为只有两份"的误判）：

- `look2hear/models/Falcon.py:1054/1123` 的 `STFTEncoder`/`STFTDecoder`：直接调 `torch.stft`/`torch.istft`
  （`n_fft=win=256, hop=128, hann 周期窗, center 默认 True`）——**Falcon 真正使用的实现**。
- `look2hear/losses/matrix.py:13` 的 `STFTBase` + `:59` 的 `STFT`：自带 sqrt-hann + 完美重构缩放的**第三套**实现，
  供 `MultistageLoss` 使用（`:114`），属于 loss 子系统（loss-metric-eval 负责）。
- `layers/stft_tfgn.py:23 Stft`：ESPnet 包装（第 4 套）；其硬依赖 `torch_complex` 来自
  `utils/complex_utils.py`（`from torch_complex import functional as FC`，`complex_utils.py:6-7`），`utils/nets_utils.py`
  再引入 espnet 的 `make_pad_mask` —— 这三者构成"STFT 生态"的导入面（§4.7）。

## 3. 关键调用链与数据流

### 3.1 Falcon 真实前向链（`models/Falcon.py`，归属 model-core，此处只给 DSP 相关接口）

```
audio_mixture [B, 32000] float32
  └─ STFTEncoder.forward                     Falcon.py:1107-1121
       torch.stft(n_fft=256, hop=128, hann, center=True)  → [B,129,251] complex64
       sqrt(|X|²+eps)                                    → [B,129,251] float32
       stack([real, imag, mag], dim=1).transpose(2,3)    → [B,3,251,129]
       ConvNormAct(3→112, k=3, is2d, causal=True)        → [B,112,251,129]
  └─ audio_bottleneck = ConvNormAct(**audio_bn_params, in_chan=112)   Falcon.py:1311
       pre_norm=cLNhw(112) → ReLU → Conv2d(112→112,k=1) → Identity     → [B,112,251,129]
  └─ video_bottleneck = ConvNormAct(kernel_size=-1)                    Falcon.py:1312
       → nn.Identity（0 参数）                                          → [B,512,50] 原样
  └─ Separator.forward(audio, video)                                   Falcon.py:1025-1052
       for i in 0..fusion_repeats(=1): audio=AudioBlock(audio); video=VideoBlock(video);
                                       audio=NeuroSCNetFusion(audio,video)
       for j in 0..audio_repeats(=1):  audio=AudioBlock(audio + audio_residual)
                                                                       → [B,112,251,129]
  └─ MaskGenerator.forward(refined, mixture_emb)                      Falcon.py:1266-1275
       PReLU → ConvNormAct(112→112,k=1) → RI_split 复数掩码乘法        → [B,1,112,251,129]
  └─ STFTDecoder.forward(emb, input_shape)                            Falcon.py:1163-1192
       view → [B,112,251,129] → causal F.pad(T,2) → ConvTranspose2d(112→2,k=3,p=1)
       → crop T[:-1] → complex → istft(length=32000)                  → [B,1,32000]
```

### 3.2 `layers` 的调用链（只有归一化/激活被真正调用）

```
models/Falcon.py:87-90      ConvNormAct.__init__ → normalizations.get(x)(chan) / activations.get(x)()
models/Falcon.py:186-188    ConvActNorm.__init__ → normalizations.get(x)((C,n_freqs) 或 C)
models/Falcon.py:370        DualPathRNN.__init__ → normalizations.get(x)((C,1))
```
此外 **无任何** `layers.cnnlayers / rnnlayers / enc_dec / stft / stft_tfgn` 的运行时调用（`[代码]` + `[验证]`）。

### 3.3 形状推演示例（至少 3 个端到端）

#### 例 1：STFTEncoder 与 STFTDecoder（输入 `[B, 32000]`）

| 步骤 | 操作 | 形状 | dtype | 依据 |
|---|---|---|---|---|
| 1 | 输入 2s/16kHz 单声道 | `[B,32000]` | float32 | `configs/Falcon.yml:161-162`（16 kHz / segment 2.0 s） |
| 2 | `torch.stft(n_fft=256, hop=128, hann, center=True, onesided=True)` | `[B,129,251]` | complex64 | `Falcon.py:1110-1116`；帧数 `1+32000/128=251`，bin 数 `256/2+1=129` |
| 3 | `this_spec=sqrt(|X|²+eps)` | `[B,129,251]` | float32 | `Falcon.py:1117` |
| 4 | `stack([real,imag,mag],1).transpose(2,3)` | `[B,3,251,129]` | float32 | `Falcon.py:1118` |
| 5 | `Conv2d(3→112,k=3,padding=(2,1))` 后 `[:,:,:-2,:]` | `[B,112,251,129]` | float32 | `Falcon.py:59,100-104`；T: `(251+4-3)+1-2=251`，F: `(129+2-3)+1=129` |
| 6 | bitstream → `ConvTranspose2d(112→2,k=3,p=1)`，causal pad(T,2)→crop(T,-1) | `[B,2,252,129]` | float32 | `Falcon.py:1149-1156,1170-1177` |
| 7 | `torch.istft(length=32000)` | `[B,32000]` | float32 | `Falcon.py:1182-1188`；`(252-1)*128=32128≥32000`，靠 `length=` 截断 |

`[验证]` 日志 §[6][6b]：实测 `(1,112,251,129)` / `(1,1,32000)`，中间帧数 `251→253→253→252`。
**反直觉点**：解码器输出的帧数比编码器多 1（因果 pad 2 帧、只裁 1 帧），多出的最后一帧对应采样
`[32000, 32256)`，不参与 `<32000` 的重建，因此无副作用（`[验证]`：解码器对帧 `<120` 的扰动响应严格为 0）。

#### 例 2：`AudioBlock` 4 尺度账本（TFRB 的实际计算图）

输入 `x=[B,112,251,129]`（`hid_chan=64`，`upsampling_depth=4`，`audionet.audio_params`）：

| 级 | 操作 | 输出形状 | 备注 |
|---|---|---|---|
| — | `gateway` ConvNormAct(112→112,k=1,groups=112) | `[B,112,251,129]` | depthwise 1×1 + PReLU（`Falcon.py:718-726`） |
| — | `projection` ConvNormAct(112→64,k=1,norm=cLNhw) | `[B,64,251,129]` | cLNhw 在此做**全窗口**统计（`Falcon.py:727-735`） |
| L0 | downsample(k=3,s=1) | `[B,64,251,129]` | `Falcon.py:749-767` |
| L1 | downsample(k=5,s=2,pad=2) | `[B,64,126,65]` | `⌊(251+4-5)/2⌋+1=126`；`⌊(129+4-5)/2⌋+1=65` |
| L2 | downsample(k=5,s=2) | `[B,64,63,33]` | |
| L3 | downsample(k=5,s=2) | `[B,64,32,17]` | 最深尺度，仅此级在 encoder 侧跑 TFRB（`Falcon.py:842-845`） |
| L3 | `DualPathRNN(dim=4,hid=24,bidir=True)` | `[B,64,32,17]` | 沿 **F=17** 双向 LSTM |
| L3 | `DualPathRNN(dim=3,hid=72,bidir=False)` | `[B,64,32,17]` | 沿 **T=32** 单向 LSTM |
| L3 | `MHSA2D(n_head=4,hid=4,n_freqs=17,causal=True)` | `[B,64,32,17]` | 沿 T 的 masked self-attn（`Falcon.py:788-790`） |
| — | `fusion_layers[i]`（GCI）注入 global(L3) | 各级原形状 | `InjectionMultiSum`，`Falcon.py:858` |
| L2 | `concat_layers[-1]` + 残差 + TFRB(L2) | `[B,64,63,33]` | `Falcon.py:868-875` |
| L1 | 逐级上采样 + TFRB(L1) | `[B,64,126,65]` | `Falcon.py:879-888` |
| L0 | 逐级上采样 + TFRB(L0) | `[B,64,251,129]` | `Falcon.py:879-888` |
| — | `interpolate(nearest)` 到残差形状 → `residual_conv(64→112,k=1)` + residual | `[B,112,251,129]` | `Falcon.py:893-894` |

关键：**`MHSA2D` 的 `n_freqs` 必须与该级运行时 F 相等**，因为 `ConvActNorm` 的 `LayerNormalization4D((out_chan,n_freqs))`
的 `γ/β` 形状 `[1,C,1,F]` 要逐元素相乘（`Falcon.py:185-188, 785-786`）。配置里 `n_freqs: 129`（`configs/Falcon.yml:39`）
是初始值，代码在 `:774` 逐级递推 `current_f=(current_f-1)//stride+1`，得到 `129→65→33→17`，与上表一致 `[验证]`
（实测模型可前向，若不一致会广播报错）。

#### 例 3：`layers/stft.py` 的 `forward_stft → inverse_stft`（`[验证]`）

| 步骤 | 形状 | dtype |
|---|---|---|
| 输入 wav | `[2,32000]` | float32 |
| `init_kernel('hann', W=256)` | K `[512,1,256]`，window `[256]` | float32 |
| `conv1d(stride=128)` | `[2,512,251]` | float32 |
| `chunk(2, dim=-2)` + onesided 取前 129 | real/imag `[2,129,251]` | float32 |
| `stack(dim=-1)` | `[2,129,251,2]` = `[B,F,T,2]` | float32 |
| `inverse_stft` → 共轭补齐 → `conv_transpose1d` | `[2,1,32256]` | float32 |
| OLA 归一化 + center 裁 `±128` | `[2,32000]` | float32 |

与 `torch.stft`/`torch.istft` 的逐项对比见 §2.2 表；round-trip 误差 `2.384e-06`。

#### 例 4：`layers.STFT` / `iSTFT`（nn.Module 形态，未被使用）

`STFT(frame_len=256, frame_hop=128, window='sqrthann', center=True)`：`wav [N,(C),S] → [N,(C),129,T,2]`；
`iSTFT` 反向。默认窗是 **`sqrthann`**（不是 hann），与 `torch.stft` 的默认不一致，跨实现比对时必须显式指定窗。

### 3.4 数据布局与 dtype 约定

- **模型内部布局**：`[B, C, T, F]`——第 3 维是时间/帧、第 4 维是频率。所有 2D 卷积都按 `(H,W)=(T,F)` 处理
  （例如因果卷积 padding 写成 `(causal_padding, self.padding)`，`Falcon.py:59,157`）。
- **STFT 输出布局**：`layers/stft.py` 为 `[N,(C),F,T,2]`；`stft_tfgn.Stft` 为 `[B,T,F,2]`；`torch.stft` 为
  `[B,F,T] complex64`；`STFTEncoder` 先 stack 再 transpose 成 `[B,3,T,F]`。**四种布局混用，是本子系统最大的踩坑源**。
- **dtype**：`real` 一律 float32；complex 只出现在 `torch.stft`/`torch.istft` 与 `torch.complex()` 构造成员
  （complex64）。`layers/stft.py` 的 `K/w` 以 float32 `nn.Parameter(requires_grad=False)` 存储。
- **复数建模方式**：Falcon 不用 complex 卷积；它把复数谱拆成 **实部/虚部两半的实通道**：
  `STFTEncoder` 输入 3 通道 `[real, imag, mag]`（`Falcon.py:1118`）；`MaskGenerator` 的 `RI_split=True`
  把 112 通道 embedding 视为 `2×56` 的复数通道，掩码按复数乘法 `(a+bi)(c+di)` 实现（`Falcon.py:1247-1259`）。

## 4. 关键实现细节与设计动机

### 4.1 STFT/iSTFT 的窗、hop、归一化、中心填充，以及与 `torch.stft` 的等价性

Falcon 实际用的 `STFTEncoder`/`STFTDecoder`（`models/Falcon.py:1054-1192`）**直接调用** `torch.stft` / `torch.istft`：

- `n_fft = win = 256`（16 ms @16k），`hop_length = 128`（50% 重叠，8 ms 帧移）；
- 窗 = `torch.hann_window(256)`（**periodic=True 默认**，与 librosa/torch.stft 一致），注册为 buffer（`:1093`、`:1161`，不可训练）；
- **`center` 未显式给出 ⇒ 默认 `True`**（torch.stft 默认）⇒ 两端 **reflect** 填充 `n_fft//2 = 128`；
- `onesided=True`（默认）⇒ 129 个 bin；`normalized=False`（默认）⇒ 无 1/√N 缩放；
- `torch.istft(..., length=input_shape[-1])` 用 `length=` 精确还原长度（`:1187`）。

因此"是否等价 `torch.stft`"的答案是：**Falcon 就是 `torch.stft` 本身**（不存在自实现偏差）。
`layers/stft.py` 那套 conv1d-DFT 实现则与 `torch.stft` **数值等价但实现不同**（§2.2 表），且默认 `center=False`、
默认窗 `sqrthann`——若拿它做对照实验，必须显式传 `window="hann", center=True`。

**因果性含义（重要 nuance，R3 重写）**：`center=True` 意味着帧 `t` 覆盖采样 `[t·hop − 128, t·hop + 128)`。

- **帧级 lookahead**：帧 `t` 依赖未来至多 `win/2 = 128` 采样 = **8 ms**。`[验证]` 日志 §[3]：在 `p=20000` 注入脉冲，
  受影响帧恰为 `[156,157]`，与理论 `[⌈(p−127)/128⌉, ⌊(p+128)/128⌋]` 一致；R3 另测：扰动采样 `s≥16000` 后 encoder
  **帧 124 仍为 0、帧 125 起非零**（§8-[V1]）。
- **encoder→decoder 串联的采样级 lookahead：解析上界 ≤127 采样（<8 ms），不是 16 ms**。推导：
  1. 输入采样 `p=128q+r`（`r∈[0,127]`）影响 encoder 帧 `n ∈ [⌈(p−127)/128⌉, ⌊(p+127)/128⌋]`，其中 `n_min=q`；
  2. 解码器因果分支的 `ConvTranspose2d(k=3,pad=1)` + `pad(k−1)`/`crop(padding)` 使其合成帧 `m` 只依赖
     encoder 帧 `m−3…m−1`（`Falcon.py:1170-1177`）；`[验证]`：mask 帧 ≥125 时输出 `samples<16000` 严格为 0；
  3. `istft(center=True)` 的输出采样 `s` 由合成帧 `⌊s/128⌋`、`⌊s/128⌋+1` 叠加；
  4. 合成 ⇒ 最早受影响输出采样 `= 128·q = p−r` ⇒ **过去泄漏 = r ≤ 127**（再减 hann 窗首点为 0 的 1 个采样）。
  `[验证]`（R3 日志 §[V2]）：`p=16000 (r=0) → 首变输出 16001`（无过去泄漏）；`p=20000 (r=32) → 19969`，泄漏 **31**；
  `p=20032 (r=64) → 19969`，泄漏 63；`p=20127 (r=31) → 20097`，泄漏 30；`p=24000 (r=64) → 23937`，泄漏 63 —— 六点全部符合 `r−1`。
- **R3 更正**：R1 版写"采样级最坏一个窗长 = 256 采样 = 16 ms"是错的——它把合成帧 `⌊n/128⌋+1` 的分析窗直接叠加，
  忽略了 decoder 因果 ConvTranspose 的 **3 帧内部 lookback 恰好吸收了 iSTFT 合成的 ±1 帧扩散**。正确表述见上。

这与 Studio 的窗口化推理并不冲突：Studio 每次处理完整 2 s 窗口（32000 采样），窗口内部允许 ≤8 ms（帧级）与
≤127 采样（串联采样级）的对齐误差，跨窗口不泄露（`studio/falcon_model/PROVENANCE.md:41-42` 明确说明
"rolling two-second windows, not sample-by-sample streaming"）。

### 4.2 因果性逐层分析（**这是本项目"因果"声明的关键 nuance**）

**统一表述（与 `01-model-core.md` R3 一致的"窗口级因果（window-causal）"）**：
Falcon 的 "causal" 是**结构层面**的——时间轴的卷积 padding 方向、RNN 方向、注意力 mask 都只指向过去；
但整体**不是逐采样严格因果**，而是**窗口级因果**：在一个 2 s 窗口内允许三类未来依赖——
① `torch.stft(center=True)` 的 ≤128 采样/8 ms 帧级 lookahead；
② `AudioBlock` 对称卷积的 ±456 ms（单遍实测 57 帧，见下）；
③ `cLNhw` 的 `GroupNorm(1,C)` 全窗口统计（无界，依赖整个窗口）。
跨窗口无状态，所以 Studio 用"2 s 窗口 + 0.5 s hop + 80 ms 右上下文"的重叠推理。
**三个 `causal` 开关来源不同、不可混用**：`enc_dec_params.causal`（编/解码器，`Falcon.py:1308,1332-1336`）、
`audio_params.layers.layer_3.causal`（MHSA2D mask，`yml:72` + `Falcon.py:789`）、**顶层 `Falcon.causal`（惰性）**
（`Falcon.py:1314-1315` → `AudioBlock.causal` 只传给 `DualPathRNN(causal=...)`，而 `DualPathRNN` 从不读 `self.causal`；
`[验证]` 见 01-A25：`causal=True/False` 两模型同权重下输出 bit-identical）。

实际是否因果，**必须逐模块看**：

| 模块 | 因果参数 | 实际时间因果性 | 证据 |
|---|---|---|---|
| `STFTEncoder` **conv 子层** (3→112,k=3) | `causal=True` | ✅ **子层**严格因果（`padding=(2,1)` 后裁 `[:-2]`） | `Falcon.py:48-67,100-104`；`[验证]` 帧 124 仍 0（仅代表 conv 子层，不能代表整体） |
| `torch.stft(center=True)`（encoder 前置） | — | ⚠️ **帧级 lookahead ≤128 采样/8 ms** | `Falcon.py:1110-1116`；`[验证]` 帧 125 起非零（§8-[V1]） |
| **`STFTEncoder` 整体** | `causal=True`（仅作用于 conv） | ⚠️ **非严格逐采样因果**（= conv 子层因果 + 上述 STFT 帧级 lookahead） | 见 §6-A36′；R3 修正 R1 的"严格因果"错误 |
| `audio_bottleneck`（pre_norm=cLNhw） | `causal=` 未传 | ❌ **全窗口统计**（GroupNorm 跨 T）+ 1×1 conv | `[验证]`：波形后 1 s ×10 → 左段 `max|Δ|=4.7e-05`（§8-[V3]） |
| `AudioBlock.gateway / projection` | `causal=False`（`:725,734`） | k=1 卷积本身无时间感受野；但 `projection` 的 `cLNhw` 非因果 | `Falcon.py:727-735` |
| `AudioBlock.downsample_layers` | `causal=False`（`:765`） | ❌ k=3/5 对称 padding；**逐级 lookahead（换到 L0 帧）= L0 ±1 / L1 ±4 / L2 ±8 / L3 ±16**（解析：`⌈(k−1)/2⌉·2^level`） | `Falcon.py:751-767`；[验证] 形状 + 单遍 57 帧 |
| `InjectionMultiSum`（GCI，fusion/concat） | `causal=False`（`:801,814`） | ❌ 仅因 **k=5 对称卷积** 非因果（±2 帧）；`interpolate(nearest)` 是**单调映射**（输出 t 只取 `⌊t·ratio⌋` 的过去/当前源），**不是**非因果致因 | `Falcon.py:283-329,319-325`；[验证] 去 cLNhw 后脉冲响应 `[98,102]`（§8-[V4]） |
| `AudioBlock` 单遍整体（去 cLNhw 的纯结构） | — | ❌ 实测 **lookahead 57 帧 = 7,296 采样 ≈ 456 ms**（扰动帧 ≥125 → 输出首变帧 68） | `[验证]` R3 §[V4]；R2 评审同值 |
| `AudioBlock.residual_conv` | `causal=False`（`:823`） | k=1，**无时间感受野 ⇒ 不能作为非因果证据**（R3 更正 R1 引用） | `Falcon.py:818-824` |
| `DualPathRNN(dim=4, bidir)` | 频轴双向 | ✅ 时间因果（沿 F 方向，不跨 T） | `[验证]` 日志 §[8]：早期 t 完全不变 |
| `DualPathRNN(dim=3, unidir)` | `causal=self.causal`（`:781`） | ✅ 时间因果（单向 LSTM 沿 T） | 同上，`max|Δ|=0` |
| `MultiHeadSelfAttention2D(causal=True)` | `layer_3.causal: true`（yml:72） | ✅ 时间因果（`tril` mask 强制 `attn_scores` 只看过去）；但整段 T 一次性算、无 KV cache ⇒ 数学因果但非流式 | `Falcon.py:520-524`；`[验证]` 早期 t `max|Δ|=0`；`causal=False` 时 `max|Δ|=5.1` |
| `STFTDecoder` convtranspose | `causal=True` | ✅ **帧级**严格因果（pad 2 帧再裁 1 帧；合成帧 m 依赖 encoder 帧 m−3…m−1 = 3 帧内部 lookback） | `[验证]` 日志 §[7b]：`<15360` 采样扰动响应严格为 0 |
| `torch.istft(center=True)` | — | ⚠️ 合成帧 `m` 对输出采样 ±win/2 扩散；但被 decoder 的 3 帧 lookback 吸收 ⇒ **串联采样级 ≤127 采样**（§4.1） | `[验证]` R3 §[V2] |
| `VideoBlock`（视频分支） | `is2d=False`, k=1 conv + 单向 LSTM + `BatchNorm1d(32)` | ✅ **eval** 下时间因果（单向 LSTM + running stats）；⚠️ **train 模式** `BatchNorm1d` 用批内统计（含未来帧）⇒ 训练期非因果 | `Falcon.py:940-972`；`[验证]`：eval 左段 0.0、train 左段 `4.8e-02`（§8-[V4b]） |
| FRCNN 前端 `frontend3D`（冻结、04 域） | `Conv3d(k=(5,7,7),padding=(2,3,3))` + `MaxPool3d(k=(1,3,3))` | ⚠️ 时间 **±2 视频帧 = ±80 ms** lookahead | `videomodels/frcnn_videomodel.py:58-70`；`[验证]`：改帧 ≥30 → 最左受影响输出帧 28（交叉引用 `04-losses-metrics-video.md`） |

**为什么"GN 是全窗口统计"却常只测到 1e-6 量级？——两个抵消机理（R3 新增，回应两份评审 P1-②）**：

1. **`CumulateLN4D` 抵消 GN 的"纯尺度型"泄漏**：`cLNhw = GlobalLayerNorm + CumulateLN4D`。
   同一 "后半段 ×10" 探针下，单独 `GlobalLayerNorm` 左段泄漏 **3.930**（强非因果），而完整 `cLNhw` 只剩 **1.435e-03**
   （相差约 3 个数量级）——第二段沿时间累积的归一化会把"整体尺度变化"重新标准化掉（§8-[V3d]）。
2. **common-mode 抵消**：对**整段所有通道施加同一常数**（common-mode）时，`GlobalLayerNorm` 与 `CumulateLN4D`
   的统计都包含 channel 维，会把它整体减掉：`[验证]` 整段+全通道 +5 → 全段 `max|Δ|=7.15e-07`；
   而整段+单通道 +5 → **5.084**；单帧+全通道 +20 → 前缀 1.165e-03、单帧+单通道 +20 → 2.07e-05（§8-[V3b]）。
   ⇒ **任何因果性数字都必须写明"单通道 vs 全通道、单帧 vs 整段、T 长度、测点层级"**，否则不可比。

**`cLNhw` 在模型里的实例密度**：论文配置下 **30 个 `LayerNormalizationHW`**（= 30 个 `GlobalLayerNorm` + 30 个
`CumulateLN4D`；另有 60 个 `LayerNormalization4D`）——来源：`audio_bottleneck` 1 + `projection` 1 +
`downsample_layers` 4 + `fusion/concat` 3×8 = 24（§8-[V7]）。这说明"全窗口统计"不是某一层的瑕疵，而是音频主干每层都有。

**结论**：编码器 conv 子层、解码器 convtranspose、`DualPathRNN`、`MHSA2D(causal=True)` 在**结构/帧级**是因果的；
但 **`cLNhw` 的 `GroupNorm(1,C)` 在全窗口上统计、`AudioBlock` 的下采样/融合卷积是 `causal=False`**，所以整个
`Separator` **不是逐采样严格因果**。`[验证]`：把后半段音频放大 10×，`Separator` 输出前 120 帧仍变化 `1.7`
（相对幅度与后段同量级），最终波形 `[0:8000]` 段 `max|Δ|=9.7e-04`、`[8000:16000]` 段 `1.65e-01`——存在真实未来信息泄漏。
这就是为什么必须用 **2 s 窗口 + 右上下文** 的窗口化推理：算法在窗口内是"窗口级因果"，跨窗口靠 hop 与 80 ms guard 掩盖。

### 4.3 TFRB 归属与实现细节（必答）

**归属结论**：TFRB 的组件全部定义在 `look2hear/models/Falcon.py`，**不在 `layers/rnnlayers.py` 或 `cnnlayers.py`**；
`rnnlayers.py` 的 `DPRNN`/`DPRNNBlock` 与 TFRB 无关（SepFormer 式 chunk RNN，无调用方）。
TFRB 用到的"底层"只有 `layers/normalizations.py` 的 `LayerNormalization4D`（以及 `torch.nn.LSTM` 直连）。

**(a) 双向频轴 LSTM — `DualPathRNN(dim=4, rnn_type="LSTM", bidirectional=True, hid_chan=24)`**（`Falcon.py:331-415`）：

- 输入 `[B,C,T,F]`；`dim==4` 时先 `transpose(-2,-1)` 成 `[B,C,F,T]`（`:386-387`）；
- 统一按 `B,C,T',F'` 解包后 `permute(0,3,1,2).view(B*F', C, T')`（`:389-392`）。
  dim=4 时 `T'=F, F'=T` ⇒ **batch = B·T（每个时间帧独立），序列维 = F（频率轴）**；
- `nn.LSTM(input_size=C, hidden_size=hid_chan, bidirectional=True)`（`:372-377`，`batch_first=False`），
  输出 `[F, B·T, 2·hid]`；`out_proj = Linear(2·hid, C, bias=False)`（`:380-383`，因为 `2·24=48 ≠ 64`）；
- 残差 `+residual` 后 `transpose` 回 `[B,C,T,F]`（`:408-413`）。
- ⇒ **沿频率轴双向：频率维无因果要求（频带同时可见），时间维不受影响。**
- `[验证]` 日志 §[9]：`rnn.input_size=64`、`hidden_size=24`、`bidirectional=True`、`out_proj=Linear(48→64)`。

**(b) 单向时间轴 LSTM — `DualPathRNN(dim=3, rnn_type="LSTM", bidirectional=False, hid_chan=72)`**：

- dim≠4 不转置；`permute(0,3,1,2).view(B*F, C, T)` ⇒ **batch = B·F（每个频点独立），序列维 = T（时间轴）**；
- 单层单向 LSTM（`batch_first=False`）⇒ 严格沿时间因果；
- `out_proj = Linear(72, 64, bias=False)`（72≠64）；残差相加（`:410`）。
- 实现细节坑：`num_direction = int(bidirectional)+1`（`:367`）仅用于算 `rnn_out_chan`；`rnn_type` 走
  `getattr(nn, self.rnn_type)`（`:372`），**默认值 `rnn_type="Comba"` 在 torch 中不存在**——`torch.nn` 没有 `Comba`
  （`[验证]`：`AttributeError: module 'torch.nn' has no attribute 'Comba'`），该默认值是遗留自 flash-linear-attention
  的分支（`:394-397`，`SISNRRepel.py:9` 有注释掉的 `from fla.layers import Comba`）。**只有配置显式传
  `rnn_type: LSTM` 才能构造成功**（`configs/Falcon.yml:55,63`）。
- 另有"死参数"：`kernel_size=8, stride=1, num_heads, head_dim, expand_v, use_output_gate, use_short_conv, ...`
  全部只存不用（`:337-352`），是 Comba 分支的残留接口。

**(c) masked 2D self-attention — `MultiHeadSelfAttention2D`（`Falcon.py:417-543`）**：

- 4 个 head，每个 head 独立 `ConvActNorm(64→4,k=1,is2d,act=PReLU,norm=LayerNormalization4D((4,n_freqs)))` 产生
  Q/K（`:445-469`），V 为 `ConvActNorm(64→16,...)`（`:470-481`）；
- `Q/K: [B,4,T,F] → cat(dim=0) [B·4,4,T,F] → transpose(1,2).flatten(2) → [B·4, T, 4F]`（`:505-510`）；
  `V: [B·4,16,T,F] → [B·4,T,16F]`（`:511-513`）；
- `attn = softmax(QKᵀ/√(C·F))`，形状 `[B·4, T, T]`（`:514-526`）——**注意力沿时间轴，频率被摊平进 embedding**；
- **因果**：`causal=True` 时用 `torch.tril` 掩码把 `t' > t` 的分数置 `-inf`（`:520-524`）⇒ 只看过去帧；
- 输出 `[B·4,T,16F] → reshape [B·4,T,16,F] → [B,4,16,T,F] → [B,64,T,F]`，再过 `attn_concat_proj` 1×1 conv + 残差（`:528-543`）。
- 计算量：`O(T²·C·F)`。T=251 时注意力矩阵 `251×251`（可接受）；这也是它只放在 `n_freqs` 与 T 最小的深层/或
  每级各一份的原因之一。

**GCI / `InjectionMultiSum`（`Falcon.py:267-329`）**：`out = local_emb(x_local) * gate(x_global→interp) + global_emb(x_global→interp)`，
其中 `gate` 用 `Sigmoid`（`:303-313`）、`global_*` 先 `F.interpolate(..., mode="nearest")` 对齐形状（`:319-325`）⇒
"门控的全局上下文注入"。全部 1×1/深度可分离 5×5 conv + `cLNhw`。

### 4.4 `cLN/cLNhw/LayerNormalization4D/BatchNorm1d` 的精确语义与广播维度（必答，汇总表）

设输入 `[B,C,T,F]`：

| 名称 | 公式 | 统计维度 | `γ/β` 形状 | 时间因果性 |
|---|---|---|---|---|
| `LayerNormalization4D((C,1))` | `(x−μ)/√(σ²+ε)·γ+β` | `dim=(1,)`：对 C（逐 b,t,f） | `[1,C,1,1]` | ✅（无时间混合） |
| `LayerNormalization4D((C,F))` | 同上 | `dim=(1,3)`：对 (C,F)（逐 b,t） | `[1,C,1,F]` | ✅（逐帧） |
| `cLNhw = LayerNormalizationHW(C)` | 先 `GroupNorm(1,C)`；再逐 f 的 `CumulateLN4D` | GN：对 (C,T,F)；cLN4D：对 (C, t'≤t)，逐 f | GN 有 `[C]`/`[C]`；cLN4D 有 `[1,C,1,1]`×2 | ❌ GN 段非因果；cLN4D 段因果 |
| `gLN = GlobalLayerNorm(C)` | `(x−μ)/√(σ²+ε)` + per-channel 仿射 | 对 (C,T,F) | `[C]`/`[C]`（GroupNorm 内） | ❌ |
| `cLN = CumulateLN(C)`（3D 输入） | `(x−μ_t)/√(σ²_t+ε)·γ+β` | 对 (C, t'≤t)（无 f 维） | `[1,C,1]`×2 | ✅ |
| `BatchNorm1d(C)`（`get` 解析到 torch 原生） | 训练：batch 统计；推理：running stats | 对 (B,T)（逐 C） | `[C]`×2 | eval 时与时间无关（但训练期跨 batch 统计） |

补充：`cLNhw` 的总参数 = `GroupNorm` 的 `2C` + `CumulateLN4D` 的 `2C` = `4C`（C=112 时 448）；
`LayerNormalization4D` 参数 = `2·C·max(F,1)`。`[验证]` 所有形状/数值断言见日志 §[4]。

### 4.5 `enc_dec.py` 的编码器/解码器参数化与"因果卷积"公式（对照说明）

虽然 Falcon 不用它，但任务要求给出参数化与 stride/padding 公式（便于与 `STFTEncoder` 对照）：

- `Encoder.forward`（`:254-281`）：`spec = conv1d(waveform, filters, stride=H, padding=P)`；
  输出帧数 `T_out = ⌊(S + 2P − W)/H⌋ + 1`，`W=kernel_size`，`H=stride`（缺省 `W//2`）。
- `Decoder.forward`（`:369-392`）：`wav = conv_transpose1d(spec, filters, stride=H, padding=P, output_padding=P_out)`；
  输出长度 `S_out = (T−1)·H − 2P + W + P_out`；再按 `length` 截断（`:389-391`）。
- `who_is_pinv`（`:47-60`）：伪逆绑定，`compute_filter_pinv`（`:201-207`）用 `torch.pinverse` + `stride/kernel_size` 缩放补偿 OLA。
- **因果卷积公式在 Falcon 里是另一套**（`ConvNormAct`，`models/Falcon.py:48-67,100-104`）：
  设 `k=kernel_size, d=dilation`，`p_causal = d·(k−1)`；卷积用**左右对称** `padding=p_causal`（1D）或
  `(p_causal, p_same)`（2D，其中 `p_same = d·(k−1)//2` 是 F 维的 same padding），输出后裁掉末尾 `p_causal`：
  - 1D：`y[t] = Σ_{i=0}^{k-1} w[i]·x[t − p_causal + i·d]` ⇒ 只用到 `t − p_causal … t`，**严格因果**，输出长度不变。
  - 2D：`y[t,f] = Σ_{i,j} w[i,j]·x[t − p_causal + i·d, f − p_same + j·d]` ⇒ **只在 T 方向因果，F 方向 same**。
  - `causal=False` 时 `padding = d·(k−1)//2`（`:34,143`）⇒ 对称 same 卷积，**非因果**。
- `Decoder`（`STFTDecoder`）的因果公式：`F.pad(x, (0,0,k−1,0))` 只在 T 的**前侧**补 `k−1`，`ConvTranspose2d(padding=(k−1)//2)`
  后 `[:, :, :-(k−1)//2, :]`（`Falcon.py:1170-1177`）⇒ 把转置卷积的 `k−1` 个"历史样本"对齐回因果。

### 4.6 Studio 的 "CPU FFT bridge" 对应什么（必答）

`studio/falcon_model/` 是训练代码的**推理版拷贝**（`studio/falcon_model/PROVENANCE.md`）：

- **目录裁剪**：只保留 `layers/{__init__.py, normalizations.py, activations.py}`，**没有 `stft.py` / `stft_tfgn.py`**
  （`studio/falcon_model/layers/__init__.py` 只有一行 docstring）；因此 Studio 侧**不需要 librosa / torch_complex /
  typeguard / espnet**。`normalizations.py`/`activations.py` 与训练侧仅注释语言不同（26 行 diff，全为中文注释）。
- **`timm` → 本地 `drop_path.py`**（`studio/falcon_model/models/Falcon.py:6` `from ..drop_path import DropPath`；
  PROVENANCE:15）——eval 模式下 DropPath 是恒等。
- **CPU FFT bridge**（PROVENANCE:20-21）= 在 `Studio/models/Falcon.py` 的 `STFTEncoder.forward` / `STFTDecoder.forward`
  里对 **Apple MPS 设备**做三处搬运（训练侧无此逻辑）：
  1. `STFTEncoder.forward`：`compute_device=x.device; stft_x = x.cpu() if x.device.type=="mps" else x;`
     → `torch.stft(stft_x,...)` → 组装 `[real,imag,mag]` → `spec.to(compute_device)` 再进卷积
     （**精确行号 `studio/falcon_model/models/Falcon.py:1111-1122`**：1110 是注释、1123 已是卷积；
     对照训练侧 `look2hear/models/Falcon.py:1107-1119`）。
  2. `STFTDecoder.forward`：先 `output_device=...`，若 MPS 则把 `decoded_separated_audio` 搬到 CPU 再
     `torch.complex(...)` 与 `torch.istft(...)`，最后 `output.to(output_device)`
     （`studio/falcon_model/models/Falcon.py:1183-1198`，对照训练侧 `:1176-1190`）。
  - 动机：MPS 后端对 complex tensor / 通用 STFT 支持不全；搬 CPU 后**算子与参数完全不变**，因此在 CPU 上
    与训练代码**逐位一致**（PROVENANCE:33 声称 max abs diff = 0）。**除设备搬运外没有任何数值改动** `[代码]`+`[验证]`。
  - **R3 用真实权重验证**（`/root/.cache/falcon_weights/falcon.pt`，只读）：Studio 副本与训练副本
    `load_state_dict(audio_model.*, strict=True)` 均 **All keys matched**（各 525 key、537,697 参数），
    同输入下 CPU 输出 **`max|Δ| = 0.000e+00`（bit-identical）**；Studio 副本 import 时 **不加载**
    librosa/torch_complex/look2hear（§8-[W1][W2]）。这坐实了 A44/A52/A53 的等价性主张。
- 另外 `studio/falcon_runtime.py:149-150` 对配置做 `copy.deepcopy`，因为 `Separator.__init__` 会 `pop` 配置键
  （`Falcon.py:991-997`，`[验证]` 日志 §[10]：构造后 `audio_params` 的 `repeats`/`shared` 被删除）。

### 4.7 依赖与"导入即加载全部"的代价（可运维问题）

- `import look2hear.models`（更准确地说 `import look2hear.layers`）会拉入：`timm`（DropPath）、`librosa`
  （layers/stft.py）、`torch_complex` + `typeguard` + `espnet`（layers/stft_tfgn.py → utils.complex_utils/nets_utils）、
  **`pytorch_lightning`**。**PL 的加载链路（R3 修正归因）**：`layers/__init__.py:20` → `stft_tfgn.py:12
  from ..utils.nets_utils import make_pad_mask` → 执行 `look2hear/utils/__init__.py` →
  **`:12 from .lightning_utils import ...`** → `lightning_utils.py:3 from pytorch_lightning.utilities import rank_zero_only`
  ⇒ **utils 包的顶层导入**，与 `BaseModel.serialize` 无关（后者另有一处函数内延迟 import，`base_model.py:81`）。
  `[验证]` R3 日志 §[V7]：干净进程 `import look2hear.layers` 后 `sys.modules` 含 PL/librosa/torch_complex/typeguard。
  （注：R2 评审曾判断"import 不会拉入 PL"，该判断不成立，见 R2 评审 N1；本版保留"会拉入"结论、修正归因。）
- `layers/stft.py:10` 用 `from distutils.version import LooseVersion`，**`distutils` 在 Python ≥3.12 已被移除** ⇒
  在 3.12+ 解释器上 `import look2hear.models` 会直接 `ModuleNotFoundError` `[推断]`（当前验证环境 3.10 可用；
  训练 requirements 未固定 python 版本，`requirements.txt` 无 python 声明）。
- `layers/stft.py:13-18` 的 `TORCH_VERSION >= LooseVersion("1.7")` 把 `str` 与 `LooseVersion` 比较，
  依赖 distutils 的 `LooseVersion.__ge__` 内部把 str 转成版本（`[验证]`：torch 2.8 下可运行）。
- `_pytorch_stft`/`_pytorch_istft` 使用 `return_complex=False`（`:401,469`），在 torch 2.8 触发
  `UserWarning: stft with return_complex=False is deprecated`（`[验证]` 运行时警告），未来版本会报错。

### 4.8 同名不同义的坑（跨文件一致性）

1. **`ConvNormAct`**：`layers/cnnlayers.py:66` 是 **1D** 版（`padding=0` 默认、`act_type="prelu"` 小写、无 `is2d`/`causal`）；
   `models/Falcon.py:9` 是 **2D/1D 通用**版（自动算 `causal_padding`、`pre_gelu`/`pre_norm` 两段式、`is2d`/`causal`）。
   `layers/__init__.py:6` 导出的是**前者**，但 Falcon 用的是**后者**（局部定义，不需要导入）。
2. **`CumulateLN`**：`normalizations.py` 里定义两次（`:51` 与 `:91`），后者覆盖前者；`cLN` 别名指向 `:91`。
3. **`LN` ≠ LayerNorm**：别名 `LN = CumulativeLayerNorm`（`:226`），而该类**没有累积**语义；
   真正的"累积 LN"是 `cLN`。命名反直觉。
4. **`STFT` 类**：`layers/stft.py:699` 与 `utils/stft.py:699` 两份同名；`losses/matrix.py:59` 第三份；含义/布局都不同。
5. **`STFTBase`**：`layers/stft.py:610` 与 `losses/matrix.py:13` 两套完全不同的基类（后者要求在构造时传 `device`，
   且 `num_bins = K.shape[0]//2`——与前者 `K.shape[0]//4+1` 不同）。
6. **`MLayerNorm.CumulateLN`**（`:51`）的 `cumsum(dim=1)` 是**错的**（应沿时间 dim=2），且 `beta` 初始化成 1。

## 5. 对外接口契约（谁调用 / 依赖 / 形状 / dtype / 约定）

### 5.1 层接口表

| 模块/类 | 输入 | 输出 | dtype | 因果/填充约定 | 谁调用 |
|---|---|---|---|---|---|
| `layers.stft.STFT(frame_len,frame_hop,window='sqrthann',center=False,mode='librosa')` | `[N,(C),S]` | `[N,(C),F,T,2]`（`return_polar` 时 mag/phase） | float32 | 默认 `center=False`；`center=True` 用 reflect pad `B/2` | 无人调用（仅 `layers/__init__` 导出 2 个函数） |
| `layers.stft.iSTFT` | `[N,(C),F,T,2]` | `[N,(C),S]` | float32 | OLA + 窗平方归一化 | 无人调用 |
| `layers.stft.forward_stft/inverse_stft` | 同上 | 同上 | float32 | 7 个布尔/字符串开关 | 无人调用（唯一 import 点 `layers/__init__.py:19`） |
| `utils.stft.STFT` | 同上 | 同上 | float32 | 同上 | 无人调用（`utils/__init__.py:1`） |
| `layers.stft_tfgn.Stft(n_fft=512,hop=128,center=True)` | `[B,S]` 或 `[B,S,C]` | `([B,T,F,2], olens)` | float32 | `center=True` 默认 | 无人调用 |
| `layers.normalizations.get(id)` | — | 归一化**类** | — | `None→nn.Identity`、str→torch 原生优先 | `models/Falcon.py:87-90,186-188,370` |
| `layers.activations.get(id)` | — | 激活**类/函数** | — | 同上 | `models/Falcon.py:88,90,185` |
| `layers.enc_dec.make_enc_dec` | — | `(Encoder, Decoder)` | float32 | 无因果概念 | 无人调用 |
| `layers.cnnlayers.*` | 多为 `[B,C,T]` | 同/变 C | float32 | 仅 `Conv1DBlock(causal=True)` 有因果版本 | 无人调用 |
| `layers.rnnlayers.*` | `[B,C,T]`/`[B,T,C]`/`[B,C,K,L]` | 同名 | float32 | DPRNN/Conformer，无全局因果保证 | 无人调用 |
| `models.STFTEncoder(win,hop,out_chan,kernel_size,stride,act_type,norm_type,bias,causal)` | `[B,S]` 或 `[B,1,S]` | `[B,out_chan,1+⌊S/hop⌋, win/2+1]` | float32 | **conv 子层**严格因果（T 方向）；**整体**因 `torch.stft(center=True)` 有 ≤128 采样/8 ms 帧级 lookahead ⇒ 非严格逐采样因果（§4.2/A36′） | `Falcon.py:1308` |
| `models.STFTDecoder(win,hop,in_chan,n_src,kernel_size,stride,bias,causal)` | `[B,n_src,in_chan,T,F]` + `input_shape` | `[B,n_src,L]` | float32 | **帧级**严格因果（`causal=True`，合成帧 m 依赖 encoder 帧 m−3…m−1 = 3 帧内部 lookback）；串联采样级过去泄漏 ≤127 采样 | `Falcon.py:1332` |
| `models.DualPathRNN(in_chan,hid_chan,dim,bidirectional,...)` | `[B,C,T,F]` | `[B,C,T,F]` | float32 | `dim=4`→频轴；`dim=3`→时间轴 | `AudioBlock`（`Falcon.py:777-782`） |
| `models.MultiHeadSelfAttention2D(in_chan,n_freqs,n_head,hid_chan,causal)` | `[B,C,T,F]` | `[B,C,T,F]` | float32 | 沿 T 的 masked attention | `AudioBlock`（`Falcon.py:788-790`） |
| `models.InjectionMultiSum(in_chan,kernel_size,norm_type,is2d,causal)` | `local[B,C,T,F]`, `global[B,C,T',F']` | `[B,C,T,F]` | float32 | `nearest` 插值对齐（**单调映射，因果**）+ k=5 对称卷积（非因果 ±2 帧）+ `cLNhw` | `AudioBlock`（`Falcon.py:796-816,858-888`） |

### 5.2 关键常量契约

- `win=256`、`hop_length=128`、`out_chan=112`、`n_freqs=129`（`configs/Falcon.yml:28-30,39`），采样率 16 kHz、segment 2.0 s（`configs/Falcon.yml:161-162`）。
- 帧数：`T = 1 + L//128`；频点数：`F = win/2 + 1 = 129`；解码器输出帧数 = 编码器帧数 + 1（§3.3 例 1）。
- 掩码复数约定：`RI_split=True` ⇒ 通道维前一半=实部、后一半=虚部（`Falcon.py:1247-1259`）。
- 音频内部布局：`[B,C,T,F]`，T 在 F 之前（`Falcon.py:95,1164,1466` 等注释）。
- Studio 侧：无 `layers/stft*.py` 依赖，`STFTEncoder/Decoder` 在 MPS 上走 CPU FFT bridge（§4.6）。

## 6. 可验证断言表

> 证据列行号以 commit `7576854` 为准。本表共 **64 条**（A01–A64，A36 已按 R3 评审改为 A36′）。
> `[验证]` 对应两份日志小节：R1 = `docs/verification/runs/layers-dsp-verify.log`（`§[n]`）；
> R3 = `docs/verification/runs/r3-layers-revision.log`（`§[Vn]`）与 `runs/r3-studio-weights.log`（`§[Wn]`）。

| 编号 | 断言 | 证据 `path:line` | 验证方式 | 置信度 |
|---|---|---|---|---|
| A01 | `layers/stft.py` 与 `utils/stft.py` 是同源复制，差异仅 2 行空白 | 两文件 785 行；diff 见 §2.12 | `diff -u`/`diff -B`/去空白 md5 | `[验证]` |
| A02 | 差异 = 1 处新增（utils 第 1 行空行）+ 1 处删除（layers:164 四空格行） | `utils/stft.py:1`；`layers/stft.py:164` | `diff` 输出 | `[验证]` |
| A03 | `layers/stft.py` 唯一 import 点是 `layers/__init__.py:19`（**限产品代码**；`docs/verification/scripts/` 的自检脚本除外） | `look2hear/layers/__init__.py:19` | grep 产品代码 | `[代码]` |
| A04 | `utils/stft.py` 唯一 import 点是 `utils/__init__.py:1` | `look2hear/utils/__init__.py:1` | grep 全仓 | `[代码]` |
| A05 | 二者在产品代码中**都没有被调用**（死代码；仅 R1 验证脚本调用） | 产品代码无 `forward_stft(`/`inverse_stft(`/`utils.STFT(` 调用 | grep + 日志 §[0] | `[代码]`+`[验证]` |
| A06 | `import look2hear.models` 会加载 `layers/stft.py` 与 `utils/stft.py` | `models/Falcon.py:7`→`layers/__init__.py:19`→`stft_tfgn.py:10-12`→`utils/__init__.py:1` | `sys.modules` 检查（日志 §[0]） | `[验证]` |
| A07 | `layers/stft.py` 顶层依赖 librosa 与 distutils | `layers/stft.py:7,10` | `sys.modules` 出现 `librosa`（日志 §[0]） | `[验证]` |
| A08 | conv1d-DFT `forward_stft` 与 `torch.stft` 数值等价（float32 误差级） | `layers/stft.py:221-286` | `max|Δ|=2.289e-05`（日志 §[1]） | `[验证]` |
| A09 | `mode="torch"` 路径与 `torch.stft` 逐位一致 | `layers/stft.py:357-413` | `max|Δ|=0`（日志 §[1]） | `[验证]` |
| A10 | `inverse_stft` 与 `torch.istft` 等价；round-trip 误差 ~2.4e-6 | `layers/stft.py:289-354` | 日志 §[2] | `[验证]` |
| A11 | `forward_stft`/`STFTBase` 默认 `center=False`（与 torch 相反） | `layers/stft.py:484,636` | 读源码 + 帧数实测 249 vs 251 | `[代码]`+`[验证]` |
| A12 | 默认窗 `sqrthann = hann**0.5`，除 rect 外 `periodic=True` | `layers/stft.py:37-38,51-53` | 读源码 | `[代码]` |
| A13 | `round_pow_of_two` 会放大 FFT 点数并改变 bin 数 | `layers/stft.py:82-89` | 200→256→129 bins；400→512→257 bins（日志 §[1]） | `[验证]` |
| A14 | 帧数公式 `T=1+⌊L/hop⌋`（center=True） | `layers/stft.py:673-682` | 32000/128→251（日志 §[2]） | `[验证]` |
| A15 | `export_jit` 是死代码且必然抛错（对 nn.Module 迭代） | `layers/stft.py:21-26` | 静态分析 | `[代码]` |
| A16 | `normalizations.get(None)` 返回 `nn.Identity` **类** | `normalizations.py:231-233` | 日志 §[4] | `[验证]` |
| A17 | `get("BatchNorm1d")` 解析为 `torch.nn.BatchNorm1d`，非本文件包装类 | `normalizations.py:237-238` | 日志 §[4] | `[验证]` |
| A18 | `GlobalLayerNorm(C)` == `nn.GroupNorm(1,C)`；统计跨 (C,T,F) | `normalizations.py:134-143` | 与手算 GroupNorm 一致（日志 §[4]） | `[验证]` |
| A19 | `cLNhw` 先做全窗口 `GroupNorm`，再做沿 T 的累积 LN ⇒ **非严格因果**；**数字强依赖扰动协议**，且 `CumulateLN4D` 抵消大部分纯尺度泄漏 | `normalizations.py:204-221` | 协议表见 §4.2/**§8-[V3]**：T=8 单帧尖峰+20 `1.165e-03`；T=251 波形×10（pre_norm）`1.43e-06`~`8.49e-05`（随随机种子/协议）；同分布替换 `1.43e-06`；GN-only ×10 `3.93` vs cLNhw ×10 `1.44e-03` | `[验证]` |
| A20 | `CumulateLN4D` 统计 = (C, t'≤t)，逐频点 f 独立；gain/bias `[1,C,1,1]` | `normalizations.py:165-202` | 与手算逐元素一致（日志 §[4]） | `[验证]` |
| A21 | `LayerNormalization4D((C,1))` 的 `dim=(1,)`，`γ` 形状 `[1,C,1,1]` | `normalizations.py:146-157` | 日志 §[4] | `[验证]` |
| A22 | `LayerNormalization4D((C,F))` 的 `dim=(1,3)`，`γ` 形状 `[1,C,1,F]` | `normalizations.py:146-157` | 日志 §[4] | `[验证]` |
| A23 | `cLN = CumulateLN`，3D 输入、累积统计、gain `[1,C,1]` | `normalizations.py:91-132,225` | 与手算一致（日志 §[4]） | `[验证]` |
| A24 | `LN` 别名指向 `CumulativeLayerNorm`，无累积语义 | `normalizations.py:74-88,226` | 读源码 | `[代码]` |
| A25 | `CumulateLN` 在文件中被定义两次，后者覆盖前者；前者 `cumsum(dim=1)` 是 bug | `normalizations.py:51,91` | 读源码 | `[代码]` |
| A26 | `MLayerNorm.beta` 初始化为 1（非 0）且该类无使用者 | `normalizations.py:27-28` | 读源码 + grep | `[代码]` |
| A27 | `activations.get('PReLU'|'ReLU'|'Tanh')` 走 `hasattr(nn,·)`；`'prelu'` 走 `globals()` 函数 | `activations.py:48-62` | 日志 §[5] | `[验证]` |
| A28 | Falcon 对 `layers` 的运行时依赖只有 `normalizations`/`activations` | `models/Falcon.py:7`；全仓 grep | grep | `[代码]` |
| A29 | `cnnlayers/rnnlayers/enc_dec` 全部类在 Falcon 路径零调用 | `layers/__init__.py:1-20`；grep 零命中 | grep | `[代码]` |
| A30 | `rnnlayers.DPRNN*` 与 Falcon 的 `DualPathRNN` 无继承/调用关系 | `rnnlayers.py:586-908` vs `models/Falcon.py:331-415` | grep + 类型检查（日志 §[11]） | `[代码]`+`[验证]` |
| A31 | `rnnlayers.DPRNNLinear.forward:882` 的 `transpose(1,1)` 是 no-op，形状不匹配 ⇒ 不可用 | `rnnlayers.py:882-884` | 静态推演 | `[代码]` |
| A32 | `enc_dec.py` 无 causal 概念，`padding` 对称 ⇒ 非因果 same 卷积 | `enc_dec.py:254-281,369-392` | 读源码 | `[代码]` |
| A33 | `STFTEncoder` 输出 `[B,112,251,129]`（B=1,L=32000） | `models/Falcon.py:1054-1121` | 日志 §[6]：实测一致 | `[验证]` |
| A34 | 编码器帧数 `T=1+L/hop=251`、频点 `F=win/2+1=129` | 同上；`configs/Falcon.yml:28-29` | 日志 §[6] | `[验证]` |
| A35 | `STFTEncoder` conv 因果裁剪后 T 不变（251）：pad(2,1)→253→裁 2 | `models/Falcon.py:59,100-104` | 形状推演 + 日志 §[6] | `[验证]` |
| **A36′** | **`STFTEncoder` 的 conv 子层严格时间因果**（帧 0–123 在扰动后半段音频下 `max|Δ|=0`）；**但整体非严格逐采样因果**：`torch.stft(center=True)` 带来 **≤128 采样/8 ms 帧级 lookahead**——扰动 `s≥16000` 时帧 124 仍为 0、**帧 125 起非零** | `models/Falcon.py:48-67,100-104,1110-1116` | R3 日志 §[V1]：`s=16000→首变帧 125`；`s=20000→156` | `[验证]` |
| A37 | `torch.stft(center=True)` 帧级 lookahead ≤ win/2 = 128 采样（8 ms） | `models/Falcon.py:1110-1116` | 脉冲探测：受影响帧 [156,157]（日志 §[3]）+ R3 §[V1] 多点 | `[验证]` |
| A38 | `STFTDecoder` **帧级**严格时间因果（causal 模式下；合成帧 m 依赖 encoder 帧 m−3…m−1） | `models/Falcon.py:1170-1177` | 日志 §[7b]：`<15360` 采样响应为 0；R3 §[V2] 串联采样级 | `[验证]` |
| A39 | 解码器内部帧数：251→(pad)253→(convT)253→(crop)252，再多 1 帧由 `length=` 截断 | `models/Falcon.py:1143,1170-1188` | 日志 §[6b] | `[验证]` |
| A40 | `DualPathRNN(dim=4)` 沿频率轴（batch=B·T，seq=F），双向 | `models/Falcon.py:386-401` | 形状推演 + 日志 §[9] | `[验证]` |
| A41 | `DualPathRNN(dim=3)` 沿时间轴（batch=B·F，seq=T），单向⇒因果 | 同上 | 日志 §[8]：早期 t 不变 | `[验证]` |
| A42 | 两个 `DualPathRNN` 都有 `out_proj`（48→64 / 72→64，bias=False） | `models/Falcon.py:380-383`；yml:50-65 | 日志 §[9] 打印 `Linear(48,64)`/`Linear(72,64)` | `[验证]` |
| A43 | `rnn_type` 默认 `"Comba"` 在 `torch.nn` 中不存在 ⇒ 默认构造必失败 | `models/Falcon.py:339,372` | 日志 §[9]：`AttributeError` | `[验证]` |
| A44 | `MHSA2D(causal=True)` 用 `tril` mask 只看过去帧 | `models/Falcon.py:520-524` | 日志 §[8]：早期 t `max|Δ|=0` vs `causal=False` 时 `5.1` | `[验证]` |
| A45 | `MHSA2D` 注意力在时间轴（`[B·n_head,T,T]`），频率摊平进 embedding | `models/Falcon.py:505-527` | 形状推演 | `[代码]` |
| A46 | `MHSA2D` 的 `n_freqs` 必须等于运行时 F（`γ` 形状 `[1,C,1,F]`） | `models/Falcon.py:185-188,785-786`；yml:39 | 读源码 + 模型可前向（日志 §[6]） | `[代码]`+`[验证]` |
| A47 | `AudioBlock` 4 级下采样 T=251→251→126→63→32、F=129→129→65→33→17 | `models/Falcon.py:749-776` | 形状推演（§3.3 例 2） | `[推导]`+`[验证]`（模型可跑通） |
| A48 | `AudioBlock` 只有最深级在 encoder 侧跑 TFRB；decoder 侧用 2,1,0 级 | `models/Falcon.py:837-848,868-888` | 读源码 | `[代码]` |
| A49 | `AudioBlock` 下采样/融合卷积 `causal=False` ⇒ 非因果；**逐级 lookahead（换到 L0 帧）L0 ±1 / L1 ±4 / L2 ±8 / L3 ±16**；去 cLNhw 后**单遍实测 57 帧 = 7,296 采样 ≈ 456 ms**；`:823` 是 k=1 `residual_conv`，**不能作为非因果证据**（有效证据：`:765` downsample、`:801` fusion、`:814` concat） | `models/Falcon.py:751-767,801,814` | R3 日志 §[V4]：扰动帧≥125 → AudioBlock 输出首变帧 68（57 帧）；R2 评审同值 | `[验证]` |
| A50 | `Separator` 构造时 `pop` 配置键（`repeats`/`shared`），需 deepcopy 复用 | `models/Falcon.py:991-997` | 日志 §[10] | `[验证]` |
| A51 | `Falcon` 用 `torch.stft`/`torch.istft`（hann、center=True、length=）而非 `layers` 的 STFT | `models/Falcon.py:1110,1182` | 日志 §[11] + 读源码 | `[代码]`+`[验证]` |
| A52 | Studio 在 MPS 上把 STFT/iSTFT 搬到 CPU 计算（CPU FFT bridge）；**编码器桥 = `:1111-1122`**（1110 是注释、1123 已是卷积），decoder 桥 `:1183-1198`；真实权重下 CPU 侧两副本 **bit-identical** | `studio/falcon_model/models/Falcon.py:1111-1122,1183-1198`；`look2hear/models/Falcon.py:1107-1119,1176-1190` | R3 日志 §[W1]-[W3]：两副本 `strict=True` 均 OK、输出 `max|Δ|=0.0`；训练侧无这几行 | `[验证]`+`[代码]` |
| A53 | Studio 侧 `falcon_model/layers/` 不含 `stft.py`/`stft_tfgn.py` ⇒ 无 librosa 依赖 | `studio/falcon_model/layers/__init__.py:1`（仅 docstring） | `find` + grep | `[代码]` |
| A54 | 训练侧与 studio 侧 `normalizations.py`/`activations.py` 仅注释差异 | `studio/falcon_model/layers/` | diff：normalizations 26 行全注释、activations 0 行 | `[验证]` |
| A55 | `losses/matrix.py` 是第三套独立 STFT 实现（供 MultistageLoss） | `look2hear/losses/matrix.py:13,59,114` | 读源码 | `[代码]` |
| A56 | `STFTEncoder` 参数 3,024、`STFTDecoder` 2,016（均无 bias） | `models/Falcon.py:1080-1091,1149-1157` | 日志 §[6] 打印 | `[验证]` |
| A57 | `video_bottleneck`（kernel_size=-1）的 `conv` 是 `nn.Identity`，0 参数；实际走 **`causal=False` 分支 → `Falcon.py:84-85`**（R1 误引 `66-68` 的 `causal=True` 分支） | `models/Falcon.py:30,84-85,1312`；yml:16-18 | 日志 §[6]：`参数=0`、`conv=Identity`；R3 §[V5] 行号核对 | `[验证]` |
| A58 | 音频路径子模块参数合计 523,113；`FalconSpeakerContrastive` 另加 spk 头 14,584，合计 537,697（“0.5M trainable”口径） | `Falcon.py:1308-1336`；`Falcon_Contrastive.py:11-50` | 日志 §[6] 打印 523,113；与 Lead `lead-model-params.log` 的 537,697 相差 14,584 | `[验证]` |

| A59 | **encoder→decoder 串联**的采样级过去泄漏 = `p mod 128`（解析上界 ≤127 采样）；实测 `p=16000(r=0)→首变 16001`（无泄漏）、`p=20000(r=32)→19969`（泄漏 31）等六点全部符合 `r−1`（hann 首点为 0） | `Falcon.py:1110-1116,1170-1177,1182-1188` | R3 日志 §[V2] | `[验证]` |
| A60 | cLNhw 的泄漏数字强依赖探针协议；且**两类抵消机理**成立：(i) `CumulateLN4D` 抵消 GN 的纯尺度泄漏（GN-only ×10 `3.93` vs cLNhw ×10 `1.44e-03`）；(ii) common-mode（整段全通道同量）被 LN-over-C 抵消（+5 → `7.15e-07`），单通道 +5 → `5.08` | `normalizations.py:134-143,165-202,204-221` | R3 日志 §[V3a-d] + R2 评审 | `[验证]` |
| A61 | `import look2hear.layers` 会经 `utils/__init__.py:12 → lightning_utils.py:3` **顶层**加载 `pytorch_lightning`（不是 `BaseModel.serialize` 的延迟导入） | `layers/__init__.py:20`；`stft_tfgn.py:12`；`utils/__init__.py:12`；`utils/lightning_utils.py:3` | R3 日志 §[V7]：`sys.modules['pytorch_lightning']` 存在 | `[验证]` |
| A62 | 论文配置下 `LayerNormalizationHW`(=cLNhw) 实例 **30 个**（对应 30 GlobalLayerNorm + 30 CumulateLN4D），另有 60 个 `LayerNormalization4D` | `Falcon.py:1308-1336` + `normalizations.py:204-221` | R3 日志 §[V7] | `[验证]` |
| A63 | 真实权重（`falcon.pt`，epoch 714）下，训练副本与 Studio 副本 **`strict=True` 均加载成功**（各 525 key、537,697 参数），同输入 CPU 输出 **`max|Δ|=0.000e+00`**；Studio 副本 import 时**不加载** librosa/torch_complex/look2hear | `studio/falcon_model/models/Falcon.py:1111-1122,1183-1198`；PROVENANCE:33 | R3 日志 §[W1][W2] | `[验证]` |
| A64 | `interpolate(mode="nearest")` 是单调映射（输出 t 只取 `⌊t·ratio⌋` 的过去/当前源）⇒ **本身因果**，不是 `InjectionMultiSum` 非因果的致因；非因果来自 k=5 对称卷积 | `Falcon.py:319-325,801,814` | 静态推演；[验证] 去 cLNhw 后脉冲响应 `[98,102]`（§8-[V4]） | `[代码]` |

> A58 说明：本脚本只统计 `encoder+audio_bottleneck+separator+mask_generator+decoder = 523,113`；
> `FalconSpeakerContrastive` 另有 speaker pooling 头（`Falcon_Contrastive.py:11-50`）约 14,584 参数，
> 合计 **537,697**，与 Lead 的 `docs/verification/runs/lead-model-params.log` 一致。

## 7. 未解疑问 / 需其它子系统确认的问题

| 编号 | 问题 | 需要谁确认 |
|---|---|---|
| Q1 | 训练侧 `causal: true` 与 `AudioBlock` 中 `causal=False` 的下采样/融合卷积、`cLNhw` 的全窗口统计**明显矛盾**：论文的 causal 主张是按"帧级因果"还是"窗口内因果"定义？是否在论文/附录中明确？ | Lead / model-core（读论文正文） |
| Q2 | `STFTDecoder` 因果路径输出 252 帧而编码器 251 帧（多 1 帧靠 `length=` 截断）是刻意设计还是 off-by-one？对非 32000 长度（如 Studio 边界窗口）是否始终安全？ | model-core / studio-backend |
| Q3 | `STFTEncoder` 用 `torch.stft` 默认 `center=True` + `pad_mode='reflect'`，在**窗口化推理**的窗口边界会产生反射伪影；Studio 的 80 ms 右上下文/guard 是否正是为此？是否评估过改用 `center=False` + 缓存？ | studio-backend / model-core |
| Q4 | `layers/stft.py` 的 `distutils`（Py3.12 移除）与 `torch.stft(return_complex=False)`（未来移除）都是**训练代码的导入/运行炸弹**；仓库 `requirements.txt` 未固定 Python/torch 上限，是否有意保留？ | ops-tests-docs / train-sys |
| Q5 | `layers/rnnlayers.py:882` `DPRNNLinear.forward` 的 `transpose(1,1)` no-op 与形状不匹配：是否存在**从未被测试**的旧模型（如 DPRNN 基线）使用它？ | loss-metric-eval / Lead |
| Q6 | `layers/stft.py` 与 `utils/stft.py` 既然是同源复制，哪份是"原始"、哪份是"复制"？（git 历史只有一个 commit，无法区分；从 import 关系看 `utils` 版本被 `utils/__init__` 当公共 API，`layers` 版本只导出两个函数 ⇒ 推测 `utils` 较早） | Lead（或查上游 look2hear 仓库） |
| Q7 | `MHSA2D` 的 `n_freqs` 依赖 `AudioBlock` 里手工递推的 `current_f`（`:774`），若修改 downsampling 结构（kernel/stride）不同步就会有静默错误（广播失败）。是否有测试覆盖？ | model-core / ops-tests-docs |
| Q8 | `stft_tfgn.Stft` 与 `complex_utils` 是 ESPnet 死代码；是否可以在发布版从 `layers/__init__.py` 移除以免安装 librosa/torch_complex/espnet？studio 已经这么做了，训练侧为何不？ | train-sys / Lead |
| Q9 | `losses/matrix.py` 的第三套 STFT 用 `hann^(1/2)` + 特定缩放实现"完美重构"，与 `layers/stft.py` 的 `sqrthann` 常数是否一致？（影响 MultistageLoss 的尺度） | loss-metric-eval |
| Q10 | Studio 的 CPU FFT bridge 在 MPS 上引入 CPU↔GPU 拷贝，实测带宽开销多少？PROVENANCE 只声明 CPU 上一致性，没有 MPS 性能数据。 | studio-backend |

## 8. 本轮验证活动记录

**环境**：`/root/.venvs/falcon/bin/python` = Python 3.10.18 + torch 2.8.0+cu128（CPU 运行）；
stub `PYTHONPATH=$PWD/docs/verification/stubs:$PWD`（timm.DropPath、typeguard.check_argument_types）。
**本轮未修改任何源码**（`git status` 仅新增 `docs/`）。

### 8.1 R1 轮（初版证据）

- **脚本**：`docs/verification/scripts/verify_layers_dsp.py`（**403 行**）
- **日志**：`docs/verification/runs/layers-dsp-verify.log`（**181 行**）

```bash
cd /root/code/github_repos/Falcon-fork
diff -u look2hear/layers/stft.py look2hear/utils/stft.py            # 15 行 / 2 hunk
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python docs/verification/scripts/verify_layers_dsp.py \
  | tee docs/verification/runs/layers-dsp-verify.log
```

| 小节 | 验证内容 | 结果摘要 |
|---|---|---|
| §[0] | 导入链 | `import look2hear.models` 后 `sys.modules` 含 `look2hear.layers.stft`、`look2hear.utils.stft`、`librosa`、`torch_complex`、`typeguard`、`timm` |
| §[1] | `forward_stft` vs `torch.stft` | center T/F：`2.289e-05`；`mode="torch"`：`0.000e+00`；round_pow_of_two 行为符合公式 |
| §[2] | `inverse_stft` vs `torch.istft` | `2.146e-06`；round-trip `2.384e-06`；`num_frames(32000)=251`；`num_bins=129`；`K.shape=(512,1,256)` |
| §[3] | STFT lookahead | 脉冲 p=20000 → 受影响帧 `[156,157]`，与理论一致 ⇒ 帧级 ≤8 ms |
| §[4] | 归一化语义 | `GN(1,C)`/`CumulateLN4D`/`LN4D((C,1))`/`LN4D((C,F))`/`cLN` 全部与手算一致；`get("BatchNorm1d")→nn.BatchNorm1d` |
| §[5] | 激活解析 | 大写名走 torch 原生；小写名走本地函数，实例等价 |
| §[6] | 端到端形状 | `[1,32000]` → encoder `(1,112,251,129)` → bottleneck `(1,112,251,129)` → separator `(1,112,251,129)` → mask `(1,1,112,251,129)` → decoder `(1,1,32000)`；子模块参数 3,024/13,104/492,312/12,657/2,016 |
| §[6b] | 解码器帧账本 | `251 → pad 253 → convT 253 → crop 252 → istft(length=32000)` |
| §[7] | 全链因果性 | encoder 帧 `<120` 为 `0.000e+00`（R3 已指出：该证据只覆盖 conv 子层，**不能**推出 encoder 整体严格因果）；bottleneck 早期 `4.7e-05`（非 0）；separator 早期 `1.7`；最终波形 `[0:8000]` 段 `9.7e-04`、`[16000:]` 段 `6.6e+00` |
| §[7b] | 解码器因果性 | `<15360` 采样严格 0；`[15360:16000)` 起 `6.3e-01`（iSTFT OLA 扩散；R3 已由 §[V2] 统一为串联口径） |
| §[8] | 子模块因果性 | `DualPathRNN(dim4/dim3)`、`MHSA2D(causal=True)` 早期帧 `0.000e+00`；`MHSA2D(causal=False)` 早期 `5.1`；`cLNhw` 早期 `1.2e-03`（**仅 T=8 尖峰探针，不代表 2 s 窗口**，见 R3 §[V3]） |
| §[9] | `DualPathRNN` 参数 | `out_proj`：dim4 `Linear(48,64)`、dim3 `Linear(72,64)`；默认 `rnn_type="Comba"` → `AttributeError` |
| §[10] | 配置破坏性 | 构造后 `audio_params` 丢 `repeats`/`shared`，`video_params` 同理 ⇒ 必须 deepcopy |
| §[11] | 编解码器归属 | `model.encoder.__module__ == "look2hear.models.Falcon"`，非 `layers.enc_dec` |

### 8.2 R3 轮（回应评审的补充验证）

- **脚本**：`docs/verification/scripts/verify_r3_layers_revision.py`（216 行）→ 日志 `runs/r3-layers-revision.log`（92 行）
- **脚本**：`docs/verification/scripts/verify_r3_studio_weights.py`（87 行）→ 日志 `runs/r3-studio-weights.log`（40 行）

```bash
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r3_layers_revision.py \
  > docs/verification/runs/r3-layers-revision.log 2>&1
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r3_studio_weights.py \
  > docs/verification/runs/r3-studio-weights.log 2>&1
```

| 小节 | 验证内容 | 结果摘要（完整见日志） |
|---|---|---|
| §[V1] | A36′：encoder 帧级 lookahead | `s≥16000 → 首变帧 125`（124 为 0）；`s≥20000 → 156`；`s≥16255 → 126`；`s≥16256 → 127` |
| §[V2] | encoder→decoder 串联采样级 | `p=16000(r=0)→16001`；`p=20000(r=32)→19969`（泄漏 31）；`p=20032(r=64)→19969`（63）；`p=20127(r=31)→20097`（30）；`p=24000(r=64)→23937`（63）— 六点全符合 `r−1` |
| §[V3] | A19 协议表 + 抵消机理 | 单帧全通道尖峰+20（T=8）`1.165e-03`；单帧单通道+20 `2.07e-05`；整段全通道+5（common-mode）`7.15e-07`；整段单通道+5 `5.084`；波形×10 → pre_norm `8.49e-05`/bottleneck `3.83e-05`；同分布替换 → `1.43e-06`/`7.15e-07`；GN-only ×10 `3.930` vs cLNhw ×10 `1.435e-03` |
| §[V4] | A49 逐级与 57 帧 | 各级形状 `[(251,129),(126,65),(63,33),(32,17)]`；去 cLNhw 后扰动帧≥125 → 输出首变帧 68 ⇒ **57 帧 = 7,296 采样 = 456 ms** |
| §[V4b] | VideoBlock 因果性 | `eval` 左段 t<30 = `0.000e+00`；`train` 左段 = `4.533e-02`（`BatchNorm1d` 批内统计） |
| §[V4c] | FRCNN 时间感受野 | 改帧 ≥30 → 最早受影响输出帧 **28**（`Conv3d k=5,p=2` ⇒ ±2 帧 = ±80 ms） |
| §[V5] | A57 行号 | `Falcon.py:30` 退化 out_chan；`:67` 是 causal=True 分支 Identity；**`:85` 是 causal=False 分支 Identity（实际使用）** |
| §[V6] | §2.1 名字计数 | `cnnlayers` **13**、`rnnlayers` **5**（AST） |
| §[V7] | 依赖链 + 实例密度 | PL 经 `utils/__init__.py:12 → lightning_utils.py:3` 顶层加载（`sys.modules` 确认）；`cLNhw` 30 个、`GlobalLayerNorm` 30、`CumulateLN4D` 30、`LayerNormalization4D` 60 |
| §[W1] | Studio 副本 import 面 + 真实权重 | `librosa/torch_complex/look2hear` 均未加载；`strict=True` OK；525 keys、537,697 参数；forward `(1,1,32000)` |
| §[W2] | 训练 vs Studio（真实权重） | 两侧 `strict=True` 均 OK；**输出 `max|Δ|=0.000e+00`**；state_dict key 完全一致 |
| §[W3] | Studio CPU FFT bridge 行号 | encoder 桥 `1111/1112/1117/1122`；decoder 桥 `1184-1186/1194/1198` |

**失败/修正记录**：

1. 首轮 R1 脚本在对比 `forward_stft` 与 `torch.stft` 时把 `ref` 多转置了一次（形状 `251 vs 129` 报错）→ 脚本 bug，已修。
2. `models.DualPathRNN` 不存在于 `look2hear.models` 命名空间（`models/__init__.py:1-2` 只导出两个类）→ 改从 `models.Falcon` 导入。
3. **R3 更正 R1 的诊断**：cLNhw 首版探针只测到 `7e-07` 不是"GN 尺度不变的巧合"单一原因，而是
   **`CumulateLN4D` 抵消纯尺度泄漏 + common-mode 抵消**两个机理（§[V3]；两份评审均指出）。
4. `Separator` 配置破坏性探针首版复用了被前面模型构造 `pop` 过的 `aconf` → 改为重新 `yaml.safe_load`。
5. **R3 新错误**（已在文档中修正）：R1 的 `STFTEncoder` "严格因果"是**错误结论**（P0，只测了前 120 帧、恰避开帧 125）；
   R1 的"采样级 16 ms/256 采样"也是**错误推导**（漏算 decoder 3 帧 lookback 的吸收）。两处均在 §4.1/§4.2/A36′/A38/A59 重写。
6. **未做**：MPS 设备实测（沙箱无 Apple GPU，Studio bridge 只做静态核对 + 真实权重 CPU bit-identical 验证）；
   训练/评测端到端（无数据、无训练需求）。

## 9. 变更日志与评审响应

### 9.1 R3 评审响应表（P0=1 / P1=4 / P2=7 + R2 遗留）

> 评审来源：`docs/reviews/round1/02-layers-dsp-review-by-model-core.md`（R2，P0×1/P1×4/P2×7）
> 与 `docs/reviews/round2/02-layers-dsp-round2-review-by-train-sys.md`（R4 复核：R3 未发生→阻塞；执行级验证 15 项）。
> 处置：**采纳 / 部分采纳 / 不采纳** 逐条列明。

| 评审项 | 处置 | 本版落点 |
|---|---|---|
| **P0-1** A36「`STFTEncoder` 严格时间因果」错误 | **采纳** | A36 改为 **A36′**（conv 子层严格因果 + `torch.stft(center=True)` ≤128 采样/8 ms 帧级 lookahead）；§4.2 新增"`STFTEncoder` 整体"行；删去"前 120 帧 Δ=0"作为充分证据的写法，改为帧 124=0/帧 125 非零（§8-[V1]） |
| **P1-①** 采样级 lookahead 256 采样/16 ms 错误 | **采纳** | §4.1 重写为"帧级 8 ms；encoder→decoder 串联过去泄漏 `= p mod 128 ≤127`（实测 p=20000→31）"；新增 A59；§4.2 `torch.istft` 行改为"被 decoder 3 帧 lookback 吸收" |
| **P1-②** A19 泄漏数字缺探针条件 + 缺 CumulateLN4D 抵消机理 | **采纳** | A19 改为"协议表 + 两机理"；§4.2 新增"为什么 GN 全窗口却常只测到 1e-6"段（CumulateLN4D 抵消 + common-mode）；新增 A60；§8.2 §[V3] 给 7 组协议数字 |
| **P1-③** A49「±8 帧」低估、`:823` 非有效证据 | **采纳** | A49 改为逐级 **L0 ±1 / L1 ±4 / L2 ±8 / L3 ±16** + 单遍 **57 帧=7,296 采样≈456 ms**；明确 `:823` 是 k=1 `residual_conv` **不能作证据**，有效证据为 `:765/:801/:814`；§4.2 表同步 |
| **P1-④** A57 行号指向 causal=True 分支 | **采纳** | A57 证据改为 `Falcon.py:30,84-85,1312`（实际走 `causal=False` 的 `:85`）；§8-[V5] 逐行核对 |
| P2-1 A03/A05 "唯一/全仓"未限定产品代码 | **采纳** | A03/A05/§2.12 表加"限产品代码；`docs/verification/scripts/` 自检脚本除外" |
| P2-2 §2.1 "15/6 个名字"计数错 | **采纳** | 改为 **13/5**（§8-[V6] AST 计数） |
| P2-3 §4.7 "import 拉入 PL"归因错（"仅 serialize 延迟导入"） | **采纳（归因改、结论保留）** | §4.7 改为经 `utils/__init__.py:12 → lightning_utils.py:3` 顶层加载；明确 `BaseModel.serialize` 是另一处延迟导入；新增 A61 + §8-[V7]。**未采纳 R2 评审"不会拉入 PL"的否定判断**（该否定与实测不符，见 R2 评审 N1） |
| P2-4 把 nearest 插值列为非因果致因 | **采纳** | §4.2 表与 §5.1 改为"nearest 单调 ⇒ 因果；非因果来自 k=5 对称卷积"；新增 A64 |
| P2-5 A52 编码器桥行号 off-by-one | **采纳** | 改为 **`:1111-1122`**（1110 注释/1123 卷积），§4.6/A52/§8-[W3] 同步；并用真实权重做 CPU bit-identical 验证（A63） |
| P2-6 §8 日志/脚本行数不符 | **采纳** | §8 重写为 R1（脚本 403 行/日志 181 行）+ R3（190/80、87/40 行）两小节，行数与 `wc -l` 一致 |
| P2-7 遗漏 VideoBlock(train) BN 非因果与 FRCNN ±80 ms | **采纳** | §4.2 表新增两行（视频分支 train/eval、FRCNN `±2帧=±80ms`），交叉引用 `04-losses-metrics-video.md`；§8-[V4b] 实测 eval=0 / train=4.8e-02、FRCNN 首变帧 28 |
| R2-N1 PL 归因双向修正（R3 不可照抄 R2 否定） | **采纳** | 见 P2-3：保留"会拉入"、修正归因，并在 §4.7 注明 R2 否定不成立 |
| R4 E15 全模型因果基线（首变采样 9612） | **部分采纳** | 该现象属 cLNhw 全局统计的下游表现，本版 §4.2 结论与之一致；未复跑 Lead 的 `verify_causality.py`（脚本不在本域，避免重复） |
| R2 评审 C1–C8 跨文档建议 | **采纳** | C1/C2/C4 见上；C3 用协议表（§4.2/A19）；C6 TFRB/GCI 映射保留（§1）；C7 80 ms 与 FRCNN 关系在 §4.2 交叉引用；C8 normalizations 26 行注释 diff 保留（A54） |
| 01-R3 的"窗口级因果（window-causal）"统一表述 + §4.6-⑨ 五协议表 | **采纳** | §4.2 开头采用与 `01-model-core.md`（R3, 975 行, sha `afe00188…db0058`）一致的表述与三段论（STFT 8 ms / 对称卷积 456 ms / cLNhw 无界）；cLNhw 数字**全部保留并标注扰动协议**，与 01-R3 §4.6-⑨ 的五协议表口径对齐（单帧尖峰 T=8 / 波形×10 / 同分布替换，另加单通道 vs 全通道、T 长度两个维度），不在 FINAL 中只留单一数字 |

**仍存分歧（明确记录，供 FINAL 裁定）**：

1. **FRCNN ±80 ms 的归属**：本版按 R2 建议列入 §4.2 表并交叉引用 04 域，但视角焦视觉前端最终结论以 `04-losses-metrics-video.md` 为准。
2. **cLNhw 单层泄漏"代表值"**：本版给出"协议表"而非单一数字；若 FINAL 必须给一个数，建议用
   "T=251/波形×10 协议下 `audio_bottleneck` 左段 ≈`3.8e-05`（本版）/`3.2e-05`（model-core R2 复算）/`4.7e-05`（R1 首测）"，量级一致。
3. **A49/57 帧的 `±1` 量化**：扰动分界不同会得到 57 或 58 帧（nearest 插值对齐所致），本版按 R2 的 split=125 口径写 57 帧。

### 9.2 版本锚点

- R1 版：757 行，sha256 `ae01042f95cef3e95961027b308b5e610a60f2b97b17dce0a7fad279a31bbd2f`（被两份评审锚定）。
- **R3 版**：**895 行**；sha256 见 `docs/verification/r3-layers-dsp-anchor.md`（文件自哈希无法内嵌于自身，故外置记录）。
- `01-model-core.md` R3 版：975 行，sha256 `afe00188097e4d7d5ada48dbcdd96703248ea7942233f3b699658e48d2db0058`（用于跨文档对齐）。

### 9.3 Changelog

| 日期 | 轮次 | 变更 | 采纳的评审意见 |
|---|---|---|---|
| 2026-10-21 | R1 | 初稿：8 节模板 + 58 条断言表 + 4 个形状推演 + diff 结论 + 静态/动态验证 | —（待 R2 评审） |
| 2026-10-21 | **R3** | ① P0 修正：A36→A36′（STFTEncoder 非严格因果，conv 子层因果 + 8 ms 帧级 lookahead），§4.2 表格新增"整体"行与三 causal 来源说明；② §4.1 采样级 lookahead 从 256 采样/16 ms 重写为 ≤127 采样（实测 p=20000→31），新增 A59；③ A19 改协议表 + 新增两抵消机理段（CumulateLN4D/common-mode）、A60；④ A49 改逐级 L0±1/L1±4/L2±8/L3±16 + 57 帧、剔除 `:823` 证据；⑤ A57 行号改 `:84-85`；⑥ §4.7 PL 归因改为 `utils/__init__→lightning_utils`；⑦ §2.1 计数 15/6→13/5；⑧ §4.2 补 VideoBlock(train) BN 与 FRCNN ±80 ms、nearest 因果性；⑨ 真实权重验证两副本 bit-identical（A63）；⑩ 统一"窗口级因果（window-causal）"表述；⑪ 断言 58→64 条（新增 A59~A64）；⑫ §8 补 R3 验证记录、§9.1 响应表/§9.2 版本锚点 | model-core R2（P0×1/P1×4/P2×7）+ train-sys R4 复核（N1/N2/E1-E15）+ Lead R3 指令 |
