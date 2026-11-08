# 02 layers + DSP 算子族（W1-02）

> 范围：`look2hear/layers/*`（8 文件 3644 L）+ `look2hear/utils/{stft,complex_utils,attn,inversible_interface,separator,nets_utils,torch_utils,get_layer_from_string}.py`。
> 仓库 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork`，HEAD `6c1d5b7`，只读，CPU 实测，`falcon` conda 环境（Py 3.11.17 / torch 2.8.0+cu128）。
> 全部结论带 `path:line`；实测数字来自本报告 §4 的 12 个脚本（`/tmp/w102/exp*.py`，不落仓库）。
> 论文 arXiv:2610.32749 已于 2026-10-23 公开。本报告 2026-11-08 经 Lead 复核，更正 MACs 口径与 lookahead 三档两处。

## 0. 结论速览

1. **`look2hear/layers/` 里 45 个算子类，Falcon 实际只用到 6 个**：`GlobalLayerNorm`、`CumulateLN4D`、`LayerNormalizationHW`、`LayerNormalization4D`、`normalizations.get`、`activations.get`。其余（cnnlayers 13、rnnlayers 15、enc_dec 6、stft 3+8 函数、stft_tfgn 1、utils/attn 1）**全部零调用**，是 Conv-TasNet/FRCNN/Asteroid/ESPnet 的遗物。
2. **Falcon 的 DSP 不在 `layers/`，在 `models/Falcon.py`**：`STFTEncoder`(:1061) / `STFTDecoder`(:1130) 直接用 `torch.stft`/`torch.istft`，`win=256, hop=128, center=True, window=hann, normalized=False, onesided=True` → 251 帧 × 129 bin。
3. **`layers/stft.py` 的卷积式 STFT 与 `torch.stft` 等价到 `max|Δ|=2.289e-05`**（fp32 噪声级，8 组 win/hop/center 全同值）；但它是死代码，且输出布局是 `(B,F,T,2)`，与 `stft_tfgn.Stft` 的 `(B,T,F,2)` **相反**——混用必炸。
4. **`layers/stft.py` 往返重建**：`center=True` 时 `max|Δ|=1.907e-06`（SI-SNR 128.5 dB）；`center=False` 时 `max|Δ|=1.126`（SI-SNR 31.8 dB）——边缘帧 OLA 覆盖不全，**默认 `center=False` 的组合不是正交变换**。
5. **`cLNhw` = `GlobalLayerNorm` ∘ `CumulateLN4D`**：前者统计轴是 `(C,T,F)` 全窗，后者沿 `T` 累积、统计只跨 `C`。实测 `cLNhw(x) == CumulateLN4D(GlobalLayerNorm(x))`，`max|Δ|=3.6e-07`。
6. **`cLNhw` 的 `GlobalLayerNorm` 在初始化态是数学空操作**：`|cLNhw(x) − CumulateLN4D(x)|max = 3.576e-07`（被后者的归一化吸收）；把 GLN 的 affine 随机化后差异跳到 `1.232`。即 GLN 段唯一的实际内容是那 2×C 个仿射参数。
7. **`cLNhw` 彻底破坏时间因果**：扰动 `t=125` 一帧，`AudioBlock` 输出 251/251 帧全变（T=251 时 lookback/lookahead 都=全窗）；换成因果归一化（`cLN4D`）后变成 lookback **57 帧（单遍）**、lookahead 0。`configs/LRS2_2Mix_Paper_v2.yml:30 causal: true` 只是**窗口级**承诺。
   lookahead 需分三档写：**单遍几何 57 帧 = 456 ms**（即上面这个 `cLN4D` 替换后的卷积感受野）、
   **整链几何 113 帧 = 904 ms**（编码器阶梯 15 帧 + 57 + 第二遍 56；另一路 110 帧 / 880 ms，差约 3%）、
   **as-shipped 窗尾 ≤2000 ms**（含 `cLNhw` 时全窗耦合）。57 帧不能单独当整链延迟用。
   `Falcon.py:772/808/821` 三处硬编码 `causal=False` 是整链 lookahead 的卷积来源（11 §5、12 §6）。
8. **本仓库没有任何流式/缓存状态**：`cache_hat`/`cache_prev`/`self.cache` 全仓 0 命中，`InversibleInterface` 只被死代码 `Stft` 实现，`Stft` 从未被实例化。
9. **`stft_tfgn.py` 里没有 TF-GN**（整个仓库都没有 TF-GN/TFGroupNorm）；该文件唯一的实质贡献是把 `typeguard.check_argument_types()` 钉进 `layers` 包的 import 链（:8,:34），从而强制 `typeguard<3`。
10. **复数算子只有一种真实形态**：`MaskGenerator.__apply_masks`(:1251) 的 RI 复数乘法；没有 complex conv / complex layernorm / modReLU。且 `mask_act=ReLU` 使复数掩码的实部虚部**同时非负**→ 每 bin 相位被限死在 [0°,90°]（实测 mask min=0.0000）。
11. **`rsync layers` 里两个真 bug（死代码，不影响 Falcon）**：`DPRNNLinear.forward` 的 `output.transpose(1, 1)`(:882) 是 no-op，导致 `nn.Linear` 维度错，**任何输入都崩**；`Bottomup_Concat_Topdown_TCN` 缺 interpolate，`cat` 维度不匹配，**构造即坏**。
12. **`VideoBlock` 的 LSTM 布局错误**：`x_enc.transpose(1,2)` 得到 `(B,T,C)` 喂给 `batch_first=False` 的 LSTM(:952) → 序列长度被解释成 1、batch=50，**时间建模完全失效**（脉冲响应实测只有 1/50 帧变化）。
13. **规模核验**：`LRS2_2Mix_Paper_v2` 配方实例化 **537,697 参数**（与 `Falcon.py:990` 注释一致）。MACs 要分两把尺子：
   ptflops 口径 **10.611 GMAC**（含 LSTM、不含视觉，与论文 10.6 G 同尺）；`torch.utils.flop_counter` 口径
   **5.382 GMAC / 10.763 GFLOPs**，因为它不记 `aten::lstm`。两个数各自成立，不能并列——2026-11-08 Lead 复核更正：
   此前"5.382 G 是真 MACs、论文把 FLOPs 写成 MACs"的判断错误，ptflops 的 conv hook 每个乘加记 1，返回值就是 MACs。

## 1. 事实基线

### 1.1 范围内文件（行数已核）

| 文件 | 行 | 在 Falcon 前向中的角色 |
|---|---|---|
| `look2hear/layers/__init__.py` | 50 | 把 8 个模块全量 re-export（:1-20） |
| `look2hear/layers/normalizations.py` | 245 | **被使用**：`cLNhw`/`LN4D` 及 `get()` |
| `look2hear/layers/activations.py` | 67 | **被使用**：`get()` 字符串→激活 |
| `look2hear/layers/stft.py` | 785 | 死代码（`forward_stft`/`inverse_stft`/`STFT`/`iSTFT`） |
| `look2hear/layers/stft_tfgn.py` | 244 | 死代码（`Stft`），但垄断 `typeguard` 依赖 |
| `look2hear/layers/cnnlayers.py` | 884 | 死代码（Conv-TasNet/FRCNN 块） |
| `look2hear/layers/rnnlayers.py` | 908 | 死代码（Asteroid DPRNN/Conformer） |
| `look2hear/layers/enc_dec.py` | 461 | 死代码（Asteroid filterbank） |
| `look2hear/utils/stft.py` | 785 | 与 `layers/stft.py` **不同文件**（md5 `796ceb06…` vs `533c481a…`），同为死代码 |
| `look2hear/utils/complex_utils.py` | 183 | 全仓无调用（`einsum/matmul/solve` 包装 `torch_complex`） |
| `look2hear/utils/attn.py` | 28 | `CrossAttention`，全仓无调用 |
| `look2hear/utils/inversible_interface.py` | 12 | 仅 `Stft` 实现 |
| `look2hear/utils/separator.py` | 69 | `separate()/torch_separate()`，全仓无调用 |
| `look2hear/utils/nets_utils.py` | 502 | 仅 `make_pad_mask` 被死代码 `stft_tfgn` 用（:12） |
| `look2hear/utils/torch_utils.py` | 42 | 仅 `tensors_to_device` 被 `audio_visual_test.py:67` 用 |
| `look2hear/utils/get_layer_from_string.py` | 42 | 全仓无调用 |

### 1.2 唯一真实入口

```
look2hear/models/Falcon.py:9    from ..layers import normalizations, activations
```
这一行**触发 `look2hear/layers/__init__.py` 全量导入**（:1-20），于是 `librosa`、`torch_complex`、`typeguard`、`distutils.version`、`soundfile` 全部进入 `import look2hear.models` 的依赖闭包——而其中 90% 的算子永远不执行。实测这 6 个依赖在 `falcon` 环境全部可用（§4 E7b）。

`normalizations.get` / `activations.get` 在 `Falcon.py` 里出现 4 处构造点（:89,:91,:188,:372）+ `AudioBlock` 内部经 `ConvNormAct` 间接使用。

### 1.3 实测环境

```
torch 2.8.0+cu128 ; typeguard 2.13.3 ; librosa / torch_complex / soundfile / distutils.version 均 OK
CUDA_VISIBLE_DEVICES= (空)  PYTHONPATH=/root/code/github_repos/Falcon-fork
/root/miniforge3/envs/falcon/bin/python
```

## 2. 机制拆解

### 2.1 算子族全清单（每类一行，参数量为实测）

约定：参数量 = 用「该类默认构造参数」或「Falcon 生产配置」实例化后 `sum(p.numel())`；`因果?` 指**该类自身设计**的时间因果性（不涉及全模型）。

#### 2.1.1 `layers/cnnlayers.py`（13 类，全部死代码）

| 类 | file:line | 输入 → 输出 | 参数（实测） | 因果? | Falcon 用? |
|---|---|---|---|---|---|
| `_Chop1d` | :7-15 | `x[..., :-p]` | 0 | 造因果 | 否 |
| `Conv1DBlock` | :18-63 | `(2,128,50)→(2,128,50)` | 201,474 | `causal=True` 时是（:43 `_Chop1d`） | 否 |
| `ConvNormAct` | :66-101 | `(2,128,50)→(2,512,50)` | 198,145 | 否（对称 pad） | **否**（Falcon 用自己的同名类 `Falcon.py:11`） |
| `ConvNorm` | :104-131 | 同上 | 198,144 | 否 | 否 |
| `NormAct` | :134-152 | 形状不变 | 257 | 否 | 否 |
| `Video1DConv` | :155-220 | `(2,128,50)→(2,512,50)`（返回 `(skip,y)`） | 132,608 | 否（`padding=(d(k-1))//2` 对称，:189） | 否 |
| `Concat` | :223-239 | `(2,128,50)+(2,64,25)→(2,512,50)` | 98,817 | 否 | 否 |
| `FRCNNBlock` | :242-389 | `(2,128,50)→(2,128,50)` | 3,839,110 | 否 | 否 |
| `Bottomup` | :392-451 | 返回 `(residual, deepest, list)` | 83,457 | 否 | 否 |
| `BottomupTCN` | :454-495 | 同上 | 2,179,585 | 否 | 否 |
| `Bottomup_Concat_Topdown` | :498-595 | `(res, list[4], top)→(2,128,50)` | 4,804,229 | 否 | 否 |
| `Bottomup_Concat_Topdown_TCN` | :598-680 | **构造即坏**（缺 interpolate，§5-6） | — | 否 | 否 |
| `FRCNNBlockTCN` | :683-795 | `(2,128,50)→(2,128,50)` | 5,922,950 | 否 | 否 |
| `TAC` | :798-884 | `(2,2,128,10,5)` 形状不变 | 296,067 | 否（跨麦平均，:852） | 否 |

用途一句话：`Conv1DBlock/ConvNormAct/ConvNorm/NormAct` = Conv-TasNet 的 TCN 积木；`Video1DConv/Concat/FRCNNBlock*/Bottomup*` = FRCNN 视频前端的上/下采样金字塔；`TAC` = FasNet-TAC 的跨通道通信块（Asteroid 系）。

#### 2.1.2 `layers/rnnlayers.py`（15 类 + 2 辅助，全部死代码）

| 类 | file:line | 输入 → 输出 | 参数（实测） | 因果? | Falcon 用? |
|---|---|---|---|---|---|
| `has_arg` | :15-31 | 反射工具 | 0 | — | 否 |
| `SingleRNN` | :34-73 | `(2,50,128)→(2,50,128)` | 132,096 | 由 `bidirectional` 决定；`forward` 里每次调 `flatten_parameters()`(:70) | 否 |
| `LSTMBlockTF` | :76-105 | `(2,128,50)→(2,128,50)` | 297,344 | 默认 `bidirectional=True` → **否** | 否 |
| `Linear` | :109-123 | xavier 初始化包装 | — | — | 否 |
| `Swish` | :126-136 | `x·σ(x)` | 0 | — | 否 |
| `Transpose` | :139-147 | 维度置换 | 0 | — | 否 |
| `GLU` | :150-162 | `chunk(2,dim)·σ` | 0 | — | 否 |
| `FeedForwardModule` | :165-180 | `(2,10,128)` 不变 | 131,968 | 否 | 否 |
| `PositionalEncoding` | :183-206 | `length→(1,L,d)` | 0（buffer） | — | 否 |
| `RelativeMultiHeadAttention` | :209-307 | 4×`(2,10,128)`→`(2,10,128)` | 82,688 | 否 | 否 |
| `MultiHeadedSelfAttentionModule` | :310-359 | 同上 | 82,944 | 默认 `is_casual=True` → 上三角 mask(:345-352) 是 | 否 |
| `ResidualConnectionModule` | :362-377 | 形状不变 | 0 | — | 否 |
| `DepthwiseConv1d` | :380-430 | `(2,128,50)` 不变 | 3,968 | `is_casual=True` 时 `pad=k-1` 再裁尾(:411-429) 是 | 否 |
| `PointwiseConv1d` | :433-468 | 同上 | 16,512 | 是（k=1） | 否 |
| `ConformerConvModule` | :471-522 | `(2,10,128)` 不变 | 54,016 | `is_casual` 透传 | 否 |
| `TransformerLayer` | :525-555 | 同上 | 401,152 | 透传 | 否 |
| `TransformerBlockTF` | :558-580 | `(2,128,50)` 不变（B,F,T 语义） | 401,408 | 透传 | 否 |
| `DPRNNBlock` | :586-637 | `(2,128,10,5)` 不变 | 594,688 | intra 双向(`:604`) → 否 | 否 |
| `DPRNN` | :640-771 | `(2,128,50)→(2,2,128,50)` | 3,667,329 | 否 | 否 |
| `DPRNNLinear` | :774-908 | **任何输入都崩**（:882） | — | 否 | 否 |

⚠ 命名陷阱：`rnnlayers.DPRNN` 与 `Falcon.py:333` 的 `DualPathRNN` **同名不同实现**，互不引用。

#### 2.1.3 `layers/enc_dec.py`（Asteroid filterbank，全部死代码）

| 类/函数 | file:line | 说明 | 参数（实测） |
|---|---|---|---|
| `make_enc_dec` | :8-71 | 按 `who_is_pinv` 生成 `(Encoder,Decoder)`；`"free"` 时 `enc=32768, dec=32768` | — |
| `Filterbank` | :114-175 | 抽象基类，`forward` 直接 `NotImplementedError`(:172-175) | — |
| `_EncDec` | :178-220 | `compute_filter_pinv`(:201) 用 `torch.pinverse` + `stride/kernel` 缩放 | — |
| `Encoder` | :223-281 | `(2,1,4096)→(2,128,31)` | 32768 |
| `multishape_conv1d` | :284-326 | 1D/2D/3D/ND 分派到 `batch_packed_1d_conv` | — |
| `Decoder` | :341-392 | `(2,128,31)→(2,1,4096)`（`length` 截断 :389-391） | 32768 |
| `multishape_conv_transpose1d` | :395-431 | OLA = `conv_transpose1d` | — |
| `FreeFB`/`free` | :434-461 | 无约束滤波器组（= `Conv1d` 权重） | — |

#### 2.1.4 `layers/normalizations.py`（**唯一被真实使用的模块**）

| 类 | 别名 | file:line | 统计轴 | 参数(C=64) | 因果? | Falcon 用? |
|---|---|---|---|---|---|---|
| `norm`/`glob_norm` | — | :11-20 | 任意 dims | — | — | 间接 |
| `MLayerNorm` | — | :23-35 | `forward` 是 `pass`(:34-35) → 返回 None | — | — | 否 |
| `GlobalLN` | — | :38-41 | 全轴 | — | 否 | 否 |
| `ChannelLN` | — | :44-48 | `dim=1` | — | 是 | 否 |
| `CumulateLN`(3D) | — | :51-61 | 累积轴写错（见 §5-8） | — | — | 否 |
| `BatchNorm` | `bN` | :64-71 | 仅 `_check_input_dim` | 128 | 否 | 否 |
| `CumulativeLayerNorm` | `LN` | :74-88 | 转置后 `nn.LayerNorm` | 128 | 是 | 否 |
| `CumulateLN`(3D) | `cLN` | :91-132 | cum over T，stats over C | 128 | 是 | 否（3D 版） |
| `GlobalLayerNorm` | `gLN` | :134-143 | `nn.GroupNorm(1,C)` → `(C,T,F)` 全窗 | 128 | **否** | **是**（作为 `cLNhw` 组件） |
| `LayerNormalization4D` | `LN4d` | :146-163 | `(1,3)` 若 `freq>1` 否则 `(1,)` | 16,512 | 是 | **是**（60 个实例） |
| `CumulateLN4D` | — | :165-202 | cum over `dim=2`(T)，stats over C | 128 | 是 | **是**（30 个实例） |
| `LayerNormalizationHW` | **`cLNhw`** | :204-221 | `GlobalLayerNorm` → `CumulateLN4D` | 256 | **否** | **是**（30 个实例） |

#### 2.1.5 `layers/stft.py`（函数式 + 类式 STFT，全部死代码）

| 对象 | file:line | 输入 → 输出 | 参数 | 说明 |
|---|---|---|---|---|
| `init_window` | :29-56 | `(wnd,frame_len)` | — | `hann/hamm/blackman/bartlett/rect/sqrthann`；`periodic=True` 对齐 librosa(:53) |
| `init_kernel` | :59-111 | `(fft_size*2,1,fft_size)` | — | DFT 矩阵 `fft(eye)/S`；`inverse` 时再 `/fft_size`(:104-106) |
| `mel_filter` | :114-158 | `(num_mels, N//2+1)` | — | `librosa.filters.mel(htk=True)` |
| `speed_perturb_filter` | :161-185 | `(dst_sr,src_sr,K)` | — | 语音变速重采样核 |
| `splice_feature` | :188-218 | `N×…×T×F → N×…×T×FD` | — | 上下文拼接（clamp 补齐） |
| `_forward_stft` | :221-286 | `(B,(C),S)→(B,(C),F,T,2)` | — | `conv1d`+DFT 核；`center` 用 reflect pad(:253-256) |
| `_inverse_stft` | :289-354 | `(N,F,T,2)→(N,S)` | — | `conv_transpose1d` + OLA 除 `denorm`(:339-352) |
| `_pytorch_stft` | :357-413 | 同 `torch.stft` | — | 纯包装 |
| `_pytorch_istft` | :416-471 | 同 `torch.istft` | — | 纯包装 |
| `forward_stft` | :474-541 | `(B,S)→(B,F,T,2)` | — | `mode ∈ {librosa,kaldi,torch}` |
| `inverse_stft` | :544-607 | `(B,F,T,2)→(B,S)` | — | 对称 |
| `STFTBase` | :610-696 | — | — | `K`,`w` 存成 `requires_grad=False` 的 `nn.Parameter`(:650-651) |
| `STFT` | :699-741 | `(2,4096)→(2,129,33,2)` | **131,328** | 全部冻结参数 |
| `iSTFT` | :744-785 | `(2,129,33,2)→(2,4096)` | **131,328** | 同上 |

#### 2.1.6 `layers/stft_tfgn.py` / `utils/attn.py` / `utils/*`

| 对象 | file:line | 输入 → 输出 | 参数 | 说明 |
|---|---|---|---|---|
| `Stft` | stft_tfgn.py:23-245 | `(2,4096)→(2,33,129,2)` | 0 | ESPnet 风格；`inverse`(:188) 走 `torch.istft` |
| `CrossAttention` | attn.py:3-29 | `(2,10,128)+(2,20,128)→(2,10,128)` | 65,664 | 返回 `(x, attn)` |
| `InversibleInterface` | inversible_interface.py:7-13 | 抽象类 | 0 | 仅 `Stft` 实现 |
| `separate/torch_separate` | separator.py:29-61 | numpy/Tensor 分派 | 0 | `:57` 有能量重标定 `out *= |in|.sum()/|out|.sum()` |
| `make_pad_mask` | nets_utils.py:64-… | `(B,)`+ref → bool mask | 0 | `stft_tfgn:182` 用于 `masked_fill_` |
| `pad_x_to_y` | torch_utils.py:5-10 | 右补零 | 0 | — |
| `get_layer` | get_layer_from_string.py:6-43 | 字符串→`torch.nn` 类 | 0 | — |

### 2.2 被用到 / 死代码（两张表，grep 全仓取证）

命令（在 repo 根执行，排除 `look2hear/layers/` 自身、`docs/`、`tools/`、`studio/falcon_model/layers/`）：

```bash
for s in Stft make_enc_dec FreeFB DPRNN DPRNNLinear LSTMBlockTF TransformerBlockTF \
         Conv1DBlock ConvNormAct ConvNorm NormAct Video1DConv Concat FRCNNBlock \
         FRCNNBlockTCN Bottomup BottomupTCN TAC forward_stft inverse_stft CrossAttention; do
  echo "$s: $(grep -rn --include=*.py "\b$s\b" . | grep -v '^./docs/' | grep -v '^./look2hear/layers/' \
        | grep -v '^./tools/' | grep -v '^./studio/falcon_model/layers/' | wc -l)"; done
```

**表 A：被 Falcon 真实使用的层（6 项）**

| 名字 | 引用点 | 实例数（生产配方） |
|---|---|---|
| `normalizations.get` | `Falcon.py:89,91,188,372` | — |
| `LayerNormalizationHW`(`cLNhw`) | 由 `norm_type: cLNhw` 解析 | **30** |
| `CumulateLN4D` | `cLNhw` 内部(:210) | 30 |
| `GlobalLayerNorm` | `cLNhw` 内部(:208) | 30 |
| `LayerNormalization4D` | `norm_type: LayerNormalization4D` | **60** |
| `activations.get` | `Falcon.py:90,92,187` | — |

归一化层清点（实测，§4 E5）：`GroupNorm` 30 / `LayerNormalizationHW` 30 / `LayerNormalization4D` 60 / `BatchNorm1d` 1（视频 projection）/ `nn.LayerNorm` **0**。注意 `gLN`、`cLN`、`bN` 三个别名在 Falcon 生产配方里**一次都没实例化**（视频 `norm_type: BatchNorm1d` 走的是 `get()` 对 `torch.nn` 的回退，见 §5-9）。

**表 B：死代码（外部引用计数全为 0，逐项确认）**

| 名字 | 外部引用 | 名字 | 外部引用 |
|---|---|---|---|
| `Stft`(stft_tfgn) | **0** | `ConvNorm` | **0** |
| `make_enc_dec` | **0** | `NormAct` | **0** |
| `FreeFB` | **0** | `Video1DConv` | **0** |
| `DPRNN`(rnnlayers) | 0（仅 `evaluated_mac_params.py:85` 注释） | `Concat` | **0** |
| `DPRNNLinear` | **0** | `FRCNNBlock`/`FRCNNBlockTCN` | **0** |
| `LSTMBlockTF`/`TransformerBlockTF` | **0** | `Bottomup*`（4 个） | **0** |
| `Conv1DBlock` | **0** | `TAC` | **0** |
| `ConvNormAct`(cnnlayers) | 69 命中**全部**是 `Falcon.py:11` 自己的同名类 | `forward_stft`/`inverse_stft` | 仅 `layers/__init__.py:19` re-export |
| `CrossAttention` | **0** | `complex_utils.*` | **0** |

唯一"半死"的是 `look2hear/utils/torch_utils.py:19 tensors_to_device`，被 `audio_visual_test.py:67` 使用（评测入口，不在模型前向）。

### 2.3 真实 DSP 路径（Falcon 的 5 段，逐层形状）

配方：`configs/LRS2_2Mix_Paper_v2.yml`（2 s / 32000 采样 / 16 kHz / batch 2）。

```
wav (B,1,32000)
 └─ STFTEncoder (Falcon.py:1114-1128)
    torch.stft(n_fft=256, hop=128, window=hann256, return_complex=True)
      → spec (B,129,251)  [center=True ⇒ 帧数 = 32000//128 + 1 = 251]
    this_spec = sqrt(|spec|² + eps)                       :1124
    spec = stack([real, imag, mag], dim=1).transpose(2,3)  :1125
      → (B,3,251,129)
    Conv2d(3→112, k=3, pad=(2,1)) + 裁 T 末 2 帧          :1126 / ConvNormAct:61,104
      → (B,112,251,129)     ← 时间因果（lookback 2 帧），频率对称 pad
 └─ audio_bottleneck (ConvNormAct, pre_norm=cLNhw)        Falcon.py:1322
      pre_norm: GroupNorm(1,112) over (C,T,F)  ← 全窗统计，非因果
      Conv2d(112→112, k=1) → (B,112,251,129)
 └─ Separator (Falcon.py:1032-1059)
      fusion_repeats=1, audio_repeats=1（repeats=2 - 1）
      ├─ AudioBlock ×2  : output (B,112,251,129)
      │   ├─ gateway  : dw Conv2d(112,1,groups=112) + cLNhw
      │   ├─ projection: Conv2d(112→64,k=1) + cLNhw + PReLU
      │   ├─ downsample[0]: dw k=3 s=1  → (B,64,251,129)
      │   │  downsample[1..3]: dw k=5 s=2 → (126,65) → (63,33) → (32,17)
      │   ├─ 只在最深 level 跑 process_f(沿F 双向LSTM) → process_t(沿T 单向LSTM) → attn(causal tril)
      │   └─ top-down: InjectionMultiSum ×4 + 上采样回到 251×129
      ├─ VideoBlock ×1 : (B,512,62) → (B,512,62)（LSTM 布局错误，见 §5-5）
      └─ NeuroSCNetFusion: Conv1d(512→112, k=1, groups=16) → unsqueeze(-1) → 广播加 (B,112,251,129)
 └─ MaskGenerator (Falcon.py:1273-1282)
      PReLU → Conv2d(112→112, k=1, act=ReLU) → masks (B,112,251,129) 全 ≥0
      RI_split: view(B,1,2,56,251,129) 与 emb.view(B,2,56,251,129)
      est = (er·mr − ei·mi) ⊕ (er·mi + ei·mr)              :1263-1266
 └─ STFTDecoder (Falcon.py:1170-1199)
      view(B*n_src,112,251,129) → F.pad(0,0,2,0)→253
      ConvTranspose2d(112→2,k=3,s=1,p=1) → (B,2,253,129) → 裁 1 → 252
      torch.complex(re,im).transpose(1,2) → (B,129,252)
      torch.istft(n_fft=256, hop=128, length=32000) → (B,1,32000)
```

解码器的帧记账（实测，§4 E11）：`251 → pad 253 → convT 253 → crop 252`，`252` 帧无 `length` 时 `istft` 会给 `(252−1)*128+256 = 32384` 采样，传 `length=32000` 后截断。

### 2.4 `cLNhw` / GN 的归一化轴：精确语义与"为什么这样选"

**代码**（`normalizations.py:204-221`）：

```python
class LayerNormalizationHW(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        self.global_layer_norm = GlobalLayerNorm(num_channels=num_channels, eps=eps)
        self.cumulate_ln      = CumulateLN4D(dimension=num_channels, eps=eps)
    def forward(self, x):        # x: (B,C,H=T,W=F)
        x = self.global_layer_norm(x)   # 归一化 (C,H,W)
        x = self.cumulate_ln(x)         # 沿 H 累积，统计跨 C
        return x
```

- `GlobalLayerNorm`(:134-143) 就是 `nn.GroupNorm(num_groups=1, num_channels=C)`：**单组 = 跨 `C,H,W` 的逐样本标量均值/方差**。实测与手算 `mean over (C,T,F)` 差 `2.384e-07`，与 `nn.GroupNorm(1,C)` 差 **0.0**。
- `CumulateLN4D`(:165-202)：`step_sum = x.sum(dim=1, keepdim=True)`(:182) → 统计跨 **C**；`cumsum(dim=2)`(:185) → 累积沿 **H=T**；`entry_cnt = arange(C, C(H+1), C)`(:189) 即"已见过 `t·C` 个元素"。逐 `(b,f)` 独立。实测与手算 `max|Δ| = 0.0`。
- 组合语义 = **先全窗 GN，再沿时间累积的 cLN**。所以"`cLNhw` 是 H(时间)+W(频率) 双轴归一化"这个直觉**只对了一半**：频率轴只在第一阶段被卷进全局标量里，第二阶段对 F **完全独立**（逐频点）。

**为什么这样选**：名字 `HW` 暗示作者想要 "LayerNorm over H and W"，即对 2D TF 图做类似 LayerNorm 的全轴归一化以消除输入增益；但为了保持时间因果又叠了一个 `CumulateLN4D`。两者叠加后**因果性被第一阶段彻底破坏**（§4 E6/E8 实测量化），所以这是个"想同时要两件事、结果只拿到后者归一化、丢掉前者因果"的组合。

**与论文公式的一致性**：论文（arXiv:2610.32749）正文没有给出 `cLNhw` 的显式公式，只在实现层面等价于 "global norm + cumulative norm"。仓库内的实现与常见 cLN（`Channel-wise LayerNorm`，Luo & Yu 2019）**不同**：标准 cLN 只用累积统计（`cLN`/`CumulateLN`），而 `cLNhw` 在其前面多了一个全窗 GN。判定：**不一致/超集**，且多出来的那一项破坏因果。详见 §6。

**冗余性（本轮新发现，量化）**：

| 检验 | 结果 |
|---|---|
| `cLNhw(x)` vs `CumulateLN4D(GlobalLayerNorm(x))` | `3.576e-07` → 定义一致 |
| `cLNhw(x)` vs `CumulateLN4D(x)`（**去掉 GN**） | `3.576e-07` → **GN 在初始化态是空操作** |
| 把 GLN 的 `norm.weight/bias` 随机化后再比 | `1.232` → GN 只剩下它的 2×C 个仿射参数在起作用 |
| `cLN4D(a·x+b)` vs `cLN4D(x)`（a=3.7,b=−2.1 逐样本仿射） | `4.768e-07` → 对逐样本仿射严格不变 |
| `cLNhw(a·x+b)` vs `cLNhw(x)` | `4.768e-07` → 同上（GN 被吸收） |

数学原因：GN 输出 `x̂=(x−μ)/σ`（μ,σ 为逐样本标量），随后 cLN 计算 `(x̂−cum_mean)/cum_std`；若把 `x` 换成 `a·x+b`，`x̂` 不变，故整条链对全局增益/偏置不变。于是 GN 的归一化**本来就多余**，`cLNhw` 相对 `CumulateLN4D` 的净增量只有 2×C 个参数（C=64 时 256 vs 128 个参数，实测）。

### 2.5 STFT / iSTFT 数学

**Falcon 生产路径**（`Falcon.py:1117-1123`）：

| 参数 | 值 | 依据 |
|---|---|---|
| `n_fft` | **256**（= `win`） | `configs/LRS2_2Mix_Paper_v2.yml:41` |
| `hop_length` | **128** | 同上 `:42` |
| `window` | `torch.hann_window(256)`，`periodic=True`（PyTorch 默认） | `Falcon.py:1100` |
| `center` | **True**（默认，未显式传） | `Falcon.py:1117-1123` 未传 → `torch.stft` 默认 `True` |
| `pad_mode` | **`"reflect"`**（默认） | 同上 |
| `normalized` | **False**（默认） | 同上 |
| `onesided` | **True**（默认） | 同上 |
| `return_complex` | **True**（唯一显式传的开关） | `Falcon.py:1122` |
| 频点 | `256//2+1 = 129` | — |
| 帧数（2 s） | `1 + 32000//128 = 251` | 实测 §4 E5 |

**加窗与尺度补偿**：`normalized=False` ⇒ 正变换无 `1/sqrt(N)`，逆变换由 `torch.istft` 内部按窗平方的 OLA 归一化（`Σw²` 除法）。因此**前向幅度与窗能量成正比**，`|spec|` 的量级≈`0.5·sqrt(256)·|X|`（hann 相干增益 0.5）。`Falcon.py:1124` 的 `this_spec = sqrt(|spec|²+eps)` 因此是"带 hann 增益的幅度"，与 `real/imag` 两通道量纲一致——**三通道特征之间没有做尺度对齐**（见 §5-4）。

**iSTFT 的重叠相加与 length**（`Falcon.py:1189-1195`）：`torch.istft(..., length=length)`，`length = input_shape[-1] = 32000`。`torch.istft` 的 `center=True` 会在末尾裁掉 `n_fft//2`。

**死代码路径的等价性**（`layers/stft.py`）：

- `_forward_stft`：`kernel = kernel * window`(:258)；`conv1d(wav, kernel, stride=hop)`(:270) 等价于"加窗后做 DFT"，`center=True` 时先 `tf.pad(wav,(pad,pad),mode="reflect")`(:253-256) 且 `pad = kernel.shape[-1]//2 = 128`。
- `_inverse_stft`：`conv_transpose1d(packed, kernel*window, stride=hop)`(:334) 完成 OLA；`denorm` 用 `conv_transpose1d(win², eye)`(:339-343) 复现 `Σw²`；`wav/(denorm+eps)`(:352)。
- `onesided` 的共轭补齐在 `:327-330`（inverse）与 `:277-280`（forward，直接切 `num_bins`）。

### 2.6 `stft_tfgn.py`：TF-GN 并不存在，真正的耦合是 `typeguard`

**拨乱反正**：全仓库 `grep -rn "TF-GN\|TFGN\|tf_group\|TFGroupNorm"` 只命中 `docs/` 的两份旧文（`docs/understanding/deepdive-20261024/02-layers-dsp-video.md` 标题、`docs/verification/phase2/…captured.json`），**源码里没有任何 TF-GN**。`stft_tfgn.py` 全文只有 `class Stft(torch.nn.Module, InversibleInterface)`(:23) 一个类，没有任何归一化。

它的真实作用是**依赖钉子**：

```
look2hear/layers/stft_tfgn.py:8   from typeguard import check_argument_types
look2hear/layers/stft_tfgn.py:34      assert check_argument_types()      ← 在 __init__ 里调用
look2hear/layers/__init__.py:20       from .stft_tfgn import Stft
look2hear/models/Falcon.py:9      from ..layers import normalizations, activations
```

→ 只要 `import look2hear.models`，就会执行 `check_argument_types` 的 **import**。`typeguard 3.x` 移除了这个函数（上游 API 改为 `@typechecked`），于是**整条模型 import 链直接 ImportError**，与 stft 是否被调用无关。这就是 `AGENTS.md` / `docs/STATUS.md:21` 强制 `typeguard 2.13.3` 的根因。

**实测行为**（2.13.3，§4 E7）：

```
typeguard 2.13.3 ; has check_argument_types: True
signature: (memo: Optional[typeguard._CallMemo] = None) -> bool
Stft(n_fft="256")  → TypeError: type of argument "n_fft" must be int; got str instead
Stft(window=123)   → TypeError: type of argument "window" must be one of (str, NoneType); got int instead
Stft(center="yes") → TypeError: type of argument "center" must be bool; got str instead
```

即 `assert check_argument_types()` 是**真的在校验**（`check_argument_types()` 无参调用时用 `sys._getframe(1)` 检查调用方 `__init__` 的实参），不是 no-op。仓库里的 `docs/verification/stubs/typeguard.py:1-6` 是一个 no-op 替身，注释明确写着 "removed in typeguard>=3"，仅供离线验证脚本用。

### 2.7 复数算子：只有 RI 复数乘法一种

全仓 `grep` 结果：**没有** complex conv、complex layernorm、modReLU；`torch.complex` 只在三处出现（`Falcon.py:1186` 解码器组装、`stft_tfgn.py:231` 逆变换组装、`complex_utils.py:21/102/106/132/136` 的包装）。

**唯一的复数算子**是 `MaskGenerator.__apply_masks`(`Falcon.py:1251-1271`) 的 RI 拆分：

```python
masks  = masks.view(B, n_src, 2, C//2, T, F)         # :1255
emb    = emb.view(B, 2, C//2, T, F)                  # :1256
mr, mi = masks[:,:,0], masks[:,:,1]                  # :1258-1259
er, ei = emb[:,0], emb[:,1]                          # :1260-1261
est_r = er*mr - ei*mi                                # :1263  复数乘法实部
est_i = er*mi + ei*mr                                # :1264  复数乘法虚部
```

**数值稳定性**：无除法、无 `1/|z|`、无 `eps` 参与 → 没有除零风险；`eps` 只出现在两处：`Falcon.py:1085`（`STFTEncoder.eps = finfo(float32).eps`，用于 `sqrt(|spec|²+eps)` 防 `sqrt` 在 0 处梯度 NaN）和 `Falcon.py:1381,1385`（SI-SNR 的分母）。

**但有一个语义坑（本轮新发现）**：`mask_act: ReLU`（`configs/LRS2_2Mix_Paper_v2.yml:108`）作用在 112 个 mask 通道上，**实部和虚部两半同时被 ReLU 变成非负**。实测 `masks.min() = 0.0000`、`<0 占比 = 0.0000`、real 半段/imag 半段 min 都是 `0.0000`。后果：每 bin 的复数掩码相位被锁在 **第一象限 [0°,90°]**，无法表达 90°–360° 的相位旋转。这是 as-built 行为，不是笔误（`ReLU` 在配置里显式指定）。

### 2.8 流式 / 缓存：**本质上是"无状态整窗重算"**

- `grep -rn "cache_hat\|cache_prev\|self.cache"` 全仓（排除 docs）**0 命中**（只有 `torch.cuda.empty_cache()` 与 `huggingface_hub.try_to_load_from_cache` 这类无关命中）。
- `InversibleInterface`(`utils/inversible_interface.py:7-13`) 只被 `Stft`(`stft_tfgn.py:23`) 实现，而 `Stft` 从未被实例化。
- `STFTEncoder/Decoder`、`AudioBlock`、`DualPathRNN`、`MaskGenerator` 都**没有** `forward(..., cache=...)` 参数，也没有 `self.xxx_cache` 成员。
- 唯一跨调用状态在 Studio 层（`studio/falcon_runtime.py`，不在本文件范围）。

**"kernel 移动"的逐帧轨迹（代替不存在的 cache 轨迹）**：一次调用吃满一个窗；窗与窗之间**没有任何递推量**，因此"前进一帧"在模型层不存在，只有"前进一 hop 重算整窗"。实测（§4 E6d/E6e）：

| 实验 | 结果 | 含义 |
|---|---|---|
| 输入整体右移 128 采样（=1 hop） | `y3[128:384]` vs `y1[0:256]` `maxAbsD = 3.877e-01` | **不是平移等变**：`center=True` 的 reflect pad 与全窗统计一起破坏了等变 |
| 2 s 整窗跑 vs 只跑前 1 s | 前 1 s 输出 `maxAbsD = 1.940e-01` | 输出依赖**整窗**统计（`cLNhw` 的 GN），无法用短窗近似 |

逐帧映射（可用来算 lookahead）：帧 `t` 覆盖采样 `[128t−128, 128t+127]`；`p` 处的脉冲影响 STFT 帧 `t ∈ [⌈(p−127)/128⌉, ⌊(p+128)/128⌋]`（实测 p=2000 → 帧 15,16；p=1000 → 帧 7,8）。再经 `ConvNormAct(pad=(2,1),causal chop 2)` → 输出帧 `t..t+2`（实测 p=2000 → 帧 15..18，lookback 2 帧、lookahead 0）。

### 2.9 搜索效率：卷积核 / group / padding / dilation 全表 + 参数量与 FLOPs 手算

**生产配方里真实存在的卷积（去重后，§4 E5b 输出）**：

| 算子 | in→out | k | stride | padding | dilation | groups | bias | 位置 |
|---|---|---|---|---|---|---|---|---|
| Conv2d | 3→112 | (3,3) | (1,1) | **(2,1)** | (1,1) | 1 | False | `STFTEncoder` `Falcon.py:1087` |
| Conv2d | 112→112 | (1,1) | 1 | 0 | 1 | 1 | True | `audio_bottleneck` `:1322` |
| Conv2d | 112→112 | (1,1) | 1 | 0 | 1 | **112** | True | `AudioBlock.gateway` `:720` |
| Conv2d | 112→64 | (1,1) | 1 | 0 | 1 | 1 | True | `AudioBlock.projection` `:729` |
| Conv2d | 64→64 | (3,3) | (1,1) | (1,1) | 1 | **64** | True | `downsample[0]`（`causal=False`）`:758` |
| Conv2d | 64→64 | (5,5) | **(2,2)** | (2,2) | 1 | **64** | True | `downsample[1..3]` `:758` |
| Conv2d | 64→64 | (5,5) | 1 | (2,2) | 1 | **64** | **False** | `InjectionMultiSum.local/global_embedding` `:285` |
| Conv2d | 64→64 | (5,5) | 1 | (2,2) | 1 | **64** | False | `InjectionMultiSum.global_gate` `:305` |
| Conv2d | 64→4 | (1,1) | 1 | 0 | 1 | 1 | True | `MHSA2D.Queries/Keys`（`hid_chan=4`）`:447` |
| Conv2d | 64→16 | (1,1) | 1 | 0 | 1 | 1 | True | `MHSA2D.Values`（`in_chan//n_head`）`:472` |
| Conv2d | 64→112 | (1,1) | 1 | 0 | 1 | 1 | True | `AudioBlock.residual_conv` `:820` |
| Conv1d | 512→512 | 1 | 1 | 0 | 1 | **512** | True | `VideoBlock.gateway` `:933` |
| Conv1d | 512→32 | 1 | 1 | 0 | 1 | 1 | True | `VideoBlock.projection` `:942` |
| Conv1d | 32→512 | 1 | 1 | 0 | 1 | 1 | True | `VideoBlock.residual_conv` `:958` |
| Conv1d | 512→112 | 1 | 1 | 0 | 1 | **16** | True | `NeuroSCNetFusion.v_proj`(`gcd(112,512)=16`) `:246` |
| ConvTranspose2d | 112→2 | (3,3) | (1,1) | (1,1) | 1 | 1 | False | `STFTDecoder` `:1156` |

**RNN**（9 个，全部 `batch_first=False`）：

| 模块 | in | hidden | layers | bidir | 参数（实测） | 扫的轴 |
|---|---|---|---|---|---|---|
| `process_f_layers[i].rnn` ×4 | 64 | 24 | 1 | **True** | 17,280 | **频率**（`dim=4` 先转置，`:389`） |
| `process_t_layers[i].rnn` ×4 | 64 | 72 | 1 | False | 39,744 | 时间（`dim=3`） |
| `video_net.rnn` | 32 | 32 | 1 | False | 8,448 | 时间（但布局错，见 §5-5） |
| 合计 | | | | | **236,544** | 占 537,697 的 44.0% |

LSTM 参数量手算校验：`4h(in+h) + 8h` 每方向。`h=24,in=64` → `4·24·64 + 4·24·24 + 8·24 = 6144+2304+192 = 8640`，双向 ×2 = **17,280** ✓ 与实测一致。

**参数量手算校验（总账）**：`sum(p.numel()) = 537,697`（=1.0×`537,697`，与 `Falcon.py:990` 注释的 537,697 完全一致）。主要项：

| 模块 | 参数 | 占比 |
|---|---|---|
| `process_t_layers`（4×LSTM 72） | 177,920 | 33.1% |
| `process_att_layers`（4×MHSA + 60 个 LN4D） | 119,732 | 22.3% |
| `process_f_layers`（4×LSTM 24 双向） | 81,920 | 15.2% |
| `fusion_layers` + `concat_layers`（8×InjectionMultiSum） | 44,544 | 8.3% |
| `video_net`（gateway/projection/rnn/residual） | 41,825 | 7.8% |
| `mask_generator` | 12,656 | 2.4% |
| `encoder`（Conv2d 3→112 k3） | 12,544 | 2.3% |
| `spk_pool_proj`（对比头，非分离主体） | 7,232 | 1.3% |

**FLOPs 手算 vs 实测（2 s 窗，32000 采样）**：

| 分项 | 实测 MACs | 手算 | 校验 |
|---|---|---|---|
| `STFTEncoder` | 0.099 G | `251·129·112·3·9 = 97.9 M` | ✓ |
| `audio_bottleneck` | 0.406 G | `251·129·112·112 = 406.1 M` | ✓ |
| `video_bottleneck` | **0.000 G** | 全 `Identity`（`kernel_size=-1`） | ✓ |
| `Separator`（AudioBlock×2 + VideoBlock + CAF） | **4.405 G** | 单次 AudioBlock ≈1.10 G + 2.20 G | — |
| ↳ 其中 `projection` | 0.232 G | `251·129·64·112 = 232.1 M` | ✓ |
| ↳ 其中 `downsample[0..3]` | 0.058 G | `(251·129·64·9 + 126·65·64·25 + 63·33·64·25 + 32·17·64·25)/1e9 = 0.036 G` | 同量级 |
| ↳ 其中 `InjectionMultiSum`×4 | 0.260 G | `3·64·25·Σ(级别面积)` | — |
| `MaskGenerator` | 0.406 G | `251·129·112·112 = 406.1 M` | ✓ |
| `STFTDecoder` | 0.066 G | `252·129·112·2·9 = 65.5 M` | ✓ |
| **合计（`FlopCounterMode` 口径，另一把尺子）** | **5.382 GMACs = 10.763 GFLOPs** | 分项合计 5.382 | ✓ 闭合 |

`FlopCounterMode` 的算子级归因：`aten.convolution 3.229 G` + `aten.bmm 1.489 G`（attention + LSTM）+ `aten.mm 0.663 G` = 5.382 G。即**卷积占 60%，其中最大单项是 top-down `InjectionMultiSum` 在满分辨率 (251×129) 上的 3 个 depthwise k=5 卷积**。

> 两把尺子：本表全部是 `torch.utils.flop_counter` 口径（不记 `aten::lstm`），合计 5.382 GMAC / 10.763 GFLOPs。
> ptflops 对同一模型同一输入返回 **10.611 GMAC**（含 LSTM、不含 FRCNN），与论文 10.6 G、README、`docs/STATUS.md:33`
> 逐位一致。2026-11-08 Lead 复核更正：此前据本表推断"论文把 FLOPs 写成 MACs"不成立，两个数只是两把尺子。
> 与视觉同尺的整窗成本是 **26.51 GMAC（视觉 15.897 G，占 60.0%）**，见 00 §4、03 §4-K。

## 3. 关键数据/参数表

| 项 | 值 | 口径/来源 |
|---|---|---|
| 模型总参数 | **537,697** | `LRS2_2Mix_Paper_v2`，全部可训练（§4 E5b） |
| 2 s 窗 MACs（ptflops，含 LSTM） | **10.611 G** | 与论文 10.6 同尺；不含视觉（05 §3.2 逐位复算） |
| 2 s 窗 MACs（`FlopCounterMode`，另一把尺子） | **5.382 G** | 不记 `aten::lstm`，不可与 10.6 并列（§4 E10/E15） |
| 2 s 窗 FLOPs（`FlopCounterMode`） | **10.763 G** | 同上 |
| 4 s 窗 MACs | 13.696 G | 同上（近似线性） |
| STFT | n_fft 256 / hop 128 / hann / center=True / normalized=False | `Falcon.py:1100,1117-1123` |
| TF 图尺寸（2 s） | 251 帧 × 129 bin | 实测 |
| 归一化层 | cLNhw ×30 / LN4D ×60 / GroupNorm ×30 / BatchNorm1d ×1 / LayerNorm ×0 | §4 E5 |
| RNN 参数 | 236,544（44.0%） | §2.9 |
| AudioBlock 感受野（含 `cLNhw`） | 全窗（T=251 → 251/251 帧） | §4 E8 |
| AudioBlock 感受野（`cLN4D` 替换后 = 单遍几何） | lookback 57 帧 / lookahead 0（整链 113 帧 / 904 ms；as-shipped 窗尾） | §4 E8、12 §3 |
| 输入中一帧扰动对输出的影响（整链） | 输出样本 `[2,4095]/4096`（4 s 窗时全变） | §4 E6 |
| `forward_stft` vs `torch.stft` | `max|Δ| = 2.289e-05` | §4 E1 |
| 往返重建（center=True） | `1.907e-06` / 128.5 dB | §4 E2 |
| 往返重建（center=False） | `1.126` / 31.8 dB | §4 E2 |
| GPU 侧 fp16 STFT | **CPU 直接报错**（MKL FFT 不支持 Half） | §4 E2c/E6f |

## 4. 实测验证

所有脚本置于远端 `/tmp/w102/`（**不写仓库**），以 `cd /root/code/github_repos/Falcon-fork && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python /tmp/w102/expN.py` 运行。每个脚本 ≪120 s。

> 踩坑提示：`/tmp/profile.py`（其他 session 遗留）会 shadow 标准库 `profile`，使 `import cProfile` 崩。脚本要放进独立子目录（本报告用 `/tmp/w102/`），否则会撞上这个影子。

### E1 `layers/stft.py` 封装 vs `torch.stft`（exp1.py）

```
--- E1a: forward_stft(mode=librosa) vs torch.stft(return_complex=False) ---
  fl=256 hop=128 center=False win=hann      got(1, 129, 31, 2) ref(1, 129, 31, 2) maxAbsD=2.289e-05
  fl=256 hop=128 center=False win=sqrthann  got(1, 129, 31, 2) ref(1, 129, 31, 2) maxAbsD=2.289e-05
  fl=256 hop=128 center=True  win=hann      got(1, 129, 33, 2) ref(1, 129, 33, 2) maxAbsD=2.289e-05
  fl=256 hop=128 center=True  win=sqrthann  got(1, 129, 33, 2) ref(1, 129, 33, 2) maxAbsD=2.289e-05
  fl=512 hop=256 center=False win=hann      got(1, 257, 15, 2) ref(1, 257, 15, 2) maxAbsD=2.289e-05
  fl=512 hop=256 center=True  win=hann      got(1, 257, 17, 2) ref(1, 257, 17, 2) maxAbsD=2.289e-05
--- E1b: 3-channel input shape ---
  in (2, 3, 4096) -> out (2, 3, 129, 33, 2)
--- E1c: mode=torch path vs torch.stft ---
  maxAbsD 0.0 shape (1, 129, 33, 2)
```

结论：(a) 卷积式 STFT 与 `torch.stft` **等价**，误差恒为 `2.289e-05`（fp32 的 conv-vs-FFT 舍入），与 win/hop/center/window 无关；(b) 输出布局是 **`(B,C,F,T,2)`**（F 在前），与 `stft_tfgn.Stft` 的 `(B,T,F,2)` **相反**；(c) `mode="torch"` 分支是纯包装，误差 **0.0**。

### E2 STFT→iSTFT 往返（exp1.py 续）

```
--- E2: STFT->iSTFT roundtrip (librosa mode, hann) ---
  fl=256 hop=128 center=False norm=False out(1, 4096) maxAbsErr=1.126e+00 SI-SNR=31.8dB
  fl=256 hop=128 center=False norm=True  out(1, 4096) maxAbsErr=1.126e+00 SI-SNR=31.8dB
  fl=256 hop=128 center=True  norm=False out(1, 4096) maxAbsErr=1.907e-06 SI-SNR=128.5dB
  fl=256 hop=128 center=True  norm=True  out(1, 4096) maxAbsErr=1.907e-06 SI-SNR=128.5dB
  fl=512 hop=256 center=False norm=False out(1, 4096) maxAbsErr=1.547e+00 SI-SNR=27.1dB
  fl=512 hop=256 center=True  norm=False out(1, 4096) maxAbsErr=2.146e-06 SI-SNR=127.9dB
--- E2b: sqrthann center=True ---   maxAbsErr 1.669e-06
--- E2c: falcon 实际路径 torch.stft/istft (win=256 hop=128 center=True) ---
  torch.float32 spec(1, 129, 33) wav(1, 4096) maxAbsErr=7.153e-07
  torch.float16 FAILED: RuntimeError: MKL FFT doesn't support tensors of type: Half
  torch.float64 spec(1, 129, 33) wav(1, 4096) maxAbsErr=0.000e+00
```

结论：`center=True` 时往返是数值精确的（`1.9e-06`，128 dB）；**`center=False` 时边缘帧 OLA 覆盖不足**，`denorm` 接近 0 → 31.8 dB 的"假损失"（内区仍是精确的）。`normalized` 开关不影响往返。**fp16 在 CPU 上直接不可用**（MKL FFT 限制）。

### E3 归一化轴语义（exp2.py）

```
=== E3a: 轴语义验证 (B,C,T,F)=(2,8,6,5) ===
  GLN vs nn.GroupNorm(1,C) maxAbsD 0.0
  GLN vs manual global(C,T,F) maxAbsD 2.384e-07
  CumulateLN4D vs manual(cum over T, stats over C) maxAbsD 0.0
  cLNhw vs cLN4D(GLN(x)) maxAbsD 3.576e-07
  cLNhw vs cLN4D(x) alone maxAbsD 3.576e-07            ← GN 是空操作
  cLN4D 对 per-sample 仿射不变性 maxAbsD 4.768e-07
  cLN4D 因果性: 改动 t>=4 后, t<=3 输出变化 maxAbs 0.0   ← 严格因果
  cLNhw 因果性: 同上 maxAbs 4.922                       ← 非因果
  cLNhw(3D) FAILED: ValueError not enough values to unpack (expected 4, got 3)
=== E3b: LayerNormalization4D dim 选择 ===
  input_dimension (8,5) -> self.dim (1, 3) gamma (1,8,1,5)
  input_dimension (8,1) -> self.dim (1,)   gamma (1,8,1,1)
```

结论：`cLNhw` **非因果**（`4.922`），`cLN4D` **严格因果**（`0.0`）；`cLNhw` 只接受 4D 输入（3D 直接 `ValueError`）。

### E4 `LayerNormalizationHW` 的 affine 才是实质（exp12.py）

```
初始: |cLNhw(x)-cLN4D(x)|max = 3.576e-07
GLN affine 置随机后: |cLNhw(x)-cLN4D(x)|max = 1.232e+00
cLNhw 参数量 = 32  cLN4D 参数量 = 16     (C=8)
```

### E5 实例化生产模型 + 归一化清点 + 端到端形状（exp3/exp3b.py）

```
total params 537697  trainable 537697
  separator.audio_net.process_t_layers                   177,920
  separator.audio_net.process_att_layers                 119,732
  separator.audio_net.process_f_layers                    81,920
  separator.audio_net.fusion_layers                       22,272
  separator.audio_net.concat_layers                       22,272
GroupNorm count: 30 | LayerNorm count: 0 | BatchNorm count: 1
cLNhw(LayerNormalizationHW) count: 30 | LN4D count: 60
  encoder conv Conv2d(3, 112, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), bias=False) causal True
  window buffer dtype/len torch.float32 256
=== 端到端 shape 轨迹 (CPU, no_grad, 2s) ===
  encoder          (2,1,32000)        -> (2,112,251,129)  float32
  audio_bottleneck (2,112,251,129)    -> (2,112,251,129)
  video_bottleneck (2,512,50)         -> (2,512,50)          ← Identity
  separator        (2,112,251,129)+(2,512,50) -> (2,112,251,129)
  mask_generator   (2,112,251,129)    -> (2,1,112,251,129)
  decoder          (2,1,112,251,129)  -> (2,1,32000)
  n_frames(2s) = 251   n_fft = 256   bins = 129
```

### E6 因果性：整链 + 受控替换（exp6/exp7b.py）

```
=== audio_bottleneck / separator 因果（L=16000, p=8000）===
[原始 cLNhw]  one fwd 0.3s
   encoder      受影响帧 [62,65] 4/126
   bottleneck   受影响帧 [0,125] 126/126
   separator    受影响帧 [0,125] 126/126
[受控 cLNhw→cLN4D]
   encoder      受影响帧 [62,65] 4/126
   bottleneck   受影响帧 [62,125] 64/126
   separator    受影响帧 [0,125] 126/126
params 原始=537697 cLN4D版=533761
```

`encoder` 严格局部（lookback 2 帧）；`audio_bottleneck` 在 `cLNhw` 下**全窗**、换成 `cLN4D` 后变成 `[62,125]`（因果）；`separator` 仍全窗——因为 T=126 时 AudioBlock 的卷积感受野（~61 帧/侧）已覆盖整窗，见 E8。

### E7 归一化在 4 s 窗上的脉冲响应（exp7.py 草案，被 E8 取代但结论一致）

> 首轮 E7 用 L=32000 跑两次整链，超过 180 s 被 kill（exit 124）。改用 E8 的隔离 AudioBlock 后 0.7 s 完成。**记录此事以提醒后续轮次：整链 CPU 前向在 T=251 上并非实时，别用双跑做脉冲实验。**

### E8 隔离 `AudioBlock` 感受野（exp8.py，关键实验）

```
=== AudioBlock 感受野（隔离，输入 (1,112,T,129)）===
  [cLNhw] T=126 1fwd 0.39s 受影响帧 [0,125] 126/126  lookback=63 lookahead=62
  [cLNhw] T=251 1fwd 0.70s 受影响帧 [0,250] 251/251  lookback=125 lookahead=125
  [cLNhw] T=501 1fwd 1.62s 受影响帧 [0,500] 501/501  lookback=250 lookahead=250
  [cLN4D] T=126 1fwd 0.70s 受影响帧 [4,125]  122/126  lookback=59 lookahead=62
  [cLN4D] T=251 1fwd 1.10s 受影响帧 [68,250] 183/251  lookback=57 lookahead=125
  [cLN4D] T=501 1fwd 2.02s 受影响帧 [196,500] 305/501 lookback=54 lookahead=250
=== VideoBlock 感受野 ===
  [video] T=50 受影响帧 [25,25] 1/50 lookback=0 lookahead=0
```

解读：
- `cLNhw`：**任意 T 都全窗耦合**（`lookback` = `lookahead` = 半窗，即全覆盖）。
- `cLN4D`：`lookback ≈ 54–59 帧`（卷积感受野，随 T 增大略降，因为下采样层数固定而窗变长）；`lookahead` 到窗尾是**累积归一化的正向传播**，不构成因果违规。
- **输出帧 `j` 只依赖输入帧 `[j−57, j]`** ⇒ 换成因果归一化后，"窗口级因果"才成立，**单遍** lookahead ≤ 57 帧 × 8 ms = **456 ms**（与 `docs/STATUS.md:50` 的 57 帧≈456 ms 数字吻合，本轮独立复现）。整链还要加编码器阶梯与第二遍，共 **113 帧 / 904 ms**；含 `cLNhw` 的 as-shipped 是窗尾 ≤2000 ms。
- `VideoBlock` 的 1/50 是 §5-5 的 LSTM 布局 bug 造成的（不是设计意图）。

### E9 `typeguard` 与 import 链（exp4.py）

```
=== E7a: typeguard 版本与 API ===
  typeguard 2.13.3 /root/miniforge3/envs/falcon/lib/python3.11/site-packages/typeguard/__init__.py
  has check_argument_types: True
  signature: (memo: Optional[typeguard._CallMemo] = None) -> bool
=== E7b: layers/__init__ 依赖链 ===
  torch_complex OK | librosa OK | typeguard OK | packaging OK | distutils.version OK | soundfile OK
=== E7c: stft_tfgn.Stft 实例化 ===
  Stft() ok -> n_fft=256, win_length=256, hop_length=128, center=True, normalized=False, onesided=True
  {'n_fft':'256'} -> TypeError type of argument "n_fft" must be int; got str instead
  {'window':123}  -> TypeError type of argument "window" must be one of (str, NoneType); got int instead
  {'center':'yes'}-> TypeError type of argument "center" must be bool; got str instead
=== E7d: Stft 输出布局 ===
  Stft(2,4096) -> out=(2, 33, 129, 2)      ← (B,T,F,2)
```

### E10 复数算子 / mask 非负性（exp12.py）

```
mask 输出 (1,112,251,129)  min 0.0000 max 2.7497  <0 占比 0.0000
RI 拆分后 real 半段 min 0.0000 (ReLU 强制非负? True)
RI 拆分后 imag 半段 min 0.0000 (ReLU 强制非负? True)
```

### E11 解码器帧记账（exp11.py）

```
decoder in view -> (1, 112, 251, 129)
decoder.causal True kernel 3 padding 1 stride 1
after causal pad (1, 112, 253, 129)
after convtranspose (1, 2, 253, 129)
after crop (1, 2, 252, 129)
spec for istft (1, 129, 252)
istft out (1, 32000)  maxlen if no length: 32384
```

### E12 FLOPs 与分项归因（exp9/exp10/exp15.py）

> 本节全部是 `FlopCounterMode` 口径（不记 `aten::lstm`），合计 5.382 GMAC；ptflops 同模型是 10.611 GMAC。两把尺子不可混用。

```
L= 32000 ( 2.0s) 总 10.763 GFLOPs = 5.382 GMACs   ←[复核注] FlopCounterMode 口径；ptflops 同模型为 10.611 GMAC
L= 64000 ( 4.0s) 总 27.393 GFLOPs = 13.696 GMACs
--- 分算子 ---
   aten.convolution   3.229 GMACs
   aten.bmm           1.489 GMACs
   aten.mm            0.663 GMACs
--- 分模块 ---
  STFTEncoder 0.099 | audio_bottleneck 0.406 | video_bottleneck 0.000
  Separator 4.405 | MaskGenerator 0.406 | STFTDecoder 0.066   （合计 5.382）
--- AudioBlock 内部 ---
  gateway 0.004 | projection 0.232 | downsample[0..3] 0.058 | InjectionMultiSum×4 0.260
```

### E13 死代码可用性巡检（exp13/exp14.py）

```
Conv1DBlock 201,474 | ConvNormAct 198,145 | ConvNorm 198,144 | NormAct 257
Video1DConv 132,608 | Concat 98,817 | FRCNNBlock 3,839,110 | FRCNNBlockTCN 5,922,950
Bottomup 83,457 | BottomupTCN 2,179,585 | Bottomup_Concat_Topdown 4,804,229
TAC 296,067
DPRNNBlock 594,688 | DPRNN 3,667,329 | LSTMBlockTF 297,344 | TransformerBlockTF 401,408
SingleRNN 132,096 | RelativeMultiHeadAttention 82,688 | ConformerConvModule 54,016
DepthwiseConv1d 3,968 | PointwiseConv1d 16,512 | FeedForwardModule 131,968
make_enc_dec(free) enc=32768 dec=32768  Encoder(2,1,4096)->(2,128,31)->Decoder(2,1,4096)
Stft params=0 out=(2,33,129,2) | STFT params=131,328 out=(2,129,33,2) | iSTFT params=131,328
CrossAttention 65,664
DPRNNLinear → RuntimeError mat1 and mat2 shapes cannot be multiplied (512x50 and 128x128)
Bottomup_Concat_Topdown_TCN → RuntimeError Sizes of tensors must match except in dimension 1
```

## 5. 坑与反直觉点

1. **`layers/stft.py` 与 `stft_tfgn.py` 的输出布局相反**。`forward_stft`/`STFT` → `(B,C,F,T,2)`（`stft.py:286,273`），`Stft.forward` → `(B,C,T,F,2)`（`stft_tfgn.py:160-166`）。任何"把两边当同一个 STFT 用"的改造都会静默算错（F/T 互换后 shape 依然合法）。实测：`STFT → (2,129,33,2)`，`Stft → (2,33,129,2)`。
2. **`layers/stft.py` 的 `STFT`/`iSTFT` 各自把 131,328 个元素塞进 `nn.Parameter(requires_grad=False)`**（:650-651）。它们进 `state_dict`、进 `sum(p.numel())`，但不可训练。若有人按 "params 应为 0.5M" 做校验，一旦误接这套层就会凭空多 262k 参数。
3. **`layers/stft.py:10` `from distutils.version import LooseVersion`**，Python 3.12 起标准库移除 `distutils` → 该文件在 3.12 上 import 即崩。当前 env 是 3.11.17 才没事。这是"环境钉死"的第二颗钉子（第一颗是 `typeguard<3`）。
4. **`Falcon.py:1125` 的三通道特征量纲不一致**：`[real, imag, mag]` 中 `mag` 是 `sqrt(re²+im²+eps)`，与 `real/imag` 同量纲，但三者**没有做方差归一化**就直接进 `Conv2d(3→112)`。紧跟其后的 `ConvNormAct` 只有 `norm_type=None`（`audio_bn_params` 没给 `norm_type`），真正的归一化发生在 `pre_norm_type: cLNhw` —— 而 `pre_norm` 作用在**整个 (C,T,F) 上**，对"三通道尺度差"只是一次全局缩放，不能逐通道校正。
5. **`VideoBlock` 的 LSTM 时间建模完全失效**：`Falcon.py:970` `x_enc.transpose(1,2)` 得 `(B,T,C)`，但 `:952` 的 `nn.LSTM` 用默认 `batch_first=False`，于是被解释成 `seq_len=B`、`batch=T`。实测脉冲响应只有 `1/50` 帧变化（`[25,25]`），即 **LSTM 在时间维上是逐帧独立的前馈变换**。720 个"时间建模"参数（8,448）名存实亡。
6. **`rnnlayers.py:882` `output.transpose(1, 1)`**：`DRPNNLinear` 里把 `(B,C,T)` 送进 `nn.Linear(C,out)` 之前写了个恒等转置，实测 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x50 and 128x128)`。即便改成 `(1,2)`，随后的 `* self.net_gate(output)` 又是 `(B,out,T) * (B,C,T)` 广播不匹配——两处都错。
7. **`Bottomup_Concat_Topdown_TCN`(`cnnlayers.py:659-680`) 缺上采样**：`bottomup[i-1]`（长度 50）与 `bottomup[i]`（长度 25）直接 `cat` → `RuntimeError: Sizes of tensors must match except in dimension 1`。对照 `Bottomup_Concat_Topdown:576-586` 有 `F.interpolate`，可知是复制粘贴时漏了。
8. **`CumulateLN`(3D, `normalizations.py:51-61`) 的累积轴写错**：`step_sum = x.sum(1, keepdim=True)` 得到 `(B,1,T)`，然后 `torch.cumsum(..., dim=1)` 在**长度为 1 的轴上**累积 → 等于没累积。所幸它是死代码（`cLN` 别名指向 3D 版 `CumulateLN`，而 Falcon 只用 `cLNhw`）。真正能用的 3D 版是 `:91-132` 的第二个 `CumulateLN`（同名类，后者覆盖前者）。
9. **`normalizations.get` 会回退到 `torch.nn`**（`:236-239`）：`hasattr(nn, identifier)` 优先，所以 `norm_type: BatchNorm1d`（`configs/LRS2_2Mix_Paper_v2.yml:89`）拿到的是 `torch.nn.BatchNorm1d`，**不是**本文件的 `bN`。同理 `act_type: "ReLU"` 走 `nn.ReLU`（`activations.py:54`），而 `"prelu"` 走本文件的 `prelu()`。大小写敏感。
10. **`cLNhw` 只接受 4D**：3D 输入在 `CumulateLN4D:179` `B,C,H,W = input.size()` 处 `ValueError`。而 `AudioBlock` 里所有 `ConvNormAct` 都传 `is2d=True`，所以现状没炸；一旦有人把 `is2d` 关掉，`norm_type: cLNhw` 立刻崩。
11. **`VideoBlock` 实际是"帧内"算子但配置里写了 `causal: true`**（`:100`），加上 E8 的 1/50 实测，说明"视频支路因果"这个说法在 as-built 上没有意义（它压根不跨帧）。
12. **`evaluated_mac_params.py` 的 MAC 口径**：脚本用 `ptflops.get_model_complexity_info`（`:88-96`），输入 `a=(1,32000)`、`v` 由视频模型产生（脚本习惯 `v=(1,1,50,64,64)` → 50 帧，因此它测的是 64×64 视觉口径）。本机无 GPU 权限跑它，本轮改用 CPU 复刻：ptflops 得到 **10.611 GMAC**（含 `aten.bmm`/LSTM 的乘加），`FlopCounterMode` 得到 5.382 GMAC（不记 `aten::lstm`）。两者差 5.23 G 全部来自 LSTM，2026-11-08 已定案，不再是 GAP。
13. **`utils/separator.py:57` 的能量重标定**：`out_wavs *= wav.abs().sum() / out_wavs.abs().sum()` —— 用 L1 总量做逐 utterance 增益校正。这段代码**全仓无调用**，但如果有人接上去做评测，会掩盖整体的幅度误差并污染 SI-SDR。
14. **`einops` 与手工 `permute` 混用**：范围内**没有 einops**（全仓 0 命中），全部是手工 `reshape/permute/transpose`。因此 §5-5、§5-6 这类布局 bug 没有形状断言兜底。`DualPathRNN:394` 的 `view(B*F_, C, T)` 依赖 `permute(...).contiguous()`，是这一族里唯一写对的（§4 已用 E8 的形状隐式验证）。
15. **in-place 操作**：`stft_tfgn.py:182` `output.masked_fill_(...)`（in-place，作用在 `transpose` 之后的非连续视图上——PyTorch 允许，但会打断 autograd 图，只在推理路径安全）；`_forward_stft:265-266` `frames[:, 1:] = ...`（in-place 覆盖 unfold 结果）；`dtype` 提升方面：`normalizations.py:117` `torch.from_numpy(entry_cnt).type(input.type())` 是唯一显式跟 dtype 对齐的地方，`fp64` 输入能正常工作（E2c 的 fp64 往返误差 0.0）。
16. **`torch.stft` 在 fp16 下直接不可用**（CPU：`MKL FFT doesn't support tensors of type: Half`；GPU 上 `torch.stft` fp16 也只支持 `n_fft` 是 2 的幂且走 cuFFT 的特定路径）。所以 `model.half()` 推理路径在当前实现下**必然崩**（E6f 实测）。

## 6. 与既有文档的冲突/更正

| # | 旧说法（出处） | 实际（本轮实测） |
|---|---|---|
| C1 | `docs/understanding/deepdive-20261024/02-layers-dsp-video.md` 标题写 "TFGN" | **仓库无 TF-GN**。`stft_tfgn.py` 无任何归一化，只是个 `Stft` 类；文件名的 `tfgn` 疑为 "tfgn" 的历史残留。"tfgn" 与 "TF-GN" 的关联是文档层面的误读。 |
| C2 | 同上 §1.1："`utils/stft.py` 与 `layers/stft.py` **逐字节同源**" | **不成立**。md5 不同（`533c481a…` vs `796ceb06…`），行数都是 785 但内容有差异。只能称"高度同源"。 |
| C3 | `docs/STATUS.md:50`："`cLNhw` 全窗 GN + AudioBlock 对称卷积，单遍 lookahead 实测 **57 帧≈456 ms**" | 数字对（我复现出 lookback 57 帧），但**归因不完整**：57 帧是"把 `cLNhw` 换成因果归一化后、一遍 AudioBlock"的单遍几何值。**as-built 含 `cLNhw` 时 lookback = lookahead = 全窗**（T=251 时 251/251 帧），不存在 57 帧这个界；整链几何是 113 帧 / 904 ms。`docs/understanding/01-model-core.md:99` 已经修正过一次（"as-built 是全局 (0,250)"），STATUS.md 未同步。**2026-11-08 Lead 复核：三档已写进 00–06；STATUS.md 本轮由 Lead 重写，旧 `:50` 的表述随之废止。** |
| C4 | `docs/understanding/01-model-core.md:32`："audio_bottleneck：cLNhw + Conv2d(112→112,k=1)" | 顺序需更正为 **pre_norm 在前**（`Falcon.py:99-101`：`pre_norm → pre_act → conv`）。`cLNhw` 归一化的是**编码器输出**，不是 conv 输出。这不是文字顺序问题：它决定了"全窗统计发生在 112 通道线性混合之前"。 |
| C5 | 论文/README："10.6G MACs" | **2026-11-08 Lead 复核更正**：ptflops 同输入返回 10.611 GMAC（含 LSTM、不含视觉），与论文数值逐位同尺，此处原判断"论文把 FLOPs 写成 MACs"作废。`FlopCounterMode` 的 5.382 GMAC 是另一把尺子（不记 `aten::lstm`），不可与 10.6 并列。整窗口径 26.51 GMAC。 |
| C6 | `docs/reviews/round1/02-layers-dsp-review-by-model-core.md:302`："实测 `sys.modules` 含 librosa/torch_complex/typeguard…" | 一致（我复核了这 6 个依赖都可用），并**补齐了链路**：`Falcon.py:9` → `layers/__init__.py:20` → `stft_tfgn.py:8`。 |
| C7 | 旧文未提 | **新增**：`cLNhw` 的 `GlobalLayerNorm` 在初始化态是数学空操作（`3.576e-07`），净增量只有 2×C 仿射参数。 |
| C8 | 旧文未提 | **新增**：`mask_act=ReLU` 使复数掩码实部/虚部同时非负 → 相位限死 [0°,90°]（实测 min 0.0000）。 |
| C9 | 旧文未提 | **新增**：`VideoBlock` LSTM 布局错误导致时间建模失效（脉冲响应 1/50 帧）。 |
| C10 | 旧文未提 | **新增**：`DPRNNLinear`、`Bottomup_Concat_Topdown_TCN` 两处死代码是**必然崩**（非"未测试"）。 |

## 7. GAP LIST

1. **（已结案）论文 "10.6G MACs" 的口径**：2026-11-08 Lead 复核后定案——ptflops 含 LSTM、不含 FRCNN，返回 10.611 GMAC，
   与论文 10.6 同尺；`FlopCounterMode` 的 5.382 GMAC 因不记 `aten::lstm` 而是另一把尺子。剩余未验项只有
   "论文是否也把即 88×88 视觉计入"（11 §3 判断为否）。
2. **`typeguard>=3` 的硬证据**：本轮只证明 2.13.3 可用且真在校验；"3.x 移除 `check_argument_types`" 依赖上游 changelog 与仓库内 stub 注释，**未在本机装 typeguard 3 复现**。需要一次隔离 venv 的复现实验（不改 `falcon` env）。
3. **`cLNhw` 在真实训练后的行为**：我只证明了"初始化态 GN 是空操作、affine 随机化后不是"。需要一个已训练 checkpoint（`/root/.cache/falcon_weights/falcon.pt`）来测 `|cLNhw(x) − cLN4D(x)|` 的真实值，判断训练是否让 GLN 段"活过来"。
4. **`process_f_layers` 沿频率轴双向 LSTM 的合理性**：`dim=4` 转置后 LSTM 扫的是 F（`Falcon.py:389`），双向在频率上没问题；但对**频率轴做双向 LSTM** 是否等价于论文的 "frequency-direction RNN"，需要与论文 App 对照（本文件范围外，交给 W1-01/W3）。
5. **`InjectionMultiSum` 的语义**：`Falcon.py:316-331` 里 `old_shape/new_shape` 用 `len(local_features.shape)//2` 取尾部维度，4D 输入时取的是 `(T,F)`，5D 时取的是 `(F,T,chunk)` —— 这套"按秩折半"的分支在 4D/5D 下的行为差异未做穷举验证（我只测了 4D）。
6. **多通道输入路径**：`STFTEncoder.unsqueeze_to_2D`(`:1102-1109`) 在 `x.ndim==3` 时断言 `s[1]==1`；Falcon 前向只喂 `(B,1,L)`。多通道（`(B,C,L)`）行为未测，理论上会被 reshape 成 `(B, C*L)` 而错。
7. **`torch.istft` 在 252 帧 vs 251 帧下的 `NOLA` 条件**：本模型固定 `length=32000` 截断，未产生告警；但若有人去掉 `length`（得到 32384）或改变 hop，需要核查 `Σw²` 是否处处 >0（`center=True` 下边缘是靠 pad 保证的）。
8. **`stft_tfgn.Stft.inverse` 的 `length=ilens.max()`**(`:242`)：批内变长时所有样本被拉到最长，短样本尾部是 `istft` 的零填充——多句批量评测若走这条死路径会引入尾部静音。需确认是否曾有人在评测里用它（目前 grep 为 0 调用，风险为 0）。
9. **`FlopCounterMode` 不含 `aten.interpolate`/`aten.copy`**：`AudioBlock` 的 top-down 有 12 次 `F.interpolate(..., mode="nearest")`，这些不计入 FLOPs 但实际有内存带宽开销。真实延迟剖分（GPU）需要另一套 profiling。
10. **Studio 侧的窗口调度**：README 说"每 0.5 s 重算 2 s 窗"。该 hop/窗长比例（4:1）对应的算力放大 16×，但 `studio/falcon_runtime.py` 不在本文件范围，需要在 Studio 轮确认它是否真的每 0.5 s 全量重算。

## 8. 证据清单

**代码（远端，只读）**

```
look2hear/layers/__init__.py        1-50
look2hear/layers/normalizations.py  9,11-20,44-48,51-61,74-88,91-132,134-143,146-163,165-202,204-221,223-229,231-245
look2hear/layers/activations.py     5-62
look2hear/layers/stft.py            10,15,29-111,221-286,289-354,357-471,474-607,610-696,699-785
look2hear/layers/stft_tfgn.py       8,23-57,99-111,160-166,168-184,188-245
look2hear/layers/cnnlayers.py       7-63,66-152,155-239,242-389,392-495,498-680,683-795,798-884
look2hear/layers/rnnlayers.py       15-105,109-580,586-637,640-771,774-908(尤其 882)
look2hear/layers/enc_dec.py         8-71,114-220,223-326,341-458
look2hear/utils/{stft,complex_utils,attn,inversible_interface,separator,nets_utils,torch_utils,get_layer_from_string}.py
look2hear/models/Falcon.py          9,11-109,241-331,333-417,419-545,693-898,900-976,978-1059,
                                    1061-1128,1130-1199,1201-1282,1284-1524
configs/LRS2_2Mix_Paper_v2.yml      18-110,111-172
```

**实测脚本（远端 `/tmp/w102/`，均不进仓库）**

| 脚本 | 覆盖 |
|---|---|
| `exp1.py` | E1（`forward_stft` vs `torch.stft`）、E2（往返/fp16） |
| `exp2.py` | E3（归一化轴语义、因果性、`get()` 分派） |
| `exp3.py` / `exp3b.py` | E5（模型实例化、参数分项、归一化清点、形状轨迹、卷积核表） |
| `exp4.py` | E9（typeguard / import 链 / `Stft` 布局） |
| `exp5.py` | E6（整链因果、位移等变、窗切分、fp16） |
| `exp6.py` / `exp7.py` | 受控替换 `cLNhw→cLN4D/LN4D`（L=4096 结论定性正确，E8 取代） |
| `exp8.py` | **E8**（隔离 AudioBlock/VideoBlock 感受野，核心实验） |
| `exp9.py` / `exp10.py` / `exp15.py` | E12（FLOPs、分项归因、手算校验） |
| `exp11.py` / `exp12.py` | E10/E11（mask 非负性、解码器帧记账、GN affine） |
| `exp13.py` / `exp14.py` | E13（死代码逐类参数/前向巡检、两处必崩 bug） |

**既有文档（已读，仅作对照，结论均已回到源码复核）**

```
docs/STATUS.md:21,50
docs/understanding/deepdive-20261024/02-layers-dsp-video.md（250 L，全文）
docs/understanding/01-model-core.md:32,95,98-99,147-148,179
docs/PHASE2_VOLCANO_LRS2_TRAINING.md:93
docs/reviews/round1/01-model-core-review-by-layers-dsp.md:170
docs/reviews/round1/02-layers-dsp-review-by-model-core.md:302,309
docs/verification/stubs/typeguard.py:1-10
```

**命令模式**

```bash
ssh -o BatchMode=yes dev_a800_2gpus '<cmd>'
cd /root/code/github_repos/Falcon-fork && \
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python /tmp/w102/expN.py
md5sum look2hear/layers/stft.py look2hear/utils/stft.py
grep -rn --include=\*.py "<symbol>" . | grep -v '^./docs/'   # 死代码引用计数
```
