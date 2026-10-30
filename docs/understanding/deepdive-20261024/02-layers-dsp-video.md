# 02 深挖 · 层与 DSP（STFT / TFGN / 归一化 / CNN / RNN / attention / 视频骨干）

> 编写：Lead（2026-10-24，接替未交付的 layers-dsp 子任务）· 仓库 `repro/phase2` HEAD `2beec38`
> 环境 conda `falcon`（Py3.11 / torch 2.8.0+cu128）· 复跑脚本 `docs/verification/scripts/verify_layers_dsp.py`（exit=0）
> 对照旧文档 `docs/understanding/02-layers-dsp.md`（895 行，Phase 0 产物）。
> 证据标记：`[代码]` `[验证]`（本次实跑）`[文档]`（旧文档/其他报告，未复跑）`[推断]`。

---

## 0. 一句话结论

**Falcon 的 DSP 主干是“STFT 编码 → 112 通道瓶颈 → 4 级 AudioBlock（每级与视频特征融合）→ TFRB（频向双向 LSTM +
时间向单向 LSTM + 因果 2D attention）→ 复数掩码 → 因果 STFT 解码”，全部组件定义在 `look2hear/models/Falcon.py`，
`look2hear/layers/**` 只贡献归一化/激活（+ 未被调用的卷积/RNN 库）。**

本次实跑确认（`verify_layers_dsp.py`，exit=0）：STFT `win=256/hop=128`、251 帧 × 129 频点、3 通道输入
`[real, imag, |X|]`；`cLNhw = CumulateLN4D(GroupNorm(1,C)(x))`（**先全窗口统计、再沿时间因果累积**）；
`DualPathRNN(dim=4)` 沿频率轴双向、`dim=3` 沿时间轴单向；`MHSA2D(causal=True)` 用 tril mask；
`Separator.__init__` 会 **pop 配置键** `repeats/shared`（Studio 用 deepcopy 规避）；
`look2hear/layers/enc_dec.py` 与 `rnnlayers.DPRNN*`、`cnnlayers.*` **在 Falcon 前向中无调用方**。

**对复现最重要的两条**：① “causal” 只能表述为**窗口级因果**——STFT `center=True` 有 ±128 采样（8 ms）帧级
lookahead，`cLNhw` 全窗口统计，AudioBlock 的下采样/融合卷积 `causal=False`（旧文档实测纯结构 lookahead
57 帧≈456 ms），因此 Studio 用 2 s 窗/0.5 s hop/80 ms guard 的分块重叠推理；② **不要为了“补架构意图”去改这些层**
（会破坏官方权重 bit-identical），`concat_layers[2]` 闲置、Comba 死分支、`enc_dec.py` 死代码都保持原样。

---

## 1. 系统边界与关键路径（file:line）

### 1.1 范围内文件与“是否真的被调用”

| 文件 | 行数 | 在 Falcon 前向中的角色 | 结论 |
|---|---|---|---|
| `look2hear/models/Falcon.py` | ~1492 | **真正的 DSP 全部在这里**：`STFTEncoder/STFTDecoder`、`AudioBlock`、`DualPathRNN`、`MultiHeadSelfAttention2D`、`InjectionMultiSum`、`MaskGenerator`、`Separator`、`Falcon/FalconSpeakerContrastive` | 核心 |
| `look2hear/layers/normalizations.py` | 245 | `cLNhw/cLN/gLN/LayerNormalization4D/BatchNorm1d` | 被大量使用 |
| `look2hear/layers/activations.py` | 67 | `PReLU/ReLU/Tanh/Sigmoid/GELU` 字符串解析 | 被使用 |
| `look2hear/layers/stft.py` | 785 | `forward_stft/inverse_stft`（librosa 卷积实现）+ `STFTBase` | **Falcon 不用**（用 `torch.stft`）；与 `utils/stft.py` 逐字节同源 |
| `look2hear/layers/stft_tfgn.py` | 244 | ESPnet 风格 `Stft`，`from typeguard import check_argument_types` | **Falcon 不用**；它是 `typeguard<3` 约束的来源 |
| `look2hear/layers/cnnlayers.py` | 884 | Conv-TasNet/FRCNN 块库（`Conv1DBlock`、`FRCNNBlock`…） | 全仓无外部调用方 |
| `look2hear/layers/rnnlayers.py` | 908 | Asteroid/SepFormer 式 `DPRNN*`、Conformer 件 | **与 Falcon 的 `DualPathRNN` 同名不同实现**，无调用方 |
| `look2hear/layers/enc_dec.py` | 461 | Asteroid filterbank `Encoder/Decoder/FreeFB` | **无调用方**（本次 `[11]` 实测：Falcon encoder 是 `models.Falcon.STFTEncoder`） |
| `look2hear/utils/{attn,complex_utils,stft,nets_utils}.py` | — | `complex_utils` 被模型/评测用到；`attn` 未被 Falcon 使用；`nets_utils` 里 espnet import 是死路径 | 部分使用 |
| `look2hear/videomodels/frcnn_videomodel.py` | ~200 | **冻结的唇读编码器**（ResNet18 前端 + `frontend3D`） | 被使用（11,186,688 参数、全冻结） |

### 1.2 端到端形状（`configs/Falcon.yml`，本次 `[6]` 实测）

```
输入 audio [1,32000] + mouth [1,1,50,88,88]
STFTEncoder          → (1,112,251,129)      # T=1+32000/128=251, F=256/2+1=129
audio_bottleneck     → (1,112,251,129)      # cLNhw + 1×1 conv
video_bottleneck     → (1,512,50)           # kernel_size<0 → Identity
Separator(AudioBlock) → (1,112,251,129)     # 4 级金字塔 + 视频融合 + TFRB
MaskGenerator        → (1,1,112,251,129)    # ReLU 掩码（n_src=1）
STFTDecoder          → (1,1,32000)          # causal convtranspose + torch.istft
```

子模块参数（本次实测，合计 **523,113**，不含训练期对比头 14,584）：
`STFTEncoder 3,024 + audio_bottleneck 13,104 + Separator 492,312 + MaskGenerator 12,657 + STFTDecoder 2,016`。

### 1.3 三条“同名不同义”的调用链（复现时最容易踩）

1. **`layers/stft.py` ≠ Falcon 用的 STFT**：Falcon 在 `STFTEncoder.forward` 里直接 `torch.stft(..., return_complex=True)`，
   再堆 `[real, imag, |X|]` 三通道（`Falcon.py:1107-1119`）——不是 `layers/stft.py` 的 librosa 卷积实现。
2. **`rnnlayers.DPRNN` ≠ Falcon 的 `DualPathRNN`**：前者是 SepFormer 式 chunk RNN，后者是“频轴双向 + 时轴单向”的
   双路径 LSTM（`Falcon.py:331-415`）。
3. **`cnnlayers` 里的 `ConvNormAct` ≠ `models/Falcon.py:9` 的 `ConvNormAct`**：同名不同实现（旧文档 §4.8 已列）。

---

## 2. 关键事实与数字（命令 + 实际输出）

复跑命令（只读、CPU、<60 s）：

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/verify_layers_dsp.py
# exit=0；日志本次存于 /root/tmp/lead-verify-layers.log（11 个 section）
```

### 2.1 STFT 参数与帧级 lookahead（`[1][3]`）

| 项 | 值 | 证据 |
|---|---|---|
| `win / hop_length` | **256 / 128**（`configs/Falcon.yml` `enc_dec_params`） | `[代码]` + `[验证]` `[0]` |
| 帧数 / 频点 | `1 + 32000//128 = 251`；`256//2 + 1 = 129` | `[验证]` `[1][2]` |
| `center=True` 的帧级 lookahead | 帧 t 覆盖采样 `[128t-128, 128t+128)` ⇒ 最多依赖未来 **128 采样 = 8 ms** | `[验证]` `[3]`：脉冲 p=20000 → 受影响帧 `[156,157]` |
| 与 `torch.stft` 等价性 | `forward_stft`（librosa 卷积）vs `torch.stft`：`max|diff|=2.289e-05`；`mode='torch'` 时 **0** | `[验证]` `[1]` |
| iSTFT 重建 | librosa 路径 vs `torch.istft` `2.146e-06`；round-trip `2.384e-06 / 9.537e-07` | `[验证]` `[2]` |
| 解码器帧记账 | mask 251 帧 → causal pad 253 → convtranspose 253 → crop 252 → iSTFT 得 32128 采样 → 截断 32000 | `[验证]` `[6b]` |

### 2.2 编码器/解码器的因果性（`[7][7b]`）

| 模块 | 后半段扰动下的早期段变化 | 结论 |
|---|---|---|
| `STFTEncoder` | `T<120: max|Δ|=0`；`T>=120: 113.9` | conv 子层严格因果；但 `torch.stft(center=True)` 仍有 8 ms 帧级 lookahead |
| `audio_bottleneck`（cLNhw） | `T<120: 4.739e-05` | 全窗口统计造成微小泄漏（被 `CumulateLN4D` 部分抵消） |
| `Separator` | `T<120: 1.706` | **非因果**（cLNhw + causal=False 卷积 + 视频融合） |
| `STFTDecoder` 自身 | `samples[0:15360]: 0`；`[15360:16000]: 0.63` | convtranspose 严格因果裁剪；`torch.istft(center=True)` 使扰动扩散 ±win/2 |

> 与旧文档一致：**“结构单向”与“逐采样因果”是两件事**；as-built 端到端首变样本 9,612（相对扰动点 −649 ms）
> 来自 AudioBlock 对称卷积 + 全窗归一化的叠加（01 §2.7 已给同口径数字）。

### 2.3 归一化：`cLNhw` 的精确语义（`[4]`）

| 组件 | 语义 | 实测 |
|---|---|---|
| `GlobalLayerNorm` (`gLN`) | `nn.GroupNorm(1, C)`：统计维度 = 跨 `(C,T,F)` 的**全窗口**统计 | `[验证]`：与手算 `mean over C,T,F` 差 `2.38e-07` |
| `CumulateLN4D` (`cLN`) | 统计 `t' <= t` 的帧（跨 C），**逐频点**独立 ⇒ 时间上因果累积 | `[验证]`：与手算一致 |
| **`cLNhw`**（Falcon 配置） | **`CumulateLN4D(GroupNorm(1,C)(x))`**：先全窗 GN，再沿时间累积 LN | `[验证]`：复合等价 `True` |
| `LayerNormalization4D((C,1))` | per-channel LN（逐帧、跨 C） | `[验证]` 等价 `LN(dim=1)` |
| `LayerNormalization4D((C,F))` | 跨 C 与 F、逐帧独立 | `[验证]` 等价 `LN(dim=(1,3))` |
| `BatchNorm1d` | 配置里用的是 **torch 原生** `nn.BatchNorm1d`（不是本仓库包装类） | `[验证]` |

关键推论：**`cLNhw` 的 GN 部分是“无界依赖整个窗口”的**，论文配置下 `cLNhw` 实例 30 个
（`audio_bottleneck 1 + projection 1 + downsample 4 + fusion/concat 3×8`，旧文档 `[V7]`），
所以“全窗统计”不是个别层的瑕疵，而是音频主干每层都有 ⇒ 只能窗口化推理。

### 2.4 TFRB 的真实实现（`[8][9]`）

| README 组件名 | 代码 | 方向性/细节 |
|---|---|---|
| **TFRB** 频率向 | `DualPathRNN(dim=4, hid_chan=24, bidirectional=True, rnn_type=LSTM)`（`Falcon.py:331-415`） | 输入 `[B,C,T,F]`→转置成 `[B,C,F,T]`→`batch=B·T`、序列=F ⇒ **沿频率轴双向**（频带可同时可见） |
| **TFRB** 时间向 | `DualPathRNN(dim=3, hid_chan=72, bidirectional=False)` | `batch=B·F`、序列=T ⇒ **沿时间轴单向（因果）** |
| attention | `MultiHeadSelfAttention2D`（`Falcon.py:417-543`），4 head、`tril` mask | `causal=True`：`t' > t` 置 `-inf`；`causal=False` 时早期帧也变化（实测 `5.100`） |
| GCI | `InjectionMultiSum`（`Falcon.py:267-329`） | `local_emb(x_local) * Sigmoid(gate(x_global→interp)) + global_emb(...)`；`interpolate(nearest)` 只取过去/当前源，**非因果致因是 5×5 对称卷积** |

`[验证]` `[8]`：`DualPathRNN dim=4/dim=3` 与 `MHSA2D(causal=True)` 在 `t<5` 的扰动响应均为 **0**；`causal=False` 的
MHSA2D 为 `5.100`；`cLNhw` 单层为 `1.202e-03`（全窗统计泄漏，量级被后续 `CumulateLN4D` 抵消）。

`[验证]` `[9]` 细节：`out_proj` 分别为 `Linear(48→64)`（2×24）与 `Linear(72→64)`；
**`DualPathRNN` 的默认 `rnn_type="Comba"` 在 torch 中不存在**（`AttributeError`），只有配置显式传 `LSTM` 才能构造
——这是 flash-linear-attention 分支的残留（`kernel_size/stride/num_heads/...` 全是死参数）。

### 2.5 视频骨干 FRCNN（冻结，`[代码]` + 旧文档 `[文档]`）

| 项 | 值 | 证据 |
|---|---|---|
| 类 | `look2hear/videomodels/frcnn_videomodel.py:FRCNNVideoModel`（`videonet_name: FRCNNVideoModel`） | `[代码]` |
| 主干 | `frontend3D = Conv3d(1→64, k=(5,7,7), stride=(1,2,2), pad=(2,3,3)) + BN3d + PReLU + MaxPool3d((1,3,3),(1,2,2))` + `ResNet(BasicBlock,[2,2,2,2])` | `[代码]` `frcnn_videomodel.py:58-70` |
| 输入/输出 | `[B,1,50,88,88]` → `[B,512,50]`（时间维 50 帧 = 2 s@25 fps 保持不变） | 旧文档 T5 `[文档]` |
| 参数量/冻结 | **11,186,688**，`requires_grad=False` 全冻结 | 01 §2.1；Phase1 T5 |
| 时间感受野 | `frontend3D` 的 `Conv3d k=5, pad=2` ⇒ **±2 视频帧 = ±80 ms** lookahead | 旧文档 §4.2 交叉验证（改帧 ≥30 → 最左受影响输出帧 28） |
| 训练/推理差异 | `train()` 覆写：**保持 BN running stats 冻结**并 `return self`（Phase 1 修复 D8） | `[代码]` `frcnn_videomodel.py:95+` |
| 权重加载 | `init_from(path)` 读 `{"model_state_dict": ...}`，`update_frcnn_parameter` 过滤前缀 | `[代码]` |
| 权重来源 | 从官方 `falcon.pt` 的 `video_model.*`（137 keys）导出为 `frcnn_128_512.backbone.pth.tar` | Phase1 F12 `[文档]` |

### 2.6 Studio 副本的 DSP 差异（`[代码]` + 旧文档 `[文档]`）

- `studio/falcon_model/` 只保留 `layers/{__init__,normalizations,activations}.py`，**没有 `stft.py/stft_tfgn.py`**
  ⇒ 不需要 librosa/torch_complex/typeguard/espnet；`timm` 被替换成本地 `drop_path.py`（eval 下恒等）。
- **CPU FFT bridge**：`STFTEncoder/STFTDecoder` 在 `x.device.type=="mps"` 时把 STFT/iSTFT 搬到 CPU 计算再搬回
  （`studio/falcon_model/models/Falcon.py:1111-1122, 1183-1198`）；除设备搬运外无数值改动 ⇒ 在 CPU 上与训练侧
  **bit-identical（max|Δ|=0）**（06 §3.2 本次已用真实权重复验）。
- `studio/falcon_runtime.py:149-150` 对 config 做 `deepcopy`，原因正是 `Separator.__init__` 会 `pop`
  `repeats/shared`（本次 `[10]` 实测：构造后 `audio_params` 少了 `repeats/shared`，模型参数从 537,697 变残缺）。

### 2.7 依赖与导入代价

- **`typeguard` 必须 <3**：`look2hear/layers/stft_tfgn.py:8,34` 用 `check_argument_types()`（4.x 已删除）。
  Falcon 主线不调用该模块，但 `look2hear.layers.__init__` 会导入它 ⇒ **升级 typeguard 会导致 `import look2hear.models` 失败**。
- `look2hear/models/__init__.py` 全量导入会把 `layers/stft.py`（librosa/torch_complex）、`videomodels`（timm/opencv）、
  `datas`（soundfile 等）一起拉进来——这是“import 慢/依赖重”的根因，也是 Studio 需要独立副本的原因。

---

## 3. 当前缺陷与风险（P0/P1）

| # | 级别 | 问题 | 状态 | 影响 | 证据 |
|---|---|---|---|---|---|
| L1 | 属性 | **非严格逐采样因果**：STFT ±128 采样、`cLNhw` 全窗 GN、AudioBlock 对称卷积 `causal=False` | 架构属性（权重依赖） | 不能宣称“sample-wise streaming / 0 延迟”；Studio 必须窗口化 | §2.2/§2.3；01 §2.7 |
| L2 | P2 | `DualPathRNN` 默认 `rnn_type="Comba"` 不存在；`kernel_size/stride/num_heads...` 死参数 | 仍在（配置显式 LSTM 规避） | 任何“按默认构造”的脚本会 AttributeError | `[验证]` `[9]` |
| L3 | P2 | `Separator.__init__` 原地 `pop` 配置键 | 仍在（Studio/入口 deepcopy 规避） | 同一 dict 二次建模型 → 参数残缺（537,697→49,081） | `[验证]` `[10]`；01 D4 |
| L4 | P2 | `layers/stft.py` 与 `utils/stft.py` 同源复制、`stft_tfgn.py` 是 espnet 复制件 | 仍在 | 双份维护；typeguard<3 约束 | 旧文档 §2.12 |
| L5 | P3 | `layers/enc_dec.py`、`rnnlayers.DPRNN*`、`cnnlayers.*` 全仓无调用方 | 仍在（死代码） | 误读风险；`DPRNNLinear.forward` 本身形状不匹配不可用 | `[验证]` `[11]`；旧文档 §2.11 |
| L6 | 属性 | `VideoBlock` 在 **train 模式**下 `BatchNorm1d` 用批内统计（含未来帧） | 架构属性 | 训练期非因果（推理 eval 因果） | 旧文档 `[V4b]` `[文档]` |
| L7 | P3 | `FRCNNVideoModel.train()` 曾经无 `return self` | **已修（Phase 1）** | 无 | `[代码]`；01 D8 |
| L8 | P3 | `timm`/`opencv`/`librosa` 被 `look2hear.layers/__init__` 连带导入 | 仍在 | 环境重、Studio 必须独立副本 | §2.7 |

---

## 4. 与旧文档（`docs/understanding/02-layers-dsp.md`）的差异清单

| 旧文档结论 | 本次复核 | 判定 |
|---|---|---|
| Falcon 前向链/形状/参数（§3.3、§6） | `verify_layers_dsp.py` exit=0，`[6]` 形状/参数逐项一致 | ✅ 确认 |
| STFT `win=256/hop=128`、251×129、±128 采样 lookahead（§4.1） | `[1][3]` 实测一致 | ✅ 确认 |
| `cLNhw = CumulateLN4D(GroupNorm(1,C))`（§4.4） | `[4]` 复合等价 `True`；单层泄漏 `1.202e-03` | ✅ 确认 |
| TFRB = 频轴双向 + 时轴单向 + tril-mask attention（§4.3） | `[8][9]` 方向性/参数/out_proj 全部一致 | ✅ 确认 |
| AudioBlock lookahead 57 帧≈456 ms（R3 §[V4]） | 本次未重跑该单遍实验（脚本不含），引用 01 §2.7 同口径 | ✅ 保持（未重跑） |
| `layers/stft.py` 与 `utils/stft.py` 逐字节同源（§2.12） | 本次未 diff（旧文档有 md5 证据）；两者均无 Falcon 调用 | ✅ 保持（未重跑） |
| `enc_dec.py`/`rnnlayers`/`cnnlayers` 无调用方（§2.9–2.11） | `[11]` 确认 Falcon encoder/decoder 是 `models.Falcon` 内部类 | ✅ 确认（旧文档对 `enc_dec` 的结论由本次实测坐实） |
| 旧 R1 曾写“严格因果”，R3 改为“窗口级因果” | 本次以 `[7][7b]` 再次确认“结构单向 ≠ 逐采样因果” | ✅ 以 R3 表述为准 |
| “Studio 副本在 CPU 上与训练侧 bit-identical”（§4.6） | 06 本次用真实权重复验 `max|Δ|=0` | ✅ 确认 |
| 旧文档行号基于 `7576854` | HEAD `2beec38` 行号有偏移（如 `Falcon.py` 1492 行、`frcnn_videomodel.py:58-70`） | ⚠️ 修正：引用以本文行号为准 |

**本轮新增/强化的结论**：
1. `verify_layers_dsp.py` 在真实 falcon 环境 exit=0（旧文档 `runs/layers-dsp-verify.log` 之外的新鲜证据）；
   注意该脚本会把 `docs/verification/stubs` 放进 `sys.path`（日志显示 `typeguard/timm -> stubs`），
   所以它验证的是“数值语义”，**不能替代**真实依赖环境的 import 检查（真实环境 import 由 01/03 验证）。
2. `[7]` 的逐层泄漏数字（Encoder 0 / bottleneck 4.7e-05 / Separator 1.706）给“窗口级因果”提供了按层分解。
3. `[9]` 坐实 Comba 死分支；`[10]` 坐实 `Separator` pop 配置是 Studio deepcopy 的直接原因。

---

## 5. 对复现的影响与建议

1. **表述统一**（写入最终报告/README 引用）：Falcon 是“**窗口级因果（window-causal）+ 单次 encoder–decoder +
   无跨窗状态**”，不是逐采样流式；证据：STFT ±128 采样、`cLNhw` 全窗 GN、AudioBlock 对称卷积、
   Studio 2 s/0.5 s/80 ms 分块。禁止写“严格因果/0 延迟”。
2. **不要改层来“修因果”**：任何把 `causal=False` 改 `True`、把 `cLNhw` 换成纯因果归一化、或补 `concat_layers[2]`
   调用的改动都会改变官方权重语义（bit-identical 被破坏），只能另开分支并重训/重评（01 §5.6）。
3. **复现训练时保持**：`win=256/hop=128`、`n_freqs=129`、`out_chan=112`、`hid_chan=64`、`repeats=2/shared=true`、
   `upsampling_depth=4`、`layer_1 dim=4 bidir`、`layer_2 dim=3 unidir`、`layer_3 MHSA causal`——5 个配置的
   `audionet` 段逐键相同（01 §5.5），8 卡训练出的 ckpt 与官方同 key/shape，可直接 strict 加载。
4. **评测别走 `look2hear/utils/separator.py`**：它会对输出做幅度重缩放（01 §5.6 第 4 条），污染 SI-SNRi/SDRi。
5. **Studio/部署**：保持“固定 2 s 窗 + 0.5 s hop + 80 ms guard”；CPU 单窗 ~543 ms（06 §3.4）> hop，
   本机 CPU 无法实时；GPU A800 单窗 p50 47.1 ms（small）/56.5 ms（big）（06 §3.4）。
6. **若要做 TFRB/GCI 消融**：只作为“架构意图”实验另开分支，不参与主表；论文未公开消融设置，结果只能定性。

---

## 6. 未解问题

| # | 问题 | 现状 | 需要谁 |
|---|---|---|---|
| Q1 | `concat_layers[2]`（5,568 参数）是否上游漏调用？ | 代码中 level-2 融合层从未被调用；Phase 1 为保 bit-identical 有意不修 | 作者确认 |
| Q2 | 论文“causal”是否指窗口级？ | 只能从代码/Studio 推断；论文未公开 | 论文/作者 |
| Q3 | `cLNhw` 的全窗统计在训练时是否有正则/稳定性作用？ | 未做消融（会破坏权重语义） | 研究性问题 |
| Q4 | `typeguard>=3` 是否可通过改 `stft_tfgn.py` 解除？ | 该模块不在 Falcon 前向链，可改但属源码修改（需授权） | 用户决策 |
| Q5 | FRCNN 时间感受野 ±2 帧是否与 LRS2 mouth 对齐误差叠加？ | P2-F 已测 mouth ±2 帧 −5~−6 dB；FRCNN 自身 ±80 ms 未单独消融 | 可选实验 |

---

## 7. 验证记录（本次实跑）

| # | 命令/脚本 | 结果 |
|---|---|---|
| 1 | `PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/verify_layers_dsp.py` | **exit=0**；11 个 section 全部输出（日志 `/root/tmp/lead-verify-layers.log`） |
| 2 | `[1]` forward_stft vs torch.stft | `max|diff|=2.289e-05`；`mode='torch'` 时 0；251 帧/129 频点 |
| 3 | `[2]` iSTFT / round-trip | `2.146e-06` / `2.384e-06`（librosa）、`9.537e-07`（torch） |
| 4 | `[3]` STFT lookahead | 脉冲 p=20000 → 帧 `[156,157]`；≤128 采样=8 ms |
| 5 | `[4]` 归一化语义 | `GlobalLayerNorm == GroupNorm(1,C)`、`cLNhw == CumulateLN4D(GroupNorm)`、`LN4D` 广播等价 |
| 6 | `[6]` 前向形状/参数 | 形状链见 §1.2；`523,113`（audio 子模块） |
| 7 | `[7][7b]` 因果探测 | Encoder 0 / bottleneck 4.739e-05 / Separator 1.706；Decoder 自身 `[0:15360]=0` |
| 8 | `[8]` 子模块因果 | DualPathRNN 早期 0；MHSA2D(causal) 0 / (non-causal) 5.100；cLNhw 1.202e-03 |
| 9 | `[9]` DualPathRNN 参数 | `out_proj 48→64 / 72→64`；Comba 默认 AttributeError |
| 10 | `[10]` Separator pop | `repeats/shared` 被删除（audio/video params 均少 2 键） |
| 11 | `[11]` enc_dec 归属 | Falcon encoder/decoder = `models.Falcon.STFTEncoder/STFTDecoder`；`layers/enc_dec.py` 无调用方 |
| 12 | `git log -- look2hear/layers look2hear/videomodels` | Phase 1 仅 `videomodels +7 行`（`train()` return self）；Phase 2 零改动（`[文档]`，01 §4.1 已核） |
