# 12 因果性与延迟

> **2026-11-10 逐篇复核提示：**本篇记录特定权重、输入、阈值和窗口下的实验。57/113 帧来自替换归一化后的几何实验；原模型存在全窗统计依赖。默认 VideoBlock 的 LSTM 将批次轴当序列处理；520 ms 视觉系统值亦仅是该实验和估算。127/377 ms 由 guard、块位置和一次前向耗时估算；采集至播放的延迟仍待实测；公式将几何前视与窗口缓存直接相加也会重复计算或混淆调度。6 秒实验的输出写 11 窗与覆盖样本 0–55000，两者按 0.5 秒块长并不一致，实际覆盖范围有待核对；不同窗长与相位的相关性或 SI-SNR 描述输出一致性；误差归因、感知质量和实时可用性需分别测量。论文 15 帧的推导来源和 80 ms guard 的设计动机仍待确认。详见 [当前状态](../../STATUS.md)。

> 子代理 W2-12。仓库 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork` @ `6c1d5b7`（只读）。
> 全部实测在 **CPU**（`CUDA_VISIBLE_DEVICES=` 置空）、真实权重 `falcon.pt`、`falcon` conda 环境、
> 单窗 32000 样本（2 s）/ 251 帧 / `no_grad`。脚本全部在 `/tmp/w212/`（本机同副本）。
> 论文原话取自 `https://arxiv.org/html/2610.32749v1`（`/tmp/falcon_paper.html`，App E）。

---

## 0. 结论速览

1. **113 帧 / 904 ms**：单窗 2 s 内，输出第 *t* 帧在几何上依赖输入到 *t*+113 帧（实测 113，解析 115）。分解：`STFT +0`（样本级 +1 帧）→ 编码器阶梯 15 帧/120 ms（论文声明的 120 ms 预算就是这个量级，见第 4 条）→ 一遍 AudioBlock +57 帧/456 ms → 第二遍再 +56 帧。
2. **30 处 `cLNhw` 把依赖推到窗尾**：`cLNhw`（= `GroupNorm(1,C)`，覆盖 C×T×F）在音频路径出现 30 处，把整窗统计量灌进每一帧。只在窗尾（帧 245）之后注入 1e-3 噪声，帧 0 仍变化 −52 dB；换成逐 (t,f) 的 `LN_c` 后记录的帧 0、20、40、80、120 为 −240 dB；该区间最大值为 −75.2895 dB，回落到 113 帧。
3. **1 种口径成立：窗口级因果**：窗内看未来（上两档），跨窗无状态。窗内既不是因果，也不等于论文声明的 120 ms。
4. **15 帧 / 120 ms 只能由对称 padding 复现**：论文 §6.4 声明音频支路「at most 15 frames」、App E 声明视觉「3 future frames = 120 ms」，而代码是对称 padding（`causal=False`）。15 = `2^4−1` 恰好等于编码器最深层的累计 lookahead；这个数只覆盖编码器单层，不含 GCI/decoder（+44 帧）与第二遍（+56 帧）。
5. **视觉 ±2 帧 / 80 ms，系统级 13 帧 / 520 ms**：`Conv3d(kernel=(5,7,7), padding=(2,3,3))` 的时间感受野是 ±2 视频帧；论文声明 3 帧/120 ms，代码实测 2 帧/80 ms。系统级视觉需求 = 2（FRCNN）+ 11.4（第一遍 57 音频帧折算）≈ 13 视频帧 = 520 ms；as-shipped 时视频帧 0…49 全部影响输出。
6. **0 个跨窗状态，但跨窗不连续**：无缓存、无 `state`、无 RNN 隐状态传递。6 s 流（3×2 s 片段拼、join 处 10 ms 交叉淡化，Studio 从 t=0 开窗）按 Studio 方案拼接 vs 整段一次前向 = **SI-SNR +14.71 dB、corr 0.9835、最大偏差 53.4% 峰值**；**剔除冷启动首窗（窗内前 1.42 s 是静音）为 +16.72 dB / 0.9895 / 28.5%**。同一段 0.5 s 内容换窗对齐（块恒在 Studio 的 `[22720,30720)`，相位 ±250 ms），与整段一致性只在 15.86–17.99 dB 间波动，**任两种对齐互相 ~17 dB**；旧稿的 8.69 dB 是"窗尾（guard=0）vs 窗中（guard=250 ms）"的单块比较，混了 guard 与相位两个变量，不能当对齐代价。
7. **80 ms guard 对 456–904 ms 的需求，但 guard 是次要旋钮**：要分两种语义量——**"丢弃窗尾"（Studio 实际可调的就是这个）** 80→456 ms：14.51→16.71 dB（4 s：13.73→17.58），**非单调**，且这部分增益来自窗口位置改变而非补未来；**"多喂右上下文"（窗口 = 2 s + guard）** 80 ms 就已 22.12 dB，加到 640 ms 仅 22.92（**+0.8**），**基本饱和**。几何上块 `[178,240]` 帧相对 113 帧（904 ms）lookahead **全部不足**（帧 ≤137 才算足上下文，块里一帧都没有），但实测代价只有 ~0.8 dB——几何"上下文饥饿"与可听/可测质量不是一回事。
8. **127 ms 墙钟 vs 904 ms 算法**：`L_e2e ≈ L_algo + GUARD(80 ms) + 块粒度(0…500 ms) + F + I/O`。A800 单窗 F = 47.1 ms（引用 `deepdive-20261024/06-studio-backend.md:402`，本轮按硬规则未跑 GPU），CPU 4 线程 F ≈ 0.6 s（实测 592–694 ms；FRCNN 119.2 + 音频 473.2）。最好情况 127 ms（A800）、平均 ≈377 ms；CPU 最好 672 ms，Studio 自报 RTF = 1.18，不能实时。
9. **两个 120 ms 不是同一个量**：wall-clock 能到 127 ms，是因为 Studio 只喂 80 ms 未来；架构自身需要 904 ms。论文的 120 ms 指视觉前端 lookahead（代码里只有 80 ms），与端到端延迟不能换算。
10. **差距从大到小**：`cLNhw` 全窗统计（→ 2000 ms）> 两遍 AudioBlock（+56 帧）> 对称 padding 阶梯（+15 帧@bottleneck、逐级累计 +59；论文声明里那个 120 ms 就出自这一级）。

## 1. 方法与口径

### 1.1 度量定义（全文统一，先定义再看数）

设被考察算子在时刻 *j* 的输入被改，输出 diff 的非零区间为 `[lo, hi]`：

| 量 | 定义 | 含义 |
|---|---|---|
| **lookahead** | `j − lo`（帧） | 输出**早于**扰动位置的部分也变了 ⇒ 该算子看未来。这是"因果性"的唯一判据 |
| **memory / 回看** | `hi − j` 或 `j − lo`（视方向） | 输出依赖多久的过去（不是因果性问题） |
| **lag（延迟偏移）** | `lo − j` > 0 | 输出整体后移，是"延迟"不是"看未来" |

> 这一步很关键：很多表格把 `hi−j`（前向传播跨度 = memory）当成 lookahead 报出来，会把因果算子误判成非因果。
> 本文所有 lookahead 都是 `j − lo`。

### 1.2 帧/样本换算

| 量 | 值 | 出处 |
|---|---|---|
| 采样率 | 16000 | `configs/Falcon.yml`（`sample_rate`） |
| `win` / `hop_length` | 256 / 128 | `configs/Falcon.yml:28-29` |
| 1 帧 | 128 样本 = **8 ms** | 计算 |
| 2 s 窗帧数 | `T = 1 + 32000/128 = 251` | 实测 `torch.stft` 输出 `n_frames=251` |
| 视频帧率 / 每视频帧对应音频帧 | 25 fps / **5 音频帧** | `studio/falcon_runtime.py:29-31`；实测 `caf` 1→5 |
| 1 视频帧 | 40 ms | 计算 |

### 1.3 四条证据线（互相独立、互相校验）

| 线 | 做法 | 作用 |
|---|---|---|
| **A 算子隔离** | 取**真实子模块**（`a.encoder`、`a.separator.audio_net` 内部各层…），喂随机张量，改 1 帧（+1.0），diff 输出 | 单算子因果性表 |
| **B 几何消融 `deNORM`** | deepcopy 整个模型，把全部 `LayerNormalizationHW/GlobalLayerNorm/BatchNorm/CumulateLN(4D)/LayerNormalization4D` 换成 `Identity` | 剥掉归一化后剩下的**纯卷积/循环几何**因果性（"若按论文 LN_c 实现会怎样"的反事实） |
| **C 端到端梯度锥** | `torch.autograd.grad(y[0,0,s], x)`，按 128 样本折帧取 RMS，对 `thr·max` 取支撑 | 真实权重下的精确依赖锥（含所有非线性） |
| **D 解析区间传播** | 用区间 `[L(j), U(j)]` 逐层推（conv/pad/stride/nearest-upsample/uni-LSTM/tril-attn），并追踪**名义对齐** `ref[j]` | 给出闭式公式，与 A/B 对账 |

**口径警告（随数字一起引用才不会被读错）**：
- `deNORM` / `LN_c` 替换**改变了函数本身**（不是等变变换），只能用来判定**结构性因果**，不能用来评价质量。
- 所有端到端数字都是 **单窗 32000 样本**；模型的 `cLNhw` 统计量随窗长变化，所以"3 s 窗"和"2 s 窗"的输出本来就不同（§4.7 正是利用这一点）。
- CPU 单窗音频前向 **523 ms**（warm，median of 3），FRCNN **119.2 ms**，故端到端实验一律用小样本次数，靠梯度而非暴力扫描。

### 1.4 复现命令骨架

```bash
ssh dev_a800_2gpus 'cd /tmp/w212 && CUDA_VISIBLE_DEVICES= \
  /root/miniforge3/envs/falcon/bin/python <script>.py'
```
`w212_common.py` 用 `studio/configs/Falcon.yml` + `studio/falcon_model/models/Falcon_Contrastive.py` 建模型，
`strict=True` 载入 `falcon.pt` 的 `audio_model.*` / `video_model.*`（与 `studio/falcon_runtime.py:200-212` 同一条路径）。

---

## 2. 逐算子因果性表

### 2.1 主表

`是否看未来` 分两列：**几何**（卷积/循环结构本身）与 **as-shipped**（叠加 `cLNhw` 之后，实测）。
"看未来多少" 以**原帧**计（1 帧 = 8 ms）。

| # | 算子 | 位置 `path:line` | 几何看未来 | as-shipped 看未来 | 跨窗 |
|---|---|---|---|---|---|
| 1 | `torch.stft(center=True)` | `look2hear/models/Falcon.py:1117-1123` | **是，+1 帧（128 样本/8 ms）** | 同 | 否 |
| 2 | `STFTEncoder.conv`（k=3, stride=1, `causal=True`） | 构造 `Falcon.py:1087-1096`；causal pad+trim `Falcon.py:48-52,102-108` | **否**（0 帧）；覆盖 `t−2..t`，中心滞后 1 帧 | 同 | 否 |
| 3 | `audio_bottleneck`（k=1 conv，`pre_norm_type=cLNhw`） | 构造 `Falcon.py:1322`；cfg `configs/Falcon.yml:21-26` | 否（k=1） | **是，整窗**（实测 −125…+125） | 否 |
| 4 | `AudioBlock.gateway`（k=1 depthwise，无 norm） | `Falcon.py:720-728` | 否 | 否（实测 0） | 否 |
| 5 | `AudioBlock.projection`（k=1，`norm_type=cLNhw`） | `Falcon.py:729-737` | 否 | **是，整窗** | 否 |
| 6 | `downsample_layers[0]`（k=3,s=1,pad=1,`causal=False`） | `Falcon.py:758-776`（`cur_stride=1`,`cur_kernel_size=3`,`causal=False`） | **是，+1 帧** | 整窗（norm） | 否 |
| 7 | `downsample_layers[1..3]`（k=5,s=2,pad=2,`causal=False`） | 同上 | **是，+2 该层帧 = +2·2^l 原帧** | 整窗（norm） | 否 |
| 8 | `process_f_layers[i]` = `DualPathRNN(dim=4, bidirectional=True)` | `Falcon.py:779-781`；cfg `:52-57` | **否**（序列轴是**频率**，不是时间；实测 0） | 否（norm 是 `LayerNormalization4D`，逐帧） | 否 |
| 9 | `process_t_layers[i]` = `DualPathRNN(dim=3, bidirectional=False)` | `Falcon.py:782-784`；cfg `:60-65` | **否**（单向 LSTM 沿时间，`Falcon.py:387-417`） | 否 | 否 |
| 10 | `process_att_layers[i]` = `MultiHeadSelfAttention2D(causal=True)` | `Falcon.py:790-794`；tril mask `Falcon.py:522-526` | **否**（`t'≤t` 掩码，实测 0；关掉 causal 立刻变 −16 帧） | 否 | 否 |
| 11 | `fusion_layers[i]` = `InjectionMultiSum`（k=5,pad=2,`causal=False`） | `Falcon.py:797-808`；实参 `Falcon.py:322-323` | **是，+2 该层帧** | 整窗（norm） | 否 |
| 12 | `concat_layers[i]` = 同 `InjectionMultiSum` | `Falcon.py:810-819` | **是，+2 该层帧** | 整窗（norm） | 否 |
| 13 | 上采样 `F.interpolate(mode="nearest")` | `Falcon.py:322-327, 895` | 否（`out[j]←in[⌊j·n_in/n_out⌋]`） | 同 | 否 |
| 14 | `residual_conv`（k=1，`causal=False`） | `Falcon.py:820-827` | 否 | 整窗（norm） | 否 |
| 15 | `MaskGenerator`（`PReLU`+k=1 conv，`output_gate=False`） | `Falcon.py:1234-1240`；cfg `:99-100` | 否（实测 0） | 否（该 `ConvNormAct` 无 `norm_type`） | 否 |
| 16 | `STFTDecoder` `ConvTranspose2d(k=3,s=1,pad=1)` + causal pad/trim | `Falcon.py:1150,1177-1187` | **否**（实测：样本 16000 只依赖 mask 帧 122–124） | 同 | 否 |
| 17 | `torch.istft(center=True, length=L)` | `Falcon.py:1189-1195` | 否（重建，帧 t 覆盖样本 ±128） | 同 | 否 |
| 18 | `video_bottleneck`（`kernel_size:-1` → `Identity`） | `Falcon.py:1323`；cfg `:17-18` | 否 | 否 | 否 |
| 19 | `VideoBlock.gateway/projection/residual_conv`（k=1，`BatchNorm1d`） | `Falcon.py:933-964`；cfg `:77-81` | 否 | 否（BatchNorm1d = 逐帧 + running stats） | 否 |
| 20 | `VideoBlock.rnn`（`nn.LSTM(bidirectional=False)`） | `Falcon.py:952-956, 967-975` | 否 | 否 | 否 |
| 21 | `NeuroSCNetFusion`（1×1 投影 + nearest 上采样 + 加法） | `Falcon.py:241-267` | 否（1 视频帧 → 5 音频帧，实测 126–130） | 否 | 否 |
| 22 | `FRCNNVideoModel.frontend3D` `Conv3d(k=(5,7,7), pad=(2,3,3))` | `studio/falcon_model/videomodels/frcnn_videomodel.py:60-70` | **是，+2 视频帧 = 80 ms** | 同 | 否 |
| 23 | **`cLNhw` 本身** = `GlobalLayerNorm(GroupNorm(1,C))` + `CumulateLN4D(H)` | `look2hear/layers/normalizations.py:134-141, 165-202, 204-213, 229` | **是，整窗（C×T×F）** | — | 否 |

**统计**：音频路径 `cLNhw` 实例 **30 个**（实测 `swapped 30 LayerNormalizationHW(cLNhw) modules`）。
`configs/Falcon.yml:14` 写着 `causal: true`，但该值只用于三处：`Falcon.py:1325-1326` 写进 `audio_params['causal']`/`video_params['causal']`、
`Falcon.py:1130-1146` 的 encoder 与 `Falcon.py:1131-1175` 的 decoder（`enc_dec_params.causal` 独立），
以及 `AudioBlock.process_t_layers`（而 `DualPathRNN.__init__` 把 `causal` 存下来后**从未使用**，`Falcon.py:333-386` 全文无 `self.causal` 分支）。

### 2.2 `causal=True` 的实际 pad 行为（易被误读，单独说清）

```python
# Falcon.py:48, 50-52
self.padding = dilation * (kernel_size - 1) // 2      # k=3 → 1
if self.causal:
    self.causal_padding = dilation * (kernel_size - 1)  # k=3 → 2
    self.conv = nn.Conv2d(..., padding=(self.causal_padding, self.padding))  # (2, 1)
# Falcon.py:102-108
if self.causal:
    x = x[:, :, :-self.causal_padding, :]              # 裁掉末尾 2 帧
```
`padding=(2,1)` 在 Conv2d 中是 `(pad_H, pad_W)`，**每个维度两侧各 pad**：H(=时间) 两侧各 2，W(=频率) 两侧各 1。
于是 `T → T+4`，conv 后 `T+2`，裁尾 2 → `T`（长度守恒）。
**语义**：`out[t] = f(in[t-2 .. t])` ⇒ **0 帧未来**，但整体**后移 1 帧**（窗口中心在 `t-1`）。
这就是为什么 STFTEncoder 的单点扰动实测影响输出帧 `125..127`（`+0..+2`）——不是看未来，是 lag。

### 2.3 一个容易搞错的点：`DualPathRNN` 的"双向"

| 路径 | `dim` | 序列轴 | 双向？ | 算不算"看未来" |
|---|---|---|---|---|
| `process_f_layers`（`layers.layer_1`） | 4 | **频率**（`Falcon.py:392-400`：`permute(0,3,1,2).view(B*F_, C, T)` 后 `T` 实为频率轴） | 是 | **不算**——每个时间帧独立跑频向 BiLSTM |
| `process_t_layers`（`layers.layer_2`） | 3 | **时间**（`Falcon.py:401-405`：`permute(2,0,1)`，seq=T） | **否** | 不算 |

实测印证：改 `process_f` 输入的时间帧 15，输出**只有**帧 15 变化（`out[+0,+0]`）；改 `process_t` 亦 `[+0,+0]`。
所以"频向双向 LSTM 是否算未来"——**不算**，它在时间上逐帧独立。
（注：`process_t` 的 0 是"1e-6 相对阈值"下的 0，单向 LSTM 的因果传播本身是无限的，见 §4.3 的说明。）

---

## 3. 解析推导（含公式）

### 3.1 单算子

- **STFT（`center=True`）**：帧 *t* 的中心样本 `= t·H`，覆盖 `[tH − W/2, tH + W/2)`，`H=128, W=256`。
  ⇒ 帧 *t* 需要到样本 `tH + 127`，即**未来 1 帧**。实测：改整帧 125 的 128 个样本 → STFT 帧 `[125,126]`（`+0,+1`）。
- **`ConvNormAct(causal=False)`**：`p = (k−1)//2` 两侧对称
  ⇒ `out[j] = f(in[j−p .. j+p])` ⇒ 未来 `p` 帧。
  k=3 → +1 帧；k=5 → +2 帧；k=1 → 0。
- **`ConvNormAct(causal=True, is2d)`**：见 §2.2 ⇒ 未来 0，lag `k−1`。
- **depthwise k=5, s=2**：`out[j] = f(in[2j−2 .. 2j+2])` ⇒ 未来 2 个**该层**帧 = `2·2^l` 原帧。
- **nearest 上采样**：`out[j] = in[⌊j·n_in/n_out⌋]` ⇒ 0。
- **单向 LSTM**：`out[j] = g(in[0..j], h_{j-1})` ⇒ 0 未来、memory=∞（窗内）。
- **tril attention**：`out[j] = Σ_{j'≤j}` ⇒ 0 未来、memory=∞（窗内）。
- **`GroupNorm(1, C)`**：统计量跨 **C×T×F** ⇒ 未来 = 整窗。

### 3.2 编码器阶梯（累计到原帧）

记 `R_l(j)` = 第 *l* 层下标 *j* 能触及的最大**原帧**（`l=0` 为 STFT 帧），`ref_l(j) = j·2^l` 为名义对齐：

```
R_0(j) = j + 1                                  (k=3, p=1)
R_1(j) = R_0(2j+2) = 2j + 3                     (k=5, s=2, p=2)
R_2(j) = R_1(2j+2) = 4j + 7
R_3(j) = R_2(2j+2) = 8j + 15
```
⇒ **`D_l = 2^{l+1} − 1`**，`D_3 = 15` 帧 = **120 ms**（编码器最深层的单层解析值。论文 §6.4 声明的 120 ms 音频预算取的就是这个数，但它不含 GCI/decoder 与第二遍）。

实测（§4.1）D1 输出帧 `62,63` 被改、D2 `31,32`、D3 `15,16`，逐级与上式一致。

> **注意**：这个 `15` 就是论文里那句 "leaving the audio branch at most **15 frames** of lookahead"（arXiv HTML §6.3, Table 7 段）。
> 它在**论文自己的假设（left-aligned padding）下应该等于 0**；只有代码的对称 padding 才给出 15。

### 3.3 一遍 `AudioBlock`

`AudioBlock` 不是纯编码器：GCI（`InjectionMultiSum`）+ 自顶向下 decoder，每一级都把"全局上下文"（来自最深层 `S_3=G`）用 k=5 depthwise 卷一次再 nearest 上采。
逐级递推（`LA_l` = 第 *l* 级输出相对原帧的 lookahead）：

```
GCI:  LA_3 = 15 + 2·2^3 = 31          # G 自身先过一遍 k=5 (pad 2) → +2 个 level-3 帧 = +16
decoder l=2: LA_2 = max(31, LA_3 + 2·2^3) = 31 + 16 = 47
decoder l=1: LA_1 = max(31, LA_2 + 2·2^2) = 47 + 8  = 55
decoder l=0: LA_0 = max(31, LA_1 + 2·2^1) = 55 + 4  = 59
```
**闭式**：`LA_pass = (2^5 − 1) + Σ_{l=1..3} 2^{l+1} = 31 + 16 + 8 + 4 = 59 帧 = 472 ms`。

### 3.4 两遍 `AudioBlock`

`Separator.forward`（`Falcon.py:1032-1059`）在 `repeats:2 / fusion_repeats:1`（cfg `:46,80`）下：
`audio → audio_net → caf(audio,video) → audio_net`，即 **AudioBlock 跑两遍**（CAF 的加法与 nearest 上采样不增加时间上下文）。

```
LA_sep = LA_pass + LA_pass − ε = 115f = 920 ms      (ε≈2 帧来自窗尾边界裁剪；115f 是解析值，论文声明里的 15 帧只是编码器单层)
```

### 3.5 加入 `cLNhw`

任一处 `cLNhw` 把 `[L(j),U(j)]` 变成 `[0, T−1]`：

```
LA(t) = (T−1) − t   ⇒  最大 250 帧 = 2000 ms，且与 t 有关（越靠前 lookahead 越大）
```

### 3.6 解析 vs 实测对账

| 量 | 解析 | 实测 | 差 |
|---|---|---|---|
| 编码器 S3 | 15 帧 / 120 ms | 15 帧（D3 输出 15,16） | 0。本行的 120 ms 是论文声明里的预算值，这里只把它当编码器单层量核对 |
| 一遍 AudioBlock | 59 帧 / 472 ms | **57 帧 / 456 ms** | −2（窗尾裁剪） |
| Separator 两遍 | 115f / 920 ms（解析值；论文声明里的 15 帧是编码器单层，不是这个量） | **113 帧 / 904 ms** | −2 |
| 全模型（`deNORM`） | 115f（解析值；与论文声明的 15 帧不同层级） | **112.4 帧 / 899 ms** | −2.6 |
| as-shipped | 250 帧 | 245→0 全窗（§4.5） | — |

**`docs/STATUS.md:41` 的"57 帧 ≈ 456 ms"是"一遍 AudioBlock"的几何值**：数值没错，但**只是全链的一半**。

> **两路实测互注**：W2-11 §2.3.1 用 tail-perturb 测得单遍 54 帧/432 ms、整链 110 帧/880 ms，与上表的解析/实测值相差约 3%；它对 AudioBlock `upsampling_depth` 2/3/4/5 给出 80/208/432/944 ms，与本节 `D_l` 阶梯同源。差异来自窗尾裁剪（ε≈2 帧）与扰动阈值，同一物理量两条测量路径，引用时用 904 ms 并注明另一路差约 3%。

---

## 4. 实测（遮挡/脉冲/跨窗，附命令与数值）

### 4.1 算子隔离（`/tmp/w212/ops.py`、`ops2.py`）

命令：`CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python ops.py`（`ops2.py` 同）

```
=== 0. torch.stft(center=True) alone ===
torch.stft        n_frames=251 one input sample 16000 (frame 125) -> frames [125,125] = [+0,+0]
torch.stft        whole frame 125 changed -> frames [125,126] = [+0,+1]  (=> frame t needs samples up to (t+1)*hop)

=== 1. STFTEncoder (stft + conv k=3 causal=True) ===
STFTEncoder       sample 16000 -> out frames [125,127] = [+0,+2]  (out T=251, F=129)

=== 2. audio_bottleneck / AudioBlock entry (cLNhw present vs ablated) ===
audio_bottleneck  [cLNhw active]        j= 125 | 1e-3: out[-125,+125] | 1e-6: out[-125,+125] | LOOKAHEAD=125f
audio_bottleneck  [norm=Identity]       j= 125 | 1e-3: out[  +0,  +0] | 1e-6: out[  +0,  +0] | LOOKAHEAD=0f
gateway    k=1 dw                       j= 125 | 1e-3: out[  +0,  +0] | 1e-6: out[  +0,  +0] | LOOKAHEAD=0f
projection k=1                          j= 125 | 1e-3: out[  +0,+125] | 1e-6: out[-125,+125] | LOOKAHEAD=125f
projection k=1 [norm=Identity]          j= 125 | 1e-3: out[  +0,  +0] | 1e-6: out[  +0,  +0] | LOOKAHEAD=0f

=== 3. down-sample ladder (geometry vs +cLNhw) ===
D0 k=3 s=1 pad=1 g=64 [cLNhw]           j= 125 | 1e-3: out[  -1,+125] | 1e-6: out[-125,+125] | LOOKAHEAD=125f
D0 k=3 s=1 [norm=Identity]              j= 125 | 1e-3: out[  -1,  +1] | 1e-6: out[  -1,  +1] | LOOKAHEAD=1f
D1 k=5 s=2 pad=2 g=64 [cLNhw]           j= 125 | 1e-3: out[ -63,  +0] | 1e-6: out[-125,  +0] | LOOKAHEAD=125f
D1 k=5 s=2 [norm=Identity]              j= 125 | 1e-3: out[ -63, -62] | 1e-6: out[ -63, -62] | LOOKAHEAD=63f
D2 k=5 s=2 pad=2 g=64 [cLNhw]           j=  63 | 1e-3: out[ -63,  -1] | 1e-6: out[ -63,  -1] | LOOKAHEAD=63f
D2 k=5 s=2 [norm=Identity]              j=  63 | 1e-3: out[ -32, -31] | 1e-6: out[ -32, -31] | LOOKAHEAD=32f
D3 k=5 s=2 pad=2 g=64 [cLNhw]           j=  31 | 1e-3: out[ -31,  +0] | 1e-6: out[ -31,  +0] | LOOKAHEAD=31f
D3 k=5 s=2 [norm=Identity]              j=  31 | 1e-3: out[ -16, -15] | 1e-6: out[ -16, -15] | LOOKAHEAD=16f
  ladder shapes -> T,F = 32,17
```
> 阶梯行的 `-63/-32/-16` 是**输入坐标系**的偏移（输出下标 62/31/15 换成输入下标后即 −63）。折算成"输出帧"后：
> D1 的 `62,63` ⇒ lookahead 1 输出帧 = 2 输入帧；D2 的 `31,32` ⇒ 1 输出帧 = 4 输入帧；D3 的 `15,16` ⇒ **16 输入帧**。与 §3.2 的 `2^{l+1}−1` 完全对上。

```
=== 4. TFRB @ bottleneck (T/8=31, F/8=17) ===
process_f   DualPathRNN dim=4 bidir LSTM over FREQ   j= 16 | 1e-3: out[  +0,  +0] | 1e-6: out[  +0,  +0] | LOOKAHEAD=0f
process_t   DualPathRNN dim=3 uni LSTM over TIME     j= 16 | 1e-3: out[  +0,  +0] | 1e-6: out[  +0,  +0] | LOOKAHEAD=0f
process_att MHSA2D causal=True (tril)                j= 16 | 1e-3: out[  +0, +15] | 1e-6: out[  +0, +15] | LOOKAHEAD=0f
process_att MHSA2D causal=False (ablation)           j= 16 | 1e-3: out[ -16, +15] | 1e-6: out[ -16, +15] | LOOKAHEAD=16f

=== 5. GCI / concat (k=5, causal=False) ===
fusion_layers[3] local=S3[31] global=G[31]          j=  15 | 1e-3[ -15, +15] 1e-6[ -15, +15] LOOKAHEAD=15f fwdspan=15f
fusion_layers[3] [norms=Id]                         j=  15 | 1e-3[  -2,  +2] 1e-6[  -2,  +2] LOOKAHEAD=2f fwdspan=2f
fusion_layers[0] S0=251,G=31                        j= 125 | 1e-3[-125,+125] 1e-6[-125,+125] LOOKAHEAD=125f fwdspan=125f
fusion_layers[0] [norms=Id]                         j= 125 | 1e-3[ -19, +20] 1e-6[ -19, +20] LOOKAHEAD=19f fwdspan=20f
concat_layers[2] S2=63,Y3=31                        j=  30 | 1e-3[  -3, +32] 1e-6[  -3, +32] LOOKAHEAD=3f fwdspan=32f
concat_layers[2] [norms=Id]                         j=  30 | 1e-3[  -3,  +6] 1e-6[  -3,  +6] LOOKAHEAD=3f fwdspan=6f
residual_conv k=1                                   j= 125 | 1e-3[  +0,  +0] 1e-6: out[+0,+0] | LOOKAHEAD=0f

=== 6. mask / decoder ===
MaskGenerator (PReLU+conv1x1+causal=False)  -> mask frames [125,125] LOOKAHEAD=0f  (output_gate=False)
MaskGenerator(mixture emb frame 125)        -> mask frames [125,125] LOOKAHEAD=0f
STFTDecoder (convT k=3 causal pad+trim, istft center=True) frame 125 -> samples [16001,16511]
  = frames [125.01,128.99]  (no sample < 16000 changed => 0 lookahead, ~3 frames reconstruction tail)
```

`ops2.py` 关键片段：
```
FRCNN out (1, 512, 50)
FRCNNVideoModel mouth frame 25 -> feature frames [23,27] = [-2,+2] video frames
VideoBlock (k=1 + uni-LSTM over 50 video frames)   j= 25 | [+0,+0] LOOKAHEAD=0f
NeuroSCNetFusion video frame 25 -> audio frames [126,130] (nearest upsample 50->251 => 5 audio frames/video frame)
AudioBlock as shipped (cLNhw everywhere)            j= 125 | [-125,+125] LOOKAHEAD=125f
AudioBlock all norms=Identity (pure geometry)       j= 125 | [ -57,+125] LOOKAHEAD=57f
```

### 4.2 端到端脉冲与遮挡（`/tmp/w212/e2e.py`）

输入：`/root/data/falcon_lrs2_2mix/cv/mix/6300370419826092098_00001_...wav`（16 kHz, 32000 样本, 峰值归一到 0.9）。
```
[timing] first fwd 511 ms / warm fwd median 523.0 ms (n=3, cpu, 2s window)

[impulse] delta at sample 16000 (frame 125) -> output peak frame 125 (0 ms later);
          frames>-40dB: 123..130 (rel -2..+5 frames)
[impulse] -60dB support: 1..249                      <-- 整窗，归一化把能量摊到全窗

[single-frame] perturb frame 125 -> |dY|>-80dB frames 0..249 (rel -125..124);
               frames>-40dB 15..249

[future-noise 1e-3 after frame 150] past(<19200) RMS change = -20.5 dB re future-change RMS;
               max|d| past=7.11e-04 vs future=7.66e-03
[future-noise] past frame change dB (re max), frames 0,30,60,90,120,140:
               [-30.2, -23.2, -20.9, -22.2, -20.5, -15.3]

[future-zero after frame 150] past RMS change = -22.77 dB re past signal
```

**解读**：
- 脉冲响应主瓣极短（−40 dB 内 ±2/+5 帧），说明"局部"感受野并不长；
- 但 −60 dB 支撑铺满整窗 1..249 —— 这就是 `cLNhw` 的全窗统计；
- **只在未来注入 1e-3 噪声，过去帧整体变化 −20.5 dB（约 10% RMS）**；把未来清零，过去变化 −22.8 dB。
  一个真正窗口级因果（113 帧）的模型，在 fcut=150 时对帧 0（相距 150 帧 > 113）应当**完全不动**。它动了 −30 dB。

### 4.3 梯度锥（真实权重、精确依赖）（`/tmp/w212/e2e2.py` §A）

`grad(y[0,0,s], x)`，按帧取 RMS，报"最后一个 > thr·max 的输入帧 − 输出帧"：
```
  2s window out frame  60 -> 0.1:+4 0.001:+186 1e-06:+189 1e-09:+189 ; backreach -60
  2s window out frame 120 -> 0.1:+8 0.001:+122 1e-06:+129 1e-09:+129 ; backreach -120
  2s window out frame 180 -> 0.1:+5 0.001:+69  1e-06:+69  1e-09:+69  ; backreach -180
  2s window out frame 240 -> 0.1:+7 0.001:+9   1e-06:+9   1e-09:+9   ; backreach -240
```
- **能量主导向 lookahead = +4…+8 帧（32–64 ms）**：绝大多数梯度能量落在很短的前视范围内；
- **−60 dB（1e-3）以下铺到窗尾**：`+186`（帧 60 → 窗尾 249）。这与"整窗归一化"一致，也与 `deNORM` 下的 `+60` 形成对照；
- `1e-9` 与 `1e-6` 相同 ⇒ 不是浮点噪声，是真实的解析依赖。

> 关于 `process_t`（单向 LSTM）在算子隔离里显示 `+0`：这是 1e-6 **相对阈值**把 16 步之后的指数衰减截掉了。
> 结构性结论（LSTM 单向 ⇒ 0 未来）由代码 `Falcon.py:401-405` 决定，这里不依赖阈值。

### 4.4 逐层传播范围（单点扰动 → 输出范围）

`/tmp/w212/e2e3.py`（几何消融）与 `ops2.py`：

| 被扰动层 | 输入帧 | 输出可影响范围 | lookahead | 备注 |
|---|---|---|---|---|
| 编码器（STFT+conv） | 125 | 125 – 127 | 0（lag 2） | `+0..+2` |
| `audio_bottleneck` | 125 | 0 – 250 | **整窗** | norm=Id 时 0 |
| `AudioBlock`（一遍，几何） | 125 | **68 – 250** | **57 帧 / 456 ms** | 实测（`e2e3.py`） |
| `AudioBlock`（一遍，几何） | 200 | 148 – 250 | 52 帧 | 窗尾裁剪 |
| `Separator`（两遍，几何） | 125 | **12 – 250** | **113 帧 / 904 ms** | 实测 |
| 全模型（`deNORM`） | 60 | 0.05 – 250 | **59.95 帧 / 480 ms** | `e2e2.py` §B |
| 全模型（`deNORM`） | 125 | 12.60 – 250 | **112.4 帧 / 899 ms** | 同上 |
| 全模型（`deNORM`） | 200 | 92.55 – 250 | 107.5 帧 / 860 ms | 对齐抖动 |
| 全模型（as-shipped） | 125 | 0 – 250 | **整窗** | `ops2.py` AudioBlock 行 |
| `MaskGenerator` | 125 | 125 | 0 | |
| `STFTDecoder` | 125 | 样本 16001–16511 | 0 | 只有重建尾巴 |

`STFTDecoder` 的反向核对（`e2e2.py` §C）——**输出样本 16000 依赖哪些 mask 帧**：
```
|dY[16000]| by perturbed mask frame: [(122, 1.61e-03), (123, 8.49e-04), (124, 3.63e-03)]
```
样本 16000 的分析帧是 125，`istft(center=True)` 理论触碰帧 124–126，实测只到 **124** ⇒ 解码器**严格因果、且 lag ≈1 帧**，0 lookahead。

### 4.5 归因实验：把 `cLNhw` 换成论文的 `LN_c`（`/tmp/w212/fix2.py`）

`LN_c`（论文 Eq.2）：`μ_{t,f} = (1/C_h)Σ_c Z_{c,t,f}` ⇒ **逐 (t,f) 的通道归一**，0 lookahead。
替换时**保留学到的 affine**（`cumulate_ln.gain/bias`），只把统计量从"全窗 GroupNorm"改成"逐点"。

```
[fix] swapped 30 LayerNormalizationHW(cLNhw) modules -> per-(t,f) channel LN, learned affine kept

=== A. future-noise 1e-3 injected at frame fcut -> how much does each PAST frame change ===
  as shipped (cLNhw = GroupNorm)     fcut=150 0:-30 20:+0 40:-25 80:-22 120:-21 140:-15
  LN_c fix (per-(t,f) stats)         fcut=150 0:-240 20:-240 40:-78 80:-70 120:-29 140:+0
  as shipped (cLNhw = GroupNorm)     fcut=245 0:-52 20:-16 40:-48 80:-33 120:-47 160:-36 200:-43 240:-5
  LN_c fix (per-(t,f) stats)         fcut=245 0:-240 20:-240 40:-240 80:-240 120:-240 160:-35 200:-18 240:-7
```
**这是本文最硬的一条证据**：
- as-shipped：窗尾（帧 245）的扰动一直传到**帧 0**，衰减只有 −52…−5 dB ⇒ lookahead = 245 帧 = **1960 ms**；
- 换成 `LN_c` 后：所列帧 0、20、40、80、120 为 **−240 dB**；整个区间最大值为 −75.2895 dB，从帧 160 起才有响应 ⇒ 依赖下界 ≈ `245 − 113 = 132` 帧，**正好等于 §3 的几何 113 帧**。
- 结论：**"整窗看未来"的元凶是 `cLNhw` 的全窗统计，30 处**；剥掉它，模型回落到 113 帧几何 lookahead。

`fix.py` 里还可以看到：换掉 norm 后立刻重测几何锥，帧 60 → `LOOKAHEAD=60.0f`、帧 125 → `112.8f`，
与 `deNORM` 的 59.95 / 112.4 一致（两种独立改法给出同一个几何极限）。

### 4.6 视觉侧

| 实验 | 结果 | 结论 |
|---|---|---|
| FRCNN 单帧扰动（`ops2.py`） | 嘴部帧 25 → 特征帧 23–27 = **±2 视频帧** | 时间感受野 5 帧、未来 2 帧 = **80 ms**（不是论文声明的 3 帧/120 ms） |
| `VideoBlock`（k=1 + uni-LSTM） | `[+0,+0]` | 0 未来 |
| `caf` 融合 | 视频帧 25 → 音频帧 126–130 | 1→5 帧 nearest，0 未来 |
| **系统级**（as-shipped，梯度） | 输出音频样本 16064 ← 视频帧 **[0,49]（全窗）** | cLNhw 把视频贡献也全球化了 |
| **系统级**（`deNORM` 反事实） | 样本 16064 ← 视频帧 **[0,38] = [+13]** | 几何视觉 lookahead = 2(FRCNN) + 11.4(第一遍 57 音频帧 ÷5) ≈ **13 视频帧 = 520 ms** |

> 后两行很重要：**"视觉只缓冲 3 帧"在系统层面完全不成立**。即使把卷积改成因果的，
> 视觉侧仍需要 ~13 个未来视频帧（520 ms），因为第一遍 AudioBlock 的 57 帧前视会把融合点之后的视频需求一起带上来。

### 4.7 跨窗（规范实验 `tools/repro/experiments/streaming_canonical.py`，原 `/tmp/w213/canonical.py`）

输入：同一 source id 的 3 条 LRS2-2Mix 片段拼接成 6.0 s（`joins=[32000,63840]`，join 处 10 ms 交叉淡化，
尾部补零到 hop 整数倍）。参考 = 整段一次前向（`cLNhw` 统计跨 6 s）。
Studio 拼接仿真严格照 `studio/web/app.js:9-10,621-647` + `studio/server.py:29,106-120,328`：
`WINDOW=32000, HOP_S=8000, GUARD=1280, FADE=320`，取 `pred[22720:30720]`，块首 320 样本与上一窗的 `pred[30720:31040]` 交叉淡化。
网格从 **t=0 的首窗（`e=9280`）** 开始，即产品从静音起播的真实口径。

```
=== Studio policy (2 s window / 0.5 s hop / 80 ms guard / 20 ms crossfade) ===
  stream 6.0 s (3 x 2 s, joins=[32000,63840] with 10 ms crossfade) ; grid from t=0 (first e=9280)
  windows=11 covered samples 0..55000
  SI-SNR(stitched vs whole) = +14.71 dB   r=0.9835  maxdev=53.4% peak  rms ratio 0.975
  per-chunk SI-SNR vs whole: 7.8 18.0 22.7 19.7 11.5 29.7 18.4 19.7 20.5 14.4 17.0
  steady state (cold-start first window dropped, 10 chunks): +16.72 dB  r=0.9895  maxdev=28.5%  rms ratio 0.987

=== ablation: hard splice (no crossfade) ===
  SI-SNR = +14.72 dB  <- 20 ms 块间淡化对 SI-SNR 零影响
  seam jump |y[n]-y[n-1]| median 1.24e-02 max 3.99e-02 ; with crossfade median 9.57e-03

=== guard sweep, fixed absolute chunk (epoch anchored, common 0..5.0 s) ===
  stitch semantics (discard window tail)  80/160/320/456/640 ms = 14.51 15.19 15.01 16.71 15.57 dB
  lookahead semantics (add right context) same grid           = 22.12 22.19 22.55 21.30 22.92 dB

=== phase sweep: chunk fixed at [22720,30720), grid start +-250 ms ===
  vs whole: 17.99 16.67 16.38 17.42 16.72 15.86 17.94 17.56 17.99 dB (spread 2.1 dB)
  any two phases against each other: min 17.02 dB

=== where does the error live? SNR by position inside the emitted chunk ===
  sub-block (each 0.1 s of every 0.5 s chunk): +22.9 +6.4 +9.8 +13.8 +8.6  dB
  window-relative frames: ['178-190', '190-203', '203-216', '216-228', '228-240']

=== magnitude of cross-window difference ===
  max|stitched-ref| = 0.4414 ; ref peak = 0.7998 (=53.4% of peak)
  correlation = 0.9835

=== attribution: what the error is made of ===
  2 s 无重叠平铺 vs 6 s 整段 = 18.66 dB   <- 窗长/统计失配（与 hop/guard/淡化无关）
  Studio 拼接 vs 2 s 平铺     = 15.26 dB   <- 拼接机械误差
```

**四条硬结论**：
1. **跨窗不连续是量级问题，不是细节问题**：与整段推理比 **+14.71 dB SI-SNR / corr 0.9835 / 最大偏差 53.4% 峰值**
   （6 s，含冷启动；稳态 +16.72 / 0.9895 / 28.5%）；RMS 比 **0.975（−0.22 dB）**，稳态 0.987——**不是在缝上掉电平**。
2. **20 ms 交叉淡化只治"咔哒"不治"内容"**：SI-SNR 完全不变（14.71 vs 硬拼接 14.72），seam 跳变中位数 1.24e-2 → 9.57e-3（−2.2 dB），
   而窗内自然差分中位数是 1.69e-2 —— seams 处的跳变本来就**不比正常波形差分更大**。
   **误差不集中在缝上**：块级 SI-SNR 从冷启动 7.8 到 29.7，joins 所在块 11.5–14.4，其余 18–30。
3. **块内位置剖面：误差集中在块内后段**：块 = 窗内帧 178–240，几何 lookahead 113 帧要求 `t ≤ 137`，
   故**整块 63 帧都在几何上不足**（原文"前 ~13 帧足上下文"自相矛盾：块里没有一帧满足 `t ≤ 137`）；
   实测块内前 0.1 s 22.9 dB，其余 6.4/9.8/13.8/8.6 dB。
4. **guard 覆盖不足，但 guard 是次要旋钮**：guard 80 ms 与几何需求（456–904 ms）差 **5.7×–11.3×**，
   但把 guard 从 80 ms 拉到 456 ms，丢弃窗尾语义下提升 **+2.2 dB**；补右上下文语义下为 22.12→21.30 dB，640 ms 时为 22.92 dB，所以 guard 不是主因。
   误差主体是 `cLNhw` 的窗长依赖：同一 ckpt 任意两种 2 s 窗方案互差 **13–19 dB**，不是 hop 或 guard。
   另外 `docs/STATUS.md` 至今未记录 Studio 选 1280（80 ms）的依据。

**相位敏感性 ~17 dB（两两）/ 2.1 dB（整段波动）** 是"这个模型不是流式"的最直接量化：同一段音频、同一段绝对时间，
把窗网格起点平移 ±250 ms，与整段的一致性只在 **15.86–17.99 dB** 间波动，任两种对齐互相 **17.02 dB**。
旧稿的 8.69 dB 不作数：那是"guard=0 的窗尾"与"guard=250 ms 的窗中"的单块比较，混了两个变量。

---

## 5. 延迟分解与实时性判定

### 5.1 公式

```
L_e2e  =  L_algo  +  L_buf  +  L_fwd  +  L_io
L_algo =  max_t ( U(t) − t ) · 8 ms          # 架构要求的未来上下文（本文 §3）
L_buf  =  GUARD + (0 … HOP_S)                # 块内位置 + 块对齐抖动
L_fwd  =  F_video + F_audio                  # 单窗前向
L_io   =  采集/编码/WebSocket/base64/播放调度（未测）
```

### 5.2 分量（实测/引用）

| 分量 | 值 | 口径/出处 |
|---|---|---|
| `L_algo` **as-shipped** | **0 – 2000 ms**（= 窗尾，随 t 变） | §4.5（fcut=245 → 帧 0 变化 −52 dB） |
| `L_algo` **几何** | **904 ms**（113 帧） | §4.4（113 帧实测 / 115 解析） |
| `L_algo` **论文声明**（不是代码事实） | 120 ms（视觉 3 帧） | arXiv HTML App E |
| `L_buf` GUARD | **80 ms**（1280 样本） | `studio/web/app.js:9`、`studio/server.py:29` |
| `L_buf` 块粒度 | 0 – 500 ms（均值 250 ms） | `HOP_S = 8000` |
| `F_video`（FRCNN，50 帧） | **119.2 ms** | 本轮 CPU 实测（4 线程） |
| `F_audio` 合计 | **473.2 ms** | 本轮 CPU 实测：encoder 2.0 / bottleneck 12.9 / separator 441.2 / mask 7.3 / decoder 5.6 |
| `F` 合计 CPU | **约 0.6 s**（4 线程实测 592–694 ms） | B=1、2 s 窗、CPU；另一次纯音频 warm 前向 median 523 ms |
| `F` 合计 A800 | **47.1 ms**（small）/ 56.5 ms（big） | 引用 `docs/understanding/deepdive-20261024/06-studio-backend.md:402-403`（本轮按硬规则未跑 GPU，见 GAP-1） |
| Studio 自报 RTF | `inference_ms/500` | `studio/server.py:330`；CPU 上 592–694/500 = **1.18–1.39 ⇒ 不实时** |

### 5.3 三个口径的"端到端延迟"

| 场景 | 最好 | 典型 | 最差 | 说明 |
|---|---|---|---|---|
| A800（F=47.1） | **127 ms** | ≈377 ms | 627 ms | 80 + 47 + (0/250/500) |
| CPU 4 线程（F≈0.6 s，592–694 ms） | **672 ms** | ≈922 ms | 1172 ms | **跟不上**，队列会持续增长 |
| 论文声明的"120 ms lookahead" | — | — | — | 指视觉前端，不是端到端 |

> 注意 `L_buf` 的块粒度项：Studio 每 0.5 s 只发一次窗，所以"块内最后一个样本"的延迟是 `80+F`，
> 而"块内第一个样本"是 `9280/16 + F = 580 + F` ms。稳态播放是连续的，队列深度恒定 ≈ 0.5 s − 80 ms。

### 5.4 判定：能不能做到论文声称的 120 ms？

**分三个问题回答，不能混**：

1. **墙钟端到端 ≤ 130 ms？** —— **尚未实测**；127 ms 只是基于历史单窗耗时和 guard 的部分估算。但这只是因为 Studio **只喂 80 ms 未来**。
2. **算法 lookahead ≤ 论文声明的 120 ms？** —— **不能**。几何 904 ms（单遍 456 ms 都已超），as-shipped 是整窗。
   代码里唯一等于这个 120 ms（论文声明值）的量是**编码器最深层的 15 帧**，它不含 GCI/decoder（+44 帧）也不含第二遍（+56 帧）。
3. **能不能在论文声明的 120 ms 内给出"足上下文"的输出？** —— **不能**，差 7.5×（904/120）。

**差在哪一层（按贡献排序）**：
| 层 | 增量 | 累积 | 修法 |
|---|---|---|---|
| `cLNhw` 全窗 GroupNorm（30 处） | 57 → **整窗（+2000 ms）** | 2000 ms | 换成论文 `LN_c`（逐 (t,f) 通道归一），§4.5 已验证可把依赖下界压回 113 帧 |
| 第二遍 AudioBlock（`repeats:2`） | **+56 帧 / 448 ms** | 904 ms | 只跑一遍，或让第二遍沿时间因果（现在几何上不是） |
| GCI + decoder 的 k=5 逐级上采 | **+44 帧 / 352 ms**（15→59） | 456 ms | 全部 depthwise 改 `causal=True`（左对齐），或把 GCI 的作用域限制在已过去的下采样块内 |
| 编码器对称 padding | **+15 帧 / 120 ms**（编码器单层几何值，等于论文声明值） | 120 ms | `causal=True`（论文文字声称的就是这个） |
| STFT `center=True` | +1 帧 / 8 ms | 8 ms | `center=False` + 1 帧补偿 |

**结论**：论文声明的 120 ms 只有在「编码器左对齐 + 单遍 + 帧局部 LN」三者同时成立时才可能；
当前代码三者全不成立，而 Studio 用 80 ms guard 把这个问题**从延迟转移成了质量**：块在几何上整块都在
113 帧 lookahead 之外（§4.7 第 3 条），但这个几何缺口在实测上只值 ~0.8 dB（§4.7 第 4 条）。

---

## 6. 与论文/文档的冲突与更正

### 6.1 与论文（arXiv:2610.32749v1 HTML）

| # | 论文原话（位置） | 代码/实测 | 判定 |
|---|---|---|---|
| P1 | "All convolutions use **left-aligned padding along time** and symmetric padding along frequency, **adding no temporal lookahead**."（§4.2 Encoder） | `downsample_layers` / `fusion_layers` / `concat_layers` 全部 `causal=False` → 对称 pad（`Falcon.py:758-776,797-819`）；实测几何 +57 帧/遍 | **不符**。论文描述的是它没实现的版本 |
| P2 | "`LN_c` … `μ_{t,f} = (1/C_h)Σ_c Z_{c,t,f}` … is **frame-local and cache-free**"（Eq.2） | 实现是 `cLNhw` = `GroupNorm(1,C)` 覆盖 C×T×F（`normalizations.py:134-141,204-213`）；30 处；实测整窗 | **不符**。这是最严重的一条 |
| P3 | "the visual branch imposes a fixed algorithmic lookahead of **120 ms** … leaving the audio branch at most **15 frames** of lookahead"（§6.3, Table 7 段） | 论文声明的 15 帧 = `2^4−1` = 代码**对称 padding** 下 S3 的累计 lookahead（§3.2/§4.1），只覆盖编码器单层；真实音频 lookahead = 113 帧 = 904 ms | **数字自相矛盾**：15 只能由 P1 否认的那种 padding 产生 |
| P4 | 论文声明："At 25 fps, the visual stream buffers **three future frames**, corresponding to 120 ms"（App E） | `Conv3d(kernel=(5,7,7), padding=(2,3,3))` → **±2 帧 = 80 ms**（`frcnn_videomodel.py:60-70`，实测 ±2） | **偏大 1 帧**；且系统级视觉需求是 ~13 帧/520 ms（§4.6） |
| P5 | "the configuration uses **no additional future-chunk buffer**"（App E） | Studio 有 `GUARD=1280`（80 ms，`app.js:9`）+ 0.5 s 块缓冲 | **不符**（若论文指训练/评测流程则另说，但部署实现有） |
| P6 | "51.3 ms GPU latency"（Table 5） | 单窗 2 s 前向；本仓 A800 47.1 ms | 数字可比，但**这是单窗前向延迟，不能代替采集至播放延迟**（§5.3：端到端 127–627 ms） |

### 6.2 与仓库文档

| # | 旧说法（位置） | 实际 | 更正建议 |
|---|---|---|---|
| D1 | "单遍 lookahead 实测 **57 帧≈456 ms**"（`docs/STATUS.md:41`；`README.md:45`；`README_zh.md:45`） | 57 帧 = **一遍 AudioBlock** 的几何值；全链 = **113 帧/904 ms**（两遍）；叠加 `cLNhw` = **整窗 ≤2000 ms** | 改写为三档：`单遍几何 57 帧/456 ms`、`整链几何 113 帧/904 ms`、`as-shipped 窗长（2 s）` |
| D2 | "`causal = 窗口级因果`：`torch.stft(center=True)` + `cLNhw` 全窗 GN + AudioBlock 对称卷积"（`STATUS.md:41`） | 三条都对，但**漏了"两遍"**，且未量化"全窗 GN"= 30 处、−52 dB@帧0 | 补上 30 处 / fcut 实验数字 |
| D3 | "跨窗无状态"（`README.md:45`） | **正确**（模型无 cache/state；`falcon_runtime.py:1-5` 自己写明 "windowed inference, not a stateful streaming architecture"） | 保持，但补一句"因此同一段音频换窗对齐，与整段的一致性只在 15.86–17.99 dB 间波动，任两种对齐互相 ~17 dB" |
| D4 | "runs live in video meetings **on a laptop**"（`README.md:43`） | CPU 单窗约 0.6 s（4 线程 592–694 ms）> 500 ms 预算（RTF 1.18–1.39）；笔记本更慢 | 改为"on a workstation GPU"或注明 RTF |
| D5 | "**streaming** audio-visual target speaker extraction"（`project_page/index.html:7,12,18,339`） | 窗口式重算，非流式 | 改为 "windowed / chunked (2 s window, 0.5 s hop)" |
| D6 | "unidirectional LSTM along the time axis to **maintain causal streaming**"（`project_page/index.html:422`） | LSTM 确实 0 未来（实测），但 `cLNhw` 与对称卷积让整链非因果 | 保留 LSTM 描述，删掉"maintain causal streaming"的因果承诺 |
| D7 | "51.3 ms GPU latency"（`project_page/index.html:12,304`） | 单窗前向吞吐 | 标注"per 2 s window forward, batch 1"，并给出端到端区间 |
| D8 | `docs/STATUS.md:44` "A800 单窗 small 47.1 ms / big 56.5 ms" | 与 `06-studio-backend.md:402-403` 一致（p50，含预处理/H2D/D2H） | 正确，可继续引用 |

### 6.3 一句可直接引用的准确表述

> **Falcon 是"窗口级因果"（window-causal），不是流式因果。**
> 单窗 2 s 内，输出第 *t* 帧在几何上依赖输入到 *t*+113 帧（≈904 ms；一遍 AudioBlock 已是 +57 帧/456 ms，
> 编码器阶梯本身 +15 帧/120 ms，这 120 ms 是论文声明的预算值、不是全模型 lookahead）；实际发布权重因 30 处 `cLNhw`（跨 C×T×F 的 GroupNorm）而依赖到窗尾，
> 即 lookahead 随窗长增长（2 s 窗时最坏 2 s）。跨窗无任何状态传递，Studio 以 2 s 窗 / 0.5 s hop / 80 ms guard /
> 20 ms 交叉淡化重算，同一段音频换窗对齐只让整段一致性在 15.86–17.99 dB 间波动（2.1 dB），任两种对齐互相 ~17 dB。视觉前端感受野为 ±2 视频帧（80 ms），
> 但系统级视觉依赖约 13 个未来视频帧（520 ms）。

### 6.4 需要改的具体措辞清单

1. `docs/STATUS.md:41`（口径 1）：57 → 拆成三档（见 D1）。
2. `README.md:45` / `README_zh.md:45`：同上；并补"换窗对齐后与整段一致性仍在 15.86–17.99 dB，任两种对齐互相 ~17 dB"。
3. `README.md:43` / `README_zh.md:43`："on a laptop" → 注明 CPU RTF 1.18。
4. `project_page/index.html:7,12,18`（meta/og/twitter description）：`streaming` → `windowed`。
5. `project_page/index.html:339`（"Single-pass streaming separation"）：→ "Single-pass multi-scale separation (windowed inference, 2 s window / 0.5 s hop)"。
6. `project_page/index.html:422`：删"maintain causal streaming"。
7. `project_page/index.html:304`：51.3 ms 加脚注"per-window forward, batch 1, RTX 3090; end-to-end ≈ 0.13–0.63 s"。
8. `studio/web/app.js:9` 注释 "80 ms of right context the server keeps past each output chunk"：应说明该值**远小于**模型 lookahead（456–904 ms），是"取舍"而非"匹配"。

---

## 7. GAP LIST

| # | 问题 | 为什么没解决 | 建议认领方式 |
|---|---|---|---|
| G1 | **A800 上的端到端延迟未实测**（只有引用的 47.1 ms 单窗 p50） | 本轮硬规则要求 `CUDA_VISIBLE_DEVICES=` 置空 | 允许 GPU 时跑 `studio/scripts` 的 benchmark 30 次，并测"采集→播放"墙钟 |
| G2 | 论文声明的 15 帧预算与代码对称 padding 的**因果方向**（是作者按代码算的，还是巧合） | 只有数值吻合（15 = 2^4−1）这一条证据 | 读论文 v2/补充材料，或直接问作者 |
| G3 | 论文"3 future visual frames"的来源 | 代码只有 ±2；可能论文用了 3 帧时间堆叠的预处理（本仓没有） | 查上游 `JusperLee/Falcon` 的 dataloader 是否有 5 帧/3 帧堆叠 |
| G4 | `LN_c` 修正后**质量**如何 | 只换了统计量、权重未重训，不能评价 | 用 `LRS2_2Mix_Paper_v2.yml` 重训 20–30 epoch 做 A/B |
| G5 | 训练 `segment: 2.0` 与推理 2 s 窗一致，模型从未见过 >2 s 上下文 —— 跨窗统计不匹配的**训练侧影响** | 需要有 >2 s 段的训练配置 | 改 `segment: 4.0` 训练对照 |
| G6 | Studio `GUARD=1280` 的取值依据无文档 | 代码里只有注释，无推导 | 测 guard ∈ {1280, 3200, 4556, 7296} 的 SI-SNR/听感曲线 |
| G7 | `big` 变体（`SpeakerContrastiveV2` + `FalconBig.yml`）因果性未单独测 | 时间预算；结构可能不同（层数/通道） | 复跑 `ops.py`/`e2e3.py` 换 config 即可 |
| G8 | 20 ms 交叉淡化只覆盖 320/8000 = 4% 的块，**听感**上是否够 | 未做主观/客观音频质量评估 | 用 DNSMOS/NISQA 对比 stitch vs full-context |
| G9 | >5 s 长音频的跨窗**漂移**（误差是否累积） | 本轮最长 4 s | 用 RealSSA-Real 的 30 s 片段做 10 窗串联，看 per-chunk SNR 是否单调下降 |
| G10 | `STFTDecoder` 输出 `T+1 = 252` 帧（convT 后 `[:-1]` 仍多 1 帧）对 `istft(length=32000)` 的**边界样本**影响 | 只做了单点核对（样本 16000），未逐样本 | 对比 `decoder` 输出帧数与 `stft` 帧数，确认最后一帧是否被 `istft` 丢弃 |

---

## 8. 证据清单

### 8.1 脚本（`/tmp/w212/`，本机同副本；仓库只读，未写入任何文件）

| 文件 | 作用 | 关键输出 |
|---|---|---|
| `w212_common.py` | 用 `studio/configs/Falcon.yml` + `falcon.pt` 建模型（`strict=True`），`run()` 单窗前向 | `[build] audio params=537,697 video=on` |
| `ops.py` | 算子隔离：STFT / encoder / bottleneck / gateway / projection / 4 级 downsample / TFRB 三件 / fusion / concat / residual | §4.1 前半 |
| `ops2.py` | fusion/concat 两参、MaskGenerator(5D)、STFTDecoder、整 AudioBlock 几何、FRCNN、VideoBlock、caf | §4.1 后半 |
| `e2e.py` | 脉冲、单帧扰动、未来噪声、未来清零、梯度锥（首次，含索引 bug） | §4.2 |
| `e2e2.py` | 修正后的梯度锥、全模型 `deNORM` 几何锥、decoder 反向核对、视觉梯度锥 | §4.3/4.4/4.6 |
| `e2e3.py` | 单遍/两遍几何、视频锥（反事实）、MaskGenerator、**分阶段计时** | §4.4/4.6/5.2 |
| `canonical.py`，仓库副本 `tools/repro/experiments/streaming_canonical.py`（149 行） | 规范跨窗实验：Studio 拼接 vs 整段、hard-splice / guard 双语义 / 相位扫描、块内位置剖面、误差归属；`python streaming_canonical.py`（6 s）/ `--seconds 4` | §4.7 |
| `xwin.py`（**已作废**） | 旧探针：视频条件被拉伸 2×（固定 50 帧嘴型喂 4 s 整段）、对齐扫描混了 guard 与相位；数字全部由 `canonical.py` 取代 | §4.7 |
| `fix.py`, `fix2.py` | 把 30 处 `cLNhw` 换成 `LN_c`（保留 affine），帧级泄漏剖面 | §4.5 |
| `analytic.py` | 区间传播 + 名义对齐，闭式推导 | §3 |

### 8.2 关键 stdout（原文，可逐条对账）

```
# 计时（e2e.py / e2e3.py）
[timing] warm fwd median 523.0 ms (n=3, cpu, 2s window)
FRCNN(video frontend, 50 frames)    119.2 ms
encoder     2.0 | bottleneck    12.9 | separator    441.2 | mask     7.3 | decoder     5.6
audio total    473.2 ms ; audio+video    592.4 ms

# 几何极限（e2e3.py）
AudioBlock pass 1 frame 125 -> out frames [68,250] LOOKAHEAD=57f (456 ms)
AudioBlock pass 1 frame 200 -> out frames [148,250] LOOKAHEAD=52f (416 ms)
Separator(2 passes) input frame 125 -> out [12,250] LOOKAHEAD=113f (904 ms)

# 全模型（e2e2.py §B，deNORM）
perturb input frame  60 -> frames [0.05,249.99] -> LOOKAHEAD = 59.95 frames (480 ms)
perturb input frame 125 -> frames [12.60,249.99] -> LOOKAHEAD = 112.40 frames (899 ms)
perturb input frame 200 -> frames [92.55,249.99] -> LOOKAHEAD = 107.45 frames (860 ms)

# 泄漏归因（fix2.py，future-noise 1e-3 @fcut=245）
as shipped (cLNhw = GroupNorm)  fcut=245  0:-52 20:-16 40:-48 80:-33 120:-47 160:-36 200:-43 240:-5
LN_c fix (per-(t,f) stats)      fcut=245  0:-240 20:-240 40:-240 80:-240 120:-240 160:-35 200:-18 240:-7

# 解析（analytic.py）
# 注：S3 的 15f/120ms 是编码器单层解析值；论文 §6.4 把它当作音频支路预算（论文声明），两者不是同一个量
 S0  LOOKAHEAD max=   1f(    8ms)
 S1  LOOKAHEAD max=   3f(   24ms)
 S2  LOOKAHEAD max=   7f(   56ms)
 S3  LOOKAHEAD max=  15f(  120ms)
AudioBlock pass (measured: 57f)      LOOKAHEAD max=  59f(  472ms) median= 54f
Separator 2 passes (measured: 113f)  LOOKAHEAD max= 115f(  920ms) median=108f
GlobalLayerNorm over (C,T,F)         LOOKAHEAD max= 250f( 2000ms) median=125f

# 跨窗（streaming_canonical.py 6，原 canonical.py）—— 替代已作废的 xwin.py
stitched vs whole (6 s, Studio from t=0) : +14.71 dB  r=0.9835  maxdev=53.4% peak
stitched vs whole (6 s, 去冷启动首窗)    : +16.72 dB  r=0.9895  maxdev=28.5%
stitched vs whole (4 s)                  : +14.14 dB  r=0.9813  maxdev=53.9%
guard(stitch) 80/160/320/456/640 ms      : 14.51/15.19/15.01/16.71/15.57
guard(lookahead) 同档                    : 22.12/22.19/22.55/21.30/22.92
align +-250 ms vs whole                  : 15.86..17.99 dB; 两两最小 17.02 dB
sub-block SNR: +22.9 +6.4 +9.8 +13.8 +8.6 dB
```

### 8.3 代码位置索引（本文引用的全部 `path:line`）

```
look2hear/models/Falcon.py:48,50-52           ConvNormAct 非因果 pad / 因果 causal_padding
look2hear/models/Falcon.py:96-108             ConvNormAct.forward（因果裁剪）
look2hear/models/Falcon.py:322-327            InjectionMultiSum 的 nearest 上采样
look2hear/models/Falcon.py:333-386            DualPathRNN.__init__（causal 只存不用）
look2hear/models/Falcon.py:387-417            DualPathRNN.forward（dim=4 走频率轴）
look2hear/models/Falcon.py:419-546            MultiHeadSelfAttention2D
look2hear/models/Falcon.py:522-526            tril 因果掩码
look2hear/models/Falcon.py:693-899            AudioBlock（gateway 720 / projection 729 /
                                              downsample 758 / process_* 779-794 /
                                              fusion 797 / concat 810 / residual 820 / forward 828）
look2hear/models/Falcon.py:900-976            VideoBlock（gateway 933 / LSTM 952 / forward 967）
look2hear/models/Falcon.py:978-1059           Separator（repeats 拆分 998-999 / audio_net 1008 / forward 1032）
look2hear/models/Falcon.py:1061-1128          STFTEncoder（torch.stft 1117 / conv 1126）
look2hear/models/Falcon.py:1130-1199          STFTDecoder（padding 1150 / causal pad 1177-1179 /
                                              convT 1181 / 裁剪 1183-1184 / istft 1189）
look2hear/models/Falcon.py:1201-1282          MaskGenerator（conv 1234-1240 / apply 1251 / forward 1273）
look2hear/models/Falcon.py:1284-1354          Falcon.__init__（encoder 1319 / bottleneck 1322-1323 /
                                              causal 写回 1325-1326 / separator 1328 /
                                              mask 1336 / decoder 1343）
look2hear/models/Falcon.py:1480-1490          Falcon.forward
look2hear/layers/normalizations.py:134-141    GlobalLayerNorm = GroupNorm(num_groups=1)
look2hear/layers/normalizations.py:165-202    CumulateLN4D（沿 H 的累积，因果）
look2hear/layers/normalizations.py:204-213    LayerNormalizationHW（= cLNhw）
look2hear/layers/normalizations.py:229        cLNhw 别名
configs/Falcon.yml:14,21-26,28-36,39-72,77-89,93-100   全部关键超参
studio/falcon_runtime.py:1-5,29-31,200-212,278-289     无状态声明 / 常量 / 载权重 / 单窗前向
studio/server.py:29,68,106-120,328-330                 GUARD/FADE / 请求校验 / OutputSlicer / 响应
studio/web/app.js:9-10,621-647                         窗长/hop/guard/交叉淡化/发送
studio/falcon_model/videomodels/frcnn_videomodel.py:60-70   Conv3d(k=5,7,7)+pad(2,3,3)
docs/STATUS.md:41,44                                   口径与效率行
docs/understanding/deepdive-20261024/06-studio-backend.md:402-403,544  A800 47.1/56.5 ms 出处
```

### 8.4 本轮**没有**做的事（避免误读）

- 没有跑 GPU（硬规则）；没有训练；没有改仓库任何文件；没有下载数据。
- 没有评价 `LN_c` 修正后的分离质量（权重不匹配，只用于因果归因）。
- 没有做主观听感；没有做 >4 s 的长音频漂移测试。
- 论文引用均来自 `arxiv.org/html/2610.32749v1` 的 HTML 文本（`/tmp/falcon_paper.html`，504,680 字节），未交叉核对 PDF 排版。
