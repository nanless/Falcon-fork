# 评审：`docs/understanding/02-layers-dsp.md` —— 评审人 `model-core`

> 被评审版本：`sha256 = ae01042f95cef3e95961027b308b5e610a60f2b97b17dce0a7fad279a31bbd2f`（757 行，R1 初稿）。
> 评审口径：`00-CONVENTIONS.md` §5 模板 + `00-round-plan.md` §"评审口径"（逐条断言核对、P0/P1/P2 定义、必须读源码）。
> 复核环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0，CPU）+ stub
> `PYTHONPATH=$PWD/docs/verification/stubs:$PWD`；无源码改动、无权重重算（MACs/精度不在本评审范围）。
> 我的复核脚本：`docs/verification/scripts/verify_r2_causality_conditions.py`；日志：`docs/verification/runs/r2-causality-conditions.log`。
> 我的 R1 文档：`docs/understanding/01-model-core.md`（下称"01"）。

---

## 1. 结论

**总评：通过（需修订）**。这是本轮质量最高的文档之一：真实路径/死代码的切分准确（`layers/` 只有
`normalizations`/`activations` 进入 Falcon），两份 `stft.py` 同源判定的证据链完整（diff 行数 + 去空白 md5 双证），
归一化语义表与主要形状/参数结论与我的独立复核**全部一致**。§6 的 58 条断言中我逐条核对了 **58 条**
（源码 + 其日志 + 我的动态复算），其中 **51 ✅ / 6 ⚠️ 部分需修订 / 1 ❌ 事实错误**。

| 级别 | 数量 | 摘要 |
|---|---|---|
| **P0** | **1** | A36「`STFTEncoder` 严格时间因果」是**错误结论**，且以 `[验证]` 名义给出的证据（只测"前 120 帧不变"）无法支持该结论；与同文 A37、§4.2 表格自相矛盾 |
| **P1** | **4** | ①§4.1/§4.2 的"采样级 lookahead 256 样本/16 ms"与 A38 矛盾（正确上界 ≤128 样本/8 ms）；②A19/A49 的因果性数字缺探针条件（1.2e-03 是 T=8 人工张量的尖峰探针，不能代表 2 s 窗口），且漏掉"`CumulateLN4D` 会抵消纯尺度泄漏"这一关键机理；③A49「跨级放大到 ±8 帧」低估（L3 为 ±16，单遍实测 57 帧）；④A57 证据行号指向 `causal=True` 分支（66-68），实际使用分支是 84-85 |
| **P2** | **7** | A03/A05 的"唯一/全仓"未限定产品代码；§2.1 导入名字数 15/6 实为 13/5；§4.7 "import 会拉入 pytorch_lightning" 与实测不符；§4.2 把 `interpolate(nearest)` 列为非因果致因之一（nearest 单调 ⇒ 因果）；A52 编码器桥行号 off-by-one；§8 日志行数/关联验证记录里脚本行数与实际不符；遗漏 VideoBlock(train) 的 BN 非因果与 FRCNN ±80 ms |

**对"完全理解"的贡献**：该文档把"layers 是死库、Falcon 自带层"这条最容易误判的边界一次性钉死，
是 01（模型架构）与 04（损失/视觉）之间缺失的一环。修订 P0/P1 后即可作为 FINAL 的层/DSP 依据。

---

## 2. 断言逐条核对表（58 条）

判定：✅ 正确；⚠️ 部分正确/需补条件或改行号；❌ 错误；⏭ 未逐条核对（仅依赖其日志自证）。
"我的证据"优先给源码 `path:line` 或我的运行结果；日志指其 `runs/layers-dsp-verify.log`。

| 断言 | 判定 | 我的核对证据 / 备注 |
|---|---|---|
| A01 两份 stft.py 同源、仅 2 行空白 | ✅ | 复跑 `diff -u` = 15 行 / 2 hunk；`grep -c '^<'`=1、`'^>'`=1；去空白 md5 两文件均 `3f7053135a33126c8d2a078b98fc2927` |
| A02 差异 = utils:1 空行 + layers:164 四空格行 | ✅ | `diff \| cat -A` 显示 `0a1  $` 与 `164d164  "    "`，与其描述逐字一致 |
| A03 `layers/stft.py` 唯一 import 点 = `layers/__init__.py:19` | ⚠️ | 产品代码唯一 ✓；但 `docs/verification/scripts/verify_layers_dsp.py:34` 也 `from look2hear.layers import stft` ⇒ "唯一"应限定"产品代码（`look2hear/`、`studio/`）" |
| A04 `utils/stft.py` 唯一 import 点 = `utils/__init__.py:1` | ✅ | grep 仅此一处（产品代码） |
| A05 二者都无调用（死代码） | ⚠️ | 产品代码零调用 ✓；但同名函数在其自己的验证脚本被调用（`verify_layers_dsp.py:71,80,97`）⇒ 同 A03 需限定范围 |
| A06 `import look2hear.models` 加载两份 stft | ✅ | 其日志 §[0] 的 `sys.modules` 列表确认（含 `layers.stft`、`utils.stft`、`librosa`、`torch_complex`） |
| A07 stft.py 顶层依赖 librosa + distutils | ✅ | `layers/stft.py:7,10`（`import librosa.filters`、`from distutils.version import LooseVersion`） |
| A08 conv1d-DFT vs torch.stft 数值等价 | ✅ | 其日志 §[1]：2.289e-05（与我读到的实现路径一致，未重复跑） |
| A09 `mode="torch"` 逐位一致 | ✅ | 日志 §[1]：0.000e+00 |
| A10 inverse/round-trip | ✅ | 日志 §[2]：2.146e-06 / 2.384e-06 |
| A11 默认 `center=False`（与 torch 相反） | ✅ | `layers/stft.py:484`（`forward_stft`）、`:636`（`STFTBase`） |
| A12 默认窗 sqrthann=hann^0.5、periodic=True | ✅ | `layers/stft.py:37-56`：`sqrthann` 定义 + `periodic=True`（rect 除外） |
| A13 round_pow_of_two 改变 bin 数 | ✅ | 日志 §[1]：200→256→129；400→512→257 |
| A14 `T=1+⌊L/hop⌋`（center=True） | ✅ | `layers/stft.py:673-682` + 日志（32000→251） |
| A15 `export_jit` 死代码且必抛错 | ✅ | `layers/stft.py:21-26`：对 `nn.Module` 迭代 + `.exportable()` 均不支持 |
| A16 `get(None)` 返回 `nn.Identity` 类 | ✅ | `normalizations.py:231-233` |
| A17 `get("BatchNorm1d")` → torch 原生类 | ✅ | `normalizations.py:237-238`（`hasattr(nn, identifier)` 优先） |
| A18 `GlobalLayerNorm` == `nn.GroupNorm(1,C)`，统计跨 (C,T,F) | ✅ | `normalizations.py:134-144`；我独立复算：GN-only 前缀扰动 Δ=4.40（见 §3-P1-②） |
| A19 cLNhw 先全窗口 GN、再累积 LN ⇒ 非严格因果 | ⚠️ | **结论正确**；但证据 `1.2e-03` 来自 **T=8,F=65,+20 单帧尖峰**人工张量（其脚本 `verify_layers_dsp.py:389-396`），不代表 2 s 窗口（T=251）量级；且未说明第二段累积 LN 会大幅抵消 GN 的纯尺度泄漏（见 §3-P1-②） |
| A20 CumulateLN4D 统计 (C,t'≤t)、逐 f 独立、gain[1,C,1,1] | ✅ | `normalizations.py:165-202` 逐行核对：`input.sum(dim=1)` → `cumsum(dim=2)`（H）、`W` 保留、`gain(1,C,1,1)` |
| A21 `LN4D((C,1))` dim=(1,)、γ[1,C,1,1] | ✅ | `normalizations.py:146-157`（`dim=(1,3) if F>1 else (1,)`） |
| A22 `LN4D((C,F))` dim=(1,3)、γ[1,C,1,F] | ✅ | 同上；运行时验证 γ 形状 `(1,4,1,17)` |
| A23 `cLN=CumulateLN` 3D、累积、gain[1,C,1] | ✅ | `normalizations.py:91-132,225` |
| A24 `LN` 别名 = `CumulativeLayerNorm`，无累积语义 | ✅ | `normalizations.py:74-88,226` |
| A25 `CumulateLN` 定义两次，:51 版 `cumsum(dim=1)` 是 bug | ✅ | `normalizations.py:51,91`；:57 `x.sum(1,keepdim=True)` 后 `cumsum(dim=1)` 在 size=1 维上 = no-op，应为 dim=2 |
| A26 `MLayerNorm.beta` 初始化 1（非 0）且无使用者 | ✅ | `normalizations.py:27-28`（gamma/beta 均 `ones`）；全仓无实例化 |
| A27 激活解析大小写两条路径、实例等价 | ✅ | `activations.py:48-62` + 日志 §[5] |
| A28 Falcon 对 layers 的运行时依赖只有 normalizations/activations | ✅ | `models/Falcon.py:7`；`studio/falcon_model/models/Falcon.py:7`、`SISNRRepel.py:8` 同 |
| A29 cnnlayers/rnnlayers/enc_dec 在 Falcon 路径零调用 | ✅ | grep：仅 `layers/__init__.py` 内部 import；`evaluated_mac_params.py:85` 只是注释 "DPRNN" |
| A30 `rnnlayers.DPRNN*` 与 `DualPathRNN` 无关系 | ✅ | 两处定义互不引用（`rnnlayers.py:586-908` vs `models/Falcon.py:331-415`） |
| A31 `DPRNNLinear.forward:882` `transpose(1,1)` no-op | ✅ | `rnnlayers.py:882`：`self.net_out(output.transpose(1, 1)).transpose(1, 2)` |
| A32 enc_dec 无 causal 概念 | ✅ | `enc_dec.py:254-281,369-392` 对称 padding |
| A33 STFTEncoder 输出 `[B,112,251,129]` | ✅ | 我的 R1 验证 §8-[C] 同值 |
| A34 T=251、F=129 | ✅ | 同上 |
| A35 conv 因果裁剪后 T 不变 | ✅ | `Falcon.py:59,100-104`；(251+4-3)+1-2=251 |
| **A36 STFTEncoder 严格时间因果**（证据：前 120 帧 max\|Δ\|=0） | ❌ | **错误**。我复算（扰动采样 ≥16000）：`encoder` 帧 123/124 的 \|Δ\|=0，**帧 125 起非零（3.91）** ⇒ 整体存在 128 样本/8 ms 帧级 lookahead。其证据只覆盖帧<120，恰好避开泄漏帧 125，不能支持"严格因果"；且与 A37、§4.2 的 `torch.stft` 行自相矛盾 |
| A37 torch.stft 帧级 lookahead ≤128 样本 | ✅ | 公式 `t∈[⌈(p−127)/128⌉, ⌊(p+128)/128⌋]` 与我一致；日志 [156,157] 与 p=20000 吻合 |
| A38 STFTDecoder 严格时间因果 | ✅ | 我复算：改 embedding 帧 ≥125（采样≥16000）→ 输出 <16000 采样 max\|Δ\|=0；帧级严格因果成立（采样级 nuance 见 P1-①） |
| A39 解码帧账本 251→253→253→252 + length 截断 | ✅ | 我复算：(252−1)·128=32128≥32000 ✓；末帧窗 `[32000,32256)` ✓ |
| A40 DualPathRNN(dim=4) 频轴、batch=B·T、双向 | ✅ | `Falcon.py:386-401`；我 R1 验证同 |
| A41 DualPathRNN(dim=3) 时间轴、batch=B·F、单向因果 | ✅ | 同上；前缀 Δ=0（我 R1 §8-[D]） |
| A42 两个 DualPathRNN 都有 out_proj（48→64 / 72→64） | ✅ | 运行时确认：`Linear(48,64,bias=False)` / `Linear(72,64,bias=False)` |
| A43 默认 `rnn_type="Comba"` 构造必失败 | ✅ | 运行时：`AttributeError: module 'torch.nn' has no attribute 'Comba'`；默认值在 `Falcon.py:339`；`SISNRRepel.py:9` 注释 `#from fla.layers import Comba as FLAComba` 确认 |
| A44 MHSA2D(causal=True) tril mask | ✅ | `Falcon.py:520-524`；causal=False 前缀 Δ=5.1（其日志）/4.68（我 R1） |
| A45 注意力沿 T、频率摊平进 embedding、[B·n_head,T,T] | ✅ | `Falcon.py:505-527`；我运行时形状核对 |
| A46 n_freqs 必须等于运行时 F | ✅ | 运行时 γ 形状 `(1,4,1,17)`（L3）；`Falcon.py:185-188,785-786` |
| A47 AudioBlock 4 级形状 251→251→126→63→32 / 129→129→65→33→17 | ✅ | 与我的 forward hook 实测逐级一致 |
| A48 只有最深级在 encoder 侧跑 TFRB，decoder 用 2/1/0 | ✅ | `Falcon.py:842-845,868-888`；我 hook 计数：每级 process 各 1 次 |
| A49 AudioBlock 下采样/融合 causal=False ⇒ 非因果 | ⚠️ | 结论 ✅；但"每级 ±1/±2、跨级放大到 ±8 帧"低估：L0 k=3 ±1、L1 ±2→±4、L2 ±2→±8、**L3 ±2→±16**（换到 L0 单位），单遍 AudioBlock 实测 lookahead **57 帧=7,296 样本≈456 ms**（关 cLNhw 后，纯结构）。另外证据里 `:823` 是 `residual_conv`（k=1，无时间感受野），不宜作为非因果证据 |
| A50 Separator `dict.pop` 破坏配置，需 deepcopy | ✅ | 我 R1 §8-[T] 复现：同一 dict 二次构造 537,697→49,081 |
| A51 Falcon 用 torch.stft/istft 而非 layers 的 STFT | ✅ | `Falcon.py:1110,1182` |
| A52 Studio 在 MPS 上 CPU FFT bridge | ⚠️ | 结论 ✅；行号小错：编码器桥实际 **1111-1122**（1110 是注释，1123 已是卷积），decoder 桥 1183-1198 ✓ |
| A53 studio layers 不含 stft.py/stft_tfgn.py ⇒ 无 librosa | ✅ | `ls studio/falcon_model/layers/` = `__init__.py/activations.py/normalizations.py`；`layers/__init__.py` 仅一行 docstring |
| A54 训练/Studio 的 normalizations 仅注释差异、activations 0 行 | ✅ | `diff` 26 行（13 对），逐行核对全部为注释/行内注释文本差异；activations diff=0 |
| A55 losses/matrix.py 第三套 STFT（:13/:59/:114） | ✅ | `look2hear/losses/matrix.py:13,59,114` |
| A56 STFTEncoder 3,024 / STFTDecoder 2,016 | ✅ | 与我的 R1 逐模块统计一致（3·112·9=3,024；112·2·9=2,016） |
| A57 video_bottleneck Identity、0 参数 | ⚠️ | 结论 ✅（`kernel_size=-1` ⇒ `Identity`、0 参数）；证据行号 `45,66-68` 指向 **causal=True** 分支，`video_bottleneck` 走 causal=False 分支 ⇒ 应为 **84-85** |
| A58 523,113 + 14,584 = 537,697 | ✅ | 与我的 R1 完全一致：encoder 3,024 + audio_bn 13,104 + separator 492,312 + mask 12,657 + decoder 2,016 = 523,113；+spk 头 14,584 = 537,697 |

统计：✅ 51、⚠️ 6、❌ 1（A36）。

---

## 3. 错误清单

### P0（事实错误 / 会误导实现）

**P0-1｜A36「`STFTEncoder` 严格时间因果」错误，且 `[验证]` 证据不成立**

- 文档：A36（§6）与 §4.2 表格首行均称 `STFTEncoder` "✅ 严格因果"，证据 "扰动后半段，前 120 帧 max|Δ|=0.000e+00"。
- 事实：`STFTEncoder = torch.stft(center=True) + ConvNormAct(causal=True)`（`Falcon.py:1110-1119`）。STFT 的 `center=True`
  使帧 f 的窗口为 `[128f−128, 128f+128)`，因此**帧 125 及以后需要未来采样**。
- 我的复算（`verify_r2_causality_conditions.py` §[R2-C2]）：扰动采样 ≥16000 后，帧 123/124 的 `max|Δ|` **恰为 0**，
  **帧 125 = 3.913e+00**（后续帧同量级）；受影响帧列表从 125 开始。
- 为什么这是 P0：①结论与服务端实现直接相关（若有人据此认为 encoder 可逐采样输出，会做出错误的流式实现）；
  ②证据 `[验证]` 标签与"只测前 120 帧"不匹配（恰好避开 125 号泄漏帧），属"声称已验证但未真正验证该结论"；
  ③同文档 A37 与 §4.2 的 `torch.stft(center=True)` 行给出相反信息（内部矛盾），会让 FINAL 汇总时选错表述。
- 修订建议（可直接替换 A36）：
  > A36′：`STFTEncoder` 的 **conv 子层**严格时间因果（帧 0–123 在"扰动后半段音频"下 max|Δ|=0）；
  > 但整体因 `torch.stft(center=True)` 存在 **帧级 lookahead：帧 124 仍为 0、帧 125 起非零**，≤128 样本（8 ms）（见 A37）。

### P1（重要遗漏 / 误导 / 行号错）

**P1-①｜§4.1 的"采样级 lookahead 256 样本 = 16 ms"与 A38 矛盾，正确上界是 128 样本 = 8 ms**

- 文档 §4.1："采样级 lookahead（含 iSTFT 的 OLA 扩散）：输出采样 n 依赖的最后一帧为 ⌊n/128⌋+1，其窗覆盖到 n+256，
  即最坏一个窗长 = 256 采样 = 16 ms"；§4.2 表格 `torch.istft` 行也写 "⚠️ 帧级 ±win/2 扩散（同上 16 ms 量级）"。
- 事实（我推导 + 实测）：解码器因果分支下 `ConvTranspose2d` 的合成帧 f 只依赖**输入帧 f−3…f−1**
  （pad 2 帧 + k=3 + `p=1` ⇒ `y[n]=Σ x[n−3…n−1]`，`Falcon.py:1170-1177`）；`iSTFT(center=True)` 下输出采样 s 由合成帧
  `⌊s/128⌋` 与 `⌊s/128⌋+1` 叠加，后者只需输入帧 `≤ ⌊s/128⌋`。因此**端到端采样级 lookahead ≤128 样本（8 ms）**，
  不是 256。实测：改 embedding 帧 ≥125（=采样≥16000）后，输出 <16000 采样 max|Δ|=**0**；单帧脉冲（输入帧 100）→
  输出样本 `[12801,13311]`，是"该帧影响更晚的输出"（向右扩散 511 样本），不是"未来依赖"。
- 影响：与 A38 的"严格因果"口径打架；FINAL 若引用 16 ms 会与 Studio 的 80 ms guard 计算对不上。
- 修订建议：把 §4.1 的采样级结论改为"≤128 样本（8 ms，来自 STFT 帧网格；帧级因果严格，采样级因 center=iSTFT
  的 ±1 帧叠加有界）"；§4.2 `torch.istft` 行改为"⚠️ 端到端采样级 lookahead ≤128 样本（8 ms）"。

**P1-②｜A19/§4.2 的 cLNhw 泄漏数字缺探针条件；且漏掉"`CumulateLN4D` 抵消纯尺度泄漏"的关键机理**

Lead 特别要求对齐"02 的 1.2e-03 / 4.7e-05 vs 01 的 1.4e-06"。我的结论：**三个数字都不算错，但必须在文档里标注探针条件，
且 02 的 1.2e-03 不能代表 2 s 窗口量级**。可复现条件表（我的日志 §[R2-C]/[R2-C1]，fp32、eval 模式、阈值 `>1e-12`）：

| 被探针模块 | 扰动条件 | 输入形状 | 前缀 max\|Δ\| | 备注 |
|---|---|---|---|---|
| `LayerNormalizationHW(64)` 单模块 | +20 单帧尖峰 @t=T−2 | `[1,64,8,65]`（02 条件） | **9.93e-04**（02 报 1.202e-03，同量级，RNG 差异） | 02 的 A19 数字来源 |
| 同上 | 同上 | `[1,64,32,65]` | 2.92e-04 | 我的 T 敏感性补测 |
| 同上 | 同上 | `[1,64,125,65]` | 7.87e-05 | 同上 |
| 同上 | 同上 | `[1,64,251,65]` | 3.91e-05 | 同上 |
| `cLNhw(112)`（模型 pre_norm） | 后半段 ×10 | `[1,112,251,129]` | **1.9e-06** | 见下方机理 |
| `audio_bottleneck`（cLNhw+Conv1×1） | **后半段音频 ×10**，split 帧 120（02 条件） | 波形 `[1,32000]` | **3.22e-05**（02 报 4.739e-05，同量级） | 我完整复现其 §[7] 条件 |
| 同上去掉 cLNhw 的 1×1 conv | 任意 | 同上 | **0.000e+00** | 泄漏源确认为 cLNhw |
| `AudioBlock`（论文配置） | 后半段 ×10 | `[1,112,251,129]` | 7.14e+00 | 下游放大 |
| `AudioBlock`（norm 全关） | 后半段 ×10 | 同上 | 1.58e+00 | 纯结构 |

**关键新机理（02 与 01 都未写）**：`cLNhw` 的第一段 `GlobalLayerNorm` 全窗口统计确实非因果（单独测 ×10 探针前缀 Δ=**4.40**），
但其后的 `CumulateLN4D` 是沿时间累积的归一化，**会把"纯尺度型"泄漏几乎完全抵消**——`cLNhw` 整体在同一探针下只剩
**1.9e-06**。这解释了为什么"GN 是全窗口统计"却在小探针上只能测到 1e-6；也说明**模型级非因果主要来自下游的对称卷积
与多层归一化串联，而不是单层 GN**（`AudioBlock` 整测 ×10 探头 Δ=7.14）。02 的 §8.3-3 把首版只测到 7e-07 归因为
"GN 尺度不变的巧合"是对的，但应升级为"CumulateLN4D 的二段归一化抵消"这一结构性解释。

- 修订建议：把 A19 的证据改为"条件 + 数字"两列（如"T=8 尖峰探针 1.2e-3；T=251 波形×10 的 audio_bottleneck 4.7e-5；
  T=251 同分布替换 7e-7"），并在 §4.2/§4.4 补一句 CumulateLN4D 的抵消效应。
- 我这边（01）会在 R3 同步采用本表，保证 FINAL 只出现一套数字。

**P1-③｜A49「跨级放大到 ±8 帧」低估，且 `:823` 不是有效证据**

- 事实：每级对称卷积的 lookahead（换到 L0 帧单位）= L0 k=3 ×1 → **±1**；L1 k=5 → ±2×2 = **±4**；L2 → **±8**；
  L3 → ±2×8 = **±16**（`Falcon.py:751-767`，`k=3(i=0)`、`k=5(i>0)`）。我实测（关掉全部 cLNhw、纯结构）：
  **单遍 `AudioBlock` 前缀 lookahead = 57 帧 = 7,296 样本 ≈ 456 ms**（`docs/verification/runs/model-core-verify4.log` §[O]）。
- A49 证据里的 `Falcon.py:823` 是 `residual_conv`（k=1，无时间感受野），列为非因果证据会误导；应只列
  `:765`（downsample，k=3/5）、`:801`（fusion_layers，k=5）、`:814`（concat_layers，k=5）。

**P1-④｜A57 证据行号指向未使用分支**

`ConvNormAct(kernel_size=-1)` 的 `nn.Identity` 出现在两个分支：causal=True 分支 `:66-68`、causal=False 分支 `:84-85`
（`Falcon.py:45-85`）。`video_bottleneck` 的 `video_bn_params` 不含 `causal`，默认 False ⇒ 实际走 **`:84-85`**。
文档 A57 引 `45,66-68,1312`，应改为 `30,84-85,1312`（`:30` 是 `out_chan := in_chan` 的退化逻辑）。

### P2（细节瑕疵）

| # | 问题 | 证据 | 修正建议 |
|---|---|---|---|
| P2-1 | A03/A05 的"唯一 import 点/全仓无调用"未限定产品代码 | `docs/verification/scripts/verify_layers_dsp.py:34,71,80,97` 有 import/调用 | 改为"产品代码（`look2hear/`、`studio/`）中唯一/零调用；验证脚本除外" |
| P2-2 | §2.1 "cnnlayers（15 个名字）、rnnlayers（6 个）" 计数错 | `layers/__init__.py:1-16` AST 计数：cnnlayers **13**、rnnlayers **5** | 改成 13 / 5 |
| P2-3 | §4.7 "`import look2hear.models` 会拉入 … `pytorch_lightning`（仅 `BaseModel.serialize` 内部延迟导入）" 自相矛盾 | 其日志 §[0] 的 `sys.modules` 无 `pytorch_lightning`；`base_model.py:81` 是函数内延迟 import | 删掉该项或改为"不会在 import 时加载，仅 `serialize()` 内部按需导入" |
| P2-4 | §4.2 把 `interpolate(nearest)` 列为 `InjectionMultiSum` 非因果的致因之一 | nearest 上采样是单调映射（输出 t 只取 ⌊t·ratio⌋ 的源，过去/当前），本身因果；`Falcon.py:319-325` | 只保留"对称 k=5 卷积"；插值改为"不引入未来依赖" |
| P2-5 | A52 编码器桥行号 off-by-one | 实际：注释 `1110`、`compute_device` `1111`、`spec.to(compute_device)` `1122`（`1123` 已是卷积） | 写 `1111-1122`（或 `1110-1124` 含注释/卷积） |
| P2-6 | §8 说日志 177 行、验证记录说脚本 777 行 | 实际日志 **181 行**、脚本 **403 行**（`wc -l`） | 更新为实际行数或删掉行数（避免版本漂移） |
| P2-7 | 因果性表漏两条对"整个系统因果"必要的视频侧事实 | ①`VideoBlock` 的 `BatchNorm1d` 在 **train 模式**用批内（含未来帧）统计 ⇒ 训练期视频分支非因果（我 R1 实测 train 前缀 Δ=4.4e-2、eval=0）；②冻结 FRCNN 前端 `Conv3d(k=(5,7,7),p=(2,3,3))` 有 **±2 视频帧 = ±80 ms** lookahead（我 R1 实测最左受影响帧 28） | 在 §4.2 表补两行并注明"归属 04 域，交叉引用 `04-losses-metrics-video.md`" |

---

## 4. 遗漏的重要行为（代码里有、文档没写）

1. **`CumulateLN4D` 对 GN 泄漏的抵消效应**（见 P1-②）：这是解释"为什么 1e-6 与 1e-3 都真实"的机制核心，属本域职责。
2. **`cLNhw` 在 Falcon 模型里的实例密度**：实测 **30 个 `LayerNormalizationHW`**（= 30 个 `GlobalLayerNorm` + 30 个
   `CumulateLN4D`；来源：`audio_bottleneck` 1 + `projection` 1 + `downsample_layers` 4 + `fusion/concat` 各 3×8=24）。
   这量化了"全局统计在音频主干的每一层都出现"，是 §4.4 结论的最强支撑，建议补一行。
3. **`MHSA2D` 的因果 mask 由 `layer_3.causal` 控制，而顶层 `Falcon.causal` 是惰性开关**：我 R1 验证 `causal=True/False`
   两个模型参数量/state_dict/输出 **bit-identical**（同权重）。文档 §4.2 说"配置 causal 只注入部分模块"方向正确，
   但没点明"顶层开关完全无效"这一反直觉事实——它正是 Q1 的答案基础，也影响 FINAL 对"causal 声明"的表述。
4. **`STFTEncoder`/`STFTDecoder` 的 `causal` 来自 `enc_dec_params`，与 `Falcon.causal` 无关**（`Falcon.py:1308-1336`）：
   文档把三者混在 §4.2 行表里，建议在 §4.2 开头加一句"三个 causal 来源不同：`enc_dec_params.causal`（编解码）、
   `layer_3.causal`（注意力）、顶层 `Falcon.causal`（惰性）"。
5. **`utils/` 侧还有第 5 处 STFT 相关实现**：`look2hear/utils/stft.py` 之外，`look2hear/losses/matrix.py` 已有（A55），
   但 `look2hear/utils/complex_utils.py` + `nets_utils.make_pad_mask` 与 `stft_tfgn` 的配对关系只在 §2.4 提了一句，
   建议在 §2.12 的"第三/第四份"清单里也列出 `utils/complex_utils`（它才是 `torch_complex` 硬依赖的直接来源）。

---

## 5. 跨模块一致性问题（与 01 / 其它文档的冲突与统一建议）

| # | 冲突点 | 02 的表述 | 01（本评审人）的表述 | 判定 | 统一建议（供 FINAL） |
|---|---|---|---|---|---|
| C1 | encoder 因果性 | A36/§4.2："`STFTEncoder` 严格时间因果" | 01-A18：conv 严格因果，但整体 8 ms lookahead | **02 错**（P0-1） | 采用 P0-1 的 A36′；01 无需改 |
| C2 | decoder 采样级 lookahead | §4.1/§4.2：≤256 样本 / 16 ms | 01-A17：帧级严格因果（未谈采样级） | **两者都需修** | 统一为"帧级严格因果；采样级 ≤128 样本 / 8 ms（iSTFT 帧网格）；实测 128 整数倍分界处前缀 Δ=0"。我将在 01-R3 补该 nuance |
| C3 | cLNhw 泄漏数字 | 1.2e-03（T=8 尖峰）、4.7e-05（波形×10） | 1.4e-06（T=251 同分布替换） | **不矛盾但缺条件**（P1-②） | FINAL 只引用"条件表"（见 P1-②），并加 CumulateLN4D 抵消机理 |
| C4 | AudioBlock lookahead 量级 | "跨级放大到 ±8 帧" | 57 帧 ≈ 7,296 样本 ≈ 456 ms（关 cLNhw 纯结构） | **02 低估**（P1-③） | 统一为"L0 ±1 / L1 ±4 / L2 ±8 / L3 ±16（L0 帧单位）；单遍实测 57 帧；两遍串联 ≈114 帧（窗口内）" |
| C5 | 参数口径 | 523,113 + 14,584 = 537,697（A58） | 537,697；11,724,385 含 FRCNN | **完全一致** ✅ | 直接采纳 |
| C6 | TFRB / GCI 命名 | §1 给出 TFRB=三件套、GCI=`InjectionMultiSum`（均 `[推断]`） | 01 未使用这两个术语 | 非冲突（01 缺） | 01-R3 增补交叉引用；FINAL 采用 02 的映射（README 图注 "bidirectional LSTM along frequency, a unidirectional LSTM along time, then masked 2D self-attention" 支持 TFRB 定义） |
| C7 | 因果性结论的工程含义 | "必须用 2 s 窗口 + 右上下文；跨窗口靠 hop 与 80 ms guard 掩盖" | 01-§4.6：Studio 80 ms guard 与 FRCNN ±2 帧（80 ms）量级吻合 `[推断]` | 兼容 ✅ | FINAL 把"80 ms 右上下文 ↔ 视频前端 ±80 ms + 音频 8 ms/有界对称卷积"写成一条，需 06 域确认 OutputSlicer 语义（06-Q） |
| C8 | Studio 副本 diff 口径 | §4.6：normalizations 26 行注释 diff；A54 | 01 未涉及 | ✅ 一致 | 08/06 文档若引用同一数字需一致（26 行 = 13 对注释行） |

---

## 6. 给作者的问题清单（需回答或补充验证）

| 编号 | 问题 |
|---|---|
| Q1 | A36 的探针 `split_t=120` 是否刻意避开帧 125？能否补测"split=帧 125/126"并把结论改为 A36′（见 P0-1）？ |
| Q2 | A19 的 1.2e-03 来自 T=8 人工张量；是否愿意补一个"论文 2 s 窗口（T=251,F=129）"下的 cLNhw 数字（我的表可直接用）？ |
| Q3 | §4.1 "采样级 lookahead 256 样本/16 ms"的推导依据是什么？我推导+实测为 ≤128 样本（见 P1-①），请确认修正。 |
| Q4 | A49 的 "±8 帧" 是否笔误（L3 应为 ±16）？能否采用我的量化（单遍 57 帧）？ |
| Q5 | A57 的 `45,66-68` 是否想引用 causal=False 分支的 `84-85`？ |
| Q6 | §2.1 的 15/6 是否笔误（实际 13/5）？ |
| Q7 | §4.7 为何把 `pytorch_lightning` 列为 import 期加载（实测未加载）？ |
| Q8 | §8 的日志 177 行与验证记录里的"脚本 777 行"与实际（181/403）不符，是否有版本漂移？ |
| Q9 | §4.2 把 nearest 插值列为非因果致因，是否愿意改为"nearest 单调、因果；非因果来自对称 k=5 卷积"？ |
| Q10 | 是否愿意补 VideoBlock(train) 的 BN 非因果与 FRCNN ±80 ms 两行（属 04 域，可交叉引用）？ |

---

## 附：本轮复核执行记录

```bash
cd /root/code/github_repos/Falcon-fork
PY=$PWD/docs/verification/stubs:$PWD

# 0) 版本锚定
sha256sum docs/understanding/02-layers-dsp.md   # 评审开始时与结束时均为 ae01042f...
wc -l docs/understanding/02-layers-dsp.md       # 757

# 1) A01/A02：同源复制
diff -u look2hear/layers/stft.py look2hear/utils/stft.py | wc -l          # 15
diff look2hear/layers/stft.py look2hear/utils/stft.py | grep -c '^[<>]'   # 2（1 增 1 删）
for f in look2hear/layers/stft.py look2hear/utils/stft.py; do tr -d '[:space:]' < $f | md5sum; done
# 均 3f7053135a33126c8d2a078b98fc2927

# 2) A03/A04/A05：import / 调用点（含 docs 与 studio）
grep -rn "from .stft import\|layers.stft import\|from utils.stft import" --include=*.py .
grep -rn "forward_stft(\|inverse_stft(" --include=*.py . | grep -v docs/verification/runs
grep -rn "stft" --include=*.py studio/ | grep -v test      # studio 无 layers/stft 依赖

# 3) A52/A53/A54：Studio 侧
ls studio/falcon_model/layers/                              # __init__/activations/normalizations
diff studio/falcon_model/layers/normalizations.py look2hear/layers/normalizations.py | wc -l   # 44
diff studio/falcon_model/layers/activations.py      look2hear/layers/activations.py | wc -l    # 0
grep -n "compute_device\|stft_x = \|spec = spec.to\|output_device\|output = output.to" \
  studio/falcon_model/models/Falcon.py    # 1111/1112/1122 ... 1184/1185/1198

# 4) A41-A46：运行时核对
PYTHONPATH=$PY /root/.venvs/falcon/bin/python - <<'EOF'
# → out_proj Linear(48,64)/Linear(72,64)；默认 Comba → AttributeError；MHSA2D out (2,64,32,17)
# → γ 形状 (1,4,1,17)
EOF

# 5) 因果性条件表（本轮新增脚本）
PYTHONPATH=$PY /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r2_causality_conditions.py \
  > docs/verification/runs/r2-causality-conditions.log
```

**关键输出摘要**（详见 `docs/verification/runs/r2-causality-conditions.log`）：

- cLNhw 单模块：GN-only ×10 探针前缀 Δ=**4.4009**；`cLNhw` 整体同探针 **1.907e-06**（第二段归一化抵消）。
- cLNhw +20 单帧尖峰、T 敏感性：T=8→**9.930e-04**；T=32→2.918e-04；T=125→7.868e-05；T=251→3.910e-05。
- `audio_bottleneck`（后半段音频 ×10，split=120，复现 02 §[7] 条件）→ **3.22e-05**（02 报 4.739e-05，同量级）。
- `AudioBlock`（norm 全关）×10 → 前缀 Δ=**1.577**；论文配置 → **7.139**。
- encoder：帧 123/124 = 0、帧 125 = 3.913（→ P0-1）。
- decoder：改 embedding 帧 ≥125 → 输出 <16000 采样 **0.000e+00**；单帧脉冲输入帧 100 → 输出样本 [12801,13311]。
- 帧账本：251 → pad 253 → convT 253 → crop 252；(252−1)·128=32128 ≥ 32000。
- 模型内 `LayerNormalizationHW` 实例数 = **30**（`GlobalLayerNorm` 30 / `CumulateLN4D` 30 / `LayerNormalization4D` 60）。

**版本复核**：评审结束时 `sha256sum docs/understanding/02-layers-dsp.md` = `ae01042f95cef3e95961027b308b5e610a60f2b97b17dce0a7fad279a31bbd2f`（757 行），
与开始时一致，**被评审文档在评审期间未变更**。

---

## 附二：R3 期间对本文评审的**自我更正**（1 条，2026-10-21）

> 触发：R3 修订 01 时按 Lead 要求用真实权重/环境复跑 `import look2hear.models` 的依赖链，发现本文 §2（P2-2 判定）有误。

- **本文此前的判定**：P2-3「§4.7 称 `import look2hear.models` 会拉入 `pytorch_lightning`（实测未加载）」判为 **错误**，
  依据是 02 文档自己的日志 §[0] 的 `sys.modules` 清单里没有 `pytorch_lightning`。
- **R3 实测更正**：`import look2hear.models`（以及单独 `import look2hear.layers`）**确实会加载 `pytorch_lightning`**。
  链路：`Falcon.py:7` → `look2hear/layers/__init__.py:19-20` → `layers/stft_tfgn.py:10-12` → `look2hear/utils/__init__.py:12`
  → `utils/lightning_utils.py:3-4`（顶层 `from pytorch_lightning.utilities import rank_zero_only`）。
  实测 `sys.modules` 含 `librosa / torch_complex / typeguard / pytorch_lightning / rich`。
- **结论**：02 文档 §4.7 的**结论（会拉入 PL）是对的**，只有括号里的**归因**（"仅 `BaseModel.serialize` 内部延迟导入"）不对——
  应改为"经 `utils/lightning_utils.py` 顶层导入"。因此：
  1. 本文 P2-3 的严重度从"结论错误"降为"**归因错误**"（P2 仍然成立，但描述需改）；
  2. 我在 01 的 §5.3/A55/§8.4 中按实测写入了正确的链路与结论；
  3. **建议 02 作者在 R3 修订时把 §4.7 的归因改为 `utils/lightning_utils.py:3-4`**（保留"会拉入 PL"的结论）。
- 依据日志：`docs/verification/runs/model-core-r3-verify.log`（附加段）；命令：
  `PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python -c "import look2hear.models, sys; print([m for m in ('librosa','torch_complex','typeguard','pytorch_lightning') if m in sys.modules])"`。
