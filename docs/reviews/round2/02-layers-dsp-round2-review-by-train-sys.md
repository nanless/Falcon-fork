# 评审（R4 复核）：`docs/understanding/02-layers-dsp.md` —— 评审人 `train-sys`

> 复核对象：`docs/understanding/02-layers-dsp.md`（**R1 版，757 行**；
> sha256 `ae01042f95cef3e95961027b308b5e610a60f2b97b17dce0a7fad279a31bbd2f`）。
> 该 sha 与 `model-core` 的 R2 评审（`docs/reviews/round1/02-layers-dsp-review-by-model-core.md`）所据版本**相同**：
> 截至 2026-10-21 05:50，**02 尚未出现 R3 修订版**（mtime 04:20 未变）。因此本次复核与 01 的 R4 处理一致——
> 是"R4 执行级验证 + R2 遗留问题状态核对"，**不是对修订版的验收**；若作者提交 R3，需再做增量复核。
> 评审人范围（`00-round-plan.md` R4 表）：`train-sys` → 02，指定动作 = **"验证归一化/STFT 的因果性声明"**。
> 独立证据：`docs/verification/scripts/verify_train_r4_layers_causal.py`、`verify_train_r4_chain_causal.py`、
> `verify_train_r4_params.py`；日志 `docs/verification/runs/train-sys-r4-layers-causal.log`、
> `docs/verification/runs/train-sys-r4-crosscheck.log`。环境 `/root/.venvs/falcon/bin/python`
> （3.10.18 + torch 2.8.0+cu128，CPU/随机权重，未加载真实权重）；stubs 注入 `timm.DropPath`/`typeguard`。
> **未修改任何源码与被评审文档**。

## 1. 结论

**未通过（阻塞）——原因与 01 的 R4 相同：R3 未发生。**

- 02 的**核心 DSP/归一化事实与主要数字经我独立执行级复算全部成立**：两份 `stft.py` 去空白 md5 相同
  （`3f705313…`）、`forward_stft` vs `torch.stft` `max|Δ|=2.098e-05`；`CumulateLN/CumulateLN4D` 严格时间因果
  （改 t=16 → 首变恰 t=16）；`GlobalLayerNorm` 非因果（首帧 Δ=0.105）；`MHSA2D(causal=True)` 未来帧 0 泄漏
  vs `causal=False` 1.667；`DualPathRNN(dim=3)` 时间因果、`dim=4` 频率双向耦合；`STFTDecoder` 帧级严格因果；
  逐模块参数 `3,024/13,104/492,312/12,657/2,016=523,113` + spk 头 `14,584` = **537,697**（与 02 A56/A58 逐项吻合）。
- 但 R2 评审（`model-core`）标注的 **P0×1 + P1×4 在 R1 版中全部未修复**，且我的独立探针**再次复现**其中
  最关键的 3 条：①A36「STFTEncoder 严格时间因果」错误（我实测 s=16000 → 首变帧 125，即 128 采样/8 ms 前瞻）；
  ②§4.1「采样级 lookahead 最坏 256 采样=16 ms」错误（我实测 encoder→decoder 串联最大过去泄漏 **31 采样**，
  上界 <128 采样；且与扰动是否对齐帧有关）；③A57 证据行号指向 `causal=True` 分支（实际使用分支在 `:85`）。
- 另外我发现 **R2 评审本身有一条判断不成立**：R2 的 P2 说「§4.7『import 会拉入 pytorch_lightning』与实测不符」，
  我实测 `import look2hear.layers` 即已加载 `pytorch_lightning`（链路：`stft_tfgn.py` → `from ..utils.nets_utils...`
  → `look2hear/utils/__init__.py:12` → `lightning_utils` → PL）。02 的"会拉入 PL"结论正确，只是括号里的
  "仅 `BaseModel.serialize` 内部延迟导入"归因错。→ **R3 不可照抄 R2 该条，否则引入新错误**（见 §3-N1）。
- 结论：**需 `layers-dsp` 先完成 R3 修订**（P0-1、P1-①~④、§6 的 Q1–Q5），再由我按本文 §2/§3 的同一清单做增量复核；
  当前版本可作为"数据大体准确、但因果性表述有一个 P0 未修"的理解草稿进入 FINAL 参考。

**问题计数（对 R1 版）：P0 = 1（沿用 R2，未修复）、P1 = 4（沿用 R2，未修复）、P2 = 9（R2 的 7 条未修复 + 本文新增 2 条）。**

## 2. 执行级验证（R4 指定动作；独立脚本，不依赖 02 的 §8）

> 全部在 `/root/.venvs/falcon/bin/python`、CPU、随机权重下完成；直接实例化
> `models/Falcon.py` 的 `STFTEncoder/STFTDecoder/DualPathRNN/MultiHeadSelfAttention2D` 与
> `layers/normalizations.py` 的真实类，**不是复刻公式**。

| # | 检查 | 我的实测 | 02 的说法 | 判定 |
|---|---|---|---|---|
| E1 | 两份 `stft.py` 同源 | `diff`（非 -u）4 行 = 1 增 1 删；去空白 md5 两者均 `3f7053135a33126c8d2a078b98fc2927` | A01/A02 同 | ✅ 精确 |
| E2 | `import look2hear.*` 的 import 副作用 | `import look2hear.layers` 后 `pytorch_lightning`、`librosa`、`torch_complex`、`typeguard`、`distutils` 均已在 `sys.modules`；链路经 `look2hear.utils.__init__:12 → lightning_utils` | §4.7 称会拉入 PL（括号归因"仅 BaseModel.serialize 延迟导入"错） | ✅ 结论成立 / ⚠ 归因需改（并纠正 R2，见 N1） |
| E3 | `STFTEncoder` 因果性 | 扰动采样 s=16000 → encoder **首变帧 125**（帧 125 窗口 [15872,16128)，前瞻上界 128 采样=8 ms）；帧 <125 全 0 | A36 称「严格时间因果（前 120 帧不变）」 | ❌ **A36 结论错误**（复现 R2 P0-1）；"前 120 帧"恰好避开 125，探针无法支持严格因果 |
| E4 | encoder→decoder **串联**的采样级泄漏 | s=16000 → 首变输出 **16001**（无过去泄漏）；s=20000 → 首变输出 **19969**（泄漏 **31** 采样）；可解析验证最大泄漏 ≤127 采样 | §4.1 称「采样级 lookahead 最坏一个窗长 = 256 采样 = 16 ms」 | ❌ **§4.1 结论错误**（复现 R2 P1-①）；正确：帧级 8 ms，串联采样级 ≤127 采样（依赖对齐） |
| E5 | `STFTDecoder` 帧级因果 | 扰动 embedding 帧 125 → 首变输出 16001（即 ≥16000） | A38 称严格时间因果 | ✅ |
| E6 | `CumulateLN` / `CumulateLN4D` | 改 t=16 → 首变恰 t=16（max\|Δ\|≈5）；3D/4D 均如此 | A20/§4.4 因果 | ✅ |
| E7 | `GlobalLayerNorm` / `cLNhw` 非因果 | GLN 改最后帧单通道 +5 → 首帧 Δ=**1.054e-01**（强非因果）；`LayerNormalizationHW` 直接输出早期帧泄漏仅 **1.4e-6**（T=8）/ **4.8e-7**（T=251），远小于 02 的 1.2e-3 → 证明"量级强依赖探针"，并复现机理：**所有通道同量扰动会被 LN-over-C 的 common-mode 抵消**（全通道 +5 → 输出差 9.5e-07；单通道 +5 → 5.0） | A19 称 1.2e-03；§4.2 称 4.7e-05 | ✅ 定性结论成立 / ⚠ 数字缺探针协议（复现 R2 P1-②，并补上抵消机理） |
| E8 | `MHSA2D` 因果 mask | `causal=True`：改最后时间帧 → 之前帧 Δ=**0.000e+00**；`causal=False`：Δ=1.667 | A44 | ✅ |
| E9 | `DualPathRNN` 方向 | `dim=3` 改最后时间帧 → 之前帧 Δ=0（时间因果）；`dim=4` 改最后频点（**单通道、非全通道同量**）→ 其余频点 Δ=6.3e-2（频率双向耦合） | A40/A41 | ✅（方法注意：全通道同量扰动会被 `LayerNormalization4D` 抵消，首版探针因此误得 0） |
| E10 | `forward_stft` vs `torch.stft` | center=True/False：shape `[2,129,251,2]`/`[2,129,249,2]`，`max|Δ|=2.098e-05` | A08/A11/A14：2.289e-05（同量级） | ✅ |
| E11 | 参数账本 | encoder 3,024 / audio_bottleneck 13,104 / separator 492,312 / mask_generator 12,657 / decoder 2,016 → 523,113；spk_pool_proj 14,584；总 **537,697** | A56/A58 | ✅ 逐项精确 |
| E12 | `CumulateLN` 定义两次 | `normalizations.py:51` 与 `:91` 两处 | A25 | ✅ |
| E13 | A57 证据行号 | `causal=False`（`video_bottleneck` 实际走的分支，`causal` 未传）的 Identity 在 **`Falcon.py:85`**；02 引的 `66-68` 是 `causal=True` 分支的 else（`:67`） | A57 引 `66-68` | ❌ 行号错（复现 R2 P1-④；精确应为 `:85`，R2 说"84-85"略偏） |
| E14 | `layers/__init__` 导入名字数 | `cnnlayers` **13** 个（TAC…Bottomup_Concat_Topdown_TCN）、`rnnlayers` **5** 个（DPRNN/DPRNNBlock/DPRNNLinear/LSTMBlockTF/TransformerBlockTF） | §2.1 称 15 个 / 6 个 | ❌ 数字错（复现 R2 P2；02 §2.1 需改 13/5） |
| E15 | 全模型因果基线（复跑 Lead 的 `verify_causality.py`） | 扰动 s=20000 → 全模型 `first_changed_sample=9612`（过去泄漏 −649 ms），separator first_frame=0 | §4.2「Separator 整体非严格因果」 | ✅ 定性成立（泄漏来源应归因 cLNhw；R3 需写清） |

**未独立复核项**：`AudioBlock` 去 cLNhw 的 57 帧 lookahead（R2 已实测，属 model-core/01 域，我未重跑）；
MPS 设备实测（沙箱无 Apple GPU，Studio bridge 只做静态核对）；真实权重下的数值。

## 3. R2 遗留问题状态（对 R1 版：全部未修复）

### P0-1｜A36「`STFTEncoder` 严格时间因果」错误（沿用 R2，本文独立复现并给出正确表述）
- 02 证据：A36 声称"扰动后半段音频，前 120 帧输出完全不变"（其日志 §[7]），但**前 120 帧恰好避开了真正
  存疑的帧 120–125**。我的 E3：扰动 s=16000 → 首变帧 **125**，帧 125 的 STFT 窗覆盖 [15872,16128)，
  即 encoder 对输入有 **≤128 采样 = 8 ms 的帧级前瞻**（来自 `torch.stft(center=True)`，`Falcon.py:1110-1116`）。
- 这同时与 02 自己 A37 的"帧级 ≤8 ms"、§4.2 表格自相矛盾。
- R3 修法（建议）：A36 改为 **A36′**："`STFTEncoder` 的 conv 部分严格时间因果（pad 2 后裁 2），
  但前置 `torch.stft(center=True)` 带来 ≤128 采样/8 ms 帧级前瞻 ⇒ 整体非严格逐采样因果"，
  并引用 E3/01-A18 的同一数字。

### P1-①｜§4.1「采样级 lookahead 256 采样 = 16 ms」错误（沿用 R2；本文给出更精确上界）
- 02 §4.1（`:426-431`）推导："输出采样 n 依赖最后一帧 ⌊n/128⌋+1，其窗覆盖到 n+256 ⇒ 16 ms"。
  我的 E4 实测 encoder→decoder 串联：最大过去泄漏 31 采样（s=20000），对齐时 0 泄漏（s=16000）；
  解析上界 = 127 采样 < 128。**正确表述**：帧级 lookahead 8 ms；encoder→decoder 串联的采样级过去泄漏
  ≤127 采样（与扰动相对帧边界的相位有关），**不存在 16 ms 的采样级前瞻**。
- 该条与 A38（decoder 严格因果）、A37（帧级 8 ms）一起构成一致的三段论；R3 请统一改写 §4.1。

### P1-②｜A19/§4.2 的 cLNhw 泄漏数字缺探针条件（沿用 R2；本文补齐"抵消机理"）
- 02 的 1.2e-03 来自其人工张量（T=8、F=65、单帧尖峰 +20）；我的直接 `LayerNormalizationHW` 探针在
  单通道尖峰下只有 1.4e-6（T=8）/4.8e-7（T=251），与 02 相差 2–3 个量级 → 数值强依赖探针。
- **缺的机理**（R2 指出、我用实验补齐）：`GlobalLayerNorm=GroupNorm(1,C)` 的统计量与 `CumulateLN4D`
  的累积统计都会**部分抵消"所有通道同量"的扰动**——全通道 +5 时输出差仅 9.5e-07（被 LN-over-C 当 common-mode
  减掉），而单通道 +5 时为 5.0。解释"1e-6 与 1e-3 都真实"的关键正在于此。
- R3 修法：A19/§4.2 附"扰动协议表"（T、扰动位置/幅度、单通道 vs 全通道、测点层级、是否去 cLNhw），
  并把机理写成一句话；FINAL 只引用协议表口径。

### P1-③｜A49「跨级放大到 ±8 帧」低估（沿用 R2；我未重跑，保持 R2 证据）
- R2 实测（去 cLNhw 纯结构）：单遍 `AudioBlock` → 输出首变帧 68（扰动帧 ≥125），即 **57 帧 lookahead**，
  两遍串联 ≈114 帧；02 的"±8 帧"低估。属 01/model-core 域，R3 请引用 R2 的测量并改写量级。

### P1-④｜A57 证据行号错（沿用 R2；本文精确为 `:85`）
- 见 E13：`causal=False` 分支的 `self.conv = nn.Identity()` 在 `Falcon.py:85`，02 引 `66-68` 是
  `causal=True` 分支的 else（`:67`）。`video_bottleneck` 未传 `causal`（默认 False）→ 实际走 `:85`。

### P2 遗留（R2 的 7 条，均未修复；我复核其中 2 条）
- §2.1 "cnnlayers 15 个名字 / rnnlayers 6 个" → 实际 **13 / 5**（E14）。
- §4.7 "import 会拉入 pytorch_lightning"（结论成立，但括号归因错，见 N1）。
- A03/A05 "唯一 import 点/全仓无调用"未限定"产品代码（`look2hear/`、`studio/`）"（R2）。
- §4.2 把 `interpolate(nearest)` 列为非因果致因之一有误（最近邻插值对时间是单调映射，保守讲不引入
  时间非因果）（R2）。
- A52 Studio bridge 行号 off-by-one（R2；与 06 文档对拍后修正）（R2）。
- §8 日志行数/脚本行数与实际不符（R2）。
- 遗漏 `VideoBlock(train)` 的 BN 非因果与 FRCNN ±80 ms（R2）。

### 本文新增

**N1｜§4.7 的 PL 归因错，且 R2 的否定判断不成立（双向修正）**
- 02 §4.7 称 "`import look2hear.models` 会拉入 … `pytorch_lightning`（仅 `BaseModel.serialize` 内部延迟导入）"。
  - "会拉入 PL" **正确**：我的 E2 在干净进程实测 `import look2hear.layers` 后 `pytorch_lightning` 已在
    `sys.modules`；链路 = `layers/__init__.py:20` → `stft_tfgn.py:12 from ..utils.nets_utils import make_pad_mask`
    → 执行 `look2hear/utils/__init__.py` → `:12 from .lightning_utils import ...` → `import pytorch_lightning`。
  - 括号归因 **错误**：不是 `BaseModel.serialize` 的延迟导入，而是 `utils/__init__` 的**顶层**导入。
  - **R2 评审该条（"与实测不符"）不成立**，R3 不要照抄。
- R3 修法：改为"通过 `utils/__init__ → lightning_utils` 顶层拉入 PL；`BaseModel.serialize` 另有自己的延迟导入"。

**N2｜cLNhw 探针的 common-mode 抵消未记录为方法论**（并入 P1-② 一起修即可，单列以便 R3 检查）
- 见 E7/E9：全通道同量扰动会被 `GlobalLayerNorm`/`LayerNormalization4D` 的 channel 维统计抵消，
  这是 02 的 1.2e-3 与 R2/我的 1e-6 级差异的**主要来源之一**；R3 的"扰动协议表"必须注明"单通道 vs 全通道"。

## 4. 遗漏的重要行为（代码里有、02 没写）

1. **encoder→decoder 的因果补偿**：decoder 的 `F.pad(k−1)`+`crop` 偏移（`Falcon.py:1170-1177`）恰好抵消
   `torch.stft(center=True)` 的帧级前瞻，使"encoder+decoder 串联"的采样级过去泄漏 <128 采样（E4）——
   这是解释"为什么整链看似有 8–16 ms 前瞻、实际采样级几乎不泄漏"的关键，02 只分别描述两端，未给串联结论。
2. **归一化探针的 common-mode 抵消机理**（E7/E9、N2）：决定了任何因果性数字是否可复现，必须写进 §4.2/§6。
3. **`import look2hear.layers` 即拉入 PL/librosa/torch_complex/typeguard/distutils 的完整链路**（E2/N1）。
4. **`LayerNormalization4D((C,1))` 在 `DualPathRNN` 里按 channel 归一化会消除"全通道同量扰动"**——评估
   DualPathRNN 的频率耦合时必须逐通道扰动（E9），否则会误判"无耦合"（我首版探针就误得 0）。
5. `DualPathRNN(dim=4)` 的 `out_proj` 输出在 `x + residual` 中占比很小（本探针 residual 差 5.0、branch 差
   6e-2 量级）——做"频轴双向"结论时应用 `rnn` 输出的直接 hook（我改用单通道扰动后成立）。

## 5. 跨模块一致性问题（02 vs 01/03/04/05/06）

| # | 冲突点 | 02 说法 | 对照文档/我的证据 | 判定 / 统一建议 |
|---|---|---|---|---|
| C1 | encoder 因果性 | A36「STFTEncoder 严格时间因果」 | 01-A18：「conv 严格因果，整体 8 ms lookahead」；我的 E3 | **02 错**；采用 A36′（E3 数字） |
| C2 | 采样级 lookahead | §4.1「256 采样=16 ms」 | 01/R2/我的 E4：≤128 采样（实测最大 31）；A37/A38 一致性 | **02 错**；统一为"帧级 8 ms；串联采样级 ≤127" |
| C3 | cLNhw 泄漏量级 | 1.2e-03（T=8 尖峰）/4.7e-05（×10） | 01：1.4e-06（T=251 同分布）；我的 E7：1.4e-6/4.8e-7（单通道） | 不矛盾但缺协议；FINAL 用"协议表"（P1-②） |
| C4 | AudioBlock lookahead | "±8 帧" | R2 实测 57 帧（关 cLNhw，单遍）；两遍 ≈114 帧 | **02 低估**（P1-③，引用 R2） |
| C5 | 与 `03-training-system` | layers 因果结论 | 03 §1.1#6/§4.x 的"非严格因果、窗口化推理"依赖本域结论 | 无冲突；03 的"窗口内近似因果"应以本域 A36′/§4.2 修正版为准 |
| C6 | 与 `06-studio-backend` | §4.6/A52-A54 Studio CPU FFT bridge 描述 | 06 §4.8/A45 有逐行 diff（净 +9 行、MPS 搬运位置） | 结论一致；02 的 A52 行号按 06 更正 |
| C7 | 与 `04-losses-metrics-video` | A55「losses/matrix.py 是第三套 STFT」 | 04 的损失域独立实现 | 无冲突 |
| C8 | 与 `05-data-pipeline` | 无直接交叠 | — | 无冲突 |

## 6. 给作者（layers-dsp）的问题清单（R3 必答）

1. **Q1** 是否接受把 A36 改为 **A36′**（conv 严格因果 + `torch.stft(center=True)` 的 ≤128 采样/8 ms 帧级前瞻，
   整体非严格逐采样因果）？探针请补"s=16000/20000"两点（E3），不要只测"前 120 帧"。
2. **Q2** §4.1 的"采样级 256 采样/16 ms"是否同意改为"帧级 8 ms；encoder→decoder 串联采样级过去泄漏 ≤127
   采样（与对齐有关，实测 31）"？（E4）
3. **Q3** A19/§4.2 是否补"扰动协议表"（T、单帧/整段、单通道/全通道、改后测点）并写明 common-mode 抵消机理？
   （E7/N2）
4. **Q4** A57 行号 `66-68` → `:85`；A03/A05 的"唯一/全仓"是否加"产品代码范围"限定；§2.1 的 15/6 → 13/5；
   §4.7 的 PL 归因按 N1 改写（并**不要**采纳 R2 的"与实测不符"结论）。
5. **Q5** A49 的"±8 帧"是否改为引用 R2 的 57 帧（单遍，去 cLNhw）/两遍 ≈114 帧？若你重测，请注明
   是否去 cLNhw（否则 cLNhw 泄漏会淹没结构 lookahead）。

## 7. 结束校验

- 复核对象 sha256（评审结束时重算）：`ae01042f95cef3e95961027b308b5e610a60f2b97b17dce0a7fad279a31bbd2f`（与开始时一致，未改动）。
- 本复核新增文件：`docs/reviews/round2/02-layers-dsp-round2-review-by-train-sys.md`；
  脚本 `docs/verification/scripts/verify_train_r4_layers_causal.py`、`verify_train_r4_chain_causal.py`、
  `verify_train_r4_params.py`；日志 `docs/verification/runs/train-sys-r4-layers-causal.log`、
  `docs/verification/runs/train-sys-r4-crosscheck.log`。
- 未修改任何源码、未修改 02 文档。
