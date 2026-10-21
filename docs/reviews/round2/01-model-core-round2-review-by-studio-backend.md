# 评审（R4 复核）：`01-model-core.md` —— 评审人 `studio-backend`

> 复核对象：`docs/understanding/01-model-core.md`（**R1 版，807 行**；sha256 `cd58a08cd7775b8ec8fb91ce13b5f9e053977afb981f0b3bd2a6c8f4cd11b30b`）。
> 该 sha 与 `layers-dsp` 的 R2 评审（`docs/reviews/round1/01-model-core-review-by-layers-dsp.md`）所据版本**相同**：
> 截至 2026-10-21 05:46，**01 尚未出现 R3 修订版**（mtime 04:28 未变）。本复核因此是“R4 执行级验证 + R2 遗留问题状态核对”，
> 不是对修订版的验收；若作者随后提交 R3，我需再做一次增量复核。
> 评审人范围（00-round-plan R4 表）：`studio-backend` → 01，指定动作 = **“用 Lead 的脚本复算参数量/形状，验证‘因果’表述”**。
> 证据：我的独立脚本 `docs/verification/scripts/r4_review_01_exec.py`，日志
> `docs/verification/runs/r4-review-01-exec.log`；环境 `/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128，
> CPU/随机权重；stubs 注入 `timm.DropPath`）。未修改任何源码/被评审文档。

## 1. 结论

**未通过（阻塞）——原因不是 01 的数据错误，而是流程缺失：R3 未发生。**

- 01 的**核心数据断言经我独立执行级复算全部成立**：参数量 537,697 / 11,724,385、逐模块账、`video_bottleneck` Identity、
  金字塔形状 `(251,129)→(126,65)→(63,33)→(32,17)`、STFT 8 ms 自然前瞻（帧 125 首变）、AudioBlock 去 cLNhw 后
  **57 帧 lookahead（实测 125−68=57，精确命中）**、顶层 `causal` 开关惰性（bit-identical）、`_mdcl_loss` 缺失路径、
  `n_src=2` RuntimeError 等（§4）。
- 但 R2 评审（`layers-dsp`）标注的 **P0×1 + P1×3 在 R1 版中全部未修复**，且我的 hook 计数**再次独立证实 P0-1**：
  `AudioBlock.concat_layers[2]` 在 forward 中**从未被调用**，其 **5,568 参数（占 0.538 M 的 1.04%）全程闲置**（§3-P0-1）。
  因此按“R4 无 P0 遗留”的验收口径，01 目前**不能通过**。
- 结论：**需 model-core 先完成 R3 修订**（回答 R2 的 P0/P1 + 本文 §6 的补充），再由我按 §2/§3 的同一清单做增量复核；
  在当前版本上，01 只能作为“数据准确但含 1 个未修 P0 的理解草稿”进入 FINAL 参考。

**问题计数（对 R1 版）：P0 = 1（沿用 R2，未修复）、P1 = 3（沿用 R2，未修复）、P2 = 4（本文新增）。**

## 2. 执行级验证（R4 指定动作；独立脚本，不依赖 01 的 §8）

> 全部在 `/root/.venvs/falcon/bin/python`、随机权重、CPU 下完成；`model = FalconSpeakerContrastive(**configs/Falcon.yml 的 audionet_config)`。
> 与 01 的 §8 脚本无关，属于“第二名 agent 独立复算”。

| 检查 | 我的实测 | 01 的说法 | 判定 |
|---|---|---|---|
| 参数总量 | audio 537,697 + video 11,186,688 = **11,724,385** | A02 同 | ✅ |
| 逐模块参数 | encoder 3,024 / audio_bottleneck 13,104 / separator 492,312（audio_net 445,766 + video_net 42,850 + caf 3,696）/ mask_generator 12,657 / decoder 2,016 / spk_pool_proj 14,584 / video_bottleneck **0** | A03/A05 同 | ✅ |
| `caf.v_proj` | `Conv1d(512,112,k=1,groups=16)`，输出 `(1,112,50)` | A13 | ✅ |
| 金字塔降采样形状（2 次 AudioBlock 调用） | `(251,129),(126,65),(63,33),(32,17)` ×2 | A08 | ✅ |
| STFT 编码器未来依赖 | 改 `samples≥16000` 后 encoder 帧 **121–124 Δ=0**、**帧 125 首变** | A18（"帧 123/124=0、125 起非零"） | ✅ 精确 |
| AudioBlock 去 cLNhw 的 lookahead | 扰动帧≥125 时输出**首变帧=68 ⇒ 125−68 = 57 帧**；max\|Δ\|=4.62 | A20（57 帧=7,296 样本≈456 ms） | ✅ 精确 |
| 顶层 `causal` 开关 | `causal=True` vs `False`：state_dict keys 同、参数同 537,697、**输出 bit-identical（max\|d\|=0）** | A25 | ✅ |
| 死 MDCL 路径 | `Falcon._mdcl_loss_hard=True`、`Falcon._mdcl_loss=False`；`Falcon.train()` + `near_audio=` → **AttributeError** | §4.9-⑭ / A? | ✅ |
| `n_src=2` | 前向 `RuntimeError: shape '[2,224,251,129]' is invalid...` | A16 | ✅ |
| 前向形状链 | encoder/audio_bottleneck `(1,112,251,129)`、separator `(1,112,251,129)`、mask `(1,1,112,251,129)`、decoder `(1,1,32000)` | A06/A15 | ✅ |
| AudioBlock/VideoBlock/caf 调用次数 | 一次 `Separator.forward`：AudioBlock 内部各层 ×2（=2 次 AudioBlock）、separator×1；**concat[2] 0 次** | A11（AudioBlock 2、VideoBlock 1、caf 1） | ✅（A11）；❌（§3.2 的 concat 计数，见 P0-1） |

**未独立复核项（保持 01 原标注）**：L=8000/16000/48000 的等长输出（我只跑 L=32000）、A37 死类 `named_modules`、
A07 train 返回 tuple、A19 的“1.4e-6”精确数值（我测到的是全模型/不同协议下的量级，见 §3-P1-3）、A24 的 8 段数值表。
原因：无真实权重/GPU，且本轮以“复核 R2 遗留 + 关键因果表述”为目标；这些项 01 已有 [验证] 记录，我无相反证据。

## 3. R2 遗留问题状态（对 R1 版：全部未修复）

### P0-1｜`concat_layers[2]` 从未被调用，5,568 参数闲置（沿用 `layers-dsp` R2 评审，本文独立复现）

- 01 §3.2 的调用链描述与 A10 隐含“解码器逐级融合 level2→1→0 都执行”。实际 `AudioBlock.forward`
  （`look2hear/models/Falcon.py:860-888`）：
  `expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1])` 用 index 3；
  随后 `for i in range(self.upsampling_depth - 3, -1, -1)` 用 index 1、0 —— **index 2 被跳过**。
- 我的 hook 计数（一次 `Separator.forward`，AudioBlock 被调 2 次）：
  `{'concat[0]': 2, 'concat[1]': 2, 'concat[3]': 2, 'downsample[*]': 2, 'fusion[0..3]': 2, 'process_f[0..3]': 2, 'separator': 1}`
  —— **没有 `concat[2]` 条目 = 0 次**；`concat_layers` 每个 `InjectionMultiSum` = **5,568** 参数，
  即 **5,568 参数闲置**（0.538 M 的 1.04%）。
- 这同时影响 01 的三处结论：§3.2 调用计数、§4.7 参数表对 `concat_layers` 的解释、§4.9 死代码清单（未登记该参数）。
  建议 R3：①修正调用计数；②明确这是“代码少了一次 level-2 融合（潜在 bug）”还是“有意跳过”，并登记 5,568 闲置参数；
  ③A10 的“每级调用次数 1”需限定为 `process_f/t/att` 与 `downsample/fusion`，`concat` 除外。

### P1-1｜A21 “InjectionMultiSum 脉冲响应 ±2 帧”只在**去掉 cLNhw** 后成立（沿用 R2）

- as-built 下 `InjectionMultiSum` 的三个门（local/global embedding、gate）都含 `LayerNormalizationHW`（`Falcon.py:283-329`），
  全局统计会把脉冲扩散到全窗。我用全模型单点扰动（s=20000）也观察到输出 **sample 1 即有非零 Δ（3.4e-6）**，
  与“±2 帧（±256 样本）”描述矛盾；01 需要在 A21 像 A20 一样显式声明“去 cLNhw 口径”。

### P1-2｜A44 “Studio 副本仅注释/DropPath 来源不同，逻辑等价”不完整（沿用 R2；本域可给出精确证据）

- `studio/falcon_model/models/Falcon.py` 相对 `look2hear/models/Falcon.py` **净 +9 行**，其中功能性改动是
  **Metal FFT bridge**：`STFTEncoder.forward` 在 MPS 时把 STFT 搬到 CPU（studio `:1110-1122`），`STFTDecoder.forward`
  把复数构造/istft 搬到 CPU（`:1183-1198`）——不是“仅注释/DropPath”。我的 06 文档 A45/§4.8 已逐行验证。
- 建议 R3 把 A44 改成“注释/DropPath + MPS CPU FFT bridge（MPS 专用，CPU/CUDA 数值等价）”，并引用 06 §4.8。

### P1-3｜A19 的 1.4e-6 与 02 文档 4.7e-5 的差异需给“扰动协议”说明（沿用 R2）

- 两者不是矛盾而是协议不同（改哪段/改动幅度/mouth 是否固定）。建议 A19 附一行“本数值的扰动协议”，
  否则 FINAL 读者会误判。

### 新增 P2（本文）

- **P2-1｜§5.3 的 `falcon_runtime` 架构校验描述只写了 small**：原文“校验 `config["audionet"]["audionet_name"]`
  必须等于 `FalconSpeakerContrastive`（`falcon_runtime.py:152-156`）”。实际是**与 `VARIANTS[variant]["architecture"]`
  比较**：small=`FalconSpeakerContrastive`，big=`SpeakerContrastiveV2`（`falcon_runtime.py:31-37,152-154`）。
  否则读者会以为 Big 也强制 small 类。建议改为“必须等于该 variant 配置的 architecture”。
- **P2-2｜§4.7 参数表的 `concat_layers ×4 = 22,272` 未区分“实例化 4 / 调用 3”**（与 P0-1 同根）。
- **P2-3｜A18 的“帧 123/124=0”可写得更精确**：STFT(`center=True`) 下首变帧 =
  `ceil((s−128)/hop) = ceil((16000−128)/128) = 125` ⇒ 帧 **≤124** 全 0（我实测 121–124 全 0）。建议给出该公式，
  避免“为什么正好 123/124”的疑问。
- **P2-4｜§4.6-③ 的“pre_norm 左段 Δ=1.4e-6”属单层协议**，而 §4.6-⑥ 的全模型表在 0–0.12 s 给出 5.1e-7；
  两者混排时需说明“单层/全模型”口径，否则读者会以为同一实验两个数。

## 4. 遗漏的重要行为 / 对“完全理解”的建议（≥3 条）

1. **`concat[2]` 闲置是“参数/算力账”的实质缺口**：01 把 0.5 M 可训练参数的故事讲得很完整，但其中 **5,568（1.04%）从不参与前向**；
   若 FINAL 声称“0.5 M 全部用于推理”，需要显式排除或确认 bug。建议在 §4.7 表尾加“闲置参数”一行并在 §4.9 登记。
2. **`Falcon`（裸类）与 `FalconSpeakerContrastive` 的 forward 契约差异未单独成表**：01 §5.2 已给三态表，但“裸 `Falcon` +
   `near_audio` → AttributeError”只在死代码段落出现；建议并入 §5.2 的契约表（训练脚本若误用裸类会直接崩）。
3. **“因果”结论的工程可操作性**：01 已给“若要做逐采样流式必须改 cLNhw/卷积/stft/KV-cache”（§4.6-⑦），很好；
   建议补一张“若要流式，改动点 → 影响参数/数值 → 需要重训？”的表，便于 FINAL 回答“为什么 Studio 只能窗口化”。
4. **与 06/07 的右上下文呼应**：01 §4.6-⑤ 已提“Studio 的 80 ms 右上下文与 ±80 ms 量级吻合 [推断]”；
   06 文档 A45/§5.5 已给出“80 ms=1280 样本”的精确切片语义，建议 01 直接引用 06 §5.5，避免 FINAL 二次推断。

## 5. 跨模块一致性

| 项 | 01 说法 | 06/其它域 | 一致性 |
|---|---|---|---|
| 参数 537,697 / 11,724,385；state_dict 525 keys | §4.7/A02/§5.4 | 06 §4.4/§8-A2/A36（525/137 keys、11,724,385） | ✅ 互证 |
| Studio 副本 `Falcon.py` 差异 | A44 “仅注释/DropPath” | 06 §4.8/A45：+9 行含 MPS bridge（`studio/.../Falcon.py:1110-1122,1183-1198`） | ❌ 01 需补（P1-2） |
| `FRCNNVideoModel.train()` 不带 `return self`（`eval()` 返回 None） | §5.3 | 06 §4.3/A41、`falcon_runtime.py:177-180` 注释 | ✅ 互证 |
| 窗口 2 s / hop 0.5 s / 80 ms 右上下文 | §4.6-⑤/⑦ | 06 §1.3/§5.5 | ✅ 互证（01 是 [推断]，06 是精确实现） |
| `audionet_name` 校验 | §5.3 只提小模型 | 06 §4.9-2 / `falcon_runtime.py:152-154`：按 variant 的 architecture 校验 | ⚠️ 01 需精确化（P2-1） |
| `n_src=1` 隐含约束 | §5.1/A16 | 06 配置 `n_src:1`（`studio/configs/Falcon.yml:12`） | ✅ |
| 权重加载策略 strict 差异 | §5.4 | 06 §4.3（strict=True；评测 strict=False、weights_only=False） | ✅ 互证 |

## 6. 给作者（model-core）R3 必做清单

1. **修 P0-1**：§3.2 调用链删掉/限定 `concat_layers[2]`；§4.7 参数表加“concat[2] 5,568 闲置”；§4.9 死代码清单登记；
   并在 §4.9 或 §7 明确“level-2 融合缺失是否为潜在代码 bug（需作者确认）”。
2. **修 P1-1/P1-2/P1-3**：A21 加“去 cLNhw 口径”；A44 补 MPS CPU FFT bridge 并引用 06 §4.8/A45；
   A19 加扰动协议说明。
3. **P2**：P2-1 `audionet_name` 按 variant 校验；P2-2 参数表区分实例化/调用；P2-3 STFT 首变帧公式；
   P2-4 单层/全模型口径。
4. **提交 R3 后通知 Lead 与我**：我将按本评审 §2/§3 做增量复核（只需重跑 `r4_review_01_exec.py` 的 hook 计数 +
   diff 三处文本，约 1 分钟）。
5. **回答**：`concat[2]` 是“有意跳过”还是“AudioBlock.forward 少了一次 level-2 融合”？这决定它是“文档遗漏”还是
   “训练代码 bug + 0.5 M 口径失真”，FINAL 必须给结论。

## 7. 复核记录

- 复核范围：01 全文 §1–§9（重点 §3.2、§4.6、§4.7、§4.9、§5.2–§5.4、§6 断言表、§8 脚本清单）。
- 独立执行：`docs/verification/scripts/r4_review_01_exec.py`（参数账/调用计数/因果/开关/死路径/形状链）；
  Lead 的 `verify_model_params.py`、`verify_causality.py` 复跑（`docs/verification/runs/r4-prep-studio-backend.log`：
  单采样点扰动首变 9612（−649 ms）、encoder 首变帧 156 对应 ±128 自然前瞻、separator 首变帧 0 为归一化泄漏）。
- 未修改源码/被评审文档；`git status` 仍只有 `docs/` 未跟踪。
- 局限：无真实权重/GPU；A07/A19/A24/A37 等未独立复跑（保持作者 [验证] 标注）；01 的 R3 尚未出现（本复核对 R1 版）。

---

## 增量复核（基于 R3 sha `afe00188…`，2026-10-21 07:10 UTC）

> 对象：`docs/understanding/01-model-core.md` **R3**（975 行，sha256
> `afe00188097e4d7d5ada48dbcdd96703248ea7942233f3b699658e48d2db0058`，与我 round2 评审所据 R1 版 `cd58a08…`（807 行）不同）。
> 方法：①逐条核对 R3 §9.1 对我 round2 评审 P0/P1/P2/建议的响应；②独立执行验证（新脚本
> `docs/verification/scripts/r4_incremental_01_r3.py`，日志 `docs/verification/runs/r4-incremental-01-r3.log`）：
> 使用 Lead 提供的本地真实权重 `/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`（未联网、未下载），
> 覆盖真实权重结构/strict 加载/Studio 运行时加载/真实权重 bit-identical/关键数字复算。
> 未修改源码与 01 文档。

### 增量结论

**通过（R4 增量复核）。P0 = 0、P1 = 0；残留 P2 = 1、P3 = 3（均为文档表述级，不阻塞 FINAL）。**
R3 已逐条修复我 round2 评审的 **P0×1、P1×3、P2×4 与 §4 建议 1/2/4**（建议 3 半采纳）；其中我 round2 提的
“A18 首变帧公式 `ceil((s−128)/hop)`”经 R3 指出是**我的公式错误**，R3 的
`f_first = ⌊(s−win/2)/hop⌋ + 1` 经我三点实测全部命中——该条按 R3 更正采纳。

### 我的独立执行级验证（与 R3 §8.4 对照，全部一次通过）

| 项 | 我的实测 | 01 R3 的说法 | 判定 |
|---|---|---|---|
| 权重文件 | `falcon.pt` 47,156,343 B / sha256 `a1888b5a…928c`；`falcon-big.pt` 49,645,087 B / `ce2a92a3…037e` | §8.4 给出的两个 sha 完全相同 | ✅ 且 44.97 MiB ≈ **“约 45 MB”**（顺带回答 08 文档 A4 的“45 MB 无法验证”） |
| `falcon.pt` 结构 | 顶层 keys=`['architecture','epoch','state_dict']`；`architecture='FalconSpeakerContrastive'`、`epoch=714`；`state_dict` **662 = audio 525 + video 137**；`mdcl_window` **不在其中** | A46 同 | ✅ 精确 |
| numel ≠ params | audio state_dict numel **537,762**（+65）、video **11,196,308**（+9,620）、合计 11,734,070 vs params **11,724,385** | A50 同（+65 / +9,620） | ✅ 精确 |
| audio/video 权重 ↔ 模型 | key 集合全等、**0 shape 不一致**、`strict=True` 载入 OK；重建参数 **537,697 / 11,186,688 / 11,724,385** | A47–A49 同 | ✅ |
| `falcon-big.pt` 结构 | keys 同三段；`architecture='SpeakerContrastiveV2'`、`epoch=715`；**654 = audio 517 + video 137**；audio numel **1,158,746**；strict 载入 OK、参数 **1,158,617** | A56 同 | ✅ 精确 |
| **Studio 运行时加载真实权重** | `FalconRuntime("small", falcon.pt).load("cpu")` → ready、audio_keys **525**、video_keys **137**、parameters **11,724,385**、warmup 882 ms；big → audio_keys **517**、params **12,345,305**、infer 输出 `(32000,)` finite | 06 域接口；A46-A49/A56 | ✅ 说明 `weights_only=True` + 本地路径 + strict 双前缀对发布 ckpt **完全兼容** |
| **真实权重 bit-identical（small）** | 同一真实权重、同一预处理/输入，Studio 运行时 vs `look2hear` 训练侧模型 CPU 前向：**max\|Δ\| = 0.0（bit-identical=True）** | A49 / PROVENANCE:33 | ✅ **真实 falcon.pt 上成立**（此前 R1 只做了合成权重版） |
| **真实权重 big 自一致** | `FalconRuntime("big", falcon-big.pt)` vs 直接 `SpeakerContrastiveV2` + `FRCNNVideoModel` 前向：**max\|Δ\| = 0.0（bit-identical=True）**；audio 517 / video 137 key、params 1,158,617 | A56 | ✅ 训练仓库无 big 类，只能做 Studio 内部自一致（已做） |
| `concat[2]` 闲置 | hook：`{concat[0]:2, concat[1]:2, concat[3]:2, fusion[0..3]:2}`、**concat[2]=0**；`concat[2]` 参数 **5,568**、state_dict key **15**、闲置占比 **1.0355%**、有效参数 **532,129** | A51/§4.9-㉑/§4.7 同 | ✅ 精确 |
| A18 首变帧公式 | s=16000→**125**、20000→**156**、16128→**126**，与 `⌊(s−128)/128⌋+1` 三点全中 | A18/A53 | ✅（我先前的 `ceil` 写法确实错，R3 正确） |
| A21a-c 脉冲 | as-built **（0,250）**；去 cLNhw **（98,102）**；`causal=True`+去 norm **（100,104）** | A21a-c 同 | ✅ 精确 |
| A19 机理（GN→CumulateLN 抵消） | `GlobalLayerNorm(112)` 单层、后半段×10：左段 **4.103**；完整 `LayerNormalizationHW` 同协议：**1.907e-6**；P1 同分布替换：**9.5e-7** | §4.6-③/⑨：GN-only ×10 **4.40**、cLNhw **1.9e-6**、P1 **9.5e-7~1.4e-6** | ✅ 同量级/关键值 1.9e-6 精确命中 |
| A44 studio diff | `diff look2hear→studio`：added **65** / removed **56** / net(studio−look2hear) **+9** | §8.4-[W5]“新增 56/删除 65（净 −9）” | ✅ 数值等价（以 studio 为 base 的相反方向），表述见 P3-N1 |

### round2 评审条目 → R3 修复状态（逐条）

| 我 round2 的编号 | 意见 | R3 状态 | 我的复核 |
|---|---|---|---|
| P0-1 | `concat[2]` 从未调用、5,568 参数闲置、§3.2/§4.7/§4.9 未登记 | ✅ 已修 | §2.1.4“实例化 8/调用 7”、§3.2 明确 `concat[2]=0`、§4.7 新增“闲置 −5,568/有效 532,129”、§4.9-㉑、A10 例外、A51；hook 实测全对 |
| P1-1 | A21 “±2 帧”缺“去 cLNhw”口径 | ✅ 已修 | A21a（as-built 0,250）/A21b（去 norm 98,102）/A21c（causal 100,104）；我独立脉冲三点全中 |
| P1-2 | A44 漏 MPS CPU FFT bridge | ✅ 已修 | A44 三项并列 + 净 +9 行；§3.4/§5.3 引 06 §4.8；studio 行号 `:1111-1122`/`:1184-1198` 与我 06 的 `:1110-1122`/`:1183-1198` 仅差“是否含注释行” |
| P1-3 | A19/A24 缺扰动协议 | ✅ 已修 | §4.6-⑨ 五协议表 + A19/A24 条件列；裁定 1.2e-3/4.7e-5/1.4e-6 协议不同；我用 GN-only 4.103 vs cLNhw 1.907e-6 独立复现机理 |
| P2-1 | §5.3 架构校验只写 small | ✅ 已修 | 改为“等于该 variant 的 `architecture`（small/big）” |
| P2-2 | §4.7 concat 未区分实例化/调用 | ✅ 已修 | “实例化 ×4（22,272），调用 3 个（其中 5,568 闲置）” |
| P2-3 | A18 建议写首变帧公式 | 🟡 采纳并**更正** | R3 指出我的 `ceil` 公式错（ceil=124 与实测 125 不符），给出 `⌊(s−128)/128⌋+1`；我三点实测全中，接受更正（A18/A53） |
| P2-4 | 单层 1.4e-6 vs 全模型 5.1e-7 口径 | ✅ 已修 | §4.6-③ 标 P1/单层；A19/A24 条件列区分 |
| §4-建议 1 | §4.7 表尾加“闲置参数”行 | ✅ 已修 | “−5,568 / 有效 532,129（再扣对比头 517,545）”；我复算 537,697−5,568=532,129 ✓ |
| §4-建议 2 | 裸 `Falcon` 误用并入 §5.2 契约 | ✅ 已修 | §5.2 三态表 + §4.9-① 互链（A32） |
| §4-建议 3 | “若要流式：改动点→参数/数值→需重训”表 | 🟡 半采纳 | §4.6-⑦ 已写四类改动点及“前两项需重训/后两项保权重”；完整表建议放 FINAL（合理） |
| §4-建议 4 | 01→06 §5.5 引用 80 ms 右上下文 | ✅ 已修 | §4.6-⑤ 已改为“06 域已完成实现级验证，不再是纯 [推断]” |

### 残留问题（不阻塞；建议 R5/FINAL 前顺手修）

- **P2-N1｜§8.4-[W5] 的 diff 方向表述易被误读**：表中写“新增 56 / 删除 65（净 −9）”，未说明 base 是 studio；
  我的命令（`diff look2hear→studio`）结果是 added 65 / removed 56 / net **+9**。数值等价，但建议写成
  “`look2hear→studio`: +65/−56（净 +9 行；以 studio 为 base 则 +56/−65/净 −9）”并统一 A44 的“净 +9 行”。
- **P3-N1**：A44 行号 `:1111-1122`/`:1184-1198` 与 06 的 `:1110-1122`/`:1183-1198` 差的是**注释行**；
  §9.1 说“含注释行”但引用起点 1111（compute_device）不含注释 1110，建议统一为 `:1110-1122`/`:1183-1198`
  （或明确“功能行范围”）。
- **P3-N2**：§4.6-③ 写“P1 协议左段 1.4e-6”而 §4.6-⑨ 写“9.5e-7~1.4e-6”；我实测 9.54e-7。
  建议统一为“**1e-6 量级（seed/协议相关；实测 9.5e-7~1.4e-6）**”，避免 FINAL 引两个不同的“单值”。
- **P3-N3**：§4.9-㉑ 标“P0 级死参数”、R3 changelog 写“修 P0”，与 R2/R4 的“文档 P0=已修复”容易混淆；
  建议改“代码侧疑似 bug（严重度 P0，文档已登记；是否补调用待作者）”以免 FINAL 误读为“仍有未修 P0”。

### 跨模块更新（由真实权重验证带出的连带结论，供 FINAL/其它域使用）

1. **06 文档 Q1 可闭环**：真实 `falcon.pt` 已用 Studio `FalconRuntime`（本地路径、`weights_only=True`、strict 双前缀）
   成功加载并通过真实权重 **CPU bit-identical（max|Δ|=0）** 对照训练侧；`falcon.pt` 顶层
   `architecture/epoch=714`、`falcon-big.pt` `SpeakerContrastiveV2/epoch=715` 与 `PROVENANCE.md:36-39` 一致。
   ⇒ 06 的 Q1/A49/P10/P11 可从“未验证”升级为“已验证（本地权重，Lead 提供）”；
   big 亦完成 Studio 内部真实权重自一致（max|Δ|=0）。06 文档本轮未改动以避免 FINAL 期间 hash 漂移，待 Lead 决定是否补记。
2. **08 文档 A4（45 MB 无法验证）**：真实文件 47,156,343 B = 44.97 MiB，与 `launch.sh:10` 的“about 45 MB”一致 [验证]。
3. **06 文档 Q7（真实图片解码）**：不受影响；权重已验证不改变 Pillow 缺失结论。

### 增量复核验证活动记录

- 命令：`PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/r4_incremental_01_r3.py`
  → 日志 `docs/verification/runs/r4-incremental-01-r3.log`（含 [W0-W4]/[S]/[H]/[F]/[A21]/[D] 全部输出）。
- 未联网；未修改源码；01 文档 sha256 复核前后均为 `afe00188…d2db0058`（未变化）。
