# 验证记录 · 01 核心模型架构（model-core）

> 对应交付物：`docs/understanding/01-model-core.md`（§8 为本文件的摘要）
> 运行环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128，CPU）
> 实验使用 stub `timm.models.layers.DropPath`、`typeguard`，以随机张量验证模型结构与数值；第 5 节另记真实权重检查。
> **当前实现：**默认 VideoBlock 的 LSTM 沿批次轴递归；本页单样本探针的结果对应 B=1。配置重复构建时的字典污染已经修复。

## 1. 脚本与日志清单

| 脚本 | 当时日志 | 覆盖内容 |
|---|---|---|
| `docs/verification/scripts/model_core_verify.py` | `docs/verification/runs/model-core-verify.log` | [A] 构造链与配置键消耗 [B] 参数总账/逐层拆解/死类 [C] 形状链 [D] 逐组件因果性/脉冲响应 [E] MDCL 数值 [F] 死代码与抗误用 |
| `docs/verification/scripts/model_core_verify2.py` | `docs/verification/runs/model-core-verify2.log` | [G] 端到端前缀扰动；[H][I] 作废（见 §4） |
| `docs/verification/scripts/model_core_verify3.py` | `docs/verification/runs/model-core-verify3.log` | [L] 返回类型契约 [M] MDCL 非零损失/梯度/β/margin [N] 与 litmodule 契约 |
| `docs/verification/scripts/model_core_verify4.py` | `docs/verification/runs/model-core-verify4.log` | [O] 严格因果/lookahead 定量 [P] 视频分支因果（FRCNN + VideoBlock） |
| `docs/verification/scripts/model_core_verify5.py` | `docs/verification/runs/model-core-r3-verify.log`（R3 合并验证日志；verify5 的输出已并入） | [R] 20 项参数量手算复核 [S] causal 开关重做 [T] config 二次构建塌缩复现 |

使用当前服务器环境重跑时，可将输出写入新目录。`model_core_verify2.py` 和 `model_core_verify4.py` 的历史异常见第 4 节。

```bash
cd /root/code/github_repos/Falcon-fork
FALCON_RUN_DIR=$(mktemp -d /root/tmp/falcon-model-probe.XXXXXX)
FALCON_STUB_PATH=$PWD/docs/verification/stubs:$PWD
for script in model_core_verify model_core_verify2 model_core_verify3 model_core_verify4 model_core_verify5; do
  PYTHONPATH="$FALCON_STUB_PATH" /root/miniforge3/envs/falcon/bin/python "docs/verification/scripts/$script.py" \
    > "$FALCON_RUN_DIR/$script.log" 2>&1 && echo "$script OK" || echo "$script FAIL"
done
printf '结果目录：%s\n' "$FALCON_RUN_DIR"
```

## 2. 关键结果（数值均来自上述日志）

### 2.1 参数量（与 Lead 独立验证一致）

```
audio_model total = 537,697 (trainable=537,697)
  encoder              3,024
  audio_bottleneck    13,104
  video_bottleneck         0     <- ConvNormAct(kernel_size=-1) => Identity
  separator          492,312
    audio_net        445,766     <- 单个 AudioBlock，被复用 2 次
    video_net         42,850
    caf                3,696
  mask_generator      12,657
  decoder              2,016
  spk_pool_proj       14,584
FRCNNVideoModel    11,186,688
合计               11,724,385
```

逐项手算复核（`model_core_verify5.py` [R]）：20/20 项 `手算 == 实测`（OK），
明细见 `docs/understanding/01-model-core.md` §4.7。

### 2.2 形状链（L=32000，2 s@16 kHz；B=1）

```
audio [1,32000] + mouth [1,512,50]
 encoder         -> [1,112,251,129]     T'=1+⌊32000/128⌋=251, F=256/2+1=129
 audio_bottleneck-> [1,112,251,129]
 video_bottleneck-> [1,512,50]（0 参数）
 AudioBlock 逐级 -> (251,129) -> (126,65) -> (63,33) -> (32,17) -> 回到 (251,129)
 separator       -> [1,112,251,129]
 mask_generator  -> [1,1,112,251,129]
 decoder         -> [1,1,32000]
train 模式返回 (est, sep_emb) = ((1,1,32000), (1,1,112,251,129))
eval  模式返回 Tensor(1,1,32000)
```

### 2.3 因果性（核心结论的量化证据）

| 模块 / 层 | 前缀扰动 max\|Δ\| | lookahead | 结论 |
|---|---|---|---|
| `STFTEncoder` | 帧 123/124 = 0，帧 125 起非零 | **128 样本 = 8 ms** | `torch.stft(center=True)` 造成 |
| `STFTDecoder` | 本次探针前缀（<16000 样本）为 0 | 本次未检出 | 此输入和切分下的结果 |
| `audio_bottleneck.pre_norm`（cLNhw） | 1.4e-6 | **无界**（全局统计） | 非因果 |
| 同上去掉 cLNhw 的 1×1 conv | **0** | 0 | 泄漏源确认为归一化 |
| `AudioBlock`（cLNhw 全关） | — | **57 帧 = 7,296 样本 ≈ 456 ms** | 对称卷积 |
| `AudioBlock`（实际配置） | 2.98（特征域） | 无界 | 全局归一化 + 对称卷积 |
| `DualPathRNN(dim=3)` 时间路径 | **0** | 0 | 单向 LSTM，因果 |
| `DualPathRNN(dim=4)` 频率路径 | **0** | 0 | 时间维被当 batch |
| `MHSA2D(causal=True)` | **0** | 0 | 三角 mask |
| `MHSA2D(causal=False)` | 4.68 | — | 对照组 |
| `InjectionMultiSum(causal=False)` | 2.70 | ±2 帧 | 对称 k=5 |
| `InjectionMultiSum(causal=True)` | **0** | 0 | 对照组 |
| `MaskGenerator` / `NeuroSCNetFusion` | **0** | 0 | 1×1 / 加性 |
| FRCNN 视觉前端 | 改帧≥30 → 最左受影响帧 28 | **±2 视频帧 = ±80 ms** | Conv3d k=(5,7,7) pad=(2,3,3) |
| `VideoBlock` eval（B=1） | **0** | 本次未检出 | B=1 的输入探针；B>1 时默认实现存在跨样本递归 |
| `VideoBlock` train | 4.4e-2 | 窗口内无界 | BatchNorm1d 批内统计 |

端到端（改 1.0–2.0 s 音频，看前 1 s 输出）：`0–0.12s: 5.1e-7`、`0.38–0.5s: 1.1e-4`、`0.75–0.88s: 1.5e-2`、`0.88–1.0s: 4.2e-2`。
Lead 独立测得前 0.5 s 为 2.97e-4，同量级。

**顶层 `causal` 开关是惰性的**：`causal=True` 与 `False` 两个模型参数量相同（537,697）、`state_dict` key 相同、
装同一权重后输出 **bit-identical**（`torch.equal == True`）。真正生效的是 `enc_dec_params.causal` 与 `layer_3.causal`。

### 2.4 MDCL 数值

- `_sisnr(est=ref) = 122.06 dB`；已知 20 dB SNR 输入 → **20.000 dB**（正确）。
- `_mag_sim(est=ref) = 135.27 dB`；20 dB 波形噪声 → **16.33 dB**（幅度域更严格）。
- `_mdcl_loss_hard`：`K=0 → 0`；`K=1 == relu(mag_sim−margin).mean()`（allclose）；`K=3` 三种 β（0/1/20）与手算
  `(softmax(β·sim)·relu(sim−margin)).sum(0).mean()` **全部 allclose=True**；梯度可回传（|grad|≈101）。
- `margin=500 → 0`（relu 截断）；`margin=-500 → 加权平均相似度`。
- 模型默认 `mdcl_margin=0.0`、`mdcl_alpha=0.15`（config 未给）。

### 2.5 死代码 / 不一致（历史探针）

`_mdcl_loss` 别名与 `__all__` 已在 Phase 1 修复；配置入口的字典污染也已通过复制配置处理。以下保留修复前的探针输出。

| 项 | 命令/证据 | 结果 |
|---|---|---|
| `Falcon.forward(near_audio≠None)` | `model_core_verify.py` [F] | `AttributeError: 'Falcon' object has no attribute '_mdcl_loss'` |
| `FalconSpeakerContrastive` 吞 `near_audio` | `model_core_verify3.py` [L] | 未抛错、静默忽略 |
| config 直接喂裸 `Falcon` | 同上 | `TypeError: unexpected keyword argument 'spk_embed_dim'` |
| `from look2hear.models import *` | [F] | `AttributeError: ... has no attribute 'AVTIGER_causal'` |
| `models.get("AVTIGER_causal")` | [F] | `ValueError: Could not interpret model name` |
| `n_src=2` 前向 | [F] | `RuntimeError: shape '[2, 224, 251, 129]' is invalid for input of size 7252896` |
| 同一 config 二次构造 | `model_core_verify5.py` [T] | 537,697 → **49,081**，`audio_net`/`video_net` 为空 `ModuleList`，**前向不报错** |
| 死类 | [B] `named_modules` 检测 | `PositionalEncoding / MultiHeadSelfAttention / FeedForwardNetwork / GlobalAttention` 均不在图中 |
| `state_dict` | [F] | 525 keys，`mdcl_window`（`persistent=False`）不在其中 |

## 3. 与 Lead 验证的交叉核对

| 结论 | Lead（`runs/lead-model-params.log`） | 本记录 | 一致 |
|---|---|---|---|
| audio_model 参数 | 537,697 | 537,697 | ✓ |
| FRCNN | 11,186,688 | 11,186,688 | ✓ |
| 合计 | 11,724,385 | 11,724,385 | ✓ |
| 逐模块拆分 | 3,024 / 13,104 / 0 / 492,312 / 12,657 / 2,016 / 14,584 | 同 | ✓ |
| `video_bottleneck` Identity/0 参数 | ✓ | ✓ | ✓ |
| L=32000 → T'=251、F=129 | ✓ | ✓ | ✓ |
| 因果性泄漏（cLNhw 全局统计） | 2.97e-4（前 0.5 s） | 1.13e-4（0–0.5 s；0–0.12 s 仅 5.1e-7） | 同量级 ✓ |
| `_mdcl_loss` 缺失 | ✓ | ✓（并补充：子类被 `**kwargs` 静默吞掉） | ✓ |

## 4. 失败与作废的探针

1. **`model_core_verify2.py` [I] 崩溃（历史版本，现已修复）**：`FRCNNVideoModel(pretrain=None).eval()` 返回 `None`。
   根因不是脚本错误，而是 `look2hear/videomodels/frcnn_videomodel.py:90-96` 重写了 `train(self, mode=True)` 却
   **没有 `return self`**，而 `nn.Module.eval()` 的实现是 `return self.train(False)`（torch 2.8.0）。
   ⇒ `FRCNNVideoModel(...).eval()` 变成 `None`，任何链式调用都会炸。改成分语句调用后验证通过（见 verify4-[P]）。
   **建议同步给 04 域（loss-metric-eval）与 06 域（studio-backend）**。
2. **`model_core_verify2.py` [H] 作废**：`lookahead()` 把"输出索引"与"输入分界（样本）"放在不同单位空间相减，
   得到无意义负数；已在 `verify4-[O]` 用统一单位重做。
3. **`model_core_verify4.py` [Q] 作废（历史配置污染；现行入口已复制配置）**：两次构造共享同一份嵌套 config dict，第一次 `Separator.__init__` 的 `dict.pop`
   污染第二次（`audio_params`/`video_params` 丢 `repeats`/`shared`），得到 557,281 vs 68,665 的假差异
   （真实差异其实来自 `spk_embed_dim` 被剔除后回落到类默认 128）。已在 `verify5-[S]` 用"每次全新 deepcopy"重做，
   结论：causal 开关 bit-identical。
4. **Lead 脚本 `verify_model_params.py` 的 TypeError**：`Falcon.__init__() got an unexpected keyword argument
   'spk_embed_dim'` 是"配置与裸 `Falcon` 不兼容"的真实证据（见理解文档 §4.9-④），脚本随后改走 `FalconSpeakerContrastive`。
5. **未做（受环境限制）**：MACs / 延迟复算（属 04 域，`evaluated_mac_params.py` 需 GPU + ptflops 自定义输入构造）；
   真实权重端到端（无 `falcon.pt`、无 `pretrain_zoo/`）；训练/精度（无数据集）。这些在理解文档 §7 的 Q1/Q9 中标注。

---

## 5. R3 版本锚点与新增验证（2026-10-21）

### 5.1 版本锚点

| 项 | 值 |
|---|---|
| 被修订文档 | `docs/understanding/01-model-core.md` |
| R1 版 sha256（R2/R4 评审所据） | `cd58a08cd7775b8ec8fb91ce13b5f9e053977afb981f0b3bd2a6c8f4cd11b30b`（807 行） |
| **R3 版 sha256** | **`afe00188097e4d7d5ada48dbcdd96703248ea7942233f3b699658e48d2db0058`**（**975 行**，2026-10-21；复核方可用该 hash 校验增量复核对象） |
| R3 新增脚本 | `docs/verification/scripts/model_core_r3_verify.py` |
| R3 新增日志 | `docs/verification/runs/model-core-r3-verify.log` |
| 真实权重 | `/root/.cache/falcon_weights/falcon.pt`（sha256 `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`）、`falcon-big.pt`（`ce2a92a3188e8069e979f23aaec21cb6d439170aaf4002b4eaa234f0486e037e`） |
| 复跑命令 | `PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/model_core_r3_verify.py > "$FALCON_RUN_DIR/model-core-r3-verify.log" 2>&1`（`FALCON_RUN_DIR` 见第 1 节） |

### 5.2 真实权重验证结论（[验证]）

1. `falcon.pt` 顶层 `dict`，keys = `state_dict / architecture / epoch`；`architecture='FalconSpeakerContrastive'`、`epoch=714`。
2. `state_dict` 共 **662** 条 = `audio_model.*` **525** 条 + `video_model.*` **137** 条；`mdcl_window` **不在其中**。
3. audio 权重与 `FalconSpeakerContrastive(configs/Falcon.yml)` **key 集合相等（525/525）、0 shape 不一致、`strict=True` 载入成功**；
   载入后参数 **537,697**。video 权重与 `FRCNNVideoModel` **137/137、strict=True OK**，参数 **11,186,688**；合计 **11,724,385**。
4. **口径陷阱**：`state_dict` 张量 numel ≠ 参数量——audio 537,762（比参数多 65 = `BatchNorm1d(32)` running stats 等 buffer）、
   video 11,196,308（多 9,620 = ResNet 的 BN buffer）⇒ 参数量必须用 `parameters()` 求和。
5. `falcon-big.pt`：`architecture='SpeakerContrastiveV2'`、`epoch=715`；audio **517** 条（state_dict numel 1,158,746）/ video 137 条
   （big 属 06 域，仅结构核对）。
6. 附：`import look2hear.models` 实测连带加载 `librosa / torch_complex / typeguard / pytorch_lightning / rich`
   （PL 来自 `look2hear/utils/lightning_utils.py:3-4`，不是 `BaseModel.serialize`）。

### 5.3 concat[2] 闲置 / Studio diff / 分界敏感性（[验证]）

- `concat_layers` 调用计数（一次 model forward，AudioBlock×2）= `{0:2, 1:2, 2:0, 3:2}`；`fusion` = `{0:2,1:2,2:2,3:2}`；
  `concat_layers[2]` = **5,568 参数**（537,697 的 **1.04%**）全程闲置，有效参数 **532,129**；其 15 个 state_dict key 仍会保存/加载。
- Studio 副本 vs 训练侧 `Falcon.py`：diff 新增 56 / 删除 65（净 **-9**，即 studio 多 9 行）；语义差异 = DropPath 来源 +
  MPS CPU FFT bridge（studio `:1111-1122`、`:1184-1198`，各 +4 行代码）。
- AudioBlock（关 cLNhw）lookahead 的分界敏感性：split=100/125/150 → **56/57/58 帧**（±1 帧量化）。
- STFT encoder 首变帧公式：`f_first = ⌊(s − win/2)/hop⌋ + 1`（s=16000→125、20000→156、16128→126；≠ `ceil((s−128)/hop)`）。

### 5.4 新增/修正的因果性口径（R3 裁定）

- 01 的 `1.4e-6`（T=251 同分布替换）、02 的 `4.7e-5`（波形×10 → 我复算 3.2e-5）、02 的 `1.2e-3`（T=8 单帧尖峰）
  **三者都正确、只是探针协议不同**；统一条件表见 01 文档 §4.6-⑨。
- 新机理：`cLNhw` 第一段 GN 单独 ×10 探针前缀 Δ=**4.40**，经第二段 `CumulateLN4D` 后仅 **1.9e-6**（纯尺度泄漏被抵消）；
  AudioBlock ×10 探针 Δ=7.14；全窗统计与对称卷积的相对贡献仍需受控实验区分。
