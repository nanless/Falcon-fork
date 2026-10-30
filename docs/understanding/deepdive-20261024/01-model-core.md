# 深挖 01 · 模型核心（Falcon 架构 / 参数 / MACs / 因果性 / 真实权重）

> 小组：model-core（“项目理解 + 模型复现规划”深挖小组）· 日期 2026-10-24
> 仓库：`/root/code/github_repos/Falcon-fork` · 分支 `repro/phase2` · HEAD `2beec38`
> 环境：**仅** `/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / timm 1.0.30 / typeguard 2.13.3 / numpy 2.2.6；2×A800 空闲）
> 证据标记：`[代码]` 读源码；`[验证]` 本轮本机复算；`[推断]` 合理推断；`[文档]` 引用已有报告/README
> 行号均为 HEAD `2beec38` 的实际行号。本轮**只读**：未改源码、未改他人文档、未做 git 写操作、未联网、未启动训练。
> 上一版：`docs/understanding/01-model-core.md`（R3，基线 commit `7576854`，975 行）。本文与它的逐条对照见 §4。

---

## 0. 一句话结论

1. **模型定义就是“官方上游快照”，没有被人改过架构**：Phase 1 只加了 2 处非数值改动（`_mdcl_loss` 别名、`__all__` 修正），Phase 2 对 `look2hear/models/**` **零改动**（§4.1 git 证据）。
2. **主线复现（论文结果 / 8 卡配方）用 `look2hear.models.FalconSpeakerContrastive` + `configs/Falcon.yml`**：音频 **537,697** 参数（基础 `Falcon` 523,113 + 训练期对比头 14,584；其中 `concat_layers[2]` 的 **5,568** 参数从不参与前向 ⇒ **实际生效 532,129**），冻结 FRCNN **11,186,688**，合计 **11,724,385**；MACs **10.611 G**（不含视觉编码器）。与 README 的 0.5 M / 11.72 M / 10.6 G 口径一致。
3. **官方权重与代码严格对应**：`falcon.pt` 的 audio 525 key / video 137 key 可 `strict=True` 零 missing 零 unexpected 载入，`architecture='FalconSpeakerContrastive'`、`epoch=714`；`falcon-big.pt` 对应 `SpeakerContrastiveV2`（audio 517 / video 137、`epoch=715`）。训练侧与 Studio 侧**同权重同输入前向 max|Δ|=0**（bit-identical）。
4. **“causal” 只能写成“窗口级因果（window-causal）”**：卷积/RNN/attention 结构单向、跨窗口无状态，但 `torch.stft(center=True)` 有 ±128 采样分析窗、`cLNhw` 全窗统计、AudioBlock 的对称 k=5 卷积不因果。关掉 `cLNhw` 后单遍 `AudioBlock` lookahead = **57 帧 = 7,296 采样 ≈ 456 ms**；as-built 端到端单点扰动首变样本 **9,612**（相对扰动点 **−10,388 采样 ≈ −649 ms**）。它是“窗口级因果”，**不是逐采样流式模型**。
5. 旧文档 56 个编号（A01–A56，A21 拆 a/b/c 共 58 行）本轮逐条复核：**53 行确认、1 行修正（A44 Studio diff 行数）、3 行推翻/已修（A32/A36/A43，Phase 1 修好）、1 行未复跑（A41，训练系统域）**；另有旧文档/FINAL 的**过期描述**需要更新（§4.3）。
6. 主要剩余风险**不在模型数值**，而在：`concat_layers[2]` 5,568 参数闲置（疑似 level-2 融合漏调用，Phase 1 为保 bit-identical 有意不修）、`Separator` 原地 `pop` 配置、`FalconSpeakerContrastive` 静默吞 `near_audio`、`n_src>1` 不支持；这些都不阻碍当前复现路线（§3/§5）。

---

## 1. 系统边界与关键路径（file:line）

### 1.1 范围内文件

| 文件 | 行数 | 职责 | 本轮关键点 |
|---|---|---|---|
| `look2hear/models/Falcon.py` | 1492 | 架构定义：17 个 `nn.Module` 类（13 在用 + 4 死代码）+ `Falcon` 主类 + MDCL 三个工具方法 | 全部可训练参数都在这里（§2.1） |
| `look2hear/models/Falcon_Contrastive.py` | 122 | `FalconSpeakerContrastive(Falcon)` + `AttentionSpeakerPool`（同空间说话人对比头） | 训练/发布权重用的就是它 |
| `look2hear/models/base_model.py` | 102 | `BaseModel`：`sample_rate`、`from_pretrain`、`load_state_dict_in_audio`、`serialize` | 无可训练参数；`get_model_args` 只回 `n_src`，不能重建模型 |
| `look2hear/models/__init__.py` | 44 | 导出 `Falcon` / `FalconSpeakerContrastive` / `get` / `register_model` | Phase 1 已修正 `__all__`（旧版指向不存在的 `AVTIGER_causal`） |
| `look2hear/utils/get_layer_from_string.py` | 42 | 按名取 `torch.nn` 层（供 layers 域） | 与 Falcon 前向无直接关系 |
| `look2hear/utils/separator.py` | 69 | `separate()/torch_separate()/numpy_separate()` 推理胶水 | 其中 `out_wavs *= wav.abs().sum()/out_wavs.abs().sum()` 会改幅度，**评测不要走它** |
| `evaluated_mac_params.py` | 151 | README 参数/MACs 的出处脚本（ptflops） | 只统计 audio_model，视觉编码器 64×64，不含在 10.6 G 内（§2.2） |

### 1.2 三条“产品线”入口（同一模型定义）

| 用途 | 入口 | 关键行 | 模型类 |
|---|---|---|---|
| 训练 | `python audio_train.py --conf_dir configs/Falcon.yml` | `audio_train.py:261` `getattr(look2hear.models, config["audionet"]["audionet_name"])(sample_rate=..., **audionet_config)` | `FalconSpeakerContrastive` |
| 评测 | `python audio_visual_test.py` | `audio_visual_test.py:287-295` 按 `audionet_name` 重建；`:301/:319` `strict=True` 载入 audio/video | 同上 |
| Studio 推理（small） | `studio/falcon_runtime.py` | `:19` `from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive`；`:33,:37` `MODEL_CLASSES`；`:157` 构造；`:168-169` `strict=True` | Studio 副本（与训练侧类文件逐字节相同） |
| Studio 推理（big） | 同上 | `:20` `SpeakerContrastiveV2`；`:35` variant 映射 | `SpeakerContrastiveV2(SISNRRepel)`，audio 1,158,617 参数 |

### 1.3 模块关系（`[代码]`）

```
Falcon (Falcon.py:1277)
├── encoder: STFTEncoder(1054)              torch.stft(win=256,hop=128,center=True) → 3ch(real,imag,mag) → Conv2d(3→112,k3,causal)
├── audio_bottleneck: ConvNormAct(112→112,k1,pre_norm=cLNhw)   (1311)
├── video_bottleneck: ConvNormAct(kernel=-1 ⇒ Identity, 0 参数) (1312)
├── separator: Separator(976)
│    ├── audio_net: AudioBlock(691) ×1 实例（shared=true，前向调用 2 次）
│    │     pyramid: 4 级 downsample（stride 2 同时降 T、F）
│    │     每级: DualPathRNN(dim=4, 频向双向 LSTM) + DualPathRNN(dim=3, 时向单向 LSTM) + MHSA2D(causal=True)  ← TFRB
│    │     fusion_layers[0..3] = InjectionMultiSum（把最深层全局特征注入各层 skip）                    ← GCI
│    │     concat_layers[0..3] = InjectionMultiSum（解码逐级融合；**index 2 从不调用**）
│    ├── video_net: VideoBlock(898) ×1（512→32 单向 LSTM→512）
│    └── caf: NeuroSCNetFusion(239) ×1（视频 512→112 groups=16，时间 50→251 nearest，加性注入）
├── mask_generator: MaskGenerator(1194)     PReLU + Conv2d(112→112,k1,ReLU)；RI_split ⇒ 112=2×56 复数掩码
└── decoder: STFTDecoder(1123)              ConvTranspose2d(112→2,k3,causal pad) → 复谱 → istft(length=L)
```

- 顶层 `causal` 开关**惰性**（`[验证]` §2.8）：只写入 `audio_params['causal']`/`video_params['causal']`，`AudioBlock` 只把它传给 `DualPathRNN`，而 `DualPathRNN` 从不读 `self.causal`（`Falcon.py:365`）；真正生效的是 `enc_dec_params.causal=True` 与 `layers.layer_3.causal=True`。
- Studio 副本关系（`[代码]`+`[验证]` §2.6）：`studio/falcon_model/models/Falcon_Contrastive.py` 与 `look2hear/models/Falcon_Contrastive.py` **逐字节相同**（`diff` 退出码 0）；`studio/.../Falcon.py` 与 `look2hear/.../Falcon.py` 只差注释语言、`DropPath` 来源（`../drop_path` vs `timm.models.layers`）、studio 独有的 MPS CPU-FFT bridge，以及 look2hear 独有的 Phase 1 `_mdcl_loss` 别名——**CPU/CUDA 前向数值等价**（`[验证]` max|Δ|=0）。`SISNRRepel.py` 是 studio `Falcon.py` 的“改名副本”（仅类名 + 2 行注释 import 差异），`SpeakerContrastiveV2` 在其上加 128 维 spk 头。

---

## 2. 关键事实与数字（命令 + 实际输出）

通用环境前缀（脚本内部还会自行插入 `docs/verification/stubs`，这是旧沙箱遗留；真实 conda 环境**不依赖 stub** 也能构造模型，见 §2.8 的清环境验证）：

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon
PYTHONPATH=$PWD/docs/verification/stubs:$PWD $ENV/bin/python <script>
```

### 2.1 参数量：537,697 / 532,129 / 523,113 / 11,186,688 / 11,724,385（[验证]）

命令与输出（`docs/verification/scripts/verify_model_params.py`，本轮用 conda 环境复跑，exit 0）：

```text
-- 音频网络各子模块参数量 --
encoder              total=     3,024
audio_bottleneck     total=    13,104
video_bottleneck     total=         0
separator            total=   492,312
mask_generator       total=    12,657
decoder              total=     2,016
spk_pool_proj        total=    14,584
AUDIO_MODEL total=537,697 trainable=537,697
FRCNN total=11,186,688 trainable=11,186,688 (未加载 pretrain 时 requires_grad=True)
AUDIO+FRCNN total=11,724,385  trainable(audio only)=537,697
```

口径拆解（`[验证]`，本机复算）：

```text
FalconSpeakerContrastive total = 537,697
  - spk_pool_proj (训练期对比头) = 14,584  -> 基础 Falcon = 523,113
  - concat_layers[2] (前向闲置) = 5,568  -> 生效 = 532,129
  - 两者都扣 = 517,545
  Falcon 基类直接构建（去掉 spk_embed_dim）: 523,113  (= 537,697-14,584=523,113)
```

- **537,697 = 0.538 M**：README `README.md:44` “0.5 M trainable”；`README.md:123` 脚注 “paper counts trainable parameters only (0.5 M for Falcon)”。0.538 M 取 1 位有效数字即 0.5 M（陈述性取整，`[推断]`，无更多论文细节可证）。
- **532,129 = 537,697 − 5,568**：`concat_layers[2]` 从未参与前向（§2.3）。
- **523,113 = 537,697 − 14,584**：去掉只训练期使用的 `spk_pool_proj`；它与“推理图”口径最接近（532,129 再扣对比头 = 517,545，两种口径都记录在案，引用时必须写明）。
- **11,724,385 = 537,697 + 11,186,688**：`README.md:120` 表格 11.72 M、`README.md:123` 注明含冻结 FRCNN；“冻结的 11.2 M”只指 FRCNN，`video_bottleneck`/`VideoBlock` 属于音频模型、可训练（`[代码]` `frcnn_videomodel.py:117` 里 `requires_grad=False`）。

### 2.2 MACs：10.611 G（[验证]）

`evaluated_mac_params.py:80-123` 的实际做法：`a=randn(1,32000).cuda()`、`v=randn(1,1,50,64,64).cuda()` → `video_model(v)` 得 `(1,512,50)` → ptflops 只对 **audio_model** 计 MACs，打印 `MACs/1e9`。CPU 复刻（`docs/verification/scripts/verify_macs_04.py`）输出：

```text
FalconSpeakerContrastive (audio) = 537,697 = 0.538 M
FRCNNVideoModel                  = 11,186,688 = 11.187 M
audio + video                    = 11,724,385 = 11.72 M  -> README '11.72 M'
ptflops(audio_model, input=(a,vemb)) -> MACs = 10.611 G, params = 0.538 M
FRCNNVideoModel (1, 50, 64, 64): MACs=7.77 G,  params=11.19 M
FRCNNVideoModel (1, 50, 88, 88): MACs=15.90 G, params=11.19 M
FRCNNVideoModel (1, 50, 96, 96): MACs=17.48 G, params=11.19 M
```

- **10.611 G 与 README `README.md:226` 的 10.6 G 一致**（1 位小数），统计口径 = 2 s 音频 + 50 帧**已编码**视觉特征的一次前向，**不含视觉编码器**；且脚本用的是 64×64 而非训练/评测实际 88×88（`[代码]` `evaluated_mac_params.py:81`）。
- **未复算**：README `README.md:226` 的 CPU 1361.2 ms / GPU 51.3 ms / RAM 125.3 MB / 显存 364.7 MB（本轮禁跑测速，标“未验证”）。

### 2.3 结构映射：TFRB / GCI ↔ 代码，金字塔形状，`concat[2]` 闲置（[代码]+[验证]）

README 图注（`README.md:67` “a TFRB at the coarsest scale, GCI fusion at every scale and a decoder of GCI and TFRB blocks”；`README.md:72` “bidirectional LSTM along frequency, a unidirectional LSTM along time, then masked 2D self-attention”）↔ 代码（`[推断]` 命名映射，但语义可互证）：

| README 术语 | 代码实现 | 证据 |
|---|---|---|
| **TFRB**（Time-Frequency Refinement Block） | `DualPathRNN(dim=4)`（频向双向 LSTM，时间维是 batch ⇒ 时间不上看未来）+ `DualPathRNN(dim=3)`（时向**单向** LSTM）+ `MultiHeadSelfAttention2D(causal=True)`（`torch.tril` mask）三件套，每级各 1 份，共 4 级 | `Falcon.py:331-415`、`:417-543`、`:777-790`；mask 在 `:520-524` |
| **GCI**（跨尺度门控注入） | `InjectionMultiSum`：`local_emb * Sigmoid(global_gate(global)) + global_emb`（3×depthwise k=5 ConvNormAct） | `Falcon.py:267-329`、`:858`、`:868`、`:882` |
| 编码侧只在最粗尺度跑 TFRB | `if i == self.upsampling_depth - 1:` 才调 `process_f/t/att` | `Falcon.py:842-845` |
| 解码侧逐级 2→1→0 跑 TFRB + 每级 GCI | 初始 `concat[-1]`；循环 `for i in range(upsampling_depth-3, -1, -1)` = `[1,0]` | `Falcon.py:868-888` |

形状推演（`[验证]` 逐层 forward hook，B=1, L=32000）：

```text
AudioBlock 输入: (1,112,251,129) 输出: (1,112,251,129)
downsample_layers 输出形状(T,F): [(0, 251, 129), (1, 126, 65), (2, 63, 33), (3, 32, 17)]
process_att_layers 记录 n_freqs: [129, 65, 33, 17]
process_att_layers 参数: [51,693, 31,213, 20,973, 15,853]
fusion_layers[0] params = 5568
concat_layers[2] params = 5568
audio_net params = 445766 | video_net params = 42850 | caf params = 3696 | separator params = 492312
```

`concat_layers[2]` 从未被调用（`[验证]` hook，一次 model forward，AudioBlock 调 2 次）：

```text
concat_layers 调用次数: {0: 2, 1: 2, 2: 0, 3: 2}
fusion_layers 调用次数: {0: 2, 1: 2, 2: 2, 3: 2}
concat_layers[2] 参数 = 5,568 | 占 audio_model 537,697 的 1.04%
有效参与前向的参数 = 532,129
concat_layers[2] 的 state_dict key 数: 15
```

- 机制：`Falcon.py:868` 用 `self.concat_layers[-1]`（index **3**）融合 N-2/N-1 两级；`:879` 的循环只遍历 `1,0` ⇒ index 2 被跳过。源码注释 `:864-867` 写的是 “Fuse the two deepest levels”，与“跳过 level-2 融合”不一致，**疑似漏调用 bug**（Phase 1 决策：为保与官方权重 bit-identical 不修，见 `docs/REPRODUCTION_PHASE1_REPORT.md` F13）。
- 后果：官方 `falcon.pt` 里这 15 个 key **有训练值、可 strict 加载、但前向从不使用**——做“参数量/有效容量”讨论时必须注明。

### 2.4 前向数据流与 mask 形式（[代码]+[验证]）

`FalconSpeakerContrastive.forward`（`Falcon_Contrastive.py:102-118`；`Falcon.forward` `Falcon.py:1469-1488`）：

```text
音频 [B,L] 16kHz
 → encoder:  torch.stft(win=256,hop=128,center=True,reflect) → spec[129,T]; 堆 3 通道(real,imag,sqrt(|s|²+eps))
             → Conv2d(3→112,k=3,causal=True,时间左 pad2/裁右2) → [B,112,T,129]，T=1+⌊L/128⌋，F=129
 → audio_bottleneck: cLNhw+Conv2d(112→112,k1) → [B,112,T,129]
 视频 [B,512,Tv]（FRCNN 输出，2 s ↔ Tv=50）
 → video_bottleneck: Identity（0 参数）
 → separator.audio_net → caf(video→112, 50→251 nearest, 广播加) → audio_net(第2遍, +audio_residual) → [B,112,T,129]
 → mask_generator: Sequential(PReLU, ConvNormAct(112→112,k1,act=ReLU))
      RI_split: 112 = 2×56；mask_real=mask[:,:,0], mask_imag=mask[:,:,1]；encoder 特征同样拆 56+56
      est_real = e_r*m_r − e_i*m_i；est_imag = e_r*m_i + e_i*m_r；再 cat 回 112
      → separated_audio_embeddings [B,1,112,T,129]
 → decoder: x.view(B*n_src,112,T,129) → causal 时间 pad2 → ConvTranspose2d(112→2,k3) → 裁尾1帧
      → (real,imag) 复谱 → istft(length=L) → [B,1,L]
```

mask 的**精确形式**（易误读，`[代码]` `Falcon.py:1221-1270`）：

- mask 是**学到的特征通道分组**上的“复数”掩码，不是 STFT 真/虚部的掩码；乘法是对 112 通道特征做双线性交互（`RI_split=True`）。
- mask 激活是 **ReLU（无上界、非 [0,1] 门）**；`output_gate/dw_gate/direct` 均为 False（本配置未启用额外 gate）。
- 输出与输入等长 `[B,1,L]`（`istft(length=L)` 保证）；`n_src=1` 是硬约束（见 §2.8）。

### 2.5 真实权重：strict 加载、key 数、epoch、architecture（[验证]）

权重（源 `/root/.cache/falcon_weights/`，指向 group-shared 缓存；sha256 与旧文档一致）：

```text
a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c  falcon.pt      (47,156,343 B)
ce2a92a3188e8069e979f23aaec21cb6d439170aaf4002b4eaa234f0486e037e  falcon-big.pt  (49,645,087 B)
```

`docs/verification/scripts/verify_real_weights.py`（本轮 conda 环境复跑，exit 0）关键输出：

```text
A. falcon.pt: dict keys=['state_dict','architecture','epoch']; entries 662; prefix {'audio_model':525,'video_model':137}
   architecture='FalconSpeakerContrastive'  epoch=714   optimizer_states=False
B. training audio_model strict load: OK (missing/unexpected empty); params=537,697
   training video_model strict load: OK; params=11,186,688
C. studio audio_model strict load: OK, params=537,697
   eval-mode forward: train (1,1,32000) vs studio (1,1,32000), max|Δ| = 0.000e+00  bit-identical: True
   train-mode est max|Δ| = 0.000e+00 | sep_emb max|Δ| = 0.000e+00
D. FalconRuntime status: ready=True variant=small architecture=FalconSpeakerContrastive checkpoint=falcon.pt
   audio_weight_keys=525 video_weight_keys=137 parameters=11724385 warmup_ms=1745.7 mode=windowed_audio_visual
   infer output (32000,) float32 finite=True | runtime vs manual max|Δ| = 0.0
E. falcon-big.pt: entries 654; {'audio_model':517,'video_model':137}; architecture='SpeakerContrastiveV2' epoch=715
   big strict load OK: audio=1,158,617 video=11,186,688 total=12,345,305 (README 12.35M)
```

结论：

- **falcon.pt ↔ `FalconSpeakerContrastive(configs/Falcon.yml)` 完全对应**：audio 525/525 key 集合相等、0 shape 不一致、`strict=True` 成功；`mdcl_window` 是 `persistent=False` 的非持久 buffer，**不在** state_dict（state_dict 恰 525 key）。
- video 137/137 ↔ `FRCNNVideoModel`，`strict=True` OK。
- **big 权重 ↔ `SpeakerContrastiveV2(studio/configs/FalconBig.yml)`** 完全对应（audio 517 key、1,158,617 参数），video 同样 137 key。
- `state_dict` 张量 numel ≠ 参数量：audio 537,762 vs 537,697（多 65 = BatchNorm running stats 等 buffer）；video 11,196,308 vs 11,186,688；big audio 1,158,746 vs 1,158,617。**参数口径必须用 `parameters()`**。

### 2.6 训练侧 vs Studio 侧：同一输入 max|Δ|=0（[验证]）

```text
studio_falcon_model/models/Falcon_Contrastive.py  ==  look2hear/models/Falcon_Contrastive.py   (diff 退出码 0，逐字节相同)
eval-mode forward:  train (1,1,32000) vs studio (1,1,32000), max|Δ| = 0.000e+00
train-mode est max|Δ| = 0.000e+00 | sep_emb max|Δ| = 0.000e+00
FalconRuntime: audio_weight_keys=525 video_weight_keys=137 parameters=11724385
infer(32000 采样, 50×96×96) → (32000,) float32；runtime vs 手工前向 max|Δ| = 0.0
```

Studio 小模型副本与训练侧差异（`[验证]` 结构化 diff）：

```text
diff studio/falcon_model/models/Falcon.py look2hear/models/Falcon.py
  diff 总行数 199 | 新增 61 | 删除 65 | 净 -4（studio 1496 行 vs look2hear 1492 行）
  去注释/空白后：新增 9 行、删除 15 行
  语义差异只有：DropPath 来源（studio ../drop_path.py vs timm）；studio 独有 MPS CPU-FFT bridge（encoder/decoder 各一处）；
               注释语言（studio 中文 / look2hear 英文）；look2hear 独有 Phase 1 的 `_mdcl_loss` 别名
```

### 2.7 因果性 / 流式：精确表述与数字（[验证]）

**统一措辞（与旧文档 §4.6-⑦、02/06 一致）**：

> **窗口级因果（window-causal）**：2 s 窗口内允许未来依赖（STFT 分析窗 ±128 采样/8 ms；关掉 `cLNhw` 后对称卷积单遍 lookahead 57 帧≈456 ms；`cLNhw` 全窗统计带来无界依赖），但模型**不携带跨窗口状态**，因此**不是逐采样因果的流式模型**。

证据 1：STFT 分析窗 ±128 采样（`Falcon.py:1110-1116` 未传 `center`，默认 `center=True, pad_mode='reflect'`；`[验证]` 脚本 `verify_causality.py`）：

```text
单采样点扰动: a2[20000] += 1.0  (win=256, hop=128)
[全模型 est] first_changed_sample=9612 last_changed_sample=31999 max|d|=7.757e-02
  改动点 s=20000: 若 f >= s 则严格逐采样因果；实际 f-s = -10388 samples (-649.25 ms)   ← 摘录，省略“若 f>=s…”判据句
encoder          : frames [156, 159]  (samples [19968, 20352])  max|d|=3.103e-01
audio_bottleneck : frames [156, 250]  (samples [19968, 32000])  max|d|=8.424e-02
separator        : frames [0, 250]    (samples [0, 32000])      max|d|=1.263e-01
mask_generator   : frames [0, 250]    (samples [0, 32000])      max|d|=3.541e-01
```

- `encoder` 首变帧 156 = ⌊(20000−128)/128⌋+1，说明**编码器本身**只有 ±128 采样（8 ms）分析窗，且时间卷积是因果的。
- 泄漏发生在 `audio_bottleneck` 之后（cLNhw 全窗统计）与 `separator`（对称卷积 + 归一化串联）：端到端首变样本被拉回到 **9,612**，相对扰动点 **−10,388 采样 ≈ −649.25 ms**。

证据 2：首变帧公式三点命中（`[验证]`，`model_core_r3_verify.py [W8]`）：

```text
s=16000: 首变帧=125 末变帧=250 | 公式 f_first=125 (一致)
s=20000: 首变帧=156 末变帧=250 | 公式 f_first=156 (一致)
s=16128: 首变帧=126 末变帧=250 | 公式 f_first=126 (一致)
# 公式：f_first = floor((s - win/2)/hop) + 1 = floor((s-128)/128)+1
```

证据 3：关掉 cLNhw 后的纯结构 lookahead（`[验证]`，`model_core_verify4.py [O]` / `model_core_r3_verify.py [W7]`）：

```text
AudioBlock(norm 全关)       : 分界 125 帧, 前缀最左受影响帧 = 68, lookahead = 57 帧 = 7296 样本
AudioBlock(实际 cLNhw)      : 分界 125 帧, 前缀最左受影响帧 = 0,  lookahead = 125 帧 = 16000 样本（无界/全窗）
audio_bottleneck(cLNhw)     : 分界 125 帧, 前缀最左受影响帧 = 0,  lookahead = 125 帧 = 16000 样本
split=100 -> 56 帧 = 7168 样本 | split=125 -> 57 帧 = 7296 样本 | split=150 -> 58 帧 = 7424 样本
7,296 采样 / 16 kHz = 456 ms
```

证据 4：逐模块因果性（`[验证]` 本轮）：

```text
STFTDecoder: 前缀(样本<16000) max|Δ| = 0.000e+00 -> 严格因果；最左受影响输出样本 = 16001
DualPathRNN(dim=4, 频向双向) 前缀帧<125 max|d| = 0.0     （时间维是 batch，不上看未来）
DualPathRNN(dim=3, 时向单向) 前缀帧<125 max|d| = 0.0
MHSA2D(causal=True)  前缀帧<125 max|d| = 0.0
MHSA2D(causal=False) 前缀帧<125 max|d| = 4.114          （对照，证明确实由 tril mask 保证因果）
InjectionMultiSum(norm=cLNhw) 脉冲@100 影响帧区间 = (0, 250)     （as-built：全局）
InjectionMultiSum(norm=None)  脉冲@100 影响帧区间 = (98, 102)    （±2 帧）
InjectionMultiSum(norm=None,causal=True) 脉冲@100 = (100, 104)   （严格因果）
audio_bottleneck.pre_norm P1 同分布替换，左段(帧<120) max|d| = 1.430511474609375e-06
模型内 LayerNormalizationHW/GlobalLayerNorm/CumulateLN4D 各 30 个（named_modules 计数）
```

证据 5：端到端改后半段音频，前 1 s 输出的泄漏量级（`[验证]`）：

```text
[    0: 2000] (0.00-0.12s) max|Δ| = 5.104e-07
[ 2000: 4000] (0.12-0.25s) max|Δ| = 3.010e-06
[ 4000: 6000] (0.25-0.38s) max|Δ| = 2.314e-05
[ 6000: 8000] (0.38-0.50s) max|Δ| = 1.127e-04
[ 8000:10000] (0.50-0.62s) max|Δ| = 8.349e-04
[10000:12000] (0.62-0.75s) max|Δ| = 3.221e-03
[12000:14000] (0.75-0.88s) max|Δ| = 1.504e-02
[14000:16000] (0.88-1.00s) max|Δ| = 4.236e-02
```

- 引用泄漏数值必须同时给扰动协议（同分布替换 / ×10 / 单帧尖峰差异可达 2–4 个数量级，旧文档 §4.6-⑨ 的 P1–P5 表仍成立）。旧文档 P1 口径 1.4e-6 与本轮实测 1.43e-6 一致。
- 视频分支（04 域，仅作端到端背景）：FRCNN 前端时间感受野 ±2 帧=±80 ms（`[验证]` 改帧≥30 → 最左受影响帧 28）；`VideoBlock` eval 因果、train 因 `BatchNorm1d` 批内统计非因果（train 前缀 Δ=4.398e-02）。Studio `OutputSlicer` 的 80 ms 右上下文与此量级吻合（06 文档）。

### 2.8 结构常量、契约与“反直觉”事实（[代码]+[验证]）

- **顶层 `causal` 开关惰性**（`[验证]`，独立 deepcopy 避免 `Separator.pop` 污染）：

```text
causal=True  params = 537,697 | causal=False params = 537,697
state_dict keys 完全一致: True
同权重同输入输出 bit-identical: True | max|Δ| = 0.0
```

- **`mdcl_window` 非持久 buffer**：`state_dict` 525 key 不含它；`persistent=False`（`Falcon.py:1339-1343`）。
- **`n_src=1` 硬约束**：`n_src=2` 构造成功但前向 `RuntimeError: shape '[2, 224, 251, 129]' is invalid for input of size 7252896`（decoder `view(B*n_src, 112*n_src, ...)` 与 MaskGenerator 输出通道不一致，`Falcon.py:1168`）。
- **`DualPathRNN` 的 `"Comba"` 分支不可达**：torch 2.8 的 `torch.nn` **没有** `Comba`（本轮实测 `hasattr(nn,"Comba") == False`），而构造时 `getattr(nn, self.rnn_type)`（`Falcon.py:372`）会先抛 AttributeError；配置里 `rnn_type: LSTM`，`if "Comba" in rnn_type` 永不成立。
- **死代码类**：`PositionalEncoding/MultiHeadSelfAttention/FeedForwardNetwork/GlobalAttention` 均不在 `named_modules`（计数 0）；`ConvNormAct` 38 个、`ConvActNorm` 52 个、`InjectionMultiSum` 8 个、`DualPathRNN` 8 个、`MHSA2D` 4 个、`AudioBlock`/`VideoBlock`/`NeuroSCNetFusion` 各 1 个。
- **导出/遮蔽**：`__all__=['Falcon','FalconSpeakerContrastive','get','register_model']`，`from look2hear.models import *` 正常；`look2hear.models.Falcon` 是**类**（包属性被同名 import 覆盖），`get('Falcon')`/`get('falconspeakercontrastive')` 正常，`get('AVTIGER_causal')` 抛 `ValueError`。
- **MDCL 两套机制**：`Falcon.forward` 的 `training and near_audio is not None` 分支（`Falcon.py:1482-1486`）走 `_mdcl_loss`（Phase 1 已加别名，裸 `Falcon` 现返回标量 loss，`[验证]` `Falcon(base) train+near_audio -> tuple` 第二项 shape `()`）；而论文配置用的 `FalconSpeakerContrastive.forward` **用 `**kwargs` 静默吞掉 `near_audio`**（`Falcon_Contrastive.py:102`），返回 `(est, sep_emb)`，不走 MDCL。两套对比机制（NASS 损失 vs `mdcl_*`）互斥，config 里只有 `spk_contrast`（NASS）。
- **`_mdcl_loss_hard` 数值**（`[验证]` 本轮抽检）：K=3、β=1/20 与手算 `Σ softmax(β·sim)·relu(sim−margin)` `allclose=True`；K=0 → 0；`_sisnr` 20 dB 输入 → 19.995 dB；`_mag_sim` 同输入 16.229 dB。
- **`AttentionSpeakerPool`**（`[验证]`）：`[1,112,251,129] → [1,64]`，`pool_spk_embed` 后 `‖z‖₂=1.0`，参数 14,584。
- **`import look2hear.models` 的隐式依赖链仍成立**（`[验证]`）：`sys.modules` 含 `librosa/torch_complex/typeguard/pytorch_lightning/timm/rich`（来自 `layers/__init__.py` 全量 import 与 `utils/lightning_utils.py`）。Studio 副本靠裁剪 layers 规避（06 域）。
- **清环境可构建**（`[验证]`，不注入 stub，`PYTHONPATH=$PWD`）：`timm 1.0.30 / typeguard 2.13.3 / numpy 2.2.6`，`FalconSpeakerContrastive` 构造成功、`audio params=537,697`、`state_dict keys=525`。stub 仅为旧脚本兼容（只影响死类 `DropPath` 与 typeguard API）。
- **config 契约**：`configs/Falcon.yml` 的 `audionet_config` 共 11 个顶层键（`n_src, pretrained_vout_chan, causal, spk_embed_dim, video_bn_params, audio_bn_params, enc_dec_params, audio_params, video_params, fusion_params, mask_generation_params`）；**没有** `mdcl_*` 键 ⇒ 用类默认 `mdcl_margin=0.0 / mdcl_alpha=0.15`（`[验证]` 读属性）。

---

## 3. 当前缺陷与风险（P0/P1 状态）

| # | 问题 | 状态（2026-10-24 HEAD） | 影响 | 证据 |
|---|---|---|---|---|
| D1 | `concat_layers[2]` 从未调用（5,568 参数闲置；疑似 level-2 融合漏调用） | **仍在（有意保留）**。Phase 1 F13 决策：为保与官方权重 bit-identical 不修，待作者确认 | 参数口径必须写“生效 532,129”；不阻碍复现（官方权重本来就没用到这层） | `Falcon.py:868,882`；hook `{0:2,1:2,2:0,3:2}`；`REPRODUCTION_PHASE1_REPORT.md` F13 |
| D2 | `Falcon.forward(near_audio)` 调用不存在的 `_mdcl_loss`（旧 P0） | **已修（Phase 1）**：`Falcon.py:1464-1467` 加 `_mdcl_loss = _mdcl_loss_hard` 别名；裸 `Falcon` 训练路径可用，数值语义不变 | 无（论文配置本来不走它） | `git show ac91800`；`[验证]` 裸 Falcon 返回 `(est, loss())` |
| D3 | `FalconSpeakerContrastive.forward(**kwargs)` 静默吞 `near_audio`（MDCL 静默失效） | **仍在**（Phase 1 F14 只修一半，授权文件外） | 只影响 MDCL 支线；主线 NASS 不受影响 | `Falcon_Contrastive.py:102`；`[验证]` 传 near_audio 无 loss |
| D4 | `Separator.__init__` 原地 `pop` `repeats/shared`，同一 dict 二次构造静默残缺（537,697→49,081） | **仍在（类内）**；入口侧已用 deepcopy 规避（Phase 1 F9） | 任何“直接用同一 config dict 连着建两个模型”的分析/脚本都会得到残缺模型（本轮 `model_core_verify4.py [Q]` 就踩到：557,281 vs 68,665） | `Falcon.py:991-996`；`[验证]` 旧 §8-[T] |
| D5 | 顶层 `causal` 开关惰性 | **仍在（设计如此/命名误导）** | 论文“causal”实际由 `enc_dec_params.causal` + `layer_3.causal` 决定；口径必须写“窗口级因果” | `[验证]` bit-identical |
| D6 | 全模型非严格因果：STFT ±128、cLNhw 全窗、对称 k=5 卷积 | **仍在（架构属性，权重依赖）** | 不能宣称“逐采样流式”；Studio 走 2 s 窗口 + 0.5 s hop + 80 ms guard | §2.7 |
| D7 | `n_src>1` 不支持 | **仍在** | 单目标提取任务 OK（论文就是 n_src=1） | `[验证]` RuntimeError |
| D8 | `FRCNNVideoModel.train()` 无 `return self`（`eval()` 返回 None） | **已修（Phase 1，videomodels +7 行）** | 无 | `git show ac91800`；Phase 1 报告 |
| D9 | `__all__` 指向不存在名字（`AVTIGER_causal`） | **已修（Phase 1）** | 无 | `models/__init__.py`；`[验证]` import * OK |
| D10 | `evaluated_mac_params.py` 只统计 audio model、视觉输入 64×64（与实际 88×88 不符） | **仍在（README 口径）** | 10.6 G 不含视觉编码器；引用时不可与“总 MACs”混同 | `[代码]` `evaluated_mac_params.py:80-123`；`[验证]` §2.2 |
| D11 | 训练/评测/Studio 三处加载策略不一致（`strict=False` vs `strict=True` vs missing 抛错） | **部分收敛（Phase 1）**：`audio_visual_test.py` 已改 strict=True（`:301/:319`） | 降低“静默残缺权重”风险 | `[代码]`；Phase 1 F8/F16 |

**结论**：P0 级“模型本体不可用”类问题（D2）已修；D2 之外的 D1/D3–D7 都是**既有架构属性/支线问题**，且**不影响官方权重复现**（官方权重 strict 加载、输出 bit-identical、RealSSA-Sim 7.774/8.479 已由 Phase 2 实测 `[文档]`）。

---

## 4. 与旧文档及 Phase 1/2 的差异清单

### 4.1 git 证据：Phase 1/2 到底改没改模型（[验证]）

```bash
git log --oneline -- look2hear/models look2hear/utils      # 结果只有 2 个提交
ac91800 复现 Phase 1：修复训练/评测入口与指标，重建数据工具链与验证资产
a7af472 Falcon: training, evaluation and Falcon Studio
git log --oneline ac91800..HEAD -- look2hear/models look2hear/utils look2hear/layers
# （空输出：Phase 2 未改动模型/工具/层）
git diff a7af472 ac91800 -- look2hear/models
#  Falcon.py: +5 行（_mdcl_loss 别名，纯别名无数值变化）
#  __init__.py: __all__ 从 ["AVTIGER_causal"] 改为 4 个实际导出
git diff a7af472 ac91800 -- look2hear/videomodels/frcnn_videomodel.py   # +7 行：train() 补 return self
git diff a7af472 ac91800 -- configs/Falcon.yml                          # +2 行：仅注释（logger/gpus 说明）
```

- **Phase 1**：`look2hear/models/Falcon.py` +5 行（别名）+ `models/__init__.py` 修 `__all__`；另修 `videomodels`（`return self`）、`system/av_litmodule.py`（logger API）、metrics/dataset/评测入口——**均不改前向数值**；Phase 1 报告 §3.4 明确“参数 537,697/11,186,688 不变、fingerprint 完全一致”。
- **Phase 2**：只新增 `configs/LRS2_2Mix*.yml`（4 个）与报告/工具；`git log ac91800..HEAD -- look2hear/**` 为空。**模型层零改动**。
- 旧文档基线 `7576854` 的 `Falcon.py` 是 1487 行，HEAD 是 1492 行（+5 = 别名 + 注释）；**旧文档对 `Falcon.py:1461` 之后的行号引用整体 +5 偏移**（本文所有行号已按 HEAD 重取）。

### 4.2 旧文档 A01–A56 逐条判定（本轮复核）

状态含义：【确认】=旧结论在本轮复算/读码下成立；【修正】=结论成立但需改数字/口径；【推翻】=旧结论已被 Phase 1 修好，不再成立。

| 编号 | 状态 | 本轮依据/变化 |
|---|---|---|
| A01 | 确认 | `[验证]` 11 个顶层键列出，与旧文档一致 |
| A02 | 确认 | `[验证]` 537,697；+FRCNN 11,724,385 |
| A03 | 确认 | `[验证]` 逐模块 3,024/13,104/0/492,312/12,657/2,016/14,584 |
| A04 | 确认 | 逐模块实测与手算吻合：encoder 3,024 / audio_bn 13,104 / separator 492,312(=445,766+42,850+3,696) / mask 12,657 / decoder 2,016 / spk 14,584；旧文档 §4.7 的 20 行手算公式本轮未逐行重算，但合计口径一致 |
| A05 | 确认 | `[验证]` video_bottleneck 0 参数、Identity 输出=输入 |
| A06 | 确认 | `[验证]` L=16000/32000/48000 → T'=126/251/376，L 等长 |
| A07 | 确认 | `[验证]` train→tuple、eval→Tensor |
| A08 | 确认 | `[验证]` (251,129)→(126,65)→(63,33)→(32,17) |
| A09 | 确认 | `[验证]` n_freqs=[129,65,33,17] |
| A10 | 确认 | `[验证]` hook：编码只最深级、解码 2→1→0，concat 例外见 A51 |
| A11 | 确认 | `[验证]` AudioBlock×2 / VideoBlock×1 / caf×1 |
| A12 | 确认 | `[代码]` `:991-1006` + hook |
| A13 | 确认 | `[代码]` `:241-265`；v_proj groups=16 |
| A14 | 确认 | `[代码]` `:1247-1259` 逐行；112=2×56 |
| A15 | 确认 | `[验证]` `[1,1,112,251,129]`→`[1,1,32000]` |
| A16 | 确认 | `[验证]` n_src=2 前向 RuntimeError |
| A17 | 确认 | `[验证]` decoder 前缀 Δ=0、首变 16001 |
| A18 | 确认 | `[验证]` 帧 ≤124 全 0、首变公式三点命中 |
| A19 | 确认 | `[验证]` P1 左段 1.43e-6；cLNhw 30 个实例 |
| A20 | 确认 | `[验证]` norm 全关 lookahead 57 帧=7296 采样=456 ms；split 100/125/150→56/57/58 |
| A21a | 确认 | `[验证]` norm=cLNhw 脉冲 (0,250) |
| A21b | 确认 | `[验证]` norm=None 脉冲 (98,102) |
| A21c | 确认 | `[验证]` causal=True+norm=None 脉冲 (100,104) |
| A22 | 确认 | `[验证]` dim=4/3 前缀 Δ=0 |
| A23 | 确认 | `[验证]` causal=True Δ=0；causal=False Δ=4.114（旧值 4.68/4.97 是随机输入量级，非精确值） |
| A24 | 确认 | `[验证]` 0–0.12 s 5.104e-7 … 0.88–1.0 s 4.236e-2 |
| A25 | 确认 | `[验证]` 参数量/state_dict/输出 bit-identical |
| A26 | 确认 | `[验证]` 改帧≥30 → 最左受影响帧 28（±2 帧=80 ms） |
| A27 | 确认 | `[验证]` eval 前缀 Δ=0；train Δ=4.398e-2 |
| A28 | 确认 | `[验证]` mdcl_margin=0.0 / mdcl_alpha=0.15，config 无此键 |
| A29 | 确认 | `[验证]` `_sisnr` 20 dB 输入 → 19.995 dB（旧文档 20.000 为不同构造，量级一致） |
| A30 | 确认 | `[验证]` `_mag_sim` 16.229 dB（旧文档 16.33 为不同构造） |
| A31 | 确认 | `[验证]` β=1/20 与手算 allclose；K=0→0 |
| A32 | **推翻（已修）** | Phase 1 加 `_mdcl_loss` 别名（`Falcon.py:1464-1467`）；裸 Falcon train+near_audio 返回 `(est, loss())`，不再 AttributeError |
| A33 | 确认 | `[验证]` 子类 train()+near_audio 仍静默吞掉（返回 (est, sep_emb)） |
| A34 | 确认 | `[代码]` config 含 `spk_embed_dim`；裸 `Falcon` 不识别（旧 [验证]） |
| A35 | 确认 | 本轮 `model_core_verify4.py [Q]` 复现“同一 dict 二次构造→残缺模型（68,665 参数）”，正是该 bug |
| A36 | **推翻（已修）** | Phase 1 修 `__all__`；`[验证]` import * OK、get 正常 |
| A37 | 确认 | `[验证]` 4 个死类 named_modules 计数 0 |
| A38 | 确认 | `[代码]` `Falcon.py:6` DropPath 仅死类使用 |
| A39 | 确认 | `[代码]` `:926-929,997,1018,1208-1209` 读/不读；`fusion_type` 等未用 |
| A40 | 确认 | `[验证]` `[1,112,251,129]`→`[1,64]`、‖z‖₂=1、14,584 参数 |
| A41 | 未复跑 | NASS 损失/梯度属训练系统域（旧 [验证] 保留；本轮未跑） |
| A42 | 确认 | `[验证]` 525 key、mdcl_window 不在其中 |
| A43 | **推翻（已修）** | Phase 1 videomodels +7：`train()` 补 `return self`；`eval()` 不再返回 None |
| A44 | **修正** | Studio↔look2hear diff 由 56/65（净 −9）变为 **61/65（净 −4）**（Phase 1 别名 +5 行）；studio 1496 vs look2hear 1492 行。语义差异清单不变：DropPath 来源、注释语言、studio MPS bridge、look2hear 别名 |
| A45 | 确认 | README 0.5 M = 0.538 M 取整；扣对比头 0.523 M |
| A46 | 确认 | `[验证]` 662 条=525+137；architecture/epoch 同；mdcl_window 不在 |
| A47 | 确认 | `[验证]` audio 525/525、0 shape 差、strict=True OK |
| A48 | 确认 | `[验证]` video 137/137、strict=True OK |
| A49 | 确认 | `[验证]` 537,697+11,186,688=11,724,385 |
| A50 | 确认 | `[验证]` audio numel 537,762（+65）、video 11,196,308（+9,620） |
| A51 | 确认 | `[验证]` hook `{0:2,1:2,2:0,3:2}`、5,568、15 key |
| A52 | 确认 | `[验证]` 56/57/58 帧 |
| A53 | 确认 | `[验证]` 125/156/126 三点命中 |
| A54 | 确认 | README 图注↔代码结构；命名映射仍属 `[推断]` |
| A55 | 确认 | `[验证]` sys.modules 含 librosa/torch_complex/typeguard/pytorch_lightning/timm/rich |
| A56 | 确认 | `[验证]` falcon-big 654 条=517+137；SpeakerContrastiveV2/epoch 715 |

### 4.3 本轮修正/新增的事实（旧文档没有或需更新）

1. **A32/A36/A43 已报废**：Phase 1 已修（`_mdcl_loss` 别名、`__all__`、FRCNN `return self`）。旧文档与 `FINAL_PROJECT_UNDERSTANDING.md` 的 P0-8/P0-3 描述**已过期**（FINAL 是 Phase 1 前的快照 commit `7576854`，见其头部）。
2. **Studio 副本关系量化**：`Falcon_Contrastive.py` 两份**逐字节相同**；`SISNRRepel.py` 只是 studio `Falcon.py` 改名副本（新增 `SpeakerContrastiveV2` 用）；big 模型 audio 参数 **1,158,617**、总计 **12,345,305**（README 12.35 M），key 517/137、epoch 715。
3. **`mdcl_window` 验证口径**：旧文档“`'mdcl_window' in sd == False`”正确；但 `dict(model.named_buffers())` 检查**必然为 True**（非持久 buffer 也在 named_buffers 里），不能拿它当证据。
4. **`torch.nn.Comba` 不存在**（torch 2.8.0+cu128 实测），彻底坐实 `DualPathRNN` 的 Comba 分支死代码。
5. **`_mdcl_loss_hard` 的实测注意点**：随机无关音频下 `_mag_sim` 为负、`relu(sim−0)` 全 0，因此 K≥2 损失常为 0（旧文档用 margin=±500 才看到非零；这是 margin 语义，不是 bug）。
6. **Phase 2 完成态**（`[文档]` `docs/REPRODUCTION_PHASE2_REPORT.md:115-159`）：RealSSA-Sim test 全量 9169 条 **SI-SNRi 7.774 / SDRi 8.479**（论文 7.7/8.2）；LRS2 contrastive 微调 **8.858/9.270**；8 卡论文配方（8×bs5=40、1000 epoch、early stop 30、β 0→1/200 ep）已落地 `configs/LRS2_2Mix_Paper_8gpu.yml` + `tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh`，本机 2 卡 smoke 3 epoch 通过；官方 falcon.pt 在 LRS2-2Mix 的 **7.03/7.71 属跨语料**（训练集 `LRS3_complex_nonmoving`）。
7. **模型不是 7 dB 差距的原因**（`[文档]` P2-B）：big 仅 +0.971/+0.911 dB；窗口化 −1.912/−2.318 dB；置零 mouth −15.1 dB ⇒ 规模/权重版本/视觉分支均非主因。

### 4.4 旧文档中未复跑/未验证项（诚实标注）

- A41（NASS 损失数值/梯度）——训练系统域，本轮未跑，保留旧 `[验证]` 记录。
- README 的延迟/内存数字（`README.md:226`：CPU 1361.2 ms、GPU 51.3 ms）——本轮禁跑测速，**未验证**。
- 真实数据集上的精度——只引用 Phase 2 报告（`[文档]`），本轮未跑评测。
- 论文正文对“causal/streaming”的正式定义——无网络，**未验证**；本报告只按 README 图注 + 实现给出“窗口级因果”。

---

## 5. 对复现的影响与建议（回答 5 个重点问题）

### 5.1 ① 三个模型类（+ Studio big）到底差在哪，主线用哪个？

| 类 | 定义位置 | 音频参数（config） | 训练期返回 | 用途 |
|---|---|---|---|---|
| `Falcon` | `look2hear/models/Falcon.py:1277` | **523,113** | `est`；`near_audio≠None` 时 `(est, mdcl_loss)`（Phase 1 后可跑） | MDCL 支线（`av_litmodule_mdcl`）；config 无法直接构建（缺 `spk_embed_dim` 识别） |
| `FalconSpeakerContrastive` | `look2hear/models/Falcon_Contrastive.py:60` | **537,697**（+14,584 spk 头） | `(est, sep_emb)`；eval 只 `est` | **主线**：`configs/Falcon.yml`、Phase 1/2 全部训练配置、官方 `falcon.pt`、Studio small |
| `SpeakerContrastiveV2` | `studio/falcon_model/models/Speaker_Contrastive_v2.py:78` | **1,158,617**（256 通道 / hid 96,48 / spk 128） | 同 FalconSpeakerContrastive | Studio big（`falcon-big.pt`，epoch 715，12.35 M） |

- 三者**前向骨架完全相同**（Falcon 家族），差异只在：是否带 spk 头、spk_embed_dim、以及 config 里的通道/层宽（big 更大）。
- **主线复现结论**：用 `look2hear.models.FalconSpeakerContrastive` + `configs/Falcon.yml`（8 卡配方 `configs/LRS2_2Mix_Paper_8gpu.yml` 的 `audionet` 段与它**完全相同**）。`Falcon` 只用于 MDCL 支线；`SpeakerContrastiveV2` 只用于 Studio big 推理（`[验证]` 1,158,617 + 11,186,688 = 12,345,305），不是论文主表（0.5 M）对应的模型。
- **不要**用 `FalconSpeakerContrastive` 跑 MDCL（`near_audio` 被 `**kwargs` 静默吞掉）；也**不要**用裸 `Falcon` 跑 NASS 训练（无 spk 头，`pool_spk_embed` 不存在）。

### 5.2 ② 前向数据流与 mask 形式（复现/评测时最容易踩的 4 个点）

1. 输入形状：音频 `[B,L]` float32 16 kHz 单声道（**不做峰值归一化**，`configs/Falcon.yml` `normalize_audio:false`）；mouth 必须是 **`[B,512,Tv]`**（FRCNN 输出格式），不是 `[B,Tv,512]`（`VideoBlock` 的 Conv1d 会把它当通道）。
2. `T' = 1+⌊L/128⌋`、`F=129`；`mouth` 的 `Tv` 可以是 50（2 s@25fps），由 `NeuroSCNetFusion` nearest 插值到 `T'`。
3. mask 是 **ReLU 非负、无上界的特征域复数掩码**（112=2×56），不是 [0,1] 软掩码、也不是 STFT 实虚部掩码；`output_gate=False`。
4. `FalconSpeakerContrastive` 的返回值**随 `self.training` 变化**（train→tuple，eval→Tensor）；Studio/评测必须 `model.eval()` 后调用，否则会拿到 tuple。

### 5.3 ③ 官方权重与代码是否严格对应？

**是，严格对应（`[验证]`）**：

- `falcon.pt`：`dict(state_dict, architecture='FalconSpeakerContrastive', epoch=714)`，662 条 = audio 525 + video 137；audio 525 key 与 `FalconSpeakerContrastive(configs/Falcon.yml)` **完全相等**（0 missing / 0 unexpected / 0 shape 不一致），`strict=True` 成功；video 137 key 与 `FRCNNVideoModel` 完全相等，`strict=True` 成功。
- `falcon-big.pt`：`architecture='SpeakerContrastiveV2', epoch=715`，654 条 = 517 + 137；与 `SpeakerContrastiveV2(studio/configs/FalconBig.yml)` `strict=True` 成功（audio 1,158,617）。
- **没有缺失/多余 key**；`mdcl_window` 是非持久 buffer（不在 state_dict），不会污染 strict 加载。
- **一个“隐藏耦合”**：`concat_layers[2]` 的 15 个 key **在 ckpt 里、能 strict 加载、但前向从不使用**——因此“权重能 strict 加载”不能反证“模型前向用到了全部权重”。
- 训练侧 vs Studio 侧：同一权重、同一输入、CPU 前向 **max|Δ|=0（bit-identical）**；Studio `FalconRuntime` 真实加载 status `parameters=11,724,385`、`audio_weight_keys=525 / video_weight_keys=137`。

### 5.4 ④ 参数量口径与“因果/流式”的精确表述

**参数口径（引用时必须带口径）**：

| 口径 | 数值 | 说明 |
|---|---|---|
| 音频模型全部可训练参数 | **537,697 = 0.538 M** | README “0.5 M trainable”按 1 位有效数字取整 |
| 其中前向真正生效 | **532,129** | 扣 `concat_layers[2]` 5,568（1.04%） |
| 扣除训练期对比头 | **523,113** | 537,697 − 14,584（spk_pool_proj，仅训练用） |
| 再扣闲置层 | **517,545** | 532,129 − 14,584 |
| + 冻结 FRCNN | **11,724,385 = 11.72 M** | 与 README 精确一致（含冻结 11.2 M 视觉编码器） |

**因果/流式表述（建议写入 FINAL 的定稿句）**：

> Falcon 是**窗口级因果（window-causal）**、**单次 encoder–decoder** 的 AVTSE 模型：时间卷积/单向 LSTM/tril-masked attention 结构上不看未来，且**不携带跨窗口状态**；但 `torch.stft(center=True)` 有 ±128 采样分析窗、`cLNhw` 全窗统计、AudioBlock 对称 k=5 卷积——关掉归一化后单遍结构 lookahead 仍为 57 帧≈456 ms，as-built 端到端首变样本 9,612（相对扰动点 −649 ms）。因此它**不是**逐采样因果的流式模型，产品侧用 2 s 窗口 + 0.5 s hop + 80 ms 右上下文的分块重叠推理。

禁止的表述：❌“严格因果/streaming（sample-wise）”；❌“0 延迟”；❌“0.5 M 全部参与推理”。

### 5.5 ⑤ 8 卡训练配方是否复用同一模型定义？有无模型层改动？

**复用同一模型定义，无任何模型层改动**：

- 5 个配置的 `audionet` 段**逐个键完全相等**（本机脚本对 `LRS2_2Mix.yml / _Paper_2gpu.yml / _Paper_2gpu_smoke.yml / _Paper_8gpu.yml` 全比 `configs/Falcon.yml`，输出 `audionet section identical: True`）：`audionet_name=FalconSpeakerContrastive`，`n_src=1, causal=true, spk_embed_dim=64, out_chan=112, hid_chan=64, repeats=2, shared=true, win=256/hop=128, upsampling_depth=4`。
- 差异只在**数据/训练/优化器**：`LRS2_2Mix.yml`（100 epoch，[0,1] GPU）、`Paper_2gpu`（1000 epoch，[0,1]）、`Paper_2gpu_smoke`（冒烟）、`Paper_8gpu`（`gpus:-1`、8×bs5=全局 40、1000 epoch、early stop 30、β 0→1/200 ep）。
- **Phase 2 未改 `look2hear/models/**`**（§4.1）；Phase 1 的模型文件改动只是别名/`__all__`（不影响前向数值，Phase 1 报告 §3.4 fingerprint 一致）。因此 8 卡配方训练出来的 ckpt 与官方 `falcon.pt` **同 key 同 shape**，可直接被 `audio_visual_test.py` / Studio `strict=True` 加载。
- 引用 Phase 2 结果时注意：官方 `falcon.pt` 的 LRS2-2Mix **7.03/7.71 是跨语料**（训练集 LRS3_complex_nonmoving），不是论文 LRS2 列（14.1/14.7）的复现；RealSSA-Sim 7.774/8.479 才是与论文同口径（`[文档]`）。

### 5.6 建议（按优先级）

1. **保住 bit-identical**：任何“补 `concat[2]` 调用 / 改 cLNhw / 改 causal padding”的改动都会让官方权重语义变化，不能用于“复现论文数字”这一目标；如需修，另开分支并重训/重评（§3 D1）。
2. **复现文档统一口径**：参数一律写成“537,697（0.538 M；生效 532,129；扣对比头 523,113）”，因果一律写“窗口级因果 + 无跨窗状态 + 非逐采样流式”（§5.4）。
3. **MDCL 支线**若要启用，需二选一：用裸 `Falcon` + `av_litmodule_mdcl`；或给 `FalconSpeakerContrastive.forward` 显式转发 `near_audio`（会改变返回契约，需同步 litmodule，本轮禁改源码，仅记录）。
4. **评测入口别用 `look2hear/utils/separator.py`**：它会对输出做幅度重缩放（`out_wavs *= wav.abs().sum()/out_wavs.abs().sum()`），会污染 SI-SNRi/SDRi；用 `audio_visual_test.py`/Studio runtime 的路径。
5. **Studio big** 只作展示/对比：12.35 M 参数、`SpeakerContrastiveV2`，与论文主表的 0.5 M small 不是同一模型；比较 big/small 差值（P2-B：+0.97 dB）时要说明参数规模与权重版本不同。

---

## 6. 未解问题

| # | 问题 | 需要谁 |
|---|---|---|
| Q1 | `concat_layers[2]` 到底是“有意跳过”还是 `AudioBlock.forward` 漏调用？官方权重里这 15 个 key 有训练值，说明作者训练时也没用到（若作者确认是 bug，会影响“0.5 M 有效容量”的论文口径） | 作者确认（Phase 1 已列 F13） |
| Q2 | 论文正文对 “causal / streaming” 的正式定义（帧级？窗口级？是否承认 cLNhw 全局统计） | Lead 对照论文 |
| Q3 | `falcon.pt` epoch 714 / `falcon-big.pt` epoch 715 与论文各表（RealSSA/LRS2/LRS3/VoxCeleb2）的对应关系；ckpt 只有 `state_dict/architecture/epoch`，无训练超参 | Lead + 训练系统 |
| Q4 | `SpeakerContrastiveV2`（big）在论文中的定位：是论文主模型还是 Studio 的更大变体？README 表只列 0.5 M/11.72 M，未解释 12.35 M | Lead 对照 README/论文 |
| Q5 | `FalconSpeakerContrastive` 静默吞 `near_audio` 是否影响论文 NASS 训练路径（NASS 用的是 sep_emb，不走 near_audio，故预期无影响；但需训练系统确认 litmodule 实际传参） | train-sys |
| Q6 | `evaluated_mac_params.py` 的 64×64 视觉输入与训练/评测 88×88 的不一致是否影响 README 的 10.6 G 可复现性（视觉编码器 MACs 未计入；audio 部分与输入帧数相关而非像素） | 04 域 |

---

## 7. 验证记录

### 7.1 环境

```text
/root/miniforge3/envs/falcon/bin/python -V  -> Python 3.11.17
python -c "import torch" -> 2.8.0+cu128 | cuda_available True | devices 2
timm 1.0.30 | typeguard 2.13.3 | numpy 2.2.6
```

> 注：`docs/verification/scripts/*` 内部会 `sys.path.insert` 旧沙箱 stub（`docs/verification/stubs`：`timm.models.layers.DropPath` + `typeguard`），对 Falcon 前向**无数值影响**（只被死类/第三方 API 用到）。本轮另做了**不注入 stub** 的清环境构建验证（§2.8）：`537,697 / 525 keys`，与带 stub 一致。

### 7.2 本轮运行清单（全部 < 5 min，未启动训练/下载/全量评测）

| 脚本/命令 | 结果 | 用途 |
|---|---|---|
| `verify_model_params.py` | exit 0 | 逐模块参数、前向形状、因果探测、`_mdcl_loss` 状态 |
| `verify_causality.py` | exit 0 | 单点扰动首变样本 9612 / −10388 采样 / −649.25 ms；逐 stage 首变帧 |
| `verify_macs_04.py` | exit 0 | MACs 10.611 G；视觉编码器 MACs 7.77/15.90/17.48 G |
| `verify_real_weights.py` | exit 0 | falcon.pt/falcon-big.pt 结构、strict 加载、Studio bit-identical、Runtime status |
| `model_core_r3_verify.py` | exit 0 | W1–W8：权重 key、concat[2] hook、lookahead 56/57/58、首变帧公式 |
| `model_core_verify4.py` | **[O][P] 通过；[Q] 因脚本自身 dict 污染崩溃（非模型问题）** | lookahead、端到端泄漏分段、视频因果；[Q] 已用独立 deepcopy 重做（§2.8），结果 bit-identical |
| 本报告 §2.1/2.3/2.6/2.7/2.8 的 heredoc 小脚本 | 全部通过 | 参数口径、hook 计数、模块计数、A19/A21/A22/A23/A29–A31/A40 复核、causal 惰性、n_src=2、导出、清环境构建 |
| `sha256sum /root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}` | 与旧文档一致 | 权重指纹 |
| `git log/diff`（只读） | 见 §4.1 | Phase 1/2 模型改动核实 |

### 7.3 本轮未做（遵守预算/禁区）

- 未启动训练、未下载数据/权重（权重已缓存）、未跑全量评测、未测延迟/显存。
- 未复跑 A41（NASS 损失数值/梯度，训练系统域）。
- 未修改任何源码/配置/他人文档；未做 git 写操作；未联网。
- 旧文档 `docs/verification/scripts/model_core_verify4.py [Q]` 的 dict 污染问题**已在本文记录**（`[Q]` 原始输出不可引用，本文用独立 deepcopy 重做了等价实验）。

### 7.4 证据文件位置

- 旧文档/基线：`docs/understanding/01-model-core.md`（R3，975 行，commit `7576854`）
- 汇总（部分过期）：`docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` §3.4/§10（P0-8 等已过期，见 §4.3）
- Phase 1 修复清单：`docs/REPRODUCTION_PHASE1_REPORT.md`（F13/F14、§3.4 数值不变量）
- Phase 2 结果：`docs/REPRODUCTION_PHASE2_REPORT.md`（§1/§4/§5.3）
- 验证脚本：`docs/verification/scripts/{verify_model_params,verify_causality,verify_macs_04,verify_real_weights,model_core_r3_verify,model_core_verify4}.py`
- 权重：`/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`（sha256 见 §2.5）
