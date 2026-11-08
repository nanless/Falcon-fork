# 00 总纲 · Falcon-fork 全景深挖（2026-11-08）

> 本轮：1 名 Lead + 8 个持久队友 + 4 个子代理，10 个 Wave-1 域深挖（01–10）+ 2 个 Wave-2 攻坚（11 论文对齐 / 12 因果性）。
> 目标仓库：`dev_a800_2gpus:/root/code/github_repos/Falcon-fork`，HEAD `6c1d5b7`（tag `repro-freeze-20261031`）。
> 环境：`/root/miniforge3/envs/falcon`（Py3.11.17 / torch 2.8.0+cu128 / 2×A800-80G）。
> **全部结论都回到源码/实测**；与 `docs/` 既有文档冲突处单列（§6）。仓库全程只读，git 状态未动。

---

## 0. 结论速览

1. **这是什么**：NeurIPS 2026 论文 *Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World
   Acoustic Scenes*（Falcon，arXiv:2610.32749，JusperLee/Falcon）的**复现 + 工程化 fork**。
   任务是 **AVTSE（音视频目标说话人提取）**：给定混合语音 + 目标说话人唇部视频，输出**单路**目标语音。
   **不是 2 源分离**——虽然 benchmark 名叫 LRS2-2Mix。
2. **模型有多小**：音频主干 **537,697 参数**（= 论文 0.5 M ✓）；其中 **5,568（1.04%）是死参数**
   （`concat_layers[2]`，清零后输出 bit-identical）。冻结视频前端 FRCNN **11,186,688**，合计 11,724,385。
3. **参数 537,697、音频 10.611 GMAC，与论文的 0.5 M / 10.6 G 同尺**。10.611 G 是 ptflops 口径：conv hook
   每个乘加记 1，且**含 LSTM**，只算音频、不含 FRCNN。同一模型换 `torch.utils.flop_counter` 只剩
   **5.382 GMAC**，因为它**不给 `aten::lstm` 记账**：两把尺子各自成立，不可与 10.6 并列。
   视觉前端（88×88×50 帧）**15.897 GMAC**，整窗 **26.51 GMAC、视觉占 60.0%**。
   02/03 早期写的"5.382 G 是真 MACs""21.19 GMAC / 视觉 74.6%"混用了尺子，2026-11-08 Lead 复核已更正。
4. **结构一句话**：STFT(win256/hop128, center=True) → 3 通道(real/imag/mag) → 112 通道 →
   **4 级 (T,F) 同时 8× 下采样的二维金字塔 AudioBlock**（每级：F 向双向 LSTM + T 向单向 LSTM +
   tril-mask 自注意力，带 top-down InjectionMultiSum）→ 复数掩码（RI 2×2 乘法）→ iSTFT。
5. **视听融合不是 cross-attention**：视频只经 `Conv1d(512→112, groups=16)` + `nearest` 上采样(50→251) +
   `unsqueeze(-1)`，做**逐帧、频率平坦的加性偏置**（`Falcon.py:241-267`）。全模型无跨模态 Q/K/V。
6. **因果性只有"窗间无状态"一层成立，lookahead 分三档**。单遍几何 **57 帧 = 456 ms**（关掉 `cLNhw`
   后一遍 AudioBlock 的感受野）；整链几何 **113 帧 = 904 ms**（编码器阶梯 15 帧 + 57 + 第二遍 56；
   另一路实测 110 帧 / 880 ms，差约 3%）；as-shipped 因 30 处 `cLNhw` 依赖到**窗尾（≤2000 ms）**：
   `cLNhw = GroupNorm(1,C)` 在 **112×251×129 ≈ 3.6 M** 元素上统计（音频路径 30 处）。实测：帧 245 之后注入 1e-3 噪声，
   **帧 0 仍变化 −52 dB**；换成论文式逐 (t,f) `LN_c` 后帧 0…120 变成 **−240 dB（数值 0）**，才回落到 904 ms。
   单点脉冲影响 **[13, 31999]**。**论文 §4.2/App E 的"无时域 lookahead / audio 15 帧=120 ms"与代码不符。**
7. **视频侧的时间建模是坏的**：`VideoBlock` 把 `(B,hid,T)` 转成 `(B,T,hid)` 喂给 `batch_first=False`
   的 LSTM ⇒ 序列维被解释成 batch，**帧间无递推**（脉冲响应只有 1/50 帧变化）。视觉的真实时间上下文
   只剩 FRCNN 的 **±2 帧 = ±80 ms**。
8. **解码器有一个 16 ms 结构性延迟**：`causal=True` 时前向 pad 2 帧、只在末尾裁 1 帧 ⇒ 净 +2 帧 = +256 采样。
   delta 权重直通实测 lag=+256、corr=1.0000；真实权重端到端 lag=0（被掩码吸收）→ **重训/换 hop 会重现**。
9. **训练口径三处失真**（v2 配置）：① 训练损失默认是 **−SNR**（非尺度不变），只有 v2 改 −SI-SDR；
   ② 论文 Table 16 的 τ(β) 0→1/1000ep 在 LRS2 上**空转**（每样本 K≡1，K==1 分支不含 β）；
   ③ margin=0.5 **失效**（真实 run cos_neg≈0.9845 ⇒ `relu(sim−0.5)` 恒正）。
10. **复现状态**：L1 结构/权重/参数 ✅（strict 加载、bit-identical、数字逐位复现）；
    RealSSA-Sim **7.774/8.479**（9,169 条，ckpt mean）、ch0 **7.977/8.452**：数值吻合（论文 7.7/8.2），
    但 ckpt（LRS3 训练）与通道（mean）两轴不同，不构成严格复现；
    LRS2 跨语料 7.029/7.713、自训 ep124 8.858/9.270；**论文 LRS2 14.1 未复现**（缺 LRS2-trained ckpt / 8 卡）；
    LRS3-2Mix / VoxCeleb2-2Mix **数据未发布**。
11. **证据质量**：数字可复算到 **5.3e-8 dB**（SI-SNRi）、跨 7 天两次全量 run 差 3.3e-7 dB；
    但 full3 DNSMOS 已作废、RealSSA-Sim 的 SDRi/PESQ 实际分母是 **9,166 不是 9,169**、
    A800 的 47.1/56.5 ms **只有 md 表格没有原始测量文件**。
12. **工程面**：Studio = 单进程回环 FastAPI + 纯静态前端 + 浏览器 MediaPipe；Space 字节级复用 Studio 运行时
    （54 个文件、约 39.87 MB，mediapipe wasm 占 98.8%）；测试 168（有/无 GPU 同值）/ studio 51+2 / node 8/8；
    **CI 的 repro job 大概率是红的**（ubuntu 浅克隆；无 GPU 那一项已修）。

---

## 1. 三条产品线，一套模型定义

| 线 | 路径 | 是什么 | 关键事实 |
|---|---|---|---|
| 训练/评测 | `look2hear/**`、`audio_train.py`、`audio_visual_test.py`、`configs/**` | 上游官方代码 + fork 的复现改造 | 训练入口可跑（CPU 2 epoch 实测通过）；评测**整段一次前向**，不切窗 |
| Studio（演示） | `studio/**`（`server.py` 419L + `falcon_model/**` 副本 + `web/**` 4302L JS） | 实时 demo：麦克风+摄像头 → 2 s 窗 → 0.5 s hop 输出 | 单进程、只绑 127.0.0.1、无鉴权/限流/TLS；模型无跨窗状态 |
| Space（云端） | `space/**`（`build.py` 组装）、`project_page/**` | HF ZeroGPU 版：录完→整段→一次推理 | **服务端零复用**（`server.py` 不进 Space）→ 无实时能力 |

**一套模型**：`studio/falcon_model/` 是 `look2hear/` 的推理副本，sha256 不同但**同权重同输入输出 bit-identical
（max|Δ|=0）**，差异只有 `timm.DropPath`→本地实现、STFT 的 MPS 分支、注释翻译。

---

## 2. 数据流与形状（2 s 窗口 / B=1，全部实测）

```
音频 32000 样本 (2 s @16k)
  └─ torch.stft(n_fft=256, hop=128, hann, center=True, reflect) → (129, 251) 复数
       └─ stack[real, imag, |·|] → (3, 251, 129) → Conv2d(3→112, k3, c2f) → (112, 251, 129)
            └─ audio_bottleneck: cLNhw + Conv1d(112→112, k1) → (112, 251, 129)

视频 mouths/<utt>.pt : uint8 (50, 96, 96, 3)
  └─ Grayscale → (50, 96, 96) → RandomCrop(88)/CenterCrop(88) → /255 → (x−.421)/.165
       └─ FRCNN(冻结, resnet trunk, 3D frontend) → (512, 50)          # T 不降采样
            └─ video_bottleneck: kernel_size=-1 ⇒ nn.Identity()      # 0 参数
                 └─ VideoBlock: PReLU-BN conv + LSTM(布局错误) → (112, 50)

融合（每级 AudioBlock 内）:
  video (112,50) --nearest--> (112,251) --unsqueeze(-1)--> (112,251,1)
  audio (112,251,129) + 上式  → (112,251,129)         # 加性偏置，频率维广播

Separator = 1 次 (AudioBlock → CAF 加性融合) + 1 次 AudioBlock，权重 shared
  AudioBlock: 251×129 --s2×3--> 32×17 （F 与 T 同时 8× 下采样）
              最深级: F向双向LSTM + T向单向LSTM + tril-mask 2D 自注意力
              解码: top-down InjectionMultiSum + nearest 上采样回 251×129 + 残差

MaskGenerator: PReLU → Conv2d(112→112, k1) → ReLU → view(B,1,112,251,129)
   RI 复数乘: (E_real,E_imag) ⊗ (M_real,M_imag)（把 112 通道硬切 0:56/56:112 当实/虚）
STFTDecoder: ConvTranspose2d(112→2, k3, s1) → complex → transpose → istft(length=32000)
   └─ 输出 (1, 1, 32000)
```

**融合/上采样细节**：`fusion_repeats` 取自 **`video_params.repeats`**（`Falcon.py:1000`），所以"视频块数"决定
"融合次数"；`audio_repeats = audio.repeats − fusion_repeats`。3 处 `F.interpolate(mode="nearest")`，
全库 `align_corners` 零出现。

---

## 3. 机制总览（7 条，逐条带证据）

1. **任务**：`n_src: 1` 出现在全部 6 份 config；数据索引只有 `mix.json + target.json`（`target=s1`）；
   `PITLossWrapper` 在实跑路径上是恒等映射。**`n_src≥2` 在代码上是坏的**：
   实测 `n_src=2` 前向直接 `RuntimeError: shape '[2,224,251,129]' is invalid`
   （`decoder.in_chan=encoder_out_chan*n_src` 与 MaskGenerator 每源通道数不自洽）。
2. **多尺度分离** = (T,F) 二维金字塔，不是频率单轴：每级 downsample 同时把 T 与 F 减半（251×129→32×17），
   最深级才跑 F/T 双向与单向 LSTM + 自注意力；解码用 4 个 `InjectionMultiSum` + 1 个
   `concat_layers[-1]` 做 top-down 融合（`concat_layers[2]` 从未调用 → 死参数）。
3. **归一化是因果性的决定性因素**：`cLNhw = GlobalLayerNorm(C,T,F) ∘ CumulateLN4D(沿 T 累积、统计跨 C)`；
   初始化态 GLN 段近似数学空操作（差 3.576e-07），净增量只有 2×C 个仿射参数，但它把整窗耦合在一起。
4. **复数掩码的相位被锁**：`mask_act=ReLU` ⇒ 实部与虚部同时非负 ⇒ 每 bin 掩码相位限死 **[0°, 90°]**。
   真实权重下掩码 **98.9% 为 0**，`|Z|/|E| = 0.039`（掩码近乎"开/关"而非"比例缩放"）。
5. **对比学习（NASS）**：`encode_stft` 在 STFT 特征上做注意力池化得 64 维嵌入；
   `nass_contrastive_loss` = α₊(1−cos_pos) + α₋·relu(cos_neg − margin)，K=1 时退化为单负样本且**不含 β**；
   K=0 时仍返回 α₊(1−cos)（≠0，与 MDCL 的 K=0→0 不一致）。
6. **MDCL**（另一条路径）：`_mdcl_loss_hard` 用**幅度谱相似度**替代波形 SI-SNR，K≥2 时按
   `softmax(β·sim)` 做难负样本加权；K=1 时退化为 `relu(sim−margin)`；K=0 时为 0。
   仅 `self.training and near_audio is not None` 时生效（`Falcon.py:1493`），推理零成本。
7. **视频条件的两处"形同虚设"**：① `VideoBlock` LSTM 布局错误 → 无时间递推；② `_encode_audio` 的
   dummy video 是 `zeros(B,512,1)`，经 VideoBlock 后被 bias 抬成 **norm=9.72 的逐帧常数向量**，
   即"零视频"仍注入一个学习到的常数偏置（数值上与 `zeros(512,50)` 差 3e-6，语义上是坑）。

---

## 4. 硬数字表（引用时带口径；MACs 有两把尺子）

| 量 | 值 | 口径 | 证据 |
|---|---|---|---|
| 音频参数（总） | **537,697** | 全部 requires_grad；含死参数 5,568 + 对比头 14,584 | 01 §2、05 §3.1 |
| 扣死参数（= 生效参数） | 532,129 | 537,697 − 5,568；`concat_layers[2]` 从未被调用，清零后 bit-identical | 01 §4（max\|Δ\|=0） |
| 扣对比头 | 523,113 | 537,697 − 14,584；`spk_pool_proj` 只在训练侧单独调用，不在 forward 里 | 05 §3.1 |
| 两者都扣（分离前向） | 517,545 | 532,129 − 14,584 = 537,697 − 5,568 − 14,584 | 01 §4、05 §3.1 |
| state_dict 条目 | 537,762 | +65 个持久化 BN buffer | 01 |
| 视频前端 | 11,186,688 | 冻结、不进 optimizer | 03 §4 |
| 端到端参数 | 11,724,385 | 537,697 + 11,186,688 | 01/05/09 一致 |
| Big 变体 | 1,158,617 / 12,345,305 | 音频 / 端到端；`SpeakerContrastiveV2` **只存在于 studio** | 01 §4、08 §4.4 |
| 音频 MACs（ptflops） | **10.611 G** | 含 LSTM、不含视觉；**与论文 10.6 G 同尺** | 01/05/11 三方复算一致 |
| 音频 MACs（FlopCounterMode） | 5.382 G | **漏 `aten::lstm`**，不可与 10.6 并列 | 02 §4 |
| 视觉 MACs | **15.897 G** | 88×88×50 帧、ptflops 同尺（96²≈17.4 G、88²×25≈7.9 G 是 flop_counter 口径） | 03 §4、11 §3 |
| 整窗合计 | **26.51 GMAC，视觉占 60.0%** | 10.611 + 15.897（同尺） | 11 §3（本轮 Lead 复核口径后更正） |
| 几何 lookahead | **113 帧 = 904 ms** | 单窗内、真实权重；解析 115 | 12 §3 |
| as-shipped lookahead | **窗尾（≤2000 ms）** | 30 处 cLNhw 全窗统计 | 12 §4 |
| 单窗延迟（CPU） | **约 0.6 s**（4 线程 592–694 ms；1 线程约 1.0 s） | B=1、2 s 窗、CPU | 12 §5 |
| 单窗延迟（A800） | 47.1 ms（small）/ 56.5 ms（big） | **只有 md 表格，无原始测量文件** | 07 §1（标为不可引用） |
| 跨窗一致性 | **+14.71 dB / r 0.9835 / 53.4% 峰值** | 6 s 流（3×2 s + 10 ms join 淡化）按 Studio 分块 vs 整段一次前向，含冷启动首窗；稳态 **+16.72 / 0.9895 / 28.5%** | 12 §4.7 |
| guard 敏感性 | 80→456 ms：丢弃窗尾 **+2.2 dB**（14.51→16.71，非单调）；补右上下文 **+0.8 dB**（22.12→22.92，饱和） | 同上，两种语义分开量 | 12 §4.7 |
| RealSSA-Sim | **7.774 / 8.479** | 9,169 条整段，ckpt mean，falcon.pt | 07 §3 |
| RealSSA-Sim ch0 | **7.977 / 8.452** | 同上，channel-1 | 07 §3 |
| RealSSA-Real DNSMOS | **OVRL 2.7941 / P808 3.3365** | full4（窗口起点已修）；**与论文 3.276 不可比** | 07 §3 |
| LRS2 跨语料 | 7.029 / 7.713 | tt 3,000，官方 falcon.pt（LRS3 训练，ep714） | 07 §3 |
| LRS2 自训 ep99 / ep124 | 8.726/9.122 / **8.858/9.270** | 单卡 100/130 epoch，旧配方（−SNR） | 07 §3 |
| mix 基线 | 4.9937 dB | tt 3,000 的 SI-SDR(mix, s1) 均值 | 07 §3 |
| LRS2-2Mix SNR | **∈[0,10] dB，中位 5.025** | `g1∈[0,5], g2=−g1`，PCM16 舍入（max\|Δ\|=3.05e-5） | 06 §4.4 |
| 训练吞吐 | 46.9 样本/s ≈ 11.7 it/s | nw=8 / bs=4 / paper-tr | 06 §4.6 |
| 1 epoch 覆盖 | ≈22–25% 唯一 mix | `train_samples=10000` vs 池 40000 | 06 §4.5、04 §5 |
| 测试 | 168 passed（约 147 s） | 无 GPU **同样 168 passed**（约 170 s）；修前为 166 passed / 2 failed，已修 | 10 §4 |
| Studio 测试 | 51+2（有 CUDA）/ 53+0（无） | pytest studio/tests | 08 §4 |
| Node 测试 | 8/8（≈4 s） | studio/tests/*.mjs | 08 §4 |
| Space 构建 | 54 个文件、约 39.87 MB | mediapipe wasm 占 98.8%；57 是误把 3 个目录计为文件 | 09 §4 |
| i18n | 226（space）/ 368（studio） | 两套 checker 覆盖不相交 | 09 §4 |

---

## 5. 坑清单（27 条，逐条带 `path:line`）

**模型/DSP**
1. `cLNhw` 是全窗 GroupNorm ⇒ **窗内完全非因果**（`layers/normalizations.py:134-146`；01 §4、02 §4）。
2. `causal=False` 硬编码在下采样/融合/上采样卷积里（`Falcon.py:772,808,821`），不是可配项。
3. `concat_layers[2]` 5,568 参数**从未被调用**（`Falcon.py:848-884`），是死参数。
4. `n_src≥2` 前向必崩（`Falcon.py:1175` vs `:1345`）——"2 源分离"的说法在代码层不成立。
5. `VideoBlock` LSTM 布局错误 ⇒ 视频无时间递推（`Falcon.py:952-955`）。
6. `mask_act=ReLU` ⇒ 复数掩码相位锁死 [0°,90°]（`Falcon.py:1229`、`:1263`）。
7. 解码器 +256 采样结构延迟（`Falcon.py:1178-1184`），重训/换 hop 会重现。
8. `causal=False` vs `True` 的 STFT 往返：center=False 时重建 SI-SNR 只有 31.8 dB（边缘 OLA 不足，02 §4）。
9. `mask_act`、`RI_split` 把 112 通道硬切实/虚部——实/虚划分是学出来的、非物理（01 §5）。
10. `torch.stft` 在 CPU fp16 直接崩（MKL FFT Half）⇒ `model.half()` 推理路径不可用（02 §4）。

**数据**
11. `idx` 被丢弃 ⇒ `shuffle` 无效、**有放回采样**、epoch 长度与数据量解耦（`avspeech_dataset_near.py:154`）。
12. LRS2 每条恰好 2.000 s ⇒ `rand_start=randint(0,0)=0`，**训练没有随机裁剪**（06 §4.1）。
13. Dataset 输出 **float64**（`transform.py:73` 的 numpy 除法），靠 litmodule `type_as` 救（08/03）。
14. `mouth_start = int(rand_start/16000*25)` 把采样率与 fps **写死**（06 §4.2）。
15. 缺帧**没有 pad**，只有 slice；换非 2 s 数据集会在 `collate` 的 `stack` 处崩。
16. `.pt` 存的是 3 通道复制灰度（`lrs2_2mix_to_falcon.py:224`），tr 多占 ≈13 GB（03 §5）。
17. near = 同 mixture 的 s2 **原始波形且未乘 g2**（06 §4.2）；RealSSA 侧 100% fallback ⇒ K=0。
18. `mouths/` 在 `falcon_lrs2_2mix`↔`_paper`、`realssa_sim`↔`_ch0` 之间是**硬链接**：
    逐目录 `du` 相加 110 G，真实并集 **62 G**（06 §4.3）。

**训练/评测**
19. `threshold_byloss=True`（默认）**丢的是好样本**（`min_loss > -30` 保留更差的；05 §4）。
20. `hparams.yaml` 含 5 处 `!!python` 标签（torch.Tensor），`yaml.safe_load` 直接失败（04 §4.3）；
    `conf.yml` 里 `main_args` 恒空，`--conf_dir/--ckpt_path` 落错 argparse 组（`parser_utils.py:143-148`）。

**工程**
21. `checkpoints/` 里 **88 个 5,269 B 废桩 ckpt**（2026-11-08 实测；每跑一次测试 +1），是 pytest 每次跑出来的副作用（`state_dict={'audio_model.w': ()}`）；
    `*.ckpt` 被 ignore，`git status` 永远干净（10 §4、08 §4）。
22. Studio **没有 pydantic**、`docs_url=None`，契约只能读 `decode_request`（`server.py:65-97`）。
23. Studio 的 "80 ms right context" 不是感受野：窗内 cLN+全局注意力让任一输入样本影响**全部 32000 输出**；
    GUARD 真正挡的是窗尾 iSTFT 边界瞬态（末 320 样本能量 5.39× 中位数）（08 §4）。
24. `space/README.md:59` 的 `pytest space/tests` 在 falcon 环境是 **4 个 ERROR**（缺 `spaces`），不是 skip（09 §4）。
25. **guard 80 ms 远小于模型所需，但代价很小**：几何上要 456–904 ms 未来上下文；块 = 窗内帧 178–240，
    而"足上下文"要求 `t ≤ 137` ⇒ **整块 63 帧都在几何上不足**；实测块内位置剖面 +22.9/+6.4/+9.8/+13.8/+8.6 dB。
    把 guard 拉到 456 ms 只回收 **+2.2 dB（丢弃窗尾语义，非单调）/ +0.8 dB（补右上下文语义，456 ms 后饱和）**
    —— 几何"上下文饥饿"与可测质量不是一回事（12 §4.7）。
26. **跨窗不连续是实打实的，但误差主体不是 hop/guard**：6 s 流（3×2 s 片段 + 10 ms join 淡化，Studio 从 t=0 开窗）
    按 Studio 方案拼接 vs 整段一次前向 = **+14.71 dB / 0.9835 / 最大偏差 53.4% 峰值**，稳态 **+16.72 / 0.9895 / 28.5%**；
    换窗对齐 ±250 ms 只让整段一致性在 **15.9–18.0 dB** 波动（2.1 dB），任两种对齐互相 **~17 dB**；
    20 ms 交叉淡化只降缝跳变（1.24e-2→9.57e-3），SI-SNR 一点没变。误差主体是 `cLNhw` 的窗长依赖：
    同一 ckpt 任意两种 2 s 窗方案互差 13–19 dB（12 §4.7）。
27. **论文的 120 ms 与代码不是同一个量**：论文 App E 说 "left-aligned padding … adding no temporal lookahead"，
    代码是**对称 padding**（`Falcon.py:48,758-819`）；论文 "audio branch at most 15 frames" 的 15 = `2^4−1`
    恰好只是**编码器最深层的累计值**（不含 GCI/decoder +44、不含第二遍 +56）（12 §6、11 §5）。

---

## 6. 与既有 `docs/` 的冲突总表（只列需要改文档的）

| # | 既有说法 | 实际 | 判定 |
|---|---|---|---|
| C1 | STATUS:41「单遍 lookahead 实测 57 帧≈456 ms」 | 数字对，但**缺限定**：57 帧是「关掉 cLNhw 后一遍 AudioBlock」的几何值；整链几何 **113 帧/904 ms**（另一路 110 帧/880 ms），as-shipped 是**窗尾 ≤2000 ms**。旧文 `docs/understanding/01-model-core.md:506,545` 写对了，是 STATUS 的压缩句把限定丢了 | 00–06 已按三档改写；`STATUS.md` 由 Lead 本轮重写，旧 `:41` 的压缩句随之废止 |
| C2 | STATUS:41「causal = 窗口级因果」 | 窗间无状态 ✅；窗内连帧级因果都不成立 | **表述需降级** |
| C3 | README/STATUS「0.5 M / 10.6 G MACs」 | 0.5 M ✓；10.6 G 是 **ptflops 含 LSTM 的 MACs**（不是 FLOPs），且不含视觉；整窗 26.51 GMAC、视觉占 60.0% | 数字对，00–06 已补"不含视觉 / 整窗 26.51 G"；README/STATUS 仍待改 |
| C4 | AGENTS.md:91「训练入口存在已知 P0（Comet logger、conf.yml 类名）」 | Comet logger 已在 `ac91800` 修；conf.yml 类名半修（带指引的 AttributeError） | **过期 11 天** |
| C5 | STATUS:64「Studio 51 passed / 2 skipped」 | 有 CUDA 时 51+2；**无 CUDA 时 53+0** | 缺前提，非错误 |
| C6 | STATUS:22「conda-env-lock.txt 115 行」 | 实际 124 行 = 9 行注释 + 115 个包（09 实测）；包名+版本与 pip freeze **完全一致**，但来源 `file://` 被改写 | 需订正行数 |
| C7 | STATUS §5「repro 168 passed」 | 有/无 GPU 均 **168 passed**（修前无 GPU 为 166+2 failed，已修，见 `test_phase1_regression.py`）；CI 用 `-k "not diff_scope"` 只选 166 | **已订正**：原判「缺前提」不成立 |
| C8 | 文档「视频 128×512 输入」的暗示 / 文件名 | 真实输入 **88×88 灰度**，512 是输出通道 | **易误读** |
| C9 | 旧文「Studio 80 ms guard = FRCNN 感受野」 | guard 是发射几何，挡窗尾瞬态；二者是巧合而非同一设计 | **归因错** |
| C10 | 「模型是 2 说话人分离（separation）」 | 是 **TSE 单输出**；PIT 恒等；n_src≥2 直接崩 | **概念需纠正** |
| C11 | FINAL_PROJECT_UNDERSTANDING「FRCNN train() 无 return self」 | HEAD 已 `return self`（`frcnn_videomodel.py:113-125`） | **已过时** |
| C12 | phase2 报告「9,169 条」 | SDRi/PESQ 实际分母 **9,166**（6 次 SDR singular + 3 次 PESQ nan，`nanmean` 静默跳过） | **分母错** |
| C13 | STATUS 引用的 full3 DNSMOS（2.7901/3.3285） | 已因窗口起点 bug 作废（`2062528`：`endpoint=LEFT+HOP`→`HOP+GUARD`） | 已作废，应从对外表格删除 |
| C14 | `docs/understanding/deepdive-20261024/00-SYNTHESIS.md:14`「论文本身仍未公开」 | 论文 2026-10-23 已公开（arXiv:2610.32749），同文件后面章节已改，§0 漏改 | **自相矛盾** |

---

## 7. 复现状态（数字可信度分级）

| 级别 | 内容 | 证据强度 |
|---|---|---|
| **A 可对外引用** | 537,697 参数、11,724,385 端到端、strict 加载、bit-identical（train↔studio）、死参数 5,568、RealSSA-Sim 数值 7.774/8.479 与 ch0 7.977/8.452（复现判定见 §0-10）、DNSMOS full4 2.7941/3.3365、LRS2 跨语料 7.029/7.713、LRS2 自训 ep124 8.858/9.270 | 有 commit 内 CSV/日志，可复算到 1e-7 dB |
| **B 有实测但口径特殊** | 整段 vs 分窗差 −1.6~−2.3 dB（协议模拟伪影）、CPU 单窗约 0.6 s（4 线程 592–694 ms；8 线程 357 ms）/ A800 47.1 ms 只有表格、1 epoch 覆盖 22% | 需同时声明口径 |
| **C 不可对外引用** | full3 四个 DNSMOS 值、把 7.029 与 14.1 并列成"复现"、A800 47.1/56.5 ms、任何"帧级因果/120 ms lookahead"表述 | 作废或口径不成立 |
| **D 不可复现** | LRS2 论文列 14.1/14.7、LRS3-2Mix 16.4/16.7、VoxCeleb2-2Mix 10.5/11.8、3090 延迟/RAM/显存、论文 6 组消融、论文"无时域 lookahead/120 ms" | 缺 ckpt / 缺数据 / 缺硬件 / 与代码矛盾 |

---

## 8. 仍未解的问题（按"值得投入"排序）

1. **LRS2 论文列 14.1**：唯一路径是 8×3090/A800 跑 `configs/LRS2_2Mix_Paper_v2.yml`（1000 ep），
   或向作者索取 LRS2-trained ckpt。当前最接近的是 8.858/9.270（单卡旧配方）。
2. **论文 tt 到底是 3,000 还是 6,000**：`falcon_lrs2_2mix_paper/tt` 索引 6000 行 = 3000 唯一 mix × {s1,s2}；
   历史 CSV 全是 3000（单向）。**所有"tt 3000"的分母都需要重核**。
3. **τ/margin 的真实作用**：K≡1 使 τ 退火无效、`relu(sim−0.5)` 恒正；要复现论文的对比学习收益，
   要复现论文的对比学习收益，就得按 App A.2 重建 K≥2 的 near 负样本（LibriSpeech + RIR）；本地 `rir/` 与
   `speech/` 已有素材。
4. **8 卡真实 DDP**：`find_unused_parameters=True` 的必需性已由 grad=None 证明，但条件化 all_gather /
   `sync_dist` / CSVLogger version 冲突只在单进程验过。
5. **RealSSA-Real 的窗口化损失**：DNSMOS 用 2 s 窗/0.5 s hop 拼，整段前向 20–153 s 会 OOM，
   混入多少损失未知。
6. **Falcon-Big 无法与训练侧交叉验证**：`SpeakerContrastiveV2`/`SISNRRepel` 只在 `studio/falcon_model/`。
7. **消融 0/6（论文 Table 7/8/9/18/19/21 一组都没做）**：TFRB、GCI、NASS、stages、big/small 只有部分；
   这是"论文级复现"最大的空洞，也是唯一不依赖 8 卡的部分（可在 2 卡上跑小配方）。
8. **`LN_c` 修正后质量未知**：把 `cLNhw` 换成论文式逐 (t,f) 归一化能把 lookahead 从窗尾降到 904 ms，
   但权重未重训 —— 需 A/B 重训才知道"因果性"与"质量"的真实 trade-off（论文的核心论点）。
9. **论文 15 帧与代码对称 padding 的 15 帧只是数值吻合**，无作者确认；论文"3 future visual frames"来源不明
   （代码 ±2，疑在上游 dataloader 的帧堆叠）。需要作者澄清或对上游 `439cbe9` 之前的历史做 diff。
10. **CI 修复**：`enable_checkpointing=False`/tmp 落盘、摘掉三个断言的 CUDA 前提、`fetch-depth`。
11. **`128_512` 命名来源**、face_alignment 版本与 LRW landmark 是否同源（低价值，纯考古）。

---

## 9. 报告导航

| # | 报告 | 一句话 |
|---|---|---|
| 01 | `01-model-core.md` | 主干逐层：加性融合、TSE、死参数 5,568、n_src=2 崩、窗内非因果、解码器 +16 ms |
| 02 | `02-layers-dsp.md` | layers/ 45 类只用 6 个；cLNhw 全窗耦合；VideoBlock LSTM 失效；mask 相位锁死 |
| 03 | `03-video-frontends.md` | 88×88 灰度（非 128×512）、FRCNN 11.19 M 冻结、视觉 15.9 GMAC、±80 ms 感受野 |
| 04 | `04-training-system.md` | 入口可跑；τ 空转、margin 失效；epoch 覆盖 22%；DDP find_unused 必需 |
| 05 | `05-losses-metrics.md` | −SNR vs −SI-SDR；threshold 丢好样本；MACs 含 LSTM 口径；tt 6000 疑问 |
| 06 | `06-data-pipeline.md` | mix 离线、SNR∈[0,10]、near=s2、idx 丢弃、硬链接 62 G、46.9 样本/s |
| 07 | `07-eval-benchmark.md` | 整段不切窗、TSE 口径、13 行数字逐条证据、5.3e-8 dB 复现、full3 作废 |
| 08 | `08-studio.md` | 单进程 FastAPI、无 pydantic、分块几何、guard 真义、51+2/8/8 |
| 09 | `09-space-project-page.md` | Space 复用 Studio 运行时、媒体 pipe 98.8%、i18n 226、CI 可过性 |
| 10 | `10-ops-repro-git.md` | 168 需 GPU、CI 大概率红、证据链最硬、upstream diff 396 文件 |
| 11 | `11-paper-alignment.md` | **论文逐条对齐**：9 个宽度全中、配方 15/16 行对上、因果/120 ms 与消融 0/6 不对 |
| 12 | `12-causality-latency.md` | **定论**：几何 904 ms、as-shipped 窗尾、跨窗 +14.71（稳态 +16.72）dB、guard 覆盖不足但只值 +0.8 dB |

---

## 10. 论文 ↔ 代码：逐条判定摘要（来自 11）

**真的对上了（可以放心引用）**
- **任务定义**：论文是 AVTSE **单目标提取**；"-2Mix" 只表示"两说话人混合、只提一路"（论文 §1 明确改词）。代码 `n_src:1` 一致。
- **9 个宽度全中**（Table 15）：Ce=112 / C=112 / Ch=64 / Cv=32 / dim(q)=64 / h_F=24 / h_T=72 / n_head=4 / L=4。
- **GCI 与 Eq.(6) 逐符号同构**；复数掩码与 Eq.(1) 逐符号同构；TFRB 三件套全部落地。
- **配方 16 行里 15 行对上**：−SI-SNR / λ_spk 1.0 / margin 0.5 / τ 0→1@1000ep / AdamW 1e-3+wd0.01 /
  ReduceLROnPlateau(15,0.5) / 早停 30 / 1000 ep / bs 4/GPU / 8 卡 / 2 s / 25 fps·50 帧 / Hann 256-128 /
  3 通道 / FRCNN 冻结 / seed 42。
- **RealSSA-Real 元数据几乎逐字命中**：54 段 / 5,039.4 s / 84.0 min（论文 54 / 5,035 s / 83:55）。

**对不上（论文说得比代码多）**

| 项 | 论文 | 代码事实 | 影响 |
|---|---|---|---|
| 时域 lookahead | §4.2/App E：「无时域 lookahead」「audio 最多 15 帧=120 ms」 | 对称 padding ⇒ 几何 113 帧=904 ms；as-shipped 窗尾 ≤2000 ms | **核心卖点不可复现** |
| `LN_c` | 「frame-local and cache-free」（Eq.2） | `cLNhw` = 全窗 `GroupNorm(1,C)`，30 处 | 同上 |
| "single encoder–decoder pass" | 摘要卖点 | `Separator` 类是 `repeats` 循环，`repeats:2/shared:true` ⇒ 同一 AudioBlock 跑两遍（融合前/后） | **2× MACs**，但论文 §4.1 也描述两次调用 → 措辞问题 |
| 视觉 future frames | 「3 future frames = 120 ms」 | `Conv3d(k=(5,7,7),pad=(2,3,3))` ⇒ ±2 帧 = 80 ms；系统级视觉 lookahead 实测 +13 视频帧=520 ms | 论文多算 1 帧 |
| 消融 | Table 7/8/9/18/19/21 六组 | 本地只做过 big/small 与 mouth（N=20），**TFRB/GCI/NASS/stages 一个都没做** | **消融 0/6** |
| near 负样本 | 1–5 近场干扰 | LRS2 用 s2（K=1，未乘 g2）；RealSSA 全 fallback（K=0） | 论文的对比学习收益无法复现 |

---

## 11. 因果性与延迟：最终定论（来自 12，Lead 采纳）

1. **一句话**：Falcon 是**窗口级因果（window-causal）**，不是流式因果；
   单窗内**几何** lookahead **113 帧 = 904 ms**；**发布权重因 30 处 cLNhw 而依赖到窗尾（≤2000 ms）**；
   **跨窗无状态**，换窗对齐 ±250 ms 只让整段一致性在 15.9–18.0 dB 间波动（任两种对齐互相 ~17 dB）。
2. **分解**：`STFT +1 帧样本级` → 编码器阶梯（对称 pad）**15 帧/120 ms** → 一遍 AudioBlock **+57 帧/456 ms**
   → 第二遍 **+56 帧** → **113 帧**。
3. **逐算子结论**：`DualPathRNN` 的 `causal` 参数**存了不用**；频率向 BiLSTM 的序列轴是频率 ⇒ **不算看未来**；
   tril attention 0；`MaskGenerator`/`STFTDecoder` 0 未来（解码器实测：样本 16000 只依赖 mask 帧 122–124）。
4. **延迟公式**：`L_e2e ≈ L_algo + GUARD(80 ms) + 块粒度(0…500 ms) + F + I/O`。
   A800 `F=47.1 ms` ⇒ 最好 **127 ms**、平均 ≈377 ms；CPU `F≈0.6 s`（4 线程 592–694 ms）⇒ **不可实时（RTF 1.18）**。
   **墙钟能到 120 ms 级，算法不能**——差 7.5×，且这个墙钟数字是靠"只喂 80 ms 未来"换来的。
5. **对外措辞要改的地方**：`STATUS.md:41` / `README.md:45` / `README_zh.md:45` 的"57 帧"要拆三档
   （单遍几何 57 / 整链几何 113 / as-shipped 窗长）；`README:43` "on a laptop" 需注 RTF 1.18；
   `project_page/index.html:7,12,18,339` 的 "streaming" → windowed；`:422` 删 "maintain causal streaming"；
   `:304` 的 51.3 ms 需注"per-window forward, 端到端 0.13–0.63 s"。
6. **一条可落地的"6 行修复"（W2-11 追加实测，Lead 认为这是本轮最有价值的行动项）**：
   论文 §4.2 说 "All convolutions use left-aligned padding along time … adding no temporal lookahead"，
   代码却有**三处硬编码 `causal=False`**——`Falcon.py:772`（downsample）、`:808`（fusion）、`:821`（concat），
   绕过了 `Falcon.py:1331` 注入的 `causal: true`，只有 `STFTEncoder` 是真 causal。
   把 30 个 `GlobalLayerNorm` 换成 Identity 后单遍仍是 **432 ms** ⇒ 说明这部分 lookahead **纯由卷积 padding 造成，
   与归一化无关**。因此把这三组改回 `causal=True`（约 6 行）后，几何 lookahead 应回落到编码器阶梯
   **15 帧 = 120 ms = 论文声明值**；再叠加"去掉/替换 cLNhw"才能同时满足论文的 frame-local 承诺。
   注意：这会改变数值，判断质量代价要重训 A/B（depth 消融：L=2/3/4/5 → 80/208/432/944 ms；
   两遍整模型 880 ms，与 12 的 904 ms 一致到约 3%）。

---

## 12. 变更记录

- 2026-11-08：本轮 10 个 Wave-1 域深挖 + 2 个 Wave-2 攻坚（论文对齐 / 因果性定论）完成并合成。
  Lead 复核了 ptflops 计数口径，更正了"10.6 G 是 FLOPs"与"视觉占 74.6%"两处误判（见 §0-3、§4）。
- 2026-11-08：00–06 七篇按"最新事实 + 中文写作规范"统一：MACs 两把尺子并列、lookahead 三档补齐、
  论文对齐事实（arXiv 已公开 / `causal=False` 三处 / `VideoBlock` LSTM 布局）落到各篇；重复背景改为引用，
  命令式表述改为陈述句加原因。07–12 未动。
