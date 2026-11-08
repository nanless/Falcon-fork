# 11 论文 ↔ 代码逐条对齐

> 子代理 W2-11 · 2026-11-08 · 远端 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork`（只读）
> 论文：*Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes*（arXiv:2610.32749v1, cs.SD, 23 Oct 2026, Sang/Kai Li/Yifan Li/Jianqiang Huang/Xiaolin Hu）
> 本文件只写「论文原文 ↔ 代码事实」的对齐证据。凡涉及"实测"均标注命令与输出；凡论文未写、代码无法自证者进 §7 GAP LIST。

---

## 0. 结论速览

1. **9/9 宽度命中**：Table 15 的 Ce=112 / C=112 / Ch=64 / Cv=32 / dim(q)=64 / h_F=24 / h_T=72 / n_head=4 / L=4 全部对上；GCI 与 Eq.(6)、复数掩码与 Eq.(1)、TFRB 三件套（频向双向 LSTM + 时向单向 LSTM + 2D attention）逐符号同构。
2. **16/16 命中、平均 13.23 人**：RealSSA-Sim test 的 9,169 条元数据逐格核 Table 11 的 near / far / RT60 / sudden 四场景 × 四列，16 格全中（§2.6）；Table 13 的平均说话人数 13.23 与实测 1 + 2.48 + 9.75 = **13.23** 两位小数一致。RealSSA-Real **54 段 / 5,039.4 s / 84.0 min**（论文 54 段 / 5,035 s / 83:55，差 0.09%）。这是本轮最强的数据侧证据。
3. **0.5 M / 10.6 G 口径闭合**：音频侧 **537,697 参数 / 10.611 GMAC**（ptflops、**含 LSTM**、**不含 FRCNN**），与论文 0.5 M / 10.6 G 同源同口径。视觉 88×88 输入是 **15.897 GMAC**，整窗 10.611 + 15.897 = **26.51 GMAC、视觉 60.0%**。
4. **5.382 GMAC 是另一把尺子（2026-11-08 更正）**：`torch.utils.flop_counter` 不给 `aten::lstm` 记账，音频侧只剩卷积 **5.381 GMAC**；5.381 + 15.897 = 21.28，于是出现「21.19 GMAC / 视觉 74.6%」这个数。它与论文的 10.6 G 不同口径，不能并列；整窗正确口径是 **26.51 GMAC / 60.0%**。
5. **2×MACs 被 "single pass" 掩盖**：`Separator` 是 SwiftNet 式的 `repeats` 循环（`Falcon.py:1032`），配置 `repeats: 2 / shared: true` 让它把**同一个 AudioBlock 跑两遍**（融合前 + 融合后），总 MACs = 2× 单块。论文 §4.1 确实写了 B_a 的两次调用，所以设计对得上；但摘要/§4.2 的 "single encoder–decoder pass" 只对每次 B_a 内部成立，读成「整模型一次前向」会漏掉一半 MACs（§3.4）。
6. **因果性对不上，差距 4–7 倍**：论文 §4.2 说 left-aligned padding、无时域 lookahead，代码 `AudioBlock` 的 downsample / fusion / concat 硬编码 `causal=False`（`Falcon.py:772,808,821`）。三档实测：**单遍几何 57 帧/456 ms**（本文 tail-perturb 得 54 帧/432 ms，另一路 57/456，差约 3%）、**整链几何 113 帧/904 ms**（本文 110 帧/880 ms，同 3%）、as-shipped 因 30 处 `cLNhw` 依赖到窗尾（2 s 窗时 **≤2000 ms**）。
7. **15 帧 / 120 ms 是论文声明，不是代码事实**：论文 §6.4 说音频支路「at most 15 frames」，App E 说视觉「three future frames = 120 ms」。代码里唯一等于 15 帧的量是编码器最深层的对称 padding 累计（`2^4−1`），它不含 GCI/decoder（+44 帧）也不含第二遍（+56 帧）；视觉前端实测 ±2 帧 = 80 ms。
8. **两个归一化实验不是同一件事**：本文 §2.3.1 把 30 个 `GlobalLayerNorm` 换成 `Identity` 后单遍仍是 432 ms ⇒ lookahead 由卷积 padding 造成；W2-12 §4.5 把 `cLNhw` 换成逐 (t,f) 的 `LN_c`（保留 affine）后，帧 0…120 变 −240 dB ⇒ 依赖下界回到 113 帧几何。前者问「纯几何有多长」，后者问「整窗统计还漏多少」，结论不冲突。
9. **16 行配方对上 15 行**：−SI-SNR / λ 1.0 / m 0.5 / τ 0→1@1000ep / AdamW 1e-3+wd 0.01 / ReduceLROnPlateau(15,0.5) / 早停 30 / 1000 ep / bs 4 / 8 卡 / 2 s / 25 fps·50 帧 / Hann 256-128 / 3 通道 / FRCNN 冻结 / seed 42。唯一实质缺口是 near 负样本语义（§4.4）。
10. **1 格数字到论文量级，但不是同 ckpt 对照**：RealSSA-Sim 官方 `falcon.pt` 得 **7.774 / 8.479**（取 `P2E-realssa-sim-metrics_combined.csv` 的 `avg` 行；论文 7.7 / 8.2，SI-SNRi 高 0.07、SDRi 高 0.28）。通道是 4ch 均值（论文 channel-1），ckpt 是 LRS3 训练（论文用 RealSSA-Sim 训练），两个轴都不同，属**数值吻合**，不构成严格复现；分母 9,169（SI-SNRi）/ 9,166（SDRi）。LRS2 14.1/14.7 本地拿不到（`falcon.pt` 训在 `LRS3_complex_nonmoving`）；LRS3 / VoxCeleb2 无数据；DNSMOS 3.276 属未发布的 RealSSA-trained ckpt。7.03 与 14.1 只能并列成「不可比」。
11. **0/6 消融复现**：论文 Table 7 / 8 / 9 / 18 / 19 / 21 六组，本地只有 P2B 的 big/small 与 mouth 消融，TFRB / GCI / NASS / stages 一个都没做过。

## 1. 证据与方法

### 1.1 论文来源与证据级别

| 项 | 值 |
|---|---|
| 来源 | `https://arxiv.org/html/2610.32749v1`（LaTeXML HTML 全量版，**L1 = 正文+公式+全部附录**） |
| 落地 | `/tmp/falcon_paper.html` 504,680 B；结构化转 `/tmp/falcon_struct.txt`（701 行 / 111,858 字符，597 个块） |
| 版本 | arXiv:2610.32749v1，`[cs.SD] 23 Oct 2026`（论文页脚自述） |
| 章节锚 | §1–§7 + App A–N（A 数据/ B 数据细节/ C 宽度/ D 配方/ E 指标与延迟/ F 基线因果化/ G 分场景/ I RNN 单元/ K 输出形式/ L 重叠率/ M 视觉缺失） |
| 表格锚 | Table 1–21（正文 3/4/5/6/7/8/9；附录 10–21） |

**HTML 转换已知噪声**（引用数字时已人工归一）：LaTeXML 会把数学串重复渲染，例如 `100{,}826100{,}826`、`−23-23 LUFS`、`7.77.7 dB`、`91.9%91.9\%`。本报告中此类数字统一取**第一次出现的值**。判定依据：同一段落的表格式行（`[row]` 分隔）给出干净值，与重复串的前半段一致。

**一处论文自身的量纲笔误**：Table 16 的 `STFT | Window 256, hop 128 (8 ms at 16 kHz), Hann` —— 256 样本 @16 kHz 是 **16 ms**，8 ms 是 hop 128。正文 §5.2 写的是 "window size 256 and hop 128 (8 ms hop at 16 kHz)"，正确。以 §5.2 为准。

### 1.2 本地代码基线

| 项 | 值 |
|---|---|
| HEAD | `6c1d5b7`（`STATUS：回填 HEAD=2062528 与 freeze tag 说明`） |
| 工作区 | `git status --porcelain` 空（干净，未被我改动） |
| 权重 | `/root/.cache/falcon_weights/falcon.pt`，`epoch: 714`，`architecture: FalconSpeakerContrastive` |
| 权重一致性 | 用 `configs/Falcon.yml` 建模型 → `load_state_dict(strict=True)` → **missing 0 / unexpected 0**（我自己跑的，非引用文档） |
| 关键配置 | `configs/Falcon.yml`（`evaluated_mac_params.py:39` 读它）、`configs/LRS2_2Mix_Paper_v2.yml`（Table 16 版） |
| 环境 | `/root/miniforge3/envs/falcon/bin/python`（torch 2.8.0+cu128, ptflops 可用） |

### 1.3 我实际做了什么（全部 CPU，`CUDA_VISIBLE_DEVICES=` 置空，未碰 GPU）

| # | 动作 | 目的 |
|---|---|---|
| M1 | `get_model_complexity_info` 测音频模型 MACs/params | 复现 10.611 G / 537,697 |
| M2 | `print_per_layer_stat=True` 逐模块分解 | 定位 LSTM 占比、解释 2× 口径差 |
| M3 | ptflops 测 FRCNN 在 64/88/96 三档分辨率（7.769 / 15.897 / 17.481 GMAC） | 解释「21.19 GMAC / 74.6%」这个口径（2026-11-08 更正：那是 `flop_counter` 口径，不是整窗正确值；整窗 26.51 GMAC / 60.0%） |
| M4 | 真实权重 + **尾部扰动**测 lookahead（音频/视频两路） | 独立给出因果性数字 |
| M5 | 把 `GlobalLayerNorm` 换成 `Identity` 后重测 | 把 lookahead 拆成"归一化 vs 卷积/注意力" |
| M6 | 逐层（encoder/bottleneck/separator/mask）定位 lookahead 引入点 | 归因 |
| M7 | 构造 `upsampling_depth ∈ {2,3,4,5}` 的 AudioBlock 测 lookahead | 直接对照论文 Table 7 与 §6.4 的"3 阶段上限" |
| M8 | 2 s 窗 vs 4 s 整段的输出互比（SI-SNR） | 量化整窗归一化的后果 |
| M9 | 重算 `P2E-*.csv` 的 SI-SNRi/SDRi 均值（SI-SNRi 9,169 条、SDRi 9,166 条） | 不采信文档，自己算 |
| M10 | 统计 `near.json`、`RealSSA-Real` 时长、mouth 张量形状 | 数据侧对齐 |

---

## 2. 逐项对齐表

> 判定记号：✅ 一致 ｜ ⚠️ 部分/口径差异 ｜ ❌ 冲突 ｜ ❓ 无法核对

### 2.1 任务定义（问题 1）

| 条目 | 论文原话（摘） | 代码事实 | 判定 | 证据 |
|---|---|---|---|---|
| 任务类型 | "audio-visual **target speaker extraction**…these methods all condition on the visual stream of a single target speaker, which by definition is target speaker extraction; therefore, we use the term AVTSE throughout this paper"（§1） | `n_src: 1`；`MaskGenerator` 输出 `n_src * in_chan` 通道；`decoder(in_chan = encoder_out_chan * n_src)` | ✅ | `configs/Falcon.yml`（`n_src: 1`）；`Falcon.py:1232`、`Falcon.py:1343-1345` |
| 混合模型 | `x = s + Σ_{k=1..K} b_k + f + n`，`b_k` 近场干扰、`f` 远场之和、`n` 环境噪声；无远场时 `f = 0`（§4.1） | 训练时只有 1 路 target + K 路 near；无 far/noise 项进入损失 | ⚠️ | 见 §4.4：near 只用于对比损失，重建损失只有单目标 |
| n_src 语义 | 单输出，`K` 是被抑制的近场干扰数，**不是**输出通道数 | `n_src=1` 输出 1 路；`K` 由 `near_audios[b]` 列表长度动态决定（0–4 分桶） | ✅ | `av_litmodule_nass_v2.py:115-132`、`:19-33`（`_NEAR_BUCKETS=[0..4]`） |
| SI-SNRi / SDRi 相对谁 | "both in dB, **relative to the unprocessed mixture**"（App E） | `look2hear/metrics/core.py` 标准 `si_snr(est,target) - si_snr(mix,target)`；CSV 里 `si-snr_i = si-snr(est) − si-snr(mix)` 代数成立（P2F 已用独立实现核到 4.8e-8 dB） | ✅ | `docs/verification/phase2/P2F-report.md` §1；`P2E-*.csv` 列名 |
| 训练损失 | `L_recon = −SI-SNR(ŝ,s)`（§4.3 末） | `sdr_type: pairwise_neg_sisdr`（v2 配置） | ✅ | `LRS2_2Mix_Paper_v2.yml:loss.train` |

**注**：`Falcon.yml`（老配置）里 `train.sdr_type: pairwise_neg_snr` + `threshold_byloss: true`，与论文 **不一致**；`LRS2_2Mix_Paper_v2.yml` 已改为 `pairwise_neg_sisdr` + `threshold_byloss: false`。**用哪个配置决定了是否对齐 Table 16**（§4.1）。

### 2.2 架构（问题 2）

| 条目 | 论文原话（摘） | 代码事实 | 判定 | 证据 |
|---|---|---|---|---|
| STFT 参数 | "window size 256 and hop 128"（§5.2）；"Hann"（Table 16） | `win: 256, hop_length: 128`；`torch.hann_window(self.win)` | ✅ | `Falcon.yml:enc_dec_params`；`Falcon.py:1100`（`register_buffer("window", torch.hann_window(self.win))`） |
| 编码器输入 | "concatenates the real part, imaginary part, and magnitude spectrum as a **three-channel** tensor"（§4.1） | `torch.stack([spec.real, spec.imag, this_spec], 1)` → Conv2d `in_chan=3` | ✅ 逐符号一致 | `Falcon.py:1122-1126` |
| Ce / C | Table 15：Ce=112、C=112 | `enc_dec_params.out_chan: 112`；`audio_bn_params.out_chan: 112` | ✅ | `Falcon.yml` |
| Ch | Table 15：64 | `audio_params.hid_chan: 64` | ✅ | 同上 |
| Cv | Table 15：32 | `video_params.hid_chan: 32` | ✅ | 同上 |
| dim(q) | Table 15：64 | `spk_embed_dim: 64` | ✅ | 同上 |
| h_F / h_T | Table 15：24 / 72 | `layer_1.hid_chan: 24`（dim=4，双向）/ `layer_2.hid_chan: 72`（dim=3，单向） | ✅ | 同上 |
| n_head | Table 15：4 | `layer_3.n_head: 4` | ✅ | 同上 |
| L（层数） | Table 15：4；"encoder has L=4 levels"（§4.2） | `upsampling_depth: 4` | ✅ | 同上 |
| 编码器下采样 | "D0 is a stride-1 depthwise 3×3; D1,D2,D3 are stride-2 depthwise 5×5"（§4.2） | `cur_stride = 1 if i==0 else 2`；`cur_kernel_size = 3 if i==0 else 5`；`groups=hid_chan`（depthwise） | ✅ 逐条一致 | `Falcon.py:760-762`、`Falcon.py:765-773` |
| TF 建模只在最深 | "Time-frequency modeling is applied **only at the bottleneck**"（Eq.4） | 编码器循环里 `if i == self.upsampling_depth - 1:` 才调 `process_{f,t,att}` | ✅ | `Falcon.py:844-848` |
| 4 个尺度一次前向 | "a single encoder–decoder pass therefore already yields four distinct TF scales"（§4.2） | 瓶颈级 3 一次 + 解码级 2/1/0 各一次 = 4 次 TFRB | ✅ | `Falcon.py:844-848` + `Falcon.py:870`, `:881-893` |
| TFRB 三段顺序 | `R_l(X)=A_l^2D(T_l(F_l^rnn(X)))`，Eq.(5)：先频向双向、再时向单向、最后 2D attention | `process_f_layers`(dim=4,bidirectional=True) → `process_t_layers`(dim=3,bidirectional=False) → `process_att_layers`（顺序在 `forward` 里逐行写死） | ✅ 顺序与方向都对 | `Falcon.py:774-800`、`Falcon.py:844-848` |
| 2D attention 形式 | "operates on the **flattened time-frequency plane** with a block-causal mask: a query at (t,f) attends to every key (t',f') with t' ≤ t"（§4.2） | 实际是**帧级** attention：`Q=[B',T,E*F]`，`attn=QKᵀ` 为 **T×T**，`torch.tril` 掩码；不是 (T·F)×(T·F) 的 block mask | ⚠️ 信息可达性等价、矩阵形状不等价（论文描述更像逐 (t,f) query；代码是每帧一个 E·F 维 query） | `Falcon.py:500-521`（`Q.transpose(1,2).flatten(start_dim=2)` → `[B',T,E*F]`；`mask=torch.tril`） |
| GCI（Eq.6） | `S̃_l = φ_loc(S_l) ⊙ σ(φ_gate(G)) + φ_glb(G)` | `injection_sum = local_emb * gate + global_emb`，三者都是 depthwise conv + LN_c，gate 用 Sigmoid | ✅ 逐符号一致 | `Falcon.py:329` |
| GCI 上采样 | "G is first upsampled to the resolution of S_l by **nearest-neighbor**" | `F.interpolate(..., mode="nearest")` | ✅ | `Falcon.py:320-325` |
| 解码器（Eq.7） | `Y_l = R_l(I^d_l(S̃_l, Y_{l+1}) + S_l)`，`+S_l` 是 identity residual | `expanded = self.concat_layers[i](x_fused[i], expanded) + downsampled_outputs[i]` 然后 `process_*[i]` | ✅ 连 `+S_l` 都在 | `Falcon.py:870`、`Falcon.py:881` |
| 输出 | `E = O(Y_0) + R`（残差分支 global skip） | `out = self.residual_conv(expanded) + residual`，`residual = self.gateway(x)`（depthwise 1×1 + PReLU） | ✅ | `Falcon.py:719-731`、`Falcon.py:896` |
| 复数掩码（Eq.1） | `Â=[M_r⊙A_r − M_i⊙A_i ; M_r⊙A_i + M_i⊙A_r]`，通道对半分 | `masks.view(B,n_src,2,C//2,T,F)`；`est_real=emb_r*m_r − emb_i*m_i`；`est_imag=emb_r*m_i + emb_i*m_r`；再 `cat` | ✅ 逐符号一致 | `Falcon.py:1251-1268` |
| 视觉前端 | "FRCNN visual backbone used in CTCNet, **frozen** during training"（§4.1） | `FRCNNVideoModel(backbone_type=resnet, pretrain=...backbone.pth.tar)`；训练时 `with torch.no_grad(): mouth_emb = self.video_model(...)` | ✅ | `Falcon.yml:videonet`；`av_litmodule_nass_v2.py:86-87` |
| VideoBlock | "the VideoBlock of SwiftNet, which aggregates temporal context with a **causal LSTM** and upsamples to the audio frame rate"（§4.1） | `nn.LSTM(input=32, hidden=32, bidirectional=False)`；上采样不在 VideoBlock 内，而在 `NeuroSCNetFusion` 里 `F.interpolate(size=T_a, mode="nearest")` | ⚠️ 功能等价，但"VideoBlock 内上采样"与代码位置不同（代码在 fusion 里） | `Falcon.py:952-957`、`Falcon.py:258-261` |
| 融合方式 | "Ṽ is expanded along the frequency axis and **added** to Ã"（§4.1） | `v_feat.unsqueeze(-1); Z = audio + v_feat`（broadcast 到 F） | ✅ | `Falcon.py:263-267` |
| Separator 单次性 | "in a **single decoder traversal**…replacing the multi-iteration recurrence of SwiftNet"（§4.2） | 类 `Separator` 本质是 `repeats` 循环（SwiftNet 血统）；但 `repeats: 2 / shared: true` ⇒ 只跑 2 次且是**融合前/融合后**两个不同位置，非迭代精化 | ⚠️ 结论等价、实现血缘不同（详见 §3.4） | `Falcon.py:1000-1005`、`Falcon.py:1032-1058` |

### 2.3 因果性与延迟（问题 3）

| 条目 | 论文原话（摘） | 代码事实 | 判定 | 证据 |
|---|---|---|---|---|
| 卷积 padding | "**All convolutions use left-aligned padding along time** and symmetric padding along frequency, **adding no temporal lookahead**"（§4.2） | 只有 `STFTEncoder` 是 causal（`causal: true` → `padding=(2,1)` 再裁尾）。`AudioBlock` 内部 **downsample / fusion / concat / gateway / projection / residual_conv 全部硬编码 `causal=False`** → 时域对称 padding | ❌ 直接矛盾 | `Falcon.py:772`、`:808`、`:821`（`causal=False,`）；对照 `Falcon.py:1084`（STFTEncoder 透传 `causal`）与 `Falcon.py:51,99-104`（causal 分支裁尾） |
| LN_c | "LN_c is **frame-local and cache-free**"；Eq.(2) 只在通道维求均值 `μ_{t,f}=(1/C_h)Σ_c Z_{c,t,f}`（§4.2） | 主卷积栈用的是 `cLNhw = LayerNormalizationHW`：**先 `GlobalLayerNorm`（= `nn.GroupNorm(1,C,H,W)`，对 C×T×F 全域统计）**，再 `CumulateLN4D`（对 H 累积）。前者是**整窗统计**，既不 frame-local 也不 cache-free | ❌ 直接矛盾 | `normalizations.py:204-220`（尤其 `:208`、`:216`）；`Falcon.yml` 里 `norm_type: cLNhw` 出现在 `audio_bn_params`、`audio_params`、`downsample_layers`、`InjectionMultiSum` 全部 |
| 视觉 lookahead | "At 25 fps, the visual stream buffers **three future frames**, corresponding to **120 ms** of lookahead"（App E；这是论文声明） | FRCNN 前端 `nn.Conv3d(1, nout, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3))` → 时间核 5、对称 pad 2 → **±2 帧 = 80 ms** | ⚠️ 同一机制、帧数不同（论文 3 帧，代码 2 帧） | `frcnn_videomodel.py:60-65` |
| 音频 lookahead 预算 | "the audio hop is 8 ms, leaving the audio branch **at most 15 frames** of lookahead; a fourth stage would exceed this budget"（§6.4）。这 15 帧/120 ms 是**论文声明** | 实测（见下表）单遍 **54 帧 = 432 ms**，整模型 **110 帧 = 880 ms**；另一路独立实测为 **57 帧/456 ms** 与 **113 帧/904 ms**，相差约 3% | ❌ 差 4–7 倍 | §2.3.1 实测 |
| 无额外缓冲 | "the configuration uses **no additional future-chunk buffer**"（App E） | 除上述卷积/归一化外，`torch.stft` 未传 `center=False`（默认 `center=True`）→ 首尾各补 `win//2=128` 样本，额外 +1 帧（8 ms）前瞻 | ⚠️ 再叠加一层 | `Falcon.py:1117-1123`（`torch.stft(x, n_fft=..., hop_length=..., window=...)`，无 `center` 参数） |
| 延迟/因果是"窗口级" | 论文通篇按"因果/streaming"叙述 | 结论：**窗口级因果 + 窗内全局非因果**。2 s 窗与 4 s 整段的输出差 **SI-SNR 16.03 dB**（同一权重、同一输入前缀） | ❌ | §2.3.2 实测 |

#### 2.3.1 lookahead 实测（M4–M7）

方法：固定真实权重，把输入**最后一个 STFT 帧**（末 128 样本 = 8 ms）替换为随机数，逐帧比较输出，取"最早被改变的输出帧"。`lookahead_frames = 250 − 最早改变帧`（hop=128 → 1 帧 = 8 ms）。

| 被测对象 | 最早改变帧 | lookahead | 说明 |
|---|---|---|---|
| `STFTEncoder` 单层 | 249 | **1 帧 = 8 ms** | `center=True` + 因果卷积，符合预期 |
| + `audio_bottleneck` | 249 | 1 帧 = 8 ms | 1×1 卷积，无新增 |
| + `Separator`（shared AudioBlock×2） | 140 | **110 帧 = 880 ms** | 引入点：AudioBlock |
| + `mask_generator` | 140 | 110 帧 = 880 ms | 1×1 卷积，无新增 |
| 单个 `AudioBlock`，depth=2 | 240 | **10 帧 = 80 ms** | 论文 Table 7 "1 stage"（0 个 stride-2 级） |
| 单个 `AudioBlock`，depth=3 | 224 | **26 帧 = 208 ms** | 论文 Table 7 "2 stages" |
| **单个 `AudioBlock`，depth=4（部署配置）** | 196 | **54 帧 = 432 ms** | 论文 Table 7 "3 stages (Ours)"；= 1 个 stride-1 + **3 个 stride-2** 级 |
| 单个 `AudioBlock`，depth=5 | 132 | **118 帧 = 944 ms** | 对应论文所说"第 **4** 个下采样阶段就越预算" |
| 整模型（2 次 AudioBlock）| 140 | **110 帧 = 880 ms** | ≈ 2×432 |
| 视频末帧扰动 → 音频输出 | 148（波形）/ 140（特征） | **≈ 800–880 ms** | 该数字与视频侧无关，是音频分支 lookahead 的传导 |
| 视频前端 `Conv3d` 本身 | — | **2 帧 = 80 ms** | `padding=(2,3,3)` |
| **对照：STATUS.md 记录** | — | 57 帧 ≈ 456 ms（单遍） | 与我 54 帧/432 ms 相差 3 帧，落在两路的 3% 差内；但 STATUS 未记录整链（我 110 帧/880 ms，另一路 113 帧/904 ms），读成「端到端 456 ms」会低估一倍 |

**口径提醒**：论文的 "stages" = **下采样阶段数**（Table 7 的 1/2/3），代码的 `upsampling_depth` = **层数**，两者差 1。`stages=3 (Ours)` ↔ `upsampling_depth=4`（L0 stride-1 + L1/L2/L3 各 stride-2）。

**归因**：把 30 个 `GlobalLayerNorm` 全换成 `Identity` 后重测，单遍仍为 432 ms ⇒ **lookahead 不是归一化造成的，而是卷积 padding 造成的**。数学上可精确解释：`downsample_layers`（k=3/k=5，对称 pad）在 L0/L1/L2/L3 分别贡献 1/2/2/2 帧，但按各层时间分辨率折算回输入帧率是 **1 + 4 + 8 + 16 = 29 帧**；`fusion_layers` 与 `concat_layers`（各 4 个 `InjectionMultiSum`，k=5 对称 pad 2）再各贡献 **2 + 4 + 8 + 16 = 30 帧**；编码侧 + 解码侧 ≈ 29×2 + 30 ≈ 88 帧，与实测 54（单遍，仅编码侧+fusion）/110（两遍）同量级。**改这三组卷积只修好 padding 这一条**：单遍几何 lookahead 会从 432 ms 降到编码器阶梯这一级（约 15 帧 / 120 ms，几何推算、未实测），不是归零；`cLNhw` 的整窗 `GroupNorm` 不在这三组卷积里（30 处），只改 padding 达不到论文 "frame-local and cache-free"（Eq.2）的声明，归一化要另改（§2.3.2）。

**两路实测相差约 3%**：本节表里的 54 帧/432 ms（单遍）与 110 帧/880 ms（整链）来自 tail-perturb 一路；W2-12 用算子隔离 + 梯度锥得到 57 帧/456 ms 与 113 帧/904 ms，解析值是 59f/472 ms 与 115f/920 ms。差异来自窗尾裁剪（ε≈2 帧）和扰动阈值，是同一个物理量的两条测量路径，不是两套结论。对外引用时用整链 113 帧/904 ms，并注明另一路差约 3%。

**本节与 W2-12 §4.5 是两个不同实验，不能读成互相矛盾**：

| 实验 | 换什么 | 保留什么 | 回答的问题 | 结果 |
|---|---|---|---|---|
| 本节 §2.3.1 | 30 个 `GlobalLayerNorm` → `Identity` | 去掉统计量与仿射 | 纯几何下前视还剩多长 | 单遍仍 **432 ms** ⇒ lookahead 由卷积 padding 造成 |
| W2-12 §4.5 | `cLNhw` → 逐 (t,f) `LN_c` | 保留学到的 affine | 整窗统计把依赖扩到多远 | 帧 0…120 变 **−240 dB** ⇒ 依赖下界回到 113 帧 |

两条证据各管一段：前者证明 padding 是几何主因，后者证明 `cLNhw` 是 as-shipped 整窗化的主因。W2-12 那次替换未重训，只能用于因果归因，不能评价质量。

#### 2.3.2 整窗归一化的后果（M8）

| 输入 | 输出 | 互比 |
|---|---|---|
| 4 s（64000 样本，`v4=[1,512,100]`）整段 | `y4` | 取 `y4[:32000]` 对比 |
| 2 s（32000 样本，`v=[1,512,50]`）单窗 | `y2` | **SI-SNR(y2, y4[:32000]) = 16.03 dB**，`max|Δ|=0.074`，`rms 差/信号 rms = 0.163` |

含义有三：
1. 论文的"streaming / causal / cache-free"在**部署形态上不可实现**：换窗长即换结果，无法增量推理；
2. `audio_visual_test.py` 的**整段一次前向**（`:23` 注释 "whole-utterance inference"）与 Studio / `eval_realssa_real.py` 的**2 s 窗**是两种不同数值协议，同一权重下不可互换；
3. 这也解释了 RealSSA-Real 为什么只能窗口化（整段前向会 O(T) 爆显存，见 `eval_realssa_real.py:383`）。

### 2.4 参数量 / MACs（问题 4）

| 条目 | 论文原话 | 代码/实测 | 判定 | 证据 |
|---|---|---|---|---|
| 参数量 | "Falcon and Swift-Net-12 both had **0.5 M** parameters"（§6.2 / Table 5） | 音频模型 **537,697**；四个口径：扣从不执行的 `concat_layers[2]`（5,568）= **532,129**（生效）、扣对比头（14,584）= **523,113**、两者都扣（分离前向）= **517,545** | ✅（0.538 M ≈ 0.5 M） | M1 实测；`Falcon_Contrastive.py:99`（`spk_pool_proj`） |
| MACs | "**10.6 G** MACs"（§6.2 / Table 5） | **10.6109 G**（ptflops） | ✅ 精确到 3 位 | M1 实测 |
| 口径：是否含视觉 | 论文未明说 | **不含**。`evaluated_mac_params.py` 先 `v = video_model(v)` 在计数**之外**算好，再把 `v` 作为输入喂给音频模型 | ✅ 音频侧 | `evaluated_mac_params.py:80-96` |
| 口径：是否含 LSTM | 论文未明说 | **含**。ptflops 的 `rnn_flops` 对 `nn.LSTM` 逐 time-step 记账；LSTM 合计 ≈ **5.02 GMAC（47%）** | ✅ 含（不含就凑不出 10.6） | `ptflops/pytorch_ops.py:105-127`；M2 分解 |
| 视觉实际成本 | 论文 Table 5 未单列 | FRCNN **64²: 7.769 / 88²: 15.897 / 96²: 17.481 GMAC**（50 帧） | — | M3 实测 |
| 整窗总成本（正确口径） | — | 10.611 + 15.897 = **26.51 GMAC**（视觉 **60.0%**） | — | M1+M3 |
| 与"21.19 G / 74.6%"的关系（2026-11-08 更正） | — | 该数 = **`torch.utils.flop_counter`** 口径：音频只剩卷积 **5.381 GMAC**（我实测 FlopCounter 给出 10.762 GFLOP ⇒ 5.381 GMAC，恰好是 ptflops 的一半，差的正是 LSTM），视觉 15.897 ⇒ 5.381+15.897 = **21.28 ≈ 21.19**，视觉占 **74.7% ≈ 74.6%** | ✅ 两个数都对，**口径不同** | M3 实测 + `FlopCounterMode` |

**结论（写报告时带的口径句）**：论文 0.5 M / 10.6 G = **音频侧、ptflops、含 LSTM、不含冻结 FRCNN、2 s/50 帧输入**。引用「21.19 G / 74.6%」时要同时说明它是 `flop_counter` 口径（LSTM 记账为 0，音频只剩 5.381 GMAC），与 10.6 G 并列会自相矛盾。2026-11-08 更正：整窗正确口径是 **26.51 GMAC、视觉 60.0%**。

MACs 分布（M2，占 10.611 G 的比例）：

| 子模块 | MACs | 占比 | 参数 |
|---|---|---|---|
| `separator.audio_net`（AudioBlock，shared×2） | 8.05 G | 75.9% | 362.95 k |
| └ `process_t_layers[0]`（时向 LSTM 72 隐） | 2.92 G | 27.5% | 44.35 k |
| └ `process_f_layers[0]`（频向双向 LSTM 24） | 1.35 G | 12.7% | 20.35 k |
| └ `process_att_layers[0]`（2D attention） | 0.684 G | 6.4% | 10.41 k |
| └ 4× `downsample_layers` | 0.089 G | 0.84% | 6.13 k |
| └ 4× `fusion_layers` + 4× `concat_layers` | 0.387 G | 3.6% | 41.4 k（含死参数 5.57 k） |
| `audio_bottleneck`（1×1 112→112 + LN） | 0.421 G | 4.0% | 12.88 k |
| `mask_generator`（1×1 112→112） | 0.417 G | 3.9% | 12.66 k |
| `encoder`（3×3 3→112） | 0.099 G | 0.9% | 3.02 k |
| `decoder`（ConvTranspose 112→2） | 0.066 G | 0.6% | 2.02 k |
| `video_net` + `caf`（视觉侧，不含 FRCNN） | 0.0022 G | 0.02% | 46.55 k |

**反直觉点**：10.6 G 里 **47% 是 LSTM**，而论文把卖点讲成"multi-scale TF modeling"；卷积（含下采样/融合）合计不到 6%。

### 2.5 训练配方（问题 5，详见 §4）

| 条目 | 论文（Table 16 / App D） | 代码（`LRS2_2Mix_Paper_v2.yml`） | 判定 |
|---|---|---|---|
| 优化器 | AdamW, lr 1e-3, wd 0.01 | `adamw / 0.001 / 0.01` | ✅ |
| 调度 | ReduceLROnPlateau, factor 0.5, patience 15 | `ReduceLROnPlateau / patience 15 / factor 0.5` | ✅ |
| 早停 | patience 30 on val loss | `patience: 30`, `monitor: val_loss/dataloader_idx_0` | ✅ |
| 最大 epoch | 1000 | `epochs: 1000` | ✅（但发布 ckpt 只到 **714**，见 §7 GAP-10） |
| batch | 4 / GPU | `batch_size: 4` | ✅ |
| 硬件 | 8× RTX 3090 | `gpus: -1`（配合 8 卡启动脚本 `NPROC=8`） | ✅ |
| 段长 | 2 s @16 kHz | `segment: 2.0`, `sample_rate: 16000` | ✅ |
| 视频 | 25 fps, Tv=50 | 数据侧 50 帧 `.pt`；`fps_len` 由 2 s×25 得出 | ✅ |
| mouth ROI | 96×96 grayscale | 存 **96×96×3 → Grayscale(1)** → **CenterCrop(88)** ⇒ 网络见 **88×88** | ⚠️ 见 §4 行 10 |
| STFT | 256/128, Hann | `win 256 / hop 128` + `hann_window` | ✅ |
| 编码器输入 | real+imag+mag 3 通道 | 同 | ✅ |
| 视觉骨干 | FRCNN（CTCNet），冻结，唇读预训练 | `FRCNNVideoModel` + `pretrain: frcnn_128_512.backbone.pth.tar` + `no_grad` | ✅ |
| 重建损失 | −SI-SNR | `pairwise_neg_sisdr`, `threshold_byloss: false` | ✅ |
| λ_spk | 1.0 | `spk_contrast.loss_weight: 1.0`（→ `contrastive_weight: 1.0`） | ✅ |
| margin m | 0.5 | `margin: 0.5` | ✅ |
| τ 退火 | 线性 0→1，前 1000 epoch，之后固定 1 | `beta_init 0.0 / beta_final 1.0 / beta_anneal_epochs 1000` | ✅ |
| seed | 42 | `seed: 42` | ✅ |
| near 负样本 | 每个近场干扰 `b_k`（含房间响应） | LRS2：K=1、取另一路源 wav；RealSSA 本地：**K=0** | ⚠️/❌ 见 §4.4 |

### 2.6 RealSSA 数据构造（问题 6）

> **重要更正**：`tt/mix/` 下除了 `<id>.wav` 还有 **`<id>_meta.json`**（9,169 个）。我第一次只数了 `.wav`，误判为"发布包无场景元数据"。元数据**齐全**，且与论文 Table 11/13 逐格吻合 —— 这是本轮最强的数据侧证据。

**逐格核对 Table 11（9,169 条 test 元数据实算）**

| 场景 | 论文 near 数 | 实测 near | 论文 far 数 | 实测 far | 论文 RT60 | 实测 RT60 | 论文 sudden | 实测 sudden | env 分量均值 |
|---|---|---|---|---|---|---|---|---|---|
| Small meeting | 1–5 | **1–5** ✅ | none | **0** ✅ | 0.3–0.6 s | **0.30–0.60** ✅ | 30%, 1 event | **30%** ✅ | 1.00 |
| Medium indoor | 1–3 | **1–3** ✅ | 10–15 | **10–15** ✅ | 0.5–0.9 s | **0.50–0.90** ✅ | 40%, 1–2 | **39%** ✅ | 2.00 |
| Large public | 1–4 | **1–4** ✅ | 10–15 | **10–15** ✅ | 1.0–2.0 s | **1.00–2.00** ✅ | 50%, 1–3 | **51%** ✅ | 2.00 |
| Outdoor | 1–4 | **1–4** ✅ | 10–15 | **10–15** ✅ | 0.05–0.1 s | **0.05–0.10** ✅ | 40%, 1–3 | **41%** ✅ | 2.00 |

**16/16 格全部命中**（`near`/`far` 取自 `components[].type`，`RT60` 取自 `room.t60`，`sudden` 为含 `type=="sudden"` 分量的样本占比）。

**整体统计核对 Table 13（test 行）**

| 指标 | 论文 Table 13（Test） | 本地实算（9,169 条） | 判定 |
|---|---|---|---|
| 样本数 | 9,169 | **9,169** | ✅ |
| 场景均衡 | "each accounting for nearly one quarter" | large 2,435 / medium 2,369 / outdoor 2,347 / **small 2,018** | ✅（small 略少 22%，因它对 near 上限更宽） |
| 近场干扰均值 | "average of **2.5**" | **2.48** | ✅ |
| 平均说话人数 | **13.23** | 1（target）+ 2.48（near）+ 9.75（far）= **13.23** | ✅✅ 两位小数完全一致 |
| sudden 噪声比例 | 30/40/50/40% 加权 ⇒ ≈40.5% | **41%**（3,728/9,169） | ✅ |
| env 噪声流 | "one WHAM! + one Hive ambient stream"（Algorithm 1 第 8 步） | far 场景 **2 条** ✅；small meeting **1 条** | ⚠️ small 场景少一条（论文未说明例外） |
| 分量总数 | — | near 22,716 / far 89,427 / env 16,320 / sudden 6,417 | — |

**其余条目**

| 条目 | 论文（§3 / App A / Table 2,11,13,14） | 本地实际 | 判定 | 证据 |
|---|---|---|---|---|
| Sim 规模 | train 100,826 / val 9,133 / test 9,169；≈153 h | **只有 tt**（9,169 mix + 9,169 target + 9,169 mouths + 9,169 meta） | ⚠️ 仅 test 可核，数量 ✅ 精确 | `ls .../tt/mix/*.wav` = 9,169；`target` = 9,169 |
| near 波形 | "each near-field interferer b_k…captured at the microphone" | `near.json` 9,169 条**全为自引用 fallback**；元数据 `provenance.note` 明写 **"RealSSA-Sim does not ship near-speaker waveforms"** | ✅ 发布策略（非 bug）/ ⚠️ 训练语义受影响 | `<id>_meta.json:provenance.note`；`near.json` K hist `{1: 9169}` 且路径 == mix |
| far 分量 | 10–15 个 LibriSpeech 干扰 | 元数据有 **路径 + sir_db（20.4–33.1 dB）**，但**无波形** | ✅ 元数据在 / ⚠️ 波形不在 | `components[].type=="far"` |
| RIR | gpuRIR + Sabine，4 元线性阵列 2.8 cm，只用 ch1 | 元数据有 `room.dimensions / mic_center / t60`、`target_position.start/end`（`moving:false`）；**无 RIR 文件/hash**；本地 `/root/data/rir` 是 PSE 生成的另一套，与 Sim 无关 | ✅ 几何元数据 / ❓ RIR hash 缺 | `_meta.json:room`；论文 §3 称"RIR hashes" |
| 场景元数据 | 论文 §3 末："released metadata records the sampled room geometry, RT60, per-source positions, RIR hashes, SIR values, and Hive label identifiers" | **有** geometry/RT60/target position/**每源 SIR**；**缺** per-source position（`position: null`）、RIR hash、Hive label id | ⚠️ 4/6 有 | `components[].position` 全 null |
| RIR 口径 | 只用 channel-1 | `falcon_realssa_sim_ch0`（`channel_policy: "0"`）✅；另有 `falcon_realssa_sim`（mean） | ✅ | `_meta.json:provenance.channel_policy` |
| 采样率/通道 | 单通道 16 kHz | `provenance.audio = "16 kHz mono PCM16"`；mouth "uint8 RGB [T,96,96,3]"，`aligned_frames 145`，`samples_per_frame 640` | ✅ | 同上 |
| 响度/峰限/增益裁 | −23 LUFS、−1 dBFS、[−40,+40] dB | 无生成脚本，无法验证 | ❓ | — |
| 随机种子 | 主种子 42，split 用不相交偏移 | 只有 tt，无法验证 | ❓ | — |
| 突发噪声语料 | **Hive** [30]（44.1 kHz，544 h） | 元数据路径显示 **`USS-Qwen/data/BBC_Sound_Effects_cut_gpfs/...flac`**（BBC Sound Effects），不是 Hive | ❓ 语料名不一致（见 GAP-8） | `<id>_meta.json` 的 `type=="sudden"` 路径 |
| 环境噪声语料 | WHAM!（连续）+ Hive（环境） | env 路径 = `wham_noise/tr/*.wav` + `BBC_Sound_Effects_cut_gpfs/*.flac` | ⚠️ 同 GAP-8 | 同上 |
| Real 规模 | 54 sessions / 5,035 s / 83:55 / 8 环境 / 10 被试 | **54 段 / 5,039.4 s / 84.0 min** | ✅ 逐项命中（+4.4 s，0.09%） | M10：`sf.info` 累加 54 个 `audio/inference/**/*.wav` |
| Real 采集 | iPhone 17 Pro，60 fps/48 kHz → 16 kHz/25 fps，不降噪/不归一 | 发布包 `audio/`（16 kHz）+ `faces/`+`landmarks/`+`mouth/` | ✅ 格式一致 | `RealSSA-Real/INFERENCE.md`、目录 |
| Real 无干净参考 | "no isolated target waveforms" | 目录确无 target；`eval_realssa_real.py` 明确只出 DNSMOS | ✅ | `eval_realssa_real.py:10` |
| 场景映射 | Table 14：study room/classroom→small；internet cafe/billiard→medium；mall/supermarket→large；street×2→outdoor | `metadata/inference_segments.jsonl`（本次未逐条核 8 环境分布） | ⚠️ 结构在，分布未核 | `RealSSA-Real/metadata/inference_segments.jsonl` |

### 2.7 评测协议（问题 7）

| 条目 | 论文 | 本地 | 判定 | 证据 |
|---|---|---|---|---|
| 输入是整段还是窗 | App B：公开 benchmark"每个 utterance 取固定 2 s clip，短补零、长随机裁（训练）"；RealSSA-Real 未规定窗长 | `audio_visual_test.py`：**整段一次前向**，batch=1（`:23` 注释）；`eval_realssa_real.py`：**2 s 窗 / 0.5 s hop / 80 ms guard** | ⚠️ 两套协议并存，论文未规定 RealSSA-Real 的窗长 ⇒ 无法判定谁"对" | `audio_visual_test.py:23,419-432`；`eval_realssa_real.py:399-400` |
| 窗调度 | — | `WINDOW=32000, HOP=8000, GUARD=1280, LEFT=22720`；取 `pred[22720:30720]`，映射到绝对 `[endpoint−9280, endpoint−1280)`；首窗 `endpoint=HOP+GUARD=9280`；有 100% 覆盖断言 | — | `eval_realssa_real.py:399-455` |
| guard 是否覆盖 lookahead | 论文认为 120 ms（论文声明）就够 | 实测整链 lookahead **880–904 ms**，guard **80 ms** 只覆盖约 9%；被取用的 `pred[22720:30720]`（窗内 1.42–1.92 s）实际依赖到窗尾 2.0 s，即**隐含 80–580 ms 未申报延迟** | ❌ | §2.3.1 + `eval_realssa_real.py:399` 算术 |
| DNSMOS 版本 | "DNSMOS [35]"（未写 P.835/P.808，只给引用号） | `DEFAULT_MODEL_TYPE="dnsmos"` = speechmos 的**标准 P.835**（`sig_bak_ovr.onnx`），同时并行输出 **P.808**（`model_v8.onnx`）；9.01 s / 1 s hop 分段取均值 | ⚠️ 论文没写版本，本地按 P.835 报 OVRL + 另附 P808 | `eval_dnsmos.py:36,68-89`；`eval_realssa_real.py:4`（"reference-free: DNSMOS P.835"） |
| 参考 ckpt | Table 3 的 3.276 属**RealSSA-Sim 训练的 ckpt** | 官方 `falcon.pt` 是 **LRS3_complex_nonmoving** 训练的（P2F 由 `Experiments/checkpoint/Falcon/conf.yml:99-102` + epoch 714 锁定）⇒ 用官方权重测 RealSSA-Real **必然**低于 3.276 | ✅（可解释） | `docs/verification/phase2/P2F-report.md` §1.4（既有证据，我未复核 conf.yml，标注为引用） |
| WER | Mega-ASR [37] | 未实现（本地无 Mega-ASR 环境） | ❓ | STATUS §6 待办 |
| speaker similarity | WeSpeaker ResNet34-LM（HF `wespeaker-voxceleb-resnet34-LM`），独立 enrollment | 未实现 | ❓ | STATUS §6 待办 |
| NISQA | [36]（Table 6） | 未实现 | ❓ | — |
| 效率测量协议 | 2 s / 16 kHz + 50 帧；3090 + Xeon 6226R；1 线程；5 次热身取后 50 次中位数；CUDA events | 脚本 `evaluated_mac_params.py` 输入 **2 s（32000）+ 50 帧**但帧尺寸用 **64×64**（`:81` `torch.randn(1,1,50,64,64)`），非 88/96 | ⚠️ MACs 不受影响（v 是 FRCNN 输出），但复现脚本本身与论文协议不完全同 | `evaluated_mac_params.py:80-81` |

### 2.8 表格数字（问题 8，逐格见 §5）

### 2.9 消融（问题 9）

| 论文消融 | 表 | 本地是否做过 | 证据 |
|---|---|---|---|
| 下采样阶段数 1/2/3 | Table 7 | **否**（我用随机权重只测了 lookahead，不是 SI-SNRi） | 本轮 M7 仅测因果性 |
| 编码器空间对比损失（Falcon/SwiftNet/Dolphin ±） | Table 8 | **部分**：P2D 的"+s2 near 对比微调"ep124 8.858/9.270 vs ep99 8.726/9.122 ⇒ **+0.13/+0.15 dB**，但 τ 退火未生效（200 epoch 配置 + 仅 124 epoch） | `P2D-eval-epoch124-contrastive.csv`、STATUS §4.5 |
| τ 调度（0 / 1 / →2 / →1） | Table 9 | **否** | — |
| TFRB 循环单元（RNN/SRU/GRU/LSTM） | Table 18 | **否** | — |
| 输出形式（复数/实数/直接 RI） | Table 19 | **否** | — |
| 视觉缺失（Random/Block drop） | Table 21 | **部分**：P2B 有 mouth 消融（`P2B-b300-small-whole-zero.csv`/`-shuffle.csv` = 视觉置零/打乱） | `docs/verification/phase2/P2B-*.csv` |
| 模型规模（small vs big） | 论文**无**此表 | 本地有：big−small = +0.97/+0.91 dB（300 条配对） | `P2B-summary-b300-big.json` |
| 重叠率分组 | Table 20（不是消融，是分析） | **否** | — |
| 分场景/分说话人数 | Table 17（App G） | **否**（需 per-sample 场景标签，发布包没有 ⇒ 见 §2.6） | — |

---

## 3. 架构/张量级对齐细节

### 3.1 一次真实前向的形状链（2 s 输入）

| 阶段 | 形状 | 代码位置 |
|---|---|---|
| 波形 | `[B, 32000]` | — |
| STFT | `[B, 251, 129]` 复数（`n_fft=256, hop=128, center=True`） | `Falcon.py:1117` |
| 3 通道堆叠 | `[B, 3, 251, 129]`（real/imag/mag，`transpose(2,3)` 后） | `Falcon.py:1125` |
| 编码器 Conv2d(3→112, k=3, pad=(2,1), causal) | `[B, 112, 251, 129]` = `A_0` | `Falcon.py:1126` |
| `audio_bottleneck`（1×1 112→112 + cLNhw） | `[B, 112, 251, 129]` = `A` | `Falcon.py:1330` |
| 视觉：`[B,1,50,96,96]` → Grayscale→Crop(88) → FRCNN | `[B, 512, 50]` = `V_0` | `frcnn_videomodel.py:77-88`；`avspeech_dataset_near.py:81` |
| `video_bottleneck`（k=-1 ⇒ Identity，0 参数） | `[B, 512, 50]` | `Falcon.yml:video_bn_params.kernel_size: -1` |
| `AudioBlock`#1（融合前精化） | `[B, 112, 251, 129]` | `Falcon.py:1038` |
| `VideoBlock`（1×1→32, LSTM 32, 1×1→512） | `[B, 512, 50]` | `Falcon.py:966-975` |
| `NeuroSCNetFusion`：`v_proj` Conv1d(512→112, groups=16) + `interpolate(50→251, nearest)` + `unsqueeze(-1)` + `+` | `[B, 112, 251, 129]` = `F` | `Falcon.py:250-267` |
| `AudioBlock`#2（= separator，融合后） | `[B, 112, 251, 129]` = `E` | `Falcon.py:1039` |
| `MaskGenerator`：PReLU → Conv1×1(112→112) → `view(B,1,2,56,...)` | `M: [B,1,112,251,129]` | `Falcon.py:1242-1244`、`:1254` |
| 复数掩码 | `Â: [B,1,112,251,129]` | `Falcon.py:1263-1267` |
| `STFTDecoder`：ConvTranspose2d(112→2,k=3) → iSTFT | `[B,1,32000]` | `Falcon.py:1180-1198` |
| 训练时额外：`spk_pool_proj` | `q: [B,64]`（L2 归一化） | `Falcon_Contrastive.py:120-131` |

### 3.2 `AudioBlock` 内部数据流（这是论文 Fig.3 的实现）

```
x [B,112,T,F]
├─ residual = gateway(x)                         # depthwise 1×1 + PReLU，causal=False（k=1，无影响）
├─ x_enc   = projection(residual)                # 1×1 112→64 + cLNhw
├─ 编码（4 级）
│   L0: Conv(64→64, k=3, s=1, dw, pad=1)         # 对称 padding ← lookahead +1 帧
│   L1: Conv(64→64, k=5, s=2, dw, pad=2)         # +2 帧@T/2
│   L2: Conv(64→64, k=5, s=2, dw, pad=2)         # +2 帧@T/4
│   L3: Conv(64→64, k=5, s=2, dw, pad=2)         # +2 帧@T/8
│   ★ 只有 L3 之后才跑 TFRB：频向双向 LSTM → 时向单向 LSTM → 2D causal attention
├─ GCI：S̃_l = InjectionMultiSum(S_l, G=S_3)      # 4 个，k=5 对称 pad ← 再 +2 帧/级
├─ 解码
│   expanded = concat[3](S̃_2, S̃_3) + S_2 ; TFRB@L2
│   for i in [1,0]: expanded = concat[i](S̃_i, expanded) + S_i ; TFRB@L_i
│   ★ concat[2] 从未被调用（死参数 5,568）
├─ expanded = F.interpolate(expanded, size=(T,F), nearest)   # L0 已是 (T,F)，实为 no-op
└─ out = residual_conv(expanded) + residual      # 1×1 64→112
```

### 3.3 TFRB 三段的实现细节与论文措辞的偏差

| 段 | 论文 | 代码 | 偏差 |
|---|---|---|---|
| 频向 LSTM | "bidirectional LSTM along frequency" | `DualPathRNN(dim=4)`：`dim==4` 时先 `transpose(-2,-1)`，再把 F 折进 batch 维度跑 `nn.LSTM(bidirectional=True)` | ✅ |
| 时向 LSTM | "unidirectional LSTM along time" | `DualPathRNN(dim=3)`，`bidirectional: false` | ✅（注意：`causal` 参数在 `DualPathRNN.forward` 里**完全没被使用**，因果性由 `bidirectional=False` 保证；`AudioBlock:783` 传进来的 `causal` 是死参数） |
| 2D attention | 在 flatten 的 TF 平面上 block-causal | 帧级 `T×T` attention + `torch.tril` | ⚠️ 见 §2.2 |
| 残差包裹 | "Each stage is wrapped in a pre-norm residual block with LN_c" | `DualPathRNN` 内 `norm → rnn → out_proj → +residual`；`MultiHeadSelfAttention2D` 内 `+residual` | ✅ |
| LN 类型 | LN_c（通道向、frame-local） | `LayerNormalization4D`（attention/RNN 内）+ `cLNhw`（卷积栈，含整窗 GN） | ❌ 卷积栈那条不符 |

### 3.4 `Separator` 的 repeats 语义（"single pass" 到底几 pass）

- `Separator.__init__`（`Falcon.py:1000-1005`）：
  `fusion_repeats = video_params.repeats`（=1）；`audio_repeats = audio_params.repeats − fusion_repeats`（=2−1=**1**）。
- `Separator.forward`（`Falcon.py:1032-1058`）：`for i in range(fusion_repeats)` 里做 **audio_net → video_net → caf**；再 `for j in range(audio_repeats)` 里只做 **audio_net**。
- 因 `shared: true`，两个循环用的是**同一个** `AudioBlock` 实例 ⇒ **总 MACs = 2× 单块**。§2.4 的 10.611 GMAC 是这两遍的总和，不是一遍。
- 论文 §4.1 确实写了两次 B_a（"audio feature A is processed by an AudioBlock B_a… yielding Ã"，"the fused feature F is processed by the separator B_a to obtain E"）⇒ **两次调用是论文的设计，不是冗余**。
- 但论文 §4.2/§6.4 的"single encoder–decoder pass"只对**每次 B_a 内部**成立。写论文/写报告时若说"整个模型一次前向"，会漏掉一半 MACs。

### 3.5 论文没提、但影响数值的三处实现

1. **`concat_layers[2]` 从不执行**（`Falcon.py:870` 与 `:881-893` 只走 `[-1]` 与 `[1,0]`）：5,568 参数（1.04%）是死重，`get_model_complexity_info` 也如实报了 0 MAC。
2. **`video_bottleneck` 是 Identity**（`kernel_size: -1` ⇒ `ConvNormAct` 里 `nn.Identity()`），论文 §4.1 "maps through a **visual bottleneck**" 在代码里是空操作。
3. **`torch.stft` 用默认 `center=True`**：首尾各补 128 样本 ⇒ 额外 1 帧（8 ms）前瞻，且使"整段/窗口"两种推理的边缘帧处理不同。

---

## 4. 训练配方对齐细节（Table 16 逐行）

> 对照对象：`configs/LRS2_2Mix_Paper_v2.yml`（2026-10-30 起为论文配方版）；老版 `configs/LRS2_2Mix_Paper_8gpu.yml` 保留作历史。

| # | Table 16 行 | 论文值 | v2 配置值 | 判定 | 备注 |
|---|---|---|---|---|---|
| 1 | Optimizer | AdamW, lr 1e-3, wd 0.01 | `optim_name: adamw / lr: 0.001 / weight_decay: 0.01` | ✅ | `optimizers.py` 里 AdamW 分支；`Falcon.yml` 同 |
| 2 | LR scheduler | ReduceLROnPlateau, factor 0.5, patience 15 | `sche_name: ReduceLROnPlateau / patience: 15 / factor: 0.5` | ✅ | `schedulers.py` |
| 3 | Early stopping | patience 30 on val loss | `monitor: val_loss/dataloader_idx_0 / mode: min / patience: 30` | ✅ | |
| 4 | Max epochs | 1000 | `epochs: 1000` | ✅ | 发布权重的 `epoch: 714` 说明实际未跑满（GAP-10） |
| 5 | Batch size per GPU | 4 | `batch_size: 4` | ✅ | global = 4×8 = 32 |
| 6 | Hardware | 8× RTX 3090 | `gpus: -1` + `tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh`（NPROC=8） | ✅ | 本地实际只在 2×A800 上跑过 100 ep |
| 7 | Audio segment | 2 s @16 kHz | `segment: 2.0 / sample_rate: 16000` | ✅ | |
| 8 | Video frame rate | 25 fps, Tv=50 | 数据侧 50 帧（`.pt` 形状实测 `[50,96,96,3]`） | ✅ | |
| 9 | Mouth ROI | 96×96 grayscale | 存 96×96×3 → `transforms.Grayscale(1)`（`:81`）→ `Normalize/255 → RandomCrop/CenterCrop(88)` → `mean .421/std .165` | ⚠️ **网络实际见 88×88**；"96×96"指存盘 ROI | `transform.py:147` 默认 `Crop_size=(88,88)`；`avspeech_dataset_near.py:81` |
| 10 | STFT | Window 256, hop 128, Hann | `win: 256 / hop_length: 128` + `hann_window` | ✅ | 论文该行 "(8 ms at 16 kHz)" 是 hop 的量纲，window 应为 16 ms |
| 11 | Encoder input | Real/Imag/Mag（3 通道） | `torch.stack([real, imag, mag], 1)` | ✅ | |
| 12 | Visual backbone | FRCNN（CTCNet），冻结，唇读预训练 | `FRCNNVideoModel` + `pretrain: /root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar`；训练 `no_grad`；`train()` 里强制 BN eval | ✅ | `frcnn_videomodel.py:113-124` |
| 13 | Reconstruction loss | `−SI-SNR(ŝ,s)` | `sdr_type: pairwise_neg_sisdr`（旧配置是 `pairwise_neg_snr` ❌） | ✅（仅 v2） | 老配置 + `threshold_byloss: true` 会做 30 dB 截断，与论文不符 |
| 14 | Contrastive weight λ_spk | 1.0 | `loss_weight: 1.0` → `contrastive_weight` | ✅ | `av_litmodule_nass_v2.py:137-138`：`total = pit_loss + 1.0 * cl_loss` |
| 15 | Contrastive margin m | 0.5 | `margin: 0.5` | ✅ | `nass_contrastive_loss.py:69-72` 用 `relu(sim − m)` |
| 16 | τ schedule | 线性 0→1 前 1000 ep，之后固定 1 | `beta_init 0.0 / beta_final 1.0 / beta_anneal_epochs 1000`；`_get_beta()` 用 `min(epoch/1000,1)` 线性插值 | ✅ | `av_litmodule_nass_v2.py:66-70` |
| 17 | Random seed | 42 | `seed: 42` | ✅ | |
| — | （论文未列）梯度裁剪 | — | v2 注释写 "grad clip 5"，但 `av_litmodule.py` 里 **grep 不到 `clip_grad`/`gradient_clip`** | ⚠️ | `grep clip look2hear/system/av_litmodule.py` 无输出 ⇒ 实际未裁剪 |

**统计：17 行中 14 行 ✅、2 行 ⚠️（mouth 88 vs 96、梯度裁剪）、1 行依赖配置版本（第 13 行）。**

### 4.4 「near 负样本」这一行是真缺口（不是笔误）

- 论文 §4.3：负样本 = 每个**近场**干扰 `b_k`，且"both captured at the microphone and therefore carrying the same room response as the mixture"，远场被排除；`K=1..K`（Eq.8）。
- 代码：`near_audios[b]` 是数据集给的变长列表 → `encode_stft` → `pool_spk_embed`（detach）→ 传进 `NASSContrastiveLoss`。形状上完全支持 K 任意。
- 实际数据：
  - **LRS2-2Mix（v2 配置训练用）**：`near.json` K=1，路径指向另一路说话人的**源 wav**（`.../s2/<id1>_<utt1>_<snr1>_<id2>_<utt2>_<snr2>.wav`）。语义上"另一路说话人"是 2 源混合里唯一合理的近场干扰 ✅，但它是**未加权的源波形**，不是"麦克风处、带房间响应"的版本 ⚠️（LRS2-2Mix 本身无 RIR，故无从谈起）。
  - **RealSSA-Sim（本地下载包）**：`near.json` 的 9,169 条**全部是自引用 fallback**（路径 == mix 路径），而数据集 `avspeech_dataset_near.py:186-193` 明确 `if near_path == mix_path: continue` ⇒ **K=0，负样本项恒为空**，`L_spk` 退化为只剩 `(1−cos(q,p))`。
- 影响：对**评测**无影响（对比损失只在 `training_step` 里算）；对"在 RealSSA 上从零训练/微调"是**静默失效**——日志里只有 `train_cl_*_near = 0` 这类分桶计数，不会报错。既有文档 STATUS §4.5 已把这条列为口径，本轮我独立复核并**升级判定为"会静默改变训练语义"**。

---

## 5. 数字对齐表

> 论文 Table 3 / 5 / 4 / 6；本地证据状态分四档：**实测复现 / 有证据但不同 ckpt / 无数据 / 未测**。

### 5.1 Table 3（主结果，逐格）

| Method | RealSSA-Real DNSMOS↑ | RealSSA-Sim SI-SNRi | RealSSA-Sim SDRi | LRS2 SI-SNRi | LRS2 SDRi | LRS3 SI-SNRi | LRS3 SDRi | VoxCeleb2 SI-SNRi | VoxCeleb2 SDRi |
|---|---|---|---|---|---|---|---|---|---|
| AV-ConvTasNet | 2.580 | 1.2 | 2.6 | 9.8 | 10.3 | 12.5 | 12.8 | 6.9 | 8.2 |
| AV-Sepformer | 2.813 | 5.5 | 6.3 | 10.6 | 11.1 | 13.4 | 13.7 | 6.9 | 8.3 |
| AV-TF-GridNet | 3.030 | 7.6 | 8.1 | 12.7 | 13.1 | 14.3 | 14.8 | 5.8 | 7.7 |
| RTFSNet-12 | 3.066 | 4.9 | 5.7 | 12.4 | 12.8 | 14.0 | 14.4 | 8.4 | 10.7 |
| CTCNet | 2.840 | 5.4 | 6.2 | 8.0 | 8.8 | 9.4 | 10.2 | 2.5 | 4.7 |
| AV-DPRNN | 2.901 | 5.4 | 6.2 | 10.2 | 10.7 | 11.9 | 12.3 | 7.8 | 9.0 |
| AV-Mossformer2 | 2.912 | 6.0 | 6.7 | 13.2 | 13.9 | 16.2 | 16.6 | 7.7 | 9.1 |
| Swift-Net-12 | 3.276 | 7.3 | 7.8 | 13.9 | 14.1 | 16.0 | 16.3 | 9.4 | 10.8 |
| Dolphin | 2.960 | 6.7 | 7.3 | 14.0 | 14.6 | 14.3 | 14.9 | 9.2 | 11.1 |
| **Falcon** | **3.276** | **7.7** | **8.2** | **14.1** | **14.7** | **16.4** | **16.7** | **10.5** | **11.8** |

### 5.2 Falcon 那一行的本地证据状态

| 单元格 | 论文 | 本地实测 | 差 | 状态 | 证据 |
|---|---|---|---|---|---|
| RealSSA-Sim SI-SNRi | 7.7 | **7.774**（mean 通道）/ **7.977**（ch0 = 论文 channel-1） | +0.07 / +0.28 | **⚠️ 数值吻合，不构成严格复现**（ckpt 与通道两个轴都不同；W1-07 §3.3 写作「巧合级吻合」，判断相同） | 取 `P2E-realssa-sim-metrics_combined.csv` 的 `avg` 行：SI-SNRi 均值 7.774307（分母 9,169）；ch0 版取 `P2E-realssa-sim-ch0-metrics_combined-20261030.csv` 的 `avg` 行，均值 7.976709。2026-11-08 更正：原文写 9,168，逐列计数复核后 SI-SNRi 是 9,169、SDRi/PESQ 是 9,166 |
| RealSSA-Sim SDRi | 8.2 | **8.479**（mean）/ **8.452**（ch0） | +0.28 / +0.25 | **⚠️ 数值吻合，不构成严格复现**（同上） | 同上（`sdr_i` 列） |
| LRS2-2Mix SI-SNRi/SDRi | 14.1 / 14.7 | **7.029 / 7.713**（官方 `falcon.pt`，tt 3,000）；自训最佳 **8.858 / 9.270**（ep124，旧配方，单卡 100 ep） | −7.07 / −6.99 | **❌ 未复现** | `P2D-eval-falcon-full.csv`、`P2D-eval-epoch124-contrastive.csv`；根因见 P2F：官方 ckpt 训在 LRS3 |
| LRS3-2Mix SI-SNRi/SDRi | 16.4 / 16.7 | 无数据（HF 仓库仅 README） | — | **无数据** | STATUS §3 |
| VoxCeleb2 SI-SNRi/SDRi | 10.5 / 11.8 | 无数据 | — | **无数据** | 同上 |
| RealSSA-Real DNSMOS | 3.276 | 官方 ckpt **OVRL 2.7941**（P.835）/ **P808 3.3365**（54 段，full4 重跑） | −0.48 | **不可比**（论文该格属 RealSSA-trained ckpt，未发布） | `DNSMOS-realssa-real-mean-full4-20261031/`；注意 P808 反而比论文高 |

### 5.3 效率表（Table 5）

| Method | Params (M) | MACs (G) | CPU 延迟 (ms) | GPU 延迟 (ms) | CPU RAM (MB) | GPU Inf. (MB) |
|---|---|---|---|---|---|---|
| AV-TF-GridNet | 6.2 | 213.8 | 5646.3 | 567.8 | 136.3 | 1015.8 |
| AV-Mossformer2 | 56.7 | 224.2 | 16670.7 | 114.9 | 330.9 | 539.8 |
| Swift-Net-12 | 0.5 | 45.3 | 4163.8 | 172.1 | 482.3 | 411.7 |
| Dolphin | 5.6 | 18.1 | 2411.0 | 87.9 | 560.2 | 390.3 |
| **Falcon** | **0.5** | **10.6** | **1361.2** | **51.3** | **125.3** | **364.7** |

本地状态：

| 指标 | 论文 | 本地 | 状态 |
|---|---|---|---|
| Params | 0.5 M | **537,697** = 0.538 M（音频侧） | ✅ 精确（口径见 §2.4） |
| MACs | 10.6 G | **10.611 G**（ptflops, 音频, 含 LSTM） | ✅ 精确 |
| GPU 延迟 | 51.3 ms | **47.1 ms**（A800 单窗 small，非 3090） | ⚠️ 硬件不同，量级一致 |
| CPU 延迟 | 1361.2 ms | 未测 | 未测 |
| CPU RAM | 125.3 MB | 未测 | 未测 |
| GPU Inf. | 364.7 MB | 未测 | 未测 |

### 5.4 相对 AV-TFGridNet 的三个比值（摘要/§6.2 的卖点）

| 论文表述 | 论文自算 | 我用论文 Table 5 原始列复算 | 判定 |
|---|---|---|---|
| 参数 −91.9% | 91.9% | `1 − 0.5/6.2 = 91.94%` | ✅ 自洽 |
| 计算 −95.0%（正文）/ **20.2×**（摘要） | 95.0% / 20.2× | `1 − 10.6/213.8 = 95.04%`；`213.8/10.6 = 20.17×` | ✅ 自洽（95.0% 与 20.2× 是同一事实两种写法） |
| GPU 延迟 −91.0%（正文）/ **11.1×**（摘要） | 91.0% / 11.1× | `1 − 51.3/567.8 = 90.96%`；`567.8/51.3 = 11.07×` | ✅ 自洽 |
| 相对 Swift-Net-12：MACs −76.6%、GPU −70.2% | 76.6% / 70.2% | `1 − 10.6/45.3 = 76.6%`；`1 − 51.3/172.1 = 70.2%` | ✅ 自洽 |
| 相对 Dolphin：−91.1% / −41.4% / −41.6% | 91.1 / 41.4 / 41.6 | `1−0.5/5.6=91.07%`；`1−10.6/18.1=41.4%`；`1−51.3/87.9=41.6%` | ✅ 自洽 |

> **结论**：这三个"卖点比值"在论文内部完全自洽，可放心引用；但它们全部依赖 MACs 口径（音频、ptflops、含 LSTM）与"AV-TF-GridNet 213.8 G 同口径"这一前提。**若换成 flop_counter 口径，Falcon 会变成 5.38 G、AV-TF-GridNet 也会变，比值不变 —— 但绝对值会与 10.6 G 对不上。**

### 5.5 其余表（本地无证据）

| 表 | 内容 | 本地状态 |
|---|---|---|
| Table 4 | RealSSA-Sim/Real 的 WER 与 speaker similarity（Falcon 25.47/43.05；32.74/42.23） | 无（需 Mega-ASR + WeSpeaker） |
| Table 6 | VoxCeleb2 vs RealSSA 训练的 NISQA/DNSMOS | 无（需 NISQA；DNSMOS 只有 RealSSA-Real 那半） |
| Table 7 | 阶段数 1/2/3 → 7.1/7.4/7.7 | 无（本轮只用随机权重测了同三档的 lookahead：80/208/432 ms） |
| Table 8 | 对比损失消融（Falcon 7.3→7.7；SwiftNet 7.3→7.5；Dolphin 6.7→7.0） | 部分（P2D +0.13/+0.15 dB，非论文协议） |
| Table 9 | τ 调度（0 → 7.5 / 1 → 7.4 / →2 → 7.5 / →1 → 7.7） | 无 |
| Table 17 | 分场景×分近场数的 16 格 SI-SNRi | 无（缺 per-sample 场景标签） |
| Table 18 | RNN/SRU/GRU/LSTM（6.2/7.4/7.3/7.7；0.08/0.15/0.20/0.25 M；1.90/2.71/4.42/5.69 G） | 无 |
| Table 19 | 复数/实数/直接 RI（7.7/7.3/7.1） | 无 |
| Table 20 | 重叠率 5 组（7.95/7.71/7.80/7.73/7.73） | 无 |
| Table 21 | 视觉缺失（p=0/0.25/0.5/0.75/1.0，Random vs Block） | 部分（P2B 的 zero/shuffle 两种，非 5 档 p） |

> 附：Table 20 有个内部不一致值得记一笔 —— 5 组样本数 1,834×4 + 1,833 = **9,169** ✅ 与 test 集一致；但最低重叠组 `0.11–0.72` 的均值被写成 0.635，而该区间中点约 0.42，"均值 0.635"说明分布严重右偏，可信；不构成错误，仅是分组不等宽。

---

## 6. 论文没写但代码有的（fork 增补）、代码没实现但论文有的（缺口）

### 6.1 代码有、论文没写

| # | 内容 | 证据 | 影响 |
|---|---|---|---|
| A1 | **`configs/LRS2_2Mix_Paper_v2.yml`** 整套论文配方配置 + 8 卡启动脚本（`run_lrs2_paper_volcano_8gpu.sh`）+ `DATA_ROOT/NPROC/global-batch` 守卫 | 仓库 `configs/`、`tools/repro/phase2/` | 使 Table 16 可执行；论文本身没给可运行配置 |
| A2 | **`evaluated_mac_params.py`**（ptflops 复现 10.611 G / 537,697 的唯一入口） | 根目录 | 口径基准；论文没给测量脚本 |
| A3 | **2 s/0.5 s/80 ms 的窗口化推理与 100% 覆盖断言** | `eval_realssa_real.py:399-455` | 工程必需；论文只说"streaming"，没给调度 |
| A4 | **`AttentionSpeakerPool`** 的具体结构（`Linear(112→28)→Tanh→Linear(28→1)` 时间注意力 + `Linear(112→64)→ReLU→Linear(64→64)`，L2 归一化） | `Falcon_Contrastive.py:11-72` | 论文 Eq.(9) 只说"averaging over frequency, adaptively pooling over time, projecting"，**没给"自适应"= 时间注意力**这一实现选择 |
| A5 | **`Falcon._mag_sim`**（MDCL 的幅度谱 SI-SNR 变体）+ `mdcl_margin/mdcl_alpha` | `Falcon.py:1389-1420` | 论文完全没有 MDCL 这条线；属 fork/上游遗留的可选损失 |
| A6 | **训练日志的 near 分桶计数**（`train_n_samples_{0,1,2,3,4plus}_near`、`train_cl_*`） | `av_litmodule_nass_v2.py:19-33,182-183` | 让"K=0"这种静默失效变得可观察（虽然不报错） |
| A7 | **`concat_layers[2]` 死参数**（5,568）与 `video_bottleneck=Identity` | `Falcon.py:870,881-893`；`Falcon.yml:kernel_size:-1` | 论文的 "visual bottleneck" 在代码里是空操作；参数表里含死重 |
| A8 | 完整的 Studio/Space/项目页/CI/166 项测试 | `studio/`、`space/`、`tools/repro/tests` | 论文没有的工程层（本报告不展开） |

### 6.2 论文有、代码没实现（或实现不到位）

| # | 论文内容 | 代码现状 | 严重度 |
|---|---|---|---|
| B1 | "All convolutions use left-aligned padding along time…no temporal lookahead"（§4.2） | AudioBlock 的 downsample/fusion/concat 硬编码 `causal=False`（`Falcon.py:772/808/821`）→ 单遍 432 ms、整链 880 ms（另一路实测 456 / 904 ms，差约 3%） | **高（直接推翻因果性声明）** |
| B2 | "LN_c is frame-local and cache-free"（Eq.2） | 主卷积栈用 `cLNhw`，含 `GlobalLayerNorm = GroupNorm(1,C,T,F)`（`normalizations.py:208,216`）→ 整窗统计，换窗长即换结果（2 s vs 4 s 差 16 dB） | **高** |
| B3 | 论文声明视觉 3 帧 = 120 ms 前瞻（App E） | FRCNN `Conv3d` 时间核 5/pad 2 → 2 帧 = 80 ms | 中 |
| B4 | 远场干扰 `f`（10–15 个）与"排除远场做负样本" | 数据里没有 far 分量可用；损失里也没有 far 项 | 中（数据侧缺失，非代码 bug） |
| B5 | Table 17 的分场景×分近场数分析 | 发布包无 per-sample 场景/近场数元数据（`mix.json` 只有路径+长度） | 中 |
| B6 | RealSSA-Sim 的 **train/val**（100,826 / 9,133） | 本地只有 `tt`；`tr/cv` 需从 TOS 恢复（STATUS 记 450 GB） | 中（工程/存储问题） |
| B7 | near 负样本（K=1..5，带房间响应） | RealSSA 本地 `near.json` 全 fallback ⇒ K=0；LRS2 只有 K=1 且是未加权源波形 | **高（训练语义）** |
| B8 | Mega-ASR / WeSpeaker / NISQA 三条评测 | 未实现 | 中（Table 4/6 无法核） |
| B9 | Table 18/19/20/21 的消融 | 未做 | 低（可后续补） |

---

## 7. GAP LIST（≤10 条，可直接认领）

| # | GAP | 为什么卡住 | 需要什么才能关闭 |
|---|---|---|---|
| G1 | **论文 LRS2-2Mix / LRS3-2Mix / VoxCeleb2-2Mix 的 SNR 采样范围**完全没写（App B 只说 "followed the protocol of prior causal AVTSE work [6,17]"） | 无法判断本地数据构建（`build_lrs2_paper_data.py`，文件名里带 ±SNR）是否与论文同分布；这直接影响 14.1 能否复现 | 问作者要数据构建脚本/参数，或从 [6]/[17] 的协议反推 |
| G2 | **14.1/14.7 到底用哪个 ckpt、哪份训练数据**：论文说"按 benchmark 训练"，但官方 `falcon.pt` 训在 LRS3_complex_nonmoving（P2F 已证） | 本地只能用官方 ckpt 做跨语料评测，得 7.03 —— 无法区分"配方不足"vs"ckpt 不对" | 要 benchmark-specific ckpt，或用 8 卡按 v2 配方重训（33–50 h） |
| G3 | **"single encoder–decoder pass" 的计数口径**：B_a 在 §4.1 被调用两次，10.6 G 是 2 次的总和 | 论文摘要/§4.2 反复用"single pass"叙事，与 MACs 口径不自明 | 作者澄清："1 pass" 指 AudioBlock 内部一次 U 型遍历 |
| G4 | **§6.4 的 "15 frames lookahead"（论文声明值）是怎么算出来的**：120 ms ÷ 8 ms = 15 帧；但代码实测 54 帧/遍 | 若论文的因果化实现（left-aligned conv）真做对了，则**本仓库的 fork 丢了 causal 透传**；若没做对，则论文声明有误 | 要官方因果化代码，或作者确认 `downsample/fusion/concat` 是否应 `causal=True` |
| G5 | **2D attention 的算子形状**：论文写 (t,f) query → (t′,f′) key 的 block-causal；代码是帧级 T×T + tril | 影响"block-causal"到底是不是这个意思；也影响 KV cache 设计 | 作者澄清；或看官方实现 |
| G6 | **Table 3 的 RealSSA-Real DNSMOS 3.276 用哪个 ckpt** | RealSSA-trained ckpt 未发布，该格不可复现（官方权重 2.7941） | 要 RealSSA-trained ckpt |
| G7 | **Table 5 的 CPU/GPU 延迟/内存无法核**：论文 3090 + Xeon 6226R、50 次中位数；本机只有 A800 且测法不同 | 51.3 / 125.3 / 364.7 三个数只能"量级接受" | 需 3090 机器 + 论文同款测量脚本 |
| G8 | **RealSSA-Sim 的 RIR/场景元数据是否随包发布**：论文说"metadata records room geometry, RT60, per-source positions, RIR hashes, SIR, Hive label ids" | 本地发布包 `mix.json` 只有路径+长度；G/Table 17 因此无法做 | 要官方 metadata（或 gpuRIR 生成脚本 + 种子 42） |
| G9 | **near 负样本的可重建路径**：论文 App A.2 给了 LibriSpeech+RIR 的做法，但脚本"will release" | 本地 RealSSA 的 K=0 会让对比损失静默退化；LRS2 的 K=1 也不是"带房间响应"的版本 | 要生成脚本；或本地按 A.2 自建（需 gpuRIR + LibriSpeech） |
| G10 | **发布 ckpt 只到 epoch 714 / 1000**：`falcon.pt` 的 `epoch: 714`，配 τ 退火 1000 epoch 的配方 ⇒ 退火只走到 β≈0.714 | 若论文最终模型跑满 1000 ep，则官方 ckpt 不是论文模型；若论文也用 714，则 τ 退火未完成 | 问作者：Table 3 的 Falcon 是 714 ep 还是 1000 ep |

---

## 8. 证据清单

### 8.1 论文侧

| 项 | 路径/锚 |
|---|---|
| HTML 原文 | 远端 `/tmp/falcon_paper.html`（504,680 B，`https://arxiv.org/html/2610.32749v1`） |
| 结构化文本 | 远端 `/tmp/falcon_struct.txt`（701 行 / 111,858 字符 / 597 块）；本地副本 `/tmp/falcon_struct.txt` |
| 抽取脚本 | 远端 `/tmp/extract_paper.py`（stdlib `html.parser`，保留 `h1-h6`/`p`/`li`/表格行） |
| 引用锚点 | §1、§3、§4.1–4.3、§5.2、§6.1–6.4、App A（A.1–A.4）、App B、App C(Table 15)、App D(Table 16)、App E、App F、App G(Table 17)、App I/K/L/M、Table 1–21 |

### 8.2 代码侧（路径均为 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork/`）

| 用途 | 文件:行 |
|---|---|
| `ConvNormAct` 因果/对称 padding | `look2hear/models/Falcon.py:45,51,94-109` |
| `NeuroSCNetFusion`（视频投影 + nearest 上采样 + 加性融合） | `Falcon.py:241-267` |
| `InjectionMultiSum`（GCI 实现，Eq.6） | `Falcon.py:269-331`（核心式在 `:329`） |
| `DualPathRNN`（频向/时向 LSTM；`causal` 死参数） | `Falcon.py:333-417` |
| `MultiHeadSelfAttention2D`（帧级 T×T + tril） | `Falcon.py:419-544`（`:497-521`） |
| `AudioBlock`（U 型 4 级；causal=False 硬编码） | `Falcon.py:693-904`（`:772,808,821`）；forward `:828-897` |
| `VideoBlock`（单向 LSTM 32） | `Falcon.py:900-976` |
| `Separator`（repeats 循环） | `Falcon.py:978-1058`（`:1000-1005,1032-1058`） |
| `STFTEncoder`（3 通道 + causal conv） | `Falcon.py:1061-1128`（`:1117-1126`） |
| `MaskGenerator`（复数掩码 Eq.1） | `Falcon.py:1201-1282`（`:1251-1268`） |
| `Falcon`（装配；`audio_params['causal']` 注入） | `Falcon.py:1284-1354`（`:1331-1332`） |
| `FalconSpeakerContrastive` / `AttentionSpeakerPool` | `look2hear/models/Falcon_Contrastive.py:11-129` |
| NASSv2 训练步（query/pos/neg 组装） | `look2hear/system/av_litmodule_nass_v2.py:73-138` |
| `NASSContrastiveLoss`（Eq.10） | `look2hear/losses/nass_contrastive_loss.py:52-77` |
| `cLNhw`/`GlobalLayerNorm`/`CumulateLN4D` | `look2hear/layers/normalizations.py:134-144,165-202,204-220` |
| FRCNN 时间维感受野 | `look2hear/videomodels/frcnn_videomodel.py:60-70` |
| 数据：Grayscale + near 跳过 fallback | `look2hear/datas/avspeech_dataset_near.py:81,168-174,186-193` |
| mouth 预处理 Crop 88 | `look2hear/datas/transform.py:147-173` |
| MACs/params 脚本 | `evaluated_mac_params.py:39,75-123` |
| 整段评测 | `audio_visual_test.py:23,419-432` |
| 窗口化评测（2 s/0.5 s/80 ms） | `tools/repro/phase2/eval_realssa_real.py:381-455`（`:399-400`） |
| DNSMOS（P.835 默认 + P.808） | `tools/repro/phase2/eval_dnsmos.py:36,68-89` |
| 配置（论文宽度） | `configs/Falcon.yml`；`configs/LRS2_2Mix_Paper_v2.yml` |

### 8.3 本轮实测命令（可重跑；全部 CPU / 无 GPU / 未改仓库）

```bash
# M1/M2/M3：参数量与 MACs（音频 537,697 / 10.611 G；FRCNN 88² = 15.897 G）
cd /root/code/github_repos/Falcon-fork && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \
  /root/miniforge3/envs/falcon/bin/python -c "…get_model_complexity_info(am,(a,v),input_constructor=…)"
# M4/M5/M6：真实权重 tail-perturb 测 lookahead（结果：encoder 1 帧；+separator 110 帧=880 ms；另一路算子隔离得 113 帧=904 ms）
# M7：AudioBlock depth 2/3/4/5 → 80 / 208 / 432 / 944 ms
# M8：2 s 窗 vs 4 s 整段 → SI-SNR 16.03 dB
# M9：读 P2E CSV 的 avg 行 → 7.774/8.479（mean）、7.977/8.452（ch0）；n=9,169（SI-SNRi）/ 9,166（SDRi）
# M10：near.json K 全 1（含自引用）；RealSSA-Real 54 段 / 5039.4 s
```

### 8.4 引用的既有本地证据（我未重跑，仅引用并标注）

| 结论 | 来源 | 我的处理 |
|---|---|---|
| 官方 `falcon.pt` 训练于 LRS3_complex_nonmoving | `docs/verification/phase2/P2F-report.md` §1.4（引 `Experiments/checkpoint/Falcon/conf.yml:99-102`） | **引用**，未复核 conf.yml；因此 LRS2 差距的"根因"判定标为引用级 |
| LRS2 自训 ep86/ep99/ep124 三条结果 | `P2D-eval-epoch{86,99,124}*.csv` | 引用数字，未重算 |
| DNSMOS full4 = 2.7941 / P808 3.3365 | `DNSMOS-realssa-real-mean-full4-20261031/` | 引用，未重算 |
| big−small = +0.97/+0.91 dB | `P2B-summary-b300-big.json` | 引用，未重算 |
| 单遍 lookahead 57 帧≈456 ms | `docs/STATUS.md` §4.1 | **与本轮 54 帧/432 ms 互证**（两路差约 3%）；并补上整链 110–113 帧 / 880–904 ms 这一 STATUS 未记录的事实 |

### 8.5 与既有文档的冲突/更正（必读）

| 旧说法（出处） | 本轮实测/核查 | 性质 |
|---|---|---|
| `STATUS.md` §4.1："单遍 lookahead 实测 **57 帧≈456 ms**" | 单遍 **54 帧 = 432 ms**；**整链（两次 AudioBlock）110 帧 = 880 ms**，另一路实测 113 帧 = 904 ms | **补正**：57 帧是单遍量级，文档未说明整链约 2 倍，容易被读成「端到端 456 ms」 |
| `STATUS.md` §3："MACs 10.611 G ✅ 精确复现" | 正确，但要补口径：ptflops + 音频侧 + **含 LSTM**。`flop_counter` 口径会得 5.381 G | **补口径**（2026-11-08 更正：否则 21.19 G / 74.6% 与 10.6 G 并列会自相矛盾；整窗正确值是 26.51 GMAC / 60.0%） |
| `STATUS.md` §3：RealSSA-Sim"7.774 / 8.479" | CSV `avg` 行同为 **7.774 / 8.479**（SI-SNRi 分母 9,169、SDRi 9,166） | 一致。2026-11-08 更正：本报告此前写 7.776/7.979，没有交代剔除规则，已按 CSV `avg` 行改回；原写的 n=9,168 也不成立 |
| `PAPER_ALIGNMENT_20261030.md`（未逐行核） | 本轮若与其冲突，以本报告的 `path:line` 为准 | 待 Lead 交叉 |
| 任务简报里"分离两路语音" | 论文与代码都是**单输出 AVTSE**（`n_src=1`） | **先拨乱反正**：任何"2 源分离"表述都不成立 |
| 任务简报里"21.19 GMAC、视觉占 74.6%" | 该数来自 `flop_counter` 口径（LSTM 记 0）；正确整窗口径是 **26.51 GMAC / 视觉 60.0%** | **2026-11-08 口径更正**：两个数都可引用，但不可与 10.6 G 混用 |

---

## 附：一句话总评

**这个 fork 把论文"能对的地方"对得非常准（宽度、算子、损失公式、配方、数据规模），把论文"最想让人相信的地方"（论文声明的 120 ms 因果、根本不用未来帧）对不上**——差距不在实现水平，而在 `AudioBlock` 里三组 `causal=False` 与 `cLNhw` 里的整窗 `GroupNorm`：前者把几何 lookahead 从编码器阶梯的 120 ms 抬到 432→880 ms，后者让"streaming"在数学上不成立。padding 这一条是约 6 行的改动（`Falcon.py:772,808,821`），改完几何 lookahead 回落到编码器阶梯 15 帧 / 120 ms（推算值，未实测）；`cLNhw` 不在那 6 行里，30 处归一化要单独换成逐 (t,f) 的 `LN_c`，仅改 padding 达不到论文 "cache-free" 的声明。改完还要重训才能验证精度不掉，这也是本轮最值得向作者/Lead 追问的一条。
