# Falcon 项目完整理解（FINAL · Lead 汇总版）

> 本文件是 8 个 subagent 多轮“理解 R1 → 交叉评审 R2 → 修订 R3 → 复核 R4”的汇总产物，由 Lead 写成。
> 证据规范：`[代码]` 读源码、`[验证]` 本机实际运行（脚本/日志在 `docs/verification/`）、`[推断]` 推断。
> 仓库：`/root/code/github_repos/Falcon-fork`（commit `7576854`；**2026-10-21 已同步上游 `689555c`，差异见 §17**）。8 份子系统深度文档见 `docs/understanding/0X-*.md`，
> 评审记录见 `docs/reviews/round1|round2/`，所有可执行验证脚本/日志见 `docs/verification/`。

---

## 0. 一页速览（先读这 10 条）

1. 这是 **NeurIPS 2026 论文《Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World
   Acoustic Scenes》的官方仓库**，包含两条相互独立、共享模型定义的产品线：
   - **训练/评测代码**（仓库根 + `look2hear/`，PyTorch Lightning）——研究代码，发布状态下**不可直接跑通**（有 P0，见 §10）。
   - **Falcon Studio**（`studio/`，FastAPI/WebSocket + 纯前端 JS）——工程完成度高，**真实权重可加载、窗口化实时推理可跑**（Lead 已实测）。
2. **模型本体极小**：音频网络 **537,697 参数（≈0.538 M，即论文“0.5 M trainable”口径）**；加冻结的
   11,186,688 参数 FRCNN 唇读编码器后总计 **11,724,385 ≈ 11.72 M**（与 README 精确一致）[验证]。
3. 架构是 **单次 encoder–decoder 的多尺度时频分离**：STFT 编码 → 瓶颈 → 4 级金字塔 AudioBlock（每级与
   VideoBlock 融合）→ 最粗尺度 TFRB（频向双向 LSTM + 时间向单向 LSTM + 因果 2D attention）→ 复数掩码 →
   因果 STFT 解码。README 图里的 **TFRB/GCI 是组件名**，代码里对应 `DualPathRNN(dim=4)+DualPathRNN(dim=3)+
   MultiHeadSelfAttention2D` 与 `InjectionMultiSum`。
4. **“causal”要精确表述为“窗口级因果（window-causal）”**：卷积/RNN/attention 结构单向，但
   `torch.stft(center=True)` 有 ±128 采样（8 ms）分析窗、AudioBlock 的下采样/融合卷积 `causal=False`
   （关掉归一化后实测 lookahead **57 帧=7,296 采样≈456 ms**）、`cLNhw` 的 `GlobalLayerNorm` 是**全窗口**统计、
   FRCNN 时间感受野 ±2 帧（±80 ms）。端到端单点扰动可影响**改动点之前 9,612 采样（≈−649 ms）**的输出 [验证]。
   它**不是逐采样流式因果模型**。
5. **Studio 的实时方案 = “固定 2 s 窗口、每 0.5 s 全窗重算”**，不是流式状态：`window 32000 / hop 8000 /
   guard 1280(80 ms)`，首窗需 9280 采样（0.58 s）+ 左上下文 22720 采样（1.42 s）；服务端 `OutputSlicer`
   输出 `pred[22720:30720]`，窗口间 20 ms（320 采样）交叉淡化。
6. **真实发布权重已由 Lead 下载并验证**（此前所有 agent 均无法验证）：`falcon.pt` = `{state_dict(525 audio +
   137 video keys), architecture: FalconSpeakerContrastive, epoch: 714}`；`falcon-big.pt` = `{...517+137,
   SpeakerContrastiveV2, epoch:715}`；两个文件只含 `audio_model.*`/`video_model.*`；**训练侧与 Studio 侧代码
   对同一真实权重、同一输入的前向输出逐位相同（max|Δ|=0）** [验证]。
7. **训练入口有 2 个 P0**：`audio_train.py:170` 使用了全仓库未定义的 `comet_logger`（NameError）；三个
   litmodule 的日志又硬依赖 Comet 专属 `self.logger.experiment.log(dict)`（换任何非 Comet logger 即
   AttributeError）。按 README 训练命令**无法启动**。
8. **评测入口/指标有 3 个 P0**：`Experiments/checkpoint/Falcon/conf.yml` 的 `audionet_name: SpeakerContrastiveV2`
   在 `look2hear.models` 中不存在（AttributeError）；该快照与 `best_k_models.json` 使用未展开的
   `${ROOT}/${DATA_ROOT}` 占位路径且仓库无 ckpt；`ALLMetricsTracker` 的 **“sdr/sdr_i”列实际是 −SNR/SNRi**
   （真正的 SDR 只在死代码 `MetricsTracker` 里），且逐句行与 avg/std 行符号相反、PESQ 实参传反。
9. **数据管线的“按排序下标对齐、无校验”与 `_meta.json` 无生成者**是两个系统性陷阱：按 README 流程
   （只跑 `process_lrs23.py` + `generate_near_json.py`）时 `<mix>_meta.json` 不存在 → `near.json` 全为
   fallback 哨兵 → **NASS 对比损失负项恒 0、MDCL 无效、val 分桶全 0near**（即论文核心贡献之一在复现流程中静默失效）。
   另：评测 Dataset 对 mono wav 的 `mean(-1)` 会退化为 0 维并在 `torch.stft` 崩溃（与 README “mono 16 kHz”矛盾）。
10. **Studio 的协议/安全/窗口切片设计与前端高度自洽**（常量、payload 长度、坐标换算、20 ms+5 ms 淡化、
    349 条 i18n 全部实测一致）；但仓库内 `server.py` 只允许 loopback + TrustedHost 白名单，**公开在线 demo
    必然是另一份部署补丁**；且“服务端 error 是否致命”在前端实现为整场停止，与后端“跳过该窗口”文案不一致。

---

## 1. 项目定位与两条产品线

| 维度 | 训练/评测产品线 | Falcon Studio |
|---|---|---|
| 目的 | 复现论文训练与 benchmark 数值 | 本机实时 AVTSE demo / 会议降噪 |
| 入口 | `audio_train.py`、`audio_visual_test.py`、`evaluated_mac_params.py` | `studio/scripts/launch.sh` → `server.py` → 浏览器 |
| 依赖面 | torch、PyTorch Lightning、librosa、torch_complex、typeguard、timm、torch_optimizer、ptflops、pystoi/pypesq、torch_mir_eval、fast_bss_eval 等 | torch、numpy、fastapi、uvicorn、huggingface_hub、soundfile（无 librosa/PL） |
| 代码量 | `look2hear/` ≈ 12 k 行 + 根脚本 ~700 行 | `studio/` ≈ 35 k 行（含 38 MB MediaPipe vendor） |
| 发布状态 | **不可直接跑通**（P0 阻塞，见 §10） | **可跑通**（真实权重已实测加载 + infer） |
| 共享 | `Falcon.py` 模型定义、`Falcon.yml` 超参、window/fps 约定 | `falcon_model/` 为推理拷贝（含 +9 行 MPS CPU FFT bridge） |

项目还有：`DataPreProcess/`（JSON 索引生成）、`Experiments/checkpoint/Falcon/`（训练快照，非权重）、
`project_page/`（GitHub Pages 项目页）、`assets/`（README 图）、四份 README、Apache-2.0 LICENSE。

---

## 2. 全景架构与端到端数据流

```
                          ┌──────────────── 训练/评测产品线 ────────────────┐
LRS2/LRS3/VoxCeleb2 ──► DataPreProcess ──► mix/target/near.json ──► AVSpeechDataModuleWithNear
   (tr/cv/tt: mix+target wav + mouths/*.pt + <mix>_meta.json)                │
                                                                  ┌─────────┴──────────┐
                                                     FalconSpeakerContrastive      FRCNNVideoModel(冻结)
                                                     (audio, look2hear/models)      (look2hear/videomodels)
                                                                  └─────────┬──────────┘
                                                              AudioVisualLightningModuleNASSv2
                                                          (PIT 损失 + NASS speaker contrastive, β 退火)
                                                                  │ PyTorch Lightning DDP Trainer
                                                     Experiments/checkpoint/Falcon/{conf.yml, ckpt, best_k_models.json}
                                                                  │ audio_visual_test.py
                                                     metrics_rank*.csv → metrics_combined.csv (SI-SNRi/伪 SDRi/PESQ/STOI)
                          └──────────────────────────────────────────────────────────────┘

                          ┌──────────────────── Falcon Studio 实时链路 ────────────────────┐
浏览器: getUserMedia(audio/video) → AudioWorklet 重采样 16 kHz → 8 s ring → 2 s 窗口/0.5 s hop
        video 30 fps → worker 节流 25 fps → MediaPipe FaceLandmarker → 96×96 灰度 JPEG → 50 帧
        → WebSocket /ws {type,id,audio(b64 f32 32000),frames(50×b64 jpeg),end_sample,reset,...}
本机 service: FastAPI server.py → Engine → FalconRuntime(严格加载) → 模型窗口推理
        ← result {audio(b64 f32 8000), start/end_sample, contiguous, rtf, inference_ms, ...}
浏览器: playback-resampler(63-tap/2048 相) + crossfade → 耳机监听 / 录音导出 WAV / BlackHole 虚拟麦克风 → 会议
                          └──────────────────────────────────────────────────────────────┘
```

宏观结论：**Studio 不训练模型；它只是把训练侧的 Falcon 前向封装成窗口化推理服务，再配合浏览器采集/追踪/播放**。
两个产品线的模型定义在数值上等价（真实权重逐位验证一致），差异仅在依赖裁剪、加载策略、MPS bridge。

---

## 3. 核心模型（`look2hear/models/Falcon.py`，1487 行）—— 深度结论

### 3.1 类图与职责（17 个类中 4 个是死代码）

| 组件 | 职责 | 关键事实 |
|---|---|---|
| `STFTEncoder` | 波形 → 复数谱特征 | `torch.stft(win=256, hop=128, center=True)`；`[B,3,T,F]`（real/imag/mag）经 1 个 `ConvNormAct`(3×3, 3→112) → `[B,112,T',F]`；参数 **3,024**；`T'=1+L/128`，`F=129` |
| `audio_bottleneck` | 通道对齐 + 归一化 | `ConvNormAct(in=112,out=112,k=1,pre_norm=cLNhw)`；参数 **13,104** |
| `video_bottleneck` | 视觉特征通道对齐 | `kernel_size=-1` + `causal=False` ⇒ `nn.Identity`、**0 参数** |
| `Separator` | 多尺度音视频融合分离 | 内含 1 个 `AudioBlock`（`repeats=2, shared=true` ⇒ 同一实例前向 2 次）、1 个 `VideoBlock`、`NeuroSCNetFusion`；参数 **492,312（82.9%）** |
| → `AudioBlock` | 4 级金字塔 + GCI 融合 + TFRB | 形状 `(T,F)`: `(251,129)→(126,65)→(63,33)→(32,17)`，编码侧**只在最粗级**跑 TFRB；解码侧按 2→1→0 级；`n_freqs` 记账 129/65/33/17 与实测一致 |
| → `DualPathRNN`（TFRB 的 RNN 部分） | 频向双向 / 时间向单向 | `dim=4`: batch=B·T、seq=F、bidirectional；`dim=3`: batch=B·F、seq=T、单向（因果）；底层 `getattr(nn,"LSTM")`，**不是** `layers/rnnlayers.py` 的 DPRNN；`out_proj` 48→64 / 72→64 |
| → `MultiHeadSelfAttention2D`（TFRB 的 attention 部分） | 时间轴 masked attention | `causal=True` 用 `tril` mask（只看过去帧）；`n_freqs` 必须等于运行时 F（γ 形状 `[1,C,1,F]`）；只在最粗级调用 |
| → `InjectionMultiSum`（GCI） | 跨尺度门控融合 | `nearest` 插值对齐 + 门控；`causal=False`（对称 k=2 卷积） |
| → `VideoBlock` | 视觉特征时间下采样 + 融合 | Conv1d stride=2 → 50→25 帧；eval 因果、**train 下 BatchNorm1d 用批统计（非因果）** |
| `MaskGenerator` | 复数掩码生成与应用 | `RI_split=True`：112 通道=56 复通道×2；复乘 `est_real=e_r m_r−e_i m_i` 等；输出 `[B,n_src,112,T',F]` |
| `STFTDecoder` | 复谱 → 波形 | 因果 ConvTranspose2d(112→2) + `torch.istft(length=L)`；参数 2,016；采样级严格因果（改动点前恒 0） |
| `AttentionSpeakerPool`（`Falcon_Contrastive.py`） | 说话人 embedding 池化 | 频向均值 → 时间注意力 → 投影 + L2 归一化；参数 14,584 |
| `FalconSpeakerContrastive` | 论文模型类 | 训练模式返回 `(est[B,1,L], sep_emb[B,1,112,T',129])`，eval 只返回 `est`；`**kwargs` 会静默吞掉 `near_audio` |
| 死代码 | `PositionalEncoding`/`MultiHeadSelfAttention`/`FFN`/`GlobalAttention`（整组不可达）；`timm.DropPath` 只为这些死类存在 | 17 类中 4 个死；`timm` 依赖理论上可移除 |

### 3.2 参数量拆解（可复算，[验证]）

| 子模块 | 参数 |
|---|---|
| encoder | 3,024 |
| audio_bottleneck | 13,104 |
| video_bottleneck | 0 |
| separator | 492,312 |
| mask_generator | 12,657 |
| decoder | 2,016 |
| spk_pool_proj（对比头） | 14,584 |
| **音频网络合计** | **537,697**（扣对比头 = 523,113；其中 `concat[2]` 的 5,568 从未参与前向 ⇒ **实际生效 532,129**，扣对比头 517,545） |
| FRCNNVideoModel（冻结） | 11,186,688 |
| **总计** | **11,724,385** |
| Falcon-Big（Studio）音频网络 / 总计 | 1,158,617 / 12,345,305 |

> 论文“0.5 M trainable”= 537,697（约 0.538 M，1 位有效数字取整）。注意 `concat_layers[2]` 的 5,568 参数
> 在前向中**从未被调用**（见 §3.4），占 1.04% ⇒ **实际生效 532,129**（√ 01-model-core R3 §4.7/A51；
> 是否属于“有意跳过”待作者确认）。另注意**统计口径陷阱**：`state_dict` 的 numel ≠ `parameters()`——
> audio 537,762（+65 个 BN buffer）、video 11,196,308（+9,620），核算参数量必须用 `parameters()` [验证]。

### 3.3 配置映射（`configs/Falcon.yml` → 构造）

- `audionet_config` 11 个顶层键：`n_src, pretrained_vout_chan(512), causal, spk_embed_dim(64), video_bn_params,
  audio_bn_params, enc_dec_params, audio_params, video_params, fusion_params, mask_generation_params`。
- `enc_dec_params`（win 256/hop 128/out_chan 112/kernel 3/causal true）同时喂 `STFTEncoder` 与 `STFTDecoder`；
  `audio_params.layers.layer_1/2/3` 决定 AudioBlock 三级 TFRB 子层；`repeats:2`+`shared:true` ⇒ **1 个实例调用 2 次**
  （不翻倍参数）；`upsampling_depth:4` ⇒ 金字塔 4 级。
- **必须 `copy.deepcopy(audionet_config)`**：`Separator.__init__` 会 `pop("repeats"/"shared")` 原地修改 dict
  [验证：同一 dict 二次构造，参数量 537,697→49,081，前向不报错、静默残缺]。
- 惰性/死配置：顶层 `causal` 写入两层 dict 但 AudioBlock 仅传给 DualPathRNN，而 DualPathRNN 不读它 ⇒
  `causal=True/False` 两个模型**参数量、state_dict、同权重输出全部 bit-identical**；`video_params.layers`、
  `fusion_type`、`mask_generator_type`、`video_params.stride` 均未被使用。

### 3.4 关键实现缺陷与死代码（本轮交叉评审确认）

1. **`concat_layers[2]` 从未被调用**（`Falcon.py:868` 用 `concat[-1]=[3]`、`:879` 只循环 i=1,0；forward hook
   计数 `concat[0]/[1]/[3]=2、concat[2]=0`）→ 5,568 参数闲置（**实际生效 532,129**）。结合源码注释 “Fuse the two deepest levels”，
   疑似 level-2 融合缺失（**待作者确认**，影响参数量口径与模型容量）。
2. `Falcon.forward` 在 `near_audio≠None` 时调用 `self._mdcl_loss(...)`，但类中只有 `_mdcl_loss_hard`
   → **AttributeError**（裸 `Falcon` 训练路径必崩）；而 `FalconSpeakerContrastive.forward(**kwargs)` 把
   `near_audio` 静默吞掉 → MDCL 在对比类下静默失效。两者叠加 = MDCL 路线事实上不可用。
3. `n_src` 只支持 1：`n_src=2` 前向 RuntimeError（decoder reshape 固定）。
4. `FRCNNVideoModel.train()` 重写**没有 `return self`** → `FRCNNVideoModel(...).eval()` 返回 `None`
   （任何链式调用会崩；评测脚本恰好用语句式调用未暴露）。
5. `models/__init__.py` 的 `__all__ = ["AVTIGER_causal"]` 指向不存在名字（`import *` 失败，`get()` 不受影响）。
6. `SISNRi` 因 `super(Loss, self)` 直接 NameError；`MixIT`/`new_PITLossWrapper`/`MultistageLoss`/`MetricsTracker`
   /`SPlitMetricsTracker` 等大量死代码；`layers/` 下 `stft.py`、`cnnlayers.py`、`rnnlayers.py`、`enc_dec.py`、
   `stft_tfgn.py` 与 `utils/stft.py` **全部零调用**（但 `import look2hear.models` 会把它们连坐加载，使
   librosa/torch_complex/typeguard 成为硬依赖）。
7. `layers/stft.py` 与 `utils/stft.py` 是**同源复制**（去空白 md5 相同 `3f705313…`，diff 仅 1 增 1 删空白）。
8. `normalizations.py` 中 `CumulateLN` 定义两次（后者覆盖前者，前者的 `cumsum(dim=1)` 是 bug）；
   别名坑：`LN` 指向 `CumulativeLayerNorm`（并非累积）、`cLN` 才是累积。

### 3.5 因果性最终裁定（多轮交叉评审 + 多协议探针统一口径）

| 破点 | 机制 | 量化（[验证]） |
|---|---|---|
| STFT 分析窗 | `torch.stft(center=True)` 每帧用 ±128 采样 | 帧级 lookahead ≤128 采样 = 8 ms |
| AudioBlock 卷积 | 下采样/融合/residual 卷积 `causal=False`（对称 padding） | 关掉 cLNhw 后单遍 lookahead **57 帧=7,296 采样≈456 ms**；逐级 L0±1/L1±4/L2±8/L3±16 帧 |
| `cLNhw` 归一化 | `GlobalLayerNorm=nn.GroupNorm(1,C)` 在 `[B,C,T,F]` 上对 (C,T,F) **全窗口**统计 | 单层探针：同分布替换 1.4e-6 / 波形×10 4.7e-5 / T=8 尖峰 1.2e-3（协议不同都正确；`CumulateLN4D` 会抵消纯尺度泄漏：GN-only 4.40 → cLNhw 整体 1.9e-6） |
| FRCNN 视觉前端 | Conv3d（5,7,7）时间 kernel 5 + 无时间下采样 | 时间感受野 ±2 帧 = ±80 ms |
| 端到端 | 上述叠加 | 改 1 个采样点，输出从 **−649 ms** 处开始变化；改未来 1 s，前 0.5 s 仍差 ~1e-4→4e-2 梯度 |

**结论**：Falcon 的 “causal” = 结构层面（因果卷积、时间单向 RNN、tril-masked attention、无跨窗状态），
**不是逐采样因果**；工程上以“2 s 窗口全窗重算 + 80 ms 右上下文 + 20 ms 淡化”实现低延迟，论文的 latency
与“streaming”应在此口径下理解。Studio 的 80 ms guard 恰好等于 FRCNN 的 ±80 ms 时间感受野，二者设计意图一致。

---

## 4. 训练系统（入口 / 配置 / Lightning / 损失）

### 4.1 装配链（`audio_train.py`，S0–S26 逐步 `file:line` 见 `03-training-system.md`）

```
YAML ──prepare_parser_from_dict──► argparse 默认值（两级 dict 展开，:201-202）
S1  datamodule = AVSpeechDataModuleWithNear(**data_config)   :49-51；setup() 后 make_loader（@property，只取一次）
S2  audio model = getattr(look2hear.models, audionet_name)(sample_rate, **deepcopy(audionet_config))  :55-58
S3  video model = FRCNNVideoModel(**videonet_config)          :61-63（加载 pretrain 并冻结）
S4  optimizer = make_optimizer(AdamW lr=1e-3, wd=0.01)        :67；scheduler = ReduceLROnPlateau(patience=15, factor=0.5)
S5  exp_dir = Experiments/checkpoint/<exp_name>；**先把 config 落盘再建 system** → dump 的 conf.yml 已被
    Separator pop 污染（缺 repeats/shared、被塞入 causal）  :82-84
S6  loss = PITLossWrapper(pairwise_neg_snr) / val: pairwise_neg_sisdr（pit_from=pw_mtx）  :91-100
S7  spk_contrast 7 键（margin/alpha_pos/alpha_neg/beta_init/beta_final/beta_anneal_epochs/loss_weight）
    全部注入 litmodule  :103-110、:129-135
S8  system = AudioVisualLightningModuleNASSv2(...)            :119-136
S9  callbacks = ModelCheckpoint(monitor=val_loss/dataloader_idx_0, mode=min, save_top_k=5, save_last,
    filename="{epoch}") + EarlyStopping(patience=30)
S10 trainer = pl.Trainer(max_epochs=1000, gradient_clip_val=5.0, strategy=DDPStrategy(find_unused_parameters=True),
    sync_batchnorm=True, limit_train_batches=1.0, num_sanity_val_steps=0, logger=comet_logger ← 未定义!, ...)
S11 fit；结束后写 best_k_models.json（indent=0）与 best_model.pth（audio_model.serialize()）
```

### 4.2 损失与训练目标

- **PIT 损失**：train 用 `pairwise_neg_snr`（SNR 非尺度不变），val 用 `pairwise_neg_sisdr` → **train/val loss
  不可直接比较**；`threshold_byloss=true` 的真实语义是“丢弃 SDR>30 dB（过好）的样本，只保留 min_loss>−30”，
  全被过滤时退回全体均值。
- **NASS speaker contrastive**（`nass_contrastive_loss.py`）：
  `L = α_pos·(1−cos(q,p)) + α_neg·Σ_k softmax_k(β·cos(q,n_k))·relu(cos(q,n_k)−margin)`；
  q=分离音频 embedding（有梯度），p=目标语音、n=near 干扰（均 detached）；**平均分母恒为 B**（无负样本也计入）；
  β 按 epoch 线性退火 `β=min(epoch/200,1)`（config 的 0→1.0/200 ep 覆盖 litmodule 默认 1.5/100）。
- **MDCL 路线**（`av_litmodule_mdcl.py`）与对比类**互斥**（期望裸 `Falcon` 返回张量，但配置模型返回 tuple），
  且 `_mdcl_loss` 缺失 ⇒ 论文配置未走 MDCL。
- val_dataloader = `[val_loader, test_loader]`（test_loss 每 10 epoch 计算一次）；1 epoch ≈ `train_samples=10000`
  个随机 2 s 片段（DDP 下每 rank 约 250×世界大小；`use_distributed_sampler` 默认生效）。

### 4.3 训练侧 P0/P1

| 级别 | 问题 | 证据 |
|---|---|---|
| P0 | `logger=comet_logger` 未定义 → `main()` NameError | `audio_train.py:170`；symtable 唯一未定义名 |
| P0 | 三个 litmodule 的 `self.logger.experiment.log(dict)` 是 Comet 专属 API；CSVLogger 实测 AttributeError，TensorBoardLogger 构造即 ModuleNotFoundError，requirements 无 comet-ml | `av_litmodule.py:153-179` |
| P1 | 新 dump 的 conf.yml（缺 repeats/shared）若用于评测，模型 537,697→49,081 参数、`strict=False` 静默丢 ~91% 权重 | [验证] monkeypatch 跑到 `audio_train.py:84` 的 dump 重建 |
| P1 | PL 2.x 断档：`accelerator=None` → ValueError；`gpus:[0..9]` 在 2 卡机 MisconfigurationException | [验证] PL 2.6.6 |
| P1 | 死配置键 `training.system/parallel`、`datamodule.data_name` 完全未被读取 | AST/grep |
| P2条件P0 | DDP per-bucket `all_gather` 只在“本 rank bucket 非空”时调用 → 2 进程 CPU DDP 实测 rank0/rank1 bucket 不对称后挂死（真实 10 卡 cv 集大时未必触发） | [验证] 2 进程复现 |

---

## 5. 评测与指标（`audio_visual_test.py` / `evaluated_mac_params.py` / `metrics/`）

### 5.1 评测流程

- `--conf_dir` 读训练快照 → `arg_dic["train_conf"]`；`--ckpt_path` 支持本地或 `hf://<org>/<repo>/<file>`（用
  huggingface_hub 下载）；不传时按 `best_k_models.json`（最低 val_loss）→ `last.ckpt` → `epoch=*.ckpt` 顺序找。**注意第三级是字符串排序取最后一个**：`sorted(glob(...))[-1]`
  在 `epoch=1000` 与 `epoch=742` 并存时会选中 **742**（并非 epoch 数值最大），04 round2 已实测复现
  （A38 原断言写反）——这是评测回退链的一个小坑。
- 不构建 DataLoader：`datamodule.setup()` 后逐条 `test_set[idx]`、**整句**推理、`rank::world_size` 分片、
  `batch=1`；`use_near=False` 只影响训练，评测仍读 `_meta.json` 的 `n_near`（结果丢弃）。
- 指标：每 rank 写 `Experiments/checkpoint/<exp>/results/metrics_rank*.csv`，最后合并为 `metrics_combined.csv`。
- **真实权重验证**：`falcon.pt` sha256 `a1888b5a…928c`、47,156,343 B（44.97 MiB）、顶层是 `{"state_dict": ..., "architecture": "FalconSpeakerContrastive", "epoch": 714}`
  → `_load_models_from_ckpt` 的 `ckpt["state_dict"]` 假设**成立**（此前 04 文档的 Q1 已关闭）[验证]。
- 真实权重整句评测实测（05 R3 §8.7）：`T=32000/F=50` 与 `T=48513/F=75` 均能完成
  `audio(1,112,T',129) ← video(1,512,F)` 融合并输出 `est(1,1,T)`（`len==T`、finite）——**整句变长推理契约成立** [验证]。

### 5.2 指标实现的三处问题（P0/P1）

1. **`ALLMetricsTracker`（评测实际使用）**：`sdr = pit_snr(...)`、`sdr_i = sdr − baseline`，写入 CSV 的
   “sdr/sdr_i” 实际是 **−SNR / SNRi**；而 `si-snr` 列取负号后为正；**逐句行与 avg/std 行符号相反**
   （`all_sdrs` 存的是 `−sdr`）。真正 SDR 只在死代码 `MetricsTracker` 里（`−fast_bss_eval.sdr_pit_loss`）。
2. **PESQ 实参传反**：`allwrapper.py:50` 调 `pesq(estimate, clean, 16000)`，而 pypesq 合同是 `pesq(ref, deg, fs)`
   → 应为 `(clean, estimate)`。
3. **口径不可比**：训练用 2 s 随机片段、评测用整句；论文 SI-SNRi/SDRi 若来自本入口，需确认实际列含义。

### 5.3 效率数字

- `evaluated_mac_params.py` 用 ptflops 只统计 **audio model**，MACs = **10.611 G**（与 README 10.6 G 一致）
  [验证]；脚本硬编码 CUDA + 硬编码 `configs/Falcon.yml`，CPU 机不可跑。
- README 的 GPU/CPU latency、RAM/显存数字无法在本环境复现（无 3090、无真实 benchmark 数据）。

### 5.4 评测侧 P0（按 README 命令逐条）

| 断点 | 证据 |
|---|---|
| `conf.yml:88 audionet_name: SpeakerContrastiveV2` 在 `look2hear.models` 不存在 → `getattr` AttributeError | [验证] hasattr=False |
| 数据路径 `${DATA_ROOT}` 未展开 → datamodule.setup() 先 FileNotFoundError | `conf.yml` 快照 |
| 无 ckpt 且 `best_k_models.json` 是 `${ROOT}` 占位（indent=2，非当前代码产物） → 不传 `--ckpt_path` FileNotFoundError | [验证] 文件内容 |
| mono wav → `AVSpeechEvalDataset` 的 `mean(-1)` 退化为 0 维 → `torch.stft` RuntimeError；训练期 val 同 dataset 也会崩 | [验证] 真实模型复现 |
| 根 requirements 缺 `huggingface_hub`（`hf://` 分支需要）；timm 实际已声明 | `requirements.txt:17` |

---

## 6. 数据管线（`look2hear/datas/` + `DataPreProcess/`）

### 6.1 磁盘布局与 JSON schema

```
$DATA/{tr,cv,tt}/
├── mix/<stem>.wav            16 kHz（README 说 mono，但代码 mean(-1) 期望多通道，见 §6.4）
├── mix/<stem>_meta.json      {"components":[{"type":"near"|"far", "path": ...}, ...]}  ← near 的唯一来源
├── target/<stem>.wav
├── mouths/<stem>.pt          (F,96,96) uint8/float？实际按 float32 存储时 Normalize 后仍 f32
└── ...
$INDEX/{tr,cv,tt}/
├── mix.json      [[abs_wav_path, n_samples], ...]
├── target.json   [[abs_wav_path, abs_mouth_pt_path, n_samples], ...]   # 第 3 字段从未被读取
└── near.json     [[[near_wav_path, n_samples], ...], ...]              # 与 mix 同序等长
```

- `process_lrs23.py` 写 mix/target.json；`generate_near_json.py` 读 `<mix>_meta.json` 写 near.json（无 meta →
  fallback `[[mix_path, n]]` 哨兵，Dataset 按 `path==mix_path` 跳过）。
- **对齐规则**：mix/target 各自 `listdir+sort` 后**按下标**配对，无文件名/长度校验 → 等量错名静默错配、
  不等量 Dataset 构造期 IndexError。
- **`_meta.json` 在仓库内没有生成者**（只有读取方）→ 按 README 流程（只跑两个 DataPreProcess 脚本）时
  near 全为空 → NASS 负项恒 0 / MDCL 关闭 / val 分桶全 `0near`。**这是“论文核心贡献之一在复现流程中静默失效”的 P0**。

### 6.2 样本与 batch 契约（权威，[验证]）

| 张量 | 形状/dtype | 说明 |
|---|---|---|
| mixtures / sources | `[B,32000] float32` | train 随机 2 s；val/test **整段变长** |
| mouths | `[B,50,88,88] float32` | 存储 96×96，Dataset 内 RandomCrop(train)/CenterCrop(88)(val/test) |
| near_audios | `list[list[Tensor[32000]]]`，K 可变 | K=0 表示无近场 |
| filenames | `list[str]` | |

- 帧对齐：`mouth_start = floor(rand_start/16000*25) = floor(rand_start/640)`，取 50 帧（2 s=50×640 采样），
  音频起点与嘴帧名义起点差 0–639 采样（≤39.9 ms）；**16000/25 硬编码**（8 kHz 会错位）。
- 归一化：`/255` 后 `(x−0.421)/0.165`；Studio `preprocess_mouth` 与训练 val 管线**算法一致**——
  当 `.pt` 存 float32 时逐位相同（max|Δ|=0），当 `.pt` 存 uint8 时训练侧为 float64、与 Studio float32
  在 float32 精度内等价（实测 raw |Δ|≈2.6e-7，非逐位 0.0；05 round2 复核 P1）[验证]。
- `near` 判定：只认 `_meta.json: components[].type=="near"`；仓库内**没有任何距离/SIR 阈值逻辑**；
  所有近场人都当负样本（不采样）；`min(K,4)` 分桶仅用于日志。
- DataModule：`__init__` 12 个配置键与 `configs/Falcon.yml` **12/12 匹配**（唯一缺省是 `train_samples=10000`）；
  `make_loader`/`make_sets` 都是 `@property`；val/test loader **硬编码 batch_size=1**（config 的 4 被忽略）；
  val_dataloader = `[cv,tt]`。

### 6.3 `DataPreProcess` 行为

- `process_lrs23.py` CLI：`--in_audio_dir/--out_dir`；输出 JSON 用绝对路径；不生成 meta/mouth。
- `generate_near_json.py` CLI：`--data_root/--out_root/--sample_rate`；读 meta → near.json；无 meta 时哨兵。
- `check_avspeech_dataset.py`：诊断脚本；当前环境因依赖缺（cv2 等）不可直接跑（静态可读）。
- 两个脚本对空 split/缺失目录容错很弱（空 split 直接崩），且 **README 未记录 mouth 目录名/.pt 格式、meta 生成方式**。

### 6.4 数据侧风险

1. mono wav 在 eval Dataset 中 `mean(-1)` 退化 0 维 → `torch.stft` 崩；训练 val 同路径 → 反证真实数据应为多声道
   （与 README 的 “mono 16 kHz” 矛盾，需作者/数据侧确认）。
2. near wav 采样率未校验（near.json 只存路径+采样数，按 16 kHz 解释 → 非 16k 时时间轴错位）。
3. `<stem>_meta.json` 缺失时程序只 WARNING（MDCL 静默关闭），且 “N/N samples have near-speaker entries”
   会把 fallback 计入，误导性日志。
4. 短于 2 s 的近场音频 offset 会被 clamp → 时间错位（源码注释承认）。

---

## 7. Studio 后端（服务 / 运行时 / 推理拷贝）

### 7.1 HTTP/WS 协议（已全表化，`06-studio-backend.md` A01–A68）

| 端点 | 关键约定 |
|---|---|
| `GET /api/status` | 加载前/后字段集合不同：加载后合并 runtime 的 `architecture/parameters/warmup_ms/audio_weight_keys/video_weight_keys/...`；`mode: sliding-window → windowed_audio_visual` |
| `POST /api/load` | body ≤1024 B；`{"variant":"small"|"big"}`；200 / 400（未知变体，或变体不可用/缺 huggingface_hub）/ 409（加载中或有活跃连接）/ 413（body 过大） |
| `WS /ws` 请求 | `{type:"infer", id, audio(b64 = 32000×f32 = **170668** chars), frames(50×b64 JPEG/PNG 96×96), window_seconds:2, hop_seconds:0.5, end_sample, reset}`；8 字段强校验；`end_sample ∈ [32000, 2^53−1]` |
| `WS /ws` 结果 | 13 字段：`{type:"result", id, audio(b64 = 8000×f32 = **42668** chars), sample_rate:16000, start_sample=endpoint−9280, end_sample=endpoint−1280, contiguous, inference_ms, rtf, output_delay_ms:80, ...}` |
| 错误 | 三类：① 协议/推理错误 → `type:"error"`，**不关闭连接**（server.py:268-272）；② 非同源 Origin → WS close **1008**（:235-237，握手阶段）；③ 单条消息 > `MAX_MESSAGE_BYTES`（2,000,000 chars）→ close **1009**（:245-247） |
| 安全 | `same_origin` 仅允许 hostname∈{localhost,127.0.0.1,::1}；TrustedHost 白名单仅 loopback；CSP 内联；无鉴权（本机应用） |

> 命名陷阱：请求 `end_sample` = 窗口末端（绝对采样），结果 `end_sample` = 输出块末端（= 窗末端 − 1280），
> **同名不同义**；前端 `app.js:678-681` 校验一致。

### 7.2 Engine 与窗口切片

- 状态机：`unloaded → loading → ready / error` + `busy` 瞬态；守卫：加载中禁切变体、有活跃 WS 连接禁切、
  同款加载 no-op、变体不可用 400、加载异常 409。
- `OutputSlicer`：窗口预测 32000 采样 → 输出 `pred[22720:30720]`（0.5 s，绝对 `[endpoint−9280, endpoint−1280)`）；
  `guard=1280`（80 ms 右上下文）；窗口间 `pred[30720:31040]`（20 ms/320 采样）与下一窗首样本交叉淡化；
  `contiguous` 仅当未 reset 且 `endpoint−last==8000`；两块拼接实测 max|Δ|=4.3e-8 [验证]。
- 首窗语义：客户端需 9280 采样（0.58 s）即可出第一块；`LEFT_CONTEXT=22720` 是**客户端私有协议常量**
  （服务端未编码，只校验 `end_sample≥WINDOW`）；首块从真实采集边界开始（1.42 s 静音左上下文不计入录音）。

### 7.3 FalconRuntime（真实加载已验证）

- 常量：`16 kHz / 25 fps / 32000 采样 / 50 帧`。
- `preprocess_mouth`：uint8 `(50,96,96)` → `[4:92, 4:92]` 中心裁剪 88×88 → `/255` → `(x−0.421)/0.165`
  → 形状 `(1,1,50,88,88)`，值域约 `[−2.55, 3.51]`（**不是 [−1,1]**）。
- 权重解析：显式本地路径 / `hf://<org>/<repo>/<file>` / 变体默认 HF repo（下载并缓存）。
- **strict=True**：发布权重必须恰好是 **small: 525 audio + 137 video / big: 517 audio + 137 video** keys；
  不匹配即失败（避免残缺加载）。
- 设备：`auto` = CUDA → MPS → CPU，且用**真实 warmup** 判定，GPU 失败自动回退 CPU；`FALCON_CPU_THREADS` 夹取 [1,8]。
- 真实权重实测：`status().parameters = 11,724,385`；`infer(32000 采样, 50×96×96)` → `(32000,) float32`；
  手工复算 `audio_model(zeros, video_model(preprocess_mouth(zeros)))` 与 runtime 输出 **max|Δ|=0** [验证]。

### 7.4 推理拷贝与 PROVENANCE（真实权重级结论）

- `studio/falcon_model/` 共 15 个 .py（4,035 行）+ PROVENANCE.md（42 行）。
- 与 `look2hear/` 的差异：`Falcon.py` 净 +9 行 = `DropPath` 本地化 import（替换 timm）+ **MPS CPU FFT bridge**
  （encoder `:1110-1122`、decoder `:1183-1198`：MPS 时把 STFT/iSTFT 搬 CPU）；其余为注释中译；
  `Falcon_Contrastive.py`/`videomodels` 逐字节相同（frcnn 仅删 `sys.path.append`/`resnet1D` import、CRLF→LF）。
- **PROVENANCE 声明「全部来自训练代码」不成立**：`SISNRRepel.py`（1,498 行）与 `Speaker_Contrastive_v2.py`（140 行）
  在 `look2hear/` 中**不存在**（全仓库 grep / git log 确认）→ Falcon-Big 的训练代码未发布。
- **等价性已用真实权重验证**：同一 `falcon.pt` + 同一输入，训练侧模型与 Studio 模型 eval/train 模式输出
  **全部 max|Δ|=0（bit-identical）** [验证]。

### 7.5 部署边界

- 仓库 `server.py` 绑定 127.0.0.1、`same_origin` 只接受 loopback、TrustedHost 白名单仅 loopback →
  **README 中的公网在线 demo 必然是另一份部署补丁**（改 host/origin/鉴权），仓库内代码不能直接对外。
- 前端把任何 `type:"error"`（包括后端“模型忙，请跳过此窗口”）当致命错误并停止整场会话 → 多标签页并发
  时非持锁页面会被打断（前后端契约 P1，产品口径待定）。

---

## 8. Studio 前端（`studio/web/`，非 vendor ≈2.85 k 行：JS 2,728 + HTML 64 + CSS 56；另 2 个前端测试 97 行）

### 8.1 常量与协议对齐（逐项 [验证] 一致）

| 常量 | 值 | 用途 / 对齐 |
|---|---|---|
| `RATE` | 16000 | 重采样目标；服务端 `SAMPLE_RATE` |
| `WINDOW` | 32000 | 2 s；`WINDOW_SAMPLES` |
| `HOP` | 8000 | 0.5 s；`hop_seconds` |
| `GUARD` | 1280 | 80 ms 右上下文（服务端切片保留量）；注意与 `FADE=320`（20 ms 交叉淡化）是不同常量 |
| `LEFT_CONTEXT` | **22720**（= WINDOW−HOP−GUARD，1.42 s） | 首窗左上下文；= 服务端 OutputSlicer 切片起点 |
| `FPS` | 25（50 帧） | `VIDEO_FPS/WINDOW_FRAMES` |
| `MAX_RECORDING_SECONDS` | 300 | 5 分钟可导出上限（超限只停录音，分离继续） |
| 首窗需求 | 9280 采样（0.58 s） | 与 `end_sample` 偏移、`start_sample` 计算互证 |

> 早期文档/任务书把 LEFT_CONTEXT 写成 22400 是错的；前端/后端/测试三方一致为 **22720**。

### 8.2 状态机与链路

- 生命周期：`Off → Opening → Preview → Starting → Running(追踪允许/阻断) → Draining → Off → Playback`；
  追踪子状态 `AUTO(loading/tracking/lost/searching/error)` / `MANUAL`；丢脸不暂停（用最后 crop，状态提示）。
- 音频链：`getUserMedia → AudioWorklet(falcon-capture, 63-tap 多相 FIR, 7.2 kHz 截止, 群延迟 31, 20 ms 包)
  → 8 s ring → 2 s 窗/0.5 s hop → base64 LE Float32 → WS`（**非 ArrayBuffer**）。
- 视频链：30 fps → worker 40 ms 节流 = 25 fps → `FaceLandmarker`(CPU, numFaces=2) → 96×96 灰度 JPEG q0.88
  → `frameWindow` 严格 25 fps 网格取最近帧（±160 ms）→ 50 帧。
- 播放：服务端 20 ms crossfade + 前端 63-tap/2048 相升采样 + 5 ms 淡入 + 提前 30 ms 调度 + 积压 >1.5 s 丢弃；
  reset 时重采样器清零。
- 录音/导出：Separated/Mixture 逐样本对齐（mixture 取请求快照）；WAV PCM16/16 k/mono；
  停止时 drain 最后 0.6–1.1 s（补零 + 重复末帧，8 s 超时），二次点击立即停。
- 显示：`spectrum` FFT1024/hop640、0–8 kHz、−90..0 dBFS、实时 200 列=8 s；录音细览 1–16× 缩放/键盘定位；
  worker 失败仅降级。
- i18n：4 个字典 **349 条**（去重 337）、EN↔中、占位符模板、切语言不丢状态、125 个 HTML 绑定全覆盖。
- P2 遗留：`requiresReacquire` 无生产者（UI 分支不可达）；`lastOutput/frameId/backendReachable` 未使用；
  error 致命 vs 后端的“跳过窗口”文案矛盾（§7.5）。

---

## 9. 运维 / 测试 / CI / 文档一致性

- 启动链：`Start Falcon.command → scripts/launch.sh（20 行：bootstrap → 确保权重 → exec server.py）→
  bootstrap.sh（venv 选择：`.venv` 或 `~/Library/Caches/FalconStudio/venv`；优先 `requirements-lock.txt`；
  Python 3.10–3.12 校验；`FALCON_PYTHON`/`FALCON_VENV`）→ server.py（端口 7860–7899 自动选、打开浏览器）`。
- 测试矩阵（本环境实测）：**8 个 Node 测试全 PASS（≈4.1 s）**；`node --check` 21/21；pytest 用最小 PIL stub
  可跑 **33 passed / 4 skipped**（本环境已有权重缓存时 35 passed/2 skipped；`test_server` 协议用例
  pytest 收集 22 项全过；真实 JPEG/PNG 解码未验证）；无 stub 时
  `test_server` 因缺 Pillow collection error（环境问题，非仓库缺陷）。
- CI：`.github/workflows/pages.yml` 仅在 push main（`project_page/**` 或 workflow 改动）或手动触发，权限
  `pages+id-token`，私有仓库跳过部署，artifact=project_page；自定义域名 `cslikai.cn` 依赖仓库 Settings（仓库内无 CNAME）。
- 依赖：根 requirements 基本不锁版本（timm 已声明；**缺 huggingface_hub**、缺 Comet；含未使用的 torch_optimizer 等）；
  Studio 用范围下限 + `requirements-lock.txt`（8/8 自洽，索引可达）。
- 文档偏差（21+ 条，详见 08）：2 个 P0（§10.1 P0-1/P0-3）、`best_k_models.json` 占位、项目页
  `large/Dolphin.wav` 大小写不一致、`Experiments/tensorboard_logs/` 未 gitignore、在线 demo vs same-origin 限制等。

---

## 10. 风险清单（按严重度）

### 10.1 P0 —— 会直接阻断“按 README 复现”

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| P0-1 | `audio_train.py:170` `comet_logger` 未定义 | symtable / 全文 grep | 训练命令 NameError |
| P0-2 | 三个 litmodule 硬依赖 Comet 的 `logger.experiment.log(dict)` | `av_litmodule.py:153-179` | 任何非 Comet logger → AttributeError（训练不跑完） |
| P0-3 | `Experiments/checkpoint/Falcon/conf.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 未导出 | `look2hear/models/__init__.py:1-2`、`audio_visual_test.py:129` | 评测命令 AttributeError |
| P0-4 | 评测快照 `${DATA_ROOT}`/`best_k_models.json` `${ROOT}` 未展开且无 ckpt | conf.yml、best_k_models.json | 不传 `--ckpt_path` → FileNotFoundError |
| P0-5 | 评测 Dataset 对 mono wav `mean(-1)` → 0 维 → `torch.stft` 崩（训练 val 同 dataset） | `avspeech_dataset.py:193-194`、`Falcon.py:1110` | 与 README “mono” 矛盾；首个样本即崩 |
| P0-6 | `<mix>_meta.json` 无生成者 → near 全 fallback → NASS 负项/MDCL 静默失效、val 分桶全 0near | `generate_near_json.py`、`avspeech_dataset_near.py:190-191`、`nass_contrastive_loss.py:73-74` | 论文核心贡献之一在复现流程中无效 |
| P0-7 | `concat_layers[2]` 从未被调用（5,568 参数闲置 ⇒ 实际生效 532,129） | `Falcon.py:868,879`、hook 计数、01 R3 §4.9-㉑/A51 | 疑似 level-2 融合缺失（影响容量/参数口径，待作者确认） |
| P0-8 | `Falcon.forward(near_audio)` 调不存在的 `_mdcl_loss`；对比子类 `**kwargs` 静默吞掉 near_audio | `Falcon.py:1480`、`Falcon_Contrastive.py:95` | MDCL 路线事实上不可用 |
| P0-9 | 指标 “sdr/sdr_i” 实为 −SNR/SNRi 且行/avg 符号相反；PESQ 实参传反 | `allwrapper.py:39-50,60-95` | 若论文数字来自本入口，SDR 口径错误 |

### 10.2 P1 —— 语义/契约/稳定性风险

- “causal” 声明 vs 实现的窗口级因果（§3.5）；顶层 `causal` 开关完全惰性。
- `Separator` 原地 `pop` 配置 → 同一 dict 二次构造 537,697→49,081 静默残缺；新 dump conf.yml 被污染。
- 评测 `strict=False` 加载 audio 权重（可能静默残缺）；Studio 用 `strict=True`（正确）。
- `FRCNNVideoModel.eval()` 返回 None（`train()` 无 return self）。
- DataModule/JSON 按下标对齐、无文件名校验；near wav 采样率未校验；mouth 帧数不足 collate crash；
  `16000/25` 硬编码。
- `best_k_models.json` 是发布占位快照（indent=2/`${ROOT}`），非当前代码产物；`best_model.pth` 路径逻辑。
- PL 2.x 断档（accelerator/gpus 参数）；DDP per-bucket all_gather 可能死锁。
- Studio：`FalconRuntime` 只支持 2 s 窗口（模型无流式状态）；前端把 error 当致命；在线 demo 部署补丁不可得；
  `SISNRRepel/Speaker_Contrastive_v2` 训练源未发布（PROVENANCE 表述需修正）。
- `root requirements` 缺 huggingface_hub（`hf://` 评测需要）；Comet 依赖缺失。
- 训练/评测/Studio 的 mouth 约定链（96 存储→88 裁剪→z-score）在 README 中未完整记录。

### 10.3 P2 —— 死代码/可维护性（不影响理解主线）

- `look2hear/layers` 大量零调用（stft/cnnlayers/rnnlayers/enc_dec/stft_tfgn）、`utils/stft.py` 同源复制；
  `layers/__init__` 全量导入使 librosa/torch_complex/typeguard 成为硬依赖（Py3.12 的 distutils 移除风险）。
- `normalizations.py` 同名类定义两次、别名语义反直觉（LN≠累积、cLN=累积）。
- 死配置键（`training.system/parallel`、`datamodule.data_name`、`video_params.layers/stride`、`fusion_type`、
  `mask_generator_type`）；`nets_utils.py` 502 行训练链 0 调用；`DPRNNLinear.forward` 的 `transpose(1,1)` no-op；
  `export_jit` 必然抛错；`SISNRi` NameError。
- 未用变量/字段（前端 `lastOutput/frameId/backendReachable`、`requiresReacquire` 死分支）；项目页媒体大小写。

---

## 11. 未解问题（需作者/线上/数据侧确认；详见各文档 §7 与评审 Q 表）

| # | 问题 | 谁可回答 |
|---|---|---|
| Q1 | 论文 SDR/SI-SNRi 的实际计算路径与列名？是否来自本仓库 `ALLMetricsTracker`？ | 作者 |
| Q2 | `_meta.json` 的真实生成工具、mouth `.pt` 的精确格式（H/W/C/fps/dtype）、RealSSA 的 near 距离/SIR 阈值 | 作者/数据侧 |
| Q3 | 真实训练数据 wav 是多声道还是 mono？（代码要求多通道，README 说 mono） | 数据侧 |
| Q4 | `concat_layers[2]` 是有意跳过还是 forward 漏调用？影响 0.5 M 口径与容量 | 作者 |
| Q5 | 顶层 `causal` 开关为何完全惰性？论文 “causal” 的正式定义（帧级/窗口级） | 作者 |
| Q6 | `comet_logger`/Comet 日志的原始意图（是否遗漏 `CometLogger` 与依赖声明） | 作者 |
| Q7 | `Experiments/checkpoint/Falcon/conf.yml` 的 `SpeakerContrastiveV2` 是笔误还是另一分支快照？发布评测应以哪份 conf 为准 | 作者 |
| Q8 | Falcon-Big（`SpeakerContrastiveV2`/`SISNRRepel`）训练代码是否会发布？PROVENANCE 应如何修正 | 作者 |
| Q9 | 在线 demo 的部署补丁（host/origin/鉴权）是否公开？ | 作者/线上 |
| Q10 | `train_samples=10000` 是否有意（1 epoch ≠ 全量数据）？`use_distributed_sampler` 改动后的 epoch 语义 | 作者 |
| Q11 | `FALCON_CPU_THREADS`、`OutputSlicer` 的 FADE/guard 参数是否在论文 latency 口径中说明 | 作者 |
| Q12 | 训练/评测的随机种子与可复现性（代码未见全局 seed 设定） | 作者 |

---

## 12. 关键数字总表（均可复算/已 [验证]）

| 数字 | 值 | 来源/验证 |
|---|---|---|
| 音频网络参数 | 537,697（其中 5,568 闲置 ⇒ 实际生效 **532,129**；扣对比头 523,113） | 构建 + `parameters()` 统计；01 R3 A51 |
| state_dict numel vs 参数 | audio 537,762 / 537,697；video 11,196,308 / 11,186,688（差值为 BN buffer） | 01 R3 §8.4 [验证] |
| 冻结 FRCNN 参数 | 11,186,688 | 同上 |
| 总参数（small） | 11,724,385 | 与 README 11.72M 一致 |
| Falcon-Big 音频/总参数 | 1,158,617 / 12,345,305 | 真实 falcon-big.pt strict 加载 |
| state_dict keys | small: 525 audio + 137 video；big: 517 + 137 | 真实权重 |
| 发布权重顶层 | `{state_dict, architecture, epoch}`；epoch 714/715 | `torch.load` |
| 帧数/频点 | T'=1+L/128（32000→251）；F=129 | 前向实测 |
| 窗口/hop/guard | 32000 / 8000 / 1280（80 ms） | 前后端常量 |
| 左上下文/首窗 | 22720（1.42 s）/ 9280（0.58 s） | `LEFT_CONTEXT=WINDOW-HOP-GUARD` |
| 输出块 | 8000 采样（0.5 s）= pred[22720:30720] | OutputSlicer |
| WS payload | 请求 audio 170,668 chars；结果 42,668 chars | b64 长度实测 |
| 训练超参 | AdamW lr 1e-3, wd 0.01, ReduceLROnPlateau(15,0.5), EarlyStopping(30), grad clip 5, epochs 1000, batch 4×N GPU, train_samples 10000 | 配置+装配链 |
| 损失 | PIT(SNR train / SI-SDR val) + NASS contrastive β 0→1.0/200 ep | 代码+数值 |
| MACs | 10.611 G（仅 audio） | ptflops 复算与 README 10.6G 一致 |
| 因果 lookahead | STFT 8 ms；AudioBlock ≈456 ms；cLNhw 全窗；端到端 −649 ms | 多协议探针 |
| Studio 测试 | 8 Node PASS；pytest 33 passed/4 skipped（PIL stub）；test_server 协议用例收集 22 项全过 | 实测 |
| i18n | 349 条（337 去重） | 字典统计+测试 |
| 推理耗时 | CPU(本沙箱, 真实权重) warmup ≈2.47 s；单窗推理 ≈0.54–0.74 s（合成/随机权重实测，含预处理）/ M5 Pro Metal ~0.12 s（README） | 实测/文档 |

---

## 13. 术语表

| 术语 | 含义 |
|---|---|
| AVTSE | Audio-Visual Target Speaker Extraction，音视频目标说话人提取 |
| RealSSA | 论文提出的真实场景模拟 benchmark（2–6 动态近/远场说话人，四类场景 + 真实录制 split） |
| causal / window-causal | 本文档口径：结构单向 + 2 s 窗口内整窗计算；非逐采样流式因果 |
| TFRB | Time-Frequency Refinement Block；代码 = DualPathRNN(dim4, freq, biLSTM) + DualPathRNN(dim3, time, uniLSTM) + MHSA2D(tril mask) |
| GCI | Gated Cross-scale Injection；代码 = InjectionMultiSum（nearest 上采样 + 门控融合） |
| cLNhw | `LayerNormalizationHW` = 全窗 GroupNorm(1,C) + 沿时间累积 LN(CumulateLN4D) |
| PIT | Permutation Invariant Training |
| NASS | NASS 风格 same-space speaker contrastive（encoder 特征空间对比） |
| MDCL | Magnitude-Domain Contrastive Loss（硬负样本、幅度谱 SI-SNR 推离），本仓库未接入 |
| near/far | `_meta.json` 中 `components[].type`；near = 近场干扰人（对比学习负样本） |
| window/hop/guard | 2 s 推理窗 / 0.5 s 输出步长 / 80 ms 右上下文 |
| Left context | 新目标开始时的 1.42 s 静音左上下文（使首块从真实采集边界开始） |
| contiguous | 服务端标记：本块与上一块是否在时间上连续（否则复位/不拼接） |
| OutputSlicer | 服务端窗口输出切片器（chunk + guard + crossfade） |
| PROVENANCE | Studio 推理拷贝的来源说明（声称来自训练代码，实际 Big 相关文件无训练源） |

---

## 14. 如何“使用/复现/继续深入”

### 14.1 想跑 Studio（推荐，已具备条件）

```bash
cd studio && ./scripts/launch.sh          # 自动选 Metal/CUDA/CPU，下载权重，打开浏览器
# 或只做离线推理：
python infer_file.py --audio mix.wav --mouth mouth.npy --out target.wav --model JusperLee/Falcon
# 或本沙箱已验证的方式（真实权重 + CPU）：
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/miniforge3/envs/falcon/bin/python - <<'PY'
from falcon_runtime import FalconRuntime
rt = FalconRuntime("small", "/root/.cache/falcon_weights/falcon.pt").load("cpu")
print(rt.status()["parameters"], rt.infer(__import__("numpy").zeros(32000, "float32"),
      __import__("numpy").full((50, 96, 96), 107, "uint8")).shape)
PY
```

### 14.2 想跑训练/评测（需先修 P0）

1. 补 `comet_logger`（改 CSVLogger/TensorBoardLogger 并把 litmodule 的 `logger.experiment.log(dict)` 换成
   `self.log_dict(...)`，或在 requirements 加 comet-ml 并定义 `CometLogger`）。
2. 把 `conf.yml` 的 `audionet_name` 改为 `FalconSpeakerContrastive`，展开 `${DATA_ROOT}`/`${ROOT}`，
   显式传 `--ckpt_path`（如 `hf://JusperLee/Falcon/falcon.pt` 或本地 `falcon.pt`，已验证含 `state_dict`）。
3. 数据侧：按真实数据通道数修正 eval Dataset 的 `mean(-1)`（加 ndim 守卫）；补齐 `<mix>_meta.json` 与
   near 生成工具；确认 mouth `.pt` 格式/帧对齐。
4. 注意 conf.yml 若由当前代码重新 dump 会缺 `repeats/shared`（模型退化 49,081 参数）→ 用带这两个键的原始
   `configs/Falcon.yml` 评测，不要用新 dump 的快照。

### 14.3 想继续深入/验证

- 子系统深度文档：`docs/understanding/01..08-*.md`（每份 700–1,250 行，含断言表与未解疑问）。
- 评审记录：`docs/reviews/round1/`（8 份交叉评审）、`docs/reviews/round2/`（复核）。
- 验证资产：`docs/verification/lead-verification.md`（Lead 的模型/因果/真实权重验证）、
  `docs/verification/scripts/`（20+ 可复跑脚本）、`docs/verification/runs/`（日志）。
- 运行环境（强制）：**专属 conda 环境 `falcon`**（`/root/miniforge3/envs/falcon`，Python 3.11），
  详见 [AGENTS.md](../../AGENTS.md) 与 `docs/understanding/VERIFY-ENV.md`；历史日志中的 `/root/.venvs/falcon` 已删除。

---

## 15. 验证与轮次状态（诚实记录）

| 文档 | R1 | R2 交叉评审 | R3 修订 | R4 复核 |
|---|---|---|---|---|
| 01-model-core | ✅ 807 行 | ✅ P0=1/P1=3（concat[2]、A21、A44…） | ✅ **975 行**，sha256 `afe00188…db0058`（§9.1 逐条响应；A21 拆 a/b/c、§4.6-⑨ 五协议表、§4.10 TFRB 映射、A46–A56） | ✅ R4 增量复核**通过**（P0=0/P1=0，残留 P2×1/P3×3；真实权重 bit-identical、A18 公式三点命中） |
| 02-layers-dsp | ✅ 757 行 | ✅ P0=1/P1=4（A36 因果、lookahead…） | ✅ **895 行，sha256 `1b6def62…92afdd`**（A01–A64：A36′、A59 ≤127 采样、A60 协议表、A61 PL 归因、A63 真实权重） | ✅ 已复核（train-sys R4 + 05 round2 复核；PL 归因已修正） |
| 03-training-system | ✅ | ✅ P0=0/P1=3/P2=10 | ✅ 700 行（§9.1 逐条响应 + §4.8 执行级验证） | ⏳ round2 复核未执行（其 R3 已含 15 项执行级验证；结论已纳入 FINAL） |
| 04-losses-metrics-video | ✅ 1,056 行 | ✅ P0=0/P1=6/P2=8 | ✅ v2 1,191 行（sha256 `4bfb41b7…ccbda9`；真实权重 + eval ckpt 回环验证） | ✅ round2 复核通过（P0=0/P1=1：A38 断言写反、`torch_utils` 行号越界等文档级残留） |
| 05-data-pipeline | ✅ 797 行 | ✅ P0=1/P1=4/P2=9 | ✅ **912 行**，sha256 `78ff36d9…8928914`（R3.1：A18 降级、A60 按 dtype 条件化、边界补齐；`_meta.json` 升 P0 主结论；真实权重整句评测 A68） | ✅ round2 复核通过（P0=0/P1=2，已在 R3.1 全部修复） |
| 06-studio-backend | ✅ 1,029 行 | ✅ P0=0/P1=1/P2=9 | ✅ v3 1,252 行 | ✅ round2（协议专项，reviewer=studio-frontend）P0=P1=0 → v3 后 P0–P3=0；Q1/P10/P11 已由真实权重升级为“已验证” |
| 07-studio-frontend | ✅ 1,100 行 | ✅ P0=0/P1=3/P2=6 | ✅ v2 1,158 行 | ✅ 作者 R4 自校（sessionEpoch 修正/引用自检，无独立 round2 文件） |
| 08-ops-tests-docs | ✅ 718 行 | ✅ P0=0（作者 2 P0 复核成立）/P1=2/P2=7 | ✅ v2 768 行（sha256 74f90d25…） | ✅ round2 复核通过（P0=0/P1=1：两处引用行号偏移；pytest 33/35 双口径复现、node 21/21、D22 独立复现） |

> 截至本文件定稿：R1（8 份文档）+ R2（8 份交叉评审）+ R3（**8/8 全部完成**）+ R4 round2 复核（01/02/04/05/06/07/08 已完成；03 未单独跑 round2，但其 R3 自带 15 项执行级验证）已完成；
> 残余仅为文档级小修（04 的 A38 断言与 torch_utils 行号、08 的两处引用行号），均已在 round2 评审核查表中记录，不影响任何技术结论；
> 05 的对应问题已在 R3.1 修复（sha256 `78ff36d9…`，`verification/runs/data_pipe/05-data-pipeline-r3.sha256` 可校验）。
> 全部评审文件与执行日志在 `docs/reviews/round2/`、`docs/verification/runs/`。
> Lead 的独立验证（本文件 §3–§9 中标 [验证] 的数字）不受上述未完成轮次影响：
> 参数/形状/因果/真实权重/协议/测试均为实际运行结果，脚本与日志可复跑。

---

## 17. 上游更新与数据可用性（2026-10-21，同步到 `689555c`）

### 17.1 上游 4 个新提交

| commit | 内容 | 对本分析的影响 |
|---|---|---|
| `607352c` | Studio：设备启动超时上限、采集卡顿诊断（`window.falconDiagnostics()`、`performanceHint`） | 07 文档 §2/§8 的部分实现细节已更新；常量和协议未变；我们重跑 `pytest studio/tests` → 35 passed/2 skipped，Node 8/8 |
| `a27b1e6` | 新增 `space/`（HF ZeroGPU Space，record-then-separate，复用 `studio/` 运行时） | 项目新增第三个交付物（推理/演示），不参与训练；其测试需要独立环境（`spaces`/`gradio`，py3.12/torch2.13） |
| `8188b91` | README：链接已发布的 RealSSA 数据集（RealSSA-Sim / RealSSA-Real） | 数据可得性变化，见 §17.2 |
| `689555c` | 中文 README 同步 + RealSSA 数据说明 | 同上 |

### 17.2 数据可用性核实（HF API，2026-10-21）

| HF 仓库 | gated | 内容/状态 |
|---|---|---|
| `JusperLee/RealSSA-Sim` | `manual`（人工审批） | 242 文件；`tr/*.tar` 202 + `cv` 19 + `tt` 19 shards，约 522 GB；每条 4ch 16k mix/target + `[T,96,96,3]` uint8 RGB mouths + 场景 JSON |
| `JusperLee/RealSSA-Real` | `manual` | 422 文件；54 段/40 录音/10 说话人/8 场景，stereo 16k + 224×224 25fps 人脸视频 + mouth NPZ + landmarks + enrollment + 逐录音转录（**无干净 target，只能 DNSMOS/WER/相似度**） |
| `JusperLee/LRS2-2Mix` | 公开 | `lrs2.tar.gz` 16.47 GB（`lrs2_rebuild/`：混合脚本 + 说话人列表 + 音频；**不含视觉**——需要 LRS2 原始视频自建嘴部 crop） |
| `JusperLee/LRS3-2Mix` | 公开 | **仅 README，无数据文件** |
| `JusperLee/VoxCeleb2-2Mix` | 公开 | **仅 README，无数据文件** |

关键限制（来自 dataset card，与我们的 P0 结论一致）：
- **RealSSA-Sim 不发布 near 干扰源波形**；`components[].path` 只是生成时来源路径；官方明说 “the original speaker-contrastive
  loss needs valid near-speaker negatives; the indexing script's fallback to the mixture is not an equivalent reproduction”。
  → NASS 对比损失的精确训练复现**在官方数据下不可行**，只能替代负样本或近似重建（见 `docs/REPRODUCTION_PLAN.md` §5.1）。
- RealSSA-Real 的 enrollment 音频不是时间对齐的干净 target → 不能算 SI-SDR。
- 仓库无 `_meta.json` 生成器、无嘴部 crop 提取流水线（LRS 系列需要）；`pretrain_zoo` 骨干权重链接仍是 README TODO，
  但**可以从 `falcon.pt` 的冻结 `video_model.*` 导出**（训练时视觉分支冻结，所以发布的权重就是该骨干）。

### 17.3 论文状态

- README 明确：**“Paper link will be added when available”**；arXiv/OpenAlex/Semantic Scholar 检索均无该论文；项目页无 paper 链接。
- 因此评测协议细节（通道选择、切段、DNSMOS/ASR/声纹模型版本）与消融设置只能依据代码/README/dataset card 推断，
  复现报告需要显式列出这些假设（见 `docs/REPRODUCTION_PLAN.md` §9）。
- **复现进展（2026-10-22）**：Phase 1（复现前置修复）已完成——训练/评测入口可跑、指标修正、mono 修复、
  视觉骨干可导出、官方权重数值指纹不变；并已在真实数据上跑通首个 benchmark：
  **LRS2-2Mix test（3000 条，官方 `falcon.pt`）SI-SNRi 7.03 / SDRi 7.71**（20 条子集 7.91/8.55；target=s1 约定已确证），
  与论文 14.1/14.7 仍有 ~7 dB 差距，需 Phase 2 排查（训练集匹配度/评测协议/嘴部对齐）。
  详见 [`docs/REPRODUCTION_PHASE1_REPORT.md`](../REPRODUCTION_PHASE1_REPORT.md)。
- **Phase 2 结论（2026-10-22）**：官方 `falcon.pt` 在 LRS2-2Mix 全量 test 的 **7.03/7.71 属“跨语料”结果**
  （训练 datamodule 为 LRS3_complex_nonmoving；README 明示论文表格按 benchmark 训练），
  协议/指标/对齐变体最多解释 ~0.4 dB、falcon-big 仅 +0.97 dB、Studio 窗口化 −1.9 dB；
  视觉分支贡献 +4.8~+15 dB（置零 mouth 后显著下降）。要复现论文 LRS2 列需按 LRS2 配方训练（进行中）。
  LRS2-2Mix 已全量转换（28,000 条，target=s1），单卡训练已启动；RealSSA 访问申请已提交待批。
  详见 [`docs/REPRODUCTION_PHASE2_REPORT.md`](../REPRODUCTION_PHASE2_REPORT.md)。
- 详细的复现步骤、缺口清单、分阶段计划与验收标准：**[`docs/REPRODUCTION_PLAN.md`](../REPRODUCTION_PLAN.md)**。


## 16. 一句话结论

**Falcon 是一个“小模型 + 冻结唇读前端 + 单次多尺度分离”的高效 AVTSE 研究工程：模型与 Studio 推理链路
真实可跑且两套代码逐位等价，但训练/评测入口按发布状态不可直接复现（Comet 日志与快照配置 P0），
“causal” 应理解为窗口级因果，数据管线与指标口径存在多处静默失效陷阱——这些正是深入理解本项目后
最需要向作者反馈/修复的部分。**
