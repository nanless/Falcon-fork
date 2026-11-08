# Falcon-fork 复现状态（STATUS · 单一事实源）

> 最后更新 2026-11-08 · 分支 `main` · HEAD `6c1d5b7`（tag `repro-freeze-20261031` 指向该提交）· 环境 conda `falcon`
> 本文件回答三个问题：哪几层复现了、每个数字的口径是什么、接下来做什么。历史报告只作过程记录，与本文冲突时以本文为准。
> 数字来源限定为本轮深挖报告 [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) 与 01–12 号报告；记忆和旧稿里的数字不再引用。

## 1 复现现状（L1–L4）

**按 L1–L4 四层算，L1 完全复现；L2 只有 RealSSA-Sim 数值吻合（实测 7.774 / 8.479 vs 论文 7.7 / 8.2，+0.07 / +0.28），但 ckpt 语料与评测通道两个变量同时不同，不构成严格复现；L2 其余 benchmark 与 L3、L4 未完成。**

- **L1 结构与权重：已复现（实测）**。音频参数 **537,697**；`falcon.pt` strict 加载零 missing；训练侧与 Studio 同输入 `max|Δ|=0`；MACs **10.611 G**（ptflops 口径，含 LSTM、不含视觉）。
- **L2 评测：只有 RealSSA-Sim 数值吻合，非同协议，不构成严格复现（实测）**。SI-SNRi / SDRi 为 **7.774 / 8.479**（四通道均值，9,169 条；SDRi 分母 9,166）；论文 7.7 / 8.2 用的是 RealSSA-Sim-trained ckpt + channel 1。官方 `falcon.pt` 训在 `LRS3_complex_nonmoving`（ep714），两个变量同时不同；ch0 复评 **7.977 / 8.452** 说明通道影响 <0.3 dB，仍不能消除 ckpt 差异。DNSMOS **2.7941 / 3.3365** 有数，与论文 3.276 同理不可比。
- **L2 未复现**：LRS2-2Mix 论文列 **14.1 / 14.7**。官方 `falcon.pt` 训在 LRS3，跨语料只有 **7.029 / 7.713**；单卡自训最好 **8.858 / 9.270**（ep124，旧配方，非论文 8 卡配方）。
- **L3 训练：未完成**。训练入口可跑（CPU 2 epoch 实测通过），论文的 8×3090 / 1000 epoch 配方没有在本机跑过。
- **L4 消融与扩展数据：空缺**。论文 Table 7/8/9/18/19/21 六组消融没有一组按论文协议完成（Table 21 的视觉缺失只有 mouth 置零 / 打乱的部分实验）；LRS3-2Mix、VoxCeleb2-2Mix 数据未发布。

任务本身是 **AVTSE 单目标说话人提取**：给混合语音和目标说话人的唇部视频，输出单路目标语音。6 份配置都是 `n_src: 1`，数据索引只有 `mix.json` + `target.json`。benchmark 名里的 "-2Mix" 指两说话人混合、只提一路，不是 2 源分离。`n_src=2` 在代码上跑不通：实测前向直接报 `RuntimeError: shape '[2, 224, 251, 129]' is invalid`（`MaskGenerator` 的通道账与 `decoder.in_chan=encoder_out_chan*n_src` 不自洽）。

层级定义：L1 = 结构 / 权重 / 参数；L2 = 评测指标；L3 = 训练复现；L4 = 消融与论文扩展 benchmark。

全文的结论标签按证据强度分三种：**实测**是本次在仓库或本机跑出来的；**推断**是由代码、日志或等价条件推出的；**未验证**是只有文档记载、没有原始测量文件的。三种标签不互相升级。

## 2 仓库与环境

### 2.1 代码与提交

| 项 | 值 |
|---|---|
| 仓库 | `dev_a800_2gpus:/root/code/github_repos/Falcon-fork` |
| 分支 / 远端 | `main`；`origin = nanless/Falcon-fork`，`upstream = JusperLee/Falcon`（已合并上游 `439cbe9`） |
| HEAD | `6c1d5b7`（2026-10-31 06:57 UTC） |
| tag | `repro-freeze-20261031`，指向 `6c1d5b7` |
| 工作区 | 有未提交改动（含本文件重写）；`docs/understanding/deepdive-20261108/` 是未跟踪的报告目录 |
| 规模 | 相对 upstream：396 文件 / +107,000 / −523 行，其中 `docs/verification` 占 72,118 行插入（67%）；真正运行的模型代码改动是 `look2hear/** +693/−196` |

### 2.2 运行环境

| 项 | 值 |
|---|---|
| 解释器 | `/root/miniforge3/envs/falcon/bin/python` |
| Python | 3.11.17 |
| torch / torchaudio / torchvision | 2.8.0+cu128 / 2.8.0+cu128 / 0.23.0+cu128 |
| pytorch-lightning | 2.6.6 |
| typeguard | 2.13.3（确实导出 `check_argument_types`，`docs/verification/stubs/typeguard.py` 的理由已失效） |
| GPU | 2×A800-80G；延迟类数字只有 CPU 实测，A800 的 GPU 数字是引用值（§4.6） |

环境漂移结论（实测）：115 个包的名称与版本和活环境 `pip freeze` 完全一致；`test_lock_portable.py` 通过。

环境里有四处与旧文档不一致，跑之前会碰到：

- `docs/verification/stubs/typeguard.py` 的 docstring 理由已失效：活环境的 typeguard 2.13.3 确实导出 `check_argument_types`。
- `espnet` 缺失；`pypesq.py` 是 792 B 的本地 shim，不在 lock 里（非发行包）。
- `torch.stft` 在 CPU fp16 下直接崩（MKL FFT 的 Half 不支持），所以 `model.half()` 的推理路径不可用。
- `hparams.yaml` 含 5 处 `!!python` 标签（torch.Tensor），`yaml.safe_load` 会失败；`conf.yml` 里 `main_args` 恒空，`--conf_dir/--ckpt_path` 落错 argparse 组（`parser_utils.py:143-148`）。

### 2.3 依赖锁

`docs/verification/conda-env-lock.txt` 物理 **124 行** = 9 行注释 + **115 个包**（无空行，文末有换行）。

包名和版本与 `pip freeze` 一一对上，但 lock 把 **18 个 `file:///root/wheelhouse/...` 来源改写成了 `pkg==version`**：来源 URL 与 sha256 只留在活环境的 `pip freeze` 输出里，lock 文件里没有。后果是 `pip install -r` 能装出同名同版本，**不能保证装出同一份二进制**（推断；18 个包集中在 `torch` / `pytorch-lightning` / `triton` / `nvidia-*`）。可移植化改写由 `test_lock_portable.py:82-95` 用 `importlib.metadata.version()` 归一后放行，测试守住的是"名字+版本"，不是"二进制"。

### 2.4 权重

| 文件 | 架构 | epoch | key 数 | 参数量 |
|---|---|---|---|---|
| `/root/.cache/falcon_weights/falcon.pt` | `FalconSpeakerContrastive`（small） | 714 | 662（525 audio + 137 video） | 音频 537,697 |
| `/root/.cache/falcon_weights/falcon-big.pt` | `SpeakerContrastiveV2`（big） | 715 | 654（517 audio + 137 video） | 音频 1,158,617 |

small 与 big 是两套结构，不是同一模型的两个尺寸：big 少 8 个 key，正是 `process_f/t_layers.*.out_proj.weight`，因为 big 的 `hid_chan×方向数 == in_chan` 使 `out_proj` 退化成 `Identity`。`SpeakerContrastiveV2` 只存在于 `studio/falcon_model/`，训练侧没有，所以 big 无法与训练侧交叉验证。Studio 不从 `frcnn_128_512.backbone.pth.tar` 预训练，视觉权重全部来自 ckpt 的 `video_model.*`。

### 2.5 数据

| 数据集 | 路径 | 单目录 du | 条数 tr/cv/tt | 备注 |
|---|---|---|---|---|
| LRS2-2Mix 源包 | `/root/data/lrs2_2mix` | 17 G | 20000 / 5000 / 3000 | 旧稿写 33 G，偏大；`mouths/*.npz` 20,392 个，`[50,96,96]` |
| Falcon LRS2（单 target） | `/root/data/falcon_lrs2_2mix` | 24 G | 20000 / 5000 / 3000 | 其中 20 G 是与 paper 共享的硬链接；`near.json` 100% 为 K=1 |
| Falcon LRS2（论文配方） | `/root/data/falcon_lrs2_2mix_paper` | 26 G | 40000 / 10000 / 6000 | 20000/5000/3000 mixture × 双向；无 `mix/`、`target/` 目录 |
| RealSSA-Sim（mean） | `/root/data/falcon_realssa_sim` | 30 G | — / — / 9169 | `mouths` 28 G 与 ch0 共享；四通道均值是本项目口径，非同协议（§4.10） |
| RealSSA-Sim（ch0） | `/root/data/falcon_realssa_sim_ch0` | 30 G | — / — / 9169 | 与 mean 共享 mouths，实际独增约 2.7 G；ch0 对齐论文 channel 1，ckpt 轴仍不同，数值吻合但不构成严格复现 |
| RealSSA-Sim 发布包 | `/root/data/RealSSA-Sim` | 38 G | 仅 tt | 19/19 shard 已下载；tr 202 / cv 19 未下载；本行只记数据位置，发布包本身不构成严格复现证据 |
| RealSSA-Real | `/root/data/RealSSA-Real` | 12 G | 54 条 inference / 34 条 original | reference-free，走 DNSMOS |

上表前四个 Falcon 目录的 `mouths/` 是硬链接：逐目录相加 110 G，`du -sch` 一次去重后的真实并集是 **62 G**。容量规划和迁移按 62 G 算，按 110 G 会高估 48 G。

### 2.6 三条产品线

| 线 | 路径 | 做什么 | 关键事实 |
|---|---|---|---|
| 训练 / 评测 | `look2hear/**`、`audio_train.py`、`audio_visual_test.py`、`configs/**` | 上游代码 + 复现改造 | 评测整段一次前向，不切窗 |
| Studio | `studio/**` | 实时演示：麦克风 + 摄像头 → 2 s 窗 → 0.5 s hop | 单进程 FastAPI，只绑 127.0.0.1，无鉴权 / 限流 / TLS |
| Space | `space/**`、`project_page/**` | HF ZeroGPU：录完整段一次推理 | 服务端零复用（`server.py` 不进 Space），没有实时能力 |

Space 复用 Studio 运行时是字节级的：`build.py` 直接拷 `studio/falcon_runtime.py`、`falcon_model/`、`configs/`、`web/`，`test_static.py:40-41` 断言 `falcon_runtime.py` 与 `Falcon.yml` 和 studio 版字节相同。构建产物 54 个文件、约 39.87 MB（换一次构建差几十到上百字节，来自被拷入的 `space/README.md` 与文件时间戳），其中 `web/vendor/mediapipe` 占 39.38 MB（98.8%），三个 `.wasm` 合计 34.47 MB。

### 2.7 关键代码位置

| 对象 | 路径 | 说明 |
|---|---|---|
| 模型定义 | `look2hear/models/Falcon.py` | `AudioBlock`、`Separator`、`MaskGenerator`、`STFTEncoder/Decoder` 都在这一个文件 |
| 层与归一化 | `look2hear/layers/normalizations.py` | `cLNhw = GlobalLayerNorm(C,T,F) ∘ CumulateLN4D`；`layers/` 45 个类只用到 6 个 |
| 视频前端 | `look2hear/models/frcnn_videomodel.py` | `frontend3D` 的 `Conv3d(k=(5,7,7), pad=(2,3,3))`；`train()` 已 `return self` |
| 数据 | `look2hear/datas/avspeech_dataset_near.py` | `__len__ = train_samples`、`idx` 默认被丢弃；`mouth_start = int(rand_start / sample_rate * mouth_fps)`，默认 16000 Hz / 25.0 fps（等价旧行为），采样率不是 16000 时告警 |
| 训练 | `audio_train.py`、`look2hear/system/av_litmodule_nass_v2.py` | 入口可跑；v2 模块负责 −SI-SDR 与 τ |
| 评测 | `audio_visual_test.py`、`tools/repro/phase2/eval_realssa_real.py` | 前者整段一次前向；后者窗口化 DNSMOS |
| Studio | `studio/server.py`（419 行）、`studio/falcon_runtime.py` | 单进程 FastAPI；`GUARD=1280`、`HOP=8000` |
| Space | `space/build.py`（80 行） | 拷 4 样、覆盖 3 个、注入 2 行 |

## 3 复现矩阵（带口径）

| 层 / Benchmark | 论文 | 实测 | 口径（ckpt / 数据 / 条数 / 算法） | 结论 | 证据 |
|---|---|---|---|---|---|
| L1 参数 | 0.5 M | 音频 **537,697**；冻结 FRCNN 11,186,688；端到端 **11,724,385** | 全部 requires_grad；扣死参数 5,568 → 532,129；再扣对比头 14,584 → 517,545 | 精确复现 | `verification/scripts/verify_model_params.py` |
| L1 MACs | 10.6 G | 音频 **10.611 G** | ptflops，含 LSTM，不含 FRCNN，2 s / 50 帧 | 同尺精确复现 | `evaluated_mac_params.py` |
| L1 权重加载 | — | strict 零 missing；训练侧 vs Studio `max\|Δ\|=0` | `falcon.pt` 525 audio keys | 复现 | `verify_real_weights.py`、`verify_r3_studio_weights.py` |
| RealSSA-Sim SI-SNRi / SDRi（mean） | 7.7 / 8.2 | **7.774 / 8.479** | 官方 `falcon.pt`（`LRS3_complex_nonmoving`，ep714）；RealSSA-Sim tt 四通道均值；整段一次前向，9,169 条；SDRi 分母 9,166 | **⚠️ 数值吻合（非同协议）**，+0.07 / +0.28；ckpt 与通道两轴同时不同，不构成严格复现 | `P2E-realssa-sim-metrics_combined.csv` |
| RealSSA-Sim（ch0，论文 channel-1） | 7.7 / 8.2 | **7.977 / 8.452** | 同上，`--channel 0`；SDRi 分母 9,166 | **⚠️ 数值吻合（非同协议）**，通道影响 <0.3 dB（SI-SNRi +0.20、SDRi −0.03）；ckpt 轴仍不同，不构成严格复现 | `P2E-realssa-sim-ch0-metrics_combined-20261030.csv` |
| RealSSA-Real DNSMOS | 3.276（P.835） | OVRL **2.7941** / P808 **3.3365** | 官方 `falcon.pt`（LRS3 训练），54 段，full4 | 不可比，非复现失败 | `DNSMOS-realssa-real-mean-full4-20261031/` |
| LRS2-2Mix 官方权重（跨语料） | 14.1 / 14.7 | **7.029 / 7.713** | 官方 `falcon.pt`（LRS3 训练，ep714），tt 3,000 | 跨语料基线，不是论文列 | `P2D-eval-falcon-full.csv` |
| LRS2-2Mix 自训 ep99（best val） | — | 8.726 / 9.122 | 单卡 100 epoch，旧配方（−SNR），val 13.2014 | 中间快照 | `P2D-eval-epoch99.csv` |
| LRS2-2Mix 自训 ep124（+s2 near 微调） | — | **8.858 / 9.270** | 单卡 130 epoch，旧配方；距论文约 5.2 / 5.4 dB（14.1−8.858 / 14.7−9.270） | 本机最好，非论文配方 | `P2D-eval-epoch124-contrastive.csv` |
| mix 基线（参考） | — | 4.9937 dB | tt 3,000 的 SI-SDR(mix, s1) 均值 | 上下文，不是模型结果 | `P2A-report.md` |
| LRS2-2Mix 论文列 | 14.1 / 14.7 | 无 | 缺 LRS2-trained ckpt；8 卡配方未跑 | **不可复现（当前）** | §6 待办 4 |
| LRS3-2Mix | 16.4 / 16.7 | 无数据 | HF 仓库仅 README | **不可复现** | `00-SYNTHESIS.md` §7 |
| VoxCeleb2-2Mix | 10.5 / 11.8 | 无数据 | 同上 | **不可复现** | 同上 |
| 效率（论文 3090） | 51.3 ms / 125.3 MB / 364.7 MB | A800 引用 47.1 / 56.5 ms | 无原始测量文件；硬件不同 | **未验证**，不可并列 | §4.6 |
| 消融 Table 7/8/9/18/19/21 | 6 组 | 0 组按论文协议完成 | 只有 big/small 与 mouth 置零 / 打乱（N=20）部分 | **未完成** | `11-paper-alignment.md` §2.9 |

口径列的含义：同一行里的数只有在这一列的条件下才成立。RealSSA-Sim 两行的 SI-SNRi 分母是 9,169，SDRi / PESQ 分母是 9,166（原因见 §4.5）；DNSMOS 行的 ckpt 与论文不同源；LRS2 跨语料与自训两行的配方、卡数、epoch 都不同，不能互相替代。**⚠️ = 数值吻合（非同协议）**：两行 RealSSA-Sim 的 ckpt 语料轴（官方权重训在 `LRS3_complex_nonmoving`，论文该格用 RealSSA-Sim-trained ckpt）与通道轴（四通道均值 vs channel 1）同时不同，不构成严格复现（§4.10）。

### 3.1 评测协议：整段、窗口、实时是三套

| 场景 | 入口 | 处理方式 | 用于哪些数字 |
|---|---|---|---|
| LRS2 / RealSSA-Sim | `audio_visual_test.py` | 整段一次前向，`istft(length=T_in)`；无滑窗、无 hop、无 pad、无拼接 | §3 的 7.774 / 8.479、7.977 / 8.452、7.029 / 7.713、8.726 / 9.122、8.858 / 9.270；其中 RealSSA-Sim 两格为数值吻合、非同协议 |
| RealSSA-Real DNSMOS | `tools/repro/phase2/eval_realssa_real.py` | 长音频按 2 s 窗 / 0.5 s hop 拼，再算 P.835 / P808 | 2.7941 / 3.3365 |
| Studio 实时 | `studio/server.py` | 2 s 窗 / 0.5 s hop / 80 ms guard，跨窗拼接 | §4.6 的 +14.71 dB 一致性实验（稳态 +16.72 dB） |
| Space | `space/app.py` | 录完整段，一次推理 | 无独立指标 |

这三套不能互相"对齐"：整段前向的 cLNhw 统计跨整段，窗口推理只统计 2 s；RealSSA-Real 的 DNSMOS 还是 9.01 s 窗均值。把 LRS2 的 7.029 与 Studio 的实时效果放在一起比较，会同时混入协议差和窗边界损失。

评测入口有覆盖率自检（`audio_visual_test.py`，全文 28 处 `raise`）：计划数 = `min(limit, len(dataset))`，rank 内比对实际产出行数与计划数，rank 间比对 rank0 写的 `coverage_rank0.json` 与各分片行数，合并后再核总行数；任一处不一致就非 0 退出。`--allow-partial` 把这套检查显式降级为 warning，并在 `metrics_counts.json` 的 `coverage` 段落写 `verified: false`。`--limit` 在多卡时仍是"先截断再按 rank 分片"，`world_size=2 --limit 5` 得到 3+2 条，总数仍是 5。

### 3.2 已经对上的论文项

复现不是只有缺口，以下四类已经逐条核对：

- **任务定义**：论文是 AVTSE 单目标提取，"-2Mix" 只表示两说话人混合、只提一路（论文 §1 明确改词），与代码 `n_src: 1` 一致。
- **9 个宽度全中**（Table 15）：Ce=112 / C=112 / Ch=64 / Cv=32 / dim(q)=64 / h_F=24 / h_T=72 / n_head=4 / L=4。
- **结构与公式同构**：GCI 与 Eq.(6) 逐符号同构；复数掩码与 Eq.(1) 逐符号同构；TFRB 三件套（F 向 BiLSTM、T 向单向 LSTM、tril-mask 自注意力）全部落地。
- **训练配方 16 行里 15 行对上**：−SI-SNR、λ_spk 1.0、margin 0.5、τ 0→1@1000ep、AdamW 1e-3 + wd 0.01、ReduceLROnPlateau(15, 0.5)、早停 30、1000 ep、bs 4/GPU、8 卡、2 s、25 fps·50 帧、Hann 256-128、3 通道、FRCNN 冻结、seed 42；唯一不对的是 near 负样本语义（§4.7）。
- **RealSSA-Real 元数据几乎逐字命中**：54 段 / 5,039.4 s / 84.0 min，论文写 54 / 5,035 s / 83:55。

## 4 口径与已知偏差

### 4.1 因果性：只能保证"窗间无状态"

"causal"在本仓库能落到实处的只有一层：**每窗从零开始算，窗间没有 hidden state、没有音频 cache**。窗内连帧级因果都不成立。lookahead 要分三档写，三档都来自同一轮实测：

| 口径 | 数值 | 具体含义 | 标签 |
|---|---|---|---|
| 单遍 AudioBlock 几何 | **57 帧 / 456 ms** | 只看一遍 AudioBlock 的卷积 / LSTM 感受野 | 实测 |
| 整链几何 | **113 帧 / 904 ms** | 编码器阶梯 15 帧 + 第一遍 57 帧 + 第二遍 56 帧（解析值 115） | 实测 |
| as-shipped | **窗尾，≤2000 ms** | 30 处 `cLNhw = GroupNorm(1,C)` 在整窗 `112×251×129 ≈ 3.6 M` 元素上统计 | 实测 |

as-shipped 一档的实测依据：在帧 245 之后注入 1e-3 噪声，**帧 0 仍变化 −52 dB**；把 `cLNhw` 换成论文式逐 `(t,f)` 的 `LN_c` 后，帧 0…120 变成 **−240 dB（数值 0）**，依赖下界回落到 113 帧。单点脉冲的影响范围是输出 **[13, 31999]**。

论文 §4.2 / App E 写的是"left-aligned padding … adding no temporal lookahead"和"audio 最多 15 帧 = 120 ms"，与代码不符：代码是三处硬编码 `causal=False` 的对称 padding（`Falcon.py:772,808,821`，绕过了 `:1342` 注入的 `causal=true`），而 15 帧 = `2^4−1` 只是编码器最深层的累计值，不含 GCI / decoder 的 +44 帧，也不含第二遍的 +56 帧。这三处已经有开关：`paper_causal=True`（默认 False = 发布行为）把对称 padding 改成透传 `causal=True`，几何 lookahead 回到编码器阶梯的 15 帧 / 120 ms；要同时满足论文的 frame-local 承诺，还得替换 `cLNhw`。两项都会改变数值，质量代价要重训 A/B 才有数。

视觉侧同样不是论文的"3 future frames = 120 ms"：FRCNN 的 `Conv3d(k=(5,7,7), pad=(2,3,3))` 给的是 **±2 帧 = ±80 ms**；系统级视觉 lookahead 实测 **+13 视频帧 = 520 ms**，因为第一遍 AudioBlock 的 57 帧前视会把融合点之后的视频需求一起带上来。

补充两处结构失真，读延迟数字时会用到：

- **视频没有时间递推**。`VideoBlock` 把 `(B,hid,T)` 转成 `(B,T,hid)` 后喂给 `batch_first=False` 的 LSTM，序列维被当成 batch；脉冲响应只有 1/50 帧变化。视觉的真实时间上下文只剩 FRCNN 的 ±2 帧。
- **解码器有一个 16 ms 结构延迟**。`causal=True` 时前向 pad 2 帧、末尾只裁 1 帧，净 +2 帧 = +256 采样。delta 权重直通实测 lag=+256、corr=1.0000；发布权重的端到端 lag=0（被掩码吸收），但重训 / 换 hop 会把这个错位带回来。

### 4.2 参数量

音频模型 **537,697** 个参数，全部 `requires_grad`；加 65 个持久化 BN buffer 后 state_dict 是 537,762 条。其中 **5,568（1.04%）是死参数**：`concat_layers[2]` 从未被调用。实测把这块清零后输出 **bit-identical**（`max|Δ|=0`）；对照组 `concat_layers[1]` / `fusion_layers[2]` 清零分别改变输出 0.441 / 0.223，说明清零测试本身有分辨力。

四个口径都从总参数出发，扣的项不同：

| 口径 | 数值 | 算法 |
|---|---|---|
| 总参数 | **537,697** | 全部 requires_grad |
| 扣死参数（生效参数） | **532,129** | 537,697 − `concat_layers[2]` 5,568 |
| 扣对比头 | **523,113** | 537,697 − `spk_pool_proj` 14,584（仍含 5,568 死参数） |
| 两者都扣（分离前向实际用到） | **517,545** | 532,129 − 14,584，也等于 523,113 − 5,568 |

端到端是 **11,724,385** = 537,697 + 冻结 FRCNN 11,186,688。引用时按口径注明扣了哪一项：同样是扣对比头，从总参数扣是 523,113，从生效参数扣是 517,545，两个数不同来自基数不同。Big 变体是音频 1,158,617 / 端到端 12,345,305。

### 4.3 MACs：两把尺子，不能并列

**10.611 GMAC（10,610,934,096）是 ptflops 口径**：含 LSTM、不含冻结 FRCNN、输入 2 s / 50 帧。它与论文 Table 5 的 10.6 G 同尺。LSTM 在这 10.611 G 里约占 **5.02 G（47%）**，卷积（含下采样 / 融合）合计不到 6%。

| 口径 | 音频 | 视觉（88×88×50） | 整窗 | 视觉占比 |
|---|---|---|---|---|
| ptflops（含 LSTM） | **10.611 G** | **15.897 G** | **26.51 G** | **60.0%** |
| `torch.utils.flop_counter`（不记 `aten::lstm`） | 5.382 G | 15.897 G | 21.28 G | 74.7% |

视觉成本随输入尺寸变化：64² 是 7.769 G，88² 是 15.897 G，96² 是 17.481 G（均 50 帧）。整窗成本应写 **26.51 GMAC、视觉 60.0%**。

`flop_counter` 的 5.382 G 与 10.611 G 都"对"，但尺子不同：前者不记 `aten::lstm`，恰好少掉一半。早先文档里的 "21.19 G / 视觉 74.6%" 就是这个口径（5.381 + 15.897 ≈ 21.28，四舍五入差异来自视觉取整），只能在注明 `flop_counter` 时使用，不能当整窗成本对外引用，也不能和 10.6 G 并列。

### 4.4 DNSMOS：只有一个版本可引用

可引用的只有 full4：**OVRL 2.7941 ± 0.4100 / P808 3.3365 ± 0.4066**，N=54，P.835。权重是官方 `falcon.pt`（LRS3_complex_nonmoving，epoch 714），不是论文 Table 3 那一格用的 RealSSA-trained ckpt（未发布），所以**与论文 3.276 不可比**；同一 ckpt 在 VoxCeleb2 版上的 2.878 与 2.7941 同量级，可作旁证。

full3 的四个值全部作废，原因是窗口起点 bug（commit `2062528` 把 `endpoint = LEFT + HOP` 改成 `HOP + GUARD`），旧初值使每条估计开头 21,440 样本（1.34 s）恒为静音。这是输出确定性错误，不是幅度大小的取舍；`DNSMOS-realssa-real-mean-full3/` 只作历史证据保留。

另外两点口径：DNSMOS 的"整段"实际是 9.01 s 窗均值（`speechmos/dnsmos.py:12,65-73`）；`score_array` 在峰值 >1 时先做峰值归一（`eval_dnsmos.py:81-83`）。工具版本是 speechmos 0.0.1.1 + onnxruntime 1.31.0。

### 4.5 条数与分母

RealSSA-Sim（非同协议口径，§4.10）的 9,169 条只对 SI-SNRi 成立。SDRi / PESQ 的实际分母是 **9,166**：日志里有 6 次 `SDR computation failed (singular matrix)` 和 3 次 `PESQ computation failed (b'No utterances detected')`，`nanmean` 静默跳过 NaN，只在 `n_*` 列留痕。三条退化句是 `00003_02_41188.wav`、`00012_02_84551.wav`、`50016_74250.wav`；mean 与 ch0 两个通道变体是同样三条，说明是数据本身退化，不是随机失败。

CSV 的 `avg` 行在 `:9171`（表头 + 9,169 + avg + std），SI-SNRi 的 `n=9169`，SDRi / PESQ 的 `n=9166`。分母需与指标同行给出：9,169 只对 SI-SNRi 成立。

LRS2 侧的分母另有一个未定项：`falcon_lrs2_2mix_paper/tt` 索引 6,000 行 = 3,000 唯一 mix × {s1, s2}，而历史 CSV 全是 3,000（单向）。所有"tt 3,000"的分母需要按 s1/s2 口径重核（推断，见 §6 待办 6）。

### 4.6 延迟：A800 数字未验证

- **A800 47.1 ms（small）/ 56.5 ms（big）是引用值，未验证**。唯一来源是 `docs/understanding/deepdive-20261024/06-studio-backend.md:402-403` 的 md 表格，没有原始 json / log。它也不能和论文 3090 的 51.3 ms 并列，硬件和测法都不同。
- **CPU 单窗约 0.6 s**（口径：B=1、2 s 窗、CPU；4 线程实测 592–694 ms，随机器负载波动，其中 FRCNN 119.2 + 音频 473.2；1 线程约 1.0 s），Studio 自报 RTF = 592/500 = **1.18**，在这台机器的 CPU 上不可实时。另一轮 warm 音频前向的中位数是 523 ms。
- **A800 端到端是推算，未实测**：按 `L ≈ L_algo + GUARD(80 ms) + 块粒度(0…500 ms) + F + I/O`，F=47.1 时最好 127 ms、平均约 377 ms。这个数依赖 §4.1 的 904 ms 几何 lookahead，不能当成"算法延迟 120 ms"。
- **跨窗一致性是量级问题**：6 s 流（3×2 s LRS2 片段拼、join 处 10 ms 交叉淡化，Studio 从 t=0 开窗）按 Studio 方案分块拼接 vs 整段一次前向 = **SI-SNR +14.71 dB、corr 0.9835、最大偏差 53.4% 峰值**；去掉冷启动首窗（稳态）= **+16.72 dB / 0.9895 / 28.5%**。20 ms 交叉淡化只把缝跳变从 1.24e-2 降到 9.57e-3，SI-SNR 一点没变。
- **guard 是次要旋钮，不是主因**：80→456 ms 在"丢弃窗尾"语义下 **+2.2 dB**（非单调），在"补右上下文"语义下只有 **+0.8 dB**（456 ms 后饱和）；换窗对齐 ±250 ms 只让整段一致性在 15.9–18.0 dB 间波动（**2.1 dB**），两个相位互相约 17 dB。几何上的上下文饥饿与实测代价不是一回事。
- **误差主体是 `cLNhw` 的窗长依赖**：同一 ckpt 任意两种 2 s 窗方案互差 13–19 dB，不是 hop 或 guard。
- **guard 80 ms 远小于模型需求，但代价很小**：块 = 窗内帧 178–240，而"足上下文"要求帧 ≤ 137，故**整块 63 帧都在几何上不足**；实测块内位置剖面 +22.9 / +6.4 / +9.8 / +13.8 / +8.6 dB，误差集中在块内后段。把 guard 拉到 456 ms 只回收 +2.2 / +0.8 dB（见上条）。Studio 的 "80 ms right context" 是发射几何、挡的是窗尾 iSTFT 边界瞬态（末 320 样本能量是窗中位数的 5.4 倍），不是模型感受野。

### 4.7 数据侧三条硬事实

1. **LRS2-2Mix 的 SNR 落在 [0, 10] dB**。配方是 `mix = 10^(g1/20)·s1 + 10^(g2/20)·s2`，`g1 ∈ [0,5]` 且 `g2 = −g1`，所以 `SNR(s1 over s2) = 2·g1 ∈ [0,10] dB`；20,000 条 tr stem 实测中位 **5.025**、均值 5.011。mix 严格等于两路之和（PCM16 舍入误差 max|Δ|=3.05e-5）。
2. **near 是同 mixture 的 s2 原始波形**，K=1，且**没有乘 g2**。因此 near 与 mixture 里的干扰分量相差 `|g2|` dB 的响度，做对比学习时这是一个未声明的不一致。RealSSA 侧 `near.json` 100% 是自引用 fallback，**有效 K=0**，NASS 只剩正项，τ 退火在 K=1 分支不生效。
3. **`mouths/` 是硬链接**：`falcon_lrs2_2mix` ↔ `_paper`、`realssa_sim` ↔ `_ch0` 共享同一批 inode；四个目录 `du` 相加 110 G，真实并集 62 G。

数据管线还有三处会影响训练解释：LRS2 每条恰好 2.000 s，`rand_start=randint(0,0)=0`，训练没有随机裁剪；`idx` 默认被丢弃，`shuffle` 无效、采样有放回、epoch 长度与数据量解耦（`respect_index=True` 才按 idx 取值，每个样本一轮出现一次）；`train_samples=10000` 对 40,000 条池子 ⇒ 1 epoch 只摸到约 22–25% 的唯一 mix，1000 epoch 约等于 250 遍全量，不是 1000 遍。

### 4.8 训练口径的三处失真

| 项 | 论文 / 配置怎么写 | 代码实际 | 影响 |
|---|---|---|---|
| 训练损失 | −SI-SNR / −SI-SDR | 默认是 **−SNR**（非尺度不变），只有 v2 配置改 −SI-SDR | 旧 run（ep99 / ep124）与论文不同尺，不能直接比 |
| τ 退火 | 0→1，前 1000 epoch 线性 | LRS2 每样本 K≡1，K==1 分支不含 β；RealSSA K=0 | LRS2 上 τ 空转，ep124 的 +0.13 / +0.15 dB 不能归因于 τ |
| margin | 0.5 | 真实 run 的 `cos_neg≈0.9845`，`relu(sim−0.5)` 恒正 | margin 不起筛选作用，只相当于一个常数偏移 |

另外两条会影响结果解释：`threshold_byloss=True`（默认）会丢掉 **loss 更小的好样本**（`min_loss > -30` 的保留逻辑实际留下了更差的样本）；发布 ckpt 只到 **epoch 714**，而配方是 1000 epoch 退火，所以官方权重的 τ 只走到 β≈0.714。8 卡 DDP 的 `find_unused_parameters=True` 必需性已由 `grad=None` 证明，但条件化 `all_gather` / `sync_dist` / CSVLogger version 冲突只在单进程验过。

### 4.9 掩码与视觉条件的两个"形同虚设"

- **复数掩码的相位被锁死**。`mask_act=ReLU` 让实部和虚部同时非负，每个 bin 的掩码相位限在 **[0°, 90°]**；真实权重下掩码 **98.9% 为 0**，`|Z|/|E| = 0.039`，更接近"开 / 关"而不是比例缩放。`RI_split` 还把 112 通道硬切成 0:56 / 56:112 当实部 / 虚部，这个划分是学出来的，不是物理上的实虚。
- **"零视频"仍注入一个学习到的常数**。`_encode_audio` 的 dummy video 是 `zeros(B,512,1)`，经 `VideoBlock` 后被 bias 抬成 `norm=9.72` 的逐帧常数向量，再作为频率无关的加性偏置加到音频上。数值上与 `zeros(512,50)` 只差 3e-6，不算 bug，但语义上视觉分支仍在贡献一个常数。
- **视听融合不是 cross-attention**。视频只经 `Conv1d(512→112, groups=16)` + `nearest` 上采样（50→251）+ `unsqueeze(-1)` 做逐帧、频率平坦的加性偏置；全模型没有跨模态 Q/K/V。`fusion_repeats` 取自 `video_params.repeats`，"视频块数"决定"融合次数"。

### 4.10 RealSSA-Sim：数值吻合，但不是严格复现

论文 7.7 / 8.2 那一格的口径是 **RealSSA-Sim-trained ckpt + channel 1**。本机用官方 `falcon.pt`（`LRS3_complex_nonmoving`，ep714）在 RealSSA-Sim tt 上整段一次前向、取四通道均值，得 **7.774 / 8.479**（SI-SNRi 高 0.07、SDRi 高 0.28）。误差都在 0.3 dB 以内，但 **ckpt 语料轴与评测通道轴同时不同**：ckpt 不是论文那一格用的 RealSSA-Sim-trained 权重，通道是 4 通道均值而论文用 channel 1。两个变量同时不同，只能判 **数值吻合，不构成严格复现**，也不能拿它反证结构或训练配方已被复现。

把通道轴单独修好（`--channel 0`，对齐论文 channel-1）得 **7.977 / 8.452**：SI-SNRi +0.20、SDRi −0.03，即通道影响 **<0.3 dB**。这个量级解释不了与论文的差异，也消不掉 ckpt 轴。要走到"严格复现"，两条路都得走：拿到 RealSSA-Sim-trained ckpt（未发布，§6 待办 5），或在同分布上从零训练到 7.7 / 8.2 一档。

L1 三行（参数 / MACs / 权重加载）不适用这条降级，结论仍是"精确复现"；本节只作用于 §3 的两行 RealSSA-Sim 评测。

## 5 验证基线

所有测试结论都带前提，但**测试基线本身不依赖 GPU**：有 GPU 与无 GPU 都是 **168 passed**（2026-11-08 双路实测）。

| 检查 | 实测 | 前提 | 耗时 / 备注 |
|---|---|---|---|
| `tools/repro/tests` | **168 passed / 0 skipped / 0 failed** | 有 GPU（`CUDA_VISIBLE_DEVICES=0`） | 约 147 s |
| 同上，无 GPU | **168 passed / 0 skipped / 0 failed** | `CUDA_VISIBLE_DEVICES=`（`cuda_available=False` 已在断言里） | 约 170 s；修前此行是 166 passed / 2 failed，两处已修（见本节末段） |
| Studio pytest | **51 passed / 2 skipped** | 有 CUDA | 18.74 s |
| 同上，无 CUDA | **53 passed / 0 skipped** | `CUDA_VISIBLE_DEVICES=` | 42.39 s；两个 `skipif(torch.cuda.is_available())` 从 skip 变 run——这是修前无 GPU 2 failed 的来源之一 |
| Studio Node | **8/8** | 无 | 约 4.0 s |
| Space 静态 | **3 passed** | 无 | 0.15 s |
| Space i18n | **226 条** | 组装后 | `check_i18n.mjs`，退出码 0 |
| 模型数值 | strict 加载 + 537,697 + bit-identical | 权重在 `/root/.cache/falcon_weights/` | `verify_model_params.py` / `verify_real_weights.py` |

测试收集数是 168（166 个 `def test_` + 2 处参数化），但**分母会随数据前提变化**：25 处 skip 里 20 处在 `test_phase2_pipeline.py`，数据被清掉或换机时会静默变 skip。引用"168"时同时说明本机数据清单（§2.5）。

测试基线（2026-11-08 实测）：有 GPU **168 passed**（约 147 s）；无 GPU **同样 168 passed**（约 170 s）。
修前无 GPU 为 166 passed / 2 failed：一处是指纹测试对 `torch` 线程数敏感（现已固定 `torch.set_num_threads`），
一处是 `test_phase1_regression` 硬断言 studio `skipped == 2`（现已放宽为 `skipped in (0, 2)`）；
另有 `audio_train.py` 校验顺序前移。见 `tools/repro/tests/test_phase1_regression.py` 与 `audio_train.py`。

### 5.1 CI 现状：待修，不是已通过

`.github/workflows/tests.yml` 的 repro job 在无 GPU + 浅克隆的 runner 上**仍会失败**（推断：由代码和本地等价条件推出，未取到 GitHub Actions 运行记录）；下面第 1 条（无 GPU）已修，第 2 条（浅克隆）仍在：

1. **无 GPU**（修前）：`ubuntu-latest` 没有 CUDA，当时 `test_phase1_regression.py` 的 `assert skipped == 2` 会在此失败；
   该断言已放宽为 `skipped in (0, 2)`，无 GPU 现在全绿（168 passed）。
2. **浅克隆**：`actions/checkout` 未设 `fetch-depth`（`:75`），`test_phase1_regression.py:113-118` 的 `_phase1_base()` 在 `git cat-file -t 689555c` 失败时显式 `pytest.fail`，并连带打掉依赖 `_landed()` 的多条断言。

同一 workflow 的 `frontend` job（`:31-69`）是安全的：只跑 Node、Space 与 lock 检查，`:66` 的 `grep "@ file://"` 与 lock 现状一致（lock 里 `file://` 出现 0 次）。修复方向见 §6 待办 1。

### 5.2 `checkpoints/` 里的废桩

`checkpoints/` 下曾有 **88 个 5,269 B 的 `.ckpt`**（2026-11-08 观测；每跑一次全量测试 +1，79→…→88；定稿时目录已清空，见 §6 待办 9），它们不是权重，是 pytest 的副作用：`state_dict = {'audio_model.w': ()}`，一个 0 维张量、1 个元素，epoch 0 / step 2。`*.ckpt` 被 gitignore，`git status` 永远干净，所以肉眼看不见。真正的冒烟 ckpt 在 `Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/*.ckpt`（约 51.9 MB）。

### 5.3 测试覆盖与两个盲区

`tools/repro/tests` 有 16 个 `test_*.py` 和 3 个 helper，共 166 个 `def test_`（收集 168 条）。覆盖的是入口可跑、参数 / MACs 基线、Phase 1 验收、数据转换契约、lock 可移植、Studio 回归，不覆盖训练收敛与听感。

两个盲区：

- **数据门控是静默降级**。25 处 skip 里 20 处在 `test_phase2_pipeline.py`；数据被清掉、换机或浅克隆时，168 条里会有一大批变 skip，`168` 的分母本身就变。`test_phase2_data.py` 的 4 个 skip 属 fail-open。引用通过数时一并写本机数据清单。
- **数值指纹对线程数敏感**。两个 fingerprint 用例把 float32 输出按字节做 sha256（`eval_sha`），`torch.set_num_threads` 改的是归约顺序，哈希随之变。跑这两条时保持默认线程数，否则换台机器对不上基线。

### 5.4 本文数字怎么复核

- **评测**：定位 CSV 的 `avg` 行（LRS2 类 `:3002`，RealSSA-Sim 类 `:9171`；后者是数值吻合、非同协议口径，§4.10），同时读 `n_*` 列确认分母；独立 NumPy 实现与 `look2hear.metrics.core` 对拍，SI-SDR 的 `maxabs` 是 4.82e-8 dB。
- **参数 / MACs**：`verify_model_params.py` 打印 537,697 与 10.611 G；死参数用"同权重清零后输出 bit-identical"对照，`concat_layers[1]` / `fusion_layers[2]` 作阳性对照。
- **一致性**：训练侧与 Studio 同输入 `max|Δ|=0`；跨 7 天的两次全量 run 差 3.3e-7 dB（SI-SNRi）。
- **证据链**：`P2D-train-metrics-1gpu.csv` 与训练产物的 sha256 都是 `c6d5704e…fef675`；ep86 / ep99 / ep124 的 ckpt sha256 与 `P2D-report.md:273-279` 逐字符一致。

## 6 待办（按价值排序）

| 序 | 事项 | 为什么先做 | 成本 | 验收 |
|---|---|---|---|---|
| 1 | 修 CI | 当前 repro job 必红（浅克隆），"测试全绿"无法被别人复核；**无 GPU 这一项已修**（skip 断言已放宽为 `skipped in (0,2)`，本地无 GPU 168 passed） | 低（估算：半天到 1 天） | 无 GPU + 浅克隆 runner 上 repro job 绿；剩余覆盖 `fetch-depth: 0`、给测试加 `enable_checkpointing=False` / tmp 落盘 |
| 2 | 因果性重训 A/B | 论文声明的是无时域 lookahead，当前 as-shipped 是整窗；这是对外表述和产品定位的根 | 中，开关已就绪，成本在 2 卡重训与评估 | `paper_causal=True` 后几何 lookahead 回到 15 帧 / 120 ms；替换 `cLNhw` 后帧级依赖落到 113 帧；A/B 的 SI-SNRi 差距有数 |
| 3 | 补消融 | 论文 6 组消融 0/6，是"论文级复现"最大的空洞，且不依赖 8 卡 | 中，2 卡小配方可跑 | Table 7 的阶段数、Table 9 的 τ 调度至少各一组，结果进 `docs/verification/` |
| 4 | LRS2 论文列 14.1 / 14.7 | 主表唯一没复现的格子；当前 8.858 / 9.270 是旧配方单卡 | 高，需 8 卡；33–50 h 是旧估算，没有实测支撑；或向作者索取 LRS2-trained ckpt | tt 3,000 SI-SNRi ≥13.6 / SDRi ≥14.2，或拿到 ckpt 后重评 |
| 5 | RealSSA-Sim 从零训练 | 现在只有官方 ckpt 的非同协议评测（数值吻合），没有同分布训练闭环 | 高，重建 near + 从 TOS 恢复 450 GB + 8 卡（时长旧估算 33–50 h，未实测） | 同分布 test 约 7.7 / 8.2（这才算同协议复现）；完成后才能做 RealSSA-trained ckpt 的 DNSMOS |
| 6 | 重核 tt 分母 | 6,000 行索引 = 3,000 mix × 2 向，历史 CSV 全是 3,000；所有跨语料结论都挂在分母上 | 低，读索引与 CSV | 明确每个 tt 数字是 s1 单向还是双向平均，写进 §3 口径列 |
| 7 | RealSSA-Real 扩展指标 | DNSMOS 只覆盖感知质量，缺说话人相似度与可懂度 | 中，需 WeSpeaker / NISQA / Mega-ASR 环境 | 54 段上给出 speaker-sim、NISQA、WER 三列 |
| 8 | 效率按论文协议重测 | A800 47.1 / 56.5 ms 无原始文件，3090 三项目前无法对标 | 低–中，需 GPU 与固定脚本 | CUDA events、median of 25、bs1、2 s，json 落 `docs/verification/` |
| 9 | 清理废桩并加守卫 | 每次测试往仓库根写 1 个假 ckpt，污染工作区且不可见 | 低 | `checkpoints/` 清空后跑全量测试不再新增；或加"仓库根不产生新文件"断言 |

成本列只有 450 GB 来自报告；33–50 h 是旧估算、没有实测支撑；其余是量级估算（推断），不是排期承诺。

本轮已把 5 个忠实化开关放进代码，默认值都等于发布行为（输出逐位不变），打开才改变数值：模型侧 `paper_causal`、`video_lstm_batch_first`（`look2hear/models/Falcon.py` 与 `studio/falcon_model/models/Falcon.py` 两份副本同构）；数据侧 `respect_index`、`mouth_fps`、`pad_mouth`（`look2hear/datas/avspeech_dataset*.py`）。A/B 用 `tools/repro/experiments/lookahead_ab.py`。

需要人决定的两项：

- 8 卡资源：排队 / 租 8×3090 或 A800，还是批准 2 卡降级先测 epoch 成本。
- 是否向作者索取 benchmark-specific ckpt、near 重生成脚本、数据构建参数（论文 App B 没写 SNR 采样范围）。

暂不推进的三项：

- **3090 的 51.3 ms / 125.3 MB / 364.7 MB 对标**：本机没有 3090，A800 硬件不同，继续测也只能"量级接受"。先按 §6 待办 8 把 A800 的原始 json 补齐。
- **Falcon-Big 与训练侧交叉验证**：`SpeakerContrastiveV2` 只在 `studio/falcon_model/`，训练侧没有对应实现，要先移植再谈对齐。
- **`128_512` 命名来源、face_alignment 版本同源**：纯考古，低价值。

## 7 常用命令

以下路径已核对存在；本轮没有逐条重跑，命令里的 `<ckpt>` 需要替换成实际文件。

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
REPO=$PWD

# 结构与权重（CPU 可跑）
PYTHONPATH=$REPO $ENV docs/verification/scripts/verify_model_params.py
PYTHONPATH=$REPO $ENV docs/verification/scripts/verify_real_weights.py
PYTHONPATH=$REPO $ENV docs/verification/scripts/verify_causality.py

# 复现测试（有 GPU 与无 GPU 均为 168 passed；约 147 / 170 s）
PYTHONPATH=$REPO $ENV -m pytest tools/repro/tests -q -rs

# Studio（Python 与 Node）
$ENV -m pytest studio/tests -q
for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done

# Space 组装与 i18n
$ENV space/build.py --out /tmp/falcon-space
node space/tests/check_i18n.mjs /tmp/falcon-space
# 注：space/tests 全目录在 falcon 环境会有 4 个 ERROR（缺 gradio/spaces），
# 单跑静态测试用 $ENV -m pytest space/tests/test_static.py -q

# 评测：整段一次前向，不切窗；独立 cwd，避免同名覆盖
mkdir -p /root/tmp/eval_run && cd /root/tmp/eval_run
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/audio_visual_test.py \
  --conf_dir $REPO/configs/LRS2_2Mix.yml --ckpt_path <ckpt> \
  --data_dir /root/data/falcon_lrs2_2mix --gpus 0

# Studio 本地起服务（默认端口从 7860 起扫描，也可显式给 --port）
$ENV studio/server.py --port 7860

# 8 卡论文配方（需 8 卡；DATA_ROOT 用 paper 版）
CONF=configs/LRS2_2Mix_Paper_v2.yml DATA_ROOT=/root/data/falcon_lrs2_2mix_paper \
  bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
```

`tools/repro/experiments/` 下四个脚本都能在 CPU 上跑，各自对应本文的一处口径：

| 脚本 | 用途 |
|---|---|
| [`lookahead_ab.py`](../tools/repro/experiments/lookahead_ab.py) | `paper_causal` / `video_lstm_batch_first` 四种组合的 A/B |
| [`ln_variant.py`](../tools/repro/experiments/ln_variant.py) | `cLNhw` 换成逐 (t,f) `LN_c` 的因果对照 |
| [`streaming_consistency.py`](../tools/repro/experiments/streaming_consistency.py) | 分窗拼接与整段前向的一致性 |
| [`video_ablation_check.py`](../tools/repro/experiments/video_ablation_check.py) | 视觉消融：换说话人、mouth 置零 / 打乱 |

> 本目录现共 5 个脚本（本轮新增流式仲裁 `streaming_canonical.py`）；用法、CPU 耗时与产出落盘见 [`tools/repro/experiments/README.md`](../tools/repro/experiments/README.md)，本轮实测证据见 [`docs/verification/deepdive-20261108/`](verification/deepdive-20261108/README.md)。

## 8 证据索引

| 主题 | 文件 | 支持什么 |
|---|---|---|
| 本轮总纲 | [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) | 全景、硬数字表、与旧文档的 14 条冲突 |
| 模型主干 | [`01-model-core.md`](understanding/deepdive-20261108/01-model-core.md) | 537,697 / 死参数 5,568 / n_src=2 崩 / 解码器 +16 ms |
| 层与 DSP | [`02-layers-dsp.md`](understanding/deepdive-20261108/02-layers-dsp.md) | `cLNhw` 全窗耦合、VideoBlock LSTM 失效、掩码相位 |
| 视频前端 | [`03-video-frontends.md`](understanding/deepdive-20261108/03-video-frontends.md) | 88×88 灰度、FRCNN 11.19 M、视觉 15.897 G |
| 训练系统 | [`04-training-system.md`](understanding/deepdive-20261108/04-training-system.md) | 入口可跑、τ 空转、margin 失效、epoch 覆盖 22% |
| 损失与指标 | [`05-losses-metrics.md`](understanding/deepdive-20261108/05-losses-metrics.md) | −SNR vs −SI-SDR、MACs 含 LSTM 的口径 |
| 数据管线 | [`06-data-pipeline.md`](understanding/deepdive-20261108/06-data-pipeline.md) | SNR ∈[0,10]、near=K=1、硬链接 62 G、46.9 样本/s |
| 评测与基准 | [`07-eval-benchmark.md`](understanding/deepdive-20261108/07-eval-benchmark.md) | 逐条 CSV 证据、9,166 分母、full3 作废；§3.3 判 RealSSA-Sim 两轴同时不同 ⇒ 数值吻合、不构成严格复现 |
| Studio | [`08-studio.md`](understanding/deepdive-20261108/08-studio.md) | 单进程 FastAPI、分块几何、guard 真义、51+2 / 8/8 |
| Space 与项目页 | [`09-space-project-page.md`](understanding/deepdive-20261108/09-space-project-page.md) | 54 文件 / 39.87 MB、i18n 226、CI 可过性 |
| 运维与复现 | [`10-ops-repro-git.md`](understanding/deepdive-20261108/10-ops-repro-git.md) | GPU 前提、CI 推断失败、废桩 ckpt、lock 124 行 |
| 论文对齐 | [`11-paper-alignment.md`](understanding/deepdive-20261108/11-paper-alignment.md) | 9 个宽度全中、配方 15/16、消融 0/6 |
| 因果与延迟 | [`12-causality-latency.md`](understanding/deepdive-20261108/12-causality-latency.md) | 113 帧 / 904 ms、as-shipped 窗尾、跨窗 +14.71（稳态 +16.72）dB |

以上 13 份 deepdive 报告在 `docs/understanding/deepdive-20261108/`，本轮未提交；clone 下来的仓库里这些链接是 404。

关键证据文件：

- 参数 / 权重：[`verify_model_params.py`](verification/scripts/verify_model_params.py)、[`verify_real_weights.py`](verification/scripts/verify_real_weights.py)、[`verify_causality.py`](verification/scripts/verify_causality.py)
- 评测：[`P2D-report.md`](verification/phase2/P2D-report.md)、[`P2E-report.md`](verification/phase2/P2E-report.md)、[`P2F-report.md`](verification/phase2/P2F-report.md)
- 逐句数据：[`P2E-realssa-sim-metrics_combined.csv`](verification/phase2/P2E-realssa-sim-metrics_combined.csv)、[`P2D-eval-epoch124-contrastive.csv`](verification/phase2/P2D-eval-epoch124-contrastive.csv)、[`P2D-eval-falcon-full.csv`](verification/phase2/P2D-eval-falcon-full.csv)
- DNSMOS：[`DNSMOS-realssa-real-mean-full4-20261031/`](verification/phase2/DNSMOS-realssa-real-mean-full4-20261031/)
- 消融与视觉条件：[`P2B-report.md`](verification/phase2/P2B-report.md)
- 阶段总结：[`REPRODUCTION_PHASE2_REPORT.md`](REPRODUCTION_PHASE2_REPORT.md)
- 环境锁：[`conda-env-lock.txt`](verification/conda-env-lock.txt)
- 数据说明：[`DATA_ACCESS_GUIDE.md`](DATA_ACCESS_GUIDE.md)
- 历史审阅：[`THREE_ROUND_REVIEW_20261030.md`](understanding/THREE_ROUND_REVIEW_20261030.md)、[`PAPER_ALIGNMENT_20261030.md`](understanding/PAPER_ALIGNMENT_20261030.md)

## 9 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-08 | 修正测试基线：实测无 GPU **同样 168 passed / 0 failed**（约 170 s），修前 166 passed / 2 failed 已由「指纹固定 `torch.set_num_threads`、studio skip 断言放宽为 `(0,2)`、`audio_train.py` 校验顺序前移」消除；§5 表、§5.1 CI、§8 命令同步，删掉「168 需 GPU 可见」这一前提 |
| 2026-11-08 | 统一 RealSSA-Sim 判定为「数值吻合（非同协议）、不构成严格复现」：§1 总结句、§3 两行、§2.5 备注、§3.1/§4.5/§6/§8 措辞同步；新增 §4.10 记录 ckpt 与通道两轴口径；L1 三行不降级；§5.2 按 `checkpoints/` 已清空更新，无 GPU 行改写通过数表述 |
| 2026-11-08 | 按 review 定稿：参数量改四层口径；Space 改 54 个文件；补评测覆盖率自检与 5 个忠实化开关；测试改 168；§7 增脚本索引 |
| 2026-11-08 | 按深挖结论全文重写：lookahead 拆三档并写明 as-shipped 依赖窗尾；参数量补生效 / 死参数口径；MACs 分 ptflops 与 `flop_counter` 两把尺子；DNSMOS 只留 full4；条数改 9,166；A800 延迟降级为未验证；测试补 GPU 前提与耗时；CI 标为待修；补废桩 ckpt、TSE 定性、数据三条硬事实 |
| 2026-10-31 | 按三轮 review 执行 Day-1 + Week-1 修复；新增本 STATUS 作为单一事实源；配置切论文 Table 16；证据与测试收口 |
| 2026-10-30 | 上游合并、论文公开校正、ch0 复评、ep86 证据恢复、三轮 review |

引用本文件用 §节号：行号每轮重写都会漂，本文里不再写 `STATUS.md:行号` 形式的引用。
