# Falcon 复现报告（2026-11-09）

> **2026-11-10 结果更新：**发布权重的训练语料及其与论文权重的身份仍待确认；RealSSA-Sim 的四通道均值与论文所述第一通道使用不同口径。默认视觉 LSTM 存在跨样本影响，历史近场增益与采样覆盖表述已有修正。完整结果与证据见[复现状态](STATUS.md)、[数据集说明](understanding/20261110-datasets.md)和[数据训练说明](understanding/20261110-data-and-training.md)。

> 对象：论文 *Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes*（Falcon，arXiv:2610.32749v1，[cs.SD] 2026-10-23）与 `nanless/Falcon-fork`。
> 代码基线：`dev_a800_2gpus:/root/code/github_repos/Falcon-fork`，HEAD `5371df7`（2026-11-08 16:35 UTC）；冻结 tag `repro-freeze-20261031` 指向 `6c1d5b7`，证据基线不等于 HEAD。环境 conda `falcon`（Python 3.11.17 / torch 2.8.0+cu128 / 2×A800-SXM4-80GB）。
> 事实源：[`STATUS.md`](STATUS.md)。本文只做汇总与指针，细节回到 STATUS 与各证据文件；两者冲突时以 STATUS 为准。数字标签：**实测**＝本轮在仓库或本机跑出，**推断**＝由代码/日志/等价条件推出，**未验证**＝只有文档记载。

## 0 结论

按 L1–L4 四层算，四层里只有一层完整复现。**L1 结构与权重：复现（实测）**——音频 537,697 个参数、10.611 GMAC（ptflops 口径）、`falcon.pt` strict 加载零 missing、训练侧与 Studio 同输入 `max|Δ|=0`。**L2 评测：只有 RealSSA-Sim 一格数值吻合（实测，7.774 / 8.479 对论文 7.7 / 8.2），但不是同协议**——本机用 LRS3 训练的官方权重评四通道均值，论文那一格用 RealSSA-Sim-trained 权重评 channel 1，ckpt 语料轴与通道轴同时不同，只能算巧合级吻合。**L3 训练：未完成**——训练入口可跑（CPU 2 epoch 实测通过），论文的 8×3090 / 1000 epoch 配方没在本机跑过，单卡旧配方最好 8.858 / 9.270，离 LRS2 论文列 14.1 / 14.7 差 5.2 / 5.4 dB。**L4 消融与扩展 benchmark：空缺**——论文 Table 7/8/9/18/19/21 六组消融 0 组按论文协议完成；LRS3-2Mix、VoxCeleb2-2Mix 数据未发布。整体判定：**结构、参数、权重、指标实现与评测口径可信；benchmark 数字只有一格量级吻合；训练的收敛质量与论文声明尚未验证。**

## 1 复现对象与判定标准

| 项 | 值 | 说明 |
|---|---|---|
| 论文 | arXiv:2610.32749v1，[cs.SD] 2026-10-23 | 证据级别 L1：LaTeXML HTML 全量版（正文+公式+App A–N），落地 504,680 B / 701 行 / 597 块；表格锚 Table 1–21 |
| 任务 | AVTSE 单目标说话人提取 | 输入混合语音 + 目标说话人唇部视频，输出单路目标语音；6 份配置全 `n_src: 1`，benchmark 名里的 "-2Mix" 只表示两说话人混合、只提一路 |
| 复现目标 | 论文 Table 3 主结果、Table 5 效率、Table 16 配方、Table 7/8/9/18/19/21 消融 | 验收容差（计划口径）：L2 指标 ±0.3 dB（同权重、同数据、同协议），训练 ±0.5 dB |
| 判定基准 | 同 ckpt 语料 + 同通道 + 同评测协议，三者同时满足才算"严格复现" | 计划 §3.1/§5.5 的验收条件；任何一轴不同即降一级，标"数值吻合（非同协议）" |

四层定义与本文用的标签：

- **L1 结构 / 权重 / 参数**：参数、MACs、strict 加载、数值一致性。可精确复算 ⇒ 写"复现（实测）"。
- **L2 评测指标**：在指定 benchmark 上跑 SI-SNRi / SDRi / DNSMOS。同协议同权重的 ±0.3 dB 内才算复现。
- **L3 训练复现**：按论文配方从零训练并达到论文 validation/test 水平；本地只到"入口可跑 + 单卡小配方"。
- **L4 消融与扩展 benchmark**：论文的组件消融与未发布数据的两个 benchmark。

三种标签不互相升级：`未验证` 的引用值不会因为被人反复引用而变成实测；`实测` 的数值也不能在换口径后继续沿用。

## 2 复现矩阵

口径列是这一行数字的成立条件，离开它数字没有意义。⚠️ = 数值吻合但非同协议。

| Benchmark / 项 | 论文值 | 本地实测 | 口径（ckpt / 数据 / 条数 / 算法 / commit） | 判定 | 证据文件 |
|---|---|---|---|---|---|
| L1 音频参数量 | 0.5 M | **537,697** | `FalconSpeakerContrastive`（small），全部 `requires_grad`；另有四层口径见 §3 | 精确复现（实测） | [`verify_model_params.py`](verification/scripts/verify_model_params.py) |
| L1 音频 MACs | 10.6 G | **10.611 G** | ptflops，含 LSTM、不含冻结 FRCNN，2 s / 50 帧；与论文 Table 5 同尺 | 同尺精确复现（实测） | `evaluated_mac_params.py`（仓库根） |
| L1 权重加载 | — | strict 零 missing；train↔Studio `max\|Δ\|=0` | `falcon.pt`（`LRS3_complex_nonmoving`，ep714）525 个 audio key | 复现（实测） | [`verify_real_weights.py`](verification/scripts/verify_real_weights.py)、[`verify_r3_studio_weights.py`](verification/scripts/verify_r3_studio_weights.py) |
| RealSSA-Sim SI-SNRi / SDRi（四通道均值） | 7.7 / 8.2 | **7.774 / 8.479** | 官方 `falcon.pt`（LRS3，ep714）；RealSSA-Sim tt 9,169 条整段一次前向；SDRi 分母 9,166 | ⚠️ 数值吻合（实测，非同协议），+0.07 / +0.28 | [`P2E-realssa-sim-metrics_combined.csv`](verification/phase2/P2E-realssa-sim-metrics_combined.csv) |
| RealSSA-Sim（channel 0，对齐论文 channel-1） | 7.7 / 8.2 | **7.977 / 8.452** | 同上，`--channel 0`；9,169 / 9,166 | ⚠️ 数值吻合（实测），通道影响 <0.3 dB，ckpt 轴仍不同 | [`P2E-realssa-sim-ch0-metrics_combined-20261030.csv`](verification/phase2/P2E-realssa-sim-ch0-metrics_combined-20261030.csv) |
| RealSSA-Real DNSMOS | 3.276（P.835） | OVRL **2.7941** / P808 **3.3365** | 官方 `falcon.pt`（LRS3），54 段，full4；speechmos 9.01 s 窗均值 | 不可比（实测），非复现失败：论文那格用未发布的 RealSSA-trained ckpt | [`DNSMOS-realssa-real-mean-full4-20261031/`](verification/phase2/DNSMOS-realssa-real-mean-full4-20261031/) |
| LRS2-2Mix（官方权重，跨语料） | 14.1 / 14.7 | **7.029 / 7.713** | 官方 `falcon.pt`（LRS3，ep714），tt 3,000，`limit=all`；ckpt 训练语料与论文列不同 | 跨语料基线（实测），不是论文列 | [`P2D-eval-falcon-full.csv`](verification/phase2/P2D-eval-falcon-full.csv) |
| LRS2-2Mix 自训 ep86 | — | 8.618 / 9.042（val 13.1182） | 单卡 100 epoch，旧配方（−SNR），同 tt 3,000 | 中间快照（实测） | [`P2D-eval-epoch86.csv`](verification/phase2/P2D-eval-epoch86.csv) |
| LRS2-2Mix 自训 ep99（best val） | — | 8.726 / 9.122（val 13.2014） | 同上 | 中间快照（实测） | [`P2D-eval-epoch99.csv`](verification/phase2/P2D-eval-epoch99.csv) |
| LRS2-2Mix 自训 ep124（+s2 near 微调） | — | **8.858 / 9.270** | 单卡 130 epoch，旧配方；距论文 −5.24 / −5.43 dB | 本机最好（实测），非论文配方 | [`P2D-eval-epoch124-contrastive.csv`](verification/phase2/P2D-eval-epoch124-contrastive.csv) |
| mix 基线（上下文） | — | 4.9937 dB | tt 3,000 的 `SI-SDR(mix, s1)` 均值，不是模型输出 | 参考值（实测） | [`P2A-report.md`](verification/phase2/P2A-report.md) |
| LRS2-2Mix 论文列 | 14.1 / 14.7 | 无 | 缺 LRS2-trained ckpt；8 卡配方未跑 | **不可复现（当前）** | §7 |
| LRS3-2Mix | 16.4 / 16.7 | 无数据 | HF 仓库只有 README，无数据文件 | **不可复现** | [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) §7 |
| VoxCeleb2-2Mix | 10.5 / 11.8 | 无数据 | 同上 | **不可复现** | 同上 |
| 效率（论文 3090） | 51.3 ms / 125.3 MB / 364.7 MB | A800 47.1 / 56.5 ms | 只有 md 表格，无原始 json/log；硬件是 A800 不是 3090 | **未验证**，不可并列 | [`STATUS.md`](STATUS.md) §4.6 |
| 结构宽度（Table 15） | 9 个宽度 | 9/9 全中 | Ce=112 / C=112 / Ch=64 / Cv=32 / dim(q)=64 / h_F=24 / h_T=72 / n_head=4 / L=4，读 `configs/Falcon.yml` | 复现（实测） | [`11-paper-alignment.md`](understanding/deepdive-20261108/11-paper-alignment.md) §2.2 |
| 训练配方（Table 16） | 17 行；STATUS 按 16 项计 | 15/16 对上 | `configs/LRS2_2Mix_Paper_v2.yml` 对照论文 Table 16；唯一实质缺口是 near 负样本语义 | 部分（实测 + 推断） | §5、[`11-paper-alignment.md`](understanding/deepdive-20261108/11-paper-alignment.md) §4 |
| 因果性 / lookahead | "无时域 lookahead"，音频 ≤15 帧 = 120 ms | 整链几何 **113 帧 = 904 ms**；as-shipped 依赖窗尾 | 真实权重，2 s 窗，单窗内 | **与论文声明冲突（实测）** | §6 |
| RealSSA-Real 元数据 | 54 段 / 5,035 s / 83:55 | 54 段 / 5,039.4 s / 84.0 min | 发布包逐段元数据求和，差 0.09% | 复现（实测） | [`11-paper-alignment.md`](understanding/deepdive-20261108/11-paper-alignment.md) §2.6 |
| 消融 Table 7/8/9/18/19/21 | 6 组 | 0 组按论文协议 | 只有 big/small 与 mouth 置零/打乱的 N=20 部分实验 | **未完成** | §6 |

## 3 结构与权重

**参数四层口径（实测，`verify_model_params.py`，HEAD `5371df7`）**：总参数 537,697（全部 `requires_grad`，state_dict 另含 65 个持久化 BN buffer ⇒ 537,762 条）；扣掉从未被调用的 `concat_layers[2]` 5,568 个死参数（1.04%）得生效参数 **532,129**；扣对比头 `spk_pool_proj` 14,584 个得 **523,113**（仍含死参数）；两者都扣、即分离前向实际用到 **517,545**。端到端是 537,697 + 冻结 FRCNN 11,186,688 = **11,724,385**。引用时说明扣的是哪一项：同叫"扣对比头"，从总算 523,113、从生效参数算 517,545。

**MACs 有两把尺子（实测，ptflops 与 `torch.utils.flop_counter` 分别复算）**：

| 口径 | 音频 | 视觉（88×88×50） | 整窗 | 视觉占比 |
|---|---|---|---|---|
| ptflops（含 LSTM） | **10.611 G** | **15.897 G** | **26.51 G** | **60.0%** |
| `flop_counter`（不记 `aten::lstm`） | 5.382 G | 15.897 G | 21.28 G | 74.7% |

10.611 G 与论文 Table 5 的 10.6 G 同尺（含 LSTM、只算音频、不含冻结 FRCNN）。5.382 G 少掉的正好是 LSTM 那一半（LSTM 占 10.611 G 的约 5.02 G / 47%），只能在注明 `flop_counter` 时使用，不能与 10.6 G 并列。视觉成本随输入尺寸变：64² 是 7.769 G、88² 是 15.897 G、96² 是 17.481 G（均 50 帧）。

**权重与数值一致性（实测）**：`falcon.pt` strict 加载零 missing（662 个 key = 525 audio + 137 video）；训练侧与 Studio 同一输入 `max|Δ|=0`；把 5,568 个死参数清零后输出 bit-identical（`max|Δ|=0`），阳性对照 `concat_layers[1]` / `fusion_layers[2]` 清零分别改变输出 0.441 / 0.223，说明清零测试本身有分辨力。big 变体是另一套结构（`SpeakerContrastiveV2`，只存在于 `studio/falcon_model/`），音频 1,158,617 / 端到端 12,345,305，少 8 个 `out_proj` key 是因为 `hid_chan×方向数 == in_chan` 使 `out_proj` 退化为 `Identity`。

**视觉前端的实际输入是 88×88（实测）**：数据盘存 96×96×3，`Grayscale(1)` → `Normalize(/255)` → `CenterCrop(88)` → mean 0.421 / std 0.165；论文 Table 16 的 "96×96" 指存盘 ROI，网络看到的是 88×88。这一点决定了视觉 MACs 的口径：按 88² 算是 15.897 G，按 96² 算是 17.481 G。

## 4 评测协议

**一次前向，全程不切窗（实测，`audio_visual_test.py:419`）**：整段 `mix` 一次送进模型，`istft(length=T_in)`，全文没有 window / hop / chunk / pad。2 s 窗 / 0.5 s hop / 80 ms guard 只属于 Studio 运行时与 RealSSA-Real 长音频评测器，三套协议不能互相对齐：整段前向的 `cLNhw` 统计跨整段，窗口推理只统计 2 s。把 LRS2 的 7.029 与 Studio 的实时效果并列，会同时混入协议差（整段→分窗 −1.9 / −2.3 dB，300 句 small / big；另跑 5 句为 −1.648）与窗边界损失。

**单路输出与指标定义**：`n_src=1`，`MaskGenerator` 输出 1 路，PIT 在 S=1 时退化为恒等，不存在 "best-of-2" 输出选择。`SI-SNRi = SI-SDR(ŝ, s1) − SI-SDR(mix, s1)`，逐句算，相对未处理混合；SDR 滤波器长度 512，`zero_mean=True`。`n_src=2` 在代码层跑不通（前向直接 `RuntimeError: shape '[2, 224, 251, 129]' is invalid`），所以"2 源分离"的说法在代码里不成立。

**条数与分母**：RealSSA-Sim 的 **9,169** 只对 SI-SNRi 成立；SDRi / PESQ 的实际分母是 **9,166**——日志里有 6 次 `SDR computation failed (singular matrix)` 和 3 次 `PESQ computation failed`，`nanmean` 静默跳过 NaN，只在 `n_*` 列留痕。三条退化句是 `00003_02_41188.wav`、`00012_02_84551.wav`、`50016_74250.wav`，mean 与 ch0 是同样三条，说明数据本身退化。CSV 的 `avg` 行：RealSSA-Sim 类 `:9171`，LRS2 类 `:3002`。LRS2 侧另有一个未定项：`falcon_lrs2_2mix_paper/tt` 索引 6,000 行 = 3,000 唯一 mix × {s1, s2}，而历史 CSV 全是 3,000（单向）。

**覆盖率与子集语义（实测，HEAD `5371df7`）**：`audio_visual_test.py` 有 28 处 `raise`，其中包含 rank 内、跨 rank、合并后三层行数自检，rank0 写 `coverage_rank0.json`；`--limit` 是子集语义，`metrics_counts.json.coverage` 会写 `is_subset=true`、`verified=false`；`--allow-partial` 只把行数自检降级为 warning。多卡时是"先截断再分片"，`--gpus 0,1 --limit 5` 得到 3+2 条，总数仍是 5。

**DNSMOS 只有一个版本可引用（实测）**：full4 的 OVRL 2.7941 ± 0.4100 / P808 3.3365 ± 0.4066，N=54，P.835，工具 speechmos 0.0.1.1 + onnxruntime 1.31.0。它的"整段"实际是 9.01 s 窗均值（`speechmos/dnsmos.py:12,65-73`），峰值 >1 时先做峰值归一。full3 的四个值作废：窗口起点 bug 让每条估计开头 21,440 样本（1.34 s）恒为静音，是输出确定性错误，不是幅度取舍。与论文 3.276 **不可比**，因为 3.276 属未发布的 RealSSA-trained ckpt，本机权重训在 LRS3。

**确定性**：官方 `falcon.pt` 跑 5 条，对历史 CSV 前 5 行的 `max|Δ|` 为 5.3e-8 dB（SI-SNRi）、PESQ 2.3e-5；相隔 7 天的两次全量 run 逐行差 3.3e-7 dB。误差在 float32 内核噪声量级，评测本身确定性够用。

## 5 训练配方对齐

按 STATUS §3.2 的 16 项口径，**15 项对上，唯一实质缺口是 near 负样本语义**：−SI-SNR、λ_spk 1.0、margin 0.5、τ 0→1@1000 ep、AdamW 1e-3 + wd 0.01、ReduceLROnPlateau(15, 0.5)、早停 30、1000 ep、bs 4/GPU、8 卡、2 s、25 fps·50 帧、Hann 256-128、3 通道、FRCNN 冻结、seed 42。11 号报告另有一张 17 行的行级对照表，并额外标出两处口径差（mouth ROI 存盘 96 与网络输入 88；v2 注释写 "grad clip 5" 但 `av_litmodule.py` 里 grep 不到裁剪实现）与一处"取决于用哪份配置"（重建损失，老配置 `pairwise_neg_snr` 对不上，v2 改 `pairwise_neg_sisdr` 才对上）。两套计数不冲突，差别只在把这些杂项算不算独立行。

配方文本对上，训练语义有三处失真（实测/推断，见 [`04-training-system.md`](understanding/deepdive-20261108/04-training-system.md)）：

1. **τ 退火在 LRS2 上无效**。每样本 `K≡1`，`K==1` 分支不含 β；RealSSA 侧 `near.json` 100% 是自引用 fallback，有效 `K=0`，对比损失只剩正项。ep124 的 +0.13 / +0.15 dB 不能归因于 τ。
2. **margin 0.5 失效**。真实 run 的 `cos_neg≈0.9845`，`relu(sim−0.5)` 恒正，margin 只相当于一个常数偏移，不起筛选作用。
3. **1 epoch 只覆盖约 22–25% 的唯一 mix**。`train_samples=10000` 对 40,000 条池子，且 `idx` 默认被丢弃、`shuffle` 无效、采样有放回，epoch 长度与数据量解耦；1000 epoch 约等于 250 遍全量，不是 1000 遍。发布 ckpt 只到 epoch 714，τ 实际只走到 β≈0.714。

另两条会影响结果解释：老配置的默认损失是 **−SNR**（非尺度不变），只有 v2 改 −SI-SDR，所以 ep99 / ep124 与论文不同尺；`threshold_byloss=True`（默认）会丢掉 loss 更小的好样本，v2 才关掉。训练吞吐实测 46.9 样本/s ≈ 11.7 it/s（nw=8 / bs=4 / paper-tr），这是 2 卡机器上的量级，不是 8 卡实测。

结论：**配方文本对齐不等于训练复现**。L3 缺的是"同配方在同一数据上跑满并达到论文水平"这一步，以及 near 负样本的语义缺口（论文要求近场干扰 `b_k` 带房间响应，本地 LRS2 侧给的是未乘 g2 的 s2 原始波形）。

## 6 与论文的实质差距

**因果性：904 ms 对声明的 120 ms（实测）**。lookahead 要分三档写，都来自同一轮实测：单遍 `AudioBlock` 几何 **57 帧 / 456 ms**；整链几何 **113 帧 / 904 ms**（编码器阶梯 15 帧 + 第一遍 57 + 第二遍 56，解析值 115）；as-shipped 因 30 处 `cLNhw = GroupNorm(1,C)` 在整窗 `112×251×129 ≈ 3.6 M` 元素上统计，**依赖到窗尾（2 s 窗时 ≤2000 ms）**。as-shipped 一档的实测依据：帧 245 之后注入 1e-3 噪声，帧 0 仍变化 −52 dB；把 `cLNhw` 换成论文式逐 `(t,f)` 的 `LN_c` 后，帧 0…120 变成 −240 dB（数值 0），依赖下界回落到 113 帧。论文 §4.2 / App E 的 "left-aligned padding … adding no temporal lookahead" 与代码不符：`AudioBlock` 的 downsample / fusion / concat 三处硬编码 `causal=False`（`Falcon.py:772,808,821`），而 15 帧 = `2^4−1` 只是编码器最深层的累计值，不含 GCI / decoder 的 +44 帧，也不含第二遍的 +56 帧。

**cLNhw 是整窗统计，不是逐帧归一**。它决定了"换窗长即换结果"：同一 ckpt 的 2 s 窗与 4 s 整段输出差 SI-SNR 16.03 dB；6 s 流按 Studio 2 s/0.5 s/80 ms 重算 vs 整段一次前向 = **+14.71 dB / r 0.9835 / 最大偏差 53.4% 峰值**，剔掉冷启动首窗的稳态是 +16.72 / 0.9895 / 28.5%。误差主体是 `cLNhw` 的窗长依赖，不是 hop 或 guard（guard 80→456 ms 只值 +2.2 / +0.8 dB）。论文说 "LN_c is frame-local and cache-free"，代码做不到。

**视觉侧三处偏差**。论文 App E 说 25 fps 下缓冲 3 帧未来帧 = 120 ms，代码 FRCNN 前端是 `Conv3d(k=(5,7,7), pad=(2,3,3))`，即 **±2 帧 = ±80 ms**；系统级视觉 lookahead 实测 **+13 视频帧 = 520 ms**，因为第一遍的 57 帧前视会把融合点之后的视频需求一起带上来。网络实际看到的是 **88×88**（盘存 96×96 → `Grayscale` → `CenterCrop(88)`），与论文 Table 16 的 "96×96" 差一档，视觉 MACs 因此在 15.897 G（88²）与 17.481 G（96²）之间取值。另外 `VideoBlock` 把 `(B,hid,T)` 当 `(T,B,hid)` 喂给 `batch_first=False` 的 LSTM，序列维被读成 batch，**帧间没有递推**，视觉的真实时间上下文只剩 FRCNN 的 ±2 帧。

**融合是加性偏置，掩码是 ReLU 限幅（实现口径，不是论文声明）**。视频只经 `Conv1d(512→112, groups=16)` + `nearest` 上采样（50→251）+ `unsqueeze(-1)` 做逐帧、频率平坦的加性偏置——这与论文 §4.1 的 "expanded along the frequency axis and added" 一致，但容易被读成 cross-attention；全模型没有跨模态 Q/K/V。复数掩码的 `mask_act=ReLU` 让实部虚部同时非负，掩码相位锁在 [0°, 90°]，真实权重下 **98.9% 的掩码为 0**、`|Z|/|E| = 0.039`，更接近"开/关"而不是比例缩放。

**消融 0/6**。Table 7（阶段数）、8（对比损失）、9（τ 调度）、18（循环单元）、19（输出形式）、21（视觉缺失）没有一组按论文协议做完，且不依赖 8 卡——这是"论文级复现"最大的空洞。已有的只是 big/small 与 mouth 置零/打乱的部分实验，N=20；其中早先"mouth 置黑 −14.4 dB"的结论是**误标**：对已归一化数组取零得到的是中灰脸（−15.98 dB），真黑帧只掉 −3.16 dB，换说话人 −11.68 dB（N=40、40 个不同说话人）。

**数据侧三条硬事实**（实测，影响数字解释）：LRS2-2Mix 的 SNR 落在 **[0, 10] dB**（`g1∈[0,5]`、`g2=−g1`，中位 5.025）；near 负样本是同一 mixture 的 s2 原始波形、且未乘 g2，与 mixture 里的干扰分量差 `|g2|` dB 响度；`mouths/` 在四个 Falcon 数据目录间是硬链接，逐目录 `du` 相加 110 G，真实并集 62 G。

## 7 不可复现项与原因

| 项 | 论文值 | 本地状态 | 原因 | 判定 |
|---|---|---|---|---|
| LRS2-2Mix | 14.1 / 14.7 | 跨语料 7.029 / 7.713；自训最好 8.858 / 9.270 | 缺 LRS2-trained ckpt；论文配方是 8×3090 / 1000 ep，本机 2 卡且只在旧配方跑到 130 ep | 不可复现（当前），需 8 卡或作者提供 ckpt |
| LRS3-2Mix | 16.4 / 16.7 | 无数据 | HF 仓库只有 README（2026-10-21 API 核实） | 不可复现（数据未发布） |
| VoxCeleb2-2Mix | 10.5 / 11.8 | 无数据 | 同上 | 不可复现（数据未发布） |
| 效率（3090） | 51.3 ms / 125.3 MB / 364.7 MB | A800 47.1 / 56.5 ms | 本机没有 3090；A800 两个数只有 md 表格，无原始 json/log；测法也不同 | 未验证，不可并列 |
| 六组消融 | Table 7/8/9/18/19/21 | 0/6 | 协议未执行；改模型需另开分支，不能破坏官方权重的 bit-identical 语义 | 未完成 |
| 论文"无时域 lookahead / 120 ms" | 声明 | 904 ms 几何 / 窗尾 as-shipped | 代码三处对称 padding + 30 处全窗归一化，与论文措辞直接矛盾 | 冲突，不是资源缺口 |
| DNSMOS 3.276 | 3.276 | 2.7941 / 3.3365 | 该格用未发布的 RealSSA-trained ckpt，本机权重训在 LRS3 | 不可比 |
| RealSSA-Sim 严格同协议 | 7.7 / 8.2 | 数值吻合，非同协议 | 两轴同时不同：ckpt 语料（LRS3 vs RealSSA-Sim）+ 通道（四通道均值 vs channel 1）；同分布训练还缺 K≥2 near 负样本 | 数值吻合，不构成严格复现 |

[`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md) §9 的 12 条风险与本文的对应：R2（near 源波形未发布）、R4（4ch 协议未声明）、R5（LRS3/Vox 无数据）、R8（指标实现 bug）、R10（RealSSA-Real 隐私限制）、R12（参数口径）分别落到 §5、§4、§7、§4、§7、§3；R1（论文未公开）、R3（访问审批）、R6（LRS2 只有音频）、R7（视觉权重缺失）已解除。未处理的两条：R9 只固定了 seed 42，没有多 seed 方差；R11 的 Space 测试需要独立环境（`spaces` / `gradio`），在 `falcon` 环境里是 4 个 ERROR 而不是 skip。

## 8 复跑指南

**环境**：可用环境只有一个，`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / torchaudio 2.8.0 / pytorch-lightning 2.6.6 / typeguard 2.13.3）。`typeguard` 需 <3（`look2hear/layers/stft_tfgn.py` 用 `check_argument_types`）；`pypesq` 是 site-packages 里 792 B 的本地 shim，不在 lock 里，重建环境要按 `AGENTS.md` §5 手写。`torch.stft` 在 CPU fp16 会崩，`model.half()` 推理路径不可用。

**权重与数据**：权重在 `/root/.cache/falcon_weights/`（`falcon.pt` / `falcon-big.pt` / 两个 `frcnn*.pth.tar`，均为软链）；数据在 `/root/data/`（`falcon_lrs2_2mix`、`falcon_lrs2_2mix_paper`、`falcon_realssa_sim`、`falcon_realssa_sim_ch0`、`RealSSA-Sim`、`RealSSA-Real`），清单与容量见 [`STATUS.md`](STATUS.md) §2.5 与 [`DATA_ACCESS_GUIDE.md`](DATA_ACCESS_GUIDE.md)。

```bash
REPO=/root/code/github_repos/Falcon-fork; ENV=/root/miniforge3/envs/falcon/bin/python
cd $REPO && export PYTHONPATH=$PWD
$ENV -V                                                    # 3.11.17
$ENV evaluated_mac_params.py --device cpu                  # 音频 0.538 M / 10.611 GMAC
$ENV docs/verification/scripts/verify_model_params.py
$ENV docs/verification/scripts/verify_real_weights.py      # strict 零 missing
$ENV -m pytest tools/repro/tests -q                        # 168 passed（有/无 GPU 同值）
mkdir -p /root/tmp/eval_falcon && cd /root/tmp/eval_falcon  # 独立 cwd，避免同名覆盖
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/audio_visual_test.py \
  --conf_dir $REPO/configs/LRS2_2Mix.yml --ckpt_path /root/.cache/falcon_weights/falcon.pt \
  --data_dir /root/data/falcon_lrs2_2mix --gpus 1 --limit 5  # 期望 si-snr_i 6.522868 / sdr_i 7.567688
```

RealSSA-Sim 评测、DNSMOS 与 Studio 起服务的完整命令见 [`STATUS.md`](STATUS.md) §7；五个 CPU 实验脚本（`lookahead_ab`、`ln_variant`、`streaming_consistency`、`streaming_canonical`、`video_ablation_check`）的用法与耗时见 [`tools/repro/experiments/README.md`](../tools/repro/experiments/README.md)。

**门禁分工（实测）**：本机手动跑 168 条复现套件（有 GPU 约 147 s、无 GPU 约 170 s，多次 run 135–209 s），它依赖 `/root/data`、`/root/.cache/falcon_weights`、`/root/tmp` 训练产物和本机环境锁，hosted runner 上都没有；CI 只跑两个 job——`frontend`（Studio Node 8/8、Space 组装 54 文件 + i18n 226 条 + 静态 3 条、lock 检查）与 `python`（导入、`studio/tests`、Space `test_clip.py -m "not slow"`），不跑 `tools/repro/tests/**`、`space/tests/test_space.py` 与 slow 用例。HEAD `5371df7` 的 `tests` workflow 实测 success（1 分 32 秒），同一 commit 把 repro 套件移出 CI 门禁。

## 9 证据索引

| 主题 | 首选文件 | 支持什么 |
|---|---|---|
| 总纲与硬数字 | [`00-SYNTHESIS.md`](understanding/deepdive-20261108/00-SYNTHESIS.md) | 12 条速览、硬数字表（含口径列）、27 条坑清单、与旧文档 14 条冲突 |
| 模型主干 | [`01-model-core.md`](understanding/deepdive-20261108/01-model-core.md) | 537,697 / 死参数 5,568 / 张量链 / 解码器 +16 ms / `n_src=2` 崩 |
| 层与 DSP | [`02-layers-dsp.md`](understanding/deepdive-20261108/02-layers-dsp.md) | `cLNhw` 全窗耦合、VideoBlock LSTM 失效、掩码相位 |
| 视频前端 | [`03-video-frontends.md`](understanding/deepdive-20261108/03-video-frontends.md) | 88×88 灰度、FRCNN 11.19 M、视觉 15.897 G |
| 训练系统 | [`04-training-system.md`](understanding/deepdive-20261108/04-training-system.md) | τ 空转、margin 失效、epoch 覆盖 22%、DDP 前提 |
| 损失与指标 | [`05-losses-metrics.md`](understanding/deepdive-20261108/05-losses-metrics.md) | −SNR vs −SI-SDR、threshold 丢好样本、MACs 含 LSTM |
| 数据管线 | [`06-data-pipeline.md`](understanding/deepdive-20261108/06-data-pipeline.md) | SNR∈[0,10]、near=K=1、硬链接 62 G、46.9 样本/s |
| 评测与 benchmark | [`07-eval-benchmark.md`](understanding/deepdive-20261108/07-eval-benchmark.md) | 逐条 CSV 证据、9,166 分母、full3 作废、可引用/不可引用清单 |
| 论文对齐 | [`11-paper-alignment.md`](understanding/deepdive-20261108/11-paper-alignment.md) | 9 个宽度、Table 16 逐行、消融 0/6、GAP LIST |
| 因果与延迟 | [`12-causality-latency.md`](understanding/deepdive-20261108/12-causality-latency.md) | 113 帧 / 904 ms、as-shipped 窗尾、跨窗 +14.71 dB |
| 本轮改动与验收 | [`14-UPDATE-20261108.md`](understanding/deepdive-20261108/14-UPDATE-20261108.md) | 改了什么、168 passed、七份 review 与 R7 终局结论、未做项 |
| 参数 / 权重脚本 | [`verification/scripts/`](verification/scripts/) | `verify_model_params.py`、`verify_real_weights.py`、`verify_causality.py` |
| 阶段报告 | [`P2B-report.md`](verification/phase2/P2B-report.md)、[`P2D-report.md`](verification/phase2/P2D-report.md)、[`P2E-report.md`](verification/phase2/P2E-report.md)、[`P2F-report.md`](verification/phase2/P2F-report.md) | 消融、训练与评测、RealSSA-Sim、指标对拍 |
| 逐句数据 | [`P2E-realssa-sim-metrics_combined.csv`](verification/phase2/P2E-realssa-sim-metrics_combined.csv)、[`P2D-eval-epoch124-contrastive.csv`](verification/phase2/P2D-eval-epoch124-contrastive.csv) | `avg` 行（`:9171` / `:3002`）与 `n_*` 分母可复算 |
| 本轮证据落盘 | [`verification/deepdive-20261108/README.md`](verification/deepdive-20261108/README.md) | 仲裁与实验的原始输出 |
| 环境锁 | [`conda-env-lock.txt`](verification/conda-env-lock.txt) | 124 行 = 9 行注释 + 115 个包；18 个 `file://` 来源被改写成 `pkg==version` |

## 10 未解问题

1. **LRS2 论文列 14.1 / 14.7 怎么补**。两条路：8×3090/A800 按 v2 配方跑满 1000 ep，或向作者索取 LRS2-trained ckpt。当前最近的 8.858 / 9.270 是旧配方单卡，不能当"接近"。
2. **tt 分母到底是 3,000 还是 6,000**。`falcon_lrs2_2mix_paper/tt` 索引 6,000 行 = 3,000 mix × {s1, s2}，历史 CSV 全是 3,000 单向；所有"tt 3,000"的数字都挂在这个未定项上。补法：读索引与 CSV，明确每个数字是 s1 单向还是双向平均。
3. **τ 与 margin 的真实作用**。K≡1 让 τ 退火无效、`relu(sim−0.5)` 恒正；要验证论文的对比学习收益，得按 App A.2 重建 K≥2 的 near 负样本（LibriSpeech + RIR，本地 `rir/` 与 `speech/` 已有素材）。这条同时是 RealSSA-Sim 严格复现的前置条件。
4. **因果性换质量的实际代价**。`paper_causal=True` 已能把几何 lookahead 拉回 15 帧 / 120 ms，但 `cLNhw` 未替换、权重未重训；A/B 的 SI-SNRi 差距目前没有数，论文"因果性换质量"的核心论点仍属未验证。
5. **8 卡真实 DDP 未验**。`find_unused_parameters=True` 的必需性已由 `grad=None` 证明，但条件化 `all_gather` / `sync_dist` / CSVLogger version 冲突只在单进程验过。
6. **RealSSA-Real 窗口化损失量级**。DNSMOS 用 2 s 窗 / 0.5 s hop 拼接，整段前向 20–153 s 会 OOM，混入多少损失未知。
7. **消融仍是 0/6**。Table 7 的阶段数与 Table 9 的 τ 调度最值得先做，2 卡小配方就能跑，不依赖 8 卡。
8. **Falcon-Big 无法与训练侧交叉验证**，以及论文 15 帧与代码对称 padding 的 15 帧只是数值吻合、视觉 "3 future frames" 来源不明——都需要作者澄清，或对上游 `439cbe9` 之前的历史做 diff。
