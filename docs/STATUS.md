# Falcon-fork 复现状态

**2026-11-10：结构、权重加载和多条评测链已有可复核记录；论文的完整训练与主要对照尚未按相同条件全部完成。** RealSSA-Sim 的部分分数与论文数值接近，但 checkpoint 训练来源、通道与论文表格之间仍有未对齐或未确认的条件。

本页以仓库代码 `6e5c1b5` 和 2026-11-10 的检查为基线。更早的实验条件和结果见 [2026-11-08 深挖](understanding/deepdive-20261108/00-SYNTHESIS.md)与 [Phase 1/2 记录](verification/README.md)。

2026-11-11 的块级因果 v4 每块处理 120 ms、不看后续块，接口及当时的训练记录见 [v4 说明](FALCON_STREAMING_120MS.md)；更早的 160 ms v3 见 [历史记录](FALCON_STREAMING_160MS.md)。

2026-11-12 新增的 [论文风格流式模型](FALCON_PAPER_LOOKAHEAD.md) 每次输出 120 ms，额外使用最多 80 ms 音视频前视，并缓存历史；它与 v4 和论文原模型是不同实现。训练使用 RealSSA-Sim 官方 tr/cv、冻结预训练 FRCNN、音频网络从零初始化；每卡 batch 1024、两卡、一次反向、无梯度累积。当前语音主目标是尺度敏感的直接波形信号误差比，静音与常数目标单独处理；具体公式、静音条件及训练和评测口径见[模型说明](FALCON_PAPER_LOOKAHEAD.md)。

2026-11-12 14:55（北京时间）只读核验时，新损失任务运行至 epoch=0、step=7，使用官方训练集；尚未完成一轮验证或测试，不能据此判断收敛或与论文对比。旧 loss 实验及异常定位见[训练异常审查](verification/20261112-lookahead-training-instability.md)，跨轮设计和操作经验见[流式训练经验与复核](verification/20261112-falcon-experience-lessons.md)。运行状态会变化，以实验日志为准：`/root/group-shared/voiceprint/experiments/falcon_paper_lookahead_20261112/sdr-w192-b1024-from-scratch-20261112.log`。

想先理解项目，可看[中文 README](../README_zh.md)；模型、数据集、训练、评测与应用分别有[模型说明](understanding/20261110-model-and-streaming.md)、[数据集说明](understanding/20261110-datasets.md)、[数据训练说明](understanding/20261110-data-and-training.md)和[评测应用说明](understanding/20261110-evaluation-and-apps.md)。本轮逐项发现与可复跑材料见[审查记录](verification/deepdive-20261110/审查记录.md)。

## 原始模型复现的四个层次

| 层次 | 现在能说什么 | 还缺什么 |
| --- | --- | --- |
| 结构和权重 | small 音频模型为 537,697 个参数元素；已有真实权重严格加载及训练侧/Studio 单输入一致性记录 | 其他设备、输入、训练模式与 Big 的一致性待测 |
| 指标评测 | 已保存 LRS2、RealSSA-Sim 和 RealSSA-Real 的部分测试结果及原始表格 | 与论文表格比较时，权重、通道和处理方式尚未全部对齐 |
| 训练 | 训练入口、单卡旧实验及短流程检查有记录 | 论文配置的完整 8 卡训练未完成；真实多卡不对称分桶同步风险未实测 |
| 消融和扩展数据 | 有视觉、时长、窗口等诊断 | 与论文完整消融表逐项同协议的结果仍不足；部分旧脚本本身存在已确认问题 |

## 已有评测结果

SI-SNRi 和 SDRi 都按每条输出分数减混合音频基线，单位 dB。下表汇总已保存的全量评测结果。

| 输入、权重与处理 | 样本 | SI-SNRi / SDRi | 可得出的结论 | 原始表格 |
| --- | ---: | ---: | --- | --- |
| LRS2，发布 `falcon.pt`，整句 | 3,000 | 7.029 / 7.713 | 该数据及权重组合的结果；发布权重的确切训练语料尚缺作者或原训练记录确认 | [P2D 官方权重](verification/phase2/P2D-eval-falcon-full.csv) |
| LRS2，旧配方自训 ep99，整句 | 3,000 | 8.726 / 9.122 | 旧单卡训练结果 | [P2D ep99](verification/phase2/P2D-eval-epoch99.csv) |
| LRS2，旧配方自训 ep124，整句 | 3,000 | 8.858 / 9.270 | 与 ep99 相比经历了更长训练，且使用不同配置 | [P2D ep124](verification/phase2/P2D-eval-epoch124-contrastive.csv) |
| RealSSA-Sim，发布 `falcon.pt`，四通道均值，整句 | 9,169 | 7.774 / 8.479 | 与论文表中数值接近，但通道口径不同，checkpoint 身份尚未确认 | [P2E 四通道均值](verification/phase2/P2E-realssa-sim-metrics_combined.csv) |
| RealSSA-Sim，同一权重，ch0，整句 | 9,169 | 7.977 / 8.452 | ch0 对应论文所述 channel 1 的编号方式；权重训练来源仍需独立证据 | [P2E ch0](verification/phase2/P2E-realssa-sim-ch0-metrics_combined-20261030.csv) |

RealSSA-Sim 两行的 SI-SNRi 有效条数为 9,169，SDRi 与 PESQ 各为 9,166。部分指标计算异常记为 NaN；模型输出仍有 9,169 条。本轮复算四份关键 CSV 的均值和分母，并将其 ID 与**当前**测试索引按文件名比较，未发现重复、缺失或多出；历史运行时索引与当前索引的字节一致性尚待核对。复核结果见[ID 集合检查](verification/deepdive-20261110/评测ID集合核查.log)和[数值检查索引](verification/deepdive-20261110/verification-inventory.json)。

论文 LRS2 表中的 14.1 / 14.7 在本仓库尚无同训练协议的完整复现。当前权重配置曾被作为 LRS3 训练的线索，但 checkpoint 未提供足够训练元数据；配置快照和 epoch 提供了线索，实际训练来源仍待确认。

RealSSA-Real 的无参考评测采用 [full4 结果](verification/phase2/DNSMOS-realssa-real-mean-full4-20261031/)，54 段的 OVRL 均值为 2.7941、P808 均值为 3.3365。当前权重与论文所用权重的身份尚待确认，评测处理也有差异。更早的 full3 因窗口起点错误作废，full4 是修正后的记录。RealSSA-Real 评测的首窗嘴部补零和无输出交叉淡化，与 Studio 不完全相同。

## 模型和数据的几个关键事实

### 输入输出与结构

原始 Falcon 是单目标说话人提取：混合语音和目标嘴部视频输入，输出一条语音。下段描述原始网络，不是新增的论文风格流式网络。原始配置使用 `n_src: 1`，已有测试中 `n_src=2` 形状不匹配。2 秒、16 kHz 的音频产生 251 个 STFT 帧和 129 个频点；50 帧视觉特征经处理后对齐音频帧。两个共享权重的音频块之间进行一次视频加性融合。

音频主干总参数元素 **537,697**；`state_dict` 中 **525 个音频 key** 的 tensor 元素合计 **537,762**，前者统计参数元素，后者统计状态字典中的全部 tensor 元素。视觉前端参数元素 **11,186,688**。small 与 Big 的视觉部分有相同的 137 个状态 key、参数张量相同，但本轮比较发现其中 60 个 BatchNorm 运行状态不同。

### 计算量的范围

`evaluated_mac_params.py` 使用 ptflops 0.7.5 默认 PyTorch 后端，记录 2 秒音频模型约 **10.611 GMAC**，含该工具支持的 LSTM 计数，不含视觉前端。同口径的已有视觉记录为 **15.897 GMAC**（50 帧、88×88），相加约 **26.51 GMAC**。这些数字对应工具支持的算子：本轮探针中显式 `torch.matmul` 被计入，而 `torch.stft`、`torch.istft` 与部分运算符形式的逐元素计算未被计入。`flop_counter` 的另一组数字采用不同的算子计数口径。探针和输出见[计算量算子检查](verification/deepdive-20261110/falcon_ptflops_ops_probe_20261110.txt)。

### 数据、采样和损失

五套开源数据的文件内容、本机目录，以及训练和测试实际使用的划分，见[数据集说明](understanding/20261110-datasets.md)。

paper 版 LRS2 训练索引有 40,000 条目标方向记录，对应 20,000 个混合；验证和测试分别是 10,000 / 6,000 条方向记录。默认有放回抽 10,000 次时，按独立均匀抽样计算，期望覆盖约 **22.12% 的方向记录**，约 **39.35% 的不同混合**；实际采样结果和多卡丢批会改变数值。

现存 LRS2 文件中的 s1、s2 已是混合分量，本轮抽查满足 `mix≈s1+s2`，因此复原现有混音时直接相加两路即可。文件名记录名义增益差，实际 RMS 比还受源信号电平影响；1,000 条系统抽样中有 48 条超出 0–10 dB。[逐条记录](verification/deepdive-20261110/能量比审查/samples.csv)保留了样本路径、哈希和计算值。

NASS 的当前单负样本 `K=1` 分支不使用 β；本轮直接比较 β=0、1、100 的损失和梯度，结果相同。`K=0` 时正项仍可能存在。索引中的 near 在读取异常等情况下可能被跳过，运行时 `K` 应以实际读取为准。详见[数据训练说明](understanding/20261110-data-and-training.md)。

## 需要优先注意的实现问题

| 问题 | 已确认到哪里 | 仍未确认什么 |
| --- | --- | --- |
| 视频 LSTM 把批次轴当作递归序列 | 发布 small 权重下，固定第二条音视频、只改变第一条真实嘴部视频，第二条输出发生变化 | 不同数据、批量大小和 Big 的整体质量影响；修改后是否需要重训以及质量代价 |
| `paper_causal` 额外裁帧 | 2 秒输入的 stride-2 中间时间长度发生变化；通用 causal kernel=1 还可能得到空时间轴 | 修正实现后的前视和模型质量 |
| 时间依赖 | 默认 `cLNhw` 包含全窗统计，窗口内的输出可依赖后续输入 | 端到端实际延迟、不同场景和设备上的效果 |
| 评测覆盖和非有限值 | 合成重复/遗漏 ID 但行数相同的分片仍可 `verified=true`；Inf 的合并计数与单分片口径不同 | 历史运行时索引与当前索引的字节一致性尚待核对 |
| 数据转换与索引 | 临时反例复现同一输出目录从 s1 切到 s2 时旧音频与新嘴部索引错配；独立排序生成器也可静默错配 | 现有数据是否存在同类错配尚待确认；完整人物身份仍待逐项核验 |
| 文件推理 | 32 位整数 WAV 误解释，标准 IEEE float32 WAV 在当前 Python 环境被拒；真实三通道嘴部 `.pt` 不符合离线入口形状 | 其他音频格式和设备环境的完整兼容性 |
| 多卡训练 | 条件式 `all_gather` 和训练后并发写出存在源码层风险 | 真实 Falcon 多卡不同 rank 分桶是否触发错误 |

以上问题没有因本次文档修订而改变。详细触发条件、复现范围和证据链接见[问题与验证状态](verification/deepdive-20261110/问题与验证状态.md)。

## 历史实验怎样引用

- `streaming_consistency.py` 的 `np.rint` 选帧会在某些对齐相位重复/跳过帧。一个 6 秒拼接输入、比较其中 4.5 秒的例子中，只改取帧规则，结果从 **18.201** 变为 **17.492 dB**；这是该样本的比较结果。实验假设源视频帧中心均匀排列；浏览器使用实际采集时间戳。
- `streaming_canonical.py` 是另一实验，两项实验使用不同的处理方式。其相位对比较原本混入相差一个 hop 的同网格组合；修正筛选后，所测 6 秒样本的两两范围由 **17.02–105.66** 改为 **17.02–20.82 dB**。输入本身是拼接音频和重复嘴部帧，也不是自然连续录音。
- `ln_variant.py` 用早期最小响应判断整个早期范围是否为零。归档中该范围的最大响应为 **−75.29 dB**，因此“前 121 帧全部为数值零”的旧说法不受该记录支持。
- `lookahead_ab.py` 的默认行比较同一次输出。另有本轮在一个 CPU 输入上进行的冻结 tag 与当前模型对照，但其他设备和输入仍待比较。
- P2F 的 `_best_lag` 对非零 lag 的波形移动方向相反。归档正文称 50 条 lag 都为 0，逐条缓存已缺失，该结论仍待逐条记录核验；零位移指标与普通评测不因该代码问题自动失效。

各实验的输入、时间轴、checkpoint、阈值和输出范围见[模型说明](understanding/20261110-model-and-streaming.md)与[审查证据](verification/deepdive-20261110/README.md)。

## 测试与运行环境

2026-11-10 在服务器 `/root/miniforge3/envs/falcon`、CPU 条件下运行完整复现套件：**166 passed、2 skipped、38 warnings**。跳过项分别依赖已经不在原 `/root/tmp` 路径下的 s1/s2 评测日志与旧训练曲线。归档训练 CSV 已通过复用原断言的补充检查，但正式套件仍是 166/2。Studio/Space 非慢速测试 **73 passed、2 deselected**；Space 静态测试 **3 passed**；Studio Node 检查 **8 项通过**。[测试日志](verification/deepdive-20261110/复现测试全集.log)和[跳过原因](verification/deepdive-20261110/跳过项核查.log)保留具体条件。

此次完整套件显式执行 `py_compile`；仓库被忽略的 135 个 `.pyc` 在测试时段同一分钟更新，各文件的先前状态和具体写入来源未逐项记录。后续测试可将缓存放到隔离位置。真实浏览器、会议设备、完整 8 卡训练和全部声画同步尚未在本轮验证。

项目环境与执行约定见 [AGENTS.md](../AGENTS.md)；目录和每份文档的用途见[文档清单](DOCUMENTS_INDEX.md)。
