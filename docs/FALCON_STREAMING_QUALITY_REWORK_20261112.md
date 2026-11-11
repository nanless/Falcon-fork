# 120 ms 流式模型质量改造与对照实验

本文记录 2026-11-12 对 RealSSA-Sim 流式训练链路的诊断、代码修改和 A/B/C 对照方案。接口与原训练流程见[流式模型说明](FALCON_STREAMING_120MS.md)，历史证据见[第 100 轮测试](verification/streaming-120ms-realssa-tt-epoch100.md)和[质量诊断](verification/streaming-120ms-quality-diagnosis.md)。

## 已知结果和改造动机

历史第 100 轮流式检查点与开源 Falcon 在 RealSSA-Sim 官方 tt 上逐条配对评测。9,169 条索引中有 9,166 条目标适合计算 SI-SDR，另外三条目标为全零。有效样本中，开源模型平均 SI-SDRi 为 **7.731 dB**，流式模型为 **4.247 dB**，相差约 **3.485 dB**。前者处理完整 2 秒，后者逐块处理 120 ms；这一差距不能单独归因于某一项结构或损失。官方 tt 已用于诊断。

旧训练已正常结束：共 136 轮、6,664 次更新，连续 25 轮未达到至少 0.01 的显著改善后触发早停；这不等同于严格的数学收敛。旧程序按显著改善规则选中第 111 轮，而对全部检查点的完整精度审计显示第 114 轮验证损失数值最低。最终 `final.pt` 与第 136 轮检查点逐字节一致；其官方 tt 全量评测已完成。

第 111 轮官方 tt 的 9,166 条有效样本上，流式 SI-SDRi 均值为 **4.254181 dB**，较开源对照低 **3.477293 dB**；第 114 轮为 **4.256274 dB**，最终第 136 轮为 **4.256584 dB**，最终模型较开源对照低 **3.474890 dB**。第 136 轮与第 114 轮的逐条均值差约 +0.000310 dB；两者的差异很小，尚不能证明流式模型已追平开源模型。检查点、训练结束审计、配对结果和源码差异见[第 111、114、136 轮复测](verification/streaming-120ms-realssa-tt-epoch111.md)。

进一步检查发现以下可分别验证的因素：

1. 历史训练对比项只有正样本对齐，没有推开其他目标的负样本。48 条验证样本的辅助向量匹配 Top-1 数从第 1 轮的 23 条降到第 100 轮的 3 条；后者的目标向量平均余弦相似度约 0.988。这是特定抽样和辅助表征的观察，不能单独证明整个分离模型的质量由表征决定。
2. 全零或常量目标的零均值能量为零，SI-SDR 对其缺乏正常解释。训练与评测需要单独处理并记录有效分母。
3. 历史视觉前端采用因果卷积，而整块到达后当前块内的后续帧已经可用。新候选允许利用这些帧，但仍不读取后续块。
4. 真实嘴图、错误嘴图和中性嘴图的诊断显示输出受视觉条件影响，应保留真实视听对齐并在同一批数据上比较。

同源片段的补充探针将 48 条目标各切成两个 1 秒片段，再分别重复到 2 秒：第 100 轮检查点的配对检索 Top-1 为 1/48，同源配对平均余弦相似度为 0.9615，其他配对为 0.9572。重复片段与真实输入分布不同，这项结果用于辅助表征诊断，不是说话人验证；脚本、结果与日志位于 `/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/identity_audit/half_segment_probe.py`、`half_segment_probe.json` 和 `half_segment_probe.log`。

这些因素构成本轮实验假设，最终效果由配对结果决定。

## 时序和视觉前端

每块包括 16 kHz 音频 1,920 点与 25 fps 视频 3 帧，表示 120 ms。系统收齐当前块后输出对应音频；块内允许互看，跨块仅使用历史状态。块首等待其余音频到达的时间与模型处理耗时是两个不同的量。

| 模式 | 前端版本 | 当前 3 帧怎样处理 | 跨块约束 |
| --- | --- | --- | --- |
| `causal` | v4 | 延续历史因果计算，使用先前帧与当前帧 | 不读取后续块 |
| `chunk_centered` | v5 | 使用前一块末尾 2 帧、当前 3 帧，右侧重复当前块最后一帧 2 次 | 不读取后续块，右侧填充值不是未来视频 |

v5 改变视觉上下文，块接口与输出长度保持一致。检查点和流式状态携带版本，v4/v5 状态不能直接混用；显式 warm-start 可迁移形状匹配的模型参数并适配训练。64 条官方 tt 的直接切换探针中，v4 平均 SI-SDRi 约 4.280 dB，未适配训练的 v5 约 2.370 dB；v5 的训练后效果由 C 与 B 的对照检验。

## 训练与评测的改变

新配置是 [`configs/RealSSAStreamingQuality.yml`](../configs/RealSSAStreamingQuality.yml)，双卡入口仍为 `tools.streaming.train_ddp`。默认 `causal`、320 通道、每卡 batch 1024、无梯度累积、正样本对比和零方差项。模板的训练和验证索引指向原始 RealSSA-Sim tr/cv，使用派生分组时需指定其索引。A/B/C 工具生成独立配置，不改变模板。

### 重建与对比损失

普通目标继续使用负 SI-SDR。常量目标改用按混合信号能量归一化的波形均方误差。默认 `silence_rms_threshold: 0.0` 只把精确常量视作退化；提高阈值也会把低变化量目标归入退化处理。日志分别计数有效语音、静音与非静音常量，默认从对比项中排除退化目标。评测记录无效样本状态并按有效样本计算指标。

正样本项采用余弦对齐，按有效样本计算；双卡训练按全局有效数量处理梯度比例，本次实验 `accum: 1`。可选 `variance_weight` 对目标辅助向量各维的批内标准差设置下限，鼓励表征具有差异；该项的目标音频编码已经 `detach`，直接更新说话人池化头而不直接向音频编码器传递梯度；分离指标如果变化，是经正样本项等其他路径产生的间接效果。该项默认关闭，B/C 权重为 0.1。每卡有效样本少于 2 条时该项为零。它既不是负样本学习，也不能保证分离质量提高。

### 检查点与验证

`warm_start` 载入兼容模型和所需说话人头参数，并重置优化器、调度器与训练步数；来源和摘要保存在新检查点中。`resume` 恢复训练状态，要求代码、数据、协议和模型结构一致，两者互斥。部分 epoch 的续训从该 epoch 开始重新遍历，不恢复中途样本位置。

验证维护实际最低损失和显著改进两个量：前者决定数值最优检查点，达到 `min_delta` 的后者用于调度与早停。因此小于阈值的实际改进仍可保存为数值最优检查点。

## 数据划分和解释范围

RealSSA-Sim 原始训练索引有 100,826 条。按目标音频来源和视频所在目录来源构造派生划分，原始 TAR 和索引保持原样：

| 集合 | 条目 | 视频目录组 | 用途 |
| --- | ---: | ---: | --- |
| 派生 tr | 80,205 | 3,534 | A/B/C 训练 |
| 派生 cv | 10,828 | 441 | 派生验证 |
| 派生 tt | 9,793 | 441 | 后续派生评测 |
| cv-fast | 512 | 210 | 本轮快速开发评测，取自派生 cv |

派生 tr/cv/tt 的目标来源标识散列与视频目录组标识散列互不重叠；这些散列用于识别来源，不是 TAR 内容的完整性摘要；本次未对干扰人和噪声来源建立完整排重，不同视频目录不能证明是不同人，同一视频目录也不能证明必定是同一人，因此目录不是说话人身份标签。对原始 tr 随机审计 4,096 条，均有 LRS3 目标源路径，未发现 `speaker`、`spk`、`identity`、`person` 字段。依据派生训练集目标来源散列计算，每个 1,024 条样本的 batch 中，组合期望约有 4.52 对来自同一目标文件、354 对来自同一视频目录；目录配对数并不代表同一说话人配对数。审计脚本及输出位于 `/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/identity_audit/`，分别为 `audit_metadata.py`、`audit_metadata.json` 和 `audit_metadata.log`。清单、摘要和实际加载检查在 `/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/source_holdout_seed20261112/`。缺失来源审计记录会阻止分组生成。

**warm-start 的数据暴露：**历史第 100 轮权重使用过原始完整 tr，因而已经接触后来划给派生 cv/tt 的来源。分组防止本轮继续训练直接使用这些验证样本，却不能把 A/B/C 称为未见来源或未见说话人测试。要回答这些泛化问题，需要采用未接触相应来源的初始权重重新训练，并核实目标、干扰人和说话人身份划分。

## A/B/C 对照

三组共享数据、初始模型与说话人头、随机种子、更新步数、评测样本和其他训练条件。起点是历史第 100 轮检查点，SHA-256 为 `fb225469889b4581a63e09883df387cfefb82f1c2d0e97c8b39d053323ee50df`。均重置优化器，使用双 A800、每卡 batch 1024、梯度累积 1、学习率 `6.25e-5`、无 warmup，最多更新 50 步。

| 组 | 视觉 | 正样本对齐 | 方差权重 | 比较目的 |
| --- | --- | --- | ---: | --- |
| A | causal / v4 | 开启 | 0 | 新训练链路基线 |
| B | causal / v4 | 开启 | 0.1 | 与 A 比较方差项 |
| C | chunk-centered / v5 | 开启 | 0.1 | 与 B 比较视觉窗口方案及其训练交互 |

C 在 warm-start 时的前向行为已经改变；B/C 的差异同时包含视觉窗口和训练交互。正对比与每卡方差项均不能单独证明模型学会了身份。50 步是短程诊断，并非收敛训练。三组采用相同派生 cv-fast 512 条和各组 `final.pt` 做逐条配对。比较工具核对配置、数据、检查点和评测来源，并按视频目录组重采样计算差值区间。该区间反映按视频目录分组抽样时的样本变动；它不覆盖不同目录之间的同一说话人相关性、跨组共享干扰源或噪声造成的相关性，也不能消除 warm-start 的来源暴露。当前每组只用一个训练 seed，区间不覆盖训练随机种子差异。代码按 seed、epoch 和索引决定样本与增强，但日志未逐步记录样本 ID，GPU 运算也未强制逐位确定，因此不能称为逐位复现。

### 已完成的 50 步结果

全部 512 条 cv-fast 样本有效，覆盖 210 个视频目录组；各组训练均完成 50 次更新。开源 Falcon 同批平均 SI-SDRi 为 **7.9719225 dB**，warm-start 前的流式起点为 **4.9800494 dB**。开源模型使用完整 2 秒，流式模型使用 120 ms 增量块。

| 结果 | A | B | C | B−A | C−B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 流式 SI-SDRi，dB | 4.9810609 | 4.9809474 | 4.5241826 | −0.0001135 | −0.4567648 |
| 视频目录组重采样 95% 区间，dB | — | — | — | [−0.0002598, 0.0000160] | [−0.5394476, −0.3642548] |
| 输出向量匹配目标的 Top-1 | 7/48 | 8/48 | 8/48 | — | — |

起始流式模型在同一 48 条表征样本上的 Top-1 为 7/48。这里是在固定候选集合中匹配同一目标的辅助表征，不是说话人身份验证；与上文历史 48 条诊断使用的样本集合不同。目录组重采样 2,000 次、seed 20261112，以被抽中组中的样本为单位求均值。A/B 差异接近零，不能从本次结果得出方差项带来质量改善；C 比 B 低约 0.457 dB，这一结果反映窗口改变及 50 步训练交互，不证明更长训练后必然如此。A 对起点的变化约 +0.0010 dB，同样未显示实质性的质量改善。

结果和逐条配对见 `/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/abc50-v2/comparison.json`；起点评测为 `initial-eval.json`，表征诊断为 `representation-initial.json` 和 `representation-final.json`。这些均为单 seed、历史检查点已接触来源的短程结果。D 组的对照结果见下一节。

### D：关闭正样本对比项的对照

D 沿用 A 的因果视觉、warm-start、数据和 50 次更新，关闭正样本对比项，只训练重建损失；两组的方差项均为零。D 与 A 使用相同配置种子，但日志没有逐步记录样本 ID，GPU 也没有强制逐位确定。本次 D 的首次配置启动失败保留在 `d50-reconstruction-v1`；修正配置后的完整 50 步及配对评测位于 `d50-reconstruction-v2`。

同一 512 条样本、210 个视频目录组的平均 SI-SDRi：起始模型 **4.98004942 dB**，A **4.98106090 dB**，D **4.98125984 dB**。起始模型与后两组相差 50 次更新；A 与 D 均为第 50 步检查点。

| 配对差值 | 平均差，dB | 视频目录组重采样 95% 区间，dB |
| --- | ---: | --- |
| D − A | +0.00019894 | [−0.00020385, +0.00079792] |
| D − 起始模型 | +0.00121042 | [−0.00944737, +0.01261597] |
| A − 起始模型 | +0.00101148 | [−0.00933788, +0.01234517] |

差别极小且区间均跨过零；在本次单 seed、已见来源的短程条件下，没有足够证据认定添加或移除正样本项带来分离质量收益。这一结果也不能证明两种训练方式无差异或等效，不能排除更长训练或其他数据下的作用。重采样为 2,000 次，以视频目录组为单位，不覆盖训练随机种子、跨组共享干扰源或同一说话人相关性。

`/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/d50-reconstruction-v2/comparison.json` 保存配对明细和区间，`summary-audit.json` 独立从评测行复算均值，并核对训练日志、检查点和退出状态；它调用与原比较相同的 `quality_d_control_v2.compare` 重算配对结果及区间，因此不是对配对或 bootstrap 算法的独立实现验证。

### 后续声纹教师的前端检查

现有 ReDimNet `b0-vox2-lm` 权重选择 `tf` 前端。对冻结模型的实际探针显示其前端使用 `no_grad`，输出不携带对输入波形的梯度；因此不能直接把该模型原样接入需要反向传播到流式输出的训练损失。下一步若采用声纹教师，应先验证可导前端和与原权重的特征一致性。探针和结果分别为 `/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/identity_audit/redimnet_gradient_probe.py` 与 `redimnet_gradient_probe.json`。

## 运行新的对照

以下命令在服务器隔离 checkout 执行。将 `RUN` 改成从未使用的独立路径。准备工具只生成配置与身份清单，不启动训练。

```bash
set -euo pipefail
REPO=/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/repo
BASE=/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112
DATA="$BASE/source_holdout_seed20261112"
RUN="$BASE/abc50-example-new"
CKPT=/root/group-shared/voiceprint/experiments/falcon_streaming_120ms_realssa_full_20261111/epochs/epoch-0099-step-00004900.pt
PY=/root/miniforge3/envs/falcon/bin/python
TORCHRUN=/root/miniforge3/envs/falcon/bin/torchrun
cd "$REPO"
PYTHONPATH=. "$PY" -m tools.streaming.prepare_quality_abcs \
  --base-config configs/RealSSAStreamingQuality.yml \
  --train-index "$DATA/tr.json" --valid-index "$DATA/cv_fast512_seed20261112.json" \
  --holdout-manifest "$DATA/manifest.json" --warm-start "$CKPT" --output-dir "$RUN" \
  --max-steps 50 --lr 6.25e-5 --warmup-steps 0 --seed 42
PYTHONPATH=. "$PY" -m tools.streaming.prepare_quality_abcs --output-dir "$RUN" --verify
for ARM in a_causal_positive b_causal_variance c_centered_variance; do
  mkdir -p "$RUN/$ARM"
  PYTHONPATH=. "$TORCHRUN" --standalone --nproc_per_node=2 \
    -m tools.streaming.train_ddp --config "$RUN/$ARM.yml" > "$RUN/$ARM/train.log" 2>&1
  PYTHONPATH=. "$PY" -m tools.streaming.evaluate_realssa \
    --tar-index "$DATA/cv_fast512_seed20261112.json" \
    --streaming-checkpoint "$RUN/$ARM/final.pt" --device cuda --threads 2 --limit 0 \
    --output "$RUN/$ARM/paired-dev512.json"
done
PYTHONPATH=. "$PY" -m tools.streaming.compare_quality_abcs \
  --experiment-dir "$RUN" --a "$RUN/a_causal_positive/paired-dev512.json" \
  --b "$RUN/b_causal_variance/paired-dev512.json" \
  --c "$RUN/c_centered_variance/paired-dev512.json" \
  --seed 20261112 --bootstrap 2000 --output "$RUN/paired-summary.json"
```

准备工具记录输入、配置、模型代码及起始检查点摘要；日志和检查点保存在各组目录。`tools.streaming.inspect_quality_embeddings --help` 提供辅助向量分析入口，比较不同组时应固定索引和随机种子。

### 复现 D 对照

D 工具从已完成的 A/B/C 清单生成 D 配置，核对除目标函数及输出路径外的配置一致性，随后配对起点、A、D；工具本身不启动训练。下面示例复用本次 `abc50-v2` 的 A 和起点评测，`DRUN` 必须是一个未使用路径；准备工具还会核对 A/B/C 清单中记录的代码、数据和权重摘要。

```bash
set -euo pipefail
REPO=/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/repo
BASE=/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112
ABC="$BASE/abc50-v2"
DRUN="$BASE/d50-reconstruction-example-new"
DATA="$BASE/source_holdout_seed20261112"
PY=/root/miniforge3/envs/falcon/bin/python
TORCHRUN=/root/miniforge3/envs/falcon/bin/torchrun
cd "$REPO"
PYTHONPATH=. "$PY" -m tools.streaming.quality_d_control_v2 prepare --abc-dir "$ABC" --output-dir "$DRUN"
PYTHONPATH=. "$PY" -m tools.streaming.quality_d_control_v2 verify --output-dir "$DRUN"
PYTHONPATH=. "$TORCHRUN" --standalone --nproc_per_node=2 \
  -m tools.streaming.train_ddp --config "$DRUN/d_reconstruction.yml" > "$DRUN/train.log" 2>&1
printf '0\n' > "$DRUN/exit.txt"
PYTHONPATH=. "$PY" -m tools.streaming.evaluate_realssa \
  --tar-index "$DATA/cv_fast512_seed20261112.json" \
  --streaming-checkpoint "$DRUN/final.pt" --device cuda --threads 2 --limit 0 \
  --output "$DRUN/eval.json"
PYTHONPATH=. "$PY" -m tools.streaming.quality_d_control_v2 compare --output-dir "$DRUN" \
  --initial "$ABC/initial-eval.json" --a "$ABC/a_causal_positive/eval.json" \
  --d "$DRUN/eval.json" --seed 20261112 --bootstrap 2000 --output "$DRUN/comparison.json"
PYTHONPATH=. "$PY" -m tools.streaming.audit_quality_d_summaries --output-dir "$DRUN" \
  --initial "$ABC/initial-eval.json" --a "$ABC/a_causal_positive/eval.json" \
  --d "$DRUN/eval.json" --comparison "$DRUN/comparison.json" --output "$DRUN/summary-audit.json"
```

汇总审计读取 A 与 D 的训练日志、`exit.txt`、检查点和评测输出，独立核算评测行均值，并复用同一比较实现核对配对结果；上面的 `exit.txt` 只在训练进程成功退出后写入。

## 已完成的链路和 CPU 检查

- A 和 C 各完成真实数据双卡 2 步试训，并续训至第 3 步，使用 16 条验证样本；这验证训练、保存、续训和验证链路，不能代表 50 步对照结果。日志位于 `/root/group-shared/voiceprint/experiments/falcon_quality_rework_20261112/abc2/` 各组的 `train.log` 和 `resume.log`。
- 损失、双卡空有效样本、评测来源和检查点定向测试记录位于同一实验目录的 `pytest-loss-final.log`、`eval-provenance-tests-final.log` 和 `merge-final-tests.log`。
- 全仓回归分两个独立进程完成：非 metrics 部分 **359 passed、2 skipped**，metrics 部分 **10 passed**，日志为实验目录下 `regression-no-metrics.log` 和 `metrics-standalone.log`。此前单进程全跑长时间停滞后已停止，记录为 `full-regression.log` 和 `full-regression-stopped.txt`；两次分开运行给出了完整测试分区结果。
- v4 与历史实现的真实输入前向一致性，以及 v5 未来块扰动和状态检查，记录于 `/root/group-shared/voiceprint/experiments/falcon_quality_revision_20261112/model_visual_probe/`。
- CPU 节奏测试使用 Xeon Platinum 8336C、4 个 PyTorch 线程、单条真实片段的 16 个 120 ms 块循环、20 次预热和 500 块计时。v5 使用从 v4 迁移的权重，未完成 v5 适配训练。处理耗时 p50 **20.60 ms**、p95 **27.60 ms**、p99 **29.86 ms**、最大 **31.63 ms**，500 块均低于 40 ms；从计划释放到完成的最大值为 **31.72 ms**。计时包含已有 96×96 嘴图处理、模型和输出复制，不包含采集、人脸检测跟踪、网络与播放。证据为 `/root/group-shared/voiceprint/experiments/falcon_quality_revision_20261112/model_visual_probe/warmstart_v5_cpu4_paced500.json`，结果对应该机器的单路测试条件。

最终质量与收敛轮数由后续完整实验确定。
