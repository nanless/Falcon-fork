# RealSSA-Sim：流式模型训练记录

当前模型为 120 ms v4，完整说明见 [120 ms 流式模型说明](FALCON_STREAMING_120MS.md)。本文的数据说明和“训练配置与两卡试训”使用当前入口；明确标为 v3 的试训与对照结果保留原始实验条件。

这条路径使用 RealSSA-Sim 的原始 TAR，直接读取真实混合音频、目标音频和目标人物的嘴图，训练 FalconStreaming v4。模型在 120 ms 块内联合使用音视频，块间只使用历史状态；公开 Falcon 的视觉权重用于初始化视觉前端并冻结，两个训练进程各占一张 GPU。

## 数据和目录

- 原始分片：`/aispeech-cfs/voiceprint/data/multimodal/realssa_sim/{tr,cv,tt}`，本次清点分别为 202、19、19 个 TAR。TAR 元素按样本包含 mix、target、mouth 和元数据；训练直接按索引读取，不必全量解压。
- 本轮已建立三个划分的全量 TAR 成员索引：训练 100,826 条、验证 9,133 条、测试 9,169 条，分别对应 202、19、19 个分片；三个划分的缺字段样本及各自内部重复 ID 均为 0。索引按成员头检查，并非全部音视频 payload 逐条解码或逐字节哈希。索引目录：`/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/index/full-S1xPSK/`。
- 使用第 0 通道（论文从 1 开始编号时的 channel 1）；混合和目标取同一通道。训练使用 `tr`，验证使用 `cv`；`tt` 保留官方测试划分；目标源重叠情况见下文。
- 先前真实数据试训及其复制的索引、日志和 checkpoint 存在 `/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/`；本轮正式训练位于 `/root/group-shared/voiceprint/experiments/falcon_streaming_120ms_realssa_full_20261111/`。音视频、数据索引和权重留在服务器，不加入 Git。
- 2026-11-11 的试训最初记录在 `/root/group-shared/voiceprint/data/falcon_realssa_trial_20261111/`，现已迁至上述实验目录。下文保留当时的命令和结果语境，文件路径统一指向迁移后位置。

## 与原训练设置的关系

| 项目 | 本路径 | 原论文或开源实现 |
| --- | --- | --- |
| 音频及嘴图 | 2 秒 16 kHz 音频、50 张 25 fps 嘴图 | 论文使用相同长度和帧率 |
| 尾块 | 模型内部输入 17 个 120 ms 块，补齐 40 ms；重建和对比正项均只使用原始 2 秒 | 原始 Falcon 处理 2 秒窗口 |
| 视觉初始化 | 从公开 `falcon.pt` 提取的 FRCNN，加载后冻结 | 论文使用唇读预训练 FRCNN 并冻结；本文件来自公开分离检查点 |
| 训练增强 | 灰度、随机 88×88 裁剪和水平翻转 | 对照原训练视觉预处理；验证采用中心裁剪 |
| 训练裁剪 | 随机起点按 640 采样点/40 ms 网格选择 | 上游数据加载器允许音频在任意采样点起始 |
| 不足 2 秒 | 音频与原始嘴图补零至 2 秒 | 对应论文附录；上游旧目录加载器默认过滤短音频、嘴图不足时报错 |
| 优化器 | AdamW、目标学习率 0.001、预热 100 次更新、weight decay 0.01；2 卡×1024、无梯度累积，全局 batch 2048 | 论文 8 卡×4，batch 32 |
| 重建损失 | 调用原项目 `pairwise_neg_sisdr` | 论文采用负 SI-SNR |
| 对比损失 | 支持仅正项；将提取声音和目标声音分别编码并复用 NASS 正项 | 论文还使用每个近场干扰源作为负样本，query 特征来自原分离器 |
| 随机抽样 | DDP 按 epoch 分布式洗牌并丢弃末尾不足的批次 | 原训练入口默认每 epoch 有放回抽样 10,000 次 |

公开 TAR 没有单独的近场干扰语音：`components[].path` 是生成时的源文件路径，不能作为已对齐的干扰波形。以混音减目标所得的是多个干扰和噪声之和；它不是论文所用的逐个近场负样本。训练器用明确的目标开关记录仅重建或正项对比方案，不能把它们称作完整复现论文对比损失。

`frcnn_128_512.backbone.pth.tar` 的 SHA256 为 `980339a9351cef0cdc39ea6e1257c59681ceac813f6e091178de5d1afcfd93bd`；元信息显示它从 `falcon.pt`、epoch 714 提取。该来源不等同于已单独核实的原始唇读预训练文件。

## 历史 v3 试训记录

2026-11-11 在两张 A800 上以第一训练 TAR 的 500 条索引为候选、第一验证 TAR 中 16 条为验证集，实际完成 2 次优化。参数为每卡 batch 4、梯度累积 4，等效 batch 32；两次更新各消耗32次样本，属于有界链路检查。训练使用负 SI-SNR 重建项，尚未使用说话人对比项。最后一次微批损失 `-0.718479`，16 条验证片段的平均重建损失 `-1.778170`；这些是损失，不是 SI-SNRi 测试结果或模型收敛证据。

检查点：`/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/run-RsDkC7/checkpoint.pt`。其 SHA256 为 `a82c468f8b43a856da8a5e3a03825803f747872e838bc4aacb80f0494f207644`；同目录保留 `train.log`、两份索引副本和验证样本回放 WAV。checkpoint 中的 `training_objective`、视觉来源、数据索引哈希、GPU 数和两步更新均可检查。已使用该 checkpoint 在 CPU 上回放验证录音，91,520 个输出采样与输入长度一致。

旧试训只使用小范围数据与重建损失；切换训练集、目标或验证集时，应从对应新协议初始化训练，不将它作为严格断点续训。块级模型与原始公开 Falcon 的训练检查点不兼容，独立比较要在同样的数据、目标、通道和评测时长下完成。

第二次试训使用 `--positive-contrastive`：在前一模式基础上，将输出与目标的有效 2 秒波形重新编码，经训练专用说话人池化头计算原 NASS 正项，正例侧停止梯度，权重为 1。该 query 并非论文原模型的掩码后隐变量，并且没有近场负项。辅助头只存在训练检查点；当时的 v3 模型状态可供当时的 v3 流式推理，当前 v4 入口需要 v4 检查点。

在相同 500 条训练候选及 16 条验证样本上，两张 A800、每卡 batch 4、梯度累积 4 完成 2 次更新。最后一个微批总损失为 `-0.716402`；固定验证集上的重建损失为 `-1.777973`。此次为未完成一个 epoch 的短试验，验证不会推动按 epoch 的学习率调度、提前停止计数或最佳模型选择。检查点位于 `/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/positive-batch32-gN7Yoa/checkpoint.pt`，SHA256 `bc533bea72a6cf8f11e181b26c07957f0688f6d2da15ff01ecf7e77198fe4d0b`；同目录包含原子保存的阶段检查点、日志及与训练记录哈希一致的数据索引副本。该数值只是链路验证。

## 准备数据索引

以下在服务器执行。索引存储 TAR 成员偏移、大小和文件身份，`--workers` 限制并发，`--cache-dir` 为逐分片断点；输出文件采用新路径。独立检查不同 split 的完整数和缺字段数后再开始长期训练。

```bash
cd /root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
SRC=/aispeech-cfs/voiceprint/data/multimodal/realssa_sim
INDEX=/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/index/example
mkdir -p "$INDEX"
"$PY" -m tools.streaming.realssa_tar --src-root "$SRC" --split tr --index "$INDEX/tr.json" --workers 4 --cache-dir "$INDEX/cache"
"$PY" -m tools.streaming.realssa_tar --src-root "$SRC" --split cv --index "$INDEX/cv.json" --workers 4 --cache-dir "$INDEX/cache"
"$PY" -m tools.streaming.realssa_tar --src-root "$SRC" --split tt --index "$INDEX/tt.json" --workers 4 --cache-dir "$INDEX/cache"
```

每次构建的最终索引路径不可已存在；中途退出后可沿用 sidecar 缓存并换新索引名。源文件的路径、大小或修改时间改变时，需要重新扫描对应分片。

## 迁移后的本地样本索引

本次迁移保留了历史 `tr`、`cv` 目录及原来的 `mix.json`、`target.json`、`near.json`，历史索引内部仍记载当时的旧路径。为读取迁移后的小样本文件，另建了适配索引目录：

`/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111-adapted-indexes/`

其中 `tr/` 和 `cv/` 各含三份 JSON；它们只改写以旧实验根目录开头的媒体路径，不复制音视频，也不改写历史索引。`relocation-report.json` 记录源文件和新文件的 SHA-256 及各自的替换数：每个划分的 mix 64、target 128、near 64，共 512 个路径。工具校验这 512 个目标文件存在；此项校验不等于逐个文件内容哈希。

使用 `TwoSecondDataset` 的验证模式实际读取所有满足长度条件的样本：训练划分 56 条、验证划分 63 条；混音和目标均为 32,000 采样点，嘴图为 50×88×88，读取结果均为有限值。该读取覆盖 mix、target 和嘴图，没有解码 near 波形或短于两秒的样本。这些本地小样本索引用于目录模式；上面的全量 TAR 索引是另一组文件，YAML 示例使用全量 TAR 索引，不使用此适配目录。

如需重新生成适配索引，给工具提供原始实验目录、迁移前根目录、迁移后根目录和一个尚不存在的输出目录，例如：

```bash
cd /root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
TRIAL=/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111
PARENT=$(mktemp -d /root/group-shared/voiceprint/experiments/falcon-index-relocation-XXXXXX)
"$PY" -m tools.streaming.relocate_local_indexes \
  --source-root "$TRIAL" \
  --old-root /root/group-shared/voiceprint/data/falcon_realssa_trial_20261111 \
  --new-root "$TRIAL" --output-root "$PARENT/indexes"
```

工具在创建输出前检查六份源索引和所有匹配的目标文件，并拒绝已存在的输出目录。报告中的输出路径可能解析为 `/shared/group-shared/...`；它与服务器上的 `/root/group-shared/...` 是同一挂载位置。重新生成的索引仅替换命中的旧目录前缀，其他字符串保留原值；不会修改历史 checkpoint 或使其绕过严格续训协议。

## 训练配置与两卡试训

训练入口为 `tools.streaming.train_ddp`，配置文件为 [configs/RealSSAStreamingDDP.yml](../configs/RealSSAStreamingDDP.yml)。配置使用全量 `tr` 和 `cv` 索引，冻结公开权重提取的视觉前端，启用重建损失和对比正项；两卡每卡 1024 条、`accum: 1`，全局 batch 为 2048。配置使用完整验证集、最多 300 个 epoch，并按验证损失调整学习率和提前停止；详细训练记录和当前启动命令见 [120 ms 模型说明](FALCON_STREAMING_120MS.md)。

当前模型每块使用 1920 点音频与 3 帧嘴图，2 秒训练片段补齐为 17 块；`model_channels: 320` 可在配置中调整音频通道数，dilation 固定为 1/2/4/8/16/32。配置键采用命令行长选项的下划线形式，例如 `batch_size` 对应 `--batch-size`。文件内相对路径相对于 YAML 文件所在目录解析；命令行给出的相对路径按执行命令时所在目录解析。未知字段、重复字段和错误类型会被拒绝。`max_steps` 限制本次进程新增的更新步数；`0` 表示不设置步数上限。`validation_limit: 0` 表示使用完整验证集。

下面是有界的两步烟测：通过命令行覆盖步数、验证条数和输出位置；正式训练使用完整验证集。每次创建独立实验目录，已有输出文件不会被覆盖：

```bash
cd /root/code/github_repos/Falcon-fork
PYROOT=/root/miniforge3/envs/falcon/bin
INDEX=/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/index/full-S1xPSK
RUN=$(mktemp -d /root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/run-XXXXXX)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none \
"$PYROOT/torchrun" --standalone --nproc_per_node=2 -m tools.streaming.train_ddp \
  --config configs/RealSSAStreamingDDP.yml --max-steps 2 --validation-limit 16 \
  --output "$RUN/checkpoint.pt" --checkpoint-dir "$RUN/epochs"
```

命令行参数优先于 YAML，例如加 `--max-steps 3` 可将本次进程新增的更新上限改为 3。数据源、验证源、视觉初始化和损失目标分别是互斥组；命令行选择该组某一项时会替代 YAML 中该组的选择。例如 `--reconstruction-only` 会替代配置里的 `positive_contrastive`。若要持久调整设置，可复制 YAML 再修改，并以 `--config` 指向新文件。配置路径、文件 SHA256 和最终生效参数会记录在 checkpoint 中。

训练器检查训练 split 为 `tr`、验证 split 为 `cv`；样本身份包含 split。训练和验证有 84 个同名 ID，经逐项对比四类数据及目标来源均不同，不能仅凭裸 ID 认定相同样本；训练器对同名候选检查目标来源和目标 WAV 字节是否相同。对全部划分目标来源元数据的审计已发现跨 split 源路径重合，具体数量与含义见下文。checkpoint 还记录索引及模型、训练器和数据器哈希、视觉来源、训练目标、设备数、batch 和随机种子。评测时使用官方 `tt` 划分，与公开 Falcon 使用相同样本及指标；真实口型准备、时间窗口和模型的不同架构需在比较表中列出。

### 恢复训练

当前版本生成的 checkpoint 可通过 `--resume` 恢复模型、优化器和调度器状态，并使用新的 `--output` 与 `--checkpoint-dir`。例如，前一次训练完成 2 步后，在同一个 shell 保留上面的 `RUN` 变量，以下命令新增 1 步，结束于累计第 3 步：

```bash
RESUME_RUN=$(mktemp -d /root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/resume-XXXXXX)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none \
"$PYROOT/torchrun" --standalone --nproc_per_node=2 -m tools.streaming.train_ddp \
  --config configs/RealSSAStreamingDDP.yml --resume "$RUN/checkpoint.pt" \
  --validation-limit 16 --max-steps 1 --output "$RESUME_RUN/checkpoint.pt" --checkpoint-dir "$RESUME_RUN/epochs"
```

恢复时会检查训练协议、数据索引、设备数及代码哈希；部分 epoch 停止后会从该 epoch 起点重新采样，不是逐样本精确续训。优化器当前学习率和调度器状态从 checkpoint 恢复；若显式指定 `lr`，它必须与 checkpoint 保存的目标学习率一致。`lr: null` 或不指定时使用 checkpoint 的目标学习率，当前学习率仍从优化器恢复。

下文 v3 历史试训检查点对应 160 ms 模型和旧训练协议；当前 v4 模型、会话和严格续训入口都不直接加载 v3 检查点。历史结果仍可读取；使用历史权重推理或恢复当时训练时，需要对应的旧版代码和协议。

## 历史 v3 YAML 配置双卡验收（2026-11-11）

当时的 v3 代码使用当时示例 YAML 和 RealSSA-Sim 全量训练、验证索引，在两张 A800 上以每卡 batch 4、梯度累积 4 完成 2 次更新；读取 `step2.pt` 后新增 1 次更新至累计第 3 步。两次均以固定前 16 条验证，重建损失分别为 -1.750885 和 -1.811937。实验记录位于 `/root/group-shared/voiceprint/experiments/falcon_yaml_ddp_final_20261111/`，包含 `train.log`、`step2.pt`、`resume.log` 和 `step3.pt`。训练配置、索引适配、DDP、数据及续训回归测试共 46 项通过（72.77 秒）。上述为训练、验证和续训链路检查，并非完整模型训练结果。

## 历史 v3 双卡试训与配对检查

当时的 v3 代码在真实训练首分片候选 500 条、cv 验证分片 16 条上，以每卡 batch 4、累积 4 步运行 2 次更新，再用同一协议恢复并完成第 3 次更新。恢复属于部分 epoch 重放，不是从数据迭代器的原位置继续。训练和恢复日志、分步 checkpoint 与评测保存在：

`/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/final-batch32-cKDxaO/`

- `checkpoint.pt`：第 2 步，SHA256 `bc46ca91829259c4a52d1b66eeda8c92dde53401b36419516cd6af5ea9398ad9`。
- `resumed.pt`：第 3 步，SHA256 `64071e68255289825af9c05a5be8b3343e464c71d3771e7bd3008aa8ded83ef5`。
- 最后一次微批训练总损失 `-1.797306`，16 条验证的平均重建损失 `-1.833156`；检查点记录损失目标为重建加正项、没有近场负项。

`tools/streaming/evaluate_realssa.py` 从固定验证索引逐条读取同一个中心裁剪的 2 秒音频、目标及 50 帧嘴图，分别评估作者公开 `falcon.pt` 与上面的第 3 步 checkpoint。16/16 条有效，去均值 SI-SDRi 均值分别是 **7.73851 dB** 与 **0.03909 dB**；逐条数据和输入、权重、代码哈希见该目录的 `paired-cv16-final-v2.json`。公开 Falcon 对整段 2 秒做一次前向，v3 采用连续 160 ms 块，新模型只训练了 3 次更新；这次结果验证配对评测和载入链路，并不代表训练收敛或同等上下文下的最终比较。

当前 v4 检查点的评测示例，输出采用新文件名；历史 v3 检查点使用当时的 v3 代码评测：

```bash
"$PYROOT/python" -m tools.streaming.evaluate_realssa \
  --tar-index "$INDEX/cv.json" --streaming-checkpoint "$RUN/checkpoint.pt" \
  --baseline-checkpoint /root/.cache/falcon_weights/falcon.pt \
  --limit 16 --device cpu --threads 4 --output "$RUN/paired-cv16.json"
```

实现回归对 TAR 索引、增强、双卡训练关键逻辑、原转换器及配对评测最终代码共 **27 项通过**（[回归日志](verification/realssa-streaming-training-20261111-pytest.log)）。两卡实训的原始音视频、嘴图、训练结果音频、模型权重及数据索引留在服务器授权目录。

## 历史 v3 全量索引双卡试训

2026-11-11 使用全量 `tr.json`（100,826 条候选）和全量 `cv.json`（9,133 条），指定 `--validation-limit 16`，以每卡 batch 4、梯度累积 4、正项对比模式完成两次优化，并以相同协议恢复至第 3 次。仅执行了少量训练步骤，并非遍历全量数据或完成模型训练。16 条验证的重建损失：第 2 步 `-1.750885`，续训第 3 步 `-1.811937`。

实验目录为 `/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/full-index-positive-v3-20261111/`，包含 `commands.txt`、`step2.log`、`step3.log`、`step2.pt`、`step3.pt`、每阶段 checkpoint 和源码哈希。checkpoint 中记录全量训练/验证索引、`validation_limit=16`、等效 batch 32、冻结视觉来源、对比正项、无 near 负项以及 84 个同名候选的核查结果。当时的 v3 `StreamingSession` 在 CPU 加载第 3 步模型并产生了完整 2,560 点有限输出。

在验证索引前 16 条、每条同一中心裁剪 2 秒、channel 0 的配对检查中，公开 Falcon 的平均 SI-SDRi 为 **7.73851 dB**，本次第 3 步流式模型为 **0.01787 dB**，16 对均有效；结果和逐条记录在本目录 `paired-cv16.json`。公开模型观察完整 2 秒，而新模型按 160 ms 分块，且训练仅 3 次更新；这是流程对照，不是最终模型质量或同时间上下文下的模型优劣结论。这 16 条也用于训练时验证，并非独立测试集。


## 官方划分中的目标来源重叠

本轮读取 tr/cv/tt 全部 119,128 条样本的 metadata，240 个分片均完成，目标路径字段缺失和解析错误均为 0。归一化目标原始路径并按不同路径去重后，tr 与 cv 有 **4,456** 个共同目标源路径，tr 与 tt 有 **4,430** 个，cv 与 tt 有 **538** 个。对应母视频目录标识的交集分别是 2,630、2,609、2,067；目录标识不是已经验证的说话人身份。这些数字是不同路径或目录的数量，不是重合样本条数。

相同源路径说明 metadata 指向同一原始目标文件；实际音频片段边界和处理是否相同，还需逐项核查。官方三划分仍可用于复现公开协议，但这些结果说明其测试集不能视为已证明的目标源独立泛化评测。需要按目标源或更严格条件隔离时，应另建并标识独立实验划分，不替换官方索引。聚合摘要位于服务器数据索引目录 `source-audit-summary.json`；脚本 `tools/streaming/audit_realssa_sources.py` 可按三个索引复查，且输出路径/视频目录的哈希，不输出明文来源或音视频数据。

按样本行计数，共源文件覆盖 tr/cv 分别 **6,156/100,826** 和 **4,575/9,133** 条；tr/tt 分别 **6,220/100,826** 和 **4,584/9,169** 条；cv/tt 分别 **559/9,133** 和 **557/9,169** 条。这里“共源”表示 metadata target 指向同一原始路径，不表示处理后的音频、混音或最终截取片段逐样本相同。聚合汇总见 [不同源数量](verification/realssa-source-audit-summary-20261111.json)及[受影响样本行数](verification/realssa-source-overlap-rows-20261111.json)。

本轮使用的验证前 16 条中，6 条目标源路径出现在完整训练候选池中。按实际随机种子重建本次第 1、2 步的 64 个训练样本，目标源与这 16 条无交集；部分 epoch 恢复的第 3 步重放前 32 条。这是对本次短训采样的检查，不保证未来遍历全量训练数据时仍无来源重叠。

另从 tr/cv 共源中以 `random.Random(19)` 抽查 5 个不同目标源，两侧嘴图张量均逐元素相同；5 对混音 WAV 与目标 WAV 的字节均不同，场景也不同。此抽样显示同一视频画面被用于不同划分的合成场景，并不表示每条混音或混响后的目标波形相同，也不能由此推断所有说话人的划分情况。

训练器说明修正后，以同一全量索引和相同参数再次完成双卡 2 步与续训第 3 步；最终目录为 `/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/final-fullindex-Knc2Jj/`，第 2 步 `step2.pt` 的 SHA256 为 `e2d3008666cc121d8e265d215cbf4fad6dca12e3efedb02cbfaa4b5796c3e46b`，第 3 步 `step3.pt` 的 SHA256 为 `aeff3345d06cdf1b9c0782ea89d4557962508a7fc2cd810a18927a0cac4ba5b3`。第 3 步与上一组 `full-index-positive-v3-20261111/step3.pt` 的模型状态逐张量完全相同；上面的 16 条配对评测对应上一组检查点。当时的续训使用了对应旧版训练器、相同数据索引与训练协议；旧 checkpoint 与当前训练器的严格续训兼容性见上文。

重做来源审计可使用以下命令；输出包含样本 ID 和源路径的未加盐哈希，逐条记录保存在服务器，不代表可以公开发布。输出目录使用新路径，缓存可用于中断后的重跑。

```bash
INDEX=/root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/index/full-S1xPSK
AUDIT=$(mktemp -d /root/group-shared/voiceprint/experiments/falcon_realssa_trial_20261111/source-audit-XXXXXX)
/root/miniforge3/envs/falcon/bin/python -m tools.streaming.audit_realssa_sources \
  --tr-index "$INDEX/tr.json" --cv-index "$INDEX/cv.json" --tt-index "$INDEX/tt.json" \
  --output-dir "$AUDIT" --cache-dir "$AUDIT/cache" --workers 4
```
