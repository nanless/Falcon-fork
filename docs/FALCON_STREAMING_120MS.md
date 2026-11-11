# 120 ms 音视频块级流式模型

当前流式模型每次接收 120 ms 混合音频和对应的 3 张嘴图，输出完整的 120 ms 目标语音。模型允许当前块内音频和视觉特征互相参考，块与块之间通过每个会话独立的历史状态衔接，当前输出不依赖后续块。

| 项目 | 当前实现 |
| --- | --- |
| 音频 | 16 kHz、单声道、每块 1920 个 float32 采样 |
| 嘴图 | 25 fps、每块 3 帧；原始 96×96 uint8，预处理为 88×88 灰度 |
| 内部音频帧 | 8 ms 一帧，每块 15 帧 |
| 音频通道 | 320，膨胀系数 1、2、4、8、16、32 |
| 历史 | 音频卷积缓存、最近音频与视觉缓存；不同会话分别保存 |
| 检查点 | v4 原非负合成，v5 块内居中视觉，v6 有符号频响残差，v7 增加已完成块的注意力缓存。由格式与模型配置共同决定加载方式 |
| 输出保护区 | 0，输出完整块 |

`FalconStreaming.forward_chunk` 接收一块音视频与上块状态，返回当前音频与新状态；`StreamingSession` 提供带序号、音频采样时钟和嘴图时间戳检查的接口。一个块收齐后才运行模型；音频块内允许互看，块首样本等待完整块，与块处理耗时是两个不同的时间量。视频帧应按其所在 40 ms 区间选择，接口中的时间戳规则见 `tools/streaming/session.py`。

## 历史 v4 训练数据和方法

训练入口为 `tools.streaming.train_ddp`，可编辑的配置为 [configs/RealSSAStreamingDDP.yml](../configs/RealSSAStreamingDDP.yml)。读取 RealSSA-Sim 原始 TAR 的训练、验证索引；训练集 100,826 条，验证集 9,133 条。原始 2 秒训练样本包含 32,000 个音频点与 50 张嘴图；为按完整块运算，补到 17 块，即 2.04 秒和 51 帧，重建损失只使用原有的 2 秒。

默认使用 2 张 A800，每卡 1024 条，`accum: 1`，全局 batch 2048。AdamW 目标学习率为 0.001，前 100 个参数更新线性预热。每完整轮使用完整验证集，验证损失较最佳结果改善至少 0.01 才更新最佳检查点；连续 6 次不满足改善条件时，学习率减半；连续 25 次不满足时早停。300 轮是上限，达到上限本身不表示已经收敛。训练中每轮记录更新次数、验证损失、学习率和检查点。

公开 RealSSA-Sim 包没有分别对齐的近场干扰源波形；当前重建损失加说话人对比正项，没有论文所需的逐个近场负项。官方划分存在目标源路径重叠，详见[数据与历史训练说明](REALSSA_STREAMING_TRAINING.md)。

从服务器项目根目录启动时，配置中的输出路径须指定到本次独立的实验目录：

```bash
cd /root/code/github_repos/Falcon-fork
RUN=/root/group-shared/voiceprint/experiments/falcon_streaming_120ms_realssa_20261111
mkdir -p "$RUN"
PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  /root/miniforge3/envs/falcon/bin/torchrun --standalone --nproc_per_node=2 \
  -m tools.streaming.train_ddp --config configs/RealSSAStreamingDDP.yml \
  --output "$RUN/final.pt" --checkpoint-dir "$RUN/epochs"
```

训练器检查点的 `positive_contrastive_definition` 仍沿用带 `v3_audio_proj` 字样的历史字段值；本轮训练实际使用当前 v4 模型的音频投影，模型版本和结构以检查点的 `format` 与 `model_config` 为准。

2026-11-11 启动的正式训练实际目录为 `/root/group-shared/voiceprint/experiments/falcon_streaming_120ms_realssa_full_20261111/`；本页命令中的路径是可修改的新实验示例。检查点记录模型配置、数据与代码摘要、训练和验证协议、优化器、调度器、步数和最佳验证值。正式试验每个 epoch 结束后保存，续训应保持数据和协议一致并选择新的输出位置。

## CPU 实测与质量评估

320 通道随机初始化模型在 Intel Xeon Platinum 8336C、4 个 PyTorch 线程、预取音视频输入的测试条件下，完成了 500 块按 120 ms 节奏释放的完整音视频推理。最终源码复测处理时间 P50 17.33 ms、P95 19.44 ms、P99 22.41 ms，最大 27.19 ms；包括嘴图预处理、视觉骨干、音频模型和输出复制，单会话测试不包括人脸检测、跟踪、模型加载、采集、网络或播放。测试素材为真实 LRS2 嘴图与音频中的 16 块循环；详细计时及输入和权重身份保存在 `/root/group-shared/voiceprint/experiments/falcon_cpu120_bench_20261111/`。单线程初测存在超过 40 ms 的块。

320 通道模型在双 A800 上以每卡 1024、无梯度累积完成了一次真实数据训练更新，并从该检查点完成续训；两卡显存采样峰值均为 26,326 MiB。记录位于 `/root/group-shared/voiceprint/experiments/falcon_120ms_320ch_1024_smoke_20261111/`。这是训练链路验证，模型质量以正式训练收敛后的测试结果为准。

`tools.streaming.evaluate_realssa` 在指定 cv 或 tt 索引上，对相同的 2 秒中心裁剪输入配对运行公开 Falcon 和流式模型，记录每条的 SI-SDR 改善以及输入、检查点、模型配置和代码摘要。公开 Falcon 使用完整 2 秒，流式模型使用 120 ms 块；比较结果需同时呈现这一时间上下文差异。不足 2 秒的片段由数据集补零到 2 秒，指标覆盖补零后的 2 秒；这不是论文的整句评测口径。评测脚本支持 `--start` 与 `--limit` 分段，实验产物统一保存在 `/root/group-shared/voiceprint/experiments/` 子目录。

[160 ms v3 设计与历史结果](FALCON_STREAMING_160MS.md)保留此前实验；v4 使用不同块长和位置编码，需使用 v4 检查点。

第 100 轮检查点的 RealSSA-Sim tt 全量中途配对结果、9,166 条有效样本统计和三个全零目标的核查，见 [测试记录](verification/streaming-120ms-realssa-tt-epoch100.md)。

模型质量差距的数据、时序、对比损失、边界及训练诊断见[质量差距逐项诊断](verification/streaming-120ms-quality-diagnosis.md)。

2026-11-12 的训练质量修复、派生数据分组、对照方案和 CPU 验证见[质量改造与 A/B/C 对照](FALCON_STREAMING_QUALITY_REWORK_20261112.md)。

后续 v6/v7 实验使用同一 120 ms 接口。v7 的本次训练配置允许当前块查询此前最多 16 个块的投影 K/V，缓存满后只保留最近历史。新训练入口仍为 `tools.streaming.train_ddp`，训练配方和结果见[历史注意力与教师蒸馏验证](verification/streaming-history-distillation-20261112.md)。
