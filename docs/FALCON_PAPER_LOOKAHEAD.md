# 120 ms 分块、80 ms 前视的 Falcon 风格模型

这条实验路径使用 120 ms 输出块，允许当前块结束后再读取 80 ms 的音频和视频，并持续携带历史状态。模型使用冻结的预训练 FRCNN、四尺度时频网络和复数掩码，在结构上向 [Falcon 论文](https://arxiv.org/html/2610.32749v1) 靠拢。它是新实现的论文风格模型，与论文和公开检查点不是同一个网络；质量需要通过训练后同协议评测确定。

## 时钟与输入输出

采样率为 16 kHz，视频为 25 fps，每帧对应 40 ms。每处理一块，输入 3200 个音频采样点和 5 张嘴图，输出前 1920 个采样点；下一次将窗口向前推进 1920 点。

```text
时间：       0       40       80       120      160      200 ms
视频：       A        B        C         D        E
             └──── 当前输出块 ────┘ └─── 允许前视 ───┘
音频：       |-------- 1920 点 --------|--- 1280 点 ---|
输出：       |-------- 1920 点 --------|
下一次：                                 A'       B' ...
                                         ↑ 从 120 ms 开始
```

每次收齐 200 ms 的输入后输出最早的 120 ms，随后每收到 120 ms 新数据再输出一块。首块起点到结果的等待为 200 ms 加模型运算；对块末附近的采样点，等待约 80 ms 加运算。图中时间是同步音视频的采集时间，不包含摄像头采集或外部传输耗时。

原 FRCNN 第一个 3D 卷积使用前后各两帧。对当前 3 帧，视觉编码器使用上一块末尾 2 帧、当前 3 帧和未来 2 帧，共 7 帧，得到与原始对称卷积对齐的当前 3 帧特征。未来两帧不提交为历史；下一次它们会作为新块前两帧再次输入。音频分离网络使用未来 80 ms 音频，视觉分离条件使用当前三帧特征；未来音频帧位置沿用当前末帧视觉特征，避免再次计算未来视觉特征造成额外前视。

从流首开始缺失的历史补零；`LookaheadSession.finish()` 对流尾缺失的音视频补零，只输出真实音频长度。会话要求每 40 ms 按顺序传入一个音频段和一个 88×88 嘴图，`sequence` 从 0 递增。输入必须已经完成音视频同步、嘴部检测和与训练一致的灰度/归一化处理。

## 网络结构与论文的关系

```text
过去 2 帧 + 当前 3 帧 + 后续 2 帧
               │
       原对称 3D 卷积 + ResNet，冻结
               │ 当前 3 帧视觉特征
               ├─────────────────────────────┐
音频：120 ms 当前 + 80 ms 前视                │
       │                                     │
Hann STFT：256 点、128 点 hop                │
       │ 实部、虚部、幅值                     │
四尺度频率编码 → 频率双向 LSTM               │
       │ → 历史时间卷积 → 时间 LSTM          │
       │ → 历史与当前/前视注意力 ← 视觉投影 ─┘
       │ → 跨尺度上下文注入与解码
       ↓
复数掩码 → 加窗重叠相加 → 当前 120 ms 目标音频
```

| 部分 | 论文 | 本实现 |
| --- | --- | --- |
| 视觉 | 唇读预训练 FRCNN，AVTSE 训练中冻结 | 从公开 `falcon.pt` 提取的 FRCNN 冻结；文件 SHA256 为 `980339a9351cef0cdc39ea6e1257c59681ceac813f6e091178de5d1afcfd93bd`，其与独立唇读预训练文件的身份未单独核实 |
| 音频输入 | Hann STFT 256、hop 128，实部、虚部、幅值 | 同样的窗长、帧移与三通道；状态缓存左侧音频以持续分帧 |
| 分离网络 | 多尺度时频编码、跨尺度交互和时频建模 | 新写的四尺度频率编码、跨尺度注入、频率双向 LSTM、时间 LSTM 与注意力，操作和参数组织并非原论文网络的逐层移植 |
| 输出 | 复数掩码 | 复数掩码及加窗重叠相加，保留跨块边界频谱 |
| 时序 | 论文报告固定视觉前视，原开源实现另有维度及归一化行为 | 每块最多使用其结束后 80 ms 原始音视频，已提交状态不包含对之后块的预测性递推状态 |
| 训练目标 | 重建损失加逐近场说话人对比损失 | 当前为语音重建损失，退化/静音目标使用归一化均方误差；暂无近场负例对比项 |

### 历史信息

- 时间卷积默认 kernel 5、dilation 16，帧移 8 ms：采样位置相差 128 ms，最远直接访问当前特征之前 512 ms；中间位置由其他时序结构处理。卷积状态只提交本块 15 帧，不提交未来 10 帧。
- 时间 LSTM 保存并传递每个频率位置的隐藏状态。状态理论上可以持续携带更早信息，实际有效记忆由训练和数据决定。
- 注意力默认保留最多 125 个已提交历史帧，按 8 ms 帧移约为 1 秒；当前块和其 80 ms 前视也参与本次注意力。`history_frames` 调整缓存长度。
- 音频窗尾、跨块频谱和 FRCNN 原始帧历史分别缓存。各个状态都属于单独的流，不能在不同说话人或不同录音间共享；新流须重置。

## 官方数据和训练配置

[配置文件](../configs/RealSSAPaperLookahead.yml) 使用官方 RealSSA-Sim 索引：训练 100,826 条，验证 9,133 条。官方测试集为 9,169 条，留给训练后的测试。主训练数据按官方 tr/cv 划分；官方不同划分存在部分源路径重叠，因此与额外按来源分组的泛化评估应分开报告。

数据入口 [realssa_lookahead.py](../tools/streaming/realssa_lookahead.py) 读取原始 TAR 的混音、目标和嘴图，使用第 0 通道；训练统一随机裁剪/水平翻转，验证中心裁剪。每个样本产生最多 2.12 秒的混音及 53 帧视频作为真实未来上下文，监督输出为前 2 秒目标语音。源素材不足时在真实结尾补零，损失只使用目标长度。

默认候选配置为两卡每卡 batch 1024、全局 2048，每个 batch 一次反向传播和一次优化器更新，**没有梯度累积**。AdamW 目标学习率 0.01，前 1000 个优化步线性预热，weight decay 0.01；最多 300 epoch，完整验证后通过 plateau 调整学习率并按验证结果提前停止。`checkpoint_chunks: true` 是反向传播时重算块激活以节省显存，不改变 batch 或把多个 batch 累积成一次更新。视觉骨干使用预训练权重并冻结；音频与分离网络初始化为新模型。

学习率 0.01 是候选值；训练日志记录每步学习率、损失和梯度范数，实际稳定性和效果以完整训练、验证结果为准。当前数据包没有各近场干扰说话人独立混响波形，`mix - target` 包含所有其他来源，不作为论文的逐人负例。

## 运行与检查

在服务器仓库根目录执行，使用 `/root/miniforge3/envs/falcon` 环境。实验目录必须使用新的 `run-name`；日志、配置身份和 checkpoint 位于配置指定的 `/root/group-shared/voiceprint/experiments/falcon_paper_lookahead_20261112/` 下。

先做有界双卡试训，下面的 batch 2 用于链路检查；完整训练仍以配置中的每卡 1024 为准：

```bash
PYROOT=/root/miniforge3/envs/falcon/bin
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none \
"$PYROOT/torchrun" --standalone --nproc_per_node=2 \
  -m tools.streaming.train_paper_lookahead \
  --config configs/RealSSAPaperLookahead.yml \
  --run-name smoke-example --batch-size 2 --train-limit 4 \
  --max-steps 1 --validation-limit 4
```

完整训练的入口如下；启动前需要先完成配置对应模型的双卡 batch 1024、时序、数值和 CPU 验证。`run-name` 替换成新的实验目录名称：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none \
/root/miniforge3/envs/falcon/bin/torchrun --standalone --nproc_per_node=2 \
  -m tools.streaming.train_paper_lookahead \
  --config configs/RealSSAPaperLookahead.yml --run-name full-example
```

完整 epoch checkpoint 可通过 `--resume` 恢复，需使用新的 `run-name`；训练器检查配置、代码、数据索引及 GPU 数等身份。中途结束的部分 epoch checkpoint 不能按完整 epoch 续训。

当前自动化检查可执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
/root/miniforge3/envs/falcon/bin/python -m pytest -q \
  tools/repro/tests/test_falcon_lookahead_visual.py \
  tools/repro/tests/test_falcon_lookahead_data.py \
  tools/repro/tests/test_paper_stream_prototype.py \
  tools/repro/tests/test_falcon_lookahead_session.py \
  tools/repro/tests/test_falcon_lookahead_stream_e2e.py \
  tools/repro/tests/test_falcon_paper_lookahead.py \
  tools/repro/tests/test_falcon_lookahead_training.py \
  tools/repro/tests/test_falcon_lookahead_train_model.py
```

视觉编码器的随机权重与公开视觉权重测试比较了整段和分块特征；完整模型会话测试检查两个块的前视上限、逐 tick 输出、短尾及恒等掩码重建。单测与小步试训分别验证时序及训练链路，最终提取质量需要对完整官方测试集计算指标。

## 双卡 batch 1024 首次更新记录

实验目录：`/root/group-shared/voiceprint/experiments/falcon_paper_lookahead_20261112/ddp-w256-bf16-b1024-01/`；过程日志为同级 `ddp-w256-bf16-b1024-01.log`。本次使用四尺度、width 256、时间 LSTM hidden 320、BF16、冻结公开视觉权重和块激活重算；两卡各 batch 1024，一次使用 2048 条官方训练样本直接更新，梯度累积次数为 1。训练器报告每卡数据由官方训练索引读取；诊断范围限定为 2048 个样本，验证仅取官方验证集 2 条。

- 第 1 次更新：rank 0 损失 `-1.239979`，裁剪前梯度范数 `94.537781`；目标学习率为 0.01，第 1 步处于 1000 步预热中，实际学习率为 `0.00001`。
- rank 0 报告的峰值已分配显存为 63.605 GiB、峰值预留显存为 69.869 GiB；训练加两条验证共记录 35.8 秒。
- 两条验证样本的平均损失为 `-0.996267`，保存了第 1 步 checkpoint。该试验只证明当前条件下能够完成一次前向、反向、更新和有界验证；它没有覆盖完整训练集或证明收敛。两条验证损失和训练损失不代表官方测试集质量，模型能否达到论文结果需要完成训练和同协议评估。

当前源码中分离核心的 SHA256 与此次日志一致（`3a842b804544071cfd0b3a7a5e18f0837e340c23cef863e0d559ae06b2fd6ed2`）；日志还保存了训练器、视觉权重和数据索引的 SHA256。修改模型或训练逻辑后，应重新核对身份并验证显存和数值。

## CPU 测量记录

`/root/group-shared/voiceprint/experiments/falcon_paper_lookahead_20261112/full_four_scale_w256_cpu4.json` 记录了四尺度早期版本的完整视觉加音频模型，宽度 256、LSTM hidden 320、历史注意力 125 帧，CPU 4 线程，预热 100 块、测量 500 块：P50 **38.43 ms**、P95 **41.28 ms**、P99 **43.02 ms**、最大 **47.08 ms**，500 块中没有超过 60 ms 的块。

该记录使用随机输入测量模型计算，不是训练质量评估；它对应测量时的版本，测量时已包含时间卷积，之后代码又增加了混合精度训练所需的类型转换；最终版本仍需重新测量。最终配置和权重需要在同一 CPU 环境重新预热、长流测量并检查 P95、P99、最大值和超过 60 ms 的块数。

可在独立新输出文件上重新测量：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
/root/miniforge3/envs/falcon/bin/python -m tools.streaming.benchmark_paper_lookahead \
  --visual-init /root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar \
  --width 256 --hidden 320 --history-frames 125 --threads 4 \
  --warmup 200 --blocks 500 \
  --output /root/group-shared/voiceprint/experiments/falcon_paper_lookahead_20261112/cpu-new-measurement.json
```
