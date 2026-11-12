> 本页记录 160 ms 历史实现；后续无后续块前视的 120 ms v4 见 [v4 模型说明](../FALCON_STREAMING_120MS.md)，2026-11-12 的 120 ms 输出、80 ms 前视实现见 [论文风格模型](../FALCON_PAPER_LOOKAHEAD.md)。

# 160 ms 音视频流式模型独立审查

日期：2026-11-10。对象为本轮新增的 FalconStreaming 派生模型、训练入口及会话接口。此记录使用随机初始化权重验证实现和数据流，不评价目标说话人提取质量；新模型不是原 Falcon 发布权重兼容版本。

## 环境与执行

- 服务器 `dev_a800_2gpus`，GPU NVIDIA A800-SXM4-80GB，Python 项目环境 `/root/miniforge3/envs/falcon`，PyTorch 2.8.0+cu128、CUDA 12.8。
- 测试运行时 `torch.backends.cuda.matmul.allow_tf32=False`，`torch.backends.cudnn.allow_tf32=True`；不同批尺寸可能选择不同 GPU 实现，比较跨样本影响时固定批大小。
- 命令：`PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_falcon_streaming_contract.py -q`。
- 随机种子和断言见测试源码。模型状态测试使用 `channels=8, dilations=(1,2)` 的轻量随机实例；会话测试使用默认架构的随机检查点。GPU 实际运行：**6 passed**；参数化范围见下表。

## 结果

| 核查 | 测试和观察 |
|---|---|
| 分块与重置 | 三块完整 `forward` 与逐块 `forward_chunk` 连续调用结果完全相同；重置后首块相同，输出长度和数值有限。 |
| 未来音频 | 修改 128、896、1280、2432、2560、5120 采样位置及其后面的输入，不改变此前完整 128 采样 hop 的输出，覆盖块内及跨块边界。该结果按 8 ms hop 口径，不表示 hop 内逐采样因果。 |
| 未来嘴图 | 修改第 2、4、8 帧及之后的图像，不改变这些帧对应的 640 采样时间位置之前的输出；扰动确实影响后续输出。 |
| 时间对齐 | 将掩码固定近似 1 后，随机输入与输出按样本对齐，在 `rtol=1e-5, atol=2e-6` 内一致，覆盖块边界；这不是对非恒等掩码的语音质量测试。 |
| 训练梯度 | 真实模型路径上的随机音视频张量和目标 loss 能反向传播到视觉输入、视觉前端与投影、音频投影；归一化层训练时使用固定统计。 |
| 批次隔离 | 固定 B=2 和第二条音视频，仅更换第一条，第二条输出在 CPU、GPU 两次探针中差值为 0；训练模式自动化测试也通过。 |
| 会话 | 校验块序号、采样起点、未来视频帧、过旧视频帧；无效请求不会推进会话状态，重置后相同输入输出相同。 |
| 现有训练数据 | `/root/data/falcon_lrs2_2mix/tr` 的 20,000 对索引可加载；抽取 2 块，混音和目标各 5120 采样，嘴图为 `[8,88,88]`。这项是入口与尺寸检查，不是全量数据质量检查。 |
| 检查点 | 实际读取到随机 v2 检查点 step=0、`random_initialization`；两步真实数据 smoke 后保存 step=2；恢复后保存 step=3、`real_data_optimization`，含 131 个优化器状态。检查点并未保存随机数及数据加载迭代器状态，因此当前是权重与优化器续训，不是逐样本精确重放。 |

批尺寸比较探针中，同一条输入单独 B=1 与置于 B=2 的最大差为 CPU `3.58e-7`、CUDA `7.69e-4`；固定 B=2、只替换另一条后 CPU 和 CUDA 差均为 0。前者不能单独作为跨样本信息泄漏的证据。

## 时间与适用范围

160 ms 表示公开接口每次提交 2560 个音频采样与 4 帧嘴图；频域处理按 128 采样 / 8 ms hop 收集、计算并输出。去除 80 ms 窗尾保护区不等于输出逐采样零等待。嘴图必须在对应 40 ms 区间起点之前采集；采集、跟踪和音视频时钟映射由外部负责。完整麦克风到播放延迟、分离质量与长时间运行表现需要对应的实测，以上自动化测试不代替这些评测。

## 本机双路 WebSocket 实测

在服务器启动 loopback 服务，使用随机检查点 `/root/tmp/falcon-streaming-check-20261110/random-v2.pt`、A800 GPU 和两个真实 TCP WebSocket 客户端。该检查点已迁至 `/root/group-shared/voiceprint/experiments/falcon_historical_runs_20261111/falcon-streaming-check-20261110/random-v2.pt`，原始记录的路径保留；迁移后文件 SHA-256 与下述记录一致。脚本位于 [`streaming_ws_concurrency.py`](scripts/streaming_ws_concurrency.py)，复现命令：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniforge3/envs/falcon/bin/python \
  docs/verification/scripts/streaming_ws_concurrency.py \
  --checkpoint /root/group-shared/voiceprint/experiments/falcon_historical_runs_20261111/falcon-streaming-check-20261110/random-v2.pt --device cuda
```

两路同输入的输出一致；单路 reset 后另一连接继续使用自己的序列，8 轮连续验证通过。启动后未经额外预热的第 0 轮两路完成耗时 338.11 ms，其中第二路服务端等待队列 325.93 ms；后 7 轮两路完成耗时 14.20～15.53 ms。该记录仅为少量本机请求，数字包括客户端发请求和收响应，不包含麦克风、摄像头跟踪和播放；不能作为端到端延迟百分位。后续服务如加入预热，其冷启动数值需要重新测量。

WebSocket 的 FastAPI TestClient 回归命令：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_falcon_streaming_server.py -q
```

CPU 随机模型测试 **3 passed**，覆盖两连接状态隔离、重置与重新连接、无效消息后状态不推进及跨源拒绝；真实 TCP 双连接见上节。

服务加入 3 块预热后，使用相同命令重新运行：服务上报预热 350.41 ms；第 0 轮两路完成 34.97 ms，后 7 轮为 13.92～17.71 ms，两路输出和重置隔离检查通过。这仍是一次短时本机测试。

关闭 CUDA 后以 `CUDA_VISIBLE_DEVICES= PYTHONPATH=. OMP_NUM_THREADS=2` 运行上述契约和服务测试，结果 **9 passed**。服务器检查 `/dev/video*` 与 `/dev/snd/*` 均为空，`DISPLAY` 未设置；本次无法在该服务器完成真实摄像头与麦克风采集到播放的闭环验证。现有 Studio 采集、嘴部跟踪与播放代码仍使用旧滑窗协议，尚未接入此增量服务。

## 身份与补充回归

- 被检模型源码 [`FalconStreaming.py`](../../look2hear/models/FalconStreaming.py) 的 SHA-256：`f3fb15d143582d60d24bc8d4cee3c53ea60250c706b18e7660d40c229f200757`。
- 随机初始化检查点 `/root/tmp/falcon-streaming-check-20261110/random-v2.pt` 的 SHA-256：`b9b606d7f90d0cd40c7ac258748111bd0c41653364a0c0be636dd2af041481c8`；它是本机临时测试产物，重新按种子生成的文件内容和散列需要以新文件核验为准。
- 预热服务后按 160 ms 间隔发送 35 块的逐块原始结果见 [`streaming-20261110-ws-paced.json`](streaming-20261110-ws-paced.json)：随机合成输入，本机 WebSocket，客户端编码到响应的 P50 11.20 ms、P95 14.23 ms、最大 30.19 ms，无块超过 160 ms。该计时不包括设备采集、人脸跟踪、跨机网络及声音播放。
- `CUDA_VISIBLE_DEVICES= PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_falcon_streaming_data.py -q`：**3 passed**，覆盖数据路径对齐、错配和输入检查；与实际 LRS2 的单样本读取检查互补。
