# Falcon 音视频增量流式 Implementation Plan

> **Execution:** Use the installed executing-plans skill when implementing this plan. Reuse existing authorization and honor the requested review checkpoints.

**Goal:** 实现无额外保护区的 160 ms 音视频因果分块推理及对应训练入口，用随机权重完成机制、数据、性能验证。

**Architecture:** 新增 Falcon 派生流式网络，视觉前端和音频处理分别缓存历史状态，按固定时间戳融合，训练和推理共用分块路径；服务会话显式管理状态、顺序和重置。

**Tech Stack:** Python、PyTorch、现有 Falcon 数据索引、CUDA A800、pytest。

---

### 任务 1：状态模型

- 新建 `look2hear/models/FalconStreaming.py`，实现四帧视觉输入、2560 采样音频输入与 2560 采样输出，跨块缓存音频、视觉和网络状态。
- 测试时序边界、未来扰动、状态复位、批次隔离和视觉输入影响，区分 8 ms 子帧成帧等待与零额外保护区。

### 任务 2：数据、训练和推理

- 新建 `configs/FalconStreaming.yml` 与 `tools/streaming/common.py`、`init.py`、`train.py`、`infer.py`，保留实际路径与参数的自解释帮助。
- 使用项目已有 mix.json、target.json 和嘴部序列，按 640 采样对齐嘴部帧，检查随机权重 checkpoint、梯度、恢复和真实数据训练步骤。
- 提供会话级状态、块序号及重置接口，核对完整输出样本数和不引入额外保护区。

### 任务 3：独立验证与文档

- 新增针对性的自动测试和独立审查记录；运行 CPU/GPU 正确性、真实数据样本训练及 A800 性能测试。
- 新增中文文档，写清音视频输入、时间戳约束、随机权重状态、训练和推理命令、延迟测量口径及未测条件；保留已有模型行为。
- 最后复核工作区变更，确保所有结论可由测试结果支持，不擅自提交。
