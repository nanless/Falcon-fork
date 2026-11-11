# 流式质量修复 Implementation Plan

> **Execution:** Use the installed `executing-plans` skill when implementing this plan. Reuse existing authorization and honor the user's requested review checkpoints.

**Goal:** 修正已验证的训练缺陷，保留 120 ms 因果接口，并以匹配预算的真实数据对照检验候选效果。

**Architecture:** 原实验运行中的源文件保持不变；在独立 checkout 中保留 v4 原行为，增加显式标记的块内居中 v5 及严格 warm-start。训练目标区分语音与退化目标，保持原对比正项基线并单独评估可选方差正则；按目标视频目录分组生成实验数据（未隔离干扰与噪声源），校验来源和预训练接触范围。

**Tech Stack:** Python、PyTorch DDP、NCCL/Gloo、PyYAML、pytest、RealSSA TAR 索引。

---

### Task 1：模型、推理与协议

**Files:** `look2hear/models/FalconStreaming.py`、`tools/streaming/common.py`、`tools/streaming/session.py`、`tools/streaming/server.py`、`tools/streaming/train.py`、`tools/repro/tests/test_falcon_streaming_visual_context.py`。

- 检验旧 v4 输出逐位一致，分开音频与视频未来块扰动；检查 v4/v5 格式、状态、实际服务和单卡保存/恢复。
- 对候选进行含视觉的真实 CPU 节奏测量，记录全部时延及 40 ms 超预算次数。

### Task 2：损失与双卡训练

**Files:** `tools/streaming/train_ddp.py`、`tools/streaming/training_config.py`、`tools/repro/tests/test_falcon_quality_losses.py`。

- 比较原无负项正项与批量实现的损失/梯度；非零常量、静音、低能非恒定样本分别核验。
- 用两 rank 测试有效数不等、单 rank 全退化及全局退化的全局正项归一和日志。
- 检验方差正则仅对辅助头生效；分别检查数值最佳与早停状态、严格恢复与显式 warm-start；包含无验证集。

### Task 3：来源分组和配对工具

**Files:** `tools/streaming/split_realssa_sources.py`、`tools/streaming/prepare_quality_abcs.py`、`tools/streaming/compare_quality_abcs.py` 及其对应 `tools/repro/tests/test_*quality*.py`、`test_realssa_source_split.py`。

- 验证索引 ID/来源分组互斥、固定种子复现、hash完整性和非法输入；实际读取 tr/cv 样本。
- 输出同数据、同权重、同预算的 A/B/C 配置；检查开发集与保留集不混用，记录旧全量检查点已见来源的事实。
- 配对比较按来源目录分组 bootstrap，核对检查点、索引、配置和代码身份并列出无效样本。

### Task 4：真实双卡和质量对照

- 先独立运行两卡真实 1024/卡 smoke、v5 正则分支、保存和严格续训；检查 GPU 内存及全局有效计数。
- 三组以固定起点、样本、随机种子、学习率和 50 个更新比较，分别记录日志、检查点及 512 条开发样本逐条结果；训练其他组前验证冻结的代码和输入摘要。
- 将对照结果用于诊断，不将 warm-start 已见过来源的实验称为未见来源泛化或训练已收敛。

### Task 5：整合和交付

- 执行受影响范围完整回归、交叉 review、文档和路径检查；对已发现的问题修复后重验。
- 保存可复现的配置、证据和结论；由服务器完成中文提交和推送，核对远端。原训练使用的源文件不在其运行中替换。
