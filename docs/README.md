# Falcon-fork 文档导航

这套文档从实际代码、已保存的评测记录和 2026-11-10 的多轮检查出发，帮助读者理解模型、准备数据、运行评测，并判断结果能支持什么。项目还保留更早的研究、复现计划和审查记录，展示各项结论的形成过程。

## 按问题查找

| 读者的问题 | 文件 | 能看到什么 |
| --- | --- | --- |
| 这个项目做什么，怎样开始 | [中文 README](../README_zh.md) / [English README](../README.md) | 输入输出、目录、环境、快速评测和主要限制 |
| 现在复现到了哪里 | [复现状态](STATUS.md) | 权重与指标条件、已有结果、测试状态和未完成项 |
| 120 ms 输出、80 ms 前视的新模型怎样训练和推理 | [论文风格流式模型](FALCON_PAPER_LOOKAHEAD.md) | 官方数据、尺度敏感波形损失、冻结 FRCNN、训练与评测口径 |
| 旧训练为何数值异常，修复了什么 | [训练异常审查](verification/20261112-lookahead-training-instability.md) | 输出增益、静音损失、多尺度放大和新训练协议 |
| 流式训练有哪些经验教训 | [Falcon 流式训练经验与复核](verification/20261112-falcon-experience-lessons.md) | 时序、视觉初始化、数据与损失、数值稳定性和评测证据 |
| 模型怎样处理音频与嘴部 | [模型、融合与时间依赖](understanding/20261110-model-and-streaming.md) | 张量形状、两次音频块、视觉融合、因果性和计算量 |
| 五套开源数据分别是什么、本机有哪一层 | [数据集说明](understanding/20261110-datasets.md) | LRS2、RealSSA-Sim、RealSSA-Real 的样本、目录、索引和空仓库 |
| 项目与多模态数据有哪些、冷端怎样取回 | [数据集全景](data_catalog/README.md) | 训练/测试数据、逐库卡片、归档回执与恢复方法 |
| 数据、采样和训练怎样工作 | [数据准备与训练](understanding/20261110-data-and-training.md) | 索引、目标方向、near、损失、配置和检查点 |
| 怎样评测和使用演示 | [评测、Studio 与 Space](understanding/20261110-evaluation-and-apps.md) | 整段/分窗/批处理、指标、输入格式、输出和风险 |
| 怎样下载、转换和核对访问条件 | [数据访问说明](DATA_ACCESS_GUIDE.md) | 历史获取记录、转换示例和许可；当前目录以数据集说明为准 |
| 怎样准备论文版 8 卡训练 | [训练准备说明](PHASE2_VOLCANO_LRS2_TRAINING.md) | 配方、路径、提交前检查与未验证项；运行会占用资源 |
| 数字和新发现从哪里来 | [2026-11-10 审查证据](verification/deepdive-20261110/README.md) | 本轮脚本、输入/权重哈希、结果和测试日志 |
| 某个旧文档在哪里 | [全部文档清单](DOCUMENTS_INDEX.md) | 117 份原有 Markdown 和本轮入口，按用途与时点分组 |

## 怎样判断一条结论

先确认文档记录的是**哪次代码版本、权重、数据与运行方式**。例如，LRS2 的单目标方向 3,000 条、论文版双目标方向索引 6,000 条，是不同的分母；RealSSA-Sim 的不同指标也有不同有效条数。逐位比较应记录输入和设备。

[复现状态](STATUS.md)汇总当前应如何理解已有数字；[本轮审查记录](verification/deepdive-20261110/审查记录.md)说明发现的旧文错误、实测反例和仍未验证的事项。2026-11-08 与 2026-10-24 的深挖目录、`docs/reviews/`、Phase 1/2 报告均保留原来的历史内容。引用旧数字时，可从原始 CSV、日志和后来的更正了解当时条件。

## 原始证据在哪里

- [`verification/`](verification/README.md)：按阶段整理的日志、CSV、模型参数和验证脚本。部分脚本使用替身或写固定临时路径，运行前先看脚本和对应说明。
- [`verification/deepdive-20261110/`](verification/deepdive-20261110/README.md)：本轮批次串扰、帧选择、ID 集合、指标汇总、测试及逐文件审阅材料。真实输入与合成输入分别标注。
- [`understanding/deepdive-20261108/`](understanding/deepdive-20261108/00-SYNTHESIS.md)：上一轮模型、数据、评测与论文分析；[逐篇复核](verification/deepdive-20261110/深挖文档复核状态.md)列出其中已发现的口径问题。
- [`understanding/deepdive-20261024/`](understanding/deepdive-20261024/00-README-%E9%97%A8%E7%89%8C.md) 和 [`reviews/`](reviews/round1/)：更早的理解、问题与审查过程。

已有历史日志和 CSV 保留原样。修正结论时新增带时间的检查和说明，不能用新结果覆盖旧数据。历史命令可能下载文件、重建索引、覆盖日志或清理临时目录；使用前先核对脚本、输出位置和文档的适用版本。
