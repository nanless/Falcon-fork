# 2026-11-11 流式模型训练复核

复核对象为本轮修正频谱合成后的 `falcon-streaming-v2`。机器为 `dev_a800_2gpus`，解释器 `/root/miniforge3/envs/falcon/bin/python`，训练检查使用 CPU、单线程，真实数据来自 `/root/data/falcon_lrs2_2mix/tr`。各任务写入新的 `/root/tmp` 目录。

## 实现调整

- 修复续训时 `--lr` 被优化器检查点覆盖：省略 `--lr` 保留检查点值；显式传入时，将所有参数组更新为指定值。全新训练的默认值为 `0.0003`，启动时打印实际参数组学习率。
- 训练启动前检查目标检查点路径，已存在的路径直接报错。保存仍使用独占创建。
- `falcon-streaming-v2` 与前一版合成语义不同。读入检查点时同时验证格式和模型状态中的标量 `int64` 版本值 2；前一版检查点需重新初始化或训练。
- 在零目标输入探针中，原 SI-SNR 损失对非零输出的 loss 为 70、最大梯度约 17.37；原损失保持不变。

## 自动回归

```bash
cd /root/code/github_repos/Falcon-fork
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniforge3/envs/falcon/bin/python -m pytest \
  tools/repro/tests/test_falcon_streaming_training_resume.py -q
```

结果：**6 passed**。其中完整 CLI 运行覆盖新训练、沿用学习率续训、显式覆盖学习率续训及已存在输出路径；其余参数化用例覆盖旧格式、缺失版本、错误版本、浮点版本和非标量版本。

## 真实数据两步训练和续训

使用 2 块连续输入、batch 1、CPU，在 `/root/tmp/falcon-streaming-v2-train-20261111-2QxOdL` 新建随机检查点，运行两步训练并再续训一步。训练与恢复均调用 `python -m tools.streaming.train`，数据参数为 `--index-dir /root/data/falcon_lrs2_2mix/tr --chunks 2 --batch-size 1 --device cpu`；第一次加 `--steps 2 --resume random-v2.pt`，第二次加 `--steps 1 --resume real-v2.pt --lr 0.0001`。

| 记录 | 值 |
| --- | --- |
| 两步训练 loss | -2.90967、0.35787 |
| 两步梯度范数 | 75.23689、49.51631（裁剪前） |
| 续训一步 loss / 梯度范数 | -5.70959 / 26.67561（裁剪前） |
| 输出检查点步数 | 0、2、3 |
| 训练 / 续训检查点学习率 | 0.0003 / 0.0001 |
| 检查点格式 | 三者均为 `falcon-streaming-v2` |

短训用于验证真实数据读取、反向传播和保存恢复；模型提取质量通过后续完整训练与评测确定。续训恢复模型、优化器和累计步数，随机数和数据迭代位置不在检查点内。


## 2026-11-11 v3 版本说明

本文件记录 v2 运行和历史结果。当前工作树的流式训练、检查点加载与因果审计已升级为 v3；复现这里的 v2 命令需使用对应历史代码版本和 v2 检查点。v3 结果见 [流式模型指南](../FALCON_STREAMING_160MS.md)。
