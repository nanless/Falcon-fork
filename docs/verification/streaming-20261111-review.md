# 2026-11-11 流式模型 CPU 复核

## 范围与修复

多轮审查覆盖模型、训练、检查点、分块会话、文件推理和浏览器采集收尾。频谱掩码先转成最长 129 tap 因果 FIR，再执行 256 点 overlap-save；逐子帧输出和梯度与时域线性卷积参考对照。旧 v1 使用不同重建语义，v2 不加载 v1 检查点。模型状态必须完整，续训显式学习率覆盖优化器中的旧值。停止采集时冲刷采集 FIR、处理和裁剪尾块，部分末块之后须重置。训练记录见 [训练复核](streaming-20261111-training-review.md)。

## CPU 连续处理

环境：Intel Xeon Platinum 8336C、PyTorch 2.8.0+cu128、CPU float32、随机 v2 检查点，单会话。每组预热 10 块，连续处理 200 块；输入由一条 LRS2 测试音视频的 12 个完整块循环构成，拼接处是人为循环。计时包含嘴图预处理、音视频模型和输出回传，不包含采集、嘴部追踪、网络与播放。

| 线程 | P50 | P95 | P99 | 最大 | 超过 140 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.071 ms | 54.344 ms | 57.006 ms | 61.028 ms | 0 |
| 4 | 17.675 ms | 19.997 ms | 20.885 ms | 21.545 ms | 0 |
| 8 | 15.145 ms | 16.106 ms | 16.667 ms | 16.779 ms | 0 |

## 160 ms 节奏测试

4 线程预热 10 块后，按 160 ms 周期处理 500 块。调度起点到完成的 P50/P95/P99/最大为 21.789/24.798/30.166/31.858 ms；超过 140 ms 为 0 块，超过 160 ms 为 0 块。按约定加 160 ms 凑块时间，P99 为 190.166 ms、最大为 191.858 ms。

## 因果检查

在同一真实音视频的连续两块上，对 14 个音频位置及之后的波形做扰动；其中 12 个位置之前存在完整 8 ms hop，其输出最大变化均为 0。其余两个位置之前没有完整 hop，因此没有前序 hop 差值。对 6 个视频帧边界及之后的画面做扰动，其对应时间之前最大变化为 0。每个测试同时确认后续输出确有变化。8 ms 内的先前样本可受同一 hop 后续样本影响，因此这是子帧级因果。

## 证据和复现

- [因果逐项数据](streaming-20261111-cpu-causality-verified.json)
- [1 线程数据](streaming-20261111-cpu-verified-1t-unpaced.json)、[4 线程数据](streaming-20261111-cpu-verified-4t-unpaced.json)、[8 线程数据](streaming-20261111-cpu-verified-8t-unpaced.json)
- [500 块定时逐项数据](streaming-20261111-cpu-verified-4t-paced-500.json)
- [测试脚本](../../tools/streaming/bench_cpu_review.py)和[因果脚本](../../tools/streaming/audit_causality.py)

脚本在测量前后核对列出的模型主文件和检查点 SHA-256；视觉骨干等其他依赖未包含在该校验中。一次测试只覆盖一条音视频和该服务器负载，随机权重不衡量分离效果。

## 自动回归

- CPU 流式模型、数据、训练、推理及服务：`pytest tools/repro/tests/test_falcon_streaming*.py -q`，39 passed。
- 浏览器时钟、音频采集与收尾：`node --test tools/streaming/web/*.test.mjs`，11 passed。
- 原 Studio：`pytest studio/tests -q -m "not slow"`，53 passed。
- 两步真实 LRS2 短训与一步续训的损失和梯度有限；续训覆盖学习率及检查点版本测试见 [训练复核](streaming-20261111-training-review.md)。

以下命令使用本轮随机检查点；`--output` 使用新的文件名。前述逐项 JSON 保留了运行参数、输入与检查点身份和每块耗时。

```bash
cd /root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
CKPT=/root/group-shared/voiceprint/experiments/falcon_historical_runs_20261111/falcon-streaming-review-20261111/random-v2-r2.pt
DATA=/root/data/falcon_lrs2_2mix/tt
OUT=$(mktemp -d /root/group-shared/voiceprint/experiments/falcon-cpu-review-repeat-XXXXXX)
for n in 1 4 8; do
  OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n "$PY" -m tools.streaming.bench_cpu_review \
    --checkpoint "$CKPT" --index-dir "$DATA" --threads "$n" --warmup 10 \
    --chunks 200 --output "$OUT/cpu-${n}t.json"
done
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -m tools.streaming.bench_cpu_review \
  --checkpoint "$CKPT" --index-dir "$DATA" --threads 4 --warmup 10 \
  --chunks 500 --paced --output "$OUT/cpu-4t-paced.json"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -m tools.streaming.audit_causality \
  --checkpoint "$CKPT" --index-dir "$DATA" --threads 4 --output "$OUT/causality.json"
```


## 2026-11-11 v3 版本说明

本文件记录 v2 运行和历史结果。当前工作树的流式训练、检查点加载与因果审计已升级为 v3；复现这里的 v2 命令需使用对应历史代码版本和 v2 检查点。v3 结果见 [流式模型指南](../FALCON_STREAMING_160MS.md)。
