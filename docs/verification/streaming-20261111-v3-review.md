# 160 ms 块级因果 v3：验证记录

## 模型和时间范围

v3 用当前块的 20 个音频特征和 4 个视觉特征做块内自注意力，块间历史来自音频因果卷积状态和视觉帧缓存。单次调用接收并输出同一时段的 2,560 个 16 kHz 样本，保护区为零。评估使用随机初始化检查点；样本级时序和运行速度不等于训练后的语音分离质量。

## 真实音视频因果审计

使用 LRS2 测试索引第一条的 3 个连续块、CPU 和随机检查点，分别扰动第 2、3 块开始的音频或视频；4 种情况下此前输出最大变化均为 0。扰动第 2 块后半段时，第 2 块前半输出的最大变化：音频扰动 0.00253437、视频扰动 0.00147227；第 1 块均无变化。输入、检查点、模型哈希和逐项结果见 [原始记录](streaming-20261111-v3-causality.json)。

独立复核发现：早期 session 对末块无效画面只检查占位时间，v3 块内注意力可能让结束之后的画面影响有效音频。修正后会话只使用覆盖有效音频的画面，其余位置重复最后一张有效画面。对 1、8、640、641、1280、1281、1920、1921、2559 个有效采样点分别反转无效位置嘴图，有效输出最大变化均为 0；末块后继续推送会被拒绝。实时会话和离线回放采用相同的末块画面补齐规则。

## 真实数据训练链路

在本服务器 `falcon` 环境，使用 `/root/data/falcon_lrs2_2mix/tr` 的真实索引、CPU、两个连续块、batch 1，从新初始化模型运行两步，loss 为 -5.32046、-3.97353；恢复检查点和优化器后运行一步，loss 为 -5.92094。另用随机权重回放 32,000 点真实音频，产生 13 块输出、补齐 1,280 点后裁回输入长度。短训练验证读取、反向传播、保存和恢复，不作为分离质量评估。

执行命令：

```bash
cd /root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
R=$(mktemp -d /root/tmp/falcon-chunk-v3-example-XXXXXX)
OMP_NUM_THREADS=4 "$PY" -m tools.streaming.init --output "$R/random.pt" --seed 17
OMP_NUM_THREADS=4 "$PY" -m tools.streaming.train --index-dir /root/data/falcon_lrs2_2mix/tr --chunks 2 --steps 2 --batch-size 1 --device cpu --output "$R/real.pt"
OMP_NUM_THREADS=4 "$PY" -m tools.streaming.train --index-dir /root/data/falcon_lrs2_2mix/tr --chunks 2 --steps 1 --batch-size 1 --device cpu --resume "$R/real.pt" --output "$R/resume.pt"
OMP_NUM_THREADS=4 "$PY" -m tools.streaming.audit_causality --checkpoint "$R/random.pt" --index-dir /root/data/falcon_lrs2_2mix/tt --threads 4 --output "$R/causality.json"
```

检查点和结果路径使用独占创建；复跑时使用新的临时目录。旧版 v1、v2 检查点与 v3 不是相同模型格式。

## CPU 性能

CPU 为 Intel Xeon Platinum 8336C，PyTorch 2.8.0，随机 v3 检查点。输入取同一条真实 LRS2 测试样本的 12 个完整块循环，重复处是人为拼接；一次处理包含嘴图裁剪归一化、音视频模型和输出拷贝，不包含采集、跟踪、网络或播放。完整输入、哈希、每块耗时及统计保存在对应 JSON。

连续 500 块按 160 ms 节奏释放，预热 10 块，4 线程：

| 测量 | P50 | P95 | P99 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| 处理耗时 | 21.651 ms | 25.768 ms | 31.290 ms | 32.865 ms |
| 计划释放到完成，含调度等待 | 21.727 ms | 25.842 ms | 31.364 ms | 32.939 ms |

500 次计划释放到完成时间均不超过 140 ms，也均不超过下一个 160 ms 周期。按块首样本等待 160 ms 加计划释放到完成的时间，本次 P99 为 191.364 ms，最大为 192.939 ms。结果对应此次单路 CPU 测量；更多并发和机器负载使用各自场景重新测量。[500 块原始结果](streaming-20261111-v3-cpu-final-paced-500.json)，[另一轮 500 块结果](streaming-20261111-v3-cpu-paced-500.json)。

各线程连续调用 100 块、未按真实节奏限速，预热 10 块：

| 线程 | P50 | P95 | P99 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 47.200 ms | 48.667 ms | 52.122 ms | 52.867 ms |
| 4 | 17.274 ms | 18.076 ms | 18.620 ms | 19.966 ms |
| 8 | 12.732 ms | 13.450 ms | 14.019 ms | 14.802 ms |

[1 线程记录](streaming-20261111-v3-cpu-unpaced-1t.json)、[4 线程记录](streaming-20261111-v3-cpu-unpaced-4t.json)、[8 线程记录](streaming-20261111-v3-cpu-unpaced-8t.json)。非限速与实时节奏结果分别测量，不能将前者视作包含调度等待的实时结果。

复测命令，输出选择新路径：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniforge3/envs/falcon/bin/python -m tools.streaming.bench_cpu_review \
  --checkpoint "$R/random.pt" --index-dir /root/data/falcon_lrs2_2mix/tt \
  --threads 4 --warmup 10 --chunks 500 --paced --output "$R/cpu-paced.json"
```

## 自动测试

流式浏览器核心使用 `node --test tools/streaming/web/stream-core.test.mjs`：6 项通过，覆盖帧选择、音频缓冲、消息时钟、压缩图片和末块。Python 回归中的多说话人指标测试在隔离进程以 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""` 单独运行，1 项通过；整套测试连续执行时，该历史指标用例长时间运行，可能与前序测试改变进程内线程状态有关，因此其余用例使用单独进程执行并记录结果。

其余 Python 项目回归通过 **217 项，跳过 2 项，另有 1 项按上述方式单独运行**；两项跳过原因是历史临时 LRS2 评测日志和历史训练 metrics.csv 不在原路径。与独立运行的 1 项合计，218 项通过、2 项跳过。结果和警告见 [Python 回归日志](streaming-20261111-v3-pytest.log)。命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q -rs \
  -k "not test_tracker_multi_source_does_not_crash_and_pesq_is_nan"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  /root/miniforge3/envs/falcon/bin/python -m pytest \
  tools/repro/tests/test_metrics.py::test_tracker_multi_source_does_not_crash_and_pesq_is_nan -q
```
