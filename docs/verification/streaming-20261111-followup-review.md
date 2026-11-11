# 流式模型后续复核：数据裁剪、末块重建和服务恢复

## 检查范围

本轮基线为 `3681fc0`。检查 v3 模型、训练数据、会话、服务和客户端；独立复核了块内注意力、跨块状态、音频重建和 CPU 结果。模型本身与此前性能测试时相同；此前性能的输入、设备、随机检查点与结果见 [v3 验证记录](streaming-20261111-v3-review.md)。

## 数据裁剪

训练的混音与目标长度由索引给出，视频从嘴图文件读取。以前裁剪起点只受音频长度约束，当嘴图比音频短、但仍足够一个训练片段时，可能随机选到视频末尾之外。现按 640 个音频采样点（40 ms）为网格，取混音、目标和嘴图都能覆盖完整训练片段的共同起点；若嘴图不足一个片段，在访问该条样本时提示文件路径。每条样本读取嘴图的次数仍为一次。

独立复核了 8、9、10、12 张可用嘴图与 8 帧训练片段的末尾合法起点，以及不足 8 张时的错误处理；验证音频、目标和嘴图保持时间对齐。复核者分别执行了数据和服务测试，并在当前工作区获得 18 项通过。

对 `/root/data/falcon_lrs2_2mix/tr/target.json` 使用 `random.Random(41)` 从 20,000 条中抽 1,000 条，逐条加载嘴图，按 `len(video) < int(indexed_samples) // 640` 检查，短于音频完整 40 ms 帧数为 0 条，读取失败为 0 条。另用 `random.Random(72)` 抽 500 条目标音频并各选一段 1.6 秒，目标全零 0 条、RMS 小于 0.001 的 0 条，最小 RMS 为 0.065653；这是该次样本的结果。

当前训练使用整段去均值 SI-SNR 损失。对 2,560 点零目标，以 `torch.manual_seed(33)` 测试随机非零输出，损失为 94.067276、梯度绝对值最大为 0.1109904；全零输出和精确常量 0.125 输出经去均值后的损失与梯度均为零。本轮保留现有训练目标，后续评估其他语料时可结合静音片段的占比与目标指标决定训练配方。

## 分块合成与服务

独立使用随机的非恒等掩码和 129 抽头直接线性卷积核对频域重建，最大绝对误差为 `4.17e-7`。原有回归测试另用高精度数据验证了随机时变掩码和梯度。

有限的极大浮点输入可能使计算结果成为非有限数。会话在提交本块状态前检查输出，现在以 `NonFiniteOutputError` 表示这一情况；服务仅对此错误记录序号及采样位置，并返回固定错误及期望的下一序号。双会话对照先积累相同历史，一路发送溢出块后以相同序号重试；其输出与未经历失败的会话逐字节相同。

## 本轮测试

联合运行流式数据、服务、模型时序、推理、状态和训练恢复测试，**58 项通过**；日志见 [联合回归](streaming-20261111-followup-pytest.log)。独立复核后新增目标音频对齐断言，数据测试再次运行 8 项通过。浏览器分块、采集和末块三套测试共 **11 项通过**。运行联合回归：

```bash
cd /root/code/github_repos/Falcon-fork
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  /root/miniforge3/envs/falcon/bin/python -m pytest -q \
  tools/repro/tests/test_falcon_streaming_data.py \
  tools/repro/tests/test_falcon_streaming_server.py \
  tools/repro/tests/test_falcon_streaming_contract.py \
  tools/repro/tests/test_falcon_streaming_inference_review.py \
  tools/repro/tests/test_falcon_streaming_state_review.py \
  tools/repro/tests/test_falcon_streaming_training_resume.py
```
