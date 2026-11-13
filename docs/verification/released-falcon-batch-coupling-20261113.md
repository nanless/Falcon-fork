# 发布 Falcon 的批次耦合与修复

在 A800 GPU1 上用发布 small 权重 FP32 诊断。冻结 LRS2 CV12 的 12 个混音、24 个目标方向沿已存配对清单读取，每条混音的 s1/s2、两套嘴图和原始混音分别通过 SHA-256 核验。RealSSA CV24 沿统一评测原始 ch0、center-valid 读取，逐条核验成员哈希、时序与评分长度。评测均为整个裁剪窗的发布模型前向。

## LRS2 目标选择与串扰

| 条件 | 双向均选对 | 正确方向 | 平均目标 SI-SDR | 平均目标 SI-SDRi |
|---|---:|---:|---:|---:|
| 发布 B1 正确嘴图 | 7/12 | 19/24 | 4.740 dB | +4.812 dB |
| 发布 B1 互换嘴图，参考身份固定 | 0/12 | 5/24 | −14.228 dB | −14.157 dB |
| 原 B2 正序 | 7/12 | 19/24 | 4.709 dB | +4.780 dB |
| 原 B2 反序并还原逐条身份 | 7/12 | 19/24 | 4.735 dB | +4.806 dB |
| 独立单步语义，B2 正序 | 7/12 | 19/24 | 4.740 dB | +4.812 dB |
| 同权重改为真正时间递推，B1 | 7/12 | 19/24 | 4.736 dB | +4.807 dB |

原 B2 正反序还原身份后，24 个目标波形差 RMS 中位数为 0.000372，最大 0.005672。固定目标在 B2 第二位置，仅切换前一位置嘴图，波形差 RMS 中位数为 0.000440，最大 0.01018。B4 加入另一个混音的两条目标并重新编排，单目标 SI-SDR 相对 B1 最低 −0.518 dB，最高 +0.880 dB。B2 重复同目标对照最高 +1.020 dB。此次所有原顺序条件的双向正确数恰好相同。

第一条混音的探针固定第二位置的音频与嘴图，只换第一位置嘴图。LSTM 的第二位置输入逐位相同，输出差 RMS 已为 0.005314，最终波形差 RMS 为 0.003251。发布路径将 [B,T,C] 送入按时间轴优先读张量的 LSTM，原实现把 B 当作序列长度。FRCNN 特征分别以 B1 和 B2 编码时，差 RMS 最大 1.97e−7。把每个样本、每帧独立过一个零状态 LSTM 步，B2 换序的最大差 RMS 降至 1.52e−8，相对原 B1 的所有 B2/B4 条件最大差约 3.42e−7。显式沿时间递推也可保持样本独立，其同权重输出及评分已有变化。

## RealSSA CV24 及代码修复验证

冻结 24 条 CV 保留统一评测的真实评分范围。三条短音频使用各自实际长度。四条成组，仅改变组内顺序。

| 默认路径 | B4 正反序波形差 RMS 中位数 | 最大 | B1 / B4 正序 / B4 反序 SI-SDRi |
|---|---:|---:|---|
| 修复前 | 1.305e−4 | 6.222e−4 | 7.147721 / 7.144224 / 7.145420 dB |
| 修复后，真实代码重跑 | 约 5e−9 | 7.076e−9 | 7.147721 / 7.147721 / 7.147721 dB |

修复前 RealSSA B4 对 B1 的逐条差值最低 −0.0785 dB，最高 +0.0550 dB。修复后 B4 对当前 B1 最大差 3.82e−6 dB，修复前后 B1 最大差 1.91e−6 dB。处理使用原发布权重 strict 加载。FP32 跨形状与运行结果允许上述微小差异。

默认 VideoBlock 分支把 [B,T,C] 变为 [1,B×T,C] 输入原 LSTM，再恢复 [B,T,C]。参数名、张量形状、加载结构和显式 `video_lstm_batch_first=True` 的时间递推均保留。CPU 定向测试覆盖逐条与批量、换序、只替换前一位置、B1 旧计算公式、显式时间递推与严格加载，共同运行模型重建测试为 20 passed。模型 `eval()` 下仍允许 GPU/批次形状导致的舍入误差；训练模式的 BatchNorm 仍按批次统计。发布模型的整窗依赖与流式延迟应继续单独评测。

## Studio 与蒸馏

Studio Small 的 `Falcon.py` 与 Big 的 `SISNRRepel.py` 各自复制了 VideoBlock。两个副本同步实施同样的独立单步修复，Space 构建沿用 Studio 副本。冻结 LRS2 前 4 个混音、8 个方向通过真实权重 strict 加载与 CPU 前向。Small 和 Big 在修复前后的全部 B1 波形 SHA 均逐条一致。

| Studio 模型 | B2 反序最大波形差 RMS，修复前 → 后 | 仅改变前一条嘴图，目标固定在第二位，最大波形差 RMS |
|---|---:|---:|
| Small | 0.0018998 → 5.18e−8 | 0.0032515 → 0 |
| Big | 0.0012173 → 0 | 0.0020995 → 0 |

蒸馏教师直接使用原模型已修复的 VideoBlock，移除了同样逻辑的外层包装。构造时检查存在 VideoBlock，且视觉 LSTM 使用默认独立单步配置。教师仍冻结参数并处于评估模式。蒸馏加三个 VideoBlock 副本的定向测试为 11 passed。

Space 的片段处理和静态资源测试通过。合并回归有 102 项通过，另有 4 项 Space HTTP 用例在启动服务时因 `falcon` 环境缺少 `spaces` 停止。该运行环境也未安装 `gradio`。两种真实权重的 CPU 核验通过了 Studio 模型入口和严格加载；ZeroGPU 服务的运行验证需使用对应的运行环境。

## 身份与原始证据

- 发布权重 `/root/.cache/falcon_weights/falcon.pt`，epoch 714，SHA-256 `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`。
- LRS2 冻结清单 `paired_selection/run64/manifest.json`，SHA-256 `b1af511c2623652bb2367b35568ae511f5d03a47e51a7991c70c608d1d2579b0`。
- RealSSA 冻结清单 `evaluation_candidate/cv24_seed1371049_center_valid.json`，SHA-256 `72a70b4fc553e5d41fba28d65139e4410a20caa54d02fd85166bba551ad907dd`。
- 修复前模型运行仓 `falcon_streaming_v9_capacity_20261112/repo`，HEAD `3767b2c74ff58467a3e21fc901c0ff57ee8945e0`，`look2hear/models/Falcon.py` SHA-256 `6956216db79cd80766126b16ae7247264e3e2de9e531c64d1a470c1b5dadfa0d`。
- 实施仓 `current_fixes/repo`，修改前 HEAD `98a63580cbf38a0be14e3e6064bc0752f0674fe0`，修复后 `Falcon.py` SHA-256 `3d21c296d1cd61c7933fcbee83ff3a1d61a5be62df885f7698ebd68629dbe375`。
- 实验脚本、JSON 和日志位于 `/root/group-shared/voiceprint/experiments/falcon_diagnosis_execution_20261113/released_batch`。主文件为 `lrs2_cv12_released_batch.json`、`realssa_cv24_released_batch.json` 和 `realssa_cv24_patched_batch.json`；对应实验脚本 `released_batch.py` 与 `released_realssa_cv24.py`。结果描述冻结样本的串扰范围，历史全量或正式训练指标需要各自的复测。
- Studio 真实权重记录为 `studio_before.json` 和 `studio_after.json`，复验脚本 `studio_released_batch.py`。Big 权重 SHA-256 `ce2a92a3188e8069e979f23aaec21cb6d439170aaf4002b4eaa234f0486e037e`。
