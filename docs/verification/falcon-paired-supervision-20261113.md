# Falcon 同混音双目标同批监督短训

在 A800 的 `main` 上比较同一混音的两个目标请求是否放在同一次更新。推理固定 600 ms 历史、120 ms 输出、80 ms 前视。每次给原混音和一个指定嘴轨，输出一条目标波形。

## 唯一改动

每两个更新共用 8 个自然响度 LRS2 混音，每个混音选同一个 120 ms 双人有声窗。两个目标在至少两个相同的 40 ms 格中都有可测语音。每组每步 8 个请求，分别有 4 个第一轨、4 个第二轨请求。开组每步包含 4 个混音的双向正确标签，关组每步包含 8 个不同混音的单向正确标签，在下一步补齐另一方向。每两步的混音、嘴图和对应参考人声逐项相同。两组都训练过换嘴的正确标签，本轮估计的是这些请求同一步更新的作用。波形重建损失按请求独立计算，两组均未额外加入跨输出损失、响度增强或静默训练。

训练使用发布 Small epoch 714 初始化，冻结 FRCNN 和模型运行统计，523,113 个可训练参数。两组同一张 A800 交替更新，FP32、关闭 TF32、AdamW 学习率 0.0003。单种子 20261154，每组 96 步，768 个请求，覆盖同样的 120 个混音。48 对相邻更新的请求清单全部一致，第一对更新还按请求核对了实际混音、嘴特征和参考波形张量摘要。两组 96 步均有限且全部触发范数 5 裁剪，裁剪前范数中位数为关组 58.90、开组 57.66。初始固定权重下，同一请求跨批排法的波形差最大为 5.96e−8。上线式逐块入口与训练窗前向的最大误差为关组 8.94e−8、开组 5.96e−8。

## 受控与真实录音结果

SI-SDRi 表示相对原混音的改善。静默数值是输出能量相对原混音的变化，越负表示泄漏越少。开口能量使用同步参考轨的拟合分量估计，越接近 0 表示保留越多。

| 指标 | 发布 Small 滚动 | 关 | 开 |
| --- | ---: | ---: | ---: |
| LRS2 自然混音双向选对，32 对 | 28 | 29 | 27 |
| LRS2 同混音 ±6 dB 两种响度都双向选对，32 对 | 14 | 24 | 25 |
| LRS2 同混音 ±12 dB 两种响度都双向选对，32 对 | 4 | 20 | 18 |
| LRS2 独说波形 SNR | 21.93 dB | 20.21 dB | 20.19 dB |
| RoomReader 既有弱目标单向，12 段，平均 SI-SDRi | −7.25 dB | +0.24 dB | +3.50 dB |
| RoomReader 既有真实三人，12 段，平均 SI-SDRi | −7.02 dB | +3.46 dB | −0.84 dB |
| EasyCom 同室远场三人及以上，8 段，平均 SI-SDRi | −3.82 dB | +0.06 dB | +1.28 dB |
| RoomReader 可见静默，11 段，平均 / 最差 | −18.02 / −1.76 dB | −12.14 / −2.19 dB | −20.25 / −3.35 dB |
| EasyCom 可见静默，8 段，平均 / 最差 | −14.12 / −1.91 dB | −9.21 / −3.85 dB | −25.89 / −7.76 dB |
| MISP 可见静默，6 段，平均 / 最差 | −5.87 / −1.51 dB | −4.62 / −2.22 dB | −18.24 / −6.64 dB |
| RoomReader 重开前，19 段，平均 / 最差泄漏 | −21.61 / −0.36 dB | −17.75 / −0.30 dB | −27.65 / −0.30 dB |
| RoomReader 重开首 120 ms，19 段，目标能量变化均值 | −27.71 dB | −12.34 dB | −24.11 dB |
| RoomReader 重开首 120 ms，目标能量下降超过 6 dB | 14/19 | 10/19 | 13/19 |

既有 RoomReader 三人样本来自 5 场，共 3.44 秒。EasyCom 三人及以上样本来自 4 场，共 18.4 秒。两份资料的多人指标方向相反。MISP 是一场原始远场录音，没有同步个人参考波形，因此只报告可见静默泄漏。

开组在 19 条开口中有 12 条目标拟合能量下降、7 条上升，配对差中位数为 −7.52 dB。开组重开前平均漏音更低，最差一段仍只比原混音低约 0.30 dB。用 A800 本机 Whisper 对照官方时标首词。Base 和 Large-v3 均只在各自 6 条干净近讲参考可辨的片段上评分，关／开首词匹配分别为 3/6、2/6 和 2/6、2/6。词级结果不足以确认吞字率。19 条对齐原幅音频保留在服务器供回听。

## 固定真实混音换嘴

从 17 个未参与先前 RoomReader 训练和诊断的会话中，筛出 8 场的 22 对短窗，合计 6.56 秒。录音为 RoomReader 发布的在线会话聚合音频。每对两次前向共用完全相同的混音和时间窗，分别指定两名参与者的独立嘴轨。一个方向只有在输出与指定人的同步参考轨比与所有其他活跃说话人更接近时才计为选对，两次都选对才计为双向成功。弱目标在评分窗里的拟合分量比另一指定目标低 6 至 14 dB。

| 指标 | 发布 Small 滚动 | 关 | 开 |
| --- | ---: | ---: | ---: |
| 所有双向选对，22 对 | 2 | 5 | 5 |
| 弱目标双向选对，11 对 | 2 | 4 | 4 |
| 弱目标方向选对，11 对 | 2 | 4 | 4 |
| 弱目标方向平均 SI-SDRi，11 对 | −12.74 dB | −0.95 dB | −6.45 dB |
| 三人及以上双向选对，16 对 | 0 | 1 | 2 |

弱目标来自 5 场、13 个目标身份，评分总长 3.44 秒。开组双向成功新增 1 对、丢失 1 对，9 对不变。两组各有 4/11 对换嘴后仍偏向同一个人。弱目标平均 SI-SDRi 开减关为 −5.49 dB，11 段有 3 段改善、8 段退步，按会话重采样区间为 [−9.95, −0.67] dB。双向成功率差的同法区间为 [−25, +20] 个百分点。三人及以上来自 7 场、4.24 秒，开组多对一对，差值区间为 [−16.7, +35.3] 个百分点。区间反映这批会话的样本差异，训练仍只有一个种子。

S26 缺一位参与者的同步音轨，整场排除。候选拒绝中 94 条未通过原混音的参考拟合，11 条嘴轨不连续，7 条缺必要媒体。两组都改善了相对发布 Small 的部分选人指标。同批开关在真实弱目标上没有提高选对数，波形质量下降，重开衰减增加，静默漏音均值降低。当前结果不支持开启该开关做长训。

## 复现

A800 产物根目录为 `/root/group-shared/voiceprint/experiments/falcon_paired_supervision_20261113`。训练使用提交 `97bbbc1`，评测代码和输入、缓存、逐条输出与权重摘要分别见各结果目录的身份文件。真实样本音频、嘴图、逐条标注、转写及模型权重留在 A800。

```sh
H=/root/group-shared/voiceprint/experiments/falcon_diagnosis_execution_20261113
P=/root/group-shared/voiceprint/experiments/falcon_paired_supervision_20261113
PY=/root/miniforge3/envs/falcon/bin/python
RELEASE=/root/.cache/falcon_weights/falcon.pt
"$PY" -m tools.streaming.rolling_paired_supervision_ab \
  --frozen "$H/init_ab_followup/budget96_v2/data_identity.json" \
  --checkpoint "$RELEASE" --output "$P/train/another_run96" --steps 96
"$PY" -m tools.streaming.evaluate_roomreader "$H/silence_factorial_20261113/reopen_frozen.json" \
  "$P/onset/another_reopen" --input-cache "$H/silence_factorial_20261113/reopen_cache" \
  --release "$RELEASE" --experiment-arm "$P/train/run96/paired_off.pt" \
  --experiment-arm "$P/train/run96/paired_on.pt" --export-reopen-waveforms
```

其余结果位于 `onset/roomreader`、`onset/reopen_words.json`、`onset/reopen_words_large_v3.json`、`independent/easycom` 和 `independent/misp.json`。EasyCom 使用历史冻结缓存及同目录历史冻结评测工具 `conditioning_real_followup/easycom_frozen.py`，MISP 使用 `tools.streaming.evaluate_misp_intervals`。首词脚本使用 A800 已有 Whisper Base 和 Large-v3，在 GPU 上以 FP32 解码，CPU 数学库线程限制为 2。首次 CPU 预检生成异常数量的线程，已停止该进程，正式两次首词结果均完整。

```sh
"$PY" -m tools.streaming.evaluate_roomreader_same_mix_pair evaluate \
  "$P/real_pair/frozen_partitioned" "$P/real_pair/another_eval" \
  --release "$RELEASE" --arm "$P/train/run96/paired_off.pt" \
  --arm "$P/train/run96/paired_on.pt"
```


真实双嘴清单与输入摘要见 `real_pair/frozen_partitioned/manifest.json`，逐条方向评分及按会话配对统计见 `real_pair/eval_formal`。
