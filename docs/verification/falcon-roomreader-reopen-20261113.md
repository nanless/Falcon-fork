# RoomReader 可见静默后重新开口的验收方法

重开评测沿用既有独立真实会话的五场与相同目标身份。使用原会话混音、指定个人视频和官方个人参考麦克风。语音分离始终只读取混音及一条目标嘴图，只返回一路声音。

1. 官方人工校订语句起点与词级时间一致到筛选窗。排除目标此前约 420 ms 内的语句、词和副语言事件。目标个人参考麦需从安静转为稳定语音，同时另一人有可测声音。
2. 围绕官方词起点，用个人麦上连续三个 10 ms 能量格定位声学起点。评分前 200 ms 为 onset 前 240 到 40 ms，保留 40 ms 保护。开口评测分别取 onset 后 120 和 240 ms。模型只跑一次完整上下文，从原输出截取三种评分窗。
3. 评分窗口要求目标嘴图连续有效。开口窗要求参考轨能对原混音做全秩线性拟合，残差不高于 −10 dB，且原混音中的目标成分能量充足、投影方向为正。
4. 开口目标能量保留用输出的目标拟合成分相对原混音的目标拟合成分计算。低于 −6 dB 的段数单报，同时保留 SI-SDRi、目标与最强干扰的拟合比、投影反向和拟合残差。这是头段能量和波形证据，词错误率需另外听辨或转写。
5. 前置静默泄漏用输出对原混音能量比计算。四臂同一原始混音、嘴图、参考轨、时间窗和可评分条件。

本轮冻结 19 条，覆盖 5 场，目标为学生的有 15 条、教师 T002 有 4 条。重开缓存全部确认嘴图可见，两个开口窗口均通过原混音拟合筛选。声学起点与官方词时刻之差为 −20 到 +110 ms。实测基线中发布权重滚动在首 120 ms 目标投影下降平均 27.71 dB，14/19 条低于 −6 dB，SI-SDRi 平均 −8.88 dB。旧流式在首 120 ms 的平均下降为 4.65 dB、2/19 条低于 −6 dB，但其开口前静默平均仍保留相对混音 −5.93 dB 的能量，发布滚动为 −21.61 dB。

## 命令

```sh
python -m tools.streaming.select_roomreader_reopen \
  --dataset-root "$ROOMREADER_ROOT" --frozen "$ROOMREADER_ORIGINAL_HOLDOUT" \
  --output "$REOPEN_MANIFEST"

python -m tools.streaming.evaluate_roomreader "$REOPEN_MANIFEST" "$REOPEN_CACHE" \
  --prepare-cache --dataset-root "$ROOMREADER_ROOT" --face-model "$FACE_MODEL"

python -m tools.streaming.evaluate_roomreader "$REOPEN_MANIFEST" "$REOPEN_RESULT" \
  --input-cache "$REOPEN_CACHE" --release "$FALCON_RELEASED" \
  --experiment-arm "$FIRST_ARM" --experiment-arm "$SECOND_ARM" \
  --experiment-arm "$THIRD_ARM" --experiment-arm "$FOURTH_ARM"
```

服务器清单、缓存、基线和四臂结果位于 `/root/group-shared/voiceprint/experiments/falcon_diagnosis_execution_20261113/silence_factorial_20261113/`。不随仓库分发原始音视频、嘴图、语料清单或权重。

## 正式四臂真实结果

| 响度反事实 | 可见静默监督 | 开口前泄漏均值／最坏 | 首 120 ms 目标能量变化均值／最坏 | 首 120 ms 低于 −6 dB | 首 240 ms 目标能量变化均值 | 首 240 ms SI-SDRi |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 关 | 关 | −11.68／−0.27 dB | −12.50／−33.09 dB | 13/19 | −12.31 dB | −5.69 dB |
| 开 | 关 | −23.31／−1.04 dB | −19.17／−57.19 dB | 15/19 | −17.27 dB | −4.93 dB |
| 关 | 开 | −16.99／−0.65 dB | −13.79／−45.31 dB | 13/19 | −12.47 dB | −4.56 dB |
| 开 | 开 | −24.06／−1.76 dB | −21.56／−67.24 dB | 13/19 | −18.90 dB | −4.20 dB |

19 条全部通过。只保留 15 个学生目标，首 120 ms 目标能量变化均值依序为 −13.24、−21.29、−15.21、−25.82 dB，低于 −6 dB 分别为 11、12、11、12 条，最坏值与全量相同。联合臂最严重的 3 段，首 120 ms 目标投影衰减为 −67.24、−47.14、−40.76 dB，相对原混音的总输出能量也分别下降 50.18、34.92、34.91 dB。三个原混音对参考拟合残差均低于 −18 dB。这是明显的过度抑制证据。最严重的拟合丢失接近测量地板，不以它推断字词丢失多少。

按实际训练运行身份和数据摘要复查，训练是 26 个新场片段，来自 S08/S10/S17/S20/S27。重开和原留出来自 S05/S06/S11/S15/S16。场次及目标身份交集为空。共享老师 T001/T002 依据原始训练选窗协议均无校订语句、词或副语言事件，已选训练窗中共享教师个人麦相对原混音的最高能量比为 −43.83 dB。测试仍单列学生，避免将老师重复出现和房间重叠混在一起。

配对逐条指标和子集报告分别为 `reopen96/rows.jsonl`、`reopen96/scope_analysis.json`，训练排除核验为 `reopen_training_exclusion_verified.json`。
