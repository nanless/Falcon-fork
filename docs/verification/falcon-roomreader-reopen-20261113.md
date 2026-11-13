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
