# 独立真实会话 RoomReader 的单目标提取短测

本轮使用 RoomReader 原发行的多人在线交谈混音、同步个人视频、个人音轨、人工校订语句和词时间戳。训练使用 LRS2 的短训两臂。RoomReader 的参与者、会话和录制流程都与 LRS2 不同。发布模型完整训练清单不可见，因此独立性基于公开的发行来源及本轮输入。

- 三人同时说话要求同一评分区间有至少三个固定参与者同时命中校订语句与词时间戳，并排除副语言标注。区间包含真实抢话和简短反馈，按评分区间合计时长，前后 640 ms 上下文只供模型使用。
- 弱目标以同步个人轨和原始会话混音判定。目标在讲话，目标相对其他活跃参与者的功率较低。每条记录保留目标与干扰分量及原混音拟合残差。
- 可见静默要求目标不落在讲话或副语言标注内，目标个人轨近静默且目标嘴图在评分帧全部有效。缺跟踪单列，静默打分不借助目标活动标注进入模型。零音频输入不作为静默验证。
- 使用指定个人视频的嘴图。每个模型只接收原会话混音及一条嘴图轨，始终输出一路语音。原始 32 kHz 音轨按 25 fps 网格转为 16 kHz。冻结清单全部使用同一 MediaPipe 模型重提嘴部，源 OpenFace CSV 可用的 28 条另作交叉复测。评分区间至少每 40 ms 有一个有效嘴部输入，缺帧不会被当作通过静默测试。
- 目标 SI-SDR 使用同步个人音轨作参考。用所有参与者同步音轨同时做线性拟合时，目标拟合分量大于最大单个干扰分量记为“目标分量胜出”。此指标的解释依赖参考对齐和拟合残差，尤其短反馈与波形伪影不能靠拟合一项判定。目标静默时只报输出对原混音的能量比，更负表示输出更安静。

## 实测结果

冻结 36 条。MediaPipe 完整评分 35 条，三人重叠 12/12（3.44 s），弱目标 12/12，可见静默 11/12（15.4 s）。静默排除 1 条口型跟踪未全覆盖。结果按条等权平均。

| 模型 | 三人重叠 SI-SDR | 三人目标分量胜出 | 弱目标 SI-SDR | 弱目标分量胜出 | 可见静默 输出/混音 | 静默高于 −10 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原混音 | -7.23 dB | 0/12 | -9.05 dB | 0/12 | 0.00 dB | 11/11 |
| 旧流式 | -8.90 dB | 1/12 | -10.85 dB | 0/12 | -5.25 dB | 10/11 |
| 发布权重整窗 | -12.86 dB | 5/12 | -14.87 dB | 3/12 | -27.54 dB | 5/11 |
| 发布权重滚动 | -14.25 dB | 5/12 | -16.29 dB | 3/12 | -18.02 dB | 5/11 |
| 发布初始化 96 步 | -5.15 dB | 8/12 | -5.25 dB | 7/12 | -25.28 dB | 0/11 |
| 从零初始化 96 步 | -13.22 dB | 0/12 | -14.44 dB | 0/12 | -21.13 dB | 0/11 |

三人同时说话的 12 条中，发布初始化的 96 步臂有 7 条波形 SI-SDR 高于原混音，目标分量胜出 8 条。输出仍包含不能由同步参与者参考轨解释的分量。发布滚动在部分短片目标分量占优，同时波形 SI-SDR 降低。从零臂在静默压低了输出，但有声目标选择仍弱。

OpenFace 交叉复测按官方 CSV 能覆盖的 28 条冻结样本运行，25 条通过嘴图覆盖要求。三人 9/9 条、弱目标 9/9 条、静默 7/10 条。三人组原混音／旧流式／发布滚动的 SI-SDR 均值 −6.26／−7.27／−11.62 dB。静默三者输出相对混音 0／−6.25／−16.48 dB。同一人目标在短重叠时语音能量不足，或分离伪影占主导，导致少量极低 SI-SDR。应和逐条输出、覆盖及误差并读。

## 复现

冻结清单使用受授权 RoomReader 根目录，不含在仓库中。清单格式为 `{"rows":[{"session":"Sxx","target":"Pnnn","start_ms":0,"end_ms":400,"group":"triple"}]}`。实际清单包含更完整的选段证据。用显式参数指定发行目录、发布权重、可选旧流式权重、最多可增设多个滚动评测权重。输出目录必须不存在。

```sh
python -m tools.streaming.evaluate_roomreader \
  "$ROOMREADER_MANIFEST" "$EVAL_DIR" \
  --dataset-root "$ROOMREADER_ROOT" --release "$FALCON_RELEASED" \
  --face-model studio/web/vendor/mediapipe/face_landmarker.task \
  --old "$FALCON_OLD_STREAMING" \
  --arm "$FALCON_WARM_AFTER_96" --arm "$FALCON_SCRATCH_AFTER_96"
```

推理依赖项目 Falcon 环境、OpenCV、SciPy、soundfile、pandas、ffprobe 和 MediaPipe 0.10.32。省略 `--face-model` 使用 RoomReader 官方 OpenFace CSV，需事先确认其时间覆盖。每条运行记录包含模型权重摘要、嘴图与混音切片摘要、目标 ID、有效时间及指标。A800 原始输出位于 `/root/group-shared/voiceprint/experiments/falcon_diagnosis_execution_20261113/independent_real_20261113/`，冻结清单位于 `/root/group-shared/voiceprint/experiments/falcon_independent_validation_20261113/roomreader_shortlist.json`。本仓提交代码和聚合结果。

语料来源为 Reverdy et al., LREC 2022，[RoomReader: A Multimodal Corpus of Online Multiparty Conversational Interactions](https://aclanthology.org/2022.lrec-1.268/)。本轮仅作非商业研究。
