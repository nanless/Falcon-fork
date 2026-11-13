# 同室实录复核与等预算短训结论

2026-11-13，A800。此轮所有前向以原始单通道会话音频和明确指定的一个嘴部轨迹为输入，每次只输出一路目标语音。训练来自 LRS2；RoomReader、EasyCom 和 MISP 来自不同的原始录制来源。它们均未参加本轮短训，也未用于调整两臂超参数。公开 Falcon 权重的完整训练样本清单未公开，来源独立性不推及其未知训练数据。

- [固定预算短训及受控 LRS2 结果](falcon-rolling-init-short-ab-20261113.md)
- [RoomReader 原始在线会话混音及同步个人参考](falcon-roomreader-independent-20261113.md)

## MISP2022 同室远场

使用 AVSD eval 的同一场录音、原生 Far_0 单通道和官方逐帧嘴框。冻结六段真实重叠，共 8.16 秒。整个评分段都有相同的三位说话者在 RTTM 中同时活动，部分帧有第四人。冻结六段目标可见且静默的区间，共 8.40 秒，目标活动边界有 80 ms 保护。三人组在同一个原始音频上，分别指定两个仍在说话的人的嘴图。

| 模型 | 可见静默输出/混音均值 | 最差静默 | 换嘴波形差/两路输出均能量 |
|---|---:|---:|---:|
| 发布整段 | −19.15 dB | −0.55 dB | −3.22 dB |
| 发布滚动 | −5.87 dB | −1.51 dB | −2.12 dB |
| 历史流式 step 1375 | −17.80 dB | −12.95 dB | −4.85 dB |
| 发布初始化 96 步 | −16.76 dB | −5.52 dB | −0.74 dB |
| 从零初始化 96 步 | −21.47 dB | −19.27 dB | −30.54 dB |

官方 AVSD 资料对该场只有近讲转写，无对应近讲 WAV。换嘴波形差可检出模型响应，无法单独核验输出属于谁。发布热启在此场的静默最差样本仍有明显串音，历史流式在该小组抑制较强。从零臂的静默更低，但三人重叠输出相对原混音平均衰减 21.90 dB，换嘴反应也很弱。

## EasyCom 同室原阵列声道

使用 AR 眼镜阵列第 5 通道原始录音、人工语音活动标注和经身份标注的目标脸框。原视频 20 fps，保持到 25 fps 网格；框内定位真实嘴部，评分帧口型覆盖至少 95%。重叠组每帧至少三人活动，静默组目标连续不说话且可见。各模型共用同一音频、口型和评分区间，按官方近讲轨做目标波形对照。原始近讲轨约 48008.6 Hz 且存在时钟微偏差。当前按标称 48 kHz 重采样，近讲参考与声学混响及麦距有差别，波形指标仅作小样本配对证据。没有近讲轨的干扰者仍计入活动人数。

| 模型 | 重叠组平均目标 SI-SDRi，8 段 | 可见静默输出/混音均值，8 段 |
|---|---:|---:|
| 发布滚动 | −3.82 dB | −14.13 dB |
| 发布初始化 96 步 | +1.46 dB | −22.03 dB |
| 从零初始化 96 步 | −9.25 dB | −22.45 dB |

重叠组累计 18.4 秒，热启组 5/8 条高于原混音。可见静默组累计 24.0 秒。增加到每组八段时发现 20 fps 视频切片可能在最后留下不足 40 ms 音频，已修复时钟尾块，重跑中前五段各项指标保持一致。受控 LRS2 的弱目标改进和 RoomReader 的真实选人结果支持继续保留发布初始化臂，同时用自然响度保真和同室泄漏阻止退化。目前是单一训练种子和 96 步，未启动长训。

## 复核

在 Falcon 环境从仓库根运行。评测按非商业研究许可执行，媒体、逐条清单、个人标注、输出波形和权重留在服务器。

```sh
python -m tools.streaming.select_misp_intervals "$MISP_SELECTION" --source "$MISP_AVSD"
python -m tools.streaming.prepare_misp_intervals "$EXPERIMENT_ROOT" "$MISP_CROPS" \
  --shortlist "$MISP_SELECTION" --source-media "$MISP_EXTRACTED_SESSION"
python -m tools.streaming.evaluate_misp_intervals --manifest "$MISP_CROPS/manifest.json" \
  --output "$MISP_RESULT" --model release96 rolling "$FALCON_WARM_AFTER_96" \
  --model scratch96 rolling "$FALCON_SCRATCH_AFTER_96"

python -m tools.streaming.select_easycom_intervals "$EASYCOM_MAIN" "$EASYCOM_SELECTION"
python -m tools.streaming.evaluate_easycom "$EASYCOM_SELECTION" "$EASYCOM_RESULT" \
  --dataset-root "$EASYCOM_MAIN" --face-model "$FACE_LANDMARKER" --released "$FALCON_RELEASED" \
  --warm "$FALCON_WARM_AFTER_96" --scratch "$FALCON_SCRATCH_AFTER_96" --per-group 8
```

MISP 及 EasyCom 的原始逐条结果在 `/root/group-shared/voiceprint/experiments/falcon_independent_validation_20261113/` 的 `misp_probe_baselines.json`、`misp_probe_init96.json` 及 `easycom/repo_recheck_tail/result.json`。原始标注、音频入口身份、嘴图切片哈希和运行日志保存在同一实验树。

数据来源为 [MISP 2022 Challenge](https://arxiv.org/abs/2303.06326)、[EasyCom](https://github.com/facebookresearch/EasyComDataset) 和 [Reverdy et al., RoomReader, LREC 2022](https://aclanthology.org/2022.lrec-1.268/)。
