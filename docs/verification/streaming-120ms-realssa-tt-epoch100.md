# 120 ms 流式模型第 100 轮：RealSSA-Sim 测试集配对评测

## 结果

使用训练过程中选中的第 100 轮检查点与公开 Falcon，对 RealSSA-Sim 官方 tt 索引逐条做配对评测。两模型使用同一条混合音频、目标音频和嘴图；总索引 9,169 条，全部遍历，9,166 条具有可计算的目标，3 条目标音频全零，SI-SDRi 无定义。均值对两模型使用同样的 9,166 条有效配对。

| 模型 | 平均 SI-SDRi | 中位数 SI-SDRi |
| --- | ---: | ---: |
| 公开 Falcon | 7.731474 dB | 7.495539 dB |
| 120 ms 流式模型，第 100 轮 | 4.246685 dB | 3.931897 dB |
| 流式减公开，逐条配对 | -3.484789 dB | -3.289778 dB |

有效配对中，流式模型在 268 条上高于公开模型；流式 SI-SDRi 低于 0 的有 285 条。这是第 100 轮训练中的一次完整测试，训练继续进行；最终模型需在收敛后重新评测。

## 权重、数据和协议

- 流式权重：`/root/group-shared/voiceprint/experiments/falcon_streaming_120ms_realssa_full_20261111/epochs/epoch-0099-step-00004900.pt`，实际第 100 轮、4,900 次更新、320 通道 v4；SHA-256 `fb225469889b4581a63e09883df387cfefb82f1c2d0e97c8b39d053323ee50df`。
- 公开 Falcon 权重 SHA-256：`a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`。
- 完整 tt 索引 SHA-256：`f72e8efec3b7f8e1c242d291b57725b170db034097887d33bda45350d571f8c3`；9,169 条、19 个 TAR。
- 使用第 0 通道、中心裁剪 2 秒；短于 2 秒的输入补零，指标使用补零后的完整 2 秒。公开 Falcon 观察完整 2 秒，新模型按 120 ms 块处理。指标为去均值 SI-SDRi，不是多抽头 SDRi、PESQ 或论文整句评测结果。
- 按索引分两段在双 A800 执行；训练同时继续。逐条核对分段范围、索引及样本 ID，确认 9,169 条均覆盖，没有重复和漏号；两段的索引、权重、代码摘要及计算协议一致。
- 官方训练和测试划分此前审计出目标源路径重叠，见 [RealSSA 数据与训练说明](../REALSSA_STREAMING_TRAINING.md)；该结果属于官方划分内的比较。

零目标样本为 `00003_02_41188`、`00012_02_84551` 和 `50016_74250`。逐条读取原始 TAR 后，三条目标音频的四个通道整段均严格全零，混音和嘴图可读；此前整句评测中相同三条的 SDRi/PESQ 也没有有效结果。它们在本次两模型配对统计中共同排除，没有记作零分。

## 原始证据

服务器目录：`/root/group-shared/voiceprint/experiments/falcon_streaming_120ms_realssa_full_20261111/midtrain_tt_full_epoch100/`。

- `part0.json`、`part1.json`：两段原始逐条结果与输入/代码/权重摘要；`part0.log`、`part1.log` 为逐条运行记录。
- `paired-full-audited-summary.json`：独立核验的覆盖、有效数和逐样本汇总，SHA-256 `b1d4b3e99fd58ae8c28c1d0dc5294c058e1d96b682023cdfbe4aad21c9fc5b68`。
- `zero-target-audit.json`：三条全零目标的原始 TAR 和四通道解码核查。

评测过程先以固定随机种子从同一官方 tt 中抽取 128 条，覆盖全部 19 个分片：公开 Falcon 和流式模型的 SI-SDRi 分别为 7.499475 dB、4.138984 dB；样本清单和结果保存在相邻 `midtrain_tt128_seed20261111_epoch100/` 目录。最终全量结果以上文 9,166 条有效配对为准。
