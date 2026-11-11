# FaceStar：高质量英语会话音视频（`facestar`）

**用途定位：B｜高质量小样本。** 抽查到两位说话人的配对音视频，可作为构造定向混音的候选；说话人数量少，本地许可字段记为 CC-NC-4.0，具体使用条件需核对来源，不宜作为通用训练主体。

## 当前目录、状态与许可

- 数据说明：FaceStar studio AV conversational speech
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/facestar/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/facestar/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/facestar/SHA256SUMS`。
- 机器记录：登记预期对象 115；状态已完成对象 115；状态冷端对象 115；冷端收据对象 115；冷端收据字节 29,817,491,437。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-NC-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://github.com/facebookresearch/facestar](https://github.com/facebookresearch/facestar)。
- 当前清单开头的对象示例：`raw/female_speaker_testset/session01.zip`、`raw/female_speaker_testset/session02.zip`、`raw/female_speaker_testset/session03.zip`、`raw/female_speaker_testset/session04.zip`、`raw/female_speaker_testset/session05.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：抽查 session ZIP 内只有对应 MP4 和 WAV，没有句子文字。
- 训练/测试划分：女声训练55场/测试7场，男声训练45场/测试5场；不同会话对应同两名说话人。
- 冷端 115 个文件，共 29,817,491,437 字节：112 个会话 ZIP，另有论文及展示材料。
- 官方按女说话人训练 55 场/测试 7 场、男说话人训练 45 场/测试 5 场发布，约 10 小时、2 人。
- 抽查女/男训练 session01，分别有 18/9 对 MP4 和 WAV。样本视频为 1300×1600、60 帧/秒，音频为 16 kHz、单声道浮点 WAV。
- 既有下一步：可用于音视频表示学习或视频压缩/噪声研究；若用于识别，应先生成并审核转写。

## 对 Falcon 的准备事项

- 没有逐字文字时不能直接进行有文字监督的识别训练。
- 只有两位说话人，规模和说话人多样性都有限；会话是长片段，需要分句和口型裁剪。
- 发布者现成 train/test 应保持隔离。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
