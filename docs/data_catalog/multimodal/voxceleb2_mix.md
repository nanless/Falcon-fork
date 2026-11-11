# VoxCeleb2-mix：目标说话人提取的分片下载（`voxceleb2_mix`）

**用途定位：A｜分离训练候选。** 冷端音频/原视频分片可按源视频和说话人建立目标；先验证分片拼接、完整成员及混音生成协议，已有 VoxCeleb2 与 AVYT 来源去重。

## 当前目录、状态与许可

- 数据说明：VoxCeleb2-mix AV target-speaker extraction
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/voxceleb2_mix/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/voxceleb2_mix/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/voxceleb2_mix/SHA256SUMS`。
- 机器记录：登记预期对象 14；状态已完成对象 14；状态冷端对象 14；冷端收据对象 14；冷端收据字节 563,291,453,440。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-SA-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix](https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix)。
- 当前清单开头的对象示例：`raw/audio_clean_part_aa`、`raw/audio_clean_part_ab`、`raw/audio_clean_part_ac`、`raw/audio_clean_part_ad`、`raw/audio_clean_part_ae`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：多语种，原VoxCeleb2来源
- 标注与格式：目标是混合语音与目标说话人提取；2026-10-16 盘点未发现逐字文本包。
- 训练/测试划分：首片实见 audio_clean/test 与 orig/test；训练与测试全量成员尚未遍历。
- **2026-11-11 当前证据**：冷端 14/14 个登记分片、563,291,453,440 字节；`audio_clean_part_aa` 至 `ag` 和 `orig_part_aa` 至 `ag` 各 7 片。`.write_test_avsr` 是测试文件，不计入发行分片。
- 两组首片 `file` 识别为 GNU tar 流；只读前 15 个成员可见 `audio_clean/test/id00017/.../*.wav` 与 `orig/test/id00017/.../*.mp4`。[14 个分片的 stat 与首片成员证据](voxceleb2-mix-headers-20261111.json)记录了这次抽查；尚未遍历完整归档或确认混音/干净目标全量配对。
- 发布者的 [数据卡用法](https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix#usage) 将 `orig*` 和 `audio_clean*` 分别按文件名顺序拼成 `orig.tar`、`audio_clean.tar`，再分别解包；当前两组各有 `aa`–`ag` 七片。操作时只选择对应 `orig_part_aa`…`ag` 或 `audio_clean_part_aa`…`ag`，不包含 `.write_test_avsr`，先核对 SHA256SUMS、目标可用空间和全部分片；本轮只读首片，没有实际拼接或验收完整 tar。
- 下一步：核验全部分片拼接、媒体成员、音视频配对和混音配方。

## 2026-11-11 首对实际解码与音轨对照

两组 `part_aa` 中的 `test/id00017/01dfn2spqyE/00001` 是一组同键成员。WAV 为 16 kHz 单声道 PCM16、68,608 点、4.288 秒；MP4 视频为 224×224、25 fps、107 帧、视频轨 4.280 秒，同时有 16 kHz 单声道 AAC 音轨。WAV 首秒及 MP4 视频首帧解码均成功。

将这一个 MP4 音轨和 WAV 均解码为 16 kHz 单声道，各为 68,608 点；在 ±3,200 点（±200 ms）搜索音轨时移，最佳 lag 为 0，重叠皮尔逊相关系数为 0.999999992。该例的音轨和 WAV 高度一致；这个结果只覆盖音频对音频，并没有确认嘴部运动与音频同步或全部分片配对。

[首对媒体、文件身份与音轨对照](../vox-first-pair-audio-sync-20261111.json)。

## 对 Falcon 的准备事项

- 已抽查两组首片开头为 tar；完整分片组的容器与所有成员尚未逐项核对。
- 冷端已有原始视频分片；需要进一步核对完整音视频和混音配对。目标分离不要求逐字文字。
- 混合和干净目标数据是同一源片段的派生版本，训练/测试必须按原始视频关联去重。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

`raw/` 可见 15 文件、563,292,502,016 字节；其中 14 个是两组原始分片，额外 1,048,576 字节 `.write_test_avsr` 为测试标记，不计入 14 片回执。两组分片的成员与拼接方法见卡片既有证据。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
