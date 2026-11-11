# VoxCeleb2（`voxceleb2`）

**用途定位：A｜说话人源候选。** 需要找到有音轨视频与已配对嘴图，按说话人和原视频隔离训练测试；与 VoxCeleb2-mix 及 AVYT 的同源片段去重。

## 当前目录、状态与许可

- 数据说明：大规模野外说话人音视频数据。
- 登记状态：`existing_cold_reference`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/voxceleb2/`；状态文件：该目录下 `status.json`。
- 冷端共用归档：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`；登记名称是关联线索，包名不证明该库每条数据可用。
- 本地许可记录：`official access terms`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：多语言；现有英语伪标注子集见派生章节
- 标注与格式：官方说话人身份与片段来源，不提供通用AVSR所需人工逐字转写
- 训练/测试划分：原说话人/视频划分须与 VoxCeleb2-mix 和 AVYT 对照，具体 Falcon 划分待建。
- 本目录 raw为空，引用联合大包；现有MMS-LLaMA训练所用媒体、英语筛选与伪转写见nested audio-visual/derived专项审计。AVYT仓库也发布Vox2 dev/test处理版本。

## 对 Falcon 的准备事项

- 优先合并已确认音画同步、可见目标说话人和目标音频对应的条目；不同仓库的预处理版本按原始 speaker/video/utterance 去重。目标语音提取不要求逐字转写，文字和语言筛选按识别或多语种专项需要另行处理。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

登记的冷端引用是共用的 `AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`，当前大小 1,980,009,103,360 字节。2026-11-11 只读查看大包开头 60 个成员，实际看到 `multimodal/audio-visual/preprocessed_data/lrs2_rebuild/audio/wav8k/min/tr/mix/` 下 WAV；前部未看到 VoxCeleb2，不能据此前部检查判断包内其余位置。这个共享包须按一次对象计数。
