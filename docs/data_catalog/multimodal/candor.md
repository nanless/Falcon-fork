# CANDOR（`candor`）

**用途定位：B｜双人会议候选。** 原始音视频与转写适合自然对话；需确定麦克风通道、目标说话人与人脸同步，并满足现有研究用途条款。

## 当前目录、状态与许可

- 数据说明：大规模自然双人对话的音视频与行为标注。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/candor/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/candor/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/candor/SHA256SUMS`。
- 机器记录：登记预期对象 166；状态已完成对象 166；状态核验字节 951,758,137,196；冷端收据对象 166；冷端收据字节 951,758,137,196。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`academic/scientific research only; no redistribution or re-identification`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://candor.usc.edu/](https://candor.usc.edu/)。
- 当前清单开头的对象示例：`raw/raw_media_part_001.zip`、`raw/raw_media_part_002.zip`、`raw/raw_media_part_003.zip`、`raw/raw_media_part_004.zip`、`raw/raw_media_part_005.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语为主；2026-10-16 盘点未逐会话做语言识别
- 标注与格式：代表分卷有各说话人原始/处理视频、混合音频、转写CSV/JSON、时间偏移和行为数据
- 训练/测试划分：166个原始媒体ZIP是发行对象数；训练测试按来源会话独立划分，2026-10-16 盘点未确认分离数据集条目数量。
- 冷端 166 个 raw_media_part_*.zip，951758137196 字节，现有归档回执确认该授权分卷集合完成。抽查第1卷含10个会话目录、视频、音频、转写；这是分卷样本，不能乘166当作核实会话总数。

## 对 Falcon 的准备事项

- 代表 transcript_audiophile.csv 列有 turn_id、speaker、start、stop、utterance、overlap 等。转写版本有多种，不能默认全部为人工金标准。需评估标注质量，裁段/裁嘴、处理双方重叠、按会话或说话人划分。学术研究使用受限，不能公开原始媒体或重识别参与者。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
