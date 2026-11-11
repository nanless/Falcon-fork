# RoomReader（`roomreader`）

**用途定位：B｜多人会议评测/候选。** 个人视频/音频与全场混音需按会话、说话人和时间轴映射；先核目标音轨是否可作为参考，再选训练或测试划分。

## 当前目录、状态与许可

- 数据说明：在线多人会话，118 名参与者、约 322 个视频。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/roomreader/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/roomreader/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/roomreader/SHA256SUMS`。
- 机器记录：登记预期对象 1,339；状态已完成对象 1,339；状态核验字节 159,508,080,509；冷端收据对象 1,339；冷端收据字节 159,508,080,509。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`RoomReader Non-Commercial Licence Agreement; source terms apply`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://sigmedia.tcd.ie/room_reader_corpus_db/](https://sigmedia.tcd.ie/room_reader_corpus_db/)；[https://sigmedia.tcd.ie/RoomReader/](https://sigmedia.tcd.ie/RoomReader/)。
- 当前清单开头的对象示例：`raw/NON COMMERCIAL LICENCE AGREEMENT_RoomReaderDataset.pdf`、`raw/README_RoomReader.txt`、`raw/annotations/continuous engagement/EngAnno_1/S01_P011_Christian.csv`、`raw/annotations/continuous engagement/EngAnno_1/S01_P017_Julian.csv`、`raw/annotations/continuous engagement/EngAnno_1/S01_P025_liam.csv`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语，含母语/非母语参与者
- 标注与格式：修订后的TextGrid/EAF、逐词/逐句标注、个人音视频与整场音视频
- 训练/测试划分：含30场会议与118名参与者；训练测试划分需按会议和说话人隔离，官方划分需复核。
- 冷端1339文件、159508080509字节；其中322MP4、176WAV、443TextGrid、30EAF，另有特征和研究文档。源README说明30场会话、118人。抽查修订TextGrid有IntervalTier、xmin/xmax和对应文字。

## 对 Falcon 的准备事项

- 先按参与者与时间窗配准、剔除无脸/遮挡/未发声段，处理修订标记与非语音事件，按场次/说话人划分。研究使用许可与在论文中展示图像/视频是两件事；公开展示需按每场参与者同意记录核查。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

递归遍历 `raw/` 得 1,339 文件、159,508,080,509 字节：322 MP4、176 WAV、443 TextGrid、324 CSV、30 EAF，其余为说明及其他文件；这是文件计数，未逐文件媒体解码。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
