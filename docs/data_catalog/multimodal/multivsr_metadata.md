# MultiVSR Metadata（`multivsr_metadata`）

**用途定位：C｜切片/字幕元数据。** 含多语种划分、视频ID及词时标，须与已获取源视频求交；不能把元数据包数当语音样本数。

## 当前目录、状态与许可

- 数据说明：MultiVSR 的划分、YouTube ID、词时间戳和人脸轨迹发布物。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/multivsr_metadata/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/multivsr_metadata/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/multivsr_metadata/SHA256SUMS`。
- 机器记录：登记预期对象 10；状态已核验对象 10；状态核验字节 42,984,746,968；冷端收据对象 10；冷端收据字节 42,984,746,968。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`mit`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/sindhuhegde/multivsr](https://huggingface.co/datasets/sindhuhegde/multivsr)。
- 当前清单开头的对象示例：`raw/.gitattributes`、`raw/README.md`、`raw/dataset_teaser.gif`、`raw/multivsr.tar`、`raw/test.csv`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：en/fr/de/es/it/ca/ru/ja/zh/pl/pt/tr/nl
- 标注与格式：五CSV划分、ytids.txt、multivsr.tar中的文本与人脸轨迹.pckl
- 训练/测试划分：原数据有13语言元数据及拆分；当前视频交集未做全量核对。
- 冷端10文件、42984746968字节。CSV实际表头file_path,transcript,length,language,start；tar成员抽样含同片段.txt/.pckl。文本示例有Text、Lang及WORD/START/END/SCORE。

## 对 Falcon 的准备事项

- 元数据记录的train_major/minor、val_major/minor、test需要保留；同源视频的不同句段不能混入互相独立的划分。MIT是发布元数据仓库声明，不自动授予源视频任意用途。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
