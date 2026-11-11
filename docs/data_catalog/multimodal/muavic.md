# MuAViC Full（`muavic`）

**用途定位：B｜多语言来源。** 源视频加 metadata 与转写/音频切片关联，可构造多语种目标；英语部分复用 LRS3，避免重复。3875/3882 是已接受的部分源视频集合。

## 当前目录、状态与许可

- 数据说明：九语种 AVSR/AVST 完整重建集，依赖 OpenSLR 音频和原视频。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/muavic/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/muavic/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/muavic/SHA256SUMS`。
- 机器记录：登记预期对象 3,882；状态已核验对象 3,875；状态核验字节 767,683,633,536；冷端收据对象 3,875；冷端收据字节 767,683,633,536。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC BY-NC 4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://github.com/facebookresearch/muavic](https://github.com/facebookresearch/muavic)。
- 当前清单开头的对象示例：`raw/source_videos/ar/test/U7yWsUX5TLY.mp4`、`raw/source_videos/ar/test/VDJ88bKCi-A.mp4`、`raw/source_videos/ar/test/XAW3SoGevgg.mp4`、`raw/source_videos/ar/test/uz6GpKZkWRc.mp4`、`raw/source_videos/ar/test/wFK0ccfE3to.mp4`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：阿拉伯语、德语、希腊语、西语、法语、意语、葡语、俄语；英语复用LRS3
- 标注与格式：本项只存原始YouTube视频；人脸轨迹在muavic_metadata，转写与切分还依赖官方OpenSLR/mTEDx/LRS3准备流程
- 训练/测试划分：源视频按语言 train/test 分布，官方切分与元数据需关联；英语复用LRS3。
- 冷端3875个源MP4、767683633536字节；计划3882，缺7。归档回执明确这是用户接受的部分归档，不能写“官方全量已完成”。8种非英语视频数为ar109/de70/el130/es1013/fr970/it500/pt830/ru253。

## 对 Falcon 的准备事项

- 需补齐/定位各语言音频与转写、按官方轨迹和时间窗裁片、裁嘴、检查音画同步、保留官方train/valid/test。7个缺视频涉及多个划分；应在实际清单显式排除并报告对应句子损失，不能按7/3882直接推算小时损失。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
