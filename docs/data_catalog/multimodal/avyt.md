# AVYT（`avyt`）

**用途定位：A｜大规模预训练候选。** talking/audio/video 可作为干净目标候选，需另外构造有可追溯干净目标的混音；dialog 是否提供逐人干净目标需逐样本确认。其 LRS2/Vox2 子集不要重复记为新源。

## 当前目录、状态与许可

- 数据说明：AVCocktail 工作中的约 1.42 TB 大规模野外派生训练集。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/avyt/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/avyt/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/avyt/SHA256SUMS`。
- 机器记录：登记预期对象 1,320；状态已核验对象 1,320；状态核验字节 1,419,416,723,248；冷端收据对象 1,320；冷端收据字节 1,419,416,723,248。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`not declared`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/nguyenvulebinh/AVYT](https://huggingface.co/datasets/nguyenvulebinh/AVYT)。
- 当前清单开头的对象示例：`raw/.gitattributes`、`raw/README.md`、`raw/avyt/silent-train-000000.tar`、`raw/avyt/silent-train-000001.tar`、`raw/avyt/silent-train-000002.tar`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语（抽查）；全量语言未重统计
- 标注与格式：label、length、sample_id 与封装音视频的 .video；转写来源及质量需沿用作者说明核对
- 训练/测试划分：来源清单包含 avyt talking/silent train、dialog train/test、lrs2 train/pretrain/test 及 vox2 test；不得把多个来源简单相加。
- 完整冻结仓库 1320 文件，其中 1318 个 tar，合计 1419416723248 字节。它同时包含 avyt 518 包、dialog 666 包、lrs2 78 包、vox2 56 包。tar 数不是样本数。

## 对 Falcon 的准备事项

- 代表 talking、silent 和 Vox2 .video 均为 96×96/25fps + 16kHz 单声道AAC。需解码适配本项目、检查长度/文本、按原始视频ID去重。仓库还含 LRS2 valid/test 和不同信噪比 test，必须明确排除评测条目。AVYT 根卡片没有清楚声明许可，不能把 AVCocktail 的 CC BY-SA 自动套给 AVYT。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
