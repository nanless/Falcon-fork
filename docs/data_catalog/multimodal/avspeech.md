# AVSpeech（`avspeech`）

**用途定位：A｜预训练候选。** 只在核对可见目标、源音轨和嘴部对齐后构造混音目标；原始和 gapfill/历史替代版不可直接加总。

## 当前目录、状态与许可

- 数据说明：大规模野外音视频语音片段。
- 登记状态：`merged_complete`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/avspeech/`；状态文件：该目录下 `status.json`。
- 独立媒体与记录：热端 `avspeech/videos/`、`file_lists/` 等；迁移件位于 `/aispeech-cfs/voiceprint/data/multimodal/_local_overflow_20261102/avspeech/`；另引用上述共用大包。避免将不同处理/重复副本相加。
- 本地许可记录：`dataset-specific terms`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://looking-to-listen.github.io/avspeech/](https://looking-to-listen.github.io/avspeech/)。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：多语言；当前未全量统计
- 标注与格式：YouTube ID、片段开始/结束时间、归一化人脸位置；没有官方逐字转写
- 训练/测试划分：暂存有 videos/train、test、legacy_unmatched 等目录；每部分历史来源/迁移范围需按清单核对。
- 媒体主要在暂存根 videos/{train,test,legacy_unmatched}/shard_*，另有 extras/superseded 和补录目录。raw 为空不表示没有媒体。status 的 merged_complete 只说明历史合并结束，后续补录另有脚本。

## 对 Falcon 的准备事项

- 2026-09-21 status 留存覆盖 1847042/2805118，2026-09-23 coverage.json 为 1862581/2805118；2026-10-16 盘点读取现有 train.txt 1888167 行、test.txt 120957 行、legacy.txt 229520 行、superseded.txt 313777 行。它们是不同日期、不同口径，当前行数不等于去重后媒体覆盖率。2026-10-16 盘点没有遍历百万媒体或重算总时长，不报“已经全量完成”。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
