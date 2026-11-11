# AudioSet Strong：有起止时间的声音类别标注（`audioset_strong`）

**用途定位：C｜噪声筛选元数据。** 只有事件 TSV，适合选噪声；本地训练 TSV 尾行曾发现截断，需来源校正，不能当音视频语料。

## 当前目录、状态与许可

- 数据说明：AudioSet Strong labels (CC-BY-4.0)
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/audioset_strong/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/audioset_strong/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/audioset_strong/SHA256SUMS`。
- 机器记录：登记预期对象 4；状态已完成对象 4；冷端收据对象 4；冷端收据字节 75,750,940。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://research.google.com/audioset/download_strong.html](https://research.google.com/audioset/download_strong.html)。
- 当前清单开头的对象示例：`raw/audioset_eval_strong.tsv`、`raw/audioset_eval_strong_framed_posneg.tsv`、`raw/audioset_train_strong.tsv`、`raw/mid_to_display_name.tsv`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：非逐字语料，类别名称为英语
- 标注与格式：TSV 描述声音类别、开始时间和结束时间，不记录说了什么。
- 训练/测试划分：训练与评测事件 TSV 分开；本地训练 TSV 存在截断记录，不能当成完整媒体集。
- 冷端只有 4 个 TSV 文件，共 75,750,940 字节，没有音视频媒体。
- 本地训练 TSV 的有效记录为 933,241 个声音区间、103,299 个独立片段，类别代码 447 种；评测 TSV 读取到 139,538 行。
- 官方页面当前写训练部分为 934,821 个区间、103,463 个片段。本地有效统计与之不一致；按二进制逐行复查发现末行只剩 3 字节 H5f，没有制表符，是明确截断行。
- 既有下一步：先保留截断证据并核对原下载验收记录，后续修复训练标签；当前仅作为辅助索引，不计入有监督AVSR样本数。

## 对 Falcon 的准备事项

- 声音类别标签不能替代逐字转写；即使复用原 AudioSet 媒体，仍不构成现成 AVSR 训练样本。
- 该训练 TSV 存在明确截断证据，不能标作完整标签；冷端 SHA 清单覆盖和历史归档状态不能替代来源内容完整性验证。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
