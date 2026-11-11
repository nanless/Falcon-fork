# ViSpeR：四语种转写、时间信息与视频编号（`visper`）

**用途定位：C｜元数据候选。** 包含语言文字与视频ID，但媒体需按来源取得并对齐；当前不能直接训练视听分离。

## 当前目录、状态与许可

- 数据说明：ViSpeR multilingual AVSR transcripts + ID lists
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/visper/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/visper/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/visper/SHA256SUMS`。
- 机器记录：登记预期对象 20；状态已完成对象 20；状态冷端对象 20；冷端收据对象 20；冷端收据字节 6,971,436,241。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-NC-2.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/tiiuae/visper](https://huggingface.co/datasets/tiiuae/visper)。
- 当前清单开头的对象示例：`raw/arabic_test.tar.gz`、`raw/arabic_train.tar.gz`、`raw/chinese_test.tar.gz`、`raw/chinese_train.tar.gz`、`raw/french_test.tar.gz`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：中文、阿拉伯语、法语、西班牙语
- 标注与格式：压缩包样本为 JSON 与视频编号 TXT；JSON 可用于后续切片和文字关联。
- 训练/测试划分：四种语言训练与测试 tar 元数据；对应原始视频未由本发行提供。
- 冷端 20 文件，共 6,971,436,241 字节，含四种语言各训练/测试的 8 个 tar.gz，以及 12 份视频 ID 表。
- 有界读取中文和阿拉伯语测试包，确认内部是 JSON 和视频编号，并非压缩视频。中文训练包开头含 1.57 GB 的 JSON 成员。
- 官方说明要求用户先根据视频列表下载媒体，说明当前发行清单和文字不能当作完整视听媒体。
- 文件名里的 cz 在这一发行中对应 Chinese 目录，不能按常见语言代码误认成捷克语。
- 既有下一步：中文及其他非英语扩展的重要候选，但当前应标成缺媒体；不把发布页总时长计入现成可训数据。

## 对 Falcon 的准备事项

- 7 GB 大小主要来自详细 JSON，不能按文件大就推断媒体已准备。
- 没有全流展开训练包，2026-10-16 盘点不报告本地已验收样本小时数。
- 后续获取媒体时需保留失效视频的缺失记录，并检查与现有 LRS3、MuAViC、AVSpeech 的重叠。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

`raw/` 实际有 20 个文件、6,971,436,241 字节，其中 8 个语言/划分 tar.gz、12 个视频 ID 文本；压缩包文件数不能当作已具备媒体的语音条数。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
