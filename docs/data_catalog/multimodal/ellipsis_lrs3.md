# ellipsis LRS3 trainval：口型数组与文字派生包（`ellipsis_lrs3`）

**用途定位：C｜视觉派生数据。** 抽查成员有 roi.npy 与文字但未见音频，且来源字段与目录名称有差异；先核源ID，再尝试连接对应音轨。

## 当前目录、状态与许可

- 数据说明：LRS3 trainval Auto-AVSR format
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/ellipsis_lrs3/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/ellipsis_lrs3/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/ellipsis_lrs3/SHA256SUMS`。
- 机器记录：登记预期对象 34；状态已完成对象 34；状态冷端对象 34；冷端收据对象 34；冷端收据字节 25,041,170,491。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-trainval-autoavsr](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-trainval-autoavsr)。
- 当前清单开头的对象示例：`raw/shards/bridge-report.json`、`raw/shards/dataset-manifest.jsonl`、`raw/shards/shard-000000.tar`、`raw/shards/shard-000001.tar`、`raw/shards/shard-000002.tar`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：代表分片按样本保存 roi.npy、txt 和 json；未见音频成员。
- 训练/测试划分：标注宣称LRS3 trainval，样本来源字段待核，缺音频；在核清身份与划分前不混入主训练。
- 冷端是 32 个未压缩 TAR 分片及 2 个清单/处理报告，共 25,041,170,491 字节。
- 处理报告写输入 31,982 条、成功 31,957 条，另有 25 条因未提取到口型而丢弃；这是发布者报告中的计数，尚非2026-10-16 盘点逐片复算。
- 代表样本的灰度口型形状为 [帧数,96,96]，25 帧/秒，有英语句子文字。
- 目录和发布页称 LRS3 trainval，但代表样本 JSON 写 source=voxpopuli，并含 vp- 开头的说话人编号；这与页面身份描述存在差异。
- 既有下一步：先核对源镜像、处理程序和 ID 与现有 LRS3 的关系；未解决来源及重复前，不放入正式合并训练。

## 对 Falcon 的准备事项

- 不能只凭 source 字段就判定真正来自 VoxPopuli，也不能忽略该差异直接宣称来源已确认。
- 2026-10-16 盘点只读首个分片的有限成员；没有把 31,957 条当成新增独立 LRS3 数据。
- 缺独立音频时，口型数组加文字只能直接支持纯视觉识别；AVSR 需要与已有 LRS3 音频准确连接。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
