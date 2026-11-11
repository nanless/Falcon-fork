# MaFi（`mafi`）

**用途定位：C｜纯视觉词级评测。** 抽查样本视频无音轨，可用于口型分支分析；单独使用无法提供 Falcon 音视频监督。

## 当前目录、状态与许可

- 数据说明：5 人、2,519 个静音单词视频的 VSR 测试集。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/mafi/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/mafi/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/mafi/SHA256SUMS`。
- 机器记录：登记预期对象 24；状态已完成对象 24；状态已核验对象 24；状态核验字节 5,128,929,084；冷端收据对象 24；冷端收据字节 5,128,929,084。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`unverified`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://osf.io/mna8j/](https://osf.io/mna8j/)。
- 当前清单开头的对象示例：`raw/Analysis/data_100words_American.csv`、`raw/Analysis/data_100words_AmericanSpeaker_AmericanParticipants.csv`、`raw/Analysis/data_100words_AmericanSpeaker_BritishParticipants.csv`、`raw/Analysis/data_100words_Australian.csv`、`raw/Analysis/data_100words_British.csv`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：文件名词标签、词可读性/感知规范CSV与行为统计
- 训练/测试划分：词级静音视频主要用于视觉测试；此次未确认 Falcon 分离训练/测试划分。
- 冷端24文件包含5视频ZIP、12CSV、1XLSX、5Python、1R；五视频ZIP经目录索引剔除伴生后分别816、121、680、315、587个MP4，合计2519；B2包另有1个XLSX。来源清单指向OSF mna8j。

## 对 Falcon 的准备事项

- status notes和registry还留有“没有可验证来源”的旧文案，与现有OSF清单和归档事实不一致；本报告保留矛盾并以实物描述，不改远端。许可字段仍 unverified，需核对OSF项目正式许可；不能用“已下载”替代用途授权。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
