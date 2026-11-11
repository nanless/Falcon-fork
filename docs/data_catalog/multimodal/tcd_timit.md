# TCD-TIMIT（`tcd_timit`）

**用途定位：B｜受控源训练/测试。** 优先采用官方说话人独立划分，利用独立干净语音、双视角视频构造混音；62人的双视角 ZIP 不是124个独立人。

## 当前目录、状态与许可

- 数据说明：受控场景高清双视角连续语音视听语料。
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/tcd_timit/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/tcd_timit/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/tcd_timit/SHA256SUMS`。
- 机器记录：登记预期对象 137；状态已完成对象 137；冷端收据对象 137；冷端收据字节 676,948,863,296。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`authenticated academic access; source terms apply`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://sigmedia.tcd.ie/tcd_timit_db/](https://sigmedia.tcd.ie/tcd_timit_db/)；[https://sigmedia.tcd.ie/TCDTIMIT/](https://sigmedia.tcd.ie/TCDTIMIT/)。
- 当前清单开头的对象示例：`raw/Additional Resources/EGillen MSc Thesis/EGillen_thesis.pdf`、`raw/Additional Resources/EGillen MSc Thesis/Thumbs.db`、`raw/Additional Resources/Modified CMU Pronouncing Dictionary/CMUdict_with_dublin`、`raw/Additional Resources/Recommended Train-Test Splits/Speaker-Dependent Baseline Sets/TCDSpkrDepTestSet.scp`、`raw/Additional Resources/Recommended Train-Test Splits/Speaker-Dependent Baseline Sets/TCDSpkrDepTrainSet.scp`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语，含爱尔兰口音
- 标注与格式：每句MP4/WAV/TXT/嘴部MAT；MLF音素强制对齐；官方说话人划分SCP
- 训练/测试划分：冻结清单为62说话人×2视角ZIP及13辅助对象；官方同时提供说话人相关和说话人独立两组基线 SCP，按具体实验选择并记录；当前未统计语句。
- **2026-11-11 当前证据**：登记 137/137 对象，冷端逐文件归档收据 137 项、676,948,863,296 字节。62 人双视角 124 ZIP 与 13 个辅助对象；旧 2026-10-16 的 130/137 已过时。

## 对 Falcon 的准备事项

- 建立每人正面/侧面与音轨的配对表。不同相机共享同一句录音，不计双倍语料；来源为账号授权非商业研究。历史记录中的 130/137 已由 2026-10-22 的 137/137 完成记录取代。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

递归遍历 `raw/` 得 137 文件、676,948,863,296 字节，其中 124 个 ZIP。说话人相关 SCP 训练／测试分别 3,752／1,736 行；说话人独立 SCP 训练／测试为 3,822／1,666 行。行数为协议清单项，不是本次解码样本数。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
