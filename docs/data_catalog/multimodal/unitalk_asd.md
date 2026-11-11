# UniTalk-ASD：是否正在说话的音视频标注（`unitalk_asd`）

**用途定位：C｜活动检测辅助。** 有脸轨、音频和说话/未说话标签，可用于筛选可见目标或验证拒识；标签不是目标干净音频。

## 当前目录、状态与许可

- 数据说明：UniTalk-ASD active speaker detection
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/unitalk_asd/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/unitalk_asd/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/unitalk_asd/SHA256SUMS`。
- 机器记录：登记预期对象 353；状态已完成对象 353；状态冷端对象 353；冷端收据对象 353；冷端收据字节 14,555,424,127。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`MIT`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD)。
- 当前清单开头的对象示例：`raw/clips_audios/train/-ASZexdSdWE.zip`、`raw/clips_audios/train/-maoolmiKi0.zip`、`raw/clips_audios/train/1JGpByLjUak.zip`、`raw/clips_audios/train/1hJn0KZ1cs8.zip`、`raw/clips_audios/train/1xrAHh2fCk8.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：需以来源及片段识别为准；本轮不推断全部语言
- 标注与格式：CSV 标签是 SPEAKING_AUDIBLE 与 NOT_SPEAKING，并不是说了什么。
- 训练/测试划分：有 train/val CSV、按脸轨及视频分包；ASD 活动帧不能直接等于可分离语句。
- 冷端 353 个文件，共 14,555,424,127 字节，包含成包音频、按脸轨分组的 JPG 帧和 CSV。
- 代表音频 ZIP 有 244 个 16 kHz 单声道 WAV；对应画面 ZIP 有 16,818 个 JPG。
- 代表 CSV 没有表头，实际 16,818 行，涉及 244 个脸轨编号；其中 2,864 行为可听说话，13,954 行为未说话。
- csv/train 与 csv/val 是来源拆分；不能把每帧标签行当作独立句子。
- 既有下一步：用于训练或检验目标脸是否正在发声，再帮助筛选其他数据；若用于识别训练，须另行补齐转写。

## 对 Falcon 的准备事项

- 即使音视频已经存在，活动标签也不能替代逐字转写。
- 口型与音轨配对需要按脸轨和时间戳处理；不能仅凭同一视频 ID 就认定全程是目标说话人的语音。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

递归遍历 `raw/` 得 353 文件、14,555,424,127 字节，其中 234 个 ZIP、119 个 CSV；活动标签、音频和脸轨需按时间及来源 ID 配对。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
