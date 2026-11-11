# MLD-VC：中文和英语视频会议识别数据（`mld_vc`）

**用途定位：B｜会议域候选。** 现有音视频可合成重叠，先修正 metadata 复数路径映射并核对句子与说话人划分；许可未声明。

## 当前目录、状态与许可

- 数据说明：MLD-VC video-conferencing AVSR
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/mld_vc/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/mld_vc/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/mld_vc/SHA256SUMS`。
- 机器记录：登记预期对象 5；状态已完成对象 5；状态冷端对象 5；冷端收据对象 5；冷端收据字节 9,914,542,126。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`not declared`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/nccm2p2/MLD-VC](https://huggingface.co/datasets/nccm2p2/MLD-VC)。
- 当前清单开头的对象示例：`raw/audio.zip`、`raw/landmarks.zip`、`raw/metadata.jsonl`、`raw/text.zip`、`raw/video.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：普通话、英语
- 标注与格式：metadata.jsonl 有音视频及人脸关键点路径；text.zip 按说话人保存 30 句中英文本。
- 训练/测试划分：20,803 条媒体实例含多个会议平台/录音条件；按源人、句子及平台建立训练测试。
- 冷端 5 个文件，共 9,914,542,126 字节；实际 audio.zip 包含 20,803 个 WAV，video.zip 包含 20,803 个 MP4。
- metadata.jsonl 正好 20,803 行，涉及 31 人；离线 6,236 行，在线 14,567 行，覆盖 DD、FS、TXHY、zoom 四类平台目录。
- text.zip 去除 Mac 系统附带项后，有 31 个真正的文字文件，各含编号、表头和 30 条句子；文字包括中文及英语。
- metadata 使用 videos/ 与 audios/ 复数路径，压缩包内却是 video/ 与 audio/ 单数。全量核对 20,803 行：原路径全部缺失，修正前缀后音视频缺失均归零；说话人文字文件也全部覆盖。
- 代表视频为 512×512、25 帧/秒，音频为 16 kHz、单声道。视频包中额外有 .metadata 和 .lock，不应把这些当作更多视频。
- 既有下一步：适合视频会议场景适配和跨平台测试；先修正路径并生成逐人逐句文字关联表，再建立说话人隔离的训练/评测划分。

## 2026-11-11 全量索引与代表媒体

逐行读取 `metadata.jsonl` 得到 **20,803 条、31 个 speaker ID**；其中 Online 14,567 条、Offline 6,236 条，四个 `platform` 标签分别为 DD 7,192、FS 4,560、TXHY 4,559、zoom 4,492。`noise_level` 字段的 40db、60db、80db 各 5,201 条，clean 5,200 条。元数据没有 `split` 字段，训练、验证、测试划分需要另外制定并按说话人/原始 utterance 防止泄漏。

将所有音频、视频和 landmarks 路径分别映射到 `audio.zip`、`video.zip`、`landmarks.zip` 中央目录，三类各 **20,803 条唯一引用、缺失 0**。`video.zip` 中除 20,803 个 MP4 外还有等量 `.lock` 和 `.metadata` 条目，它们不是额外视频。此检查是成员名匹配，不是 20,803 条媒体全量解码或时钟校验。

抽一条 `Online/speaker_21/TXHY/9`：`60db` 音频为 16 kHz 单声道 4.288 秒，视频为 512×512、25 fps、4.360 秒；同一目录 `clean` 音频为 3.328 秒、视频 3.400 秒。四个文件首秒/首帧解码通过。两个条件录制时长不同，不能直接将 `clean.wav` 当作 `60db.wav` 的逐采样同步目标；需核对起止时间、内容和身份后再配对。

[全量元数据与归档成员核验](../mld-vc-index-probe-20261111.json)；[代表样本解码](../archive-media-probe-20261111.json)。

## 对 Falcon 的准备事项

- 20,803 是包含说话人、平台和录音条件的媒体实例数，不是 20,803 条不同句子。
- 同句不同平台和不同噪声条件高度相关，划分应按说话人和录制来源，避免随机逐文件拆分。
- 还需确认 utt=0…29 与文字行的准确对应、离线重复录制的关系；当前未完成中文模型接入验收。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

## 2026-11-11 冷端归档直查

直接读取冷端音频与视频 ZIP 的中央目录：WAV 和 MP4 各 20,803 个；去掉最外层 audio/video 目录和扩展名后，20,803 个相对路径键一一对应，双方独有和重复键均为 0。单用文件名会因 clean、40db 等名字复用而错误合并；此结果未逐对验证同步。 [逐项统计与执行范围](../cold-zip-pair-audit-20261111.json)；[只读复核脚本](../cold_zip_pair_audit.py)。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
