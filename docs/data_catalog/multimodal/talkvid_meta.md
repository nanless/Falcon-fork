# TalkVid 元数据及小规模展示/评测媒体（`talkvid_meta`）

**用途定位：C｜元数据/小样素材。** 已有少量音视频和人物信息，非完整训练媒体；需确认目标音轨、视频及使用范围。

## 当前目录、状态与许可

- 数据说明：TalkVid metadata (URL-list only)
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/talkvid_meta/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/talkvid_meta/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/talkvid_meta/SHA256SUMS`。
- 机器记录：登记预期对象 951；状态已完成对象 951；状态冷端对象 951；冷端收据对象 951；冷端收据字节 632,073,719。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-NC-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/FreedomIntelligence/TalkVid](https://huggingface.co/datasets/FreedomIntelligence/TalkVid)。
- 当前清单开头的对象示例：`raw/TalkVid-bench/age/audios/videovideo-hOQAsNg8aw-scene3_scene3.m4a`、`raw/TalkVid-bench/age/audios/videovideo2482_4v01hRqzlQg-scene69_scene9.m4a`、`raw/TalkVid-bench/age/audios/videovideo2525_H3yZUfZV9UM-scene35_scene10.m4a`、`raw/TalkVid-bench/age/audios/videovideo2535_y3-nmMtxtYM-scene5_scene3.m4a`、`raw/TalkVid-bench/age/audios/videovideo4R1yqW6RFck-scene1_scene28.m4a`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：多语种，包含中文与英语
- 标注与格式：代表 JSON 是片段边界、人物及画质信息；画面描述不等于说话转写。
- 训练/测试划分：现有 bench 元数据和少量媒体不等于来源完整训练集；训练测试尚未完成关联。
- 冷端 951 个文件，共 632,073,719 字节，确有少量音视频，不应写成完全没有媒体。
- age/test_age.json 有 105 项，字段含音视频相对路径、起止时间、帧数、人物信息、质量分数；未见逐字转写字段。
- 发布页把 TalkVid 定位为语音驱动说话人视频生成，并区分大规模训练素材说明与 500 片段 TalkVid-Bench。
- 当前下载范围不是论文所述完整 1,244 小时媒体。
- 既有下一步：可作为多语种和画面多样性的候选索引；若另行获得并整理训练媒体，需要补转写、口型和同步验证。

## 对 Falcon 的准备事项

- 不能把大规模数据说明的小时数当作本地已落盘时长。
- bench 的年龄、语言等分类可能复用同一片段，需按唯一 ID 去重。
- 缺逐字文字和大规模原媒体时，不能直接计入有监督AVSR训练。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

递归遍历 `raw/` 得 951 文件、632,073,719 字节：643 MP4、305 M4A、3 JSON。文件数不自动等于按时间对齐的音视频配对数。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
