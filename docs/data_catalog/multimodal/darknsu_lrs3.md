# Darknsu LRS3 镜像：已裁口型视频和音频（`darknsu_lrs3`）

**用途定位：B｜LRS3镜像替代。** 有口型视频和音频，可用于补缺或构造混音；需按ID还原原数据拆分及使用条款，与 LRS3 去重。

## 当前目录、状态与许可

- 数据说明：LRS3 all video + wav
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/darknsu_lrs3/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/darknsu_lrs3/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/darknsu_lrs3/SHA256SUMS`。
- 机器记录：登记预期对象 2；状态已完成对象 2；状态冷端对象 2；冷端收据对象 2；冷端收据字节 62,975,957,463。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`Apache-2.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/Darknsu/lrs3_all_video_wav](https://huggingface.co/datasets/Darknsu/lrs3_all_video_wav)。
- 当前清单开头的对象示例：`raw/video.tar.gz`、`raw/wav.tar.gz`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：顶层仅 video.tar.gz 与 wav.tar.gz，未发现独立文本清单。
- 训练/测试划分：以 LRS3 原始划分和视频ID为准，第三方镜像不可直接视为独立新增样本。
- 冷端 2 个 gzip 压缩包，共 62,975,957,463 字节。
- 有界读取开头的媒体表明，视频为 96×96、25 帧/秒，音频为 16 kHz、单声道；样本共享 LRS3 式视频 ID。
- 部分文件名带 _0、_1 等二次分段编号，不能直接假定与原始 LRS3 文件名逐条相同。
- 既有下一步：只在原有口型或音频缺失时作为候选替代来源，先完成 ID、文字及原始划分核对。

## 对 Falcon 的准备事项

- 2026-10-16 盘点没有遍历两个大压缩流的全部成员，不报告全量片段数。
- 来源是第三方镜像，不是 LRS3 作者原发行；状态里的许可证字段不能替代原数据许可。
- 不能与已有 LRS3 原始和 derived 处理结果重复计算训练规模；必须排除测试和验证样本。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
