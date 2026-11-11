# AVCocktail（`avcocktail`）

**用途定位：B｜外部场景评测。** 51 个源视频有三套切片及文字，可检验目标可见和真实干扰；是否有逐目标干净源需另核，不能直接假定可算 SI-SNR。

## 当前目录、状态与许可

- 数据说明：多人、噪声和重叠语音条件的鸡尾酒会 AVSR 挑战集。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/avcocktail/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/avcocktail/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/avcocktail/SHA256SUMS`。
- 机器记录：登记预期对象 206；状态已核验对象 206；状态核验字节 3,438,836,255；冷端收据对象 206；冷端收据字节 3,438,836,255。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`cc-by-sa-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/nguyenvulebinh/AVCocktail](https://huggingface.co/datasets/nguyenvulebinh/AVCocktail)。
- 当前清单开头的对象示例：`raw/.gitattributes`、`raw/README.md`、`raw/labels/video_0/label-eval-0.tar`、`raw/labels/video_1/label-eval-0.tar`、`raw/labels/video_10/label-eval-0.tar`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语（代表标签）；未逐条统计
- 标注与格式：独立 label tar + 视频片段开始/结束时间、sample_id；按说话人检测、固定和标准切段三种方式发布
- 训练/测试划分：video_0～video_50 为评测来源，每个有ASD/fixed/gold三种分段；按源视频隔离。
- 冷端 206 个发行文件，包含 51 个标签 tar、153 个视频 tar 与 2 个仓库文件；后者是 51 个源视频的三种切段，不是 153 个独立场景。代表片段确有 96×96、25 帧/秒视频和 16kHz 单声道音频。

## 2026-11-11 标签与首个分段实查

逐一读取 **51 个标签 tar 的首成员**，51 个都是 WebVTT；这 51 个字幕成员合计记录 1,578 个 cue。另检查 153 个视频 tar 的首个分段头部，`asd`、`fixed`、`gold` 各 51 个包；每种是该来源的不同切分，不能将 153 个包视作 153 个独立来源。这次没有遍历全部分段成员。

示例 `video_0/asd-chunk-eval-0.tar` 第一个分段为 `seg-0_344_712`，记录起止 13.76–28.48 秒；视频为 96×96、25 fps、369 帧、14.760 秒，音频为 16 kHz 单声道 AAC、14.720 秒。首帧和首秒解码成功；对应 label tar 的首成员是 WebVTT，包含 69 个 cue。字幕用于文字或时间对照，不等同于同步干净目标音频；有参考提取指标仍需另找目标参考。

[逐包首成员和分段头核验](../avcocktail-headers-probe-20261111.json)；[示例片段解码](../archive-media-probe-20261111.json)。

## 对 Falcon 的准备事项

- 需写读取器，把独立标签包与各片段的时间窗、目标轨迹对应；核对重叠期间应输出哪个人的文本，不能直接复用单人整段识别标签。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
