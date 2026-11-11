# CUAVE group 派生包：双人灰度人脸与音频特征（`cuave_partial`）

**用途定位：B｜双人重叠专项。** MAT 含双人画面、按帧索引音频及说话状态；先核对目标个人参考与音频可逆格式，不能仅靠活动标签训练分离。

## 当前目录、状态与许可

- 数据说明：CUAVE group sequences (partial)
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/cuave_partial/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/cuave_partial/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/cuave_partial/SHA256SUMS`。
- 机器记录：登记预期对象 1；状态已完成对象 1；状态冷端对象 1；冷端收据对象 1；冷端收据字节 640,156,002。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`not stated`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://people.csail.mit.edu/siracusa/avdata/](https://people.csail.mit.edu/siracusa/avdata/)。
- 当前清单开头的对象示例：`raw/cuave-group-aligned.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语，数字和有限句式
- 标注与格式：官方页面称 MAT 内含 labels；该标注主要涉及两人的说话状态，不能据此认定是词序列。
- 训练/测试划分：本地为 group 部分派生包；说话人隔离和训练测试划分尚未确认。
- 冷端 cuave-group-aligned.zip 有 22 个 MAT 文件，大小合计 640,156,002 字节，属于 CUAVE 的 group 部分派生包。
- 来源页说明每个 MAT 含两人的 75×50 灰度帧、按帧索引的音频及 MFCC，还包含两人同时说话时的状态标签。
- 使用现有 Conda 读取 g01_aligned.mat 的字段：video 是 1×2 元胞数组，audioIndexed 为 534×957，MFCC 为 13×957，labels 为 1×957 的数值数组，另有帧号、帧率及采样率；未出现逐字文本字段。
- 暂存 recovery 保留约 461 MB 的失败分片；完整包位于冷端。
- 既有下一步：若需监督识别，应先读取 MAT 字段并取得准确句子标注；现阶段不并入主训练清单。

## 对 Falcon 的准备事项

- 这是部分 group 序列，不是完整 CUAVE 原始发行。
- 2026-10-16 盘点只读取一个代表 MAT 的字段说明及 labels 数组，没有展开全部视频矩阵；状态值与说话人的完整对应仍需核对来源定义。
- 未在当前目录确认逐字句子清单。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
