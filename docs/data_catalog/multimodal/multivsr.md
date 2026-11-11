# MultiVSR Full Videos（`multivsr`）

**用途定位：B｜多语言候选。** 先用已取得源视频和 metadata 求交，按原视频/说话人划分、裁目标脸并制作可确认干净目标；12,000小时是发行规模，不能当当前可用训练时长。

## 当前目录、状态与许可

- 数据说明：约 12,000 小时、13 种语言的原视频重建集。
- 登记状态：`downloading`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/multivsr/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/multivsr/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/multivsr/SHA256SUMS`。
- 机器记录：登记预期对象 124,416；状态已核验对象 75,998；状态核验字节 10,572,295,769,172；冷端收据对象 75,991；冷端收据字节 10,571,152,579,195。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`source-video terms vary`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://github.com/Sindhu-Hegde/multivsr](https://github.com/Sindhu-Hegde/multivsr)。
- 当前清单开头的对象示例：`raw/source_videos/---ZLAsraJY.mp4`、`raw/source_videos/--9Icp6IITs.mp4`、`raw/source_videos/--D6bQ0mY9Y.mp4`、`raw/source_videos/--F-B0HOeVY.mp4`、`raw/source_videos/--FWKJ7ME8Y.mp4`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：元数据列出13语言；本项未逐视频语言识别
- 标注与格式：本项为源MP4；转写、词时间戳与轨迹在multivsr_metadata
- 训练/测试划分：官方metadata规定原划分，已得视频与划分求交后才可报告各语种训练测试数量。
- **2026-11-11 状态快照**：正在重建，登记预期 124,416 源视频，状态已核验约 75,998；冷端收据时间与状态存在滚动差异。失败是请求/状态计数，不可直接当独立永久缺失数。

## 对 Falcon 的准备事项

- 必须按源视频ID生成成功/缺失交集，保留train/val/test，报告每语言、划分的有效句子与小时。failure_count不是唯一永久缺失总数，也不能加到成功数上充当覆盖。源平台条款与元数据MIT许可分别核对。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
