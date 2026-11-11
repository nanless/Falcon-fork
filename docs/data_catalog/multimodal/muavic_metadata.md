# MuAViC Metadata（`muavic_metadata`）

**用途定位：C｜切片辅助元数据。** 说话人脸轨迹和 mean-face 需与 MuAViC 视频、对应音频组合，不能独立训练分离。

## 当前目录、状态与许可

- 数据说明：MuAViC 九语种官方人脸轨迹和 mean-face 元数据。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/muavic_metadata/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/muavic_metadata/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/muavic_metadata/SHA256SUMS`。
- 机器记录：登记预期对象 10；状态已核验对象 10；状态核验字节 15,182,727,260；冷端收据对象 10；冷端收据字节 15,182,727,260。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-NC-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://github.com/facebookresearch/muavic](https://github.com/facebookresearch/muavic)。
- 当前清单开头的对象示例：`raw/20words_mean_face.npy`、`raw/ar_metadata.tgz`、`raw/de_metadata.tgz`、`raw/el_metadata.tgz`、`raw/en_metadata.tgz`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：9语种
- 标注与格式：9个语言.tgz中的.pkl人脸轨迹及20words_mean_face.npy
- 训练/测试划分：语言 train/test 元数据辅助源视频切分；自身不是训练媒体。
- 冷端10个文件、15182727260字节，归档路径与清单一致。抽查de_metadata.tgz的前20成员确见语言/train、test目录和源视频ID.pkl；未执行任何pickle内容。

## 对 Falcon 的准备事项

- 需与MuAViC源视频、官方语音/文本和切分一同准备。不要直接加载不可信pickle；正式读取应使用项目已确认的可信发布物与受控转换。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
