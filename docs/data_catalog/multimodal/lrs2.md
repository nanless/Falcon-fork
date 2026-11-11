# LRS2（`lrs2`）

**用途定位：A｜项目基线源。** 原始源与已构造的 LRS2-2Mix 必须按源 ID 去重；联合冷包的总量不能直接作为 LRS2 数量。项目现用清单及评测数字见 ../01-project-data.md。

## 当前目录、状态与许可

- 数据说明：BBC 野外英语句子级唇读/视听语音数据。
- 登记状态：`existing_cold_reference`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/lrs2/`；状态文件：该目录下 `status.json`。
- 冷端共用归档：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`；登记名称是关联线索，包名不证明该库每条数据可用。
- 本地许可记录：`official access terms`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：官方发行包含句子文字和时间信息；当前本目录尚未直接读到联合包内部的对应标注
- 训练/测试划分：项目当前训练/验证/测试实际索引参见项目数据文档；联合包的原始切分需单独核验。
- 本目录 raw 为空，状态为 existing_cold_reference，引用 AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar。包头抽样确见 lrs2_rebuild/audio/wav8k/min/tr/mix 等派生混合音频目录，但这不足以证明原始LRS2发行全量齐全。

## 对 Falcon 的准备事项

- 优先交叉核对 nested audio-visual 中的已解包LRS2和训练用清单；本条引用目录不单独贡献媒体文件数或容量。联合大包不是已经就绪的标准AVSR训练目录。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
