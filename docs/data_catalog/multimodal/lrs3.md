# LRS3（`lrs3`）

**用途定位：A｜合成混音候选。** 从有音轨的已配对原始语音和嘴图构造混音，保留源验证/测试；需核对其他 LRS3 镜像和派生数据以免重复。

## 当前目录、状态与许可

- 数据说明：TED/TEDx 野外英语句子级唇读/视听语音数据。
- 登记状态：`existing_cold_reference`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/lrs3/`；状态文件：该目录下 `status.json`。
- 冷端共用归档：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`；登记名称是关联线索，包名不证明该库每条数据可用。
- 本地许可记录：`official access terms`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：官方句子转写；现有派生目录另有已完成训练和评测的清单
- 训练/测试划分：现有派生训练清单与官方验证/测试需按源ID对齐；与 AVYT 和 MuAViC 英语部分去重。
- 本目录 raw 为空，状态 existing_cold_reference，指向联合大包；训练实际使用的已处理媒体与清单位于 nested multimodal/audio-visual/derived，另作专门审计，不能把此占位目录误当缺失数据。

## 对 Falcon 的准备事项

- 以派生数据章节报告的正式训练清单、测试清单和最近实测结果为准；2026-10-16 盘点未重复遍历该目录，也未重新核实官方原始发行全量一致性，不能据已完成训练推断联合包所有条目完整。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
