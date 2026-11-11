# AudioSet（`audioset`）

**用途定位：C｜干扰/噪声源。** 音频事件可用于环境声增强；先定位联合包实际内容、来源版本和 train/test，事件标签不能直接当目标语音。

## 当前目录、状态与许可

- 数据说明：YouTube 音频事件数据及标签。
- 登记状态：`existing_cold_reference`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/audioset/`；状态文件：该目录下 `status.json`。
- 冷端共用归档：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`；登记名称是关联线索，包名不证明该库每条数据可用。
- 本地许可记录：`dataset-specific terms`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://research.google.com/audioset/](https://research.google.com/audioset/)。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：多语言；事件标签不代表转写语言
- 标注与格式：音频事件类别；不是逐字语音转写
- 训练/测试划分：事件音频 train/eval 的来源版本需确认，混音噪声也应按源隔离。
- 本目录只有状态与说明，raw 为空，引用联合冷端大包。包名和注册记录写明 AudioSet；2026-10-16 盘点包头抽样没有走到它的内部位置，不能确认 AudioSet 的具体版本、文件数、音视频配对与有效率。

## 对 Falcon 的准备事项

- 先定位联合包中 AudioSet 的实际目录或现有解包结果，明确 balanced/unbalanced/eval 等发行范围；训练用噪声与验证噪声应分开。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
