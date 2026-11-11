# Lombard GRID（`lombard_grid`）

**用途定位：B｜噪声/侧脸专项。** 有独立音频、正面和侧面视频，可研究噪声下发音；需按标识核对逐条配对，两视角同一录音不能当双倍独立语句。

## 当前目录、状态与许可

- 数据说明：54 名说话人、正常/噪声条件、正面与侧面双视角。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/lombard_grid/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/lombard_grid/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/lombard_grid/SHA256SUMS`。
- 机器记录：登记预期对象 5；状态已核验对象 5；状态核验字节 2,485,068,233；冷端收据对象 5；冷端收据字节 2,485,068,233。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`cc-by-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://zenodo.org/records/3736465](https://zenodo.org/records/3736465)。
- 当前清单开头的对象示例：`raw/lombardgrid_alignment.zip`、`raw/lombardgrid_audio.zip`、`raw/lombardgrid_front.zip`、`raw/lombardgrid_json.zip`、`raw/lombardgrid_side.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：句子编码、每人JSON、音素起点/时长JSON；正面/侧面视频与独立音频
- 训练/测试划分：54人双视角，训练/测试划分需按人和原录音显式建立。
- 冷端完整5包：alignment、audio、front、json、side。音频5390个WAV，正面和侧面各5390个MOV，每人JSON54个，音素对齐JSON5361个。

## 对 Falcon 的准备事项

- 统计得到 5,390 个音频文件和 5,361 个音素对齐文件，数量相差 29；具体缺失项需按标识核对。样本JSON中的 TRANSCRIPTION 可为“-”，需要结合句子编码、STATUS与官方词表生成文字，不能把“-”当标签。按说话人和录音成组划分；使用音素目标时只取已对齐交集。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

## 2026-11-11 冷端归档直查

直接读取冷端音频与正面视频 ZIP 的中央目录：各 5,390 个 WAV/MOV，不带扩展名的文件名各不重复，名称集合匹配 5,390，双方独有为 0。该结果不等于已逐对检查音视频同步和内容。 [逐项统计与执行范围](../cold-zip-pair-audit-20261111.json)；[只读复核脚本](../cold_zip_pair_audit.py)。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
