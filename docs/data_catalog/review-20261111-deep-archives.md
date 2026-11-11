# 六个优先数据集的归档深查

本次直接从 `/aispeech-cfs/voiceprint/data/multimodal` 读取索引、压缩包目录和少量媒体；小样本通过任务专用临时目录解码，检查后自动清理。数据源、成员名称、文件大小和样本哈希见下方 JSON；全量读取索引或 ZIP 中央目录与抽样解码是不同范围。

| 数据集 | 直接检查及数量 | 对 Falcon 的意义 |
| --- | --- | --- |
| [Mandarin AVSR](multimodal/mandarin_avsr.md) | 训练的混音、目标索引各 40,000 行；各人数组 8,000。训练、验证、测试包各读取前 10 个成员，看到相应 `s1` 音频；已有单个预览视频首帧解码 | 三类配对接近目标格式；完整媒体、嘴部和权限仍须按实验检查。前 10 个成员不能用于推算整套划分数量 |
| [ASPIRE](multimodal/aspire.md) | ZIP 成员有 5,940 WAV 与 5,940 MOV 同 stem；实际解码一对：44.1 kHz 双声道 3.000 秒音频、1080p/30 fps 视频轨 3.033 秒 | 有目标音视频候选，需要重采样、下混、嘴部提取及真实时钟核对 |
| [Chinese-LiPS](multimodal/chinese_lips.md) | 三划分 ZIP 的 30,341／1,959／3,908 个同 stem 音视频；实际解码测试集一对：16 kHz 音频 12.921 秒、25 fps 视频 13.000 秒 | 可按原始划分准备中文目标并构造混音；该单例时长差不能直接解释为起始错位 |
| [MLD-VC](multimodal/mld_vc.md) | 20,803 行、31 个说话人；音频、视频、关键点引用在各自 ZIP 中央目录均缺失 0。代表 utterance 的 `60db` 与 `clean` 音频分别 4.288／3.328 秒 | 三种归档的名称可对应；不同条件文件不应未经对齐就作为逐采样混音/目标配对 |
| [AVCocktail](multimodal/avcocktail.md) | 51 个标签 tar 的首成员均为 WebVTT；153 视频 tar 的首分段头分为三种方式各 51 个包。抽一段，音频和视频分别为 14.720／14.760 秒，成功解码 | 可用于真实场景或文本时间参考；字幕不是目标干净语音，三种切分不能算作独立源视频 |
| [VoxCeleb2-mix](multimodal/voxceleb2_mix.md) | 两组 14 个分片；首片各抽一对同键 WAV/MP4，音轨各 68,608 点，±200 ms 搜索中 lag=0、相关约 0.999999992 | 该单例音频对音频高度一致；仍需验证目标嘴部时间轴、所有分片及混音配方 |

证据：[代表媒体解码](archive-media-probe-20261111.json)、[MLD-VC 索引](mld-vc-index-probe-20261111.json)、[AVCocktail 归档头](avcocktail-headers-probe-20261111.json)、[Mandarin 归档前部](mandarin-first-headers-20261111.json)、[Vox 同键音轨](vox-first-pair-audio-sync-20261111.json)。各 JSON 旁有同名 Python 复核脚本；完整训练可用性还需对所选范围核对目标身份、帧时钟及许可。

[返回数据总览](README.md)
