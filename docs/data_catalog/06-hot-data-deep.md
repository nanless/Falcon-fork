# 共享盘实际驻留的视听数据：深入核对

检查时间：2026-11-11。本文关注能从服务器共享盘直接读取的媒体和清单。`/root/group-shared/voiceprint/data/multimodal` 的真实路径是 `/shared/group-shared/voiceprint/data/multimodal`，`findmnt` 显示所属挂载是 GPFS；这批共享盘文件与 `/aispeech-cfs` 的冷端挂载是不同的存储位置。完整来源、冷端相关归档和其他登记项见[数据集总览](02-multimodal-datasets.md)。

## 1. 现存数据的入口

| 项目 | 服务器实际入口 | 当前看到的内容 | 对 Falcon 的作用 |
| --- | --- | --- | --- |
| LRS3 原媒体 | `multimodal/audio-visual/preprocessed_data/lrs3/{pretrain,trainval,test}` | 三个真实目录；历史索引分别记录 118,516／31,982／1,321 个 MP4 及文字 | 可选说话人音视频，再提取与音频时间对应的嘴图、构造干扰和混音 |
| LRS3 处理媒体 | `multimodal/audio-visual/derived/avsr_reproduction/mms_lrs3_24s/` | 视频、WAV、标签，训练和测试 TSV/WRD 文件另在 `manifests/` | 已有 96×96 视频与 16 kHz 音频的样本；仍要构造与目标对应的混音和 Falcon JSON |
| VoxCeleb2 原媒体与派生 | `multimodal/audio-visual/preprocessed_data/voxceleb2/` 及 `derived/avsr_reproduction/mms_vox2_1759h_v1/` | `dev`、`test`、元数据和多阶段预处理目录；见下方清单 | 按原说话人、视频 ID 去重划分，确认画面与目标声音，构造混音 |
| AVSpeech 本地遗留原片 | `avspeech/videos/legacy_unmatched/` | **229,520 个 MP4**，本次遍历按扩展名计数、无遍历错误 | 可挑选素材；文件名时间不必等于实际容器时长，重新查音轨、时间与说话人可见性 |
| AVSpeech 处理阶段 | `derived/avsr_reproduction/avspeech_scaled_v1/{conformed,preprocessed,labels,lists}` | 分阶段处理媒体、伪标签与缺失清单；`manifest` 和 `sr` 目录当前为空 | 源、重编码、嘴图和文本是关联产物，不应重复当作独立素材；须建立 Falcon 混音和目标配对 |
| 旧 LRS2／Vox2 重建 | `preprocessed_data/{lrs2_rebuild,vox2_rebuild}` | 音频、嘴图、面部、关键点、混音 SCP 及处理脚本 | 可参考旧混音与预处理方式；应校验绝对路径、采样率、同目标嘴图和划分 |
| 迁往冷端的派生集 | `derived/avsr_reproduction/{avspeech_pilot_v1,muavic_wf_v1,muavic_lrs3_build_v1}` | 本地目录仅见 `MOVED_TO_COLD.md` | 需要时按各标记和冷端清单恢复，不能把标记文件当成媒体 |

表中 `multimodal/audio-visual/` 是相对共享盘根 `/root/group-shared/voiceprint/data/multimodal/` 的路径。历史数字表示此前的目录/清单统计，本次对下列重点清单做了全行重算；目录仍需按实际实验的索引检查。

## 2. 两套实际 TSV 的全行统计

两套清单均为六列 TSV，首行是根路径 `/`；后面每行包含来源标识、视频路径、WAV 路径、帧数、音频采样数和第六个数值字段。其同名 `.wrd` 提供文本行。新流式 Falcon 并不直接读取这种 TSV/WRD，也不以文本作为分离监督。

| 清单 | 划分 | 数据行 | 来源分组 | 累计音频时长 | 文本行 |
| --- | --- | ---: | --- | ---: | ---: |
| `mms_llama_433h_nocap_srlength_v6_20261004` | train | 164,916 | 全部 LRS3 | 436.312 小时 | 164,916 |
| 同上 | test | 1,321 | 全部 LRS3 | 0.851 小时 | 1,321 |
| `mms_llama_1759h_v6_datarepair_20261004` | train | 820,468 | LRS3 164,916；Vox2 655,552 | 1,758.012 小时 | 820,468 |
| 同上 | test | 1,321 | 全部 LRS3 | 0.851 小时 | 1,321 |

时长按 TSV 音频采样点数除以 16,000 计算；TSV 首行不是数据。两套 LRS3 数据和 test 行会重叠，不能把表格各行简单相加。两组训练清单分别每 20,000 行抽取一个索引，9／9 与 42／42 个抽查位置的音频与视频路径都存在；测试各每 100 行抽查，14／14 条路径存在。这是路径抽查，不代表逐文件解码或同步验证。

LRS3 首条 train 样本是 H.264、96×96、25 fps、37 帧（1.480 秒），对应 PCM float32、16 kHz 单声道 WAV、24,576 点（1.536 秒），视频首帧实际解码通过；两者总长度差 **56 ms**。首条 test 样本同样有 56 ms 长度差。Vox2 首条 train 样本是 96×96、25 fps、236 帧（9.440 秒），WAV 150,528 点（9.408 秒），长度差 32 ms，视频首帧也解码通过。长度不同不能单独证明起始时间错位；做 Falcon 的 160 ms 分块前仍应检查源时间和可用帧。

按清单的 `音频采样数/16000` 与 `视频帧数/25` 计算，LRS3 训练清单有 **35,094／164,916** 行的绝对时长差超过 40 ms；组合训练清单有 **35,096／820,468** 行超过 40 ms，两套 test 清单各 **324／1,321** 行超过 40 ms。阈值及统计代码见 [机器证据](hot-manifest-audit-20261111-v2.json) 与 [复核脚本](audit_hot_manifests.py)。这些数字依据清单元数据，不等同于全体媒体头重新测量，也不能由此确定画面和音频起点的偏移。

组合训练清单的 Vox2 共 655,552 行，按**当前物理路径字符串**可识别 625,683 行来自 `preprocessed_dev_seg16`、29,500 行来自 `preprocessed_test_seg16`、369 行来自修复或其他路径；369 行不能只按目录名简单归入 dev 或 test。尤其 `preprocessed_test_seg16` 的素材已出现在该训练清单中，如果为 Falcon 设计独立测试集，需要先按源视频和说话人核对使用历史。

## 3. AVSpeech 本地原片与链接

本次递归读取 `legacy_unmatched/` 的文件名称与类型，得到 **229,520 个 MP4**、遍历错误 0。首三个取到的样本均由 `ffprobe` 识别为 H.264 1280×720 视频与 AAC 44.1 kHz 双声道音轨；三条不是全库统计。其中一个文件名标记区间长度 3.000 秒，容器时长为 **7.686 秒**，说明不能仅从名字把它当作对应时长的精确切片；其原因及与目标说话人的对齐关系需继续核查。

`avspeech_scaled_v1/source_flat/` 本次全目录计 **200,638 个符号链接**，抽查前 20 个目标有 **20 个不存在**；所见目标指向已经迁到冷端的 `avspeech/videos/test/` 路径。这说明该入口抽样不能直接读取原片；200,638 个链接的目标未在本次全部逐个 `stat`。`conformed/` 与 `preprocessed/` 是另行存放的处理阶段，并不因 `source_flat` 抽样悬空而被认定全部不可读。

`avspeech_scaled_v1/lists/` 现有原始 ID、相对 ID 与 missing 清单；`labels/` 有多个 JSONL 伪标签及日志；`manifest/`、`sr/` 当前为空。历史盘点的大小和完整阶段计数见[派生数据说明](multimodal/derived.md)，不能把目录或文字存在等同于 Falcon 的监督清单。

## 4. 接入 Falcon 的次序

1. 从清单抽取**目标说话人**对应音频和嘴部画面，确认音视频源时间、帧率及目标发声；文字用于检索或识别评估，不替代目标声源。
2. 固定源视频、说话人和会话的训练／验证／测试划分，检查 LRS3、Vox2、AVSpeech、AVYT 与项目现有 LRS2 的重叠；已有 MMS 清单的 `test` 可能参与过别的模型训练。
3. 用独立说话人音频与噪声按记录的增益、随机种子和时间构造混音；保留同步目标音频、干扰身份和嘴图。连续说话视频不能仅因一人可见就自动视为无其他声音的干净目标。
4. 输出 Falcon 的 `mix.json` 与 `target.json`，保留逐条来源、起止和变换版本；16 kHz 单声道、25 fps 嘴图要满足所选入口的尺寸和格式。先抽查音画、训练一小批并独立测试，再决定扩大范围。

本次检查没有移动、删除或大规模复制媒体。结果以 [清单逐项 JSON](hot-manifest-audit-20261111-v2.json) 和本文列出的路径、抽样范围为准。

[返回数据全景](README.md)
