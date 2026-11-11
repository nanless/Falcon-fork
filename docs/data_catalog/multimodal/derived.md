# 嵌套视听派生数据：与 Falcon 的关系

统一前缀为服务器 `/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/`。2026-11-11 检查了下列路径；其中 MuAViC 两个热端目录只保留迁移标记。样本和时长数字来自 2026-10-16 的完整目录或清单盘点，属于历史口径。不同版本、重编码、分片和软链接与源媒体相互关联，不相加为独立数据。

| 处理链 | 路径（相对统一前缀） | 历史数量与用途 | Falcon 接入 |
| --- | --- | --- | --- |
| LRS3 原片 | `preprocessed_data/lrs3/{pretrain,trainval,test}` | 118,516、31,982、1,321 个 MP4 及同名 TXT；样本内有音轨 | 保持官方拆分，提取目标音轨和嘴图，再合成干扰；原片与子片按视频ID去重 |
| MMS LRS3 媒体与清单 | `derived/avsr_reproduction/mms_lrs3_24s/`；`derived/avsr_reproduction/manifests/mms_llama_433h_nocap_srlength_v6_20261004/` | 六列 TSV：train 164,916 行、test 1,321 行；训练音频累计约 436.312 小时，10 帧门槛后 164,667 条 | 可作为干净音视频候选，但 MMS 清单不是 Falcon 的 `mix.json`/`target.json`；须建立分离混音索引。历史实验用 LRS3 test 选模型，不能称为盲测 |
| VoxCeleb2 原媒体 | `preprocessed_data/voxceleb2/` | dev/test 历史元数据计数 1,092,009/36,237 个视频；WAV 与 M4A 是同源不同编码；TXT 是人脸轨迹及身份，并非逐字转写 | 按说话人、源视频生成目标音频与嘴图，按原 ID 避免与 VoxCeleb2-mix/AVYT 重叠 |
| MMS LRS3+Vox2 v6 | `derived/avsr_reproduction/mms_vox2_1759h_v1/`；`derived/avsr_reproduction/manifests/mms_llama_1759h_v6_datarepair_20261004/` | 清单历史总 820,468 行，其中 Vox2 dev 626,039、Vox2 test 29,513；总音频约 1,758.012 小时。英语识别文字为模型伪标签 | `test` 已参与该 MMS 训练；是否拿作 Falcon 训练及测试须先定独立口径。Vox2 身份、人脸轨迹不保证目标始终发声 |
| MuAViC-English / Whisper-Flamingo | 热端 `derived/avsr_reproduction/{muavic_wf_v1,muavic_lrs3_build_v1}/` 仅含 `MOVED_TO_COLD.md`；实际分卷在冷端 `_local_overflow_20261102/multimodal/audio-visual/derived/avsr_reproduction/` 同名目录 | `muavic_wf_v1` 为3个 tar、`muavic_lrs3_build_v1` 为6个 tar；历史 `en/train` 299,646 行，其中 28,577 条音视频长度差 >0.1 s、27,431 条 >1 s；`en/valid` 1,200、`en/test` 1,321 | 先按冷端清单校验/恢复，再按原始时间区间重建与复核；迁移标记不代表媒体在热端 |
| AVSpeech 分阶段产物 | `derived/avsr_reproduction/avspeech_scaled_v1/` | 2026-10-16 记录 conformed 1,913,250 MP4 / 1,913,167 WAV，预处理 832,782 MP4/WAV；manifest 和语速目录当时为空 | 阶段之间同源，不累计。按唯一片段键核目标音视频、保留原来源 split；历史伪标签中的 578 条源键为 test 不应自动归 train |
| LRS2 原样媒体与旧 rebuild | `preprocessed_data/lrs2/`、`preprocessed_data/lrs2_rebuild/` | 历史前者 48,165 MP4 与 TXT；旧 rebuild 含短音频、NPZ 嘴部及混音产物，多种文件不可直接相加 | 优先对照项目实际 LRS2-2Mix 清单；旧路径及切片时钟需重写并核对 |
| Vox2 旧分离产物 | `preprocessed_data/vox2_rebuild/` | 历史 `mix_2_spk_tr.scp` 100,000 行、`mix_2_spk_cv.scp` 10,000 行；行内容是音频路径与增益 | 有助追踪配方；需重新验证源文件、音量、目标嘴部及拆分，历史绝对路径不自动可用 |
| AudioSet 与辅助 | `AudioSet/`、`derived/avsr_reproduction/auto_avsr_landmarks/` | 音频事件素材与人脸关键点，不是额外语音分离样本 | 可用于噪声增强或视觉处理；按源ID划分并查许可 |

AVSpeech 的 `avspeech_pilot_v1/` 热端也仅保留 `MOVED_TO_COLD.md`；`avspeech_scaled_v1/` 存在分阶段目录。可使用各自迁移标记和冷端 `manifest.jsonl`/`SHA256SUMS` 确认恢复位置。

历史记录来自项目内 [2026-10-16 选录](historical-20261016-selected.json)，主要是数据与清单盘点，不是全体媒体逐一解码结果。目标口型必须与目标干净音频在同一时间轴；新增 160 ms 模型还需嘴部帧可用时间不晚于对应 40 ms 区间起点。

[返回多模态总览](../02-multimodal-datasets.md)
