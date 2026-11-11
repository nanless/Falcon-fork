# 2026-11-11 冷端 a–m 目录直接检查

对登记表中首字母 a–m 的 **24 项**直接读取 `/aispeech-cfs/voiceprint/data/multimodal/<slug>/raw` 第一层，记录目录和实际文件名。表中数字是目录直接子项，不是媒体样本数；展示最多前 20 个名称，嵌套目录需继续查看。共享 tar 引用和未物化条目不具备单独 `raw/`。

| 登记项 | 冷端 raw 第一层 | 文件 / 子目录 | 部分实际名称 |
| --- | ---: | ---: | --- |
| [aspire](multimodal/aspire.md) | 1 | 1 / 0 | `ASPIRE.zip` |
| [audioset](multimodal/audioset.md) | — | — | 共享冷端引用：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| [audioset_strong](multimodal/audioset_strong.md) | 4 | 4 / 0 | `audioset_eval_strong.tsv`、`audioset_eval_strong_framed_posneg.tsv`、`audioset_train_strong.tsv`、`mid_to_display_name.tsv` |
| [avcocktail](multimodal/avcocktail.md) | 4 | 2 / 2 | `.gitattributes`、`README.md`、`labels`、`videos` |
| [avspeech](multimodal/avspeech.md) | — | — | 共享冷端引用：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| [avyt](multimodal/avyt.md) | 6 | 2 / 4 | `.gitattributes`、`README.md`、`avyt`、`dialog`、`lrs2`、`vox2` |
| [candor](multimodal/candor.md) | 166 | 166 / 0 | `raw_media_part_001.zip`、`raw_media_part_002.zip`、`raw_media_part_003.zip`、`raw_media_part_004.zip`、`raw_media_part_005.zip`、`raw_media_part_006.zip` 等 |
| [chinese_lips](multimodal/chinese_lips.md) | 19 | 19 / 0 | `image.zip`、`meta_all.csv`、`meta_test.csv`、`meta_test.json`、`meta_train.csv`、`meta_valid.csv` 等 |
| [cuave_partial](multimodal/cuave_partial.md) | 1 | 1 / 0 | `cuave-group-aligned.zip` |
| [darknsu_lrs3](multimodal/darknsu_lrs3.md) | 2 | 2 / 0 | `video.tar.gz`、`wav.tar.gz` |
| [ellipsis_lrs3](multimodal/ellipsis_lrs3.md) | 1 | 0 / 1 | `shards` |
| [facestar](multimodal/facestar.md) | 5 | 0 / 5 | `female_speaker_testset`、`female_speaker_trainset`、`male_speaker_testset`、`male_speaker_trainset`、`paper_materials` |
| [grid](multimodal/grid.md) | 36 | 36 / 0 | `alignments.zip`、`audio_25k.zip`、`jasagrid.pdf`、`s1.zip`、`s10.zip`、`s11.zip` 等 |
| [lombard_grid](multimodal/lombard_grid.md) | 5 | 5 / 0 | `lombardgrid_alignment.zip`、`lombardgrid_audio.zip`、`lombardgrid_front.zip`、`lombardgrid_json.zip`、`lombardgrid_side.zip` |
| [lrs2](multimodal/lrs2.md) | — | — | 共享冷端引用：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| [lrs3](multimodal/lrs3.md) | — | — | 共享冷端引用：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| [lrs_combined](multimodal/lrs_combined.md) | — | — | 没有单独 raw；查看该库卡片及共享引用 |
| [mafi](multimodal/mafi.md) | 5 | 0 / 5 | `Analysis`、`Behavioural Data`、`Norms`、`Scripts`、`Videos Folder` |
| [mandarin_avsr](multimodal/mandarin_avsr.md) | 9 | 9 / 0 | `cv.tar.gz`、`preview.tar.gz`、`tr_1mix.tar.gz`、`tr_2mix.tar.gz`、`tr_3mix.tar.gz`、`tr_4mix.tar.gz` 等 |
| [mld_vc](multimodal/mld_vc.md) | 5 | 5 / 0 | `audio.zip`、`landmarks.zip`、`metadata.jsonl`、`text.zip`、`video.zip` |
| [muavic](multimodal/muavic.md) | 1 | 0 / 1 | `source_videos` |
| [muavic_metadata](multimodal/muavic_metadata.md) | 10 | 10 / 0 | `20words_mean_face.npy`、`ar_metadata.tgz`、`de_metadata.tgz`、`el_metadata.tgz`、`en_metadata.tgz`、`es_metadata.tgz` 等 |
| [multivsr](multimodal/multivsr.md) | 1 | 0 / 1 | `source_videos` |
| [multivsr_metadata](multimodal/multivsr_metadata.md) | 10 | 10 / 0 | `.gitattributes`、`README.md`、`dataset_teaser.gif`、`multivsr.tar`、`test.csv`、`train_major.csv` 等 |

## 进一步直接读取的对象

- AVCocktail 的冷端 raw 经目录遍历为 206 个文件，其中 204 个 tar；AVYT 为 1,320 个文件，其中 1,318 个 tar。FaceStar 115 个文件中 112 个 ZIP；Ellipsis-LRS3 为 32 个 tar 和两个清单/报告文件。
- [ZIP 成员名称逐项统计](cold-zip-pair-audit-20261111.json)：ASPIRE 5,940 个 WAV 与 5,940 个 MOV，Chinese-LiPS 的 train/val/test 分别 30,341/1,959/3,908 个 WAV 与 MP4，以及 Lombard GRID 音频与正面视频各 5,390 个成员，分别按去扩展名的文件名（stem）核对，集合一一对应；MLD-VC 音频与视频按去掉各自顶层目录和扩展名后的相对路径键有 20,803 个相同键。
- MaFi 冷端 raw 逐层列出 24 个文件，包括 12 个 CSV、5 个 Python 脚本和 5 个视频 ZIP；MuAViC 元数据目录为 9 个语言 `.tgz` 与 1 个均值人脸 `.npy`；MultiVSR 元数据目录包括 5 个 CSV、1 个 tar 及辅助文件。MuAViC `source_videos/` 下可见 `ar/de/el/es/fr/it/pt/ru`；MultiVSR 的 `source_videos/` 有大量 MP4，数量和归档状态持续变化。
- 上述 ZIP 检查只读取中央目录，名称匹配不证明音视频内容相同、逐条时间同步、目标人物身份或训练许可。`cuave-group-aligned.zip` 读取到 22 个 `.mat` 成员；GRID `s1.zip` 读取到 1,000 个 `.mpg` 与一个 `.db` 文件。

逐库完整背景、训练准备及许可证记录见[多模态总览](02-multimodal-datasets.md)。本页机器记录见 [cold-direct-a-m-20261111.json](cold-direct-a-m-20261111.json)。
