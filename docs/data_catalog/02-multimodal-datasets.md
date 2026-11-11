# 多模态数据全目录：Falcon 训练、评测与冷存取用

## 怎样读这份目录

本页回答“多模态目录里有什么、哪些能用于目标说话人提取、使用前要做什么”。当前项目**实际使用**的训练与测试数据、数量、训练入口和现有结果，见 [项目数据与实验](01-project-data.md)。这里的 37 个登记库是候选目录，不等于项目已用过 37 个库。

**盘点时间与依据：**2026-11-11 00:44（北京时间），在 `dev_a800_2gpus` 逐项读取登记表、每库 `status.json`、冷端归档收据与校验清单开头；源媒体与格式优先采用 2026-10-16 有界实查，并在 37 份卡片中标明。动态下载数字会变化；[00:44 逐库快照](multimodal/snapshot-20261111-0044.json) 与 [00:46 独立快照](catalog-snapshot-20261111.json) 分别留档，数字可能不同。

Falcon 训练所需的是同一目标人的 **混合音频、干净目标音频和同步嘴部轨迹**；文字并非分离损失的必备输入。可供构造混音的干净单人音视频属于候选；只有嘴图、字幕、事件标签、音频或 URL 的数据，不能单独当作一条已就绪的训练样本。客观分离测试还需与混音同时间轴的目标参考；没有参考的真实录音可做听感或无参考评价。

```text
已有单人干净音轨 + 同步嘴部画面
    └─ 依说话人/原视频划分 ─ 合成干扰和噪声 ─ mix + target + mouth
已有混音 + 干净目标 + 嘴部画面
    └─ 对齐目标身份/时钟 ─ 转换采样率、帧率和索引 ─ Falcon 训练/评测
只有字幕、人脸轨迹、URL 或纯视觉
    └─ 补齐媒体和目标配对，或作为视觉/活动检测辅助数据
```

## 最值得优先核对的候选

1. [Mandarin AVSR](multimodal/mandarin_avsr.md)：当前中文目标分离数据最接近现有训练输入。`tr_all.tar.gz` 实查 40,000 行、1–5mix 各 8,000，目标音频与嘴图路径配对；嘴部为 `.npz`，需转换为当前加载器所需格式。先核对发布授权、目标身份及全量媒体，保留 cv/tt；[索引逐行与归档校验记录](multimodal/mandarin-avsr-index-check-20261111.json)。另对一条 3mix 预览视频完成可寻址探测和首帧解码：384×384、25 fps、267 帧、10.68 秒；这是单例，[解码记录](multimodal/mandarin-avsr-preview-decode-20261111.json)。
2. [VoxCeleb2-mix](multimodal/voxceleb2_mix.md)：冷端记录 14/14 分片；已抽查干净 WAV 与原始 MP4 首片对应说话人/视频 ID。先核验分片拼接与全包内容，生成目标嘴图和混音配方；不要仅以包名断言已提供可直接训练的混音；[分片与首片抽样记录](multimodal/voxceleb2-mix-headers-20261111.json)。另有一组首片测试样本的 WAV 与 MP4 音轨实际解码，音频对音频相关系数约 1，见[单例音轨探针](vox-first-pair-audio-sync-20261111.json)；这只是单例，尚未验证嘴部与音频同步。
3. [Chinese-LiPS](multimodal/chinese_lips.md)：已处理 train/val/test 历史盘点分别 30,341/1,959/3,908 对，16 kHz、25 fps 的代表样本可作为中文目标语音候选；核对目标音质后再构建分离混音。原始训练分卷缺 `train.zip` 末卷，与独立处理版 ZIP 区分；记录的许可为 CC BY-NC-SA。
4. [AVYT](multimodal/avyt.md)、[AVSpeech](multimodal/avspeech.md)、[LRS3](multimodal/lrs3.md) 与 [MultiVSR](multimodal/multivsr.md)：适合扩充说话人、语言和场景，但要先筛选真实有声目标、建立干净源，按原视频去重。AVYT 已混入 LRS2/Vox2；MultiVSR 仍在按视频 ID 重建，其状态计数会变化。
5. [Real-World AVSE](multimodal/real_world_avse.md)、[AVCocktail](multimodal/avcocktail.md)、[VoxMM](multimodal/voxmm.md) 和 [RoomReader](multimodal/roomreader.md)：适合独立的真实重叠、视觉退化和会议场景评测；需逐项核对参考目标和指标，避免把只含文字/视频的条目当作有干净语音参考。

## 37 个登记项逐库入口

对象列是状态或收据中的**文件/分片数**，不是音视频片段数。按 2026-11-11 00:44 CST 快照，`source_verified` 可能已逐文件迁至冷端，`archived` 只描述冻结或已批准归档范围。

| 数据集 | Falcon 用途 | 状态 | 对象数量口径 |
| --- | --- | --- | --- |
| [AVSpeech](multimodal/avspeech.md) | A｜预训练候选 | `merged_complete` | 未统计；见卡片 |
| [AVYT](multimodal/avyt.md) | A｜大规模预训练候选 | `archived` | 冷端收据 1,320 |
| [chinese_lips](multimodal/chinese_lips.md) | A｜中文合成混音候选 | `source_verified` | 冷端收据 19 |
| [LRS2](multimodal/lrs2.md) | A｜项目基线源 | `existing_cold_reference` | 未统计；见卡片 |
| [LRS3](multimodal/lrs3.md) | A｜合成混音候选 | `existing_cold_reference` | 未统计；见卡片 |
| [mandarin_avsr](multimodal/mandarin_avsr.md) | A｜分离训练优先 | `source_verified` | 冷端收据 9 |
| [VoxCeleb2](multimodal/voxceleb2.md) | A｜说话人源候选 | `existing_cold_reference` | 未统计；见卡片 |
| [voxceleb2_mix](multimodal/voxceleb2_mix.md) | A｜分离训练候选 | `source_verified` | 冷端收据 14 |
| [aspire](multimodal/aspire.md) | B｜真实噪声专项 | `source_verified` | 冷端收据 1 |
| [AVCocktail](multimodal/avcocktail.md) | B｜外部场景评测 | `archived` | 冷端收据 206 |
| [CANDOR](multimodal/candor.md) | B｜双人会议候选 | `archived` | 冷端收据 166 |
| [cuave_partial](multimodal/cuave_partial.md) | B｜双人重叠专项 | `source_verified` | 冷端收据 1 |
| [darknsu_lrs3](multimodal/darknsu_lrs3.md) | B｜LRS3镜像替代 | `source_verified` | 冷端收据 2 |
| [facestar](multimodal/facestar.md) | B｜高质量小样本 | `source_verified` | 冷端收据 115 |
| [GRID](multimodal/grid.md) | B｜受控源训练/测试 | `archived` | 冷端收据 36 |
| [Lombard GRID](multimodal/lombard_grid.md) | B｜噪声/侧脸专项 | `archived` | 冷端收据 5 |
| [mld_vc](multimodal/mld_vc.md) | B｜会议域候选 | `source_verified` | 冷端收据 5 |
| [MuAViC Full](multimodal/muavic.md) | B｜多语言来源 | `archived` | 冷端收据 3,875 |
| [MultiVSR Full Videos](multimodal/multivsr.md) | B｜多语言候选 | `downloading` | 冷端收据 75,991 |
| [ravdess](multimodal/ravdess.md) | B｜情绪和表情测试 | `source_verified` | 冷端收据 49 |
| [real_world_avse](multimodal/real_world_avse.md) | B｜保留开发/测试 | `source_verified` | 冷端收据 4 |
| [RoomReader](multimodal/roomreader.md) | B｜多人会议评测/候选 | `archived` | 冷端收据 1,339 |
| [TCD-TIMIT](multimodal/tcd_timit.md) | B｜受控源训练/测试 | `source_verified` | 冷端收据 137 |
| [VoxMM / LRS-VoxMM](multimodal/voxmm.md) | B｜长对话评测 | `archived` | 冷端收据 8 |
| [AudioSet](multimodal/audioset.md) | C｜干扰/噪声源 | `existing_cold_reference` | 未统计；见卡片 |
| [audioset_strong](multimodal/audioset_strong.md) | C｜噪声筛选元数据 | `archived` | 冷端收据 4 |
| [ellipsis_lrs3](multimodal/ellipsis_lrs3.md) | C｜视觉派生数据 | `source_verified` | 冷端收据 34 |
| [LRS Combined](multimodal/lrs_combined.md) | C｜尚未物化 | `derived_not_materialized` | 未统计；见卡片 |
| [MaFi](multimodal/mafi.md) | C｜纯视觉词级评测 | `archived` | 冷端收据 24 |
| [MuAViC Metadata](multimodal/muavic_metadata.md) | C｜切片辅助元数据 | `archived` | 冷端收据 10 |
| [MultiVSR Metadata](multimodal/multivsr_metadata.md) | C｜切片/字幕元数据 | `archived` | 冷端收据 10 |
| [talkvid_meta](multimodal/talkvid_meta.md) | C｜元数据/小样素材 | `source_verified` | 冷端收据 951 |
| [unitalk_asd](multimodal/unitalk_asd.md) | C｜活动检测辅助 | `source_verified` | 冷端收据 353 |
| [videocc](multimodal/videocc.md) | C｜URL元数据 | `archived` | 冷端收据 1 |
| [visper](multimodal/visper.md) | C｜元数据候选 | `source_verified` | 冷端收据 20 |
| [vlr](multimodal/vlr.md) | C｜纯视觉候选 | `source_verified` | 冷端收据 8 |
| [WildVSR](multimodal/wildvsr.md) | C｜纯视觉外部评测 | `archived` | 冷端收据 1 |

## 37 项以外的目录和数据

- **联合冷端大包**：`/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`，当前文件长度 1,980,009,103,360 B。LRS2、LRS3、VoxCeleb2、AudioSet 及 AVSpeech 的 `existing_cold_reference` 指向该包，**共享包只计一次**；包名和局部成员不能证明各发行版完整性。
- **RealSSA 冷端**：`/aispeech-cfs/voiceprint/data/multimodal/realssa_real/`、`realssa_sim/` 不在这 37 项登记表内。`_download/_archive_done.json` 的 2026-10-23 记录写入 cold 目标且 `ok=true`；项目实际 RealSSA 划分与现有结果见 [项目数据](01-project-data.md)。本轮不将历史回执当逐文件重算。
- **嵌套已有处理集**：`/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/` 下有 `preprocessed_data/`、`avspeech_download/`、`derived/avsr_reproduction/` 及 `AudioSet/`。这些是源数据的处理结果、模型或配套资产，不应再计算为独立原始来源。包括 LRS3、Vox2、MuAViC-English、AVSpeech 和 LRS2 等；历史入口见 [派生资料](multimodal/derived.md)。
- **AVSpeech 冷端溢出**：`/aispeech-cfs/voiceprint/data/multimodal/_local_overflow_20261102/avspeech/` 下含 `videos/`、`_gapfill/` 和 `extras/`；它们是原热端对象迁移件，不是又下载的一整套 AVSpeech。
- **目录和日志**：`_catalog/` 是登记、归档、审计和运行状态；`_download/` 是 RealSSA 下载/归档记录；`multimodal/` 是嵌套数据树。它们不是独立语料。旧快照中的空归档记录 `r` 在本次两根目录浅层核对中未发现，不当作数据集。

## 接入、拆分与数量口径

1. 先确定目标任务与评测口径：同一时段的混合声、单个目标干净声、目标嘴图必须可按 ID 和时间对上。带噪音频不自动等于混音+干净目标；静音嘴图不自带音轨。
2. 按原始说话人、源视频和会话建立 train/validation/test，先排除现有 LRS2、LRS3、VoxCeleb2、AVSpeech 与 AVYT 派生项的交叉，再做合成混音；同一原片的多种噪声、切片、编码和视角不要分到训练与独立测试。
3. 记录源录音、目标 ID、干扰 ID、起止时间、混音比例、噪声、视频帧时钟及转换版本；当前项目需要 16 kHz 单声道及按时间对齐的 25 fps 嘴图，具体张量形状以各入口为准。新流式模型每 160 ms 使用 4 帧，所选帧可用时间不得晚于对应 40 ms 区间起点。
4. 对候选集先小规模建立索引、查路径、验证采样率与同步、抽听并核对目标身份，再按拆分核对数量。记录按混音条目、唯一源视频、说话人、小时和已解码/失败数分别报告，不把包数或重叠派生版本当独立样本。
5. 许可按真实来源和用途确认。`other`、`未声明` 或镜像自身标识不是完整授权说明；AVYT 的本地许可未声明、Mandarin AVSR 为 other，Chinese-LiPS 和 Real-World AVSE 的现有记录含非商业条件。

## 冷端按需取用

服务器的 `/aispeech-cfs` 当前显示为 `cfs-fuse` 挂载；目录与历史归档记录将它作为 TOS 冷端入口。每库卡片列出具体 `raw/` 和 `SHA256SUMS`。本轮只读取目录、状态与少量样本，没有批量下载、解包或重算所有哈希。

后续选择**一个明确文件或一个必要拆分**时，先检查冷端收据和目标盘剩余空间，把文件复制到新的任务目录，再用相同相对路径核对该对象的 SHA-256；不要按联合大包名称推断完整发行范围。分卷包先核全组和可用容器格式，合并输出写新路径，保留原分片及清单。下面是可在有足够空间时使用的单文件示例；示例本轮未执行：

```bash
set -euo pipefail
SRC=/aispeech-cfs/voiceprint/data/multimodal/chinese_lips
OUT=/root/tmp/falcon-chinese-lips-task
mkdir -p "$OUT/raw"
rsync -a --ignore-existing "$SRC/raw/processed_train.zip" "$OUT/raw/"
cd "$OUT"
grep -F "  raw/processed_train.zip" "$SRC/SHA256SUMS" | sha256sum -c -
```

冷端路径是访问入口，不等于每个对象已经恢复到训练工作目录。后续恢复数据及制作索引时，再记录真实文件数、样本数、耗时、失败项和分母。
