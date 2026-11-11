# 多模态冷端存储与按需恢复

检查时间：2026-11-11（亚洲/上海）。本页记录只读盘点的现场结果、历史回执和按需恢复方法。本次没有批量恢复数据；为核查媒体格式，单独读取并抽取了一条预览视频，指定小型归档的哈希核对见逐库证据。训练、测试内容和适配判断见同目录其他数据集文档。

## 已挂载目录：直接查看与读取

`/aispeech-cfs` 已挂载，可以直接用 `ls`、`stat`、`ffprobe` 和读取文件等普通文件操作访问，无需先执行 TOS 下载。2026-11-11 本轮直接列出 `realssa_sim/{tr,cv,tt}`，分别看到 **202、19、19** 个 tar；三组第一个 tar 的各前 16 MiB 中均看到 `mix.wav`、`target.wav`、`mouths.pt` 和场景 JSON，并分别实际读取各一个样本的音频头与嘴图。详见[项目数据说明](01-project-data.md)。

同一挂载内共享大包的文件大小为 1,980,009,103,360 字节；本轮仅读取前 16 MiB 和头 20 个成员，实际看到 LRS2-rebuild 的 8 kHz 训练混音路径。该局部不代表整个大包的所有成员。直接文件清单见 [第二次挂载直查清单](aispeech-cfs-direct-listing-20261111-v2.json)；[第一次清单](aispeech-cfs-direct-listing-20261111.json)保留原貌。

## 1. 从本地挂载到 TOS 对象

服务器 `dev_a800_2gpus` 上 `findmnt -T /aispeech-cfs` 返回源 `cfs-fuse`、文件系统 `fuse.cfs-fuse`；服务器挂载配置 `/root/.cfs/config` 的非敏感字段给出：

```text
本地挂载点：/aispeech-cfs
TOS 挂载前缀：tos://aitogether-privater-cloudfs-st/aiplatform/aispeech-cfs
```

所以挂载路径 `/aispeech-cfs/voiceprint/data/multimodal/<相对路径>` 对应配置中的 `tos://aitogether-privater-cloudfs-st/aiplatform/aispeech-cfs/voiceprint/data/multimodal/<相对路径>`。例如共享包的映射是：

```text
/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar
tos://aitogether-privater-cloudfs-st/aiplatform/aispeech-cfs/voiceprint/data/multimodal/AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar
```

挂载路径本次可列出并 `stat`，文件长度为 **1,980,009,103,360 字节**。本次只读使用服务器默认 `tosutil` 配置查询这个具体 TOS 前缀，返回 **403 AccessDenied**；默认对象客户端凭据的列举权限与现有挂载的读取权限不同。上述 URI 是挂载配置提供的映射，并未通过 TOS API 成功核验对象或权限。未在文档中保存或展示密钥、签名 URL 或临时令牌。

共享 tar 被登记为 LRS2、LRS3、AVSpeech、VoxCeleb2 和 AudioSet 的既有冷端引用；这是同一个包，不能把包大小按五套数据重复求和。包名与登记说明用于定位，不单独证明其中每套数据的版本、内容或完整性。本次没有遍历整个共享大包或重新计算其哈希。

## 2. 37 个登记项的冷端位置与回执

下表按照服务器现有 `_catalog/dataset_registry.json` 的 **37 个唯一 slug** 生成；“本机状态”来自当前热端 `status.json` 的 `state`，无该文件时注明注册初始状态。整库回执为冷端 `<slug>/archive_receipt.json`；增量回执为 `<slug>/incremental_receipt.json`。文件数和字节数是回执记录的载荷文件数及原始字节数，不是样本数、小时数，也不是本次重新计算的校验结果。增量归档覆盖已记录文件，不自动代表上游整套发行版完整。具体文件清单及哈希以同目录 `SHA256SUMS` 和历史回执为依据。

| 注册 slug | 当前热端状态 | 冷端证据类型 | 回执文件数 | 回执字节数 | 挂载路径 |
| --- | --- | --- | ---: | ---: | --- |
| `lrs2` | existing_cold_reference | 共享 tar 引用 | — | 共享包计一次 | `AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| `lrs3` | existing_cold_reference | 共享 tar 引用 | — | 共享包计一次 | `AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| `lrs_combined` | derived_not_materialized | 本地派生，未默认生成 | — | — | — |
| `tcd_timit` | source_verified | 逐文件增量回执 | 137 | 676,948,863,296 | `/aispeech-cfs/voiceprint/data/multimodal/tcd_timit/raw` |
| `grid` | archived | 整库回执 | 36 | 16,211,527,582 | `/aispeech-cfs/voiceprint/data/multimodal/grid/raw` |
| `lombard_grid` | archived | 整库回执 | 5 | 2,485,068,233 | `/aispeech-cfs/voiceprint/data/multimodal/lombard_grid/raw` |
| `roomreader` | archived | 整库回执 | 1339 | 159,508,080,509 | `/aispeech-cfs/voiceprint/data/multimodal/roomreader/raw` |
| `candor` | archived | 整库回执 | 166 | 951,758,137,196 | `/aispeech-cfs/voiceprint/data/multimodal/candor/raw` |
| `voxceleb2` | existing_cold_reference | 共享 tar 引用 | — | 共享包计一次 | `AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| `avspeech` | merged_complete | 共享 tar 引用 | — | 共享包计一次 | `AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| `audioset` | existing_cold_reference | 共享 tar 引用 | — | 共享包计一次 | `AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar` |
| `wildvsr` | archived | 整库回执 | 1 | 91,755,398 | `/aispeech-cfs/voiceprint/data/multimodal/wildvsr/raw` |
| `avcocktail` | archived | 整库回执 | 206 | 3,438,836,255 | `/aispeech-cfs/voiceprint/data/multimodal/avcocktail/raw` |
| `avyt` | archived | 整库回执 | 1320 | 1,419,416,723,248 | `/aispeech-cfs/voiceprint/data/multimodal/avyt/raw` |
| `muavic_metadata` | archived | 整库回执 | 10 | 15,182,727,260 | `/aispeech-cfs/voiceprint/data/multimodal/muavic_metadata/raw` |
| `muavic` | archived | 整库回执（明确部分） | 3875 | 767,683,633,536 | `/aispeech-cfs/voiceprint/data/multimodal/muavic/raw` |
| `multivsr_metadata` | archived | 整库回执 | 10 | 42,984,746,968 | `/aispeech-cfs/voiceprint/data/multimodal/multivsr_metadata/raw` |
| `multivsr` | downloading | 逐文件增量回执 | 76003 | 10,573,315,866,489 | `/aispeech-cfs/voiceprint/data/multimodal/multivsr/raw` |
| `mafi` | archived | 整库回执 | 24 | 5,128,929,084 | `/aispeech-cfs/voiceprint/data/multimodal/mafi/raw` |
| `voxmm` | archived | 整库回执 | 8 | 43,846,182,092 | `/aispeech-cfs/voiceprint/data/multimodal/voxmm/raw` |
| `audioset_strong` | archived | 整库回执 | 4 | 75,750,940 | `/aispeech-cfs/voiceprint/data/multimodal/audioset_strong/raw` |
| `videocc` | archived | 整库回执；另有增量回执 | 1 | 203,044,884 | `/aispeech-cfs/voiceprint/data/multimodal/videocc/raw` |
| `cuave_partial` | source_verified | 逐文件增量回执 | 1 | 640,156,002 | `/aispeech-cfs/voiceprint/data/multimodal/cuave_partial/raw` |
| `talkvid_meta` | source_verified | 逐文件增量回执 | 951 | 632,073,719 | `/aispeech-cfs/voiceprint/data/multimodal/talkvid_meta/raw` |
| `visper` | source_verified | 逐文件增量回执 | 20 | 6,971,436,241 | `/aispeech-cfs/voiceprint/data/multimodal/visper/raw` |
| `real_world_avse` | source_verified | 逐文件增量回执 | 4 | 9,350,891,169 | `/aispeech-cfs/voiceprint/data/multimodal/real_world_avse/raw` |
| `mld_vc` | source_verified | 逐文件增量回执 | 5 | 9,914,542,126 | `/aispeech-cfs/voiceprint/data/multimodal/mld_vc/raw` |
| `aspire` | source_verified | 逐文件增量回执 | 1 | 12,960,167,179 | `/aispeech-cfs/voiceprint/data/multimodal/aspire/raw` |
| `unitalk_asd` | source_verified | 逐文件增量回执 | 353 | 14,555,424,127 | `/aispeech-cfs/voiceprint/data/multimodal/unitalk_asd/raw` |
| `ravdess` | source_verified | 逐文件增量回执 | 49 | 25,600,214,208 | `/aispeech-cfs/voiceprint/data/multimodal/ravdess/raw` |
| `ellipsis_lrs3` | source_verified | 逐文件增量回执 | 34 | 25,041,170,491 | `/aispeech-cfs/voiceprint/data/multimodal/ellipsis_lrs3/raw` |
| `facestar` | source_verified | 逐文件增量回执 | 115 | 29,817,491,437 | `/aispeech-cfs/voiceprint/data/multimodal/facestar/raw` |
| `mandarin_avsr` | source_verified | 逐文件增量回执 | 9 | 49,977,304,020 | `/aispeech-cfs/voiceprint/data/multimodal/mandarin_avsr/raw` |
| `darknsu_lrs3` | source_verified | 逐文件增量回执 | 2 | 62,975,957,463 | `/aispeech-cfs/voiceprint/data/multimodal/darknsu_lrs3/raw` |
| `vlr` | source_verified | 逐文件增量回执 | 8 | 70,453,584,679 | `/aispeech-cfs/voiceprint/data/multimodal/vlr/raw` |
| `chinese_lips` | source_verified | 逐文件增量回执 | 19 | 93,069,232,024 | `/aispeech-cfs/voiceprint/data/multimodal/chinese_lips/raw` |
| `voxceleb2_mix` | source_verified | 逐文件增量回执 | 14 | 563,291,453,440 | `/aispeech-cfs/voiceprint/data/multimodal/voxceleb2_mix/raw` |

表中 `MultiVSR` 仍在变化：查询时 `status.json` 为 `downloading`，回执与 `SHA256SUMS` 在本次复核中由较早读取的 75,991 项变化到 **76,003 项**，表示存在增量写入，表格仅是生成时快照。`voxceleb2_mix/raw` 除 14 个回执记录对象外还可见 `.write_test_avsr`，该标记不在 14 行 `SHA256SUMS` 的口径中，不计为音视频样本。`videocc` 同时有整库与增量回执，指向同一份载荷，不能重复计数。`muavic` 整库形式的回执明确写了获批准的部分归档：预期 3,882 项、实归档 3,875 项，另有 7 项未获得；不能把它写成完整上游视频。

`realssa_sim` 和 `realssa_real` 不是上述 37 项注册表行，却各自在 `/aispeech-cfs/voiceprint/data/multimodal/realssa_sim` 与 `/aispeech-cfs/voiceprint/data/multimodal/realssa_real` 直接存在。前者目录可见 `tr/`、`cv/`、`tt/` 和 `_MANIFEST.json`，后者根目录可见 audio、faces、landmarks、metadata、mouth、references、transcripts 等数据及清单，另有 `raw/videos/`；这两个目录不能根据 catalog 回执表被误判为缺失，具体 split、采样数量和样本身份以项目数据盘点为准。

### 常用目标提取候选的实际载荷样子

以下是仅取每个目录前若干名称的只读抽查，并非完整清单：

- `mandarin_avsr/raw` 有 `tr_1mix.tar.gz` 至 `tr_5mix.tar.gz`、`tr_all.tar.gz`、`cv.tar.gz`、`tt.tar.gz` 和 `preview.tar.gz`，具备混合数据分包的明确命名；是否有与每路目标对应的嘴部画面、清晰目标和时钟，需对选定分包查成员和加载器。
- `voxceleb2_mix/raw` 有 `audio_clean_part_aa`…`ag` 和 `orig_part_aa`…`ag`，各 7 片。[发布者数据卡](https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix#usage) 给出的恢复方法是两组分别按名称顺序拼为 `audio_clean.tar`、`orig.tar` 再解包；`.write_test_avsr` 不是分卷。恢复前逐片核对清单和可用空间，输出放到新的独立目标。本轮只读取两组首片开头，没有实际拼接或验收完整 tar。
- `real_world_avse/raw` 有 `track1_dev.zip`、`track2_dev.zip`、`Testing_set_GT.zip`；`aspire/raw` 有 `ASPIRE.zip`；它们属于增强/真实噪声候选，作为目标说话人提取监督时还要核对目标身份、同步视频和参考语音。
- `facestar/raw` 有按男女说话人及 train/test 划分的目录；`mld_vc/raw` 有 `audio.zip`、`video.zip`、`landmarks.zip`、`text.zip`、`metadata.jsonl`；`avcocktail/raw` 可见 `videos/` 和 `labels/`。这些路径为后续核对说话人、音轨、嘴部及配对提供入口。
- `avyt/raw` 可见 `avyt/`、`dialog/`、`lrs2/`、`vox2/` 等子项，与既有 LRS2/VoxCeleb2 可能共享来源；合并前应以原始视频及说话人身份去重、检查训练和测试泄漏。

## 3. 单独的本地溢出归档

`/aispeech-cfs/voiceprint/data/multimodal/_local_overflow_20261102` 是曾经从热端迁出的本地副本，不是上述共享 tar，也不在本次 `_catalog` 的归档行内。本次只读找到以下与 AVSpeech / AVSR 派生相关的 `receipt.json`、`manifest.jsonl`、`SHA256SUMS` 和 `RESTORE.md`。`all_verified` 是历史归档回执记录，未在本次重新读取全部分卷哈希；实际分卷数是本次只读目录计数。

| 溢出区相对路径 | 回执源文件数 | 原文件字节数 | 回执分卷数 / 本次可见 | 历史 all_verified |
| --- | ---: | ---: | ---: | --- |
| `avspeech/videos/train` | 1,888,171 | 1,720,138,186,380 | 87 / 87 | True |
| `avspeech/videos/test` | 120,957 | 111,658,448,215 | 6 / 6 | True |
| `avspeech/extras/superseded` | 313,777 | 432,847,664,600 | 15 / 15 | True |
| `avspeech/_gapfill/avspeech-test.zip` | 1 | 48,575,346,822 | 1 / 1 | True |
| `multimodal/audio-visual/derived/avsr_reproduction/avspeech_pilot_v1` | 30,516 | 15,541,744,261 | 2 / 2 | True |
| `multimodal/audio-visual/derived/avsr_reproduction/muavic_wf_v1` | 604,337 | 79,622,672,044 | 3 / 3 | True |
| `multimodal/audio-visual/derived/avsr_reproduction/muavic_lrs3_build_v1` | 537,728 | 154,140,879,053 | 6 / 6 | True |

`avspeech/videos/train`、`avspeech/videos/test` 的热端目录保留了 `MOVED_TO_COLD.md`，明确指向上述冷端分卷；历史记录显示这份迁移件与共享大包是不同的冷端保存对象。溢出区还有 `codex_artifacts/`、`miniforge3_envs/` 历史产物或环境归档，不属于本页的数据集样本，不能计入音视频数量。

## 4. 以后需要时如何取回

当前先使用已可读取的 cfs-fuse 挂载路径，针对确定的数据集、split 和具体文件建立任务专属目标目录；不要先把整包复制到训练目录。原始载荷保留冷端不动。普通 `<slug>/raw/` 载荷可先查同一目录的 `archive_receipt.json` 或 `incremental_receipt.json` 和 `SHA256SUMS`，只复制所需文件到新目标，并对已取回文件逐一核对清单中相应 SHA-256 与大小。数据许可及用户、机构和训练用途的适配情况以来源条件和现有授权记录复核；本次没有变更权限或开始取回。

对于 `_local_overflow_20261102` 分卷，先读取该具体目录的 `receipt.json`、`RESTORE.md` 和 `manifest.jsonl`，按清单记录定位成员与分卷，再检查分卷校验和与目标空间。清单每行是 JSON 数组，首项是 tar 相对成员名，第四项是分卷编号；不是按文件名猜分卷。可以先在**新的**目标目录按选定分卷和成员提取，而不是覆盖现有训练数据。

本次对 AVSpeech `train/shard_0000.tar` 只读取了最前面的 tar 成员名，实际以 `train/shard_*/...` 开头，热端 `MOVED_TO_COLD.md` 也写明应在 `.../avspeech/videos` 下恢复。因此若希望得到 `新目标/videos/train/...`，示例为：

```bash
COLD=/aispeech-cfs/voiceprint/data/multimodal/_local_overflow_20261102/avspeech/videos/train
DEST=/path/to/new-target/videos
# 先检查 manifest.jsonl 和 SHA256SUMS，确定确实需要的成员及其分卷。
mkdir -p "$DEST"
tar -C "$DEST" -xf "$COLD/shard_0000.tar" -- 'train/实际需要的相对成员路径.mp4'
```

命令中的成员名须替换为所选分卷清单中的确切值；上例是按单文件取回的格式，不是本次执行记录。冷端同目录 `RESTORE.md` 中“videos 的上一级”文字存在歧义，而实际 tar 成员有 `train/` 前缀；恢复时应根据实查成员名选定 `-C`，避免生成错误层级。完整恢复 87 卷 train 和 6 卷 test 属于大规模数据操作，应先核对实际目标容量、任务身份和清单范围，之后再按任务单独安排。

## 5. 本页证据与口径

- 服务器热端注册表：`/root/group-shared/voiceprint/data/multimodal/_catalog/dataset_registry.json`；整库历史记录：同目录 `archive_manifest.jsonl`。注册表 `initial_state` 是初始状态，不代替现行 `status.json`。
- 冷端逐库回执：`/aispeech-cfs/voiceprint/data/multimodal/<slug>/archive_receipt.json` 或 `incremental_receipt.json`；具体载荷校验清单 `SHA256SUMS`。
- 冷端溢出：`.../_local_overflow_20261102/<相对路径>/{receipt.json,manifest.jsonl,SHA256SUMS,RESTORE.md}`；热端 `avspeech/videos/{train,test}/MOVED_TO_COLD.md`。
- 本次检查为注册表与目录、回执及少量 tar 头部的只读核对；没有对所有媒体、完整 tar、授权状态或对象 API 作成功的逐项验证。数据可训练性、可用于商业用途以及跨数据集身份重叠均需以专项证据另行确认。
