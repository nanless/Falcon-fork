# 多模态后 18 库及冷端独立复核

2026-11-11 对 37 张登记项数据卡按 slug 字母排序，独立复核第 20～37 张及 `derived.md`；前 19 张由其他审查者处理。本页记录检查范围和已经修改的内容。

## 逐库覆盖

对下列 18 库，逐一核对 [00:44 登记和状态快照](multimodal/snapshot-20261111-0044.json) 中 `state`、许可字段、冷端回执文件数和字节数，以及数据卡中引用的历史盘点。快照中未给出独立回执的共享包引用按原样记录；MultiVSR 仍在增量写入，快照数值只对应其采集时间。

| 数据卡 | 00:44 状态 | 冷端回执文件数 | 审阅重点 |
| --- | --- | ---: | --- |
| [MLD-VC](multimodal/mld_vc.md) | source_verified | 5 | 音视频与文字路径、会议场景划分 |
| [MuAViC](multimodal/muavic.md) | archived | 3,875 | 预期 3,882、已接受的七项缺失、语种和划分 |
| [MuAViC metadata](multimodal/muavic_metadata.md) | archived | 10 | 人脸轨迹元数据与源媒体分开 |
| [MultiVSR](multimodal/multivsr.md) | downloading | 75,991 | 回执与状态时间不同、持续增量、元数据与媒体求交 |
| [MultiVSR metadata](multimodal/multivsr_metadata.md) | archived | 10 | 文本/轨迹与视频分开；元数据许可与视频来源区别 |
| [RAVDESS](multimodal/ravdess.md) | source_verified | 49 | 语音/歌唱、音频/视频版本和表演者划分 |
| [Real-World AVSE](multimodal/real_world_avse.md) | source_verified | 4 | 开发测试用途、有参考目标音频的分母 |
| [RoomReader](multimodal/roomreader.md) | archived | 1,339 | 参与者、单路和全场音视频、非商业条款 |
| [TalkVid metadata](multimodal/talkvid_meta.md) | source_verified | 951 | 小样媒体与大规模发行说明的区别 |
| [TCD-TIMIT](multimodal/tcd_timit.md) | source_verified | 137 | 实际完成对象和官方两类拆分 |
| [UniTalk-ASD](multimodal/unitalk_asd.md) | source_verified | 353 | 发声活动标签与干净目标、文字的区别 |
| [VideoCC](multimodal/videocc.md) | archived | 1 | URL/场景描述与语音媒体的区别 |
| [ViSpeR](multimodal/visper.md) | source_verified | 20 | 元数据和对应源视频、来源重叠 |
| [VLR](multimodal/vlr.md) | source_verified | 8 | 七分包、抽样音轨与其余媒体范围 |
| [VoxCeleb2](multimodal/voxceleb2.md) | existing_cold_reference | 共享包引用 | 不把同一包按数据集重复计数 |
| [VoxCeleb2-mix](multimodal/voxceleb2_mix.md) | source_verified | 14 | 两组 7 分片、官方拼接方法、测试文件排除 |
| [VoxMM](multimodal/voxmm.md) | archived | 8 | VoxMM 与 LRS-VoxMM 的媒体与划分区别 |
| [WildVSR](multimodal/wildvsr.md) | archived | 1 | 抽样视频无音轨的范围 |

程序化对照 18/18 卡的状态、许可字段及有记录的回执文件数和字节数，未发现与所引 00:44 快照的数值不一致。派生链见 [派生数据说明](multimodal/derived.md)；直接核对热端 `muavic_wf_v1`、`muavic_lrs3_build_v1` 和 `avspeech_pilot_v1` 三目录，均只有 `MOVED_TO_COLD.md`。冷端位置及分卷见[冷端与恢复说明](03-storage-and-restore.md)。

## 发现及修正

- 根据当前 `status.json` 和冷端回执，TCD-TIMIT 完成 137/137 个冻结对象；已将历史缺 7 个的旧表述更正，并实查官方目录同时有 speaker-dependent 和 speaker-independent 两组 SCP。
- VLR 为 8 个已登记对象，仅对历史抽样的 part5/part7 视频报告无音轨；不把该抽样外推为整个数据集的媒体实况。
- VoxCeleb2-mix 有 `audio_clean_part_aa`～`ag` 和 `orig_part_aa`～`ag` 两组共 14 片；`.write_test_avsr` 不属于发行分片。两组首片的有限 tar 头部成员和文件 stat 见 [独立证据](multimodal/voxceleb2-mix-headers-20261111.json)。发布者数据卡说明两组分别按名称顺序拼接为两个 tar；本次未拼接、解包完整媒体或重算所有分片哈希。
- VoxCeleb2 和 VideoCC 的 Falcon 目标提取准备事项区分了目标语音配对与识别转写；Real-World AVSE 的建议明确为目标说话人提取和视觉退化评测；VoxMM 的 247/42 划分与 LRS-VoxMM 的 test 抽样分开。MLD-VC 修正了句子与说话人划分措辞。
- 本次冷端实查确认 `/aispeech-cfs` 的 cfs-fuse 配置映射到 TOS 前缀，默认 `tosutil` 对共享包对象只读查询返回 403；不能把挂载可读等同于默认对象客户端具备权限。AVSpeech 迁移分卷抽查到 `train/` 相对路径，恢复目标应按成员路径确定。

## 文档检查与证据边界

后 18 张卡、`derived.md` 与 [冷端说明](03-storage-and-restore.md) 的相对链接共 75 处，逐一检查可解析；冷端文档的恢复命令通过 shell 语法检查。状态、回执、部分目录和 tar 头部为只读核对；历史样本和数量各按页面标注的日期及对应记录引用。没有在本轮对所有媒体重新解码、对全部冷端数据重算哈希或拉取整库。
