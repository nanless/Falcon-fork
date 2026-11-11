# 多模态数据卡交叉复核记录

复核日期：2026-11-11。覆盖按登记 slug 字母顺序排列的前 19 张数据卡：对照 00:44 状态与回执快照，逐卡检查登记状态、已记录许可字段、预期/完成/核验/回执数字、冷热目录和校验清单存在性、卡片相对链接，并将用途描述与 2026-10-16 历史资料对照。数据集源状态或许可记录不是训练授权结论。

| 数据卡 | 本轮复核要点 |
| --- | --- |
| [ASPIRE](multimodal/aspire.md) | 状态、1 个对象和字节数一致；WAV/MOV 各 5,940 是文件计数，尚需逐条核对配对与同步。 |
| [AudioSet](multimodal/audioset.md) | 登记引用共享大包；包名不证明内部版本、媒体总量和可用性。 |
| [AudioSet Strong](multimodal/audioset_strong.md) | 4 个 TSV 的状态与回执一致；历史记录中的训练 TSV 截断不能用归档状态抹去。 |
| [AVCocktail](multimodal/avcocktail.md) | 206 个对象与状态一致；51 个源视频的三类切段不是 153 个独立场景。 |
| [AVSpeech](multimodal/avspeech.md) | `merged_complete` 属历史状态；不同日期清单和 gapfill、冷端迁移件不能直接相加。 |
| [AVYT](multimodal/avyt.md) | 1,320 个发行对象状态及回执一致；原始素材去重、目标音轨、划分和许可需核对。 |
| [CANDOR](multimodal/candor.md) | 166 个 ZIP 与回执字节数一致；单个分卷抽样不能乘以分卷数推断全量会话数。 |
| [Chinese-LiPS](multimodal/chinese_lips.md) | 19 个发行对象、历史的 36,208 行对应与 ID 集合记录明确；目标音质需实听，许可记录含非商业。 |
| [CUAVE partial](multimodal/cuave_partial.md) | 1 个派生对象和字节数一致；MAT 状态标签、波形及单人目标关系需进一步核对。 |
| [Darknsu LRS3](multimodal/darknsu_lrs3.md) | 2 个压缩包记录一致；局部媒体抽样不是全量成员和与 LRS3 官方 ID 的全量配对。 |
| [Ellipsis LRS3](multimodal/ellipsis_lrs3.md) | 34 个对象记录一致；口型与文字无音频是代表分片抽查，来源字段的差异保留待核。 |
| [FaceStar](multimodal/facestar.md) | 115 个对象记录一致；音视频配对限已核对会话，记录非商业许可字段。 |
| [GRID](multimodal/grid.md) | 36 个归档对象记录一致；33 位有视频的说话人与 34 位音频/标签人数区分。 |
| [Lombard GRID](multimodal/lombard_grid.md) | 5 个对象记录一致；5,390 个音频与 5,361 个对齐文件相差 29，具体缺失项须按 ID 核对。 |
| [LRS2](multimodal/lrs2.md) | 登记是共用冷包引用；项目训练索引见[项目数据](01-project-data.md)，不把整个联合包计作 LRS2。 |
| [LRS3](multimodal/lrs3.md) | 登记是共用冷包引用；派生集和 AVYT/MuAViC 的共同源需去重。 |
| [LRS Combined](multimodal/lrs_combined.md) | 状态为 `derived_not_materialized`，不是额外独立媒体。 |
| [MaFi](multimodal/mafi.md) | 24 个对象状态与回执一致；代表视频无声，许可记录 `unverified`。 |
| [Mandarin AVSR](multimodal/mandarin_avsr.md) | 9 个对象记录一致；40,000 行为索引数，路径和长度逐行检查不等于全量媒体解码。 |

## 修正

- ASPIRE：将等数 WAV/MOV 直接说成 5,940 个已验证文件对改为文件计数及待核对的配对关系。
- Lombard GRID：将音频数与音素对齐数相差 29，改为说明这是数量差，缺失项需按标识确认；逐条音视频匹配列为待核事项。
- FaceStar、Ellipsis LRS3：将音视频配对、音轨缺失限定为抽查范围。
- Chinese-LiPS：已有音视频 ID 配对，但未据此断言目标音质均为干净语音。
- Mandarin AVSR：把训练集的说法限定为索引中列有的混音、目标和嘴图路径；CANDOR 与 LRS3 的历史检查说明标明 2026-10-16。

## Mandarin AVSR 单例复核

[索引检查](multimodal/mandarin-avsr-index-check-20261111.json)记录 `mix.json` 与 `s1.json` 各 40,000 行，1–5 人混音各 8,000 行，文件名与长度错配均为 0；压缩包哈希与冷端清单一致。预览 `tt_3mix_000001` 只有一个代表样本，并非全库媒体检验。

[单例视频解码证据](multimodal/mandarin-avsr-preview-decode-20261111.json)记录：从 `preview.tar.gz` 用非可寻址管道直接送 ffmpeg 时首帧解码报 `partial file`；容器 `moov` 在 `mdat` 后。另将此 414,538 字节单个成员提取到临时普通文件后，ffprobe 显示 MPEG-4、384×384、25 fps、267 帧、10.68 秒，ffmpeg 首帧解码成功，音频 170,880 点 / 16 kHz 也是 10.68 秒。临时文件已清理；该操作没有解包整套数据。

## 项目数据总览复核

[数据入口 README](README.md) 中的 LRS2 单方向 20,000/5,000/3,000、双方向 40,000/10,000/6,000 及其 20,000/5,000/3,000 唯一混音、RealSSA-Sim 两种通道各 9,169、RealSSA-Real 54 段均与项目索引或元数据核对一致。候选排序是基于任务适配的使用建议：Mandarin 索引和单条预览、VoxCeleb2-mix 首片成员、AVCocktail 和 AVYT 的历史代表样本不代表全库配对已验收。

本次未遍历或重算这些大型数据集的全部媒体哈希，也未给出逐条音画同步质量结论；登记快照、历史观察、抽样证据和本次单例解码的时间与范围各自记录。
