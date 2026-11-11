# 多模态目录之外的相邻数据

检查时间：2026-11-11。下列数据通过 `/aispeech-cfs` 文件系统挂载直接查看，位于 37 项多模态登记表之外。按需读取时使用列出的现有路径；这里列举目录或代表文件，不把它们与 37 项计数相加。项目训练需要目标人同一时段的混音、目标语音和嘴图；其他形式可用于合成干扰、候选源或真实场景评测。

| 路径 | 实际直接看到 | 潜在用途及接入 |
| --- | --- | --- |
| `/aispeech-cfs/voiceprint/data/MCoRec` | `train_only_central_videos.zip`、`train_without_central_videos.zip`、两份 dev zip；`processed/` 48 个 train tar、3 个 valid tar。首个 tar 头部可见 `.label/.length/.sample_id/.video`，未解包全量 | 本地 README 描述多人会话和目标人视角；训练部分有近讲音频/个人视频，dev 只提供中心视角，需单独核对通道、时间同步和目标参考。README 条款限定非商业研究与授权，原始可识别素材和分享需按协议处理。 |
| `/aispeech-cfs/voiceprint/data/TalkVid` | `data/` 中 `filtered_video_clips.json` 和带 captions 的 JSON；`videos/` 有 MP4 和配套 info.json | 抽查一段视频为 3840×2160、约 23.976fps，48kHz 双声道音轨、约 602.461 秒。可作视听来源候选；目标人嘴图、干净目标和同步须逐条建立；其与 `multimodal/talkvid_meta` 的重叠先核对。 |
| `/aispeech-cfs/avatar/DATA/AVSpeech` | `videos/` 中有 MP4，另有 `video_clips/`、`video_person/`、`video_noperson/` 和 `copy_manifest.csv` | 抽查一个 MP4 为 1280×720、25fps、16kHz 单声道、3.44 秒。可查目标清晰度及嘴部后构造混音，注意与其他 AVSpeech 源/派生内容重复。 |
| `/aispeech-cfs/avatar/DATA/MEAD` | `videos/` 有 MP4，另有 `video_clips/`、`video_person/`、`copy_manifest.csv` | 抽查一个 MP4 为 1920×1080、25fps、16kHz 单声道、3.8 秒。可作表情/说话风格候选或专项评测；未检查全量说话人、许可或配对。 |
| `/aispeech-cfs/voiceprint/data/AISHELL-5` | `train.tar.gz`、`Dev.tar.gz`、`Eval1.tar.gz`、`Eval2.tar.gz`、`noise.tar.gz` | Dev 包有界前缀看到 `Dev/001/DA03.wav` 和 `.TextGrid`；可作为场景、标注和干扰声音候选，目标人视频及同步干净语音尚未核实。 |
| `/aispeech-cfs/voiceprint/data/AliMeeting` | `Train_Ali_far.tar.gz`、`Train_Ali_near.tar.gz`、`Eval_Ali.tar.gz`、`Test_Ali.tar.gz` | 有界读取两个训练包开头：near 为 `Train_Ali_near/audio_dir/...SPK....wav`，far 为 `Train_Ali_far/audio_dir/...MS....wav`；可核查近远场音频对齐并作为会议混响/干扰素材，当前抽查未见目标人视频。 |
| `/aispeech-cfs/voiceprint/data/AMI` | 8 个 `Array1-*.tar.gz`、`headset.tar.gz`、`ami_manual_1.6.1.tar.gz` | 标注包前部看到 XML 摘要等条目，可进一步核对近讲与阵列通道的时序与授权；本次没找到视频并不说明该数据集所有发行内容均无视频。 |
| `/aispeech-cfs/voiceprint/data/AISHELL-4`、`CHiME6`、`Dipco`、`NotSoFar-1` | AISHELL-4 的 tr/test 压缩包、CHiME6 的 train/dev/eval 及转写、Dipco 的 `.tgz`；NotSoFar 存在 `benchmark-datasets` 的 train/dev/eval 子目录 | 可分别作为语音干扰、场景和会议评测素材候选；数据清单、音视频配对、目标参考和使用条款需要逐项核对。NotSoFar 本地说明将该公开数据限定于挑战活动用途。 |

## 检查范围

根目录及上述文件名使用 `os.scandir` 浅层列示；MCoRec 的 51 个 processed tar 是目录文件计数，首片仅读取前 16 MiB 的前 20 个成员。AliMeeting、AISHELL-5 和 AMI 的压缩包仅读取前 8 MiB，不能据此确定全包内容。三个 MP4 各用 `ffprobe` 查看一个样本的容器流信息，未进行全库解码、同步或音质检查；这些目录可能与本项目内其他数据有来源重叠。未下载、解包或修改数据。

[返回数据全景](README.md)

另外完整读取两个 `copy_manifest.csv`：AVSpeech 有 **9,972 行**、MEAD 有 **1,824 行**，各自 `status` 均为 `ok` 且目标路径不重复。这是历史复制清单的行数与状态，本次没有逐行验证这些目标现在仍存在，也不把它们与多模态目录中同源记录相加。
