# 服务器现存噪声与房间响应：怎样辅助 Falcon 训练

核查时间：2026-11-11。这些路径直接从服务器本地或共享盘读取，与多模态冷端目录分开记录。它们主要用于合成干扰和混响，不能代替目标人的嘴部视频与同步参考语音。

## 噪声材料

`/root/group-shared/voiceprint/data/noise/` 当前可见下列目录。它们可能包含原音频、处理副本、分段目录和处理脚本，不把目录数量当成互不重复的音频数量。

| 目录 | 可见内容或入口 | 面向本项目的用途 |
| --- | --- | --- |
| `AudioSet_balanced_nohuman` | `segments_10s/` | 可核对实际内容后筛选环境声；目录名不代替逐条标签核验 |
| `DEMAND-noise` | `PSTATION/`、`DKITCHEN/`、`NPARK/` 等场景目录及 48k ZIP | 根据场景构造背景噪声；保留原场景与训练/测试拆分 |
| `DNS5_noise_fullband_vad_filtered` | 大量独立 WAV；抽查见下文 | 作为候选噪声，仍需核对内容及来源 |
| `QUT-noise`、`chime4noise` | 各有 `segments_10s/` 和原素材目录 | 场景噪声候选；抽查分段、采样率及与其他集合的重复 |
| `freesound_downloaded_noises` | 两个已过滤来源目录及 `segments_10s/` | 声音事件候选；具体声音及使用范围分别核对 |
| `manually_recorded_noises` | 键盘、椅子、门声等录音和分段目录 | 可补充具体设备与生活声；记录采集条件和使用范围 |
| `musan`、`wham-noise` | 各有原素材及 `segments_10s/` | 候选背景或干扰材料；MUSAN 内不同类别不应一概视为非人声 |
| `rirs_noises` | `rirs_noises.zip` 和 `RIRS_NOISES/` | 房间响应或附带噪声候选；核对每个子集的含义 |

历史 `RESAMPLING_UPDATE.md` 描述的是分段脚本增加 16 kHz 重采样，并不表示整个目录所有原始文件都是 16 kHz。2026-11-11 用 `soundfile.info` 实读两条：

- `manually_recorded_noises/chair_movement_android_9.wav`：**48 kHz、2 通道、PCM16，390.228 秒**。
- `DNS5_noise_fullband_vad_filtered/door_Freesound_validated_257372_44.wav`：**48 kHz、1 通道、PCM16，10 秒**。

这是两条代表音频头，不是对整库采样率、音质或语音含量的全量检查。生成 16 kHz 训练片段时，需要保留源文件、转换参数和 split 关系；不要把原声和分段副本当不同录音。

## 本机房间响应

`/root/data/rir/paired_pse_steam_v4_nearfield_office/` 当前存在 `manifest.jsonl`、`manifest.validated.jsonl`、`scenes/` 和 `splits/`。目录中的历史 `validation_summary.json` 记录 **200,044 个有效场景**；历史 `splits/split_report.json` 记录 train **196,119**、valid **1,969**、test **1,956**，并记录路径和划分检查结果。这些数字来自保存的摘要，本次没有重新遍历全部 200,044 个场景或计算全部文件哈希。

本次实读 validated manifest 第一行，并相对该数据根解析 `scenes/ab0cd771c36eb105.npz`：文件存在，`rirs` 为 **float32 `[5,9121]`**，另有 `direct_path_samples` 和 `metadata_json`；该记录的采样率为 **16 kHz**，场景类型为 corridor。一个 RIR 不是一条语音：它描述房间与声源到麦克风的声学响应，可与目标、干扰、噪声合成混响样本。不同通道/声源在文件中的语义、目标麦克风选择及近远场方式，要按生成配置和元数据确认。

## 做训练清单时怎样组合

1. 从视听源确定目标人的音频与嘴图，记录人物身份和同一时间区间。
2. 分别选其他人的语音、非目标噪声及需要的 RIR，明确采样率、卷积、音量和混音起点。
3. 生成的 `mix`、目标参考和嘴部轨迹要保持同一采样时钟；记录是否把目标定义为干声、混响声或指定麦克风的目标分量。
4. 先确定源说话人、录音及场景划分，再派生多个混音；把重复源或同一个场景的不同处理版本放进不同 split 会影响独立评测。

这些目录是候选来源，当前 `tools/streaming/train.py` 不会自动读取噪声或 RIR 并现场混音；要先生成音频及索引，或另行实现并验证数据管线。

[返回数据全景](README.md)

## 本次一级目录检查

2026-11-11 对上述 10 个噪声目录分别检查**直属第一层**，记录名称类型并抽读最多两条音频头；[完整机器记录](resident-noise-shallow-20261111.json)保留每一行及截断标记。`DEMAND-noise` 第一层有 19 个 ZIP、20 个子目录；`manually_recorded_noises` 第一层有 130 个 WAV。`DNS5_noise_fullband_vad_filtered` 扫描到 25,000 个一级条目后停止，其中 24,998 个 WAV、1 个目录和 1 个 Markdown，因此这里只说明至少看到了这些条目，不把它当该库总数。其他目录的一级多是子目录；本次没有递归统计所有后代或逐条检查声音。
## 路径对应的挂载

2026-11-11 使用 `findmnt -T` 核查：`/root/data` 落在 `/root` 的 GPFS 挂载；`/root/group-shared/voiceprint/data/noise` 和多模态热端落在 `/shared/group-shared` 的 GPFS 挂载；`/aispeech-cfs` 是 `cfs-fuse` 挂载。这里的“本机或共享盘现存”指这些路径目前可直接读取，不代表每个文件都有独立副本，也不对物理介质作推断。
