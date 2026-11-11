# Real-World AVSE 2026：中文真实重叠及视觉退化测试数据（`real_world_avse`）

**用途定位：B｜保留开发/测试。** 优先作中文真实重叠、遮挡和音画偏移评测；目标 WAV 只覆盖部分条目，不能把全部文字/视频计为有参考源的 SI-SNR 样本。

## 当前目录、状态与许可

- 数据说明：Real-World AVSE Challenge dev+test (ISCSLP 2026)
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/real_world_avse/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/real_world_avse/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/real_world_avse/SHA256SUMS`。
- 机器记录：登记预期对象 4；状态已完成对象 4；状态冷端对象 4；冷端收据对象 4；冷端收据字节 9,350,891,169。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-NC-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/Rezzsl/Real-World_AVSE_Challenge_dev_and_test_sets_ISCSLP_2026](https://huggingface.co/datasets/Rezzsl/Real-World_AVSE_Challenge_dev_and_test_sets_ISCSLP_2026)。
- 当前清单开头的对象示例：`raw/Testing_set_GT.zip`、`raw/assets/challenge_icon.png`、`raw/track1_dev.zip`、`raw/track2_dev.zip`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：普通话
- 标注与格式：实际 ZIP 内有 s1.txt、s2.txt 中文逐句文字，不只是语音分离标签。
- 训练/测试划分：track1_dev、track2_dev、Testing_set_GT 是开发/测试包；逐目标有参考音频的数量只覆盖其中一部分。
- 冷端共 4 文件、9,350,891,169 字节，核心是 track1_dev.zip、track2_dev.zip、Testing_set_GT.zip，另有展示图。
- 两份开发包分别有 4,284、5,250 个目标文字文件；测试包有 18,396 个目标文字文件。三包每条目标文字均有同名目标 MP4。
- 只有 1,800、2,196、7,812 个文字项同时有同名目标 WAV；不能说每个文字项都已配干净目标音轨。
- 抽样混合音频为 16 kHz、单声道，目标人脸视频为 256×256、25 帧/秒。
- 发布者明确这些是开发与测试数据；第二赛道还包含遮挡、低分辨率、卡帧与音视频不同步等退化。
- 既有下一步：作为中文目标说话人提取和视觉失真评测候选；保留官方划分，按可用目标参考与评价指标制作评测清单后再跑基线。

## 对 Falcon 的准备事项

- 同一源片段在不同混合、重混合和视觉退化版本中出现，不可简单相加为独立样本量。
- 目标视频对应的是 s1 还是 s2、输入混合声和目标文字的关联，仍需按清单逐项确认。
- 开发集和测试集不能在仍用它们报告成绩的同时并入训练。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

`raw/` 实际 4 文件、9,350,891,169 字节；其中 `track1_dev.zip` 中央目录见 4,284 个目标 MP4、4,284 个 TXT、3,942 个 WAV 和 4,284 个 PKL。首／中／末三个视频样本均为 256×256、25 fps，仅含视频流；音频要取对应 WAV。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
