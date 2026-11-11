# mandarin_avsr：中文目标说话人语音分离数据（`mandarin_avsr`）

**用途定位：A｜分离训练优先。** 训练索引列有 1～5 人混音、目标音频及嘴部路径；优先核对媒体、路径转换和同一目标人。cv/tt 保持为验证/测试，先确认授权。

## 当前目录、状态与许可

- 数据说明：Mandarin AVSS mixtures
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/mandarin_avsr/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/mandarin_avsr/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/mandarin_avsr/SHA256SUMS`。
- 机器记录：登记预期对象 9；状态已完成对象 9；状态冷端对象 9；冷端收据对象 9；冷端收据字节 49,977,304,020。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`other`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/playwithmino/mandarin_avsr](https://huggingface.co/datasets/playwithmino/mandarin_avsr)。
- 当前清单开头的对象示例：`raw/cv.tar.gz`、`raw/preview.tar.gz`、`raw/tr_1mix.tar.gz`、`raw/tr_2mix.tar.gz`、`raw/tr_3mix.tar.gz`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：普通话
- 标注与格式：代表样本含 mix.wav、label.wav、video.mp4、info.json；label.wav 是干净目标音频，不是文字标签。
- 训练/测试划分：训练 tr/1mix～5mix 各8,000索引；cv 与 tt 为验证、测试，当前未逐卷统计其样本数。
- **2026-11-11 直接复核**：`tr_all.tar.gz` 的 `mix.json` 和 `s1.json` 各 40,000 行；1、2、3、4、5 人混合各 8,000 行。配对文件名及长度错配均为 0；长度 32,000～260,480 点，按 16 kHz 合计约 92.98 小时。40,000 条嘴部路径均为 `.npz`。
- 预览 `tt_3mix_000001`：混音与 `label.wav` 均为 16 kHz 单声道、170,880 点；通过可寻址的单个预览视频成员复核，视频为 MPEG-4、384×384、25 fps、267 帧、10.68 秒，首帧解码成功；音频 170,880 点、16 kHz 同为 10.68 秒。原非可寻址管道探测的宽高为 0；该 MP4 的 `moov` 位于 `mdat` 之后，直接管道首帧解码失败。`info.json` 记录目标人、2 个干扰、重叠与 SIR。`label.wav` 是目标音频。未逐条解包验证全量媒体。
- 既有下一步：优先用于中文重叠语音专项；如做目标识别，先确认来源并建立文字映射，保持源数据拆分，明确输入是混合声还是干净声。

## 2026-11-11 验证、测试包实际成员

对 `cv.tar.gz`、`tt.tar.gz`、`tr_1mix.tar.gz` 各读取前 10 个 tar 成员，分别看到 `cv/3mix/audio/s1/*.wav`、`tt/3mix/audio/s1/*.wav` 和 `tr/1mix/audio/s1/*.wav`。这证明这些包的前部有对应划分的目标音频文件；没有遍历归档后部，不能由此计算 cv/tt 总条数或确认所有嘴图均齐。具体成员名、大小和脚本见 [归档前部检查](../mandarin-first-headers-20261111.json)。

## 对 Falcon 的准备事项

- 2026-10-16 盘点未遍历所有压缩流，未验证所有 40,000 个路径与媒体齐全。
- 若后续确认 lips_ 片段来自 Chinese-LiPS，可再尝试关联其 TEXT；在此之前不能假定二者必然对应。关联后仍须防止同一源片段跨训练/测试。
- 这是分离数据，不因目录名含 avsr 就自动成为可直接跑识别训练的数据。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。
- [2026-11-11 两份索引、预览和压缩包哈希检查](mandarin-avsr-index-check-20261111.json)：40,000 行为索引计数；两个压缩包哈希与冷端清单吻合。
- [同一预览视频首帧解码复核](mandarin-avsr-preview-decode-20261111.json)：直接管道失败；提取 414,538 字节单个成员为可寻址临时文件后首帧解码成功。检查的是一条预览的首帧，不是所有样本或全段解码。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
