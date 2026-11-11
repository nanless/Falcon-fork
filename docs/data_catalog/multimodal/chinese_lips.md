# Chinese-LiPS：中文音频、口型和逐句文字（`chinese_lips`）

**用途定位：A｜中文合成混音候选。** 处理版训练集有按 ID 对应的音频与嘴部视频；核对目标音质后可按说话人合成 1～N 人混音，官方验证、测试单独保留。许可含非商业条件。

## 当前目录、状态与许可

- 数据说明：Chinese-LiPS Mandarin AVSR
- 登记状态：`source_verified`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/chinese_lips/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/chinese_lips/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/chinese_lips/SHA256SUMS`。
- 机器记录：登记预期对象 19；状态已完成对象 19；状态冷端对象 19；冷端收据对象 19；冷端收据字节 93,069,232,024。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC-BY-NC-SA-4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://huggingface.co/datasets/BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS)；[https://huggingface.co/datasets/BAAI/Chinese-LiPS/tree/main](https://huggingface.co/datasets/BAAI/Chinese-LiPS/tree/main)。
- 当前清单开头的对象示例：`raw/image.zip`、`raw/meta_all.csv`、`raw/meta_test.csv`、`raw/meta_test.json`、`raw/meta_train.csv`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：普通话
- 标注与格式：CSV 的 TEXT 列是中文逐句文字，ID 与已处理音视频名称可直接关联。
- 训练/测试划分：processed train/val/test 历史全ID对应数量30,341/1,959/3,908；原始分卷和处理包分开。
- 冷端 19 个文件，共 93,069,232,024 字节。其中 processed_train.zip、processed_val.zip、processed_test.zip 是独立可读 ZIP。
- 三个已处理包分别有 30,341、1,959、3,908 对 MP4 和 WAV；对应 CSV 行数完全相同，总计 36,208 条。
- 2026-10-16 盘点对三个拆分全部核对 ID：每条文字都能找到同 ID 音频和视频，文字无空值；拆分之间样本 ID、说话人编号前缀都无交叉。
- 样本口型视频为 96×96、25 帧/秒；样本音频为 16 kHz、单声道。三个拆分按 ID 前缀分别涉及 175、11、21 个说话人编号。
- 原始 train.z01 至 train.z08 共 8 卷，但冷端没有 train.zip；2026-10-16 盘点查看的发布页也未列出这个终卷。已处理的训练 ZIP 不依赖这些原始分卷。
- 既有下一步：新增中文训练优先项：使用官方 train/valid/test 拆分制作清单，先保持现有测试隔离，再核对中文模型及音视频同步。

## 2026-11-11 实际测试样本

从 `processed_test.zip` 读取同 stem 的 `130_42_M_TY_064.wav` 和 `.mp4`：音频为 16 kHz 单声道、12.921 秒；视频为 96×96、25 fps、325 帧、13.000 秒，未含音频轨。音频首秒和视频首帧解码均成功。此为一条例子，容器时长相差约 79 ms；该差值本身不能确定起始时刻与嘴部同步误差。使用前按目标片段时间核对，构造干扰与混音并保持原划分。

[实际媒体探针](../archive-media-probe-20261111.json)。

## 对 Falcon 的准备事项

- 2026-10-16 盘点验证的是所有 ID 的存在和代表媒体头，不是对全部媒体逐帧解码、听写或同步验收。
- 原始训练分卷不能仅因 8 卷都在而宣称可完整解压。无需为使用已处理包而重复下载。
- 当前 MMS-LLaMA 的英语配置不能默认直接适用中文；仍需检查文字规范、分词器、模型语言能力和训练清单。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

## 2026-11-11 冷端归档直查

直接读取冷端 processed 三个 ZIP 的中央目录：train/val/test 分别有 30,341/1,959/3,908 个 WAV 文件名与同名 MP4 一一对应，三组均无单侧独有名称及重复名称。该结果检查成员名称，未逐条解码媒体或验证同步。 [逐项统计与执行范围](../cold-zip-pair-audit-20261111.json)；[只读复核脚本](../cold_zip_pair_audit.py)。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
