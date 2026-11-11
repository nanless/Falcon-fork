# VoxMM / LRS-VoxMM（`voxmm`）

**用途定位：B｜长对话评测。** LRS-VoxMM 含音视频文本与干扰条件，可研究长时对齐和重叠；VoxMM 人脸轨迹并不代表所有目标全程可见。

## 当前目录、状态与许可

- 数据说明：自然长对话丰富转写、音频、人脸轨迹及含视频的 LRS-VoxMM 基准。
- 登记状态：`archived`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/voxmm/`；状态文件：该目录下 `status.json`。
- 冷端载荷：`/aispeech-cfs/voiceprint/data/multimodal/voxmm/raw/`；校验清单：`/aispeech-cfs/voiceprint/data/multimodal/voxmm/SHA256SUMS`。
- 机器记录：登记预期对象 8；状态核验字节 43,846,182,092；冷端收据对象 8；冷端收据字节 43,846,182,092。文件/分片数不是语音条数；动态任务数字以该次快照为准。
- 本地许可记录：`CC BY 4.0`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://mmai.io/projects/voxmm/](https://mmai.io/projects/voxmm/)。
- 当前清单开头的对象示例：`raw/lrs_voxmm/3dist.tar.gz`、`raw/lrs_voxmm/noise.tar.gz`、`raw/lrs_voxmm/original.tar.gz`、`raw/voxmm/vmm_facetrack.zip`、`raw/voxmm/vmm_interjection_list.txt`；这是有界示例，不是文件类型全量统计。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语（抽查）；未全量语言统计
- 标注与格式：VoxMM有289份音频、289份转写JSON、289份人脸轨迹JSON和train247/test42；LRS-VoxMM有原版/三干扰源/噪声包
- 训练/测试划分：VoxMM 的 split 文件历史盘点为 train 247 / test 42；LRS-VoxMM 抽样见 test，完整来源划分需单独核实。接入前按原始说话人、视频和会话建立隔离。
- 冷端8文件、43846182092字节。VoxMM四ZIP含音频/metadata/facetrack/split；没有独立源视频ZIP。LRS-VoxMM original与3dist包的有界抽样明确见test目录下成套MP4/WAV/TXT，TXT有词时间。

## 对 Falcon 的准备事项

- train/test划分需要照搬；原版与不同干扰版本为同源，不能跨训练/测试分开。LRS-VoxMM 的 2026-10-16 盘点抽查只见test，不据包名假设另有独立训练集。下载8包完整不等于所有VoxMM原YouTube视频已到位。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。

[返回多模态总览](../02-multimodal-datasets.md)

## 2026-11-11 挂载目录直接检查

递归遍历 `raw/` 得 8 文件、43,846,182,092 字节：4 ZIP、3 tar.gz 和 1 TXT。VoxMM 与 LRS-VoxMM 的载荷和划分需分别检查，包数不能作为对话数。 详细范围见[直接文件核查](../direct-cold-n-z-20261111.md)。
