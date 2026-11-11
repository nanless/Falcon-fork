# LRS Combined（`lrs_combined`）

**用途定位：C｜尚未物化。** LRS2/LRS3 组合是派生名称，不是额外可加和的独立数据；生成前需明确源划分与去重。

## 当前目录、状态与许可

- 数据说明：LRS2 与 LRS3 的本地派生合并集。
- 登记状态：`derived_not_materialized`；检查时间 2026-11-11 00:44（北京时间）。状态描述登记范围，并非 Falcon 训练清单验收。
- 服务器暂存目录：`/root/group-shared/voiceprint/data/multimodal/lrs_combined/`；状态文件：该目录下 `status.json`。
- 当前为 `derived_not_materialized`：没有独立冷端训练载荷，按需由已确认的 LRS2、LRS3 派生。
- 本地许可记录：`inherits source terms`；镜像或派生集的标识须与原素材条款一起核对。
- 来源：[https://www.robots.ox.ac.uk/~vgg/data/lip_reading/](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/)。

## 内容、规模与格式

以下样本/配对/说话人数字来自 2026-10-16 的原始媒体或索引盘点；当前登记数量见上节。两者是不同时间、不同统计口径。

- 语言：英语
- 标注与格式：继承LRS2/LRS3标签
- 训练/测试划分：没有独立物化及独立拆分，沿原始 LRS2/LRS3 定义产生。
- raw为空，状态 derived_not_materialized。此名称表示按需合并来源，不是独立发行，也不是另存的一套可直接读取媒体。

## 对 Falcon 的准备事项

- 须先固定两数据集的原始视频ID、官方划分与排除规则，再生成清单。不要把目录存在理解为合并训练数据已经建好。
- 面向目标说话人提取：明确 `混合音频 → 对应目标干净音频 → 同一目标口型` 三者映射、拆分和原始说话人/视频 ID；当前数据准备完成度以实际新清单与抽检为准。

## 证据范围

- 00:44 登记/状态快照：[本轮只读清单](snapshot-20261111-0044.json)；00:46 独立快照：[项目机器清单](../catalog-snapshot-20261111.json)；历史样本和数据说明：[2026-10-16 选录](historical-20261016-selected.json)。
- 2026-11-11 读取了本库登记、状态及冷端收据和校验清单开头；样本内容使用上述标注日期的直接检查或历史盘点。


[2026-11-11 冷端目录第一层直查](../cold-direct-a-m.md)。

[返回多模态总览](../02-multimodal-datasets.md)
