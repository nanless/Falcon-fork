# 数据目录盘点口径与核验时间

这份说明用于阅读同目录的[登记快照](catalog-snapshot-20261111.json)、[项目训练与测试数据](01-project-data.md)及逐库资料。计数分别描述登记行、索引行、文件、媒体片段、物理混音与目标方向，使用前先确认对应分母。

## 采集时间与覆盖

- 多模态登记快照 `captured_at` 为 **2026-11-11 00:46:16.575404 +08:00**，相当于 **2026-11-10 16:46:16.575404 UTC**；它读取当时的 registry、各数据集状态及可见归档回执字段。
- 冷端顶层登记外对象于 **2026-11-11 00:49:10.036388 +08:00** 另外记录，与主快照并非同一瞬间。对象冷存储通过 `/aispeech-cfs/voiceprint/data/multimodal` 挂载；`/root/group-shared/voiceprint/data/multimodal` 是登记表所用的热端根。
- 逐项比对 registry 与 JSON：**37 个登记 slug、37 条快照，无遗漏或重复**。登记表中的名称、描述、来源、许可字段、初始状态及既有冷端引用与快照字段一致。这是登记覆盖检查，不是 37 项媒体完整性检查。
- 热端根中除 37 个登记名之外有 **7 个顶层对象**：`_catalog`、`_download`、`AUTHORIZATION_STATUS_zh.md`、`DATASET_CATALOG_zh.md`、`existing_sources.json`、`multimodal`、`README_zh.md`。它们包含目录和管理文件，不可按 7 个新数据集计数。
- 冷端根中除登记名之外有 **5 个顶层对象**：`_catalog`、`_local_overflow_20261102`、`AVSpeech_Vox2_LRS2_LRS3_AudioSet.tar`、`realssa_real`、`realssa_sim`。共用大 tar 中记录了多个数据来源，但该 tar 是一个物理对象；内容及完成范围以各自证据为准。
- 快照列出的 **32 份历史回执路径**在 2026-11-11 的本次复核时均存在；快照字段与当时读取到的回执字段不能替代对每个大文件重新计算哈希。动态写入的回执会在快照之后继续变化。

## 状态、回执和时间字段

`status.json` 是数据集在某一时刻的状态记录；归档回执描述一次归档或递增归档的记录。两者可能由不同进程在不同时间更新，文件数与字节数不能只凭两条记录一起出现，就解释为同一时刻的最终总数。

以 MultiVSR 的主快照为例：状态为 `downloading`，状态记录的 `verified_files=76003`、`verified_bytes=10573315866489`；快照中的历史 `incremental_receipt.json` 记录 `file_count=75991`、`total_bytes=10571152579195`、`updated_at=2026-11-10T16:41:28Z`。回执与状态不在同一更新点，不能将两组值相加或取其一作为未加限定的全量结果。复核时数据仍在变化，使用前应重新读取当前状态、回执与目录证据。

部分状态源文件同时含数值 `updated_at` 和字符串 `updated_at_utc`。这两个字段在 RoomReader、CANDOR、MaFi、VideoCC 的源记录中存在不一致；快照分别保留了源值，没有把其中一个改写成另一个。解释日期时应注明引用的是哪个字段及时间类型，并回到相应状态文件与回执核对事件。

## 项目索引与媒体检查分层

[项目数据文档](01-project-data.md)中，LRS2 单方向、双方向及 RealSSA-Sim 两种通道方案的**完整 JSON 索引**被读取，用于统计行数、唯一混音/目标/嘴部路径、近场自引用和索引采样点时长。这是索引全行统计。

文件可读性采用另一范围：每个存在的索引划分等距抽查 30 行，检查其中混音、目标、嘴图及 near 文件是否存在；再实读每个划分首条音频头与嘴图张量。RealSSA-Real 按目录及 54 行元数据计数，并抽读一条嘴图和一条注册 WAV。RealSSA-Sim 共享目录按划分数了 202、19、19 个 tar。以上不是逐个媒体解码、全库内容去重或逐文件哈希；数据集级来源、内容和恢复条件应结合逐库证据。

## 避免重复计算

- LRS2 双方向索引把一段混音分别与两个目标说话人配对，40,000 个训练方向对应 20,000 个不同混音；方向小时不能当作独立混音小时。
- RealSSA-Sim 的四通道平均与第 0 通道目录是同一批 9,169 条测试混音的不同转换，不能相加成 18,338 条独立测试；两个目录可共享嘴部硬链接。
- 原始媒体、提取音频、重编码视频、嘴部裁剪、混音、训练索引、软链接和归档副本是不同处理层，不默认代表不同原始样本。
- `du` 可见结果取决于命令是否同时遍历硬链接，路径字节总和、去重内容字节和实际物理占用也不是同一指标。容量、文件数量和训练样本数量分别记录。
- 注册语音、模型权重、脚本、空状态目录和管理报告不计为训练媒体；许可、标注、目标语音与嘴图对齐条件须按数据集独立核实。

## 敏感字段检查

对快照做了已知格式模式扫描：GitHub token、Hugging Face token、AWS access-key 格式、私钥标记和常见带签名查询参数均未命中。该检查覆盖列出的模式，不等同于对任意形式的秘密或私人数据进行全面识别。快照记录的数据来源、时间及状态应以本页注明的范围理解。

## 以后如何刷新登记快照

只读工具 [snapshot_catalog.py](snapshot_catalog.py) 重新读取登记表、各状态文件和冷端回执，输出到新的 JSON 文件。它记录采集开始与结束时间；采集期间仍可能有数据集状态变化。文件名应保留日期或任务标识，以便与历史快照比较。

```bash
cd /root/code/github_repos/Falcon-fork
OUT=$(mktemp -d /root/tmp/falcon-data-inventory-XXXXXX)
python3 docs/data_catalog/snapshot_catalog.py --output "$OUT/catalog.json"
```

快照工具只读取登记、状态、回执及目录入口，不遍历所有媒体，不改变下载进程和数据本体。历史文档仍以各自标注时间与证据为准。

## 直接读取挂载目录

[第二次直接清单](aispeech-cfs-direct-listing-20261111-v2.json) 逐个列出 `/aispeech-cfs/voiceprint/data/multimodal` 的顶层目录及其 `raw/` 内容；每个目录最多读取 50 项，`truncated` 标明未列全，`omitted_names` 记录因名称含敏感模式而省略的项。第一版 [直接清单](aispeech-cfs-direct-listing-20261111.json) 保留原貌，未记录每层省略计数。[A–M 直接检查](cold-direct-a-m.md)和[N–Z 直接检查](direct-cold-n-z-20261111.md)包含更深入的目录与包内抽查。

以后可用只读脚本重查实际挂载：

```bash
cd /root/code/github_repos/Falcon-fork
OUT=$(mktemp -d /root/tmp/falcon-cold-listing-XXXXXX)
python3 docs/data_catalog/list_mounted_data.py --limit 50 --output "$OUT/listing.json"
```

目录条目上限限制输出体量，不等同全量媒体统计；每个数据集已另行记录实际遍历或 ZIP 中央目录检查的范围。
