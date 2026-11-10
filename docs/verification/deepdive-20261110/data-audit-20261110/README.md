# 2026-11-10 历史数据二轮复核

## 范围与方法

本次对 `docs/verification/` 当时的 211 个 `.csv`、`.json`、`.log`、`.txt` 文件逐文件记录路径、字节数和 SHA-256，并检查 UTF-8 解码；明细见 [清单](inventory.json)。范围包括本轮新增的 `deepdive-20261110` 原证据；不包括源码、Markdown、二进制和 Python 缓存。验证目录外还核查了 `Experiments/checkpoint/Falcon/best_k_models.json` 与 `studio/web/vendor/mediapipe/manifest.json` 的结构。发布权重的完整训练来源仍待确认。

全部 55 个 JSON 通过 Python JSON 解析器以及拒绝 `NaN`、`Infinity` 常量的严格常量检查；26 个 CSV 均能读取，行宽一致。对含数值指标的 CSV，先去除首列为 `avg` 或 `std` 的汇总行，再逐列统计有限值与均值；原有 `avg` 汇总行与重算值差值均不超过 0.00011，统计详见 [逐表重算](csv-recalculation.json)。没有发现以 `snt_id`、`segment_id` 命名的 ID 列存在重复。P2B 摘要、三份 DNSMOS 摘要和两份 counts JSON 与各自 CSV 共做了 70 项均值或计数对照，均在误差阈值内；逐项见[摘要交叉核对](summary-crosscheck.json)。比较不同版本时还需核对数据与评测条件。

117 个日志逐份检查首部和末尾，分类、终态提示及 SHA-256 见[逐文件日志表](log-review.tsv)；[日志摘录目录](log-catalog.json)另外收集了 1,222 条匹配运行条件、结果或异常关键词的行。第一轮仅抽看部分命中上下文；后续分三组检查全部日志的主要内容和终态，个别长列表未逐项核对，具体深度见各组记录。各组记录链接至原始日志。真实失败包括 RealSSA-Real 早期 `full`、`full2` 各 54 段 NameError，随后 `full3` 又有已记录的窗口起点错误；这三次均不是有效的完整结果；后续 full4 的结果另行记录。另有预期异常测试和旧版测试失败，须结合各日志与日期判断。`P2B-big-full.log` 仅有 100/3000 进度而无完整汇总；旧版 `P2E-realssa-sim-ch0-eval.log` 明确推迟，2026-10-30 的另一份日志及 CSV 才是后续完成结果；`train-sys-ddp-bucket-hang.log` 末尾停在 gather 前。

## 重要数值复算

- LRS2 公开权重：3,000 条，SI-SNRi 7.029101、SDRi 7.712925。
- LRS2 ep99：3,000 条，SI-SNRi 8.726148、SDRi 9.121848；ep124：3,000 条，SI-SNRi 8.858192、SDRi 9.269574。
- RealSSA-Sim 四通道均值：SI-SNRi 9,169 个有效值、均值 7.774307；SDRi 与 PESQ 各 9,166 个有效值，SDRi 均值 8.479395。ch0 对应值分别为 7.976709 / 8.452396，同样具有各指标不同的有效数。
- RealSSA-Real full4：54 段，CSV 重算 OVRL 2.79405185、P808 3.3365 左右；以原 summary 的四位小数 OVRL 2.7941 / P808 3.3365 报告。论文对应训练条件、本地 checkpoint 身份和实现细节见[当前复现状态](../../../STATUS.md)及[评测说明](../../../understanding/20261110-evaluation-and-apps.md)。

上述数字根据仓库中保存的结果文件复算；模型推理和音频指标沿用历史运行记录。`csv-recalculation.json` 里的空白、无法解析和非有限值统一列作 excluded；应回到逐句文件与日志区分具体原因。历史证据原件没有覆盖。复核脚本为 [逐表重算](recompute_csv.py) 和 [摘要比对](check_summaries.py)；每次执行在 `/root/tmp` 下新建独立目录保存结果，不改历史证据。

2026-11-10 对照当前 LRS2 测试索引，公开权重两份 CSV、ep86、ep99 两份 CSV 和 ep124 共六份逐句表均为 3,000 个不同 ID，集合与当前索引一致；各文件和索引哈希见 [六份 LRS2 身份核对](lrs2-six-id-sets.json)。核对范围包括当前索引及文件名集合。对应[核对脚本](check_lrs2_ids.py)每次写入独立临时目录。

项目验证目录之外另有 3 份 JSON 与 3 份 CSV，已记录文件身份、行数和相关统计，见[验证目录外的资料清单](outside-verification-inventory.json)。其中训练 smoke 的 9 行记录训练过程。

日志再复核：已对排序第 1–39 份完整逐行阅读，并逐项记录实验条件、终态和使用边界，见[第 1–39 份逐行语义审阅](log-semantic-review-001-039.tsv)。

[第 79–117 份日志语义审阅](log-semantic-079-117.md)记录了该组条件、终态与局限；其中第 86 份有一行 33,312 字符的参数键列表未逐名核对，详见分组记录。

[第 40–78 份日志语义审阅](../log-review-040-078.md)逐份列出运行条件、终态和边界。三组记录共同覆盖盘点时的 117 份日志；各组注明未逐项核对的长列表和其他限制。

第 40–78 份日志的[补读台账](log-complete-review-040-078.tsv)说明逐文件阅读深度；第 52、55 份跳过重复动态进度条，正文和转换进度已读。

第 52、55 份日志中未逐条人工阅读的 386 与 522 条重复动态进度，已用[进度序列机器解析](progress-sequence-audit.json)逐条检查：全部成功解析，两个文件各三阶段到终点且无越界。这项检查使用程序解析进度数字。
