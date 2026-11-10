# 文档正文复核台账：冻结清单第 1–47 篇

基线为 [140 篇冻结清单](document-freeze-20261110.txt) 的前 47 行。表中记录各篇的全文阅读情况。历史报告的实验条件与后续更正分别见原文和现行状态。

| 序号 | 路径 | 本轮全文 | 结论或缺口 |
| --- | --- | --- | --- |
| 1 | `AGENTS.md` | 是 | 操作约定及环境已阅读；单输入开关对照需写明范围，已通知主审补限定。 |
| 2 | `CONTRIBUTING.md` | 是 | 现行贡献说明：测试数带日期，注意历史证据保留和 CI 边界。 |
| 3 | `README.md` | 是 | 英文项目入口：权重来源与单输入、分窗边界有说明；部分结论需结合证据看。 |
| 4 | `README_zh.md` | 是 | 中文项目入口：结果分母、批次影响及输入格式有条件说明；未全面重跑示例。 |
| 5 | `docs/CHANGELOG.md` | 是 | 提交记录以 2026-11-09 快照为界，历史变更摘要不等同当前结论。 |
| 6 | `docs/DATA_ACCESS_GUIDE.md` | 是 | 数据指南完整读取：混音能量、通道、路径和索引风险均有补充；历史访问/容量不能视为实时。 |
| 7 | `docs/DOCUMENTS_INDEX.md` | 是 | 文档索引全文读取：按当前入口与历史分组；文档数随本轮新台账增加会变化，应按最终清单更新。 |
| 8 | `docs/PAPER_REPORTED_RESULTS.md` | 是 | 论文结果表明确转录自旧 README，未逐项对原论文重新验真。 |
| 9 | `docs/PHASE2_VOLCANO_LRS2_TRAINING.md` | 是 | 候选 8 卡训练 runbook：核对了 K=1、绝对路径和环境限制；未启动训练。 |
| 10 | `docs/README.md` | 是 | 导航按时点区分现行与历史；新增审查记录需要在最终收口更新。 |
| 11 | `docs/REPRODUCTION_PHASE1_REPORT.md` | 是 | 历史 Phase1 §3.5 将权重谱系和论文目标方向写成确定，§7 嘴部素材假设过期；已加置顶更正，正文保留。 |
| 12 | `docs/REPRODUCTION_PHASE2_REPORT.md` | 是 | 历史 Phase2 §1/3/5.1 对权重、差距归因、full3、通道和损失收益判断过度；已加置顶更正。 |
| 13 | `docs/REPRODUCTION_PLAN.md` | 是 | 历史计划含旧跨语料结论、同协议和 10 卡旧说法；已加置顶更正，依据仓库论文 Table 16 对照。 |
| 14 | `docs/REPRODUCTION_REPORT.md` | 是 | 现行复现报告的均值/分母、full3、权重身份、跨样本与 K 边界已逐项阅读；没有重新运行全量。 |
| 15 | `docs/REPRODUCTION_REPORT_20261109.md` | 是 | 2026-11-09 原版报告为完整历史快照；权重谱系、near 增益、覆盖率、时延与结果等多处有被后续证据推翻的断言；应依现行状态阅读。 |
| 16 | `docs/STATUS.md` | 是 | 当前 STATUS 明确有效分母、权重来源未确认、full3 作废及测试限制；未重复全部实验。 |
| 17 | `docs/reviews/round1/01-model-core-review-by-layers-dsp.md` | 是 | 历史 round1 模型评审完整读取：含 concat[2] 与全窗归一化、MPS bridge 的旧版发现和部分未独立复核项；路径与源码行号属历史版本。 |
| 18 | `docs/reviews/round1/02-layers-dsp-review-by-model-core.md` | 是 | 历史 round1 DSP 评审全文：涉及 STFT 因果、归一化探针条件与后续对 PL 导入的自我更正；部分旧因果推断须结合现行专题。 |
| 19 | `docs/reviews/round1/03-training-system-review-by-loss-metric-eval.md` | 是 | 历史 round1 训练评审全文：指出 88 像素、整段评测、配置污染和旧版日志问题；评审基于旧代码，修复状态以当前源码为准。 |
| 20 | `docs/reviews/round1/04-losses-metrics-video-review-by-data-pipe.md` | 是 | 历史 round1 评测审查全文：旧版 mono、PESQ 顺序与数据 dtype 风险，审查者也注明其参考权重格式而非直接读文件；后续修复须对照现行代码。 |
| 21 | `docs/reviews/round1/05-data-pipeline-review-by-train-sys.md` | 是 | 历史 round1 数据审查全文：meta 缺失 K=0 仍有正项、test loader 参与验证、stub 和采样率限制；旧版缺陷不自动等同现状。 |
| 22 | `docs/reviews/round1/06-studio-backend-review-by-studio-frontend.md` | 是 | 历史 round1 Studio 后端评审全文：审查状态码、窗口坐标、设备与依赖；PIL stub 和无真实权重限制已保留，32-bit WAV 后续有更精确实测。 |
| 23 | `docs/reviews/round1/07-studio-frontend-review-by-ops-tests-docs.md` | 是 | 历史 round1 前端评审全文：追踪、采样坐标、测试引用及 80ms guard 与 20ms 淡化的差别；性能数字仅限当时环境。 |
| 24 | `docs/reviews/round1/08-ops-tests-docs-review-by-studio-backend.md` | 是 | 历史 round1 运维评审全文：旧版同源限制、PIL 替身及 CI 等问题；未验证线上部署方式，不能把假设当成当前证实。 |
| 25 | `docs/reviews/round2/01-model-core-round2-review-by-studio-backend.md` | 是 | 历史 round2 模型评审全文：最初阻塞仅指当时 R1，末尾增量复审记录后续通过；真实权重一致性仅限记录的输入与设备。 |
| 26 | `docs/reviews/round2/02-layers-dsp-round2-review-by-train-sys.md` | 是 | 历史 round2 DSP 评审全文：旧版因果表述与扰动条件缺失；其结论对应指定旧版，不代表当前未修。 |
| 27 | `docs/reviews/round2/04-losses-metrics-video-round2-review-by-ops-tests-docs.md` | 是 | 历史 round2 指标评审全文：checkpoint、指标与形状核验及排序断言纠错；旧 tracker 行为与现行实现不可混用。 |
| 28 | `docs/reviews/round2/05-data-pipeline-round2-review-by-layers-dsp.md` | 是 | 历史 round2 数据评审全文：uint8 与 float32 预处理存在约 2.6e-7 差异，旧版断言等级和 stub 边界有残留。 |
| 29 | `docs/reviews/round2/06-studio-backend-round2-review-by-studio-frontend.md` | 是 | 历史 round2 Studio 协议评审全文：sessionEpoch 非零坐标差异与后续 v3 处置均已阅读；线上部署形式仍需外部证据。 |
| 30 | `docs/reviews/round2/08-ops-tests-docs-round2-review-by-loss-metric-eval.md` | 是 | 历史 round2 运维评审全文：旧 CI、权重、PIL stub 和依赖条件逐项记录；运行结果仅适用于当时环境。 |
| 31 | `docs/understanding/00-CONVENTIONS.md` | 是 | 历史协作规范：旧环境、无权重结论与当前不同，不是当前仓库操作规则。 |
| 32 | `docs/understanding/00-project-map.md` | 是 | 历史项目侦察 v0 自标待验证，旧依赖、代码行数及数据可用性不可直接作为当前事实。 |
| 33 | `docs/understanding/00-round-plan.md` | 是 | 历史多轮协作计划，列出当时验收目标与分工，不证明后续所有问题已经消失。 |
| 34 | `docs/understanding/01-model-core.md` | 是 | 历史模型专题 975 行已分段全文读取；默认视频轴、旧 config 原地修改、MDCL 别名和 FRCNN eval 返回值的过时结论已加置顶提示；前视数值须保留探针条件。 |
| 35 | `docs/understanding/02-layers-dsp.md` | 是 | 全文逐行读至 895 行；历史 stub 与旧版源码分析，已在原文顶部更正默认 VideoBlock 轴、全窗统计、配置深拷贝、仓库内 make_pad_mask、有限输入对照边界；历史正文保留 |
| 36 | `docs/understanding/03-training-system.md` | 是 | 全文逐行读至 700 行；历史版 logger、设备策略、配置污染、MDCL 缺方法已变化；10 卡模板不能证明论文训练身份；旧 DDP 分桶风险仍见当前源码；已加顶部更正，原文保留 |
| 37 | `docs/understanding/04-losses-metrics-video.md` | 是 | 全文逐行读至 1191 行；更正历史 mono、FRCNN eval、指标、配置污染与原始嘴部说法；论文评测窗及权重语料身份未证明；已加顶部提示并保留历史正文 |
| 38 | `docs/understanding/05-data-pipeline.md` | 是（转交复读） | 另一组已全文复读，具体范围及更正见 [38–41 补充台账](document-supplement-038-041.tsv)。 |
| 39 | `docs/understanding/06-studio-backend.md` | 是（转交复读） | 另一组已全文复读，具体范围及更正见 [38–41 补充台账](document-supplement-038-041.tsv)。 |
| 40 | `docs/understanding/07-studio-frontend.md` | 是（转交复读） | 另一组已全文复读，具体范围及更正见 [38–41 补充台账](document-supplement-038-041.tsv)。 |
| 41 | `docs/understanding/08-ops-tests-docs.md` | 是（转交复读） | 另一组已全文复读，具体范围及更正见 [38–41 补充台账](document-supplement-038-041.tsv)。 |
| 42 | `docs/understanding/20261110-data-and-training.md` | 是 | 现用数据训练专题全文读取：采样期望、K=0/K=1 及构建风险有证据和局限；未重跑多卡。 |
| 43 | `docs/understanding/20261110-evaluation-and-apps.md` | 是 | 现用评测应用专题全文读取：全量结果、首窗、批次和 WAV 边界与状态页基本一致；原“跨语料结果”措辞已反馈主审限定。 |
| 44 | `docs/understanding/20261110-model-and-streaming.md` | 是 | 现用模型专题全文读取：视频轴、归一化与因果开关均有条件说明；未扩大单样本结论。 |
| 45 | `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` | 是 | 全文逐行读至 690 行；历史总纲多处模型、训练和评测结论已变；权重身份、目标约定、full3、嘴部尺寸、因果和有限输入等价边界已加顶部更正 |
| 46 | `docs/understanding/PAPER_ALIGNMENT_20261030.md` | 是 | 全文逐行读至 368 行；旧论文对照将权重训练身份、因果归因和数据重生成写得过满，full3 作废且 mean/ch0 不能证明严格复现；已加顶部提示 |
| 47 | `docs/understanding/THREE_ROUND_REVIEW_20261030.md` | 是 | 全文逐行读至 149 行；150 候选与现存 captured 43 条覆盖边界、跨语料未知、full3 作废、K=0 正项仍在，均已加顶部更正 |

本组直接全文读取 43 篇，另 4 篇由另一组全文复读，合计覆盖 47 篇；后者的范围与发现见补充台账。
