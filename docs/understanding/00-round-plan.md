# 多轮理解—评审—修订—复核计划（Lead 维护）

> 目标：直到对项目“完全理解”为止。每轮都有明确产出、负责人、验收口径。轮次之间由 Lead 读文档、
> 检查证据、写评审指派消息驱动下一轮。

## 轮次总览

| 轮次 | 名称 | 产出 | 谁做 | 进入下一轮的条件 |
|---|---|---|---|---|
| R0 | 侦察 | `00-CONVENTIONS.md`、`00-project-map.md`、`VERIFY-ENV.md`、`docs/verification/lead-verification.md` | Lead | 8 个 owner 已领任务 |
| R1 | 理解 | `docs/understanding/01..08-*.md`（8 节模板 + 断言表≥30 + 未解疑问） | 8 个 owner 并行 | 8 份文档落盘 + 每人汇报 |
| R2 | 交叉评审 1 | `docs/reviews/round1/`（逐条断言核对 + P0/P1/P2 + 遗漏 + 跨模块问题 + 问题清单） | 见下表（环形成对，互评不同域） | 8 份评审落盘 + 作者已知晓 |
| R3 | 修订深化 | 各 owner 文档 v2（修订 + 回答 Q + 补遗漏 + Changelog） | 8 个 owner | 每份文档含“对 review 意见的逐条响应” |
| R4 | 复核 + 执行级验证 | `docs/reviews/round2/`（换人复核 + 至少 3 条“执行级验证”结论） | 见下表（换配对） | 无 P0 遗留；关键数字/断言有 [验证] 证据 |
| R5 | 汇总 | `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` + 术语表 + FAQ + 风险清单 + 阅读路径；Lead 最终一致性检查 | Lead（+ 2 名 owner spot-check） | 主文档与 8 份子系统文档互不矛盾 |

## R2（交叉评审 1）指派表

| 评审人 | 评审对象 |
|---|---|
| model-core | `02-layers-dsp.md`（layers 的实际使用与形状） |
| layers-dsp | `01-model-core.md`（层数学/因果性/形状） |
| train-sys | `05-data-pipeline.md`（datamodule↔训练契约） |
| loss-metric-eval | `03-training-system.md`（损失/指标在训练中的用法） |
| data-pipe | `04-losses-metrics-video.md`（评测脚本如何消费数据） |
| studio-backend | `08-ops-tests-docs.md`（脚本/测试/CI 与服务的配合） |
| studio-frontend | `06-studio-backend.md`（WS/HTTP 协议前后端一致性） |
| ops-tests-docs | `07-studio-frontend.md`（测试约束、常量、脚本一致性） |

## R4（复核 + 执行级验证）指派表

| 评审人 | 复核对象 | 执行级验证要求 |
|---|---|---|
| model-core | `06-studio-backend.md` | diff `falcon_model` vs `look2hear`，核对 strict-load/权重 key |
| studio-backend | `01-model-core.md` | 用 Lead 的脚本复算参数量/形状，验证“因果”表述 |
| layers-dsp | `05-data-pipeline.md` | 用真实/构造张量验证 mouth/audio 形状与归一化契约 |
| data-pipe | `03-training-system.md` | 验证 data_config→litmodule 的键匹配（可 import 级） |
| train-sys | `02-layers-dsp.md` | 验证归一化/STFT 的因果性声明（可跑 Lead 因果脚本） |
| loss-metric-eval | `08-ops-tests-docs.md` | 跑 requirements/测试矩阵，复核“可运行”标注 |
| ops-tests-docs | `04-losses-metrics-video.md` | 跑/模拟 `audio_visual_test.py` 的关键函数，复核 CSV/ckpt 逻辑 |
| studio-frontend | `06-studio-backend.md`（第二评审人，协议专项） | 前端常量 ↔ 后端常量逐项对照并标注一致/不一致 |

## 评审口径（对评审人也一样严格）

1. **逐条核对断言表**：✅正确 / ❌错误 / ❓无法确认；错误必须给正确证据（path:line）。
2. **P0 定义**：事实错误（会误导实现/复现）、关键数字错误、“声称已验证但实际没验证”。
3. **P1**：重要遗漏（关键流程/接口/配置项没写）、证据行号错误。
4. **P2**：措辞/结构/细节瑕疵。
5. 评审人必须自己读**被评审区域源码**，不能只读文档；至少抽查 10 条断言到源码。
6. 每份评审必须指出：本文档对“完全理解项目”还缺什么（≥3 条具体建议）。

## 完成定义（“完全理解”的验收口径）

- 能回答：项目做什么/给谁用；两条产品线的端到端数据流；每个模块的职责、接口契约（形状/单位/常量）、
  关键设计动机；配置项如何映射到代码；如何训练/评测/推理/部署；数字从哪来、如何复算；
  与论文/README 的一致与偏差；已知 bug/死代码/风险；未解问题清单。
- 8 份子系统文档 + 主文档互相引用、无矛盾；关键断言有 `path:line` 证据；≥1 轮独立复核；
  可运行验证有命令与日志。
- Lead 能对任意一份文档抽查 5 条断言并全部通过。
