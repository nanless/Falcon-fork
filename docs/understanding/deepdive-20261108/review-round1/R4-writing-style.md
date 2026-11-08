# R4 写作规范审查（review-4）

审查对象：`dev_a800_2gpus:/root/code/github_repos/Falcon-fork` 工作区改动（HEAD `6c1d5b7` + 未提交重写）。
规范：`plain-chinese-writing`（SKILL + user-preferences / sentence-editing / structure-and-detail / review-workflow）与 `document-rewrite` SKILL。
只读仓库；除本报告外未写任何文件。

## 0 结论速览

**覆盖**：9 份通读。6 份本轮改动（`docs/STATUS.md`、`README.md`、`README_zh.md`、`AGENTS.md`、`space/README.md`、`studio/README.md`，`git diff --stat` 共 +548/−134 行）；3 份本轮新增（`docs/understanding/deepdive-20261108/{00-SYNTHESIS,11-paper-alignment,12-causality-latency}.md`，1,629 行）。另有 5 份 phase2 报告**未改**，见 §4。

**判定：达标 5 份，需返工 4 份。** 文风本身（空话、八股、口号、机械排比）几乎找不到问题——我按八股连接词、口号、咨询黑话、夸张形容词等 20 个模式逐个检索，零命中，结论先行和"实测/推断/未验证"三标签是这批文档做得最好的部分。真正的问题是**同一件事在不同文件里给了不同的数字和不同的证据强度**，且有两处数字用文档自己给的命令复现不出来。

最严重的 3 个模式：

1. **同一结论被标成两种依据强度。** RealSSA-Sim 7.774/8.479：`07-eval-benchmark.md:263` 明写"两轴同时不同，不能写 ✅ 复现"、`11-paper-alignment.md:433` 判"巧合级吻合，不算复现"；而 `docs/STATUS.md:115` 写"复现，+0.07/+0.28"，`README.md:253` 与 `README_zh.md:253` 对外写"**RealSSA-Sim ✅**"。这是本轮最可能被外部读者直接引用的一条判定，三份文档三种说法。
2. **同一指标两套数字，其中一套复现不出来。** 跨窗一致性：STATUS:225 / README:46 / 12:17 都写 **+10.83 dB**；`studio/README.md:122` 写 **+18.2 dB**。我按 studio/README 给出的命令在本机 CPU 跑 `tools/repro/experiments/streaming_consistency.py`（2026-11-08 实测）：`guard 1280 80ms → +18.20 dB, r=0.9925, maxdev/peak 0.188`——与 studio/README 一致，与 STATUS/README 差 7.4 dB。文档里没有一句说明这两个数差在片段还是差在协议。
3. **"文件/行/包"这类可数项数错，且错了还写进了"订正"位置。** `docs/STATUS.md:93` 与 `00-SYNTHESIS.md:161` 写 Space 构建"57 文件 / 39,869,169 B"；实测 `find /tmp/r4-space-check -type f | wc -l` = **54**，57 是 `find | wc -l` 把 `falcon_model/`、`configs/`、`web/` 三个目录也数进去了。同类：`00-SYNTHESIS.md:223` 写 lock 文件"117 个 `==`"，实际 124 行 = 9 行注释 + **115 个包**（117 是含两行注释里的 `torch==`）。

## 1 逐份评分表

记号：✅ 达标；⚠️ 有具体问题（见 §2 条目号）；❌ 需实质修改；n/a 该体裁不适用。

| 文档 | 结论先行 | 数字先行 | 依据标注 | 空话 | 命令口气 | 结构 | 判定 |
|---|---|---|---|---|---|---|---|
| `docs/STATUS.md` | ✅ 开头 3 段给出 L1–L4 结论 | ⚠️ #3（57 文件） | ⚠️ #1 #2 | ✅ 零命中 | ⚠️ #15 | ⚠️ #10 #11 #12 | 基本达标，需小修 7 处 |
| `README.md` | ✅ | ✅ | ⚠️ #1 | n/a（英文） | n/a | ✅ | 需小修 2 处（#1 #19） |
| `README_zh.md` | ✅ | ✅ | ⚠️ #1 | ✅ | n/a | ✅ | 需小修 2 处（#1 #19） |
| `AGENTS.md` | ✅ | ✅ | ✅ | ✅ | ❌ #17 | ✅ | 需返工：命令口气 |
| `space/README.md` | ✅ | ⚠️ #3 #4 | ✅ | n/a（英文） | n/a | ✅ | 需小修数字（54 vs 57） |
| `studio/README.md` | ✅ | ⚠️ #2 | ✅ | n/a（英文） | n/a | ✅ | 需与 STATUS 对接（+18.2） |
| `00-SYNTHESIS.md` | ✅ 12 条速览，第一条即定义与结论 | ⚠️ #3 #5 #6 #20 | ⚠️ #5 | ✅ | ✅ | ⚠️ #13 #21 | 基本达标，需小修 7 处 |
| `11-paper-alignment.md` | ✅ | ⚠️ #7 | ❌ #8 | ⚠️ #9 | ✅ | ⚠️ #9 | **需返工**（数字 + 断言强度） |
| `12-causality-latency.md` | ✅ | ✅ | ✅ 三标签用得最严 | ✅ | ✅ | ✅ | **达标**（仅 #18 #19 两处措辞） |

## 2 问题清单

| # | 文件:行 | 原文摘录 | 问题类型 | 建议改法 |
|---|---|---|---|---|
| 1 | `docs/STATUS.md:115`；`README.md:253`；`README_zh.md:253` | "复现，+0.07 / +0.28"；"**RealSSA-Sim ✅**" | 依据标注（最严重：推断/巧合级被写成实测级 ✅） | 按 07:263 的判法改：数字照给，标签降为"数值接近、ckpt 与通道两轴不同，不构成复现"；README 的 ✅ 换成 ⚠️ 或写明"数值吻合、非同 ckpt" |
| 2 | `docs/STATUS.md:225`；`README.md:46`；`12:17` vs `studio/README.md:122` | "SI-SNR 只有 **+10.83 dB**" vs "**+18.2 dB** SI-SNR" | 数字先行 / 同指标多值，且文档命令复现不出文档数 | 统一到一条可复现命令；写明片段（4 s 两段/6 s 三段）与是否含 crossfade；对外只留一个数并注明另一组条件 |
| 3 | `docs/STATUS.md:93`；`00-SYNTHESIS.md:50,161` | "构建产物 57 文件 / 39,869,169 B" | 数字错误（目录被数成文件） | 改 **54 文件**；字节数按当次构建重测（我在本轮改动后测得 `du -sb` 39,871,191 B，因 README 也被拷进构建） |
| 4 | `space/README.md:52,69` vs `STATUS.md:93` | "54 files and 39,865,947 B" | 同一构建两个字节数、两处文件数 | 文件数统一 54；字节数标注"随 space/README.md 改动变化"，或只写 MB 量级 |
| 5 | `00-SYNTHESIS.md:133` | "生效参数 532,129 ｜ 扣对比头" | 口径标错（532,129 = 537,697 − 死参数 5,568；扣对比头是 523,113） | 口径列改"扣死参数"，与 `STATUS.md:184-187` 的口径表对齐 |
| 6 | `00-SYNTHESIS.md:223` | "实际 124 行 / 117 个 `==`（09 实测）" | 数字错误（117 含 2 行注释） | 改"115 个包（124 行 = 9 行注释 + 115 行包）"，与 AGENTS:41 一致 |
| 7 | `11-paper-alignment.md:20,433,588,608` | "**7.776 / 8.479**"、"ch0 7.979" | 数字与仓库 CSV 不符（CSV `:9171` 行为 7.774307、7.976709） | 直接用 CSV `avg` 行的数；若要给重算值，写清分母与剔除规则，不要写成对 STATUS 的"更正" |
| 8 | `11-paper-alignment.md:143,617` | "**只要把 `causal=True` 透传进这三组卷积，lookahead 立刻归零 —— 这是……唯一实质差距**"；"两处合计只需改约 6 行代码即可与论文声明一致" | 断言超过证据（与 12、STATUS 冲突：改后是 15 帧/120 ms，且 `cLNhw` 不在那 6 行里） | 改为"回落到编码器阶梯 15 帧/120 ms"；删"唯一实质差距"，改成"padding 这一条的修法" |
| 9 | `11-paper-alignment.md:615-617` | "## 附：一句话总评 …… 差距不在实现水平，而在……" | 套路结尾 + 复述全文 + 拔高 | 删整节；结论已在 §0 逐条列过 |
| 10 | `docs/STATUS.md:7` | "## 1 一句话现状"（下面 5 条 bullet + 1 段） | 标题与实际内容不符 | 改"## 1 复现现状（L1–L4）" |
| 11 | `docs/STATUS.md:33` | "本次只重写本文件；……工作区其余未提交改动不属于本次改动" | 审查/过程叙述进正文 | 挪到 §9 变更记录；表格该行只留仓库与分支事实 |
| 12 | `docs/STATUS.md:58,75` | "旧稿写"115 行"，应读作"115 个包"；"旧稿写 33 G，偏大" | 审查记录进正文（应为结论） | 直接写现值与口径；订正过程放 `docs/verification/phase2/REGRESSION-docs.md` 或 §9 |
| 13 | `00-SYNTHESIS.md:252-253` | "要复现论文的对比学习收益，要复现论文的对比学习收益，就得按 App A.2……" | 重复句（编辑遗留） | 删重复半句 |
| 14 | `00-SYNTHESIS.md:22,345`；`12:19` | "2026-11-08 Lead 复核已更正"；"本轮按硬规则未跑 GPU" | 内部流程语进正文 | 改成事实句："GPU 数字未在本轮测量，见 §7" |
| 15 | `docs/STATUS.md:216` | "写结果时把分母跟着指标一起写，不要用一个"9,169 条"概括全部列" | 命令口气（对读者下指令） | 改"分母需与指标同行给出；9,169 只对 SI-SNRi 成立" |
| 16 | `docs/STATUS.md:225-226` | "75%（50/63 帧）的帧在上下文不足下算出" | 数字可核性（50/63 与"前约 13 帧"的换算没写） | 补一句"63 = 每个 0.5 s 块的帧数（窗内帧 178–240），其中 13 帧足上下文 ⇒ 50/63" |
| 17 | `AGENTS.md:3,7,25,126-131,174` | "（运行环境强制）"、"禁止使用"、"必须"、"不得把包装进系统 Python"、"新命令一律使用 `falcon` 环境" | 命令口气（SKILL 第 12 条：红线可留，但要写原因与后果） | 改为"事实 + 条件 + 后果"：如"系统 Python 3.8 无项目依赖，跑训练会在 import 阶段失败"；§6 已改好的写法可作模板 |
| 18 | `12:19` | "本轮按硬规则未跑 GPU" | 内部规则外泄 | "本轮未测 GPU；该值引自 …" |
| 19 | `12:17`；`README.md:46` | "两次预测相差 **+8.69 dB**" | 偏差用"+"号，易读成提升 | 去掉"+"，写"相差 8.69 dB" |
| 20 | `00-SYNTHESIS.md:239` | "A800 单窗 357 ms CPU / 47.1 ms 只有表格" | 主体混写（357 是 CPU 数、47.1 是 A800 引用值） | 拆成两项 |
| 21 | `00-SYNTHESIS.md:218-224` | "STATUS:41 …"、"STATUS:22"、"STATUS:63" | 交叉引用行号已随本轮重写失效 | 改成小节号（§4.1/§2.3/§5），或注明"旧版行号" |
| 22 | `docs/STATUS.md:303`；`12:518-521` | "论文核心卖点是……"；"D1 … 改写为三档" | 轻度问题：前者"核心卖点"是口号式概括、后者是给别人的改稿清单 | 前者改"论文声明的是无时域 lookahead"；后者保留（属本轮交付的措辞清单） |
| 23 | `README.md:49`；`README_zh.md:49` | "592 ms（re-measured at 694 ms）……RTF 1.2–1.4" | 两个实测值并列未说明差异来源 | 写"592 ms（一次复测 694 ms，机器负载不同）"或只留区间 |
| 24 | `studio/README.md:99` vs `12:465-472` | "designed delay is about 0.7–0.8 s" vs 典型 377 ms / CPU 672 ms | 同一延迟三个数，读者无法判断 | 注明"设计值(含播放缓冲) / 前向实测 / 端到端推算"三者的口径 |

## 3 跨格式误伤检查

对照 `git diff` 与当前文件逐项看，**没有发现表格被压平、链接变裸文本、代码块缩进被破坏**：

- 表格：STATUS 表从 HEAD 的 4 张增到 16 张，全部保留 `|---|` 分隔行与列数；README/README_zh 只加行不改列；`space/README.md`、`studio/README.md` 的表格未动。
- 代码块：围栏成对（STATUS 2 个、README 18、AGENTS 14、studio 16、12 号 34）。README.md 新增的 `### Parameter and MAC counting (needs one GPU)` 与小节标题都落在围栏之外，围栏没有被标题切断。
- 链接：STATUS 30 条相对链接里 13 条指向本轮新增的 deepdive 报告，逐条在服务器上核对存在（`docs/understanding/deepdive-20261108/01…12`、`P2E-realssa-sim-metrics_combined.csv`、`DNSMOS-realssa-real-mean-full4-20261031/` 等）；README/README_zh/studio/space 的 `README_zh.md`、`PROVENANCE.md`、`LICENSE` 均存在。
- 唯一"格式级"损伤不是 Markdown 而是**定位失效**：SYNTHESIS 与 12 号用 `STATUS.md:41/44/22` 这样的旧行号引用，STATUS 重写后这些行号指向别的内容，读者按图索骥会看错（见 §2 #21）。

## 4 不适用 / 例外说明

- **英文文档**：`README.md`、`space/README.md`、`studio/README.md` 是英文，plain-chinese-writing 明确把英文长文排除在适用范围外。我只查了它们的结构、数字与依据标签，不评中文文风（表里记 n/a）。这三份的实质问题（#1 #2 #3 #4）都是数字/判定，不是语言。
- **本轮未改，不在审查范围**：`docs/verification/phase2/{P2D-report,P2E-report,README,QA-report,P0-FIX-report}.md` 与 `P2A/P2B/P2C/P2F/REGRESSION-docs`。依据：`git status --porcelain docs/verification/` 只有 `scripts/data_pipe/verify_dataset2.py` 一处改动，上述 md 的 mtime 均为 2026-10-30（最后一次改动在 commit `2062528`）。按 document-rewrite"保留历史原稿"的要求，它们不应随本轮重写；其中 `P2E-report` 把 9,169 条同时用于 SDRi 的旧口径，已由 `STATUS.md:212-216` 更正，phase2 侧保持原样是对的。
- **AGENTS.md 的命令口气有体裁理由**：它是给 agent 与开发者的约定文件，用"禁止/必须"是行业惯例；但 SKILL 第 12 条把"对其他智能体的说明"也算在内，故计为需返工（§2 #17），而不是直接判违规。本轮新写的 §6 几段（存基线、证据只追加、提交由用户决定）已经改成"原因 + 后果"，可直接当改写模板。
- **两份实测由我完成，供核对**：① `space/build.py --out /tmp/r4-space-check` → 54 文件、`du -sb` 39,871,191 B；② `tools/repro/experiments/streaming_consistency.py`（CPU、6 s、3 段）→ guard 80 ms 时 `+18.20 dB / r 0.9925 / 0.188`。两项都只写 `/tmp`，未改仓库。
