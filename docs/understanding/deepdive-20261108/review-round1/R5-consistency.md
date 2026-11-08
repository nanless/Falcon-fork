# R5 全局一致性复核（review-5）

> 仓库 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork`，HEAD `6c1d5b7`，工作区 **37 项改动**（含本轮重写的 `docs/STATUS.md` 402 行）+ 2 个未跟踪目录。只读复核，未改仓库任何文件。
> 复核方式：全仓 grep/逐文件对读；`space/build.py` 与 4 个新脚本只做 `--help` + 一次构建到 `/tmp`（不写仓库、不跑长任务）。

## 0 结论速览

- 不一致 **23 条**：其中 **11 条会造成误解**（会被直接引用错数、或让人重复做已经做过的工作）。
- 最严重三条：① 参数量"扣对比头"有两个值（523,113 / 517,545），STATUS 已判 523,113 基准错，但 4 份报告仍在用；② Space 构建三个数（54 / 57 文件，39,865,947 / 39,869,169 / 实测 39,870,088 B）；③ 本轮代码已落地 `paper_causal` / `video_lstm_batch_first` 两个开关，但**全仓文档 0 处提及**，STATUS 仍写"改约 6 行才能修"。
- 结构性风险：`docs/STATUS.md` 是"单一事实源"，但它有 **14 个 markdown 链接指向未跟踪的 `deepdive-20261108/`**，未 commit 时全部 404；同时 74 处 `STATUS.md:行号` 引用全部失效（旧 141 行文件已不存在）。

## 1 锚点漂移清单

本轮 `docs/STATUS.md` 从 141 行重写为 402 行，**所有行号锚点失效**。共 **74 处** `STATUS[:.]md?:行号` 引用（`docs/understanding/deepdive-20261108/` 内），全部落在下面 9 个旧行号上：

| 旧锚点（引用次数） | 旧文件该行是什么 | 新文件同行为 | 正确锚点 |
|---|---|---|---|
| `:41`（7） | `LRS2-2Mix 自训 ep124 8.858/9.270`（**旧文件此处本就不是 lookahead，引用从一开始就偏了 9 行**） | `\| Python \| 3.11.17 \|` | `§4.1`（L161 单遍 / L162 整链 / L163 as-shipped） |
| `:50`（10） | "causal=窗口级因果…57 帧≈456 ms"（正确位置） | 空行 | `§4.1` L159–167 |
| `:33`（8） | 参数 537,697 / 532,129 / 523,113 + MACs 10.611 G | `\| 工作区 \| 本次只重写本文件…\|` | `§4.2` L178–189、`§4.3` L193–202（或 `§3` L112–113） |
| `:44`（3） | 效率（3090）行，含 A800 47.1/56.5 ms | `\| typeguard \| 2.13.3 …\|` | `§4.6` L222（或 `§3` L125） |
| `:63`（4） | `166 passed / 136.5 s` | 空行 | `§5` L258（**新值 355.47 s**） |
| `:64`/`:64-65`（4） | Studio 51 passed/2 skipped；Node 8/8 | `\| 文件 \| 架构 \| epoch …` 权重表头 | `§5` L260–262 |
| `:22`（3） | `conda-env-lock.txt 115 行` | 空行 | `§2.3` L58 |
| `:35`（2） | `9,169 条` | 空行 | `§4.5` L214 |
| `:3,20` / `:23` / `:40` / `:53,77` / `:78` / `:83`（各 1） | 各自旧内容 | 全错位 | 改为 §节号 |
| "STATUS 141 行"（3：`07-eval:557`、`09-space:432`、`10-ops:487`） | — | 现 **402 行** | 删行数或改 402 |

**建议**：本轮起 `STATUS` 一律用 `§节号` 引用（新文件已有稳定 `§1–§9`），并在 STATUS §9 加一句"禁止引用本文件行号"。
另：`00-RECON-LEAD.md:76` 写"最后更新 2026-10-31"，实际已是 **2026-11-08**。

## 2 多说法对照表

### 2.1 lookahead —— 口径已拆三档，残留 2 处旧说法

| 位置 | 说法 |
|---|---|
| `docs/STATUS.md` L161–163；`README.md` L47–48；`README_zh.md` L47–48；`project_page/index.html` L333 | 三档齐全：**57 帧/456 ms（单遍）→ 113 帧/904 ms（整链）→ 窗尾 ≤2000 ms（as-shipped）**，且注明 15/120 ms 是论文声明 |
| `11-paper-alignment.md` L16–18,118,130–145 | 另有一套 **54 帧/432 ms、110 帧/880 ms**（tail-perturb 路），文内自注"与 57/113 差约 3%，同一物理量两条测量路径" |
| `00-RECON-LEAD.md` L50–52；`00-SYNTHESIS.md` L28–30,143–144 | 同 11（并明确"对外用 113/904"） |
| `12-causality-latency.md` L371 | as-shipped 写 **1960 ms**（245 帧）；STATUS 写 **≤2000 ms** |
| `12-causality-latency.md` L433 | "`docs/STATUS.md` 未记录 guard=1280 的依据" ← **STATUS §4.6 L226 已记录**，旧说法 |
| `09-space-project-page.md` L389,418 | "项目页没有任何对应注记，需补" ← **`index.html:333` 本轮已补**（三档齐全），旧说法 |

**应统一为**：三档 57/456、113/904、窗尾 ≤2000 ms；54/432 与 110/880 只作"另一路实测（差 3%）"脚注，不进结论行。`12:433` 与 `09:389,418` 两条陈述应删除。

### 2.2 MACs —— 两把尺子已分清，只剩四舍五入与作废值

| 位置 | 说法 |
|---|---|
| `STATUS` L113,193–202；`README*` L244；`evaluated_mac_params.py` L22–29,71–75,145 | ptflops **10.611**（含 LSTM、不含 FRCNN）/ 视觉 88² **15.897** / 整窗 **26.51**、视觉 **60.0%**；`flop_counter` **5.382** / 21.28 / **74.7%** |
| `11-paper-alignment` L178；`02-layers-dsp` L457–459 | 同口径，但写成 **21.19 / 74.6%（≈21.28/74.7%）** |
| `01-model-core` L19,330–332,421；`05-losses-metrics` L402,594–596,605；`03` L19–20,420,427 | **18.380 G**（64² 视觉 + 音频混算）保留为"原始输出/已作废" |
| `REPRODUCTION_PLAN.md` L22,86,133 | 只写 "10.611 G"，**无"音频侧/不含视觉"限定** |

**应统一为**：`10.611 GMAC（ptflops、音频侧、含 LSTM、不含 FRCNN）`；整窗 `26.51 GMAC / 视觉 60.0%`；`flop_counter` 口径写 `5.382 GMAC / 21.28 G / 74.7%`（**把 21.19/74.6 全部改成 21.28/74.7**，或在括号内注明是同一数的四舍五入）。18.380 只允许出现在"作废"标注里（现已满足）。`REPRODUCTION_PLAN` 三处补限定语。

### 2.3 参数量 —— 真冲突（1 条，高）

| 位置 | 说法 |
|---|---|
| `STATUS` L186,189 | 分离前向实际用到 = **517,545**（532,129 − 对比头 14,584）；并明确"旧稿 523,113 = 537,697 − 14,584，基准不一致，不混用" |
| `11-paper-alignment` L172 | "减对比头（14,584）= **523,113**；减 concat[2] = 532,129"（从总参数扣） |
| `05-losses-metrics` L388,795 | "537,697 = **523,113** + 14,584 ⇒ 与 STATUS:33 逐位吻合"（引用已废止的口径） |
| `07-eval-benchmark` L224 | "生效 532,129；**扣对比头 523,113**"（把两个基准串在一句） |
| `REPRODUCTION_PLAN.md` L401 | 建议写法仍是"生效 532,129；**扣对比头 523,113**" |
| 其余 28 处（README/STATUS/01/02/05/09/11 等） | 537,697 / 5,568 / 532,129 / 11,724,385 / 1,158,617 / 12,345,305 全部一致 |

**应统一为**：总 537,697；生效 532,129；**分离前向 517,545**；523,113 仅在"总参数 − 对比头"这一句里出现并加"≠ 分离前向"。

### 2.4 Space 构建 —— 三个数，且本轮实测与三者都不同

| 位置 | 说法 |
|---|---|
| `STATUS` L93,L379；`00-SYNTHESIS` L50,161；`09-space` L11,41,190,281,439 | **57 文件 / 39,869,169 B**（39.87 MB） |
| `space/README.md` L52,L69 | **54 文件 / 39,865,947 B**（39.87 MB） |
| 本轮实测（`space/build.py --out /tmp/r5-space`，工作区含本轮改动） | **54 文件 / 39,870,088 B** |

**应统一为**：`54 个文件、约 39.87 MB（2026-11-08 重测 39,870,088 B）`。**"57 文件"是错的**（多算 3 个，疑似把 `__pycache__` 或临时文件算进目录），三处 `39,869,169` 与 `39,865,947` 都不等于当前构建。字节数随代码改动漂移，建议只锁"54 文件 + ~39.87 MB"。

### 2.5 视觉消融 —— N 与 Δ/绝对值混用（高）

先说清事实：任务描述里的 **−15.98 / −11.68 / −3.16 在仓库中不存在**（`grep` 零命中），不要按这三个数去找。

| 位置 | 说法 | N |
|---|---|---|
| `P2F-report.md` L42,101,134；`QA-report.md` L167；`10-ops-repro-git.md` L476 | mouth 置零 **−14.4 dB**（绝对值），换说话人 **−23.3 dB**，基线 7.911 | **20** |
| `P2B-report.md` L19–21,76–77 | mouth 置零 **−8.173 dB**（绝对值），配对 Δ**−15.089**；shuffle **−5.977**，Δ**−12.894**，基线 6.916 | **300** |
| `STATUS` L15,126 | "只有 big/small 与 mouth 置零 / **打乱（N=20）** 部分" | 把打乱也归到 N=20（**错**，打乱只有 N=300 的 −5.977） |
| `10-ops` L476 | 已正确标注"−14.4/−23.3 是 20 条样本" | — |

**应统一为**：`置零：N=20 −14.4 dB / N=300 −8.17 dB（配对 Δ−15.09）`；`打乱：N=300 −5.98 dB（Δ−12.89）`；`换说话人：N=20 −23.3 dB`。STATUS §3/§4.9 的"（N=20）"必须拆开，否则读者会把 −8.17 与 −14.4 当矛盾。

### 2.6 测试 —— STATUS 自相矛盾 1 处 + 4 份报告引用已废止的 136.5 s

| 位置 | 说法 |
|---|---|
| `STATUS` L254,258,267,284 | **166**（164 `def test_` + 2 参数化）；GPU 可见 166/0/0，**355.47 s**；无 GPU 164 passed/2 failed，173–186 s |
| `STATUS` L288 | "数据被清掉时 **162 条**里会有一大批变 skip" ← **与 166/164 自相矛盾**（无出处） |
| `STATUS` L260–262 | Studio 51+2（有 CUDA）/ 53+0（无）；Node 8/8；Space 静态 3 passed；i18n 226 |
| `10-ops-repro-git.md` L12,83,350,455,471 | 仍称"STATUS 记 136.5 s，实测 355.47 s，对不上" ← 新 STATUS 已改 355.47 s，"冲突"已不存在，这条应销案 |
| `00-SYNTHESIS` L158–159,222,224；`09-space` L16,167–169,240,400 | 166 / 164+2f / 51+2 / 53+0 / CI 只选 164（`-k "not diff_scope"`）——与 STATUS 一致 |

**应统一为**：166（GPU 可见）/164+2f（无 GPU）/355.47 s；删掉 L288 的"162"；`10-ops` 的 136.5 s 段落改为"旧值，STATUS 已于 2026-11-08 更正为 355.47 s"。

### 2.7 分母 —— 基本统一，残留 0.002 dB 双值

| 位置 | 说法 |
|---|---|
| `STATUS` L12,115–116,128,214–218；`07-eval` L14,226–227；`10-ops` L432；P0-FIX/P2D/P2E/QA/README | SI-SNRi **9,169**、SDRi/PESQ **9,166**（6 次 SDR 奇异 + 3 次 PESQ 无语音）——已一致 |
| `11-paper-alignment` L20,433,608 | SI-SNRi 均值写 **7.776**（STATUS/00 写 **7.774**）；ch0 写 **7.979**（STATUS 写 **7.977**） |
| `STATUS` L218,307；`00-SYNTHESIS` L250 | tt 分母 **3,000 vs 6,000** 是显式未决项（6,000 行 = 3,000 mix × 2 向），口径已标注"待重核"——一致，不算冲突 |

**应统一为**：9,169 / 9,166 不动；均值统一用 CSV `avg` 行 **7.774 / 8.479 / 7.977 / 8.452**，`11` 的 7.776/7.979 加"我重算，差 0.002"脚注或直接改成 7.774/7.977。

### 2.8 环境锁 —— "115 包"对，"117 个 `==`"是计数错误

| 位置 | 说法 |
|---|---|
| 实测 `conda-env-lock.txt` | **124 物理行** = 9 注释 + **115 包**；`grep -c "=="` = **117**，因为**第 3 行与第 7 行两条注释里也含 `==`** |
| `STATUS` L58；`AGENTS.md` L40–41 | 115 个包、物理 124 行 —— **正确** |
| `09-space` L164,243,399；`10-ops` L77 | "124 行 / **117 个 `==`**"，并据此推断"114→117 条包后行数没同步" —— **推断错误**，117 不是包数 |
| `P0-FIX-report` L41,114；`REGRESSION-docs` L36,49；`QA-report` L67,152；`three-round-review…json` L48 | 仍写"**115 行**"（应为 115 包 / 124 行），属历史快照 |

**应统一为**：`124 物理行 = 9 行注释 + 115 个包`；删掉 `09/10` 的"117 个 `==`"与其推断；历史 phase2 报告保留原文但已在顶栏注明"以 STATUS 为准"即可。

## 3 代码 / 文档一致性

| # | 事实 | 代码 | 文档 | 判定 |
|---|---|---|---|---|
| C1 | `paper_causal`（默认 **False** = 发布行为） | `look2hear/models/Falcon.py` L707,713,772,808,821；`studio/falcon_model/models/Falcon.py` 同构 L707–821 | **全仓 0 处提及**（README/AGENTS/STATUS/12 篇 deepdive 全无） | ❌ 名字与默认值无处可查 |
| C2 | `video_lstm_batch_first`（默认 **False**） | `look2hear/.../Falcon.py` L925,931,966,982；studio 同构 | 同上，0 处 | ❌ |
| C3 | `respect_index`（默认 **False**） | `avspeech_dataset.py` L105,133,195–207；`avspeech_dataset_near.py` L67,95,185–197 | `STATUS` L102/§4.7 仍写"`idx` 被丢弃、`shuffle` 无效、采样有放回"，当作不可改行为 | ❌ 默认行为描述对，但**没告诉读者有开关** |
| C4 | `mouth_fps`（默认 **25.0**）、`pad_mouth`（默认 **False**） | `avspeech_dataset_near.py` L77–113,216–228；`avspeech_dataset.py` L57–94 | `STATUS` L102 写"`mouth_start` **写死 16000 与 25**"；`03-video-frontends.md` L480,503 同 | ❌ **与代码不符**：现为 `int(rand_start/sample_rate*mouth_fps)`，16000/25 只是默认值 |
| C5 | `--allow-partial` + 覆盖率自检 | `audio_visual_test.py` L21,100–101,126–128,487–529,555；写入 `coverage_rank0.json`；`raise` 现 **28 处** | `STATUS` L141："全文 23 处 `raise`，**没有一条校验跑了几条/共几条**，只能靠 `utterances=N` 和 `metrics_counts.json`"；`07-eval` L18 同 | ❌ **本轮新增的能力被文档否定**（数字也旧：23→28） |
| C6 | `paper_causal` 使"改 6 行"过时 | 开关已存在，打开即得论文式 causal padding | `STATUS` L167"即使把这三组卷积改回 `causal=True`（约 6 行）"、L303 待办 2"改动约 6 行"；`12-causality` L507/L496 | ❌ 会让人重复实现已有开关；正确写法是 `paper_causal=True`（A/B 用 `tools/repro/experiments/lookahead_ab.py`） |
| C7 | `--device` 语义 | `evaluated_mac_params.py`：`choices={auto,cuda,cpu}`，auto/cuda 无 GPU 时 **exit 2**；`streaming_consistency.py`：`choices={cpu,cuda,mps}`，默认 cpu；`video_ablation_check.py`：**自由字符串**，默认 `cpu` | README L210 只描述 `evaluated_mac_params` 的三个值 | ⚠️ 三个脚本三种语义，同名不同义 |
| C8 | `--wrong-mode` / `--select` | `video_ablation_check.py` L202–216（`--select {first,spread}`、`--wrong-mode {other,next,s2}`） | 仅脚本自身 `--help`/docstring，无任何 md 引用 | ⚠️ 可用但不可发现 |
| C9 | `--guard-ms` | **不存在**；对应开关是 `streaming_consistency.py --guards`（ms 逗号列表，默认 `80,160,320,456,640`） | `studio/README.md` L149 用 `--guards 80,456`（正确） | ❌ 任务书/若他处写 `--guard-ms` 会直接报错 |
| C10 | 4 个新脚本 `--help` 与行为 | 逐个 `timeout 30 --help` 全部 rc=0；`ln_variant`/`lookahead_ab`/`streaming_consistency`/`video_ablation_check` 的 usage、默认值、docstring 与 argparse 一致 | `video_ablation_check --seed` 自注"reserved"，`--stage` 只有 `variants` 一个取值，均已在 help 里说明 | ✅ 自洽（唯一小瑕：`--stage {variants}` 单值 choices 属冗余写法） |

## 4 未跟踪目录引用与死文件

**A. 指向 `docs/understanding/deepdive-20261108/` 的 markdown 超链接：14 个，全部在 `docs/STATUS.md`**（L5 + L370–382 的 13 行表）。该目录未 commit ⇒ 推送到 GitHub 后 14 个链接全 404，而 STATUS 自称"单一事实源"。

**B. 非链接的路径引用（点了也不会跳，但同样是未跟踪路径）7 处**：`README.md` L51,258、`README_zh.md` L51,258、`AGENTS.md` L111,171、`project_page/index.html` L333（对外页面直接指向一个不存在于仓库的文件）。

**C. 路径写法错误的相对引用 16 处（7 个文件）**：`docs/DATA_ACCESS_GUIDE.md` L17,427,430；`docs/REPRODUCTION_PLAN.md` L16,84,338,398；`phase2/P0-FIX-report.md` L7,79；`P2D-report.md` L9,235；`P2E-report.md` L10,263；`QA-report.md` L10；`phase2/README.md` L16,173 —— 写成裸 `deepdive-20261108/07`，从 `docs/` 或 `docs/verification/phase2/` 出发都解析不到，应为 `docs/understanding/deepdive-20261108/07-eval-benchmark.md`。

**D. `tools/repro/experiments/` 的索引状态（2 个死文件）**：

| 脚本 | 被谁引用 |
|---|---|
| `streaming_consistency.py` | ✅ `studio/README.md` L113,145,147,149（含 3 条可跑命令） |
| `ln_variant.py` | ✅ `look2hear/layers/normalizations.py` L8,188,285（注释里指路） |
| `lookahead_ab.py` | ❌ **零引用**（只在自身 docstring 里）——正是 C1/C2 两个开关的 A/B 工具 |
| `video_ablation_check.py` | ❌ **零引用**——正是 §2.5 视觉消融（−8.17/−14.4）的复现工具 |

没有任何 README/STATUS/AGENTS 段落索引 `tools/repro/experiments/`。`__pycache__/` 已被 `.gitignore` 覆盖，无害。

## 5 建议的统一动作（可直接派活）

1. **STATUS 行号 → 节号（一次性 74 处）**：在 12 篇 deepdive 里批量替换 `STATUS.md:41`→`STATUS.md §4.1`、`:50`→`§4.1`、`:33`→`§4.2/§4.3`、`:44`→`§4.6`、`:63`→`§5`、`:64-65`→`§5`、`:22`→`§2.3`、`:35`→`§4.5`；删 `07-eval:557`/`09-space:432`/`10-ops:487` 的"141 行"。同时在 `STATUS §9` 增一行：`引用本文件请用 §节号，不要用行号（每轮重写会漂）`。
2. **参数量口径（4 文件）**：`11-paper-alignment.md:172` 改为"总 537,697 − 对比头 14,584 = 523,113（≠ 分离前向）；分离前向 = 532,129 − 14,584 = **517,545**"；`05-losses-metrics.md:388,795` 同改；`07-eval-benchmark.md:224` 把"生效 532,129；扣对比头 523,113"改为"生效 532,129；分离前向 517,545"；`REPRODUCTION_PLAN.md:401` 同上。
3. **Space 构建（3 文件统一到 54 文件）**：`STATUS.md:93` 与 `:379` 的"57 文件 / 39,869,169 B"→"**54 文件、约 39.87 MB（2026-11-08 重测 39,870,088 B）**"；`00-SYNTHESIS.md:50,161`、`09-space-project-page.md:11,41,190,281,439` 同步；`space/README.md:52,69` 的 39,865,947 也更新为 39,870,088。
4. **把两个已在代码里的开关写进文档（消除 C1/C2/C6）**：`STATUS.md` L167 与 §6 待办 2 改为"已实现 `paper_causal`/`video_lstm_batch_first`（默认 False＝发布行为），A/B 用 `tools/repro/experiments/lookahead_ab.py`，不需要再改 6 行"；`AGENTS.md §3` 增 5 行表：`paper_causal=False`、`video_lstm_batch_first=False`、`respect_index=False`、`mouth_fps=25.0`、`pad_mouth=False`，各一行说明"默认值 = 发布行为，开启会改变数值/需重训"。
5. **修 4 处已过时的观察（各自一行）**：`STATUS.md:141` 改为"入口已有覆盖率自检（`coverage_rank0.json` + `--allow-partial`），全文 28 处 `raise`"；`STATUS.md:102` 的"`mouth_start` 写死 16000 与 25"改为"`mouth_start=int(rand_start/sample_rate*mouth_fps)`，16000/25 仅默认值；`idx` 丢弃可用 `respect_index=True` 关闭"；`12-causality-latency.md:433` 删"STATUS 未记录 guard 依据"；`09-space-project-page.md:389,418` 删"项目页需补注记"（`index.html:333` 已有）。
6. **补索引与提交（避免死文件/404）**：在 `README.md` L258 附近增一节 `tools/repro/experiments/`（4 行表格，标注 `lookahead_ab.py`＝开关 A/B、`video_ablation_check.py`＝视觉消融复现、`ln_variant.py`＝归一化对照、`streaming_consistency.py`＝跨窗一致性）；`docs/STATUS.md §8` 表头加一句"以下 13 个文件位于未提交目录，clone 后不存在"；**建议本轮一并 commit `deepdive-20261108/` 与 `experiments/`**，否则 STATUS 的 14 个链接与 README/项目页的引用全部悬空。
