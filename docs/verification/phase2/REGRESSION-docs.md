# Phase-2 回归 + 文档一致性核对（task-25 · `studio-frontend`）

> **快照：2026-10-21T19:45Z（v1，终检完成）** · 分支 `repro/phase2`（HEAD `af74cca`；**未 commit、未切分支**）
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / 2×A800）+ 系统 `node v22`
> 写入范围：`docs/verification/phase2/`（本文件）、`docs/REPRODUCTION_PLAN.md`（仅状态小节）、
> `docs/DATA_ACCESS_GUIDE.md`（仅状态/链接/证据措辞）；**未改源码、未 commit**。
> 状态：**v1（终检）**——P2-A…F 报告全部落盘并逐项核对；回归终值 **111 passed / 1 skipped / 0 failed**（QA 终检 19:52Z 独占重跑；96/6 为 19:29Z 中间值）；文档一致性/链接/HF 事实已核；
> 2 处报告 P3 勘误已由 owners 修复并复核；P2-D 正式训练 19:26:41Z 已启动（结果待跑完）、RealSSA 仍待审批（见 §6）。

---

## 1. 回归结果（实际执行，2026-10-21 18:57–19:29Z）

| # | 项目 | 命令 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | Studio Python | `$ENV -m pytest studio/tests -q` | **35 passed, 2 skipped**（33.33 s，18:57Z） | ✅ 与 Phase 1 基线一致 |
| 2 | Studio Node（8 个） | `for f in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$f"; done` | **8/8 rc=0**（18:57Z） | ✅ |
| 3 | Space 组装 + i18n | `$ENV space/build.py --out /root/tmp/space-verify2` → `node space/tests/check_i18n.mjs /root/tmp/space-verify2` | `Assembled /root/tmp/space-verify2`；**`i18n: 226 Chinese strings translated`**（18:57Z） | ✅ 与 Phase 1 相同 |
| 4 | 导入面 | `PYTHONPATH=$PWD $ENV -c "import look2hear.{models,layers,system,metrics,losses,datas,videomodels,utils}, audio_train, audio_visual_test"` | **10/10 imports OK**（18:57Z） | ✅ |
| 5 | 新增测试（全量） | `$ENV -m pytest tools/repro/tests -q` | **111 passed, 1 skipped, 0 failed**（QA 终检 19:52Z 独占重跑；中间值 96/6 为 19:27–19:29Z）；历史：19:04Z 82/12（94）→ 19:16Z AGENTS.md 致 2 failed → QA 19:20/19:23 白名单修复 → 19:26Z 守卫 2 passed；P2-C 数据使 5 条数据测试由 skip 转 passed | ✅ 全绿（QA 终检独占重跑后以终值替换） |
| 6 | Phase2 QA 框架 | `$ENV -m pytest tools/repro/tests/test_phase2_data.py tools/repro/tests/test_phase2_pipeline.py -q` | **12 passed**（数据 5 + 管线 7；P2-C 数据落盘后 5 条由 skip 转 passed） | ✅ |
| 7 | 官方权重指纹 | QA `test_phase2_official_weight_fingerprints_unchanged` | 4 条指纹与 Phase 1 基线**逐位一致**（Phase2 未改数值） | ✅ |
| 8 | Phase2 源码保护 | QA `test_phase2_diff_scope_guard` | `audio_train.py`/`audio_visual_test.py`/`look2hear/`/`studio/`/`configs/Falcon.yml` 与 main **零 diff** | ✅ |

**历史对照（同一工作区，仅记录不掩盖）**：
`18:48Z` = 75 passed / 1 failed / 6 skipped（82 collected，Phase2 白名单未更新）
→ `18:58Z` = 80 passed / **2 failed** / 12 skipped（94 collected，P2-G 新增测试已加入）
→ `19:04Z` = **82 passed / 12 skipped**（全绿）→ `19:29Z` = **96 passed / 6 skipped**（102 collected，终值；6 条 skip 均为 Phase1 回归中依赖分支 diff 的条件测试：`T1/T3/T5 尚未落地…`、`configs/Falcon.yml 未被改动`、`audio_train.py 未改动`）。

### 1.1 两轮失败项与闭环（均为“进行中状态”，非代码缺陷）

| 失败项 | 时间 | 原因 | 闭环 |
|---|---|---|---|
| `test_phase1_regression.py::test_diff_scope_only_allowed_files` | 18:48Z / 18:58Z | Phase2 的 tracked 文档改动（`docs/DATA_ACCESS_GUIDE.md`、`docs/REPRODUCTION_PHASE1_REPORT.md`、`docs/REPRODUCTION_PLAN.md`）+ QA 维护的测试文件 + untracked `configs/LRS2_2Mix.yml` 不在 Phase1 白名单 | P2-G 于 18:59:55 更新 `ALLOWED_TRACKED`/`ALLOWED_UNTRACKED`（`docs/*`、`tools/repro/*`、`configs/LRS2_2Mix.yml`），并新增 `test_phase2_diff_scope_guard` 单独断言源码零改动（见 §1 #8）；19:04Z 全绿 |
| `test_phase2_pipeline.py::test_phase2_scripts_compile_and_help` | 18:58Z | `tools/repro/phase2/gap_analysis.py --help` rc=1：`ModuleNotFoundError: No module named 'look2hear'`（脚本未自举 `sys.path`，测试子进程未设 `PYTHONPATH`） | P2-A 19:02Z 更新脚本；19:03Z 单测复跑 **1 passed**；19:04Z 全量全绿 |
| `test_diff_scope_only_allowed_files` + `test_phase2_diff_scope_guard` | 19:18Z（新） | 环境合规更新把 `AGENTS.md` 改为 tracked-modified（AGENTS §4 要求；lock 已同步 115 行），两个守卫的白名单未含 `AGENTS.md` | 已通知 P2-G；**QA 19:20/19:23 更新两处白名单（加入 `AGENTS.md`）**，19:26Z 两条守卫复跑 **2 passed** → 闭环 |

> 说明：19:20Z QA 的中间全量跑曾出现 1 failed（`test_phase2_reports_have_evidence`，pytest 启动时收集的旧代码 + P2D 报告自述“进行中”却引用未生成日志）；QA 单测复验 1 passed，本任务 19:27Z 全量复跑已无该失败。18:48Z 的 82 collected 与 18:58Z 的 94 collected 差异来自 P2-G 新增
> `test_phase2_data.py`（5 条）与 `test_phase2_pipeline.py`（7 条）；skip 数 6→12 是因为 P2-C 转换数据
> （`/root/data/falcon_lrs2_2mix`）与 P2 报告尚未落盘，属**预期 skip**，不是失败。

### 1.2 环境 / 资源观察（影响复跑，不是回归失败）

- **GPU 争用**：P2-D 的 bs8 冒烟（`P2D-smoke-bs8.log`）在 18:55Z 因 `CUDA out of memory` 失败——GPU 1 被
  外部进程（PID 2514085）占用 **70.23 GiB**，非配置/代码问题；同配置 bs4 冒烟 `exit=0`（100 step，73 s，
  `P2D-smoke.log`）。P2-B 的 `P2B-full-*` CSV 在 19:05Z 启动时为 0 B（并发大任务排队中）。
  → **正式训练/全量评测前需独占 GPU**（`nvidia-smi` 确认），复跑命令见 §5。
- **新增依赖已入锁（19:16Z）**：P2-E 为 DNSMOS 安装 `onnxruntime==1.31.0`、`speechmos==0.0.1.1`（+ protobuf/flatbuffers），
  `docs/verification/conda-env-lock.txt` 已更新为 **115 行**（4 个包均在内）、`AGENTS.md §2` 已增记 Phase2 评测依赖（AGENTS §4 合规闭环）。
  ⚠️ 该 `AGENTS.md` tracked 修改随后触发两个 diff-scope 守卫失败（`tracked modified: AGENTS.md`），已通知 P2-G 更新白名单（见 §1.1 表）。

---

## 2. 文档一致性核对（P2 报告 / GUIDE / PHASE1_REPORT / PLAN）

### 2.1 RealSSA 申请状态：发现不一致并已统一（本次修复）

- **原始不一致**：提交 `af74cca` 只把 GUIDE §0 汇总表的 **RealSSA-Sim** 行改为“申请已提交（2026-10-22）待批”，
  但同一文件 §0 **RealSSA-Real** 行、§1 表（标题“未提交任何申请”、Sim/Real 两行“未获批”、末行“未提交任何访问申请”）、
  §4.2/§5.2 gated 行、§4.3/§5.3 现状行仍是“未获批 / 未提交”的旧表述；而
  `docs/REPRODUCTION_PLAN.md:294`、`docs/REPRODUCTION_PHASE1_REPORT.md:155` 均已写“**Sim / Real 申请已提交（2026-10-22）待批**”。
- **处置**（GUIDE 内 9 处，仅状态措辞、不改数据结论）：§0 Real 行、§1 标题、§1 Sim/Real 两行、§1 “未做”行、
  §4.2 gated 行、§4.3 现状行、§5.2 gated 行、§5.3 现状行；§11 增加“更新记录”两条（含本条）。
- **现状**：GUIDE / PLAN / PHASE1_REPORT / `phase2/README.md` 四处口径一致 ✅：
  “Sim/Real 申请已提交（2026-10-22，`nanless`/USTC/CN），待人工审批，数据文件仍 403”。
- 附：`docs/DATA_ACCESS_GUIDE.md` 的 `git diff main` 由 2 行变为 32 行（20+/12-），全部为本任务状态同步与
  §3.3 链接/§11 证据措辞修复；**无数据集大小/许可/格式数字变化**。

### 2.2 HF 事实复核（2026-10-21T19:09Z，经 mihomo 代理 `127.0.0.1:17897`，API `?full=true`）

| 数据集 | gated | usedStorage（API） | license tag | siblings | 与 GUIDE 对照 | 判定 |
|---|---|---|---|---|---|---|
| `JusperLee/LRS2-2Mix` | false | **16,470,508,590 B** | apache-2.0 | 3 | GUIDE：16.47 GB 压缩（= 16,470,508,590 B）、本地 tar 同字节 | ✅ |
| `JusperLee/RealSSA-Sim` | **manual** | **521,532,631,040 B**（521.53 GB） | cc-by-nc-nd-4.0 | **242** | GUIDE：521.53 GB / 242 files / 240 shards | ✅ |
| `JusperLee/RealSSA-Real` | **manual** | **12,842,268,608 B**（12.84 GB） | 无 license tag（card 研究条款） | **422** | GUIDE：12.84 GB / 422 文件 / 无通用 license | ✅ |
| `JusperLee/LRS3-2Mix` | false | 0 B | apache-2.0 | 2（README+`.gitattributes`） | GUIDE：仅 README 622 B，无数据 | ✅ |
| `JusperLee/VoxCeleb2-2Mix` | false | 0 B | apache-2.0 | 2（README+`.gitattributes`） | GUIDE：仅 README 756 B，无数据 | ✅ |

本地数据侧（与 GUIDE/PHASE1_REPORT 一致）：`/root/data/lrs2_2mix/lrs2.tar.gz` = **16,470,508,590 B**
（= HF usedStorage，Phase 1 已核 sha256 `ebdfdf31…767e`）；解压 `lrs2_rebuild/` ≈ 17 GB（成员 17,390,672,317 B）。

### 2.3 中间证据 ↔ Phase 1 基线（v0 快照；结论已被 §2.4 报告核对取代）

已出现的中间证据与 Phase 1 数字**自洽**（同一评测实现）：

| 证据 | 数字 | 与 Phase 1 基线关系 |
|---|---|---|
| P2-F 独立复算（`P2F-results.json`，n=50） | CSV avg **SI-SNRi 7.029 / SDRi 7.713**；独立实现 vs `look2hear.metrics.core` vs CSV，max\|Δ\| = **4.8e-8 / 1.8e-7 dB** | 与 Phase 1 全量 3000 的 **7.03 / 7.71** 一致；指标实现互证 ✅ |
| P2-A 20 条锚点（`P2A-anchor-first20.log`） | SI-SNRi **7.91** / SDRi **8.55** | = Phase 1 20 条子集 7.91/8.55 ✅ |
| P2-B 300 条（`P2B-summary-b300.json`、`logs/P2B-*-b300.log`） | small whole **6.92 / 7.49**；small windowed **5.00 / 5.76**；big whole **7.89 / 8.40**；big windowed **5.56 / 6.26** | 全量 7.03/7.71 的子集波动范围内；**整句 > 窗口化**（约 −1.9～−2.3 dB），big > small（+0.97 dB）——均待 P2-B 报告定稿 |
| P2-A 指标变体（同 20 条） | `zm0` 7.913、plain-SNRi 8.097、SDR512i 8.548、SDR256i 8.380、SDR64i 8.043、seg2 7.971、seg4 7.901、tgt-rms-scale 7.912 | 各变体相对基线 **< 0.2 dB（SDR 滤波长度除外）** → 不能解释 ~7 dB 差距（待 P2-A 报告定稿） |
| P2-F 数据自检（全 3000） | `max\|mix−(s1+s2)\|=3.05e-5`、peak=0.9 3000/3000、baseline SI-SDR(mix,s1)=**4.994**、mouths npz **20392**、eval pt 3000、`target==s1` 5/5 | 与 Phase 1 / GUIDE / P2-A 数据自检一致 ✅ |
| P2-C 全量转换（19:23Z，`P2C-convert.log` + 独立抽查） | tr/cv/tt = **20000/5000/3000**；三表（mix/target/near）等长；`target.json` 项 `[target,mouth,n]`、mouth `.pt` **[50,96,96,3] uint8**、n=32000；输出 **24 GB**（估 23.32 GiB） | 与 GUIDE/PHASE1_REPORT 的 LRS2 结构一致；QA `test_phase2_data.py` 由 5 skip → **5 passed** ✅ |

### 2.4 已落盘报告逐项核对（v1：P2-A…F + QA）

| 报告 | 关键主张 | 独立核对（文件/命令） | 判定 |
|---|---|---|---|
| `P2A-report.md` | 200 条×14 变体：基线 **6.95/7.62**；所有合理变体收益 **≤0.4 dB**；anchor **7.911/8.546**；mouth=s2 −16.71、置黑 +2.16、±5/12 帧 −3.4~−5.9、96 不裁 −0.04、1ch ±0.00、不归一化 −0.45、截 1 s −0.48、补零 3 s −5.13、峰/RMS 归一化 ±0.00；数据自检 3.05e-5 | `P2A-small-200.json` 逐值核对：`si_snr_i` 6.9547、`seg4` 7.2371(=+0.282)、`sdr512_i` 7.6224、anchor 7.9106/8.5463；数据自检与 P2-F 全 3000 一致；§3.6 对 P2-B/P2-F 的交叉引用一致 | ✅ 一致 |
| `P2B-report.md` | 300 条：small 6.916/7.492、small windowed 5.004/5.762、big 7.887/8.403、big windowed 5.569/6.270；配对 +0.971/−1.912/−2.318；视觉消融 −15.089/−12.894/−4.375；窗口化诊断 16.06 dB/corr 0.980 | 逐句 CSV 聚合、`[RESULT]`/summary JSON、`P2B-analysis.log`、`windowed_diag.py` 均核对；3000 基线行列序已由 owner 修正（§2.6） | ✅（P3 已修） |
| `P2C-report.md` | 28000 条、**23.36 GiB**、mouths 15516、sha256、7 tests、near fallback→K=0 | 独立验证：tr/cv/tt = 20000/5000/3000、三表等长、mouth `.pt` `[50,96,96,3]` uint8、`conversion_stats.json` 存在、**两个 sha256 与报告逐字节一致**、`test_lrs2_convert.py` **7 passed**、log 117 行 | ✅ 完全一致 |
| `P2D-report.md` | 冒烟 100 step/73 s/2.97 it/s/val 1.946；conf 保留 `repeats/shared`（F9）；**正式训练 19:26:41Z 启动**（单卡 bs4，PID 4002894） | `P2D-smoke.log`、`/root/tmp/p2d_smoke/.../conf.yml:40-41,83-84`、`ps` 确认 PID 运行中、`P2D-train.log` 已出 Trainer/参数表；eval smoke 5 条为未训练基线（报告已声明“仅证明管线”） | ⚠️ 报告 §5 “P2-C 数据未就绪”为陈旧表述（数据 19:24 已就绪、训练已启动）；训练结果待跑完 |
| `P2E-report.md` | DNSMOS 离线链路（speechmos/onnxruntime）、模型 sha256、参考值 3.276、**无 RealSSA 指标结论** | `P2E-dnsmos-*.json`、`P2E-smoke-official-loader.log`（3.2677/2.7496/2.3637/3.1704）、CSV 逐项一致；`REPRODUCTION_PLAN.md:77` 有 3.276±0.05；env delta 已并入 lock（§1.2） | ✅ 一致 |
| `P2F-report.md` | 对拍 **4.8e-8/1.8e-7 dB**；baseline 4.994；target=s1 Δ+24.5；PIT 0；lag 0；无削波；8k +0.11；**发布权重训练集=LRS3_complex_nonmoving** | `P2F-results.json` 逐项一致；**独立复核** `Experiments/checkpoint/Falcon/conf.yml:99-102`（LRS3_complex_nonmoving）、`:105`（exp_name Falcon）、`best_k_models.json` epoch=714、`PROVENANCE.md:37-38`、`README.md:211/228` 均属实；P5 基线笔误已由 owner 修正（§2.6） | ✅（P3 已修；结论成立） |
| `QA-report.md`（v0） | 白名单修复、官方权重指纹不变、94→102 collected、Phase2 守卫 | QA 中间跑 88/1/6（旧收集 + 报告证据检查未打补丁）；task-25 复跑 **96/6/0**；两条守卫 19:26Z **2 passed** | ✅（QA 终检独占重跑后以终值替换） |

### 2.5 关键结论：7.03/7.71 是“跨语料”结果（P2-F 归因，已独立复核）

P2-F 由仓库内证据链给出归因，本任务逐条复核均属实：

1. `Experiments/checkpoint/Falcon/conf.yml:99-102`：发布权重训练 run 的 `train_dir/valid_dir/test_dir` 均为
   `${DATA_ROOT}/small/DataPreProcess/**LRS3_complex_nonmoving**/{tr,cv,tt}`；`:105` `exp_name: Falcon`。
2. `Experiments/checkpoint/Falcon/best_k_models.json`：最优 ckpt = `epoch=714`；`studio/falcon_model/PROVENANCE.md:37-38`
   说明 `falcon.pt`/`falcon-big.pt` 分别由 epoch 714/715 导出 → **发布权重是 LRS3 派生数据训练**。
3. `README.md:211` 的 Falcon 行：LRS2-2Mix **14.1/14.7** 与 RealSSA/LRS3/VoxCeleb2 列并列；`README.md:228`
   写明 “Dolphin and AV-Mossformer2 were re-implemented as causal variants **under our training protocol**” →
   论文数字是“按 benchmark 分别训练”的口径，不是单一跨语料模型的 zero-shot 结果。

**含义**：本仓库的 **7.03/7.71 = LRS3-trained 官方权重在 LRS2-2Mix 上的跨语料泛化结果**，不是论文 LRS2-2Mix 列
（14.1/14.7）的复现；P2-A（指标/协议 ≤0.4 dB）、P2-B（big 仅 +0.97 dB、窗口化 −1.9~−2.3 dB）、P2-F（P1–P10 全部排除）
共同把“评测协议/指标/对齐/target/PIT/缩放”排除。要复现 14.1/14.7 需 **LRS2-2Mix 训练的 ckpt**（P2-D 进行中）或作者提供
benchmark-specific ckpt。本任务已在 `docs/REPRODUCTION_PLAN.md` 的 Phase 2/3 状态小节记录该结论并链接 P2-F；
**建议 Lead 在 FINAL/`REPRODUCTION_PHASE1_REPORT.md` 定稿时补同样标注**（超出 task-25 写作用域，未改）。

### 2.6 报告 P3 勘误（已由 owners 修复，本任务复核）

| 报告 | 勘误 | 状态 |
|---|---|---|
| `P2B-report.md:81` | 3000 条基线行 SI-SNR/SDR 对调（12.813↔12.023） | ✅ 已修为 **SI-SNR 12.023 / SI-SNRi 7.029 / SDR 12.813 / SDRi 7.713**，并加列序说明；本任务复核 CSV avg 行一致 |
| `P2F-report.md:101`（§4-P5） | mouth 消融基线误写为 N=50 的 7.535（实为 N=20 的 **7.911**） | ✅ 已修为 7.911 并注明“与 P3/P4 的 N=50 基线 7.535 是不同子集”；`P2F-results.json`/`protocol_check.py` 同步、§9 变更日志已记勘误 |
| `P2D-report.md` §5 | 仍列“P2-C 数据未就绪”为阻塞（实际 19:24 已就绪、19:26 训练已启动） | ⏳ 陈旧表述，不影响数据/结论；建议 owner 更新（已告知） |

---

## 3. 链接 / 路径审计

### 3.1 Markdown 相对链接

`docs/**/*.md` + 根 `README.md`/`README_zh.md` 共 **62 个文件、38 条非 http/# 相对链接，0 条真 404**
（4 条为正文里的字面量 `` `[...](...)` `` 占位，非链接：`docs/understanding/08-ops-tests-docs.md:273`、
`docs/verification/phase1/REGRESSION-docs.md:151`、`docs/verification/phase2/README.md:123`、
`docs/verification/phase2/REGRESSION-docs.md:140`，与 Phase 1 结论一致）。

### 3.2 反引号仓库路径

**910 条**（全 docs）含 `/` 且带已知扩展名的反引号路径，按“仓库根 + `look2hear/`、`studio/`、`tools/repro/`、
`DataPreProcess/`、`docs/verification/`、`docs/`、`configs/` 前缀 + 文档所在目录”解析后，**61 条 unresolved**，分类：

- **55 条为示意/复合路径（非缺陷）**：`exp_dir/conf.yml`、`exp_dir/best_k_models.json`、`mix.json/target.json`、
  `results/metrics_rank{rank}.csv`、`hf://...`、数据集内部路径（`metadata/*.json`、`audio_process/mix_2_spk_tr.txt`）、
  以及 `models/SISNRRepel.py` 这类相对 `studio/falcon_model/` 的简写。
- **1 条真实 P3**：`docs/REPRODUCTION_PLAN.md:138` 写 `DataPreProcess/realssa_to_falcon.py`，实际 Phase 1 落在
  `tools/repro/realssa_to_falcon.py`（同名脚本；PLAN 为计划期命名）。按本任务“PLAN 只加状态/链接”的约束**未改正文**，
  建议下一轮修正（不影响可复现性：README §3.1 命令用的是正确路径）。
- **2 条预期未来产物**：`docs/REPRODUCTION_PLAN.md:345/407` 的 `docs/REPRODUCTION_REPORT.md`（Phase 6 交付物，尚未到）。
- **1 条已解决**：`docs/verification/phase2/README.md:13` 的 `tools/repro/tests/test_lrs2_convert.py` 现已落盘（P2-C），
  QA 侧另有 `test_phase2_data.py`；README 已更新。
- **2 条已处理**：GUIDE 的 LRS3 深链（§3.3）；`docs/verification/phase1/REGRESSION-docs.md` 里 `.../lip_reading/lrs3.html`
  为历史记录（保留，不追改历史报告）。
- **P2 报告新增反引号路径 128 条（v1 追加审计）**：19 条 unresolved 均为复合/相对简写（`mix/target/near.json`、
  `P2E-dnsmos-synthetic.log/.json`、`version_0/metrics.csv`、RealSSA 数据集内 `scripts/load_inference.py`、
  `exp_dir/...` 等），**无真实 404**。

### 3.3 外链 HTTP 复核（经代理，2026-10-21T19:09Z）

| URL | HTTP | 处置 |
|---|---|---|
| `https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html` | **404**（上游站点问题；索引页自身也链到该 404） | GUIDE:327 已改为入口页 `.../lip_reading/`（200）并注明深链 404 |
| `https://www.robots.ox.ac.uk/~vgg/data/lip_reading/` | 200 | — |
| `https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html` | 200 | 保留 |
| `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html` | 200 | 保留 |
| 5 个 HF 数据集页（LRS2-2Mix / RealSSA-Sim / RealSSA-Real / LRS3-2Mix / VoxCeleb2-2Mix） | 均 200 | — |

### 3.4 Phase 1 遗留 3 项复核

| 遗留项 | 现状 | 判定 |
|---|---|---|
| `docs/verification/model-core.md:16` 引用不存在的 `runs/model-core-verify5.log` | 已改为 `runs/model-core-r3-verify.log`（存在） | ✅ 已闭环 |
| `docs/understanding/05-data-pipeline.md` changelog 引用 `docs/verification/{runs,scripts}/data_pipe` | 两个目录均存在（5 个日志/5 个脚本） | ✅ 已闭环 |
| GUIDE:327 LRS3 深链 404、GUIDE:394 “部分 `tar tzf`” + `/tmp/ds_cards/` | 本次修复（入口页；完整清单 249,670 条；卡片副本在仓库内 `dataset-cards/`） | ✅ 已闭环 |

---

## 4. 本次对文档的最小改动（仅状态/链接/证据措辞，无结论变化）

| 文件 | 改动 | 约束核对 |
|---|---|---|
| `docs/verification/phase2/README.md` | Phase 2 索引/状态/复跑命令（新建） | ✅ 本任务写作用域 |
| `docs/REPRODUCTION_PLAN.md` | 在 Phase 2 节后新增“Phase 2 执行状态（18:50Z）”小节、Phase 3 节末新增“Phase 3 状态”小节；v1 更新为 19:45Z 快照（P2-A…F 报告已落盘 + P2-F 跨语料归因 + P2-D 训练已启动 + RealSSA 待批），仍只含状态与链接 | ✅ 只加状态与链接，未改结论 |
| `docs/DATA_ACCESS_GUIDE.md` | §0 Real 行、§1 表、§4.2/§4.3/§5.2/§5.3 的 RealSSA 申请状态同步（9 处）；LRS3 链接 404→入口页；§11 证据措辞（完整 tar 清单/卡片仓库内路径）；§11 更新记录 | ✅ “如需”范围；未改数据结论 |
| `docs/verification/phase2/REGRESSION-docs.md` | 本文件 | ✅ 本任务写作用域 |

**未做**（按约束）：未改任何源码、未 commit、未切分支、未改他人 P2 报告/脚本/数据。

---

## 5. 复跑命令汇总（Phase 2 全链路；完整模板见 `phase2/README.md` §3）

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python

# 0) 环境自检
$ENV -V && $ENV -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())"
$ENV -c "import sys;sys.path.insert(0,'.');import look2hear.models,audio_train,audio_visual_test;print('imports OK')"

# 1) 本回归（task-25）
$ENV -m pytest studio/tests -q                                    # 期望 35 passed, 2 skipped
for f in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$f"; done   # 8/8
$ENV space/build.py --out /root/tmp/space-verify && node space/tests/check_i18n.mjs /root/tmp/space-verify  # i18n: 226
$ENV -m pytest tools/repro/tests -q                               # 19:29Z 实测 96 passed, 6 skipped（102 collected）

# 2) LRS2-2Mix 数据转换（P2-C；目标 /root/data/falcon_lrs2_2mix）
$ENV tools/repro/lrs2_2mix_to_falcon.py --src /root/data/lrs2_2mix/lrs2_rebuild \
     --out /root/data/falcon_lrs2_2mix --target s1 --near-mode fallback --jobs 8

# 3) LRS2-2Mix 评测（官方权重；small/big、整句/窗口化）
$ENV audio_visual_test.py --conf_dir configs/Falcon.yml --ckpt_path /root/.cache/falcon_weights/falcon.pt \
     --data_dir /root/tmp/lrs2_eval_tt_full --limit 300 --gpus 0
$ENV audio_visual_test.py --conf_dir studio/configs/FalconBig.yml --ckpt_path /root/.cache/falcon_weights/falcon-big.pt \
     --data_dir /root/tmp/lrs2_eval_tt_full --limit 300 --gpus 0

# 4) 7 dB 差距排查 / 协议独立验证（P2-A/P2-F）
$ENV tools/repro/phase2/gap_analysis.py --help
$ENV tools/repro/phase2/protocol_check.py --help

# 5) 训练（P2-D；正式训练前务必 nvidia-smi 独占 GPU）
#    19:26:41Z 已启动：单卡 bs4（PID 4002894，日志 P2D-train.log，cwd /root/tmp/p2d_train）
bash tools/repro/phase2/start_p2d_single_gpu.sh 4
# 训练 ckpt 评测（脚本已端到端验证）
bash tools/repro/phase2/eval_p2d.sh <ckpt> /root/tmp/lrs2_eval_tt_full 5 0

# 6) DNSMOS / RealSSA 预置（P2-E；RealSSA 数据 403，仅工具冒烟）
$ENV tools/repro/phase2/eval_dnsmos.py --help
$ENV tools/repro/phase2/eval_realssa_real.py --help
```

> 网络：HF 访问必须经 mihomo 代理 `http://127.0.0.1:17897`（`export https_proxy=... http_proxy=...`），
> 或使用 `tools/repro/fetch_data.py` 的 `--proxy`；直连/`hf-mirror.com` 在本沙箱不可用。

---

## 6. 待办与风险（v1 终检后仍开放）

1. **P2-D 正式训练运行中**（19:26:41Z 启动，单卡 bs4，PID 4002894）：**epoch 0 已完成（19:45Z）——val_pit_sisnr **6.73 dB**（cv）、test_pit_sisnr **6.81 dB**（tt 3000；`metrics.csv`）**，训练继续中；跑完/到可评测节点后用 `eval_p2d.sh` 复测并与官方权重（7.03/7.71）、论文（14.1/14.7）对比，结果由 P2-D 报告 v1 记录（本任务不等训练结束）。
2. **QA 终检独占重跑**：`pytest tools/repro/tests -q -rs`（当前 96/6/0；QA 报告 v1 落盘后以终值替换 §1 #5/#6）。
3. **RealSSA 纪律**：Sim/Real 数据仍 403（申请待批）；任何报告不得产出 RealSSA 指标结论，只能记录工具预置与
   合成冒烟（P2-E 已按此执行；P2-F/P2-A 亦未使用 RealSSA）。
4. **跨语料标注**：建议 Lead 在 `FINAL_PROJECT_UNDERSTANDING.md`/`REPRODUCTION_PHASE1_REPORT.md` 定稿时补
   “7.03/7.71 = LRS3-trained 官方权重在 LRS2-2Mix 的跨语料结果”（§2.5；本任务未改这两份文档）。
5. **P3 残留**：`docs/REPRODUCTION_PLAN.md:138` 的 `DataPreProcess/realssa_to_falcon.py` → 实际 `tools/repro/realssa_to_falcon.py`；
   `P2D-report.md` §5 的“P2-C 数据未就绪”陈旧表述。均不影响可复现性/结论。
6. **GPU**：P2-D 训练与其它评测共享 GPU0（外部进程占用）；后续全量评测/双卡训练需 Lead 协调独占窗口。

---

## 7. Changelog

- **2026-10-21T19:50Z（v1.1）**：P2-D 训练 epoch 0 完成（val 6.73 / test 6.81 dB，`metrics.csv`），§6 更新；其余同 v1。
- **2026-10-21T19:45Z（v1，终检）**：P2-A…F 报告全部落盘并逐项核对（§2.4）；P2-F 跨语料归因独立复核（§2.5）；2 处报告 P3 勘误由 owners 修复并复核（§2.6）；回归终值 **96 passed/6 skipped（102 collected）**；链接审计更新（62 文件/30 链接/0 真 404）；PLAN 状态小节升 19:45Z；README §1 全部状态 ✅/⚠️ 回填。
- **2026-10-21T19:31Z（v0.5）**：白名单修复后全量复跑 **96 passed/6 skipped（102 collected，146 s）**；QA 回信确认 AGENTS.md 授权注释 + 终检独占重跑计划；§1/§1.1 更新。
- **2026-10-21T19:28Z（v0.4）**：QA 已把 `AGENTS.md` 加入两处白名单，19:26Z 守卫复跑 2 passed（环境合规→守卫失败→闭环）；P2-A 200 条变体、P2-B visual ablation、P2-C 全量转换的中间证据已可交叉核对。
- **2026-10-21T19:25Z（v0.3）**：P2-C 全量转换完成（20000/5000/3000，24 GB），独立抽查 + QA 数据测试 5 passed；README §1 同步。
- **2026-10-21T19:19Z（v0.2）**：环境合规更新（lock 115 行 + AGENTS §2）导致两个 diff-scope 守卫因 `AGENTS.md` tracked-modified 失败；已通知 P2-G 加白名单，§1/§1.1/§1.2/§6 记录该状态。
- **2026-10-21T19:14Z（v0.1）**：P2-D/P2-E 报告落盘，补 §2.4 逐项核对（数字/证据一致）；README §1 同步。
- **2026-10-21T19:10Z（v0）**：首版。跑齐 Studio/Node/Space/imports/tools 回归（最终 82 passed/12 skipped 全绿，
  含两轮失败闭环记录）；完成 RealSSA 状态一致性修复（GUIDE 9 处）、HF 5 数据集事实复核、链接/路径审计
  （Markdown 0 真 404；反引号路径 61 条 unresolved 已分类）、Phase 1 遗留 3 项复核；列出 P2 报告落盘后的 v1 待办。
