# R7 终局验证 — Falcon-fork @ 6c1d5b7

> **2026-11-10 复核提示：**本文记录 2026-11-08 的测试与工作区快照。本轮服务器 CPU 结果为 166 passed / 2 skipped，跳过项依赖缺失的历史临时文件；证据及完整条件见现行状态。见[复现状态](../../../STATUS.md)。

只读验证。主机 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork`，解释器 `/root/miniforge3/envs/falcon/bin/python`，Node v22.23.3。
验证前后 `git status --short` 均为 **46 条**、`git diff --stat` 均为 **42 files changed, 2413 insertions(+), 478 deletions(-)**、`checkpoints/*.ckpt` 均为 **0**；未新增/删除任何跟踪或未跟踪条目（脚本只写入 `/tmp`）。

## 0 可交付判定

**代码/测试/证据三轴可交付；文档口径不可交付（1 处必须改）。**

| 维度 | 结论 |
|---|---|
| 测试全绿 | ✅ GPU 168/0、CPU 168/0、Studio 51+2 与 53+0、Node 8/8、Space 3/3 |
| 证据未污染 | ✅ 冻结证据 0 改动，测试不落 ckpt（0→0） |
| 断链 | ✅ 本轮改动的 33 个 `.md` 真实 404 = **0** |
| 数字残留 | ❌ 当时 **9 个文件 17 处**把「无 GPU 166 passed / 2 failed」当作当前事实 → **已修**（现统一为有/无 GPU 均 168 passed） |
| 判为**本轮引入** | 是。理由见 §3：使 CPU 转绿的两处修改（`test_studio_pytest_regression` 放宽 `skipped==2` 硬断言、指纹线程钉死）都在本轮未提交 diff 内，而"无 GPU 会 2 failed"的叙述也是本轮新写的 —— 属**轮内自相矛盾**，非历史遗留 |

结论（当时）：**先改这一处口径即可交付**——**已改并复核**：`CUDA_VISIBLE_DEVICES= python -m pytest tools/repro/tests -q` → **168 passed / 0 failed**（170.39 s），有 GPU 同样 168 passed。

## 1 八项实测

| # | 项 | 结果 | 判定 |
|---|---|---|---|
| 1 | `CUDA_VISIBLE_DEVICES=0 pytest tools/repro/tests -q` | **168 passed, 42 warnings in 147.12s**（总 148 s） | ✅ |
| 2 | `CUDA_VISIBLE_DEVICES= pytest tools/repro/tests -q` | **168 passed, 38 warnings in 170.47s**；前置断言 `cuda_available=False device_count=0` | ✅ 符合 168/0 预期 |
| 3 | `pytest studio/tests -q` 有/无 CUDA | 有：**51 passed, 2 skipped in 18.66s**；无：**53 passed in 30.39s** | ✅ 51+2 / 53+0 精确吻合 |
| 4 | Node `check_*.mjs/.cjs` + `web/*.test.cjs` | **8/8 PASS**（i18n、mouth_tracking、playback_resampler、spectrum、capture_dsp、tracking_ui、capture-worklet、video-worker） | ✅ |
| 5 | Space build + i18n + static | build 1 s → **54 文件**；`i18n: 226 Chinese strings translated`；`test_static.py` **3 passed** | ✅ |
| 6 | `checkpoints/*.ckpt` | **跑前 0 → 跑后 0**（`find` 复核 0；无 11-08 之后新增 ckpt） | ✅ 新增 `conftest.py` 守卫生效 |
| 7 | 证据未改 | `docs/verification/**` 下 `.csv/.log/.json` 改动 **0**；`git diff --name-only` 的 `csv\|log\|json\|pth\|pt` **为空**；暂存区为空 | ✅（例外见 §4） |
| 8a | 改动 `.md` 相对链接 | 33 文件 / **77 条相对链接 / 75 解析 / 2 "断"** | ✅ 2 条是正文里的字面量 `` `[...](...)` `` 占位（`phase2/README.md:140`、`REGRESSION-docs.md:150` 自述"4 条字面量占位"）→ **真实 404 = 0** |
| 8b | 残留旧数字 | `57 文件`、`610 行`、`Falcon.py:767/803/816` → **各 0 命中**；`166 passed` → **22 行 / 10 文件** | ❌ 见 §3 |
| 8c | 作废口径数字 | `.md` 命中 65 行（27 带"作废/旧/full3"标记）；**current-facing 文档仅 2 处，均带标记**；`README*.md` **0 命中** | ✅ 实质通过 |

要通过/不通过计：**7 项通过，1 项（8b）不通过**（8 拆为 a/b/c 时为 8/9 子项通过）。

## 2 改动面清点

- `git status --short | wc -l` = **46** = 42 已跟踪修改 + 4 未跟踪条目
- `git diff --stat` 末行：**42 files changed, 2413 insertions(+), 478 deletions(-)**
- 按目录（已跟踪修改计数）：`docs` 13 · `tools` 7 · `look2hear` 7 · 根文件 6（`audio_train.py`、`audio_visual_test.py`、`evaluated_mac_params.py`、`AGENTS.md`、`README.md`、`README_zh.md`）· `studio` 4 · `configs` 3 · `space` 1 · `project_page` 1
- 未跟踪目录：`docs/understanding/deepdive-20261108` **14 文件 / 761K**；`docs/verification/deepdive-20261108` **9 文件 / 58K**（7 `.json` + `lookahead-ab.txt` + `README.md`，**无 csv/log**）；`tools/repro/experiments` **13 文件 / 362K**；另 `tools/repro/tests/conftest.py`

## 3 断链与残留

**8b 不通过（本轮唯一硬伤）**。使 CPU 转绿的两处修改都在本轮未提交 diff 内：
`test_phase1_regression.py` 把 Studio 回归从 `assert skipped == 2` 放宽为"0 failed + 总量不下降"，注释明确写"无 GPU：53 passed + 0 skipped"；指纹测试钉死 `set_num_threads(64)`。
于是**"无 GPU 2 failed" 当前已被证伪**（实测 168/0），但 9 个文件仍以现在时断言它：

- `docs/STATUS.md:264`（"**168 passed 只在 ≥1 张 GPU 可见时成立**"）、`:269`（"**2 failed / 166 通过（共 168）**"）← 最显眼的公开口径
- `AGENTS.md:69`、`:127`
- `docs/REPRODUCTION_PLAN.md:394`
- `docs/verification/phase2/QA-report.md:12/:24/:231`、`REGRESSION-docs.md:13/:30/:269`、`README.md:128/:173`、`P0-FIX-report.md:10/:120`
- `docs/understanding/deepdive-20261108/10-ops-repro-git.md:13/:82`、`00-SYNTHESIS.md:160`

**已修（2026-11-08 收口）**：上述文件全部改为「有 GPU 168 passed / 无 GPU 同样 168 passed / 0 failed」，`STATUS.md` 的「只在 ≥1 张 GPU 可见时成立」前提句已删除；以下保留为 review 当时的现场记录。

判为**本轮引入**（轮内矛盾）：修改与叙述同属本轮，改完测试未回收叙述。`09-space-project-page.md:240/:400` 的 "166 **collected**" 与 `REPRODUCTION_PLAN.md:397` 的"旧文不能再当基线"属正当语境，不计入。

**8c 实质通过**：`2.7901/3.3285` 仅在 `REPRODUCTION_PLAN.md:341`（"full3 作废"）与 `DATA_ACCESS_GUIDE.md:14`（"旧 run"）出现于 current-facing 文档，均带作废标记；`STATUS.md:208` 明写 full3 作废，C9/C13 的"STATUS 仍并列引用 full3"已被修掉。其余命中在 `deepdive-20261024`、`PAPER_ALIGNMENT_20261030`、`REPRODUCTION_PHASE2_REPORT` 等**本轮未改动**的日期快照内，属历史存档。`9169 条` 是正确事实（RealSSA-Sim test 集规模），非作废值。

## 4 证据完整性

- `git status --short` **未出现**任何 `docs/verification/**` 下的 `.csv/.log/.json`（普通视图只显示目录项；`-uall` 展开后亦无）。
- **唯一例外（允许）**：本轮新增的 `docs/verification/deepdive-20261108/` 下 7 个 `.json` —— `canonical.json`、`canonical-4s.json`、`canonical-6s.json`、`ln-variant.json`、`lookahead-ab.json`、`streaming-consistency.json`、`video-ablation.json`（均 `??` 未跟踪，未修改任何既有证据）。
- `git diff --name-only | grep -E "\.(csv|log|json|pth|pt)$"` → **空**；暂存区 → 空。
- 冻结证据（`P2E-*.csv/.log`、`DNSMOS-*/summary.json`、`three-round-review-*.json`）内容零改动；`checkpoints/` 0→0，测试不再在仓库根落 `.ckpt`。

## 5 剩余风险（按严重度）

1. **【高·必须改】9 文件 17 处"无 GPU 2 failed"已为假**。任何人跑一条 CPU 命令即可证伪，直接削弱交付可信度。改法：把该口径统一为"CPU/GPU 均 168 passed / 0 failed；CUDA 隐藏后耗时约 170 s"，并删掉 `STATUS.md` 第 5 节"只在 ≥1 GPU 可见时成立"的前提句。 **（已处理）**
2. **【高·需用户决策】未跟踪目录不 commit → GitHub 上 404**。`STATUS.md`(15)、`REPRODUCTION_PLAN.md`(4)、`README.md`(2)、`README_zh.md`(2)、`AGENTS.md`(2)、`DATA_ACCESS_GUIDE.md`(2) 共 **27 个不同目标**指向 `deepdive-20261108/*.md`，而 `git ls-files` 对这三个未跟踪目录返回 **0**。本地链接检查全过，恰是因为文件在磁盘上；一推送就 404。**除非用户提交这 36 个未跟踪文件，此风险持续存在。**
3. **【中】`tools/repro/experiments/`（13 文件 / 362K）同为未跟踪**，若文档引用它亦有同样问题；且它是产物目录，需确认是否该入库。
4. **【低】耗时口径漂移**：文档写 GPU 148.07 s / CPU 154.63 s，本轮实测 147.12 s / 170.47 s。CPU 侧差 16 s（≈10%），建议改写成区间或"约 170 s"，避免再次被当固定基线。
5. **【低】`conftest.py` 为新增未跟踪文件**，是"测试不落 ckpt"的唯一保障；若提交时漏掉它，第 6 项的 0→0 立刻失效（87 个历史 `.ckpt` 正是这样堆出来的）。
