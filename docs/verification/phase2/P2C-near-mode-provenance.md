# P2C 补充：`near.json` 的 s2 负样本来源与 `--near-mode other-speaker` 复现说明

> 任务：P0-3（`task-12`）· owner `data-pipeline` · 2026-10-24
> 分支 `repro/phase2`（HEAD `2beec38`）；未 git commit；未改动 `/root/data/falcon_lrs2_2mix`（只读）。
> 相关：`docs/verification/phase2/P2C-report.md`、`P2C-convert.log`、`P2D-report.md`、`P2D-near-train.log`、
> `docs/REPRODUCTION_PHASE2_REPORT.md` §4.1、`tools/repro/lrs2_2mix_to_falcon.py`。

---

## 0 一句话结论

**磁盘上 `/root/data/falcon_lrs2_2mix/<split>/near.json` 的 K=1 "s2 作 near" 语义，原先由一条未入库的一次性命令生成（mtime 2026-10-22T16:37:54），导致 `conversion_stats.json`/P2-C 报告/QA 报告仍写 fallback(K=0) 的文档-磁盘漂移；现已在 `tools/repro/lrs2_2mix_to_falcon.py` 增加 `--near-mode other-speaker`，可确定性地、一键重建与磁盘**逐字节同构**的 `near.json`，并用 5 条真实样本 + 3 个不依赖 24G 数据的单测验证。**

---

## 1 当前磁盘状态（2026-10-24 只读实测）

| 项 | 值 | 证据 |
|---|---|---|
| 数据集 | `/root/data/falcon_lrs2_2mix` | — |
| `near.json` 条数 | tr 20000 / cv 5000 / tt 3000（与 mix/target 等长） | `[验证]` `json.load` |
| 每条形态 | `[[ "<abs>/audio/wav16k/min/<split>/s2/<stem>.wav", 32000 ]]`（K=1，外层 1 元素） | `[验证]` 前 2 条 + 全量逐条检查 |
| near 目标 | `--target s1` 约定下 = 同一条 mixture 的 **s2 源波形**（不是 mix、不是随机片段） | `[验证]` `near == s2` 逐样本 `max|Δ|=0.0`；`near == mix` `max|Δ|≈0.46–0.79` |
| `near.json` mtime | **2026-10-22T16:37:54Z**（三个 split 同秒） | `[验证]` `stat` |
| `conversion_stats.json` mtime / 内容 | 2026-10-21T19:24:05Z；其中 `near_mode: "fallback"`、`generated_at: 2026-10-21T19:23:28Z` | `[验证]` |
| P2-C 转换日志 | `P2C-convert.log:112` 明确 "near=fallback identical to mix sentinel for all 3 splits"（2026-10-21T19:28Z） | `[文档]` |
| 训练期证据 | `P2D-near-train.log:15-18`：`Loaded near.json with 20000 entries` / `20000/20000 samples have near-speaker entries` / `Near file access OK: …/tr/s2/…wav` | `[验证]` |

结论：**`falcon_lrs2_2mix` 的 `mix/target/mouths` 仍是 P2-C 的产物（2026-10-21），只有 `near.json` 在 2026-10-22 被原地替换成 s2 负样本**；`conversion_stats.json` 与 P2-C/QA 报告没有同步更新，因此出现"报告说 K=0、磁盘实际 K=1"的漂移。

---

## 2 为什么会漂移（历史时间线）

| 时间（UTC） | 事件 | 证据 |
|---|---|---|
| 2026-10-21 19:22–19:24 | P2-C 全量转换完成：`near.json` = fallback（`[[mix,n]]`，Dataset 跳过 → K=0）；`conversion_stats.json` 记录 `near_mode=fallback` | `P2C-convert.log:112`、`conversion_stats.json` |
| 2026-10-21 19:26 | P2-D 正式训练启动（单卡 bs4、100 epoch），`use_near: true` 但数据 K=0 → 对比损失只有正项 | `REPRODUCTION_PHASE2_REPORT.md` §4.1 |
| 2026-10-22 **16:37:54** | **一次性命令原地重写 `{tr,cv,tt}/near.json`**：每条改成该 mixture 的 s2 源 wav（K=1）；仓库中没有任何 committed 脚本能产出该形态 | `stat` mtime；`git log -S`/grep 无生成器 |
| 2026-10-22 18:21 | 提交 `6d2df43` + `141c751`：新增 `run_contrastive_after_training.sh` 与 `LRS2_2Mix_near_s2.yml`（`use_near: true`，从 100 epoch ckpt 续训到 130） | `git show --stat` |
| 2026-10-22 19:55 → 2026-10-23 03:12 | contrastive 微调运行完成（130 epoch），日志显示 K=1 生效；结果 test SI-SNRi **8.858** / SDRi **9.270**（无 near 基线 8.618/9.042，**+0.24/+0.23 dB**） | `P2D-near-train.log`、`REPRODUCTION_PHASE2_REPORT.md:62-70` |

原因判断（`[推断]`，有 mtime/日志/git 证据支撑）：当时为了尽快启动 contrastive 微调，用一条临时 Python 命令把 fallback 的 `near.json` 原地改成了 "s2 负样本"，但没有把它固化成转换器选项，也没有回写 `conversion_stats.json`/P2-C 报告。于是：
- **P2-D 的 contrastive 结果（+0.24/+0.23 dB）的数据来源就是这个未入库的改写**；
- 任何人按 P2-C 文档重跑只会得到 K=0，**无法复现该实验**。

> 语义提醒：s2 是**同一 mixture 里的干扰源波形**（远场、干净、本身已在 mix 中），**不是**论文意义上的 near-field 泄漏波形；LRS2-2Mix 官方没有发布 near 源。任何使用该 near 的训练结果都必须声明这是 "other-speaker / non-official negatives" 近似（见 `lrs2_2mix_to_falcon.py` 模块 docstring 的 `--near-mode` 段）。

---

## 3 新增选项：`--near-mode other-speaker`

实现：`tools/repro/lrs2_2mix_to_falcon.py`（689 行，sha256 `e6e5da92…`）

- CLI：`--near-mode {fallback,raw_audio,other-speaker}`（默认仍为 `fallback`）。
- 语义：**K=1、确定性、不依赖随机/seed**：
  - `--target s1`（默认）→ 第 i 条 near = 第 i 条 mixture 的 **s2** 源 wav；
  - `--target s2` → 第 i 条 near = 该 mixture 的 **s1** 源 wav；
  - 路径来源：`<src>/audio/wav16k/min/<split>/{s1,s2}/<stem>.wav`（`Path(...).resolve()` 绝对路径）。
- `near.json` schema 与磁盘完全同构：`[[abs_other_speaker_wav, n_samples]]`，顺序 = `mix.json` 顺序（sorted stem），`n_samples = sf.info(wav).frames`。
- 校验：`validate_outputs` 对 other-speaker 额外检查 K=1、near 路径必须等于期望的 s1/s2 源文件、16 kHz、`frames ≥ 32000`、`near.json` 的 n 与 wav 帧数一致。
- `--dry-run`：不写任何文件；额外打印 `planned near entries`（按 split 的 parsed 数，total）与 `target=s1 -> near=s2` 方向说明。
- `--near-per-sample` / `--seed` 仅对 `raw_audio` 生效；`other-speaker` 恒 K=1。
- `conversion_stats.json` 的 `near_mode` 会如实写 `"other-speaker"`（重跑后即消除 stats 与 JSON 的不一致）。

### 3.1 重建当前磁盘数据的官方命令（**本次未执行**，避免改写 24G 主数据）

转换器是 re-entrant 的：`mix/target/mouths` 已存在且大小合理时只跳过，**只会重建三份 JSON**。因此如需把磁盘上的 `near.json` 变成"有据可查的产物"，执行：

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/lrs2_2mix_to_falcon.py \
  --src /root/data/lrs2_2mix/lrs2_rebuild \
  --out /root/data/falcon_lrs2_2mix \
  --splits tr,cv,tt --near-mode other-speaker --jobs 16 --validate-samples 200
# 预期：conversions 全部 skipped_existing（不重写 wav/mouth）；三份 near.json 与当前磁盘内容一致；
#       conversion_stats.json 的 near_mode 变为 "other-speaker"
```

> 本次任务**没有**执行该命令（约束：不改 `/root/data/falcon_lrs2_2mix`）。因此当前磁盘 `conversion_stats.json` 仍写 `fallback`，属**已知且已解释**的漂移，直到有人显式重跑上面的命令。

### 3.2 小规模等价性验证（本次已执行）

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/lrs2_2mix_to_falcon.py \
  --src /root/data/lrs2_2mix/lrs2_rebuild --out /root/tmp/lrs2_otherspk_check \
  --splits tt --limit 5 --near-mode other-speaker --overwrite --validate-samples 5
```

输出要点：

```
== dry-run near-mode... （非 dry-run 时无此行）
== converting split tt (expected 5 samples, jobs=8)
  [tt] INDEX READY: /root/tmp/lrs2_otherspk_check/tt/mix.json (5 entries)
== validation (sampled) ==
  tt: checked=5 length_ok=5 mouth_ok=5 mix_sum_ok=5 max_mix_sum_err=3.05e-05 near_ok=5 problems=0
```

逐条 diff（新产物 vs 磁盘 `/root/data/falcon_lrs2_2mix/tt/near.json` 同 id 条目）：

| # | 新产物 near 路径（节选） | 与磁盘同 id 条目 | `near == s2` | `near == mix` |
|---|---|---|---|---|
| 0 | `…/tt/s2/6330311066473698535_00011_0.53084_…_-0.53084.wav` | **完全一致** | `max|Δ|=0.0` | `0.460` |
| 1 | `…/tt/s2/6330311066473698535_00011_2.6743_…_-2.6743.wav` | 完全一致 | 0.0 | 0.736 |
| 2 | `…/tt/s2/6330311066473698535_00011_2.9639_…_-2.9639.wav` | 完全一致 | 0.0 | 0.789 |
| 3 | `…/tt/s2/6330311066473698535_00011_3.0079_…_-3.0079.wav` | 完全一致 | 0.0 | 0.734 |
| 4 | `…/tt/s2/6330311066473698535_00011_3.2108_…_-3.2108.wav` | 完全一致 | 0.0 | 0.687 |

→ **5/5 条 near 条目（路径 + n_samples）与当前磁盘逐条完全相同**；且 near 波形逐样本等于 s2 源 wav（`max|Δ|=0.0`）、明显不等于 mix。**schema 无不一致。**（临时产物在 `/root/tmp/lrs2_otherspk_check`，可删。）

### 3.3 单测（不依赖 24G 数据）

`tools/repro/tests/test_lrs2_convert.py` 新增 3 个用例（tmp 目录造 2 对假 s1/s2 wav + mouths）：

- `test_near_mode_other_speaker_target_s1`：K=1、绝对路径、路径 == `<src>/.../s2/<stem>.wav`、n=32000、near==s2（0.0）且 near!=mix；`conversion_stats.near_mode=="other-speaker"`、`validation.problems==[]`。
- `test_near_mode_other_speaker_target_s2`：`--target s2` 时 near == s1、target 音频 == s2 字节、mouth 属于 s2 utterance。
- `test_near_mode_other_speaker_dry_run_reports_counts`：`--dry-run` 不创建 out 目录，stdout 含 `near-mode=other-speaker` / `planned near entries` / `target=s1 -> near=s2`。

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_lrs2_convert.py -q
# 10 passed in 57.32s   （原 7 + 新增 3）
```

---

## 4 P2-D contrastive 实验的数据来源（+0.24/+0.23 dB）

| 项 | 值 |
|---|---|
| 训练配置 | `tools/repro/phase2/LRS2_2Mix_near_s2.yml`（`use_near: true`、`train_dir=/root/data/falcon_lrs2_2mix/tr`、从 100 epoch ckpt 续训到 130） |
| 训练数据 | `/root/data/falcon_lrs2_2mix`（单目标、target=s1、**near.json = s2 负样本 K=1**） |
| near 来源 | 2026-10-22T16:37:54 的未入库原地改写；**自本文件起**可用 `--near-mode other-speaker` 一键复现（§3.1） |
| 结果 | test SI-SNRi **8.858** / SDRi **9.270**（无 near 基线 8.618/9.042 → **+0.24 / +0.23 dB**） |
| 报告口径 | 必须声明：near 是 "other-speaker / non-official negatives" 近似，不是 near-field 泄漏源；K=1；训练规模单卡 bs4/130 epoch，与论文 10×A800/1000 epoch 不同 |

---

## 5 变更文件清单（未 commit）

| 文件 | 状态 | 说明 |
|---|---|---|
| `tools/repro/lrs2_2mix_to_falcon.py` | 修改（689 行，sha256 `e6e5da9288aa353061356354698b8f77eab5f0a03f60a2e1467bd61d4b2cb920`） | 新增 `--near-mode other-speaker` + docstring + validate + dry-run 计数 |
| `tools/repro/tests/test_lrs2_convert.py` | 修改（318 行，sha256 `505204b55fcda0a530874df11654a30ecfc8b7cd386f2c3f15ec8d3018f005ed`） | 新增 3 个 other-speaker 用例 |
| `docs/verification/phase2/P2C-near-mode-provenance.md` | 新增（本文件） | 漂移来源/时间线/复现命令/验证 |
| `docs/verification/phase2/P2C-report.md` | 末尾追加指针 | 见 §11 之后 |

> 未 git commit；未改 `/root/data/falcon_lrs2_2mix`；临时验证产物在 `/root/tmp/lrs2_otherspk_check`（仓库外，可删）。
