# Phase-1 回归 + 文档一致性核对（task-17 · `studio-frontend`）

> **终检快照：2026-10-21T17:30Z（第二轮，全绿）** · 分支 `repro/phase1`（HEAD `689555c`，未 commit、未切分支）
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / CUDA 2×A800）+ 系统 `node v22`
> 写入范围：仅本目录（`docs/verification/phase1/`）：本文件 + `REGRESSION-docs-evidence.json`；**未改源码/他人文档**。
> 状态：**终检完成（FINAL，第五轮 17:42Z sha 复核）**。最新文档 sha：`DATA_ACCESS_GUIDE.md`=`1683e943…`、
> `REPRODUCTION_PLAN.md`=`933077be…`、`T6-report.md`=`2c56a2d0…`、`REPRODUCTION_PHASE1_REPORT.md`=`92c64d58…`、
> `phase1/README.md`=`0a003dff…`、`T5-report.md`=`ed270313…`；**回归全部通过（tools/repro 82 passed）**；
> GUIDE/PLAN 的 P1/P2 已清零，仅剩 2 个其它文档的 P2 + 少量 P3（见 §6）。

---

## 1. 回归结果（终检实际执行，2026-10-21 17:15–17:16Z）

| # | 项目 | 命令 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | Studio Python | `/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q` | **35 passed, 2 skipped**（19.4 s，17:15Z） | ✅ 满足 ≥35/2 |
| 2 | Studio Node（8 个） | `for f in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$f"; done` | **8/8 rc=0**（i18n 8 checks/**363 条**；tracking_ui **23 条**；其余 PASS） | ✅ |
| 3 | Space 组装 + i18n | `python space/build.py --out /root/tmp/space-verify` → `node space/tests/check_i18n.mjs /root/tmp/space-verify` | `Assembled /root/tmp/space-verify`；**`i18n: 226 Chinese strings translated`**，rc=0 | ✅ |
| 4 | 导入面 | `PYTHONPATH=$PWD python -c "import look2hear.{models,layers,system,metrics,losses,datas,videomodels,utils}, audio_train, audio_visual_test"` | **10/10 OK** | ✅ |
| 5 | 新增测试 | `python -m pytest tools/repro/tests -q` | **82 passed / 82 collected**（66.1 s，17:28Z；17:15Z 首轮 76/1 的越界守卫已闭环，见 §1.1） | ✅ |
| 6 | Space Python 测试（环境边界） | `python -m pytest space/tests -q`（`falcon` 环境） | 24 passed, 4 errors `RuntimeError`（gradio/spaces 缺失） | ✅ 与“需独立 Space 环境”一致，不算回归失败 |

历史对照：16:35Z=70 passed → 16:58Z=77 passed → 17:08Z 出现 `configs/Falcon.yml` 越界注释 → 17:15Z=76 passed/1 failed
（守卫触发）→ 17:26Z QA 更新白名单并新增注释级断言 → **17:28Z=82 passed 全绿**。

### 1.1 越界守卫事件：已闭环（Lead 授权 + QA 白名单 + 注释级断言）

首轮 17:15Z 唯一失败为 `test_diff_scope_only_allowed_files`：`tracked modified: configs/Falcon.yml`。经查是 T1
（train-sys）为落实 `REVIEW-training-stack.md:48` P3-2 **经 Lead 授权**而加的 **2 行注释**（`# logger: csv …`、
`# gpus: [0,1] …`；`git diff --stat` = `configs/Falcon.yml | 2 +`，键值不变）。处置与证据：

- QA 已把 `configs/Falcon.yml` 加入 `ALLOWED_TRACKED`（`tools/repro/tests/test_phase1_regression.py:428-438`，注释明确
  “authorized by Lead: comment-only semantics docs”），并新增 `test_config_tracked_changes_are_comment_only()`
  （同文件 `:529-549`）：用 `yaml.safe_load(git show main:configs/Falcon.yml)` 与工作区对比，**只允许注释级改动**；
- 事件记录见 `QA-report.md:214-225`（授权说明 + 9 键 = 9 键 + 新断言）；T1-report 同步更新（sha `c64bdcc7…`）；
- 复跑：**17:28Z `pytest tools/repro/tests -q` = 82 passed**（新增 5 项测试后总数 82），守卫测试单独跑 `1 passed`。

结论：该 P2 已闭环，无需 Lead 再决定；配置改动为注释级、语义零变化（并有测试持续约束）。

并发说明：T3/T4 写入期（16:31Z）曾出现瞬时 `5 failed/65 passed`（见 `T2-combined-suite-note.log`），文件稳定后全绿；
本次终检为独占运行（17:15Z 后无实现者写文件），失败原因是**工作区状态**而非测试抖动。

---

## 2. `docs/DATA_ACCESS_GUIDE.md`（新版 sha `b0f3687c…`）核对

方法：HF API `?full=true` + `/tree/main?recursive=true`（经本机 mihomo 代理 `127.0.0.1:17897`）、T6 的 card 快照
`/tmp/ds_cards/`（sha256 与我独立抓取的 5 份 README 完全一致）、T6 `fetch_data.py --dry-run` 只读探测（真实 403）、
LRS2 完整 tar 流式分析（`T6-lrs2-structure.txt` + 我自己的 `tar tzf` 249,670 行清单）。明细见 `REGRESSION-docs-evidence.json`。

### 2.1 终检确认正确的关键事实

| Guide 主张（新版行号） | 独立证据 | 判定 |
|---|---|---|
| Sim/Real `manual` gated；本机 `nanless` 未获批 403；token 37 B（GUIDE:18-19,151,203） | API `gated: manual`；dry-run `GatedRepoError 403`；`whoami=nanless` | ✅ |
| RealSSA-Sim 242 files / **521.53 GB**；tr 202=441.42 / cv 19=39.97 / tt 19=40.14 GB（GUIDE:18,174） | tree：242 files、521,532,645,652 B；split 字节与 guide 逐字节一致 | ✅ |
| Sim 100,826 / 9,133 / 9,169；500/shard；4ch 16k PCM16；`mouths.pt`[T,96,96,3]；无 near 源；CC BY-NC-ND 4.0（GUIDE:18,174-175,179-186） | card 原文 + API license tag | ✅ |
| RealSSA-Real 422 files / **12.84 GB**；raw 11.38 GB；54 seg/40 rec/10 spk/8 scene/83.99 min；stereo 16k；face 224×224 25fps；mouth `[T,96,96]`; landmarks `[T,68,2]`; 无 clean target→不能 SI-SDR；无通用 license（GUIDE:19,206,222-247,232） | tree 逐目录计数 + card 原文 | ✅ |
| **LRS2-2Mix 为完整 2 s AV 包**（新版已修正）：16.47 GB 压缩 / **17.39 GB 解压**；`mix/s1/s2`（tr 20,000 / cv 5,000 / tt 3,000）；`mouths` npz(50,96,96)；faces mp4；landmarks；create_mix 脚本/清单；**无需 LRS2 原始视频**（GUIDE:17,23,99-135,137-163） | 我的完整 `tar tzf`（249,670 entries；成员总大小 17,390,672,317 B）+ T6-lrs2-structure.txt；mouth/landmark npz 实测 shape 见 §2.3 | ✅ |
| LRS2 HF 标签 apache-2.0 但源数据受 LRS2/TED 条款，不可再分发（GUIDE:97,317-328） | API license tag + card + 数据内容（BBC 来源） | ✅ |
| LRS3/Vox 公开但仅 README（622/756 B，usedStorage=0） | API：gated=false、usedStorage=0、siblings 2 个 | ✅ |
| gated 元数据可读、非 README 文件 403、不替用户申请 | dry-run 输出与脚本行为 | ✅ |

### 2.2 新版 GUIDE 的残留问题（第三轮：已基本清零）

第二轮提出的 GUIDE P1/P2 已被 T6 全部修正并经我复核：

- ✅ **P1 已修复**：原 `GUIDE:297` 的 “README.md:67 称 all audio is mono 16 kHz” 已删除；新版不再做该错误引用
  （README 实际在多处写明多声道/单声道输入的关系，见 `README.md:82/85/87`；guide 现按 LRS2/RealSSA 事实描述）。
- ✅ **P2 已修复**：`GUIDE:18/220` 改为“**242 files / 240 shards**”；`GUIDE:222` Sim 表单改为“**3 个 checkbox**”；
  `GUIDE:136` 的“旧计划更正”改为引用 `REPRODUCTION_PLAN.md §5.2（已修订）`。
- ✅ **LRS2 AV 事实正确**：`GUIDE:17,23,96,105-147,208` 与 T6-report/我的完整 tar 实测一致
  （16.47 GB 压缩 / 17.39 GB 解压、mix/s1/s2 tr20000/cv5000/tt3000、mouths uint8(50,96,96)、faces/landmark、target=s1、
  `mix==s1+s2`、8k 版本、raw_audio 20,392、无需 LRS2 原始视频、不可再分发）。

仍存的问题（均 P3）：
- **P3-1（GUIDE:327）**：LRS3 Oxford 链接 `.../lip_reading/lrs3.html` 实测 **HTTP 404**（索引页自身也链该 404 页，属上游
  站点问题）→ 建议改链索引页 `https://www.robots.ox.ac.uk/~vgg/data/lip_reading/`。
- **P3-2（GUIDE:394）**：证据描述仍写“部分 `tar tzf` 结构”，且 card 证据仍在 `/tmp/ds_cards/`（临时目录）→ 建议改为
  “完整清单见 `T6-report.md`/`T6-lrs2-structure.txt`”并把关键 sha 固化（本报告 evidence JSON 已固化一份）。
- **P3-3**：`GUIDE:17` 的 TL;DR 与 `GUIDE:18` 现已分别用 AV/240 shards 表述，内部自洽；未发现新事实错误。

### 2.3 我独立复核的 LRS2 完整 tar 事实（与 T6-report 一致）

`/root/data/lrs2_2mix/lrs2.tar.gz`：**16,470,508,590 B**，sha256 `ebdfdf31…767e`（与 HF LFS `x-linked-etag` 一致）。

```text
entries 249,670（files 249,621 + dirs 49）；成员总大小 17,390,672,317 B = 17.39 GB（压缩包 16.47 GB）
ext: .wav 188,392 / .npz 40,784 / .mp4 20,392 / .py 13 / .txt 11 / .mat 6 / .scp 3
audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}: tr 20,000 / cv 5,000 / tt 3,000 组（wav8k 同构）
mouths/ 20,392 个 npz：data uint8 (50,96,96)   ← 实测抽取
landmark/ 20,392 个 npz：data float32 (50,1,68,2)（T6 补充：少数帧 2 个人脸 list）  ← 实测抽取
faces/ 20,392 个 mp4：224×224、25 fps、50 帧（T6 解码核实）
raw_audio/{train,val,test}: 19,649 / 476 / 267 = 20,392 个 wav（16k mono PCM16，2.000 s）
create_mix/：create_wav_2speakers.m / activlev.m / maxfilt.m / mix_2_spk_{tr,cv,tt}.scp / mix_2_spk_min_* 列表
scp 每行 `<wav1> <snr1> <wav2> <snr2>`；mix 文件名 `<id1>_<snr1>_<id2>_<snr2>.wav`；T6 实测 mix==s1+s2（max|Δ|=3.05e-5）
train.txt 20,779 行；video_path.txt 20,468/20,469 行（wc 与 T6 计法差 1）；filename_input.csv 20,392 行
```

结论：**“LRS2-2Mix 纯音频/无视频”的旧说法必须废弃**；它是完整 AV 数据，`s1/s2` 即 scaled target，`mouths` 直接匹配
Falcon 的 `(50,96,96)` 输入（转 `.pt` 时补 channel 维）。

---

## 3. `docs/REPRODUCTION_PLAN.md`（新版 sha `345625df…`，17:15Z）一致性

### 3.1 已修正/正确的部分

- §4.2 数据表、§5.2 主体、§8 RealSSA-Real 容量已按 AV 事实重写：LRS2 完整 AV、Real 12.84 GB、raw_audio 20,392、
  faces 224×224/25fps、mouths (50,96,96)、landmark 近似 (50,1,68,2)、`mix==s1+s2` 等——与 T6-report/我的实测一致 ✅。
- §1.1 上游同步（4 commits、origin/main）、§1.2 环境（Python 3.11.17/torch 2.8.0+cu128/PL2.6.6/39 个验证脚本）、
  权重缓存、Studio 35/2、Node 8/8、space 226 条 i18n、§11.1 四个 commit 描述、§11.2 RealSSA 两行、MACs 10.611G、
  README 指标数字 7.7/8.2/14.1/14.7/16.4/16.7/10.5/11.8/3.276 均复核通过 ✅。

### 3.2 新版 PLAN 一致性（第五轮：P2/P3 基本清零）

第二轮提出的 PLAN P2 全部由 Lead 修复并经我复核（`PLAN` sha `933077be…`）：

- ✅ §5.2 标题/R6 已改 AV 口径；TL;DR、§5.2、§8（17.39 GB）、Phase3、§11.2、待办勾选均已按 AV 事实更新；
- ✅ 视觉骨干 F12 状态改为 “✅ 已从 falcon.pt 导出并验证”；§6.2/§6.3 引用、`bounded()`、R12 参数口径均已修正；
- ✅ RealSSA-Real 12.84 GB、`mix==s1+s2`、s1/s2 target convention、LRS2 完整 AV 均正确。

现仅剩 P3（不影响复现结论）：
| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| P3-1 | PLAN:107-109 | 资产表仍用 🟡 表示 “可获取但需动作”（gated 审批/公开已下载）——属状态快照语义，可接受 | 可选：加“状态截至 2026-10-21” |
| P3-2 | PLAN:112 | “LRS2/LRS3/VoxCeleb2 原始语料 🟡 用于生成嘴部 crop/视觉输入”——LRS2 已含视觉，不再需要原始语料 | 改为“仅 LRS3/Vox 需要原始语料” |

## 4. 其它 phase1 文档与证据

- **T6-report.md**（`35b2a892…`）：内容与我的独立 tar 分析一致，且诚实指出 Lead 给的 20,396/84,017 应为 20,392/84,000；
  其 §7 明确建议 Lead 修订 PLAN 旧结论（与 §3.2 一致）。✅
- **T6-lrs2-structure.txt**（`3f8bd1fc…`）：完整清单/格式/混音公式证据；本轮已用它交叉验证 GUIDE 与 PLAN。✅
- **QA-report.md**（`bcac4c71…`）：Gate 表第 6 项在 16:37 时点写 “T1–T6+REVIEW+REGRESSION 均已落盘”，当时 T6/REGRESSION
  尚未落盘；现在 T6-report ✓、REGRESSION=本文件 ✓，该项自动成立（P3 时点差）。其 P2-3 对 video 指纹约定（baseline
  training=True、我独立复现 `2afcea08…`/sum 7832.2563，eval 得 `41e79f0d…`/sum 5955.73）与 phase1 README Gate 5
  的表述仍建议互相引用（见 §6 P2-8）。
- **REVIEW-training-stack.md**：P3-2 建议在 `configs/Falcon.yml` 加 logger 注释，正是本次 `test_diff_scope_only_allowed_files`
  失败的直接触发来源；P3-4 的 `--conf_dir` 默认值已闭环（T1 改为 `configs/Falcon.yml`）。
- **REPRODUCTION_PHASE1_REPORT.md**：§1/§5 的 “70 passed” 已被 77 取代（P3 时点差）；§9 相关文档在 T6/本文件落盘后齐全。
- phase1 README Gate 表：1 ✅（77/76+1）、2 ✅（integration-smoke：1 epoch/val_loss 12.188/ckpt）、3 ✅（eval smoke CSV）、
  4 ✅（35/2 + Node 8/8）、5 ⚠️（eval/train/sep 与 baseline 相同；video 需按 training-mode 约定，见 §6 P2-8）、
  6 ✅（T1–T6/QA/REVIEW/REGRESSION 齐全）。

---

## 5. docs 内部链接 / 路径审计（终检）

- **Markdown 相对链接**：`docs/**/*.md` 36 个文件共 4 条非 http/# 相对链接，**0 条 404**（1 条误报：08 文档表格里的
  字面量 `` `[...](...)` ``，不是链接）。新参考目标 `docs/REPRODUCTION_PHASE1_REPORT.md`、`docs/DATA_ACCESS_GUIDE.md`、
  `docs/verification/phase1/T6-report.md` 均存在。
- **反引号仓库路径**（832 个抽查）：真实缺口 2 处——`docs/verification/model-core.md:16` 的
  `docs/verification/runs/model-core-verify5.log` 不存在（脚本存在）；`docs/understanding/05-data-pipeline.md` changelog
  的 `docs/verification/data_pipe` 不存在（实际在 `scripts/data_pipe/`、`stubs/data_pipe/`）。
- **外部 URL**（代理实测）：GUIDE 的 5 个 HF 数据集页均 **HTTP 200**；Oxford VGG：LRS2 200、VoxCeleb2 200、
  LRS3 页面 **404**（上游站点；见 §2.2 P3-1）。
- 计划/未来文件（非 404 bug）：`docs/REPRODUCTION_REPORT.md`（Phase 6 产出）；环境/数据类故意不存在：
  `Experiments/tensorboard_logs/`、`studio/.venv`、`configs/NASSv2_lrs3_complex_nonmoving.yml`、
  `Experiments/checkpoint/Falcon/{last.ckpt,results/metrics_combined.csv}`、`studio/models/`、`docs/reviews/roundN/`。

---

## 6. 残留问题清单（终检第五轮，已收敛）

**P1：无。**

**P2（均属 phase1 之外的旧文档，建议后续顺手补）**
1. `docs/verification/model-core.md:16`：`runs/model-core-verify5.log` 缺失（脚本 `scripts/model_core_verify5.py` 存在）。
2. `docs/understanding/05-data-pipeline.md`（changelog）：`docs/verification/data_pipe` 路径不存在（实际在 `scripts/data_pipe/`、`stubs/data_pipe/`）。

**P3**
1. `GUIDE:327` LRS3 Oxford 链接 404（上游站点）；`GUIDE:394` 仍写“部分 `tar tzf`”+`/tmp/ds_cards` 临时证据。
2. `PLAN:107-109/112` 资产表状态快照语义与“原始语料”行（LRS2 已含视觉）。
3. `REPRODUCTION_PHASE1_REPORT.md:3` 日期 2026-10-22 与 UTC mtime 差一天（时区），建议标注时区。
4. `QA-report.md` 的 config 事件已闭环（Lead 授权 + QA 白名单 `test_phase1_regression.py:428-438` + 注释级断言
   `:529-549`）；17:28Z 全套 **82 passed**。

## 7. 已关闭的等待项

- ✅ LRS2 下载完成（16,470,508,590 B；sha256 `ebdfdf31…767e` = HF LFS `x-linked-etag`；`T6-data-fetch.log`）。
- ✅ `T6-report.md`、`T6-lrs2-structure.txt`、`DATA_ACCESS_GUIDE.md`（AV 修正版）均已落盘并复核。
- ✅ `REPRODUCTION_PLAN.md` 已按 AV 事实重写 §5.2/§8（但 §11.2/TL;DR 等残留见 §3.2）。
- ✅ `configs/Falcon.yml` 注释级改动：Lead 已授权，QA 已加白名单 + 注释级断言，82 passed（§1.1）。**无待决定项。**

---

## 8. 证据与复跑命令

```bash
ENV=/root/miniforge3/envs/falcon/bin/python
$ENV -m pytest studio/tests -q
for f in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$f"; done
$ENV space/build.py --out /root/tmp/space-verify && node space/tests/check_i18n.mjs /root/tmp/space-verify
$ENV -m pytest tools/repro/tests -q
PYTHONPATH=$PWD $ENV -c "import look2hear.models, look2hear.layers, look2hear.system, look2hear.metrics, look2hear.losses, look2hear.datas, look2hear.videomodels, look2hear.utils, audio_train, audio_visual_test; print('OK')"
export https_proxy=http://127.0.0.1:17897 http_proxy=http://127.0.0.1:17897
$ENV tools/repro/fetch_data.py --dataset all --dry-run --proxy http://127.0.0.1:17897
sha256sum /tmp/ds_cards/JusperLee__*/README.md /root/tmp/lrs2-full-list.txt docs/DATA_ACCESS_GUIDE.md docs/REPRODUCTION_PLAN.md
tar tzf /root/data/lrs2_2mix/lrs2.tar.gz > /root/tmp/lrs2-full-list.txt   # 249,670 entries
```

证据附件：`docs/verification/phase1/REGRESSION-docs-evidence.json`（HF API 数字、split 字节、5 份 card sha256、
完整 tar 统计/抽样、baseline/T5 指纹、最新文档 sha256、终检回归结果与失败项）。
本轮只新增本目录两个文件；未改源码（`git status` 中源码改动均来自其他 T 任务）。

---

## 9. 终检记录（最新第五轮 sha，2026-10-21T17:42Z）

| 项 | 值 |
|---|---|
| 分支/HEAD | `repro/phase1` @ `689555c`（未 commit） |
| `DATA_ACCESS_GUIDE.md` | sha256 `1683e943297e307558639ba4b1a24756b153239aa52a9b00ee7a090df8bd19a2`（17:34Z） |
| `REPRODUCTION_PLAN.md` | sha256 `933077bec0c63b8d0c5ee6703c4f2d6f77640183ef7e98bbdeb21a52bed5abc0`（17:40Z） |
| `T6-report.md` | sha256 `2c56a2d0a70163f034220dafe5a8ed679f4b7bc3e071db5f1b2ce446ef0f749a`（17:35Z） |
| `REPRODUCTION_PHASE1_REPORT.md` | sha256 `92c64d5856863f4ac2ddb66b0b1860f902b16893366e2cff28d1d308fe921d93`（17:37Z，82 passed） |
| `phase1/README.md` / `T5-report.md` | `0a003dff…` / `ed270313…`（已加入 video 指纹 training=True 约定说明） |
| `QA-report.md` / `T1-report.md` | `e00eea0a…` / `c64bdcc7…`（含 config 授权与白名单说明） |
| 其它 | `T6-lrs2-structure.txt`=`3f8bd1fc…`；`baseline-fingerprint.json`=`df78be29…`（未变）；LRS2 tar sha256 `ebdfdf31…767e`（= HF LFS） |
| 回归 | studio 35/2 ✅；Node 8/8 ✅；space i18n 226 ✅；imports 10/10 ✅；tools/repro **82 passed/82 collected** ✅（17:28Z） |
| 文档结论 | GUIDE：**无 P1/P2**；PLAN：**无 P1/P2**（仅 2 条 P3 状态语义）；phase1 README/T5 指纹约定已补；仅剩 2 个旧文档 P2 证据路径（§6）。**可进入 R5/FINAL。** |
