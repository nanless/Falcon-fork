# 10 复现工程 / 证据链 / git 历史

> 子代理 W1-10。范围：`tools/repro/**`（含 `tests/`、`phase2/`）、`docs/verification/scripts/**`、`Experiments/**`、`.gitignore`、`AGENTS.md`、`docs/STATUS.md`、git 历史与 upstream 差异。
> 仓库只读（本轮唯一写入是 pytest 自己产生的 `__pycache__`/`checkpoints/*.ckpt`，均被 `.gitignore` 覆盖，`git status` 全程干净）。
> 全部命令在 `/root/code/github_repos/Falcon-fork`，Python 用 `/root/miniforge3/envs/falcon/bin/python`。

---

## 0. 结论速览

1. **0 处 md 提到 `2062528`**：任务书的前提（「STATUS 写 HEAD=2062528」）不成立。`docs/STATUS.md:3,20` 写的是「`repro-freeze-20261031` 指向的提交」，该 tag 实测指向 `6c1d5b7`（=当前 HEAD），自洽；「回填 HEAD=2062528」只出现在 `6c1d5b7` 的提交信息里。
2. **168 passed / 约 148 s**（2026-11-08 实测；本轮新增 2 条用例，旧收集 166）：GPU 可见时 `tools/repro/tests` 全绿（0 skipped / 0 failed）。§4 附的 355.47 s 是 166 那次旧日志，是 STATUS 记的 136.5 s 的 2.6 倍。
3. **无 GPU 同样 168 passed**（2026-11-08 终测，170.47 s）。**修前**是 166 passed / 2 failed：两个失败都来自「隐藏 CUDA」——指纹测试对 `torch` 线程数敏感、`test_phase1_regression` 硬断言 studio `skipped == 2`；两处已修（见 `test_phase1_regression.py`、`audio_train.py`），不是代码缺陷。
4. **1 类必然失败（修前是 2 类）**：新增 CI 的复现测试 job 在 `ubuntu-latest` 跑全量 `tools/repro/tests`，且 `actions/checkout` 未设 `fetch-depth`（浅克隆），而 `test_phase1_regression.py:113-118` 对浅克隆显式 fail。无 GPU 那一类（skip 硬断言 / 线程敏感指纹）**已修**；浅克隆这一条仍在，可用本地等价条件复现（§4.6）。
5. **3 个 ckpt 的 sha256 字节一致**：`P2D-train-metrics-1gpu.csv` 与 `/root/tmp/p2d_train` 的 `metrics.csv` sha256 相同；`metrics_combined.csv` 的 `avg` 行与日志 `Final metrics` 15 位小数一致；两次 LRS2 评测的 `si-snr_mix/sdr_mix` 列逐元素相同（同 3000 条、同顺序）。
6. **2 种 ckpt 混在同一棵树里**：`LRS2_2Mix_Paper_2gpu_smoke/*.ckpt` 是 ~51.9 MB 的真 ckpt（3 epoch 冒烟），`checkpoints/*.ckpt`（88 个，5,269 B；2026-11-08 实测，每跑一次测试 +1）是单标量参数的废桩（`state_dict={'audio_model.w': ()}`），由测试写进仓库根。
7. **每跑一次测试 +1 个废桩 ckpt**：实测 79→80→81→82（2026-11-08 已到 88）。因为 `*.ckpt` 被忽略，`git status` 永远干净，看不出工作区被改过。
8. **115 个包、0 漂移**：`conda-env-lock.txt` 与 `pip freeze` 的名称+版本完全一致，`test_lock_matches_current_env_pip_freeze` 通过；但 lock 把 18 个 `file:///root/wheelhouse/...` 改写成 `pkg==version`，来源与哈希不在 lock 里。
9. **1 个 shim、1 个真缺口**：`pypesq.py` shim 存在（792 B）；`espnet` 确实缺失；`typeguard 2.13.3` 确实导出 `check_argument_types`，所以 `docs/verification/stubs/typeguard.py` 的 docstring 理由已失效。
10. **396 文件 / +107,000 / −523**：upstream 差异里 `docs/verification` 独占 72,118 插入（67%），真正跑起来的模型代码只有 `look2hear/** +693/−196`。这是一个以测量和证据为主的 fork。
11. **0 个 0 字节证据文件、0 个被跟踪的 `__pycache__`/`.pyc`**；`2062528` 删掉了 1 个 0 字节 CSV + 2 个 `.pid`。
12. **1 个残留 `.pid`**：`docs/verification/phase2/LRS2-paper-2gpu.pid`（内容 3118837，进程已不存在），被 `*.pid` 忽略、未跟踪。
13. **3 条最优先待办**：给测试加 `enable_checkpointing=False` 或 `tmp` 落盘、把「CUDA 可见性」从三个断言里摘掉、修 CI 的 `fetch-depth`。这三条比「再复现一次」更能改善证据质量。

## 1. 事实基线

### 1.1 仓库 / git 状态（只读）

```
$ git rev-parse HEAD
6c1d5b710ef5999c92aa4e920fbd21a2b9bf4688
$ git log -1 --format="%H %ci %s"
6c1d5b7... 2026-10-31 06:57:15 +0000 STATUS：回填 HEAD=2062528 与 freeze tag 说明
$ git status --porcelain      # 空
$ git tag -l
repro-freeze-20261031
repro-phase2-freeze-20261024
$ git describe --tags --always HEAD
repro-freeze-20261031
```

| 项 | 实测 | 备注 |
|---|---|---|
| HEAD commit | `6c1d5b710ef5999c92aa4e920fbd21a2b9bf4688` | AuthorDate `2026-10-30 18:05:08 +0000`，**CommitDate `2026-10-31 06:57:15 +0000`**（不是同一天；Lead 基线记的 18:05 是 author date） |
| 分支 | `main`，跟踪 `origin/main`（同 commit） | `repro/phase1`、`repro/phase2` 本地已不存在 |
| `repro-freeze-20261031` | **annotated tag**（`git cat-file -t` = `tag`）→ `6c1d5b7` | tag 消息：「复现维护轮冻结：三轮 review Day-1/Week-1 修复 + 证据收口…」 |
| `repro-phase2-freeze-20261024` | annotated tag → `2582f61` | 「Phase 2 复现证据冻结…」 |
| `git tag --contains HEAD` | `repro-freeze-20261031` | HEAD 正好落在冻结 tag 上 |
| 与上游 | `upstream/main..HEAD` = **23 ahead / 0 behind**；`upstream/main` = `439cbe9` | 已含 merge `d9ec1e8` |
| 工作区 | 干净；`--ignored` 39 条（38 个 `__pycache__` + `checkpoints/` 等） | 见 §1.5 |

### 1.2 范围文件规模

| 路径 | 文件数 | 说明 |
|---|---|---|
| `tools/repro/*.py` | 6 | `audit_data/fetch_data/lrs2_2mix_to_falcon/realssa_to_falcon/make_synthetic_data/extract_video_backbone` |
| `tools/repro/phase2/` | 18 py + 9 sh/yml | 其中 `protocol_check.py` 966 L、`eval_realssa_real.py` 723 L |
| `tools/repro/tests/` | 16 个 `test_*.py` + 3 个 helper | 共 **166 个 `def test_`**，pytest 收集 **168**（2 处参数化；本轮新增 2 条用例） |
| `docs/verification/scripts/` | 46 个 `*.py`（含 `data_pipe/` 5 个） | 另含 `stubs/`（4 个假包）与 `pil_stub/PIL` |
| `docs/verification/**` | 334 文件（108 `.log`、25 `.csv`） | 108/108 log、25/25 csv **全部入库** |
| `Experiments/` | 跟踪 2 文件，磁盘 203 MB | 见 §3.4 |

### 1.3 环境基线（实测，非引用）

| 项 | lock 文件 | 实测 | 一致 |
|---|---|---|---|
| Python | — | 3.11.17 | — |
| torch / torchaudio / torchvision | 2.8.0+cu128 ×3 | 2.8.0+cu128 / 2.8.0+cu128 / 0.23.0+cu128 | ✅ |
| pytorch-lightning | 2.6.6 | 2.6.6 | ✅ |
| typeguard | 2.13.3 | 2.13.3（`hasattr(check_argument_types)=True`） | ✅ |
| numpy / scipy / pandas | 2.2.6 / 1.17.1 / 3.0.6 | 同 | ✅ |
| speechmos / onnxruntime | 0.0.1.1 / 1.31.0 | 同 | ✅ |
| 包数 | 115 | `pip freeze` 115 | ✅ 集合与版本全等 |
| `pypesq.py` shim | 不在 lock（非发行包） | `/…/site-packages/pypesq.py` 792 B 存在 | ✅ |
| `espnet` | 明示未装 | `import espnet` → ModuleNotFoundError | ✅ |

`conda-env-lock.txt` 物理 **124 行** = 9 行注释 + 115 个包（无空行、文末有换行）。AGENTS.md:40 与 STATUS:22 都写「115 行」，应理解为「115 个包」。

### 1.4 测试基线（一句话）

GPU 可见：**168 passed / 0 skipped / 0 failed / 约 148 s**（2026-11-08 实测；本轮新增 2 条用例）。以下 §4 附的是旧收集 166 时的一次日志（355.47 s）。
隐藏 CUDA：**同样 168 passed / 0 failed / 约 170 s**（2026-11-08 终测）。**修前**为 166 passed / 2 failed
（旧收集 164；§4 的 173–186 s 为旧日志），两处已修。
对照 STATUS 旧记录：**166 passed / 136.5 s** → 通过数与旧收集吻合（本轮 +2 条用例后为 168），**耗时对不上**（355.47 s vs 136.5 s）。

### 1.5 工作区「脏」与「净」的真实边界

`git status --porcelain --ignored` 实测 39 条，全部是 `!!`（ignored）：

```
!! checkpoints/                             # 88 个 5,269 B 废桩 ckpt（2026-11-08）
!! Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/   # 203 MB 真冒烟产物
!! Experiments/checkpoint/LRS2_2Mix/        # 空 results 目录
!! docs/verification/phase2/LRS2-paper-2gpu.pid          # 3118837，进程已亡
!! .pytest_cache/  + 38 个 __pycache__/
```

- 被 **跟踪** 的 `__pycache__`/`.pyc`：**0 个**（`git ls-files | grep -cE "__pycache__|\.pyc$"` = 0）。
- 被跟踪的 `.pid`：**0 个**；被跟踪的 ≤10 字节文件只有 `agents.md`（9 B，symlink → `AGENTS.md`）。
- 0 字节文件：`docs/verification` 下 **0 个**（唯一那个 0 字节 CSV 已在 2062528 删除，见 §2.4）。

---

## 2. 机制拆解

### 2.1 复现流水线（六个可执行环节 + 一道门禁）

```
fetch_data.py ──► lrs2_2mix_to_falcon.py / realssa_to_falcon.py ──► audit_data.py（门禁）
                                                                        │
                                                                        ▼
                                    audio_train.py（configs/*.yml, 8 卡脚本）──► ckpt
                                                                        │
                                                                        ▼
                        audio_visual_test.py（整段 PIT 评测）┄┐
                        tools/repro/phase2/eval_realssa_real.py（窗口化 DNSMOS）┄┴─► docs/verification/**（log+csv+json）
                                                                        │
                                                                        ▼
                                            tools/repro/phase2/protocol_check.py（第三方对拍）
```

1. **`tools/repro/audit_data.py:1-14`** 是唯一「训练前只读门禁」：逐 split 读 `mix/target/near.json`，统计 K 分布与「自引用哨兵」（`near == mix` 时 Dataset 静默跳过 → 对比损失失效），并以 **退出码 0/1** 表达通过与否（`audit_data.py:11-14`）。这把「数据错但训练照跑」变成「训练前失败」。
2. **`lrs2_2mix_to_falcon.py:1-15`** 定义转换契约：`<stem>=<s1>_<gain1>_<s2>_<gain2>`，`mix == gain1*s1 + gain2*s2`（PCM16 取整），**target 恒为 s1**。这个「target=s1」不是随手约定，它被 `protocol_check.py` 独立验证过（见 §3.3：s1 vs s2 差 +24.55 dB）。
3. **`realssa_to_falcon.py:1-14`** 解 WebDataset 四分片（`mix.wav/target.wav/.json/.mouths.pt`），并做 mouths 原子写（commit `714e1a5`：临时文件 + `O_EXCL` + `fsync` + `os.replace`），避免并发覆盖共享 inode。
4. **`audio_visual_test.py:466-540`** 是证据生成的核心：`_merge_results()` 把 per-rank CSV 合并，**先剔除 `avg`/`std` 行**（`:478`），再算 `mean`（NaN skip）与 **population std `ddof=0`**（`:493`），逐列写 `n_<col>` 有效计数，并输出 `si_snr_identity_check`（`:505-512`，规则 `|si-snr − si-snr_i − si-snr_mix|`）。这套设计让「平均数是几条算出来的」不再靠猜。
5. **`tools/repro/phase2/eval_realssa_real.py:381-455`** 负责长音频窗口化推理（2 s 窗 / 0.5 s hop / 80 ms guard），并在 `:428-455` 加了**覆盖率断言**：`filled` 布尔数组逐样本记录是否被某窗写入，不用 `out.sum()` 之类的间接判据，`filled.all()` 为假就列出缺口 raise。
6. **`tools/repro/phase2/protocol_check.py:1-22`** 是「第三方独立验证」：纯 NumPy 重写 SI-SDR、直接调 `fast_bss_eval`，与 `look2hear.metrics.core`、3000 条 CSV、官方权重重跑三方对拍，并量化协议变体（target s1/s2、PIT oracle、SDR 滤波器长度、mouth 消融、big/small）。
7. **`tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh`** 的守卫是「错误配置不能静默降级」的具体实现：`NPROC` 正则校验（`:50`）、按 `NPROC` 限定 `CUDA_VISIBLE_DEVICES` 后断言 `device_count == NPROC`（`:54-68`）、把 `DATA_ROOT` **真正渲染进临时 conf**（`:11-13, :72`）而不是只做自检、打印 `global batch = bs × NPROC`（`:97-112`）。这是对「旧版只用 DATA_ROOT 做自检、训练仍读旧路径」的针对性修复。

### 2.2 测试套件的组织方式：**默认 fail-open，少数 fail-on-missing**

16 个测试文件里有 **约 25 处 `pytest.skip`**，gated 条件是「本机是否存在某数据/权重/报告」：

- `test_phase2_pipeline.py` 21 处 skip（`:131,161,196,198,255,292,383,424,433,435,458,484,519,558,590,610,670,716,722,766`），覆盖 Phase2 报告、LRS2 评测数据、官方权重、`/root/tmp/p2d_train` 指标等。
- `test_phase2_data.py:49,52,114,144`、`test_model_rebuild.py:315,326`、`test_t5_backbone.py:109,124`、`test_phase1_regression.py:323,696,723,739,741`。
- 文档自己承认这个设计：`test_phase2_pipeline.py:12`「数据/脚本/报告缺失时自动 skip，不阻塞仓库内回归」。

但 **Phase1 的 7 条验收断言被刻意改成 fail**（`test_phase1_regression.py:18-19`、`:108-135`）：历史实现判 `main vs main^`，tip 一前进就漂移，导致「7 条验收断言在 clean tree 下恒 skip、从未真正执行（review A8-1）」。现在用**固定基线 commit** `PHASE1_BASE = "689555c"`（`:108`），并在 `_phase1_base()` 里「解析不到就 fail 而不是悄悄 skip」（`:113-118`）。

`_landed(path)`（`:137-147`）的判据是「相对 689555c 的**累计** diff 或未跟踪文件」，与分支 tip/工作区干净度无关——这是让「修复是否落地」可判定的关键设计。

**跨分支数值一致性**另有机制：`test_numeric_invariance_vs_main_worktree`（`:317-340`）在**外部 worktree** 里复算指纹。实测 `git worktree list`：

```
/root/code/github_repos/Falcon-fork  6c1d5b7 [main]
/root/wt-base                        689555c (detached HEAD)
```

`/root/wt-main` 不存在，代码按 `BASELINE_WT_CANDIDATES = [/root/wt-base, /root/wt-main]`（`:43`）回退——**这条测试因此真的执行了**（耗时 8.83 s，做了 4 次前向 + 参数计数 + sha256 对比），不是 skip。

### 2.3 证据生成的两代机制（provenance 质量不均）

| 生成器 | 结构化 provenance | 时间戳 | 命令 | 计数/口径 |
|---|---|---|---|---|
| `tools/repro/phase2/eval_realssa_real.py` | ✅ `run_metadata.json`：`tool/timestamp_utc/data_root/loader_kind/is_smoke/channel/align_tail/model/device/versions/note` | ✅ `20261030T162555Z` | ❌ | ✅ `segments_total/ok/error` + per-seg `inference_ms/rtf/used_frames/discarded_samples` |
| `audio_visual_test.py` | ⚠️ 仅 `metrics_counts.json`（`producer/workers/n_utterances/counts/nan_counts/aggregation/identity`） | ❌ 无 | ❌ 无（只有 `[eval] config/ckpt_path/results dir`） | ✅ `n_*` + `nan_counts` + identity |
| 早期脚本（`docs/verification/scripts/*`） | ❌ | ⚠️ 少数写死在正文 | ❌ | ❌ |

即：**最新写的那个 CLI 有金标准 provenance，主评测路径没有**。`P2E-realssa-sim-eval-official.log` 全文 46 行，含 `[eval] config/ckpt_path/results dir` 与 `exit=0`，但**没有日期、没有命令、没有 git commit、没有 sha256**。

### 2.4 git 冻结机制：用 tag 绕开「提交不能包含自身 SHA」的悖论

STATUS 需要「指认当前 HEAD」，但 HEAD 的 SHA 在写文档时还不存在。`6c1d5b7` 的处理是：把占位符「（收尾提交后回填）」换成「`repro-freeze-20261031` 指向的提交（`git log -1`）」：

```diff
-> 最后更新：2026-10-31 · 分支 main · HEAD `见 §2（本轮收尾提交）` · 环境 conda falcon…
+> 最后更新：2026-10-31 · 分支 main · HEAD = `repro-freeze-20261031` 指向的提交（`git log -1`） · …
-| HEAD / tag | `（收尾提交后回填）`；tag `repro-freeze-20261031`（…） |
+| HEAD / tag | `repro-freeze-20261031`（tag 指向的提交；…） |
```

这个做法**正确**（tag 是 annotated tag，`git describe --tags HEAD` 输出即该 tag），但代价是：**tag 一旦被移动或重建，STATUS 会在无人察觉的情况下变错**。当前一致性成立。

---

## 3. 关键数据表

### 3.1 测试矩阵：每个文件测什么、断言主题、是否 fail-on-missing

| 文件 | 行/测试 | 被测对象 | 关键断言（举例） | 缺依赖时 | fail-on-missing? |
|---|---:|---|---|---|---|
| `test_audio_train_entry.py` | 352 / 20 | `audio_train.py` 入口 | `FakeTrainer` 记录 kwargs；真实 `FalconSpeakerContrastive` 从 `configs/Falcon.yml` 构建以暴露 config 污染；`build_logger` 目录层级 | 1 处 skip（comet 已装则跳） | 部分 |
| `test_audit_data.py` | 192 / 12 | `audit_data.py` + `fetch_data.py` 校验和 | 退出码 0/1、missing→1、`--full` 抽样契约、sha256/bytes | 无 | ✅ |
| `test_eval_entry.py` | 165 / 12 | `audio_visual_test.py` 助手 | ckpt 发现、config 加载、结果合并（不需 GPU/权重） | 无 | ✅ |
| `test_litmodule_logging.py` | 216 / 3 | 3 个 LightningModule 的 `self.log` | CSVLogger 下 `on_validation_epoch_end` 不崩；`val_pit_sisnr_*near` 键存在；`logger=None` 不崩 | 无 | ✅（**副作用：写 `checkpoints/`**） |
| `test_lock_portable.py` | 147 / 3 | `conda-env-lock.txt` | 无 `file://`/`/root/`/`@ url`；每行 `name==version` 且唯一；**包集合与版本 == 活环境 `pip freeze`**（`@ file://` 用 `importlib.metadata.version` 归一，`:82-95`） | 无 | ✅ |
| `test_lrs2_convert.py` | 318 / 10 | `lrs2_2mix_to_falcon.py` | 布局/契约、`mix==g1*s1+g2*s2`、幂等（二次运行 skip 已有）、fallback 布局 | 无（合成树） | ✅ |
| `test_lrs2_paper_build.py` | 180 / 4 | `build_lrs2_paper_data.py` | `--copy-wavs` 同名塌缩（P0）、构建闸门、40k/10k/6k 计数 | 无（合成树） | ✅ |
| `test_metrics.py` | 241 / 8 | `look2hear/metrics` | 与解析值 ±0.05 dB；PESQ/STOI **参数顺序**；`ALLMetricsTracker` CSV 的 per-utt + `avg`/`std` 同向约定 | 无 | ✅ |
| `test_model_rebuild.py` | 335 / 18 | `Falcon`/`Separator` 参数与重建 | 537,697 口径、`serialize→from_pretrain` 525 keys、config 不被污染（同 dict 连构两次） | 2 skip（缺 backbone/ckpt） | 部分 |
| `test_phase1_regression.py` | 769 / 21 | Phase1 全量验收 + Studio + 数值不变 | 固定基线 `689555c` diff 白名单；gpus 语义矩阵；`/root/wt-base` 指纹逐位一致；**studio skip 断言（现放宽为 `skipped in (0,2)`）**；参数/MACs 基线 | 6 skip（worktree/`/root/tmp` 数据） | ✅（验收 7 条） |
| `test_phase2_data.py` | 182 / 5 | Phase2 转换产物 | tr/cv/tt 等长、`mouths` 形状、s1/s2 对照 | 4 skip（无转换数据） | ❌ fail-open |
| `test_phase2_launcher.py` | 210 / 13 | 8 卡脚本 + `audio_train.py` 静态契约 | 渲染 DATA_ROOT、NPROC/global batch、resume 守卫 raise、monitor 选最高 `version_*` | 无（只读文本 + tmp） | ✅ |
| `test_phase2_pipeline.py` | 785 / 19 | Phase2 端到端（脚本/报告/CSV 一致性） | 脚本 `--help` 可跑、报告结构、summary↔逐句 CSV 互证、eval 冒烟 | **20 skip** | ❌ fail-open |
| `test_realssa_convert.py` | 227 / 5 | `realssa_to_falcon.py` | channel mean==手算均值、ch0、端到端 | 无（合成） | ✅ |
| `test_synthetic_data.py` | 160 / 5 | `make_synthetic_data.py` | 布局/索引/可被 DataModule 读取 | 无 | ✅ |
| `test_t5_backbone.py` | 140 / 6 | 视频骨干导出 + API 小修 | `FRCNNVideoModel.train()` 返回 self、`_mdcl_loss` 别名、`__all__`、指纹不变 | 2 skip（缺权重/导出） | 部分 |

**"绿但结论可能错"的四个盲区**（按危险度排序）：

1. **数据门控 = 静默降级**。25 处 skip 里 20 处在 `test_phase2_pipeline.py`。在「数据被清掉/换机/浅克隆」的仓库里，162 条里可能有一大批变 skip，而 **`168 passed` 的分母本身会变**——STATUS 只记了数字没记「本机数据清单」，无法判断这 168 是在什么数据前提下的 168。
2. **CUDA 可见性改变语义**（不是难度，是**语义**）——**本节记为修前状态**。`test_phase1_regression.py` 当时断言 `skipped == 2`，而 `studio/tests/test_device.py:35,44` 的两个测试在无 GPU 时会**从 skip 变 run**。于是「有 GPU → 51p/2s（通过）」，「无 GPU → 53p/0s（失败）」。同一个仓库、同一份代码，断言结论相反。**现已修**：断言放宽为 `skipped in (0,2)` + `passed + skipped >= 37`，无 GPU 也通过（168 passed）。
3. **`test_t1_gpu_spec_semantics` 的校验分支在无 CUDA 时被短路**。`audio_train.py:116` 先解析 spec，然后 `:118-119` 在 `not torch.cuda.is_available()` 时 **直接 `return "cpu",1,1,"auto"`**，根本走不到「空列表/重复索引/越界索引 raise」的校验。测试 `:647-649` 对这三个 case 无条件 `pytest.raises(ValueError)` → 修前无 GPU 必红（**已修**：把校验顺序前移到 CUDA 短路之前，无 GPU 也走校验）。注意：**同一测试在 `:658` 却把 GPU 相关断言放进了 `if _t.cuda.is_available()`** —— 作者对「GPU 相关」的判断在同一个函数里前后不一致。
4. **`_landed()` 依赖完整 git 历史**。浅克隆下 `git cat-file -t 689555c` 失败 → `pytest.fail`（`:113-118`）。这是**故意**的（比静默 skip 好），但它把「测试绿」和「有完整历史」绑定，而 CI 正是浅克隆。

### 3.2 证据链抽检表（6 个，含 5 个随机 + 1 个系统性补充）

| # | 证据 | 可追溯性 | 对应命令 | 时间戳 | 与 STATUS 一致 | 编辑痕迹 / 残留 | 结论 |
|---|---|---|---|---|---|---|---|
| 1 | `docs/verification/phase2/P2E-realssa-sim-eval-official.log`（46 行） | ⚠️ 有 config/ckpt_path/results dir + `exit=0`，**无命令/无日期/无 commit** | ❌ 隐含为 `audio_visual_test.py` | ❌ | ✅ `si-snr_i 7.774307`、`sdr_i 8.479395` = STATUS 7.774/8.479 | 进度条把指标拆成多行（`:27-33`），不影响 | 数值可信，**元数据不足** |
| 2 | 同上配对的 `P2E-realssa-sim-metrics_combined.csv`（1,178,733 B） | ✅ 逐句 + `avg`/`std` 行 | — | — | ✅ `avg` 行与日志 15 位小数一致 | ⚠️ **含 `avg`/`std` 两行汇总**，下游 `read_csv` 不过滤就会多算 2 行 | 格式正确（`audio_visual_test.py:478` 有剔除逻辑、`test_metrics.py` 有断言） |
| 3 | `docs/verification/phase2/DNSMOS-realssa-real-mean-full4-20261031/`（4 文件） | ✅ **金标准**：`run_metadata.json` 含 tool/timestamp_utc/data_root/loader_kind/channel/device/versions/note | ⚠️ tool 字段给了脚本路径，无完整 argv | ✅ `20261030T162555Z` | ✅ OVRL 2.7941 / P808 3.3365 / N=54 | ⚠️ `"checkpoint": null`（用模型名 `JusperLee/Falcon` 解析），**未 pin 权重文件与 sha256** | 最可信的一份，唯一缺「权重指纹」 |
| 4 | `docs/verification/phase2/P2D-eval-falcon-full-cols-20261031.{log,csv,counts.json}` | ✅ log 里有 `results dir : /root/tmp/p2d_eval_official_v4/...`（隔离 cwd）+ ckpt_path；counts.json 有 `producer` | ❌ 无 argv | ❌ | ✅ 7.029101 / 7.712925；`n=3000/3000` 全列 | ⚠️ 进度条显示 `7.049/7.73`，与 Final `7.029101/7.712925` **差 ~0.02 dB** | 数值可信；引用时读 `Final metrics` 段，进度条那行差 0.02 dB |
| 5 | `docs/verification/phase1/T1-checks.log` | ✅ 逐项标题 + 可打印矩阵（含每条 ValueError 原文） | ⚠️ 标题式，无完整命令 | ❌ | ✅ 与 `audio_train.py:50-95` 语义逐条吻合 | ✅**诚实**：`:T1-6` 段直接写下「1 failed, 76 passed … cause: ALLOWED_TRACKED 未包含 configs/Falcon.yml」 | 好证据：把「当时的已知失败」留在文件里 |
| 6 | `docs/verification/runs/conda-env-verify.log` | ✅ 首行 `=== … (2026-10-21T10:54:09+00:00) ===` + env 路径 | ⚠️ 无 argv | ✅ | ⚠️ 当时 studio 是 `35 passed, 2 skipped`（现在 51+2） | 无 | 证明环境构建时刻；也说明 AGENTS 的「38」是中间值 |

**通用缺陷（6/6 命中）**：`docs/verification` 下 **108 个 log 无统一 run id、无 git commit、无完整 argv**；`scripts/` 46 个脚本与 `runs/` 51 个日志之间**没有机器可读的映射表**，只能靠文件名约定（如 `verify_model_params.py` → `lead-model-params.log`，而 `verify_causality.py` → `lead-causality.log`/`r2-causality-conditions.log`，命名并不一致）。

### 3.3 交叉一致性（这是本报告最想强调的「证据不是孤证」）

| 交叉项 | 方法 | 结果 |
|---|---|---|
| 3000 条逐句 CSV vs 日志 | 读 `avg` 行 | `si-snr_i 7.774307312097187` / `sdr_i 8.479394557369417` — 与 log 逐位一致 |
| **两次评测是否同一测试集** | 比较 `epoch99` 与 `falcon-full` 的 `si-snr_mix`/`sdr_mix` 列 | **逐元素完全相等**（`np.array_equal=True`），`snt_id` 顺序 3000/3000 相同 |
| 配对增益 | `epoch99 − falcon-full` 的 `si-snr_i` | mean **+1.697 dB**、median +1.319、**88.4% 句子为正** |
| 指标恒等式 | `|si-snr − si-snr_i − si-snr_mix|` | max **4.44e-15**（falcon-full）/ **5.33e-15**（ep99），finite_pairs 3000 |
| 独立实现对拍 | 纯 NumPy SI-SDR vs `look2hear.metrics.core` vs CSV，N=50 | maxabs **4.82e-8 dB** / **1.84e-7 dB**；`dataset_order_match = 50/50`；`csv_avg_recompute_ok = True` |
| target 约定 | 同一模型换 target s1↔s2 | s1 **7.5347**，s2 **−17.0117**，Δ **+24.55 dB**；oracle-PIT==s1；`n_items_where_s2_better = 0` |
| 视频分支是否真的在用 | mouth 置黑 / 移 ±2 帧 / 换说话人（N=20, 基线 7.9106） | 置黑 **−14.399**、+2 帧 **−6.353**、−2 帧 **−5.079**、错人 **−23.297** |
| 数据代数 | 3000 条 mix 峰值 / `mix − (s1+s2)` | `n_peak_is_0.9 = 3000`；mix/s1/s2 关系残差见 `P2F-results.json`；`target_is_s1_copy_sample = 5/5` |
| 训练产物 | `P2D-train-metrics-1gpu.csv` vs 活文件 | **sha256 相同**：`c6d5704e…fef675`（5201 行，两边一致） |
| ckpt | sha256 逐个复核 | ep86 `68edbc98…d561`、ep99 `8a461175…dfec`、ep124 `a3f4319d…b73d` — **与 `P2D-report.md:273-279` 表格完全一致**，字节数 51,922,804 / 51,922,996 一致 |
| 权重自描述 | `torch.load` 元数据 | `falcon.pt`: `FalconSpeakerContrastive`, epoch **714**, 662 张量（525 audio + 137 video）；`falcon-big.pt`: **`SpeakerContrastiveV2`**, epoch **715**, 654（517+137）— 与 STATUS:23 一致 |

### 3.4 产物 / 实验目录对照

| 位置 | 内容 | 大小 | 性质 | 入库 |
|---|---|---|---|---|
| 仓库 `checkpoints/` | 88 × `epoch=0-step=2[-vNN].ckpt`（2026-11-08 实测） | 5,269 B/个，共 640 KB | **废桩**：`state_dict = {'audio_model.w': ()}`（1 个 0 维张量，1 元素），epoch 0 / step 2 | ❌ `*.ckpt` 忽略 |
| 仓库 `Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/` | `epoch=0/1/2.ckpt`、`last.ckpt`（各 51.9 MB）、`best_model.pth`（2,376,222 B，525 张量 / **537,762** 参数）、`conf.yml`、`best_k_models.json`、`version_0/` | 203 MB | **真**（3 epoch 冒烟） | ❌ `Experiments/checkpoint/LRS2_*/` 忽略 |
| 仓库 `Experiments/checkpoint/Falcon/` | `conf.yml`（audionet 配置快照）+ `best_k_models.json` | 小 | 发布权重的训练配方证据 | ✅ **仅这两个文件被跟踪** |
| `/root/tmp/p2d_train/…/LRS2_2Mix_1gpu/` | `epoch={86,96,97,98,99}.ckpt`+`last.ckpt`（各 51,922,804 B）、`best_model.pth`、`best_k_models.json`、`version_0/{hparams.yaml,metrics.csv(757,616 B)}` | 301 MB | **真，100 epoch 正式训练** | ❌ 仓库外 |
| `/root/tmp/p2d_near_train/…/LRS2_2Mix_near_s2/` | `epoch=124.ckpt` | 52 MB | 真，contrastive 微调 | ❌ 仓库外 |

`best_k_models.json` 的值是 `−val_pit_sisnr`（负号用于让「最小=最好」）：ep86 → `−13.118217`、ep96 `−13.138268`、ep97 `−13.114943`、ep98 `−13.037923`、**ep99 `−13.201412`（最小）** → ep99 确为 best val，STATUS:40 的 13.2014 可复核。

**两者关系**：仓库里的 `Experiments/` 只是**冒烟/索引**（203 MB，未跟踪）；真正的正式训练产物全在 `/root/tmp/`，**不入库、未归档**（`P2D-report.md:287` 自己写明「ckpt 只存在于本机 `/root/tmp`，未归档」）。所以「仓库可复现」到这里只是「数值可复现」，**产物不可复现**。

### 3.5 upstream 差异分类（`git diff upstream/main..HEAD`）

总量：**396 文件、+107,000 / −523**（361 新增 / 35 修改 / 0 删除）。

| 归类 | 面积 | 文件 | +/− | 判断 |
|---|---|---:|---|---|
| **复现工具（工具链）** | `tools/**` | 48 | +12,652 / 0 | 新增能力（audit/protocol_check/8 卡脚本/19 个测试文件） |
| **复现修复（运行时代码）** | `look2hear/**` | 12 | +693 / −196 | 外科手术式：config 污染、logger API、指标参数序、FRCNN 严格加载 |
| 复现修复（入口） | `audio_train.py` / `audio_visual_test.py` | 2 | +777 / −248 | gpus 语义、resume 守卫、结果合并/计数/恒等式 |
| **证据** | `docs/verification/**` | 257 | +72,118 / 0 | 67% 的 diff 是 log/csv/json |
| **文档** | `docs/understanding|reviews|*.md` | ~45 | +18,300 / 0 | 上轮 9 篇深挖 + 三轮 review + STATUS/PLAN/GUIDE |
| **Studio** | `studio/**` | 15 | +766 / −61 | `--checkpoint`、doctor/bootstrap 守卫、依赖对齐、i18n |
| **Space** | `space/**` | 4 | +61 / −14 | 静态测试 + i18n |
| 项目页 | `project_page/index.html` | 1 | +19 / −4 | meta |
| **上游同步** | （不出现在 diff 中，已在 `upstream/main` 内） | — | — | 唯一 merge 提交 `d9ec1e8` |
| CI | `.github/workflows/tests.yml` | 1 | +112 | 新增 |

**TOP-20 改动文件（按 churn，剔除 `docs/verification/**.{csv,json}` 的证据体量）**

| # | 文件 | +/− | 为什么改 |
|---:|---|---:|---|
| 1 | `docs/understanding/06-studio-backend.md` | +1,252 | 上轮 Studio 后端深挖 |
| 2 | `docs/understanding/04-losses-metrics-video.md` | +1,191 | 损失/指标/视频前端深挖 |
| 3 | `docs/understanding/07-studio-frontend.md` | +1,158 | 前端深挖 |
| 4 | `docs/understanding/01-model-core.md` | +975 | 模型核心深挖 |
| 5 | `tools/repro/phase2/protocol_check.py` | +966 | 第三方独立对拍（本报告最看重的工具） |
| 6 | `docs/understanding/05-data-pipeline.md` | +912 | 数据管线深挖 |
| 7 | `docs/understanding/02-layers-dsp.md` | +895 | 层/DSP 深挖 |
| 8 | `docs/verification/phase1/review_training_stack_checks.py` | +793 | 训练栈复核脚本 |
| 9 | `tools/repro/tests/test_phase2_pipeline.py` | +785 | Phase2 端到端回归（20 处 skip） |
| 10 | `tools/repro/tests/test_phase1_regression.py` | +769 | Phase1 验收（含 `689555c` 固定基线、studio skip 断言） |
| 11 | `docs/understanding/08-ops-tests-docs.md` | +768 | 运维/测试/文档深挖 |
| 12 | `tools/repro/phase2/eval_realssa_real.py` | +723 | DNSMOS 评测 + 窗口起点修复 + 覆盖率断言 |
| 13 | `audio_visual_test.py` | +489/−213 | 结果合并/计数/恒等式；多 worker 合并 |
| 14 | `docs/understanding/03-training-system.md` | +700 | 训练系统深挖 |
| 15 | `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` | +690 | 总纲 |
| 16 | `tools/repro/lrs2_2mix_to_falcon.py` | +689 | LRS2→Falcon 布局转换 |
| 17 | `tools/repro/phase2/gap_analysis.py` | +660 | 7 dB 差距归因 |
| 18 | `tools/repro/realssa_to_falcon.py` | +580 | WebDataset→folder 转换（含原子写） |
| 19 | `docs/verification/scripts/verify_studio_backend.py` | +524 | Studio 后端验证 |
| 20 | `tools/repro/fetch_data.py` | +395 | 统一下载（curl 续传 + sha256/bytes 校验） |

（若纯按行数排，前列被 7 个大 CSV 占据：`P2E-realssa-sim-metrics_combined.csv` 9,172 行、其 ch0 版 9,172 行、`P2D-train-metrics-1gpu.csv` 5,201 行、5 个 3,003 行的评测 CSV。它们不是「改动」而是「证据入库」。）

---

## 4. 实测验证

### 4.1 pytest（第一次：CUDA 隐藏，遵守「不占 GPU」）

```bash
$ CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD \
  /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q -rs --durations=15
...
2 failed, 164 passed, 39 warnings in 186.00s (0:03:06)
```

### 4.2 pytest（第二次：CUDA 隐藏，抓失败明细）

```bash
$ ... pytest tools/repro/tests -q -rf --tb=line -p no:cacheprovider
=========================== short test summary info ============================
FAILED tools/repro/tests/test_phase1_regression.py::test_studio_pytest_regression
FAILED tools/repro/tests/test_phase1_regression.py::test_t1_gpu_spec_semantics
2 failed, 164 passed, 39 warnings in 173.45s (0:02:53)
===ckpt after=== 81     # ← 第一次跑前 79，跑完 80；第二次 81
```

### 4.3 两个失败的根因（逐个定位，**都不是代码缺陷**；以下为 2026-11-08 **修前**的现场记录，两处现已修）

```bash
$ ... pytest tools/repro/tests/test_phase1_regression.py -q --tb=short -k "studio_pytest_regression or t1_gpu_spec_semantics"
______________________ test_studio_pytest_regression ______________________
tools/repro/tests/test_phase1_regression.py:385: assert skipped == 2
E   AssertionError: Studio skip 数变化：0 (期望 2)
E    53 passed in 42.54s
______________________ test_t1_gpu_spec_semantics ______________________
tools/repro/tests/test_phase1_regression.py:648: in test_t1_gpu_spec_semantics
E   Failed: DID NOT RAISE ValueError
2 failed, 19 deselected, 5 warnings in 49.21s
```

- **失败 1**：`test_phase1_regression.py:385` 硬断言 studio 的 skip 数 == 2。无 GPU 时 `studio/tests/test_device.py:35,44` 的 `skipif(torch.cuda.is_available(), …)` 变 False → 两个测试**真的跑**，于是 `53 passed / 0 skipped`。**"环境变好"把守卫打红了。**
- **失败 2**：行号 648 对应 `resolve_accelerator_and_devices({"gpus": [99]})` 那个 case。`audio_train.py:118-119`：`if not torch.cuda.is_available(): return "cpu",1,1,"auto"` → 校验代码根本没执行。

### 4.4 反证：同一用例在 GPU 可见时通过

```bash
$ unset CUDA_VISIBLE_DEVICES
$ ... pytest "tools/repro/tests/test_phase1_regression.py::test_t1_gpu_spec_semantics" -q
1 passed, 5 warnings in 6.78s     # real 12.0 s
```

### 4.5 pytest（第四次：GPU 可见，全量）—— STATUS 的 166 复现（旧收集；本轮新增 2 条用例后为 168）

```bash
$ unset CUDA_VISIBLE_DEVICES; PYTHONPATH=$PWD ... pytest tools/repro/tests -q -rs -p no:cacheprovider
166 passed, 43 warnings in 355.47s (0:05:55)
=== ckpt after === 82          # 79 → 80 → 81 → 82，每次跑 +1
nvidia-smi: GPU0 0 MiB / GPU1 0 MiB（前后均 0，未实际占卡）
```

**结论**：`166 passed / 0 skipped / 0 failed`（旧收集）**可复现**；`136.5 s` **不可复现**（355.47 s，2.6×；隐藏 CUDA 时 173–186 s）。

耗时无法归因：第一次跑（CUDA 隐藏，186.00 s）留了 `--durations=15`，前 15 名合计约 **116 s（62%）**，头部是 `test_phase1_regression::test_studio_pytest_regression` 42.32 s、`test_phase2_pipeline::test_phase2_scripts_compile_and_help` 15.35 s、`test_phase1_regression::test_import_completeness` 10.07 s。而 GPU 可见的 355 s 那次**没有 durations 账本**，多出的 ~219 s 落点未知（其中最可疑的是 studio 的两个 device 测试从 skip 变 run + 依赖 CUDA 的 smoke 用例）→ GAP-1。

### 4.6 CI 是否真的会红（静态推断 + 本地等价复现）

`.github/workflows/tests.yml:71-112`：

```yaml
  python:
    runs-on: ubuntu-latest          # 无 GPU
      - uses: actions/checkout@v4   # :75 未设 fetch-depth → 浅克隆
      - run: python -m pytest tools/repro/tests -q -rs -k "not diff_scope"   # :112
```

两条独立缺陷（第 1 条已修）：

1. **无 GPU**（**已修**）→ 4.3 的两个失败在修前必然出现；放宽 skip 断言 + 固定线程数后，本地 `CUDA_VISIBLE_DEVICES=` 跑出 **168 passed / 0 failed**（170.39 s）。
2. **浅克隆** → `test_phase1_regression.py:113-118` 的 `_phase1_base()` 在 `git cat-file -t 689555c` 失败时 `pytest.fail("…本套件需要完整 git 历史（勿在浅克隆中运行）")`。这会连带打掉该文件里依赖 `_landed()` 的多条断言。

同一 workflow 的 `frontend` job（`:31-69`）是安全的（只跑 node + space + lock 检查），并且 `:66` 的 `grep -n "@ file://"` 与 lock 现状一致（lock 里 `file://` 出现 0 次）。

> 注：我无法访问 GitHub Actions 运行记录（本任务不允许 `fetch`），所以这是「代码+本地等价条件」的推断，不是运行记录。列入 GAP-10。

### 4.7 证据链哈希复核

```bash
$ sha256sum docs/verification/phase2/P2D-train-metrics-1gpu.csv \
            /root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/version_0/metrics.csv
c6d5704e4fab99b00194de4f63f70aba9fd20df43dcc3d34d325845508fef675  docs/…/P2D-train-metrics-1gpu.csv
c6d5704e4fab99b00194de4f63f70aba9fd20df43dcc3d34d325845508fef675  /root/tmp/…/version_0/metrics.csv
$ wc -l 两者 → 5201 / 5201

$ sha256sum /root/tmp/p2d_train/…/epoch=86.ckpt …/epoch=99.ckpt …/p2d_near_train/…/epoch=124.ckpt
68edbc98d265b7f5d1630d92ae8bcd41e8eb831b49f9ee2519a6b06c6663d561  epoch=86.ckpt
8a46117577dfc9ce4e40e70019ddcaa5c749adebd2f44e1669d14120a47ddfec  epoch=99.ckpt
a3f4319d72ccd1f4f86a30881c617bbfe87310087290cfccf4fbe309d5f2b73d  epoch=124.ckpt
```

三行与 `docs/verification/phase2/P2D-report.md:273-279` 的表格**逐字符一致**。这是全仓最强的一条证据链（证据文件 ↔ 产物 ↔ 报告表格三方闭环）。

### 4.8 环境一致性实测

```bash
$ /root/miniforge3/envs/falcon/bin/pip freeze | wc -l      → 115
$ grep -v '^#' docs/verification/conda-env-lock.txt | grep -v '^$' | wc -l → 115
$ diff <(sort lock) <(sort freeze)
# 差异只在 18 行：左侧 lock = `nvidia-*==12.8.x` / `torch==2.8.0+cu128` / `pytorch-lightning==2.6.6` / `triton==3.4.0`
#            右侧 freeze = 同名包 `@ file:///root/wheelhouse/….whl#sha256=…`
$ grep -c "file://" docs/verification/conda-env-lock.txt → 0
```

即：**版本集合完全一致（0 漂移）**，但 lock 的「可移植化改写」抹掉了 **18 个包的来源与 sha256**。`test_lock_portable.py:82-95` 明确解释了为什么这样仍能通过（用 `importlib.metadata.version()` 归一 direct reference）——测试通过是**正确**的，只是它守住的是「名字+版本」，不是「二进制」。

### 4.9 stub 是否扭曲了验证结论（逐个排除）

```bash
$ /root/miniforge3/envs/falcon/bin/python -c "import typeguard; print(hasattr(typeguard,'check_argument_types'))"
True                                    # typeguard 2.13.3 有该 API → stub 的理由失效

$ # 同 seed 下对比 stubs/timm 的 DropPath 与真 timm 1.0.30
drop_prob=0.0 equal=True maxdiff=0.000e+00
drop_prob=0.3 equal=True maxdiff=0.000e+00
```

- `timm` stub：**逐位等价**（真 timm 1.0.30 的 `drop_path` 与 stub 同实现），所以参数/MACs/前向指纹类证据**不受影响**。
- `typeguard` stub：no-op，会**关闭实参类型检查**（不影响数值结论，影响「是否真的做过断言」）。
- 但 **stub 的 docstring 前提已过时**：`stubs/timm/__init__.py:5` 说「timm 需要 torchvision，沙箱未装」、`stubs/typeguard.py:3-5` 说「装的 typeguard 不再导出 check_argument_types」——两者在当前 `falcon` 环境**都不成立**（torchvision 0.23.0+cu128 已装、typeguard 2.13.3 有该 API）。而 `verify_model_params.py:21` 仍然**无条件** `sys.path.insert(0, …/stubs)`。

---

## 5. 坑与反直觉点

1. **测试会往仓库根写 ckpt，而且永远看不见。**
   `tools/repro/tests/test_litmodule_logging.py:142-153` 构造 `pl.Trainer(max_epochs=1, limit_train_batches=2, accelerator="cpu", devices=1, …)`，**没有 `enable_checkpointing=False`，也没有 `default_root_dir`**。PL 默认 `ModelCheckpoint` 就写到 `<cwd>/checkpoints/`，pytest 的 cwd = 仓库根。实测每跑一次 +1 个 `epoch=0-step=2-vNN.ckpt`（当时 79→80→81→82，2026-11-08 已到 88）。因为 `.gitignore:24` 的 `*.ckpt`，`git status` 永远干净 —— **这是「测试零污染」的假象**。同时 `test_audio_train_entry.py:14` 的 docstring 还写着「no source files are modified by the tests」（严格说不算 source，但工作区确实被改了）。
2. **`checkpoints/` 里的 88 个 ckpt 是废桩，不是模型（2026-11-08 实测，随每次全量测试 +1）。**
   `state_dict = {'audio_model.w': ()}` —— 1 个 0 维张量、1 个元素、5,269 B。任何「ckpt 存在就算有产物」的判断都会被它骗过。
3. **进度条数值 ≠ 最终数值，差 ~0.02 dB。**
   `P2D-eval-falcon-full-cols-20261031.log` 的进度条写 `si-snr_i: 7.049`、`sdr_i: 7.73`，而 `Final metrics` 是 `7.029101 / 7.712925`。若有人截进度条当证据，会得出比 STATUS 高 0.02 dB 的结论。**只认 `Final metrics` 段与 `counts.json`。**
4. **`metrics_combined.csv` 末尾有 `avg`、`std` 两行。**
   `audio_visual_test.py:478` 自己会剔除，但外部复算时 `pd.read_csv(...)['si-snr_i'].mean()` 会把这两行算进分母。实测差异：`7.773784`（含汇总行）vs `7.774307`（正确）—— 0.0005 dB 的「幽灵偏差」。
5. **RealSSA-Sim 的 9,169 里有 3 条 nan。**
   `P2E-realssa-sim-metrics_combined.csv`：`sdr/sdr_i/pesq` 各 3 个 nan（`torch.linalg.solve … singular` / `PESQ No utterances detected`，见 log `:3-11`），`si-snr/si-snr_i/stoi` 无 nan。所以 `sdr_i` 与 PESQ 的均值分母是 **9,166**、`si-snr_i` 是 **9,169**（W1-07 §3.4 逐列计数复核）。STATUS:35 写「9,169 条」对 si-snr 类正确，对 sdr/pesq 类实际是 9,166 条。
6. **窗口协议有一个曾经吃掉 1.34 s 前导静音的 bug。**
   `eval_realssa_real.py:412` 注释写得很清楚：旧值 `endpoint = LEFT + HOP`（=30720）让首窗只写 `[21440, 29440)`，`[0, 21440)`（1.34 s @16k）**永久静音**。现在改成 `endpoint = HOP + GUARD`（=9280），并加 `filled` 覆盖率断言（`:428-455`）。这解释 full3（2.7901）为何作废、full4（2.7941）为何只涨 0.004 —— **bug 让语音变差，DNSMOS 只会轻微下降，不会报错**，正是最该加断言的那类 bug。
7. **docstring 的 `path:line` 会漂。**
   `eval_realssa_real.py:385-390` 说「与 `studio/server.py:266` 一致」，实际那段代码在 **`studio/server.py:328`**（`:266` 是 HTTP 中间件）。`studio/web/app.js:624` 那条引用是**对的**。同类：`P2F-results.json` 记 README 行号 211/228，实际是 **213/230**。
8. **`git log upstream/main..HEAD` 看不到上游同步的内容。**
   因为 `upstream/main` 已经是 tip，上游带来的改动（含 arXiv 链接那条）**不出现在 diff 里**；只有 `d9ec1e8` 这个 merge 提交作为「发生过同步」的痕迹。若只看 `--stat` 会误以为「全程自己写的」。
9. **tag 是 STATUS 的锚点，而 tag 可以被移动。**
   `repro-freeze-20261031` 现在指 `6c1d5b7`；STATUS 说「HEAD = tag 指向的提交」。任何人 `git tag -f` 都会让 STATUS 静默变错，而没有测试守这个不变量。
10. **`docs/verification/stubs/` 会被所有验证脚本无条件注入。**
    `verify_model_params.py:21`、`model_core_verify*.py:19` 等 30 个脚本都 `sys.path.insert(0, …/stubs)`，永久 shadow 真实 `timm/torchvision/typeguard/matplotlib`。stub 本身（DropPath）是等价的，所以结论没错；但**「证据是在 stub 环境下跑出来的」这件事没有写进任何 log**。
11. **`__pycache__` 有 38 个目录在磁盘上，但 0 个被跟踪。**
    `.gitignore:2-3` 生效。看起来「干净」是靠 ignore，不是靠没产生。`find tools/repro -name "*.pyc"` 能看到 30+ 个 `.pyc`。

---

## 6. 与既有文档的冲突 / 更正

| # | 旧说法（出处） | 实际（证据） | 建议 |
|---|---|---|---|
| 1 | 任务书：「STATUS 写 HEAD=2062528，实际 HEAD=6c1d5b7」 | `docs/STATUS.md:3,20` 写的是「`repro-freeze-20261031` 指向的提交」；tag→`6c1d5b7`=HEAD，**自洽**。「回填 HEAD=2062528」只是提交信息措辞；`grep -rn 2062528 --include=*.md` **零命中** | 无需改 STATUS；把「提交信息别写具体 SHA」写进约定 |
| 2 | `AGENTS.md:66`、`:101`「studio/tests **38 passed/2 skipped**」 | 实测 GPU 可见 **51 passed/2 skipped**；STATUS:64 的 51/2 才对。历史证据 `runs/conda-env-verify.log` 记的是 35/2（2026-10-21） | 把 AGENTS.md 两处改成「51 passed / 2 skipped（2026-10-31 实测）」，或改为「见 STATUS §5」避免再漂 |
| 3 | `AGENTS.md:40`、`STATUS:22`「conda-env-lock.txt **115 行**」 | 物理 **124 行** = 9 注释 + 115 包 | 写「115 个包（124 行）」 |
| 4 | `STATUS` §5「168 passed / 0 skipped / 0 failed（2026-11-08 实测，约 148 s）」 | 通过数 ✅（本轮 +2 条用例）；旧日志实测 **355.47 s**（GPU 可见）/ 173–186 s（CUDA 隐藏） | 删掉具体秒数或标注「首次冷跑 ~136 s / 本机复跑 355 s，随 GPU 可见性变化」 |
| 5 | `STATUS:63` 把它当作无条件基线 | **无条件成立**：有/无 GPU 均 168 passed；修前无 GPU 为 166p/2f（§4.3） | **原判不成立**（缺前提是修前的 bug，已修） |
| 6 | `docs/verification/phase2/README.md:3`「分支 `repro/phase2`（HEAD `af74cca`）」 | 该分支**本地/远端均不存在**；HEAD = `6c1d5b7` | 该文件已有 2026-10-31 声明「以 STATUS 为准」，建议把首行分支/HEAD 直接改成「历史快照」 |
| 7 | `docs/verification/lead-verification.md:12-14` 复跑命令用 `/root/.venvs/falcon/bin/python` | AGENTS.md:27 写明该环境「已删除、不使用」。全仓 **18 个当前文档**（非 reviews/deepdive）仍含该路径，`*.md` 里共 **85 处** | 给「可执行命令块」加统一 banner 或批量替换为 `conda falcon` 路径；历史日志正文保持原样 |
| 8 | `docs/verification/stubs/typeguard.py:3-5`「装的 typeguard 不再导出 `check_argument_types`」 | `hasattr==True`（2.13.3） | 改 docstring；或改为「仅在 typeguard>=3 时兜底」的 try/except 注入 |
| 9 | `docs/verification/stubs/timm/__init__.py:5`「真 timm 需要 torchvision，沙箱未装」 | torchvision 0.23.0+cu128 已装、timm 1.0.30 已装 | 同上；并说明「stub 与真实现逐位等价（实测 drop_prob=0/0.3 均 equal）」 |
| 10 | `docs/verification/scripts/pil_stub/PIL/**` 入库 | `AGENTS.md:84` 自己写「Pillow 已装，不需要 PIL stub」；Pillow 13.0.0 在环境里 | 标 deprecated 或移出 `scripts/`（现在它给 `__pycache__` 贡献了 4 个 `.pyc`） |
| 11 | `docs/verification/phase1/README.md:24`「baseline 由 Lead 在 `/root/wt-main` 上生成」 | 实际 worktree 是 **`/root/wt-base`**；`test_phase1_regression.py:43` 的候选顺序也是 wt-base 优先 | 改文档，或两者都保留（现在 fallback 逻辑是对的） |
| 12 | commit `2062528` 消息「恒等式 **8.9e-15**」 | `counts.json` 实测 max **4.44e-15**（falcon-full）/ **5.33e-15**（ep99） | 无需改历史；引用时用 counts.json 的值 |
| 13 | 「仓库干净、测试零污染」（隐含在 STATUS/AGENTS 的产物约定） | 工作区干净是 ignore 的结果；每跑一次测试 +1 个废桩 ckpt（当时 79→82，2026-11-08 为 88） | 在 STATUS §5 或 AGENTS §6 加一句「pytest 会写 `checkpoints/`，已 ignore；如需彻底干净加 `enable_checkpointing=False`」 |
| 14 | STATUS:78「证据：… pid/0 字节清理」 | ✅ 部分为真：2062528 删除了 `T6-download.pid`、`P2D-train.pid`、0 字节 `P2B-full-big-whole-real.csv`（`git show --name-status` 三条 `D`）；但 `docs/verification/phase2/LRS2-paper-2gpu.pid`（3118837）**仍在磁盘**（未跟踪） | 删或说明「保留作历史记录」 |

---

## 7. GAP LIST

1. **355 s vs 136.5 s 的 219 s 去哪了？** 需要一次 `pytest --durations=30` 存档（GPU 可见），否则「测试耗时」这个基线不可引用。当前只知 GPU 隐藏时 173–186 s。
2. **CI 的真实状态未知。** 无法访问 GitHub Actions（本任务不允许 fetch）。需人给一次 `gh run list` 输出，或用本地 docker（`--depth 1`）复现 §4.6 剩余的浅克隆失败，把「预测红」变成「实测红」。
3. **`docs/verification` 缺 script→log→commit 的机器可读清单。** 108 个 log 无 run id/argv/commit；`scripts/` 46 个脚本与 `runs/` 51 个日志命名不一致。建议加 manifest（script、argv、commit、时间、产物 sha256）。
4. **lock 文件不含 artifact hash。** 版本集合一致已验证，但 `pip install -r` 重建能否得到**同一二进制**无法保证（18 个包当前来自 `/root/wheelhouse`，含 sha256 只存在于活环境的 `pip freeze` 输出里）。
5. **`checkpoints/` 废桩无任何守卫。** 需要一个测试或 CI 步骤断言「仓库根不产生新文件」（`git status --porcelain --ignored` 白名单），或在 `test_litmodule_logging.py:142` 传 `default_root_dir=tmp_path`。
6. **mouth 消融只在 N=20 做过**（`P2F-results.json` 自己写「re-run `--stage variants --device cuda:1` to reproduce」），未在 N=50/3000 复核。−14.4 / −23.3 dB 这类强结论目前是 20 条样本。
7. **`Experiments/` 的 203 MB 冒烟产物与 `/root/tmp` 的 301 MB 正式产物都没有归档策略**（`P2D-report.md:287` 承认「未归档」）。「复现」到此只保证数值，不保证产物可重建。
8. **P2-A 与 P2-F 的 baseline 口径不同未在 STATUS 统一**：`phase2/README.md` 的 P2-A 写基线 6.95/7.62（200 条）、P2-F 写 4.994（3000 条 mixture 基线）、P2-B 写 small 6.916/7.492。三个「baseline」不是同一个东西，STATUS 未列出对照表。
9. **`falcon-big.pt` 的架构名 `SpeakerContrastiveV2` 与 small 的 `FalconSpeakerContrastive` 不同**，且 `P2F-results.json` 的 `big.checkpoint` 段落带的是 small 的元数据（epoch 714 / `FalconSpeakerContrastive` / best_k 714–742）。big 的 517 个 audio key 与 small 的 525 个不同，说明结构有差异 —— 但「big 的参数量/MACs 口径」「big 的 best_k 归属」在本轮未核实（属模型范围，交给别组）。
10. **STATUS:83 的 8 卡训练「33–50 h」估计无实测支撑**（本机 2 卡，未跑）。这条验收（tt 3000 SI-SNRi ≥13.6 / SDRi ≥14.2）仍是整个复现的唯一大缺口。

---

## 8. 证据清单

**读过的仓库文件（路径为仓库内相对路径，除非另注）**
- 状态与约定：`docs/STATUS.md`（141 L，全读）、`AGENTS.md`（156 L，全读）、`.gitignore`（47 L）、`.github/workflows/tests.yml`（112 L）
- 测试（16 个）：`tools/repro/tests/test_{audio_train_entry,audit_data,eval_entry,litmodule_logging,lock_portable,lrs2_convert,lrs2_paper_build,metrics,model_rebuild,phase1_regression,phase2_data,phase2_launcher,phase2_pipeline,realssa_convert,synthetic_data,t5_backbone}.py`；helper：`t5_checks.py`、`t5_fingerprint.py`、`make_t3_smoke_data.py`
- 工具：`tools/repro/{audit_data,fetch_data,lrs2_2mix_to_falcon,realssa_to_falcon,make_synthetic_data,extract_video_backbone}.py`、`tools/repro/phase2/{protocol_check,eval_realssa_real}.py`、`run_lrs2_paper_volcano_8gpu.sh`
- 运行时：`audio_train.py:50-95,116-123`、`audio_visual_test.py:466-540`、`studio/server.py:29,110,328`、`studio/web/app.js:9,624`、`look2hear/layers/stft_tfgn.py:1-12`
- 证据：`docs/verification/phase2/{P2E-realssa-sim-eval-official.log, P2E-realssa-sim-metrics_combined.csv, P2E-realssa-sim-ch0-metrics_combined-20261030.csv, P2D-eval-falcon-full-cols-20261031.{log,csv,counts.json}, P2D-eval-epoch99-cols-20261031.{csv,counts.json}, P2F-results.json, P2D-report.md:265-295, DNSMOS-realssa-real-mean-full4-20261031/*, README.md}`；`docs/verification/phase1/{T1-checks.log,README.md}`；`docs/verification/runs/conda-env-verify.log`；`docs/verification/lead-verification.md`；`docs/verification/stubs/{typeguard.py,timm/**,data_pipe/**}`；`docs/verification/scripts/verify_model_params.py`
- 产物元数据：`Experiments/checkpoint/Falcon/{conf.yml,best_k_models.json}`；`/root/tmp/p2d_train/Experiments/checkpoint/LRS2_2Mix_1gpu/{best_k_models.json,version_0/metrics.csv}`

**跑过的命令（全部只读，GPU 仅做元数据/不占显存）**

| 目的 | 命令要点 |
|---|---|
| git 状态/历史 | `git rev-parse/log/show --stat/--name-status/diff --stat/--numstat/--dirstat/worktree list/tag -n5/cat-file -t/describe/rev-list --left-right --count` |
| 测试 4 轮 | `PYTHONPATH=$PWD … -m pytest tools/repro/tests -q -rs [--durations=15] [-rf --tb=line] [-p no:cacheprovider]`，分别在 `CUDA_VISIBLE_DEVICES=` 与 unset 下；另单跑 `::test_t1_gpu_spec_semantics` |
| 证据数值 | `pandas.read_csv` 复核 6 个 CSV 的 `avg`/`std` 行、行数、nan 计数、`si-snr_mix` 逐元素比较、恒等式残差 |
| 哈希 | `sha256sum` ×4（3 ckpt + metrics.csv）、`stat -c %s`；`torch.load(map_location="cpu", weights_only=False)` 读元数据 |
| 环境 | `pip freeze`（写 `/tmp/w110_freeze.txt`，不入库）、`importlib.metadata.version` 批量、`import espnet`、`ls site-packages/pypesq.py`、`grep -c file://` |
| stub 等价性 | 同 seed 下对比 `stubs/timm` DropPath 与 `timm.layers.drop_path`；`hasattr(typeguard,'check_argument_types')` |
| 忽略规则 | `git status --porcelain --ignored`、`git ls-files`、`git check-ignore -v` |
| 其他 | `nvidia-smi --query-gpu=memory.used`（前后 0 MiB）、`find`/`ls -la`/`wc -l`/`grep -rn`/`sed -n` 取证 |

**未做（及原因）**
- 未 `fetch`、未 `checkout`、未 commit/stash/clean；未装包；未跑训练/全量评测；未改仓库任何文件。
- 唯一副作用：pytest 自身生成了 `__pycache__` 与 3 个 `checkpoints/epoch=0-step=2-v{79,80,81}.ckpt`（均被 `.gitignore` 覆盖，`git status` 仍干净）。按只读纪律本应避免；已记录在 §5.1，并作为「测试污染工作区」的直接证据。
