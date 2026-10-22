# Phase 2 QA 报告（task-24，owner: layers-dsp）

> 分支 `repro/phase2` @ `af74cca`（main `ac91800` = Phase1 合并后）；**未 commit、未改 Phase1/2 源码**。
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / 2×A800）。
> 写作用域：`tools/repro/tests/test_phase2_*.py`（+ 对 `test_phase1_regression.py` 的基线路径/白名单/分阶段
> skip 条件维护）、`docs/verification/phase2/`。
> 本报告 **v1 = 终检**（快照时间 2026-10-21T19:52Z；P2-D 正式训练仍在进行，见 §6）。
> 日志：`docs/verification/phase2/QA-regression.log`。

## 1. 结论（v1 终检）

| 验收项 | 结果 | 证据 |
|---|---|---|
| pristine 基线 worktree `/root/wt-base`（detached `689555c`） | ✅ 已建并保持干净 | `git worktree list`（§2） |
| `test_phase1_regression.py` 跨分支基线优先 `/root/wt-base`、fallback `/root/wt-main`（断言不变） | ✅ | `BASELINE_WT_CANDIDATES` + `_baseline_worktree()`；数值不变量测试通过（§2） |
| 全量回归 `pytest tools/repro/tests -q -rs`（112 collected） | ✅ **111 passed / 1 skipped / 0 failed**，106.62s | §3.1；skip = `configs/Falcon.yml` 未被改动（条件跳过） |
| Studio 回归 `pytest studio/tests -q` | ✅ 35 passed / 2 skipped，18.81s | §3.2 |
| 官方权重 4 条指纹（eval / train-est / sep_emb / video） | ✅ 与 Phase1 基线**逐位一致**（Phase2 未改数值） | §3.3 |
| Phase2 新增 QA 测试 23 条（data 5 + pipeline 18） | ✅ 全绿（数据未落盘时自动 skip） | §4 |
| P2-C 转换数据（28000 条 / 23.36 GiB / target=s1 / mix=s1+s2） | ✅ 5/5 数据 QA 通过 | §5.3 |
| P2-A 200 条指标变体（报告表 mean±std/Δ） | ✅ 由 JSON 独立复算一致 | §5.1 |
| P2-B 7 组 CSV + 6 组配对统计 | ✅ 独立复算与报告**逐项一致**（均值 3 位小数、胜场数、总体 std） | §5.2 |
| P2-D 训练/评测管线 + 数字 | ✅ 冒烟与评测数字可追溯；正式训练 epoch 0 完成、持续中 | §6 |
| P2-E DNSMOS（环境/lock/sha256/分数域） | ✅ 版本与 conda lock 一致，4 个 ONNX sha256 齐全，分数 ∈ [1,5] | §5.5 |
| P2-F 指标实现独立对拍 | ✅ 自实现 vs `look2hear.metrics.core` vs 评测 CSV：最大差 **4.8e-8 / 1.8e-7 dB** | §5.6 |
| diff 范围守卫（Phase1/2 源码零改动） | ✅ protected 源码 vs main 差异 = **0**；tracked 改动仅文档/QA 测试 | §7 |
| 未 commit / 未改源码 | ✅ | `git status`（§7） |

**结论：Phase 2 交付物在“数值不变量、数据契约、指标实现、报告可追溯性、改动范围”五个维度全部通过
独立 QA；无 P0/P1。唯一未收口项是 P2-D 正式训练仍在进行（外部 GPU 资源限制），其已完成部分（冒烟、
epoch 0 曲线、评测管线）均通过校验，见 §6/§8。**

## 2. 基线 worktree 与跨分支基线修复

- Phase1 合并后 `main` = `ac91800`（不再是 upstream `689555c`），`/root/wt-main` 也随之更新，
  不能再作为“Phase1 前”的数值基线。QA 新建并保持：
  ```
  /root/code/github_repos/Falcon-fork  af74cca [repro/phase2]   ← 主工作区（Phase2）
  /root/wt-base                        689555c (detached HEAD)  ← pristine 基线（Phase1 前）
  /root/wt-main                        ac91800 [main]            ← 仅作 fallback
  ```
- `tools/repro/tests/test_phase1_regression.py` 改为：
  ```python
  BASELINE_WT_CANDIDATES = [Path("/root/wt-base"), Path("/root/wt-main")]
  MAIN_WT = BASELINE_WT_CANDIDATES[0]
  def _baseline_worktree() -> Path: ...   # 取第一个存在 Falcon.py 的候选
  ```
  **断言未改动**；`test_numeric_invariance_vs_main_worktree` 通过（与 689555c 数值逐位一致）。
- Phase1 验收测试在合并后原本会因 `_landed()`（“相对 main 有未提交改动”）全部 skip（6 条）。
  本次把 `_landed()` 改为 merge-aware：`main` vs `main^`（= upstream `689555c`）的已合并改动也算落地，
  于是 T1/T3/T5 验收测试在 Phase2 分支上重新生效（skip 由 6 降为 1，见 §3.1），**断言未改**。

## 3. 回归结果（终值）

### 3.1 全量 repro 测试

```bash
cd /root/code/github_repos/Falcon-fork
$ENV/bin/python -m pytest tools/repro/tests -q -rs --durations=10
# → 111 passed, 1 skipped, 0 failed, 43 warnings in 106.62s (0:01:46)
#   collected = 112
#   SKIPPED [1] test_phase1_regression.py:576: configs/Falcon.yml 未被改动（条件跳过：只有该文件被改时才校验注释级改动）
```

- 运行时间窗口内 P2-D 正式训练（PID 4002894，GPU0 ~83%）与其它 Phase2 评测并发，耗时属并发负载下数值。
- 首次（v0）失败项 `test_diff_scope_only_allowed_files` 已闭环：Phase2 时代的 `docs/*`、`tools/repro/*`、
  `configs/LRS2_2Mix.yml`、以及 **AGENTS.md 环境合规更新（§2/§4 + lock 115 行）** 均已授权进白名单
  （含注释说明“authorized: env compliance”），**源码保护不变**。
- 覆盖内容（112 条）包含：Phase1 全部验收/回归（21 条，T1–T5 重新生效）、Phase2 数据契约 5 条、
  Phase2 管线/指纹/证据/范围守卫 18 条、其余 68 条 Phase1 既有测试（训练入口、评测入口、指标、转换器、
  synthetic/realssa、litmodule、T5 骨干）。

### 3.2 Studio

```bash
$ENV/bin/python -m pytest studio/tests -q     # → 35 passed, 2 skipped in 18.81s
```
与 Phase1 基线（35 passed / 2 skipped）一致；全量 repro 套件内的 `test_studio_pytest_regression` 亦通过。

### 3.3 官方权重指纹（Phase2 不得改变任何数值）

| 指纹 | 值（与 `docs/verification/phase1/baseline-fingerprint.json` 比对） |
|---|---|
| `falcon.pt` sha256 / size | `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c` / 47,156,343 B |
| 参数量 audio / video / total | 537,697 / 11,186,688 / 11,724,385 |
| eval forward sha256 | `ad0f20a0e67e8e750058bfea69946ca1ee5ff9d7ec6d3caa1f212cb0e8f40757`（sum 0.3195922） |
| train forward est / sep sha256 | `5f01a59658471af0955cf6c18c09bb7c0788f844c9e0db334db36fa0a7e7545a` / `8a8e5c2a77eb0d78efd1ee1e9106b7b709326658fc13b1be3c67003c60bddc16` |
| video forward sha256 | `2afcea082697fdd859b28e4abbcf854abc942b074e9107a96bba9d636c19dd40`（sum 7832.2563） |

约定：seed(0) 造输入 → 构建 → strict load → eval/train forward；video 指纹**不调用 `.eval()`**
（Phase1 基线约定，BN3d 用默认 training 批内统计）。以上 4 条全部逐位一致。

## 4. Phase2 QA 框架（23 条，`tools/repro/tests/test_phase2_{data,pipeline}.py`）

| 文件 | 条数 | 覆盖 |
|---|---|---|
| `test_phase2_data.py` | 5 | P2-C 转换数据契约：JSON 一致性 / mouth 对齐+uint8 `[F,96,96,3]`（`F==n//640`）/ target=s1 且 mix=s1+s2 / near 条目 / split 计数 |
| `test_phase2_pipeline.py` | 18 | ①官方权重 4 指纹；②脚本 `py_compile`+`--help` 冒烟；③P2A–F 报告证据（进行中报告允许 pending 证据）；④LRS2 端到端评测冒烟（**优先 P2-C 转换 tt**，官方权重 CPU `--limit 2`）；⑤评测 CSV avg/std 复算；⑥rebuild 计数；⑦Phase2 diff 范围守卫；⑧P2B summary↔CSV；⑨P2C 转换日志↔索引；⑩P2A anchor 数据不变量；⑪P2E DNSMOS 环境+sha256+分数域；⑫P2F 指标实现对拍；⑬LRS2 配置不变量；⑭P2D 冒烟证据；⑮P2B 报告表+配对统计独立复算；⑯P2A 200 条报告↔JSON；⑰P2D 训练曲线（K=0/beta/val 正）；⑱P2D 评测 CSV↔报告数字 |

所有涉及外部数据/报告/产物的测试均“缺失即 `pytest.skip`”，不阻塞仓库内回归；本次终检全部实际执行
（无 skip）。评测冒烟使用 `/root/data/falcon_lrs2_2mix/tt`（P2-C 产物）而非 Phase1 遗留目录，
因此同时验证了“转换数据 → Dataset → 官方权重推理 → CSV 指标”全链路。

## 5. P2-A..F 独立校验（不采信报告，只采信 CSV/JSON/日志）

### 5.1 P2-A（`P2A-report.md` / `P2A-small-200.json` / `P2A-anchor-first20.json`）

- 数据侧不变量（200 条 + anchor）：`mix_r2 ≈ 0.99999999`；`target_equals_s1` max|Δ| = **0**；
  `mouth_equals_s1_npz` = **1.0**；全部 32000 samples / 16 kHz / 单声道。
- 200 条基线（official `falcon.pt`，whole 2 s，mouth_s1_88）：**SI-SNRi 6.955 ± 3.134 / SDRi 7.622 ± 2.800**；
  报告表 7 个核心行 mean±std 与 Δ 均由 JSON 独立复算一致（负号已归一化比对）。
- 变体上界：zero_mean=False −0.016、2×1 s 分段 +0.146、4×0.5 s 分段 +0.282、SDR 512→64 −0.352 dB
  ⇒ **指标/协议变体合计最多回收 ~0.35 dB**，不能解释 ~7 dB 差距（与 P2-F 结论一致）。
- 错误目标 target=s2：−16.998 ± 5.113（Δ −23.95 dB）；best-of-s1/s2（PIT 上界）6.976（+0.021）
  ⇒ 单源 target=s1 约定正确、无需 PIT。

### 5.2 P2-B（`P2B-report.md` / `P2B-b300-*.csv` / `P2B-summary-b300*.json`）

QA 用 `test_phase2_p2b_paired_stats_and_report_consistency` 从 7 个逐句 CSV 独立复算（剔除 `avg/std` 行）：

| variant / protocol / visual | n | SI-SNR | SI-SNRi | SDR | SDRi | PESQ | STOI |
|---|---|---|---|---|---|---|---|
| small / whole / real | 300 | 11.856 | **6.916** | 12.533 | **7.492** | 2.276 | 0.908 |
| small / windowed / real | 300 | 9.944 | 5.004 | 10.804 | 5.762 | 1.949 | 0.884 |
| small / whole / zero | 300 | −3.233 | −8.173 | −0.062 | −5.103 | 1.355 | 0.466 |
| small / whole / shuffle | 300 | −1.038 | −5.977 | 1.685 | −3.357 | 1.363 | 0.540 |
| big / whole / real | 300 | 12.827 | **7.887** | 13.445 | **8.403** | 2.503 | 0.921 |
| big / windowed / real | 300 | 10.508 | 5.569 | 11.311 | 6.270 | 2.093 | 0.893 |
| big / whole / zero | 300 | 8.452 | 3.512 | 9.309 | 4.267 | 2.000 | 0.813 |

- 上表数值与报告表 3 位小数一致（最大舍入差 5e-4）；每个 CSV 的 `avg`/`std` 行 = 逐句均值/总体 std（复算通过）。
- 配对统计独立复算与报告一致：**big−small +0.971 dB（258/300 胜）**；small windowed−whole **−1.912 dB
  （33/300）**；big windowed−whole −2.318 dB（33/300）；zero-mouth −15.089 dB（4/300）；
  shuffle −12.894 dB（7/300）；big zero −4.375 dB（26/300）。
- 结论采信：视觉分支真实生效（置零/错配大幅掉点），窗口化协议比整句低 ~1.9–2.3 dB，big 仅比 small 好 ~1 dB。

### 5.3 P2-C（`P2C-report.md` / `P2C-convert.log` / `/root/data/falcon_lrs2_2mix`）

- 条数：tr **20000** / cv **5000** / tt **3000**（= 28000，源 `audio/wav16k/min/*` 一一对应）；
  `missing_mouth_npz=0`；每 split `INDEX READY` 与 `converted` 数字均与索引条目数一致（QA 复算）。
- 磁盘 23.36 GiB（24 GB），tr/cv/tt 转换耗时 33.6/11.0/5.9 s（16 并行，整次运行 237.9 s）；
  mouths 按 target_id 去重 28000 → 15516（省 ~17 GB）。
- 数据契约（QA 5 条测试全过）：mix/target/near 三表等长、路径存在、`n_samples == wav 帧数`；
  mouth `uint8 [F,96,96,3]`、`F==n//640`、命名 == mix 名前两段（s1_id）；target == rebuild s1、
  mix == s1+s2（max|Δ|<1e-3）；near 为 fallback 自引用 ⇒ 训练期 K=0（P2-D 指标已证实）。

### 5.4 P2-D 见 §6。

### 5.5 P2-E（`P2E-report.md` / `P2E-*.json` / `P2E-env-delta.txt` / lock）

- 环境合规：`speechmos==0.0.1.1`、`onnxruntime==1.31.0`、`protobuf==7.36.2`、`flatbuffers==25.12.19`
  已进 `docs/verification/conda-env-lock.txt`（115 行）且 AGENTS.md §2 记录（QA 断言 lock 含 4 包）。
- DNSMOS 模型：4 个 ONNX（`sig.onnx`/`bak_ovr.onnx`/`sig_bak_ovr.onnx`/`model_v8.onnx`）字节数与 sha256
  均记录在 `P2E-dnsmos-synthetic.json`；4 个合成 case 的 P.835/P.808 分数全部有限且 ∈ [1,5]。
- LRS2-2Mix 冒烟（2 条）：`num_ok==num_files==2`、`num_errors==0`；RealSSA-Real 2 段：
  est `sig 3.175 / bak 2.664 / ovrl 2.246 / p808 3.147`，mix 基线 `2.717 / 1.551 / 1.535 / 2.658`
  （数字可追溯；样本量小，仅作管线可用性证据）。

### 5.6 P2-F（`P2F-report.md` / `P2F-results.json`）

- 指标实现：自实现 vs `look2hear/metrics/core.py` **max|Δ| = 4.82e-8 dB**；vs Phase1 全量评测 CSV
  **1.84e-7 dB**；mixture baseline 公式 **4.47e-8 dB**（SDR baseline 8.9e-14）；`csv_avg_recompute_ok=True`、
  `dataset_order_match=50/50`；逐句三源（my/core/csv）QA 复算一致（阈值 1e-5 / 1e-4）。
- 协议自洽：3000/3000 条 `max(|mix|,|s1|,|s2|)==0.9`；`mix−(s1+s2)` max 3.05e-5（16-bit PCM 1 LSB）、
  相对 RMS 1.9e-4；全部 clip 32000 samples/16 kHz；target=s1 的 SI-SDRi 7.535 vs s2 −17.012（Δ+24.55）；
  对齐 lag 恒 0、无削波、尺度不变（1e-8 dB）、zero-mean Δ−0.0005 dB、8 kHz Δ+0.11/0.21 dB。
- mouth 消融：置零 −14.4 dB、错说话人 −23.3 dB ⇒ 视觉分支生效且目标正确。
- 结论采信：7.03/7.71 属“协议可信的跨语料结果”，非论文 LRS2 列协议复现；缺口是 LRS2-2Mix 训练出的
  checkpoint（P2-D），不是评测/指标修复。

## 6. P2-D 训练/评测数字（快照 2026-10-21T19:52Z）

| 项 | 数字 | 证据 |
|---|---|---|
| 2-GPU 100-step 冒烟 | 2.97 it/s（global bs8 ≈23.8 clips/s）、wall 73 s；val 1.946 / test 1.912 dB | `P2D-smoke.log`、`P2D-report.md` §3 |
| 1-GPU 冒烟 | `P2D-smoke-1gpu.log`（配置/显存/吞吐） | `tools/repro/phase2/smoke_lrs2_1gpu.yml` |
| 正式训练 | PID 4002894，2026-10-21T19:26:41Z 启动；单卡 bs4、train_samples=10000 ⇒ **2500 step/epoch**、epochs=100 | `P2D-train.pid` / `P2D-train.log` / `/root/tmp/p2d_train/.../metrics.csv` |
| epoch 0（已完成） | train_loss_epoch **−6.697**；**val_pit_sisnr 6.732**、test_pit_sisnr 6.815 dB；beta=0；K=0（0near=4，1near+ 全 0） | metrics.csv（QA `test_phase2_p2d_training_curve_sane` 解析） |
| epoch 1（进行中） | beta=0.005（=1/200，符合 0→1 over 200 epoch）；训练 step 持续 | 同上 |
| 评测管线冒烟 | 100-step 冒烟 ckpt 在 5 条 LRS2 tt：sdr 3.585 / sdr_i −1.784 / si-snr 2.811 / si-snr_i −2.490 / pesq 1.141 / stoi 0.762（**仅证明管线可用，不作为结果**） | `P2D-eval-epoch_0.csv`（QA 复算 avg=逐句均值，且报告数字可追溯） |
| 正式训练后续 | ⏳ 训练持续中（受外部 GPU 占用限制，单卡 ~14 min/epoch；100 epoch 远超本轮时间窗） | — |

> QA 已把“训练曲线合理性”固化为测试：epoch 0 step < 2500、K=0 语义、beta 从 0 单调增长、
> 损失有限且不恶化、val PIT SI-SNR 有限为正。后续 epoch 落盘后可直接重跑该测试复核。

## 7. 工作区 / diff 范围守卫

- `git diff --name-only main`（tracked 改动 7 个，全部文档/QA 测试，**无源码**）：
  `AGENTS.md`、`docs/DATA_ACCESS_GUIDE.md`、`docs/REPRODUCTION_PHASE1_REPORT.md`、`docs/REPRODUCTION_PLAN.md`、
  `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md`、`docs/verification/conda-env-lock.txt`、
  `tools/repro/tests/test_phase1_regression.py`。
- untracked 54 个：`configs/LRS2_2Mix.yml`、`docs/verification/phase2/*`、`tools/repro/phase2/*`、
  `tools/repro/lrs2_2mix_to_falcon.py`、`tools/repro/tests/test_{lrs2_convert,phase2_data,phase2_pipeline}.py`。
- **受保护源码 vs `main` 差异 = 0**：`audio_train.py`、`audio_visual_test.py`、`look2hear/`、`studio/`、
  `configs/Falcon.yml`（Phase2 守卫断言）。未 `git commit`、未切分支、未清理工作区。
- 白名单授权记录：`docs/*`、`tools/repro/*`、`configs/LRS2_2Mix.yml`、`AGENTS.md`（环境合规 §2/§4 + lock）、
  `space/*`；Phase1 测试中的 `configs/Falcon.yml` 注释级改动仍由独立测试保护。

## 8. 发现与风险（不阻塞，均已记录）

1. **P2-D 正式训练未完成**（P2-G 唯一开放项）：epoch 0 已通过 QA，但 100 epoch 需要数小时；论文级
   LRS2-2Mix 数字需训练完成后再评测。属资源/时间限制，非实现缺陷。
2. `P2B-full-big-whole-real.csv` 为 0 字节占位（全量 big 评测因时间窗中止）；报告已注明“未跑全量”，
   不影响已交付的 300 条配对结论。建议后续清理或补跑。
3. `P2D-train.log` 在 epoch 结束前只含启动段（PL 进度条写入方式导致缓冲）；逐 step 曲线以
   `/root/tmp/p2d_train/.../metrics.csv` 为准（QA 测试读取该文件）。
4. AGENTS.md §2 称 DNSMOS “模型 sha256 见 `P2E-env-delta.txt`”，实际 sha256 在
   `P2E-dnsmos-synthetic.json` / `P2E-report.md`；`P2E-env-delta.txt` 含安装命令/许可/依赖说明。属文档指向小瑕疵。
5. 并发负载：终检期间 P2-D 训练与其它评测并发（GPU0 ~83%），全量套件 106.62s 属该负载下数值；
   数值不变量与指纹均通过，不受影响。
6. 全量套件唯一 skip（`configs/Falcon.yml` 未被改动）是条件测试的预期行为；如需强制覆盖，可对
   `main` 做一次注释级改动后重跑。

## 9. 复现命令（全部在 `falcon` 环境）

```bash
cd /root/code/github_repos/Falcon-fork
$ENV/bin/python -m pytest tools/repro/tests -q -rs          # 111 passed, 1 skipped（本机快照）
$ENV/bin/python -m pytest tools/repro/tests/test_phase2_data.py -q
$ENV/bin/python -m pytest tools/repro/tests/test_phase2_pipeline.py -q
$ENV/bin/python -m pytest studio/tests -q                   # 35 passed, 2 skipped
git worktree list                                           # 校验 /root/wt-base = 689555c detached
```

## 10. 变更记录

- **v0（19:03Z）**：框架落盘（数据 5 条 + 管线 7 条）、初始回归（1 failed 为 QA 自身白名单，已修）、
  worktree 基线路径修复、Studio 35/2、指纹逐位一致。
- **v0.1（19:35Z）**：AGENTS.md 环境合规白名单闭环（2 守卫 2 passed）；中间全量 88 passed/6 skipped
  （旧收集导致的报告证据失败，已修）；P2-C 数据落盘 → 数据 QA 5 passed。
- **v1（19:52Z，本版）**：Phase2 QA 扩到 23 条（新增 P2A/P2B/P2C/P2D/P2E/P2F 证据复算 11 条）；
  `_landed` merge-aware 恢复 Phase1 T1/T3/T5 验收覆盖（skip 6→1）；终检
  **111 passed / 1 skipped / 0 failed, 106.62s**；P2-A..F 数字独立复算全部一致；P2-D epoch 0 通过；
  无源码改动、未 commit。**task-24 保持 in_progress（P2-D 训练未完成，等 Lead 收口）。**
