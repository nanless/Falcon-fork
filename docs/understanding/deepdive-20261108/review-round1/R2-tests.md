# R2 · 测试实跑与红绿归因

> **2026-11-10 复核提示：**本页记录当时的对照实验、测试数和临时路径。本轮 CPU 结果为 166 passed / 2 skipped，条件和跳过原因见[复现状态](../../../STATUS.md)；历史 checkpoint 的用途和保留状态需逐项确认。

- 机器：`dev_a800_2gpus`（A800×2，跑测试时 0% 占用）；仓库 `Falcon-fork`，HEAD `6c1d5b7`，工作区 25 改 + 2 未跟踪。
- 解释器 `/root/miniforge3/envs/falcon/bin/python`（torch 2.8.0+cu128 / py3.11.17），`PYTHONPATH=<repo>`，`OMP_NUM_THREADS=4`。
- 纯 HEAD 对照副本：`/tmp/r2_head`（`git clone` 自本仓，`git status` 干净，HEAD 同为 `6c1d5b7`）。
- 只读仓库：未改任何受版本控制文件；pytest 自身产生的 6 个 `checkpoints/*.ckpt` 桩已按 mtime 精确删除（见 §4），仓库回到 87 个桩。

## 0 结论速览

| # | 测试项 | 当前工作区 | 纯 HEAD `6c1d5b7` | 本轮新增红 |
|---|---|---|---|---|
| 1 | `tools/repro/tests`（有 CUDA） | **4 failed / 164 passed**，144.5s | **4 failed / 162 passed**，129.8s | **0** |
| 2 | `tools/repro/tests`（无 CUDA） | **6 failed / 162 passed**，159.1s | **6 failed / 160 passed**，148.3s | **0** |
| 3a | `studio/tests`（有 CUDA） | 51 passed / 2 skipped，18.5s | 51 passed / 2 skipped，19.4s | 0 |
| 3b | `studio/tests`（无 CUDA） | 53 passed，40.1s | 53 passed，41.2s | 0 |
| 4 | Node 8 个 check/test 脚本 | 8/8 PASS | 8/8 PASS | 0 |
| 5 | `space` build + i18n + static | build rc=0 / i18n 226 条 / 3 passed | 文件未改动，未复跑 | 0 |
| 6 | 4 个新实验脚本 `--help` | 4/4 rc=0（≤1.5s） | 文件为新增，无基线 | — |

- **结论：本轮改动没有让任何原本绿的用例变红。** 全部红点都在纯 HEAD 副本上原样复现（同名同数量）。
- 4 条指纹失败与线程设置有关：4 条指纹用例只在 `OMP_NUM_THREADS=4` 下红，用 torch 默认 64 线程时与基线**逐位相同**（见 §3 证据）。
- **建议合并**（无新增红）；但 §5 第 1、2 条（线程固定、checkpoints 污染）建议合并前一并修掉，成本极低。

## 1 实测输出

### 1.1 `tools/repro/tests`（有 CUDA）
```
$ PYTHONPATH=$PWD OMP_NUM_THREADS=4 python -m pytest tools/repro/tests -q -rs
4 failed, 164 passed, 43 warnings in 144.48s (0:02:24)
FAILED tools/repro/tests/test_phase1_regression.py::test_eval_forward_fingerprint_unchanged
FAILED tools/repro/tests/test_phase1_regression.py::test_train_forward_fingerprint_unchanged
FAILED tools/repro/tests/test_phase1_regression.py::test_numeric_invariance_vs_main_worktree
FAILED tools/repro/tests/test_phase2_pipeline.py::test_phase2_official_weight_fingerprints_unchanged
```
首行报错（4 条同源）：
- `AssertionError: eval 输出指纹变化！若 T1–T5 修复未触碰前向，说明引入了数值变化`
  `assert 'a09288c50e9d...3f7a20735e962' == 'ad0f20a0e67e...12cb0e8f40757'`（test_phase1_regression.py:240）
- `assert 'c0d92eac8ad0...befb257342be9' == '5f01a5965847...36fa0a7e7545a'`（:250，train_est）
- `test_numeric_invariance_vs_main_worktree`：在 `/root/wt-base`（689555c，clean）复算得到**同样的** `a09288…`，与 baseline 不等（:331）
- `test_phase2_official_weight_fingerprints_unchanged`：`audio_params/video_params` 两条断言通过，卡在 `eval_sha`（test_phase2_pipeline.py:110）

### 1.2 `tools/repro/tests`（无 CUDA）
```
$ CUDA_VISIBLE_DEVICES= ... pytest tools/repro/tests -q -rf
6 failed, 162 passed, 39 warnings in 159.14s
```
= 上述 4 条 + 2 条只在无 GPU 时红的：
- `test_studio_pytest_regression`：`AssertionError: Studio skip 数变化：0 (期望 2)`
- `test_t1_gpu_spec_semantics`：`Failed: DID NOT RAISE ValueError`（断言只在 GPU 语义下成立）

### 1.3 `studio/tests`
```
（有 CUDA） 51 passed, 2 skipped in 18.52s
（无 CUDA） 53 passed in 40.07s
```

### 1.4 Node
```
PASS studio/tests/check_i18n.mjs / check_mouth_tracking.mjs / check_playback_resampler.mjs / check_spectrum.mjs
PASS studio/tests/check_capture_dsp.cjs / check_tracking_ui.cjs
PASS studio/web/capture-worklet.test.cjs / video-worker.test.cjs      → 8/8 PASS，无 FAIL 行
```

### 1.5 space
```
$ python space/build.py --out /tmp/r2_space     → Assembled /tmp/r2_space（rc=0）
$ node space/tests/check_i18n.mjs /tmp/r2_space → i18n: 226 Chinese strings translated（rc=0）
$ python -m pytest space/tests/test_static.py -q → 3 passed in 0.16s
```

### 1.6 新实验脚本冒烟（各 ≤60s）
```
lookahead_ab rc=0 / ln_variant rc=0 / streaming_consistency rc=0 / video_ablation_check rc=0
```
全部 1s 级返回 usage，无卡死、无超时。

## 2 HEAD 对照

| 项 | HEAD | 现在 | 差 |
|---|---|---|---|
| repro 有 CUDA | 4F / 162P | 4F / 164P | 失败集合**完全相同**；P +2 = 新增用例 |
| repro 无 CUDA | 6F / 160P | 6F / 162P | 同上 |
| studio 有/无 CUDA | 51P+2S / 53P | 相同 | 无 |
| Node | 8/8 | 8/8 | 无 |

- **HEAD 既有红（4 条，有 CUDA）**：`test_eval_forward_fingerprint_unchanged`、`test_train_forward_fingerprint_unchanged`、`test_numeric_invariance_vs_main_worktree`、`test_phase2_official_weight_fingerprints_unchanged`。
- **HEAD 既有红（+2 条，无 CUDA）**：`test_studio_pytest_regression`、`test_t1_gpu_spec_semantics`。
- **现在红 = HEAD 红，新增红 = 0 条**。
- 用例总数 166 → 168：`tools/repro/tests/test_metrics.py` 新增 `test_sisnri_instantiable_and_forward`、`test_pit_pw_pt_2d_does_not_raise`，两条**均通过**。

## 3 新增红的归因

以下记录 4 条既有指纹失败的线程对照实验：

`eval_sha` 的计算方式：
- `tools/repro/tests/test_phase1_regression.py:52-59` → `_tensor_sha(t) = sha256(t.numpy().tobytes())`，即 **float32、CPU、contiguous 原始字节**的 sha256（无量化、无舍入、无容差）。
- 调用点：`:199 eval_sha`、`:202 train_est_sha`、`:204 sep_sha`、`:207 video_sha`；`video_sha` 通过 → 只音频前向受影响。
- 对照侧：`docs/verification/phase1/baseline-fingerprint.json`；`test_phase2_pipeline.py:110` 只比 `eval_sha`。

**漂移性质：所测输出的 float32 原始字节随线程数变化**：参数数（537,697 / 11,186,688）、state_dict key 数（525 / 137）、输出 shape 全部与基线一致；只有字节级数值不同。标量证据：

| 线程数 | `eval_sha`(前 12) | `eval_sum` |
|---|---|---|
| 1 | `fe52176dc4aa` | 0.3195900321006775 |
| 4 | `a09288c50e9d` | 0.3195908069610597 |
| 8 | `c245e424f14f` | 0.3195924162864685 |
| 16 | `8702f48341a4` | 0.3195921182632446 |
| 32 | `39fafc4eaa66` | 0.3195907473564148 |
| **64（默认，未设 OMP）** | **`ad0f20a0e67e`** | **0.3195921778678894 = 基线逐位相同** |

- 基线 `sum=0.3195921778678894` 与 64 线程结果**完全相等**，`est_sha/sep_sha` 也同时命中基线。
- 结论：float32 归约/卷积的求和顺序随 intra-op 线程数变化，末位抖动（4 线程相对基线 |Δsum|≈1.4e-6，相对 4.3e-6）。**与本次 diff 无关**：`/tmp/r2_head`（HEAD）与 `/root/wt-base`（689555c，clean，另有独立复算）在 4 线程下都产出同一个 `a09288…`。
- 最小复现（两条命令的差只在线程数）：

```bash
cd /root/code/github_repos/Falcon-fork
# 红：按约定 OMP=4
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python -m pytest \
  tools/repro/tests/test_phase1_regression.py::test_eval_forward_fingerprint_unchanged \
  tools/repro/tests/test_phase2_pipeline.py::test_phase2_official_weight_fingerprints_unchanged -q
# 绿：不设 OMP（torch 默认 64 线程）
env -u OMP_NUM_THREADS CUDA_VISIBLE_DEVICES= /root/miniforge3/envs/falcon/bin/python -m pytest \
  tools/repro/tests/test_phase1_regression.py tools/repro/tests/test_phase2_pipeline.py -q
# → 38 passed（4 条指纹全绿，仅剩 2 条只在有 GPU 时才成立的断言）
```

无 CUDA 那 2 条的归因：`test_phase1_regression.py` 内嵌断言把「有 GPU」写死（studio 期望恰好 2 skip、T1 期望 `ValueError`），属环境断言，非本轮代码问题。

## 4 `checkpoints/` 污染复核（结论：**没修掉**）

- 我进场时：`checkpoints/*.ckpt` = **87** 个（其中 v81–v86 为今日更早的其他 review 跑出，非我产生）。
- 跑完 6 次 pytest 后 = **93**，`du -sh checkpoints` = 752K；每个新桩 **5,269 B**，命名 `epoch=0-step=2-vNN.ckpt`。
- 单点归因（隔离实跑）：只跑 `test_litmodule_logging.py::test_fit_without_logger_does_not_crash` → 92 → **93（+1）**。
- 根因：`tools/repro/tests/test_litmodule_logging.py:142` 的 `pl.Trainer(...)` 未传 `default_root_dir=str(tmp_path)`、也未 `enable_checkpointing=False`，Lightning 默认 `ModelCheckpoint` 落到 `cwd/checkpoints/`；`test_fit_with_csvlogger` 只把 **logger** 指到 tmp_path，checkpoint 仍落仓库根。
- HEAD 侧同样 +1/次（`/tmp/r2_head` 跑 2 次 → 2 个桩），说明**该污染是既有问题，不是本轮引入**。
- **关键事实纠正：本轮没有改 `test_litmodule_logging.py`**——`git diff tools/repro/tests/test_litmodule_logging.py` 为空，`git status` 里测试目录只有 `M tools/repro/tests/test_metrics.py`；该文件最后一次改动是 `ac91800`。所以「已被修掉」的前提不成立。
- 我造成的 6 个桩（v87–v92）已按文件名精确 `rm` 并复核回到 **87**，未触碰其他桩。

## 5 合并前必修清单

1. **[阻塞·4 条红]** 指纹用例固定线程：`test_phase1_regression.py` 的 `fingerprints` fixture 开头加 `torch.set_num_threads(64)`（或从基线 JSON 读线程数并在断言消息里打印 `torch.get_num_threads()`）。否则任何按项目约定 `OMP_NUM_THREADS=4` 跑的人都会看到 4 红，且会被误判成「数值回归」。
2. **[阻塞·污染]** `test_litmodule_logging.py:142` 的 `Trainer` 补 `enable_checkpointing=False`（或 `default_root_dir=str(tmp_path)`）；顺手清掉 `checkpoints/` 里 87 个 5,269 B 死桩（752K）。
3. **[阻塞·事实]** 更正 review 前提：「本轮改了 `test_litmodule_logging.py` 修掉污染」与 diff 不符（该文件零改动），结论需按「未修」重述。
4. **[非阻塞]** 无 GPU 时 `test_studio_pytest_regression`（断言 studio skip==2）与 `test_t1_gpu_spec_semantics`（断言 `ValueError`）应改 `skipif(not torch.cuda.is_available())`，否则 `CUDA_VISIBLE_DEVICES=` 的常规跑法恒 2 红。
5. **[非阻塞]** `test_numeric_invariance_vs_main_worktree` 依赖 `/root/wt-base` 存在且 clean；在干净 CI/新机器上会 skip（等于没跑），建议把该目录准备写进 README 或在缺失时 fail。
6. **[无需动作]** `test_metrics.py` 新增 2 条用例、Node 8 个脚本、space build/i18n/static、4 个新实验脚本冒烟全绿。

## 6 未跑 / 受限项

- 未跑训练/评测全量（按要求），GPU 仅用于 pytest；期间两卡 0% 占用。
- `space` 未在 `/tmp/r2_head` 复跑：本轮该目录只有 `space/README.md` 改动，build/i18n/test 代码与 HEAD 一致，故不做基线。
- 指纹全部在 CPU 上计算（`map_location="cpu"`，无 `.cuda()`），**CUDA 前向数值未验证**；GPU/CPU 一致性不在本次范围。
- 单条命令均 ≤400s；最慢项为 1.2（159.1s）、1.1（144.5s），未分组也一次跑完。
- `/tmp/r2_head` 与 `/root/r2probe/probe_fp.py`（探针脚本）保留在远端，可复算；均在仓库外。
