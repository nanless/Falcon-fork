# Phase-1 QA 报告：独立回归与数值不变量验证（task-15）

> owner：`layers-dsp` · 分支 `repro/phase1` @ 基线 `689555c`（**未 commit、未切分支、未改源码**）
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / PL 2.6.6 / 2×A800，CPU 验证）
> 写作用域：`tools/repro/tests/test_phase1_regression.py`、`docs/verification/phase1/`（本报告 + `QA-regression.log`）
> 回归脚本：`tools/repro/tests/test_phase1_regression.py`（16 条独立 pytest 用例）
> 完整日志：`docs/verification/phase1/QA-regression.log`（命令 + 输出 + 指纹实测值）
> 基线：`docs/verification/phase1/baseline-fingerprint.json`（Lead 在未改动 main worktree 生成）+ 干净
> `main` worktree `/root/wt-main`（跨分支对比用）

## 1. 结论

**PASS（Phase-1 Gate G1 的 QA 部分通过；无阻塞项）。P0 = 0、P1 = 0、P2 = 6。**

| 验收项（Gate G1） | 结果 | 证据 |
|---|---|---|
| 1) `pytest tools/repro/tests -q` 全过 | ✅ **82 passed / 0 failed**（终检 v2；含 QA 21 条 + 各 T* 61 条） | QA-regression.log §3 |
| 2) 冒烟训练可存 ckpt / loss 有值 | ✅（Lead `integration-smoke.log`：1 epoch、val_loss=12.188、`epoch=0.ckpt` + `best_model.pth`） | `integration-smoke.log`；T1/T2 报告 |
| 3) 冒烟评测可出 CSV / 指标量级/符号 | ✅ 流程跑通 + 符号正确（我的独立指标测试：好估计为正；`est=mix ⇒ 提升恰 0.000`） | QA-regression.log §2；本报告 §5 |
| 4) `pytest studio/tests -q` = 35 passed / 2 skipped | ✅ **35 passed, 2 skipped** | QA-regression.log §4 |
| 5) **官方权重数值指纹不变**（4 条 sha256 全部对齐 baseline） | ✅ **eval / train-est / sep_emb / video 全部 sha256 相同** | 本报告 §3；QA-regression.log §6b |
| 6) 每个 T* 报告含 file:line/命令/结果/遗留 | ✅（T1–T6 报告 + REVIEW + REGRESSION 均已落盘） | `docs/verification/phase1/*.md` |

**关键数字（独立重算，非引用实现者日志）**：
`audio params 537,697`、`video 11,186,688`、`total 11,724,385`；`eval sha256=ad0f20a0…f40757`；
`train est 5f01a596…e7545a`；`sep_emb [1,1,112,251,129] 8a8e5c2a…bddc16`；`video [1,512,50] 2afcea08…dd40`（sum=7832.2563）。

**P2 清单（非阻塞，建议收尾顺手处理）**：

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| P2-1 | T1：`audio_train.py:2` 的 `import sys` 现已未使用（清理残留） | `grep -n "sys\." audio_train.py` 无命中 | 删除该 import |
| P2-2 | T2：日志 dict 中移除 `"epoch"` 字段（原 Comet x 轴），改由 PL `on_epoch=True` 提供 `epoch` 列；key 集与旧 CSV/Comet 不完全相同 | `av_litmodule.py:157-186`；T2-report §2 | 已在 T2 报告说明；如需 100% 兼容可补同名 dummy 字段（不必） |
| P2-3 | T5 报告称"baseline 的 video 指纹无法复现（试了 8 种约定）"——**QA 已定位原因**：baseline 视频指纹是在**未调用 `.eval()`**（FRCNN 默认 `training=True`，BN 用批内统计）下生成；评测时先用 `.eval()` 会得到不同 sum（5955.7 vs 7832.3）。T5 自己的 pre/post 指纹自洽但 ≠ baseline 的 video sha | `T5-report.md:105-106`；本报告 §3.2 | 在 T5 报告/README 补一句"baseline video 指纹约定 = 默认 training 模式（不 evall）"；QA 套件已固化该约定 |
| P2-4 | T3：legacy `SPlitMetricsTracker`（`splitwrapper.py`）仍是旧式负号/SNR 语义，且仍在 `look2hear.metrics.__all__` 中导出；评测链路已改用 `ALLMetricsTracker`，故不影响结果，但 API 不一致 | `look2hear/metrics/splitwrapper.py`；`metrics/__init__.py` | Phase 2 统一或标注 deprecated |
| P2-5 | T3：`requirements.txt` 仍列 `torch-mir-eval`，但 `wrapper.py` 已移除该 import（仅历史死路径） | `requirements.txt:24` vs `wrapper.py:13` | 保留环境兼容即可；或标明"legacy" |
| P2-6 | **流程**：多 agent 并发跑 `pytest tools/repro/tests` 时耗时 ~65s→146s+（GPU/DDP 争用）；实现者写入文件期间会出现瞬时失败（我 16:38 观测到 4 个 T3 测试瞬时失败，文件稳定后 20 passed） | 本报告 §6 观察 | QA/CI 终检应独占运行；不要在他人在写文件时下结论 |

## 2. 验证矩阵（命令 / 结果 / 证据）

| # | 命令 | 结果 | 日志 |
|---|---|---|---|
| V1 | `python -m pytest tools/repro/tests/test_phase1_regression.py -q` | **16 passed**（40.2 s） | QA-regression.log §2 |
| V2 | `python -m pytest tools/repro/tests -q` | **70 passed**（54.1 s，独占运行） | §3 |
| V3 | `python -m pytest studio/tests -q` | **35 passed, 2 skipped**（15.7 s） | §4 |
| V4 | `python -c "import look2hear.models, ..., audio_train, audio_visual_test"` | **IMPORTS_OK (12 modules)** | §5 |
| V5 | 官方权重指纹（QA 套件断言 + 独立重算） | 4/4 sha256 与 baseline 一致；参数一致 | §6/§6b |
| V6 | `/root/wt-main`（干净 main）子进程重算 4 指纹并与本分支逐位比较 | 全部相同（`max|Δ|=0` 等价） | §2（`test_numeric_invariance_vs_main_worktree`） |
| V7 | `git diff main` 范围守卫 + `py_compile` | 仅允许的 11 个文件被改；11 个 .py 全部编译通过 | §1/§7 |
| V8 | 指标第二意见（fast_bss_eval 直算 + 解析式） | T3 的 SDR/SDRi/SI-SDR/SI-SDRi 与独立计算一致（±1e-4 / ±5e-3 dB）；`est=mix ⇒ 提升 0` | §2（指标两条用例） |

## 3. 数值不变量（本 QA 的核心交付）

### 3.1 官方权重 + 固定种子指纹（与 baseline 逐位一致）

| 指纹 | 形状 | sha256 | baseline |
|---|---|---|---|
| eval 前向 | `[1,1,32000]` | `ad0f20a0e67e8e750058bfea69946ca1ee5ff9d7ec6d3caa1f212cb0e8f40757`（sum 0.3195922） | ✅ 相同 |
| train 模式 est | `[1,1,32000]` | `5f01a59658471af0955cf6c18c09bb7c0788f844c9e0db334db36fa0a7e7545a` | ✅ 相同 |
| train 模式 sep_emb | `[1,1,112,251,129]` | `8a8e5c2a77eb0d78efd1ee1e9106b7b709326658fc13b1be3c67003c60bddc16` | ✅ 相同 |
| 视频前端 | `[1,512,50]` | `2afcea082697fdd859b28e4abbcf854abc942b074e9107a96bba9d636c19dd40`（sum 7832.2563） | ✅ 相同 |

**算子顺序（QA 套件已固化为可复现约定）**：
`torch.manual_seed(0)` → `audio=randn(1,32000)`、`mouth_emb=randn(1,512,50)` → 构建 `FalconSpeakerContrastive`
（构建消耗 RNG）→ `load_state_dict(strict=True)` → eval 前向 → train 前向；
`torch.manual_seed(1)` → `frames=randn(1,1,50,88,88)` → 构建 `FRCNNVideoModel` → **不调用 `.eval()`**
（默认 `training=True`，BN3d 用批内统计）→ 前向。**这解释了 T5 报告无法复现 baseline video 指纹的原因（P2-3）**。

### 3.2 修复不改变数值的跨分支证据

- `test_numeric_invariance_vs_main_worktree`：在 `/root/wt-main`（干净 main，未改动）用**独立子进程**重算上述
  4 条指纹 + 参数量，并与本分支逐项比对 ⇒ **全部相同**。同时证明 baseline JSON 本身可复现（main 侧值 = JSON 值）。
- T5 自己的 pre/post 指纹 JSON（`T5-pre-fix-fingerprint.json` / `T5-post-fix-fingerprint.json`）内部逐位相同，
  且 eval/train est 与 Lead baseline 一致（video 例外见 P2-3）。

## 4. 代码审查（T1–T5 diff 独立评审）

方法：`git diff main -- <files>` 逐 hunk 读 + `py_compile` + QA 套件的范围/语法/接口守卫 + 抽查运行行为。
总体：**未发现数值改变、接口破坏、未定义名、日志 key 丢失或异常被吞掉的 P0/P1 问题**。分任务评估：

| 任务 | 关键改动 | 评估 |
|---|---|---|
| **T1** `audio_train.py` | `build_logger()`（csv/tb/comet/none，缺依赖清晰报错）；`resolve_accelerator_and_devices()`（PL2 `accelerator/devices/strategy`，10 卡配置在 2 卡机收敛并告警）；conf dump 前移 + `deepcopy`；resume/收尾稳健性 | ✅ 与 REPORT 一致；`git diff` 中模型/优化器/损失构造语句未变；无 `comet_logger` 残留；P2-1（unused sys） |
| **T2** `look2hear/system/av_litmodule.py` | 4 处 `logger.experiment.log` → 末尾 `log_dict(metrics, on_step=False, on_epoch=True)`；keys/数值语义保留（`learning_rate/val_pit_sisnr/val_pit_sisnr_*near/test_pit_sisnr`） | ✅ 无 logger 时不崩；CSV 列验证（`test_litmodule_logging.py` 5 passed）；P2-2（epoch 字段） |
| **T3** `look2hear/metrics/*` + `audio_visual_test.py` | 新增 `core.py`（正号 SI-SDR/SDR/SDRi 唯一定义）；`ALLMetricsTracker`/`MetricsTracker` 改用 core；PESQ 实参修正；eval 入口支持 `--conf_dir/--data_dir/--ckpt_path(--hf)/--limit`、env 展开、epoch 数值排序、CSV 合并 | ✅ 我的独立指标第二意见与其一致；eval 入口 `--help`/合成 smoke 可跑；P2-4/P2-5 |
| **T4** `look2hear/datas/*` + 2 个新工具 | eval/train/旧 dataset 的 mono `ndim` 守卫（F5）；`make_synthetic_data.py`；`realssa_to_falcon.py`（tarfile 流式、4ch→mono、meta/JSON） | ✅ guard 正确（mono 1-D、stereo 通道均值）；新脚本 10 tests passed；**T4 改动也覆盖 `check_avspeech_dataset.py`（旧副本，仍在 datas 作用域内）** |
| **T5** `models/Falcon.py` / `models/__init__.py` / `videomodels/frcnn_videomodel.py` + 导出脚本 | `_mdcl_loss = _mdcl_loss_hard` 别名；`__all__` 修正；`FRCNNVideoModel.train()` `return self`；`extract_video_backbone.py` | ✅ 纯别名/返回值/导出名改动，零数值变化（§3.2）；导出 137 keys 与 `update_frcnn_parameter` 兼容 |

**审查中确认的接口契约**：
- `look2hear.metrics` 新增 8 个导出（`si_sdr/si_sdr_improvement/sdr/sdr_improvement/pesq_metric/stoi_metric` +
  2 常量），旧 3 个 tracker 名称保留 ⇒ 旧调用方不会 ImportError；
- `audio_visual_test.py` 顶层只定义 `parser`，`parse_args()` 在 `__main__` 内 ⇒ `import audio_visual_test` 安全；
- `from look2hear.models import *` / `models.__all__` 与 `models.get()` 均可用（QA 套件断言）；
- 改动范围：仅有 T1–T5 约定的 11 个文件（+ 新增 `metrics/core.py`、`tools/repro/*`），**无越界源码改动**（QA 套件断言）。

## 5. 指标第二意见（不依赖 T3 的测试）

合成信号（固定 seed）：`target=s1`、`interference=s2`、`mixture=s1+s2`、`estimate=s1+0.3·s2+noise`。
- **独立参考**：SDR 用 `fast_bss_eval.sdr_pit_loss(..., filter_length=512)` 直接计算；SI-SDR 用 numpy float64
  **解析式**（零均值、投影 α 公式）计算；SDRi/SI-SDRi 用差值。
- **T3 实现**：`look2hear.metrics.core.sdr/sdr_improvement/si_sdr/si_sdr_improvement` 与上述参考逐项一致
  （SDR ≤1e-4 dB、SI-SDR ≤5e-3 dB）；`ALLMetricsTracker` 写入 CSV 的逐句行/avg/std 行同号，
  `avg==逐句值`、单句 `std==0`；`est=mix` 时 SI-SDRi/SDRi 提升 **恰为 0**（与 Lead integration-smoke §5 一致）。
- 结论：T3 的"正号/越高越好"语义与 `fast_bss_eval`/解析定义一致；此前"−SNR 当 SDR、行/avg 反号、PESQ 反参"
  三类问题已消除。注：T3 的合成 smoke 指标为负值（随机 target/mix 不相关），只证明"链路通"，不作为指标正确性证据。

## 6. 边界与未覆盖

1. **未在 GPU 上跑完整 QA 数值指纹**（CPU 已足够；fingerprint 与 device 无关，仅 T3 eval smoke 在 GPU 跑通）。
2. **未用真实 RealSSA/LRS2 数据**：T3/T4 用合成数据；指标"合理量级"的最终确认需 Phase 2 真实数据。
3. **DDP 多卡训练**未在本 QA 复跑（peer review `REVIEW-training-stack.md` 已做 2-GPU toy DDP；Lead integration smoke 为单卡）。
4. **T6 LRS2 下载仍在后台**（16:45 时 9.47 GB / 16.47 GB ≈57.5%），不影响本次 QA 结论。
5. **并发运行注意**：多 agent 同时跑全量 pytest 会因 GPU/DDP 争用变慢（~65s→146s+），且实现者写文件期间
   会出现瞬时失败（我在 16:38 见过 4 个 T3 用例瞬时失败，文件稳定后 20 passed）。终检必须独占运行。

## 7. 复跑命令（QA 终检）

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon

# 1) QA 独立回归（16 条：基线/参数/4 条数值指纹/跨分支不变量/import/Studio/指标二次意见/守卫）
$ENV/bin/python -m pytest tools/repro/tests/test_phase1_regression.py -q

# 2) 全量新测试（70 条）
$ENV/bin/python -m pytest tools/repro/tests -q

# 3) Studio 回归（35 passed / 2 skipped）
$ENV/bin/python -m pytest studio/tests -q

# 4) 单独重算官方权重指纹（与 baseline 对比）
$ENV/bin/python - <<'PY'
import copy, hashlib, sys, torch, yaml
from pathlib import Path
REPO=Path('/root/code/github_repos/Falcon-fork'); sys.path.insert(0,str(REPO))
import look2hear.models as models
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel
sha=lambda t: hashlib.sha256(t.detach().to('cpu').contiguous().numpy().tobytes()).hexdigest()
cfg=yaml.safe_load((REPO/'configs/Falcon.yml').read_text()); ac=copy.deepcopy(cfg['audionet']['audionet_config'])
ck=torch.load('/root/.cache/falcon_weights/falcon.pt',map_location='cpu',weights_only=False)['state_dict']
a={k[12:]:v for k,v in ck.items() if k.startswith('audio_model.')}
v={k[12:]:v for k,v in ck.items() if k.startswith('video_model.')}
torch.manual_seed(0); audio=torch.randn(1,32000); mouth=torch.randn(1,512,50)
m=models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(ac)); m.load_state_dict(a,strict=True).eval()
with torch.no_grad(): y=m(audio,mouth)
torch.manual_seed(1); vin=torch.randn(1,1,50,88,88)
vm=FRCNNVideoModel(pretrain=None); vm.load_state_dict(v,strict=True)
with torch.no_grad(): vy=vm(vin)
print('eval', sha(y)); print('video', sha(vy))
PY
# 期望：eval ad0f20a0…f40757 / video 2afcea08…dd40
```

## 8. 变更日志

| 日期 | 轮次 | 变更 |
|---|---|---|
| 2026-10-21 | V1（phase1 实现轮，本报告初版） | 建立 16 条独立 QA 回归；完成 T1–T5 diff 审查（P0=0/P1=0/P2=6）；4 条官方权重指纹与 baseline 全部对齐；跨 `/root/wt-main` 数值不变量通过；复跑 70/35+2 全绿；定位 baseline video 指纹约定（T5 报告遗留问题的答案） |

## 9. 终检（基于最新 diff sha，2026-10-21 17:0x）

> 触发：源码更新（train-sys 修 T1 P2：GPU spec 语义/CSVLogger 目录/`--conf_dir` 默认；T5 仅改文档；T2/T3/T4 源码未再改）。
> **本轮 diff sha256：`d854fa7d780d28ba700f9c8abd00f2304aa4ff2606016113fbb0fff2b8d96c1b`**（`git diff main`；
> 11 files, +951/−419；`audio_train.py` = **409 行 / 18,800 B**）。
> 说明：该 hash **与上一轮 QA 锚点相同** —— `audio_train.py` 终版（16:48:47）在上一轮锚点（16:50）之前已落盘；
> 但上一份 `QA-regression.log` 的**首个快照/首次 QA 套件跑在 16:48 修改之前**，因此本轮在最终代码上**全量重跑**并覆盖日志。

### 9.1 终检结果（全部实跑）

| # | 检查 | 结果 |
|---|---|---|
| F1 | QA 独立回归（16 条） | **16 passed**（46.4 s） |
| F2 | 全量 `tools/repro/tests` | **77 passed / 0 failed**（57.5 s；T1 新增 7 条 GPU 语义用例，13→20） |
| F3 | Studio 回归 | **35 passed, 2 skipped** |
| F4 | import 完整性 | **IMPORTS_OK (12 modules)** |
| F5 | 官方权重 4 条指纹 vs baseline | **全部逐位一致**：eval `ad0f20a0…f40757`(sum 0.3195922)、train est `5f01a596…e7545a`、sep `8a8e5c2a…bddc16`、**video `2afcea08…dd40`(sum 7832.2563，training=True 口径)** |
| F6 | 参数 | **537,697 / 11,186,688 / 11,724,385** |
| F7 | 改动范围 + py_compile | 仅 T1–T5 约定 11 个 .py；全部编译通过；无越界源码改动 |
| F8 | 跨 `/root/wt-main` 数值不变量 | PASS（QA 套件子进程逐项比对） |

### 9.2 T1 GPU spec 改动的独立复核（Lead 重点）

- **语义已修正**（`audio_train.py:51-90` `_normalize_gpus_spec`）：`list/tuple/range/"0,1"` = **GPU 索引**；
  `int n>0` = **数量**；`0` = CPU（legacy）；`-1` = 全部可见（legacy）；`None/""` = 默认单卡。
  实测矩阵 9/9 与文档一致（`[7]` 现在是"索引 7"而不是"7 张卡"，修复了旧的歧义）。
- **校验在 resolver**（`:93-161`）：空列表 / 重复索引 / 越界索引 → `ValueError`（实测 3/3 命中）；
  `configs/Falcon.yml` 的 10 卡列表在 2 卡机上 → `devices=[0,1]` + `DDPStrategy` + WARNING（实测）；
  `gpus=0→cpu`、`-1→[0,1]+DDP`、`2→devices=2+DDP`、`None→单卡` 均符合预期。
- **CSVLogger 单层目录**（`:186-190`）：`CSVLogger(save_dir=<…/checkpoint>, name=<exp>)` ⇒
  `log_dir = …/checkpoint/<exp>/version_0`（实测），不再出现 `<exp>/<exp>/version_0` 重复目录。
- **`--conf_dir` 默认** = `configs/Falcon.yml`（`:34-40`），与 `REPRODUCTION_PLAN` F3/F4 一致。
- **数值/训练逻辑不变（证据）**：① 4 条官方权重指纹与 baseline 全部逐位一致（F5）；
  ② 模型/视频模型/优化器/损失/system 构造语句与 `main` **逐行等值**（QA 脚本比对，见 QA-regression.log §8b）；
  ③ 与 main 的差异仅：删除未使用的 `ReduceLROnPlateau` import、**scheduler 行仅空白格式差**
  （`sche_name"])(` → `sche_name"] )(`，调用语义相同）、exp_dir/conf dump 前移（F9 预期）。
  ⇒ **T1 本轮改动不影响"数值/训练逻辑不变"结论；上次 review 的 GPU 语义 P2 已闭环。**

### 9.3 残留问题（终检后）

- **P0 = 0、P1 = 0。**
- **P2（5 条，均非阻塞）**：
  1. T1 清理项：`audio_train.py:2` 的 `import sys` 仍未使用；scheduler 行有空白格式瑕疵（`sche_name"] )(`）——纯 cosmetic；
  2. T2 日志 dict 移除 `"epoch"` 字段（改由 PL epoch 列提供；已在 T2 报告说明）；
  3. T3 legacy `SPlitMetricsTracker`（splitwrapper.py）仍旧负号/SNR 语义且仍被 `metrics.__all__` 导出（未被评测链路使用，建议 Phase 2 统一/标 deprecated）；
  4. `requirements.txt` 仍列 `torch-mir-eval`（wrapper.py 已不再 import，保留环境兼容即可）；
  5. 流程提醒：多 agent 并发跑全量 pytest 会变慢（~65→146s）且写入期间有瞬时失败 ⇒ 终检需独占运行（本轮已独占）。
  - **已闭环**：P2-3（T5 报告 video 指纹口径）——T5-report（16:54 更新）已补 `modes`/`mode_notes`，
    明确 baseline video 指纹为 `training=True`、未调用 `.eval()` 的口径，与本 QA 结论一致。

**终检结论：PASS。Phase-1 Gate G1 的 QA 部分在最新源码上全部通过；本轮锚定 `git diff main` sha256 `d854fa7d…`。**

## 10. 终检 v2（config 白名单修复 + LRS2-2Mix 真实数据 sanity，2026-10-21 17:2x）

> 触发：① 经 Lead 授权，train-sys 给 `configs/Falcon.yml` 加 **2 行注释**（logger/gpus 语义），导致 QA 的
> `test_diff_scope_only_allowed_files` 白名单缺该文件而失败；② Lead 用官方权重在 **LRS2-2Mix 真实数据**上
> 跑通评测，邀请纳入端到端 sanity。本轮**独占重跑**全部 QA。
> **新 diff sha256：`49101a329fba3d158a3eb0f2160b476aa969c0c31c1c11cf041dc2e4148514bf`**
> （`git diff main`；12 files, +958/−419；`audio_train.py`=409 行/19,107 B，17:08 版）。
> 上一轮锚点 `d854fa7d…` 仅缺 `configs/Falcon.yml` 与 audio_train 的负整数 GPU 校验更新。

### 10.1 白名单修复（P0 级 QA 自身缺陷 → 已闭环）

- `ALLOWED_TRACKED` 增加 `configs/Falcon.yml`，并新增独立断言
  `test_config_tracked_changes_are_comment_only()`：
  1. `yaml.safe_load(git show main:configs/Falcon.yml) == yaml.safe_load(工作区)` ⇒ **键值完全一致**（9 键 = 9 键）；
  2. `git diff` 的新增/删除行必须全部为空行或 `#` 注释 ⇒ **仅 2 行注释**（见 QA-regression.log §7b 原文）。
  这样既接受授权改动，又防止未来把配置键值改动混进 Phase-1。

### 10.2 终检 v2 结果（全绿）

| # | 检查 | 结果 |
|---|---|---|
| V1 | QA 独立套件（**21 条**：原 16 + config 注释守卫 + T1 GPU spec + LRS2×3） | **21 passed**（66.7 s） |
| V2 | 全量 `tools/repro/tests` | **82 passed / 0 failed**（85.3 s）= 原 77 + QA 新增 5 |
| V3 | Studio | **35 passed, 2 skipped** |
| V4 | import | **IMPORTS_OK (12 modules)** |
| V5 | 官方权重 4 指纹 vs baseline | **全部逐位一致**（eval `ad0f20a0…` / est `5f01a596…` / sep `8a8e5c2a…` / video `2afcea08…`） |
| V6 | 范围/语法守卫 | 仅约定 12 个文件（含授权的 `configs/Falcon.yml`）；py_compile OK；无越界 |
| V7 | 跨 `/root/wt-main` 不变量 | PASS |
| V8 | T1 GPU spec 复核（17:08 版） | 语义矩阵 9/9；`-2/-5` 等非 `-1` 负整数 → `ValueError`；空表/重复/越界 → `ValueError`；10 卡配置→`[0,1]+DDP`；CSVLogger 单层目录；**模型/视频/优化器/损失/system 构造语句与 main 逐行等值** |

### 10.3 LRS2-2Mix 真实数据端到端 sanity（新增，非仓库数据）

数据（Lead 提供，仅本机）：`/root/data/lrs2_2mix/lrs2_rebuild`（原始 LRS2）、
`/root/tmp/lrs2_eval_tt_full/tt`（3000 条 Falcon 格式：mix/target/mouths + JSON）、
`/root/tmp/lrs2_eval_s{1,2}/tt`（20 条目标约定对照）。全量 3000 条评测在 QA 时仍在后台运行（QA 不依赖其结果）。

QA 新增 3 条测试（数据缺失时自动 skip，不阻塞仓库内回归）：
1. **`test_lrs2_mix_equals_s1_plus_s2`**：独立验证混音定义——`mix == target(s1) + target(s2)`，
   3 个样本 **R² = 1.00000、系数 [1,1]**（PCM16 量化级误差）⇒ s1/s2 目录的 target 确实是两个源、且 `target=s1` 约定成立。
2. **`test_lrs2_eval_aggregate_matches_lead_evidence`**：解析 Lead 的 20 条评测日志，复算并断言：
   - `target=s1`: **SI-SNRi 7.911 / SDRi 8.547 / SI-SNR 13.601 / SDR 14.309 / PESQ 2.312 / STOI 0.923**；
   - `target=s2` 对照: SI-SNRi **4.603**；
   - `s1 − s2 > 2 dB` ⇒ **目标约定确认为 speaker1**，评测管线自洽（与论文 LRS2-2Mix 14.1/14.7 同量级，待全量确认）。
3. **`test_lrs2_eval_end_to_end_smoke`**：官方权重 + 真实数据、CPU、`audio_visual_test.py --limit 2` 端到端重跑
   （cwd=临时目录，不写仓库），断言：exit 0、`metrics_combined.csv` 含 2 条逐句 + avg/std、全部有限值、
   `SI-SNR>0 / SDR>3 / SI-SNRi>0 / SDRi>0`（实测 2 条 SI-SNRi 7.45、SDRi 8.37，与 20 条口径一致）。

### 10.4 终检 v2 残留问题

- **P0 = 0、P1 = 0。**
- **P2（5 条，非阻塞；与 §9.3 相同，P2-3 已闭环）**：① T1 清理项（`audio_train.py:2 import sys` 未用、
  scheduler 行空白格式 `sche_name"] )(`）；② T2 移除 `"epoch"` 日志字段（已说明）；③ T3 legacy
  `SPlitMetricsTracker` 仍旧符号语义（未使用）；④ `requirements.txt` 仍列 torch-mir-eval（兼容保留）；
  ⑤ 流程：并发全量 pytest 会变慢/瞬时红，终检需独占（本轮已独占）。
- 备注：仓库内 `Experiments/checkpoint/*/results/` 被 `.gitignore:20 results/` 忽略，评测产物不会污染 QA 范围守卫。

**终检 v2 结论：PASS。最新源码（`git diff main` sha256 `49101a32…`）上：QA 21/21、全量 82/82、Studio 35+2、4 指纹全等；
LRS2-2Mix 真实数据端到端可跑通且 target=s1 约定经独立验证。**
