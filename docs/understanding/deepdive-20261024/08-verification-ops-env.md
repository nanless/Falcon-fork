# 08 验证资产 / 环境 / 运维（深挖版）

> 组员：`verify-ops` · 任务：共享任务 `task-8` · 分支 `repro/phase2`（HEAD `2beec38`，1 个未提交改动）· 2026-10-24
> 环境：**只用** `/root/miniforge3/envs/falcon/bin/python`（conda `falcon`，Python 3.11.17，torch 2.8.0+cu128，2×A800）
> 范围：`docs/verification/**`（scripts / runs / phase1 / phase2 的报告·日志·JSON·CSV）、`AGENTS.md`、`conda-env-lock.txt|yml`、
> `tools/repro/tests`、`.github/**`、git 状态与历史。只读分析；唯一写入 = 本文件。
> 证据标记：`[代码]` 直接读源码/文件；`[验证]` 本轮实际运行命令确认；`[推断]` 由证据推断；`[文档]` 仅文档转述。
> 参考旧文档：`docs/understanding/08-ops-tests-docs.md`（768 行，R3 v2，分析对象 commit `7576854`）。

---

## 0. 一句话结论

**本机环境（conda falcon / torch 2.8.0+cu128 / 2×A800）完好、可自洽复跑，关键指标数字也基本都能在 CSV/JSON/日志里对上；但"已复现"结论的证据硬度分层极不均匀，且大量一手证据并未入库。** 具体地：`docs/verification` 在磁盘上 226 个文件、git 只跟踪 129 个（`*.log` 被根 `.gitignore` 全量忽略，`runs/` 45 个日志 **0 个入库**）；RealSSA-Sim `7.774/8.479`、自训 `8.858/9.270` 的逐句 CSV 都在被 gitignore 的 `Experiments/` 下，Phase-1 全量 `7.03/7.71` 的 CSV 更已被同路径的 RealSSA 评测**覆盖**；`docs/verification/phase2/README.md`、`REPRODUCTION_PLAN.md`、`DATA_ACCESS_GUIDE.md` 三处索引/状态落后实际 2–3 天（仍写 RealSSA 403、训练进行中）；`tools/repro/tests` 当前 **104 passed / 1 failed / 7 skipped**（QA 记录的 `111/1` 已过期，失败由 RealSSA 全量 CSV 里 3 个空 SDR 单元格触发）。**最硬的证据是**：权重 sha256、TOS 归档字节数与 rsync 退出码、P2-F 独立对拍 4.8e-8 dB、P2-A/P2-B 的入库 JSON/CSV；**最软的是**：8 卡/火山引擎方案（纯 runbook，零日志）、"训练继续中/GPU 被占"（已结束、GPU 已空）、以及被覆盖 CSV 支撑的两个全量 LRS2 结论。

---

## 1. 环境自检结果（命令 + 实际输出）

### 1.1 解释器 / torch / CUDA

```bash
$ /root/miniforge3/envs/falcon/bin/python -V
Python 3.11.17

$ PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -c \
  "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda, torch.backends.cudnn.version())"
torch 2.8.0+cu128 / cuda_available True / device_count 2 / cuda 12.8 / cudnn 91002
```

`[验证]` 与 `AGENTS.md §1/§2` 声明一致（Python 3.11.17、torch 2.8.0+cu128、2 卡）。

### 1.2 项目 import 自检

```bash
$ PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -c "
import sys; sys.path.insert(0,'.')
import look2hear.models, look2hear.system, look2hear.metrics
import audio_train, audio_visual_test
print('imports OK')"
timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated...
imports OK
```

`[验证]` 5 个目标 import 全部通过，唯一告警是 timm 的弃用 FutureWarning（非错误）。

```bash
$ ... -c "import pypesq; print(pypesq.__file__)"
/root/miniforge3/envs/falcon/lib/python3.11/site-packages/pypesq.py
$ ... -c "from importlib.metadata import version; print(version('typeguard'), version('pesq'), version('speechmos'))"
typeguard 2.13.3 / pesq 0.0.4 / speechmos 0.0.1.1
```

`[验证]` **pypesq 兼容层在环境里、但不在仓库、不在 lock 文件**（`grep -c pypesq conda-env-lock.txt = 0`）；重建环境必须手工创建（`AGENTS.md §5` 有 shim 代码）。注意 `typeguard` 2.13.3 **没有 `__version__` 属性**，用 `typeguard.__version__` 会 AttributeError——这是库自身行为，不是环境损坏。

### 1.3 回归测试

```bash
$ /root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q
35 passed, 2 skipped in 19.05s          # 37 collected（test_server 单跑 22 passed）

$ PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q
1 failed, 104 passed, 7 skipped, 43 warnings in 111.87s (112 collected)
```

`[验证]` Studio 侧与 `AGENTS.md §2`/旧文档完全一致（35/2）。**repro 侧与 QA 记录的 `111 passed / 1 skipped` 不一致**，原因已定位：

1. **1 failed** = `tools/repro/tests/test_phase2_pipeline.py::test_phase2_eval_csv_aggregation_consistent`
   （`test_phase2_pipeline.py:245`，`ValueError: could not convert string to float: ''`）。触发文件是
   `Experiments/checkpoint/Falcon/results/metrics_combined.csv`（RealSSA-Sim 全量结果，见 §2），其中 3 行
   `sdr/sdr_i/pesq` 为空串：
   - 第 859 行 `00003_02_41188.wav,,,-80.0,0.0,,0.0`
   - 第 3454 行 `00012_02_84551.wav,,,-80.0,0.0,,0.0`
   - 第 8820 行 `50016_74250.wav,,,-80.0,0.0,,0.0`

   对应评测日志 `/root/tmp/realssa_sim_eval_official.log` 开头即有 `SDR computation failed (torch.linalg.solve ...
   input matrix is singular.); recording nan` 与 `PESQ computation failed (b'No utterances detected')`——即 3/9169 条
   退化样本被写成空串，测试的 `float(r[col])` 没有容错。**这是"新增证据数据破坏既有回归"，不是代码回归。**
2. **7 skipped** 全部在 `tools/repro/tests/test_phase1_regression.py`（`:512/:521/:530/:547/:564/:576/:600`），
   skip 理由都是"当前无被改动的 Python 文件 / T1、T3、T5 尚未落地改动 / `configs/Falcon.yml` 未被改动"。
   Phase-1 改动**已 commit**（`repro/phase1 = ac91800`、HEAD 即 main），所以这批"diff 范围守卫"从"运行并 PASS"
   退化为"条件跳过"——**回归覆盖在提交后自动失效**（见 §4 风险 R7）。

### 1.4 Node 测试（Studio 前端）

```bash
$ for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done
check_i18n.mjs rc=0 · check_mouth_tracking.mjs rc=0 · check_playback_resampler.mjs rc=0 · check_spectrum.mjs rc=0
check_capture_dsp.cjs rc=0 · check_tracking_ui.cjs rc=0 · capture-worklet.test.cjs rc=0 · video-worker.test.cjs rc=0
```

`[验证]` 8/8 rc=0，与旧文档"8 个 Node 测试全过"一致。

### 1.5 磁盘 / GPU / 环境锁

```bash
$ df -h / /root
overlay                     20G  5.9G   14G  31% /
fs_vepfs-cnbjc90fab4fcf40  164T  151T   14T  92% /root

$ nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
0, NVIDIA A800-SXM4-80GB, 81920 MiB, 2 MiB, 0 %, 535.129.03
1, NVIDIA A800-SXM4-80GB, 81920 MiB, 2 MiB, 0 %, 535.129.03
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]      # 空：无计算进程
```

`[验证]` 根分区只剩 14 GB（`/` 20 GB），大产物必须写 `/root`（vepfs）。**GPU 当前完全空闲（2 MiB / 0%）**——
与 Phase 2 报告中"GPU0 55.6 GB 被外部进程占用、无法双卡"（`P2D-gpu-snapshot.txt` 记录 2026-10-21T19:26:41Z
外部 PID 2514085 占 55.6/71.9 GB）是**历史快照**，现在该进程已退出（见 §4 R5）。

```bash
$ wc -l docs/verification/conda-env-lock.txt docs/verification/conda-env.yml
115 docs/verification/conda-env-lock.txt
 94 docs/verification/conda-env.yml
```

`[验证]` lock 实为 **115 行**（Phase 2 加入 speechmos/onnxruntime/protobuf/flatbuffers 后），而 `AGENTS.md §2`
仍写"pip freeze，111 行"——**文档过期**（旧 README §1 已正确写 115）。lock 中关键包：
`torch @ file:///root/wheelhouse/torch-2.8.0+cu128-...whl`、`pytorch-lightning @ file:///root/wheelhouse/...`、
`numpy==2.2.6`、`typeguard==2.13.3`、`speechmos==0.0.1.1`、`onnxruntime==1.31.0`、`pesq==0.0.4`、`fastapi==0.156.8`、`pillow==13.0.0`。

**环境可重建性风险（重要）**：`conda-env-lock.txt` 里有 **18 个包**是 `@ file:///root/wheelhouse/<绝对路径>.whl`
（torch/torchaudio/torchvision/pytorch-lightning + 14 个 nvidia-* CUDA 库）。在**新机器**上直接
`pip install -r conda-env-lock.txt` 会因 `/root/wheelhouse` 不存在而失败；`conda-env.yml` 的 pip 段虽写
`torch==2.8.0+cu128`，但该本地版本不在 PyPI，仍需先加 `--index-url https://download.pytorch.org/whl/cu128`。
即：**lock 文件只是"本机快照"，不是可移植锁**；`AGENTS.md §5` 只提示"需先装好 torch cu128 轮子"，
未提示 18 个 `file://` 绝对路径与 `pypesq` shim 缺失。`[代码+验证]`

---

## 2. 关键数字出处表（逐条抽验）

> 复算方法：对 CSV 的 `avg` 行独立重算逐句均值（有限值集合），对日志取 `Final metrics` 段；所有数字均在本机复现。
> 表中"CSV 行"为文件内 1-based 行号（含表头）。

| # | 结论 / 数字 | 一手证据（文件:行） | 证据类型 | 本轮复算 |
|---|---|---|---|---|
| N1 | **RealSSA-Sim 全量 9169 条：SI-SNRi 7.774 / SDRi 8.479**（论文 7.7/8.2） | `Experiments/checkpoint/Falcon/results/metrics_combined.csv:9171`（avg 行：`sdr 10.644734 / sdr_i 8.479395 / si-snr 9.980200 / si-snr_i 7.774307`）；`/root/tmp/realssa_sim_eval_official.log`（`Final metrics ... exit=0`） | 本机实测 | ✅ avg=逐句均值；SDR 类 n=9166（3 条空值），SI-SNR 类 n=9169；去掉 3 个 0 值后 si-snr_i=7.7769（+0.003，不影响结论） |
| N2 | **RealSSA-Real DNSMOS（54/54 段）：SIG 3.2209 / BAK 3.6921 / OVRL 2.7901 / P808 3.3285**；mix 基线 2.4176/1.7908/1.7599/2.6765 | `docs/verification/phase2/P2E-realssa-real-full3.log`（`N=54 SIG=3.2209 ... exit=0`，且逐段 54 行）；`/root/data/falcon_eval/realssa_real_mean_full3/summary.json`（`segments_ok: 54`） | 本机实测 | ✅ 日志均值与 summary.json 一致；前两次尝试 `P2E-realssa-real-full{,2}.log` 以 `exit=1` 失败，最终数字来自第 3 次 |
| N3 | **LRS2-2Mix 官方权重全量 3000：SI-SNRi 7.029 / SDRi 7.713**（跨语料） | `/root/tmp/lrs2_eval_tt_full.log`（`Final metrics si-snr_i 7.029101 / sdr_i 7.712925`）；`docs/REPRODUCTION_PHASE1_REPORT.md` §3.5 | 本机实测 | ⚠️ 数字可对日志，但**对应 CSV 已被 N1 的 RealSSA 评测覆盖**（同路径 `Experiments/checkpoint/Falcon/results/metrics_combined.csv`） |
| N4 | **LRS2-2Mix 300 条子集：small 6.916 / 7.492**（P2-D eval 文件即此） | `docs/verification/phase2/P2D-eval-falcon.pt.csv:302`（avg `sdr_i 7.491614 / si-snr_i 6.916240`）；`P2B-summary-b300.json` 同值 | 本机实测（已入库） | ✅ 注意：**`P2D-eval-falcon.pt.csv` 支撑的是 6.916/7.492（300 条），不是 7.03/7.71（3000 条）**——任务书里"7.03/7.71 = §3.5 + 该 CSV"的对应关系需修正 |
| N5 | **自训 LRS2（无 near，ep86）：SI-SNRi 8.618 / SDRi 9.042** | `/root/tmp/p2d_eval_epoch86_full.log`（`Final metrics 8.617919 / 9.041506`，3000 条，`exit=0`） | 本机实测 | ⚠️ 对应 CSV 已被 N6 覆盖，仅存日志（且日志在仓库外） |
| N6 | **contrastive 微调（ep124）：SI-SNRi 8.858 / SDRi 9.270**（PESQ 2.576 / STOI 0.943） | `Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv:3002`（avg `8.858191910560924 / 9.269574165939565`）；`/root/tmp/p2d_eval_contrastive_epoch124.log` | 本机实测 | ✅ 日志 final 与 CSV avg 完全一致；3000 条、整句（limit=all） |
| N7 | 自训 val/test 曲线：best val **13.390929 @ep124**，test 13.7618 @ep120；无 near 训练 100 epoch 完成，ep99 val_loss −13.20141 / test −13.330 | `/root/tmp/p2d_near_train/Experiments/checkpoint/LRS2_2Mix_near_s2/version_0/metrics.csv`（1560 行，30 个 val 行、3 个 test 行）；`docs/verification/phase2/P2D-train.log` 末尾 `Finished Training` | 本机实测 | ✅ 但**报告写"epoch 86 最佳 val 13.118、训练继续中"已过期**：训练 2026-10-22 19:54 结束（100/100 epoch），ep99 val 13.201 更高；评测用的仍是 ep86 ckpt |
| N8 | **RealSSA-Sim 归档 242 文件 / 521,532,645,652 B（≈521.5 GB）**；`bad=[]`；rsync 26m48s @309 MB/s、`rc=0 src_removed=True` | TOS `realssa_sim/_MANIFEST.json`（242 files / 521532645652 / bad=[]）；`/root/group-shared/.../_download/realssa_console.log`（`521,532,645,847 100% 309.26MB/s 0:26:48 (xfr#243, to-chk=0/247)`、`rsync rc=0 src_removed=True`）；`_archive_done.json` 两条 `ok:true` | 本机实测 + 外部存储 | ✅ TOS 实盘 243 文件、总 521,532,645,847 B（=242 数据集文件 521,532,645,652 + `_MANIFEST.json` 195 B）；tr 202 / cv 19 / tt 19 与 HF 一致 |
| N9 | **RealSSA-Real 12.84 GB / 423 文件** | TOS `realssa_real/` 实盘 423 文件、12,843,107,273 B；`docs/REPRODUCTION_PHASE2_REPORT.md` §5.4 | 本机实测 + 外部存储 | ⚠️ 与 HF API / 旧 README 的 **422 文件 / 12,842,268,608 B** 差 1 文件、838,665 B，未解释（见 §4 R11） |
| N10 | **官方权重 sha256**：`falcon.pt` = `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`（47,156,343 B）、`falcon-big.pt` = `ce2a92a3188e8069e979f23aaec21cb6d439170aaf4002b4eaa234f0486e037e`（49,645,087 B） | 本机 `sha256sum`；与 `docs/verification/phase1/baseline-fingerprint.json:8`、`docs/verification/phase2/QA-report.md:84`、`docs/verification/model-core.md:161` 完全一致 | 本机实测 | ✅ 但注意：`/root/.cache/falcon_weights/*` 现为**符号链接**（2026-10-23 06:18 建），指向 `/root/group-shared/voiceprint/models/a800_2gpus/cache/falcon_weights/`；该挂载不可用则权重与依赖权重的测试全部失效 |
| N11 | **P2-F 独立对拍最大差 4.82e-8 dB**（vs `core.py`）、**1.84e-7 dB**（vs Phase-1 CSV） | `docs/verification/phase2/P2F-results.json`（`4.823487742555699e-08`、`1.8408684887560867e-07`）；`P2F-protocol-check.log:49`（`max|my − core.py| = 4.823e-08 dB ; max|my − Phase-1 CSV| = 1.841e-07 dB`） | 本机实测（JSON 已入库，日志未入库） | ✅ 数字自洽；⚠️ 脚本 `tools/repro/phase2/protocol_check.py:49` 硬编码读 `Experiments/checkpoint/Falcon/results/metrics_combined.csv`，**该 CSV 现在是 RealSSA 的 9169 行**，故"vs Phase-1 CSV"分支已不可原样重跑（会读到错误 CSV） |
| N12 | **P2-A 14 变体：基线 6.9547/7.6224，最大正收益 +0.3405 dB** | `docs/verification/phase2/P2A-small-200.json`（`models.small.variants.mouth_s1_88.si_snr_i.mean=6.954677`、`sdr512_i.mean=7.6224`；`snr_i_plain.mean=7.2952` → Δ+0.3405） | 本机实测（已入库） | ✅ 复算：14 个 `variants` 相对基线**全部 ≤0**（最大 −0.000），+0.34 来自基线块里的 **plain-SNR（尺度相关、非标准指标）**；标准口径下最大正收益是 4×0.5s 分段 **+0.282**，其余为负（nonorm −0.451、crop1s −0.478、zero mouth −4.80、shift±5 −11.5/−12.8、target=s2 −23.95）。报告"±0.35 dB"成立 |
| N13 | **P2-B**：small 6.916/7.492、big 7.887/8.403、small windowed 5.004/5.762、big windowed 5.569/6.270 | `docs/verification/phase2/P2B-b300-*.csv`（各 300 行+avg/std）、`P2B-summary-b300*.json` | 本机实测（已入库） | ✅ 逐项对上；big−small = +0.971/+0.911，窗口化 = −1.912/−2.318。⚠️ `P2B-full-big-whole-real.csv` 是 **0 字节已入库文件**（报告 §6 已声明"主动终止留下的空文件，可忽略"） |
| N14 | epoch 0：SI-SNRi 1.5169 / SDRi 1.8298（tt 300 条子集） | `docs/verification/phase2/P2D-eval-epoch_0.csv:302`（avg `sdr_i 1.829785 / si-snr_i 1.516857`） | 本机实测（已入库） | ✅ 与 `REPRODUCTION_PLAN.md` "≈1.52" 一致 |
| N15 | DNSMOS ONNX 模型 4 个 sha256 | `docs/verification/phase2/P2E-dnsmos-synthetic.json` vs 本机 `sha256sum site-packages/speechmos/dnsmos_models/*.onnx` | 本机实测（已入库） | ✅ 4/4 完全一致（`sig_bak_ovr` 269fbe…f6edd、`model_v8` 924648…f6e91、`sig` d2fbdb…549d7、`bak_ovr` 5f335c…d1308） |
| N16 | LRS2-2Mix 转换数据 28,000 条 / 23.36 GiB | `/root/data/falcon_lrs2_2mix`（71,526 文件、25,078,171,954 B = 23.36 GiB） | 本机实测 | ✅ 与 P2-C/报告一致 |
| N17 | 论文配方数据 26 GB（双向 target，tr 40000/cv 10000/tt 6000） | `/root/data/falcon_lrs2_2mix_paper`（19,444 文件、26,929,975,969 B = 25.08 GiB / 26.93 GB） | 本机实测 | ✅ 存在；⚠️ **但没有任何该数据上的训练结果**（见 §3 L3） |

**抽验中发现的数字级不一致（需报告方修正/声明）**

1. **同一数据集三个字节数**：HF API/README 写 `521,532,631,040 B`；`_MANIFEST.json` 写 `521,532,645,652 B`（=242 文件实盘和）；
   rsync 控制台写 `521,532,645,847 B`（=242 文件 + `_MANIFEST.json` 195 B）。三个都对，但报告若混用会显得对不上，**应注明口径**。
2. **RealSSA-Real 文件数**：本机/TOS 423 个 vs HF API 422 个（旧 `DATA_ACCESS_GUIDE.md` 采信 422）。
3. **训练时长**：报告"100 epoch 预计 ~30 h（~18 min/epoch）"，实际 2026-10-21 19:26:41Z → 2026-10-22 19:54 ≈ **24 h27m（~14.7 min/epoch）**；估计值未被回填为实测。
4. **`P2D-eval-falcon.pt.csv` 口径**：300 条子集（6.916/7.492），**不是** 3000 条全量（7.03/7.71）；任务书中的对应关系需修正。

---

## 3. 证据可信度分级表（重点交付）

> 分级定义：
> **L1 = 硬证据**（可离线复跑 + 证据文件已入库 git）；**L2 = 硬但未入库**（本机实测、证据存在，但 git 跟踪不到或已丢失）；
> **L3 = 转述/仅报告**（只有文字结论，无一手日志/数据）；**L4 = 估计/未验证**（明确是外推、估计或从未执行）。
> "是否足够"按"第三方 clone 仓库后能否复核"判定。

| 结论 | 证据文件 | 证据类型 | 级别 | 是否足够（第三方复核） |
|---|---|---|---|---|
| 权重 sha256 / 参数量 / 4 条数值指纹不变 | `baseline-fingerprint.json`、`QA-report.md:84`、`model-core.md:161` + `verify_real_weights.py` | 本机实测 | **L1** | ✅ 足够（脚本 + 基线 JSON 均入库；但需权重与 18 个 `file://` 轮子） |
| P2-F 指标对拍 4.8e-8 / 1.8e-7 dB | `P2F-results.json`（入库）+ `P2F-protocol-check.log`（未入库） | 本机实测 | **L1（对 core.py 部分）/ L2（vs Phase-1 CSV 部分）** | ⚠️ 部分：JSON 在库，但脚本读的 Phase-1 CSV 已被覆盖 |
| P2-A 14 变体 ≤0.34 dB | `P2A-small-200.json`、`P2A-anchor-first20.json`（均入库）+ 脚本 `gap_analysis.py` | 本机实测 | **L1** | ✅ 足够（需 LRS2 eval 数据，见 R8） |
| P2-B small/big/窗口化/视觉消融 | `P2B-b300-*.csv`、`P2B-summary-b300*.json`（均入库） | 本机实测 | **L1** | ✅ 足够（同上需数据） |
| RealSSA-Sim 7.774/8.479（9169 条） | CSV 在 `Experiments/…/metrics_combined.csv`（**gitignore**）、日志 `/root/tmp/…`（**仓库外**） | 本机实测 | **L2** | ❌ 不足：repo 内只有报告数字；复跑需 40.1 GB tt 原始 shard（本机在 `/root/data/RealSSA-Sim/tt`）或从 TOS 冷端恢复 |
| RealSSA-Real DNSMOS 3.3285 等 | `P2E-realssa-real-full3.log`（**未入库**）、`/root/data/falcon_eval/…/summary.json`（**仓库外**） | 本机实测 | **L2** | ❌ 不足：报告表格无入库一手件；复跑需 RealSSA-Real 数据（本机 `/root/data/RealSSA-Real` 仍在） |
| 官方权重 LRS2 全量 7.03/7.71 | Phase-1 报告 §3.5 + `/root/tmp/lrs2_eval_tt_full.log`（**仓库外**）；原 CSV **已被覆盖** | 本机实测 | **L2（偏弱）** | ❌ 不足：无逐句 CSV，仅报告与一条仓库外日志 |
| 自训 8.618/9.042（ep86） | `/root/tmp/p2d_eval_epoch86_full.log`（**仓库外**）；CSV 已被覆盖 | 本机实测 | **L2（偏弱）** | ❌ 不足 |
| contrastive 8.858/9.270（ep124） | `Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv`（**gitignore**）+ `/root/tmp/…` 日志 | 本机实测 | **L2** | ❌ 不足（但 CSV 仍在磁盘，可人工取出） |
| 训练完成（100 epoch，best val 13.201/13.391） | `/root/tmp/p2d_train|p2d_near_train/…/metrics.csv` + `P2D-train.log`（**未入库**） | 本机实测 | **L2** | ❌ 不足；报告里"训练继续中/最佳 13.118"更是**过期转述** |
| RealSSA 下载/归档 521.5 GB、242 文件、`ok:true` | TOS 实盘 + `_MANIFEST.json` + `_archive_done.json` + `realssa_console.log`（**均不在仓库**） | 本机实测 + 外部存储 | **L2** | ⚠️ 需 TOS 访问权；字节数/文件数可核对，但归档日志不在库 |
| DNSMOS ONNX sha256 | `P2E-dnsmos-synthetic.json`（入库）+ 本机复算 | 本机实测 | **L1** | ✅ 足够（speechmos wheel 版本已 pin） |
| Studio 35/2、Node 8/8、5 个目标 import | 本轮实测（10/10 全量 import 见旧 `QA-report`） | 本机实测 | **L1** | ✅ 足够（命令在 README/AGENTS 里） |
| tools/repro 回归全绿（111 passed/1 skipped） | `QA-report.md:16/60`、`REGRESSION-docs.md:7/20` | 转述（当时的日志未入库） | **L3（已过期）** | ❌ 与当前实况不符（104/1/7），见 §1.3 |
| LRS2 论文配方 8 卡 / 火山引擎训练 | `docs/PHASE2_VOLCANO_LRS2_TRAINING.md`（runbook）、`configs/LRS2_2Mix_Paper_8gpu.yml`、`run_lrs2_paper_volcano_8gpu.sh` | 计划/文档 | **L3** | ❌ 零执行日志、零结果；`grep -rln "火山|volcano|8gpu" docs/verification/` = 0 |
| 论文配方数据 26 GB 已构建（双向 target、40k/10k/6k） | `/root/data/falcon_lrs2_2mix_paper`（本机实盘） | 本机实测（数据）+ 报告 | **L2（数据）/ L3（结果）** | ⚠️ 数据存在但无训练/评测结果 |
| big 全量 ≈8.0/8.5 | `P2B-report.md:85`（明确"仅估计，未跑全量"） | 估计 | **L4** | ⚠️ 已声明为估计，可接受；引用时不得写成实测 |
| "GPU0 55.6 GB 被占、无法双卡" | `P2D-gpu-snapshot.txt`（2026-10-21 快照） | 历史快照 | **L4（已过期）** | ⚠️ 现在 GPU 全空（§1.5），不能再作为限制理由 |
| RealSSA-Sim "仍 403、无任何指标" | `P2E-report.md`、`README.md §1`、`PLAN`、`GUIDE` | 过期转述 | **L4（已推翻）** | ❌ 与 §2 N1/N2 矛盾（数据已批、已评测、已归档） |

**结论：哪些"已复现"最硬、哪些最软**

- 最硬（L1）：权重指纹/参数量、指标实现对拍（对 `core.py`）、P2-A/P2-B 的变体矩阵、DNSMOS 模型哈希、环境自检、Studio/Node 回归。
- 中等（L2）：**RealSSA-Sim 7.774/8.479**、**RealSSA-Real DNSMOS**、自训 8.858/9.270——数字本身经复算无误，但"证据在库外"，
  且 RealSSA-Sim 的**评测协议与论文是否同口径未被证明**（见 §4 R4）。
- 最软（L3/L4）：8 卡/火山方案、训练进度类描述（"继续中""GPU 被占"）、以及一切引用 `P2E-report.md`/`README.md`/`GUIDE`
  的"RealSSA 403"表述。

---

## 4. 证据缺口与风险（按严重度）

### R1（P0，证据完整性）**所有 `*.log` 被 `.gitignore` 忽略，一手执行日志几乎全部未入库**

- 根 `.gitignore` 第 11 行是 `*.log`，导致 `docs/verification` 下 **226 个文件只有 129 个被 git 跟踪**；
  `git status --ignored docs/` 列出 **58 条** ignored 条目；`docs/verification/runs/` 45 个日志 **0 个入库**。
- Phase 2 目录 64 个文件中 35 个入库，**29 个未入库全是日志**：`P2D-train.log`、`P2D-eval-*.log`、
  `P2E-realssa-real-full{,2,3}.log`、`P2E-realssa-sim-convert.log`、`P2F-protocol-check.log`、`QA-regression.log`、
  `logs/P2B-*.log`、`LRS2-paper-2gpu-smoke.log` 等。
- 后果：报告大量以"日志为证据"（如 P2-F、P2-E），但 **clone 仓库后这些证据不存在**；只有 `[文档]` 级转述。
- 建议：对证据日志 `git add -f`（或把 `docs/verification/**/*.log` 加白名单 `!docs/verification/**`），
  至少把每份报告的"最终 metrics 段"另存为 `*.txt`/`*.json`（`*.txt` 不在忽略列表，现有 5 个 txt 已入库）。

### R2（P0，证据被覆盖）**同名输出路径导致全量评测 CSV 被后一次评测覆盖**

- `audio_visual_test.py` 固定写 `<cwd>/Experiments/checkpoint/<exp_name>/results/metrics_combined.csv`
  （`audio_visual_test.py:386-396`、`_merge_results` 在 `:466`），`exp_name` 来自配置。
- 事实链：Phase-1 的 LRS2 全量 3000 条 CSV（7.03/7.71）→ 被 RealSSA-Sim 全量（exp_name=Falcon）**覆盖**；
  自训 ep86 的 LRS2 CSV（8.618/9.042）→ 被 ep124（8.858/9.270）**覆盖**。当前只剩：
  `Experiments/checkpoint/Falcon/results/metrics_combined.csv` = RealSSA-Sim 9169；
  `Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv` = ep124。
- 连带后果：`tools/repro/phase2/protocol_check.py:49` 的"vs Phase-1 CSV"对拍分支现在会读到 RealSSA CSV → **不可重跑**。
- 建议：评测脚本增加 `--out_dir/--tag` 或把结果写 `results/<split>_<ckpt>_<timestamp>/`；报告引用结果时写全路径+行号。

### R3（P0，文档/索引过期）**三份索引仍停留在"RealSSA 403、训练进行中"**

| 文件 | 过期表述 | 实际 |
|---|---|---|
| `docs/verification/phase2/README.md:3` | HEAD `af74cca` | HEAD `2beec38`（差 5 个提交） |
| `README.md` §1 P2-E 行 | "RealSSA 仍 403 → 无 RealSSA 指标结论" | 数据已获批、已全量评测（N1/N2）、已归档 TOS |
| `README.md` §1 P2-D 行 | "正式训练已启动…训练继续中" | 2026-10-22 19:54 已 100 epoch 结束 |
| `README.md` §3.5 回归 | "96 passed/6 skipped（102 collected）" | 现 112 collected；QA 后为 111/1；本轮 104/1/7 |
| `docs/REPRODUCTION_PLAN.md` Phase 2 状态 | "申请已提交（待批）…RealSSA 仍 403" | 已获批并完成 |
| `docs/DATA_ACCESS_GUIDE.md:18-19,34-35,218` | "数据文件仍 403" | 已下载/已归档 |
| `AGENTS.md §2` | lock 111 行 | 115 行 |
| `docs/verification/phase2/P2E-report.md` | 通篇"待审批、不伪造指标" | 报告冻结在审批前，RealSSA 结果只写在 `REPRODUCTION_PHASE2_REPORT.md §5.1` |

- `README.md` 索引本身与磁盘文件的**链接**基本一致：9 条 markdown 链接仅 1 条误报（`...`），P2-A…F 报告、脚本、
  CSV 均存在；但 **59 个文件中 45 个未被 README 提及**（日志/CSV/JSON 类），索引粒度只到"报告"级。

### R4（P1，科学口径）**RealSSA-Sim 7.774/8.479 的评测协议与论文未必同口径**

- `[代码]` `AVSpeechEvalDataset.__getitem__`（`look2hear/datas/avspeech_dataset.py:181-218`）**不做定长裁剪/补零**：
  `rand_start=0, stop=None`，整段读入后直接送模型；`audio_visual_test.py` 也按整句推理。
- `[验证]` RealSSA-Sim tt 片段长度实测 **4.6–5.8 s**（`soundfile.info` 抽样：73600–92800 样本 @16 kHz），
  因此 N1 是"**整段 5 s 级 clip**"结果，而不是 2 s 窗/0.5 s hop 的流式协议。
- `[验证]` 同一模型在 LRS2 上：整句 6.916 vs 窗口化 5.004（P2-B，−1.91 dB）。若论文 RealSSA 口径是流式/2 s 窗，
  N1 与论文 7.7/8.2 的"+0.07/+0.28"接近就**可能是协议差异抵消后的巧合**，不能直接写成"复现论文主结果"。
- 建议：报告必须声明"整段（非流式）协议"，并补一个窗口化对照（同一 9169 条、2 s 窗/0.5 s hop）后才能说"同协议复现"。
- 另：RealSSA-Real 的 DNSMOS 因整段 OOM 改用了"Studio 等价 2 s 窗/0.5 s hop"（`REPRODUCTION_PHASE2_REPORT.md §5.1`），
  与 Sim 的协议又不同；P808 3.3285 vs 论文 3.276（+0.05）的对比同样缺"论文口径"证据。

### R5（P1，状态过期）**"训练进行中/GPU 被外部占用"已不成立**

- `[验证]` `ps aux | grep audio_train` 无进程；`nvidia-smi` 无 compute apps、显存 2 MiB；`P2D-train.log` 末尾
  `Trainer.fit stopped: max_epochs=100 reached` + `Finished Training`（mtime 2026-10-22 19:54）。
- `docs/verification/phase2/P2D-train.pid` 仍是 `4002894`（**已入库的 stale pid**，进程早已退出）；
  `LRS2-paper-2gpu.pid` 未入库。
- 报告 §4.1/§6.2 的"训练继续中""GPU0 55.6 GB 被外部占用、无法双卡"均为历史状态；现在若要补跑双卡/8 卡实验，
  资源已具备（但仍只有 2 卡）。

### R6（P1，环境可重建性）lock 不可移植 + pypesq shim 不在锁文件

- 18 个 `@ file:///root/wheelhouse/...whl`（torch/PL/14 个 nvidia-*）；`pypesq.py` shim 在 site-packages、lock 里只有 `pesq==0.0.4`。
- `requirements.txt`（根）**无 `huggingface_hub`**（`grep -ci hugging = 0`）→ 按 README 装训练环境后 `hf://` 评测命令失败（旧文档 D3 仍成立）。
- `studio/requirements.txt:3` 仍写 `torch>=2.14.1,<3`（与 AGENTS 声明的 2.8.0 冲突，AGENTS 已注明"以本文件为准"）。
- `[验证]` `/root/wheelhouse` 5.4 GB 仍在，可用于离线重装；`/` 只剩 14 GB，`TMPDIR=/root/tmp` 必须设置。
- 建议：lock 改为 `pip freeze` 之外再存一份"index-url + 包名版本"的可移植清单（或用 `--no-deps` 的 wheelhouse tarball）。

### R7（P1，回归盲区）测试在提交后自失效 + 无 CI

- Phase-1 的 7 条 diff 守卫（`test_phase1_regression.py`）依赖"工作区相对 main 有改动"，提交后全部 skip（§1.3）。
  即：**"源码与 main 零 diff"这类守卫只在开发期有效，长期回归里等于没有。**
- `.github/workflows/pages.yml` 是唯一 CI（35 行，只发项目页）；**没有 pytest/Node CI**，回归全靠人工。
- `tools/repro/tests` 的 CSV 一致性测试对"空值/nan"无容错，被新数据打挂（R2/N1），说明"证据数据 → 回归"耦合脆弱。
- 覆盖盲区：评测协议（窗口化 vs 整段）、DNSMOS 全量、归档完整性、`pypesq` shim 存在性、lock 可安装性均无自动测试。

### R8（P1，数据可得性）第三方复核缺的数据/步骤

| 缺失项 | 现状 | 复核代价 |
|---|---|---|
| RealSSA-Sim tr/cv（441 GB + 40 GB） | 组共享源目录已删（`src_removed=True`），仅在 **TOS 冷端** `/aispeech-cfs/.../realssa_sim/{tr,cv,tt}` | 需 TOS 读权限 + 恢复 521.5 GB |
| RealSSA-Sim tt（40.1 GB） | 本机 `/root/data/RealSSA-Sim/tt` 19 个 tar 仍在（40,140,881,920 B）+ 转换后 `/root/data/falcon_realssa_sim/tt`（29.77 GiB） | 本机可复跑；外部需重新下载（gated，需审批） |
| RealSSA-Real（12.84 GB） | 本机 `/root/data/RealSSA-Real` 仍在 + TOS 归档 | 本机可复跑 |
| LRS2-2Mix 原始包 | `/root/data/lrs2_2mix/lrs2_rebuild`（17.39 GB）+ 转换后 23.36 GiB | 公开可下载（16.47 GB tar，sha256 `ebdfdf31…767e`） |
| 官方权重 | `/root/.cache/falcon_weights/` 是 **symlink** → `/root/group-shared/...` | 外部从 HF 下载（需网络/镜像） |
| 训练 ckpt（ep86/ep124/best） | `/root/tmp/p2d_train`、`/root/tmp/p2d_near_train`（各 ~52 MB/ckpt，`*.ckpt` 被 gitignore） | 仅本机；外部无法复核 8.858/9.270 的权重来源 |
| RealSSA-Sim 全量评测命令 | **仓库内无记录**（`realssa_sim_tt_pipeline.sh` 只覆盖 1-shard smoke，且默认目录 `realssa_sim_raw/realssa_falcon` 与实际 `falcon_realssa_sim` 不一致） | 需按日志头部 `configs/Falcon.yml` + `--data_dir /root/data/falcon_realssa_sim/tt` 反推 |
| 火山引擎 8 卡环境 | 只有 runbook，无任何节点/日志/结果 | 无法复核 |

### R9（P2，归档与字节口径）同一数据三个字节数、Real 文件数 422 vs 423

见 §2 "数字级不一致" 1–2。建议报告统一采用 `_MANIFEST.json` 口径（242 files / 521,532,645,652 B，dataset-only），
并注明 rsync 总数 521,532,645,847 B 含 manifest；Real 的 423 vs 422 需查 HF tree 再定论。

### R10（P2，仓库卫生）0 字节 CSV 入库、pid 入库、`Experiments/` 被整体忽略

- `docs/verification/phase2/P2B-full-big-whole-real.csv` = 0 字节但已入库（报告 §6 已说明"可忽略"）。
- `P2D-train.pid`（stale）已入库；`*.pid` 后来才加进 `.gitignore`（现 40 行，旧文档记 35 行）。
- `.gitignore` 现新增 `Experiments/checkpoint/LRS2_*/`、`Experiments/checkpoint/Falcon/results/`、`*.pid`——
  这解释了为什么 8.858 与 7.774 的 CSV 不入库，但**也意味着这两个核心结论永远不会有逐句证据进仓库**。

### R11（P2，口径/归属）本机 vs 火山引擎 vs 外部

- Phase 2 的**全部实测结果**都来自本机 2×A800（单卡 bs4 训练、单卡评测）；火山引擎只出现在方案/runbook
  （`docs/PHASE2_VOLCANO_LRS2_TRAINING.md`）与 commit message 的"方案落地"里，**没有一次远程执行**。
  报告目前没有把"本机实测"与"计划中的 8 卡"混写（这点是对的），但 `REPRODUCTION_PHASE2_REPORT.md §5.3`
  标题"8 卡方案"容易被误读为已执行，建议显式加"（未执行）"。
- RealSSA 数据的"实测"发生在 TOS 冷端归档/组共享下载链路上（rsync 309 MB/s 是**本机→TOS** 的本地挂载速度，
  不是 HF 下载速度）；HF 下载速度（~17 MiB/s、最低 0.12 MB/s）只在 `realssa_monitor.log`（仓库外）。

### R12（P2，许可）RealSSA 数据不可再分发

- RealSSA-Sim/Real 为 CC BY-NC-ND/研究授权，**不可转发**；派生转换物（`/root/data/falcon_realssa_sim`）同样受限。
- 因此"把数据/CSV 放进仓库"本身不可行——这解释了 R1/R2 的一部分，但**不能解释日志不入库**（日志不含数据本体）。
  报告应把"证据不可入库"与"许可限制"分开陈述，避免用许可为缺失日志开脱。

---

## 5. 与旧文档 `docs/understanding/08-ops-tests-docs.md` 逐条对照

> 旧文档分析对象是 `7576854`（Phase 0 上游状态），本轮对象是 `2beec38`（Phase 2 结束）。逐条给【确认/修正/推翻】。

| # | 旧文档断言 | 本轮结论 | 判定 |
|---|---|---|---|
| O1 | Studio pytest = **35 passed / 2 skipped** | 实测 35/2（19.05 s，37 collected，test_server 22） | **确认** |
| O2 | pytest 需注入 `docs/verification/scripts/pil_stub/` + `-p no:typeguard` 才能跑 | 现在 **Pillow 13.0.0 已装**，直接 `pytest studio/tests` 即 35/2，无需 stub、无需 `-p no:typeguard` | **修正**（旧文档是缺 Pillow 时代的权宜；该 stub 路径仍存在但已不必要） |
| O3 | 8 个 Node 测试全部 rc=0 | 实测 8/8 rc=0 | **确认** |
| O4 | `.github/workflows/pages.yml` 35 行、只在 main+公开仓库部署 | `wc -l` = 35，触发/权限/`if: !private` 未变 | **确认** |
| O5 | 依赖四文件行数 29 / 11 / 110 / 3 | 实测 29 / 11 / 110 / 3 | **确认** |
| O6 | 根 `.gitignore` 35 行 | 现在 **40 行**（新增 `Experiments/checkpoint/LRS2_*/`、`Experiments/checkpoint/Falcon/results/`、`*.pid`） | **修正** |
| O7 | D1 P0：`audio_train.py:170` 未定义 `comet_logger`，训练必崩 | `grep -rn comet_logger --include=*.py` 仅剩注释/文档；`audio_train.py:169 build_logger()` 已实现（CSV/TB/Comet/none） | **推翻（已修复，P0 关闭）** |
| O8 | D2 P0：README 评测命令用 `Experiments/checkpoint/Falcon/conf.yml`，其 `audionet_name: SpeakerContrastiveV2` 不可 import | `Experiments/checkpoint/Falcon/conf.yml:88` **仍是** `SpeakerContrastiveV2`；`configs/Falcon.yml:10` 是 `FalconSpeakerContrastive`；`audio_visual_test.py:288-292` 现在抛出**明确的 AttributeError 提示**（指出 conf 快照过期），且默认配置改为 `configs/Falcon.yml`；但 `README.md:185/188` 仍写旧 conf 路径 → **命令仍不可运行** | **部分确认（P0 未闭环；已从"静默失败"变"显式诊断"）** |
| O9 | D3 P1：根 requirements 缺 `huggingface_hub` | `grep -ci hugging requirements.txt` = 0 | **确认** |
| O10 | D4 P1：`best_k_models.json` 的 `${ROOT}` 不展开 → 不传 ckpt 必 FileNotFound | 文件仍在、逻辑未变（本轮未复跑，代码未改） | **确认（未修）** |
| O11 | D20/Q6：权重体积/内容未验证（"45 MB 无法验证"） | 已实测 47,156,343 B（44.97 MiB）+ sha256 + 662 tensors；`baseline-fingerprint.json` 入库 | **修正（已升级为 [验证]）** |
| O12 | D15：MACs/延迟未复算 | `docs/verification/runs/r3-macs.log` 已复算：audio MACs **10.611 G**、params 0.538 M、FRCNN 7.77 G@64²/15.90 G@88²；延迟仍未复算 | **修正（MACs 已复算，延迟仍缺）** |
| O13 | D17：`.gitignore` 无 `tensorboard_logs/` | 仍未加（训练会在 `Experiments/tensorboard_logs/` 落盘；但 `Experiments/checkpoint/LRS2_*/` 与 `Falcon/results/` 已忽略） | **确认（未修，风险降低但仍存在）** |
| O14 | §8.5 未验证清单：权重下载/加载、MACs、Metal、浏览器 QA、Pages 部署等 | 权重与 MACs 已补；Metal/浏览器/Pages 仍未验证（本轮同） | **部分确认** |
| O15 | "本域无 CI，只有 Pages" 的隐含结论 | 仍然成立：无 pytest/Node CI | **确认** |
| O16 | 旧文档未覆盖 `docs/verification/**` 的入库完整性 | 本轮新增结论：**226 文件仅 129 入库，`*.log` 全被忽略、`runs/` 0 入库** | **新增（旧文档盲区）** |
| O17 | 旧文档基于 `/root/.venvs/falcon`（已删除的旧验证环境） | 现在唯一合法环境是 conda `falcon`（`AGENTS.md §1`）；旧文档里的 venv 路径只能用于复现历史证据 | **修正（环境口径已切换）** |
| O18 | 旧文档称"训练侧依赖基本不锁 → 论文数字不可精确复现" | 现在有 `conda-env-lock.txt`（115 行）与 `conda-env.yml`，但 **18 个包是 `/root/wheelhouse` 绝对路径**，仍不是可移植锁 | **部分修正（有锁但不可移植）** |

---

## 6. 对复现报告（`REPRODUCTION_PHASE2_REPORT.md`）的具体要求

1. **补 §5.5 "证据索引表"**：每个关键数字给出「文件路径 + 行号/CSV 行 + 是否入库 + 复跑命令」四列；
   对未入库证据（`/root/tmp/*.log`、`Experiments/checkpoint/**/results/*.csv`）显式标注 `[本机，未入库]`。
2. **修正过期状态**：§1 TL;DR 的"训练继续中"→"已于 2026-10-22 19:54 完成 100 epoch（best val 13.201）"；
   §4.1 的"epoch 86 最佳 13.118"→"ep99 val 13.201；已评测 ckpt 为 ep86"；§6.2 的"GPU0 55.6 GB 被占"→"历史快照，现已释放"。
3. **RealSSA 结论加协议声明**：N1 必须写明"整段（4.6–5.8 s clip）推理、非 2 s 流式"，并说明与论文 7.7/8.2 的可比性未证；
   N2 必须写明"2 s 窗/0.5 s hop + channel=mean + align_tail"；两者不可互相替代。
4. **把 §5.1 的 RealSSA 结果从"总报告内嵌表"升级为独立 P2 报告**（`P2E-report.md` 目前冻结在"403/无指标"，自相矛盾），
   或至少在 P2E 报告头部加"后续更新见总报告 §5.1"的指针。
5. **数字口径统一**：RealSSA-Sim 用 `_MANIFEST.json` 的 242 文件 / 521,532,645,652 B，并注明 rsync 总数含 manifest；
   RealSSA-Real 的 423 vs 422 必须解释或改为"本机实测 423"。
6. **更正 `P2D-eval-falcon.pt.csv` 的引用口径**：它是 300 条子集（6.916/7.492），不是全量 7.03/7.71；
   全量证据目前只有 `/root/tmp/lrs2_eval_tt_full.log`（未入库）。
7. **声明 8 卡/火山引擎"未执行"**：§5.3 标题加"（方案，未执行）"，并写明"本报告所有实测均在本机 2×A800"。
8. **补"证据保全"承诺**：后续评测必须用唯一输出目录（避免 R2 覆盖），并把最终 metrics 段另存 `*.txt`（可入库）。
9. **提交未提交内容**：§5.4（当前工作区唯一未提交改动）应 commit；`origin/repro/phase2` 落后 2 个提交
   （`29e19c7`、`2beec38`），虽然 `origin/main` 已含这两个提交，但分支 ref 应同步。
10. **明确"不可入库"与"未入库"的区别**：RealSSA 数据受 CC BY-NC-ND 不可分发（合理），但**日志/摘要 JSON 不受此限**，
    不应以许可为由让日志缺失（R12）。

---

## 7. 未解问题

- **Q1** 论文 RealSSA-Sim 的评测协议（整段 / 2 s 窗 / hop、通道选择、是否对齐）到底是什么？没有它，N1 的"+0.07/+0.28 dB"
  不能算"同协议复现"。作者 issue（`JusperLee/Falcon#1`）尚无回复证据。
- **Q2** `P2E-realssa-real-full{,2}.log` 两次 `exit=1` 的具体失败原因（是否 OOM/脚本 bug）未归档；只保留成功第 3 次，
  失败尝试的教训丢失。
- **Q3** RealSSA-Real 423 vs 422 文件差在哪一个文件？是否 HF 仓库在 Phase 1 之后新增/改动了文件？
- **Q4** `metrics_combined.csv` 的 3 条空 SDR 是否应记为 `-inf`/`nan` 而非空串？评测脚本（`ALLMetricsTracker`）的
  CSV 写出逻辑对 nan 的处理未定义，导致回归测试崩溃——是否需要修 `audio_visual_test.py` 或测试？
- **Q5** 训练完成后 best ckpt 是 ep99（val 13.201），但评测用了 ep86（val 13.118）。为何不评 ep99/best_model？
  8.858 与"ep99 评测"的差是否会改变"contrastive +0.24 dB"的结论？
- **Q6** `/root/.cache/falcon_weights` 改为 symlink 到 group-shared 后，若该挂载掉线，多少测试/脚本会失败？
  是否需要把权重复制回本地（47 MB + 49 MB + 90 MB，代价很小）？
- **Q7** 8 卡/火山引擎方案是否有任何后续执行计划？当前本机 GPU 已空，双卡复跑 paper 配方（`LRS2_2Mix_Paper_2gpu_smoke.yml`）
  是否值得做，以至少给出"配方可跑通 + 趋势"的证据？
- **Q8** `docs/verification/runs/` 的 45 个日志全部未入库，但它们被 `docs/understanding/**` 大量引用为 `[验证]`。
  这些引用在 clone 后是否应降级为 `[文档]`，或统一 force-add 日志？

---

## 8. 验证记录（本轮实际执行的命令与输出摘要）

| # | 命令 | 结果 |
|---|---|---|
| V1 | `/root/miniforge3/envs/falcon/bin/python -V` | `Python 3.11.17` |
| V2 | `PYTHONPATH=$PWD … -c "import torch; …"` | `torch 2.8.0+cu128 / cuda True / 2 devices / cuda 12.8 / cudnn 91002` |
| V3 | `PYTHONPATH=$PWD … -c "import look2hear.models/system/metrics, audio_train, audio_visual_test"` | `imports OK`（仅 timm FutureWarning） |
| V4 | `… -c "import pypesq; print(pypesq.__file__)"`；`importlib.metadata` | shim 在 `site-packages/pypesq.py`；`typeguard 2.13.3 / pesq 0.0.4 / speechmos 0.0.1.1` |
| V5 | `/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q` | **35 passed, 2 skipped in 19.05s**（37 collected） |
| V6 | `PYTHONPATH=$PWD … -m pytest studio/tests/test_server.py -q` | **22 passed in 0.86s** |
| V7 | `PYTHONPATH=$PWD … -m pytest tools/repro/tests -q` | **1 failed, 104 passed, 7 skipped, 43 warnings in 111.87s**（112 collected） |
| V8 | 同上 `-q -rs`（后台任务） | 失败 = `test_phase2_eval_csv_aggregation_consistent`（`test_phase2_pipeline.py:245`，空串）；7 skip 全在 `test_phase1_regression.py:512/521/530/547/564/576/600` |
| V9 | `for t in studio/tests/check_*.mjs … ; do node "$t"; done` | 8/8 `rc=0` |
| V10 | `df -h / /root` | `/` 20G/5.9G/14G/31%；`/root` 164T/151T/14T/92% |
| V11 | `nvidia-smi --query-gpu=…`；`--query-compute-apps=…` | 2× A800-80GB，2 MiB used，0% util，driver 535.129.03；**无计算进程** |
| V12 | `wc -l conda-env-lock.txt conda-env.yml`；`grep -c "@ file:///root/wheelhouse"` | lock 115 行 / yml 94 行；**18 个 file:// 绝对路径** |
| V13 | 复算 `Experiments/checkpoint/Falcon/results/metrics_combined.csv` 的 avg/std | 9169 行逐句：si-snr_i mean 7.7743、sdr_i 8.4794（SDR n=9166），与 avg 行一致；3 条空值行定位 859/3454/8820 |
| V14 | 读 `/root/tmp/realssa_sim_eval_official.log` | `Final metrics si-snr_i 7.774307 / sdr_i 8.479395`，`exit=0`；开头有 3 组 SDR singular + PESQ "No utterances detected" |
| V15 | 读 `P2E-realssa-real-full3.log` + `summary.json` | `N=54 SIG=3.2209 BAK=3.6921 OVRL=2.7901 P808=3.3285`；`segments_ok 54`；full/full2 均 `exit=1` |
| V16 | 复算 `Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv` avg | `8.858191910560924 / 9.269574165939565`（3000 行），与 `p2d_eval_contrastive_epoch124.log` 一致 |
| V17 | 读 `/root/tmp/p2d_eval_epoch86_full.log`、`lrs2_eval_tt_full.log` | `8.617919/9.041506`；`7.029101/7.712925`（均在仓库外） |
| V18 | 读 `P2D-train.log` 尾部、`P2D-train.pid`、`p2d_near_train/.../metrics.csv` | 100 epoch 完成；stale pid 4002894；near-run best val 13.390929@ep124、test 13.7618@ep120 |
| V19 | 复算 `P2A-small-200.json` 14 变体 | 基线 6.9547/7.6224；最大正收益 +0.3405（plain-SNR），标准口径最大 +0.282，其余 ≤0 |
| V20 | 读 `P2B-b300-*.csv` avg 行 | small 6.9162/7.4916；big 7.8872/8.4033；small windowed 5.0042/5.7624；big windowed 5.5687/6.2701 |
| V21 | `grep` P2F JSON/log | `4.823487742555699e-08`、`1.8408684887560867e-07`；脚本 `protocol_check.py:49` 硬编码 CSV 路径 |
| V22 | `sha256sum` 权重 4 文件；对比文档 | falcon.pt `a1888b5a…928c`/47,156,343 B；big `ce2a92a3…037e`/49,645,087 B；**与 baseline/QA 文档一致**；权重为 symlink |
| V23 | `sha256sum` speechmos ONNX 4 文件 | 与 `P2E-dnsmos-synthetic.json` **4/4 一致** |
| V24 | `find/stat` TOS 归档（不 `du` 大目录） | sim：243 文件 / 521,532,645,847 B（242 数据集文件 521,532,645,652 + manifest）；real：423 文件 / 12,843,107,273 B；`_MANIFEST.json` `bad=[]` |
| V25 | 读 `_archive_done.json` + `realssa_console.log` | `ok:true`×2；rsync `521,532,645,847 100% 309.26MB/s 0:26:48 (xfr#243)`、`rc=0 src_removed=True` |
| V26 | 读 `realssa_status.json` | `expected_bytes 521,532,645,847`、`phase archive`、`note "rsync -> TOS"` |
| V27 | `find`/`os.walk` 统计转换数据 | `falcon_lrs2_2mix` 23.36 GiB（71,526 文件）；`falcon_lrs2_2mix_paper` 25.08 GiB；`falcon_realssa_sim` 29.77 GiB；`RealSSA-Sim/tt` 19 tar = 40,140,881,920 B |
| V28 | `soundfile.info` 抽样 RealSSA-Sim tt wav | 73600–92800 样本 @16 kHz = **4.6–5.8 s**（非 2 s） |
| V29 | 读 `avspeech_dataset.py:181-218`、`audio_visual_test.py:389-470` | eval 数据集整段读取、无裁剪/补零；评测结果写固定 `Experiments/checkpoint/<exp>/results/metrics_combined.csv` |
| V30 | git 状态 | `## repro/phase2`；`M docs/REPRODUCTION_PHASE2_REPORT.md`（+9 行 §5.4）；HEAD `2beec38`；`origin/repro/phase2` = `9121598`（**落后 2**：29e19c7、2beec38）；`origin/main` = `2beec38` |
| V31 | `git ls-files` vs `find` | `docs/verification`：入库 129 / 磁盘 226；`runs/` 入库 **0** / 45；phase2：入库 35 / 磁盘 64（29 个日志未入库） |
| V32 | 读 `phase2/README.md` 并做链接存在性检查 | 9 条 md 链接仅 1 条误报（`...`）；但 59 文件中 45 个未被索引；状态/HEAD/回归数字全部过期 |
| V33 | 读 `REPRODUCTION_PLAN.md` Phase 2 段、`DATA_ACCESS_GUIDE.md:18/34/218` | 仍写"RealSSA 待批/403"，与实际（已获批、已评测、已归档）矛盾 |
| V34 | 读旧文档 `08-ops-tests-docs.md`（768 行） | 完成 §5 的 18 条逐条对照（确认 8 / 修正 5 / 部分 3 / 推翻 1 / 新增 1） |
| V35 | `grep -rln "火山\|volcano\|8gpu" docs/verification/` | 0 命中 → 火山/8 卡无任何执行证据 |

> 说明：本轮**未**修改任何源码/配置/他人文档，未 `pip install`，未联网，未启动训练/下载，未做 git 写操作；
> 唯一写入是本文件（`docs/understanding/deepdive-20261024/08-verification-ops-env.md`）。
> 对 521 GB 的 TOS 目录只做 `os.walk` 元数据统计（未 `du` 全量内容），单条命令均 <5 min。

## 8. 2026-10-24 执行更新（P0 收口，Lead 复核）

本轮已按 §4 的 R1–R7 执行修复（用户授权"该干的干了、该修的修了"），完整清单见
`docs/verification/phase2/P0-FIX-report.md`。状态变化：

| 审计项 | 处置 | 结果 |
|---|---|---|
| R1 日志未入库 | `.gitignore` 增加负向规则 `!docs/verification/**/*.log\|csv` + `runs/` | `git add -n docs/verification` = 109 文件 / 3.2 MB |
| R2 CSV 被覆盖 | RealSSA-Sim 逐句 CSV、ep124 逐句 CSV、DNSMOS per-segment 复制入库；官方权重全量 + ep99 全量复跑 | `P2E-realssa-sim-metrics_combined.csv`（9,172 行）、`P2D-eval-epoch124-contrastive.csv`、`DNSMOS-realssa-real-mean-full3/`、`P2D-eval-{falcon-full,epoch99}.*` |
| R3 索引/状态过期 | 6 份文档加 2026-10-24 状态横幅/追加节（GUIDE / PLAN / phase2 README / P2E / PHASE2_REPORT / FINAL） | 见各文件 |
| R4 协议口径 | RealSSA-Sim 明确"整段推理、channel=mean"；ch0 敏感性转换完成、评测待跑 | `P2E-realssa-sim-ch0-convert.log`（9,169 条） |
| R6 lock 不可移植 | 18 行 `file:///root/wheelhouse` → `pkg==version` | 115 行不变、`grep file://`=0、PEP508 0 错 |
| R7 回归盲区 | CSV 空串 → `_as_float`；`.gitignore` 加入两处 diff-scope 守卫白名单 | `tools/repro/tests` **109 passed / 6 skipped / 0 failed**；`studio/tests` 35/2 |
| R8 数据可得性 | TOS 归档/本地数据未动；证据复制入库 | 不变 |

仍未完成（不在 P0）：LRS2 8 卡论文配方（P2）、RealSSA-Sim 从零训练（P3）、3090 延迟（P4）。
