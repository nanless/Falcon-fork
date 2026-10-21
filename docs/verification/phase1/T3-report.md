# Phase1 T3 报告：评测指标修复（F7/F8）+ `audio_visual_test.py` 可运行化

> 任务：`task-11`（分支 `repro/phase1`，**未 commit**）
> 依据：`docs/REPRODUCTION_PLAN.md` §6.2 F7/F8、§6.3；`docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` §5
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / fast-bss-eval 0.1.4 / pypesq+pesq shim / pystoi）
> 写作用域：`look2hear/metrics/`、`audio_visual_test.py`、`tools/repro/tests/`、`docs/verification/phase1/`（未碰其它源码）

## 1. 摘要（结论先行）

- **F7 指标语义已修**：评测实际使用的 `ALLMetricsTracker` 现在输出**正号、越高越好**的
  `sdr / sdr_i`（真 SDR，`fast_bss_eval.sdr_pit_loss(..., filter_length=512)`）与
  `si-snr / si-snr_i`（标准零均值 SI-SDR + PIT）；逐句行与 `avg/std` 行**同一符号约定**；
  PESQ 改为 `pesq(clean, estimate, fs)`；STOI 顺序核对无误保持不变。
- **F8 之外的可运行化已做**：`audio_visual_test.py` 支持
  `--conf_dir configs/Falcon.yml` + `--data_dir`、`${DATA_ROOT}` 等占位符展开、
  本地 `.pt`（Lightning ckpt / `falcon.pt` / 裸 state-dict）与 `hf://` 权重、
  `_find_best_ckpt` 的 **epoch 数值排序**、`--limit N`、更清晰的错误信息；
  评测 worker 不再依赖 train/valid split，直接对 test index 构造 `AVSpeechEvalDataset` 做**整句**推理。
- **验证（全部实跑）**：
  - `pytest tools/repro/tests/test_metrics.py tools/repro/tests/test_eval_entry.py -q` → **20 passed**（12.15 s）；
  - **官方权重** `/root/.cache/falcon_weights/falcon.pt` + 合成样本 1 条，`--limit 1` 端到端跑通：
    strict 加载 audio 525 keys / video 137 keys，整句 32000 采样/50 帧输出 → CSV；
    日志 `docs/verification/phase1/T3-eval-smoke.log`，CSV 快照 `T3-eval-smoke-metrics.csv`；
  - `python -c "import audio_visual_test"` OK（`T3-import-check.log`）；
  - `${DATA_ROOT}` 路径（不传 `--data_dir`）同样跑通；三种失败场景给出明确错误并以退出码 1 结束。
- **遗留**：真实 LRS2/RealSSA 数据仍缺（冒烟值不可用作论文结论）；`hf://` 分支与多卡 `mp.spawn`
  未在本轮实跑；老 `Experiments/checkpoint/Falcon/conf.yml` 的 `audionet_name` 仍为
  `SpeakerContrastiveV2`（现在会抛出可读错误，不静默）。

## 2. 改动清单（含 file:line）

### 2.1 `look2hear/metrics/`

| 文件 | 改动 | 关键位置 |
|---|---|---|
| `core.py`（**新增**） | 指标单一事实来源；明确 SI-SDR / SI-SDRi / SDR / SDRi / PESQ / STOI 定义，统一正号；float64 + CPU 计算保证确定性 | `SDR_FILTER_LENGTH=512`（:69）；`PIT_SI_SDR`（:75，`threshold_byloss=False`，避免 >30 dB 好样本被过滤成 nan）；`si_sdr`（:152）、`si_sdr_improvement`（:158）、`sdr`（:168）、`sdr_improvement`（:180）、`pesq_metric`（:196，`(clean, estimate, fs)`）、`stoi_metric`（:208） |
| `allwrapper.py` | `ALLMetricsTracker` 改为调用 `core`：`sdr/sdr_i` 从“−SNR/SNRi”改成真 SDR/SDRi；逐句行与 `avg/std` 同符号；PESQ 实参与 STOI 顺序修正；多源时 PESQ/STOI 记 `nan`；新增 `sample_rate` / `sdr_filter_length` 可选构造参数（默认 16000/512，向后兼容） | `CSV_COLUMNS`（:34）、`_compute`（:63）、`final`（avg/std 行） |
| `wrapper.py` | 遗留 `MetricsTracker` 语义同步修正（真 SDR + 零均值 SI-SDR + 正号一致），删除未使用的 `torch_mir_eval` 硬依赖；CSV 列保持 5 列 | `class MetricsTracker`（:29）起 |
| `__init__.py` | 额外导出 `core` 的指标函数与常量（`si_sdr`/`sdr`/…） | 全文 |

指标定义（与 `core.py` docstring 一致）：

```
SI-SDR  = 10 log10( ||alpha*s||^2 / ||e - alpha*s||^2 ),  alpha = <e,s>/||s||^2（零均值、PIT）
SI-SDRi = SI-SDR(estimate) - SI-SDR(mixture)
SDR     = -fast_bss_eval.sdr_pit_loss(estimate, clean, filter_length=512)   # BSS-Eval-v4 风格多抽头
SDRi    = SDR(estimate) - SDR(mixture)
PESQ    = pypesq.pesq(ref=clean, deg=estimate, fs=16000)
STOI    = pystoi.stoi(x=clean, y=estimate, fs=16000, extended=False)
```

> 关于 SDR 的 ±0.05 dB 对拍：`fast_bss_eval` 的最优 512 抽头滤波相对解析 SDR 有极小正偏差
> （2 s 信号约 +0.067 dB，10 s 约 +0.015 dB），因此 `test_metrics.py` 的解析对拍用 **10 s 信号**
> 并锁定 ±0.05 dB；同时另有一条“与 `fast_bss_eval` 直接调用逐位相等（1e-6）”的测试，
> 保证实现与所选库口径一致。

### 2.2 `audio_visual_test.py`（329 → 约 560 行）

| 功能 | 位置 | 说明 |
|---|---|---|
| 新 CLI | `--conf_dir`（默认改为存在的 `configs/Falcon.yml`）、`--data_dir`、`--ckpt_path`、`--gpus`、`--limit`（:70-104） | `--gpus ''/cpu` 走 CPU，单 GPU 默认 `0` |
| `${VAR}` 展开 | `_expand_env_vars` / `_list_unresolved_vars`（:123-160） | 递归展开 config 中所有字符串；对残留占位符打印警告 |
| config 组装 | `_get_train_conf`（:165-180）、`_prepare_config`（:170） | 同时接受原始 YAML 与 `{"train_conf": {...}}`；校验 `audionet/videonet/datamodule`；`--data_dir` 覆盖 test index；数据目录缺失给出可执行提示 |
| 数据目录解析 | `_resolve_index_dir`（:148） | 支持 `<data_dir>/tt/` 或 `<data_dir>/` 直接含 `mix.json`/`target.json` |
| best ckpt | `_find_best_ckpt`（:214） | best_k_models（最低 val_loss 且本地存在）→ last.ckpt → `epoch=*.ckpt` **按 epoch 数值**取最大（`epoch=1000` > `epoch=742`） |
| ckpt 读取 | `_extract_state_dict`（:255）、`_load_models_from_ckpt`（:280） | 支持 Lightning ckpt / 发布 `falcon.pt`（`{state_dict, architecture, epoch}`）/ 裸 state dict；audio & video 均 **strict=True**，失败给出“config 与权重不匹配（如 conf.yml 缺 repeats/shared）”的可读错误；对 `best_model.pth`（仅音频、无前缀）明确拒绝 |
| 评测 worker | `process_worker`（:365） | 直接构造 `AVSpeechEvalDataset`（不需要 train/valid）；整句、batch=1、`rank::world_size` 分片；`--limit` 全局截断；mask/mouth 形状校验；CUDA/CPU 自动 |
| 结果合并 | `_merge_results`（:467） | 排除每 rank 的 `avg/std` 行，只用逐句行重算**全局唯一** avg/std；写 `metrics_combined.csv` 并清理 shard |
| main | `main`（:503） | 统一打印 config/test index/ckpt/results 目录；异常打印 traceback + 退出码 1 |
| `hf://` | `__main__`（:540-560） | 同原逻辑：spawn 前 `hf_hub_download`；缺 `huggingface_hub` 给安装提示；格式校验 |

**兼容性**：
- CSV 列名与文件位置不变（`Experiments/checkpoint/<exp>/results/metrics_combined.csv`），
  仅列语义变为“正号/越高越好”；下游若按旧错误符号解读需要更新。
- `ALLMetricsTracker.save_file` 仍为位置参数；新增参数均有默认值（旧调用 `ALLMetricsTracker(save_file=...)` 不变）。
- `SPlitMetricsTracker` 未改动（死代码，3 源拆分语义复杂，不在本任务范围）。
- 评测现在**不需要** `train/valid` index；这是行为改进，不影响输出路径。

### 2.3 测试/工具（`tools/repro/tests/`）

| 文件 | 内容 |
|---|---|
| `test_metrics.py`（新增） | 8 个测试：SI-SDR/SI-SDRi 解析值 ±0.05 dB；SDR 解析值 ±0.05 dB（10 s 信号）；SDRi 符号（好估计>0、坏估计<0）；与 `fast_bss_eval` / `PairwiseNegSDR` 直接调用对拍；PESQ/STOI 实参顺序（monkeypatch 记录参数）；CSV 逐句/avg/std 符号与数值一致性；多源不崩且 PESQ=nan |
| `test_eval_entry.py`（新增） | 12 个测试：epoch 数值排序、best_k 最低值优先且跳过不存在文件、last.ckpt 优先级、缺 ckpt 报错；`${VAR}` 展开/data_dir 覆盖/缺 test_dir 报错；ckpt state-dict 三种格式与 `best_model.pth` 拒绝；`_merge_results` 只用逐句行重算 avg/std；CPU fallback |
| `make_t3_smoke_data.py`（新增） | 生成 1 条（参数化 n）合成 eval index：stereo 16k wav（双通道相同，兼容 mono 守卫前后）、`mouths/*.pt` uint8 RGB `[F,96,96,3]`、`mix.json`/`target.json`/`_meta.json`。**独立于 T4 的 `make_synthetic_data.py`**（后者面向训练目录） |

## 3. 验证记录（命令 + 结果，全部实跑）

### V1 单元测试（conda `falcon`）

```bash
cd /root/code/github_repos/Falcon-fork
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_metrics.py -q
#   8 passed in 4.24s           -> docs/verification/phase1/T3-metrics-tests.log
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_eval_entry.py -q
#   12 passed in 7.56s          -> docs/verification/phase1/T3-eval-entry-tests.log
/root/miniforge3/envs/falcon/bin/python -m pytest \
    tools/repro/tests/test_metrics.py tools/repro/tests/test_eval_entry.py -q
#   20 passed, 5 warnings in 12.15s -> docs/verification/phase1/T3-combined-tests.log
```

解析值对拍样例（`test_metrics.py`）：SI-SDR 20 dB 的构造信号实测与解析值差 <1e-6 dB；
SI-SDRi 0→20 dB；SDR（10 s，alpha=0.3）解析值 10.4576 dB 与实测差 ~0.015 dB（<0.05）；
PESQ/STOI 参数顺序通过记录型 mock 断言 `(clean, estimate, 16000)`；CSV `avg` 行与逐句均值差 <1e-9。

### V2 端到端冒烟（**官方权重** + 合成样本，`--limit 1`）

```bash
# 1) 合成 eval index（1 条、2 s、stereo 16 kHz、50 帧 96×96 RGB）
/root/miniforge3/envs/falcon/bin/python tools/repro/tests/make_t3_smoke_data.py \
    --out /tmp/t3_smoke/data --seconds 2 --n 1

# 2) 评测（仓库根不在 cwd，避免写仓库内的 Experiments/；PYTHONPATH 指向仓库）
cd /tmp/t3run
PYTHONPATH=/root/code/github_repos/Falcon-fork \
/root/miniforge3/envs/falcon/bin/python /root/code/github_repos/Falcon-fork/audio_visual_test.py \
    --conf_dir /root/code/github_repos/Falcon-fork/configs/Falcon.yml \
    --data_dir /tmp/t3_smoke/data \
    --ckpt_path /root/.cache/falcon_weights/falcon.pt \
    --gpus 0 --limit 1
```

结果（`docs/verification/phase1/T3-eval-smoke.log`，CSV 快照 `T3-eval-smoke-metrics.csv`）：

```
[eval] test index  : /tmp/t3_smoke/data/tt
[eval] ckpt_path   : /root/.cache/falcon_weights/falcon.pt
✓ audio_model loaded: 525 keys from falcon.pt
✓ video_model loaded: 137 keys from falcon.pt
[eval] test index: ... | utterances=1 | this rank=1 | device=cuda:0
✅ Combined metrics saved to: /tmp/t3run/Experiments/checkpoint/Falcon/results/metrics_combined.csv
```

```csv
snt_id,sdr,sdr_i,si-snr,si-snr_i,pesq,stoi
u0000.wav,-5.905,-4.019,-11.478,-9.427,1.046,0.056
avg,      -5.905,-4.019,-11.478,-9.427,1.046,0.056
std,       0.0,    0.0,    0.0,    0.0,   0.0,  0.0
```

说明：该合成样本不带真实语义（随机嘴部 + 单音目标），模型输出劣于混合，故冒烟数值为负——
**这是合理的度量结果，不是符号 bug**；“正号/越高越好、行与 avg 同符号”由 V1 的解析单测保证。
冒烟关注的是：权重 strict 加载、整句变长推理（32000 采样/50 帧 → 32000 采样输出）、
指标后端可运行、CSV 结构（逐句 + 唯一 avg/std）与 shard 清理。

### V3 `${DATA_ROOT}` 路径（不传 `--data_dir`）

```bash
# conf: test_dir: ${DATA_ROOT}/tt  （其余数据键仍为 "--"）
DATA_ROOT=/tmp/t3_smoke/data PYTHONPATH=/root/code/github_repos/Falcon-fork \
/root/miniforge3/envs/falcon/bin/python /root/code/github_repos/Falcon-fork/audio_visual_test.py \
  --conf_dir /tmp/t3_smoke/conf_data_root.yml \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt --gpus 0 --limit 1
# exit=0，同一份指标被写入 /tmp/t3run2/Experiments/checkpoint/Falcon/results/metrics_combined.csv
```

### V4 错误路径（清晰报错 + exit 1）

| 场景 | 结果（均写入 `T3-eval-smoke.log`） |
|---|---|
| 旧 `Experiments/checkpoint/Falcon/conf.yml`（`audionet_name: SpeakerContrastiveV2`） | `AttributeError: look2hear.models has no attribute 'SpeakerContrastiveV2' ... 发布权重用 FalconSpeakerContrastive；shipped conf 是过期快照` |
| `--data_dir /tmp/does-not-exist-t3` | `FileNotFoundError: No JSON index found for split 'tt' under ... Pass --data_dir ... or set DATA_ROOT ...` |
| 不传 `--ckpt_path` 且 exp 目录无 ckpt | `FileNotFoundError: No checkpoint found in ... Pass --ckpt_path explicitly ... or train first` |

### V5 导入检查

```bash
PYTHONPATH=. /root/miniforge3/envs/falcon/bin/python -c "import audio_visual_test"
# import audio_visual_test OK   -> docs/verification/phase1/T3-import-check.log
```

## 4. 与并发任务的关系 / 依赖

- **T4（data-pipe）**：`look2hear/datas/avspeech_dataset.py` 的 mono 守卫在并行修；本任务的
  冒烟 fixture 用 **stereo** wav，因此在 F5 落地前后都能跑（冒烟跑通时 T4 的改动已在工作区）。
  `audio_visual_test.py` 现在直接实例化 `AVSpeechEvalDataset`，不依赖 `AVSpeechDataModuleWithNear.setup()`
  的 train/valid 目录。
- **T5（model-core）**：`FRCNNVideoModel.train()` 改为 `return self` 后，`video_model.eval()` 返回模型而非
  None；本脚本仍以语句形式调用 `.eval()`，修复前后行为都正确。
- **T1/T2（train-sys）**：本任务不改训练入口/litmodule；`audio_train.py` 的改动不影响评测路径。
- **T15（QA）**：其 `test_phase1_regression.py` 会独立复核指标实现，可直接与
  `look2hear/metrics/core.py` 对拍（本报告的 V1 已给出解析值口径）。

## 5. 遗留问题 / 风险

1. **无真实数据**：LRS2-2Mix 只有音频、无嘴部；RealSSA gated 未获取。因此冒烟只证明“管线可运行 +
   指标实现正确”，**不能**产出论文级 SI-SDRi/SDRi 数字；真实数据到位后需按同一命令替换
   `--data_dir`（及通道策略）复跑。
2. **`hf://` 分支未实跑**：本机已有本地权重缓存，未触发下载路径；代码与原逻辑一致且
   `huggingface_hub` 已在 conda 环境，但建议有网时用 `--ckpt_path hf://JusperLee/Falcon/falcon.pt`
   冒烟一次（会写 HF 缓存，不写仓库）。
3. **多卡 `mp.spawn` 未实跑**：本机 2×A800 可用但冒烟只用了 1 卡；`--limit N` 的语义是
   “全局前 N 条再按 rank 轮转切片”，当 `N < world_size` 时部分 rank 无样本（其 shard 只有 nan avg/std，
   merge 会正确跳过）；如需多卡复跑，建议 `--limit` ≥ world_size 或 0。
4. **SDR 计算成本**：`filter_length=512` 的 `fast_bss_eval` 每信号约 0.2–0.3 s（CPU；A800 上更快），
   每个 utterance 要算 estimate + mixture 两次；大测试集（几千条）会明显变慢。默认按任务要求固定 512，
   未加 `--sdr-filter-length` 开关（如需可后续加，但会偏离任务口径）。
5. **PESQ/STOI 仅 n_src=1**：多源时写 `nan`（评测入口配置 n_src=1；若未来多源评测，需要重新定义）。
6. **快照 conf.yml 仍未修**：`Experiments/checkpoint/Falcon/conf.yml` 的 `audionet_name` 仍是
   `SpeakerContrastiveV2`（F3 属训练/配置侧）；本脚本现在会给出明确错误与修复提示，但不会自动改写。
7. **指标口径仍非论文确认版**：SI-SDR 用零均值 + PIT；SDR 用 fast_bss_eval 512-tap BSS-Eval-v4 风格；
   论文若使用不同 SDR 实现/窗长/通道策略，数值不可直接对齐（需在 Phase2 报告中注明协议）。

## 6. 复跑命令汇总

```bash
ENV=/root/miniforge3/envs/falcon/bin/python
cd /root/code/github_repos/Falcon-fork

# 单元测试
$ENV -m pytest tools/repro/tests/test_metrics.py -q
$ENV -m pytest tools/repro/tests/test_eval_entry.py -q

# 冒烟（需官方权重；生成 1 条合成样本后从临时 cwd 跑）
$ENV tools/repro/tests/make_t3_smoke_data.py --out /tmp/t3_smoke/data --seconds 2 --n 1
mkdir -p /tmp/t3run && cd /tmp/t3run
PYTHONPATH=/root/code/github_repos/Falcon-fork $ENV \
  /root/code/github_repos/Falcon-fork/audio_visual_test.py \
  --conf_dir /root/code/github_repos/Falcon-fork/configs/Falcon.yml \
  --data_dir /tmp/t3_smoke/data \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt --gpus 0 --limit 1
```

产物：`docs/verification/phase1/{T3-metrics-tests,T3-eval-entry-tests,T3-combined-tests,T3-eval-smoke,T3-import-check}.log`、
`T3-eval-smoke-metrics.csv`、`T3-smoke-fixture-mix.json`、本报告。
