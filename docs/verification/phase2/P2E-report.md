# Phase 2-E · RealSSA 评测预置（DNSMOS + RealSSA-Real 管线 + Sim 评测命令）

> 任务：`task-22`（owner `ops-tests-docs`，状态 in_progress，未 complete）；分支 `repro/phase2`（HEAD `af74cca`）。
> 环境：conda `falcon`（`/root/miniforge3/envs/falcon/bin/python`，Python 3.11.17，torch 2.8.0+cu128）。
> 约束遵守：未改 Phase1 源码、未 commit；新增文件仅在 `tools/repro/phase2/` 与 `docs/verification/phase2/`；
> RealSSA-Sim/Real 仍 gated（申请已提交、待批，数据文件 403），**本任务不伪造任何 RealSSA 指标**。
> 关联：`docs/DATA_ACCESS_GUIDE.md` §4/§5、`docs/verification/phase1/dataset-cards/RealSSA-{Sim,Real}-README.md`、
> `docs/REPRODUCTION_PLAN.md` §4/§5.4/§7、`tools/repro/realssa_to_falcon.py`。

---

## 0. 摘要

| 交付 | 状态 | 证据 |
|---|---|---|
| DNSMOS P.835 离线实现（speechmos/ONNX）装入 conda falcon | ✅ | `P2E-env-delta.txt`；`P2E-dnsmos-synthetic.log/.json` |
| `tools/repro/phase2/eval_dnsmos.py`（wav → SIG/BAK/OVRL/P808，合成音频冒烟） | ✅ | 合成 4 例 PASS；LRS2 2 例实测 |
| `tools/repro/phase2/eval_realssa_real.py`（54 段 → 整段分离 → DNSMOS CSV；`--limit/--channel/--device`） | ✅ 预置完成；真实数据待审批 | `P2E-smoke-realssa-real.log/.csv/.json`；官方 loader adapter 用 mock API 验证 |
| RealSSA-Sim tt 转换+评测命令（复用 `realssa_to_falcon.py`，1 shard dry-run） | ✅ dry-run 通过（403 状态明确） | `tools/repro/phase2/realssa_sim_tt_pipeline.sh`；`P2E-sim-pipeline.log` |
| 审批后执行清单/风险 | ✅ | 本报告 §5/§6 |

**结论**：工具链已可离线运行并通过接口冒烟；审批通过后按 §5 执行即可出 DNSMOS（RealSSA-Real）与 SI-SNRi/SDRi（RealSSA-Sim）。
RealSSA-Real **只能算 DNSMOS**（无干净 target，不能算 SI-SDR）；DNSMOS 目标参考值为 **3.276 ±0.05**（`REPRODUCTION_PLAN.md` §4 表）。

---

## 1. DNSMOS 实现选型与依赖安装

### 1.1 选型对比

| 候选 | 是否离线 | 许可 | 结论 |
|---|---|---|---|
| **PyPI `speechmos==0.0.1.1`**（Microsoft AEC/DNS/PLC organizers） | ✅ wheel **自带** DNSMOS P.835 + P.808 ONNX 模型（无需运行时下载） | **MIT** | **采用** |
| `torchmetrics==1.9.0` 的 `DeepNoiseSuppressionMeanOpinionScore`（Apache-2.0） | ❌ 首次运行从 `raw.githubusercontent.com/microsoft/DNS-Challenge` 下载 3 个 onnx 模型（`~/.torchmetrics`），沙箱网络不稳/非离线 | Apache-2.0 | 不采用（仅作备选记录） |
| 微软 DNS-Challenge 原始 ONNX 手工接入 | ⚠️ 需自行落盘模型 | MIT（DNS-Challenge） | 由 speechmos 覆盖，无需重复 |

### 1.2 已安装（conda falcon，2026-10-21）

```bash
/root/miniforge3/envs/falcon/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "onnxruntime==1.31.0"
/root/miniforge3/envs/falcon/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps "speechmos==0.0.1.1"
```

| 包 | 版本 | 许可 | 来源/备注 |
|---|---|---|---|
| speechmos | 0.0.1.1 | MIT（Microsoft） | wheel 9,439,732 B，自带 `speechmos/dnsmos_models/{sig_bak_ovr,model_v8,sig,bak_ovr}.onnx` |
| onnxruntime | 1.31.0 | MIT | CPU 版；DNSMOS 推理后端 |
| protobuf | 7.36.2 | BSD-3-Clause | onnxruntime 依赖（新增） |
| flatbuffers | 25.12.19 | Apache-2.0 | onnxruntime 依赖（新增） |

模型文件（可复现性，sha256 见 `P2E-dnsmos-synthetic.json`）：
- `sig_bak_ovr.onnx` 1,157,965 B sha256 `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`（P.835 主模型）
- `model_v8.onnx` 224,860 B sha256 `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91`（P.808）
- 另含 `sig.onnx` / `bak_ovr.onnx`（AEC 相关，未用）

> ⚠️ **环境变更合规**：`AGENTS.md §4` 要求安装后更新 `docs/verification/conda-env-lock.txt` 与 AGENTS §2 包清单；
> 本 task 写 scope 仅 `tools/repro/phase2` + `docs/verification/phase2`，故把精确 delta 写入 **`docs/verification/phase2/P2E-env-delta.txt`**，
> 请 Lead/Phase2-QA 合并进 lock/AGENTS（否则新机器重建环境会缺 4 个包）。

### 1.3 DNSMOS 协议（写入报告时必须声明）

- 模型：DNSMOS **P.835**（SIG/BAK/OVRL）+ P.808；speechmos 内部按 **9.01 s 窗 / 1 s hop** 分段、逐段推理后取均值；输入要求 **16 kHz、[-1,1] 浮点**。
- 非个性化模式（`model_type="dnsmos"`）；论文 RealSSA 未说明 personalized，默认标准 P.835。
- 输出：`sig_mos / bak_mos / ovrl_mos / p808_mos`；本工具额外记录 `duration_s/elapsed_s`。
- **参考无关指标**：RealSSA-Real 无干净 target → 只能报 DNSMOS；不要与 SI-SDR/SI-SNRi 混表。

---

## 2. `tools/repro/phase2/eval_dnsmos.py`

```bash
# 合成音频自检（离线 ONNX 链路，不依赖数据）
python tools/repro/phase2/eval_dnsmos.py --synthetic --synthetic-log docs/verification/phase2/P2E-dnsmos-synthetic.json

# 单文件/目录 → CSV（16 kHz；非 16k 加 --resample；stereo 自动取均值）
python tools/repro/phase2/eval_dnsmos.py --input est.wav --out dnsmos.csv --summary-json dnsmos_summary.json
python tools/repro/phase2/eval_dnsmos.py --input /path/to/est_dir --recursive --out dnsmos.csv
```

- CSV 列：`file,duration_s,sig_mos,bak_mos,ovrl_mos,p808_mos,elapsed_s,error`；汇总 JSON 含均值/标准差/版本/模型 sha256。
- 可被 `eval_realssa_real.py` 以 `from eval_dnsmos import score_array` 复用（同一进程内 ONNX session 复用）。
- 首次调用约 20–30 s（session 初始化），之后 2 s 音频约 3–4 s（含 speechmos 的 9 s 窗重复填充）。

**合成冒烟结果**（`P2E-dnsmos-synthetic.log`，PASS；只验证链路，不代表任何数据/论文数值）：

| case | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| synthetic_harmonic | 1.496 | 1.254 | 1.337 | 2.182 |
| synthetic_harmonic_noise | 1.258 | 1.045 | 1.081 | 2.173 |
| synthetic_silence | 2.514 | 3.472 | 1.840 | 2.147 |
| synthetic_noise | 1.137 | 1.095 | 1.109 | 2.076 |

LRS2-2Mix 2 条真实混合音频（`P2E-dnsmos-lrs2-smoke.csv`）：SIG 2.40/3.03，BAK 1.51/1.59，OVRL 1.52/1.55，P808 2.77/2.55（2.0 s 文件）。

---

## 3. `tools/repro/phase2/eval_realssa_real.py`

### 3.1 读取/对齐（严格按 dataset card 约定）

- 数据根：`--data /root/data/realssa_real`（或 `$REALSSA_REAL_DIR`）；只读 `inference` view（`audio/inference/**/*.wav`、`mouth/**/*.npz`、`metadata/inference_segments.jsonl`）。
- **官方 loader 优先**：若 `<data>/scripts/load_inference.py` 存在，`import` 后优先使用官方 `read_recordings(root)` + `load_sample(root, index=…/inference_id=…, align_tail=…, channel=…)`；
  失败/不存在时回退到内置 loader（metadata jsonl 或目录扫描），CSV/metadata 的 `loader_kind` 会标明来源。
- `align_tail=True`（默认）：`T = min(mouth_frames, floor(audio_samples/640), landmark_frames)`；音频保留前 `T*640` 样本、mouth 保留前 `T` 帧；记录 `discarded_samples`；若丢弃 >70 ms+1 帧则 warning（card 规定 native 音频最多比 mouth 长 70 ms）。
- `--channel mean|0|1`：stereo native 音频取均值 / 左 / 右；**默认 mean**（card 明确 channel 策略需报告）。
- 视觉：mouth `uint8 [T,96,96]` 直接送 Falcon（与训练/Studio 相同 96→88 center crop）；若误给 landmark 形状会报清晰错误；RGB 4D 自动转 luma。
- 分离：`FalconRuntime.from_pretrained("JusperLee/Falcon"|"JusperLee/Falcon-Big", device).load()` + `infer_file.extract(runtime, audio, mouth)`（整段推理；模型输入按 2 s 窗口因果前向，输出与 mouth 帧对齐）。
- 指标：对 estimate 算 DNSMOS；`--baseline` 同时算 mixture DNSMOS（同段对照）。**不计算 SI-SDR**（无 clean target）。
- 输出：`<out-dir>/realssa_real_dnsmos_per_segment.csv`（逐段：id/recording/scene/channel/align/audio_s/mouth_frames/used_frames/used_s/discarded/inference_ms/rtf/4×DNSMOS[/4×mix_DNSMOS]/error）、
  `summary.json`（整体 + 分场景 mean/std）、`run_metadata.json`（命令/版本/模型/loader/是否 smoke）。
- 安全：默认不保存分离音频；`--save-estimates` 才写 `<out-dir>/estimates/`（仍属参与者数据，须在获批范围内使用/删除）。

### 3.2 CLI

```bash
# 无数据：清晰等待提示（exit 2），不会写任何结果
python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real --dry-run

# 审批通过、下载后（建议先 --limit 5 --dry-run 看清单，再小样本，再全量）
python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real --limit 5 --dry-run
python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real \
    --channel mean --device cuda --baseline --limit 5 \
    --out-dir /root/data/falcon_eval/realssa_real_mean_pilot
python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real \
    --channel 0 --device cuda --baseline --out-dir /root/data/falcon_eval/realssa_real_ch0
```

### 3.3 接口冒烟（无数据时；不伪造 RealSSA 结果）

| 冒烟 | 命令 | 结果 |
|---|---|---|
| 合成音频 + fallback loader + DNSMOS（`--skip-model`） | `--smoke-synthetic --limit 1 --skip-model` | ✅ CSV `P2E-smoke-synthetic-per-segment.csv`：used=2.00 s，SIG=1.971/BAK=2.829/OVRL=1.646 |
| **LRS2-2Mix 构造 mock RealSSA-Real 目录 + 真实 falcon.pt 分离 + DNSMOS** | `--smoke-lrs2 /root/data/lrs2_2mix/lrs2_rebuild --limit 2 --baseline` | ✅ `P2E-smoke-realssa-real.log/.csv/.json`：loader=fallback:metadata-jsonl；model load 17.2 s (cuda)；2 段均 used=2.00 s；OVRL **2.364/2.128**（mixture 基线 **1.523/1.547**） |
| **官方 loader adapter**（mock `scripts/load_inference.py` 实现 card API） | `--data <mock> --limit 1 --baseline` | ✅ `P2E-smoke-official-loader.log`：loader=**official:read_recordings**，SIG/BAK/OVRL/P808 = 3.2677/2.7496/2.3637/3.1704，与 fallback 路径逐位一致 |
| loader 清单 dry-run | `--data <mock> --dry-run` | ✅ `P2E-smoke-loader-dryrun.log`：列出 2 段、不加载模型 |
| `--channel 0` 路径 | `--smoke-synthetic --limit 1 --skip-model --channel 0` | ✅ 通过 |

> ⚠️ 以上 LRS2 mock 结果只是**接口验证**，不是 RealSSA-Real 数据、不代表论文 3.276；CSV 文件名带 `SMOKE_` 前缀、metadata 标注 `is_smoke=true`。
> 官方 `load_sample` 的**真实**返回结构仍需在数据获批后首次运行时核对（当前用 mock 覆盖了 `read_recordings`+`load_sample(inference_id/index, align_tail, channel)` 的调用约定）。

---

## 4. RealSSA-Sim tt 转换 + 评测命令（复用 `tools/repro/realssa_to_falcon.py`）

已提供 `tools/repro/phase2/realssa_sim_tt_pipeline.sh`（默认 dry-run；`--execute` 才真正执行）：

```bash
# dry-run：只读检查 gated 授权 + 打印完整命令（不下载/不转换/不评测）
bash tools/repro/phase2/realssa_sim_tt_pipeline.sh

# 审批通过后：1 个 tt shard（tt-000000，≈2.33 GB）→ 转换 → --limit 20 评测
bash tools/repro/phase2/realssa_sim_tt_pipeline.sh --execute

# 可选参数
SHARD=tt-000005 CHANNEL=0 LIMIT=5 bash tools/repro/phase2/realssa_sim_tt_pipeline.sh --execute
FULL=1 bash tools/repro/phase2/realssa_sim_tt_pipeline.sh --execute   # tt 全量 19 shards ≈40.1 GB
```

脚本内部命令（等价手工执行）：

```bash
ENV=/root/miniforge3/envs/falcon/bin/python
# 0) 授权只读检查（当前预期 403 → 等审批）
$ENV tools/repro/fetch_data.py --dataset realssa-sim --shard tt-000000 --dry-run --proxy http://127.0.0.1:17897

# 1) 下载 1 个 test shard（获批后）
$ENV tools/repro/fetch_data.py --dataset realssa-sim --shard tt-000000 --out /root/data/realssa_sim_raw --proxy http://127.0.0.1:17897

# 2) 转换（4ch→mono；mouths.pt 直用；meta→near.json fallback；含 mix/target/near JSON）
$ENV tools/repro/realssa_to_falcon.py --src /root/data/realssa_sim_raw/tt/tt-000000.tar \
    --out /root/data/realssa_falcon --splits tt --channel mean
#    可先 --dry-run 检查配对/计数（不写盘）

# 3) 小样本评测（官方权重；同通道 target → SI-SNRi/SDRi；--limit 20）
PYTHONPATH=$PWD $ENV audio_visual_test.py --conf_dir configs/Falcon.yml \
    --data_dir /root/data/realssa_falcon/tt --ckpt_path /root/.cache/falcon_weights/falcon.pt \
    --gpus 0 --limit 20

# 4) tt 全量（19 shards）：下载 --pattern 'tt/*.tar'，转换 --src /root/data/realssa_sim_raw --splits tt，
#    评测去掉 --limit；论文目标 SI-SNRi 7.7 / SDRi 8.2（±0.3），ch0 与 mean 两种通道都要记录。
```

**当前 dry-run 结果**（`P2E-sim-pipeline.log`）：`gated 授权检查 ❌ 未获批（GatedRepoError 403）`；1 shard 匹配 `tt/tt-000000.tar` 2.33 GB；
后续命令已打印但未执行。**待人工审批后执行**（不伪造结果）。

---

## 5. 审批通过后的执行清单（可直接照做）

1. **确认授权**：`hf auth login`（账号 `nanless`）→ `python tools/repro/fetch_data.py --dataset realssa-sim --dry-run` 应显示已获批（非 403）。
2. **RealSSA-Sim**：
   - `bash tools/repro/phase2/realssa_sim_tt_pipeline.sh --execute`（1 shard → 转换 → `--limit 20` 评测）先验证链路；
   - 通过后 `FULL=1 bash ... --execute`（tt 全量 19 shards）→ `--channel mean` 与 `--channel 0` 两套 SI-SNRi/SDRi；
   - 报告写：channel、align（converter 默认 min 对齐）、ckpt、eval 命令、与论文 7.7/8.2 对比。
3. **RealSSA-Real**：
   - `python tools/repro/fetch_data.py --dataset realssa-real --full`（≈12.84 GB）；
   - `python tools/repro/phase2/eval_realssa_real.py --data /root/data/realssa_real --limit 5 --dry-run` 核对 54 段清单；
   - `--channel mean` / `0` / `1` 各跑一遍（`--baseline` 留 mixture 对照），输出 CSV+summary；
   - 报告写：DNSMOS 版本/模型 sha256、channel、align_tail、模型与 ckpt、逐段/分场景/整体均值；与论文 **3.276 ±0.05** 对比；**不得**出现 SI-SDR。
4. **环境合规**：把 `P2E-env-delta.txt` 的 4 行合并进 `docs/verification/conda-env-lock.txt` + AGENTS §2（AGENTS §4 要求）。
5. **回归**：跑一次 `pytest studio/tests`（35 passed/2 skipped 基线）与 `eval_dnsmos --synthetic`，确认安装未破坏环境。

---

## 6. 风险 / 限制 / 文档一致性

| 项 | 说明 | 处理 |
|---|---|---|
| RealSSA 数据仍 403 | 申请已提交、待人工审批（GUIDE §4/§5）；无法在本任务内产生真实数值 | 本报告只做接口冒烟；真实指标待批后按 §5 跑 |
| RealSSA-Real 无 SI-SDR | 无 clean target，enrollment 不是时间对齐参考（card 明确） | 只报 DNSMOS（+可选 WER/speaker-sim）；目标 3.276 |
| DNSMOS 版本敏感 | speechmos 内置 DNS-Challenge P.835 模型；论文官方 DNSMOS 版本未说明 | 报告固定 `speechmos 0.0.1.1` + 模型 sha256；跨版本不可直接比 |
| 官方 loader 适配 | 真实 `scripts/load_inference.py` 的返回字段未能在数据 403 时验证；已用 mock API 覆盖调用约定 | 首次真实运行时核对 `loader_kind` 与 CSV；若字段不同，按 `load_sample` 返回 dict 归一化逻辑小改即可 |
| 计算成本 | RealSSA-Real 54 段共 83.99 min（avg ~93 s/段）；分离 RTF ~0.06–0.1，DNSMOS 9.01 s/1 s hop 约 30–40 s/段（CPU onnxruntime）；`--baseline` 翻倍 | 先 `--limit 5`；全量预计 ~30–60 min；必要时去掉 baseline/分批 |
| 通道/对齐口径 | 论文未声明 channel 与整段/窗口口径 | 三种 channel 都记录；align_tail=True 默认；报告写清 |
| `docs/DATA_ACCESS_GUIDE.md` 状态行不一致 | 第 18 行写“申请已提交待批”，第 39 行仍写“未提交任何访问申请”（Phase1 残留）；不在本 task 写 scope | 建议 Lead/QA 修 GUIDE 第 39 行（避免读者误解） |
| 环境 lock/AGENTS 未更新 | AGENTS §4 要求安装后更新 lock + §2；本 task 写 scope 不含这两处 | 见 `P2E-env-delta.txt`；请 Lead/QA 合并 |

---

## 7. 证据文件索引（均在 `docs/verification/phase2/`）

- `P2E-dnsmos-synthetic.log/.json`：合成音频 4 例 + 模型 sha256 + 版本（PASS）。
- `P2E-dnsmos-lrs2-smoke.csv/.json`：LRS2 2 条真实音频的 DNSMOS 冒烟。
- `P2E-smoke-realssa-real.log/.csv/.json/.metadata`：LRS2 mock RealSSA-Real + 真实 falcon.pt 整段分离 + DNSMOS（SMOKE，非数据集结果）。
- `P2E-smoke-official-loader.log`：mock 官方 `load_inference.py` API → official adapter 验证。
- `P2E-smoke-loader-dryrun.log`：mock 目录 loader dry-run 清单。
- `P2E-smoke-synthetic-per-segment.csv`：合成 + fallback loader + DNSMOS（--skip-model）。
- `P2E-sim-pipeline.log`：RealSSA-Sim tt pipeline dry-run（403 状态 + 完整命令）。
- `P2E-env-delta.txt`：新增依赖 freeze 行/安装命令/许可/来源（待并入 lock）。
- 工具：`tools/repro/phase2/{eval_dnsmos.py,eval_realssa_real.py,realssa_sim_tt_pipeline.sh}`。
- 原始输出（仓库外）：`/root/data/falcon_eval/_smoke_lrs2/`、`/root/data/falcon_eval/_smoke_official/` 等。

## 8. 验收核对（task-22）

- ✅ DNSMOS 可离线实现已装入 conda falcon，来源/许可/版本/模型 hash 已记录；`--synthetic` PASS。
- ✅ `eval_dnsmos.py`：wav→SIG/BAK/OVRL/P808，CSV/JSON，合成+LRS2 冒烟。
- ✅ `eval_realssa_real.py`：54 段约定（official loader 优先 + fallback）、align_tail/channel/limit/device/baseline、逐段+汇总 CSV、无数据清晰提示、LRS2/合成接口冒烟、不伪造 RealSSA 结果。
- ✅ RealSSA-Sim tt 转换+评测命令（复用 `realssa_to_falcon.py`），1 shard dry-run 通过并明确“待审批后执行”。
- ✅ 报告+日志齐备；未改 Phase1 源码、未 commit；task-22 暂不 complete（等 Lead 验收）。

## 9. 2026-10-24 状态更新（终态）

- **RealSSA-Sim / RealSSA-Real 已于 2026-10-22 获批并下载**：Sim 521,532,645,652 B / 242 文件（240 shard），
  Real 12.84 GB；已 rsync 归档到 TOS `/aispeech-cfs/voiceprint/data/multimodal/`（`_archive_done.json` 两条 `ok:true`）。
  本报告 §0/§5 的“403/待审批”描述作废。
- **RealSSA-Sim test（9,169 条，channel=mean，整段推理，官方 falcon.pt）**：SI-SNRi **7.774307** / SDRi **8.479395** /
  PESQ 2.093770 / STOI 0.817290（论文 7.7 / 8.2）。证据：`P2E-realssa-sim-eval-official.log`、
  `P2E-realssa-sim-metrics_combined.csv`（9,171 行=9,169 逐句+avg/std）、`P2E-realssa-sim-convert.log`。
- **RealSSA-Real（54 段，channel=mean，align_tail=True，窗口化分离后整段 DNSMOS）**：
  SIG 3.2209 / BAK 3.6921 / OVRL 2.7901 / P808 3.3285；mixture 基线 2.4176 / 1.7908 / 1.7599 / 2.6765。
  证据：`DNSMOS-realssa-real-mean-full3/{summary.json,run_metadata.json,per_segment.csv}`。
- **论文 DNSMOS 3.276 的列名仍未确认**：按 P808 为 +0.05，按 OVRL 为 −0.49；报告必须同时给两列并注明口径。

> **2026-10-30 论文公开后校正（arXiv:2610.32749）**：论文的 “Real-world DNSMOS 3.276” 引用 **DNSMOS P.835**（不是 P.808），
> 且该数字对应 **RealSSA-trained ckpt**；我们用的官方 `falcon.pt` 在同一数据上 P.835 OVRL = **2.7901**，
> 与 3.276 **不可直接比较**（P808 3.3285 只是数值巧合）。另：论文 RealSSA-Sim **只用 channel-1**（代码 `--channel 0`），
> 我们此前用 4ch mean；ch0 复评已排期。详见 `docs/understanding/PAPER_ALIGNMENT_20261030.md`。

---

## 10. 2026-10-31 A4-1 修复后重跑（full4，**替换 full3**）

### 10.1 bug 与修复（review A4-1）

- **位置**：`tools/repro/phase2/eval_realssa_real.py` 的 `_extract_windowed()`（旧 `:405,424`）。
- **旧行为**：`endpoint = LEFT + HOP`（=30,720），而每个窗输出的是 `pred[LEFT:LEFT+HOP]`，对应绝对
  `[endpoint-GUARD-HOP, endpoint-GUARD)`。因此首窗写 `[21,440, 29,440)`，**`[0, 21,440)`（21,440 样本 = 1.34 s@16k）
  永远不被任何窗口写入**，每条估计的音频开头恒为静音。已发布的 full3 四指标即建立在该输出上。
- **修复**：首窗 `endpoint = HOP + GUARD`（=9,280），与 Studio 语义对齐——`studio/web/app.js:624`
  首窗 `end = sessionEpoch + HOP + GUARD`、`studio/server.py:266` 发出
  `start_sample = endpoint-GUARD-HOP, end_sample = endpoint-GUARD`；此时 `abs_start = endpoint-(WINDOW-LEFT) = 0`。
- **覆盖率断言**（新增）：循环内用 `filled[np]` 记录每个写入样本；循环后若 `not filled.all()`，压缩未覆盖区间并
  `raise RuntimeError("windowed inference coverage failure: … gaps(sample[start,end))=[…]")`；
  整段输出必须 100% 被写，不允许静默留白。

### 10.2 修复验证（CPU/GPU 两路）

- 合成探针（fake runtime，`_extract_windowed` 直接调用）：n=16,000 / 19,200 / 20,000 / 40,000 样本全部
  **coverage=100%**、输出长度==输入长度；同一循环用旧初值 `LEFT+HOP` 时未覆盖恰为
  `21,440/64,000` 样本、区间 `[0, 21440)`（即 1.34 s 静音前缀），新初值 `HOP+GUARD` 未覆盖 0 样本。
- 真实数据：full4 全 54 段运行中覆盖率断言**全部通过**（任何一段 <100% 都会 raise 并中止）；
  首段估计前 21,440 样本 `nonzero_frac=1.000`（旧代码为 0），首 1 s `mean|x|=0.0204`。
- **mixture 基线逐位不变**（见下表 4 个 mix_* delta=+0.0000）→ 证明差异只来自分离输出修复，DNSMOS 侧无漂移。

### 10.3 full3（作废） vs full4（有效）

**结论：旧 full3 的 SIG/BAK/OVRL/P808 因首块 1.34 s 静音前缀作废，引用/图表一律改用 full4；full3 仅保留为历史
证据（`…full3/`），不得再作为“官方权重 RealSSA-Real DNSMOS”引用。**

| 指标 | full3（旧，作废） | **full4（新，有效）** | Δ（new−old） | n |
|---|---|---|---|---|
| SIG | 3.2209 ± 0.3691 | **3.2226 ± 0.3703** | +0.0017 | 54/54 |
| BAK | 3.6921 ± 0.2640 | **3.6981 ± 0.2598** | +0.0060 | 54/54 |
| OVRL | 2.7901 ± 0.4082 | **2.7941 ± 0.4100** | +0.0040 | 54/54 |
| P808 | 3.3285 ± 0.4045 | **3.3365 ± 0.4066** | +0.0080 | 54/54 |
| mixture SIG | 2.4176 | 2.4176 | +0.0000 | 54/54 |
| mixture BAK | 1.7908 | 1.7908 | +0.0000 | 54/54 |
| mixture OVRL | 1.7599 | 1.7599 | +0.0000 | 54/54 |
| mixture P808 | 2.6765 | 2.6765 | +0.0000 | 54/54 |

分场景 OVRL（full3 → full4）：big_indoor 2.5934 → **2.6024**；mid_indoor 2.3942 → **2.4069**；
outdoor 2.7766 → **2.7608**；small_indoor 3.0958 → **3.1067**（其余分场景见 `summary.json`）。

**幅度解释**：单段整段 DNSMOS 用 9.01 s 窗/1 s hop，1.34 s 静音只影响首窗约 15% 的内容，且静音在 P.835 中
BAK/OVRL 上被平滑 → 均值只移动 +0.002~+0.008 MOS；但**输出本身错误（每条开头 1.34 s 静音）是确定性的**，
因此必须重跑/替换，不能以“数值几乎不变”为由沿用 full3。

### 10.4 复跑命令与产物

```bash
cd /root/tmp/p2e_real_dnsmos_full4        # 独立 cwd（不在仓库内）
OUT=/root/code/github_repos/Falcon-fork/docs/verification/phase2/DNSMOS-realssa-real-mean-full4-20261031
PYTHONPATH=/root/code/github_repos/Falcon-fork:/root/code/github_repos/Falcon-fork/studio:/root/code/github_repos/Falcon-fork/tools/repro/phase2 \
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/falcon/bin/python \
  /root/code/github_repos/Falcon-fork/tools/repro/phase2/eval_realssa_real.py \
  --data /root/data/RealSSA-Real --channel mean --align-tail --device cuda \
  --model JusperLee/Falcon --baseline --out-dir "$OUT"
```

- 产物：`DNSMOS-realssa-real-mean-full4-20261031/{summary.json,run_metadata.json,realssa_real_dnsmos_per_segment.csv,P2E-realssa-real-full4.log}`
- `run_metadata.json`：timestamp `20261030T162555Z`、channel=mean、align_tail=true、device=cuda、
  speechmos 0.0.1.1 / onnxruntime 1.31.0、baseline=true、segments 54/54/0、used_s 合计 5,037.6 s（=83.96 min）、RTF 均值 0.102（median 0.0985，GPU 0）。
- 备注：`--save-estimates` 本轮未开（该开关对 `inference_id` 含 `/` 的嵌套 id 会因父目录不存在而报错，
  属既有 P3 问题，不影响指标）；覆盖率证据由 §10.2 的合成探针 + 运行期断言提供。

### 10.5 附带：评测 CSV 增加 mixture 基线列与有效计数（A4-3 / A4-4）

- `look2hear/metrics/allwrapper.py`：`CSV_COLUMNS` **只在末尾追加**（既有 7 列顺序/位置不变）：
  `si-snr_mix, sdr_mix, n_sdr, n_sdr_i, n_si-snr, n_si-snr_i, n_pesq, n_stoi, n_si-snr_mix, n_sdr_mix`。
  逐句行填 8 个新值列、`n_*` 留空；`avg`/`std` 行的 `n_*` 为真实有效计数（不再“静默 nanmean”）。
- `audio_visual_test.py::_merge_results`：合并时按列统计 finite 值写 `metrics_counts.json`
  （`counts`/`nan_counts`/`n_utterances`/`si_snr_identity_check`），并在 avg/std 行回填 `n_*`。
- 重跑 LRS2-2Mix test 全量 3,000 条（ep99 与官方 `falcon.pt` 各一次；GPU0，独立 cwd `/root/tmp/p2d_eval_*_v4`）：

| run | 旧列 avg（sdr / sdr_i / si-snr / si-snr_i / pesq / stoi） | 新列 avg（si-snr_mix / sdr_mix） | 逐句旧列 max\|Δ\| |
|---|---|---|---|
| LRS2 ep99 | 14.222311 / 9.121848 / 13.719868 / 8.726148 / 2.577971 / 0.943988（与旧 CSV **一致**） | **4.993720 / 5.100463** | sdr 3.3e-7、si-snr 3.3e-7、pesq 4.3e-4、stoi 1.8e-8 dB/MOS |
| LRS2 官方 falcon.pt | 12.813387 / 7.712925 / 12.022821 / 7.029101 / 2.195634 / 0.916350（与旧 CSV **一致**） | **4.993720 / 5.100463** | sdr 2.2e-7、si-snr 2.3e-7、pesq 9.4e-5、stoi 3.9e-8 dB/MOS |

- 离线恒等式：`max|si-snr − si-snr_i − si-snr_mix| = 8.9e-15`、
  `max|sdr − sdr_i − sdr_mix| = 8.9e-15`（3000/3000 句）→ SI-SNRi/SDRi 可离线复核（A4-4 闭环）。
- 有效计数：两 run 均 `n=3000/3000`（6 个旧列 + 2 个新列全部 finite，0 nan）。
- 证据：`P2D-eval-epoch99-cols-20261031.{csv,log,counts.json}`、
  `P2D-eval-falcon-full-cols-20261031.{csv,log,counts.json}`（旧 6 列 CSV 保留不动，便于对拍）。


