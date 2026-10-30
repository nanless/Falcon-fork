# P2E · RealSSA-Sim 通道敏感性（ch0 vs mean）· 转换完成 / 评测推迟

> 任务：`task-13`（P1 通道敏感性，owner `model-core`）· 日期 2026-10-24
> 环境：conda `falcon`（`/root/miniforge3/envs/falcon/bin/python`，Python 3.11.17，torch 2.8.0+cu128）
> 约束遵守：未 git commit、未改源码/他人文档、未联网、未下载数据、未 kill 任何进程；写作用域仅
> `docs/verification/phase2/P2E-realssa-sim-ch0-*`；未写入/覆盖 `/root/data/falcon_realssa_sim`（mean 版）。
> 关联：`tools/repro/realssa_to_falcon.py`、`docs/verification/phase2/P2E-report.md`、`docs/REPRODUCTION_PHASE2_REPORT.md`。

---

## 0. 摘要（状态一表）

| 项 | 状态 | 证据 |
|---|---|---|
| RealSSA-Sim tt → ch0 转换（9169 条） | ✅ **完成** | `P2E-realssa-sim-ch0-convert.log`（exit 0） |
| 转换自检（条数/通道记录/对齐/near fallback/id 与 mean 一致） | ✅ **通过** | 本报告 §3 |
| 官方权重 `falcon.pt` 在 ch0 上全量评测（9169 条） | ⏸ **推迟（未完成，无指标）** | 见 §4；`P2E-realssa-sim-ch0-eval.log` 已标注 DEFERRED |
| 与 mean 基线（SI-SNRi 7.774 / SDRi 8.479）的 Δ | ⏸ 待评测后计算 | §5 给出判定规则与脚本思路 |
| 产物落盘 | ✅ 转换日志 + 本报告 | `P2E-realssa-sim-ch0-{convert.log,eval.log,report.md}` |

一句话：**ch0 数据已就绪并通过自检；评测被 Lead 的 GPU 统一排期中止（P0 LRS2 全量评测占满 2 卡），本轮不排，指标留待统一排期后重跑。**

---

## 1. 背景与通道口径

- RealSSA-Sim 的 mixture / target 是 **4 声道 PCM16 16 kHz**；论文与 README **未声明** 4ch→mono 的通道协议。
- mean 基线（已有）：`/root/data/falcon_realssa_sim`，4 通道算术平均，官方 `falcon.pt` 全量 test 9169 条 =
  **SI-SNRi 7.774 / SDRi 8.479**（`P2E-realssa-sim-metrics_combined.csv` 的 `avg` 行，本轮复核一致）。
- 本任务：用**同一转换脚本、同一对齐策略、只换 `--channel 0`**（mixture 与 target 用同一通道），量化通道选择的影响。
- 判定规则（Lead 指定）：若 `|Δ| > 0.3 dB`，报告中必须写明“**通道协议必须显式声明**”，否则认为通道策略影响可忽略。

---

## 2. 转换执行

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/realssa_to_falcon.py \
  --src /root/data/RealSSA-Sim --out /root/data/falcon_realssa_sim_ch0 \
  --splits tt --channel 0 2>&1 | tee docs/verification/phase2/P2E-realssa-sim-ch0-convert.log
# exit=0（转换约 12 min CPU，19 个 shard 全量读取）
```

转换日志尾部（原样摘录）：

```text
[realssa_to_falcon] src=/root/data/RealSSA-Sim out=/root/data/falcon_realssa_sim_ch0 channel=0 align=True dry_run=False shards=19
[realssa_to_falcon] split tt: 19 shard(s)
  tt: converted 9169 sample(s); skipped incomplete=0, errors=0
[realssa_to_falcon] building mix/target/near JSON via DataPreProcess for ['tt']
  tt: mix.json=.../tt/mix.json target.json=.../tt/target.json near.json=.../tt/near.json (9169 samples)
  Wrote 9169 entries to .../tt/near.json  (missing/fallback: 9169, near_per_sample min=1 max=1 avg=1.00)
[realssa_to_falcon] NOTE: RealSSA-Sim ships no near-speaker waveforms ... near.json entries fall back to
  mixture sentinels -> the dataset skips them -> K=0 -> the NASS contrastive loss stays inactive.
[realssa_to_falcon] done. Point the DataModule at: train_dir=.../tr valid_dir=.../cv test_dir=.../tt
```

- 磁盘：输出 **30 GB**（`tt/mouths` 28 GB、`tt/mix` 1.4 GB、`tt/target` 1.3 GB）；源 `RealSSA-Sim/tt` 38 GB 只读。
- 未使用 `--limit-*`、未用 `--overwrite`（输出目录预先不存在），因此是**全量 19 shards / 9169 条**。

---

## 3. 转换产物自检（本轮实测，全部通过）

| 检查项 | 命令/方法 | 结果 |
|---|---|---|
| 索引条数 | `json.load(tt/{mix,target,near}.json)` | mix=**9169**、target=**9169**、near=**9169** |
| 文件数 | `glob` 计数 | mix `\*.wav`=9169 + `\*_meta.json`=9169；target wav=9169；mouths `.pt`=9169 |
| 通道记录 | 扫描全部 9169 个 `_meta.json` 的 `provenance.channel_policy` | `{'0': 9169}`（全部为 `"0"`；mean 版为 `"mean"`） |
| 帧对齐 | `_meta.json` 的 `provenance.aligned_frames` | min=25 / max=188 / mean=115.19；`samples_per_frame=640` |
| 采样数对齐 | 抽 50 条 `mix.wav` 的 `frames % 640` | 0 条不整除（与 mouth 帧对齐成立） |
| 通道确已生效 | 同 id 对比 ch0 vs mean 的 `mix.wav` | 3 条抽样 `max|Δ|`=0.1645/0.2043/0.5186，长度相同 ⇒ 不是同一波形 |
| near fallback | `near.json` 逐条检查路径 | 9169/9169 全部指向 `.../mix/<id>.wav`（mixture sentinel；RealSSA-Sim 不发布 near 波形）⇒ 评测/训练时 K=0 |
| 与 mean 可比性 | ch0 vs mean 的 9169 条 id 集合 | **完全相同**（only_ch0=0，only_mean=0）⇒ 未来评测是严格同集对比 |

抽样 meta 示例（`tt/mix/00001_00_10182_meta.json`）：

```json
{"provenance": {"channel_policy": "0", "audio": "16 kHz mono PCM16", "mouths": "uint8 RGB [T,96,96,3]",
                "aligned_frames": 145, "samples_per_frame": 640,
                "dataset": "RealSSA-Sim", "note": "...near-speaker waveforms not shipped..."}}
```

**结论**：ch0 转换产物完整、通道记录可追溯、与 mean 版严格同集同长度、near 语义与 mean 版一致（均为 fallback/K=0）。

---

## 4. 评测状态：推迟（DEFERRED，无任何指标）

- 2026-10-24T14:14Z 曾启动一次全量评测（cwd=`/root/tmp/p2e_sim_ch0`），
  **运行中被 Lead 的 GPU 统一排期 kill（SIGKILL）**——当时 Lead 已把 2×A800 排给两个 P0 全量 LRS2 评测
  （PIDs 987371/987372，预计 30–45 min，之后可能还有其它 GPU 任务）。
  **本轮未产出任何 per-utterance 结果**：`metrics_rank0.csv` 为 0 字节（已删除），`P2E-realssa-sim-ch0-eval.log` 已加 DEFERRED 头注。
- Lead 决定：ch0 全量 9169 条约需 **2–4 h**，本轮不排；待统一排期后重跑。
- 因此本报告**不含** ch0 的 SI-SNRi/SDRi，也不含 Δ——不做任何推断或占位数字。

### 4.1 运维发现（对后续排期有用，`[验证]`）

`audio_visual_test.py` 的 `_run_workers()` 会 **无条件覆盖** `CUDA_VISIBLE_DEVICES`：

```python
# audio_visual_test.py:447-457
gpus = str(config.get("gpus", "0")).strip().lower()
use_cuda = gpus not in ("", "cpu", "none") and torch.cuda.is_available()
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpus"))   # ← 覆盖外部环境变量
```

⇒ **固定显卡必须用 `--gpus <物理卡号>`，不要依赖外部 `CUDA_VISIBLE_DEVICES`**（会被覆盖）。
本轮第一次尝试传了 `CUDA_VISIBLE_DEVICES=1` 但 `--gpus 0`，结果落在物理 GPU0 上（与当时 train-system 共用），
发现后已自行 kill 并用 `--gpus 0`（当时 GPU0 空闲）重跑；第二次运行中途被 Lead 排期 kill。

### 4.2 待排期后的重跑命令（单卡，独立 cwd 防覆盖）

```bash
mkdir -p /root/tmp/p2e_sim_ch0 && cd /root/tmp/p2e_sim_ch0
PYTHONPATH=/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python \
  /root/code/github_repos/Falcon-fork/audio_visual_test.py \
  --conf_dir /root/code/github_repos/Falcon-fork/configs/Falcon.yml \
  --data_dir /root/data/falcon_realssa_sim_ch0/tt \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt \
  --gpus <空闲物理卡号> 2>&1 | tee /root/code/github_repos/Falcon-fork/docs/verification/phase2/P2E-realssa-sim-ch0-eval.log
# 产物：/root/tmp/p2e_sim_ch0/Experiments/checkpoint/Falcon/results/metrics_combined.csv
# 落盘：cp .../metrics_combined.csv docs/verification/phase2/P2E-realssa-sim-ch0-metrics_combined.csv
```

---

## 5. 与 mean 基线的 Δ：待评测后填写（方法已定）

| 指标 | mean 基线（9169 条，已复核） | ch0（9169 条） | Δ = ch0 − mean |
|---|---|---|---|
| SI-SNRi | **7.774307** | 待评测 | 待评测 |
| SDRi | **8.479395** | 待评测 | 待评测 |
| SI-SNR | 9.980200 | 待评测 | 待评测 |
| SDR | 10.644734 | 待评测 | 待评测 |
| PESQ | 2.093770 | 待评测 | 待评测 |
| STOI | 0.817290 | 待评测 | 待评测 |

> mean 基线取自 `P2E-realssa-sim-metrics_combined.csv` 的 `avg` 行，本轮已独立重算（逐条均值 si-snr_i=7.773784、
> sdr_i=8.478822，与 `avg` 行 7.774307/8.479395 的差异仅来自 `avg` 行按列均值 vs 逐条均值，量级 <1e-3 dB）。
> 判定：若 `|Δ SI-SNRi| > 0.3 dB` 或 `|Δ SDRi| > 0.3 dB`，在最终报告中加入
> “**RealSSA-Sim 的 4ch→mono 通道协议必须显式声明**（mean vs ch0 差 X dB）”。

---

## 6. 产物清单（写作用域内）

| 文件 | 内容 |
|---|---|
| `docs/verification/phase2/P2E-realssa-sim-ch0-convert.log` | 全量转换原始日志（19 shards → 9169 条，exit 0） |
| `docs/verification/phase2/P2E-realssa-sim-ch0-eval.log` | 评测尝试的日志，已加 **DEFERRED** 头注（无指标） |
| `docs/verification/phase2/P2E-realssa-sim-ch0-report.md` | 本报告 |
| `/root/data/falcon_realssa_sim_ch0/`（仓库外） | ch0 数据集：`tt/mix|target|mouths` + `mix.json/target.json/near.json`，30 GB，9169 条 |
| `/root/data/falcon_realssa_sim/`（仓库外，**未改动**） | mean 基线数据集（只读引用） |

---

## 7. 下一步建议

1. **统一排期后重跑 §4.2**（预计 2–4 h 单卡）；用 `--gpus <物理卡号>` 固定卡，别再依赖环境变量。
2. 若时间紧，可先跑 `--limit 1000` 做 ch0 快检（与 mean 的同 1000 条对比），粗判 Δ 量级，再决定是否值得全量。
3. 评测完成后：`cp` combined CSV → 按 §5 填 Δ；若 `|Δ|>0.3 dB`，在 `P2E-report.md`/FINAL 的 RealSSA 口径处加“通道协议必须显式声明”。
4. 注意：ch0 与 mean 的 near.json 都是 mixture fallback（K=0），两者评测语义一致，不存在 near 差异混杂。

---

## 8. 未做 / 约束遵守记录

- 未启动新的评测（Lead 排期指令后未再起任何 GPU 任务）；未 kill 任何进程（第一次 kill 的是**本任务自己的**评测进程，用于纠正落卡）。
- 未 git commit、未 checkout、未改源码/配置/他人文档；仅新增/追加本任务前缀文件。
- 未覆盖 mean 数据目录；未删除任何已有证据。
- 未产出任何 ch0 指标——**评测未完成，明确标记为未验证**，不以任何方式外推。
