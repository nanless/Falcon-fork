# Phase 2-C 报告：LRS2-2Mix → Falcon 训练格式（全量转换完成）

- 负责人：`data-pipe`
- 分支：`repro/phase2`（未 commit；只写 scope 内文件）
- 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / soundfile 0.14.0）
- 任务：`task-20`（Phase2-C：LRS2-2Mix 全量转 Falcon 格式 + 索引/near 生成）
- 完成时间：2026-10-21 19:24 UTC（全量转换耗时 **237.9 s ≈ 4.0 min**，16 并行）
- 日志：`docs/verification/phase2/P2C-convert.log`（117 行，含审计/每 split 进度/校验/DataModule 集成检查）

---

## 1. 交付物与路径（已就绪，训练可直接用）

```
/root/data/falcon_lrs2_2mix/
├── tr/  cv/  tt/
│   ├── mix/<stem>.wav          # 32000 采样（2.000 s）16 kHz mono PCM16
│   ├── target/<stem>.wav       # target=s1（与 mix 同名 stem）
│   ├── mouths/<target_id>.pt   # uint8 RGB [50,96,96,3]，按 target utterance id 去重
│   ├── mix.json                # [[abs_mix, 32000], ...]
│   ├── target.json             # [[abs_target, abs_mouth_pt, 32000], ...]
│   └── near.json               # [[[abs_mix, 32000]], ...]  （fallback 模式：mixture 哨兵）
├── conversion_stats.json       # 机器可读统计/校验/索引信息
└── （无 conversion manifest 文件；一切以 JSON + stats 为准）
```

- **条数**：tr 20000 / cv 5000 / tt 3000 = **28000 条**（与源 `audio/wav16k/min/*` 一一对应；`missing_mouth_npz=0`）。
- **磁盘**：**23.36 GiB**（`du -sh` = 24 GB）：tr 21.44 GiB / cv 1.21 GiB / tt 0.70 GiB。
- `/root` 当前余量 15 T，未受影响。

## 2. 转换统计

| split | mix | target | mouths(pt) | 时长 | 转换耗时 | 输出大小 |
|---|---|---|---|---|---|---|
| tr | 20000 | 20000 | 14773 | 20000×2.000 s | 33.6 s | 21.44 GiB |
| cv | 5000 | 5000 | 476 | 5000×2.000 s | 11.0 s | 1.21 GiB |
| tt | 3000 | 3000 | 267 | 3000×2.000 s | 5.9 s | 0.70 GiB |
| **合计** | **28000** | **28000** | **15516** | **~15.6 h** | 转换 ~50 s；整次运行 237.9 s | **23.36 GiB** |

- mouths 去重：同一 target utterance 被多条 mixture 复用，`.pt` 按 `<target_id>.pt` 只存一份（28000 → 15516），节省约 17 GB。
- 整次运行 237.9 s 中，主要耗时不是转换本身，而是**网络文件系统上的目录扫描/索引构建**（tr 20000 条 JSON 路径 + 28k 文件 stat/sf.info）。

## 3. 数据格式与约定

| 项 | 约定 | 证据 |
|---|---|---|
| 采样率/声道 | 16 kHz mono PCM16 | 全部源 wav 实测（`sf.info`），转换逐字节复制 |
| 样本长度 | 32000 采样 = 2.000 s（全部 28000 条） | `conversion_stats.json` → `sample_lengths={32000: N}` |
| 视频帧 | 25 fps；mouths 50 帧，`frames == samples//640` | npz 原数据 `uint8 [50,96,96]` → 补 3 通道 |
| mouth 文件 | `torch.uint8 [50,96,96,3]`，3 通道为灰度帧复制（Dataset 先 Grayscale 再 crop 88） | `test_lrs2_convert.py` 逐元素断言 |
| target | **s1**（paper/约定默认；`--target s2` 可切换，且 mouth 同步用 s2 id） | 转换器 `--target` |
| stem | 保留源 recipe：`<s1_id>_<gain1>_<s2_id>_<gain2>`（如 `5535..._00004_2.4022_6160..._00015_-2.4022`） | `STEM_RE`；与源文件名一致 |
| mix/target 对齐 | `mix == gain1*s1 + gain2*s2`（PCM16 舍入内） | 抽样校验 max err **3.05e-05** |
| JSON 绝对路径 | 全部为绝对路径；`target.json` 第 3 字段 = 采样数（DataLoader 目前不读，但保持契约） | `mix.json`/`target.json` 头 |
| near | 默认 **fallback**：`[[abs_mix, 32000]]`（Dataset 跳过 → K=0） | 见 §5 |

## 4. 校验结果（全部通过）

### 4.1 抽样校验（转换器内建，每 split 200 条，共 600 条）

| split | checked | length_ok | mouth_ok | mix_sum_ok | max\|mix-(s1+s2)\| | near_ok | problems |
|---|---|---|---|---|---|---|---|
| tr | 200 | 200 | 200 | 200 | 3.05e-05 | 200 | **0** |
| cv | 200 | 200 | 200 | 200 | 3.05e-05 | 200 | **0** |
| tt | 200 | 200 | 200 | 200 | 3.05e-05 | 200 | **0** |

（`near_ok` 在 fallback 模式下检查 near 条目 == mixture 哨兵；raw_audio 模式会检查外部 near 文件格式/说话人不同。）

### 4.2 全量存在性检查（非抽样）

```
JSON existence check: 28000 samples, missing files=0 (elapsed 16.7s)
near=fallback identical to mix sentinel for all 3 splits
```

- 对 3 个 split 的 28000 条 JSON 逐条检查 `mix/target/mouths` 三个路径：**0 缺失**。
- `mix.json`/`target.json`/`near.json` 三者逐 split 等长（20000/5000/3000）。

### 4.3 训练侧集成验证（与 train-sys 的 `configs/LRS2_2Mix.yml` 完全一致）

用转换后的目录实例化真实 `AVSpeechDataModuleWithNear`：

```
train batch: mix (4, 32000) float32 | target (4, 32000) | mouth (4, 50, 88, 88) float64
             near K [0, 0, 0, 0] | names ['6074..._-3.4864.wav', ...]
val batch  : mix (1, 32000) | mouth (1, 50, 88, 88) | n_near (1,)
```

- 形状/长度与训练预期一致；mouth 为 float64 是 **uint8 `.pt` 存储的既有行为**（litmodule 内 `type_as` 转 float32）。
- `configs/LRS2_2Mix.yml` 的 `train_dir/valid_dir/test_dir` 已指向 `/root/data/falcon_lrs2_2mix/{tr,cv,tt}`，无需改动。

## 5. near 说明（重要，影响论文指标解读）

- **LRS2-2Mix 发布物只有 mixture 和两个源（s1/s2），没有 near-field 干扰源波形**。因此默认 `--near-mode fallback` 写入 mixture 哨兵 `[[mix_path, n]]`；Dataset 按 `near_path == mix_path` 跳过 → **每个样本 K=0**。
- 后果（与 Phase1 数据域结论一致）：NASS 对比损失只剩正样本项（负项恒 0）、MDCL 恒 0、val 分桶全进 `0near`。**这不是本转换器的 bug，而是数据可得性限制**；任何用该数据训练出的模型都不包含“近场负样本对比学习”的贡献，报告指标时必须显式声明。
- 转换器已实现可选 `--near-mode raw_audio`（**本次全量未使用**，留作消融实验）：对每条 mixture 从同 split 的 `raw_audio/{train,val,test}` 中确定性选取 1 个 **不同说话人** 的 2 s 片段写入 `near.json`。其风险：raw_audio 是干净单说话人片段（无 RIR、无响度/阵列匹配、非真实干扰源），只能作为 “non-official negatives” 近似（`docs/REPRODUCTION_PLAN.md` §5.1 Step 4 option 2）。若使用，必须在报告中标注协议。
- 若后续要复现论文的对比损失，需要：① 向作者索取 near 源波形；② 或用 RealSSA-Sim 的 component 重建（plan §5.1 Step 4 option 3，高成本）。

## 6. 转换器实现要点（`tools/repro/lrs2_2mix_to_falcon.py`，644 行）

- **CLI**：`--src`（默认 `/root/data/lrs2_2mix/lrs2_rebuild`）、`--out`、`--splits tr,cv,tt`、`--target s1|s2`、`--jobs`（默认 8，本次 16）、`--limit`（每 split 前 N 条）、`--near-mode fallback|raw_audio`、`--near-per-sample`、`--seed`、`--dry-run`、`--overwrite`、`--validate-samples`、`--mix-tol`。
- **解析**：`<s1_id>_<gain1>_<s2_id>_<gain2>` 正则；校验 s1/s2/mix 存在、npz 存在（缺失计入 `missing_mouth_npz` 并跳过该条，不中断）。
- **并行**：`ProcessPoolExecutor`；每个样本：原子复制 mix/target（tmp+`os.replace`）、按 target id 去重写 mouth `.pt`（仅首次出现的 mixture 负责写，避免 28k→15.5k 的重复写放大）。
- **可重入**：已有且大小合理的 mix/target/mouth 默认跳过（`--overwrite` 强制重写）；JSON 索引每次从输出目录重建。
- **增量门禁**：**每个 split 转换完成后立即写该 split 的 `mix/target/near.json`**（tr 的 mix.json 19:22 出现，早于 cv/tt 完成），便于 train-sys 提前启动。
- **统计/校验**：输出 `conversion_stats.json`（每 split 条数/长度分布/字节数/转换耗时 + 抽样校验 + 索引路径），供报告与下游脚本复用。
- **容错**：单样本错误只记录并继续（最多打印前 10 条），不中断全量。

## 7. 训练使用方式（train-sys 已确认）

```bash
# configs/LRS2_2Mix.yml 的 data_config 已指向 /root/data/falcon_lrs2_2mix/{tr,cv,tt}
/root/miniforge3/envs/falcon/bin/python audio_train.py --conf_dir configs/LRS2_2Mix.yml
```

- `use_near: true` 可保持（数据侧 fallback → K=0，无对比负项；不会报错）；若想显式关闭对比分支，可改 `use_near: false`（需 litmodule 支持，属 P2-D 范围）。
- `train_samples: 10000` 决定每 epoch 抽样条数（F11）；`batch_size=4`、`num_workers=8` 与数据规模匹配。
- 期望 train batch：`mix/target [B,32000] f32`、`mouth [B,50,88,88]`（float64→`type_as` f32）、`near_audios` 为空列表。
- 评测（tt 3000 条）可用 `audio_visual_test.py` + `configs/LRS2_2Mix.yml`（P2-D/P2-E 负责指标口径）。

## 8. 测试与日志

- 单元测试：`tools/repro/tests/test_lrs2_convert.py`（245 行，**7 passed**，23.4 s）：
  - fallback 端到端（布局/JSON schema/target=s1/mouth 3 通道/字节级 target 复制/mix==s1+s2/stats）；
  - `--target s2`（target 与 mouth 均切换 s2）；
  - `--near-mode raw_audio`（路径/格式/不同说话人）；
  - `--dry-run` 不写文件；重入跳过 existing；`--limit` 与 `missing_mouth_npz` 计数；mix 配方不一致会被 validation 报告。
  - 运行：`PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_lrs2_convert.py -q`
- 全量运行日志：`docs/verification/phase2/P2C-convert.log`（含源审计、每 1000 条进度、每 split INDEX READY、最终 summary、POST-RUN 存在性/DataModule 集成检查）。
- 复现命令：
  ```bash
  /root/miniforge3/envs/falcon/bin/python -u tools/repro/lrs2_2mix_to_falcon.py \
      --jobs 16 --out /root/data/falcon_lrs2_2mix --near-mode fallback --validate-samples 200 \
      2>&1 | tee docs/verification/phase2/P2C-convert.log
  ```

## 9. 已知限制 / 后续

1. **无 near 源波形** → 默认 K=0（见 §5）；论文对比损失贡献无法用本数据复现（需数据侧资产或替代负样本方案）。
2. **target 约定 s1**：LRS2-2Mix 的 s1 为视频说话人（paper 约定）；`--target s2` 仅用于消融（mouth 也来自 s2，但需确认视觉目标一致性）。
3. **样本仅 2.0 s**：训练随机 2 s 窗=整段（`segment=2.0`），不存在短样本过滤；评测也是整段 2 s（与 RealSSA 变长评测不同，指标口径需在 P2-E 说明）。
4. **mouth 为源 npz 原样补 3 通道**：未做 RealSSA 式 RGB→灰度显式转换（Dataset 的 Grayscale 会做）；与真实训练资产格式一致（4-D channel-last uint8）。
5. **未对 s1/s2 做语音级校验**（如 SI-SDR 上界、静音/削波统计）；mix==s1+s2 已抽样验证；如需更强质量报告可在 P2-E 补充。
6. 本次转换脚本未处理 `--limit` 与断点混合的边界（resume 时若改 limit，JSON 会按已存在文件重建，符合预期）。

## 10. 变更文件清单（未 commit；scope 内）

| 文件 | 状态 | sha256 |
|---|---|---|
| `tools/repro/lrs2_2mix_to_falcon.py` | 新增（644 行） | `632dbf157efcd098f6884df897c3bb79a0e065de84087c5b8ae19294307bbb8d` |
| `tools/repro/tests/test_lrs2_convert.py` | 新增（245 行） | `6bbcfa78f63f6189cfaf6d544a2e8b64b963304bf692b727e60f3d4f524346a7` |
| `docs/verification/phase2/P2C-convert.log` | 新增（117 行） | — |
| `docs/verification/phase2/P2C-report.md` | 新增（本报告） | — |

> 数据产物在 `/root/data/falcon_lrs2_2mix`（仓库外，不入 git）；未修改 Phase 1 源码，未 git commit。

## 11. 结论

- **LRS2-2Mix 全量 28000 条已转换完成**，`{tr,cv,tt}/{mix,target,mouths}` + `mix/target/near.json` 全部就绪，抽样/全量存在性/训练侧 DataModule 三层验证通过；train-sys 可立即用 `configs/LRS2_2Mix.yml` 启动 P2-D 训练。
- 数据侧唯一缺口是 near 负样本（数据发布物限制），已在 `near.json` 中如实 fallback，并在本报告与 `docs/REPRODUCTION_PLAN.md` 中标注；任何训练结果需同时声明 “K=0 / 无对比负项”。

---

## 12 补记（2026-10-24，P0-3 / task-12）：near.json 的 s2 负样本来源

> 本报告 §5 与 `conversion_stats.json` 的 "near=fallback（K=0）" 只描述 **2026-10-21 的首次转换**。
> 自 **2026-10-22T16:37:54Z** 起，磁盘 `/root/data/falcon_lrs2_2mix/{tr,cv,tt}/near.json` 已被一条未入库命令原地改写为 **"s2 作 near"（K=1，每条 `[[abs_s2_wav, 32000]]`）**；P2-D 的 contrastive 微调（test SI-SNRi 8.618→8.858 / SDRi 9.042→9.270，+0.24/+0.23 dB）用的就是这一版。
> 该改写现已固化为 `tools/repro/lrs2_2mix_to_falcon.py --near-mode other-speaker`（确定性、K=1），可一键重建与磁盘**逐条同构**的 `near.json`；5 条真实样本验证路径/n_samples 完全一致且 `near==s2`（max|Δ|=0.0），单测 `pytest tools/repro/tests/test_lrs2_convert.py` → 10 passed。
> 漂移原因、时间线、重建命令与 P2-D 数据来源详见 **`docs/verification/phase2/P2C-near-mode-provenance.md`**；重跑转换器后 `conversion_stats.near_mode` 会如实变为 `other-speaker`。
