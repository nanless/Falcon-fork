# Phase 1 · T4 报告：数据层修复 + 数据工具（合成数据 / RealSSA 转换器）

- 负责人：`data-pipe`
- 分支：`repro/phase1`（基线 `689555c`；**未 git commit / 未切换分支**）
- 运行环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17，torch 2.8.0+cu128，torchvision/opencv/soundfile 齐全）
- 验收依据：`docs/REPRODUCTION_PLAN.md` §5.1 / §6.1 F5 / §6.3；`docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` §6
- 日志（完整命令与输出）：`docs/verification/phase1/T4-data-check.log`

---

## 1. 摘要（TL;DR）

1. **F5 已修**：`AVSpeechEvalDataset` 的 mono `mean(-1)` 加 `ndim` 守卫（训练集/诊断副本同步），单声道 eval 音频由 **0 维**恢复为 **1-D 全长**；stereo 仍按通道取均值。已用合成 mono 数据 + 回归测试验证。
2. **`tools/repro/make_synthetic_data.py` 已交付并验证**：一条命令生成 DataModule 可直接读取的 `{tr,cv,tt}/{mix,target,mouths,near}` + `mix/target/near.json`（JSON 由仓库自己的 `DataPreProcess` 代码生成），mono 16 kHz、`mouths.pt` 为 `uint8 RGB [T,96,96,3]`、meta 带可用 near 负样本（K=1，可冒烟 NASS 对比分支）。**T1/T3 可直接使用**。
3. **`tools/repro/realssa_to_falcon.py` 已交付并验证**：流式读 WebDataset TAR（`tarfile`，无 webdataset 依赖）→ 仓库 folder 格式；4ch→mono（`--channel mean|0|1|2|3`）；帧对齐 `frames=min(mouth, mix//640, target//640)`；`--limit-shards/--limit-samples/--dry-run`；自动生成精简 `_meta.json` 并调用 `DataPreProcess` 生成三份 JSON。**用伪 tar 端到端跑通**（真实 gated shard 未下载）。
4. 回归：T4 测试 **10 passed**；`pytest studio/tests -q` = **35 passed, 2 skipped**。`tools/repro/tests` 全量快照中 `test_eval_entry.py` 有 3 个失败属**其它任务的在编测试**（非 T4 文件、非本次改动引起，见 §5）。

---

## 2. F5：mono wav 在 eval Dataset 中退化为 0 维（已修）

**问题**（`docs/REPRODUCTION_PLAN.md` §6.1 F5）：`sf.read` 对 mono wav 返回 `(samples,)`，原 eval 路径直接 `torch.from_numpy(x).mean(-1)` 会对**时间维**求均值 → 0 维张量 → `mix[None]` 变成 `[1]` → `torch.stft` 抛 `RuntimeError`；训练期 `validation_step` 走同一 Dataset，也一样崩。

**修复**（模式与训练集 `avspeech_dataset_near.py:176-177` 一致：`ndim > 1` 才取通道均值）：

| 文件 | 位置 | 改动 |
|---|---|---|
| `look2hear/datas/avspeech_dataset.py` | `AVSpeechTrainDataset.__getitem__`（legacy） | `mean(-1)` → `mean(-1) if ndim>1 else tensor`，附 F5 注释 |
| `look2hear/datas/avspeech_dataset.py` | `AVSpeechEvalDataset.__getitem__`（**评测/验证实际路径**） | 同上 |
| `look2hear/datas/check_avspeech_dataset.py` | 两份旧拷贝类的 train/eval 路径 | 同上（避免诊断脚本复现旧 bug） |
| `look2hear/datas/avspeech_dataset_near.py` | 训练集 mix/source/near 读取 | 原本已有守卫，**无需改动**（其 eval 用的是 `AVSpeechEvalDataset`，随基类修复） |

**验证（合成 mono 数据，`docs/verification/phase1/T4-data-check.log` §3）**：

```
eval item : mix (48000,) ndim 1 | src (48000,) | mouth (75, 88, 88) | n_near 1
            mix[None] (1, 48000) | mouth[None,None] (1, 1, 75, 88, 88)
```

- 回归测试 `test_eval_dataset_mono_guard`（断言 `ndim==1`、长度=wav 采样数、`mouth_frames == samples//640`）。
- `test_eval_dataset_stereo_still_uses_channel_mean` 证明多声道行为不变（与读回文件的通道均值一致）。
- `test_check_copy_mono_guard` 证明诊断副本同步修复。
- 未改 `audio_visual_test.py`（另一任务负责），但该修复正好消除其 `model(mix[None], ...)` 的 mono 崩溃输入。

---

## 3. `tools/repro/make_synthetic_data.py`（合成数据生成器，T1/T3 冒烟关键）

### 3.1 输出布局（DataModule 直接可读）

```
<out>/<split>/                       split ∈ {tr, cv, tt}
├── mix/<stem>.wav                   mono 16 kHz PCM16 混合音频
├── mix/<stem>_meta.json             RealSSA 风格 meta：components[near/far/env]（near 指向真实合成 wav）
├── target/<stem>.wav                mono 16 kHz PCM16 目标语音
├── mouths/<stem>.pt                 uint8 RGB [T, 96, 96, 3]（25 fps）
├── near/<stem>_near<k>.wav          mono 16 kHz 合成近场干扰人（供 NASS 对比损失冒烟）
├── mix.json / target.json / near.json
```

- **JSON 由仓库真实 `DataPreProcess/process_lrs23.py` 生成**（`preprocess_one_dir` + 内嵌 `generate_near_json`），不是复制逻辑：绝对路径、near.json 与 mix.json 同序等长、near 条目含 `[path, n_samples]`。
- 生成规则：`frames = round(seconds*25)`，`samples = frames*640` → 音频/嘴部**精确对齐**（mouth frame k ↔ 采样 `[640k, 640(k+1))`）；默认 `--seconds 3` = 48000 采样 / 75 帧（满足训练 2 s 随机窗）。
- 合成信号：谐波+包络的目标语音、独立干扰语音、低通环境噪声；`mix = 0.70*target + 0.25*interference + 0.05*env`；near 为独立谐波信号（≥2 s，长度=mixture 长度）。
- meta 带 `scene/room/components(SIR,position)/target_position/transcript`，便于 Phase-2 对齐 RealSSA 结构；`components[].path` 对 near 是**真实可用路径**（K≥1）。
- 校验：`--seconds < 2.0` 直接报错（训练窗口需要 32000 采样）；已有非空 `--out` 需 `--overwrite`。

### 3.2 CLI

```bash
/root/miniforge3/envs/falcon/bin/python tools/repro/make_synthetic_data.py \
    --out /tmp/falcon_smoke/data --n 2 --seconds 3 --seed 0 [--splits tr,cv,tt] [--near 1] [--overwrite]
```

输出末尾直接打印 DataModule 配置片段（`train_dir/valid_dir/test_dir`）。

### 3.3 已验收的形状（3 s 数据、n=2）

| 对象 | 结果（`T4-data-check.log` §3） |
|---|---|
| train `AVSpeechTrainDatasetWithNear` 单条 | `mix (32000,) f32`、`src (32000,) f32`、`mouth (50,88,88)`、`near K=1`（`(32000,)`） |
| eval `AVSpeechEvalDataset` 单条 | `mix (48000,) ndim=1`、`src (48000,)`、`mouth (75,88,88)`、`n_near=1` |
| `AVSpeechDataModuleWithNear` train loader（B=2） | `mix (2,32000) f32`、`mouth (2,50,88,88)`、`near K=[1,1]`、near 元素 `(32000,)` |
| val loader（B=1） | `mix (1,48000)`、`mouth (1,75,88,88)`、`n_near (1,)` |

> 注意：`mouths.pt` 以 uint8 存储 → transform 后 mouth 张量是 **float64**（历史已验证行为，litmodule `type_as` 兜底）；这是真实 RealSSA 格式路径，不是错误。

---

## 4. `tools/repro/realssa_to_falcon.py`（RealSSA-Sim TAR → folder 格式）

### 4.1 输入契约（来自 HF dataset card，仓库 `README.md:80-90` 同步说明；缓存在 `/root/tmp/RealSSA-Sim-README.md`）

- `{tr,cv,tt}/{tr,cv,tt}-NNNNNN.tar`，每 shard ≤500 例；每例连续 4 个成员：
  `<id>.mix.wav`（4ch PCM16 16k）、`<id>.target.wav`（4ch PCM16 16k）、`<id>.json`（scene/room/components(type/source/SIR/position)/target_position/transcript）、`<id>.mouths.pt`（uint8 RGB `[T,96,96,3]`）。
- component type ∈ {`near`,`far`（干扰语音）,`env`,`sudden`（噪声）}；**`components[].path` 是生成期路径，不可下载**（不是 near 波形）。

### 4.2 输出与策略

```
<out>/<split>/mix/<id>.wav        mono 16 kHz PCM16（--channel 策略）
<out>/<split>/target/<id>.wav     同通道 mono 16 kHz PCM16
<out>/<split>/mouths/<id>.pt      uint8 RGB [T,96,96,3]（原样，按帧对齐裁剪）
<out>/<split>/mix/<id>_meta.json  精简 meta：components[{type,path,source,sir_db,position}] + provenance
<out>/<split>/{mix,target,near}.json   由 DataPreProcess 生成
```

- **通道**：`--channel mean`（4ch 平均）或 `0/1/2/3`（默认 `mean`；mixture 与 target 用同一策略，符合 card 的“报告通道协议”建议）。采样率必须 16 kHz，否则清晰报错（不静默重采样）。
- **帧对齐（默认开）**：`frames = min(mouth_frames, len(mix)//640, len(target)//640)`，audio/target 截到 `frames*640`、mouths 截到 `frames`；`--no-align` 关闭。此规则与 dataset 的 640 采样/帧一致。
- **meta**：每个 component 都保留字符串 `path`（生成期 provenance 或 `realssa-sim://<id>/componentN`），因此 `generate_near_json` 可以安全地 `os.path.exists` 失败并走 fallback；可选 `--near-dir`（文件 `<id>.near*.wav`）把外部负样本真正接进 meta/near.json。
- **性能**：每 shard 只打开一次 tar、用 `TarInfo` 偏移逐成员取数（O(1)/member，不在 500 例 shard 上反复线性扫描）。
- **CLI**：`--src`（含 `{tr,cv,tt}` 的根，或单个 `.tar`）、`--out`、`--splits`、`--channel`、`--limit-shards`、`--limit-samples`、`--dry-run`、`--near-dir`、`--no-index`、`--no-align`、`--overwrite`；`--src` 不存在/无 shard 时给出明确错误与下载提示（含 gated 数据说明）。

### 4.3 伪 tar 端到端验收（`T4-data-check.log` §7；`tools/repro/tests/test_realssa_convert.py`）

- 构造 3 例完整 + 1 例缺 JSON/mouths 的 shard（4ch wav/JSON/mouths.pt 按 card 布局）：
  - 转换结果：`converted 2`（1 例 audio 48000→75 帧；1 例 mouth 60 帧→audio 38400）、`skipped incomplete=1`；
  - 输出均为 mono 16 kHz：`00001: 48000 samples / 1ch`、`00002: 38400 samples / 1ch`；`frames == samples/640` 均 True；mouths `(F,96,96,3) uint8`；
  - `mix/target/near.json` 2/2/2；无 near 源时 near.json 为 mixture 哨兵（fallback）；
  - `--channel 0` 与原始 4ch 第 0 通道逐样本一致；`--channel mean` 与 4 通道均值一致；
  - `--dry-run` 不创建输出目录；`--src` 缺失 → 非 0 退出 + "does not exist"；`--near-dir` 时 meta/near.json 正确接入外部 near（`[path, 3*16000]`）。

### 4.4 已知限制（写进使用说明）

1. **未在真实 RealSSA-Sim shard 上跑过**（gated，未下载）：成员名/JSON 字段按公开 dataset card 实现，解析对未知字段是容错的；拿到真实 shard 后建议先 `--dry-run --limit-shards 1`。
2. **RealSSA-Sim 不发布 near 源波形** → 默认 `near.json` 全 fallback → K=0 → NASS 对比损失不生效；需要 `--near-dir` 或 plan §5.1 Step 4 的替代负样本方案。
3. 通道协议论文未声明：复现实验必须报告 `--channel` 选择。
4. 真实 meta 的 SIR/position 字段名可能与本实现假设略有差异（已做多键容错 + 原 JSON key 列表写入 `provenance.source_json_keys`，便于核对）。

---

## 5. 验证与验收记录（全部实际执行）

| # | 命令 / 检查 | 结果 |
|---|---|---|
| V1 | `python tools/repro/make_synthetic_data.py --out /tmp/t4_smoke/data --n 2 --seconds 3` | 3 split × 2 例，每例 48000 采样 / 75 帧（3 s）；三份 JSON 生成成功 |
| V2 | 合成数据 train/eval dataset 各 1 条 + DataModule 两个 loader | §3.3 形状全部符合；F5 mono 为 1-D；near K=1 |
| V3 | `pytest tools/repro/tests/test_synthetic_data.py tools/repro/tests/test_realssa_convert.py -q` | **10 passed**（F5 regression、stereo、check 副本、转换器 5 项） |
| V4 | 伪 RealSSA tar 端到端（`--channel 0` / `mean` / `dry-run` / 缺 src / `near-dir`） | 全部通过（§4.3） |
| V5 | `pytest studio/tests -q`（Lead 要求回归） | **35 passed, 2 skipped**（16.0 s） |
| V6 | `pytest tools/repro/tests -q`（全量快照，含其它任务的在编测试） | 67 passed / **3 failed**（仅 `test_eval_entry.py::test_prepare_config_*`，属其它任务在编代码/测试，与 T4 变更无关；T4 的 10 项全过） |

完整日志：`docs/verification/phase1/T4-data-check.log`（§0–§9）。

---

## 6. 变更文件清单（仅允许范围；未 git commit）

| 文件 | 状态 | sha256 |
|---|---|---|
| `look2hear/datas/avspeech_dataset.py` | 修改（F5，2 处） | `5be594d22b1e8264532e7dbfa270ac1ea7b2420e8dcc9d768a87aace68cd6879` |
| `look2hear/datas/check_avspeech_dataset.py` | 修改（F5，2 处） | `1fb76fbe0e8359cd327ba5093cb67eeaedb6069fa972d2533adf7731eb8f26ce` |
| `tools/repro/make_synthetic_data.py` | 新增（355 行） | `01c49ffdd7a6bad4ec9dcf5f1f5d8772a3dd44aad0890dd40eb8a2cebd71940a` |
| `tools/repro/realssa_to_falcon.py` | 新增（548 行） | `aeb0fc352d4da60586a5a272949eea0727256ccfdaa96ec89d425c217d318154` |
| `tools/repro/tests/test_synthetic_data.py` | 新增（160 行） | `57351332208ca4e0c12a2679faa28b4f5cf6d9e6aae1b714d12b11eaa38286ed` |
| `tools/repro/tests/test_realssa_convert.py` | 新增（227 行） | `f56ebcef473214786ccc9e765b72c59174dde71d6b9547c06f9adec88373e339` |
| `docs/verification/phase1/T4-data-check.log` | 新增（验证日志） | —（见文件） |
| `docs/verification/phase1/T4-report.md` | 新增（本报告） | — |

> `git status` 中 `audio_train.py`/`audio_visual_test.py`/`look2hear/metrics/*`/`look2hear/models/*`/`look2hear/system/*` 的修改属于其它并行任务，**非 T4 改动**；T4 只写了上表文件。

---

## 7. 给 T1 / T3 的交接（可直接用）

```bash
# 生成合成冒烟数据（每个 split 2 条，3 s）：
/root/miniforge3/envs/falcon/bin/python tools/repro/make_synthetic_data.py \
    --out /tmp/falcon_smoke/data --n 2 --seconds 3 --seed 0
```

- 把 `configs/Falcon.yml`（或临时 conf）里 `datamodule.data_config` 的
  `train_dir/valid_dir/test_dir` 分别指向 `/tmp/falcon_smoke/data/{tr,cv,tt}` 即可；
  其余键保持原值（`segment=2.0`、`sample_rate=16000`、`batch_size=4`、`use_near=true`）。
- **不需要改代码**：合成数据含真实可用的 near wav（K=1），NASS 对比分支会真实执行。
- 注意：`train_samples=10000` 默认仍决定 epoch 长度（见 `docs/REPRODUCTION_PLAN.md` F11）；小于 batch×卡数时 loader 可能 drop_last 丢样本，冒烟时可用更小的临时值或 `--n` 增大。
- 该数据仅用于“能跑通/形状对/loss 非 NaN”的管线冒烟，**不得用于任何质量指标**。

---

## 8. 结论

- T4 的三项交付（F5 数据修复、合成数据生成器、RealSSA-Sim 转换器）均已完成并经过执行级验证；T1/T3 现在可以用合成数据跑训练/评测冒烟，无需等待 gated 数据。
- 后续（Phase 2）拿到 RealSSA-Sim 访问权后，建议顺序：`--dry-run --limit-shards 1` → 单 shard 转换 → 检查 channel/SIR/mouth 对齐 → 再批量；near 负样本按 plan §5.1 Step 4 选择方案并显式声明。
