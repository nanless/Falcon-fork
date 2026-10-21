# Falcon 复现 Phase 1 执行报告（Gate G1）

> 日期：2026-10-22 · 分支：`repro/phase1`（基于上游 `689555c`，**未 commit**）
> 计划：[`docs/REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md) §6/§7 · 环境：[`AGENTS.md`](../AGENTS.md)（conda `falcon`）
> 所有命令/日志可在 `docs/verification/phase1/` 找到。

## 1. 结论（TL;DR）

**Gate G1 基本通过**：训练入口与评测入口在 conda `falcon` 环境已可端到端跑通（合成数据 + 官方权重），
指标语义修正且有独立数值验证，数据层 mono 问题已修，视觉骨干可从官方权重导出，官方权重数值指纹**未变**。

| Gate G1 项 | 状态 | 证据 |
|---|---|---|
| `audio_train.py` 可启动并完成 1 epoch（CSVLogger、PL2、config 快照） | ✅ | `/root/tmp/phase1_smoke/train-smoke.log`；`integration-smoke.log` §2 |
| `audio_visual_test.py` 可用官方权重/训练 ckpt 端到端评测 | ✅ | `integration-smoke.log` §3/§4；`T3-eval-smoke.log` |
| 指标语义正确（正号、行/avg 一致、PESQ 顺序） | ✅ | `T3-report.md`；Lead 独立 sanity（good_est `sdr 36.176 / sdr_i 19.158 / si-snr 36.108 / si-snr_i 19.162`；est=mix 时 `i=0.000`） |
| 数据层 mono 不再崩、合成数据可被 DataModule 读取 | ✅ | `T4-report.md`；`integration-smoke.log` §1 |
| 视觉骨干可导出并 strict 加载（11,186,688 参数、全冻结） | ✅ | `T5-report.md`；`/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar` |
| 官方权重数值指纹不变（修复不改数值） | ✅ | `eval/train/sep` sha256 与 `baseline-fingerprint.json` 全等 |
| 新增测试全过 + 回归 | ✅ | `pytest tools/repro/tests` = **82 passed**（QA 新增 5 条后）；`pytest studio/tests` = **35 passed, 2 skipped** |
| 训练栈独立评审 | ✅ 通过（P2×2 已闭环） | `REVIEW-training-stack.md`（105 项检查 PASS=104/WARN=1/FAIL=0；P0/P1=0） |
| **真实数据评测（LRS2-2Mix）** | ✅ **首个真实 benchmark 结果** | 官方权重、**3000 条 tt 全量：SI-SNRi 7.03 / SDRi 7.71**；20 条子集 7.91/8.55；target=s1 已确证（见 §3.5） |
| 独立 QA（回归/数值/跨分支/指标第二意见） | ✅ 通过（P0=0/P1=0/P2=5，终检全绿） | `QA-report.md` §9、`QA-regression.log`；跨分支指纹一致、SDR 对拍 ≤1e-4 dB |

## 2. 修复清单（F1–F14）与落地位置

| # | 修复 | 状态 | 位置/说明 |
|---|---|---|---|
| F1 | `comet_logger` 未定义 → config 驱动 logger（默认 CSVLogger；csv/tb/comet/none） | ✅ | `audio_train.py`（`build_logger`；`none`→Trainer 传 `False` 避免 PL2 默认 logger） |
| F2 | 3 个 litmodule 的 Comet 专属 `logger.experiment.log` → `self.log_dict` | ✅ | `look2hear/system/av_litmodule.py`（NASSv2/MDCL 继承生效）；key/语义保留 |
| F3/F4 | 评测快照类名/占位路径问题 | ✅（工作绕过） | `audio_visual_test.py` 默认 `configs/Falcon.yml`、`--data_dir`、`${VAR}` 展开、清晰错误；旧 conf.yml 不再静默 |
| F5 | eval Dataset mono `mean(-1)` → 0 维崩溃 | ✅ | `look2hear/datas/avspeech_dataset.py`（eval + legacy train）+ `check_avspeech_dataset.py` 加 ndim 守卫 |
| F6 | PL 2.x 兼容（accelerator/devices/DDP） | ✅ | `audio_train.py`（`resolve_accelerator_and_devices`；2-GPU toy DDP 已验证） |
| F7 | 指标语义（sdr=−SNR、符号、PESQ 反参） | ✅ | 新增 `look2hear/metrics/core.py` 单一实现；`allwrapper.py`/`wrapper.py` 重写 |
| F8 | 评测入口可运行化（conf/data_dir/ckpt/limit/数值排序） | ✅ | `audio_visual_test.py`（329→555 行；strict 加载、整句推理、CSV 合并重算） |
| F9 | config dump 被 `Separator.pop` 污染（缺 repeats/shared） | ✅ | dump 前移 + deepcopy；双向证据见 `REVIEW-training-stack.md` |
| F12 | 视觉骨干 `frcnn_128_512.backbone.pth.tar` 缺链接 | ✅ **从 falcon.pt 导出** | `tools/repro/extract_video_backbone.py`；小/大模型骨干 key 相同但 60/137 张量值不同（必须 `--variant`） |
| F13 | `concat_layers[2]` 从未调用（5,568 参数闲置） | ⏸ **按计划保持现状** | 为保 bit-identical；是否修由作者确认 |
| F14 | `_mdcl_loss` 不存在 | ✅ 一半 | `Falcon._mdcl_loss = _mdcl_loss_hard` 别名；**`Falcon_Contrastive.py:95` 仍静默吞 `near_audio`（未在授权文件内，列为 Phase-2 待办）** |
| F15 | DDP per-bucket all_gather 死锁风险 | ⏳ 未做 | 2-GPU toy DDP 通过，但真实 10 卡未验 |
| F16 | 发布 ckpt 格式/占位路径 | ✅ | `audio_visual_test.py` 支持 falcon.pt 的 `{state_dict, architecture, epoch}` 与 Lightning ckpt；实测 strict 加载 525/137 keys |

## 3. 冒烟结果（真实执行）

### 3.1 训练冒烟（合成数据，1 epoch）

```
config: configs/Falcon.yml + synthetic data (tr/cv/tt 各 3 条, 3 s, 16 kHz mono)
trainer: max_epochs=1, batch=2, train_samples=8, devices=[0], CSVLogger
结果: 4 train steps + 1 val；val_loss=12.188（= -SI-SDR，监控 min 正确）；
      epoch=0.ckpt / last.ckpt / best_model.pth / best_k_models.json / conf.yml 均生成
```
- 注意：`val_loss = PIT 负损失`；`val_pit_sisnr = -val_loss = -12.188`（初始化模型在随机合成数据上的 SI-SDR），
  符号约定正确（min val_loss = max SI-SDR）。
- 合成数据**仅用于管线冒烟**，数值无指标意义。

### 3.2 评测冒烟

| 权重 | 结果 |
|---|---|
| 训练 1 epoch 的 ckpt | 端到端跑通，audio 525 + video 137 keys strict 加载，CSV 生成 |
| 官方 `falcon.pt` | 同上；`--limit 1`、整句推理、`metrics_combined.csv` 生成 |
| 合成数据上的指标 | 数值为负属正常（模型对随机无意义音频），符号/一致性由单测与 sanity 保证 |

### 3.3 指标独立 sanity（Lead）

| 场景 | sdr | sdr_i | si-snr | si-snr_i |
|---|---|---|---|---|
| est = 0.9·target + 小噪声 | 36.176 | 19.158 | 36.108 | 19.162 |
| est = mixture（基线） | 17.018 | **0.000** | 16.945 | **0.000** |

行与 avg 一致、improvement 语义正确、PESQ 顺序修正后合理（1.85 vs 1.03）。

### 3.4 数值不变量

官方权重固定种子下：`eval` / `train(est+sep)` sha256 与 `baseline-fingerprint.json` **完全一致**；
参数 `537,697 / 11,186,688` 不变；T5 的 `return self`/alias/`__all__` 未改变任何前向数值。

### 3.5 LRS2-2Mix 真实数据评测（首个真实 benchmark 结果）

**数据**：`/root/data/lrs2_2mix/lrs2_rebuild`（已解压，见 §10）。转换方式：
`audio/wav16k/min/tt/mix/*.wav` → `mix/`；`audio/wav16k/min/tt/s1/*.wav` → `target/`（`mix == s1+s2` 实测成立，
target convention 用 speaker1）；`mouths/<s1_id>.npz` 的 `uint8 (50,96,96)` → stack 3 通道 `(50,96,96,3)` → `.pt`；
按仓库格式写 `mix.json/target.json/near.json`；`--data_dir /root/tmp/lrs2_eval_tt_full`。

**20 条 tt 子集结果（官方 `falcon.pt`，`--limit 20`，GPU A800）**：

| target 约定 | SI-SNR | SI-SNRi | SDR | SDRi | PESQ | STOI |
|---|---|---|---|---|---|---|
| **speaker1（s1，正确约定）** | 13.60 | **7.91** | 14.31 | **8.55** | 2.31 | 0.923 |
| speaker2（s2，错误约定，对照） | −1.12 | 4.60 | 0.31 | 5.68 | 1.25 | 0.612 |

结论：
- **评测管线在真实数据上自洽**：目标约定选对时 SI-SNRi/SDRi 显著更高、STOI 0.92，选错时明显下降 ⇒ target=s1；
- 20 条子集 SI-SNRi 7.91 / SDRi 8.55，**低于论文 LRS2-2Mix 的 14.1 / 14.7**（差 ~6 dB）。
  可能原因（待全量/协议对齐后判断）：① 20 条随机子集不代表全量；② 论文的训练数据/配方与发布权重不同
  （配置快照为 LRS3/LRS2 混合，实际训练集未知）；③ 评测协议差异（论文可能用不同分段/通道/嘴部 crop 约定）；
  ④ 指标实现口径（我们用 fast_bss_eval 512-tap SDR + 标准 SI-SDR）。
**全量 3000 条 tt 评测（官方 `falcon.pt`，已完成）**：

| 指标 | 值 |
|---|---|
| SI-SNR / SI-SNRi | 12.023 / **7.029** |
| SDR / SDRi | 12.813 / **7.713** |
| PESQ / STOI | 2.196 / 0.916 |

与论文 LRS2-2Mix 的 14.1 / 14.7 相比差 ~7 dB。可能原因（Phase 2 需逐项排除）：
① **发布权重与论文 LRS2 列可能不是同一训练集**（配置快照是 LRS3/RealSSA 风格；论文不同 benchmark 列可能各自训练）；
② 评测协议差异（目标源约定 s1 已由 s1/s2 对照确认；分段/缩放/指标实现对 SDRi 影响可达数 dB）；
③ 嘴部 crop/对齐与训练分布不同（LRS2-2Mix 的 mouths npz 由 AVSS 流水线生成，虽然尺寸一致但对齐细节可能不同）；
④ 权重实际训练数据未知（README 未说明 falcon.pt 的训练集组合）。
**下一步**：等 RealSSA-Sim 访问批准后，用官方权重在 RealSSA-Sim test 上评测（论文 7.7/8.2，falcon.pt 更可能对应这个训练分布）；
同时用 falcon-big 在同数据上做对照、并复现论文的 LRS2 训练配方（若确认 paper 的 LRS2 列来自单独训练）。

## 4. 新增工具（`tools/repro/`）

| 脚本 | 用途 |
|---|---|
| `make_synthetic_data.py` | 生成 DataModule 可直接读取的合成数据（含真实可读 near wav，NASS 分支真实执行） |
| `realssa_to_falcon.py` | RealSSA-Sim WebDataset TAR → 仓库 folder 格式（4ch→mono、掩码 88 crop 前保留 RGB、meta/JSON 生成；伪 tar 端到端验证） |
| `extract_video_backbone.py` | 从 `falcon.pt`/`falcon-big.pt` 导出 `frcnn_128_512.backbone.pth.tar` 兼容文件（137 keys、全冻结） |
| `fetch_data.py` | 统一数据获取（LRS2-2Mix 公开下载；RealSSA gated 提示；HF_ENDPOINT 可配） |
| `tests/` | 70 个 pytest（入口/日志/指标/数据/转换器/骨干/回归） |

## 4.1 文档闭环（2026-10-22 终态）

- `docs/DATA_ACCESS_GUIDE.md`：398 行，通道引用已改为 `README.md:82/85/87`；RealSSA-Sim 242 files/240 shards；申请表单 3 checkbox；
  新增 **§3.3 如何把 LRS2-2Mix 转成 Falcon 训练/评测格式**（`mix/`、`target=s1`、`mouths npz→.pt 补 3 通道`、
  `process_lrs23.py` 生成 JSON、`use_near=false` 等），并用 3 样本/split 冒烟验证 Dataset 输出 `(32000,)/(32000,)/(50,88,88)`。
- `docs/REPRODUCTION_PLAN.md`：按实测更新 LRS2-2Mix 全 AV 结构、17.39 GB 解压、s1/s2 直接可得、官方权重实测 7.03/7.71。
- `docs/verification/phase1/`：T1–T6 报告、QA 报告（21 用例/82 全量）、训练栈评审（105 项）、Studio/文档回归、集成冒烟日志齐备。
- 证据归档：`docs/verification/phase1/dataset-cards/`（6 份数据集卡）、`T6-lrs2-structure.txt`、`T6-lrs2-listing-sample.txt`。

## 5. 测试与质量

| 检查 | 结果 |
|---|---|
| `pytest tools/repro/tests -q` | **82 passed / 82 collected**（QA 终检，67 s） |
| `pytest studio/tests -q` | **35 passed, 2 skipped** |
| Studio Node 测试 | 8/8（Phase 0 基线，本轮改动未触及前端） |
| T1/T2 独立评审 | P0=0/P1=0/P2=2/P3=4（74 项检查 PASS；P2 为 GPU 选择语义/空列表校验，已反馈修复） |
| 独立 QA（layers-dsp，**终检**） | **PASS**：QA 16 passed、全量 **77 passed / 0 failed**、studio 35/2、imports 12 模块 OK、4 条官方权重指纹与 baseline 逐位一致、跨 `/root/wt-main` 重算一致、SDR/SI-SDR 对拍 ≤1e-4/≤5e-3 dB、`est=mix ⇒ 提升恰 0`、diff 范围守卫通过；**终检锚定 `git diff main` sha256 `d854fa7d…96c1b`** |
| 训练栈增量复评 | ✅ 通过（P2-1/P2-2 闭环；105 项检查 PASS=104/WARN=1/FAIL=0；残留仅 P3：config 语义注释缺失等，已授权 train-sys 补） |
| 代码卫生 | 无 debug 残留；改动文件 AST 全部通过；`git status` 仅目标文件被修改（未 commit） |

## 5.1 文档终检（第三轮，studio-frontend）

- GUIDE：P1/P2 全部修复（README 引用、242 files/240 shards、3 checkbox、旧计划提示），LRS2 AV 事实双向核实；剩余仅 2 条 P3（上游 Oxford 链接 404、证据路径说明）。
- PLAN：LRS2 旧口径已全部更新（本文件 §5.2/§8/Phase3/§11.2/待办；R6 风险已更正为“包内含视觉”）；剩余 P3（§6.3 F12 引用、`bound()` 笔误、R12 口径）。
- REGRESSION-docs：最新 sha 复核通过；回归全绿（82 / 35+2 / Node 8/8 / Space i18n 226 / imports 10/10）。

## 6. 遗留与风险（Phase 2 前置）

1. **真实数据状态（Phase 1 结束时）**：
   - RealSSA-Sim / RealSSA-Real：仍为人工审批 gated（尚无 token 授权），见 `docs/DATA_ACCESS_GUIDE.md`；
   - **LRS2-2Mix：已下载完成**（16,470,508,590 B，sha256 校验通过）。
     **重大发现**：发布包不是纯音频，而是完整 AV 重建数据集——含 `audio/wav16k/min/{tr,cv,tt}/mix/*.wav`（2 s，16 kHz）、
     **`mouths/<id>.npz`（uint8 (50,96,96)，正好是 Falcon 输入格式）**、`faces/<id>.mp4`、`landmark/*.npz`、
     `raw_audio/{train,test}/*.wav`、`create_mix/*.scp`（源+SNR）与 MATLAB 混音脚本。
     ⇒ **LRS2-2Mix 的 AV 复现不需要 LRS2 原始视频**；而且 `mix == s1+s2` 实测成立（max|Δ|=3.05e-5），**target 参考直接可得**（s1/s2 正在包内；target convention 需固定并报告）。
     详见 `docs/REPRODUCTION_PLAN.md` §5.2 与本报告 §10。
   - LRS3-2Mix / VoxCeleb2-2Mix：HF 仓库只有 README（无数据文件），不可复现（需联系作者）。
2. **F14 另一半**：`Falcon_Contrastive.py:95` 仍静默吞 `near_audio`（不影响 NASS 训练，但 MDCL 路线仍不可用）。
3. **F13 决策**：`concat_layers[2]` 按计划保持现状（保 bit-identical）；若论文架构意图是调用它，需要作者确认。
4. **未实跑**：`hf://` 权重分支（代码保留，需联网）、多卡 `mp.spawn` 评测、真实 10 卡训练（DDP bucket 死锁风险 F15）。
5. **指标口径**：SDR 采用 fast_bss_eval 512-tap（BSS-Eval v4 风格）；若作者论文用不同实现，需在 Phase 2 用真实数据对齐口径并记录差异。
6. **Space 测试**需要独立环境（gradio/spaces，py3.12/torch2.13），不要装进 `falcon`。
7. **pypesq** 用 `pesq 0.0.4` + shim（见 `AGENTS.md`）；PESQ 数值与 pypesq 同源实现，但版本差异需记录。
8. **训练栈 P2/P3**：`gpus` list vs int 语义、空列表校验（train-sys 正在修）；`audio_train.py` 未使用 `import sys`；CSVLogger 目录重复、`--conf_dir` 旧默认值（P3）。
9. **QA/复评记录的 P2（非阻塞）**：`audio_train.py` 未使用 `import sys` 与 scheduler 行格式（train-sys 正在清理）；T2 日志 dict 去掉了 `"epoch"`（改由 PL epoch 列）；T3 的 legacy `SPlitMetricsTracker` 仍是旧负号语义（未使用，Phase-2 标 deprecated）；`requirements.txt` 仍列 `torch-mir-eval`（兼容保留）；`configs/Falcon.yml` 缺 logger/gpus 语义注释（已授权 train-sys 补注释，不改默认值）。
10. **video 指纹约定（QA 定位）**：`baseline-fingerprint.json` 的 video 项是在 **FRCNN 默认 `training=True`（未调 `.eval()`）** 下生成的；
    调 `.eval()` 会得到不同 sum（5955.7 vs 7832.2563）。QA 用例已固定该约定；Phase-2 复跑请沿用或重新生成基线时显式记录模式。

## 7. Phase 2 衔接（下一步）

1. 等 LRS2-2Mix 下载完成 → `T6` 结构核实 → 构建 LRS2-2Mix 音频/评测链路（嘴部 crop 需 LRS2 原视频 + `crop_mouth_from_video.py`）。
2. 提交 RealSSA-Sim/Real 访问申请（见 `docs/DATA_ACCESS_GUIDE.md`）；批准后：单 shard → 转换器 → 官方权重 test 评测（目标 RealSSA-Sim SI-SNRi 7.7 / SDRi 8.2）。
3. 用 LRS2-2Mix + 官方权重复现 14.1 / 14.7（若拿到视频/嘴部）。
4. 负样本方案决策（RealSSA-Sim 无 near 源波形）→ 训练复现（Phase 4）。
5. 清理：本分支改动经评审后由用户决定是否 commit/PR（当前未 commit）。

## 8. 复跑命令

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon
# 新测试 + 回归
$ENV/bin/python -m pytest tools/repro/tests -q
$ENV/bin/python -m pytest studio/tests -q
# 合成数据 + 训练冒烟
$ENV/bin/python tools/repro/make_synthetic_data.py --out /root/tmp/phase1_smoke/data --n 3 --seconds 3 --near 0 --overwrite
cd /root/tmp/phase1_smoke && PYTHONPATH=/root/code/github_repos/Falcon-fork \
  $ENV/bin/python /root/code/github_repos/Falcon-fork/audio_train.py --conf_dir /root/tmp/phase1_smoke/smoke.yml
# 官方权重评测冒烟
PYTHONPATH=/root/code/github_repos/Falcon-fork $ENV/bin/python \
  /root/code/github_repos/Falcon-fork/audio_visual_test.py \
  --conf_dir /root/code/github_repos/Falcon-fork/configs/Falcon.yml \
  --data_dir /root/tmp/phase1_smoke/data \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt --limit 1 --gpus 0
# 视觉骨干导出
$ENV/bin/python tools/repro/extract_video_backbone.py --variant small
```

## 9. 相关文档

- 计划：[`docs/REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md)
- 数据获取：[`docs/DATA_ACCESS_GUIDE.md`](DATA_ACCESS_GUIDE.md)
- 各任务报告：`docs/verification/phase1/T1..T6-report.md`、`QA-report.md`、`REVIEW-training-stack.md`、`REGRESSION-docs.md`
- 冒烟日志：`docs/verification/phase1/integration-smoke.log`
- 基线指纹：`docs/verification/phase1/baseline-fingerprint.json`
