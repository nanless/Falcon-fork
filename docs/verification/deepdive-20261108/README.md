# deepdive-20261108 实验证据（4 个 experiments 脚本 + 流式仲裁）

> **2026-11-10 复核：**`lookahead_ab.py` 的默认输出比较属于自比较；流式取帧与相位配对存在边界问题；消融记录中的 mouth 标识尚未核实对应的说话人身份。[逐篇核查](../deepdive-20261110/深挖文档复核状态.md)和[模型与流式说明](../../understanding/20261110-model-and-streaming.md)列出具体修正。

> 生成：**2026-11-08**（UTC 14:50–14:57 / 北京 22:50–22:57）
> 主机 `di-20260825114741-2bbpx`（2×A800；本轮**纯 CPU**）
> 仓库 `/root/code/github_repos/Falcon-fork` @ `6c1d5b7`（工作树 **dirty**：跑测时 `git status --porcelain` 44 条，其中 `look2hear/` 7 个）
> 环境 conda `falcon`：Python 3.11.17 / torch 2.8.0+cu128 / numpy 2.2.6
> 线程：`CUDA_VISIBLE_DEVICES=` + `OMP_NUM_THREADS=4`，脚本内 `torch.set_num_threads(4)`
> 权重：`/root/.cache/falcon_weights/falcon.pt`（→ `/root/group-shared/...` 软链），epoch 714
> **sha256 `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`**
>
> 本目录保存实验产物。`lookahead-ab.txt` 是同一命令的 stdout，其中记录了 JSON 未包含的 `max|Δ|` 数字。

## 1 文件清单与生成命令

公共前置：

```bash
REPO=/root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
cd $REPO
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
mkdir -p /tmp/ev
```

| 文件 | 生成命令 | 实测耗时 | 关键数字（详见 §2） |
|---|---|---|---|
| `lookahead-ab.json`（+ `.txt`） | `$ENV tools/repro/experiments/lookahead_ab.py --json /tmp/ev/lookahead-ab.json` | **24.8 s**（61.3 s user，264% CPU） | 音频「看未来」125.0 帧（4 组同值）；`default` 与自身比较；`paper_causal` max\|Δ\|=13.01；LSTM 递推 1→7 帧 |
| `ln-variant.json` | `$ENV tools/repro/experiments/ln_variant.py --variant both --allow-dirty --json /tmp/ev/ln-variant.json` | **12.7 s** | 回看 218→110 帧；帧 0..120 最小响应 −40.2→−240 dB；输出电平 +44.6 dB（169.5×） |
| `streaming-consistency.json` | `$ENV tools/repro/experiments/streaming_consistency.py --json /tmp/ev/streaming-consistency.json` | **33.7 s** | guard 80 ms：**+18.20 dB** / r 0.9925 / 峰值偏差 0.188（6 s 流、3 条片段） |
| `video-ablation.json` | `$ENV tools/repro/experiments/video_ablation_check.py --stage variants --n 40 --select spread --out /tmp/ev/video` | **93.2 s**（2,329 ms/条） | real `+7.113`；置零 **−15.98**；黑帧 **−3.16**；替换 mouth **−11.68**（N=40，全部成对） |
| `canonical.json`（汇总） | 由下面两条输出生成（同目录 `streaming_canonical.py`，见 §4） | — | 6 s：14.71 / 16.72；4 s：14.14 / 17.52；两两最小 17.02 / 17.05 |
| `canonical-6s.json`（原始） | `$ENV tools/repro/experiments/streaming_canonical.py 6 --json <out>` | **88.6 s**（复跑 75.8 s） | 见 `canonical.json` |
| `canonical-4s.json`（原始） | `$ENV tools/repro/experiments/streaming_canonical.py 4 --json <out>` | **56.0 s**（复跑 60.3 s） | 见 `canonical.json` |

`canonical-6s.json` / `canonical-4s.json` 是深挖轮 `/tmp/w213/out_{6,4}s.json` 的**逐字节副本**；
入仓库后已用 §4 的脚本原地复跑校验：**除 `elapsed_s`（计时）外所有字段逐位相同**。

## 2 每个 JSON 的关键数字

### `lookahead-ab.json` — 两个忠实化开关的 A/B

| 配置 | 音频看未来(帧) | 视频看未来(帧) | 与 `default` 输出 max\|Δ\| | 单窗耗时(ms) |
|---|---|---|---|---|
| `default` | 125.0 | 25.0 | 0（`sha256[:16]=ad0532a6e994921f`；自身比较） | 429 |
| `paper_causal` | 125.0 | 25.0 | 13.01 | 428 |
| `video_lstm_batch_first` | 125.0 | 25.0 | 1.295e−02 | 432 |
| `both` | 125.0 | 25.0 | 13.01 | 457 |

`n_audio_keys` 四组均为 525（strict 装载成功）。`video_lstm_batch_first` 打开后，扰动第 25 帧
影响 1/50 → **7/50** 帧（>1e−6·max 口径 5/50），本次 B=1 输入在默认配置下帧间无递推；B>1 时默认 LSTM 沿批次轴递归。

### `ln-variant.json` — `cLNhw` → 逐 (t,f) LayerNorm

| 变体 | 帧 0..120 最小响应 | 首响应帧 | 回看帧数 | 2s-vs-4s SI-SNR | 输出 rms |
|---|---|---|---|---|---|
| `as_shipped` | −40.2 dB | 0 | 218 | 10.12 dB | 0.00041 |
| `ln_c` | **−240 dB（区间最小值）** | 108 | 110 | 10.04 dB | 0.06978（**169.5×**） |

替换 30 个 `LayerNormalizationHW` → `LayerNormalization4D`（模型内 4D 实例 90 个）。该区间最大响应为 −75.29 dB，首响应帧为 108。
`meta.params = 537,762` 是 **state_dict 张量元素和（含 buffer）**；正文的 537,697 是
`nn.Parameters` 口径，差 65，**不是矛盾**。

### `streaming-consistency.json` — 分窗拼接 vs 整段（6 s，3 条同源片段，无 crossfade）

| guard | 80 ms | 160 ms | 320 ms | 456 ms | 640 ms |
|---|---|---|---|---|---|
| SI-SNR vs 整段 | **18.20** | 18.74 | 18.72 | 18.45 | 16.69 |
| r | 0.9925 | 0.9934 | 0.9933 | 0.9929 | 0.9895 |

换窗对齐 ±250 ms：+17.81 dB；两两 19.83 dB。`checkpoint_sha256` 已写进 JSON 本体；
`parameters = 11,724,385`（= 音频 537,697 + FRCNN 11,186,688）。
发射区间从 `first_end=32000` 起算，跳过 Studio 的冷启动首窗；两项实验还采用不同的视频帧索引。

### `video-ablation.json` — 视觉消融（N=40，`select=spread`，40 个不同 mouth 标识）

| 条件 | mean SI-SDRi | 相对 real 的成对 Δ（95% CI） | 更好/更差 |
|---|---|---|---|
| real（真口型） | **+7.113** | — | — |
| zero_feat（预处理后置零 = 中灰） | −8.871 | **−15.98** [−19.71, −12.26] | 1/39 |
| black_pix（原始像素 0） | +3.949 | **−3.16** [−4.67, −1.66] | 3/37 |
| wrong_speaker（替换 mouth，身份待核实） | −4.563 | **−11.68** [−14.98, −8.37] | 0/40 |
| mixture（输入本身） | +5.340 | — | — |

`forward_verify_max_abs_err = 0.0`（快速路径与 `model.forward` 数值一致），`skipped = []`，
`elapsed_s = 93.16` < 120 s（未分片）。原始 CSV 当前尚未找到；§1 的 `--out` 可生成新的明细。

### `canonical.json` / `canonical-6s.json` / `canonical-4s.json` — 流式仲裁

| 口径 | 6 s 流 | 4 s 流 |
|---|---|---|
| Studio 从 t=0（**含冷启动首窗**） | **+14.71 dB** / r 0.9835 / 峰值偏差 53.4% | **+14.14** / 0.9813 / 53.9% |
| 剔除冷启动首窗（steady） | **+16.72** / 0.9895 / 28.5% | **+17.52** / 0.9913 / 20.7% |
| 输出块硬切（无 20 ms 块间淡化） | +14.72 | +14.14 |
| 2 s 窗 / 2 s hop 参考平铺 vs 整段 | +18.66 | +23.18 |
| 任意两种窗对齐互相 SI-SNR（最小） | **17.02**（−62.5 / +62.5 ms） | **17.05** |
| guard 80→640 ms（同一 0–5.0 s 区间） | 14.51 / 15.19 / 15.01 / **16.71** / 15.57 | 13.73 / 14.95 / 14.92 / **17.58** / 15.99 |

`canonical.json` 是这两份原始文件的**摘要索引**（含 ckpt sha256、脚本来源、逐项数字）；
原始逐块与逐对齐明细在 `canonical-6s.json` / `canonical-4s.json`。

## 3 与正文引用值对照（哪些一致、哪些要改）

| 量 | 本目录（可复现） | 正文/其他文档 | 判断 |
|---|---|---|---|
| 开关 A/B：`paper_causal` 改输出 | max\|Δ\|=13.01，音频「看未来」125 帧不变 | 00-SYNTHESIS/01/02/11：整链几何 **113 帧/904 ms**；as-shipped 窗尾 ≤2000 ms | **一致（口径不同）**：125 帧是含 `cLNhw` 全窗耦合的端到端量，113 帧是隔离归一化后的几何量，不可互换 |
| `cLNhw` 非因果 | 30 个实例；帧 0 即响应（−40.2 dB） | 正文「30 处 `cLNhw`」、−40.2 dB | **一致** |
| `cLNhw` 回看帧数 | 218 → 110（尾部 32 帧区块扰动） | 12 §2：单帧扰动 57 帧（单遍）/ 113（整链） | **不冲突**：回看随扰动位置变化，见脚本自身说明 |
| 跨窗一致性 | 6 s：**14.71**（含首窗）/ **16.72**（稳态）；4 s：**14.14 / 17.52** | STATUS §4.6、README、12 §4.7、00-SYNTHESIS 已统一为本目录值 | **已一致（2026-11-08 回填）**：旧稿 +10.83 出自未入库的 `/tmp/w212/xwin.py`，已作废 |
| 跨窗一致性（studio/README 口径） | `streaming_consistency` guard 80 → **18.20 / 0.9925 / 0.188** | `studio/README.md` 写 +18.2 / 0.9925 / 0.188 | **逐位一致** |
| guard 收益 | 丢弃窗尾 80→456 ms：+14.51→+16.71（+2.2 dB）；补右上下文 80→640 ms：+0.8 dB | 正文已改为同一口径（+2.2 / +0.8 dB） | **已一致** |
| 视觉消融 | −15.98 / −11.68 / −3.16（N=40） | P2B 旧口径 N=300：−15.089 / −12.894 / −4.375 | **同向、同量级**；deepdive 正文尚未引用这组数（本目录即为其出处） |
| 视觉参数量 | `parameters=11,724,385`（音频 537,697 + FRCNN 11,186,688） | 03-video-frontends「FRCNN 11.19 M」 | **一致** |

## 4 仲裁脚本入库（`tools/repro/experiments/streaming_canonical.py`）

* 来源：`falcon-deepdive-20261108/_scratch/R6/canonical.py`，149 行，
  sha256 `31d0877ebb59b057fa1a890b0d4e9f87e3c058a3b2a0b9bc4dc97e8481214f05`。
* 只改三处**非数值**内容：① 补 shebang 与来源/用途 docstring；② `sys.path` 从 `/tmp/w212`
  换成同目录 `_falcon_cpu_common.py`（原 `w212_common.py` 的随仓库副本，仅把仓库根改为按
  `__file__` 推导）；③ 加 `--json` / `--seconds`。**计算逻辑逐行未改。**
* 复跑校验：`--seconds 6` / `--seconds 4` 与 `/tmp/w213/out_{6,4}s.json` 对比，
  **仅 `elapsed_s` 不同**（75.8 vs 88.6；60.3 vs 56.0），其余字段逐位相同。
* 确定性旁证：`lookahead_ab` / `ln_variant` / `streaming_consistency` 各跑两遍，
  除计时字段（`ms`、`inference_seconds`、`whole_pass_seconds`）外**所有指标逐位相同**。

## 5 复跑

```bash
# 前置见 §1；把 <out> 换成你要落盘的位置
$ENV tools/repro/experiments/lookahead_ab.py --json <out>/lookahead-ab.json
$ENV tools/repro/experiments/ln_variant.py --variant both --allow-dirty --json <out>/ln-variant.json
$ENV tools/repro/experiments/streaming_consistency.py --json <out>/streaming-consistency.json
$ENV tools/repro/experiments/video_ablation_check.py --stage variants --n 40 --select spread --out <out>/video
$ENV tools/repro/experiments/streaming_canonical.py 6 --json <out>/canonical-6s.json
$ENV tools/repro/experiments/streaming_canonical.py 4 --json <out>/canonical-4s.json
```

`git check-ignore -v docs/verification/deepdive-20261108/*.json` **无输出**（未被忽略），
`.gitignore` 未改动。
