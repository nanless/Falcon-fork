# `tools/repro/experiments/` — 复现实验脚本索引

> 本目录的 5 个脚本是正文里那些**量化结论**的可执行出处。它们在此之前只被脚本自己的
> docstring 提到（"零引用死文件"），本文件负责把它们接进索引；实测证据落在
> [`docs/verification/deepdive-20261108/`](../../../docs/verification/deepdive-20261108/)。
>
> 共同约束：**纯 CPU**（`CUDA_VISIBLE_DEVICES=`）、`eval()` + `no_grad()`、固定 seed、
> **不训练**、默认只写 `/tmp`，不写仓库。权重固定 `/root/.cache/falcon_weights/falcon.pt`
> （`sha256 a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`，epoch 714）。

通用前置（conda `falcon`：Python 3.11.17 / torch 2.8.0+cu128）：

```bash
REPO=/root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
cd $REPO
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=    # 4 线程，纯 CPU
```

## 脚本一览

| 脚本 | 解决什么问题 | 典型耗时（4 线程） | 产出怎么落盘 |
|---|---|---|---|
| [`lookahead_ab.py`](lookahead_ab.py) | `paper_causal` / `video_lstm_batch_first` 两个忠实化开关打开后到底改了什么：端到端「看未来帧数」、与发布输出的 `max\|Δ\|`、逐卷积几何诊断、视频 LSTM 是否沿帧轴递推 | 约 25 s | `--json <path>`；不给则只打印 Markdown 表格 |
| [`ln_variant.py`](ln_variant.py) | 把 `cLNhw`（`GroupNorm(1,C)` 全窗统计）换成逐 `(t,f)` 的 LayerNorm4D 后，窗内因果能恢复多少、不重训的代价多大 | 约 13 s | `--json <path>`；`look2hear/` 有未提交改动时需加 `--allow-dirty` |
| [`streaming_consistency.py`](streaming_consistency.py) | Studio 的 2 s 窗 / 0.5 s hop 分窗拼接，与整段一次前向差多少；guard 与窗对齐各扫一遍 | 约 35 s | `--json <path>` |
| [`video_ablation_check.py`](video_ablation_check.py) | 视觉分支贡献多少：真口型 / 置零 / 黑帧 / 换说话人 4 条件配对比较（`--n 40 --select spread`） | 约 95 s（40 条 × 4 条件） | `--out <dir>`，写 `video_ablation_per_sample.csv` + `summary.{json,csv,txt}`；默认 `/tmp/falcon_video_ablation` |
| [`streaming_canonical.py`](streaming_canonical.py) | 仲裁口径（深挖轮 `R6/canonical.py` 的逐行副本）：同一段流的 Studio 拼接 / 剔冷启动 / 硬切 / 参考平铺 / guard / 对齐六组量 | 约 60–90 s（6 s 流）、约 55–60 s（4 s 流） | `--json <path>`；缺省 `/tmp/w213/out_<N>s.json`；依赖同目录 [`_falcon_cpu_common.py`](_falcon_cpu_common.py) |

## 典型用法

```bash
# 1. 两个忠实化开关的 A/B（表格 + JSON）
$ENV tools/repro/experiments/lookahead_ab.py --json /tmp/lookahead.json

# 2. cLNhw 变体的因果对照
$ENV tools/repro/experiments/ln_variant.py --variant both --allow-dirty --json /tmp/ln.json

# 3. 分窗拼接 vs 整段（默认 6 s 流、guard 80/160/320/456/640 ms）
$ENV tools/repro/experiments/streaming_consistency.py --json /tmp/streaming.json

# 4. 视觉消融 N=40，均匀取样（>120 s 时按 --start 分片）
$ENV tools/repro/experiments/video_ablation_check.py --stage variants --n 40 --select spread --out /tmp/video

# 5. 仲裁口径：Studio 拼接流式 vs 整段前向
$ENV tools/repro/experiments/streaming_canonical.py 6 --json /tmp/canonical-6s.json
$ENV tools/repro/experiments/streaming_canonical.py 4 --json /tmp/canonical-4s.json
```

每个脚本都支持 `--help`（5 个全部 `exit 0`）。把结果**作为仓库证据**时，不要改脚本默认路径，
而是把 `--json` / `--out` 指到 `docs/verification/<轮次>/` 下，并在该目录的 `README.md` 里
记下命令、权重 sha256、日期与线程数——`docs/verification/deepdive-20261108/` 就是这么做的。

## 两个已知坑

* **`streaming_consistency.py` 与 `streaming_canonical.py` 不是同一协议**：前者从
  `first_end = 32000` 起算（跳过 Studio 的冷启动首窗）且无 crossfade，6 s 得 **+18.20 dB**；
  后者从 `t=0` 起算（含 1.42 s 静音首窗）并带 10 ms join 淡化，6 s 得 **+14.71 dB**，
  剔除首窗后 **+16.72 dB**。引用时必须写清口径。
* **`ln_variant.py` 的「回看帧数」随扰动位置变化**：尾部 32 帧区块扰动测得 `218 → 110` 帧，
  单帧扰动测得 57 帧（单遍几何）。这不是同一物理量，别混用。
