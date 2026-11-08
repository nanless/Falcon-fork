# Falcon-fork 深挖轮 · Lead 基线事实（2026-11-08）

> 本轮子代理先读本文件。这里只写已核实的事实、口径和本轮的统一写法；领域细节由各子代理自己挖。
> 2026-11-08 Lead 复核后补入三条统一事实：MACs 两把尺子、lookahead 三档、论文对齐；00–06 各篇以这三条为准。

## 0. 本轮的边界

1. **目标仓库在远端**：`dev_a800_2gpus:/root/code/github_repos/Falcon-fork`；本机 cwd 只用来写报告。访问走
   `ssh -o BatchMode=yes dev_a800_2gpus '<命令>'`（免密已通，root 账号），长命令加 `timeout 120`。仓库不 rsync 到本机：
   模型代码在远端读更快，也免得把 62 G 数据卷进来。
2. **仓库只读**：本轮不改、不移、不删仓库内文件，也不跑 `git commit/checkout/stash/clean`。这轮只做核对，
   报告写在本机 `falcon-deepdive-20261108/`；动了工作区，`git status` 就没法再当"未改代码"的证据。
3. **Python 用 `falcon` conda 环境**：`/root/miniforge3/envs/falcon/bin/python`，`PYTHONPATH=$PWD`。
   `/usr/bin/python3` 与 `/root/.venvs/falcon` 缺本仓库钉死的依赖（`typeguard==2.13.3` 等，AGENTS.md 有记录），
   用它们跑出来的形状/参数量结论不可信。
4. **计算量上限**：不启动训练、不做全量评测、不下载数据、不装包。允许读代码、`grep`、`git log/show`、
   形状/参数量/张量流级别的 CPU 单窗前向（`torch.no_grad()`）、跑已存在的单测；单次命令 ≤120 s，
   `CUDA_VISIBLE_DEVICES=` 置空。2 张 A800 留给正式任务。
5. **先拨乱反正**：`docs/` 下已有 2026-10-24 deepdive 9 篇、2026-10-30 三轮 review、STATUS.md。可以读、可以引用，
   但既有结论不能直接当结论用——每条关键结论回到源码或证据文件复核，冲突时以代码为准，并在报告里单列
   「与既有文档的冲突」。

## 1. 项目一句话

Falcon（arXiv:2610.32749，JusperLee/Falcon，2026 年因果音视频语音分离）在 LRS2-2Mix 上做**窗口级因果的音视频
目标说话人提取**：输入混合语音 + 目标说话人唇部视频，输出**单路**目标语音（`n_src: 1`，不是 2 源分离）。
音频主干 0.5 M 参数 / 10.6 G MAC。本仓库 `nanless/Falcon-fork` 是「复现 + 工程化（Studio / HF Space / 项目页）」的 fork。
论文 2026-10-23 已在 arXiv 公开（arXiv:2610.32749），11 号报告做过逐条对齐。

## 2. 已核实事实

| 项 | 值 |
|---|---|
| HEAD | `6c1d5b7` (2026-10-30 18:05 +0000) `STATUS：回填 HEAD=2062528 与 freeze tag 说明` |
| 前两个提交 | `2062528` 复现维护轮修复；`0fdad70` 三轮 review |
| 分支/远端 | `main`；`origin=ssh://git@ssh.github.com:443/nanless/Falcon-fork.git`；`upstream=https://gh-proxy.com/https://github.com/JusperLee/Falcon.git`（已合并 439cbe9） |
| tags | `repro-freeze-20261031`、`repro-phase2-freeze-20261024` |
| 工作区 | `git status` 干净（无输出） |
| 规模 | 923 个非 .git 文件，py 39529 行；其中 `look2hear/**` py 约 8.5k 行（含顶层入口） |
| 环境 | `/root/miniforge3/envs/falcon`：Py 3.11.17、torch 2.8.0+cu128、pytorch-lightning 2.6.6、typeguard 2.13.3 |
| GPU | 2× A800-SXM4-80GB，当前 0 MiB 占用 |
| 权重 | `/root/.cache/falcon_weights/{falcon.pt, falcon-big.pt, frcnn_128_512.backbone.pth.tar, ...big...}`（软链到 group-shared） |
| 数据 | `/root/data/{lrs2_2mix, falcon_lrs2_2mix, falcon_lrs2_2mix_paper, falcon_realssa_sim, falcon_realssa_sim_ch0, RealSSA-Real, RealSSA-Sim, falcon_eval, rir, lists, pretrained_models, models}` |
| 训练产物 | `/root/tmp/p2d_train`、`/root/tmp/p2d_near_train`（见 P2D-report §8） |
| 磁盘 | `/` overlay 20G（16G avail）；`/root` 164T VEPFS（17T avail） |
| MACs（音频，ptflops） | **10.611 GMAC**：含 LSTM、不含 FRCNN，与论文 10.6 G 同尺 |
| MACs（音频，`torch.utils.flop_counter`） | 5.382 GMAC：不记 `aten::lstm`，是另一把尺子，不可与 10.6 并列 |
| MACs（视觉，88×88×50 帧） | **15.897 GMAC** |
| 整窗 MACs | **26.51 GMAC，视觉占 60.0%**（10.611 + 15.897，同尺） |
| lookahead（单遍几何） | 57 帧 / 456 ms：关掉 `cLNhw` 后一遍 AudioBlock 的感受野 |
| lookahead（整链几何） | **113 帧 / 904 ms**；另一路实测 110 帧 / 880 ms，差约 3% |
| lookahead（as-shipped） | 窗尾 ≤2000 ms：30 处 `cLNhw` 在整窗上统计 |
| 论文对齐 | `causal=False` 硬编码在 `Falcon.py:772/808/821`（改约 6 行可回到 120 ms）；`VideoBlock` LSTM 布局错；整窗非帧级因果 |

## 3. 仓库骨架

```
AGENTS.md / README.md / README_zh.md / agents.md(软链)（该软链已于 2026-11-09 删除，入口只保留 `AGENTS.md`）
audio_train.py (456L)  audio_visual_test.py (875L)  evaluated_mac_params.py (5.3K)
configs/{Falcon,LRS2_2Mix,LRS2_2Mix_Paper_2gpu,LRS2_2Mix_Paper_2gpu_smoke,LRS2_2Mix_Paper_8gpu,LRS2_2Mix_Paper_v2}.yml
look2hear/
  models/{Falcon.py 1524L, Falcon_Contrastive.py 129L, base_model.py 108L, __init__.py 44L}
  layers/{cnnlayers 884L, rnnlayers 908L, stft 785L, enc_dec 461L, normalizations 245L, stft_tfgn 244L, activations 67L, __init__ 50L}
  system/{av_litmodule 247L, av_litmodule_nass_v2 185L, av_litmodule_mdcl 173L, optimizers 106L, schedulers 111L}
  datas/{avspeech_dataset 334L, avspeech_dataset_near 337L, check_avspeech_dataset 349L, transform 187L}
  losses/{matrix, mixit, nass_contrastive_loss, pit_wrapper, sisnri}
  metrics/{allwrapper, core, splitwrapper, wrapper}
  utils/{attn, complex_utils, get_layer_from_string, inversible_interface, lightning_utils, nets_utils, parser_utils, separator, stft, torch_utils}
  videomodels/{frcnn_videomodel, light_videomodel, resnet, resnet1D, resnet_videomodel, shufflenetv2, wujian_videomodel}
DataPreProcess/  Experiments/checkpoint/  studio/{falcon_model,configs,scripts,tests,web}  space/  project_page/
tools/{repro/{phase2,tests},...}  docs/{understanding,reviews,verification,*.md}  assets/  checkpoints/
```

## 4. 既有文档地图（可读，需校验）

- `docs/STATUS.md` — 单一事实源（复现矩阵/口径/待办），最后更新 2026-10-31。
- `docs/understanding/deepdive-20261024/00-SYNTHESIS.md` + 01–09（上轮 9 子代理深挖）。
- `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md`、`PAPER_ALIGNMENT_20261030.md`、`THREE_ROUND_REVIEW_20261030.md`。
- `docs/reviews/round{1,2}/*.md`（交叉评审）、`docs/verification/**`（脚本+日志+CSV 证据）。
- `docs/REPRODUCTION_PLAN.md`、`docs/REPRODUCTION_PHASE{1,2}_REPORT.md`、`docs/DATA_ACCESS_GUIDE.md`、`docs/PHASE2_VOLCANO_LRS2_TRAINING.md`。

## 5. 报告格式（所有子代理统一）

写到 `falcon-deepdive-20261108/NN-<topic>.md`，结构：

```
# NN <topic>
## 0. 结论速览（≤15 行，可执行的判断，不是目录）
## 1. 事实基线（文件/行号/命令输出为证）
## 2. 机制拆解（逐层：输入→张量形状→算子→输出；含公式/伪代码）
## 3. 关键数据/参数表（表格）
## 4. 实测验证（跑了什么命令，输出是什么，附关键 stdout 片段）
## 5. 坑与反直觉点（每条含 file:line）
## 6. 与既有文档的冲突/更正（逐条：旧说法 → 实际）
## 7. GAP LIST（还没弄懂的具体问题，写成可被下一轮直接认领的任务，≤10 条）
## 8. 证据清单（用到的文件路径、日志、CSV、命令）
```

要求：
- 每条非平凡结论带 `path:line`（可用 `sed -n`/`grep -n` 取证）。没有行号的结论，下一轮无法复核。
- 数字给口径（哪份数据集、多少句、哪个 ckpt、哪个 commit）。同一个数字换一把尺子就不能并列写。
- 不确定的写进 GAP LIST，不编。
- 报告 400–900 行，信息密度优先，不复述 README。
