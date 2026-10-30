# LRS2-2Mix 论文配方训练（8 卡 / 火山引擎）Runbook

> 目标：复现论文 LRS2-2Mix 列 **SI-SNRi 14.1 / SDRi 14.7**。
> 背景：官方发布权重是 LRS3 训练的（`LRS3_complex_nonmoving`，epoch 714），在 LRS2-2Mix 上 7.03/7.71 属跨语料；
> 论文表格 “under our training protocol” 按 benchmark 分别训练，需要 LRS2 专属 ckpt（已向作者提 issue：
> <https://github.com/JusperLee/Falcon/issues/1>）。
> **2026-10-30 校正**：论文 arXiv:2610.32749 已公开，App D Table 16 给出训练配方（我们此前只能从 config 推断）。
> 本 runbook 已切到 **v2 = Table 16 配方**（`configs/LRS2_2Mix_Paper_v2.yml`）；旧
> `configs/LRS2_2Mix_Paper_8gpu.yml`（bs5 / −SNR / τ200）保留并标记 **LEGACY**，只用于复跑旧实验。
> 论文原文要点见 `docs/understanding/PAPER_ALIGNMENT_20261030.md` §2.1。

## 1. 配方（论文 App D Table 16，v2）

| 项 | 论文 Table 16 | v2 配置（`configs/LRS2_2Mix_Paper_v2.yml`） | LEGACY（`..._8gpu.yml`，保留） |
|---|---|---|---|
| 并行/硬件 | **8×RTX3090**，DDP | 8 卡、DDP（`gpus: -1`） | 8 卡 |
| batch | **4/GPU ⇒ global 32** | `batch_size: 4` ⇒ 8×4 = **32** | 5/GPU ⇒ 40 |
| 重建损失 | **`L_recon = −SI-SNR`** | `loss.train.sdr_type: pairwise_neg_sisdr`（`threshold_byloss: false`，逐对均值不截断） | `pairwise_neg_snr` |
| 对比温度 τ（代码里叫 β） | **0→1 线性退火，前 1000 epoch**，之后固定 1 | `spk_contrast.beta_anneal_epochs: 1000` | 200 |
| 对比权重 / margin | λ_spk = 1.0 / m = 0.5 | `loss_weight: 1.0` / `margin: 0.5` | 同 |
| optimizer | AdamW lr 1e-3 / wd 0.01 | 同 | 同 |
| scheduler | ReduceLROnPlateau(patience 15, factor 0.5) | 同 | 同 |
| 梯度裁剪 | 5.0 | `gradient_clip_val=5.0`（`audio_train.py` 硬编码） | 同 |
| epochs / early stop | 1000 / patience 30 | `epochs: 1000`、`early_stop.patience: 30` | 同 |
| seed | 42 | `training.seed: 42` → `audio_train.set_seed()` → `pl.seed_everything(seed, workers=True)`，启动日志打印 | 未设置 |
| train_samples | 未说明（沿用） | 10,000 /epoch | 同 |
| 数据 | LRS2-2Mix `tr` + 双向 target | 见 §2（`falcon_lrs2_2mix_paper`，40k） | 同 |
| 视觉骨干 | FRCNN 冻结 | 从 `falcon.pt` 导出 | 同 |

**v2 vs LEGACY 的三个实质差异**：① 训练损失 −SI-SNR（v2）vs −SNR（legacy）；② τ（β）退火 **1000 epoch**（v2）
vs 200（legacy）；③ bs **4/GPU ⇒ global 32**（v2，与论文一致）vs bs5 ⇒ global 40（legacy，为凑旧推断的 10×4=40）。
`exp_name` 已区分（`LRS2_2Mix_Paper_v2` / `LRS2_2Mix_Paper_8gpu`），**两套结果不可混用**。

**双向 target（沿用旧方案，论文未说明其 target 采样）**：LRS2-2Mix 中 s1 恒为较响说话人（snr1>0），若只训练
target=s1，模型可走“选更响”捷径。我们的数据构建对每条 mixture 生成两条样本：
`(target=s1, mouth=s1, near=s2)` 与 `(target=s2, mouth=s2, near=s1)`，训练样本 40,000 条。

**near 负样本**：该数据集没有 near-field 泄漏源；用**另一说话人（干扰源）**作为 NASS 对比损失的负样本，
这是可用的近似（论文若用 near-field 模拟，指标可能更高）。注意 K=1 时 softmax 权重恒为 1 → **τ 退火在 LRS2 上
实际不生效**；论文的 τ 消融只能在 RealSSA-Sim（K≥2）上复现。报告中必须声明该差异。

## 2. 数据准备（在火山引擎节点上执行）

LRS2-2Mix 是公开数据（HF `JusperLee/LRS2-2Mix`，`lrs2.tar.gz` 16.47 GB）：

```bash
# 1) 下载并解压（16.5GB → 17.4GB 成员）
#    有镜像用镜像；没有则 HF 直连/proxy
curl -L -o lrs2.tar.gz "https://huggingface.co/datasets/JusperLee/LRS2-2Mix/resolve/main/lrs2.tar.gz"
mkdir -p /data/lrs2_2mix && tar xzf lrs2.tar.gz -C /data/lrs2_2mix   # 得到 lrs2_rebuild/

# 2) 构建“论文配方”数据（双向 target + near；音频默认只被 JSON 引用，不复制）
git clone https://github.com/nanless/Falcon-fork   # 或你自己的 fork
cd Falcon-fork
$PYTHON_BIN tools/repro/phase2/build_lrs2_paper_data.py \
  --src /data/lrs2_2mix/lrs2_rebuild --out /data/falcon_lrs2_2mix_paper --jobs 32
# 期望：tr 40000 / cv 10000 / tt 6000 条；mouths 18,691 / 476 / 267 个 .pt；共 ~26GB
# 构建末尾自动跑 validate_build 闸门（计数 / target≠near / mouth 说话人==target 说话人 /
# near 非自引用覆盖 100%），任何不符打印细节并 exit 2。
```

**自包含数据（`--copy-wavs`，跨机迁移/火山引擎节点推荐）**：加 `--copy-wavs` 后每个 mixture 复制
**5 份 wav**（mix 1 + target 2 + near 2），不再是 1 份；磁盘增量 **约 +9 GB**（旧文档的 +7 GB 已作废）。
布局（review A5-1 修复：s1/s2 分名，避免同名文件互相覆盖/静默错配）：

```
<out>/<split>/mix/<stem>.wav
<out>/<split>/target/<stem>.s1.wav      # target=s1 的条目
<out>/<split>/target/<stem>.s2.wav      # target=s2 的条目
<out>/<split>/near/<stem>.s1.wav        # target=s2 时的负样本（s1 波形）
<out>/<split>/near/<stem>.s2.wav        # target=s1 时的负样本（s2 波形）
```

即 **near 永远指向“另一说话人”**（target=s1 → near=s2；target=s2 → near=s1），K=1，near 非自引用；
target wav 与 mouth `.pt` 的说话人一致。构建后必须过两道闸门：

```bash
# 闸门 1：构建脚本内置 validate_build（exit 2 = 标签/计数/覆盖不自洽，禁止拿去训练）
$PYTHON_BIN tools/repro/phase2/build_lrs2_paper_data.py \
  --src /data/lrs2_2mix/lrs2_rebuild --out /data/falcon_lrs2_2mix_paper \
  --jobs 32 --copy-wavs

# 闸门 2：独立审计（要求每 split coverage=100.00%、fallback=0、exit 0）
$PYTHON_BIN tools/repro/audit_data.py /data/falcon_lrs2_2mix_paper --full
# 参数是位置参数 data_root；需要机器可读结果时追加 --json <path>
```

> 我们的本地构建：`/root/data/falcon_lrs2_2mix_paper`（26GB，默认不复制音频、JSON 引用
> `/root/data/lrs2_2mix/lrs2_rebuild` 的音频）。跨节点复现请用 `--copy-wavs`（+9GB）。

## 3. 环境

与仓库 `AGENTS.md` §5 一致：Python 3.11、torch 2.8.0+cu128、pytorch-lightning 2.6.6、typeguard<3、timm、fast-bss-eval、
soundfile、pypesq（pesq shim）等。**本文所有命令统一用 conda `falcon` 解释器**（禁止系统 `python3`）：

```bash
PYTHON_BIN=/root/miniforge3/envs/falcon/bin/python
$PYTHON_BIN -m pip install -r docs/verification/conda-env-lock.txt   # 先装好匹配 CUDA 的 torch
# 自检
$PYTHON_BIN -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # 期望 8
```

## 4. 视觉骨干预训练权重

```bash
$PYTHON_BIN tools/repro/extract_video_backbone.py \
  --ckpt /root/.cache/falcon_weights/falcon.pt \
  --out  /root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar
```
（`falcon.pt` 从 HF `JusperLee/Falcon` 下载；config 里 `videonet.videonet_config.pretrain` 指向该文件。）

## 5. 启动训练（8 卡，v2）

```bash
cd /path/to/Falcon-fork
CONF=configs/LRS2_2Mix_Paper_v2.yml \
DATA_ROOT=/data/falcon_lrs2_2mix_paper \
NPROC=8 \
bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
```

脚本（2026-10-30 加固）现在会：
1. 按 `NPROC` 设置 `CUDA_VISIBLE_DEVICES=0..N-1`，并断言 `NPROC == torch.cuda.device_count()`（不等直接退出）；
2. 用 `$PYTHON_BIN` 把 `DATA_ROOT` **真正渲染**进一份临时 conf（`$TMPDIR/conf_<pid>.yml` 的
   `train_dir/valid_dir/test_dir` 三处），训练读这份 conf，并打印**生效**的三条路径——不再出现
   “DATA_ROOT 只用于自检、训练仍读 config 里写死路径”的错位；
3. 自检 `{tr,cv,tt}/{mix,target,near}.json` 与视觉骨干（骨干路径从渲染后的 conf 读取）；
4. 打印 `global batch = batch_size × NPROC`（v2 默认 4×8=**32**）与每 epoch 的 optimizer step 估算；
5. 后台启动 `audio_train.py --conf_dir <rendered>`，日志/PID/渲染后 conf 路径落盘
   （`LRS2-paper.log`、`LRS2-paper.pid`、`LRS2-paper.confpath`）。

**先在节点上做一次不启动自检（推荐）**：

```bash
DRY_RUN=1 DATA_ROOT=/data/falcon_lrs2_2mix_paper NPROC=8 \
  bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
```

平台要求单进程入口时也可直接跑（**注意：直接跑不会渲染 DATA_ROOT**，需自己把 config 的
`datamodule.data_config.*_dir` 指到真实路径；或复用 `$TMPDIR/conf_<pid>.yml`）：

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON_BIN audio_train.py \
  --conf_dir <rendered_conf.yml>
```

（PL 2.x 在 `gpus: -1` + 8 可见卡时自动 DDP；`training.seed: 42` 会在启动日志打印 `seed = 42`。）

## 6. 监控 / 续训 / 导出

```bash
# 监控（val SI-SNR、最佳 epoch；按 glob version_* 取最新，续训后的 version_1/2 也能读到）
$PYTHON_BIN tools/repro/phase2/monitor_lrs2_paper.py --exp LRS2_2Mix_Paper_v2 --watch 300

# 续训（中断后从 last.ckpt 恢复）——路径不存在会直接 SystemExit，不再静默从头训练
$PYTHON_BIN audio_train.py --conf_dir <rendered_conf.yml> --ckpt_path <exp_dir>/last.ckpt
# 或：CKPT_PATH=<exp_dir>/last.ckpt bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh

# 导出 ckpt 后用 Phase-1 修好的评测器在 LRS2-2Mix test 上评测（目标 14.1/14.7）
PYTHONPATH=$PWD $PYTHON_BIN audio_visual_test.py --conf_dir configs/LRS2_2Mix.yml \
  --data_dir /root/data/falcon_lrs2_2mix --ckpt_path <最佳.ckpt> --gpus 0
```

> 注意 1：训练数据用 `falcon_lrs2_2mix_paper`（双向 target，40k 条）；**评测**仍用 `falcon_lrs2_2mix/tt`
> （3000 条，target=s1，与 benchmark 约定一致；目标约定已由 s1/s2 对照确证）。
> 注意 2：续训时 CSVLogger 会新建 `version_<N+1>`，monitor 已按 `version_*` 取最新；
> 若 checkpoint 根目录不在默认的 `/root/tmp/*/Experiments/checkpoint` 与 `./Experiments/checkpoint`
> 下，用 `--root <dir>`（可重复）追加搜索根。`--ckpt_path` 缺失会被
> `audio_train.resolve_resume_checkpoint()` 拦成 `SystemExit`（A3-04）——
> 这是为了防止“以为在续训、实际从零训练”的长时间误跑。

## 7. 时间估算（修正版，2026-10-30）

| 方案 | 每 epoch（含 cv+tt 验证） | 100 epoch | 1000 epoch（上限） | 依据 |
|---|---|---|---|---|
| 单卡 A800 bs4 | ≈14.7 min（2500 step + cv5000/tt3000） | **≈24.5 h** | ≈245 h（≈10 天，不可行） | P2-D 实测 metrics.csv |
| 2 卡 A800 bs4 | ≈7–10 min | ≈12–17 h | **≈120 h**（≈5 天） | 2 卡 smoke 2.97 it/s 外推 |
| 8 卡（v2：bs4 ⇒ global 32） | ≈2–3 min | ≈3–5 h | **≈33–50 h** | 8 卡吞吐外推 |

- EarlyStopping(patience 30) 通常 150–300 epoch 收敛 ⇒ 8 卡 **约 5–15 h**，2 卡 **约 18–50 h**。
- 结论：**只有 8 卡节点现实**；单卡 1000 epoch ≈10 天、2 卡 ≈5 天，仅适合做趋势/冒烟。
- 旧版“8 卡 33–50 h”不变；旧版“2 卡≈1/4 吞吐”的模糊表述改为上表口径（2 卡约 120 h / 1000 ep）。

## 8. 风险与差异声明

1. **配方版本**：v2 = 论文 Table 16（−SI-SNR、τ/1000、bs4/GPU ⇒ global 32、seed 42）；LEGACY = 旧 8 卡方案
   （−SNR、τ/200、bs5/GPU ⇒ global 40）。`exp_name` 已区分（`LRS2_2Mix_Paper_v2` vs `LRS2_2Mix_Paper_8gpu`），
   **两套结果不可混用**，报告必须写明用的哪份配置。
2. **硬件差异**：论文 8×RTX3090（24 GB），我们 8×A800（80 GB）。同 batch/同超参下数值预期等价，但 cuDNN/cuBLAS
   实现与吞吐不同；论文未声明确定性设置。
3. **LRS2-2Mix 无 near-field 源** ⇒ 用另一说话人作 NASS 负样本（非论文原始 near 模拟）；K=1 时 softmax 权重恒为 1，
   **τ 退火在 LRS2 上实际不生效**（论文 τ 消融只能在 RealSSA-Sim K≥2 上复现）。
4. **target 约定**：训练双向（s1/s2），评测按 benchmark 约定 target=s1；论文未说明训练侧 target 采样方式。
5. **seed 42**：论文 §5.2 给出，但完整确定性设置（cuDNN、数据顺序、worker seed）未给；单 seed 有方差，
   正式结论建议 3 seed（42/43/44）并报告均值±std。
6. **数据混合**：论文 LRS2 配方引用 prior work [6,17]，**未给 Falcon 自己的 LRS2 数据混合脚本**；我们的 40k
   双向 target 是推断，须在结果中声明。
7. 若作者回复 issue 提供 LRS2 ckpt/配置，优先用官方 ckpt 评测，再用本方案对照。
