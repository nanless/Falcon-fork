# LRS2-2Mix 论文配方训练（8 卡 / 火山引擎）Runbook

> **2026-11-10 说明**：本手册介绍候选训练方案。先读[数据集说明](understanding/20261110-datasets.md)、[数据与训练详解](understanding/20261110-data-and-training.md)和[数据访问指南](DATA_ACCESS_GUIDE.md)；真实机器、可见 GPU、目标路径与训练授权须在启动前核对。

> 目标：复现论文 LRS2-2Mix 列 **SI-SNRi 14.1 / SDRi 14.7**。
> 背景：历史配置名含 `LRS3_complex_nonmoving`、epoch 714，但已发布权重的完整训练谱系尚待作者或原始训练记录确认；在 LRS2-2Mix 上的 7.03/7.71 是历史评测结果；
> 论文表格的训练和目标口径仍需与作者确认，若希望复现 LRS2 专属训练则需要相应 checkpoint（已向作者提 issue：
> <https://github.com/JusperLee/Falcon/issues/1>）。
> **2026-10-30 校正**：论文 arXiv:2610.32749 已公开，App D Table 16 给出训练配方（我们此前只能从 config 推断）。
> 本 runbook 已切到 **v2：参考 Table 16 的候选配方**（`configs/LRS2_2Mix_Paper_v2.yml`）；旧
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
| 数据 | 目标方向采样未明确 | 见 §2（`falcon_lrs2_2mix_paper`，40k） | 同 |
| 视觉骨干 | FRCNN 冻结 | 从 `falcon.pt` 导出 | 同 |

**v2 vs LEGACY 的三个实质差异**：① 训练损失 −SI-SNR（v2）vs −SNR（legacy）；② τ（β）退火 **1000 epoch**（v2）
vs 200（legacy）；③ bs **4/GPU ⇒ global 32**（v2，与论文一致）vs bs5 ⇒ global 40（legacy，为凑旧推断的 10×4=40）。
`exp_name` 已区分（`LRS2_2Mix_Paper_v2` / `LRS2_2Mix_Paper_8gpu`），**两套结果不可混用**。

**双向 target（沿用旧方案，论文未说明其 target 采样）**：文件名中的正增益不保证 s1 的实际 RMS 恒高于 s2；[1,000 条能量抽样](verification/deepdive-20261110/能量比审查/summary.json)中有 26 条为负。仅用 s1 仍可能产生选择偏差，双向设计可供比较。我们的数据构建对每条 mixture 生成两条样本：
`(target=s1, mouth=s1, near=s2)` 与 `(target=s2, mouth=s2, near=s1)`，训练样本 40,000 条。

**near 负样本**：该数据集没有 near-field 泄漏源；用**另一说话人（干扰源）**作为 NASS 对比损失的负样本，
这是与原配方不同的近似，其指标需要通过实验测量。注意 K=1 时 softmax 权重恒为 1 → **τ 退火在 LRS2 上
实际不生效**；只有存在至少两个有效负样本时，才可能检验权重随温度变化的效果。报告中必须声明该差异。

## 2. 数据准备（在火山引擎节点上执行）

LRS2-2Mix 是公开数据（HF `JusperLee/LRS2-2Mix`，`lrs2.tar.gz` 16.47 GB）。以下代码块按同一 shell 会话顺序展示；`/data` 和源码目录是示例路径，运行前替换为平台实际路径。先将 `PYTHON_BIN` 设置为平台中已配置好的 Falcon Python 解释器；以下值只适用于本开发服务器，火山引擎节点应使用其实际环境路径：

```bash
export PYTHON_BIN=/root/miniforge3/envs/falcon/bin/python

# 下面下载、解压和构建会写盘，须先检查目标目录、空间和数据使用条件。
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

**复制音频（`--copy-wavs`）**：加 `--copy-wavs` 后每个 mixture 复制
**5 份 wav**（mix 1 + target 2 + near 2），不再是 1 份；磁盘增量 **约 +9 GB**（旧文档的 +7 GB 已作废）。
JSON 中记录的是构建时的绝对路径，因此文件集中复制并不保证换机器或换目录后索引仍可用；应在目标节点重建索引，或在隔离副本上明确改写路径并逐条验证。布局（review A5-1 修复：s1/s2 分名，避免同名文件互相覆盖/静默错配）：

```
<out>/<split>/mix/<stem>.wav
<out>/<split>/target/<stem>.s1.wav      # target=s1 的条目
<out>/<split>/target/<stem>.s2.wav      # target=s2 的条目
<out>/<split>/near/<stem>.s1.wav        # target=s2 时的负样本（s1 波形）
<out>/<split>/near/<stem>.s2.wav        # target=s1 时的负样本（s2 波形）
```

即 **near 永远指向“另一说话人”**（target=s1 → near=s2；target=s2 → near=s1），K=1，near 非自引用；
target wav 与 mouth `.pt` 的说话人一致。构建脚本末尾运行内置验证；随后可单独运行审计。不要在已有输出上再次执行构建：`--copy-wavs` 会复用已存在文件，而路径/计数检查不能证明波形身份。核对目标、嘴部和音频内容后再用于训练：

```bash
# 构建阶段已运行 validate_build，不重复执行构建
# 闸门 2：独立审计（要求每 split coverage=100.00%、fallback=0、exit 0）
$PYTHON_BIN tools/repro/audit_data.py /data/falcon_lrs2_2mix_paper --full
# 参数是位置参数 data_root；需要机器可读结果时追加 --json <path>
```

> 我们的本地构建：`/root/data/falcon_lrs2_2mix_paper`（26GB，默认不复制音频、JSON 引用
> `/root/data/lrs2_2mix/lrs2_rebuild` 的音频）。跨节点迁移需要同时处理 JSON 绝对路径；`--copy-wavs` 会增加约 9 GB，单独使用不能保证可移植。

## 3. 环境

与仓库 `AGENTS.md` §5 一致：Python 3.11、torch 2.8.0+cu128、pytorch-lightning 2.6.6、typeguard<3、timm、fast-bss-eval、
soundfile、pypesq（pesq shim）等。**本开发服务器示例使用 conda `falcon` 解释器**；其他平台请设置该平台的实际解释器，不能照搬本机绝对路径：

```bash
export PYTHON_BIN=/root/miniforge3/envs/falcon/bin/python
$PYTHON_BIN -m pip install -r docs/verification/conda-env-lock.txt   # 先装好匹配 CUDA 的 torch
# 自检
$PYTHON_BIN -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # 期望 8
```

## 4. 视觉骨干预训练权重

```bash
$PYTHON_BIN tools/repro/extract_video_backbone.py \
  --src /root/.cache/falcon_weights/falcon.pt \
  --dst  /root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar
```
（`falcon.pt` 从 HF `JusperLee/Falcon` 下载；config 里 `videonet.videonet_config.pretrain` 指向该文件。） 启动脚本缺骨干时打印的旧提示仍写 `--ckpt/--out`；工具实际只接受这里的 `--src/--dst`。

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

**节点上的不启动训练自检**：仍会检查 GPU 数量、创建目录并渲染临时配置；并非无副作用。脚本按 NPROC 重设 CUDA_VISIBLE_DEVICES，须先核对调度器的设备映射。当前双卡服务器不能通过 NPROC=8 的检查：

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

（8 卡时需核对 Lightning 实际设备和分布式策略；`training.seed: 42` 会在启动日志打印。脚本给出的 global batch 是 batch_size × NPROC，未将梯度累积计算在内。）

## 6. 监控 / 续训 / 导出

```bash
# 监控（val SI-SNR、最佳 epoch；按搜索结果的 version 数字选取；跨根目录不保证时间上最新）
$PYTHON_BIN tools/repro/phase2/monitor_lrs2_paper.py --exp LRS2_2Mix_Paper_v2 --watch 300

# 续训（中断后从 last.ckpt 恢复）——路径不存在会直接 SystemExit，不再静默从头训练
$PYTHON_BIN audio_train.py --conf_dir <rendered_conf.yml> --ckpt_path <exp_dir>/last.ckpt
# 或：CKPT_PATH=<exp_dir>/last.ckpt bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh

# 导出 ckpt 后用 Phase-1 修好的评测器在 LRS2-2Mix test 上评测（目标 14.1/14.7）
PYTHONPATH=$PWD $PYTHON_BIN audio_visual_test.py --conf_dir configs/LRS2_2Mix.yml \
  --data_dir /root/data/falcon_lrs2_2mix --ckpt_path <最佳.ckpt> --gpus 0
```

> 注意 1：训练数据用 `falcon_lrs2_2mix_paper`（双向 target，40k 条）；**评测**仍用 `falcon_lrs2_2mix/tt`
> （3000 条，target=s1 是本地约定；s1/s2 对照不能证明论文测试目标方向）。
> 注意 2：续训时 CSVLogger 会新建 `version_<N+1>`，monitor 按 `version_*` 数字选择，跨实验根目录不能保证是时间上最新；
> 若 checkpoint 根目录不在默认的 `/root/tmp/*/Experiments/checkpoint` 与 `./Experiments/checkpoint`
> 下，用 `--root <dir>`（可重复）追加搜索根。`--ckpt_path` 缺失会被
> `audio_train.resolve_resume_checkpoint()` 拦成 `SystemExit`（A3-04）——
> 这是为了防止“以为在续训、实际从零训练”的长时间误跑。

## 7. 时间和资源估算的边界

历史记录曾依据单卡 metrics 和双卡 smoke 外推 8 卡每 epoch 2–3 分钟、1000 epoch 33–50 小时；这是历史外推；目标节点的 8 卡耗时和早停时间需要实测。磁盘、数据加载、验证、通信、GPU 型号及调度都会改变耗时。规划时先在获授权的目标节点做有界实测，记录实际吞吐与验证耗时，再决定资源。

## 8. 风险与差异声明

1. **配方版本**：v2 参考论文 Table 16（−SI-SNR、τ/1000、bs4/GPU ⇒ global 32、seed 42）；LEGACY = 旧 8 卡方案
   （−SNR、τ/200、bs5/GPU ⇒ global 40）。`exp_name` 已区分（`LRS2_2Mix_Paper_v2` vs `LRS2_2Mix_Paper_8gpu`），
   **两套结果不可混用**，报告必须写明用的哪份配置。
2. **硬件差异**：论文使用 8×RTX3090；目标环境是否有 8 卡必须实际核对。不同 GPU、分布式和数值实现可能改变结果，不能预设数值等价。
3. **LRS2-2Mix 无 near-field 源** ⇒ 用另一说话人作 NASS 负样本（非论文原始 near 模拟）；K=1 时 softmax 权重恒为 1，
   **τ 退火在 LRS2 上实际不生效**（温度效应需至少两个有效负样本；RealSSA-Sim 发布分片未提供独立 near 波形）。
4. **target 约定**：训练双向（s1/s2），本地评测按 target=s1；论文未说明训练侧 target 采样方式。
5. **seed 42**：论文 §5.2 给出，但完整确定性设置（cuDNN、数据顺序、worker seed）未给；单 seed 有方差，
   正式结论建议 3 seed（42/43/44）并报告均值±std。
6. **数据混合**：论文 LRS2 配方引用 prior work [6,17]，**未给 Falcon 自己的 LRS2 数据混合脚本**；我们的 40k
   双向 target 是推断，须在结果中声明。
7. 若作者回复 issue 提供 LRS2 ckpt/配置，优先用官方 ckpt 评测，再用本方案对照。

### 2026-11-10 补充：实际训练前需要理解的代码行为

- 每 epoch `train_samples=10000` 是有放回随机抽取，40,000 个方向样本的单 epoch 期望覆盖约 22.12%；20,000 个混音至少被抽到一个方向的期望覆盖约 39.35%。不是顺序遍历完整训练集。
- K=1 时温度权重恒为 1；小样本梯度实验与源码均表明该设置下 β 不改变结果。K=0 时仍有正项，关闭 near 不等于关闭 NASS。
- 当前 `video_lstm_batch_first=False` 在 batch 大于 1 时存在跨样本交互；改成 True 会改变计算及权重行为，不能作为无影响的推理开关。`paper_causal=True` 存在 stride 2 过裁和 kernel 1 空输出风险，不能仅凭开关认定实现了论文时延。
- DDP 下条件性 `all_gather` 有不同步风险，训练后多 rank 导出可能竞态；这是源码层面风险，不能当作已发生或已通过真实多卡验证。请先检查[数据与训练详解](understanding/20261110-data-and-training.md)。
