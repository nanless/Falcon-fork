# LRS2-2Mix 论文配方训练（8 卡 / 火山引擎）Runbook

> 目标：复现论文 LRS2-2Mix 列 **SI-SNRi 14.1 / SDRi 14.7**。
> 背景：官方发布权重是 LRS3 训练的（`LRS3_complex_nonmoving`，epoch 714），在 LRS2-2Mix 上 7.03/7.71 属跨语料；
> 论文表格 “under our training protocol” 按 benchmark 分别训练，需要 LRS2 专属 ckpt（已向作者提 issue：
> <https://github.com/JusperLee/Falcon/issues/1>）。本文档给出**我们自己按论文配方重训**的完整方案。

## 1. 配方（与论文公开信息对齐）

| 项 | 值 | 说明 |
|---|---|---|
| 并行 | **8×A800（80GB）**，DDP | 论文 10 卡；8 卡用 bs5 对齐全局 batch |
| batch | **5/GPU × 8 = 40** | 论文 10×4=40 |
| lr / wd | AdamW **1e-3** / 0.01 | config 原值 |
| 调度 | ReduceLROnPlateau(patience 15, factor 0.5) | config 原值 |
| 梯度裁剪 | 5.0 | config 原值 |
| epochs | **1000**，EarlyStopping(monitor=val_loss, patience=30) | 论文训练协议 |
| NASS β | 0.0 → 1.0 over **200 epochs** | config 原值；负样本见下 |
| train_samples | 10,000 /epoch | config 原值 |
| 数据 | LRS2-2Mix `tr`（20,000 条）+ 双向 target | 见 §2 |
| 视觉骨干 | 冻结，`tools/repro/extract_video_backbone.py` 从 `falcon.pt` 导出 | 11,186,688 参数 |
| 种子 | 未固定（论文未给）；复现实验建议固定并多 seed | |

**双向 target（关键改进）**：LRS2-2Mix 中 s1 恒为较响说话人（snr1>0），若只训练 target=s1，模型可走“选更响”捷径。
论文大概率在 s1/s2 间随机（mouth 对应切换）。我们的数据构建对每条 mixture 生成两条样本：
`(target=s1, mouth=s1, near=s2)` 与 `(target=s2, mouth=s2, near=s1)`，训练样本 40,000 条。

**near 负样本**：该数据集没有 near-field 泄漏源；用**另一说话人（干扰源）**作为 NASS 对比损失的负样本，
这是可用的近似（论文若用 near-field 模拟，指标可能更高）。报告中必须声明该差异。

## 2. 数据准备（在火山引擎节点上执行）

LRS2-2Mix 是公开数据（HF `JusperLee/LRS2-2Mix`，`lrs2.tar.gz` 16.47 GB）：

```bash
# 1) 下载并解压（16.5GB → 17.4GB 成员）
#    有镜像用镜像；没有则 HF 直连/proxy
curl -L -o lrs2.tar.gz "https://huggingface.co/datasets/JusperLee/LRS2-2Mix/resolve/main/lrs2.tar.gz"
mkdir -p /data/lrs2_2mix && tar xzf lrs2.tar.gz -C /data/lrs2_2mix   # 得到 lrs2_rebuild/

# 2) 构建“论文配方”数据（双向 target + near，输出 ~26GB mouths + JSON；音频直接引用原始 wav）
git clone https://github.com/nanless/Falcon-fork   # 或你自己的 fork
cd Falcon-fork
/root/miniforge3/envs/falcon/bin/python tools/repro/phase2/build_lrs2_paper_data.py \
  --src /data/lrs2_2mix/lrs2_rebuild --out /data/falcon_lrs2_2mix_paper --jobs 32
# 期望：tr 40000 / cv 10000 / tt 6000 条；mouths 18,691 / 476 / 267 个 .pt；共 ~26GB
```

> 若要自包含数据（音频也复制进输出目录，约 +7GB）：加 `--copy-wavs`。
> 我们的本地构建：`/root/data/falcon_lrs2_2mix_paper`（26GB，JSON 引用 `/root/data/lrs2_2mix/lrs2_rebuild` 的音频）。

## 3. 环境

与仓库 `AGENTS.md` §5 一致：Python 3.11、torch 2.8.0+cu128、pytorch-lightning 2.6.6、typeguard<3、timm、fast-bss-eval、
soundfile、pypesq（pesq shim）等。火山引擎镜像若已有 CUDA/PyTorch，按 lock 文件安装即可：

```bash
pip install -r docs/verification/conda-env-lock.txt   # 先装好匹配 CUDA 的 torch
# 自检
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # 期望 8
```

## 4. 视觉骨干预训练权重

```bash
python tools/repro/extract_video_backbone.py \
  --ckpt /root/.cache/falcon_weights/falcon.pt \
  --out  /root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar
```
（`falcon.pt` 从 HF `JusperLee/Falcon` 下载；config 里 `videonet.videonet_config.pretrain` 指向该文件。）

## 5. 启动训练（8 卡）

```bash
cd /path/to/Falcon-fork
CONF=configs/LRS2_2Mix_Paper_8gpu.yml \
DATA_ROOT=/data/falcon_lrs2_2mix_paper \
bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
```

脚本会做自检（8 GPU、数据 JSON、骨干文件）→ 后台启动 `audio_train.py` → 日志写入
`docs/verification/phase2/LRS2-paper-<UTC>.log`，PID 写 `LRS2-paper.pid`。

如果平台要求单进程入口，也可直接：

```bash
python audio_train.py --conf_dir configs/LRS2_2Mix_Paper_8gpu.yml
```

（PL 2.x 在 `gpus: -1` + 8 可见卡时自动 DDP。）

## 6. 监控 / 续训 / 导出

```bash
# 监控（val SI-SNR、最佳 epoch）
python tools/repro/phase2/monitor_lrs2_paper.py --exp LRS2_2Mix_Paper_8gpu --watch 300

# 续训（中断后从 last.ckpt 恢复）
python audio_train.py --conf_dir configs/LRS2_2Mix_Paper_8gpu.yml \
  --ckpt_path <exp_dir>/last.ckpt

# 导出 ckpt 后用 Phase-1 修好的评测器在 LRS2-2Mix test 上评测（目标 14.1/14.7）
PYTHONPATH=$PWD python audio_visual_test.py --conf_dir configs/LRS2_2Mix.yml \
  --data_dir /root/data/falcon_lrs2_2mix --ckpt_path <最佳.ckpt> --gpus 0
```

> 注意：训练数据用 `falcon_lrs2_2mix_paper`（双向 target，40k 条）；**评测**仍用 `falcon_lrs2_2mix/tt`（3000 条，target=s1，
> 与 benchmark 约定一致；目标约定已由 s1/s2 对照确证）。

## 7. 时间估算（8×A800）

- 8 卡 bs5：约 250 step/epoch@每卡；单卡实测 bs4 ≈5.5 it/s ⇒ 训练 ~1 min/epoch + 分布式验证 ~0.5–1 min/epoch
  ⇒ **约 2–3 min/epoch（含验证）**。
- 1000 epoch 上限 ≈ **33–50 h**；EarlyStopping(30) 通常 150–300 epoch 收敛 ⇒ **约 5–15 h**。
- 本地 2 卡等价配置（bs4×2×accum5=40）约为 1/4 吞吐，可用于 1–2 epoch 的 smoke 验证（见 `run_lrs2_paper_2gpu_smoke.sh`）。

## 8. 风险与差异声明

1. LRS2-2Mix 无 near-field 源 ⇒ 用另一说话人作 NASS 负样本（非论文原始 near 模拟）。
2. target 约定：训练双向（s1/s2），评测按 benchmark 约定 target=s1；若论文训练用不同约定，需调整。
3. 论文未公开训练细节（数据混合、seed、warmup 等），本方案为“最接近的合理复现”。
4. 若作者回复 issue 提供 LRS2 ckpt/配置，优先用官方 ckpt 评测，再用本方案对照。
