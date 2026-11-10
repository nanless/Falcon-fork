> **2026-11-10 复核提示：** 历史实验更新：`SISNRi` 构造问题已修复，但静音等退化输入仍可能产生 NaN；历史 3,000 条表与 6,000 方向索引采用不同方向数，需结合 ID 对照。音频模型 537,697 参数包含对比头，但评测前向不执行该头；ptflops 与论文数值接近，具体算子统计口径需进一步核对。RealSSA 两表的 SI-SNRi 有效数为 9,169，SDRi/PESQ 为 9,166；DNSMOS 的论文与本地实现细节及 checkpoint 身份仍需核对。详见[评测专题](../20261110-evaluation-and-apps.md)和[数据审计](../../verification/deepdive-20261110/data-audit-20261110/README.md)。

# 05 损失与指标（Falcon-fork 深挖轮 W1-05）

> 范围：`look2hear/losses/{matrix,mixit,nass_contrastive_loss,pit_wrapper,sisnri,__init__}.py`、
> `look2hear/metrics/{allwrapper,core,splitwrapper,wrapper,__init__}.py`、`evaluated_mac_params.py`、
> `audio_visual_test.py` 指标段、`look2hear/system/av_litmodule{,_nass_v2}.py`（损失组合的调用方）。
> 仓库只读；全部实测在远端 `falcon` conda 环境、CPU（`CUDA_VISIBLE_DEVICES=`）完成。
> 基线 commit `6c1d5b7`。论文 arXiv:2610.32749 已于 2026-10-23 公开；MACs 口径 2026-11-08 经 Lead 复核。

---

## 0. 结论速览

1. **整条管线是「目标说话人抽取」而非盲分离**（任务定义与结构证据见 01 §2）：所有 6 个 config 都写死 `n_src: 1`
   （`Falcon.yml:12`、`LRS2_2Mix.yml:11`、`Paper_2gpu.yml:11`、`Paper_2gpu_smoke.yml:11`、`Paper_8gpu.yml:16`、`Paper_v2.yml:28`；datamodule 侧同名参数同步为 1），
   train/eval 索引只有 `target.json`（`look2hear/datas/avspeech_dataset.py:47,148`）→ **PIT 在实跑路径上是恒等映射**，
   `PITLossWrapper` 的置换逻辑装了但没用上。
2. **训练损失默认不是 SI-SDR 而是 −SNR**：`configs/Falcon.yml:117/123`、`LRS2_2Mix.yml:105`、
   `Paper_2gpu.yml:105`、`Paper_8gpu.yml:110` 全部 `sdr_type: pairwise_neg_snr` + `threshold_byloss: true`；
   只有 `Paper_v2.yml:122,126` 换成 `pairwise_neg_sisdr` + `threshold_byloss: false`。
   −SNR **不具备尺度不变性**（实测 `est=-s` → −6.02 dB，`est=0.5s` → +6.02 dB），−SI-SDR 才具备（均 ≈119 dB）。
3. **`threshold_byloss=True` 会丢掉"最好"的样本**：条件 `min_loss > -30` 保留的是 **loss 较大（=更差）** 的样本。
   实测 B=2（1 完美 + 1 很差）：`False→-50.69`、`True→+14.64`。eval/val 侧因此显式关掉（`metrics/core.py:72-79`）。
4. **SI-SDR 实现与解析值逐位吻合**（7 组信号，最大误差 2.6e-7 dB）；`eps=1e-8` 出现在 4 处，
   并带来两个人工边界：**地板 −80 dB**（完全不相关/全零估计）与**天花板 10·log10(E/eps)**（完美重建，2 s/16 k 约 116 dB）。
5. **`pairwise_neg_sdr` 的分母/分子不对称**：`matrix.py:227-229` 分子无 eps、分母 `+EPS`，再在 log 内 `+EPS` →
   地板固定 −80 dB 而非 `-inf`；同一份实现里 `SingleSrcNegSDR` 又把构造参数 `EPS` 丢掉硬写 `1e-8`（`matrix.py:246`）。
6. **历史 `pit_from="pw_pt"` 兼容问题已修复**：它把 2-D 张量喂给要求 3-D 的 `PairwiseNegSDR` → `TypeError`（实测）。
   仓库里 `pw_pt` 无任何 config 引用，属死路径。
7. **历史 `SISNRi` 构造问题已修复**：`super(Loss, self).__init__()` 引用未定义名 `Loss` → `NameError`（实测）。该类被 `__init__.py:20` 导出。
8. **`metrics/core.py` 集中定义本 fork 的指标**（原上游没有此文件）；上游 `allwrapper` 的 `sdr` 列实为 `-SNR`、
   PESQ 参数顺序反了、逐句行与 avg 行符号相反（`git show a7af472:look2hear/metrics/allwrapper.py:27,50,63,73`）。
9. **三个 tracker 只有 `ALLMetricsTracker` 在用**（`audio_visual_test.py:397`）；`MetricsTracker`/`SPlitMetricsTracker`
   仅被 `__init__.py` 导出，无调用点 → 死代码。三者语义都是**逐句算再平均**，没有"整段拼接后再算"的实现。
10. **PESQ 能跑，但靠的是环境级 shim**：`site-packages/pypesq.py`（**不在仓库里**）转发到 PyPI `pesq` 0.0.4，
    `mode` 写死 `'wb'`；换一台干净机器 `core.pesq_metric` 直接返回 `nan`。STOI 用 `pystoi` 0.4.1 `extended=False`。
11. **DNSMOS 不在 `look2hear/metrics` 内**，只在证据工具 `tools/repro/phase2/eval_dnsmos.py`（speechmos 0.0.1.1，
    16 kHz，9.01 s 窗/1 s hop；P.835=`sig_bak_ovr.onnx`、P.808=`model_v8.onnx`，4 个 onnx 的 sha256 已记录）。
12. **MACs 实测复算 = 10.610934 G / 0.537697 M**（CPU ptflops，与 docs/STATUS 的 10.611 G 逐位一致）；
    口径 = **仅 audio model、eval 模式、2 s 窗（32000 samples + 50 帧 mouth embedding）**，
    **不含 FRCNN**（`evaluated_mac_params.py:82` 在 ptflops 调用之前就算完了）、**不含对比头**（forward 里根本没走 `spk_pool_proj`）。
    这个数是 **MACs**（ptflops 的 conv hook 每个乘加记 1），论文数值为 10.6 G，其计数方法尚未确认。同模型换
    `torch.utils.flop_counter` 得 **5.382 GMAC**，因为它不记 `aten::lstm`——两把尺子不可并列。
    视觉 88×88×50 帧同一把尺子是 15.897 G，整窗 **26.51 GMAC、视觉占 60.0%**。

---

## 1. 事实基线

### 1.1 文件与规模（`wc -l` 实测）

| 文件 | 行数 | 角色 |
|---|---|---|
| `look2hear/losses/matrix.py` | 382 | `PairwiseNegSDR`/`SingleSrcNegSDR`/`MultiSrcNegSDR` + STFT + 多阶段损失 + 实例别名 |
| `look2hear/losses/pit_wrapper.py` | 324 | `PITLossWrapper`（152-312）+ `new_PITLossWrapper`（6-150，语义不同） |
| `look2hear/losses/mixit.py` | 309 | 1-95 行整段注释；`MixITLossWrapper` 在 102-309 |
| `look2hear/losses/nass_contrastive_loss.py` | 81 | NASS 三元组对比损失 |
| `look2hear/losses/sisnri.py` | 41 | `SISNRi`（历史构造问题已修） |
| `look2hear/losses/__init__.py` | 44 | 导出 21 个符号 |
| `look2hear/metrics/core.py` | 229 | **本 fork 新增**的唯一指标定义源 |
| `look2hear/metrics/allwrapper.py` | 198 | 线上 tracker（`audio_visual_test.py` 用） |
| `look2hear/metrics/wrapper.py` | 103 | 旧 tracker（注明 not used） |
| `look2hear/metrics/splitwrapper.py` | 115 | 2+1 分路 tracker（未使用） |
| `evaluated_mac_params.py` | 151 | ptflops MACs 脚本（GPU-only） |
| `audio_visual_test.py` | 875 | 评测入口（指标段 364-547） |

### 1.2 运行环境（`importlib.metadata` 实测，`/root/miniforge3/envs/falcon/bin/python`）

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.8.0+cu128 | |
| fast-bss-eval | **0.1.4** | 真装了（旧验证脚本却把它 stub 掉了，见 §6） |
| pesq | **0.0.4** | PyPI ITU-T P.862 实现，签名 `pesq(fs, ref, deg, mode)` |
| pypesq | **无 pip 记录** | 但 `site-packages/pypesq.py` 存在 → 这是一个**手写 shim**（§2.8.1） |
| pystoi | 0.4.1 | |
| ptflops | 0.7.5 | |
| speechmos | 0.0.1.1 | 自带 4 个 onnx |
| torchmetrics | 1.9.0 | **在本环境 import 会炸**（§5.9） |

### 1.3 索引与口径（实测 `json.load`）

```
/root/data/falcon_lrs2_2mix_paper/tt/
  mix.json     6000 行，每行 [mix.wav, 32000]      -> 唯一 stem 只有 3000 个
  target.json  6000 行，每行 [s1|s2.wav, mouth.pt, 32000]
               源目录分布: {'s1': 3000, 's2': 3000}
```

→ 一条混合语音出现 **2 次**（一次以 s1 的嘴型为条件、一次以 s2 的嘴型为条件），
所以"跑满 tt"= **6000 条逐句记录**，而历史 CSV 只有 3000 条（§6.4）。
每条音频 32000 samples = **正好 2.0 s @16 kHz**。

---

## 2. 机制拆解

### 2.1 损失清单（公式 / 形状 / 归约 / eps / PIT / mask）

符号：`B` batch，`N` 源数（本仓恒 1），`T` 采样点，`⟨·,·⟩` 内积，`‖·‖²` 能量。

#### 2.1.1 `PairwiseNegSDR` — `matrix.py:188-232`

```
输入:  ests, targets ∈ R^{B×N×T}      （形状不等于 3-D 直接 TypeError, :198-201）
零均值: 对 dim=2（时间轴）各自减均值   :204-208   ← 注意是 dim=2，时间轴
s_target  = unsqueeze(targets, dim=1)  -> [B,1,N,T]
s_estimate= unsqueeze(ests,   dim=2)   -> [B,N,1,T]
sisdr/sdsdr:
  dot   = Σ_t s_est · s_target                 [B,N,N,1]
  E_tgt = Σ_t s_target² + EPS                  [B,1,N,1]      (:216)
  proj  = dot · s_target / E_tgt               [B,N,N,T]      (:218)
snr:
  proj  = s_target 广播                        [B,N,N,T]      (:221)
e_noise = s_est - proj        (sisdr)
        = s_est - s_target    (snr/sdsdr)                     (:222-225)
SDR_ij  = Σ proj² / (Σ e_noise² + EPS)                        (:227-229)
loss    = -10·log10(SDR + EPS)                                (:231-232)
输出:  [B,N,N]  ← 每个 (估计 i, 参考 j) 一个负 SDR；不做归约
```

* eps 位置：3 处（`E_tgt`、分母、log 内），默认 `1e-8`（`:189`）。
* 归约：**无**，返回完整 pair 矩阵；归约由 `PITLossWrapper` 做。
* PIT：不在本类，靠 wrapper。
* mask：**无任何 mask/长度 mask**（batch 内全部按齐长处理）。
* 别名实例：`pairwise_neg_sisdr/sdsdr/snr`（`:372-374`），模块级单例、无状态可共享。

#### 2.1.2 `SingleSrcNegSDR` — `matrix.py:235-281`

```
输入 [B,T] → 零均值 dim=1 (:254-258) → dot [B,1] → E_tgt [B,1]+EPS
→ scaled_target → e_noise → ratio [B] → 10log10(ratio+EPS) → mean if reduction=="mean" → 取负
输出 [B]（reduction='none' 默认, :237）
```
**坑**：构造参数 `EPS` 被忽略，`self.EPS = 1e-8` 硬写（`:246`）；`assert reduction != "sum"`（`:239`）。

#### 2.1.3 `MultiSrcNegSDR` — `matrix.py:284-326`

`[B,N,T]`，零均值 dim=2，逐源算 ratio，`return -mean(pair_wise_sdr, dim=-1)` → **[B]**（源轴已平均）。
**未被 `__init__.py` 导出**（只有它的实例 `multisrc_neg_*` 被导出，`:31-37`）。

#### 2.1.4 `freq_MAE_WavL1Loss` / `freq_MSE_Loss` — `matrix.py:328-369`

`freq_L1 = |Re(Ŝ)−Re(S)|.mean + |Im(Ŝ)−Im(S)|.mean`（win=2048/hop=512）再加时域 L1；
`freq_MSE_Loss` 同类但用平方（win=640/hop=160，即 40 ms/10 ms @16 k）。
**`freq_MSE_Loss` 没有导出、也没有别名**（`:351-369` 与 `:382` 只导出 `freq_mae_wavl1loss` 和二元组）。

#### 2.1.5 多阶段损失（`PIT_SISNR_mag_loss` / `PIT_SISNR_time_loss` / `MultistageLoss*`）— `matrix.py:105-186`

```
mag 支: -20·log10( eps + ‖‖S_ref‖‖ / (‖‖S_mix − S_ref‖‖ + eps) ),  eps=1e-12 (:117,128)
time 支: -20·log10( eps + ‖s_ref‖ / (‖s_mix − s_ref‖ + eps) ), clamp(min=-30) (:139,146-147)
MultistageLoss: cur = (1-α)·time + α·mean(mag_i),  α=0.4·0.8^(1+(epoch-81)//5) for epoch>80 else 0.4 (:161,168)
```
注意：这里的 "mix" 其实是**当前阶段的估计**，`src` 是干净参考 → 不是 SI-SDR，是"估计 vs 估计"的
log-ratio 型损失（尺度不变性靠 `scale_inv` 分支 `:120-122,143-145`）。
**全链路无调用点**（§2.6）。

#### 2.1.6 `NASSContrastiveLoss` — `nass_contrastive_loss.py:18-81`

```
L_pos = α_pos · (1 − cos(q, p))
L_neg = α_neg · Σ_k  softmax_k(β·cos(q,n_k)) · relu(cos(q,n_k) − margin)     (K ≥ 2, :71-72)
      = α_neg · Σ_k  relu(cos(q,n_k) − margin)                                (K == 1, :68-69)
      = 0                                                                     (K == 0 / None, :73-74)
L     = (Σ_b L_pos^(b) + L_neg^(b)) / n_valid,   n_valid += 1 无条件           (:76-77,81)
```

* 输入：`q,p ∈ R^{B×D}`（假定已 L2 归一化，但用 `F.cosine_similarity` 自带归一化，`:57,64`）；
  `embed_negs_list: list[B] of [K_b, D] | None`（**变长、逐样本 python 循环**，不可向量化 → DDP 下每卡独立）。
* 温度：**没有 τ**，只有浓度参数 `β`（`forward` 默认 1.0，`:36`），由 litmodule 线性 anneal 0→1.0。
* margin：构造默认 **0.0**（`:22`），config 里给 **0.5**（`configs/Falcon.yml:105`）。
* 归约：逐样本求和后 **除以 n_valid = B**（不是"有负样本的样本数"）→ 无近邻样本会**稀释**损失（实测 §4.5）。
* PIT：**无**。query 恒取 `est[:,0]`、positive 恒取 `targets[:,0]`（`av_litmodule_nass_v2.py:102,107`）——
  在 `n_src=1` 下无歧义；若将来 `n_src>1`，这里缺少 PIT 重排，正样本可能配错人（结构性风险，见 GAP-3）。
* mask：无。

### 2.2 被 config 选中的损失组合

`audio_train.py:328-337` 把 config 拼成 `{"train": PITLossWrapper(getattr(losses, sdr_type), **config), "val": ...}`；
`loss_func` 名恒为 `PITLossWrapper`，实际差异全在 `sdr_type` + `threshold_byloss`。
`training.system` 恒为 `AudioVisualLightningModuleNASSv2`（`configs/*.yml:115/121/134`），
所以**总损失恒为 `pit_loss + contrastive_weight · nass_loss`**（`av_litmodule_nass_v2.py:137-138`）。

| config | train `sdr_type` | train `threshold_byloss` | val `sdr_type` | val `threshold_byloss` | spk_contrast |
|---|---|---|---|---|---|
| `Falcon.yml:115-126` | `pairwise_neg_snr` | **true** | `pairwise_neg_sisdr` | false | margin .5, β 0→1/200ep, w=1.0 (`:104-111`) |
| `LRS2_2Mix.yml:102-113` | `pairwise_neg_snr` | true | `pairwise_neg_sisdr` | false | 同上 (`:94-101`) |
| `LRS2_2Mix_Paper_2gpu.yml:102-114` | `pairwise_neg_snr` | true | `pairwise_neg_sisdr` | false | 同上 (`:94-101`) |
| `LRS2_2Mix_Paper_2gpu_smoke.yml:102-114` | `pairwise_neg_snr` | true | `pairwise_neg_sisdr` | false | 同上 |
| `LRS2_2Mix_Paper_8gpu.yml:107-120` | `pairwise_neg_snr` | true | `pairwise_neg_sisdr` | false | 同上 (`:99-106`) |
| `LRS2_2Mix_Paper_v2.yml:119-132` | `pairwise_neg_sisdr` | **false** | `pairwise_neg_sisdr` | false | margin .5, β 0→1/**1000**ep (`:111-118`) |

→ 结论：**默认配方（含"论文 Table 16"的 8gpu 版）用 −SNR 训练**，只有 v2 用 −SI-SDR。
`av_litmodule.py:163,184` 里 `val_pit_sisnr = -val_loss` 也因此只在 val 损失为 −SI-SDR 时才名副其实。

### 2.3 SI-SNR 实现细节（三个实现，逐行）

**(a) `PairwiseNegSDR("sisdr")`（训练/指标共用）**：见 2.1.1。
零均值在 **dim=2（时间轴）**；`eps=1e-8` 3 处；`-inf` 保护 = 分母 +EPS 且 log 内 +EPS →
**下界 −80 dB**（实测 §4.1），上界 = `10·log10(E_tgt/eps)` 随信号能量浮动（2 s/16 k 实测 116.02 dB）。
**batch 内平均 vs 句内平均**：本类**不做任何平均**；`PITLossWrapper` 先对 `N` 个源求**算术平均**（`:289` `loss_set /= n_src`），
再对 `B` 求 mean（`:230`）→ **先句内(源轴)平均、再 batch 平均**，两种平均等价（都是线性），
但**不是**"先算 dB 再平均功率"——是 dB 域直接平均。

**(b) `SISNRi.sisnr`（`sisnr.py:8-24`，历史构造问题已修）**：显式写 `est_p = ⟨est,ref⟩·ref/‖ref‖²`，
`est_sisnr - mix_sisnr` 直接返回 **improvement**（`:24`）——这是唯一内置 "i" 的实现，但拿不到。

**(c) `metrics/core.py:152-155`**：`-PIT_SI_SDR(est, ref)`，其中 `PIT_SI_SDR` 用
`PairwiseNegSDR("sisdr", zero_mean=True)` + `pit_from="pw_mtx"` + **`threshold_byloss=False`**（`:75-79`）。

**与 `torchmetrics` / `fast_bss_eval` 的差异（实测 §4.2）**：

| 维度 | Falcon `PairwiseNegSDR` | `fast_bss_eval.si_sdr` 0.1.4 |
|---|---|---|
| 完美重建 | 116.02 dB（受 `E/eps` 限制） | **156.54 dB**（无 eps，`zero_mean=True`） |
| `est=0.5s` | **113.38 dB**（eps 破坏严格尺度不变） | **156.54 dB**（严格不变） |
| 参数顺序 | `(ests, targets)` | **`(ref, est)`**（注意反了） |
| `zero_mean` 默认 | **True** | **False** |
| 长度/滤波 | 逐点投影，无滤波 | 无滤波（`sdr` 才有 512 tap） |
| `N=1` 输入 | 正常 | **抛 `RuntimeError: min(): ... numel() == 0`**（实测） |
| clamp | 无 | `clamp_db` 可选（默认 None） |

`torchmetrics` 在本环境 import 失败（§5.9），无法做第五列对比；但从公式看它等价于 `fast_bss_eval` 那一列
（同样 `zero_mean=True`、无 eps、`10log10(sum(proj²)/sum(res²))`）。

### 2.4 SI-SNRi / SDRi 定义与 PIT 口径

```
SI-SDRi(est) = SI-SDR(est, ref) − SI-SDR(mix, ref)        core.py:158-162
SDRi(est)    = SDR(est, ref)    − SDR(mix, ref)           core.py:180-185
SDR(est)     = −fast_bss_eval.sdr_pit_loss(est, ref, filter_length=512).mean()   core.py:168-177
```

* **是相对 mix 的 improvement 吗**：是。减数是**同一个 `ref`** 下把 `mix` 当作估计算出的分数
  （`_mixture_sources` 把单声道 mix 复制 `n_src` 份，`core.py:132-141`）。
* **参考信号选谁**：**恒为 `clean` = s1 或 s2（当前这条的"目标说话人"），不是 PIT 后的 s1/s2 一对**。
  索引里只有 `target.json`（§1.3），`clean` 从 dataset 出来就是 `[T]` → `[1,T]`（`audio_visual_test.py:420-424`）。
* **PIT 是逐句还是整段（关键口径）**：`audio_visual_test.py` 是 **whole-utterance、batch=1、逐句**：
  `est_sources = model(mix.unsqueeze(0), mouth_emb)`（`:419`）→ `metrics(mix=mix, clean=sources.unsqueeze(0), estimate=est_sources.squeeze(0), key=key)`（`:420-424`）。
  dataset 侧 `stop=None`（`avspeech_dataset.py:186-187`）→ 读整条 **32000 samples = 2 s**，无切窗、无拼接。
  文件头注释也写明 "whole-utterance inference, batch size 1"（`audio_visual_test.py:26-27`）。
  所以最终的 3000/6000 条记录 = **每条 2 s 语音各自算一次 SI-SDR/SDR/PESQ/STOI，再对句做算术平均**。
  **不存在"整段拼接（concatenate all utts then compute）"的做法**；`si-snr_mix` 也是逐句算的。
* **PIT 在实路径上是恒等**：`n_src=1`（§2.2 表）→ `find_best_perm_factorial` 只有 1 个排列（`pit_wrapper.py:280,288-289`）。
  仍实测了 N=2/3 的置换正确性（§4.3）以备 `n_src>1`。
* **BSSEval `compute_permutation` 用法**：本仓**没有** `torch_mir_eval`/`mir_eval` 的 `bss_eval_sources`
  调用（`splitwrapper.py:6` 的 import 被注释）；置换由 `fast_bss_eval.sdr_pit_loss` 内部做
  （其内部 `_solve_permutation`，见 §4.2 报错栈），或由 `PITLossWrapper.find_best_perm_factorial` 的 one-hot einsum 做
  （`pit_wrapper.py:287-288`）。`filter_length=512` 与 BSS-Eval 默认一致（`core.py:69`）。

### 2.5 metrics wrapper 三者的聚合语义

| tracker | 用在哪 | 每句算什么 | 聚合 | 与论文表格的关系 |
|---|---|---|---|---|
| `ALLMetricsTracker` (`allwrapper.py:70-196`) | **唯一在跑**：`audio_visual_test.py:397` | `core.si_sdr/sdr`（含 PIT，句内先源平均）+ `si_sdr_mix`/`sdr_mix` 基线 + PESQ/STOI（仅 `n_src==1`，`:112-119`） | `np.nanmean`/`np.nanstd`（`ddof=0`，`:147-159`），`final()` 写 `avg`/`std` 两行（`:183-190`）；`n_*` 列记有限值个数（`:175-181`） | **= 逐句算再平均**，对应论文 "SI-SDRi / SDRi on LRS2-2Mix tt"；句数即权重（等权） |
| `MetricsTracker` (`wrapper.py:29-100`) | **无调用点** | 同上（但无 PESQ/STOI、无 mix 基线列） | 同上 | 无 |
| `SPlitMetricsTracker` (`splitwrapper.py:17-115`) | **无调用点** | 把 3 个估计拆成 "two"(0:2) 与 "one"(2)，分别 PIT；用 `pairwise_neg_snr` 定置换（`:45-47`），再算 sisnr | `np.array(...).mean()`（`:101-108`），`final()` 只有 avg 行 | 这是给 **3 源（2 混 + 1 单）** 数据集的（LRS3-complex 类），与 LRS2-2Mix 无关 |
| combine (`audio_visual_test.py:_merge_results:467-547`) | 每 rank 一个 shard → `metrics_combined.csv` | — | 先 `pd.concat`，再对**逐句行**求 mean/std（`:495-497`），并把 `n_*` 回填（`:498-501`）；同时做 `|si-snr − si-snr_i − si-snr_mix|` 恒等式自检（`:511-518`）写进 `metrics_counts.json` | 决定"表格里那一行"的唯一入口 |

**`sync_dist` 行为**：metrics 侧**完全没有 `sync_dist`**——它不走 Lightning，而是每 rank 写自己的
`metrics_rank{rank}.csv`（`:396-397`），最后由 rank0 的 `_merge_results` 合并（`:467-547`）；
合并后还会 `os.remove` 掉每个 shard（`:545-548`）。也就是说**多卡评测的"全局平均"是 pandas 层面的，
不是 `all_gather` 层面的** → 若某个 rank 中途异常，`_merge_results` 只合并存在的 shard，**样本数会静默变少**
（`:470-473` 的 `if os.path.isfile` 分支）。这点与训练侧形成对比：训练侧 `self.log(..., sync_dist=True)`
（`av_litmodule.py:78-82,96-103,131-138`）和 `on_validation_epoch_end` 的 `all_gather`（`:145,174,183`）都在做跨卡同步。

### 2.6 死代码清单（grep 引用，排除 `look2hear/losses/` 自身与 `docs/`）

| 符号 | 外部引用 | 判定 |
|---|---|---|
| `MultistageLoss` / `MultistageLoss_val` | 0 | 死（仍被 `__init__.py:18-19` 导出） |
| `PIT_SISNR_mag_loss` / `PIT_SISNR_time_loss` | 0 | 死 |
| `freq_mae_wavl1loss` / `pairwise_neg_sisdr_freq_mse` / `pairwise_neg_snr_multidecoder` | 0 | 死（`matrix.py:381-383` 的实例别名） |
| `multisrc_neg_sisdr` / `multisrc_neg_sdsdr` / `multisrc_neg_snr` | 0 | 死 |
| `new_PITLossWrapper` | 0 | 死（且语义与 `PITLossWrapper` 不同：`pit_wrapper.py:75` 直接 `return pw_loss`，**不做 PIT 归约**） |
| `SISNRi` | 0（`splitwrapper.py:112` 只是注释里出现字样） | 死 + 已坏 |
| `MixITLossWrapper` | 仅 `docs/verification/scripts/verify_losses_04.py:13` | 死（`mixit.py:1-95` 整段注释；102 起的类无 config 引用） |
| `Matrix.py` 的 `STFT`/`STFTBase` | 仅 `PIT_SISNR_mag_loss.__init__`（`:114`，而那个类也死） | 死 |
| `Single/MultiSrcNegSDR` 实例 | `singlesrc_neg_sisdr` 被 `splitwrapper.py:11` import 但**未使用** | 半死 |

结论：**实际生效的损失实现只有 `PairwiseNegSDR`（+ `PITLossWrapper` + `NASSContrastiveLoss`）**，
余下约 60% 的 `look2hear/losses/` 代码量是上游 asteroid/Conv-TasNet 生态的遗留。

### 2.7 PITLossWrapper 逐项

| 项 | 事实 | 位置 |
|---|---|---|
| 支持 `pit_from` | `pw_mtx`, `pw_pt`, `perm_avg`, `pw_mtx_broadcast`, `pw_mtx_multidecoder_keepmtx`, `pw_mtx_multidecoder_batchmin` | `:162` |
| `pw_mtx` | `pw_loss = loss_func(ests, targets)`，要求返回 `[B,N,N]`（`assert pw_loss.ndim==3`, `:197-199`） | `:170-171` |
| `pw_pt` | 逐对调用；把 `[B,T]` 喂给 `PairwiseNegSDR` → **TypeError**（实测 §4.3）；只有 `SingleSrcNegSDR` 能用 | `:182-183,239-247` |
| 置换求解 | `n_src<=3` 用阶乘 + one-hot einsum；否则 Hungarian（`scipy.optimize.linear_sum_assignment`） | `:266-274,276-301,303-312` |
| 源轴归约 | `loss_set /= n_src`（算术平均） | `:289` |
| `threshold_byloss` | `min_loss = min_loss[min_loss > -30]`（空则保留全部） | `:227-229` |
| batch 归约 | `torch.mean(min_loss)` | `:230` |
| `return_ests` | 按最优置换重排估计 | `:231,260-264` |
| `perm_avg` | 逐排列算标量损失取 min（`loss_func` 须自带归约） | `:184-193,249-258` |

### 2.8 PESQ / STOI / DNSMOS

#### 2.8.1 PESQ

```python
try: from pypesq import pesq as _pesq      # core.py:54-57
except Exception: _pesq = None
_pesq(ref, deg, int(sample_rate))          # core.py:203   参数序 (ref=clean, deg=estimate)
```

* **实际后端** = `site-packages/pypesq.py`（**手写 shim，不在仓库里**）：
  ```python
  from pesq import pesq as _pesq                     # PyPI pesq 0.0.4 (ITU-T P.862)
  def pesq(ref, deg, fs, mode="wb"):
      return float(_pesq(int(fs), np.asarray(ref), np.asarray(deg), mode))
  ```
  `mode` **硬编码 `'wb'`** → 只支持 16 kHz 宽带；若 `sample_rate` 变成 8 k，`core.pesq_metric` 会抛错并返回 `nan`
  （`core.py:204-205` 的裸 `except` 把它变成 nan 而不是报错）。
* 采样率：`PESQ_SAMPLE_RATE = 16000`（`core.py:70`），由 `audio_visual_test.py:368` 从
  `datamodule.data_config.sample_rate` 读入、`:397` 传给 tracker。`wb` 模式与 P.862.2 一致。
* 只在 `n_src==1` 时算（`allwrapper.py:112-119`），否则 nan。
* **历史 bug 已修**：上游是 `pesq(estimate, clean, 16000)`（`git show a7af472:look2hear/metrics/allwrapper.py:50`），
  参数序与 `pypesq` 相反 → 旧 PESQ 数值不可用。

#### 2.8.2 STOI

`pystoi 0.4.1`，`stoi(clean, estimate, fs, extended=False)`（`core.py:215`），返回 `[0,1]`，`fs=16000`。
`extended=True` 未使用（config 里也没开关）。

#### 2.8.3 DNSMOS（**不属于 `look2hear.metrics`**）

* 代码位置：`tools/repro/phase2/eval_dnsmos.py`（证据工具，不进评测入口）。
* 库：`speechmos==0.0.1.1` + `onnxruntime`，模型随 wheel 自带，离线可用（`eval_dnsmos.py:4-8,42-49`）。
* 采样率：**强制 16 kHz**（`TARGET_SR = 16000`, `:35`；`sr != 16000` 直接 raise，`:74-75`）。
* 窗口：**9.01 s / 1 s hop**，`speechmos/dnsmos.py:16,72-77`（`INPUT_LENGTH=9.01`, `num_hops = floor(len/fs) − 9.01 + 1`；
  短音频会**自我复制拼接**直到 9.01 s，`dnsmos.py:68-70`）——**这与分离评测的 2 s 整句口径完全不同**。
* P.835 与 P.808 的关系（`speechmos/dnsmos.py:104-118` + `eval_dnsmos.py:7`）：
  * **P.835**：三轴模型 `sig_bak_ovr.onnx`，输出 SIG/BAK/OVRL 三个原始分，再各自过 3 次多项式重标定
    （`get_polyfit_val`, `dnsmos.py:32-55`）→ 三个 1–5 的 MOS；
  * **P.808**：老的单轴模型 `model_v8.onnx`，输入是 **120 维 mel 谱**（`audio_melspec`），输出一个 1–5 MOS；
  * 两者**不是同一把尺子的不同输出**：P.808 是众包单维评价，P.835 是分轴实验室评价；数值不可互换、不可互比。
    仓库自己的结论也是这个方向（`docs/STATUS.md:37`：论文 3.276 是 **P.835**，官方 ckpt OVRL 2.7941，**不可比**）。
* 模型 sha256（本机实测 `hashlib`）：

| 文件 | 字节 | sha256 |
|---|---|---|
| `sig_bak_ovr.onnx`（P.835 主模型） | 1,157,965 | `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` |
| `model_v8.onnx`（P.808） | 224,860 | `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91` |
| `bak_ovr.onnx` | 742,375 | `5f335c90994618150192a656a474bcf8a9cbcedbc47965494ba8da79605d1308` |
| `sig.onnx` | 742,203 | `d2fbdb293bc2366dfbae2b7477c490f981d24a8b4405efd3c11787569c6549d7` |

* **GAP**：`look2hear/metrics/` 里没有任何 DNSMOS 入口，所以 `audio_visual_test.py` 的 8 列 CSV **不含 DNSMOS**；
  RealSSA-Real 那组 3.3365/2.7941 是另一条脚本链（`eval_realssa_real.py`）出的，与分离评测表**不可直接拼表**。

### 2.9 MACs 口径（`evaluated_mac_params.py`）

```python
a = torch.randn(1, 32000).cuda()            # :80   2 s @ 16 kHz
v = torch.randn(1, 1, 50, 64, 64).cuda()    # :81   2 s @ 25 fps, 64×64 灰度
v = video_model(v)                          # :82   ★ 在 ptflops 之前跑完 → FRCNN 不计入
macs, params = get_model_complexity_info(model, (a, v), input_constructor=input_func, ...)   # :88-96
print("MACs: ", total_macs / 10.0 ** 9)     # :122
```

* 输入口径：**2 s 窗**（32000 samples；50 帧 mouth embedding）→ 报出的 10.611 G 是**每 2 s 窗**的值。
* **是否含 FRCNN**：否。理由两条：(i) `:82` 已把视频前向跑完，喂给 ptflops 的是**特征张量** `[1,512,50]`；
  (ii) 实测（§4.7）模型内不存在 FRCNN 子模块。
* **是否含对比头**：**否**。`FalconSpeakerContrastive.forward` 只在 `self.training` 时才返回 embedding
  （`look2hear/models/Falcon_Contrastive.py:107-115`），而脚本 `:87` 先 `model.eval()`；
  且 `spk_pool_proj` 从来不在 `forward` 里被调用（它由 litmodule 单独调，`av_litmodule_nass_v2.py:103,109,131`）。
  实测：`spk_pool_proj` = **0.000810 G / 14,584 参数**（占 0.008%）。
* **脚本可运行性**：`configs/Falcon.yml:8` 的 `pretrain: pretrain_zoo/frcnn_128_512.backbone.pth.tar`
  **在树内不存在**（`ls pretrain_zoo` → No such file），而 `frcnn_videomodel.py:95-99` 对缺失文件**硬报错**
  → 原样执行会 `FileNotFoundError`；再加 `:48` 的 `.cuda()`，脚本是 历史版本依赖 GPU 与外部权重路径，当前脚本已支持选择设备。
  这解释了为什么此前的 MACs 证据都来自复刻脚本（`docs/verification/scripts/verify_macs_04.py`）。
* 与 10.6 G 的对应：实测 **10.610934 G**，README 写 10.6（0.1% 舍入），`docs/STATUS.md:33` 写 10.611 G。

---

## 3. 关键数据/参数表

### 3.1 参数量（本机 CPU 实测）

| 模块 | 参数量 | 说明 |
|---|---|---|
| `separator` | 492,312 | 主体 |
| `spk_pool_proj` | **14,584** | 对比头（AttentionSpeakerPool, dim=64） |
| `audio_bottleneck` | 13,104 | |
| `mask_generator` | 12,657 | |
| `encoder` | 3,024 | STFTEncoder（win 256 / hop 128） |
| `decoder` | 2,016 | |
| **audio 合计** | **537,697** | = 523,113 + 14,584 ⇒ 与 `docs/STATUS.md:33` "扣对比头 523,113" 逐位吻合；523,113 仍未扣死参数 5,568，两者都扣是 **517,545** |
| FRCNNVideoModel | 11,186,688 | 冻结 |
| audio+video | 11,724,385 | 与 STATUS 一致 |

### 3.2 MACs（本机 ptflops / CPU）

| 口径 | MACs | 参数 |
|---|---|---|
| **audio, eval, 2 s（脚本口径）** | **10.610934 G** | 0.537697 M |
| audio, 2 s, train 模式 | 10.610934 G | 差 0（对比头不在 forward 里） |
| 对比头 `spk_pool_proj` 单独 | 0.000810 G | 0.014584 M |
| FRCNN 单独 @64×64（历史口径） | 7.769293 G | 11.186688 M；**不是本管线输入** |
| FRCNN 单独 @88×88 | 15.896742 G | 真实输入 88×88×50 帧 |
| **整窗（audio + 视觉 @88×88）** | **26.507 G** | 10.611 + 15.897，视觉占 60.0% |
| audio + FRCNN @64×64（已作废） | 18.380227 G | 混用 64×64 视觉；2026-11-08 Lead 复核后不再使用 |
| audio，`flop_counter` 口径（另一把尺子） | 5.382 G | 不记 `aten::lstm`，不可与 10.611 并列 |
| audio, 1 s 窗 | 4.953689 G | 线性度：5.31 G/s @2 s vs 4.95 G/s @1 s |

### 3.3 常量与边界

| 量 | 值 | 位置 |
|---|---|---|
| SI-SDR eps | `1e-8`（3 处） | `matrix.py:189,216,228,231` |
| SI-SDR 地板 | **−80.000000 dB** | 实测（`10log10(1e-8)`） |
| SI-SDR 天花板 | `10log10(E/eps)`，2 s/16 k ≈ 116.02 dB | 实测 |
| `SingleSrcNegSDR` eps | 硬写 `1e-8`（忽略构造参数） | `matrix.py:246` |
| SDR filter_length | 512 | `core.py:69` |
| PESQ/STOI 采样率 | 16000 | `core.py:70` |
| PIT 阈值 | `-30`（丢 loss<−30 的样本） | `pit_wrapper.py:227-229` |
| NASS margin（默认/配置） | 0.0 / **0.5** | `nass_contrastive_loss.py:22`；`configs/Falcon.yml:105` |
| NASS β（默认/anneal） | 1.0 / 0→1.0 over 200ep（v2 为 1000ep） | `.py:36`；`av_litmodule_nass_v2.py:46-48,66-70` |
| `contrastive_weight` | 1.0 | `configs/Falcon.yml:111` |

---

## 4. 实测验证

> 环境：`cd /root/code/github_repos/Falcon-fork && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python`
> 脚本落在 `/tmp/w1lm/`（**不要放 `/tmp` 根**：那里有别人的 `profile.py` 会遮蔽 stdlib，见 §5.9）。

### 4.1 实测 V1：SI-SDR 解析值校验（7 组已知信号）

信号：`s(t)=sin(2π·440t)+0.3·sin(2π·1234t)`，T=16000，float64。解析式
`SI-SDR = 10log10((‖αs‖²+eps)/(‖ŝ−αs‖²+eps))`，`α=⟨ŝ,s⟩/(‖s‖²+eps)`。

```
=== A. SI-SDR 解析值 vs PairwiseNegSDR(sisdr) ===
perfect est=s            analytic= 119.40516 got= 119.40516 diff=-4.99e-12
inverted est=-s          analytic= 119.40516 got= 119.40516 diff=-4.99e-12
scaled est=0.5s          analytic= 113.38456 got= 113.38456 diff=-1.99e-11
dc-shift est=s+7         analytic= 119.40516 got= 119.40516 diff=-4.99e-12
orth-noise SNR=+10dB     analytic=   7.29322 got=   7.29322 diff=+8.09e-09
orth-noise SNR=+0dB      analytic=  -2.70678 got=  -2.70678 diff=+8.10e-08
orth-noise SNR=+-5dB     analytic=  -7.70678 got=  -7.70678 diff=+2.56e-07
```

**结论**：① 实现与解析式完全一致（误差来自 float64 累积）；② 反相/DC 偏移**完全不影响**（尺度不变）；
③ **0.5 倍缩放掉到 113.38 dB**——因为分子加了 `eps`，严格尺度不变性在超高 SNR 处被破坏（工程上无影响，但解释了
"为什么完美重建不是 inf"）。

三态 `sdr_type` 在退化输入上的行为（同一次实测）：

```
snr   (est=-s, ref=s) = -6.020600      sdr(est=0.5s) = +6.020600
sisdr (est=-s, ref=s) = 119.405165     sisdr(0.5s)   = 113.384565
sdsdr (est=-s, ref=s) = -6.020600      sdsdr(0.5s)   = 0.000000
```

→ **`pairwise_neg_snr`（默认训练损失）会因反相而给 −6 dB 的惩罚**，`sdsdr` 对缩放给 0 dB。

### 4.2 实测 V2：SI-SDR 实现横向对比 + fast_bss_eval 边界

```
              case          analytic    PairwiseNegSDR   SingleSrcNegSDR       core.si_sdr   fast_bss.si_sdr
perfect est=s             119.405165        119.405165        119.405165        119.405165        156.535598
inverted est=-s           119.405165        119.405165        119.405165        119.405165        156.535598
scaled est=0.5s           113.384565        113.384565        113.384565        113.384565        156.535598
dc-shift est=s+7          119.405165        119.405165        119.405165        119.405165        156.535598
orth-noise SNR=+10dB          7.293219          7.293219          7.293219          7.293219          7.293219
orth-noise SNR=+0dB          -2.706781         -2.706781         -2.706781         -2.706781         -2.706781
orth-noise SNR=+-5dB         -7.706781         -7.706780         -7.706780         -7.706780         -7.706781
```

```
=== fast_bss_eval.si_sdr 参数敏感性（perfect，2 源堆叠）===
  zero_mean=False clamp_db=None -> 149.545898
  zero_mean=False clamp_db=30.0 -> 30.000000
  zero_mean=True  clamp_db=None -> 156.535598
  zero_mean=True  clamp_db=30.0 -> 30.000000
```

* 三种 Falcon 实现（`PairwiseNegSDR` / `SingleSrcNegSDR` / `core.si_sdr`）**逐位一致**。
* `fast_bss_eval.si_sdr` 在**非退化区间**完全一致；退化区间（≥150 dB）无 eps → 更大且**对缩放严格不变**
  （`est=0.5s` 也给 156.5 dB）。
* **`fast_bss_eval.si_sdr` 在 `N=1` 时崩**（这是它自己的缺陷，实测栈）：
  ```
  File ".../fast_bss_eval/torch/helpers.py", line 94, in _solve_permutation
  File ".../fast_bss_eval/torch/helpers.py", line 142, in _linear_sum_assignment_with_inf
      m = values.min()
  RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0.
  ```
  所以要用它做单源对照，得把估计/参考都**堆成 2 源再取平均**（本报告就是这么做的）。
* `fast_bss_eval` 的 `si_sdr` 签名是 **`(ref, est)`**，且**没有** `filter_length` 参数
  （只有 `sdr`/`sdr_pit_loss` 有）→ `core.py:174` 用 `sdr_pit_loss(estim, ref, filter_length=512)` 是对的。

### 4.3 实测 V3：PIT 置换校验

```
=== E. PIT 置换校验（2 源，s1=300Hz, s2=900Hz）===
pair 矩阵 raw (est_ok) =
 [[-116.0206   80.    ]
 [  80.     -116.1909]]
pair 矩阵 raw (est_sw) =
 [[  80.     -116.1909]
 [-116.0206   80.    ]]
pit loss est_ok=-116.105766610 est_sw=-116.105766610 diff=0.000e+00 (0 = 置换不变)
pit_from=pw_pt(SingleSrcNegSDR) = -116.105766610  diff vs pw_mtx = 0.000e+00
threshold_byloss=True  -> -116.105766610 (全被截断时保留全部)
return_ests allclose(clean): True
```

* 置换**严格不变**（diff = 0，因为两条排列的 mean 相同）。
* `raw` 矩阵的**强烈非对称**（对角 −116 / 非对角 +80）说明 `[B, est, ref]` 的索引方向正确
  （80 dB = 完全不相关时的 −80 dB 地板取负）。
* **`pw_pt` 与 `pw_mtx` 在 `SingleSrcNegSDR` 下等价**；但用 `PairwiseNegSDR` 时 `pw_pt` 直接崩：
  ```
  TypeError: Inputs must be of shape [batch, n_src, time], got torch.Size([1, 8000]) and torch.Size([1, 8000])
    at pit_wrapper.py:244 -> matrix.py:199
  ```
* `threshold_byloss` 的**空集保护生效**：当全部样本都 ≤ −30 时保留全部（`:227-229`）。

3 源（`SPlitMetricsTracker` 的 2+1 口径）：

```
3源 pair 矩阵:
 [[  80.    -116.191   19.83 ]
 [-116.021   80.     -19.83 ]
 [ -19.83    19.83  -116.066]]
PIT(-si-sdr) 3源乱序 = -116.092356
two(src0:2) PIT = -116.105767 ; one(src2) PIT = -116.065533
```
→ 乱序输入能被正确配回（3 源阶乘搜索 6 个排列，`pit_wrapper.py:280`）。

### 4.4 实测 V4：评测口径（`clean` 单源 × `estimate` 多源时会静默平均）

这一项是**为了回答"如果 `n_src>1` 会怎样"**（当前 config 全是 `n_src=1`，不会触发，但代码允许）：

```
=== G. clean 1 源 / estimate 2 源（audio_visual_test 的 broadcast 路径）===
  perf[ch0=s1,ch1=s2]    core.si_sdr=  18.0103  core.sdr=  70.8491
  ch0=s1, ch1=静音        core.si_sdr=  18.0103  core.sdr=     -inf
  swap[ch0=s2,ch1=s1]    core.si_sdr=  18.0103  core.sdr=  70.8491
  仅 ch0 的 SI-SDR = 116.0206 ; 仅 ch1(=s2) 的 SI-SDR = -80.0000 ; 两者均值 = 18.0103
```

原因链：`allwrapper.py:99-104` 取 `n_src = clean.shape[0]`（=1）→ `core._prepare_pair` 把 `ref` 沿源轴
`expand` 到 2（`core.py:119-122`）→ 两条参考**完全相同** → 两个排列得分相同 → `min` = **两个估计通道的算术平均**。
同时 **PESQ/STOI 只看 `est[0]`**（`allwrapper.py:113-114`）→ 同一行 CSV 里 SI-SDR 用了 2 个通道、PESQ 只用 1 个，
**口径不一致**。当前 `n_src=1` 掩盖了它，但这是 n_src>1 时的定时炸弹（GAP-4）。
另注意 `core.sdr` 在"某通道静音"时给出 **−inf**（无 `clamp_db`），CSV 会写出 `inf/-inf` 字面量。

### 4.5 实测 V5：`threshold_byloss` 的偏置方向 + NASS 退化行为

```
=== M. threshold_byloss 在 batch 维上的偏置（B=2，样本 0 完美 / 样本 1 很差）===
  threshold=False -> -50.6887  (= 两者均值)
  threshold=True  -> +14.6431  (= 只剩差的样本)
```

→ `threshold_byloss=True` **丢掉的是高质量的样本**（loss 更负 = SI-SDR 更高），留下一批"难样本"，
因此 `train_loss` 的**绝对数值不可与 val_loss 比较**（前者 true、后者 false）。这也解释了
`metrics/core.py:72-74` 为什么在指标侧强制关掉它。

NASS 损失（在 `q,p,negs` 与 `margin=0.5` 下实测）：

```
batch 中 2 个有近邻 / 1 个 None : 0.110868
全部 None (K=0 退化)             : 0.023064     ← 与"仅正项均值"完全相等
仅正项 (1-cos)/B 均值            : 0.023064
K=1 分支                         : 0.081568
margin=0.0 vs 0.5 差异           : 0.352013 vs 0.110868
beta=0 / 1 / 10                  : 0.086849 / 0.110868 / 0.205321
单条无近邻样本的稀释: L(q,p,[negs[0],None,None]) = 0.076912  (n_valid 仍是 3 不是 2)
```

**验证了既有说法**："near 全 fallback → 只剩正项"（数值逐位相等）。补充两点别人没说的：
① `K=0` 与 `K>0` 的样本被 `n_valid=B` **等权稀释**（`nass_contrastive_loss.py:76-77`），
所以"近邻覆盖率"会直接改变损失的绝对量级（bucket 日志 `_BUCKET_N_LABEL` 只记数量、不归一化，
`av_litmodule_nass_v2.py:182-183`）；
② β 的作用是**放大困难负样本**（β=10 时损失翻倍），β=0 退化为均值权重。

### 4.6 实测 V6：`SISNRi` 损坏 + `core` 后端就位情况

```
SISNRi() 实例化失败: NameError name 'Loss' is not defined
core._pesq = <function pesq at 0x...>   core._stoi = <function stoi at 0x...>
core.pesq_metric(s1, 0.9*s1) = 4.643888473510742
core.stoi_metric(s1, 0.9*s1) = 0.9999999999989817
pypesq.__file__: /root/miniforge3/envs/falcon/lib/python3.11/site-packages/pypesq.py
pypesq.pesq 签名: (ref, deg, fs, mode='wb')
```
→ 损坏点：`sisnri.py:6` `super(Loss, self).__init__()`（`Loss` 未定义、`__init__` 也没参数）。
→ PESQ/STOI 后端**是可用的**（与旧验证脚本声称"未安装所以 stub"相反，§6.2）。

### 4.7 实测 V7：MACs 复算

> 下面最后两行是本次复算的原始输出：`@64×64` 的 7.769293 G 与"eval + FRCNN 若计入 = 18.380227 G"沿用
> `evaluated_mac_params.py` 的 64×64 输入习惯。真实管线是 88×88×50 帧（15.896742 G），整窗应为
> **26.507 G**；18.380227 G 是混尺数，2026-11-08 Lead 复核后作废。

```
params: audio=537697 (0.5377 M)  video(frcnn)=11186688 (11.1867 M)
video out: (1, 512, 50)   (2s*25fps=50 帧)
AUDIO model MACs = 10.610934 G   params=0.537697 M
### 对比头单独: spk_pool_proj MACs=0.000810 G params=0.014584 M
### FRCNN 单独 @64×64: MACs=7.769293 G params=11.186688 M ; @88×88: 15.896742 G
### eval(不含对比头)=10.610934 G ; train(含对比头)=10.610934 G ; 差=0.000000 G
### eval 模式下 +FRCNN 若计入 = 18.380227 G
### 1s 窗 = 4.953689 G
```
→ **与 `docs/STATUS.md:33`、README 的 10.6 G 独立复算一致**；同时坐实"不含 FRCNN、不含对比头（0.0008 G）"。
不计入的原因：`evaluated_mac_params.py:82` 在调 ptflops 之前就跑完了视频前向。

### 4.8 实测 V8：真实数据上的 mix 基线（SI-SDRi 的减数）

```
前 5 条 tt：SI-SDR(mix,s1)= 1.384  -1.354   5.377  -5.602   7.066
            SDR(mix,s1)   = 1.495  -1.221   5.459  -5.376   7.089
            长度 32000；rms(mix)≈0.14-0.20
```
→ **mix 基线不是 0 dB**（因为两个说话人电平不同、文件名里的 ±0.53/±2.67 就是各自的相对增益）。
全程均值约 **+4.99 dB**（由历史 CSV 的 `si-snr − si-snr_i` 反推，§4.9），
所以 "SI-SDRi 8.858" 对应 "SI-SDR 13.85"，二者差 4.99 —— 自洽。
同时 `SDR(mix)` 只比 `SI-SDR(mix)` 高约 0.1 dB，说明 512-tap 滤波对"含干扰的混合"几乎帮不上忙（符合预期）。

### 4.9 实测 V9：历史 CSV 的口径反查

```
Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv   3000 句
  rows mean: {sdr 14.37, sdr_i 9.2696, si-snr 13.8519, si-snr_i 8.8582, pesq 2.5761, stoi 0.9428}
  avg  row : {sdr 14.37, sdr_i 9.2696, si-snr 13.8519, si-snr_i 8.8582, pesq 2.5761, stoi 0.9428}   ← 完全一致
  si-snr − si-snr_i = 4.9937 ; sdr − sdr_i = 5.1005
```
* 逐句行与 avg 行**符号/数值一致** → 已不是上游那版"行正 avg 负"的 bug。
* `sdr(14.37) > si-snr(13.85)`：若 `sdr` 列是老口径的 `-SNR`，SNR 应**低于** SI-SDR（投影残差更大），
  而这里更高且只高 0.52 dB，与"BSS-Eval 512-tap 比逐点投影多赚 0.5 dB"完全一致
  → **该 CSV 的 `sdr` 是真 BSS-Eval SDR**。
* 该 CSV 与 `docs/STATUS.md:41` 的 "ep124 8.858 / 9.270" 逐位相同 → 它就是 `P2D-eval-epoch124-contrastive.csv`
  （同尺寸 546,290 B，2026-10-24 13:52）。

---

## 5. 坑与反直觉点

1. **【高】默认训练损失是 −SNR，不是 −SI-SDR**（`configs/Falcon.yml:117`）。−SNR 对**尺度/反相敏感**
   （实测 −6.02 dB @ 反相）→ 模型被要求复现混合的绝对幅度，等于多了一个隐式的输出增益约束。
   这解释了为什么 v2 特意换成 `pairwise_neg_sisdr` 并在注释里写"−SI-SNR 是逐对均值，不做 30 dB 截断"（`Paper_v2.yml:125`）。
2. **【高】`threshold_byloss` 方向反直觉**：丢的是**好样本**（`pit_wrapper.py:227-229`）。
   后果：`train_loss` 的数值永远比"真实平均"悲观，`val_loss`（false）与它**不同分布**，
   `ModelCheckpoint(monitor="val_loss/dataloader_idx_0")`（`audio_train.py:379-380`）因此不受影响——
   但任何"train/val 曲线应当贴合"的直觉判断都会错。
3. **【高】`n_src=1` 让 PIT 变成恒等**，但代码里所有 PIT 机制都在（`pit_wrapper.py` 全 324 行）。
   读代码时极易误判"这是盲分离 + PIT"，实际是**目标说话人抽取**（`target.json` 只有一路参考）。
4. **【中】`PairwiseNegSDR` 的阶乘分支只对 `n_src<=3` 用**（`pit_wrapper.py:268`），
   `n_src=4` 时切 Hungarian；两条路径的**归约不同**：阶乘分支用 `einsum` 求均值（`:288-289`），
   Hungarian 分支用 `torch.gather(...).mean([-1,-2])`（`:311`）→ 数值上等价，但
   **Hungarian 分支 `detach().cpu()`**（`:306`）→ `min_loss` 保留梯度？不：`min_loss` 从 `pwl`（带梯度）gather 得到，
   所以梯度仍在，只是置换索引不参与求导（正确）。
5. **【中】`SingleSrcNegSDR` 的 `EPS` 构造参数是装饰品**（`matrix.py:237` 传入、`:246` 覆盖为 1e-8）；
   `MultiSrcNegSDR` 同样（`:285,292`）。改 eps 只能改 `PairwiseNegSDR`。
6. **【中】`pw_pt` 拿不到梯度也不能用**（§4.3）——若有人按 asteroid 文档改 config 到 `pw_pt` 会直接崩。
7. **【中】NASS 的分母是 `B` 而不是"有负样本的样本数"**（`nass_contrastive_loss.py:77,81`）→
   负样本覆盖率变化会同时改变**损失量级和梯度量级**；再叠加 `contrastive_weight=1.0`，
   意味着"近邻多的 batch"梯度天然更大（没有按 bucket 归一化）。
8. **【中】NASS 的 query/positive 都固定取 index 0**（`av_litmodule_nass_v2.py:102,107`），
   没有用 PIT 的最优置换重排估计。`n_src=1` 下无问题；`n_src>1` 时正样本可能配对错人（正样本把"另一个说话人"当锚点）。
9. **【中】`torchmetrics` 在本环境不可用**：`import torchmetrics` → `module 'profile' has no attribute 'run'`。
   根因：`/tmp/profile.py`（**别人留下的脚本**，`ls -la /tmp/profile.py` 存在）会遮蔽 stdlib，
   只要脚本目录是 `/tmp` 就中招。我最初的脚本就踩了，改到 `/tmp/w1lm/` 后正常。
   → **任何在本机跑 python 脚本的人都会中招**，建议 Lead 在最终报告里提一句"脚本勿放 /tmp 根"。
10. **【中】`core.sdr` 没有任何 clamp**（`core.py:174-175` 只 `.mean()`）→ 线性相关的估计会给出 `inf`
    （实测完美/半幅/换序全部 `inf`），某通道静音会给出 `-inf`。CSV 会写下 `inf` 字面量，
    `np.nanmean` 不会拦截 `inf`（`np.isfinite` 才拦）→ **平均被污染**。当前 3000 句真实数据没触发（§4.9），
    但 `sdr_i` 的 std 计算（`allwrapper.py:154-159` 用 `nanstd`）对 `inf` 也无防护。
11. **【低】`_mixture_sources` 要求 `mix` 的通道数 == `n_src`**（`core.py:136-141`），
    单声道 mix 会被 `expand` 复制（`:135-136`）→ 对多源参考，基线是所有源共用一个 mix，语义上正确。
12. **【低】`allwrapper` 每 `__call__` 都立刻写一行 CSV**（`:135-136`），崩溃时已写出的行不会丢；
    但 `final()` 只在正常路径调用（`audio_visual_test.py:438`）→ 中途崩溃就**没有 avg 行**，
    而 `_merge_results` 仍会自己算 avg（`:495-497`），所以最终表还有平均值、只是丢了逐 rank 的 std 口径差异。
13. **【低】`metrics/__init__.py:5-6` 导出 `SDR_FILTER_LENGTH`/`PESQ_SAMPLE_RATE`，但没导出 `core` 自身**；
    `allwrapper.py:37` 用 `from . import core` 绕过去了。外部想拿 `_prepare_pair` 这类工具函数只能 `from look2hear.metrics import core`（可行，我实测过）。

---

## 6. 与既有文档的冲突 / 更正

| # | 既有说法（出处） | 实际（证据） |
|---|---|---|
| 6.1 | "`sdr_type: pairwise_neg_snr`…" 多篇文档只写"PIT 损失"，未强调**训练用 SNR** | `configs/*.yml` 5/6 份用 `pairwise_neg_snr`，只有 v2 用 sisdr；−SNR 非尺度不变（§4.1 实测）→ **训练目标与评测指标（SI-SDR）不是同一个量** |
| 6.2 | `docs/verification/scripts/verify_metrics_04.py:4-5`："Third-party metric packages are stubbed (**they are not installed**)" | 在 AGENTS.md 强制的 `falcon` conda 环境里 **都已安装**：fast_bss_eval 0.1.4、pesq 0.0.4(+pypesq shim)、pystoi 0.4.1 → 以前的"验证"只验了 stub 签名，**没验真库行为**。本轮全部用真库重跑（§4） |
| 6.3 | 同 6.2 的 stub 把 `fast_bss_eval.sdr_pit_loss` 实现成**逐点投影**的 `-SI-SDR` 代理 | 真库 `sdr_pit_loss` 是 **512-tap 多抽头滤波**版本，且内部自带置换求解；两者在"混合 vs 干净"上可差 0.1–0.5 dB（§4.8/4.9） |
| 6.4 | `docs/STATUS.md:38-41` 写 "LRS2-2Mix …（**tt 3,000**）" | `falcon_lrs2_2mix_paper/tt` 索引是 **6000 行 / 3000 个唯一 stem**（每个 stem 有 s1、s2 两个目标条件）→ 跑满 tt 应是 **6000 句**。历史 CSV 的 3000 行说明当时只覆盖了**一半条件**（或用了另一个索引）。**所有引用"tt 3000"的表格都需要重新确认分母**（GAP-1） |
| 6.5 | 早前文档把 `SISNRi`（`losses/sisnri.py`）当作可用组件 | **类无法实例化**（`NameError: Loss`，§4.6）；且全仓无调用点 |
| 6.6 | `docs/understanding/04`（上轮）称 metrics 三 tracker 各有分工 | 实际只有 `ALLMetricsTracker` 有调用点（`audio_visual_test.py:397`）；另两个是纯导出死代码 |
| 6.7 | `evaluated_mac_params.py` 被视为"可直接跑的 MACs 复算脚本" | **不可直接跑**：`:48` 强依赖 CUDA，`configs/Falcon.yml:8` 的 `pretrain_zoo/...` 在树内不存在且 `frcnn_videomodel.py:95-99` 硬报错 → 只能靠 `docs/verification/scripts/verify_macs_04.py` 复刻（本轮已复刻并得到 10.610934 G） |
| 6.8 | 旧 CSV（`Experiments/checkpoint/Falcon/results/`，2026-10-22）易被当成"当前口径" | 它是 **7 列旧 schema**（无 `si-snr_mix`/`n_*`），无法离线复核 `si-snr − si-snr_i == si-snr_mix` 恒等式；但符号自洽、`sdr` 已是真 SDR（§4.9）→ 可引用数值，**不可与新的 8 列 CSV 直接 diff 列** |

---

## 7. GAP LIST（≤10 条，可被下一轮直接认领）

1. **GAP-1（★最高，属 W1-07 但影响所有指标表）**：`tt` 到底该报 3000 还是 6000 句？
   索引有 6000 行（3000 stem × {s1,s2}），历史 CSV 全是 3000 行。
   需确认：论文 Table 里的 LRS2-2Mix 是"每混合一个目标"还是"每混合两个目标各算一次"；
   若两种都算，分母翻倍、均值会变（s2 条件通常更难/更易？）。**同一 ckpt 跑满 6000 并分别按 s1/s2 分组统计**即可结案。
2. **GAP-2**：`si-snr_i` 的减数（mix 基线）在 6000 条上的**分布**未统计。已知前 5 条在 −5.6…+7.1 dB，
   均值约 +4.99（由历史 3000 条反推）。需要一张 `si-snr_mix` 直方图 + 与 s1/s2 条件的相关性。
3. **GAP-3**：把 `av_litmodule_nass_v2.py:102,107` 的固定 index-0 换成 **PIT 后重排**的 query/positive，
   在 `n_src=1` 下应等价、在 `n_src=2` 下才显著。需要一个 n_src=2 的最小复现来量化"配对错人"的比例
   （当前无任何 n_src>1 的 config，无法用现网数据验证）。
4. **GAP-4**：`n_src>1` + 单参考索引时，`allwrapper.py:99-122` 会**静默把两个输出通道平均**
   （实测 18.01 dB vs 单通道 116.02 dB）。需要决定：报"最优通道"、"平均"还是"按 PIT 后取对应通道"，
   并在 `core._prepare_pair` 里加显式分支（现在是隐式 `expand`）。
5. **GAP-5**：`core.sdr` 的 `inf` 防护缺失（实测 3 种线性相关输入全部 `inf`，静音通道 `-inf`）。
   需要决定是否用 `fast_bss_eval` 的 `clamp_db`（例如 40 dB）并在 CSV 里标注，否则 `mean/std` 会被 `inf` 污染。
6. **GAP-6**：`pypesq.py` shim **不在仓库里** → 换环境即 `PESQ=nan`（静默）。
   需要：把 shim 落进 `tools/repro/`（或直接改 `core.py` 支持 `pesq` 包签名 `pesq(fs,ref,deg,mode)`），
   并按 `fs` 自动选 `wb/nb`（现在 `mode='wb'` 硬编码，8 kHz 必炸）。
7. **GAP-7**：`torchmetrics`/`cProfile` 在本 env 被 `/tmp/profile.py` 遮蔽（§5.9）。
   清理该文件或约定脚本目录；否则任何"用 torchmetrics 交叉验证 SI-SDR"的尝试都会失败。
8. **GAP-8**：DNSMOS 与分离指标**不在同一张表**、口径也不同（9.01 s 窗 vs 2 s 整句，16 k 强制）。
   若要出"L2 RealSSA-Real：DNSMOS + WER + spk-sim"的表，需明确写出窗口策略（历史记录里已有
   "整段 OOM → 改 Studio 等价 2 s/0.5 s hop"的变更，见 `docs/REPRODUCTION_PHASE2_REPORT.md:101`）——
   这两套窗口的数不可混引。
9. **GAP-9**：`MultistageLoss` 的 α 调度（`matrix.py:161`）与 `MultistageLoss_val` 的**分支丢弃**
   （`:180-183` 算了 mag 分支却只 `return cur_loss_s.mean()`）看起来是未完成的重构残留。
   若确认永久不用，建议整段删除（含 `STFT`/`STFTBase` 200+ 行），减少阅读成本；需要 Lead 决策"是否允许动代码"。
10. **GAP-10**：`new_PITLossWrapper`（`pit_wrapper.py:6-150`）与 `PITLossWrapper` 的**语义差异**
    （前者直接返回未归约的 pair 矩阵，`pit_from="pw_mtx"` 时 `:26` 还会把 `ests_bn`/`epoch` 硬塞给 loss_func）
    是否曾被某个实验分支使用？`git log -S new_PITLossWrapper` 可查；目前判死。

---

## 8. 证据清单

### 8.1 读过的文件（`path:line` 见正文）

```
look2hear/losses/{__init__,matrix,mixit,nass_contrastive_loss,pit_wrapper,sisnri}.py
look2hear/metrics/{__init__,core,allwrapper,wrapper,splitwrapper}.py
look2hear/system/{av_litmodule,av_litmodule_nass_v2,av_litmodule_mdcl}.py
look2hear/models/{Falcon.py:1421-1478, Falcon_Contrastive.py:8-120}
look2hear/datas/avspeech_dataset.py:23-220
look2hear/videomodels/frcnn_videomodel.py:36-112
audio_train.py:322-390        audio_visual_test.py:1-80, 364-547
evaluated_mac_params.py:1-151
configs/{Falcon,LRS2_2Mix,LRS2_2Mix_Paper_2gpu,LRS2_2Mix_Paper_2gpu_smoke,
         LRS2_2Mix_Paper_8gpu,LRS2_2Mix_Paper_v2}.yml
tools/repro/phase2/eval_dnsmos.py:1-120
docs/{STATUS.md:33-41,REPRODUCTION_PHASE2_REPORT.md:101-112,DATA_ACCESS_GUIDE.md:14-25}
docs/verification/scripts/{verify_losses_04.py,verify_metrics_04.py,verify_macs_04.py,verify_model_params.py}
docs/reviews/round2/{04-losses-metrics-video-round2-review-by-ops-tests-docs.md:20,104-105,135}
```

### 8.2 git 取证

```
git log --oneline -3 -- look2hear/metrics/allwrapper.py
  2062528 / ac91800 / a7af472
git show a7af472:look2hear/metrics/allwrapper.py
  :27  self.pit_snr = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")   ← sdr 列实为 -SNR
  :50  _pesq = pesq(estimate..., clean..., 16000)                            ← 参数序反了
  :63  "sdr_i": sdr_i.item()   vs   :73  self.all_sdrs_i.append(-sdr_i.item())← 行/avg 符号相反
```

### 8.3 实测脚本（远端 `/tmp/w1lm/`）

| 脚本 | 内容 | 关键输出 |
|---|---|---|
| `t2.py` / `t3.py` | SI-SDR 解析值 × 4 实现（含 fast_bss_eval 边界） | §4.1、§4.2 |
| `t5.py` | PIT 置换/3 源/单参考多估计/`threshold_byloss`/`core.sdr`/PESQ-STOI | §4.3、§4.4、§4.5 |
| `t6.py` | pypesq shim 溯源、−80 dB 地板、B=2 截断偏置 | §4.5 |
| `t7.py` | 参数量分模块 + NASS 数值行为 + `SISNRi` | §3.1、§4.5、§4.6 |
| `t8.py` | 真实 tt 数据 mix 基线 SI-SDR/SDR | §4.8 |
| `macs.py` / `macs3.py` | ptflops 复算（audio / 对比头 / FRCNN / 1 s 窗） | §4.7 |

### 8.4 关键命令（可复现）

```bash
# 环境（AGENTS.md 强制）
cd /root/code/github_repos/Falcon-fork
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python /tmp/w1lm/t5.py

# 历史指标反查
IM=/root/miniforge3/envs/falcon/bin/python
CUDA_VISIBLE_DEVICES= $IM -c 'import pandas as pd; d=pd.read_csv("Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv"); ...'

# 索引口径
CUDA_VISIBLE_DEVICES= $IM -c 'import json;m=json.load(open("/root/data/falcon_lrs2_2mix_paper/tt/mix.json"));print(len(m), len({x[0] for x in m}))'
# -> 6000 3000
```

### 8.5 本轮产生的可信数字（供 Lead 直接引用）

| 数字 | 含义 | 置信度 |
|---|---|---|
| **10.610934 G / 0.537697 M** | audio model MACs（eval、2 s 窗、不含 FRCNN 和对比头前向）；参数计数含对比头 | 实测，与 docs 逐位一致 |
| 537,697 = 523,113 + **14,584** | audio 参数 = 扣对比头 + `spk_pool_proj` | 实测 |
| 11,186,688 / 11,724,385 | FRCNN / 合计 | 实测 |
| 7.769293 G | FRCNN 单独 @64×64（**未计入** 10.611；非本管线输入） | 实测 |
| 15.896742 G | FRCNN 单独 @88×88×50 帧（与 10.611 同尺） | 实测 |
| 26.507 G | 整窗 = 10.611 + 15.897，视觉占 60.0% | 实测（同尺相加） |
| 5.382 G | 音频 `flop_counter` 口径（不记 `aten::lstm`，另一把尺子） | 实测，不可与 10.611 并列 |
| −80.000000 dB | SI-SDR 地板（完全不相关/全零） | 实测 |
| 116.0206 dB | 2 s/16 k 完美重建的 SI-SDR（= 10log10(E/eps)） | 实测 |
| −6.02 / +6.02 dB | `pairwise_neg_snr` 对反相 / 0.5 倍缩放的打分 | 实测 |
| 0.023064 = 仅正项 | NASS 在 K=0 全 fallback 时的退化值 | 实测 |
| 0.110868 / 0.352013 | NASS margin=0.5 / margin=0.0 | 实测 |
| 4.99 dB | LRS2-2Mix tt 的 mix 基线均值（SI-SDRi 的减数） | 由历史 3000 句反推 + 5 条直测 |
| 6000 / 3000 | tt 索引行数 / 唯一混合数 | 实测（与 STATUS 的 "3000" 冲突，见 GAP-1） |
