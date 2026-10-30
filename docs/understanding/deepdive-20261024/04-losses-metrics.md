# 04 深挖：损失函数与评测指标（PIT / NASS / SI-SNR / SDR / PESQ / STOI / DNSMOS）

> 任务：共享任务 `task-4`（深挖小组 `losses-metrics`）· 只读分析 + 只写本文件
> 仓库/基线：`/root/code/github_repos/Falcon-fork`，分支 `repro/phase2`，HEAD `2beec38`
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Py3.11.17 / torch 2.8.0+cu128 / fast-bss-eval 0.1.4 /
> pesq 0.0.4 + `pypesq.py` shim / pystoi 0.4.1 / speechmos 0.0.1.1 / onnxruntime 1.31.0），命令前 `PYTHONPATH=$PWD`
> 证据标记：`[代码]` 直接读源码并给出 file:line；`[文档]` 仓库内文档/README 记载；`[验证]` 本次实际运行；
> `[推断]` 由代码+文档合理推断。**本任务不修改源码、不重跑全量评测、不联网、不 pip 安装。**
> 对照对象：`docs/understanding/04-losses-metrics-video.md`（旧文档，1191 行，基于 main `7576854` 的**修复前**代码）、
> `docs/REPRODUCTION_PHASE1_REPORT.md`、`docs/verification/phase2/{P2A-report.md,P2F-report.md,P2F-results.json}`。

---

## 0 一句话结论

**当前 `look2hear/metrics/core.py` 这一套定义（SI-SDR 去均值 + 源维 PIT；SDR=`-fast_bss_eval.sdr_pit_loss(
filter_length=512)`；PESQ 宽带 16 kHz 且 `ref=clean, deg=estimate`；STOI `extended=False`）是唯一可信的评测口径**：
用它复算官方权重在 RealSSA-Sim test 全量 9,169 条得到 **SI-SNRi 7.774 / SDRi 8.479**，与论文 7.7 / 8.2 分别相差
**+0.07 / +0.28 dB**（同分布锚点闭合）；而 LRS2-2Mix 的 14.1 / 14.7 **不是指标口径问题**——指标/协议变体合计
≤0.4 dB（P2-A 14 变体、P2-F 独立对拍 4.8e-8 dB），差的是 **LRS2-2Mix 训练出的 checkpoint**（发布 `falcon.pt`
是 LRS3 训练，跨语料只有 7.03 / 7.71）。另有三条必须写进复现报告的“同名不同义”：
① 训练 PIT 损失是 **负 SNR**（`pairwise_neg_snr`，`configs/Falcon.yml:117`），验证/评测是 **SI-SDR**，二者不可比；
② 训练/验证用的 `val_pit_sisnr = −val_loss` 是 **SI-SDR(est)**，不是 **SI-SNRi**（没有减 mixture 基线）；
③ NASS 对比损失在 RealSSA（无 near 波形）上 **K=0 → 只剩正项**；LRS2 用 s2 当 near 时 **K=1 → 代码跳过 softmax，
β 退火完全失效**；MDCL 是死代码，**不在论文最终模型**里。

---

## 1 边界与关键路径

### 1.1 文件边界（本任务的读写范围）

| 文件 | 行数 | 角色 | 是否被主路径使用 |
|---|---|---|---|
| `look2hear/losses/matrix.py` | 383 | `PairwiseNegSDR`（负 SI-SDR/SD-SDR/SNR 矩阵）、`SingleSrc/MultiSrcNegSDR`、多阶段/频域死损失 | ✅ `PairwiseNegSDR` 被 PIT wrapper 使用 |
| `look2hear/losses/pit_wrapper.py` | 325 | `PITLossWrapper`（6 种 `pit_from`、阈值、全排列/匈牙利）、未完成的 `new_PITLossWrapper` | ✅ train/val 都用 `pw_mtx` |
| `look2hear/losses/nass_contrastive_loss.py` | 81 | NASS 同空间说话人对比损失（正项 + hard-negative 负项） | ✅ `av_litmodule_nass_v2.py:53` |
| `look2hear/losses/sisnri.py` | 42 | `SISNRi` 对象（SI-SNR 改善量） | ❌ 不可实例化（`NameError`），死代码 |
| `look2hear/losses/mixit.py` | 309 | MixIT 包装器（4 估计/2 混合） | ❌ 全仓库无调用 |
| `look2hear/losses/__init__.py` | 44 | 导出 9 个负 SDR 单例、PIT/MixIT、NASS、SISNRi | ✅ 训练入口 `getattr` 取单例 |
| `look2hear/metrics/core.py` | 229 | **指标唯一入口**：SI-SDR/SI-SDRi/SDR/SDRi/PESQ/STOI | ✅ `allwrapper`/`wrapper` 全部走它 |
| `look2hear/metrics/allwrapper.py` | 148 | `ALLMetricsTracker`：逐句 7 列 CSV + avg/std | ✅ `audio_visual_test.py:66,397` |
| `look2hear/metrics/wrapper.py` | 103 | 旧 `MetricsTracker`（5 列、无 PESQ） | ❌ 无入口调用（保留 API） |
| `look2hear/metrics/splitwrapper.py` | 115 | `SPlitMetricsTracker`（3 源 `one/two` 拆分） | ❌ 无入口调用（legacy） |
| `look2hear/metrics/__init__.py` | 27 | 导出 3 个 tracker + core 指标函数 | ✅ |
| `audio_visual_test.py` | 555 | 离线评测入口：配置/权重加载、逐句推理、CSV 合并 | ✅ 主评测入口 |
| `tools/repro/phase2/eval_dnsmos.py` | 281 | wav → SIG/BAK/OVRL/P808（speechmos ONNX） | ✅ RealSSA-Real 用 |
| `tools/repro/phase2/eval_realssa_real.py` | 697 | RealSSA-Real 54 段整段/窗口推理 + DNSMOS（channel/align_tail） | ✅ RealSSA-Real 用 |
| `tools/repro/phase2/{gap_analysis.py,protocol_check.py,windowed_diag.py}` | — | P2-A/P2-F 的**独立**指标实现与协议变体 | ✅ 但与本域 `core.py` 独立 |

> 边界声明：视觉编码器（`look2hear/videomodels/**`）、训练系统（`look2hear/system/**` 的 optimizer/DDP）、
> 数据管线（`look2hear/datas/**`）只在“接口契约”深度出现（本文件只讲它们如何喂损失/指标）。
> 论文 RealSSA-vs-LRS2 的训练配方差距归 P2-D/训练域，本文件只给出“指标口径不是原因”的量化证据。

### 1.2 四条关键路径（按调用方向）

**① 训练损失链（PIT + NASS）**

```
configs/Falcon.yml:114-126  loss.train = PITLossWrapper(pairwise_neg_snr, pit_from=pw_mtx, threshold_byloss=true)
                            loss.val   = PITLossWrapper(pairwise_neg_sisdr, ..., threshold_byloss=false)
configs/Falcon.yml:104-111  spk_contrast = {margin .5, α_pos/α_neg 1, β 0→1/200, weight 1}
audio_train.py:282-297      loss_func = {"train": ..., "val": ...}（getattr(look2hear.losses, 单例名)）
audio_train.py:299-331      AudioVisualLightningModuleNASSv2(contrastive_weight/margin/alpha/beta...)
av_litmodule_nass_v2.py:95  pit_loss = loss_func["train"](est_sources[B,1,T], targets)
av_litmodule_nass_v2.py:102-134  query/positive/negatives → NASSContrastiveLoss(q,p,negs,β(t))
av_litmodule_nass_v2.py:137-138  total_loss = pit_loss + contrastive_weight * cl_loss
```

**② NASS 负样本链（决定“K=0/K=1”语义）**

```
near.json（DataPreProcess/generate_near_json.py 生成）
  → avspeech_dataset_near.py:185-222 每条样本读 near wav 列表（fallback 路径 == mix 路径时跳过 :190-192）
  → av_litmodule_nass_v2.py:115-132  每样本 K=len(near_audios[b])；K=0 时 append None
  → nass_contrastive_loss.py:61-74   K=0 → l_neg=0（只剩正项）；K=1 → relu（无 softmax）；K≥2 → softmax(β·cos)
```

**③ 评测指标链（audio_visual_test → ALLMetricsTracker → core）**

```
audio_visual_test.py:397     metrics = ALLMetricsTracker(save_file=...)
audio_visual_test.py:420-438 每句 metrics(mix=..., clean=sources[None], estimate=est, key=...)
allwrapper.py:63-92          core.si_sdr / core.sdr / core.pesq_metric / core.stoi_metric
audio_visual_test.py:476-486 合并各 rank CSV：剔除 avg/std 行 → 重算 avg/std → metrics_combined.csv
```

**④ DNSMOS 链（RealSSA-Real）**

```
eval_realssa_real.py:243-305  读 stereo wav → channel mean/0/1；align_tail 截到 min(mouth,floor(samples/640))
eval_realssa_real.py:381-437  窗口化分离（2 s 窗 / 0.5 s hop / 80 ms guard，OOM 规避）
eval_realssa_real.py:646      score_array(estimate) → speechmos DNSMOS（9.01 s 窗 / 1 s hop，逐窗均值）
```

### 1.3 三个“同名不同义”速查

| 名字 | 训练/日志里实际是什么 | 评测里实际是什么 | 差异量级（实测） |
|---|---|---|---|
| “PIT loss” | `PairwiseNegSDR("snr")` 的 PIT 归约（尺度相关 SNR，且 `threshold_byloss=True` 过滤 SNR>30 dB 样本） | — | `est=2·s` 时训练 SNR 损失 = **0.000 dB**，而 SI-SDR = **116.191 dB** |
| `val_pit_sisnr` | `−val_loss` = SI-SDR(est)（`av_litmodule.py:163`） | SI-SNRi = SI-SDR(est) − SI-SDR(mix) | LRS2 实测 val 13.391 vs test SI-SNR 13.612 vs **SI-SNRi 8.858**（差 ≈4.75 dB 基线） |
| CSV `sdr/sdr_i`（旧代码） | `−SNR/SNRi`（`PairwiseNegSDR("snr")`） | 现在的真 SDR/SDRi（fast_bss_eval 512-tap） | 同一合成样本旧 `−22.139` / 新 `+22.208`；真实 LRS2 上 SDR−SI-SDR ≈ +0.79 dB |

---

## 2 关键事实与数字（命令 + 输出）

### 2.0 复跑命令（全部 CPU、无 GPU、总耗时 < 3 min）

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python

# ① 损失函数对拍（官方 Phase-1 脚本，真实环境无需 stub）
PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_losses_04.py            # <10 s，全 PASS

# ② 指标 tracker 脚本（**已过期**，见 §3 第 17 行；在真实环境会崩，见 §7）
PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_metrics_04.py           # exit=1 RuntimeError

# ③ 本任务自写的 10~120 行对拍（手算 SI-SDR / PIT / old-vs-new tracker / DNSMOS / P2-F 抽验）
PYTHONPATH=$PWD $ENV /tmp/lm/deepdive_verify.py                               # <30 s，见 §7.2
$ENV -m pytest tools/repro/tests/test_metrics.py -q                           # 8 passed in 4.35 s
```

### 2.1 PIT SI-SNR：定义、permutation 与 SI-SNRi 的关系

**定义**（`core.py:152-156` 通过 `PIT_SI_SDR = PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=True),
pit_from="pw_mtx", threshold_byloss=False)`，`core.py:75-79` / `matrix.py:188-232`）：

```
e = est − mean(est);  s = clean − mean(clean)
α = <e, s> / (‖s‖² + 1e-8)
SI-SDR = 10·log10( ‖α·s‖² / (‖e − α·s‖² + 1e-8) )          # dB，越大越好
PIT：SI-SDR_PIT = max_{π∈S_S} (1/S) Σ_i SI-SDR(est_{π(i)}, clean_i)   （S=n_src）
SI-SDRi = SI-SDR_PIT(est, clean) − SI-SDR_PIT(mix, clean)              （core.py:158-162）
```

- `[代码]` 逐对矩阵：`pair_wise_dot` / `s_target_energy` / `pair_wise_proj`（`matrix.py:212-218`），
  `e_noise = E − P`（`matrix.py:224-225`），`pair_wise_sdr = ΣP²/(Σe_noise²+EPS)`（`matrix.py:227-229`），
  取 `−10log10` 得损失（`matrix.py:230-232`）。
- `[代码]` PIT 排列：`pit_from="pw_mtx"` 时 `find_best_perm`（`pit_wrapper.py:204-207`）在 `n_src≤3` 时
  **穷举全排列**（`pit_wrapper.py:276-301`，用 one-hot + `einsum` 求每个排列的源平均），否则 scipy 匈牙利
  （`pit_wrapper.py:303-312`）。`threshold_byloss` 是训练 wrapper 的默认 `True`，会**丢掉最佳排列损失 > −30 dB
  （即 SI-SDR>30 dB）的“过好”样本**（`pit_wrapper.py:227-230`）；指标侧显式关掉它（`core.py:78`），否则近完美
  估计会从均值里消失。
- `[验证]` 三种 `sdr_type` 与手算逐元素 `max|diff|=0`；`zero_mean=True` 对 +3 直流不变（3.05e-5）、`False` 时差
  47.815 dB；SNR 对 `est→2·est` 变化 −4.015 dB 而 SI-SDR 变化 0（尺度不变）——`docs/verification/scripts/
  verify_losses_04.py` §[1]-§[3]。
- `[验证]` **PIT 确实在换排列**（2 源、带噪、估计通道故意交换）：
  `PIT SI-SDR = 20.895 dB`（aligned 与 swapped 相同），而同索引平均（不做 PIT）swapped 为 **−48.247 dB**
  → PIT 增益 **69.14 dB**；单源时 PIT 退化为取矩阵唯一元素（Falcon `n_src=1`，`configs/Falcon.yml:12`）。
- `[验证]` **SI-SNRi 算术**：同一构造下 `SI-SDR(est)=22.114329`、`SI-SDR(mix)=6.017407`、
  `si_sdr_improvement=16.096922 = 22.114329−6.017407`（差 0.00e+00）；`est=mix` 时改善恰为 **+0.00e+00 dB**。
- `[验证]` **与 `matrix.py` 一致**：`core.si_sdr` 与手写 numpy float64 公式在 1 s 谐波+噪声上差 **1.83e-10 dB**
  （`test_metrics.py` 另用正交噪声做解析值对拍，容差 ±0.05 dB，8 passed）。

**与 SI-SNRi 的关系（要点）**：评测报的 `si-snr_i` 是 **SI-SDR 的改善量**，基线是“同一 mix 对同一 clean 的
SI-SDR”；不是“SI-SDR(est) 本身的绝对值”。论文表中的 SI-SNRi 与训练日志 `val_pit_sisnr` 差的就是这个基线
（LRS2 官方权重：SI-SNR 12.023、SI-SNRi 7.029 ⇒ 基线 4.994 dB，与数据 SIR 均值 4.974 dB 吻合）。

### 2.2 NASS 对比损失：正/负样本、margin、β、K 的特殊情形

**公式**（`nass_contrastive_loss.py:1-11` 文件头注释与 `:52-77` 实现）：

```
对每个样本 b（q,p L2 归一化，D 维）：
  L_pos = α_pos · (1 − cos(q_b, p_b))                                     # :57-58
  K_b = len(N_b)（near 说话人个数）
  K_b = 0 : L_neg = 0                                                     # :73-74  → 只剩正项
  K_b = 1 : L_neg = α_neg · relu(cos(q_b, n_1) − margin)                  # :68-69  → 无 softmax
  K_b ≥ 2 : w_k = softmax_k(β · cos(q_b, n_k))                            # :71
            L_neg = α_neg · Σ_k w_k · relu(cos(q_b, n_k) − margin)        # :72
  loss = Σ_b (L_pos + L_neg) / B                                          # :76-80（分母恒为 B）
```

- `[代码]` **near 波形 vs mix fallback**（这是 RealSSA 复现的关键）：`generate_near_json.py` 在没有 near 源时
  把 near 条目写成 **mix 自己的路径**作哨兵；`avspeech_dataset_near.py:190-192` 检测 `near_path == mix_path`
  就 `continue`，最终 `near_audios[b] = []` → K=0。`tools/repro/realssa_to_falcon.py:482-489` 明确打印
  “RealSSA-Sim ships no near-speaker waveforms … near.json entries fall back to mixture”。
- `[代码]` **K=0 时只剩正项且仍参与训练**：`nass_contrastive_loss.py:74` 置零负项，`:77` 仍 `n_valid += 1`，
  所以 batch 平均分母恒为 B（不是“有效负样本数”）；`n_valid==0` 只有 B=0 才可能触发（`:79-80`）。
- `[代码]` **K=1 时 β 完全失效**：`:68-69` 直接 `relu`，不计算 `softmax(β·cos)` → **β 退火对 K≤1 的样本无任何
  影响**（softmax 恒为 1，数学上等价，但“β 调度”这一复现项在 K=1 数据上不可观测）。
- `[代码]` `margin=0.5`（`configs/Falcon.yml:105`）：负样本 cos < 0.5 时被 `relu` 截断，不再贡献梯度；
  只有“已经像目标说话人”的 hard negative 才推。
- `[代码]` β 退火（`av_litmodule_nass_v2.py:66-70`）：`β(t) = β_init + (β_final−β_init)·min(epoch/anneal, 1)`；
  `configs/Falcon.yml:108-110` = `0→1 over 200 epochs`，与发布 run 快照
  `Experiments/checkpoint/Falcon/conf.yml:134-141` 一致；system 默认值 `beta_final=1.5/anneal=100`
  （`av_litmodule_nass_v2.py:46-48`）只是兜底，`audio_train.py:304-306` 总会显式传参。
- `[验证]` 与独立 Python 手算逐项对拍：`β=0/1/5 → 0.959351 / 0.967698 / 1.003706`，`max|diff| ≤ 5.96e-8`；
  构造“负样本 cos 全 < margin=1.5”时损失 = **1.5187**（只剩正项）；全 batch 无负样本时损失 = **0.7594**
  （= mean 正项），证明 K=0 分支行为正确（`verify_losses_04.py` §[10]）。
- `[代码]` 内部**不 detach** `p`/`N`（`:57,:64` 直接对入参求 cos），梯度只经 q 有定义是调用方的责任；
  `av_litmodule_nass_v2.py:106-109,129-131` 用 `torch.no_grad()` 包住 positive/negative 侧（`verify_losses_04.py`
  的梯度探针显示 loss 内部对 p 也有梯度 → 调用方必须保持 no_grad）。
- `[文档]` **论文最终模型用的是 NASS（encoder-space 说话人对比），不是 MDCL**：`README.md:58`
  “An encoder-space speaker contrastive loss suppresses near-field leakage at zero inference cost…”；
  发布 run 快照 `Experiments/checkpoint/Falcon/conf.yml:161` `system: AudioVisualLightningModuleNASSv2`。

### 2.3 MDCL：确实是死代码（且不在论文最终模型里）

| 证据 | file:line | 结论 |
|---|---|---|
| MDCL 实现（幅度谱 SI-SNR 加权推离，`_mdcl_loss_hard`） | `look2hear/models/Falcon.py:1410-1462` | 代码存在且数学完整（K=0→0，K=1→relu，K≥2→softmax(β·sim)·relu(sim−margin)） |
| `Falcon.forward` 的 MDCL 分支 | `Falcon.py:1482-1486` | 只有 `self.training and near_audio is not None` 才返回 `(est, mdcl_loss)` |
| 谁传 `near_audio`？ | `Falcon_Contrastive.py:94-95` `forward(..., **kwargs)` | 对比子类 **静默吞掉** `near_audio`；基类 system 也不传（`av_litmodule.py:66-73` 只传 mix/mouth）→ 分支不可达 |
| MDCL Lightning module | `look2hear/system/av_litmodule_mdcl.py:37` | **未导出**：`look2hear/system/__init__.py:1-11` 的 `__all__` 无它；`audio_train.py:26` 只 import NASSv2 |
| 任何配置选中 MDCL？ | `grep system: configs/*.yml` → 只有 `AudioVisualLightningModuleNASSv2` | 无 |
| 别名修复 | `Falcon.py:1464-1467` `_mdcl_loss = _mdcl_loss_hard`（Phase-1 F14 半修） | 不再 `NameError`，但仍无调用者 |
| 论文配置快照 | `Experiments/checkpoint/Falcon/conf.yml:161` system=NASSv2；`:101` `use_near: true` | 发布模型训练走 NASS，不走 MDCL |

`[推断]`（强）：MDCL 是历史/备用路线，**论文最终模型未使用**；`docs/REPRODUCTION_PLAN.md:59-60,258` 同样写
“配置文件显示最终模型用 NASS 对比损失，MDCL 代码路径实际不可用”。因此复现主线不需要 MDCL；
若要做 MDCL 消融，需先补 `AudioVisualLightningModuleWithMDCL` 的接线（且它与对比子类的 tuple 返回语义冲突）。

### 2.4 指标历史 bug 与现状（实测 old vs new）

`[验证]` 直接把上游 `a7af472:look2hear/metrics/allwrapper.py`（修复前）与当前实现跑在同一合成样本上
（1 s 谐波 ref、mix=ref+0.5·noise、est=ref+0.02·noise，16 kHz；命令见 §7.2-③）：

| 行 | 旧 `allwrapper`（a7af472） | 新 `allwrapper`（当前） |
|---|---|---|
| 逐句 `sdr` | **−22.139**（实际是 −SNR 的损失值） | **+22.208**（真 SDR，fast_bss_eval 512-tap） |
| 逐句 `sdr_i` | −16.095 | +16.141 |
| `avg` 行 `sdr` / `sdr_i` | **+22.139 / +16.095**（把逐句行再取负） | +22.208 / +16.141（与逐句同号） |
| 逐句 `si-snr` / `si-snr_i` | 22.143 / 16.166 | 22.143 / 16.166（两版一致，旧代码此处本来就对） |
| `pesq` | **2.73**（`pesq(estimate, clean)` 传反） | **1.03**（`pesq(clean, estimate)`） |

- `[代码]` 旧 bug 位置：`sdr = self.pit_snr(...)`（旧 `allwrapper.py:38-46`，`pit_snr=PairwiseNegSDR("snr")`
  即 −SNR）；`row["sdr"]=sdr.item()`（`:62`）而累加 `append(-sdr.item())`（`:72`）→ 逐句/avg 符号相反；
  `pesq(estimate, clean, 16000)`（`:50`）。旧 `MetricsTracker` 还有 `PairwiseNegSDR("sisdr", zero_mean=False)`
  （旧 `wrapper.py:30`）。
- `[代码]` 现修复位置：`allwrapper.py:72-79` 全部改走 `core.sdr` / `core.si_sdr` / `core.pesq_metric`；
  `core.sdr` = `−fast_bss_eval.sdr_pit_loss(est, ref, filter_length=512)`（`core.py:168-178`）；
  `core.pesq_metric(clean, estimate)`（`core.py:196-205` → shim `pypesq.pesq(ref, deg, fs)` →
  `pesq.pesq(fs, ref, deg, mode="wb")`，shim 见 AGENTS §5）；CSV 读写完全一致（`allwrapper.py:95-105`）。
- `[验证]` 当前 tracker 的逐句行 == avg 精确成立：3 个样本（good / est=mix / 2×ref）下
  `sdr` 逐句 `[22.195, 6.155]`、avg `14.175`（=`mean(逐句)`，差 <1e-9）；`est=mix` 的 `si-snr_i = 0.000e+00`；
  `std` 行是 `np.nanstd`（ddof=0）。
- `[验证]` 合并逻辑也已修（旧 B10）：`audio_visual_test.py:476-486` 先剔除 `avg/std` 行、再用逐句行重算
  `avg`/`std`（旧代码把所有行 concat 后再求均值，会把 avg 行当样本重复计入）。
- `[验证]` **真实数据上的符号/数值**：P2-F 全 3000 条 `metrics_combined.csv` 的 avg 为
  `sdr 12.813 / sdr_i 7.713 / si-snr 12.023 / si-snr_i 7.029 / pesq 2.196 / stoi 0.916`，
  且 `si-snr − si-snr_i = 4.994 dB` 与数据 SIR 均值 4.974 吻合（P2F-results.json `metrics.csv_avg`）。

### 2.5 SDR / PESQ / STOI 的当前口径（逐项实测）

| 指标 | 当前实现 | 关键参数 | 实测口径证据 |
|---|---|---|---|
| SDR | `−fast_bss_eval.sdr_pit_loss(est, ref)`（`core.py:168-178`） | `filter_length=512`、`zero_mean=False`（fast_bss_eval 默认）、内部含 PIT | `[验证]` 同一样本 `sdr_pit_loss(est,ref,512)=−22.1946` → `core.sdr=22.1946`；`SDR(mix)=6.1554`、`SDRi=16.0392`；filter 64/128/256/512 = `22.1290/22.1382/22.1583/22.1946`（抽头越长越高）；`SDR(2×est)=22.1946`（尺度不变） |
| SI-SDR | `core.si_sdr`（`core.py:152-156`） | `zero_mean=True`、PIT、float64 | `[验证]` 与手算差 1.83e-10 dB |
| SI-SNRi | `core.si_sdr_improvement`（`core.py:158-162`） | 基线 = `SI-SDR(mix, clean)` | `[验证]` `est=mix` → 0.000e+00 |
| PESQ | `core.pesq_metric(clean, estimate)`（`core.py:196-205`） | 16 kHz、**wideband `mode="wb"`**、`ref=clean` 在前 | `[验证]` 合成的近干净 est：正确顺序 1.0297，传反 2.7158（差 1.69 MOS）；真实 LRS2 句对：1.1196 vs 1.1830 |
| STOI | `core.stoi_metric(clean, estimate)`（`core.py:208-217`） | `pystoi.stoi(x=clean, y=deg, fs, extended=False)` | `[验证]` 真实 LRS2 句对 `STOI(clean,mix)=0.764728`、传反 0.729854（非对称 ≈0.035） |
| 通用 | `core._to_float64_cpu`（`:82-88`）统一 detach→CPU→float64；`_prepare_pair`（`:115-129`）支持 `[T]/[S,T]/[1,S,T]` 并做源/批广播 | — | `[验证]` `allwrapper` 对 `clean=[1,T]`、`estimate=[1,T]` 正常 |

- `[代码]` 异常保护：SDR/PESQ/STOI 失败返回 `nan` 并 warning（`core.py:173-177,204-205,216-217`），
  不再让一条坏样本炸掉整轮评测；`ALLMetricsTracker` 对 `nan` 用 `np.nanmean/np.nanstd`
  （`allwrapper.py:107-119`），与 `_merge_results` 的 pandas `mean/std(skipna)` 一致。
- `[验证]` `est=2×ref` 这种退化输入上 `core.sdr` 返回 **nan**（fast_bss_eval 奇异），而 SI-SDR 仍给 116 dB；
  这是“SDR 可能对退化样本 nan”的真实边界，不是 bug（有 warning + nan 处理）。
- `[代码]` `wrapper.py` 的 `MetricsTracker` 也已重写为走 `core`（`wrapper.py:42-67`，`zero_mean=True` 的 SI-SDR、
  真 SDR），但**仍然没有入口调用它**（只保留 API）；`splitwrapper.py:41-42` 仍是
  `PITLossWrapper(pairwise_neg_snr)` 的 legacy 语义：列名 `one_snr/two_snr` 是**真 SNR（正号）不是 SDR**，
  逐句与 avg 同号（实测 `one_snr 13.839`、avg 同行）；它只用于 3 源拆分，主评测不用。

### 2.6 DNSMOS 口径（speechmos 版本、模型 sha、9.01 s/1 s、短音频循环拼接）

`[代码]` 实现：`tools/repro/phase2/eval_dnsmos.py:35-36`（16 kHz、`model_type="dnsmos"`）、`:55-66`（打印模型
sha256）、`:68-91`（`score_array`：峰值 >1 时按峰值归一化，调用 `speechmos.dnsmos.run`）、`:93-110`（读 wav、
stereo 取均值、非 16k 可选重采样）。

`[验证]` 本机 `sha256sum` 与 P2E 记录完全一致：

| 模型文件 | bytes | sha256 | 用途 |
|---|---|---|---|
| `dnsmos_models/sig_bak_ovr.onnx` | 1,157,965 | `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` | **P.835 主模型**（SIG/BAK/OVRL） |
| `dnsmos_models/model_v8.onnx` | 224,860 | `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91` | P.808 |
| `dnsmos_models/sig.onnx` / `bak_ovr.onnx` | 742,203 / 742,375 | `d2fbdb29…` / `5f335c90…` | AEC 相关，未用 |

`[代码]` `speechmos/dnsmos.py`（site-packages，非仓库）的推理口径：
`INPUT_LENGTH=9.01`、`hop_len_samples=fs=1 s`，逐窗 9.01 s、窗间 1 s、对窗结果**取均值**；输入限定
`[-1,1]`、16 kHz；**音频短于 9.01 s 时 `while len(audio)<len_samples: audio=np.append(audio, audio)` 循环拼接**
（即 2 s clip 被重复拼到 16 s 再取 7 个窗）。

`[验证]` 口径敏感性实测（本次会话，合成谐波，与 `eval_dnsmos.py:122-130` 的 `synthetic_cases` 同构）：

```
2.00 s 合成音: SIG=1.4956 BAK=1.2543 OVRL=1.3370 P808=2.1819
  循环拼接: 32000 样本(2.00 s 内容) -> 256000 样本(16.00 s) -> 窗数 num_hops=7
1.00 s 合成音: OVRL=1.3607（与 2 s 差 +0.024；拼接后同样 16 s/7 窗）
9.01 s 合成音: OVRL=1.2189（单窗，与 2 s 差 -0.118）
与 P2E-dnsmos-synthetic.json 的 synthetic_harmonic 对拍: SIG 1.4956 vs 1.4959、
BAK 1.2543 vs 1.2545、OVRL 1.3370 vs 1.3371、P808 2.1819 vs 2.1821（差 ≤3e-4，float32/64 舍入）
```

⇒ **2 s clip 的 DNSMOS 不是“官方 9 s 窗”口径**：它是在“重复拼接的伪 16 s 信号”上取 7 个 9.01 s 窗的均值；
短音频被周期性重复（1 s 与 2 s 都拼到 16 s、7 窗，但重复周期不同），因此**结果对输入长度敏感**。这解释了为什么 `P2E-report.md:93` 的
LRS2 2 s 冒烟值（OVRL 1.52/1.55）与 RealSSA-Real 长段（OVRL 2.79）不可横向比较。

`[代码]` RealSSA-Real 的 DNSMOS 协议（`eval_realssa_real.py`）：native stereo `channel=mean|0|1`
（`:228-240`，默认 mean）；`align_tail=True` 时 `T=min(mouth_frames, floor(audio_samples/640))`，只保留前
`T*640` 样本、最多丢 ~70 ms（`:243-305`, 默认 `:549-551`）；分离用窗口化推理 **2 s 窗 / 0.5 s hop / 80 ms
guard**（`:381-437`，避免长录音 OOM），DNSMOS 对**拼接后的整段估计**计算（`:646`），并可选 `--baseline`
对 mixture 同段计算（`:649`）。

`[文档]` RealSSA-Real 54/54 段（channel=mean, align_tail=True, 窗口化推理）结果：
SIG **3.2209** / BAK **3.6921** / OVRL **2.7901** / P808 **3.3285**（mixture 基线 OVRL 1.7599 / P808 2.6765；
`docs/REPRODUCTION_PHASE2_REPORT.md:98-107`）。论文 “Real-world DNSMOS 3.276”（`README.md:211` 表）
**只有 P808 口径接近（3.3285，+0.05）**；若论文是 OVRL 口径则差 0.49 → DNSMOS 列口径仍未确定（见 §6）。

### 2.7 P2-F “独立实现 vs core.py 最大差 4.8e-8 dB” 抽验

`[验证]` 两级抽验（CPU，无 GPU、无重跑评测）：

1) 用 `docs/verification/phase2/P2F-results.json` 的 `metrics.per_item`（N=50）**重算** `|my − core|`：

```
si_sdr   : max 1.43e-08
si_sdr_i : max 4.82e-08   ← 报告主数字的来源
sdr      : max 2.96e-12
sdr_i    : max 3.05e-12
全局最大 = 4.823e-08 dB，与 JSON 记录 maxabs_my_vs_core_db=4.823e-08 一致 = True
```

2) 用真实数据**复现其中一小步**：读 `/root/tmp/lrs2_eval_tt_full/tt` 前 3 条 mix/target，
用 numpy 独立公式算 `SI-SDR(mix, s1)`，与 `core.si_sdr(mix, s1)` 比 → 差 `3.16e-8 / 1.26e-8 / 8.54e-9 dB`；
再用 P2F 的 `si_sdr − si_sdr_i` 反推基线 → 与重算的 `core.si_sdr(mix,s1)` **差 0.00e+00**（3/3 条）。
⇒ “4.8e-8 dB 量级”可信（它对应 `si_sdr_i` 这一列浮点累积)，且 P2-F 的 CSV/基线算术自洽。

### 2.8 论文数字与我们的数字（同一口径下）

| 数据集 / 权重 | SI-SNRi | SDRi | 论文 | 差（SI-SNRi / SDRi） | 来源 |
|---|---|---|---|---|---|
| **RealSSA-Sim test 全量 9,169** / 官方 `falcon.pt` | **7.774** | **8.479** | 7.7 / 8.2 | **+0.07 / +0.28** | `docs/REPRODUCTION_PHASE2_REPORT.md:120-128` |
| RealSSA-Sim test 前 1000（快速估计） | 7.935 | 8.754 | 7.7 / 8.2 | +0.24 / +0.55 | 同上 `:110-116` |
| **LRS2-2Mix test 全量 3,000** / 官方 `falcon.pt`（跨语料） | 7.029 | 7.713 | 14.1 / 14.7 | −7.07 / −6.99 | `docs/REPRODUCTION_PHASE1_REPORT.md:101-105` |
| LRS2-2Mix test 20 条锚点 | 7.911 | 8.547 | — | — | `P2A-anchor-first20.json:anchors_first20`（与 Phase-1 7.911/8.547 一致） |
| LRS2-2Mix 训练复现（130 epoch，s2-near 微调） | 8.858 | 9.270 | 14.1 / 14.7 | −5.24 / −5.43 | `docs/REPRODUCTION_PHASE2_REPORT.md:58-67` |

`[验证]` 论文 7.7/8.2 的“可复算性”只在 RealSSA-Sim 成立（同分布 ckpt）；它给出两个独立结论：
(a) 我们的 SI-SDR/SDR 定义与论文一致到 +0.07 dB（SI-SNRi）；(b) 指标口径变体最多 ±0.4 dB（§2.9），
SDRi 的 +0.28 落在实现差异内（filter length、BSS-Eval 版本）。
因此 **14.1/14.7 的 7 dB 缺口不能用“指标定义”解释**（与 P2-A/P2-F 结论一致），只能归因于训练数据/ckpt。

### 2.9 指标/协议变体的量化（P2-A 与 P2-F 的原始数字）

`[验证]`（P2-A，200 条 LRS2 子集，官方权重；`docs/verification/phase2/P2A-small-200.log:61-75`）：

| 变体 | SI-SNRi | Δ vs 基线 | SDRi | Δ vs 基线 |
|---|---|---|---|---|
| 基线（SI-SDR 去均值 + SDR 512-tap） | 6.955 ± 3.134 | — | 7.622 ± 2.800 | — |
| SI-SDR `zero_mean=False` | 6.939 | −0.016 | — | — |
| plain SNR improvement（尺度相关） | 7.295 | **+0.340** | — | — |
| SDR filter 512 / 256 / 128 / 64 | — | — | 7.622 / 7.537 / 7.410 / 7.270 | −0.085 / −0.212 / −0.352 |
| 2×1 s 段平均 / 4×0.5 s 段平均 | 7.101 / 7.237 | +0.146 / +0.282 | — | — |
| target 能量缩放 / RMS 缩放 | 6.955 / 6.955 | ±0.000 | — | — |
| target=s2（错误目标） | −16.998 | −23.95 | — | — |
| best-of-s1/s2（PIT 上界） | 6.976 | +0.021 | — | — |

`[验证]`（P2-F，N=50，独立实现；`P2F-results.json:variants.metric_variants`）：
`SI-SDR zero_mean=False −0.000484`；`SDR 512→2048 +0.257`；`change_sign=True` 恰为相反数；
8 kHz 口径 `SI-SDR +0.111 / SDR +0.211`；`SI-SDR 尺度不变 1.3e-8`；估计/目标互相关最优 lag 恒 0。

⇒ **所有“合理”口径变体合计 ≤0.4 dB**；能改变 7 dB 的只有数据/权重/训练，不是指标。

---

## 3 bug 修复状态表（旧文档 P0/B 编号 → 当前代码）

| # | 问题（旧文档编号） | 旧行为（file:line @ a7af472） | 当前状态（file:line @ 2beec38） | 证据 | 结论 |
|---|---|---|---|---|---|
| 1 | CSV `sdr/sdr_i` 实为 −SNR/SNRi（B8/P0-2） | `allwrapper.py:27,38-46` 用 `pairwise_neg_snr` | `allwrapper.py:72-75` → `core.sdr`（fast_bss_eval 512-tap） | old −22.139 vs new +22.208 | ✅ **已修** |
| 2 | 逐句行与 avg/std 行符号相反（B8） | `allwrapper.py:62` 写 `sdr.item()`，`:72` 累加 `-sdr.item()` | `allwrapper.py:95-105` 行/累加同源；`107-140` nan 均值 | 逐句==avg（差<1e-9） | ✅ **已修** |
| 3 | PESQ 实参传反（B9） | `allwrapper.py:50` `pesq(estimate, clean)` | `allwrapper.py:78` `core.pesq_metric(clean, est)`；`core.py:203` `_pesq(ref, deg)` | 1.03 vs 2.73（合成）；1.1196 vs 1.1830（真实） | ✅ **已修** |
| 4 | `MetricsTracker` SI-SDR 用 `zero_mean=False` | 旧 `wrapper.py:30,33` | `wrapper.py:47-49` 走 `core.si_sdr`（zero_mean=True） | P2-A `zero_mean` 差 −0.016 dB | ✅ **已修** |
| 5 | `torch_mir_eval` 死依赖导致 import 失败 | 旧 `wrapper.py:6` | 已删；`core.py:49-62` 三个后端 `try/except` 可选 | `import look2hear.metrics` OK | ✅ **已修** |
| 6 | PESQ/STOI 在 `n_src>1` 时崩溃 | 旧 `allwrapper.py:50-58` 直接 `squeeze(0)` | `allwrapper.py:77-84`：`n_src!=1` → warn + nan | 代码 | ✅ **已修** |
| 7 | 多卡合并把每 rank 的 avg/std 当样本再平均（B10） | 旧 `audio_visual_test.py:291-307` | `audio_visual_test.py:476-486` 剔除 avg/std 后重算 | 代码 | ✅ **已修** |
| 8 | `_find_best_ckpt` epoch 字符串排序（A38） | 旧 `audio_visual_test.py:103-106` `sorted()[−1]` | `:238-246` 正则取 epoch 整数后 `max` | 代码 | ✅ **已修** |
| 9 | audio 权重 `strict=False` 静默丢键（B11 风险） | 旧 `:139` `strict=False` | `:301` `strict=True` + 可操作报错 | 代码 | ✅ **已修** |
| 10 | 训练 conf 被 `Separator.pop` 污染（B11/Q3） | 旧 `audio_train.py:54-58` 建模后才 dump | `audio_train.py:245-247` 建模前 `deepcopy` 落盘 | 代码 | ✅ **已修** |
| 11 | 评测无 CPU 回退（B6） | 旧 `:179-180` 强制 `cuda.set_device` | `:446-460` `_run_workers` 无 CUDA 时 `cpu` 单进程 | 代码 | ✅ **已修** |
| 12 | `${VAR}` 占位符不展开（B3） | 旧无展开 | `audio_visual_test.py:107-134` `_expand_env_vars` + 未解析提示 | 代码 | ✅ **已修** |
| 13 | 发布权重/裸 `state_dict` 加载（B5/Q1） | 旧 `:119-120` 硬要求 `ckpt["state_dict"]` | `:254-277` `_extract_state_dict` 兼容裸 dict；`weights_only=True` 优先 | Phase-1 真实权重验证 | ✅ **已修** |
| 14 | `SISNRi()` 不可实例化 | `sisnri.py:6` `super(Loss, self)` NameError | 未修（仍是死代码） | `SISNRi()` → NameError | ❌ **未修（死代码）** |
| 15 | `new_PITLossWrapper` 不做 PIT 归约 | `pit_wrapper.py:75` 直接返回矩阵 | 未修（实验性，无调用者） | 返回 `[B,S,S]` | ❌ **未修（死代码）** |
| 16 | `NASSContrastiveLoss` 不在 `__all__` | `losses/__init__.py:21` import、`:23-44` 缺它 | 未修（训练侧直接 from 模块 import，不影响运行） | 代码 | ❌ 未修（低危） |
| 17 | `verify_metrics_04.py` 已过期 | — | 脚本按“修复前 + stub 环境”写；真实环境 `torch.allclose(float64,float)` 崩、且叙述的行/avg 符号结论已不适用 | exit=1 `RuntimeError` | ⚠️ **脚本需重写**（本任务不改他人脚本） |
| 18 | `SingleSrc/MultiSrcNegSDR` 忽略 `EPS` 入参 | `matrix.py:246,292` 写死 1e-8 | 未变 | 代码 | ❌ 未修（影响 <1e-6 dB） |
| 19 | MDCL 分支不可达 / 对比子类吞 `near_audio` | `Falcon.py:1482-1486`、`Falcon_Contrastive.py:94-95` | 仅加了 `_mdcl_loss` 别名（`Falcon.py:1467`）；仍不可达 | §2.3 | ⚠️ **半修**（不影响 NASS 主线） |
| 20 | `splitwrapper` 仍是 legacy SNR（列名叫 snr，不是 SDR） | `splitwrapper.py:41-42` | 未变（无入口调用） | 实测正值同号 | ⚠️ 保留（legacy） |

> 结论：**主评测路径（audio_visual_test → ALLMetricsTracker → core）的 1–13 全部已修**；未修的 14–20
> 都是死代码/legacy/脚本卫生问题，不影响当前复现结论。

---

## 4 与旧文档（04-losses-metrics-video.md）及 P2-A/P2-F 的差异

### 4.1 对旧文档逐条判定

| 旧文档位置/断言 | 判定 | 依据（当前代码/实测） |
|---|---|---|
| §2.1.2 `PairwiseNegSDR` 数学（三种 `sdr_type`、`zero_mean`、`EPS`） | ✅ 确认 | 未变；手算 diff=0；`zero_mean` 3.05e-5 vs 47.815 dB |
| §2.1.3 `PITLossWrapper` 6 种 `pit_from`、`threshold_byloss` 语义 | ✅ 确认 | 未变；`22.393810 / 33.590714 / −6.194741 vs −36.891914` 复跑一致 |
| §2.1.3 `new_PITLossWrapper` 未完成、不做 PIT | ✅ 确认 | 未变（TypeError / 原样返回矩阵） |
| §2.1.4 MixIT 14 个有序二划分、仓库未用 | ✅ 确认 | `17.777866`（手算一致）/ `19.077686` |
| §2.1.5 NASS 公式/K=1 无 softmax/分母恒为 B/β 调度 | ✅ 确认 | 未变；β=0/1/5 对拍 ≤5.96e-8；新增“K=1 ⇒ β 失效”这一后果（§2.2） |
| §2.1.6 `SISNRi()` 抛 `NameError`、数学正确 | ✅ 确认 | 未变；`__new__` 绕过构造对拍 1.9e-6 |
| §2.2.1 `MetricsTracker` 是“唯一用真 SDR 的实现、zero_mean=False、torch_mir_eval 死依赖” | ❌ **推翻（已重写）** | 现在它也用 `core`（真 SDR + zero_mean=True），且无 torch_mir_eval 依赖 |
| §2.2.2 `ALLMetricsTracker` 的 `sdr` 实为 −SNR、行/avg 反号、PESQ 传反 | ❌ **推翻（已修）** | 见 §2.4；`allwrapper.py:63-105` 全部重写 |
| §2.2.4 `metrics/__init__` 因 import wrapper 而强制依赖 `torch_mir_eval/fast_bss_eval` | ⚠️ **修正** | `core.py:49-62` 后端可选；无包时对应指标返回 nan 而非 import 崩溃 |
| §2.4-B `_load_models_from_ckpt` 音频 `strict=False`、裸 state_dict 不兼容 | ⚠️ **修正** | 现在 `strict=True`（`:301`）、裸 state_dict 兼容（`:254-277`） |
| §2.4-C 评测“无 DataLoader、整句、batch=1、强制 CUDA” | ⚠️ **部分修正** | 前三项不变；CUDA 现在可 `--gpus cpu` 回退（`:446-460`） |
| §2.4-D 合并 CSV 把 avg/std 当样本（B10） | ❌ **推翻（已修）** | `:476-486` 剔除后重算 |
| §3.4 B8/B9（指标 bug） | ❌ **推翻（已修）** | 见 §2.4 |
| §4.1 “训练 SNR、验证 SI-SDR、数值不可比” | ✅ 确认并**量化** | `est=2·s`：训练 SNR 损失 0.000 dB vs SI-SDR 116.191 dB |
| §4.4 NASS “若全 batch 无负样本只剩正项” | ✅ 确认 | `K=0` 实测损失 = 正项 = 0.7594；并指出 RealSSA 正是这种情况 |
| §4.6-8 “整句 vs 2 s 窗口口径不可横比” | ✅ 确认（对本域） | LRS2/RealSSA-Sim 评测是整句 2 s clip；RealSSA-Real 是窗口化推理 + 整段 DNSMOS |
| §5.2 CSV 契约（全部正 dB、avg/std） | ✅ 确认 | 逐句==avg；`est=mix` → i=0 |
| §6 A23–A25（sdr=−SNR、符号反、PESQ 反） | ❌ **推翻（已修）** | 同上 |
| §6 A27（MetricsTracker zero_mean=False / 真 SDR） | ⚠️ **修正** | zero_mean 已改 True，SDR 口径保留 |
| §7 Q5（论文 SDRi 是哪条路径） | ⚠️ **部分回答** | 现在有 fast_bss_eval 真 SDR；RealSSA-Sim 复现 8.479 vs 8.2（+0.28）→ 论文 SDRi 大概率同族（BSS-Eval-v4/512-tap），但仍非逐位确认 |
| §8.1 用 `/root/.venvs/falcon` + stub 跑验证 | ⚠️ **环境过期** | 本任务用 conda `falcon` 真实包；`verify_metrics_04.py` 因此失效（§3 #17） |
| 旧文档**未覆盖**：DNSMOS、RealSSA 评测、论文数字对账、P2-A/P2-F | ➕ **新增** | §2.6/§2.8/§2.9 |

### 4.2 与 P2-A / P2-F 的关系（不是重复，而是交叉验证）

- `P2A-report.md` / `P2A-small-200.json`：200 条 LRS2 子集、14 个协议变体、`gap_analysis.py` 自写 SI-SDR
  （`gap_analysis.py:190-236`）与直接 `fast_bss_eval`（`:225-236`）；本文件引用其 §3.2 表并补上“为什么
  SI-SDRi 的 +0.34 plain-SNR 变体不是论文口径”（论文 SI-SNRi 与我们的 7.774 对得上，plain-SNR 会变成 ~8.1，
  反而偏离 7.7）。
- `P2F-report.md` / `P2F-results.json`：独立 NumPy SI-SDR + `fast_bss_eval.sdr`（`protocol_check.py:70-101`），
  与 `core.py` 对拍 4.8e-8 dB；本文件 §2.7 做了**两级抽验**（JSON 重算 + 真实数据基线复算）确认可信。
- 两份报告与本文件的**分歧点**：都没有讨论“训练损失 ≠ 评测指标”（同义不同名）、“NASS K=0/K=1 的 β 失效”、
  “DNSMOS 短音频循环拼接”。这三点是本次新增的、对**训练复现**有直接影响的结论。
- 一致性：P2-A/P2-F 的“协议/指标 ≤0.4 dB”与本文件 §2.9 同向同量级；RealSSA-Sim 7.774/8.479 与
  P2-F 对 `core.py` 的可信性结论互相支撑（同一指标实现同时闭合 7.7/8.2 与 7.03/7.71 两条数据）。

---

## 5 对复现的影响与建议

### 5.1 论文 7.7/8.2 与 14.1/14.7 用哪套指标定义最可信？

**答：用当前 `look2hear/metrics/core.py` 这一套（SI-SDR 去均值 + PIT；SDR=fast_bss_eval 512-tap；
PESQ wb；STOI ext=False；整句推理（LRS2 为 2 s clip，RealSSA-Sim 为对齐后的原生长度）；channel=mean）。** 理由：

1. **同分布锚点闭合**：RealSSA-Sim test 9,169 条（发布权重）SI-SNRi **7.774** vs 论文 **7.7**（+0.07 dB），
   这是唯一“数据+权重+指标”三者都在手且能对上的锚点；SDRi +0.28 dB 落在口径变体范围内
   （P2-A：512→64 tap 差 −0.35；P2-F：512→2048 差 +0.26）。
2. **独立实现对拍**：P2-F 独立实现 vs core 4.8e-8 dB；本次再抽验通过（§2.7）。定义错误不可能产生这种一致性。
3. **协议变体上界 ≤0.4 dB**：即使把口径全换成最有利的变体，LRS2 的 7.03 也只能到 ~7.4，离 14.1 差 ~6.7 dB；
   RealSSA-Sim 的 7.774 也不会变成 14（数据集不同，不能这样比，但说明口径不是数量级因素）。
4. **14.1/14.7 的不可复现原因是 ckpt**：发布 `falcon.pt` 训练集为 LRS3（`Experiments/checkpoint/Falcon/
   conf.yml:99-102`），LRS2 列是论文按 benchmark 训练（`README.md:228` “under our training protocol”）；
   我们 LRS2 自训 130 epoch 只到 8.858/9.270（单卡 bs4，对比论文 10×A800/bs40/1000 epoch）。
   ⇒ 比较 14.1/14.7 必须用 **LRS2-trained ckpt + 同一 core.py 口径**；否则应标“跨语料结果”。

### 5.2 训练损失与评测指标“同名不同义”的处置

1. **不要用 `train_pit_loss`（−SNR）与 `val_pit_sisnr`（= −val_loss = SI-SDR(est)）互相比较**，更不要与论文 SI-SNRi 比：
   - 训练：`pairwise_neg_snr` + `threshold_byloss=True`（`configs/Falcon.yml:116-120`）；SNR 对幅度敏感
     （2× 完美估计 = 0 dB），且 SNR>30 dB 的样本会被阈值丢掉。
   - 验证：`pairwise_neg_sisdr` + `threshold_byloss=False`（`:121-126`），`val_pit_sisnr=−val_loss`
     （`av_litmodule.py:163`）= **SI-SDR(est)**，不含 mixture 基线。
   - 复现做早停/选 ckpt 时，建议额外记录**评测口径的 SI-SNRi/SDRi**（或至少记录 `SI-SDR(mix)` 基线），
     否则 ckpt 选择目标（max SI-SDR）≠ 论文报告目标（max SI-SNRi）。LRS2 实测：val 13.391/13.762 vs
     SI-SNRi 8.858（基线 4.9 dB）。
2. **若要严格对齐论文训练目标**，先向作者确认训练用的是 SNR 还是 SI-SDR（仓库 config 是 SNR，但论文未公开
   训练细节）；两种目标对“幅度校准”的行为不同，会影响早期训练与最终 SDR/SI-SDR 的差（LRS2 实测 +0.79 dB）。
3. **报告必须同时给 4 个数 + 基线**：SI-SDR、SI-SDRi、SDR、SDRi（例：LRS2 官方权重 12.023/7.029/12.813/7.713），
   否则读者无法判断“SDRi 是 filtered SDR 还是别的实现”。

### 5.3 对比损失负样本缺失对训练复现的影响

| 场景 | near 来源 | K | β 是否生效 | 实测影响 |
|---|---|---|---|---|
| 论文原始训练（LRS3 等） | near.json 的 near-field 干扰源 | 可变（≥1，可能 ≥2） | K≥2 时生效（softmax） | 未知（数据不可得） |
| **RealSSA-Sim 训练**（若要复现） | **无 near 波形**（README:87；`realssa_to_falcon.py:482-489` 回退 mix 哨兵，dataset 跳过） | **0** | **不生效** | 对比损失只剩正项 → “远离干扰”的约束完全缺失 |
| LRS2-2Mix 训练（本次 130 epoch 微调） | 用 s2 说话人当 near（`configs/LRS2_2Mix_near_s2.yml`；与 Phase-2 报告 §5.3 的 8 卡“双向 target”论文配方数据是两套） | **1** | **不生效**（K=1 跳过 softmax） | +0.24/+0.23 dB（8.618/9.042 → 8.858/9.270） |

结论与建议：
- **RealSSA 上无法忠实复现论文的对比损失负样本设置**：README 明确说 “The original negative-speaker
  contrastive training setup requires those source signals”，而 RealSSA 元数据里的源路径不可下载
  （`README.md:87`）。若在 RealSSA-Sim 上训练，NASS 实际退化为“正样本对齐”，**这必须写进复现差异声明**。
  好消息是：官方发布权重本身就是 RealSSA 训练分布，推理侧 7.774/8.479 已复现，训练侧差异暂不影响该结果。
- **K=1 时 β 退火是 no-op**（`nass_contrastive_loss.py:68-69`）：LRS2 s2-near 实验里 β 0→1/200 只是配置装饰，
  不能把增益归因于“hard negative 加权”；增益来自正项 + 单个硬负样本的 margin 项。若想做 β 消融，必须构造
  K≥2 的负样本（如多条 near 或 batch 内负样本），否则消融无意义。
- **每 batch 的 K 分布要记录**（`av_litmodule_nass_v2.py:113-118,182-183` 已按 0/1/2/3/4+ 分桶 log）：
  RealSSA 全 0、LRS2 全 1，这是复现报告里必须声明的两个极端。
- 负样本质量：LRS2 用 s2 作为 near 是“真干扰说话人”，语义上合理，但与论文的 near-field（混响近场）分布不同；
  RealSSA 若要构造替代负样本（batch 内其他样本的 target 等），只能算消融，不能声称复现原方法。

### 5.4 DNSMOS 不可复算性/版本敏感性

- **版本敏感**：结果依赖 `speechmos==0.0.1.1` 内置的 DNSMOS ONNX（P.835 `sig_bak_ovr.onnx` 1,157,965 B
  sha256 `269fbe…`；P.808 `model_v8.onnx` 224,860 B sha256 `924648…`）。换 speechmos 版本/换 torchmetrics
  `DeepNoiseSuppressionMeanOpinionScore`（会另下模型）都会改变数值。**报告必须写版本 + sha256 + model_type**
  （`eval_dnsmos.py:55-66` 已自动记录；`P2E-dnsmos-synthetic.json` 已有一份）。
- **短音频口径不可比**：2 s clip 会被 speechmos 循环拼接到 16 s、取 7 个 9.01 s 窗均值（本次实测
  窗数=7；1 s/2 s 拼接后同为 16 s/7 窗，9.01 s 单窗 OVRL 与 2 s 差 −0.118）。因此 `P2E-dnsmos-lrs2-smoke.csv` 的 2 s OVRL（1.52/1.55）
  不能与 RealSSA-Real 长段（2.79）或论文 3.276 直接比；**能比的是同长度/同拼接规则的数值**。
- **论文 3.276 的列名未定**：我们 RealSSA-Real（54 段、channel=mean、align_tail、窗口化推理）
  P808 **3.3285**（+0.05）、OVRL **2.7901**（−0.49）、SIG 3.2209、BAK 3.6921。
  `[推断]` 论文 “DNSMOS 3.276” 更像 **P808**（或另一种 OVRL 模型），必须在论文/作者确认前**同时报 OVRL 与 P808**。
- **channel / align_tail / windowed-vs-whole 三个口径必须一起报**：RealSSA-Real native stereo，
  `channel=mean` 是默认但论文未声明；`align_tail=True` 会丢最多 ~70 ms 尾部；分离是 2 s 窗/0.5 s hop/80 ms
  guard 的窗口化推理（整段前向 OOM），而 LRS2/RealSSA-Sim 评测是整句 2 s clip（无窗口拼接）。
  P2-B 已测得 LRS2 上窗口化比整句低 1.9–2.3 dB（SI-SNRi/SDRi）→ DNSMOS 的“窗口化”也会带入类似偏差，
  不能把 RealSSA-Real 的 DNSMOS 与“整段推理”结果混表。
- 建议：给 `eval_dnsmos.py` 增加 `--min-duration` 警告（<9.01 s 时显式记录“tiled/repeated”标志），
  summary 里记录 `n_windows`；RealSSA-Real 报告里固定 `channel=mean`（另跑 0/1 作敏感性）。

### 5.5 评测执行层面

- 用 `configs/Falcon.yml` + `--data_dir <index>` + `--ckpt_path` 的显式组合（避免旧 `Experiments/
  checkpoint/Falcon/conf.yml` 的 `SpeakerContrastiveV2` 类名/`${DATA_ROOT}` 占位符陷阱；
  `audio_visual_test.py:289-296` 现在会给出清晰 `AttributeError`）。
- 接收别人的 CSV 时先验证：`si-snr − si-snr-i == SI-SDR(mix)`、`sdr − sdr_i == SDR(mix)`（P2-F 已用此
  反推基线）；若不等或符号为负，说明是旧版/别的口径。
- 不要把 `metrics_combined.csv` 的 avg/std 行再平均（本文件 §3 #7 已修，但外部脚本可能仍在这样做）。

---

## 6 未解问题

| Q | 问题 | 现状/证据 | 需要谁 |
|---|---|---|---|
| Q1 | 论文 “Real-world DNSMOS 3.276” 是 **OVRL** 还是 **P808**（还是别的 DNSMOS 实现）？ | 我们 P808 3.3285（+0.05）、OVRL 2.7901（−0.49）；`README.md:211` 与 Swift-Net 值相同（3.276），无法区分 | 作者/论文 |
| Q2 | 论文 SDRi 的实现（BSS-Eval v3/v4、filter length、是否 `change_sign`）| 我们用 fast_bss_eval 512-tap；RealSSA-Sim SDRi +0.28、LRS2 变体 512→64 差 −0.35 | 作者/论文 |
| Q3 | 论文训练损失是 SNR 还是 SI-SDR？`configs/Falcon.yml:117` 是 `pairwise_neg_snr`，但论文/README 未写 | 代码是 SNR；`threshold_byloss=True` 还会丢 >30 dB 样本 | 作者/论文 |
| Q4 | 论文 PESQ 是 wideband 还是 narrowband、是否用 `extended` STOI | 我们用 wb + `extended=False`（`core.py:196-217`）；PESQ 传反会差 0.06~1.69 MOS | 作者/论文 |
| Q5 | 论文 RealSSA-Sim 的 channel 选择（4ch→mean？ch0？）与 align 规则 | 我们用 mean + min 对齐；`P2E-report.md` §4 也要求 ch mean/0 两套都报，目前只有 mean 结果 | 数据卡/作者 |
| Q6 | 论文 LRS2-2Mix 的训练配方与对应 ckpt | 发布 ckpt=LRS3（`conf.yml:99-102`）；GitHub issue #1 已问；P2-D 训练中 | 作者/P2-D |
| Q7 | 原始训练 near.json 的 K 分布（决定 NASS 负项与 β 是否真的起作用）| 发布权重是推理产物，无法反推；LRS3 数据不可得 | 作者/数据 |
| Q8 | speechmos 内置 ONNX 与“论文用 DNSMOS 官方实现”的数值一致性 | 只有 sha256 记录（§2.6）；不同实现/版本可差 0.1~0.3 MOS | 作者/复算 |
| Q9 | MDCL 是否是论文消融的一部分 | 代码/配置/发布快照都不走 MDCL（§2.3）；论文未公开 | 作者/论文 |
| Q10 | `docs/verification/scripts/verify_metrics_04.py` 的维护归属（已过期，本任务无权改） | 真实环境 exit=1；其断言基于修复前代码 | Lead/Phase-2 QA |

---

## 7 验证记录

### 7.1 环境自检

```bash
$ENV/bin/python -V                                   # Python 3.11.17
$ENV/bin/python -c "import torch;print(torch.__version__)"   # 2.8.0+cu128
$ENV/bin/python -c "import fast_bss_eval,pesq,pystoi,speechmos;print('metric backends OK')"
# 实测版本：fast-bss-eval 0.1.4 / pesq 0.0.4 / pystoi 0.4.1 / speechmos 0.0.1.1 / onnxruntime 1.31.0
PYTHONPATH=$PWD $ENV -c "import look2hear.models, look2hear.metrics, look2hear.losses, audio_visual_test; print('imports OK')"
# -> imports OK（timm 只有 FutureWarning）
```

### 7.2 本次实际运行（全部 CPU，未用 GPU，未联网）

| # | 命令 | 结果 | 耗时 |
|---|---|---|---|
| ① | `PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_losses_04.py` | 全 PASS（§2.1/§2.2 的数字全部来自该输出） | <10 s |
| ② | `PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_metrics_04.py` | **exit=1**：`RuntimeError: Double did not match Float`（脚本 stub 与真实 float64 实现冲突）；且其打印的“逐句/avg 反号、PESQ 第 1 实参是 estimate”是修复前叙事，与当前代码不符 → 判定脚本过期 | <5 s |
| ③ | `PYTHONPATH=$PWD $ENV /tmp/lm/deepdive_verify.py` | 9 组探针：SI-SDR 手算 1.83e-10 dB、SI-SNRi 恒等、PIT 20.895 vs −48.247、训练 SNR vs SI-SDR（0.000 vs 116.191 dB）、SDR 512-tap（22.1946）、PESQ 正/反（1.0297/2.7158）、tracker 行==avg、P2-F 4.823e-8 重算、DNSMOS 2 s→7 窗 | <30 s（含 DNSMOS 2 次推理） |
| ④ | old-vs-new tracker：`git show a7af472:look2hear/metrics/allwrapper.py` + 同一合成样本 | 旧：逐句 `sdr=−22.139` / avg `+22.139` / PESQ 2.73；新：`+22.208` / `+22.208` / 1.03 | <10 s |
| ⑤ | P2-F 真实数据小步复现：`/root/tmp/lrs2_eval_tt_full/tt` 前 3 条 mix/target | numpy SI-SDR vs `core.si_sdr` 3.16e-8/1.26e-8/8.54e-9；`si_sdr−si_sdr_i` 反推基线差 0.00e+00 | <5 s |
| ⑥ | `sha256sum .../speechmos/dnsmos_models/*.onnx` | 与 `P2E-dnsmos-synthetic.json` 四条 sha256 完全一致 | <1 s |
| ⑦ | `$ENV -m pytest tools/repro/tests/test_metrics.py -q` | **8 passed in 4.35 s**（解析值 ±0.05 dB、SDR 对拍、PESQ/STOI 实参顺序、CSV 行/聚合一致） | 4.35 s |

> 未做（受任务约束/成本）：不重跑 `audio_visual_test.py` 全量评测、不重跑 P2-A/P2-F 的 200/3000 条 GPU 推理、
> 不下载任何模型/数据；P2-A/P2-F 的数字直接引用其 JSON/日志并做算术/小步抽验。

### 7.3 临时脚本（未写入仓库，“只写一份文档”约束）

`/tmp/lm/deepdive_verify.py` 的核心可复现片段（可直接在 repo 根运行）：

```python
import numpy as np, torch, sys
sys.path.insert(0, ".")
import look2hear.metrics.core as core            # 手算 SI-SDR
T, t = 16000, np.arange(16000) / 16000.0
ref = 0.3*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*440*t)
rng = np.random.default_rng(0); est = ref + 0.02*rng.standard_normal(T)
s = ref - ref.mean(); e = est - est.mean()
a = (e*s).sum()/((s*s).sum()+1e-8)
manual = 10*np.log10(((a*s)**2).sum()/(((e-a*s)**2).sum()+1e-8))
print(manual, core.si_sdr(est, ref))             # 22.1223949939 / 22.1223949941

# 训练损失（-SNR） vs SI-SDR：est = 2*ref
from look2hear.losses import PITLossWrapper, PairwiseNegSDR
tr = -PITLossWrapper(PairwiseNegSDR("snr"), pit_from="pw_mtx", threshold_byloss=True)(
        torch.tensor(2*ref)[None,None], torch.tensor(ref)[None,None]).item()
print(tr, core.si_sdr(2*ref, ref))               # 0.000 dB（SNR） vs 116.191 dB（SI-SDR）
```

`/tmp/lm/old_allwrapper.py` = `git show a7af472:look2hear/metrics/allwrapper.py` 仅把
`from ..losses import` 改成 `from look2hear.losses import`（其余原样），用于 §2.4 的 old-vs-new 对照。

### 7.4 与旧文档/P2 报告的证据对照说明

- 本文件所有 `[验证]` 数字都给出了命令与输出摘要；临时脚本未入库（遵守“只写一份文档”）。
- 旧文档 `docs/understanding/04-losses-metrics-video.md` 的 §2.1（损失数学）、§2.1.3（PIT）、§4.1（训练/验证
  口径不同）仍然成立，可直接沿用；§2.2（tracker）、§3.4 B8/B9、§4.6 P0-2、§6 A23–A25 已被 Phase-1 修复**推翻**，
  引用旧文档时必须以本文件 §2.4/§3 为准。
- P2-A/P2-F 报告与本文件结论同向：**指标/协议最多解释 0.4 dB；RealSSA-Sim 7.7/8.2 是可信锚点；LRS2 14.1/14.7
  缺的是 LRS2-trained ckpt**。本文件新增的三个训练侧结论（训练 SNR ≠ 评测 SI-SDRi、NASS K=0/K=1 与 β 失效、
  MDCL 死代码）应并入最终复现报告。
```

## 8. 2026-10-30 论文公开后的校正（arXiv:2610.32749）

- **DNSMOS 口径确定**：论文 “Real-world DNSMOS” 引用 **P.835**（非 P.808），单值 3.276 属 RealSSA-trained ckpt；
  同一官方 `falcon.pt` 上我们的 P.835 OVRL 2.7901 不可与之比（P808 3.3285 是巧合）。
- **训练损失**：论文 App D Table 16 明示 **L_recon = −SI-SNR**、λ_spk=1.0、margin 0.5、**τ 0→1 线性退火跨前 1000 epoch**、
  8×RTX3090、bs 4/GPU（global 32）、seed 42；仓库 `configs/Falcon.yml` 的 `pairwise_neg_snr` 与 β/200 epoch 属偏离配方，
  复现报告必须声明该差异。
- **RealSSA-Sim 评测通道**：论文用 **channel-1**（对应代码 `--channel 0`），我们此前用 4ch mean；
  通道差异量级待 ch0 复评给出（预计 ±0.1~0.3 dB，可能解释我们 +0.07/+0.28 的一部分）。
- **near 负样本**：论文 App A.2 说明 near 干扰由 LibriSpeech 源按 RIR 仿真生成，metadata/脚本承诺发布 →
  “发布包无 near 波形”不等于不可复现，而是需要按论文参数重建（成本高，列为 P3 的备选方案）。
