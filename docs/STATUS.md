# Falcon 复现状态总览（STATUS · 单一事实源）

> 最后更新：2026-10-31 · 分支 `main` · HEAD = `repro-freeze-20261031` 指向的提交（`git log -1`） · 环境 conda `falcon`（Py3.11 / torch 2.8.0+cu128 / 2×A800）
> 本文件是“当前结论 + 证据位置 + 待办”的唯一事实源；历史规划/报告仅作过程记录。
> 详细问题清单见 [`docs/understanding/THREE_ROUND_REVIEW_20261030.md`](../understanding/THREE_ROUND_REVIEW_20261030.md)；论文协议见 [`PAPER_ALIGNMENT_20261030.md`](../understanding/PAPER_ALIGNMENT_20261030.md)。

---

## 1. 一句话

**L1 推理/结构已完全复现；L2 评测两列有结论（RealSSA-Sim 主结果闭合，LRS2-2Mix 为跨语料）；L3/L4 未完成**：LRS2 论文级训练依赖 8 卡（配置已切到论文 Table 16），RealSSA-Sim 从零训练需先重建 near 负样本；LRS3-2Mix / VoxCeleb2-2Mix 数据未发布，不可复现。当前无阻塞性 P0（review 的 1 个数据 P0 已修）。

---

## 2. 仓库与环境

| 项 | 值 |
|---|---|
| 分支 / 远端 | `main`；`origin = nanless/Falcon-fork`，`upstream = JusperLee/Falcon`（已合并 439cbe9） |
| HEAD / tag | `repro-freeze-20261031`（tag 指向的提交；含本次 review 修复与证据） |
| 环境 | `/root/miniforge3/envs/falcon/bin/python`：Python 3.11.17、torch 2.8.0+cu128、pytorch-lightning 2.6.6、typeguard 2.13.3 |
| 依赖锁 | [`docs/verification/conda-env-lock.txt`](verification/conda-env-lock.txt)（115 行，无 `file://`；顶部含 cu128 两步重建命令） |
| 权重 | `/root/.cache/falcon_weights/`：`falcon.pt`（small，epoch 714，525+137 keys）、`falcon-big.pt`（big，epoch 715，517+137）、导出的 `frcnn_128_512.backbone.pth.tar` |
| 数据 | LRS2-2Mix 原始/转换：`/root/data/lrs2_2mix`（33G）、`falcon_lrs2_2mix`（24G，28k）、`falcon_lrs2_2mix_paper`（26G，40k/10k/6k）；RealSSA-Sim tt：`falcon_realssa_sim`（mean，30G）与 `falcon_realssa_sim_ch0`（ch0，30G），各 9,169 条；RealSSA-Real 12G；TOS 归档 Sim 521.5 GB / Real 12.84 GB |
| 训练产物 | `/root/tmp/p2d_train`（100 epoch，best ep99 13.2014）、`/root/tmp/p2d_near_train`（微调 ep124 13.3909）；sha256 记录见 `P2D-report.md` §8 |

---

## 3. 复现矩阵

| 层 / Benchmark | 论文 | 实测 | 结论 | 证据 |
|---|---|---|---|---|
| L1 参数 / MACs | 0.5 M / 10.6 G | 音频 537,697（生效 532,129；扣对比头 523,113）+ 冻结 FRCNN 11,186,688 = 11,724,385；MACs 10.611 G | ✅ 精确复现 | `docs/verification/scripts/verify_model_params.py` |
| L1 权重 / 一致性 | — | `falcon.pt` strict 零 missing；训练侧 vs Studio 同输入 max\|Δ\|=0 | ✅ | `verify_real_weights.py`、`verify_r3_studio_weights.py` |
| RealSSA-Sim SI-SNRi / SDRi（mean） | 7.7 / 8.2 | **7.774 / 8.479**（9,169 条，整段） | ✅ +0.07/+0.28 | `P2E-realssa-sim-{eval-official.log,metrics_combined.csv}` |
| RealSSA-Sim（ch0，论文 channel-1） | 7.7 / 8.2 | **7.977 / 8.452**（9,169 条） | ✅ +0.28/+0.25；通道影响 <0.3 dB | `P2E-realssa-sim-ch0-{eval-20261030.log,metrics_combined-20261030.csv}` |
| RealSSA-Real DNSMOS | 3.276（P.835，RealSSA-trained ckpt） | 官方 `falcon.pt`：P.835 **OVRL 2.7941** / P808 **3.3365**（full4，2026-10-31 修复窗口起点 bug 后重跑；full3 的 2.7901/3.3285 作废） | ❌ 不可比（论文 3.276 属 RealSSA-trained ckpt）；**非复现失败** | `DNSMOS-realssa-real-mean-full4-20261031/`、P2E-report §10 |
| LRS2-2Mix 官方权重（跨语料） | 14.1 / 14.7 | **7.029 / 7.713**（tt 3,000） | 跨语料基线，不是论文列 | `P2D-eval-falcon-full.csv` |
| LRS2-2Mix 自训 ep86（无 near） | — | 8.618 / 9.042（val 13.1182） | 单卡 100 epoch 中间快照 | `P2D-eval-epoch86.csv` |
| LRS2-2Mix 自训 ep99（best val） | — | **8.726 / 9.122**（val 13.2014） | 单卡 100 epoch 最佳 | `P2D-eval-epoch99.csv` |
| LRS2-2Mix 自训 ep124（+s2 near 对比微调） | — | **8.858 / 9.270**（val 13.3909） | 距论文 −5.24/−5.43 | `P2D-eval-epoch124-contrastive.csv` |
| LRS3-2Mix | 16.4 / 16.7 | 无数据 | ❌ 不可复现 | HF 仓库仅 README |
| VoxCeleb2-2Mix | 10.5 / 11.8 | 无数据 | ❌ 不可复现 | 同上 |
| 效率（3090） | 51.3 ms / 125.3 MB / 364.7 MB | 未测（A800 单窗 small 47.1 ms / big 56.5 ms） | ⚠️ 硬件不同 | Studio 后端报告 |

---

## 4. 口径与已知偏差（写报告必须带）

1. **“causal” = 窗口级因果**：`torch.stft(center=True)` + `cLNhw` 全窗 GN + AudioBlock 对称卷积（`causal=False`），单遍 lookahead 实测 **57 帧≈456 ms**，不是论文 App E 的 120 ms；Studio 用 2 s 窗 / 0.5 s hop / 80 ms guard 重算，跨窗无状态。
2. **DNSMOS**：论文用 **P.835**；3.276 属 **RealSSA-trained ckpt**，与官方 `falcon.pt` 不同 ckpt，不可直接比；报告必须同时给 OVRL 与 P808 并注明版本（speechmos 0.0.1.1 + 模型 sha256）。
3. **RealSSA-Sim 通道**：论文 channel-1（代码 `--channel 0`）；mean 与 ch0 差 +0.20/−0.03 dB（SI-SNRi/SDRi），两通道均在 ±0.3 内。
4. **训练配方**：论文 Table 16 = **−SI-SNR、λ_spk 1.0、margin 0.5、τ 0→1/1000 epoch、8×RTX3090、bs4/GPU（global 32）、seed 42**；仓库旧配置是 `pairwise_neg_snr` + β/200。v2 配置 `configs/LRS2_2Mix_Paper_v2.yml` 已按论文切换；历史 P2-D（8.618/8.726/8.858）明确标注为旧配方。
5. **near 负样本**：RealSSA 发布包无 near 波形（near.json 全 fallback，K=0，NASS 只剩正项）；LRS2 现用 s2 作 near（K=1，τ 退火不生效，增益 +0.24/+0.23 dB 不能归因于 τ）。论文 App A.2 给出可重建路径（LibriSpeech+RIR），尚未实施。
6. **参数口径**：引用时必须写“537,697（生效 532,129；扣对比头 523,113；+冻结 FRCNN = 11,724,385）”。

---

## 5. 验证基线（本轮收尾后回填最终值）

| 检查 | 期望 | 命令 |
|---|---|---|
| 复现工具测试 | **166 passed / 0 skipped / 0 failed**（2026-10-31 实测，136.5 s；Phase1 7 条验收断言已改为 fail-on-missing） | `PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q -rs` |
| Studio Python | **51 passed / 2 skipped**（2026-10-31） | `pytest studio/tests -q` |
| Studio Node | 8/8 | `for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done` |
| Space static | **3 passed**（build+i18n+运行时复用断言）；组装后 i18n 226 条 | `pytest space/tests/test_static.py -q`；`python space/build.py --out /tmp/falcon-space && node space/tests/check_i18n.mjs /tmp/falcon-space` |
| 模型数值不变 | strict 加载 + 537,697 + bit-identical | `docs/verification/scripts/verify_{real_weights,model_params}.py` |

---

## 6. 待办（按优先级；详细见三轮 review）

### 已完成（2026-10-31）
- [x] 数据构建 P0：`--copy-wavs` target/near 同名塌缩修复 + 标签一致性闸门
- [x] 评测 P1：RealSSA-Real 窗口首块起点 bug + 覆盖率断言 + 54 段 DNSMOS 重跑
- [x] 模型 P1：config 污染（deepcopy）、ckpt 自描述、FRCNN 严格加载
- [x] 训练：Table 16 v2 配置、8 卡脚本 DATA_ROOT/NPROC/global-batch 守卫、续训缺失退出
- [x] 证据：Phase1 skip 守卫修复、pid/0 字节清理、lock 可移植测试、ckpt sha256
- [x] Studio：`--checkpoint`、bootstrap/doctor 安全守卫、依赖版本对齐、Space 静态测试/CI、i18n 5 条、项目页 meta
- [x] 文档：本 STATUS、PLAN/GUIDE/PHASE2/README 过期正文、Dolphin.wav 误判、README 因果/ckpt 注记

### 仍开放
- [ ] **LRS2 8 卡论文配方训练**：`configs/LRS2_2Mix_Paper_v2.yml` + `run_lrs2_paper_volcano_8gpu.sh`（需 8 卡；8 卡约 33–50 h；2 卡 1000 ep 约 120 h，需实测）。验收：tt 3,000 SI-SNRi ≥13.6 / SDRi ≥14.2。
- [ ] **RealSSA-Sim 从零训练**：按论文 App A.2 重建 near 负样本（LibriSpeech+RIR）→ TOS 恢复 tr/cv 450 GB → 转换 100,826/9,133 → 8 卡训练（33–50 h）。验收：同分布 test ≈7.7/8.2；完成后才可做 RealSSA-trained ckpt 的 DNSMOS。
- [ ] **RealSSA-Real 扩展指标**：WeSpeaker speaker-sim、NISQA、Mega-ASR（WER）——需额外环境。
- [ ] **LRS3-2Mix / VoxCeleb2-2Mix**：数据未发布，维持不可复现。
- [ ] **效率补测**：3090 延迟/RAM/显存（本机只有 A800）。
- [ ] **可选清理**：`concat_layers[2]` 闲置（需作者确认）、MDCL 死代码、TFRB/GCI 消融（另开分支）。

### 需要人决定
- [ ] 8 卡资源：排队/租用 8×3090 或 A800，还是批准 2 卡降级（先实测 epoch 成本）。
- [ ] 是否向作者索取 benchmark-specific ckpt / near 重生成脚本（不发邮件则由你决定渠道）。

---

## 7. 常用命令

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python

# 结构与权重
PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_model_params.py
PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_real_weights.py
PYTHONPATH=$PWD $ENV docs/verification/scripts/verify_causality.py

# 测试
PYTHONPATH=$PWD $ENV -m pytest tools/repro/tests -q
$ENV -m pytest studio/tests -q
for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done

# 评测（独立 cwd，防同名覆盖；--gpus 用物理卡号）
mkdir -p /root/tmp/eval_run && cd /root/tmp/eval_run
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/audio_visual_test.py \
  --conf_dir $REPO/configs/LRS2_2Mix.yml --ckpt_path <ckpt> \
  --data_dir /root/data/falcon_lrs2_2mix --gpus 0

# 8 卡论文配方（需 8 卡）
CONF=configs/LRS2_2Mix_Paper_v2.yml DATA_ROOT=/data/falcon_lrs2_2mix_paper \
  bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
```

---

## 8. 证据索引

- 理解与规划：[00 总纲](../understanding/deepdive-20261024/00-SYNTHESIS.md)、[09 基准对齐](../understanding/deepdive-20261024/09-benchmark-alignment.md)
- 三轮 review：[THREE_ROUND_REVIEW_20261030.md](../understanding/THREE_ROUND_REVIEW_20261030.md)
- 论文对齐：[PAPER_ALIGNMENT_20261030.md](../understanding/PAPER_ALIGNMENT_20261030.md)
- Phase 1/2：[REPRODUCTION_PHASE1_REPORT.md](REPRODUCTION_PHASE1_REPORT.md)、[REPRODUCTION_PHASE2_REPORT.md](REPRODUCTION_PHASE2_REPORT.md)、[P0-FIX-report](verification/phase2/P0-FIX-report.md)
- 评测证据：`docs/verification/phase2/{P2D-eval-*,P2E-realssa-*,DNSMOS-realssa-real-*}`、`docs/verification/phase2/P2D-report.md`
- 环境锁：[conda-env-lock.txt](verification/conda-env-lock.txt)、[AGENTS.md](../AGENTS.md)

---

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-31 | 按三轮 review 执行 Day-1 + Week-1 修复；新增本 STATUS 作为单一事实源；配置切论文 Table 16；证据/测试收口 |
| 2026-10-30 | 上游合并、论文公开校正、ch0 复评、ep86 证据恢复、三轮 review |
