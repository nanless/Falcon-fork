# Falcon 复现计划（Reproduction Plan）

> 编写时间：2026-10-21（仓库已同步上游 `689555c`）。配套：`docs/understanding/FINAL_PROJECT_UNDERSTANDING.md`（理解结论）、
> `AGENTS.md`（运行环境强制 conda `falcon`）、`docs/verification/`（已有验证脚本/日志）。
> 本文回答三件事：**复现什么、已经有什么、还差什么、按什么顺序做。**

> **状态更新（2026-10-24）**：本计划写于 2026-10-21，Phase 2/3 状态已大幅变化（RealSSA 已获批并完成评测、
> LRS2 已完成单机训练 8.858/9.270、8 卡论文配方待执行）。**最新状态与规划 v2 以
> `docs/understanding/deepdive-20261024/00-SYNTHESIS.md` 与 `09-benchmark-alignment.md` 为准**；
> 本文保留为历史执行计划（§6 的 F1–F9/F12/F17 多已修复，§7 的 Phase 2/3 状态行已过期）。
>
> **2026-10-30 更新**：上游已同步 `689555c → 439cbe9`（HF 下载计数 + README/项目页 arXiv 链接）；
> **论文已于 2026-10-23 公开：arXiv:2610.32749**，因此本文 §2“论文未公开”及其派生推断作废，
> 协议核对见 `docs/understanding/PAPER_ALIGNMENT_20261030.md`（论文公开后对齐，进行中）。

---

## 0. TL;DR（先看这段）

| 层级 | 内容 | 当前状态 | 结论 |
|---|---|---|---|
| **L1 推理复现** | 用官方权重跑 Studio / 离线推理，验证参数量、形状、数值一致性 | ✅ **已完成** | 真实权重可加载，训练侧与 Studio 侧输出 **bit-identical**；参数 11.72 M / 可训练 0.5377 M；MACs 10.611 G 与 README 一致 |
| **L2 评测复现** | 用官方权重在 RealSSA / LRS2-2Mix / LRS3-2Mix / VoxCeleb2-2Mix 上复现表中指标 | ⚠️ **可做，但要先拿数据 + 修评测代码** | RealSSA-Real 可下载（gated）；RealSSA-Sim 可申请（gated，522 GB）；**LRS2-2Mix 公开且已下载（16.47 GB，完整 AV：mix/s1/s2 + mouths/faces/landmarks）**；**LRS3-2Mix / VoxCeleb2-2Mix 的 HF 仓库没有数据文件** |
| **L3 训练复现** | 从零训练 Falcon，复现主表指标 | ❌ **当前不可直接做** | 缺：RealSSA-Sim 访问权限、数据格式转换、Comet/PL 等 P0 修复；**对比学习负样本（near 源波形）官方未发布**，只能近似或改配方 |
| **L4 贡献/消融复现** | NASS 对比损失、MDCL、TFRB/GCI、因果性分析 | ⚠️ **部分可做** | 架构/因果性可复算（已做）；对比损失需替代负样本方案；~~论文未公开~~（已公开 arXiv:2610.32749），消融设置见 `PAPER_ALIGNMENT_20261030.md` |

**最大的三个卡点**：① ~~论文还没有公开~~（**2026-10-23 已公开 arXiv:2610.32749**，协议见 `docs/understanding/PAPER_ALIGNMENT_20261030.md`）；
② RealSSA-Sim 是需人工审批的 gated 数据集，且**不含近场干扰源的原始波形**（官方 dataset card 明确说明）；
③ LRS3-2Mix / VoxCeleb2-2Mix 训练数据未随 HF 数据集发布。

**建议路线**：先做 **L2-RealSSA（用官方权重）** → 再谈 L3 训练。L2 不需要数据集里的近场源波形，也不需要视频骨干预训练权重，
是性价比最高、最能验证 “我们真的理解了模型” 的一步。

---

## 1. 当前基线（已完成，不需重做）

### 1.1 仓库状态

- 本地 fork 已同步上游：`7576854 → 689555c`（4 个新提交），并已推送到 fork（`origin/main`）。
- 上游更新内容：
  - **RealSSA 数据集发布**（README 增加 RealSSA-Sim / RealSSA-Real 链接与说明；`689555c` 同步中文版）。
  - 新增 `space/`：Hugging Face ZeroGPU Space（record-then-separate，复用 `studio/` 运行时，`build.py` 组装）。
  - Studio 健壮性更新：设备启动超时上限、采集卡顿诊断（`window.falconDiagnostics()`）、i18n 动态更新。
  - ~~**论文仍未公开**~~：2026-10-23 已公开 arXiv:2610.32749（README 已更新链接）。
- 训练/评测代码（`look2hear/`、`audio_train.py`、`audio_visual_test.py`、`DataPreProcess/`）**未变化** → 我们之前发现的 P0 全部仍然存在。

### 1.2 环境与验证资产（已完成）

- conda 环境 **`falcon`**：Python 3.11.17、torch 2.8.0+cu128（2×A800 可用）、pytorch-lightning 2.6.6、全部训练/评测依赖；
  详见 `AGENTS.md`、`docs/verification/conda-env-{lock.txt,yml}`。
- 官方权重：`/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`（已验证格式、strict 加载、bit-identical）。
- 验证脚本：`docs/verification/scripts/`（参数量/形状/因果性/真实权重/评测流程/指标/MACs 等 20+ 脚本）。
- 测试：Studio `pytest studio/tests` → **35 passed / 2 skipped**；Node 测试 8/8；`space/tests` 的 Python 测试需要单独的
  Space 依赖环境（`spaces`/`gradio`，见 §4.10），不属于 `falcon` 环境。

---

## 2. 论文状态与对复现的影响

- ~~**没有公开论文**~~：2026-10-23 已公开 **arXiv:2610.32749**（28 页），README/项目页已有链接；协议核对见 `PAPER_ALIGNMENT_20261030.md`。
- 数据集 card 只说 “accompanies Falcon (NeurIPS 2026)”，未给方法细节。
- 影响：
  1. **RealSSA 生成参数**（房间尺寸范围、RT60、SIR、场景比例）只能从数据集 JSON 统计反推（拿到数据后）；
  2. **评测协议细节**（通道选择、切段方式、DNSMOS 版本、WER/说话人相似度模型）未定义，只能按 README + 数据集 card 的
     “建议协议”执行并**显式记录偏差**；
  3. **消融实验设置**（对比损失 margin/α/β、MDCL 是否用于最终模型）只能按配置推断（配置文件显示最终模型用 NASS 对比损失，
     MDCL 代码路径实际不可用）；
  4. 论文表格中 “Real-world DNSMOS” 只有 DNSMOS，与 RealSSA-Real “无干净参考、只能算参考无关指标” 一致。

> 结论：**复现要以代码 + 配置 + README + 数据集 card 为准，论文只能等公开后补差异说明。**

---

## 3. 复现目标与验收标准（按论文表）

### 3.1 L2 指标目标（README 表）

| Benchmark | 指标 | 论文值 | 验收容差（建议） |
|---|---|---|---|
| RealSSA（Sim，test 9,169 条） | SI-SNRi / SDRi | 7.7 / 8.2 | ±0.3 dB（同权重、同数据、同协议） |
| LRS2-2Mix | SI-SNRi / SDRi | 14.1 / 14.7 | ±0.3 dB |
| LRS3-2Mix | SI-SNRi / SDRi | 16.4 / 16.7 | ±0.3 dB（数据未发布 → 暂不可做） |
| VoxCeleb2-2Mix | SI-SNRi / SDRi | 10.5 / 11.8 | ±0.3 dB（数据未发布 → 暂不可做） |
| RealSSA-Real | DNSMOS | 3.276 | ±0.05（参考无关指标，波动正常） |
| 效率 | 参数量 / MACs / GPU 延迟 | 0.5 M / 10.6 G / 51.3 ms(3090) | 参数量/MACs 精确可复算（已完成 0.5377 M / 10.611 G） |

### 3.2 L3 训练复现验收

- 训练 loss 曲线可复现（PIT + 对比损失，β 退火 0→1.0/200 epoch）；
- 在 RealSSA-Sim **validation** 上 SI-SNRi 随训练上升，reasonable 的早期 checkpoint 接近官方权重水平；
- 完整训练后用 test split 复现 7.7 / 8.2（容差 ±0.5 dB，因训练随机性/算力差异放宽）；
- 记录所有与论文不一致的地方（见 §9）。

---

## 4. 资产清单与缺口（逐项）

图例：✅ 已有；🟡 可获取但需动作；❌ 缺失/阻塞。

### 4.1 代码与权重

| 资产 | 状态 | 说明 / 动作 |
|---|---|---|
| Falcon 代码（训练+评测+Studio） | ✅ | 已同步 `689555c` |
| 论文配置 `configs/Falcon.yml` | ✅ | 训练/评测唯一可信配置（`Experiments/checkpoint/Falcon/conf.yml` 是过期快照，类名错误） |
| `falcon.pt` / `falcon-big.pt` | ✅ | 已缓存并验证（可复用于 L2 评测与 L3 初始化） |
| 视频骨干 `frcnn_128_512.backbone.pth.tar` | ✅ **已从 falcon.pt 导出并验证** | README 的 TODO 链接仍缺失；但训练时视觉分支**冻结**，`falcon.pt` 的 `video_model.*`（137 keys）就是该骨干。Phase 1 已实现并验证：`tools/repro/extract_video_backbone.py` → `FRCNNVideoModel(pretrain=…)` 加载成功（11,186,688 参数、全冻结、`[1,1,50,88,88]→[1,512,50]`），见 `docs/verification/phase1/T5-report.md`（§6.2 F12） |
| Falcon-Big 训练代码（`SISNRRepel`/`SpeakerContrastiveV2`） | ❌ | 仅存在于 Studio 推理副本；训练侧未发布 → Falcon-Big 只能做推理复现 |

### 4.2 数据集

| 数据集 | 状态 | 内容 / 限制 |
|---|---|---|
| **RealSSA-Sim** | 🟡 gated（人工审批） | 522 GB，WebDataset TAR（tr 202 / cv 19 / tt 19 shards，500 条/shard）；每条：`mix.wav` 4ch 16k PCM16、`target.wav` 4ch、`.json`（场景/房间/RT60/components type,SIR,位置/transcript）、`mouths.pt` uint8 RGB `[T,96,96,3]`。**不含 near 源的原始波形** |
| **RealSSA-Real** | 🟡 gated（人工审批） | 54 段/40 录音/10 说话人/8 场景，共 83.99 min；stereo 16k audio、224×224 25fps 人脸视频、嘴部 NPZ、68 点 landmarks、enrollment 音频、逐录音 transcript。**无干净 target，不能算 SI-SDR**；用于 DNSMOS / WER / speaker-similarity |
| **LRS2-2Mix** | 🟡 公开（HF，非 gated） | `lrs2.tar.gz` **16,470,508,590 B 已下载并 sha256 校验**（`ebdfdf31…`）；tr/cv/tt 各 20000/5000/3000 组 2 s `mix/s1/s2` + mouths npz + faces + landmarks + create_mix 脚本；**完整 AV 数据，可直接用于复现**（需要 ≥40GB 解压空间） |
| **LRS3-2Mix** | ❌ 数据未发布 | HF 仓库只有 README（无 tar/数据文件） |
| **VoxCeleb2-2Mix** | ❌ 数据未发布 | 同上（仓库仅 README） |
| LRS2 / LRS3 / VoxCeleb2 原始语料 | 🟡 需自行申请许可 | 用于生成嘴部 crop / 视觉输入；有各自的访问协议（非公开下载） |
| RealSSA 近场干扰源波形 | ❌ 未发布 | dataset card 明确：`components[].path` 只是生成时来源路径，不是可下载的 near 波形；“the original speaker-contrastive loss needs valid near-speaker negatives; the indexing script's fallback to the mixture is not an equivalent reproduction” |
| 嘴部 crop 提取/对齐流水线 | ❌ 不在本仓库 | 训练需要 `mouths/*.pt`（RGB→灰度 88×88 crop 在训练 transform 内做）；RealSSA-Sim 自带 crop，LRS2/LRS3/VoxCeleb2 需参考 CTCNet / Swift-Net 的数据预处理 |
| 近场 `_meta.json` 生成器 | ❌ 不在本仓库 | 本仓库只读、无写；复现时需自建（RealSSA-Sim 的 `.json` 可转换，但 near 波形仍缺） |

### 4.3 评测工具与参考

| 资产 | 状态 | 说明 |
|---|---|---|
| SI-SNR / SDR 评测代码 | ✅（有 bug 需修） | `audio_visual_test.py` + `look2hear/metrics` 的 “sdr” 列实为 −SNR、PESQ 实参传反（我们已确认）；建议改用 `fast_bss_eval` 重新实现 SDRi，SI-SNRi 可复用 |
| DNSMOS 实现 | ❌ | 论文 “Real-world DNSMOS” 需要 DNSMOS P.835 模型（通常用微软 DNSMOS ONNX 或官方工具）；仓库无 |
| WER / speaker-similarity 工具 | ❌ | RealSSA-Real card 提到 WER/说话人相似度评估协议，但仓库无 ASR/声纹模型 |
| 基线模型实现（AV-ConvTasNet 等 10 个） | ❌ | 表里数字来自 Swift-Net/原论文；复现对照可只引用数字，不必重跑 |
| 效率测试脚本 | ✅（部分） | `evaluated_mac_params.py`（MACs=10.611 G 已复算），延迟需在 3090 上测（本机 A800 可测但非论文硬件） |

---

## 5. 数据准备方案（详细）

### 5.1 RealSSA-Sim → 仓库训练格式（主训练集，建议优先）

**Step 1：申请访问**
- HF 页面 `JusperLee/RealSSA-Sim` → Request access（填姓名/单位/国家/用途，勾选非商用学术研究）。
- 人工审批（时间未知）；批准后 `hf auth login`。
- 建议先只下 1 个 test shard 验证管线（`tt/tt-000000.tar`），再下全量（522 GB）。

**Step 2：写转换脚本 `tools/repro/realssa_to_falcon.py`**（新增，不改上游逻辑；Phase 1 已落地该脚本）：
```
输入：RealSSA-Sim/{tr,cv,tt}/*.tar
输出：<DATA>/{tr,cv,tt}/mix/<id>.wav        # mono 16k（通道策略见 Step 3）
      <DATA>/{tr,cv,tt}/target/<id>.wav     # 同通道 mono 16k
      <DATA>/{tr,cv,tt}/mouths/<id>.pt      # uint8 RGB [T,96,96,3]（与训练 transform 期望一致！）
      <DATA>/{tr,cv,tt}/mix/<id>_meta.json  # components 列表（type/path/SIR/position；path 仅 provenance）
      <INDEX>/{tr,cv,tt}/{mix,target,near}.json   # 由现有 DataPreProcess 脚本生成
```
- 关键点：**嘴部数据不用自己转**。训练里的 `transform.py` 期望输入是 `[T,96,96,3]` RGB（先 Grayscale、再 CenterCrop/RandomCrop 88），
  RealSSA-Sim 的 `mouths.pt` 正好是 uint8 RGB `[T,96,96,3]`（我们的分析已确认训练 `.pt` 为 4-D channel-last）。
- TAR 是 WebDataset 格式（每条 4 个成员，顺序不保证），转换时按 `<id>.mix.wav` 前缀配对即可。
- 帧对齐：`T_visual = min(mouth_frames, floor(audio_samples/640))`，与现有 loader 的 640 采样/帧一致。

**Step 3：通道策略（必须做实验并记录）**
- RealSSA-Sim 是 4 声道；README 未声明论文用哪个通道/如何下混。
- 建议同时准备：`ch0`、`mean`（4 通道平均）两版；评测时报告通道协议（这也符合 dataset card 的要求）。
- 注意：`look2hear/datas/avspeech_dataset.py` 的 eval 路径对 mono wav 有 bug（`mean(-1)` → 0 维），先修（§6.2 F5）。

**Step 4：近场负样本（对比学习的关键缺口）**
- 官方发布**没有** near 源波形。可选方案（按推荐度排序）：
  1. **跳过对比损失**：训练时 `use_near=false` + 从 `av_litmodule_nass_v2` 换成只算 PIT 的配置（需要小改或加开关），
     复现 “无对比损失” 的消融基线，并与论文值对比 → 可量化 “对比损失贡献”，但**不是**论文最终模型。
  2. **替代负样本（近似）**：用同 batch 内其它样本的 target/干扰说话人（不同 speaker）作为 negatives，
     或从 LRS3/LibriSpeech 随机采不同说话人音频（带相同 RIR 条件可选）。实现上把 `near.json` 指向这些替代音频，
     并在报告里显式声明 “non-official negatives”。这是工程上最可行的训练复现路径。
  3. **物理重建（高成本）**：dataset card 给出 components 的来源类型（干扰语音 100% 来自 LibriSpeech train-clean-360）、
     SIR、位置与房间几何（JSON 里有 room dims / mic center / T60 / target 位置 / component 位置）。
     理论上可以用 LibriSpeech 干信号 + 相同几何用 gpuRIR 重卷积近似重建 near 波形；但随机种子/实现细节未知 → 只能近似。
     建议作为 “research option”，不建议作为主线。
- 注意：现有 `generate_near_json.py` 的 fallback 是 `[[mix_path, n]]` 哨兵，Dataset 会跳过 → K=0，这**不是**合法复现（官方 card 也这么警告）。

### 5.2 LRS2-2Mix（次训练/评测集；公开，**完整 AV 重建数据**，已下载解压）

**已完整下载并实测结构**（`lrs2.tar.gz` 16,470,508,590 B；sha256 `ebdfdf31…` 与 HF LFS oid 一致；
共 **249,670 个 entry**；成员总大小 **17.39 GB**（建议预留 ≥40 GB 解压空间）。**关键更新：这个发布包是完整的 AV 重建数据集，包含嘴部/landmark/人脸视频**：

```
lrs2_rebuild/                                  # 249,670 entries；成员总大小 17.39 GB（压缩 16.47 GB）
├── audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}/*.wav   # 2.000 s / 16k mono PCM16；tr 20000 / cv 5000 / tt 3000 组
├── audio/wav8k/min/{tr,cv,tt}/{mix,s1,s2}/*.wav    # 8 kHz 版本
├── raw_audio/{train,val,test}/<id>.wav      # 19,649 / 476 / 267 = 20,392 条原始说话人 2 s 片段（16k）
├── mouths/<id>.npz                          # data: uint8 (50, 96, 96) 灰度嘴部（25 fps × 2 s，正好 Falcon 输入格式）
├── landmark/<id>.npz                        # data: float32 (50,1,68,2)（约 1% 帧有 2 张脸）
├── faces/<id>.mp4                           # 20,392 个 224×224 / 25 fps / 50 帧人脸视频
├── create_mix/                              # MATLAB 造混音脚本 + scp + scaling.mat + crop_mouth_from_video.py
├── transform.py / utils.py                  # AVSS 数据加载/人脸对齐（cv2/skimage）
└── train.txt(20,779) / video_path.txt(20,468) / filename_input.csv / retest/dev/redev 等
```

**T6 实测要点（2026-10-22）**：
- `mix == s1 + s2` 实测成立（20/组×3 split，max|Δ|=3.05e-5，纯 16-bit 量化误差）⇒ **s1/s2 已是 scaled source，target 参考直接可得，无需从 raw_audio 重建**；
  target convention（s1 vs s2 谁是目标）论文未声明，复现时需固定并报告。
- split 无泄漏：tr↔raw train（18,691/19,649 ids）、cv↔val（476/476）、tt↔test（267/267）。
- 由于是**完整 AV 数据**（含 mouths/faces/landmarks），LRS2-2Mix 的 AV 评测/训练**不需要 LRS2 原始视频**。
- 建议预留 ≥40 GB 解压空间（实测成员总大小 17.39 GB）；`mouths` 需转成 `(50,96,96,3)` uint8（Falcon transform 用 3→1 的 Grayscale）。

结论（对复现意义重大）：
- **LRS2-2Mix 的视觉输入齐全**：`mouths/*.npz` 是 `uint8 (50,96,96)`（转成 `(50,96,96,3)` 供仓库 transform 使用；
  仓库 `transform = torchvision.Grayscale(num_output_channels=1)` 需要 3 通道输入）；
  `faces/*.mp4`、`landmark/*.npz` 也在，**不需要 LRS2 原始视频**（除非要做不同 crop 的消融）。
- **音频是 2 s 混合片段**，文件名带两个说话人 id 与各自 SNR；**s1/s2 目录随包发布，且实测 `mix == s1 + s2`**
  （max|Δ|=3.05e-5）⇒ **target 参考直接可得**；`raw_audio` + `create_mix`（scp/`create_wav_2speakers.m`/`scaling.mat`）
  仅用于核查或自定义混合，不是 `target` 的必要来源。target convention（s1 vs s2）论文未声明，必须固定并报告
  （我们用 speaker1 与 speaker2 对照，s1 明显更优，已确认 s1）。
- 训练/评测切分：`create_mix` 与 `train.txt`/`video_path.txt` 提供列表；mix 文件名可解析出两个说话人及其 SNR；
  split 无泄漏（tr↔raw train、cv↔val、tt↔test）。
- 仍需 LRS2 原始语料许可？**不需要**（发布包已含所需 media）；但数据集仍受 LRS2/TED 条款约束（见 DATA_ACCESS_GUIDE）。
- `audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}` 与 `audio/wav8k/...` **均已确认存在**（旧版本“只有 mix”的判断已过时）。

### 5.3 LRS3-2Mix / VoxCeleb2-2Mix

- 两个 HF 仓库都**没有数据文件**（仅 README）→ 目前无法复现这两列。
- 可行路径：联系作者要数据；或用 LRS3 / VoxCeleb2 原始语料自行按 README 描述（随机两人混合、SNR −5~5 dB、相同 split 大小）
  重建 —— 但混合随机种子/说话人对未知，**无法得到与论文完全一致的测试集**，只能做 “同分布” 复现。

### 5.4 RealSSA-Real（L2 评测，DNSMOS）

- 申请 gated 访问 → 下载完整 snapshot（含 face/mouth NPZ）。
- 用官方 `scripts/load_inference.py`（`align_tail=True, channel='mean'`）读取 sample：
  `audio float32 16k`、`mouth uint8 [T,96,96]`、`enrollment`、`metadata`。
- 评测：`extract(FalconRuntime.from_pretrained("JusperLee/Falcon"), audio, mouth)` → 估计语音 → **DNSMOS**；
  另可算 WER（ASR 模型自选）与 speaker similarity（声纹模型自选），但**不能算 SI-SDR**（无干净参考）。
- 报告时必须写：通道策略（mean/0/1）、tail alignment、DNSMOS 模型版本。

### 5.5 评测数据切分与协议（必须在报告里写清楚）

- 训练：随机 2 s 片段（`train_samples=10000`/epoch，可重复采样）。
- 验证/测试：**整段**音频（现有 eval Dataset 行为），不是 2 s 窗口。
- 指标：SI-SNRi / SDRi 用 “估计 vs 同通道 target、对齐时长” 计算；RealSSA-Sim 用 test split；RealSSA-Real 用 DNSMOS。
- 与论文对齐的最大不确定性：通道选择、是否整段、SDR 的定义（我们已发现仓库指标实现有 bug）。

---

## 6. 代码修复清单（训练/评测复现前必须做）

> 所有条目都在 `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` §10 有详细证据。按优先级排列；工作量是粗略估计。

### 6.1 P0：不修就无法启动

| # | 问题 | 位置 | 修法 | 工作量 |
|---|---|---|---|---|
| F1 | `comet_logger` 未定义 | `audio_train.py:170` | 改为 `CSVLogger`/`TensorBoardLogger`（或按原意装 comet-ml 并定义 `CometLogger`） | 0.5 h |
| F2 | 三个 litmodule 调用 Comet 专属 `self.logger.experiment.log(dict)` | `av_litmodule.py:153-179` 等 | 改 `self.log_dict(...)`；或加兼容分支 | 2 h |
| F3 | 评测快照类名错误 | `Experiments/checkpoint/Falcon/conf.yml:88` | 改为 `FalconSpeakerContrastive`；或统一用 `configs/Falcon.yml` + 显式 `--ckpt_path` | 0.5 h |
| F4 | 快照/占位路径 | conf.yml 的 `${DATA_ROOT}`、`best_k_models.json` 的 `${ROOT}` | 展开为真实路径；或改用 `configs/Falcon.yml` 重写数据字段 | 0.5 h |
| F5 | eval Dataset mono→0 维 → `torch.stft` 崩 | `avspeech_dataset.py:193-194` | 加 `ndim` 守卫（同训练 set 写法） | 0.5 h |
| F6 | PL 2.x 不兼容 | `audio_train.py:158-167` | `devices=`/`accelerator="gpu"/"cpu"`，去掉旧 `gpus` 参数；DDP strategy 调整 | 2 h |

### 6.2 P1：影响正确性/指标可信度

| # | 问题 | 位置 | 修法 | 工作量 |
|---|---|---|---|---|
| F7 | “sdr/sdr_i” 实为 −SNR/SNRi；行/avg 符号相反 | `look2hear/metrics/allwrapper.py:39-95` | 用 `fast_bss_eval` 重算真正 SDRi；或至少重命名列并统一符号；增加 SI-SNRi 校验 | 3 h |
| F8 | PESQ 实参传反 | `allwrapper.py:50` | `pesq(clean, estimate, fs)`（pypesq 语义） | 0.2 h |
| F9 | 训练产生的 conf.yml 被 `Separator` pop 污染（重复/shared 丢失） | `audio_train.py:82-84` + `Falcon.py:991-996` | dump 前 deepcopy，或先 dump 再建模型 | 1 h |
| F10 | `test_loader` 语义/val 组合理解 | `av_litmodule.py:214-216` | 明确 `[cv,tt]` 每 epoch 验证、每 10 epoch test_loss（现有行为，只需文档化，不用改） | 0 |
| F11 | `train_samples=10000` 决定 epoch 大小 | `configs/Falcon.yml` + DataModule | 复现论文时保持默认；若改全量遍历需说明 | 0 |
| F12 | 视觉骨干权重缺链接 | README TODO | **从 `falcon.pt` 的 `video_model.*`（137 keys，冻结）导出为 `frcnn_128_512.backbone.pth.tar`**（写 10 行脚本） | 0.5 h |
| F13 | `concat_layers[2]` 从未被调用（5,568 参数闲置） | `Falcon.py:868,879` | 复现官方权重时**保持现状**（保证 bit-identical）；若要复现“论文架构意图”，加调用并单独说明 | 1 h（决策） |
| F14 | `_mdcl_loss` 不存在 / 对比类静默吞 near_audio | `Falcon.py:1480`、`Falcon_Contrastive.py:95` | 复现主线用 NASS（不走 MDCL）；若要 MDCL 消融，补实现 | 2 h |
| F15 | DDP per-bucket all_gather 可能死锁 | `av_litmodule.py:164-169` | 改成固定数量 all_gather 或改为单卡验证 | 3 h |
| F16 | `best_k_models.json` 占位/无 ckpt | 仓库自带快照 | 用 `--ckpt_path` 显式指定权重（已可用；`falcon.pt` 含 `state_dict` 已实测） | 0 |
| F17 | 训练日志/指标名不统一（comet vs CSV） | litmodules | 统一 `self.log_dict` 的键名 | 1 h |

### 6.3 新增脚本（建议放到 `tools/repro/`，不改上游目录结构）

| 脚本 | 用途 |
|---|---|
| `extract_video_backbone.py` | 从 `falcon.pt` 导出 visual backbone（`{"model_state_dict": {...}}`） |
| `realssa_to_falcon.py` | RealSSA-Sim WebDataset → 仓库 folder 格式（§5.1） |
| `make_near_json.py` | 生成 `near.json`（官方/替代 negatives 方案） |
| `eval_realssa.py` | 官方权重 → RealSSA-Sim test → SI-SNRi/SDRi（正确实现） |
| `eval_realssa_real.py` | RealSSA-Real → DNSMOS (+ WER/speaker sim 可选) |
| `train_smoke.py` | 1 GPU、小样本、100 steps 的冒烟训练 |
| `repro_report.py` | 汇总指标 → markdown 表（对照 README） |

---

## 7. 分阶段执行计划

### Phase 0 —— 基线（已完成）

> **Phase 1 执行结果见 [`docs/REPRODUCTION_PHASE1_REPORT.md`](REPRODUCTION_PHASE1_REPORT.md)**：
> Gate G1 基本通过（训练/评测入口可跑、指标修正、mono 修复、视觉骨干可导出、数值指纹不变；QA P0/P1=0）。
- 环境、权重、代码同步、L1 验证、Studio 测试。**产出**：`FINAL_PROJECT_UNDERSTANDING.md`、`docs/verification/`。

### Phase 1 —— 代码修复 + 冒烟（1–2 天）
1. 修 F1–F6、F9、F12；写 `tools/repro/` 脚本骨架。
2. 用**随机小数据**（几条假 wav + `.pt`）跑 `audio_train.py` 20–50 steps，确认 loss 能降、checkpoint 能存/读。
3. 用官方权重跑 `audio_visual_test.py`（修好的版本）在 1 条样本上，确认 SI-SNR 输出正确、SDR 定义正确。
4. 从 `falcon.pt` 导出 video backbone 并验证 `FRCNNVideoModel` 可加载。
- **Gate G1**：训练/评测入口在 conda `falcon` 环境可跑通（不依赖论文/大数据）。

### Phase 2 —— RealSSA-Sim 数据管线 + L2 评测（3–7 天 + 审批等待）

> **状态（2026-10-22）**：RealSSA-Sim / RealSSA-Real 访问申请已提交（待批）；LRS2-2Mix 已下载解压，
> Phase 2 并行执行：① 7 dB 差距排查（falcon-big/协议变体）；② LRS2-2Mix 全量转换 + 训练复现；
> ③ RealSSA 数据管线/DNSMOS 工具预置；④ Phase 2 QA。详见 `docs/REPRODUCTION_PHASE1_REPORT.md` 与后续 Phase 2 报告。
>
> **Phase 2 执行状态（2026-10-21T19:55Z，task-25 维护）**：索引与复跑命令见
> [`docs/verification/phase2/README.md`](verification/phase2/README.md)，回归/一致性见
> [`docs/verification/phase2/REGRESSION-docs.md`](verification/phase2/REGRESSION-docs.md)。**P2-A/B/C/E/F 报告已落盘**：
> P2-A（7 dB 排查：指标/协议变体 ≤0.4 dB）、P2-B（big +0.97 dB、窗口化 −1.9~−2.3 dB、视觉分支刚需）、
> P2-C（LRS2-2Mix 全量 28000 条 → `/root/data/falcon_lrs2_2mix`，23.36 GiB）、P2-E（DNSMOS/RealSSA 工具预置，
> **RealSSA 仍 403 待批、无指标结论**）、P2-F（独立对拍 ≤1.8e-7 dB；**归因：发布 `falcon.pt` 训练集为
> `LRS3_complex_nonmoving`（`Experiments/checkpoint/Falcon/conf.yml:99-102`）→ 7.03/7.71 属跨语料结果**）。
> **P2-D 正式训练 19:26:41Z 已启动**（单卡 bs4，日志 `docs/verification/phase2/P2D-train.log`，PID 4002894），
> **LRS2-2Mix 数据转换完成（P2-C）**：28000 条（tr 20000 / cv 5000 / tt 3000，target=s1）→
> `/root/data/falcon_lrs2_2mix`，**23.36 GiB**，JSON/near 齐全（`docs/verification/phase2/P2C-report.md`）。
> **单卡正式训练已启动**（19:26:41Z，bs4；epoch 0 完成：`val_pit_sisnr` 6.732 / `test_pit_sisnr` 6.815，
> 为**绝对 SI-SDR**（非 SI-SDRi；epoch0 在 tt 300 条的 SI-SNRi≈1.52，见 `P2D-eval-epoch_0.csv`）；
> 100 epoch 预计 **≈30 h**（单卡 ~18 min/epoch），结果由 P2-D 报告 v1 更新）。
> **RealSSA-Sim/Real 仍待审批**（数据 403，本 Phase 不产出 RealSSA 指标）；P2-G QA **v1 终检通过**
> （`pytest tools/repro/tests` 111 passed/1 skipped，无源码改动，见 `docs/verification/phase2/QA-report.md`）。
> 已知起点不变：官方 `falcon.pt` 在 LRS2-2Mix tt 全量 3000 条 SI-SNRi **7.03** / SDRi **7.71**
> （`REPRODUCTION_PHASE1_REPORT.md` §3.5）；**P2-F 归因：该权重训练集为 LRS3 派生数据 → 7.03/7.71 属跨语料结果**，
> 论文 14.1/14.7 需 LRS2-trained ckpt。
1. 申请 RealSSA-Sim 访问；下载 1 个 test shard 验证转换器；再下 tt（19 shards）做评测。
2. 转换器落地（4ch→mono、mouths.pt 直用、meta.json→near.json）。
3. 用官方权重在 RealSSA-Sim test 上跑：
   - ch0 与 mean 两种通道；
   - 整段 vs 2 s 窗（论文口径未知，都记录）；
   - 目标：SI-SNRi 7.7 / SDRi 8.2（±0.3 dB）。
4. 顺带修正指标实现，和 README 表核对。
- **Gate G2**：RealSSA-Sim 指标复现成功（这是“模型 + 数据 + 评测”三条链路都通的标志）。

### Phase 3 —— RealSSA-Real + LRS2-2Mix + 其它（1–2 周，取决于数据许可）
1. RealSSA-Real：gated 审批 → 下载 → DNSMOS 评测（目标 3.276）；WER/speaker-sim 可选。
2. LRS2-2Mix：**已下载解压完成**（16.47 GB tar → 17.39 GB 成员）→ 用 `mix/s1/s2` + `mouths/*.npz` 直接构建 Falcon 数据；
   评测目标 14.1 / 14.7。**进展（2026-10-22，Phase 1 顺手完成）**：官方 `falcon.pt` 全量 tt（3000）实测
   SI-SNRi **7.03** / SDRi **7.71**（20 条子集 7.91/8.55；target=s1 已确证），与论文差 ~7 dB，需排查
   发布权重训练集匹配度/评测协议/嘴部对齐（详见 `docs/REPRODUCTION_PHASE1_REPORT.md` §3.5）。
3. LRS3-2Mix / VoxCeleb2-2Mix：仓库无数据 → 列为“不可复现”，或向作者申请；不要用自行混合的数据冒充。
- **Phase 3 状态（2026-10-21T19:45Z，task-25 维护）**：LRS2-2Mix 已完成下载/解压、首轮全量评测（官方 `falcon.pt`：
  SI-SNRi 7.03 / SDRi 7.71，`REPRODUCTION_PHASE1_REPORT.md` §3.5）与全量转换（P2-C，28000 条/23.36 GiB）；
  **P2-F 归因**：发布权重为 LRS3 派生数据训练，7.03/7.71 是跨语料结果；复现 14.1/14.7 需 P2-D 训练 ckpt 或
  作者 benchmark-specific ckpt（P2-D 训练中）。RealSSA-Real 的 DNSMOS 工具已在 P2-E 预置（**待数据审批**，
  审批前不产出 RealSSA 结果）；LRS3-2Mix / VoxCeleb2-2Mix 维持“仓库无数据文件”。
  任务索引：[`docs/verification/phase2/README.md`](verification/phase2/README.md)。
- **Gate G3**：LRS2-2Mix 指标复现（或明确记录阻塞原因）。

### Phase 4 —— 训练复现（1–2 周，含调参）
1. 训练数据：RealSSA-Sim tr（100,826 条），2 s 随机片段；batch 4/GPU × 2 GPU = 8，grad accumulation 5 → 等效 batch 40
   （与论文 10 卡 ×4 对齐）；lr 按线性缩放或保持 1e-3 并说明。
2. 负样本方案（先选 §5.1 Step 4 的方案 1 或 2，明确记录）。
3. 冒烟训练：1 GPU、1k steps、看 val SI-SNRi 是否上升。
4. 正式训练：1000 epochs（或 early stopping），2×A800 估计 2–4 天/次（见 §8）；跑 2–3 个配置（含/不含对比损失）。
5. 用训练 checkpoint 跑 Phase 2/3 的评测流程，对比官方权重与论文值。
- **Gate G4**：训练收敛且 test SI-SNRi 接近论文（容差 ±0.5 dB）或给出合理偏差分析。

### Phase 5 —— 消融与效率（3–5 天）
- 对比损失消融（有/无 / 不同 margin、β 退火）；
- 因果性/窗口推理实验（复用 `docs/verification` 脚本）；
- 效率：参数量/MACs/延迟（本机 A800，注明与 3090 的差异）；
- Studio/离线推理一致性（已做，可复用）。
- **Gate G5**：报告 + 表格（对照论文/README）。

### Phase 6 —— 报告与差异说明
- 汇总：复现结果表、与论文差异、缺失/假设、数据协议、随机种子、硬件、耗时；
- 产出 `docs/REPRODUCTION_REPORT.md`（模板见 §10）。

---

## 8. 算力与存储估算

| 项 | 估计 | 说明 |
|---|---|---|
| 训练数据（RealSSA-Sim） | **522 GB**（全量）；tt 19 shards 约 49 GB | 建议先下 tt + cv 一小部分，再下 tr |
| LRS2-2Mix | **16.47 GB** tar → 解压 **17.39 GB**（已下载完成，sha256 校验；建议预留 ≥40 GB） | 公开 |
| RealSSA-Real | **约 12.84 GB**（422 文件；54 段 84 min + 视频/landmark/enrollment） | gated |
| 2×A800 训练速度（估） | 0.15–0.4 s/step（batch 40 等效，混精） | 模型音频 0.54 M + 冻结视觉 11.2 M；需实测 |
| 1000 epochs（10k 样本/epoch，等效 batch 40） | 250k 步 × ~0.25 s ≈ **17 h**（乐观）~ 70 h（保守） | early stopping 可能更早 |
| 评测（tt 9,169 条整段） | 单卡 A800：约 10–30 min（含 IO） | 取决于整段长度分布 |
| 存储（checkpoints/logs） | < 50 GB | 模型很小 |

> 注：以上是 2×A800 的估算；论文用 10 卡。实际以 smoke 实测为准。

---

## 9. 风险与未决问题（复现报告必须列出）

| # | 风险/缺口 | 影响 | 应对 |
|---|---|---|---|
| R1 | ~~论文未公开~~ | 已于 2026-10-23 公开（arXiv:2610.32749） | 按论文 Table 16/App A.2 执行；差异写报告 |
| R2 | **RealSSA-Sim 近场源波形未发布**（官方 card 明确） | NASS 对比损失无法精确复现 | 方案 1/2/3（§5.1 Step 4），报告中明确“非官方负样本” |
| R3 | RealSSA-Sim gated 人工审批 | 时间不可控 | 尽早提交申请；先用 1 个 shard 验证管线 |
| R4 | RealSSA-Sim 4ch、协议未声明 | 通道选择影响指标 | ch0/mean 两种都跑并报告 |
| R5 | LRS3-2Mix / VoxCeleb2-2Mix 无数据 | 两列论文指标无法复现 | 联系作者；否则在报告中标注“不可复现” |
| R6 | ~~LRS2-2Mix 只有音频~~ **已更正：包内含 mouths/faces/landmarks（完整 AV）** | 无需 LRS2 原始视频；仅自定义 crop/对齐消融才需要 | 用包内 mouths npz（转 `(50,96,96,3)`）即可 |
| R7 | 视觉骨干权重链接缺失 | 从零训练缺初始化 | 从 `falcon.pt` 导出（已验证可行） |
| R8 | 评测指标原始实现有 bug（SDR=−SNR、PESQ 反参） | 论文数字可能来自不同实现 | 重写 SDR/SI-SNR 评测；与 README 对照后再下结论 |
| R9 | 训练随机性/种子未设 | 指标有波动 | 固定 seed、多 seed 平均、报告方差 |
| R10 | RealSSA-Real 隐私限制 | 不能公开人脸/音频；不能做 SI-SDR | 只报 DNSMOS/WER/相似度聚合值；遵守 access terms |
| R11 | `space/` 测试需单独环境（`spaces`/`gradio`，py3.12/torch2.13） | 不要污染 `falcon` 环境 | 如需跑 Space：单独 venv/conda 环境 |
| R12 | 论文 “0.5 M” vs 实测 537,697（≈0.538 M）/ 实际参与前向 532,129 | 报告口径 | 统一写“参数量 537,697（其中 `concat[2]` 的 5,568 未参与前向 ⇒ 生效 532,129；扣对比头 523,113）” |

---

## 10. 待办清单（按角色/顺序，可直接勾选）

**立刻（不需要外部依赖）**
- [ ] F1–F6 代码修复 + `tools/repro/` 脚本骨架
- [ ] 从 `falcon.pt` 导出 video backbone（F12）并验证
- [ ] 写 RealSSA 转换器骨架 + 用 1 条真实 shard 验证（等访问权限）
- [x] ~~检查 LRS2-2Mix `lrs2.tar.gz` 内部结构~~（✅ 已下载解压；完整 AV：mix/s1/s2 + mouths/faces/landmarks，见 §5.2）
- [ ] 重写评测指标（SDR/SI-SNRi，修正符号与 PESQ）

**需要审批/外部**
- [ ] 提交 RealSSA-Sim / RealSSA-Real 访问申请
- [ ] 申请 LRS2/LRS3/VoxCeleb2 原始语料（若要做对应 benchmark）
- [ ] 联系作者要 LRS3-2Mix/VoxCeleb2-2Mix 数据或论文
- [ ] 选定 DNSMOS / ASR / speaker-embedding 工具并固定版本

**训练前**
- [ ] 数据转换完成并抽查 20 条（形状、时长、对齐、meta）
- [ ] 冒烟训练 1k steps，loss 下降
- [ ] 决定负样本方案（§5.1 Step 4）

**报告前**
- [ ] 跑完 L2 评测（RealSSA-Sim；可选 LRS2-2Mix）
- [ ] 跑完 L3 训练（至少 1 个完整配置）
- [ ] 写 `docs/REPRODUCTION_REPORT.md`（表格 + 偏差 + 假设）

---

## 11. 附：本轮上游更新与我们的验证备注

### 11.1 上游更新（`7576854 → 689555c`，4 commits）

- `607352c` Studio：设备启动超时上限 + 采集卡顿诊断（`bounded()`、`window.falconDiagnostics()`、`performanceHint`）。
  我们的 `falcon` 环境重跑 `pytest studio/tests` → **35 passed / 2 skipped**，Node 8/8 通过。
- `a27b1e6` 新增 `space/`（HF ZeroGPU Space）：`build.py` 从 `studio/` 组装 Space + `app.py`/`clip.py` + 前端；
  其 Python 测试需要 `spaces`/`gradio` 依赖环境（`space/requirements.txt`：torch 2.13、py3.12），**不要装进 `falcon`**。
  `space/build.py --out /root/tmp/falcon-space` 已实测成功；`node space/tests/check_i18n.mjs <assembled>` → 226 条中文翻译通过。
- `8188b91` / `689555c` README：RealSSA 数据集发布说明 + 中文同步；**论文仍写 “Paper link will be added when available”**。

### 11.2 数据可得性核实（2026-10-21，HF API）

| HF 仓库 | gated | 文件 |
|---|---|---|
| `JusperLee/RealSSA-Sim` | `manual` | 242 个（`tr/*.tar` 202 + `cv` 19 + `tt` 19 + README/attrs） |
| `JusperLee/RealSSA-Real` | `manual` | 422 个 |
| `JusperLee/LRS2-2Mix` | 公开 | `README.md`、`lrs2.tar.gz`（**16.47 GB**，sha256 `ebdfdf31…`）；`lrs2_rebuild/`：`audio/{wav16k,wav8k}/min/{tr,cv,tt}/{mix,s1,s2}`（20,000/5,000/3,000 组 2 s）+ `raw_audio` 20,392 + **mouths npz (50,96,96)** + faces mp4 + landmarks + `create_mix` MATLAB 脚本/scp + `crop_mouth_from_video.py`；**完整 AV 数据** |
| `JusperLee/LRS3-2Mix` | 公开 | **仅 README**（无数据文件） |
| `JusperLee/VoxCeleb2-2Mix` | 公开 | **仅 README**（无数据文件） |

### 11.3 汇报口径

- 对上级/作者沟通时用一句话：**“推理与参数已完全复现；评测复现需要 RealSSA/LRS 数据与代码修复；
  训练复现被三件事卡住：论文未公开、RealSSA-Sim 需审批且缺 near 源波形、LRS3/Vox 数据未发布。”**
- 本文件中所有“已确认”结论都可在 `docs/verification/` 找到脚本与日志；需要我把 Phase 1 的修复与脚本先落地，可以直接开工。
