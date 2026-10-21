# 评审（R4 复核）：`docs/understanding/04-losses-metrics-video.md` v2 —— 评审人 `ops-tests-docs`

> 复核对象：`docs/understanding/04-losses-metrics-video.md` **R3 v2**（1,191 行，55 条断言，2026-10-21 落盘）。
> **sha256（复核开始）**：`4bfb41b7d692ab0af9c6e26a1743e377c610871e8a8702b97178b58065ccbda9`
> **sha256（复核结束复校验）**：`4bfb41b7d692ab0af9c6e26a1743e377c610871e8a8702b97178b58065ccbda9`（**未变化**；本轮未修改 04 文档与任何源码）。
> R2 评审：`docs/reviews/round1/04-losses-metrics-video-review-by-data-pipe.md`（P0=0 / P1=6 / P2=8 / ❓3）。
> 本轮职责（00-round-plan R4 表）：**第二评审人（执行级验证）**。指定动作 = 用真实权重
> `/root/.cache/falcon_weights/falcon.pt` 与 `audio_visual_test._load_models_from_ckpt` 复核 04 的“ckpt 格式可行”结论；
> 复跑 `docs/verification/scripts/verify_*_04.py` ≥2 个；核对 MACs 10.611 G vs README 10.6 G；
> 并逐条核对 R2 的 P1×6 / P2×8 / 3 条 ❓ 是否闭合。
> 方法：读 04 v2 全文 + 对照源码；独立执行 5 个 04 验证脚本 + 2 个新执行级检查（§6）；核对全部被修正行号；
> 扫描 v2 新增 §8.2-6/7、Q13、A51-A55 是否引入新错误。未修改任何源码（`git status` 仍只有 `docs/` 未跟踪）。

## 1. 结论

**通过（R4 复核）；P0 = 0，P1 = 1，P2 = 3，P3 = 2（建议 R5 前修掉 P1 + 3 个 P2）。**

- **R2 意见闭合度：P1×6 全部闭合；P2×8 中 7 条正确修复、P2-8 修复本身有误（见 §2/§3-P2-1，非结论性）；3 条 ❓ 全部用 R3 复跑关闭（A35/A47/A49 已改 `[验证]`）。**
- **执行级验证全部 PASS 且与 04 的结论一致**（§6）：真实权重格式/`_load_models_from_ckpt` 实跑成功（525+137 keys、strict=True、537,697/11,186,688）；
  Big 权重 517+137、Big audio 1,158,617 → 总 12,345,305≈12.35 M；MACs 复算 **10.611 G / 0.538 M**（README 10.6 G/0.5 M，偏差 0.1%）；
  视觉 MACs 7.77/15.90/17.48 G 未计入；`_find_best_ckpt` 断点/`epoch` 字符串排序 bug/`conf.yml` 类名污染/整句评测/mono 退化路径均复现。
- **新发现问题仅文档级细节**：A38 断言把“字符串排序选中哪个 ckpt”写反（**P1，低影响但属断言事实错误**，正文 §2.4-A 正确）；
  `torch_utils.py:19-44` 引用了不存在的行（文件仅 42 行，正确 `:19-42`，**P2，R2 评审建议 44 本身有误，v2 照抄**）；
  §3.2 流程图仍写 `mouth [T,88,88] float64`（未随 P1-3 条件化，**P2**）；mono 复跑脚本位于 `/tmp/dp-verify`（**P2 可复现性**）。
  不影响 04 的主结论，修完即可作为 FINAL 的 04 基线。

## 2. R2 意见修复逐条核对

### 2.1 P1×6

| R2 条目 | v2 处置（位置） | 我的独立复核 |
|---|---|---|
| P1-1 Q1/B5 从“必然 KeyError”关闭 | Q1 关闭、B5 标注已关闭、§2.4-E 行 165/168 改为“✅ ckpt 格式 / ❌ 因 B2+B3 失败”、新增 A51/A52（`04:541-547,623,773,1004-1005,1018`） | ✅ **成立**。我实跑 `verify_r3_weights_04.py`：`falcon.pt` 顶层 `{architecture:'FalconSpeakerContrastive',epoch:714,state_dict}`、525+137 keys、strict 加载 OK；`audio_visual_test._load_models_from_ckpt(falcon.pt, configs/Falcon.yml)` 返回 `FalconSpeakerContrastive/FRCNNVideoModel`、参数 **537,697/11,186,688**。PROVENANCE.md:36-38 原文一致。 |
| P1-2 mono 升级为条件性 P0-3 + 训练期 val | B7 升级、§4.6 P0 编号、Q4 改写、A44/A53 重写（`04:775,855-860,1021,997,1006`） | ✅ 采纳正确：`avspeech_dataset.py:193-194` 无条件 `.mean(-1)`、训练集 `avspeech_dataset_near.py:176-177` 有 ndim 保护、`av_litmodule.py:86-119` 的 validation 使用同一 `AVSpeechEvalDataset`；`runs/r3-mono-eval-repro.log` 存在（我读了 fixture/形状段）。完整 RuntimeError 未重复执行（依赖 /tmp 脚本，见 P2-3）。 |
| P1-3 dtype 条件化 | §2.3.2 改条件表述、新增 A55、Q13（`04:479-483,1008,1030`） | ⚠️ **基本修复**。§2.3.2/A55 已条件化（uint8→float64、float32→float32、下游 `.float()`/`type_as` 统一）；但 §3.2 流程图 `04:713` 仍残留无条件 `float64` → 见 P2-2。 |
| P1-4 `avspeech_dataset_near.py:166-172` 行号 | 改 `:167-174`（管线调用与切片 169-174）（`04:471`） | ✅ 源码核对：167 注释、168 `mouth_start`、169-174 `lipreading_preprocessing_func(...)[mouth_start:...]`，正确。 |
| P1-5 `tensors_to_device` 返回类型 | 改“长度 5 的 list（`:35-36`），解包”；引用范围 `:19-44`（`04:575-576`） | ⚠️ **返回类型修复正确**（`torch_utils.py:35-36`：`elif isinstance(tensors,(list,tuple)): return [tensors_to_device(...)]`）；但 `:19-44` 越界 → 见 P2-1。 |
| P1-6 整句 vs 2 s 口径 | §4.6-8/9 新增、§3.2/§5.2 补、A54 新增（`04:730-733,873-878,920-921,1007`） | ✅ 成立：`avspeech_dataset.py:179-191` `start=0, stop=None` 整段读；`Falcon.py:1166` 取 `input_shape[-1]`、`:1182-1188` `istft(length=...)`；eval-flow 实测 1 s 样本返回 16000 点且 `T_out=16000`。 |

### 2.2 P2×8

| R2 条目 | v2 处置 | 复核 |
|---|---|---|
| P2-1 `:170` 不能“体现返回 None” | 改“`:169-170` 以语句形式调用（未链式赋值）”（`04:450`） | ✅ 源码 169-170 为两条独立语句；`verify_video_04.py` 实测 `FRCNNVideoModel().eval()` 返回 `None`、不链式赋值安全。 |
| P2-2 K=1 softmax 措辞 | 改“K=1 省去计算、softmax 恒为 1、等价非语义差异”（`04:248-249`） | ✅ 与 `nass_contrastive_loss.py:68-72` 一致。 |
| P2-3 `make_sets[2]` | 改为 `_, _, test_set = datamodule.make_sets`（189）（`04:569`） | ✅ 源码 189 行即此解包。 |
| P2-4 setup 行号 277-293 | 改 `:277-295`（`04:567,941`） | ✅ 源码 277 `def setup`、279-295 构造 train/val/test 三个 dataset。 |
| P2-5 mouth `.pt` 4-D 格式 | §3.2 标 `[推断]`、新增 Q13（`04:711,1030`） | ✅ 采纳；与 05 Q2 对齐。 |
| P2-6 表格混用 ✅/❌ | 拆为“逻辑 ✅；数据 ❌（仓库无 ckpt）”（`04:624`） | ✅ 已拆。 |
| P2-7 P0 未编号 | §2.4/§4.6 加 P0-1/2/3 编号与重排说明（`04:508-510,855-860`） | ✅ 编号自洽，原 ckpt 风险已关闭并让位给条件性 mono。 |
| P2-8 `torch_utils.py` 引用范围 | “改为 `:19-44`”（`04:576`、§9.1） | ❌ **修复有误**：`wc -l look2hear/utils/torch_utils.py` = **42**，函数实际 `19-42`；`:43-44` 不存在 → 见 P2-1（R2 建议值本身错误，v2 照抄）。 |

### 2.3 3 条 ❓

| 条目 | v2 处置 | 我的复核 |
|---|---|---|
| A35 shufflenet 64 崩 | 改 `[验证]`，引 `runs/r3-video.log`（`04:988,1165`） | ✅ 我复跑 `verify_video_04.py`：`shufflenet 分支对 64x64 输入 -> RuntimeError: Given input size: (1024x2x2). Calculated output size: (1024x0x0)`；`globalpool=AvgPool2d(int(input_size/32))`（`shufflenetv2.py:160`）。⚠️ “96 正常”仍属 [推断]（r3-video.log 只对 shufflenet 做了 88 前向；96 是 resnet 分支）→ P3-1。 |
| A47 MACs | 改 `[验证]`，`runs/r3-macs.log` | ✅ 我复跑：**MACs=10.611 G / params=0.538 M**；audio+video 参数 11,724,385=11.72 M。 |
| A49 视觉 MACs | 改 `[验证]` | ✅ 我复跑：64²=7.77 G、88²=15.90 G、96²=17.48 G；脚本只对 audio model 调 ptflops（`evaluated_mac_params.py:88`）、视觉输入 64×64（`:81`）。 |

### 2.4 R2 遗漏 1–6 / 跨模块 1–7

均已在 v2 落实：遗漏 1（训练期 val 同 dataset）→B7/Q4/A53；2（读 `_meta.json`）→§2.4-C 补充；3（评测不过滤 <2 s）→§2.4-C 补充；
4（无截断/输出长度==T）→§3.2/A54；5（`use_near=False` 与三目录要求）→§2.4-C/§5.4；6（评测不建 val/test dataloader）→§3.3 后注。
跨模块 1/2/3 的修正与 05/06/03 域结论一致；跨模块 6（README 命令失败原因归属）已明确为 B2+B3（+B4），与 08 文档 D2/D4 一致。

## 3. R4 新发现问题

### P1（事实错误/误导）

**P1-1｜A38 断言把“字符串排序最终选中哪个 ckpt”写反（04:991）。**
- v2 原文：`epoch=1000.ckpt` 会因字符串排序排在 `epoch=742.ckpt` 之前**被选为“最新”**。
- 实际代码 `audio_visual_test.py:103-106`：`ckpts = sorted(glob("epoch=*.ckpt")); return ckpts[-1]`。升序排序下 `"epoch=1..." < "epoch=7..."`，
  故 1000 **排在最前**、`[-1]` 取到的是 **742**（错误地当作最新；真正最新是 1000）。
- 我执行 `verify_eval_flow_04.py` §[1d]：临时目录同时放 `epoch=742.ckpt`/`epoch=1000.ckpt` → 返回 **742**（脚本输出 “字符串排序结果 = epoch=742.ckpt（>=4 位 epoch 时会选错）”）。
- v2 正文 §2.4-A 第 3 条（04:529-532）表述**正确**（“1000 排在 742 之前…可能选错最新；返回的是 742”），仅 A38 断言行写反。
- 建议 R5：A38 改为“`epoch=1000.ckpt` 字符串排序在 742 之前，`[-1]` 选中 742 并将其误当最新（真正最新 1000 被跳过）”。

### P2（细节）

**P2-1｜`torch_utils.py:19-44` 越界（04:576；§9.1 P2-8 行同）。**
`wc -l look2hear/utils/torch_utils.py` = **42**；`def tensors_to_device` 在 19 行、函数体结束于 42 行（`38-42` 为 dict/else 分支）。
`:43-44` 不存在。根因是 R2 评审写“函数实际到 :44”，v2 采纳了该错误建议（不是新增代码错）。
建议改为 `look2hear/utils/torch_utils.py:19-42`（返回 list 的 `:35-36` 正确，可保留）。

**P2-2｜§3.2 视觉链流程图仍写无条件 `float64`（04:713）。**
`→ mouth [T,88,88] float64` 与 P1-3 修复后的 §2.3.2（04:479-483）、A55（04:1008）自相矛盾：dtype 取决于 `.pt` 存储（uint8→float64，float32→float32）。
我执行 eval-flow §[8] 的 fixture 为 uint8，输出 `mouth(25,88,88) float64`（与 v2 现文一致），但该值不是常量。
建议改为 `→ mouth [T,88,88]（dtype 由 .pt 决定；uint8→float64 / float32→float32）`。

**P2-3｜mono 复跑脚本不在仓库内，第三方不可复现（04:1065-1067、§8.1）。**
v2 的 R3 mono 复跑命令为 `PYTHONPATH=/tmp/dp-verify/stubs:... python /tmp/dp-verify/verify_p0_eval.py`；脚本在 `/tmp`（我确认当前存在，12,312 B），
仓库内只有日志 `runs/r3-mono-eval-repro.log`（我读过：fixtures/mono item/…，含真实模型 RuntimeError）。但 `/tmp` 不随仓库交付，FINAL 复核者无法重跑。
建议把该脚本移入 `docs/verification/scripts/verify_mono_eval_04.py`（与其它 6 个脚本同目录）并在 §8.1 更新命令；或明确引用 `docs/verification/data-pipe-supplement.md` 的可复现路径。

### P3（建议，非阻塞）

- **P3-1｜A35 “88/96 正常”中 96 无直接执行证据**：`runs/r3-video.log`/`verify_video_04.py` 对 **shufflenet** 只做了 88×88 前向（96×96 只测了 resnet 分支；`globalpool=AvgPool2d(3)` 需 96/32=3，可通过推导得出）。建议标 `[推断]` 或补一条 shufflenet-96 前向。
- **P3-2｜A55 的 `transform.py:147-168` 应为 `147-169`**：`get_preprocessing_pipelines` 定义 147-169（168 是 `preprocessing["test"]=preprocessing["val"]`，169 `return`）。不影响结论。

## 4. 遗漏的重要行为 / 对“完全理解”的建议（≥3 条）

1. **Q8 可由本轮执行级结果关闭**：`evaluated_mac_params.py:81` 的 64×64 只用于让 `video_model` 生成 `mouth_embedding`；
   FRCNN 输出形状恒为 `(1,512,50)`（与 88×88/96×96 相同），而 ptflops 只统计 audio model（`:88`）→ **64 vs 88 不改变 10.6 G**。
   我复跑 MACs 得 10.611 G，视觉 MACs 单独为 7.77/15.90/17.48 G 均未计入。建议 04 把 Q8 从“作者是否有意”改为“对报告 MACs 无影响（已由 04-R4 复算确认）”。
2. **Q12 可部分关闭**：`mp.spawn` 的 args 是纯 yaml dict；我实测 `pickle.dumps` 对 `configs/Falcon.yml`/`Experiments/checkpoint/Falcon/conf.yml` 均成功
   （2,678/3,019 bytes，全部 str/float/int/list/dict），pickle 风险可降为“多卡冒烟未测（无 2 卡运行环境）”。建议写入 §8.2。
3. **建议在 §8 记录真实权重文件的 size/sha256**：现 A51/A52 只引用 `runs/r3-weights.log` 的键/形状；我实测
   `falcon.pt` = 47,156,343 B、sha256 `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`，
   `falcon-big.pt` = 49,645,087 B。把文件哈希写入 04 可让“真实权重验证”成为可跨环境复核的硬证据。
4. **建议补“B10 CSV 污染的量化影响”**：04 只定性指出 avg/std 行会被 `pd.concat` 当样本再平均（`audio_visual_test.py:291-307`）。
   可用假 CSV（1 rank 各 N 行 + avg/std）量化均值偏差，给出“多卡时偏差 X%”的具体数字，属于对复现指标的补充。
5. **建议 04 Q2 与 08 D2 交叉引用**：`conf.yml:88` 的 `SpeakerContrastiveV2` 与 `_load_models_from_ckpt` 的 `getattr` 失败，
   在 08 文档已作为 **P0-2/D2** 独立记录（`docs/understanding/08-ops-tests-docs.md` D2）；两域结论一致，建议互相引用避免 FINAL 重复/冲突。

## 5. 跨模块一致性

| 涉及 | 04 的说法 | 本轮核对结论 |
|---|---|---|
| 04 ↔ 08（ops/README 审计） | P0-1：README:165/168 不可复现（B2+B3+B4） | ✅ 与 08 D2（P0）/D4 一致；README 行号 165/168/174/206 我逐一核对无误 |
| 04 ↔ 06（Studio/权重） | Q1/B5 关闭；PROVENANCE:36-40；`_load_models_from_ckpt` 实跑成功 | ✅ 我独立执行同一函数：525+137、strict、537,697/11,186,688；与 06 域（PROVENANCE/权重格式）无冲突 |
| 04 ↔ 05（数据） | mono→P0-3；整句评测；dtype 条件化；行号 167-174/277-295 | ✅ 与 05 §8.6/A5/D3 一致；源码行号正确 |
| 04 ↔ 03（训练） | B11/Q3：`Separator.pop` 污染 config → `conf.yml` 缺键 → 评测退化 49,081 | ✅ 我复跑 eval-flow：537,697→49,081；`audio_train.py:54-58`（建模）→`:82-84`（dump）链路正确 |
| 04 ↔ 01/02（模型/层） | FRCNN `T_out=T_in`、`stride` 死配置、88 裁剪归属 | ✅ 我复跑 video：5 种形状 `T_out=T_in`；`VideoBlock.forward` 不用 `self.stride`（`Falcon.py:923,964-976`） |
| 04 ↔ README | MACs 10.6 G / params 0.5 M / 11.72 M / Big 12.35 M | ✅ 复算 10.611 G vs README:206 `10.6`（0.1% 舍入）；537,697+11,186,688=11,724,385；1,158,617+11,186,688=12,345,305 |
| 04 ↔ README（评测命令） | 不可开箱复现，主因 B2/B3+B4，不是 ckpt 格式 | ✅ 我用真实权重走 shipped conf → `torch.load` 成功后仍 `AttributeError: SpeakerContrastiveV2`（eval-flow §[5]），结论正确 |

## 6. 执行级验证结论（R4 要求）

环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0）；`PYTHONPATH=$PWD/docs/verification/stubs:$PWD`；真实权重 `/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`；Node v22.23.3。全部命令在仓库根执行，未联网下载、未改源码。

| # | 验证 | 命令/证据 | 结果（与 04 结论对比） |
|---|---|---|---|
| E1 | **真实权重 + `audio_visual_test._load_models_from_ckpt`**（Lead 指定动作） | `verify_r3_weights_04.py`（我实跑，输出 `/tmp/r4-weights-check.log`） | ✅ `falcon.pt` 顶层 `{architecture:'FalconSpeakerContrastive',epoch:714,state_dict}`；**525 audio + 137 video** keys，key/shape 与本地模型 0 缺失/0 多余/0 形状不符；strict=True 加载 OK；`_load_models_from_ckpt(falcon.pt, configs/Falcon.yml)` → `FalconSpeakerContrastive/FRCNNVideoModel`、参数 **537,697/11,186,688**。`falcon-big.pt` = `{architecture:'SpeakerContrastiveV2',epoch:715,state_dict}`、517+137、strict OK、`Big audio=1,158,617` → 总计 12,345,305。**04 的“ckpt 格式可行 / Q1-B5 关闭”成立。** |
| E2 | **MACs 10.611 G vs README 10.6 G** | `verify_macs_04.py`（我实跑，`/tmp/r4-macs-check.log`） | ✅ `ptflops(audio_model, input=(a,vemb))` = **10.611 G / 0.538 M**；README:206 为 `10.6`、README:34/103 为 `0.5 M` → 舍入一致（10.611→10.6；0.5377→0.5）。视觉单独：64²=7.77 G、88²=15.90 G、96²=17.48 G，均未计入（脚本只统计 audio model）。 |
| E3 | **视觉编码器形状/冻结/变体** | `verify_video_04.py`（我实跑） | ✅ 96/88/25/1/64 五种输入均 `[B,512,T]` 且 `T_out=T_in`；`.eval()` 返回 None；`update_frcnn_parameter` 后全冻结、BN 保持 eval；shufflenet 64² RuntimeError、88² 正常（96² 见 P3-1）；pretrain 缺失 FileNotFoundError。 |
| E4 | **评测流程/断点** | `verify_eval_flow_04.py`（我实跑） | ✅ 仓库自带目录 `FileNotFoundError`；best_k 取 min、last.ckpt 回退；**epoch 字符串排序选中 742（非 1000）**（A38 据此判错）；config 污染 537,697→49,081；伪造 ckpt 525+137 回环；shipped conf → AttributeError；整段音频 16000 点/25 帧；mono 探针 `mix.shape=()`；前向 `(1,512,25)`/`(1,1,16000)`、CSV 7 列。 |
| E5 | **指标口径（P0-2）** | `verify_metrics_04.py`（我实跑） | ✅ per-utt `sdr` == `PairwiseNegSDR('snr')` 损失值（5.481166 vs 5.481166）；avg 行符号相反；est×2 → “sdr”+6.3 dB 而真 SI-SDR 尺度不变；PESQ 第 1 实参是 estimate（pypesq 合同 `pesq(ref,deg,fs)` → 传反）；STOI 顺序正确；zero_mean 口径差 14.294 vs 28.163 dB。 |
| E6 | **外部包合同（04 §8.3 的独立复核）** | 从 PyPI 取 sdist（未安装） | ✅ `pypesq 1.2.4`：`def pesq(ref, deg, fs=16000, normalize=False)`（`pypesq/__init__.py:7`）；`fast-bss-eval 0.1.4`：`sdr_pit_loss(est, ref, filter_length=512, zero_mean=False, ...)` 内部 `return sdr(ref, est, change_sign=True, ...)`（`torch/metrics.py:283-302`）→ 与 04 §8.3/A25/A27 一致。asteroid 0.6.0 未复核。 |
| E7 | **Q12/mono 补充** | `pickle.dumps` yaml configs；`ls /tmp/dp-verify/verify_p0_eval.py`；读 `runs/r3-mono-eval-repro.log` | ✅ 两套 config 均可 pickle（2,678/3,019 B）；mono 复跑脚本当前在 `/tmp` 但不在仓库（P2-3）；其日志与 data-pipe-supplement 互证。 |
| E8 | **04 文档完整性** | `sha256sum` 前后复校验；`wc -l`；断言表 A01-A55 | ✅ 1,191 行；55 条断言（R2 时 50 → 新增 A51-A55）；sha256 评审前后未变（`4bfb41b7…da9`）。 |

**未执行/未独立复核**：mono 的完整真实模型 RuntimeError 我未重跑（依赖 /tmp 脚本；已读日志 + eval-flow 形状探针）；`fast_bss_eval`/`asteroid` 之外的第三方数值；真实 LRS2/LRS3 数据端到端评测（无数据）；多卡 `mp.spawn` 冒烟（无多卡执行环境）。

## 7. 给作者的问题清单（R5 前可修）

| # | 问题 | 建议 |
|---|---|---|
| Q1 | A38 断言把“选中 742”写成“1000 被选为最新”（正文 §2.4-A 正确） | 按 §3-P1-1 改写 A38；或直接引用正文第 3 条 |
| Q2 | `torch_utils.py:19-44` 越界（R2 建议值有误） | 改 `:19-42`（§2.4-C.6 与 §9.1 P2-8 两处） |
| Q3 | §3.2 流程图残留 `float64` | 改为条件 dtype 措辞（与 A55 一致） |
| Q4 | mono 复跑脚本在 `/tmp`，仓库内不可复现 | 移入 `docs/verification/scripts/` 并更新 §8.1 |
| Q5 | Q8 是否可按本轮结果关闭？ | 建议关闭为“64×64 不影响 10.6 G（已复算）”，保留“作者意图未知”一句即可 |
| Q6 | Q12 是否可按 pickle 实测降级？ | 建议改为“pickle 已验证（本轮），多卡冒烟仍待有 GPU 环境” |
| Q7 | A35 的“96 正常”是否补 shufflenet-96 执行或标 [推断]？ | R5 二选一 |
| Q8 | 是否把真实权重文件 size/sha256 写入 §8？ | 建议加（见 §4-3，可提升真实权重证据强度） |
| Q9 | A55 的 `transform.py:147-168` 是否改 `147-169`？ | 细节，可顺手改 |

## 8. 复核记录

- 被复核文档：`docs/understanding/04-losses-metrics-video.md` v2（1,191 行）；
  sha256 开始/结束均为 `4bfb41b7d692ab0af9c6e26a1743e377c610871e8a8702b97178b58065ccbda9`（未变化）。
- 执行活动：5 个 04 脚本（weights/macs/video/eval_flow/metrics）全部实跑成功（输出另存 `/tmp/r4-*.log`，与 04 `runs/r3-*.log` 一致）；
  另做 2 项新验证（E6 外部包 sdist 合同、E7 pickle/脚本位置检查）；读源码核对 40+ 处 `file:line`。
- 未修改源码、04 文档或任何他人文档；`git status` 仅 `docs/` 未跟踪（本文件为新增）。
- 结论：**建议通过（Minor 修订）**。执行级证据足以支撑 04 的核心结论（ckpt 格式可行、MACs 10.6 G、mono 条件性 P0-3、整句口径、指标 P0-2）；R5 前修 P1-1 + P2-1/P2-2/P2-3 即可。
