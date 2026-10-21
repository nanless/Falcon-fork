# 评审：`docs/understanding/05-data-pipeline.md` —— 评审人 `layers-dsp`（R4/round2 复核）

> 复核对象：`docs/understanding/05-data-pipeline.md` R3 定稿（作者 `data-pipe`）。
> **评审开始 sha256**：`30b9d96903bae6193822d77c1568c2768243d4c7f2fb1b7aebd45078d95bd179`（900 行，
> 与作者记录 `docs/verification/runs/data_pipe/05-data-pipeline-r3.sha256` 一致）。
> **评审结束 sha256**：同上 `30b9d969…bd179`（**未变化**；本评审基于该版本，无漂移）。
> 复核口径：`00-CONVENTIONS.md` §5 + Lead 的本轮重点清单（执行级验证 + 遗留核对）。
> 独立证据：`docs/verification/scripts/review_05_data_pipeline_checks.py`，日志
> `docs/verification/runs/review_layers/05-review-checks.log`；另**复跑了作者的两套验证套件**
> （`verify_dataset2.py` 57 PASS / 0 FAIL；`verify_preprocess.py` 19 PASS / 0 FAIL）与
> `verify_realweight_eval.py` 关键结论。环境 `/root/.venvs/falcon/bin/python`（3.10 + torch 2.8，CPU）；
> 真实权重 `/root/.cache/falcon_weights/falcon.pt`（只读，不联网/不训练）。**未修改任何源码与被评审文档。**

## 1. 结论

**通过（需小修）**。05 的 R3 是一次高质量修订：核心 P0/P1 结论经我独立执行级复核**全部成立**，且部分数字
（真实权重 sha/size/keys、整句前向形状、`_meta` 六步链、dtype 条件化、57/57 与 19/19 套件）逐项精确复现。
仅剩 **2 个 P1（文档内部一致性问题）与 4 个 P2**：

| 级别 | 数量 | 摘要 |
|---|---|---|
| **P0** | **0** | 未发现影响主结论的事实错误；R3 的两项核心升格（val loop、`_meta.json`）经验证成立 |
| **P1** | **2** | ① **A18 "降级"未落到断言表置信度列**（表内仍写 `[代码][验证]`，与 §8.5/§9.1 自相矛盾）；② **A60/§3.9/§8.4 的"uint8 输入 → max\|Δ\|=0.0"不可复现**（实测 2.574e-07；逐位相等只在训练侧输入 float32 时成立，且与 §5.4 自己的 dtype 条件化冲突） |
| **P2** | **4** | §8.4(1) 的训练侧输出形状/dtype 描述与实际管线不符；§8.7 命令用 `/tmp` stub（非仓库内）；§9.1 旧版哈希只有简写；A60 前提列与 §8.4 表述张力 |

**对"完全理解"的贡献**：05 把"数据侧可复现性"这条最容易被忽略的断点（`_meta.json` 缺失 ⇒ NASS 负项静默失效）
钉死，并用真实权重关闭了 04/06 的 ckpt 格式疑问；本轮只需修两处文档内部一致性即可作为 FINAL 数据域依据。

## 2. 断言逐条核对表

判定：✅正确；⚠️部分正确/需补条件；❌错误；⏭未逐条核对。编号沿用 05 文档。

### 2.1 Lead 指定重点（R3 核心修正）

| 条目 | 判定 | 我的独立复核（执行级） |
|---|---|---|
| **P0-1 / A64**：`val_dataloader=[val,test]`，`trainer.fit` 每 epoch 遍历 [cv,tt]，每 10 epoch 在 tt 上算 `test_loss`；`trainer.test()` 从未调用 | ✅ | 源码直读确认：`av_litmodule.py:214-216` `return [self.val_loader, self.test_loader]`；`:125` `if current_epoch % 10 == 0 and dataloader_idx == 1:`；`:174` `if current_epoch % 10 == 0:`；`audio_train.py:51` `datamodule.make_loader`（无括号）、`:180` `trainer.fit(system)`、全文无 `trainer.test(`（grep False）。`[验证]` 日志 §[V5] |
| **P1-1 / A65/A67**：README 流程无 `_meta.json` → fallback → 哨兵跳过 → K=0 → NASS 负项/ MDCL / val 分桶静默失效 | ✅ | 四环直验：① `README.md/README_zh.md` 中 `_meta` **0 命中**；② 全仓 `_meta.json` 只有**读取者**（`process_lrs23.py:55`、`generate_near_json.py:48`、`avspeech_dataset.py:204`），无写入者；③ fallback `process_lrs23.py:69-71` / `generate_near_json.py:62-65`；④ 哨兵跳过 `near.py:188-191`；⑤ `nass_contrastive_loss.py:71-74` 无负样本 → `l_neg = embed_query.new_zeros(1)`；⑥ `av_litmodule_mdcl.py:114-115` `if n_near == 0: continue`；⑦ val 分桶 `av_litmodule.py:161-169` + `avspeech_dataset.py:201-210`（n_near 只来自 meta） |
| **A18 降级**（自建 torchvision stub 的 1 通道路径不是真实依赖验证） | ⚠️ **部分落实** | 前提列已加"1 通道依赖真实 torchvision（未验证）"，但**置信度列仍为 `[代码][验证]`**；§8.5 与 §9.1-P1-2 均声称"已降级为 [代码]+[推断]" → 三处自相矛盾（P1-1） |
| **dtype 条件化**（§5.4 / A19 / A23） | ✅ | 独立复现：numpy 2.2.6 下 `((uint8−0.0)/255.0).dtype == float64`、`float32→float32`；训练 val 管线对 `(50,96,96) uint8` 输出 **`(50,88,88) float64`**（min −2.5515 / max 3.5091）。与 §5.4 表述一致。`[验证]` §[V3] |
| **near 必须 16 kHz**（P1-7 / A66 / Q16） | ✅ | `near.json` 只写 `[path, sf.info().frames]`（`generate_near_json.py:54-60`；`process_lrs23.py:61-67`）；Dataset 用 `sf.read(start,stop)` 且 `n_samples < seg_len` 比较、无重采样/无校验（`near.py:193,198-203`）。契约与风险表述准确 |
| **mouth `[B,50,88,88]`**（§5.4 / A16/A17/A23） | ✅ | `seg_len=int(2.0*16000)=32000`、`fps_len=int(2.0*25)=50`（`near.py:77-78`）；`mouth_start=int((rand_start/16000)*25)`（`near.py:168`）、取 `[ms:ms+50]`；`AVSpeechTrainDatasetWithNear` 返回 `mouth (50,88,88)`（作者套件 D3 + 我复跑 57 PASS） |
| **§5.5 config 逐键对照 / A67 评测链不走 DataLoader** | ✅ | `audio_visual_test.py:189` `_, _, test_set = datamodule.make_sets`（无 DataLoader）；`avspeech_dataset_near.py:283-295` 三个 dataset 均在 `setup()` 构建 |

### 2.2 A60 / A68 专项（Lead 指定）

| 条目 | 判定 | 复核结果 |
|---|---|---|
| **A60**：Studio `preprocess_mouth` 与训练 val 管线逐元素等价（`(50,96,96) uint8`，**max diff 0.0**） | ❌（条件缺失，P1-2） | 用**两边真实代码**跑同一 `(50,96,96) uint8`：训练管线（`transform.get_preprocessing_pipelines()["val"]`）返回 **float64**，Studio 返回 **float32**；`max|Δ| = 2.574e-07`（raw）/ **2.384e-07**（训练输出 cast 到 float32 后），`np.array_equal` = **False**。逐位一致（0.0）**只在训练侧输入 float32 时成立**（我测变体 B：`.pt` 为 float32 ⇒ 精确相等 True）⇒ 应写成条件式（见 §3-P1-2） |
| A60 的中间步骤 `CenterCrop(88) == [4:92,4:92]`（§8.3/A20） | ✅ | 梯度帧逐元素：训练管线 vs 手算 `(x/255−0.421)/0.165` `max|Δ| = 0.000e+00`；crop 位置 4:92 正确 |
| **A68**：`falcon.pt` 顶层 `{state_dict,architecture,epoch}`、662 key（525+137）、strict OK；`T=32000/F=50` 与 `T=48513/F=75` 均 `len==T`、finite | ✅ **精确复现** | sha256 `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`、size **47,156,343 B**、architecture `FalconSpeakerContrastive`、epoch 714、662 key（525/137）；audio+video `strict=True` 均 All keys matched（537,697 + 11,186,688 = 11,724,385）。前向：`T=32000,F=50` → video `(1,512,50)`、audio `(1,112,251,129)`、est `(1,1,32000)` len==T True finite True；`T=48513,F=75` → video `(1,512,75)`、audio `(1,112,380,129)`、est `(1,1,48513)` ✔。与作者 `runs/data_pipe/verify_realweight.log` 逐行一致。`[验证]` §[V1][V2] |
| **A64 的前提列** "每 epoch 遍历；test_loss 仅 epoch%10==0" | ✅ | 与源码/PL 语义一致（非 10 的 epoch 里 tt loader 仍被遍历但不计算 loss） |

### 2.3 其它抽查（R3 修订点与作者套件）

| 条目 | 判定 | 说明 |
|---|---|---|
| A16/A17/A20/A21/A22 | ✅ | 行号与公式逐一核对（`near.py:77-78,168`；`transform.py:82-100,147-169`；`near.py:168` 硬编码 16000） |
| A11/A12/A13/A14/A15 | ✅ | `near.py:111-122`（过滤 + 反序）、`:149-150`（`__len__` 返回 samples）、`:154`（忽略 idx 随机）与文档一致 |
| A35 `n_near` 来自 `_meta.json` | ✅ | `avspeech_dataset.py:201-210`：读 `mix_path.replace(".wav","_meta.json")`、计 `type=="near"`、异常静默→0 |
| A45/A67 评测返回 5 元组、不走 collate | ✅ | `avspeech_dataset.py:213`；`audio_visual_test.py:189,229-231` |
| A55/A56/A57/A58/A59（预处理脚本边界） | ✅ | 作者 `verify_preprocess.py` 我复跑 **19 PASS / 0 FAIL** |
| A1–A63 其余断言 | ✅（抽样） | 作者 `verify_dataset2.py` 我复跑 **57 PASS / 0 FAIL**（`TOTAL FAILURES: 0`），与 §8.2 声称一致 |

## 3. 错误清单

### P1-1（文档内部一致性）：A18 的"降级"没有落到断言表置信度列

- **文档现状**（三处互相矛盾）：
  - `05:651`（§6 A18）：置信度列 = **`[代码][验证]`**；
  - `05:803`（§8.5）："R3 已把该断言降级为 **[代码]+[推断]**"；
  - `05:870`（§9.1 P1-2）："A18 降级为 **[代码]+[推断]** 并加条件列"。
- **事实**：A18 的核心争议是"1 通道 `.pt` 能否工作"依赖自建 torchvision stub（stub 显式 `if img.shape[-3]==1: return img`），
  真实 torchvision 行为未验证 ⇒ 正确置信度应是 `[代码]+[推断]`（或 `[代码]`），**不应保留 `[验证]`**。
  作者只改了"前提/条件"列，漏改置信度列。
- **影响**：05 是 FINAL 的数据域依据；断言表是其他评审人/Lead 快速核对的主入口，"降级"是最初评审的 P1-2 采纳项，
  表内不一致会让 FINAL 误把"1 通道可用"当已验证事实。
- **修正**：A18 置信度列改 `[代码]+[推断]`（与 §8.5/§9.1 一致）；或若坚持 `[验证]`，则 §8.5/§9.1 改为
  "条件列已补、置信度仍为验证（因 …）"并给出理由。二选一，三者必须一致。

### P1-2（可复现性）：A60/§3.9/§8.4 的"uint8 输入 max|Δ|=0.0"不成立

- **文档断言**：
  - `05:412`（§3.9）："已用两边真实代码交叉验证：同一 `(50,96,96)` **uint8** 输入，输出逐元素 **max|diff| = 0.0**"；
  - `05:693`（A60）："（`(50,96,96) uint8`，max diff 0.0）"；
  - `05:789-793`（§8.4(1)）："同一 `(50,96,96) uint8` 输入 … train → `(1,1,50,88,88)` **float32** … max|diff| = 0.0"。
- **我的独立实测**（两边真实代码、同 seed、同一 `uint8 (50,96,96)`）：
  | 训练侧输入 | 训练管线输出 dtype | raw `max|Δ|` | cast f32 后 `max|Δ|` | 逐位相等 |
  |---|---|---|---|---|
  | **uint8**（真实 Dataset 路径：`torch.load`→permute→Grayscale→squeeze→numpy） | **float64** | **2.574e-07** | **2.384e-07** | **False** |
  | float32（假设 `.pt` 存 float32） | float32 | 0.0 | 0.0 | True |
- **根因**：`Normalize(0.0,255.0)` 在 numpy 上把 uint8 提升为 **float64**（05 自己在 §5.4 写了这一点），
  而 Studio `preprocess_mouth` 先 `astype(np.float32)/255.0` 再做两次减法/除法（全程 float32）⇒ 两者相差
  float32 舍入级（~1 ulp）。**0.0 只在训练侧输入本来就是 float32 时成立**（即 `.pt` 为 float32 的情形，
  恰好是 A19 的前提，但不是 §3.9/A60 所写的 uint8 情形）。
- **影响**：结论"数值等价"仍然成立（float32 精度内），但"逐位/0.0"是**过度声明**；若 FINAL 引用"0.0"，
  会被严格复现者推翻。同时 §8.4(1) 写 train 输出 "float32" 与 §5.4"uint8→float64"自相矛盾（见 P2-1）。
- **修正建议**：改为条件式表述，例如：
  > "同一 `(50,96,96) uint8` 输入下，训练 val 管线（numpy float64）与 Studio `preprocess_mouth`（float32）在
  > float32 精度内等价：raw `max|Δ|=2.6e-07`、cast 后 `2.4e-07`；**若 `.pt` 以 float32 存储（训练侧输入为 float32），
  > 两者逐位一致（max|Δ|=0.0）**。"
  并在 §8.4(1) 写明训练侧输出的包装步骤（`[None,None]` / `.float()`）与实际 dtype。

### P2 级

| # | 问题 | 证据 | 修正建议 |
|---|---|---|---|
| P2-1 | §8.4(1) 声称 train 输出 `(1,1,50,88,88) float32`，但真实 `get_preprocessing_pipelines()["val"]` 返回 `(50,88,88)`（uint8 输入时 float64）；`(1,1,·)`/`float32` 是额外手工包装 | `transform.py:161-167`；§8.4(1) | 写明包装步骤：`out[None,None]` + `.float()`（或对比前统一 cast），并给出实际 dtype |
| P2-2 | §8.7 运行命令用 `PYTHONPATH=/tmp/dp-verify/stubs:…`（非仓库内、`/tmp` 可被清理）；而作者已把等价 stub 补交到 `docs/verification/stubs/data_pipe/` | `05:856`；`docs/verification/scripts/data_pipe/README.md` | 命令改为 `PYTHONPATH=$PWD/docs/verification/stubs:$PWD/docs/verification/stubs/data_pipe:$PWD`（脚本内部已自动插入 repo stub 路径，可直接简化） |
| P2-3 | §9.1 的版本锚（`f1c90fa3…`/`dcd32572…`）只有 8 位简写，且未落 anchor 文件；R4 复核无法校验旧版本 | `05:863` | 与当前版本一样，把 R2/R3 各版 `sha256sum` 全文写入 `docs/verification/runs/data_pipe/`（当前版已有 `05-data-pipeline-r3.sha256` ✓，旧版可补一行） |
| P2-4 | A60 的"前提/条件"列写"数值等价；抽取端一致性未验证"，与 §8.4 的"max|diff=0.0"存在表述张力（若按条件式修订 P1-2，本条自动消失） | `05:693,789-793` | 随 P1-2 一并改为"float32 精度内等价；float32 `.pt` 时逐位一致" |

## 4. 遗漏的重要行为（代码里有、05 未写或写得不够）

1. **`preprocess_mouth` 只接受固定 50 帧窗口**（`studio/falcon_runtime.py:95-105` 断言 `frames.shape == (WINDOW_FRAMES,96,96)`），
   因此 A60 的等价性**只覆盖 2 s 窗口（F=50）**；05 §8.7 的变长评测路径（F=75）不能复用 Studio helper。
   建议在 §3.9 补一句边界（不影响结论，但避免读者把 A60 推广到任意 F）。
2. **`_meta.json` fallback 的"哨兵"不可区分**：fallback 写入 `[mix_path, mix_n]`，Dataset 以 `near_path == mix_path` 判定跳过
   （`near.py:188-191`）；若真实场景确实存在"near 文件就是 mix 文件"的退化用例，会被静默当成 fallback 跳过——
   05 已把 fallback 流程写清，但未指出这一歧义（属 RealSSA 数据生成端的弱约定，可并入 Q16/Q3）。
3. **训练期 val 循环的 2 个 loader 会各跑一遍 `AVSpeechEvalDataset`**：文档已写；补一句"tt 集在每个 epoch 都被遍历
   （只是非 10 的 epoch 不算 loss）"能避免"每 10 epoch 才读 tt"的误读——R3 前提列已部分表达，正文 §3.4 可再点明。

## 5. 跨模块一致性问题

| # | 冲突点 | 05 表述 | 对照文档/证据 | 判定 / 统一建议 |
|---|---|---|---|---|
| C1 | ckpt 格式 | A68：`{state_dict, architecture, epoch}`、662 key、strict OK | 04 文档 Q1/B5（data-pipe 的 04 评审已关闭）、`studio/falcon_model/PROVENANCE.md:36-40`、我独立复现 | **一致 ✅**，FINAL 可直接引用 A68 关闭 04-B5 |
| C2 | mouth 形状 88×88 | §1.3/§5.4：`[B,50,88,88]`（96×96 存储帧中心裁 88） | 03 文档曾写 96（05 §9.1-C1 记为 03 的错误）、`frcnn_videomodel.py` 接受 88、作者套件 D3 | **05 正确 ✅**；建议 FINAL 以 05+04 为准，提醒 03 修订 |
| C3 | `val_dataloader` 组成 / `test_loss` | A64：`[cv,tt]` 每 epoch、每 10 epoch `test_loss`、`trainer.test()` 未调用 | 03 文档 V4（11-epoch 实测）、源码 `av_litmodule.py:214-216,125-140,174-181` | **05 已与 03 一致 ✅** |
| C4 | 真实权重整句契约 | A68/§8.7：T 任意、F 任意、`len==T` | 我的 02/01 文档：`T'=1+⌊T/128⌋`、`F=win/2+1=129`、`istft(length=T)`、`NeuroSCNetFusion` nearest 对齐 | **一致 ✅**（我独立复算 `T=48513→T'=380` 与 05 相同） |
| C5 | `_meta.json` 严重度 | 05 §1.4 升为 P0 主结论 | 03/04 文档是否已同步"README 流程 NASS 负项静默失效"？ | **待确认（跨文档）**：建议 FINAL 在训练/损失域显式引用 05-A65，避免"论文贡献可复现"表述冲突 |
| C6 | Studio 数据契约 | §3.9：Studio 重实现同一套预处理，与训练 val 逐元素一致 | 06 文档 Studio-side 描述、我的 A60 条件式结论 | 结论方向一致；**建议 06 引用时带上 P1-2 的 float32 精度条件**，不要写"逐位 0.0" |

## 6. 给作者的问题清单（R4 请回答/小修）

| # | 问题 | 期望动作 |
|---|---|---|
| Q1 | A60 的"0.0"是在 uint8 还是 float32 输入下测的？我的实测：uint8→2.574e-07 / cast 后 2.384e-07；float32→0.0。 | 按 P1-2 条件式改写 A60/§3.9/§8.4(1)，并给出两行实测数字 |
| Q2 | A18 的置信度列是否同意改为 `[代码]+[推断]`，与 §8.5/§9.1 对齐？（或说明保留 `[验证]` 的理由） | 三处一致 |
| Q3 | §8.4(1) 的训练侧包装（`[None,None]`/`.float()`）能否写明？当前"train → float32"与 §5.4 dtype 条件化矛盾 | 补一行伪代码/说明 |
| Q4 | §8.7 的运行命令能否改用仓库内 stub 路径（`docs/verification/stubs/data_pipe`），避免依赖 `/tmp`？ | 更新命令 |
| Q5 | 是否同意在 §3.9 补一句"`preprocess_mouth` 仅支持固定 50 帧窗口，A60 的等价性不覆盖变长 F"？ | 补边界说明 |
| Q6 | 旧版哈希（`f1c90fa3…`/`dcd32572…`）是否补全文到 `runs/data_pipe/`？ | 便于 R4/FINAL 校验版本链 |

## 7. 本轮复核活动记录（执行级）

```bash
cd /root/code/github_repos/Falcon-fork
# 0) 版本锚定（评审开始/结束各一次）
sha256sum docs/understanding/05-data-pipeline.md   # 开始=结束=30b9d969…bd179（900 行）

# 1) 我的独立复核脚本（A68/整句契约/dtype/Preprocess 等价/源码断言行）
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python docs/verification/scripts/review_05_data_pipeline_checks.py \
  | tee docs/verification/runs/review_layers/05-review-checks.log

# 2) 复跑作者的两套验证套件（执行级，非只读日志）
export PYTHONPATH=$PWD/docs/verification/stubs:$PWD/docs/verification/stubs/data_pipe:$PWD
PY=/root/.venvs/falcon/bin/python
$PY docs/verification/scripts/data_pipe/verify_dataset2.py      # -> 57 [PASS]，TOTAL FAILURES: 0
$PY docs/verification/scripts/data_pipe/verify_preprocess.py    # -> 19 [PASS]，TOTAL FAILURES: 0

# 3) 交叉核对作者真实权重日志
cat docs/verification/runs/data_pipe/verify_realweight.log      # 与我的 §[V1][V2] 逐行一致
```

**关键结果**：ckpt sha/size/keys/epoch/strict 全对；`T=32000/F=50` 与 `T=48513/F=75` → `est len==T`、finite；
uint8→float64、float32→float32（与 §5.4 一致）；preprocess 等价 2.574e-07（raw）/2.384e-07（cast）≠ 0.0；
A18 置信度列仍 `[代码][验证]`（与 §8.5/§9.1 矛盾）；57/57、19/19 套件复现通过。

**失败/边界**：MPS/CUDA 未测（沙箱无 GPU，05 亦未声称）；真实训练数据/`_meta.json` 资产仍缺（05 Q2/Q3 已列）；
我未复跑 `verify_p0_eval.py`（其结论与 05 §8.6 一致，且需构建模型，非本轮重点）。
