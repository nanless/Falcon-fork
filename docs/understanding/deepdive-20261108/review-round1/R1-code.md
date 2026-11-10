# R1 逐行代码审查（Falcon-fork，HEAD 6c1d5b7 + 工作区改动）

> **2026-11-10 复核提示：**本文评的是当时的 HEAD 和工作区；子集 `verified` 的历史问题已有修订，但当前覆盖检查仍主要对行数，重复 ID 与遗漏可能同时存在。默认路径对照对应当次输入、测试环境和版本。当前边界见[复现状态](../../../STATUS.md)。

审查对象：`look2hear/models/Falcon.py`、`studio/falcon_model/models/Falcon.py`、`look2hear/losses/{sisnri,pit_wrapper}.py`、
`look2hear/datas/{avspeech_dataset,avspeech_dataset_near}.py`、`look2hear/layers/normalizations.py`、`audio_visual_test.py`、
`evaluated_mac_params.py`、`tools/repro/phase2/eval_realssa_real.py`、`tools/repro/phase2/*.sh`、`configs/*.yml`、
`docs/verification/scripts/data_pipe/verify_dataset2.py`、`tools/repro/tests/test_metrics.py`。
实验在 `/tmp/r1` 的 `git archive` 副本上进行。

## 0 结论速览

- **P0 = 0 / P1 = 2 / P2 = 9。建议本轮合并**；其中 **P1-1（coverage 戳）建议合并前改掉（约 5 行）**，P1-2 可在合并后立即改文档+默认值。
- 所测模型默认路径**逐位一致**：`configs/Falcon.yml` 建 `FalconSpeakerContrastive`，HEAD vs 工作区 → 525 个 state_dict 键、参数哈希、前向输出全等（`out_sum = -71.21968841552734`）。
- **checkpoint 兼容已实测**：旧 `model_args`（无两个新键）/ 新 `model_args` 都能 `BaseModel.from_pretrain`，`strict=True` 无 missing/unexpected；`nn.LSTM(batch_first=)` 不改变参数名与形状（537,697 参数不变）。
- 数据集默认路径：`AVSpeechTrainDatasetWithNear`（默认 16 kHz）**逐位一致**；plain `AVSpeechTrainDataset`（默认 **8000**）**不一致**（P1-2）。
- **`paper_causal=True` 不是"纯 padding 开关"**：stride=2 的 3 个 downsample 层输出长度 50→48，且末帧输入对输出**零影响**（P2-3）。
- `verify_dataset2.py` 的 H3/H4 是**真断言**（读取实现产物），实跑 `TOTAL FAILURES: 0`；`pytest tools/repro/tests/test_metrics.py` → 10 passed。
- 唯一"会静默少报/多报"的实质风险在 coverage 产物（P1-1）与 plain 数据集默认值（P1-2），其余为口径/文档级。

## 1 发现表

| 编号 | 文件:行 | 严重度 | 现状 | 问题 | 修复建议 | 证据命令 |
|---|---|---|---|---|---|---|
| F1 | `audio_visual_test.py:555-625`（`_check_coverage`）、`:654` | **P1** | `--limit 3`（数据集 100）时 coverage 报 `verified: true, planned_total: 3`；落盘的 `metrics_counts.json.coverage` 里**没有** `limit`/`dataset_len`/`subset` 字段 | `--limit` 子集被盖上"覆盖率已验证"章：下游只读 `metrics_counts.json` 的人无法区分"全测试集 5000 条"和"前 500 条 smoke"，与 docstring「`--limit` 的数不得当全集引用」直接矛盾（假阴性） | report 里补 `"limit"`, `"dataset_len"`, `"subset": planned_total < dataset_len`；`verified` 只在 `not subset and not problems` 时为 true（或新增 `--allow-limit` 才允许 subset 的 verified），并把这三个字段写进 `metrics_counts.json` | 见 §3 T1（`_check_coverage` 单测：`{"case":"limit-subset", ... "verified": true, "planned_total": 3}`） |
| F2 | `look2hear/datas/avspeech_dataset.py:129`（`sample_rate=8000`）+ `:230`（`rand_start/self.sample_rate`） | **P1** | plain `AVSpeechTrainDataset` 默认 `sample_rate=8000`，新式子是 `rand_start/8000*25`，旧式子是硬编码 `rand_start/16000*25` → `mouth_start` **翻倍** | 该类（及 `AVSpeechDataModule:357`）的**默认路径不是零漂移**，而 docstring `:118` 写"the defaults reproduce it exactly"、模块注释 `:29-31` 写"mouth crops 以 16 kHz 为基准"——两处都假；更糟的是 `_warn_mouth_sample_rate(8000)` 会警告"crops 假设 16 kHz"，而同一次构造里代码按 8 kHz 算，警告与算术互相打脸 | 明确口径二选一：①mouth 映射锚定 `MOUTH_AUDIO_SR`（保持旧行为，`mouth_fps` 只改帧率），②承认 8 kHz 才是对的，则把该类默认改 16000 或显式传参并改掉 docstring/注释；同时把 warning 文案改成"本类 sample_rate≠16000，mouth_start 将按 sample_rate 计算（旧版硬编码 16000）" | `ds_eq2.py`：默认 8000 → plain 哈希 `9255e1cf...`（且 `ValueError`），显式 16000 → 与 HEAD **同为** `819abe6f19fd8b8d` |
| F3 | `look2hear/models/Falcon.py:707-712`（注释）+ `:772/:808/:821`（三处开关）；studio `:712-…` 同构 | **P2** | 注释称 True 表示三处"follow `self.causal`，i.e. left-aligned causal padding" | 底层 `ConvNormAct`（`:11`，`causal=True`）是"两侧 padding=k-1、再裁掉尾部 k-1"：stride=1 时恰好等价左对齐（已实测因果），但 **stride=2 时输出长度 50→48（k=5）/ 20→19（k=3），且末帧输入对任何输出都没有影响**——每级 downsample 丢 2 帧，decoder 里 `F.interpolate(..., size=residual.shape)` 把它静默重采样掉。所以 `paper_causal=True` 改的不只是 padding，还有时间栅格 | 要么给 stride>1 实现真正的 left-only padding，要么在注释里写清"改变每级时间长度、由 interpolate 吸收"；并加一条回归测试（扰动末帧 → 输出不变**且** out_len 不变）；另外 `paper_causal` 没有任何 config key（`configs/*.yml` 无此键），只能改 hparams 才能开，建议显式暴露或标注 "experimental" | `causal_probe2.py`：`{"k":5,"stride":2,"causal_out":18,"sym_out":20,"changed_positions_tail":[]}` |
| F4 | `look2hear/losses/pit_wrapper.py:124`、`:289` | **P2** | 新增 `.reshape(B)` 要求 loss 返回 `[B]`/`[B,1]` | 标量返回的 loss（旧代码靠 broadcast 静默写入每一行）现在直接 `RuntimeError: shape '[3]' is invalid for input of size 1`。仓库内 `pw_pt` 无实际调用（所有 config 都是 `pw_mtx`），但这是未记录的破坏性变更 | 改成 `flat = loss.reshape(-1)`，`assert flat.numel() == B` 并给出"loss 必须返回 [B] 或 [B,1]，got ..."的显式错误；或在 docstring 里声明该契约 | `pit_edge.py`：`{"scalar_loss_pw_pt": "RAISED", "err": "RuntimeError: shape '[3]' is invalid for input of size 1"}` |
| F5 | `docs/verification/scripts/verify_losses_04.py:235-239`（不在本轮 diff，但被 `sisnri.py` 修复推翻） | **P2** | 该脚本仍断言 `SISNRi()` 抛 NameError，并用 `SISNRi.__new__` 绕过 `__init__` | `sisnri.py` 修好后该脚本会打印"实例化成功 (意外)"，成为"文档化 bug 仍在"的反向证据；且 `__new__` 绕过是旧 workaround，现在可直接构造 | 删掉 try/except 与 `__new__`，改成 `SISNRi()` 正常构造 + 断言 `isinstance(loss, nn.Module)`（与 `test_metrics.py:249-273` 对齐），措辞改成"已修复（W1-05）" | `sed -n 235,246p docs/verification/scripts/verify_losses_04.py` |
| F6 | `audio_visual_test.py:496-502` | **P2** | 新增 worker 内 `if n_done != expected_local: raise` | `n_done` 在每个循环迭代末尾 `+=1`，与 `len(local_indices)` 恒等；真正的漏跑（异常/进程死）根本走不到这行 → 该断言近乎恒真，只能防未来的 `continue` | 保留但注释说明"只为防未来 continue"，别把它当覆盖率保证；或改成记录 `set(evaluated_keys)` 与 `local_indices` 一一对账（对账成本几乎为零） | 读码 + §3 T1 单测（缺行只被 merge 侧 `_check_coverage` 抓到） |
| F7 | `audio_visual_test.py:586-601` | **P2** | 只比较"每个 rank 的行数"和"总行数" | 同一 rank 内"重复评估 A、漏掉 B"、或跨 rank 重复同一句，行数可以完全对上 → 假阴性（合并 CSV 里出现重复 `snt_id`，仍报 verified） | merge 后加 `dup = utterances["snt_id"].duplicated().sum()`，非 0 即计入 problems（键即 `mix[idx][0]` 文件名，天然唯一） | 读码；`_check_coverage` 只吃 `rows_per_rank`/`n_rows`，没有集合信息 |
| F8 | `evaluated_mac_params.py:129` | **P2** | `except (FileNotFoundError, KeyError, ValueError)` 才回退零 embedding | 文档承诺"backbone 缺失就回退"，但权重文件**存在却损坏/截断**时 `torch.load` 抛 `RuntimeError`，脚本直接崩，与承诺不符 | 追加 `RuntimeError`（并把原始异常打进 WARNING），或先 `torch.load` 探测再决定 | 读码；`pretrain_zoo/` 不存在 → 走 fallback（实测 warning 命中） |
| F9 | `evaluated_mac_params.py:186`（`V_FRAMES = 50`）、`:218/:222` | **P2** | 假嘴部张量写死 50 帧 | 50 = 2 s × 25 fps，与 `configs/Falcon.yml` 的 2 s 窗口绑定；一旦 segment/sample_rate 改了，`mouth_embedding` 与 `audio_mixture` 帧数不匹配（ptflops 前向直接报错或口径失真），脚本没有断言 | 从 `datamodule.data_config` 的 `segment`（或 audio 长度）推 `V_FRAMES`，并断言 `model(a, v)` 形状 | 读码 |
| F10 | `tools/repro/phase2/run_lrs2_paper_2gpu_smoke.sh:1-4` | **P2** | 头注释把 40 叫 "global batch = 2×4×5" | 同一轮里 `run_lrs2_paper_volcano_8gpu.sh:114-125` 已把 `global batch` 定义为 `bs×NPROC`（8），`effective` 才乘 accum（40）——两个脚本对同一术语给了相反定义，正是"引用数字时搞混"的源头 | smoke 头注释改成 "effective batch 40（global batch 8 × accum 5）"，与 volcano 的措辞统一 | 两文件对照 |
| F11 | `tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh:117-123` | **P2** | 新增 `steps/epoch ≈ train_samples/(bs×NPROC×accum)` | 该式假设 `len(train_dataset) == train_samples`（默认 `respect_index=False` 成立）。若本轮引入的 `respect_index=True` 被打开，`__len__` 变成 `len(self.mix)`（如 40000），打印的步数会差 len(mix)/train_samples 倍——而脚本不读数据集，无法自检 | 打印里注明"assumes len(train_set)==train_samples (respect_index=False)"，或在渲染配置时若发现 `respect_index` 为真则打 WARNING | 读码 + `avspeech_dataset_near.py:185-188` |

## 2 逐项通过清单

1. **默认路径零漂移 —— 通过（带 1 个例外，见 F2）**
   Falcon 侧：`paper_causal`/`video_lstm_batch_first` 默认 `False`，`self.causal if self.paper_causal else False` 在 `self.causal` 已赋值（`:718`）之后求值，短路只回落到字面 `False`，与旧代码同；`Entity` 无关。实测 HEAD vs 工作区：525 keys、参数哈希、输出 `out_sum` 全等。数据侧：`int((rand_start/self.sample_rate)*self.mouth_fps)` 在 `sample_rate=16000, mouth_fps=25.0` 时与旧 `int((rand_start/16000)*25)` 逐位同（25.0==25，同一浮点运算链）；`AVSpeechTrainDatasetWithNear` 默认路径哈希 `819abe6f19fd8b8d` 与 HEAD 完全一致。`_fit_mouth_frames` 在 `got==expected` 时原样返回，不改数值。**例外**：plain `AVSpeechTrainDataset` 默认 8000 → F2。`None`/`False` 混淆检查：`respect_index`/`pad_mouth` 只做真值判断，`None` 等价 False，无 `or` 短路误用；`int(segment*mouth_fps)` 的截断方向与旧 `int(segment*25)` 一致。

2. **state_dict 兼容 —— 通过**
   `nn.LSTM(batch_first=)` 只影响 I/O 布局，不进入 `_parameters`：`batch_first=False/True` 两种构造的 `n_params` 都是 537,697，state_dict 键集合与哈希相同（`bfe6024bda7b7095`）。`get_model_args()` 两个新键（`Falcon.py:1545-1546`）全部在 ctor 签名内（实测 `not_in_ctor: []`）；用**旧** `model_args`（并手工剔除 `audio_params/video_params` 里的新键）与**新** `model_args` 分别 `BaseModel.from_pretrain`，都无 warning、无 missing/unexpected；`load_state_dict(HEAD_sd, strict=True)` 通过；`Falcon(**falcon.get_model_args())` 二次往返参数/键不变（523,113 / 518 keys）。studio 拷贝的 hunk 与 look2hear 逐字相同（`paper_causal` 出现 9 次，含 3 处开关），`studio/falcon_runtime.py:190` 从 `configs/*.yml` 的 `audionet_config` 构造、`:206` strict 载入，配置里没有新键 → 默认 False，不受影响。

3. **异常与边界 —— 基本通过（1 条 P2）**
   `_promote_single_source`：2-D/2-D→升秩，3-D/3-D→原样，混排→`ValueError`（实测 `ests.ndim=2/targets.ndim=3` 报错）；`_call_pair_loss` 用 `"shape" not in str(exc)` 过滤，实测 `PairwiseNegSDR` 的 TypeError 文案是 `"Inputs must be of shape [batch, n_src, time], got ..."`（含 shape）→ 命中；只重试**一次**，不存在无限重试；含 "shape" 的无关 TypeError 被重试一次后原样抛出（实测 `calls=2`，最终消息保留）→ 不吞错。`_fit_mouth_frames` 在 `num_workers=2` 下实测：`pad_mouth=False` → 主进程收到 `ValueError`（"Caught ValueError in DataLoader worker process 0" 前缀 + 原始消息），`pad_mouth=True` → `mouth [2,50,88,88]` 正常 stack；`got>expected` 分支因切片不可达（已注明），`got==0` 不会进 pad 分支。`respect_index=True`：`__len__` = `len(self.mix)`（fixture 8→7 条，与 `sources` 等长），`idx=-1`→末条，`idx=len`→`IndexError: ... idx=7 out of range for 7 samples`，边界正确。唯一遗留：`.reshape(B)` 的标量兼容性（F4）。

4. **覆盖率断言 —— 通过（但产物口径有 P1，见 F1/F6/F7）**
   单测 `_check_coverage`/`_merge_results`：①不均匀 4 卡 `[3,3,2,2]` → verified；②零句 rank `[2,0,0]`（world_size>planned）→ verified（rank 也会写出 header+avg/std，`read_csv` 不报 EmptyDataError，`rows_per_rank=0` 与 planned 0 相符）；③缺 1 行 → `RuntimeError`（per-rank 与总数两条都报）；④同③加 `--allow-partial` → 仅 WARNING 且 `verified:false` + `reason`。worker 侧 `planned_total` 断言把"limit 先切、再 `indices[rank::world_size]`"钉住，`planned_per_rank` 用 `range(planned_total)[r::]` 复算，与真实切片等价。**假阴性**剩下的两处：`--limit` 子集被盖 verified 章（F1）、重复键+漏键的行数守恒（F7）；worker 内 `n_done` 断言近乎恒真（F6）。

5. **`evaluated_mac_params.py` —— 通过（2 条 P2）**
   实测 `CUDA_VISIBLE_DEVICES=` + `--device auto` → **exit 2**，stderr 打印 `NO_GPU_MSG`（与 docstring/`--help` 承诺一致）；`--device cpu` → 打印 WARNING 后跑通；`--device cuda`（真卡）→ 跑通。两种设备都是 `MACs = 10.611 GMAC`、`Params = 0.538 M`，与文档 10.611/0.538/15.897/26.51 的三口径说明自洽，没有"承诺与行为不一致"。`CONFIG_PATH` 改为脚本同目录，`print(config=...)` 输出绝对路径（从任意 cwd 都能跑）。`pretrain_zoo/` 缺失 → 命中 fallback，打印 WARNING 并用 `(1,512,50)` 零 embedding 完成计数；数值与有 FRCNN 时一致（ptflops 只统计 audio model），符合承诺。P2：异常捕获范围（F8）与 `V_FRAMES` 写死（F9）。

6. **shell/config —— 通过（2 条 P2）**
   `bash -n` 两个脚本都 OK；`set -euo pipefail` 下 `NPROC="${NPROC:-8}"` 先赋值再校验（`:32-58`），heredoc 里 `nproc=int(sys.argv[2])` 不依赖 shell 变量，`accum` 取 `int(...get(...,1) or 1)` → `null/0` 都落到 1，无未定义变量。算式自洽：v2 配置 `bs=4, accum=1`、`NPROC=8` → `global=32, effective=32` 不打 WARNING；换 `accum=5` → `effective=160` 打 WARNING；`steps=10000//32=312` 与 Lightning+DistributedSampler 把 10000 均分到 8 卡后的步数一致（旧式漏乘 accum，确实高估）。`configs/*.yml` 用 `git diff -U0` 确认**只有注释行**（`LRS2_2Mix.yml` +1、`_Paper_8gpu.yml` +1、`_Paper_v2.yml` +3），无键值变化；注释里的数字已核对：2gpu 配置 `bs4/accum5`（4×2×5=40、改 4 后 4×2×4=32）、legacy 8gpu `bs5/accum1`（8×5=40）、v2 `-SI-SNR/beta1000/seed42`、smoke `pairwise_neg_snr/beta200/无 seed`、`train_samples=10000` 有放回 → `1-e^{-0.25}≈22%` 唯一覆盖，全部对得上。P2：术语（F10）与步数公式前提（F11）。

7. **`verify_dataset2.py` —— 通过（新行为被真断言）**
   实跑 `TOTAL FAILURES: 0`。H3a 断言 `ValueError` 且消息含 `"25 frame(s)"` 与 `"50 are required"`（实跑原文一致）；H3b 断言 `i1[2].shape[0]==50` **且** 尾 25 帧逐帧等于第 24 帧（`np.array_equal`，真读张量，不是恒真）；H4 从返回 clip 的像素反解帧号（`(px*0.165+0.421)*255`，即 fixture 把帧号写进像素），实跑 `mouth_start(from clip)=31 == expected 31`，并同时断言旧的 16000 口径会得到 15（反例保留）与 `warnings` 命中 `sample_rate=8000`；F2/J1 从" documenting 0-dim bug"改成 F5-guard 的 1-D 且带精确长度 `(48000,)`；I2 从 `extra == ["train_samples"]` 放宽为"所有 extra 都有默认值"（真检查，实跑 `extra=['train_samples','respect_index','mouth_fps','pad_mouth']`）。没有把断言改写成恒真。

## 3 建议的修法清单（可直接派给 owner）

- **T1（P1-1，`audio_visual_test.py`）**：`_check_coverage`（`:555`）读取 `plan["limit"]`/`plan["dataset_len"]`，在 `report`（`:604`）里补 `"limit"`, `"dataset_len"`, `"subset": planned_total < dataset_len`；`verified = (not problems) and not subset`（subset 时即便无 problems 也 `reason="limited subset"`），并让 `_merge_results`（`:654`）把这三个键写进 `metrics_counts.json`。验收：`--limit 3`（数据集 100）→ `verified:false, subset:true`；全量 → `verified:true`。
- **T2（P1-2，`look2hear/datas/avspeech_dataset.py`）**：定口径后二选一。若沿用旧行为：`:230` 改回锚定 `MOUTH_AUDIO_SR`（`rand_start / MOUTH_AUDIO_SR * self.mouth_fps`），`_warn_mouth_sample_rate` 只作提示；若承认 8 kHz 正确：改 `:129` 默认值并在 `:118` docstring 与 `:29-31` 模块注释里删掉"defaults reproduce it exactly / crops 以 16 kHz 为基准"的表述，改 `_warn_mouth_sample_rate` 文案为"mouth_start 将按 sample_rate=x 计算（旧版硬编码 16000，数值会变）"。验收：`ds_eq2.py` 默认构造下 HEAD 与工作区哈希一致（或明确记录差异）。
- **T3（P2-3，`look2hear/models/Falcon.py:707-712` 及 studio 同构位置）**：把注释从"left-aligned causal padding"改成"三处改跟随 `self.causal`；`ConvNormAct(causal=True)` 对 stride>1 会在裁剪后少 2 帧，由 decoder 的 `F.interpolate` 吸收"；并加一条测试：`paper_causal=True` 下 `AudioBlock` 的 downsample 输出长度与末帧敏感性（期望 out_len 不变、扰动末帧不改任何输出）；若要让 `paper_causal=True` 真能复现论文，再给 `ConvNormAct` 加 stride>1 的 proper left-pad。
- **T4（P2-4，`look2hear/losses/pit_wrapper.py:124`、`:289`）**：`.reshape(B)` → `flat = loss.reshape(-1)` + `if flat.numel() != B: raise ValueError("pw_pt loss must return [B] or [B,1], got shape ...")`；`test_metrics.py` 补一条标量 loss 的报错用例。
- **T5（P2-5，`docs/verification/scripts/verify_losses_04.py:235-246`）**：删掉 NameError 断言与 `__new__` 绕过，改成正向断言（对齐 `test_metrics.py:249-273`）。
- **T6（P2-6/7，`audio_visual_test.py:496-502`、`:586-601`）**：worker 内改记 `evaluated_keys` 集合并与 `local_indices` 对账（或注释声明"仅防 continue"）；merge 侧加 `duplicated("snt_id")` 检查。
- **T7（P2-8/9，`evaluated_mac_params.py:129`、`:186`）**：`except` 增加 `RuntimeError` 并把原异常打进 WARNING；`V_FRAMES` 从 `datamodule.data_config` 的 `segment`/`sample_rate` 推导并断言 `model(a, v)` 形状。
- **T8（P2-10/11，两个 `.sh`）**：smoke 头注释统一术语为 "effective batch 40（global 8 × accum 5）"；volcano 的步数打印加 "assumes len(train_set)==train_samples (respect_index=False)" 或在 `respect_index` 为真时 WARNING。

## 4 未覆盖范围

- **`paper_causal` / `video_lstm_batch_first` 没有任何测试或 verification 脚本**：本轮 diff 只保证"默认 False 零漂移"，两个 True 分支的正确性（因果性、论文对齐、LSTM 真沿 25 fps 递归后的数值）无断言；我也只做到"能跑通、形状/参数量不变"。
- **`respect_index=True` 无任何断言**：`verify_dataset2.py` 只覆盖 mouth 三个 knob 中的两个；`respect_index` 的 `__len__`/打乱语义我用手工 fixture 验了（len=7、7 个唯一 key、`-1`/越界正确），但仓库内没有回归保护。
- `video_lstm_batch_first=False`（发布行为）下 LSTM 是**沿 batch 轴递归**（`[B,T,C]` 被当成 `[T=B, batch=T]`），即不同语句之间互相耦合；这是既有 bug 的事实描述，但我没有验证它对训练/推理数值的实际影响量级（属模型侧审查范围）。
- `studio/falcon_model/models/Falcon.py` 只做了与 look2hear 的逐字 hunk 比对 + 加载路径读码，没有单独 import studio 副本跑前向（`studio/falcon_runtime.py` 的 strict 载入无真 checkpoint 可跑）。
- `audio_visual_test.py` 的端到端（真 ckpt + 多卡）没跑：`Experiments/checkpoint/Falcon` 下无 `.ckpt`；覆盖面靠 `_check_coverage`/`_merge_results` 单测 + 代码路径核对。
- `normalizations.py` 仅文档（已用 `git diff -U0` 确认为纯 docstring/注释，无代码行），其中"30 个 cLNhw / 60 个 LayerNormalization4D / -40.2 dB 非因果"等量化结论未复算（依赖 `tools/repro/experiments/ln_variant.py`）。
- 报告只审"diff 是否正确"，未审 `docs/STATUS.md`、`README*`、`project_page/index.html`、`space/README.md`、`docs/understanding/`（不在此次 diff 清单内）。
