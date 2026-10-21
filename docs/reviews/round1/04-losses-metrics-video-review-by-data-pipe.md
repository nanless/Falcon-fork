# 评审：`04-losses-metrics-video.md` —— 评审人 `data-pipe`

- **评审对象**：`docs/understanding/04-losses-metrics-video.md`（1056 行）
- **sha256（评审开始）**：`d02806e2b790eef1cca9c4e78669af4f99fd4ce76a1cbc5217e985bfbafaa810`
- **sha256（评审结束复核）**：`d02806e2b790eef1cca9c4e78669af4f99fd4ce76a1cbc5217e985bfbafaa810`（未变；本次评审未修改 04 文档与任何源码）
- **评审人**：`data-pipe`（数据管线域；R2 交叉评审）
- **评审范围**：按 Lead 指定的数据/评测/视觉接口交叉点逐条核对到源码；损失数学与指标数值属其他域，只做抽样核对。重点：①评测是否使用 DataLoader、dataset 输出形状、整句/对齐、`make_sets`/`use_near=False`；②88×88 裁剪归属与训练/评测/Studio 三处一致性；③FRCNN `[B,1,T,88,88]→[B,512,T]` 与 50 帧关系、`video_params.stride:2` 死配置；④P0-3（`ckpt["state_dict"]` 与 `falcon.pt` 格式）；⑤Q4（mono wav）。
- **独立证据来源**：
  - `docs/understanding/05-data-pipeline.md` §3.5/§3.7/§5.4/§8.6；
  - `docs/verification/data-pipe-supplement.md`（本轮新增；mono-eval P0 用真实模型复现）；
  - 复现脚本 `/tmp/dp-verify/verify_p0_eval.py`，日志 `/tmp/dp-verify/logs/verify_p0_eval.log`；
  - 04 自带脚本/日志 `docs/verification/scripts/verify_*_04.py`、`docs/verification/runs/04-*.log`（5 个日志均存在，已交叉引用）。

---

## 1. 结论

**需修订（P0：0 条；P1：6 条；P2：8 条）。**

总体评价：这是一份质量很高的子系统文档——50 条断言 + 12 条 Q + 5 个验证脚本/日志齐全；在我复核的数据/评测/视觉接口相关结论中，**绝大多数与源码一致**（评测确实不用 DataLoader、逐条 `test_set[idx]`、整句音频、batch=1；88×88 裁剪确实在 dataset 预处理而非 FRCNN；FRCNN 时间维不降采样；`video_params.stride` 确实是死配置；README 评测命令不可直接复现；`ALLMetricsTracker` 的 "sdr" 实为 −SNR）。因此**没有发现需要推翻结论的 P0 事实错误**。

需要修订的 P1 集中在五处：
1. **Q1/B5 关于 `falcon.pt` 的"必然 KeyError"风险可以关闭**：仓库内 `studio/falcon_model/PROVENANCE.md:36-40` 明确说明发布文件持有 `state_dict` 键，04 遗漏了这条仓库内证据；README:168 命令的真实阻断点是 conf.yml 的 `audionet_name`（B2）与 `${DATA_ROOT}`（B3），不是 ckpt 格式。
2. **mono 风险后果被低估**：本轮用真实模型坐实——单声道 wav → eval Dataset 退化为 0 维 → `mix[None]` 为 `(1,)` → `torch.stft` 抛 `RuntimeError`，且该异常不被评测脚本捕获；**训练期 `val_dataloader` 使用同一 dataset，也会在第一次验证时崩溃**（因此"能训练出论文模型"反证真实数据应为多声道）。
3. **`dtype=float64` 结论缺少前提**：它来自验证 fixture 用 uint8 `.pt`；float32 `.pt` 时输出 float32；真实 `.pt` 存储 dtype 未知。
4. **一处证据行号错误**（`avspeech_dataset_near.py:166-172` 应为 169-174）。
5. **`tensors_to_device` 返回类型误述**（返回 `list` 而非 5 元组）。
6. **遗漏"评测整句 vs 训练 2 s 窗口"的指标口径讨论**。

---

## 2. 断言逐条核对表（A01–A50）

> 图例：✅ = 本轮读源码核对（行号/语义一致）或与 04 自证日志、我的独立验证一致；❓ = 属于损失/指标数值或第三方包细节，本轮未复算，未发现矛盾；❌ = 发现错误。
> 说明：04 的断言表本身**没有**一条是 ❌；下列问题主要出现在正文/表格叙述（§3 已列）。

| # | 结论 | 核对内容 / 备注 |
|---|---|---|
| A01 | ✅ | `matrix.py:372-380` 确为模块级单例：`pairwise_neg_{sisdr,sdsdr,snr}` 等 |
| A02 | ✅ | `PairwiseNegSDR` 在 `matrix.py:188-232`；输出 `[B,est,tgt]` 负 dB，符号 `−10log10` 一致 |
| A03 | ✅ | `zero_mean` 分支 `matrix.py:204-208`；数值由 04 log 自证 |
| A04 | ✅ | SNR/SI-SDR 尺度差异源自投影/非投影（`matrix.py:212-232`），与结论一致 |
| A05 | ✅ | `SingleSrcNegSDR` `matrix.py:235-281`；`assert reduction != "sum"` 在 :239 |
| A06 | ✅ | `self.EPS = 1e-8` 写死（`matrix.py:246,292`），构造参数 `EPS` 被忽略——属实（小 bug 判断正确） |
| A07 | ✅ | `MultiSrcNegSDR` 同索引逐源平均，无排列匹配（`matrix.py:294-326`） |
| A08 | ✅ | `PITLossWrapper` 6 种 `pit_from` 白名单在 `pit_wrapper.py:162`；`pw_mtx` 直接调用 loss_func（:170-171） |
| A09 | ✅ | 三模式等价结论由其日志自证；代码路径一致 |
| A10 | ✅ | `pw_mtx_broadcast` 的 0.5×第二损失在 `pit_wrapper.py:233-234` |
| A11 | ✅ | `threshold_byloss` 语义（保留 `min_loss>-30`，全过滤时退回全体）在 `pit_wrapper.py:227-230`；属实且是容易误读的点 |
| A12 | ✅ | `n_src<=3` 全排列 / >3 匈牙利：`pit_wrapper.py:266-274,303-312`（`linear_sum_assignment`） |
| A13 | ✅ | `return_ests=True` 返回重排后的估计：`pit_wrapper.py:231,260-264` |
| A14 | ✅ | `new_PITLossWrapper`（:6-150）：`forward(ests, ests_bn, targets, epoch,...)`（:22）、`pw_mtx` 分支 4 参调用（:25-26）、不归约 `return pw_loss`（:75）——与 `PairwiseNegSDR` 不兼容属实 |
| A15 | ✅ | `MixITLossWrapper`（`mixit.py:102`）、`parts_mixit_gen`（:250-260）；数值由 04 log 自证（14 个有序划分） |
| A16 | ✅ | 全仓库 grep 无 MixIT 调用者——属实（死代码） |
| A17 | ✅ | NASS 公式与实现（`nass_contrastive_loss.py:52-77`）一致；K=1 分支见 P2-2 |
| A18 | ✅ | `n_valid` 无条件 +1（:77），分母恒为 B；`:79-81` 仅 B=0 触发——属实 |
| A19 | ✅ | 损失内部不 detach；system 侧 positive/negative 均 `no_grad`（`av_litmodule_nass_v2.py:106-109,129-131`）——属实 |
| A20 | ✅ | β 调度 `_get_beta` 与 config 值一致（`av_litmodule_nass_v2.py:66-70`；`configs/Falcon.yml:108-110` 实为 margin/alpha/beta 区段，值 0.0/1.0/200 正确） |
| A21 | ✅ | `total_loss = pit_loss + weighted_cl`（:137-138），日志 `train_spk_contrast_loss` 记 `weighted_cl`（:175）——属实 |
| A22 | ✅ | `sisnri.py:6` `super(Loss, self)` 引用未定义 `Loss` → `NameError`，类不可实例化——属实 |
| A23 | ✅ | `audio_visual_test.py:47,208` 只用 `ALLMetricsTracker`；其 sdr 来自 `PairwiseNegSDR("snr")`（`allwrapper.py:27,38-46`）——属实 |
| A24 | ✅ | 逐句 `row["sdr"]=sdr.item()`（:62） vs 累加 `-sdr.item()`（:72）→ 逐句/avg 符号相反——属实 |
| A25 | ✅ | `allwrapper.py:50` 实参顺序 `(estimate, clean, 16000)` vs pypesq `(ref, deg, fs)` → 传反——属实（外部包合同由 04 §8.3 核对） |
| A26 | ✅ | STOI 传参 `(clean, estimate)` 正确（`allwrapper.py:53-58`）——属实 |
| A27 | ✅ | `wrapper.py:30,33` `zero_mean=False`；:44-45 `-fast_bss_eval.sdr_pit_loss(...)`——属实 |
| A28 | ✅ | 全仓库仅 `audio_visual_test.py` import `ALLMetricsTracker`；`MetricsTracker`/`SPlitMetricsTracker` 无入口——属实 |
| A29 | ✅ | FRCNN resnet 分支参数 11,186,688、输出 `[B,512,T]`（本人独立实测 `(2,1,50,88,88)→(2,512,50)`） |
| A30 | ✅ | `frcnn_videomodel.py:90-95` `train()` 无 `return`，故 `.eval()` 返回 None——属实 |
| A31 | ✅ | `update_frcnn_parameter`（:106-118）跳过 `tcn`、加载 strict、置 `requires_grad=False`——属实 |
| A32 | ✅ | 96→88 裁剪在 dataset 预处理与 Studio `[:,4:92]`、不在 FRCNN 内——与我的独立验证一致（Studio vs val 管线 `max|diff|=0`） |
| A33 | ✅ | `T_out=T_in` 属实；融合层 nearest 插值 `Falcon.py:257-258`；"25fps→12.5fps 下采样不存在"正确 |
| A34 | ✅ | 只有 `FRCNNVideoModel` 被 config 选中：`configs/Falcon.yml:2-8`、`studio/configs/Falcon.yml:3`、`studio/configs/FalconBig.yml:166`——属实 |
| A35 | ❓ | shufflenet 64×64 崩溃不复算（非本域；其日志 `04-video.log` 存在） |
| A36 | ✅ | `_find_best_ckpt` 优先级：`audio_visual_test.py:78-108` 与描述一致 |
| A37 | ✅ | `best_k_models.json` 确为 `${ROOT}/Falcon/epoch=*.ckpt` 占位符（5 条）；仓库该目录仅 conf.yml+json → 默认评测 `FileNotFoundError`——属实 |
| A38 | ✅ | `sorted(glob("epoch=*.ckpt"))[-1]`（:101-106）字符串排序 → 742 会排在 1000 之后被选为"最新"——属实 |
| A39 | ✅ | `_load_models_from_ckpt`（`audio_visual_test.py:111-172`）：要求 `ckpt["state_dict"]`（:119-120）、audio `strict=False`（:139）、video pretrain 置 None（:149-152）、video `strict=True`（:160）——代码属实；但"falcon.pt 格式未知"的推断见 P1-1 |
| A40 | ✅ | `Experiments/checkpoint/Falcon/conf.yml:88` `audionet_name: SpeakerContrastiveV2`；`look2hear/models/__init__.py:1-2` 只有 `Falcon`/`FalconSpeakerContrastive` → `AttributeError`——属实 |
| A41 | ✅ | `Separator.__init__` 的 `dict.pop("repeats"/"shared")`（`Falcon.py:991-996`）污染传入 config；`audio_train.py:54-58` 直接传 config——属实（参数 49,081 由 04 日志自证；建议补 `audio_train.py:82-84` 的落盘行） |
| A42 | ✅ | `process_worker` 不用 DataLoader、按 `rank::world_size` 切片、`test_set[idx]`、batch=1（`audio_visual_test.py:189-193,228-229,249-250`）——属实；注意正文"返回 5 元组"错误（P1-5） |
| A43 | ✅ | 评测读整段音频（不截 2 s）——属实（`avspeech_dataset.py:179-187`）；"嘴部 25 fps"来自数据约定（dataset 不校验 fps，见 P2-5） |
| A44 | ✅ | mono wav → `mean(-1)` 0 维（`avspeech_dataset.py:193-194` vs 训练集 guard `near.py:176-177`）——属实；**后果低估**见 P1-2 |
| A45 | ✅ | 结果路径 `cwd/Experiments/checkpoint/<exp_name>/results/metrics_{rank,combined}.csv` 与合并/删除逻辑（`audio_visual_test.py:196-208,284-313`）——属实 |
| A46 | ✅ | 合并时把每 rank 的 avg/std 行当普通样本再次平均（:291-307）——属实 |
| A47 | ❓ | MACs 10.611 G 未复算（本域外；`04-macs.log` 存在，参数 0.538 M 与 Lead 验证/我构建的 537,697 一致） |
| A48 | ✅ | README:100/103/206 数字与 537,697 + 11,186,688 = 11,724,385 相符——属实 |
| A49 | ❓ | 视觉 MACs 未复算（依赖 ptflops；本域外） |
| A50 | ✅ | `evaluated_mac_params.py:48,51-53` 要求 CUDA 且保留 config 的 pretrain 路径——代码属实 |

**小结**：50 条断言中 ✅ 45 条、❓ 3 条（A35/A47/A49，均本域外未复算）、❌ 0 条。问题集中在正文叙述行（见 §3），不在断言表。

---

## 3. 错误清单

### P0（事实错误）——0 条

未发现会让复现结论方向错误的 P0 事实错误；04 识别的项目级 P0（B2 类名不匹配、B8 sdr 列非 SDR、B4 无权重）均成立。

### P1（重要遗漏或误导）——6 条

**P1-1｜Q1/B5：`falcon.pt` "是否 KeyError"可从仓库内证据关闭为"非必然 KeyError"**（04:534-537, 600, 739, 962）
- 04 结论："`_load_models_from_ckpt` 只认 `ckpt["state_dict"]`；Studio 用 `ckpt.get("state_dict", ckpt)` 兼容两种；沙箱无法访问 HF → 未解（Q1）"。
- 遗漏证据链：`studio/falcon_model/PROVENANCE.md:36-40` 明确写道发布的 `falcon.pt`/`falcon-big.pt` **"They hold `state_dict`, the architecture name and the epoch"**；`README.md:107` 说文件只含 `audio_model.*`/`video_model.*` 权重；`studio/falcon_runtime.py:160-161` 用 `ckpt.get("state_dict", ckpt)` 后按 `audio_model.`/`video_model.` 前缀拆分并 `strict=True` 加载（:162-169）。
- 推论：`audio_visual_test.py:119-120` 的 `ckpt["state_dict"]` 与发布文件格式**兼容**，不会因"裸 state_dict"而 `KeyError`；README:168 命令实际失败点是 **B2**（`Experiments/checkpoint/Falcon/conf.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 在 `look2hear/models/__init__.py:1-2` 不存在 → `AttributeError`）与 **B3**（`${DATA_ROOT}` 未展开），不是 ckpt 格式。
- 建议：Q1 改为"已由仓库文档解答（PROVENANCE 声明；残留：未实测文件头）"；B5 从"断点"降级/删除；§2.4-E 行 600 由 "❓ 取决于格式" 改为 "❌ 因 B2/B3 失败"。

**P1-2｜mono wav 的后果被低估：不仅是"形状错误"，而是评测命令与训练期验证双双崩溃**（04:741 B7、965 Q4、946 A44）
- 04 写"单声道数据会让 `mix[None]` 形状错误（需 data-pipe 确认…）"。
- 本轮用真实代码+真实模型复现（`docs/verification/data-pipe-supplement.md` §2；日志 `verify_p0_eval.log`）：
  1) `AVSpeechEvalDataset` item：`mixture.shape=()`、`source.shape=()`（`avspeech_dataset.py:193-194`）；`mix[None].shape=(1,)`；
  2) 真实 `FalconSpeakerContrastive`（537,697 参数）+ `FRCNNVideoModel`（11,186,688）复刻 `audio_visual_test.py:229-231` → `RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (128,128) at dimension 2 of input [1,1,1]`（链路 `Falcon.py:1095-1097` reshape `(1,)→(1,1)` → `:1110-1116` `torch.stft(n_fft=256,center=True)`）；
  3) 该异常**不被捕获**：`model(...)` 在 try（:233）之外，try 只捕 `ValueError` 且仅 "all zeros"/"nan"（:237-242）→ README:165/168 评测命令在第一个测试样本即中止；
  4) 训练期 `val_dataloader` 走同一 `AVSpeechEvalDataset`（`avspeech_dataset_near.py:286-290,309-318`；`av_litmodule.py:86-119`），`default_collate` 后 `mixture (1,)`、`mouth (1,F,88,88)` → 同一 RuntimeError：**若数据为 mono，训练第一次验证就崩**；因此"论文模型能训练出来"反证真实训练/评测数据应为多声道（或验证/评测路径被替换过）。
- 建议：B7 明确标注"若数据 mono 则 P0（阻断 README 评测）"；Q4 的"后果"部分关闭、只保留"真实数据通道数待数据侧确认"（与 05 Q1 对齐）；§3.4 增加"训练期验证同样受影响"。

**P1-3｜`dtype=float64` 缺前提条件**（04:474, 687）
- 04 写"返回 mouth.shape=(25,88,88), dtype=float64"、"mouth [T,88,88] float64"。
- 该结果来自其 fixture 把帧存成 **uint8**（`docs/verification/scripts/verify_eval_flow_04.py:177` `(np.random.rand(T,96,96,1)*255).astype(np.uint8)`）→ numpy 中 `uint8 - 0.0 → float64`。若 `.pt` 为 float32（torch.save 常见），输出为 **float32**（我的 05 A5/D3 实测）；真实 `.pt` 存储 dtype 未知（05 Q2）。
- 下游会统一：评测 `torch.from_numpy(mouth[None,None]).float()`（`audio_visual_test.py:230`）；训练 `mouth_in.type_as(mixtures)`（`av_litmodule_nass_v2.py:87`）→ FRCNN 输入始终 float32。
- 建议：改为条件表述"输出 dtype 取决于 `.pt` 存储 dtype（uint8→float64、float32→float32），FRCNN 前统一转 float32"，并注明 fixture 存储类型。

**P1-4｜证据行号错误（按 Lead 规则行号错=P1）**（04:468）
- 04 写"数据集在 `avspeech_dataset_near.py:166-172` 对整段 `[T,96,96]` 应用"。
- 实际：166 空行、167 注释、168 `mouth_start=...`、**169-174 才是管线调用与切片**（`self.lipreading_preprocessing_func(...)[mouth_start: self.fps_len+mouth_start]`）。
- 建议：改为 `avspeech_dataset_near.py:167-174`（或 `:169-174`）。

**P1-5｜`tensors_to_device` 返回类型误述**（04:565-566）
- 04 写"`tensors_to_device(test_set[idx], device)` 返回 5 元组"。
- 实际：`torch_utils.py:35-36` 对 tuple 返回 **list**（`return [tensors_to_device(tens, device) for tens in tensors]`），随后解包为 5 个变量；"5 元组"应改为"长度 5 的 list"（解包行为不受影响）。

**P1-6｜遗漏"评测整句 vs 训练 2 s 段"的指标口径讨论**（04: §4.6 只讨论 SNR/SDR）
- 数据域结论：评测数据集整句读取、无 2 s 截断/补齐（`avspeech_dataset.py:179-191`）；模型内部把视频帧 nearest 插值到音频帧数（`Falcon.py:257-258`），`STFTDecoder` 用 `istft(length=input_shape[-1])` 严格还原输入长度（`Falcon.py:1166,1182-1188`）——输出长度 == 输入 T（实测 `T=48513, F=75 → out (1,1,48513)`）。
- 这意味着 SI-SDR/SDR 是**整句口径**（不是论文/Studio 的 2 s 窗口口径）；04 讨论"sdr 列非 SDR"时未指出该窗口口径差异，对"论文指标能否复现"的判断不完整。建议补入 §4.6/§7（与 05 Q5 对齐）。

### P2（细节瑕疵）——8 条

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| P2-1 | 04:448 | "`audio_visual_test.py:170` … 体现了这一点"——:170 只是 `video_model.to(device).eval()` 未链式赋值，并不能"体现返回 None" | 改为"代码未链式赋值（`:169-170`），规避了该陷阱" |
| P2-2 | 04:251-252 | "文档注释的 `softmax_k(β·cos)` 只对 K>1 成立"——K=1 时 softmax 恒为 1，公式自然退化成立；`nass_contrastive_loss.py:68-69` 只是省算分支，不是语义差异 | 删除该"不一致"说法或改"代码对 K=1 省去 softmax 计算（结果等价）" |
| P2-3 | 04:559 | "`test_set = datamodule.make_sets[2]`（189）"——实际是 `_, _, test_set = datamodule.make_sets` 解包 | 贴原文 |
| P2-4 | 04:557-558 | setup 行号 `avspeech_dataset_near.py:277-293` → 实际 277-295（test 数据集在 291-295） | 改 `:277-295` |
| P2-5 | 04:685 | `torch.load(mouth.pt)[T,96,96,1]` 作为事实——仓库无 mouth `.pt` 生成器/资产，该 4-D channel-last 格式是从 `permute(0,3,1,2)` 反推 | 标 `[推断]` 并加入未解疑问（与 05 Q2 对齐） |
| P2-6 | 04:601 | 表格同一行混用 "✅ 逻辑一致 / ❌ 数据不满足" | 拆行或加注 |
| P2-7 | 04:502-503 | 提及 "P0 级…" 但未编号；Lead 讨论中已出现 "P0-3" 指代 | 显式编号 P0-1/P0-2…，便于评审引用 |
| P2-8 | 04:566 | `torch_utils.py:19-42` 引用范围，函数实际到 :44 | 改 `:19-44` |

---

## 4. 遗漏的重要行为（代码里有、04 没写）

1. **训练期验证也受 mono 影响**（见 P1-2）：`AVSpeechEvalDataset` 是训练 `val_dataloader` 的 dataset，不仅评测入口。
2. **评测逐条会读相邻 `<mix>_meta.json`**：`AVSpeechEvalDataset.__getitem__` 总是尝试 `mix_path.replace(".wav","_meta.json")` 统计 `n_near`（`avspeech_dataset.py:201-210`），即使评测脚本把该值丢弃（`audio_visual_test.py:229` 的 `_`）；缺失时静默 0（try/except）。→ 评测数据目录若不带 meta 也能跑，但每样本多一次文件 IO；04 未提。
3. **评测不过滤 <2 s 短音频**：训练集会过滤（`avspeech_dataset_near.py:111-122`），评测 `AVSpeechEvalDataset` 全量加载（`:161-171`）→ 训练/评测样本集合不同，04 未提。
4. **eval 无截断/补齐、输出长度严格 == T**：04 只给出常数"50 帧→251 帧"的例子；实际比例随句长变化（`T'=1+floor(T/128)`），且模型靠 `istft(length=...)` 精确保长。建议在 §3.2/§4.5 明确。
5. **`use_near=False` 的完整行为**：只影响 train（`near=[]`、不读 near.json）；eval 数据集仍构造并读 meta；`setup()` 需要 **train/valid/test 三个目录**的 `mix.json`+`target.json`（`avspeech_dataset_near.py:277-295`）。04 提了 train_dir，未提 valid_dir/test_dir 同样必须存在。
6. **DataModule 里的 `val_dataloader`/`test_dataloader` 在评测路径完全没有被创建/调用**：评测只用 `make_sets`；`audio_train.py` 里 `test_loader` 传入后也从未被消费（04 已说"不用 DataLoader"，可补这一句更完整）。

---

## 5. 跨模块一致性问题

| 涉及 | 04 的说法 | 实际/建议 |
|---|---|---|
| 04 ↔ 05（数据管线） | mono 只作为 Q4 风险；`dtype=float64`；行号 166-172 | 与 05 §3.5/§3.7/§8.6 + 补充验证对照：mono 后果应升级、dtype 需条件化、行号改 169-174；R3 建议直接引用 `data-pipe-supplement.md` |
| 04 ↔ 06（Studio） | Q1：ckpt 格式未知，靠 Studio `get("state_dict", ckpt)` 兼容 | `PROVENANCE.md:36-40` + `README:107` + `falcon_runtime.py:160-169` 已可定论：发布文件持有 `state_dict`（含 audio/video 前缀）；Q1 应关闭 |
| 04 ↔ 03（训练） | A41/Q3：config 被 `Separator` pop 污染 | 污染链路还应包含 `audio_train.py:82-84`（`yaml.safe_dump(config, conf.yml)` 把污染后的 config 落盘，导致 README 评测命令用到的 conf.yml 缺键）——04 只引 `:54-58`；建议与 train-sys 03 对齐 |
| 04 ↔ 01（模型核心） | `video_params.stride:2` 死配置；FRCNN `T_out=T_in`；融合 nearest 插值 | 与源码/我的实测一致（`Falcon.py:923,964-976`；`Falcon.py:257-258`），无矛盾 |
| 04 ↔ 02（层/DSP） | 88×88 裁剪归数据/Studio，FRCNN 内不裁剪 | 与 02 应有结论一致（建议交叉引用 02 文档） |
| 04 ↔ README | README:165/168 评测命令"不可直接复现（P0）" | 判断正确；**原因归属需修正**：主因是 B2（conf.yml 类名）与 B3（`${DATA_ROOT}`），不是 ckpt 格式（见 P1-1）；B4（无权重）在 `--ckpt_path` 给出时应消失 |
| 04 ↔ 05（数据准备） | 04 未涉及 DataPreProcess | 不影响本域结论；如需衔接可引用 05 §2.6/§5.9（README 第二条命令冗余、`${DATA_ROOT}` 不展开） |

---

## 6. 给作者的问题清单

1. **Q1/B5**：是否接受用 `studio/falcon_model/PROVENANCE.md:36-40`（发布文件持有 `state_dict`+architecture+epoch）与 `README:107` + `falcon_runtime.py:160-169` 关闭"可能 KeyError"，并据此把 §2.4-E 行 600 改写为"因 B2/B3 失败"？
2. **mono 风险**：是否将 B7 升级为"条件性 P0（mono 数据下评测/验证必崩）"，并把"训练期 `validation_step` 同样受影响 → 反证真实数据多声道"写入 §3.4/Q4？真实数据通道数仍需数据侧确认（05 Q1）。
3. **dtype**：是否把"float64"改为"由 `.pt` 存储 dtype 决定（uint8→float64；float32→float32；FRCNN 前统一 float32）"？
4. **口径**：是否补一节"评测整句推理 vs 训练 2 s 随机段"的说明（数据域已确认 eval 整句、输出长度==T；见 05 §3.5/§8.6）？
5. **行号/类型**：是否修正 `avspeech_dataset_near.py:166-172 → 169-174`、`tensors_to_device` 返回 list、`make_sets` 解包写法、setup `:277-295`？
6. **mouth `.pt` 格式**：是否把 `[T,96,96,1]` 标为 `[推断]` 并加入未解疑问（与 05 Q2 一致；仓库内无生成脚本/资产）？

---

## 附：本次评审的独立验证活动

- 复现 mono-eval P0：`PYTHONPATH=/tmp/dp-verify/stubs:$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_p0_eval.py`；日志 `/tmp/dp-verify/logs/verify_p0_eval.log`（真实模型：audio 537,697 + video 11,186,688 参数；mono → RuntimeError；stereo/odd-length → 输出 `(1,1,T)`）。
- 静态核对 04 全部 50 条断言的行号/语义（重点 A01-A50 中数据/评测/视觉相关约 30 条），另读源码：`avspeech_dataset.py`、`avspeech_dataset_near.py`、`transform.py`、`audio_visual_test.py`、`frcnn_videomodel.py`、`Falcon.py`/`Falcon_Contrastive.py`、`metrics/allwrapper.py`/`wrapper.py`、`losses/nass_contrastive_loss.py`/`matrix.py`/`pit_wrapper.py`/`mixit.py`/`sisnri.py`、`utils/torch_utils.py`、`studio/falcon_model/PROVENANCE.md`、`studio/falcon_runtime.py`、README。
- 04 文档 sha256 评审前后一致（`d02806e2…`，见文首）；评审未修改 04 文档与任何源码。
