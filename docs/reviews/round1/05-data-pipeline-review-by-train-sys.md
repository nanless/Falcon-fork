# 评审：`05-data-pipeline.md` —— 评审人 `train-sys`

- 被评审文档：`docs/understanding/05-data-pipeline.md`。
- **版本锚定（评审期间文件被作者更新，已核对）**：
  - 评审开始：`sha256 = f1c90fa3…`，**777 行**（R1 初版）；
  - 评审结束：`sha256 = fde1ed6a…`（`fde1ed6a3943582e018a8c8156191c9e9a285818c3bf02feb01d6141b153c442`），**797 行**——
    作者追加了 §8.6「R2-prep 补充验证」（+20 行，见其 Changelog）；
  - 已核对：**第 1-777 行内容未变**（抽查错误点仍在原行号：`test_loader` 第 329 行、P2-8 第 446 行、Q14 第 686 行），
    故本文所有 `05:NNN` 行号对两版均有效；新增 §8.6 我已单独复核（见 §3 末尾「§8.6 附注」与 §5/C12）。
- 评审范围：按 Lead 指定重点（batch 形状权威结论、`make_loader/make_sets`、`_meta.json` 产生者、
  JSON 对齐、near 哨兵、短音频）+ 与 `03-training-system.md` 的交叉一致 + 抽查行号与动态验证可复现性。
- 本评审**未修改** 05 文档与任何源码；只新增本文件。

## 1. 结论

**结论：需修订（P0 × 1，P1 × 4，P2 × 9）。主体质量高**：8 节模板完整、63 条断言中约 60 条成立
（2 条需加条件、1 条不成立）；我抽查约 90 处 `path:line` 引用，仅 4 处存在 ±1~2 行的轻微偏差，
无实质性错引；其动态验证**可复现**——我在本机重跑其两个脚本均 `TOTAL FAILURES: 0`
（`verify_dataset2.py` 57 项、`verify_preprocess.py` 19 项，见 §1.1）。05 的两个核心贡献经我复核成立：
(1) batch 形状权威结论 `mouths [B,50,88,88]`（96 存储 → 88 crop）正确；
(2) `<mix>_meta.json` 不由本仓库任何脚本生成、且缺它时近场负样本静默为空——事实正确，但**严重度被低估**（见 P1-1）。

需修订的实质问题：
- **P0-1（事实错误）**：`test_loader` 并非"trainer.fit 不会消费/死路径"——它是第二个 validation
  dataloader，每 epoch 参与遍历，且每 10 epoch 计算 `test_loss`（详见 §3）。
- **P1-1（关键结论缺失/严重度低估）**：`_meta.json` 缺失不是"可选"边角，而是按 README 复现时
  **全量样本 K=0** → NASS 对比损失负项恒为 0（只剩正项）、MDCL 关闭、val 分桶全部塌进 `0near`。
- **P1-2 / P1-3（验证有效性）**：A18「1 通道与 3 通道 .pt 均可工作」是其**自建 torchvision stub** 的行为，
  不是真实依赖下的验证；「batch dtype = float32」只在 .pt 以 float32 存储时成立（uint8 存储 → float64）。
- **P1-4（契约缺口）**：near wav 采样率未纳入契约/未校验（near.json 只存 frames，Dataset 默认按 16 kHz 解释）。
- **P2 × 9**：4 处行号/引用瑕疵、2 处内部口径不一致、2 处工程风险未提、1 处验证产物未入
  `docs/verification/`（影响 R4 复现）。

### 1.1 本次评审的验证记录（可复现）

环境：`/root/.venvs/falcon/bin/python`（3.10.18 / torch 2.8.0 / PL 2.6.6）；05 的自建 stubs 位于
`/tmp/dp-verify/stubs/`（本机仍在，未写入仓库）。

| # | 命令 / 检查 | 结果 |
|---|---|---|
| R1 | `sha256sum docs/understanding/05-data-pipeline.md`（评审前后） | 开始 `f1c90fa3…`(777 行) → 结束 `fde1ed6a…`(797 行)：作者追加 §8.6（+20 行）；1-777 行未变 |
| R2 | `PYTHONPATH=/tmp/dp-verify/stubs:$PWD /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_dataset2.py` | 重跑通过，`TOTAL FAILURES: 0`（与 05 §8.2 一致） |
| R3 | 同上跑 `/tmp/dp-verify/verify_preprocess.py` | 重跑通过，`TOTAL FAILURES: 0`（19/19） |
| R4 | 独立复现 FRCNN 形状契约：`FRCNNVideoModel(pretrain=None).eval()(torch.randn(2,1,50,88,88))` | `(2,512,50)` float32 —— 支持 05 的 A48/§5.4（本次记录） |
| R5 | numpy dtype 流：`(uint8帧-0)/255` vs `(float32帧-0)/255` | uint8→float64（两级 Normalize 后仍 float64）；float32→float32 —— 见 P1-3 |
| R6 | 逐行抽查 05 的约 90 处引用（`avspeech_dataset*.py`/`transform.py`/`process_lrs23.py`/`generate_near_json.py`/`av_litmodule*.py`/`frcnn*`/`audio_visual_test.py`/Studio runtime） | 仅 §3 的 P2-1/P2-2/P2-3/P2-5 有轻微偏差 |
| R7 | 读 `/tmp/dp-verify/stubs/torchvision/transforms.py` | 其 `Grayscale.__call__` 显式含 `if img.shape[-3] == 1: return img`（P1-2 证据） |
| R8 | `grep -rn "_meta.json\|components"` 全仓库 | 仅 `process_lrs23.py`、`generate_near_json.py`、`avspeech_dataset.py` **读取**，无生成者 —— 支持 05 §3.2(d) |

## 2. 断言逐条核对表（✅正确 / ❌错误 / ⚠成立但需加条件 / ❓无法确认）

| 断言 | 判定 | 说明（证据核对） |
|---|---|---|
| A1 布局 `{tr,cv,tt}/{mix,target,mouths}`、mouth 路径由 target 替换 | ✅ | `process_lrs23.py:8-28,84-99` 逐行相符 |
| A2 mix.json 2 元组 | ✅ | `:19-20,31-32` |
| A3 target.json 3 元组 | ✅ | `:21-28` |
| A4 near.json 嵌套 list、与 mix 同序 | ✅ | `generate_near_json.py:43-67,73-78` |
| A5 `len(near)==len(mix)` | ✅ | fallback `:62-73`；fixture 8/8/8 |
| A6 meta 只读 `type/path` | ✅ | `process_lrs23.py:35-43`；`avspeech_dataset.py:208` 另读 `components` |
| A7 只收 `type=="near"` | ✅ | `:40-42`；fixture 2 near+1 far→K=2 |
| A8 无 meta/仅 far/缺文件 → fallback | ✅ | `:69-71`；fixture 6/8 fallback |
| A9 JSON 路径绝对 | ✅ | `:11` `os.path.abspath` |
| A10 建索引可重入、字节一致 | ✅ | 其 #1-#2 日志；我重跑 `verify_preprocess.py` 亦 0 FAIL |
| A11 短于 seg_len 的训练样本三表同步删 | ✅ | `avspeech_dataset_near.py:111-122` |
| A12 过滤后顺序为 JSON 反序 | ✅ | 倒序遍历 + append（`:111-122`） |
| A13 `__len__==samples` | ✅ | `:147-150`；`samples` 默认 10000（`:250`） |
| A14 `__getitem__` 忽略 idx | ✅ | `:154` `mix_index = random.randint(...)` |
| A15 窗口 `rand_start∈[0,n-seg_len]`、32000 点 | ✅ | `:156-157` |
| A16 `seg_len=32000`,`fps_len=50` | ✅ | `:77-78` |
| A17 `mouth_start=floor(rand_start/640)`、50 帧 | ✅ | `:168-174` |
| A18 `.pt` 4-D channel-last、**1 通道与 3 通道均可工作** | ⚠ | 4-D/permute 部分 ✅（`:169-174`）；"1 通道可用"仅在 `/tmp/dp-verify/stubs/torchvision/transforms.py` 的自写 `Grayscale` 下成立（其显式 `if img.shape[-3]==1: return img`），官方 torchvision 未在本环境验证（缺包且网络不可达）→ 该 [验证] 不能覆盖真实依赖，见 P1-2 |
| A19 train 管线、输出 `(T,88,88) f32` | ⚠ | 管线与 88 crop ✅（`transform.py:153-158`）；**f32 仅当 .pt 为 float32**（其 fixture 即 float32），uint8 → float64，见 P1-3 |
| A20 val 管线 = `[4:92,4:92]` | ✅ | `transform.py:161-167`；其逐元素比对 0 diff |
| A21 `sample_rate≠16000` 时 mouth_start 错位 | ✅ | `near.py:168` 字面量 16000；其 8 kHz 实测 15 vs 31 与代码推算一致 |
| A22 嘴部帧不足 → collate stack RuntimeError | ✅ | `near.py:168-174,40-42`；其 75/40 帧实测 |
| A23 训练 collate 形状/dtype 权威结论 | ⚠ | 形状 `[B,32000]/[B,32000]/[B,50,88,88]/list[str]/list[list[Tensor]]` ✅（`near.py:23-43`）；`float32` 需加条件（P1-3） |
| A24 near_audios K 可变且可为 0 | ✅ | `near.py:184-220`；`nass_v2.py:115-132` |
| A25 near 偏移 `min(rand_start,n-seg_len)`、<seg_len 跳过 | ✅ | `:193-199` |
| A26 fallback 哨兵被 Dataset 跳过 | ✅ | `:188-191` |
| A27 `use_near=False` → 全空 | ✅ | `:94-101,186-187` |
| A28 near.json 缺失 → WARNING，训练继续 | ✅ | `:94-101` |
| A29 当前配置关闭音频归一化 | ✅ | `configs/Falcon.yml:163`；`near.py:179-182` |
| A30 开启时 target 复用 mixture 的 std | ✅ | `near.py:176-182` |
| A31 静音段 eps 保护 | ✅ | `near.py:16-20` |
| A32 评测不过滤短音频、len=mix.json | ✅ | `avspeech_dataset.py:161-171` |
| A33 评测整段读取 | ✅ | `:179-187` |
| A34 评测嘴部整段 val 管线 `(F,88,88)` | ✅ | `:189-191` |
| A35 评测 5 元组、`n_near` 来自 meta | ✅ | `:201-212`（我复核 `:204` 替换、`:208` 计数、`:212` 返回） |
| A36 mono → 0 维退化（P0 风险） | ✅ | `:193-194` 无 ndim 守卫（训练集 `near.py:176-177` 有守卫，对比成立） |
| A37 训练集有 ndim 守卫 | ✅ | `near.py:176-177` |
| A38 13 参数 / 12 键全命中、仅缺 train_samples | ✅ | 我用 AST 独立核对：`signature params 13, config keys 12, 无多余键, 缺 train_samples` |
| A39 conf.yml 的 data_config 键集合相同 | ✅ | `Experiments/checkpoint/Falcon/conf.yml:90-103`（12 键相同，值为 `${DATA_ROOT}` 占位） |
| A40 `make_loader`/`make_sets` @property、每次新建 | ✅ | `near.py:331-337` |
| A41 训练 loader 参数 | ✅ | `:297-307` |
| A42 val/test loader `batch_size=1` 硬编码 | ✅ | `:309-329`（config 4 只影响 train） |
| A43 setup() 建 3 dataset → 只跑 test 也要 train/valid | ✅ | `near.py:277-295`；`audio_visual_test.py:187-189` |
| A44 评测 data_config 剔除 use_near/train_samples | ✅ | `audio_visual_test.py:183-189`（精确到行） |
| A45 评测不走 DataLoader、逐条索引 | ✅ | `:192-193,229-231` |
| A46 `tensors_to_device(tuple)` 返回 list | ✅ | `torch_utils.py:19-40`（`isinstance(x,(list,tuple))→[...]`） |
| A47 litmodule 解包 5 元组、mouth 4-D→unsqueeze | ✅ | `nass_v2.py:73-90`（`:82` `mouth.ndim==4`，`:83` unsqueeze，`:87` 过 video） |
| A48 FRCNN `[B,1,50,88,88]→[B,512,50]` | ✅ | `frcnn_videomodel.py:75-83`；**我独立复现** `(2,1,50,88,88)→(2,512,50)`（R4） |
| A49 near 全部使用、bucket 仅日志 | ✅ | `nass_v2.py:19,113-118,182-183` |
| A50 K=0/1/>1 三种损失分支 | ✅ | `nass_contrastive_loss.py:62-74` |
| A51 near 唯一判据 meta、无距离/SIR 阈值 | ✅ | 全仓库 grep + 读码 |
| A52 两脚本 CLI 参数、`--sample_rate` 未使用 | ✅ | `process_lrs23.py:108-121`；`generate_near_json.py:82-90,39`（process_split 收到 sample_rate 但函数体未用） |
| A53 空 split → `min([])` ValueError | ✅ | `generate_near_json.py:73-78` |
| A54 缺输入目录 → FileNotFoundError | ✅ | `process_lrs23.py:12` |
| A55 缺 mix.json → Skipping | ✅ | `generate_near_json.py:97-99` |
| A56 mix/target 等量错名 → 静默错配 | ✅ | `process_lrs23.py:12-13`（各自排序）+ Dataset 无校验 |
| A57 表长度不一致 → IndexError | ✅ | `near.py:111-122` |
| A58 check 脚本不可运行（硬编码 `${DATA_ROOT}`） | ✅ | `check_avspeech_dataset.py:317` |
| A59 check 脚本是旧副本、无引用 | ✅ | 类定义 `:30,133,210`；`__main__` `:312-347`；grep 无引用 |
| A60 Studio `preprocess_mouth` 与 val 管线逐元素等价 | ✅ | `studio/falcon_runtime.py:95-105` 代码级等价（`frames[:,4:92,4:92]/255` → `(x-.421)/.165`）与 `transform.py:161-167` 一致；其动态 diff=0 可信（我未重跑 Studio 侧） |
| A61 Studio 常量 16k/25fps/32000/50/96 | ✅ | `falcon_runtime.py:24-27`（我复核 :24-27 恰为 SAMPLE_RATE/VIDEO_FPS/WINDOW_SAMPLES/WINDOW_FRAMES；96 来自 preprocess_mouth 校验） |
| A62 `datas/__init__` 只导出旧模块 | ✅ | `look2hear/datas/__init__.py:1,5` |
| A63 audio_train 用 `make_loader`、audio_visual_test 用 `make_sets` | ✅ | `audio_train.py:51`；`audio_visual_test.py:189` |

**缺失/不成立的断言（应补）**：
1. ❌ 无"test_loader 是否被消费"的正确断言——相反 §3.4 写了错误结论（P0-1）。
2. ❌ 无"`_meta.json` 缺失 → K=0 → 负项为 0"的端到端断言（只在 Q6 提及；应作为主结论，P1-1）。
3. ⚠ A18/A23 的验证层级应降级/加条件（P1-2/P1-3）。
4. ❌ 无 near wav 采样率契约/校验断言（P1-4）。
5. ❌ 无 dtype 的条件化结论（uint8 vs float32 存储）与 litmodule `type_as` 兜底说明。

## 3. 错误清单

### P0（事实错误）

**P0-1 `test_loader` 被 `trainer.fit` 消费（05 三处写反）**
- 05 位置：`05:329`（§3.4 "test_loader 只被保存，trainer.fit 不会消费；训练脚本不跑 test"）、
  `05:446`（P2-8 "test_loader 从未被消费"）、`05:686`（Q14 "test_loader 是死路径"）。
- 代码事实：`AudioVisualLightningModule.val_dataloader()` 返回 **`[self.val_loader, self.test_loader]`**
  （`look2hear/system/av_litmodule.py:214-216`）；`audio_train.py:180` `trainer.fit(system)` 会让 PL 的
  evaluation loop 遍历**两个** dataloader；`validation_step` 在 `dataloader_idx==1` 且
  `current_epoch % 10 == 0` 时计算并记录 `test_loss`（`av_litmodule.py:125-140`）。
- 外部证据：我 R1 的 dummy 模型 11-epoch 实测中，`trainer.logged_metrics` 出现
  `test_loss/dataloader_idx_1`（`docs/verification/runs/train-sys-litmodules.log`），且 `03` 文档 V4 已记录。
- 影响：读者会以为 tt 集训练期完全不参与（实际每 epoch 都被遍历、每 10 epoch 计算损失），并据此得出
  错误的 P2-8/Q14 建议（"删除/标注 test_loader"）。
- 修正建议（3 处）：改为"`test_loader` 作为第二个 validation dataloader，**每 epoch 参与数据遍历**，
  仅每 10 epoch 计算/记录 `test_loss`；`trainer.test()` 从未被调用"。
- 级别判定：按 00-CONVENTIONS §5（P0 = 事实错误）列 P0；文字修正成本低。

### P1（重要遗漏 / 误导）

**P1-1 `_meta.json` 的严重度被低估，缺"按 README 复现 → 对比训练信号静默退化"的完整结论**
- 05 已正确发现 meta 不由仓库生成（`05:284-286`、§8.5），但把它放在"可选输入"位置
  （`05:19` "可选；只用于生成 near.json"、`05:462`），后果只出现在 P1-6 与 Q6。
- 完整证据链（我逐环节复核）：
  1. README 数据准备只有两条命令、从不生成 meta（`README.md:148-158`）；
  2. `process_lrs23.py:59-71` / `generate_near_json.py:62-65`：无 meta → **fallback 成 `[mix_path, n]`**；
  3. `avspeech_dataset_near.py:190-191`：`near_path == mix_path` 的 fallback 条目被**静默跳过**；
  4. 因此按 README 流程构建的数据，**每个训练样本 `near_audios=[]`（K=0）**；
  5. `av_litmodule_nass_v2.py:115-120` → K=0 → `nass_contrastive_loss.py:73-74` `l_neg=0`，
     对比损失退化为**仅正项** `α_pos(1−cos(q,p))`；MDCL 分支 `mdcl_total` 同理恒 0（`av_litmodule_mdcl.py:100-126`）；
  6. val 端 `n_near` 同样来自 meta（`avspeech_dataset.py:201-211`）→ 全为 0 →
     `av_litmodule.py:161-169` 所有样本只进 `val_pit_sisnr_0near`。
- 结论：论文贡献之一的"近场干扰人负样本对比学习"在当前 README 的可复现流程下**静默失效**（训练不报错）。
  这不是 05 的事实错误，而是**主结论缺失/严重度低估** → P1（若按"论文可复现性"口径，建议 Lead 升格为 P0）。
- 修正建议：把 `_meta.json` 从"可选"改为"**论文对比损失必需的输入资产（仓库不提供其生成器）**"，
  在 §1/§4.7 增加上述 6 步后果链，并在 Q6 给出明确结论（不再是开放问题）。
- 附：`05:19` "只用于生成 near.json" 也与 `05:390`（eval 用 meta 统计 n_near）自相矛盾 → 一并修正（P2-8）。

**P1-2 A18「1 通道与 3 通道 .pt 均可工作」不是真实依赖下的验证**
- 证据：05 §8.1 自建 `/tmp/dp-verify/stubs/torchvision/transforms.py`，其 `Grayscale.__call__`
  **显式** `if img.shape[-3] == 1: return img`（我 R7 读取确认）；真实 torchvision 的
  `Grayscale → rgb_to_grayscale` 是否接受单通道未在本环境验证（沙箱缺 torchvision、网络不可达）。
  仓库代码 `near.py:169-174` 在 `Grayscale(...)` 后 `.squeeze(1)`，说明作者预期 Grayscale 输出
  恰为 1 通道（这更像 3 通道输入的产物）。
- 影响：若真实 `.pt` 为 1 通道且官方 torchvision 拒绝，则 Dataset 直接报错——05 会把该风险掩盖为"已验证可用"。
- 修正：A18 降级为 [代码]+[推断]，注明"1 通道可用性依赖 torchvision 行为，未验证；3 通道→1 通道路径最可信"
  （其实 05 的 Q2 已承认真实格式未知，只需在断言表同步降级）。

**P1-3 「batch dtype = float32 权威结论」过度概括**
- 证据：05 的 fixture 用 float32 存 `.pt`（`/tmp/dp-verify/make_fake_data.py:33-42`
  `np.full(..., dtype=np.float32)`），故 D3/A23 得 float32；但真实 `.pt` 格式未知，README:132 的
  推理格式是 **uint8 96×96**（"same format as the training data"）。读取链
  `near.py:169-174`（torch→numpy）→ `transform.py:155-158`（`Normalize(0.,255.)` 等）在 numpy 下：
  uint8 − 0.0 → float64（我 R5 实测：uint8→float64，float32→float32）。
- 因此若真实 `.pt` 是 uint8，batch `mouths` 将是 **float64**（而非 05 §5.4/A23/D3 的 float32）。
  功能上无碍：`av_litmodule_nass_v2.py:83-87` 用 `mouth_in.type_as(mixtures)` 转 float32，但 05 未提兜底。
- 修正：把"权威 dtype"改为条件表述（uint8 存储→float64，再被 litmodule 转 float32；float32 存储→直接 float32）。

**P1-4 near wav 采样率未纳入契约、代码也不校验**
- 证据：near.json 只写 `[path, sf.info().frames]`，**不写 sample_rate**
  （`process_lrs23.py:61-67`；`generate_near_json.py:54-60`）；Dataset 用 `n_samples < self.seg_len`
  （`near.py:193`）和 `sf.read(start,stop)`（`:198-203`）直接按 16 kHz 解释——若 near wav 是 48 kHz，
  "帧数"比较与时间轴都会错位（片段长度只有 1/3）。
- 05 §5.1 只对 mix/target 标注"16 kHz"，near 行写"建议与 mix 等长"（`05:462-463`）→ 契约不完整。
- 修正：在 §5.1/§5.2 注明 near wav 必须 16 kHz 或 Dataset 未做重采样；建议列入 P1/P2 风险。

### P2（细节瑕疵）

| 编号 | 问题 | 05 位置 | 证据 / 修正 |
|---|---|---|---|
| P2-1 | 引用 `README:132,134` 说明嘴部格式，但 README:134 是 `### Python API`，格式只出现在 :132 | `05:575` | `grep -n "### Python API" README.md` → 134。改为 `README:132` |
| P2-2 | `__len__` 引用 `:171`，实际 `__len__` 在 `:173-174`（:171 是 `self.length = orig_len`） | `05:118` | `look2hear/datas/avspeech_dataset.py:171,173-174` |
| P2-3 | 评测数据构造引用 `audio_visual_test.py:178-189`，实际构造代码在 `:183-189`（:178 是 `def process_worker`） | `05:48` | 精确到 `:183-189` |
| P2-4 | "README 只有下载 TODO" 不准确：README 唯一 TODO 是 backbone 权重（`README.md:158` HTML 注释），数据/meta 无任何下载或生成说明 | `05:286` | 改为"README 完全未提 meta 产生方式"（反而更严重） |
| P2-5 | `requirements.txt:6-7` 中 torchvision 实际在 `:7`（`:6` 是 torchaudio）；其余 9/11/13-14/23-24 正确 | `05:760` | 改为 `:7,11,13-14,23-24` |
| P2-6 | 未提 `wav_path.replace(out_filename,"mouths")` 是**全路径字符串替换**：数据根路径若含 "target" 子串，mouth 路径会被破坏；且建索引不校验 `.pt` 是否存在 | `process_lrs23.py:25` | 建议 §3.1/§4.7 补 P2：应改为 `os.path.join(dirname, "mouths", stem+".pt")` |
| P2-7 | 未提每样本 `torch.load(.pt)` **整段加载**只为取 50 帧的 I/O 开销；eval 每条读 `_meta.json` | `near.py:171`；`avspeech_dataset.py:206` | 建议补性能注记（num_workers=8 下影响被摊薄） |
| P2-8 | 内部口径矛盾：`05:19` 说 meta "只用于生成 near.json"，但 `05:390`/§3.2(d) 与 `avspeech_dataset.py:201-211` 表明 eval/val 也用它 | `05:19` vs `05:390` | 统一为"生成 near.json + 评测/验证统计 n_near" |
| P2-9 | 验证脚本/fixtures/logs 主体仍在 `/tmp/dp-verify/`（仅摘要落了 `docs/verification/data-pipe-supplement.md`），未按 `VERIFY-ENV.md §4` 把脚本与完整日志落 `docs/verification/{scripts,runs}/`；清理后 R4 无法独立复核 | `05:697,706-719,769,773` | 建议补交脚本与日志（我已重跑两脚本 0 FAIL，见 §1.1 R2/R3；脚本本体仍在 /tmp，可原样拷贝） |

### §8.6 附注（作者 R2-prep 追加内容，评审人已复核）

作者在我评审期间追加的 §8.6（`05:771-788`）用**真实模型**坐实了 mono-eval P0，并把评测入口的数据路径契约
（不用 DataLoader/整句/无 2 s 截断/模型内插值/`use_near=False` 行为）写清楚。我抽查其引用的 5 处跨模块行号，
**全部准确**：`Falcon.py:257-258`（`F.interpolate(..., mode="nearest")`）、`Falcon.py:1095-1097`
（`unsqueeze_to_2D` 的 `(1,)→(1,1)`）、`Falcon.py:1110-1116`（`torch.stft(..., center=True)`）、
`Falcon.py:1166,1182-1188`（`istft(length=input_shape[-1])`）。其"mono 会在 `torch.stft` 抛 RuntimeError、
异常不被 `try` 吞掉"的结论与我的 A36 判断一致，且**强化了** 05 的 P0（从"推断风险"升级为[验证]）。
该追加**不影响**本文 P0-1/P1-1~P1-4 的判断；P2-9 已部分改善（摘要文档已落盘，脚本/日志仍在 /tmp）。

## 4. 遗漏的重要行为（代码有、05 没写或写错）

1. **val 循环实际由 `[cv, tt]` 两个 loader 组成**，`test_loss` 每 10 epoch 计算一次
   （`av_litmodule.py:125-140,214-216`）——05 三处写反（P0-1）。
2. **meta 缺失时 K=0 的端到端后果链**（P1-1），以及 val 分桶指标全部塌进 `0near`。
3. **dtype 的条件化结论 + litmodule `type_as` 兜底**（P1-3）。
4. **near wav 采样率契约**（P1-4）。
5. **`wav_path.replace` 全局替换风险 & mouths `.pt` 存在性不校验**（P2-6）。
6. **`.pt` 整段加载/每条 meta I/O 的代价**（P2-7）。
7. `audio_train.py:44` 的 `use_near` 仅用于打印；真正开关是 `AVSpeechTrainDatasetWithNear(use_near=...)`
   （`near.py:284`）——05 覆盖了 use_near 行为，但未点出入口处这层"打印用"语义（小）。
8. `AVSpeechTrainDatasetWithNear.__init__` 的 near 自检只读**第一条非空 near 的前 100 帧**
   （`near.py:133-145`）——05 描述为"启动自检"是对的，但未指出它无法发现后续文件的不可读/采样率问题（可并入 P1-4）。

## 5. 跨模块一致性问题

| # | 05 与谁冲突 | 判定 | 证据 / 处理 |
|---|---|---|---|
| C1 | 05 `mouths [B,50,88,88]`（§5.4/A23） vs **03** 文档 §3.3/§5.2/§5.3 写的 `mouth[B,Tv,96,96]` | **05 正确，03 错误** | `transform.py:147-158`（默认 `Crop_size=(88,88)`）→ 输出 88×88；03 的 litmodule 逻辑（`mouth.ndim==4→unsqueeze(1)`、`mixtures.shape[-1]`）与 H/W 无关、自洽，但数字写错。**我在 R3 修正 03**；05 无需改（可加一句"litmodule 对 H/W 无假设"） |
| C2 | 05 "test_loader 不被消费" vs **03** §3.4/A21 "val_dataloader=[val,test]，test_loss 每 10 epoch" | **03 正确，05 错误** | `av_litmodule.py:125-140,214-216`；03 有 11-epoch 实测日志 `train-sys-litmodules.log`。05 需改（P0-1） |
| C3 | 05 未写 val loader 组成 vs 03 明确 `[cv,tt]` | 03 更完整，05 应补 | 同上 |
| C4 | 05 §5.4 说 val loader mouths `[1,F,88,88]` vs 03 无此类断言 | 一致（03 只写训练 batch 形状但 H/W 数字错，见 C1） | — |
| C5 | 05 §4.2/A13 `train_samples=10000`（1 epoch≠全量） vs 03 §4.7/A17（1 epoch=10000 随机片段、10 卡 250 step/epoch） | 一致 | `near.py:147-150,250`；Falcon.yml 无该键 |
| C6 | 05 P1-6/A49 bucket 仅日志 vs 03 §3.4.2（`train_n_samples_*` 键、`val_pit_sisnr_*near`） | 一致 | `nass_v2.py:19,113-118,182-183`；`av_litmodule.py:161-169` |
| C7 | 05 A18/A23 dtype/channel vs **03** 未断言 dtype | 03 无冲突；但 03 的 `near_audios list[list[Tensor]]` 与 05 一致 | — |
| C8 | 05 A60 Studio 等价 vs **06**（未产出） | 待 06 复核"抽取端（MediaPipe 裁剪）"是否也一致；05 的 Q8 已正确标注边界 | `studio/falcon_runtime.py:95-105` 只证明数学等价 |
| C9 | 05 P0（eval mono 退化） vs **04**（未产出） | 需 04 确认评测数据通道数/评测指标口径；05 的 Q1 已挂起，结论合理 | `avspeech_dataset.py:193-194`；README:67 mono |
| C10 | 05 §5.5 `n_src` 未被使用 vs **01**（模型 `n_src=1` 契约） | 无冲突：datamodule 的 `n_src` 仅校验 `∈{1,2}` 后不再使用（`near.py:256-257,262`），模型 `n_src` 来自 `audionet_config` | — |
| C11 | 05 与 `00-project-map.md`：project map 未给 mouth 形状，但 03 用了 96 | 05 的校正有效（project map §4.1 只说 "mouth 帧"，未定尺寸）；建议 Lead 在 FINAL 采纳 88 | `transform.py:147-158` |
| C12 | 05 §8.6 的"eval 整句靠模型内插值/istft 精确还原长度"（`Falcon.py:257-258,1166,1182-1188`） vs 01/02（model-core / layers-dsp） | 待 01/02 复核；我抽查其行号全部准确、机制与源码相符 | `Falcon.py:250-262,1095-1097,1110-1116,1160-1190` |

## 6. 给作者的问题清单

1. **Q-A（P0-1）** 是否同意把 §3.4:329、P2-8、Q14 三处的 `test_loader` 结论改为
   "第二个 validation dataloader，每 epoch 遍历、每 10 epoch 计算 `test_loss`；`trainer.test()` 未被调用"？
   若你当时是有意区分"训练脚本不调用 `trainer.test()`"，建议把这句话与"fit 的 val loop 会消费 test_loader"分开写，避免歧义。
2. **Q-B（P1-1）** 是否同意把 `_meta.json` 从"可选"升格为"论文对比损失必需的输入资产"，
   并在 §1 结论 / §4.7 主表加入"README 流程 → 全 fallback → K=0 → 负项=0 / MDCL 关闭 / val 分桶全 0"的
   明确结论？我认为这是本文档最有价值的发现，但目前藏在 Q6/P1-6 里，Lead 与其他读者很可能会漏掉。
3. **Q-C（P1-2/P1-3）** 是否同意把 A18 的"1 通道 .pt 可用"降级为 [推断]（stub 行为），
   并把 A23/§5.4 的 `float32` 改为条件结论（uint8 存储 → float64 → litmodule `type_as` 转 float32）？
   如果你有真实 `.pt` 样例的 dtype 信息，请直接给出并保留 [验证]。
4. **Q-D（P1-4）** near 源 wav 的采样率在数据生成端是否保证 16 kHz？若不能保证，建议在 §5.1/§4.7
   补一条"Dataset 不做重采样、只按 16 kHz 解释 frames"的风险项。
5. **Q-E（P2-9）** 能否把 `/tmp/dp-verify/{verify_dataset2.py,verify_preprocess.py,stubs,logs}` 摘要
   补交到 `docs/verification/scripts/` 与 `docs/verification/runs/`？我重跑两者均 0 FAIL，
   但你补交后 R4 复核人可以不依赖 `/tmp` 独立复现（Lead 的 VERIFY-ENV §4 有此要求）。
