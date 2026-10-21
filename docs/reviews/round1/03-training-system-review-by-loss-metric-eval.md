# 评审：`docs/understanding/03-training-system.md` —— 评审人 `loss-metric-eval`

> 评审对象：`docs/understanding/03-training-system.md`（569 行，R1 初稿）
> 评审对象 sha256（评审开始时）：`16ad59ea43a7397f1395b024e1eccbf6bec3072b0d8587324816f6f9b575780f`
> 评审范围：以 Lead 指定的 5 组重点（S0–S26 装配链 / 损失组合对照 / P0 `comet_logger`+Comet API / P1 PL 2.x+死配置+DDP 死锁 / conf.yml 四处差异与评测复现）为主线，
> 并对 A01–A46 断言表逐条核对，抽查正文全部 `file:line` 证据。
> 评审方式：**独立读源码** + **独立运行脚本**（不只看 03 自己的日志）：
> - 我自写 `docs/verification/scripts/verify_r2_review_03.py`（symtable、PL 2.6.6 契约、round-trip、best_k 格式、Falcon AST、依赖、torch.load），日志 `docs/verification/runs/r2-review-03.log`；
> - 独立重跑 03 的 `verify_train_pl_contract.py`（结果另存 `docs/verification/runs/r2-review-03-pl-contract.log`）、`verify_train_undefined_globals.py`、`verify_train_config_mutation.py`；
> - 交叉阅读 `04-losses-metrics-video.md`、`05-data-pipeline.md`、`08-ops-tests-docs.md` 的对应结论。
> **评审对象 sha256（评审结束时）仍为** `16ad59ea…b575780f` —— 评审期间 03 文档未被改动（未施加任何修改）。

---

## 1. 结论

**结论：通过（需修订）—— P0 = 0，P1 = 3，P2 = 10。**

- 03 的两条 P0（`comet_logger` 未定义、litmodule 硬依赖 Comet `logger.experiment.log(dict)`）**独立复核成立**：
  symtable 显示 `audio_train.py` 中 `main` 唯一未定义全局名就是 `comet_logger`；`CSVLogger.experiment` 为
  `ExperimentWriter`，调用 `.log({})` 精确复现 `AttributeError: 'ExperimentWriter' object has no attribute 'log'`。
- S0–S26 装配链、`datamodule` 不传 Trainer、`make_loader` 只取一次、ModelCheckpoint/EarlyStopping 参数、
  best_k_models/best_model.pth 写出、PL 2.6.6 契约（`accelerator=None` 的 `ValueError`、`devices=[0..9]`
  的 `MisconfigurationException`）、死配置键、DDP bucket 死锁机制、conf.yml 与 `configs/Falcon.yml` 的
  4 处内容差异 —— 以上**全部复核为正确**，行号抽查未发现错误。
- 03 的损失组合与 04 完全一致：train=`pairwise_neg_snr` + `threshold_byloss=true`、val=`pairwise_neg_sisdr`
  + `threshold_byloss=false`；`spk_contrast` 7 个键在 `audio_train.py:103-110` 读取、`:129-135` 全部注入；
  β=0→1.0 / 200 epoch；NASS 公式（含 K=1 跳过 softmax）与 04 §2.1.5 一致。
- 主要问题集中在 **数据/视觉接口形状**与**跨模块后果**：03 把嘴部张量写成 96×96（实际 88×88）、把
  train 的 2 s/50 帧形状错误地套到 val/test（实际整段、变长），且未写出 config 被 `dict.pop` 污染后
  对评测入口的具体后果。这三条是 P1；其余为行文/口径细节 P2（明细见 §3）。
- 断言表质量很高：A01–A46 中 43 条 ✅、3 条需措辞修订（A04/A42/A37 口径，均非事实硬错），
  没有发现 ❌ 级断言错误；主要错误出现在 §3.3/§5.2/§5.3 正文的**形状契约**上。

---

## 2. 断言逐条核对表（A01–A46）

标记：✅ 正确（本轮独立复核通过）；⚠️ 结论方向正确但表述需改；❓ 无法独立确认。
“核对方式”列：`源`=读源码/静态；`动`=我独立运行；`日志`=核对 03 的验证日志（可复现的脚本仍存在）。

| # | 断言概括 | 判定 | 核对方式与说明 |
|---|---|---|---|
| A01 | 唯一训练入口，默认 `--conf_dir` 不存在 | ✅ | 源：`configs/` 仅 `Falcon.yml`；`audio_train.py:30-40` |
| A02 | `comet_logger` 未定义，Trainer 构造前 NameError | ✅ | **动**：自写 symtable 输出 `[('main','comet_logger')]`；全仓库源码仅 `audio_train.py:170` |
| A03 | 不实例化任何 logger，只建未用目录 | ✅ | **动**+源：grep 仅 03 的验证脚本含 logger 名；`audio_train.py:153-154` |
| A04 | epoch-end 走 Comet 专属 `logger.experiment.log(dict)` | ⚠️ | **动**：CSVLogger 复现 AttributeError ✅；但“任何非 Comet logger”过强（见 P2-8） |
| A05 | `main_args={}`、`exp_dir` 是新建键 | ✅ | **动**：round-trip `main_args == {}`；`parser_utils.py:143-148`、`audio_train.py:77-80` |
| A06 | YAML→argparse 只展开两级，三级 dict 整体传递 | ✅ | **动**：`loss.train` 仍为 dict；`parser_utils.py:31-43` |
| A07 | `data_config` 12 键与 DataModule 形参匹配；`train_samples` 用默认 | ✅ | 源（04 亦验证）：`avspeech_dataset_near.py:237-252` |
| A08 | DataModule 不是 LightningDataModule，从不传 Trainer | ✅ | 源：`avspeech_dataset_near.py:226`；`audio_train.py:49-51` vs `:161-173` |
| A09 | `getattr(look2hear.models,...)` 构造，`__all__` 陈旧不影响 | ✅ | 源：`models/__init__.py:1-6` |
| A10 | 视觉模型加载 pretrain 后冻结参数与 BN 统计 | ✅ | 源+04 §2.3.2 [验证]：`frcnn_videomodel.py:86-95,106-118` |
| A11 | 无 `pretrain_zoo/` → 构造 FileNotFoundError | ✅ | 动（04 脚本）：`frcnn_videomodel.py:87`。注：完整装配链中数据 setup 更早失败（P2-9） |
| A12 | 优化器外置构造，`configure_optimizers` 返回之 | ✅ | **动**：`configure_optimizers` 返回结构 `([opt],[{monitor, scheduler}])`；`av_litmodule.py:183-208` |
| A13 | ReduceLROnPlateau(patience=15,factor=.5)，monitor 注入 | ✅ | 源：`audio_train.py:69-74`；`av_litmodule.py:194-196` |
| A14 | ModelCheckpoint 参数齐全 | ✅ | **动**：重跑 `verify_train_pl_contract.py` 输出 `ModelCheckpoint kwargs: OK`；`audio_train.py:139-147` |
| A15 | EarlyStopping patience=30，与调度器 15 不同 | ✅ | **动**：同上 `EarlyStopping kwargs: OK`；`Falcon.yml:134-138` |
| A16 | grad clip 5.0 / limit 1.0 / sanity 0 / sync_bn | ✅ | 源：`audio_train.py:167-172` |
| A17 | 无 accumulate → 1；全局 batch=4×卡数 | ✅ | 源：`audio_train.py:161-173`；`Falcon.yml:131,164` |
| A18 | `accelerator=None` 在 PL 2.6.6 非法 | ✅ | **动**：`ValueError: You selected an invalid accelerator name: accelerator=None` |
| A19 | `devices=[0..9]` 在 2 卡机器抛 MisconfigurationException | ✅ | **动**：`MisconfigurationException: ... machine only has: [0, 1]` |
| A20 | resume 仅 `os.path.exists`，不展开 `~`/env，静默从头 | ✅ | 源：`audio_train.py:176-180` |
| A21 | val_dataloader 返回 `[val, test]`，test_loss 每 10 epoch 才算 | ✅ | **动**：日志键含 `test_loss/dataloader_idx_1`；`av_litmodule.py:125-140,214-216` |
| A22 | 多 dataloader 自动加后缀 | ✅ | **动**：logged keys `val_loss/dataloader_idx_0`/`test_loss/dataloader_idx_1` |
| A23 | 基类 training_step 不解包 2 元组 → 与 Contrastive 不兼容 | ✅ | 日志+源：`PITW(tuple_ests): AttributeError: 'tuple' object has no attribute 'size'`；`matrix.py:198` |
| A24 | NASSv2 显式解包、用 `emb[:,0]` 当 query | ✅ | 源：`av_litmodule_nass_v2.py:90,102-103` |
| A25 | total = PIT + weight·CL（0.9065=0.6565+0.25） | ✅ | 03 日志 `train-sys-litmodules.log:50`；公式与 04 §2.1.5 一致 |
| A26 | β 线性 clamp；E≤0→β_final | ✅ | 日志 `β(epoch){0:0,1:.005,50:.25,100:.5,150:.75,200:1,250:1}`；`nass_v2.py:66-70` |
| A27 | 生效 β 0→1.0/200（非默认 1.5/100） | ✅ | 源：`audio_train.py:103-110,129-135`；`Falcon.yml:104-111` |
| A28 | NASS 公式/K=1/K=0 语义 | ✅ | 源（04 §2.1.5 已数值对拍）；00 无矛盾。补：平均分母恒为 B（03 未提，见 O-4） |
| A29 | MDCL total = PIT + α·mean_b(L_hard)，β 同 NASSv2、默认 1.5 | ✅ | 源：`Falcon.py:1410-1462`、`av_litmodule_mdcl.py:67-71,153-155`；日志 MDCL fit OK |
| A30 | MDCL 模块不可达（未导出、无 import） | ✅ | 源：`system/__init__.py:1-11`；grep 仅自命中（我复核） |
| A31 | `training.system/parallel`、`datamodule.data_name` 未被读取 | ✅ | **动**：grep `audio_train.py` 无 `parallel/data_name/["system"]` 读取 |
| A32 | `spk_contrast` 7 键读取并注入 | ✅ | 源：`:103-110,129-135` |
| A33 | exp_dir 与 conf.yml 建模后落盘 | ✅ | 源：`audio_train.py:77-84` |
| A34 | 建模原地改 config：+causal、−repeats/shared | ✅ | **动**：重跑 `verify_train_config_mutation.py`，输出与 03 完全一致 |
| A35 | conf.yml 键集无法由当前代码复现 | ✅ | **动**：两 YAML 扁平 diff 显示 conf 同时含 `audio_params.causal` 与 `repeats/shared`（当前代码不可能） |
| A36 | conf.yml 的 `SpeakerContrastiveV2` 不存在 | ✅ | **动**：`getattr` AttributeError（04 A40 同结论） |
| A37 | best_k 5 条、负 val_loss、最低 epoch=714 | ✅ | **动**：`min=-9.984025… @ epoch=714`；“6 行”是 wc 口径（文本 7 行），建议统一（P2-5） |
| A38 | PL 2.6.6 `best_k_models` 值为 Tensor，`.item()` 成立 | ✅ | **动**：重跑脚本输出 `('Tensor', tensor(...))` + `.item()` |
| A39 | 收尾 `torch.load`（默认）可加载 Lightning ckpt | ✅（中→实测微型） | **动**：默认 `torch.load` 加载含 callbacks/loops 等键的 ckpt OK；真实 ckpt 未测，与文档“中”一致 |
| A40 | 训练链实际只用 `parser_utils.*` + `print_only` | ✅ | 源：04/03 均验证；`print_only` 调用行列表不全（P2-4） |
| A41 | nets_utils 10 函数仅 `make_pad_mask` 被 import，`Stft` 从不实例化 | ✅ | 源：`layers/stft_tfgn.py:12,182`、`layers/__init__.py:20`；grep 无 `Stft(` 调用 |
| A42 | `separator.py`、`get_layer_from_string.py` 无调用点 | ⚠️ | separator ✅；`get_layer` 在 `utils/__init__.py:14` 被 import 并在 `:38` 再导出（仅无调用点）→ 措辞需改（P2-3） |
| A43 | `MyRichProgressBar` 引用未定义名 | ✅ | **动**：symtable 输出 `MetricsTextColumn`/`CustomProgress`；`lightning_utils.py:93-94` |
| A44 | per-bucket `all_gather` 不对称 → DDP 挂死 | ✅ | 源：`av_litmodule.py:164-169` 的 `if outputs:` 守卫 + 03 的 2 进程日志（结尾停在 bucket occupancy，未重跑） |
| A45 | torch_optimizer 依赖 pytorch-ranger；沙箱 import 失败 | ✅ | **动**：`import torch_optimizer → ModuleNotFoundError: pytorch_ranger`；metadata `requires=['torch','pytorch-ranger>=0.1.1']` |
| A46 | dict-scheduler 分支直接读 `sched["interval"]` → 潜在 KeyError | ✅ | 源：`av_litmodule.py:198-206` |

**统计**：✅ 43、⚠️ 3（A04/A42/A37 口径）、❌ 0；断言表整体可信。

---

## 3. 错误清单

### P0（事实错误、会误导实现/复现）：0 条

03 的两条 P0 均独立复核成立（见 §2 A02/A04、§5 的动/源证据），无漏报/误报。

### P1（重要错误/遗漏，需修订）：3 条

**P1-1 嘴部张量形状写成 96×96，实际是 88×88（3 处正文）**

- 03 证据：`03-training-system.md:246`（§3.3 训练 batch）、`:414`（§5.2 video_model 契约）、
  `:428-429`（§5.3 train/val-test batch 表）。
- 实际：`AVSpeechTrainDatasetWithNear` 与 `AVSpeechEvalDataset` 都会套
  `get_preprocessing_pipelines()["train"/"val"]`（`transform.py:147-168`：
  `Normalize(0,255)→RandomCrop/CenterCrop(88)→Normalize(.421,.165)`），
  输出 `[Tv,88,88]`；`avspeech_dataset_near.py:166-174`、`avspeech_dataset.py:138,189-192`。
  FRCNN 输入因此是 `[B,1,Tv,88,88]`（`frcnn_videomodel.py:75-84` 不做裁剪）。
- 独立验证：R1 伪造 1 s 样本实测 `mouth=(25,88,88)`、与手工 CenterCrop 对拍 `max|diff|=2.57e-7`
  （`docs/verification/runs/04-eval-flow.log` §[8]；`04` 文档 §2.3.2/§3.2）。
- 跨文档：`05-data-pipeline.md:65` 已明确指出“**mouth 张量不是 `[B,F,T,96,96]`**：实际为 `[B,50,88,88]`”；
  03 的写法与 05 直接冲突。
- 修正建议：§3.3/§5.2/§5.3 统一改为 `mouth[B,Tv,88,88]`、`forward(mouth[B,1,Tv,88,88])`，
  并注一句“96×96 是磁盘存储格式，Dataset 预处理裁到 88×88”。

**P1-2 val/test 的“2 s / 50 帧”形状契约错误，且漏掉“评测/验证是整段推理”**

- 03 证据：`03-training-system.md:430`（§5.3「形状」行）= `T=segment*16000=32000；Tv=segment*25=50`，
  上下文被放在 train/val/test 三行表格之下（`:428-429` 的 val/test 行也写了 `mouth[Tv,96,96]`）。
- 实际：只有 **train** 用 `segment=2.0` 随机取 2 s / 50 帧（`avspeech_dataset_near.py:152-174`）；
  **val/test 用 `AVSpeechEvalDataset`，`rand_start=0, stop=None` 整段读取、整段嘴部帧**
  （`avspeech_dataset.py:179-192`），因此 `T`/`Tv` 随音频时长**可变**（`Tv≈25*时长`），
  这就解释了 `audio_visual_test.py:228-232` 逐条 `test_set[idx]` + `mouth[None,None]` 的用法。
- 独立验证：R1 伪造 1 s 样本返回 `mix=16000` 点、`mouth=(25,88,88)`（`runs/04-eval-flow.log` §[8]）。
- 跨文档：`05-data-pipeline.md:119-120,345,633` 明确“整段读取/整段帧、评测逐条索引”；03 与 05 冲突。
- 修正建议：§5.3 拆成两行（train：32000/50 固定；val/test：整段可变 T、Tv≈25·T/16000），
  并在 §3.3 补一句“评测/验证不走 DataLoader、不做 2 s 窗”。

**P1-3 config 被模型构造污染后，对评测入口的具体后果未写（跨模块遗漏）**

- 03 已正确发现污染（§4.5、A34：建模后 `repeats/shared` 消失、多出 `causal`），但只写了
  “任何依赖 conf.yml 复现模型的工具都必须接受这些差异（→ 与 eval 交叉核对）”，没有给出**可执行的后果**。
- 实际后果（04 已复现）：`audio_visual_test._load_models_from_ckpt` 用 `config["train_conf"]` 重建模型
  （`audio_visual_test.py:128-139`）；若 conf.yml 缺 `audio_params.repeats/shared`，`Separator` 变成 0 层，
  模型参数量从 **537,697 掉到 49,081**，随后 `load_state_dict(..., strict=False)` 只打印 missing 数量、
  **静默丢弃 ~91% 权重**，评测继续跑但结果是错的（`docs/verification/runs/04-eval-flow.log` §[3]；
  `04` 文档 §2.4 P1/§7 Q3）。注意仓库自带的 `Experiments/checkpoint/Falcon/conf.yml` 反而**没有**被污染
  （仍含 repeats/shared，却多了 causal）——所以它是旧脚本/手工快照，而不是当前 `audio_train.py` 的产物
  （03 A35 已证）。
- 修正建议：§4.5/§5.5 增加一段“对评测链的后果 + 交叉引用 04 §2.4/§7 Q3”，并把它列为 Q2/Q3 的前提。

### P2（细节瑕疵，共 10 条）

| # | 位置 | 问题 | 证据 | 修正建议 |
|---|---|---|---|---|
| P2-1 | `03:131`（§2.8 表） | “on_validation_epoch_end ... lr/val_loss 全部走 `self.logger.experiment.log`（153-179）”不准确：`lr` 走 `self.log`（146-152），`val_loss` 在 `validation_step`（96-103）；`experiment.log` 只记 `learning_rate`(153-155)、`val_pit_sisnr`(156-158)、per-bucket(169)、`test_pit_sisnr`(177-179) | `av_litmodule.py:142-181` | 改成“只有 Comet 专属指标走 experiment.log；lr 仍由 self.log” |
| P2-2 | `03:224`（S10） | “33 个超参”无法复算：`audionet_config` 顶层 11 键、叶子 70、`**kwargs` 12 个 | `configs/Falcon.yml:9-101`；`runs/r2-review-03.log` §[7] | 删除该数字或定义口径（如“11 个顶层键”） |
| P2-3 | `03:176`（§2.11） | `get_layer_from_string.py` “无任何 import/调用”不准确：`look2hear/utils/__init__.py:14` 有 import、`:38` 在 `__all__` 再导出；只是无调用点 | `utils/__init__.py:14,38`（我 grep 复核） | 改为“被 utils/__init__ 再导出但全仓库无调用点” |
| P2-4 | `03:174`（§2.11） | `print_only` 使用行号列表不全（22 是 import 行；实际调用还有 71/87/112/177/182） | `grep -n "print_only(" audio_train.py` | 改为“训练链所有打印均走 print_only（45,48,54,66,71,87,112,177,182）” |
| P2-5 | `03:91`、A37 | `best_k_models.json “（6 行）”` 与 A37 `:1-7` 混用 wc/文本行两种口径（wc=6、splitlines=7、无尾换行）；并且未指出**该文件格式也不是当前代码产物**（indent=2 + `${ROOT}` 字面量 vs 代码 `indent=0` + 绝对路径） | 自写脚本 §[5]：条目行前导空格={2}，`json.dumps(indent=0)` 不同格式；`audio_train.py:186`；`08-ops-tests-docs.md:242` 已指出 | 注明两种行数口径；补一句“该文件为发布占位快照，不是本仓库脚本产物（同 08:242）” |
| P2-6 | `03:103`（§2.5） | “audio_train.py:25 直接按模块路径 import NASSv2，不经过 package `__init__`”误导：`audio_train.py:18 import look2hear.system` 已执行 `system/__init__.py`（其又 import optimizers/schedulers/litmodules） | `audio_train.py:18,25`；`system/__init__.py:1-4` | 改为“第 25 行直接 import 子模块；包 `__init__` 在第 18 行已被执行” |
| P2-7 | `03:220`（S6） | “读 use_near（:44）仅用于打印”不准确：该键仍随 `**data_config` 传入 DataModule（`audio_train.py:49`），并控制训练集 near 加载 | `audio_train.py:44-49`；`avspeech_dataset_near.py:271,284`（near.py:284 `use_near=self.use_near`） | 说明“局部变量只用于打印；配置键本身被 DataModule 消费” |
| P2-8 | `03:27-30`（§1.1 #2）、A04 | “任何非 Comet logger 都会 AttributeError”过强：本环境 `TensorBoardLogger` 在**构造**时即 `ModuleNotFoundError: Neither tensorboard nor tensorboardX`；WandB 等第三方 logger 的 experiment 常直接暴露 `.log` | 自写脚本 §[2] 输出；本仓库 requirements 无 comet/wandb | 限定为“PL 内置 CSVLogger/TensorBoardLogger 等（及本仓库依赖范围内）的 experiment 无 `.log`” |
| P2-9 | `03:25-26`（§1.1 #1） | “当前仓库状态下训练入口无法直接运行（comet_logger P0）”作为**首个**失败原因不成立：按仓库原样（`data_config.train_dir='--'`、无 `pretrain_zoo/`、无 ckpt）会在 S8 数据 setup 或 S11 视觉建模更早失败；`comet_logger` 是“环境齐备时的代码层 P0” | `configs/Falcon.yml:157-159`；`avspeech_dataset_near.py:115-121`；`frcnn_videomodel.py:86-88` | 补一句失败顺序：数据/权重缺失（更早）→ `comet_logger`(P0) → Comet API(P0) |
| P2-10 | `03:111-113`（§2.6）、A45 | “沙箱 venv 用 `--no-deps` 安装”无法从仓库验证（依赖元数据已核实：`torch-optimizer` requires `pytorch-ranger>=0.1.1`；import 确实失败） | 自写脚本 §[8]；`optimizers.py:3-19` | 改为“本沙箱 venv 缺少 `pytorch_ranger`（torch_optimizer 的声明依赖），非仓库代码问题” |

---

## 4. 遗漏的重要行为（代码里有、03 没写；按影响排序）

1. **评测/验证是整段推理，且 `AVSpeechEvalDataset` 对单声道 wav 有 `.mean(-1)` 退化风险**
   （`avspeech_dataset.py:193-194`；04 §2.4 P1/§7 Q4 已列 P0 风险）。03 只把 val/test 当普通 loaders，
   没有指出 `audio_visual_test.py` 直接索引 dataset、batch=1、无 collate 的契约 → 这也是 P1-2 的根因。
2. **config 污染 → 评测侧“空 Separator + strict=False 静默丢权重”的具体链路**（P1-3；04 已复现 49,081 参数）。
3. **NASS 平均分母恒为 B**：`NASSContrastiveLoss` 的 `n_valid` 无条件 +1（`nass_contrastive_loss.py:77`），
   无负样本的样本也计入分母、只贡献正样本项（04 §2.1.5 [验证]）。03 的 §3.4.1 公式没有写这一点，
   若他人按“仅对有效负样本平均”复现会有数值差异。
4. **`_mdcl_loss_hard` 的批量语义**：方法对 `[B,T]` 输入返回 **batch 均值**（`Falcon.py:1448,1462`），
   而 MDCL litmodule 以 `B=1` 逐样本调用后自己再求 `Σ_b/B`（`av_litmodule_mdcl.py:122-126,154-155`）。
   03 的公式把它写成样本级 `L_hard(b)`（在此调用方式下等价），但没说明方法本身是均值，复用时有歧义。
5. **DDP 数据采样语义**：`train_samples=10000` 且 `__getitem__` 忽略 idx 随机抽样（`avspeech_dataset_near.py:152-157`），
   叠加 PL 2.x 默认 `use_distributed_sampler=True` 后，每 rank 实际是“对同一数据池的 250 次随机抽样”
   （10 卡时），并非严格 epoch 遍历。03 只写“1 epoch=10000 个随机 2 s 片段”，建议补 `[推断]` 与验证。
6. **val 循环每 epoch 都跑 `test_loader`**：`validation_step` 对 `dataloader_idx==1` 只有
   `epoch%10==0` 才计算 `test_loss`，其余 9 个 epoch 仍会完整迭代 `tt`（大 test 集时每 epoch 白跑一次；
   `av_litmodule.py:125-140,214-216`）。03 写了“每 10 epoch 才算”，没提示这层成本。
7. **`audio_train.py:18 import look2hear.system` 的间接依赖链**（`system/__init__ → optimizers → torch_optimizer → pytorch_ranger`）
   会在干净环境 import 阶段就失败——比后续 P0 更早；03 §2.6 有环境备注，但 §1.1/§8 未把它列为首个阻塞。
8. **`save_hyperparameters(config_to_hparams(config))` 的副作用**：把 `training.gpus` 列表变成 tensor、
   把 `None` 变字符串写进 hparams/ckpt；且 config 里含 `options.conf_dir`、`main_args.exp_dir` 等运行路径
   （`av_litmodule.py:46,223-241`；`parser_utils` 输出），对 ckpt 可移植性有影响 —— 03 只在表里提了
   `save_hyperparameters`，未讨论这些副作用。

---

## 5. 跨模块一致性问题

1. **与 05-data-pipeline 直接冲突（最重要）**：03 §3.3:246、§5.2:414、§5.3:428-429 写 mouth `96×96` 且
   val/test `T=32000/Tv=50`；05 文档 `:11/:65/:119-120/:345/:633` 明确 mouth=`[B,50,88,88]`（96 只是存储格式，
   Dataset 裁 88）、评测整段读取、逐条索引。→ 按 P1-1/P1-2 修订后两文档一致；建议 Lead 在 FINAL 文档中
   把“mouth 88×88、train 2 s/50 帧、eval 整段变长”列为统一数据契约。
2. **与 08-ops-tests-docs 的互补（不矛盾但 03 缺）**：08 `:242` 已指出 shipped `best_k_models.json`
   的缩进 2 与 `${ROOT}` 字面量不符合当前 `audio_train.py:184-186` 的 `indent=0`+绝对路径格式；
   03 §2.4/A37 只描述文件内容、未指出“不是当前代码产物”。两文档应互相引用（P2-5）。
3. **与 04 的评测链路一致**：03 §5.5/Q2 的“conf.yml `audionet_name: SpeakerContrastiveV2` 不存在 →
   评测 AttributeError（`audio_visual_test.py:129`）”与 04 §2.4 B2/A40 完全一致；03 Q3（best_k 占位路径、
   无 ckpt）与 04 §3.4 B4 一致。补充完整断点顺序（04 §3.4）：不带 `--ckpt_path` 时先
   `_find_best_ckpt` FileNotFoundError（B4）；带 `--ckpt_path hf://…` 时在 `_load_models_from_ckpt` 的
   `getattr` 抛 AttributeError（B2）；且 `_load_models_from_ckpt` 只认 `ckpt["state_dict"]`（04 Q1，falcon.pt
   格式未知）。→ README:165 与 README:168 两条命令**都会失败**（原因不同），建议 03 §5.5 引用 04 §3.4。
4. **疑问去重**：03 Q2/Q3 与 04 Q1/Q2/Q5、08 的 Q2 高度重叠；建议 Lead 在 FINAL/`00-` 层维护“共享 Q 表”
   （falcon.pt 格式、conf.yml 类名、SDRi 口径、单声道数据），避免各文档给出不同版本答案。
5. **与 01-model-core**：03 §4.3/Q9 关于 `Falcon.forward` 调不存在的 `_mdcl_loss`、`FalconSpeakerContrastive`
   训练/推理双返回契约，应在 01 文档里有对应条目（04 只做接口引用）。建议 03 Q9 明确“由 01 在 FINAL 中确认”。
6. **conf.yml 差异的精确量化**：我程序化 diff 确认 03 的“4 处差异（audionet_name + 3 数据目录）”正确；
   除此之外只有 6 个“运行期新增键”（`audio_params.causal`、`video_params.causal`、`main_args.exp_dir`、
   `options.{ckpt_path,conf_dir,help}`），与 04 §2.4/§3.4 一致，无矛盾。

---

## 6. 给作者（train-sys）的问题清单与建议

1. **形状契约修订**：请确认是否同意把 mouth 全部改为 88×88、并区分 train（2 s/50 帧）vs val/test（整段/变长）。
   若你们坚持 96×96，需要解释为何与 05/04 的动态验证（`(25,88,88)`）不一致；建议直接采用 05 的结论。
2. **conf.yml 污染→评测后果**：你们已用隔离脚本证明“建模后键集变化”，但还没有证明“`audio_train.py`
   实际 `safe_dump` 出来的 conf.yml”就是缺键版本（当前 `main()` 在 comet_logger 处就崩，跑不到 dump）。
   能否写一个最小复现（monkeypatch 掉 Trainer/comet_logger，跑到 `:84` 为止）并确认 dump 键集？确认后请
   在 §4.5 写评测后果，并与 04 §2.4/§7 Q3 互相引用（这是本域与 train-sys 最需要对齐的一处）。
3. **`best_k_models.json` 的写法**：建议在 §2.4 补一句“该文件为发布占位快照：`indent=2`+`${ROOT}` 字面路径，
   与当前代码 `indent=0`+绝对路径不符，不是本仓库脚本产出”（与 08:242 对齐），并把“6 行/wc”与“7 文本行”
   口径写清（A37 的 `:1-7` 其实是对的）。
4. **补 DDP 数据采样语义**：请确认/补验证 PL 2.6.6 的 `use_distributed_sampler` 默认值对
   `LightningModule` 返回的 DataLoader 是否生效；若生效，10 卡时每 rank 每 epoch 是 250 个随机 batch，
   与“1 epoch=10000 segments”的叙述应合并成一句更准确的描述。
5. **Q 清单收敛**：Q1（comet_logger 替换）与 Q2/Q3 建议在 FINAL 层与 04 Q1/Q2、08 Q2 合并编号，
   避免同一问题在 3 份文档各自表述；另外 Q5（epoch 语义/早停是否触发）可用 `best_k_models.json` 的
   epoch 分布（714–742）做一次反推，写进文档会更有说服力。

---

## 7. 结束校验

- 评审对象 sha256（评审结束时重算）：`16ad59ea43a7397f1395b024e1eccbf6bec3072b0d8587324816f6f9b575780f`（与开始时一致，未改动）。
- 本评审新增文件：`docs/reviews/round1/03-training-system-review-by-loss-metric-eval.md`；
  评审支撑脚本/日志：`docs/verification/scripts/verify_r2_review_03.py`、
  `docs/verification/runs/r2-review-03.log`、`docs/verification/runs/r2-review-03-pl-contract.log`。
- 未修改任何源码，未修改 03 文档。
