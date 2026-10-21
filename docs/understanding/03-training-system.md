# 03 训练系统（入口 / 配置 / Lightning modules / 优化调度 / 训练协议）

> 方向负责人：`train-sys`。行号以当前 `main`（commit `7576854`）为准，静态核对 + 沙箱内
> 动态验证（torch 2.8.0 / pytorch_lightning 2.6.6，见 `docs/understanding/VERIFY-ENV.md`）。
> 证据标注：`[代码]` 直接读源码确认；`[推断]` 由源码合理推断但未直接运行证明；`[验证]` 实际运行脚本确认
> （脚本与日志见 `docs/verification/scripts/verify_train_*`、`docs/verification/runs/train-sys-*`）。

## 1. 子系统职责与总体位置

**结论：训练系统是一条「YAML → argparse 回填 → 手工逐件实例化 → `pl.Trainer` → 收尾导出」的
两段式（`__main__` + `main()`）脚本链，核心装配逻辑全部在 `audio_train.py` 的 203 行里；
LightningModule 只负责损失组合与日志，不负责数据/优化器构造。当前仓库状态下训练入口无法直接运行
（`comet_logger` 未定义，P0），且模型/优化器/调度器的实例化顺序带来若干可复现性缺陷。**

- 定位：训练产品线的入口层，上游是 `DataPreProcess/` 产出的 JSON 索引与 `look2hear/datas/` 数据类，
  下游是 `Experiments/checkpoint/<exp_name>/`（`conf.yml`、`epoch=*.ckpt`、`last.ckpt`、
  `best_k_models.json`、`best_model.pth`）与 Comet 日志；模型本体在 `look2hear/models/`（由 `01` 负责）。
- 代码规模：`audio_train.py` 203 行；`look2hear/system/*.py` 共 4 个文件（litmodule 240 + nass_v2 185 +
  mdcl 173 + optimizers 106 + schedulers 112 + `__init__` 11）；本域 utils 共 4 个被列出的工具文件
  （`lightning_utils` 102、`parser_utils` 151、`separator` 69、`get_layer_from_string` 42、
  `nets_utils` 502、`torch_utils` 42）。>80% 的代码是历史模板 / 死代码（详见 §2、§4.6）。

### 1.1 本轮最重要的 6 条结论（详细证据见 §3–§6）

1. **P0 不可运行（代码层）**：`audio_train.py:170` 使用 `logger=comet_logger`，该名字在全仓库无任何定义 →
   `main()` 内抛 `NameError`，`Trainer` 构造前即失败。`[代码]`+`[验证]`（`verify_train_undefined_globals.py`）。
   **但按仓库原样运行会更早失败**：`data_config.{train,valid,test}_dir='--'` 使 `datamodule.setup()` 在
   `avspeech_dataset_near.py:89-92` 抛 `FileNotFoundError`；数据就绪后，`pretrain_zoo/` 缺失也会让视觉模型在
   `frcnn_videomodel.py:86-88` 抛错。正确失败顺序：**数据/权重缺失（更早）→ `comet_logger`（P0）→ Comet 日志 API（P0）**。
2. **P0 日志耦合**：三个 litmodule 共用的 `on_validation_epoch_end` 用 Comet 专属 API
   `self.logger.experiment.log(dict)`（`av_litmodule.py:153-158,161-171,174-179`）；训练链未实例化任何 logger，
   `requirements.txt` 也没有 `comet-ml`。**在本仓库依赖范围内**：PL 内置 `CSVLogger` 实测
   `AttributeError: 'ExperimentWriter' object has no attribute 'log'`；`TensorBoardLogger` 在 PL 2.6.6 下
   构造即 `ModuleNotFoundError: Neither tensorboard nor tensorboardX`。第三方 logger（Comet/WandB）的
   `experiment` 若自带 `.log` 则可工作，但均不在 requirements 中。`[验证]`
3. **P1 配置无约束**：`training.system`、`training.parallel`、`datamodule.data_name` 三个 YAML 键
   **完全未被读取**（litmodule 类、DDP 策略、DataModule 类都是硬编码）；改它们不改变行为。`[代码]`
4. **P1 版本断档**：`audio_train.py:158-173` 写的是 PL 1.x 用法（`accelerator=None` 表示自动、
   `devices=<list>`、`logger=` 必填），在 `requirements.txt` 未钉版本的 PL 2.6.6 下：
   `accelerator=None` 抛 `ValueError`（CPU 机器必崩）；`configs/Falcon.yml` 的 `gpus: [0..9]` 在
   少于 10 卡的机器抛 `MisconfigurationException`。`[验证]`
5. **P1 复现性**：`Experiments/checkpoint/Falcon/conf.yml` 与当前代码不自洽——其
   `audionet_name: SpeakerContrastiveV2` 在仓库不存在（`getattr` 已验证 `AttributeError`），
   且它同时保留 `audio_params.repeats/shared` 又含 `audio_params.causal`，而当前模型的
   `Separator.__init__` 会 `pop` 前两者、`Falcon.__init__` 会写入后者（均已实测）。`[代码]`+`[验证]`
   **R3 执行级补充（更严重）**：最小复现已证实当前 `audio_train.py` 实际 dump 的 conf.yml **缺** `repeats/shared`；
   若把它用于评测，模型参数会从 537,697 掉到 **49,081（9.1%）** 且 `strict=False` 静默继续 → 评测给出错误结果（见 §4.5/§5.5）。
6. **P2（条件性 P0）DDP 验证日志死锁**：`av_litmodule.py:164-169` 的 per-bucket `all_gather` 只在
   「本 rank 该 bucket 非空」时调用 → 各 rank 非空 bucket 集合不同时集合通信次数不对称。2 进程 CPU
   DDP 实测：rank0 占 `{0:1,4:1}`、rank1 占 `{1:2}` 后进程挂死（>6 min 无输出，需 `kill -9`）。`[验证]`

## 2. 文件清单与逐文件解析

### 2.1 `audio_train.py`（203 行）—— 唯一的训练入口

职责：**所有对象的实例化与装配**；`main()` 串行执行「数据 → 声学模型 → 视觉模型 → 优化器 → 调度器 →
实验目录 → 损失 → 对比损失超参 → LightningModule → callbacks → Trainer → fit → 导出」。

| 区段 | 行号 | 内容 | 关键常数/备注 |
|---|---|---|---|
| 导入 | 1-27 | `torch.cuda.empty_cache()` 在 import 时执行；导入 `look2hear.{models,losses,system,videomodels,utils}`、`AVSpeechDataModuleWithNear`、`AudioVisualLightningModuleNASSv2` | `warnings.filterwarnings("ignore")`（27） |
| 参数 | 30-40 | `--conf_dir`（默认 `configs/NASSv2_lrs3_complex_nonmoving.yml`，**文件不存在**）、`--ckpt_path`（默认空） | 33、37-40 |
| `main(config, ckpt_path)` | 43-192 | 见 §3 装配链 | — |
| `__main__` | 195-203 | 二次解析 YAML 并用 `parser_utils` 回填 argparse，`arg_dic` 传入 `main` | 196-203 |

关键点：
- `torch.cuda.is_available()` 决定 `gpus=None`、`accelerator=None`（158-159）；**与 YAML 的 `gpus` 合并语义**
  是「无卡时忽略 YAML」，但有卡时完全按 YAML 列表。
- 输入输出：无返回值；副作用是写 `Experiments/checkpoint/<exp_name>/conf.yml`（82-84）、
  `best_k_models.json`（184-186）、`best_model.pth`（191-192），以及 Lightning 的 `epoch=*.ckpt`/`last.ckpt`。
- `datamodule` 实例只用于 `setup()` + `make_loader`（49-51），**从未传给 `Trainer`**；`Trainer` 通过
  LightningModule 的 `train_dataloader/val_dataloader`（`av_litmodule.py:210-216`）取数据。 `[代码]`

### 2.2 `configs/Falcon.yml`（171 行）—— 发布用论文配置模板

结构：`videonet`（2-8）/`audionet`（9-101）/`spk_contrast`（104-111）/`loss`（114-126）/
`training`（129-138）/`optimizer`（141-144）/`scheduler`（147-151）/`datamodule`（154-168）/`exp`（170-171）。
它是**给人改的模板**：`train_dir/valid_dir/test_dir` 三处占位 `--`（157-159），`pretrain` 指向尚需下载的
视觉 backbone（8）。所有键都只被 `audio_train.py` 消费；`main_args`/`options` 两个区段由 argparse
回填产生，模板里没有。

### 2.3 `Experiments/checkpoint/Falcon/conf.yml`（168 行）—— 历史实验配置快照

- 是 `audio_train.py:82-84` `yaml.safe_dump(config)` 的产物格式：键按字母序、含 `options`（125-128）与
  `main_args.exp_dir`（119-120）两个「回填区段」——这与 `parser_utils.parse_args_as_dict` 在
  Python 3.10 下的输出形状一致（`main_args={}` 后被 `audio_train.py:77-80` 写入 `exp_dir`）。
- 但内容不自洽（P1）：`audionet_name: SpeakerContrastiveV2`（88）在仓库不存在；
  数据路径是 `${DATA_ROOT}/small/DataPreProcess/LRS3_complex_nonmoving/{tr,cv,tt}`（99-102），
  与 README 的 LRS2/LRS3 布局和 `configs/Falcon.yml` 的占位 `--` 都不同；`options.conf_dir` 指向
  `${DATA_ROOT}/small/configs/small_1.yml`（127）——**说明真正的论文运行不是用 `configs/Falcon.yml`**，
  而是另一个 `small_1.yml` + `small/` 数据树，发布时做了两次不同的路径匿名化。
- `[推断]` `conf.yml` 大概率由「dump 早于模型实例化」的旧版脚本产生：当前代码里
  `Falcon.__init__` 会写入 `audio_params.causal`/`video_params.causal`（`Falcon.py:1314-1315`），
  `Separator.__init__` 会 `pop` 掉 `repeats`/`shared`（`Falcon.py:991-996`）；实测建模后这两个 dict 变成
  `[…, causal, …]` 且 `repeats/shared` 消失。而 conf.yml 同时含 `causal` 与 `repeats/shared`，
  因此**不可能**由当前代码在任意 dump 顺序下直接产生。

### 2.4 `Experiments/checkpoint/Falcon/best_k_models.json`（`wc -l`=6 / 文本 7 行，无尾换行）

`{<ckpt_path>: <monitor_value>}` 共 5 条（与 `save_top_k=5` 一致），值为 `val_loss/dataloader_idx_0`
（= −SI-SDR），最低者 `epoch=714.ckpt`（-9.984025）即「best」。`[代码]`（audio_train.py:139-147,184-186）
路径形如 `${ROOT}/Falcon/epoch=714.ckpt`，与 conf.yml 的 `${ROOT}/Experiments/checkpoint/Falcon`
前缀不一致（少 `Experiments/checkpoint`）→ 匿名化口径不一致，当前目录结构下无法 `os.path.exists` 命中；
仓库内也确实没有任何 `.ckpt` 文件（`ls` 只有两个 json/yml）。`[代码]`+`[验证]`
**该文件也不是当前代码产物**：其缩进为 2（条目行前导 2 空格）、路径为 `${ROOT}` 字面量；而
`audio_train.py:186` 是 `json.dump(best_k, f, indent=0)` + `ModelCheckpoint(dirpath=exp_dir)` 的机器绝对路径。
→ 属发布时手工/旧脚本生成的占位快照（与 `08-ops-tests-docs.md:242` 一致）。

### 2.5 `look2hear/system/__init__.py`（11 行）

只导出 `make_optimizer`、`DPTNetScheduler`、`AudioVisualLightningModule`、
`AudioVisualLightningModuleNASSv2`；**`AudioVisualLightningModuleWithMDCL` 未被导出**（1-11）。
`audio_train.py:25` 直接按**子模块**路径 import NASSv2；但包 `__init__` 早在 `:18 import look2hear.system`
时已执行（其又 import optimizers/schedulers/litmodules），并非"绕过包初始化"。`[代码]`

### 2.6 `look2hear/system/optimizers.py`（106 行）

- `make_optimizer(params, optim_name="adam", **kwargs)`（51-68）→ `get(optim_name)(params, **kwargs)`。
- `get(identifier)`（88-106）：大小写不敏感地在模块 `globals()` 中查（100-105），支持
  `torch.optim` 的 Adam/AdamW/SGD/… 与 `torch_optimizer` 的十几种（3-19）。
- 训练链实际使用：`adamw` → `torch.optim.AdamW`，`lr=1e-3, weight_decay=0.01`（Falcon.yml:142-144）。`[代码]`
- 依赖 `torch_optimizer`（3-19）：该包 0.3.0 的元数据声明依赖 `pytorch-ranger>=0.1.1`（`pip show` 实测）。
  本沙箱 venv **缺少 `pytorch_ranger`**，干净进程 `import torch_optimizer` → `ModuleNotFoundError: pytorch_ranger`；
  这是**沙箱环境缺依赖，不是仓库代码问题**（正常 `pip install -r requirements.txt` 会拉取该依赖）。`[验证]`（见 §8）

### 2.7 `look2hear/system/schedulers.py`（112 行）

- `BaseScheduler`（6-56）：自定义 step-wise 调度器基类，与 Lightning 无关。
- `DPTNetScheduler`（58-109）：Noam warmup + 指数衰减（95-109），由 `system/__init__.py` 导出，
  **但 `audio_train.py` 从不使用它**（改用 `torch.optim.lr_scheduler.ReduceLROnPlateau`，
  audio_train.py:70-74）。整个文件在当前训练链上是死代码。`[代码]`

### 2.8 `look2hear/system/av_litmodule.py`（240 行）—— 基类

| 成员 | 行号 | 语义 |
|---|---|---|
| `flatten_dict` | 9-18 | 递归扁平化 config（仅 `MutableMapping`） |
| `__init__` | 22-51 | 保存 audio/video/optimizer/loss_func/3 个 loader/scheduler/config；`default_monitor="val_loss/dataloader_idx_0"`（45）；`save_hyperparameters(config_to_hparams(config))`（46）；初始化 val/test 输出 buffer 与 5 个 `_val_bucket_outputs`（48-51） |
| `forward` | 53-64 | `mouth` 4D→5D（59-61）；`no_grad` 下过 `video_model`（62-63）；`return self.audio_model(wav, mouth_emb)`（64） |
| `training_step` | 66-84 | 解包 `mixtures, targets, mouth, *_ = batch`（67）；`self(mixtures, mouth)`（69）；targets 2D→3D（70-71）；`loss_func["train"]`（73）；`self.log("train_loss", on_epoch=True, prog_bar=True, sync_dist=True)`（75-82）；返回 `{"loss": loss}` |
| `validation_step` | 86-140 | `dataloader_idx==0` 才是 val（88-122）：`val_loss`（96-103）+ 逐样本 bucket 损失（106-120）；`epoch%10==0 and dataloader_idx==1` 时用 test loader 算 `test_loss`（125-140）；否则隐式返回 `None` |
| `on_validation_epoch_end` | 142-181 | `self.log("lr", …)` 走 PL 日志（146-152，**不是** experiment.log）；`self.logger.experiment.log`（仅 Comet 支持）记录 `learning_rate`（153-155）、`val_pit_sisnr`（156-158）、按 bucket 的 `val_pit_sisnr_*near`（161-171）、`test_pit_sisnr`（174-179）；清空 buffer（170-181） |
| `configure_optimizers` | 183-208 | 无 scheduler 直接返回 optimizer（185-186）；单 scheduler 包装成 list（188-189）；`ReduceLROnPlateau` 自动包成 `{"scheduler":…, "monitor": default_monitor}`（194-196）；dict 分支 `setdefault` monitor/frequency 后读 `sched["interval"]`（198-206） |
| `train_dataloader`/`val_dataloader` | 210-216 | `val_dataloader` 返回 **`[val_loader, test_loader]`**（216）→ val 循环有两个 dataloader |
| `on_save_checkpoint` | 218-221 | 把整个 `self.config` 存进 ckpt 的 `training_config` 键 |
| `config_to_hparams` | 223-241 | flatten 后 `None→"None"`、`list/tuple→torch.tensor` |

关键契约：`forward` 把 `self.audio_model(...)` 的返回值**原样透传**；而 `FalconSpeakerContrastive.forward`
在 `self.training` 时返回 `(separated_audios, separated_audio_embeddings)` 二元组、否则只返回波形
（`look2hear/models/Falcon_Contrastive.py:94-118`）。**基类 `training_step` 不做解包**（69,73），
所以基类 + `FalconSpeakerContrastive` 在训练时会把 tuple 喂给 `PITLossWrapper`：
实测 `AttributeError: 'tuple' object has no attribute 'size'`（`matrix.py:198`）。`[验证]`

### 2.9 `look2hear/system/av_litmodule_nass_v2.py`（185 行）—— 当前唯一在用的 module

- 头部注释（1-11）直接写明 6 步流程：forward → query=pool(sep_emb) → positive=encode(target)（detach）
  → negative=encode(near)（detach）→ NASS loss（β annealing）→ `total = pit + w_cl * cl`。
- `__init__`（39-56）：`*args/**kwargs` 直传基类；额外 7 个超参（contrastive_weight/margin/alpha_pos/
  alpha_neg/beta_init/beta_final/beta_anneal_epochs），构造 `NASSContrastiveLoss`（53）。
- `_trim_or_pad`（59-64）：把 near 波形裁/补到与 mix 同长。
- `_get_beta`（66-70）：`β(e) = β_init + (β_final−β_init)·min(e/E,1)`，`E<=0` 直接返回 `β_final`。
- `training_step`（73-185）严格解包 5 元组（74）：`mixtures, targets, mouth, _, near_audios = batch`。
- 日志键见 §3.4.2。

### 2.10 `look2hear/system/av_litmodule_mdcl.py`（173 行）—— 未接入的 MDCL 分支

- 与 NASSv2 同构：`_trim_or_pad`（60-65）、`_get_beta`（67-71）公式完全相同（默认 `beta_final=1.5`、
  `beta_anneal_epochs=200`，46-53）。
- 差别在损失空间：**不做 speaker embedding 对比**，而是逐样本调用
  `self.audio_model._mdcl_loss_hard(est_b, near_b_wavs, beta=β)`（123-125），即
  `look2hear/models/Falcon.py:1410-1462` 的「STFT 幅度域 hard-negative 排斥」；
  `total = pit_loss + mdcl_alpha · (Σ_b L_b / B)`（154-155），`mdcl_alpha`/`mdcl_margin` 来自模型属性
  （默认 0.15/0.0，`Falcon.py:1291-1292`）。
- 前 5 个 step 在 rank0 打印 `_mag_sim` 诊断（105-151）。
- **接入状态**：全仓库无任何文件 import 它（`grep` 只命中自身），`system/__init__.py` 不导出，YAML 的
  `training.system` 也未被读取 → 当前入口不可达，属历史分支。`[代码]`
- 兼容性：它调用 `self.audio_model(...)` 后**按单张量使用**（91）；与 `FalconSpeakerContrastive`
  （training 时返回 tuple）不兼容，只适配普通 `Falcon`。`[推断]`（依据 §2.8 的实测 tuple 报错）

### 2.11 本域 utils 逐文件

| 文件 | 行数 | 职责与关键签名 | 训练链使用情况 |
|---|---|---|---|
| `utils/parser_utils.py` | 151 | `prepare_parser_from_dict(dic, parser)`（5-43）把**两级** YAML 变成 argparse 组/参数；`parse_args_as_dict(parser, return_plain_args=False)`（117-151）反向变 dict；`str_int_float/str2bool/*`（46-114）类型转换 | **在用**：`audio_train.py:196-203`。注意它只展开**第二级**键：`loss.train.config` 这类第三级 dict 是「默认值整体」传递（36-39），不生成子参数 |
| `utils/lightning_utils.py` | 102 | `print_only`（22-24，`@rank_zero_only` 包装 rich.print）；`RichProgressBarTheme`（26-49）；`BatchesProcessedColumn`（51-58）；`MyMetricsTextColumn`（60-80）；`MyRichProgressBar`（82-103） | 训练链只用 `print_only`（`audio_train.py:22` import；调用点 45,48,54,66,71,87,112,177,182）。`RichProgressBarTheme/MyMetricsTextColumn/BatchesProcessedColumn` 只被 `audio_visual_test.py:48,212-217`（评测）使用；`MyRichProgressBar` **从未被使用**，且其 `_init_progress` 引用了未定义的 `MetricsTextColumn`、`CustomProgress`（93-94）→ 一旦启用必 `NameError`。`[代码]`+`[验证]`（静态未定义名分析） |
| `utils/separator.py` | 69 | `get_device`（8-18）、`Separator` 接口（21-26）、`separate/torch_separate/numpy_separate`（29-69） | **全仓库无调用点**（仅自引用）→ 死代码。`[代码]` |
| `utils/get_layer_from_string.py` | 42 | `get_layer(l_name, library=torch.nn)`（6-43）按名字取层，大小写不敏感 + difflib 建议 | 被 `look2hear/utils/__init__.py:14` import、`:38` 再导出，但**全仓库无调用点** → 对训练链是死代码。`[代码]` |
| `utils/torch_utils.py` | 42 | `pad_x_to_y`（5-10）、`shape_reconstructed`（13-16）、`tensors_to_device`（19-42） | 仅 `tensors_to_device` 被 `audio_visual_test.py:48,229,250` 使用；其余两个只被 `look2hear/utils/__init__.py:2` 再导出，无调用点。训练链不使用本文件。`[代码]` |
| `utils/nets_utils.py` | 502 | 10 个函数：`to_device`（12-31）、`pad_list`（34-61）、`make_pad_mask`（64-181）、`make_non_pad_mask`（184-270）、`mask_by_length`（273-301）、`th_accuracy`（304-324）、`to_torch_tensor`（327-392）、`get_subsample`（395-473）、`rename_state_dict`（476-487）、`get_activation`（490-502，lazy import espnet） | **训练链上 0 个函数被真正调用**：唯一被 import 的是 `make_pad_mask`（`layers/stft_tfgn.py:12` 使用于 `forward` 的 `:182`，且 `look2hear/utils/__init__.py:16` 再导出），但 `stft_tfgn.Stft` 在仓库中**从未被实例化**（`Falcon.py` 只在 `:1054/:1123` 定义并使用自己的 `STFTEncoder/STFTDecoder`）→ 其余 9 个函数完全无用。`[代码]` |

> 结论：任务要求的「`nets_utils.py` 502 行里哪些在训练链上」答案是——**没有一个函数在训练链的执行路径上被调用**；
> 只有 `make_pad_mask` 被 import（经 `layers/__init__.py:20` → `stft_tfgn.py` 的 import 副作用），而该
> 模块的 `Stft` 类没有被任何模型使用。

## 3. 关键调用链与数据流（每步 file:line）

### 3.1 阶段 A：`__main__` 的 YAML→argparse 二次解析（audio_train.py:195-203）

```
S0  python audio_train.py --conf_dir configs/Falcon.yml            # README.md:162
S1  args = parser.parse_args()                                     # audio_train.py:198
      --conf_dir 默认 configs/NASSv2_lrs3_complex_nonmoving.yml    # :33（[验证] configs/ 下只有 Falcon.yml）
      --ckpt_path 默认 ""                                          # :36-40
S2  def_conf = yaml.safe_load(open(args.conf_dir))                 # :199-200
S3  parser = prepare_parser_from_dict(def_conf, parser=parser)     # :201；parser_utils.py:5-43
      · 每个一级键建一个 argument_group（:32）
      · 二级键 → --<key>，default=<值>（:36-39）；list 值 → --<key> 整体（:33-35）
      · 三级嵌套 dict（loss.train.config 等）只作为「默认值」保留，不建子参数
S4  arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)  # :202；parser_utils.py:117-151
      · 每个 group 的 {dest: 值}（:140-142）
      · Python≥3.10：把 "positional arguments" 改名为 main_args（:143-145）→ main_args == {}（空组！）
      · “options” 组保留 --conf_dir/--ckpt_path/--help（:147-148 只对 <3.10 改名）
S5  main(arg_dic, ckpt_path=args.ckpt_path)                        # :203
```

`[验证]` 用真实 `configs/Falcon.yml` 跑 round-trip：top-level = `['options','videonet','audionet',
'spk_contrast','loss','training','optimizer','scheduler','datamodule','exp','main_args']`；
`main_args={}`；`options={'help':None,'conf_dir':...,'ckpt_path':''}`；所有配置组内容与 YAML 一致
（`loss.train` 保持为 dict，`training.early_stop`/`scheduler.sche_config`/`datamodule.data_config` 亦然）。
日志：`docs/verification/runs/train-sys-parser-roundtrip.log`。

> 设计含义：`config["main_args"]` 在 Python 3.10+ 下是空 dict，所以 `audio_train.py:77-80` 的
> `config["main_args"]["exp_dir"] = ...` 是**新建键**而不是覆盖 CLI；`--conf_dir` 只存在于
> `config["options"]` 里，随后被 `yaml.safe_dump` 一起写进 conf.yml（因此 conf.yml 有 `options` 区段）。
> `[代码]`+`[验证]`

### 3.2 阶段 B：`main()` 装配链（audio_train.py:43-192）

| 步骤 | 对象 | 证据 | 参数/行为 |
|---|---|---|---|
| S6 | 读 `use_near` | :44 | `config["datamodule"]["data_config"].get("use_near", False)`；**局部变量只用于打印**，但配置键本身仍随 `**data_config` 传入 DataModule（:49）并控制训练集 near 加载（`avspeech_dataset_near.py:271,284`） |
| S7 | `datamodule = AVSpeechDataModuleWithNear(**data_config)` | :49 | 12 个键 → 12 个形参（`avspeech_dataset_near.py:237-252`）；`train_samples` 未给 → 默认 10000（:250） |
| S8 | `datamodule.setup()` | :50 | 构建 train/val/test 三个 Dataset（:277-295）；train 用 `samples=10000`（:283） |
| S9 | `train_loader, val_loader, test_loader = datamodule.make_loader` | :51 | `make_loader` property（:331-333）→ `train_dataloader()` bs=4/shuffle/drop_last（:297-307）、`val_dataloader()` bs=1（:309-318）、`test_dataloader()` bs=1（:320-329） |
| S10 | `model = getattr(look2hear.models, "FalconSpeakerContrastive")(sample_rate=16000, **audionet_config)` | :55-58 | `audionet_config` 顶层 **11** 键（嵌套 dict 作为子模块参数整体传入，叶子键约 70 个——不宜笼统写"N 个超参"）；`models/__init__.py:2` 导入；`__all__` 只有 `AVTIGER_causal`（:4-6）但 `getattr` 不受影响 `[验证]`；参数量 537,697 全可训练 `[验证]`（Lead 亦有同结论） |
| S11 | `video_model = getattr(look2hear.videomodels, "FRCNNVideoModel")(**videonet_config)` | :61-63 | `pretrain` 非空 → `init_from` → `torch.load(pretrain_zoo/frcnn_128_512.backbone.pth.tar)`（`frcnn_videomodel.py:71-73,86-88`）；**该文件不存在**（`ls pretrain_zoo` 失败）→ 无 backbone 时构造即 `FileNotFoundError` `[代码]`；`update_frcnn_parameter` 把全部 `requires_grad` 置 False（:116-118），并重写 `train()` 冻结 BN 统计（:90-95） |
| S12 | `optimizer = make_optimizer(model.parameters(), **config["optimizer"])` | :67 | `AdamW(params, lr=1e-3, weight_decay=0.01)`（`optimizers.py:51-68,88-106`） |
| S13 | `scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=15, factor=0.5)` | :69-74 | 仅当 `sche_name` 非空（:70）；默认 mode='min'、threshold 等用 torch 默认 |
| S14 | `exp_dir = os.getcwd()/Experiments/checkpoint/<exp_name>`；`makedirs`；`yaml.safe_dump(config)` → `exp_dir/conf.yml` | :77-84 | `exp_name=Falcon`（Falcon.yml:171）；dump **发生在模型实例化之后**（S10 在前），因此 dump 时 config 已被模型构造污染（见 §4.5） |
| S15 | 损失函数字典 | :91-100 | train=`PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx", threshold_byloss=True)`；val=`PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False)`（Falcon.yml:114-126；`losses/__init__.py:1-27`；`matrix.py:372-374`）；`PITLossWrapper.forward(ests, targets)`（`pit_wrapper.py:168-236`） |
| S16 | 读 `spk_contrast` 7 个超参 | :103-110 | `loss_weight=1.0, margin=0.5, alpha_pos=1.0, alpha_neg=1.0, beta_init=0.0, beta_final=1.0, beta_anneal_epochs=200`（Falcon.yml:104-111）；**确实全部传入** litmodule（:119-136） |
| S17 | `system = AudioVisualLightningModuleNASSv2(audio_model, video_model, loss_func, optimizer, train/val/test_loader, scheduler, config, +7 个 spk 超参)` | :119-136 | 构造链：`nass_v2.py:39-56` → `av_litmodule.py:22-51`；`save_hyperparameters(config_to_hparams(config))`（:46,223-241） |
| S18 | `ModelCheckpoint(exp_dir, filename="{epoch}", monitor="val_loss/dataloader_idx_0", mode="min", save_top_k=5, verbose=True, save_last=True)` | :139-147 | checkpoint 目录=exp_dir；best_k=5；同时存 last.ckpt |
| S19 | `if early_stop: callbacks.append(EarlyStopping(**config["training"]["early_stop"]))` | :149-150 | monitor 同上、mode=min、patience=30、verbose=True（Falcon.yml:134-138） |
| S20 | `logger_dir = cwd/Experiments/tensorboard_logs`，`makedirs(<logger_dir>/<exp_name>)` | :153-154 | **建目录但从不使用**；没有实例化任何 logger `[代码]`（全仓库 grep 无 `CSVLogger/TensorBoardLogger/CometLogger`） |
| S21 | `gpus = config["training"]["gpus"] if cuda else None`；`distributed_backend = "cuda" if cuda else None` | :158-159 | 有卡时 devices=[0..9]；无卡时 devices=None + accelerator=None |
| S22 | `trainer = pl.Trainer(max_epochs=1000, callbacks=[ckpt, es], default_root_dir=exp_dir, devices=gpus, accelerator=distributed_backend, strategy=DDPStrategy(find_unused_parameters=True), limit_train_batches=1.0, gradient_clip_val=5.0, logger=comet_logger, sync_batchnorm=True, num_sanity_val_steps=0)` | :161-173 | **`logger=comet_logger` 未定义 → NameError（P0）**；未设置 `accumulate_grad_batches`（默认 1）、`val_check_interval`（默认 1.0 = 每 epoch）、`check_val_every_n_epoch`（默认 1）、`precision`（默认 32） |
| S23 | `trainer.fit(system[, ckpt_path=...])` | :176-180 | 见 S24/B；val 循环 = `[val_loader, test_loader]`（`av_litmodule.py:214-216`） |
| S24 | resume：仅当 `ckpt_path` 非空且 `os.path.exists(ckpt_path)` | :176-179 | 路径不存在时**静默从头训练**（无警告）；`~`/环境变量不展开（`os.path.exists` 不 expanduser）`[代码]` |
| S25 | 收尾：`best_k = {k: v.item() ...}` → `exp_dir/best_k_models.json`（`indent=0`） | :184-186 | `ModelCheckpoint.best_k_models` 的值在 PL 2.6.6 是 **Tensor**（实测），`.item()` 成立；PL 1.x 可能是 float，`.item()` 会崩（版本风险 `[推断]`） |
| S26 | `torch.load(checkpoint.best_model_path)` → `system.load_state_dict(ckpt["state_dict"])` → `system.cpu()` → `system.audio_model.serialize()` → `exp_dir/best_model.pth` | :188-192 | serialize 见 `base_model.py:80-96`；产物只含音频模型 state_dict + `model_name/model_args/infos` |

### 3.3 阶段 C：一次训练 step 的数据流（以在用的 NASSv2 为例）

```
train_loader (batch_size=4, drop_last=True, collate_with_near)        # avspeech_dataset_near.py:297-307,23-43
  → batch = (mixtures[B,T=32000], targets[B,T], mouth[B,Tv=50,88,88], filenames, near_audios[list[list[T]]])
    （磁盘 mouth 帧为 96×96，Dataset 的 train 管线 RandomCrop 到 88×88；见 §5.3）
  → nass_v2.training_step(batch, batch_nb)                            # :73
      · targets[B]→[B,1]                                              # :76-77
      · mouth[B,50,88,88]→[B,1,50,88,88]；no_grad 过 video_model     # :82-87
      · est_sources, sep_embs = audio_model(mixtures, mouth_emb)      # :90
            est_sources [B,1,T]；sep_embs [B,n_src=1,112,T',129]
            （112 = audio_bn_params.out_chan=Falcon.yml:23；129 = audio_params.n_freqs=Falcon.yml:39；
             形状为 Lead 实测，见 docs/verification/runs/lead-model-params.log）
      · pit_loss = loss_func["train"](est_sources, targets)           # :95（负 SNR，PIT 退化 n_src=1）
      · β = _get_beta()                                               # :99,66-70
      · embed_query = pool_spk_embed(sep_embs[:,0])   [B,D] 有梯度    # :102-103
      · embed_pos   = pool_spk_embed(encode_stft(targets[:,0]))  detached   # :106-109
      · embed_negs[k] = pool_spk_embed(encode_stft(near_wavs[k])) detached  # :112-132
      · cl_loss = NASSContrastiveLoss(query, pos, negs, β)            # :134；nass_contrastive_loss.py:31-81
      · total = pit_loss + contrastive_weight(1.0) * cl_loss          # :136-138
      · 相似度监控：cos(query,pos)、cos(query,neg) avg/max、neg_active_ratio  # :140-167
      · 写 7+N 个 train_* 标量（见 §3.4.2）                            # :169-183
      · return {"loss": total_loss}                                   # :185
```

**注意：以上是训练路径；验证/评测路径不同**：`val_dataloader` 用 `AVSpeechEvalDataset`，整段读取
（`avspeech_dataset.py:179-192`，`rand_start=0, stop=None`），不做 2 s 窗，`T`/`Tv` 随音频时长可变
（`Tv≈25·T/16000`）；评测入口 `audio_visual_test.py` 更彻底——**不走 DataLoader**，直接逐条索引 dataset
（`:189,192-193`），`mouth` 以 numpy `(Tv,88,88)→[None,None]` 送视觉模型（`:229-231`）。`[代码]`

`[验证]` 用 dummy audio/video 模型 + 真 `PITLossWrapper/NASSContrastiveLoss` 跑 PL Trainer 11 epoch：
`train_loss_epoch=0.9065 == train_pit_loss_epoch(0.6565) + train_spk_contrast_loss_epoch(0.25)`
（weight=1.0），证明损失组合公式就是 `pit + w_cl·cl`。日志 `docs/verification/runs/train-sys-litmodules.log`。

### 3.4 损失组合与日志（三个 module 的精确差异）

#### 3.4.1 损失公式

- **PIT 项（三个 module 相同）**：`PITLossWrapper` + `pairwise_neg_snr`（train）、`pairwise_neg_sisdr`（val）；
  `n_src=1` 时置换只有一个（PIT 退化为逐样本负 SNR/SI-SDR 均值）。`matrix.py:372-374`、`pit_wrapper.py:168-236`。
- **NASS v2 对比项**（`av_litmodule_nass_v2.py:97-138` + `nass_contrastive_loss.py:31-81`）：
  ```
  L_cl = α_pos·(1 − cos(q,p)) + α_neg·Σ_k w_k·relu(cos(q,n_k) − margin)
  w_k  = softmax_k(β·cos(q,n_k))            # K=1 时跳过 softmax，退化为权重 1
  K=0 时 L_neg=0
  L    = L_PIT + w_cl·L_cl                  # w_cl = spk_contrast.loss_weight
  β(e) = β_init + (β_final−β_init)·min(e/β_anneal_epochs, 1)；E<=0 → β_final
  ```
  有效值（Falcon.yml:104-111 经 audio_train.py:103-110,119-136 注入）：`w_cl=1.0, margin=0.5,
  α_pos=α_neg=1.0, β: 0→1.0, E=200 epochs`。
  `[验证]` β 数值：e=0→0.0, e=1→0.005, e=50→0.25, e=100→0.5, e=200→1.0, e=250→1.0；E=0→β_final。
- **MDCL 项**（`av_litmodule_mdcl.py:96-155` + `Falcon.py:1410-1462`）：
  ```
  L_hard(b) = Σ_k softmax_k(β·s_k)·relu(s_k − mdcl_margin),  s_k = _mag_sim(est_b, near_k)（STFT 幅度域 SI-SNR, dB）
  K=0→0；K=1→relu(s−margin)
  L = L_PIT + mdcl_alpha·(Σ_b L_hard(b)/B)      # mdcl_alpha 默认 0.15、mdcl_margin 默认 0.0
  ```
  β 调度公式与 NASSv2 完全一致（`av_litmodule_mdcl.py:67-71`），但默认 `β_final=1.5, E=200`（:49-51）。
  `Falcon.yml` 没有 `mdcl_margin/mdcl_alpha` 键 → 若启用 MDCL 实际用 0.0/0.15。
- **基类**：只有 `L = L_PIT`（`av_litmodule.py:73`）。
- **平均口径（易错点）**：`NASSContrastiveLoss` 的 `n_valid` 对每个 batch 样本**无条件 +1**
  （`nass_contrastive_loss.py:77`）→ 平均分母恒为 B（K=0 的样本只贡献正项、不被跳过）。
- **MDCL 的批量语义**：`Falcon._mdcl_loss_hard` 对 `[B,T]` 输入返回 **batch 均值**（`Falcon.py:1448,1462`）；
  MDCL litmodule 以 `B=1` 逐样本调用后再求 `Σ_b/B`（`av_litmodule_mdcl.py:122-126,154-155`），
  本调用方式下与"样本级 L_hard"等价，但复用该方法时需注意它自带 batch 归约。

#### 3.4.2 指标名与去向（`[验证]` 实际跑 fit 后 `trainer.logged_metrics` / fake-Comet 收到的键）

| 来源 | 键 | API | 备注 |
|---|---|---|---|
| NASSv2 training_step | `train_loss`（on_epoch+prog_bar）、`train_pit_loss`、`train_spk_contrast_loss`、`train_beta`、`train_cos_sim_pos`、`train_cos_sim_neg_avg`、`train_cos_sim_neg_max`、`train_neg_active_ratio`、`train_n_samples_{0,1,2,3,4plus}_near` | `self.log` | `on_step=True, on_epoch=True, sync_dist=True` |
| MDCL training_step | `train_loss`、`train_pit_loss`、`train_mdcl_loss`、`train_beta`、`train_mdcl_{0,1,2,3,4plus}_near`、`train_n_samples_*` | `self.log` | 同 DDP 固定键集合 |
| val（基类） | `val_loss` → 实际键名 `val_loss/dataloader_idx_0` | `self.log` | **被 ModelCheckpoint / EarlyStopping / ReduceLROnPlateau 三者同时 monitor** |
| test-in-val（每 10 epoch） | `test_loss` → `test_loss/dataloader_idx_1` | `self.log` | epoch%10==0 才**计算/记录**（`av_litmodule.py:125`）；但 `tt` loader **每个 epoch 仍会被完整迭代**（validation loop 含两个 dataloader，`:214-216`）→ 大 test 集时每 epoch 白跑一遍数据加载。`[代码]` |
| on_validation_epoch_end | `lr` | `self.log` | `prog_bar=True` |
| Comet 专属 | `learning_rate`、`val_pit_sisnr`、`val_pit_sisnr_{0,1,2,3,4plus}_near`、`test_pit_sisnr` | `self.logger.experiment.log({...})` | **只有 Comet API 支持**；CSVLogger 实测 AttributeError |

### 3.5 收尾/产物文件与格式

| 文件 | 生产者 | 格式/内容 |
|---|---|---|
| `Experiments/checkpoint/<exp>/conf.yml` | audio_train.py:82-84 | 回填后的完整 config（含 `options`、`main_args.exp_dir`） |
| `.../epoch=<N>.ckpt`、`last.ckpt` | PL ModelCheckpoint | 标准 Lightning ckpt；额外含 `training_config`（`av_litmodule.py:218-221`） |
| `.../best_k_models.json` | audio_train.py:184-186 | `{ckpt_path: float(val_loss)}`，5 条 |
| `.../best_model.pth` | audio_train.py:188-192 | `{model_name, state_dict(音频模型), model_args, infos{torch_version, pytorch_lightning_version}}`（`base_model.py:80-96`） |

> 注意：`best_model.pth` 只含音频模型，与 README:107 「发布权重含 `audio_model.*` 和 `video_model.*`」
> 的 Lightning-ckpt 口径**不是同一格式**（→ Q7）。

## 4. 关键实现细节与设计动机

### 4.1 为什么是「脚本装配」而不是 LightningDataModule
`AVSpeechDataModuleWithNear` 是**普通类**，没有继承 `pl.LightningDataModule`（`avspeech_dataset_near.py:226-252`），
所以 audio_train 手工 `setup()` 后取 loader，再挂到 LightningModule 上（`train_dataloader/val_dataloader`
都在 module 里）。好处是少一层 PL 生命周期；代价是 `Trainer` 不感知 datamodule，`datamodule.data_name`
等配置键变成装饰品（§1.1-3）。

### 4.2 模型/优化器/调度器的「外置构造」顺序
`optimizer` 在 `model` 之后、LightningModule 之前构造（:67），然后作为属性传入 system
（`av_litmodule.py:37`），`configure_optimizers` 原样返回 `[self.optimizer]`（:208）。参数对象同一份，
因此语义等价于「在 configure_optimizers 里建优化器」，但有一个副作用：**优化器状态在 resume 前后依赖
PL 把 ckpt 的 optimizer state 灌回同一批对象**（S24），这是标准路径，没发现问题。`[代码]`+`[推断]`

### 4.3 `FalconSpeakerContrastive.forward` 的 training/eval 双契约（与 model-core 交叉核对）
- 训练：返回 `(separated_audios, separated_audio_embeddings)`（`Falcon_Contrastive.py:116-117`）。
- eval：只返回 `separated_audios`（:118）。
- 这决定了三个 litmodule 的写法：基类在 training_step 不解包（坏），NASSv2 显式解包（好），MDCL 按单张量
  （只适配普通 `Falcon`）。**验证**：把 tuple 喂 `PITLossWrapper` → `AttributeError: 'tuple' object has no
  attribute 'size'`（`matrix.py:198`）。
- 同时注意 `Falcon.forward`（`Falcon.py:1464-1480`）在 `near_audio is not None` 时会调 `self._mdcl_loss(...)`（:1480），
  而类里只有 `_mdcl_loss_hard`（:1410）——该分支在任何入口都没被触发（训练调用只传 2 个位置参数），
  与 Lead 在 `VERIFY-ENV.md` §5 的发现一致。`[代码]`

### 4.4 对比损失为什么「同空间 + detach」
positive/negative 都过 `encode_stft` 并 `no_grad`（`nass_v2.py:106-109,129-131`），只有 query 分支有梯度；
embedding 由 `pool_spk_embed` 做 L2 normalize（`Falcon_Contrastive.py:82-91`），符合
`NASSContrastiveLoss` 文档假设（`nass_contrastive_loss.py:10`）。负样本来自 `near.json`
（近场干扰说话人，`DataPreProcess/generate_near_json.py`，由 05 负责），按每样本 K 个、K>4 归入 bucket 4。

### 4.5 配置被模型构造原地污染（影响 conf.yml 与复现）
`audio_train.py:57` 把 `config["audionet"]["audionet_config"]` 以 `**` 展开传给模型——顶层复制但**嵌套
dict 仍是同一对象**。模型构造时：
- `Falcon.__init__` 写入 `self.audio_params['causal']=...`、`self.video_params['causal']=...`
  （`Falcon.py:1314-1315`）→ config 多出两个键；
- `Separator.__init__` `pop("repeats")`、`pop("shared")`（`Falcon.py:991-996`）→ config 少两个键。
而 conf.yml dump 在建模之后（:82-84）→ 落盘配置 = 被污染后的配置。`[验证]` 实测建模前后
`audio_params`: `[…] → [causal…]`（`repeats/shared` 消失）、`video_params` 同理；参数量 537,697。
这就是 `audio_visual_test.py:118` 注释「Use deepcopy to prevent Separator.__init__'s dict.pop from
polluting config」的来由。**对评测链的具体后果（R3 执行级验证，脚本 `verify_train_r3_*.py`，日志
`docs/verification/runs/train-sys-r3-exec.log`）**：

1. [B] 用 monkeypatch 掉 `pl.Trainer`/`comet_logger` 的方式把**真实 `audio_train.main()`** 跑到 `:84`，
   实际 dump 的 conf.yml 中 `audio_params`/`video_params` 均 `causal=True`、`repeats/shared` 缺失；
2. [C] 用该 dump 的 `audionet_config` 重建模型（复刻 `_load_models_from_ckpt` 的 `getattr(...)(**cfg)`）：
   参数从 **537,697 → 49,081（9.1%）**；`load_state_dict(full_sd, strict=False)` 返回
   **missing=0、unexpected=505**（即 ckpt 中 505 个张量无处可去），而 `audio_visual_test.py:133-140`
   只打印数量后继续；
3. [C] `Separator` 的 `fusion_repeats/audio_repeats/video_repeat` 全为 0（`Falcon.py:991-996`）→
   `Separator.forward` 的 0 次循环退化为恒等（`Falcon.py:1025-1054`），模型仍能输出 `[1,1,T]`
   → **评测静默给出错误结果**（不是崩溃）。
4. 对照：仓库自带 conf.yml 仍含 `repeats/shared`（重建=537,697 参数），它是旧快照；而**当前代码的新 dump**
   才会触发上述退化。→ 与 04 文档 §2.4/§7 Q3 一致，是 Q2/Q3 的前提。

### 4.6 遗留/半成品代码清单（本域）
1. `comet_logger` 缺定义 + Comet-only 日志 API + 无 comet-ml 依赖（P0 组合）。
2. `logger_dir/tensorboard_logs` 建目录但不建 logger（:153-154）。
3. `training.system` / `training.parallel` / `datamodule.data_name` 未被读取（Falcon.yml:130,132,155）。
4. `AudioVisualLightningModuleWithMDCL` 未导出、未接入（`system/__init__.py:1-11`；全仓库无 import）。
5. `DPTNetScheduler` 导出但不用（`schedulers.py:58-109`）。
6. `MyRichProgressBar` 未使用且引用未定义名（`lightning_utils.py:93-94`，静态分析验证）。
7. `nets_utils.py` 中 9/10 函数无调用点；`separator.py`、`get_layer_from_string.py` 整体无调用点。
8. `Falcon.forward` 的 `_mdcl_loss` 调用指向不存在的方法（`Falcon.py:1480`）。
9. `configure_optimizers` 的 dict-scheduler 分支直接读 `sched["interval"]`（`av_litmodule.py:201`），
   用户传不带 `interval` 的 dict 会 `KeyError`（当前配置走 `ReduceLROnPlateau` 实例分支，未触发）。

### 4.7 与 README「训练协议」的逐项核对（README.md:161）

| README 声明 | 代码事实 | 结论 |
|---|---|---|
| DDP | `strategy=DDPStrategy(find_unused_parameters=True)` 硬编码（audio_train.py:167）；YAML `parallel: ddp` **未读取** | ✅ 行为一致，但配置键无效 |
| AdamW lr 1e-3 | `optim_name: adamw`、`lr: 0.001`（Falcon.yml:141-144）→ `torch.optim.AdamW` | ✅ |
| ReduceLROnPlateau | `sche_name: ReduceLROnPlateau`，`patience=15, factor=0.5`（Falcon.yml:147-151）；monitor 由 litmodule 注入 `val_loss/dataloader_idx_0` | ✅（README 未给 patience/factor；patience 与 early-stop 的 30 不同） |
| early stopping 30 | `EarlyStopping(monitor="val_loss/dataloader_idx_0", mode="min", patience=30)`（Falcon.yml:134-138） | ✅ |
| grad clip 5 | `gradient_clip_val=5.0`（audio_train.py:169） | ✅ |
| （README 未提）weight decay | `weight_decay=0.01`（Falcon.yml:144） | 补充 |
| （README 未提）batch/epoch | `batch_size=4`（Falcon.yml:164）×10 卡 = 全局 40；`epochs=1000`（:133）；无梯度累积（accumulate=1）；1 epoch=10000 个随机 2 s 片段（`train_samples` 默认 10000，`avspeech_dataset_near.py:250,283`；`__getitem__` 忽略 idx 随机采样，:152-157） | **README 未说明**；且 `epoch` 不是「遍历数据集」语义 |
| （README 未提）验证集组成 | `val_dataloader=[cv, tt]`（`av_litmodule.py:216`）；tt 上的 `test_loss` 每 10 epoch 才计算 | 补充 |
| gpus 10 卡 | `gpus: [0..9]`（Falcon.yml:131） | ✅ 与 config 一致，但**不是所有机器可跑**（实测 2 卡机器 PL 报错） |
| DDP 下常量 | `sync_batchnorm=True`、`num_sanity_val_steps=0`、`limit_train_batches=1.0`（:168-172） | README 未提 |

### 4.8 PL 2.x 数据采样与 hparams 副作用（R3 新增）

- **`use_distributed_sampler` 默认 True**（PL 2.6.6 `Trainer.__init__` 参数默认值经 `inspect.signature` 实测），
  且对 LightningModule 返回的 DataLoader 同样生效：我的 2 进程 CPU DDP 探针
  （`verify_train_ddp_bucket_gather.py`）中两个 rank 拿到的 val 样本下标是**分片交错**的
  （rank0 得样本 0、2；rank1 得 1、3）→ 证明 PL 把原 DataLoader 重新包了 `DistributedSampler`
  （PL 源码 `pytorch_lightning/trainer/connectors/data_connector.py:236-243`）。`[代码]`+`[验证]`
- 结合 `train_samples=10000`、batch=4、10 卡：每 rank 每 epoch = `10000/10/4 = 250` 个 batch，
  且 `__getitem__` 忽略 idx 随机抽样（`avspeech_dataset_near.py:152-157`）→ **不是**严格意义的
  "遍历数据集一次"，而是"每 rank 对同一数据池做 250 次随机抽样"。`[推断]`（10 卡组合未实跑）
- **`save_hyperparameters(config_to_hparams(config))` 的副作用**（`av_litmodule.py:46,223-241`）：
  flatten 后 `list/tuple → torch.tensor`（如 `training.gpus`）、`None → "None"`（如 `enc_dec_params.act_type`），
  且 config 含 `options.conf_dir`、`main_args.exp_dir` 等**运行期路径** → 均写入 hparams 与每个 ckpt 的
  `hyper_parameters`，影响 ckpt 的可移植性/可读性。`[代码]`

## 5. 对外接口契约

### 5.1 YAML 键 → 消费者（训练入口）

| YAML 路径 | 消费者（file:line） | 语义 / 约束 |
|---|---|---|
| `videonet.videonet_name` / `videonet_config` | audio_train.py:61-63 | `getattr(look2hear.videomodels, name)`；`pretrain` 必须是存在的文件 |
| `audionet.audionet_name` / `audionet_config` | :55-58 | `getattr(look2hear.models, name)`；`sample_rate` 由 datamodule 提供 |
| `spk_contrast.*` | :103-110 → :129-135 | 7 个超参全部进 NASSv2 构造；缺键用 litmodule 默认 |
| `loss.train/val.{loss_func,sdr_type,config}` | :91-100 | 两次 `getattr(look2hear.losses, …)` |
| `training.gpus` | :158 | 仅在 CUDA 可用时使用；列表长度须 ≤ 实际卡数 |
| `training.epochs` | :162 | `max_epochs` |
| `training.early_stop` | :149-150 | 原样 `EarlyStopping(**dict)` |
| `training.system` / `training.parallel` | **无** | 死键（误导） |
| `optimizer.*` | :67 | `make_optimizer(model.parameters(), **dict)`；须含 `optim_name` |
| `scheduler.sche_name` / `sche_config` | :70-74 | `getattr(torch.optim.lr_scheduler, name)(optimizer=…, **dict)`；`sche_name: ""` → 不建调度器 |
| `datamodule.data_name` | **无** | 死键 |
| `datamodule.data_config.*` | :49 | **逐键匹配** `AVSpeechDataModuleWithNear.__init__`（`[验证]` 12/12 命中，仅 `train_samples` 用默认）；见 §5.3 |
| `exp.exp_name` | :78, :154 | 仅决定目录名 |

### 5.2 LightningModule 构造契约（audio_train.py:119-136 → nass_v2.py:39-56 → av_litmodule.py:22-33）

```
AudioVisualLightningModuleNASSv2(
  audio_model: nn.Module,           # 必须实现 forward(wav, mouth_emb)->(est[B,n_src,T], emb) when training,
                                    #   以及 pool_spk_embed / encode_stft
  video_model: nn.Module,           # forward(mouth[B,1,Tv,88,88])->[B,512,Tv]（96×96 存储→Dataset 裁 88）
  loss_func: dict {"train","val"} of nn.Module(ests,targets)->scalar,
  optimizer: torch.optim.Optimizer,
  train_loader/val_loader/test_loader: DataLoader,
  scheduler: LRScheduler | list | None,
  config: dict,                     # 全量 config，存 hparams 与 ckpt
  contrastive_weight,margin,alpha_pos,alpha_neg,beta_init,beta_final,beta_anneal_epochs: float/int
)
```

### 5.3 数据 batch 契约（与 05-data-pipeline 交叉核对）

| loader | batch 结构 | 证据 |
|---|---|---|
| train | `(mixtures[B,32000], sources[B,32000], mouths[B,50,88,88], filenames[list[str]], near_audios[list[list[Tensor[32000]]]])` | `collate_with_near`，`avspeech_dataset_near.py:23-43`；train DS `__getitem__` 返回 5 元组（:223）；2 s 随机窗（:156-157）、50 帧（:78） |
| val / test | `(mixture[T], source[T], mouth[Tv,88,88], filename[str], n_near:int)`，`T`/`Tv` **整段变长** | `AVSpeechEvalDataset.__getitem__` 返回值（`avspeech_dataset.py:212`）；整段读取（:179-192）；`Tv≈25·T/16000`；`n_near` 从 `<mix>_meta.json` 的 `components[type=="near"]` 计数（:202-212） |
| 形状 | train：`T=2.0*16000=32000`、`Tv=2.0*25=50`；val/test：整段变长 `T`、`Tv≈25·T/16000`（如 1 s→25 帧） | `avspeech_dataset_near.py:77-78,156-157`；`avspeech_dataset.py:179-192` |
| 评测额外契约 | **不走 DataLoader**：逐条 `test_set[idx]`、无 collate/batch；`mouth` 为 numpy `(Tv,88,88)` → `[None,None]` 送视觉模型 | `audio_visual_test.py:189,192-193,229-231` |

`[验证]` 用 `ast` 提取 `AVSpeechDataModuleWithNear.__init__` 形参并与 `Falcon.yml` 的 `data_config` 比对：
`config keys not in signature: []`；`signature params without config: ['train_samples']`（默认 10000）。
→ **任务要求的「是否完全匹配」答案是：完全匹配（12/12），无拼写错误；唯一未显式给出的是 `train_samples`。**

### 5.4 日志/监控契约
- `self.log` 的键 → PL logger（本应写 TensorBoard/CSV/Comet）与 `trainer.logged_metrics`；
  被 callbacks 依赖的只有 `val_loss/dataloader_idx_0`。
- `self.logger.experiment.log({...})` 的键 → 仅 Comet；这是当前代码的**硬依赖**（P0，见 §8）。
- val 的两个 dataloader 决定键后缀：`val_loss/dataloader_idx_0`、`test_loss/dataloader_idx_1`
  （`[验证]` 单进程 CPU fit 实测键名）。

### 5.5 产物契约（下游消费者）
- `audio_visual_test.py` 依赖 `Experiments/checkpoint/<exp>/{conf.yml,best_k_models.json,*.ckpt}`
  与 `conf.yml` 的 `audionet_name`；**当前 conf.yml 的 `SpeakerContrastiveV2` 不存在** → 评测会
  `AttributeError`（`audio_visual_test.py:129`；`[验证]` `getattr`）。→ Q2/Q3（需 eval owner 确认）。
- **两种失败形态**（与 04 文档 §2.4/§3.4 对齐）：① 用仓库自带 conf.yml（含 repeats/shared，但
  `audionet_name` 不存在）→ `getattr` AttributeError；② 若修好类名、但用当前 `audio_train.py` 新 dump 的
  conf.yml（缺 repeats/shared）→ 模型退化为 **49,081 参数（9.1%）**，`load_state_dict(strict=False)`
  只报 505 unexpected keys、不中断，评测**静默错误**（§4.5）。README:165 / README:168 两条评测命令
  在本仓库状态下都会失败或给出错误结果。
- `best_model.pth` 的格式（serialize）与 README:107 的发布权重格式不同 → Q7。

## 6. 可验证断言表

| 编号 | 断言 | 证据 path:line | 验证方式 | 置信度 |
|---|---|---|---|---|
| A01 | 训练入口唯一，且默认 `--conf_dir` 指向不存在的文件 | audio_train.py:30-35；`configs/` 只有 Falcon.yml | 静态 + `ls configs/` | 高 |
| A02 | `main()` 使用未定义名字 `comet_logger`，Trainer 构造前即 `NameError` | audio_train.py:170；全仓库仅此一处 | symtable 未定义名分析（`verify_train_undefined_globals.py`） | 高 |
| A03 | 入口不实例化任何 logger（无 CSV/TensorBoard/Comet），只建了未使用的 `tensorboard_logs` 目录 | audio_train.py:153-154；`grep CometLogger\|CSVLogger\|TensorBoardLogger` 为空 | grep + 静态 | 高 |
| A04 | litmodule 的 epoch-end 使用 Comet 专属 `logger.experiment.log(dict)`；限于本仓库依赖范围（PL 内置 logger）必失败，第三方 logger 若自带 `.log` 可工作 | av_litmodule.py:153-158,161-171,174-179；CSVLogger/TensorBoardLogger 实测 | 读码 + 实测 | 高 |
| A05 | `config["main_args"]` 在 Python 3.10 下为空 dict，`exp_dir` 是新建键 | parser_utils.py:143-148；audio_train.py:77-80 | 真实 YAML round-trip 运行 | 高 |
| A06 | YAML→argparse 只展开两级键，三级 dict 作为整体默认值 | parser_utils.py:31-43 | round-trip：`loss.train` 仍为 dict | 高 |
| A07 | `datamodule.data_config` 12 个键与 `AVSpeechDataModuleWithNear.__init__` 完全匹配；`train_samples` 用默认 10000 | Falcon.yml:156-168；avspeech_dataset_near.py:237-252 | AST 形参提取比对 | 高 |
| A08 | `AVSpeechDataModuleWithNear` 不是 `pl.LightningDataModule`，实例从未传给 Trainer | avspeech_dataset_near.py:226-252；audio_train.py:49-51 vs 161-173 | 读码 | 高 |
| A09 | 音频模型经 `getattr(look2hear.models, name)` 构造，`__all__` 陈旧但不影响 | audio_train.py:55-58；models/__init__.py:1-6 | 读码 + `getattr` 实测 | 高 |
| A10 | 视觉模型在 `pretrain` 存在时加载 ckpt 并冻结全部参数（requires_grad=False）与 BN 统计 | frcnn_videomodel.py:71-73,86-88,90-95,116-118；Falcon.yml:8 | 读码 | 高 |
| A11 | `pretrain_zoo/` 不存在 → 按 Falcon.yml 从零训练会在构造视觉模型时 FileNotFoundError | frcnn_videomodel.py:87；`ls pretrain_zoo` 失败 | 静态 + ls | 高 |
| A12 | 优化器在 LightningModule 外创建，`configure_optimizers` 原样返回 | audio_train.py:67；av_litmodule.py:183-208 | 读码 + 实测返回结构 `([opt],[{scheduler,monitor}])` | 高 |
| A13 | 调度器是 `torch.optim.lr_scheduler.ReduceLROnPlateau(patience=15, factor=0.5)`，monitor 由 litmodule 注入 | audio_train.py:70-74；Falcon.yml:147-151；av_litmodule.py:194-196 | 读码 + 实测 | 高 |
| A14 | `ModelCheckpoint` monitor=`val_loss/dataloader_idx_0`, mode=min, save_top_k=5, save_last=True, filename="{epoch}" | audio_train.py:139-147 | 读码 + PL 2.6.6 实测构造/落盘 | 高 |
| A15 | `EarlyStopping` 参数来自 `training.early_stop`（patience=30），与调度器 patience=15 不同 | audio_train.py:149-150；Falcon.yml:134-138 | 读码 + 实测 | 高 |
| A16 | `gradient_clip_val=5.0`、`limit_train_batches=1.0`、`num_sanity_val_steps=0`、`sync_batchnorm=True` | audio_train.py:167-172 | 读码 + 实测 | 高 |
| A17 | 无 `accumulate_grad_batches` → 默认 1；全局 batch = 4×卡数 | audio_train.py:161-173；Falcon.yml:164,131 | 读码 + PL Trainer 参数语义 | 高 |
| A18 | `accelerator=None` 在 PL 2.6.6 非法 → CPU 机器训练必崩 | audio_train.py:158-159,166；PL 实测 `ValueError` | 运行 `verify_train_pl_contract.py` | 高 |
| A19 | `devices=[0..9]` 在 2 卡机器抛 MisconfigurationException | Falcon.yml:131；PL 实测 | 运行 `verify_train_pl_contract.py` | 高 |
| A20 | resume 只在 `os.path.exists(ckpt_path)` 时生效，否则静默从头训练；不展开 `~`/环境变量 | audio_train.py:176-180 | 读码 | 高 |
| A21 | `val_dataloader` 返回 `[val_loader, test_loader]`；`test_loss` 每 10 epoch 才算 | av_litmodule.py:214-216,125-140 | 读码 + 11-epoch fit 实测 | 高 |
| A22 | 多 dataloader 下日志键自动带后缀 `val_loss/dataloader_idx_0`、`test_loss/dataloader_idx_1` | av_litmodule.py:96-103,131-138 | PL 2.6.6 实测键名 | 高 |
| A23 | 基类 training_step 不解包 audio_model 的 2 元组返回 → 与 FalconSpeakerContrastive 不兼容 | av_litmodule.py:69,73；Falcon_Contrastive.py:116-118；matrix.py:198 | `PITLossWrapper(tuple_ests)` 实测 AttributeError | 高 |
| A24 | NASSv2 显式解包 `(est, emb)` 并只用 `emb[:,0]` 当 query | av_litmodule_nass_v2.py:90,102-103 | 读码 + fit 实测 | 高 |
| A25 | NASSv2 total = PIT + weight·CL（weight=1.0 时数值可分解） | av_litmodule_nass_v2.py:136-138 | 实测 0.9065 = 0.6565+0.25 | 高 |
| A26 | β 调度为线性 in epoch，clamp 到 β_final；E≤0 直接 β_final | av_litmodule_nass_v2.py:66-70 | 实测 e=0/1/50/100/150/200/250 与 E=0 | 高 |
| A27 | 生效的 β 范围是 0→1.0/200 epoch（不是 litmodule 默认的 0→1.5/100） | audio_train.py:103-110,129-136；Falcon.yml:104-111 | 读码 + 实测 | 高 |
| A28 | NASS 损失 = α_pos(1−cos+) + α_neg·Σ softmax(βcos)·relu(cos−margin)，K=1 跳过 softmax、K=0 无负项 | nass_contrastive_loss.py:52-81 | 读码（含公式注释 :1-11） | 高 |
| A29 | MDCL total = PIT + mdcl_alpha·mean_b(L_hard)，β 公式与 NASSv2 相同、默认 β_final=1.5 | av_litmodule_mdcl.py:67-71,153-155；Falcon.py:1410-1462 | 读码 + 3-epoch fit 实测 | 高 |
| A30 | MDCL 模块在仓库中不可达（未导出、无 import） | system/__init__.py:1-11；`grep av_litmodule_mdcl` 仅自命中 | grep 静态 | 高 |
| A31 | `training.system`、`training.parallel`、`datamodule.data_name` 未被读取 | audio_train.py 全文 grep；Falcon.yml:130,132,155 | 静态 | 高 |
| A32 | `spk_contrast` 7 个键确实被读取并逐一传入 litmodule | audio_train.py:103-110,129-135 | 读码 | 高 |
| A33 | exp_dir = `cwd/Experiments/checkpoint/<exp_name>`，conf.yml 在建模后落盘 | audio_train.py:77-84 | 读码 | 高 |
| A34 | 建模会原地修改 config：新增 `causal`、删除 `repeats/shared` | Falcon.py:991-996,1314-1315；audio_train.py:55-58 | 实测建模前后 dict 键变化（`verify_train_config_mutation.py`） | 高 |
| A35 | 因此 conf.yml 的键集无法由当前代码直接复现（含 `causal` 又含 `repeats/shared`） | conf.yml:9-44；A34 | 对比实测 | 中高（推断） |
| A36 | conf.yml 的 `audionet_name: SpeakerContrastiveV2` 在仓库不存在 → eval/重训 `getattr` 抛 AttributeError | conf.yml:88；`getattr` 实测 | 运行 import probe | 高 |
| A37 | best_k_models.json 5 条、值为负 val_loss（−SI-SDR），最低 epoch=714；文件为 `indent=2`+`${ROOT}` 占位，**不是**当前代码 `indent=0`+绝对路径产物 | best_k_models.json:1-7；audio_train.py:184-186；08:242 | 读文件 + 读码 + 格式对照 | 高 |
| A38 | `ModelCheckpoint.best_k_models` 的值在 PL 2.6.6 是 Tensor，`.item()` 成立 | audio_train.py:184；PL 实测 | 运行 `verify_train_pl_contract.py` | 高 |
| A39 | 收尾 `torch.load` 未指定 `weights_only`，但普通 Lightning ckpt 在 torch 2.8 下可加载 | audio_train.py:188；torch 2.8 实测 | 运行 `verify_train_pl_contract.py` | 中（实测用微型 ckpt） |
| A40 | 本域 utils 中训练链实际用到的只有 `parser_utils.*` 与 `print_only`；`nets_utils` 0 个函数被调用 | utils/__init__.py:1-16；nets_utils.py 全函数；layers/stft_tfgn.py:12,182 | 全仓库 grep + 读码 | 高 |
| A41 | `nets_utils.py` 502 行、10 个函数中仅 `make_pad_mask` 被 import，且其唯一使用者 `Stft` 从未被实例化 | nets_utils.py:64；stft_tfgn.py:12,182；layers/__init__.py:20；Falcon.py:1054,1123 | 读码 + grep | 高 |
| A42 | `separator.py` 全仓库无调用点；`get_layer_from_string.get_layer` 被 `utils/__init__.py:14,38` import/再导出但无调用点 | separator.py:29-69；get_layer_from_string.py:6-43；utils/__init__.py:14,38 | grep | 高 |
| A43 | `lightning_utils.MyRichProgressBar` 引用未定义名 `MetricsTextColumn/CustomProgress` | lightning_utils.py:93-94 | symtable 未定义名分析 | 高 |
| A44 | 基类 `on_validation_epoch_end` per-bucket `all_gather` 依赖各 rank bucket 占用对称，否则 DDP 挂死 | av_litmodule.py:164-169 | 2 进程 CPU DDP 实测挂死（rank0 {0,4} vs rank1 {1}） | 高（机制）；真实规模影响中（推断） |
| A45 | `torch_optimizer` 0.3.0 元数据依赖 `pytorch-ranger>=0.1.1`；沙箱 venv（--no-deps）缺它导致 `import look2hear.system` 失败 | optimizers.py:3-19；pip metadata | pip show/`import` 实测 | 高 |
| A46 | `configure_optimizers` 的 dict-scheduler 分支读取未 setdefault 的 `sched["interval"]` → 潜在 KeyError | av_litmodule.py:198-206 | 读码 | 中（未触发路径） |
| A47 | 当前 `audio_train.py` 实际 dump 的 conf.yml 缺 `audio_params/video_params.repeats/shared`、含 `causal` | audio_train.py:55-84；Falcon.py:991-996,1314-1315 | R3 最小复现（monkeypatch Trainer/comet_logger 跑到 :84） | 高 |
| A48 | 用 A47 的 dump 重建模型：参数 537,697→49,081（9.1%）；`strict=False` 得 missing=0/unexpected=505；forward 仍输出 `[1,1,T]` → 评测静默错误 | Falcon.py:991-996,1025-1054；audio_visual_test.py:128-140 | R3 执行（`verify_train_r3_polluted_eval.py`） | 高 |
| A49 | 仓库自带 conf.yml 仍含 repeats/shared（重建=537,697）→ 是旧快照，非当前代码产物 | conf.yml:9-44；A35 | R3 执行（`verify_train_r3_eval_pollution.py`） | 高 |
| A50 | PL 2.6.6 `use_distributed_sampler` 默认 True 且对 LightningModule 的 DataLoader 生效（2 进程 DDP 样本按 rank 分片） | PL 源码 data_connector.py:236-243；`verify_train_ddp_bucket_gather.py` | inspect 默认值 + DDP 探针日志 | 高（机制）/中（10 卡组合为推断） |
| A51 | `NASSContrastiveLoss` 平均分母恒为 B（`n_valid` 每样本无条件 +1，K=0 不被跳过） | nass_contrastive_loss.py:48-81 | 读码 | 高 |
| A52 | `save_hyperparameters(config_to_hparams(config))`：`gpus`→tensor、`None`→"None"、hparams 含 `options/main_args` 运行路径 | av_litmodule.py:46,223-241 | 读码 | 高 |
| A53 | 失败顺序：`data_config` 占位 `'--'`/缺 `pretrain_zoo` 比 `comet_logger` 更早失败 | avspeech_dataset_near.py:89-92；frcnn_videomodel.py:86-88；audio_train.py:170 | 读码 + 最小复现（R3-B 数据就绪后才走到 Trainer） | 高 |
| A54 | `TensorBoardLogger` 在 PL 2.6.6 本环境构造即 `ModuleNotFoundError`（缺 tensorboard） | PL 实测 | 运行 | 高 |

## 7. 未解疑问 / 需其它子系统确认的问题（Q1..Q11）

- **Q1（作者/Lead）** `comet_logger` 本应指向什么？仓库没有 comet-ml 依赖，是否应改成
  `CSVLogger`/`TensorBoardLogger`？若改，`av_litmodule.py` 三处 `logger.experiment.log` 必须重写
  （`self.log(...)` 只能在 step 上下文用；per-bucket 指标需换 `self.log_dict` 或自定义 Metric）。
  这是训练能否跑起来的第一阻塞点。
- **Q2（loss-metric-eval / ops-tests-docs）** `Experiments/checkpoint/Falcon/conf.yml` 的
  `audionet_name: SpeakerContrastiveV2` 不存在；README:165 的评测命令会 `AttributeError`
  （`audio_visual_test.py:129`）。评测侧是否有 alias/兼容逻辑？我未读完评测入口，请 eval owner 确认。
  **R3 补充**：与 04 §2.4/§3.4 对齐后，两条 README 评测命令各有独立失败点——不带 `--ckpt_path` 时
  `_find_best_ckpt` 先 `FileNotFoundError`（无 ckpt）；带 `hf://` 时在 `_load_models_from_ckpt` 的
  `getattr` 抛 AttributeError；若修好类名，还有 §4.5 的 49,081 参数静默错误。
- **Q3（eval / 作者）** `best_k_models.json` 的路径前缀 `${ROOT}/Falcon/` 与 conf.yml 的
  `${ROOT}/Experiments/checkpoint/Falcon` 不一致（少一级目录）。仓库内无 ckpt，评测的
  `_find_best_ckpt` 回退链在真实环境能否命中？发布权重（hf://）与本地 ckpt 的期望格式是否不同？
  **R3 补充**：该文件缩进=2、路径含 `${ROOT}` 字面量，与当前代码 `indent=0`+绝对路径不符 → 它是发布占位
  快照（§2.4/A37，与 08:242 一致）；即使路径可解析，其条目也指向仓库内不存在的 ckpt。
- **Q4（作者）** val per-bucket `all_gather` 的 DDP 条件（A44）。真实 10 卡 + 完整 cv 集时，每个 rank
  每个 bucket 是否必然非空？若不能保证，应改成「固定 5 次 gather，空 bucket 传零张量」。
  这决定它是 P2 还是 P0。
- **Q5（作者）** 论文训练协议的 epoch 语义：配置 `epochs: 1000`，但 1 epoch = 10000 个随机片段
  （`train_samples` 默认），并非遍历数据集；实际 best ckpt 在 epoch 714/735/738/741/742（B37），
  early-stop patience=30 是否真的触发过？论文写的训练轮数/总步数是多少？
  **R3 推断**：best=epoch 714，top-5 中最后一次更优是 epoch 742，之后无更优记录 → patience=30 下
  早停点约在 epoch 772；总步数 ≈ 772×250 ≈ 19.3 万/rank（10 卡）。`[推断]`（依据 best_k_models.json + 配置）
- **Q6（作者/ops）** `training.system`、`training.parallel`、`datamodule.data_name` 是否被仓库外的
  启动脚本（例如内部 `small_1.yml` 工作流）读取？还是纯遗留？
- **Q7（eval / ops）** 训练收尾产出的 `best_model.pth`（`serialize()` 格式，只含音频模型，
  `base_model.py:80-96`）与 README:107 所说「包含 `audio_model.*` 和 `video_model.*`」的发布格式不同。
  哪个是 HuggingFace `falcon.pt` 的真实格式？`best_model.pth` 是否还有下游消费者？
- **Q8（作者）** `AudioVisualLightningModuleWithMDCL`（β_final 默认 1.5）与 NASSv2（config β_final=1.0）
  哪条是论文/消融最终方案？MDCL 是否只为消融保留、不打算接入发布入口？
- **Q9（model-core）** `Falcon.forward` 的 `self._mdcl_loss(...)`（Falcon.py:1480）在任何入口都不可达，
  且方法不存在；是否应删除或改名 `_mdcl_loss_hard`？另外 `FalconSpeakerContrastive` 的
  training/eval 双返回契约（Falcon_Contrastive.py:116-118）是否论文推理路径也依赖（Studio 是否会
  因 `model.eval()` 得到单返回值）——归 01/06 交叉确认。
- **Q10（data-pipe 05）——R3 已结案**：`_meta.json` **不由本仓库任何脚本生成**（全仓库 grep：仅
  `process_lrs23.py:35-43`、`generate_near_json.py:27-36`、`avspeech_dataset.py:201-212` **读取**，无写入者）。
  按 README 数据准备流程（README:148-158）操作时 meta 缺失 → `process_lrs23.py:69-71` 全 fallback →
  `avspeech_dataset_near.py:190-191` 跳过 fallback → **所有训练样本 K=0** → NASS 负项恒 0（只剩正项）、
  MDCL 关闭、val `n_near` 全 0 → 分桶指标全部塌进 `val_pit_sisnr_0near`。结论：对比学习的负样本依赖一个
  **仓库外资产**；`val_pit_sisnr_*near` 只有在真实 meta 存在时才有意义（与 05 文档 §3.2(d)/§8.6 一致）。
- **Q11（Lead/作者）** PL 2.x `use_distributed_sampler=True` 下（§4.8），"1 epoch=10000 个随机段"的表述与
  "每 rank 250 个随机 batch" 需要在论文/README 澄清吗？它影响总步数与 `ReduceLROnPlateau` 的步进语义
  （每 epoch 一次，`av_litmodule.py:194-196`）。

## 8. 本轮验证活动记录

环境：`/root/.venvs/falcon/bin/python` = Python 3.10.18 + torch 2.8.0+cu128（`cuda.is_available()=True`，
`device_count=2`）+ pytorch_lightning 2.6.6；按 `VERIFY-ENV.md` 注入
`PYTHONPATH=$PWD/docs/verification/stubs:$PWD`（timm/typeguard stub）。系统 python3=3.8 未使用。
脚本：`docs/verification/scripts/verify_train_*.py`；日志：`docs/verification/runs/train-sys-*.log`。

| # | 命令 | 结果摘要 |
|---|---|---|
| V1 | `python docs/verification/scripts/verify_train_pl_contract.py` | ModelCheckpoint(`filename="{epoch}"`,monitor=...,save_top_k=5,verbose,save_last) ✅；EarlyStopping(patience=30,verbose) ✅；CPU+CSVLogger 双 val loader fit ✅；实际键名 `['val_loss/dataloader_idx_0','test_loss/dataloader_idx_1','lr',...]`；`best_k_models` 值类型 Tensor 且 `.item()` ✅；`torch.load`（默认 weights_only）加载 Lightning ckpt ✅；`Trainer(accelerator=None)` → `ValueError`；`devices=[0..9]` 于 2 卡机器 → `MisconfigurationException` |
| V2 | 用真实 YAML 跑 `prepare_parser_from_dict` + `parse_args_as_dict` | 组键/`main_args={}`/`options`/嵌套 dict 保持，全部符合 §3.1（日志 `train-sys-parser-roundtrip.log`） |
| V3 | `verify_train_config_mutation.py`（`PYTHONPATH=…`） | 建模前 `audio_params` 11 键（含 repeats/shared）→ 建模后 10 键（含 causal、无 repeats/shared）；参数量 537,697 全可训练 |
| V4 | `verify_train_litmodules.py` | dummy audio/video + 真损失：NASSv2 11-epoch fit ✅，`train_loss 0.9065 = pit 0.6565 + cl 0.25`；β={0:0.0,1:0.005,50:0.25,100:0.5,150:0.75,200:1.0,250:1.0}，E=0→1.5；`configure_optimizers` 返回 `([opt],[{monitor:'val_loss/dataloader_idx_0', ReduceLROnPlateau}])`；Comet 风格键 `val_pit_sisnr(_*near)/test_pit_sisnr/learning_rate`；CSVLogger 下 `AttributeError: 'ExperimentWriter' object has no attribute 'log'`；`PITLossWrapper(tuple_ests)` → `AttributeError: 'tuple' object has no attribute 'size'`；MDCL 3-epoch fit ✅（`train_mdcl_loss/train_beta` 键存在） |
| V5 | `verify_train_ddp_bucket_gather.py`（2 进程 CPU DDP，gloo） | 构造 rank0 bucket `{0:1,4:1}`、rank1 `{1:2}` 后**进程挂死**（>6 min 无输出，`timeout`/`kill -9` 终止）→ 证实 `av_litmodule.py:164-169` 的集合通信不对称死锁机制 |
| V6 | `verify_train_undefined_globals.py`（symtable） | `audio_train.py` 未定义全局名恰为 `comet_logger`（`main` 内）；`lightning_utils.py` 未定义 `MetricsTextColumn`、`CustomProgress` |
| V7 | `verify_train_imports.py`（stub `pytorch_ranger` 后） | `import look2hear.{utils,system,losses}`、`look2hear.system.av_litmodule_nass_v2/_mdcl`、`optimizers`、`schedulers` 全 ✅；`look2hear.models` 有 `Falcon/FalconSpeakerContrastive`、**无 `SpeakerContrastiveV2`**；干净进程 `import torch_optimizer` ❌ `ModuleNotFoundError: pytorch_ranger`（venv 用 --no-deps 安装所致，非仓库 bug） |
| V8 | `getattr(look2hear.models,"SpeakerContrastiveV2")` | `AttributeError`（配合 V7） |
| V9 | `ast` 提取 `AVSpeechDataModuleWithNear.__init__` 形参 vs Falcon.yml `data_config` | 12/12 匹配；`train_samples` 走默认 10000 |
| V10 | `ls configs/`、`ls pretrain_zoo`、`ls Experiments/checkpoint/Falcon/`、`grep comet/CSVLogger/TensorBoardLogger` | configs 只有 Falcon.yml（默认 conf 不存在）；pretrain_zoo 不存在；实验目录只有 conf.yml+best_k_models.json（无 ckpt）；仓库无 logger 实例化、无 comet-ml |
| V11 | **R3** `verify_train_r3_convdump.py`：monkeypatch `pl.Trainer`/`comet_logger`，把真实 `audio_train.main()` 跑到 `:84` | `[B]` 成功 dump conf.yml → `audio_params/video_params: causal=True, repeats=False, shared=False`；`options={ckpt_path,conf_dir,help}`、`main_args={exp_dir}`；拦截到 Trainer kwargs（devices=[0], accelerator='cuda', gradient_clip_val=5.0, limit_train_batches=1.0, sync_batchnorm=True, num_sanity_val_steps=0, max_epochs=1000, logger=None）→ 证 A33/A47/A53；日志 `train-sys-r3-exec.log` |
| V12 | **R3** `verify_train_r3_polluted_eval.py`：用 V11 dump 重建模型 + `strict=False` 加载完整 state_dict + eval 前向 | `[C]` 参数 537,697 → **49,081（9.1%）**；`missing=0, unexpected=505`（ckpt 侧多 505 个张量）；退化模型 `forward` 仍输出 `(1,1,32000)` → 静默错误（A48） |
| V13 | **R3** `verify_train_r3_eval_pollution.py`：用仓库自带 conf.yml 的 audionet_config 重建 | `[A]` 参数仍 537,697、strict 加载 missing=0/unexpected=0；该 conf 同时含 causal 与 repeats/shared → 旧快照（A49） |
| V14 | **R3** `inspect.signature(pl.Trainer.__init__)` + V5 DDP 探针 | `use_distributed_sampler` 默认 **True**；2 进程 DDP 下 rank0 得样本 0/2、rank1 得 1/3（分片交错）→ A50 |
| V15 | **R3** `pl.loggers.TensorBoardLogger("/tmp/tb")`；`grep print_only audio_train.py`；audionet_config 顶层键计数 | TensorBoardLogger → `ModuleNotFoundError: Neither tensorboard nor tensorboardX`（A54）；print_only 调用点 9 处；audionet_config 顶层 11 键（P2-2/P2-4 证据） |
| V16 | **R3** 读码核对：`NASSContrastiveLoss.n_valid` 无条件 +1；`_mdcl_loss_hard` 返回 batch 均值；`save_hyperparameters` 副作用 | A51/A52 成立（`nass_contrastive_loss.py:77`；`Falcon.py:1448,1462`；`av_litmodule.py:46,223-241`） |

**失败/限制说明**：
1. 沙箱 venv 缺 `torchvision`/`cv2`/`espnet`/`pytorch_ranger` → 无法 import `look2hear.datas`（数据类）
   与直接 import `look2hear.system`（需 stub `pytorch_ranger`），因此**没有**真实 dataloader 参与的端到端
   训练（这也符合「无数据/无权重不训练」的约定）。V4 用 dummy 模型隔离验证了 litmodule 逻辑。
2. 无 LRS 数据、无 `pretrain_zoo/`、无 `.ckpt` → 无法验证真实 batch 形状/dtype、ckpt resume 全链路、
   10 卡 DDP 行为；相关结论标注为 `[代码]`/`[推断]`。
3. `torch.load` 的 A39 只用微型 ckpt 验证；真实 ckpt 含 `training_config`（纯 Python 基本类型）与大量
   tensor，`weights_only=True` 下预期可加载，但未用真实 ckpt 验证（标 `[推断]` 风险）。
4. V5 的挂死测试不会在 CI 中重跑（脚本保留，标注需 `timeout`+`kill`）。

## 9. 变更日志（Changelog）

- 2026-10-21 / R1 初稿（`train-sys`）：建立 8 节模板 + Changelog；完成 `audio_train.py` 的 YAML→Trainer
  逐步装配链（S0–S26）、三个 litmodule 的逐行解析与损失公式、两份配置的逐键差异表、utils 使用情况审计、
  46 条可验证断言（R3 后 54 条）；动态验证覆盖 parser round-trip、PL 2.6.6 契约、dummy 模型 fit（NASSv2/MDCL）、
  config 污染、2 进程 DDP 死锁复现。发现 P0：`comet_logger` 未定义、Comet 日志 API 硬依赖；P1：
  死配置键、PL 2.x 版本断档、conf.yml 与代码不自洽。

- 2026-10-21 / **R3 修订**（`train-sys`，依据 `docs/reviews/round1/03-training-system-review-by-loss-metric-eval.md`，
  评审结论 P0=0 / P1=3 / P2=10）：
  1. **形状契约修正**：mouth 全部 96×96 → **88×88**（§3.3/§5.2/§5.3）；val/test 由"2 s/50 帧"改为
     "整段读取、变长 T/Tv"（§3.3 新增说明、§5.3 拆行、§5.5 增评测契约）。[P1-1/P1-2]
  2. **conf.yml 污染 → 评测后果**：新增执行级验证并写入 §4.5/§1.1#5/§5.5/A47–A49——当前代码 dump 的 conf
     缺 `repeats/shared`；若用于评测，参数 537,697→49,081（9.1%）、`strict=False` 报 505 unexpected keys
     但不中断、模型仍能输出 → 静默错误（与 04 §2.4/§3.4 对齐）。[P1-3]
  3. **新增 §4.8**：PL 2.x `use_distributed_sampler=True` 的数据采样语义（DDP 探针实证分片、每 rank
     250 随机 batch）与 `save_hyperparameters` 副作用（gpus→tensor、None→"None"、含运行期路径）。[遗漏 5/8]
  4. **断言表**：修订 A04/A37/A42 口径，新增 A47–A54（含 R3 两项执行级验证、失败顺序、NASS 分母、DDP 采样、
     hparams 副作用、TensorBoardLogger 构造失败）。§8 新增 V11–V16。
  5. **P2 逐条修正**：§1.1 失败顺序、§2.4 行数口径+格式说明、§2.5 import 链、§2.6 依赖措辞、
     §2.8 on_validation_epoch_end 日志归属、§2.11 get_layer/print_only、S6 use_near、S10 超参口径、
     §3.4.1 NASS 分母/MDCL 批量语义、§3.4.2 test_loader 每 epoch 遍历成本、§5.5 两种失败形态、§7 Q10 结案 + Q11。
  6. 未修改任何源码；新增验证脚本 `docs/verification/scripts/verify_train_r3_*.py`、
     日志 `docs/verification/runs/train-sys-r3-exec.log`。

### 9.1 对 R2 评审意见的逐条响应（评审人：loss-metric-eval）

| 评审条目 | 处置 | 本文位置 |
|---|---|---|
| P1-1 mouth 96×96 → 88×88 | ✅ 已采纳 | §3.3、§5.2、§5.3 |
| P1-2 val/test 应为整段/变长 | ✅ 已采纳（并补"评测不走 DataLoader"） | §3.3 注、§5.3、§5.5 |
| P1-3 conf 污染 → 评测后果 | ✅ 已采纳并升级为执行级验证（49,081 参数/静默错误） | §1.1#5、§4.5、§5.5、A48/A49 |
| P2-1 on_validation_epoch_end 日志归属 | ✅ 已修正（lr 走 self.log） | §2.8 |
| P2-2 "33 个超参" | ✅ 删除，改为顶层 11 键口径 | S10 |
| P2-3 get_layer "无 import" | ✅ 改为"被 __init__ 再导出但无调用点" | §2.11、A42 |
| P2-4 print_only 行号不全 | ✅ 补全 9 个调用点 | §2.11 |
| P2-5 best_k "6 行" 口径 + 非当前产物 | ✅ 已采纳（两种口径 + 格式说明） | §2.4、A37 |
| P2-6 import 链误导 | ✅ 已修正（:18 已执行包 __init__） | §2.5 |
| P2-7 use_near 仅打印 | ✅ 已修正（局部变量 vs 配置键） | S6 |
| P2-8 "任何非 Comet logger"过强 | ✅ 已限定为本仓库依赖范围 | §1.1#2、A04 |
| P2-9 失败顺序 | ✅ 已采纳（数据/权重更早失败） | §1.1#1、A53 |
| P2-10 "--no-deps" 不可验证 | ✅ 已改为"沙箱缺 pytorch_ranger（声明依赖）" | §2.6 |
| 遗漏 1 eval 整段 + mono 风险 | ✅ 整段契约已补；mono P0 引 04/05（跨域） | §3.3 注、§5.3 |
| 遗漏 2 conf 污染→评测链路 | ✅ 见 P1-3 | §4.5 |
| 遗漏 3 NASS 分母恒为 B | ✅ 已补 | §3.4.1、A51 |
| 遗漏 4 `_mdcl_loss_hard` 自带 batch 归约 | ✅ 已补 | §3.4.1 |
| 遗漏 5 DDP 采样语义 | ✅ 已补（含探针证据） | §4.8、A50 |
| 遗漏 6 val 每 epoch 迭代 test_loader | ✅ 已补 | §3.4.2 |
| 遗漏 7 import 链更早失败 | ✅ 已并入失败顺序 | §1.1#1、A53 |
| 遗漏 8 save_hyperparameters 副作用 | ✅ 已补 | §4.8、A52 |
| Q1 形状契约是否同意 | ✅ 接受 04/05/评审结论（88×88、train vs val/test 分离） | §3.3/§5.3 |
| Q2 要求最小复现 dump 键集 | ✅ 已完成（monkeypatch 跑到 :84，V11 日志） | §4.5 [B]、A47 |
| Q3 best_k 格式说明 | ✅ 已采纳 | §2.4、A37 |
| Q4 use_distributed_sampler 默认值 | ✅ 已验证 True 且对 module 提供的 loader 生效 | §4.8、A50、V14 |
| Q5 Q 清单收敛 / epoch 反推 | ◐ 部分采纳：已补早停≈772 轮推断；Q 编号合并留 FINAL（Lead） | §7 Q5/Q11 |
