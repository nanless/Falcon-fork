# 03 训练系统深挖（audio_train.py / litmodules / configs / PL 2.6.6）

> 组员：`train-system`（共享任务 `task-3`）· 日期：2026-10-24
> 仓库：`/root/code/github_repos/Falcon-fork`，分支 `repro/phase2`，HEAD **`2beec38`**
> 环境：conda `falcon` = `/root/miniforge3/envs/falcon/bin/python`（Py 3.11.17 / torch 2.8.0+cu128 /
> pytorch-lightning 2.6.6 / 2×A800）；所有命令前置 `PYTHONPATH=$PWD`，未使用 `/usr/bin/python3`。
> 证据标记：`[代码]` 直接读源码；`[验证]` 本轮实际运行（命令+输出见 §7）；`[推断]` 由代码/数据合理推断未直接运行；
> `[文档]` 引自仓库既有报告（Phase1/Phase2/旧 03 文档）。
> 本文件只读分析、只写本文件；未改源码、未启动真实训练、未下载数据。

---

## 0 一句话结论

**在 `repro/phase2` HEAD `2beec38` 上，训练入口已从"发布态不可运行"变为"可运行且已实测跑通"**：
`audio_train.py` 重写为 417 行（原 203 行），F1（`comet_logger` 未定义）、F2（Comet 专属
`logger.experiment.log`）、F5（eval mono 0 维）、F6（PL 2.x devices）、F9（conf 被 pop 污染）、F17（日志键统一）
以及新增的 `training.accumulate_grad_batches` 均已修复/落地；本机 2 卡 DDP + grad-accum 3 epoch smoke
（`configs/LRS2_2Mix_Paper_2gpu_smoke.yml`）与 100 epoch 单卡 LRS2 训练均有产物证据。
**仍开放**：F3/F4（发布快照 `Experiments/checkpoint/Falcon/conf.yml` 的类名/占位路径仍在，仅靠评测入口绕开）、
F15（DDP per-bucket `all_gather` 非对称死锁风险仍在代码里，`av_litmodule.py:170-176`）；
**"从零训练 RealSSA-Sim"尚缺 4 件事**：① tr/cv 全量数据从 TOS 归档恢复到本地并转换（本地只有 tt 9,169 条）；
② 仓库没有 RealSSA-Sim 训练配置（只有 LRS2/Falcon 模板）；③ near 负样本方案（官方不发布 near 波形，现为 K=0）；
④ 8 卡节点（本机 2 卡且常被占）。论文级 RealSSA 训练超参（epochs/数据混合/通道策略）未公开，只能按
`configs/Falcon.yml` 快照推断。

---

## 1 边界与关键路径

### 1.1 范围

本域文件（行数，HEAD `2beec38`）：

| 文件 | 行数 | 角色 | 本轮结论 |
|---|---:|---|---|
| `audio_train.py` | 417 | 唯一训练入口（装配全部对象） | 已重写：3 个纯函数 helper + `main()` |
| `look2hear/system/av_litmodule.py` | 247 | LightningModule 基类（val/test/日志/优化器） | F2 已修；F15 未修 |
| `look2hear/system/av_litmodule_nass_v2.py` | 185 | **当前唯一在用**的训练 module（NASS 对比） | 逻辑未变，继承基类修复 |
| `look2hear/system/av_litmodule_mdcl.py` | 173 | MDCL 分支（未接入，无 import） | 仍未接入 |
| `look2hear/system/optimizers.py` | 106 | `make_optimizer` / `get` | 未变 |
| `look2hear/system/schedulers.py` | 111 | `DPTNetScheduler`（死代码） | 未变 |
| `look2hear/utils/lightning_utils.py` | 102 | `print_only`（训练链唯一在用） | `MyRichProgressBar` 仍引用未定义名（死代码） |
| `look2hear/utils/parser_utils.py` | 151 | YAML→argparse 回填 | 未变 |
| `configs/Falcon.yml` | 173 | 发布配置模板（10 卡 / 数据占位 `--`） | 数据占位仍在 |
| `configs/LRS2_2Mix.yml` | 154 | LRS2 单卡/2 卡（100 epoch，单向 target） | Phase2 实测用 |
| `configs/LRS2_2Mix_Paper_2gpu.yml` | 155 | 2×bs4×accum5 = 有效 40 | Phase2 新增 |
| `configs/LRS2_2Mix_Paper_2gpu_smoke.yml` | 155 | 3 epoch / train_samples 2000 | 已实测通过 |
| `configs/LRS2_2Mix_Paper_8gpu.yml` | 153 | 8×bs5 = 全局 40 | 未在 8 卡上跑过 |
| `Experiments/checkpoint/Falcon/conf.yml` | 168 | 发布态历史快照（含 `${DATA_ROOT}`/`SpeakerContrastiveV2`） | F3/F4 未修 |
| `Experiments/checkpoint/Falcon/best_k_models.json` | 5 条 | 发布态占位（`${ROOT}` 路径） | F4 未修 |
| `tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh` | 63 | 8 卡启动脚本（自检→后台启动） | 可用，有 3 个小坑（§5.3） |
| `tools/repro/phase2/monitor_lrs2_paper.py` | 59 | 读 `metrics.csv` 输出 epoch/val/ETA | 硬编码 `version_0`（§6） |

### 1.2 装配链（`main()`，`audio_train.py:235-406`）

```
S0  __main__ (:409-417): argparse --conf_dir（默认 configs/Falcon.yml :32-39）/ --ckpt_path (:40-44)
      yaml.safe_load → prepare_parser_from_dict → parse_args_as_dict → main(arg_dic, ckpt_path)
S1  exp_dir = cwd/Experiments/checkpoint/<exp_name>；makedirs；deepcopy 后写 conf.yml (:240-248)
S2  use_near 打印 (:250-251)；DataModule(**data_config).setup()；make_loader (:253-257)
S3  音频模型 getattr(look2hear.models, audionet_name)(sample_rate, **audionet_config) (:260-264)
S4  视觉模型 getattr(look2hear.videomodels, videonet_name)(**videonet_config) (:267-269)
S5  优化器 make_optimizer(model.parameters(), **config["optimizer"]) (:272-273)
S6  调度器 getattr(torch.optim.lr_scheduler, sche_name)(optimizer=..., **sche_config) (:275-280)
S7  损失 train=PIT(pairwise_neg_snr) / val=PIT(pairwise_neg_sisdr) (:283-296)
S8  spk_contrast 7 超参读取 (:299-312)
S9  system = AudioVisualLightningModuleNASSv2(...) (:315-332)
S10 ModelCheckpoint(exp_dir, filename="{epoch}", monitor="val_loss/dataloader_idx_0",
      mode=min, save_top_k=5, save_last=True) + 可选 EarlyStopping (:335-346)
S11 logger = build_logger(config, exp_dir)（F1） (:349-350)
S12 accelerator/devices/num_devices/strategy = resolve_accelerator_and_devices(training)（F6） (:353-359)
S13 accumulate_grad_batches = int(training.get("accumulate_grad_batches",1) or 1) (:361-362)
S14 pl.Trainer(max_epochs, accumulate_grad_batches, callbacks, default_root_dir=exp_dir,
      devices, accelerator, strategy, limit_train_batches=1.0, gradient_clip_val=5.0,
      logger=(logger or False), sync_batchnorm=True, num_sanity_val_steps=0) (:363-377)
S15 trainer.fit(system[, ckpt_path=...])；ckpt 不存在时 WARNING 并从头训练 (:380-386)
S16 收尾：best_k_models.json（indent=0）→ best_model.pth（audio_model.serialize()） (:391-406)
```

要点：`AVSpeechDataModuleWithNear` **不是** `pl.LightningDataModule`（`avspeech_dataset_near.py:226`），
实例从未传给 `Trainer`；Trainer 通过 litmodule 的 `train_dataloader/val_dataloader` 取数据
（`av_litmodule.py:217-223`）。`config["training"]["system"]`、`["parallel"]`、`datamodule.data_name`
**仍是死键**（全文件 grep 无读取点，`audio_train.py` 只用 `training.gpus/logger/epochs/early_stop/accumulate_grad_batches`）。

### 1.3 一次训练 step 的数据流（NASSv2，`av_litmodule_nass_v2.py:73-185`）

```
train_loader（bs=4/5，drop_last=True，collate_with_near；__getitem__ 忽略 idx 随机采样）
 → batch = (mixtures[B,32000], targets[B,32000], mouth[B,50,88,88], filenames, near_audios[list[list[T]]])
   （avspeech_dataset_near.py:23-43, 152-223；2 s = 32000 samples，25 fps × 2 s = 50 帧，88×88 由 transform 裁）
 → training_step：
   targets[B]→[B,1]；mouth[B,50,88,88]→[B,1,50,88,88]；no_grad 过冻结 video_model → mouth_emb
   est_sources, sep_audio_embs = audio_model(mixtures, mouth_emb)   # 训练态返回二元组
   pit_loss = loss_func["train"](est_sources, targets)
   β = β_init + (β_final-β_init)·min(epoch/E,1)                     # :66-70
   query  = pool_spk_embed(sep_audio_embs[:,0])                     # 有梯度
   pos    = pool_spk_embed(encode_stft(targets[:,0]))   (detach)    # :106-109
   negs   = pool_spk_embed(encode_stft(near_wavs[b]))   (detach)    # :112-132，逐样本 K_b
   cl_loss = NASSContrastiveLoss(query,pos,negs,β)                  # nass_contrastive_loss.py:31-81
   total = pit_loss + contrastive_weight·cl_loss                    # :136-138
   log 8 个 train_* 标量 + 5 个桶计数（固定键集合，DDP 安全）        # :169-183
```

`FalconSpeakerContrastive.forward`（`look2hear/models/Falcon_Contrastive.py:94-120`）在 `self.training`
时返回 `(separated_audios, separated_audio_embeddings)`、eval 时只返回波形；基类 `training_step`
**不解包**（`av_litmodule.py:69,73`），因此基类 + 该模型训练会 `AttributeError: 'tuple' object has no
attribute 'size'`——入口固定用 NASSv2，故不触发（§7 V4 复测仍复现该不兼容）。

---

## 2 关键事实与数字（命令 + 输出）

### 2.1 环境与导入 `[验证]`

```
$ PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -c "import audio_train"
[OK] import audio_train 6.8 s
[OK] look2hear.system.av_litmodule / av_litmodule_mdcl / av_litmodule_nass_v2 / optimizers / schedulers
torch 2.8.0+cu128 pl 2.6.6 cuda True 2
```
- `audio_train.py:4` 在 **import 时**执行 `torch.cuda.empty_cache()`（副作用，不影响正确性）。
- 三个 litmodule 均可在 conda `falcon` 环境干净 import；`torch_optimizer` 依赖（`pytorch_ranger`）在
  conda 环境**已具备**（旧文档 §2.6 的沙箱缺依赖问题不存在于当前环境）`[验证]`。
- `look2hear.system.__all__` = `[make_optimizer, AudioVisualLightningModule,
  AudioVisualLightningModuleNASSv2, DPTNetScheduler]`——**MDCL 未导出**（`system/__init__.py:1-11`）。

### 2.2 配置解析 `[验证]`（6 个 YAML 全部可 `yaml.safe_load`）

| 配置 | audionet_name | gpus | epochs | bs | accum | 全局 batch | train_samples | logger | 数据目录 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `configs/Falcon.yml` | FalconSpeakerContrastive | `[0..9]` | 1000 | 4 | 缺省 1 | 4×卡数 | 缺省 10000 | 缺省 csv | `--`（占位） |
| `configs/LRS2_2Mix.yml` | 同 | `[0,1]` | 100 | 4 | 1 | 8 | 10000 | csv | `/root/data/falcon_lrs2_2mix/{tr,cv,tt}` |
| `configs/LRS2_2Mix_Paper_2gpu.yml` | 同 | `[0,1]` | 1000 | 4 | **5** | **40** | 10000 | csv | `/root/data/falcon_lrs2_2mix_paper/*` |
| `..._Paper_2gpu_smoke.yml` | 同 | `[0,1]` | 3 | 4 | 5 | 40 | 2000 | csv | tr=paper / cv,tt=`/root/tmp/paper_smoke` |
| `..._Paper_8gpu.yml` | 同 | **-1** | 1000 | **5** | 1 | 5×卡数 | 10000 | csv | `/root/data/falcon_lrs2_2mix_paper/*` |
| `Experiments/checkpoint/Falcon/conf.yml` | **SpeakerContrastiveV2（不存在）** | `[0..9]` | 1000 | 4 | 缺省 1 | — | 缺省 10000 | 无 | `${DATA_ROOT}/small/...` |

- `[验证]` `hasattr(look2hear.models,"SpeakerContrastiveV2") = False`；`FalconSpeakerContrastive = True`。
- `[验证]` `conf.yml` 文本含 `${DATA_ROOT}`、`${ROOT}`；`best_k_models.json` 5 条路径全为 `${ROOT}/Falcon/epoch=*.ckpt`，
  `os.path.exists` 全 False。
- `[验证]` `resolve_accelerator_and_devices` 在本机（2 卡）实际解析：
  - `Falcon.yml` `[0..9]` → `devices=[0,1]`（clamp + WARNING）；`-1` → `[0,1]`；
  - `CUDA_VISIBLE_DEVICES=""` → `("cpu",1,1,"auto")`；
  - `>1 device` → `DDPStrategy(find_unused_parameters=True)`。
- **陷阱**：`LRS2_2Mix_Paper_8gpu.yml` 用 `gpus: -1` + `bs 5`，全局 batch = 5×**可见卡数**。
  在 2 卡机上直接跑会静默变成 global batch 10（本机实测解析：`devices=[0,1]`、global batch 10、
  1000 step/epoch），**不是**论文的 40。8 卡脚本自带 `device_count()>=8` 断言（`run_..._8gpu.sh:41`），
  但直接 `python audio_train.py --conf_dir ..._8gpu.yml` 没有保护。

### 2.3 训练协议数字（由配置 + DataModule + PL 语义计算）`[代码]`+`[验证]`

| 项 | 值 | 依据 |
|---|---|---|
| 1 epoch 的样本数 | `train_samples`（默认 **10000**） | `avspeech_dataset_near.py:147-150`（`__len__` 返回 `samples`）、`:250` 默认值 |
| 采样语义 | `__getitem__` **忽略 idx**，`random.randint` 抽 mix + 随机 2 s 窗 | `:152-157` |
| DataLoader | `shuffle=True, drop_last=True, batch_size=config`；PL 2.6.6 默认 `use_distributed_sampler=True`，再包一层 DistributedSampler | `:297-307`；`inspect.signature(pl.Trainer)` 实测默认 True |
| 8 卡（bs5，accum1） | 每 rank 1250 样本 → **250 batch → 250 optimizer step/epoch** | 10000/8/5 |
| 2 卡（bs4，accum5） | 每 rank 5000 样本 → 1250 batch → **250 optimizer step/epoch** | 10000/2/4/5 |
| 论文 10 卡（bs4，accum1） | 每 rank 1000 样本 → 250 batch → **250 step/epoch** | 10000/10/4 |
| 单卡 bs4（LRS2_2Mix.yml） | 2500 batch → **2500 step/epoch**（实测 metrics.csv `last_step=249999` @100 epoch） | `[验证]` |
| 验证协议 | `val_dataloader = [cv_loader, tt_loader]`：**每 epoch 两个 loader 都被完整遍历**；`dataloader_idx=0` 记 `val_loss` 与分桶 SI-SDR；`dataloader_idx=1` 仅在 `current_epoch % 10 == 0` 时计算/记录 `test_loss` | `av_litmodule.py:221-223, 86-140` |
| test_loss 周期 | epoch 0,10,20,…（**0 基**） | `av_litmodule.py:125,181` |
| β 退火 | `0 → 1.0` 线性、`beta_anneal_epochs=200`：ep0=0、ep1=0.005、ep20=0.1、ep50=0.25、ep100=0.5、ep200=1.0（E≤0 直接 1.0） | 5 个 config 的 `spk_contrast` 一致；`av_litmodule_nass_v2.py:66-70`；`[验证]` 数值复算 |
| 优化器 | `AdamW(lr=1e-3, weight_decay=0.01)` | `optimizers.py:51-68,88-106`；config `optimizer` |
| 调度器 | `ReduceLROnPlateau(patience=15, factor=0.5)`，monitor 由 litmodule 注入 `val_loss/dataloader_idx_0` | `audio_train.py:275-280`；`av_litmodule.py:195-206`（注入在 `:201-202`） |
| 梯度裁剪 | `gradient_clip_val=5.0` | `audio_train.py:372` |
| EarlyStopping | monitor `val_loss/dataloader_idx_0`、mode min、**patience 30** | `audio_train.py:345-346`；config `training.early_stop` |
| 其它 Trainer 常量 | `limit_train_batches=1.0`、`sync_batchnorm=True`、`num_sanity_val_steps=0`、`check_val_every_n_epoch=1`、`precision` 未设（fp32） | `audio_train.py:371-377`；PL 默认值 `[验证]` |
| ckpt | `save_top_k=5`、`filename="{epoch}"`、`save_last=True`、monitor 同上 | `audio_train.py:335-343` |
| 续训 | `--ckpt_path` 存在 → `trainer.fit(system, ckpt_path=...)`；不存在 → WARNING + 从头 | `audio_train.py:380-386` |

### 2.4 日志与产物 `[验证]`

- `build_logger`（`audio_train.py:169-232`）实测分支：
  `csv→CSVLogger`、`None/False/"none"→None`（Trainer 收到 `False`）、`tensorboard→RuntimeError`（本环境无
  tensorboard 包）、`comet→RuntimeError`（无 comet-ml）、非法值→`ValueError`。
- CSVLogger 落盘位置：`save_dir=os.path.dirname(exp_dir)`、`name=exp_name` → `<cwd>/Experiments/checkpoint/<exp>/version_N/metrics.csv`。
- `[验证]` 已有 metrics.csv 列（来自 Phase2 单卡训练）：`epoch, learning_rate, lr, step,
  train_{loss,pit_loss,spk_contrast_loss,beta,cos_sim_pos,cos_sim_neg_avg,cos_sim_neg_max,neg_active_ratio,
  n_samples_*near}_{step,epoch}, val_loss/dataloader_idx_0, val_pit_sisnr, val_pit_sisnr_*near,
  test_loss/dataloader_idx_1, test_pit_sisnr`。即 F2 的 4 个 Comet-only 键已全部进入 CSV。
- 收尾产物（`audio_train.py:391-406`）：`best_k_models.json`（`indent=0`，值为 Tensor `.item()`）、
  `best_model.pth`（`audio_model.serialize()`，只含音频模型，与发布权重 `falcon.pt` 格式不同）。

### 2.5 已验证的"跑通"证据（不重启训练，只引用既有产物）

- `[验证]` 2 卡 DDP + accum=5 的 3 epoch smoke（`docs/verification/phase2/LRS2-paper-2gpu-smoke.log`）：
  `Epoch 0, global step 50`（=250 batch/5）→ 说明 accum 生效；`val_loss 0.624 → 0.243 → -0.125`；
  ckpt/best_model/conf 正常落盘；**无 DDP 死锁**（该数据 val `n_near` 全 0，桶集合对称）。
- `[验证]` 单卡 LRS2 100 epoch 的 metrics.csv：100 个 epoch 行，`val_pit_sisnr` 6.73→13.20 dB，
  `test_pit_sisnr` 每 10 epoch 一行（ep0 6.81 → ep40 12.56）；`last_step=249999`。
- `[验证]` `verify_train_config_mutation.py`：建模前 `audio_params` 含 `repeats/shared`，建模后变成含
  `causal`、无 `repeats/shared`（参数量 537,697 不变）——**污染机制仍在**，但 dump 已前移到建模之前。
- `[验证]` 真实 2 卡 smoke 的 conf.yml（`Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/conf.yml`）
  保留 `repeats/shared`、**没有**注入 `causal` → **F9 修复在真实运行中生效**。

---

## 3 P0/P1 修复现状表（对照 `docs/REPRODUCTION_PLAN.md` §6 的 F1–F17）

图例：✅ 已修并验证 / 🟡 已缓解（文件未改但有运行期保护）/ ❌ 未修 / ➖ 非本域。

| # | 问题 | 现状 | 现在代码长什么样（file:line） | 是否仍会炸 |
|---|---|---|---|---|
| F1 | `comet_logger` 未定义 | ✅ | `build_logger(config, exp_dir)` `audio_train.py:169-232`；调用 `:349-350`；`logger if logger is not None else False` `:374`（避开 PL2 `logger=None` 自动建默认 logger 的坑） | 不会。symtable 实测 `audio_train.py` 未定义全局名 = **0** `[验证]` |
| F2 | litmodule 用 Comet 专属 `self.logger.experiment.log(dict)` | ✅ | `av_litmodule.py:154-160` 注释 + `:161-164` metrics 字典 + `:186` `self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)`；NASSv2/MDCL 继承生效 | 不会。全仓库 `look2hear/system/*.py` 无**可执行** `logger.experiment`（grep 仅注释）`[验证]`；CSVLogger / 无 logger 两种 fit 均通过 |
| F3 | `Experiments/checkpoint/Falcon/conf.yml:88` `audionet_name: SpeakerContrastiveV2` 不存在 | 🟡（文件未修，评测绕开） | 文件仍是 `SpeakerContrastiveV2`；`audio_visual_test.py:84` 默认 `configs/Falcon.yml`，`:290-294` 对旧快照给出明确 `AttributeError` 提示 | 训练不再读该文件（`--conf_dir` 默认 `configs/Falcon.yml`）；**评测**若显式 `--conf_dir Experiments/.../conf.yml` 仍会炸（预期行为，有清晰报错） |
| F4 | `${DATA_ROOT}` / `${ROOT}` 占位 | 🟡（文件未修，评测绕开） | `conf.yml:99-102`（数据）、`:120`（main_args.exp_dir）、`:127`（options.conf_dir）；`best_k_models.json` 5 条 `${ROOT}/Falcon/epoch=*.ckpt` | 训练不读；评测入口 `audio_visual_test.py:110-117` 用 `os.path.expandvars` 展开，缺变量时 `:199` 报错。`best_k_models.json` 仍是死引用（仓库无 ckpt） |
| F5 | eval dataset mono → `mean(-1)` 变 0 维 → `torch.stft` 崩 | ✅ | `avspeech_dataset.py:198-201`（eval）、`:114-120`（legacy train）；训练集 `avspeech_dataset_near.py:176-177,206`；`check_avspeech_dataset.py:121-123,202-204` | 不会（单声道保留 1-D，多声道才取均值） |
| F6 | PL 2.x `accelerator/devices/gpus` | ✅ | `_normalize_gpus_spec` `audio_train.py:50-95`（list=索引 / int=卡数 / 0=CPU / -1=全部 / bool/负数非法）；`resolve_accelerator_and_devices` `:98-166`；Trainer `:363-377` | 不会。CPU 路径实测 `("cpu",1,1,"auto")`；2 卡 clamp 实测；非法值提前 `ValueError` `[验证]` |
| F9 | conf.yml 被 `Separator.pop` 污染（丢 `repeats/shared`） | ✅ | `audio_train.py:236-248`：`exp_dir` 建好后**先** `copy.deepcopy(config)` 写 `conf.yml`，**再**建模 | 不会。真实 smoke 的 conf.yml 实测保留 `repeats/shared`、无 `causal` `[验证]`；`verify_train_config_mutation.py` 证明建模仍会原地改 config |
| F15 | DDP per-bucket `all_gather` 死锁 | ❌ | `av_litmodule.py:170-176`：`for bucket,name in _bucket_names.items(): outputs=...; if outputs: avg=...; avg=self.all_gather(avg)` —— **仍只在"本 rank 该桶非空"时调用集合通信** | **可能**。机制未变（旧文档 2 进程 CPU DDP 实测挂死 `[文档]`）；真实 2 卡 LRS2 smoke 未触发（val 全 K=0 或全 K=1，各 rank 桶集合相同）`[验证]`。RealSSA-Sim 的 cv meta 每样本 `near` 1–4+ 个 → 桶分布不均匀，**小 val 集/分片不均时会踩** |
| F17 | 训练日志/指标名不统一 | ✅ | 统一走 `self.log` / `self.log_dict`（`av_litmodule.py:186`；NASSv2 `:169-183`；MDCL `:157-171`） | 不会 |
| 新增 | `training.accumulate_grad_batches` 支持 | ✅ | `audio_train.py:361-362` 读取 + `:365` 传入 Trainer；`Paper_2gpu*.yml:128` = 5、`Paper_8gpu.yml:126` = 1 | 不会；2 卡 smoke 实测 global step 50/epoch（250 batch ÷5）`[验证]` |
| F7/F8/F10–F14/F16 | 指标/评测/骨干/模型侧 | ➖ | 见 Phase1 报告 §2；F13 保持现状（bit-identical），F14 只加了 `_mdcl_loss` 别名，`Falcon_Contrastive.py:95` 仍吞 `near_audio` | 非本域，见 `04`/`06` 文档 |

### 3.1 仍未修/仍不可信的清单（本域）

1. **F15 未修**（`av_litmodule.py:170-176`）：per-bucket `all_gather` 数量依赖各 rank 桶占用；建议改成
   「固定 5 次 gather，空桶传零张量」或把分桶指标移出 DDP 验证（单卡汇总）。这是 8 卡 RealSSA-Sim 训练前
   唯一必须评估的 P0 级正确性风险。
2. **F3/F4 文件未修**：`Experiments/checkpoint/Falcon/conf.yml` 与 `best_k_models.json` 仍是发布占位；
   训练已不依赖，评测靠默认配置绕开。若要"一条命令复跑 README 评测"，仍需改写这两个文件（或删除）。
3. **死键仍误导**：`training.system` / `training.parallel` / `datamodule.data_name` 不参与行为（`audio_train.py` 无读取）。
4. **`lightning_utils.MyRichProgressBar`** 仍引用未定义 `MetricsTextColumn`/`CustomProgress`（`:93-94`），
   symtable 实测 2 个未定义名；当前无人使用，属死代码。
5. **`configure_optimizers` dict 分支** `sched["interval"]` 未 `setdefault`（`av_litmodule.py:208`）→ 传不带
   `interval` 的 dict scheduler 会 `KeyError`（当前配置走 `ReduceLROnPlateau` 实例分支，不触发）。
6. **`gpus: -1` 的全局 batch 语义**：见 §2.2 陷阱；8 卡脚本有断言，直接跑 config 无保护。

---

## 4 与旧文档（`docs/understanding/03-training-system.md`，700 行）及 Phase1/2 的差异

> 旧文档行号以 `main@7576854`（发布态）为准；HEAD `2beec38` 中 `audio_train.py` 从 203 → 417 行、
> `av_litmodule.py` 240 → 247 行，**旧文档所有行号引用均已失效**，只能按语义对照。

### 4.1 逐条【确认 / 修正 / 推翻】

| 旧文档结论 | 判定 | 现状与证据 |
|---|---|---|
| §1「训练入口无法直接运行（`comet_logger` 未定义，P0）」 | **推翻** | F1 已修；入口可 import、可启动；Phase1 单卡 smoke + Phase2 2 卡 smoke 均跑通 `[验证]`/`[文档]` |
| §1.1#2「三个 litmodule 共用 Comet 专属 `logger.experiment.log`，本仓库依赖范围内必失败」 | **推翻** | 已改 `self.log_dict`（`av_litmodule.py:186`）；CSVLogger / 无 logger 实测通过 `[验证]` |
| §1.1#3「`training.system`/`parallel`/`datamodule.data_name` 完全未被读取」 | **确认** | 仍为死键（grep `audio_train.py` 无读取） |
| §1.1#4「PL 1.x 写法，PL 2.6.6 下 CPU 必崩 / 10 卡列表在少卡机器崩」 | **推翻（已修）** | F6：`resolve_accelerator_and_devices`（`audio_train.py:98-166`）；实测 CPU 路径与 2 卡 clamp `[验证]` |
| §1.1#5「`conf.yml` 与代码不自洽（类名不存在；同时含 causal 与 repeats/shared 不可能由当前代码产生）」 | **部分修正** | 类名/占位仍在（F3/F4 未修）；但"不可能由当前代码产生"已过时——当前代码 dump 前移+deepcopy，**真实 dump 保留 repeats/shared 且不注入 causal**（F9，`[验证]`） |
| §1.1#6「DDP per-bucket `all_gather` 死锁」 | **确认（未修）** | 代码仍在 `av_litmodule.py:170-176`；F15 未做；2 卡 LRS2 smoke 未触发（桶对称） |
| §2.1「`audio_train.py` 203 行；默认 `--conf_dir` 指向不存在的 `configs/NASSv2_lrs3_complex_nonmoving.yml`」 | **修正** | 现 417 行；默认 `configs/Falcon.yml`（`:32-39`） |
| §2.1「`datamodule` 只用于 setup/make_loader，从不传给 Trainer」 | **确认** | 同旧（`audio_train.py:253-257`；`av_litmodule.py:217-223`） |
| §2.2「`configs/Falcon.yml` 数据三处占位 `--`、`pretrain` 指向 `pretrain_zoo/`」 | **确认** | `Falcon.yml:159-161,8`（仍为发布模板；实际训练用 `LRS2_2Mix*.yml` 的绝对路径） |
| §2.3「`Experiments/.../conf.yml` 是 dump 产物格式、含 `options`/`main_args`、内容不自洽」 | **确认** | 结构一致（`options`/`main_args` 在 `:119-128`）；类名/占位未修 |
| §2.4「`best_k_models.json` 5 条、`indent=2`、`${ROOT}` 字面量，非当前代码产物」 | **确认** | 现代码 `indent=0`+绝对路径（`audio_train.py:391-396`）；该文件仍是发布占位 |
| §2.5「`system/__init__.py` 不导出 MDCL」 | **确认** | `system/__init__.py:1-11` 仍未导出 |
| §2.6「沙箱 venv 缺 `pytorch_ranger` 导致 `import torch_optimizer` 失败」 | **推翻（环境已具备）** | conda `falcon` 实测 `[OK] import torch_optimizer (without stub)` |
| §2.8「基类 `training_step` 不解包 tuple，与 `FalconSpeakerContrastive` 不兼容」 | **确认** | `av_litmodule.py:69,73`；`[验证]` `PITLossWrapper(tuple)` 仍 `AttributeError` |
| §2.9/§2.10「NASSv2 185 行 / MDCL 173 行、MDCL 未接入」 | **确认** | 行数一致；全仓库仍无 MDCL import |
| §2.11「训练链 utils 只用 `parser_utils` 与 `print_only`；`MyRichProgressBar` 引用未定义名」 | **确认** | `print_only` 现有 **19** 个调用点（`audio_train.py`）；`lightning_utils.py:93-94` 未定义名仍在 `[验证]` symtable |
| §3.1「YAML→argparse 二次解析；Python≥3.10 下 `main_args={}`，`exp_dir` 是新建键」 | **确认** | `parser_utils.py:117-151`；真实 dump 的 `main_args={'exp_dir': ...}` `[验证]` |
| §3.2 S20「建 `tensorboard_logs` 目录但从不实例化 logger」 | **推翻** | 已改为 `build_logger`（F1）；目录逻辑删除 |
| §3.2 S22「`logger=comet_logger` → NameError」 | **推翻** | 见 F1 |
| §3.2 S24「resume 路径不存在时静默从头」 | **修正** | 现在有 WARNING（`audio_train.py:384-385`）；仍不展开 `~`/环境变量（`os.path.exists` 原样） |
| §3.4.2「Comet 专属键（learning_rate/val_pit_sisnr/test_pit_sisnr）只进 Comet」 | **推翻** | 全部进 CSV（`[验证]` metrics.csv 列） |
| §3.4.2「test_loss 每 10 epoch 才算，但 tt loader 每 epoch 仍被完整遍历」 | **确认** | `av_litmodule.py:125,181,221-223`；这是每 epoch 的固定开销（8 卡 Paper 数据 cv 10k + tt 6k） |
| §4.5「conf 污染 → 评测 49,081 参数（9.1%）静默错误」 | **修正（当前代码不再产生污染 dump）** | F9 已修；但**历史 dump / 旧快照**仍可能触发；评测侧已加 deepcopy + strict 加载 + 明确报错（Phase1） |
| §4.7「README 协议：DDP / AdamW 1e-3 / ReduceLROnPlateau / early stop 30 / grad clip 5」 | **确认** | `audio_train.py:365,372,275-280,345-346`；`Falcon.yml:143-153,136-140` |
| §4.7「batch/epoch 口径：bs4×卡数=全局 40、epochs 1000、无梯度累积、1 epoch=10000 随机片段」 | **修正** | 无梯度累积已过时（新增 `accumulate_grad_batches`）；8 卡配方 bs5×8=40、2 卡 bs4×2×accum5=40；epoch 语义不变（随机 10000/epoch） |
| §4.8「PL 2.6.6 `use_distributed_sampler` 默认 True，且对 module 提供的 loader 生效」 | **确认** | `inspect.signature(pl.Trainer)` 默认 True `[验证]`；2 卡 smoke 的 per-rank 分片行为一致 |
| §5.1 YAML 键消费表 | **修正** | 新增 `training.logger`（`:177`）、`training.accumulate_grad_batches`（`:361`）；`training.gpus` 语义重写（索引/卡数/0/-1）；`training.system/parallel/data_name` 仍死键 |
| §6 A17「无 `accumulate_grad_batches` → 默认 1」 | **推翻** | 已支持配置，默认仍 1（`:361`） |
| §6 A18/A19「`accelerator=None` 崩 / `devices=[0..9]` 在 2 卡崩」 | **推翻** | F6 已修；`verify_train_pl_contract.py` 仍显示**裸 PL API** 会崩，但入口不再这样用 `[验证]` |
| §6 A45「沙箱缺 `pytorch_ranger`」 | **推翻** | 环境已具备 |
| §6 A53「失败顺序：数据/权重缺失比 `comet_logger` 更早」 | **修正** | `comet_logger` 已不存在；当前最早失败点是 `data_config` 占位 `--`（`Falcon.yml`）或视觉骨干文件缺失 |
| §7 Q1「`comet_logger` 本应指向什么」 | **结案** | `build_logger`：csv（默认）/ tensorboard / comet / none |
| §7 Q4「per-bucket all_gather 是 P2 还是 P0」 | **未结案** | 仍未修；结论：**取决于数据/分片**，LRS2 对称数据不触发，RealSSA-Sim 桶不均时有风险 |
| §7 Q5「epoch 语义 / 早停点反推」 | **部分确认** | epoch 仍是"每 rank 250 次随机抽样"；单卡 LRS2 100 epoch 实测 `last_step=249999` 支持"非遍历数据集"结论 |
| §8 V1–V16 验证记录 | **部分过时** | `verify_train_litmodules.py` 中"NASSv2 fit(CSVLogger) 预期失败"现在**通过**（脚本自身标注 unexpected）；`verify_train_pl_contract.py` 结论仍有效 `[验证]` |

### 4.2 Phase1/Phase2 报告口径核对

- Phase1 报告 §2 的 F1–F14 修复清单与本轮代码一致（F1/F2/F5/F6/F9 ✅；F3/F4"工作绕过"✅；F13 保持；
  F14 半修）`[文档]`+`[代码]`。
- Phase1 `REVIEW-training-stack.md` 的 P3-2（`configs/Falcon.yml` 未登记 `training.logger` 注释）**已闭环**：
  `Falcon.yml:130-131` 现有 `# logger: csv ...`、`# gpus: [0, 1] ...` 注释 `[代码]`。
- Phase2 §5.3 的"8 卡方案"与本轮核对一致：`configs/LRS2_2Mix_Paper_8gpu.yml` 8×bs5、1000 epoch、
  early stop 30、β 0→1/200；`accumulate_grad_batches` 新增支持；本地 2 卡 smoke 通过 `[文档]`+`[代码]`。
- Phase2 `P2D-report.md` 的单卡 100 epoch 训练数据与本轮 metrics.csv 对得上（`val_pit_sisnr` ep0 6.73、
  最佳 ~13.2–13.4；test 每 10 epoch）`[文档]`+`[验证]`。

---

## 5 对复现的影响与清单

### 5.1 论文配方 vs 代码默认配置的差异

| 维度 | 论文/README 声明 | 代码默认（`configs/Falcon.yml`） | 8 卡配方（`Paper_8gpu.yml`） | 差异影响 |
|---|---|---|---|---|
| 卡数 / 全局 batch | 10 卡 × bs4 = 40（由 config 推断，README 未明说） | `gpus:[0..9]`、bs4 → 40（10 卡机） | 8 卡 × bs5 = 40 | 等价全局 batch；**per-rank 梯度累积语义不同**（accum1） |
| optimizer | AdamW 1e-3 / wd 0.01 | 同 | 同 | 无 |
| scheduler | ReduceLROnPlateau(15, 0.5) | 同 | 同 | 无（monitor = val_loss） |
| epochs / 早停 | 1000 / patience 30 | 同 | 同 | 无 |
| β 退火 | 0→1 over 200 epoch | 同 | 同 | 无 |
| train_samples | 未说明 | 缺省 10000 | 10000 | 1 epoch 非"遍历数据集" |
| 数据 | 未公开（按 benchmark 分别训练） | 占位 `--` | LRS2 双向 target 40k / near=另一说话人 | **最大不确定性**：数据混合/双向 target/near 方案均为推断 |
| near 负样本 | NASS 需要近场负样本 | `use_near: true`（数据无 near 时 K=0） | near=另一说话人（K=1） | 论文若用 near-field 模拟，指标可能更高；LRS2 结果不可直接对标 |
| 通道策略（RealSSA） | 未声明 4ch 如何下混 | — | — | RealSSA 训练需先做实验（mean vs ch0） |
| 日志 | README 未说明 | 缺省 csv | csv | 无 |
| 视觉骨干 | 冻结 FRCNN（来自发布权重） | `pretrain_zoo/...`（缺失） | `/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar` | 已由 Phase1 T5 导出 |

**结论**：代码默认配置能对齐论文"训练协议"的可公开部分（优化器/调度/裁剪/早停/β/epochs），
**对齐不了**论文未公开的部分（数据混合、near 负样本、通道策略、双向 target、实际训练轮数/种子）。

### 5.2 "从零训练 RealSSA-Sim"还差什么（重点问题①）

数据现状（本轮实测）：

| 资产 | 位置 | 现状 |
|---|---|---|
| RealSSA-Sim `tt`（test 9,169 条） | `/root/data/falcon_realssa_sim/tt`（30 GB，已转 Falcon folder） | ✅ 可训练格式（mix/target/mouths + JSON），**但 tt 是 test split，不能用于训练** |
| RealSSA-Sim 原始 tar（tt 19 shards，38 GB） | `/root/data/RealSSA-Sim/tt` | ✅ |
| RealSSA-Sim `tr`（202 shards，412 GB）/ `cv`（19 shards，38 GB） | TOS 归档 `/aispeech-cfs/voiceprint/data/multimodal/realssa_sim/{tr,cv}`（`_archive_done.json` `ok:true`） | 🟡 **未恢复到本地、未转换**；本地源目录已删 |
| RealSSA-Real | TOS `/aispeech-cfs/.../realssa_real`（12 GB） | ✅（评测用，与训练无关） |
| 视觉骨干 | `/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar` | ✅（Phase1 T5） |
| LibriSpeech（可选负样本源） | TOS `/aispeech-cfs/voiceprint/data/librispeech/*.tar.gz`（含 train-clean-360） | 🟡 存在但未解包 |

**缺口清单（按阻塞顺序）**：

1. **数据恢复**：把 `realssa_sim/{tr,cv}`（450 GB tar）从 TOS 恢复到本地（或直接流式读取 TOS tar），
   用 `tools/repro/realssa_to_falcon.py --src <shard>.tar --out <dir> --splits tr,cv --channel mean`
   转成 Falcon folder 格式（100,826 / 9,133 条）。磁盘预算：tar 450 GB + 转换产物（16k mono wav +
   mouths 引用/复制）≥ 600 GB（`docs/DATA_ACCESS_GUIDE.md` §4 已给该口径）。转换脚本支持逐 shard 流式处理，
   不必整包解压 `[代码]`（`realssa_to_falcon.py:41-55,101-160`）。
2. **训练配置缺失**：仓库没有任何 RealSSA-Sim 训练配置（`configs/` 只有 `Falcon.yml` + 4 个 LRS2 配置）。
   需要新建：`exp_name`、`train/valid/test_dir`、`batch_size`、`train_samples`、`gpus`、`logger`、
   `accumulate_grad_batches`。超参只能沿用 `configs/Falcon.yml`（lr 1e-3、AdamW、ReduceLROnPlateau、
   clip 5、epochs 1000、early stop 30、β 0→1/200、train_samples 10000）——**论文 RealSSA 的实际训练轮数/
   数据混合未公开**，属推断。
3. **near 负样本方案**（决定 NASS 损失是否成立）：
   - 现状：转换器生成的 `near.json` 是 **fallback 自引用**（`[mix_path, n]`），训练集 `__getitem__`
     会跳过自引用（`avspeech_dataset_near.py:186-191`）→ **K=0**，NASS 只剩正项 `α_pos(1-cos(q,p))`，
     `β` 退火对负项无效。实测 `falcon_realssa_sim/tt/near.json` 9,169 条全部 K=1 但都是自引用 `[验证]`。
   - 选项 A：接受 K=0 训练（与论文配方不等价，须在结果里声明）。
   - 选项 B：用 meta `components[type=="near"].path` 指向的 **LibriSpeech 语句**近似重建负样本
     （TOS 有 librispeech tar.gz；路径可映射 speaker/chapter/utt）。但 meta **只有 SIR、没有时间偏移**
     （`position: null`，无 start/duration）→ 只能随机裁剪近似，不是原 near 波形。dataset card 明确
     "fallback 不是等价复现"。
   - 选项 C：仿 LRS2 方案，用同 batch/同场景的其它说话人当负样本（非官方）。
4. **硬件**：需要 8 卡（本机 2×A800，且 GPU 常被外部进程占用）。单卡外推 ~72 min/epoch（Phase2 P2D 实测口径），
   8 卡配方 250 step/epoch；按 2 卡 smoke 的 3.82 it/s 外推 ≈1–2 min/epoch（含验证）→ 1000 epoch 上限 33–50 h，
   early stop 通常 5–15 h `[文档]`+`[推断]`。
5. **F15**：RealSSA-Sim cv meta 的 `near` 数量 1–4+ 分布不均（抽 200 条统计：1near 44、2near 59、3near 59、
   4+ 38，**没有 0near**）`[验证]`。全量 cv 9,133 条、8 卡分片时每 rank ~1,142 条，各桶大概率非空
   → 死锁概率低 `[推断]`；但小规模 smoke/自定义 val 子集时容易踩，建议先修 F15 或先单卡跑通。
6. **合规**：RealSSA-Sim 为 `cc-by-nc-nd-4.0` + gated（仅非商用学术，禁止转发/再分发含人脸/声音的派生文件）
   → 转换产物**不能进仓库**，只能留本地/TOS（与 `docs/DATA_ACCESS_GUIDE.md` §4 一致）。

### 5.3 若要复跑 8 卡火山引擎实验：启动命令与前置检查（重点问题④）

**启动（仓库内脚本，推荐）**：

```bash
cd /path/to/Falcon-fork
CONF=configs/LRS2_2Mix_Paper_8gpu.yml \
DATA_ROOT=/data/falcon_lrs2_2mix_paper \
LOG_DIR=/data/logs \
bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
```

**前置检查清单**（脚本已覆盖 ①②③，④–⑨ 需人工确认）：

1. **8×A800 可见**：脚本 `:38-42` 断言 `torch.cuda.device_count() >= 8`（本机 2 卡 → 会在此退出，**不会误启动**；
   我未执行该脚本以免误触发训练，仅静态核对 + `bash -n` 语法检查通过 `[验证]`）。
2. **数据 JSON**：脚本 `:43-45` 检查 `$DATA_ROOT/{tr,cv,tt}/mix.json` 存在并统计条数。
   **坑**：脚本用 `python3`（系统 Py 3.8）只做 `json.load` 计数——虽能跑但违反 AGENTS.md §1 环境强制；
   且 `DATA_ROOT` 只用于**自检**，训练实际读的是 config 里写死的 `/root/data/falcon_lrs2_2mix_paper`
   （`Paper_8gpu.yml:139-141`）→ 若覆盖 `DATA_ROOT` 为别的路径，会出现"自检通过、训练读旧路径"的错位。
3. **视觉骨干**：脚本 `:46-51` 检查 `$BACKBONE`（默认 `/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar`）；
   缺失时给出导出命令（`tools/repro/extract_video_backbone.py`）。
4. **Python 环境**：`PYTHON_BIN` 默认 `/root/miniforge3/envs/falcon/bin/python`（`:20`）；节点上若无该路径需显式覆盖。
5. **依赖**：torch cu128 / PL 2.6.6 / typeguard<3 / timm / soundfile 等（`docs/verification/conda-env-lock.txt`）。
6. **全局 batch 校验**：确认 `device_count=8` 且 `Paper_8gpu.yml` 的 `bs=5`、`accum=1` → global 40
   （若节点只有 4 卡，`gpus:-1` 会让 global batch 变 20 而**不报错**——必须先看启动日志里
   `[audio_train] accelerator=... num_devices=...` 一行）。
7. **日志/监控**：训练日志 `docs/verification/phase2/LRS2-paper-<UTC>.log`、PID `LRS2-paper.pid`；
   `python tools/repro/phase2/monitor_lrs2_paper.py --exp LRS2_2Mix_Paper_8gpu --watch 300`（实测可用，
   读取 `version_0/metrics.csv` 输出 `epochs/last/best val_pit_sisnr/last_step` `[验证]`）。
8. **续训**：`python audio_train.py --conf_dir configs/LRS2_2Mix_Paper_8gpu.yml --ckpt_path <exp>/last.ckpt`
   （PL 恢复 epoch/step/optimizer/scheduler/callbacks）。**注意**：CSVLogger 在新进程会分配 `version_1`
   （实测：同 save_dir/name 的第二个 logger → `version: 1`）→ 监控脚本只找 `version_0`，续训后监控会读到旧文件；
   建议续训后用 `--watch` 前先确认最新 `version_*` 目录，或把监控改成 `glob(version_*)` 取最新。
9. **F15 预检**：若 val 集较小或分桶不均，先在 smoke 上跑 1 epoch 确认无挂死（2 卡 LRS2 数据已通过）。

**备选启动方式**（平台要求单进程入口时）：
`PYTHONPATH=$PWD python audio_train.py --conf_dir configs/LRS2_2Mix_Paper_8gpu.yml`
（PL2 在 `gpus:-1` + 8 可见卡时自动 DDP；同样受 §2.2 的"全局 batch 随可见卡数变化"影响）。

### 5.4 8 卡 vs 论文 10 卡的等价性

- **相同**：全局 batch 40、optimizer/scheduler/裁剪/早停/β/epochs、每 epoch 250 optimizer step。
- **不同**：per-rank batch 5 vs 4（单卡显存与 BN 统计口径不同）；DDP world size 8 vs 10（NCCL 通信量、
  `sync_batchnorm` 统计范围不同）；论文是否真的 10 卡只是从 `Falcon.yml:133` 的 `gpus:[0..9]` 推断 `[文档]`。
- 结论：8 卡是**可接受近似**，但不是严格等价；报告里应写明"8×bs5 对齐全局 batch 40"。

---

## 6 未解问题

1. **F15 是否在 8 卡 RealSSA-Sim 上真实触发？** 需要真实 cv 分片 + 多 rank 实测（旧文档用 2 进程 CPU DDP
   构造非对称桶已复现挂死 `[文档]`）。建议修法：固定 5 次 gather / 空桶零张量。
2. **RealSSA-Sim 训练超参**：论文/README 未公开训练数据混合、epochs、通道策略、near 负样本构造；
   只能按 `configs/Falcon.yml` 推断。是否值得先向作者提 issue（已有 `JusperLee/Falcon#1` 只问 LRS2）？
3. **near 负样本可否用 LibriSpeech 近似重建？** meta `components[].path` 有 LibriSpeech utt 路径，但无时间
   偏移/时长；TOS 有 librispeech tar.gz。需要实验确认"随机裁剪近似"对 NASS 是否有正向作用。
4. **`Experiments/checkpoint/Falcon/conf.yml` / `best_k_models.json`**：应改写为真实路径+正确类名，还是删除
   以免误导？（当前训练不读、评测靠默认配置绕开，但 README 评测命令仍会踩。）
5. **`gpus: -1` 的隐式全局 batch**：是否应加"配置声明 world_size 断言"（如 `training.expected_world_size`），
   让 8 卡配方在非 8 卡机上**直接报错**而不是静默降级？
6. **`training.system`/`parallel`/`datamodule.data_name` 死键**：保留（历史兼容）还是删除/校验？
7. **测试套件回归**：`pytest tools/repro/tests` 本轮 **104 passed / 7 skipped / 1 failed**
   （`test_phase2_pipeline.py::test_phase2_eval_csv_aggregation_consistent`：对空字符串 `float('')` 报
   `ValueError`，属 Phase2 评测 CSV 聚合测试，与训练域无关）。该失败是测试自身对空列不健壮，还是
   数据/产物回归？需 Phase2 负责人确认。
8. **`audio_train.py:399` 的 `torch.load`**：未显式 `weights_only`，torch 2.8 默认 `True`（实测签名
   `weights_only=None` → 内部按 True 处理）；真实 ckpt 含 `training_config`（纯 YAML 类型）与
   `hyper_parameters`（含 `gpus` → Tensor、`None` → "None"）预期可加载 `[推断]`，未用真实 ckpt 验证。
9. **`persistent_workers=true` + `num_workers=8`**：8 卡时每 rank 8 worker → 64 进程；RealSSA-Sim
   长音频 IO 下是否需要调小/关掉，需实测。

---

## 7 验证记录

环境：`/root/miniforge3/envs/falcon/bin/python`（3.11.17，torch 2.8.0+cu128，PL 2.6.6，2×A800）；
所有命令 `cd /root/code/github_repos/Falcon-fork && PYTHONPATH=$PWD`。**未启动任何真实训练/下载。**

| # | 命令 | 结果摘要 |
|---|---|---|
| V1 | `python -c "import audio_train"` + 逐个 import 3 个 litmodule / optimizers / schedulers / utils | 全 OK（audio_train 6.8 s；torch 2.8.0+cu128 / PL 2.6.6 / cuda True / 2 卡） |
| V2 | `verify_train_imports.py` | 全 `[OK]`（含 `torch_optimizer` 无 stub）；`look2hear.models` attrs = `Falcon/FalconSpeakerContrastive` |
| V3 | `verify_train_pl_contract.py` | ModelCheckpoint/EarlyStopping kwargs OK；CPU+CSVLogger 双 val loader fit OK；键名 `val_loss/dataloader_idx_0`、`test_loss/dataloader_idx_1`、`lr`；`best_k_models` 值 Tensor 且 `.item()` OK；`torch.load` 默认参数 OK；裸 PL `accelerator=None` → `ValueError`、`devices=[0..9]`@2 卡 → `MisconfigurationException`（入口已不用这两种写法） |
| V4 | `verify_train_litmodules.py` | NASSv2 11-epoch fit OK（`train_loss 1.2275 = pit 0.9775 + cl 0.25`）；β `{0:0.0,1:0.005,50:0.25,100:0.5,150:0.75,200:1.0,250:1.0}`、E≤0→1.5（脚本实例默认 β_final=1.5；config 生效值 1.0，见 §2.3）；`configure_optimizers` = `([opt],[{monitor: val_loss/dataloader_idx_0, ReduceLROnPlateau}])`；Comet 键已进 CSV（`learning_rate/val_pit_sisnr(_*near)/test_pit_sisnr`）；**CSVLogger fit 现在 OK**（脚本原预期失败）；MDCL 3-epoch fit OK；`PITLossWrapper(tuple)` 仍 `AttributeError` |
| V5 | `verify_train_config_mutation.py` | 建模后 `audio_params/video_params` 丢 `repeats/shared`、多 `causal`；参数 537,697 全可训练（污染机制仍在） |
| V6 | `verify_train_undefined_globals.py audio_train.py look2hear/system/*.py look2hear/utils/lightning_utils.py` | `audio_train.py`/3 litmodule = **0** 未定义全局名；`lightning_utils.py` = 2（`MetricsTextColumn`、`CustomProgress`，死代码） |
| V7 | 6 个 YAML 全量解析 + 关键字段打印 | 见 §2.2；`SpeakerContrastiveV2` 不存在；`conf.yml`/`best_k_models.json` 占位与死路径确认 |
| V8 | `resolve_accelerator_and_devices` × 5 配置 + `CUDA_VISIBLE_DEVICES=""` | 2 卡：`[0..9]→[0,1]` clamp、`-1→[0,1]`、CPU→`("cpu",1,1,"auto")`；>1 卡→`DDPStrategy` |
| V9 | 全局 batch/step 复算 + `inspect.signature(pl.Trainer)` + `torch.load` 签名 | 8gpu 配置在本机解析为 2 卡（global 10）→ 陷阱确认；PL 默认 `accumulate_grad_batches=1`、`use_distributed_sampler=True`、`check_val_every_n_epoch=1`、`sync_batchnorm=False`（入口显式 True） |
| V10 | `build_logger` 全分支探测 | csv→CSVLogger；None/False/none→None；tensorboard/comet→RuntimeError（缺包）；非法→ValueError |
| V11 | CSVLogger 版本探测（同 save_dir/name 建两个） | 第一个 `version=0`，第二个 `version=1` → 续训会新建 `version_1`（监控脚本硬编码 `version_0` 的坑） |
| V12 | `pytest tools/repro/tests/test_audio_train_entry.py test_litmodule_logging.py -q` | **25 passed**（9.07 s） |
| V13 | `pytest tools/repro/tests -q` | **104 passed / 7 skipped / 1 failed**（失败项 `test_phase2_eval_csv_aggregation_consistent`，空字符串转 float；非本域） |
| V14 | `monitor_lrs2_paper.py --exp LRS2_2Mix_1gpu` | 读到 `/root/tmp/p2d_train/.../version_0/metrics.csv`：`epochs=100 last=ep99 val_pit_sisnr=13.2014 best=ep99 13.2014 last_step=249999` |
| V15 | 真实 2 卡 smoke 产物核对（`LRS2-paper-2gpu-smoke.log`、`Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/conf.yml`） | 3 epoch 完成、global step 50/epoch（accum5 生效）、无 DDP 死锁；conf 保留 `repeats/shared` 无 `causal`（F9 生效） |
| V16 | `bash -n tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh` + 静态核对 | 语法 OK；`:41` 8 卡断言、`:44` 用系统 `python3`、`DATA_ROOT` 仅自检、`NPROC` 未使用；**未执行脚本**（避免误启动训练） |
| V17 | RealSSA-Sim 数据盘点：`/root/data/falcon_realssa_sim/tt`（9,169）、`/root/data/RealSSA-Sim/tt`（19 shards）、TOS `realssa_sim/{tr:202, cv:19, tt:19}`（412G/38G/38G）、`_archive_done.json` `ok:true` | 见 §5.2；`near.json` 全为自引用 fallback（K=1 但训练时跳过 → K=0）；meta `components` 有 `near` 1–4+（无 0near）→ F15 风险与负样本方案依据 |
| V18 | 单卡 LRS2 100 epoch `metrics.csv` 统计 | 100 个 epoch 行；`val_pit_sisnr` ep0 6.7325 → ep99 13.2014；`test_pit_sisnr` ep0 6.8147 / ep40 12.5586；`last_step=249999` |

**未完成/受限项**：
1. 未跑 F15 的 DDP 挂死复现（旧脚本需 >6 min 且必须 `timeout`+`kill -9`，本轮预算单条 <5 min，且会占用 GPU/进程）；
   结论沿用旧文档实测 + 当前代码静态核对。
2. 未在 8 卡上验证 `Paper_8gpu.yml`（本机 2 卡）；未执行 8 卡启动脚本（避免误启动训练）。
3. 未用真实 ckpt 验证 `torch.load`（`weights_only` 默认 True）全链路；`[推断]` 可加载。
4. 未转换 RealSSA-Sim tr/cv（450 GB，超出本轮"便宜验证"预算，且会占用大量磁盘/IO）。
5. RealSSA-Sim 训练配置尚不存在，未编写（本任务只读，仅给缺口清单）。
