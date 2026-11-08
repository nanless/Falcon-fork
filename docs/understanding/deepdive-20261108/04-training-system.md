# 04 训练系统与配置（audio_train.py / litmodules / configs / 论文 Table 16 对齐）

> 子代理 `w1-train-system`（共享任务 `task-4`）· 2026-11-08 · 仓库 `/root/code/github_repos/Falcon-fork`
> HEAD `6c1d5b7`（2026-10-30）· env `falcon` = `/root/miniforge3/envs/falcon/bin/python`
> （Py 3.11.17 / torch 2.8.0+cu128 / pytorch-lightning 2.6.6 / 2×A800-SXM4-80GB，本轮**未占卡**）
> 证据标记：`[代码]` 读源码；`[实测]` 本轮实际运行（命令+输出见 §4/§8）；`[推断]` 由代码+数据推出未直接运行。
> 只读分析，未改仓库任何文件；临时脚本只写 `/root/tmp/w104/`。
> 论文 arXiv:2610.32749 已于 2026-10-23 公开，Table 16 对齐以公开版为准（§3.4）。
> 本文件不重复 MACs 与 lookahead 数字：音频 10.611 GMAC 是 ptflops 含 LSTM、不含 FRCNN 的口径，
> 整窗 26.51 GMAC；lookahead 三档见 00 §4 与 12 §3。

---

## 0. 结论速览

1. **入口是活的**：当前 HEAD 的 `audio_train.py` 能跑通。我在 CPU 上用**真实 config + 同一套装配调用**
   重建了完整对象图并跑完 2 epoch（§4.2）；AGENTS.md:91 点名的「Comet logger / conf.yml 类名」两个 P0
   **已在 `ac91800` 修掉**，AGENTS.md 那句话过期 11 天（§6）。
2. **三个 litmodule 只有 NASSv2 会被实例化**（`audio_train.py:26,356` 硬编码），`training.system` 键是死的；
   `av_litmodule_mdcl.py`（173 行）与 `look2hear/system/schedulers.py`（111 行）在产品路径上**全死**。
3. ⚠️ **β（论文里的 τ）退火在 LRS2-2Mix 上是空转**：数据每样本恰好 1 个 near（K=1），而
   `NASSContrastiveLoss` 在 `K==1` 走的是**不含 β** 的分支（`nass_contrastive_loss.py:68-69`）。
   Table 16 的「τ 0→1/1000ep」在 v2 配置里逐项落地，却在数学上无法生效（§3.4、§5-P0-α）。
4. ⚠️ **margin=0.5 近似失效**：真实 run 里 `cos_sim_pos≈0.98`、`cos_sim_neg≈0.95`、
   `neg_active_ratio=1.0`（`metrics.csv` 实证），`relu(sim−0.5)` 恒为正 → 负项退化成「无阈值余弦排斥」。
5. **−SI-SNR vs −SNR 的差异是真的**：v2 `pairwise_neg_sisdr`（投影残差）vs LEGACY `pairwise_neg_snr`
   （直接相减、尺度敏感），`matrix.py:222-225`。且两者**都没有下界截断**，v2 还关掉了唯一的
   `threshold_byloss` 过滤（`configs/…v2.yml:126`）→ 退化估计可产生 −48 dB 级 loss（实测 §4.2）。
6. **global batch**：v2 + 8 卡 = 4×8×1 = **32** ✓ 对齐论文；但派生的 `_2gpu` 用 accum 5 → **40**
   （`run_lrs2_paper_2gpu_smoke.sh:2` 自称「与论文对齐」是错的，要 32 得 accum 4）；本机 2 卡跑 v2 只有 8。
7. **epoch 语义**：`train_samples=10000` = 「10000 个随机 2 s 窗口」，池子 40000 → 1 epoch 只覆盖
   ≈22% 的唯一 mix；而**验证每 epoch 跑满 cv 10000 条（bs=1）**，FLOPs 与训练同量级（§5-P1-γ）。
8. **实测正确性**：`ReduceLROnPlateau` 收到的是**当 epoch** 的 val_loss（实测 `[36.90, 34.83]`），
   与 Lightning 组合**没有滞后坑**；`optimizer.step()` 计数 4/4 与 `accumulate_grad_batches=2` 相符。
9. **坑**：`config_to_hparams` 把 list 转 `torch.Tensor` → 真实 `hparams.yaml` 含 5 处 `!!python` 标签，
   `yaml.safe_load` 直接失败（`conf.yml` 可安全读，§4.3）。
10. **坑**：`conf.yml` 里 `main_args` 是空字典，`--conf_dir/--ckpt_path` 落到 `options:` 组
    （`parser_utils.py:143-148` 在 Py≥3.10 取错组名，`[实测]` §4.1）。
11. **坑**：启动脚本打印的 steps/epoch 把 accum 漏在分母外（打印 1250，真值 250，`§4.4`）。
12. **坑**：`AVSpeechEvalDataset` 靠 `<mix>_meta.json` 数 near，论文数据集**没有**该文件 →
    val 分桶指标全塌到 `val_pit_sisnr_0near`（真实 run 的 metrics.csv 只有这一列，§4.5）。
13. **配置无漂移**：`studio/configs/Falcon.yml` 与 `configs/Falcon.yml` **值完全相同**（仅 2 行注释差），
    Space 由 `space/build.py:32` 从 `studio/configs/` 拷贝（§3.5）。
14. **DDP 要开 `find_unused_parameters=True`**：不是因为对比头，而是 `concat_layers.2` 那 15 个张量
    **真的拿不到梯度**（`[实测]` grad=None 清单，§4.2）。不开这个开关，多卡会在第一次 backward 报
    unused-parameter 错误；这条同时给「P0-7 二层融合未接通」补上了梯度级证据。
15. **本轮无法验证**：8×3090 的卡数、真实多卡 DDP、1000 epoch 收敛行为（只有 2 卡，且本轮不占卡）。

---

## 1. 事实基线

### 1.1 本域文件（HEAD `6c1d5b7`）

| 文件 | 行数 | 角色 | 本轮判定 |
|---|---:|---|---|
| `audio_train.py` | 456 | 唯一训练入口，装配全部对象 | 可运行（§4.2） |
| `look2hear/system/av_litmodule.py` | 247 | LightningModule 基类（val/test/日志/优化器装配） | 在用；仍有 1 处 DDP 隐患（`170-176`） |
| `look2hear/system/av_litmodule_nass_v2.py` | 185 | **唯一在用的** system（NASS 对比） | 在用 |
| `look2hear/system/av_litmodule_mdcl.py` | 173 | MDCL 分支 | **死**（只被 tests/verification 导入） |
| `look2hear/system/optimizers.py` | 106 | `make_optimizer`/`get` 注册表 | `make_optimizer` 在用；`register_optimizer` 死 |
| `look2hear/system/schedulers.py` | 111 | `DPTNetScheduler` | **死**（入口用 `torch.optim.lr_scheduler`） |
| `look2hear/system/__init__.py` | 11 | 导出 | 缺 `WithMDCL` 导出（进一步说明它不在产品路径） |
| `look2hear/utils/parser_utils.py` | 151 | YAML→argparse 回填 | 在用，有版本分支 bug |
| `configs/*.yml` | 154/155/155/158/172/173 | 6 个配置 | §3.2/3.3 |
| `tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh` | 171 | 8 卡启动（自检+渲染+拉起） | 可用，1 个打印 bug（§4.4） |
| `tools/repro/phase2/run_lrs2_paper_2gpu_smoke.sh` | 17 | 2 卡 smoke | 注释与事实不符（§6） |
| `tools/repro/phase2/run_p2d_train.sh` | 63 | P2-D 正式训练 | `cd $WORK` 后产物落到仓库外（有意） |

### 1.2 依赖事实（`[实测]`）

```
torch_optimizer 0.3.0        comet_ml MISSING     tensorboard MISSING
pytorch_lightning 2.6.6      torch 2.8.0+cu128    soundfile 0.14.0   torchvision 0.23.0+cu128
```
→ `look2hear/system/__init__.py:1` 一进来就 `from .optimizers import …`，而 `optimizers.py:3-19`
硬依赖 `torch_optimizer`：**torch_optimizer 缺失会让 `import look2hear.system` 直接失败**（当前已装，OK，
但它是隐式硬依赖，`requirements` 里没写）。
`comet_ml`/`tensorboard` 缺失意味着 `training.logger: comet|tensorboard` 两条分支在本环境**只能抛错**。

### 1.3 数据事实（`[实测]`，决定 epoch/批/桶语义）

| split | mix.json 条数 | 段长 | 每样本 near 数（near.json 直方图 0/1/2/3/4/5+） | `_meta.json` 存在率 |
|---|---:|---|---|---:|
| `falcon_lrs2_2mix_paper/tr` | 40000 | 32000 (=2.0 s) 全部 | `[0, 40000, 0, 0, 0, 0]` | **0 / 40000** |
| `falcon_lrs2_2mix_paper/cv` | 10000 | 32000 全部 | `[0, 10000, 0, 0, 0, 0]` | **0 / 10000** |
| `falcon_lrs2_2mix_paper/tt` | 6000 | 32000 全部 | `[0, 6000, 0, 0, 0, 0]` | **0 / 6000** |

- near 路径是 `…/min/tr/s2/<same-stem>.wav`（**另一说话人的 clean 源**），与 mix 路径不同 →
  `avspeech_dataset_near.py:190` 的「同路径跳过」不会触发 → **训练集 K 恒等于 1**（实测 batch 见 §4.3）。
- `_meta.json` 全缺 → `avspeech_dataset.py:208-216` 的 `n_near` 恒为 0 → **验证集每个样本都落进 bucket 0**。

---

## 2. 机制拆解

### 2.1 入口全流程：argv → conf → Data → Module → Trainer

`audio_train.py:448-456` 的 `__main__` 是唯一入口；**没有 hydra**，配置回填靠 `parser_utils`。

| 步 | 位置 | 代码做什么 | 语义要点 |
|---|---|---|---|
| S0 | `audio_train.py:31-44` | 建 `ArgumentParser`，只加 `--conf_dir`（默认 `configs/Falcon.yml`）、`--ckpt_path`（默认 `""`） | 两个参数**只在这里定义一次**；其余参数由 S1 动态生成 |
| S1 | `:452-453` | `yaml.safe_load(open(args.conf_dir))` → `def_conf` | 6 个 yml 都是 2 层嵌套（`videonet.audionet_config` 等） |
| S2 | `:454` → `parser_utils.py:5-43` | 按「第一层=argparse group，第二层=`--key`」展平；`--audionet_config` 的 `type=dict`、`default=<整个子 dict>` | **共 29–32 个 flat action**（实测）；深层键不展平，整个 dict 作为一个 default 传下去 |
| S3 | `:455` → `parser_utils.py:117-151` | `parse_args_as_dict` 反向拼回 `{group: {key: value}}` | `[实测]` 6 个 yml 的 round-trip 全部 **`== yaml`**（§4.1） |
| S4 | `:456` | `main(arg_dic, ckpt_path=args.ckpt_path)` | `ckpt_path` 从 `plain_args` 取，不走 arg_dic |
| S5 | `main:271` | `resolve_resume_checkpoint`：空→None；文件不存在→`SystemExit` | `:248-253`「拒绝静默从头训」 |
| S6 | `:274-275` | `set_seed(config["training"].get("seed"))` → `pl.seed_everything(seed, workers=True)` | 只有 v2 配了 42；其余 5 个 config **无 seed** → 不设种子 |
| S7 | `:281-289` | `exp_dir = cwd/Experiments/checkpoint/<exp_name>`；`deepcopy` 后写 `conf.yml` | 注释 `:277-280` 说明要在建模型**之前** dump（否则 `repeats/shared` 被 pop、`causal` 被注入） |
| S8 | `:291-298` | `use_near` 读取；`AVSpeechDataModuleWithNear(**data_config)`；`setup()`；`make_loader` → (train,val,test) | `data_name` 键**不读**，直接 import 类 |
| S9 | `:301-305` | `getattr(look2hear.models, audionet_name)(sample_rate=…, **audionet_config)` | `sample_rate` 来自 `datamodule.data_config`，不是 audionet 段 |
| S10 | `:308-310` | `getattr(look2hear.videomodels, videonet_name)(**videonet_config)` | FRCNN 加载冻结骨干（缺文件会 `FileNotFoundError`，`frcnn_videomodel.py:96-101`） |
| S11 | `:313-321` | `make_optimizer(model.parameters(), **config["optimizer"])`；`torch.optim.lr_scheduler.<sche_name>(optimizer=…, **sche_config)` | **优化器只包 audio_model**（视频权重不在 `param_groups` 里，`[实测]`=0 个） |
| S12 | `:328-337` | 两个 loss：`getattr(look2hear.losses, loss_func)(getattr(look2hear.losses, sdr_type), **config)` | `PITLossWrapper(PairwiseNegSDR 实例, pit_from, threshold_byloss)` |
| S13 | `:340-353` | 读 7 个 `spk_contrast.*`（缺省值写在 `.get` 里） | 缺键不会 KeyError（唯一做了默认的段） |
| S14 | `:356-373` | `AudioVisualLightningModuleNASSv2(...)` | **`training.system` 不读**（`[代码]` 直接 import 类） |
| S15 | `:376-387` | `ModelCheckpoint(exp_dir, filename="{epoch}", monitor="val_loss/dataloader_idx_0", save_top_k=5, save_last=True)`；可选 `EarlyStopping(**early_stop)` | monitor 要写 list-loader 的自动后缀键 |
| S16 | `:390` | `logger = build_logger(config, exp_dir)` | 默认 `csv` |
| S17 | `:394-400` | `resolve_accelerator_and_devices(config["training"])` | 见 §2.6 |
| S18 | `:402-418` | `pl.Trainer(max_epochs, accumulate_grad_batches, callbacks, default_root_dir, devices, accelerator, strategy, limit_train_batches=1.0, gradient_clip_val=5.0, logger, sync_batchnorm=True, num_sanity_val_steps=0)` | 全部硬编码，只有 epochs/accum 来自 config |
| S19 | `:421-425` | `trainer.fit(system, ckpt_path=resume_from)` 或 `fit(system)` | **不复用 conf 里的 loader**：DataModule 只用来取三个 loader，真正喂数据走 `system.train_dataloader()` |
| S20 | `:430-445` | 写 `best_k_models.json`；把 best ckpt 载回、`cpu()`、`audio_model.serialize()` → `best_model.pth` | `best_model.pth` 只有 audio（实测 2,376,222 B） |

**注意 S19 的双重身份**：`AVSpeechDataModuleWithNear` 继承的是**普通 object**（`avspeech_dataset_near.py:226`
`class AVSpeechDataModuleWithNear:`，`super().__init__()` 只是 object），它不是 `pl.LightningDataModule`。
`make_loader` 是 `@property`（`:331-333`），所以 `audio_train.py:298` 拿到的是元组。真正生效的 loader 是
`av_litmodule.py:217-223` 的三个方法。→ 这意味着 **`Trainer` 的 `use_distributed_sampler`（PL 2.x 默认 True）
会在 `system.train_dataloader()` 返回的 loader 上换 `DistributedSampler`**，而 **val/test 是长度为 2 的 list**，
每个 loader 各自被换 sampler（每 rank 拿到 `10000/N` 与 `6000/N` 条）。

#### 2.1.1 `args` 默认值与语义（全量）

| 参数 | 默认 | 语义 | 谁消费 |
|---|---|---|---|
| `--conf_dir` | `configs/Falcon.yml` | 唯一的配方来源 | `:452` |
| `--ckpt_path` | `""`（空=从头训） | 续训 ckpt；路径不存在时 `SystemExit`（拒绝静默从头训） | `:451,456` → `:271` |
| `--audionet_name` | `FalconSpeakerContrastive` | 音频模型类名 | `:302` |
| `--audionet_config` | 整个子 dict | 模型超参 | `:304` |
| `--videonet_name` | `FRCNNVideoModel` | 视觉模型类名 | `:308` |
| `--videonet_config` | dict | 视觉超参 | `:309` |
| `--train` / `--val` | 两个 dict | **注意**：`loss` 段的两个子键被展平成 `--train`/`--val`（组内键名冲突风险） | `:329-336` |
| `--system` | `AudioVisualLightningModuleNASSv2` | **死键** | 无 |
| `--gpus` | 视 config（list / int） | 卡选择（§2.6） | `:116` |
| `--parallel` | `ddp` | **死键** | 无 |
| `--epochs` | 视 config | `max_epochs` | `:405` |
| `--early_stop` | dict | `EarlyStopping(**…)` | `:386-387` |
| `--logger` | `csv`（Falcon.yml 无此键→`.get("logger","csv")`） | 见 `build_logger` | `:177` |
| `--accumulate_grad_batches` | 缺省 1（v2=1，2gpu/smoke=5，8gpu=1） | 梯度累积 | `:402` |
| `--seed` | 缺省 None（只有 v2=42） | `pl.seed_everything` | `:274` |
| `--optim_name/--lr/--weight_decay` | `adamw/0.001/0.01` | `make_optimizer(**optimizer 段)` | `:314` |
| `--sche_name/--sche_config` | `ReduceLROnPlateau/{patience:15,factor:0.5}` | `torch.optim.lr_scheduler.<name>` | `:317-321` |
| `--data_name` | `AVSpeechDataModuleWithNear` | **死键** | 无 |
| `--train_dir/--valid_dir/--test_dir/--batch_size/--num_workers/--pin_memory/--persistent_workers/--train_samples/--use_near/--segment/--normalize_audio/--n_src/--sample_rate` | 见 §3.2 | 直接进 DataModule 构造 | `:296` |
| `--exp_name` | 视 config | `exp_dir` 与 logger name | `:281,178` |

#### 2.1.2 实际是死代码的分支（逐条）

| # | 位置 | 内容 | 判定依据 |
|---|---|---|---|
| D1 | `audio_train.py:194-208` | TensorBoard 分支 | 本环境无 `tensorboard` → 只会抛 `RuntimeError`（`[实测]` 依赖检查） |
| D2 | `audio_train.py:209-227` | Comet 分支 | 无 `comet_ml`；且 6 个 config 无一处 `logger: comet` |
| D3 | `av_litmodule.py:125-140` | test-loss 分支 | 条件 `current_epoch % 10 == 0` → **10 个 epoch 里 9 个不执行**（真实 run 的 metrics.csv 只在 epoch 0 出现 `test_loss/dataloader_idx_1`） |
| D4 | `av_litmodule.py:208-213` | `interval` assert 分支 | 只有传入 **dict** 才走到；`audio_train.py:319` 传的是 **scheduler 对象** → 恒走 `:201-203`。若真传 dict 会 `KeyError: 'interval'`（`setdefault` 只补了 monitor/frequency） |
| D5 | `av_litmodule.py:114-118` | `n_near_batch` 的 list/scalar 兼容分支 | 真实 collate 给的是 tensor（`[实测]` `vb[4]=tensor([0])`） |
| D6 | `av_litmodule_nass_v2.py:20-26` | `_BUCKET_LABEL` | 定义后**从未引用**（`grep` 只命中定义行） |
| D7 | `av_litmodule_mdcl.py` 整文件 | MDCL system | `audio_train.py` 与 `system/__init__.py` 都不 import 它 |
| D8 | `schedulers.py` 整文件 | `BaseScheduler`/`DPTNetScheduler` | 入口走 `torch.optim.lr_scheduler`；`schedulers.py` 仅被 `system/__init__.py:2` 导出 |
| D9 | `optimizers.py:71-85` | `register_optimizer` | 全仓无调用 |
| D10 | `nass_contrastive_loss.py:68-72` | `K>=2` 的 softmax(β) 分支 | 本数据集 K≡1 → **β 永不生效**（§5-P0-α） |
| D11 | `audio_train.py:412` | `limit_train_batches=1.0` | 常量；控制 epoch 长度的旋钮其实是 `train_samples` |
| D12 | `configs/*.yml: training.system / training.parallel / datamodule.data_name` | 3 个键 | 全仓 `grep` 只有 tests 断言其**值**，无消费者 |
| D13 | `audio_train.py:124-132` | `gpus: 0` / `gpus: -1` / 缺省 三路分支 | 可达但不被任何 config 命中（6 个 config：list 或 -1） |

### 2.2 三个 LightningModule 的差异表

| 维度 | `AudioVisualLightningModule`（基类，247L） | `…NASSv2`（185L） | `…WithMDCL`（173L） |
|---|---|---|---|
| 是否在用 | 在用（作为父类 + val/test/optimizer） | **在用**（唯一实例化） | 死 |
| `training_step` | `:66-84` 只算 PIT | **重写** `:73-185` | 重写 `:74-173` |
| forward 调用 | `self(mixtures, mouth)` → `forward:53-64` → `audio_model(wav, mouth_emb)`，**eval 时返回 Tensor、train 时返回 2 元组**（`Falcon_Contrastive.py:116-118`） | **不用 `self(...)`**，直接 `self.audio_model(mixtures, mouth_emb)` 解包 2 值（`:90`） | 同 NASSv2，但只取 1 值（`:91`） |
| loss 组成 | `PIT(est,tgt)` | `PIT + λ·NASS(query=pool(emb[:,0]), pos=pool(enc(tgt)), negs=[pool(enc(near_k))])`；`λ=contrastive_weight`（`:137-138`） | `PIT + mdcl_alpha·(Σ_b relu(mag_sim(est_b, near)−0.5)/B)`（`:154-155`） |
| β 退火 | — | `_get_beta:66-70`，`beta_init+(final-init)*min(epoch/anneal,1)` | 同名同实现 `:67-71`，默认 anneal=200（NASSv2 默认 100） |
| 负样本粒度 | — | 逐样本循环 `:115-132`，每个 near 一次 `encode_stft`（no_grad） | 逐样本循环 `:109-127`，每个 near 一次 `_mdcl_loss_hard`（**有梯度**，作用在波形上） |
| grad=None 风险 | 无（PIT 走全网） | 无新增 | `_mdcl_loss_hard` 反传到 `est_b`，与 PIT 共享图 → 额外开销 |
| logging key | `train_loss`（`:75-82`） | `train_loss / train_pit_loss / train_spk_contrast_loss / train_beta / train_cos_sim_pos / train_cos_sim_neg_avg / train_cos_sim_neg_max / train_neg_active_ratio / train_n_samples_{0,1,2,3,4plus}_near`（`:170-183`） | `train_loss / train_pit_loss / train_mdcl_loss / train_beta / train_mdcl_{0..4plus}near / train_n_samples_{…}`（`:158-171`） |
| `configure_optimizers` | `:190-215` 共用 | 继承 | 继承 |
| 额外依赖 | — | `audio_model.pool_spk_embed`、`encode_stft`（`Falcon_Contrastive.py:72-91`） | `audio_model._mdcl_loss_hard`、`_mag_sim`、`mdcl_alpha/mdcl_margin`（`Falcon.py:1389-1478`） |
| 实例化入口 | `system/__init__.py:3` | `audio_train.py:26,356` | 无 |

**结论**：`training.system` 这个键（6 个 config 全写 `AudioVisualLightningModuleNASSv2`）**不参与任何选择**；
把 v2 里的 `system` 改成 `WithMDCL` 不会有任何效果，只会让 `conf.yml` 的语义撒谎。

### 2.3 NASSv2 的 training_step 张量流（逐行，形状实测）

```
batch(dict-less tuple) ← collate_with_near  [avspeech_dataset_near.py:23-43]
  mixtures [B,32000] f32   ← 4 路 mono mean 后 stack   [实测 (4,32000)]
  targets  [B,32000] f32                              [实测 (4,32000)]  → :76-77 unsqueeze → [B,1,T]
  mouth    [B,50,88,88] f64（Grayscale+CenterCrop+Normalize 的输出仍是 f64）
  filenames list[str]
  near_audios list[list[Tensor[T]]]                    [实测 (4,)=>[1,1,1,1]]

:86-87  with no_grad: mouth_emb = video_model(mouth.unsqueeze(1).type_as(mixtures))   [B,512,50]
:90     est_sources, sep_audio_embs = audio_model(mixtures, mouth_emb)
        est_sources    [B,1,32000]        [实测 (1,1,32000)]
        sep_audio_embs [B,1,112,251,129]  [实测 (1,1,112,251,129)]   ← N=112, T'=1+L/128=251, F=129
:95     pit_loss = loss_func["train"](est_sources, targets)  → 标量
:102-103 query_feat = sep_audio_embs[:,0] [B,112,251,129] → pool_spk_embed → [B,64]（带梯度）
:106-109 with no_grad: pos = pool(encode_stft(targets[:,0])) → [B,64]
:115-132 逐 b：near_wavs [K,32000] → encode_stft → pool → [K,64]（no_grad）
:134    cl_loss = NASSContrastiveLoss(query,pos,negs,beta)   → [1] 标量
:137-138 total = pit + λ·cl
:141-167 相似度监控（cos_pos / cos_neg mean,max / neg_active_ratio）
:170-183 8 + 5 个 key 全部 sync_dist=True
```

**一个容易漏看的耦合**：`:90` 依赖 `FalconSpeakerContrastive.forward` 在 `self.training=True` 时返回
**2 元组**（`Falcon_Contrastive.py:116-118`）。而 `validation_step` 走 `self(...)` 并在 eval 模式下拿**单值**
（`av_litmodule.py:127`）。这两处一旦谁把 `num_sanity_val_steps` 调 >0 且模型没被 PL 置 eval，就会 `ValueError`。
当前 `audio_train.py:417` 写死 `num_sanity_val_steps=0`，所以是「靠一个别的参数保命」。

### 2.4 优化器 / 调度器

**注册表机制**（`optimizers.py`）：
- `make_optimizer(params, optim_name="adam", **kwargs)`（`:51-68`）→ `get(optim_name)(params, **kwargs)`。
- `get`（`:88-106`）把 `globals()` 全部 **小写化** 成字典查表 → `"adamw"` 命中 `torch.optim.AdamW`（`:2`）。
- 因为 `config["optimizer"]` 的键**恰好叫 `optim_name`**，`**kwargs` 才会把 lr/wd 留给 `get(...)` 的调用。
  改键名（如 `optimizer_name`）会立刻把整个 dict 塞进优化器构造器 → `TypeError`。危险的隐式契约。
- `[实测]` 实际 `param_groups`：`{'lr':0.001,'betas':(0.9,0.999),'eps':1e-08,'weight_decay':0.01,'decoupled_weight_decay':True}`
  → 确认是 **AdamW**（decoupled）+ lr 1e-3 + wd 0.01（论文一致）。
- 副作用：`get` 的查表包含 `get`、`make_optimizer`、`register_optimizer` 本身 → `optim_name: "get"` 会返回函数。

**调度器**（`audio_train.py:317-321`）：`getattr(torch.optim.lr_scheduler, sche_name)`，
`sche_config={patience:15, factor:0.5}` → `ReduceLROnPlateau(mode='min', factor=0.5, patience=15,
threshold=1e-4, cooldown=0, min_lr=0)`（其余取 torch 默认）。

**Lightning 组合（`av_litmodule.py:190-215`）**：
```
若 self.scheduler is None → 直接返回 optimizer（:192-193）
否则：非 list → 包成 list（:195-196）
  sched 非 dict 且是 ReduceLROnPlateau → {'scheduler': sched, 'monitor': 'val_loss/dataloader_idx_0'}（:201-203）
返回 ([optimizer], [spec])
```
→ 最终 spec 实测为 `[{'scheduler': 'ReduceLROnPlateau', 'monitor': 'val_loss/dataloader_idx_0'}]`，
**没有 `interval`/`frequency` 键**，由 PL 补默认（interval=epoch, frequency=1）。

**有没有坑？`[实测]` 结论：没有滞后坑。** 我给 `scheduler.step` 打了记录器，两个 epoch 收到
`[36.901161193847656, 34.82929992675781]`，正好等于 epoch 0 / epoch 1 的 `val_loss/dataloader_idx_0`
（metrics.csv 同值）→ PL 2.6.6 在 **同一 epoch 的 val 结束后**才 step，不滞后。
真正的坑只有两个：
- `patience=15` 是 **epoch 级** 的，而 1 epoch = 10000 个随机窗口（§5-P1-γ），即 patience 实际等待
  ≈15×10⁴ 个窗口；与论文「patience 15」名义一致、语义未必一致（论文一个 epoch 覆盖多少数据未披露）。
- `:146-152` 在 `on_validation_epoch_end` 里 `self.log("lr", …)`，与 `:162` 的 `learning_rate` 重复
  （metrics.csv 两列同值）。

### 2.5 损失链

| 组件 | 位置 | 语义 |
|---|---|---|
| `PITLossWrapper(loss_func, pit_from="pw_mtx", threshold_byloss=…)` | `pit_wrapper.py:152-324` | 先算 `[B,n_src,n_src]` 成对矩阵（`:171`），再 `find_best_perm`（`:205`），`threshold_byloss=True` 时**丢掉 loss<-30 的样本**（`:227-230`），最后 `mean` |
| `n_src=1` 的后果 | — | 排列只有 1 种 → PIT **退化为普通 SI-SNR 均值**，PIT 机制在这里是空壳 |
| `PairwiseNegSDR("sisdr")` | `matrix.py:188-235`，别名 `:372` | `e_noise = s_est − proj`（投影残差）→ 真 SI-SDR；`zero_mean=True`；**无上/下界** |
| `PairwiseNegSDR("snr")` | 别名 `:374` | `e_noise = s_est − s_target` → 尺度敏感的 −SNR |
| `NASSContrastiveLoss` | `nass_contrastive_loss.py:18-81` | `L = α₊(1−cos(q,p)) + α₋·Σ_k q_β(k)·relu(cos(q,n_k)−m)`；`K==1` 时 `l_neg = α₋·relu(sim−m)`（**无 β**）；**K==0 时仍保留 `l_pos`**（`:73-77` `n_valid` 无条件 +1）→ 实测 K=0 且随机初始化时 `cl=0.5369`（≠0） |
| `_mdcl_loss_hard` | `Falcon.py:1421-1475` | 幅度谱版；**K==0 返回 0**（`:1453-1454`），与 NASS 的 K=0 行为**不一致** |

### 2.6 分布式与批

| 项 | 实际值 | 位置 |
|---|---|---|
| `_normalize_gpus_spec` | `None/""`→none；`"0,1"`/list→indices；int>0→count；`0`→cpu；`-1`→all；int<−1→ValueError | `audio_train.py:50-95` |
| `resolve_accelerator_and_devices` | 无 CUDA→`("cpu",1,1,"auto")`；indices 会**裁剪到可见卡并 warning**；空 list/重复/index 越界→`ValueError` | `:98-166` |
| strategy | `num_devices>1` → `DDPStrategy(find_unused_parameters=True)`，否则 `"auto"` | `:165` |
| `replace_sampler_ddp` | **不存在**（`[实测]` PL 2.6.6 的 `Trainer.__init__` 签名里查无此项）；实际用 `use_distributed_sampler`（默认 **True**，`[实测]`）→ DDP 下自动把 `shuffle=True` 的 loader 换成 `DistributedSampler(shuffle=True)` | `:404-418` 两个 kwargs 都没传 |
| `sync_batchnorm` | `True`（单卡时 PL 不动；多卡会把含 BN 的模块转 SyncBN） | `:416` |
| `find_unused_parameters=True` | 需要开：实测 15 个 `separator.audio_net.concat_layers.2.*` 张量 `grad=None`（§4.2），不开会在多卡报错 | `:165` |
| `gradient_clip_val` | 5.0（硬编码） | `:413` |
| `accumulate_grad_batches` | config（v2=1 / 2gpu=5 / 8gpu=1） | `:402` |
| `limit_train_batches` | 1.0（硬编码，不暴露） | `:412` |
| `num_sanity_val_steps` | 0（硬编码，且 val 很贵，见 §5-P1-γ） | `:417` |
| train sampler | `shuffle=True` → PL 换成 `DistributedSampler(shuffle=True)`；**但 `__getitem__` 忽略 idx**（`avspeech_dataset_near.py:154`）→ 分片只影响「各 rank 抽多少条」，不影响抽哪条 | `:297-307` |
| `drop_last` | train **True**（`:305`）、val/test **True**（`:317,328`，bs=1 时无效果） | — |
| `num_workers` / `pin_memory` / `persistent_workers` | 8 / True / True（val/test 也用同一组，即 bs=1 也开 8 worker） | `:302-304,313-316` |
| global batch | `batch_size × n_gpu × accumulate` | 实测：smoke run 50 step/epoch = 2000/(4×2×5) ✓；dry-run 打印 1250 是**错的**（§4.4） |
| metric 聚合 | **训练路径完全不 import `look2hear.metrics`**；聚合只有 `self.log(..., sync_dist=True)`（均值归约）+ `on_validation_epoch_end` 里手工 `all_gather` | `[实测]` grep 无命中 |

**DDP 聚合的唯一隐患（仍未修）**：`av_litmodule.py:170-176`
```python
for bucket, name in _bucket_names.items():
    outputs = self._val_bucket_outputs[bucket]
    if outputs:                      # ← 只在本 rank 该桶非空时才进入
        avg = torch.mean(torch.stack(outputs))
        avg = torch.mean(self.all_gather(avg))   # ← 集合通信
```
条件化集合通信：若某桶在 rank0 非空、rank1 为空 → rank1 直接跳过 `all_gather` → **NCCL 挂死**。
本数据集因为 `_meta.json` 全缺（n_near≡0）**恰好对称**，所以真实 2 卡 smoke 没踩（与旧文档 §V15 一致）；
一旦补上 `_meta.json`（或换 RealSSA-Sim 的不均匀 near 数）就会踩。

---

## 3. 关键数据/参数表

### 3.1 六个 config 的日志 key（由代码推导 + 实测列名核对）

| key | on_step | on_epoch | sync_dist | 来源 |
|---|:-:|:-:|:-:|---|
| `train_loss` | 默认 True | True | True | `av_litmodule_nass_v2.py:172` |
| `train_pit_loss` / `train_spk_contrast_loss` / `train_beta` | True | True | True | `:174-176` |
| `train_cos_sim_pos` / `_neg_avg` / `_neg_max` / `train_neg_active_ratio` | True | True | True | `:177-180` |
| `train_n_samples_{0,1,2,3,4plus}_near` | True | True | True | `:182-183` |
| `val_loss/dataloader_idx_0` | False（val hook 默认） | True | True | `av_litmodule.py:96-103` |
| `test_loss/dataloader_idx_1` | False | True | True | `:131-138`（仅 epoch%10==0） |
| `lr` | — | True | True | `:146-152` |
| `learning_rate` / `val_pit_sisnr` | False | True | False（已手工 all_gather） | `:161-164,186` |
| `val_pit_sisnr_{0,1,2,3,4plus}near` | False | True | False（已手工 all_gather） | `:167-176` |
| `test_pit_sisnr` | False | True | False | `:181-184` |

真实 run 的 `metrics.csv` 列名与上表**逐字吻合**（§4.5），且只有 `val_pit_sisnr_0near`（证明 bucket 1–4 恒空）。

### 3.2 配置逐键字典（语义 + 消费者）

`data_config` 的 13 个键与 `AVSpeechDataModuleWithNear.__init__`（`avspeech_dataset_near.py:237-252`）
**一对一**，多一个键就 `TypeError`。

| 键 | 值（v2 为准） | 语义 | 消费者 `path:line` |
|---|---|---|---|
| `videonet.videonet_name` | `FRCNNVideoModel` | 视觉前端类名 | `audio_train.py:308` |
| `videonet_config.backbone_type` | `resnet` | ResNet18 主干 | `frcnn_videomodel.py:41-44` |
| `…relu_type` | `prelu` | 前端激活 | `:54-56` |
| `…width_mult` | `1.0` | 仅 shufflenet 用（resnet 下**忽略**） | `:47` |
| `…pretrain` | `/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar` | 冻结骨干权重 | `:86-112` `init_from` |
| `audionet.audionet_name` | `FalconSpeakerContrastive` | 音频类名（带对比头） | `:301-302` |
| `audionet_config.n_src` | `1` | 只出 1 路（目标说话人提取，不是 2 源分离） | `Falcon.py:1302,1336-1341` |
| `…pretrained_vout_chan` | `512` | 视频特征通道（FRCNN backend_out） | `Falcon.py:1323` |
| `…causal` | `true` | 注入到 `audio_params/video_params["causal"]` | `Falcon.py:1325-1326` |
| `…spk_embed_dim` | `64` | 对比 embedding 维度 | `Falcon_Contrastive.py:63-69` |
| `…video_bn_params.kernel_size` | `-1` | 全窗卷积（ConvNormAct） | `Falcon.py:1323` |
| `…audio_bn_params.{pre_norm_type,pre_act_type,out_chan,kernel_size,is2d}` | `cLNhw/ReLU/112/1/true` | 音频 bottleneck | `Falcon.py:1322` |
| `…enc_dec_params.{win,hop_length,out_chan,kernel_size,stride,bias,act_type,norm_type,causal}` | `256/128/112/3/1/false/null/null/true` | STFT 编解码（16 ms 窗 / 8 ms hop） | `Falcon.py:1319,1343` |
| `…audio_params.*`（n_freqs 129 / hid_chan 64 / kernel 5 / stride 2 / cLNhw / PReLU / upsampling_depth 4 / repeats 2 / shared true / layers 3 层） | — | 音频分离主干；`repeats`/`shared`/`layers` 被 `Separator` **pop** | `Falcon.py:1328-1334`、`Separator.__init__` |
| `…video_params.*`（hid 32 / k 3 / s 2 / BatchNorm1d / PReLU / repeats 1 / shared / is2d false / layers.layer_1） | — | 视频细化分支（`stride`/`layers` 实际不被使用，旧文档 §10.3） | 同上 |
| `…fusion_params.{fusion_type,fusion_shared,kernel_size,is2d}` | `NeuroSCNetFusion/true/4/true` | 融合块；`fusion_type` 是**死键**（类型写死） | `Falcon.py:1332` |
| `…mask_generation_params.{mask_generator_type,mask_act,RI_split,is2d}` | `MaskGenerator/ReLU/true/true` | 掩码生成；`mask_generator_type` 死键 | `Falcon.py:1336-1341` |
| `spk_contrast.margin` | `0.5` | 余弦 margin | `audio_train.py:342` → `NASSContrastiveLoss` |
| `…alpha_pos / alpha_neg` | `1.0 / 1.0` | 正/负项权重 | `:343-344` |
| `…beta_init / beta_final` | `0.0 / 1.0` | τ 起点/终点 | `:345-346` |
| `…beta_anneal_epochs` | **1000**（v2）；LEGACY 200 | τ 线性退火长度 | `:347` → `nass_v2:69` |
| `…loss_weight` | `1.0` | 论文 λ_spk | `:341` |
| `loss.train.loss_func` | `PITLossWrapper` | 包装器类 | `:329` |
| `loss.train.sdr_type` | `pairwise_neg_sisdr`（v2） / `pairwise_neg_snr`（其余） | 重建损失种类 | `:330` |
| `loss.train.config.pit_from` | `pw_mtx` | 成对矩阵模式 | `:331` |
| `loss.train.config.threshold_byloss` | `false`（v2） / `true`（其余） | 是否丢弃 loss<-30 的样本 | `:331` → `pit_wrapper.py:227-230` |
| `loss.val.*` | 同构，恒 `pairwise_neg_sisdr` + `threshold_byloss:false` | 验证口径 | `:333-336` |
| `training.system` | `AudioVisualLightningModuleNASSv2` | **死键** | 无 |
| `training.gpus` | `-1`（v2/8gpu）；`[0,1]`（2gpu/smoke/2Mix）；`[0..9]`（Falcon） | 卡选择 | `:116` |
| `training.parallel` | `ddp` | **死键**（strategy 由卡数决定） | 无 |
| `training.epochs` | 1000 / 100 / 3 | max_epochs | `:405` |
| `training.early_stop.{monitor,mode,patience,verbose}` | `val_loss/dataloader_idx_0/min/30/true` | 早停 | `:386-387` |
| `training.logger` | `csv`（Falcon.yml **无此键**→默认 csv） | csv/tb/comet/none | `:177-232` |
| `training.accumulate_grad_batches` | 1（Falcon.yml 无） / 1 / 5 | 梯度累积 | `:402` |
| `training.seed` | 42（**只有 v2 有**） | 全局种子 | `:274` |
| `optimizer.optim_name/lr/weight_decay` | `adamw/0.001/0.01` | 优化器 | `:314` |
| `scheduler.sche_name` | `ReduceLROnPlateau` | 调度器类 | `:317-319` |
| `scheduler.sche_config.patience/factor` | `15/0.5` | 调度参数 | `:320` |
| `datamodule.data_name` | `AVSpeechDataModuleWithNear` | **死键** | 无 |
| `data_config.train_dir/valid_dir/test_dir` | `/root/data/falcon_lrs2_2mix_paper/{tr,cv,tt}` | 索引目录（含 mix/target/near.json + mouths/） | `:296` → `avspeech_dataset_near.py:85-97` |
| `…n_src / sample_rate / segment` | `1 / 16000 / 2.0` | 判据：`seg_len=32000`、`fps_len=50` | `:77-78` |
| `…normalize_audio` | `false` | 是否按 mix 标准差归一 | `:179-182` |
| `…batch_size` | 4（8gpu 是 5） | 每卡批 | `:299-300` |
| `…num_workers` | 8 | worker 数 | `:302` |
| `…pin_memory / persistent_workers` | `true / true` | — | `:303-304` |
| `…use_near` | `true` | 是否读 near.json（决定 K） | `:186-220` |
| `…train_samples` | 10000（smoke 2000） | **`__len__` 的返回值**（不是遍历数据集） | `:147-150` |
| `exp.exp_name` | `LRS2_2Mix_Paper_v2` 等 | 产物目录名 + logger name | `:281-282,178,193` |

### 3.3 差异矩阵（6 个 config）

`+` = 与 v2 相同。只列**有差异**的键：

| 键 | `Falcon.yml` | `LRS2_2Mix.yml` | `…_2gpu.yml` | `…_2gpu_smoke.yml` | `…_8gpu.yml` | `…_v2.yml` |
|---|---|---|---|---|---|---|
| 定位 | 发布模板 | Phase2-D 实跑 | 2 卡复现 | 3 epoch 冒烟 | **LEGACY** | **论文 Table 16** |
| `videonet_config.pretrain` | `pretrain_zoo/…`（**相对路径，仓库内不存在**） | `/root/.cache/…` | + | + | + | + |
| `loss.train.sdr_type` | `pairwise_neg_snr` | 同 | 同 | 同 | 同 | **`pairwise_neg_sisdr`** |
| `loss.train.config.threshold_byloss` | `true` | `true` | `true` | `true` | `true` | **`false`** |
| `spk_contrast.beta_anneal_epochs` | 200 | 200 | 200 | 200 | 200 | **1000** |
| `training.gpus` | `[0..9]` | `[0,1]` | `[0,1]` | `[0,1]` | `-1` | `-1` |
| `training.epochs` | 1000 | **100** | 1000 | **3** | 1000 | 1000 |
| `training.logger` | （缺省→csv） | csv | csv | csv | csv | csv |
| `training.accumulate_grad_batches` | （缺省→1） | （缺省→1） | **5** | **5** | 1 | 1 |
| `training.seed` | — | — | — | — | — | **42** |
| `data_config.train_dir` | `--`（占位） | `/root/data/falcon_lrs2_2mix/tr` | `/root/data/falcon_lrs2_2mix_paper/tr` | 同 + `valid/test`→`/root/tmp/paper_smoke/*` | `/root/data/falcon_lrs2_2mix_paper/tr` | 同 |
| `data_config.batch_size` | 4 | 4 | 4 | 4 | **5** | 4 |
| `data_config.train_samples` | 10000 | 10000 | 10000 | **2000** | 10000 | 10000 |
| `exp.exp_name` | Falcon | LRS2_2Mix | LRS2_2Mix_Paper_2gpu | …_smoke | …_8gpu | …_v2 |
| **有效 batch（8 卡口径）** | 32 | 32 | — | — | **40** | **32** ✓ |
| **有效 batch（2 卡口径）** | 8 | 8 | **40** | **40** | — | 8 |

`LRS2_2Mix.yml` vs `LRS2_2Mix_Paper_8gpu.yml`：除上表外**逐键相同**（两者都是 legacy 配方），
差别只在 epochs（100 vs 1000）、gpus、batch_size（4 vs 5）、数据目录（`falcon_lrs2_2mix` vs `_paper`）。
`…_2gpu.yml` 与 `…_2gpu_smoke.yml`：只差 `epochs`(1000→3)、`valid/test_dir`、`train_samples`(10000→2000)、`exp_name`。
`Falcon.yml` 与其余 5 个：除 `pretrain` 相对路径、`train_dir: --`、`gpus:[0..9]`、无 logger/accum/seed 外，
**模型段与损失段完全一致**（即：默认配置 = legacy 配方 + 占位数据路径）。

额外三个非 `configs/` 的配方（`tools/repro/phase2/`）全部是 legacy 配方（`neg_snr` + β/200）：

| 文件 | gpus | epochs | bs | train_samples | 备注 |
|---|---|---|---|---|---|
| `LRS2_2Mix_1gpu.yml` | 单卡 | 100 | 4 | 10000 | 旧单卡复跑 |
| `smoke_lrs2.yml` | 单卡 | 1 | 4 | 800 | 索引目录 `index_smoke` |
| `smoke_lrs2_bs8.yml` | 单卡 | 1 | 8 | 800 | 同上 |
| `LRS2_2Mix_near_s2.yml` | — | — | — | — | near 方案变体（本轮未展开） |

### 3.4 论文 Table 16 对齐（以 `configs/LRS2_2Mix_Paper_v2.yml` 为准）

> 论文 2026-10-23 已在 arXiv 公开（arXiv:2610.32749），本表按公开版 Table 16 逐项核对。

| 论文项 | 论文值 | v2 落地 | 判定 | 证据 |
|---|---|---|---|---|
| 重建损失 | −SI-SNR | `pairwise_neg_sisdr` + `threshold_byloss:false` | ✅ 对齐 | `v2.yml:122,126`；`matrix.py:222-225` |
| λ_spk | 1.0 | `spk_contrast.loss_weight: 1.0` | ✅ | `v2.yml:118` → `audio_train.py:341` |
| margin m | 0.5 | `margin: 0.5` | ⚠️ **数值对齐、语义失效** | `v2.yml:112`；实测 cos_neg≈0.95、`neg_active_ratio=1.0` |
| τ 0→1，前 1000 epoch | 0→1/1000ep | `beta_init 0 / beta_final 1.0 / anneal 1000` | ⚠️ **配置对齐、数学上无法生效** | `v2.yml:115-117`；`nass_contrastive_loss.py:68-69`（K==1 不带 β） |
| τ 之后固定 1 | 1 | 需 epoch≥1000 才到 1.0，而 `epochs:1000`（epoch∈[0,999]）→ 最大 β=0.999 | ⚠️ 差 1 个 epoch | `nass_v2:69` `min(epoch/1000,1)`；`v2.yml:137` |
| 硬件 8×RTX3090 | 8×3090 | 只有 2×A800；`gpus:-1` 在 2 卡机上 → 2 卡 | ❌ **无法表达**（硬件不可得） | §1 环境 |
| batch 4/GPU、global 32 | 4×8=32 | `batch_size:4` + `gpus:-1` + `accum:1`；真 8 卡 → 32 | ✅ 配方正确；本机只有 8 | `v2.yml:165,135,144` |
| epoch 语义 | 未披露 | 1 epoch = 10000 个**随机 2 s 窗口**（池 40000，覆盖≈22%） | ❓ **无法对齐**（论文未给 epoch 定义） | `avspeech_dataset_near.py:147-157`；§1.3 |
| seed 42 | 42 | `training.seed: 42` → `pl.seed_everything(42, workers=True)` | ✅ | `v2.yml:145`；`audio_train.py:265` |
| AdamW lr 1e-3 / wd 0.01 | — | ✅ | ✅ | `v2.yml:147-149` |
| ReduceLROnPlateau(15, 0.5) | — | ✅（factor 0.5 / patience 15） | ✅ | `v2.yml:151-154` |
| grad clip 5 | 5 | 硬编码 5.0 | ✅ | `audio_train.py:413` |
| early stop patience 30 | 30 | ✅ `monitor=val_loss/dataloader_idx_0` | ✅ | `v2.yml:138-142`；`:386-387` |
| max epochs 1000 | 1000 | ✅ | ✅ | `v2.yml:137` |

**一句话**：v2 把「能写进 yml 的都写了」，但 **τ 与 margin 这两个对比损失的核心旋钮在 LRS2-2Mix 数据上
数学失效**，而「8×3090 / global 32 / epoch 定义」这三项在本机/本仓库里**无法真正落地或核验**。

### 3.5 Studio / Space 侧的配置副本

- `studio/configs/Falcon.yml`（171 行）vs `configs/Falcon.yml`（173 行）：`diff` 只报 `130,131d129`
  ——即仅少两行注释，**所有键值完全一致**（md5 不同只因注释）。
- `studio/configs/FalconBig.yml`（166 行）：**big 架构专属**，`configs/` 下**没有**对应训练配置
  → Falcon-Big 无训练配方（与既有结论"Big 训练源未发布"一致）。
- Space 侧：`space/build.py:32` `shutil.copytree(STUDIO/"configs", out/"configs")`，
  `space/tests/test_static.py:41` 断言 build 产物与 studio 源 **字节相同** → **只有一份真源（studio）**。
- Studio 只消费 `audionet.audionet_config` + `videonet.*`（`studio/falcon_runtime.py:187-196`），
  `training.*` / `datamodule.*` / `loss.*` 在 Studio 侧**不被读取**（它们只是训练模板的搭车字段）。
- 结论：**训练配置与 Studio 配置之间没有值漂移**，但存在"同一份文件承担两种用途"的耦合：
  改 `audionet_config` 会同时改变训练与线上推理的模型结构。

---

## 4. 实测验证

### 4.1 实测 1：6 个 yml 的解析回填（`/root/tmp/w104/cfg.py`）

复刻 `audio_train.py:448-456` 的 argv 路径（`argparse` → `prepare_parser_from_dict` → `parse_args_as_dict`）：

```
CONF: configs/LRS2_2Mix_Paper_v2.yml
  group titles: ['positional arguments', 'options', 'videonet', 'audionet',
                 'spk_contrast', 'loss', 'training', 'optimizer', 'scheduler', 'datamodule', 'exp']
  arg_dic top keys: ['options', 'videonet', ..., 'exp', 'main_args']
  main_args: {}                      ← 空！
  options  : {'help': None, 'conf_dir': 'configs/LRS2_2Mix_Paper_v2.yml', 'ckpt_path': ''}
  roundtrip == yaml ? True           ← 6 个配置全部 True
  n_flat_args: 32 (Falcon.yml 29 / 2Mix 30 / 其余 31)
  training: {... 'accumulate_grad_batches': 1, 'seed': 42}
  bs/accum: 4 1 | epochs 1000 | gpus -1
```
**6 个 config 的 round-trip 全部字节级相等**（`rt == def_conf`），说明 `--xxx` 覆盖机制不会意外改语义。

**发现**：`main_args` 是 `{}`。原因是 `parser_utils.py:143-148`：
```python
if sys.version_info.minor >= 10:
    args_dic["main_args"] = args_dic["positional arguments"]   # Py3.11 里这个组是空的
```
Py ≥3.10 的**可选参数组标题已改名为 `'options'`**，所以 `--conf_dir/--ckpt_path` 落进 `options`。
后果：真实 `conf.yml` 同时含 `main_args:`（只有后面 `main()` 补的 `exp_dir`）和
`options: {help, conf_dir, ckpt_path}`（`[实测]` `Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/conf.yml:118,124-126`）。
不影响功能（`ckpt_path` 从 `plain_args` 取），但**任何按 `conf.yml["main_args"]["conf_dir"]` 读配置的工具都会 KeyError**。

### 4.2 实测 2：CPU 端 2 epoch 冒烟（真实 config、真实装配、假数据）

`/root/tmp/w104/smoke_fit.py`：按 `audio_train.py` 的同一条调用链建模型/优化器/调度器/损失/system，
用手工 collate 的假 batch（`mix[4,32000]`、`tgt[4,32000]`、`mouth[4,50,88,88]`、`near=[[...]]×4`）
喂 `pl.Trainer(max_epochs=2, accumulate_grad_batches=2, limit_train_batches=4, accelerator="cpu",
callbacks=[ModelCheckpoint(monitor="val_loss/dataloader_idx_0")])`：

```
[W1] audio params: 537697 video params: 11186688
Epoch 0, global step 2: 'val_loss/dataloader_idx_0' reached 36.90116 (best 36.90116), saving model ...
[W1] optimizer.step() calls: 4 (expect 4 = 2 epochs x 4 batches / accum 2)
[W1] scheduler.step() calls: 2 monitored values: [36.901161193847656, 34.82929992675781]
[W1] best_model_path: epoch=1.ckpt score: tensor(34.8293)
[W1] best_k: {'epoch=0-v1.ckpt': 36.9012, 'epoch=1.ckpt': 34.8293}
[W1] video_model.training after fit: True | audio model.training: True
[W1] ||delta_audio||=1.711600e+00  ||delta_video||=0.000000e+00
  delta[spk_pool_proj.proj.0.weight] = 1.527765e-01
  delta[spk_pool_proj.temporal_attn.0.weight] = 1.284932e-01
[W1] video params present in optimizer: 0
[W1] optimizer groups: [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,
                         'weight_decay': 0.01, 'decoupled_weight_decay': True, ...}]
[W1] configure_optimizers scheduler spec: [{'scheduler': 'ReduceLROnPlateau',
                                            'monitor': 'val_loss/dataloader_idx_0'}]
```
据此确认 6 件事：
1. **monitor 键解析正确**：`ModelCheckpoint`/`EarlyStopping` 的 `val_loss/dataloader_idx_0` 能命中，
   不会 `MisconfigurationException`（这是最容易坏的地方）。
2. **梯度确实流到对比头**（`spk_pool_proj.*` 有 1.5e-1 的位移）→ `query` 分支不是 no_grad。
3. **视频权重零更新**（`||delta_video||=0.0`）且**不在优化器里**（0 个交集）。
4. **调度器不滞后**（见 §2.4）。
5. **`accumulate_grad_batches` 语义正确**：4 batches / accum 2 → 2 step/epoch。
6. **`metrics.csv` 列名**与 §3.1 完全一致（含 `val_pit_sisnr_0near` 而无 1..4near）。

同一套代码的**无用参数**探针（`/root/tmp/w104/probe_c.py`，B=1 单次 forward+backward）：
```
[W1C] est (1, 1, 32000) sep_embs (1, 1, 112, 251, 129)
[W1C] pit 48.267066955566406 cl(K=0) 0.5368615984916687
[W1C] params: 522 | grad=None: 15 | grad all-zero: 1
[W1C] grad=None list:
    separator.audio_net.concat_layers.2.local_embedding.conv.weight
    ... (共 15 个，全是 concat_layers.2.*)
```
→ ① `find_unused_parameters=True` 是**必需**的（否则 DDP 直接报错）；
 ② 给既有 P0-7「`concat_layers[2]` 从未被调用」补上**梯度级证据**；
 ③ `PIT` 在随机估计下可达 **48.27**（= −48 dB SI-SNR），证明损失**没有下界**；
 ④ `K=0` 时 `NASSContrastiveLoss` 仍返回 α₊ 项（0.5369），**不等于 0**（与 `_mdcl_loss_hard` 的 K=0→0 不一致）。

### 4.3 实测 3：真实 DataModule 单 batch + 冻结 BN + hparams 安全（`/root/tmp/w104/probe_b.py`）

```
[W1a] video BN buffers: 40 ... n_changed: 0 / 40        ← 冻结生效
[W1a] vm.train() returns: FRCNNVideoModel | vm.eval() returns: FRCNNVideoModel   ← P1「eval() 返回 None」已修
[W1b] len(train DS)= 10000  len(val DS)= 10000  len(test DS)= 6000
[W1b] train loader: bs 4 drop_last True workers 8 pin True sampler RandomSampler steps/epoch 2500
[W1b] val loader:   bs 1 drop_last True steps 10000
[W1b] train batch: mix (4, 32000) tgt (4, 32000) mouth (4, 50, 88, 88) torch.float64 n_near/list: [1, 1, 1, 1]
[W1b] val batch: n_tuple 5 ... 5th: tensor([0]) <class 'torch.Tensor'>
[W1c] hparams.yaml has !!python tag: True | occurrences: 5
[W1c] yaml.safe_load FAILED: ConstructorError ... 'tag:yaml.org,2002:python/object/apply:torch._utils._rebuild_tensor_v2'
[W1c] conf.yml safe_load: OK
```
要点：① 训练集**每样本恰好 1 个 near**（K≡1，直接证据）；② val 的第 5 元素是 `tensor([0])`（`_meta.json` 缺失
→ n_near≡0）；③ `mouth` 是 **float64**，模块里靠 `.type_as(mixtures)` 转 float32（`:87`），
意味着 collate 阶段多搬一倍带宽；④ `hparams.yaml` 含 5 个 `!!python` 标签（`yaml.safe_load` 失败），
而 `conf.yml` 是纯 YAML（可安全读）——`config_to_hparams` 的 `torch.tensor(list)` 副作用（`av_litmodule.py:246-247`）。

### 4.4 实测 4：启动脚本 dry-run（`DRY_RUN=1`）

真 8 卡脚本在本机 2 卡上、`NPROC=2`：
```
torch: 2.8.0+cu128 | cuda: True | visible gpus: 2 | expected (NPROC): 2
[render] effective train_dir = /root/data/falcon_lrs2_2mix_paper/tr   (cv/tt 同)
[render] batch_size = 4 | seed = 42 | loss.train.sdr_type = pairwise_neg_sisdr | beta_anneal_epochs = 1000
global batch = batch_size x NPROC = 4 x 2 = 8 (x accum 1 = 8 effective)
optimizer steps/epoch ~= train_samples / global_batch = 10000 / 8 = 1250
data ok: tr 40000 / cv 10000 / tt 6000（mix+target+near 全部存在）
backbone ok: /root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar
[dry-run] would launch: … audio_train.py --conf_dir /root/tmp/conf_1442139.yml
```
→ 自检链（GPU 数 / 渲染 conf / 数据索引 / 骨干）**全部通过**；`DATA_ROOT` 真的被渲染进临时 conf。

同一个脚本换 `CONF=configs/LRS2_2Mix_Paper_2gpu.yml`：
```
[render] batch_size = 4 | seed = None | loss.train.sdr_type = pairwise_neg_snr | beta_anneal_epochs = 200
global batch = 4 x 2 = 8 (x accum 5 = 40 effective)
optimizer steps/epoch ~= train_samples / global_batch = 10000 / 8 = 1250     ← 错，真值 250
```
→ **两个问题**：① 打印公式 `samples // global_batch` 漏了 accum（`run_lrs2_paper_volcano_8gpu.sh:111-112`），
accum>1 时高估 5 倍；② 顺手暴露 `_2gpu.yml` 其实是 **legacy 配方（neg_snr / β200 / 无 seed）**，
并非 Table 16。

### 4.5 实测 5：真实训练产物复核（2026-10-23 的 2 卡 smoke）

`Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/`（3 epoch，已完成，`best_model.pth` 2,376,222 B，
每个 ckpt 51.9 MB）：
- `version_0/metrics.csv` 列：`epoch, learning_rate, lr, step, test_loss/dataloader_idx_1, test_pit_sisnr,
  train_beta_{step,epoch}, train_cos_sim_neg_{avg,max}, train_cos_sim_pos, train_loss_{step,epoch},
  train_n_samples_{0..4plus}near ×2, train_neg_active_ratio, train_pit_loss, train_spk_contrast_loss,
  val_loss/dataloader_idx_0, val_pit_sisnr, val_pit_sisnr_0near`
  → **只有 `_0near` 一列**（bucket 1–4 恒空，§1.3 的直接后果）。
- 数值：epoch0 `train_pit_loss=-1.7629`、`train_spk_contrast_loss=0.5004`、`cos_sim_pos=0.9840`、
  `cos_sim_neg_avg=0.9845`、`cos_sim_neg_max=0.9845`、`neg_active_ratio=1.0`、`train_n_samples_1near=4.0`；
  epoch0/1/2 的 `train_beta=0.0/0.005/0.01`（= epoch/200，验证 `_get_beta` 公式）。
- `best_k_models.json` = `{epoch=0: 0.6236, epoch=1: 0.2430, epoch=2: -0.1248}`（绝对路径，无占位符）。
- 每 epoch 优化步数 = **50** = 2000/(4×2×5) ✓（与 §4.4 的真值一致，反证脚本打印的 1250 是错的）。
- `cos_sim_pos≈0.984` 与 `cos_sim_neg≈0.9845` 几乎相等 → 初始化时投影头**近简并**，`margin=0.5` 远小于
  基线相似度 → `neg_active_ratio` 恒 1.0。

---

## 5. 坑与反直觉点

**P0-α｜β（τ）退火在 LRS2-2Mix 上完全不生效**
`nass_contrastive_loss.py:62-72`：`K==1` 走 `l_neg = α₋·relu(sim−m)`，**β 不进入计算**。
而 §1.3/§4.3 实测每个样本恰好 1 个 near → K≡1 → `beta_init/beta_final/beta_anneal_epochs`
（`configs/…v2.yml:115-117`）只体现在 `train_beta` 这条**日志曲线**上（真实 run：0/0.005/0.01），
**对梯度零影响**。论文 Table 16 的 τ 调度在 LRS2-2Mix 上不可复现；只有在 RealSSA-Sim（near 1–5 个）才可能有意义。
→ 任何"我们复现了 τ 调度"的表述都不成立。

**P0-β｜margin 0.5 语义失效**
实测 `cos_sim_neg_avg≈0.95–0.98`、`neg_active_ratio=1.0`（§4.5）→ `relu(sim−0.5)` 恒正，
负项等价于 `α₋·(cos(q,n)−0.5)`，margin 完全没有"门槛"作用，只是给 loss 加了一个常数偏移。
真正控制排斥强度的是 `loss_weight=1.0` 与 embedding 的几何。

**P1-α｜验证集每 epoch 全量跑一遍，成本 ≈ 训练**
`val_dataloader` 是 `[val_loader, test_loader]`（`av_litmodule.py:221-223`），`AVSpeechDataModuleWithNear`
把它们写死 `batch_size=1`（`avspeech_dataset_near.py:313,324`），长度 = 整个 cv/tt。
1000 epoch ⇒ **10,000,000 次 bs=1 验证前向**（cv）+ 6,000,000 次 test loader 迭代（其中 9/10 不做前向）。
而每 epoch 训练只有 10000 个窗口（8 卡 312 step）。→ 论文级 1000 epoch 的墙钟时间里，
**验证很可能比训练更贵**，且 `num_sanity_val_steps=0` 也不救。

**P1-β｜`validation_step` 里对同一批数据算两遍 loss**
`av_litmodule.py:95` 算 batch loss，`:108-120` 又对每个样本 `est[i:i+1]` 再算一次（为了分桶）。
bs=1 时这两次**完全等价**（多余 1 倍 loss 计算 + 1 倍显存驻留），且分桶结果因 `_meta.json` 缺失**全是桶 0**。

**P1-γ｜`train_samples` 决定的 epoch ≠ 遍历数据集**
`__len__` 返回 `samples`（`:147-150`），`__getitem__` **忽略 idx** 用 `random.randint` 取随机 mix 与随机起点
（`:154-156`）→ 1 epoch = 10000 次有放回随机抽样；池子 40000 ⇒ 期望唯一 mix ≈ 40000(1−e^{−0.25}) ≈ **8844（22%）**。
⇒ `epochs:1000` 总曝光 ≈ 10⁷ 窗口，但**没有"第 N 轮看完全部数据"的语义**；`patience=15`（epoch 级）
实际等价于"15×10⁴ 个窗口没进步"。分布式下 `idx` 被忽略还意味着 DistributedSampler 的分片**只控数量不控内容**。

**P1-δ｜`hparams.yaml` 不是安全 YAML**
`config_to_hparams`（`av_litmodule.py:230-248`）把 list 转 `torch.tensor`，PL 又把 hparams 用
`yaml.dump` 落盘 → 真实文件里 5 处 `!!python/object/apply:torch._utils._rebuild_tensor_v2` +
`!!python/tuple` + `OrderedDict`（§4.3）。`yaml.safe_load(hparams.yaml)` **直接抛 ConstructorError**。
`conf.yml` 是干净的（deepcopy 在模型构造前，`audio_train.py:277-289`）——**要读配置请读 conf.yml，不要读 hparams.yaml**。

**P1-ε｜`configure_optimizers` 的 dict 分支是颗地雷**
`av_litmodule.py:204-213`：对 dict 形式只 `setdefault` 了 `monitor`/`frequency`，**没有 `interval`**，
紧接着 `if sched["interval"] == "batch"` → 传 dict 必 `KeyError`。当前配置永远不会触发（传的是对象），
但如果有人按 PL 官方文档写成 `{"scheduler": ..., "interval": "epoch"}` 反而**能过**（因为有 interval 键），
写成 `{"scheduler": ...}` 就崩。

**P1-ζ｜DDP 条件化 `all_gather`**
见 §2.6。补充：这也是 `find_unused_parameters=True` 需要保留的原因（§4.2 的 15 个 grad=None 张量）；
把它"优化"掉，多卡会在第一次 backward 报 unused-parameter 错误。

**P2-α｜`Falcon.yml` 是"能解析不能跑"的模板**
`pretrain: pretrain_zoo/frcnn_128_512.backbone.pth.tar`（`:8`）——**该目录在仓库里不存在**（`[实测]` ls 失败）；
`train_dir/valid_dir/test_dir: --`（`:159-161`）。默认 `python audio_train.py` 会先死在数据（`open('--/mix.json')`），
再死在骨干（`frcnn_videomodel.py:96-101` 的清晰 `FileNotFoundError`）。默认值 `configs/Falcon.yml` 只有"模板"意义。

**P2-β｜`gpus: [0..9]` 在 2 卡机上静默降级**
`audio_train.py:150-163`：越界索引被裁到可见集合 + 一条 warning → `configs/Falcon.yml` 在本机等于 2 卡。
global batch 从 40 变 8 而**训练不会失败**。同理 `gpus: -1` 的"8 卡配方"在任何 <8 卡机器上都会静默改写配方
（`run_lrs2_paper_volcano_8gpu.sh:64-68` 只断言 `device_count == NPROC`，`NPROC` 又可以被环境变量改成 2）。`[实测]` §4.4。

**P2-γ｜`torch.cuda.empty_cache()` 在 import 期执行**（`audio_train.py:4`）——副作用式初始化 CUDA；
DDP 子进程 re-exec 时也会跑一遍。无功能问题，但在"多 rank 同时 import"时是多余的 CUDA 上下文触碰。

**P2-δ｜`PIT` 在 `n_src=1` 时是空壳**
`pit_wrapper.py:205` 对 1 个源做"最优排列" = 恒等 → `PITLossWrapper` 实际就是 `mean(-SI-SDR)`。
配置里写 `PITLossWrapper` 会让人误以为做了 2 源排列。

---

## 6. 与既有文档的冲突 / 更正

| 既有说法 | 出处 | 实际（本轮证据） |
|---|---|---|
| 「训练入口存在已知 P0（Comet logger、conf.yml 类名等）」 | `AGENTS.md:91-92`（2026-10-24） | **P0-1 已在 `ac91800` 修**：`a7af472:170` 的 `logger=comet_logger` 全文件仅出现 1 次（未定义）→ 建 `Trainer` 即 `NameError`；当前 `audio_train.py:390` 已改为 `build_logger(config, exp_dir)`。AGENTS.md 未更新（11 天滞后）。「conf.yml 类名」**仍然坏**（见下一行）。 |
| 「`Experiments/checkpoint/Falcon/conf.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 未导出 → 评测 AttributeError」 | `FINAL_PROJECT_UNDERSTANDING.md:471`（P0-3） | **半修**：快照文件仍在（`git ls-files` 仍跟踪），但 `audio_visual_test.py:288-293` 已把它变成**带修复指引的 AttributeError**。即"坏但可诊断"，不是"坏且莫名其妙"。 |
| 「`av_litmodule.py:153-179` 硬依赖 Comet `logger.experiment.log(dict)`」 | 同上 P0-2 | **已修**：当前 `av_litmodule.py:154-186` 用 `log_dict`，注释写明原因；`[实测]` CSVLogger 下 `learning_rate/val_pit_sisnr*` 全部正常落盘。 |
| 「DDP per-bucket `all_gather` 可能死锁」 | 旧 03 文档 §V15、`FINAL…:488` | **确认未修**（`av_litmodule.py:170-176`）；`[实测]` 但因 `_meta.json` 全缺、桶集合对称而**不会触发**。补数据后才会暴露。 |
| 「RealSSA-Sim 的 near 负样本现为 K=0」 | 旧 03 文档 §0 | **对 LRS2-2Mix 不成立**：`falcon_lrs2_2mix_paper/*/near.json` 存在且每样本 1 条、指向 `s2/` 的 clean 源（§1.3）。K=0 只适用于 RealSSA 发布包。 |
| 「2×bs4×accum5 = 有效 40，与论文对齐」 | `run_lrs2_paper_2gpu_smoke.sh:2` | **错**：论文 global 是 **32**（4×8），40 是 LEGACY 8gpu（5×8）的值。2 卡对齐 Table 16 应 `accum=4`（4×2×4=32）。 |
| 「1 epoch = 每 rank 250 次随机抽样」 | 旧 03 文档 §V15 | **按 bs 而定**：单卡 bs4 → 2500 step；2 卡 accum5 → 50 step（真实 run 实测）；8 卡 accum1 → 312 step。真正的定义是 `train_samples/(bs×nGPU×accum)`。 |
| 「`v2` 配置落地了论文 Table 16」 | `docs/STATUS.md:53,77`、`PHASE2_VOLCANO_LRS2_TRAINING.md:8` | **配置层落地，机制层未落地**：τ 在 K=1 下无效（§5-P0-α）、margin 失效（§5-P0-β）、硬件 8×3090 不可得、epoch 定义无对齐依据（§3.4）。 |
| 「训练/评测的随机种子未见全局 seed」 | `FINAL…:521`（Q12） | **已支持**：`audio_train.py:256-266` + `v2.yml:145` 的 `seed: 42`（`pl.seed_everything(…, workers=True)`）。其余 5 个 config 仍未设种子。 |
| 「`FRCNNVideoModel.eval()` 返回 None / BN 统计在训练中漂移」 | `FINAL…:484`、`FINAL…:488` | **已修**：`frcnn_videomodel.py:113-125` 重写 `train()` → 返回 `self`；`[实测]` 40 个 running buffer **0 漂移**。 |
| 「`audio_train.py` 417 行」 | 旧 03 文档 §1.1 | 现在 **456 行**（`2062528` 加入 v2/seed/logger 相关）。 |

---

## 7. GAP LIST（可被下一轮直接认领）

| # | 问题 | 为什么重要 | 建议做法 |
|---|---|---|---|
| G1 | **β 在 K=1 下无效这件事，作者是否知情？** | 决定"τ 调度"是否论文贡献的一部分，直接影响消融复现 | 查论文 §4.3 Eq.8-10 的 softmax 是否对 K=1 有定义；查 RealSSA 场景 near 数（1–5）→ 只有那里能用 |
| G2 | `margin=0.5` 是在什么 embedding 尺度下定标的？ | 实测 cos≈0.95 基线使 margin 失效，若论文用同一实现则论文的对比损失也基本退化为余弦排斥 | 需要在 RealSSA 数据上测 `cos_sim_pos/neg` 分布；本仓库只有 LRS2 的 3 epoch 证据 |
| G3 | **8 卡真实 DDP 未验证** | `find_unused_parameters`、条件化 `all_gather`、`sync_dist` 都只在单进程下测过 | 需 8 卡节点（火山/2×A800 不够）跑 ≥2 epoch，重点看是否 hang、`train_loss` 是否=各 rank 均值 |
| G4 | `1000 epoch × 10⁴ val` 的实际墙钟 | 决定论文级复现是否可行（可能验证比训练贵） | 在 2 卡上测 1 个 epoch 的 train/val 时间比；必要时把 val 改成子集+`limit_val_batches`（需改代码，当前未暴露） |
| G5 | `train_samples=10000` 是否是刻意（22% 覆盖/epoch） | 直接决定"1000 epoch"到底等于多少 data passes | 论文未披露 epoch 定义；需作者确认或按「data passes」重新标定 |
| G6 | `concat_layers.2` 的 5,568 个无用参数是否为 bug | 影响 0.5M 参数口径与容量论证（已有梯度级证据 §4.2） | 需作者确认；若是有意（level-2 只用两层），应删代码而不是留着吃 `find_unused_parameters` 的性能税 |
| G7 | val 分桶指标是否曾被真正读过 | `_meta.json` 全缺 ⇒ 分桶永远是桶 0，等于白写 40 行代码 | 要么生成 `_meta.json`（`generate_near_json.py` 顺带产出），要么删掉分桶逻辑 |
| G8 | `hparams.yaml` 的 `!!python` 标签是否影响 `load_from_checkpoint` | 影响"从 ckpt 恢复"的可移植性（含 torch 版本绑定） | 用 `yaml.load(..., Loader=UnsafeLoader)` 的替代：在 `config_to_hparams` 里把 list 转成 `str(list)` |
| G9 | `2gpu` 配置的 accum 该是 4 还是 5 | 决定 2 卡实验是否与论文 global 32 可比 | 若要 32 → accum 4；若要沿用 legacy 40 → 改文档措辞（脚本注释现在是错的） |
| G10 | Studio 复用训练 `audionet_config` 的耦合风险 | 改训练结构会静默改变线上模型 | 需要一次"Studio 侧 config 快照 vs 训练 config"的契约测试（当前 `space/tests/test_static.py:41` 只测拷贝，不测语义） |

---

## 8. 证据清单

### 8.1 读过的文件（HEAD `6c1d5b7`）

```
audio_train.py (456)                                  look2hear/system/{__init__,av_litmodule,av_litmodule_nass_v2,
look2hear/utils/{parser_utils,lightning_utils}.py      av_litmodule_mdcl,optimizers,schedulers}.py
look2hear/datas/{avspeech_dataset,avspeech_dataset_near}.py
look2hear/losses/{__init__,matrix,pit_wrapper,nass_contrastive_loss}.py
look2hear/models/{__init__,Falcon,Falcon_Contrastive,base_model}.py
look2hear/videomodels/frcnn_videomodel.py
configs/{Falcon,LRS2_2Mix,LRS2_2Mix_Paper_2gpu,LRS2_2Mix_Paper_2gpu_smoke,
         LRS2_2Mix_Paper_8gpu,LRS2_2Mix_Paper_v2}.yml
tools/repro/phase2/{run_lrs2_paper_volcano_8gpu,run_lrs2_paper_2gpu_smoke,run_p2d_train}.sh
tools/repro/phase2/{LRS2_2Mix_1gpu,smoke_lrs2,smoke_lrs2_bs8}.yml
studio/{falcon_runtime.py,configs/Falcon.yml,configs/FalconBig.yml}   space/{build.py,tests/test_static.py}
AGENTS.md  docs/understanding/{FINAL_PROJECT_UNDERSTANDING.md,PAPER_ALIGNMENT_20261030.md}
docs/understanding/deepdive-20261024/03-training-system.md  docs/STATUS.md  docs/PHASE2_VOLCANO_LRS2_TRAINING.md
Experiments/checkpoint/{Falcon/conf.yml,Falcon/best_k_models.json,
                        LRS2_2Mix_Paper_2gpu_smoke/{conf.yml,best_k_models.json,version_0/{hparams.yaml,metrics.csv}}}
```

### 8.2 本轮执行过的命令（全部只读 + CPU）

```bash
# 0) 基线与依赖
ssh dev_a800_2gpus 'cd /root/code/github_repos/Falcon-fork && git log --oneline -3 && wc -l …'
ssh dev_a800_2gpus '/root/miniforge3/envs/falcon/bin/python -c "import torch_optimizer,comet_ml,…"'
ssh dev_a800_2gpus 'git show a7af472:audio_train.py | grep -n comet'      # P0-1 证据
ssh dev_a800_2gpus 'git show ${c}:audio_train.py | grep -n logger='      # ac91800..2062528 修复轨迹

# 1) 数据事实
/root/miniforge3/envs/falcon/bin/python  # 统计 mix.json 长度 / near.json 直方图 / _meta.json 存在率

# 2) 实测 1：配置解析（6 个 yml round-trip）
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python /root/tmp/w104/cfg.py

# 3) 实测 2：CPU 2-epoch 冒烟 + grad=None 探针
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD …/python -u /root/tmp/w104/smoke_fit.py    # 日志 /root/tmp/w104/smoke.log
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD …/python -u /root/tmp/w104/probe_c.py

# 4) 实测 3：真实 DataModule 单批 + BN 冻结 + hparams 安全
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD …/python -u /root/tmp/w104/probe_b.py

# 5) 实测 4：启动脚本 dry-run（不启动训练、不占显存）
DRY_RUN=1 NPROC=2 LOG_DIR=/root/tmp/w104/dryrun  bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh
DRY_RUN=1 NPROC=2 CONF=configs/LRS2_2Mix_Paper_2gpu.yml LOG_DIR=/root/tmp/w104/dryrun2 \
  bash tools/repro/phase2/run_lrs2_paper_volcano_8gpu.sh

# 6) 实测 5：真实 run 产物
cat Experiments/checkpoint/LRS2_2Mix_Paper_2gpu_smoke/version_0/metrics.csv
```

### 8.3 本轮产出的临时脚本（仓库外，可复跑）

| 文件 | 作用 |
|---|---|
| `/root/tmp/w104/cfg.py` | 6 个 yml 的 argparse 回填 + round-trip 校验 |
| `/root/tmp/w104/smoke_fit.py` | CPU 2 epoch 冒烟（真 config/真装配/假 batch），打印 step 计数、scheduler 收到的 monitor 值、参数位移、metrics.csv |
| `/root/tmp/w104/probe_b.py` | FRCNN BN 冻结探针 + 真实 DataModule 单批 + `hparams.yaml` 安全性 |
| `/root/tmp/w104/probe_c.py` | 单次 forward/backward 的 `grad=None` 清单（DDP unused-parameters 证据） |
| `/root/tmp/w104/smoke.log` | 冒烟完整 stdout/stderr |
| `/root/tmp/w104/ckpt/`、`/root/tmp/w104/logs/` | 冒烟产出的 ckpt 与 CSVLogger 目录 |

> 本轮**未做**：真实 GPU 训练（本轮不占卡）、8 卡 DDP、1000 epoch、`audio_train.py` 端到端（10000 条/epoch 的
> val 在 CPU 上不可行）；这些都在 §7 GAP LIST 里。
