# 评审：Phase-1 训练栈改动（T1 `audio_train.py` + T2 `look2hear/system/`）

> 评审人：`studio-backend`（task-16，**独立评审**，未参与实现、未修改源码）
> 分支：`repro/phase1` @ 基线 `689555c`（未 commit）。评审对象（评审时 sha256）：
> `audio_train.py` = `4857c2d6…ab89f39`；`look2hear/system/av_litmodule.py` = `35d76d3f…18558e3`
> （`look2hear/system/` 下其余文件无改动；`git diff main` 的其它文件属于 T3/T4/T5，不在本评审范围）。
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / PL 2.6.6 / 2×A800）。
> 独立验证脚本：`docs/verification/phase1/review_training_stack_checks.py`
> 日志：`docs/verification/phase1/REVIEW-training-stack-checks.log`（**74 项检查全部 PASS**，含 2 个 smoke fit、2-GPU toy DDP、真实权重指纹）
> 参考实现者报告：`T1-report.md`、`T2-report.md`（评审时已同步到最终代码，未发现与实现不符的结论）。
> 声明：评审人只写 `docs/verification/phase1/`（本文件 + 检查脚本/日志 + `_ddp_smoke_toy.py`），未改源码、未 commit。
> **增量复评（2026-10-21 17:06 UTC）：见文末 §7** —— `audio_train.py` 已由实现者按 P2 反馈更新为 409 行
> （sha256 `71cb2bfb…5b155f`），§7 含 P2 闭环证据与重跑结果（T2 文件 hash 未变）。

## 1. 结论

**通过（T1/T2 可合入 Phase-1，满足 Gate G1 的训练栈部分）。P0 = 0、P1 = 0、P2 = 2、P3 = 4。**

- **F1（comet_logger 未定义）**：已彻底修复。`audio_train.py` 不再引用 `comet_logger`；新增 `build_logger()` 按
  `training.logger` 构造 `CSVLogger`（默认）/`TensorBoardLogger`/`CometLogger`/无 logger，缺可选依赖时给出清晰
  `RuntimeError`；`logger: none` 时向 Trainer 传 **`False`**（而不是 `None`，避免 PL2 自动创建默认 logger）。
  我独立验证：symtable 无未定义全局名、真实 `pytest` 13 项通过、真实 smoke 训练 CSV 落盘。
- **F6（PL 2.x Trainer）**：已修复。去掉 `gpus=`/`accelerator=None`，改为 `devices`/`accelerator`/`strategy`
  （`audio_train.py:60-113,305-318`）；CPU / 单卡 / 多卡 / 10 卡配置在 2 卡机自动收敛到 `[0,1]+DDP`。
  我独立验证：8 组 resolver 矩阵全对；真实 2-GPU **toy DDP fit 成功**；真实 smoke 单卡 GPU fit 成功。
- **F9（conf dump 被 `Separator.pop` 污染）**：已修复。conf 快照前移到建模之前并用 `copy.deepcopy`
  （`audio_train.py:185-192` 先于 `:205` 建模）。我独立验证：**建模时 conf 已存在**；落盘 conf 保留
  `audio_params/video_params` 的 `repeats`/`shared`、无注入的 `causal`；同时运行期 config 仍按原语义被污染
  （正/反双向证据）。
- **F2（litmodule 的 Comet-only `self.logger.experiment.log`）**：已修复。`on_validation_epoch_end`
  改为 `metrics` 字典 + `self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)`
  （`look2hear/system/av_litmodule.py:142-188`，`log_dict` 在 `:186`）；`learning_rate/val_pit_sisnr/
  val_pit_sisnr_*near/test_pit_sisnr` 键全部保留，`"epoch"` 由 PL 原生 epoch 列提供（CSVLogger 有 `epoch` 列）。
  三个 litmodule 中只有基类含 Comet 调用（4 处），NASSv2/MDCL 继承同一实现，故基类修复即覆盖三者；
  我独立静态确认 `look2hear/system/*.py` 无可执行 `logger.experiment`，并用真实模型 + 真实 Trainer 跑通三模块。
- **数值/训练逻辑变化**：**未发现任何变化**。T2 的 diff hunk 全部落在 `on_validation_epoch_end` 内；三模块的
  `training_step`/`validation_step` 源码在 main 与新代码间**逐字节等价**（AST unparse 比较）；T1 中
  `model/video_model/optimizer/loss_func/system/checkpoint` 的构造语句与 main **逐字节等价**；真实权重
  eval/train/video 前向 sha256 与 `baseline-fingerprint.json`/`T5-post-fix-fingerprint.json` **完全相同**。

### 需修（非阻塞，建议 Phase-1 收尾前顺手修）

| 级别 | # | 问题 | 位置 | 证据 | 修法 |
|---|---|---|---|---|---|
| P2-1 | 单元素**索引列表**被当成"卡数"：`gpus: [1]` 期望选 GPU 1，实际 `devices=1` → 用 GPU 0；`gpus: [7]` → `devices=2`（用 0,1） | `audio_train.py:48-57`（`_as_int_list` 丢失 int/list 原类型）+ `:91-107`（`len(requested)==1` 分支当 count） | 独立实测：`[1] -> ('gpu',1,1,'auto')`、`[7] -> ('gpu',2,2,DDP)`；仓库默认 `[0..9]` 不受影响 | 保留原类型：**list/tuple 一律视为索引**、**scalar int 视为卡数**（并映射 legacy `0`→CPU 或报错、`-1`→全部可见卡）；补 `[1]/[7]` 单测 |
| P2-2 | `gpus: []` → `devices=[]`，`num_devices=0`，把非法配置推迟到 PL 抛 `MisconfigurationException`（信息不友好） | `audio_train.py:91-107` | 独立实测：`gpus=[] -> ('gpu',[],0,'auto')`；`Trainer(devices=[]) -> MisconfigurationException: ... is not a valid input` | resolver 内校验：空列表 → 明确 `ValueError`（或回退默认单卡）；重复索引 `[0,0]` 同样校验 |
| P3-1 | legacy 标量语义变化：PL1 的 `gpus: 0`（=CPU）/`gpus: -1`（=全部 GPU）现在都映射为 GPU0 | `audio_train.py:74-79` | 独立实测 `0 -> ('gpu',[0],1,'auto')`、`-1 -> ('gpu',[0],1,'auto')` | 仓库配置均为列表，不影响 Gate；在 docstring/REPRODUCTION_PLAN 注明新语义，或显式兼容旧值 |
| P3-2 | 新配置键 `training.logger`（以及 `training.comet_project`）未在 `configs/Falcon.yml` 或 Phase-1 文档中登记，默认值只能从代码发现 | `audio_train.py:122-131`；`configs/Falcon.yml:129-138` 无 `logger` 键 | 读码 | 在 `configs/Falcon.yml` 的 `training:` 下加注释示例（`# logger: csv  # csv / tensorboard / comet / none`），并在 Phase-1 README 记录 |
| P3-3 | CSVLogger 目录嵌套重复：`Experiments/checkpoint/<exp>/<exp>/version_N/metrics.csv`（exp_dir 已含 exp_name） | `audio_train.py:137`（`CSVLogger(save_dir=exp_dir, name=exp_name)`） | 独立 smoke 实测路径 `.../phase1_review_smoke/phase1_review_smoke/version_0/metrics.csv` | 改 `name="logs"` 或 `save_dir=os.path.dirname(exp_dir)`；纯路径美观问题 |
| P3-4 | 文档小漂移：T1 报告的行号引用相对最终代码偏移约 +2~+5 行（如 `build_logger` 实为 `:115-177`、Trainer 实为 `:305-318`）；`--conf_dir` 默认值仍指向不存在的旧配置（报告已列为遗留） | `T1-report.md` §2；`audio_train.py:31-35` | 读码对照 | 评审不要求改；建议 T1 收尾时把默认值改为 `configs/Falcon.yml`（复现 UX）并同步行号 |

> 另：`audio_train.py` 仍有未使用 `import sys`（基线与当前一致，非本次引入，P3 忽略）。
> 说明：P2-1/P2-2 只影响"手工挑 GPU/空列表"的非默认配置；**仓库默认 `gpus: [0..9]` 在 2 卡机与
> 单卡机上的行为均正确**，因此不阻塞 Phase-1 Gate。

## 2. 独立验证方法与结果（与 T1/T2 报告互相独立）

### 2.1 静态检查（`review_training_stack_checks.py` §A，全部 PASS）

| 检查 | 结果 |
|---|---|
| symtable 未定义全局名：`audio_train.py` + 3 个 litmodule | 全部 **无**（`comet_logger` 已消除） |
| AST：`audio_train.py` 中 `comet_logger` Name 节点 | 0 个 |
| AST：`pl.Trainer(...)` 仅 1 处，kwargs = 11 个 PL2 合法键，**无 `gpus=`** | 通过 |
| `look2hear/system/*.py` 可执行 `logger.experiment` | 0 处（仅注释） |
| T2 diff hunk 范围 | `[(153,12),(175,2),(184,3)]` ⊆ `on_validation_epoch_end` 行域 `(142,188)` |
| `training_step`/`validation_step` 源码与 main 对比（AST unparse） | 三个模块全部**逐字节相同** |
| `audio_train.py` 中 `model/video_model/optimizer/loss_func/system/checkpoint` 构造语句 vs main（AST unparse） | 全部**逐字节相同**（训练数值路径未被触碰） |
| T2 `log_dict` 参数 | `on_step=False, on_epoch=True, logger=True` ✓；metric keys ⊇ `learning_rate/val_pit_sisnr/test_pit_sisnr` ✓ |

### 2.2 Logger 与 PL2 语义（§B，全部 PASS）

- `build_logger`：默认/`csv` → `CSVLogger`（save_dir=exp_dir）；`tensorboard`（本环境无 `tensorboard` 包）→
  清晰 `RuntimeError` 提示安装；`comet`（无 `comet-ml`）→ 清晰 `RuntimeError`；`none/null/false/None/False` → `None`；
  `wandb/True/3` → `ValueError`。
- **PL2 语义确认**：`Trainer(logger=None)` 会**创建默认 CSVLogger**；`Trainer(logger=False)` 才无 logger。
  T1 的 `logger if logger is not None else False`（`audio_train.py:315`）正确避开了该坑（这是 v1 修复过一次的真实问题）。
- resolver 矩阵（monkeypatch CUDA 可见性）：CPU→`("cpu",1,1,"auto")`；2 卡 + `[0..9]`→`("gpu",[0,1],2,DDP)`；
  `[0]`→`("gpu",[0],1,"auto")`；`8`→`("gpu",2,2,DDP)`；`"0,1"`→`[0,1],DDP`；`"1"`→`1,"auto"`；`None`→`[0],1`；
  真实 2×A800 主机 → `("gpu",[0,1],2,DDPStrategy)`。

### 2.3 `main()` 装配与 F9（§C，全部 PASS）

用 fake DataModule/Video/Trainer + **真实 audio model** 运行 `audio_train.main()`（不训练）：
- 建模回调记录 `conf.yml` **在模型构造时已存在** → dump 顺序前移生效；
- 落盘 conf：`audio_params/video_params` 均含 `repeats`/`shared`，`audio_params` **无** `causal`，`main_args.exp_dir` 正确；
- 同一进程运行期 config 被污染（`repeats` 消失、`causal` 注入）→ 正/反双向证明"dump 的是 deepcopy"；
- Trainer kwargs：`CSVLogger`、`accelerator=gpu`、`devices=[0,1]`、`strategy=DDPStrategy`、`limit_train_batches=1.0`、
  `gradient_clip_val=5.0`、`sync_batchnorm=True`、`num_sanity_val_steps=0`、无 `gpus`；
- `training.logger: none` → Trainer 收到 **`False`**。

### 2.4 真实 smoke 训练（§D/E，Gate G1-2 的训练栈部分）

`audio_train.main()` + 真实 `FalconSpeakerContrastive`（官方 `configs/Falcon.yml`）+ fake 视觉前端/fake 批次
（2 train batches / 1 val / 1 test，32000 样本），**单卡 GPU + CSVLogger**：

```text
True   smoke: training completed without exception
True   smoke: global_step == 2 (2 train batches)
True   smoke: conf.yml + best_k_models.json + best_model.pth + *.ckpt written
True   smoke: CSVLogger produced metrics.csv
True   smoke: metrics.csv preserves baseline keys (+bucket, +train keys)
True   smoke: learning_rate value == optimizer lr (1e-3)          # 0.0010000000474974513
True   smoke: val_pit_sisnr == -val_loss (semantics preserved)    # 评审本次运行：63.764671325683594 / -63.764671325683594（绝对值随随机初始化而变，关系恒成立）
True   smoke: no NaN/Inf in logged numeric cells
True   smoke: epoch-end metrics logged once (on_epoch=True semantics)
True   F9: smoke dumped conf keeps repeats/shared, no causal
True   smoke: best_k_models.json 值为 float 且路径均存在（epoch=0.ckpt）
```

**无 logger smoke**（`training.logger: none`）：训练完成、Trainer `logger=False`、**不产生 metrics.csv**、
checkpoint/best_model.pth 仍正常导出 → F2 的"无 logger 不崩"在真实模型/真实 Trainer 下成立。

### 2.5 数值/训练逻辑不变（§F，全部 PASS）

| 指纹 | 结果 |
|---|---|
| eval forward sha256 / mean vs `baseline-fingerprint.json` | `ad0f20a0…f40757` / `9.9872559e-06` **完全一致** |
| train mode (est, sep) sha256 vs baseline | `5f01a596…e7545a` / `8a8e5c2a…bddc16` **完全一致** |
| video forward（T5 协议，seed=0）vs `T5-post-fix-fingerprint.json` | `1b7255f3…ff3f` **完全一致**（T5 pre/post 两份也相同） |

### 2.6 多卡语义（§G）

- resolver 在真实 2×A800 主机得到 `("gpu",[0,1],2,DDPStrategy(find_unused_parameters=True))`；
- 用等价 kwargs 的 **toy 2-GPU DDP fit**（`_ddp_smoke_toy.py`，CSVLogger、`sync_batchnorm=True`）**成功**；
- 真实 Falcon 2-GPU 训练未运行（见 §4 局限）；PL2 参数合法性已由 toy DDP + 真实单卡 smoke 共同覆盖。

### 2.7 回归测试（评审人复跑）

```bash
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audio_train_entry.py tools/repro/tests/test_litmodule_logging.py -q
# -> 18 passed (13 T1 + 5 T2)
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q      # 全量共享测试（含其它任务）
# -> 70 passed in 60.73s
/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q           # Gate G1-4
# -> 35 passed, 2 skipped in 17.62s
```

## 3. 逐项验收对照（Gate G1 训练栈部分）

| 验收点 | 状态 | 证据 |
|---|---|---|
| F1 `audio_train.py` 可在无 comet 环境启动 | ✅ | §2.2/§2.4；`build_logger` 全分支 |
| F6 PL 2.6 Trainer 参数合法（单卡/多卡/CPU） | ✅ | §2.2/§2.6；toy DDP + 真实单卡 fit |
| F9 conf dump 保留 `repeats`/`shared` 且未被建模污染 | ✅ | §2.3/§2.4（正/反双向） |
| F2 三 litmodule 在 CSVLogger/无 logger 下不崩、key 语义保留 | ✅ | §2.1/§2.4；CSV 列与数值一致性 |
| 不改训练数值/逻辑 | ✅ | §2.1（源码逐字节等价）+ §2.5（指纹全等） |
| 冒烟训练可保存 checkpoint 且 loss 有数值 | ✅ | §2.4（2 steps、checkpoint/best_model/CSV） |

## 4. 局限与未覆盖（诚实声明）

1. **真实 Falcon 2-GPU DDP 训练未运行**：只做了 toy DDP fit + 真实单卡 smoke。F15（per-bucket
   `all_gather` 在 DDP 下可能死锁）属另一个已知风险、与本改动无关；如集成阶段要跑 2 卡，建议先小步跑
   `av_litmodule` 的 validation 路径观察是否挂起。
2. **TensorBoard/Comet 成功路径未实测**：环境未安装 `tensorboard`/`comet-ml`，只验证了缺包时的清晰报错；
   成功构造/落盘需在装包环境补测（T1 报告同样声明）。
3. **smoke 用的是 fake 视觉前端与 fake 批次**（为了不依赖 LRS 数据/backbone）；真实视觉前端前向已由
   §2.5 指纹覆盖，真实数据链路由 T4/T3 的合成数据 + 集成 Gate 覆盖。
4. 真实数据训练收敛性/速度不在本评审范围（Phase-1 Gate 只要求冒烟跑通）。
5. 评审时点 `pytest tools/repro/tests -q` 已 70 passed（此前的 T3 metrics 失败已被其作者修复）；T3/T4/T5/T6
   的改动与报告不在本评审范围（其文件 diff 未评审）。

## 5. 复跑命令（评审全过程可复现）

```bash
cd /root/code/github_repos/Falcon-fork
# 1) 完整独立评审脚本（74 项；含 2 次真实 smoke 训练 + 2-GPU toy DDP + 真实权重指纹，约 90 s）
/root/miniforge3/envs/falcon/bin/python docs/verification/phase1/review_training_stack_checks.py
#    -> RESULT: 74 checks | PASS=74 WARN=0 FAIL=0
#    -> 日志 docs/verification/phase1/REVIEW-training-stack-checks.log

# 2) 实现者回归测试
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audio_train_entry.py \
    tools/repro/tests/test_litmodule_logging.py -q                 # 18 passed
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q  # 70 passed
/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q       # 35 passed, 2 skipped

# 3) 独立重跑 smoke 训练（脚本内自动建临时目录，不污染仓库；结果路径打印在 stdout）
#    见 review_training_stack_checks.py 的 §D/§E 输出；关键检查项见 §2.4。
```

## 6. 评审结论与给 Lead 的建议

- **建议合入**：T1/T2 已把 Phase-1 训练栈从"必然崩溃"修到"真实模型 2-step smoke + CSV 落盘 + checkpoint
  导出全部通过"，且经独立验证**未引入任何数值/训练逻辑变化**。
- **P2-1/P2-2 建议修**（各 ~10 行）：在 `_as_int_list`/`resolve_accelerator_and_devices` 中保留
  list-vs-scalar 类型语义并校验空/重复设备列表；若本轮不修，请在 Phase-1 文档中写清
  "单卡请用 `gpus: [0]`/scalar 卡数；选非 0 号卡请用 `CUDA_VISIBLE_DEVICES` 或 ≥2 元素列表"。
- **P3-2 建议修**：在 `configs/Falcon.yml` 的 `training:` 增加注释形式的 `logger: csv` 示例，
  让 `tensorboard/comet/none` 可发现（不改变默认行为）。
- T2 报告已覆盖 F2 的三模块口径（基类 4 处调用，两个子类继承）并在报告中注明，与实现一致。
- 本评审未发现需要回退/重做的 P0/P1；不阻塞 Gate G1。

---

## 7. 增量复评（P2 闭环，2026-10-21 17:06 UTC）

> 触发：train-sys 按上轮 P2 反馈修改 `audio_train.py`（**409 行**，sha256
> `71cb2bfb877102cc003f0f04e89309f8b780b289dbd5ee0d525674a4695b155f`）；
> `look2hear/system/av_litmodule.py` **未变化**（sha256 `35d76d3f…b18558e3`，仍为上轮已评版本）。
> 方法：只读 `git diff main -- audio_train.py` + 重跑独立脚本
> `review_training_stack_checks.py`（脚本已按新语义扩充，**105 项检查**；日志
> `REVIEW-training-stack-checks.log`）+ `pytest tools/repro/tests/test_audio_train_entry.py`
> + T1/T2 联合测试 + 真实权重指纹。未改源码、未 commit。

### 7.1 增量结论

**通过（P2 已闭环，无新 P0/P1）。** 上轮 **P2-1、P2-2 全部修复**，P3-3（CSV 目录）、P3-4（`--conf_dir`
默认值）也已修复；P3-1（legacy 语义）按新契约显式实现并告警。独立脚本 **105 项检查：PASS=104、WARN=1
（唯一 WARN 为文档承诺未落地，见 7.4-R1）、FAIL=0**；T1 测试 **20 passed**（上轮 13），T1+T2 **25 passed**；
指纹与上轮/基线完全一致；真实 2-step smoke 与 2-GPU toy DDP 均通过。

### 7.2 P2/P3 闭环核对（逐条证据）

| 上轮条目 | 新实现 | 我的独立复核 |
|---|---|---|
| **P2-1** int-vs-list 语义丢失 | 新增 `_normalize_gpus_spec()`（`audio_train.py:51-90`）：`None/""`→none；list/tuple/range/逗号串→**indices**；scalar >0→**count**；`0`→**cpu**；`-1`→**all**；bool/float/dict→ValueError。`resolve_accelerator_and_devices()`（`:93-161`）按 kind 分派 | 12 组 `_normalize_gpus_spec` 矩阵全对；resolver 实测：`[1]`→`("gpu",[1],1,auto)`（**不再误用 GPU0**）、`"1"`→`[1]`、`8`→`("gpu",2,2,DDP)`、`0`→cpu、`-1`→`[0,1],DDP`；✅ 闭环 |
| **P2-2** 非法设备配置延迟到 PL 才报错 | resolver 内校验：空列表 / 重复索引 / 全部越界 → 清晰 `ValueError`（含可见卡提示）；部分越界 → 过滤+告警（这是 10 卡默认配置在 2 卡机可用的必要条件） | 实测 `[]`→“empty list...”、`[0,0]`→“duplicate GPU indices”、`[7]`→“does not match any visible GPU (visible: [0,1])”、`"a,b"`→“not a comma-separated integer list”；`[0,7]`→`[0]`+warning；✅ 闭环（且不再出现 `Trainer(devices=[])` 的 PL `MisconfigurationException`） |
| **P3-1** legacy `0`/`-1` 语义 | 显式实现：`0`→CPU（打印 legacy 告警）、`-1`→全部可见卡 | 实测 `0`→`("cpu",1,1,auto)`、`-1`→`("gpu",[0,1],2,DDP)`；✅ 闭环 |
| **P3-3** CSV 目录重复 | `CSVLogger(save_dir=os.path.dirname(os.path.abspath(exp_dir)), name=exp_name)`（`:188`） | 真实 smoke：`metrics.csv` 路径 = `<exp_dir>/version_0/metrics.csv`（**单层**，无 `<exp>/<exp>`）；✅ 闭环 |
| **P3-4** `--conf_dir` 旧默认值 | 默认改为 `configs/Falcon.yml`（`:35`），help 说明旧配置不随仓库发布 | AST/正则检查 `default="configs/Falcon.yml"`；✅ 闭环 |
| **P3-2** `training.logger` 文档 | ❌ 未修（configs/Falcon.yml 仍未登记；见 7.4-R1） | WARN |

### 7.3 重跑验证记录（本轮）

```bash
/root/miniforge3/envs/falcon/bin/python docs/verification/phase1/review_training_stack_checks.py
#   -> RESULT: 105 checks | PASS=104 WARN=1 FAIL=0
#      （含：真实 2-step smoke、no-logger smoke、2-GPU toy DDP、真实权重指纹、34 个
#        _normalize_gpus_spec/resolver 用例、静态 diff-scope/源码等价性检查）
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audio_train_entry.py -q
#   -> 20 passed（上轮 13；新增 _normalize_gpus_spec / 单索引 / 越界/空/重复 / legacy / 字符串索引等用例）
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audio_train_entry.py \
    tools/repro/tests/test_litmodule_logging.py -q
#   -> 25 passed
```

关键运行时复核（与上轮同口径）：

| 项 | 结果 |
|---|---|
| 真实模型 2-step smoke（CSVLogger） | 通过；`global_step=2`；`<exp>/version_0/metrics.csv`（单层）；`learning_rate=1e-3`、`val_pit_sisnr == -val_loss`、无 NaN/Inf、epoch 行唯一；ckpt/best_k_models/best_model.pth 齐全 |
| no-logger smoke（`training.logger: none`） | 通过；Trainer 收到 `False`；不写 CSV；checkpoint/best_model 正常 |
| 2-GPU toy DDP（resolved `DDPStrategy`+`sync_batchnorm`） | `DDP TOY FIT OK` |
| F9 dump 顺序/内容 | 不变：建模前落盘、deepcopy 保留 `repeats/shared` 且无 `causal`；运行期 config 仍被污染 |
| 数值指纹 | eval `ad0f20a0…f40757`、train est `5f01a596…e7545a` / sep `8a8e5c2a…bddc16`、video `1b7255f3…ff3f` —— 与 baseline/T5 **完全一致** |
| 静态不含未定义名 / 无 `logger.experiment` / T2 函数源不变 | 全部通过（与上轮同） |

### 7.4 残留问题（P3，不阻塞；本轮仅新增 1 条 WARN + 3 条观察）

- **R1（WARN，上轮 P3-2 未修 + docstring 新承诺未落地）**：`_normalize_gpus_spec` docstring 写
  “Semantic contract (documented in ``configs/Falcon.yml`` / Phase-1 README)”，但
  `configs/Falcon.yml` 仍无 `logger`/gpus 语义注释（`git diff main -- configs/` 为空），
  `docs/verification/phase1/README.md` 也无对应说明；`training.logger`/`training.comet_project`
  仍只能从代码发现。**修法**：在 `configs/Falcon.yml` 的 `training:` 加 2–3 行注释示例
  （`# logger: csv  # csv | tensorboard | comet | none`、`# gpus: 2 / [0,1] / 0(CPU) / -1(全部)`），
  或不写 config 而改 docstring 指向 `T1-report.md`。
- **R2（观察）**：scalar **任何负数**都映射为 `"all"`（docstring 只写 `-1`）；`gpus: -5` 会被静默当成
  “全部 GPU”。建议 `value == -1` 之外抛 `ValueError`，或 docstring 写明“任意负数=all”。
- **R3（观察）**：单层目录只修了 `csv` 分支；`tensorboard`/`comet` 分支仍 `save_dir=exp_dir, name=exp_name`
  （`:191-231`），将来启用时会回到 `<exp>/<exp>/version_N`。可统一为同一 `dirname` 口径。
- **R4（观察）**：无 CUDA 时 resolver 在 spec 校验前直接返回 CPU，因此 CPU-only 机器上 `gpus: []`/重复索引
  不会报错（GPU 机器会）。对 CPU 复现无影响；若要求配置在任意机器都严格校验，可把校验移到 CUDA 判断之前。

### 7.5 增量复评结论

**T1 的 P2 反馈已完整闭环，无回归**：新实现正确区分“索引列表 vs 卡数”，非法配置在 resolver 内获得清晰
`ValueError`，CSV 日志目录单层化、`--conf_dir` 默认值修复；`git diff` 其余部分（F1/F6/F9 主体、T2 日志改造）
与上轮已评版本一致（T2 文件 hash 未变）。建议合入 Phase-1；残留仅 §7.4 的 P3 文档/一致性小项，
不阻塞 Gate G1。若 Lead 希望把 R1 一并收尾，只需在 `configs/Falcon.yml` 加 2–3 行注释（不改默认行为）。
