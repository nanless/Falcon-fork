# T1 报告：训练入口 `audio_train.py` 修复（F1/F6/F9 + 评审 P2/P3 收尾）

> 分支：`repro/phase1`（未切换、未 commit）。环境：`/root/miniforge3/envs/falcon/bin/python`
> （Python 3.11.17 / torch 2.8.0+cu128 / pytorch-lightning 2.6.6 / 2×A800）。
> 范围：`audio_train.py` + `configs/Falcon.yml`（Lead 授权，仅注释）+ 本报告/测试/日志。
> 测试：`tools/repro/tests/test_audio_train_entry.py`（pytest，20 个用例）。
> **版本说明**：本报告按独立评审 `docs/verification/phase1/REVIEW-training-stack.md`（studio-backend，
> 结论 P0=0/P1=0/P2=2/P3=4）分两轮收尾：① P2-1/P2-2/P3-3/P3-4 修复；② Lead 授权 A 后再做：
> `configs/Falcon.yml` 注释示例、`_normalize_gpus_spec` 负数语义（仅 -1=全部）、未用 `import sys` 与
> scheduler 空白清理。行号已同步到最终代码（**audio_train.py 414 行**）。

## 1. 结论

**T1 完成：训练入口在 PL 2.6.6 下可装配、可跑真实 GPU smoke、且不改变数值路径。**
F1 / F6 / F9 全部修复；评审 P2-1（int-vs-list GPU 语义）、P2-2（非法 gpus 提前校验）、
P3-1（legacy `0`/`-1` 语义显式化）、P3-3（CSVLogger 双重 exp 名目录）、P3-4（`--conf_dir` 默认值 + 行号漂移）、
P3-2（`training.logger`/`gpus` 语义文档，config 注释形式）均已收尾。

关键验收（全部实际执行，证据见 §4）：
- `pytest tools/repro/tests/test_audio_train_entry.py -q` → **20 passed**；
- `pytest tools/repro/tests -q` → **82 passed**（QA 已将 `configs/Falcon.yml` 加入白名单并加“仅注释”守卫）；
- `pytest studio/tests -q` → **35 passed, 2 skipped**；
- **真实 GPU smoke**（`/root/tmp/phase1_smoke/smoke.yml`，`gpus: [0]`，1 epoch=4 train step）：
  `accelerator=gpu devices=[0] strategy=auto`，CSV / ckpt / best_model.pth / conf.yml 全部落盘；
- **F9**：落盘 `conf.yml` 的 `audio_params/video_params` 保留 `repeats`/`shared`、无注入的 `causal`；
  同一进程运行期 config 仍被 `Separator` pop（正/反双向证据）；
- **数值指纹不变**：eval/train 前向 sha256 与 `docs/verification/phase1/baseline-fingerprint.json` 完全一致，
  video 前向与 `T5-post-fix-fingerprint.json` 一致，参数量 537,697 / 11,186,688 不变；
- **config 注释无行为变化**：`yaml.safe_load(configs/Falcon.yml)` 与 `main` 版**解析结果逐键相等**，
  `training.gpus` 仍为 `[0..9]`，未新增 `logger` 键值（只有注释）。

## 2. 改动清单（file:line，基于最终 `audio_train.py`，共 414 行）

| # | 对应 | 改动 | 位置 |
|---|---|---|---|
| 1 | F1 | `build_logger(config, exp_dir)`：读 `config["training"]["logger"]`，默认 `csv` → `CSVLogger`；支持 `tensorboard`/`tb`（缺包清晰 `RuntimeError`）、`comet`/`comet_ml`（缺 `comet-ml` 清晰 `RuntimeError`，支持 `COMET_API_KEY`/`training.comet_project`）、`none`/`null`/`false` → `None`；未知值 `ValueError` | `:169-232` |
| 2 | F1 | 删除未定义名 `comet_logger`；logger 在 Trainer 构造前由 config 创建；logger 为 `None` 时向 Trainer 传 `False`（PL2 中 `None` 会创建默认 logger，`False` 才是关闭） | `:348-350`、`:371` |
| 3 | F6 + **P2-1** + **P3-1** | `_normalize_gpus_spec()`：**list/tuple/range/逗号串一律视为 GPU 索引**，**scalar int 一律视为卡数**；legacy `0`→CPU、**仅 `-1`→全部可见卡**（其它负数 → `ValueError`，docstring 已写明）；`gpus: [1]` 正确选 GPU 1 | `:50-95` |
| 4 | F6 + **P2-2** | `resolve_accelerator_and_devices()`：`accelerator="gpu"/"cpu"`；索引列表→`devices=list`，卡数→`devices=int`；**空列表/重复索引/越界索引/非法负数在 resolver 内抛清晰 `ValueError`**（附可见卡提示）；10 卡配置在 N 卡机自动裁到可见子集并告警；>1 卡用 `DDPStrategy(find_unused_parameters=True)`，否则 `"auto"`；无 CUDA → CPU fallback | `:98-166` |
| 5 | **P3-3** | CSVLogger 目录去重：`CSVLogger(save_dir=os.path.dirname(os.path.abspath(exp_dir)), name=exp_name)` → 日志写到 `<exp_dir>/version_N/metrics.csv`，不再出现 `<exp>/<exp>/version_N` | `:193` |
| 6 | **P3-4** | `--conf_dir` 默认值改为 `configs/Falcon.yml`（README 复现命令所用配置），help 提示旧 NASSv2 配置不随仓库提供 | `:34` |
| 7 | F6 | `Trainer(...)` 去掉 `gpus=`/`accelerator=None`：`devices=devices, accelerator=accelerator, strategy=strategy`；保留 `max_epochs`/`callbacks`/`default_root_dir`/`limit_train_batches=1.0`/`gradient_clip_val=5.0`/`sync_batchnorm=True`/`num_sanity_val_steps=0` | `:352-374` |
| 8 | F9 | exp_dir 构造与 `conf.yml` dump 前移到 **建模之前**；写盘内容为 `copy.deepcopy(config)`（`safe_config`）；运行期 `config["main_args"]["exp_dir"]` 同步设置 | `:236-252`（dump `:243-252`，建模 `:259-269`） |
| 9 | 稳健性 | `--ckpt_path` 非空但文件不存在 → 告警后从头训练；resume 分支保留 | `:376-385` |
| 10 | 稳健性 | 收尾：`best_k_models.json` 值兼容 float/Tensor；`torch.load(..., map_location="cpu")`；无 best ckpt 时告警并跳过 `best_model.pth` | `:387-413` |
| 11 | **P3-2（授权 A）** | `configs/Falcon.yml` 的 `training:` 下新增 2 行**注释**（`# logger: csv …`、`# gpus: [0, 1] …`），不新增键值 → 解析结果与 `main` 逐键相同 | `configs/Falcon.yml:129-130` |
| 12 | 清理 | 删除未使用 `import sys`；scheduler 行空白格式 `["sche_name"] )(` → `["sche_name"])(`（纯格式） | `:1-3`、`:274-276` |

## 3. 验证命令与结果（conda `falcon` 环境）

```bash
cd /root/code/github_repos/Falcon-fork

# 静态
/root/miniforge3/envs/falcon/bin/python -c "import ast; ast.parse(open('audio_train.py').read())"   # AST OK
# symtable：undefined globals -> NONE
# configs/Falcon.yml：yaml 解析结果 == main 版（注释不生效）；training.gpus 不变、无 logger 键

# 单测（monkeypatch Trainer/DM/video；真实 audio model；不训练）
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_audio_train_entry.py -q    # 20 passed
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests -q                             # 76 passed + 1 已知白名单失败
/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q                                  # 35 passed, 2 skipped

# 真实 GPU smoke（fixture 由集成阶段提供；cwd 在仓库外，避免污染仓库）
cd /root/tmp/phase1_smoke_retest
PYTHONPATH=/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python /root/code/github_repos/Falcon-fork/audio_train.py \
  --conf_dir /root/tmp/phase1_smoke/smoke.yml
#  -> accelerator=gpu devices=[0] strategy=auto, 4 train step, CSV/ckpt/best_model.pth/conf.yml 落盘

# 权重指纹
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/tests/t5_fingerprint.py --out /tmp/fp.json
#  -> eval/train sha == baseline-fingerprint.json；video sha == T5-post-fix-fingerprint.json；params 不变
```

pytest 覆盖点（20 条，摘录）：
1. `ast.parse`；2. symtable 无未定义全局名；3. CLI 仍有 `--conf_dir`/`--ckpt_path` 且默认值为 `configs/Falcon.yml`；
4-5. `_normalize_gpus_spec` 的 int/list/str/None/legacy/非法负数语义矩阵；
6. 默认 logger 为 `CSVLogger` 且 `log_dir=<exp_dir>/version_N`（无双重 exp 名）；
7. `logger: none` → `None`；8. 未知值 → `ValueError`；9. comet 缺包 → 清晰 `RuntimeError`；
10-14. resolver：`[0..9]`→`[0,1]+DDP`、`[1]`→GPU1、`8`→2 卡、`-1`→全部、`0`→CPU、`"0,1"`→索引；
15-18. resolver 非法配置：空列表 / 重复索引 / 越界索引 / `-2` → `ValueError`；无 CUDA → CPU；
19. `main()` 装配：logger/accelerator/devices/strategy/grad clip/limit/sync_bn/sanity/max_epochs/callbacks +
`conf.yml` 保留 `repeats/shared` 且无 `causal` + 运行期 config 被污染（F9 双向证据）+ CSV 路径不重复；
20. `main()` + `logger:none` → Trainer 收到 `logger=False` 不崩。

## 4. 证据文件

- `docs/verification/phase1/T1-checks.log`：环境、AST、symtable、**gpus resolver 12 组矩阵**、
  `main()` 装配与 F9 键检查、config 解析等价性、全量测试已知失败说明；
- `docs/verification/phase1/T1-pytest.log`：本文件 20 passed + 全目录结果 + studio 回归 + config diff + diffstat；
- `docs/verification/phase1/T1-gpu-smoke.log`：真实 GPU smoke、产物清单、CSV 路径、F9 conf 检查、指纹对比。

## 5. 遗留问题 / 风险（含已知/延后项）

1. **已知/延后（Lead 确认保持现状）**：
   - **R3**：`training.logger: tensorboard|comet` 时 Logger 目录仍会有 `save_dir/name/version_N` 的层级
     （TB/Comet 自带目录约定，未做去重；仅 CSV 分支修了双重 exp 名）；
   - **R4**：无 CUDA 时 `resolve_accelerator_and_devices()` 直接返回 CPU，不校验 `gpus` spec
     （非法值只在 GPU 机器上提前报错）；
   - `training.parallel`、`training.system`、`datamodule.data_name` 仍为未读取的死配置键（F 清单外）；
   - `--ckpt_path` resume 沿用 PL `fit(ckpt_path=...)`，`~`/环境变量不展开。
2. **已闭环**：QA 已把 `configs/Falcon.yml` 加入 `ALLOWED_TRACKED` 并新增“仅注释”守卫测试；全量 `tools/repro/tests` 现 **82 passed**（见 `T1-pytest.log`）。
3. tensorboard/comet **成功构建**未实测（环境未安装对应包，只验证清晰报错路径）；启用需装包并更新环境锁定。
4. 多卡仅验证到 Trainer 入参 + reviewer 的 2-GPU toy DDP；真实多卡大样本训练属集成阶段。
5. 本任务按约定**未运行超过 smoke 的训练、未 git commit**；本组源码改动仅 `audio_train.py`、`configs/Falcon.yml`
   （注释）、`look2hear/system/av_litmodule.py`（T2）。
