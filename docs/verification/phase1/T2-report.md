# T2 报告：Lightning 模块日志 API 修复（F2，去 Comet 硬依赖）

> 分支：`repro/phase1`（未切换、未 commit）。环境：`/root/miniforge3/envs/falcon/bin/python`
> （Python 3.11.17 / torch 2.8.0+cu128 / pytorch-lightning 2.6.6 / CPU 实测，本机 2×A800 未参与）。
> 范围：`look2hear/system/av_litmodule.py`（唯一含 Comet 调用的文件）、
> `look2hear/system/av_litmodule_nass_v2.py`、`look2hear/system/av_litmodule_mdcl.py`（复核无 Comet 调用，继承修复）；
> 新增 `tools/repro/tests/test_litmodule_logging.py`；日志与报告在 `docs/verification/phase1/`。未触碰他人 scope。

## 1. 结论

**T2 完成：三个 litmodule 均可在 `CSVLogger` 与 `logger=False`（无 logger）下正常 fit，不再依赖 Comet API。**
实际执行结果：
- `pytest tools/repro/tests/test_litmodule_logging.py -q` → **5 passed**（CPU，dummy 模型，每个模块真实跑 2 个 train step + 2 个 val step）；
- 三个模块（base / NASSv2 / MDCL）分别用 `CSVLogger` 跑 `pl.Trainer(max_epochs=1)` 均无 `AttributeError`，CSV 产出并含原指标列；
- **无 logger**（`Trainer(logger=False)`，`trainer.logger is None`）同样跑通，`self.log_dict` 不崩（不再出现 `'NoneType' object has no attribute 'experiment'`）；
- 原有 key 全部保留：`learning_rate`、`val_pit_sisnr`、`val_pit_sisnr_{0,1,2,3,4plus}_near`、`test_pit_sisnr`（实测 epoch 0 记录到 `learning_rate / val_pit_sisnr / val_pit_sisnr_0near / val_pit_sisnr_1near / test_pit_sisnr`）；
- `grep -rn "logger.experiment" look2hear/system/*.py` → **无可执行命中**（仅剩说明性注释）；
- 三个文件 `symtable` 未定义全局名扫描 → 均为 **NONE**；
- 回归 `pytest studio/tests -q` → **35 passed, 2 skipped**（与 AGENTS.md 基线一致）。

证据：`docs/verification/phase1/T2-litmodule-check.log`（pytest + grep + symtable + CSV 证据）、
`docs/verification/phase1/T2-studio-regression.log`、`docs/verification/phase1/T2-combined-suite-note.log`。

## 2. 改动清单（file:line，基于修改后文件）

| # | 文件 | 改动 | 位置 |
|---|---|---|---|
| 1 | `look2hear/system/av_litmodule.py` | 删除 4 处 Comet 专属 `self.logger.experiment.log({...})`，改为先组装 `metrics` 字典、在 `on_validation_epoch_end` 末尾统一 `self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)` | 旧 :153-179 → 新 `:142-188`（`log_dict` 在 `:186`） |
| 2 | 同上 | key/数值语义保持不变：`learning_rate`=当前 lr（float）、`val_pit_sisnr`=`-val_loss`（已 all_gather 均值）、`val_pit_sisnr_*near`=`-avg`（逐 bucket，已 all_gather）、`test_pit_sisnr`=`-test_loss`（每 10 epoch 分支）；原 dict 中的 `"epoch"` 是 Comet 的 x 轴字段，改由 PL 的 `on_epoch=True` 原生按 epoch 记录（已在本报告与代码注释中说明） | `:160-186` |
| 3 | 同上 | 保留原 `self.log("lr", ...)`（`on_epoch=True, prog_bar=True, sync_dist=True`）与 buffer 清理逻辑不变；未改损失/训练/返回结构 | `:145-153`、`:187-189` |
| 4 | `look2hear/system/av_litmodule_nass_v2.py` | 复核：无 `logger.experiment` / Comet 调用（仅 `self.log`），无需改动；修复由基类继承生效 | 全文件 |
| 5 | `look2hear/system/av_litmodule_mdcl.py` | 同上（仅有 `print(... file=sys.stderr)` 的 MDCL debug 输出，非 logger API） | 全文件 |

> 说明：任务描述里的"3 个 litmodule 的 `self.logger.experiment.log(dict)`"实际只存在于**基类** 1 个文件（4 处调用），
> NASSv2/MDCL 通过继承使用同一 `on_validation_epoch_end`；因此修复基类即同时修复三个模块，测试对三个模块逐一验证。

## 3. 验证命令与结果（conda `falcon` 环境）

```bash
cd /root/code/github_repos/Falcon-fork

# 1) 三模块 + CSVLogger / 无 logger 的回归测试
/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_litmodule_logging.py -q
#   -> 5 passed, 38 warnings（3 个 CSVLogger fit + 无 logger fit + 代码静态检查）

# 2) Studio 回归
/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q
#   -> 35 passed, 2 skipped in 21.54s

# 3) 静态检查
grep -rn "logger.experiment" look2hear/system/*.py | grep -v ":[0-9]*: *#"   # -> 无可执行命中
#   symtable 未定义名：av_litmodule / nass_v2 / mdcl -> NONE（见 T2-litmodule-check.log）
```

pytest 覆盖点（5 条）：
1. `AudioVisualLightningModule` + CSVLogger：fit 通过；`learning_rate/val_pit_sisnr/val_pit_sisnr_*near/test_pit_sisnr` 均在 `trainer.logged_metrics`；`metrics.csv` 存在且包含这些列；
2. `AudioVisualLightningModuleNASSv2` + CSVLogger：同上（含对比损失路径）；
3. `AudioVisualLightningModuleWithMDCL` + CSVLogger：同上（含 `_mdcl_loss_hard` 调用路径）；
4. `logger=False`（`trainer.logger is None`）：NASSv2 fit 不崩、指标仍写入 `trainer.logged_metrics`；
5. 三个 system 模块源码中不存在可执行的 `logger.experiment` 调用（注释除外）。

## 4. 遗留问题 / 风险（不在 T2 范围或需集成阶段确认）

1. **未在 DDP 多卡下实测**：`log_dict` 的 epoch 级记录在单进程下已验证；DDP 下 per-bucket 的
   `all_gather` 不对称死锁（F15）与本次改动无关、仍是独立风险（本任务不改采样/汇总逻辑）。
2. `val_pit_sisnr_*near` 只在对应 bucket 非空时记录（原语义如此）；小验证集时部分 bucket 键会缺失，
   不是 bug，但 FINAL 文档应说明"按命中 bucket 记录"。
3. `"epoch"` 字段不再作为指标写入（原 Comet x 轴）；若下游有脚本从 Comet 读取 `{"epoch": ...}`，
   需改为使用 PL/CSV 的 `epoch` 列（CSVLogger 自带 `epoch` 列）。已在代码注释与本报告记录。
4. `self.log_dict` 在 epoch-end hook 使用 `on_epoch=True` 且**未额外加 `sync_dist`**：因为 `val_pit_sisnr`
   与分桶值在此前已手动 `all_gather`，各 rank 值相同；如后续有人改为"每 rank 本地值"，需自己加 `sync_dist=True`。
5. 集成阶段仍需 `train_smoke.py`（真实数据/backbone）验证 CSV/TensorBoard 实际落盘与训练曲线；
   本任务按约定**未运行真实训练、未 git commit**。
6. 备注：`pytest tools/repro/tests -q`（整个共享目录）当前有 5 个失败，均来自其它 agent 正在并行修改的
   `test_eval_entry.py / test_metrics.py / test_synthetic_data.py`（见 `T2-combined-suite-note.log`），
   与本任务改动的两个文件无关（我的测试单跑 13 + 5 全过）。
