# Phase 1（复现前置修复）执行说明与验收

> 目标：让 `audio_train.py` / `audio_visual_test.py` 在 conda `falcon` 环境可运行、指标语义正确、
> 数据层对 mono 与 RealSSA 兼容、视觉骨干可导出，并建立可复跑的冒烟与回归。
> 上游分支：`main`（= 上游 `689555c`）；工作分支：`repro/phase1`（不 commit，最终由用户/Lead 决定）。
> 计划全文：`docs/REPRODUCTION_PLAN.md` §6/§7。

## 任务分工

| 任务 | 负责人 | 写入范围 | 产出 |
|---|---|---|---|
| T1 训练入口（F1/F6/F9） | train-sys | `audio_train.py` | `T1-report.md`、`T1-*.log`、`tools/repro/tests/test_audio_train_entry.py` |
| T2 litmodule 日志（F2） | train-sys | `look2hear/system/` | `T2-report.md`、`T2-litmodule-check.log` |
| T3 指标 + 评测入口（F7/F8） | loss-metric-eval | `look2hear/metrics/`、`audio_visual_test.py` | `T3-report.md`、`T3-eval-smoke.log`、`tools/repro/tests/test_metrics.py` |
| T4 数据层 + 数据工具（F5） | data-pipe | `look2hear/datas/`、`tools/repro/{make_synthetic_data,realssa_to_falcon}.py` | `T4-report.md`、`T4-data-check.log`、`tools/repro/tests/test_realssa_convert.py` |
| T5 视觉骨干 + 模型小修（F12/F13/F14） | model-core | `look2hear/models/Falcon.py`、`look2hear/models/__init__.py`、`look2hear/videomodels/frcnn_videomodel.py`、`tools/repro/extract_video_backbone.py` | `T5-report.md`、`T5-backbone-check.log` |
| T6 数据获取 + 访问指南 | ops-tests-docs | `tools/repro/fetch_data.py`、`docs/DATA_ACCESS_GUIDE.md` | `T6-report.md`、`T6-data-fetch.log` |
| QA 独立回归 | layers-dsp | `tools/repro/tests/test_phase1_regression.py` | `QA-report.md`、`QA-regression.log` |
| 评审（训练栈） | studio-backend | `docs/verification/phase1/` | `REVIEW-training-stack.md` |
| 回归 + 文档一致性 | studio-frontend | `docs/verification/phase1/` | `REGRESSION-docs.md` |

## 关键基线：`baseline-fingerprint.json`

由 Lead 在**未改动的 `main` worktree**（`/root/wt-main`）上生成，用于验证“修复不改变数值”：

- 官方权重 `falcon.pt`：audio keys 525、video keys 137、size 47,156,343
- 参数：audio **537,697** / video **11,186,688**
- 固定种子（`torch.manual_seed(0)`，audio `[1,32000]` randn，mouth_emb `[1,512,50]` randn）：
  - eval 输出 `[1,1,32000]`，sha256 = `ad0f20a0…f40757`
  - train 模式 est sha256 = `5f01a596…e7545a`，sep_emb `[1,1,112,251,129]` sha256 = `8a8e5c2a…bddc16`
- 视频前端（seed 1，`[1,1,50,88,88]` randn）：输出 `[1,512,50]`，sha256 = `2afcea08…dd40`

> 任何 Phase1 改动后若这些 sha256 变化，必须给出理由（预期：T5 的 `_mdcl_loss` 别名/`return self`/`__all__`
> 都不应改变数值；T1–T4 不触碰前向）。
>
> **video 指纹约定（QA 已定位）**：`video_forward` 是在 **FRCNN 默认 `training=True`、未调用 `.eval()`**
> 下生成的（此时 BatchNorm3d 用批内统计）。若先调 `.eval()` 会得到不同结果（sum 5955.7 而非 7832.2563）。
> QA 回归用例已固化该约定；后续重新生成基线时请显式记录 `model.training` 状态。
> 另见 T5-report §3.2/§4 与 `T5-pre/post-fix-fingerprint.json` 的 `modes`/`mode_notes` 字段（QA 已定位并跨引）。

## 验收口径（Phase 1 Gate G1）

1. `pytest tools/repro/tests -q` 全过（各任务新增测试）。
2. 冒烟训练：合成数据 + `audio_train.py`（1 epoch / 少量 step / CSVLogger）能保存 checkpoint 且 loss 有数值。
3. 冒烟评测：`audio_visual_test.py --limit 1`（合成数据 + 官方权重）能输出 CSV，SI-SNRi/SDRi 为合理量级。
4. `pytest studio/tests -q` 维持 35 passed / 2 skipped；Studio Node 8/8。
5. 官方权重数值指纹不变（与 `baseline-fingerprint.json` 对比 max|Δ|=0 或 sha256 相同）。
6. 每个 T* 报告含：改动 file:line、验证命令、结果、遗留/风险。

## 复跑（Lead 集成时执行）

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon
$ENV/bin/python -m pytest tools/repro/tests -q
$ENV/bin/python -m pytest studio/tests -q
# 合成数据 + 训练/评测冒烟（脚本由 T4/T1/T3 提供后补充具体命令）
```
