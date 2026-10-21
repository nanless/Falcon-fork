# Phase-1 T5 报告：视觉骨干导出验证 + 模型小修

> 分支 `repro/phase1`（未 commit）· owner `model-core` · 2026-10-21
> 执行环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 + torch 2.8.0+cu128，CPU 运行；GPU 未参与）
> 对应任务：`task-13`（Phase1-T5）· 计划条目：`docs/REPRODUCTION_PLAN.md` §6.2 F12 / §6.3 extract 脚本 / FINAL §10 P0-8、P1-`eval() 返回 None`

## 1. 结论

**全部完成，且已用官方权重证明"修复前后前向数值 bit-identical"。**

- F12 视觉骨干导出：新增 `tools/repro/extract_video_backbone.py`，从 `falcon.pt`/`falcon-big.pt` 的 `video_model.*`（137 keys）
  导出 `{"model_state_dict": ...}` 兼容文件（默认写 `/root/.cache/falcon_weights/`，**不写仓库**）；
  `FRCNNVideoModel(pretrain=<导出文件>)` 加载成功：**11,186,688 参数、requires_grad 全 False、
  `[1,1,50,88,88] -> [1,512,50]`**，且与手动 `load_state_dict` 路径输出 bit-identical。
- 三处小修（`FRCNNVideoModel.train()` 返回 self、`Falcon._mdcl_loss` 别名、`models.__all__`）：
  修复前后 eval/train/video 三路前向输出 **sha256 完全相同、max|Δ|=0**；并与 Lead 的
  `docs/verification/phase1/baseline-fingerprint.json` 精确对齐（eval sha `ad0f20a0…`、train est sha `5f01a59…`）。
- 回归：`pytest studio/tests -q` → **35 passed, 2 skipped**（符合预期）；新增 T5 测试 6 passed。

## 2. 改动清单（file:line）

| 文件 | 改动 | 行号（改后） | 说明 |
|---|---|---|---|
| `look2hear/videomodels/frcnn_videomodel.py` | `FRCNNVideoModel.train()` 末尾 `return self`（并补 docstring 解释 BN 冻结语义） | `:102`（docstring `:91-96`） | 修 `.eval()` 返回 `None`（`nn.Module.eval` 实现为 `return self.train(False)`）。**不改变数值**：仅返回值 |
| `look2hear/models/Falcon.py` | 类属性别名 `_mdcl_loss = _mdcl_loss_hard` | `:1464-1468` | 让 `Falcon.forward(..., near_audio=...)` 不再 `AttributeError`；同一函数对象 ⇒ 数值不变 |
| `look2hear/models/__init__.py` | `__all__` 由不存在的 `["AVTIGER_causal"]` 改为实际导出 | `:4-9` | `__all__ = ["Falcon", "FalconSpeakerContrastive", "get", "register_model"]`；`from look2hear.models import *` 恢复可用 |
| `tools/repro/extract_video_backbone.py` | **新增**（~200 行） | — | F12 导出脚本：`--variant small|big`、`--src/--dst`、`--force`、`--dry-run`、`--expect-keys`；带 meta 与回读校验 |
| `tools/repro/tests/t5_checks.py` | 新增（可执行验证脚本，输出即 T5 日志） | — | 6 组检查：[1] imports/`__all__` [2] API 修复 [3] 导出 [4] 加载/前向 [5] big 变体 [6] 修复前后指纹对比 |
| `tools/repro/tests/t5_fingerprint.py` | 新增（确定性指纹采集，pre/post 共用） | — | 固定输入约定：`seed=0`，audio `[1,32000]`、mouth_emb `[1,512,50]`、frames `[1,1,50,88,88]` |
| `tools/repro/tests/test_t5_backbone.py` | 新增（pytest 回归，6 用例） | — | 权重缺失自动 skip；覆盖 `__all__`/eval 返回 self/alias 数值等价/near_audio 路径/导出/加载 |
| `docs/verification/phase1/` | 新增日志/指纹/本报告 | — | `T5-backbone-check.log`、`T5-pre-fix-fingerprint.json`、`T5-post-fix-fingerprint.json`、`T5-report.md` |

> 未改动 metrics/datas/audio_train/litmodules/其它模型文件；未 `git commit`；未向仓库写入大文件（导出的 44.8 MB 文件在 `/root/.cache/falcon_weights/`）。

## 3. 验证命令与结果（全部实际执行）

### 3.1 一键验证（日志 = `docs/verification/phase1/T5-backbone-check.log`）

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/tests/t5_checks.py \
  | tee docs/verification/phase1/T5-backbone-check.log
# -> ALL T5 CHECKS PASSED
```

| 小节 | 检查 | 结果 |
|---|---|---|
| [1] | `import look2hear.models, look2hear.videomodels`；`models.__all__`；`from look2hear.models import *`；`models.get("Falcon"/"FalconSpeakerContrastive")` | OK（star-import 得到 4 个名字） |
| [2] | `Falcon._mdcl_loss is Falcon._mdcl_loss_hard`；`FRCNNVideoModel.train(True)/eval()` 返回 self 且 BN 保持 eval；`Falcon.forward(near_audio=[...])` 正常返回 `(est,(mdcl))` | OK；alias 与 hard 在相关信号上 `loss=22.339954` 两者 `torch.equal=True` |
| [3] | `extract_video_backbone.py --variant small --force` | 137 keys / 11,196,308 张量元素 / 44.8 MB；`torch.load(dst)["model_state_dict"]` 回读 137 keys、meta 存在、key 集合与源一致 |
| [4] | `FRCNNVideoModel(pretrain=<导出>)` | `params=11,186,688 trainable=0 all_frozen=True`；forward `(1,1,50,88,88)->(1,512,50)` finite；与手动加载输出 sha 相同（`1b7255f3…`） |
| [5] | `--variant big`（falcon-big.pt，`SpeakerContrastiveV2`/epoch 715，state_dict 654 条） | 导出成功；与 small 的 video tensors **key 相同、值不同（60/137 个张量不同）** ⇒ 必须按变体导出，不能混用 |
| [6] | pre-fix vs post-fix 指纹（官方权重） | eval/train/video 三路 shape/sum/mean/sha256 **全部 SAME**，params 相同；eval `ad0f20a0…`、train est `5f01a59…` 与 Lead baseline 精确一致 |

### 3.2 数值不变性的证据文件

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/tests/t5_fingerprint.py \
  --out docs/verification/phase1/T5-post-fix-fingerprint.json
```

- `T5-pre-fix-fingerprint.json`（改前）：eval `sha256=ad0f20a0e67e8e75…`、train est `5f01a596…`、sep `8a8e5c2a…`、video `1b7255f3…`。
- `T5-post-fix-fingerprint.json`（改后）：同名 sha **逐位相同**，sum/mean 差值 `= 0.0`。
- 与 Lead `baseline-fingerprint.json` 的 eval/train 条目**完全一致**（说明分支基线未漂移）。
- **模式约定（QA 已定位）**：两个指纹均由 `t5_fingerprint.py` 生成 —— audio 模型按用途分别 `model.eval()`（eval_forward）/
  `model.train()`（train_forward），**video 模型显式 `video_model.eval()`**；而 `baseline-fingerprint.json` 的 video 指纹是
  **FRCNNVideoModel 默认 `training=True` 且未调用 `.eval()`** 时采集的（构造时 BN 处于训练态，只有显式 `.train()` 才会冻结 BN），
  其 `sum=7832.2563` 与本文件的 `eval()` 口径（`sum=5965.4599609375`）不同。**这属模式约定差异、不是数值漂移**；
  两个 JSON 已补 `modes`/`mode_notes` 字段固定该约定（QA 已在 baseline 侧固化 training=True 口径）。

### 3.3 提取脚本 CLI 行为

```bash
# dry-run（不写文件）
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/extract_video_backbone.py --dry-run
# 缺失 checkpoint 的清晰报错（exit=2）
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/extract_video_backbone.py --src /nonexistent/x.pt --dry-run
# 导出（small 默认路径）
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/extract_video_backbone.py --force
```

产物（仓库外）：`/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar`（44,830,218 B）、
`...backbone.big.pth.tar`（44,830,790 B）。

### 3.4 回归测试

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_t5_backbone.py -q
# 6 passed in 8.96s
/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q
# 35 passed, 2 skipped in 16.84s
```

### 3.5 Import 测试

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -c \
  "import look2hear.models, look2hear.videomodels; print('imports OK')"
```

## 4. 遗留问题 / 注意事项

1. **F14 的另一半未修（不在本任务允许文件内）**：`FalconSpeakerContrastive.forward(..., **kwargs)` 仍会静默吞掉
   `near_audio`（`look2hear/models/Falcon_Contrastive.py:95`）。本任务只加了 `Falcon._mdcl_loss` 别名，让**裸 `Falcon`**
   的 MDCL 路径可用；子类的静默吞参需另一任务（或训练侧显式调用 `_mdcl_loss_hard`，如 `av_litmodule_mdcl.py` 已这样做）。
2. **F13（`concat_layers[2]` 置闲）按计划保持现状**：为满足"数值不变"（官方权重 bit-identical），本轮**未**改 `AudioBlock.forward`；
   该问题仍记录在 FINAL §10 P0-7 / `01-model-core.md` §4.9-㉑（实际生效参数 532,129）。是否修由 Lead 决策。
3. **small 与 big 的视觉骨干权重不同**（137 keys 相同、60 个张量数值不同）。训练 small 配置用 `falcon.pt` 导出的文件，
   训练 big 配置必须用 `--variant big`（默认导出到独立的 `.big.pth.tar`，避免覆盖）。已写入脚本帮助与 meta。
4. **video 指纹的模式口径（QA 已定位原因，非数值问题）**：`baseline-fingerprint.json` 的 video 指纹是在
   `FRCNNVideoModel` **默认 `training=True` 且未调用 `.eval()`** 时生成的（此时 BatchNorm3d 用批内统计），故 `sum=7832.2563`；
   一旦调用 `.eval()` 则改用 running stats，得到 `sum≈5955.7`（本报告实测 `5965.46`，输入约定差异）。
   本报告 §3.2 的两个指纹用**显式 `video_model.eval()`** 口径，仅用于"改前 vs 改后"自比较（同名 sha 逐位相同 ⇒ 不变性成立），
   **不用于跨口径比较**。约定已固化：`T5-pre/post-fix-fingerprint.json` 新增 `modes` / `mode_notes` 字段；
   建议 QA 的回归也沿用"baseline = training=True（不 eval）、T5 指纹 = eval()"两套口径并显式标注。
5. 噪声提示：导入 `look2hear.models` 会连带 `timm`/`librosa`/`torch_complex`/`typeguard`/`pytorch_lightning`
   （R3 已记录，非本轮引入）。

## 5. 复跑命令（验收用）

```bash
cd /root/code/github_repos/Falcon-fork
export PYTHONPATH=$PWD
PY=/root/miniforge3/envs/falcon/bin/python

$PY tools/repro/tests/t5_checks.py | tee docs/verification/phase1/T5-backbone-check.log   # ALL T5 CHECKS PASSED
$PY -m pytest tools/repro/tests/test_t5_backbone.py -q                                     # 6 passed
$PY -m pytest studio/tests -q                                                              # 35 passed, 2 skipped
$PY tools/repro/extract_video_backbone.py --dry-run                                        # inspect only
$PY tools/repro/extract_video_backbone.py --force                                          # write cache export
```
