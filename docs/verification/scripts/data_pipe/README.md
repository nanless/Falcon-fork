# data-pipe 验证脚本（05 数据管线，R3 补交，对应 05 §8）

来源：作者 R1/R2/R3 期间的 `/tmp/dp-verify/`（原文保留），按 VERIFY-ENV.md §4 补交到仓库，供 R4 独立复核。

## 依赖

- 解释器：`/root/.venvs/falcon/bin/python`（3.10 + torch 2.8）。
- 仓库缺 torchvision/cv2/matplotlib，本目录配套 stub：`docs/verification/stubs/data_pipe/`
  （`torchvision.transforms.Grayscale` 按官方 luma 权重实现；仅用于数据管线 import，见 05 §8.1 保真边界）。
- 另需仓库自带 stub：`docs/verification/stubs/`（timm/typeguard，加载模型时用）。

## 快速复核（在仓库根执行）

```bash
export PYTHONPATH=$PWD/docs/verification/stubs:$PWD/docs/verification/stubs/data_pipe:$PWD
PY=/root/.venvs/falcon/bin/python

# 1) 生成合成数据（写入 /tmp/dp-verify，可重复执行）
$PY docs/verification/scripts/data_pipe/make_fake_data.py
# 2) 建索引（真实 DataPreProcess 代码）
$PY DataPreProcess/process_lrs23.py --in_audio_dir /tmp/dp-verify/data --out_dir /tmp/dp-verify/index
# 3) 数据管线动态验证：57 PASS / 0 FAIL
$PY docs/verification/scripts/data_pipe/verify_dataset2.py
# 4) DataPreProcess/check 脚本验证：19 PASS / 0 FAIL
$PY docs/verification/scripts/data_pipe/verify_preprocess.py
# 5) mono-eval P0 复现（需真实模型权重？不需要——用配置构建模型，无权重）
$PY docs/verification/scripts/data_pipe/verify_p0_eval.py
# 6) 真实权重评测形状验证（需 /root/.cache/falcon_weights/falcon.pt；可选）
$PY docs/verification/scripts/data_pipe/verify_realweight_eval.py
```

日志：`docs/verification/runs/data_pipe/`。
说明：脚本内的数据/fixture 路径默认 `/tmp/dp-verify`（与作者运行环境一致）；如 /tmp 已清理，先跑第 1 步重新生成。
