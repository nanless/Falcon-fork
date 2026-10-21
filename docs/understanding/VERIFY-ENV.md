# 验证环境说明（VERIFY-ENV）—— 项目专属 conda 环境

> **2026-10-21 起：本项目的唯一允许运行环境是专属 conda 环境 `falcon`。**
> 强制约定见仓库根 [`AGENTS.md`](../../AGENTS.md)；本文件只补充“怎么用这个环境做验证”。
> 早期文档/日志中出现的 `/root/.venvs/falcon` 是搭建 conda 环境之前的临时环境，**已删除**，
> 仅保留用于复现当时的历史证据。

## 0. 快速使用（强制）

```bash
# 交互式
source /root/miniforge3/etc/profile.d/conda.sh
conda activate falcon

# 非交互式（脚本 / subagent，推荐）
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python <script>.py
```

- 环境路径：`/root/miniforge3/envs/falcon`（Python 3.11）
- conda/mamba：`/root/miniforge3/bin/conda`、`/root/miniforge3/bin/mamba`
- 依赖锁定：`docs/verification/conda-env-lock.txt`（pip freeze）、`docs/verification/conda-env.yml`（conda export）
- **禁止**：系统 `/usr/bin/python3`（3.8）、`/root/.venvs/falcon`（废弃）、其他项目 venv/conda env。

自检：

```bash
/root/miniforge3/envs/falcon/bin/python -V
/root/miniforge3/envs/falcon/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. 环境事实

| 项 | 值 |
|---|---|
| 解释器 | `/root/miniforge3/envs/falcon/bin/python`（Python **3.11.17**） |
| 深度学习 | torch **2.8.0+cu128** / torchaudio 2.8.0+cu128 / torchvision 0.23.0+cu128 / triton 3.4.0 / pytorch-lightning **2.6.6** / torchmetrics 1.9.0 |
| 训练侧依赖 | numpy **2.2.6**、scipy 1.17.1、pandas 3.0.6、PyYAML 6.0.3、soundfile 0.14.0、librosa 0.11.0、torch-complex 0.4.4、**typeguard 2.13.3**、timm 1.0.30、torch-optimizer 0.3.0、opencv-python 5.0.0.93、matplotlib 3.11.2、fast-bss-eval 0.1.4、torch-mir-eval 0.4、pystoi 0.4.1、ptflops 0.7.5、rich 15.0.0、tqdm 4.70.1 |
| Studio 依赖 | fastapi 0.156.8、uvicorn 0.54.0、Pillow 13.0.0、huggingface_hub 2.2.0、einops 0.8.2、pytest 9.1.1、httpx 0.28.1 |
| Node | 系统 `node v22.23.3`（不在 conda env 内；JS 测试直接用它） |
| GPU | 2×A800；驱动 535.129 支持 cu12x，`torch.cuda.is_available()` 应为 True；CPU 也能完整跑（2 s 窗口≈2.5 s） |
| 真实权重 | `/root/.cache/falcon_weights/falcon.pt`、`falcon-big.pt`（Lead 从 hf-mirror 下载并验证） |
| 数据 | 无 LRS2/LRS3/RealSSA 数据、无 `pretrain_zoo/` → 训练/评测不可端到端复现，只能做代码级 + 最小验证 |

## 2. 关键 pin 与已知取舍

| 项 | 约定 | 原因 |
|---|---|---|
| `typeguard<3` | **必须** | `look2hear/layers/stft_tfgn.py:8` 用 `check_argument_types`，typeguard 4.x 已移除该 API；否则 `import look2hear.models` 直接失败 |
| `torch==2.8.0+cu128` | 固定 | 本机驱动 535 只支持 cu12x（PyPI 默认 torch 2.14 是 cu13 构建，GPU 不可用）；且此前所有验证结论（参数量/形状/因果性/真实权重 bit-identical/Studio 测试）都在 2.8.0+cu128 上复现通过 |
| `numpy==2.2.6` | 固定 | 与已验证结论一致；避免 2.5.x 与旧库的潜在 ABI/行为差异 |
| `espnet` | **未安装** | 仅 `look2hear/utils/nets_utils.py:493` 函数内可选 import（死代码路径）；装它可能与 typeguard<3 冲突。确需时单独评估 |
| `pypesq` | **用 `pesq` + 兼容层替代** | `pypesq 1.2.4` 在 numpy 2.x 下构建失败（依赖已移除的 numpy.distutils）；已装 PyPI `pesq 0.0.4`（同为 ITU-T P.862），site-packages 放 `pypesq.py` shim 映射 `pypesq.pesq(ref,deg,fs) → pesq.pesq(fs,ref,deg)`，使 `look2hear.metrics`/`audio_visual_test` 可导入、PESQ 可计算（自检：ref 对自身 4.644、加噪 1.019） |
| Studio 的 `torch>=2.14.1` 声明 | 不采用 | 与本机驱动不兼容；功能上 2.8.0 已验证可跑 Studio 全部测试与真实权重推理 |

## 3. stub 说明（保留但通常不再需要）

`docs/verification/stubs/` 下有 `timm`（DropPath）与 `typeguard`（check_argument_types）两个 stub，
是 conda 环境建立之前“无依赖”阶段用的。`falcon` 环境已装**真包**，正常情况不需要 stub。

注意：`docs/verification/scripts/verify_*.py` 脚本内部仍会 `sys.path.insert(0, docs/verification/stubs)`，
因此这些脚本会**优先加载 stub**（行为与真包等价：DropPath(drop_prob=0) 恒等、typeguard 断言为 no-op）。
这是可复现历史结论的刻意选择，不是环境问题；如需测真包行为，把该行去掉或换目录即可。

## 4. 已验证可用的命令

```bash
cd /root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon

# 1) 模型构建/参数量/形状/因果性/死代码
PYTHONPATH=$PWD $ENV/bin/python docs/verification/scripts/verify_model_params.py
PYTHONPATH=$PWD $ENV/bin/python docs/verification/scripts/verify_causality.py

# 2) 真实发布权重（格式/strict 加载/双代码库 bit-identical/FalconRuntime）
PYTHONPATH=$PWD $ENV/bin/python docs/verification/scripts/verify_real_weights.py

# 3) Studio Python 测试（Pillow 已装，无需 stub；会真实解码 JPEG/PNG）
$ENV/bin/python -m pytest studio/tests -q      # 实测 35 passed, 2 skipped（20.9 s）

# 3b) 项目入口模块 import（真实依赖，不注入 stub）
$ENV/bin/python -c "import sys;sys.path.insert(0,'.');import look2hear.models,look2hear.system,look2hear.metrics,audio_train,audio_visual_test;print('imports OK')"

# 3c) GPU 可用性（A800 ×2）
$ENV/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
#   -> 2.8.0+cu128 True 2

# 4) Studio JS 测试（系统 node）
for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done

# 5) 前端常量的协议级验证等
ls docs/verification/scripts/
```

## 5. 历史环境（已删除）

- `/root/.venvs/falcon`（Python 3.10 + torch 2.8.0+cu128，无 timm/torchvision/Pillow/pytest）：2026-10-21
  早期为完成分析临时搭建，**建好 conda `falcon` 后已删除**。`docs/` 中 2026-10-21 之前的日志/评审
  引用该路径只用于复现历史证据。
- `/root/code/github_repos/Wav2TextGrid/.venv`：**不属于本项目**，不要再用。
- `/tmp/dp-verify`、`/tmp/falcon-env-install.log`：临时目录/安装日志，可丢弃。

## 6. 与本文件相关的规范

- 新增/升级依赖 → 更新 `docs/verification/conda-env-lock.txt`，并在 `AGENTS.md` §2 记录。
- 影响已验证结论的变更 → 重跑对应 `docs/verification/scripts/` 脚本并在日志中注明新版本。
- 运行环境相关问题（torch/驱动/依赖）与项目代码问题（P0/P1）要分开定位：
  项目代码问题见 `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` §10。
