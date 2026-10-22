# AGENTS.md — Falcon 项目工作约定（运行环境强制）

> 本文件适用于在此仓库（`/root/code/github_repos/Falcon-fork`）工作的所有 AI agent 与人类开发者。
> 最后更新：2026-10-21。**任何与本文件冲突的旧命令/文档，以本文件为准。**

## 1. 运行环境（强制，最高优先级）

- 本项目**唯一允许**的 Python 运行环境是专属 conda 环境 **`falcon`**。
- 位置：`/root/miniforge3/envs/falcon`（Python 3.11）；conda/mamba 位于 `/root/miniforge3/bin/`。
- 激活（交互式）：

  ```bash
  source /root/miniforge3/etc/profile.d/conda.sh
  conda activate falcon
  ```

- 非交互式（脚本 / CI / subagent 推荐，等价且无需激活）：

  ```bash
  /root/miniforge3/envs/falcon/bin/python  <script>.py
  /root/miniforge3/envs/falcon/bin/pip     <args>
  /root/miniforge3/envs/falcon/bin/pytest  <args>
  ```

- **禁止**使用以下解释器运行本项目：
  - `/usr/bin/python3`（系统 Python 3.8，无项目依赖）；
  - `/root/.venvs/falcon`（早期临时验证环境，**已于建 conda 环境后删除**；历史日志中出现的该路径仅用于复现当时证据）；
  - `/root/code/github_repos/Wav2TextGrid/.venv` 或其他项目的 conda env/venv。
- 自检（应为 3.11.x 且 torch 可导入）：

  ```bash
  /root/miniforge3/envs/falcon/bin/python -V
  /root/miniforge3/envs/falcon/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

- Node（Studio 前端 JS 测试）不使用 conda 环境，直接用系统 `node v22`。

## 2. 环境内容

- 精确锁定：`docs/verification/conda-env-lock.txt`（`pip freeze`，111 行）与 `docs/verification/conda-env.yml`（conda export，94 行）。
- 实测关键版本（2026-10-21 建好并验证）：
  - Python **3.11.17**；torch **2.8.0+cu128**、torchaudio 2.8.0+cu128、torchvision 0.23.0+cu128、triton 3.4.0；
  - pytorch-lightning **2.6.6**、torchmetrics 1.9.0、lightning-utilities；
  - numpy **2.2.6**、scipy 1.17.1、pandas 3.0.6、PyYAML 6.0.3、soundfile 0.14.0、librosa 0.11.0、torch-complex 0.4.4；
  - **typeguard 2.13.3**、timm 1.0.30、torch-optimizer 0.3.0、opencv-python 5.0.0.93、matplotlib 3.11.2；
  - fast-bss-eval 0.1.4、torch-mir-eval 0.4、pystoi 0.4.1、ptflops 0.7.5、rich 15.0.0、tqdm 4.70.1；
  - Studio：fastapi 0.156.8、uvicorn 0.54.0、Pillow 13.0.0、huggingface_hub 2.2.0、einops 0.8.2、pytest 9.1.1、httpx 0.28.1。
  - Phase 2 评测依赖（2026-10-22 增装，用于 RealSSA-Real DNSMOS）：**speechmos 0.0.1.1**（MIT，wheel 自带 DNSMOS P.835/P.808 ONNX 模型）、
    **onnxruntime 1.31.0**（MIT）、protobuf 7.36.2、flatbuffers 25.12.19；安装命令/许可见 `docs/verification/phase2/P2E-env-delta.txt`；DNSMOS ONNX 模型 sha256 见
    `docs/verification/phase2/P2E-dnsmos-synthetic.json` 与 `P2E-report.md`。
- 训练/评测侧：`torch 2.8.0+cu128`（CUDA 12.8 构建，CPU 同样可用）、`torchaudio`、`torchvision`、`pytorch-lightning`、
  `numpy 2.2.6`、`scipy`、`pandas`、`pyyaml`、`soundfile`、`librosa`、`torch-complex`、**`typeguard<3`**、`timm`、
  `torch-optimizer`、`opencv-python`、`matplotlib`、`fast-bss-eval`、`torch-mir-eval`、`pypesq`、`pystoi`、`ptflops`、`rich`、`tqdm`
- Studio 侧：`fastapi`、`uvicorn[standard]`、`Pillow`、`huggingface_hub>=2`、`einops`、`pytest`、`httpx`
- 关键说明：
  - **`typeguard` 必须 <3**：`look2hear/layers/stft_tfgn.py` 使用 `check_argument_types`，4.x 已移除该 API，
    升级会导致 `import look2hear.models` 失败。
  - **`espnet` 未安装**：仅 `look2hear/utils/nets_utils.py:493` 的函数内可选 import（死代码路径）；
    安装可能与 `typeguard<3` 冲突。如确需，先评估依赖冲突再装。
  - **`pypesq` 用替代方案**：`pypesq 1.2.4` 在 numpy 2.x 下无法构建（setup.py 依赖已移除的 numpy.distutils）。
    已安装 PyPI `pesq 0.0.4`（同一 ITU-T P.862 实现），并在 site-packages 放置 `pypesq.py` 兼容层
    （`pypesq.pesq(ref, deg, fs)` → `pesq.pesq(fs, ref, deg)`），使 `look2hear.metrics` / `audio_visual_test.py`
    可正常导入与计算。该 shim **不在仓库内**（属环境内容），重启环境需按 §5 重建。
  - 环境已经验证：真实权重 strict 加载 + CPU/GPU 推理、`pytest studio/tests` **35 passed/2 skipped**、
    8 个 Node 测试全过、`import look2hear.models/system/metrics`、`import audio_train/audio_visual_test` 均 OK。
  - **GPU**：本机 2×A800（驱动 535.129），cu128 轮子可用；CPU 也可完整运行（2 s 窗口前向约 2.5 s）。
  - torch 固定 **2.8.0+cu128** 是刻意选择：本项目此前所有验证结论（参数量、形状、因果性、真实权重
    bit-identical、Studio 协议测试）均在该版本上复现通过。

## 3. 常用命令（全部在 `falcon` 环境内）

```bash
cd /root/code/github_repos/Falcon-fork

# 模型结构 / 参数量 / 形状 / 因果性
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/verify_model_params.py
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/verify_causality.py

# 真实发布权重（已缓存在 /root/.cache/falcon_weights/；缺失时用 mirror 重下）
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/verify_real_weights.py

# Studio Python 测试（Pillow 已装，不需要 PIL stub）
/root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q

# Studio JS 测试（系统 node v22）
for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done
```

> 训练/评测入口存在已知 P0（Comet logger、conf.yml 类名等），**不要把它们误判为环境问题**；
> 详见 `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` §10。

快速自检（应全部通过）：

```bash
$ENV/bin/python -V                                                   # 3.11.17
$ENV/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
#   -> 2.8.0+cu128 True 2
$ENV/bin/python -c "import sys;sys.path.insert(0,'.');import look2hear.models, look2hear.system, look2hear.metrics, audio_train, audio_visual_test;print('imports OK')"
$ENV/bin/python -m pytest studio/tests -q                            # 35 passed, 2 skipped
```

## 4. 依赖安装 / 升级规则

- 只能安装到 `falcon` 环境；安装后必须：
  1. 更新 `docs/verification/conda-env-lock.txt`（`pip freeze > ...`）；
  2. 在本文件 §2 记录新增/变更的包；
  3. 若影响已有验证结论，重跑 `docs/verification/scripts/` 对应脚本。
- 不得把包装进系统 Python 或其它环境来"绕过"。
- 不得升级 `typeguard>=3`、不得随意替换 `numpy==2.2.6` / `torch==2.8.0+cu128`（否则已验证结论失效）。
- 模型权重只放在 `/root/.cache/falcon_weights/` 或 `~/.cache/huggingface`，不要提交进仓库。

## 5. 重建环境（新机器 / 环境损坏）

```bash
/root/miniforge3/bin/mamba create -n falcon python=3.11 -y
ENV=/root/miniforge3/envs/falcon
$ENV/bin/pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
$ENV/bin/pip install pytorch-lightning "numpy==2.2.6" scipy pandas pyyaml soundfile tqdm rich packaging \
  timm librosa torch-complex torch-optimizer opencv-python matplotlib fast-bss-eval torch-mir-eval \
  pypesq pystoi ptflops
$ENV/bin/pip install "typeguard<3"
$ENV/bin/pip install fastapi "uvicorn[standard]" "Pillow>=12,<14" "huggingface_hub>=2" einops pytest httpx
```

- 也可以用锁定文件重建：`$ENV/bin/pip install -r docs/verification/conda-env-lock.txt`（需先装好 torch 的 cu128 轮子）。
- 网络提示：本机 `pypi.org` 曾不可达，建环境时使用镜像
  `-i https://pypi.tuna.tsinghua.edu.cn/simple` 与 PyTorch cu128 轮子
  `--index-url https://download.pytorch.org/whl/cu128`；已下载的轮子缓存在 `/root/wheelhouse/`
  （与 pip 缓存硬链接，5.4 GB，可用于离线重装）；`TMPDIR=/root/tmp`（`/` 只有 20 GB，/tmp 易满）。
- **pypesq 兼容层**（锁定文件不含它，需手动创建）：
  ```bash
  $ENV/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pesq==0.0.4
  cat > $ENV/lib/python3.11/site-packages/pypesq.py <<'EOF'
  import numpy as np
  from pesq import pesq as _pesq
  def pesq(ref, deg, fs, mode="wb"):
      return float(_pesq(int(fs), np.asarray(ref), np.asarray(deg), mode))
  EOF
  ```
- 注意：仓库根 `requirements.txt` 未 pin `typeguard`（默认会装 4.x 导致导入失败），
  `studio/requirements.txt` 的 torch 约束（`>=2.14.1`）与本机驱动不兼容；**以本文件为准**。

## 6. 仓库 / 产物约定

- `docs/` 下是本项目的分析产物（`understanding/` 文档、`reviews/` 评审、`verification/` 脚本与日志），
  可以继续追加，**不要删除**。复现计划见 **`docs/REPRODUCTION_PLAN.md`**（数据/缺口/分阶段执行/验收）。
- 历史文档/日志中出现的 `/root/.venvs/falcon`、`/tmp/dp-verify` 等路径是旧环境/临时目录，
  只用于复现历史证据；**新命令一律使用 `falcon` 环境**。
- 修改源码不在本约定范围内；如需修 P0，先与用户确认并单独开分支。
- 不要 `git commit` / `git checkout` / 清理工作区，除非用户明确要求。
