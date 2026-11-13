# AGENTS.md — Falcon 项目工作约定

> **2026-11-10 补充：**本文件保留环境和仓库操作约定。模型、数据、实验与测试的现状和证据边界以 [docs/STATUS.md](docs/STATUS.md)、[数据集说明](docs/understanding/20261110-datasets.md) 及 [2026-11-10 审查记录](docs/verification/deepdive-20261110/审查记录.md) 为入口；下文带日期的测试数字是历史记录。本轮 CPU 复现套件为 166 passed / 2 skipped，两项因历史临时资产缺失跳过。
>
> 历史默认 VideoBlock 在 batch>1 时有跨样本影响。2026-11-13 起当前修复分支的默认分支以独立单步复用发布 B=1 语义，详见 [发布模型批次耦合](docs/verification/released-falcon-batch-coupling-20261113.md)；`paper_causal=True` 会额外裁短 stride-2 的中间特征，`lookahead_ab.py` 的默认行比较同一次输出；跨版本比较另见状态页。发布的 LRS2 s1/s2 已包含混合比例，名义 gain 差不是逐条波形 RMS 比；10,000 次有放回抽样约覆盖 22.12% 方向条目、39.35% 不同混合（均为期望）。这些结论、条件和仍未完成的验证均见状态页。

> 本文件适用于在此仓库（`/root/code/github_repos/Falcon-fork`）工作的所有 AI agent 与人类开发者。
> 环境与代码记录始于 2026-11-08，2026-11-10 已补充复核。环境与测试结果可能随时间变化；遇到冲突时，先核对当前代码、现行状态页及原始证据。
> 入口说明：仓库根此前另有一个 `agents.md` 符号链接指向本文件，两个名字都被 git 跟踪；macOS/Windows 的大小写不敏感文件系统 checkout 时会互相覆盖。该软链已删除，**入口只保留 `AGENTS.md`**。

## 1. 运行环境

- 本项目的依赖只装在专属 conda 环境 **`falcon`** 里。换个解释器照样能读写文件，但凡是 import 项目模块的命令会在导入阶段失败。
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

- 下面三个解释器跑不了本项目，看到它们出现就把命令里的路径换成 `falcon`：
  - `/usr/bin/python3`：系统 Python 3.8，没有 torch / timm / lightning，`import look2hear.models` 直接 `ModuleNotFoundError`；
  - `/root/.venvs/falcon`：早期临时验证环境，建 conda 环境后已删除，现在指向它是 `No such file or directory`；历史日志里出现这个路径，只用于复现当时的证据；
  - 其他项目的 conda env / venv（如 `/root/code/github_repos/Wav2TextGrid/.venv`）：依赖版本与本项目不同，跑起来会在导入或第一个张量操作上失败，报错位置和真正的原因隔得很远。
- 自检（应为 3.11.x 且 torch 可导入）：

  ```bash
  /root/miniforge3/envs/falcon/bin/python -V
  /root/miniforge3/envs/falcon/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

- Node（Studio 前端 JS 测试）不使用 conda 环境，直接用系统 `node v22`。

## 2. 环境内容

- 精确锁定：`docs/verification/conda-env-lock.txt`（`pip freeze` 快照，**115 个包、物理 124 行**；
  2026-11-08 实测 `wc -l` = 124、以 `pkg==` 开头的行 = 115，差值是 9 行文件头注释。2026-10-24 已把 18 个
  `pkg @ file:///root/wheelhouse/...` 改写成 `pkg==version`，可直接 `pip install -r` 重建）与
  `docs/verification/conda-env.yml`（conda export，94 行）。
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
  - **`typeguard` 停在 2.13.3**：`look2hear/layers/stft_tfgn.py` 使用 `check_argument_types`，typeguard 3.x 起移除了这个 API，
    升到 3.x 或 4.x 会让 `import look2hear.models` 直接失败；要升级得先把 `layers/stft_tfgn.py` 里那几处改成新 API。
  - **`espnet` 未安装**：仅 `look2hear/utils/nets_utils.py:493` 的函数内可选 import（死代码路径）；
    它自带的 `typeguard` 依赖会与 §上一条的 `typeguard<3` 冲突。确需安装时先算清依赖再动手。
  - **`pypesq` 用替代方案**：`pypesq 1.2.4` 在 numpy 2.x 下无法构建（setup.py 依赖已移除的 numpy.distutils）。
    已安装 PyPI `pesq 0.0.4`（同一 ITU-T P.862 实现），并在 site-packages 放置 `pypesq.py` 兼容层
    （`pypesq.pesq(ref, deg, fs)` → `pesq.pesq(fs, ref, deg)`），使 `look2hear.metrics` / `audio_visual_test.py`
    可正常导入与计算。该 shim **不在仓库内**（属环境内容），重启环境需按 §5 重建。
  - 环境已经验证：真实权重 strict 加载 + CPU/GPU 推理、`pytest studio/tests` **有 CUDA 51 passed / 2 skipped，
    无 CUDA 53 passed**（2026-11-08 实测；那 2 个 skip 是只在 CUDA 上运行的用例）、`pytest tools/repro/tests`
    **有 GPU 168 passed、无 GPU 同样 168 passed**（2026-11-08 双路实测）、8 个 Node 测试全过、
    `import look2hear.models/system/metrics`、`import audio_train/audio_visual_test` 均 OK。
  - **GPU**：本机 2×A800（驱动 535.129），cu128 轮子可用；CPU 也可完整运行（2 s 窗、B=1 单窗前向约 0.6 s，4 线程实测 592–694 ms；1 线程约 1.0 s）。
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

> `docs/verification/scripts/*.py` 会在每个脚本里无条件执行 `sys.path.insert(0, <repo>/docs/verification/stubs)`
> （例如 `verify_model_params.py:9,20`），而这个目录里放着 `timm/`（只实现了 `DropPath`）、`typeguard.py`
> （`check_argument_types` 空实现）和 `data_pipe/matplotlib/` 三个替身。2026-11-08 实测：加上 stubs 后
> `timm.__file__` 指向 `docs/verification/stubs/timm/__init__.py`，`typeguard` 同样被覆盖；`torchvision`
> 本身仍解析到 site-packages，但因为真实 `timm` 不再加载，`timm → torchvision` 这条导入链整个不参与。
> 这些脚本原本是给没有 timm/typeguard 的沙箱写的，在 `falcon` 环境（timm 1.0.30、typeguard 2.13.3 都在）
> 里跑会静默改用替身实现。引用那里的 log 或结论时，写明"证据来自 stub 环境"；涉及 timm 行为或
> typeguard 参数校验的结论，不能拿它当依据。

> 训练/评测入口的两个旧 P0 现在一半已修、一半变成可诊断的报错。把它们当成环境问题会白查一轮：
>
> - **Comet logger 已修**（`ac91800`）。`a7af472` 里未定义的 `comet_logger` 会让建 `Trainer` 时抛
>   `NameError`；现在 `audio_train.py:169` 的 `build_logger(config, exp_dir)` 按 `training.logger`
>   分派，`comet_ml` 未安装时只有 `logger: comet` 这一条分支报错，并给出安装提示（`audio_train.py:209-227`）。
> - **`conf.yml` 类名仍坏，但报错带修复指引**。`Experiments/checkpoint/Falcon/conf.yml:88` 的快照写着
>   `audionet_name: SpeakerContrastiveV2`，而 `look2hear.models` 只导出 `FalconSpeakerContrastive`；
>   `audio_visual_test.py:316-321` 会抛一个说明该快照已过期的 `AttributeError`。换成 `configs/Falcon.yml`，
>   或把那一行改成 `FalconSpeakerContrastive`，评测即可继续。
> - 训练入口可跑：CPU 2 epoch 实测通过（`docs/understanding/deepdive-20261108/04-training-system.md` §4.2）。

快速自检（结果及跳过项取决于当前环境）：

```bash
FALCON_ENV_DIR=/root/miniforge3/envs/falcon
$FALCON_ENV_DIR/bin/python -V                                                   # 3.11.17
$FALCON_ENV_DIR/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
#   -> 2.8.0+cu128 True 2
$FALCON_ENV_DIR/bin/python -c "import sys;sys.path.insert(0,'.');import look2hear.models, look2hear.system, look2hear.metrics, audio_train, audio_visual_test;print('imports OK')"
$FALCON_ENV_DIR/bin/python -m pytest studio/tests -q                            # 51 passed + 2 skipped (CUDA) / 53 passed (CPU)
$FALCON_ENV_DIR/bin/python -m pytest tools/repro/tests -q -rs                    # 历史 168 passed；2026-11-10 CPU 为 166 passed / 2 skipped，见 docs/STATUS.md
```

`tools/repro/tests` 收集到 168 个用例：166 个 `def test_`，其中 `test_litmodule_logging.py::test_fit_with_csvlogger`
参数化成 base / nass_v2 / mdcl 三例。**有 GPU 与无 GPU 都是 168 passed**（2026-11-08 实测：约 147 s / 约 170 s，
`cuda_available=False` 已在断言里）。修前无 GPU 是 166 passed / 2 failed，两处已修：指纹测试现在固定
`torch.set_num_threads`，`test_phase1_regression` 的 studio skip 硬断言放宽为 `skipped in (0, 2)`
（另 `audio_train.py` 校验顺序前移）。耗时在不同 run 之间从约 150 s 到 355 s 都出现过
（`docs/STATUS.md` §5 记的是 355 s，差额尚未归因），引用耗时写清是哪一次。两个 fingerprint 用例对
`torch.set_num_threads` 敏感：`eval_sha` 是 float32 输出字节的 sha256，线程数一变末几位就变，看着像回归其实不是；
测试里已固定线程数，手工复核时也要先固定（例如 `OMP_NUM_THREADS=4`）。

### 3.1 五个影响数值的开关（默认值 = 发布行为）

| 开关 | 默认 | 打开 / 改动后改变什么 | 代价 |
|---|---|---|---|
| `paper_causal` | `False` | `AudioBlock` 的 `downsample` / `fusion` / `concat` 三处卷积原本硬编码 `causal=False`；True 时改为透传 `self.causal`，即论文 §4.2 的左对齐 causal padding，逐卷积前瞻从 2 帧降到 0 帧 | 在该次 CPU 单输入对照中相对默认开关输出变化（`max\|Δ\|≈13`），要可比得重训；端到端"看未来"仍到窗尾，因为 30 处 `cLNhw` 是全窗 `GroupNorm`，padding 不是瓶颈 |
| `video_lstm_batch_first` | `False` | False 时逐样本逐帧独立执行一次 LSTM 零状态步，复用发布 B=1 语义；True 才沿 25 fps 帧轴递推 | 在该次 CPU 单输入对照中输出变化（`max\|Δ\|≈1.3e-2`），等于换了视频编码器，与发布权重的结果不能混用 |
| `respect_index` | `False` | False 是历史行为：`__getitem__` 忽略 `idx`、每次从全量索引随机取，`DataLoader(shuffle=True)` 因此无效、采样有放回；True 严格按 `idx` 取，`__len__` 变成 `len(mix)`，一个 epoch 每条样本过一次 | 采样序列与历史 run 不同，loss 曲线不能直接对比；音频裁剪起点仍随机 |
| `mouth_fps` | `25.0` | 嘴部裁剪帧率，决定 `mouth_start = int(rand_start / sample_rate * mouth_fps)` 与所需帧数 `int(segment * mouth_fps)` | 25.0 精确复现历史写死的 `(rand_start/16000)*25`；改成别的值要保证数据本身是该帧率，否则唇动与音频错位 |
| `pad_mouth` | `False` | False 时帧数不足 `int(segment * mouth_fps)` 直接 `ValueError`；True 复制末帧补齐 | 补齐会在唇动里插静帧，短片段的结果与重采样的片段不等价 |

前两个是 `look2hear/models/Falcon.py` 里 `AudioBlock` / `VideoBlock` 的构造参数，A/B 用
`tools/repro/experiments/lookahead_ab.py`（纯 CPU、约 40 s、真实权重）；后三个是 `avspeech_dataset.py` /
`avspeech_dataset_near.py` 的 dataset 构造参数，`configs/*.yml` 里没有对应键，要用得自己接线。
默认开关保留发布 B=1 路径；2026-11-13 起消除了历史默认 B>1 的样本串扰。`lookahead_ab.py` 的默认值比较是输出与自身比较，用于记录本次默认输出；跨版本逐位比较需另行执行。本轮另对 small 冻结 tag 与当前模型做过一个 CPU 输入的权重对照，范围见状态页；Big、其他输入和设备不能由此推定。

## 4. 依赖安装 / 升级规则

- 依赖装进 `falcon` 环境。装完按下面三步收尾，否则环境重建或结论复核时会缺依据：
  1. 将 `pip freeze` 保存到带日期的新快照，保留历史 `docs/verification/conda-env-lock.txt`；
  2. 在本文件 §2 记录新增/变更的包；
  3. 若影响已有验证结论，重跑 `docs/verification/scripts/` 对应脚本。
- 装到系统 Python 或其他项目环境里，跑出来的结果与 `falcon` 不是同一套依赖，不能当本项目的证据；系统 Python 3.8 本身也装不上 torch 2.8。
- `typeguard` 3.x 移除了 `check_argument_types`，升级会让 `import look2hear.models` 失败；要升级需先改 `layers/stft_tfgn.py`。
- `numpy==2.2.6` 与 `torch==2.8.0+cu128` 是现有全部结论的测量基准（参数量、形状、因果性、真实权重 bit-identical、Studio 协议测试都在这一版上验过）。换版本等于换尺子，这些结论要重跑一遍才能继续引用。
- 模型权重放在 `/root/.cache/falcon_weights/` 或 `~/.cache/huggingface`：`falcon.pt` 45 MB、`falcon-big.pt` 48 MB、两个 FRCNN 主干各 43 MB，提交进仓库会让每次 clone 多出近 180 MB 二进制，而且 Git 历史里的权重删不干净。

## 5. 重建环境（新机器 / 环境损坏）

```bash
/root/miniforge3/bin/mamba create -n falcon python=3.11 -y
FALCON_ENV_DIR=/root/miniforge3/envs/falcon
$FALCON_ENV_DIR/bin/pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
$FALCON_ENV_DIR/bin/pip install pytorch-lightning "numpy==2.2.6" scipy pandas pyyaml soundfile tqdm rich packaging \
  timm librosa torch-complex torch-optimizer opencv-python matplotlib fast-bss-eval torch-mir-eval \
  pesq==0.0.4 pystoi ptflops
$FALCON_ENV_DIR/bin/pip install "typeguard<3"
$FALCON_ENV_DIR/bin/pip install fastapi "uvicorn[standard]" "Pillow>=12,<14" "huggingface_hub>=2" einops pytest httpx
```

- 也可以用锁定文件重建：`$FALCON_ENV_DIR/bin/pip install -r docs/verification/conda-env-lock.txt`（需先装好 torch 的 cu128 轮子）。
- 网络提示：本机 `pypi.org` 曾不可达，建环境时使用镜像
  `-i https://pypi.tuna.tsinghua.edu.cn/simple` 与 PyTorch cu128 轮子
  `--index-url https://download.pytorch.org/whl/cu128`；已下载的轮子缓存在 `/root/wheelhouse/`
  （与 pip 缓存硬链接，5.4 GB，可用于离线重装）；`TMPDIR=/root/tmp`（`/` 只有 20 GB，/tmp 易满）。
- **pypesq 兼容层**（锁定文件不含它，需手动创建）：
  ```bash
  $FALCON_ENV_DIR/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pesq==0.0.4
  cat > $FALCON_ENV_DIR/lib/python3.11/site-packages/pypesq.py <<'EOF'
  import numpy as np
  from pesq import pesq as _pesq
  def pesq(ref, deg, fs, mode="wb"):
      return float(_pesq(int(fs), np.asarray(ref), np.asarray(deg), mode))
  EOF
  ```
- 注意：仓库根 `requirements.txt` 未 pin `typeguard`，直接装会拿到 3.x/4.x（两者都没有 `check_argument_types`），
  `import look2hear.models` 会失败；`studio/requirements.txt` 的 torch 约束为 `>=2.8,<3`，具体 CUDA 构建仍需匹配服务器驱动；
  两份都按本文件的口径装。

## 6. 仓库 / 产物约定

- `docs/` 下是本项目的分析产物（`understanding/` 文档、`reviews/` 评审、`verification/` 脚本与日志），
  按需追加即可，已有文件保持原样（删掉一份日志，引用它的结论就失去追溯路径）。复现计划见 **`docs/REPRODUCTION_PLAN.md`**（数据/缺口/分阶段执行/验收）。
  2026-10-24 那一轮的理解快照见 **`docs/understanding/deepdive-20261024/`**（历史快照；数字与结论以 `docs/STATUS.md` 为准）；
  2026-11-08 一轮深挖见 **`docs/understanding/deepdive-20261108/00-SYNTHESIS.md`**（01–10 域深挖 + 11 论文对齐 + 12 因果性定论）。
  `docs/verification/**` 下的 `.log`/`.csv` 已通过 `.gitignore` 负向规则纳入版本控制（证据要能复核）。
- 历史文档/日志中出现的 `/root/.venvs/falcon`、`/tmp/dp-verify` 等路径是旧环境/临时目录，
  只用于复现历史证据；新命令用 `falcon` 环境，否则会像 §1 说的那样在导入阶段失败。
- 改源码前先存基线：动手前各记一份 `git status --short` 与 `git diff --stat`，改完再比一次。
  没有基线时，事后分不清哪几行是本轮改的、哪几行是之前留下的；基线记录写进当轮日志，不放进仓库根目录。
- 证据只追加：`docs/verification/**` 下的 `.log`/`.csv` 是结论可复算的唯一凭据，覆盖既有行会让历史结论
  失去追溯路径。修正结论时新增一份带日期的文件，并在开头写清它替代哪一份、为什么。
- 提交由用户决定：`git commit` / `git checkout` / 清理工作区都等用户明确要求；默认只改工作区文件，
  交付时报出改了哪几个文件、各增删多少行。
- 修上游 P0 先确认范围再单独开分支；直接改在主干上会和 `repro-freeze-20261031` 这份证据基线混在一起，
  之后无法判断某条结论对应的代码状态。
