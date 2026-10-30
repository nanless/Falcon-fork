# 08 运维、测试、CI、文档与配置一致性

> 负责人：`ops-tests-docs`（R1 初稿 v1 → **R3 修订 v2**；依据 `docs/reviews/round1/08-ops-tests-docs-review-by-studio-backend.md`，逐条响应见 §9.1）
> 版本哈希：R1 评审基线 sha256 = `75692199722df17160157529bb95073e47787e471369f352afe0844869084b98`；R3 交付 sha256 见 §9.1 末行（写入前计算）
> 分析对象：commit `7576854`（`git log --oneline -1`，工作区干净，只有未跟踪的 `docs/`）
> 证据约定：`[代码]` 直接读源码；`[推断]` 由源码/文档推断；`[验证]` 本轮实际运行命令确认。
> 本轮约束：未修改任何源码/脚本/配置；未 `pip install`；未下载权重；未 commit。

## 1. 子系统职责与总体位置

**一句话结论**：本域覆盖 Falcon 的两条"最后一公里"——①Studio 的 macOS/Linux 一键安装与启动链（`studio/scripts/` + 两个 `.command` + `server.py` 的 CLI/端口/浏览器逻辑）；②仓库的"可信度基础设施"——测试矩阵（8 个 Node 测试全部可跑且通过；4 个 pytest 文件在注入仓库内最小 PIL stub 后可完整运行）、CI 发布（Pages 仅 main + 公开仓库）、依赖策略（Studio 用"范围下限 + 精确 lock"，训练侧基本不锁），以及四份 README 与代码的逐条一致性审计。**本域最重要的产出不是复述 README，而是偏差清单**：共确认 **2 个 P0**（`audio_train.py` 训练入口因未定义变量必然崩溃；README 给出的评测命令因 `conf.yml` 的 `audionet_name` 无法 import 必然崩溃）、**若干 P1/P2**（`best_k_models.json` 的 `${ROOT}` 占位路径、根 requirements 缺 `huggingface_hub`、`evaluated_mac_params.py` 硬编码 CUDA、项目页 `large/Dolphin.wav` 大小写、公网在线 demo 与本地 loopback 限制冲突（D22）等）。

### 1.1 范围与位置

| 范围 | 位置 | 在系统中的角色 |
|---|---|---|
| Studio 安装/启动 | `studio/scripts/{bootstrap,launch,doctor}.sh`、`studio/scripts/check_environment.py`、`studio/{Start Falcon.command,Setup Meeting Audio.command}` | 用户唯一入口；负责选 Python、建 venv、装依赖、下载权重、拉起 `server.py` |
| 会议音频 | `studio/scripts/install-blackhole.sh` | 可选、显式的 BlackHole 2ch 安装器；启动器永不调用 |
| 测试 | `studio/tests/**`（4 pytest + 6 Node + `browser/`）、`studio/web/*.test.cjs`（2） | 覆盖 DSP、i18n、跟踪几何、UI 控制流、服务协议、权重解析、设备选择 |
| 依赖 | 根 `requirements.txt`（训练/评测）、`studio/requirements{,-lock,-dev}.txt` | 两套互不相同的 Python 环境 |
| CI/页面 | `.github/workflows/pages.yml`、`project_page/index.html` | 把静态项目页发布到 `https://cslikai.cn/Falcon/`（媒体在 `oss.cslikai.cn`） |
| 实验产物 | `Experiments/checkpoint/Falcon/{conf.yml,best_k_models.json}` | 评测的数据/模型配置快照 + best ckpt 选择元数据 |
| 忽略规则 | `.gitignore`、`studio/.gitignore` | 防止权重/环境/日志入库 |
| 文档 | 根 `README.md`、`README_zh.md`、`studio/README.md`、`studio/README_zh.md` | 用户可见的唯一"说明书"，本域逐条核对 |

### 1.2 本域 Top 结论（详见 §4.7 偏差清单）

1. **P0** 训练入口不可运行：`audio_train.py:170` 使用未定义的 `comet_logger`（全仓库仅此一处出现），`pl.Trainer(...)` 构造即 `NameError`；根 `requirements.txt` 也没有 `comet_ml`。[代码+验证（全仓库 grep）]
2. **P0** README 给出的评测命令不可运行：`Experiments/checkpoint/Falcon/conf.yml:88` 写 `audionet_name: SpeakerContrastiveV2`，但 `look2hear/models/__init__.py:1-2` 只导出 `Falcon` / `FalconSpeakerContrastive`；`audio_visual_test.py:129` 用 `getattr(look2hear.models, ...)` → `AttributeError`。本环境注入 stub 后实测 `hasattr(look2hear.models,'SpeakerContrastiveV2') == False`。[验证]
3. **P1** `best_k_models.json` 的 5 个 key 是字面量 `${ROOT}/Falcon/epoch=*.ckpt`；`audio_visual_test.py:83-94` 只做 `os.path.exists(best_path)`，**不展开 `${ROOT}`**，且仓库里没有任何 `*.ckpt`（被 `.gitignore:24` 忽略）→ 不传 `--ckpt_path` 时评测最终 `FileNotFoundError`（`audio_visual_test.py:103-107`）。[代码+验证（json 解析/存在性）]
4. **P1** 根 `requirements.txt`（29 行）没有声明 `huggingface_hub`，而 README 的 `--ckpt_path hf://...` 命令在 `audio_visual_test.py:323-326` 需要它 → 全新训练环境按 README 安装后该命令 `ModuleNotFoundError`。[代码+验证（全量 import 审计）]
5. **P1** `python evaluated_mac_params.py` 只能跑在有 CUDA 的机器上：`evaluated_mac_params.py:48,77-81,86` 多处 `.cuda()`；且硬编码读 `configs/Falcon.yml`（`:39`），README 未提示。[代码]
6. ~~**P1（待外部确认）** 项目页 large 场景 `Dolphin.wav` 大小写~~ → **2026-10-30 已确认非缺陷**：`large/Dolphin.wav` HTTP 200、`large/dolphin.wav` 404，大写 D 是 OSS 真实 key，禁止规范化。
7. **测试现状（R3 修订）**：8 个 Node 测试**全部通过**（实测 ≈4.0–4.1 s，见 §2.4/§8）；pytest 四文件在**注入仓库内最小 PIL stub**（`docs/verification/scripts/pil_stub/`，仅提供 `Image.new/open/save/convert`）后**可完整运行**：评审基线 **33 passed / 4 skipped**（当时无权重缓存，`test_server` 协议用例全过），本轮复跑为 **35 passed / 2 skipped**（本环境 HF 缓存已有 small/big 权重，`test_infer_file` 2 项不再 skip；详见 §8.2）。真实 JPEG/PNG 解码仍属 Pillow 职责、未验证。
8. **启动链认知修正**：任务描述里归给 `launch.sh` 的"venv 选择/锁优先/端口 7860-7899/Metal-CUDA 探测/自动开浏览器"，实际分布在 `bootstrap.sh`（venv/依赖，`:12-83`）、`server.py`（端口/浏览器，`:304-329`）、`falcon_runtime.py`（设备，`:81-92,183-196`）；`launch.sh` 只有 20 行，只做"调 bootstrap → 确保权重 → exec server.py"（`:6-20`）。[代码]
9. **测试面修正 Lead 侦察图**：`00-project-map.md:71` 写 `check_*.mjs/cjs（5 个）`，实际 `studio/tests/` 有 **6 个**（4 `.mjs` + 2 `.cjs`），另有 `studio/web/*.test.cjs` **2 个**；`test_*.py` 4 个正确。[验证（ls/wc）]
10. **CI 与仓库状态一致**：`pages.yml` 只在 push `main` 且路径命中 `project_page/**` 或本 workflow 时触发，并有 `workflow_dispatch`；deploy job 用 `if: ${{ !github.event.repository.private }}` 保证私有仓库不部署；artifact 路径是 `project_page`。[代码+验证（YAML 解析）]
11. **P1（R3 新增，D22）README 公网在线 demo 与本地同源限制冲突**：`README.md:21`/`README_zh.md:23`/`studio/README.md:16`/`studio/README_zh.md:16` 提供 `https://u464727-dmja-7896a455.bjb3.seetacloud.com:8443/` 在线 demo（并明说摄像头/麦克风数据发往该服务器），但 `server.py:175-179` 的 `same_origin` 只接受 loopback origin（hostname ∈ localhost/127.0.0.1/::1），`:196` 的 `TrustedHostMiddleware` 白名单也只有 loopback/testserver。我实测：公网 origin + 同 host → `same_origin=False`；`TestClient` 注入公网 `Host` → **HTTP 400 "Invalid host header"** ⇒ 线上必然是另一份部署补丁/反向代理改写，本仓库 `server.py` 不能原样提供该 demo，README 未说明。[代码+验证]

## 2. 文件清单与逐文件解析

### 2.1 `studio/scripts/` 五个脚本

#### 2.1.1 `launch.sh`（20 行，100755）

职责：**唯一的"正常启动"入口**，输出/副作用是拉起本地服务。它是"组合器"，不含环境选择逻辑。

| 行 | 内容 | 说明 |
|---|---|---|
| `:2` | `set -euo pipefail` | 严格模式 |
| `:4-5` | `SCRIPT_DIR` / `APP_DIR` | `APP_DIR = studio/` |
| `:6` | `PYTHON_BIN="$("$SCRIPT_DIR/bootstrap.sh")"` | **契约**：bootstrap stdout 只有一行 python 路径，所有消息走 stderr（见 `bootstrap.sh:2,25,85`） |
| `:9` | 用 `$PYTHON_BIN -c 'from falcon_runtime import weights_source; print(weights_source("small"))'` 判断是否已缓存 | 依赖 `huggingface_hub.try_to_load_from_cache`（`falcon_runtime.py:48-56`） |
| `:10` | 首次运行提示"下载 JusperLee/Falcon（约 45 MB）" | 与参数量推导自洽：11,724,385 × 4 B ≈ 46.9 MB（44.7 MiB，fp32 state_dict，不含 optimizer）[推断]；文件真实体积/内容未验证 |
| `:11-14` | `resolve_checkpoint("small")` 失败 → 提示 `HF_ENDPOINT=https://hf-mirror.com` 重试并 `exit 1` | 任何 import/下载失败都被归因成"下载失败"，措辞略误导（P2） |
| `:17-18` | 中文 banner | `printf ... >&2`，只走 stderr |
| `:19` | `cd -- "$APP_DIR"` | 保证 `server.py` 的相对路径（`web/`、`configs/`）正确 |
| `:20` | `exec "$PYTHON_BIN" "$APP_DIR/server.py" "$@"` | `--device/--model/--port/--no-browser` 全部透传给 `server.py` |

#### 2.1.2 `bootstrap.sh`（85 行，100755）

职责：**选出/创建可用的 Python 环境并确保依赖**；成功后 stdout 只打印 python 路径。是安装链的核心。

逐段：

| 行 | 逻辑 | 备注 |
|---|---|---|
| `:8-10` | `valid_python()`：可执行且 `(3,10) <= version_info[:2] < (3,13)` | 与 `check_environment.py:18-19`、两个 README 的 "Python 3.10–3.12" 一致 |
| `:12-18` | 目标环境目录优先级：`FALCON_VENV` → `studio/.venv`（若其中 python 合法）→ `$HOME/Library/Caches/FalconStudio/venv` | README 承诺的"复用 .venv，否则装到 iCloud 之外的缓存目录" |
| `:19-22` | 相对路径转绝对（基于 `$PWD`） | `FALCON_VENV=rel/path` 可用 |
| `:24-27` | 若目标环境 python 合法**且** `check_environment.py --quiet` 通过 → 打印路径、`exit 0` | "复用"条件 = 依赖版本全部达标；否则走重装 |
| `:29-33` | 目标目录存在但 python 不合法 → 报错并提示换 `FALCON_VENV`，**不删除** | 避免误删用户数据 |
| `:34-47` | `SOURCE_PYTHON = FALCON_PYTHON`（非法则报错）；否则按 `python3.11→3.12→3.10→python3` 找 | 顺序暗示 README 推荐 3.11 |
| `:48-60` | Finder 不继承终端 PATH，追加硬编码候选：`/opt/homebrew/bin/python3.11`、`/usr/local/bin/python3.11`、`/Library/Frameworks/.../3.11`、`~/.local/share/uv/python/cpython-3.11.*/bin/python3.11` | 只覆盖 3.11（Finder 场景的兜底） |
| `:61-64` | 全找不到 → 明确中文指引（装 3.11 / 设 `FALCON_PYTHON` / 看 README）exit 1 | |
| `:65-68` | `mkdir -p` 父目录 + `$SOURCE_PYTHON -m venv "$ENV_DIR"` | |
| `:70-83` | 安装依赖：**优先 `requirements-lock.txt`**（`:71-75`）；有 `uv` 用 `uv pip install --python ...`（`:76-78`），否则 `ensurepip` 后 `pip install`（`:79-83`） | 锁优先是 README 明确承诺；uv 路径不要求 venv 里先有 pip |
| `:84` | 安装后再跑一次 `check_environment.py`（非 quiet，输出到 stderr） | `set -e` 下若校验失败会中止，不会打印路径 |
| `:85` | 打印 `$ENV_DIR/bin/python` | 供 `launch.sh:6`/`doctor.sh:5` 捕获 |

关键设计：**stdout 单值契约**使 `launch.sh`/`doctor.sh` 可以 `PYTHON_BIN="$(bootstrap.sh)"` 组合；同时也是脆弱点——任何子命令的 stdout 泄漏都会污染返回值（当前实现都把消息定向到 stderr）。

#### 2.1.3 `doctor.sh`（32 行，100755）

职责：诊断。与 `bootstrap.sh` 的关系是**"先调用 bootstrap，再补充设备/权重信息"**：

| 行 | 内容 |
|---|---|
| `:5` | `PYTHON_BIN="$(bootstrap.sh)"` —— 因此 `doctor.sh` **可能触发安装**（不是纯只读诊断） |
| `:6` | `check_environment.py` 完整输出（版本/import 检查） |
| `:7-20` | heredoc 内：`platform.platform()`/`machine()`、`torch.__version__`、`torch.backends.mps.is_built()/is_available()`、`torch.cuda.is_available()`（附 GPU 名） |
| `:21-30` | 遍历 `falcon_runtime.VARIANTS`，逐个 `weights_source()`：`cached`/`huggingface`/未装 hub；`small` 不可用则 `exit 1` |
| `:31` | 结束语："这里只检查环境和模型文件；实际加载与分离结果请在应用中确认" |

**bootstrap 与 doctor 差异**：bootstrap = 建环境+装依赖（有副作用，成功即返回 python 路径）；doctor = bootstrap + 依赖版本/import 检查 + 平台/MPS/CUDA + 权重缓存状态（只读诊断 + 可能的隐式安装）。README `studio/README.md:112` 的 "check dependencies, Metal / CUDA and model downloads" 与实际相符，但未提醒它会先安装。

#### 2.1.4 `check_environment.py`（54 行，mode 100644，需用 python 调用）

职责：**不依赖 pip 的依赖自检**（"Verify supported package versions and their native imports without pip"）。检查项清单：

1. `:18-19` Python 版本必须 3.10/3.11/3.12（否则报错）。
2. `:20-39` 逐行解析 `studio/requirements.txt`（同目录 `../requirements.txt`）：
   - 去注释（`#` 后）与空行（`:21-24`）；
   - 要求每行严格是 `pkg[extra]>=min,<max`（`:25-27`）——**没有容错**：若未来改成 `==`/裸包名会 `ValueError` 崩溃（P2 契约风险）；
   - `metadata.version(pkg)` + "数字前缀元组比较"（`:28-37`），避免 `packaging` 依赖；
   - 缺失包或越界 → 记 errors（`:38-39`）。
3. `:40-44` 实际 import 8 个模块：`torch, numpy, yaml, einops, fastapi, uvicorn, PIL, websockets`。注意**没有 import `huggingface_hub`**（它只被版本检查，不在 import 清单里）。
4. `:45-50` `--quiet` 只抑制成功提示；**错误始终打印到 stderr**；有错返回 1。`bootstrap.sh:24` 正依赖这个返回码决定是否重装。

### 2.2 `studio/scripts/install-blackhole.sh`（72 行，100755）

职责：**可选**的 BlackHole 2ch 安装器；`install-blackhole.sh:2` 明言 "Explicit optional setup. Never called by the ordinary Falcon launcher."（全仓库仅 `Setup Meeting Audio.command` 引用它）。

| 行 | 逻辑 |
|---|---|
| `:3` | `set -euo pipefail` |
| `:5` | 驱动路径 `/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver`（macOS） |
| `:6-14` | 找 brew：先 `command -v brew`，再 `/opt/homebrew/bin/brew`、`/usr/local/bin/brew`（与 bootstrap 的 Finder 兜底思路一致） |
| `:16-31` | `--check`：只读。报告驱动是否存在、brew 路径、官方 GitHub 链接；**未做 uname 判断**，非 macOS 上也能输出（Linux 实测 exit 0）；已存在时提示"若设备列表仍没有请重启 macOS" |
| `:32-35` | `--help/-h`：中文用法 |
| `:36-38` | 其它参数 → stderr + `exit 2` |
| `:40-43` | 真正安装前才检查 `uname -s == Darwin`，否则 `exit 1` |
| `:44-48` | 已存在驱动 → 提示无需重复安装 + 重启建议，`exit 0` |
| `:49-52` | 无 Homebrew → 给官方安装包 URL，`exit 1` |
| `:54-68` | 中文说明块：将执行 `brew install --cask blackhole-2ch`；会改系统音频驱动目录、可能要求管理员密码、**脚本不会重启电脑/不会改默认输入输出** |
| `:70` | `HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ANALYTICS=1 "$BREW_BIN" install --cask blackhole-2ch` |
| `:71-72` | 完成后提示重启 macOS 及会议接线（Falcon 会议输出/会议麦克风 = BlackHole 2ch；系统/会议扬声器 = 耳机） |

与 README 对照：`studio/README.md:70`（"double-click Setup Meeting Audio.command, or run ./scripts/install-blackhole.sh (--check only reports status). It uses the Homebrew cask blackhole-2ch, may ask for an administrator password and needs a macOS restart. The normal launcher never installs drivers."）逐条与代码一致。[代码]

### 2.3 两个 `.command` 启动器（macOS Finder）

| 文件 | 行 | 内容 | 要点 |
|---|---|---|---|
| `studio/Start Falcon.command` | `:1` | `#!/bin/bash` | 没有 `set -e`（要捕获退出码） |
| | `:2` | `APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"` | 与脚本同目录（`studio/`） |
| | `:3` | `"$APP_DIR/scripts/launch.sh" "$@"` | 透传参数 |
| | `:4-8` | 退出码非 0/130 时打印错误指引并 `read` 等回车 | 130 = Control-C，正常退出不打扰 |
| | `:9` | `exit "$EXIT_CODE"` | |
| `studio/Setup Meeting Audio.command` | `:2-3` | 调 `install-blackhole.sh "$@"` | 可传 `--check` |
| | `:4-6` | **总是**打印"按回车关闭此窗口"并 `read` | 安装/检查后都停留 |
| | `:7` | `exit "$EXIT_CODE"` | |

`file` 显示两者均为 "Bourne-Again shell script, UTF-8 Unicode text executable"（LF，无 CRLF/BOM 问题）；`git ls-files -s` 显示 `100755`；`bash -n` 通过。[验证] README 中 `studio/Start Falcon.command` 的大小写/空格与实际文件名完全一致（`README.md:115`、`README_zh.md:117`、`studio/README.md:24`、`studio/README_zh.md` 同）。

### 2.4 `studio/tests/**` 与 `studio/web/*.test.cjs`：逐个测试解析 + 可运行矩阵

测试总量：`studio/tests/` 下 4 个 pytest（`test_device/test_infer_file/test_server/test_weights`，共 305 行）、6 个 Node 测试（4 `.mjs` + 2 `.cjs`，共 1196 行）、`browser/`（手工浏览器 QA：`index.html` + `test.js` + 官方测试人像 `portrait.jpg`）、`serve_browser.py`（QA 服务器，非测试）；`studio/web/` 下 2 个 `*.test.cjs`。

**Node 测试（纯 Node，无浏览器/摄像头/MediaPipe/网络）**

| 文件 | 测什么（证据行） | 依赖 | 当前环境 | 本轮实测 |
|---|---|---|---|---|
| `tests/check_i18n.mjs`（200 行） | 用最小 DOM 加载**真实** `web/i18n.js`：8 项检查（阻塞 localStorage、中文词典全量可译且保留占位符、HTML 静态 key、错误/动态文案、切换保留状态），审计 349 条词典（`:200`） | node `assert/fs/path/url` | ✅ 可运行 | rc=0，85 ms，"8 localization checks passed (349 dictionary entries audited)" |
| `tests/check_capture_dsp.cjs`（64 行） | 在 `vm` + 假 `AudioWorkletProcessor` 中跑**真实** `web/capture-worklet.js`：16/44.1/48 kHz 时钟、2 s=32000 样本、时间戳补偿、1 kHz 通带、12 kHz 混叠抑制、时钟空洞恢复 | node `assert/fs/path/vm` | ✅ 可运行 | rc=0，515 ms，输出 `{"test":"PASS",...}` |
| `tests/check_mouth_tracking.mjs`（173 行） | 导入**真实** `web/mouth-tracker.js` 的 `newMouthTrack/updateMouthTrack/mouthGeometry/clampROI`：嘴部几何、缩放不随张嘴抖动、丢脸 coasting、4 帧重锁、跳变确认、多人选择、epoch 竞态、编码失败、停止（`:173` 汇总） | 只需 node（无 MediaPipe） | ✅ 可运行 | rc=0，63 ms，PASS |
| `tests/check_playback_resampler.mjs`（180 行） | 导入**真实** `web/playback-resampler.js`，自带 radix-2 FFT 独立测量：44.1/48 kHz 通带 ±0.1 dB、镜像/混叠 < -60 dB、6 段 0.5 s 分块与整体逐位一致、稳态长度、reset 清 FIR 历史（`:86,103,127,143,171`） | node（16k→设备率上采样 DSP） | ✅ 可运行 | rc=0，1586 ms，ALL PASS |
| `tests/check_spectrum.mjs`（136 行） | 用 `data:` URL 动态 import **真实** `web/spectrum-dsp.js`：FFT_SIZE/HOP/FREQUENCY_ROWS/16 kHz/DB_FLOOR=-90、频率行映射、满幅/-6/-20 dBFS 校准、DC/Nyquist 不翻倍、NaN/Inf 静音、LiveSpectrum 分块等价、整段 overview 列数/耗时（`:136` 打印 JSON） | node（Buffer/base64） | ✅ 可运行 | rc=0，971 ms，输出 `{"test":"PASS",...}` |
| `tests/check_tracking_ui.cjs`（443 行） | `vm` 执行**真实** `web/app.js`（`:10-12` 正则剥掉 3 个 ESM import），用 `web/index.html` 的 id 集合构造最小 DOM/音频 stub：18 项控制流（停止/首窗在途/延迟响应/录音切片/频谱/预览/模式切换竞态等，`:441-442`） | node；无摄像头/模型/浏览器 | ✅ 可运行 | rc=0，433 ms，"18 tracking UI control-flow checks passed" |
| `web/capture-worklet.test.cjs`（51 行） | capture worklet DSP 的另一组断言：16/44.1/48 kHz、1 kHz 通带、12 kHz 混叠、FIR 时延、时钟空洞（`:51`） | node | ✅ 可运行 | rc=0，373 ms，PASS |
| `web/video-worker.test.cjs`（46 行） | `web/video-worker.js`：25 fps 采样、灰度 ROI、wall-clock 消息、VideoFrame 关闭、错误清理（`:45`） | node（不需要 MediaPipe，因为用 stub/合成帧） | ✅ 可运行 | rc=0，42 ms，PASS |

> 结论：8 个 Node 测试**全部真实执行、全部 rc=0**，总耗时约 4.1 s。`node --check` 对 `studio/web/*.js`（13 个）、`tests/*.mjs|cjs`（6 个）、`web/*.test.cjs`（2 个）共 **21** 个文件全部通过（13 + 4 + 2 + 2；R1 曾误写 19）。[验证]

**Python 测试（pytest）**

| 文件 | 测什么 | 依赖 | 当前环境可运行？ | 本轮实测 |
|---|---|---|---|---|
| `tests/test_weights.py`（70 行） | 权重解析：short link 与文件名、`hf://<org>/<repo>/<file>`、本地文件覆盖、缓存状态、下载失败提示镜像、`from_pretrained` 短链 → variant/device（`:26-70`） | pytest + torch + `falcon_runtime`（会 import `falcon_model`/torch；**不联网**：hub 被 `monkeypatch.setitem(sys.modules,...)` 假替身，`:17-24`） | ✅ 可运行（本环境实测） | **7 passed in 1.74s** [验证] |
| `tests/test_device.py`（49 行） | `pick_device("auto")` CUDA→MPS→CPU 顺序、显式设备、不可用时报错；GPU 预热失败自动回退 CPU（`:35` 需"无 CUDA + 权重已缓存"才跑）；无 GPU 时显式 cuda 报错（`:44`） | pytest + torch | ✅ 可运行（部分 skip） | **4 passed, 2 skipped in 1.92s**；2 个 skip 条件见 `:35`（需权重已缓存且无 CUDA）、`:44`（需无 GPU）[验证] |
| `tests/test_infer_file.py`（37 行） | 离线整段推理与窗口 runtime 数值一致、任意长度/文件往返、时长裁剪 | pytest + numpy + torch + **Falcon 权重**（模块级 `pytestmark`，`weights_source != "cached"` 时整体 skip，`:13`） | ✅（R3 修订）本环境 HF 缓存已有 small/big 权重 → 可运行；无权重机器上为 skip | R1：**2 skipped in 2.15s**（无权重）；R3 复跑：**2 passed in 68.28s**（权重已缓存，§8.2）[验证] |
| `tests/test_server.py`（149 行） | 服务协议：WS wire 往返/时间戳（HOP/GUARD）、非法 payload 拒绝（11 类）、OutputSlicer 交叉淡化与跳变重置、跨域 Origin 拒绝（HTTP 403 / WS 异常）、坏 JSON 不断连、id 布尔/精度拒绝、variant 加载错误、坏 `/api/load`、CSP/WASM MIME 与本地 vendor 资产（`:35-149`） | pytest + numpy + **fastapi/httpx（TestClient）** + **Pillow**；全部用 `StubEngine`（`:29-34`），不联网、无权重 | ✅ **可运行（R3 修订）**：注入仓库内最小 PIL stub（`docs/verification/scripts/pil_stub/`，仅 `Image.new/open/save/convert`）+ `-p no:typeguard` 后协议用例全部 PASSED（单跑收集 **22** 项 = 9 个 test 函数 + parametrize 展开）；真实 JPEG/PNG 解码未验证（Pillow 职责） | **22 passed**（单跑 0.91 s）[验证]；四文件合计评审基线 33 passed/4 skipped、本轮 35 passed/2 skipped（§8.2） |

**R3 可运行性修订（P1-2 响应）**：缺 Pillow 时可用仓库内最小 PIL stub 完整跑协议逻辑：
```bash
PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python -m pytest studio/tests -q -p no:typeguard
```
评审基线（当时无权重缓存）**33 passed, 4 skipped**；`test_server` 协议用例全过（评审文字口径 19 项，按 pytest 收集实为 **22** 项 = 9 函数 + 参数展开 11+4；33 = 22 + test_weights 7 + test_device 4）。本轮复跑（HF 缓存已有 small/big 权重）**35 passed, 2 skipped**。`-p no:typeguard` 用于绕开 `docs/verification/stubs/typeguard.py` 遮蔽真实 typeguard 包导致的插件加载失败（R1 记录见 §8.2）。真实 JPEG/PNG 解码未执行。

补充：`tests/serve_browser.py`（14 行）不是 pytest 用例，而是 QA 服务器：`create_app(Engine())`（`:10-11`，默认 `autoload=True`，见 `server.py:182,187`）会在启动后台加载模型并**解析/下载权重**（`falcon_runtime.py:58-78`），然后 `uvicorn.run(... port=int(os.environ.get('FALCON_QA_PORT', 7866)))`（`:14`）。README 示例用 `FALCON_QA_PORT=8766`（`studio/README.md:187`）覆盖默认值，两者不矛盾，但默认 7866 与示例 8766 并存易混淆（P2）。

**浏览器 QA（手工）**：`tests/browser/index.html` 加载 `test.js`；`test.js:7` 起真实 `Worker('/static/video-worker.js')`、`:24` 用本地 `portrait.jpg`、`:59` 输出 `library:'MediaPipe 1.0.1'` 的报告；断言"真实人脸稳定检测 ≥5 帧、ROI 合法、短暂丢脸不暂停输出、长丢失后重新锁定、手动模式仍出 crop"。`portrait.jpg`（176 KB）与 vendor `manifest.json:3` 的 MediaPipe 1.0.1 版本号一致；文件来源"官方测试人像"无法离线验证。当前环境**无浏览器/无 MediaPipe/无权重**，未运行。

### 2.5 依赖文件四份

| 文件 | 行数 | 内容与策略 | 关键证据 |
|---|---|---|---|
| 根 `requirements.txt` | 29 | **训练/评测侧**；绝大多数**不锁版本**（`torchaudio,torchvision,pytorch-lightning,numpy,scipy,pandas,pyyaml,soundfile,tqdm,rich,packaging,timm,librosa,torch-complex,typeguard,espnet,torch-optimizer,opencv-python,matplotlib,fast-bss-eval,torch-mir-eval,pypesq,pystoi,ptflops`），唯一下限 `torch>=1.13`；文件头自述 "TODO: pin exact versions from the training machine ... they were not recorded with the checkpoints" | `:1-4` 注释、`:5` torch>=1.13、`:17` timm、`:25-29` 指标/算力包；**无 `huggingface_hub`**（`grep -ci hugging` = 0） |
| `studio/requirements.txt` | 11 | **Studio 侧**：**范围约束**（"tested minimum versions; keep compatible major versions"）：`torch>=2.14.1,<3`、`numpy>=2.4.6,<3`、`PyYAML>=6.0.3,<7`、`einops>=0.8.2,<1`、`fastapi>=0.142.2,<1`、`uvicorn[standard]>=0.54.0,<1`、`Pillow>=12.3.0,<13`、`huggingface_hub>=2.2.0,<3` | `:1-11`；Python 版本约束不在文件里，而在 `check_environment.py:18-19` / `bootstrap.sh:9` / README |
| `studio/requirements-lock.txt` | 110 | `uv pip compile requirements.txt --python-version 3.11 --output-file requirements-lock.txt` 生成的**精确 pin**（39 个包，含传递依赖）；`torch==2.14.1`、`uvicorn==0.54.0`、`fastapi==0.142.2`、`huggingface-hub==2.2.0`、`websockets==17.2`、`httpx2==2.15.1` 等；`bootstrap.sh:71-75` 优先使用它 | `:1-2` 生成命令；逐个包 `==`；39 个包 [验证] |
| `studio/requirements-dev.txt` | 3 | `-r requirements.txt` + `pytest==9.1.1` + `httpx==0.28.1`；供 `studio/README.md:182` 的 pytest 命令 | `:1-3` |

**一致性验证**：用脚本把 lock 与 `requirements.txt` 的 8 个下限逐条比对（名称大小写/下划线归一化），8/8 落在 `[min,<max)` 内 → **lock 与 requirements 自洽**。[验证] 版本策略差异（训练侧不锁、Studio 锁）本身是合理取舍，但训练侧"未记录训练机版本"（`:3-4` TODO）意味着论文数字不可精确复现，属已知风险。

**可用性旁证**：本环境可访问的包索引对 studio 的**每个精确下限版本**（torch 2.14.1、numpy 2.4.6、PyYAML 6.0.3、einops 0.8.2、fastapi 0.142.2、uvicorn 0.54.0、Pillow 12.3.0、huggingface-hub 2.2.0、pytest 9.1.1、httpx2 2.15.1）均返回 HTTP 200（`/pypi/<pkg>/<ver>/json`），即这些下限在本环境中可满足；`pip install` 被本轮约束禁止，未做真实安装。[验证]

### 2.6 `.github/workflows/pages.yml`（35 行）

| 行 | 内容 | 结论 |
|---|---|---|
| `:1-2` | 注释：发布 `project_page/` 到 `https://cslikai.cn/Falcon/`；demo 音视频由 `https://oss.cslikai.cn/demos/` 提供，不在仓库 | 与 `project_page/index.html` 实际引用一致（35 条外部 URL，31 唯一）[验证] |
| `:3` | `name: Project page` | |
| `:5-9` | 触发：`push` 到 `main` 且路径命中 `project_page/**` 或本 workflow；另有 `workflow_dispatch` | **只有 main + 相关路径**；README 改动不会重新部署（合理） |
| `:11-14` | 权限：`contents: read`、`pages: write`、`id-token: write` | 符合 GitHub Pages Actions 官方要求 |
| `:16-18` | 并发：`group: pages`、`cancel-in-progress: true` | 避免并发部署 |
| `:20-27` | 单 job `deploy`；`if: ${{ !github.event.repository.private }}`——**私有仓库不部署**；`environment: github-pages` + `url: ${{ steps.deployment.outputs.page_url }}` | 与 `00-project-map.md:39` 的"私有仓库不部署"一致 |
| `:28-35` | 步骤：`actions/checkout@v4`、`configure-pages@v5`、`upload-pages-artifact@v3`（`path: project_page`）、`deploy-pages@v4` | artifact 是**整个 `project_page/` 目录**（index.html + figures/ + logo.png） |

风险/未解：workflow **没有 `cname` 输入，`project_page/` 里也没有 `CNAME` 文件**（`find project_page -type f` 仅 5 个文件），因此自定义域名 `cslikai.cn/Falcon/` 只能依赖仓库 Settings → Pages 的外部配置，离线不可验证（Q8）。[代码/未验证]

### 2.7 `project_page/index.html`（801 行，静态单页）

结构（行号为当前文件）：

- `:1-14` `<!DOCTYPE html>`、`lang="en"`、viewport、title、`favicon=logo.png`；**外部依赖**：Google Fonts（`:10-12`）+ Font Awesome CDN（`:13`）；无本地 JS/CSS 文件。
- `:15-260` 全部 CSS 内联（`:root` 变量、排版、表格、demo 卡片、响应式等）。
- `:262-293` 导航 + hero：会议 badge、标题、作者、链接按钮（Code=GitHub、🤗 Models、Live demo=`https://u464727-dmja-...:8443/`）、4 个 stats（7.7 dB / 16.4 dB / 0.5M / 51.3 ms）。
- `:295-687` 8 个 section：abstract/contributions/realssa/method/demos/results/ablation/citation；`method` 用 `figures/{Overall_Pipeline,Separator,TFRB}.pdf.png`（`:385,402,412`）。
- `:419-566` 4 个 demo 场景（small/medium/large/outdoor），每个含 `<video>`（`https://oss.cslikai.cn/demos/<scene>/s1.mp4`）+ base `<audio>` + 5 个方法按钮（base/falcon/swiftnet/dolphin/TFGrid 的 `.wav` 全在 OSS）。
- `:567-639` 结果表（quality/efficiency/real-world），`:640-665` 消融表（downsampling/RNN/contrastive/τ），`:666-686` BibTeX。
- `:687-801` 内联 JS：滚动进度条、复制 BibTeX、tab 切换、demo 音频与视频秒级同步（`hardSync`，阈值 0.15 s）、lightbox、DOMContentLoaded 初始化。

**一致性核对**：
- 相对资源 `logo.png`、`figures/*.pdf.png` 全部存在；页内锚点 `#abstract...#citation` 全部有对应 id。[验证]
- 外部媒体**统一在 `oss.cslikai.cn/demos/<scene>/`**：`small/medium/large/outdoor` × `{s1.mp4, base,falcon,swiftnet,dolphin,TFGrid .wav}`；`large` 的 Dolphin 为 **`Dolphin.wav`**（`:517`）而其余为小写 `dolphin.wav`；2026-10-30 实测 `large/Dolphin.wav`=200、`large/dolphin.wav`=404 ⇒ **非缺陷，禁止改成小写**。
- 页面数字与 README 结果表/效率表逐项一致：7.7/8.2、14.1/14.7、16.4/16.7、10.5/11.8、DNSMOS 3.276；0.5M/10.6G/1361.2/51.3/125.3/364.7；摘要里的 91.9%/20.2×/11.1× 可分别由 (6.2-0.5)/6.2、213.8/10.6、567.8/51.3 复算 ≈91.9%/20.17/11.07。[代码]
- 页面插图与 README 插图是**两份独立渲染**（`project_page/figures/*.pdf.png` 209/75/105 KB vs `assets/Overall_Pipeline.png` 165 KB、`Separator.png` 105 KB、`TFRB.png` 64 KB），内容对应但文件不同 → 双源维护风险（P2）。
- 页面依赖 Google Fonts + cdnjs：与 Studio 的 "no CDN"（`studio/README.md:20`）**不冲突**（该承诺只针对 Studio 界面），但项目页离线/受限网络下样式会退化（P2）。

### 2.8 `Experiments/checkpoint/Falcon/{conf.yml,best_k_models.json}`

**`conf.yml`（168 行）**：训练时由 `audio_train.py:77-84` 写出的实验配置快照（`os.getcwd()/Experiments/checkpoint/<exp_name>/conf.yml`），评测（`audio_visual_test.py`）读取它来构建 audio/video 模型、取得数据配置和 `exp.exp_name`。关键行：

| 行 | 内容 | 角色/问题 |
|---|---|---|
| `:1-87` | `audionet` 完整配置（audio_params/enc_dec_params/bn/fusion/mask 等，small 维度：`out_chan:112`、`hid_chan:64`、`spk_embed_dim:64`） | 与 `configs/Falcon.yml` 的 `audionet_config` 键值基本一致 |
| `:11` / `:73` | `audio_params.causal: true`（`:11`）与 `video_params.causal: true`（`:73`） | `configs/Falcon.yml` 无这两个嵌套键（`:45` 是顶层 `audionet_config.causal`，`:49` 是 `enc_dec_params.causal`；均与 root 配置共有）；但 `Falcon.__init__` 会强制写回 `params['causal']=self.causal`（`look2hear/models/Falcon.py:1314-1315`），**无实际影响** [代码]（R3 修正行号，原写 `:45`） |
| `:88` | `audionet_name: SpeakerContrastiveV2` | **P0**：`look2hear.models` 未导出该类（见 §4.7 D2） |
| `:90-103` | `datamodule.data_config`，其中 `:99-102` 三个目录是字面量 `${DATA_ROOT}/...` | 需用户手工改（README 已说明"Evaluation reads the data paths from that conf.yml (edit them)"） |
| `:105` | `exp.exp_name: Falcon` | 评测据此拼 `Experiments/checkpoint/Falcon/` |
| `:119-120` | `main_args.exp_dir: ${ROOT}/Experiments/checkpoint/Falcon` | 被 `audio_visual_test.py:196-198/273-275` 的 `cwd` 重算取代，未实际使用 |
| `:126` | `options.ckpt_path: ''` | 与 CLI `--ckpt_path` 同名但不同层（评测读 `arg_dic["ckpt_path"]`，不是这里） |
| `:127` | `options.conf_dir: ${DATA_ROOT}/small/configs/small_1.yml` | 历史痕迹，未使用 |
| `:142-159` | `training.gpus: [0..9]` 等 | 评测只用到 `exp_name`/data/audionet/videonet，training 段不生效 |

**`best_k_models.json`（6 行/wc，5 条记录）**：`ModelCheckpoint(monitor="val_loss/dataloader_idx_0", mode="min", save_top_k=5)`（`audio_train.py:139-147`）的 `best_k_models` 快照，训练结束时由 `audio_train.py:184-186` 写入 `exp_dir/best_k_models.json`；评测 `_find_best_ckpt`（`audio_visual_test.py:78-107`）优先读取它并 `min(best_k, key=best_k.get)` 选"验证损失最低"的 ckpt。

问题（P1，详见 §4.7 D4）：
- 5 个 key 都是**字面量** `${ROOT}/Falcon/epoch={714,735,738,741,742}.ckpt`（`best_k_models.json:2-6`），`os.path.exists("${ROOT}/...")` 恒为 False（实测 5/5 False）→ 选择逻辑必然穿透到 `last.ckpt`/`epoch=*.ckpt`，而仓库里没有任何 ckpt（`.gitignore:24` 忽略 `*.ckpt`；`find Experiments -type f` 只有这两个文件）。
- 文件格式与代码生成的格式不符：`audio_train.py:185-186` 用 `json.dump(..., indent=0)` 且 key 是 `ModelCheckpoint` 写入的**绝对路径**；提交文件是缩进 2、`${ROOT}` 占位 → [推断] 该文件是从原始训练机手工拷贝/占位而来，不是本仓库脚本产出。
- 数值为负（-9.98…），与 `mode="min"` 及负 SI-SNR 损失一致；最小项是 `epoch=714`（-9.984025）。[代码]

结论：这两个文件的**角色**是"配置快照 + ckpt 选择元数据"，用于让 `audio_visual_test.py` 在训练后自动找最优 ckpt；但它们对**未训练的新克隆不可用**，README `:174` 的自动选择描述需要补充前置条件（先训练，或显式 `--ckpt_path`）。

### 2.9 `.gitignore`（根 35 行 + `studio/.gitignore` **14 行**）

根 `.gitignore`：Python 缓存（`__pycache__`、`*.py[cod]`、`.pytest_cache`）、`.DS_Store`、日志（`*.log`、`wandb/`、`runs/`、`lightning_logs/`）、数据/输出（`data/`、`datasets/`、`outputs/`、`results/`）、权重（`pretrain_zoo/`、`*.ckpt/pt/pth/pth.tar/onnx`）、压缩包。`studio/.gitignore` 额外覆盖 `.venv/`、`venv/`、`*.py[cod]`、`*.ckpt/pt/pth`、`recordings/`、`outputs/`。

- [验证] `git check-ignore`：`studio/.venv/bin/python` 被 `studio/.gitignore:1` 命中（**不会**被误提交，Lead 提示的 `.venv` 隐患已被 studio 级规则解决）；`Experiments/checkpoint/Falcon/last.ckpt` 被根 `:24 *.ckpt` 命中。
- **缺口（P2）**：`audio_train.py:153-154` 会创建 `Experiments/tensorboard_logs/<exp_name>/`，根 `.gitignore` 的 `lightning_logs/`/`runs/`/`*.log` 都不覆盖它 → 训练后 `git status` 出现未跟踪目录（`git check-ignore` 实测 NOT IGNORED）。[验证]
- 另有：根 `.gitignore` 未列 `.venv/`，但 Studio 目录自带忽略文件已覆盖；若未来在仓库根建 venv 仍会漏。

### 2.10 四份 README

| 文件 | 行数 | 结构 | 相互链接 |
|---|---|---|---|
| `README.md` | 251 | logo/标题/作者/NeurIPS/链接行 → News → Overview（3 张图）→ RealSSA → Installation（两条产品线两套环境）→ Model weights（表）→ Inference（Studio + 离线 + Python API）→ Training and evaluation → Results（质量表 + 效率表）→ Repository layout → Citation/License/Acknowledgements | EN↔ZH 互链、`studio/README.md`、`studio/README_zh.md`、HF×2、Project page、Live demo、`PROVENANCE.md`、`LICENSE` |
| `README_zh.md` | 253 | 与 EN 逐节镜像（代码块除注释与示意图文字外 ASCII 等价 [验证]） | 同上（反向） |
| `studio/README.md` | 187 | logo → 标题/在线 demo 警告 → 总述 → Quick start 7 步 → Models → Mouth tracking → Meetings（接线图/表格）→ How the live pipeline works（阶段表）→ Command line（launch 参数、离线、Python API）→ Troubleshooting（13 条）→ Project layout → Tests | 根 README、`README_zh.md`、`../assets/logo*.png`、HF、BlackHole、pytorch.org、MediaPipe |
| `studio/README_zh.md` | 187 | 与 EN 镜像（`:182-187` 测试段与 EN 行号一致） | 同上 |

[验证] 4 个文件所有相对 Markdown 链接（`[...](...)` 非 http/# 目标）均解析到存在的路径，**0 缺失**；代码块中出现的命令路径（`scripts/launch.sh`、`tests/...`、`configs/Falcon.yml` 等）也均存在。EN/ZH 的差异仅限翻译文字与 ASCII 示意图里的标注（属预期）。四份 README 与代码的**逐条偏差**见 §4.7。

## 3. 关键调用链与数据流

### 3.1 启动链（Finder → 浏览器）逐段解析

```
[Finder 双击]
  └─ studio/Start Falcon.command:2             APP_DIR = studio/
     :3                                        "$APP_DIR/scripts/launch.sh" "$@"
        └─ launch.sh:4-5                       SCRIPT_DIR=studio/scripts, APP_DIR=studio
           :6                                  PYTHON_BIN="$(bootstrap.sh)"   ← 只取 stdout
              └─ bootstrap.sh:12-18            选 ENV_DIR:
                   FALCON_VENV > studio/.venv(合法) > ~/Library/Caches/FalconStudio/venv
                 :24-27                       若 python 合法 && check_environment --quiet 通过 → 直接返回
                 :29-33                       已有目录但 python 不合法 → 报错退出（不删除）
                 :34-60                       找源 Python（FALCON_PYTHON / python3.11,3.12,3.10,python3 / Finder 兜底路径）
                 :65-68                       创建 venv
                 :70-83                       安装：requirements-lock.txt 优先；uv 否则 ensurepip+pip
                 :84                          再跑 check_environment.py（失败则 set -e 中止）
                 :85                          stdout: <venv>/bin/python
           :9                                 $PYTHON_BIN -c '...weights_source("small")'
              └─ falcon_runtime.py:48-56      huggingface_hub.try_to_load_from_cache → "cached"/"huggingface"/None
           :10-15                           非 cached → 打印"首次下载约 45MB" → resolve_checkpoint("small")
              └─ falcon_runtime.py:58-78      hf_hub_download(repo="JusperLee/Falcon", file="falcon.pt")
                 （失败 → FileNotFoundError("... Set HF_ENDPOINT ...") → launch.sh:12-13 提示镜像并 exit 1）
           :19                                cd studio/
           :20                                exec $PYTHON_BIN server.py "$@"
              └─ server.py:289-297            CLI: --host{127.0.0.1,localhost} --port --device{auto,cuda,mps,cpu}
                                                    --model{JusperLee/Falcon[, -Big]} --no-browser
                 :299-303                    variant_for(--model) 校验
                 :304-314                    未指定 --port：对 7860..7899 逐个 bind 测试，取第一个空闲
                 :315-316                    显式端口范围 1024..65535
                 :318                        url = http://<host>:<port>
                 :319-329                    后台线程轮询 GET /api/status，返回 app=="falcon-studio" 时 webbrowser.open(url)
                                              （最多 60 次 × 0.5 s = 30 s）
                 :332-333                    uvicorn.run(create_app(Engine(device, variant)), ...)
                    └─ server.py:182-196      create_app(autoload=True) → lifespan → Engine.start_load()
                       └─ server.py:126-146   后台线程 _load → FalconRuntime(variant).load(device)
                          └─ falcon_runtime.py:133-200
                               142-158        resolve_checkpoint → 读 studio/configs/<variant>.yml → 校验
                                              architecture/videonet 名 → 构建 audio/video model
                               160-172        torch.load(weights_only=True)、拆分 audio_model.*/video_model.*
                                              → 两者都必须非空（:164-165）→ strict=True 加载（:168-169）
                               175-180        FALCON_CPU_THREADS（默认 4）→ eval + requires_grad_(False)
                               181-196        pick_device(device)；auto 时 GPU 预热失败 → warning + CPU 重试
```

要点：
- **端口/浏览器/设备逻辑不在 `launch.sh`**，而在 `server.py`/`falcon_runtime.py`；`launch.sh` 只是传参（`launch.sh:20`）。任务书里"launch.sh 内选 venv/端口"的描述需按此修正。
- `--device` 的语义链：`launch.sh "--device cuda"` → `server.py:294` → `Engine(device)`（`:332`）→ `FalconRuntime.load`（`:133`）→ `pick_device`（`falcon_runtime.py:81-92`）；`auto` = CUDA→MPS→CPU，且 GPU **预热**失败会回退 CPU（`:183-196`）。
- `weights_source` 的三种返回值含语义：`"cached"`/`"huggingface"`（hub 已装但未缓存）/`None`（hub 未装）。`launch.sh:9` 只区分 `"cached"` 与否；`None` 时 `resolve_checkpoint` 会给出"huggingface_hub is not installed"的 FileNotFoundError，但 `launch.sh:12` 统一显示"Download failed / 用镜像"，**错误归因不准确**（P2）。
- 首次启动权重下载是**两段式**：`launch.sh` 先下 small（保证 `--model Falcon` 默认可跑）；若用户启动后切 Falcon-Big，`server.py` `/api/load` → `Engine.start_load` 线程内再按需下载 big（`server.py:150-160` 调用 `FalconRuntime.load`）。

### 3.2 `bootstrap.sh` 环境选择决策树

```
FALCON_VENV 已设置? ──是──> ENV_DIR=FALCON_VENV
        │否
studio/.venv/bin/python 合法(3.10≤v<3.13 且可执行)? ──是──> ENV_DIR=studio/.venv
        │否
ENV_DIR=$HOME/Library/Caches/FalconStudio/venv

ENV_DIR/bin/python 合法 && check_environment.py --quiet 通过?
   ├─是 → 复用：打印路径，exit 0（:24-27）
   └─否
       ENV_DIR 已存在?
         ├─是（且 python 不合法）→ 报错，提示换 FALCON_VENV，exit 1（:29-33）
         └─否
              FALCON_PYTHON 设置?
                ├─是：必须合法，否则报错 exit 1（:34-38）
                └─否：依次找 python3.11/3.12/3.10/python3（:39-47）
                      再找 Homebrew/官方安装器/uv 的硬编码路径（:48-60）
              找到? ├─否 → 中文指引 exit 1（:61-64）
                    └─是 → mkdir + venv（:65-68）→ 安装依赖（:70-83）→ 校验（:84）→ 打印路径（:85）
```

### 3.3 `doctor.sh` 链

`bootstrap.sh`（可能装依赖）→ `check_environment.py`（版本+import 全量报错）→ heredoc：平台/CPU、torch 版本、MPS built/available、CUDA available+GPU 名、每个 variant 的 `weights_source`；`small` 若连 huggingface_hub 都没有则 exit 1。**它不检查**：媒体设备（摄像头/麦克风）、浏览器、BlackHole、`studio/web/vendor` 完整性——这些由 README 的 Troubleshooting 手工排查。

### 3.4 训练 / 评测 / 算力三条命令行链（与配置文件、元数据的配合）

```
训练：python audio_train.py --conf_dir configs/Falcon.yml
  audio_train.py:199-203   yaml.safe_load(conf_dir) → prepare_parser_from_dict → arg_dic
  :43-58   datamodule = AVSpeechDataModuleWithNear(**data_config)（签名见 datas/avspeech_dataset_near.py:237-252）
           audio model = getattr(look2hear.models, audionet_name)(**audionet_config)   ← FalconSpeakerContrastive ✓
  :61-63   video model = FRCNNVideoModel(**videonet_config)，pretrain=pretrain_zoo/frcnn_128_512.backbone.pth.tar
  :66-74   optimizer=adamw(lr 1e-3, wd .01) / scheduler=ReduceLROnPlateau(patience 15,factor .5)
  :77-84   写 Experiments/checkpoint/<exp_name>/conf.yml（注意：在模型构造之后写，Separator.__init__ 已 pop 掉 repeats/shared）
  :139-150 ModelCheckpoint(monitor=val_loss/dataloader_idx_0, mode=min, save_top_k=5, save_last) + EarlyStopping(patience 30)
  :161-173 pl.Trainer(devices=gpus, accelerator=cuda/None, strategy=DDPStrategy(find_unused_parameters=True),
                      gradient_clip_val=5.0, logger=comet_logger ← ✗ NameError)
  :184-192 写 best_k_models.json（绝对路径）、best_model.pth
评测：python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml --gpus 0
  :315-329 读 conf → arg_dic["train_conf"]；hf:// 前缀 → huggingface_hub.hf_hub_download
  :271-281 CUDA_VISIBLE_DEVICES + mp.spawn（单卡直接 process_worker）
  :196-201 exp_dir=cwd/Experiments/checkpoint/<exp_name>；ckpt_path = CLI --ckpt_path 或 _find_best_ckpt(exp_dir)
  :78-107  best_k_models.json(min) → last.ckpt → epoch=*.ckpt → 否则 FileNotFoundError
  :119-168 torch.load ckpt → getattr(look2hear.models, conf.audionet_name) ← ✗ SpeakerContrastiveV2
  :298     results/metrics_combined.csv（各 rank 先写 metrics_rankN.csv 再合并删除）
算力：python evaluated_mac_params.py
  :39      硬编码 open("configs/Falcon.yml")（必须 cwd=仓库根）
  :44-53   audio model + video model；:77-96 .cuda() + torch.randn(1,32000)、mouth (1,1,50,64,64)
  :122-123 输出 MACs/Params（G/M）
```

### 3.5 CI 发布链

`push main`（路径 `project_page/**` 或 `.github/workflows/pages.yml`）/ 手工 dispatch → `deploy` job（公开仓库才跑）→ checkout → configure-pages → upload artifact(`project_page/`) → deploy-pages → `https://cslikai.cn/Falcon/`（自定义域名依赖仓库设置；媒体 URL 指向 `oss.cslikai.cn`，不随仓库发布）。

### 3.6 测试执行链

- Node：`node tests/check_i18n.mjs` 等 8 个文件直接跑（无构建、无 npm 依赖）；`check_tracking_ui.cjs` 会用 `vm` 跑真实 `web/app.js`，并用 `web/index.html` 的 id 集合校验 DOM 契约。
- pytest：`pip install -r requirements-dev.txt` → `python -m pytest tests/`；`test_weights/test_device` 无需网络/权重；`test_infer_file` 需已缓存权重否则整体 skip；`test_server` 需 fastapi TestClient + Pillow。
- 浏览器：`FALCON_QA_PORT=8766 python tests/serve_browser.py` → Chrome 打开 `/__test/` → 点按钮；使用本地 vendor MediaPipe + `portrait.jpg`，但 `create_app(Engine())` 的 autoload 会后台尝试加载/下载模型（见 §2.4）。

## 4. 关键实现细节与设计动机

### 4.1 "stdout 单值"脚本组合模式

`bootstrap.sh:2` 明确 "stdout contains only the selected Python path; messages go to stderr"，`launch.sh:6`/`doctor.sh:5` 用命令替换获取路径。好处：单一事实来源（环境选择只有一处）；风险：任何误输出到 stdout 都会污染路径（当前 5 个脚本一致遵守）。`check_environment.py --quiet` 也是为这个契约服务（`bootstrap.sh:24`）。

### 4.2 为什么把 venv 放到 `~/Library/Caches/FalconStudio/venv`

README 解释 "outside iCloud"（`studio/README.md:115`）。动因 [推断]：macOS 用户常把 `~/Documents` 放 iCloud，venv 内的符号链接/大文件会被同步；Cache 目录不参与 iCloud 且可被系统清理（可重建，`bootstrap.sh` 会自动重装）。同时优先复用 `studio/.venv` 方便开发者/CI。`requirements-lock.txt` 优先（`bootstrap.sh:71-75`）是为了可复现；lock 由 uv 对 Python 3.11 编译（`requirements-lock.txt:2`），而 bootstrap 允许 3.10–3.12，故 3.12 安装 lock 时依赖平台 wheel 的回退能力（`uv pip compile` 输出无 `python_version` 标记）→ 轻微未验证风险（Q10）。

### 4.3 设备探测与 GPU 回退设计

`pick_device`（`falcon_runtime.py:81-92`）只做"是否可用"的静态判断；真正的脆弱点是某些 GPU 驱动/算子能通过 `is_available` 但跑不了模型，所以 `load()` 在 `auto` 模式下**先真实预热一次**（`_warmup`，`:202-205`），失败则警告并重试 CPU（`:183-196`）。这解释了 README "auto tries CUDA, then Metal, and falls back to the CPU if the GPU warm-up fails"（`studio/README.md:115`）不是空话，而是有对应实现；也解释了 `test_device.py:35` 那个"报告可用但实际不可用"的测试为何存在。显式 `--device cuda/mps` 不可用时直接 `RuntimeError`（`:86-90`），不做静默回退——与 README "严格使用该设备"的语义一致。

### 4.4 端口选择与浏览器自动打开

`server.py:304-314` 用"bind 后立即关闭"探测 `7860..7899`；这是常见但存在 TOCTOU 竞态（探测与 uvicorn 真正 bind 之间有窗口）[推断，P2 风险]。未指定端口时 `parser.error` 明确报 "No free local port between 7860 and 7899"（`:314`）。显式 `--port` 必须 1024–65535（`:315-316`），且**不再做占用检查**（由 uvicorn 自行报错）。浏览器线程轮询 `/api/status` 直到 `app == "falcon-studio"`（`:319-329`），注意 Engine 在模型未加载时也会返回 status（`server.py:105-116` 的 `ready:false`；R3 修正行号），所以**浏览器通常在模型就绪前就打开**，这是预期 UX（页面自带加载动画）。`--no-browser` 可关闭（`:296,319`）。

### 4.5 本地安全边界（与 README 承诺相关）

`server.py:175-179` `same_origin`：无 `Origin`（命令行/测试）放行；有 Origin 时要求 `origin.netloc == Host` 且 hostname ∈ {localhost,127.0.0.1,::1}。HTTP 中间件对**所有**请求检查并返回 403（`:198-203`）；WS 单独检查（`:235-237`）。`TrustedHostMiddleware` 只允许 `127.0.0.1/localhost/[::1]/testserver`（`:196`）。（R3 修正行号；该 loopback 限制与 README 公网在线 demo 的矛盾见 D22）CSP `script-src 'self' 'wasm-unsafe-eval'`、禁 `unsafe-eval`（`:210`）。这与 `studio/README.md:177`（"restricted to same-origin local pages"）一致；`test_server.py:86-96` 覆盖 403/WS 拒绝。

### 4.6 `Experiments/checkpoint` 两个文件的"来历"与可信度 [推断]

- `conf.yml` 的 `audionet_name` 与 `configs/Falcon.yml` 不一致、且包含 `configs/Falcon.yml` 没有的嵌套 `causal` 键、`options.conf_dir` 指向 `${DATA_ROOT}/small/configs/small_1.yml`——这些痕迹表明它是**原始（未发布）训练仓库某次实验**的配置快照，而不是本仓库 `audio_train.py` 用 `configs/Falcon.yml` 训练后写出的文件（否则 `audionet_name` 应为 `FalconSpeakerContrastive`）。
- `best_k_models.json` 的 `${ROOT}` 占位与缩进同样不符合本仓库 `audio_train.py:184-186` 的写出格式（应为绝对路径、`indent=0`）→ 也是从原始环境带过来的**元数据展示**，不是可直接执行的文件。
- 因此 README 把它们描述为 "config snapshot and best-checkpoint metadata"（`README.md:220`）是准确的；但 README `:174` 的"自动选 best ckpt"流程在**本次发布的仓库**中无法直接执行，需要先训练或显式 `--ckpt_path`（P1，建议 README 补充一句）。

### 4.7 ★ README 与代码一致性审计：偏差清单（重点交付）

> 行号均为当前 commit；`严重度`：P0=文档命令必然失败/事实错误；P1=重要遗漏或误导；P2=细节。
> 中文 README 与英文逐条对应（行号见"位置"列），偏差同时存在于两版；`README_zh.md` 行号单列。

| # | README 位置 | README 陈述（摘） | 代码/仓库证据 | 偏差描述 | 严重度 |
|---|---|---|---|---|---|
| D1 | `README.md:162`；`README_zh.md:164` | `python audio_train.py --conf_dir configs/Falcon.yml` 可训练 | `audio_train.py:170` `logger=comet_logger`，全仓库 `.py` 仅此一处（`grep -rn comet_logger --include=*.py` = 1 行），无定义/无 import；根 `requirements.txt` 无 `comet_ml` | 构造 `pl.Trainer` 时 `NameError: name 'comet_logger' is not defined`，训练无法启动 | **P0** |
| D2 | `README.md:165,168`；`README_zh.md:167,170` | `python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml ...` 可评测 | `Experiments/checkpoint/Falcon/conf.yml:88` `audionet_name: SpeakerContrastiveV2`；`look2hear/models/__init__.py:1-2` 只导入 `Falcon`,`FalconSpeakerContrastive`（`__all__` 还写错成 `AVTIGER_causal`，`:4-6`）；`audio_visual_test.py:129` `getattr(look2hear.models, name)`；本环境实测 `hasattr(...,'SpeakerContrastiveV2')==False` | 构建 audio model 时 `AttributeError: module 'look2hear.models' has no attribute 'SpeakerContrastiveV2'`；两条 README 评测命令（含 `--ckpt_path hf://`）都会在此处失败。**首次失败点取决于数据是否已配置**（R3 补，回应评审 P2-4）：`process_worker` 先 `datamodule.setup()`（`audio_visual_test.py:187-188`）→ `AVSpeechEvalDataset` 打开 `<test_dir>/mix.json`（`avspeech_dataset.py:142-150`），而 conf.yml 的目录是字面量 `${DATA_ROOT}`（`:99-102`）→ 未按 README 先改数据路径时首个异常是 `FileNotFoundError`；数据就绪后才在 `_load_models_from_ckpt`（`audio_visual_test.py:201-202`）触发 AttributeError。 | **P0**（结论不变：命令不可运行） |
| D3 | `README.md:168`；`README_zh.md:170` | `--ckpt_path hf://JusperLee/Falcon/falcon.pt` | `audio_visual_test.py:323-326` 在分支内 `from huggingface_hub import hf_hub_download`；根 `requirements.txt`（29 行）**无 huggingface_hub**（`grep -ci hugging`=0） | 只按 README 安装训练环境时该命令 `ModuleNotFoundError: huggingface_hub` | P1 |
| D4 | `README.md:174`；`README_zh.md:176` | 不传 `--ckpt_path` 时自动加载 best：`best_k_models.json` 最低项 → `last.ckpt` → 最后 `epoch=*.ckpt` | `audio_visual_test.py:78-107` 逻辑存在；但 `best_k_models.json:2-6` 的 key 是字面量 `${ROOT}/Falcon/epoch=*.ckpt`（`os.path.exists` 不展开 → 5/5 False）；仓库无任何 `*.ckpt`（`find Experiments -type f` 仅 2 文件；`.gitignore:24` 忽略 `*.ckpt`） | 新克隆上不传 `--ckpt_path` → 最终 `FileNotFoundError`（`audio_visual_test.py:103-107`）；README 未提示"需先训练/权重文件" | P1 |
| D5 | `README.md:89`；`README_zh.md:91` | 训练/评测 "PyTorch >= 1.13" | 根 `requirements.txt:5` `torch>=1.13`；但 `README.md:28`/`README_zh.md:30` badge 写 "PyTorch 2.x"；`pytorch-lightning`/`timm` 未锁（`requirements.txt:8,17`）；代码用 `torch.load(..., weights_only=...)`（`audio_visual_test.py:119`、`studio/falcon_runtime.py:160`） | 同一份 README 内部矛盾（2.x vs >=1.13）；未锁的 PL/timm 实际会拉更高 torch，"1.13" 不是验证过的组合 | P1 |
| D6 | `README.md:171`；`README_zh.md:173` | `python evaluated_mac_params.py` | `evaluated_mac_params.py:48,77,78,80,81,86` 多处 `.cuda()`；`:39` 硬编码 `open("configs/Falcon.yml")`；`:36` `--exp_dir` 定义了但从未使用 | 只能在**有 CUDA、cwd=仓库根**的机器跑；README 未说明前置条件；CPU-only 环境直接失败 | P1 |
| D7 | `README.md:193`；`README_zh.md:195` | 效率表输入为 "2 s, 16 kHz clip with 50 mouth frames" | `evaluated_mac_params.py:81` 构造 `torch.randn(1,1,50,64,64)`（64×64），而文档/推理管线是 96×96（`falcon_runtime.py:27` `WINDOW_FRAMES=50`，`preprocess_mouth` 取 96→88，`:95-105`） | 复现脚本嘴部输入尺寸与文档管线不一致（因 MACs 只统计 audio model，mouth_embedding 固定 (1,512,50)，数值不受影响；但增加了"看起来不一致"的复现成本） | P2 |
| D8 | `README.md:105`；`README_zh.md:107` | "A local file can be passed with `--checkpoint`" | 仅 `studio/infer_file.py:91` 与 `FalconRuntime(checkpoint_path)` 提供；`server.py:292-297` **没有** `--checkpoint`；`launch.sh:20` 原样透传 | 若用户对 `./scripts/launch.sh` 使用 `--checkpoint` 会被 server argparse 拒绝；README 未限定该参数只属于 `infer_file.py`/Python API | P2 |
| D9 | `README.md:92-93`；`README_zh.md:94-95` | Studio 安装：`pip install -r studio/requirements.txt` | 启动器的官方路径是 `scripts/bootstrap.sh`（建 venv、锁文件、Python 3.10–3.12 校验，`bootstrap.sh:8-83`）；直接 pip 安装到当前解释器**不校验版本**（只有 `check_environment.py:18-19` 会） | 两条并列的安装路径语义不同；照 README 直装可能装进 3.13/训练环境，与 Studio 的 3.10–3.12 承诺不符 | P2 |
| D10 | `README.md:129`；`README_zh.md:131` | `cd studio` 后 `python infer_file.py ...` | 启动器创建的 venv 在 `studio/.venv` 或 `~/Library/Caches/FalconStudio/venv`（`bootstrap.sh:12-18`），**不在 PATH**；README 未给 venv 的 python 路径 | 用户按字面执行 `python` 可能命中无 torch/fastapi 的系统 Python | P2 |
| D11 | 不是 README 偏差，仓库级缺陷 | — | `audio_train.py:33` 默认 `configs/NASSv2_lrs3_complex_nonmoving.yml`（`ls configs/` 只有 `Falcon.yml`）；`audio_visual_test.py:65` 默认 `Experiments/checkpoint/NASSv2_lrs3_complex_nonmoving_img89/conf.yml`（不存在） | 不带 `--conf_dir` 运行两条命令都会 `FileNotFoundError`；README 都显式传参所以未被触发，但默认值是陈旧/危险默认 | P1 |
| D12 | `studio/README.md:187` | 浏览器跟踪测试："uses a public test portrait and no camera" | `tests/serve_browser.py:10-11` `create_app(Engine())` autoload=True（`server.py:182,187`）→ `Engine.start_load`（`:126-146`）后台加载并可能 `hf_hub_download`（`falcon_runtime.py:58-78`） | 浏览器 QA 启动时会尝试下载权重（首次）；README 未提示需网络/权重缓存；另 `FALCON_QA_PORT` 默认 7866（`serve_browser.py:14`）与示例 8766 不同（示例显式覆盖，可接受） | P2 |
| D13 | `studio/README.md:182-184`；`README_zh.md` 同 | `pip install -r requirements-dev.txt` + `python -m pytest tests/` + node 循环 | `tests/test_infer_file.py:13`、`tests/test_device.py:35,44` 有 skip 条件；`test_server.py` 需 Pillow（`studio/requirements.txt:9` 有声明） | **R3 修订**：注入最小 PIL stub 后可运行（评审基线 33 passed/4 skipped，§2.4/§8.2），不是不可验证；README 仍未说明部分测试会 skip（缺权重/无 GPU）及 dev 额外依赖 | P2 |
| D14 | `studio/README.md:101` | 频谱面板显示最近 ~8 s | `spectrum-dsp.js:5-11`（16 kHz、FFT 1024、HOP 640、128 行、DB_FLOOR=-90、MAX_RECORDING_SAMPLES=4.8M=300 s）；未能定位"~8 s"常量 | 0–8 kHz / −90 dBFS 可验证；"~8 s"需前端文档（07）确认具体列数/缓冲来源 | P2（Q7） |
| D15 | 四份 README | 参数/延迟数字（0.5M/11.72M） | `docs/verification/runs/lead-model-params.log:14` `AUDIO_MODEL total=537,697 trainable=537,697`、`:18` `AUDIO+FRCNN total=11,724,385` | 与 `README.md:34,100,103` 一致（0.5M 为论文口径，实测 0.5377M；11.72M 精确匹配）；MACs 10.6G 与延迟 51.3/1361.2 ms 本轮**未尝试复算**（验证 venv 实有 `ptflops 0.7.5` 且 `torch.cuda.is_available()==True`（2×A800）；复算需固定输入与训练侧 import，超出本轮预算；R3 修正了 R1 的“无 GPU/ptflops”失实表述） | 一致/部分未验证 |
| D16 | `project_page/index.html:517` | 项目页 large 场景 Dolphin 按钮 | `large/Dolphin.wav` 为 OSS 真实 key（实测 200；小写 `large/dolphin.wav` 为 404） | **非缺陷**（2026-10-30 线上核实）；禁止规范化大小写 | 已关闭 |
| D17 | `.gitignore` | 忽略构建产物 | `audio_train.py:153-154` 创建 `Experiments/tensorboard_logs/<exp>/`；根 `.gitignore` 无 `tensorboard_logs/`（实测 NOT IGNORED） | 训练后 `git status` 出现未跟踪目录，易误提交日志 | P2 |
| D18 | `00-project-map.md:71` | `tests/ check_*.mjs/cjs（5 个）` | 实际 `studio/tests/` 6 个（4 mjs+2 cjs）+ `studio/web/*.test.cjs` 2 个（`wc -l`/`ls`） | Lead 侦察图数字错误，本域已更正 | P2 |
| D19 | 仓库/README 未写 | — | `check_environment.py:25-27` 强制要求 requirements 行含 `>=` 与 `,<`；`bootstrap.sh:24` 依赖它 | 未来若把 Studio 依赖改为 `==`（或加 hash/环境标记），bootstrap 会抛 `ValueError` 而非友好提示 | P2（Q12） |
| D20 | `README.md:107` | 权重"只包含 audio_model.* 和 video_model.*" | `falcon_runtime.py:164-169` 要求 ckpt 同时含两类前缀、`strict=True` 否则报错；`infer_file`/评测路径同理 | 代码侧约束与 README 一致；但**权重文件本身**未下载，无法验证"只包含"与体积 45 MB | 未验证（Q6） |
| D21 | `README.md:174` | 训练代码与 Studio runtime 对 `falcon.pt` 输出完全一致 | 无法在本环境验证（无权重）；两侧代码差异见 `studio/falcon_model/PROVENANCE.md` | 需 backend/model-core 复验；README 把它写成已验证事实 | 未验证（Q6） |
| D22 | `README.md:21`；`README_zh.md:23`；`studio/README.md:16`；`studio/README_zh.md:16` | 提供公网在线 demo `https://u464727-dmja-7896a455.bjb3.seetacloud.com:8443/`，并称“camera and microphone data are sent to that server” | `server.py:175-179` `same_origin` 要求 origin hostname ∈ {localhost,127.0.0.1,::1}；`server.py:196` `TrustedHostMiddleware` 仅允许 loopback/testserver。实测：`same_origin(公网 origin, 同 host)=False`；`TestClient` 注入公网 `Host` → **HTTP 400 "Invalid host header"** | 本仓库代码**不能**以公网域名原样提供该 demo（同源/主机校验会拒绝），线上必为另一份部署补丁或反向代理改写；README 未说明该前提，读者会误以为仓库代码可直接公网部署 | **P1（R3 新增；待部署方确认，Q15）** |

补充：**没有发现** README 与代码矛盾的以下项（均为一致，计入 §6 断言表）：Studio 端口/浏览器、窗口 2 s/步长 0.5 s/80 ms/0.58 s/1.42 s/5 分钟/4 帧/2.5 s、venv 位置/lock 优先/Python 3.10–3.12/FALCON_PYTHON/FALCON_VENV、训练超参（DDP/AdamW 1e-3/ReduceLROnPlateau/早停 30/梯度裁剪 5）、评测输出 CSV 路径与 ckpt 选择顺序、数据预处理脚本参数、BlackHole 行为、`.command` 文件名/可执行位、Pages 触发/权限/私有仓库逻辑。

## 5. 对外接口契约

### 5.1 脚本接口

| 接口 | 契约 | 证据 |
|---|---|---|
| `scripts/launch.sh [server args...]` | stdout：无（exec 后交棒 server）；stderr：提示；退出码：bootstrap/下载失败=1，运行中 = server 退出码；把参数透传 `server.py` | `launch.sh:2,6,9-15,20` |
| `scripts/bootstrap.sh` | **stdout 仅一行 venv python 路径**；stderr 全消息；成功 0，失败 1；副作用：可能创建 venv / pip 安装 / 联网 | `bootstrap.sh:2,25,85` |
| `scripts/doctor.sh` | stdout：平台/torch/MPS/CUDA/模型缓存状态；退出码：`small` 无法下载=1，否则 0；副作用：先跑 bootstrap（可能安装） | `doctor.sh:5-31` |
| `scripts/check_environment.py [--quiet]` | stdout：成功提示（quiet 时无）；stderr：错误列表；退出码 0/1 | `check_environment.py:13-50` |
| `scripts/install-blackhole.sh [--check|--help]` | `--check` 只读、恒 0；无参安装需 macOS，非 macOS=1；其它参数=2；会调用 `brew install --cask blackhole-2ch`、不重启 | `install-blackhole.sh:16-72` |
| `Start Falcon.command` | 双击 → launch.sh；失败时暂停等回车（130 除外）；透传参数 | `Start Falcon.command:2-9` |
| `Setup Meeting Audio.command` | 双击 → install-blackhole.sh；总是暂停等回车 | `Setup Meeting Audio.command:2-7` |
| 环境变量 | `FALCON_VENV`（环境目录）、`FALCON_PYTHON`（源解释器）、`HF_ENDPOINT`（HF 镜像）、`FALCON_DEVICE`（server 端默认设备，`server.py:183`）、`FALCON_QA_PORT`（QA 服务器端口，`tests/serve_browser.py:14`）、`FALCON_CPU_THREADS`（推理线程数，`falcon_runtime.py:175`） | 各文件行号 |

### 5.2 `server.py` CLI 契约（`launch.sh` 的透传目标）

`--host {127.0.0.1,localhost}`（默认 127.0.0.1）、`--port`（默认自动 7860–7899）、`--device {auto,cuda,mps,cpu}`（默认 auto）、`--model`（默认 `JusperLee/Falcon`，或 `JusperLee/Falcon-Big`）、`--no-browser`（`server.py:289-297`）。HTTP/WS 接口：`GET /api/status`、`POST /api/load`、`WS /ws`、`GET /`（`server.py:213-215,217-231,233,278`）。`status` 里对外暴露 `sample_rate=16000, video_fps=25, window_seconds=2, hop_seconds=.5, output_delay_ms=80`（`server.py:107-112`），这些常量与 README Studio 数字一致。

### 5.3 测试契约

- Node 测试**无 package.json/npm 依赖**，直接 `node <file>`；Node ≥ 22（本环境 v22.23.3）。
- `check_*.mjs` 用 ESM import 真实 `web/*.js`；`check_*.cjs` 用 `vm` 跑真实 worklet/app；修改这些 `web/*.js` 的导出签名会直接破坏测试（这是它们作为契约的价值）。
- pytest 文件用 `sys.path.insert(0, studio/)` 导入 `server`/`falcon_runtime`（各文件头部 `sys.path.insert(0, ...parents[1])`），因此**必须在 `studio/` 下运行**（`cd studio && python -m pytest tests/`）。
- skip 条件本身是契约：`test_infer_file.py:13`（权重未缓存→skip）、`test_device.py:35`（需无 CUDA+权重缓存）、`:44`（需无 GPU）。[代码]

### 5.4 依赖契约

- **两套解释器不通用**：根 requirements 面向训练/评测（torch>=1.13、无 fastapi/Pillow）；Studio 面向推理服务（torch≥2.14.1、fastapi/uvicorn/Pillow/huggingface_hub）。README 明确"separate environments"（`README.md:78`）。
- Python：Studio 3.10/3.11/3.12（`bootstrap.sh:9`、`check_environment.py:18-19`）；训练侧 README badge "3.10+"（`README.md:27`）但 requirements 未限制上限。
- 版本策略：Studio = 下限即"已验证最小值" + `<major` 上界 + 精确 lock；训练侧 = 不锁 + TODO（`requirements.txt:3-4`）。lock 与 requirements 8/8 自洽 [验证]。
- 已知缺口：根 requirements 缺 `huggingface_hub`（D3）；`check_environment.py` 不 import/校验 huggingface_hub（`:40-44` 清单里没有它，只在版本循环里检查）。
- **测试客户端依赖风险（R3 新增，评审 P3-1）**：`studio/requirements-dev.txt:3` 为 `httpx==0.28.1`；本轮 pytest 运行 Starlette `TestClient` 时出现告警 "Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead"（对应 `requirements-lock.txt:38` 的 `httpx2==2.15.1`）。当前 fastapi/starlette 组合下 33 passed 不受影响，但升级 starlette 后 `requirements-dev` 组合可能失效，需在锁定依赖时一并复核。

### 5.5 CI 契约

触发：`push main` + path filter（`project_page/**`、workflow 自身）+ `workflow_dispatch`；权限：`pages:write` + `id-token:write` + `contents:read`；部署前提：仓库公开（`!github.event.repository.private`）；产物：`project_page/` 目录；环境：`github-pages`。自定义域名不在 workflow 中声明（见 Q8）。

### 5.6 项目页外部契约

- 媒体：`https://oss.cslikai.cn/demos/{small,medium,large,outdoor}/{s1.mp4,base.wav,falcon.wav,swiftnet.wav,dolphin.wav,TFGrid.wav}`（large 的 `Dolphin.wav` 大写为真实 key，见 D16）。
- 字体/图标：Google Fonts + cdnjs Font Awesome（`index.html:10-13`），断网时降级。
- 图片：`logo.png` + `figures/*.pdf.png`（均在 `project_page/` 内，随 artifact 发布）。
- 同一时刻只播放一个 scene/一个方法（tab 切换 + `pauseAllSyncedMedia`，`:721-724`）；video muted，音频由 video 时间轴驱动（`:734-790`）。

## 6. 可验证断言表

| 编号 | 断言 | 证据 path:line | 验证方式 | 置信度 |
|---|---|---|---|---|
| A1 | 两个 `.command` 是可执行 bash、LF/UTF-8、文件名大小写与 README 一致 | `studio/Start Falcon.command:1-9`、`studio/Setup Meeting Audio.command:1-7`、`git ls-files -s`（100755） | `file`/`git ls-files -s`/`bash -n` [验证] | 高 |
| A2 | `Start Falcon.command` 把参数透传 launch.sh，仅失败（非 0/130）时暂停等回车 | `Start Falcon.command:3-8` | 读码 | 高 |
| A3 | `launch.sh` 用命令替换从 `bootstrap.sh` 只取 python 路径并使用 | `launch.sh:6`；`bootstrap.sh:2,25,85` | 读码 | 高 |
| A4 | 首次启动用 `weights_source("small")` 判断缓存，并显示"约 45 MB"下载提示；45 MB 与参数量推导自洽（11,724,385×4 B≈44.7 MiB） | `launch.sh:9-10`；`falcon_runtime.py:48-56`；`docs/verification/runs/lead-model-params.log:18` | 读码 + 算术推导 [推断] | 高（文件真实体积/内容未验证） |
| A5 | 权重下载失败提示 `HF_ENDPOINT` 镜像并退出 1 | `launch.sh:11-14`；`falcon_runtime.py:70-78` | 读码 | 高 |
| A6 | `launch.sh` 以 `exec python server.py "$@"` 把 CLI 参数交给 server | `launch.sh:20`；`server.py:292-297` | 读码 | 高 |
| A7 | venv 目录优先级：`FALCON_VENV` > `studio/.venv`（合法）> `~/Library/Caches/FalconStudio/venv` | `bootstrap.sh:12-18` | 读码 | 高 |
| A8 | Python 版本门限 3.10 ≤ v < 3.13（bootstrap 与 check_environment 一致） | `bootstrap.sh:8-10`；`check_environment.py:18-19` | 读码 | 高 |
| A9 | 目标环境仅在 `check_environment.py --quiet` 通过时复用 | `bootstrap.sh:24-27`；`check_environment.py:45-50` | 读码 | 高 |
| A10 | 已存在的非法环境不会被删除，报错并提示换 `FALCON_VENV` | `bootstrap.sh:29-33` | 读码 | 高 |
| A11 | `FALCON_PYTHON` 非法即报错；否则按 3.11/3.12/3.10/python3 + Finder 硬编码路径查找 | `bootstrap.sh:34-64` | 读码 | 高 |
| A12 | 安装依赖优先 `requirements-lock.txt`，其次 `requirements.txt`；有 uv 用 uv，否则 ensurepip+pip | `bootstrap.sh:70-83` | 读码 | 高 |
| A13 | bootstrap 安装后再跑一次 check_environment，失败即中止（不打印路径） | `bootstrap.sh:2,84-85` | 读码 | 高 |
| A14 | doctor = bootstrap + check_environment + 平台/torch/MPS/CUDA/模型缓存；small 不可下载时 exit 1 | `doctor.sh:5-31` | 读码 | 高 |
| A15 | check_environment 检查 Python 版本、8 个包版本边界、8 个模块 import；`--quiet` 只静默成功提示 | `check_environment.py:18-44,45-50` | 实际运行（两个解释器）[验证] | 高 |
| A16 | check_environment 要求每个依赖行严格为 `pkg>=min,<max`，否则 ValueError | `check_environment.py:21-27` | 读码 | 高 |
| A17 | BlackHole `--check`/`--help` 只读且 exit 0；无参安装非 macOS exit 1；未知参数 exit 2 | `install-blackhole.sh:16-43` | Linux 实测 `--check`/`--help` exit 0 [验证] | 高 |
| A18 | BlackHole 安装用 `brew install --cask blackhole-2ch`，带 HOMEBREW_NO_AUTO_UPDATE/ANALYTICS，不重启、不改默认设备 | `install-blackhole.sh:54-72` | 读码 | 高 |
| A19 | BlackHole 安装器不会被正常启动链调用（只有 Setup Meeting Audio.command 引用） | `install-blackhole.sh:2`；`Setup Meeting Audio.command:3`；全仓库 grep | 读码 | 高 |
| A20 | Setup Meeting Audio.command 无论结果如何都暂停等回车 | `Setup Meeting Audio.command:4-6` | 读码 | 高 |
| A21 | 端口未指定时在 7860–7899 内取第一个可 bind 的空闲端口；无空闲报错；显式端口须 1024–65535 | `server.py:304-316` | 读码 | 高 |
| A22 | 浏览器自动打开：后台轮询 `/api/status`（app=="falcon-studio"），60×0.5 s 上限；`--no-browser` 关闭 | `server.py:319-329,296` | 读码 | 高 |
| A23 | 设备 auto 顺序 CUDA→MPS→CPU；显式不可用报错；auto 下 GPU 预热失败回退 CPU | `falcon_runtime.py:81-92,183-196`；`pick_device` 由 pytest 覆盖 | 读码 + pytest `test_device` [验证] | 高 |
| A24 | 权重解析：本地文件直用；`hf://<org>/<repo>/<file>`；默认 short link；缺 hub 时给出安装/镜像提示 | `falcon_runtime.py:58-78` | pytest `test_weights` 7 passed [验证] | 高 |
| A25 | 严格加载：ckpt 必须同时含 `audio_model.*`/`video_model.*`，`strict=True`；架构名必须匹配 variant | `falcon_runtime.py:152-156,164-169` | 读码 | 高 |
| A26 | 窗口常量 16 kHz/25 fps/32000 样本/50 帧；2 s 窗、0.5 s hop、80 ms guard | `falcon_runtime.py:24-27`；`server.py:28-31`；`server.py:107-112` | 读码 | 高 |
| A27 | 8 个 Node 测试全部可运行且通过（含 2 个 web/*.test.cjs） | 见 §2.4 表；各测试文件末行 PASS 输出 | 实际逐个运行（rc=0，总 ~4.1 s）[验证] | 高 |
| A28 | **21** 个 JS 文件 `node --check` 语法通过（13 web + 4 tests mjs + 2 tests cjs + 2 web test） | `studio/web/*.js`、`tests/*.mjs|cjs`、`web/*.test.cjs` | 实际运行 21/21 [验证]（R3 修正 R1 的 19） | 高 |
| A29 | `test_weights.py` 不联网（fake hub）且 7 项通过 | `test_weights.py:17-24`；运行结果 | pytest 实测 7 passed/1.74s [验证] | 高 |
| A30 | `test_device.py` 4 passed 2 skipped（skip 条件由权重缓存/GPU 决定） | `test_device.py:35,44`；运行结果 | pytest 实测 4 passed,2 skipped/1.92s [验证] | 高 |
| A31 | `test_infer_file.py` 在无权重时整体 skip（模块级 pytestmark） | `test_infer_file.py:13`；运行结果 | pytest 实测 2 skipped/2.15s [验证] | 高 |
| A32 | `test_server.py` 用 StubEngine + TestClient，需要 Pillow；**注入仓库内最小 PIL stub 后可运行**，协议用例全部 PASSED（单跑收集 22 项），真实 JPEG/PNG 解码未验证 | `test_server.py:9-10,29-34`；`docs/verification/scripts/pil_stub/`；`docs/verification/runs/review-08-by-studio-backend.log` | pytest 实测：单跑 22 passed；四文件评审基线 33 passed/4 skipped、本轮 35 passed/2 skipped [验证] | 高 |
| A33 | 浏览器 QA 需真实 MediaPipe/Chrome；测试报告版本 1.0.1 与 vendor manifest 一致 | `tests/browser/test.js:7,24,59`；`web/vendor/mediapipe/manifest.json:3` | 读码 + 读 manifest [验证] | 中高 |
| A34 | `serve_browser.py` autoload 会在后台加载/下载模型；默认端口 7866，可用 FALCON_QA_PORT 改 | `tests/serve_browser.py:10-11,14`；`server.py:182,187`；`server.py:126-146` | 读码 | 高 |
| A35 | pages.yml 只在 main + 相关路径触发，另有 workflow_dispatch；权限/并发/私有仓库跳过/artifact=project_page | `pages.yml:5-35` | 读码 + YAML 解析 [验证] | 高 |
| A36 | 项目页相对资源存在、锚点自洽；35 条外部 URL（31 唯一）全部指向 oss.cslikai.cn 等 | `project_page/index.html`（全文）；脚本检查 | 脚本检查 [验证] | 高 |
| A37 | 项目页 stats/结果数字与 README 一致（7.7/16.4/0.5M/51.3 与两表全行） | `index.html:286-291,576-639` vs `README.md:180-206` | 人工逐项对照 | 高 |
| A38 | 四份 README 的所有相对链接均存在 | 4 个 README 全文 | 脚本检查（0 缺失）[验证] | 高 |
| A39 | 根 README 与中文版代码块（去注释/非 ASCII 后）等价；Studio 版仅示意图标注不同 | README 全文 | 脚本比对 [验证] | 高（限定范围） |
| A40 | `best_k_models.json` 5 条、key 为字面 `${ROOT}/Falcon/epoch=*.ckpt`、路径 5/5 不存在、min=epoch=714 | `best_k_models.json:1-7`；`audio_visual_test.py:83-94` | json 解析 + os.path.exists [验证] | 高 |
| A41 | `conf.yml` 与 `configs/Falcon.yml` 的差异：`audionet_name`（SpeakerContrastiveV2 vs FalconSpeakerContrastive）、data 目录占位符 `${DATA_ROOT}`、多 2 个嵌套 `causal` 键（`conf.yml:11` audio_params / `:73` video_params；`:45` 是顶层 audionet_config.causal）、多 main_args/options | 两文件全文；`Falcon.py:1314-1315` | Python yaml 展平对比 [验证]（R3 修正 causal 行号） | 高 |
| A42 | 嵌套 `causal` 差异无实际影响（`Falcon.__init__` 强制写回） | `look2hear/models/Falcon.py:1314-1315` | 读码 | 高 |
| A43 | `audio_train.py` 的 `comet_logger` 未定义（全仓库唯一出现） | `audio_train.py:170`；`grep -rn comet_logger --include=*.py` | 全仓库 grep [验证] | 高 |
| A44 | `look2hear.models` 未导出 `SpeakerContrastiveV2`，但导出 `FalconSpeakerContrastive` | `look2hear/models/__init__.py:1-2` | 注入 stub 后 import 实测 [验证] | 高 |
| A45 | 根 requirements 完全未声明 huggingface_hub；timm **有**声明（`:17`） | `requirements.txt:1-29`；`grep -ci hugging`=0 | grep + import-AST 审计 [验证] | 高 |
| A46 | Studio requirements 8 个下限与 lock 8/8 自洽；lock 由 uv 对 Python 3.11 编译 | `studio/requirements.txt:1-11`；`requirements-lock.txt:1-2` | 脚本逐条比对 [验证] | 高 |
| A47 | 沙箱可达包索引对 studio 全部精确下限版本返回 200（可满足性旁证） | `/pypi/<pkg>/<ver>/json` 11 次查询 | curl 实测 [验证] | 中（受索引/网络环境限制） |
| A48 | `.gitignore` 不覆盖 `Experiments/tensorboard_logs/`；`studio/.venv` 由 `studio/.gitignore:1` 覆盖 | 根 `.gitignore:1-35`；`studio/.gitignore:1`；`audio_train.py:153-154` | `git check-ignore` [验证] | 高 |
| A49 | `evaluated_mac_params.py` 硬编码 CUDA 与 configs 路径，`--exp_dir` 未使用 | `evaluated_mac_params.py:36,39,48,77-81,86` | 读码 | 高 |
| A50 | `check_environment.py` 的 import 清单不含 huggingface_hub（版本检查里却有） | `check_environment.py:20-44` | 读码 | 高 |
| A51 | `audio_train.py` 默认 `--conf_dir` 指向不存在的 `configs/NASSv2_lrs3_complex_nonmoving.yml`；`audio_visual_test.py` 默认 conf 也不存在 | `audio_train.py:33`；`audio_visual_test.py:65`；`ls configs`/`find Experiments` | 读码 + ls [验证] | 高 |
| A52 | `Experiments/checkpoint` 只有 2 个文件，无权重/ckpt | `find Experiments -type f` | 实测 [验证] | 高 |
| A53 | ~~项目页 large 场景 Dolphin 大小写~~ | 实测 `large/Dolphin.wav`=200、`large/dolphin.wav`=404 | curl [验证 2026-10-30] | 无（非缺陷） |
| A54 | `00-project-map.md` 的 "check_*.mjs/cjs（5 个）" 不准确（实际 6+2） | `00-project-map.md:71`；`ls studio/tests/check_* studio/web/*.test.cjs` | 实测 [验证] | 高 |
| A55 | 根 README/中文版所有可验证的 Studio 数字（端口、2s/hop、80ms、0.58/1.42s、5min、4帧、2.5s）与代码一致 | 见 §4.7 末尾"未发现矛盾"清单及 A21/A26 | 读码逐项核对 | 高 |
| A56 | README 公网在线 demo 与仓库 `same_origin`/`TrustedHost` 的 loopback 限制冲突（D22） | `README.md:21`；`README_zh.md:23`；`studio/README.md:16`；`studio/README_zh.md:16`；`server.py:175-179,196` | 实测 `same_origin(公网 origin, 同 host)=False`；`TestClient` + 公网 `Host` → HTTP 400 [验证] | 高 |
| A57 | pytest 四文件注入最小 PIL stub 后可运行（test_server 协议用例全过）：评审基线 33 passed/4 skipped | `docs/verification/scripts/pil_stub/`；`docs/verification/runs/review-08-by-studio-backend.log`；本轮复跑 35 passed/2 skipped（权重已缓存） | pytest 实测 [验证] | 高 |

（断言 **57** 条，≥30 达标；未标 [验证] 的多为纯读码事实。R3 新增 A56/A57 并更新 A4/A28/A32/A41。）

## 7. 未解疑问 / 需其它子系统确认的问题

| 编号 | 问题 | 需要谁确认 |
|---|---|---|
| Q1 | `Experiments/checkpoint/Falcon/conf.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 是笔误（应为 `FalconSpeakerContrastive`）还是来自另一个训练分支？发布版 `falcon.pt` 是否应该用这份 conf 评测？（P0 直接相关）**R2 06 域初答：不是笔误，是私有训练分支/类名时代痕迹（`PROVENANCE.md` 与 `best_k_models` 的 epoch=714 互证）；发布训练代码缺该类 ⇒ 评测不可运行，仍需作者确认** | `train-sys` / `model-core` |
| Q2 | `best_k_models.json` 的 `${ROOT}` 占位是谁/何时替换？原训练机的 `${ROOT}` 应指向什么目录（`<cwd>/Experiments/checkpoint`？）？该文件是否本就是"展示用"而非可执行元数据？ | `train-sys` |
| Q3 | `audio_train.py:170` 的 `comet_logger` 原计划是什么（CometLogger）？是否漏了 `import comet_ml` 与对应 requirement？训练入口在发布前是否真跑通过？ | `train-sys` |
| Q4 | `audio_train.py:77-84` 在模型构造**之后**写 conf.yml，而 `Separator.__init__` 会 `pop("repeats"/"shared")`（`look2hear/models/Falcon.py:991-996`）——但提交的 `conf.yml` 仍含这些键，说明它不是本脚本产出。发布的 conf.yml 与训练代码是否来自不同版本？ | `train-sys` / `model-core` |
| Q5 | `evaluated_mac_params.py:81` 用 64×64 嘴部输入、README 写"50 mouth frames"但管线是 96×96→88；论文 MACs 10.6 G 具体基于什么输入/是否含视觉模型？能否用 Lead 的 verify 脚本复算 MACs？ | `loss-metric-eval` / `model-core` |
| Q6 | 权重文件是否只含 `audio_model.*`/`video_model.*`、体积是否约 45 MB（`launch.sh:10`）？训练代码与 Studio runtime 输出"完全一致"能否用真实权重验证？**R2 06 域初答：代码侧 `falcon_runtime.py:162-169` 要求两前缀非空 + strict；体积按参数量推 44.72 MiB；用合成权重已证明 look2hear 类与 Studio CPU 前向 bit-identical，但真实 `falcon.pt` 未验证** | `studio-backend` / `model-core` |
| Q7 | `studio/README.md:101` 的 "~8 s" 频谱显示窗口来自哪个常量/列数（`spectrum-dsp.js` 只有 FFT/HOP/128 行/4.8M 上限）？"0.7–0.8 s 设计延迟"如何推导？0.12 s(Metal)/0.65 s(CPU) 本轮无法验证。 | `studio-frontend`（07） |
| Q8 | Pages 自定义域名 `https://cslikai.cn/Falcon/`：仓库里无 `CNAME`、workflow 也未设 `cname` → 是否在仓库 Settings 配置？审计结论需要"外部设置"备注。 | Lead |
| Q9 | `project_page` 的 OSS 媒体可达性与 `large/Dolphin.wav` 大小写：需线上 HEAD 检查（本环境对任意域名探测被工具策略拦截）。 | Lead / 项目页维护者 |
| Q10 | `requirements-lock.txt` 是 `uv pip compile --python-version 3.11` 生成的，而 bootstrap 允许 3.10/3.12；在 3.12 上安装 lock 是否仍自洽（平台 wheel/语法）？是否需要为每个版本生成 lock？ | Lead / 运维 |
| Q11 | `check_environment.py` 的 `>=,<` 强制格式是否要固化为契约（并在 README 说明）？否则未来 pin 成 `==` 会让 bootstrap 崩溃。 | Lead / 运维 |
| Q12 | 根 requirements 是否应补 `huggingface_hub`（支持 `hf://` 评测）与 `comet_ml`（若恢复 logger）？ | Lead / `train-sys` |
| Q13 | `README.md:129` 的 `python infer_file.py` 是否应改为显式使用 `studio/.venv/bin/python` 或 `FALCON_VENV` 路径？ | Lead/文档 |
| Q14 | 浏览器 QA（`serve_browser.py`）默认会自动加载/下载模型，是否符合设计（是否应 `autoload=False` 只测 MediaPipe）？**R2 06 域建议：应显式 `create_app(Engine(), autoload=False)`；浏览器 QA 只需 MediaPipe/追踪，不应触发首次 45 MB 权重下载** | `studio-backend` |
| Q15 | **（R3 新增，D22）** 公网在线 demo（`README.md:21` 等）如何绕过本仓库 `server.py:175-179` 的 loopback 同源限制与 `:196` TrustedHost 白名单？线上是否用另一份部署补丁/反向代理（放开 Origin/Host）？README 是否应加注“在线版由独立部署提供，仓库代码默认仅本地”？ | Lead / 部署方 / 作者 |

## 8. 本轮验证活动记录（命令、输出摘要、失败原因）

> 原则：只跑不影响仓库状态/不联网下载大文件的命令；系统 python3 = 3.8（无 torch/pytest），Node = v22.23.3；python 侧使用 Lead 提供的 `/root/.venvs/falcon/bin/python`（3.10.18，torch 2.8.0，pytest 9.1.1，fastapi，httpx，无 PIL/mediapipe/cv2）。

### 8.1 Node（全部实际执行）

```text
$ node --version
v22.23.3

$ cd studio
$ for f in tests/check_i18n.mjs tests/check_capture_dsp.cjs tests/check_mouth_tracking.mjs \
         tests/check_playback_resampler.mjs tests/check_spectrum.mjs tests/check_tracking_ui.cjs \
         web/capture-worklet.test.cjs web/video-worker.test.cjs; do node "$f"; done

结果（rc/耗时/末行输出）：
check_i18n.mjs             rc=0   85 ms   8 localization checks passed (349 dictionary entries audited).
check_capture_dsp.cjs      rc=0  515 ms   {"test":"PASS",...}
check_mouth_tracking.mjs   rc=0   63 ms   PASS: mouth geometry, stable scale/articulation, ... (长篇汇总)
check_playback_resampler.mjs rc=0 1586 ms ALL PASS: check_playback_resampler
check_spectrum.mjs         rc=0  971 ms   {"test":"PASS", ...}
check_tracking_ui.cjs      rc=0  433 ms   18 tracking UI control-flow checks passed.
web/capture-worklet.test.cjs rc=0 373 ms  PASS: 16/44.1/48 kHz clocks, ... 
web/video-worker.test.cjs  rc=0   42 ms   PASS: 25 fps sampling, ...
（合计 8/8 通过，≈3.97–4.1 s；评审与复跑一致）

$ for f in web/*.js tests/*.mjs tests/*.cjs web/*.test.cjs; do node --check "$f"; done
21/21 OK（13 个 web/*.js + 4 个 tests/*.mjs + 2 个 tests/*.cjs + 2 个 web/*.test.cjs；R3 修正 R1 的 19）
```

### 8.2 Python pytest（R3 修订：注入 PIL stub 后完整运行）

```text
# R3 路径：仓库内最小 PIL stub + 绕开 typeguard stub 插件冲突
$ PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python -m pytest studio/tests -q -p no:typeguard

评审基线（日志 review-08-by-studio-backend.log，当时无权重缓存）：
  33 passed, 4 skipped in 5.77s
  └─ test_server 协议用例全过（评审文字口径 19 项；pytest 实际收集 22 项 = 9 test 函数 + parametrize 11+4；
     33 = 22 + test_weights 7 + test_device 4）

本轮复跑（R3；HF 缓存已有 small/big 权重）：
  test_server 单跑     → 22 passed in 0.91s
  test_infer_file 单跑 → 2 passed in 68.28s（此前无权重时 2 skipped）
  四文件合计           → 35 passed, 2 skipped in 73.56s（2 skipped 均为 test_device 条件跳过）

说明与边界：
- 真实 JPEG/PNG 解码未验证（PIL stub 只覆盖尺寸/格式/convert('L') 路径；Pillow 在 studio/requirements.txt:9 声明）。
- 不加 stub 的 PYTHONPATH 下仍会因缺 PIL collection error；加 docs/verification/stubs 时必须同时
  `-p no:typeguard`——stub 的 typeguard.py 遮蔽真实 typeguard 包，导致 pytest 插件入口
  `typeguard._pytest_plugin` 加载失败（R1 现象，R3 确认绕法）。
- 未运行 browser QA（`serve_browser.py`）：无浏览器/MediaPipe，且 `create_app(Engine())` 会触发权重下载（Q14）。
```

### 8.3 脚本/配置静态与安全运行验证

```text
$ bash -n scripts/bootstrap.sh scripts/doctor.sh scripts/install-blackhole.sh scripts/launch.sh \
         "Start Falcon.command" "Setup Meeting Audio.command"
全部 OK（6/6）

$ file studio/*.command studio/scripts/*.sh
均为 "Bourne-Again shell script, UTF-8 Unicode text executable"（LF，无 CRLF/BOM）

$ git ls-files -s studio/*.command studio/scripts/*
两个 .command 与 4 个 .sh = 100755；check_environment.py = 100644（由 python 调用，不需要执行位）

$ python3 -m py_compile server.py falcon_runtime.py infer_file.py tests/*.py tests/serve_browser.py
OK（3.8 语法级编译通过；注意不能 import，缺 torch/fastapi；仅生成被 .gitignore 忽略的 __pycache__，未改源码）

$ ./scripts/install-blackhole.sh --check
尚未找到 BlackHole 2ch 驱动。 / Homebrew：未找到；可使用官方安装包。 / 官方说明：https://github.com/ExistentialAudio/BlackHole
exit 0；--help 同样 exit 0（只读模式符合代码契约）

$ python3 scripts/check_environment.py            # 系统 3.8
exit 1：请使用 Python 3.10、3.11 或 3.12；缺少 torch/PyYAML/einops/fastapi/uvicorn/huggingface_hub；
        numpy 1.24.4<2.4.6；Pillow 10.4.0<12.3.0；torch/yaml/... import 失败

$ /root/.venvs/falcon/bin/python scripts/check_environment.py   # 3.10 + torch2.8
exit 1：torch 2.8.0 需 >=2.14.1；numpy 2.2.6 需 >=2.4.6；缺 einops/Pillow/websockets；huggingface_hub 0.35.3 需 >=2.2.0
（结论：真实校验逻辑有效；沙箱环境不满足 Studio 的版本下限，因此 bootstrap 在此环境会尝试重装——未执行 bootstrap/launch，
  以免触发 venv 创建与 pip 安装。）

未执行（有意跳过，原因逐条）：
- scripts/bootstrap.sh / launch.sh / .command 的整体执行：会创建 venv 并 pip 安装/联网下载（违反本轮约束）；
  其各组成部分已由 check_environment 直接运行、其余分支静态核对。
- serve_browser.py / tests/browser：无浏览器/MediaPipe，且会下载权重。
- audio_train.py / audio_visual_test.py / evaluated_mac_params.py：**未尝试执行**（不是“没有环境”——验证 venv 实有
  pytorch-lightning 2.6.6 / ptflops 0.7.5 / torch.cuda.is_available()==True（2×A800），但没有 LRS2/LRS3 数据与 ckpt，
  且脚本会写 Experiments/；audio_train 另必然 NameError，见 §4.7 D1；MACs 复算需固定输入与训练侧全量 import，超出本轮预算）。
- 对 oss.cslikai.cn / huggingface.co / cslikai.cn 的 URL 可达性探测：本环境工具策略未放行（未验证）。

R3 补充验证（回应评审 P1-1/P1-2/P2-*）：
$ wc -l studio/.gitignore → **14**（R1 误写 18）
$ node --check（13 web/*.js + 4 tests/*.mjs + 2 tests/*.cjs + 2 web/*.test.cjs）→ **21/21 OK**（R1 误写 19）
$ pytest studio/tests -q -p no:typeguard（PYTHONPATH 注入最小 PIL stub + stubs）→ 见 §8.2（基线 33p/4s；本轮 35p/2s）
$ same_origin("https://u464727-...:8443", "u464727-...:8443") → **False**；same_origin("http://localhost:7860","localhost:7860") → True
$ TestClient(create_app(Engine(), autoload=False), base_url="http://u464727-...:8443").get("/api/status")
  → **400 "Invalid host header"**（TrustedHostMiddleware 拒绝，server.py:196）
$ importlib.metadata.version("ptflops") → **0.7.5**；torch.cuda.is_available()=**True**、device_count()=**2**（2×A800）
$ 权重体积推导：11,724,385 × 4 B = 46,897,540 B = **44.72 MiB ≈ 46.9 MB**（[推断]；与 launch.sh:10 “约 45 MB”自洽）
```

### 8.4 配置/一致性/依赖检查

```text
$ python -c "import yaml; yaml.safe_load(open('.github/workflows/pages.yml'))"
name=Project page；on=['push','workflow_dispatch']；permissions={contents:read,pages:write,id-token:write}；
if=${{ !github.event.repository.private }}；artifact path=project_page  ✔

$ python（yaml 展平）对比 conf.yml vs configs/Falcon.yml / studio/configs/FalconBig.yml
conf vs root：4 处值差异（audionet_name 与 3 个 data 目录）+ conf 多 2 个嵌套 causal + main_args/options
conf vs big ：键集合一致，但所有维度（out_chan 112/256、hid_chan 64/96、spk_embed_dim 64/128…）与 exp_name 不同
→ conf.yml 是"small 维度 + SpeakerContrastiveV2 名字"的混合体（D2/A41）

$ diff configs/Falcon.yml studio/configs/Falcon.yml
无差异（两份 Falcon-small 配置完全相同）

$ python（json）解析 Experiments/checkpoint/Falcon/best_k_models.json
5 entries；min=${ROOT}/Falcon/epoch=714.ckpt（-9.9840）；os.path.exists 5/5 False；全部 key 均以 ${ROOT} 开头

$ python 解析 4 份 README 的 Markdown 相对链接
README.md / README_zh.md / studio/README.md / studio/README_zh.md：missing = NONE

$ python 对比 README.md vs README_zh.md 代码块
去掉注释/非 ASCII 后 8/8 个代码块等价；studio 版仅 ASCII 接线示意图的标签不同（预期）

$ python 静态 import 审计（根训练/评测代码，ast 解析 100+ 文件）
第三方 import 集合：torch/torchaudio/torchvision/pytorch_lightning/numpy/scipy/pandas/yaml/soundfile/tqdm/rich/
packaging/timm/librosa/torch_complex/typeguard/espnet/torch_optimizer/cv2/matplotlib/fast_bss_eval/torch_mir_eval/
pypesq/pystoi/ptflops/**huggingface_hub** → 与 requirements.txt 对照，唯一缺失声明 = huggingface_hub（D3/A45）

$ python 注入 docs/verification/stubs 后 import look2hear.models
SpeakerContrastiveV2=False；FalconSpeakerContrastive=True（D2/A44）

$ python 比对 studio/requirements-lock.txt 与 studio/requirements.txt
8/8 个下限满足（torch 2.14.1、numpy 2.4.6、PyYAML 6.0.3、einops 0.8.2、fastapi 0.142.2、uvicorn 0.54.0、
Pillow 12.3.0、huggingface_hub 2.2.0）；lock 共 39 包（A46）

$ curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/<pkg>/<floor-version>/json
torch 2.14.1 / numpy 2.4.6 / PyYAML 6.0.3 / einops 0.8.2 / fastapi 0.142.2 / uvicorn 0.54.0 / Pillow 12.3.0 /
huggingface_hub 2.2.0 / pytest 9.1.1 / httpx2 2.15.1 / torch 2.8.0 → 全部 200（可满足性旁证；未实际安装）

$ git check-ignore -v studio/.venv/bin/python
studio/.gitignore:1:.venv/  （已覆盖）
$ git check-ignore -v Experiments/tensorboard_logs/Falcon/events.out.tfevents.1
（无输出 → NOT IGNORED，D17）
$ git check-ignore -v Experiments/checkpoint/Falcon/last.ckpt
.gitignore:24:*.ckpt（已覆盖）
```

### 8.5 未验证清单

- 权重下载/加载/推理一致性（R1 未下载；**R3 复核时发现本环境 HF 缓存已有 small/big 权重**，但未据此重做端到端一致性验证）。
- Metal/CUDA 真实行为、M5 Pro 0.12 s / CPU 0.65 s、10.6G MACs、51.3/1361.2 ms（**未尝试复算**：venv 有 ptflops 0.7.5 且 CUDA 可见 2×A800，但需固定输入/训练侧数据与 ckpt；R3 修正“无 GPU/ptflops”失实表述）。
- macOS 上 Finder 双击 `.command`、brew 安装 BlackHole（Linux 沙箱，无 brew）。
- 浏览器 QA（MediaPipe 真实检测）、Pages 实际部署与自定义域名、OSS 媒体可达性/大小写结果。
- startup 链的端到端执行（避免 pip 安装）。

## 9. 变更日志（Changelog）

| 日期/轮次 | 改了什么 | 依据 |
|---|---|---|
| 2026-10-21 / R1 | 初稿：新增本文件。完成启动链逐段解析（launch/bootstrap/doctor/check_environment/.command/BlackHole）、测试矩阵与 8 个 Node 测试 + pytest 实测结果、CI/项目页/依赖/实验产物分析、55 条断言表、21 条 README 偏差清单（2×P0）、14 条未解疑问、完整验证记录 | 本域 task-8；`docs/understanding/00-CONVENTIONS.md` 8 节模板；Lead 的 `VERIFY-ENV.md` 与消息提示 |
| 2026-10-21 / R3 | 依据 R2 评审（studio-backend）修订：新增 **D22**（公网在线 demo vs loopback 同源/TrustedHost，P1）与 §9.1 逐条响应；测试矩阵改为“**注入 PIL stub 可运行**”（基线 33 passed/4 skipped、本轮 35 passed/2 skipped；test_server 协议用例全过，单跑 22 passed）；更正 `studio/.gitignore` 14 行、`node --check` **21/21**、conf.yml causal 行号 **:11/:73**、server.py 行号偏移（175-179/235-237/105-116）；D2 补“首次失败点”（FileNotFoundError→AttributeError 两阶段）、A4 补 45 MB 参数量推导、D15 改“未尝试复算”（ptflops 0.7.5/CUDA×2 可见）；新增 Q15、A56/A57；§5.4 补 httpx/httpx2（P3-1）风险。断言 55→57（含 2 P0 保留） | `docs/reviews/round1/08-ops-tests-docs-review-by-studio-backend.md` + Lead R3 必改 1–4 |

### 9.1 R2 评审意见逐条响应（reviewer: studio-backend）

| 评审项 | 等级 | 回应 | 落点 |
|---|---|---|---|
| P1-1 在线 demo vs 本地同源限制 | P1 | **采纳**：新增 D22（§4.7）+ §1.2 第 11 条 + A56 + Q15；实测 `same_origin(公网)=False`、公网 `Host` → HTTP 400；README 未说明“线上为独立部署补丁” | §1.2、§4.7、§6、§7 |
| P1-2 test_server 可运行（PIL stub） | P1 | **采纳**：§2.4 测试矩阵改“✅ 可运行（注入 `docs/verification/scripts/pil_stub/` + `-p no:typeguard`）”，§8.2 重写；保留“真实 JPEG/PNG 解码未验证”边界；引用评审脚本/日志 | §2.4、§6 A32、§8.2 |
| P2-1 `studio/.gitignore` 行数 | P2 | 18→**14** | §2.9 |
| P2-2 `node --check` 文件数 | P2 | 19→**21**（13+4+2+2，实测 21/21） | §2.4、A28、§8.1 |
| P2-3 `conf.yml` causal 行号 | P2 | **:45→:11**（audio_params）并补 `:73`（video_params）；注明 `:45` 是顶层 `audionet_config.causal` | §2.8、A41 |
| P2-4 D2 首次失败点 | P2 | 补：未配置数据时先 `datamodule.setup()` → `FileNotFoundError`（mix.json）；数据就绪后才 `_load_models_from_ckpt` → AttributeError | §4.7 D2 |
| P2-5 A4 “45 MB 无法验证” | P2 | 改为“与参数量推导自洽（11,724,385×4 B≈44.7 MiB≈46.9 MB）[推断]；文件体积/内容未验证” | §2.1.1、A4、§8.3 |
| P2-6 D15 环境陈述失实 | P2 | 改为“**未尝试复算**”；注明 venv 实有 ptflops 0.7.5、`torch.cuda.is_available()==True`（2×A800） | §4.7 D15、§8.5 |
| P2-7 行号偏移 | P2 | 已改：`same_origin` 175-179、WS 同源 235-237、`status()` 105-116 | §4.4、§4.5 |
| P3-1 httpx/httpx2 | P3 | §5.4 增风险注记（`requirements-dev.txt:3` httpx==0.28.1 vs lock 的 httpx2==2.15.1；Starlette TestClient 告警；升级 starlette 需复核） | §5.4 |
| §4 遗漏 1–5 | — | 1/2/3/5 已入正文/矩阵；4（`serve_browser.py` autoload）已在 Q14 补 06 域建议；2 随 P1-2 修复 | 全文 |
| X1（Q1 答案） | — | 已把 06 域初答写入 Q1（私有分支痕迹、非笔误、评测不可运行待作者确认） | §7 Q1 |
| X2（Q6 答案） | — | 已写入 Q6（两前缀+strict 代码证据、体积 44.72 MiB、合成权重 bit-identical、真实权重未验证） | §7 Q6 |
| X3（Q14 答案） | — | 已写入 Q14（建议 `create_app(Engine(), autoload=False)`） | §7 Q14 |
| 保留未决 | — | Q10/Q11/Q12/Q13 按评审建议保留，待 Lead 裁决（不擅自在 R3 下结论） | §7 |

> **R3 交付 sha256**（`sha256sum docs/understanding/08-ops-tests-docs.md`，在写入本行之前计算；本行写入后文件哈希会再次变化）：`88cd19789af19cdcc89819ab288d82af740b0934c859621df4308e4c31507922`。
> 说明：写入该哈希行后文件哈希会再次变化；R1 评审基线为 `75692199722df17160157529bb95073e47787e471369f352afe0844869084b98`（评审方记录）。
