# 评审：`08-ops-tests-docs.md` —— 评审人 `studio-backend`

> 被评审文档：`docs/understanding/08-ops-tests-docs.md`（718 行，R1 初版，92 条断言）
> 评审时 sha256（开始/结束）：`75692199722df17160157529bb95073e47787e471369f352afe0844869084b98`（**评审期间未变化**）
> 证据基线 commit `7576854`；评审侧证据日志 `docs/verification/runs/review-08-by-studio-backend.log`；
> 复用脚本 `docs/verification/scripts/verify_studio_backend.py` + `docs/verification/scripts/pil_stub/`（本域 R1 产出）。
> 环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128 + fastapi/uvicorn/pytest + `ptflops 0.7.5`，
> CUDA 可见 2×A800；无 Pillow/mediapipe/cv2），Node v22.23.3。
> 评审侧独立复核方式：读源码 + 实际执行（不依赖被评审文档的脚本），所有命令见证据日志。

## 1. 结论

**需修订（Minor，整体质量很高）**。

被评审文档把“运维/测试/CI/四份 README 一致性”做成了可操作的偏差清单（D1–D21），证据密度高、诚实标注验证边界，
其中**两个 P0 经我独立复核全部成立**：
- P0-1 `audio_train.py:170` 的 `comet_logger` 未定义（全仓库唯一出现，`grep -rn comet_logger --include=*.py` = 1 行）
  → `pl.Trainer(...)` 构造即 `NameError`；
- P0-2 `Experiments/checkpoint/Falcon/conf.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 在 `look2hear.models`
  中不存在（`hasattr` 实测 False），`audio_visual_test.py:129` 的 `getattr` 会失败（与我的 06 文档 Q2/Q3 互证）。

但作为“偏差清单 + 测试矩阵”交付物，存在 **2 个 P1（重要遗漏/可补强）与 7 个 P2（行号/算术/环境陈述）**，
其中 P1-1 是 README-vs-代码审计的实质遗漏（在线 demo），P1-2 关系到本域最重要的契约（HTTP/WS 协议）能否被测试矩阵覆盖。

**问题计数：P0 = 0（新增；作者原有 2 个 P0 复核为真）；P1 = 2；P2 = 7；P3 = 1。**

## 2. 断言逐条核对表

> ✅=正确；❌=错误/需修订；❓=无法确认（含我未独立复核）。编号沿用被评审文档 A1–A55。
> “我的复核”列全部来自我自己的命令/读码，不引用 08 文档的脚本。

### 2.1 脚本 / 安装 / 启动链（A1、A3–A14、A17–A23、A34）

| 编号 | 结论 | 我的复核（独立） | 备注 |
|---|---|---|---|
| A1 | ✅ | `git ls-files -s`：两个 `.command`/四个 `.sh` 均 100755，`check_environment.py` 100644；文件名与 README 一致 | — |
| A2 | ✅ | `Start Falcon.command:3-8` 读码确认（透传 + 非 0/130 暂停） | 未动态执行（会启动服务） |
| A3 | ✅ | `launch.sh:6` 命令替换 + `bootstrap.sh:2,25,85` stderr 约定 | — |
| A4 | ✅（但"45 MB 无法验证"可改进，见 P2-5） | `launch.sh:9-10` 读码；体积可从参数量推导 ≈44.72 MiB（见 P2-5） | 建议改口径 |
| A5 | ✅ | `launch.sh:11-14` → `resolve_checkpoint` 的 `HF_ENDPOINT` 提示在 `falcon_runtime.py:76-78` | — |
| A6 | ✅ | `launch.sh:20` `exec ... server.py "$@"` | — |
| A7 | ✅ | `bootstrap.sh:12-18` 三级优先级（FALCON_VENV → studio/.venv → Library/Caches） | — |
| A8 | ✅ | `bootstrap.sh:9` 与 `check_environment.py:18-19` 均为 `(3,10) <= v < (3,13)` | — |
| A9 | ✅ | `bootstrap.sh:24-27` 依赖 `check_environment.py --quiet` 返回码；我实跑该脚本 exit=1（本验证环境 torch 2.8/numpy 2.2 低于 Studio 下限）——行为与文档一致 | — |
| A10 | ✅ | `bootstrap.sh:29-33` 不删除已有目录 | — |
| A11 | ✅ | `bootstrap.sh:34-64`、硬编码候选路径读码 | — |
| A12 | ✅ | `bootstrap.sh:70-83` lock 优先、uv/ensurepip+pip 双路径 | — |
| A13 | ✅ | `bootstrap.sh:84-85` | — |
| A14 | ✅ | `doctor.sh:5-31`（含 small 不可下载 `SystemExit(1)`，`:29-30`） | — |
| A17 | ✅ | 实跑 `./scripts/install-blackhole.sh --check` → 中文只读输出、exit 0 | Linux 下也 exit 0，与文档一致 |
| A18/A19/A20 | ✅ | `install-blackhole.sh:54-72`、`.command:2-7` 读码；全仓库仅 `Setup Meeting Audio.command` 引用安装脚本 | — |
| A21 | ✅ | `server.py:304-316` 端口扫描读码（与 06 文档 §2.1.7 一致） | — |
| A22 | ✅ | `server.py:319-329`/`:296` 读码；浏览器在模型就绪前即打开（status ready=false 也返回） | — |
| A23 | ✅ | `falcon_runtime.py:81-92,183-196`；我的 `test_device` 实跑 4 passed/2 skipped；`pick_device` 在 A800 上 auto→cuda | — |
| A34 | ✅ | `tests/serve_browser.py:10 `create_app(Engine())`（autoload 默认 True，`server.py:182,187`）；默认端口 7866（`:14`） | 我给出 Q14 的建议答案（§6-Q14） |

### 2.2 测试矩阵（A27–A33）——**本域最关心的部分**

| 编号 | 结论 | 我的复核（独立） | 备注 |
|---|---|---|---|
| A27 | ✅ | 我逐个实跑 8 个 Node 测试：**8/8 rc=0**，总耗时 ≈3.97 s（89/484/65/1505/951/442/384/44 ms） | 与文档“~4.1 s”一致 |
| A28 | ❌（数量错，见 P2-2） | 实跑 `node --check`：**21/21 OK**（13 `web/*.js` + 4 `tests/*.mjs` + 2 `tests/*.cjs` + 2 `web/*.test.cjs`） | 文档写 19 |
| A29 | ✅ | 我跑全部 4 个 pytest 文件：`33 passed, 4 skipped`；其中 `test_weights` 7 passed 与文档一致 | 见 A32 |
| A30 | ✅ | `test_device` 4 passed/2 skipped（我的环境同样因有 CUDA 跳过 2 项） | — |
| A31 | ✅ | `test_infer_file` 2 skipped（模块级 `pytestmark`，无权重） | — |
| A32 | ❌（结论过强，见 P1-2） | 用仓库内最小 Pillow 替身（`docs/verification/scripts/pil_stub`，仅实现 `Image.new/open/save/convert`）注入 PYTHONPATH 后，`test_server.py` **全部 19 个用例通过**；4 文件合计 `33 passed, 4 skipped in 5.77s`。因此“本沙箱只能 collection error”只是缺 1 个可替身的第三方包，不是不可验证 | 建议测试矩阵改为“可运行（注入 PIL stub）”，并标注真实图片解码仍属 Pillow 职责 |
| A33 | ❓ | 无浏览器/MediaPipe/权重，未跑；读码与 manifest 版本一致这点成立 | 维持作者 [未验证] |

补充：文档 §8.2 说“按 Lead 提示加 stub PYTHONPATH 会因 typeguard 插件加载失败”——我实测同样现象，但加
`-p no:typeguard` 即可绕开（且 PIL stub 目录与 `docs/verification/stubs` 可同时使用）。这是 R3 可补的操作细节（P1-2）。

### 2.3 CI / 页面 / 依赖 / 元数据 / README（A24–A26、A35–A55）

| 编号 | 结论 | 我的复核（独立） | 备注 |
|---|---|---|---|
| A24 | ✅ | `falcon_runtime.py:58-78` 读码 + `test_weights` 7 passed（我复跑） | — |
| A25 | ✅ | `falcon_runtime.py:152-156,164-169`；我另用自造 ckpt 验证 strict 四类失败（06 文档 §8-A7） | — |
| A26 | ✅ | `server.py:28-31,107-112`；`falcon_runtime.py:24-27` 与我 06 文档常量表逐项一致 | — |
| A35 | ✅ | `pages.yml:5-35` 读码（main+path filter、workflow_dispatch、私有仓库 if、artifact=project_page） | — |
| A36 | ✅ | 我实测 `project_page/index.html` 外部 URL：**35 条 / 31 唯一**，与文档一致 | — |
| A37 | ✅（抽查） | 抽查 7.7/8.2、16.4/16.7、0.5M、51.3 与 README 表一致（其余未逐项） | — |
| A38/A39 | ❓ | 未独立跑（作者脚本已覆盖，低风险） | — |
| A40 | ✅ | `best_k_models.json` 5 条、key 为字面 `${ROOT}/Falcon/epoch=*.ckpt`、`min` 项 = epoch 714（−9.984）；`os.path.exists` 不展开 → 5/5 False | 我另核：`Experiments/` 仅 2 文件、无 ckpt |
| A41 | ⚠️ 基本正确但行号错（P2-3） | 展开对比成立：`audionet_name` 不同、`${DATA_ROOT}` 占位、多 `main_args/options`；**多出的 2 个嵌套 `causal` 是 `audio_params.causal`（`:11`）与 `video_params.causal`（`:73`），不是 `:45`**（`:45` 是顶层 `audionet_config.causal`） | 见 P2-3 |
| A42 | ✅ | `look2hear/models/Falcon.py:1314-1315`（我的 studio 副本对应 `:1323-1324`）强制写回 causal | — |
| A43 | ✅ | `grep -rn comet_logger --include=*.py .` → 仅 `audio_train.py:170`；根 requirements 无 comet_ml | — |
| A44 | ✅ | `hasattr(look2hear.models,'SpeakerContrastiveV2')==False`（我 R1 已实测，PYTHONPATH stubs） | — |
| A45 | ✅ | `grep -ci hugging requirements.txt` = 0；`timm` 在 `:17` | — |
| A46 | ✅ | lock 39 个包（`grep -cE '^[a-z0-9_-]+=='` = 39）；8 个下限逐条落在 `[min,<max)` | — |
| A47 | ❓ | 依赖外部包索引可用性，我未复核（环境网络策略不一）；作为“旁证”标注中可接受 | — |
| A48 | ✅ | 实跑 `git check-ignore -v`：`studio/.venv/bin/python` → `studio/.gitignore:1`；`Experiments/checkpoint/Falcon/last.ckpt` → `.gitignore:24`；`Experiments/tensorboard_logs/x` **NOT ignored** | — |
| A49 | ✅ | `evaluated_mac_params.py:36,39,48,77-81,86` 读码（`.cuda()` 多处；`--exp_dir` 未使用） | — |
| A50 | ✅ | `check_environment.py:20-44` import 清单无 huggingface_hub | — |
| A51 | ✅ | `audio_train.py:33` 默认 `configs/NASSv2_lrs3_complex_nonmoving.yml`；`audio_visual_test.py:65` 默认不存在路径 | — |
| A52 | ✅ | `find Experiments -type f` = 2 个文件 | — |
| A53 | ✅ | `grep -n "dolphin\|Dolphin" project_page/index.html`：`:517 large/Dolphin.wav` vs 其余小写 | — |
| A54 | ✅ | `00-project-map.md:71` 写“check_*.mjs/cjs（5 个）”；实际 tests 6 + web 2 | — |
| A55 | ✅（抽查） | 与我的 06 文档 §1.3/§5.9 常量表一致；未发现矛盾 | — |

**未独立复核**：A2、A15（作者已实跑、我仅验证 exit 码语义）、A16、A18–A20 的动态部分、A33、A37 全表、A38、A39、A47、A55 全表 —— 均无理由怀疑，建议作者在 R3 把关键 [验证] 输出贴入日志即可。

## 3. 错误清单

### P0（事实错误 / 会误导复现）

无新增。作者的两个 P0 复核成立；仅 D2 的“首次失败点”需精确化（见 P2-4，不影响 P0 定级）。

### P1（重要遗漏 / 误导）

- **P1-1 偏差清单缺“在线 demo vs 本地同源限制”这一条真实 README 偏差。**
  README 的 Live demo 指向公网地址：`README.md:21`、`README_zh.md:23`、
  `studio/README.md:16`（“It runs on a GPU server, so camera and microphone data are sent to that server”）、
  `studio/README_zh.md:16`；而本仓库 `server.py` 的同源/主机限制是：
  `same_origin` 要求 origin hostname ∈ {localhost,127.0.0.1,::1}（`server.py:175-179`），
  `TrustedHostMiddleware` 白名单只有 loopback/testserver（`server.py:196`）。
  我的实测（证据日志）：`same_origin("https://u464727-...:8443", 同host)` = **False**；
  `TestClient` 注入 `Host: u464727-...:8443` → **400**。
  ⇒ 公网 demo 不能由本文件 `server.py` 原样提供服务，必然是另一份部署补丁（放开 Origin/Host 或反向代理改写）。
  这是“README 承诺 vs 仓库代码”的实质性偏差，应进 §4.7 偏差清单（建议编号 D22，严重度 P1/待部署方确认），
  并在 §7 增补问题项。该项同时与我的 06 文档 §4.6/Q4、A54 互证。
- **P1-2 测试矩阵低估 `test_server.py`：它在本验证环境可跑，且是本域（Studio 协议）最重要的可执行契约。**
  文档 §1.2-7 / §2.4 / A32 / D13 均称“本沙箱缺 Pillow → collection error，属环境问题”。
  这在“不注入替身”的前提下成立，但：`test_server.py` 只在 `decode_request` 里用 Pillow 解码 96×96 图，
  测试图片是合成单色图；用**最小 PIL 替身**即可完整跑协议逻辑。我实测：
  ```
  PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python -m pytest studio/tests/test_server.py \
      studio/tests/test_weights.py studio/tests/test_device.py studio/tests/test_infer_file.py -q -p no:typeguard
  → 33 passed, 4 skipped in 5.77s   （test_server 19 项全部 PASSED）
  ```
  （`-p no:typeguard` 用于绕开文档 §8.2 提到的 typeguard 插件入口问题；真实图片解码仍未验证，这一点应保留。）
  影响：偏差清单/测试矩阵是 08 域的核心交付，而“HTTP/WS 协议是否可验证”直接决定 06/07 域契约的可信度；
  建议 R3 把测试矩阵中 `test_server` 从“❌ 不可运行”改为“✅ 可运行（注入 PIL stub + `-p no:typeguard`；
  真实 JPEG/PNG 解码未验证）”，并引用我的脚本/日志作为可复现证据。
  *此项不是作者造假（作者如实写了失败原因），而是遗漏了低成本可运行路径，且因此未对协议层做任何动态验证。*

### P2（细节 / 行号 / 算术 / 环境陈述）

- **P2-1 `studio/.gitignore` 行数错误**：文档 §2.9 写“根 35 行 + `studio/.gitignore` 18 行”，
  实测 `wc -l studio/.gitignore` = **14**（根 `35` 正确）。
- **P2-2 `node --check` 文件数错误**：A28 与 §8.1 写“19/19 OK（13 + 4 + 2 + 2）”，
  但 13+4+2+2 = **21**；实测 `13 个 web/*.js + 4 个 tests/*.mjs + 2 个 tests/*.cjs + 2 个 web/*.test.cjs = 21/21 OK`。
  （§2.4 单独写的“8 个 Node 测试”本身正确。）
- **P2-3 `conf.yml` 行号错**：§2.8 表“`:45` → `audio_params.causal: true`”。
  实测该文件：`audio_params:` 起于 `:9`，嵌套 `causal: true` 在 **`:11`**；`:45` 是**顶层 `audionet_config.causal`**；
  `video_params.causal` 在 `:73`。文档“conf.yml 比 `configs/Falcon.yml` 多 2 个嵌套 causal 键”的**结论正确**，
  但证据行号应改为 `:11`（audio_params）与 `:73`（video_params）。
- **P2-4 D2/§1.2-2 的“首次失败点”不精确**：`audio_visual_test.py` 的 `process_worker` 先
  `datamodule = AVSpeechDataModuleWithNear(**data_cfg); datamodule.setup()`（`:185-190`），
  `setup()` 内构造 `AVSpeechEvalDataset(json_dir=test_dir)`（`avspeech_dataset_near.py:277-296`），
  该数据集会 `open(<test_dir>/mix.json)`（`avspeech_dataset.py:145-150`）。
  而 conf.yml 的三个目录是字面量 `${DATA_ROOT}/...`（`conf.yml:99-102`）。
  ⇒ **在未按 README 要求先改数据路径时，第一异常是 FileNotFoundError（数据 JSON），不是 AttributeError**；
  AttributeError 是在“数据路径已改好/数据存在”后 `_load_models_from_ckpt`（`:201-202`）才出现。
  README `:174` 确实要求用户先改数据路径，所以两种情形的结论（命令不可用）都不变，但文档应说明“首次失败点取决于是否已配置数据”。
- **P2-5 A4“45 MB 无法验证”可改为可佐证**：README/`launch.sh:10` 的“约 45 MB”与参数量自洽——
  `11,724,385 × 4 B = 46,897,540 B = 44.72 MiB ≈ 46.9 MB`（fp32 state_dict，不含 optimizer）。
  [推断] 建议改为“与参数量推导一致（44.7 MiB），文件真实体积/内容未验证”。
- **P2-6 D15 的环境陈述失实**：“MACs 10.6G 与延迟…本轮无法复算（无 GPU/ptflops 环境）”——
  验证 venv 实际装有 `ptflops 0.7.5`，且 `torch 2.8.0+cu128` 在此机器 `torch.cuda.is_available() == True`（2×A800）。
  正确表述应为“未尝试复算（脚本依赖训练侧全量 import / 需固定输入与后端）”或“时间预算外”，
  而不是“没有环境”；否则低估了可验证范围（这与 A47 的“包索引 200”同属环境能力陈述，应统一口径）。
- **P2-7 少量行号偏移（低价值但建议统一）**：§4.5 `server.py:175-180`（`same_origin` 实为 `175-179`）、
  `:233-236`（WS 同源检查实为 `235-237`）；§4.4 引用 `server.py:107-118`（`status()` 实为 `105-116`）。
- **P3-1（未来风险，非当前错误）**：`studio/requirements-dev.txt:3` `httpx==0.28.1`，而当前 fastapi/starlette
  TestClient 已提示 “Using httpx with starlette.testclient is deprecated; install httpx2 instead”（我跑 pytest 时的告警）。
  这不影响当前测试（33 passed），但升级 starlette 后可能破坏 `requirements-dev` 组合；建议在 §5.4 依赖契约
  加一句“test 客户端依赖 httpx 旧路径”的风险注记。

## 4. 遗漏的重要行为（代码里有、文档没覆盖）

1. **在线 demo 的部署补丁问题**（P1-1）：README 承诺外网 demo 可用，但本仓库 `same_origin`/`TrustedHost`
   只允许 loopback；文档在 §2.10 只描述 README 有“在线 demo 警告”，未把它当作“代码-文档偏差”审计。
2. **`test_server` 的可替代验证路径**（P1-2）：测试矩阵没有说明“缺 Pillow 时可用最小替身验证协议逻辑”，
   导致 08 域对 06/07 最核心的 WS/HTTP 契约没有任何动态验证记录。
3. **`FalconRuntime` 权重体积与参数量自洽**（P2-5）：A4 把 45 MB 归为“无法验证”，其实可解析佐证；
   同类的“权重只含 `audio_model.*`/`video_model.*`”（D20）也可由 `falcon_runtime.py:162-169` 的
   “两个前缀必须非空 + strict”给出更强的**代码侧**证据（作者已引用，可再明确结论）。
4. **`serve_browser.py` 的 `autoload` 语义**（作者 Q14 已提出）：我确认 `create_app(Engine())` 默认
   `autoload=True`，浏览器 QA 只需要 MediaPipe/追踪，不需要模型 → 建议 QA 服务器显式 `autoload=False`
   （或 `create_app(Engine(), autoload=False)`），避免首次 QA 触发 45 MB 权重下载。此为对作者 Q14 的回答。
5. **作者 Q1/Q6 可从本域得到部分答案**（详见 §5），建议 R3 在 §7 直接引用 06 文档结论，避免 Q 悬空。

## 5. 跨模块一致性问题

| # | 与 08 文档相关内容 | 我的 06 域结论/证据 | 建议 |
|---|---|---|---|
| X1 | D2（P0）：conf.yml 的 `SpeakerContrastiveV2` 在 `look2hear.models` 缺失；Q1 问“笔误还是另一训练分支” | **不是笔误**：`PROVENANCE.md:6-11` 同样把 `SISNRRepel.py`/`Speaker_Contrastive_v2.py` 归为 Big 架构源码，而这两文件只在 `studio/falcon_model/` 存在（`look2hear/` 无源，git 仅 2 commit）；`Experiments/checkpoint/Falcon/best_k_models.json` 的 best 恰为 `epoch=714`，与 `PROVENANCE.md:36-39` 的“falcon.pt 由 epoch 714 导出”一致 ⇒ conf.yml 来自“私有训练分支/另一个类名时代”的痕迹，而非笔误 | R3 把 Q1 回答为“来源分支痕迹 + 发布训练代码缺类 ⇒ 评测不可运行”，并标注需作者确认 |
| X2 | Q6：权重只含两前缀？体积 45 MB？训练代码 vs Studio 输出一致？ | 代码侧：`falcon_runtime.py:162-169` 要求两前缀非空 + strict；体积按参数量推 44.72 MiB；一致性：我 R1 用**同一合成权重**证明
`look2hear` 类与 Studio 运行时 CPU 前向 **bit-identical（max|Δ|=0）**、state_dict key/形状全等（audio 525/video 137），但**真实 falcon.pt 未验证**（无权重、禁下载） | R3 引用 06 文档 §4.8/§4.9/§8-A6，并保留“真实权重未验证” |
| X3 | Q14：`serve_browser.py` autoload 是否合理？ | 确认 `create_app(Engine())` 默认 autoload=True（`server.py:182,187`）且会触发 `FalconRuntime.load` → 可能下载权重（`falcon_runtime.py:58-78`）；浏览器 QA 不需要模型 | 建议改 `autoload=False`，R3 写入答案 |
| X4 | D8：`--checkpoint` 只属于 `infer_file.py`/Python API，`server.py` 无此参数 | 我 06 文档 §4.4/§5.2 同结论；服务端只接受 variant，自定义权重仅 Python API/CLI | 一致，无需改 |
| X5 | A25/A26：窗口/时长/采样常量 | 与 06 文档 §1.3 常量表逐项一致（16k/25/32000/8000/1280/80 ms；输出 8000=0.5 s） | 一致 |
| X6 | D14（~8 s 频谱）/D12（QA 端口） | 属 07 域；我的 06 文档只涉及采集端 16 kHz/25 fps 契约 | 交给 07/R4 |
| X7 | D7（MACs 脚本 64×64 mouth） | 06 域：运行时 mouth 是 96→88；MACs 脚本与推理输入不一致但只影响视觉分支输入形状（audio model 的 `pretrained_vout_chan=512` 固定），数值影响有限；具体 MACs 复算归 04/model-core | 与 04 域 Q5 对齐，勿在 08 域下结论 |

## 6. 给作者的问题清单（R3 需回应）

1. **P1-1**：是否同意把“在线 demo 公网 URL vs `same_origin`/`TrustedHost` loopback 限制”加入 §4.7 偏差清单
   （建议 D22）？若你认为“线上是另一份部署、不算 README 偏差”，请在文档中显式说明判定口径并给出反证。
2. **P1-2**：是否采纳“test_server 可运行（可注入 PIL stub）”的结论，并引用
   `docs/verification/scripts/verify_studio_backend.py` / `docs/verification/runs/review-08-by-studio-backend.log`
   更新测试矩阵与 A32/D13？（真实 JPEG/PNG 解码未验证的边界应保留。）
3. **P2-3**：能否修正 `conf.yml` 的 causal 行号（`audio_params.causal: :11`、`video_params.causal: :73`）？
4. **P2-4**：D2 是否补充“未配置数据路径时首个异常是 FileNotFoundError（mix.json），配置后才是 AttributeError”？
5. **P2-5/P2-6**：A4 的 45 MB 是否改为“与参数量推导一致（44.7 MiB）[推断]”？
   D15 的“无 GPU/ptflops 环境”是否改为“未尝试复算”并注明 venv 实际具备 ptflops/CUDA？
6. **P2-1/P2-2**：`studio/.gitignore` 14 行、`node --check` 21 个文件，请修正。
7. **Q1/Q6/Q14 的回答**：是否采纳 §5 的 X1/X2/X3 结论并在 §7 标注“已由 06 域回答”？
8. **Q10/Q11/Q12/Q13**（lock/Python 版本、check_environment 格式契约、根 requirements 补 huggingface_hub、
   README `python infer_file.py` 路径）属 ops/Lead 口径，请保留待 Lead 裁决，不要在 R3 擅自下结论。

---

## 附：评审侧复核活动记录（摘要；完整命令见 `docs/verification/runs/review-08-by-studio-backend.log`）

| 活动 | 结果 |
|---|---|
| `wc -l` 文档/脚本 | 08 文档 718 行；launch 20 / bootstrap 85 / doctor 32 / install-blackhole 72 / check_environment 54；`studio/.gitignore` **14**（文档写 18 ❌） |
| Node 8 个测试逐个执行 | **8/8 rc=0**，≈3.97 s（与文档 ~4.1 s 一致） |
| `node --check` 全量 | **21/21 OK**（文档写 19 ❌，算术错误） |
| 4 个 pytest 文件（PIL stub + `-p no:typeguard`） | **33 passed, 4 skipped in 5.77 s**；`test_server` 19 项全过（文档称不可收集 ❌/可补强） |
| `git ls-files -s` / `git check-ignore -v` | `.command`/`.sh` 100755，`check_environment.py` 100644；`studio/.venv`、`*.ckpt` 已被忽略；`Experiments/tensorboard_logs/` NOT ignored（A48 ✅） |
| `same_origin` / `TrustedHost` 实测 | 公网 demo origin → `same_origin=False`；`Host=<public>` → HTTP 400（P1-1 证据） |
| `check_environment.py --quiet`（验证 venv） | exit=1（版本下限不满足；与 A9/A15 行为一致） |
| `ptflops` / CUDA | `ptflops 0.7.5`；`torch.cuda.is_available()==True`（P2-6 证据） |
| `best_k_models.json` / conf.yml / project_page URL | 5 条 epoch 记录、min=714；conf.yml 168 行、causal 行号 11/34/45/49/73/79；外部 URL 35/31 唯一（A36/A40 ✅，A41 行号 ❌） |
| 权重体积推导 | 11,724,385×4 B = 44.72 MiB ≈ 46.9 MB（P2-5） |
| 文档 sha256 | 评审前后均为 `75692199…84b98`（未变化，本评审基于该版本） |
