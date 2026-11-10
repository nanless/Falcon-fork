# 09 Space / 项目页 / CI

> **历史记录更正（2026-11-10）**：此文保存 11 月 8 日的代码和测试快照；现状以[当前状态](../../STATUS.md)和[评测与应用专题](../20261110-evaluation-and-apps.md)为准。§3.5 将发布权重训练来源称作 LRS3 互证，当前身份仍待作者或原始训练记录确认；该文 164/166 与 168/166 测试计数、54 与原始输出 57 文件数分别对应不同时间和计数口径。§2.2 提到 `/api/status`，因此“只有一个动态端点”只能指分离调用，不包括状态接口。

> 子代理 W1-09 · 2026-11-08 · 目标仓库 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork` · HEAD `6c1d5b7`（2026-10-31 06:57 +0000）
> 范围：`space/**`、`project_page/**`、`.github/**`、`assets/`、根 `README.md` / `README_zh.md`
> 仓库只读；唯一写入 = `/tmp/falcon-space-w109`（构建产物）+ 本报告。

## 0. 结论速览（≤15 行）

1. **1 个分离端点及 1 个状态端点**：`space/app.py` 用 `gr.Server`（Gradio 6 Server 模式）+ `@server.api`，提供 `POST /gradio_api/call/separate` 与 `GET /api/status`；前端是手写的 Studio 静态页，静态资源挂 `/static`（`space/app.py:60,63,92,97`）。不是 Gradio Blocks，也不是纯 FastAPI。
2. **0 行服务侧复用、字节级模型侧复用**：`studio/server.py`（FastAPI + WebSocket，419 行）没有拷进 Space，所以 Space 没有实时/WS 能力，采用录完上传、服务端按最多 32 窗一批推理的方式。`build.py` 直接拷 `studio/falcon_runtime.py`、`falcon_model/`、`configs/`、`web/`，`test_static.py:40-41` 断言 `falcon_runtime.py` 与 `Falcon.yml` 和 studio 版字节相同。
3. **54 个文件 / 约 39.87 MB**：其中 `web/vendor/mediapipe` 独占 39.38 MB（98.8%），3 个 `.wasm` 合计 34.47 MB；`build.py` 0.19 s 完成，无警告。旧记 57 个文件 / 39,869,169 B 把 `falcon_model/`、`configs/`、`web/` 三个目录也计成了文件。
4. **226 = 122 + 93 + 10 + 1**：`node space/tests/check_i18n.mjs` 报 `i18n: 226 Chinese strings translated`，与 STATUS 的 226 逐字一致；来源是 `web/app.js` 122 + `web/index.html` 93 + `clip.py` 10 + `app.py` 1。合并后总表 442 条（studio 368 + space 74），两套 checker 覆盖不相交。
5. **3 passed in 0.15s**：`pytest space/tests/test_static.py -q` 与 STATUS 一致；`test_clip.py -m "not slow"` 20 passed / 2 deselected；`test_space.py` 在 `falcon` 环境 4 errors（缺 `gradio`/`spaces`），属预期内。
6. **12.08 MB 在用、23.36 MB 用不到**：`FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm', true)` 的第二个参数走 module 变体，实际只加载 `vision_wasm_module_internal.{js,wasm}`；另外两套占构建 58.6%。
7. **0 处落盘**：`space/app.py` / `space/clip.py` 无 `open(...w)`/`save`/`tempfile`；整帧视频不上传，只上传 96×96 嘴部裁剪。i18n 会把语言偏好写 `localStorage['falcon.language']`（`studio/web/i18n.js:7,110`）。
8. **164 与 166 两个数**：CI 只跑 164 个 repro 测试（`-k "not diff_scope"`，`tests.yml:111-112`），STATUS 记的 166 是全量收集数。其余等价命令本地全绿：8/8 node、3 passed、20 passed/2 deselected、imports OK、lock 门槛达标。
9. **10.611 G 与 26.51 G**：README 的 537,697 / 11,724,385（11.72 M）/ 10.611 G 全部复现。10.611 G 是 ptflops 音频侧、含 LSTM、不含视觉；视觉 88² 输入 15.897 GMAC，整窗合计 **26.51 GMAC、视觉 60.0%**（W2-11 §2.4）。Big 总参 12,345,305、音频 MACs 24.202 G。
10. **历史 CPU 命令失败**：`evaluated_mac_params.py:48,77-78` 硬编码 `.cuda()`，README:193 却把它列在无 GPU 前提的命令块里 → 无卡机器直接 `RuntimeError: No CUDA GPUs are available`（实测）。
11. **0 个 Space/Studio 入口**：项目页（817 行，内联 JS 111 行，`node --check` 通过）与 README 主表数字同源，但既没有 Space/Studio 链接，也没有 README:45 的 lookahead 更正注记。
12. **1 个零引用资产**：`assets/logo-source.png`（463 KB，1254²）零引用；`project_page/figures/*.pdf.png` 与 `assets/*.png` 同名不同源（尺寸/哈希都不同）。
13. **4 个 ERROR 而不是 skip**：`pytest space/tests`（space/README.md:59 的命令）在 `falcon` 环境 3.69 s 全红——`test_space.py:19` 只打了 `slow` 标记，没有「缺 gradio 就 skip」的保护。

## 1. 事实基线（path:line 为证）

### 1.1 文件清单与规模（`git ls-files` 实测，全部已跟踪，无未提交改动）

| 区域 | 文件 | 行/字节 |
|---|---|---|
| `space/` 代码 | `build.py` 80 行、`app.py` 133 行、`clip.py` 148 行 | 361 行 |
| `space/web/` | `app.js` 903 行、`index.html` 59 行（单行极长）、`i18n-space.js` 81 行 | 1043 行 |
| `space/tests/` | `test_static.py` 46、`test_clip.py` 193、`test_space.py` 102、`conftest.py` 2、`check_i18n.mjs` 42 | 385 行 |
| `space/` 文档 | `README.md` 63 行（HF 卡片，带 YAML front matter）、`README_zh.md` 43 行 | — |
| `space/` 依赖 | `requirements.txt` 8 行 | — |
| `.github/workflows/` | `tests.yml` 112 行、`pages.yml` 35 行 | 147 行 |
| `project_page/` | `index.html` 817 行（55,637 B）、`logo.png` 13,561 B、`figures/` 3 张 | 465 KB 合计 |
| `assets/` | 7 个文件 | 1.2 MB 合计 |
| 根 README | `README.md` / `README_zh.md` 各 276 行（数字集合完全一致，见 §3.5） | — |

`space/tests/__pycache__/*.pyc` 存在于工作区但**未被跟踪**（`.gitignore:2-3` 覆盖 `__pycache__/`、`*.py[cod]`；`git ls-files space` 只有 14 个文件）。

### 1.2 构建产物树（实测 `build.py --out /tmp/falcon-space-w109`，54 文件）

```
LICENSE                    12K     ← REPO/LICENSE（build.py:36）
README.md / README_zh.md    4K     ← space/README*.md（build.py:34），即 HF Space 卡
app.py / clip.py            8K     ← space/（build.py:34）
requirements.txt            4K     ← space/（build.py:34）
falcon_runtime.py          16K     ← studio/（build.py:30）
configs/{Falcon.yml,FalconBig.yml}
falcon_model/            212K      ← studio/falcon_model（models/ + layers/ + videomodels/ + PROVENANCE.md）
web/                     38M      ← studio/web（build.py:33）后被 space/web 覆盖 3 个文件（build.py:37-38）
  app.js(覆盖) index.html(覆盖) i18n-space.js(新增)
  i18n.js(注入修改) i18n-{static,dynamic,errors,audio}.js audio-inspector.* style.css
  capture-worklet.js video-worker.js mouth-tracker.js playback-resampler.js
  spectrum-{dsp,worker}.js logo.png
  vendor/mediapipe/{LICENSE,NOTICE.md,manifest.json,vision_bundle.mjs,face_landmarker.task,wasm/×6}
```

**没有**进入构建的 studio 内容：`server.py`（419 行实时服务）、`infer_file.py`、`tests/`（含 `test_*.cjs`）、`scripts/`、`README*.md`、`requirements.txt`——即“实时/离线 CLI/会议输出”这些入口在 Space 中不存在。

### 1.3 Space 卡（部署元数据，HF 读 `space/README.md` front matter）

`space/README.md:1-18`：`sdk: gradio` / `sdk_version: 6.29.1` / `python_version: "3.12.12"` / `app_file: app.py` / `license: apache-2.0` / `models: JusperLee/Falcon, JusperLee/Falcon-Big` / `preload_from_hub` 两行（`falcon.pt`、`falcon-big.pt`）。`test_static.py:31` 用 `startswith("---\ntitle: Falcon Studio")` 把首行锁死，防止 front matter 被误删。

`space/requirements.txt:1-8`：`torch==2.13.0`、`numpy==2.4.6`、`PyYAML==6.0.3`、`einops==0.8.2`、`pillow==12.3.0`、`huggingface_hub==2.2.0`、`spaces==0.51.3`；注释明说 **gradio 由 front matter 的 sdk_version 提供**，不在此文件。`fastapi` 也未列——`app.py:19-20` 用的 `FileResponse`/`StaticFiles` 来自 gradio 的传递依赖。

## 2. 机制拆解

### 2.1 `build.py`：装配 = 拷 4 样 + 覆盖 3 个 + 注入 2 行

1. **空目录保护**：`build.py:27-28` 目标非空直接 `SystemExit`（我实测先 `rm -rf` 再跑）。
2. **忽略模式**：`build.py:21` `__pycache__`、`*.pyc`、`.DS_Store`、`*.test.cjs` → 所以 `studio/web/*.test.cjs` 不进 Space，`test_static.py:30` 反向断言。
3. **四个 copytree**：`falcon_model`、`configs`、`web`（`build.py:31-33`）+ 单文件 `falcon_runtime.py`（`:30`）。
4. **覆盖顺序**：先拷 studio/web，再 `for item in (SPACE/"web").iterdir(): copy2(item, out/"web"/...)`（`build.py:37-38`）→ `space/web/{app.js,index.html,i18n-space.js}` 覆盖/新增，studio 的 `style.css`、`audio-inspector.*`、`video-worker.js` 等保持原样。
5. **i18n 注入（唯一“改文件”步骤）**：`build.py:39-44` 读 `web/i18n.js`，先用 `count(...)!=1` 断言锚点唯一（`:41-42`，改形状会立刻报错并提示“update I18N_IMPORT/I18N_SPREAD”），再插 `import spaceMessages from './i18n-space.js';` 并把合并行改成 `...audioMessages, ...spaceMessages`。
   → 结果：`spaceMessages` **最后展开、优先级最高**，可覆盖 studio 同 key 文案（实测有 4 处不同译法，见 §3.3）。
6. **推送**：`build.py:48-58` — `create_repo(repo_type="space", space_sdk="gradio", exist_ok=True)` → `upload_folder(delete_patterns="*")`（**先删后传，Space 侧文件与本次构建严格同步**）→ `get_space_runtime` 若不是 `zero-a10g` 就 `request_space_hardware(repo_id,"zero-a10g")`。
7. 未指定 `--out` 但给了 `--push` 时用 `TemporaryDirectory` 组装后再推（`build.py:74-76`）。

### 2.2 服务端：Gradio Server 模式（不是 Blocks、不是纯 FastAPI）

- `app.py:10` **第一行** `import spaces`（注释：必须早于 torch，ZeroGPU 要求）；`app.py:18-23` 再 import gradio / fastapi / `clip` / `falcon_runtime`。
- `app.py:60` `server = gr.Server(docs_url=None, redoc_url=None, openapi_url=None)` — 关掉 OpenAPI 文档面。
- 路由表：`POST /gradio_api/call/separate`（`app.py:63-80` 的 `@server.api(name="separate")`，由 gradio 生成），`GET /api/status`（`:83-89`），`GET /`（`:92-94`），`mount /static`（`:97`）。
- **静态 vs 动态界线**：页面、JS、wasm、模型权重文件全是静态文件；动态端点包括 `separate` 和状态接口。`index.html:57` `<script type="module" src="/static/app.js">`。
- **安全头**：`app.py:104-128` 自定义 ASGI 中间件，逐响应改写 `cache-control`（`/static/vendor/` 才给 86400，其余 `no-cache`，`:118`），加 `nosniff`/`no-referrer`/`permissions-policy: camera=(self), microphone=(self)`/CSP。CSP 见 `app.py:99-101`：`script-src 'self' 'wasm-unsafe-eval'`（放行 wasm 编译）、`worker-src 'self' blob:`、`frame-ancestors 'self' https://huggingface.co`（**与 Studio 唯一的实质差异**：`studio/server.py:269` 是 `frame-ancestors 'none'`，且 `connect-src` 多放行 `ws://127.0.0.1:*`）。
- `app.py:130-133` 本地 `server.launch(...)`：`ssr_mode=False`、`mcp_server=False`、`run_history=False`、`footer_links=[]`——即关掉 gradio 6 的服务端渲染/聊天历史/MCP，伺候纯手写前端。

### 2.3 请求协议（一条 JSON 走完整个分离）

`space/clip.py:105-148` `decode_request` 的契约：

| 字段 | 类型/约束 | 证据 |
|---|---|---|
| `variant` | `"small"`/`"big"`，缺省 small | `clip.py:110-112` |
| `audio` | base64(小端 float32)，长度 1–60 s；上限 `4*(60*16000*4+2)//3`≈5.12 MB base64 | `clip.py:114,121-122` |
| `frames` | `[base64(JPEG/PNG 96×96)]`，**单帧 base64 ≤ 40,000 字符**，条数 ∈ [len(audio)/(4·640), 3000] | `clip.py:28-29,125-127,137-144` |
| `times` | 每帧采集时刻（clip 采样轴，非降序，`[0, len+16000]`） | `clip.py:128-133` |

安全细节：`base64.b64decode(validate=True)`（`:117,141`）、`len(raw)%4` 检查（`:121`）、`np.isfinite` + `max|x|<=4` 幅度闸门（`:123-124`）、PIL 只认 `JPEG/PNG` 且 `size==(96,96)`（`:142-143`）、**同帧去重**（`decoded` 字典，按住画面的重复帧只解码一次，`:134,139-147`）。

响应（`app.py:76-80`）：`audio`(base64 float32) / `sample_rate` 16000 / `windows` / `gpu_ms` / `rtf` / `device` / `clipped_samples` / `variant`。`app.py:83-89` 的 `/api/status` 还暴露 `max_seconds`、`window_seconds`=2.0、`hop_seconds`=0.5、两变体的 `parameters`（实测 11,724,385 / 12,345,305）。

### 2.4 `clip.py`：把“实时 0.5 s 循环”折叠成“整段批处理”

常量（`clip.py:22-29`）：`SR=16000`、`FPS=25`、`WINDOW=32000`(2 s)、`HOP=8000`(0.5 s)、`GUARD=1280`(80 ms)、`FADE=320`(20 ms)、`FRAME=640`、`LEFT=WINDOW-HOP-GUARD=22720`、`MAX_SECONDS=60`。

- **窗口数** `plan(n)=ceil(n/HOP)`（`clip.py:32-34`）——与 Studio 每 0.5 s 发一窗完全对齐（`test_clip.py:130` 断言 `[plan(n) for n in (1,8000,8001,16000,960000)] == [1,1,2,2,120]`）。
- **取音** `window_audio(audio,w)`：窗口覆盖 `[w*HOP-LEFT, +32000)`，**窗外补零**（`clip.py:37-44`）。第 0 窗因 `start=-22720` 只有 9280 个真实样本（0.58 s），前 1.42 s 是零——这与 Studio 冷启动的空环形缓冲**同构**，所以不是 bug，是“第一块输出等于冷启动 Studio”的必然结果（我核算：保留段 `[22720,30720)` 恰对应 clip[0:8000]，见 `stitch` 的 `start,stop`）。
- **取画** `window_frames(times,w)`：50 个槽中心 `w*HOP-LEFT+(i+0.5)*FRAME`，`searchsorted` + 前后比较取**时间最近**、平局取更早（`clip.py:54-57`；`test_clip.py:119-126` 专门测平局）。
- **缝合** `stitch`：每窗取 `[WINDOW-GUARD-HOP, WINDOW-GUARD)` 即 8000 点，前 320 点与上一窗前视 320 点做 20 ms 线性交叉淡化，末尾裁剪到真实长度并 `clip(-1,1)`，统计 `clipped`（`clip.py:60-74`）。
  对照 `studio/server.py:100-119` 的 `OutputSlicer.take`：公式同（`start,stop` 同式、`FADE` 同、ramp 同），**差别是 Studio 有 `contiguous = endpoint-last_endpoint==HOP` 门槛**（`server.py:112-113`），Space 无此门槛而是无条件交叉淡化——因为 Space 的窗口永不跳窗，两者等价；`clip.py:9-10` 的 docstring 把这个前提写明。
- **视觉预处理** `normalize_mouth`：96→中心裁 88、/255、`(x-0.421)/0.165`（`clip.py:77-80`），与 `studio/falcon_runtime.py:131-140` 的 `preprocess_mouth` 同参（`test_clip.py:18,102` 直接 import 后者做基准）。
- **批处理** `separate`：`batch=32`，一次 `video_model(lips[B,1,50,88,88])` → `audio_model(wave[B,32000], emb[B,512,50])`，取 `output[:,0]`（`clip.py:83-102`）；非有限输出直接抛错（`:99-100`）。

### 2.5 GPU 配额策略

`app.py:46-49`：`gpu_seconds = min(30, 10 + plan(len(audio))//8)` → 1 s 片请求 10 s、8 s 片 12 s、60 s 片（plan=120）25 s；`@spaces.GPU(duration=gpu_seconds)`（`:52`）把“申请时长”与“实际用量”解耦（注释 `:47-48`：申请值要留在访客配额内，实际按用量计；实测 8 s small 0.2 s 热/1.2 s 冷、60 s big 2.5 s）。并发：`concurrency_limit=2 if ZERO_GPU else 1`（`:63`）。设备判定：`SPACES_ZERO_GPU∈{1,true}` → cuda，否则 `FALCON_DEVICE` 或 cpu（`:27,33`），**两模型在 import 期就 `.to(device)`**（`:31-43`）。

### 2.6 浏览器端能力边界（问题的核心答案）

| 环节 | 在哪跑 | 证据 |
|---|---|---|
| 摄像头/麦克风采集 | 浏览器，`getUserMedia`，`echoCancellation/noiseSuppression/autoGainControl=false` | `app.js:162,175-177` |
| 16 kHz 重采样 | AudioWorklet `capture-worklet.js` | `app.js:200-206` |
| 嘴部检测 | **浏览器 Web Worker**（`new Worker('/static/video-worker.js',{type:'module'})`）+ MediaPipe Face Landmarker，`delegate:'CPU'`、`runningMode:'VIDEO'`、`numFaces:2`、三档 confidence 0.6 | `app.js:369-394`；`studio/web/video-worker.js:29-37` |
| 帧来源 | `MediaStreamTrackProcessor` 优先，退化到 `createImageBitmap(video)`/canvas | `app.js:369-370,456-493` |
| 96×96 灰度裁剪 | Worker 内 OffscreenCanvas；退化路径用 `canvas.toDataURL('image/jpeg',.88)` | `app.js:493`；`studio/web/video-worker.js:12-13` |
| Falcon 前向 | **只在服务端 GPU**（ZeroGPU） | `app.py:52-57` |
| 上传时机 | **录完才发**，一次 POST + SSE 取结果（`/gradio_api/call/separate`） | `app.js:585-616,628` |
| 上传内容 | base64 float32 音频 + 每帧 96×96 JPEG base64 + 时间戳；**整帧视频不上传** | `app.js:543,628`；README.md:32-33 |
| 结果 | 只在浏览器内存/`Blob`，可试听、导出 16 kHz mono WAV | `app.js:750-766,767-777` |
| 不落盘 | 服务端 grep 无 `open(...'w')`/`save`/`tempfile`；仅 `Image.open(BytesIO)` 读内存 | `space/app.py`、`space/clip.py:141` |

**`vendor/mediapipe/wasm` 的真实用途**（这个问得准）：`video-worker.js:30` 把它当 `FilesetResolver.forVisionTasks(basePath,true)` 的 basePath。反编译 `vision_bundle.mjs:1` 里的路径公式：

```js
wasmLoaderPath = `${basePath}/vision_${`wasm${arg?"_module":""}${simd?"":"_nosimd"}_internal`}.js`
wasmBinaryPath = 同上 + .wasm        // arg = forVisionTasks 的第 2 个参数，默认 false
```

传 `true` ⇒ 走 **module 变体** ⇒ 浏览器实际取 `vision_wasm_module_internal.js/.wasm`（323,415 + 11,756,972 = 12.08 MB）。目录里另有 `vision_wasm_internal.*`（非 module SIMD，12.08 MB）与 `vision_wasm_nosimd_internal.*`（非 module 无 SIMD，11.28 MB），**共 23.36 MB 用不到**（占 39.87 MB 构建的 58.6%）。副产物风险：若浏览器无 SIMD，按公式会请求 `vision_wasm_module_nosimd_internal.js`——**该文件不存在**（只有 `vision_wasm_nosimd_internal.js`）→ 404 → `video-worker.js:48` 的“自动跟踪无法加载，请重试或切换手动模式”。

CSP 兼容性核对：wasm 编译需 `'wasm-unsafe-eval'` ✓（`app.py:99`）、module worker 需 `worker-src 'self'` ✓、wasm/模型 fetch 属 `connect-src 'self'` ✓、播放用 `blob:` ✓（`media-src`）。MIME 也过关：实测 Python 3.11 `mimetypes` 对 `.mjs`→`application/javascript`、`.wasm`→`application/wasm`，配 `nosniff` 不会拦（`face_landmarker.task` 无映射 → `text/plain`，但它是 `fetch→arrayBuffer`，不受 nosniff 影响）。

### 2.7 i18n 机制

- **key = 中文原文**，值 = 英文（`i18n-space.js:2` 注释：中文 key 兼作 zh 译文）。`t(key)`：`language==='zh'` 返回 key 本身，否则查表（`studio/web/i18n.js:61-66`）。
- **语言集只有 2 个**：`supported = new Set(['en','zh'])`（`i18n.js:8`），默认 `en`（`:12`），持久化到 `localStorage['falcon.language']`（`:7,110`），`<html lang>` 随之切换（`:90`）。
- **带占位符的句子也能互译**：`i18n.js:19-33` 把含 `{name}` 的条目编译成正则模板，**双向**匹配（zh→en、en→zh），用于 Worker/Python 回传的“语言中立”错误串（`:28-29` 注释）。
- **合并顺序**：static → dynamic → errors → audio → **space**（`i18n.js:6` + `build.py:43-44`）；同 key 后者胜，且**静默**（无冲突告警）。
- **缺翻译检测**（`check_i18n.mjs`）：只扫 `web/app.js`、`clip.py`、`app.py` 的**单引号/双引号字面量**（`:18-22`，且排除含换行的长串）+ `index.html` 的 `data-i18n*` 属性值（`:26`），然后对每个 key 断言 ① 英文里不含汉字（`:35`）、② 占位符集合相同（`:36`）、③ `index.html` 首屏英文等于 `t(key)`（`:38`）。“中文”这个 key 被显式跳过（`:32`，语言下拉项）。
- **硬编码中文漏网（我补测）**：`space/web/app.js`/`app.py`/`clip.py` 里**没有任何含汉字的模板字面量**（反引号串 0 命中）；`index.html` 里**没有非 `data-i18n*` 属性携带汉字**（唯一例外是语言下拉的裸文本 `中文`，`index.html:28`，checker 白名单）；运行时对全部 226 个 key 调 `t()`，**返回汉字者 0 个**。也就是说：扫描器有盲区（模板串、非 i18n 属性、Worker 内的中文），但实际代码**恰好没踩**。

### 2.8 CI 机制

**`tests.yml`**：触发 `push` / `pull_request` / `workflow_dispatch`（`:18-21`），`contents: read`（`:23-24`），`concurrency: tests-${{ github.ref }}` + `cancel-in-progress`（`:26-28`）。

| job | 步骤（行） | 等价验证（我本地跑的） |
|---|---|---|
| `frontend`（ubuntu，py3.11 + node22，不装 torch/gradio） | Studio Node 测试 8 个（`:44-50`，glob `studio/tests/check_*.mjs|*.cjs` + `studio/web/*.test.cjs`） | **8/8 PASS** |
| | Space 组装 + i18n（`:51-55`） | `build.py` 0.19 s；`check_i18n` 226 条 |
| | `pytest space/tests/test_static.py -q`（`:56-57`） | **3 passed** |
| | 锁文件门槛（`:58-69`）：存在 2 文件、`conda-env-lock.txt` ≥100 行、禁 `@ file://` | 124 行 = 9 行注释 + 115 个包 / 0 命中 ✓ |
| `python`（CPU torch 2.8.0 + torchvision 0.23.0） | 装依赖 + `pypesq` 兼容 shim（`:79-103`） | 无法复现装包（不装包规则）→ GAP |
| | 导入闸门 `import look2hear.*, audio_train, audio_visual_test`（`:104-106`） | **imports OK** |
| | `pytest studio/tests -q`（`:107-108`） | **53 passed**（STATUS 记 51+2 skip，见 §6） |
| | `pytest space/tests/test_clip.py -q -m "not slow"`（`:109-110`） | **20 passed, 2 deselected** |
| | `pytest tools/repro/tests -q -rs -k "not diff_scope"`（`:111-112`） | 收集 **166**，其中 2 条命中 `diff_scope` → CI 实际 164 |

`tests.yml:10-15` 明确写了**故意不跑**什么：`space/tests/test_space.py`（要 gradio/spaces）与 `test_clip.py -m slow`（要真权重），以及 `diff_scope` 两个历史任务写域守卫为何留在本地。这段注释与代码一致，是本仓库文档质量最高的部分之一。

**`pages.yml`**：`push` 到 `main` 且 `paths: project_page/**, .github/workflows/pages.yml`，或手动（`:5-9`）；权限 `pages: write` + `id-token: write`（`:11-14`）；`if: ${{ !github.event.repository.private }}`（`:22-23`，**仓库私有就静默跳过**）；artifact 只打包 `project_page`（`:31-33`），`environment: github-pages`。

### 2.9 项目页与资产

- 单文件静态页（817 行）：内联 CSS（`index.html:29-268`）+ 内联 JS（`:705-816`，111 行，`node --check` 通过），无构建步骤，符合 `pages.yml` 直接发目录的形态。
- 结构：`abstract`(309) / `contributions`(332) / `realssa`(356) / `method`(389) / `demos`(433) / `results`(581) / `ablation`(654) / `citation`(680)。
- 图片：`figures/Overall_Pipeline.pdf.png`（2411×524，`:399`）、`Separator.pdf.png`（1662×803，`:416`）、`TFRB.pdf.png`（3242×625，`:426`）、`logo.png`（128×128，`:20,275`）——**全部相对路径**，与 `pages.yml:31-33` 只发 `project_page/` 自洽。
- 外链：arXiv `abs/2610.32749`（`:295`）、GitHub（`:296`）、HF 模型（`:297`）、Live demo `https://u464727-dmja-7896a455.bjb3.seetacloud.com:8443/`（`:298`，**临时隧道域名**）；字体走 Google Fonts（`:25`）、图标走 cdnjs（`:26`）→ 项目页**不是**自包含的（与 Space 的 `'self'` CSP 完全相反）。
- 演示：4 个场景 `small/medium/large/outdoor` × 5 个系统 `Base / Falcon / Swift-Net-12 / Dolphin / AV-TF-GridNet`，音频和视频都来自 `https://oss.cslikai.cn/demos/...`（`:451,453` 等 26 条 URL），JS 用 `video.currentTime` 对齐隐藏 `<audio>`，偏差 >0.15 s 才 seek（内联脚本 `index.html:705-816` 的 `syncAudio` 段）。**只读配置、未发请求**（遵守任务约束）。

## 3. 关键数据表

### 3.1 构建产物构成（实测）

| 项 | 值 | 备注 |
|---|---|---|
| 文件数 | **54** | 按 `-type f` 计数（本轮复测）；旧记 57 = 54 个文件 + 3 个目录被计成文件 |
| 总字节 | **约 39.87 MB**（`du -sb` 39,872,115 B） | 本轮复测；旧记 39,869,169 B，差 2.9 KB，不影响 39.87 MB |
| `web/vendor/mediapipe` | **39.38 MB** | 占 **98.8%**（`du -sb` 39,380,889 B） |
| 其中 3 个 `.wasm` 二进制 | 34.47 MB | 34,474,168 B（配 .js 共 35.44 MB，占 88.9%） |
| 其中 `face_landmarker.task` | 3.76 MB | 3,758,596 B |
| 其中 `vision_bundle.mjs` | 155 KB | 155,439 B |
| `falcon_model/` | 212 KB | 模型代码副本 |
| py/js 代码 | < 300 KB | — |
| 构建耗时 | **0.19 s** | `time`（含 python 启动） |

### 3.2 MediaPipe wasm 三套变体（`studio/web/vendor/mediapipe/manifest.json` + 实测尺寸）

| 文件名 | js | wasm | 何时被选中 | 本 Space 是否用到 |
|---|---|---|---|---|
| `vision_wasm_internal.*` | 323,377 | 11,756,954 | 非 module + 有 SIMD | ✗ |
| `vision_wasm_module_internal.*` | 323,415 | 11,756,972 | **module + 有 SIMD**（`forVisionTasks(path,true)`） | ✓ **唯一在用** |
| `vision_wasm_nosimd_internal.*` | 323,180 | 10,960,242 | 非 module + 无 SIMD | ✗ |
| `vision_wasm_module_nosimd_internal.*` | — | — | module + 无 SIMD（公式会算出来） | **不存在 → 404** |

### 3.3 i18n 组成（实测 `node audit_i18n.mjs`）

| 表 | 条数 | 英文里含汉字 |
|---|---|---|
| `i18n-static.js` | 120 | 0 |
| `i18n-dynamic.js` | 165 | 0 |
| `i18n-errors.js` | 60 | 0 |
| `i18n-audio.js` | 23 | 0 |
| `i18n-space.js`（Space 增量） | **74** | 0 |
| **合并后总计** | **442** | **0** |

226 个被检查 key 的来源：`web/app.js` **122** + `web/index.html` **93** + `clip.py` **10** + `app.py` **1** = **226**（与 checker 打印完全一致）。

同 key 跨表重复 **20 处**，其中 **4 处英文不一致**（`spaceMessages` 后展开、静默覆盖 studio 译法）：

| 中文 key | i18n-errors（被覆盖） | i18n-space（生效） |
|---|---|---|
| `未知模型版本。` | Unknown model variant. | Unknown model version. |
| `音频数据无效或超出允许范围。` | Audio contains invalid or out-of-range samples. | Audio data is invalid or out of range. |
| `嘴部图像必须为 96×96 的 JPEG 或 PNG。` | Mouth frames must be 96×96 JPEG or PNG images. | Mouth images must be 96×96 JPEG or PNG. |
| `无法读取嘴部图像。` | Unable to read a mouth frame. | Could not read a mouth image. |

（studio 侧另有一套 checker：`node studio/tests/check_i18n.mjs` 打印 `10 localization checks passed (368 dictionary entries audited).` —— 368 恰好 = 442 − 74，两套 checker 分工互补。）

### 3.4 测试矩阵：声称 vs 实测

| 检查 | 声称（`docs/STATUS.md`） | 我的实测 | 一致？ |
|---|---|---|---|
| Space static | `:66` 3 passed | **3 passed in 0.15s** | ✓ |
| Space i18n | `:66` 226 条 | **226** | ✓ |
| Studio Node | `:65` 8/8 | **8/8 PASS** | ✓ |
| Studio Python | `:64` 51 passed / 2 skipped | **53 passed / 0 skipped**（`CUDA_VISIBLE_DEVICES=` 空） | ✗ 见 §6 |
| 复现工具 | `:63`（旧记 166 passed，现为 168 passed） | **168 collected**；有/无 GPU 均 168 passed（CI 用 `-k "not diff_scope"` 选 166） | ✓ |
| Space clip 排程 | CI `tests.yml:9,109-110` 隐含 | **20 passed / 2 deselected** | ✓ |
| Space HTTP/SSE | 明确不跑 | **4 errors**（缺 gradio/spaces） | ✓（预期内） |
| 锁文件 | `:22` “115 行” | **124 行 = 9 行注释 + 115 个包** | ✗ 见 §6 |

### 3.5 README 事实声明逐条核对

| # | README 声明（行） | 核对结果 | 证据 |
|---|---|---|---|
| 1 | 0.5 M 可训练参数（`:43,125,228`） | ✓ 音频侧 **537,697**（=0.538 M） | 我 CPU 实测 `sum(p.numel())` |
| 2 | 总参 11.72 M（`:122`） | ✓ **11,724,385**（537,697 + 11,186,688 冻结 FRCNN） | 实测；`STATUS:33` 同值 |
| 3 | Falcon-Big 12.35 M（`:123`） | ✓ **12,345,305**（音频 1,158,617 + 11,186,688） | 实测（README 未给 Big 的 MACs） |
| 4 | MACs 10.6 G（`:228`） | ✓ **10.611 G**（音频网络，**不含**冻结视觉编码器） | 实测 ptflops；`STATUS:33`。同口径补两项：视觉 88² 输入 **15.897 GMAC**，整窗 **26.51 GMAC、视觉 60.0%**（W2-11 §2.4）。2026-11-08 更正：任务简报里的「21.19 GMAC / 视觉 74.6%」是 `torch.utils.flop_counter` 的另一把尺子（不给 `aten::lstm` 记账，音频只剩 5.381 GMAC），两个数不能并列 |
| 5 | Big = `SpeakerContrastiveV2`（wider）（`:123`） | ✓ `studio/configs/FalconBig.yml:88`；`falcon_runtime.py:36` | — |
| 6 | Small = `FalconSpeakerContrastive`、`configs/Falcon.yml`（`:122`） | ✓ `studio/configs/Falcon.yml:10`；`falcon_runtime.py:34` | — |
| 7 | “causal = 窗口级因果，57 帧≈456 ms，非论文 120 ms”（`:45`） | ✓ 内部自洽：`Falcon.yml:28-29` win 256/hop 128 → 8 ms/帧 × 57 = 456 ms。这里的 120 ms 是**论文声明**（App E「visual stream buffers three future frames」与 §6.4「at most 15 frames」），不是代码事实；本条引用的 57 帧也只是一遍 AudioBlock 的几何值 | `STATUS:50` 同口径；三档数字见 §5-12 |
| 8 | 视觉编码器冻结、11.2 M（`:125`） | ✓ 11,186,688 | 实测 |
| 9 | 每次加载会 HEAD `config.yaml` 计入下载量，`HF_HUB_DISABLE_TELEMETRY=1` 可关（`:127`） | ✓ `falcon_runtime.py:39-40,100-113`（后台线程、best-effort） | — |
| 10 | 权重只含 `audio_model.*`/`video_model.*`（`:129`） | 未复核（属模型轮范围） | GAP |
| 11 | `falcon.pt` 来自 `LRS3_complex_nonmoving`、epoch 714（`:232`） | 配置快照指向 LRS3，发布权重与该次训练的身份关系尚未确认 | 但**不在** `checkpoints/` 里有可执行证据链 → GAP |
| 12 | 跨语料 LRS2 7.03/7.71、RealSSA-Sim 7.774/8.479（`:232`） | ✓ 与 `STATUS:35,38` 一致。7.03 是训练来源未知的发布权重在 LRS2 上的基线，不是论文 14.1 的复现；RealSSA-Sim 的 7.774 用 channel mean、SDRi 分母 9,166（W1-07 §3.4） | — |
| 13 | RealSSA-Sim 100,826/9,133/9,169、240 shards（`:84,87`） | ✓ `DATA_ACCESS_GUIDE.md:227,226` | — |
| 14 | RealSSA-Real 54 段/40 录音/10 spk/8 场景/83.99 min、original view 34 录音（`:85`） | ✓ `DATA_ACCESS_GUIDE.md:282` | — |
| 15 | 效率表 91.9% / 20.2× / 11.1×（`:61`） | ✓ 算术自洽：1−0.5/6.2=91.9%；213.8/10.6=20.2；567.8/51.3=11.07 | 同行表格 |
| 16 | 51.3 ms GPU 延迟 / 125.3 MB RAM / 364.7 MB 显存（`:228`） | ✗ 未复现（3090 vs 本机 A800；`STATUS:44` 记 A800 small 47.1 ms / big 56.5 ms） | GAP |
| 17 | Studio 在 M5 Pro Metal 每窗约 0.12 s（`:144`） | ✗ 无法复现（无 Apple 硬件） | GAP |
| 18 | Space 上限 60 s、录完再分（`:135`） | ✓ `clip.py:27`、`app.py:86` | — |
| 19 | 两个数据集 gated/manual、许可条款（`:84-85,89`） | ✓ `DATA_ACCESS_GUIDE.md:24`、`REGRESSION-docs.md:74` | — |
| 20 | 徽章 “Python 3.10+”（`:37`）vs Studio 要求 “3.10–3.12”（`:114`）vs Space “3.12.12” | 三处口径不同但不冲突（三个不同产物） | 提示级 |

`README_zh.md` 与 `README.md` 各 276 行，数字集合**完全相同**（脚本比对：仅英文有 = ∅，仅中文有 = ∅），未发现中英不一致。

## 4. 实测验证（命令 + 输出片段）

### 4.1 构建（`/root/miniforge3/envs/falcon/bin/python`，repo root）

```
$ python space/build.py --out /tmp/falcon-space-w109
Assembled /tmp/falcon-space-w109
0.03s user 0.08s system 61% cpu 0.188 total          # 退出码 0
$ find /tmp/falcon-space-w109 -type f | wc -l   →  57
$ du -sb /tmp/falcon-space-w109                 →  39869169
$ du -sh …/web                                  →  38M
```

### 4.2 i18n（node v22.23.3）

```
$ node space/tests/check_i18n.mjs /tmp/falcon-space-w109
i18n: 226 Chinese strings translated                  # 退出码 0
```
与 `docs/STATUS.md:66`「组装后 i18n 226 条」**逐字命中**。

### 4.3 Space 静态测试

```
$ CUDA_VISIBLE_DEVICES= python -m pytest space/tests/test_static.py -q
...                                                    [100%]
3 passed in 0.15s
```
三条分别是 `test_build_merges_space_files`、`test_build_reuses_the_studio_runtime_unchanged`、`test_every_chinese_string_has_an_english_translation`（第三依赖 node，`test_static.py:44` 有 `skipif`）。

### 4.4 我自己加的审计（写在 `/tmp/falcon-space-w109/`，未进仓库）

```
$ node audit_i18n.mjs
i18n-static 120 keys … i18n-space 74 keys … sum of tables 442
{ 'web/app.js': 122, 'app.py': 1, 'clip.py': 10, 'index.html': 93 } total 226
0/74 space keys undiscovered
untranslated at runtime: 0 []
$ node refcheck.mjs
… /static/{app.js,audio-inspector.css,capture-worklet.js,logo.png,style.css,video-worker.js} 全 OK，missing: 0
video-worker.js: /static/vendor/mediapipe/{wasm,face_landmarker.task} OK
```

### 4.5 clip 排程等价性（CI 同款命令）

```
$ python -m pytest space/tests/test_clip.py -q -m "not slow"
....................                                   [100%]
20 passed, 2 deselected in 6.44s
```
（2 条 slow 是 `test_real_model_matches_studio_live_path[small|big]`，真权重 CPU，CI 故意不跑。）

### 4.6 Space HTTP 测试在 `falcon` 环境的表现（反例）

```
$ python -m pytest space/tests/test_space.py -q
E  ModuleNotFoundError: No module named 'spaces'
ERROR space/tests/test_space.py::test_page_status_and_headers
ERROR space/tests/test_space.py::test_separate_round_trip[small]
ERROR space/tests/test_space.py::test_separate_round_trip[big]
ERROR space/tests/test_space.py::test_separate_reports_invalid_clip
4 errors in 3.69s
```
即 `space/README.md:59` 的 `python -m pytest space/tests` 在仓库默认环境里是**红**的（不是 skip）：`test_space.py:19` 只打了 `slow` 标记，没有任何“缺 gradio 就 skip”的保护。

### 4.7 CI 其余等价命令

```
$ for t in <8 个 node 测试>; do node "$t"; done
8/8 PASS（check_i18n.mjs 自报 “368 dictionary entries audited”）
$ python -m pytest studio/tests -q          → 53 passed in 42.72s
$ python -m pytest tools/repro/tests --collect-only -q → 166 tests collected in 6.44s（其中 2 条含 diff_scope）
$ python -c "… import look2hear.models, …, audio_train, audio_visual_test" → imports OK
$ wc -l docs/verification/conda-env-lock.txt → 124；grep -c "==" → 117；grep "@ file://" → 无命中
```

### 4.8 参数量 / MACs（CPU，`CUDA_VISIBLE_DEVICES=`）

```
[small] audio params=537697 trainable=0 video(frozen FRCNN)=11186688 total=11724385 runtime.parameter_count=11724385
[small] mouth embedding shape (1, 512, 50)
[small] audio MACs=10.611 G
[big]   audio params=1158617 … total=12345305
[big]   audio MACs=24.202 G
```
（`trainable=0` 是因为推理态全部 `requires_grad=False`，不影响“0.5 M 可训练参数”的口径。）

### 4.9 `evaluated_mac_params.py` 的无卡表现

```
$ CUDA_VISIBLE_DEVICES= python evaluated_mac_params.py
  File ".../evaluated_mac_params.py", line 48, in <module>
    audiomodel.cuda()
RuntimeError: No CUDA GPUs are available
```
同脚本还有两处硬编码值得记：`:39` 写死 `configs/Falcon.yml`（必须 repo root cwd）、`:81` 用 `torch.randn(1,1,50,64,64)` 构造视觉输入（**64×64**，与真实 96→88 不同；但 MACs 只统计 `audiomodel`，视觉部分不计入）。

### 4.10 项目页内联 JS 语法

```
$ python -c "抽取 <script> …" > page.js && node --check page.js
project_page inline JS: syntax OK   (111 行)
```

## 5. 坑与反直觉点

1. **`vendor/mediapipe` 一半以上是死重量**：39.87 MB 构建里 23.36 MB（58.6%）永远不会被这台 Space 取用（`vision_bundle.mjs:1` 的路径公式 + `video-worker.js:30` 传 `true`）。想瘦身只需保留 module+SIMD 那套，或按 SIMD 探测结果保留两套。
2. **无 SIMD 浏览器的回退路径是坏的**：公式会拼出 `vision_wasm_module_nosimd_internal.js`，仓库里没有这个文件 → 404 → 用户在 UI 上只看到“自动跟踪无法加载”（`video-worker.js:48`），根因在网络面板。
3. **Space 与 Studio 的 CSP 只差一处，但那一处是生死线**：`frame-ancestors 'self' https://huggingface.co`（`app.py:101`）vs Studio 的 `'none'`（`server.py:269`）。改错就整页在 HF iframe 里白屏，`test_space.py:76` 专门断言这串。
4. **第一块 0.5 s 输出是在 1.42 s 静音上下文里算出来的**：`LEFT=22720`（`clip.py:26,37-44`），第 0 窗只有 0.58 s 真实音频。它与“冷启动 Studio”等价（`test_clip.py:45-61`），但用户会以为前 0.5 s 质量差是模型问题。
5. **缝合函数少了 Studio 的 `contiguous` 门槛**：`server.py:112-113` 会检查 `endpoint-last_endpoint==HOP` 才交叉淡化，`clip.py:66-71` 无条件淡化。当前等价，但**一旦有人给 Space 加“跳窗/丢帧”逻辑，这两条路径会静默分叉**。
6. **i18n 合并在 4 个 key 上静默改写 studio 译法**（§3.3 表），没有冲突告警；`build.py:41-42` 只保护“锚点形状”，不保护“语义覆盖”。
7. **`build.py` 用 `delete_patterns="*"` 上传**（`build.py:53`）：Space 侧任何手工热修都会被下次 push 抹掉；同时 `--push` 不校验 front matter 的 `sdk_version` 与 `requirements.txt` 是否自洽。
8. **设备判定只靠一个环境变量**：`SPACES_ZERO_GPU`（`app.py:27`）不在 `∈{1,true}` 里就落到 CPU，而 `@spaces.GPU` 装饰器照样生效 → 表现为“GPU 申请到了但推理慢 100 倍”，日志里只有 `device:"cpu"`（`app.py:79`）。GAP：该变量名是否由 HF 运行时真实注入，无法离线证实。
9. **`pytest space/tests` 在仓库默认环境是红的**（§4.6），而 `space/README.md:59` 正是这么写的；按依赖拆开跑才对：`test_static.py`（py+node）/ `test_clip.py -m "not slow"`（falcon 环境）/ `test_space.py`（Space 依赖）。
10. **历史 `evaluated_mac_params.py` 写死 `.cuda()`**（`:48`），README:193 却把它和纯 CPU 命令并列 → 新人第一条命令就炸（§4.9）。
11. **项目页的 Live demo 是临时隧道**（`:298` `u464727-dmja-7896a455.bjb3.seetacloud.com:8443`），README:21 也链它；这类域名通常随实例回收而过期，是外审最容易点名的死链风险。
12. **项目页对外说 “Causal”，却缺 README 的自我更正**：`index.html:12,18,314,392,422` 只用“causal / causal constraints”，而 README:45 写“实测 lookahead 57 帧≈456 ms，不是论文的 120 ms”。那 120 ms 是**论文声明**（视觉 3 帧前瞻），代码实测三档为：单遍几何 57 帧/456 ms、整链几何 113 帧/904 ms、as-shipped 因 30 处 `cLNhw` 依赖到窗尾（2 s 窗时 ≤2000 ms，W2-12）。公开页面比仓库文档更“硬”。
13. **项目页不链 Space，也不链 Studio**：只有 Paper/Code/🤗 Models/Live demo（`:295-298`）。README:22-23 链了 Space 和 studio/README，两边入口不对齐。
14. **`project_page/` 没有 `CNAME`**，但 canonical/og:image 全用 `https://cslikai.cn/Falcon/`（`:8,14,19`），`pages.yml:1` 注释也这么写 → 自定义域必须在仓库设置外部维护；artifact 自带的信息无法自证。
15. **消融表不可复现**：`index.html:654-680` 给出 Downsampling / Recurrent Unit / Contrastive Loss / τ 四组数字（如 LSTM 单元 0.25 M / 5.69 G），但仓库 `configs/` 只有 6 个整模型 yml，**没有任何消融配置** → 项目页这部分只能信论文。

## 6. 与既有文档的冲突 / 更正

| # | 旧说法 | 实测/实际 | 判定 |
|---|---|---|---|
| C1 | `docs/STATUS.md:64`「Studio Python 51 passed / 2 skipped（2026-10-31）」 | 我在 `CUDA_VISIBLE_DEVICES=` 下跑 **53 passed / 0 skipped**（42.72 s） | **总数一致（53），拆分不同**。原因可定位：`studio/tests/test_device.py:35` 在“有 CUDA 或无缓存权重”时 skip、`:44` 在“有 CUDA”时 skip，`test_infer_file.py:13` 在权重未缓存时整模块 skip。STATUS 的数字是**有 GPU 可见**时跑出来的；空 `CUDA_VISIBLE_DEVICES` 时这两条会真跑。两处都对，但 STATUS 没写这个前提 → 建议补口径。 |
| C2 | `docs/STATUS.md:22`「conda-env-lock.txt（115 行…）」 | 实际 **124 行 = 9 行注释 + 115 个包**；文件头注释自称“共 115 个包”（`docs/verification/conda-env-lock.txt:1`）与实测一致。旧记的 117 是把第 3、7 行注释里的 `torch==` 也算成了包 | **口径写错**：115 是包数，不是文件行数（文件 124 行）。CI 门槛只要 ≥100 行，所以不影响绿灯。 |
| C3 | `docs/STATUS.md:63`「复现工具测试 166 passed」 | 现为 **168 passed / 168 collected**（有/无 GPU 同值）；CI 用 `-k "not diff_scope"` 选 **166**（`tests.yml:111-112`，且 `:12-15` 解释了原因） | **不是错，是两套口径**：168 = 全量本地；166 = CI。建议 STATUS 并列写清。 |
| C4 | `docs/STATUS.md:44`「效率（3090）… 未测（A800 单窗 small 47.1 ms / big 56.5 ms）」 | 我补测到的确定性数字：small **10.611 G MACs**、big **24.202 G MACs**、总参 11,724,385 / 12,345,305 | **新增**（README 只给 small 的 10.6 G，Big 的 MACs 全仓库无记载）。 |
| C5 | 隐含口径：README:228「Falcon（ours）0.5 M / 10.6 G」与基线同列比较 | MACs 只统计**音频网络**（`evaluated_mac_params.py:86-96` 只对 `audiomodel` 调 ptflops；视觉 embedding 是外部常量 `v`，见 `:80-82`），冻结 FRCNN 不参与。本机补测视觉 88² 为 **15.897 GMAC**，整窗 **26.51 GMAC、视觉 60.0%** | **口径需要标注**：与“AV-TF-GridNet 213.8 G”这种端到端数字是否同口径，本仓库无法自证 → GAP。论文表里的 10.6 G 只对应音频侧 |
| C6 | `docs/STATUS.md:66`「Space static 3 passed；226 条」 | 实测 3 passed in 0.15s；`i18n: 226 Chinese strings translated` | **一致**（本轮唯一被完整复核通过的 Space 断言）。 |
| C7 | README:45 / STATUS:50「causal = 窗口级因果，57 帧≈456 ms」 | 算术自洽（`Falcon.yml:28-29` hop 128 @16 kHz = 8 ms/帧），但它只是一遍 AudioBlock 的几何值；整链几何 **113 帧/904 ms**，as-shipped 依赖窗尾（W2-12）。项目页没有任何对应注记 | **仓库内部一致、对外不一致** → 项目页需补，且补的时候要给三档而不是一个数。 |
| C8 | `space/README.md:49-51` 分工声明（test_static/check_i18n 只需 py+node、test_clip 在 falcon、HTTP 测试需 Space 依赖） | 与实测完全吻合 | **准确**（但 `:59` 的 `pytest space/tests` 一键命令与之矛盾，见 §5.9）。 |
| C9 | `tests.yml:4`「Studio Node tests (8)」/`:51`「226 Chinese strings」 | 8 文件 glob 命中 8、8/8 PASS；226 精确命中 | **准确**。 |
| C10 | `docs/STATUS.md:23`「falcon.pt（small，epoch 714，525+137 keys）」/ README:232「来自 LRS3_complex_nonmoving」 | 仓库内互证（`P2F-report.md:36` 引 `Experiments/checkpoint/Falcon/conf.yml:99-102`），但**证据链在 docs 里，不在可执行代码里**；`checkpoints/` 目录不含可复算脚本 | **不算冲突，算证据薄弱** → GAP。 |

## 7. GAP LIST（≤10 条，可被下一轮直接认领）

1. **`SPACES_ZERO_GPU` 是否由 HF 运行时真实注入？** 若不注入，`app.py:27,33` 会让 Space 静默跑 CPU。认领方式：查 HF ZeroGPU 文档/`spaces` 包（`spaces==0.51.3`）源码里该环境变量名的出处，或读一次真实 Space 启动日志；给出结论与（若需要）改用 `torch.cuda.is_available()` 的方案对比。
2. **`space/requirements.txt` 的 7 个 pin 是否都真实存在**（`torch==2.13.0`、`numpy==2.4.6`、`pillow==12.3.0`、`huggingface_hub==2.2.0`、`spaces==0.51.3`、`PyYAML==6.0.3`、`einops==0.8.2`）以及 `sdk_version: 6.29.1` 是否有效。认领方式：只查 PyPI/HF 元数据（不装包），列出每个 pin 的最接近真实版本；若有不存在项，给出可用的最低可用版本组合。
3. **CI `python` job 的 `pip install` 行是否真的能装上**（`tests.yml:85-89` 里的 `torch-complex`/`torch-mir-eval`/`fast-bss-eval`/`ptflops` 等包名与版本约束）。认领方式：静态对照 PyPI 名称，或跑一次 `pip download --no-deps -d /tmp` 验证（需授权联网），输出“哪些名字是错的”。
4. **`window_frames` 在长录音 + 丢帧场景的等价性仍未覆盖**：`test_clip.py` 的 `capture_times` 只造 25 fps 规整抖动（`:64-67`），没有“某段完全无脸→沿用上一帧”“时间戳跳跃”等真实 Worker 输出。认领：新增 3–5 个退化用例（含 `times` 有重复值、`len(times)` 远小于 `len(audio)/FRAME`），验 `window_frames` 与 `studio_windows` 仍一致或明确记录差异。
5. **`clip.stitch` 缺 `contiguous` 守卫的隐患**：认领 = 给 `stitch` 加 endpoint 连续性参数并补一条单测（人为跳过一窗，断言输出与 Studio `OutputSlicer.take(contiguous=False)` 一致），使两条路径在未来改动时不会静默分叉。
6. **wasm 体积瘦身可行性**：实测只有 `vision_wasm_module_internal.*`（12.08 MB）在用，23.36 MB 为死重量。认领：核实“无 SIMD 浏览器”占比 → 决定保留 2 套还是 1 套；给出 build.py 的过滤补丁（只拷需要的变体）与对 `manifest.json` sha256 校验的影响评估。
7. **`evaluated_mac_params.py` 的 CPU 化**：`:48,77-78` 硬编码 `.cuda()`，`:81` 的 64×64 输入也与真实 96→88 不符（虽不影响音频 MACs）。认领：改成 `--device cpu` + 用 `preprocess_mouth` 真实预处理构造 embedding，并给出“改动前后 MACs 完全一致”的对照输出（我已在 §4.8 给出 CPU 基线 10.611 G / 24.202 G）。
8. **项目页与仓库文档的口径对齐**：补 README:45 的“窗口级因果/456 ms”注记、补 Space/Studio 入口链接、把临时隧道 Live demo 换成稳定入口或标注“临时”。认领：出一份 `project_page/index.html` 最小 diff（仅文案 + 2 个链接），并给出 meta/canonical 是否需要同步的判断。
9. **`cslikai.cn/Falcon/` 的托管真身**：仓库无 `CNAME`，`pages.yml:31-33` 只发 `project_page/`。认领：确认是 GitHub Pages 自定义域（仓库设置项）还是外部镜像；若前者，给出可提交的 `project_page/CNAME` 内容与 `pages.yml` 是否需要 `cname:` 的结论。
10. **Big 变体的独立质量证据缺失**：仓库有 `falcon-big.pt`（epoch 715）与 24.202 G MACs 的实测，但 README 结果表、STATUS 复现矩阵、项目页均无 Big 的任何 SI-SNRi/SDRi/DNSMOS。认领：跑一次 Big 在 RealSSA-Sim tt 子集（如 100 条）的评测，产出与 small 的对照表（1 GPU、限时），否则在 README 标注“Big 未经质量评测，仅演示用”。

## 8. 证据清单

**读过的源文件（repo 相对路径）**
- `space/build.py`(80) `space/app.py`(133) `space/clip.py`(148) `space/requirements.txt`(8)
- `space/web/index.html`(59) `space/web/app.js`(903) `space/web/i18n-space.js`(81)
- `space/tests/{conftest.py,test_static.py,test_clip.py,test_space.py,check_i18n.mjs}`
- `space/README.md`(63) `space/README_zh.md`(43)
- `.github/workflows/{tests.yml(112),pages.yml(35)}`
- `project_page/index.html`(817) `project_page/figures/*.png` `project_page/logo.png`
- `assets/{Overall_Pipeline.png,TFRB.png,separator.png,realssa_scenes.jpg,logo.png,logo-tile.png,logo-source.png}`
- 根 `README.md`(276) `README_zh.md`(276) `docs/STATUS.md`(141) `.gitignore`
- 交叉引用：`studio/server.py:29,100-119,269`、`studio/falcon_runtime.py:32-40,85-90,100-113,131-140,143-210`、`studio/web/{i18n.js,video-worker.js,mouth-tracker.js,app.js}`、`studio/configs/{Falcon.yml,FalconBig.yml}`、`studio/web/vendor/mediapipe/{manifest.json,vision_bundle.mjs}`、`evaluated_mac_params.py:39,48,77-96`、`docs/DATA_ACCESS_GUIDE.md:24,226-227,282`、`docs/verification/phase2/{P2E-report.md,P2F-report.md}`

**跑过的命令（全部远程、只读仓库、不装包、不占 GPU）**
```bash
ssh dev_a800_2gpus 'cd /root/code/github_repos/Falcon-fork && …'
# 1  git log/ls-files/status；find space .github project_page assets
# 2  python space/build.py --out /tmp/falcon-space-w109          → 54 files / 约 39.87 MB / 0.19 s
# 3  node space/tests/check_i18n.mjs /tmp/falcon-space-w109      → i18n: 226
# 4  pytest space/tests/test_static.py -q                        → 3 passed
# 5  pytest space/tests/test_clip.py -q -m "not slow"            → 20 passed, 2 deselected
# 6  pytest space/tests/test_space.py -q                         → 4 errors（缺 spaces）
# 7  pytest studio/tests -q                                      → 53 passed
# 8  8 个 studio node 检查                                        → 8/8 PASS（含 368 条词典审计）
# 9  pytest tools/repro/tests --collect-only -q                  → 166 collected（2 条 diff_scope）
# 10 python -c "import look2hear.…, audio_train, audio_visual_test" → imports OK
# 11 /tmp/falcon-space-w109/mac_cpu.py（自写，CPU）              → 537,697 / 11,724,385 / 10.611 G；1,158,617 / 12,345,305 / 24.202 G
# 12 /tmp/falcon-space-w109/audit_i18n.mjs（自写）               → 442 条表 / 226 来源 / 4 处译法冲突
# 13 /tmp/falcon-space-w109/refcheck.mjs（自写）                 → 静态引用 missing=0
# 14 mimetypes 检查 / wc -l lock / node --check page.js
```
**本次新建的临时产物（均在 `/tmp/falcon-space-w109/`，不属于仓库）**：构建树、`audit_i18n.mjs`、`refcheck.mjs`、`mac_cpu.py`、`page.js`。
