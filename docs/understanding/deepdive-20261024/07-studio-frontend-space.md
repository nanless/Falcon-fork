# 07 深挖：Studio 前端 + Space + project_page

> 小组：项目理解 + 模型复现规划深挖（studio-frontend）。只读分析，只写本文件。
> 仓库 `/root/code/github_repos/Falcon-fork`，分支 `repro/phase2`，HEAD `2beec38`。
> 基线：旧文档 `docs/understanding/07-studio-frontend.md`（1158 行，基线 commit `7576854`，当时 `studio/web/app.js` 1165 行）。
> 本文当前代码行号全部以 HEAD 为准（`app.js` 已 1223 行，见 §5）。
> 证据标记：`[代码]` 直接读源码；`[验证]` 本次实际运行（命令/输出见 §8）；`[推断]` 由源码/实测合理推断；`[文档]` 引用仓库已有文档。
> 范围：`studio/web/**`、`studio/tests/check_*.mjs|check_*.cjs`、`space/**`、`project_page/**`、`tools/**`、`.github/**`。

---

## 0. 一句话结论

Studio 前端是**无框架原生 ES Module 的完整实时采集/分离/回放客户端**（16000/32000/8000/1280/25 这组常数与 `studio/server.py` 逐项一致，WebSocket 私有协议两端互校），
成熟度高、可演示；本次新增的「采集卡顿诊断 + 设备启动超时上限」也已有 5 条 Node 控制流测试覆盖（Node 8/8 RC=0、`check_tracking_ui.cjs` 23 条）。
`space/` 不是独立实现而是**用 `build.py` 从 `studio/` 组装出来的 ZeroGPU 变体**：复用 runtime/模型/配置/全部 web 模块，只替换 `app.py`（Gradio 6 Server + `@spaces.GPU`）、`clip.py`（整段离线窗口调度）与两个前端文件；
本次实测组装 54 文件/39 MB、i18n 226 条中文全有英文译文（与 `REPRODUCTION_PLAN` §11.1 一致），`space/tests/test_clip.py` 在 `falcon` 环境 **20 passed/2 deselected**（其窗口/裁剪/拼接结果与 Studio 实时路径逐样本对齐）。
主要缺口是**运行期依赖与工程耦合**：Space 本体需要 py3.12/torch2.13/gradio 6.29.1（本机不可跑，只能静态+纯算法验证）；Studio 服务端**没有本地权重覆盖参数**且沿用已失效的 `models/` 报错文案；`i18n-errors.js` 与 `falcon_runtime.py` 现行英文异常出现 **5 处漂移**（中文界面会漏出英文）。

---

## 1. 边界与关键路径

### 1.1 范围与文件清单（HEAD 实测行数）

| 区域 | 文件 | 行数 | 角色 |
|---|---|---|---|
| Studio 前端 | `studio/web/app.js` | 1223 | 状态机/设备/WS/追踪 UI/录音/会议/诊断 |
| | `studio/web/index.html` | 64 | DOM 骨架，125 个 `data-i18n*` 绑定、69 个 id、仅 `<script type=module src=/static/app.js>`（`:62`） |
| | `studio/web/capture-worklet.js` | 71 | AudioWorklet：任意采样率 → 16 kHz（63-tap 多相 FIR） |
| | `studio/web/playback-resampler.js` | 129 | 16 kHz → AudioContext 采样率显式低通升采样 |
| | `studio/web/video-worker.js` / `mouth-tracker.js` | 166 / 149 | 25 fps 抓帧 + MediaPipe 检测 / 纯几何 ROI 跟踪 |
| | `studio/web/spectrum-dsp.js` / `spectrum-worker.js` | 208 / 9 | 1024 点 FFT 实时频谱与录音概览 |
| | `studio/web/audio-inspector.js` | 346 | 实时频谱 + 录音细览 UI |
| | `studio/web/i18n.js` + 4 字典 | 124 + 120/165/55/23 | 合并字典、模板反查、DOM 重渲染 |
| | `studio/web/vendor/mediapipe/**` | 38 MB | MediaPipe tasks-vision 1.0.1（本地化，无 CDN） |
| Studio 测试 | `studio/tests/check_i18n.mjs` 等 6 个 + `studio/web/*.test.cjs` 2 个 | 200/173/180/136/64/**518**/51/46 | Node 直测（本次 8/8 通过，§3.1） |
| Space | `space/build.py` | 80 | 从 `studio/` 组装、可选 push（`zero-a10g`） |
| | `space/app.py` | 133 | Gradio 6 `gr.Server` + `@spaces.GPU`，启动时加载 small+big |
| | `space/clip.py` | 148 | 整段录音 → Studio 同款窗口调度 + 批量推理 + 交叉淡化 |
| | `space/web/{app.js,index.html,i18n-space.js}` | 903 / 59 / 81 | record-then-separate 前端（无 WS） |
| | `space/tests/{check_i18n.mjs,test_clip.py,test_space.py,conftest.py}` | 42/193/116/2 | build 断言、窗口等价、HTTP/SSE 往返、i18n |
| 项目页 | `project_page/index.html` + `figures/*` + `logo.png` | 801 | 单文件静态论文项目页（内联 CSS/JS） |
| CI | `.github/workflows/pages.yml` | 35 | 只发布 `project_page/` 到 GitHub Pages |
| 工具 | `tools/repro/**` | 29 个 `.py` + 8 个 `.sh` | Phase1/2 数据与验证工具（含对 `studio/`、`space/` 的改动范围守卫） |

### 1.2 关键路径（本任务要求讲清的 7 条）

```
[实时] 麦克风 → capture-worklet(16 kHz,20 ms 包) → onPCM(ring 8 s + 漂移重锚)
      → maybeInfer(2 s 窗口/0.5 s hop/单飞) → frameWindow(50 帧 25 fps 网格)
      → WS /ws {infer, audio b64 f32, frames[50] b64 JPEG, end_sample=end+22720, reset}
      → server.decode_request → Engine.infer(非阻塞锁) → OutputSlicer(取 [22720:30720], 20 ms 交叉淡化)
      → WS result {audio 8000 样本, start/end_sample, contiguous, inference_ms}
      → handleResult(校验/录音对齐/电平) → playSamples(升采样/5 ms 淡入/抖动缓冲) → 耳机监听/虚拟声卡
[Space] 麦克风+摄像头 → 同一个 capture-worklet/video-worker（拷贝）→ 录制整段(≤60 s, 带时间戳的 96×96 crop)
      → buildClip(裁掉首尾无嘴部画面的部分) → POST /gradio_api/call/separate → SSE → clip.py 批量窗口 → WAV 试听/导出
[项目页] project_page/index.html（静态） → .github/workflows/pages.yml 上传 project_page → GitHub Pages；音视频在 oss.cslikai.cn
```

---

## 2. 数据流与消息协议表

### 2.1 实时采集：麦克风 → 16 kHz 定钟（`capture-worklet.js`）

| 步骤 | 位置 | 关键值/行为 |
|---|---|---|
| 目标率与比例 | `capture-worklet.js:8-9` | `targetRate=16000`、`ratio=sampleRate/16000`（对任意设备率通用） |
| 抗混叠 FIR | `:12-28` | 63 taps、128 相位、Hamming 窗 sinc、截止 `min(7200, .45*fs)`、系数归一化 |
| 群延迟补偿 | `:13,56` | `groupDelay=(63-1)/2=31`；`packetStart=(nextPosition-groupDelay)/sampleRate` |
| 输出静音 | `:35-36` | 连接到 destination 的输出恒填 0（原始麦克风绝不直通监听；`check_capture_dsp.cjs` 覆盖） |
| 时钟跳变复位 | `:38-42` | `expectedFrame !== currentFrame` → 清 history/相位/包缓冲 |
| 20 ms 包 | `:29,59-64` | `postMessage({samples:Float32Array(320), startTime}, [buffer])`（transfer） |

实测（本次重跑）：44.1 kHz 通带 RMS 0.70795、12 kHz 混叠 −60.68 dB、时间戳误差 RMS 0.00099；48 kHz 为 0.70672 / −60.13 dB / 0.00039，`clock_gap_reset:true`（§3.1 第 4 项）。

主线程 `onPCM`（`app.js:299-324`）：首次包锚定 `audioOrigin`；`|startTime - expected| > 80 ms` 视为掉包/挂起，重锚 + `resetTrackingWindow()`（`:303-307`）；
8 s ring 写入（`:15,309-313`，常量 `RATE*8=128000`），累计 `samples`，算输入 RMS 并刷新电平/波形/频谱，最后调用 `maybeInfer()`（`:315-323`）。

### 2.2 视频：25 fps 追踪 → 96×96 灰度 crop

- `startVideoCapture`（`app.js:446-530`）三档降级：`worker`（Worker + `MediaStreamTrackProcessor`，`:454-464`）→ `fallback`（Worker 检测 + 主线程 40 ms 抓帧，`:515-518`）→ 无 Worker（仅手动模式本地裁剪，`:526-529`）。
- worker 消息闭环：`ready` 握手 5 s 超时（`:468-477`）、`tracking` 状态（`:498`）、`frame`（`:499-506`，`wallTime` 换算到 AudioContext 时间轴：`time = context.currentTime - max(0, wallNow - message.wallTime)/1000`）、`ack/frame-done` 解除 `bitmapBusy`（`:491-496`）、`stopped/error` 停止采集（`:482-514`）。
- `addFrame`（`:531-537`）：入队并保持 ≤ `FPS*6=150` 帧 / 6 s 窗口；同时刷新诊断时间戳 `captureHealth.frameAt`（`:533`）。
- `frameWindow(startTime,endTime)`（`:577-596`）：在严格 25 fps 网格 `startTime+(n+.5)/25` 上取最近帧，共 50 帧；首窗前段用「epoch 后第一帧」合成历史（`:578-581,589`）；覆盖不足（首帧晚于窗口起点 120 ms、末帧早于终点 120 ms）返回 `null`（`:582-584`）；单帧偏差 >160 ms 直接放弃整窗（`:592`）。
- 服务端不感知帧时间戳，A/V 对齐完全由这个函数负责 → 是重要耦合点（§2.6）。

### 2.3 WebSocket 协议（`studio/server.py` ↔ `app.js`）

**客户端 → 服务端**（唯一发送点 `app.js:647`）：

```json
{"type":"infer","id":17,"audio":"<base64 LE Float32×32000>","frames":["<b64 JPEG>", "...×50"],
 "window_seconds":2,"hop_seconds":0.5,"end_sample":<end+22720>,"reset":false}
```

| 字段 | 前端产生 | 服务端校验（`server.py`） | 实测结论 |
|---|---|---|---|
| `type` | `"infer"` | 必须等于 `"infer"`（`:36`） | ✅ |
| `id` | `++state.sequence`（`:642`） | str/int、`len<=80`、int ≤2^53−1（`:252-256`） | ✅ |
| `audio` | `encodeFloat32`：LE Float32 → base64（`app.js:597-604`） | 长度必须 `4*((32000*4+2)//3)=170668`、`validate=True`、`<f4`、shape (32000,)、finite、max\|x\|≤4（`:43-50`） | ✅ 长度 170668，往返 bit-identical [验证] |
| `frames` | `frameWindow` 的 50 个 base64 | 必须是长度 50 的 list；每项 ≤40000 字符；PIL 可解码；**96×96**；**JPEG 或 PNG**；转 `L` 后 stack (50,96,96) uint8（`:51-63`） | ✅ 真 JPEG 与真 PNG 均通过；64×64 被拒 [验证] |
| `window_seconds`/`hop_seconds` | 字面量 2 / .5（`:647`） | 必须 ==2 / ==0.5（`:38-39`） | ✅ |
| `end_sample` | `end + LEFT_CONTEXT(22720)` | 非 bool int，`32000 ≤ x ≤ 2^53−1`（`:64-66`） | ✅ 首窗 = `sessionEpoch+32000` |
| `reset` | `needsReset \|\| missed>0`（`:637-649`） | 必须 bool（`:40-41`） | ✅ |
| 整条 | — | `≤2,000,000 B`，超则 close 1009（`:30,244-246`）；uvicorn `ws_max_size` 同值（`:332-333`） | ✅ 实测最坏约 0.65 MB |

**服务端 → 客户端**（`server.py:262-267`）：

| 字段 | 服务端值 | 前端用途/校验 | 结论 |
|---|---|---|---|
| `type` | `"result"`/`"error"` | 白名单，`error` 抛异常→停止（`app.js:718,728`） | ✅ |
| `id` | 回显 | 必须等于 `inFlight.id`（`:719`） | ✅ |
| `audio` | base64 LE Float32×8000 | 解码 + 非有限即抛（`:730`，`decodeFloat32:605-617`） | ✅ |
| `sample_rate` | 16000 | 必须 ===16000（`:729`） | ✅ |
| `inference_ms`/`rtf` | round(ms,1)/round(ms/500,3) | 显示（缺省本地计时，`:768-771`） | ✅ |
| `window_rtf`/`device`/`clipped_samples` | — | 前端未消费 | 冗余字段 |
| `start_sample`/`end_sample` | `endpoint-GUARD-HOP` / `endpoint-GUARD`（`:266`） | 必须整数、`end=inFlight.end+offset-GUARD`、`start=end-HOP`，再减回偏移（`:733-739`） | ✅ 数值链见下 |
| `output_delay_ms` | 80 | 性能提示参数（`:772`） | ✅ |
| `contiguous` | `not reset and endpoint-last==HOP`（`:82`） | 决定是否 reset 升采样器 + 5 ms 淡入（`:776,805,813`） | ✅ |

**时间戳语义链**（本次脚本实测）：`end_sample_client = end_real + 22720` → 服务端 `endpoint` → 回包 `end_sample = endpoint − 1280 = end_real + 21440`；
前端减偏移后得真实输出块 `[end_real−9280, end_real−1280)`，而 `OutputSlicer` 取 `pred[22720:30720]`，故协议块与请求音频第 22720 样本起**逐样本对齐**（`app.js:1021-1029` 的 `inputSpan` 偏移恒为 22720）。
本次实测：`slicer` 首块 `len=8000`、`contiguous=False`、等于 `pred[22720:30720]`；第二块 `contiguous=True`、交叉淡化头 `max|Δ|=0.0`；回包时间戳折回真实轴为 `[8000,16000)`（§3.3）。

### 2.4 推理调度（单飞 + 跳窗 + 排空）

- `maybeInfer`（`app.js:618-654`）：`running` 且无 `inFlight` 且 WS OPEN 且 `audioOrigin` 已定且追踪允许，且 `samples-sessionEpoch ≥ HOP+GUARD=9280`（0.58 s，`:621`）；`end = sessionEpoch+GUARD+floor((samples-sessionEpoch-GUARD)/HOP)*HOP`，首窗强制 `end=sessionEpoch+HOP+GUARD`（`:622-624`）；排空时每次只前进一个 hop（`:626`）。
- 跳窗：`missed = round((end-lastSentEnd)/HOP)-1`，累计 `skipped` 并提示，`reset=true`（`:636-641`）。
- 60 s `inFlight` 超时直接停采集（`:643-645`）；发送成功即 `lastSentEnd=end`、`needsReset=false`（`:648-649`）。
- `handleResult`（`app.js:715-783`）：id/epoch/采样率/长度/时间戳五重校验 → `lastResultEnd` 去重（`:740-741`）→ 写录音（Separated 与 Mixture 同长度对齐，`:745-759`）→ 播放 → `maybeInfer/continueDrain`。
- 停止排空（`app.js:900-941`）：补零到 hop 边界 + 重复最后一帧，按顺序送完剩余窗口；`finishDrain` 从尾部裁掉补零（`recording`/`mixture` 同步），等已调度音频播完（≤2 s）再 `stopCapture`；二次点击立即停。

### 2.5 播放链路

`playSamples`（`app.js:784-829`）：积压 >1.5 s 丢弃并重排（`:787-789`）→ `context.sampleRate≠16000` 时用 `createPlaybackUpsampler`（63 taps / 2048 相 / 7.4 kHz 截止）流式升采样（`:802-810`；动机是 Chrome `AudioBufferSourceNode` 跨率播放只做 2 点线性插值、无低通，见 `playback-resampler.js:1-17`）→ 非连续块 5 ms 淡入（`:813-817`）→ `when=max(now+30ms, playAt)` 精确排程（`:824-826`）→ 同时接 `monitorGain`（耳机）与 `meetingGain`（虚拟声卡）。

### 2.6 前端 ↔ 后端协议耦合点（改一处需同步哪里）

| 耦合点 | 前端位置 | 必须同步的对象 | 破坏方式 |
|---|---|---|---|
| 五常数 `16000/32000/8000/1280/25` | `app.js:6-11` | `server.py:28-29`、`falcon_runtime.py:24-27`、`space/clip.py:22-25` | 窗口/时间戳错位，或 `decode_request` 直接拒绝 |
| `LEFT_CONTEXT=22720` 私有偏移 | `app.js:10` | 无服务端常量，但 = `WINDOW-GUARD-HOP`（`server.py:80`）；`inputSpan` 依赖它 | 回包时间戳校验失败（`app.js:733-737`）→ 停止 |
| 请求/回包字段名与类型 | `app.js:647` / `:715-783` | `decode_request`（`server.py:35-67`）、`result` 字典（`:262-267`） | 任一端改名/改类型即 100% 不可用 |
| 帧图像契约 96×96 灰度 JPEG q=.88 | `video-worker.js`（经 `app.js:499-506`） | `server.py:59-60`（96×96 JPEG/PNG）、`falcon_runtime.preprocess_mouth` 88×88 | 报“嘴部图像必须为 96×96…” |
| 错误文案（中/英） | `i18n-errors.js` | `server.py` / `falcon_runtime.py` 的异常字符串 | 字典漂移→中文界面漏英文（本次实测 5 处，§4） |
| CSP / 静态路径 `/static/*` | `index.html:8-11,62`；worklet/worker/vendor 加载 | `server.py:198-211`（CSP）、`:282-283`（挂载） | 脚本/worker/wasm 被 CSP 拦 |
| `i18n.js` 结构 | `studio/web/i18n.js:1-6` | `space/build.py:22-23,41-44`（`I18N_IMPORT`/`I18N_SPREAD` 形状守卫） | build 直接 `SystemExit`（这是**有意的**失败快） |
| 窗口/裁剪/拼接算法 | `app.js maybeInfer/frameWindow/requestStop` | `space/clip.py`（重写）+ `space/tests/test_clip.py:25-61`（Studio 的 Python 转写） | `test_clip.py` 逐样本比对失败 |
| Space 前端副本 | `space/web/app.js`（903 行 fork） | `studio/web/app.js` | 双份维护，诊断阈值已出现差异（3000 vs 5000 ms 启动宽限） |

### 2.7 Space（record-then-separate）协议

Space 前端**没有 WebSocket**（`grep -n "WebSocket\|/ws" space/web/app.js` 无命中），改为 Gradio 的 `/call` + SSE：

1. 录制：`startRecording` 预分配 `maxSeconds*16000` 的 Float32 缓冲（`space/web/app.js:516-531`）；`onPCM` 只在 `running` 时写入（`:240`）；`addFrame` 把每个 crop 的绝对样本位置记进 `clipFrames`（`:448-455`，`position=(time-audioOrigin)*RATE-clipStart`）。
2. 收尾：`buildClip`（`:534-544`）以「第一帧前 20 ms」为起点、以「最后一帧 + 160 ms（Studio 的 stale 上限）」为终点裁掉首尾无嘴部画面的部分，中间按 25 fps 的 1/4 下限校验帧密度；`<1 s` 或帧过少直接报错不上传。
3. 上传：`callSeparate`（`:585-616`）POST `/gradio_api/call/separate`（headers 带 `x-gradio-user: app` 与 ZeroGPU 签名头，`:563-583` 通过 `postMessage` 与父页面握手）→ GET 事件流，解析 `event: complete/error`；5 min 超时 abort（`:622`）。
4. 服务端：`space/app.py:63-80` 的 `server.api(name="separate")` 调用 `separate_on_gpu`（`:52-57`，`@spaces.GPU(duration=gpu_seconds)`）→ `clip.separate`（`clip.py:83-102`）按 `plan()` 窗口数批量前向。
5. 回包：`{audio(b64 f32 整段), sample_rate, windows, gpu_ms, rtf, device, clipped_samples, variant}`（`app.py:77-80`）；前端 `applyResult` 校验长度一致后生成 WAV（`space/web/app.js:654-671`）。

Space 与 Studio 的**窗口数学完全同源**：`clip.plan = ceil(samples/HOP)`（`clip.py:32-34`）、`window_audio` 零填充区间 `[w*HOP-LEFT, w*HOP+HOP+GUARD)`（`:37-44`）、`window_frames` 取最近 crop（`:47-57`）、`stitch` 复刻 `OutputSlicer` 的 `[WINDOW-GUARD-HOP, WINDOW-GUARD)` + 20 ms 交叉淡化（`:60-74`）。

---

## 3. 测试结果（命令 + 实际输出）

### 3.1 必跑项①：8 个 Node 前端测试（`node v22.23.3`，全部 rc=0）

```bash
for t in studio/tests/check_*.mjs studio/tests/check_*.cjs studio/web/*.test.cjs; do node "$t"; done
```

| 测试 | 实测输出（摘录） |
|---|---|
| `check_i18n.mjs` | 8 项全 PASS；末行 `8 localization checks passed (363 dictionary entries audited).` |
| `check_mouth_tracking.mjs` | `PASS: mouth geometry, stable scale/articulation, continuous output without a face, automatic re-lock, jump confirmation, multi-face choice, fallback ownership, epoch races, encoding failure and stop.` |
| `check_playback_resampler.mjs` | 6 项 PASS：通带 ±0.1 dB、镜像 ≤−60 dB、6×0.5 s 分块与连续 push 逐位一致、长度精确、reset 清历史、identity → `ALL PASS` |
| `check_spectrum.mjs` | `"test":"PASS"`；`fft=1024 hop=640 rows=128`；5 min 概览 `samples=4800000, columns=1600, fft_windows=9600, milliseconds≈688`（性能观测值） |
| `check_capture_dsp.cjs` | `"test":"PASS"`；44100: passband_rms 0.70795 / alias −60.68 dB / timestamp_error 0.00099；48000: 0.70672 / −60.13 dB / 0.00039；`clock_gap_reset:true` |
| `check_tracking_ui.cjs` | **23** 条 PASS（真实 `app.js` + DOM stub），末行 `23 tracking UI control-flow checks passed. Real detector/camera/meeting validation is separate.` |
| `studio/web/capture-worklet.test.cjs` | `PASS: 16/44.1/48 kHz clocks, 1 kHz passband, 12 kHz alias rejection, FIR timing, clock-gap recovery.` |
| `studio/web/video-worker.test.cjs` | `PASS: 25 fps sampling, grayscale ROI, wall-clock messages, VideoFrame closure, error cleanup.` |

其中 `check_tracking_ui.cjs` 的 23 条包含本次新增的三类（与 `607352c` 的 diff 对应）：
`a frozen microphone is reported separately from an inference stall`、`device enumeration never blocks ready capture`、
`preview deadline releases devices and permits retry`、`suspended audio cannot hide permission request or hang startup`、`late permissions after timeout do not leak devices`。
即**卡顿诊断与设备启动超时这次是有测试的**，不是裸写。

### 3.2 必跑项②：Space 组装 + i18n

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python space/build.py --out /root/tmp/falcon-space-deepdive
node space/tests/check_i18n.mjs /root/tmp/falcon-space-deepdive
```

实际输出：

```
Assembled /root/tmp/falcon-space-deepdive          # build rc=0（纯标准库，无需 space 依赖）
i18n: 226 Chinese strings translated               # rc=0，与 REPRODUCTION_PLAN §11.1 的 226 一致
```

产物结构核对（等价于 `space/tests/test_space.py:74-79` 的断言，本次逐条手工复算）：

| 断言 | 实测 |
|---|---|
| 54 个文件 / 39 MB | ✅（vendor 34 MB + task 3.6 MB 占绝大部分） |
| `web/i18n.js` 含 `import spaceMessages from './i18n-space.js'` 与 `...audioMessages, ...spaceMessages` | ✅（`build.py:43-44`） |
| `web/app.js` 是 Space fork（首行 `// Falcon Studio on a ZeroGPU Space`） | ✅（studio 版被覆盖） |
| 无 `*.test.cjs`、无 `__pycache__` | ✅（`IGNORE` 规则 `build.py:21`） |
| `README.md` 以 `---\ntitle: Falcon Studio` 开头 | ✅（Space 卡 frontmatter） |
| 与 `studio/web` 的差异 | 只多 `i18n-space.js`，只少两个 `*.test.cjs`，`app.js`/`index.html` 被 Space 版替换 |

i18n 226 条的来源分解（自写脚本，镜像 `space/tests/check_i18n.mjs:17-26` 的扫描口径）：

| 来源 | 含汉字字面量 | 去重后 |
|---|---|---|
| `space/web/app.js` | 136 | — |
| `space/clip.py` | 12 | — |
| `space/app.py` | 1 | — |
| `space/web/index.html`（`data-i18n*`） | 110 | — |
| 合计（去掉语言选项「中文」） | — | **226** |
| 其中由 `space/web/i18n-space.js`（74 条）满足 | 74 | — |
| 由 Studio 的 363 条字典满足 | 152 | — |
| 缺失 | **0** | — |

### 3.3 补充实测③：协议互通（真实 PIL 解码，修正旧文档的“未覆盖”）

旧文档 §8.4 用 PIL stub，明确标注“JPEG 真实解码未覆盖”。本次 `falcon` 环境已有 Pillow，直接导入真实 `studio/server.py` 跑通：

```
server constants: {'SR': 16000, 'FPS': 25, 'WINDOW': 32000, 'HOP': 8000, 'GUARD': 1280, 'FADE': 320, 'MAX_MESSAGE_BYTES': 2000000}
LEFT_CONTEXT (client formula) = 22720 = 1.42 s
client audio b64 length: 170668 server expected: 170668
decode_request OK: (32000,) float32 (50, 96, 96) uint8 endpoint 32000 bit-identical: True
  window_seconds=3 -> rejected: 当前模型使用 2 秒窗口、0.5 秒步长。
  hop_seconds=.4   -> rejected: 当前模型使用 2 秒窗口、0.5 秒步长。
  reset=1          -> rejected: 重置标记必须为布尔值。
  frames=49        -> rejected: 每个窗口需要 50 帧同步嘴部图像。
  audio short      -> rejected: 音频必须是 32000 个 16 kHz float32 采样。
  audio NaN        -> rejected: 音频数据无效或超出允许范围。
  end_sample bool / <WINDOW -> rejected: 音频时间戳无效。
  frame 64x64      -> rejected: 嘴部图像必须为 96×96 的 JPEG 或 PNG。
  frame PNG 96x96  -> ACCEPTED（按 server.py:59 设计即允许 PNG）
  frame not image  -> rejected: 无法读取嘴部图像。
slicer first: len (8000,) contiguous False; chunk equals pred[22720:30720]: True
slicer second: contiguous True; crossfade head max|delta| = 0.0
server timestamps {'start_sample': 30720, 'end_sample': 38720} -> client real axis (8000, 16000) match: True
PROTOCOL-VERIFY-V2 PASS
```

### 3.4 补充实测④：`space/clip.py` 与 Studio 实时路径等价性

`space/tests/test_clip.py` 只依赖 numpy/PIL/torch/fastapi（都在 `falcon` 环境），可直接跑非 slow 部分：

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest space/tests/test_clip.py -q -m "not slow" -p no:cacheprovider
# 20 passed, 2 deselected in 2.33s
```

覆盖：5 组时长/随机种子的窗口与 crop 选择与 Studio 转写逐样本一致、`clip.separate` 与「逐窗 preprocess_mouth + 模型 + `OutputSlicer`」在 1e-6 内一致、tie 取更早 crop、`plan/stitch` 长度与截断、`decode_request` 的 12 类拒绝样例（含非 float32、NaN、时间戳倒序、超长帧、非图像）。
**这两个 deselected 是 `@pytest.mark.slow` 的真实权重用例**（`test_clip.py:135-150`：加载 small/big 并断言 `array_equal`），本次未跑（避免长耗时/占 GPU）。

### 3.5 必跑项③：Space 运行时依赖（未执行原因，如实记录）

`space/app.py` 与 `space/tests/test_space.py` **本次未执行**，原因是硬依赖缺失，而非跳过：

```
gradio  MISSING
spaces  MISSING
torch   2.8.0+cu128        # falcon 环境
Python  3.11.17            # falcon 环境
```

`space/requirements.txt` 要求 `torch==2.13.0`、`numpy==2.4.6`、`pillow==12.3.0`、`spaces==0.51.3`，
`space/README.md:6-8` 声明 `sdk: gradio`、`sdk_version: 6.29.1`、`python_version: "3.12.12"`；
而项目唯一允许的环境是 py3.11 + torch2.8（`AGENTS.md` §1/§4）。按任务约束**不安装**，因此：
`space/app.py`（`gr.Server`/`@server.api`/`@spaces.GPU` 的真实行为）、`test_space.py`（启动子进程 + `/gradio_api/call` 往返 + 响应头）只能做静态阅读，其结论在本文中一律标 `[代码]`。
可执行的部分已尽量执行：`build.py`（无依赖）、`check_i18n.mjs`（Node）、`test_clip.py`（算法层）。

---

## 4. 缺陷与风险

| # | 级别 | 问题 | 证据 | 影响 |
|---|---|---|---|---|
| D1 | **P1（复现演示）** | Studio 服务端无法指定本地权重：`server.py:290-298` 只有 `--host/--port/--device/--model/--no-browser`；权重经 `resolve_checkpoint` 走 HF 缓存/下载（`falcon_runtime.py:58-78`）。而 `infer_file.py:91,98` 是唯一支持 `--checkpoint` 的入口 | [代码] | 离线演示必须在 `~/.cache/huggingface` 预置权重，否则首启失败；这是“最小可跑路径”的最大卡点 |
| D2 | P2 | **失效的 `models/` 报错文案 + `available` 分支**：`server.py:119-124` 的 `available = weights_source() is not None` 在装了 `huggingface_hub` 时恒为真（返回 `'huggingface'`），`:120` 注释与 `:141` 报错“尚未安装…权重：models/…”、`i18n-errors.js:22` 的同一句都指向已不存在的 `models/` 目录 | [代码] | 用户按提示去建 `models/` 无效；真正的修复（HF 缓存/镜像）没说 |
| D3 | P2 | **i18n 反查漂移 5 处**：`falcon_runtime.py` 现行英文异常在 `i18n-errors.js` 中无对应模板 → 中文界面原样显示英文。实测 RAW：`Could not download … Set HF_ENDPOINT …`、`huggingface_hub is not installed …`、`CUDA is unavailable in this Python process; …`、`device must be 'auto', 'cpu', 'cuda' or 'mps'`（字典仍是旧的“auto、cpu 或 mps”，`i18n-errors.js:47`）、`Metal warmup failed; using CPU: …`（字典是 `Metal warmup unavailable…`，`:53`） | [验证] | 首启失败（D1）恰好落在未翻译文案上，中文用户看到英文 |
| D4 | P2 | Space 前端是 `studio/web/app.js` 的 903 行 fork，无共享模块；两份诊断阈值已不一致（Studio 启动宽限 5000 ms `app.js:54` vs Space 3000 ms `space/web/app.js:48`） | [代码] | 双份维护，任何修复要改两处；`space/web/app.js` 的录制/裁剪/上传逻辑**无 JS 级单测** |
| D5 | P2 | 60 s Space 录音的上传体量：1500 个 crop（实测 96×96 灰度 JPEG q.88：纯灰 439 B / 梯度 1077 B / 嘴部状 789 B / 纯噪声 7110 B）× base64 + 5.1 MB 音频 base64 | [验证]+[推断] | 中位约 8–13 MB、最坏约 19 MB 的单条 JSON；浏览器内存里还要同时存 `clipAudio`(3.84 MB) 与 base64 字符串，低端设备有压力 |
| D6 | P2 | Studio 服务端无 CSRF/多标签页隔离之外的限流；`Engine.infer` 非阻塞锁在锁被占时直接抛“另一个窗口正在使用模型”（`server.py:163-164`），前端把任意 `type:error` 当致命错误 `stopCapture`（`app.js:728,779-781`） | [代码] | 多标签页/瞬时忙会把可恢复错误升级为停止整场（旧文档 Q5 仍未改） |
| D7 | P2 | 追踪失效路径仍不可达：worker/`mouth-tracker.js` 从不产生 `requiresReacquire`，但 `app.js:381,392,395` 三处 UI 分支依赖它 | [代码]+[验证 grep] | “目标变化，请重新捕捉”文案是死分支（与旧文档一致，未变） |
| D8 | P2 | Space `app.py` 启动时**同时**把 small+big 载入并搬上 CUDA（`app.py:31-43`），ZeroGPU 每次调用只临时分配一张切片 GPU（`space/README.md:26`） | [代码]+[文档] | 显存翻倍；小 GPU 档位（如 zero-a10g 的切片）可能吃紧，且未做失败降级 |
| D9 | P3 | Space 的 CSP 放宽到 `frame-ancestors 'self' https://huggingface.co`，且没有 Studio 的 same-origin/TrustedHost 中间件（`app.py:99-128`） | [代码] | 公网服务按需设计，但本地以 `app.py` 起服务时缺少来源校验 |
| D10 | P3 | `project_page/index.html` 依赖 Google Fonts、Font Awesome CDN、`oss.cslikai.cn` 媒体与一个 seetacloud 临时隧道 URL（`:284`）；论文链接不存在（README:40 “Paper link will be added when available”），但页内已写 `@inproceedings{sang2026falcon, NeurIPS 2026}`（`:670-678`） | [代码]+[文档] | 离线/断网时排版与 demo 全废；引用信息早于论文公开，复现引用口径需谨慎 |
| D11 | P3 | 前端单测盲区：`audio-inspector.js` 的缩放/拖拽/键盘/进度条无测试；真实 MediaPipe/摄像头/虚拟声卡路径只有 `studio/tests/browser/test.js`（本环境无 Chrome）；`space/web/app.js` 无 Node 测试；`project_page`/`pages.yml` 无任何测试 | [代码] | 见 §6.4 |

---

## 5. 与旧文档 `07-studio-frontend.md` 的逐条差异

> 旧文档基线 `7576854`（`app.js` 1165 行）。`7576854..HEAD` 触及本范围的只有两个 commit：
> `607352c Bound device startup and expose capture stall diagnostics` 与 `a27b1e6 Add the Hugging Face ZeroGPU Space and link it in the READMEs`。
> 因此旧文档的**协议/算法/常量结论全部仍成立**，需要修正的是行号、以及新增能力/新增子系统的缺章。

### 5.1 【确认】（结论仍正确，本次复核通过）

| 旧文档结论 | 复核方式 | 结论 |
|---|---|---|
| 常数 `RATE=16000/WINDOW=32000/HOP=8000/GUARD=1280/FPS=25/MAX_RECORDING_SECONDS=300` 与后端逐项一致 | 读 `app.js:6-12` + `server.py:28-30` + 脚本打印 | ✅ 不变 |
| `LEFT_CONTEXT=22720`（1.42 s），**不是** 22400；且 == `OutputSlicer` 切片起点 | 脚本打印 `22720`、`pred[22720:30720]` 逐样本相等 | ✅ 不变（这是全文最关键的数值） |
| 音频走 base64 LE Float32（非 ArrayBuffer），长度 170668；帧是 50×base64 JPEG | 脚本实测长度 170668、50 帧、bit-identical 往返 | ✅ 不变 |
| 服务端时间戳 `start=endpoint−GUARD−HOP`、`end=endpoint−GUARD`，前端逐字段校验并减偏移 | 脚本实测 `(30720,38720) → (8000,16000)` | ✅ 不变 |
| `OutputSlicer` 首块 `contiguous=False`、后续 `contiguous=True`、20 ms 交叉淡化 | 脚本实测首块重合、第二块淡化头 `max\|Δ\|=0` | ✅ 不变 |
| 采集 worklet 63-tap / 128 相 / 截止 `min(7200,.45fs)` / 31 样本群延迟补偿 / 输出静音 | `check_capture_dsp` + `capture-worklet.test` 实测数字一致 | ✅ 不变 |
| 播放重采样 63-tap / 2048 相 / 7.4 kHz、分块逐位一致、identity 返回 `null` | `check_playback_resampler` 6 项 PASS | ✅ 不变 |
| 单飞 + 跳窗（`reset=true`）+ 排空 + 二次点击立即停 + drain 期间丢包 | `check_tracking_ui` 23 条中对应 7 条 PASS | ✅ 不变 |
| i18n 中文原文为键、英文为值、反查表、模板正则、`data-i18n*` 回放、首启英文、localStorage 键 | `i18n.js:1-124` 逐行核对 | ✅ 不变 |
| 125 个 `data-i18n*` 绑定（120 唯一键）、69 个 DOM id、`app.js` 66 个字面量 id 全存在 | 自写脚本计数 | ✅ 不变 |
| `requiresReacquire` 死路径；`lastOutput/frameId/backendReachable` 死字段；`frame-done` 兼容分支 | grep 复核 | ✅ 不变（`app.js:20-22,491,760`） |

### 5.2 【修正】（内容仍对，但数字/行号/结论需更新）

| # | 旧文档说法 | 现状 | 修正后 |
|---|---|---|---|
| M1 | 行号基线 `app.js` 1165 行（全文引用的 546/593/659… 等行号） | `app.js` **1223 行**（`bounded`+`captureHealth` 共 +58 行，插入点在 `:37-74`，其后所有行号 +~35/58） | 引用一律以本文 HEAD 行号为准（如 WS 发送 `:647`、`handleResult:715`、`requestStop:900`、`stopCapture:942`） |
| M2 | 字典 349 条 / 去重 337 键 / 12 个同名键 | 实测 **363 条 / 351 唯一键 / 12 个重复键且值完全相同**（`i18n-dynamic.js` 由 151 → 165，新增 14 条：6 条诊断提示 + 8 条设备启动/超时文案） | 363 / 351 / 12 |
| M3 | `check_i18n.mjs` 审计 349 条 | 实测末行 `8 localization checks passed (363 dictionary entries audited).` | 363 |
| M4 | `check_tracking_ui.cjs` 443 行、**18** 条断言 | 现 **518 行、23 条断言**；新增 5 条覆盖冻结麦克风诊断、设备枚举不阻塞、预览超时释放设备、挂起音频、超时后迟到授权不泄漏设备 | 518 行 / 23 条 |
| M5 | 降级矩阵无「采集卡顿」行；`C10 超时表` 无设备启动相关项 | 新增 `captureHealth`（`app.js:41-67`）：5 s 启动宽限后每秒检查；输入/帧 3 s 无更新、`inFlight` >5 s、无结果 >5 s、输入 RMS<1e-5、输出 RMS<1e-6 → `performanceHint` 文案 + 一次 `console.warn`；`window.falconDiagnostics()` 暴露聚合快照（无 PCM/图像/设备标识） | 新增 §2.1/§2.2 与 §5.5 行；超时改为 `enumerateDevices 8 s`、`getUserMedia 45 s`、`context.resume 8 s`、`cameraVideo.play 12 s`、`addWorklet 15 s`（`app.js:182,245,248,266,269`） |
| M6 | `openCapture` 里 `await enumerateDevices()`（设备枚举会阻塞“开始”） | 改为 fire-and-forget：`void enumerateDevices().catch(...)`（`app.js:288`），且枚举本身 8 s 上限 + generation 守卫（`:182-183`） | 首启不再被慢设备驱动拖住 |
| M7 | 旧文档 §8.4：“JPEG 真实解码/96×96 格式校验未被本次执行覆盖” | 本次在 `falcon` 环境（Pillow 已装）真实解码 50 张 96×96 JPEG 并通过 `decode_request`；64×64 被拒、PNG 被接受 | **该局限已消除**（§3.3） |
| M8 | 旧文档 §8 环境为 `/root/.venvs/falcon`（3.10.18，无 PIL） | 现为 `/root/miniforge3/envs/falcon`（3.11.17，有 Pillow/fastapi/torch）；`space/tests/test_clip.py` 因此可跑 | 旧环境说明作废（`AGENTS.md` 已规定唯一环境） |
| M9 | 旧文档 scope 只覆盖 `studio/web/` | `space/`、`project_page/`、`tools/`、`.github/` 全部缺失 | 本文新增 §1.1、§2.7、§6.3、§6.5 |

### 5.3 【推翻】/【新增】

| 项 | 说明 |
|---|---|
| 推翻：`node --check` 13 个文件 | 现范围内共 **24 个** JS/MJS 文件（+`space/web/{app,app?}.js`、`i18n-space.js`、`space/tests/check_i18n.mjs` 等），本次全部 `node --check` 通过（`checked=24 fail=0`） |
| 新增：`space/` 子系统 | 组装方式、ZeroGPU 约束、record-then-separate、`clip.py` 作用、226 条 i18n、依赖缺口（§1.1/§2.7/§3.2/§3.4/§3.5/§6.3） |
| 新增：`project_page/` + `.github/` | 静态单页 + Pages 工作流 + 与 README 论文素材的关系（§6.5） |
| 新增：`tools/repro` 对本范围的守卫 | `tools/repro/tests/test_phase1_regression.py:468-492` 把 `studio/` 列为“保护源码”、`space/*` 列为“允许的新文件”；`test_phase2_pipeline.py:283-300` 同类守卫；`test_phase1_regression.py:356-361` 会跑 `pytest studio/tests` 做回归 → **动 `studio/` 会触发其它阶段的 diff-scope 测试失败**（复现时需注意） |
| 新增：Space 与 Studio 的常数重复 | `space/clip.py:22-25` 复制了 SR/FPS/WINDOW/HOP/GUARD/FADE；`space/web/app.js:9-11` 复制 RATE/FPS/STALE → 第三、第四处需要同步的地方 |

---

## 6. 工程完成度评估

### 6.1 可演示性（高）

- 本地：`studio/scripts/launch.sh` → `server.py` 自选 7860–7899 端口、自动开浏览器（`server.py:304-331`），首屏即完整 UI；模型可 `/api/load` 也可启动自动加载（`lifespan`，`server.py:185-192`）。
- 浏览器内全栈自包含：字体/脚本/MediaPipe wasm+model 全部同源（`vendor/mediapipe/manifest.json`，38 MB），CSP 只放 `wasm-unsafe-eval`（`server.py:210`）——**可离线演示 UI 与追踪**，只有权重需要网络。
- 演示断点只有一个：**权重**（D1）。若 `~/.cache/huggingface` 无 `falcon.pt`，`/api/load` 会走下载；本次机器无法联网，所以“最小可跑路径”必须预置缓存（§6.5）。

### 6.2 可维护性（中）

- 优点：无构建步骤、无框架、模块边界清楚（i18n/采集/播放/频谱/追踪各自独立文件），Node 直测门槛低（`node xxx.mjs` 即可）；新加的诊断是**单一职责 + 可注入式**（`window.falconDiagnostics()` 只输出聚合量，便于用户在控制台自查）。
- 风险：`app.js` 1223 行仍是单文件巨型状态机（50+ 字段 `state`，`app.js:13-31`），事件绑定与状态迁移耦合；`space/web/app.js` 是 fork 而非共享模块（D4）；字典以中文原文为键，新增文案忘记入字典不会报错（`t()` 未命中原样返回，`i18n.js:44-59`），只能靠 `check_i18n` 的 HTML 校验兜底。
- 质量信号：测试密度不错（8 个 Node 测试 + 518 行控制流测试 + 2 个 Python）；本次实测 8/8 通过、无 flaky。

### 6.3 可移植性 / Space 与 Studio 的差异（重点）

| 维度 | Studio（本地/GPU 服务器实时） | Space（ZeroGPU） |
|---|---|---|
| 交互模型 | 推流：每 0.5 s 一次 WS 请求（`app.js:618-654`） | 录制整段（≤60 s）→ 一次性 POST + SSE（`space/web/app.js:585-616`） |
| 传输 | `ws://host/ws`，JSON 文本，单条 ≤2 MB | Gradio `/gradio_api/call/separate` + 事件流，body 为 `{data:[payload]}` |
| 推理调度 | 单窗口、单飞、`OutputSlicer` 状态跨请求 | 整段窗口批量（batch=32，`clip.py:83-102`），`stitch` 一次重放交叉淡化 |
| 设备 | `cuda`/`mps`/`cpu`（`pick_device`，`falcon_runtime.py:81-92`） | `SPACES_ZERO_GPU=1` → `cuda`，否则 `FALCON_DEVICE`（默认 cpu）（`app.py:27,33`） |
| GPU 配额 | 自己的机器，无配额 | 每次调用申请 GPU，按访客 HF 账号计配额；`gpu_seconds=min(30, 10+windows//8)`（`app.py:46-49`），60 s 录音=120 窗 → 请求 25 s，实测只用 2.5 s（`space/README.md:39`、`app.py:47-48` 注释） |
| 模型加载 | 按需 `/api/load`（一次一个变体） | 启动时 small+big 全加载并搬上 cuda（`app.py:31-43`）→ 显存×2（D8） |
| 功能裁剪 | 耳机监听、会议虚卡、5 min 录音、实时频谱 | 无监听/无会议/只保留一段结果；`space/web/app.js:32-35` 明确隐藏输出实时频谱并改文案 |
| 结果一致性 | 基准 | `clip.py` 复刻窗口/裁剪/淡化；`test_clip.py` 在 mock 上 1e-6 内一致、真实权重 `array_equal`（slow 用例，本次未跑） |
| 部署 | 手动/脚本 | `build.py --push` → `HfApi.upload_folder(delete_patterns="*")` + 请求 `zero-a10g`（`build.py:48-58`） |

结论：Space 是**同源派生的受限部署档**，不是第二套实现——除 `app.py`/`clip.py`/2 个前端文件外全部由 `build.py:30-38` 拷贝；代价是两处常数/算法需人工同步（§5.3）。

### 6.4 浏览器兼容 / 麦克风权限 / 延迟边界

| 维度 | 事实 | 证据 |
|---|---|---|
| 安全上下文 | `getUserMedia` 前检查 `isSecureContext`；`127.0.0.1` 视为安全；否则提示用启动脚本地址 | `app.js:224` |
| 权限弹窗顺序 | 在用户手势内先 `context.resume()`（不 await），再发 `getUserMedia`，兼容 Safari 的“必须手势内 resume” | `app.js:232-240`；45 s 超时 `:245`；8 s 音频启动超时 `:248` |
| 迟到授权 | 授权在超时/取消之后才返回时，按 `generation` 关闭轨道，避免泄漏设备 | `app.js:241-246,249`；测试 `late permissions after timeout do not leak devices` |
| 采集约束 | `echoCancellation/noiseSuppression/autoGainControl` 全关、单声道（分离模型要求干净混合） | `app.js:239` |
| 后台标签页 | 仅 `videoMode==='worker'`（Chrome 的 `MediaStreamTrackProcessor`）可后台继续；其它模式 `visibilitychange` 直接停采集并保留片段 | `app.js:1209-1214`；说明文案见 `index.html:61` |
| 浏览器差异 | 无 Worker/OffscreenCanvas → 自动追踪不可用、只能手动；无 MSTP → 主线程 40 ms 抓帧 | `app.js:454-455,526-529` |
| 会议输出 | 需 `setSinkId`（Chrome），并拒绝把虚拟麦克风当输入（防反馈） | `app.js:217-219,845-848` |
| 延迟边界 | 首输出需 0.58 s 有效输入（`HOP+GUARD=9280`）；稳态设计延迟 0.7–0.8 s（0.5 s hop + 80 ms guard + 推理 + 播放缓冲）；快于实时时 0.5 s 更新 | `app.js:621-624`；`studio/README.md:99` |
| 过载 | 单飞 + 跳窗 + `reset`；`inFlight` 60 s 超时后停采集；播放积压 >1.5 s 丢弃重排 | `app.js:636-645,787-789` |
| Space 侧 | 必须 HTTPS（HF 页面iframe），`postMessage` 握手拿 ZeroGPU 签名头，5 min 分离超时 | `space/web/app.js:563-583,622` |

### 6.5 前端单测覆盖盲区（明确列出）

| 盲区 | 现状 | 风险 |
|---|---|---|
| `audio-inspector.js` 交互（缩放 1–16×、pan、点击/拖动/键盘 seek、进度条、ARIA） | 仅读码；Node 测试只覆盖 worker 协议 | 中：可视化坏了不影响分离，但“录音细览”是卖点 |
| 真实 MediaPipe / 摄像头 / 虚拟声卡 | `studio/tests/browser/test.js` 需 Chrome + 人像；本环境不可跑 | 中：追踪质量、检测耗时无实测证据 |
| `space/web/app.js`（录制/裁剪/上传/SSE 解析） | **无任何 JS 测试**；只被 `check_i18n.mjs` 扫字符串 | 高：`buildClip` 的裁剪边界、SSE 事件解析、ZeroGPU 握手都只靠代码评审 |
| `space/app.py` 运行时 | 需 gradio/spaces，本机不装 → 只静态 | 高：`gr.Server`/`@server.api`/`@spaces.GPU` 行为未经执行验证 |
| `project_page` / `pages.yml` | 无测试 | 低：静态页，但外链/媒体失效不会被发现 |
| 协议端到端（真 WS + 真模型） | `check_tracking_ui` 用 stub 的 WebSocket/fetch；真推理需权重 | 中：本次用脚本做了 `decode_request`/`OutputSlicer` 的**真实**互通（§3.3），但没有浏览器↔服务端全链路 |
| 采集/播放重采样 96 kHz | 旧文档已标 `[推断]`，仍无实测用例 | 低 |

### 6.6 复现演示的“最小可跑路径”（按优先级）

**P0 本地实时（唯一能演示完整功能的路）**
```bash
cd /root/code/github_repos/Falcon-fork
# 1) 确认权重来源（本次机器无网络：必须提前把 falcon.pt 放进 HF 缓存）
PYTHONPATH=studio /root/miniforge3/envs/falcon/bin/python -c \
  "from falcon_runtime import weights_source; print(weights_source('small'))"   # 期望 'cached'
# 2) 起服务（本机 2×A800，auto 会选 cuda；纯 CPU 加 --device cpu）
cd studio && PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python server.py --host 127.0.0.1 --port 7860 --device auto
# 3) 浏览器打开 http://127.0.0.1:7860 → Enable devices → 等 AUTO 绿框 → Start separation
```
反例记录：`server.py` **没有** `--checkpoint`；离线场景只能用 `infer_file.py --checkpoint <本地文件>` 做离线单条推理（`infer_file.py:91,98`），或先把文件放进 `~/.cache/huggingface/hub/models--JusperLee--Falcon/...`。

**P1 Space（本机不可跑，只能离线验证部分）**
```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python space/build.py --out /root/tmp/falcon-space-deepdive   # ✅ 实测通过
node space/tests/check_i18n.mjs /root/tmp/falcon-space-deepdive                                               # ✅ 226 条
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest space/tests/test_clip.py -q -m "not slow"   # ✅ 20 passed
# 完整 Space 需要独立 py3.12 环境：pip install -r space/requirements.txt gradio==6.29.1 然后 cd build && python app.py
```

**P2 项目页**：`project_page/index.html` 直接双击可看（字体/媒体需联网）；发布由 `pages.yml` 在 `main` 且 `project_page/**` 变更时触发（`.github/workflows/pages.yml:5-9`），演示音视频不随仓库发布（在 `oss.cslikai.cn`）。

---

## 7. 未解问题

| # | 问题 | 需谁确认 | 现状 |
|---|---|---|---|
| Q1 | Space 的 `gr.Server`/`@server.api`/`@spaces.GPU(duration=callable)` 是否与 `sdk_version: 6.29.1` + `spaces==0.51.3` 真实匹配？`duration` 回调收到的参数是否就是 `(variant, audio, crops, times)`？ | 有 HF ZeroGPU 权限的人 | 本机无 gradio/spaces，未执行；`test_space.py` 只覆盖本地 CPU 模式（`concurrency_limit=1`），ZeroGPU 分支完全未测 |
| Q2 | `gpu_seconds` 的 10 基准 + `windows//8` 是否足够覆盖 cold start？注释称 60 s Big 用 2.5 s，但请求 25 s 只是配额上限还是排队权重？ | Space 维护者 | 影响访客配额消耗与排队，无法本地验证 |
| Q3 | Studio 是否应补 `--checkpoint`/`FALCON_CHECKPOINT`（D1）？`infer_file.py` 已有，服务端缺失是遗漏还是有意？ | studio-backend / Lead | 影响离线复现演示 |
| Q4 | `server.py:141` 的 `models/{checkpoint}` 报错与 `available` 语义（D2）如何修：改文案、或恢复本地 `models/` 查找？ | studio-backend | 当前提示会把用户引向无效路径 |
| Q5 | D3 的 5 条英文异常是否补进 `i18n-errors.js`？是否应把 `falcon_runtime.py` 的异常消息也纳入 `check_i18n` 的自动审计（目前只审 HTML 与固定样例）？ | frontend + backend | 补字典是低风险改动，但需同步英文模板 |
| Q6 | 60 s 录音 ~8–19 MB 的 POST 是否会被 Space 网关/浏览器限制拦？是否需要分片或降低帧率？ | Space 维护者 | 仅按实测 JPEG 尺寸推断，无真实上传证据 |
| Q7 | Space 前端与 Studio 前端的 fork 是否应抽公共模块（D4）？抽的话 `build.py` 的“拷贝 studio/web”组装模型要改。 | Lead / 架构 | 现状靠 `check_i18n` 与 build 断言兜底 |
| Q8 | `project_page` 的 NeurIPS 2026 bibtex 与“论文未公开”不一致（D10），复现报告的引用口径怎么写？ | Lead / 文档负责人 | `REPRODUCTION_PLAN` §2 已记录“无公开论文”，最终文档需统一 |
| Q9 | Space 同时加载 small+big（D8）在 zero-a10g 上是否 OOM？是否应懒加载 big？ | Space 维护者 | 未验证 |
| Q10 | 旧文档 Q5（`type:error` 一律停会话）、Q8（跳窗导致 WAV 时间轴省略 gap）仍未改；产品层是否接受？ | Lead / 产品口径 | 本次复核：代码未变，仍按“可接受但需声明”处理 |

---

## 8. 验证记录

> 环境：`/root/miniforge3/envs/falcon/bin/python` = 3.11.17、torch 2.8.0+cu128、Pillow/fastapi 已装；`gradio`/`spaces` **未装**；`node v22.23.3`。
> 所有命令在 `/root/code/github_repos/Falcon-fork` 执行；未改源码、未装包、未联网、未做 git 写操作。新增文件仅本文件与 `/tmp` 下一次性脚本/产物。

### 8.1 必跑项①：8 个 Node 测试（8/8 rc=0）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `node studio/tests/check_i18n.mjs` | rc=0；`8 localization checks passed (363 dictionary entries audited).` |
| 2 | `node studio/tests/check_mouth_tracking.mjs` | rc=0；1 条总 PASS（含几何/尺度/连续输出/重锁/多人脸/epoch/失败清理） |
| 3 | `node studio/tests/check_playback_resampler.mjs` | rc=0；6 条 PASS + `ALL PASS` |
| 4 | `node studio/tests/check_spectrum.mjs` | rc=0；`"test":"PASS"`；`fft_windows=9600`、`milliseconds≈688` |
| 5 | `node studio/tests/check_capture_dsp.cjs` | rc=0；`"test":"PASS"`；44.1/48 kHz 数字见 §3.1 |
| 6 | `node studio/tests/check_tracking_ui.cjs` | rc=0；`23 tracking UI control-flow checks passed.` |
| 7 | `node studio/web/capture-worklet.test.cjs` | rc=0；1 条总 PASS |
| 8 | `node studio/web/video-worker.test.cjs` | rc=0；1 条总 PASS |

合计 **8/8**，与 `docs/verification/runs/r4-review-08-node.log`（`rc=0` ×8、`node --check checked=21 fail=0`）一致；本次 `node --check` 覆盖 24 个文件（含 space）。

### 8.2 必跑项②：Space 组装 + i18n

| 命令 | 结果 |
|---|---|
| `PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python space/build.py --out /root/tmp/falcon-space-deepdive` | rc=0，`Assembled /root/tmp/falcon-space-deepdive`；54 文件 / 39 MB |
| `node space/tests/check_i18n.mjs /root/tmp/falcon-space-deepdive` | rc=0，`i18n: 226 Chinese strings translated` |

### 8.3 补充脚本（全部实际执行）

| 脚本（`/tmp`，未入库） | 用途 | 关键输出 |
|---|---|---|
| `/tmp/verify_frontend_protocol_v2.py` | 真实 PIL 走 `decode_request`/`OutputSlicer`/时间戳链 | `PROTOCOL-VERIFY-V2 PASS`（§3.3） |
| `/tmp/count_frontend.mjs` | 字典条数/重复键/index.html 绑定/DOM id/中文变量 | 363/351/12；125 绑定(120 唯一)/69 id；`app.js` 66 refs 全命中；`app.js` 中文 178(163 唯一) |
| `/tmp/space_i18n_breakdown.mjs` | 226 条来源分解 | app.js 136 + clip.py 12 + app.py 1 + html 110 → 226；space 字典 74、studio 字典 152、缺失 0 |
| `/tmp/i18n_gap_check2.mjs` | 反查 `falcon_runtime.py` 英文异常的 zh 映射 | 10 条 ZH、**5 条 RAW**（D3） |
| `python -m pytest space/tests/test_clip.py -q -m "not slow"` | clip.py ↔ Studio 等价性 | `20 passed, 2 deselected in 2.33s` |
| JPEG/载荷估算（PIL 实测） | 96×96 灰度 JPEG q=.88 尺寸与 60 s 载荷 | 439/1077/789/7110 B；60 s 约 8–19 MB |
| `node --check` ×24 | 语法 | `checked=24 fail=0` |

### 8.4 未执行 / 未能验证（如实记录）

| 项 | 原因 | 替代 |
|---|---|---|
| `space/app.py` 本地启动、`space/tests/test_space.py`（6 个测试项） | 需 `gradio`/`spaces`（且官方目标是 py3.12/torch2.13）；本环境按规定**不安装** | 静态阅读 + 复算 `test_space.py:74-79` 的 build 断言（§3.2） |
| `test_clip.py` 的 2 个 `@pytest.mark.slow`（真实权重 small/big 与 Studio 路径 `array_equal`） | 需加载真实权重（CPU 慢，且避免长跑） | 非 slow 的 mock 等价用例已 20 passed |
| `space/build.py --push`（HF 上传 + 请求 zero-a10g） | 需 HF token + 联网，违反约束 | 静态阅读 `build.py:48-58` |
| 真实浏览器 + 摄像头/麦克风/虚拟声卡/MediaPipe | 无 GUI/设备；`tests/browser/test.js` 需 Chrome | 读码 + Node 控制流测试 |
| `pytest studio/tests`（Python 侧 Studio 测试） | 属 studio-backend 范围，本次未跑 | 旧文档/AGENTS 记录 35 passed/2 skipped |
| HF 权重下载 | 本机不可联网 | 只检查 `weights_source` 的判定逻辑（D1） |

### 8.5 遵守的约束

未 `pip install`、未联网、未改仓库源码/他人文档、未执行任何 git 写操作（`git status` 仅新增 `docs/understanding/deepdive-20261024/07-studio-frontend-space.md`）。
所有临时脚本与组装产物在 `/tmp`、`/root/tmp/falcon-space-deepdive`。
