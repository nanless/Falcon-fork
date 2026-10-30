# 06 深挖：Studio 后端 / 运行时 / 窗口化实时推理协议

> 小组：项目理解 + 模型复现规划深挖（成员 `studio-backend`）。只读分析，只写本文件。
> 仓库 `/root/code/github_repos/Falcon-fork`，分支 `repro/phase2`，HEAD `2beec38`。
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128 / 2×A800，本任务期间 GPU 空闲）；
> 权重 `/root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}`（HF 缓存，未联网、未训练、未改源码）。
> 基线：旧文档 `docs/understanding/06-studio-backend.md`（1252 行，commit `7576854`）。
> 证据标记：`[代码]` 直接读源码；`[验证]` 本次实际运行（命令/输出见 §8）；`[推断]` 由源码/实测合理推断；`[文档]` 引用仓库已有文档/日志。

---

## 0. 一句话结论

**Studio 后端是一个只绑定 loopback、单进程、单模型、串行推理的「固定 2 s 窗口 + 每 0.5 s 全窗重算」演示服务，不是生产服务，也不是论文意义上的流式模型。**
本次用**真实发布权重**验证了：协议算术（32000/8000/1280/22720/9280/320）、strict 加载（small 525+137 keys、big 517+137 keys）、
Studio 副本与训练侧代码在 CPU 上**全链路 bit-identical（max|Δ|=0）**；实测 A800 单窗前向 **p50 47.1 ms（small）/ 56.5 ms（big）**、
本机 CPU **543 ms/窗**（> 500 ms hop，**CPU 无法实时**）。
同时必须把边界说清：P2-B 的 300 条配对显示**窗口化协议本身比整句评测低 −1.912/−1.730 dB（small）、−2.318/−2.133 dB（big）**，
README 的 **51.3 ms 是 3090 上单次 2 s 片段前向**、不是端到端流式延迟；因此 Studio 能证明「模型/权重/协议跑得通、且与训练代码等价」，
**不能**证明「论文流式质量/延迟已复现」，更不能直接对外部署（无鉴权、单推理锁、Origin/Host 只允许 loopback）。

---

## 1. 边界与关键路径

### 1.1 文件清单（HEAD 实测行数）

| 文件 | 行数 | 角色 |
|---|---|---|
| `studio/server.py` | 337 | FastAPI 应用：REST + WebSocket、`OutputSlicer` 时间轴切片、`Engine` 状态机、同源/主机安全边界、静态文件 |
| `studio/falcon_runtime.py` | 259 | 权重解析/下载、模型构建、strict 加载、设备选择、嘴部预处理、2 s 窗口前向；不依赖 FastAPI |
| `studio/infer_file.py` | 105 | 离线整段推理 CLI（WAV + mouth.npy → WAV），复用 `FalconRuntime` 的模型 |
| `studio/falcon_model/**` | 15 个 `.py` 共 4,035 行 + `PROVENANCE.md` 42 行 | 训练侧模型代码的推理拷贝（small：`Falcon.py`/`Falcon_Contrastive.py`；big：`SISNRRepel.py`/`Speaker_Contrastive_v2.py`） |
| `studio/configs/Falcon.yml` | 171 | small 训练配置（与仓库根 `configs/Falcon.yml` 一致） |
| `studio/configs/FalconBig.yml` | 166 | big 训练运行快照（`audionet_name: SpeakerContrastiveV2`） |
| `studio/scripts/check_environment.py` | 54 | 按 `studio/requirements.txt` 校验版本与 import（**本环境实测不通过**，见 §3.5） |
| `studio/tests/*.py` | 5 个（`test_server.py`/`test_weights.py`/`test_device.py`/`test_infer_file.py`/`serve_browser.py`） | 协议、权重解析、设备、离线推理、浏览器 QA 服务 |

**自基线以来后端零改动**：`git log 7576854..HEAD -- studio/{server.py,falcon_runtime.py,infer_file.py,configs,tests,falcon_model,scripts/check_environment.py}` 为空；
唯一相关提交 `607352c` 只改了前端 `app.js`（新增 `bounded()`/`window.falconDiagnostics()`/`checkCaptureHealth()`）与 `i18n-dynamic.js`、前端测试、`studio/README*.md` 说明。
因此旧文档的后端行号/结论在本 HEAD 仍适用，本文只对**验证环境、真实权重结论、前端诊断**做增量。

### 1.2 进程 / 线程 / 依赖方向

- 进程：单 `uvicorn` worker（`studio/server.py:332-333`），全局唯一 `Engine`（`server.py:182-195,287`）。
- 线程：`Engine.start_load()` 起 1 个 daemon 加载线程（`server.py:145-146`）；推理在 WS 协程里经
  `asyncio.to_thread(engine.infer, ...)` 丢到默认线程池（`server.py:260`）。
- 并发控制：全局非阻塞 `infer_lock`，整个服务同一时刻只有一次前向（`server.py:102,163-172`）；
  `load_lock` 保护加载/切变体（`server.py:101,130-146`）；`active_connections` 在 WS 连接期间 +1/−1（`server.py:239,276`）。
- 依赖方向：`web/*`（浏览器）→ `server.py` → `falcon_runtime.py` → `falcon_model/*`（只依赖 torch/numpy/yaml）；
  `infer_file.py` → `falcon_runtime.py`。`server.py` 不 import 模型类（`server.py:106,121,150` 的函数内延迟 import），
  因此 `tests/test_server.py` 可用 `StubEngine` 在无权重时测协议（`studio/tests/test_server.py:24-32`）。
- 不落盘：服务端只处理内存中的窗口；录音/导出在浏览器内存（`server.py:1`）。

### 1.3 三条关键链（[代码]）

**A. 启动 / 权重加载**（`server.py:126-158` + `falcon_runtime.py:133-200`）：

```
main() 选端口 7860-7899（server.py:304-316）→ create_app(Engine(device, variant))（:332）
  lifespan → engine.start_load()（:185-192）→ 加载线程 Engine._load()（:148-158）
    → FalconRuntime(variant).load(device)
       resolve_checkpoint()（falcon_runtime.py:142）→ 本地 / hf:// / HF repo 下载
       yaml.safe_load(config)（:146-148）→ deepcopy audionet/videonet（:149-150）→ videonet.pretrain=None（:151）
       架构名/视频架构校验（:152-156）→ 构建 audio+video（:157-158）
       torch.load(weights_only=True)（:160）→ 按 audio_model./video_model. 拆前缀（:162-163）
       strict=True 加载两个子模型（:168-169）→ 统计 keys/参数（:170-171）
       torch.set_num_threads(夹取 [1,8]，默认 4)（:175-176）→ eval()/requires_grad_(False)（:177-180）
       pick_device()（:181）→ _warmup() 真前向（:185-195,202-205）→ ready=True
```

**B. 实时请求**（`studio/web/app.js:618-648` → `server.py:233-276`）：

```
前端 maybeInfer()：samples-sessionEpoch ≥ HOP+GUARD(9280)（app.js:621）
  计算 end、首窗强制 end=sessionEpoch+HOP+GUARD（:622-624）→ 组 2 s 音频（:631-634）
  WS 发 {type:"infer", id, audio:b64, frames:[b64×50], window_seconds:2, hop_seconds:.5,
         end_sample:end+22720, reset}（:647）
服务端：同源校验（server.py:235-237）→ receive_text → JSON/id 校验（:249-256）
  → decode_request（:257）→ busy 检查（:258-259）→ engine.infer（:260）
  → slicer.take（:261）→ send_json result（:262-267）；异常回 error 不关连接（:268-272）
前端 handleResult：校验 id/sample_rate/长度/时间戳（app.js:715-741）→ 减 offset 回真实时间轴 → 播放/录音
```

**C. 离线整段**（`infer_file.py:85-101`）：`read_wav`（:28-38）→ `read_mouth`（:47-59）→ `FalconRuntime.load`（:98）
→ `extract` 一次前向跑完整段（:69-82）→ `write_wav`（:41-44,100）。注意这是**整段前向**，与 Studio 的 2 s 滑窗不等价（§4.2-R8）。

### 1.4 本文不覆盖

浏览器采集/追踪/回放/会议输出（`studio/web/**`，含 `bounded()`/`falconDiagnostics()` 的完整前端语义）见同目录 `07-studio-frontend-space.md`；
`space/**`（ZeroGPU 变体）同见 07。本文只在前端与后端契约相关处引用 `app.js` 行号。

---

## 2. 协议参数表

### 2.1 常量与算术（全部 [代码]，算术 [验证]）

| 量 | 值 | 换算 | 证据 |
|---|---|---|---|
| 采样率 `SR` | 16000 Hz | — | `server.py:28`；`falcon_runtime.py:24` |
| 视频帧率 `FPS` | 25 fps | — | `server.py:28`；`falcon_runtime.py:25` |
| 窗口 `WINDOW` | 32000 采样 | **2.000 s** | `server.py:28`；`falcon_runtime.py:26` |
| 窗口帧数 `WINDOW_FRAMES` | 50 帧 | 2 s × 25 fps | `falcon_runtime.py:27`；`server.py:51-52` |
| hop `HOP` | 8000 采样 | **0.500 s** | `server.py:28` |
| 右上下文 `GUARD` | 1280 采样 | **80 ms** | `server.py:29` |
| 交叉淡化 `FADE` | 320 采样 | **20 ms** | `server.py:29` |
| 左上下文 `LEFT_CONTEXT` | 22720 采样 | **1.420 s** = 32000−8000−1280 | `app.js:10`；`server.py:80` |
| 首窗所需真实输入 | 9280 采样 | **0.580 s** = hop+guard | `app.js:621,624`；推导见 §2.2 |
| 每块输出 | 8000 采样 | 0.500 s | `server.py:80-81,262-263` |
| 单请求音频 b64 | 170668 字符 | 32000×4 B → base64 | `server.py:43`；[验证] §3.3 |
| 单结果音频 b64 | 42668 字符 | 8000×4 B → base64 | `server.py:263`；[验证] §3.3 |
| 单条 WS 消息上限 | 2,000,000 字符 | `MAX_MESSAGE_BYTES`，同时传给 uvicorn `ws_max_size` | `server.py:30,244,333` |
| 时间戳上限 | 2^53−1 | JS 精确整数上限 | `server.py:31,65,255` |
| 嘴部图 | 50 × 96×96 JPEG/PNG | 服务端 `convert("L")`；模型输入裁到 88×88 | `server.py:51-63`；`falcon_runtime.py:95-104` |
| 输出时间保护 | 80 ms | 结果字段 `output_delay_ms` | `server.py:111,267` |

### 2.2 时间轴：请求/响应/切片（[代码] + [验证]）

两套坐标（与旧文档一致，本文复验）：

- **绝对采样坐标 `s`**：浏览器打开设备后的累计采样（`app.js:15,300-315`），用于录音/回放对齐；
- **协议坐标 `P`**：请求/响应的 `end_sample`/`start_sample` 所在坐标，`P = s + LEFT_CONTEXT = s + 22720`
  （`app.js:643,647`；服务端只校验 `end_sample ≥ WINDOW`，不感知 22720，`server.py:64-67`）。

```
协议窗口：[P-32000, P)                       （P = 请求 end_sample）
输出切片：pred[WINDOW-GUARD-HOP : WINDOW-GUARD] = pred[22720:30720]
        = 绝对 [P-22720-9280, P-22720-1280)   （server.py:80-81,266-267）
future  ：pred[30720 : 30720+320]            （下一窗交叉淡化用，server.py:86）
右保护带：[P-1280, P) 丢弃（80 ms）
```

相邻窗连续（未 reset 且 `endpoint−last==8000`）时：`chunk[:320] = future*(1−ramp) + chunk[:320]*ramp`，
`ramp=linspace(0,1,320)`（`server.py:82-86`）；不连续（首窗/reset/跳窗）不淡化，由前端 5 ms 淡入兜底（`app.js:811-816`）。

**首窗为什么是 9280/0.58 s**（[代码] + [验证]）：

1. 前端等 `samples - sessionEpoch ≥ HOP + GUARD = 9280`（`app.js:621`）；
2. 首窗强制 `end = sessionEpoch + HOP + GUARD`（`app.js:624`），即**真实输入恰好 9280 采样**；
3. 请求 `end_sample = end + 22720`（`app.js:647`），窗口覆盖协议 `[end+22720-32000, end+22720)`
   = 真实 `[sessionEpoch-22720, sessionEpoch+9280)`，前 22720 采样补零、嘴部用首帧填充（`app.js:631-634`；`frameWindow` 合成左上下文，`app.js:578-600`）；
4. 输出绝对区间 = `[end-9280, end-1280)` = **`[sessionEpoch, sessionEpoch+8000)`**，正好从“开始分离”的第一个真实样本起，不丢开头；
5. 因此首块需要 9280 真实采样（其中最后 1280 只作为右上下文，不进输出），之后每 8000 采样出一个块。

数值例（[验证] §3.3）：

| `sessionEpoch` | 首窗请求 `end_sample` | 首块协议 start/end | 首块绝对区间 |
|---|---|---|---|
| 0 | 32000 | 22720 / 30720 | [0, 8000) |
| 160000（预览 10 s 后开始） | 192000 | 182720 / 190720 | [160000, 168000) |

**稳态输出年龄**（[推断]，由 `app.js:621-624` 公式）：发送时采集头 `samples = end + r`，`r ∈ [0, HOP)`；
输出块末端真实时间 = `end − GUARD`。所以**算法侧输出滞后 = GUARD + r ∈ [80, 580) ms（均值 ≈330 ms）**；
再加推理（实测 GPU 47–57 ms）与播放缓冲，对应 `studio/README.md:99` 的“设计延迟约 0.7–0.8 s”。

### 2.3 REST 接口（[代码] + [验证] §3.3/§8）

| 端点 | 请求 | 响应/语义 |
|---|---|---|
| `GET /api/status` | 无 | 未加载：`ready/loading/error/device/model/variant/variants/sample_rate/video_fps/window_seconds/hop_seconds/output_delay_ms/busy/mode="sliding-window"/app="falcon-studio"`（`server.py:105-112`）。加载后合并 `FalconRuntime.status()`：`architecture/parameters/audio_weight_keys/video_weight_keys/warmup_ms/last_inference_ms/warning/mode="windowed_audio_visual"`（`server.py:113-116`；`falcon_runtime.py:215-235`） |
| `POST /api/load` | body ≤ **1024 B** 的 JSON 对象；`{"variant":"small"\|"big"}`（缺省沿用当前变体） | 200 返回 status 快照（加载可能仍在进行，`loading=true`）；400=未知变体/非对象/坏 JSON/**变体不可用**（`weights_source()` 为 None，即缺 `huggingface_hub`）；409=加载中切别的变体/有活跃 WS 连接；413=body 过大（`server.py:217-231,126-146`） |
| `GET /` | 无 | `web/index.html`（`server.py:278-280`） |
| `/static`、`/` 静态挂载 | 无 | `web/` 目录（`server.py:282-283`） |
| `WS /ws` | 见 §2.4 | 见 §2.4 |

安全边界（[代码]）：`--host` 只允许 `127.0.0.1|localhost`（`server.py:292`）；`TrustedHostMiddleware` 白名单
`127.0.0.1/localhost/[::1]/testserver`（`server.py:196`）；HTTP 中间件 `same_origin` 校验（`server.py:175-179,198-202`）；
响应头 `nosniff/no-referrer/Permissions-Policy(camera,microphone=self)/no-store/CSP`（`server.py:204-210`）。
**没有鉴权、没有限流**（见 §4.3）。

### 2.4 WebSocket 帧协议（[代码] + [验证]）

**传输形态**：全部是 **text JSON**，没有二进制帧；音频与图像各自 base64。
- 音频：`<f4` little-endian、**mono 16 kHz**、恰好 32000 采样 = 128000 B → b64 170668 字符（`server.py:42-50`；[验证] §3.3）；
- 图像：**50 张** 96×96 JPEG/PNG 的 base64，每张 ≤40000 字符；服务端用 Pillow 打开、校验尺寸/格式，`convert("L")` 成 uint8（`server.py:51-63`）；
  前端采集是 96×96 灰度/彩色裁片（`video-worker.js:12,89-90`；手动模式灰度为 `0.299R+0.587G+0.114B`，`app.js:570-574`）；
- 采样率：浏览器以设备采样率采集，AudioWorklet 用 63-tap 多相 FIR 重采样到 16 kHz（`capture-worklet.js:1-4,8-9,16-28`），
  每包 320 样本（20 ms）并带 `startTime`（`:57-61`）；服务端只接受 16 kHz；
- 通道：worklet 对多声道取平均（`capture-worklet.js:45-48`）→ **mono**；前端 8 s 环形缓冲（`app.js:15`）供 2 s 窗口取数；
- 单连接内**请求-响应串行**：`while True` 先 `receive_text` → 处理 → `send_json` 才读下一条（`server.py:242-267`）；
- 每连接独立 `OutputSlicer`（`server.py:240`），因此新会话从非连续开始、不与旧会话交叠。

**请求字段**（`server.py:35-67`）：

| 字段 | 必填 | 类型 | 校验 |
|---|---|---|---|
| `type` | ✓ | str | 必须 `"infer"` |
| `id` | ✓ | str/int | 非 bool；`len(str(id))≤80`；int 时 `\|id\|≤2^53−1` |
| `audio` | ✓ | str | base64 长度恰为 `4*((32000*4+2)//3)=170668`；解码 `<f4`；shape `(32000,)`；全 finite；`max\|amp\|≤4` |
| `frames` | ✓ | list[str] | 恰 50 个；每个 ≤40000 字符；96×96 JPEG/PNG |
| `window_seconds` | – | number | 缺省 2；出现时必须 `==2` |
| `hop_seconds` | – | number | 缺省 0.5；出现时必须 `==0.5` |
| `end_sample` | – | int | 缺省 32000；bool/float 拒绝；`32000 ≤ end_sample ≤ 2^53−1` |
| `reset` | – | bool | 缺省 false；true = 新会话，不做跨窗淡化 |

**结果字段**（`server.py:262-267`；[验证] 字段/类型见 §3.3/§8）：

| 字段 | 含义 |
|---|---|
| `type:"result"`、`id` | 回显 |
| `audio` | base64(`<f4` × 8000) = 42668 字符，0.5 s 输出块 |
| `sample_rate` | 16000 |
| `start_sample` / `end_sample` | `endpoint−9280` / `endpoint−1280`（**响应 end_sample 与请求 end_sample 同名不同义**） |
| `inference_ms` | 含 `preprocess_mouth` 的模型前向耗时（不含协议/图片解码） |
| `rtf` / `window_rtf` | `inference_ms/500` / `inference_ms/2000` |
| `device` | 实际设备（取自 `engine.status()`） |
| `output_delay_ms` | 80（右上下文） |
| `contiguous` | 是否与上一块连续（未 reset 且 `endpoint−last==8000`） |
| `clipped_samples` | 本块 `\|x\|>1` 被裁剪的样本数（前端未消费） |

**错误/关闭**：`{"type":"error","id","message"}` 覆盖 ValueError/RuntimeError，**不关闭连接**，后续消息可继续（`server.py:268-272`；[验证]）；
非同源 WS 在 accept 前 `close(1008)`（`server.py:235-237`）；单条消息 >2,000,000 字符 `close(1009)`（`server.py:244-246`）；
uvicorn 另设 `ws_max_size=2_000_000, ws_max_queue=1`（`server.py:333`）。

### 2.5 为什么是“固定 2 s 窗口、每 0.5 s 全窗重算”，而不是流式状态

**直接证据（作者自述）**：`falcon_runtime.py:1-6` 明说 “windowed inference, not a stateful streaming architecture”；
`PROVENANCE.md:41-42` “model exposes no streaming state … not sample-by-sample streaming”；`studio/README.md:99`
“Every hop recomputes the full 2 s window, because the model exposes no streaming state”。

**代码层根因（[代码]）**：

1. **无跨窗状态接口**：`FalconRuntime._forward` 只做 `preprocess_mouth → video_model → audio_model → separated[0,0]`
   （`falcon_runtime.py:237-246`），不传/不返回任何隐状态；模型 `forward(wav, video_embedding)` 没有 KV cache/RNN 状态入参。
2. **时间注意力是窗口内因果，但没有跨窗缓存**：`AudioBlock` 实际用的是 `MultiHeadSelfAttention2D`
   （`studio/falcon_model/models/Falcon.py:417,494-530`；由 `:788-789` 构建），`causal=True` 时用 `torch.tril`
   对 T 维做因果 mask（`:520-525`）；Q/K/V 全部在当前窗口内计算，没有跨 hop 的 KV cache。
3. **归一化统计覆盖整窗**：配置用 `cLNhw`（`Falcon.yml:21,43`；`FalconBig.yml:8,40`），实现是
   `LayerNormalizationHW = GlobalLayerNorm + CumulateLN4D`（`studio/falcon_model/layers/normalizations.py:204-229`）；
   `GlobalLayerNorm` 是 `nn.GroupNorm(num_groups=1, num_channels=C)`（`:134-143`），作用在 `(B,C,T,F)` 上时
   **统计量覆盖 C×T×F（含窗口尾部/未来帧）**，再叠加沿 T 的累积归一化（`:177-201`）。因此输出不是逐样本可增量计算的。
4. **训练/推理口径**：模型按 2 s 片段训练（`Falcon.yml:162 segment: 2.0`），推理保持同样的 2 s 窗口 + 全窗统计，
   是“不改模型、原样 strict 加载发布权重”的唯一低成本路径。

**实测佐证（[验证]，合成输入，见 §3.3）**：

- 把窗口尾部 1280 采样（`30720:32000`，位于 guard 内）放大 10 倍，**已发出的输出块 `pred[22720:30720]` 仍整体变化**
  （8000 个样本中 7988 个变化 >1e-6；块内 max|Δ|=0.0172，而该输入下输出 rms≈0.0003）→ 输出依赖窗口内“未来”样本，不能逐样本流式；
- 同一段绝对采样从“窗口末端”与“下一窗起点”两个位置预测，重叠区（1280 采样）相对差 `rms(d)/rms(signal)≈0.98`（合成随机输入）
  → 预测随窗口位置漂移，因此必须重算 + 用 `GUARD`/`FADE` 平滑拼接，而不是复用旧状态。

**权衡表（[推断] + 实测）**：

| 维度 | 固定 2 s 全窗重算（现状） | 状态化流式（需改模型/重训） |
|---|---|---|
| 计算 | 每 0.5 s 算 2 s → **4× 放大**；A800 实测 47 ms/窗 → 每输出秒 0.094 RTF；CPU 543 ms/窗 → 1.09 RTF | 理论 ~1×，但要 KV cache/增量归一化 |
| 延迟 | 模型不是瓶颈；端到端 ≈0.7–0.8 s（hop+guard+推理+播放缓冲） | 理论更低，需重新验证 |
| 内存 | 2 s 音频仅 128 KB；实测峰值显存 383 MiB（small）/568 MiB（big）；全窗注意力 `T'=251` 很小 | 需缓存历史状态，长会话内存增长 |
| 一致性 | 与训练 2 s 片段一致；每窗独立、无状态漂移；checkpoint 原样加载 | 训练/推理不匹配风险，归一化统计跨窗漂移 |
| 质量 | 已实测比整句低 −1.9~−2.3 dB（P2-B） | 未知，需重训/实验 |

结论：**内存不是主因，延迟有余量但 CPU 不够；核心是“一致性/工程成本”——不改模型、原样加载发布权重，
用 4× 算力和 80 ms 保护带 + 20 ms 交叉淡化换取与训练口径一致的输出。**

### 2.6 客户端超时与卡顿诊断（后端契约的运行时配套；[代码]，与 07 交叉）

`607352c` 新增的前端诊断不属于后端，但直接决定“后端慢/卡”时用户看到什么：

| 机制 | 位置 | 行为 |
|---|---|---|
| `bounded(promise, ms, message)` | `app.js:69-75` | 给设备/浏览器 Promise 加超时；超时抛中文错误 |
| 设备启动上限 | `app.js:182,245,248,266,269,469` | 枚举 8 s、`getUserMedia` 45 s、AudioContext resume 8 s、摄像头 play 12 s、AudioWorklet 15 s、视频采集初始化 5 s |
| 模型等待 | `app.js:165,666-670` | `/api/load` fetch 180 s；ready 轮询 1.5 s/次、总 180 s；WS 连接 10 s（`:678`） |
| 单次推理 | `app.js:643-645` | 前端 60 s 定时器，超时停采集并提示“检查模型服务和计算设备” |
| `window.falconDiagnostics()` | `app.js:40-52` | 只暴露聚合量：running/opening/audioState/videoMode/tracking/samples/frames/sentWindows/recordingSeconds/pendingMs/inputRms/outputRms/各 ageMs/warning；**不含 PCM、图像、cookie、设备标识** |
| `checkCaptureHealth()` | `app.js:53-67` | 每秒检查：输入 >3 s 无更新、画面 >3 s 无更新、inFlight >5 s、结果 >5 s、输入 RMS <1e-5、输出 RMS <1e-6；写 `performanceHint` 并 `console.warn` 一次 |
| 跳窗提示 | `app.js:636-641` | 计算 `missed`，提示“已跳过 N 个旧窗口” |
| 跟不上提示 | `app.js:772` | `rtf > 1` → “推理慢于采集速度，将自动跳过积压窗口”；否则“跟得上实时采集 · 输出保护区 80 ms” |

服务端本身**没有请求级超时**（`asyncio.to_thread` 未包 `wait_for`，`server.py:260`），也没有加载超时；
只能靠前端 60 s / 180 s 兜底。`/api/status` 的 `busy/last_inference_ms/warmup_ms/warning` 是诊断字段，
前端只消费其中一部分（见 07）。

---

## 3. 关键实测

### 3.1 仓库自带测试（必跑）

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest studio/tests -q -rs
```

结果（[验证]，2026-10-24）：

```
....ss...............................                                    [100%]
SKIPPED [1] studio/tests/test_device.py:35: needs a machine without CUDA and the cached JusperLee/Falcon weights
SKIPPED [1] studio/tests/test_device.py:44: checks the error without a GPU
35 passed, 2 skipped in 47.80s   （热缓存复跑 18.92s）
```

- `37 collected = 35 passed + 2 skipped`；2 个 skip 是 **GPU 条件用例**（本机有 CUDA，故跳过“GPU 失败回退 CPU”与“无 GPU 报错”两条）。
- `test_infer_file.py` 的 2 项本次**真跑**（权重已缓存）：2 s 片段 `infer_file.extract` 与 `runtime.infer` `rtol=1e-5` 一致；
  3.7 s 任意长度输出 `(92×640,)` 且有限——旧文档的“无权重 skip”已不成立。
- `test_server.py` 覆盖 wire 往返/时间戳、11 类非法 payload、`OutputSlicer` 交叉淡化与跳窗、同源拒绝、错误不杀会话、
  bool/超精度 id、变体切换守卫、CSP/wasm 静态资源。

### 3.2 真实权重：strict 加载 + 与训练侧等价（必跑）

**(a) R3 脚本**（`docs/verification/scripts/verify_r3_studio_weights.py`，CPU，<1 min）：

```bash
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/miniforge3/envs/falcon/bin/python docs/verification/scripts/verify_r3_studio_weights.py
```

关键输出（[验证]）：

```
ckpt keys: ['state_dict','architecture','epoch'] | architecture: FalconSpeakerContrastive | epoch: 714
audio_model.* keys=525  video_model.* keys=137  total=662
studio strict=True load: <All keys matched successfully>
studio params: 537,697
training strict=True load: <All keys matched successfully>
train vs studio 输出 max|Δ| = 0.000e+00 -> bit-identical: True
state_dict 两副本 key 完全一致: True
```

方法：Studio `FalconSpeakerContrastive`（`studio/falcon_model/models/Falcon_Contrastive.py:60`）与训练侧
`look2hear.models.FalconSpeakerContrastive` 用各自配置构建、加载同一份 `audio_model.*`、喂**同一随机音频 + 同一 512×50
嘴部 embedding**，CPU eval 前向比较。它证明 **audio 模型代码路径逐位一致**；视频侧由 state_dict key/shape 全等
（137/137）+ `frcnn_videomodel.py` 仅删 3 行 import/路径 hack 佐证。

**(b) 全链路真实权重复验（本次新增，CPU，~1 min）**：直接用真实 `falcon.pt` 同时跑
`FalconRuntime` 全链路与“训练侧 `FalconSpeakerContrastive` + 训练侧 `FRCNNVideoModel`”全链路，
输入同一 `audio(32000,) f32` 与 `frames(50,96,96) uint8`：

```bash
PYTHONPATH=$PWD/studio:$PWD /root/miniforge3/envs/falcon/bin/python - <<'PY'
# 构建 rt=FalconRuntime("small").load("cpu")；训练侧类从 configs/Falcon.yml 构建并 strict 加载 ckpt 的
# audio_model.* / video_model.*；同一 preprocess_mouth 后分别前向，比较 video embedding 与最终输出。
PY
```

结果（[验证]）：

```
video embedding max|Δ| = 0.0
FULL pipeline real falcon.pt CPU: max|studio-training| = 0.0 | bit-identical = True
params studio runtime = 11,724,385 | keys = 525 + 137
```

即 **PROVENANCE 的“CPU bit-identical（falcon.pt，max abs diff 0）”本次用真实发布权重在完整链路（含视频分支）复现**；
旧文档的 Q1“真实权重未验证”关闭。注意：这是 **同一 2 s 窗口**的代码等价，不是“窗口化协议 = 整句协议”（后者见 §6.1/P2-B）。

**(c) 两份权重与 strict 加载**（[验证]）：

| 项 | `falcon.pt`（small） | `falcon-big.pt`（big） |
|---|---|---|
| 文件大小 | 47,156,343 B | 49,645,087 B |
| 顶层 keys | `state_dict/architecture/epoch` | 同 |
| `architecture` | `FalconSpeakerContrastive` | `SpeakerContrastiveV2` |
| `epoch` | 714 | 715 |
| audio/video keys | 525 / 137 | **517** / 137 |
| 模型参数量 | audio 537,697 + video 11,186,688 = **11,724,385** | audio 1,158,617 + video 11,186,688 = **12,345,305** |
| strict 加载 | `<All keys matched successfully>`（audio/video） | 同（Studio 侧；训练侧无该类） |
| 差异 | — | big 少 8 个 `separator.{process_f,process_t}_layers.*.out_proj.weight`（`DualPathRNN.out_proj` 退化为 Identity，`studio/falcon_model/models/Falcon.py:379-383`）；audio 360 个公共 key 形状不同；**video 60/137 张量不同**（两次训练 run，见 P2-B §2.1） |

`studio/configs/FalconBig.yml:88` 的 `audionet_name: SpeakerContrastiveV2` 与 ckpt 元数据一致；
`falcon_runtime.py:152-156` 会拒绝配置/变体不匹配。

### 3.3 协议数值与窗口切片（本次新增，<10 s）

```bash
PYTHONPATH=$PWD/studio CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/falcon/bin/python - <<'PY'
# 打印常量、b64 长度、首窗 epoch=0/160000、OutputSlicer 三窗、窗口位置差、future 扰动实验
PY
```

关键输出（[验证]）：

```
constants window=32000 hop=8000 guard=1280 fade=320 left=22720
audio b64 chars: 170668 | output b64 chars: 42668
epoch=0: first real end(abs)=9280 req.end_sample=32000 -> abs out=[0,8000) real samples=9280 (0.580s = hop+guard)
epoch=160000: first real end(abs)=169280 req.end_sample=192000 -> abs out=[160000,168000) real samples=9280
window start(abs)=0 endpoint=32000 slice=[22720,30720) contiguous=False n=8000 clipped=0
window start(abs)=8000 endpoint=40000 slice=[30720,38720) contiguous=True n=8000 clipped=0
window start(abs)=16000 endpoint=48000 slice=[38720,46720) contiguous=True n=8000 clipped=0
stitched 3 chunks vs absolute timeline allclose: True max|d|=2.98e-08
```

`tests/test_server.py:74-83` 与 `verify_studio_backend.py` B1 另有相同结论（旧文档记 4.3e-8，本次 2.98e-8，均为 float32 精度量级）。
窗口位置/未来扰动实验（合成随机输入，输出幅度很小，故只看相对量）：

```
same abs samples [30720,32000): n=1280 max|d|=0.0043 rms(d)=0.0012 rms(sig)≈0.0012 -> rms(d)/rms(sig)≈0.98  (随机帧)
scale window tail 30720:32000 x10 -> emitted chunk 7988/8000 样本变化，max|d|=0.0172，rms(out)≈0.0003
```

说明：`GUARD` 丢掉的只是“最不可靠的末端 80 ms”，**不能消除全窗统计带来的未来依赖**；这正是必须重算 + 淡化拼接的原因（§2.5）。

### 3.4 CPU / GPU 单窗延迟（真实权重，A800）

命令：`FalconRuntime(variant).load(device)` 后先额外 warm 一次，再各计时（含 `preprocess_mouth`、H2D/D2H、`.cpu().numpy()`）：

| variant / device | load | `warmup_ms` | n | p50 | p90 | min / max | `rtf=p50/500` | 峰值显存 |
|---|---|---|---|---|---|---|---|---|
| small / cuda | 15.7 s | 2057.2 | 30 | **47.1 ms** | 48.2 | 46.9 / 48.5 | **0.094** | 383 MiB |
| big / cuda | 10.5 s | 131.2 | 30 | **56.5 ms** | 57.1 | 56.3 / 64.0 | **0.113** | 568 MiB |
| small / cpu | 11.4 s | 1131.9 | 5 | **543.3 ms** | 634.5 | 533.2 / 634.5 | **1.087** | — |

解读（[验证]+[推断]）：

- **GPU 有 10× 余量**：47 ms 对 500 ms hop → 每输出秒计算 RTF 0.094（2 s 窗/0.5 s hop = 4× 放大后仍远低于 1）。
- **CPU 不够**：543 ms > 500 ms → `rtf≈1.09`，前端会跳窗/提示；这与 `studio/README.md:99`“CPU may skip windows”一致。
- `load` 含 `torch.load`（47 MB）、构建、warmup，不含下载；首次进程内 CUDA 初始化会显著抬高 warmup（2057 ms），
  同进程第二次加载（big）warmup 仅 131 ms；首次 10 次计时未额外 warm 时 small 中位数约 74 ms，多 warm 一次后稳定在 ~47 ms（时钟爬升）。
- **A800 ≠ 3090**，本表只用于“数量级/是否实时”判断，不可直接与论文 51.3 ms 比较（§6.2）。

### 3.5 `check_environment.py` 在本环境不通过（[验证]）

```bash
/root/miniforge3/envs/falcon/bin/python studio/scripts/check_environment.py
# exit=1
torch: 已安装 2.8.0+cu128，应用需要 >=2.14.1,<3
numpy: 已安装 2.2.6，应用需要 >=2.4.6,<3
Pillow: 已安装 13.0.0，应用需要 >=12.3.0,<13
```

原因：`studio/requirements.txt:3-9` 写的是**远高于本项目锁定版本的版本下限**（torch≥2.14.1、numpy≥2.4.6、Pillow<13），
与本项目强制的 `torch==2.8.0+cu128 / numpy==2.2.6`（`AGENTS.md` §2）冲突。结论：**该脚本不能作为本仓库的环境门禁**；
Studio 实际运行以 conda `falcon` 为准，测试/权重验证均已通过（§3.1–3.2）。另：`einops` 只在 `check_environment.py:40`
的 import 列表出现，`studio/**/*.py` 运行期无 import（死依赖）。

### 3.6 主验证脚本（额外跑，作为交叉证据）

`PYTHONPATH=$PWD/docs/verification/stubs:$PWD python docs/verification/scripts/verify_studio_backend.py` 本次完整通过（<1 min，合成权重 + 协议层）：

- B0：常量 `16000/25/32000/8000/1280/320/2_000_000/9007199254740991`；`start=22720 end=30720`；audio b64=170668；14 类非法请求全 ValueError；
- B1：首块 8000、`contiguous=False`；第二块 `contiguous=True`；拼接在 float32 精度内连续（旧日志 4.28e-8，本次 inline 2.98e-8）；跳窗 reset；clip 计数；
- B5：result 13 字段类型全表；`rtf=0.05/window_rtf=0.013`（stub 25 ms）；时间戳 22720/30720、30720/38720；错误不杀会话；
- B6：PNG 帧接受、`|amp|=4.0` 接受/4.0001 拒绝、消息 2,000,000 字符内得 error/超限 close 1009、`/api/load` 400/409/413；
- B7：`FalconRuntime.status()` 字段类型全表（`parameters=11,724,385`、`audio_weight_keys=525`、`video_weight_keys=137`）。

---

## 4. 缺陷与风险

### 4.1 P1（影响“能不能实时/能不能演示好”）

| 编号 | 问题 | 证据/影响 |
|---|---|---|
| R1 | **CPU 无法实时**：543 ms/窗 > 500 ms hop → 跳窗、录音缺口、提示“推理慢于采集速度” | §3.4；`app.js:636-641,772`；`studio/README.md:99` |
| R2 | **窗口化协议本身掉点**：300 条配对 small −1.912/−1.730 dB、big −2.318/−2.133 dB（SI-SNRi/SDRi） | `docs/verification/phase2/P2B-report.md` §1、§3.1；P2-B 模拟器按 Studio `OutputSlicer` 语义复刻 |
| R3 | **首窗质量先天较弱**：只有 0.58 s 真实输入 + 1.42 s 补零/首帧；README 自认“initial separation quality can differ” | `app.js:621-635`；`studio/README.md:99` |
| R4 | **并发/多标签页不友好**：全服务单 `infer_lock`；第二个连接收到“模型忙/另一个窗口”错误，前端把任何 `type:"error"` 当致命并停整场会话 | `server.py:163-172,258-259`；`app.js:728,780-783`（07 域已交叉确认） |
| R5 | **无请求级超时/取消**：服务端不设 `wait_for`，前向 hang 只能等前端 60 s | `server.py:260`；`app.js:643-645` |
| R6 | **加载无超时 + 首次需联网下载**：`resolve_checkpoint` 走 HF；`models/` 本地目录并不存在（错误文案误导）；下载/网络慢时只能等 | `falcon_runtime.py:58-78,142-144`；`server.py:140-141`；README 的 mirror 说明 |
| R7 | **对外部署被代码硬限制**：loopback + same_origin + TrustedHost 白名单，反向代理/公网直接 403/400；无鉴权/限流/TLS | `server.py:175-179,196,292`；README 在线 demo 必然是另一份部署补丁（[文档] `studio/README.md:16`） |
| R8 | **`infer_file.py` 整段前向 ≠ Studio 滑窗**：docstring 说 “matching the evaluation protocol”（评测确实是整句），但若拿它当“实时输出”会高估约 2 dB | `infer_file.py:1-13,69-82`；P2-B |

### 4.2 P2（工程/维护性）

| 编号 | 问题 | 证据 |
|---|---|---|
| R9 | 每个 result 都调 `engine.status()` → 重建 `variants()` → 每个 variant 两次 `weights_source()` 文件系统查询（0.5 s 热路径） | `server.py:265,118-124`；`falcon_runtime.py:48-55` |
| R10 | `status()` 可能出现 `error` 与 `ready=True` 并存（手动注入 error 时）；正常路径不可达但字段语义不互斥 | `server.py:113-116`（`ready` 只看 runtime）+ 旧文档 §5.6 |
| R11 | `Engine.infer` 的 finally 先 `busy=False` 再 `infer_lock.release()` → 存在“busy 已 false、锁未放”的竞态窗口，错误文案两种 | `server.py:163-172` |
| R12 | 依赖表不可用：`requirements.txt` 版本下限不存在；`requirements-lock.txt` 与项目环境不符；`check_environment.py` exit 1；`einops` 死依赖 | §3.5；`studio/requirements.txt:3-9` |
| R13 | `read_wav` 的 32-bit float 路径不校验范围/不 clip，与 WS 路径 `max|amp|≤4` 不对称 | `infer_file.py:34-38`；`server.py:49` |
| R14 | 无 metrics/健康检查/结构化日志；只有 `/api/status` 与 uvicorn 日志；`clipped_samples/window_rtf/warning` 前端未消费 | `server.py:105-116,262-267` |
| R15 | 模型切换必须断开所有 WS；`/api/load` 不能指定设备/本地 checkpoint；设备只能启动时 `--device`/`FALCON_DEVICE` | `server.py:137-146,183,294` |
| R16 | 模块级 `app = create_app()`（`autoload=True`）与 `main()` 再建 Engine → `uvicorn server:app` 会在 lifespan 自动加载；import 本身不加载（旧文档“不 autoload”表述不准确） | `server.py:182-192,287,332` |

### 4.3 对外演示/部署缺口清单（[代码]+[推断]）

| 缺口 | 现状 | 至少需要 |
|---|---|---|
| 鉴权/授权 | 无；本机任意进程可 `/api/load`、`/ws` | token/session、来源校验、TLS |
| 网络边界 | `--host` 只 loopback；Origin/Host 白名单只 loopback | 反向代理下改写 Origin/Host 白名单、HTTPS（`getUserMedia` 需 secure context） |
| 并发/排队 | 单模型单 `infer_lock`，忙即报错；无应用层队列 | 按连接/用户排队、超时丢弃、背压 |
| 限流 | 仅消息大小 2 MB；无 QPS/连接数/帧率限制 | 连接数、QPS、单会话窗口速率限制 |
| 资源隔离 | 无 per-session 配额；模型切换要断全部会话 | 会话隔离、GPU 显存/队列上限、优雅停机 |
| 模型供给 | 首次从 HF 下载；无本地 `models/` 目录、无校验/版本固定 | 预置权重、sha256/版本校验、离线镜像 |
| 浏览器兼容 | Chrome 优先；依赖 secure context、getUserMedia、AudioWorklet、OffscreenCanvas/VideoFrame、MediaPipe WASM（CSP 需 `wasm-unsafe-eval`）、会议输出需 Chrome `setSinkId` | 明确支持矩阵/降级路径（Safari/Firefox 仅部分可用） |
| 可观测性 | 仅 `/api/status` + 日志 | 指标、追踪、错误聚合、审计 |
| 数据合规 | 音频/嘴部图发到服务端；仓库版只在本机 | 隐私声明、传输加密、留存策略 |

### 4.4 `falcon-big` 是否只能推理？（结论：是，发布仓库无训练侧代码）

| 证据 | 内容 |
|---|---|
| `look2hear/models/__init__.py` | 只导出两个模型类 `Falcon`、`FalconSpeakerContrastive`；**没有** `SpeakerContrastiveV2`/`SISNRRepel` |
| `look2hear/models/` 目录 | 只有 `Falcon.py`、`Falcon_Contrastive.py`、`base_model.py` |
| 仓库根 `configs/` | 只有 `Falcon.yml`、`LRS2_2Mix*.yml`；**没有 `FalconBig.yml`**（big 配置只在 `studio/configs/FalconBig.yml`） |
| `audio_train.py:260-262` | `getattr(look2hear.models, config["audionet"]["audionet_name"])` → 指向 big 配置会 `AttributeError` |
| `audio_visual_test.py:287-291` | 现行代码在缺类时**明确报错**（Phase 1 修复）：`look2hear.models has no attribute 'SpeakerContrastiveV2'`；`Experiments/checkpoint/Falcon/conf.yml:88` 正是 `SpeakerContrastiveV2` → 该快照评测不可直接用 |
| Studio 侧文件 | `SISNRRepel.py`(1498) = Studio `Falcon.py`(1496) + 2 行注释 import + 类名；`Speaker_Contrastive_v2.py`(140) = `Falcon_Contrastive.py`(122) + 18 行 docstring + base 类改 `SISNRRepel`（[验证] `diff`，与旧文档一致） |
| 影响 | big 只能按 Studio 代码推理（P2-B 即如此）；**无法在发布代码上训练/微调 big**，其训练代码与数据配方未公开 → 复现计划不能把 big 当“可训练基线” |

---

## 5. 与旧文档 `06-studio-backend.md` 的逐条对照

> 旧文档基线 commit `7576854`；本次 HEAD `2beec38`，**后端文件自基线未变**（§1.1），故下列差异主要来自“验证环境/权重/新增前端诊断/后续 P2-B”。

| # | 旧文档说法 | 本次判定 | 说明 |
|---|---|---|---|
| 1 | 验证环境 `/root/.venvs/falcon`、无 Pillow、无权重、禁止下载 | **修正** | 本次用项目强制 conda `falcon`（Pillow 13、真实权重已缓存），协议层不再需要 PIL stub |
| 2 | pytest 33 passed / 4 skipped | **修正** | 本次 **35 passed / 2 skipped**（`test_infer_file` 2 项因权重缓存而真跑；2 个 skip 是 GPU 条件用例） |
| 3 | 真实 `falcon.pt` 的 key/输出一致“未验证”（Q1） | **推翻（已解决）** | 真实权重 strict 加载 + R3 audio 等价 + 本次**全链路（含视频）max\|Δ\|=0**；Q1 关闭 |
| 4 | A49 置信度“高（真实 falcon.pt 未验证）” | **升级为 [验证]** | 见 §3.2(b)；条件：同一 2 s 窗口、CPU、eval |
| 5 | big 的两份代码在训练仓库不存在；PROVENANCE 来源表 2 项不准确 | **确认** | `look2hear/models` 只有 2 个模型类；根 `configs/` 无 FalconBig.yml；`audio_train.py:261` getattr 会失败 |
| 6 | 旧文档引 `audio_visual_test.py:129-132` 用 getattr 构模型 | **修正行号** | 现行 `audio_visual_test.py:287-291` 已改为显式缺类报错（Phase 1 修复），语义仍是“不能加载 SpeakerContrastiveV2 快照” |
| 7 | 首窗 9280 / 左上下文 22720 / 切片 22720:30720 / FADE 320 | **确认** | §2.1/§2.2；新增 `sessionEpoch=160000` 数值例 |
| 8 | 旧文档“任务书的 22400 无代码依据” | **确认** | 代码/前端均为 22720 = 32000−8000−1280 |
| 9 | CPU 延迟（合成权重 small≈537 ms、big≈767 ms） | **确认量级** | 本次真实权重 CPU small p50 **543 ms**；GPU 新增 **47/56 ms**；big CPU 未单独测（本次只测 GPU big） |
| 10 | GPU 延迟“无法验证/无 3090” | **补充（仍不能对标 3090）** | 本次 A800 p50 47.1/56.5 ms，只说明“单窗远小于 500 ms”；论文 51.3 ms 口径见 §6.2 |
| 11 | 窗口化 vs 整句“不等价/可能不同”（旧 Q3，无量化） | **补充量化** | P2-B 300 条配对：small −1.912/−1.730 dB、big −2.318/−2.133 dB；信号级 SNR 16.06 dB、corr 0.980 |
| 12 | 论文 51.3 ms 未分析口径 | **新增结论** | `README.md:213-226` 明确是“2 s、16 kHz、50 帧、RTX 3090/Intel Xeon、batch=1、trainable params only”的**单次片段前向**；不是端到端流式延迟 |
| 13 | `app = create_app()`“创建 Engine（不 autoload）”（A62） | **修正** | `create_app(autoload=True)`；import 时不加载，但 `uvicorn server:app` 的 lifespan 启动会 `start_load()`（`server.py:185-192,287`） |
| 14 | 部署边界：仓库 `server.py` 不能直接对外（A54/A63） | **确认** | 本次未改代码；`--host`/Origin/Host 白名单仍只 loopback |
| 15 | `models/` 错误文案把“缺包”误报成“缺权重”（Q8） | **确认** | `server.py:120-121,140-141` 未变；`resolve_checkpoint` 无本地 `models/` 逻辑 |
| 16 | 依赖版本 `torch>=2.14.1` 等“未验证是否有差异”（Q6/Q7） | **补充实测** | `check_environment.py` 在本环境 **exit 1**（3 项版本不符）；`einops` 运行期未 import |
| 17 | `infer_file` 任意长度“仅代码级确认”（A51） | **升级 [验证]** | `test_infer_file.py` 本次真跑通过（2 s 一致、3.7 s 输出形状/有限） |
| 18 | 并发/`busy` 竞态、热路径 `status()`、无超时（§4.11） | **确认** | 代码未变；前端 `bounded()/falconDiagnostics()` 是新增诊断（§2.6），未改变“无服务端超时” |
| 19 | 前端把 error 当致命（Q5） | **确认** | `app.js:728,780-783` 未变；07 域同结论 |
| 20 | 旧文档“M5 Pro 0.12/0.65 s 无法验证” | **确认（仍未验证）** | 本机 A800/CPU 不可比；保留为未解问题 |

---

## 6. 对复现演示的价值与边界

### 6.1 Studio 是不是“生产可用/论文指标”的真实证据？

**不是；它是“模型 + 权重 + 协议能跑通且与训练代码等价”的真实证据，是 demo 级工具。** 具体：

- **能证明**：真实权重 strict 加载；2 s 窗口前向数值正确（与训练侧 CPU 全链路 max|Δ|=0）；协议/时间轴/错误处理完整；
  GPU 单窗远低于 hop（A800 0.094–0.113 RTF）；`infer_file` 与评测整句路径可复用同一运行时。
- **不能证明**：生产可用（无鉴权/并发/限流/超时/浏览器矩阵，§4.3）；论文流式质量（窗口化自身 −1.9~−2.3 dB，P2-B）；
  论文延迟（51.3 ms 是单窗前向，不是端到端，§6.2）；CPU 实时（543 ms > 500 ms）。
- **对复现报告的正确用法**：定性演示/样例生成可用 Studio；**报 SI-SNR/SDR 必须用整句评测协议**（`audio_visual_test.py`），
  若要用 Studio 的实时数字，必须显式标注“windowed protocol，比整句低约 2 dB，首窗仅 0.58 s 真实上下文”。

### 6.2 与论文“streaming / 51.3 ms on 3090”的关系

| 口径 | 数值 | 含义 | 证据 |
|---|---|---|---|
| 论文 GPU latency | **51.3 ms** | RTX 3090、2 s/16 kHz/50 帧、batch=1、**单次片段前向**（表注“trainable parameters only”是参数列口径，延迟是否含冻结 FRCNN/预处理未说明） | `README.md:213-226` |
| 本机 A800 单窗前向 | **47.1 ms**（small）/ **56.5 ms**（big） | Studio 每个 hop 实际执行的模型前向（含嘴部预处理、H2D/D2H） | §3.4 |
| Studio 每 hop 计算 RTF | 0.094 / 0.113 | 前向耗时 / 0.5 s hop | 由 §3.4 推导 |
| Studio 设计端到端延迟 | **≈0.7–0.8 s** | 0.5 s hop + 80 ms guard + 推理 + 播放缓冲 | `studio/README.md:99` |
| 算法输出年龄（发送时） | **80–580 ms（均值≈330 ms）** | `GUARD + r`，`r∈[0, HOP)` | `app.js:621-624` 推导 |

**结论**：论文 51.3 ms 与 Studio 的“每 hop 一次 2 s 前向”属于**同一量级的单窗前向**（本次 A800 实测同量级；论文口径是否含冻结 FRCNN/预处理未说明，见 §7-Q1），
但论文数字**不包含** 0.5 s 分块、80 ms 保护带、播放缓冲、采集/编码/网络，也不是“每输出一个采样”的延迟；
Studio 的 0.7–0.8 s 才是系统级设计延迟。论文的 “causal/streaming” 是**窗口级因果**（窗口内时间因果 mask + 全窗统计），
不是样本级流式；Studio 的 2 s/0.5 s 包装与论文延迟口径**不可直接互换**。

### 6.3 若要对外演示/部署，缺什么（按优先级）

1. **P0 安全/网络**：鉴权、TLS/HTTPS（`getUserMedia` secure context）、Origin/Host 白名单改造、反代适配；
2. **P0 并发**：请求队列 + 超时 + 背压；把“模型忙”从致命错误改为可跳过（前后端契约一起改）；
3. **P1 资源/限流**：连接数、QPS、每会话 hop 速率、GPU 显存/队列上限；模型切换与优雅停机；
4. **P1 模型供给**：预置权重/离线镜像、版本与 sha256 校验、本地 `models/` 或显式 `--checkpoint`；
5. **P1 可观测性**：指标/日志/错误聚合；把 `warning`（GPU 降级 CPU）、`last_inference_ms`、跳窗数暴露到 UI/监控；
6. **P2 浏览器矩阵**：Chrome 优先，明确 Safari/Firefox 的降级（AudioWorklet/OffscreenCanvas/MediaPipe WASM/setSinkId）；
7. **P2 质量策略**：首窗补零/弱上下文的产品提示；或改为“先整句、后流式”的两段式演示。

### 6.4 对复现计划的具体建议

- **演示端**：用 Studio（本地 loopback）做定性样例；GPU 机器上 small/big 都能实时（p50 47/56 ms），
  CPU 机器明确告知会跳窗（543 ms/窗）。
- **指标端**：一律用整句评测（`audio_visual_test.py` + 修正后的 metrics）；不要把 Studio 窗口化输出当作论文数字来源。
- **等价性**：引用本文 §3.2(b) 的“真实权重全链路 max|Δ|=0”作为“Studio 副本 = 训练代码”的证据；
  引用 §3.2(c) 说明 big 只能推理、无训练侧代码。
- **部署**：若要公网 demo，按 §6.3 清单另开分支改 `server.py`（当前分支不要动源码）。

---

## 7. 未解问题

1. **论文 51.3 ms 的精确测量口径**：是否含视频分支/预处理/H2D-D2H？是否 3090 上 2 s 片段单次前向？
   README 只给“2 s clip, 50 frames, batch=1, trainable params only”（`README.md:213`），无法从代码复算。
2. **M5 Pro 0.12 s（Metal）/0.65 s（CPU）**：本机无 Apple 设备，未验证（`studio/README.md:99`）。
3. **big 的训练代码/配方**：`SpeakerContrastiveV2`/`SISNRRepel` 仅存在于 Studio；训练代码、数据配方、loss 细节未发布（§4.4）。
4. **README 在线 demo 的部署补丁**：公网地址与 `same_origin`/`TrustedHost` 矛盾，仓库内无对应代码（`studio/README.md:16`；旧文档 Q4）。
5. **窗口化掉点的进一步拆解**：P2-B 归因为“首块 0.58 s 真实上下文 + 边缘零填充 + 分块上下文”，但三者各自贡献未拆分。
6. **MediaPipe 裁片分布 vs 训练裁片**：Live 模式是 landmark 引导的方形裁片，与训练数据预处理不完全一致（07/P2-A 已量化部分变体，但非本文范围）。
7. **浏览器兼容性实测**：本次未启服务、未跑浏览器（只读代码 + Node 测试不在本文范围），Safari/Firefox 行为未验证。
8. **CPU 实时目标硬件**：本机 x86 A800 主机 CPU 543 ms/窗；目标用户 M 系列/其他 CPU 的表现未知。
9. **`space/` 变体**与 Studio 的运行时差异（ZeroGPU、整段 60 s）由 07 覆盖，本文未展开。

---

## 8. 验证记录

### 8.1 环境自检（[验证]）

```bash
/root/miniforge3/envs/falcon/bin/python -V                      # Python 3.11.17
/root/miniforge3/envs/falcon/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 2.8.0+cu128 True 2
git rev-parse --short HEAD                                      # 2beec38
```

### 8.2 命令与结果

| # | 命令 | 结果 | 耗时 |
|---|---|---|---|
| 1 | `PYTHONPATH=$PWD python -m pytest studio/tests -q -rs` | **35 passed, 2 skipped**（37 collected） | 47.8 s（复跑 18.9 s） |
| 2 | `PYTHONPATH=$PWD/docs/verification/stubs:$PWD python docs/verification/scripts/verify_r3_studio_weights.py` | strict OK（525+137）；**max\|Δ\|=0**；params 537,697 | <1 min |
| 3 | 本次 inline：真实 `falcon.pt` 全链路（Studio `FalconRuntime` vs 训练侧 audio+FRCNN） | video embedding max\|Δ\|=0；**全链路 max\|Δ\|=0（bit-identical）**；11,724,385 params | ~1 min（CPU） |
| 4 | `PYTHONPATH=$PWD/docs/verification/stubs:$PWD python docs/verification/scripts/verify_studio_backend.py` | 全段 OK（A 运行时 + B 协议；B5 13 字段、时间戳 22720/30720 等） | <1 min |
| 5 | inline：GPU/CPU 延迟（真实权重；额外 warm 后 30/5 次） | small/cuda p50 **47.1 ms**、big/cuda **56.5 ms**、small/cpu **543.3 ms**；峰值显存 383/568 MiB | ~1.5 min |
| 6 | inline：协议数值（b64、首窗 epoch 0/160000、OutputSlicer 三窗、future 扰动、窗口位置差） | 170668/42668；9280/22720/30720/320；拼接 max\|d\|=2.98e-8；尾部×10 改变 7988/8000 输出样本 | <10 s（不含模型加载） |
| 7 | inline：两份 ckpt 元数据/键/视频张量比较 | small 525+137/epoch 714/47,156,343 B；big 517+137/epoch 715/49,645,087 B；big 少 8 个 out_proj；video 60/137 不同 | <5 s |
| 8 | `python studio/scripts/check_environment.py` | **exit 1**：torch/numpy/Pillow 三项版本不符 | <2 s |
| 9 | `git log 7576854..HEAD -- studio/{server.py,falcon_runtime.py,infer_file.py,configs,tests,falcon_model,scripts/check_environment.py}` | 空（后端自基线未改）；`607352c` 仅前端诊断 | <1 s |

### 8.3 限制（诚实声明）

- 未启动 `server.py`/uvicorn、未占端口、未跑浏览器（遵守任务约束）；协议层用 `TestClient` + `verify_studio_backend.py` + 仓库测试覆盖。
- 未联网、未下载、未训练；权重来自 `/root/.cache/falcon_weights/` 的 HF 缓存符号链接。
- GPU 延迟是 **A800** 而非论文 3090；只用于“是否实时”的判断，不作论文对标。
- “窗口位置/未来扰动”实验用**合成随机音频 + 随机嘴部帧**，模型输出幅度很小，故只报告相对量；真实语音/真实嘴部下的绝对幅度会不同，但“非逐样本流式”的结论由代码结构（全窗 GroupNorm/无状态接口）支撑。
- 未修改任何源码/配置/他人文档；本文是唯一写入文件。
- 与 07 前端文档的分工：前端 `bounded()/falconDiagnostics()` 的完整行为、Node 测试、Space 差异以 07 为准，本文只列与后端契约相关部分。
