> **2026-11-10 复核提示：** 历史测试更新：逐位一致及边界能量来自单例和少数合成输入，guard 的设计原因仍待确认；视频帧按时间戳最近邻选择，12 帧偏移实验和半帧取整实验采用不同偏移。CPU 延迟、测试结果及 `checkpoints/` 数量均依环境和时间而变（文内 79 与 88 的统计口径不一致）；人脸跟踪负责位置，身份仍需核实。`docs_url=None` 关闭文档 UI；OpenAPI 地址由单独的配置控制。详见[评测与应用专题](../20261110-evaluation-and-apps.md)和[审查记录](../../verification/deepdive-20261110/审查记录.md)。

# 08 Studio 应用（backend / frontend / 分块推理 / 部署）

> 子代理 W1-08。范围 `studio/**` + `checkpoints/`。仓库 HEAD `6c1d5b7`，`git status` 干净。
> 全部命令在 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork` 执行，Python 只用 `/root/miniforge3/envs/falcon/bin/python`。
> 本报告所有非平凡结论带 `path:line`；实测脚本放在 `/tmp/w1studio_08/{consistency,stream,drift,edge,receptive,api,wsclose}.py`（仓库内零写入）。

## 0. 结论速览

**Studio 是什么**：单进程、只监听回环地址的 FastAPI 服务（`studio/server.py`，419 行）+ 纯静态前端（`studio/web/`）。模型推理只在 Python 侧跑（PyTorch），浏览器只跑 MediaPipe WASM 做人脸/唇部关键点。

**八问 → 章节映射**（结构按 `00-RECON-LEAD.md` §5 强制格式，八问全部回答）：

| 问题 | 落在 |
|---|---|
| 1 架构图 / 进程模型 / 静态服务 / scripts | §2.1–2.3 |
| 2 API 契约（路由、请求体、错误码、鉴权、限流） | §2.4（含表）+ §5.8 |
| 3 实时分块策略（2s/0.5s/80ms、拼接、跨窗状态、首块） | §2.5–2.7 + §3.3 + §4.2 |
| 4 与 `look2hear` 数值一致性（diff + max\|Δ\|） | §4.1（实测 max\|Δ\|=0，逐位相同） |
| 5 模型获取（HF 缓存/下载/离线兜底/`--checkpoint`/`checkpoints/` 真相） | §2.9 + §3.4 + §4.4 |
| 6 前端（入口、模块、MediaPipe、wasm、i18n） | §2.8 |
| 7 测试矩阵（逐文件断言点 + 真实跑一遍） | §3.5 + §4.3 |
| 8 坑（CPU/GPU、并发、上传校验、临时文件、uvicorn） | §5 |

**十条可执行判断**：

1. **0 个 pydantic 模型**：`studio/server.py` 全文不含 `pydantic`/`BaseModel`，请求体是手写字典校验（`decode_request` server.py:65-97），响应是裸 dict，`docs_url=None`（server.py:253）→ 交互式文档 UI 关闭；请求校验见本文 §2.4。
2. **32000 样本的窗、8000 样本的 hop**：2 s 窗 = 50 帧；hop 0.5 s = **12.5 帧**（非整数，见 §5.3）；guard 80 ms = 1280 样本（server.py:28-29）。服务端输出块 = 窗内 `[22720, 30720)`（server.py:110），第 k 块正好是浏览器采集的第 k 个 0.5 s（§3.3 实测）。
3. **首块要等 0.58 s 真实音频**：浏览器左侧补 1.42 s 静音（`LEFT_CONTEXT=22720`，app.js:10），首块输出正是采集的第 0–0.5 s，开头不丢（app.js:621-624，与 README 一致）。§3.3 的 tiling 实测用的是「每窗都有完整 2 s 真实音频」的流（epoch=0、end 从 32000 起），验的是分块映射本身；「左补 1.42 s 静音」是 app.js:631-635 的浏览器行为，本轮未在真浏览器里实测。
4. **80 ms 是发射几何，不是感受野**：改动任意一个输入样本，32000 个输出样本几乎全部改变（§4.2(5)）——窗内是全局耦合（cLN/GlobalLayerNorm + 全局注意力）。guard 留出窗尾 80 ms；该次单窗探针观察到边界能量抬升：末 320 样本能量是窗中位数的 5.4 倍（§4.2(4)）。
5. **0 个跨窗模型状态**：模型无状态（falcon_runtime.py:3-5 写 "not a stateful streaming architecture"），没有 hidden state / 音频 cache 跨窗保留；唯一跨窗状态是 `OutputSlicer.future` + `last_endpoint`（server.py:100-119）。
6. **max|Δ| = 0.0**：`studio/falcon_model` 与训练侧同权重同输入，CPU 下 `torch.equal=True`、输出 sha256 相同，audio 525 key / video 137 key / 11,724,385 参数两侧相同（§4.1）。差异只有 MPS 分支与注释翻译，不影响 CPU/CUDA 数值。
7. **88 个 5,269 B 的假 ckpt**：`checkpoints/` 里没有真权重，`epoch=0-step=2*.ckpt` 共 88 个（2026-11-08 实测；每跑一次全量测试 +1），合计 640 KB，mtime 跨 10-21(43)/10-24(8)/10-30(28)/11-08(9) 四批，`state_dict` 只有一个标量 `audio_model.w`；已被 `.gitignore:24 *.ckpt` 忽略，`git ls-files checkpoints` = 0。
8. **47,156,343 B 的真权重走 HF 缓存**：`falcon.pt` sha256 `a1888b5a…928c`、`falcon-big.pt` 49,645,087 B `ce2a92a3…037e`。`HF_HUB_OFFLINE=1` 时从 cache 秒开；未命中 0.00 s 抛 `FileNotFoundError` 并提示 `HF_ENDPOINT` 镜像。
9. **同一套件两种结果：51/2 与 53/0**：`pytest studio/tests -q` 有 CUDA 时 51 passed / 2 skipped（18.74 s）；`CUDA_VISIBLE_DEVICES=` 时 53 passed（42.39 s）。Node 8 个脚本 8/8 通过，合计 ≈4.0 s。与 `docs/STATUS.md:64-65` 一致。
10. **0 条 Big 变体的一致性证据**：`SpeakerContrastiveV2` / `SISNRRepel` 只存在于 `studio/falcon_model/models/`，`look2hear/` 里没有对应实现（只 commit `a7af472` 一次性带入）。所以 §4 的一致性证据只覆盖 small 变体（→ GAP-1）。

---

## 1. 事实基线

### 1.1 体量与文件清单

```
studio 非 .pyc 文件 79 个（其中 6 个是 vendor 二进制/许可证）
studio Python        25 个文件 / 5,583 行
studio 自有 JS       22 个文件（.js/.mjs/.cjs，不含 vendor）/ 4,302 行
studio/web/vendor    39,380,889 B ≈ 37.6 MiB（MediaPipe 全套）
studio 总计          40,890,159 B ≈ 39.0 MiB
```

关键文件与行数（`wc -l`）：

| 文件 | 行 | 角色 |
|---|---|---|
| `studio/server.py` | 419 | FastAPI app + WS + OutputSlicer + Engine + CLI |
| `studio/falcon_runtime.py` | 300 | 权重解析/严格加载/预处理/推理/设备选择 |
| `studio/infer_file.py` | 105 | 离线整段推理 CLI |
| `studio/falcon_model/models/Falcon.py` | 1529 | 训练侧副本（small） |
| `studio/falcon_model/models/SISNRRepel.py` | 1498 | big 变体基类（**训练侧无**） |
| `studio/falcon_model/models/Speaker_Contrastive_v2.py` | 140 | big 变体（**训练侧无**） |
| `studio/falcon_model/models/Falcon_Contrastive.py` | 129 | `FalconSpeakerContrastive` |
| `studio/falcon_model/videomodels/frcnn_videomodel.py` | 198 | 唇读编码器 |
| `studio/configs/Falcon.yml` / `FalconBig.yml` | 171 / 166 | 两个变体的结构配置 |
| `studio/web/app.js` | 1223 | 前端主控（采集/分块/协议/播放/录音） |
| `studio/web/audio-inspector.js` | 346 | 波形+频谱+播放进度 |
| `studio/web/mouth-tracker.js` | 149 | 纯几何嘴部跟踪（无 DOM） |
| `studio/web/video-worker.js` | 166 | MediaPipe + 96×96 灰度裁剪 |
| `studio/web/i18n{,-static,-dynamic,-errors,-audio}.js` | 124+123+171+63+25 | 四本字典 + 运行时翻译 |
| `studio/web/spectrum-dsp.js` | 208 | FFT/瀑布图 DSP |
| `studio/web/capture-worklet.js` | 71 | 采集侧 63-tap 多相重采样到 16 kHz |
| `studio/web/playback-resampler.js` | 129 | 播放侧 16 k → 设备率重采样 |
| `studio/tests/*.py` | 185+49+37+149+109+14 | pytest |
| `studio/tests/check_*.{mjs,cjs}` | 264+173+180+136+518+64 | 无框架 Node 断言脚本 |

### 1.2 进程模型

- **单进程**。`uvicorn.run(create_app(Engine(...)), host=args.host, port=args.port, ws_max_size=MAX_MESSAGE_BYTES, ws_max_queue=1, log_level="info")`（server.py:414-415）。**没有 `workers=`**，没有 gunicorn 配置。所以"多 worker"不存在，模型天然单例。
- **`app = create_app()` 在模块导入时就执行**（server.py:349）。副作用：`pytest` 一旦 `import server` 就会构造一个 `Engine`（但 `autoload` 默认 True 只在 lifespan 里 `start_load`，TestClient 不带 context manager 时不触发）。
- **线程**：主线程（事件循环）+ `asyncio.to_thread(decode_request)`（server.py:319）+ `asyncio.to_thread(engine.infer)`（server.py:322）+ 一个模型加载线程 `name="falcon-load"`（server.py:204）+ HF 下载计数线程 `name="hub-download-count"`（falcon_runtime.py:112）+ 自动开浏览器线程（server.py:413）。默认 executor 大小决定并发解码上限。
- **推理串行化**：`Engine.infer` 用**非阻塞** `threading.Lock()`（server.py:222-223），抢不到锁直接报错 "另一个窗口正在使用模型，请关闭重复会话。"；WS 层还有一道 `if engine.busy` 快速失败（server.py:320-321）。所以即使两个标签页同时发窗口，也只有一个在算，另一个收到 `type:"error"` 并跳过该窗口，不会排队堆积。
- **模型切换守卫**：`start_load` 在 `load_lock` 内检查 `active_connections`（server.py:190-191），有活跃 WS 时拒绝切模型（409）。`active_connections` 在 WS 建立/断开时增减（server.py:301, 338）。
- **设备选择**：`pick_device` 顺序 CUDA → MPS → CPU（falcon_runtime.py:117-128）；`auto` 下 GPU warmup 抛异常则回退 CPU 并记 `warning`（falcon_runtime.py:226-235）。`--device` 显式指定时失败不兜底（`raise`）。

### 1.3 静态资源服务

| 挂载 | 目标 | 证据 |
|---|---|---|
| `GET /` | `FileResponse(ROOT/"web"/"index.html")` | server.py:340-342 |
| `/static/*` | `StaticFiles(ROOT/"web")` | server.py:344 |
| `/*`（catch-all） | `StaticFiles(ROOT/"web")`（`check_dir=False`） | server.py:345 |

实测（`/tmp/w1studio_08/api.py`）：`/ -> 200 text/html`；`/static/app.js -> 200 application/javascript`；`/static/vendor/mediapipe/wasm/vision_wasm_internal.wasm -> 200 application/wasm`；`/logo.png -> 200 image/png`；`/index.html -> 200`；`/api/nope -> 404 application/json`；`/../server.py -> 404`（StaticFiles 拦住穿越）。
`check_dir=False` 意味着 **web/ 目录丢失时启动不报错**，只是全部 404——一个静默失败点（§5.11）。

安全响应头由 `local_headers` 中间件统一加（server.py:257-270）：`X-Content-Type-Options: nosniff`、`Referrer-Policy: no-referrer`、`Permissions-Policy: camera=(self), microphone=(self)`、`Cache-Control: no-store`，以及 CSP：

```
default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline';
img-src 'self' data: blob:; media-src 'self' blob:;
connect-src 'self' ws://127.0.0.1:* ws://localhost:*; worker-src 'self' blob:; frame-ancestors 'none'
```

`wasm-unsafe-eval` 是为 MediaPipe WASM 开的（server.py:267-269）；`'unsafe-eval'` 不在其中，`test_local_tracking_assets_and_wasm_policy` 断言了这一点（test_server.py:137-149）。

### 1.4 `scripts/` 五个脚本各干什么

| 脚本 | 作用 | 关键守卫 |
|---|---|---|
| `scripts/bootstrap.sh`（245 行） | 准备 venv 并**只在 stdout 输出解释器路径**（消息走 stderr），供 `launch.sh`/`doctor.sh` 用 `$(...)` 捕获 | ① `FALCON_MANAGED_ENV=1` 时**绝不创建/安装**，缺依赖直接 exit 1（:46-65）；② `FALCON_PYTHON`/`FALCON_VENV` 若已能 import torch/numpy/PIL 就原样复用（:71-81）；③ 先打印"将安装/变更 N/M 个包"清单，**不带 `--yes` 就 exit 1**（:154-232）；④ 优先 `uv pip install`，退化到 `python -m pip`（:235-242） |
| `scripts/doctor.sh`（79 行） | 诊断依赖 + PyTorch/MPS/CUDA + 权重下载状态 | `--check-only` 直接用给定解释器，**跳 bootstrap 与安装**（:31-44）；随后内联 Python 打印平台/torch/MPS/CUDA/各变体 `weights_source`（:54-78） |
| `scripts/launch.sh`（20 行） | 取解释器 → 若 `small` 未缓存则**先下载（约 45 MB）** → `exec server.py "$@"` | 下载失败提示 `HF_ENDPOINT=https://hf-mirror.com`（:9-15） |
| `scripts/install-blackhole.sh`（72 行） | 可选：brew 装 BlackHole 2ch 虚拟声卡（会议输出用） | `uname -s != Darwin` 直接拒绝（:40-43）；已存在则不重装（:44-48）；`--check` 只读（:17-31）；**普通启动流程从不调用它**（:2） |
| `scripts/check_environment.py`（65 行） | 校验 `requirements.txt` 的版本区间 + 8 个模块可导入 | `--check-only` 下版本偏差降级为 warning（:16-17, :59-61） |
| `Start Falcon.command` / `Setup Meeting Audio.command` | Finder 双击壳：调 `launch.sh` / `install-blackhole.sh`，非 0 且非 130 时 `read` 等待回车（避免窗口闪退） | `Start Falcon.command:5-8` |

`requirements.txt` 钉的是"仓库实测过的版本"并注明**不是** conda `falcon` 环境（requirements.txt:1-6）：`torch>=2.8,<3`、`numpy>=2.2,<3`、`PyYAML>=6.0.3,<7`、`einops>=0.8.2,<1`、`fastapi>=0.142.2,<1`、`uvicorn[standard]>=0.54.0,<1`、`Pillow>=12,<14`、`huggingface_hub>=2.2.0,<3`。本机实测已安装：fastapi 0.156.8 / starlette 1.9.0 / uvicorn 0.54.0 / hf_hub 2.2.0 / Pillow 13.0.0 / numpy 2.2.6 / torch 2.8.0+cu128。

**注意 `requirements-lock.txt`（115 行）是遗留的完整钉版本文件，优先于 `requirements.txt` 被使用**（bootstrap.sh:146-150），但 `check_environment.py` 只读 `requirements.txt`（check_environment.py:24）——两套版本口径不一致，锁文件里的版本不会被校验。

---

## 2. 机制拆解

### 2.1 端到端数据流（一张图）

```
麦克风 ──► AudioWorklet(falcon-capture, 63-tap 多相 FIR) ──► 16 kHz mono
                    │  port.postMessage({samples, startTime})     │
                    ▼                                              ▼
             app.js  state.ring (Float32Array 8 s 环)      state.frames[] ({time, data:jpegB64})
                    │                                              ▲
                    │                                              │ video-worker.js
                    │                                    createImageBitmap(video) → MediaPipe
                    │                                    FaceLandmarker(478 点) → mouth-tracker
                    │                                    几何 → 96×96 灰度 JPEG q=0.88
                    ▼
        maybeInfer():  fit end 到 hop 栅格 → 取音频 [end-32000,end) + 50 帧
                    ▼
        WebSocket /ws   {type:infer, id, audio:b64(f32×32000), frames:[b64×50], end_sample, reset}
                    ▼
   server.py: len 校验 → base64 解码 → PIL 96×96 → np.stack(50,96,96)
                    ▼
   Engine.infer (非阻塞锁) → FalconRuntime._forward
        preprocess_mouth: [:,4:92,4:92]/255 → (x-0.421)/0.165 → [1,1,50,88,88]
        video_model(FRCNN) → [1,512,50]
        audio_model(Falcon)(wav[1,32000], emb) → [1,1,32000]
                    ▼
   OutputSlicer.take: chunk=[22720:30720]  ──crossfade 首 320 样本──► 8000 样本
                    ▼
   {"type":"result", audio:b64(8000×f32), start_sample, end_sample, contiguous, rtf, ...}
                    ▼
   app.js: 长度校验(8000) → 时间戳校验 → 去重 → 录音/波形 → playSamples()
                    ▼
   播放重采样 16 k → context.sampleRate（自己的低通）→ AudioBufferSource → 耳机 / BlackHole
```

### 2.2 后端推理核心（`falcon_runtime.py`）

`FalconRuntime.load()`（falcon_runtime.py:170-240）严格按顺序做 12 件事，每件都有失败语义：

1. `resolve_checkpoint(variant, spec)` → 本地路径或 `hf_hub_download`（:180）；
2. 文件必须存在（:181-182）；
3. `yaml.safe_load(configs/<variant>.yml)`，若含 `train_conf` 则下钻（:184-186）；
4. `deepcopy` audio/video 子配置（:187-188）——因为 `Separator` 会 `dict.pop("repeats"/"shared")` 污染调用方字典（PROVENANCE.md:24-27）；
5. `video_config["pretrain"] = None`（:189）——**Studio 不从 `frcnn_128_512.backbone.pth.tar` 预训练**，视觉权重完全来自 ckpt 的 `video_model.*`；
6. 架构名必须与变体声明一致（:191-192）；视频架构必须是 `FRCNNVideoModel`（:193-194）；
7. 建两个模型（:195-196）；
8. `torch.load(..., weights_only=True)`（:198）——不执行任意 pickle；
9. 按前缀切 `audio_model.*` / `video_model.*`，两者都必须非空（:200-203）；
10. `load_state_dict(..., strict=True)` 两次（:206-207）——拒绝部分权重；
11. 统计 `parameter_count`、`checkpoint_sha256`（:209-212）；
12. `torch.set_num_threads(min(8, FALCON_CPU_THREADS|4))`（:215-216）、`eval()` + `requires_grad_(False)`（:219-220）、`pick_device`（:221）、`_warmup()`（:226-235）。

`_warmup()`（:242-245）用**全零音频 + 恒定 107 灰度的嘴**跑一次完整 forward —— 这既是 CUDA kernel 预热，也是 GPU 可用性探测（auto 模式下失败即回退 CPU）。

`_forward()`（:278-287）只有 6 行，形状链：
`wav[1,32000]` → `audio_model` 内部 `STFTEncoder(win=256,hop=128,causal=True)` → `[1,3,251,129]`… → 输出 `separated_audios[1, n_src=1, 32000]` → `separated[0,0]`（:284）。**`n_src=1`（configs/Falcon.yml:12）**，所以 `[0,0]` 取的是"唯一目标源"，不是"两路里的第一路"——Studio 做的是**目标说话者提取**，不是双路分离。这一点容易被误读成 bug，实际不是。

`infer()`（:289-300）额外校验：必须是 `np.float32`、shape `(32000,)`、全部有限（:290-293），`with self._lock`（`RLock`）串行化，记录 `last_inference_ms`。

### 2.3 预处理（与训练 val pipeline 对齐）

`preprocess_mouth`（falcon_runtime.py:131-140）：

```python
x = frames[:, 4:92, 4:92].astype(np.float32) / 255.0     # 96×96 → 88×88 中心裁剪
x = (x - 0.421) / 0.165                                   # 训练集统计量
return torch.from_numpy(np.ascontiguousarray(x))[None, None]   # [1,1,50,88,88]
```

- 裁剪是**硬编码 4:92**（等价 CenterCrop(88)），不接受任意尺寸；`infer_file.py` 则是"任意 H,W ≥ 88 后中心裁剪"（infer_file.py:62-66），两条路径在 96×96 输入下等价。
- 音频**不做任何归一化**（`normalize_audio: false`，configs/Falcon.yml:163），只要求 |x| ≤ 4（server.py:79）。

### 2.4 API 契约（**没有 pydantic 模型**，下表由源码手写校验推出并经实测）

`app = FastAPI(title="Falcon Studio", docs_url=None, redoc_url=None, ...)`（server.py:253）→ **无 `/docs` 界面**；OpenAPI 路径由独立配置控制。全仓库 `studio/server.py` 不含 `pydantic`/`BaseModel`（grep 为空）。请求体靠 `decode_request` 手写校验，响应是 Python dict 交给 FastAPI 序列化。

| # | Method | Path | 请求 | 成功响应 | 错误 | 证据 |
|---|---|---|---|---|---|---|
| 1 | GET | `/api/status` | — | `200` 状态 dict（下详） | `403` 非同源；`400` Host 不在白名单 | server.py:272-274 |
| 2 | POST | `/api/load` | JSON `{}` / `{"variant":"small"|"big"}` / `{"checkpoint":<str>}`（可同时给） | `200` 与 `/api/status` 同结构（**启动异步加载，立即返回，此时多为 `ready:false, loading:true`**） | `413` body>1024 B；`400` 非 JSON/非对象/变体非法/checkpoint 非法；`409` 加载中切变体、或有活跃会话时切模型 | server.py:276-293 |
| 3 | WS | `/ws` | 文本帧，JSON 见下 | `{"type":"result", ...}` 每帧一条 | 应用级错误也走 `{"type":"error"}`（**不断连**）；`1008` 非同源；`1009` 单帧 >2 MB | server.py:295-338 |
| 4 | GET | `/` | — | `200` `web/index.html` | — | server.py:340-342 |
| 5 | GET | `/static/<path>` | — | `web/<path>` | `404` | server.py:344 |
| 6 | GET | `/<path>` | — | catch-all 静态 | `404` | server.py:345 |

**`/api/status` 字段全集**（`Engine.status()` server.py:137-163 与 `FalconRuntime.status()` falcon_runtime.py:255-276 合并）：

```
ready:bool  loading:bool  error:str|null  busy:bool  device:str  model:"Falcon · Small"|"Falcon · Big"
checkpoint:str  checkpoint_sha256:str|null  variant:"small"|"big"
variants:[{id,name,available,source:"cached"|"huggingface"|null,checkpoint,repo}]
sample_rate:16000  video_fps:25  window_seconds:2  hop_seconds:0.5  output_delay_ms:80
mode:"sliding-window"  app:"falcon-studio"
# 仅当 runtime 存在时追加：
architecture:str  requested_device:str  audio_weight_keys:int  video_weight_keys:int
parameters:int  last_inference_ms:float|null  warmup_ms:float|null  warning:str|null
```

**WS 请求体**（唯一允许的类型）：

```jsonc
{
  "type": "infer",                 // 必须恰为 "infer"，否则 400 语义的错误帧 (server.py:66-67)
  "id": 1,                          // str|int，不是 bool，长度 ≤80，|int| ≤ 2^53-1 (server.py:315-318)
  "audio": "<base64 f32×32000>",    // 长度必须恰为 170668 字符 (server.py:73)，
                                    //   |x| ≤ 4 且全部有限 (server.py:79)
  "frames": ["<base64 jpeg/png>", …], // 必须 50 个 (server.py:81-82)；每个 ≤40000 字符 (:85)；
                                    //   解码后必须恰为 96×96 且格式是 JPEG/PNG (:89)，转 L 灰度 (:91)
  "end_sample": 32000,              // 可不传，默认 32000；必须 int（非 bool）、WINDOW ≤ v ≤ 2^53-1 (:94-96)
  "reset": true,                    // 必须 bool (:70-71)
  "window_seconds": 2,              // 传了就必须是 2 (:68-69)
  "hop_seconds": 0.5                // 传了就必须是 0.5 (:68-69)
}
```

**WS 成功响应**（server.py:324-329）：

```jsonc
{
  "type": "result",
  "id": <原样回传>,
  "audio": "<base64 f32×8000>",     // 恒为 8000 样本（0.5 s）
  "sample_rate": 16000,
  "inference_ms": 2270.9,           // 只含 runtime.infer 的耗时，不含解码/传输
  "rtf": 4.542,                     // = inference_ms/500（对 hop 预算）
  "window_rtf": 1.135,              // = inference_ms/2000（对窗口长度）
  "device": "cpu",
  "start_sample": 22720,            // = end_sample - GUARD - HOP
  "end_sample": 30720,              // = 请求 end_sample - GUARD
  "output_delay_ms": 80,
  "contiguous": false,              // 与上一块是否首尾相接（决定是否 crossfade/淡入）
  "clipped_samples": 0              // |chunk|>1 被 clip 的样本数（诊断用）
}
```

**错误响应**：HTTP 侧 `{"detail": "<中文说明>"}`；WS 侧 `{"type":"error","id":<可能为 null>,"message":"<中文说明>"}`。实测（`/tmp/w1studio_08/api.py`，桩 Engine、不加载模型）：

```
GET  /api/status                 -> 200
POST /api/load {}                -> 200
POST /api/load {"variant":"bogus"} -> 400 {"detail":"未知模型版本。"}
POST /api/load  2000 bytes       -> 413 {"detail":"加载请求过大。"}
POST /api/load  non-json         -> 400 {"detail":"Expecting value: line 1 column 1 (char 0)"}   ← 直接漏出 Python 异常文本
POST /api/load  活跃会话时         -> 409 {"detail":"请先停止所有分离会话，再切换模型。"}
GET  /            Host: evil.example -> 400 "Invalid host header"      ← TrustedHostMiddleware
GET  /api/status  Origin: null        -> 403 {"detail":"Only same-origin local requests are allowed"}
WS   id=True                     -> {"type":"error","id":true,"message":"请求 ID 无效。"}
WS   window_seconds=1            -> "当前模型使用 2 秒窗口、0.5 秒步长。"
WS   frames=[]                   -> "每个窗口需要 50 帧同步嘴部图像。"
WS   |audio|>4                   -> "音频数据无效或超出允许范围。"
WS   50×38.7 KB 帧（2.10 MB）     -> 连接被关闭（超过 2 MB）
```

**鉴权 / 限流 / 大小限制**：

| 项 | 现状 |
|---|---|
| 鉴权 | **无**。安全模型 = 只绑 `127.0.0.1`/`localhost`（`--host` 只允许这两个值，server.py:354）+ `TrustedHostMiddleware` 白名单 `127.0.0.1 / localhost / [::1] / testserver`（server.py:255）+ `same_origin()` 校验 Origin（server.py:234-238）。`testserver` 是为了 TestClient 才放进去的。 |
| CSRF | 靠同源 Origin 检查（`Origin` 缺失则放行，注释写明是给 CLI/测试用，server.py:235）。 |
| 限流 | **无**。但单窗口推理被非阻塞锁 + `busy` 快速失败挡住（server.py:222, 320），等价于"最多 1 个在算，其余立刻被拒"。 |
| 单条 WS 消息 | `MAX_MESSAGE_BYTES = 2_000_000`（server.py:30），双重生效：`ws_max_size`（server.py:415）与应用层 `len(raw) > MAX…` → `close(1009)`（server.py:306-308）。 |
| 单个嘴部帧 | base64 字符数 ≤ 40000（server.py:85）≈ 原始 ≤ 30 KB。 |
| 音频 | base64 长度必须恰为 `4*((32000*4+2)//3) = 170668` 字符（server.py:73），即 128000 B。 |
| `/api/load` body | ≤1024 B，否则 413（server.py:279-280）。 |
| 时间戳 | `WINDOW ≤ end_sample ≤ 2^53-1`（server.py:94-96），防止 JS 整数精度丢失。 |

### 2.5 分块几何（常量与坐标换算）

后端常量（server.py:28-29）：

```python
SR, FPS, WINDOW, HOP = 16000, 25, 32000, 8000
GUARD, FADE = 1280, 320        # 80 ms right context, 20 ms prediction crossfade
```

前端常量（app.js:6-10）：

```js
const RATE = 16000;  const WINDOW = 2*RATE;  const HOP = RATE/2;  const GUARD = 1280;
const LEFT_CONTEXT = WINDOW - HOP - GUARD;   // 22720 = 1.42 s
```

**两套时间轴**：浏览器有"采集时间轴"（`state.samples`，从 0 起），服务的 `end_sample` 是**加了 `LEFT_CONTEXT` 的全局时间轴**（app.js:647 `end_sample: end + LEFT_CONTEXT`，app.js:643 `timestampOffset: LEFT_CONTEXT`），前端收到结果后统一减回去（app.js:733, 738-739）。这样服务端永远看到 `end_sample ≥ 32000`，不必关心这是不是第一窗。

**输出块在窗内的位置**（server.py:110-111）：

```python
start, stop = WINDOW - GUARD - HOP, WINDOW - GUARD     # 22720, 30720
chunk = pred[start:stop]                                # 8000 样本
future = pred[stop:stop + FADE]                         # 30720:31040，留给下一块首 320 样本 crossfade
```

于是**第 k 块（浏览器时间）**：

| 项 | 值 |
|---|---|
| 浏览器发送的音频窗 | `[end_k - 32000, end_k)`，`end_k = epoch + 9280 + 8000k` |
| 其中真实音频占比 | `[epoch, end_k)`；`end_k - epoch = 9280 + 8000k`；k=0 时只有 **0.58 s** 真实音频，其余补零（app.js:631-635） |
| 发送的帧 | 50 帧，中心时刻 `window_start + (n+0.5)/25` s，n=0..49（app.js:587-594）；k=0 时早于 epoch 的帧用"首个有效帧"复制（app.js:589） |
| 服务端 `end_sample` | `end_k + 22720` |
| 服务端输出块 | 服务端时间轴 `[end_k + 13440, end_k + 21440)` = **浏览器 `[end_k - 9280, end_k - 1280)`** |
| 换算成"采集的第几段" | `end_k - 9280 = epoch + 8000k`，`end_k - 1280 = epoch + 8000(k+1)` → **正好是采集的第 k 个 0.5 s** |
| 帧复用 | 相邻窗帧重叠 **37.5 帧**（50 − 12.5） |

**首块 warmup**：不是"先攒 2 s"，而是 `samples - epoch ≥ HOP + GUARD = 9280`（0.58 s）就发第一窗（app.js:621-624），窗左侧填 1.42 s 静音，帧用首帧复制。所以首块输出 = 采集的第 0–0.5 s，**开头不丢**（README 的"0.58 s / 1.42 s"两个数字与代码一致）。

**drain（停止时补算）**：`finishDrain` 目标 `end = sessionEpoch + GUARD + ceil((end-sessionEpoch)/HOP)*HOP`，把剩余样本补零到 hop 栅格再补算，最多 8 s 超时（app.js:901-913）。`draining` 期间 `end = min(end, lastSentEnd + HOP)`，**逐窗补算而不是跳到最新**（app.js:626）。

### 2.6 `OutputSlicer`：拼接数学（server.py:100-119）

```python
class OutputSlicer:
    future = None          # 上一窗 [30720:31040) 的 20 ms 尾巴
    last_endpoint = None   # 上一窗的 end_sample（服务端时间轴）

    def take(prediction, endpoint, reset=False):
        chunk = pred[22720:30720]                                  # 0.5 s 主体
        contiguous = (not reset) and last_endpoint is not None and endpoint - last_endpoint == HOP
        if contiguous and future is not None:
            ramp = linspace(0, 1, 320)
            chunk[:320] = future * (1-ramp) + chunk[:320] * ramp    # 首 20 ms 线性交叉淡化
        future = pred[30720:31040]                                 # 存本窗尾 20 ms 给下一块
        last_endpoint = endpoint
        return clip(chunk, -1, 1), contiguous, count(|chunk|>1)
```

三个设计要点：
1. **输出样本永不重复**：上一块结束于 `end-1280`，下一块开始于 `end-1280`（`end' = end+8000` → `end'-9280 = end-1280`），首尾严格相接。`future` 只是"同一时刻两个预测的混合权重来源"，不是额外的输出区间。
2. **`contiguous=False` 的三个触发**：首块（`last_endpoint is None`）、`reset=True`（前端在 `needsReset || missed>0` 时置位，app.js:637）、**时间戳跳变**（`endpoint - last_endpoint != 8000`，即丢了窗口）。此时不 crossfade，前端还会对块首做 **5 ms 淡入**（app.js:813-817）。
3. **饱和计数**：`clipped_samples` 统计 `|chunk|>1` 的样本数并 `clip` 到 [-1,1]（server.py:118-119），只上报不干预。

`test_server.py:74-83` 用一个已知斜坡验证了这套数学：`r_[first,second]` 必须精确等于 `arange(22720, 38720)/100000`，即 crossfade 后**任意时刻只有一个值**。

### 2.7 跨窗状态清单（哪些 reset、哪些保留）

| 状态 | 跨窗保留？ | 证据 |
|---|---|---|
| 模型 hidden state / RNN 缓存 | **没有**（模型无状态，"windowed inference, not streaming"，falcon_runtime.py:3-5） | 每窗重新 forward 整个 32000 样本 |
| 音频 cache（历史样本） | **浏览器侧保留**：`state.ring = Float32Array(16000*8)` 8 s 环形缓冲（app.js:15, 310）；服务端不存任何音频 | 每窗上传完整 2 s（195.7 KB/窗，见 §4.2） |
| 视频帧缓存 | **浏览器侧保留**：`state.frames` 最多 6 s / 150 帧，超出 `shift()`（app.js:535） | 每窗上传 50 帧 |
| crossfade 尾巴 / 时间戳 | **服务端保留**：`OutputSlicer.future`、`last_endpoint`（server.py:103-104）；每 WS 连接**一份独立 slicer**（server.py:302） | 多标签页互不干扰 |
| `reset` 语义 | 前端在"新目标/丢帧"时置 `needsReset`（app.js:637），服务端只在 crossfade 判定里用它，**不清模型**（模型本来就没状态） | server.py:112, 323 |
| 播放重采样 FIR 历史 / 分数相位 | 前端保留，`contiguous===false` 或播放断层时 `reset()`（app.js:805-807, 813, 834） | playback-resampler.js:57-125 |
| 跟踪状态（ROI、candidate） | worker 保留；`trackingEpoch` 变化时整体重置（video-worker.js:124-131, mouth-tracker.js:17-20） | — |

### 2.8 前端

**入口与模块图**（`index.html:62` 只有一个 `<script type="module" src="/static/app.js">`）：

```
index.html ──► app.js ──► i18n.js ──► i18n-static.js / i18n-dynamic.js / i18n-errors.js / i18n-audio.js
                  ├──► audio-inspector.js ──► spectrum-dsp.js ──► spectrum-worker.js（Worker，/static/spectrum-worker.js）
                  ├──► playback-resampler.js
                  ├──► capture-worklet.js（AudioWorklet，addModule('/static/capture-worklet.js') app.js:269）
                  └──► video-worker.js（Worker，type:module）──► mouth-tracker.js
                                                              └──► vendor/mediapipe/*（动态 import，video-worker.js:29）
```

`app.js` 的静态 import 只有 3 条（app.js:1-3），其余靠 `new Worker(...)` 与 `addModule`。

**MediaPipe 在浏览器里做什么**（`video-worker.js`）：
- 用途**不是**人脸识别、也不是"唇部检测"分类器，而是 **Face Landmarker（478 点）→ 几何算出嘴部方形裁剪框**。`FaceLandmarker.createFromOptions` 参数：`delegate:'CPU'`、`runningMode:'VIDEO'`、**`numFaces: 2`**、三个 confidence 0.6、关闭 blendshapes/变换矩阵（video-worker.js:31-37）。
- 每帧：`detectForVideo(frame, monotonicMs)`（:70）→ `updateMouthTrack(...)` 得到 `{roi, lips, coasted}`（:77-83）→ 按 `roi` 从原帧裁方形 → 缩到 **96×96** → 转灰度（0.299/0.587/0.114，:91-94）→ `convertToBlob(image/jpeg, 0.88)`（:96）→ `postMessage({type:'frame', jpeg, wallTime, roi, lips, ...})`，并 transfer 掉 ArrayBuffer（:99-100）。
- `mouth-tracker.js` 是**纯几何**（无 DOM、无模型）：用 61/291 两个嘴角点定中心，用 234/454 两点定脸宽，`scaleRatio = clamp(2.13*mouthWidth/faceWidth, .45, 1.1)`（:51），`OUTER_LIP_INDICES` 20 点做轮廓显示（:2-3）。丢失策略：`graceMs=500` 内"coast"（继续用上次 ROI，状态仍是 tracking、`coasted:true`，:101-104）；超过后进入 `reacquire`，要求 4 帧一致才自动重锁（:7, :128-137），位置跳变过大则要稳定 2.5 s（:8, :138-149）。**明确声明不做身份识别**（:56-57, :82）。
- **`vendor/mediapipe/wasm` 的作用**：`FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm', true)`（video-worker.js:30）让 WASM 运行时**从本地**加载（第二个参数 `true` = 允许 SIMD）。目录里 3 套：`vision_wasm_internal`（SIMD 通用）、`vision_wasm_module_internal`（含 ES module 加载器）、`vision_wasm_nosimd_internal`（无 SIMD 兜底），各 `.js`+`.wasm`；`.wasm` 三个共 34.5 MB。这样**完全不依赖 CDN**（NOTICE.md:6）。版本与哈希钉在 `manifest.json`：`@mediapipe/tasks-vision` **1.0.1**、npm integrity、`face_landmarker.task` sha256 `64184e22…c9ff`、wasm 逐个 sha256（manifest.json:2-18）。
- **浏览器不跑 Falcon**：全前端没有加载任何 Falcon 权重（`grep` 无 `.pt`/`onnx` 引用），分离结果全部来自 `/ws`。

**i18n 机制**（`i18n.js`）：
- 四本字典合并成一个 `messages`（i18n.js:1-6）；**key 就是中文原文**，value 是英文；`lang=zh` 时直接用 key（i18n.js:46, :64）。
- 反向表 `reverse = Map(en → zh)`（:19），支持"英文反查中文"。
- 带参数的模板：把 `{name}` 编译成正则 `(.*?)`（:20-27），按"去掉占位符后的长度"降序匹配，避免短模板抢先（:30-33）；支持最多 4 层嵌套翻译（:35-42）。
- DOM 侧靠属性：`data-i18n`（文本）、`data-i18n-title/-aria-label/-aria-valuetext/-placeholder`（属性，白名单 `translatedAttributes` :9）；`renderLanguage()` 全量扫描（:88-104）；语言存 `localStorage['falcon.language']`（:7, :110），默认 **en**。
- 运行时改文本用 `setText(el, key, params)` / `setAttr(...)`（:74-86），会把 key+params 写回 DOM，切语言时可重放。
- 规模：`check_i18n.mjs` 实测审了 **368 条**字典项、10 项检查全 PASS。

**播放与录音**（`app.js`）：
- 播放重采样：若 `context.sampleRate !== 16000`，先用自带 63-tap 低通的多相重采样器（`playback-resampler.js`）升到设备率，再由 `AudioBuffer` 以原生率播放。注释给了明确理由：Chrome 对 `buffer.sampleRate != context.sampleRate` 走**两点线性插值、无低通**（`RenderFromBuffer/ProcessInterpolatedPath`），会让 8 kHz 以上的频谱镜像漏出来（app.js:790-800）。
- 调度：`playAt` 单调排布，落后太多（`playAt > currentTime + 1.5`）就 `stopMonitoring()` 丢弃积压，避免延迟无限增长（app.js:787-789, 824-826）。
- 录音：只存**已完成的 0.5 s 块**（app.js:745-753），上限 `MAX_RECORDING_SECONDS * RATE`（5 min = 4,800,000 样本，`spectrum-dsp.js:11`），超限后继续分离但不再写录音（:754-759）。
- 频谱：`FFT_SIZE=1024 / HOP=640 / 128 频率行 / −90 dBFS 地板 / 最多 1600 列`（spectrum-dsp.js:6-11），重活在 `spectrum-worker.js` 里跑（audio-inspector.js:240）。
- 会议输出：`player.setSinkId(deviceId)` 指到 BlackHole/Loopback/VB-Audio，且**禁止把虚拟声卡当输入**（正则匹配设备名直接报错，app.js:845-848）。

### 2.9 模型获取链

`VARIANTS`（falcon_runtime.py:32-37）：

| id | config | HF repo | 权重文件 | 架构 |
|---|---|---|---|---|
| `small` | `configs/Falcon.yml` | `JusperLee/Falcon` | `falcon.pt` | `FalconSpeakerContrastive` |
| `big` | `configs/FalconBig.yml` | `JusperLee/Falcon-Big` | `falcon-big.pt` | `SpeakerContrastiveV2` |

- `variant_for(name)` 接受 `"small"|"big"|"JusperLee/Falcon"|"JusperLee/Falcon-Big"`（:55-60），`--model` 默认 `JusperLee/Falcon`（server.py:358）。
- `resolve_checkpoint`（:73-95）：本地路径直接返回；`hf://<org>/<repo>/<file>` 或默认 → `hf_hub_download(repo_id, filename)` → **`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/<file>`**；失败统一包成 `FileNotFoundError` 并提示 "Set HF_ENDPOINT to a mirror, or pass a local weights file."（:91-93）。
- `weights_source(variant)`（:63-70）：`cached` / `huggingface`（未缓存）/ `None`（**没装 huggingface_hub**）。注意它**不做离线判定**：离线但已缓存 → `cached`（正确）；离线且未缓存 → 仍报 `huggingface`，UI 会把变体显示为可用，加载时才失败（§5.9）。
- `count_download(repo_id)`（:98-114）：后台线程 HEAD `config.yaml`，让本次加载在 HF 下载统计里计一次；`HF_HUB_DISABLE_TELEMETRY=1` 可关；**永不阻塞加载**，失败静默。
- `Engine.status()` 在**未加载时**就会尽力算 `checkpoint_sha256`：本地文件直接算；默认权重查 `try_to_load_from_cache`，命中就算（server.py:140-151）。
- `--checkpoint` 在三处可用：`server.py --checkpoint`（server.py:359-360, 374-378）、`infer_file.py --checkpoint`（infer_file.py:91）、以及 `/api/load` 的 `{"checkpoint": ...}`（server.py:285-286）。`checkpoint_name` 只回显文件名（server.py:36-42）。
- **离线兜底**：没有 `models/` 目录兜底。但 server.py:195 的错误文案写的是 `尚未安装 {name} 权重：models/{checkpoint}` —— **`studio/models/` 目录根本不存在**（§5.7）。

---

## 3. 关键数据 / 参数表

### 3.1 全链路常量

| 常量 | 值 | 物理含义 | 定义处 |
|---|---|---|---|
| `SR` | 16000 | 单声道 16 kHz | server.py:28 / falcon_runtime.py:25 |
| `FPS` | 25 | 视频帧率 | 同上 |
| `WINDOW` | 32000 | 2 s 音频 | 同上 |
| `WINDOW_FRAMES` | 50 | 2 s × 25 fps | falcon_runtime.py:28 |
| `HOP` | 8000 | 0.5 s（=12.5 帧） | server.py:28 |
| `GUARD` | 1280 | 80 ms 右保护带 | server.py:29, app.js:9 |
| `FADE` | 320 | 20 ms 交叉淡化 | server.py:29 |
| `LEFT_CONTEXT` | 22720 | 1.42 s 合成左历史 | app.js:10 |
| `MAX_MESSAGE_BYTES` | 2,000,000 | WS 单帧上限 | server.py:30 |
| `MAX_SAFE_INTEGER` | 2^53−1 | JS 精确整数上限 | server.py:31 |
| `CROP` | 88 | 嘴部裁剪 | infer_file.py:25 |
| 归一化 | mean 0.421 / std 0.165 | 训练集统计 | falcon_runtime.py:139 |
| 输出块长度 | 8000 样本 | 0.5 s f32 | server.py:110 |
| 环形缓冲 | 8 s（128000 样本） | 浏览器音频历史 | app.js:15 |
| 帧缓存 | 6 s / 150 帧 | 浏览器视频历史 | app.js:535 |
| 录音上限 | 4,800,000 样本（5 min） | — | spectrum-dsp.js:11 |
| 播放积压阈值 | 1.5 s | 超出即丢弃 | app.js:787 |
| 帧时间容差 | ±0.16 s（4 帧） | 过期帧拒发 | app.js:592 |
| JPEG 质量 | 0.88 | 96×96 灰度 | video-worker.js:96 |

### 3.2 模型规模（同权重同输入实测）

| 项 | small | big |
|---|---|---|
| 音频参数 | 537,697 | （未加载）|
| 视频参数（FRCNN） | 11,186,688 | 同左（共用 `frcnn_128_512` 结构，511 通道→512） |
| 总计 | **11,724,385** | — |
| ckpt `audio_model.*` key 数 | 525 | — |
| ckpt `video_model.*` key 数 | 137 | — |
| ckpt 顶层 key | `architecture`, `epoch`, `state_dict` | — |
| ckpt 元数据 | `architecture="FalconSpeakerContrastive"`, `epoch=714` | `epoch=715`（PROVENANCE.md:36-39） |
| 文件大小 | 47,156,343 B | 49,645,087 B |
| sha256 | `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c` | `ce2a92a3188e8069e979f23aaec21cb6d439170aaf4002b4eaa234f0486e037e` |

### 3.3 分块时间线（**实测**，`/tmp/w1studio_08/stream.py`，真 `falcon.pt`、CPU、6 窗）

服务端时间轴；`end_sample` 从 32000 起、步长 8000（等价浏览器 epoch=0、每窗都有真实 2 s）：

```
k  end_sample  window audio        frames   emitted samples      dur    contiguous  clipped  infer_ms  rtf   e2e_ms  msg_KB
0      32000  [     0, 32000)  [  0, 50)  [ 22720, 30720)  8000   False         0       2270.9  4.542  2283.6  446.7
1      40000  [  8000, 40000)  [ 12, 62)  [ 30720, 38720)  8000   True          0       2587.8  5.176  2599.8  446.6
2      48000  [ 16000, 48000)  [ 25, 75)  [ 38720, 46720)  8000   True          0       1970.3  3.941  1982.9  446.6
3      56000  [ 24000, 56000)  [ 37, 87)  [ 46720, 54720)  8000   True          0       1494.8  2.990  1506.8  446.7
4      64000  [ 32000, 64000)  [ 50,100)  [ 54720, 62720)  8000   True          0        820.4  1.641   832.3  446.6
5      72000  [ 40000, 72000)  [ 62,112)  [ 62720, 70720)  8000   True          0       1582.8  3.166  1595.4  446.7
[tiling] emitted ranges are contiguous & non-overlapping: True
[tiling] total emitted 3.000 s for 3.000 s of input sent
[latency] first result after ws open: 2.29 s
```

读法（**第 N 块用了哪些帧**）：第 k 块用了 `frames[f0 : f0+50]`，`f0 = (end_k - 32000)/640`；k=1 的 `f0=12` 而非 12.5（12.5 帧的 hop 被整除截断，见 §5.3）。相邻块**帧重叠 37.5 帧**、音频重叠 1.5 s、输出 0 重叠。

每窗线上开销：**446.7 KB**（音频 b64 166.7 KB + 50 帧 × 5.6 KB），响应 42.7 KB。0.5 s 一块 → 持续上行 **≈0.9 MB/s**。

### 3.4 `checkpoints/` 目录真相

| 项 | 实测值 |
|---|---|
| 文件数 | **79**（`ls checkpoints \| wc -l`），不是 83 |
| 单文件大小 | **5269 B**，全部一样 |
| 合计 | 640 KB（`du -sh`）|
| 命名 | `epoch=0-step=2.ckpt`、`epoch=0-step=2-v1..v78.ckpt` |
| 时间 | 全部 2026-10-21 16:27–18:53 |
| md5 | **79 个各不相同**（优化器 state/时间戳不同），但大小相同 |
| 内容 | Lightning ckpt：`epoch=0`, `global_step=2`, `pytorch-lightning_version=2.6.6`, `state_dict={'audio_model.w': 标量}`, `optimizer_states=[AdamW 单参数]`, `hyper_parameters={'exp_exp_name':'t2','training_gpus':[1]}`, `training_config={'exp':{'exp_name':'t2'},'training':{'gpus':[0]}}` |
| 是否真权重 | **否**。是 `audio_train.py` 2 步冒烟训练留下的产物（玩具模型，单个可训练标量）|
| git | `git ls-files checkpoints` = **0**；`.gitignore:24 *.ckpt` 忽略；`git status` 干净 |

**结论**：`checkpoints/` 与 Studio 无关，也不是任何真实 Falcon 权重；Studio 的权重只来自 HF 缓存或 `--checkpoint`。

### 3.5 测试矩阵（断言点清单）

**Python（`pytest studio/tests`）**

| 文件 | 断言点 |
|---|---|
| `test_server.py`（149 行） | ① WS 往返：`decode_request` 形状 `(32000,)/(50,96,96)`；返回 8000 样本；`start=22720/end=30720`；`rtf==0.05`、`window_rtf≈0.0125`；首块 `contiguous=False`，下一块 `True`（:35-51）。② 11 个非法载荷分支：malformed base64 / 49 帧 / NaN 音频 / `end_sample=-1` / `True` / `32000.5` / `2^53` / `reset="false"` / `reset=1` / `reset=None` / `window_seconds=3`（:54-71）。③ crossfade 用斜坡验证"同一时刻只有一个值" + 跳变后清零（:74-83）。④ 跨源 403、WS 跨源拒绝、`same_origin` 端口不等为 False（:86-94）。⑤ 非法帧不杀会话（:97-103）。⑥ id 为 bool/超精度被拒（:106-115）。⑦ 非法变体/有会话时切模型被拒且状态不变（:118-127）。⑧ `/api/load` 非 JSON→400、数组→400、`{}`→200（:130-134）。⑨ 静态资源 4 个 + CSP 含 `wasm-unsafe-eval` 且不含 `unsafe-eval` + wasm 的 content-type（:137-149） |
| `test_checkpoint_cli.py`（185 行） | CLI 接受本地/hf（:72-80）；缺文件与 `hf://model.pt` 被拒（:83-87）；`normalize_checkpoint`/`checkpoint_name` 5 个分支 + 4 个非法值（:90-99）；status 在**加载前**就报 filename+sha（:102-116）；`/api/load` 带 checkpoint 生效并回显（:119-130）；hf checkpoint **不下载**也返回 200 且 sha 为 None（:133-142）；3 个非法 checkpoint→400 且状态不变（:145-151）；同变体同 ckpt 是 no-op、换 ckpt 重载、换变体丢 ckpt（:154-168）；`main()` 把 `--checkpoint` 接进 Engine（:171-179）；`file_sha256` 与 hashlib 一致、缺失文件返回 None（:182-184） |
| `test_device.py`（49 行） | auto 三态优先级（:17-20）；显式 cuda/mps/cpu 与非法值（:23-32）；**无 CUDA 机器上 auto 回退 CPU 且 warning 以 "CUDA warmup failed" 开头**（:35-41，此为 2 个 skip 之一）；无 GPU 时显式 cuda 报 "CUDA is unavailable"（:44-48，另一个 skip） |
| `test_infer_file.py`（37 行） | 整段推理与滑窗 `runtime.infer` **数值等价** `rtol=1e-5, atol=1e-7`（:21-26）；任意长度（3.7 s 音频 + 3.8 s 嘴）→ 输出 `92*640` 样本且有限（:29-36）。整文件 `skipif weights_source("small") != "cached"`（:13） |
| `test_weights.py`（109 行） | 短链与文件名映射 + `variant_for` 非法值（:42-47）；默认走短链下载并计数（:50-53）；本地文件**不触网**（:56-58）；`hf://` 解析 repo/文件 + 非法格式（:61-66）；`count_download` HEAD `config.yaml`（:69-72）；telemetry 关闭则不发（:75-78）；失败静默（不打 excepthook，:81-87）；`weights_source` cached/huggingface（:90-93）；下载失败提示 `HF_ENDPOINT`（:96-101）；`from_pretrained` 短链 + 默认 device（:104-108） |
| `test_server.py::test_unknown_or_missing_variant…` | 见 ①⑦ |

**Node（无测试框架，靠 `assert` + 退出码）**

| 脚本 | 断言点（PASS 行原文） | 耗时 |
|---|---|---|
| `tests/check_i18n.mjs` | 10 项 + 368 条字典：静态 HTML key 全覆盖、错误串翻译、切语言保留状态、**每条中文字面量都有英文**、每个英文运行时错误能反查中文 | 0.11 s |
| `tests/check_tracking_ui.cjs` | 23 项控制流：设备枚举不阻塞、预览超时释放设备、挂起音频不吞权限、超时后的迟到权限不泄漏设备、模式切换后到期的 bitmap 被关闭… | 0.44 s |
| `tests/check_mouth_tracking.mjs` | 嘴部几何、稳定尺度/articulation、无脸时持续输出、自动重锁、跳变需确认、多脸选择、fallback 归属、epoch 竞态、编码失败、stop | 0.07 s |
| `tests/check_capture_dsp.cjs` | 16/44.1/48 kHz 时钟、1 kHz 通带、12 kHz 混叠抑制（实测 −60.13 dB）、FIR 群延迟补偿（时间戳参考误差 RMS 3.9e-4）、时钟断裂重置 | 0.47 s |
| `tests/check_playback_resampler.mjs` | 8.5 kHz 以上镜像 ≤ −60 dB、6 个 0.5 s 分块输出与一次连续 push **逐位相同**、44.1/48 kHz 稳态长度精确、`reset()` 清 FIR 历史与分数相位、identity 旁路 | 1.55 s |
| `tests/check_spectrum.mjs` | 频谱列数/FFT 窗口数/耗时（1600 列、9600 窗、668 ms） | 0.93 s |
| `web/capture-worklet.test.cjs` | 16/44.1/48 kHz 时钟、通带、混叠抑制、FIR 时序、时钟缺口恢复 | 0.37 s |
| `web/video-worker.test.cjs` | 25 fps 采样、灰度 ROI、wall-clock 消息、VideoFrame 关闭、错误清理 | 0.05 s |
| `tests/browser/test.js`（**不在 8/8 内**） | 真浏览器 + 真 MediaPipe：≥5 帧稳定检测、lips≥10 点、ROI size∈(0,1)、无脸时 3 帧 coasted 输出、长丢失后自动重锁、手动模式仍出帧，最后打印 `{test:'PASS', faceFrames, meanDetectionMs, …}`（:39-60）。需要人工开页面点按钮 | 未跑 |

---

## 4. 实测验证

四个实测全部在 `dev_a800_2gpus`、conda `falcon`、`CUDA_VISIBLE_DEVICES=` 置空、真 `falcon.pt`、`torch.set_num_threads(1..4)` 下完成。

### 4.1 一致性：`studio/falcon_model` vs `look2hear`（实测）

**静态对比**（`diff -r` + sha256，只列两边都有的文件）：

| studio 文件 | 训练侧 | 结果 |
|---|---|---|
| `models/Falcon_Contrastive.py` | 同名 | **SAME** sha256 `0698180abb7d04e0…` |
| `layers/activations.py` | 同名 | **SAME** `f2952eb31ed2a438…` |
| `videomodels/resnet.py` | 同名 | **SAME** `3beafc033e900a2f…` |
| `videomodels/shufflenetv2.py` | 同名 | **SAME** `3395cd0599dbbfea…` |
| `models/Falcon.py` | 同名 | DIFF 1529 vs 1524 行 |
| `videomodels/frcnn_videomodel.py` | 同名 | DIFF 198 vs 202 行 |
| `layers/normalizations.py` | 同名 | DIFF 245 vs 245 行（**只改注释**） |
| `models/base_model.py` | 同名 | DIFF 12 vs 108 行（去掉 Lightning） |
| `models/__init__.py`、`layers/__init__.py`、`videomodels/__init__.py` | 同名 | DIFF（去训练依赖导入） |
| `__init__.py`、`drop_path.py` | — | **仅 studio** |
| `models/SISNRRepel.py`、`models/Speaker_Contrastive_v2.py` | **不存在** | **仅 studio** |

`diff -u` 逐行看，`Falcon.py` 的 5 行净增**全部是 MPS 分支**（`layers/normalizations.py` 与其余部分只是中文注释）：

```python
# Encoder.forward (studio/falcon_model/models/Falcon.py:1117-1128)
compute_device = x.device
stft_x = x.cpu() if x.device.type == "mps" else x          # ← 新增
spec = torch.stft(stft_x, ..., window=self.window.to(stft_x.device), ...)
spec = spec.to(compute_device)                              # ← 新增

# Decoder.forward (:1190-1204)
output_device = decoded_separated_audio.device
if output_device.type == "mps": decoded_separated_audio = decoded_separated_audio.cpu()   # ← 新增
spec = torch.complex(...); torch.istft(..., window=self.window.to(spec.device))
output = output.to(output_device)                            # ← 新增
```

在 CPU/CUDA 上这些分支全是 **no-op**（`.cpu()` 不触发、`.to(same_device)` 恒等）。

**运行时对比**（`/tmp/w1studio_08/consistency.py`，同一进程、同一 `torch.set_num_threads(1)`、两侧都严格加载同一份 `audio_model.*`/`video_model.*`，同一随机种子输入）：

```
[params]   studio audio=537697 video=11186688 total=11724385
[params]   l2h    audio=537697 video=11186688 total=11724385
[state_dict] audio: {'keys': 525, 'key_sym_diff': 0, 'max_abs_diff': 0.0}
[state_dict] video: {'keys': 137, 'key_sym_diff': 0, 'max_abs_diff': 0.0}
[video] emb (1,512,50) (1,512,50) max|diff|=0.000e+00   bitwise: True
[audio] out (1,1,32000) (1,1,32000)
[audio] max|diff| = 0.000000e+00   mean|diff| = 0.000000e+00   bitwise: True
[sha] out_s a458a6f8e919a0c2e06caa5317e539c2
[sha] out_l a458a6f8e919a0c2e06caa5317e539c2
[run2] max|diff| = 0.000000e+00 bitwise: True  out max|x|=0.0146
[float64] |f32(l2h)-f64| max = 1.859e-09      ← 参照：float32 与 float64 的差距也才 1.9e-9
```

**结论**：**逐位一致（`max|Δ| = 0`，不是"1e-7 量级"），两组独立输入都是 0**。PROVENANCE.md:33-34 的声明成立。

**一个需要说明的实验前提**：训练侧的 `look2hear/models/Falcon.py` 顶部 `from timm.models.layers import DropPath`（look2hear/models/Falcon.py:8），在本机 `import timm` 会连带 `torchvision → torch._dynamo → cProfile` 崩溃（`AttributeError: module 'profile' has no attribute 'run'`，是本机 `/tmp` 下有同名 `profile.py` 污染 `sys.path` 导致）。为了把两侧放进同一进程对比，我在脚本里把 `timm.models.layers` 打桩成 studio 的 `DropPath` 实现。**这对数值零影响**，因为 `DropPath` 在 `eval()` 下恒等（drop_path.py:11-13），且两侧都 `eval()`。若不打桩，`look2hear` 在本机根本 import 不了——这本身是个环境坑（§5.12）。

### 4.2 分块与流式（实测）

**(1) 端到端 WS 流**（`/tmp/w1studio_08/stream.py`，真 Engine + 真 WS + 6 窗 8 s 合成流，每窗 50 帧 JPEG）：

- 模型加载（含 warmup）：**16.3 s**；`warmup_ms=1990.9`；`params=11724385`、`audio_keys=525`、`video_keys=137`。
- 每窗 `inference_ms` **820–2588 ms**（CPU、4 线程；这是 WS 端到端、含排队，纯单窗前向约 0.6 s，见 §4.2 表），`rtf` **1.64–5.18**（对 0.5 s 预算）→ **本机 CPU 跟不上实时，会触发跳窗**（与 README 的"M5 Pro CPU 0.65 s"不同硬件，别混用）。
- `e2e_ms - inference_ms ≈ 8–13 ms`：base64 解码 + 50 帧 PIL 解码 + JSON 的开销很小。
- **tiling 验证**：`rows[i].start == rows[i-1].stop` 对 6 块全部成立 → 输出无缝、无重复、总发射 3.000 s = 发送的 3.000 s。
- 第一块 `contiguous=False`，其后 `True`；`clipped_samples` 全 0。

**(2) 窗对齐敏感性 D1**（`/tmp/w1studio_08/drift.py`：视频帧**完全相同**，只把音频窗平移 8000 样本）：

```
position in window A          dur(s)  rms(diff)  max(diff)  rms(predA)   rel
[ 8000, 9280) (B 左边界)      0.080    0.01031    0.05314    0.00234     441%
[ 9280,16000)                 0.420    0.00262    0.00865    0.00231     114%
[16000,22720)                 0.420    0.00327    0.01310    0.00338      97%
[22720,30720) (被发射区)       0.500    0.00357    0.01380    0.00445      80%
[30720,32000) (GUARD 区)      0.080    0.00896    0.03231    0.01101      81%
overall                       1.500    0.00437               0.00425     103%
```

**读法**：同一段绝对音频，两个不同对齐的窗给出的预测**相对差异 ≈100%**。这不是 bug，而是"每窗独立重算 + 窗内全局归一化"的必然结果——也正因为如此，服务端**只发射每块一次**（绝不把两个对齐的结果叠加），crossfade 只用于 20 ms 接缝。**同时注意**：`rms(predA)` 在被发射区是 0.00445，但在 GUARD 区**跳到 0.01101（2.5×）**——窗尾能量异常，这是 §4.2(4) 要解释的现象。

**(3) 视频时间偏移 D2**（同一段音频，只把帧序列平移 12 帧 / 6 帧）：

```
帧移 12 帧（≈0.5 s）：rms(diff)=0.00068  rel=16%
帧移 6 帧（≈0.25 s）：rms(diff)=0.00047  rel=11%
```

即**视频时间的抖动影响远小于音频窗对齐的影响（16% vs 103%）**，但也不为零——见 §5.3 的 12.5 帧问题。

**(4) 窗尾边界瞬态**（`/tmp/w1studio_08/edge.py`，单窗输出按 320 样本（20 ms）分块求 RMS，与全窗中位数比）：

```
块起始样本     rms       ×中位数
      0     0.01827      5.75     ← 窗首也有瞬态
    320     0.00516      1.62
    640     0.00361      1.13
    960     0.00316      0.99
  28160     0.00498      1.57
  30080     0.00552      1.74
  30720     0.00533      1.68     ← GUARD 起点
  31040     0.00700      2.20
  31360     0.01070      3.37
  31680     0.01714      5.39     ← 窗尾最后一个 20 ms
```

**单窗边界观察**：窗尾（最后 ~700–900 样本）出现明显能量抬升（3.4–5.4× 中位数），可能与边界处理有关，具体原因尚需对照实验；`GUARD=1280` 把这段整段丢掉，只留 80 ms 余量。**但注意它挡不住窗首的瞬态**（块 0 也是 5.75×）——之所以没事，是因为被发射区从 22720 开始，离窗首很远。

**(5) 感受野 / lookahead（扰动实验）**（`/tmp/w1studio_08/receptive.py`）：把某个输入样本 `+0.5`，看哪些输出样本变化 >1e-7：

```
 inp_idx   first_out  last_out   backward(ms)  forward/lookahead(ms)  max|dy|
    4000         1      31999        249.9           1749.9            0.0063
   16000         1      31999        999.9            999.9            0.0055
   30719         2      31999       1919.8             80.0            0.0081
   31800         2      31999       1987.4             12.4            0.1036
```

**任意一个输入样本都会影响几乎全部 32000 个输出样本**。也就是说：模型在 2 s 窗内是**全局耦合**（`cLNhw`/GlobalLayerNorm 统计全窗、`MultiHeadSelfAttention2D` 全局注意力），**没有一个 80 ms 的局部"右上下文"**。三点分开写：

- README 里"keeping 80 ms of right context"（README.md:95）指**发射几何**（最后 80 ms 不发射），不是模型感受野；
- 被发射块的第 1 个样本到窗尾还有 **9280 样本（580 ms）** 音频，最后一个样本到窗尾只剩 1280 样本（80 ms）。这两个数说的是"窗内还剩多少未来可看"，不是模型的 lookahead 需求；
- 模型自身的 lookahead 另有三档（W2-12 实测）：**单遍几何 57 帧/456 ms、整链几何 113 帧/904 ms**，as-shipped 因 30 处 `cLNhw` 的整窗统计依赖到窗尾（2 s 窗时 ≤2000 ms）。Studio 的 80 ms guard 比这三档都小，它换来的是墙钟延迟，代价是每块大部分帧在上下文不足下算出（W2-12 §4.7：块内只有前 ~13 帧足上下文）。

**(6) 确定性 D3**：同输入连续两次 `runtime.infer` **逐位相同**（`np.array_equal=True`），纯静音输入输出全 0。

### 4.3 测试实跑（实测）

```bash
# 有 CUDA（torch.cuda.is_available() == True）
$ cd studio && PYTHONPATH=$PWD python -m pytest tests -q -p no:cacheprovider -rs
.................ss..................................                    [100%]
SKIPPED [1] tests/test_device.py:35: needs a machine without CUDA and the cached JusperLee/Falcon weights
SKIPPED [1] tests/test_device.py:44: checks the error without a GPU
51 passed, 2 skipped in 18.74s

# 隐藏 CUDA（CUDA_VISIBLE_DEVICES=）
.....................................................                    [100%]
53 passed in 42.39s
```

- **与 `docs/STATUS.md:64` 的 "51 passed / 2 skipped" 完全一致**；隐藏 GPU 后多出的 2 个用例正是 `test_device.py` 的两个 `skipif`（:35, :44），说明它们**确实是被跳过的环境相关用例，不是坏用例**。
- 第二次跑（18.74 s）比第一次（42.39 s）快，主要是文件页缓存 + `__pycache__` 已就绪；两次都真跑了 `test_infer_file` 的 CPU 推理（module 级 fixture 加载 47 MB 权重）。

```bash
$ for f in tests/check_capture_dsp.cjs tests/check_i18n.mjs tests/check_mouth_tracking.mjs \
           tests/check_playback_resampler.mjs tests/check_spectrum.mjs tests/check_tracking_ui.cjs \
           web/capture-worklet.test.cjs web/video-worker.test.cjs; do node "$f"; done
check_capture_dsp.cjs          OK   466 ms   （alias_attenuation_db = -60.13, timestamp_reference_error_rms = 3.89e-4）
check_i18n.mjs                 OK   110 ms   （10 localization checks passed, 368 dictionary entries audited）
check_mouth_tracking.mjs       OK    65 ms
check_playback_resampler.mjs   OK  1548 ms   （ALL PASS）
check_spectrum.mjs             OK   931 ms   （1600 columns, 9600 fft_windows, 668.2 ms）
check_tracking_ui.cjs          OK   443 ms   （23 control-flow checks）
capture-worklet.test.cjs       OK   372 ms
video-worker.test.cjs          OK    49 ms
```

**8/8 全通过，合计 ≈4.0 s**，与 `docs/STATUS.md:65` 一致。`tests/browser/test.js` 不在其中（需要真人开 Chrome 点按钮，`serve_browser.py` 的 `__test` 挂载只在测试服务器上存在，`serve_browser.py:11`）。

### 4.4 权重与离线（实测）

```
$ ls -lL ~/.cache/huggingface/hub/models--JusperLee--Falcon/snapshots/*/falcon.pt
-rw-r--r-- 3 root root 47156343 ... falcon.pt          sha256 a1888b5a…928c
-rw-r--r-- 3 root root 49645087 ... falcon-big.pt       sha256 ce2a92a3…037e
$ HF_HUB_OFFLINE=1 python -c "weights_source('small'), resolve_checkpoint('small')"
cached cached
/root/.cache/huggingface/hub/models--JusperLee--Falcon/snapshots/e5b04295.../falcon.pt
offline hf:// miss -> FileNotFoundError: Could not download x.pt from Hugging Face repo nope/nope:
                     An error happened while trying to locate the file on the Hub and we cannot find
                     the requested files in ...  (# elapsed 0.00s —— 不触网、不挂起)
```

`/root/.cache/falcon_weights/` 里还有 `falcon.pt`/`falcon-big.pt`/`frcnn_128_512.backbone{,.big}.pth.tar`，它们是**同一 inode 的硬链**（`ls -l` 显示 link count 3、同一时间戳 10-21 06:36）——Studio 不用它们，仓库里也没有引用（recon §2 提到的"软链到 group-shared"实际是硬链）。

### 4.5 实测的边界与口径声明

- 所有推理都是 **CPU**（`CUDA_VISIBLE_DEVICES=` 置空），**未测 A800 上的 CUDA 单窗时延**（→ GAP-4）。STATUS 里 "A800 单窗 small 47.1 ms / big 56.5 ms" 不是本报告的测量。
- 合成输入是"220 Hz + 660 Hz 正弦 + 高斯噪声"，不是真实语音 → 输出的绝对能量很小（rms 3e-3~4e-3），**§4.2(2)(4) 的相对比值（rel%、×中位数）要按"近静音输入"读**，别当成语音上的绝对结论。
- 一致性实验对训练侧做了 timm 打桩（§4.1 末）与 `torch.set_num_threads(1)`；两侧在同一进程、同一线程数下比较，是 apples-to-apples。

---

## 5. 坑与反直觉点

1. **没有自定义 pydantic 请求模型，文档界面关闭** —— `docs_url=None, redoc_url=None`（server.py:253），`server.py` 全文无 `pydantic`。任何"看 `/docs` 拿契约"的做法都会失败；契约只能读 `decode_request`（server.py:65-97）。**校验错误信息是中文散文**，机器不可解析（只有 WS 的 `type:"error"` 是结构化的）。
2. **`/api/load` 是异步的、立即返回**（server.py:204-205 起线程），但它返回的是**加载开始前/刚开始时**的 status，`ready` 还是 `false`。前端因此要轮询 `/api/status` 最多 180 s（app.js:666）。`test_checkpoint_cli.py:110-112` 靠 `load_thread.join(5)` 才拿到最终状态。
3. **0.5 s hop = 12.5 帧，永远对不齐**。`HOP=8000` 样本、`1 帧 = 640` 样本 → 12.5。前端用"最近邻 + ±0.16 s 容差"取整（app.js:587-593），因此**相邻窗的视频时间轴相差半帧（20 ms），取整方向固定不变**（`findIndex`/`while` 单调推进），实测视频偏移 12 帧造成 rel=16% 的输出变化（§4.2(3)）。任何"音频视频严格同期"的假设在这里不成立。
4. **该次 CPU 测试的推理耗时超过 hop 预算**。4.1 的 `rtf` 是 `inference_ms/500`，本机 CPU 实测 1.64–5.18（server.py:326）。跳过时前端置 `reset=True`（app.js:637），于是**没有 crossfade**、块首要 5 ms 淡入（app.js:813-817），并提示"已跳过 N 个旧窗口"（app.js:640）。所以 CPU 上听到的是断续的、带淡入的拼接——不是 bug，是设计。
5. **GIL 不是瓶颈，锁才是**。推理跑在 `asyncio.to_thread`（server.py:322），torch 算子大量释放 GIL；真正的串行化来自 `Engine.infer` 的非阻塞锁（server.py:222-223）+ `engine.busy`（server.py:320）。**两个标签页会互相抢锁**，后到的那个每次都被拒（收到 error 帧），不会排队也不会崩。
6. **`OutputSlicer` 是"每 WS 连接一份"**（server.py:302），不是全局。所以两个标签页各自 crossfade、各自 `last_endpoint`；`engine.busy` 只保证不并行算。**换成多 worker 部署会直接破坏这一点**（每个 worker 有自己的 slicer、自己的模型、4×内存）——这也解释了为什么代码里写死单进程。
7. **`models/` 不存在**。`server.py:195` 的文案 `尚未安装 {name} 权重：models/{checkpoint}` 指向一个**从未存在**的目录；实际权重只来自 HF cache 或 `--checkpoint`。排查"权重没装"时别去找 `studio/models/`。
8. **`variants[].available` 不看离线状态**（falcon_runtime.py:63-70）：只要装了 `huggingface_hub` 就返回 `"huggingface"`，UI 于是把 Falcon-Big 显示为可加载；真按下去才在 `resolve_checkpoint` 里 0.00 s 失败（§4.4）。`available=False` 只在**没装 huggingface_hub** 时出现。
9. **没有任何写盘**（除了用户显式指定的输出文件）。`server.py` 只有 `Image.open(io.BytesIO(...))`（:88）、`urllib.request.urlopen`（:407）、`webbrowser.open`（:409）；`falcon_runtime.py` 只 `open(path,"rb")` 读（:47）；`infer_file.py` 的 `wave.open(...,"wb")`（:43）是唯一写文件的地方，且路径由 `--out` 给定。**没有临时文件，因此没有清理逻辑**——也就没有"临时文件泄漏"这个坑，反过来说**也没有崩溃后的现场留存**。
10. **上传校验的真实边界**：单条 WS 消息 2 MB（server.py:30, 306-308）**先于**单帧 40000 字符（:85）生效——50 个合法上限的帧（40 KB/帧）就已经 2 MB+，实测 50×38.7 KB 直接断连（`/tmp/w1studio_08/wsclose.py`）。正常一窗 195.7 KB（166.7 KB 音频 + 28.6 KB 帧）。`/api/load` 的 1 KB 上限（:279）意味着**不能通过 body 传路径列表**。
11. **`StaticFiles(..., check_dir=False)`**（server.py:344-345）：`web/` 缺失时服务照常启动，`/` 之外全部 404/500，没有任何启动期报错。
12. **训练侧 `look2hear` 在本机 import 不了**（`timm → torchvision → torch._dynamo → cProfile` 崩溃，因为 `/tmp/profile.py` 之类同名模块污染 `sys.path`）。任何"在同一进程里对比 Studio 与训练侧"的脚本都要先给 `timm.models.layers` 打桩（§4.1），否则 `import look2hear` 直接崩。这是**环境**坑，不是代码坑，但会让 `tools/repro/*` 的部分脚本表现得很怪。
13. **`NOTICE.md` 与代码不一致**：NOTICE.md:7 写 "VIDEO mode, **a single face**, CPU WebAssembly"，而 `video-worker.js:34` 是 **`numFaces: 2`**（注释说明是为了区分"回头的目标"和"第二个人"）。许可证/说明文件未跟上代码。
14. **README 的 QA 端口写错**：README.md:193 说 `FALCON_QA_PORT=8766 python tests/serve_browser.py`，而 `serve_browser.py:14` 的默认值是 **7866**（`int(os.environ.get('FALCON_QA_PORT', 7866))`）。照 README 写 `8766` 能用（因为显式设了环境变量），但默认端口文档错了。
15. **`requirements-lock.txt` 优先却不被校验**：bootstrap 用锁文件（bootstrap.sh:146-150），`check_environment.py` 只校验 `requirements.txt`（:24）。版本漂移警告可能指向一套包、实际装的另一套。
16. **`uvicorn` 启动参数只有 4 个**（server.py:414-415）：`ws_max_size=2 MB`、`ws_max_queue=1`、`log_level="info"`、host/port。**没有 `workers`、没有 `timeout_keep_alive`、没有 `proxy_headers`、没有 TLS**。`ws_max_queue=1` 意味着**客户端连发两个窗口、服务端没读完时，第二个可能被丢**——配合前端"一次只有一个 `inFlight`"（app.js:619）刚好安全。
17. **端口是"扫描"出来的**：无 `--port` 时从 7860 试到 7899（server.py:386-396），先 `bind` 试探再交给 uvicorn ——**存在 TOCTOU 竞态**（bind 成功后关闭、uvicorn 再绑），并发启动两个实例时可能都拿到同一个端口然后其一失败。
18. **`--host` 只接受 `127.0.0.1`/`localhost`**（server.py:354 的 `choices`），局域网访问要改代码；`TrustedHostMiddleware` 白名单里还留着 `testserver`（server.py:255），属于测试专用值进了生产配置。
19. **`checkpoint_sha256` 的语义分两种**：本地/已缓存 → 真实哈希；`hf://` 未下载 → `None`（`test_checkpoint_cli.py:141` 断言了这点）。做"权重指纹核对"时别把 `None` 当失败。
20. **模型切换会重建整个 runtime**（server.py:202 `self.runtime = None`），旧模型的显存/内存靠 GC 回收；`load_thread` 是 daemon，进程退出时不等待。切换期间 `ready=false`，前端要等 `/api/status` 转真。

---

## 6. 与既有文档的冲突 / 更正

| # | 旧说法（出处） | 实际情况（证据） |
|---|---|---|
| 1 | `checkpoints/` 有 **83 个** ckpt 文件（本轮任务书） | **88 个**（2026-11-08 实测；每跑一次全量测试 +1），每个 5269 B，合计 640 KB；mtime 跨 10-21/10-24/10-30/11-08 四批，`state_dict` 只有一个标量 `audio_model.w`；`git ls-files checkpoints`=0、被 `.gitignore:24 *.ckpt` 忽略（§3.4） |
| 2 | recon §2"`/root/.cache/falcon_weights/*`（**软链**到 group-shared）" | `ls -l` 显示 link count=3 的**硬链**（同 inode、同 mtime），不是软链；且 Studio **不用**这个目录（§4.4） |
| 3 | README 流水线表"keeping **80 ms of right context**"（README.md:95） | 80 ms 是**发射几何**（输出块止于 `end-1280`），**不是**模型感受野。扰动实测：任一输入样本影响全部 32000 个输出样本，窗内全局耦合（§4.2(5)） |
| 4 | `docs/STATUS.md:50`"单遍 lookahead 实测 57 帧≈456 ms；Studio 用 2 s 窗/0.5 s hop/80 ms guard 重算" | **方向一致，口径要补**：本报告独立复现了"窗级因果"（跨窗无状态）与 cLN/全局注意力造成的窗内全局耦合。456 ms 只是一遍 AudioBlock 的几何值，整链是 **904 ms**，as-shipped 依赖到窗尾；§4.2(5) 的 580 ms/80 ms 是窗内可用余量，不是 lookahead |
| 5 | `docs/STATUS.md:34`"训练侧 vs Studio 同输入 max\|Δ\|=0" | **成立且更强**：不只输出 `max|Δ|=0`，**加载后的 525+137 个 state_dict 张量逐位相同**，输出 sha256 相同，两组独立输入都是 0（§4.1） |
| 6 | `docs/STATUS.md:64-65`"Studio Python 51 passed/2 skipped；Node 8/8" | **复现**：51/2（18.74 s）；`CUDA_VISIBLE_DEVICES=` 时 53/0（42.39 s）；Node 8/8（≈4.0 s）（§4.3） |
| 7 | `studio/falcon_model/NOTICE.md:7`"a single face" | `video-worker.js:34` 是 `numFaces: 2`（§5.13） |
| 8 | `README.md:193`"`FALCON_QA_PORT=8766`" | `serve_browser.py:14` 默认 **7866**（§5.14） |
| 9 | `server.py:195` 错误文案"`models/{checkpoint}`" | `studio/models/` 目录不存在；权重只来自 HF cache / `--checkpoint`（§5.7） |
| 10 | PROVENANCE.md:20-22"On Apple Metal (MPS), STFT and inverse STFT run on the CPU" | 成立；补一句：这是 `studio/falcon_model/models/Falcon.py` 相对训练侧**唯一的净增逻辑**（5 行），CPU/CUDA 上是 no-op（§4.1） |
| 11 | 隐含假设"`separated[0,0]` 是双路分离的第一路" | `n_src=1`（configs/Falcon.yml:12），`[0,0]` 是**唯一目标源**；Studio 做目标说话者提取（§2.2） |
| 12 | 隐含假设"Studio 与训练侧同构、Big 亦然" | `SpeakerContrastiveV2`/`SISNRRepel` **只存在于 studio**（look2hear 无同名类、无引用），Big 无法用训练侧交叉验证（§4.1、GAP-1） |

---

## 7. GAP LIST（可被下一轮直接认领）

1. **GAP-1｜Falcon-Big 的数值一致性无法验证**。`studio/falcon_model/models/Speaker_Contrastive_v2.py` + `SISNRRepel.py` 在 `look2hear/` 无对应实现（grep 为空，`git log` 只有 `a7af472` 一次性带入）。要做的事：从上游 `JusperLee/Falcon` 取 `SpeakerContrastiveV2` 的原始文件做 sha256 对比；或在 `look2hear` 补一个等价实现再跑 `falcon-big.pt` 的同输入 `max|Δ|`。**验收**：`falcon-big.pt` 上 `max|Δ|=0` 或给出差异清单。
2. **GAP-2｜真实语音上的 lookahead / 边界瞬态测量**。本报告 §4.2(4)(5) 用的是合成音（220/660 Hz + 噪声），输出能量 3e-3 量级，相对比值不稳。要做的事：用 `LRS2-2Mix` 的 20 条真实混合 + 真嘴部帧，重跑 edge.py 与 receptive.py，给出"窗尾瞬态宽度"的分布，判断 `GUARD=1280` 是过保守还是不够。
3. **GAP-3｜12.5 帧 hop 的实际影响**。已证"视频偏移 12 帧 → rel 16%"，但**未证**"最近邻取整导致的实际偏移序列"（相邻窗 ±0.5 帧交替？单调累积？）以及它对 SI-SNR 的影响。要做的事：从 app.js:587-593 的取整逻辑推出偏移序列，再用真实数据测两种取整策略（floor vs round）的指标差。
4. **GAP-4｜A800 上的 CUDA 单窗时延未测**。本报告全 CPU；`rtf` 1.64–5.18 只是这台机器的 CPU。要做的事：`CUDA_VISIBLE_DEVICES=0` 跑 stream.py，记录 `inference_ms`/`warmup_ms`/显存，验证 `torch.set_num_threads` 与 `--device auto` 的取舍。
5. **GAP-5｜并发实测缺失**。§2.2/§5.5-5.6 的并发结论是代码级推理。要做的事：两个 TestClient 同时连 `/ws` 各发 3 窗，统计 error 帧比例、`active_connections` 行为、以及"有活跃会话时 `/api/load` 409"是否真的会触发。
6. **GAP-6｜`spectrum-worker` 的列数/耗时与主线程关系**。实测 1600 列 / 9600 FFT / 668 ms（check_spectrum.mjs），但**未测**它在 Worker 里对主线程帧率的影响，也未测 `recordingView` 在 5 min（480 万样本）录音下的内存与耗时。
7. **GAP-7｜浏览器真机 MediaPipe 未跑**。`tests/browser/test.js` 需要真人开 Chrome（`serve_browser.py` 的 `__test` 挂载），本轮无法自动化。要做的事：在 macOS + Chrome 上跑一次，记录 `faceFrames/meanDetectionMs/steadyP95Ms`，与 `check_mouth_tracking` 的纯几何断言互补。
8. **GAP-8｜`count_download` 在真实网络/镜像下的行为**。只测了单测桩。要做的事：`HF_ENDPOINT=https://hf-mirror.com` 下冷启动一次，确认 HEAD `config.yaml` 不会拖慢加载、失败不刷日志。
9. **GAP-9｜`audio-inspector` 的 Mixture/目标对齐是否逐样本精确**。前端用 `inputSpan(result.start_sample, kept.length, received)` 从 ring 里取同段混合（app.js:748），但 `received.audio` 是**发送时的快照**、ring 是活缓冲，`LEFT_CONTEXT` 偏移与 `sessionEpoch` 之外的重叠区没有实测验证。要做的事：导出 WAV 逐样本比对。
10. **GAP-10｜`space/` 与 `studio/web` 的 i18n 字典是否同源**。STATUS 说 Space 组装后 226 条、Studio 368 条。要做的事：diff 两套 `i18n-*.js`，确认是否有漂移（这决定 `space/build.py` 是否能安全复用 Studio 前端）。

---

## 8. 证据清单

**代码（远端 `/root/code/github_repos/Falcon-fork`，HEAD `6c1d5b7`）**

- `studio/server.py`（419 行）— 全部 8 节的主要证据：常量 :28-31、`decode_request` :65-97、`OutputSlicer` :100-119、`Engine` :122-231、`same_origin` :234-238、`create_app` :241-346、CLI :352-379、`main` :382-415
- `studio/falcon_runtime.py`（300 行）— :32-37 VARIANTS、:43-52 sha256、:63-95 权重解析、:98-114 count_download、:117-128 pick_device、:131-140 预处理、:170-240 load、:242-245 warmup、:255-276 status、:278-300 forward/infer
- `studio/infer_file.py`（105 行）— :62-66 预处理、:69-82 extract、:85-101 CLI
- `studio/falcon_model/PROVENANCE.md`（42 行）— :13-22 改动清单、:33-34 逐位一致声明、:36-39 导出说明
- `studio/falcon_model/models/Falcon.py` — :1117-1128 Encoder 的 MPS 分支、:1190-1204 Decoder 的 MPS 分支、:1488-1507 forward（`n_src=1` 语义）
- `studio/falcon_model/videomodels/frcnn_videomodel.py`（198 行）— :76-86 forward、:87-110 `init_from`、:111-129 `train()` 返回 self、:155 `update_frcnn_parameter`
- `studio/configs/Falcon.yml` / `FalconBig.yml` — :10-15 架构与 `n_src=1`、:27-36 enc_dec、:38-72 audio_params、:74-95 video_params
- `studio/web/app.js`（1223 行）— :6-10 常量、:15 环形缓冲、:531-537 addFrame、:577-596 frameWindow、:618-654 maybeInfer、:715-783 handleResult、:784-835 playSamples/stopMonitoring、:901-913 drain
- `studio/web/video-worker.js`（166 行）— :29-37 MediaPipe 初始化、:53-103 processFrame、:88-96 96×96 灰度 JPEG、:104-120 pump
- `studio/web/mouth-tracker.js`（149 行）— :2-11 索引与策略、:38-54 mouthGeometry、:83-109 updateMouthTrack、:111-149 reacquire/relock
- `studio/web/i18n.js`（124 行）— :1-9 字典与属性白名单、:19-33 反向表与模板、:44-59 translateMessage、:88-124 renderLanguage/setLanguage
- `studio/web/index.html`（64 行）— :7 `data-i18n`、:28 语言选择、:62 唯一 script 入口
- `studio/web/vendor/mediapipe/manifest.json` — :2-18 版本 1.0.1 + 全部 sha256；`NOTICE.md:3-8` 来源与"single face"表述
- `studio/scripts/{bootstrap.sh,doctor.sh,launch.sh,install-blackhole.sh,check_environment.py}` — §1.4 表中行号
- `studio/requirements.txt:7-15`、`requirements-dev.txt:1-3`、`requirements-lock.txt`、`studio/.gitignore:1-14`、根 `.gitignore:24`
- `studio/tests/{test_server,test_checkpoint_cli,test_device,test_infer_file,test_weights}.py`、`serve_browser.py:11-14`
- `studio/tests/{check_capture_dsp.cjs,check_i18n.mjs,check_mouth_tracking.mjs,check_playback_resampler.mjs,check_spectrum.mjs,check_tracking_ui.cjs}`、`studio/web/{capture-worklet,video-worker}.test.cjs`、`studio/tests/browser/test.js:39-60`
- `look2hear/models/Falcon.py:8`（`timm` 导入）、`look2hear/models/__init__.py:1-2`（只导出 Falcon / FalconSpeakerContrastive）
- `configs/Falcon.yml:10`、`configs/LRS2_2Mix_Paper_v2.yml:26`（训练侧只用 FalconSpeakerContrastive）

**文档（仅用于对照，结论均已回源码复核）**

- `docs/STATUS.md:34`（一致性）、`:50`（窗口级因果/456 ms）、`:64-65`（51/2、8/8）
- `studio/README.md`（:24-35 快速开始、:36-50 模型、:51-61 唇部跟踪、:88-104 流水线、:105-131 CLI、:185-193 测试、:168-183 目录）
- `studio/README_zh.md`（同结构中文版）

**实测脚本与输出**（`dev_a800_2gpus:/tmp/w1studio_08/`，仓库内零写入）

| 脚本 | 目的 | 关键输出 |
|---|---|---|
| `consistency.py` | Studio vs look2hear 权重与输出 | `max|Δ|=0.0`、`torch.equal=True`、sha256 `a458a6f8…` 两侧相同、params 11,724,385 |
| `stream.py` | 真 Engine + 真 WS 6 窗 | §3.3 时间线表、tiling=True、加载 16.3 s、warmup 1990.9 ms、446.7 KB/窗 |
| `drift.py` | 窗对齐/视频偏移敏感性 | D1 rel 103%（发射区 80%）、D2 16%/11%、D3 确定性 True |
| `edge.py` | 窗尾瞬态剖面 | 末 320 样本 5.39× 中位数、GUARD 起点 1.68× |
| `receptive.py` | 扰动感受野 | 任一输入 → 全窗输出变化、窗尾 80 ms 可获 1280 样本未来 |
| `api.py` | 路由/错误码/静态资源 | §2.4 实测块 |
| `wsclose.py` | WS 尺寸上限 | 195.7 KB 正常、2.10 MB 断连 |

**命令清单（可复现）**

```bash
ssh dev_a800_2gpus 'cd /root/code/github_repos/Falcon-fork/studio && \
  PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tests -q -p no:cacheprovider -rs'
ssh dev_a800_2gpus 'cd /root/code/github_repos/Falcon-fork/studio && \
  for t in tests/check_*.mjs tests/check_*.cjs web/*.test.cjs; do node "$t"; done'
ssh dev_a800_2gpus 'cd /tmp/w1studio_08 && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=/root/code/github_repos/Falcon-fork/studio:/root/code/github_repos/Falcon-fork \
  /root/miniforge3/envs/falcon/bin/python consistency.py'
ssh dev_a800_2gpus 'cd /tmp/w1studio_08 && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=/root/code/github_repos/Falcon-fork/studio \
  /root/miniforge3/envs/falcon/bin/python stream.py'
ssh dev_a800_2gpus 'cd /root/code/github_repos/Falcon-fork && diff -rq studio/falcon_model look2hear | grep -v __pycache__'
ssh dev_a800_2gpus 'cd /root/code/github_repos/Falcon-fork && ls checkpoints | wc -l && md5sum checkpoints/*.ckpt | awk "{print \$1}" | sort -u | wc -l'
```
