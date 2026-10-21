# 06 Studio 后端：本地推理服务 + 运行时 + 模型推理拷贝

> 范围：`studio/{server.py, falcon_runtime.py, infer_file.py}`、`studio/falcon_model/**`、
> `studio/configs/{Falcon.yml, FalconBig.yml}`、`studio/requirements*.txt`、`studio/README.md` 运行时章节。
> 证据行号以 `main`（commit `7576854`）为准；标注约定：`[代码]`=读源码确认，`[推断]`=合理推断，
> `[验证]`=本轮实际运行命令确认（命令与输出见 §8，原始日志 `docs/verification/runs/studio-backend-verify.log`）。
> 验证环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128 + fastapi 0.156 + pytest 9.1），
> **无 Pillow**（协议层验证用 `docs/verification/scripts/pil_stub` 最小替身，真实图片解码未验证）、
> **无 Falcon 权重且禁止下载**（运行时验证用“随机权重 + 自造 ckpt”完成，见 §8）。

---

## 1. 子系统职责与总体位置

**一句话结论**：Studio 后端是一个**只绑定 loopback、单进程、单模型对（audio+video）、串行推理**的本地服务：
`server.py` 负责 HTTP/WS 协议、时间轴切片与安全边界；`falcon_runtime.py` 负责“配置→模型→严格加载权重→
预处理→2 s 窗口前向”的纯 torch 运行时；`infer_file.py` 复用同一运行时做离线整段推理；
`falcon_model/` 是训练侧模型代码的推理拷贝（有差异，见 §4.8）；`configs/` 是两份模型超参。

### 1.1 边界与上下游

| 方向 | 对象 | 契约 | 证据 |
|---|---|---|---|
| 上游（输入） | 浏览器前端（07 域） | `GET /api/status`、`POST /api/load`、`WS /ws`；2 s 音频（32000×f32）+ 50 帧 96×96 灰度嘴部 | `studio/server.py:35-67,233-276`；`studio/web/app.js:564-600` |
| 下游（计算） | `falcon_model` 模型 + torch | `FRCNNVideoModel`(B,1,T,**88,88**→B,512,T；96×96 是协议/采集尺寸，送入模型前已由 `preprocess_mouth` 裁到 88×88) + `FalconSpeakerContrastive`/`SpeakerContrastiveV2`(audio[B,32000], mouth_emb→[B,1,32000]) | §3.5 形状链 [验证] |
| 权重来源 | Hugging Face Hub（首次下载，缓存 `~/.cache/huggingface`） | `JusperLee/Falcon` → `falcon.pt`；`JusperLee/Falcon-Big` → `falcon-big.pt` | `studio/falcon_runtime.py:31-36,58-78` |
| 旁路 | `infer_file.py`（离线 CLI） | 整段 WAV + mouth.npy → 目标 WAV | `studio/infer_file.py:1-12,69-101` |
| 无持久化 | 服务端不写音视频到磁盘 | 只有内存中的预测窗口；录音在浏览器内存 | `studio/server.py:1`；`studio/README.md:20` |

### 1.2 进程 / 线程模型

- 进程：唯一 Python 进程，`uvicorn` 单 worker（`studio/server.py:332-333`：`ws_max_size=2_000_000, ws_max_queue=1`）。
- 全局唯一 `Engine` 实例：`create_app()` 里创建一个，`app.state.engine = engine`（`server.py:182-195`）。
- 线程：`Engine.start_load()` 起 1 个 daemon 加载线程（`server.py:145`）；推理在 WS 协程里用
  `asyncio.to_thread(engine.infer, ...)` 丢到默认线程池（`server.py:260`）。
- 并发控制：全局 `infer_lock`（`threading.Lock`，非阻塞获取）保证**整个服务同时只有一次前向**（`server.py:102,163-172`）；
  `load_lock` 保护加载状态（`server.py:101,130-146`）。多个 WS 连接可同时存在（`active_connections`，`server.py:96,239,276`），
  但切模型前必须所有连接关闭（`server.py:137-138`）。
- 每个 WS 连接独立 `OutputSlicer`（`server.py:240`），因此新会话从“非连续”开始（不与旧会话交叠）。

### 1.3 关键数字速查（全部 [代码]）

| 量 | 值 | 单位 | 证据 |
|---|---|---|---|
| 采样率 | 16000 | Hz | `server.py:28`；`falcon_runtime.py:24` |
| 视频帧率 | 25 | fps | `server.py:28`；`falcon_runtime.py:25` |
| 窗口音频 | 32000 = 2.0 s | 采样 | `server.py:28`；`falcon_runtime.py:26` |
| 窗口帧数 | 50 = 2.0 s×25 | 帧 | `server.py:51-52`；`falcon_runtime.py:27` |
| hop | 8000 = 0.5 s | 采样 | `server.py:28` |
| 右上下文 GUARD | 1280 = 80 ms | 采样 | `server.py:29,111,267` |
| 交叉淡化 FADE | 320 = 20 ms | 采样 | `server.py:29,84-86` |
| 每次回传音频 | 8000 = 0.5 s | 采样| `server.py:80-81,262-264` |
| 源音频字节 | 128000 = 32000×4 | 字节 | `server.py:43-46` |
| 单条 WS 消息上限 | 2 000 000 | 字符 | `server.py:30,244` |
| 时间戳上限 | 2^53−1（JS 精确整数） | 采样 | `server.py:31,65,255` |
| 前端左上下文 | 22720 = 2 s−0.5 s−80 ms = 1.42 s | 采样 | `web/app.js:10,593` |
| 首窗所需真实输入 | 9280 = hop+guard = 0.58 s | 采样 | `web/app.js:567-570` |

---

## 2. 文件清单与逐文件解析

### 2.1 `studio/server.py`（337 行）

职责：FastAPI 应用 + WebSocket + 引擎状态 + 音频时间轴切片 + 本地同源安全边界。**不含模型代码**，
模型类通过 `falcon_runtime` 间接使用（`server.py:106,121,150`）。

#### 2.1.1 常量（`server.py:1-31`）
`SR, FPS, WINDOW, HOP = 16000, 25, 32000, 8000`；`GUARD, FADE = 1280, 320`；
`MAX_MESSAGE_BYTES = 2_000_000`；`MAX_SAFE_INTEGER = 2**53 - 1`。

#### 2.1.2 `decode_request(data: dict) -> (audio, frames, endpoint)`（`server.py:35-67`）
把 WS JSON 反序列化为 `(np.float32[32000], np.uint8[50,96,96], int)`，**所有字段强校验**（细节表见 §5.3）：
- 只接受 `type=="infer"`；`window_seconds`/`hop_seconds` 若出现必须严格等于 2/0.5（`server.py:36-39`）；
- `reset` 必须是 `bool`（`server.py:40-41`）；
- `audio` 必须是 base64 str 且长度恰为 `4*((32000*4+2)//3)=170668` 字符，解码为 `<f4`、形状 `(32000,)`、
  全部有限、`max|amp| ≤ 4`（`server.py:42-50`）；
- `frames` 必须是长度恰为 50 的 list，每项 base64 str ≤40000 字符，解码后必须是 **96×96 的 JPEG 或 PNG**，
  再 `convert("L")` 成 uint8 灰度（`server.py:51-63`）；
- `end_sample`：int（bool 不算）、`32000 ≤ end_sample ≤ 2^53−1`，缺省 32000（`server.py:64-67`）。

#### 2.1.3 `OutputSlicer`（`server.py:70-89`）
一个 WS 连接一个实例；把 32000 点预测切成 8000 点回传块并做相邻窗交叉淡化、`[-1,1]` 裁剪与越界计数。
语义与时序见 §5.5（本轮已用数值脚本逐点验证）。

#### 2.1.4 `Engine`（`server.py:92-172`）
服务端状态机 + 模型变体管理 + 串行推理（§5.6）。
- `status()`（`:105-116`）合成 `/api/status` 返回体（模型未加载时只有基础字段；加载后合并 `FalconRuntime.status()`）；
- `variants()`（`:118-124`）返回 `[{id,name,available,source,checkpoint,repo}]`；
- `start_load(variant)`（`:126-146`）校验并起加载线程；
- `_load()`（`:148-158`）构造 `FalconRuntime`、`load(device)`，异常写入 `self.error`；
- `infer(audio, frames)`（`:160-172`）非阻塞锁 + 计时，返回 `(prediction, inference_ms)`。

#### 2.1.5 `same_origin(origin, host)`（`server.py:175-179`）
`origin` 缺失 → 放行（非浏览器客户端）；否则要求 scheme ∈ {http,https}、`netloc == Host 头`、
且 hostname ∈ {localhost, 127.0.0.1, ::1}。安全边界与实测见 §4.6。

#### 2.1.6 `create_app(engine=None, autoload=True)`（`server.py:182-284`）
- lifespan：`autoload` 时 `engine.start_load()`，`ValueError/RuntimeError` 写入 `engine.error`（`:185-192`）；
- `TrustedHostMiddleware(allowed_hosts=["127.0.0.1","localhost","[::1]","testserver"])`（`:196`）；
- HTTP 中间件：先 `same_origin` 校验（`:200-202`），再统一加 `X-Content-Type-Options/Referrer-Policy/
  Permissions-Policy/Cache-Control/CSP` 安全头（`:204-210`）；
- 路由：`GET /api/status`（`:213-215`）、`POST /api/load`（`:217-231`）、`WS /ws`（`:233-276`）、
  `GET /` 返回 `web/index.html`（`:278-280`），`/static` 与 `/` 都挂 `web/` 静态目录（`:282-283`）。
- 模块级 `app = create_app()`（`:287`）：`uvicorn server:app` 也可启动（此时 `FALCON_DEVICE` 生效，`autoload=True`）。

#### 2.1.7 `main()`（`server.py:290-333`）
`--host` 只能 `127.0.0.1|localhost`（`:292`）；`--device auto|cuda|mps|cpu`（`:294`）；`--model` 接 HF 短链或 small/big
（`:296,299-303`）；未指定端口时在 **7860–7899** 找空闲端口（`:304-316`）；`--no-browser` 关闭自动开浏览器；
用 `urllib` 轮询 `/api/status` 直到 `app=="falcon-studio"` 再打开浏览器（`:320-331`）；`uvicorn.run(..., ws_max_size=MAX_MESSAGE_BYTES, ws_max_queue=1)`（`:332-333`）。

### 2.2 `studio/falcon_runtime.py`（259 行）

职责：权重解析/下载、模型构建、严格加载、设备选择、嘴部预处理、窗口前向。**无服务端依赖，可单独作 Python API。**

| 符号 | 行号 | 说明 |
|---|---|---|
| `SAMPLE_RATE=16000` `VIDEO_FPS=25` `WINDOW_SAMPLES=32000` `WINDOW_FRAMES=50` | 24-27 | 与 server 常量一致 |
| `VARIANTS` | 31-36 | 见 §4.4 变体表 |
| `MODEL_CLASSES` | 37 | `{"FalconSpeakerContrastive": ..., "SpeakerContrastiveV2": ...}` |
| `variant_for(name)` | 40-45 | 接受 `small`/`big` 或两个 HF 短链；其他 → `ValueError`（错误信息列出两个 repo） |
| `weights_source(variant)` | 48-55 | 返回 `"cached"`/`"huggingface"`/`None`；**只查本地缓存，不联网**（`try_to_load_from_cache`） |
| `resolve_checkpoint(variant, checkpoint)` | 58-78 | 权重解析顺序见 §4.4 |
| `pick_device(requested)` | 81-92 | `auto` → CUDA > MPS > CPU；显式 cuda/mps 不可用时报错 |
| `preprocess_mouth(frames)` | 95-104 | `uint8(50,96,96)` → `float32(1,1,50,88,88)`；96→88 中心裁剪、/255、z-score（**不是 [-1,1]**，§4.5） |
| `FalconRuntime.__init__` | 110-131 | variant 校验、config 路径、未加载状态（`device="unloaded"`, `ready=False`） |
| `.load(device="auto")` | 133-200 | 权重解析→配置→构建→`torch.load(weights_only=True)`→前缀拆分→**strict** 加载→统计 keys/参数→`torch.set_num_threads`→`eval()`/`requires_grad_(False)`→`pick_device`→warmup→ready（顺序与源码 `:142-196` 一致） |
| `._warmup()` | 202-205 | 两个模型 `.to(device)` 后跑一次真前向：全零音频 + 全 107 灰度帧（§4.5） |
| `.from_pretrained(name, device)` | 207-213 | `variant_for` + `load`；**无 checkpoint 参数**（自定义权重只能走构造器/`infer_file --checkpoint`） |
| `.status()` | 215-235 | 返回 §5.1 中的运行时字段（不取推理锁，状态查询不阻塞推理） |
| `._forward(audio, frames)` | 237-246 | `preprocess_mouth`→video→audio→`separated[0,0]`；校验输出 `(32000,)` 且有限 |
| `.infer(audio, frames)` | 248-259 | 输入校验 + 推理锁 + 记 `last_inference_ms` |

### 2.3 `studio/infer_file.py`（105 行）

离线整段推理：**一次前向跑完整段**（不是滑窗），复用 `FalconRuntime` 的模型与设备。
- `read_wav`（`:28-38`）：仅接受 mono 16 kHz WAV；16-bit PCM→`/32768`，32-bit float 直读，其他位宽退出；
- `write_wav`（`:41-44`）：`clip(-1,1)*32767` → 16-bit PCM；
- `read_mouth`（`:47-59`）：`.npy`/`.npz`（取第一个数组）/`.pt`（`weights_only=True`），`squeeze` 成 `(T,H,W)` 且 `H,W ≥ 88`；
- `preprocess`（`:62-66`）：任意尺寸中心裁剪 88×88 → `/255` → `(x-0.421)/0.165`（与运行时同分布，但允许 H,W>96）；
- `extract`（`:69-82`）：`frames=min(len(mouth), len(audio)*25//16000)`，要求 ≥25 帧（1 s），
  `samples=frames*640`；audio 截到 `samples`，mouth 截到 `frames`；直接调 `runtime.video_model` +
  `runtime.audio_model`（**绕过 `_forward` 的 2 s 形状校验**），返回 `output[:samples]`；
- `main`（`:85-101`）：`--audio/--mouth/--out/[--model/--checkpoint/--device]`；音视频时长差 >1 帧时提示“trim”；
  权重默认仍走 HF 短链（首次下载），`--checkpoint` 可本地或 `hf://`。
- 语义注意：整段前向与“2 s 滑动窗口”结果**不等价**（归一化统计/注意力覆盖整段），文档 `infer_file.py:12`
  声称“matching the evaluation protocol”，评测侧是否整段前向需由 04/05 域确认（Q3）。

### 2.4 `studio/falcon_model/`（16 个文件 = 15 个 `.py` 共 **4035 行** + `PROVENANCE.md` 42 行；合计 4077 行）

| 文件 | 行数 | 角色 | 与 `look2hear/` 对应关系（[验证] §8-A） |
|---|---|---|---|
| `models/Falcon.py` | 1496 | `ConvNormAct/ConvActNorm/DualPathRNN/MHSA2D/GlobalAttention/AudioBlock/VideoBlock/Separator/STFTEncoder/STFTDecoder/MaskGenerator/Falcon` | `look2hear/models/Falcon.py`(1487) **+9 行**：DropPath import + Metal FFT bridge；其余为注释翻译 |
| `models/Falcon_Contrastive.py` | 122 | `AttentionSpeakerPool` + `FalconSpeakerContrastive`(small 架构) | 与 `look2hear/models/Falcon_Contrastive.py` **逐字节相同** |
| `models/SISNRRepel.py` | 1498 | `SISNRRepel`(big 架构底座) | **look2hear 无对应文件**；= Studio `Falcon.py` +2 行（2 个注释掉的 import）+ 类名替换 |
| `models/Speaker_Contrastive_v2.py` | 140 | `SpeakerContrastiveV2`(big 架构) | **look2hear 无对应文件**；= Studio `Falcon_Contrastive.py` +18 行模块 docstring + base 类改 `SISNRRepel` |
| `models/base_model.py` | 12 | 仅 `BaseModel.__init__` + `sample_rate()` | `look2hear/models/base_model.py`(102) 删掉训练/序列化 API（§4.8） |
| `drop_path.py` | 16 | timm `DropPath` 的无依赖替代 | 新文件（look2hear 从 `timm` import） |
| `layers/activations.py` | 67 | 激活 `get()` | 与 look2hear **逐字节相同** |
| `layers/normalizations.py` | 245 | 归一化 `get()` 等 | 仅注释中译（13+/13− = 26 行，净 0） |
| `videomodels/frcnn_videomodel.py` | 122 | 冻结唇读前端 `FRCNNVideoModel` | 去掉 `import sys; sys.path.append("../../")` 与 `resnet1D` import；CRLF→LF；加 5 行头注释；其余逐行相同 |
| `videomodels/resnet.py` | 140 | ResNet trunk | 与 look2hear **逐字节相同** |
| `videomodels/shufflenetv2.py` | 173 | 备用 backbone | 与 look2hear **逐字节相同** |
| 4×`__init__.py`（`falcon_model`/`models`/`layers`/`videomodels`） | 1 | 仅 docstring | 不导出符号（全部用子模块绝对/相对 import） |
| `PROVENANCE.md` | 42 | 来源声明 | 逐条核验见 §4.8：**1 条不成立（Big 两文件无训练侧源）、2 条依赖真实权重无法验证** |

### 2.5 `studio/configs/Falcon.yml`（171 行）与 `FalconBig.yml`（166 行）

- `Falcon.yml`：与仓库根 `configs/Falcon.yml` **逐字节相同**（[验证]，`diff` 0 行）——手写训练配置，
  含 `videonet/audionet/loss/training/optimizer/scheduler/datamodule/exp`；无 `train_conf` 包裹。
  模型超参：`audionet_name: FalconSpeakerContrastive`（`:10`），`n_src:1`（`:12`），`pretrained_vout_chan:512`（`:13`），
  `causal:true`（`:14`），`spk_embed_dim:64`（`:15`），`audio_bn_params.out_chan:112`（`:23`），
  `enc_dec_params.{win:256,hop_length:128,out_chan:112,kernel_size:3,stride:1,bias:false,causal:true}`（`:27-36`），
  `audio_params.{hid_chan:64,kernel_size:5,stride:2,upsampling_depth:4,repeats:2,shared:true,n_freqs:129}`（`:38-48`），
  `video_params.{hid_chan:32,...}`（`:74-89`），`fusion NeuroSCNetFusion k=4`（`:91-95`），
  `mask_generation {MaskGenerator, ReLU, RI_split:true}`（`:97-101`），`videonet FRCNNVideoModel(resnet,prelu,1.0,pretrain)`（`:2-8`）。
- `FalconBig.yml`：**一次训练运行被 `yaml.safe_dump` 落盘的完整配置**（键按字母序；含
  `datamodule` 里未展开的 `${DATA_ROOT}`、`main_args.exp_dir`、`options.conf_dir: configs/FalconBig.yml`、`training/optimizer/loss` 等），
  但运行时只读 `audionet`/`videonet`（`falcon_runtime.py:146-158`），多余键不影响。
  `audionet_name: SpeakerContrastiveV2`（`:88`）；`out_chan:256`（`:6,53`）；`hid_chan:96`（`:12`）；
  `layer_1.hid_chan:48`（`:19`）、`layer_2.hid_chan:96`（`:27`）；`spk_embed_dim:128`（`:68`）；`video hid_chan:64`（`:74`）。
- 两份配置都会被 `falcon_runtime.load()` 读取；`if "train_conf" in config` 分支（`:147-148`）兼容评测侧的包装格式。

### 2.6 `studio/requirements*.txt`

| 文件 | 内容 | 备注 |
|---|---|---|
| `requirements.txt`（11 行） | `torch>=2.14.1,<3`、`numpy>=2.4.6,<3`、`PyYAML>=6.0.3,<7`、`einops>=0.8.2,<1`、`fastapi>=0.142.2,<1`、`uvicorn[standard]>=0.54.0,<1`、`Pillow>=12.3.0,<13`、`huggingface_hub>=2.2.0,<3` | `requirements.txt:3-11` |
| `requirements-lock.txt`（110 行） | uv 编译的精确版本：`torch==2.14.1`(:79)、`numpy==2.4.6`(:54)、`fastapi==0.142.2`(:18)、`huggingface-hub==2.2.0`(:40)、`pillow==12.3.0`(:60)、`pyyaml==6.0.3`(:68)、`einops==0.8.2`(:16)、`uvicorn==0.54.0`(:103)；python-version 3.11（`:2`） | 版本与本文验证环境不同（torch 2.8/numpy 2.2/Pillow 缺失），见 Q7 |
| `requirements-dev.txt`（3 行） | `-r requirements.txt` + `pytest==9.1.1` + `httpx==0.28.1` | 测试依赖 |

运行时实际 import 第三方包的集合（`grep` 全 `studio/falcon_model` + 三个入口）只有
`torch/numpy/yaml/PIL/fastapi/uvicorn/huggingface_hub`；**`einops` 没有运行时 import**（只在 requirements 与
`scripts/check_environment.py:40` 出现）→ 死依赖（[代码]，P2）。

### 2.7 `studio/README.md` 运行时章节要点（与代码核对）

| README 说法 | 代码核对 | 证据 |
|---|---|---|
| 本地服务、`127.0.0.1:7860` 或 7860–7899、数据不落盘 | 一致（端口扫描 7860-7899） | `README.md:20`；`server.py:304-316` |
| 两个模型 small/big，首次从 HF 短链下载并缓存，离线可用 | 一致（`resolve_checkpoint` + `weights_source`） | `README.md:34-47`；`falcon_runtime.py:48-78` |
| 权重 strict 加载，不匹配报错、绝不放原始混合声 | 一致（`strict=True` + WS error 不回退） | `README.md:47`；`falcon_runtime.py:168-169`；`server.py:268-272` |
| 2 s 窗口 / 0.5 s hop / 80 ms 右上下文 / 交叉淡化 / 重采样 | 一致（服务端切 + 前端重采样，见 §5.5/§5.9） | `README.md:86-101`；`server.py:70-89`；`web/app.js:726-771` |
| 首输出 ~0.58 s 输入 + 1.42 s 静音左上下文 | 一致（`HOP+GUARD=9280`、`WINDOW-HOP-GUARD=22720`） | `README.md:97`；`web/app.js:10,567-570` |
| 稳态延迟 ~0.7–0.8 s；每 hop 重算整窗，非流式 | 一致（`OutputSlicer` 80 ms 保护 + 0.5 s hop） | `README.md:97`；`falcon_runtime.py:1-6`；`PROVENANCE.md:41-42` |
| M5 Pro 单窗 ~0.12 s（Metal）/ ~0.65 s（CPU） | **无法验证**（无 Falcon 权重、无 M5；本环境 A800 CPU 实测见 §8，不可比） | `README.md:97` |
| `python infer_file.py ... --checkpoint` | 一致（`--checkpoint` 本地/hf://） | `README.md:117-124`；`infer_file.py:91,98` |
| Python API `FalconRuntime.from_pretrained(...)` + `infer` | 一致（返回 `(32000,)` f32） | `README.md:126-137`；`falcon_runtime.py:207-213,248-259` |
| 只暴露三个 endpoint，同源本地页面 | 一致（另有 `/` 与 `/static` 静态路由） | `README.md:177`；`server.py:213-283` |

---

## 3. 关键调用链与数据流

### 3.1 启动 / 权重加载链（[代码]）

```
server.py main() :290-333
  ├─ variant_for(--model)                      falcon_runtime.py:40-45
  ├─ 端口选择 7860-7899                         server.py:304-316
  ├─ create_app(Engine(device, variant))        server.py:182-196,332
  │    └─ lifespan → engine.start_load()        server.py:185-192
  └─ uvicorn.run(..., ws_max_size=2e6)          server.py:332-333

Engine.start_load(variant) :126-146
  └─ Thread(target=Engine._load) :145
       └─ FalconRuntime(variant).load(device)  falcon_runtime.py:133-200
            1. resolve_checkpoint()            :142  → 本地 / hf:// / HF repo（hf_hub_download）
            2. is_file 校验                     :143-144
            3. yaml.safe_load(config)           :146-148
            4. deepcopy audionet/videonet 配置   :149-151
            5. videonet.pretrain = None         :151  （推理不需要 frcnn 预训练主干文件）
            6. 架构名比对（VARIANT.architecture）:152-154
            7. MODEL_CLASSES[arch](sample_rate=16000, **audio_cfg) :157
            8. FRCNNVideoModel(**video_cfg)     :158
            9. torch.load(weights_only=True)→state_dict→按 audio_model./video_model. 拆前缀 :160-165
           10. strict=True 加载两个子模型        :168-169
           11. 统计 keys/参数、torch.set_num_threads :170-176
           12. eval() + requires_grad_(False)    :177-180
           13. pick_device(device)               :181
           14. _warmup()（真前向）               :185-195
           15. ready=True                        :196
```

### 3.2 实时请求链（浏览器 → 服务 → 浏览器；[代码]+[验证]）

```
浏览器 capture-worklet 重采样到 16 kHz / 25 fps 抽帧 + MediaPipe 嘴部 96×96 灰度
  web/capture-worklet.js:1-6,14-66；web/video-worker.js（07 域）
  → maybeInfer(): 满足 samples-sessionEpoch ≥ HOP+GUARD(9280)              app.js:564-567
      end = 整到 HOP 网格；首窗强制 end=sessionEpoch+HOP+GUARD             app.js:568-570
      audio[i] = ring[sample]（session 之前的样本为 0）                     app.js:577-581
      frames = frameWindow(...) 50 帧（合成左上下文用首帧）                 app.js:523-541
      WS 发送 {type:'infer', id, audio:b64, frames:[b64×50],
               window_seconds:2, hop_seconds:.5,
               end_sample: end+LEFT_CONTEXT(22720), reset}                  app.js:589-593

server.py WS handler :233-276
  → 同源校验（失败 close 1008） :235-237
  → 读 text，长度 > 2e6 → close 1009 :243-246
  → json 解析 + id 校验（str/int，≤80 字符，int 在 ±(2^53−1)） :249-256
  → asyncio.to_thread(decode_request) :257
  → engine.busy? → 抛 RuntimeError :258-259
  → asyncio.to_thread(engine.infer) :260
        └─ FalconRuntime.infer → _forward → preprocess_mouth→video→audio→output[32000]
  → slicer.take(prediction, end_sample, reset) :261
  → ws.send_json({"type":"result", ...}) :262-267
  异常：ValueError/RuntimeError → {"type":"error", id, message}（连接不关闭） :268-269
        其他 → 记录日志 + 通用错误消息 :270-272

浏览器 handleResult :659-725
  → 校验 id、sample_rate==16000、解码长度==HOP :663,673-675
  → 校验 start_sample/end_sample 与请求窗口严格匹配 :677-681
  → offset 减回真实时间轴、跳过重复块 :682-685
  → 录音/mixture 对齐、播放（contiguous 控制淡入/重采样 reset） :686-718
```

### 3.3 模型切换链（`POST /api/load`；[代码]）

```
前端 loadModel(variant) → POST /api/load {"variant": "big"}   app.js:124-143
server.py:217-231 → 请求体 ≤1024B(:219-221) → JSON object(:223-226)
 → engine.start_load(variant) :226
      variant 合法性 → 正在加载? → 已就绪同款? → 有活跃连接? → 权重可用?
      → 置 variant/runtime=None/loading=True/error=None → 起线程 :142-146
 → 立即返回 engine.status()（可能仍 loading）：前端 1.5 s 轮询 /api/status 直至 ready
```

### 3.4 离线整段链（[代码]）

```
infer_file.main :85-101
  read_wav(:28-38) → read_mouth(:47-59) → 时长差提示(:95-97)
  → FalconRuntime(variant_for(--model), --checkpoint).load(--device) :98
  → extract(:69-82)：整段 waveform[B,L] + mouth[T,88,88] → 一次前向 → (L,) 输出
  → write_wav 16-bit PCM :100
```

### 3.5 形状 / dtype 链（[验证]，见 §8-A5/A6）

```
audio  (32000,) f32 → unsqueeze → (1,32000)
        → STFTEncoder → (1,112,251,129)     # win=256, hop=128 ⇒ T=251, F=129
        → audio_bottleneck → (1,112,251,129)
mouth  (50,96,96) uint8 → preprocess_mouth → (1,1,50,88,88) f32
        → FRCNNVideoModel → (1,512,50)      # 时间维不降采样
        → video_bottleneck = Identity（配置 kernel_size=-1 ⇒ 0 参数）→ (1,512,50)
separator 融合 → (1,112,251,129) → mask_generator → (1,1,112,251,129)
        → decoder → (1,1,32000) f32 → [0,0] → (32000,) f32
```

---

## 4. 关键实现细节与设计动机

### 4.1 为什么是“2 s 窗口 + 80 ms 右上下文”而不是真正流式

- 模型**没有任何流式状态**（无 KV cache、无 RNN 隐状态缓存），代码/文档都明说 windowed inference：
  `falcon_runtime.py:1-6`、`studio/falcon_model/PROVENANCE.md:41-42`。
- 更根本的原因：`cLNhw = LayerNormalizationHW` 内含在**整段 T 上的全窗统计**（`GlobalLayerNorm` 用
  `GroupNorm(1,C)` 在 `[B,C,T,F]` 上统计，`layers/normalizations.py:134-143,204-229`；01 域已给因果性结论），
  因此即使逐层“时间因果”，也无法逐采样复现——每次必须重算整个 2 s 窗口。
- 右上下文 80 ms（1280 样本，`server.py:29`）是“窗口末端预测质量差”的保护带：iSTFT/卷积在末尾缺少未来
  上下文，故丢弃末端 1280 样本；输出块取末端前 0.5 s（§5.5）。
- 频率轴不做因果（F 方向可双向），时间轴单向后向（01/02 域详述），与本服务的窗口化策略一致。

### 4.2 `OutputSlicer`：不重复、不丢样、相邻窗交叉淡化

设计目标（`server.py:70-71` docstring）：**绝不重复输出已发出的采样**，并对相邻窗同时间点预测做淡化。
- 每次输出块固定 8000 样本（0.5 s），窗口每前进 8000 样本恰好衔接上一块末尾 → 时间轴连续（`[验证]` §8-B1，
  两块拼起来与绝对时间轴 allclose，max|diff|=4.3e-8）。
- 相邻窗重叠区（上一窗“右上下文”的首 20 ms）用上一窗预测值 `future` 与新窗预测交叉淡化（§5.5 图）。
- `contiguous=False`（首窗/reset/跳窗）不做淡化，交由前端做 5 ms 淡入（`web/app.js:753-759`）。
- 输出 `clip(-1,1)` 并回传 `clipped_samples`（`server.py:88-89,267`）；`clipped_samples` 前端目前未消费（P2）。

### 4.3 strict loading + 配置 deepcopy（两个真实陷阱）

- `Separator.__init__` 对 `audio_params`/`video_params` 执行 `dict.pop("repeats")`/`pop("shared")`
  （`falcon_model/models/Falcon.py:991-997`），会**原地修改配置 dict**；因此 `falcon_runtime.py:149-150`
  先 `copy.deepcopy`。PROVENANCE 也专门提到（`PROVENANCE.md:24-26`）。
  [验证] 同一份 config dict 连续构造两个模型，第二个参数量会塌缩（01 域验证到 49,081）；本域验证 deepcopy 后
  原配置 `repeats/shared` 仍在（§8-A2）。
- `strict=True`（`falcon_runtime.py:168-169`）拒绝“缺 key 静默用部分权重”的情况；注释明确说明动机
  （`:166-167`）。[验证] 改名一个 audio key → `RuntimeError: Error(s) in loading state_dict for ...`；
  只给 audio 不给 video → `ValueError: Checkpoint must contain both ...`；small ckpt 装进 big 模型 → `RuntimeError`
  （§8-A7）。
- `torch.load(weights_only=True)`（`:160`）避免执行 pickle 任意对象；**同仓库另一种姿态**：
  `audio_visual_test.py:119` 读 Lightning ckpt 用 `weights_only=False`（需要反序列化其中的 Python 对象），
  两种加载口径的差异归 04 域确认是否有意。PROVENANCE 说导出文件只含
  `state_dict`+架构名+epoch（`PROVENANCE.md:36-39`），与本调用兼容（**但真实 falcon.pt 未验证，Q1**）。
- 权重 key 映射：ckpt 的 `audio_model.*`/`video_model.*` 前缀来自 LightningModule 属性名；运行时按前缀拆出
  （`:162-163`）。视频子模型另有 137 个 key（含 BatchNorm3d running stats）；音频 small 525 / big 517 个 key
  （big 少 8 个是因为 `DualPathRNN.out_proj` 在 big 配置下退化为 `nn.Identity`，§4.10）。

### 4.4 权重解析顺序、变体表、设备与环境变量

**变体表（`falcon_runtime.py:31-36`；参数为本轮 [验证] §8-A2）：**

| variant | config | weights | HF repo | 架构类 | audio 参数 | video 参数 | 总参数 |
|---|---|---|---|---|---|---|---|
| `small`（默认） | `configs/Falcon.yml` | `falcon.pt` | `JusperLee/Falcon` | `FalconSpeakerContrastive` | 537,697 | 11,186,688 | **11,724,385**（11.72 M） |
| `big` | `configs/FalconBig.yml` | `falcon-big.pt` | `JusperLee/Falcon-Big` | `SpeakerContrastiveV2` | 1,158,617 | 11,186,688 | **12,345,305**（12.35 M） |

> README 表（`README.md:100-103`）的 11.72 M / 12.35 M 与“含冻结 11.2 M FRCNN”口径**逐一吻合**。
> 论文“0.5 M trainable”与实际 audio 参数 537,697（0.538 M）是取整口径差异（P2，见 Q13）。

**权重解析顺序（`resolve_checkpoint`，`falcon_runtime.py:58-78`）：**

1. `checkpoint` 参数（`FalconRuntime(variant, checkpoint_path)` 或 `infer_file --checkpoint`）：
   - 非 `hf://` 开头 → **直接当本地路径**，原样返回；文件不存在时在 `load()` 里抛
     `FileNotFoundError: Falcon checkpoint not found: ...`（`:143-144`）；
   - `hf://<org>/<repo>/<file>` → `rpartition("/")` 拆 repo/file，任一为空 → `ValueError`（`:64-68`）；
2. 无 `checkpoint` → `VARIANTS[variant]` 的 `repo`+`weights` → `hf_hub_download`（`:66,70-78`），
   缓存 `~/.cache/huggingface`；
3. 下载异常 → `FileNotFoundError`，错误信息提示设 `HF_ENDPOINT` 镜像或传本地文件（`:76-78`）；
4. `huggingface_hub` 未安装 → `FileNotFoundError("huggingface_hub is not installed ...")`（`:70-73`）。

**没有任何“权重环境变量”**：`FALCON_CHECKPOINT`/`FALCON_MODEL` 之类不存在。Studio 运行时相关环境变量只有：

| 变量 | 作用 | 位置 |
|---|---|---|
| `FALCON_DEVICE` | `uvicorn server:app` 路径的默认设备（`main()` 用 `--device` 覆盖） | `server.py:183,294,332` |
| `FALCON_CPU_THREADS` | CPU 线程数，默认 4，夹在 [1,8] | `falcon_runtime.py:175` [验证] 2→2、999→8、0→1 |
| `HF_ENDPOINT` | huggingface_hub 镜像 | `falcon_runtime.py:76-78`（提示文案）、`README.md:44` |
| `FALCON_PYTHON` / `FALCON_VENV` | 启动脚本选解释器/环境 | `scripts/bootstrap.sh:12-13,34`（08 域） |
| `FALCON_QA_PORT` | 浏览器测试服务器端口 | `tests/serve_browser.py:14`（08 域） |

**设备选择（`pick_device`，`:81-92`）**：`auto`→CUDA→MPS→CPU；显式 `cuda` 无 CUDA 报
`RuntimeError("CUDA is unavailable ...")`，显式 `mps` 无 MPS 报 `RuntimeError("Apple Metal/MPS is unavailable ...")`；
非法值 `ValueError`。[验证] 本机 `auto`→`cuda`（A800），`tpu/gpu/None`→ValueError。
GPU `auto` 下 warmup 失败会**降级 CPU 并写 warning**（`:185-195`），显式 GPU 则直接抛错。

### 4.5 预处理与 warmup 的真实数值

`preprocess_mouth`（`falcon_runtime.py:95-104`）：
- 输入必须是 `np.uint8` 且形状**恰好** `(50,96,96)`（`:97-100`，否则 ValueError；[验证] 49 帧/88×88/float32 均被拒）；
- `x = frames[:, 4:92, 4:92] / 255`，再 `(x-0.421)/0.165`（`:101-104`）；
- 对应训练侧 val pipeline：`Normalize(0,255) → CenterCrop(88,88) → Normalize(0.421,0.165)`
  （val pipeline `look2hear/datas/transform.py:161-167`，函数定义 `:147`；`CenterCrop` 的 `delta=(96-88)/2=4` 在同文件 `:82-100`；
  dataset 用 `["val"]`：`look2hear/datas/avspeech_dataset.py:138`）。裁剪与 /255 可交换，顺序无影响。
- **值域不是 [-1,1]**：灰度 0 → **−2.5515**，255 → **+3.5091**；107 → −0.00844（≈ 均值 0.421×255≈107.4）。
  [验证] §8-A4。输出 (1,1,50,88,88) f32。
- 灰度化：服务端用 `PIL.Image.convert("L")`（`:61`）；训练数据本身是灰度裁剪；两者近似（ITU-R 601-2 vs
  数据预处理），未做逐像素等价验证（Q9）。

`_warmup`（`:202-205`）：两个模型 `.to(device)` 后跑一次 `_forward(zeros(32000), full((50,96,96),107))`。
- 107 的选择 [推断]：使归一化后 ≈0（即验证集均值），让 warmup 走“典型输入”路径；无代码注释说明。
- warmup 同时完成 lazy 初始化（cuDNN/MPS kernel、内存分配），因此 `status.ready=True` 时首窗不会额外慢。
- [验证] 合成权重下 CPU warmup ≈720 ms（small）/821 ms（big），与随后 `infer` ≈537/767 ms 同量级（§8-A5）。

`_forward`（`:237-246`）与 `infer`（`:248-259`）：

| 检查 | 规则 | 位置 |
|---|---|---|
| audio 类型 | `np.ndarray` + `dtype==float32` | `:249` |
| audio 形状 | `(32000,)` | `:249` |
| audio 数值 | 全部 finite | `:251-252` |
| frames | 交给 `preprocess_mouth` 校验：uint8 (50,96,96) | `:239`；`:97-100` |
| ready | 未就绪 → `RuntimeError(error or "Falcon model has not finished loading.")` | `:253-255` |
| 输出 | `separated[0,0].float().cpu().numpy().copy()`，形状 `(32000,)`、全部 finite，否则 `RuntimeError("Falcon produced invalid output; this window was not played.")` | `:243-246` |
| 归一化 | **没有** peak/RMS 归一化、没有 clip（clip 只在 server 层） | `:238-246`、`server.py:89` |
| 锁 | `RLock` 串行；`last_inference_ms`（1 位小数） | `:253-258` |

### 4.6 安全模型：loopback + same-origin + TrustedHost + CSP

**结论**：这是“本机可信、浏览器页面可信、网络不可达”的模型，**不是鉴权**；任何本机进程都能直接调 API。

| 层 | 规则 | 证据 |
|---|---|---|
| 监听地址 | CLI 只接受 `127.0.0.1`/`localhost`（不能 `--host 0.0.0.0`） | `server.py:292` |
| HTTP 同源 | 所有 HTTP 路由经中间件；`Origin` 缺失放行；存在则要求 scheme∈{http,https} 且 `netloc==Host` 且 hostname∈{localhost,127.0.0.1,::1}，否则 403 | `server.py:198-202,175-179` |
| WS 同源 | accept 之前用同一函数校验，失败 `close(1008)` | `server.py:235-237` |
| 防 DNS rebinding | `TrustedHostMiddleware(allowed_hosts=[127.0.0.1,localhost,[::1],testserver])`，Host 非法 → 400（http/ws 都生效） | `server.py:196`；[验证] Host=evil.example→400 |
| 响应头 | `nosniff`、`no-referrer`、`Permissions-Policy: camera=(self), microphone=(self)`、`no-store`、CSP（全文）：`default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; worker-src 'self' blob:; frame-ancestors 'none'` | `server.py:204-210`；[验证] §8-B4 |
| 资源 | 静态文件仅 `web/`；`/api/status` 无认证 | `server.py:278-283` |

边界推论（[推断]/[代码]）：
- 反向代理 / 远端部署时，`Origin` 的 hostname 不在 {localhost,127.0.0.1,::1} → **403**；`Host` 不在白名单 → 400。
  因此 `README.md:16` 宣传的在线 demo（`https://u464727-...seetacloud.com:8443`）**不可能**由本文件的
  `server.py` 原样提供服务，必然是另一份部署补丁（Q4）。
- 同源比较是字符串级 `netloc==Host`：页面开在 `localhost` 而代理把 Host 改写成 `127.0.0.1` 会被拒（P2 提示）。
- 无鉴权/无限流：本机任意进程、任意用户可调 `/api/load`、`/ws`。模型页若在浏览器被恶意扩展注入则等同本机权限（设计如此）。

### 4.7 错误如何回传前端

| 层 | 机制 | 证据 |
|---|---|---|
| 模型加载（异步） | `Engine.error = str(exc)`；`/api/status` 的 `error` 字段；前端全局 4 s 轮询（`app.js:1164`）；`startSession` 等模型时每 1.5 s 轮询一次（`app.js:608-619`）+ 180 s 超时 → 弹错 | `server.py:154-158,107-108,115`；`web/app.js:96-108,608-619,1163-1164` |
| `POST /api/load` 同步错误 | `ValueError/JSONDecodeError`→400；`RuntimeError`→409；请求体>1024B→413；成功→200 + 当前 status | `server.py:217-231` |
| WS 协议/推理错误 | `{"type":"error","id":...,"message":...}`，**不关闭连接**，后续消息继续 | `server.py:268-272`；[验证] §8-B5 |
| WS 连接级错误 | 非同源→1008；消息>2e6→1009（`ws.receive_text` 层还会因 uvicorn `ws_max_size` 直接断） | `server.py:235-237,243-246,332-333` |
| 前端 | 把 `type:"error"` 当**致命错误**：`throw` → `stopCapture()` + 提示（Q5：与“请跳过此窗口”语义冲突） | `web/app.js:672,721-724` |
| 语言 | 服务端消息分两套：`server.py` 校验消息为中文、`falcon_runtime.py` 为英文；前端 i18n 表通过“中文原文↔英文译文”双向映射统一本地化 | `i18n-errors.js:3-58`；`i18n.js:44-62` |
| 兜底 | 未捕获异常→日志 + “推理失败，请查看启动终端中的错误信息。”；**永不把输入混合声当输出播放** | `server.py:270-272`；`README.md:47` |

### 4.8 `falcon_model/` vs `look2hear/`：PROVENANCE 逐条核验（**本节为核心**）

> 方法：`diff -u`、`cmp`、`diff --strip-trailing-cr`（完整命令见 §8-A/B）。**不修改源码**。

**(a) 文件级结论表**

| Studio 文件 | look2hear 对应 | diff 结果 | 差异归因 |
|---|---|---|---|
| `models/Falcon.py` | `look2hear/models/Falcon.py` | **净 +9 行**（diff 文本 +65/−56；4 个 hunk 含代码改动 = 1 个 import 替换净 0 + 3 个 Metal bridge；其余 8 个 hunk 为注释中译净 0） | ① `from timm.models.layers import DropPath`（`look2hear/models/Falcon.py:6`）→ `from ..drop_path import DropPath`（`studio/falcon_model/models/Falcon.py:6`）；② **Metal FFT bridge**（编码器 +4 行、解码器 +4/+1 行）；③ 注释被翻译成中文（净 0 行） |
| `models/Falcon_Contrastive.py` | `look2hear/models/Falcon_Contrastive.py` | `cmp` 相同（0 字节差异） | 无 |
| `models/base_model.py` | `look2hear/models/base_model.py` | 102 → 12 行 | 只保留构造器 + `sample_rate()`；删掉 `_unsqueeze_to_3d/pad_to_appropriate_length/forward/load_state_dict_in_audio/from_pretrain/serialize/get_state_dict/get_model_args`（全部是训练/序列化 API，推理不用） |
| `models/SISNRRepel.py` | **无** | — | = Studio `Falcon.py` **+2 行**（`#from sru import SRU`、`#from fla.layers import Comba as FLAComba`）+ `class Falcon`→`class SISNRRepel`（1496→1498） |
| `models/Speaker_Contrastive_v2.py` | **无** | — | = Studio `Falcon_Contrastive.py` **+18 行**（模块 docstring）+ base 类 `Falcon`→`SISNRRepel`（122→140） |
| `videomodels/frcnn_videomodel.py` | `look2hear/videomodels/frcnn_videomodel.py` | 去 CRLF 后仅 3 处：删 `import sys`、`sys.path.append("../../")`、`from .resnet1D import ResNet1D, BasicBlock1D`；加 5 行注释头 | 训练仓库的路径 hack 与未使用 import；**forward/参数无变化** |
| `videomodels/resnet.py`、`shufflenetv2.py` | 同名 | `cmp` 相同 | 无 |
| `layers/activations.py` | 同名 | `cmp` 相同 | 无 |
| `layers/normalizations.py` | 同名 | 13+/13− = **26 行**，净 0 | 仅注释中译（如 `:170,181,213`） |
| `drop_path.py` | 新文件（训练侧用 timm） | — | timm `DropPath` 的无依赖重实现；eval 下恒等，train 下按 drop_prob 逐样本乘 `bernoulli(keep)/keep` |

**(b) `1496 vs 1487` 的 +9 行来源（逐行归因）**

`diff -u look2hear/models/Falcon.py studio/falcon_model/models/Falcon.py` 共 12 个 hunk：**4 个含代码改动**
（1 个 DropPath import 替换净 0 行 + 3 个 Metal FFT bridge 共 +9 行），其余 **8 个是注释中译**（行数一一对应，净 0）：

| 位置（Studio 行号） | 增加内容 | 行数 |
|---|---|---|
| `STFTEncoder.forward`（`studio/.../Falcon.py:1110-1122`） | MPS 时把输入搬到 CPU 做 `torch.stft`，再把 spec 搬回计算设备 | **+4** |
| `STFTDecoder.forward`（`:1183-1198`） | MPS 时先构造复数 spec 于 CPU、`torch.istft` 在 CPU 上跑（复数张量 MPS 支持不全） | **+4** |
| 同上 | `output = output.to(output_device)` 搬回计算设备 | **+1** |

其余差异：`from timm.models.layers import DropPath` → `from ..drop_path import DropPath`（1 换 1，净 0）；
`Separator`/`FiLM` 等注释与 docstring 中译。**没有任何 nn.Module 结构、超参、forward 公式改动**——由
state_dict key+shape 完全相等佐证（§8-A3：audio 525/525、video 137/137、形状全等）。

**(c) Metal FFT bridge 的具体语义（重要边界）**

```python
# studio/falcon_model/models/Falcon.py:1110-1122（STFTEncoder.forward）
compute_device = x.device
stft_x = x.cpu() if x.device.type == "mps" else x          # 仅 MPS 改道 CPU
spec = torch.stft(stft_x, n_fft=..., hop_length=..., window=self.window.to(stft_x.device), return_complex=True)
spec = torch.stack([spec.real, spec.imag, (spec.abs().pow(2)+eps).sqrt()], 1)...
spec = spec.to(compute_device)                              # 结果搬回 MPS
# :1183-1198（STFTDecoder.forward）
if decoded_separated_audio.device.type == "mps": decoded_separated_audio = decoded_separated_audio.cpu()
spec = torch.complex(...); output = torch.istft(spec, ..., window=self.window.to(spec.device), length=length)
output = output.to(output_device)
```
- 只在 `device.type == "mps"` 时改变执行位置；CPU/CUDA 路径与原实现逐运算相同（`.to(同 device)` 为恒等），
  所以 PROVENANCE 的“CPU 上 bit-identical”在代码上成立（[验证] 合成同权重下差值恰为 0，§8-A6）。
- FFT 参数（win/hop/window/`length`）完全未变，因此**数值语义不变**，只是设备迁移。
- 该 bridge 也改变了 MPS 上“模型整体在 MPS”的假设：每窗前向有 2 次 20 万级样本的 H2D/D2H 拷贝（[推断] 这是
  M5 Pro 上 0.12 s/window 仍可达的前提；未见实测数据）。
- 训练侧文件不含此 bridge；[推断] 在 MPS 上训练侧 `torch.stft(return_complex=True)`/`torch.istft` 会因复数张量支持不全而报错
  （PROVENANCE `:20-22` 的现象描述），这也是“Studio 专用改动”的合理理由。

**(d) PROVENANCE 声明逐条判定**

| # | PROVENANCE 声明（行号） | 判定 | 证据/说明 |
|---|---|---|---|
| P1 | `falcon_model/` 是训练代码 `look2hear/` 的 inference-only 拷贝（`:3-4`） | **部分不成立** | 8/10 文件可对上；`models/SISNRRepel.py`、`models/Speaker_Contrastive_v2.py` 在 `look2hear/` **完全不存在**（`find`/`grep` 无；`git log --all -S 'class SISNRRepel' -- look2hear` 无结果；仓库只有 2 个 commit `a7af472`/`7576854`——注意不带 `-- look2hear` 时该查询会命中 `a7af472`，因为 Studio 文件在该 commit 加入） |
| P2 | 文件→来源映射表（`:6-11`） | **2 条无法验证** | 上条；其余 `Falcon.py/Falcon_Contrastive.py/frcnn/resnet/shufflenet/normalizations/activations` 均可对上 |
| P3 | 推理专用改动；**不删任何可学习层**（`:13`） | 成立 | state_dict key 数与形状与训练侧两模型完全一致（525/137），[验证] §8-A3 |
| P4 | timm `DropPath` 换 `drop_path.py`；eval 恒等（`:15`） | 成立 | `drop_path.py:1,11-16`；`Falcon.py:6,590,651`；eval 分支返回 x |
| P5 | base 类只保留无参构造器 + sample-rate（`:16-17`） | 成立 | `models/base_model.py:1-12` |
| P6 | 删除未用 import（`ResNet1D`、`sys.path` 改动）与训练专用包级 import（`:18-19`） | 成立 | `frcnn_videomodel.py` diff 仅这 3 行 + 注释头；4 个 `__init__.py` 只有 docstring |
| P7 | MPS 上 STFT/iSTFT 跑 CPU；FFT 参数/window/输出长度不变；其余在所选设备（`:20-22`） | 成立 | `Falcon.py:1110-1122,1183-1198`；参数未变 |
| P8 | 配置 deepcopy 因 `Separator` 用 `dict.pop`；strict=True 加载两个前缀（`:24-27`） | 成立 | `falcon_runtime.py:149-150,162-169`；`Falcon.py:991-997` |
| P9 | 嘴部遵循训练 val 变换（uint8 灰度 25 fps、/255、96→88、mean .421 std .165）；音频无 peak/RMS 归一化（`:29-31`） | 成立 | `falcon_runtime.py:95-104`；`look2hear/datas/transform.py:161-167`；[验证] 值域 [-2.5515,3.5091] |
| P10 | CPU 上此拷贝与训练代码对 `falcon.pt` 输出 bit-identical（参数 11,724,385，最大绝对差 0）（`:33-34`） | **代码层成立、真实权重未验证** | 参数 11,724,385 [验证] 精确吻合；同权重同输入 CPU 输出 max diff=0 [验证]（合成权重）；真实 `falcon.pt` 未加载（禁止下载）→ Q1 |
| P11 | 发布文件由 epoch 714/715 导出；含 `state_dict`/架构名/epoch；不含 optimizer/Lightning/hyper-params（`:36-39`） | **部分可佐证、未验证** | `Experiments/checkpoint/Falcon/best_k_models.json` 最优 ckpt 恰为 `epoch=714.ckpt`（val_loss −9.984，最低）→ 714 一致；其余无法验证（无权重） |
| P12 | 滚动 2 s 窗口；模型无流式状态（`:41-42`） | 成立 | `falcon_runtime.py:1-6`；`Falcon.forward` 无状态；§4.1 |

### 4.9 README “training code 与 Studio runtime 对 falcon.pt 输出一致”在代码层如何成立

README 原话：`README.md:174`“The training code (`look2hear/`) and the Studio runtime (`studio/falcon_model/`) produce
identical output for `falcon.pt`; the Studio copy only drops training-time dependencies and adds a CPU FFT bridge for
Apple Metal”。代码层成立链条（每步有证据）：

1. **权重 key 机制**：训练时两个模型是 LightningModule 的属性 `audio_model`/`video_model`，`state_dict` 前缀为
   `audio_model.*`/`video_model.*`（PROVENANCE `:36-39`）；Studio 按同前缀切分后 `strict=True` 装入同名子模块
   （`falcon_runtime.py:162-169`）。[验证] Studio 模型与训练模型的 state_dict **key 集合与形状完全一致**
   （audio 525、video 137，§8-A3），所以同一份权重可无转换互换。
2. **类名映射**：`configs/Falcon.yml` 写 `FalconSpeakerContrastive`（`:10`）→ studio 拷贝的
   `Falcon_Contrastive.py` 与训练侧逐字节相同。**但**发布 ckpt 的导出配置
   `Experiments/checkpoint/Falcon/conf.yml:88` 写的是 `SpeakerContrastiveV2`（宽度参数仍是 small 值：
   out_chan 112 / hid_chan 64 / spk_embed_dim 64，见该文件 `:1-68`）。这两个类在本仓库中
   **结构完全相同**：`SISNRRepel` = `Falcon` + 2 行注释 import + 类名；`SpeakerContrastiveV2` = `FalconSpeakerContrastive`
   + 18 行 docstring + base 类名（§2.4、§4.8）。[验证] 用同一（small）配置构造两个类：state_dict keys/shapes 全等、
   参数量同为 537,697、可 strict 互装（§8-A6 附加验证）→ **类名差异不影响 key 与输出**。
3. **前向路径**：`Falcon.forward`（`models/Falcon.py:1473-1492`）与 `FalconSpeakerContrastive.forward`
   （`Falcon_Contrastive.py:94-118`）/`SpeakerContrastiveV2.forward`（`Speaker_Contrastive_v2.py:112-136`）在
   `eval()` 下都只返回 `separated_audios`；Studio 不调用训练分支（`self.training=False`）。
4. **无训练依赖副作用**：`base_model.py` 删掉的只是 `from_pretrain/serialize` 等；DropPath/BatchNorm 在 eval
   恒等/用 running stats；`FRCNNVideoModel.train()` 的 BN 冻结覆盖未被触发（`falcon_runtime.py:177-180`）。
5. **实证**：用同一套合成权重分别构造 “训练侧 `look2hear.models.FalconSpeakerContrastive` + 训练侧
   `look2hear.videomodels.FRCNNVideoModel`” 和 “Studio `FalconRuntime`”，在 CPU、同一预处理、同一输入下前向，
   **max|Δ| = 0.0（bit-identical）**（§8-A6）。
6. **成立的条件/缺口**：a) 真实 `falcon.pt` 的 key 名与形状未验证（Q1）；b) 上一条“训练侧类”用的是
   `FalconSpeakerContrastive`，而发布 ckpt 的 conf 写 `SpeakerContrastiveV2`，发布 `look2hear/` 里没有这个类
   （`getattr(look2hear.models,"SpeakerContrastiveV2")` → AttributeError，[验证]）；
   `audio_visual_test.py:129-132` 正是这样构建模型的 → 该评测脚本对
   `Experiments/checkpoint/Falcon/conf.yml` 会失败（Q3，跨到 04 域）。所以“训练代码与 Studio 一致”目前只能
   通过“Studio 内部两份拷贝 + 训练侧 Falcon.py”间接证明，**不是**用发布训练代码直接加载发布 ckpt 证明的。

### 4.10 FalconBig（SpeakerContrastiveV2, wider）vs small 配置差异表

配置来源：small `studio/configs/Falcon.yml`（= 训练根 `configs/Falcon.yml`）；big `studio/configs/FalconBig.yml`。
下表只列**模型相关**差异（其余键相同或与服务无关）：

| 键 | small | big | 影响 |
|---|---|---|---|
| `audionet_name` | `FalconSpeakerContrastive`（`Falcon.yml:10`） | `SpeakerContrastiveV2`（`FalconBig.yml:88`） | 类名不同、结构同构（§4.9-2） |
| `enc_dec_params.out_chan` | 112（`Falcon.yml:30`） | 256（`FalconBig.yml:53`） | STFT 编码器输出通道；决定 bottleneck/解码器输入 |
| `enc_dec_params.{win,hop_length,kernel_size,stride,bias,causal}` | 256/128/3/1/false/true（`:27-36`） | 256/128/3/1/false/true（`:46-55`） | 不变 |
| `audio_bn_params.out_chan` | 112（`:23`） | 256（`:6`） | audio bottleneck 通道；`Separator.audio_bn_chan` |
| `audio_params.hid_chan` | 64（`:40`） | 96（`:12`） | AudioBlock 内部宽度 |
| `audio_params.layers.layer_1.hid_chan` | 24（`:51`） | 48（`:19`） | DualPathRNN(F) 宽度；`bidirectional:true` |
| `audio_params.layers.layer_2.hid_chan` | 72（`:59`） | 96（`:27`） | DualPathRNN(T) 宽度；`bidirectional:false` |
| `audio_params.{n_freqs,kernel_size,stride,upsampling_depth,repeats,shared}` | 129/5/2/4/2/true（`:39-48`） | 129/5/2/4/2/true（`:39-44`） | 不变 |
| `spk_embed_dim` | 64（`:15`） | 128（`:68`） | speaker 对比头投影维度（推理不使用，只参与 state_dict） |
| `video_params.hid_chan` | 32（`:75`） | 64（`:74`） | VideoBlock 投影/RNN 宽度 |
| `videonet` | FRCNN(resnet,prelu,1.0,pretrain)（`:2-8`） | 同（`FalconBig.yml:160-166`） | 完全不变（11,186,688 参数两者相同） |
| 其余 | `n_src:1`、`pretrained_vout_chan:512`、`causal:true`、fusion `NeuroSCNetFusion k=4 shared`、mask `ReLU/RI_split/2d` | 同 | 不变 |

**参数量差异（[验证] §8-A2，逐模块）：**

| 模块 | small | big | Δ |
|---|---|---|---|
| encoder（STFTEncoder） | 3,024 | 6,912 | +3,888 |
| audio_bottleneck | 13,104 | 66,816 | +53,712 |
| separator.audio_net | 445,766 | 847,254 | +401,488 |
| separator.video_net | 42,850 | 100,546 | +57,696 |
| separator.caf（NeuroSCNetFusion） | 3,696 | 768 | **−2,928**（见下注） |
| mask_generator | 12,657 | 65,793 | +53,136 |
| decoder | 2,016 | 4,608 | +2,592 |
| spk_pool_proj | 14,584 | 65,920 | +51,336 |
| video_bottleneck | 0 | 0 | 0（Identity，`Falcon.py:30` + `kernel_size=-1`） |
| **audio 合计** | **537,697** | **1,158,617** | **+620,920** |
| video（FRCNN） | 11,186,688 | 11,186,688 | 0 |
| **总计** | **11,724,385** | **12,345,305** | **+620,920** |

> `separator.caf` 在 big 反而更小是**正常**的：`NeuroSCNetFusion` 仅在 audio/video 通道不等时建一个分组 1×1 Conv1d，`groups=gcd(vin,ain)`（`Falcon.py:239-246`）。small：`gcd(512,112)=16` → 参数 `112·512/16+112=3,696`；big：`gcd(512,256)=256` → `256·512/256+256=768`。通道更宽但分组更细，参数量反而下降。

补充发现（[验证]）：**big 的 audio state_dict 比 small 少 8 个 key**——`separator.{process_f,process_t}_layers.*.out_proj.weight`。
原因：`DualPathRNN.out_proj` 仅在 `rnn_out_chan != in_chan` 时是 `nn.Linear`，否则 `nn.Identity`
（`Falcon.py:380-383`）。small：`AudioBlock.hid_chan=64`，layer_1 `24×2=48≠64`、layer_2 `72≠64` → 8 个 Linear；
big：`hid_chan=96`，layer_1 `48×2=96==96`、layer_2 `96==96` → 8 个 Identity。这也说明“big 更大”的同时，
其 RNN 输出恰好无需回投影（设计上的巧合，不是 bug）。

### 4.11 协议时序、并发与失败模式（R3 补充；回应 R2 评审 §4 遗漏 1–7）

#### 4.11.1 多连接并发与 `busy` / `infer_lock` 的可观察竞态
- 服务端允许 N 个 WS 连接，但全服务只有一对模型和一个 `infer_lock`；`uvicorn(..., ws_max_queue=1)`（`server.py:333`，
  验证环境 uvicorn 0.54.0）——[推断] 其语义是框架层“每连接待处理消息队列上限为 1”（具体实现依 uvicorn/websockets 版本），
  因此**服务端不提供应用层排队**。
- 处理顺序：WS handler 先查 `engine.busy`（`server.py:258-259`）→ `asyncio.to_thread(engine.infer)`（`:260`）→
  `Engine.infer` 用**非阻塞** `infer_lock.acquire`，失败抛“另一个窗口正在使用模型…”（`:163-164`）。
- `Engine.infer` 的 `finally` 是**先 `busy=False` 再 `infer_lock.release()`**（`:169-172`），因此存在
  “busy 已 False 但锁尚未释放”的窗口：另一连接可通过 busy 检查、随后收到“另一个窗口”错误。
- 两种错误文案语义不同（服务端不是排队而是明确拒绝），前端都按致命处理（Q5/P1）。客户端应保持 single-flight
  （前端确实如此：`web/app.js:565,588`），多标签页则可能触发下述时序。

| 事件顺序（A、B 两个连接） | A 看到 | B 看到 |
|---|---|---|
| A 推理中，B 发请求 | — | `{"type":"error","message":"模型忙，请跳过此窗口后重试。"}` |
| A 进入 finally（busy=False、锁未放） | — | 通过 busy 检查 → acquire 失败 → `"另一个窗口正在使用模型，请关闭重复会话。"` |
| A 空闲，B 发请求 | — | 正常 `result` |

#### 4.11.2 每个 `result` 回包都重建一次 `status()`
`server.py:265` 写 `"device": engine.status().get("device")`；`Engine.status()` 每次都会重建 `variants()`
（`:118-124`），对每个 variant 调 `weights_source()` 两次（`:122`）→ 每个 0.5 s 回包约 4 次
`try_to_load_from_cache` 文件系统查询 + 构造对象。功能正确但属热路径浪费；[推断] 可在加载完成时缓存
`device`/`variants`（未改代码，仅记录）。**另注意错误放大路径**：`engine.status()` 在 `send_json` 之前同步调用
（`server.py:265`），若 `variants()`/`weights_source()` 因缓存目录权限等抛异常，会落入 `except Exception`
通用错误分支（`:270-272`）→ 前端看到“推理失败，请查看启动终端中的错误信息。”，而真实原因是设备/状态查询失败。

#### 4.11.3 无请求级超时 / 取消
`await asyncio.to_thread(engine.infer, ...)`（`server.py:260`）**没有** `asyncio.wait_for`；若前向 hang，
服务端不会主动断开或记录，只能等前端 60 s 定时器（`web/app.js:589-591`）触发 `stopCapture`，或进程退出。

#### 4.11.4 `variants[].available` 的组合语义（`server.py:118-124`；`falcon_runtime.py:48-55`）

| 情形 | `weights_source` | `available` | `source` | 行为 |
|---|---|---|---|---|
| 已缓存 | `"cached"` | true | `"cached"` | 可直接加载（离线可用） |
| hub 已装、未缓存 | `"huggingface"` | true | `"huggingface"` | UI“首次使用将下载”；加载时 `hf_hub_download`，失败进 error |
| hub 未安装 | `None` | **false** | `None` | 选项禁用；`/api/load` → **400**（文案误报“缺权重”，Q8） |
| 离线但已缓存 | `"cached"` | true | `"cached"` | 可加载（符合“下载一次后离线”） |
| 离线未缓存 | `"huggingface"`（只查本地缓存，不联网） | true | `"huggingface"` | 加载线程下载失败 → S3 error |
| 期望的本地 `models/` 目录 | ——**无此逻辑** | — | — | Q8 建议实现或删除相关文案 |

#### 4.11.5 `TrustedHost` 对 WebSocket 表现为“握手拒绝”
HTTP 侧非法 Host → 400；WS 侧 Starlette 会把 400 写到握手响应，客户端表现为**握手失败/断连**
（TestClient 抛 `WebSocketDenialResponse`，R2 评审实测），浏览器触发 `onerror/onclose`（`web/app.js:623-632`）。
因此 §4.6 的“Host 非法 → 400（http/ws 都生效）”在 WS 侧应理解为“握手被拒”。

#### 4.11.6 `infer_file.read_wav` 的 32-bit float 路径不对称
16-bit PCM → `/32768` 归一到 [-1,1]；32-bit float 则直接 `np.frombuffer(..., "<f4")`（`infer_file.py:34-38`），
**不 clip、不校验范围**；而 WS 路径有 `|amp|≤4` 的保护（`server.py:49`）。
[推断] 属“信任离线输入（32-bit float WAV 通常已归一化）”的设计；若输入超范围会原样进模型，使用时应自行保证。

#### 4.11.7 模块级 `app` 与“双 Engine”
`server.py:287` 的 `app = create_app()` 在 **import 时**创建一个 `Engine`（不 autoload）；`main()` 又
`uvicorn.run(create_app(Engine(args.device, variant)), ...)` 创建第二个（`:332`）。`python server.py` 用后者，
`uvicorn server:app` 用前者；因此任何 `import server` 的测试/工具会多一个无害 Engine 实例，
自定义引擎必须显式 `create_app(engine, autoload=False)`（仓库测试正是如此）。

#### 4.11.8 失败注入矩阵（服务端行为 ↔ 前端可见结果）

| 场景 | 服务端行为 | 前端可见 |
|---|---|---|
| WS 中途断开 | `WebSocketDisconnect` → finally `active_connections -= 1`（`:273-276`） | `onclose` → stopCapture + 提示（`app.js:632`） |
| 单次推理 hang | 无超时（§4.11.3） | 60 s 定时器 → stopCapture（`app.js:589-591`） |
| 加载中收到 infer | runtime None → `{"error":"模型仍在加载，请稍后重试。"}` | 正常流程不会发生（等 ready 才连 WS，`app.js:608-619`） |
| `/api/load` 400/409/413 | 对应 HTTP 状态 + `detail` | `loadModel` 抛错并 notice（`app.js:130-136`） |
| 首次下载失败/超时 | 加载线程 error → `status.error`；runtime None | 等待循环/4 s 轮询显示“加载失败”+ 原文（`app.js:96-108`） |
| 缺 `huggingface_hub` | `available=false`；`/api/load`→400 | 变体选项禁用；手动加载报 400 |
| 帧数/音频/end_sample 非法 | `{"type":"error",...}`（连接存活） | `throw` → stopCapture（会话终止） |
| 消息 >2,000,000 字符 | close 1009 | `onclose` → stopCapture |
| 非同源页面 | HTTP 403 / WS close 1008 | 页面无法调用（安全边界） |
| 模型忙 / 锁竞争（§4.11.1） | 两种 error 文案 | 致命 → stopCapture（Q5） |

#### 4.11.9 部署形态（明确结论）
即使把 `server.py` 绑到 `0.0.0.0` 或放在反向代理后面，`same_origin`（`server.py:175-179`）与
`TrustedHostMiddleware`（`:196`）都会拒绝远端 `Origin`/`Host`。所以**仓库内 `server.py` 不能直接对外部署**；
对外至少需要：放开/改写 Origin 与 Host 白名单、TLS、鉴权与限流。README 的“在线 demo”即属此类部署补丁（Q4）。

---

## 5. 对外接口契约

> 所有 JSON 字段名/类型均以实际运行抓包核对（§8-B4/B5）。时间单位：采样点（16 kHz）；音频 base64 为
> **little-endian float32** 原始字节；`inference_ms` 为毫秒。

### 5.1 `GET /api/status`（`server.py:105-116,213-215`）

未加载时返回基础字段；一旦 `Engine.runtime` 存在，会合并 `FalconRuntime.status()`（`:215-235`），
并再次覆盖 `loading/error/busy/variant/ready`（`:115`）。

| 字段 | 类型 | 单位/含义 | 来源 |
|---|---|---|---|
| `app` | `"falcon-studio"` | 就绪探测标记（`main()` 打开浏览器前判定） | `server.py:112,326` |
| `ready` | bool | 模型可推理（= `not loading and runtime.ready`） | `:107,115` |
| `loading` | bool | 加载线程进行中 | `:107` |
| `error` | str\|null | 最近一次加载错误 | `:107,156` |
| `busy` | bool | 正在执行一次前向 | `:111,165-171` |
| `device` | str | 加载前=请求值(`auto/cpu/cuda/mps`)；加载后=实际设备 | `:108`；`falcon_runtime.py:221` |
| `model` | str | `Falcon · Small` / `Falcon · Big` | `:108` |
| `variant` | `"small"`\|`"big"` | 当前/目标变体 | `:109` |
| `variants` | list[obj] | `{id,name,available,source,checkpoint,repo}`；`source∈{"cached","huggingface",null}` | `:109,118-124` |
| `checkpoint` | str | 权重文件名（加载后为实际解析到的文件名） | `:108`；`falcon_runtime.py:223` |
| `sample_rate` | int | 16000 Hz | `:110` |
| `video_fps` | int | 25 | `:110` |
| `window_seconds` | number | 2（加载后为 2.0） | `:110`；`falcon_runtime.py:226` |
| `hop_seconds` | number | 0.5 | `:110` |
| `output_delay_ms` | int | 80（右上下文 1280/16000） | `:111` |
| `mode` | str | 加载前 `"sliding-window"`；加载后被运行时覆盖为 `"windowed_audio_visual"` | `:112`；`falcon_runtime.py:234` |
| `architecture`* | str | `FalconSpeakerContrastive`/`SpeakerContrastiveV2` | `falcon_runtime.py:220` |
| `requested_device`* | str | 请求的设备（auto/cpu/...） | `:222` |
| `audio_weight_keys`* | int | 525（small）/517（big） | `:227` |
| `video_weight_keys`* | int | 137 | `:228` |
| `parameters`* | int | 11,724,385 / 12,345,305 | `:229` |
| `last_inference_ms`* | float\|null | 上一次含嘴部预处理的前向耗时（1 位小数；不含协议解码/图片解码） | `:230`；`falcon_runtime.py:258` |
| `warmup_ms`* | float\|null | warmup 总耗时（1 位小数） | `:231`；`falcon_runtime.py:195` |
| `warning`* | str\|null | GPU warmup 失败降级 CPU 的说明 | `falcon_runtime.py:192,233` |

`*` = 仅加载后出现。前端实际消费：`ready/loading/error/variant/variants/device`（`web/app.js:79-110`），
其余为诊断/日志用途（`warmup_ms/parameters/weight_keys/warning` 前端未显示，P2）。

### 5.2 `POST /api/load`（`server.py:217-231`）

| 项 | 规则 |
|---|---|
| 请求体 | ≤ **1024 字节**（超 → 413，`:219-221`）；JSON **对象**（数组/标量 → 400，`:223-226`） |
| 字段 | `{"variant": "small"\|"big"}`；缺省 → 用当前 `/` 初始变体（`payload.get("variant")` → `start_load(None)`，`:226`） |
| 200 | 返回 `engine.status()` 快照；**加载可能是异步进行的**（若刚触发，`loading=true`） |
| 400 | `ValueError`（未知变体、请求非对象、bad json、**变体不可用**——见下） |
| 409 | `RuntimeError`（加载中切别的变体 / 有活跃 WS 连接） |
| 413 | body > 1024 B |
| 不能做 | 指定设备、指定本地 checkpoint（服务端只能 variant；设备在启动时 `--device` 固定） |
| 前端 | `POST` 180 s 超时；`startSession` 里另用 1.5 s 轮询 `/api/status` 等 ready（`app.js:124-143,608-619`） |

`start_load` 的守卫顺序（`server.py:126-146`）：
① 未知变体 → ValueError；② 正在加载且变体不同 → RuntimeError；③ 正在加载同变体 → no-op return；
④ 同变体且已加载 → no-op return；⑤ `active_connections>0` → RuntimeError；⑥ 变体不可用 → **ValueError→400**
（触发条件：`weights_source()` 返回 None，即 `huggingface_hub` 未安装；此时错误文案“尚未安装 … 权重：models/…”把缺包误报成缺权重文件，见 Q8）；⑦ 置 `variant`、清空 `runtime`、
`loading=True/error=None`，起 daemon 线程。

### 5.3 `WS /ws` 请求（client → server；`server.py:35-67,233-276`）

单条消息 = 一个 JSON object 文本（≤2,000,000 字符；超 → close 1009）。
**单连接内请求-响应串行**：`server.py:242-272` 的 `while True` 先 `await ws.receive_text()`，处理并
`await ws.send_json(...)` 后才读下一条 ⇒ 同一连接不会交错回包，前端只需按 `id` 匹配单飞请求。
**上限兜底**：`2^53−1` 只由服务端校验（`server.py:65,255`）；前端用 JS `number` 递增 id/时间戳且不检查该上限
（远小于安全整数），越界由服务端拒绝。字段：

| 字段 | 必填 | 类型 | 取值/单位 | 校验与错误消息 |
|---|---|---|---|---|
| `type` | ✓ | str | 必须 `"infer"` | 否则 `未知请求类型。` |
| `id` | ✓ | str\|int | 不能 bool；`len(str(id))≤80`；int 时 `\|id\|≤2^53−1` | `请求 ID 无效。` / `请求 ID 超出精确整数范围。`；错误响应里未解析出 id 时为 `null` |
| `audio` | ✓ | str | base64(32000×`<f4`)=**170668** 字符；`max\|amp\|≤4`；全部 finite | `音频必须是 32000 个 16 kHz float32 采样。` / `音频编码无效。` / `音频数据无效或超出允许范围。` |
| `frames` | ✓ | list[str] | **恰 50 个** base64，单个 ≤40000 字符；解码须为 **96×96 JPEG/PNG**（RGB 也接受，服务端转 `L`） | `每个窗口需要 50 帧同步嘴部图像。` / `嘴部图像大小无效。` / `嘴部图像必须为 96×96 的 JPEG 或 PNG。` / `无法读取嘴部图像。` |
| `window_seconds` | – | number | 缺省 =2；出现时必须 `==2`（`2.0` 可） | `当前模型使用 2 秒窗口、0.5 秒步长。` |
| `hop_seconds` | – | number | 缺省 =0.5；出现时必须 `==0.5` | 同上 |
| `end_sample` | – | int | 缺省 32000；`32000 ≤ end_sample ≤ 2^53−1`；bool/float 拒绝。**协议时间轴的窗口末端**（含前端合成左上下文，见 §5.5） | `音频时间戳无效。` |
| `reset` | – | bool | 缺省 false；true = 新会话/重置，不跨窗交叉淡化 | `重置标记必须为布尔值。` |

通道/内存事实：单条消息上限 2 MB（`server.py:30`）与单帧上限 40000 字符叠加后，**实际每帧 b64 上限约
(2,000,000−170,668)/50 ≈ **36.6 千字符**（上限，未扣 `"frames":[...]` 等 JSON 开销）**，超过会先被 uvicorn `ws_max_size` 断开（1009）而不是得到
`{"type":"error"}`（[验证] §8-B6：2,000,000 字符边界内得到 error、边界外 close 1009）。

### 5.4 `WS /ws` 响应（server → client）

**(1) `result`（`server.py:262-267`；[验证] 字段/类型/数值见 §8-B5）**

| 字段 | 类型 | 单位/含义 |
|---|---|---|
| `type` | `"result"` | |
| `id` | str\|int | 回显请求 id |
| `audio` | str | base64(8000×`<f4`) = **42668** 字符；即 0.5 s 输出块 |
| `sample_rate` | int | 16000 |
| `inference_ms` | number | 含 `preprocess_mouth` 的模型前向耗时（ms，1 位小数，`server.py:167-169`）；不含协议/图片解码 |
| `rtf` | float | `inference_ms/500`（相对 0.5 s 输出时长；>1 表示跟不上） |
| `window_rtf` | float | `inference_ms/2000`（相对 2 s 窗口） |
| `device` | str | 实际设备；取自 `engine.status()["device"]` |
| `start_sample` | int | `endpoint - GUARD - HOP` = 输出块起始（协议时间轴） |
| `end_sample` | int | `endpoint - GUARD` = 输出块结束（**注意与请求 `end_sample` 含义不同**） |
| `output_delay_ms` | int | 80（右上下文） |
| `contiguous` | bool | 本块与上一块是否“端点恰好 +HOP 且未 reset” |
| `clipped_samples` | int | 本块中 `\|x\|>1` 被裁剪的样本数（前端未消费） |

**(2) `error`（`server.py:268-272`）**：`{"type":"error","id": <str\|int\|null>, "message": str}`；
覆盖 `ValueError`（协议/校验）与 `RuntimeError`（未就绪/忙/推理异常）；未知异常记录日志后返回通用消息。
**连接不关闭**，后续请求继续可用（[验证]）。

**(3) 连接关闭码**：非同源 `1008`（`:235-237`）；消息超限 `1009`（`:244-246`）；其余异常由框架关闭。

### 5.5 `OutputSlicer` 的 endpoint / 右上下文语义（与前端配合）

**两套坐标必须区分（R3 修订 P2-8；R4 修订 P2-R1：原点不是 `sessionEpoch`）**：

| 坐标系 | 定义 | 原点 | 用途 |
|---|---|---|---|
| **绝对采样坐标**（in-stream，`s`） | 浏览器自**打开设备**起累计的采集样本下标（`state.samples`，`app.js:15,211`）；`app.js:637` `state.sessionEpoch = state.samples` 只是标记“开始分离”的位置 | 打开设备后的第 1 个采集样本 | 录音/回放/mixture 对齐（`app.js:686-694`） |
| **协议坐标**（wire，`P`） | 请求/响应 `end_sample`、`start_sample` 所在坐标 | 协议 0 = 绝对采样 `−22720`（首窗窗口左边界；前 1.42 s 为合成音频/首帧） | `server.py:64-67,266` 的消息字段 |

换算：**`P = s + LEFT_CONTEXT(22720)`**（对任意 `sessionEpoch` 都成立）；前端收到结果后减 22720 转回绝对采样
（`app.js:682-683`）。注意 **`sessionEpoch` 不是协议原点**；若定义“相对开始点的坐标” `r = s − sessionEpoch`，
则 `P = r + sessionEpoch + 22720`（多出 `sessionEpoch` 项）。

**数值例（[验证] §8-B5 + 前端公式；`sessionEpoch=0` 与 `sessionEpoch=160000` 两行对照）**：

| 场景 | 窗 | 请求 `end_sample`（协议） | 协议坐标窗口 | 返回 `start`/`end`（协议） | 折算绝对采样 |
|---|---|---|---|---|---|
| 打开设备即开始（`sessionEpoch=0`） | 第 1 窗 | `end+22720 = 32000` | `[0, 32000)` | `22720 / 30720` | `[0, 8000)` |
| 同上 | 第 2 窗 | `40000` | `[8000, 40000)` | `30720 / 38720` | `[8000, 16000)` |
| 预览 10 s 后开始（`sessionEpoch=160000`） | 第 1 窗 | `sessionEpoch+32000 = 192000` | `[160000, 192000)` | `182720 / 190720` | `[160000, 168000)` |

> 换算校验：`22720 − 22720 = 0`（绝对）；`182720 − 22720 = 160000`（绝对）⇒ 两种场景的首块都正好从
> “开始分离后的第一个真实样本”开始；相邻窗首尾相接（第 2 窗协议 `30720 − 22720 = 8000` = 上一块末尾）。

以下图以**协议坐标**表示（`endpoint = 请求 end_sample`；window 覆盖 `[endpoint-32000, endpoint)`）。
**首窗协议窗口 = `[sessionEpoch, sessionEpoch+32000)`；图中的 `32000/22720/30720` 等数值只对应 `sessionEpoch=0` 的示例。**

```
endpoint-32000                endpoint-9280      endpoint-1280   endpoint
|------ 2 s window audio ------|---- 0.5 s 输出块 ----|-- 80 ms 右上下文 --|
                              ↑22720(相对)          ↑30720(相对)
输出块 = pred[WINDOW-GUARD-HOP : WINDOW-GUARD] = pred[22720:30720]  (server.py:80-81)
future = pred[30720 : 30720+320] = 右上下文前 20 ms，留给下一窗交叉淡化 (server.py:86)
下一窗 endpoint' = endpoint + 8000：
  输出块' = [endpoint-1280, endpoint+6720)，与上一块首 20 ms 绝对时间重叠
  → chunk'[:320] = future*(1-ramp) + chunk'[:320]*ramp, ramp=linspace(0,1,320)   (server.py:83-85)
```

- 连续性判定：`contiguous = (not reset) and last_endpoint is not None and endpoint-last_endpoint == HOP`
  （`:82`）。不连续（首窗/reset/跳窗）时**不**做淡化，仅记录 `last_endpoint`、刷新 `future`。
- 输出长度恒 8000 样本（0.5 s）；`[验证]` 两块拼接与绝对时间轴 allclose（max|Δ|=4.3e-8）。
- 时间戳回传：`start_sample = endpoint-9280`、`end_sample = endpoint-1280`（`:266-267`）。
- 前端配合（`web/app.js`）：
  - `GUARD=1280`、`LEFT_CONTEXT=WINDOW-HOP-GUARD=22720`（`app.js:9-10`）；发送 `end_sample = 绝对 end + LEFT_CONTEXT`
    （`:593`）。`sessionEpoch=0` 时首窗 `end_sample` 恰为 32000；**一般首窗 = `sessionEpoch+32000`**，
    窗口左边界（绝对采样）= `sessionEpoch−22720`（前 1.42 s 补零/首帧；`app.js:570,577-581`）。
  - 首窗需要真实输入 `HOP+GUARD=9280`=0.58 s（`:567-570`）；因此首块绝对采样区间 = `[sessionEpoch, sessionEpoch+8000)`，
    从采集边界开始，不丢开头（与 `README.md:97` 一致）。
  - 收到结果后校验 `end_sample === 请求end + 22720 - 1280` 且 `start_sample === end_sample - 8000`，不符即停止播放
    （`:677-681`），再减 22720 转回绝对采样（`:682-683`）。
  - `contiguous` 用于播放：false 时重置上采样器并做 5 ms 淡入（`:718,747-759`）；与服务端 20 ms 淡化分工明确。
- `[推断]` 设计动机：模型对“窗口最末端 80 ms”预测最不可靠（缺未来上下文+全窗归一化），丢掉它并把它作为
  下一窗的重叠淡化区，可同时满足“低延迟（0.5 s 块）”“无重复样本”“无爆音”三个约束。

### 5.6 `Engine` 状态机（`server.py:92-172,105-116`）

**状态变量**：`runtime`（None 或已加载的 FalconRuntime）、`loading`、`error`、`busy`、`active_connections`、`variant`。
**对外派生**：`ready = (not loading) and (runtime is not None and runtime.ready)`（`:115`）。

| 状态 | `runtime` | `loading` | `error` | `ready` | 进入事件 |
|---|---|---|---|---|---|
| S0 unloaded | None | False | None | False | 进程启动（`autoload=False` 时）/初始 |
| S1 loading | None（切换时先清空） | True | None | False | 启动 autoload / `POST /api/load` 触发 |
| S2 ready | not None & `ready=True` | False | None | True | 加载线程成功（`:152-153,196`） |
| S3 error | None | False | str | False | 加载线程异常（`:154-158`）/ lifespan 捕获（`:188-191`） |
| busy（瞬态，叠加在 S2） | — | — | — | — | WS 进入 `Engine.infer`，`busy=True`，finally 复位（`:165-171`） |

转移表：

| 起点 | 事件 | 守卫 | 终点 |
|---|---|---|---|
| S0/S3 | `start_load(v)` | v 合法且权重可用 | S1（线程） |
| S1 | `start_load(same v)` | — | S1（no-op） |
| S1 | `start_load(other v)` | — | 拒绝：`RuntimeError("模型加载中，请等待完成后再切换。")` |
| S2 | `start_load(same v)` | — | S2（no-op） |
| S2 | `start_load(other v)` | `active_connections==0` | S1；否则 `RuntimeError("请先停止所有分离会话，再切换模型。")` |
| S1 | 线程成功 | — | S2 |
| S1 | 线程异常 | — | S3（`error=str(exc)`） |
| S2 | WS 推理 | `infer_lock` 空闲 | busy=True → S2；锁被占 → error `"另一个窗口正在使用模型，请关闭重复会话。"` |
| any(WS) | `runtime is None` | — | error `error or "模型仍在加载，请稍后重试。"` |

**已知状态一致性瑕疵（P2）**：[验证] 手动把 `engine.error` 置为字符串而 `runtime.ready=True` 时，
`status()` 会同时返回 `error="boom"` 与 `ready=True`（error 来自 engine，ready 只看 runtime，`:115`）。
正常路径不可达（切换时 runtime 先清空），但 `status()` 的字段语义应保证互斥。

### 5.7 Python API / CLI 契约

```python
from falcon_runtime import FalconRuntime
m = FalconRuntime.from_pretrained("JusperLee/Falcon")       # 或 "JusperLee/Falcon-Big"/"small"/"big"
m = FalconRuntime("small", "/path/falcon.pt").load("cpu")   # 自定义本地权重/hf://
out = m.infer(np.zeros(32000, np.float32),                 # 必须 f32 (32000,)
              np.zeros((50, 96, 96), np.uint8))            # 必须 uint8 (50,96,96)
# out: np.float32 (32000,)；可能超出 [-1,1]（服务端才 clip）
m.status()   # §5.1 的运行时字段
```
`infer_file.py`：`python infer_file.py --audio mix.wav --mouth mouth.npy --out target.wav [--model ...] [--checkpoint ...] [--device auto|cuda|mps|cpu]`
（`infer_file.py:85-101`；输入要求见 `:1-12`）。

### 5.8 依赖方向

```
browser (07)  ──HTTP/WS──▶ server.py ──▶ falcon_runtime.py ──▶ falcon_model/* (torch only)
                                   └──(静态文件)──▶ web/*
infer_file.py ─────────────────────────▶ falcon_runtime.py ──▶ falcon_model/*
server.py ✗不 import falcon_model 的类；只 import VARIANTS/weights_source/FalconRuntime（延迟 import，便于测试注入）
```
`server.py:106,121,150` 的 import 都在函数内（延迟），因此 `tests/test_server.py` 可用 StubEngine 在无 torch 环境下测协议
（[验证] 但本环境有 torch；stub 仅用于隔离模型加载）。

### 5.9 前后端常量对照（与 07 域交叉核对）

| 常量 | 后端 | 前端 | 一致性 |
|---|---|---|---|
| 采样率 | `server.py:28` =16000 | `app.js:6` RATE=16000 | ✅ |
| 窗口 | `:28` =32000 | `app.js:7` =2*RATE | ✅ |
| hop | `:28` =8000 | `app.js:8` =RATE/2 | ✅ |
| 帧率 | `:28` =25；`falcon_runtime.py:25` | `app.js:11` FPS=25 | ✅ |
| 右上下文 | `:29` GUARD=1280 | `app.js:9` GUARD=1280 | ✅ |
| 左上下文 | `:80` 切片起点 WINDOW-GUARD-HOP=22720 | `app.js:10` LEFT_CONTEXT=22720 | ✅（任务书里的 22400 无代码依据） |
| 帧数 | `:51-52` 50 | `app.js:533` `FPS*2` | ✅ |
| 输出块 | `:80-81` 8000 | `app.js:675` 断言 samples.length===HOP | ✅ |
| 交叉淡化 | `:29` FADE=320（20 ms，服务端） | 5 ms 淡入仅在 discontinuous 块（`app.js:753-759`） | ✅ 分工不同 |
| 消息字段 | `decode_request`（§5.3） | `app.js:593` 发送全部字段 | ✅ |
| 结果字段 | `server.py:262-267` | `app.js:663-718` 消费 id/type/audio/sample_rate/inference_ms/rtf/start/end/contiguous/output_delay_ms；**不消费** window_rtf/clipped_samples/device | ✅（后三者为诊断字段） |

---

## 6. 可验证断言表

> 68 条。`[代码]`=源码直读；`[验证]`=本轮实际运行（§8）；同一断言两者都给时取更强证据。
> 列“验证方式”中的 §8 编号对应第 8 节脚本输出段。

| 编号 | 断言 | 证据 path:line | 验证方式 | 置信度 |
|---|---|---|---|---|
| A01 | 服务端常量 SR/FPS/WINDOW/HOP = 16000/25/32000/8000，GUARD/FADE=1280/320 | `studio/server.py:28-29` | [代码] + §8-B0 打印 | 高 |
| A02 | `GUARD=1280` 样本 = 80 ms，`FADE=320` = 20 ms，输出块 8000 = 0.5 s | `server.py:29,80-81`；`falcon_runtime.py:24-27` | [验证] §8-B0 算术 | 高 |
| A03 | 一次请求必须是 32000 个 `<f4` 采样（b64 恰好 170668 字符），50 帧 96×96 JPEG/PNG | `server.py:43-63` | [验证] §8-B0 valid/invalid 表 | 高 |
| A04 | 请求 `end_sample` 缺省 32000，允许范围 `[32000, 2^53−1]`，bool/float 拒绝 | `server.py:64-66` | [验证] §8-B0 | 高 |
| A05 | `window_seconds`/`hop_seconds` 若出现必须严格 2/0.5 | `server.py:38-39` | [验证] §8-B0 | 高 |
| A06 | 结果块长度恒 8000 样本，b64 42668 字符，sample_rate=16000 | `server.py:262-264` | [验证] §8-B5 | 高 |
| A07 | 结果 `start_sample=endpoint−9280`、`end_sample=endpoint−1280` | `server.py:266-267,80-81` | [验证] §8-B5（→22720/30720） | 高 |
| A08 | `rtf=inference_ms/500`、`window_rtf=inference_ms/2000` | `server.py:264` | [验证] §8-B5（25→0.05/0.013） | 高 |
| A09 | 结果 `end_sample` 与请求 `end_sample` 含义不同（前者是输出块末端） | `server.py:64,266`；`web/app.js:678-681` | [代码]+[验证] | 高 |
| A10 | `OutputSlicer` 取 `pred[22720:30720]`，`future=pred[30720:31040]` | `server.py:80-86` | [验证] §8-B1 数值 | 高 |
| A11 | 相邻窗（endpoint+HOP）输出块在绝对时间轴连续、不重复（allclose max 4.3e-8） | `server.py:80-87` | [验证] §8-B1 | 高 |
| A12 | 连续窗口首 20 ms 用上一窗 `future` 与新预测交叉淡化（ramp 0→1） | `server.py:82-86` | [验证] §8-B1 | 高 |
| A13 | `contiguous` 仅当未 reset 且 `endpoint−last_endpoint==HOP` | `server.py:82` | [验证] §8-B1 | 高 |
| A14 | 输出被 clip 到 [-1,1] 并统计 `clipped_samples` | `server.py:88-89,267` | [验证] §8-B1（8000） | 高 |
| A15 | 非 32000 或含 NaN 的预测被拒绝 | `server.py:77-79` | [验证] §8-B1 | 高 |
| A16 | HTTP 同源检查覆盖所有路由；无 Origin 放行；非同源→403 | `server.py:198-202,175-179` | [验证] §8-B2/B4（evil→403，10 case 全对） | 高 |
| A17 | WS 在 accept 前同源检查，失败 close 1008 | `server.py:235-237` | [验证] §8-B5 | 高 |
| A18 | `TrustedHostMiddleware` 白名单 4 项；Host=evil.example→400 | `server.py:196` | [验证] §8 补充（200/400 表） | 高 |
| A19 | 响应带 CSP/nosniff/no-referrer/Permissions-Policy/no-store | `server.py:204-210` | [验证] §8-B4 打印 | 高 |
| A20 | `/api/load` body>1024→413；bad json/非对象→400；成功→200 | `server.py:219-231` | [验证] §8-B4（400/413/200） | 高 |
| A21 | `/api/load` 遇到 `RuntimeError`→409 | `server.py:229-230` | [验证] §8-B6 | 高 |
| A22 | WS 单条消息 >2,000,000 字符 → close 1009；恰 2,000,000 得到 JSON error 且连接存活 | `server.py:243-246` | [验证] §8-B6 | 高 |
| A23 | WS `error` 不关闭连接，后续 result 正常 | `server.py:268-272` | [验证] §8-B5 | 高 |
| A24 | `id` 必填、bool/超 2^53/超 80 字符被拒 | `server.py:252-256` | [验证] §8-B5 + `tests/test_server.py:106-115` | 高 |
| A25 | 模型忙时 WS 返回 `{"type":"error","message":"模型忙，请跳过此窗口后重试。"}` | `server.py:258-259` | [代码]+[验证] 锁占用→RuntimeError 文案 | 高 |
| A26 | 无活跃连接才允许切换变体；加载中不允许切到别的变体；已加载同款 no-op | `server.py:130-141` | [代码] + `tests/test_server.py:118-127`（[验证] 33 passed） | 高 |
| A27 | `ready=not loading and runtime.ready`；加载失败时 error 经 `/api/status` 回传 | `server.py:115,154-158`；[验证] §8-B7 | [验证] | 高 |
| A28 | FalconRuntime 常量与变体表：small→Falcon.yml/falcon.pt/JusperLee/Falcon/FalconSpeakerContrastive；big 对应 Big 三项 | `falcon_runtime.py:31-37` | [验证] §8-A1 | 高 |
| A29 | `variant_for` 接受 small/big 与两个 HF 短链，其余 ValueError | `falcon_runtime.py:40-45` | [验证] §8-A1 | 高 |
| A30 | 权重解析顺序：显式本地路径→原样；`hf://repo/file`→拆分下载；缺省→VARIANTS 的 repo/weights；无环境变量可覆盖 | `falcon_runtime.py:58-78` | [验证] §8-A1（本地路径、坏 hf://）；`tests/test_weights.py` 7 项通过 | 高 |
| A31 | `weights_source` 只查缓存不联网，返回 cached/huggingface/None | `falcon_runtime.py:48-55` | [验证] §8-A1（本机 huggingface） | 高 |
| A32 | `pick_device auto` 顺序 CUDA>MPS>CPU；显式不可用报错；非法值 ValueError | `falcon_runtime.py:81-92` | [验证] §8-A1 + `tests/test_device.py` 通过 | 高 |
| A33 | `preprocess_mouth` 输入必须 uint8 (50,96,96)，输出 (1,1,50,88,88) f32，96→88 中心裁剪（外圈 4 px 丢弃） | `falcon_runtime.py:95-104` | [验证] §8-A4（外圈改写不影响输出） | 高 |
| A34 | 归一化是 `(x/255−0.421)/0.165`，值域约 [−2.5515, 3.5091]，**不是 [-1,1]** | `falcon_runtime.py:101-104`；`look2hear/datas/transform.py:161-167` | [验证] §8-A4 | 高 |
| A35 | 运行时构建 small 音频 537,697 / video 11,186,688 / 总 11,724,385；big 1,158,617 / 11,186,688 / 12,345,305 | `falcon_runtime.py:157-158`；配置见 §4.10 | [验证] §8-A2 | 高 |
| A36 | Studio 模型 state_dict 与训练侧 key 集合+形状完全一致（audio 525 / video 137）；可 strict 互换 | `falcon_runtime.py:162-169`；`look2hear/models/Falcon.py` | [验证] §8-A3 | 高 |
| A37 | `FalconSpeakerContrastive` 与 `SpeakerContrastiveV2` 在**同一配置**下结构完全相同、可 strict 互装 | `Falcon_Contrastive.py:60,94-118`；`Speaker_Contrastive_v2.py:78,112-136` | [验证] §8-A6 附加 | 高 |
| A38 | 严格加载失败：缺 video 前缀→ValueError；改名 key→RuntimeError；small 权重装 big→RuntimeError；文件缺失→FileNotFoundError | `falcon_runtime.py:143-144,164-169` | [验证] §8-A7 | 高 |
| A39 | ckpt 允许 `{"state_dict": ...}` 或直接前缀 state_dict（`ckpt.get("state_dict", ckpt)`） | `falcon_runtime.py:161` | [验证] §8-A7（plain 通过） | 高 |
| A40 | `torch.load(..., weights_only=True)` 避免执行任意对象 | `falcon_runtime.py:159-160` | [代码] | 高 |
| A41 | `_warmup` 跑真前向（zeros audio + 107 灰度帧）后才 ready；GPU auto 失败降级 CPU 并写 warning | `falcon_runtime.py:185-196,202-205` | [代码]+[验证] §8-A5（warmup_ms≈720/821） | 高 |
| A42 | CPU 线程数 `max(1,min(8,FALCON_CPU_THREADS or 4))` | `falcon_runtime.py:174-176` | [验证] §8 补充（2→2、999→8、0→1） | 高 |
| A43 | `infer` 只做 f32/形状/finite 校验；frames 形状由 `preprocess_mouth` 校验；输出 `(32000,)` f32 且有限，否则 RuntimeError | `falcon_runtime.py:243-252,97-100` | [验证] §8-A8 | 高 |
| A44 | 前向形状链：mouth(1,1,50,88,88)→FRCNN(1,512,50)→Identity；audio(1,32000)→(1,112,251,129)→…→(1,1,32000) | `falcon_runtime.py:237-246`；`Falcon.py:1286-1345` | [验证] §3.5 | 高 |
| A45 | `falcon_model/models/Falcon.py` 相对 `look2hear/models/Falcon.py` 净 +9 行，全部来自 DropPath import + Metal FFT bridge（+4/+4/+1），无结构改动 | `studio/.../Falcon.py:6,1110-1122,1183-1198`；`look2hear/.../Falcon.py:6` | [验证] §8-A diff 归因 | 高 |
| A46 | `SISNRRepel.py`(1498) = Studio `Falcon.py`(1496) + 2 行注释 import + 类名替换；`Speaker_Contrastive_v2.py`(140) = `Falcon_Contrastive.py`(122) + 18 行 docstring + base 类名 | `SISNRRepel.py:5,9,1288`；`Speaker_Contrastive_v2.py:1-17,23,78` | [验证] §8-A diff | 高 |
| A47 | 两份 big 文件在 `look2hear/` 中无源（find/grep/git 均无）→ PROVENANCE 来源表对此 2 项不准确 | `PROVENANCE.md:6-11`；仓库仅 2 commit | [验证] §8-A3 附加（`git log --all -S 'class SISNRRepel' -- look2hear`） | 高 |
| A48 | `Falcon_Contrastive.py`、`activations.py`、`resnet.py`、`shufflenetv2.py` 与 look2hear 逐字节相同；`normalizations.py` 仅注释；`frcnn_videomodel.py` 仅删 3 行 import/path hack + 头注释 + CRLF→LF | `studio/falcon_model/**` | [验证] §8-A（cmp/diff --strip-trailing-cr） | 高 |
| A49 | 同权重同输入下 Studio 运行时与训练侧类 CPU 前向 bit-identical（max\|Δ\|=0.0）——合成权重版 | `falcon_runtime.py:237-246` vs `look2hear/models/Falcon_Contrastive.py:94-118` | [验证] §8-A6 | 高（真实 falcon.pt 未验证） |
| A50 | FalconBig 与 small 的差异只在宽度类超参（§4.10 表）；video 完全同构 | `configs/FalconBig.yml`；`configs/Falcon.yml` | [验证] §8-A2 + 配置 flat-diff | 高 |
| A51 | `infer_file` 一次前向跑完整段，允许任意长度（≥25 帧），输出长度 = 帧数×640 | `infer_file.py:69-82` | [代码]（`tests/test_infer_file.py` 因无权重 skip） | 中（未跑） |
| A52 | Studio 服务端不写盘；只有 `infer_file.py` 写 WAV | `server.py:1`；`infer_file.py:41-44,100` | [代码] | 高 |
| A53 | `mode` 字段加载前为 `sliding-window`、加载后被覆盖为 `windowed_audio_visual` | `server.py:112,115`；`falcon_runtime.py:234` | [验证] §8-B3/B7 | 高 |
| A54 | 在线 demo（公网地址）与 `same_origin`/`TrustedHost` 白名单矛盾——本文件服务无法直接对外 | `server.py:175-179,196`；`README.md:16` | [代码]+[推断] §4.6 | 中（部署分支未知） |
| A55 | 4 个 `__init__.py` 不导出符号；全部使用相对/绝对子模块导入 | `studio/falcon_model/*/__init__.py` | [代码] | 高 |
| A56 | 多连接并发：服务端不排队；`busy` 检查与 `infer_lock` 非阻塞获取之间存在“busy=False 但锁未释放”窗口，错误文案两种（模型忙 / 另一个窗口） | `server.py:163-164,169-172,258-259`；`web/app.js:565,588` | [代码] + §4.11.1 | 高 |
| A57 | 推理无请求级超时：`asyncio.to_thread(engine.infer, ...)` 未包 `wait_for`；只能等前端 60 s 定时器或进程退出 | `server.py:260`；`web/app.js:589-591` | [代码] + §4.11.3 | 高 |
| A58 | 每个 `result` 回包都调用 `engine.status()` → 重建 `variants()` → 每 variant 2 次 `weights_source()` 缓存查询（0.5 s 热路径） | `server.py:265,118-124,122`；`falcon_runtime.py:48-55` | [代码] + §4.11.2 | 高 |
| A59 | `variants[].available = weights_source() is not None`（缓存/hub 已装/hub 缺失三态）；缺 hub 时 `/api/load` 返回 400 且错误文案误报“缺权重文件” | `server.py:118-124,140-141`；`falcon_runtime.py:48-55` | [验证] §8-B3/B6 + §4.11.4 | 高 |
| A60 | `TrustedHost` 对 WS 表现为握手拒绝（TestClient `WebSocketDenialResponse`），不是普通 400 响应 | `server.py:196` | [验证] R2 评审 §2.3-8（本文件 §8.4） | 高 |
| A61 | `infer_file.read_wav` 的 32-bit float 路径不 clip、不校验范围；与 WS 路径 `\|amp\|≤4` 不对称 | `infer_file.py:34-38`；`server.py:49` | [代码] + §4.11.6 | 高 |
| A62 | 模块级 `app = create_app()` 在 import 时创建 Engine（不 autoload）；`main()` 再建一个 → “双 Engine”，测试 import 有副作用但无害 | `server.py:287,332` | [代码] + §4.11.7 | 高 |
| A63 | 仓库内 `server.py` 不能直接对外部署：`same_origin` 与 `TrustedHost` 拒绝远端 Origin/Host；在线 demo 必须是部署补丁 | `server.py:175-179,196`；`README.md:21`；`studio/README.md:16` | [验证] `same_origin(公网)=False`、Host=公网→400（评审日志） | 高 |
| A64 | 协议坐标 `P` 与绝对采样坐标 `s` 的关系是 `P = s + 22720`（与 `sessionEpoch` 无关）；首窗请求 `end_sample = sessionEpoch + 32000`（`sessionEpoch=0` 时才恰为 32000） | `web/app.js:593,637`；`server.py:64-67,266` | [代码] + [验证] §8-B5（sessionEpoch=0 两窗）+ R4 执行级 §6-E3 | 高 |
| A65 | 单 WS 连接内请求-响应串行：`while True` 中 `receive_text` → 处理 → `send_json` 后才读下一条（不会交错回包） | `server.py:242-267` | [代码] + §5.3 | 高 |
| A66 | `2^53−1` 上限只由服务端校验；前端用 JS `number` 递增 id/时间戳且不检查该上限（服务端兜底） | `server.py:65,255`；`web/app.js:588,593` | [代码] + §5.3 | 高 |
| A67 | 同仓库两种 `torch.load` 安全姿态：Studio `weights_only=True`（`falcon_runtime.py:160`），评测 `weights_only=False`（`audio_visual_test.py:119`） | `falcon_runtime.py:160`；`audio_visual_test.py:119` | [代码] + §4.3 | 高 |
| A68 | `result` 路径在 `send_json` 前同步调用 `engine.status()`；若 `variants()/weights_source()` 抛异常，经 `except Exception` 表现为“推理失败”通用错误 | `server.py:265,270-272` | [代码] + §4.11.2 | 高 |

---

## 7. 未解疑问 / 需其它子系统或 Lead 确认

**P0（事实/可复现性级）**

- **Q1（需 Lead/HF 权限）**：真实 `falcon.pt`/`falcon-big.pt` 的键集合、形状、文件内元数据（`state_dict`/架构名/
  epoch）与“CPU bit-identical（max diff 0）”无法验证——本环境无权重且明令禁止下载。本文只能证明
  “同权重下代码路径 bit-identical + 参数量 11,724,385 精确吻合”。R4 若允许，建议仅下载一次权重到隔离目录做
  严格加载 + 与训练代码逐点对比（不跑数据）。
- **Q2（需 model-core/Lead）**：PROVENANCE 声称 `falcon_model/` 全部来自 `look2hear/`，但
  `models/SISNRRepel.py`、`models/Speaker_Contrastive_v2.py` 在发布训练仓库中**不存在**（仓库只有 2 个 commit）。
  这两份 Big 代码的真实来源（作者私有训练仓库？后续分支？）需要作者/Lead 确认，文档表述应改为
  “Falcon/small 拷贝自 look2hear；Big 两份为 Studio 自带（训练侧未发布）”。
- **Q3（需 loss-metric-eval / 04 域）**：`audio_visual_test.py:129-132` 用
  `getattr(look2hear.models, conf.train_conf.audionet.audionet_name)` 构建模型，而
  `Experiments/checkpoint/Falcon/conf.yml:88` 是 `SpeakerContrastiveV2`（look2hear 无此类）→ 该评测脚本对仓库内
  实验快照会 `AttributeError`。需确认：发布评测流程是否另有脚本/分支，或这是本 fork 的回归。
  同时 `infer_file.py:12` “whole clip in one pass, matching the evaluation protocol”是否与评测 dataloader 的
  segment 行为一致，也需 04/05 域确认。

**P1（重要遗漏/误导性行为）**

- **Q4（已由 §4.11.9 回答；保留“部署方确认”项，不再计入未解问题）**：`README.md:16` 的在线 demo 与外网访问，与 `same_origin`（hostname 必须 localhost）
  和 `TrustedHostMiddleware` 白名单冲突 → 线上必然运行了另一份 server 补丁；仓库内代码不能复现该部署，
  应在文档/README 中说明。
- **Q5（需 studio-frontend / Lead 产品口径）**：服务端 `busy` 错误文案建议“跳过此窗口”，但前端对所有
  `type:"error"` 一律 `throw` → 停止整场会话（`web/app.js:672,721-724`）。多标签页/open 两个 WS 时，非持有
  推理锁的页面会遇到该错误并停止。要么前端把 busy 当可跳过、要么服务端改排队/拒绝新连接。**这是前后端契约
  层面的真实不一致**（07 域 Q5 与本文互证）。
- **Q6（需 ops-tests-docs / 08 域）**：`requirements.txt:3` `torch>=2.14.1`/lock `torch==2.14.1` 与本文验证环境
  torch 2.8 不同；`numpy>=2.4.6`、`Pillow>=12.3.0` 同理。本文所有 [验证] 都在 torch 2.8/numpy 2.2/无 Pillow 下完成，
  **新版本上是否有 API 行为差异未验证**（尤其 `torch.stft/istft`、`torch.load(weights_only=)`、Starlette 版本）。
- **Q7（需 ops-tests-docs）**：`requirements.txt:6` 的 `einops` 在运行时代码中未被 import（仅 doctor 检查），
  属死依赖；`Pillow` 是 `server.py` 的硬依赖（无它服务无法启动）却在当前验证环境缺失。
- **Q8（需 Lead 产品口径）**：`server.py:120-121` 注释与 `:141` 错误文案引用 `models/<file>` 本地权重目录，
  但代码中**没有**任何“本地 models/ 优先”路径（`resolve_checkpoint` 只会 hf 下载或用户显式 `--checkpoint`），
  仓库也没有 `studio/models/`。该文案**可达但仅在缺 `huggingface_hub` 时**：此时 `weights_source()` 返回 None →
  `variants[].available=false` → `start_load` 抛这条 ValueError（`server.py:140-141`），于是“缺 Python 包”被误报成
  “缺权重文件”。建议区分两种原因；或实现本地 `models/` 优先探测。

**P2（细节/待确认）**

- **Q9（需前端/追踪域）**：服务端把任意 RGB JPEG 转 `L`（PIL ITU-R 601-2）；训练数据是数据预处理生成的灰度裁剪。
  两者灰度化公式/量化是否完全一致未验证（不影响协议，可能轻微影响音质）。
- **Q10（需 07 域）**：`result` 中的 `clipped_samples`/`window_rtf`/`device`，以及 `/api/status` 中的
  `warmup_ms`/`parameters`/`weight keys`/`warning` 前端均未消费（诊断用途）。若产品需要“GPU 降级 CPU”提示，应显示 `warning`。
- **Q11（需 07 域）**：`LEFT_CONTEXT` 应为 **22720**（`app.js:10` = `32000−8000−1280`），与后端切片起点
  `server.py:80` 完全一致；任务书/Lead 提示中的 22400 无代码依据（07 文档 Q1 同结论）。
- **Q12（需 model-core）**：`Falcon.forward` 在 `training and near_audio is not None` 分支调用不存在的
  `self._mdcl_loss`（`Falcon.py:1489`，类中只有 `_mdcl_loss_hard`）：`hasattr(Falcon,'_mdcl_loss')==False`。
  该分支在 Studio 推理（eval）不可达，且 `FalconSpeakerContrastive`/`SpeakerContrastiveV2` 覆写了 forward；
  但若有人用 `Falcon` 直接训练会 AttributeError（与 01 域结论一致，记录为跨文档一致项）。
- **Q13（需 Lead/论文口径）**：README“0.5 M trainable parameters”与实际 audio 537,697（0.538 M）差 7.5%；
  是取整/口径（是否含 spk 头？）问题，建议 FINAL 文档统一说明。
- **Q14（观察）**：`Engine.status()["device"]` 加载前是“请求值”（`auto`）、加载后是“实际值”
  （`cpu`/`cuda`/`mps`），同名字段语义变化；前端大写显示导致用户先看到 “AUTO” 再变 “CUDA/CPU”。
- **Q15（观察）**：`MODEL_CLASSES` 支持两个类，但 `VARIANTS` 把类名写死（`architecture`），
  切换变体时若配置与类名不符会在 `:152-154` 抛 `ValueError("Unexpected architecture for ...")`——这是有意的
  “配置/权重配套”保护，不是 bug。

### 7.1 对 R2 评审（`studio-frontend`）意见的逐条响应（R3 修订）

被评审版本：`06` R1 1025 行版；评审文件 `docs/reviews/round1/06-studio-backend-review-by-studio-frontend.md`
（结论：P0=0、P1=1、P2=9、P3=3；55 条断言 54 成立、A54 无法独立确认）。**全部采纳**（P2-5 的 `out_proj`：R2 评审所据 1025 行版为 `384-388`，R3 核对源码为 `380-383`）：修订后本文件为 v2。

| 评审编号 | 评审意见 | 处置 | 修订位置 |
|---|---|---|---|
| P1-1 | §5.2 把“无可用权重”归 409，实际 `ValueError→400` | 采纳：409 行删除“无可用权重”，单列 400 行并给触发条件 | §5.2、§5.2 守卫⑥ |
| P2-1 | `git log --all -S 'class SISNRRepel'` “均无”不准确 | 采纳：改为 `... -- look2hear` 无结果；说明不带路径会命中 `a7af472`（Studio 文件在该 commit 加入） | §4.8(d) P1、A47、§8.1#5 |
| P2-2 | Q8“该文案永远不可达”与代码不符 | 采纳：改为“可达，但仅在缺 `huggingface_hub` 时；文案把缺包误报成缺权重” | §7-Q8 |
| P2-3 | `normalizations.py` diff 28 应为 26 | 采纳：三处统一 `13+/13− = 26` | §2.4、§4.8(a)、§8-A3 |
| P2-4 | §1.1 FRCNN 输入写成 96×96，实际模型输入 88×88 | 采纳：改 88×88，并注明 96×96 只是协议/采集尺寸 | §1.1 |
| P2-5 | 三处行号漂移 | 部分已先行修正（`out_proj` 380-383）；其余采纳：`NeuroSCNetFusion` 239-246、`audio_visual_test.py` 129-132 | §4.10、§4.9-6 |
| P2-6 | §2.4 “16 个文件/4035 行”口径不一致 | 采纳：标题改为“15 个 `.py` 4035 行 + `PROVENANCE.md` 42 行 = 4077 行” | §2.4 |
| P2-7 | CSP 摘录不完整且未标注节选 | 采纳：给全量 CSP（含 style/img/media/worker） | §4.6 |
| P2-8 | §5.5 坐标定义混用（协议坐标/采样坐标） | 采纳：新增“协议坐标 vs 行内采样坐标”对照表 + 两窗数值例（32000/40000 → 22720/30720/38720 → [0,8000)/[8000,16000)） | §5.5 |
| P2-9 | §2.2 `.load()` 步骤顺序 | 采纳：按源码 `:142-196` 顺序重写（strict→统计参数→set_num_threads→eval/freeze→pick_device→warmup） | §2.2 |
| P3-1 | 36.5 KB 应 ≈36.6 千字符并扣 JSON 开销 | 采纳 | §5.3 |
| P3-2 | `inference_ms`“纯前向”不精确 | 采纳：改“含 `preprocess_mouth` 的前向耗时” | §5.1、§5.4 |
| P3-3 | 4 s 轮询与 startSession 1.5 s 等待混写 | 采纳：区分全局 4 s 轮询与 `startSession` 1.5 s 等待循环 | §4.7 |
| §4 遗漏 1–7 | 并发/竞态、热路径 status、无超时、available 语义、WS 握手拒绝、infer_file float 不对称、双 Engine | 采纳：新增 §4.11.1–4.11.7 + 失败注入矩阵 §4.11.8 + 部署结论 §4.11.9 | §4.11 |
| A54 | 在线 demo 冲突无法独立确认 | 同意：保留 [推断] 标注；并在 §4.11.9 明确“仓库内 `server.py` 不能直接对外部署，需补丁清单” | §4.11.9、§7-Q4 |

**对评审 §6 的 8 个问题逐一回答：**

1. **Q1（P1-1 409→400）**：采纳。409 只保留 `RuntimeError`（加载中切变体/有活跃连接）；“无可用权重”单列为 400，
   触发条件写入守卫⑥（`weights_source()` 为 None ⇔ 缺 `huggingface_hub`）。
2. **Q2（坐标定义）**：采纳，§5.5 已给两套坐标换算表与两窗数值例；首个 wav 的 1.42 s 合成左上下文属于协议坐标。
3. **Q3（证据口径）**：采纳：`normalizations` 26 行、`out_proj` 380-383、`NeuroSCNetFusion` 239-246、
   `audio_visual_test.py` 129-132、`falcon_model` 计数口径。另确认评审指出的“28 行”源于我早期用
   `diff | grep -c '^[+-]'` 把 `---/+++` 头计入了 2 行。
4. **Q4（§1.1 形状）**：采纳，改为 88×88（模型输入）；协议/采集是 96×96。
5. **Q5（并发/缓存 status）**：采纳补 §4.11.1/§4.11.2；把 `device/variants` 缓存到加载完成时属代码优化建议
   （`server.py:265` 每个 result 都重建一次 `status()`），本轮遵守“不改源码”约束，仅记录为 [推断] 优化项。
6. **Q6（infer_file float 不 clip）**：[推断] 属“信任离线输入（32-bit float WAV 约定已归一化）”的设计，
   已在 §4.11.6 记录不对称；建议后续给 `read_wav` 加范围校验（P2，未改源码）。
7. **Q7（真实权重验证）**：当前约束（无权重且禁止下载）下做不了；R4 若 Lead 授权，建议仅下载一次到隔离目录
   做 strict load + 训练侧/Studio 逐点对比（不跑数据）。FINAL 措辞保持：“合成权重下代码级 bit-identical
   （max|Δ|=0），真实 `falcon.pt` 未验证”（§4.9-6、A49 置信度列、§7-Q1）。
8. **Q8（与 08 的 test_server 口径统一）**：已在 R2 评审 `08` 时提出 P1-2（PIL stub + `-p no:typeguard`
   可跑通全部 4 个 pytest 文件 = 33 passed/4 skipped）；本文件 §8 已声明 stub 边界与“真实 JPEG/PNG 解码未验证”。
   建议 FINAL 统一为：“缺 Pillow 时需注入最小 stub 才能跑协议测试；真实图片解码未验证；测试本身在完整 Studio
   环境可直接运行”。

> 另：评审 §4 提到的“多 WS 连接背压矩阵/失败注入矩阵/部署形态”已并入 §4.11；评审 §2.3 的独立复跑
> （pytest 33 passed/4 skipped、OutputSlicer 数值、state_dict/bit-identical、形状链、diff 统计）与本文件结果一致，
> 作为 R3 的交叉验证记录在 §8.4。

### 7.2 对 R4 复核（`studio-frontend` round2）意见的响应（R4 修订）

复核文件：`docs/reviews/round2/06-studio-backend-round2-review-by-studio-frontend.md`
（结论：**通过**，P0=0、P1=0、P2=2、P3=3；R2 的 P1×1/P2×9/P3×3 全部修复经其逐条复核；执行级验证 E1–E5 全 PASS）。
本轮（v3）对 R4 新发现全部处理：

| R4 编号 | 意见 | 处置 | 位置 |
|---|---|---|---|
| P2-R1 | §5.5 双坐标原点写成 `sessionEpoch`，一般情形不成立 | 采纳：坐标系改为“**绝对采样坐标**（打开设备起）+ **协议坐标**”，换算 `P = s + 22720` 与 `sessionEpoch` 无关；补 `sessionEpoch=160000` 对照数值例（首窗 192000 → 182720/190720 → 绝对 [160000,168000)）；前端 bullet 同步修正 | §5.5、§5.9 关联说明 |
| P2-R2 | 图/例未注明 `sessionEpoch=0` 前提 | 采纳：图上方明确“首窗协议窗口 = `[sessionEpoch, sessionEpoch+32000)`；图中数值仅对应 sessionEpoch=0 示例” | §5.5 |
| P3-R1 | Q4 未收口（§4.11.9 已答仍列 P1） | 采纳：Q4 标注“已由 §4.11.9 回答；保留部署方确认项，不计入未解问题” | §7-Q4 |
| P3-R2 | §7.1 的 `out_proj` 溯源表述与所据版本不符 | 采纳：改为“R2 评审所据 1025 行版为 384-388，R3 核对源码为 380-383” | §7.1 |
| P3-R3 | §4.11.1 `ws_max_queue` 语义写成事实 | 采纳：标 `[推断]` 并注明验证环境 uvicorn 0.54.0 / 依实现版本 | §4.11.1 |
| §4 建议 1 | 补 `sessionEpoch>0` 端到端数值例 | 采纳（同 P2-R1） | §5.5 |
| §4 建议 2 | 补“同连接内请求-响应串行” | 采纳：§5.3 新增单连接顺序保证（`server.py:242-267`） | §5.3、A65 |
| §4 建议 3 | 补 `2^53−1` 上限由服务端兜底、客户端不检查 | 采纳：§5.3 新增“上限兜底” | §5.3、A66 |
| §4 建议 4 | 补 `weights_only` 两种安全姿态对比 | 采纳：§4.3 补 Studio `True` vs 评测 `False`（`audio_visual_test.py:119`），归 04 域确认 | §4.3、A67 |
| §4 建议 5 | 补 `status()` 异常放大路径 | 采纳：§4.11.2 补“设备查询失败会表现为推理失败” | §4.11.2、A68 |

> R4 执行级验证（评审方 E1–E5：常量矩阵 5/5、前端 `encodeFloat32`→`decode_request` bit-exact 往返、
> 双窗协议链 max|Δ|=4.28e-8、部署矛盾 same_origin/400/403、R3 静态抽查）与本文件 §8 结果一致，
> 交叉记录见 §8.4；本轮修订为文档-only（未改源码/脚本）。

---

## 8. 本轮验证活动记录

**环境/命令约定**：`cd /root/code/github_repos/Falcon-fork`；解释器 `/root/.venvs/falcon/bin/python`（3.10.18，
torch 2.8.0+cu128，CUDA 可用 2×A800，**Pillow 缺失**，无 Falcon 权重）。协议层需要 Pillow，因此用仓库内最小替身
`docs/verification/scripts/pil_stub/PIL`（仅模拟 server 用到的 `Image.new/open/save/convert`，**不测试真实
JPEG/PNG 解码**）。系统 `python3`=3.8 无 torch/fastapi，未使用。

### 8.1 命令清单

| # | 命令 | 目的 |
|---|---|---|
| 1 | `diff -u look2hear/models/Falcon.py studio/falcon_model/models/Falcon.py` | 1496 vs 1487 差异归因 |
| 2 | `diff -u --strip-trailing-cr look2hear/videomodels/frcnn_videomodel.py studio/falcon_model/videomodels/frcnn_videomodel.py` | FRCNN 差异（CRLF 干扰） |
| 3 | `cmp` 逐对比较 `Falcon_Contrastive.py/activations.py/resnet.py/shufflenetv2.py`；`diff` `normalizations.py`/`base_model.py`/`SISNRRepel.py`/`Speaker_Contrastive_v2.py` | PROVENANCE 文件级核验 |
| 4 | `diff configs/Falcon.yml studio/configs/Falcon.yml` | 根配置与 Studio 配置一致性（0 行） |
| 5 | `grep -rn "SISNRRepel\|SpeakerContrastiveV2"` + `git log --all -S 'class SISNRRepel' -- look2hear`（限定路径） | 训练仓库是否存在 Big 代码（不限定路径会命中 a7af472 的 studio 文件） |
| 6 | `PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_studio_backend.py` | 主验证脚本（A 运行时 + B 协议），日志 `docs/verification/runs/studio-backend-verify.log` |
| 7 | `PYTHONPATH=<pil_stub>:<stubs>:$PWD /root/.venvs/falcon/bin/python -m pytest studio/tests/test_server.py studio/tests/test_weights.py studio/tests/test_device.py studio/tests/test_infer_file.py -q -p no:typeguard` | 仓库自带 Python 测试 |
| 8 | `FALCON_CPU_THREADS={2,999,0} ... FalconRuntime('small', fake_ckpt).load('cpu')` | 线程数夹取验证 |
| 9 | `mimetypes.guess_type('a.wasm')`、Starlette `parse_host_header`/`TrustedHostMiddleware` 源码、`TestClient` 注入不同 Host 头 | 静态资源 MIME 与 Host 校验细节 |
| 10 | 手写脚本：逐层调用 `encoder/audio_bottleneck/video_bottleneck/separator/mask_generator/decoder/pool_spk_embed` 打印形状；同一 small 配置构造 `FalconSpeakerContrastive` 与 `SpeakerContrastiveV2` 并 strict 互装 | §3.5 形状链与类互换（A37/A44） |

### 8.2 关键输出摘要

- **A1**：`VARIANTS` 正确；`variant_for` 三值 ok，坏值 ValueError；`weights_source` 均 `"huggingface"`（缓存空）；
  本地路径原样返回；`hf://model.pt` → ValueError；`pick_device auto→cuda`、`cpu→cpu`、`tpu/gpu/None`→ValueError。
- **A2**：small `audio=537,697 video=11,186,688 total=11,724,385`；big `audio=1,158,617 total=12,345,305`；
  两者 `video_bottleneck` 0 参数/Identity；构造后原 config 的 `repeats/shared` 仍在（deepcopy 生效）。
- **A3**：`studio vs training: keys_equal=True shapes_equal=True n_studio=525 n_training=525`（audio），
  video 137/137；`Falcon_Contrastive.py` cmp 相同；`resnet/shufflenetv2/activations` cmp 相同；
  `normalizations` 仅注释差异（13+/13− = 26 行，净 0）；`frcnn` 去 CRLF 后 3 行 import/path + 头注释。
- **A4**：`preprocess_mouth` → `(1,1,50,88,88)` f32；gray107 → −0.008437；gray0/255 → −2.5515/3.5091；
  外圈 4 px 改写不影响输出（中心裁剪证毕）；49 帧/88×88/float32 全被 ValueError 拒绝。
- **A5**：合成 ckpt 加载：small `ready=True device=cpu audio_keys=525 video_keys=137 parameters=11,724,385
  warmup_ms=720 load_s=1.25`，`infer` → `(32000,) f32 finite last_inference_ms=536.5`；
  big `audio_keys=517 parameters=12,345,305 warmup_ms=821`，`infer` → `(32000,) f32, 766.7 ms`。
  （注入权重为随机初始化，仅验证流程/形状，不代表音质/真实延迟。）
- **A6**：`FRCNNVideoModel.eval()` 返回 None（验证 `falcon_runtime.py:178` 注释）；small
  `max|studio−training| = 0.000e+00 (bit-identical=True)`（训练侧类 + 同权重）；big 同配置自一致 0.0
  （训练仓库无 `SpeakerContrastiveV2`，只能自一致）。另：同一 small 配置下
  `FalconSpeakerContrastive` vs `SpeakerContrastiveV2` keys/shapes 全等、参数量均为 537,697、可 strict 互装。
- **A7**：`音频键-only→ValueError("Checkpoint must contain both audio_model and video_model weights.")`；
  `改名 key→RuntimeError(Error(s) in loading state_dict...)`；`small 权重装 big→RuntimeError`；
  `缺文件→FileNotFoundError`；`无 {"state_dict":} 包裹的带前缀 state_dict→接受`。
- **A8**：未加载→`RuntimeError: Falcon model has not finished loading.`；f64/长度不对→ValueError；
  NaN→`ValueError: Audio contains non-finite samples.`；frames 非法→`ValueError: Mouth frames must be a uint8 NumPy array.`
- **B0**：常量 16000/25/32000/8000/1280/320/2_000_000/9007199254740991；
  `start=22720 end=30720`；valid decode → `(32000,) f32 (50,96,96) u8 endpoint=32000`；缺省 end_sample=32000；
  audio b64=170668；13 个非法请求全部 ValueError（消息与 §5.3 一致）。
- **B1**：首块 `n=8000 contiguous=False first=0.2272 last=0.30719`；第二块 `contiguous=True`，首样本 0.3072
  （=future 值），fade 后 0.3104；两块拼接 vs 绝对时间轴 `allclose True max|diff|=4.3e-8`；
  跳窗 `contiguous=False`；clipping `max=1.0 clipped_samples=8000`；短/NaN 预测 ValueError。
- **B2**：`same_origin` 10 组真值表全部符合预期（含 ipv6、端口不一致、host 字符串不一致、file://、evil 域名）。
- **B3**：fresh `Engine.status()` 基础字段（无 architecture/parameters），`variants[].source="huggingface"`；
  挂 FakeRuntime 后 ready=True；runtime+loading=True→ready=False；锁占用时 `infer` →
  `RuntimeError: 另一个窗口正在使用模型，请关闭重复会话。`；并记录 stale-error+ready=True 的状态瑕疵（Q/P2）。
- **B4**：`GET /api/status` 200；安全头齐全（CSP 全文见日志）；Host 头测试：`127.0.0.1:7860/localhost:7860/[::1]:7860/testserver`→200，`evil.example`→400；`POST /api/load {}`→200、bad json→400、`[]`→400、
  1025B→413；evil Origin→403；`GET /`→200、静态 index→200、mediapipe mjs→200、wasm→200 `application/wasm`。
- **B5**：result 字段 13 个、类型全表打印（`inference_ms` 在 stub 下为 int，真实为 float）；audio b64=42668→8000 样本；
  `rtf=0.05/window_rtf=0.013`；时间戳 22720/30720，第二窗 30720/38720；error schema `['id','message','type']`；
  非 json 消息 error 的 id=None；49 帧/bool id 的 error 文案正确；出错后 session 仍收 result（连接存活）；
  evil Origin WS → `WebSocketDisconnect`。
- **B6**：PNG 帧接受；`|amp|=4.0` 接受、4.0001 拒绝；恰 2,000,000 字符 → JSON error（会话存活）；
  >2,000,000 → close frame `{'type':'websocket.close','code':1009}`；`start_load` 抛 RuntimeError→409；
  真实 Engine `POST /api/load {"variant":"bogus"}`→400（在下载前被 ValueError 拦截，未联网）。
- **B7**：`Engine._load` 捕获异常 → `error="simulated checkpoint failure" loading=False`，`/api/status` 呈现；
  `FalconRuntime.status()` 加载后字段/类型全表（见日志）。
- **补充（命令 8/10）**：`FALCON_CPU_THREADS=2→torch.get_num_threads()=2`、`999→8`、`0→1`（夹取生效）；
  形状链见 §3.5；同一 small 配置下两类 keys/shapes 全等、参数量均 537,697、strict 互装成功。
- **pytest（命令 7）**：`33 passed, 4 skipped`（跳过：2 个 GPU 回退用例因本机有 CUDA，2 个 `test_infer_file` 因无权重）。
  说明：仓库自带 `test_server.py`/`test_weights.py`/`test_device.py` 在本环境可真实运行；Pillow 用 stub 替代。

### 8.3 失败/受限记录（诚实声明）

1. **Pillow 缺失**：`import server` 需要 `PIL`；首次尝试 `pytest` 因 `typeguard` 插件入口点报错，用 `-p no:typeguard`
   规避；图片解码路径用 stub 替代，**真实 JPEG/PNG 解码未验证**（Q9）。
2. **无权重/禁止下载**：任何 `from_pretrained`/默认 `resolve_checkpoint` 会联网下载，未执行；所有运行时验证用
   随机初始化 + 自造 `{"state_dict": audio_model.*+video_model.*}` ckpt（`/tmp/falcon-studio-verify-*`），
   真实 falcon.pt 的 key/输出一致性未验证（Q1）。HF 元数据查询被运行环境自动审查拒绝，未重试。
3. **CPU 延迟不代表产品**：本机 A800 CPU 上随机权重 small 单窗 ≈537 ms、big ≈767 ms（warmup 另计），
   属于 x86 服务器 CPU；README 的 M5 Pro 0.12/0.65 s 无法验证。
4. **`tests/test_infer_file.py` 2 项 skip**：需要已缓存 Falcon 权重；`infer_file` 逻辑仅做代码级确认。
5. **未验证真实 MediaPipe/浏览器路径**（07 域），也不涉及本域协议（服务端只收 JPEG/PNG base64）。

### 8.4 R2 评审核对（交叉验证，`studio-frontend` 独立复跑）

R2 评审（`docs/reviews/round1/06-studio-backend-review-by-studio-frontend.md` §2.3）在**同一验证环境**独立复跑并
与本文件结果一致，可作为 §8.2 的交叉验证：

| 项目 | 评审人独立结果 | 本文件 §8.2 结果 | 一致 |
|---|---|---|---|
| 4 个 pytest 文件（PIL stub + `-p no:typeguard`） | `33 passed, 4 skipped`（3.51 s） | `33 passed, 4 skipped`（4.17–5.77 s） | ✅ |
| `OutputSlicer` 数值 | chunk=8000、future=320、拼接 max\|Δ\|=4.281997678656779e-08、clip 后 clipped=8000 | 同（4.3e-8、8000/320） | ✅ |
| state_dict / 参数 | audio 525/525、video 137/137、keys/shapes/params 全等；small 537,697；big 1,158,617 | 同 | ✅ |
| 形状链 / bit-identical | FRCNN(1,512,50)、encoder(1,112,251,129)、…、decoder(1,1,32000)；`max\|studio−training\|=0.0` | 同 | ✅ |
| deepcopy 塌缩 | 同 dict 二次构造 537,697→49,081 | 同（引用 01 域结论） | ✅ |
| `Falcon.py` diff | 12 hunk、+65/−56、net +9、bridge +4/+4/+1 | 同 | ✅ |
| `normalizations.py` | 13+/13−=26 行 | 本文件 R1 写 28 行（含 `---/+++` 头 2 行）→ R3 已改 26 | 已修正 |
| TrustedHost 对 WS | 非法 Host → `WebSocketDenialResponse`（握手拒绝）；合法可连 | 本文件 §4.6 R3 已补“WS 侧为握手拒绝” | 已补充 |

> 结论：协议层与模型等价性两类核心 [验证] 经两名 agent 独立复现，结果一致；唯一差异是 R1 的
> `normalizations` 计数口径（已按评审修正）。

### 8.5 R4 复核记录（round2，`studio-frontend`）

R4 复核方在 v2（1201 行）上完成 5 项执行级验证并全部 PASS（见其评审 §6）：E1 前端/后端常量矩阵 5/5 相等、
`LEFT_CONTEXT=22720`=服务端切片起点；E2 前端 `encodeFloat32` → 服务端 `decode_request` 音频 **bit-exact**
（170668 b64 字符 → (32000,) f32，max|diff|=0）；E3 双窗协议链 32000/40000 → 22720/30720/38720 → 绝对
[0,8000)/[8000,16000)，拼接 max|Δ|=4.28e-8；E4 R3 静态抽查（26 行 diff、行号、CSP、计数）全对；E5 部署矛盾
（same_origin 公网=False、Host 公网→400、evil Origin→403）复现。回归 pytest 33 passed/4 skipped 一致。
本轮据其 P2-R1/P2-R2 与 3 条 P3 修订 §5.5/§7/§4.11 并新增断言 A64–A68（详见 §7.2）。

---

## 9. 变更日志（Changelog）

| 日期 | 轮次 | 变更 | 采纳的评审意见 |
|---|---|---|---|
| 2026-10-21 | R1 | 初版：完成 8 节模板；协议/状态机/窗口切片/运行时/模型拷贝 diff/配置差异/结论；断言表 55 条；§8 附可复现验证脚本与日志 | — |
| 2026-10-21 | R3 | v2 修订：采纳 `studio-frontend` R2 评审全部意见（P1-1 状态码拆分、P2-1…P2-9、P3-1…P3-3）并逐条响应（§7.1）；新增 §4.11 协议时序/并发/失败模式（8 小节 + 失败注入矩阵 + 部署结论）；§5.5 改为“协议坐标 vs 行内采样坐标”双坐标表 + 数值例；修正 26 行/行号/计数口径；交叉验证记录并入 §8.4 | `docs/reviews/round1/06-studio-backend-review-by-studio-frontend.md`（P0=0/P1=1/P2=9/P3=3，全部采纳） |
| 2026-10-21 | R4 | v3 修订：R4 复核（`studio-frontend` round2，P0=0/P1=0/P2=2/P3=3）全部处理——§5.5 双坐标改为“绝对采样坐标 + 协议坐标”（`P=s+22720`，不再以 sessionEpoch 为原点）+ `sessionEpoch=0/160000` 对照例；§5.3 补单连接串行与 2^53 兜底；§4.3 补两种 `torch.load` 姿态；§4.11.1 标 `[推断]`、§4.11.2 补错误放大路径；Q4 收口；断言 63→68；新增 §7.2 | `docs/reviews/round2/06-studio-backend-round2-review-by-studio-frontend.md`（R4 通过） |
