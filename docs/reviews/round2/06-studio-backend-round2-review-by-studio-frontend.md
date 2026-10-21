# 评审（R4 复核）：`docs/understanding/06-studio-backend.md` v2 —— 评审人 `studio-frontend`

> 复核对象：`docs/understanding/06-studio-backend.md` **v2**（1201 行，R3 修订版，2026-10-21 05:19 落盘）。
> R2 评审：`docs/reviews/round1/06-studio-backend-review-by-studio-frontend.md`（P0=0、P1=1、P2=9、P3=3）。
> 本轮职责（00-round-plan R4 表）：**第二评审人，协议专项 + 执行级验证**；指定动作 =“前端常量 ↔ 后端常量逐项对照并标注一致/不一致”。
> 方法：①逐条核对 R3 是否修复 R2 全部意见（读 v2 原文 + 对照源码）；②独立执行 3 项执行级验证（§6）；
> ③扫描 v2 新增章节（§4.11、§5.5 双坐标、§7.1）是否引入新错误；④前端常量矩阵与 06 §5.9/07 域交叉核对。
> 未修改任何源码（`git status` 仍只有 `docs/` 未跟踪；`git diff --stat` 为空）。

## 1. 结论

**通过（R4 复核）。P0 = 0，P1 = 0，P2 = 2，P3 = 3。**

- R2 提出的 **P1×1、P2×9、P3×3 全部修复**，逐条核对见 §2；对 R2 遗漏 1–7 新增的 §4.11（协议时序/并发/失败模式/部署）
  与源码逐项吻合，是本轮质量提升最大的一节。
- 执行级验证 3 项全部 PASS（§6）：前端常量矩阵 5/5 一致 + `LEFT_CONTEXT=22720` 与服务端切片起点相等；
  前端 `encodeFloat32` → 服务端 `decode_request` 音频 **bit-exact** 往返；双窗协议链
  （32000/40000 → 22720/30720/38720 → 采样 [0,8000)/[8000,16000)，拼接 max|Δ|=4.28e-8）。
- 残留问题均为文档级 P2/P3：§5.5 双坐标定义在 `sessionEpoch>0`（先预览再开始分离）时仍不严格成立（§3 P2-R1），
  以及 3 处小口径问题。不阻塞 FINAL，但建议 R5 前一并修掉，避免协议文档给第三方客户端错误换算。

## 2. R2 意见修复逐条核对

| R2 条目 | v2 处置 | 复核结果（评审人证据） |
|---|---|---|
| P1-1 §5.2 “无可用权重”→409 错误 | §5.2 409 行只留 RuntimeError；400 行新增“变体不可用”；守卫⑥写明 `weights_source()=None ⇔ 缺 huggingface_hub` | ✅ 源码 `server.py:140-141` 为 `ValueError`，`:227-228` 映射 400；表格与守卫⑥已自洽（`06:725-738`） |
| P2-1 `git log -S` 证据 | 改为 `git log --all -S 'class SISNRRepel' -- look2hear` 无结果，并注明不带路径会命中 a7af472 | ✅ 我复跑：不带路径 → a7af472；带 `-- look2hear` → 空（`06:514,1049`） |
| P2-2 Q8 文案“永远不可达” | 改为“可达但仅在缺 huggingface_hub 时，且把缺包误报为缺权重” | ✅ 与 `falcon_runtime.py:48-55` + `server.py:140-141` 一致（`06:1014-1018`） |
| P2-3 normalizations 28→26 | 三处统一 `13+/13− = 26` | ✅ 我统计 diff -u：added 13/deleted 13 = 26（`06:160,470,1121`） |
| P2-4 FRCNN 输入 96→88 | §1.1 改 `88,88` 并注明 96×96 是协议/采集尺寸 | ✅ 与 §3.5/`preprocess_mouth` 一致（`06:25`） |
| P2-5 行号漂移 | out_proj `380-383`、NeuroSCNetFusion `239-246`、audio_visual_test `129-132` | ✅ `grep -n out_proj` → 380/405；`class NeuroSCNetFusion` → 239；`getattr(...)` → 129（`06:596,600,1053`） |
| P2-6 文件/行数口径 | 标题改“15 个 .py 4035 行 + PROVENANCE.md 42 行 = 4077 行” | ✅ `wc -l` 逐文件求和 4035；+42=4077（`06:149`） |
| P2-7 CSP 摘录不完整 | §4.6 给全量 CSP（style/img/media/worker） | ✅ 与 `server.py:210` 逐指令一致（`06:432`） |
| P2-8 §5.5 坐标混用 | 新增“协议坐标 vs 行内采样坐标”表 + 两窗数值例 | ⚠️ **大部分修复，但一般情形仍有残留** → 见 §3 P2-R1 |
| P2-9 `.load()` 步骤顺序 | 按源码重写：strict→统计→set_num_threads→eval/freeze→pick_device→warmup | ✅ 与 `falcon_runtime.py:142-196` 顺序一致（`06:127`） |
| P3-1 36.5→36.6 千字符 | §5.3 改为 `(2,000,000−170,668)/50 ≈ 36.6 千字符（未扣 JSON 开销）` | ✅ 算术 36,586.6 字符（`06:756`） |
| P3-2 `inference_ms` 措辞 | §5.1/§5.4 改为“含 preprocess_mouth 的前向耗时” | ✅ `06:714`；与 `server.py:167-169`、`falcon_runtime.py:239,258` 一致 |
| P3-3 4 s / 1.5 s 轮询 | §4.7 区分全局 4 s 轮询与 startSession 1.5 s 等待循环 | ✅ `app.js:1164` vs `app.js:608-619`（`06:446`） |
| R2 遗漏 1–7 | 新增 §4.11.1–4.11.9 + 失败注入矩阵 + 部署结论 | ✅ 逐项与源码核对（下表） |
| A54（在线 demo） | 保留 [推断]，§4.11.9 给“仓库内 server.py 不能直接对外部署 + 补丁清单” | ✅ 我执行 `same_origin(公网)=False`、HTTP Host=公网→400、evil Origin→403（§6-E5） |

R2 遗漏 1–7 的 v2 落实核对：

| 遗漏 | v2 位置 | 复核结果 |
|---|---|---|
| 1 多连接 busy/lock 竞态 | §4.11.1 | ✅ 与 `server.py:163-164,169-172,258-259` 一致；`busy=False` 先于 `release()` 的窗口真实存在 |
| 2 每个 result 重建 status | §4.11.2 | ✅ `server.py:265` 调用 `engine.status()`；`variants()` 对每 variant 调 2 次 `weights_source` |
| 3 无请求级超时 | §4.11.3 | ✅ `server.py:260` 只有 `to_thread`，无 `wait_for`（`grep` 确认） |
| 4 available 三态/组合语义 | §4.11.4 | ✅ 与 `server.py:118-124`、`falcon_runtime.py:48-55` 一致 |
| 5 TrustedHost 对 WS | §4.11.5 | ✅ 我用 TestClient 复现 `WebSocketDenialResponse` |
| 6 infer_file float 不对称 | §4.11.6 | ✅ `infer_file.py:34-38` 直读不 clip；`server.py:49` 有 `\|amp\|≤4` |
| 7 双 Engine/import 副作用 | §4.11.7 | ✅ `server.py:287` 与 `:332` 两条路径 |

## 3. R4 新发现问题

### P2（建议 R5 前修）

**P2-R1｜§5.5 双坐标表在 `sessionEpoch>0` 时仍不严格成立（R2-P2-8 的残留）。**

- v2 表（`06:789-794`）写：
  - “**协议坐标** 原点 = 前端 `sessionEpoch - LEFT_CONTEXT`”；
  - “**行内采样坐标** 原点 = 用户点开始分离时的 `sessionEpoch`”；
  - “换算：**协议坐标 = 行内采样坐标 + LEFT_CONTEXT(22720)**”。
- 但前端实际发送的是 **绝对采样下标**：`app.js:637` `state.sessionEpoch = state.samples`（`samples` 自打开设备起累计），
  `app.js:570` 首窗 `end = sessionEpoch + HOP + GUARD`，`app.js:593` `end_sample = end + LEFT_CONTEXT`。
  因此一般情形是：
  - 绝对采样坐标 `s`（`state.samples`），**协议坐标 `P = s + 22720`**；协议 0 只对应 `s = -22720`，
    **与 `sessionEpoch` 无关**（`sessionEpoch=0` 时 `sessionEpoch-22720` 才恰好等于 −22720）。
  - 若用相对 `sessionEpoch` 的坐标 `r = s − sessionEpoch`，则 `P = r + sessionEpoch + 22720`，不是 `r + 22720`。
  - 首窗请求是 `end_sample = sessionEpoch + 32000`（不是恒 32000）；协议窗口 `[sessionEpoch, sessionEpoch+32000)`；
    返回 22720/30720（epoch=0）或 `sessionEpoch+22720 / sessionEpoch+30720`（一般）；折回绝对采样仍是
    `[sessionEpoch, sessionEpoch+8000)`。
- 影响：`sessionEpoch=0` 只在“打开设备后立刻点开始”时成立；正常用户会先预览/等模型加载，`sessionEpoch` 常为几万到几十万，
  协议时间戳整体平移。**浏览器客户端自身没问题**（它全用绝对值），但文档给第三方/CLI 客户端的换算公式会差一个
  `sessionEpoch`，且 §5.5 数值例（`06:796-804`）未声明“示例假设 sessionEpoch=0”。
- 修正建议：
  1) 表格改为“绝对采样坐标（`state.samples`，原点=打开设备后的首个采集样本）”，换算 `协议 = 绝对 + 22720`；
  2) 若保留“行内采样坐标”概念，定义 `r = samples − sessionEpoch` 并写 `协议 = r + sessionEpoch + 22720`；
  3) 数值例加前提“假设 `sessionEpoch=0`”，并补一个 `sessionEpoch=160000`（预览 10 s 后开始）的对照行：
     首窗请求 192000 → 返回 182720/190720 → 绝对采样 [160000,168000)；
  4) `06:825` 的“使首窗 `end_sample` 恰为 32000（窗口左边界 = `sessionEpoch-22720`…）”改为
     “`sessionEpoch=0` 时首窗 32000；一般首窗 = `sessionEpoch+32000`，窗口左边界（绝对采样）= `sessionEpoch-22720`”。
- 关联：我自己的 `07` 文档 R1/R3 版也在 `§4.2`/`§5.3`/A4 隐含 `sessionEpoch=0`，已在 **R4 自校**中补限定并新增
  `07 §7-Q13`（见 `07` 变更日志 R4 行），因此本 P2 只需 06 侧同步。

**P2-R2｜§5.5 图/例的“窗口覆盖 `[endpoint-32000, endpoint)`”与 `sessionEpoch` 的绝对性未区分**（同一根因的另一处，可并入 P2-R1 一起改）。
- 图（`06:806-817`）本身在协议坐标下正确；但与 §5.5 开头“绝对时间轴原点”并列时容易让读者误以为首窗窗口恒从协议 0 开始。
- 建议在图上方加一句“首窗协议窗口 = `[sessionEpoch, sessionEpoch+32000)`；下列数值以 `sessionEpoch=0` 为例”。

### P3（可改可不改）

- **P3-R1｜§7-Q4 未收口**：Q4（在线 demo 与 same_origin 冲突）仍列在 “P1（重要遗漏/误导性行为）” 清单（`06:1012-1015`），
  但 §4.11.9 已给出明确结论（`06:674-677`）。建议把 Q4 标为“已由 §4.11.9 回答，保留为部署方确认项”，避免 FINAL 重复统计。
- **P3-R2｜§7.1 的 out_proj 溯源表述**：`06:1043-1044` 称“`out_proj` 一条已在 R1 自审时先行改为 380-383，评审基于 1025 行旧版可见”。
  我 R2 评审引用的是 1025 行 R1 原文（当时为 `384-388`），与“已先行改为 380-383”表述略有出入；不影响结论，建议改为
  “评审指出后核实为 380-383（评审所据 1025 行版为 384-388）”。
- **P3-R3｜§4.11.1 的 `ws_max_queue` 断言**：`06:607-608` 把“`uvicorn(ws_max_queue=1)` 表示框架层每连接只保留 1 条待处理消息”
  写成事实；该语义取决于 uvicorn/websockets 版本实现细节，建议标 `[推断]` 或注明依据的 uvicorn 版本（本环境未装 uvicorn 运行时验证）。

## 4. 遗漏的重要行为 / 对“完全理解”的建议（≥3 条）

1. **补一个 `sessionEpoch>0` 的端到端数值例**（修 P2-R1 时顺带）：这是 Studio 最常见路径（先预览、等模型 ready 再开始），
   目前两份文档（06/07）都默认从 0 开始，FINAL 若照抄会留下“首窗 end_sample 恒 32000”的错误印象。
2. **补“同连接内请求-响应是串行顺序”的协议假设**：`server.py:242-272` 的 `while True` 在 `await ws.send_json(...)` 之后才
   读下一条消息，因此单连接内不会交错回包；但 v2 §4.11 只讲了多连接与 `ws_max_queue`，没写这条**单连接顺序保证**
   （对前端 `inFlight.id` 匹配的理解很重要）。建议 §5.3 加一句。
3. **补“协议时间戳上限与前端计数器”的关系**：`server.py:65,255` 用 `2^53−1` 限制；前端 `state.sequence`/`state.samples`
   是 JS `number` 递增，理论上永不越界（实际使用时长内远小于 2^53），但文档可显式说明“客户端不自行检查该上限，
   由服务端兜底”；这也解释为什么客户端只处理 `type/error` 两类回包。
4. **补 `audio_visual_test.py` 与 runtime 的 `torch.load` 安全口径对比**：`falcon_runtime.py:160` 用 `weights_only=True`，
   而 `audio_visual_test.py:120` 用 `weights_only=False`（04 域评测侧）；06 强调了前者但未对比后者，属于“同仓库两种加载
   安全姿态”的完整性缺口（可在 §4.3 或 Q3 提一句，归 04 域确认）。
5. **补“`Engine.status()` 热路径调用对异常路径的影响”**：`server.py:265` 在回包时同步调用 `status()`；若 `variants()`/
   `weights_source()` 抛异常（例如 HF 缓存目录权限问题），`send_json` 前就会异常进入 `except Exception` 的通用错误分支
   （`:270-272`），即“设备查询失败”会表现为“推理失败”。v2 §4.11.2 只写了性能，未写这条错误放大路径。

## 5. 跨模块一致性（含前后端常量矩阵）

### 5.1 前后端常量矩阵（R4 指定动作；执行级验证见 §6-E1）

| 常量 | 前端证据 | 后端证据 | 结果 |
|---|---|---|---|
| 采样率 | `app.js:6` RATE=16000 | `server.py:28` SR=16000；`falcon_runtime.py:24` SAMPLE_RATE=16_000 | ✅ 相等 |
| 窗口 | `app.js:7` WINDOW=2×RATE=32000 | `server.py:28` WINDOW=32000；`falcon_runtime.py:26` WINDOW_SAMPLES=32_000 | ✅ 相等 |
| hop | `app.js:8` HOP=RATE/2=8000 | `server.py:28` HOP=8000；`OutputSlicer` 切片宽（`:80`） | ✅ 相等 |
| 右上下文 | `app.js:9` GUARD=1280（80 ms） | `server.py:29` GUARD=1280；`decode_request`/`result` 时间戳（`:266`） | ✅ 相等 |
| 左上下文/切片起点 | `app.js:10` LEFT_CONTEXT=WINDOW−HOP−GUARD=**22720**（1.42 s） | `server.py:80` 切片起点 `WINDOW−GUARD−HOP=22720`（无同名常量） | ✅ 数值与语义相等（06 §5.9 / 07 §5.4 同结论；任务书 22400 无依据） |
| 帧率/帧数 | `app.js:11` FPS=25；`frameWindow` 50 帧（`:533`） | `server.py:28` FPS=25；`frames` 必须 50（`:51`）；`falcon_runtime.py:25,27` VIDEO_FPS=25/WINDOW_FRAMES=50 | ✅ 相等 |
| 交叉淡化 | 前端只做不连续块 5 ms 淡入（`app.js:757`） | `server.py:29` FADE=320（20 ms，连续块）；`OutputSlicer.take`（`:83-86`） | ✅ 分工互补，不冲突 |
| 录音上限 | `app.js:12` MAX_RECORDING_SECONDS=300；`audio-inspector.js:315` RATE×300 | 无后端常量；`spectrum-dsp.js:11` 4,800,000 样本 | ✅ 前端内部一致、无后端契约 |
| 协议端点 | `app.js:593` `end_sample=end+22720`（end 为绝对采样下标） | `server.py:64-67` 只校验范围，不解释偏移 | ✅ 私有契约（07 §5.2 已标注；P2-R1 需补 `sessionEpoch>0`） |

### 5.2 与 07 文档的互证/差异

- 06 v2 §5.9 与 07 §5.4 常量表逐项一致；06 §5.5 与 07 §5.2/§5.3 的 end/start 公式一致；06 Q5 与 07 Q5 是同一契约问题
  （前端 `type:"error"` → `stopCapture`），两文档均已登记，等待 Lead 产品口径。
- 唯一差异是 `sessionEpoch=0` 假设：06 §5.5 数值例默认 0；07 的 R1/R3 版 §4.2/§5.3/A4 也默认 0。**两文档需同步补一般情形**
  （我已在 R4 自校 07；06 侧见 P2-R1）。
- 06 §4.11.5/§4.11.9 与 07 §1.2 的安全/静态资源描述一致（CSP 全量、same-origin、TrustedHost）；06 §4.11.6 与 07
  §3.7 的录音链路无冲突（服务端不写盘，WAV 只在浏览器生成）。

### 5.3 与 08（ops/tests）的口径

06 v2 §7.1-Q8 已说明“缺 Pillow 时注入 stub 才能跑 test_server”，与 08 文档的“缺 Pillow 收集失败”并不矛盾，但 FINAL 仍需
统一表述（R4 未发现新问题）。

## 6. 执行级验证结论（R4 要求，共 5 项）

> 环境：`/root/.venvs/falcon/bin/python` 3.10.18 + torch 2.8.0+cu128；`docs/verification/stubs` + `docs/verification/scripts/pil_stub`。
> 所有脚本在 `/tmp`（不改仓库源码）。

### E1｜前端常量 ↔ 后端常量矩阵（R4 指定动作）——PASS

脚本从 `studio/web/app.js` 提取常量字面量，并 `import server`、`from falcon_runtime import ...` 逐项断言：

```text
RATE           frontend=  16000 backend=  16000 equal=True
WINDOW         frontend=  32000 backend=  32000 equal=True
HOP            frontend=   8000 backend=   8000 equal=True
GUARD          frontend=   1280 backend=   1280 equal=True
FPS            frontend=     25 backend=     25 equal=True
LEFT_CONTEXT   frontend=22720 (=1.42s) slicer_start=22720 equal=True
FADE           backend=320 (20ms) (frontend 5ms fade-in only)
runtime consts SAMPLE_RATE=16000 VIDEO_FPS=25 WINDOW_SAMPLES=32000 WINDOW_FRAMES=50
```

结论：**5/5 常量相等；`LEFT_CONTEXT`=服务端切片起点=22720；runtime 4 常量一致** → 06 §5.9 的“✅”表经执行级验证成立。

### E2｜前端 `encodeFloat32` → 服务端 `decode_request` 音频 bit-exact 往返 —— PASS

用 Node 从 `app.js:543-550` **原样抽取** `encodeFloat32` 函数，构造与 `app.js:593` 完全同形的 WS JSON
（`type/id/audio/frames×50/window_seconds:2/hop_seconds:.5/end_sample:32000/reset:true`），交服务端 `decode_request`：

```text
payload chars = 172596   audio b64 = 170668
decoded audio shape/dtype: (32000,) float32   endpoint: 32000   frames: (50, 96, 96) uint8
audio bit-exact roundtrip: True   max|diff| = 0.0
```

结论：**前端编码器与服务端解码器在音频路径上 bit-exact**；`end_sample=32000`、50 帧、96×96 形状均被接受。
局限（如实）：venv 无真实 Pillow，`frames` 用仓库 `pil_stub` 的 MAGIC 编码（96×96/JPEG）通过校验；真实 JPEG 解码未执行
（真实 96×96 JPEG 已由系统 python3+PIL 单独生成并校验 439 B/96×96，但与 stub 解码不等价）。

### E3｜双窗协议链：请求 32000/40000 → 返回 22720/30720/38720 → 前端减法 —— PASS

```text
win1 request end_sample=32000 -> result start/end = 22720 / 30720 contiguous=False chunk=8000
win2 request end_sample=40000 -> result start/end = 30720 / 38720 contiguous=True  chunk=8000
frontend subtract offset -> win1 [0,8000)  win2 [8000,16000)
join continuity max|diff| = 4.281997678656779e-08 contiguous2 = True
```

结论：`OutputSlicer` 的切片/时间戳/`contiguous`/交叉淡化与 06 §5.5/§4.2 一致；两块首尾相接、无重复（max|Δ|=4.28e-8，
与 06 §8-B1 的 4.3e-8 相符）。该例是 `sessionEpoch=0` 的情形（P2-R1 的一般情形需再补例）。

### E4｜R3 修复的静态抽查（normalizations/行号/CSP/计数）——PASS

- `diff -u` 统计 `normalizations.py` = **13+/13− = 26 行**（v2 已写 26）；
- `grep -n out_proj` → 380（v2 已写 380-383）；`class NeuroSCNetFusion` → 239、`math.gcd` → 243（v2 已写 239-246）；
- v2 §4.6 CSP 与 `server.py:210` 逐指令相同（含 style/img/media/worker）；
- v2 §2.4 =“15 py 4035 行 + PROVENANCE 42 = 4077”与 `wc -l` 一致。

### E5｜部署矛盾（A63/§4.11.9）执行级复核 —— PASS

```text
same_origin(localhost pair) = True
same_origin(public origin)  = False
same_origin(no origin)      = True
HTTP Host=127.0.0.1 status  = 200
HTTP Host=public status     = 400
HTTP evil Origin status     = 403
```

结论：仓库内 `server.py` 确实不能直接对外部署（与 A63/§4.11.9 一致）；在线 demo 必为部署补丁。

### 附：回归测试

```text
PYTHONPATH=<pil_stub>:<stubs>:$PWD /root/.venvs/falcon/bin/python -m pytest \
  studio/tests/test_server.py studio/tests/test_weights.py studio/tests/test_device.py studio/tests/test_infer_file.py \
  -q -p no:typeguard
→ 33 passed, 4 skipped, 1 warning in 3.63s
```

与 v2 §8.2/§8.4 的 `33 passed, 4 skipped` 一致；R3 为文档-only 修改（`git diff --stat` 空），协议行为无回归。

## 7. 给作者的问题清单（R5 前可修）

1. **（P2-R1）** 是否同意把 §5.5 改成“绝对采样坐标”表述，并补 `sessionEpoch>0` 的对照数值例？这是本轮唯一实质残留。
2. **（P2-R2）** 是否在图/数值例旁注明“以 `sessionEpoch=0` 为例”？
3. **（P3-R1）** 是否把 Q4 从 P1 清单移到“已由 §4.11.9 回答”？
4. **（P3-R2）** §7.1 的 out_proj 溯源表述是否按 §3-P3-R2 修正？
5. **（P3-R3）** §4.11.1 的 `ws_max_queue` 语义是否标 `[推断]` 或注明 uvicorn 版本？
6. **（§4 建议 2/3/4/5）** 是否补“单连接请求-响应顺序保证”“2^53 上限由服务端兜底”“`weights_only` 两种姿态对比”
   “status() 调用异常会表现为推理失败”这四条完整性说明？

## 8. 复核记录

- 复核范围：v2 全文（重点 §2.2、§4.6、§4.8–4.11、§5.1–5.6、§5.9、§7.1、§8.4、§9）；
- 独立复跑：pytest（33 passed/4 skipped）、常量矩阵、wire roundtrip、双窗协议链、部署矛盾、diff/行号静态抽查；
- 未修改任何源码或 06 文档；本文件为 R4 唯一新增。

## 9. 复核后更新：06 v3 已修复本轮全部问题（2026-10-21 05:36）

在本评审提交后，`studio-backend` 已将 06 更新为 **v3**（1252 行，2026-10-21 05:36），逐条落实本评审 §3/§4/§7：

| 本评审条目 | v3 处置（复核结果） |
|---|---|
| P2-R1 §5.5 坐标原点 | ✅ 已改为“**绝对采样坐标**（`state.samples`，自打开设备起）+ **协议坐标**”，明确 `P = s + 22720` 与 `sessionEpoch` 无关；并给出 `P = r + sessionEpoch + 22720`（§5.5, `06:796-805`） |
| P2-R2 数值例前提 | ✅ 已注明“首窗协议窗口 = `[sessionEpoch, sessionEpoch+32000)`；图中数值仅对应 `sessionEpoch=0` 示例”，并新增 `sessionEpoch=160000` 对照行（`06:807-819`） |
| P3-R1 Q4 未收口 | ✅ Q4 标注“已由 §4.11.9 回答；保留部署方确认项，不再计入未解问题”（`06:1020`） |
| P3-R2 out_proj 溯源 | ✅ §7.2 已重述口径（R4 响应表） |
| P3-R3 `ws_max_queue` 语义 | ✅ §4.11.1 已标 `[推断]`（R4 响应表） |
| §4 建议 1–5 | ✅ 全部采纳：`sessionEpoch>0` 例；§5.3 补“单连接请求-响应串行”与“2^53 由服务端兜底”；§4.3 补两种 `torch.load` 姿态；§4.11.2 补 `status()` 异常放大路径；新增断言 **A64–A68**（63→68 条），新增 §7.2 R4 响应表 |

**复核结论（更新）**：06 v3 对 R4 全部 P2/P3 与遗漏建议处置正确；我手工复核 `sessionEpoch=160000` 数值例：
首窗 `end=160000+9280=169280` → `end_sample=192000` → 返回 `182720/190720` → 减 22720 = `[160000,168000)`，
**与 `app.js:637,570,593` + `server.py:266` 推导一致**。因此 **06 最终状态：P0=0、P1=0、P2=0、P3=0，可进入 FINAL 汇总**。
本评审的 R4 结论由“通过（P2=2/P3=3 残留）”更新为“**通过（无残留问题）**”；v3 为文档-only 修改，未改源码。
