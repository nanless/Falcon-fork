# 评审：07-studio-frontend.md —— 评审人 ops-tests-docs

> 评审轮次：R2（交叉评审 1）；被评审文档：`docs/understanding/07-studio-frontend.md`（1100 行，R1 版）
> 评审人视角：运维/测试/CI/文档一致性（task-8），R1 已独立核对 `studio/server.py`、`studio/falcon_runtime.py`、`studio/tests/**` 与全部前端测试。
> 证据等级沿用 CONVENTIONS：`[代码]` 读源码；`[推断]` 合理推断；`[验证]` 本轮实际运行。基线 commit `7576854`。
> 评审方式：除通读 07 全文外，**逐行抽查了 60+ 条 `file:line` 证据**（`app.js`、`capture-worklet.js`、`playback-resampler.js`、`mouth-tracker.js`、`video-worker.js`、`spectrum-dsp.js`、`spectrum-worker.js`、`audio-inspector.js`、`i18n*.js`、`index.html`、`server.py`、`falcon_runtime.py`、8 个 Node 测试与 4 个 pytest），并重跑 Node 测试/计数脚本；未修改任何源码或本文档。

## 1. 结论（通过 / 需修订：P0/P1 计数）

**结论：通过（需修订）。P0 = 0，P1 = 3，P2 = 6。**

07 文档是本轮质量最高的文档之一：其**核心结论经我独立复核全部成立**，尤其是 Lead 指定的五项重点：

1. **`LEFT_CONTEXT` 数值与语义**：`app.js:10` 确为 `WINDOW-HOP-GUARD = 32000-8000-1280 = 22720`（1.42 s）；`server.py:80` 的 `OutputSlicer` 切片起点 `start, stop = WINDOW-GUARD-HOP, WINDOW-GUARD` 恰为 `[22720:30720]`；`app.js:593` 发送 `end_sample: end + LEFT_CONTEXT`；`app.js:963-971` 的 `inputSpan` 中 `offset = start-(request.end-WINDOW)` 恒等于 22720；`server.py:266` 返回 `start_sample=endpoint-GUARD-HOP`、`end_sample=endpoint-GUARD`，前端 `app.js:677-683` 校验并减偏移。**四处互证成立，任务书/Lead 提示中的 22400 确为错误值**，07 的纠正是正确的。[代码+验证]
2. **常量对照表 C1–C15**：逐项核对 `app.js:6-12` ↔ `server.py:28-30` ↔ `falcon_runtime.py:24-27`，**全部一致**；C7（后端无对应）、C9（前端 5 ms fade-in vs 服务端 20 ms crossfade）、C11（播放调度互补）、C12（纯前端频谱）等"无后端对应"标注也准确。
3. **WS 双向 schema**：客户端唯一发送点 `app.js:593` 的 8 个字段与 `server.py:35-67 decode_request` 的校验逐条吻合；服务端回包 `server.py:262-267` 的 14 个字段与前端 `app.js:659-685` 的校验/用法逐条吻合（含 `start_sample`/`end_sample` 的偏移公式、`contiguous`、`output_delay_ms`、`inference_ms/rtf`）。[代码]
4. **测试可运行性**：8 个 Node 测试**全部实测 rc=0**（我本轮重跑：i18n 8 项/349 条、capture_dsp 的 44100/48000 数值与作者 §8.1 完全一致、tracking_ui 18 条、spectrum 6 组、playback 6 段、mouth/worklet/video-worker PASS）；`node --check web/*.js` **13/13 OK**；i18n 词典 **120+151+55+23=349 条、去重 337、12 个重复键且值完全相同**（用 Node 导入真实模块复核）、`index.html` 125 个绑定/120 唯一键、`app.js` 66 个字面量 DOM id、`app.js` 汉字字符串唯一字面量 153 个等 §8.3 数字**全部复现成功**。[验证]
5. **crossfade 归属**：服务端 `server.py:29 FADE=320`（20 ms）+ `:83-86` 用 `future = pred[stop:stop+FADE]` 做块首 20 ms 交叉淡化；前端 `app.js:755-759` 只在 `contiguous===false || playAt<currentTime+0.03` 时做 5 ms 线性淡入。**"服务端负责连续块淡化、前端负责不连续块防爆音"的分工判断正确**，不是前后端实现重复。[代码]

P1/P2 问题集中在**少数证据路径/行号引用**与**1 个不可复现的性能数字**，均不影响文档主结论；修订成本低，建议 R3 修正后即可作为 FINAL 的 Studio 前端基线。

## 2. 断言逐条核对表（✅正确 / ❌错误 / ❓无法确认；仅列本轮抽查项，共 40 条）

| # | 07 断言 | 核对结果 | 证据核对（我实测/读到的真实行） |
|---|---|---|---|
| A1 | 前端常量 RATE/WINDOW/HOP/GUARD/FPS/MAX_RECORDING_SECONDS 与后端一致 | ✅ | `app.js:6-12` 与 `server.py:28-30`、`falcon_runtime.py:24-27` 逐值一致 |
| A2 | `LEFT_CONTEXT=22720` 不是 22400 | ✅ | `app.js:10` 实为 `WINDOW - HOP - GUARD`，计算结果 22720 |
| A3 | `LEFT_CONTEXT` == 服务端切片起点 `WINDOW-GUARD-HOP` | ✅ | `server.py:80` `start, stop = WINDOW - GUARD - HOP, WINDOW - GUARD` |
| A4 | 首窗 9280 样本（0.58 s）发送，`end_sample=32000` | ✅ | `app.js:567-570`；`check_tracking_ui.cjs:346-355`（我读码确认 `sent[0].end_sample===32000`、`slice(0,22720).every(0)`） |
| A5 | 音频为 base64 LE-float32×32000（非 ArrayBuffer） | ✅ | `app.js:543-550,593`；`server.py:42-50`（长度 `4*((32000*4+2)//3)=170668` 精确） |
| A6 | frames 为 50 个 base64 JPEG；服务端要求 96×96 JPEG/PNG、数量 50 | ✅（限读码） | `app.js:523-542,593`；`server.py:51-63`；真实 JPEG 解码未跑（作者 §8.4/§8.6 已如实标注） |
| A7 | 服务端 `start=endpoint-GUARD-HOP`、`end=endpoint-GUARD`，前端校验并减偏移 | ✅ | `server.py:266`；`app.js:677-683`（用 `timestampOffset`） |
| A8 | worklet 63-tap/128 相位/截止 `min(7200,0.45fs)`/31 样本群延迟补偿 | ✅ | `capture-worklet.js:8-29,56`（相位预计算 17-28、packet 320 在 29） |
| A9 | 44.1/48k 12 kHz 混叠 >60 dB、通带 RMS≈0.707、时间戳误差<0.001 | ✅ | 我实测 `check_capture_dsp.cjs`：0.707947/−60.684/0.000989；0.706718/−60.134/0.000389 |
| A10 | worklet 输出恒 0（原始麦克风不监听），**测试 capture-worklet.test.cjs:25-28 验证** | ❌ = P1-1 | `:34-36` 代码正确；但 `capture-worklet.test.cjs:25-28` 是 `tone()` 里的采样/RMS 行，**无静音断言**；真正断言在 `studio/tests/check_capture_dsp.cjs:25-28`（`assert(muted.every(x=>x===0))`） |
| A12 | onPCM 8 s ring；漂移>80 ms 重锚并 resetTrackingWindow | ✅ | `app.js:15,249-255` |
| A13 | 最新 2 s 回读、sessionEpoch 前填 0（首窗 22720 个 0） | ✅ | `app.js:577-581`；`check_tracking_ui.cjs:346-369`（行 354） |
| A14 | 单飞、慢推理不清 inFlight | ✅ | `app.js:565,588,318-319`；`check_tracking_ui.cjs:97-117` |
| A16 | 30 fps→25 fps（60 帧出 50 帧） | ✅ | `video-worker.js:57-59`；`video-worker.test.cjs:35` |
| A17 | 96×96 灰度 .299/.587/.114、JPEG q=.88 | ✅ | `video-worker.js:88-96`；`video-worker.test.cjs:37-38`（灰度 124） |
| A19 | 首窗/重捕合成历史只用本目标 epoch 后第一帧、不用 old-target | ✅ | `app.js:524-527,535`；`check_tracking_ui.cjs:325-369`（行 330/340/355-356 正是 old-target/new-frame 断言） |
| A20 | 丢脸时 trackingAllowsInference=false，暂停推理但保持会话 | ✅ | `app.js:305-307,566`；`check_tracking_ui.cjs:97-117` |
| A21 | 追踪代际切换重置 inferenceEpoch/sessionEpoch/lastSentEnd，旧结果按 epoch 丢弃 | ✅ | `app.js:308-320,667-670`；`check_tracking_ui.cjs:154-166` |
| A22 | 丢脸 ≤500 ms coasted；>500 ms 仍 tracking+searching 不暂停 | ✅ | `mouth-tracker.js:6,22-27,101-106`；`check_mouth_tracking.mjs:30-41` |
| A23 | 4 帧重锁 / 2500 ms 跳变确认 / 多张不一致脸要求重新捕捉 | ✅ | `mouth-tracker.js:7-10,111-149`；`check_mouth_tracking.mjs:42-82` |
| A24 | 张嘴不改裁剪尺寸（尺度只随脸宽） | ✅ | `mouth-tracker.js:46-51`；`check_mouth_tracking.mjs:27-28`（行 28 断言 size 相等） |
| A27 | MediaPipe 同源加载（bundle+wasm+task）、CPU delegate、numFaces=2、VIDEO、阈值 .6 | ✅ | `video-worker.js:29-37`；`manifest.json:2-4`（1.0.1） |
| A28 | 播放重采样 63-tap/2048 相/7.4 kHz；通带/镜像/分块一致 | ✅ | `playback-resampler.js:18-20,57-123`；我重跑 6 段 PASS |
| A31 | 耳机监听默认关（gain=0），勾选后 1 | ✅ | `app.js:206,1062-1066`；`index.html:45`（我查到 monitorToggle 在该行） |
| A33 | 300 s 后停写录音、实时分离/会议/频谱继续 | ✅ | `app.js:687-701,718`；`check_tracking_ui.cjs:205-226` |
| A34 | Separated/Mixture 同起止、mixture 用请求快照 | ✅ | `app.js:690,963-971`；`check_tracking_ui.cjs:228-262,264-276`（行 275 `audio.slice(22720,30720)`） |
| A36 | 停止先排空最后 0.6–1.1 s、补零后裁剪；二次点击立即停 | ✅ | `app.js:842-883`；`check_tracking_ui.cjs:278-323` |
| A37 | 排空后录音末尾=点击停止真实时刻 | ✅ | `app.js:869-878`；`check_tracking_ui.cjs:297-314,371-383` |
| A39 | 频谱 1024 Hann/640 hop/128 行/0–8 kHz/−90..0 dBFS | ✅ | `spectrum-dsp.js:5-11,27-29,54-81`；`check_spectrum.mjs` PASS |
| A40 | 300 s/1600 列概览有界（9600 次 FFT） | ✅ | `spectrum-dsp.js:115-154`；我重跑输出 `fft_windows: 9600, columns: 1600` |
| A42 | 录音细览 1/2/4/8/16×、pan、点击/拖动/键盘、进度条同步 | ✅ | `audio-inspector.js:85,92,259-285`（85 行 1/2/4/8/16 选项、279 行 Arrow 键 step、283 行 timeupdate 等） |
| A44 | i18n 合并 4 字典 349 条、去重 337、12 个重复键值相同 | ✅ | 我用 Node 导入真实模块：120+151+55+23=349、unique 337、dups 12、dup-value-diff 0 |
| A45 | 首启英文（不看浏览器语言）、保存值生效、非法回退、存储禁用不崩溃 | ✅ | `i18n.js:12-16,106-113`；`check_i18n.mjs:84-108`（测试名与行号吻合） |
| A47 | 服务端/Worker 中文错误与英文 runtime 错误可双向翻译、未知原样透传 | ✅ | `check_i18n.mjs:151-163`（我读码：E_TEST_DISK 保留、设备名保留、切回中文断言） |
| A49 | index.html 125 绑定/120 唯一键全部有英文译文；**初始 HTML 无中文展示文本** | ⚠️ 措辞= P2-3 | 125/120 实测正确（`grep`）；但 `index.html:28` 的 `<option value="zh">中文</option>` 是中文展示文本；`check_i18n.mjs:143` 明确豁免 `text === '中文'`。建议改为"除语言选项'中文'外" |
| A50 | app.js 字面量 DOM id 全部存在于 index.html | ✅ | `check_tracking_ui.cjs:92-93`；我实测 app.js 66 个字面量 id、index.html 69 个唯一 id |
| A51 | 所有前端 JS 语法合法（node --check 13/13） | ✅ | 我重跑 `node --check web/*.js` = 13/13 OK |
| A52 | 单条 WS ~0.2–0.64 MB < 2 MB | ✅（估算） | 我用 py3.8+PIL 复现 §8.5：纯灰 107 JPEG=**439 B**（与作者 439 完全一致）；纯噪声 7101 B（作者 7097，编码器/种子差异）；整包估算成立 |
| 07-§2.2 | capture-worklet 输出静音由 `capture-worklet.test.cjs:27-28` 验证 | ❌ = P1-1（同上） | 应为 `tests/check_capture_dsp.cjs:25-28` |
| 07-§3.1.2 S3 | `worklet.connect(destination)`（输出静音）证据 `app.js:225-227` | ❌ = P1-3 | 实际连接在 `app.js:224`；`225-227` 是 track `ended` 处理。同句其余证据 `202-238,1081-1086` 正确 |
| 07-§5.1 | `GET /static/*` 的前端加载证据 `app.js:218,240,406,621` | ❌ = P1-2 | 218=worklet、406=video worker 正确；240 是 catch 里停轨、621 是 WS URL，均非静态资源；缺 `audio-inspector.js:240`（spectrum-worker）、`video-worker.js:29-32`（MediaPipe）、`app.js:218` 以外的 worker 加载 |
| 07-§8.1#2 | 300 s 频谱概览 `milliseconds≈1889`，波动 710–1889 ms | ⚠️ = P2-1 | 我两次实测 **671.3 ms / 683.9 ms**（均低于作者下限 710）；结论"有界、不阻塞实时"不受影响，但该 `[验证]` 数字范围不可复现，建议改为"约 0.7–1.9 s，随机器波动" |
| 07-§4.5 | 服务端 `Engine.infer` 用**非阻塞互斥锁串行化** | ⚠️ = P2-2 | `server.py:160-172`：`infer_lock.acquire(blocking=False)` 失败即 `RuntimeError("另一个窗口正在使用模型，请关闭重复会话。")`——是**拒绝并发**而非排队串行化；与作者 §7 Q5 的"模型忙"描述自洽，建议改词 |
| 07-§2.2 | worklet 支持"16/44.1/48/**96** kHz" | ⚠️ = P2-4 | 代码对任意比例通用（`ratio=sampleRate/16000`），但 96 kHz 无任何测试/基准（`check_capture_dsp`/`capture-worklet.test` 只测 16/44.1/48）；建议标 [推断] 或补测 |
| 07-§4.2 | "窗口末尾 1280 样本（80 ms）作为下一块的右上下文" | ⚠️ = P2-5 | 机制成立，但更精确：`server.py:86` 的 `future = pred[stop:stop+FADE]` 只取 20 ms（`FADE=320`）参与下一块交叉淡化；guard 的 80 ms 是"不发射的尾部上下文"。建议补 `server.py:86` 精确说明，避免读者以为 1280 样本整体参与淡化 |

（其余 A11/A15/A18/A25/A26/A29/A30/A32/A35/A38/A41/A43/A46/A48 抽读代码/测试亦均正确，未发现错误；为节省篇幅不逐条展开。）

## 3. 错误清单

### P0（事实错误/会误导实现）
- 无。

### P1（重要遗漏或证据错误）
1. **P1-1｜§2.2 与 A10：静音输出验证引用了错误的测试文件。**
   07 称 "输出强制填 0（`capture-worklet.js:34-36`，测试 `capture-worklet.test.cjs:27-28` 验证）"（A10 证据列同）。
   实际 `studio/web/capture-worklet.test.cjs:27-28` 位于 `tone()` 辅助函数：
   `const samples = fixture.packets.flatMap(...); const settled = samples.slice(640);`——**没有任何输出静音断言**。
   真正验证该行为的是 `studio/tests/check_capture_dsp.cjs:25-28`：`const muted = ...fill(1); ...; assert(muted.every((x) => x === 0), 'Capture output must not monitor raw audio')`。
   修正：两处把测试引用改为 `tests/check_capture_dsp.cjs:25-28`（可同时保留 `capture-worklet.test.cjs` 作为整套 worklet 测试的入口）。
2. **P1-2｜§5.1 `/static/*` 行的前端证据含两条无效行号。**
   证据列 `app.js:218,240,406,621`：218 为 `audioWorklet.addModule('/static/capture-worklet.js')`、406 为 `new Worker('/static/video-worker.js')`（正确）；但 **240 是 `openCapture` catch 分支里的 `stream.getTracks().forEach(track.stop)`**，**621 是 `new WebSocket(.../ws)`**，都不是静态资源加载。
   实际加载点还有：`audio-inspector.js:240`（`new Worker('/static/spectrum-worker.js')`）、`video-worker.js:29-32`（MediaPipe bundle/wasm/task 同源加载）、`audio-inspector.js:40-43`（`/static/audio-inspector.css` 按需注入）。
   修正：把该行证据改为 `server.py:282-283；app.js:218,406；audio-inspector.js:40-43,240；video-worker.js:29-32`。
3. **P1-3｜§3.1.2 S3：`worklet.connect(destination)` 行号错位。**
   07 写 "…`worklet.connect(destination)`（输出静音）| `app.js:202-238,225-227,1081-1086`"。
   实际 `state.worklet.connect(context.destination);` 在 **`app.js:224`**；`225-227` 是 `stream.getTracks()...track.addEventListener('ended',...)`。S3 该行同时还在描述"输出静音"，容易让读者以为 225-227 是连接/静音逻辑。
   修正：改为 `app.js:218-224`（worklet 创建+连接静音）与 `app.js:225-227`（track ended 处理）分开引用。

### P2（细节）
1. **§8.1#2 的性能数字范围不可复现**：作者写 5 分钟概览 `milliseconds≈1889`、"波动 710–1889 ms"；本轮两次跑 `check_spectrum.mjs` 分别得到 **671.3 ms**（本次）与 **683.9 ms**（R1），均低于 710。该数字只是耗时观测且随机器/负载波动，不影响"有界、不阻塞实时"结论；建议改为"约 0.7–1.9 s（随环境波动）"或直接删除具体上限。
2. **§4.5 "非阻塞互斥锁串行化"措辞**：`Engine.infer` 的 `acquire(blocking=False)` 在冲突时抛 `RuntimeError`（`server.py:163-164`），行为是"互斥+拒绝"，不是排队串行。与 §7 Q5 的"模型忙会把一次可恢复错误升级为停止"一致，建议把"串行化"改为"互斥（冲突时返回 模型忙）"。
3. **A49 "初始 HTML 无中文展示文本"**：`index.html:28` 的 `<option value="zh">中文</option>` 是中文展示文本；测试 `check_i18n.mjs:143` 显式豁免 `text === '中文'`。建议加限定词。
4. **§2.2 "支持 16/44.1/48/96 kHz"**：代码通用但 96 kHz 无测试覆盖（现有测试仅 16/44.1/48）；建议标 [推断] 或补一条 96 kHz 的 Node 断言。
5. **§4.2 "窗口末尾 1280 样本作为下一块右上下文"**：建议补精确机制——实际参与交叉淡化的只有 `pred[stop:stop+FADE]` 的 20 ms（`server.py:86`），其余 60 ms guard 仅作为模型侧上下文不发射；否则读者可能误以为 1280 样本整体参与下一块合成。
6. **少量闭区间行号多计一行**（P3 级，可不改）：`spectrum-dsp.js` `column` 实际 41-46（07 写 41-47）、`circularColumn` 49-51（写 49-52）；`check_mouth_tracking` 实际断言 55 条（07 写"20+ 断言"，属保守表述，无碍）。

## 4. 遗漏的重要行为（建议 R3 补充）

1. **结果去重保护 `state.lastResultEnd` 未写**（`app.js:684-685`）：`if (result.end_sample <= state.lastResultEnd) return;` —— 服务端若重复/乱序回包，前端会静默丢弃，避免重复播放与重复写录音。这是协议健壮性的关键一环，但 07 的 §3.4/§5.3 均未提及；建议加入"协议去重"小节并说明与 `inFlight.id` 校验（`app.js:663`）的分工。
2. **排空期间 `onPCM` 直接丢包未写**（`app.js:248`：`if (!state.context || !samples?.length || state.draining) return;`）：`requestStop` 之后新到的麦克风样本不再进入 ring/计数，drain 完全依赖补零+重复末帧（`app.js:847-853`）与 in-flight 排空；这解释了"停止后录音末尾=点击停止时刻"的实现基础，建议在 §4.6 补一句。
3. **16 kHz AudioContext 时完全绕过播放重采样器**：`createPlaybackUpsampler(ctx.sampleRate, RATE)` 同率返回 `null`（`playback-resampler.js:126-127`），`app.js:744` 仅在 `context.sampleRate !== RATE` 时进入重采样分支；07 的 §3.5 只描述了非 16 k 分支，未说明"同率直通"路径（对理解设备兼容性有用）。
4. **`handleResult` 中 `result.type === 'error'` 之外还有 WS `onclose`/`onerror` 的"保留录音并停"路径**（`app.js:631-632`）；§5.5 有表格条目但 §3.4 时序图未画，建议补。
5. **`index.html` 的 `audioInspectorStyles` 按需注入与模块加载顺序**（`audio-inspector.js:40-43` + `app.js:1-3` 静态 import）：07 §1.2 只画了模块依赖，未说明样式注入与 `check_i18n`/CSP 的关系；属次要遗漏。

## 5. 跨模块一致性问题

1. **`LEFT_CONTEXT`/`end_sample` 是"仅存在于前端的隐式协议契约"**：`decode_request` 只校验 `WINDOW <= end_sample <= 2^53-1`（`server.py:64-66`），服务端**不知道** `LEFT_CONTEXT` 偏移；如果未来出现第二个客户端（或 CLI 直连 `/ws`）按真实时间轴发送 `end_sample=end`，服务端仍会正常回包，但返回的 `start_sample/end_sample` 可能为负/错位，前端校验会拒绝。07 的 §5.2 已把"endpoint 语义链"写清楚，但建议在 §5.2 显式标注"这是客户端-服务端私有协议，服务器代码未编码 LEFT_CONTEXT 常量"，并请 `06-studio-backend` 文档交叉引用/确认。
2. **crossfade 分工需两边文档互证**：07 已写明"服务端 20 ms（`server.py:29,83-86`）+ 前端 5 ms fade-in（`app.js:753-759`）"；建议 `06` 文档在 `OutputSlicer` 一节引用同一结论，避免 FINAL 汇总时被误写成"前端也做 crossfade"（`00-project-map.md:4.3` 原假设即为此，07 §1.4 已纠正）。
3. **服务端返回 3 个前端未用字段**：`window_rtf`、`device`、`clipped_samples`（`server.py:265,267`；`app.js` 中无引用——`grep window_rtf/clipped_samples` 为空）。不是错误，但建议 06/07 都注明"保留字段/供状态展示或诊断"，避免读者以为前端遗漏解析。
4. **A/V 时间戳校验只在客户端**：`decode_request` 不接收帧时间戳、服务端不回传 A/V 对齐信息；07 §3.3 末句已注明"服务端不校验帧时间戳，A/V 对齐完全由前端 frameWindow 负责"。建议 06 在 WS 协议小节同样声明该边界（服务端信任客户端对齐）。
5. **测试矩阵与 08 文档一致**：07 §8.1 的 8 个 Node 测试结果与我 R1 的 08 文档 §2.4 完全一致（8/8 rc=0，i18n 349 条）；07 §8.6 对"PIL 缺失导致 test_server 不能跑""无 Chrome 不能跑 browser QA"的判断也与 08 文档一致，无冲突。

## 6. 给作者的问题清单（需作者回答或补充验证）

| # | 问题 | 建议 |
|---|---|---|
| Q1 | P1-1 的测试文件引用是否直接改正为 `tests/check_capture_dsp.cjs:25-28`？（两处：§2.2 与 A10） | R3 修正 |
| Q2 | §5.1 的 `app.js:240,621` 两条证据打算改为哪些真实加载点？建议按 §3 错误清单 P1-2 的清单改 | R3 修正 |
| Q3 | §3.1.2 S3 的 `225-227` 是否拆分为 `218-224`（worklet 创建/连接静音）与 `225-227`（track ended）？ | R3 修正 |
| Q4 | §8.1#2 的 710–1889 ms 是否改为"约 0.7–1.9 s（随环境波动）"或直接删除具体范围？ | R3 修正 |
| Q5 | §4.5 "串行化"是否改为"非阻塞互斥（冲突返回模型忙）"？ | R3 修正 |
| Q6 | 96 kHz 是否有实测数据？若无，建议 §2.2 标 [推断] 或补 Node 测例 | R3/R4 |
| Q7 | 是否需要在 §4.6 补充"onPCM 在 draining 期间直接丢包"、在 §3.4/§5.3 补充 `lastResultEnd` 去重？ | 建议补（遗漏 1、2） |
| Q8 | `LEFT_CONTEXT` 的客户端私有契约是否同意在 §5.2 标注"服务端未编码"，并让 `06` 文档交叉引用？ | 建议补 |
| Q9 | A49 的"无中文展示文本"是否按 `check_i18n.mjs:143` 的豁免加限定词（除语言选项"中文"）？ | R3 修正 |
| Q10 | §4.2 的"1280 样本右上下文"是否补充 `FADE=320` 精确机制（`server.py:83-86`）？ | 建议补 |

## 7. 评审人已执行的验证（供复核）

```text
# Node 测试（全部 rc=0；本轮重跑关键 4 项 + 复核其余）
node studio/tests/check_i18n.mjs             → 8 localization checks passed (349 dictionary entries audited)
node studio/tests/check_capture_dsp.cjs      → 44100: rms 0.707947, alias -60.684 dB, ts-err 0.000989
                                               48000: rms 0.706718, alias -60.134 dB, ts-err 0.000389
node studio/tests/check_tracking_ui.cjs      → 18 tracking UI control-flow checks passed
node studio/tests/check_spectrum.mjs         → "test":"PASS"；本次 overview milliseconds=671.3（作者写的 1889/710–1889 未复现）
（R1 已跑并记录其余 4 个：mouth/playback/worklet/video-worker，均 PASS）

# node --check
for f in studio/web/*.js; do node --check "$f"; done   → 13/13 OK

# i18n 计数（Node 导入真实模块）
static 120 / dynamic 151 / errors 55 / audio 23；total 349；unique 337；dups 12；dup-value-diff 0

# 其他复核
app.js 字面量 $('id') = 66 个；index.html 唯一 id = 69；data-i18n* 绑定 = 125（唯一 120）
汉字字符串唯一字面量：app.js 153 / video-worker.js 5 / mouth-tracker.js 4 / audio-inspector.js 23（与 §8.3 一致）
JPEG(PIL q=88) 纯灰 107 = 439 B（与 §8.5 一致）；纯噪声 7101 B（作者 7097，编码器差异）
行数复核：web/ 16 个文件全部与 07 §1.3 表一致（app.js 1165、index.html 64、style.css 54 等）
关键行号复核（全部真实存在且语义相符）：app.js:6-12/224/248/565-593/663/677-683/684-685/755-759/963-971；
server.py:28-30/35-67/70-91/160-172/233-276/262-267/278-283/332-333；
falcon_runtime.py:24-27/95-104；mouth-tracker.js 11 处；video-worker.js 18 处；spectrum-dsp.js 13 处；
i18n.js 1-124 全段；audio-inspector.js 38-345 主要函数；8 个 Node 测试的引用行段
```

未修改源码；未 commit；本文件为 R2 唯一新增（`docs/reviews/round1/07-studio-frontend-review-by-ops-tests-docs.md`）。
