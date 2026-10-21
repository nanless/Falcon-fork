# 评审：`docs/understanding/06-studio-backend.md` —— 评审人 `studio-frontend`

> 评审对象：`docs/understanding/06-studio-backend.md`（1025 行，R1 版本）。
> 评审人范围：**WS/HTTP 协议前后端一致性 + 服务端行为**（R2 指派）；同时抽查协议之外的断言。
> 评审依据：我**自己重读**了 `studio/server.py`、`studio/falcon_runtime.py`、`studio/infer_file.py`、
> `studio/falcon_model/**`、`studio/configs/**`、`studio/tests/**`、`Experiments/checkpoint/Falcon/{conf.yml,best_k_models.json}`、
> `audio_visual_test.py`、`look2hear/**` 相关文件；并独立复算/复跑了 §2.3 列出的检查（不是只读被评审文档）。
> 证据格式 `path:line`；`[复核]` = 本次评审实际运行确认，`[读码]` = 评审人读源码确认。

## 1. 结论

**总体：通过（需小幅修订）。P0 = 0，P1 = 1，P2 = 9，P3 = 3。**

06 文档质量高：协议字段/常量/窗口切片/错误路径与源码逐项吻合；`LEFT_CONTEXT=22720`（并指出任务书 22400 无依据）
与 07 文档结论一致；PROVENANCE 逐条核验（含“Big 两份文件 look2hear 无源”）是本轮最有价值的发现之一。
协议相关的核心断言（A01–A27、A44–A49）经本次独立复算**全部成立**；§5.2 有一处 HTTP 状态码映射错误（P1），
其余是证据行号、计数口径、CSP 摘录与坐标表述等细节问题（P2/P3）。修订后可作为 FINAL 的协议依据。

## 2. 断言逐条核对表

> 抽查方式：优先“评审人独立复跑/复算”，其次“读码核对”。§2.3 给出复跑命令与输出摘要。
> 未复跑的项（如 GPU/MPS warmup、真实权重）保持作者原标注，并在置信度栏注明“未独立复核”。

| 断言（摘录） | 判定 | 评审人证据 / 复算结果 |
|---|---|---|
| A01 服务端常量 SR/FPS/WINDOW/HOP=16000/25/32000/8000，GUARD/FADE=1280/320 | ✅ | `server.py:28-29` [读码]；`scripts/verify_studio_backend.py` 输出同值，我另用 import server 打印同值 [复核] |
| A02 GUARD=80 ms、FADE=20 ms、输出块 8000=0.5 s | ✅ | `server.py:29,80-81`；1280/16000=0.08、320/16000=0.02、8000/16000=0.5 [复核算术] |
| A03 请求 32000 个 `<f4`（b64 恰 170668 字符）、50 帧 96×96 JPEG/PNG | ✅ | `server.py:43-63`；我脚本构造合法 payload 通过、49 帧/超长帧/短音频均被拒 [复核]（真实 JPEG 解码未覆盖，见 §2.3 局限） |
| A04 `end_sample` 缺省 32000、范围 `[32000, 2^53−1]`、bool/float 拒 | ✅ | `server.py:64-66`；`end_sample=True/-1/31999/32000.5/2^53` 全部 ValueError [复核] |
| A05 `window_seconds/hop_seconds` 必须 2/0.5 | ✅ | `server.py:38-39`；3 与 0.4 均被拒 [复核] |
| A06 结果块 8000 样本、b64 42668 字符、sample_rate=16000 | ✅ | `server.py:262-264`；`len(b64(8000×f32))=42668` [复核计算] |
| A07 `start_sample=endpoint−9280`、`end_sample=endpoint−1280` | ✅ | `server.py:266-267`；`OutputSlicer.take` 实际返回 `pred[22720:30720]` [复核运行] |
| A08 `rtf=ms/500`、`window_rtf=ms/2000` | ✅ | `server.py:264`；25 ms→0.05/0.013 [读码/算术] |
| A09 请求/响应 `end_sample` 含义不同 | ✅ | `server.py:64,266`；`web/app.js:678-681` 的校验式 [读码] |
| A10 切片 `[22720:30720]`、`future=[30720:31040]` | ✅ | 我独立 import `server.OutputSlicer` 运行：chunk 8000、`future` 320 [复核] |
| A11 相邻窗拼接连续（allclose，max 4.3e-8） | ✅ | 我独立复算 `\|diff\|max=4.281997678656779e-08`，与作者 4.3e-8 一致 [复核] |
| A12 连续窗首 20 ms 用上一窗 future 交叉淡化 | ✅ | `server.py:82-86`；拼接结果 = 绝对时间轴 [复核] |
| A13 `contiguous` 条件 | ✅ | `server.py:82`；我的第二窗 `contiguous=True`、跳窗 `False` [复核] |
| A14 输出 clip[-1,1] + `clipped_samples` | ✅ | 全 4.0 预测 → `clipped_samples=8000`、输出 max=1.0 [复核] |
| A15 非 32000 或含 NaN 预测被拒 | ✅ | `server.py:77-79` [读码]；作者 §8-B1 [未独立复核] |
| A16 HTTP 同源：无 Origin 放行、非同源 403 | ✅ | `server.py:175-179,198-202`；`test_server.py` 同源用例在复跑中通过 [复核 pytest] |
| A17 WS accept 前同源校验失败 close 1008 | ✅ | `server.py:235-237`；`test_server.py` 用例通过 [复核 pytest] |
| A18 TrustedHost 白名单 4 项、非法 Host→400（http/ws） | ✅ | `server.py:196`；我额外实测 WS 非法 Host 得到 `WebSocketDenialResponse`（握手拒绝），合法 Host 可连 [复核] |
| A19 CSP/nosniff/no-referrer/Permissions-Policy/no-store | ✅ | `server.py:204-210`；`test_server.py` 的 CSP 断言通过 [复核 pytest]（注意作者 CSP 摘录不完整，见 P2-7） |
| A20 `/api/load` body>1024→413、bad json/非对象→400、成功→200 | ✅ | `server.py:217-231`；`test_server.py` 通过 [复核 pytest] |
| A21 `/api/load` RuntimeError→409 | ✅ | `server.py:229-230` [读码]（此断言行本身正确；但 §5.2 表格把“无可用权重”也归 409 是错的，见 P1-1） |
| A22 消息 >2,000,000 → close 1009；恰好 2,000,000 → error 且连接存活 | ✅ | `server.py:30,243-246`；作者 §8-B6；`test_server.py` 边界用例通过 [复核 pytest 部分] |
| A23 WS error 不关闭连接、后续 result 正常 | ✅ | `server.py:268-272`；`test_server.py:test_websocket_validation_error_does_not_kill_session` 通过 [复核 pytest] |
| A24 id 校验（bool/超 2^53/超 80 字符拒绝） | ✅ | `server.py:252-256`；参数化用例通过 [复核 pytest] |
| A25 busy 时返回“模型忙，请跳过此窗口后重试。” | ✅ | `server.py:258-259`；作者 §8-B3 锁占用实测 [未独立复核] |
| A26 变体切换守卫 | ✅ | `server.py:130-141`；`test_server.py::test_unknown_or_missing_variant...` 通过 [复核 pytest] |
| A27 ready = not loading and runtime.ready；错误经 status 回传 | ✅ | `server.py:107,115,154-158` [读码]；作者 §8-B7 [未独立复核] |
| A28–A32 运行时变体/权重解析/设备选择 | ✅ | `falcon_runtime.py:31-92`；`test_weights.py` 7 项、`test_device.py` 通过（2 skip）[复核 pytest]；我另验证本机 CUDA 可用、`pick_device('auto')='cuda'` [复核] |
| A33 `preprocess_mouth` → (1,1,50,88,88) | ✅ | 我实际调用得到 `(1,1,50,88,88)` [复核] |
| A34 归一化值域 [−2.5515, 3.5091]、不是 [-1,1] | ✅ | 公式 `(x/255−0.421)/0.165` 手算：0→−2.5515、255→3.5091、107→−0.00844 [复核] |
| A35 参数 small 537,697 / video 11,186,688 / 总 11,724,385；big 1,158,617 / 总 12,345,305 | ✅ | 我用 `studio/configs/*.yml` 实际构造模型统计：small `537697/11186688/11724385`、big `1158617/11186688/12345305` [复核] |
| A36 Studio 与训练侧 state_dict keys+形状一致（audio 525 / video 137） | ✅ | 同配置实际构造：audio 525/525 keys_equal=True shapes_equal=True params_equal=True；video 137/137 同 [复核] |
| A37 `FalconSpeakerContrastive` 与 `SpeakerContrastiveV2` 同配置结构相同、可 strict 互装 | ✅ | 实际构造：keys/shapes 全等、参数均 537,697、`load_state_dict(strict=True)` 成功 [复核] |
| A38 严格加载失败路径 | ✅ | `falcon_runtime.py:143-144,164-169`；作者 §8-A7 [未独立复核] |
| A39 ckpt 允许 `{"state_dict":...}` 或裸前缀 state_dict | ✅ | `falcon_runtime.py:161` [读码] |
| A40 `weights_only=True` | ✅ | `falcon_runtime.py:160` [读码] |
| A41 warmup 真前向 + GPU 失败降级 CPU | ✅（CPU 数值未独立复核） | `falcon_runtime.py:185-196,202-205` [读码]；无权重，未独立复跑 warmup |
| A42 CPU 线程数夹取 | ✅ | `falcon_runtime.py:174-176` [读码] |
| A43 infer 输入校验 + 输出校验 | ✅ | `falcon_runtime.py:243-252,97-100` [读码] |
| A44 形状链 mouth(1,1,50,88,88)→FRCNN(1,512,50)→(1,112,251,129)→…→(1,1,32000) | ✅ | 我用 forward hook 实测：`frcnn (1,512,50) / encoder (1,112,251,129) / audio_bottleneck 同 / separator 同 / mask_generator (1,1,112,251,129) / decoder (1,1,32000)` [复核] |
| A45 Falcon.py 净 +9 行全部来自 DropPath import + Metal FFT bridge（+4/+4/+1） | ✅ | `diff -u` 实测 12 hunk、added 65/deleted 56/net +9；三个 code hunk 分别为 `1107,15→19`、`1176,6→10`、`1183,10→11`，精确 +4/+4/+1 [复核] |
| A46 SISNRRepel = Studio Falcon.py +2 注释 import + 类名替换；Speaker_Contrastive_v2 = Falcon_Contrastive.py +18 行 docstring + 基类替换 | ✅ | 实际 diff：+2 行（`#from sru`/`#from fla...`）+ 类名/`super` 两处改名；+18 行 module docstring + import/base 两处 [复核] |
| A47 Big 两份文件在 look2hear 无源 → PROVENANCE 来源表 2 项不准确 | ✅（证据措辞需修正） | `find/grep` 确认 look2hear 无 `SISNRRepel.py`/`Speaker_Contrastive_v2.py`；但 `git log --all -S 'class SISNRRepel'` 实际**返回 a7af472**（studio 文件在该 commit 加入），作者“git log 均无”的说法需限定 `-- look2hear`，见 P2-1 |
| A48 逐字节/注释级差异表 | ✅（normalizations 行数有误） | `Falcon_Contrastive/activations/resnet/shufflenetv2` cmp 相同；`normalizations.py` 仅注释差异，但实际 13+/13−=26 行，作者写 28 行（P2-3）；`frcnn` 去 CRLF 后 3 处 + 5 行头注释成立 [复核] |
| A49 同权重同输入 CPU 前向 bit-identical（合成权重 max\|Δ\|=0） | ✅ | 我独立构造 Studio/训练侧模型、互装 state_dict、同输入前向：`max\|Δ\|=0.0` [复核] |
| A50 Big vs small 配置差异仅宽度类超参 | ✅ | 我核对 `FalconBig.yml` 行号：out_chan 6/53、hid 12、layer 19/27、spk 68、video 74、audionet_name 88 全部吻合 [读码/复核] |
| A51 infer_file 一次整段前向、输出=帧数×640 | ✅ | `infer_file.py:69-82` [读码] |
| A52 服务端不写盘 | ✅ | `server.py:1`；`infer_file.py:41-44,100` [读码] |
| A53 mode 字段加载前 sliding-window、加载后 windowed_audio_visual | ✅ | `server.py:112,115`；`falcon_runtime.py:234` [读码] |
| A54 在线 demo 与 same_origin/TrustedHost 冲突 | ❓ 无法独立确认 | 代码层矛盾成立，但线上部署分支不在仓库；保持原 Q4 结论 [读码/推断] |
| A55 4 个 `__init__.py` 仅 docstring、不导出符号 | ✅ | 实际查看 4 个文件均只有一行 docstring [复核] |

**断言表结论：55 条中 54 条成立、1 条（A54）无法独立确认；无 P0 级事实错误。** 但在断言表之外的正文
（§5.2 状态码表、§7 Q8、§4.6 CSP 摘录、§1.1 形状）存在下节问题。

### 2.3 评审人独立复算记录（命令与输出摘要）

环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128，CUDA 2×A800 可用）+ 
`docs/verification/stubs`（timm/typeguard）+ `docs/verification/scripts/pil_stub`（仅协议层）。

```bash
# 1) 仓库自带 Python 测试（作者声称 33 passed/4 skipped）——我复跑一致：
PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python -m pytest studio/tests/test_server.py studio/tests/test_weights.py \
  studio/tests/test_device.py studio/tests/test_infer_file.py -q -p no:typeguard
# → 33 passed, 4 skipped, 1 warning in 3.51s

# 2) 我用 /tmp/review_slicer.py 独立 import server.OutputSlicer：
#    chunk=8000 contiguous=False future=320；第二窗 contiguous=True；
#    拼接 vs 绝对时间轴 allclose=True，max|diff|=4.281997678656779e-08；
#    全 4.0 预测 → clipped_samples=8000, max|out|=1.0；
#    b64 长度：请求 170668、结果 42668。

# 3) /tmp/review_backend_check2.py：构造 Studio 与训练侧模型，互装 state_dict：
#    audio studio_keys=525 train_keys=525 keys_equal=True shapes_equal=True params_equal=True
#    video 137/137 同；small params=537,697；big audio=1,158,617

# 4) /tmp/review_equiv.py：preprocess_mouth (1,1,50,88,88)；forward hook 形状链见 A44；
#    训练侧 vs Studio 同权重同输入 CPU 输出 max|studio−training|=0.0

# 5) /tmp/review_deepcopy.py：同一 config dict 连续构造 → 537,697 → 49,081（塌缩）；
#    每模型 deepcopy → 537,697；FSC vs SpeakerContrastiveV2（同 small 配置）keys/shapes 全等、strict 互装成功

# 6) /tmp/review_modules.py：small/big 逐模块参数量与 06 §4.10 表逐项一致
#    （encoder 3024/6912、audio_bottleneck 13104/66816、separator.audio_net 445766/847254、…、audio 合计一致）

# 7) diff 统计：Falcon.py 12 hunk/+65/−56/net+9；bridge +4/+4/+1；
#    normalizations.py 13+/13−=26（作者写 28）；frcnn --strip-trailing-cr 仅 3 行 import/path 差异

# 8) TrustedHost 对 WebSocket 的行为（作者称 http/ws 都生效）：实测
#    WS Host=evil.example → WebSocketDenialResponse；Host=127.0.0.1 → 连接成功。
```

**局限（与作者相同的诚实声明）**：无 Falcon 权重/禁止下载 → 真实 `falcon.pt` 的 key/输出一致性、
真实 warmup 延迟、M5 Pro 数据均未复核；PIL stub 不解真实 JPEG，服务端图片解码路径未复核（作者已声明）。

## 3. 错误清单

### P0（事实错误 / 会误导实现或复现）

无。

### P1（重要遗漏或误导性行为）

**P1-1 §5.2 `POST /api/load` 状态码表把“无可用权重”归入 409，实际是 400。**
- 证据：`Engine.start_load` 在变体不可用时抛 **ValueError**（`studio/server.py:140-141`：
  `raise ValueError(f"尚未安装 {selected['name']} 权重：models/{selected['checkpoint']}")`），而 `POST /api/load`
  的 `except ValueError` 映射为 **HTTP 400**（`server.py:227-228`）；`except RuntimeError` 才是 409（`:229-230`）。
- 自相矛盾：同一文档 §5.2 的守卫清单 ⑥ 已正确写“变体不可用 → ValueError”（`docs/understanding/06-studio-backend.md:660-662`），
  但表格 409 行又写“无可用权重”（`:653`）。
- 修正建议：409 行只保留“加载中切别的变体 / 有活跃 WS 连接”；把“无可用权重（ValueError→400）”单列，
  并注明实际触发条件（`huggingface_hub` 缺失时 `weights_source()` 返回 None，见 P2-2）。
- 影响：前端/集成方若按 409 处理该错误会走错分支（当前前端只展示 `body.detail`，故无功能故障，但契约文档会误导 R4/FINAL）。

### P2（细节瑕疵 / 证据偏差）

**P2-1 §4.8(d)/Q2 “`git log --all -S 'class SISNRRepel'` 均无” 不准确。**
- 证据：我实际运行 `git log --all --oneline -S 'class SISNRRepel'` → 返回 `a7af472`（即 studio/falcon_model
  在该初始 commit 中被加入）；只有限定路径 `-- look2hear` 才为空。§8.1 命令清单第 5 条也写的是不带路径的版本。
- 结论（look2hear 无 Big 源码）仍然正确，仅证据表述需修正为“`git log --all -S ... -- look2hear` 无结果”。

**P2-2 §7 Q8 “该错误文案永远不可达（仅 huggingface_hub 缺失时触发别的话术）” 与代码不符。**
- 证据：`weights_source()` 在 `huggingface_hub` 缺失时返回 **None**（`falcon_runtime.py:48-55`）→
  `available=False`（`server.py:118-124`）→ `start_load` 恰好抛出这条 `尚未安装 ... 权重：models/...`
  的 **ValueError**（`server.py:140-141`）。所以该文案**可达**，且此时文案会误导（真实原因是缺 Python 包，
  不是缺权重文件）。
- 修正建议：删除“永远不可达”，改为“仅在缺 `huggingface_hub`（或未来实现本地 models/ 探测）时可达；
  当前文案把缺包误报为缺权重，建议区分”。

**P2-3 `normalizations.py` diff 行数：28 应为 26。**
- 证据：`diff -u look2hear/layers/normalizations.py studio/falcon_model/layers/normalizations.py` 统计
  13 added + 13 deleted = **26** 行（3 个 hunk，全部注释中译）。
- 出现位置：§2.4 文件表（`06:160`）、§4.8(a) 表（`06:470`）、§8-A3（`06:967`）“28 行，净 0”。三处需统一为 26。

**P2-4 §1.1 下游形状写成 `FRCNNVideoModel`(B,1,T,**96,96**→B,512,T)，实际模型输入是 88×88。**
- 证据：`preprocess_mouth` 先 `frames[:,4:92,4:92]`（96→88）再送 `video_model`（`falcon_runtime.py:95-104,241`）；
  训练 val 管线同样是 `Normalize(0,255)→CenterCrop(88,88)`（`look2hear/datas/transform.py:147-166`；
  `look2hear/datas/avspeech_dataset.py:138`）。96×96 是**协议/采集**尺寸，88×88 才是**模型**输入。
- 这与文档 §3.5 `(50,96,96) uint8 → preprocess → (1,1,50,88,88)` 自相矛盾；建议 §1.1 改为 88×88。

**P2-5 三处行号漂移（证据应精确到行）。**
- `DualPathRNN.out_proj` 引 `Falcon.py:384-388`，实际 `self.out_proj = (` 在 **380** 行、条件表达式 380-383；
  384 起已是 `def forward`（我的 `grep -n out_proj`：380、405）。
- `NeuroSCNetFusion` 引 `Falcon.py:241-247`，实际 `class NeuroSCNetFusion` 在 **239**、`math.gcd` 243、
  `self.v_proj = nn.Conv1d` 244（`:239-246`）。
- §4.9-6 引 `audio_visual_test.py:128-130` 的 `getattr` 构造，实际 `getattr(...)` 在 **129-132**（128 是 deepcopy）。

**P2-6 §2.4 标题“`studio/falcon_model/`（16 个文件，4035 行）”计数口径不一致。**
- 4035 = 15 个 `.py` 行数之和（我按 `wc -l` 逐文件相加复核）；加上表中 `PROVENANCE.md`（42 行）应为 4077，
  或标题写“15 个 .py（4035 行）+ PROVENANCE.md”。

**P2-7 §4.6 CSP 摘录不完整且未标注“节选”。**
- 实际下发（`server.py:210`）还包含 `style-src 'self' 'unsafe-inline'`、`img-src 'self' data: blob:`、
  `media-src 'self' blob:`。其中 `media-src blob:` 是前端 `<audio>` 播放录音 Blob URL 的直接依赖，
  `img-src data: blob:` 与前端资源相关；文档只列部分指令容易被误读为完整 CSP。
- 建议：给全文，或明确标注“以下仅列与本服务行为相关的部分”。

**P2-8 §5.5 坐标定义混用，易误导协议实现者。**
- 文档先定义“window 覆盖**绝对采样** [endpoint−32000, endpoint)”（这是**协议坐标**），紧接着写
  “绝对时间轴原点 = 前端 `sessionEpoch - LEFT_CONTEXT`”（这是**采样坐标**）。两者不是同一坐标系。
- 正确关系（我按 `app.js:593,677-683` 和 `server.py:64-67,266` 反推）：**协议坐标 = 行内采样坐标 + LEFT_CONTEXT(22720)**；
  首窗窗口左边界（采样坐标）= `sessionEpoch − 22720`；协议坐标 0 ↔ 采样坐标 −22720。
- 建议：显式写两套坐标的换算，并给一个数值例：首窗请求 `end_sample=32000` → 返回 `start=22720, end=30720`
  → 采样坐标 `[0,8000)`；第二窗请求 40000 → 返回 30720/38720 → 采样坐标 `[8000,16000)`（与作者 §8-B5 一致）。

**P2-9 §2.2 `.load()` 步骤顺序与源码不符（细节）。**
- §2.2 摘要写“……strict 加载→**eval/freeze→线程数→设备**→warmup”，源码顺序是：
  统计 keys/参数（`:170-173`）→ `torch.set_num_threads`（`:174-176`）→ `eval()/requires_grad_`（`:177-180`）
  → `pick_device`（`:181`）。§3.1 的列表顺序是对的，§2.2 需对齐。

### P3（措辞/精度）

- **P3-1** §5.3 “实际每帧 b64 上限约 …≈36.5 KB”：按 `(2,000,000−170,668−JSON 开销)/50` 计算约 **36.6 KB**
  （1,829,332/50=36,586.6 字符），且需扣除 `"frames":[...]` 的 JSON 开销，实际更小；建议写“≈36.6 千字符（上限，未扣 JSON 开销）”。
- **P3-2** §5.4 `inference_ms` 写成“`_forward` 纯前向耗时”：`Engine.infer` 计时覆盖 `runtime.infer`，其中包含
  `preprocess_mouth`（`falcon_runtime.py:239`）与锁内开销（`server.py:167-169`）；“不含协议/图片解码”是对的，
  但“纯前向”不精确，建议改“含嘴部预处理的前向耗时”。
- **P3-3** §4.7 “前端 4 s 轮询 + 180 s 加载等待”：4 s 是全局 `setInterval(fetchStatus,4000)`（`app.js:1164`），
  但 `startSession` 的模型等待循环是每 **1.5 s** 一次 `fetchStatus()`（`app.js:608-619`）；建议区分两个轮询。

## 4. 遗漏的重要行为（代码里有、文档没写）

1. **多 WS 连接并发/背压矩阵缺失，且 `busy` 检查与 `infer_lock` 之间存在可观察竞态。**
   `WS handler` 先看 `engine.busy`（`server.py:258-259`）再调 `Engine.infer`；`infer()` 在锁获取失败时抛
   “另一个窗口正在使用模型…”（`:163-164`）。而 `finally` 里是**先 `busy=False` 再 `infer_lock.release()`**
   （`:169-172`），因此存在“busy 已 False 但锁仍被持有”的窗口：另一连接此时通过 busy 检查、随后拿到
   “另一个窗口”错误。文档只给单连接语义与两种错误文案，未给 2 连接时序/竞态说明。
   建议补一张两客户端时序表（A 推理中 / B 发请求 → B 收到 `"模型忙"` 或 `"另一个窗口"`），并说明
   `ws_max_queue=1`（`:333`）意味着服务端不做应用层排队、客户端应保持 single-flight（前端恰好如此，`app.js:565,588`）。
2. **每次 `result` 回包都调用完整 `engine.status()`。**
   `server.py:265` `"device": engine.status().get("device")` → `status()` 会重建 `variants()`（`:118-124`），
   对每个 variant 调 `weights_source()`（每 variant 2 次 `try_to_load_from_cache` 文件系统查询）。
   即每 0.5 s 回包做 4 次 HF 缓存查询 + 构造对象。文档 §2.1.4 只写“status() 合成返回体”，未指出这条
   “推理热路径上的状态重建”，也没评估是否应在加载完成时缓存 `device/variants`。
3. **服务端没有请求级超时/取消：infer 卡住时 WS handler 无限等待。**
   `await asyncio.to_thread(engine.infer, ...)`（`server.py:260`）没有 `wait_for`；若模型 hang，服务端不会
   主动释放连接/记录错误，只能等前端 60 s 超时断开（`app.js:589-591`）或进程退出。文档 §4.7 只写错误回传，
   未写“无超时”这一失败模式，也没有与前端 60 s 超时联动的说明。
4. **`variants[].available` 的时序语义/离线语义未展开。** `available` 等价于“`weights_source()` 非 None”
   （`server.py:118-124`），而 `weights_source` 只查本地 HF 缓存（`falcon_runtime.py:48-55`）：所以
   “可下载但未缓存”时 available=True（`source="huggingface"`）；“无 huggingface_hub”时 available=False 且
   错误文案误导（见 P2-2）；没有“本地 `models/` 目录优先”的探测（见作者 Q8）。建议把这些组合列成表。
5. **`/ws` 握手被 TrustedHost 拒绝时是“握手拒绝”而非普通 HTTP 400 响应。**
   我实测非法 Host 在 TestClient 中表现为 `WebSocketDenialResponse`（Starlette 会把 400 响应写到 WS 握手）。
   文档 §4.6 写“Host 非法 → 400（http/ws 都生效）”方向正确，但可补一句“WS 侧表现为握手失败/断连，
   浏览器看到的是连接错误”，便于和前端 `onerror/onclose` 分支（`app.js:623-632`）对上。
6. **`infer_file.read_wav` 的 32-bit float 路径不 clip、不校验范围。**
   `read_wav` 对 4 字节宽直接 `np.frombuffer(..., '<f4')`（`infer_file.py:34-38`），没有像 16-bit 路径那样
   归一化/范围检查；若 WAV 内含 >1 样本，会原样进入模型（服务端 WS 路径反而有 `|amp|≤4` 校验，`server.py:49`）。
   文档 §2.3 未提这个不对称；建议在“离线整段链”中注明。
7. **模块级 `app = create_app()` 在 import 时创建一个 Engine，而 `main()` 又 create 一个。**
   `server.py:287` 与 `server.py:332` 是两条启动路径：`python server.py` 用后者，`uvicorn server:app` 用前者；
   但 import 时模块级 app 已创建（虽不 autoload），因此测试/工具 import server 会额外实例化一个 Engine。
   文档 §2.1.6 只提“模块级 app 也可启动”，未说明 import 副作用与“双 Engine”事实（无害但会影响测试隔离理解）。

**“要完全理解项目还缺什么”（≥3 条具体建议）**
1. 补“协议时序 + 多客户端并发”一章：单连接数值例（两窗 endpoint 32000→40000 与 start/end 映射）+ 双连接
   行为表 + `ws_max_queue=1` 的背压含义 + 无请求级超时的失败模式。
2. 补“失败注入矩阵”：`/ws` 中途断开（前端保留录音）、推理超时（服务端无超时/前端 60 s）、加载中收到
   `/ws` 请求、`/api/load` 400/409/413 的真实响应、缺 `huggingface_hub`/无网/首次下载失败的 `available`
   与错误文案；最好给出每个场景下前端会看到什么（与 07 文档联动）。
3. 补“部署形态差异”：README 在线 demo 与 `same_origin`/TrustedHost 的矛盾（Q4）目前只是提问；建议明确
   “仓库内 `server.py` 不能直接对外部署”，或指出所需补丁清单（反向代理 Host/Origin 白名单、认证、TLS）。
4. 把 `infer_file` 与 `audio_visual_test.py` 的“评测协议”关系（Q3）交叉到 04/05 域并给出结论链接，
   否则“整段前向 matching evaluation protocol”的说法仍悬空。

## 5. 跨模块一致性问题

| # | 模块/文档 | 一致性判断 | 说明与建议 |
|---|---|---|---|
| X1 | 06 ↔ 07（本文） | ✅ 一致 | 协议字段、`LEFT_CONTEXT=22720`、`GUARD=1280`、首窗 9280、`start/end_sample` 公式、`contiguous` 分工全部互证；06 §5.9 与 07 §5.4 可互相引用行号，避免 FINAL 重复推导 |
| X2 | 06 Q5 ↔ 07 Q5 | ✅ 一致（且是真实契约问题） | 服务端 `busy/error` 建议“跳过窗口”，前端 `type:"error"` 一律停止整场（`app.js:672,721-724`）；建议 Lead 在 R3 前给出产品口径，否则 R4 仍会重复提 |
| X3 | 06 ↔ 01（model-core） | ✅ 一致 | 06 §4.4/§4.10 的 537,697 / 11,724,385 / 12,345,305、deepcopy 塌缩到 49,081、`FalconSpeakerContrastive` 的 bit-identical 与 01 §4.7/§8 完全一致；建议 06 P10/A49 直接引用 01 的验证，减少重复实验 |
| X4 | 06 ↔ 04（loss/metric/eval） | ✅ 结论互证、⚠️ 表述可统一 | 06 Q3 与 04 的 P0（`conf.yml:88` 的 `SpeakerContrastiveV2` 不在 `look2hear.models` → AttributeError）是同一事实；04 已把 `infer_file`/评测协议问题登记为 Q，建议 06 的 Q3 明确“由 04 域回答”，并在 04 文档回链 |
| X5 | 06 ↔ 08（ops/tests） | ⚠️ 口径不一致 | 06 §8 用 PIL stub 复跑 `test_server.py` 得到 **33 passed/4 skipped**；08 文档写“test_server 因沙箱 venv 缺 Pillow 只能收集失败”。两者都不算错（06 明确声明了 stub），但 FINAL 需要统一为“在缺 Pillow 环境下需 stub 才能跑，且图片解码未验证”，避免读者误以为测试可用性有矛盾 |
| X6 | 06 ↔ 08（requirements） | ✅ 已互相标注为 Q | 06 Q6/Q7 与 08 的依赖表都指出 `torch>=2.14.1` 与验证环境 2.8 的差异、`einops` 死依赖；建议 FINAL 用一张“声明版本 vs 本沙箱验证版本 vs 缺包”表统一 |
| X7 | 06 §5.9 与 07 的 `variants` 消费 | ✅ 一致 | 前端消费 `ready/loading/error/variant/variants/device`；06 §5.1 的“其余字段诊断用途”结论与 07 §5.1 一致；`window_rtf/clipped_samples/device` 未消费也一致 |
| X8 | 06 与 README | ✅ 基本一致 | 06 §2.7 的逐条核对（含 `output_delay_ms=80`、0.58 s/1.42 s、7860–7899、strict load）与 README 一致；README 的 M5 Pro 0.12 s 无法验证已如实标注 |

## 6. 给作者的问题清单（需回答/修订）

1. **（P1-1）** 是否同意把 §5.2 的“无可用权重”从 409 移到 400？请补一行“实际触发条件：缺 `huggingface_hub`
   时 `weights_source()=None`”，并与 Q8 一起修正文案误导（P2-2）。
2. **（P2-8）** 请把 §5.5 的坐标定义改成两套显式换算（协议坐标 = 采样坐标 + 22720），并补两窗数值例。
3. **（P2-3/P2-5/P2-6）** 请统一证据口径：`normalizations.py` 26 行；`out_proj` 380-383；
   `NeuroSCNetFusion` 239-246；`audio_visual_test.py` 129-132；`falcon_model` 文件/行数表述（15 py=4035 +
   PROVENANCE 42）。
4. **（P2-4）** §1.1 的 FRCNN 输入形状是否改为 88×88（并注明 96×96 只是协议/采集尺寸）？
5. **（§4 遗漏 1/2/3）** 是否补“多连接并发 + busy/锁竞态 + `ws_max_queue=1` + 无请求超时”小节？
   是否考虑把 `engine.status()` 从每次回包中移出（缓存 variants/device）？
6. **（§4 遗漏 6）** `infer_file` 的 32-bit float WAV 不做 clip/范围校验是有意（信任离线输入）还是遗漏？
7. **（Q1）** 真实 `falcon.pt`/`falcon-big.pt` 的键/输出一致性验证是否可在 R4 的隔离目录做一次（仅下载不跑数据）？
   若不允许，请明确 FINAL 的措辞为“合成权重下代码级 bit-identical，真实权重未验证”，避免被理解为已端到端验证。
8. **（X5）** 请与 ops-tests-docs 对齐 `test_server.py` 的可运行性口径（stub vs Pillow），避免 08 与 06 在 FINAL 中矛盾。

## 7. 变更日志

| 日期/轮次 | 内容 |
|---|---|
| 2026-10-21 / R2 | 初版评审：逐条核对 55 条断言（54 成立、1 无法独立确认）；独立复跑 pytest（33 passed/4 skipped）、`OutputSlicer` 数值、参数/state_dict/bit-identical/形状链/diff 统计/TrustedHost-WS 行为；判 P0=0、P1=1（§5.2 状态码）、P2=9、P3=3；提出 7 条遗漏与 4 条“完全理解”建议、8 个作者问题。未修改源码。 |
