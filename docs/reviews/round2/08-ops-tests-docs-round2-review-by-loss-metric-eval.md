# 评审：`docs/understanding/08-ops-tests-docs.md`（R3 v2，round2 复核）—— 评审人 `loss-metric-eval`

> 评审对象：`docs/understanding/08-ops-tests-docs.md`（768 行，R3 v2）
> 评审对象 sha256（评审开始时，`sha256sum` 实测）：`74f90d25acb7f3bb0f1b0ddd2446af353bd1c3b62882513de5faab8b809e3e58`
> （与 Lead 提供的目标哈希一致；评审结束时复校仍为该值，见 §7——本轮未修改 08 文档与任何源码。）
> 复核性质：round2 复核（换人）。重点：①R3 是否落实 studio-backend 的 R2 评审（D22、test_server 可运行口径、
> `node --check` 21/21、`studio/.gitignore` 14 行、A4 45 MB 推导、D15 修正、conf.yml 行号）；
> ②执行级验证：独立跑 pytest / Node / D22 协议探测 / 依赖与实验产物核对；③08 的 README 偏差清单中
> 与**评测/依赖**相关的条目（缺 `huggingface_hub`、`comet` 依赖、`best_k` 占位）是否仍与源码一致。
> 证据约定同 00-CONVENTIONS；本轮我新增日志：`docs/verification/runs/r4-review-08-{pytest-cache,pytest-nocache,node,scripts,d22}.log`。

---

## 1. 结论

**通过（可小修，无需回退）—— P0 = 0，P1 = 1，P2 = 5。**

- **R3 对 studio-backend R2 评审的整改已逐项落实**，且我用**独立复算**确认了 R3 的关键口径：
  - **D22**（公网 demo vs loopback 同源限制）：§1.2 #11 / §4.7 D22 / A56 / Q15 均已新增；我独立复现
    `same_origin(公网 origin, 同 host)=False`、`TestClient` 注入公网 `Host → HTTP 400 "Invalid host header"`、
    loopback `Host → 200 {"app":"falcon-studio"}`（日志 `runs/r4-review-08-d22.log`）。
  - **test_server 可运行口径**：§2.4/§8.2 已改为“注入仓库内最小 PIL stub 后可运行”，我独立复算
    **单跑 22 passed**、全量 **35 passed / 2 skipped**（当前 HF 缓存有权重，92.39 s）；把 `HF_HOME` 指向空目录
    复现**评审基线 33 passed / 4 skipped**（2.56 s）——与 08 的口径**逐项吻合**。
  - **`node --check` 21/21**：我按 13 `web/*.js` + 4 `tests/*.mjs` + 2 `tests/*.cjs` + 2 `web/*.test.cjs`
    实际跑出 **checked=21 fail=0**；8 个 Node 测试 **8/8 rc=0**。
  - **`studio/.gitignore` 14 行**（R1 曾写 18）、**A4 的 45 MB 参数量推导**（已加 `[推断]` 与算术）、
    **D15 的失实表述修正**（改为“本轮未尝试复算”，并注明 venv 实有 ptflops/CUDA×2）、
    **conf.yml causal 行号**（`:11` audio_params / `:73` video_params / `:45` 顶层）——均已落实且我核对到源码。
  - `bash -n` 6/6、脚本文件类型/执行位、`check_environment.py` 双解释器实测、`install-blackhole.sh --check`
    退出码等 R3 验证记录，抽查一致（日志 `runs/r4-review-08-scripts.log`）。
- 08 的**评测/依赖相关偏差清单仍成立**：根 `requirements.txt`（29 行）无 `huggingface_hub`（`grep -ci hugging`=0）、
  无 `comet`；`best_k_models.json` 5 条 key 全为字面 `${ROOT}/...` 且 **5/5 不存在**；`audio_visual_test.py:129`
  的 `getattr` 与 `conf.yml:88` 不一致——与 04 R3、03 的结论一致。
- 残留问题仅为**引用行号两处偏移**（P1）与**可升级/可补充的细节**（P2），不改变 08 的 P0/P1 结论。
- 另外，**04 域 R3 新获得的真实权重/MACs 证据**（见 §5）可以直接回答 08 的 Q5/Q6/D15/D20；08 当前 v2 写于
  该证据之前，建议下一版引用并关闭对应条目（这是“可升级”，不是错误）。

---

## 2. 断言逐条核对表（A1–A57）

图例：✅ 本轮独立核对（执行或读源码到行号）；✅* 与 08 自证/studio-backend 证据一致且本轮抽查源码通过；
⚠️ 结论成立但表述/证据需更新；❓ 本轮未复核（外部环境/浏览器/线上服务等）。

| 断言 | 判定 | 本轮核对说明 |
|---|---|---|
| A1 `.command` 可执行/LF/文件名一致 | ✅ | `bash -n` 6/6；`file` 均 Bourne UTF-8；`git ls-files -s` 两个 `.command`=100755 |
| A2 参数透传/失败暂停 | ✅* | 读码一致（08 证据行） |
| A3 launch 只取 python 路径 | ✅* | `launch.sh:6` 与 `bootstrap.sh` stdout 契约一致 |
| A4 缓存判断 + “约 45 MB”推导 | ⚠️ | 推导 11,724,385×4 B=44.72 MiB 成立；**真实 `falcon.pt` 已可测=47,156,343 B=44.97 MiB**（与“约45MB”自洽，差值来自 state_dict 的 buffer）→ 建议升为 `[验证]` |
| A5 下载失败提示镜像 | ✅* | 读码 |
| A6 exec 透传 CLI | ✅* | 读码 |
| A7 venv 优先级 | ✅* | `bootstrap.sh:12-22` 结构与 08 描述一致 |
| A8 Python 3.10–3.12 | ✅* | `bootstrap.sh`/`check_environment.py` 一致 |
| A9 复用条件 | ✅* | 读码 |
| A10 非法环境不删除 | ✅* | 读码 |
| A11 FALCON_PYTHON/查找顺序 | ✅* | 读码 |
| A12 lock 优先 + uv/pip | ✅* | `bootstrap.sh:70-83` 结构一致 |
| A13 安装后二次校验 | ✅* | 读码 |
| A14 doctor 链 | ✅* | 读码 |
| A15 check_environment 版本/8 包/8 import | ✅ | 双解释器实跑：system 3.8 与 falcon 3.10 均 exit=1 且错误信息与 08 §8.3 一致；import 清单不含 huggingface_hub |
| A16 行格式强制 `>=,<` | ✅ | `check_environment.py:24-27`（split 会失败） |
| A17 BlackHole `--check/--help` 只读 | ✅ | `--check` exit=0（Linux 实测） |
| A18 BlackHole 安装行为 | ✅* | 读码 |
| A19 安装器不被启动链调用 | ✅* | 读码/grep |
| A20 Setup Meeting Audio 总暂停 | ✅* | 读码 |
| A21 端口 7860–7899 / 显式范围 | ✅* | `server.py:304-316` 读码一致 |
| A22 浏览器轮询/--no-browser | ✅* | `server.py:319-329,296` 读码一致 |
| A23 设备 auto/CUDA→MPS→CPU/预热回退 | ✅ | pytest `test_device` 实测 4 passed/2 skipped；`pick_device` 行号 81 一致 |
| A24 权重解析 | ✅ | pytest `test_weights` 7 passed |
| A25 strict 加载/架构名校验 | ✅* | `falcon_runtime.py:152-169` 抽查一致 |
| A26 常量 16k/25fps/32000/50/2s/0.5s/80ms | ✅* | 常量行抽查一致 |
| A27 8 个 Node 测试全过 | ✅ | **8/8 rc=0**（本轮逐个跑，`runs/r4-review-08-node.log`） |
| A28 `node --check` 21/21 | ✅ | **checked=21 fail=0**（13+4+2+2） |
| A29 test_weights 7 passed | ✅ | collect 7；全量跑通过 |
| A30 test_device 4 passed 2 skipped | ✅ | collect 6；全量跑 4p/2s（2 skip 条件由 GPU 可用决定） |
| A31 test_infer_file 无权重整体 skip | ✅ | `HF_HOME=/tmp/hf_empty` 下 2 skipped，复现基线 33p/4s |
| A32 test_server 注入 PIL stub 后可运行、单跑 22 | ✅ | 单跑 **22 passed（1.08 s）**；无 PIL stub 时 collection error（15 collected + 1 error）也复现 |
| A33 浏览器 QA 需浏览器/MediaPipe | ❓ | 本轮未运行（无浏览器/MediaPipe），读码与 manifest 一致 |
| A34 serve_browser autoload/端口 | ✅* | 读码 |
| A35 pages.yml 触发/权限/私有仓库/artifact | ✅ | YAML 解析：name/permissions/job if/artifact path 与 08 一致；触发 push(main, paths)+workflow_dispatch |
| A36 项目页相对资源/外部 URL | ❓ | 未复跑脚本（非本域重点） |
| A37 项目页数字与 README 一致 | ✅* | 抽查 7.7/16.4/0.5M/51.3 与 README 一致 |
| A38 四份 README 相对链接 0 缺失 | ❓ | 未复跑 |
| A39 EN/ZH 代码块等价 | ❓ | 未复跑 |
| A40 best_k 5 条/`${ROOT}`/5/5 不存在/min=714 | ✅ | json 解析：5 条、全部 `${ROOT}`、`os.path.exists` 0/5 |
| A41 conf 与 root 差异（`audionet_name`/data/causal 行号） | ✅ | `:11`/`:73`/`:45` 逐行核对；差异结论与 04 R3 一致 |
| A42 causal 写回无影响 | ✅* | `Falcon.py:1314-1315` 读码 |
| A43 comet_logger 未定义 | ✅ | 全仓库仅 `audio_train.py:170`；根 requirements 无 comet |
| A44 `SpeakerContrastiveV2` 不存在 | ✅ | `look2hear/models/__init__.py:1-2` |
| A45 根 requirements 无 huggingface_hub、timm 有 | ✅ | `grep -ci hugging`=0；timm 行存在 |
| A46 Studio 8 下限与 lock 8/8、39 包 | ✅* | lock 110 行/39 个 `==`；下限脚本未复跑（沿用 08 证据） |
| A47 包索引可满足性旁证 | ❓ | 未重跑 11 次 curl（非本域重点） |
| A48 `.gitignore` 缺口/覆盖 | ✅ | `git check-ignore`：`Experiments/tensorboard_logs/...` NOT IGNORED；`studio/.venv` 与 `*.ckpt` 已覆盖 |
| A49 evaluated_mac_params 硬编码 CUDA/config、`--exp_dir` 未用 | ✅ | 读码（并见 §3 P2-1：还缺 pretrain 前置条件） |
| A50 check_environment import 清单不含 hub | ✅ | 读码（8 模块 import 清单） |
| A51 默认 conf 路径不存在 | ✅ | `configs/` 仅 Falcon.yml；`audio_visual_test.py:65` 路径不存在 |
| A52 Experiments 只有 2 文件 | ✅ | `find` 实测 |
| A53 `large/Dolphin.wav` 大小写 | ✅ | `index.html:517` 大写 vs `:451,484,550` 小写（1 vs 3） |
| A54 project-map “5 个”不准确 | ✅ | `00-project-map.md:71` 原文确认；实际 6+2 |
| A55 README Studio 数字与代码一致 | ✅* | 抽查端口/2s/hop/80ms 与代码一致 |
| A56 公网 demo vs loopback 冲突 | ✅ | **独立复现**：`same_origin=False`、公网 Host→400、loopback→200（`runs/r4-review-08-d22.log`） |
| A57 pytest 注入最小 PIL stub 后可运行（基线 33p/4s、本轮 35p/2s） | ✅ | **两项均独立复算**：空 HF 缓存 33p/4s（2.56 s）、当前缓存 35p/2s（92.39 s）；日志见 §7 |

**统计**（共 57 行 = A1–A57）：**✅ 27、✅* 24、⚠️ 1**（A4 可升级）、**❓ 5**（A33/A36/A38/A39/A47，外部环境/浏览器/线上项未复核）——**无 ❌**。

---

## 3. 错误清单

### P0（事实错误/复现阻断）：0 条

R3 的 2 个 P0（D1 `comet_logger`、D2 评测命令 `AttributeError`）经复核**仍成立**，且与 04 R3/03 的结论一致。

### P1（证据行号错误，按 00-CONVENTIONS “行号错=P1”）：1 条（两处引用偏移，同一类）

**P1-1 两处 `audio_visual_test.py` 行号偏移**

- `08:242`（§2.8 conf.yml 表）：写 “被 `audio_visual_test.py:196-198/273-275` 的 `cwd` 重算取代”。
  实际：`process_worker` 的 `exp_dir` 在 **196-198**；`main()` 的 `exp_dir` 在 **285-287**（`273-275` 是
  `gpu_list/world_size/print`）。证据：`audio_visual_test.py:270-288`。
- `08:30` 与 `08:429`（D4）：写 “最终 `FileNotFoundError`（`audio_visual_test.py:103-107`）”。
  实际 raise 在 **`:108`**；`103-107` 是 `glob/if/print/return` 段（`sed -n '98,110p'` 核对）。
- 修正：`:196-198/285-287` 与 `:108`（或 `:101-108`）。不影响结论，属证据引用精度问题。

### P2（细节瑕疵/可升级项）：5 条

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| P2-1 | D6（§4.7） | D6 只列 “CUDA + cwd=仓库根” 两个前置；但 `evaluated_mac_params.py:51-53` 用 `**videonet_config` 构造视频模型，**保留 `pretrain`**（`configs/Falcon.yml:8`）→ 无 `pretrain_zoo/frcnn_128_512.backbone.pth.tar` 时先 `FileNotFoundError`（我在 04 A50 已复核）。 | D6 补“还需 pretrain_zoo 的视觉 backbone（否则 FileNotFoundError）” |
| P2-2 | D7（§4.7） | 写 “文档/推理管线是 96×96（`falcon_runtime.py:27`…`preprocess_mouth` 取 96→88）”——**96×96 是存储/接口格式，模型输入是 88×88**（`falcon_runtime.py:99-104`；04 §2.3.2 [验证]）；同句自相矛盾。 | 改为“96×96 存储 → 中心裁剪 88×88 后送 FRCNN”；64×64 只是 MACs 脚本的 mouth_emb 生成输入，不影响 MACs 数值 |
| P2-3 | D15/Q5/D20/Q6（§4.7/§7/§8.5） | 写于 04 R3 之前：“MACs 未尝试复算”“权重文件未下载/无法验证体积与内容”。**这些现已有 [验证] 证据**（见 §5）：MACs=10.611 G（audio-only）、`falcon.pt` 662 张量=525+137、47,156,343 B=44.97 MiB；`falcon-big.pt` 517+137、49,645,087 B。 | 引用 04 R3（`runs/r3-weights.log`、`runs/r3-macs.log`）后关闭/更新 Q5、Q6、D20，D15 改为“MACs 已由 04 复算、延迟仍未复算” |
| P2-4 | A4（§6） | 45 MB 仍是 `[推断]`（推导 44.72 MiB）；真实文件 44.97 MiB（差 state_dict buffer）。结论成立，但可升级。 | 升为 `[验证]`：`falcon.pt` 47,156,343 B=44.97 MiB ≈ “about 45 MB”；注明与纯参数推导的 0.25 MiB 差来自 buffer |
| P2-5 | `08:767`（§9.1 末行） | 自引用哈希（`88cd1978…`）是写入该行**之前**的哈希；当前文件哈希是 `74f90d25…`，二者必然不同。虽已注释说明，但对复核者仍易误判。 | 删掉自引用行，或改为“R3 提交时哈希（外部记录）：…”并注明当前文件哈希会变 |

---

## 4. 遗漏的重要行为（代码里有、08 未写 / R3 后新增事实）

1. **评测/验证还可能因单声道数据直接崩溃**（04 的 P0-3、05 §8.6、data-pipe supplement §2）：即使 D2 的
   `audionet_name` 与 `${DATA_ROOT}` 都修好，若真实数据是 mono，`AVSpeechEvalDataset` 会把 `mix` 退化为
   0 维并在 `torch.stft` 抛 `RuntimeError`（评测入口不捕获；训练期 val 同样崩）。08 的 D2/D3 只列了两类
   阻断点，建议加一句“数据需为多声道（见 04 P0-3/05 §8.6）”，否则读者会误以为修完 conf.yml 就能评测。
2. **`evaluated_mac_params.py` 的 pretrain 前置条件**（P2-1）与 **`audio_train.py:4` import 期
   `torch.cuda.empty_cache()`**：08 在启动链/依赖章节未提这两个可执行前置；后者会在 import 阶段初始化
   CUDA 上下文（对纯 CPU 机器无影响，但属于训练入口副作用）。
3. **真实权重已存在后，§8.5 的“未验证清单”第 1 条（权重下载/加载/推理一致性）应拆细**：
   “下载/格式/体积/严格加载”已可验证（04 R3：`falcon.pt`/`falcon-big.pt` 的 525+137/517+137 keys、
   `strict=True`、真实 `_load_models_from_ckpt`）；“训练代码与 Studio runtime 输出完全一致”仍只有
   `PROVENANCE.md` 声明（CPU bit-identical）与合成权重证据，仍需 backend/model-core 复核。
4. **test_infer_file 单文件 ~68–90 s**（本轮全量 92.39 s）：08 已记录耗时，但未提示它明显拉长 CI 时间；
   若未来加 CI，建议 `-k "not infer_file"` 或单独 job（P2 级建议）。

---

## 5. 跨模块一致性问题

| 涉及 | 04 R3（本域）已有证据 | 对 08 v2 的影响/建议 |
|---|---|---|
| `--ckpt_path hf://` 评测（08 D3） | 04 R3：`falcon.pt` 顶层 `{architecture:'FalconSpeakerContrastive', epoch:714, state_dict}`，525+137 keys，真实 `_load_models_from_ckpt` 成功 → **ckpt 格式不是阻断点**；真实阻断是 D2（conf.yml 类名）+ 数据路径（04 B2/B3/B4） | 08 D2/D3 结论正确；建议 D3 注一句“若环境已装 hub，仍会被 D2 阻断”，避免读者误以为装 hub 即可 |
| MACs（08 Q5/D15） | 04 R3：ptflops 复算 **10.611 G / 0.538 M**（audio-only；视觉 64²/88²/96² = 7.77/15.90/17.48 G 未计入；README 10.6 G 对齐） | 关闭 Q5（本域答复）；D15 改为“MACs 已复算一致，延迟未复算” |
| 权重体积/内容（08 Q6/D20/A4） | 04 R3：`falcon.pt` 662 张量 = 525+137、47,156,343 B=44.97 MiB；`falcon-big.pt` 517+137、49,645,087 B=47.35 MiB；`PROVENANCE.md:36-40` 的 epoch/architecture 声明与实测一致 | 更新 Q6/D20/A4；仍保留“输出一致未独立复验”的边界 |
| `conf.yml` 来源（08 Q1/D2/A41） | 04 R3 实测 `falcon.pt.architecture='FalconSpeakerContrastive'`、epoch=714 → 发布 small 权重对应 FalconSpeakerContrastive；`conf.yml` 的 `SpeakerContrastiveV2` 属旧/私有分支痕迹（与 08 Q1 的 06 域初答一致） | 可在 Q1 补一句“已由真实权重 metadata 进一步坐实” |
| mono 条件性 P0（04/05） | 04 P0-3、05 §8.6、`docs/verification/data-pipe-supplement.md` §2：评测与训练期 val 均崩；本域 R3 独立重跑（`runs/r3-mono-eval-repro.log`） | 08 D2 增补引用（见 §4-1） |
| 03 训练系统 | 03：`comet_logger` P0、config 污染、conf.yml 不能由当前代码产生；08 D1/D4/A41 与之完全一致 | 无矛盾；建议 08 Q2/Q4 引用 03 §4.5/§9.1 的结论 |
| 06 Studio / PROVENANCE | 06 结论：`falcon_runtime.py` strict 加载两前缀；发布文件含 `state_dict`/architecture/epoch | 08 D20/Q6 可更新（§5 第 3 行） |

---

## 6. 给作者（ops-tests-docs）的问题清单与建议

1. **请修正 P1-1 两处行号**：`§2.8` 的 `273-275 → 285-287`、D4 的 `103-107 → 108`（或 `101-108`）。
2. **是否接受用 04 R3 的真实权重/MACs 证据更新 Q5/Q6/D15/D20？** 具体数字见 §5；若接受，建议在 §7 把
   Q5 标“已由 loss-metric-eval 回答”、Q6 拆成“格式/体积已验证 / 输出一致性未验证”。
3. **D6 补 pretrain 前置条件、D7 改“96×96 存储→88×88 模型输入”**（P2-1/P2-2）；这两处会直接影响用户
   复现 `evaluated_mac_params.py` 与理解视频输入尺寸。
4. **A4 是否升级为 `[验证]`**（真实文件 44.97 MiB）？同时说明与纯参数推导 44.72 MiB 的 buffer 差异（P2-4）。
5. **自引用哈希行是否删除/外置**（P2-5）？当前写法会让复核者把 `88cd1978…` 与本轮 `74f90d25…`
   对比时产生困惑。
6. **D2 是否把 mono 条件性 P0 加进“评测命令不可复现”的原因链**（引用 04 P0-3 / 05 §8.6）？这是本轮
   round2 复核中发现的最值得补的一句“跨域前提”，否则 08 的评测偏差清单不完整。

---

## 7. 结束校验与本轮执行级验证记录

- 评审对象 sha256（评审结束时重算）：`74f90d25acb7f3bb0f1b0ddd2446af353bd1c3b62882513de5faab8b809e3e58`
  ——与开始时一致，**未修改 08 文档、未修改任何源码**（`git status` 仅未跟踪的 `docs/`）。
- 本轮独立执行（命令与日志）：

```bash
cd /root/code/github_repos/Falcon-fork
# 1) pytest（当前 HF 缓存有权重）→ 35 passed, 2 skipped in 92.39s
PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python -m pytest studio/tests -q -p no:typeguard \
  > docs/verification/runs/r4-review-08-pytest-cache.log
# 2) pytest（空 HF_CACHE 复现评审基线）→ 33 passed, 4 skipped in 2.56s
HF_HOME=/tmp/hf_empty PYTHONPATH=... -m pytest studio/tests -q -p no:typeguard \
  > docs/verification/runs/r4-review-08-pytest-nocache.log
# 3) test_server 单跑 → 22 passed in 1.08s；collect: device 6 / infer 2 / server 22 / weights 7
# 4) 8 个 Node 测试 → 8/8 rc=0；node --check → checked=21 fail=0
#    → docs/verification/runs/r4-review-08-node.log
# 5) D22 协议探测 → same_origin(public)=False；公网 Host→400 "Invalid host header"；loopback→200 falcon-studio
#    → docs/verification/runs/r4-review-08-d22.log
# 6) bash -n 6/6、file、git ls-files -s、check_environment 双解释器、install-blackhole --check
#    → docs/verification/runs/r4-review-08-scripts.log
```

- 静态核对（本轮实际命令摘要）：`wc -l studio/.gitignore`=14；`node --check` 21 文件；根 `requirements.txt`
  =29 行且 `grep -ci hugging`=0、`grep -ci comet`=0；`best_k_models.json` 5 条/全 `${ROOT}`/0 存在；
  `git check-ignore` 验证 `*.ckpt`/`.venv`/`tensorboard_logs` 三种情况；`conf.yml` 第 11/45/73/88/99-102/
  119-120/126-127/142-159 行逐行核对；`index.html:517` vs `:451,484,550`；`00-project-map.md:71`。
- 结论：**08 R3 可信、可进入 R5 汇总**；P1-1 两处行号与 5 条 P2 均为小修，不阻塞。
