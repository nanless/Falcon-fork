# 00 总纲 · Falcon 项目全景、当前状态与复现规划 v2

> 编写：Lead（2026-10-24）· 仓库 `/root/code/github_repos/Falcon-fork`，分支 `repro/phase2`，HEAD `2beec38`
> 环境：conda `falcon`（Python 3.11.17 / torch 2.8.0+cu128 / 2×A800 80GB，当前全空）
> 本文记录 2026-10-24 的项目理解；2026-10-31 后的状态和待办汇总在 `docs/STATUS.md`。`THREE_ROUND_REVIEW_20261030.md` 记录当时的问题清单。
> **2026-11-08 校正**：参照 `docs/understanding/deepdive-20261108/` 更新与后续事实冲突的内容。
> 已改 7 处：§0-1 论文已于 2026-10-23 公开（arXiv:2610.32749）；§0-4 补 RealSSA-Sim 实际分母；§0-5/§4.2/R6 改用 full4 DNSMOS 并写明口径；§0-6 的「3 份索引」状态已修；§7 两行旧文档状态已标已修；§8-1、§8-4 标为已解决。
>
> 本文是 9 个子代理深挖文档（01–09）的汇总入口。**每一条关键结论都指向 01–09 中的证据章节**；
> 数字与口径以 01–09 为准，本文只做交叉索引与决策建议。

---

## 0. 项目与复现结果

Falcon 是音视频目标说话人提取模型：输入混合语音和目标人物的嘴部画面，输出该人物的一路语音。仓库同时包含训练、评测与 Falcon Studio 窗口化演示。

- **模型规模：**small 音频模型 537,697 个参数，冻结视觉前端 11,186,688 个，两者合计 11,724,385 个；音频侧 ptflops 计数为 10.611 GMAC。音频模型中 5,568 个参数属于当前前向未调用的模块。
- **窗口与因果性：**STFT 使用 256 采样窗和 128 采样 hop。去除全窗归一化影响的特定实验中，单遍 AudioBlock 几何前视为 57 帧、约 456 ms，两遍整链为 113 帧、约 904 ms。发布实现的归一化使用整个输入窗口的统计量；Studio 每 0.5 秒处理一个 2 秒窗口。
- **RealSSA-Sim：**历史全量评测的 SI-SNRi 为 7.774 dB、SDRi 为 8.479 dB，有效样本分别为 9,169 和 9,166 条。论文报告 7.7 / 8.2 dB；发布 checkpoint 的完整训练来源待确认，评测通道与论文口径需一并列出。
- **LRS2-2Mix：**发布权重历史结果为 7.029 / 7.713 dB，自训 epoch 124 结果为 8.858 / 9.270 dB；论文表格为 14.1 / 14.7 dB。论文同协议训练结果仍待完成。
- **RealSSA-Real：**full4 对 54 段得到 DNSMOS OVRL 2.7941、P808 3.3365；full3 因窗口起点错误作废。各次结果的窗口、通道与权重记录见[当前复现状态](../../STATUS.md)。
- **数据与训练：**LRS2 的单方向与双方向索引使用不同的样本数；RealSSA-Sim near 负样本的构造和 8 卡训练需要对应数据、配置与运行记录。

模型、数据、训练和应用的现行解释见 [模型与流式处理](../20261110-model-and-streaming.md)、[数据集说明](../20261110-datasets.md)、[数据与训练](../20261110-data-and-training.md)及[评测与应用](../20261110-evaluation-and-apps.md)。

---

## 1. 阅读指南（按你想回答的问题）

| 你想知道 | 读 | 一句话答案 |
|---|---|---|
| 模型结构/参数/因果性/真实权重是否对得上 | **01-model-core** | 537,697（生效 532,129）/11,724,385/10.611 G；falcon.pt strict 525+137 keys、ep714；train↔studio max|Δ|=0；窗口级因果 |
| STFT/归一化/TFRB/视频骨干怎么实现 | **02-layers-dsp-video** | `cLNhw = 累积LN(全窗GN)`；TFRB=频双向LSTM+时单向LSTM+tril attention；FRCNN 冻结 11.19 M/±80 ms |
| 训练入口能不能跑、怎么跑、8 卡怎么配 | **03-training-system** | F1–F9/F12/F17 已修并实测；F15（DDP gather）仍在；8 卡 runbook 有 9 项前置检查与 3 个坑 |
| 指标怎么算、7.03 为什么不等于 14.1 | **04-losses-metrics** | 评测实现见 `look2hear/metrics/core.py`；已测指标变体差异 ≤0.4 dB，14.1 与实测差异还需结合训练协议和权重核对 |
| 数据在哪、怎么转、near 负样本怎么办 | **05-data-pipeline** | LRS2 链路可从零重建；RealSSA 只有 tt 本地、tr/cv 在 TOS；LRS2 的 near.json 已被一次性脚本改成 s2（K=1） |
| Studio 能不能演示/部署、延迟多少 | **06-studio-backend** | 真实权重全链路等价；A800 单窗 47.1/56.5 ms，CPU 543 ms（不可实时）；demo 级、非生产 |
| 前端/Space/i18n/工程完成度 | **07-studio-frontend-space** | 原生 ES Module 实时客户端，Node 8/8；Space 是从 studio 组装的 ZeroGPU 变体；i18n 5 处漂移 |
| 哪些证据可信、缺什么 | **08-verification-ops-env** | 数字基本都能对上，但证据分级严重不均；日志/CSV 大量未入库或被覆盖 |
| 论文每个数字对齐到哪、下一步规划 | **09-benchmark-alignment** | 16 行基准矩阵 + 口径差异 + P0–P5 规划 v2 + 证据审计补遗 |

---

## 2. 项目是什么（两条产品线，一套模型定义）

`look2hear/` 是上游官方代码（NeurIPS 2026 Falcon）：模型、层、损失、指标、数据、Lightning 训练系统。
`studio/` 是官方 Web 演示（FastAPI + WebSocket + 原生 ES Module 前端）：真实权重、2 s 窗口化推理、
麦克风/摄像头实时目标说话人提取。两者共享同一模型定义，`studio/falcon_model/` 是推理副本（去掉训练依赖、
加 Apple Metal 的 CPU FFT bridge），**在 CPU 上对同一权重/输入逐位一致（max|Δ|=0）**（01 §2.6、06 §3.2）。

- 官方发布权重：`falcon.pt`（small，`FalconSpeakerContrastive`，epoch 714，audio 525 + video 137 keys）；
  `falcon-big.pt`（big，`SpeakerContrastiveV2`，epoch 715，audio 517 + video 137 keys，总 12,345,305 参数）。
  **训练侧只有 small 的代码**，big 只能推理（06 §4.4）。
- 论文按 benchmark 分别训练（README:228 “under our training protocol”），但只发布了 1 个（+big）权重；
  `Experiments/checkpoint/Falcon/conf.yml` 含 `LRS3_complex_nonmoving` 和 ep714 线索；发布权重的实际训练数据仍待原始训练记录确认。

---

## 3. 系统地图（模块 → 深挖文档 → 关键文件）

| 子系统 | 深挖文档 | 关键文件 | 关键结论 |
|---|---|---|---|
| 模型核心 | 01 | `look2hear/models/{Falcon.py,Falcon_Contrastive.py}` | 0.538 M 音频 / 11.72 M 总 / 10.611 G MACs；4 种参数口径 |
| 层与 DSP | 02 | `look2hear/layers/*`、`utils/*`、`videomodels/*` | 只有 normalizations/activations 被真正使用；其余是库/死代码 |
| 训练系统 | 03 | `audio_train.py`、`look2hear/system/*`、`configs/*.yml` | 入口已可运行；8 卡配方就绪；F15 未修 |
| 损失/指标 | 04 | `look2hear/{losses,metrics}/*`、`audio_visual_test.py` | core.py 实现评测指标；训练、验证和评测中的名称含义不同 |
| 数据管线 | 05 | `look2hear/datas/*`、`DataPreProcess/*`、`tools/repro/*` | LRS2 可重建；RealSSA 训练未就绪（tr/cv 在 TOS、near 缺失） |
| Studio 后端 | 06 | `studio/{server.py,falcon_runtime.py,infer_file.py}` | 32000/8000/1280/22720/320；GPU 实时、CPU 不可 |
| Studio 前端/Space | 07 | `studio/web/*`、`space/*`、`project_page/*` | 8/8 Node 测试；Space 组装 54 文件/i18n 226 条 |
| 验证/运维 | 08 | `docs/verification/**`、`tools/repro/tests` | 证据分级；日志未入库；索引过期 |
| 基准对齐/规划 | 09 | `README.md`、Phase1/2 报告、火山 runbook | 16 行矩阵；LRS2 8 卡为候选训练路线 |

---

## 4. 当前状态（2026-10-24）

### 4.1 四层复现状态

| 层 | 状态 | 证据级别 | 说明 |
|---|---|---|---|
| **L1 推理/结构** | ✅ 完成 | L1 | 参数/MACs/权重 strict/bit-identical/Studio/因果性全部复算（01、06） |
| **L2 评测** | ⚠️ 部分 | RealSSA-Sim L2、LRS2 L2 | RealSSA-Sim 7.774/8.479；通道和权重身份待核；RealSSA-Real DNSMOS 见后续校正；LRS2 14.1 未复现；LRS3/Vox 不可复现 |
| **L3 训练** | ⚠️ 单机小配方 | L2 | LRS2 130 epoch 8.858/9.270；8 卡 1000 epoch 未跑；RealSSA-Sim 从零训练未开始 |
| **L4 消融/效率** | ⚠️ 少量 | L1（P2-A/B） | 视觉/窗口化/模型规模/mouth 偏移/协议变体已做；TFRB/GCI、NASS 配方消融、3090 延迟未做 |

### 4.2 基准矩阵（摘要；完整 16 行见 09 §1）

| Benchmark | 论文 | 实测 | 结论 |
|---|---|---|---|
| RealSSA-Sim SI-SNRi / SDRi | 7.7 / 8.2 | mean **7.774/8.479**；ch0（论文通道）**7.977/8.452** | 数值接近；通道和权重身份待核 |
| RealSSA-Real DNSMOS | 3.276 | P808 **3.3365** / OVRL **2.7941**（full4；full3 的 3.3285/2.7901 作废） | ⚠️ 口径已定：论文是 P.835，论文结果使用 RealSSA-trained ckpt（2026-10-30 校正） |
| LRS2-2Mix SI-SNRi / SDRi | 14.1 / 14.7 | 官方权重 7.029/7.713；自训 8.618/9.042；+对比 8.858/9.270 | ❌ 未复现（差 5–7 dB） |
| LRS3-2Mix / VoxCeleb2-2Mix | 16.4/16.7、10.5/11.8 | 无数据 | ❌ 不可复现 |
| 参数量 / MACs | 0.5 M / 10.6 G | 537,697 / 10.611 G | ✅ |
| 延迟（3090） / RAM / 显存 | 51.3 ms / 125.3 MB / 364.7 MB | 未测（A800：47.1/56.5 ms 单窗） | ⚠️ 硬件不同 |

### 4.3 资产与磁盘（实测）

| 资产 | 位置 | 大小/条数 | 状态 |
|---|---|---|---|
| RealSSA-Sim 全量 | TOS `/aispeech-cfs/voiceprint/data/multimodal/realssa_sim` | **521,532,645,652 B / 242 文件（240 shard）** | ✅ 归档 `ok:true`（rsync 309 MB/s, 26m48s）；本地仅 tt 19 shard（38G） |
| RealSSA-Sim tt 转换 | `/root/data/falcon_realssa_sim/tt` | 30G / **9,169 条** | ✅ 可评测（near 全 fallback→K=0） |
| RealSSA-Real | 本地 `/root/data/RealSSA-Real` 12G；TOS 12.84G | 422/423 文件 | ✅ 已评测 DNSMOS |
| LRS2-2Mix 原始 | `/root/data/lrs2_2mix/lrs2_rebuild` | 33G（tar 16.47 + 解压 17.39） | ✅ |
| LRS2 单目标转换 | `/root/data/falcon_lrs2_2mix` | 24G / tr20k cv5k tt3k | ✅ 训练+评测；near.json 已被改成 s2（K=1） |
| LRS2 论文配方数据 | `/root/data/falcon_lrs2_2mix_paper` | 26G / tr40k cv10k tt6k（双向） | ✅ 待 8 卡训练 |
| 官方权重 | `/root/.cache/falcon_weights/`（symlink 到 group-shared） | falcon.pt / falcon-big.pt / backbone | ✅ |
| 训练产物 | `/root/tmp/p2d_train`、`/root/tmp/p2d_near_train` | 100 epoch + 30 epoch ckpt | ⚠️ 临时目录，未入库 |
| 磁盘 | `/` 20G（14G avail）；`/root` 164T（14T avail） | — | RealSSA 全量转换需 ≥600 GB / 全量约 0.8–0.9 TB |

### 4.4 测试与环境现状

- `pytest studio/tests`：**35 passed / 2 skipped**（37 collected；2 skip 为 GPU 条件用例）——06 本次实测。
- Node 前端测试：**8/8**（RC=0）——07 本次实测。
- `pytest tools/repro/tests`：**104 passed / 1 failed / 7 skipped**（QA 记录“111/1”已过期；失败由 CSV 空 SDR 触发）——08。
- `import look2hear.{models,system,metrics}`、`audio_train`、`audio_visual_test`：OK（08 §1.2）。
- `studio/scripts/check_environment.py`：**exit=1**（它要求 torch>=2.14.1 / numpy>=2.4.6 / Pillow<13，与项目强制版本冲突）→ 不能作为门禁（06 §3.5）。
- 环境锁：`conda-env-lock.txt` 含 **18 个 `@ file:///root/wheelhouse`** 绝对路径、`pypesq` shim 不在锁内；重建需按 `AGENTS.md` §5（08 §4-R6）。

---

## 5. 复现规划 v2（完整版见 09 §4）

> 与旧 `docs/REPRODUCTION_PLAN.md` 的差异：旧计划的 Phase 2 “RealSSA 待审批/待评测”已被现实超越（已批/已下载/已评测）；
> Phase 3 的“用官方权重复现 LRS2 14.1”已被证伪（需 LRS2-trained ckpt）；Phase 4 的“训练复现”现在有了单机 8.858 起点与
> 8 卡 runbook。下表是**只剩信息增量的动作**。

| 阶段 | 内容 | 验收 | 成本/依赖 |
|---|---|---|---|
| **P0** | 证据收口：把 `/root/tmp`、`/root/data` 下的 eval 日志/JSON 复制进 `docs/verification/phase2/` 并取消 `*.log` 全局忽略（或 `!docs/verification/**/*.log`）；修 CSV 覆盖问题；修 `DATA_ACCESS_GUIDE`/`phase2/README`/`P2E-report` 过期状态；修 1 个回归失败；补测 `epoch=99`；commit+tag | 第三方 clone 后能复核每个“已复现”数字；`pytest tools/repro/tests` 全绿 | 0.5–1 天，无 GPU |
| **P1** | RealSSA 评测补强：ch0 vs mean、整句 vs 窗口、big 对照；RealSSA-Real DNSMOS 长窗/拼接 + P808/OVRL 双报 + 向作者确认 3.276 口径 | 通道/协议/口径敏感性表；DNSMOS 结论可判定 | 1–2 天，1 GPU |
| **P2** | **LRS2-2Mix 8 卡论文配方**（主战场）：数据迁移 → 9 项前置检查 → 训练 → tt 3,000 评测 | SI-SNRi ≥13.6、SDRi ≥14.2（±0.5）；报 seed/最佳 epoch/全部差异 | 2–3 天 + 5–15 h×8 卡 |
| **P3** | RealSSA-Sim 从零训练（可选）：TOS 恢复 tr/cv 450 GB → 转换 100,826/9,133 → 新配置 → 训练；near 三选一并声明 | val SI-SNRi 上升；test 7.7±0.5（同分布） | 5–7 天；合规限制 |
| **P4** | 消融/效率：NASS 有/无、多 seed、K≥2 的 β 消融、TFRB/GCI（另开分支）、延迟/RAM/显存 | 消融表 + 效率表（注明硬件/单 seed） | 3–5 天 |
| **P5** | 最终报告 `docs/REPRODUCTION_REPORT.md`：四类状态矩阵、口径、差异声明、复跑命令、证据索引 | 可复现报告 | 1–2 天 |

**Gate 更新**：G1（入口可跑）✅ 已过；G2（RealSSA-Sim 主结果）✅ 已过；G3（LRS2 列）⏳ 待 8 卡；
G4（训练复现）⏳ 部分（单机小配方）；G5（消融/效率）⏳ 未开始。

---

## 6. 跨领域风险 Top 12（合并 01–09）

| # | 风险 | 级别 | 证据/影响 |
|---|---|---|---|
| R1 | **证据不在 git**：`*.log` 被忽略、CSV 被覆盖、关键日志在 `/root/tmp` | P0 | 08 §4-R1/R2；“已复现”无法第三方复核 |
| R2 | **文档/索引过期**：仍写 RealSSA 403、训练进行中、best 13.118（2026-11-08 已修：403→已归档、训练→已结束、best→ep99 13.2014） | P0 | 08 §4-R3；与磁盘/事实矛盾 |
| R3 | **LRS2 14.1 训练配方未公开**：8 卡配方是“合理复现”不是论文配方；near=另一说话人、K=1 使 β no-op | P0（科学） | 04 §5.3；09 §5 |
| R4 | **RSSA-Sim 训练未就绪**：tr/cv 在 TOS、无训练配置、near 缺波形（K=0） | P1 | 03 §5.2；05 §2 |
| R5 | **RealSSA-Sim 评测协议未证同论文**：整段 4.6–5.8 s clip，非 2 s 流式 | P1（科学） | 08 §4-R4 |
| R6 | **DNSMOS 口径曾未定**：3.276 是 P808 还是 OVRL？（2026-10-30 已定：论文为 P.835、属 RealSSA-trained ckpt；窗口化推理偏差仍在） | P1 | 04 §5.4；09 §1；论文 arXiv:2610.32749 |
| R7 | **“causal/streaming/51.3 ms”表述风险**：窗口级因果、51.3 ms 是 3090 单窗前向 | P1（对外） | 01 §5.4；06 §0 |
| R8 | **同名不同义**：训练 PIT=负 SNR、val=SI-SDR(est)、评测=SI-SNRi | P1（报告） | 04 §5.2 |
| R9 | **8 卡 DDP 风险**：F15 per-bucket all_gather、`gpus:-1` 全局 batch 随卡数变、DATA_ROOT 只用于自检、CSVLogger version_1 | P1 | 03 §5.3 |
| R10 | **环境可重建性**：lock 绝对路径、pypesq shim、typeguard<3、check_environment 版本冲突 | P2 | 08 §4-R6；AGENTS |
| R11 | **Studio 非生产**：无 `--checkpoint`、无鉴权/TLS、单飞锁、i18n 漂移、Space 双份维护 | P2 | 06 §4；07 §4 |
| R12 | **数据/许可/磁盘**：RealSSA cc-by-nc-nd+gated（派生数据不得入库/转发）；`/root` 92% 满；RealSSA 本地与 TOS 重复 | P2 | 05 §2.3；08 §4-R12 |

---

## 7. 与旧文档的关系（哪些结论过期）

| 旧文档 | 过期/需修正的内容 | 处理建议 |
|---|---|---|
| `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md`（2026-10-22） | §10 的 F1–F6 等 P0 多数已修；"训练入口不可跑"已不成立；LRS2 7.03/7.71 的归因已由 Phase 2 完成 | 保留历史，勿再引用其 P0 清单作为现状；以 03 为准 |
| `docs/REPRODUCTION_PLAN.md`（2026-10-21） | Phase 2/3 状态、RealSSA gated、LRS2 目标“用官方权重复现 14.1”、F1–F12 待办 | 以 09 §5 的差异表为准；P0–P5 取代其分阶段计划 |
| `docs/REPRODUCTION_PHASE2_REPORT.md` | §5 编号混排（5.1–5.4）；“训练继续中/最佳 13.118”过期；§5.4 未提交 | P0 修复：补 ep99 评测、更正训练结束时间、提交 §5.4 |
| `docs/verification/phase2/{README,P2D-report,P2E-report,QA-report}` | README 停在 `af74cca`；P2D 只写到 epoch 0；P2E 写 403；QA 写 111/1（2026-11-08 已按 deepdive-20261108 统一校正） | P0 统一更新索引与状态 |
| `docs/DATA_ACCESS_GUIDE.md` | 仍写 RealSSA 403 待批（多处；2026-11-08 已改为“已获批/已下载/已归档”，403 留作历史） | P0 更新为“已获批/已下载/已归档” |
| `docs/understanding/0X-*.md`（Phase 0，8 份 ~900 行） | 行号/结论部分过期（comet、PL1.x、严格因果等） | 01–08 已做逐条【确认/修正/推翻】，引用以深挖版为准 |

---

## 8. 未解问题 Top 10

1. ~~论文未公开~~：2026-10-23 已公开（arXiv:2610.32749），训练配方、消融设置、评测协议已可逐条核对
   （见 `deepdive-20261108/11`）；论文宣称的“无时域 lookahead/120 ms”与代码不符，改列本表第 9 条。
2. 官方 `falcon.pt` 在 `conf.yml` 留有 `LRS3_complex_nonmoving`/ep714 线索；权重的实际训练数据和与 RealSSA-Sim 的关系等待作者或原始训练记录确认。
3. LRS2 14.1 需要 LRS2-trained ckpt；作者 issue #1 未见回复。
4. ~~RealSSA-Real DNSMOS 3.276 的列名未定~~：2026-10-30 已定，论文用 **P.835**，论文结果使用 RealSSA-trained ckpt；本机 full4 OVRL 2.7941 不可直接比。
5. LRS3-2Mix / VoxCeleb2-2Mix 数据不存在。
6. `concat_layers[2]` 5,568 参数闲置是否为上游漏调用。
7. F15 DDP all_gather 在 8 卡的挂死风险未验证。
8. best ckpt（ep99）已补测，结果为 8.726/9.122；跨 seed 稳定性仍待比较（目前单 seed）。
9. RealSSA-Sim near 负样本无法忠实复现（官方未发布）。
10. 3090 延迟、RAM 和显存尚待在对应硬件上测量。

---

## 9. 证据索引

**本目录（9 份深挖 + 总纲）**：

| 文件 | 行数 | 内容 |
|---|---|---|
| `00-SYNTHESIS.md` | 本文件 | 全景/状态/规划 v2/风险/索引 |
| `01-model-core.md` | 586 | 参数四口径、MACs、TWRB/GCI 映射、真实权重、因果性、A01–A56 判定 |
| `02-layers-dsp-video.md` | 250 | STFT/归一化/TFRB/MHSA/FRCNN/typeguard；本次 `verify_layers_dsp.py` exit=0 |
| `03-training-system.md` | 463 | 训练入口修复现状、协议数字、8 卡前置检查、RealSSA 训练缺口 |
| `04-losses-metrics.md` | 608 | PIT/NASS/MDCL、指标 bug 修复、DNSMOS、三名不同义、P2-F 抽验 |
| `05-data-pipeline.md` | 484 | 数据资产表、mix=s1+s2 实测、near.json 漂移、再复现步骤 |
| `06-studio-backend.md` | 622 | 窗口协议/延迟/权重/部署边界；pytest 35/2 |
| `07-studio-frontend-space.md` | 504 | 数据流/WS 协议/Node 8/8/Space 226 i18n |
| `08-verification-ops-env.md` | 437 | 环境自检、数字出处、证据分级、R1–R12 缺口 |
| `09-benchmark-alignment.md` | 362 | 16 行矩阵、口径差异、P0–P5、证据审计补遗 |

**仓库内一手证据**：`docs/verification/{phase1,phase2,runs,scripts}/`（明细见 08 §2）；
`Experiments/checkpoint/Falcon/{conf.yml,best_k_models.json}`；`configs/LRS2_2Mix_Paper_8gpu.yml`；
`tools/repro/phase2/*`。
**仓库外一手证据（P0 需固化）**：`/root/tmp/realssa_sim_eval_official.log`、`/root/tmp/p2d_eval_{epoch86_full,contrastive_epoch124}.log`、
`/root/tmp/p2d_train|p2d_near_train/**/metrics.csv`、`/root/data/falcon_eval/realssa_real_mean_full3/{summary.json,*.csv}`（历史 full3 因窗口起点错误作废；full4 结果见当前复现状态）、
`/root/data/RealSSA-Real`、TOS 归档 `_MANIFEST.json`/`_archive_done.json`。

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-24 | 9 个子代理并行深挖（01–09），Lead 汇总 00；独立复核 RealSSA-Sim/LRS2/DNSMOS 数字、证据审计、基准矩阵与规划 v2 |

> 历史记录按实验日期阅读；现行数据与协议口径见 [当前复现状态](../../STATUS.md)。

## 11. 执行记录（2026-10-24）

用户授权后已执行 P0 收口（完整清单：`docs/verification/phase2/P0-FIX-report.md`）：

- **证据入库**：9 类关键 eval/训练证据复制进 `docs/verification/phase2/`；`.gitignore` 放行 verification 日志/CSV。
- **补测（已完成）**：ep99 全量 **8.726/9.122**（真实最佳 val 13.2014）+ 官方权重全量 **7.029/7.713**
  （恢复被覆盖的官方权重评测证据，与 Phase-1 完全一致）。
- **可复现**：`--near-mode other-speaker`（s2 负样本）落地并逐条验证；CSV 空串回归修复；conda lock 去绝对路径。
- **文档**：6 份过期状态加 2026-10-24 校正；`AGENTS.md` 更新（lock 115 行、deepdive 入口、证据入库约定）。
- **回归**：`tools/repro/tests` **109 passed / 6 skipped / 0 failed**；`studio/tests` 35/2。
- **仍在跑/待办**：P1 ch0 评测（转换已完成）、P2 8 卡训练、P3 RealSSA 从零训练。

**2026-10-30 追加**：上游已同步（`689555c → 439cbe9`，HF 下载计数 + arXiv 链接）；**论文已公开 arXiv:2610.32749**
（RealSSA-Sim 1–5 近场 + 10–15 远场干扰、encoder-space speaker contrastive loss），此前“论文未公开”类结论作废；
仓库分支统一为 `main` 并已 push（`origin/main=d076be9`，远端多余分支已清理）；协议对齐见
`docs/understanding/PAPER_ALIGNMENT_20261030.md`（DNSMOS=P.835、RealSSA channel-1、−SI-SNR/τ1000、near 可重生成）。ep86 逐句 CSV 已补跑恢复；
RealSSA-Sim ch0（论文通道）全量评测完成：**7.977/8.452**，与 mean 差 +0.20/−0.03 dB（通道影响 <0.3）。
