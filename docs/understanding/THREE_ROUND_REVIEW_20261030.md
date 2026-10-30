# Falcon 项目三轮详细 Review（2026-10-30）

> 执行：10 个 subagent/轮 × 3 轮（独立审计 → 交叉复核 → 闭环核实）+ 10 个 subagent 压缩复核（补足被输出上限截断的部分）
> 仓库 `main` @ `a3eb0d7`（工作区 clean）· 环境 conda `falcon`（Py3.11 / torch 2.8.0+cu128 / 2×A800 空闲）
> 论文已公开：arXiv:2610.32749（2026-10-23）· 关联：[00 总纲](deepdive-20261024/00-SYNTHESIS.md)、[09 基准对齐](deepdive-20261024/09-benchmark-alignment.md)、[PAPER_ALIGNMENT](PAPER_ALIGNMENT_20261030.md)

---

## 0. 方法与总体结论

三轮共产出 **150 条候选问题**，闭环核实后 **124 条 real_open / 17 条 already_done**。第一轮大 workflow 的返回结果被工具输出上限截断（66 KB 上限，丢约 120 KB），因此补跑一轮 **10 agent 压缩复核**，按 10 个子域各取 ≤5 条最高优先级问题，得到可执行结论。完整 captured JSON 见 `docs/verification/phase2/three-round-review-20261030-captured.json`。

**总体判断：模型数值与核心复现结论没有新的 P0；真正阻塞交付的是三类问题——**

1. **数据构建有 1 个真 P0**：`build_lrs2_paper_data.py --copy-wavs` 会让 target/near 同名塌缩，静默产出错配标签；而 runbook 恰好推荐该 flag。
2. **评测/证据口径有 P1**：RealSSA-Real 窗口化评测首块起点算错（每条估计前 1.34 s 恒为静音），已发布的 54 段 DNSMOS 建立在该输出上；CSV 无有效计数与 mixture 基线列，关键数字不可离线复核；自训 ckpt 无哈希且只在 `/tmp`。
3. **训练配方与论文不一致 + 8 卡依赖**：所有 LRS2 配置仍是 `pairwise_neg_snr` + β/200，论文 Table 16 是 −SI-SNR + τ/1000 + 8×3090 bs4 seed42；RealSSA-Sim 无训练配置且 near 全 fallback（K=0，NASS 只剩正项）。按现状启动长训，结果无法归因到论文配方。

---

## 1. P0（1 条，立即修）

| ID | 问题 | 证据 | 动作 |
|---|---|---|---|
| A5-1 | `build_lrs2_paper_data.py --copy-wavs` 使 target/near 同名塌缩：条目 A 的 near 变成 target 自身、条目 B 的 target 与 mouth 来自不同说话人，静默错配；runbook §2 推荐该 flag，测试零覆盖 | `build_lrs2_paper_data.py:96-106` 中 s1p/s2p 同为 `target/<stem>.wav`，`if not d.exists()` 跳过第二次复制；`PHASE2_VOLCANO_LRS2_TRAINING.md:49` | 输出名改 `<stem>.s1.wav`/`<stem>.s2.wav`，断言 target≠near；构建后校验 mouth 说话人==target 说话人、每 split 计数 40000/10000/6000，不符非零退出；补 `tools/repro/tests` 两条路径用例 |

---

## 2. P1（13 条，按“阻塞交付”排序）

| ID | 问题 | 证据（摘要） | 动作 |
|---|---|---|---|
| A4-1 | **RealSSA-Real 窗口化评测首块起点错**：`endpoint = LEFT + HOP` 应为 `HOP + GUARD`，每条估计前 21,440 样本（1.34 s）永不写入 → 前段恒静音；已发布的 SIG 3.2209 / OVRL 2.7901 建立在该输出上 | `tools/repro/phase2/eval_realssa_real.py:405,424`；对照 Studio 首窗 `app.js:624`/`server.py:266`；模拟 1 s 段输出 100% 为 0 | 改初值为 `HOP+GUARD`，对输出覆盖率加断言（filled 必须 100%），`--save-estimates` 留证后重跑 54 段 DNSMOS（约 20 min），更新 summary/对齐表 |
| A1-1 | 模型构造原地 `pop` 污染 config，且污染已固化进真实 `.ckpt` 的 hparams/training_config：由 ckpt 的 hyper_parameters 重建得 **49,081 参数**（原 537,697），audio_net 为空 ModuleList；conf.yml 反而干净 | `Falcon.py:991-996`；`av_litmodule.py:227`；实测 `last.ckpt` 缺 4 个 `repeats/shared` 键 | Falcon/Separator 入口 deepcopy 子配置；`audio_train.py` 建模前 deepcopy 给 system；补回归“由 ckpt hparams 重建=537,697” |
| A2-1 / A9-1 | **论文称因果、音频 ≤15 帧（120 ms），代码实测 456 ms（57 帧）且 AudioBlock 两遍**；README 仍写 causal/实时。端到端单点扰动可波及全部更早输出（first_changed=1，过去段 max\|Δ\|=1.6e-3） | 论文 §4.2/§6.4/App E；`Falcon.py:765,801,814` `causal=False`；`02-layers-dsp.md:482`；`README.md:43` | 文档统一为“窗口级因果、单遍 456 ms、无有限 lookahead”，撤回 120 ms/≤15 帧；真流式另建左侧 padding 分支并重训，不与发布权重结论混写 |
| A3-01 / A9-2 / A10-1 | **训练配方仍是 SNR + β/200**（论文 −SI-SNR + τ/1000 + 8×3090 bs4 seed42）：`Falcon.yml`、4 个 LRS2 配置、Paper_8gpu 全部一致；runbook 还写“β/200、seed 未给、论文未公开” | `configs/LRS2_2Mix_Paper_8gpu.yml:100,105,146`；`Falcon.yml:117,110`；`PHASE2_VOLCANO_LRS2_TRAINING.md:12,18,22,121` | 8 卡配置改 `pairwise_neg_sisdr` + β1000 + bs4（global 32）+ seed42；runbook 对照表逐行标注“论文/仓库/判定”；旧配置标 legacy |
| A3-02 | RealSSA-Sim **无训练配置**，且 near.json 9169/9169 自引用 → K=0，NASS 负项恒 0、β 退火空转，与论文 Eq.8-10 不等价 | `ls configs/*.yml` 无 RealSSA；`/root/data/falcon_realssa_sim/tt/near.json` 实测全 fallback；`avspeech_dataset_near.py:190-191`；`nass_contrastive_loss.py:73-74` | 开训前按论文 App A.2 重建 near（LibriSpeech+RIR）或同 batch 他说话人近似，并新建 RealSSA 配置；短期无法重建则标注“正项-only 近似”，不用于对照 Table 3/6 |
| A5-2 | near 审计把 fallback 哨兵计为“有 near”（打印 “Near file access OK”），实际 K=0；生成端静默写哨兵且 exit 0；唯一独立校验器 `check_avspeech_dataset.py` 仍写死 `${DATA_ROOT}` 不可运行 | `avspeech_dataset_near.py:126-132,190-191`；`generate_near_json.py:64-65,89`；`check_avspeech_dataset.py:319` | near.json 记 `{mode,sample_rate,生成命令}`；Dataset 按哨兵真计数并打印 K 分布、fallback>0 时 warn；`--strict` 模式；修好或新增 `audit_data.py` |
| A5-3 | LRS2 源包与 TOS 恢复均无校验和：`fetch_data.py` 无 `--sha256`，runbook 裸 curl 16.47 GB；paper 数据 JSON 引用仓库外绝对路径，不自包含 | `fetch_data.py` 无 hash 校验；`DATA_ACCESS_GUIDE.md:102,109` 有 sha256 但未落地；`build_lrs2_paper_data.py` 默认 copy_wavs=false | `fetch_data.py --sha256`（LRS2 `ebdfdf…`；RealSSA 用 `_MANIFEST.json`）；TOS 恢复后校验 242 文件/521,532,645,652 B；paper 数据修 A5-1 后用 `--copy-wavs` 或输出相对路径清单 |
| A8-1 | **clean tree 下 7 个 skip 恰好全是 Phase1 验收不变量**（T1 comet、T3 metrics、T5 骨干、compile 等），从未执行；“108 passed” 高估覆盖 | `tools/repro/tests/test_phase1_regression.py:101-117` 用 `git diff main^` 判“落地”，tip 之后提交即 False；`-rs` 实测 7 条全为该判据 | `_landed()` 改 689555c..HEAD 累计 diff（或缺证据即 fail），验收 `-rs` 输出 0 skipped |
| A6-1 | Studio 服务端**没有本地/自训 checkpoint 入口**（仅 infer_file.py 有），LRS2 ep86/99/124 与 RealSSA 自训权重无法在官方前端加载 | `studio/server.py:151,290-297`；`infer_file.py:91` | CLI + `/api/load` 增加 `--checkpoint`/`FALCON_CHECKPOINT`，透传 `FalconRuntime`；`/api/status` 回显文件名与 sha256 |
| A6-2 / A7-01 | **Studio/Space/仓库三套互斥 torch 版本**：studio 要求 `torch>=2.14.1`/`numpy>=2.4.6`/`Pillow<13`，lock 锁 2.14.1/2.4.6/12.3.0，Space 锁 2.13.0，而唯一已验证环境是 2.8.0+cu128/2.2.6/13.0.0；`bootstrap.sh` 失败后会按 lock 原地覆盖安装到 `FALCON_VENV` 指向的 falcon env | `studio/requirements.txt:3,4,9`；`requirements-lock.txt:79,54,60`；`space/requirements.txt:2`；`bootstrap.sh:24-27,70-83` | 以 conda lock 为单一事实源重生成/删除 studio 锁；Space 版本单独标注；bootstrap 在目标环境导入自检通过时零安装返回；`--check-only` 只告警 |
| A4-2 | **缺 WER / speaker-sim / NISQA**，论文 Table 4 的 4 个数值无法交付；仓库无任何 ASR/声纹/质量模型实现 | `PAPER_ALIGNMENT_20261030.md:152-153,227-228`（WER=Mega-ASR、sim=WeSpeaker ResNet34-LM、NISQA）；`grep` 无实现 | 二选一并写进对齐表：接 WeSpeaker+NISQA 并把模型名/版本/sha 写入 run_metadata（Mega-ASR 可得性另查）；或明确标 `not attempted` |
| A10-2 | 唯一剩余主线依赖 8 卡，本机只有 2×A800；ETA 估计互相冲突（8 卡 33–50 h vs 单卡 100 ep 实测 24.5 h → 2 卡 1000 ep 约 120 h） | `nvidia-smi -L`；`run_lrs2_paper_volcano_8gpu.sh:41` 断言 ≥8；`09:196`；`P0-FIX-report.md:94` | 锁定 8 卡节点（排队/预算/owner）或正式批准 2 卡降级（bs5+梯度累积），先实测 epoch 成本并更新 ETA/验收 |
| A10-3 | **没有 STATUS / 最终复现报告**；`REPRODUCTION_PLAN`、`DATA_ACCESS_GUIDE`、`PHASE2_REPORT` 正文仍写论文未公开、RealSSA 403、DNSMOS ±0.05、训练继续中（只在顶部加了横幅） | `ls docs/REPRODUCTION_REPORT.md` 不存在；`PLAN:27,86,116,320,391`；`REPORT:16-17,105-107,178`；`GUIDE:24` | 建 `docs/STATUS.md` 作为唯一事实源（结论/证据级别/复跑命令），就地改写过期正文，再生成最终报告；AGENTS 只指向 STATUS |

---

## 3. P2（按子域，均建议在 8 卡长训前处理）

| 子域 | 问题（ID） | 动作 |
|---|---|---|
| 模型 | `get_model_args` 只有 `{n_src}`，`best_model.pth` 被自家评测拒绝（A1-2）；MDCL/NASS 两套损失契约分裂、`near_audio` 被 kwargs 吞掉（A1-3）；`causal` 开关惰性（A1-4） | 返回完整可重建 args 或强制随包 conf.yml；只保留一套对比损失入口、margin/alpha/beta 进配置；或删模型侧 MDCL |
| 层/骨干 | FRCNN 预训练**静默部分加载**：缺键随机初始化后整网冻结、无报错，默认 `pretrain_zoo/` 不存在（A2-2）；`requirements.txt` 未 pin `typeguard<3`，而触发死导入的 `layers/stft_tfgn.py` 全仓无模型引用（A2-3） | `update_frcnn_parameter` 显式校验 missing/unexpected；`init_from` 先查路径；pin typeguard 或删死导入 |
| 训练 | 8 卡脚本 `DATA_ROOT` 只用于自检、训练读配置写死路径、`NPROC` 未用（A3-03/S3-06）；续训 ckpt 缺失会静默从零训练、监控只读 `version_0`（A3-04）；F15 条件 `all_gather` 未修（A3-05）；paper cv 无 `_meta.json` → val near 桶全 0near（S3-08） | 脚本渲染 conf 并断言生效路径/8 卡/global batch；ckpt 缺失即退出；监控 glob `version_*`；固定 5 次 gather；val near 计数改从 near.json |
| 评测 | CSV 静默 nanmeans 且无有效计数（A4-3）；丢弃 `si-snr_mix/sdr_mix` 使 SI-SNRi 不可离线复核（A4-4）；`metrics_combined.csv` 固定路径静默覆盖、shards 删除无 provenance（S4-06）；eval 工具零单测（R2-S3-123） | 增 `n_*` 列与 mix 基线列；写 `run_provenance.json`，已存在默认拒绝覆盖；补 eval 工具测试 |
| 数据 | paper 构建非原子写、静默丢条、失败仍 exit 0（A5-4）；Dataset 三份 JSON 长度无断言（S5-04）；RealSSA 归档在世界可写挂载且派生目录无 NOTICE（A5-5） | 复用 `_atomic_json_dump` + 最小 .pt 校验；长度断言；0700 目录 + NOTICE + 交付检查 |
| Studio/前端 | `available` 恒真、无本地 `models/` 回退（A6-4）；权重未 pin revision/hash（A6-3）；i18n 5 条 runtime 英文异常漏译（A7-03）；Space 测试被 `slow` 整模块挡住且无 CI（A7-02）；`doctor.sh` 在版本检查处中止（R2-S5-131）；项目页缺 OG/canonical/description（A7-04） | 三态权重探测 + local 回退；pin revision + sha256 校验；补 i18n 用例并接入 CI；doctor `--check-only`；补 meta |
| 证据 | freeze tag 落后 HEAD 9 个提交（缺 ch0/ep86 证据）（A8-2）；自训 ckpt 无 sha256 且只在 `/tmp`（A8-3）；lock 无 cu128 源说明（A8-4）；入库 2 个失效 pid + 1 个 0 字节 CSV（A8-5）；phase2/README 的 P2-D/P2-E 状态行仍是 10-21（R2-S9-143） | 重打 `repro-freeze-20261030` 并 push；ckpt sha+归档；lock 顶部加两步重建命令；清理 pid/0 字节；更新状态行 |
| 论文/规划 | README 用单 ckpt 宣称 paper results（A9-3）；消融缺 stage/τ/mask/overlap 配置（A9-5）；P3/P4 依赖不存在的 near 波形却计划 K≥2 β 消融（A10-5）；无 pytest/Node CI、基线 35/2、109/6、108/7 互相矛盾（A10-4） | README 标注 ckpt 训练集与跨语料；补最小消融开关；P3/P4 降级声明；加最小 CI 并统一基线 |

---

## 4. P3（可选/清理项）

- Studio 副本丢 `_mdcl_loss` 别名、1492 行复制必然再漂移（A1-5/A2-4）；`wujian_videomodel.py` 4D/5D 契约相反（S2-6）；`Falcon.forward` 不校验 mouth 帧与音频时长匹配（R2-S1-112/113）；`get_model_args` 不可重建（S1-03）。
- `eval_dnsmos.py` 对 |x|>1 静默峰值归一化，可虚高 OVRL 最多 +1.16 MOS（S4-04）；`eval_realssa_real.py` 的 part_order 重排是死代码且默认路径不存在（S4-07）。
- `08-ops-tests-docs.md` 仍把 `Dolphin.wav` 大小写写成待确认 P1（实际 `large/Dolphin.wav`=200、`large/dolphin.wav`=404，是真实 key）（S7-12）。
- ch0 报告 cp 文件名旧（S8-10）；0 字节 CSV/PID（A8-5）；`stubs/typeguard.py` 与 13 个脚本仍写 `/root/.venvs/falcon`、stubs 会遮蔽真实 timm（S2-5）。

---

## 5. 文档不一致清单（本轮实测，建议与 STATUS 一起修）

| 文件 | 现状 | 应为 |
|---|---|---|
| `docs/REPRODUCTION_PLAN.md` | :27/:61/:391/:457 仍写“论文未公开 / Paper link will be added”（顶部横幅已声明作废） | 正文就地改/划线，指向 PAPER_ALIGNMENT |
| `docs/REPRODUCTION_PHASE2_REPORT.md` | :16-17/:105-107/:178 仍写“训练继续中 / RealSSA 403 / P808 3.3285 最接近” | 顶部加 historical 横幅并改这三处为终态 |
| `docs/verification/phase2/README.md` | P2-D（:22）仍写“训练继续中”、P2-E（:16）仍写 403、P2-G 仍写 96/6 | 更新为终态与当前 108/7 |
| `docs/verification/phase2/P2E-report.md` | :6 头注仍写“RealSSA 仍 gated/403”（§9 已校正） | 改头注并指向 §9 |
| `docs/understanding/deepdive-20261024/04-losses-metrics.md` | :532 Q1 仍问 DNSMOS 是 OVRL 还是 P808（§8 已答） | 标记已解决 |
| `docs/understanding/deepdive-20261024/09-benchmark-alignment.md` | :13-15 结论仍写“DNSMOS 口径未定”、:35 同一行两种说法；:406-407 写 origin/main=d076be9 | 改 P.835 已确认；origin/main=a3eb0d7 |
| `AGENTS.md` | :66 写 studio 35/2（现 38/2）；:101 写 tests 35/2；无当前 tools/repro 基线 | 统一写当前基线 + 采集时间/HEAD |
| `docs/understanding/08-ops-tests-docs.md` | D16 仍把 Dolphin.wav 大写当 P1 待确认 | 改为“已确认真实 key，禁止规范化” |
| 多个文档 | 111/1、109/6、108/7、96/6、35/2、38/2 六种基线并存 | 只保留“0 failed”判据 + 当前快照 |
| 仓库 | 0 字节 `P2B-full-big-whole-real.csv`、2 个 `*.pid` 被跟踪 | 删除/移出索引并加断言 |

---

## 6. 已确认完成 / 不需要重做的（17 条 already_done 摘要）

参数四口径与 MACs（537,697 / 11,724,385 / 10.611 G）；官方权重 strict 加载与 train↔Studio bit-identical；RealSSA-Sim mean 7.774/8.479 与 ch0 7.977/8.452（两通道均 ±0.3）；LRS2 官方跨语料 7.029/7.713；ep86/99/124 三份逐句 CSV；near-mode other-speaker 与 5 条逐条一致验证；CSV 空串回归修复；conda lock 去 `file://`；P0 证据 141 文件入库；upstream 439cbe9 合并与冲突解决；Studio tests 38/2、Node 8/8；git 分支收口为 main 并推送。

---

## 7. 建议执行顺序

### 第 1 天（无 GPU，可并行）
1. **A5-1**（唯一 P0）：修 `--copy-wavs` 命名并加标签一致性闸门。
2. **A4-1**：修窗口起点 bug + 覆盖率断言，重跑 54 段 RealSSA-Real DNSMOS（GPU ~20 min），更新 P2E summary 与对齐表。
3. **A8-1**：修 `_landed()` 判据，让 7 个 Phase1 守卫真正执行。
4. **A10-3**：建 `docs/STATUS.md`，按 §5 清单改过期正文；重打并 push `repro-freeze-20261030`。
5. **A8-5 / 小清理**：删除入库 pid 与 0 字节 CSV；ch0 报告 cp 文件名；`Dolphin.wav` 文档。

### 第 1 周（单卡）
6. **A1-1/A1-2**：ckpt 自描述 + deepcopy 修复 + 重建回归。
7. **A2-2**：FRCNN 严格加载校验（8 卡前的阻塞项）。
8. **A3-03/A3-04**：8 卡脚本渲染路径、断言 8 卡/global batch、ckpt 缺失退出、监控读最新 version。
9. **A4-3/A4-4**：CSV 增有效计数与 mixture 基线列。
10. **A5-2/A5-3**：`audit_data.py` + near 真计数；fetch/TOS 校验和。
11. **A6-1**：Studio `--checkpoint`；**A6-2/A7-01**：三套 torch 版本收口与 bootstrap 守卫。

### 需要外部资源（不在本机可完成）
12. **8 卡 LRS2 论文配方**：先把配置切到论文 Table 16（−SI-SNR、τ/1000、bs4、seed42），再排队/租 8 卡（预计 33–50 h；若用 2 卡需先实测，1000 ep 约 120 h）。
13. **RealSSA-Sim 从零训练**：先按 App A.2 重建 near（LibriSpeech+RIR），再恢复 TOS tr/cv 450 GB、转换 100,826/9,133 条、8 卡 33–50 h；完成后才可能做 RealSSA-trained ckpt 的 DNSMOS（论文 3.276）与 Table 6 对照。
14. **RealSSA-Real 扩展指标**：WeSpeaker（speaker-sim）、NISQA、Mega-ASR（WER）——需额外环境。
15. **LRS3-2Mix / VoxCeleb2-2Mix**：数据未发布，维持不可复现结论；不写信、不伪造。

---

## 8. 复核入口

- 本报告对应的三轮工作流：`falcon-3round-review`（30 agents；150 候选/124 open/17 done）与 `falcon-review-converge`（10 agents，补足截断）。
- 原始 captured 问题：`docs/verification/phase2/three-round-review-20261030-captured.json`（43 条闭环条目，含 final_verdict/priority/evidence/next_action）。
- 既有证据/规划：[09 基准对齐](deepdive-20261024/09-benchmark-alignment.md)、[PAPER_ALIGNMENT](PAPER_ALIGNMENT_20261030.md)、[P0-FIX-report](../verification/phase2/P0-FIX-report.md)、[P2D-report](../verification/phase2/P2D-report.md)。

---

## 9. 2026-10-31 执行状态（本轮修复）

已修复并验证（详见 `docs/STATUS.md` 与各文件 diff）：

| review ID | 状态 | 结果 |
|---|---|---|
| A5-1（P0） | ✅ 修复 | `--copy-wavs` 改 `target/near/<stem>.s1|s2.wav` + 构建闸门（计数/等长/target≠near/mouth 说话人），测试 5 例 |
| A4-1（P1） | ✅ 修复+重跑 | 窗口首块 `HOP+GUARD` + 覆盖率断言；full4：OVRL 2.7941 / P808 3.3365（full3 作废；mixture 基线逐位不变） |
| A1-1/A1-2（P1） | ✅ 修复 | deepcopy 消除 config 污染（同 dict 连构 537,697/537,697）；`get_model_args` 完整可重建，serialize→from_pretrain 525 keys、max\|Δ\|=0 |
| A2-2（P2） | ✅ 修复 | FRCNN 严格校验 missing/unexpected/shape，缺键/坏 payload raise；官方骨干 137 keys strict、11,186,688 全冻结 |
| A3-01/A3-03/A3-04（P1/P2） | ✅ 修复 | `LRS2_2Mix_Paper_v2.yml`（−SI-SNR、τ/1000、bs4、seed42）；8 卡脚本渲染 DATA_ROOT/断言 NPROC/global batch；`--ckpt_path` 缺失退出；监控 glob version_* |
| A4-3/A4-4（P2） | ⏳ 进行中 | CSV 增加 mix 基线列与逐列 n_*，ep99/官方权重重跑（losses-metrics 收尾中） |
| A5-2/A5-3（P1） | ✅ 部分 | 新增 `audit_data.py`（K 分布/哨兵/可读性）与 `fetch_data.py --sha256/--bytes`；Dataset 侧真 K 计数与 near.json 元数据仍待做 |
| A6-1/A6-2/A7-01..04（P1/P2） | ✅ 修复 | Studio `--checkpoint`/status sha；bootstrap/doctor 安全守卫；依赖版本对齐；Space static 测试+CI；i18n 5 条；项目页 meta |
| A8-1/A8-3/A8-4/A8-5（P1/P2） | ✅ 修复 | Phase1 7 个 skip 改为 fail-on-missing 并全部执行；pid/0 字节清理；lock 顶部两步重建命令+可移植测试；ckpt sha256 记录 |
| A10-3（P1） | ✅ 修复 | 新增 `docs/STATUS.md` 单一事实源；PLAN/GUIDE/PHASE2/README/04/09/08/AGENTS 过期正文就地修正；README 中英补因果/ckpt 注记 |

仍开放（需资源或后续轮次）：LRS2 8 卡论文配方训练、RealSSA-Sim near 重建+从零训练、RealSSA-Real WER/sim/NISQA、LRS3/Vox 数据、3090 延迟、Studio big 副本（SISNRRepel/Speaker_Contrastive_v2）config 污染与 get_model_args、Dataset 侧 K 计数/near 元数据、`--save-estimates` 嵌套 id 路径问题。
