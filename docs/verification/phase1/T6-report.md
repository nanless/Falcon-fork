# Phase 1 · T6 执行报告：数据获取（LRS2-2Mix 下载/结构核实 + RealSSA 访问指南）

> 任务：`task-14`（负责人 `ops-tests-docs`）；分支 `repro/phase1`；日期 2026-10-21。
> 交付：`tools/repro/fetch_data.py`、`docs/DATA_ACCESS_GUIDE.md`、`docs/verification/phase1/`
> （`T6-data-fetch.log`、`run_lrs2_download.sh`、`T6-download.pid`、`T6-lrs2-structure.txt`、本报告）。
> 环境：`/root/miniforge3/envs/falcon/bin/python`（3.11.17）；Node 不涉及；数据全部在仓库外 `/root/data/`。
> 未修改任何源码、未 `git commit`、未提交任何 HF 访问申请。

---

## 0. 结论（TL;DR）

1. **LRS2-2Mix 下载完成并校验通过**：`lrs2.tar.gz` **16,470,508,590 B**，sha256
   `ebdfdf31335f288cc2e4b1690a0cc2f260e1826a68a020723ad4ce59a033767e`，与 HF `x-linked-etag` **一致**；
   下载 2026-10-21T16:31:18Z → 16:54:08Z（≈22m50s，≈12 MB/s，断点续传脚本后台运行），日志 `T6-data-fetch.log`。
2. **重要更正：LRS2-2Mix 不是“纯音频”**。完整扫描 249,670 entries（未解压流式 `tar tzvf`）确认它是
   **完整 2 s AV 重建包**：`audio/wav16k|wav8k/min/{tr,cv,tt}/{mix,s1,s2}`（各 28,000 组，2.000 s）+
   `mouths` npz (50,96,96) + `faces` mp4 (224×224,25fps,50 帧) + `landmarks` + 混音脚本/清单。
   ⇒ **不需要 LRS2 原始视频即可复现 LRS2-2Mix 的 AV 评测/数据构建**；`docs/REPRODUCTION_PLAN.md` §5.2/§11.2 的
   “7,234 wav、无视频/嘴部”结论**已过时**（建议 Lead 同步修订）。
3. **目标参考已天然具备且可逐样本精确复核**：实测 `mix == s1+s2`（20 样本/组×3 split，max|Δ|=3.05e-5，纯 16-bit 量化），
   `s1`/`s2` 即 create_wav_2speakers.m 生成的 scaled target；`scaling.mat` 保存 scaling_16k(N,2)/scaling16bit_16k(N,1)，
   如需按原始公式重建可精确复算（无需再从 raw_audio 反推）。
4. **HF 访问状态**：token 存在且有效（whoami=`nanless`）；**RealSSA-Sim / RealSSA-Real 均未获批**（dataset card/清单可读，
   非 README 数据文件 403 `GatedRepoError`）；LRS3-2Mix / VoxCeleb2-2Mix 仓库**只有 README，无数据文件**；未提交任何申请。
5. **`tools/repro/fetch_data.py` 已交付并通过验收**：`--dry-run` 三数据集全部可跑；gated 未授权时清晰打印申请入口/步骤
   并返回 rc=3；LRS2 走 curl 断点续传；支持 `--shard/--pattern/--full/--out/--endpoint/--proxy`；实测小文件真实下载通过。
6. **网络注意**：本机 `hf-mirror.com` TLS 不可达（`SSL_ERROR_SYSCALL`），实际走 `huggingface.co` + 本机 mihomo 代理
   `127.0.0.1:17897`；脚本/指南均已写明。

---

## 1. LRS2-2Mix 下载执行记录

| 项 | 值 |
|---|---|
| 数据集 | `JusperLee/LRS2-2Mix`（HF dataset；API `gated=False`, private=False） |
| 源 URL | `https://huggingface.co/datasets/JusperLee/LRS2-2Mix/resolve/main/lrs2.tar.gz` |
| 目标文件 | `/root/data/lrs2_2mix/lrs2.tar.gz` |
| 大小 | **16,470,508,590 B**（HF API 元数据 `lrs2.tar.gz` 同值；README 口径 ≈16.47 GB） |
| sha256 | `ebdfdf31335f288cc2e4b1690a0cc2f260e1826a68a020723ad4ce59a033767e` |
| HF 校验 | resolve HEAD 的 `x-linked-etag="ebdfdf31…767e"` = 本地 sha256（LFS 校验通过） |
| 运行方式 | `setsid nohup bash docs/verification/phase1/run_lrs2_download.sh`（后台持久；PID 记录 `T6-download.pid`） |
| 断点续传 | `curl -L --retry 5 --retry-delay 5 -C -`；中断后重跑同一脚本即可续传（`fetch_data.py --dataset lrs2-2mix` 等价） |
| 时间/吞吐 | 2026-10-21T16:31:18Z → 16:54:08Z（≈22m50s）；平均 ≈12 MB/s；日志含起止/rc/sha256/进度快照 |
| 本地占用 | tar 16.47 GB + 解压 `lrs2_rebuild/` 17.39 GB（`du` ≈17 GiB）；建议预留 ≥40 GB（保守 60 GB） |
| 镜像说明 | 任务建议 `HF_ENDPOINT=https://hf-mirror.com`，但本机 TLS 失败；改用 `huggingface.co` + 本机代理（`--proxy` 或 `http(s)_proxy`）。脚本仍支持 `HF_ENDPOINT` 指向可用镜像 |

> 解压由并行验证进程完成（日志 `/root/tmp/lrs2-extract.log`，`tar xzf` rc=0，耗时 ≈3m44s），本报告用解压树与 tar 清单双重核实；
> 数据目录在仓库外，未纳入 git。

## 2. tar 结构核实（完整清单 + 抽样格式验证）

完整证据见 **`docs/verification/phase1/T6-lrs2-structure.txt`**；摘要：

- **清单规模**：`tar tzvf` 249,670 entries（249,621 files + 49 dirs）；成员总大小 **17,390,672,317 B ≈ 17.39 GB**。
- **音频 splits（16k 与 8k 各一份）**：`audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}`，tr 20,000 / cv 5,000 / tt 3,000 组（mixture 共 28,000；wav 共 16k 84,000 + 8k 84,000 + raw 20,392 = 188,392）。
  - 每组 **2.000 s**（16k=32,000 / 8k=16,000 样本）、mono、PCM_16；实测 tr/cv 各 100 + tt 全部 3000。
  - 文件名 `<id1>_<snr1>_<id2>_<snr2>.wav`，`snr2=-snr1`、`snr1∈(0,5]`（28000 条全部可解析；mean|snr|≈2.504）。
  - `mix/scaling.mat` 保存 `scaling_16k(N,2)` 与 `scaling16bit_16k(N,1)`（8k 同）。
- **源音频**：`raw_audio/{train,val,test}` = 19,649 / 476 / 267 = **20,392** 条（16 kHz mono PCM16，2.000 s）。
- **视觉**：`mouths/<id>.npz` = **uint8 (50,96,96)**（2 s@25 fps 灰度，正好 Falcon 2 s 窗口输入）；
  `landmark/<id>.npz` = object array len 50（每帧 1 个、少数 2 个 float32 (68,2)）；`faces/<id>.mp4` = **20,392 个，50 帧，25 fps，224×224**。
  raw/mouth/face/landmark 的 id 集合完全一致（各 20,392）。
- **无 split 泄漏**：tr 使用 18,691 个 raw train ids（19,649 中）；cv 使用全部 476 个 val；tt 使用全部 267 个 test。
- **混音脚本/清单**：`create_mix/create_wav_2speakers.m`（+`activlev.m`/`maxfilt.m`；原 MIT/三菱 MERL 脚本，Apache-2.0）、
  `mix_2_spk_{tr,cv,tt}.scp`（20,000/5,000/3,000 行，`wav1 snr1 wav2 snr2`）、`mix_2_spk_min_*` 名单；
  `crop_mouth_from_video.py` + `utils.py`/`transform.py`/`20words_mean_face.npy` 也在（**本包已带 mouths，无需再裁剪**）。
- **文本辅助**：`train.txt` 20,779 / `retrain.txt` 19,649 / `dev.txt` 492 / `retest.txt` 267 / `redev.txt` 476 / `test.txt` 296 /
  `video_path.txt` 20,469 / `filename_input.csv` 20,392。
- **历史遗留**：`audio_process/mix_2_spk_tr.txt` 是 50,000 行绝对路径旧清单（与 `create_mix/scp` 的 20,000 对不重合）；实际音频产物以
  `create_mix/mix_2_spk_*.scp` + `audio/` 目录为准，勿用旧清单。
- **与 Lead 数字的核对**：Lead 通读时给 20,396 raw / 84,017 wav16k；本报告精确值为 **20,392 raw / 84,000 wav16k**（+84,000 wav8k + 20,392 raw = 188,392 wav 总）。其余（AV 完整、2 s、mouths (50,96,96)、landmark (50,1,68,2) 近似、target 可重建）均确认。

## 3. “如何构建 LRS2-2Mix 训练/评测数据”结论

1. **不需要 LRS2 原始视频**：mouths 已随包发布（50×96×96 uint8），faces/landmarks 也在；旧计划所担心的 LRS2 视频/landmark
   门槛不存在。`crop_mouth_from_video.py` 仅在需要重新裁嘴部时才有用。
2. **音频**：训练/评测直接用 `audio/wav16k/min/<split>/mix/*.wav` 与同名的 `s1`/`s2`（**约定 target=s1、干扰=s2**；
   论文 convention 未声明时需在报告固定并说明；s1/s2 可互换做双目标对照）。
   `mix == s1+s2` 已实测成立（16-bit 量化误差内）⇒ target 参考无需从 raw_audio 重建；若必须重建，按
   `create_wav_2speakers.m`（activlev 归一化 + `10^(snr/20)` + min 长度 + 统一 `0.9/max_amp`）与 `scaling.mat` 可逐样本精确复算。
3. **嘴部转 Falcon `.pt`**：训练 loader（`look2hear/datas/avspeech_dataset_near.py:168-174`）先 `.permute(0,3,1,2)`，
   因此须把 3-D 灰度 npz 补 channel 维：
   `torch.save(torch.from_numpy(np.load(npz)['data']).unsqueeze(-1), "<split>/mouths/<id>.pt")` → uint8 `(50,96,96,1)`。
   转完后 `Grayscale→squeeze(1)` 得到 `(T,88,88)`，与训练 transform 一致。
4. **索引**：按 `DataPreProcess/process_lrs23.py` 生成 `mix.json/target.json`；`near.json` 无法合法生成（本包**无 near 源**）→
   `use_near=false` 或替代负样本（见 `REPRODUCTION_PLAN.md` §5.1-Step4、`DATA_ACCESS_GUIDE.md` §4/§9）。
   **已用 3 样本/split 冒烟验证**（§7）：转目录 + `process_lrs23.py` + `AVSpeechTrainDatasetWithNear(samples=10000)` 返回 `mix(32000,)/src(32000,)/mouth(50,88,88)`。
5. **评测协议**：tt 3,000 组，每组 2.000 s（模型窗口=整段）→ 按 clip 计算 SI-SNRi/SDRi；与论文表对比时声明 target convention、
   16k/8k、是否使用 s1+s2 双目标平均。
6. **空间/许可**：压缩 16.47 GB + 解压 17.39 GB，建议 ≥40 GB（保守 60 GB）；数据受 **LRS2（BBC 来源）与源视频条款**约束，
   HF 卡片 `apache-2.0` 标签不能覆盖源数据条款；faces/mouths/音频为派生数据，**不得再分发**。

## 4. `tools/repro/fetch_data.py` 验收

```bash
# 1) dry-run（验收项；三数据集）
$ python tools/repro/fetch_data.py --dataset all --dry-run --proxy http://127.0.0.1:17897
  lrs2-2mix   : gated=False, lrs2.tar.gz 16.47 GB → OK
  realssa-sim : gated=manual, README+tt/tt-000000.tar 2.17 GB → 授权检查 ❌ GatedRepoError 403 + 申请指引
  realssa-real: gated=manual, README+INFERENCE.md 21.49 KB → 授权检查 ❌ 403 + 申请指引
  rc=0（dry-run 只读；gated 未获批在下载模式下返回 rc=3）
# 2) 真实下载冒烟（小文件）：--pattern README.md --out /tmp/... → Fetching 1 files 100%，文件 694 B ✓
# 3) LRS2 大文件路径：curl -C - 断点续传；若已存在完整文件则 [skip]；实测由 run_lrs2_download.sh 完成整包下载
```
要点：`--dataset {realssa-sim,realssa-real,lrs2-2mix,all}`、`--out`、`--pattern`（可重复）、`--shard`（`tt` / `tt-000000` / `tt/xxx.tar`）、
`--full`、`--dry-run`、`--endpoint`、`--proxy`；支持 `HF_ENDPOINT`；gated 只提示不申请；不绕过 403；默认输出 `/root/data/<dataset>`；
公开大文件断点续传；其余 `snapshot_download(allow_patterns=...)`；未获批 rc=3。
> 说明：LRS2 整包下载在 `fetch_data.py` 完成前已由本任务的后台脚本启动，为避免两个写者并发写同一文件，未再用 fetch_data.py 重复触发；
> 其 resume 分支由代码审阅 + 小文件实测覆盖，重跑 `--dataset lrs2-2mix` 会检测到完整文件并 `[skip]`（可安全再跑）。

## 5. HF 账号 / 授权状态（只读，未提交申请）

- token 文件：`/root/.cache/huggingface/token` 存在（37 B）；`HfApi().whoami()` → **`nanless`（user）**，token 有效。
- `JusperLee/LRS2-2Mix`：**公开**（可下载，已完成）。
- `JusperLee/RealSSA-Sim`：`gated=manual`；**本账号未获批**（非 README 文件 dry-run → `GatedRepoError 403`）。申请字段/流程见指南 §4.1。
- `JusperLee/RealSSA-Real`：`gated=manual`；**未获批**（同上）。申请字段/流程见指南 §5.1。
- `JusperLee/LRS3-2Mix` / `JusperLee/VoxCeleb2-2Mix`：公开，但 HF 仓库**只有 README（无数据文件）**（文件数 2，622/756 B）。
- 未提交任何 access request、未修改账号、未分享 token。

## 6. `docs/DATA_ACCESS_GUIDE.md` 覆盖清单

逐数据集给出：URL、gated 状态/申请表单字段、许可证与禁止事项、体积/格式/通道/采样率、下载命令（`fetch_data.py` + `huggingface_hub`）、
通道与对齐/评测协议、批准前能/不能做什么；并明确写入：
- **RealSSA-Sim 无 near 源波形**（only mix/target/mouths/JSON；对比学习负样本需替代方案）；
- **RealSSA-Real 无 SI-SDR 参考**（只能 DNSMOS/WER/speaker-similarity；GT 是整段 recording）；
- **LRS3-2Mix / VoxCeleb2-2Mix 的 HF 仓库无数据文件**；
- LRS2-2Mix 的正确结构/构建方式（本篇 §2/§3，已按实测更新，修正旧计划）。

> 说明：`docs/DATA_ACCESS_GUIDE.md` 已按 studio-frontend 的 R2 复核修订（242 files/240 shards、3 个 checkbox、
> `README.md:82/85/87` 的 4ch/stereo/mono-loader 说明），并新增 **§3.3「如何转成 Falcon 训练/评测格式」**（mouths→.pt 补 3 通道、
> target=s1、mix/target/near JSON、直接可执行代码 + 本机 3 样本/split 冒烟结果）；旧计划提示已改为引用 `REPRODUCTION_PLAN.md §5.2（已修订）`。
> 数据集卡原文与 tar 清单样本已复制到 `docs/verification/phase1/dataset-cards/` 与 `T6-lrs2-listing-sample.txt`（完整清单 `/root/data/lrs2_2mix/lrs2.tar.list`）。

## 7. 风险 / 未完成 / 建议

| 项 | 状态/建议 |
|---|---|
| RealSSA-Sim/Real 数据 | ⏳ 待数据集所有者人工审批；本任务只读确认未授权。获批后按指南 §4.4/§5.3 操作 |
| LRS3-2Mix / VoxCeleb2-2Mix | ❌ HF 无数据；需联系作者或用原始 LRS3/VoxCeleb2 自建（随机种子未知，只能同分布复现） |
| REPRODUCTION_PLAN 旧结论 | ⚠️ §5.2/§11.2 的 “7,234 wav、无视频/嘴部、纯音频” 已过时，建议 Lead 按本报告修订 |
| `audio_process/mix_2_spk_tr.txt` | ⚠️ 50,000 行旧清单（绝对路径，与 20,000 对新清单不重合）——勿用；以 `create_mix/*.scp` 为准 |
| landmark 格式 | 每帧通常 1 个 (68,2)，少数帧 2 个（多脸），不是严格 `(50,1,68,2)`；设计接口时按 list/首脸处理 |
| target convention | ⚠️ 论文未说明 target=s1 还是 s2；评测/训练需固定并声明（s1/s2 均已提供，可做双目标对照） |
| near 源 | ❌ 官方未发布（LRS2-2Mix 与 RealSSA 同）；训练对比损失只能替代/关闭，需在报告标注 |
| 代码 P0 | 评测/训练入口仍有已知 P0（comet_logger、conf.yml 类名、mono 0 维、指标口径等），数据已就绪不改变这些 blocker |
| 磁盘 | tar 16.47 GB + 解压 17.39 GB；转换副本与 RealSSA 全量（521.5 GB）需另行规划（≥600 GB for RealSSA-Sim 全量） |

## 8. 验收核对（task-14）

- ✅ `tools/repro/fetch_data.py --dry-run` 可跑（含 gated 提示、`--shard`、`--pattern`、`--full`、`--endpoint/--proxy`）。
- ✅ LRS2 下载已启动并完成、记录日志（`docs/verification/phase1/T6-data-fetch.log`）；sha256 与 HF etag 一致。
- ✅ tar 完整结构核实（未解压全量、流式清单 + 解压后抽样格式验证）；构建结论写入指南 §3。
- ✅ `docs/DATA_ACCESS_GUIDE.md`（5 个数据集 + HF 账号/授权 + 通道/对齐/评测 + 合规）。
- ✅ 本机 HF token/授权只读核查，未提交申请。
- ✅ GUIDE 已按 studio-frontend R2 复核修订（242 files/240 shards、3 checkbox、README:82/85/87、删除旧计划过时提示）并新增 §3.3 转换步骤；3 样本/split 转换链路冒烟通过。
- ✅ 未修改源码、未 git commit；task-14 暂不 complete（等 Lead 验收）。

## 9. 下一步（供 Lead 排期）

1. 按指南 §3.3 把 LRS2-2Mix 转成 Falcon 训练/评测目录 + JSON（可立即做；A/V 完整）。
2. 用 LRS2-2Mix tt（3000 个 2 s clip）跑官方权重评测，得到 LRS2-2Mix 列的复现实验（需先修指标/评测 P0）。
3. RealSSA-Sim/Real 提交申请（需用户/作者配合），获批后先下 1 个 `tt` shard 验证转换管线。
4. 训练侧：LRS2-2Mix 可用替代负样本/关闭对比损失做消融；RealSSA-Sim 全量获批前不建议启动正式训练。
