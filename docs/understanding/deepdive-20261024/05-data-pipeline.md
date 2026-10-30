# 05 深挖 · 数据管线与数据资产（LRS2-2Mix / RealSSA-Sim / RealSSA-Real）

> 小组：Falcon 项目"理解 + 模型复现规划"深挖 · 成员 `data-pipeline` · 任务 `task-5`
> 仓库：`/root/code/github_repos/Falcon-fork`，分支 `repro/phase2`，HEAD `2beec38`（2026-10-24）
> 环境：`/root/miniforge3/envs/falcon/bin/python`（Python 3.11.17 / torch 2.8.0+cu128）
> 证据标记：`[代码]` 源码直读；`[验证]` 本轮实际命令/抽样结果；`[推断]` 由代码+数据合理推断；`[文档]` 只来自已有报告/数据卡
> 约束遵守：只读分析；只写本文件；未改源码、未改他人文档、未 git 写操作、未 pip 安装、未联网。

---

## 0 一句话结论

**LRS2-2Mix 这一侧的数据链路已基本"从零可重建"（原始包 + 28000 条单目标转换 + 40000/10000/6000 论文配方双向数据，抽样校验全过）；RealSSA 这一侧只有 test 可评测、训练不成立：本地只有 RealSSA-Sim `tt`（9169 条/19 shard）的转换产物，`tr/cv` 只存在于 TOS 归档（521.5 GB）且未转换，官方也未发布 near-speaking 波形（K=0），而 LRS2 单目标数据集磁盘上的 `near.json` 又被一份未入库脚本改成了 "s2 作负样本"（K=1），与 P2-C/QA 报告的 fallback(K=0) 说法不一致——因此"从零训练 RealSSA-Sim"与"论文 14.1/14.7 严格复现"目前都不是数据就绪状态。**

---

## 1 边界与关键路径

### 1.1 本文件覆盖的东西

| 类别 | 路径 | 说明 |
|---|---|---|
| 数据集类 | `look2hear/datas/avspeech_dataset.py`（334 行） | `AVSpeechTrainDataset` / `AVSpeechEvalDataset`（评测与训练期验证实际用的类）/ 旧 `AVSpeechDataModule`（死代码） |
| 数据集类 | `look2hear/datas/avspeech_dataset_near.py`（337 行） | `AVSpeechTrainDatasetWithNear`、`collate_with_near`、`AVSpeechDataModuleWithNear`（唯一训练入口） |
| 变换 | `look2hear/datas/transform.py`（188 行） | 嘴部 numpy 管线：Normalize/CenterCrop/RandomCrop/HorizontalFlip |
| 调试副本 | `look2hear/datas/check_avspeech_dataset.py`（349 行） | `avspeech_dataset.py` 的旧拷贝 + `__main__` 导出样本，写死 `${DATA_ROOT}`（:319）不可直接运行 |
| 建索引 | `DataPreProcess/process_lrs23.py`（121 行）、`DataPreProcess/generate_near_json.py`（105 行） | 生成 `mix/target/near.json`；`process_lrs23.py:96-105` 已内嵌 near 生成 |
| 转换器 | `tools/repro/lrs2_2mix_to_falcon.py`（644 行） | LRS2-2Mix `lrs2_rebuild` → Falcon folder 格式 + JSON |
| 转换器 | `tools/repro/realssa_to_falcon.py`（548 行） | RealSSA-Sim WebDataset TAR → Falcon folder 格式 + `_meta.json` + JSON |
| 合成数据 | `tools/repro/make_synthetic_data.py`（355 行） | 冒烟用合成数据（mono、`mouths [T,96,96,3]`、可带 near） |
| 获取 | `tools/repro/fetch_data.py`（328 行） | LRS2 断点续传 / RealSSA gated 提示 / `--shard/--pattern/--full/--dry-run` |
| 视觉骨干 | `tools/repro/extract_video_backbone.py`（180 行） | 从 `falcon.pt` 导出 `frcnn_128_512.backbone.pth.tar`（T5 已验证） |
| Phase2 数据/评测脚本 | `tools/repro/phase2/{build_lrs2_paper_data.py,download_realssa_shared.py,eval_realssa_real.py,eval_dnsmos.py,realssa_sim_tt_pipeline.sh,...}` | 论文配方构建、RealSSA 下载归档、DNSMOS/RealSSA-Real 评测 |
| 文档 | `docs/DATA_ACCESS_GUIDE.md`、`docs/verification/phase1/dataset-cards/*.md`、`docs/verification/phase1/{T4,T5,T6}-report.md`、`docs/verification/phase2/{P2C,P2E}-report.md`、`docs/REPRODUCTION_PHASE2_REPORT.md` | 获取/许可/历史验证 |

### 1.2 磁盘 → 张量主链路（当前仓库真实路径）

```
【原始资产】
LRS2-2Mix:  lrs2_rebuild/audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}/*.wav  (2.000s, 16k mono)
            lrs2_rebuild/mouths/<speaker_utt>.npz  uint8 (50,96,96)
            lrs2_rebuild/raw_audio/{train,val,test}/*.wav  (2.000s 单说话人干净片段)
RealSSA-Sim: {tr,cv,tt}/*.tar  WebDataset：<id>.mix.wav(4ch) / .target.wav(4ch) / .json / .mouths.pt[T,96,96,3] uint8
        │
        │  tools/repro/lrs2_2mix_to_falcon.py      （单目标：target=s1，near=fallback 或 raw_audio）
        │  tools/repro/phase2/build_lrs2_paper_data.py（双向：s1/s2 各一条，near=另一说话人）
        │  tools/repro/realssa_to_falcon.py        （4ch→mono，align frames=min(...)，meta→near.json fallback）
        │  DataPreProcess/{process_lrs23.py,generate_near_json.py}（亦可单独重建 JSON）
        ▼
【Falcon folder 格式】<out>/<split>/{mix,target,mouths} + mix.json/target.json/near.json
        │
        │  look2hear/datas/avspeech_dataset_near.py:46-223（训练）
        │  look2hear/datas/avspeech_dataset.py:130-219（评测/验证）
        ▼
【样本】train: (mix[32000] f32, src[32000] f32, mouth[50,88,88] f64, basename, near list)   near.py:152-223
        eval : (mix[T] f32, src[T] f32, mouth[F,88,88] f64, basename, n_near)              avspeech_dataset.py:181-219
        │
        ▼
【batch】collate_with_near: mix[B,32000] f32 / src[B,32000] f32 / mouth[B,50,88,88] f64 / names / K 可变 near  near.py:23-43
```

### 1.3 关键约定（本轮实测确认）

- 采样率 16 kHz、25 fps、**640 samples/frame**、训练窗 2.0 s = 32000 samples = 50 帧。`[代码]` `avspeech_dataset_near.py:77-78,168`；`[验证]` LRS2 全部 32000、RealSSA 全部 `n % 640 == 0`。
- 训练：每条样本在 `__getitem__` 内重新 `random.randint` 选样本、随机 `rand_start ∈ [0, n-seg_len]`；`__len__` 返回 `samples`（= `train_samples`）而非数据量 → "伪 epoch + 有放回采样"。`[代码]` `avspeech_dataset_near.py:147-157`。
- 评测：整段读 `start=0, stop=None`，不过滤短样本，`__len__ = len(mix.json)`；返回 5 元组，第 5 项 `n_near` 来自 `mix/<stem>_meta.json` 的 `components[].type=="near"` 计数。`[代码]` `avspeech_dataset.py:161-219`。
- 视觉：`mouths/<stem>.pt` 按 `[T,96,96,3]` uint8 存 → `permute(0,3,1,2)` → `torchvision.transforms.Grayscale(1)` → `squeeze(1)` → numpy → `transform.py` 管线裁到 88×88。`[代码]` `avspeech_dataset_near.py:169-174`、`transform.py:147-169`。
- 音频通道：训练集/评测集均是"多通道才 `mean(-1)`、mono 保持 1-D"（F5 修复后）。`[代码]` `avspeech_dataset_near.py:176-177`、`avspeech_dataset.py:114-120,198-201`。

---

## 2 数据资产表（本轮实测）

### 2.1 本地 `/root/data`（rc 路径为实测）

| 资产 | 路径 | `du -sh` | 条数/规模 | 内容与状态 | 许可/约束 |
|---|---|---|---|---|---|
| LRS2-2Mix 原始包 | `/root/data/lrs2_2mix` | **33G** | `lrs2.tar.gz` 16G（16,470,508,590 B，sha256 `ebdfdf31…`）；`lrs2_rebuild/` 17G | 完整 2 s AV 包：wav16k+wav8k、`mouths` 20392 npz、`faces` 20392 mp4、`raw_audio` 19649/476/267、混音脚本/清单 | HF 标签 apache-2.0，但内容来自 BBC LRS2，需遵守 LRS2 研究条款；派生 mouths/faces 不可再分发 `[文档]` GUIDE:94-98 |
| LRS2 单目标转换 | `/root/data/falcon_lrs2_2mix` | **24G**（23.36 GiB） | tr 20000 / cv 5000 / tt 3000 | `{mix,target,mouths}` + 三份 JSON；target=s1；mouths 14773/476/267 个 `.pt`；**near.json 实为 s2 负样本（K=1，非文档写的 fallback）** | 同上；数据由本机转换，与 LRS2 同许可 |
| LRS2 论文配方 | `/root/data/falcon_lrs2_2mix_paper` | **26G** | tr **40000** / cv **10000** / tt **6000**（每 mixture 双向两条） | 只有 `mouths/`（18691/476/267）+ 三份 JSON；音频引用 `lrs2_rebuild` 原路径；near=另一说话人（K=1） | 同上 |
| RealSSA-Sim 原始（仅 test） | `/root/data/RealSSA-Sim` | **38G** | `tt/` 19 个 tar（tt-000000…tt-000018） | 仅 test split；WebDataset 4 成员/样本、shard ≤500 样本；README 在根 | CC BY-NC-ND 4.0（TED 等），仅非商业学术、禁再分发 `[文档]` card:3,26-27 |
| RealSSA-Sim 转换（仅 test） | `/root/data/falcon_realssa_sim` | **30G** | `tt/` **9169** 条 | mix/target 16k mono（mean 通道）+ mouths `[T,96,96,3]` + `mix/<id>_meta.json` 9169 个 + 三份 JSON；near 全 fallback(K=0)；**无 tr/cv** | 同上；派生转换物同样受 non-commercial/禁再分发约束 |
| RealSSA-Real 原始 | `/root/data/RealSSA-Real` | **12G** | 423 文件；inference 54 段 / references 10 / faces·mouth·landmarks 各 54 | 真实场景 stereo 16k（`audio/inference/*.wav`）；官方 loader `scripts/load_inference.py`；54 段共 83.99 min | **无通用数据集 license**；参与者个人数据，仅研究用途、禁再识别/转发 `[文档]` card:80,317 |
| TOS 归档 · Sim 全量 | `/aispeech-cfs/voiceprint/data/multimodal/realssa_sim` | 归档 manifest **521,532,645,652 B**（≈521.5 GB） | 243 文件 = README + .gitattributes + `_MANIFEST.json` + **240 shard**（tr 202 / cv 19 / tt 19）；实测 `tt/` 40.14 GB | **唯一保存全部 tr/cv/tt 的副本** | CC BY-NC-ND 4.0 + 各噪声源条款 |
| TOS 归档 · Real 全量 | `/aispeech-cfs/voiceprint/data/multimodal/realssa_real` | 12.84 GB（423 文件） | 与本地 `/root/data/RealSSA-Real` 同构 | 全量 | 同上 |
| 评测产物 | `/root/data/falcon_eval` | 未计（小） | `realssa_real_mean_full{,2,3}`、`_smoke_*` 等 | RealSSA-Real DNSMOS 逐段 CSV/summary/metadata；`full`/`full2` 为失败运行（0/54 ok），`full3` 成功（54/54） | 派生分离结果属参与者数据，按 card 用完删除 |
| 权重缓存 | `/root/.cache/falcon_weights` | 44.8 MB + `falcon.pt` 47 MB | `frcnn_128_512.backbone.pth.tar` 等 | 非数据资产，T5 导出 | Apache-2.0（权重） |

### 2.2 归档/删除状态（重点问题④的一半）

- 组共享 staging 目录 `/root/group-shared/voiceprint/data/multimodal/{realssa_sim,realssa_real}` **已不存在**；`_download/_archive_done.json`（2026-10-23T17:31:18Z）两条 `ok:true` → 报告"源目录已删"指的就是这两个 staging 目录。`[验证]`
- 但 `/root/data/RealSSA-Sim`（38G）与 `/root/data/RealSSA-Real`（12G）**仍在**，与 TOS 重复；报告 §5.4 的"源目录已删除"若被读成"本地所有旧副本都删了"是**不准确**的。`[验证]` `docs/REPRODUCTION_PHASE2_REPORT.md:166-169`
- 也就是说：**RealSSA-Sim 的 tr/cv 只存在 TOS**；tt 有两份（TOS + 本地原 tar + 本地转换产物）。

### 2.3 数据许可/合规摘要（报告必须声明）

| 数据 | 许可条款 | 对复现的约束 |
|---|---|---|
| LRS2-2Mix | HF 卡片 `apache-2.0`，但源自 BBC LRS2（研究用途，需向 Oxford VGG 申请）；faces/mouths/音频是派生数据 | 论文数字可复现；**不得再分发**派生数据；商用需另核 LRS2 条款 `[文档]` GUIDE:98,371-373 |
| RealSSA-Sim | **CC BY-NC-ND 4.0**（TED 内容）+ 各噪声源（如 BBC Sound Effects RemArc 非商业） | 仅非商业学术；禁止转发/再分发（含 mouth crops、分离结果）；转换产物同约束 `[文档]` card:3,26-27,74,89 |
| RealSSA-Real | **无通用 license**；参与者授权 + card 的 permitted-use | 仅研究；禁再识别/公开参与者媒体；**不能算 SI-SDR**（无时间对齐 clean target）`[文档]` card:80,317；GUIDE:273-274 |
| 模型权重 | Apache-2.0 | 权重公开 ≠ 数据授权 `[文档]` card:68 |

---

## 3 关键实测数字（命令 + 输出）

> 全部命令在仓库根执行，前缀 `PYTHONPATH=$PWD`，解释器 `/root/miniforge3/envs/falcon/bin/python`；只读，单条 <5 min。

### 3.1 目录存在性/大小/条数

```bash
du -sh /root/data/{falcon_lrs2_2mix,falcon_lrs2_2mix_paper,falcon_realssa_sim,RealSSA-Sim,RealSSA-Real,lrs2_2mix}
# 24G  falcon_lrs2_2mix
# 26G  falcon_lrs2_2mix_paper
# 30G  falcon_realssa_sim
# 38G  RealSSA-Sim
# 12G  RealSSA-Real
# 33G  lrs2_2mix
df -h / /root
# overlay         20G  6.2G   14G  32% /
# fs_vepfs…      164T  151T   14T  92% /root
```

- 六个目录的大小与任务描述一致。`[验证]`
- `lrs2_2mix` 33G = tar 16G + 解压 17G + 少量 smoke（`falcon_smoke` 14M、`index_smoke` 6.5K、`extract` 5.4M、`lrs2.tar.list` 33M）。`[验证]`
- TOS 侧只做 `ls/count`（未 `du` 521 GB）：`realssa_sim` 243 文件、`tr/cv/tt` = 202/19/19、`tt` 目录 40.14 GB；`realssa_real` 423 文件。`[验证]`

### 3.2 JSON 条数与等长（三套数据全部逐 split 检查）

| 数据集 | split | mix.json | target.json | near.json | 等长 |
|---|---|---|---|---|---|
| `falcon_lrs2_2mix` | tr/cv/tt | 20000/5000/3000 | 20000/5000/3000 | 20000/5000/3000 | ✅ |
| `falcon_lrs2_2mix_paper` | tr/cv/tt | **40000/10000/6000** | 40000/10000/6000 | 40000/10000/6000 | ✅ |
| `falcon_realssa_sim` | tt | **9169** | 9169 | 9169 | ✅ |

`[验证]` 直接 `json.load` 计数；`len(mix)==len(target)==len(near)` 对 9 个 split 全部成立。

### 3.3 抽样校验：`mix == s1 + s2`、`target == s1`

每 split 随机 5 条（`random.Random(0)`）与 LRS2 原始 `audio/wav16k/min/<split>/{s1,s2}` 对比：

```
[tr:6116148535082467916_] mix(16000Hz,1ch,32000) tgt(ch1,32000) mouth(50,96,96,3)/uint8
   mix-(s1+s2)=3.05e-05  tgt-s1=0.00e+00  ratio=640.0  near_ok=True
[cv …] / [tt …] 共 15 条全部同值
```

- `max|mix-(s1+s2)| = 3.05e-05` = PCM16 一个 LSB（1/32768），即**逐位成立**（16-bit 量化内）。`[验证]`
- `max|target-s1| = 0.0`（转换器按字节复制 s1）→ **target=s1 约定成立**；路径里不含 `/s1/` 只是复制到了 `target/` 目录。`[验证]`
- near 条目路径全部存在、可读，`n_samples=32000`。`[验证]`

### 3.4 mouths 形状 / dtype / 640 对齐

```python
torch.load(mouths_pt, weights_only=True)  # (50, 96, 96, 3), torch.uint8, 3 通道逐元素相同, 值域 0..167（该样本）
transforms.Grayscale(1)(t.permute(0,3,1,2)).squeeze(1)  # (50,96,96)
get_preprocessing_pipelines()["val"](...)               # (50,88,88) float64
val == (g[:,4:92,4:92]/255 - 0.421)/0.165               # True
```

- LRS2 全部 50 帧 = 32000/640；论文配方/RealSSA 抽样 10+20 条全部 `mouth_frames == n_samples//640`。`[验证]`
- `.pt` 为 **uint8** 时，`Normalize(0.0,255.0)` 在 numpy 中提升为 float64 → batch `mouth` 是 **float64**（下游 `type_as`/`.float()` 兜底，功能无碍）。`[验证]` 与旧文档 §5.4 条件化结论一致。

### 3.5 near.json 的真实状态（本轮最重要的"文档-磁盘漂移"）

| 数据 | 磁盘上的 near 语义 | 数量 | 证据 |
|---|---|---|---|
| `falcon_lrs2_2mix` | **每条 = 该 mixture 的 s2 波形**（K=1） | 20000/5000/3000，`near_fallback=0` | `[验证]` 逐条路径检查 + `P2D-near-train.log:17-18` |
| `falcon_lrs2_2mix_paper` | **每条 = 另一说话人**（target=s1 时 near=s2，反之 s1，K=1） | 40000/10000/6000 | `[验证]` 抽样 + `[代码]` `build_lrs2_paper_data.py:103-106` |
| `falcon_realssa_sim` | **fallback：near = mix 自身**（K=0，Dataset 哨兵跳过） | 9169/9169 fallback | `[验证]` `near[i][0][0]==mix[i][0]` 全命中；`P2E-realssa-sim-convert.log:191` |

但：`conversion_stats.json` 明确写 `near_mode: "fallback"`（:5），`P2C-convert.log:112` 写 "near=fallback identical to mix sentinel for all 3 splits"，P2-C/QA 报告也按 fallback 叙述。而 `falcon_lrs2_2mix/*/near.json` 的 **mtime = 2026-10-22 16:37:54**（转换/统计是 2026-10-21 19:22–19:24），内容已变为 s2 路径。仓库中能生成该形态的脚本只有 `build_lrs2_paper_data.py`（输出到 `_paper` 目录），`lrs2_2mix_to_falcon.py` 的 fallback/raw_audio 两种模式都**不会**产出 "near=s2"——说明它是**未入库的一次性命令**改写的。`[验证]`+`[推断]`

### 3.6 真实 Dataset 实际跑批（训练/评测行为直证）

```python
AVSpeechTrainDatasetWithNear("/root/data/falcon_lrs2_2mix_paper/tr", segment=2.0, samples=8, use_near=True)
# Loaded near.json with 40000 entries; 40000/40000 samples have near-speaker entries (K=1)
# batch: mix (4,32000) float32 | src (4,32000) | mouth (4,50,88,88) float64 | K=[1,1,1,1]
# near inner: (32000,) float32；mouth range [-2.55, 3.25]

AVSpeechTrainDatasetWithNear("/root/data/falcon_lrs2_2mix/tr", …)
# Loaded near.json with 20000 entries; K=[1,1,1,1]

AVSpeechEvalDataset("/root/data/falcon_realssa_sim/tt")
# len=9169；[0] mix(92800,) mouth(145,88,88) ratio=640.0 n_near=2
# [5000] mix(59520,) mouth(93,88,88) n_near=2

AVSpeechEvalDataset("/root/data/falcon_lrs2_2mix/tt")
# [0] mix(32000,) mouth(50,88,88) n_near=0（无 _meta.json）
```

要点：
- **训练随机 2 s 切片在 LRS2 上是"退化"的**：LRS2 每条恰好 32000 samples（转换统计 `sample_lengths={32000:N}`），所以 `rand_start` 恒为 0；"随机 2 s 切片"只剩**空间**增广（RandomCrop 88 + HorizontalFlip 0.5）。真正随机时间切片只在 >2 s 的数据（如 RealSSA）上生效。`[验证]`+`[代码]` `avspeech_dataset_near.py:154-157`
- **评测整段**已用 RealSSA-Sim（3.72–5.80 s 等变长）实证：mix 与 mouth 都整段返回。`[验证]`
- RealSSA-Sim 的 `_meta.json` 里 `components[].type=="near"` 仍存在（抽样 `[4,4,1,1,1]` 个），因此 **eval 的 `n_near` 会 >0**，但训练 near 音频为 0 条 —— "meta 说有 near / 实际没有 near 波形"的语义不一致（见 §4-D2）。`[验证]`

### 3.7 RealSSA-Sim 转换保真度（tar 原样 ↔ falcon 转换产物）

抽 `tt-000000.tar` 的第一个样本 `00001_00_10182`：

```
source: mix (92800,4) @16k | target (92800,4) | mouths (145,96,96,3) uint8
json:   components 含 near(sir≈9.98/14.63 dB)、far…，path 全为 /gpfs-… provenance
converted: mix 92800 样本 mono；mouth 145 帧
mean(4ch) vs converted  max|Δ| = 2.29e-05  （PCM16 量化内）
ch0       vs converted  max|Δ| = 0.1645     （≠ mean，证明转换确实用了 --channel mean）
target mean vs converted max|Δ| = 2.29e-05
mouths torch.equal(源前 135 帧, 转换) = True
frames = min(145, 92800//640, 92800//640) = 145，长度 92800 = 145*640
```

→ **RealSSA-Sim 转出的 falcon 格式完整、通道策略与对齐均正确**（channel=mean、align=True，见 `P2E-realssa-sim-convert.log:1`）。`[验证]`

### 3.8 论文配方 40000/10000/6000 双向构建核对

| 检查 | 结果 |
|---|---|
| mix.json 条数 | tr 40000 / cv 10000 / tt 6000 ✅ |
| 唯一 mix 路径数 | tr 20000（每条 mixture 复用两次）✅ |
| 偶数下标 target | 全为 `.../s1/<stem>.wav`（tr 20000/20000）✅ |
| 奇数下标 target | 全为 `.../s2/<stem>.wav`（tr 20000/20000）✅ |
| 配对 mouth 不同（s1 mouth vs s2 mouth） | 20000/20000 对均不同 ✅ |
| near = 另一说话人 | 抽样逐条成立 ✅ |
| mouths 数 | 18691 / 476 / 267 ✅（与 `conversion_stats.json` 一致） |

结论：`build_lrs2_paper_data.py` 的产物与 `docs/REPRODUCTION_PHASE2_REPORT.md:149-153`、`docs/PHASE2_VOLCANO_LRS2_TRAINING.md:24-46` 的描述**完全一致**。`[验证]`

### 3.9 回归测试与旧验证脚本

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest \
  tools/repro/tests/{test_lrs2_convert,test_realssa_convert,test_synthetic_data,test_phase2_data}.py -q
# 22 passed in 45.83s
```

`[验证]` 说明转换器/合成数据/数据契约测试在当前环境仍全绿。

### 3.10 其他实测数字

- RealSSA-Sim tt 长度：min 16000（1.0 s）、max 120320（7.52 s）；**1047/9169（11.4%）短于 32000**；`n % 640 == 0` 全成立。若用于训练，`AVSpeechTrainDatasetWithNear` 会把 <2 s 的样本静默丢弃（`avspeech_dataset_near.py:111-122`），且 RealSSA tr 的长度分布未知。`[验证]`+`[代码]`
- RealSSA-Real 成功评测（`realssa_real_mean_full3/summary.json`）：54/54 段 ok，SIG 3.2209 / BAK 3.6921 / OVRL **2.7901** / P808 **3.3285**（channel=mean、align_tail、窗口化推理）；`full`/`full2` 为 0/54 的失败运行残留。`[验证]`+`[文档]` P2E/REPORT:98-109
- near 负样本确实被训练消费：`P2D-near-train.log:15-18` "Loaded near.json with 20000 entries / 20000/20000 samples have near-speaker entries / Near file access OK: …/tr/s2/…wav"，且该次微调带来 test SI-SNRi 8.618→**8.858**、SDRi 9.042→**9.270**（`REPRODUCTION_PHASE2_REPORT.md:62-70`，[文档]）。`[验证]`（日志存在）+`[文档]`

---

## 4 缺陷与风险（按影响排序）

### D1（P0，可复现性）单目标 `falcon_lrs2_2mix` 的 near.json 被未入库脚本改写，文档/磁盘/统计三方不一致

- 事实：`conversion_stats.json` / `P2C-convert.log:112` / P2-C/QA 报告 = fallback(K=0)；磁盘 `{tr,cv,tt}/near.json`（mtime 2026-10-22T16:37:54）= s2 负样本(K=1)；仓库内**没有**能生成该形态的 committed 脚本（`lrs2_2mix_to_falcon.py` 只支持 fallback/raw_audio；`build_lrs2_paper_data.py` 只写 `_paper`）。`[验证]`+`[推断]`
- 后果：任何人按仓库重跑 P2-C 只会得到 K=0；而 P2-D 的 contrastive 微调结果（+0.24 dB）依赖这份"来源不明"的 near.json —— **该实验不可由仓库一键复现**。
- 建议：新增 `tools/repro/phase2/make_near_from_s2.py`（或给 `lrs2_2mix_to_falcon.py` 加 `--near-mode other-speaker`），并在 P2-C/QA 报告加勘误。

### D2（P0，数据缺口）RealSSA 无 near 波形 → NASS/MDCL 对比损失在 RealSSA 上不可复现

- 官方卡明确：shards 只发 mix/target，`components[].path` 是生成时 provenance，**"indexing script's fallback to the mixture is not an equivalent reproduction"**。`[文档]` `docs/verification/phase1/dataset-cards/RealSSA-Sim-README.md:213`
- 本机实测：`falcon_realssa_sim/tt/near.json` 9169/9169 fallback（K=0）；转换器 `realssa_to_falcon.py:36-45` 已如实写该限制；`--near-dir` 可接外部负样本（本机无）。
- 语义副作用：`_meta.json` 中仍保留 `type=="near"` 组件 → `AVSpeechEvalDataset` 的 `n_near=2`，而训练 K=0；如果直接拿它做验证分桶，会得到"有 near 但训练从不使用"的误导性日志。`[验证]`

### D3（P0，训练未就绪）"从零训练 RealSSA-Sim"不成立

- 本地只有 `tt` 转换产物；`tr`（100,826）/`cv`（9,133）**没有任何本地副本或转换产物**，只在 TOS 归档（521.5 GB，240 shard）。`[验证]`
- 还需先决定：① 是否用 `--channel mean`（论文未声明，卡片只给 inference 示例 ch0 并声明"不是协议断言" `[文档]` card:185,211）；② near 负样本从哪来（D2）；③ RealSSA tr 里 <2 s 样本的丢弃率未知；④ 转换产物体积（见 D7）。
- 因此当前 RealSSA 只能做 **test 评测**（官方权重已复现 7.774/8.479，`REPRODUCTION_PHASE2_REPORT.md:120-131` [文档]），不能做"从零训练"。

### D4（P1，协议）论文 14.1/14.7 与现有 LRS2 数据/训练口径的差异

1. **near 负样本不是 near-field 泄漏源**：论文的 speaker-contrastive 负样本是近场干扰人（高 SIR 泄漏）；LRS2 无 near 源，本复现用 s2/s1（远场、干净、且本来就在 mix 里）替代 → 只能算 "non-official negatives"，指标不可与论文严格对齐。`[代码]` `lrs2_2mix_to_falcon.py:29-41`、`build_lrs2_paper_data.py:6-8`；`[文档]` `docs/PHASE2_VOLCANO_LRS2_TRAINING.md:28-29`
2. **target 约定**：论文未写 target=s1/s2；本仓库评测用 `falcon_lrs2_2mix/tt`（3000，target=s1，与 benchmark 视频说话人一致），而论文配方训练集/验证集用 `_paper`（tr 40000/cv 10000/**tt 6000**）。若把 `_paper/tt` 当 benchmark，会把每条 mixture 计两次（双向），与论文"3000 test mixtures"口径不同。`[验证]`+`[文档]` `docs/PHASE2_VOLCANO_LRS2_TRAINING.md:107-108`
3. **训练规模**：论文 10×A800、全局 bs40、1000 epoch；本机只有 2×A800（且 GPU0 被外部占用），已跑的是单卡 bs4、130 epoch → 8.858/9.270，差 ~5 dB。`[文档]` `REPRODUCTION_PHASE2_REPORT.md:55-70`
4. **s1 恒更响的捷径**：LRS2-2Mix 名字里 `snr1>0`（T6 实测 snr2=-snr1），只训 target=s1 会鼓励"选更响"，双向 target 正是为消除该捷径（`PHASE2_VOLCANO_LRS2_TRAINING.md:24-26`）。`[文档]`+`[验证]`（stem 全部可解析、snr 符号）

### D5（P1，行为）LRS2 上"随机 2 s 切片"退化为整段

所有 LRS2 wav 恰为 32000 samples，`rand_start = random.randint(0, 0) = 0`；因此训练与评测的音频窗口实际一致，数据增广只有空间裁剪/翻转。对 RealSSA 这类变长数据才真正随机切片。任何"训练用了随机时间窗"的表述在 LRS2 上都是名义上的。`[验证]`+`[代码]` `avspeech_dataset_near.py:154-157`

### D6（P1，文档漂移）`docs/DATA_ACCESS_GUIDE.md` 权限状态已过期

- GUIDE 仍写 RealSSA "申请已提交待批 / 数据文件仍 403"（:18、:34、:35、:218、:273），但实际 2026-10-22 已获批、已下载、已归档 TOS（`_archive_done.json`、`REPRODUCTION_PHASE2_REPORT.md:95-131,162-169`）。`[验证]`+`[文档]`
- P2E 报告自己也指出 GUIDE 第 18/39 行状态行不一致（`P2E-report.md:215`），但仍未修。属他人写 scope，本文件只记录。

### D7（P1，磁盘/体积）全量 RealSSA-Sim 转换的磁盘压力

- 现状：`/` 20G 只剩 14G；`/root` 164T 用 92%、剩 14T。RealSSA-Sim 原始 521.5 GB 只在 TOS。
- 体积外推：仅 tt（19 shard / 38G 原始）转换后 30G，其中绝大多数是 mouths（uint8 `[T,96,96,3]`，每帧 27 KB）。按卡片的 tr 100,826 / cv 9,133 推算，全量转换产物约 **0.3–0.4 TB**，加原始 0.52 TB ≈ **0.8–0.9 TB**；`/root` 放得下但需要规划，`/` 完全放不下（`TMPDIR=/root/tmp` 的约定必须保持）。`[验证]`+`[推断]`

### D8（P2）RealSSA-Real 评测的失败残留

`/root/data/falcon_eval/realssa_real_mean_full{,2}/summary.json` 是 54 段全 error 的失败运行（loader 报错），只有 `full3` 是 54/54 成功。若复现者/报告直接 glob 目录，容易取到错误结果。`[验证]`

### D9（P2）旧调试副本仍不可运行

`look2hear/datas/check_avspeech_dataset.py` 已同步 F5 守卫（:122-123、:203-204），但 `__main__` 仍写死 `${DATA_ROOT}/av-tiger-lrs3/...`（:319），直接运行必 FileNotFoundError；无 CI 引用。`[代码]`

### D10（P2）近场/负样本契约的工程细节（仍未修）

- `near.json` 只存 `[path, n_samples]`，**不含采样率**；Dataset 按 16 kHz 解释、不重采样、不校验（`generate_near_json.py:54-60`；`avspeech_dataset_near.py:193-203`）。LRS2 s2 恰为 16 kHz 故无碍；若有 48 kHz near 源则时间轴错位。`[代码]`
- `n_samples < seg_len` 的 near 条目被静默跳过；读取偏移被 `min(rand_start, n-seg_len)` 钳位 → 短 near 文件与 mix 不同步（注释已承认）。`[代码]` `avspeech_dataset_near.py:186-199`
- `target.json` 第 3 字段 `n_samples` 至今未被任何 Dataset 读取（`target.json` 契约存在但不校验长度）。`[代码]`

### D11（P2）训练类 `__len__`/随机采样 + `train_samples` 配置

`configs/LRS2_2Mix.yml:152`、`LRS2_2Mix_near_s2.yml:151`、`LRS2_2Mix_Paper_8gpu.yml:151` 已显式给 `train_samples: 10000`（旧文档 P1-3 的"YAML 缺键"在这些配置上已修正）；但 `configs/Falcon.yml` 的 `data_config` 仍无 `train_samples`（grep 无命中），默认 10000。`[验证]`

### D12（P2）论文配方数据不自包含

`falcon_lrs2_2mix_paper` 不复制 wav（`conversion_stats.json` `copy_wavs:false`），JSON 直接引用 `/root/data/lrs2_2mix/lrs2_rebuild/...`。删除/移动 LRS2 原始解压目录会使 26G 的 paper 数据全部失效；迁移到云端（火山引擎）必须同时带 `lrs2_rebuild` 或重跑 `--copy-wavs`。`[代码]`+`[验证]`

---

## 5 与旧文档 `05-data-pipeline.md`（912 行，R3）逐条对照

> 旧文档行号以 `main@7576854` 为准；本表只列"结论级"差异。

| # | 旧文档说法（位置） | 本轮证据 | 判定 |
|---|---|---|---|
| 1 | "没有真实 LRS2/LRS3/RealSSA 数据，无 meta 样例、无训练 .pt 资产"（§8.5、§6 A18 条件列） | `/root/data` 下 LRS2 全量、RealSSA-Sim tt 转换产物、RealSSA-Real 全量均在；真实 `.pt` = `[T,96,96,3]` uint8（LRS2 由 npz 补通道、RealSSA 原生） | **推翻（条件已满足）** |
| 2 | **P0**：`AVSpeechEvalDataset` mono `mean(-1)` 无守卫 → 0 维 → `torch.stft` 崩；评测与训练期验证必崩（§4.7 P0、§8.6） | 已在 `ac91800` 修复：`avspeech_dataset.py:114-120`（train/legacy）、`:198-201`（eval）加 `ndim` 守卫；`check_avspeech_dataset.py:122-123,203-204` 同步；T4 报告 10 passed | **修正（已修，旧 P0 关闭）** |
| 3 | near 源缺口：LRS2/RealSSA 都无 near 波形，fallback→K=0，NASS 负项=0（§1.4、P1-6、A65） | RealSSA 仍成立（9169/9169 fallback，官方卡 :213 明说）；**但 `falcon_lrs2_2mix` 磁盘 near.json 已被改成 s2 负样本、K=1**，且来源脚本未入库 | **部分确认；LRS2 现状需修正（新增 D1）** |
| 4 | `_meta.json` 是 NASS 对比损失的必要输入、仓库不生成（§1.4、§3.2(d)、Q3/Q6） | 对 **RealSSA** 正确（`realssa_to_falcon.py:238-287` 生成 meta，但 near 路径是 provenance→fallback）；对 **LRS2** 不成立：两个转换器直接写 near.json，不依赖 meta（LRS2 转换产物里**没有** `_meta.json`，实测 eval `n_near=0`） | **修正（区分数据源）** |
| 5 | `mouths` 格式未知，`[T,96,96,C]`，C=3 最可信但 1ch 未验证（Q2、A18） | 真实数据全部 `[T,96,96,3]` uint8，3 通道逐元素相同；`Grayscale` 后 `(T,96,96)`；实测 val = `[4:92,4:92]` 中心裁剪 | **确认并解答（C=3，uint8）** |
| 6 | 2 s = 32000 = 50 帧 × 640（§3.7、A16） | LRS2 28000 条全 32000；RealSSA 全部 `n%640==0`，`mouth_frames == n//640`；tar↔转换 mouth `torch.equal` | **确认（跨两套数据）** |
| 7 | 训练随机 2 s 片、评测整段（§3.6、§4.6、A15/A33） | 代码行为确认；**但 LRS2 全 32000 → `rand_start≡0`，时间随机切片退化**（新增 D5）；RealSSA 变长数据实测整段评测 | **确认代码；修正语义（LRS2 上退化）** |
| 8 | 通道策略：读取时 `mean(-1)` 归约，隐含多通道；Q1 未知 mono/multi（§5.8、Q1） | LRS2 = mono；RealSSA-Sim 原始 4ch、转换取 `mean`（实测 4ch-mean vs 转换 max|Δ|=2.29e-5、ch0 vs 转换 0.1645 证明是 mean）；RealSSA-Real = stereo（mean/0/1 可选）；F5 已加单通道守卫 | **修正（三套数据各自的通道事实）** |
| 9 | `target.json` 第 3 字段从未被读取、target/mix 长度不校验（P1-2） | `[代码]` 仍成立：两个 Dataset 只用 `[0]/[1]`；转换器抽样验证长度一致 | **确认** |
| 10 | `train_samples` 不在 YAML、默认 10000（P1-3） | LRS2 三个训练配置已显式写 `train_samples: 10000`；`configs/Falcon.yml` 仍缺 | **部分修正（仅 LRS2 配置）** |
| 11 | `check_avspeech_dataset.py` 是不可运行旧拷贝、无守卫（P2-2、A58） | 守卫已同步；`${DATA_ROOT}` 占位符仍在 `:319`，直接运行仍失败 | **部分修正（守卫已修、占位符未修）** |
| 12 | `datas/__init__.py` 只导出旧 `AVSpeechDataModule`（§2.1） | 仍是 5 行、只导出旧类（`look2hear/datas/__init__.py:1-5`） | **确认** |
| 13 | `get_preprocessing_pipelines`：train=Normalize(0,255)+RandomCrop(88)+Flip(0.5)+Normalize(.421,.165)；val=CenterCrop(88)（§2.4、A19/A20） | 真实 uint8 `.pt` 上逐元素验证 val == `(x/255-.421)/.165`（中心 4:92）为 True；train 输出 `(50,88,88)` f64 | **确认** |
| 14 | LRS2-2Mix 是"纯音频/7,234 wav"的旧计划说法（旧文档 §5.9 引用 REPRODUCTION_PLAN；T6 已更正） | 实测 `lrs2_rebuild`：wav16k 28,000 组 + mouths 20,392 + faces 20,392 + raw_audio 20,392；T6 tar 249,670 entries | **推翻（以 T6/本轮实测为准）** |
| 15 | near 源采样率不入契约、Dataset 不重采样（P1-7、A66） | 代码未变；LRS2 s2 = 16 kHz 故当前无影响 | **确认（潜在风险仍在）** |
| 16 | Studio 与训练 val 管线数值等价（§3.9、A60） | 本任务未复核（跨产品线，属 studio 文档）；本轮只验证训练侧变换 | **未复核（沿用 [文档]）** |
| 17 | `AVSpeechTrainDatasetWithNear.__len__=samples`、`__getitem__` 忽略 idx、有放回（§4.2、A13/A14） | 代码未变（`avspeech_dataset_near.py:147-157`）；本轮 `len(ds)=8`（samples=8）而池 40000 已实测 | **确认** |
| 18 | `test_loader` 是第二个 validation loader（A64） | 代码未变；本轮未跑训练，沿用 train-sys 结论 | **确认（[文档]+[代码]）** |
| 19 | RealSSA-Real 只能 DNSMOS、不能 SI-SDR（Q/§5） | 已用官方 loader 全量跑通 54/54；`full3` P808 3.3285/OVRL 2.7901 | **确认并已执行** |
| 20 | RealSSA-Sim 训练/评测数据可得性（旧文档只按"gated 未获批"讨论） | 已获批：tt 已转换 9169；tr/cv 只在 TOS；near 仍缺 | **修正（状态推进，仍不满足训练）** |
| 21 | 旧文档 P2-13 I/O：训练每样本 `torch.load` 整段 `.pt` 只取 50 帧 | LRS2 `.pt` 恰好 50 帧（无浪费）；RealSSA tt 平均 ~140 帧，训练时会整段 load（浪费 ~3×） | **部分修正（依数据）** |
| 22 | 旧文档 P2-12 mouth 路径由字符串替换生成（`replace("target","mouths")`） | `process_lrs23.py:25` 未变；LRS2 转换器不用该路径（直接按 id 生成） | **确认（仅 DataPreProcess 路径）** |

---

## 6 再复现一次所需的数据与步骤

### 6.0 前置

```bash
cd /root/code/github_repos/Falcon-fork
export PYTHONPATH=$PWD
ENV=/root/miniforge3/envs/falcon/bin/python
# 磁盘：/root 剩余需 >= 60 GB（LRS2 链路）/ >= 1 TB（RealSSA 全量）；TMPDIR=/root/tmp
```

### 6.1 LRS2-2Mix 单目标（28,000 条，约 24G）——**已就绪，可跳过；从零重跑如下**

```bash
# 1) 获取（公开；走代理）——已有 /root/data/lrs2_2mix/lrs2.tar.gz 则跳过
$ENV tools/repro/fetch_data.py --dataset lrs2-2mix --proxy http://127.0.0.1:17897
# 2) 解压（tar 16.47GB → lrs2_rebuild 17.39GB）
mkdir -p /root/data/lrs2_2mix && tar xzf /root/data/lrs2_2mix/lrs2.tar.gz -C /root/data/lrs2_2mix
# 3) 转换（target=s1；fallback → K=0，与 P2-C 原始产物一致）
$ENV tools/repro/lrs2_2mix_to_falcon.py --jobs 16 \
  --out /root/data/falcon_lrs2_2mix --near-mode fallback --validate-samples 200
#    期望：tr/cv/tt = 20000/5000/3000；mouths 14773/476/267；总 ~23.36 GiB；validation problems=0
# 4)（可选，当前磁盘上那份 K=1 的来源）用 s2 作 near 负样本 —— 注意：仓库暂无此脚本（D1），需补
# 5) 校验（只读）
$ENV -m pytest tools/repro/tests/test_lrs2_convert.py tools/repro/tests/test_phase2_data.py -q
```

### 6.2 LRS2-2Mix 论文配方（40,000/10,000/6,000，约 26G；依赖 6.1 的 `lrs2_rebuild`）

```bash
$ENV tools/repro/phase2/build_lrs2_paper_data.py \
  --src /root/data/lrs2_2mix/lrs2_rebuild \
  --out /root/data/falcon_lrs2_2mix_paper --jobs 12
# 期望：tr 40000/cv 10000/tt 6000；mouths 18691/476/267；音频不复制（JSON 引用原 wav）
# 训练：configs/LRS2_2Mix_Paper_8gpu.yml（8×bs5=40、lr1e-3、1000 epoch；需 8 卡环境）
# 本机 2 卡可跑 2gpu_smoke（2×bs4×accum5）验证 DDP/双向/near 通路
bash tools/repro/phase2/run_lrs2_paper_2gpu_smoke.sh
```

> 注意：`_paper` 数据**不自包含**；迁移到训练节点时必须同时带 `lrs2_rebuild` 或重跑 `--copy-wavs`（D12）。benchmark 评测仍用 `falcon_lrs2_2mix/tt`（3000 条，target=s1），不要用 `_paper/tt`（6000 双向）。

### 6.3 RealSSA-Sim：只做评测（当前就绪）

```bash
# 已有 /root/data/falcon_realssa_sim/tt（9169 条，30G），直接评测：
PYTHONPATH=$PWD $ENV audio_visual_test.py --conf_dir configs/Falcon.yml \
  --data_dir /root/data/falcon_realssa_sim/tt \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt --gpus 0
# 论文口径：SI-SNRi 7.7 / SDRi 8.2；本机已复现 7.774/8.479（channel=mean，align=min）
# 若要换通道复跑：重新转换并显式声明 channel
$ENV tools/repro/realssa_to_falcon.py --src /root/data/RealSSA-Sim --out /root/data/falcon_realssa_sim_ch0 \
  --splits tt --channel 0
```

### 6.4 RealSSA-Sim：从零训练（**未就绪**，需要的步骤）

1. 从 TOS 取全量：`rsync -a /aispeech-cfs/voiceprint/data/multimodal/realssa_sim/ /root/data/realssa_sim_raw/`（521.5 GB，240 shard）。
2. 决定通道协议（mean vs 0）并记录；near 方案：无官方 near → 接受 K=0，或自备 `--near-dir <id>.near*.wav`（目前没有）。
3. 转换 tr/cv/tt：`$ENV tools/repro/realssa_to_falcon.py --src /root/data/realssa_sim_raw --out /root/data/falcon_realssa_sim_full --channel mean`（预计 0.3–0.4 TB，数小时；逐 shard 可断点）。
4. 数据自检：`conversion_stats` 的 `skipped_incomplete/skipped_error`、`n % 640 == 0`、mouth `[T,96,96,3]`、`near.json` 是否 K=0（预期是）。
5. 训练时注意 <2 s 样本会被丢弃（tt 有 11.4%），先统计 tr 的丢弃率。
6. 训练配置需新建（现有配置都指向 LRS2）；对比损失分支应 `use_near:false` 或接受 K=0 语义。

### 6.5 RealSSA-Real：DNSMOS 评测（当前就绪）

```bash
$ENV tools/repro/phase2/eval_realssa_real.py --data /root/data/RealSSA-Real \
  --channel mean --device cuda --baseline --out-dir /root/data/falcon_eval/realssa_real_mean_full4
# 已有成功结果 full3：54/54，OVRL 2.7901 / P808 3.3285；论文 DNSMOS 目标 3.276±0.05
# 注意 full/ full2 是失败残留（D8），不要混用
```

### 6.6 验收清单（每项都能在当前机器只读重跑）

- [ ] `du -sh` 六个目录与 §2 表一致；
- [ ] 三套数据 `len(mix)==len(target)==len(near)`；
- [ ] LRS2 抽样 `mix==s1+s2`（max|Δ| ≤ 3.05e-5）、`target==s1`（0.0）；
- [ ] 全部 mouth `frames == n_samples//640`、dtype uint8、`shape[1:]=(96,96,3)`；
- [ ] `AVSpeechTrainDatasetWithNear` 取 batch：`(B,32000)` f32 / `(B,50,88,88)` / K 符合预期；
- [ ] `AVSpeechEvalDataset` 在 RealSSA-Sim：整段 `(T,)` + `(F,88,88)`；
- [ ] `pytest tools/repro/tests/test_{lrs2_convert,realssa_convert,synthetic_data,phase2_data}.py` → 22 passed。

---

## 7 未解问题

| # | 问题 | 影响 | 谁能答 |
|---|---|---|---|
| Q1 | `falcon_lrs2_2mix/{tr,cv,tt}/near.json`（s2 负样本）由哪条命令、在 2026-10-22 16:37 生成？为何不入库？ | P2-D contrastive 结果（+0.24 dB）的一键复现 | 原作者/Lead；本文件已给出建议补脚本 |
| Q2 | 论文 LRS2-2Mix 训练的 near 负样本具体怎么构造？是否就是"另一说话人"？ | 14.1/14.7 是否可达；负样本语义 | 作者 issue `JusperLee/Falcon#1`（未回复） |
| Q3 | 论文 LRS2-2Mix 训练是否真的双向 target？s1/s2 如何挑选 video？ | 评估口径与数据量 | 作者/论文补充材料 |
| Q4 | RealSSA-Sim 论文用的通道（mean 还是 ch0）？ | SI-SNRi 比较（mean 已复现 7.774/8.479，但论文未声明） | 作者；当前结果需声明 mean |
| Q5 | RealSSA-Sim tr/cv 全量转换是否值得做（0.3–0.4 TB + 训练算力）？数据在 TOS 是否长期可用？ | "从零训练 RealSSA" 的可行性 | Lead/用户 |
| Q6 | 是否有外部 near 波形可接（`--near-dir`）用于 RealSSA 对比学习？ | 论文贡献之一能否在 RealSSA 上复现 | 用户/组内数据 |
| Q7 | `_meta.json` 里 near 组件存在但无波形时，是否应从转换产物中删除 near 组件（避免 val `n_near` 误导）？ | 验证分桶语义 | 数据侧/训练侧 |
| Q8 | RealSSA tr 的时长分布/丢弃率（<2 s 样本比例）？ | 训练的样本浪费率 | 需转换 tr 后统计 |
| Q9 | LRS2 8 kHz 音频是否有人用？（`audio/wav8k` 已发布） | 无（当前全部 16k） | 无人 |
| Q10 | `docs/DATA_ACCESS_GUIDE.md` 的权限状态由谁更新（仍写 403）？ | 新复现者会误判 | Lead/文档 owner |

---

## 8 验证记录

### 8.1 环境与约束

- 解释器：`/root/miniforge3/envs/falcon/bin/python`（3.11.17，torch 2.8.0+cu128 可导入；未用 `/usr/bin/python3`）。
- 所有命令均 `cd /root/code/github_repos/Falcon-fork && PYTHONPATH=$PWD`；只读数据；未安装包、未联网、未 git 写。
- 单条命令最长 <5 min；大数据目录只用 `ls/find -maxdepth/计数/抽样`，未对 TOS 做 `du`。

### 8.2 本轮实际执行的命令（摘要）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `du -sh` 六个数据目录 | 24G/26G/30G/38G/12G/33G（与任务描述一致） |
| 2 | `df -h / /root` | `/` 20G 用 32%；`/root` 164T 用 92% |
| 3 | `ls/find` TOS `realssa_{sim,real}` | sim 243 文件（202/19/19 shard）、tt 40.14 GB；real 423 文件 |
| 4 | `json.load` 计数 9 个 split | 三表等长；paper 40000/10000/6000；realssa 9169 |
| 5 | 抽样 15 条 LRS2：`sf.read` 对比 s1/s2 | `max|mix-(s1+s2)|=3.05e-5`；`target-s1=0.0` |
| 6 | 抽样 mouth：`torch.load` + `Grayscale` + 管线 | `(50,96,96,3) uint8` → `(50,88,88) f64`；`val == center-crop` True |
| 7 | 全局 near 检查 | 单目标 near=s2（20000/5000/3000）；paper near=other；realssa fallback 9169/9169 |
| 8 | 实例化真实 Dataset/DataLoader 取 batch | `(4,32000) f32`、`(4,50,88,88) f64`、K=1；RealSSA eval `(92800,)`、`(145,88,88)`、n_near=2 |
| 9 | 解 1 个 RealSSA tar 样本 vs 转换产物 | mean 通道 max|Δ|=2.29e-5、ch0=0.1645、mouth 逐元素相等、frames=min 对齐 |
| 10 | paper 双向结构全量检查 | 偶数 s1/奇数 s2 各半；配对 mouth 全不同；唯一 mix=20000 |
| 11 | `pytest tools/repro/tests/{4 files} -q` | **22 passed in 45.83s** |
| 12 | RealSSA-Sim 长度分布 | min 1.0 s / max 7.52 s；1047 条 <2 s；`n%640==0` 全真 |
| 13 | 查 near.json mtime vs conversion_stats/日志 | near.json 2026-10-22T16:37:54；转换统计 2026-10-21T19:24（D1 证据） |
| 14 | 查 `_archive_done.json`、`_MANIFEST.json`、`falcon_eval/*/summary.json` | 归档 ok；Sim manifest 521,532,645,652 B；RealSSA-Real full3 54/54 ok |
| 15 | `git log/show` 查 F5 修复与 near 相关提交 | `ac91800` 修 F5；`6d2df43/141c751` 只加配置/自动续训脚本，**没有** near.json 生成脚本 |

### 8.3 验证边界（没做的事）

- 未跑训练/评测端到端（属 GPU 长任务）；训练期 K=1 行为只从 `P2D-near-train.log:15-18` 与 Dataset 取批实证，未重跑 130 epoch。
- 未对 RealSSA-Sim 全量 tar 逐样本校验（只解 1 个 shard 的 1 个样本 + 19 shard 计数）；未统计 tr 的真实样本数（卡片 100,826 未用本地数据复核）。
- 未 `du` TOS 521 GB 目录（只 stat/ls/count），未核对 TOS 与本地 tar 的 per-file sha256。
- 未复核 Studio 嘴部预处理等价性（旧文档 A60），未复核 `test_loader` 双验证 loader 的训练实测（沿用旧结论）。
- 未验证 `lrs2_2mix_to_falcon.py --near-mode raw_audio` 的实际产物（本轮全量数据用的是 fallback / 未入库的 s2 改写）。

### 8.4 本轮产出

- 本文件：`docs/understanding/deepdive-20261024/05-data-pipeline.md`（唯一写入）。
- 未修改任何源码、他人文档、数据或仓库历史。
