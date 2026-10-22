# Falcon 数据获取与访问指南（Phase 1 · T6）

> 适用分支：`repro/phase1`（commit 基线见 `docs/REPRODUCTION_PLAN.md`）。
> 目标：把“复现需要哪些数据、从哪拿、能不能拿、拿到后怎么变成仓库可用格式、评测协议是什么”一次讲清，避免
> Phase 1 的其他任务在数据上重复踩坑。配套：`docs/REPRODUCTION_PLAN.md` §5/§11（方案与资产核实）、
> `tools/repro/fetch_data.py`（统一获取脚本）、`docs/verification/phase1/T6-data-fetch.log`（下载日志）、
> `docs/verification/phase1/T6-report.md`（执行报告，含 tar 结构分析）。
> 本文所有“可访问/已授权/大小”结论均为 **2026-10-21 本机实测**（HF API + dry-run + 只读探测）；数据卡内容摘自各
> HF 数据集 README（`JusperLee/*`），链接与许可以数据集所有者页面为准。

---

## 0. TL;DR

| 数据集 | HF 仓库 | gated | 本机可下载？ | 体积 | 含什么 | 能支撑的复现 |
|---|---|---|---|---|---|---|
| **LRS2-2Mix** | `JusperLee/LRS2-2Mix` | ❌ 公开 | ✅ **是（已下载完成并核实）** | **16.47 GB 压缩 / 17.39 GB 解压**（单文件 `lrs2.tar.gz`） | **完整 2 s AV 重构包**：`mix/s1/s2`（16k+8k，tr 20000/cv 5000/tt 3000）+ `mouths` npz(50,96,96) + `faces` mp4 + `landmarks` + 混音脚本/清单 | L2 的 LRS2-2Mix 列（AV，完整）+ 训练数据转换；**无需 LRS2 原始视频** |
| **RealSSA-Sim** | `JusperLee/RealSSA-Sim` | ✅ manual | 🟡 **申请已提交（2026-10-22，账号 `nanless`/USTC/CN），待人工审批；数据文件仍 403** | **521.53 GB**（**242 files / 240 shards**：tr 202 + cv 19 + tt 19；tr 441.42 / cv 39.97 / tt 40.14 GB；另 2 个文件为 README/.gitattributes） | 4ch 16k mix/target、`mouths.pt`（uint8 RGB [T,96,96,3]）、scene JSON；**没有 near 源波形** | L2 RealSSA-Sim 评测 + L3 训练（需转换；对比损失负样本需替代方案） |
| **RealSSA-Real** | `JusperLee/RealSSA-Real` | ✅ manual | 🟡 **申请已提交（2026-10-22，账号 `nanless`/USTC/CN），待人工审批；数据文件仍 403** | **12.84 GB**（422 文件） | 真实场景 AV 输入（stereo 16k）、face MP4/mouth NPZ/landmarks、10 条 enrollment、40 条 recording GT；**无 SI-SDR 参考** | L2 RealSSA-Real：DNSMOS / WER / speaker-similarity；**不能算 SI-SDR** |
| **LRS3-2Mix** | `JusperLee/LRS3-2Mix` | ❌ 公开 | ⚠️ 仓库**无数据文件** | 0 B（仅 README 622 B） | 只有数据卡 | ❌ 不能复现该列；需联系作者或自建 |
| **VoxCeleb2-2Mix** | `JusperLee/VoxCeleb2-2Mix` | ❌ 公开 | ⚠️ 仓库**无数据文件** | 0 B（仅 README 756 B） | 只有数据卡 | ❌ 同上 |

**一句话**：现在唯一能立即下载的是 LRS2-2Mix（**完整 2 s AV 数据**：音频 mix/s1/s2 + mouths + faces + landmarks，16.47 GB 压缩/17.39 GB 解压）；RealSSA-Sim/Real 需人工审批；LRS3/Vox 的 2Mix
发布仓库里没有数据。训练用的 **near 源波形官方未发布**，RealSSA-Real **不能算 SI-SDR**（无干净参考）。

---

## 1. 本机 HF 账号、token 与网络现状（只读核查；访问申请已由用户另行提交，见 §11）

| 项 | 实测结果 | 证据/命令 |
|---|---|---|
| token 文件 | 存在：`/root/.cache/huggingface/token`（37 B） | `ls -l /root/.cache/huggingface/token` |
| 登录身份 | `whoami()` → **`nanless`（type=user）**，token 有效 | `python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"` |
| RealSSA-Sim 授权 | 🟡 **申请已提交待批（2026-10-22）**；仍**未获批**：dry-run 探测 `tt/tt-000000.tar` → `GatedRepoError 403` | `python tools/repro/fetch_data.py --dataset realssa-sim --dry-run` |
| RealSSA-Real 授权 | 🟡 **申请已提交待批（2026-10-22）**；仍**未获批**：dry-run 探测 `INFERENCE.md` → `GatedRepoError 403` | 同上 `--dataset realssa-real` |
| gated 元数据可见性 | ✅ 可读：数据集卡/文件清单/大小对未获批账号可见（HF 设计），但**数据文件不可下载** | `HfApi().dataset_info(...)` 成功；非 README 文件 403 |
| LRS2-2Mix | ✅ 公开，可下载 | `--dataset lrs2-2mix --dry-run` 列出 `lrs2.tar.gz` 16.47 GB |
| LRS3/VoxCeleb2-2Mix | ✅ 公开但只有 README（文件数 2） | `dataset_info` 文件清单 |
| **未做** | 未修改账号、未分享 token（访问申请由用户于 2026-10-22 自行提交，见 §11） | 只读探测 |

**网络（本机特殊点，重要）**：
- `HF_ENDPOINT=https://hf-mirror.com` 在本沙箱 **TLS 握手失败**（`SSL_ERROR_SYSCALL`；TCP 443 可连但无 HTTP 响应），
  `huggingface.co` 直连也超时；**必须走本机 mihomo 代理 `http://127.0.0.1:17897`**（`/root/.bashrc` 已配置，但非交互 shell 不自动生效）。
- 因此本指南/脚本的统一姿势：
  ```bash
  export https_proxy=http://127.0.0.1:17897 http_proxy=http://127.0.0.1:17897
  # 或给脚本传 --proxy http://127.0.0.1:17897
  ```
  `tools/repro/fetch_data.py` 支持 `--proxy`、`--endpoint`、以及环境变量 `HF_ENDPOINT`/`http(s)_proxy`；
  若某天 hf-mirror 恢复，可直接 `--endpoint https://hf-mirror.com`（无需改代码）。
- 磁盘：`/root`（vepfs，15 TB 可用）；LRS2 全量 + RealSSA-Sim 全量需预留 **≥600 GB**。

---

## 2. 统一获取脚本 `tools/repro/fetch_data.py`

```bash
# 查看计划（不写盘）：文件、大小、gated/授权状态、申请入口
python tools/repro/fetch_data.py --dataset all --dry-run --proxy http://127.0.0.1:17897

# LRS2-2Mix：公开大文件，断点续传下载到 /root/data/lrs2_2mix/lrs2.tar.gz
python tools/repro/fetch_data.py --dataset lrs2-2mix --proxy http://127.0.0.1:17897
# 中断后再执行同一命令即续传（内部 curl -C -）

# RealSSA-Sim（获批后）：先取 1 个 test shard 验证管线（≈2.4 GB），再按需扩
python tools/repro/fetch_data.py --dataset realssa-sim --shard tt-000000 --proxy ...
python tools/repro/fetch_data.py --dataset realssa-sim --pattern 'tt/*.tar' --proxy ...   # test 全量 ≈40.1 GB
python tools/repro/fetch_data.py --dataset realssa-sim --full --proxy ...                # 全量 ≈521.5 GB

# RealSSA-Real（获批后）：inference 评估所需完整 snapshot
python tools/repro/fetch_data.py --dataset realssa-real --full --proxy ...

# 输出目录
#   默认：/root/data/{lrs2_2mix,realssa_sim,realssa_real}
#   --out 指定：单数据集时就是目标目录；--dataset all 时为父目录（内部再分子目录）
```

脚本行为要点：
- `--dry-run`：**不写盘**；列出匹配文件与大小；gated 数据集做只读授权探测并打印申请链接/步骤。
- `--shard`：RealSSA 简写 `tt`（整个 split）→ `tt/*.tar`；`tt-000000` → `tt/tt-000000.tar`；也可传 `tr/cv/tt`。
- **gated 未获批时退出码 3**，并打印申请 URL、表单要求、`hf auth login` 步骤；**不会**替用户提交申请、不会绕过限制。
- LRS2 大文件走 curl 断点续传；其余走 `snapshot_download(allow_patterns=...)` 并保留 HF 缓存/`.incomplete` 语义。
- 所有下载默认写 `/root/data/`（仓库外），不会污染 git。

---

## 3. LRS2-2Mix（公开，唯一可立即获取的数据）

### 3.1 URL / 状态 / 许可证

| 项 | 值 |
|---|---|
| HF 数据集 | <https://huggingface.co/datasets/JusperLee/LRS2-2Mix> |
| gated | **否**（公开）；HF `license` 标签：`apache-2.0` |
| 文件 | `README.md`（694 B）、`lrs2.tar.gz`（**16,470,508,590 B ≈ 16.47 GB**） |
| 本地状态（2026-10-21） | ✅ **已下载并解压完成**：sha256 `ebdfdf31335f288cc2e4b1690a0cc2f260e1826a68a020723ad4ce59a033767e`（与 HF `x-linked-etag` 一致）；成员总大小 **17,390,672,317 B ≈ 17.39 GB**；`mix/s1/s2` tr 20000/cv 5000/tt 3000；实测 **`mix==s1+s2`**（16-bit 量化误差内）；`mouths` **uint8 (50,96,96)**（抽查 200/20392 全部一致）；`raw_audio` 全部 **2.000 s/16k/mono**（抽查 500/20392） |
| 流量 | 单文件，建议断点续传；下载日志 `docs/verification/phase1/T6-data-fetch.log` |
| ⚠️ 许可注意 | HF 标签是 apache-2.0，但内容来自 **BBC LRS2**（研究用途，LRS2 自身有独立申请/许可）；商用/再分发前务必核对 LRS2 条款。数据集卡只描述“两说话人随机混合、SNR −5~5 dB、11h/3h/1.5h”。 |

### 3.2 tar 内容结构（2026-10-21 完整下载并实测；249,670 entries）

下载（已完成，见 `docs/verification/phase1/T6-data-fetch.log`）：
- HF API：`gated=False`；单文件 `lrs2.tar.gz` = **16,470,508,590 B**；本地 sha256 = `ebdfdf31335f288cc2e4b1690a0cc2f260e1826a68a020723ad4ce59a033767e`，
  与 HF resolve 的 `x-linked-etag` **一致**；`hf-mirror.com` 本机 TLS 不可达，实际走 `huggingface.co` + 本机 mihomo 代理（见 §1）。
- 压缩 16.47 GB → 解压 **17.39 GB**（tar 成员总大小 17,390,672,317 B）；建议预留 **≥40 GB**（含压缩包/解压/转换副本，保守 60 GB）。
- 无需解压全量即可查看：`tar tzvf lrs2.tar.gz | head` / `awk -F. '{print $NF}' | sort | uniq -c`；完整核对见 `docs/verification/phase1/T6-lrs2-structure.txt`。

```text
lrs2_rebuild/
├── raw_audio/{train,val,test}/<clipid>_<utt>.wav     # 19,649+476+267 = 20,392 条，16 kHz mono PCM16，2.000 s
├── audio/
│   ├── wav16k/min/{tr,cv,tt}/{mix,s1,s2}/*.wav       # tr 20000 / cv 5000 / tt 3000 组，2.000 s mono 16k PCM16
│   │   └── scaling.mat                                # scaling_16k(N,2)、scaling16bit_16k(N,1)（8k 同）
│   └── wav8k/min/{tr,cv,tt}/{mix,s1,s2}/*.wav        # 8 kHz 版本，与 16k 一一对应
├── mouths/<id>.npz          # data: uint8 (50,96,96)（2 s@25 fps 灰度）；id = raw_audio 同 id
├── landmark/<id>.npz        # data: object array len 50；每帧 1（少数 2）个 float32 (68,2) 人脸点
├── faces/<id>.mp4           # 20,392 个，224×224、25 fps、50 帧
├── create_mix/              # create_wav_2speakers.m / activlev.m / maxfilt.m + mix_2_spk_{tr,cv,tt}.scp + min_* lists
├── audio_process/           # create_mixtures.py（Python 版）、mix_2_spk_*.txt（历史绝对路径清单）
├── crop_mouth_from_video.py / detectface.py / utils.py / transform.py / 20words_mean_face.npy
├── filename_input.csv / video_path.txt / train.txt / retrain.txt / dev.txt / retest.txt / redev.txt / test.txt
└── load_npz.py / resave.py / __pycache__/ …（作者机历史痕迹）
```

**结构核实结论**（详见 `T6-lrs2-structure.txt`）：
- **split 与规模**：mix/s1/s2 各 28,000 组（tr 20,000 / cv 5,000 / tt 3,000）；源 raw 20,392 条（train 19,649 / val 476 / test 267）。
  tr 使用 18,691 个 train ids、cv 用全部 476 个 val、tt 用全部 267 个 test → **无跨 split 泄漏**。
- **每个样本恰好 2.000 s**：16k=32,000 样本、8k=16,000 样本，mono PCM16（随机 tr/cv 各 100 + tt 全部 3000 验证）。
- **mix = s1 + s2**（20 个/组×3 split，max|Δ|=3.05e-5，纯 16-bit 量化）→ **`s1`/`s2` 已是精确的 scaled target 参考，无需再从 raw_audio 重建**。
  若需重建/换 convention：按 `create_wav_2speakers.m` 的公式（activlev 归一化 + `10^(snr/20)` 加权 + min 长度 + 统一 `0.9/max_amp` 缩放）与 `scaling.mat` 可逐样本精确复算。
- **命名**：`<id1>_<snr1>_<id2>_<snr2>.wav`，`snr2=-snr1`，`snr1∈(0,5]`（等价源间能量比 0~10 dB，逐样本）；mix/scp 列表一一对应。
- **视觉**：`mouths` 为 `(50,96,96)` uint8（2 s@25 fps，**正好是 Falcon 2 s 窗口的嘴部输入**）；faces 为 224×224/25fps/50 帧；landmarks 为逐帧 (68,2)。
- **文本清单**：`create_mix/mix_2_spk_{tr,cv,tt}.scp` = 20,000/5,000/3,000 行（`wav1 snr1 wav2 snr2`，相对 `../raw_audio/{train,val,test}`）；
  `audio_process/mix_2_spk_tr.txt` 是 50,000 行旧版绝对路径清单（与 scp 20,000 对不重合，**勿用**）；
  `train.txt` 20,779 / `retrain.txt` 19,649 / `dev.txt` 492 / `retest.txt` 267 / `redev.txt` 476 / `test.txt` 296 / `video_path.txt` 20,469 / `filename_input.csv` 20,392。
- **计划口径**：LRS2-2Mix 的正确结构与构建方式见 `docs/REPRODUCTION_PLAN.md` §5.2（已修订）；本节与 `docs/verification/phase1/T6-lrs2-structure.txt` 为 2026-10-21 实测证据（188,392 个 wav + 20,392 mouths/faces/landmarks 的完整 AV 包）。

### 3.3 如何转成 Falcon 训练/评测格式（可直接执行；target=s1、mouths→.pt 补 3 通道）

目标目录（`DATA` 与 `INDEX` 均在仓库外；不要放进 git）：
```text
$DATA/{tr,cv,tt}/mix/<mixture_id>.wav       # 来自 audio/wav16k/min/<split>/mix/
$DATA/{tr,cv,tt}/target/<mixture_id>.wav    # 来自同 split 的 s1/（target=s1 convention）
$DATA/{tr,cv,tt}/mouths/<mixture_id>.pt     # uint8 [50,96,96,3]，来自目标说话人 id1 的 mouths/<id1>.npz
$INDEX/{tr,cv,tt}/{mix,target,near}.json    # 由 DataPreProcess/process_lrs23.py 生成
```
其中 `mixture_id = <id1>_<snr1>_<id2>_<snr2>`（= mix 文件名去 `.wav`）；**target=s1 ⇒ target 说话人 = id1**，
嘴部取 `mouths/<id1>.npz`（不是 mixture id）。若改用 s2 作 target，则 target 取 `s2/`、嘴部取 `<id2>.npz`，并保持全库一致。

**步骤 A/B：复制音频 + 生成 mouths `.pt`（3 通道，匹配训练 loader 的 `permute(0,3,1,2)→Grayscale→squeeze(1)`）**
```python
import os, re, shutil
import numpy as np, torch

SRC  = "/root/data/lrs2_2mix/lrs2_rebuild"
DATA = "/root/data/lrs2_2mix/falcon"
PAT  = re.compile(r"^(\d+_\d+)_(-?\d+(?:\.\d+)?)_(\d+_\d+)_(-?\d+(?:\.\d+)?)$")
for split in ("tr", "cv", "tt"):
    for d in ("mix", "target", "mouths"):
        os.makedirs(f"{DATA}/{split}/{d}", exist_ok=True)
    mixdir = f"{SRC}/audio/wav16k/min/{split}/mix"
    for fn in sorted(os.listdir(mixdir)):
        if not fn.endswith(".wav"):
            continue
        mid = fn[:-4]
        m = PAT.match(mid); assert m, mid
        id1 = m.group(1)                                  # target speaker (s1)
        shutil.copy(f"{mixdir}/{fn}",                f"{DATA}/{split}/mix/{mid}.wav")
        shutil.copy(f"{SRC}/audio/wav16k/min/{split}/s1/{fn}", f"{DATA}/{split}/target/{mid}.wav")
        mouth = np.load(f"{SRC}/mouths/{id1}.npz")["data"]   # uint8 (50,96,96) 灰度
        mouth = np.repeat(mouth[..., None], 3, axis=-1)      # -> (50,96,96,3) RGB（灰度复制，既满足 3 通道约定也兼容 1 通道）
        torch.save(torch.from_numpy(mouth), f"{DATA}/{split}/mouths/{mid}.pt")
```
> 说明：3 通道是训练 `.pt` 的约定格式（RealSSA-Sim 的 mouths 也是 `[T,96,96,3]`）；`Grayscale(num_output_channels=1)`
> 在 3 通道输入上会正确转回单通道（已在本机 falcon 环境实测），之后 `squeeze(1)` 得 `(T,96,96)`，再 center/random crop 88。

**步骤 C：生成 mix/target 索引 JSON（near.json 只可能是 fallback，训练需 `use_near=false`）**
```bash
cd /root/code/github_repos/Falcon-fork
/root/miniforge3/envs/falcon/bin/python DataPreProcess/process_lrs23.py \
  --in_audio_dir /root/data/lrs2_2mix/falcon \
  --out_dir      /root/data/lrs2_2mix/index
# 产出 index/{tr,cv,tt}/{mix,target,near}.json
```
- `process_lrs23.py` 用 `target` 路径替换出 `mouths/<mid>.pt`，与本目录布局一致；`mix.json` 的条目为 `(wav_path, n_samples)`。
- 注：`process_lrs23.py` 要求三个 split 的 `mix/target` 目录都存在（空 split 会触发 `min()` 报错；本数据集三 split 均非空）。
- 本数据包**没有 near 源波形**：`near.json` 会被写成 `[[mix_path, n]]` 的 fallback（Dataset 在 `use_near=true` 时会跳过 fallback → K=0）；
  训练时请用 `use_near=false`（或自行构造替代负样本并在报告标注“非官方负样本”），**不要把 fallback 当合法近场源**。

**步骤 D：快速自检（可选但推荐）**
```python
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")
from look2hear.datas.avspeech_dataset_near import AVSpeechTrainDatasetWithNear
ds = AVSpeechTrainDatasetWithNear(json_dir="/root/data/lrs2_2mix/index/tr",
                                  sample_rate=16000, segment=2.0, use_near=False,
                                  samples=10000)          # 直接实例化必须给 samples；否则 __len__ 返回 None
mix, src, mouth, name, near = ds[0]
print(mix.shape, src.shape, mouth.shape, name)   # 期望 (32000,), (32000,), (50,88,88), "<mid>.wav"
```
本机已用 **3 样本/split** 实测跑通该链路：`process_lrs23.py` → `AVSpeechTrainDatasetWithNear` 返回
`mix(32000,) / src(32000,) / mouth(50,88,88)`（`samples=10000`；dtype 为 float64，训练 litmodule 会 `type_as(mixtures)` 统一为 float32）。
注意：clip 恰好 2.000 s（=segment），`rand_start` 恒为 0；若扩展为长片段需同步调整 `segment/fps_len`。
已知小坑：`process_lrs23.py` 在某个 split 为空时会因 `min(empty)` 报错（LRS2-2Mix 三个 split 均非空，不受影响）；
也可用 `look2hear/datas/check_avspeech_dataset.py` 对整个索引做一致性检查。

**步骤 E：评测口径**
tt 3,000 组、每组 2 s → 模型 2 s 窗口正好覆盖整段，可按 clip 计算 SI-SNRi/SDRi（target=s1，固定声明；也可 s1/s2 双目标平均）；
评测入口仍走 `audio_visual_test.py` 的 folder eval（整句读取=2 s clip）；与论文对比时注明 target convention、16k/8k、是否双目标平均。

## 4. RealSSA-Sim（gated manual，主训练/评测集）

### 4.1 URL / 状态 / 大小

| 项 | 值 |
|---|---|
| HF 数据集 | <https://huggingface.co/datasets/JusperLee/RealSSA-Sim> |
| gated | **manual**（人工审批）；本机账号 `nanless` 申请已提交（2026-10-22）**待批**，当前仍 **403** |
| 许可证 | **CC BY-NC-ND 4.0**（TED 内容），另有各噪声源条款；仅限**非商业学术研究** |
| 规模 | **242 个文件 / 240 个 shard**：**521.53 GB**；`tr/` 202 shards=441.42 GB、`cv/` 19 shards=39.97 GB、`tt/` 19 shards=40.14 GB；另有 README/.gitattributes 2 个文件 |
| 样例数 | train 100,826 / validation 9,133 / test 9,169（shard 500 条；4 个训练样本缺 JSON/mouth 被排除） |
| 申请表单字段（card 定义） | Full name / Affiliation / Country / Intended research use 四个 text/country 字段 + **3 个 checkbox**：非商业学术用途、同意 LRS3/TED/各源数据条款、不转发数据/不共享账号 |
| 联系 | <thucs.kaili@gmail.com>（访问/删除/撤回请求） |

### 4.2 格式（card 原文要点）

WebDataset tar shards `{tr,cv,tt}/{tr,cv,tt}-NNNNNN.tar`，每个例 `<id>` 为连续 4 个成员：

```text
<id>.mix.wav      4-channel PCM16 mixture, 16 kHz
<id>.target.wav   4-channel PCM16 target speech, 16 kHz
<id>.json         scene/room(dimensions, mic center, T60)/components(type,source,SIR,position)/target position/transcript
<id>.mouths.pt    uint8 RGB mouth crops, torch tensor [T,96,96,3]
```
- `components.type ∈ {near, far, env, sudden}`；`path` 字段只是生成时来源路径（provenance，不可当作本地路径）。
- **没有 near 源波形**（card 明确：“The shards ship mixtures and target speech, not the original individual near-speaker signals”）。
  对比学习需要的 near negatives 官方未发布 → 只能：关闭对比损失做消融、用同 batch/外部不同说话人近似 negatives、
  或用 card 的房间几何 + LibriSpeech 干信号重卷积近似重建（高成本，非主线）。`generate_near_json.py` 的 `[[mix_path,n]]`
  哨兵不是合法近场源（Dataset 会跳过，K=0），不能拿来冒充复现。
- **嘴部数据已提供**（96×96 RGB，uint8，T 帧），训练转换不需要 LRS3 原始视频；但**禁止转发/再分发**含人脸/声音的派生文件。

### 4.3 本机现状与获批后操作

- 当前：可读 README/文件清单/大小；访问申请已提交（2026-10-22，待批），下载数据仍返回 403（`GatedRepoError`）。
- 获批后：
  ```bash
  hf auth login                     # 用获批账号；或 export HF_TOKEN=...
  export https_proxy=http://127.0.0.1:17897 http_proxy=http://127.0.0.1:17897
  python tools/repro/fetch_data.py --dataset realssa-sim --shard tt-000000   # 先 1 个 shard 验证管线
  python tools/repro/fetch_data.py --dataset realssa-sim --pattern 'tt/*.tar' # test 全量（评测）
  python tools/repro/fetch_data.py --dataset realssa-sim --full               # 训练全量（≥600 GB 空余）
  ```
- 建议先用 1 个 `tt-*` shard 验证：tar 读取 → mix/target/mouth 形状 → 转仓库格式 → 2 s 窗口前向 → SI-SDRi 流程。

### 4.4 转换与评测要点

- tar 布局**不能**直接喂当前 folder-based loader；需转换为 `<DATA>/{tr,cv,tt}/{mix,target,mouths}/` 并在报告写清转换规则
  （`T_visual = min(mouth_frames, floor(audio_samples/640))`；`640`=16 kHz×40 ms=音频 STFT hop）。
- **通道策略必须做实验并报告**：RealSSA-Sim 是 4 通道；README/card 未声明论文用哪个通道/如何下混。
  建议至少准备 `ch0` 与 `mean`（4 通道平均）两版，评测时按 card 要求报告 “channel selection”。
- 评测协议：`SI-SNRi / SDRi = 估计 vs 同通道、对齐时长的 target`；test split；整句 vs 2 s 窗口口径要在报告注明。
- near 缺失对训练的影响：见 §4.2 与 §9；**不要**把对比损失当已复现。

---

## 5. RealSSA-Real（gated manual，真实场景评测）

### 5.1 URL / 状态 / 大小

| 项 | 值 |
|---|---|
| HF 数据集 | <https://huggingface.co/datasets/JusperLee/RealSSA-Real> |
| gated | **manual**（人工审批）；申请已提交（2026-10-22）**待批**，当前仍 **403** |
| 许可证 | **没有单独的通用数据集 license**；按 card 的 “Access, consent and permitted use” 研究用途条款（个人数据、参与者授权） |
| 规模 | 422 个文件，**12.84 GB**（其中 `raw/` 11.38 GB 原始视频/参考录音，`mouth/` 0.74 GB，`audio/` 0.60 GB 等） |
| 内容 | 两个 view：`inference`（54 个物理段/40 个逻辑录音/10 speakers/8 scenes/83.99 min）与 `original_recordings`（34 个逻辑录音/71.12 min）；无 train/dev split（只有 test） |
| 申请表单字段（card 定义） | Full name / Affiliation / Country / Intended research use + 3 个 checkbox：仅用于 card 允许的研究、不识别/联系参与者、不转发 |
| 联系 | <thucs.kaili@gmail.com> |

### 5.2 目录与格式（card 原文要点）

```text
audio/inference/            54 个 native stereo PCM16 WAV, 16 kHz（loader 按需取 channel mean / ch0 / ch1）
audio/original_recordings/  34 个 full original-extraction mono float32 WAV, 16 kHz
faces/                      54 个 target-face MP4，224×224，25 fps
mouth/                      54 个 NPZ：data = uint8 [T,96,96]
landmarks/                  54 个 NPZ：data = float32 [T,68,2]
references/                 10 条独立 enrollment 录音（mono PCM16）
raw/                        videos(44)/transcripts/references/reference_texts 等原始资产
metadata/                   joins/provenance/audit；metadata.jsonl/views/original/metadata.jsonl
scripts/                    load_inference.py 等 loader
```
- 官方 loader：`load_sample(root, index, align_tail=True, channel='mean')` → `audio float32 16k`、`mouth uint8 [T,96,96]`、
  `landmarks`、`speaker_reference`；`T=min(mouth frames, landmark frames, face frames, floor(audio_samples/640))`，
  最多丢弃 ≤70 ms 音频；`align_tail=False` 保留原生长度（见 card “Time alignment and channel policy”）。
- 评估协议（card）：**WER**（把同一逻辑录音的 segment 预测按 `part_order` 拼起来，再与整段 GT 比；40 条 GT 是
  whole-logical-recording，不是 54 条独立句）；**speaker similarity**（用 10 条独立 enrollment 参考）；
  DNSMOS 可另算。**没有干净参考波形 → 不能算 SI-SDR**（card 明确 “clean-target SI-SDR is not supported”）。
- 禁止：转发/再分发含人脸/声音/派生文件；识别参与者；公开论文/网页展示参与者面部/声音需书面许可；用完删除。

### 5.3 本机现状与获批后操作

- 当前：metadata/card 可读；访问申请已提交（2026-10-22，待批），数据文件仍 403。
- 获批后：
  ```bash
  hf auth login
  export https_proxy=http://127.0.0.1:17897 http_proxy=http://127.0.0.1:17897
  python tools/repro/fetch_data.py --dataset realssa-real --full     # 12.84 GB，需完整 snapshot（含 face/mouth NPZ）
  # 然后按 card：python scripts/load_inference.py /path/to/RealSSA-Real --index 0 --align-tail
  # Falcon 推理：studio/falcon_runtime + infer_file.extract（card 给了可运行示例）
  ```
- 报告必须写：channel（mean/0/1）、align_tail、DNSMOS 模型版本/跑法；不得声称 SI-SDR。

---

## 6. LRS3-2Mix / VoxCeleb2-2Mix（HF 仓库无数据文件）

| 数据集 | HF 仓库 | 状态 | 文件清单 | 数据卡要点 |
|---|---|---|---|---|
| LRS3-2Mix | `JusperLee/LRS3-2Mix` | 公开 | **仅 `.gitattributes` + `README.md`（622 B）** | LRS3 两人混合；28h train / 3h val / 1.5h test；较 LRS2 干净 |
| VoxCeleb2-2Mix | `JusperLee/VoxCeleb2-2Mix` | 公开 | **仅 `.gitattributes` + `README.md`（756 B）** | VoxCeleb2 Dev 取 5% 造 train/val；56h / 3h / 1.5h；噪声/混响更复杂 |

**结论**：这两列**当前无法从 HF 获取数据**（下载 `snapshot_download` 只会得到 README）。可行路径：
1. 联系作者/数据集所有者索取（不在本仓库可控范围）；
2. 用原始 LRS3 / VoxCeleb2 语料按卡片描述自建（随机两人、SNR −5~5 dB、按相同 split 规模）——但**混合随机种子/说话人对未知**，
   只能做“同分布复现”，不能宣称与论文测试集一致；
3. 原始语料本身也要申请：LRS3 <https://www.robots.ox.ac.uk/~vgg/data/lip_reading/>（TED，研究用途；深链
   `lrs3.html` 实测 404，改用该入口页；
   LRS3 terms），VoxCeleb2 <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>（YouTube 条款，研究用途）。
   LRS2 <https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html>（BBC，需签署研究用途协议）。

---

## 7. 跨数据集：通道 / 对齐 / 评测协议对照

| 数据集 | 原生音频 | 通道 | 采样率 | 视觉/嘴部 | 推荐评测协议 | 参考可用性 |
|---|---|---|---|---|---|---|
| LRS2-2Mix | tar 内 `audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}` | **mono** | 16 kHz（另有 8k） | ✅ `mouths` npz(50,96,96 uint8) + `faces` mp4 224×224 + `landmarks` | SI-SNRi/SDRi（每 clip 2 s，=整段；target=s1 或 s2） | ✅ mix/s1/s2（s1/s2 即 scaled target） |
| RealSSA-Sim | WebDataset tar 内 `.mix.wav/.target.wav` | **4ch PCM16** | 16 kHz | ✅ `mouths.pt` uint8 RGB [T,96,96,3] | SI-SNRi/SDRi（同通道、对齐时长；test split；报告 channel 策略） | ✅ target（同通道）；❌ near 源 |
| RealSSA-Real | `audio/inference/*.wav`（stereo）/ original（mono） | **stereo**（loader 可 mean/ch0/ch1） | 16 kHz | ✅ face MP4 + mouth NPZ + landmarks | DNSMOS + WER（整段拼接 GT）+ speaker-similarity；**无 SI-SDR** | ✅ enrollment / GT 文本；❌ clean target |
| LRS3-2Mix | —（无数据） | — | — | — | 若自建：同 LRS2 口径 | ❌ |
| VoxCeleb2-2Mix | —（无数据） | — | — | — | 若自建：同 LRS2 口径 | ❌ |

**与仓库 loader 的通道差异（README 已明确提示）**：`README.md:82`/`:85` 说明 RealSSA-Sim 是 **4-channel 16 kHz mixtures**、
RealSSA-Real 是 **stereo 16 kHz inputs**；`README.md:87` 明确 “多声道 shard 需要适配本仓库的 mono, folder-based training loader”。
因此复现时必须做**通道选择/下混**（如 `mean`、`ch0`、`ch1`），并在报告中声明；不要默认 `mono` 复现多通道数据集。
另：模型输入是 **2 s 窗口**（Studio/训练），而仓库 eval Dataset 是 **整句**（无 2 s 截断/补齐）；两者指标不可直接横向比较，
须在报告中同时写明 “model window” 与 “metric window”。

---

## 8. 批准前能做什么 / 不能做什么

| 现在（未获批） | 获批后 |
|---|---|
| ✅ LRS2-2Mix：下载+解压 → 完整 2 s AV（mix/s1/s2 + mouths + faces + landmarks）→ 转 Falcon 数据并跑通 LRS2-2Mix **AV** 评测/训练数据构建（无需 LRS2 原始视频） | ✅ RealSSA-Sim：1 shard 验证 → test 全量 → 训练全量；L2 评测 + L3 训练（需转换/负样本方案） |
| ✅ 读 RealSSA-Sim/Real 的 dataset card、文件清单、大小、loader 协议（HF 元数据公开） | ✅ RealSSA-Real：DNSMOS/WER/speaker-similarity；报告 channel/align 策略 |
| ✅ 用合成数据/伪 ckpt 跑通评测代码与指标实现（见 `tools/repro/make_synthetic_data.py` 等） | ✅ 与论文数字对齐（受通道、整句口径、SDR 实现 bug 影响，需同时修 F7/F8 等 P0/P1） |
| ✅ 修训练/评测代码 P0（Comet logger、conf.yml 类名、mono→0 维、PL 2.x 接口等，见 REPRODUCTION_PLAN §6） | ⚠️ 训练：near 源缺失 → 必须替代负样本或关对比损失，并显式标注“非官方配方” |
| ✅ 做 L1 推理复现（真实权重，已完成） | ❌ LRS3-2Mix / VoxCeleb2-2Mix：HF 无数据，仍需联系作者或自建 |
| ❌ 不能下载/使用 RealSSA-Sim/Real 数据文件（403）；不能提交他人名义申请 | ❌ 仍不能转发/再分发 RealSSA 与 LRS 派生数据 |

---

## 9. 许可与合规要点（必须随复现报告声明）

- **RealSSA-Sim**：CC BY-NC-ND 4.0（TED 内容）；仅非商业学术研究；禁止商业使用、转发/再分发（含 mouth crops/分离结果）、
  识别/跟踪/合成媒体。各噪声源另有条款（WHAM!/BBC/RemArc/AudioSet/VGGSound/FSD50K 等）。
- **RealSSA-Real**：无通用 dataset license；按 card 的研究用途+参与者授权条款；含可识别个人数据，禁止再识别/转发/公开媒体；
  访问是个人化的、可被撤销；用完删除。联系 `thucs.kaili@gmail.com`。
- **LRS2/LRS3**：BBC/TED 内容，需向 Oxford VGG 申请并遵守 LRS2/LRS3 研究条款；HF 2Mix 仓库的 `apache-2.0` 标签
  **不能覆盖**源数据条款。
- **VoxCeleb2**：YouTube 视频，遵守 VoxCeleb2 条款/YouTube 版权，研究用途。
- 公共模型/权重（Apache 2.0）**不等于**数据集授权：模型公开与数据 gated 是两件事（RealSSA card 明确说明）。

---

## 10. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `GatedRepoError 403` | 账号未获批（RealSSA-Sim/Real） | 按 §4.1/§5.1 申请；批准后 `hf auth login`；只读 `--dry-run` 可先确认状态 |
| `SSL_ERROR_SYSCALL` / `hf-mirror.com` 连接失败 | 本机 hf-mirror TLS 不可达 | 用 `--endpoint https://huggingface.co` + `--proxy http://127.0.0.1:17897`（或 export http(s)_proxy） |
| `RepositoryNotFoundError 404` | 把 **dataset** 当 **model** 查（`model_info` vs `dataset_info`），或仓库名错 | 用 `api.dataset_info(repo)` / `hf_hub_download(repo_type="dataset", ...)` |
| LRS2 下载中断 | 网络/代理抖动 | 重跑同一 `fetch_data.py` 命令，curl `-C -` 续传；日志看 `T6-data-fetch.log` |
| 磁盘不足 | RealSSA-Sim 全量 521.5 GB + LRS2 16.5 GB | 先备 ≥600 GB；先下 `tt/tt-000000.tar` 验证管线 |
| token 无效/过期 | token 文件旧 | `hf auth login` 用获批账号重新登录；不要提交他人 token |

---

## 11. 证据与更新记录

- 生成时间：2026-10-21（Phase 1 · T6）。
- 依据：HF API 实测（`HfApi().dataset_info` 文件/大小/gated）、`hf_hub_download(dry_run=True)` 授权探测、
  `whoami()` 身份、LRS2-2Mix 流式下载与**完整 `tar tzf` 清单**（249,670 条，见
  `docs/verification/phase1/T6-lrs2-structure.txt`）、各数据集 README（原文已复制到仓库内
  `docs/verification/phase1/dataset-cards/`，关键内容摘入本文）。
- 下载执行与 tar 完整结构分析见 `docs/verification/phase1/T6-report.md` 与 `docs/verification/phase1/T6-lrs2-structure.txt`；下载日志 `docs/verification/phase1/T6-data-fetch.log`；
  数据集卡原文与 tar 清单样本已复制到 `docs/verification/phase1/dataset-cards/` 与 `docs/verification/phase1/T6-lrs2-listing-sample.txt`（完整 `tar tzvf` 清单另存 `/root/data/lrs2_2mix/lrs2.tar.list`）（已完成：16,470,508,590 B，sha256 `ebdfdf31…767e`，与 HF x-linked-etag 一致）。
- 对照文档：`docs/REPRODUCTION_PLAN.md` §5（数据方案）/§11（可得性核实）；`AGENTS.md`（环境/网络约定）。
- 更新记录：
  - 2026-10-22（Phase 2 · task-25）：修正 §6.2 的 LRS3 深链（`lrs3.html` 实测 404 → 入口页）、§11 证据措辞
    （“部分 `tar tzf`” → 完整清单；“`/tmp/ds_cards/`” → 仓库内 `dataset-cards/` 副本）。未改动任何数据结论。
  - 2026-10-22（Phase 2 · task-25）：同步 RealSSA 访问申请状态（§0 汇总表 RealSSA-Real 行、§1 表、§4.2/§5.2
    gated 行、§4.3/§5.3 现状行）：Sim/Real 均为“申请已提交（2026-10-22，`nanless`/USTC/CN）待批，数据文件仍 403”，
    与 `REPRODUCTION_PLAN.md` Phase 2 状态、`REPRODUCTION_PHASE1_REPORT.md` §6 一致。数据集大小/许可/格式数字未变。
