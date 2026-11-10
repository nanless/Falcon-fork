> **2026-11-10 复核提示：** 历史数据更新：文件名增益差和源信号本身的能量共同决定实测能量比；发布包 s1/s2 波形已按混音缩放，near 不是未经增益的原始波形。K=1 时 β 不参与负项计算；K=0 时仍保留正项。有放回抽取 10,000 次，期望覆盖 40,000 个方向条目的约 22.12%；8000 Hz 时固定除以 16000 导致帧起点为正确值的一半。17,280 个样本在 16 kHz 是 1.08 秒。数据构建命令可能写入现有目录，执行前应核对输出位置。详见[数据集说明](../20261110-datasets.md)、[数据与训练专题](../20261110-data-and-training.md)和[审查记录](../../verification/deepdive-20261110/审查记录.md)。

# 06 数据管线与落盘数据

> 子代理 W1-06 · 2026-11-08 · 目标仓库 `dev_a800_2gpus:/root/code/github_repos/Falcon-fork`（HEAD `6c1d5b7`）
> 范围：`look2hear/datas/**`、`DataPreProcess/**`、`tools/repro/**` 数据构建脚本、`/root/data/**` 目录与清单。
> 仓库与数据目录全程只读（仅 `ls`/`du`/读 json/读少数 wav 头与 `.pt` 头）。所有 python 均为 `/root/miniforge3/envs/falcon/bin/python`。
> 论文 arXiv:2610.32749 已于 2026-10-23 公开。MACs 与 lookahead 的统一口径见 00 §4、§11，本文件不重复。

---

## 0. 结论速览

1. **这个仓库做的是"目标说话人提取（TSE）"，不是 2 说话人分离**（任务定义与模型侧证据见 01 §2）：`n_src: 1`，`__getitem__` 只返回 **1 路 target**，没有 `s1/s2` 两个源字段。"2mix" 只描述数据层的两条源混合（`configs/LRS2_2Mix.yml:9`、`look2hear/datas/avspeech_dataset_near.py:282-284`）。
2. **`mix` 是离线混好的，dataset 里不做任何混音**。混音发生在 LRS2-2Mix 发布包里：`mix ≈ s1 + s2（发布波形已含增益）`，`g1∈[0,5] dB` 且 `g2 = -g1` ⇒ **名义增益差落在 [0,10] dB，中位数 5.025 dB**（实测 20000 条 tr stem，§4.4）。dataset 只按随机起点切窗。
3. **训练期"随机裁剪"在 LRS2 上是假的**：LRS2 每条正好 2.000 s / 32000 点，`segment=2.0` ⇒ `rand_start = randint(0, 0) = 0`（`avspeech_dataset_near.py:156`）。音频窗口恒定，只有视频侧的 `RandomCrop(88)` 与 `HorizontalFlip(0.5)` 还在随机（`transform.py:141-152`）。
4. **视频侧读的是预存 tensor，不是文件**：`mouths/<target_utt_id>.pt`，uint8 `[T,96,96,3]`，LRS2 恒 `T=50`；`torch.load(...).permute(0,3,1,2)` → Grayscale → `[T,96,96]` → `RandomCrop(88,88)` → `[50,88,88]`，**dtype 被 numpy 提升成 float64**（实测 §4.1）。
5. **音视频对齐是"帧 k 覆盖 640 点"的硬约定**，但代码把采样率**写死 16000**：`mouth_start = int((rand_start / 16000) * 25)`（`avspeech_dataset_near.py:168`、`avspeech_dataset.py:108`）。若 `sample_rate≠16000` 直接错位（当前所有 config 都是 16000，未触发）。
6. **没有 pad，只有 slice**。长度不够就短一截，靠 `collate` 的 `torch.stack` 隐式要求同 batch 等长 ⇒ 只有 `T` 恒定（LRS2）时才敢 `batch_size>1`；val/test 一律 `batch_size=1`。
7. **near 的语义是"同一 mixture 里的已缩放干扰源波形"**，不是 near-field 泄漏：`/root/data/falcon_lrs2_2mix/<split>/near.json` 100% 指向 `lrs2_2mix/lrs2_rebuild/audio/wav16k/min/<split>/s2/*.wav`，K 恒 = 1，fallback = 0（实测 §4.2）。RealSSA-Sim 侧 **100% fallback（自引用哨兵 ⇒ 被 dataset 跳过 ⇒ K=0）**，NASS 对比损失在 K=0 时仍保留正项。
8. **`--copy-wavs` 的 P0（target/near 同名塌缩）已在 `2062528` 修掉**，修在 `build_lrs2_paper_data.py:98-117`（`.s1.wav`/`.s2.wav` 分名）+ `validate_build()` 闸门（`:182-260`，失败 `exit 2` `:303-306`）。
9. **落盘数字有两处"账面"陷阱**：`falcon_lrs2_2mix` 与 `_paper` 的 `mouths/` 是**同一批 inode 的硬链接**，`falcon_realssa_sim` 与 `_ch0` 的 `mouths/` 同理。单目录 `du` 会把 4 个目录算成 110 G，**一次 `du` 去重后真实并集 = 62 G**（实测 §4.3）。
10. **`falcon_lrs2_2mix/conversion_stats.json` 与磁盘状态不一致**：stats 写 `near_mode: "fallback"`（2026-10-21），而 `near.json` 是 2026-10-22 被原地改成 s2 的。仓库里已有 `docs/verification/phase2/P2C-near-mode-provenance.md` 承认并解释了这个漂移，我独立复核成立。
11. **`check_avspeech_dataset.py` 不是"不变量检查器"**：它是 `avspeech_dataset.py` 的整份拷贝 + 一个 `__main__` **dump 脚本**（写 10 条样本到 `mixture/`/`sources/`/`sources_mouths/`）。真正的不变量闸门是 `tools/repro/audit_data.py`。
12. **eval 集高度复用**：cv 5000 条 mixture 只来自 **476 条 utterance / 68 个说话人**（10.5× 复用），tt 3000 条来自 267 条 / 144 人（11.2×）；tr 是 1.07×。所以 val/test 的方差不能用"独立同分布"口径去读（实测 §4.7）。
13. **吞吐实测**：`num_workers=8, bs=4`，paper-tr（mouth 1.38 MB/条）稳态 **46.9 样本/s ≈ 11.7 it/s**；8 卡 global batch 32 时数据侧只有 ~1.5 step/s 的上限（§4.6）。

---

## 1. 事实基线

### 1.1 代码文件与规模（实测 `wc -l`）

| 文件 | 行数 | 角色 |
|---|---|---|
| `look2hear/datas/avspeech_dataset.py` | 334 | `AVSpeechTrainDataset`（4 元组，无 near）+ `AVSpeechEvalDataset`（5 元组，带 `n_near`）+ `AVSpeechDataModule` |
| `look2hear/datas/avspeech_dataset_near.py` | 337 | `collate_with_near` + `AVSpeechTrainDatasetWithNear`（5 元组）+ `AVSpeechDataModuleWithNear` ← **训练真正用的** |
| `look2hear/datas/check_avspeech_dataset.py` | 349 | 上述两类的**拷贝** + dump `__main__`；**不是**校验器 |
| `look2hear/datas/transform.py` | 187 | 视频预处理流水线（Normalize/RandomCrop/HFlip/CenterCrop） |
| `look2hear/datas/__init__.py` | 5 | 只导出 `AVSpeechDataModule`（非 near 版！） |
| `DataPreProcess/process_lrs23.py` | 119 | 文件夹 → `mix.json`/`target.json`；**内含一份** `generate_near_json` |
| `DataPreProcess/generate_near_json.py` | 113 | **独立第二份** `near.json` 生成器（读 `_meta.json` 的 `components[type=="near"]`） |
| `tools/repro/lrs2_2mix_to_falcon.py` | 689 | LRS2-2Mix → Falcon 布局（`--target`/`--near-mode`/`--copy-wavs` 无） |
| `tools/repro/phase2/build_lrs2_paper_data.py` | 311 | **论文配方**：双向 target + 构建后闸门（P0 修复处） |
| `tools/repro/realssa_to_falcon.py` | 580 | RealSSA-Sim WebDataset TAR → 文件夹布局（`--channel`/`--align`/`--near-dir`） |
| `tools/repro/make_synthetic_data.py` | 355 | 合成冒烟数据；`build_indexes()`(`:93-134`) 复用 `DataPreProcess` 的真代码 |
| `tools/repro/audit_data.py` | 205 | **真正的数据契约闸门**（JSON 等长 / K 分布 / fallback / 采样文件校验） |
| `tools/repro/fetch_data.py` | 395 | HF 下载（`JusperLee/LRS2-2Mix` 公开；RealSSA-* gated） |

### 1.2 被训练入口实际选中的路径

- `audio_train.py:25` `from look2hear.datas.avspeech_dataset_near import AVSpeechDataModuleWithNear`
- `audio_train.py:296` `datamodule = AVSpeechDataModuleWithNear(**config["datamodule"]["data_config"])`
- `audio_train.py:257-266` `set_seed()` → `pl.seed_everything(seed, workers=True)`
- `audio_visual_test.py:65,372` 评测直接实例化 `AVSpeechEvalDataset(json_dir=test_dir, ...)`

结论：`avspeech_dataset.py` 里的 `AVSpeechTrainDataset` / `AVSpeechDataModule`（4 元组）在"训练/评测"两条主路径上**都没被用到**，属于上游遗留死路（见 §5.9）。

### 1.3 数据根（只读实测）

```
/root/data/                      277G（du -h --max-depth=1 的"归因"总和，含硬链接去重）
├── lrs2_2mix/                    17G   LRS2-2Mix 发布包解压（lrs2_rebuild/）
├── falcon_lrs2_2mix/             24G   tr20k+cv5k+tt3k，target=s1，near=s2
├── falcon_lrs2_2mix_paper/       26G   双向 target，40k/10k/6k，只有 mouths + json
├── RealSSA-Sim/                  38G   只有 tt/ 19 个 tar（发布包共 240 shard）
├── falcon_realssa_sim/           30G   tt 9169，channel=mean
├── falcon_realssa_sim_ch0/       30G   tt 9169，channel=0
├── RealSSA-Real/                 12G   真实场景 AV（无 SI-SDR 参考）
├── falcon_eval/                  53K   3 个评测输出目录（DNSMOS csv + run_metadata）
├── lists/                       113G   （不在本轮范围）
├── rir/                          36G   （不在本轮范围）
└── pretrained_models/ 1.4G / models 1.5K / speech 26K / train_data 1.0K
```

---

## 2. 机制拆解

### 2.1 三层对象与数据流

```
mix.json [[mix_wav, n_samples], ...]
target.json [[target_wav, mouth_pt, n_samples], ...]      ← 注意 3 元组，不是 s1/s2 两个列表
near.json [[[near_wav, n_samples], ...K 条], ...]         ← 与上面逐条对齐
        │
        ▼
AVSpeechTrainDatasetWithNear.__getitem__(idx)  ← idx 完全不使用
        │  (mixture[T], target[T], mouth[F,88,88], filename, near_audios list)
        ▼
collate_with_near → (mixture[B,T], target[B,T], mouth[B,F,88,88], list[str], list[list[Tensor]])
        ▼
AudioVisualLightningModuleNASSv2.training_step  (av_litmodule_nass_v2.py:73-74)
        │  mouth → video_model（冻结）→ mouth_emb
        │  audio_model(mixtures, mouth_emb) → est_sources[B,1,T], sep_audio_embs
        ▼
   pit_loss (PITLossWrapper, n_src=1 → 1×1 无置换) + weight·NASS 对比损失(query=est, pos=target, neg=near)
```

### 2.2 `__getitem__` 逐行（以 near 版为准，`avspeech_dataset_near.py:149-222`）

| 行 | 代码 | 含义 / 影响 |
|---|---|---|
| `:154` | `mix_index = random.randint(0, len(self.mix)-1)` | **忽略 `idx`**。`shuffle=True` 形同虚设，等价于"每条都从全表有放回抽样" |
| `:156` | `rand_start = random.randint(0, self.mix[mix_index][1] - self.seg_len)` | 训练随机起点；LRS2 `n=32000=seg_len` ⇒ 恒 0 |
| `:158-163` | `sf.read(mix, start=rand_start, stop=rand_start+seg_len, dtype="float32")` | 只读 2 s 窗，不整段读 |
| `:164-166` | `source, _ = sf.read(target, ...)` 同窗口 | target 与 mix **共用同一 `rand_start`**，天然同步 |
| `:168` | `mouth_start = int((rand_start / 16000) * 25)` | 16000 **写死**；`rand_start/640` 向下取整 |
| `:169-175` | `torch.load(mouth_pt, weights_only=True).permute(0,3,1,2)` → `Grayscale(1)` → `.squeeze(1).numpy()` → `lipreading_preprocessing_func(train)` → `[mouth_start: fps_len+mouth_start]` | 预存 tensor → 灰度 → 裁剪 88×88 → 取 50 帧 |
| `:177-178` | `torch.from_numpy(x).mean(-1) if x.ndim>1 else torch.from_numpy(x)` | F5 修复：单声道不能对时间轴求均值（否则塌成 0 维，`torch.stft` 崩）。修复点：`docs/REPRODUCTION_PLAN.md §6.1` |
| `:180-183` | `if self.normalize_audio:` 用 **mixture 的 std** 归一 target | 默认 `False`（所有 config 都是 false） |
| `:186-219` | near 读取（见 §2.6） | |
| `:221-222` | `return mixture, source, source_mouth, filename, near_audios` | **5 元组** |

### 2.3 Dataset 契约表

训练集（`AVSpeechTrainDatasetWithNear`）：

| key | 形状 | dtype | 含义 |
|---|---|---|---|
| `mixture` | `[T]`，`T = segment·sr` = 32000（LRS2）/ 32000（RealSSA 任取 2 s 窗） | float32 | 已混好的 2 路混合波形（**离线**生成，见 §2.4） |
| `source`（=target） | `[T]`，与 mixture 同窗 | float32 | **只有 1 路**目标说话人（视频里那个人） |
| `source_mouth` | `[F,88,88]`，`F = segment·25` = 50 | **float64** | 灰度唇部 ROI，`Normalize(0.421,0.165)` 后 |
| `filename` | `str` | — | `basename(mix_wav)`，只用于日志/落盘命名，**不参与 loss** |
| `near_audios` | `list[Tensor[T]]`，K 条 | float32 | 负样本波形（near 版独有） |

评测集（`AVSpeechEvalDataset`，`avspeech_dataset.py:219`）：`(mixture[1,T], source[1,T], mouth[1,F,88,88], filename, n_near:int)` —— `n_near` 从 `<mix_stem>_meta.json` 的 `components[type=="near"]` 数出来（`:208-217`），LRS2 没有 meta ⇒ 恒 0。

**关键事实：没有 `s1`/`s2` 两个返回字段。** 因此"PIT 的源顺序问题"在本仓库不存在：`n_src=1`，`PITLossWrapper` 的 `pw_mtx` 是 1×1 矩阵，置换群平凡（`configs/LRS2_2Mix.yml:4-8,109-118`）。s1/s2 的"顺序"只在**数据条目**层面存在：paper 配方把同一条 mixture 拆成 `(target=s1, mouth=s1, near=s2)` 与 `(target=s2, mouth=s2, near=s1)` 两条（`build_lrs2_paper_data.py:118-122`），从而把"谁更响"这条捷径打断。

### 2.4 `mix` 是怎么混的：离线，且 SNR 有硬约束

发布包里 `create_mix/create_wav_2speakers.m:73-94`：

```
inwav1_snr = C{2}(i);  inwav2_snr = C{4}(i);
mix_name = [wav1, '_', snr1, '_', wav2, '_', snr2]
weight_1 = 10^(inwav1_snr/20);  weight_2 = 10^(inwav2_snr/20)
```

文件名 stem 即配方：`<s1_id>_<g1>_<s2_id>_<g2>`。实测 20000 条 tr stem（§4.4）：`g1∈[0.000,5.000]`，`g2∈[-5.000,-0.000]`，且**逐条 `g1 == -g2`**（`True`）。于是：

```
名义增益差 = g1 - g2 = 2·g1 ∈ [0, 10] dB     （中位 5.025，均值 5.011）
mix ≈ s1 + s2（发布波形已含增益）                （PCM16 舍入，实测 max|mix-(s1+s2)| = 3.05e-5 = 1 LSB）
```

**用户问题里的"SNR 范围/随机种子"答案**：SNR 在**发布包生成时**就已固定并编码进文件名，dataset 侧**没有任何 SNR 采样、没有种子、没有在线混音**。dataset 里唯一的随机性是"哪条样本 + 哪个起点 + 视频 crop/flip"。

副作用：文件名中 s1 的名义增益高于 s2；实测源波形能量比还受源信号影响。不做双向 target 的话，模型可以只学"挑更响的那路"而不看嘴唇 —— 这正是 paper 配方要双向的原因（`build_lrs2_paper_data.py:4-8`）。

### 2.5 视频侧：预存 tensor，切片而非 pad

- **存的是什么**：LRS2 发布包 `lrs2_rebuild/mouths/<speaker>_<utt>.npz`，key=`data`，uint8 `[50,96,96]`；转换器 `np.repeat(mouth[:,:,None], 3, axis=-1)` 变成 `[50,96,96,3]` 存 `.pt`（`lrs2_2mix_to_falcon.py:212-228`）。实测：`pt (50,96,96,3) uint8`，**三个通道逐元素相同**，且 `pt[...,0] == npz['data']`（§4.5）。
- **为什么是 3 通道而不是 1**：为了复用 `torchvision.transforms.Grayscale(1)` 这条路（`avspeech_dataset_near.py:81`），灰度化后与单通道等价。
- **长度与音频不一致怎么办**：**只 slice，不 pad**（`:174`）。对齐约定是"帧 k 覆盖 `[640k, 640k+640)`"，因为 `mouth_start = rand_start//640`、`fps_len = segment·25`、`seg_len = segment·16000 = 640·fps_len` 三者自洽。
  - RealSSA 侧把这条约定**强制**为不变量：转换器取 `frames = min(mouth_frames, len(mix)//640, len(target)//640)` 并把音/视/目标一起截到 `frames`（`realssa_to_falcon.py:354-366`）。实测 3 条样本 `n_samples/640` 全为整数、`mouth.T == n/640` 精确成立（§4.5）。
  - LRS2 侧靠 `MIN_MOUTH_PT_BYTES = 50·96·96·3` 的字节闸门把 `T≥50` 卡住（`lrs2_2mix_to_falcon.py:98`）；`T>50` 时取前 50 帧。
  - **失败模式**：若某条 `T<50`，切片得到 `[T',88,88]`（`T'<50`），`collate_with_near` 的 `torch.stack` 会直接抛 shape 不匹配。当前数据未触发（tr/cv/tt 的 mouth 全 50 帧）。
- **训练 vs 验证的裁剪策略不同**：
  - 训练：`RandomCrop(88)` + `HorizontalFlip(0.5)`（`transform.py:141-146`），音频窗口 `rand_start` 随机。
  - 验证/测试：`CenterCrop(88)`，无 flip（`transform.py:147-153`），音频**整段**（`rand_start=0, stop=None`，`avspeech_dataset.py:184-186`），mouth **不切片**（`:193-197`，返回全 `T` 帧）。
  - ⇒ 验证时 mouth 长度随样本变（RealSSA 145/115/93…），但 `batch_size=1` 所以不爆。

### 2.6 near 侧：语义、K、fallback、消费

**(a) 语义**。`near.json` 的 schema（`generate_near_json.py:1-20`、`avspeech_dataset_near.py:50-58`）：

```
[ [[near_path_1, n_samples_1], [near_path_2, n_samples_2], ...],   # 第 0 条 mixture 的负样本
  [[near_path_1, n_samples_1]],                                    # 第 1 条
  ... ]
```

与 `mix.json` **逐条、同序**对齐。三种可能的填充方式（`lrs2_2mix_to_falcon.py:33-50`）：

| `--near-mode` | near 指向 | K | 语义 |
|---|---|---|---|
| `fallback`（转换器默认） | `mix` 自身 | 1（自引用哨兵） | dataset 静默跳过 ⇒ **有效 K=0** ⇒ NASS 只有正项 |
| `other-speaker` | 同一 mixture 的**另一条源** | 1 恒定 | "远场、干净、但确实在混音里"的干扰源；**不是 near-field 泄漏** |
| `raw_audio` | 从 `raw_audio/<split>/` 抽的**不同说话人** 2 s 干净片段 | `--near-per-sample` | 近似替代（无 RIR、无响度匹配），只配做"非官方负样本"实验 |

磁盘实况（§4.2 实测）：`falcon_lrs2_2mix` 与 `_paper` 全部 `other-speaker`（K=1，fallback=0）；`falcon_realssa_sim[_ch0]` 全部 `fallback`（100% 自引用）。

**(b) Dataset 里怎么用**（`avspeech_dataset_near.py:186-219`）：

```python
for near_path, n_samples in self.near[mix_index]:
    if near_path == mix_path:            # :190  哨兵：跳过，避免把 mix 当负样本
        continue
    if n_samples < self.seg_len:         # :193  太短整条丢
        continue
    near_start = min(rand_start, n_samples - self.seg_len)   # :198  夹到合法区间
    near_stop = near_start + self.seg_len
    near_wav, _ = sf.read(near_path, start=near_start, stop=near_stop, dtype="float32")
```

三条"静默降级"路径：自引用跳过、过短跳过、读失败只 `warnings.warn`（`:206-217`，注释里说明用 `warnings` 是因为 DataLoader worker 里的 `print` 会被吞）。**这三种情况都不会报错，只会让 K 变小**——这是最容易被忽略的一点。

**(c) 训练里怎么变成 NASS 的正负对**（`av_litmodule_nass_v2.py:73-133`）：

- query = `pool_spk_embed(sep_audio_embs[:, 0])`（**带梯度**，来自估计源）
- positive = `pool_spk_embed(encode_stft(targets[:,0]))`（`no_grad`）
- negatives = 每个 near 波形 `_trim_or_pad(·, seg_len)`（`:60-66`，短补零长截断）后过同一个 encoder（`no_grad`）
- `cl_loss = NASSContrastiveLoss(embed_query, embed_pos, embed_negs_list, beta)`；`total = pit_loss + weight·cl_loss`（`:134-136`）
- β 线性退火：`beta_init → beta_final`，`progress = min(current_epoch/beta_anneal_epochs, 1)`（`:68-72`）。**若 K 恒 = 1 且 β 从 0 起退火，β 不参与单负样本分支，调整 β 不改变该分支的损失**（STATUS 已指出，我复核认可）。

**(d) K 的分桶日志**：train 侧按 `min(len(near),4)` 记 `train_n_samples_{0..4}near`（`:112-132`）；val 侧按 `n_near` 记 `val_pit_sisnr_{0..4}near`（`av_litmodule.py:104-121,150-159`）。因为 eval 的 `n_near` 只从 `_meta.json` 读，而 LRS2 没有 meta、RealSSA 的 meta 里没有 `type=="near"` 组件 ⇒ **所有 val 都落 bucket 0**，这些指标实际只有一个桶有值。

### 2.7 collate 与 batch 构造

`collate_with_near`（`avspeech_dataset_near.py:23-44`）：

```python
mixtures, sources, mouths, filenames, near_list = zip(*batch)
mixtures = torch.stack(mixtures)                       # [B,T]
sources  = torch.stack(sources)                        # [B,T]
mouths   = torch.stack([m if tensor else from_numpy])   # [B,F,88,88]
return mixtures, sources, mouths, list(filenames), list(near_list)
```

- **变长处理**：`mixture/target/mouth` 要等长（`stack` 的硬要求）；`near_audios` 是 `list[list[Tensor]]`，**不进 stack**，由 `training_step` 逐样本堆（`:123-127`）再 `_trim_or_pad` 到 `seg_len`。⇒ 这是"K 可变 + B 可变长"能被支持的唯一原因。
- **DataLoader 参数**（`avspeech_dataset_near.py:298-326`）：train `bs=4, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collate_with_near`；val/test `bs=1, shuffle=False, drop_last=True`。
- **一个 epoch 的长度**：`__len__ = train_samples`（默认/配置 10000，`:147`），与数据集实际条数（20k / 40k）**解耦**。paper 配置 40000 条 / 10000 采样 ⇒ 每个 epoch 只摸到 25% 的条目，且**有放回**。1000 epoch ⇒ 10 M 样本 ≈ 40k 全量的 250 遍。
- **`drop_last=True` + `bs=1`**：val/test 不丢样本（N 整除 1），train 2500 iter/epoch（10000/4）严格整除，也不丢。
- **随机性来源链**：`audio_train.py:265 pl.seed_everything(42, workers=True)` → `lightning_fabric/utilities/seed.py:85-110 pl_worker_init_function` 用 `torch.initial_seed()-worker_id` 派生 base_seed，再分别 `torch.manual_seed / random.seed / np.random.seed`。**注意 `persistent_workers=True` ⇒ `worker_init_fn` 只跑一次**，worker 内 `random` 状态跨 epoch 连续推进（所以不同 epoch 采样不同，但不可从"第 n 个 epoch"精确复算）。config 里 `training.seed: 42` 只在 `Paper_v2` 有（`configs/LRS2_2Mix_Paper_v2.yml` 头注释）。

### 2.8 数据构建全链路

#### (A) LRS2-2Mix：`lrs2.tar.gz` → `lrs2_rebuild` → Falcon 布局

**来源**：`JusperLee/LRS2-2Mix`（公开，单文件 `lrs2.tar.gz` 16,470,508,590 B，sha256 `ebdfdf31335f288cc2e4b1690a0cc2f260e1826a68a020723ad4ce59a033767e`；`fetch_data.py:50-60`、`docs/DATA_ACCESS_GUIDE.md:23`）。**不需要原始 LRS2 视频**，包里已带 `mix/s1/s2`（16k+8k）、`mouths` npz、`faces` mp4、`landmarks` 与混音脚本。

**解压后结构**（实测 `ls /root/data/lrs2_2mix/lrs2_rebuild`）：

```
audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}/<stem>.wav   2.000 s 16 kHz mono PCM16
audio/wav16k/min/tr/scaling.mat
mouths/<speaker>_<utt>.npz                            uint8 [50,96,96]
raw_audio/{train,val,test}/<speaker>_<utt>.wav        raw train 19649 条
faces/ landmark/ load_npz/ create_mix/*.scp,*.m       {train,val,test,retrain,redev,dev}.txt
```

**Step 1 — 生成训练布局（单 target）**：

```bash
cd /root/code/github_repos/Falcon-fork
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/lrs2_2mix_to_falcon.py \
  --src /root/data/lrs2_2mix/lrs2_rebuild \
  --out /root/data/falcon_lrs2_2mix \
  --splits tr,cv,tt --target s1 --near-mode fallback --jobs 16 --validate-samples 200
# 产物：<out>/<split>/{mix,target,mouths}/ + mix.json/target.json/near.json + conversion_stats.json
# 实测记录：conversion_stats.json elapsed_seconds=237.9，generated_at 2026-10-21T19:23:28Z
```

（`--near-mode` 之后被独立改成 `other-speaker` 重建 `near.json`，见 §6.1；对应命令见 `P2C-near-mode-provenance.md §3.1`。）

**Step 2 — 论文配方（双向 target）**：

```bash
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/phase2/build_lrs2_paper_data.py \
  --src /root/data/lrs2_2mix/lrs2_rebuild --out /root/data/falcon_lrs2_2mix_paper \
  --splits tr,cv,tt --jobs 12            # 不加 --copy-wavs ⇒ JSON 直接引用源 wav，不复制音频
# 产物：40k/10k/6k 条 + mouths/<utt>.pt + conversion_stats.json（含 gate 结果）
```

**参数语义表**：

| 参数 | 作用 | 备注 |
|---|---|---|
| `--target {s1,s2}` | 选哪条源当 target（同时决定 mouth 用谁的） | 默认 `s1`；`--target s2` 会把 near 换成 `s1`（`:316-323`） |
| `--near-mode {fallback,raw_audio,other-speaker}` | 见 §2.6 | 默认 `fallback` |
| `--near-per-sample` / `--seed` | 只对 `raw_audio` 生效 | 用 `md5(f"{seed}:{stem}")` 做确定性抽样（`:259-278`） |
| `--copy-wavs`（仅 paper 脚本） | 把音频复制进输出目录（自包含） | **P0 就出在这个分支**，见下 |
| `--jobs` | 线程/进程数 | paper 用 `ThreadPoolExecutor`（mouth 转换），lrs2 转换器用进程池 |
| `--limit` / `--expect-entries` | 小样本 & 闸门期望条数 | `--expect-entries tr=4,cv=2` |

**采样率**：全链路 **16 kHz** 硬编码（`lrs2_2mix_to_falcon.py:92 SAMPLE_RATE=16000`、`MOUTH_H=MOUTH_W=96`、`FRAME=640`）。发布包另有 8 kHz 版本未使用。

**SNR**：见 §2.4，`[0,10] dB`，不参与转换参数。

**Step 3 — P0（`--copy-wavs` 同名塌缩）修在哪一行**：

- 旧实现（review A5-1 描述）：`s1p = s2p = target/<stem>.wav`，`if not d.exists(): copy` ⇒ 第二次复制被跳过 ⇒ 条目 A 的 `near` 变成 `target` 自身（自引用 ⇒ K=0），条目 B 的 `target` 与 `mouth` 来自**不同说话人**（静默错配）。证据：`docs/understanding/THREE_ROUND_REVIEW_20261030.md:25`。
- **修复**：`tools/repro/phase2/build_lrs2_paper_data.py:98-117`
  - `:102` `tgt_s1 = tgt_dir / f"{stem}.s1.wav"`、`:103` `tgt_s2 = ... ".s2.wav"`
  - `:104-105` `near_s1/near_s2 = near_dir / f"{stem}.s1.wav" | ".s2.wav"`
  - `:109-111` 五个 `(src, dst)` 逐对独立判断 `if not dst.exists(): copyfile` ⇒ 不再互相踩
  - `:114-115` `s1p,s2p = tgt_s1,tgt_s2`；`near_of_s1, near_of_s2 = near_s2, near_s1`
- **配套闸门**：`validate_build()`（`:182-260`）检查 ① 三份 JSON 等长 ② 条数 == `PAPER_ENTRIES`（`:48` = 40000/10000/6000）③ `near != target`（自引用即报错）④ 由 stem 配方与文件名/父目录交叉推断 target 说话人，要求 `mouth_id == target_id`（`:165-180,229-235`）⑤ `near n_samples == mix n_samples`；任何不符 **`exit 2`**（`:303-306`）。
- 修复 commit：`2062528`（`git log -S "near_of_s1"` 唯一命中）。

#### (B) RealSSA-Sim：240 个 WebDataset shard → 训练布局

**来源**：`JusperLee/RealSSA-Sim`（gated/manual，521.5 GB，tr 202 + cv 19 + tt 19 shard；`fetch_data.py:61-70`、`docs/DATA_ACCESS_GUIDE.md:24`；README 实测见 `/root/data/RealSSA-Sim/README.md`：`tr 100,826 / cv 9,133 / tt 9,169`，4 条训练样本缺 JSON/mouth 已被官方剔除）。

**每条样本 4 个成员**（`realssa_to_falcon.py:5-12`）：`<id>.mix.wav`（**4 声道** 16 kHz）、`<id>.target.wav`（4 声道）、`<id>.json`（scene/room/components(type,source,SIR,position)）、`<id>.mouths.pt`（uint8 `[T,96,96,3]`）。

**命令**：

```bash
# 单 shard / 全量 tt
PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/realssa_to_falcon.py \
  --src /root/data/RealSSA-Sim/tt --out /root/data/falcon_realssa_sim \
  --splits tt --channel mean            # 或 --channel 0 → /root/data/falcon_realssa_sim_ch0
# 流水线封装：SHARD=tt-000000 CHANNEL=0 LIMIT=5 bash tools/repro/phase2/realssa_sim_tt_pipeline.sh --execute
#             FULL=1 bash .../realssa_sim_tt_pipeline.sh --execute     # 19 shards
```

**参数语义**：

| 参数 | 作用 | 代码 |
|---|---|---|
| `--channel {mean,0,1,2,3}` | 4 声道降为单声道策略；**mix 与 target 用同一策略** | `:179-190`（`mean` 取 4 声道均值；`k` 取第 k 声道）；`--channel 0` 时 `:189` 校验声道数足够 |
| `--align/--no-align` | 默认 `frames = min(mouth_frames, len(mix)//640, len(target)//640)` 并三段同时截断 | `:354-366` |
| `--near-dir` | 外部 near 波形（`<id>.near*.wav`），否则写 meta 后仍走 fallback | `:230-238,547` |
| `--limit-shards/--limit-samples/--dry-run` | 限流 | `:449-470` |

**写出的额外文件**：`<split>/mix/<id>_meta.json`（`build_meta()` `:240-...`）。它把 RealSSA 的 provenance `components[].path` **保留为字符串但标注"不可下载"**，并显式 `provenance.note` 写"RealSSA-Sim does not ship near-speaker waveforms"。这也解释了为什么 `generate_near_json.py` 在这份数据上必然产出 fallback：`components[].path` 不指向本地文件 ⇒ `os.path.exists()` 全 False ⇒ `sample_nears` 为空 ⇒ 塞 `[[mix_path, n]]`。

**采样率**：16 kHz，`_read_multichannel()` 遇到非 16k **直接抛错不重采样**（`:166-172`）——这是好事（不静默改数据）。

#### (C) RealSSA-Real：不转换，不走 `look2hear.datas`

`/root/data/RealSSA-Real`（12 G）保持发布包原样：`audio/{inference,original_recordings}/<scene>/*.wav`、`mouth/<scene>/*.npz`、`faces/`、`landmarks/`、`metadata/*.jsonl`。评测由 `tools/repro/phase2/eval_realssa_real.py` 自己扫目录（`_segments_from_metadata` / `_segments_from_scan`，`:126-202`），**不经过 `mix.json/target.json` 契约**，因为它是 reference-free（DNSMOS P.835）。实测 manifest 行数：`raw_segments 44`、`inference_segments 54`、`recordings 40`、`original_recordings 34`。

---

## 3. 关键数据 / 参数表

### 3.1 落盘核对表（全部实测，2026-11-08）

| 数据集 | 路径 | 大小（单目录 du） | 句子数 tr/cv/tt | 有 mouth? | manifest | 与 STATUS.md 一致? |
|---|---|---|---|---|---|---|
| LRS2-2Mix 源包 | `/root/data/lrs2_2mix` | 17 G | 20000 / 5000 / 3000 | 有（`mouths/*.npz` 20392 个，`[50,96,96]`） | 无 json，用 `create_mix/*.scp` + 目录 | ✅（STATUS:24 写 33G，**偏大**，见 §6.2） |
| Falcon LRS2（单 target） | `/root/data/falcon_lrs2_2mix` | 24 G（其中 20 G 是与 paper 共享的硬链接） | 20000 / 5000 / 3000（共 28000） | 有，14773 / 476 / 267 个 `.pt` | `mix.json`/`target.json`/`near.json` + `conversion_stats.json` | ✅ 条数一致（"28k"） |
| Falcon LRS2（论文配方） | `/root/data/falcon_lrs2_2mix_paper` | 26 G（= mouths 25 G + json） | **40000 / 10000 / 6000**（20000/5000/3000 mixture × 双向） | 有，18691 / 476 / 267 | `mix.json`/`target.json`/`near.json` + `conversion_stats.json`；**无 `mix/`、`target/` 目录**（音频原地引用 `lrs2_rebuild`） | ✅ 条数一致（"40k/10k/6k"） |
| Falcon RealSSA-Sim（mean） | `/root/data/falcon_realssa_sim` | 30 G（`mouths` 28 G 与 ch0 共享） | — / — / 9169 | 有，9169（`T` 变长 25–171 帧） | 同上 + `mix/*_meta.json` ×9169 | ✅（"mean，30G，9169"） |
| Falcon RealSSA-Sim（ch0） | `/root/data/falcon_realssa_sim_ch0` | 30 G（`mouths` 与 mean 共享 → 实际独增 ~2.7 G） | — / — / 9169 | 有，9169 | 同上 | ✅ |
| RealSSA-Sim 发布包 | `/root/data/RealSSA-Sim` | 38 G | 仅 tt：**19/19 shard** 已下载（官方 tr 202 / cv 19 / tt 19） | 包内 `mouths.pt` | tar 内嵌，无外部 manifest | ⚠️ 只下了 tt（STATUS 未声明"仅 tt"这一点在数据行里） |
| RealSSA-Real 发布包 | `/root/data/RealSSA-Real` | 12 G | 54 条 inference 段 / 34 条 original_recording | `mouth/<scene>/*.npz` | `metadata/{inference_segments,raw_segments,recordings,original_recordings}.jsonl` | ✅ 12 G |

**硬链接实测**（决定"账面大小"）：

```
$ stat -c "%i %h %n" .../falcon_lrs2_2mix/tr/mouths/5535415699068794046_00004.pt
1345944711 2 .../falcon_lrs2_2mix/tr/mouths/5535415699068794046_00004.pt
$ find /root/data/falcon_lrs2_2mix_paper -inum 1345944711
/root/data/falcon_lrs2_2mix_paper/tr/mouths/5535415699068794046_00004.pt      ← 同一 inode
$ stat -c "%i %h" .../falcon_realssa_sim/tt/mouths/00001_00_10182.pt ...ch0/...
1345892819 2 / ... /1345892819 2                                                ← 同一 inode
```

⇒ 4 个 Falcon 目录**并集只占 62 G**（`du -sch` 一次去重），而"逐目录相加"是 110 G。

### 3.2 `near.json` K 分布（全量，非抽样）

| 数据集 | split | N | K=0 | K=1 | K≥2 | fallback（自引用哨兵） | 有效 K |
|---|---|---|---|---|---|---|---|
| falcon_lrs2_2mix | tr / cv / tt | 20000 / 5000 / 3000 | 0 | 100% | 0 | 0 (0.0%) | 1 |
| falcon_lrs2_2mix_paper | tr / cv / tt | 40000 / 10000 / 6000 | 0 | 100% | 0 | 0 (0.0%) | 1 |
| falcon_realssa_sim | tt | 9169 | 0 | 100% | 0 | **9169 (100%)** | **0** |
| falcon_realssa_sim_ch0 | tt | 9169 | 0 | 100% | 0 | **9169 (100%)** | **0** |

### 3.3 数据内容统计（实测）

| 项 | tr | cv | tt |
|---|---|---|---|
| mixture 数 | 20000 | 5000 | 3000 |
| unique 说话人（s1∪s2） | **2884** | **68** | **144** |
| unique utterance（s1∪s2） | 18691 | 476 | 267 |
| mixture / utterance | 1.07 | **10.50** | **11.24** |
| mouth 文件数（单 target 布局） | 14773 | 476 | 267 |
| mouth 文件数（双向布局） | 18691 | 476 | 267 |
| 每条长度 | 恒 32000 点 = 2.000 s | 同 | 同 |
| 格式 | 16 kHz / mono / PCM_16 | 同 | 同 |
| SNR(s1/s2) | [0,10] dB，中位 5.025 | 同（同一配方） | 同 |

RealSSA-Sim tt：9169 条，**变长** 17280–97280 点（1.08–6.08 s；500 条抽样里 640 点整数倍全覆盖），`mouth.T == n_samples/640` 精确。

### 3.4 config 里的 datamodule（6 个配置全表）

| config | train_dir | segment | bs | workers | use_near | train_samples |
|---|---|---|---|---|---|---|
| `Falcon.yml` | `--`（占位） | 2.0 | 4 | 8 | true | 默认 10000 |
| `LRS2_2Mix.yml` | `falcon_lrs2_2mix/tr` | 2.0 | 4 | 8 | true | 10000 |
| `LRS2_2Mix_Paper_2gpu.yml` | `_paper/tr` | 2.0 | 4 | 8 | true | 10000 |
| `LRS2_2Mix_Paper_2gpu_smoke.yml` | `_paper/tr`（cv/tt 指 `/root/tmp/paper_smoke`） | 2.0 | 4 | 8 | true | 2000 |
| `LRS2_2Mix_Paper_8gpu.yml` | `_paper/tr` | 2.0 | **5** | 8 | true | 10000 |
| `LRS2_2Mix_Paper_v2.yml` | `_paper/tr` | 2.0 | 4 | 8 | true | 10000 |

全部 `n_src: 1`、`sample_rate: 16000`、`normalize_audio: false`。**没有一个 config 用 `falcon_realssa_sim`**（RealSSA 只做评测，与 STATUS 的"RealSSA 从零训练未做"一致）。

---

## 4. 实测验证

> 全部脚本落在 `/tmp/w106dp/`（避免 `/tmp/*.py` 命名污染：本机 `/tmp` 里有别的进程留下的同名脚本，第一次跑撞上 `AttributeError: module 'profile' has no attribute 'run'`，是 `/tmp` 被当作 `sys.path[0]` 后的 stdlib 影子问题，不是仓库问题）。

### 4.1 T1 — 真跑一个 batch（`falcon_lrs2_2mix`，训练配置原样）

```python
dm = AVSpeechDataModuleWithNear(train_dir="/root/data/falcon_lrs2_2mix/tr", ..., sample_rate=16000,
        segment=2.0, batch_size=4, num_workers=0, train_samples=8, use_near=True)
dm.setup(); tr, va, te = dm.make_loader
```

```
[AVSpeechTrainDatasetWithNear] Loaded near.json with 20000 entries.
Drop 0 utts(0.00 h) from 20000 (shorter than 32000 samples)
[AVSpeechTrainDatasetWithNear] 20000/20000 samples have near-speaker entries.
[AVSpeechTrainDatasetWithNear] Near file access OK: /root/data/lrs2_2mix/lrs2_rebuild/audio/wav16k/min/tr/s2/6298351355700308615_00023_4.734_...wav
len(train_ds)= 8 len(val_ds)= 5000 len(tt_ds)= 3000
BATCH mixture (4, 32000) torch.float32 min=-0.8365 max=0.9000 mean=0.002816
BATCH target  (4, 32000) torch.float32
BATCH mouth   (4, 50, 88, 88) torch.float64 min=-2.5515 max=2.5346
BATCH fnames  ['6248200740571759847_00007_3.4392_...wav', ...]        # 4 条
BATCH n_near  [1, 1, 1, 1]  [(32000,), (32000,), (32000,), (32000,)]
ITEM len 5   mixture (32000,) target (32000,) mouth (50, 88, 88)  <fname>  K=1
VAL batch len 5 [(1, 32000), (1, 32000), (1, 50, 88, 88)] n_near= tensor([0])
```

要点：**mouth 是 float64**（`Normalize(0.0,255.0)` 在 float64 numpy 上做算术）；`n_near` 在 val 上是 0（LRS2 无 `_meta.json`）；`train_samples=8` 时 `len(train_ds)=8` 而底层有 20000 条。

### 4.2 T2 — near.json K 分布（全量 4+4 个 split）

```python
k = collections.Counter(len(x) for x in near)
fb = sum(1 for x,m in zip(near,mix) if any(e[0]==m[0] for e in x))
```

```
falcon_lrs2_2mix         tr: N= 20000 K={1: 20000} fallback=0 (0.0%)
falcon_lrs2_2mix         cv: N=  5000 K={1: 5000} fallback=0 (0.0%)
falcon_lrs2_2mix         tt: N=  3000 K={1: 3000} fallback=0 (0.0%)
falcon_lrs2_2mix_paper   tr: N= 40000 K={1: 40000} fallback=0 (0.0%)
falcon_lrs2_2mix_paper   cv: N= 10000 K={1: 10000} fallback=0 (0.0%)
falcon_lrs2_2mix_paper   tt: N=  6000 K={1: 6000} fallback=0 (0.0%)
falcon_realssa_sim       tt: N=  9169 K={1: 9169} fallback=9169 (100.0%)
falcon_realssa_sim_ch0   tt: N=  9169 K={1: 9169} fallback=9169 (100.0%)
```

**K 的分布是退化的（全 1）**，所以"K 的含义"只能用语义区分（自引用 vs 他人源），不能用 K 的数值区分——这正是 STATUS §5 那条"K=1 时 τ 退火不生效"的根源。

样本形态：

```
falcon_lrs2_2mix/tr        mix[0]    = [.../tr/mix/<stem>.wav, 32000]
                           target[0] = [.../tr/target/<stem>.wav, .../tr/mouths/5535415699068794046_00004.pt, 32000]
                           near[0]   = [[.../lrs2_rebuild/audio/wav16k/min/tr/s2/<stem>.wav, 32000]]
falcon_lrs2_2mix_paper/tr  mix[0]    = [.../lrs2_rebuild/audio/wav16k/min/tr/mix/<stem>.wav, 32000]   ← 不复制
                           target[0] = [.../lrs2_rebuild/.../tr/s1/<stem>.wav, .../paper/tr/mouths/<s1id>.pt, 32000]
                           near[0]   = [[.../lrs2_rebuild/.../tr/s2/<stem>.wav, 32000]]
falcon_realssa_sim/tt      near[0]   = [[/root/data/falcon_realssa_sim/tt/mix/00001_00_10182.wav, 92800]]  ← 自引用
```

### 4.3 T3 — 条数与磁盘核对（`find`/`ls | wc -l`/`du`）

```
### falcon_lrs2_2mix        tr mix=20000  target=20000  mouths=14773
                            cv mix=5000   target=5000   mouths=476
                            tt mix=3000   target=3000   mouths=267
### falcon_realssa_sim/tt   mix=18338 (9169 wav + 9169 _meta.json)  target=9169  mouths=9169
### falcon_lrs2_2mix_paper  tr mouths=18691  cv mouths=476  tt mouths=267
$ du -h --max-depth=3 /root/data/falcon_lrs2_2mix   → tr/mouths 20G, tr 22G, 总 24G
$ du -h --max-depth=3 /root/data/falcon_lrs2_2mix_paper → tr/mouths 25G, 总 26G
$ du -sch <4 个 falcon 目录>  → 24G + 5.1G + 30G + 2.7G = 62G total
```

自洽校验（很有说服力）：**mouth 文件数 == unique target utterance 数**。单 target 布局 `14773/476/267` 精确等于 §3.3 里 `unique_s1_ids`；双向布局 `18691` 精确等于 `unique(s1∪s2)`。说明去重逻辑（`mouths/<target_id>.pt`）没漏没重。

### 4.4 T4 — SNR / gain 分布（20000 条 tr stem，regex 解析 + 波形验证）

```
N stems tr = 20000
gain1 (s1) min/med/max = 0.000/2.513/5.000
gain2 (s2) min/med/max = -5.000/-2.513/-0.000
snr=g1-g2 min/med/mean/max = 0.000/5.025/5.011/10.000
sign(g1) all positive? True   sign(g2) all negative? True   g1==-g2 all? True
tr/cv/tt: frames_dist(400 sampled)={32000:400}  sr=16000 ch=1 fmt=PCM_16
mix-(s1+s2): max|Δ| = 3.05176e-05 (×3 条)
```

⇒ 文件名中的名义增益差为 `[0,10] dB`；发布 s1/s2 波形已含增益，抽查的 mix 与二者之和相差不超过 PCM16 一个量化级。源波形实际能量比并非逐条等于名义增益差。

### 4.5 T5 — mouth `.pt` 内部结构与 RealSSA 帧对齐

```
pt (50, 96, 96, 3) torch.uint8
channels identical: True
npz keys ['data'] shape (50, 96, 96) uint8
npz == pt[...,0]: True

realssa tt:  mix 00001_00_10182.wav frames=92800 sr=16000 ch=1 | target frames=92800 | mouth (145,96,96,3) | n/640=145.000
             mix 00001_00_11054.wav frames=73600                                  | mouth (115,...)        | n/640=115.000
             mix 00020_01_50940.wav frames=59520                                  | mouth (93,...)         | n/640=93.000
             frames dist(200): min=17280 max=97280
```

⇒ ① LRS2 的 3 通道是**复制的灰度**，`Grayscale` 不丢信息；② RealSSA 的 `n_samples` 全为 640 的整数倍（转换器 `--align` 的强制结果），帧对齐精确。

### 4.6 T6 — DataLoader 吞吐（paper-tr，`num_workers=8, bs=4`）

```
batch shapes: ((4,32000),(4,32000),(4,50,88,88),[1,1,1,1])
first-iter 48 samples in 2.20s -> 21.9 samples/s (含 worker 启动)
steady     80 samples in 1.71s -> 46.9 samples/s; 11.7 it/s
```

⇒ 每条样本要读 64 KB×2 音频 + 1.38 MB mouth（`torch.load` 反序列化 + 灰度 + crop）。8 卡 bs4（global 32）时数据侧上限 ≈1.5 step/s；**paper 的 1000 epoch × 2500 iter 在数据侧就是个瓶颈**（[推断]，未做端到端测速）。

### 4.7 T7 — 说话人/utterance 复用统计（§3.3 的算法）

```
tr: mixtures=20000 unique_speakers=2884 unique_utterances=18691 avg_mix_per_utt=1.07
cv: mixtures=5000  unique_speakers=68   unique_utterances=476   avg_mix_per_utt=10.50
tt: mixtures=3000  unique_speakers=144  unique_utterances=267   avg_mix_per_utt=11.24
tr: mixtures per s1 utterance min/med/max = 1/1.0/2 ; utterances used once = 9546
```

`cv` 只有 68 个说话人、476 条 utterance；`tt` 144 人 / 267 条，且 **s1 池 == s2 池**（unique_s1 = unique_s2 = 476 / 267）⇒ cv/tt 是"同一小池内部交叉配对"。样本复用会产生相关性，置信区间需要按说话人或 utterance 依赖结构估计。

### 4.8 T8 — RealSSA `--channel` 语义验证（从原始 tar 反推）

```
shard members=2000 sample=00001_00_10182 sr=16000 shape=(92800, 4)
max|falcon_realssa_sim     - mean(4ch)| = 2.29e-05     ← 等于 PCM16 舍入
max|falcon_realssa_sim_ch0 - ch0|       = 0            ← 精确相等
max|mean - ch0| = 0.165 ; corr(mean, ch0) = 0.9692
```

⇒ `falcon_realssa_sim` = `--channel mean`，`falcon_realssa_sim_ch0` = `--channel 0`，与 STATUS 记载一致；两个目录**同一批 9169 个 id**，只差声道策略。

### 4.9 T9 — 官方闸门 `audit_data.py` 复跑

```
$ python tools/repro/audit_data.py /root/data/falcon_lrs2_2mix --sample 60
[tr] entries=20000 K=min1 max1 mean1.00 hist={'1': 20000} fallback=0 coverage=100.00% sampled=60
[cv] entries=5000  ...  [tt] entries=3000  ...
[audit] total_entries=28000 total_fallback=0 status=OK        exit=0

$ python tools/repro/audit_data.py /root/data/falcon_lrs2_2mix_paper --sample 60
[audit] total_entries=56000 total_fallback=0 status=OK        exit=0
```

（脚本对 RealSSA 的预期是 `exit 1`，文档自己写明 `audit_data.py:21`——因为 near 全 fallback。我没重复跑，靠 §4.2 的全量 K 统计已足够。）

### 4.10 T10 — paper 配方的配对结构验证

```
entries= 40000  unique mix paths= 20000
pairs share mix path: True
pair0: tgt=5535415..._2.4022_..._-2.4022.wav mouth=5535415699068794046_00004.pt near=<同 stem>.wav
pair1: tgt=<同 stem>.wav                      mouth=6160231649911868423_00015.pt near=<同 stem>.wav
target==near? 0        # 无一例自引用
```

注意 pair0/pair1 的 `target` **basename 相同**（都叫 `<stem>.wav`），但父目录不同（`.../s1/` vs `.../s2/`）。这解释了 §2.8 的 P0：**一旦 `--copy-wavs` 把它们压成同一个目录，仅靠 basename 就无法区分**，所以修法是加 `.s1/.s2` 后缀而不是换目录。

---

## 5. 坑与反直觉点

1. **`idx` 被丢弃**（`avspeech_dataset_near.py:154` / `avspeech_dataset.py:90`）。`shuffle=True` 完全不生效；`__len__` 是 `train_samples` 而不是数据集大小；同一 epoch 内会重复采到同一条。改 `train_samples` 就等于改"一个 epoch 走多少数据"，跟"数据量"无关。
2. **"随机裁剪"在 LRS2 上是 no-op**：`n_samples == seg_len == 32000` ⇒ `randint(0,0)=0`（`:156`）。所以 LRS2 训练时音频窗口**恒为整条**，只有视频 crop/flip 在随机。要真正做 random crop，得让 `segment < 2.0`。
3. **`mouth_start` 写死 16000**（`:168`、`avspeech_dataset.py:108`）。`sample_rate=8000` 时 `mouth_start` 会是正确值的一半，音视频直接错位。当前 6 个 config 都是 16000，属于"潜伏 bug"。
4. **没有 pad，只有 slice**（`:174`）。`T_mouth < fps_len` 时返回短张量，`bs>1` 的 `torch.stack` 直接崩。LRS2 靠"全部 50 帧"过关，不是靠代码保护。
5. **mouth dtype 是 float64**（实测 §4.1）。`Normalize(0.,255.)` 在 uint8→float64 上做，`collate` 后整 batch 是 float64；模型 `mouth_in.type_as(mixtures)` 才降到 float32。数据侧因此多一倍内存/拷贝。
6. **near 使用已缩放的另一说话人源**：发布包中的 s1/s2 波形已按混音比例缩放；训练索引引用对应对侧波形。
7. **`near_path == mix_path` 的哨兵是静默的**（`:190`）。fallback 数据不会崩、不会 warn，只会让 K=0、对比损失退化成"只有正项"。RealSSA 全量 9169 条都是这个状态，而训练日志只会打 `9169/9169 samples have near-speaker entries`（因为 `near.json` 确实每条都有 1 条 entry）——**这个日志看起来像"有负样本"，其实是"有哨兵"**。这是最容易误判的一处。
8. **训练集索引顺序是反的**：`__init__` 从 `len-1` 往 0 遍历并 append（`:112-122`），所以 `self.mix` 是 `mix.json` 的逆序。训练无所谓（随机采样），但如果有人拿 `train_ds.mix[i]` 去对 `mix.json[i]` 就会错位。`AVSpeechEvalDataset` 是正序（`avspeech_dataset.py:162-165`）。
9. **`AVSpeechDataModule`（非 near）是死路但留在 `__init__.py` 导出**：它的 train dataset 返回 **4 元组**，而 `AudioVisualLightningModuleNASSv2.training_step` 解包 **5 元组**（`av_litmodule_nass_v2.py:74`）⇒ 一旦有人 `from look2hear.datas import AVSpeechDataModule` 配 NASS 训练，会在第一个 batch 上 `ValueError: not enough values to unpack`。`audio_train.py:25` 绕开了 `__init__.py`，所以主路径没炸。
10. **`check_avspeech_dataset.py` 不是检查器**：整份拷贝 dataset 类 + `__main__` 把 10 条样本 dump 成 `mixture/*.wav`、`sources/*.wav`、`sources_mouths/*.pt`（`:315-347`），**不校验任何不变量**。真正的不变量检查在 `tools/repro/audit_data.py`。`check_*` 的文件名不能当作"已经有校验"的证据。
11. **两份 `near.json` 生成器**：`DataPreProcess/process_lrs23.py:46-83` 与 `DataPreProcess/generate_near_json.py:39-78` 逻辑几乎一样但**不是同一份代码**（前者被 `make_synthetic_data.build_indexes()` 调用，后者是独立 CLI）。前者在空 split 上会因 `min(near_counts)` 崩（`:80`），`make_synthetic_data.py:100-102` 特意绕了这个坑。schema 漂移风险实际存在。
12. **val/test 的 `n_near` 来自 `_meta.json`，而 LRS2 没有 meta**（`avspeech_dataset.py:208-217`）。所以 `val_pit_sisnr_{1..4}near` 这些桶永远是空的，只有 `_0near` 有值。看报表时别以为"分桶评测"生效了。
13. **`drop_last=True` 用在 val/test 上**（`:313,324`），`bs=1` 所以不丢样本；但如果谁把 val `bs` 调大就会**静默丢掉尾批**（无 warning）。
14. **`falcon_lrs2_2mix_paper` 不自包含**：`mix.json`/`target.json` 指向 `/root/data/lrs2_2mix/lrs2_rebuild/...` 绝对路径，只有 mouth 是自己那份。删/移 `lrs2_2mix` 会让 paper 数据集和 `falcon_lrs2_2mix` 的 `near.json` **同时**失效（A5-3 已提到，未实施）。
15. **`falcon_lrs2_2mix` 的 `near.json` 反向指向 `lrs2_2mix`**，但 `mix/target/mouths` 是自己那份拷贝 ⇒ 这是一个"半自包含"数据集，最容易在迁移时只搬一半。
16. **cv/tt 的检查集极端复用**（§4.7）：cv 只有 68 个说话人 / 476 条 utterance 撑起 5000 条 mixture。`val_loss` 的 epoch 间波动会显著小于"5000 条独立样本"的直觉。
17. **`train_samples=10000` 与 40000 条数据**：paper 配置下每个 epoch 只过 25% 的条目且**有放回**。所以"1000 epoch"其实 ≈ 250 遍全量，不是 1000 遍。

---

## 6. 与既有文档的冲突 / 更正

### 6.1 `conversion_stats.json` 说 `fallback`，磁盘是 `other-speaker`（**独立复核成立**）

| 口径 | 值 | 证据 |
|---|---|---|
| `conversion_stats.json` | `"near_mode": "fallback"`，`generated_at 2026-10-21T19:23:28Z`，mtime 2026-10-21 19:24 | 实测 `cat` + `ls -l` |
| 磁盘 `near.json` | 100% 指向 `lrs2_rebuild/.../s2/...`，K=1，fallback=0；mtime 2026-10-22 16:37 | 实测 §4.2 + `stat` |
| 训练日志 | `Loaded near.json with 20000 entries` / `20000/20000 samples have near-speaker entries` | P2D-near-train.log（文档引用，我未复跑训练） |

既有文档 `docs/verification/phase2/P2C-near-mode-provenance.md` 已完整记录这个漂移（含 mtime、时间线、以及"未入库的一次性命令"判断）。**我的独立实测与它一致**，并补充两点它没写的：
- 不仅 `falcon_lrs2_2mix`，`falcon_lrs2_2mix_paper` 的 `conversion_stats.json` 也**不记录** `near` 的 K/来源（只有 `note` 一句话），无法从 stats 反推 near 语义。
- paper 的 `near` 与单 target 版的 `near` 指向**同一批文件**（同一个 s2 wav），所以两套数据的负样本是同一批波形，只是配对方向不同。

### 6.2 STATUS.md 的磁盘数字是"逐目录归因"，不是"独占字节"

- STATUS:24 写 `falcon_lrs2_2mix（24G，28k）` / `falcon_lrs2_2mix_paper（26G…）` / `falcon_realssa_sim（mean，30G）` / `falcon_realssa_sim_ch0（ch0，30G）`。**逐目录各自成立**，但四者之和 110 G ≠ 实际并集 **62 G**（§3.1 硬链接实测）。写容量规划/迁移时要按 62 G 算，按 110 G 会高估 48 G。
- STATUS:24 写 LRS2 源包 33 G；实测 `du -sh /root/data/lrs2_2mix` = **17 G**（`docs/DATA_ACCESS_GUIDE.md:23` 也写"16.47 GB 压缩 / 17.39 GB 解压"）。STATUS 的 33 G 疑为把 `falcon_lrs2_2mix` 的部分算进去了，或早期含 tar 时的口径 → **按 17 G 修正**。

### 6.3 "LRS2-2Mix 有 s1/s2 两个源" vs 数据集契约

既有文档（如 `FINAL_PROJECT_UNDERSTANDING.md` 一类）若把任务描述成"2 说话人分离 + PIT"，在**数据契约层**是不准确的：`n_src=1`、`target.json` 只有一路 target、`PITLossWrapper` 是 1×1。准确表述是"**目标说话人提取（TSE）**，输入是 2 源混合，输出是 1 路目标；s1/s2 的角色由 `target.json` 的条目决定，通过双向构造来消除'更响者'偏差"。

### 6.4 `check_avspeech_dataset.py` 的"检查"名不副实

若既有文档把它列为"数据不变量校验"，应更正为"样本 dump 工具"。不变量校验的实际承担者是 `tools/repro/audit_data.py`（exit code 0/1 语义明确，`:18-24`）。

---

## 7. GAP LIST

1. **训练期 val/test 是否真的用 `center crop` 的 88×88 + 全 50 帧**？`AVSpeechEvalDataset` 的 mouth **不切 50 帧**（返回全 `T`），LRS2 恰好 `T=50` 所以等价；但 RealSSA 的 `T` 可以到 171 → 验证时 mouth 长度不是 50。**下游 `video_model` 对变长 `T` 的行为（是否按最小时长截断 audio？）未验证** —— 属 W1 模型组范围，但数据侧确实不是定长。（建议：给 RealSSA 做一次 `bs=1` 前向验证 `video_model` 输出的 `T'` 与 `mixtures` 的 `T` 比例。）
2. **`raw_audio` near 模式从未在磁盘上落地过**，`select_near()` 的 `strict/loose` 回退与 `per_sample>1` 的路径只有单测覆盖（`tools/repro/tests/test_lrs2_convert.py`）。若论文 App A.2 的 near 重建要做，这里是唯一已有入口。
3. **RealSSA-Sim tr/cv 未下载**（只有 tt 19 shard），所以"RealSSA 从零训练"缺 450 GB 数据；`STATUS:84` 的待办成立，但**下载源/时间/带宽**没有任何量化记录。
4. **`lrs2_rebuild/raw_audio/train` 只有 19649 条**，而 tr 有 18691 条 unique utterance —— 两者的差集/映射关系没查（可能 `raw_audio` 含 cv/tt 之外的片段，也可能是截断）。若走 `raw_audio` near 模式，候选池的实际覆盖率未知。
5. **`falcon_lrs2_2mix` 与 `falcon_lrs2_2mix_paper` 的 mouth `.pt` 是否**逐字节**相同？** 只验证了 1 个 inode 相同 + 文件数一致；没做全量 `md5`。若有人重跑 paper 构建（不同 `--jobs`），可能产生"同 id 不同内容"的文件（转换器有 `if dst.exists(): skip`，理论上安全）。
6. **`mouths` 硬链接是谁建的、用什么命令建的**（`ln`？`cp -l`？rsync `-H`？）没有任何记录。这决定了"删一个目录会不会连带删另一个"（硬链接不会，但 `rm -rf` 后空间释放要等最后一个引用消失）。迁移数据前需要先弄清这一点。
7. **`falcon_realssa_sim` 与 `_ch0` 的音频内容不同但 `mix.json` 的 `n_samples` 相同**（同 id 同长度）。没验证"mean 与 ch0 的目标波形是否只差一个线性组合"（理论上不是）。对 DNSMOS 类指标的影响未评估。
8. **eval 的 `n_near` 对 RealSSA 恒 0 是因为 meta 里没有 `type=="near"`**——但 `build_meta()` 在 `--near-dir` 缺省时确实一条都不加。如果将来加外部 near，`AVSpeechEvalDataset` 会开始产出非 0 桶，与现有报表不可比。**属潜在口径断裂点**，需在加数据时同步声明。
9. **`train_samples=10000` 的来历**：没有任何注释/commit 说明为什么是 10000（不是 20000/40000）。疑似沿用上游 LRS3 配置。这直接影响"epoch"的定义与论文 Table 16 的 epoch 数对齐，属**未解释的关键超参**。
10. **`segment=2.0` 与 LRS2 的 2.000 s 完全相等**是巧合还是论文设定？仓库无记录。若论文实际是"随机 crop 2 s from longer clips"，则当前 LRS2 训练**与论文不一致**（无随机裁剪）。需对照论文 §5.2 原文确认。

---

## 8. 证据清单

### 代码（远端仓库，只读）

```
look2hear/datas/avspeech_dataset.py            334L   TrainDataset(4元组)/EvalDataset(5元组)/DataModule
look2hear/datas/avspeech_dataset_near.py       337L   collate_with_near(:23-44) / TrainDatasetWithNear(:60-222) / DataModule(:224-337)
look2hear/datas/check_avspeech_dataset.py      349L   拷贝 + dump __main__(:315-347)
look2hear/datas/transform.py                   187L   get_preprocessing_pipelines(:131-153)
look2hear/datas/__init__.py                      5L   只导出 AVSpeechDataModule
look2hear/system/av_litmodule.py               108L   validation_step 5元组解包(:86-121)
look2hear/system/av_litmodule_nass_v2.py       185L   training_step 5元组(:73-74) / near→neg(:112-133) / β退火(:68-72)
DataPreProcess/process_lrs23.py                119L   preprocess_one_dir(:8-44) / generate_near_json(:46-83)
DataPreProcess/generate_near_json.py           113L   process_split(:39-78)
tools/repro/lrs2_2mix_to_falcon.py             689L   STEM_RE(:96) / convert_one(:188-232) / build_indexes(:280-335) / other-speaker(:315-323)
tools/repro/phase2/build_lrs2_paper_data.py    311L   PAPER_ENTRIES(:48) / build_split(:72-150) / copy-wavs 修复(:98-117) / validate_build(:182-260) / exit2(:303-306)
tools/repro/realssa_to_falcon.py               580L   select_channel(:179-190) / align(:354-366) / build_meta(:240-...) / --channel(:539)
tools/repro/make_synthetic_data.py             355L   build_indexes(:93-134)
tools/repro/audit_data.py                      205L   真正的数据闸门
tools/repro/fetch_data.py                      395L   DATASETS(:50-80) / --sha256
audio_train.py                                 456L   datamodule(:296) / set_seed(:257-266)
audio_visual_test.py                           875L   AVSpeechEvalDataset(:372)
```

### 数据文件（远端，只读实测）

```
/root/data/falcon_lrs2_2mix/{tr,cv,tt}/{mix.json,target.json,near.json}
/root/data/falcon_lrs2_2mix/conversion_stats.json            (near_mode=fallback, 与磁盘不符)
/root/data/falcon_lrs2_2mix_paper/{tr,cv,tt}/{mix.json,target.json,near.json}
/root/data/falcon_lrs2_2mix_paper/conversion_stats.json      (含 gate 结果 tr=40000 等)
/root/data/falcon_realssa_sim[_ch0]/tt/{mix.json,target.json,near.json} + mix/*_meta.json ×9169
/root/data/lrs2_2mix/lrs2_rebuild/{audio/wav16k/min/{tr,cv,tt}/{mix,s1,s2}, mouths/*.npz, raw_audio/{train,val,test}, create_mix/*.scp,*.m}
/root/data/RealSSA-Sim/{README.md, tt/*.tar ×19}
/root/data/RealSSA-Real/{audio,mouth,metadata/*.jsonl}
/root/data/falcon_eval/realssa_real_mean_full{,2,3}/{summary.json,realssa_real_dnsmos_per_segment.csv,run_metadata.json}
```

### 命令（可复现）

```bash
# 环境
ENV=/root/miniforge3/envs/falcon/bin/python; cd /root/code/github_repos/Falcon-fork; export PYTHONPATH=$PWD

# 条数/K分布/配对（本报告 §4.2 §4.3 §4.10）
$ENV /tmp/w106dp/probe1.py   # dataset+batch 真实 shape
$ENV /tmp/w106dp/probe5.py   # 说话人/utterance 复用
$ENV /tmp/w106dp/probe7.py   # paper 双向配对
$ENV /tmp/w106dp/probe8.py   # mouth .pt 结构
$ENV /tmp/w106dp/probe9.py   # SNR/gain 分布

# 官方闸门
$ENV tools/repro/audit_data.py /root/data/falcon_lrs2_2mix --sample 60        # exit 0
$ENV tools/repro/audit_data.py /root/data/falcon_lrs2_2mix_paper --sample 60  # exit 0
$ENV tools/repro/audit_data.py /root/data/falcon_realssa_sim --sample 50      # 预期 exit 1（near 全 fallback）

# 磁盘（注意：多目录一次 du 会去重硬链接）
du -h --max-depth=3 /root/data/falcon_lrs2_2mix
du -sch /root/data/falcon_lrs2_2mix /root/data/falcon_lrs2_2mix_paper \
        /root/data/falcon_realssa_sim /root/data/falcon_realssa_sim_ch0    # → 62G total
stat -c "%i %h %n" /root/data/falcon_lrs2_2mix/tr/mouths/5535415699068794046_00004.pt
```

### 相关既有文档（已交叉核对）

```
docs/STATUS.md:24,54,74,84
docs/DATA_ACCESS_GUIDE.md:23-27
docs/verification/phase2/P2C-near-mode-provenance.md:1-80        （§6.1 交叉核对）
docs/understanding/THREE_ROUND_REVIEW_20261030.md:25,99,135     （A5-1 P0 与修复确认）
docs/REPRODUCTION_PLAN.md §5.1/§5.2/§6.1                        （F5 单声道修复、near 三选项）
```
