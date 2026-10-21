# 05 · 数据管线（datas / DataPreProcess / transform / JSON 索引）

> 负责人：`data-pipe`（R1 初版，R3 修订）。范围：`look2hear/datas/{avspeech_dataset.py, avspeech_dataset_near.py, check_avspeech_dataset.py, transform.py, __init__.py}`、`DataPreProcess/{process_lrs23.py, generate_near_json.py}`、README 数据准备章节。
> 约定：`[代码]`=源码直接确认；`[推断]`=由源码/文档合理推断；`[验证]`=实际运行确认（命令与日志见 §8）。行号以 `main@7576854` 为准。
> 版本锚定：R1 初版 `f1c90fa3…`（777 行）→ R2-prep/评审锚点 `fde1ed6a…`（797 行）→ 评审后补 dtype 说明 `dcd32572…`（799 行）→ **R3 修订版（本文件，900 行）**。R3 定稿 sha256 记录在 `docs/verification/runs/data_pipe/05-data-pipeline-r3.sha256`（由复核人 `sha256sum` 复核）；评审锚点见 `docs/reviews/round1/05-data-pipeline-review-by-train-sys.md`；本版在 §9.1 逐条响应评审意见。
> 本文件之外未修改任何源码。

---

## 1. 子系统职责与总体位置

**一句话结论**：本子系统把磁盘上的 LRS2/LRS3/VoxCeleb2/RealSSA 风格数据（`mix/`、`target/` wav + 预抽取的 `mouths/*.pt` + RealSSA 的 `mix/*_meta.json`）编译成 3 份 JSON 索引（`mix.json`/`target.json`/`near.json`），再在训练/评测时按固定契约产出：**2.0 s@16 kHz 音频（32000 采样点）+ 50 帧 88×88 灰度嘴部特征 + 可变数量的近场干扰人音频**，供 Lightning 模型与 NASS 说话人对比损失消费。**数据模拟（RealSSA 场景生成）与嘴部帧抽取都不在本仓库内**，仓库只消费它们。

### 1.1 完整链路（磁盘 → JSON → 样本 → collate → batch 张量）

```
【磁盘】
$DATA/{tr,cv,tt}/
├── mix/<stem>.wav              16 kHz 混合音频（每个 <stem> 一个样本）
├── mix/<stem>_meta.json        RealSSA 场景元数据（**NASS 对比损失必需资产；仓库不提供生成器**；用于 near.json + 评测/验证 n_near 统计）
├── target/<stem>.wav           16 kHz 目标说话人 clean 音频（与 mix 同 stem）
└── mouths/<stem>.pt            预先抽取的嘴部帧（4-D channel-last，仓库内不生产；格式由 dataset 反推）
        │
        │  DataPreProcess/process_lrs23.py:84-105   （内嵌 generate_near_json，process_lrs23.py:46-81）
        ▼
【JSON 索引】$INDEX/{tr,cv,tt}/
├── mix.json      [[abs_wav, n_samples], ...]                       process_lrs23.py:19-20,31-32
├── target.json   [[abs_wav, abs_mouth_pt, n_samples], ...]         process_lrs23.py:21-28,31-32
└── near.json     [[[abs_near_wav, n_samples], ...], ...]           generate_near_json.py:43-71
        │
        │  look2hear/datas/avspeech_dataset_near.py
        ▼
【Dataset 样本】AVSpeechTrainDatasetWithNear.__getitem__ → 5 元组        near.py:152-223
  (mixture[32000] f32, source[32000] f32, mouth[50,88,88] *, filename str, near_audios list[[32000] f32])
        │                                        * mouth dtype 依 .pt 存储类型：float32→f32，uint8→f64，见 §5.4
        │  collate_with_near  near.py:23-43
        ▼
【batch 张量】train loader (B=config 4)  near.py:297-307
  mixtures [B,32000] f32 · sources [B,32000] f32 · mouths [B,50,88,88]（dtype 同上，litmodule 以 type_as 兜底转 f32）
  filenames list[str] (len B) · near_audios list[list[Tensor[32000] f32]]（K 可变）
        │
        ▼
【下游】AudioVisualLightningModuleNASSv2.training_step  av_litmodule_nass_v2.py:73-134
  mouth [B,50,88,88] --unsqueeze(1)--> [B,1,50,88,88] --FRCNNVideoModel--> mouth_emb [B,512,50]
  mixtures [B,32000] + mouth_emb --> FalconSpeakerContrastive --> est_sources [B,n_src,32000]
  近场音频 as 负样本 --> NASSContrastiveLoss（训练期唯一用途）
```

评测链不同：`audio_visual_test.py:183-189` 用同一 DataModule 但只取 `make_sets` 的 dataset（**不走 DataLoader/collate**），逐条 `test_set[idx]` 取整段音频与整段嘴部序列做整句推理（§3.5）。Studio 产品线不读 JSON，而是在 `studio/falcon_runtime.py:95-105` **重实现**同一套嘴部预处理，并已与训练仓库 `val` 管线逐元素一致（§3.9，[验证]）。

### 1.2 文件清单与消费者

| 文件 | 行数 | 职责 | 谁消费 |
|---|---|---|---|
| `look2hear/datas/__init__.py` | 5 | 只导出旧版 `AVSpeechDataModule` | 无实际消费者（见 2.1） |
| `look2hear/datas/avspeech_dataset.py` | 327 | `AVSpeechTrainDataset`（旧训练集）、`AVSpeechEvalDataset`（**仍被 near 模块复用**）、`AVSpeechDataModule`（旧，死代码） | `avspeech_dataset_near.py:13` 复用 EvalDataset |
| `look2hear/datas/avspeech_dataset_near.py` | 337 | `AVSpeechTrainDatasetWithNear`、`collate_with_near`、`AVSpeechDataModuleWithNear`（**当前唯一训练入口**） | `audio_train.py:24,49`、`audio_visual_test.py:46,187` |
| `look2hear/datas/transform.py` | 187 | 嘴部帧 numpy 变换管线（Normalize/RandomCrop/CenterCrop/HorizontalFlip） | 两个数据集的 `lipreading_preprocessing_func` |
| `look2hear/datas/check_avspeech_dataset.py` | 347 | 旧数据集类的手工拷贝 + `__main__` 导出调试样本 | 无（不可直接运行，见 2.5） |
| `DataPreProcess/process_lrs23.py` | 121 | 生成 `mix.json`/`target.json`，并（内嵌）生成 `near.json` | 数据准备 CLI |
| `DataPreProcess/generate_near_json.py` | 105 | 独立生成 `near.json` | 数据准备 CLI（README 推荐，但步骤已内嵌，见 2.6/2.7） |
| `README.md` / `README_zh.md` | 146-175 / 148-177 | 数据准备说明 | 用户 |

### 1.3 对 `00-project-map.md` 假设的校正

- 宏观链路（project map §4.1）方向正确：`tr/cv/tt → JSON → AVSpeechDataModuleWithNear(segment=2.0, 16kHz) → NASSv2`。但 **mouth 张量不是 `[B,F,T,96,96]`**：实际为 `[B,50,88,88]`（96×96 存储帧在 Dataset 中裁剪为 88×88），见 §5.4。`[代码]`+`[验证]`
- project map 说 “near.json（对比学习负样本索引）” 正确；但 **“分桶”不在 Dataset 内**，而在 Lightning 模块里仅用于日志分组（`min(K,4)`），不影响采样。`[代码]`（av_litmodule_nass_v2.py:19,113-118）
- project map §3 的 `datas/` 行数正确（327/337/347/187）；但 **`datas/__init__.py` 只导出旧模块**，当前训练/评测都用完整路径 import near 模块。`[代码]`
- 数据准备只需**一个**脚本即可产出三份 JSON（`process_lrs23.py` 已内嵌 near 生成）；README 却列出两条命令，第二条是幂等冗余。`[代码]`+`[验证]`

### 1.4 关键结论（R3 升格）：`_meta.json` 是 NASS 对比损失的必要输入，而 README 流程不生成它

**结论（本文档最重要的可复现性发现）**：按 `README.md:148-158` 的数据准备流程复现时，**每个训练样本的近场负样本数 K=0**，NASS 说话人对比损失退化为**只有正项**（负项恒 0）、MDCL 日志/损失恒 0、验证指标全部分进 `0near` 桶——即论文贡献之一的“近场干扰人负样本对比学习”在不报错的情况下**静默失效**。`[代码]`+[验证]

完整证据链（每一环均已核对源码，并由 §8.2 的 fixture 实测支撑）：

1. README 的数据准备只有两条命令、**从不生成 `_meta.json`**（`README.md:148-158`；全仓库 grep 只有 `process_lrs23.py`/`generate_near_json.py`/`avspeech_dataset.py` **读取**它，无任何生成者）。
2. 无 meta 时，`process_lrs23.py:69-71` / `generate_near_json.py:62-65` 写入 fallback `[[mix_path, mix_n]]`。
3. Dataset 读到 fallback 哨兵（`near_path == mix_path`）时**静默跳过**（`avspeech_dataset_near.py:188-191`）→ `near_audios=[]`。
4. 训练侧：K=0 → `embed_negs_list[b]=None`（`av_litmodule_nass_v2.py:115-122`）→ `NASSContrastiveLoss` 的 `l_neg=0`（`nass_contrastive_loss.py:73-74`）→ 对比损失只剩 `α_pos(1−cos(q,p))`。
5. MDCL 分支同样 `if n_near == 0: continue` → `mdcl_total` 恒 0（`av_litmodule_mdcl.py:114-126`）。
6. 验证侧：`n_near` 也来自 meta（`avspeech_dataset.py:201-211`）→ 全为 0 → 所有样本只进 `val_pit_sisnr_0near`（`av_litmodule.py:161-169`）。

→ 影响：**用本仓库 + README 数据流程训练，NASS 对比损失实际不生效**，train/val 日志中的 `train_spk_contrast_loss`、`*_near` 分桶指标会给出误导性的“正常”数值。该资产需要由 RealSSA/数据生成端提供（仓库未开源）；见 Q3/Q6（R3 已把“文档层后果”结案，仅剩“原始训练数据是否另有 meta 生成器”需数据侧确认）。

---

## 2. 文件清单与逐文件解析

### 2.1 `look2hear/datas/__init__.py`（5 行）

```python
from .avspeech_dataset import AVSpeechDataModule   # :1
__all__ = ["AVSpeechDataModule"]                    # :3-5
```

- 只导出**旧版** `AVSpeechDataModule`，没有导出 `AVSpeechDataModuleWithNear`/`AVSpeechTrainDatasetWithNear`。两个入口脚本都按完整路径导入 near 模块（`audio_train.py:24`、`audio_visual_test.py:46`）。`[代码]`
- `audio_visual_test.py:45` 还有一句 `import look2hear.datas`，会连带 import 旧 `avspeech_dataset.py`（需要 torchvision/matplotlib），但没有实际用到其符号。`[代码]`

### 2.2 `look2hear/datas/avspeech_dataset.py`（327 行）

#### 2.2.1 `normalize_tensor_wav(wav_tensor, eps=1e-8, std=None)`（:17-21）

```python
mean = wav_tensor.mean(-1, keepdim=True)      # :18
if std is None: std = wav_tensor.std(-1, keepdim=True)  # :19-20
return (wav_tensor - mean) / (std + eps)      # :21
```
逐段（最后一个维度 = 时间）去均值、除标准差；`eps=1e-8` 防静音段除零。near 模块里有一份完全相同的拷贝（`avspeech_dataset_near.py:16-20`）。`[代码]`

#### 2.2.2 `AVSpeechTrainDataset`（:23-122）——旧版训练集（当前训练不用）

| 项 | 值 / 行为 | 证据 |
|---|---|---|
| 构造签名 | `json_dir, sample_rate=8000, segment=4.0, normalize_audio=False, samples=None` | :24-31 |
| 默认值 | 8 kHz / 4 s，与当前配置（16 kHz / 2 s）不同 | :27-28 |
| 预处理管线 | `get_preprocessing_pipelines()["train"]`；`transforms.Grayscale(num_output_channels=1)` | :38-39 |
| `seg_len` / `fps_len` | `int(segment*sample_rate)` / `int(segment*25)` | :42-43 |
| `self.test` | `self.seg_len is None` —— **恒为 False 的死分支**（`int(None*rate)` 会先抛 TypeError） | :45 |
| JSON | 读 `mix.json` + `target.json` | :46-56 |
| 过滤 | 倒序遍历，`mix[i][1] < seg_len` 的样本从 mix/target 中同步删除并统计 drop 数 | :65-81 |
| `__len__` | 返回 `samples`（不是真实样本池大小） | :82,84-85 |
| `__getitem__` | 随机 `mix_index`（忽略 `idx`）；随机 `rand_start ∈ [0, n-seg_len]`；读 audio+mouth；可选归一化；返回 **4 元组** | :87-122 |
| `mouth_start` | `int((rand_start / 16000) * 25)`（硬编码 16 kHz/25 fps） | :108 |
| 返回 | `(mixture, source, source_mouth, basename)` | :122 |

#### 2.2.3 `AVSpeechEvalDataset`（:125-212）——**评测/验证实际使用的 dataset**

| 项 | 值 / 行为 | 证据 |
|---|---|---|
| 构造签名 | `json_dir, sample_rate=8000, normalize_audio=False` | :126-131 |
| 预处理管线 | `get_preprocessing_pipelines()["val"]`（CenterCrop 88，无随机增广） | :138 |
| 过滤 | **不过滤**短音频；按 JSON 升序逐条 append（顺序 = JSON 顺序） | :161-164 |
| `__len__` | `orig_len` = `len(mix.json)` | :171,173-174 |
| 整段读取 | `start=0, stop=None`，不做 2 s 窗 | :179-187 |
| 嘴部 | 整段帧（不分窗）经 val 管线 → `(F,88,88)` | :189-191 |
| 通道归约 | `torch.from_numpy(x).mean(-1)`，**无 ndim 守卫**（mono → 0 维张量；见 P0 风险） | :193-194 |
| 归一化 | 可选，同一 `normalize_tensor_wav` | :196-199 |
| 近场计数 | 读 `mix路径.replace(".wav","_meta.json")`，数 `components[].type=="near"`，异常静默 → `n_near` | :201-210 |
| 返回 | `(mixture, source, source_mouth, basename, n_near)` **5 元组** | :212 |

#### 2.2.4 `AVSpeechDataModule`（:214-313）——旧版，死代码

- 签名/默认值与 near 版基本一致（`sample_rate=8000, segment=4.0, batch_size=64, train_samples=10000`），但训练集用旧 `AVSpeechTrainDataset`（无近场负样本）。`[代码]` :214-229
- `setup()` 建 3 个 dataset（:253-270）；`make_loader`/`make_sets` 是 **`@property`**（:307-313），与 near 版同构。
- **无消费者**：全仓库没有任何代码 import `AVSpeechDataModule` 之外的使用者（入口脚本都走 near 版）。`[代码]`（grep 结果见 §8.4）

### 2.3 `look2hear/datas/avspeech_dataset_near.py`（337 行）——当前训练主路径

#### 2.3.1 `collate_with_near(batch)`（:23-43）

```python
mixtures, sources, mouths, filenames, near_list = zip(*batch)   # :37
mixtures = torch.stack(mixtures)                                # :38
sources  = torch.stack(sources)                                 # :39
mouths   = torch.stack([m if torch.is_tensor(m) else torch.from_numpy(m) for m in mouths])  # :40-42
return mixtures, sources, mouths, list(filenames), list(near_list)  # :43
```
docstring 明确约定输出形状（:30-36）：`mixtures [B,T]`、`sources [B,T]`、`mouths [B,fps_len,H,W]`、`filenames list[str]`、`near_audios list[list[Tensor[T]]]`。**只有 train loader 使用这个 collate**。`[代码]`

#### 2.3.2 `AVSpeechTrainDatasetWithNear`（:46-223）

构造签名（:60-68）：
```python
json_dir, sample_rate=16000, segment=4.0, normalize_audio=False, samples=None, use_near=True
```
| 项 | 值 / 行为 | 证据 |
|---|---|---|
| near.json 文件格式说明 | 嵌套 list，与 mix.json 同序 | :50-58 |
| `seg_len` / `fps_len` | `int(segment*sample_rate)` / `int(segment*25)` | :77-78 |
| 预处理 | train 管线 + `Grayscale(1)`，打印管线 | :80-82 |
| JSON 路径 | `<json_dir>/mix.json`、`target.json`、`near.json` | :85-87 |
| 读 near 条件 | `use_near and os.path.exists(near_json)`；否则 `near_infos=None`，缺失时打印 WARNING | :94-101 |
| 过滤 | 倒序删短样本，同时删 `target_infos[i]` 与 `near_infos[i]`（保持三表对齐） | :111-122 |
| 统计打印 | `samples_with_near` = **非空 near 列表**的样本数（含 fallback，见 P2） | :130-132 |
| 启动自检 | 挑第一条非空 near 读 100 帧，打印 OK/FAILED | :133-145 |
| `__len__` | `samples`（= `train_samples`，默认 10000；不是数据量） | :147-150 |
| `__getitem__` | 随机 `mix_index`（忽略 idx）+ 随机 `rand_start`；读 mix/target；读嘴部并按帧号切片；可选归一化；读近场负样本 | :152-223 |
| 返回 | `(mixture, source, source_mouth, basename, near_audios)` **5 元组** | :222-223 |

关键细节（详见 §3.6/§3.7/§3.8）：倒序过滤后 `self.mix` 顺序是 **JSON 顺序的反序**（:111-122 倒序遍历 + `append`）；目标/混合音频读取有 `ndim>1` 守卫（:176-177），而 eval 数据集没有（见 P0 风险）。

#### 2.3.3 `AVSpeechDataModuleWithNear`（:226-337）——**唯一训练 DataModule**

构造签名（:237-252）：
```python
train_dir, valid_dir, test_dir, n_src=1, sample_rate=16000, segment=4.0,
normalize_audio=False, batch_size=4, num_workers=8, pin_memory=True,
persistent_workers=False, train_samples=10000, use_near=True
```
| 方法 | 行为 | 证据 |
|---|---|---|
| `__init__` | 校验 3 个 dir 非 None、`n_src ∈ {1,2}`；只做赋值，不建 dataset | :253-271 |
| `setup()` | train = `AVSpeechTrainDatasetWithNear`（带 `use_near`/`train_samples`）；val/test = `AVSpeechEvalDataset`（**固定用 16k 配置值，不传 segment**） | :277-295 |
| `train_dataloader()` | `batch_size=config`, `shuffle=True`, `drop_last=True`, `collate_fn=collate_with_near` | :297-307 |
| `val_dataloader()` / `test_dataloader()` | **`batch_size` 硬编码 1**（config 的 4 被忽略），default collate，`drop_last=True` | :309-329 |
| `make_loader` | **`@property`** 返回 `(train, val, test)` 3 个 DataLoader，每次访问新建 | :331-333 |
| `make_sets` | **`@property`** 返回 `(data_train, data_val, data_test)` | :335-337 |

该类**不继承** `pl.LightningDataModule`，因此不能 `Trainer.fit(datamodule=...)`；入口脚本手工 `setup()` + 取 loader 再传给 LightningModule。`[代码]`

### 2.4 `look2hear/datas/transform.py`（187 行）

纯 numpy/PIL 风格的手写变换（非 torchvision），只用于嘴部帧 `(T,H,W) float32`。`[代码]`

| 类/函数 | 行为 | 证据 |
|---|---|---|
| `Compose(preprocess)` | 顺序执行 | :16-36 |
| `RgbToGray` | `cv2.cvtColor` 逐帧转灰；**当前数据集路径不用**（用的是 `torchvision.transforms.Grayscale`） | :39-56 |
| `Normalize(mean,std)` | `(frames-mean)/std`（广播到 numpy 数组） | :59-79 |
| `CenterCrop(size)` | 以 `(th,tw)` 中心裁剪 | :82-100 |
| `RandomCrop(size)` | `random.randint` 随机左上角裁剪 | :103-124 |
| `HorizontalFlip(flip_ratio)` | 以概率翻转每帧（`cv2.flip`，原地修改） | :127-144 |
| `get_preprocessing_pipelines(Crop_size=(88,88))` | 返回 `{"train","val","test"}`；`mean,std=(0.421,0.165)`；train=`Normalize(0,255)→RandomCrop(88)→HorizontalFlip(0.5)→Normalize(.421,.165)`；val=`Normalize(0,255)→CenterCrop(88)→Normalize(.421,.165)`；test=val | :147-169 |
| `get_preprocessing_opt_pipelines` | 另一套 torchvision 版本，**无调用者** | :171-187 |

- `Normalize(0.0, 255.0)` 语义就是除以 255（`(x-0)/255`），把 uint8/0-255 映射到 0-1；再按 LRW 的均值/方差标准化。`[代码]` :153-159
- `HorizontalFlip` 用 `cv2.flip(frames[index], 1)` 原地改 numpy 数组（:141-143）。`[代码]`

### 2.5 `look2hear/datas/check_avspeech_dataset.py`（347 行）

- 用途：**手工调试/导出脚本**。它是 `avspeech_dataset.py` 的一份**旧拷贝**（同样两个 Dataset + DataModule 类），仅额外加了 `__main__`：用写死的 `${DATA_ROOT}/av-tiger-lrs3/...` 路径实例化 `AVSpeechTrainDataset(samples=10)`，把 10 条随机样本写成 `look2hear/datas/{mixture,sources,sources_mouths}/` 下的 wav/pt 文件。`[代码]` :312-347
- 与主实现的漂移：**不含 near 相关类**；train/eval 的 `mean(-1)` 无 ndim 守卫；没有 `_meta.json` 读取。`[代码]`
- 不可按原样运行：路径是未展开的占位符（:317）；实测 `python look2hear/datas/check_avspeech_dataset.py` → `FileNotFoundError: '${DATA_ROOT}/av-tiger-lrs3/DataPreProcess/LRS3_multi/tr/mix.json'`。`[验证]`（§8.3 #17）
- 导入兼容：try `.transform`，失败时把自身目录加入 `sys.path` 再 `from transform import ...`（:17-22），所以能作为脚本启动。`[代码]`
- 无任何 CI/入口引用（grep 仅自身）。`[代码]`

### 2.6 `DataPreProcess/process_lrs23.py`（121 行）

| 函数 | 行为 | 证据 |
|---|---|---|
| `preprocess_one_dir(in_audio_dir, out_dir, out_filename, data_type)` | `abspath` 输入目录 → `listdir` + **sort** → 只收 `.wav` → `sf.SoundFile` 取帧数 → 按 `out_filename` 分支写 schema → `makedirs(out_dir)` → `json.dump(..., indent=4)` | :8-32 |
| └ mix 分支 | `(abs_wav, n_samples)` | :19-20 |
| └ target 分支 | `(abs_wav, abs_wav.replace("target","mouths").replace(".wav",".pt"), n_samples)` | :21-28 |
| `get_all_near_paths(meta_path)` | 读 `_meta.json` 的 `components`，返回 `type=="near"` 的 `path` 列表 | :35-43 |
| `generate_near_json(mix_json_path, mix_dir, out_json_path)` | 逐 mix 条目：`<stem>_meta.json` → near 路径 → 存在且可读则 `[path, sf.info().frames]`；否则 fallback `[[mix_path, mix_n]]` → 写 `near.json`（indent=2）+ 统计 | :46-81 |
| `preprocess(args)` | 对 `tr/cv/tt` × `["mix","target"]` 调 `preprocess_one_dir`，随后（mix.json 存在时）调内嵌 `generate_near_json` | :84-105 |
| CLI | `--in_audio_dir`（default `"--"`）、`--out_dir`（default `"LRS3_multi"`）；usage 文案 `"WHAM data preprocessing"`（复制残留） | :108-121 |

### 2.7 `DataPreProcess/generate_near_json.py`（105 行）

- 用途：独立/可重跑地生成 `near.json`；逻辑与 `process_lrs23.py:46-81` 基本相同（`get_all_near_paths` :27-36、`process_split` :39-78）。`[代码]`
- CLI：`--data_root`（default `"--"`）、`--out_root`（default `"--"`）、`--sample_rate`（default 16000，**解析后从未使用**）。`[代码]` :82-90
- 主流程：对 `tr/cv/tt`，`mix_json = out_root/<split>/mix.json`，`mix_dir = data_root/<split>/mix`，`out_json = out_root/<split>/near.json`；`mix.json` 不存在则打印 `Skipping <split>` 并继续。`[代码]` :92-101
- 输出：`os.makedirs(dirname(out_json))` 后 `json.dump(..., indent=2)`；统计打印 `near_per_sample min/max/avg`。`[代码]` :69-78

### 2.8 README 数据准备章节（EN `README.md:146-175`，ZH `README_zh.md:148-177`）

README 的说法：数据遵循 LRS2/LRS3 布局（`tr/cv/tt`，各含 `mix/`、`target/` wav 16 kHz 与目标说话人嘴部裁剪）；两条命令建索引；编辑 `configs/Falcon.yml` 的 `data_config.{train,valid,test}_dir`；脚本不展开环境变量；visual backbone 放 `pretrain_zoo/`。`[代码]` README:148,154-158
与实现的对照见 §5.9；主要缺口：**没有说明 `mouths/` 目录名与 `.pt` 张量格式、没有说明 `_meta.json`、第二条命令其实是冗余的**。`[代码]`+`[验证]`

---

## 3. 关键调用链与数据流

### 3.1 磁盘目录布局与命名规则

```
$DATA/
├── tr/  cv/  tt/                          # 3 个 split；process_lrs23.py:87 固定遍历这三个
│   ├── mix/
│   │   ├── <stem>.wav                     # 16 kHz 混合音频
│   │   └── <stem>_meta.json               # 可选；RealSSA 场景元数据（本仓库不生产）
│   ├── target/
│   │   └── <stem>.wav                     # 16 kHz 目标 clean 音频（与 mix 同 stem）
│   └── mouths/
│       └── <stem>.pt                      # 嘴部帧张量（4-D channel-last，见 §5.2）
└── （近场干扰源的 wav 位置自由，由 _meta.json 的 path 指定）

$INDEX/                                     # process_lrs23.py 的 --out_dir
├── tr/  cv/  tt/
│   ├── mix.json  target.json  near.json
```

- **命名/对齐规则**：`mix/`、`target/` 各自 `listdir` 后**独立字典序排序**（`process_lrs23.py:12-13`），再按下标一一对应；`mouths/<stem>.pt` 由 target 路径字符串替换得到（:25）。因此三者的**唯一对齐依据是同名 stem + 相同排序**，代码**不校验**（见 P1-1）。`[代码]`
- 目标/近场路径全部是**绝对路径**（`os.path.abspath`，:11），JSON 与机器绑定。`[代码]`+`[验证]`

### 3.2 JSON schema 与对齐规则

#### (a) `mix.json` —— `[[str abs_wav_path, int n_samples], ...]`
- 生成：`preprocess_one_dir` mix 分支（`process_lrs23.py:19-20,31-32`），`indent=4`。
- 消费：训练集用 `[0]` 读音频、`[1]` 做长度过滤/随机窗上界（`avspeech_dataset_near.py:89-90,111-122,156-157,160-162`）；评测集只用 `[0]`（`avspeech_dataset.py:182-184`）。
- `n_samples` = wav 帧数（`sf.SoundFile.__len__`），单位采样点；16 kHz 下 2 s = 32000。

#### (b) `target.json` —— `[[str abs_wav_path, str abs_mouth_pt_path, int n_samples], ...]`
- 生成：target 分支（`process_lrs23.py:21-28`）。`mouth_pt = abs_wav.replace("target","mouths").replace(".wav",".pt")`（:25）→ 即 `<DATA>/<split>/mouths/<stem>.pt`。
- 消费：`[0]` target wav、`[1]` mouth .pt（`avspeech_dataset_near.py:91-92,163-165,171`；eval `avspeech_dataset.py:185-187,190`）。
- **第三字段 `n_samples` 从未被读取**（两个 Dataset 都只用 `[0]/[1]`）→ target 与 mix 不等长不会被发现（P1-2）。`[代码]`

#### (c) `near.json` —— `[[[str abs_near_wav, int n_samples], ...], ...]`
- 语义：**外层与 `mix.json` 同序等长**（一条 mix 对应一个内层 list），内层是这条样本可用的近场干扰人音频列表。`[代码]`（generate_near_json.py:43-67,73-78）
- 生成规则：
  1. `<stem>_meta.json` 存在 → 取 `components[].type=="near"` 的 path（`generate_near_json.py:27-36`）；
  2. 路径存在且 `sf.info()` 可读 → 追加 `[path, frames]`（:54-60）；
  3. 一个都没有（无 meta / 只有 far / 文件缺失 / 不可读）→ **fallback `[[mix_path, mix_n]]`**（:62-65）；
  4. 因此 `len(near.json) == len(mix.json)` 恒成立（除非外部破坏）。`[代码]`+`[验证]`
- 消费：训练集按 `mix_index` 取 `near[mix_index]`（`avspeech_dataset_near.py:103-127,186-220`）；`use_near=False` 时完全不加载。
- fallback 的哨兵语义：`near_path == mix_path` → Dataset 跳过（:190-191），避免把“估计≈混合”当负样本惩罚。`[代码]`+[验证]

#### (d) `<stem>_meta.json`（**输入侧，非本仓库生成**）——`{"components": [{"type": "near"|..., "path": "..."}, ...], ...}`
- 代码只读 `components[].type` 与 `components[].path` 两个字段（`process_lrs23.py:35-43`；`avspeech_dataset.py:208`）。其他字段（若存在，例如距离/SIR/类型细分）**全部被忽略**。`[代码]`
- 该文件的产生脚本不在仓库中（RealSSA 数据生成未开源）；**README 完全未提其产生方式**（唯一 TODO 是 backbone 权重下载，`README.md:158`）。缺它时 near.json 全为 fallback → K=0，见 §1.4。`[代码]`+`[推断]`

### 3.3 建索引链（可操作化）

```
$ python DataPreProcess/process_lrs23.py --in_audio_dir $DATA --out_dir $INDEX
  for split in [tr, cv, tt]:                        # process_lrs23.py:87
    for spk in [mix, target]:                       # :86
      preprocess_one_dir($DATA/<split>/<spk>, $INDEX/<split>, spk, split)   # :8-32
        listdir + sort                              # :12-13
        skip 非 .wav                                 # :15-16
        n = len(sf.SoundFile(wav))                  # :18
        mix → [abs, n]；target → [abs, mouths/*.pt, n]  # :19-28
        write $INDEX/<split>/<spk>.json (indent 4)  # :29-32
    if $INDEX/<split>/mix.json missing: print skip; continue   # :101-103
    generate_near_json($INDEX/<split>/mix.json, $DATA/<split>/mix, $INDEX/<split>/near.json)  # :46-81,96-105
      for each mix entry: basename → $DATA/<split>/mix/<stem>_meta.json  # :54-55
        parse components, keep type=="near", keep readable files         # :35-43,58-67
        if empty → fallback [[mix_path, mix_n]]                          # :69-71
      write near.json (indent 2) + stats                                 # :73-81
```

- **可重入/覆盖**：全部用 `"w"` 覆盖写；无临时文件/追加。对同一输入重跑，输出 JSON **字节级相同**（已测）。`[验证]` §8.3 #2
- **容错**：
  - 缺整个输入目录 → `os.listdir` 抛 `FileNotFoundError`，脚本中止（未捕获）。`[验证]` #9
  - 某 split 为空目录 → `mix.json`/`target.json`/`near.json` 写成 `[]` 后，`generate_near_json` 的统计行 `min([])` 抛 `ValueError`，脚本以非 0 退出（JSON 已写）。`[验证]` #10
  - 缺 `<split>/mix.json`（独立 `generate_near_json.py`）→ 打印 `Skipping` 继续。`[验证]` #14
  - meta 缺失/近场文件缺失/不可读 → 静默 fallback，**不报警**（只体现在统计 `missing/fallback` 计数）。`[代码]`+[验证]
  - `mix/` 与 `target/` 文件名集合不一致但数量相同 → **静默错配**（各自排序后错位，不抛错）。`[验证]` #15-16
  - `mix/` 比 `target/` 多文件 → 后续 Dataset 构造时 `del target_infos[i]` / 下标访问抛 `IndexError`（无显式校验）。`[验证]` #H1

### 3.4 训练链（`audio_train.py` → DataModule → DataLoader → LightningModule）

```
audio_train.py:44   use_near = config["datamodule"]["data_config"].get("use_near", False)
audio_train.py:49   datamodule = AVSpeechDataModuleWithNear(**data_config)
audio_train.py:50   datamodule.setup()
audio_train.py:51   train_loader, val_loader, test_loader = datamodule.make_loader   # @property
audio_train.py:55-58 model = FalconSpeakerContrastive(sample_rate=data_config["sample_rate"], ...)
audio_train.py:119-136 system = AudioVisualLightningModuleNASSv2(..., train_loader=..., val_loader=..., test_loader=...)
audio_train.py:161-173 trainer = pl.Trainer(...); trainer.fit(system)
```
- 注意：DataLoader 被**手工取回后作为参数传给 LightningModule**（`av_litmodule.py:36-43` 中保存 `self.train_loader`），而不是 PL 的自动 datamodule 集成。`[代码]`
- **`test_loader` 不是死路径（R3 修正，原 R1 结论错误）**：`AudioVisualLightningModule.val_dataloader()` 返回 **`[self.val_loader, self.test_loader]`**（`av_litmodule.py:214-216`）。`trainer.fit(system)`（`audio_train.py:180`）每个 epoch 都会让 PL 的 evaluation loop 遍历**两个** loader：`dataloader_idx==0` 是 cv 验证集（计算 `val_loss`），`dataloader_idx==1` 是 tt 测试集；当 `current_epoch % 10 == 0` 时在 tt 上计算并记录 `test_loss`（`av_litmodule.py:125-140`，epoch 末再汇总于 `on_validation_epoch_end`：`:174-181`）。**`trainer.test()` 从未被调用**（`audio_train.py:174-182` 只有 `trainer.fit`）。`[代码]`+[验证]（train-sys 11-epoch dummy 实测 `test_loss/dataloader_idx_1`，见 03 文档 V4）
- **训练期 val 循环由两份 `AVSpeechEvalDataset` 组成（cv + tt）**：`datamodule.setup()` 把 val/test 都建成 `AVSpeechEvalDataset`（`avspeech_dataset_near.py:286-295`，与评测入口共用同一 dataset 类/同一整句语义）；故训练期验证与 `audio_visual_test.py` 的数据读取行为完全一致（整句、无 2 s 窗口、CenterCrop 88、读 meta 得 `n_near`）。`[代码]`（与 03/04 文档的接口一致）
- 每个 epoch 的 batch 数 = `train_samples / batch_size / world_size`（`train_samples=10000` 默认、batch=4、DDP 10 卡时单卡 250 个 batch）。`[推断]`（由 :147 `__len__` + config:164 + :131 gpus 推导）

### 3.5 评测链（`audio_visual_test.py`，**不走 DataLoader**）

```
audio_visual_test.py:183-186  data_cfg = {k:v for k,v in train_conf[...]["data_config"].items()
                                          if k not in ("use_near","train_samples")}
audio_visual_test.py:187      datamodule = AVSpeechDataModuleWithNear(**data_cfg, use_near=False)
audio_visual_test.py:188      datamodule.setup()
audio_visual_test.py:189      _, _, test_set = datamodule.make_sets          # @property
audio_visual_test.py:192-193  indices = range(len(test_set)); local = indices[rank::world_size]
audio_visual_test.py:229      mix, sources, mouth, key, _ = tensors_to_device(test_set[idx], device)
audio_visual_test.py:230      mouth_emb = video_model(torch.from_numpy(mouth[None, None]).float())
audio_visual_test.py:231      est_sources = model(mix[None], mouth_emb)
```
- 评测**逐条**索引 dataset（无 collate、无 pin_memory/persistent_workers、无 batch），`mouth` 是 numpy `(F,88,88)` → `[None,None]` → `(1,1,F,88,88)`。`[代码]`
- 因此评测的完整句长 = mix wav 全长、嘴部帧数 = `.pt` 全长；**不是 2 s 窗口**。`[代码]`+[验证]
- 多卡按 `rank::world_size` 切分 test index（:192-193）。`[代码]`
- 该路径**要求 train_dir/valid_dir/test_dir 三者都可用**（`setup()` 会建 3 个 dataset），只跑 test 也一样。`[验证]` §8.2 J2
- **与训练期验证共用 `AVSpeechEvalDataset`**：训练时 `val_dataloader()` 返回 `[cv_loader, tt_loader]`（`av_litmodule.py:214-216`），两个 loader 的 dataset 都是本类（`avspeech_dataset_near.py:286-295`）→ 训练/评测的样本读取契约完全一致（整句音频、整段嘴部、5 元组含 `n_near`）；区别只在评测脚本丢弃 `n_near` 且逐条前向。`[代码]`（与 03/04 对齐）

### 3.6 单样本读取时序（训练 vs 评测）

| 步骤 | 训练 `AVSpeechTrainDatasetWithNear.__getitem__` | 评测 `AVSpeechEvalDataset.__getitem__` |
|---|---|---|
| 1. 选样本 | `mix_index = random.randint(0, len(mix)-1)`，**忽略 idx**（near.py:154） | 用 `idx`（avspeech_dataset.py:176-187） |
| 2. 选窗口 | `rand_start ∈ [0, n_samples-seg_len]`，`stop=rand_start+32000`（near.py:156-157） | `start=0, stop=None` 整段（:179-180） |
| 3. 读 mix/target | `sf.read(..., start, stop, dtype="float32")`（near.py:160-165） | 整段（:182-187） |
| 4. 通道归约 | `ndim>1 → mean(-1)`，mono 保持 1-D（near.py:176-177） | `mean(-1)` **无守卫**（:193-194） |
| 5. 嘴部帧窗口 | `mouth_start=int(rand_start/16000*25)`；切片 `[mouth_start:mouth_start+50]`（near.py:168-174） | 整段帧（:189-191） |
| 6. 嘴部预处理 | `permute(0,3,1,2)`→Grayscale→squeeze→numpy→**train** 管线（90°随机增广） | 同样→**val** 管线（中心裁剪，无增广） |
| 7. 归一化 | 可选，`normalize_audio`（默认 false）（near.py:179-182） | 可选，同函数（:196-199） |
| 8. 近场负样本 | 读 `near[mix_index]`（near.py:184-220） | 只数 `_meta.json` 里 near 数量（:201-210） |
| 9. 返回 | `(mix[32000], src[32000], mouth[50,88,88], basename, near_list)` | `(mix[T], src[T], mouth[F,88,88], basename, n_near)` |

### 3.7 mouth 帧与音频时间对齐规则

- **常数**：`sample_rate=16000`、`fps=25`、`segment=2.0`。`seg_len=int(2.0*16000)=32000`（near.py:77），`fps_len=int(2.0*25)=50`（:78）。一帧 = 16000/25 = **640 采样点 = 40 ms**；2 s = 50 帧 × 640 = 32000 采样点（严格相等）。`[代码]`+[验证]
- **帧号公式**：`mouth_start = int((rand_start / 16000) * 25) = floor(rand_start / 640)`（near.py:168；旧训练集同式 :108）。因此嘴部第 0 帧的“名义起始采样点” = `mouth_start*640`，音频窗口实际起点 `rand_start` 与之相差 `rand_start % 640 ∈ [0,639]` 个采样点（**0–39.94 ms，最多不足 1 帧**）。`[代码]`+[推断]
- **50 帧的来历**：`fps_len = int(segment*25)` 由 segment 决定，而不是视频文件的实际帧率；只要数据按 25 fps 抽取就对齐。`[代码]`
- **硬编码风险**：`mouth_start` 除的是字面量 `16000`、fps 用的字面量 `25`（near.py:168/78），与构造参数 `sample_rate` 脱钩；只有 `sample_rate=16000` 时正确。实测 `sample_rate=8000` 时 mouth_start 应为 31 而实际算成 15 → 视觉/音频错位约 0.6 s。`[验证]` §8.2 H4
- **尾部/短文件**：切片是 numpy 的“安全但静默”行为——若 `.pt` 帧数 < `mouth_start+50`，返回帧数 <50，随后 `collate_with_near` 的 `torch.stack` 因各样本帧数不同抛 `RuntimeError`；Dataset 不检查。实测：75 帧 vs 40 帧的嘴部文件混批 → `RuntimeError: stack expects each tensor to be equal size, but got [25,88,88] and [50,88,88]`。`[验证]` §8.2 H3
- **评测端**：不用切片，整段帧 + CenterCrop(88)；训练用 RandomCrop(88)，两者空间对齐位置不同（中心 vs 随机），但 Studio 推理用中心裁剪，与评测一致（§3.9）。`[代码]`+[验证]

### 3.8 near-field 负样本链路与“near”的定义

```
_meta.json(外部) --type=="near"--> near.json 内层列表
    → Dataset: 跳过 fallback（path==mix_path）、跳过 n_samples<32000 的近场文件、读 [min(rand_start, n-32000), +32000)
    → collate: 变长 list[list[Tensor[32000]]]
    → NASSv2.training_step: K=len(near_audios[b]); bucket=min(K,4) 仅用于日志分桶
       stack [K,32000] → audio_model.encode_stft → pool_spk_embed → neg_embs [K,D]
    → NASSContrastiveLoss: K=1 时 relu(sim-margin) 求和；K>1 时 softmax(β·sim) 加权硬负样本；K=0 时 l_neg=0
```

- **“near”的唯一判据**：`_meta.json` 的 `components[].type == "near"`（`process_lrs23.py:40-42`；`generate_near_json.py:33-35`）。仓库内**没有任何距离阈值、SIR 阈值或远近分界逻辑**——这些定义在未开源的 RealSSA 数据生成流程里（README:45,67 只描述“2–6 个动态近/远场说话人，距离与 SIR 按场景采样”）。`[代码]`+`[推断]`
- **far-field 干扰人不作为负样本**：只取 `type=="near"`，`far` 组件被过滤。`[代码]`+[验证]
- **负样本选择**：不做采样/分桶选择，**全部可用近场干扰人都作为负样本**；分桶只出现在 Lightning 日志（`train_n_samples_0near..4plus_near`）。`[代码]`（av_litmodule_nass_v2.py:19-33,113-118,182-183）
- **K=0 的语义**：该样本只贡献对比损失的 `l_pos`，负项为 0（NASSContrastiveLoss:62-74）；训练不中断，但 MDCL 对比约束缺失。`[代码]`
- **K>1 的语义**：`softmax(beta*sim_neg)` 给更相似的负样本更高权重（`beta` 从 `beta_init` 线性退火到 `beta_final`，config:104-111 → 0→1 over 200 epochs）。`[代码]`
- **校验/过滤细节**：近场文件 `n_samples < seg_len` 的条目在 `__getitem__` 里被跳过（:193-194）；读取偏移 `min(rand_start, n_samples-seg_len)`（:198-199）——**当近场文件短于 mix 时会在末端“钳位”，时间轴与 mix 不再对齐**（代码注释承认这是对“pre-segmented clip”的妥协）。`[代码]`+[验证] §8.2 C6
- **只用于训练**：val/test 只从 `_meta.json` 统计 `n_near` 做分组指标（`avspeech_dataset.py:201-210`；`av_litmodule.py:86-119`），不读近场音频。`[代码]`

### 3.9 Studio 侧映射（跨产品线一致性）

`studio/falcon_runtime.py:95-105` 的 `preprocess_mouth(frames)`：要求 `uint8 (50,96,96)`（**固定 50 帧窗口**，形状断言在 :99-100，变长 F 会抛 `ValueError`），执行 `frames[:,4:92,4:92]/255` 再 `(x-0.421)/0.165`（全程 float32），返回 `[1,1,50,88,88]`。其数学与训练仓库 `get_preprocessing_pipelines()["val"]` 一致（`transform.py:161-167`：CenterCrop(88)=96→88 的中心 4:92、Normalize(0,255)、Normalize(.421,.165)）。**两边真实代码交叉验证（R3.1 条件化）：同一 `(50,96,96)` uint8 输入下等价到 float32 精度（raw `max|Δ|=2.574e-07`、cast 后 2.384e-07）；仅当 `.pt` 以 float32 存储（训练侧输入 float32、val 输出 float32）时逐位一致（`max|Δ|=0.0`）。** 注意训练 val 管线实际返回 `(50,88,88)`（uint8 输入时 float64），`[1,1,·]`/float32 是对比时的手工包装。`[验证]` §8.4(1)

---

## 4. 关键实现细节与设计动机

### 4.1 为什么是 2.0 s / 32000 点 / 50 帧
`segment=2.0`（configs/Falcon.yml:162）→ `seg_len=32000`、`fps_len=50`；与论文/Studio 的 2 s 窗口、16 kHz、25 fps、50 帧完全一致（studio/falcon_runtime.py:24-27）。训练随机取 2 s 段，是为了让模型只见过“窗口内”分布，与流式推理窗口匹配；评测则整句推理。`[代码]`+`[推断]`

### 4.2 “伪 epoch”：`__getitem__` 忽略 idx + `__len__=train_samples`
`AVSpeechTrainDatasetWithNear.__getitem__` 内部重新 `random.randint` 选样本（near.py:154），`__len__` 返回 `samples`（train_samples，默认 10000，:147-150）。于是：
- 一个 epoch 的长度由 `train_samples`（此处默认 10000）而非数据集大小决定；
- DataLoader 的 `shuffle=True` 实际无意义（索引本就随机）；
- 采样是**有放回**的，短数据/长数据都会被均匀抽样。
`[代码]`+[推断] 该设计在 `configs/Falcon.yml` 中没有 `train_samples` 键（§5.5），因此实际生效值 10000，README 未说明（P1-3）。

### 4.3 归一化策略（注意：当前配置关闭）
`normalize_audio: false`（configs/Falcon.yml:163）→ 当前训练/评测**都不做音频归一化**；代码支持的可选路径是：`mixture` 用自身 std 去均值/除 std，**target 复用 mixture 的 std**（near.py:179-182；旧版 :117-120），从而保持 target 相对混合的比例；静音段由 `eps=1e-8` 保护为 0（实测）。`[代码]`+[验证]

### 4.4 视觉预处理：96×96 存储 → 88×88 输入
- 训练/评测都把 `mouths/<stem>.pt` 当 4-D channel-last 张量：`permute(0,3,1,2)` → `Grayscale(1)` → `squeeze(1)` → numpy `(T,96,96)`；再走 transform 管线（near.py:169-174）。
- 88×88 来自 LRW 唇读惯例（`get_preprocessing_pipelines` default `Crop_size=(88,88)`，transform.py:147-151），配套均值/方差 `(0.421, 0.165)`（:152）；冻结的 FRCNN 视觉主干即在此输入上预训练（configs/Falcon.yml:8 `frcnn_128_512.backbone.pth.tar`、README:107）。`[代码]`+`[推断]`
- 训练增广：RandomCrop（空间随机位置）+ HorizontalFlip(0.5)；评测/Studio 用 CenterCrop（96→88 中心 4:92）。`[代码]`+[验证]

### 4.5 近场负样本的设计动机（对应论文“encoder-space speaker contrastive loss”）
NASS 风格同空间说话人对比：query=分离出的目标嵌入（带梯度）、positive=target 波形嵌入、negative=**近场干扰人**嵌入；近场泄漏是真实场景中最难区分的目标/干扰（近场 SIR 高），因此把近场说话人作为负样本能在推理零开销下抑制泄漏。far-field 干扰人不是同一“空间尺度”，故不作负样本。`[代码]`（av_litmodule_nass_v2.py:97-134；nass_contrastive_loss.py:1-10）+`[推断]`（动机与论文口径，README:45,67）

### 4.6 训练/评测/流式三者的窗口语义
- 训练：随机 2 s 段（near.py:156-174）。
- 训练期验证：整句（AVSpeechEvalDataset），batch=1。
- 评测：整句逐条（audio_visual_test.py:229-231）。
- Studio：固定 2 s 窗口 + 0.5 s hop（非本子系统范围，仅接口对齐）。
`[代码]`+[推断] **训练段 vs 评测整句的统计口径差异**需要 loss/metric 文档确认（Q5）。

### 4.7 已发现的缺陷与风险（P0/P1/P2）

| 级别 | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P0** | `AVSpeechEvalDataset` 对 mono wav 执行无守卫 `mean(-1)` → `mixture/source` 退化为 0 维张量；随后 `mix[None]` 变成 `[1]` 送模型 → 真实模型在 `torch.stft` 抛 `RuntimeError`；**评测入口与训练期验证（同一 spec 的 `[cv,tt]` 双 loader）都会在首个样本崩溃** | `avspeech_dataset.py:193-194`；`av_litmodule.py:214-216,125-140`；实测 mono → `shape=()`、`RuntimeError`（§8.7） | 若真实数据是 mono（README:67 声称 RealSSA “mono 16 kHz”），**评测命令与训练验证均不可用**；能训练成功则反证真实数据为多声道（Q1） |
| P1-1 | 三份 JSON 只靠“排序后下标”对齐，无文件名/长度校验：数量相同但名字错位 → **静默错配**；数量不同 → 构造期 `IndexError` | `process_lrs23.py:12-13,19-28`；实测 §8.3 #15-16/H1-H2 | 数据工程错误无提示，训练目标错位 |
| P1-2 | `target.json` 第三字段 `n_samples` 从未使用；target 比 mix 短时 `sf.read` 静默返回短数组，批内长度不一致才在 collate 崩 | `avspeech_dataset_near.py:160-165`；`avspeech_dataset.py:182-187` | 数据质量问题延迟到 batch 层才暴露 |
| P1-3 | `train_samples` 不在 YAML 中，默认 10000 决定 epoch 大小；README 无说明 | `avspeech_dataset_near.py:147-150,250`；configs/Falcon.yml:156-168 | 复现者以为 epoch=数据集大小 |
| P1-4 | 嘴部 `.pt` 帧数不足/超过音频时长时无校验：短 → collate `RuntimeError`；长 → 静默截断到 50 帧 | `avspeech_dataset_near.py:168-174`；实测 H3 | 数据抽取参数不一致时训练中断 |
| P1-5 | `mouth_start`/fps 硬编码 `16000`/`25`，与 `sample_rate` 参数脱钩；改采样率即错位 | `avspeech_dataset_near.py:168`；`avspeech_dataset.py:108`；实测 H4 | 配置自由度陷阱 |
| **P1-6（R3 升格为主结论，见 §1.4）** | `_meta.json` 缺失不是“可选边角”：按 README 数据流程无任何脚本生成它 → near.json 全部 fallback → Dataset 静默跳过哨兵 → **K=0** → NASS 负项恒 0（只剩正项）、MDCL 恒 0、val 分桶全进 `0near`；且 `samples_with_near` 统计把 fallback 计入，日志“7/7 samples have near-speaker entries”具有误导性 | `process_lrs23.py:69-71`；`generate_near_json.py:62-65`；`avspeech_dataset_near.py:188-191,130-132`；`av_litmodule_nass_v2.py:115-122`；`nass_contrastive_loss.py:73-74`；实测 G3 | 论文贡献之一静默失效（训练不报错、日志看似正常） |
| P1-7 | **near 源 wav 采样率未纳入契约**：near.json 只存 `[path, frames]`（不存 sample_rate），Dataset 直接按 16 kHz 解释 frames/seg_len 并 `sf.read(start,stop)`，**不做重采样、不校验**；若 near wav 是 48 kHz，帧数比较与时间轴都会错位（有效片段只有 1/3） | `process_lrs23.py:61-67`；`generate_near_json.py:54-60`；`avspeech_dataset_near.py:193,198-203` | 近场负样本内容错位（弱标签），且无告警 |
| P2-1 | 空 split 目录：JSON 写 `[]` 后统计 `min([])` 抛错，脚本非 0 退出 | `generate_near_json.py:73-78`；实测 #10 | 空集场景不可用 |
| P2-2 | `check_avspeech_dataset.py` 是不可运行的旧拷贝（写死 `${DATA_ROOT}`，与主实现漂移），且无 CI 引用 | :312-347,317；实测 #17 | 易误导、不可复用 |
| P2-3 | README 第二条命令冗余（`process_lrs23.py` 已生成 near.json）；README **完全未提** `mouths/` 目录名、`.pt` 4-D 格式、`_meta.json` 的必要性（只说“mouth crops”） | `process_lrs23.py:96-105`；README:148-158；实测 #11-13 | 数据准备说明不完整（meta 缺失已升格 P1-6） |
| P2-4 | 近场文件短于 mix 时偏移被钳位 → 时间轴失配；注释已承认 | `avspeech_dataset_near.py:195-199` | 负样本内容可能错位（弱标签） |
| P2-5 | fallback 哨兵依赖绝对路径字符串完全相等；若符号链接/相对路径归一不同则自惩罚 | :188-191 | 工程健壮性 |
| P2-6 | `process_lrs23.py` 默认值 `"--"`、`generate_near_json.py` 默认值 `"--"` 且 `--sample_rate` 未使用；usage 文案是 "WHAM data preprocessing" | `process_lrs23.py:109-118`；`generate_near_json.py:83-90` | 易误用 |
| P2-7 | `preprocess_one_dir` 的 `data_type` 形参未使用；`sf.SoundFile` 未 close | `process_lrs23.py:8,14,18` | 代码卫生 |
| P2-8（R3 修正） | val/test DataLoader 硬编码 `batch_size=1`、`drop_last=True`；config 的 batch_size 只对 train 生效。**`test_loader` 并非死路径**：它是第二个 validation dataloader（`[val,test]`），每 epoch 被遍历、每 10 epoch 记录 `test_loss`；仅 `trainer.test()` 从未调用 | `avspeech_dataset_near.py:309-329`；`av_litmodule.py:214-216,125-140`；`audio_train.py:180` | 配置键语义与直觉不符（batch_size 只作用于 train），但 tt 集在训练期确有（每 10 epoch 一次指标的）参与 |
| P2-9 | `AVSpeechDataModule`/旧训练集为死代码；`datas/__init__.py` 不导出当前实现 | avspeech_dataset.py:23-313；`__init__.py:1-5` | 维护困惑 |
| P2-10 | 文件名取 `path.split("/")[-1]`，Windows 路径会返回全路径（本仓库面向 Linux） | `avspeech_dataset_near.py:222`；`avspeech_dataset.py:122` | 跨平台小问题 |
| P2-11 | `num_workers=8 + persistent_workers=true` 组合正确；但若使用者把 num_workers 改为 0 而不同时关 persistent_workers，PyTorch 直接抛 `ValueError` | 实测；configs/Falcon.yml:165-167 | 调参陷阱（非 bug） |
| P2-12 | mouth 路径由 `wav_path.replace(out_filename,"mouths")` 生成，是**全路径字符串替换**：若 `$DATA` 根路径含 "target" 子串，mouth 路径会被破坏；且建索引**不校验** `.pt` 是否存在（缺失延迟到训练 `torch.load` 才报错） | `process_lrs23.py:25` | 数据工程陷阱；建议改为 `os.path.join(dirname, "mouths", stem+".pt")` 并校验存在 |
| P2-13 | I/O 开销：训练每个样本 `torch.load` **整段** `.pt` 只为窗口取 50 帧（多帧/长视频浪费）；eval 每条额外读一次相邻 `_meta.json`（`n_near` 随后被丢弃） | `avspeech_dataset_near.py:171`；`avspeech_dataset.py:206` | `num_workers=8` 可摊薄，但仍是可观的重复 I/O |
| P2-14 | README 唯一的 TODO 是 backbone 权重下载（`README.md:158` HTML 注释），**数据/meta/`.pt` 的生成方式只字未提** | README:148-158 | 复现者无法从 README 得知 meta/ mouths 从哪来（与 P1-6 相关） |

---

## 5. 对外接口契约

### 5.1 数据目录契约（训练/评测共用）

| 路径 | 必需 | 格式 | 说明 |
|---|---|---|---|
| `<split>/mix/<stem>.wav` | ✅ | 16 kHz wav（通道数见 Q1） | 混合音频；长度 ≥ 2 s 才会参与训练（评测不过滤） |
| `<split>/target/<stem>.wav` | ✅ | 16 kHz wav，与 mix 同 stem | 目标 clean 音频 |
| `<split>/mouths/<stem>.pt` | ✅ | torch tensor，4-D channel-last（`[T,96,96,C]`，**C=3 最可信**；C=1 的可用性依赖真实 torchvision，未验证，见 P1-2） | 25 fps 嘴部帧；仓库内**不生产**；存储 dtype 决定输出 dtype（§5.4） |
| `<split>/mix/<stem>_meta.json` | **✅（NASS 对比学习/val 分桶必需）** | `{"components":[{"type":..., "path":...}, ...]}` | 仓库**不生成**；缺失 → near.json 全 fallback → K=0 → 对比损失静默失效（§1.4/Q6） |
| near 源 wav（由 meta 指定） | 可选（必需 meta 时配套） | **16 kHz wav**（Dataset 不重采样，只按 16 kHz 解释 frames），建议与 mix 等长 | <2 s 的会被 Dataset 跳过；采样率错误会错位（P1-7） |

### 5.2 JSON schema 契约（形式化）

```jsonc
// mix.json  (indent=4)
[ [ "/abs/path/tr/mix/u0.wav", 48000 ], ... ]                       // 2 元组

// target.json (indent=4)
[ [ "/abs/path/tr/target/u0.wav", "/abs/path/tr/mouths/u0.pt", 48000 ], ... ]  // 3 元组；[2] 未被使用

// near.json (indent=2)
[ [ ["/abs/path/near/u0_n1.wav", 48000], ["/abs/path/near/u0_n2.wav", 48000] ],  // 与 mix.json 同序等长
  [ ["/abs/path/tr/mix/u0copy.wav", 48000] ],   // fallback：与 mix[1] 完全相同（Dataset 会跳过）
  ... ]
```

对齐规则：**列表下标对应**；`len(near.json)==len(mix.json)==len(target.json)`；路径必须绝对且可直接 `sf.read`/`torch.load`。`[代码]`+[验证]
- **near.json 不含采样率字段**：只写 `[path, sf.info().frames]`（`generate_near_json.py:54-60`；`process_lrs23.py:61-67`），Dataset 直接用 `n_samples < seg_len` 与 `sf.read(start,stop)` 按 16 kHz 解释（`avspeech_dataset_near.py:193,198-203`）→ near 源 wav 必须是 16 kHz，否则帧数比较与时间轴都会错位（P1-7）。`[代码]`

### 5.3 Dataset item 契约

| 序号 | 训练（WithNear） | 评测（EvalDataset） |
|---|---|---|
| 0 | `mixture: Tensor float32 [32000]` | `mixture: Tensor float32 [T]`（T=整句；mono 数据时退化为 0 维，见 P0） |
| 1 | `source: Tensor float32 [32000]` | `source: Tensor float32 [T]` |
| 2 | `mouth: np.ndarray [50,88,88]`（train 管线；dtype 依 .pt 存储类型，见 §5.4） | `mouth: np.ndarray [F,88,88]`（val 管线，F=整句帧数；dtype 同上） |
| 3 | `filename: str`（mix 的 basename） | 同左 |
| 4 | `near_audios: list[Tensor float32 [32000]]`，K 可变（可为空；README 流程下恒为空，见 §1.4） | `n_near: int`（`_meta.json` 中 near 组件数；同样依赖 meta） |
| 长度 | `__len__ == samples`（默认 10000），与实际样本池无关 | `__len__ == len(mix.json)` |

### 5.4 collate 后的 batch 形状与 dtype（权威结论，dtype 条件化 R3）

**训练 loader**（`collate_fn=collate_with_near`，config `batch_size=4`，`shuffle=True`，`drop_last=True`）：

| 张量 | 形状 | dtype | 证据 |
|---|---|---|---|
| `mixtures` | `[B, 32000]`（B=4） | `torch.float32` | near.py:38；[验证] D1 |
| `sources` | `[B, 32000]` | `torch.float32` | near.py:39；[验证] D2 |
| `mouths` | `[B, 50, 88, 88]`（**不是 96×96，也不是 5-D**） | **依 `.pt` 存储 dtype**：float32 存储 → `torch.float32`；uint8 存储 → `torch.float64`（见下） | near.py:40-42；[验证] D3 + 真实推理格式 |
| `filenames` | `list[str]`，长度 B | — | near.py:43；[验证] D4 |
| `near_audios` | `list[list[Tensor[32000]]]`，长度 B；内层 K 可变（实测跨两 batch 的 K 序列为 `[0,2,0,0]`，即单 batch 内可为 0 或 2） | `torch.float32` | near.py:43,184-220；[验证] D5 |

**dtype 的条件化结论（R3 采纳 train-sys P1-3）**：
- 读取链：`torch.load(.pt)` → `permute(0,3,1,2)` → `Grayscale` → `.squeeze(1).numpy()` → `Normalize(0.0,255.0)`（`near.py:169-174`；`transform.py:153-158`）。在 numpy 中 `uint8 − 0.0 → float64`、`float32 − 0.0 → float32`。
- **若 `.pt` 为 uint8**（README:132 的推理格式即 uint8 96×96，训练资产大概率同源）：batch `mouths` 为 **float64**——本轮用真实权重 + uint8 `.pt` 的评测复现实测 `mouth(50/75,88,88) float64`（§8.7）。
- **若 `.pt` 为 float32**：batch `mouths` 为 float32（R1 fixture 即如此，D3）。
- **下游兜底**：训练 `mouth_in = mouth.unsqueeze(1); ... type_as(mixtures)`（`av_litmodule_nass_v2.py:83-87`）；评测 `torch.from_numpy(mouth[None,None]).float()`（`audio_visual_test.py:230`）→ FRCNN 输入**始终 float32**，功能无碍，但"batch dtype=float32"不是无条件事实。
- 真实 `.pt` 存储 dtype 仍未知（Q2）；§8.7 用真实权重验证了 uint8 路径，但未取得真实训练 `.pt` 资产。

**验证/测试 loader**（default collate，`batch_size=1`，`shuffle=False`，`drop_last=True`）：

| 张量 | 形状 | dtype | 证据 |
|---|---|---|---|
| `mixtures` | `[1, T]`（多声道数据）；**mono 数据退化为 `[1]`**（0 维张量 collate，见 P0） | `torch.float32` | 实测 D8：mono fixture 得 `(1,)` |
| `sources` | `[1, T]` | `torch.float32` | 实测 |
| `mouths` | `[1, F, 88, 88]` | 依 `.pt` 存储 dtype（uint8→float64；float32→float32），评测脚本随后 `.float()` | 实测 D8（F=62）+ §8.7 真实权重（uint8 .pt → float64） |
| `filenames` | `list[str]` 长度 1 | — | 实测 |
| `n_near` | `[1]` | `torch.int64` | default_collate 把 int 变成 tensor；实测 D8 |

**下游再变形（litmodule/video model，[验证]）**：
- `mouths [B,50,88,88]`（4-D）→ `unsqueeze(1)` → `[B,1,50,88,88]` → `FRCNNVideoModel` → `mouth_emb [B,512,50]`（时间步不降采样，与帧数相等）。证据：`av_litmodule_nass_v2.py:80-87`、`frcnn_videomodel.py:75-83`，实测 `(2,1,50,88,88)→(2,512,50)` f32。
- 评测：`mouth (F,88,88)` numpy → `torch.from_numpy(mouth[None,None]).float()` → `[1,1,F,88,88]` → FRCNN → `[1,512,F]`（变 F 可用）。

### 5.5 `AVSpeechDataModuleWithNear.__init__` vs `configs/Falcon.yml:datamodule.data_config` 逐键对照

YAML 键（`configs/Falcon.yml:156-168`，共 12 个）逐个核对（[验证]：`inspect.signature` + 真实构造，§8.2 I）：

| # | config key | YAML 值 | ctor 参数（默认值） | 存在 | 匹配 | 说明 |
|---|---|---|---|---|---|---|
| 1 | `train_dir` | `--` | `train_dir`（必填） | ✅ | ✅ | 需改成 `$INDEX/tr` |
| 2 | `valid_dir` | `--` | `valid_dir`（必填） | ✅ | ✅ | 需改成 `$INDEX/cv` |
| 3 | `test_dir` | `--` | `test_dir`（必填） | ✅ | ✅ | 需改成 `$INDEX/tt` |
| 4 | `n_src` | 1 | `n_src=1` | ✅ | ✅ | 仅校验 `∈[1,2]`，**下游未使用**；target.json 也只有单 target |
| 5 | `sample_rate` | 16000 | `sample_rate=16000` | ✅ | ✅ | 进 `seg_len` 与 audio model；但 mouth 公式硬编码 16k |
| 6 | `segment` | 2.0 | `segment=4.0` | ✅ | ✅ | 覆盖默认 4.0 → 32000 点 / 50 帧 |
| 7 | `normalize_audio` | false | `normalize_audio=False` | ✅ | ✅ | 当前关闭 |
| 8 | `batch_size` | 4 | `batch_size=4` | ✅ | ✅ | 仅 train loader 生效；val/test 硬编码 1 |
| 9 | `num_workers` | 8 | `num_workers=8` | ✅ | ✅ | — |
| 10 | `pin_memory` | true | `pin_memory=True` | ✅ | ✅ | — |
| 11 | `persistent_workers` | true | `persistent_workers=False` | ✅ | ✅（值覆盖默认） | 需 `num_workers>0`，否则 torch 抛错（P2-11） |
| 12 | `use_near` | true | `use_near=True` | ✅ | ✅ | 控制 near.json 加载与负样本 |
| — | `train_samples` | **缺失** | `train_samples=10000` | ❌ | — | **缺失键，默认 10000 生效**（决定 epoch 样本数，P1-3） |
| — | `data_name` | `AVSpeechDataModuleWithNear` | 不是 ctor 参数 | — | — | 位于 `datamodule:` 层，`audio_train.py:24` 直接 import 类，**该键被忽略** |

**结论：12/12 个 YAML 键都能被 `__init__` 接受，无多余键；唯一“缺失”是 `train_samples`（用默认值）；`data_name` 不参与构造。** `[验证]` §8.2 I1-I4
- `Experiments/checkpoint/Falcon/conf.yml:90-103` 的 `data_config` 键集合与上表**完全相同**（含 `${DATA_ROOT}` 占位符，需编辑）。`[验证]` I4
- `studio/configs/Falcon.yml:155-168` 也带一份相同 data_config（Studio 推理不使用，仅配置快照）。`[代码]`

### 5.6 入口构造方式与 `make_loader`/`make_sets` 一致性

| 入口 | 构造方式 | 证据 |
|---|---|---|
| 训练 | `AVSpeechDataModuleWithNear(**config["datamodule"]["data_config"])` → `.setup()` → `make_loader`（@property，取 3 个 DataLoader） | audio_train.py:48-51 |
| 评测 | 复制 data_config 并剔除 `"use_near","train_samples"` → `AVSpeechDataModuleWithNear(**data_cfg, use_near=False)` → `.setup()` → `make_sets`（@property，取 3 个 Dataset） | audio_visual_test.py:183-189 |

- `make_loader`/`make_sets` 都是 **`@property`**（near.py:331-337），**不是** PL 的 `train_dataloader()` 自动发现机制；每次访问都会**新建** DataLoader（实测 `dm.make_loader[0] is not dm.make_loader[0]`），训练脚本只访问一次，无影响。`[代码]`+[验证]
- README 完全没有提到 `make_loader`/`make_sets`（内部 API，不算错误，但文档缺口）。`[代码]`
- `audio_train.py` 里 `train_loader` 的 `shuffle=True` + Dataset 内部随机选样同时存在，属冗余（不影响正确性）。`[代码]`

### 5.7 与下游模块的契约（需其他子系统文档确认的部分已标注）

| 下游 | 期望输入 | 证据 |
|---|---|---|
| `AudioVisualLightningModuleNASSv2.training_step` | 5 元组 `(mixtures, targets, mouth, _, near_audios)` | av_litmodule_nass_v2.py:73-74 |
| `AudioVisualLightningModule.validation_step` | 5 元组 `(..., n_near_batch)`；`n_near` 可为 tensor/list/scalar；`dataloader_idx==0` 为 cv、`==1` 为 tt（每 10 epoch 算 `test_loss`） | av_litmodule.py:86-119,125-140 |
| `AudioVisualLightningModule.val_dataloader`（PL 接口） | 返回 **`[self.val_loader, self.test_loader]`**（两个 loader 的 dataset 均为 `AVSpeechEvalDataset`） | av_litmodule.py:214-216 |
| **训练期验证 vs 评测** | 共用 `AVSpeechEvalDataset`（整句、5 元组含 `n_near`）；评测只把 `n_near` 丢弃并用 `test_set[idx]` 逐条前向 | `avspeech_dataset_near.py:286-295`；`audio_visual_test.py:189,229` |
| `FRCNNVideoModel.forward` | `[B,1,F,H,W]`（H=W=88），返回 `[B,512,F]` | frcnn_videomodel.py:17-19,75-83；实测 |
| `FalconSpeakerContrastive.forward` | `audio_mixture [B,T]`, `mouth_embedding [B,512,T_v]` | Falcon_Contrastive.py:94-105 |
| `audio_train` 的 audio model | `sample_rate=config.datamodule.data_config.sample_rate`（16000） | audio_train.py:55-58 |
| NASS 对比损失 | `embed_query [B,D]`、`embed_pos [B,D]`、`embed_negs_list list[[K,D]|None]`、`beta` | nass_contrastive_loss.py:31-81 |
| 评测 metrics | 1-D `mix[T]`、`clean[1,T]`、`estimate[n_src,T]`、key str | audio_visual_test.py:229-241 |
| Studio 推理 | `audio (32000,) float32`、`frames (50,96,96) uint8`（内部转 `[1,1,50,88,88]`） | studio/falcon_runtime.py:95-105,248-250 |

### 5.8 单位/精度/帧率约定（全链路）

| 量 | 约定 | 证据 |
|---|---|---|
| 采样率 | 16000 Hz（配置与 backbone/Studio 一致） | configs/Falcon.yml:161；studio/falcon_runtime.py:24 |
| 视频帧率 | 25 fps；每帧 640 采样点（40 ms） | near.py:78,168；studio/falcon_runtime.py:25 |
| 训练段长 | 2.0 s = 32000 采样点 = 50 帧 | near.py:77-78；[验证] C3 |
| 存储嘴部帧 | 96×96（README:132 推断）；Dataset 裁剪到 88×88 | near.py:169-174；transform.py:147-169 |
| 音频 dtype | `float32`（`sf.read(dtype="float32")`） | near.py:160-165 |
| 嘴部 dtype | **条件化**：`.pt` float32 存储 → numpy/tensor float32；uint8 存储 → float64；下游 `.float()`/`type_as` 统一 f32 | near.py:169-174；§5.4；[验证] D3 + §8.7 |
| near 源 wav 采样率 | **必须 16 kHz**（Dataset 不重采样；near.json 只存 frames，按 16 kHz 解释；代码不校验） | `avspeech_dataset_near.py:193,198-203`；`generate_near_json.py:54-60` |
| JSON 数值 | 采样点数 `int`；路径 `str`（绝对） | process_lrs23.py:11,18-28 |
| 通道 | 读取时 `mean(-1)` 归约（隐含多通道数据；见 Q1） | near.py:176-177；avspeech_dataset.py:193-194 |

### 5.9 README 数据准备声明 vs 代码事实

| README 声明 | 代码事实 | 结论 |
|---|---|---|
| `tr/cv/tt`，各含 `mix/`、`target/` wav（16 kHz）与目标说话人嘴部裁剪（README:148） | 实际还需要 `<split>/mouths/<stem>.pt`（4-D channel-last）；**README 未提目录名/格式，也未提 `_meta.json`** | ❌ 文档缺口（P2-3/P2-14，且 meta 缺失已升格 P1-6） |
| 先 `process_lrs23.py` 再 `generate_near_json.py`（README:154-155） | `process_lrs23.py` 已内嵌生成 near.json（:96-105）；第二条命令只做幂等重写 | ⚠ 冗余步骤（P2-3） |
| `near.json` 索引近场干扰人，作为说话人对比损失负样本（README:158） | 语义正确；但“near”由外部 `_meta.json` 定义，far-field 被排除；**README 流程不生成 meta → 全 fallback → 负样本实际为空** | ❌ 按 README 复现时对比损失静默失效（§1.4/P1-6） |
| 编辑 config 的 3 个 dir，脚本不展开环境变量（README:158） | `process_*` 不做变量展开；`audio_visual_test.py` 直接 `yaml.safe_load`（无 expandvars）；`parser_utils.py` 中也无展开逻辑 | ✅ |
| 评测从 `conf.yml` 读数据路径（README:174） | 正确；但 `Experiments/checkpoint/Falcon/conf.yml:100` 是 `${DATA_ROOT}` 占位符，必须手工改 | ✅（README 已提示编辑） |
| 推理 `mouth.npy` 为 `(T,96,96)` 25 fps“与训练数据相同格式”（README:132） | Studio 预处理与训练 val 管线**逐元素一致**（[验证]）；但训练 `.pt` 是 4-D channel-last，而推理是 3-D uint8 | ⚠ 表述不完全一致（J5/Q8） |
| “RealSSA 全 mono 16 kHz”（README:67） | 评测 Dataset `mean(-1)` 隐含多通道输入（mono 会退化 0 维） | ❌/❓ 潜在矛盾（P0/Q1） |
| 训练命令 `python audio_train.py --conf_dir configs/Falcon.yml`（README:162） | 与 `audio_train.py:30-35` 的默认路径不同（默认指向不存在的 NASSv2 yml），但显式传参正确 | ✅ |
| 评测命令 `python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml --gpus 0`（README:165） | 与 `audio_visual_test.py:63-73` 的参数定义一致；`--ckpt_path` 支持 `hf://`（:323-327） | ✅ |

---

## 6. 可验证断言表

| 编号 | 断言 | 证据 path:line | 验证方式 | 前提/条件 | 置信度 |
|---|---|---|---|---|---|
| A1 | 数据布局固定 `{tr,cv,tt}/{mix,target,mouths}`；mouths 由 target 路径字符串替换得到 | `DataPreProcess/process_lrs23.py:8-28,84-99` | 读码 + fixture 跑通 | — | [代码][验证] |
| A2 | `mix.json` 每项 `[abs_wav, n_samples]`（2 元组） | `process_lrs23.py:19-20,31-32` | 运行生成后解析 JSON | — | [代码][验证] |
| A3 | `target.json` 每项 `[abs_wav, abs_mouth_pt, n_samples]`（3 元组） | `process_lrs23.py:21-28` | 同上 | — | [代码][验证] |
| A4 | `near.json` 为“外层与 mix 同序、内层 `[path,n]` 列表”的嵌套 list | `generate_near_json.py:43-71,73-78` | 同上 | — | [代码][验证] |
| A5 | `len(near.json) == len(mix.json)`，fallback 保证不缺项 | `generate_near_json.py:62-73`；`near.py:111-122` | fixture：8/8/8 | — | [代码][验证] |
| A6 | `_meta.json` 代码仅读 `components[].type/path` | `process_lrs23.py:35-43`；`generate_near_json.py:27-36` | 读码 | — | [代码] |
| A7 | 只把 `type=="near"` 组件写入 near.json；far 被排除 | `process_lrs23.py:40-42` | fixture：u0 有 2 near+1 far → K=2 | — | [代码][验证] |
| A8 | 无 meta/仅 far/文件缺失 → fallback `[mix_path, mix_n]` | `process_lrs23.py:69-71` | fixture 6/8 fallback | — | [代码][验证] |
| A9 | JSON 内路径为绝对路径 | `process_lrs23.py:11` | 解析 JSON | — | [代码][验证] |
| A10 | 建索引脚本可重入（覆盖写、两次结果字节相同） | `process_lrs23.py:29-32,75-76` | sha256 对比 | — | [验证] |
| A11 | 短于 `seg_len` 的训练样本被删除，mix/target/near 同步删 | `near.py:111-122` | fixture：1.0 s → 7/8 | — | [代码][验证] |
| A12 | 过滤后 `self.mix` 顺序为 JSON 反序 | `near.py:111-122` | 实测 stems 顺序 | — | [验证] |
| A13 | `__len__` 返回 `samples`（train_samples，默认 10000）而非数据量 | `near.py:147-150`；`near.py:250` | inspect/实测 `len==4`（fixture samples=4） | — | [代码][验证] |
| A14 | `__getitem__` 忽略传入 idx，内部随机选 mix_index | `near.py:154` | monkeypatch randint 取证 | — | [代码][验证] |
| A15 | 训练窗口 `rand_start∈[0,n-seg_len]`，长度恰 32000 | `near.py:156-157` | 强制 rand_start=10000 后逐采样比对 | — | [代码][验证] |
| A16 | `seg_len=int(2.0*16000)=32000`、`fps_len=int(2.0*25)=50` | `near.py:77-78` | 实测属性值 | — | [代码][验证] |
| A17 | `mouth_start=floor(rand_start/640)`，读 50 帧 | `near.py:168-174` | rand_start=10000 → 帧 15；逐帧值比对 | — | [代码][验证] |
| A18 | 嘴部帧加载：4-D channel-last → permute(0,3,1,2) → Grayscale → squeeze → numpy；**3 通道 .pt 路径 [代码]；1 通道仅在自建 torchvision stub 下验证 [推断]（官方行为未验证）** | `near.py:169-174`；`docs/verification/stubs/data_pipe/torchvision/transforms.py`（自写 Grayscale） | 自建 stub 下 1/3 通道均通过；真实 torchvision 未验证 | 3 通道最可信；1 通道依赖真实 torchvision | [代码]+[推断] |
| A19 | 训练管线 `Normalize(0,255)→RandomCrop(88)→Flip(0.5)→Normalize(.421,.165)`；输出 `(T,88,88) f32` | `transform.py:147-160` | 数值/形状断言（max diff 0） | .pt 为 float32 | [代码][验证] |
| A20 | val 管线 `Normalize(0,255)→CenterCrop(88)→Normalize(.421,.165)`；`CenterCrop==[4:92,4:92]` | `transform.py:161-167` | 梯度帧逐元素比对 | — | [代码][验证] |
| A21 | `sample_rate≠16000` 时 mouth_start 仍按 16000 计算 → 错位 | `near.py:168`；`avspeech_dataset.py:108` | 8 kHz 实测：15 vs 31 | 数据 25 fps；sample_rate=16k | [验证] |
| A22 | 嘴部帧不足时返回 <50 帧，collate `torch.stack` 抛 RuntimeError | `near.py:168-174,40-42` | 75/40 帧混批实测 | — | [验证] |
| A23 | 训练 collate 输出：`[B,32000] f32`、`[B,32000] f32`、`[B,50,88,88] f32`、`list[str]`、`list[list[Tensor[32000] f32]]` | `near.py:23-43` | 真实 DataLoader 取 batch | .pt float32→f32；uint8→f64（type_as 兜底） | [验证] |
| A24 | near_audios 内层 K 可变且可为 0 | `near.py:184-220`；`av_litmodule_nass_v2.py:115-132` | 实测 batch 内 `[0,2,0,0]` | — | [代码][验证] |
| A25 | 近场读取偏移 = `min(rand_start, n_samples-seg_len)`；<seg_len 的条目跳过 | `near.py:193-199` | 强制 rand_start 后逐采样比对 | — | [代码][验证] |
| A26 | fallback 条目（path==mix_path）被 Dataset 跳过 | `near.py:188-191` | fixture 中 fallback 样本 K=0 | — | [代码][验证] |
| A27 | `use_near=False` 时不加载 near.json，所有 near 列表为空 | `near.py:94-101,186-187` | 实测 | — | [代码][验证] |
| A28 | `use_near=True` 但 near.json 缺失 → 只打印 WARNING，训练继续（K=0） | `near.py:94-101` | 删除 near.json 实测 | — | [代码][验证] |
| A29 | `normalize_audio` 关闭时不做任何音频归一化（当前 config） | `configs/Falcon.yml:163`；`near.py:179-182` | 读码 + config 解析 | 当前 config normalize_audio=false | [代码] |
| A30 | 若开启归一化：mixture 去均值/unit std，target 复用 mixture 的 std，eps=1e-8 | `near.py:16-20,179-182` | 数值断言（mean≈0, std≈1） | — | [代码][验证] |
| A31 | 静音段归一化结果有限（eps 保护） | `near.py:16-20` | 全零输入实测 max=0 | — | [验证] |
| A32 | 评测 Dataset 不过滤短音频，`len==len(mix.json)` | `avspeech_dataset.py:161-171` | fixture 8/8 | — | [代码][验证] |
| A33 | 评测读整段音频（start=0,stop=None），不切 2 s 窗 | `avspeech_dataset.py:179-187` | fixture 48000 点原样返回 | 数据非 mono | [代码][验证] |
| A34 | 评测嘴部为整段帧经 val 管线裁剪 `(F,88,88) f32` | `avspeech_dataset.py:189-191` | F=75 实测 | 数据非 mono | [代码][验证] |
| A35 | 评测返回 5 元组，第 5 项 `n_near` 来自 mix 同名 `_meta.json` 的 near 计数 | `avspeech_dataset.py:201-212` | fixture：2 near → n_near=2；缺 meta → 0 | 数据非 mono；存在相邻 _meta.json | [代码][验证] |
| A36 | mono wav 进评测 Dataset → `mix/source` 退化为 0 维张量（无 ndim 守卫） | `avspeech_dataset.py:193-194` | mono fixture 实测 `shape=()`；stereo → `(T,)` | 输入为 mono | [验证] |
| A37 | 训练 Dataset 有 ndim 守卫：mono→`(T,)`，stereo→`mean` 后 `(T,)` | `near.py:176-177` | 双 fixture 实测 | — | [代码][验证] |
| A38 | `AVSpeechDataModuleWithNear.__init__` 有 13 个参数、12 个 config 键全部命中，唯一缺席 `train_samples`（默认 10000） | `near.py:237-252`；`configs/Falcon.yml:156-168` | inspect.signature + 构造 | — | [验证] |
| A39 | `Experiments/checkpoint/Falcon/conf.yml` 的 data_config 键集合与 `configs/Falcon.yml` 相同 | `Experiments/checkpoint/Falcon/conf.yml:90-103` | YAML 解析对比 | — | [验证] |
| A40 | `make_loader`/`make_sets` 都是 `@property`，返回 3 元组，每次访问新建 | `near.py:331-337` | 属性类型与对象同一性断言 | — | [代码][验证] |
| A41 | 训练 loader：`batch_size=4`、`shuffle=True`、`drop_last=True`、`collate_fn=collate_with_near` | `near.py:297-307` | 读码 + DataLoader 取 batch | — | [代码][验证] |
| A42 | val/test loader 硬编码 `batch_size=1`（config 的 4 被忽略） | `near.py:309-329` | 实测 batch 形状 `[1,...]` | — | [代码][验证] |
| A43 | `setup()` 同时构建 3 个 dataset → 只跑 test 也要求 train/valid 的 JSON 存在 | `near.py:277-295`；`audio_visual_test.py:187-189` | 故意指向不存在目录 → FileNotFoundError | — | [代码][验证] |
| A44 | `audio_visual_test` 的 data_config 构造：剔除 `use_near/train_samples` 后强制 `use_near=False` | `audio_visual_test.py:183-189` | 复刻该 7 行真实可构造 | — | [代码][验证] |
| A45 | 评测不使用 DataLoader，逐条 `test_set[idx]`（无 collate） | `audio_visual_test.py:192-193,229-231` | 读码 | 数据非 mono | [代码] |
| A46 | `tensors_to_device(tuple)` 返回 list（tuple→list），tensor 迁移，str/int 原样 | `look2hear/utils/torch_utils.py:19-40` | 实测类型 | — | [代码][验证] |
| A47 | litmodule 训练解包 5 元组，mouth 4-D → unsqueeze(1) → FRCNN | `av_litmodule_nass_v2.py:73-90` | 读码 + FRCNN 前向实测 | — | [代码][验证] |
| A48 | FRCNN 输入 `[B,1,50,88,88]` → 输出 `[B,512,50]` f32（时间步不降采样） | `frcnn_videomodel.py:75-83` | 真实模型 CPU 前向实测 | 输入 H=W=88（dataset 已裁） | [验证] |
| A49 | 近场负样本在训练中全部使用（不采样/不分桶）；`bucket=min(K,4)` 仅为日志分桶 | `av_litmodule_nass_v2.py:19,113-118,182-183` | 读码 | — | [代码] |
| A50 | K=0 → `l_neg=0`（仅正项）；K=1 → relu 求和；K>1 → softmax(β·sim) 加权 | `nass_contrastive_loss.py:56-74` | 读码（+ loss 文档 R2 核对） | — | [代码] |
| A51 | `near` 定义唯一依据是 `_meta.json` 的 `type=="near"`；仓库内无距离/SIR 阈值 | `process_lrs23.py:35-43`；`generate_near_json.py:27-36` | 全仓库 grep + 读码 | — | [代码] |
| A52 | `process_lrs23.py` CLI 2 参数；`generate_near_json.py` 3 参数且 `--sample_rate` 未使用 | `process_lrs23.py:108-121`；`generate_near_json.py:82-90` | `--help` + 传 8000 输出字节不变 | — | [代码][验证] |
| A53 | 空 split → 写 `[]` 后 `min([])` 抛 `ValueError`（脚本非 0 退出） | `generate_near_json.py:73-78` | 空目录实测 | — | [验证] |
| A54 | 缺输入目录 → 未捕获 `FileNotFoundError` | `process_lrs23.py:12` | 实测 | — | [验证] |
| A55 | 缺 `<split>/mix.json`（独立脚本）→ 打印 `Skipping` 继续 | `generate_near_json.py:97-99` | 实测 | — | [代码][验证] |
| A56 | mix/target 等量但文件名错位 → 生成脚本静默错配；Dataset 亦不报错 | `process_lrs23.py:12-13`；`near.py:103-127` | 构造 a/c vs b/c 实测 | — | [验证] |
| A57 | JSON 表长度不一致 → Dataset 构造期 `IndexError`（无显式校验） | `near.py:111-122` | 截短 target/near.json 实测 | JSON 表长度不一致 | [验证] |
| A58 | `check_avspeech_dataset.py` 作为脚本运行即 FileNotFoundError（写死 `${DATA_ROOT}`） | `check_avspeech_dataset.py:317` | 实际运行 | — | [验证] |
| A59 | `check_avspeech_dataset.py` 是旧副本（无 near 类、无 ndim 守卫），无 CI/入口引用 | `check_avspeech_dataset.py:30-310` | 读码 + grep | — | [代码] |
| A60 | Studio `preprocess_mouth` 与训练 val 管线**在 float32 精度内数值等价**（uint8 `.pt`：raw `max|Δ|=2.574e-07`、cast 后 2.384e-07）；**仅 float32 `.pt` 时逐位一致（0.0）**；等价性只覆盖固定 `(50,96,96)` 窗口（变长 F 被 Studio 拒绝） | `studio/falcon_runtime.py:95-105,99-100`；`transform.py:161-167` | 两边真实代码交叉运行（两种 `.pt` dtype） | uint8→精度内；float32→逐位 0.0；F=50 | [代码][验证] |
| A61 | Studio 常量：16 kHz / 25 fps / 32000 / 50 帧 / 96×96 | `studio/falcon_runtime.py:24-27` | 读码 | — | [代码] |
| A62 | `datas/__init__.py` 只导出旧 `AVSpeechDataModule`；入口脚本按完整路径 import near 模块 | `look2hear/datas/__init__.py:1-5`；`audio_train.py:24`；`audio_visual_test.py:46` | grep | — | [代码] |
| A63 | `audio_train.py:51` 用 `datamodule.make_loader`（无括号），`audio_visual_test.py:189` 用 `make_sets` | 两文件行号 | 读码 | — | [代码] |

| A64 | **`test_loader` 是第二个 validation loader**：`val_dataloader()` 返回 `[val_loader, test_loader]`；`trainer.fit` 每 epoch 遍历 [cv,tt]，每 10 epoch 在 tt 上计算/记录 `test_loss`；`trainer.test()` 从未被调用 | `av_litmodule.py:214-216,125-140,174-181`；`audio_train.py:180` | 读码 + train-sys 11-epoch 实测（03 文档 V4） | 每 epoch 遍历；test_loss 仅 epoch%10==0 | [代码][验证] |
| A65 | **README 流程缺 `_meta.json` → near.json 全 fallback → Dataset 跳过哨兵 → K=0** → NASS 负项=0、MDCL 恒 0、val 分桶全 `0near`（§1.4 六步链） | `process_lrs23.py:69-71`；`generate_near_json.py:62-65`；`avspeech_dataset_near.py:188-191`；`av_litmodule_nass_v2.py:115-122`；`nass_contrastive_loss.py:73-74`；`av_litmodule_mdcl.py:114-126`；`avspeech_dataset.py:201-211`；`av_litmodule.py:161-169` | 源码链 + fixture G3 实测 | README 数据流程且无外部 meta 生成器 | [代码][验证] |
| A66 | near.json 只存 `[path, frames]`，**不存 sample_rate**；Dataset 按 16 kHz 解释 frames/seg_len 并 `sf.read(start,stop)`，不重采样、不校验 | `generate_near_json.py:54-60`；`process_lrs23.py:61-67`；`avspeech_dataset_near.py:193,198-203` | 读码 | near 源 wav 非 16 kHz 时成立 | [代码] |
| A67 | **训练期验证与评测共用 `AVSpeechEvalDataset`**（整句、5 元组含 `n_near`）；训练 `val_dataloader=[cv,tt]`，评测取 `make_sets[2]` 并丢弃 `n_near` | `avspeech_dataset_near.py:286-295`；`av_litmodule.py:214-216`；`audio_visual_test.py:189,229` | 读码 + 实测 | — | [代码][验证] |
| A68 | **真实 `falcon.pt` 顶层 = `{state_dict, architecture, epoch}`**，662 keys（525 audio_model.* + 137 video_model.*），strict 加载 OK；真实权重下整句评测 `T=32000/F=50` 与 `T=48513/F=75` 均输出 `len==T`、finite | `/root/.cache/falcon_weights/falcon.pt`（sha256 `a1888b5a…`）；`§8.7` | 真实权重加载 + 前向（CPU，无网络） | 权重来自 Lead 提供的本地缓存；未做完整 CSV 评测 | [验证] |
（共 68 条；R3 新增 A64-A68，并为全表新增“前提/条件”列；断言编号与 §8 的验证日志对应，A64-A68 见 §8.7/§9.1。）

---

## 7. 未解疑问 / 需其它子系统确认的问题

| 编号 | 问题 | 为什么重要 | 需谁确认 |
|---|---|---|---|
| **Q1** | 实际训练/评测数据（LRS2/LRS3/VoxCeleb2/RealSSA）的 wav 是**单声道还是多声道**？README:67 声称 RealSSA 为 mono 16 kHz，但 `AVSpeechEvalDataset` 无守卫 `mean(-1)`（0 维退化），而训练集有 ndim 守卫 | 决定 P0 是“真 bug”还是“数据隐式多通道”；直接决定评测入口是否可用 | Lead / 数据提供方；loss-metric-eval 交叉确认 |
| **Q2** | `mouths/<stem>.pt` 由哪个工具生成？准确格式是 `(T,96,96,1)` 还是 `(T,96,96,3)`、fps、与音频时间轴如何对齐？仓库内无生成脚本 | 数据准备可复现性；训练集与推理 `(T,96,96) uint8` 的一致性（A18/A60） | Lead（外部数据资产）；studio-backend |
| **Q3** | `_meta.json` 的完整 schema 与生成脚本；RealSSA 中 **near/far 的距离阈值、SIR 定义**；“near” 是按距离还是按 SIR 判定？ | 负样本语义与论文口径；near 定义不在仓库（A51） | Lead / RealSSA 数据提供方 |
| **Q4** | `train_samples=10000` 是训练机上有意为之还是缺省巧合？是否应写入 `configs/Falcon.yml` / 实验 conf？ | 影响复现的 epoch 大小与总训练量（P1-3） | train-sys / Lead |
| **Q5** | 训练用随机 2 s 段、评测用整句（A33/A45），论文指标按哪种口径统计？SI-SDR 是整句还是 2 s 窗口平均？ | 指标可比性；worst-case 评测结果解读 | loss-metric-eval（04 文档） |
| **Q6（R3 结案：文档层）** | 按 README 数据流程复现时，LRS2/LRS3/VoxCeleb2/RealSSA 训练数据**是否另有未开源的 `_meta.json` 生成器**？若无，near.json 全 fallback → K=0 → NASS 负项恒 0、MDCL 关闭、val 分桶全 `0near`（§1.4/P1-6 已把该后果链定死） | 论文贡献之一是否在实际训练中生效；README 流程无法复现该贡献 | 数据提供方（是否有生成器）/ Lead |
| **Q7** | `n_src: 1` 与 target.json 单 target 的契约：模型是否只支持单目标？多目标数据如何组织？ | DataModule 接受 `n_src∈{1,2}` 但未使用（A38），文档易误导 | model-core / train-sys |
| **Q8** | Studio 的 `mouth.npy (T,96,96) uint8` 是浏览器端抽取格式；训练 `.pt` 是 4-D channel-last（存储 dtype 也未知）。两者抽取工具（帧率/对齐/裁剪窗口/dtype）是否完全一致？ | 训练-推理分布一致性（A60 只证明预处理数学一致，不证明抽取一致） | studio-backend / studio-frontend |
| **Q9** | 评测数据目录是否同样需要 `mix/*_meta.json`（n_near 统计依赖）？评测发布的 RealSSA 是否带 meta？ | val 分组指标（0near..4plus）与 n_near 解析（A35） | loss-metric-eval / Lead |
| **Q10** | `num_workers=8 + persistent_workers=true` 与 near 列表在 worker 中的反序列化/`warnings.warn` 行为是否符合预期？是否有 worker 泄漏（`persistent_workers=True` + 每次 `make_loader` 新建）？ | 训练稳定性/性能；本轮验证用 num_workers=0 未覆盖 | train-sys（R2 交叉核对） |
| **Q11** | 训练机上 `conf.yml` 的 `${DATA_ROOT}` 占位符是否有外部展开工具（README 声称脚本不展开）？ | 评测可操作性（A44/B 组） | ops-tests-docs / Lead |
| **Q12** | `check_avspeech_dataset.py` 是要保留的调试工具还是可删的历史遗留？ | 代码卫生；若保留需修掉写死路径 | ops-tests-docs |
| **Q13** | RealSSA 评测集是否严格按 `{tr,cv,tt}/{mix,target,mouths}` 布局发布？real-world 录制 split 的 mouth 抽取是否同样 25 fps？ | 评测入口能否直接跑（A43） | Lead / 数据提供方 |
| **Q14（R3 修正）** | 评测完全不经过 DataLoader（A45），故 `batch_size/num_workers/pin_memory/persistent_workers` 在**评测**中均不生效（正确）。但 **`test_loader` 不是死路径**：训练期 `val_dataloader()` 返回 `[val_loader, test_loader]`，每 epoch 遍历、每 10 epoch 记 `test_loss`（`av_litmodule.py:214-216,125-140`；P2-8 已修正）。是否应在 README/训练文档中明确“训练期 tt 集会参与验证循环”？ | 配置键语义与训练期行为误解 | train-sys / ops-tests-docs |
| **Q15** | 是否要求 `target/<stem>.wav` 与 `mix/<stem>.wav` 采样点严格等长（target.json 的 `n_samples` 未被校验，P1-2）？数据生成端是否有保证？ | 训练批次稳定性 | 数据提供方 / Lead |
| **Q16** | near 源 wav 的采样率是否保证 16 kHz？near.json 不写 sample_rate、Dataset 不重采样也不校验（P1-7）；若不能保证，建议数据端写入采样率或建索引时校验/统一 | 近场负样本时间轴与内容正确性 | 数据提供方 / Lead |

---

## 8. 本轮验证活动记录

### 8.1 环境与手段

- 解释器：`/root/.venvs/falcon/bin/python`（Python 3.10.18，torch 2.8.0+cu128，numpy 2.2.6，soundfile 0.13.1，pandas 2.3.3，tqdm 4.67.1，pyyaml，pytorch_lightning 2.6.6；**缺 torchvision/cv2/matplotlib**）。
- 仓库 `docs/verification/stubs/` 只有 `timm`/`typeguard` 两个 stub（见 `docs/understanding/VERIFY-ENV.md:17-34`），不足以 import `look2hear/datas/*`（需要 `torchvision.transforms`、`cv2`、`matplotlib`）。
- 因此**新建了 /tmp/dp-verify/stubs/**（不写入仓库、不修改源码）：
  - `torchvision/transforms.py`：`Grayscale`（按 torchvision `rgb_to_grayscale` 的 ITU-R 601-2 亮度权重实现，支持 `(...,C,H,W)`）、`Compose/Normalize/RandomCrop/CenterCrop`；
  - `cv2.py`：`cvtColor`/`flip` 最小实现（数据路径实际只用 `Grayscale`，`RgbToGray` 不被调用）；
  - `matplotlib`：空壳 + `axes/figure/colors/pyplot`（仅为满足 `avspeech_dataset.py:15` 的 import 及 torchmetrics 的探测）。
  - **保真边界**：`look2hear/datas/transform.py` 的变换类是纯 numpy 代码，验证时**原样使用仓库代码**；只有 `torchvision.transforms.Grayscale` 是 stub（形状/通道语义与官方一致；官方实现不可在沙箱运行）。`Normalize/Crop` 等若被 stub 之外的官方版本替换，语义相同。
- 合成数据：`/tmp/dp-verify/make_fake_data.py` 生成
  - `$DATA/data/tr`: `u0`（meta: 2 near + 1 far）、`u1`（无 meta）、`u2`（meta 仅 far）、`u3`（meta 指向不存在的 near 文件）、`u4`（meta: 短 near 20000 点 + 正常 near 48000 点）、`u5`（1.0 s，短于 2 s）、`u0copy`（3 通道 mouth .pt）、`u0grad`（梯度帧，用于验证 CenterCrop）；
  - `cv`/`tt` 各 2 条 2.5 s；
  - 近场 wav 为 ramp 信号，便于精确校验读取偏移；mouth .pt 每帧填充常量 `f`，便于精确校验帧号/归一化。
- 运行命令（全部在仓库根执行；fixtures 仍写在 `/tmp/dp-verify`，可重复生成；stub/脚本使用仓库内路径，R3.1 起不再依赖 `/tmp` 的 stub）：
  ```bash
  export PYTHONPATH=$PWD/docs/verification/stubs:$PWD/docs/verification/stubs/data_pipe:$PWD
  PY=/root/.venvs/falcon/bin/python
  # 1) 合成数据 + 建索引（真实 DataPreProcess 代码）
  $PY docs/verification/scripts/data_pipe/make_fake_data.py
  $PY DataPreProcess/process_lrs23.py \
      --in_audio_dir /tmp/dp-verify/data --out_dir /tmp/dp-verify/index
  # 2) 数据管线动态验证（57 条断言）
  $PY docs/verification/scripts/data_pipe/verify_dataset2.py
  # 3) DataPreProcess + check 脚本验证（19 条断言）
  $PY docs/verification/scripts/data_pipe/verify_preprocess.py
  ```
  日志（补交版）：`docs/verification/runs/data_pipe/verify_dataset.log`（57 PASS / 0 FAIL）、`verify_preprocess.log`（19 PASS / 0 FAIL）；运行时输出与 `/tmp/dp-verify/logs/` 原文一致。

### 8.2 数据集动态验证（`verify_dataset2.py`，57/57 PASS）

覆盖点与关键输出：
- **A 组（变换数学）**：A1-A9 PASS。`(75,96,96,1)` mouth → `(75,88,88) float32`；训练管线输出与手算 `(x/255-0.421)/0.165` `max|diff|=0.00e+00`；val `CenterCrop(88)` 与 `[4:92,4:92]` `max|diff|=0.00e+00`；训练 RandomCrop 不改变帧数（75→75）。
- **B 组（索引/过滤对齐）**：B1-B8 PASS。1.0 s 样本被删（8→7）；mix/target/mouth 三表 stem 对齐；u0 近场 K=2（far 被排除）；fallback=6 条；`len(dataset)==samples==4`；保留顺序为 JSON 反序 `['u4','u3','u2',...]`。
- **C 组（`__getitem__` 对齐，monkeypatch `random.randint`）**：C1-C6 PASS。`rand_start=10000` 时 mixture/source 与 `sf.read(start=10000,stop=42000)` **逐采样相等**；mouth 首帧 = `/255-.421)/.165 = -2.19501`（= 帧 15 的归一化值）；返回 `[50,88,88] float32`；filename=basename；u4 近场短文件被跳过、正常文件按 `min(10000, 48000-32000)=10000` 读取（逐采样相等）。
- **D 组（collate/batch）**：D1-D8 PASS。`(3,32000)`/`(3,32000)`/`(3,50,88,88)` 全 `torch.float32`；filenames `list[str]`；`near_audios` 逐 batch K 分布 `[0,2,0,0]`、元素形状 `(32000,)`；`len(dataset)=samples` → 4 样本/bs3 得批次 `[3,1]`；config 的 DataModule train loader 得 `(4,32000)` 5 元组；val loader 得 `(1,)` mix 与 `(1,62,88,88)` mouth、`n_near=tensor([0])`。
- **E 组（归一化）**：E1 PASS。mix 段 `mean≈-1.55e-08 / std=1.0000`；target 用 **mixture 的 std**（target 自身 std 0.4796 → 归一化后 0.9955）。
- **F 组（评测 Dataset）**：F1-F8 PASS。不过滤（8/8）；mono wav → `mix.shape=()`（0 维，P0 证据）；stereo → `(40000,)`；整段 mouth `(75,88,88) f32`；`n_near=2`（u0）；缺 meta → 0；gradient mouth 与 `[4:92,4:92]` 公式逐元素一致。
- **G 组（use_near）**：G1-G3 PASS。`use_near=False` → near 全空且不读文件；`use_near=True` + near.json 缺失 → WARNING 后近场全空（训练可继续）。
- **H 组（边界/缺陷复现）**：H1 PASS（target.json 截短 → `IndexError: list assignment index out of range`）；H2 PASS（near.json 截短 → 同错）；H3 PASS（40 帧 vs 75 帧 mouth → collate `RuntimeError: stack expects each tensor to be equal size`）；H4 PASS（8 kHz：seg_len=16000，mouth_start=15 而正确应为 31）；H5 PASS（全零信号归一化后 max=0、有限）。
- **I 组（config/接口）**：I1-I7 PASS。`inspect.signature` 13 参数 vs YAML 12 键：缺失键 `[]`、仅 `train_samples`（默认 10000）不在 YAML；`Experiments/checkpoint/Falcon/conf.yml` 键集合一致；`make_loader`/`make_sets` 均 property 且每次访问新建。
- **J 组（评测构造路径）**：J1-J3 PASS。复刻 `audio_visual_test.py:183-189` 的过滤+构造+setup+make_sets 成功；只跑 test 但 valid_dir 不存在 → `FileNotFoundError: .../cv/mix.json`（A43 证据）；`tensors_to_device(tuple)` → list[str,int,ndarray,Tensor,Tensor]。

### 8.3 预处理脚本验证（`verify_preprocess.py`，19/19 PASS）

- #1-#2：`process_lrs23.py` 退出码 0；重跑输出 9 个 JSON **字节级相同**（幂等/覆盖）。
- #3-#8：schema 与语义：mix 2 元组、target 3 元组（mouth 路径 `.../tr/mouths/u0.pt`）、near 嵌套 list 且与 mix 等长（K 序列 `[2,1,1,1,1,1,2,1]`）、只收 near（far 排除）、无 meta/缺文件/仅 far → 6 条 fallback（下标 `[1,2,3,4,5,7]`）。
- #9：`--in_audio_dir` 指向不存在目录 → 未捕获 `FileNotFoundError: '.../tr/mix'`（非 0 退出）。
- #10：空 split → `mix.json`/`near.json` 写成 `[]`，随后 `min() arg is an empty sequence` 崩溃（非 0 退出）。
- #11-#14：独立 `generate_near_json.py` 退出 0，输出与内嵌生成**字节相同**；`--sample_rate 8000` 输出仍相同（参数未使用）；缺 `cv/tt/mix.json` → `Skipping cv/tt`，`tr/near.json` 仍生成。
- #15-#16：mix=`[a,c]` / target=`[b,c]`（等量错名）→ 脚本退出 0 生成错配 JSON；Dataset 正常构造且 `mix=['c','a'] vs target=['c','b']` → **静默错配**（P1-1 证据）。
- #17：`look2hear/datas/check_avspeech_dataset.py` 直接运行 → `FileNotFoundError: '${DATA_ROOT}/av-tiger-lrs3/DataPreProcess/LRS3_multi/tr/mix.json'`。
- #18-#19：`--help` 只列参数、不显示默认值（`process_lrs23.py` 默认 `--` 需读源码确认）；`generate_near_json.py` 三参数齐全。

### 8.4 补充交叉验证（真实代码，非 stub 替代部分）

1) **Studio vs 训练仓库嘴部预处理（R3.1 条件化）**：用两边真实代码、同一 `(50,96,96)` 帧，两种 `.pt` 存储 dtype：

   | 训练侧输入 | train 输出（`["val"]` 管线） | studio 输出 | raw `max|Δ|` | cast f32 后 `max|Δ|` |
   |---|---|---|---|---|
   | **uint8**（真实推理格式） | `(50,88,88)` **float64** | `(1,1,50,88,88)` float32 | **2.574e-07** | **2.384e-07**（`array_equal=False`） |
   | **float32**（假设 `.pt` 存 float32） | `(50,88,88)` float32 | `(1,1,50,88,88)` float32 | **0.0（逐位相等）** | 0.0 |

   说明：`get_preprocessing_pipelines()["val"]` 实际返回 **`(50,88,88)`**；`(1,1,·)`/float32 是对比时的手工包装（`torch.from_numpy(out)[None,None].float()`；Studio 自身返回 `[1,1,·]`）。**A60 的正确结论是“float32 精度内数值等价（uint8 输入时 raw 2.574e-07 / cast 2.384e-07）；仅当 `.pt` 以 float32 存储时逐位一致（0.0）”**。
   **边界**：`preprocess_mouth` 断言输入恰为 `(50,96,96)`（`studio/falcon_runtime.py:99-100`），实测 `(75,96,96)` 抛 `ValueError: Expected 50 grayscale 96×96 mouth crops; got (75, 96, 96).` → **A60 的等价性只覆盖 2 s 固定 50 帧窗口，不覆盖 §8.7 的变长 F 评测路径**（Studio helper 不能直接复用于 F=75）。`[验证]`
2) **FRCNN 输入契约**：`look2hear/videomodels/frcnn_videomodel.py` 真实实例（`pretrain=None`，CPU）：
   ```
   input (2,1,50,88,88) → mouth_emb (2,512,50) torch.float32
   ```
   验证 A48，并说明 `[B,50,88,88] → unsqueeze(1)` 的契约正确。
3) **静态审计**：`grep` 全仓库确认 `AVSpeechDataModuleWithNear` 仅被 `audio_train.py:24,49`、`audio_visual_test.py:46,187` 使用；旧 `AVSpeechDataModule`/`AVSpeechTrainDataset` 除 `datas/__init__.py` 与 `check_avspeech_dataset.py` 外无消费者；`requirements.txt` 含 `torch`/`torchvision`/`opencv-python`/`matplotlib`/`soundfile`/`pandas`/`tqdm`（`requirements.txt:5,7,11,13-14,23-24`，torchvision 在 :7），与 `datas/` 的 import 一致。

### 8.5 未能验证 / 验证边界

- **没有真实 LRS2/LRS3/RealSSA 数据、没有 `_meta.json` 样例、没有真实训练 mouth `.pt` 资产**：`.pt` 的准确通道数/帧率/抽取方式/存储 dtype 仍是**从 `permute(0,3,1,2)` 与 README 反推**（Q2）。**A18 的“1 通道 .pt 可用”仅在自写 torchvision stub 下成立（其 `Grayscale` 显式 `if img.shape[-3]==1: return img`）**，真实 torchvision 行为未验证 → R3/R3.1 已把该断言降级为 **[代码]+[推断]**（§6 A18 置信度列已同步），3 通道→1 通道路径最可信。
- **真实权重已可用（R3 新增）**：`/root/.cache/falcon_weights/falcon.pt`（47 MB）与 `falcon-big.pt` 已落盘（非本仓库资产）；本轮用它完成了**加载 + 整句评测前向形状**验证（§8.7），但**仍未做端到端训练/完整评测**（缺数据与指标依赖）。
- **未用真实 `num_workers=8 + persistent_workers=true`**：本轮用 `num_workers=0, persistent_workers=False`（否则 torch 直接报 `persistent_workers option needs num_workers > 0`）。config 组合本身合法（config:165-167），但多 worker 行为（尤其 near 列表序列化/warnings）未测（Q10）。
- **评测入口未真正运行完整流程**：`audio_visual_test.py` 依赖 `pystoi/pypesq/ptflops` 等与完整数据目录；本轮复刻其数据构造 7 行（J1-J3）+ 用真实权重跑逐条前向（§8.7），未跑 CSV/metrics 全流程。
- **`Normalize(0.0,255.0)` 的 dtype 细节**：实测 numpy 2.2 下 uint8→float64、float32→float32；旧版 numpy 的标量提升规则不同，但 numpy≥1.20 下结论一致（[推断]）。
- **未修改任何源码**；验证脚本/日志已按 VERIFY-ENV §4 补交到仓库（P2-9）：`docs/verification/scripts/data_pipe/`、`docs/verification/runs/data_pipe/`、`docs/verification/stubs/data_pipe/`（原 `/tmp/dp-verify/` 保留）。

### 8.6 R2-prep 补充验证：mono-eval P0 复现 + 评测入口数据路径契约（2026-10-21）

> 完整版见 `docs/verification/data-pipe-supplement.md`；脚本 `docs/verification/scripts/data_pipe/verify_p0_eval.py`、日志 `docs/verification/runs/data_pipe/verify_p0_eval.log`（原 `/tmp/dp-verify/` 副本保留）。

**P0 已用真实模型坐实**（[验证]）：
- mono fixture（1 声道 wav，40000 点）→ `AVSpeechEvalDataset` item：`mixture.shape=()`、`source.shape=()`（`avspeech_dataset.py:193-194` 无守卫的 `mean(-1)` 把一维数组对时间求均值）、`mix[None].shape=(1,)`；
- 复刻 `audio_visual_test.py:229-231`：视频模型正常（`(1,1,62,88,88)→(1,512,62)`），音频模型在 `torch.stft` 抛
  `RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (128,128) at dimension 2 of input [1,1,1]`
  （链路 `Falcon.py:1095-1097` reshape `(1,)→(1,1)` → `Falcon.py:1110-1116` `torch.stft(n_fft=256, center=True)`）；
- 该异常不会被吞：`audio_visual_test.py:231` 的 `model(...)` 在 `try`（:233 起，:237 只捕获 ValueError）之外 → **README:165 的评测命令若数据为 mono 必然在第一个样本崩溃**；
- 训练期验证路径同样受影响：`default_collate` mono item → `mixture (1,)`、`mouth (1,62,88,88)` → 同一 RuntimeError（`avspeech_dataset_near.py:286-290,309-318`；`av_litmodule.py:86-119`）→ 反证“能训练成功则真实数据应为多声道”，README:67 “RealSSA … mono 16 kHz” 的表述需核实（Q1）；
- 修法：`avspeech_dataset.py:193-194` 改为与 `avspeech_dataset_near.py:176-177` 相同的 ndim 守卫（mono 保持 1-D、多声道才 mean）；实测修复后 `fixed_mix (40000,)`、`est_sources (1,1,40000)` 正常。另建议在 dataset/DataModule 启动时打印或断言首样本形状。

**评测入口数据路径断言（供 R2 评审 04 使用，[代码]+[验证]）**：
- AST 审计 `audio_visual_test.py`：`DataLoader`/`train_dataloader`/`val_dataloader`/`test_dataloader`/`make_loader` 均不出现；只有 `setup`/`make_sets`/`tensors_to_device`（`:189,192-193,229-231`）→ 逐条 `test_set[idx]`、无 collate、无 batch；
- eval dataset 输出：`mix (T,) f32`、`src (T,) f32`、`mouth (F,88,88) f32 numpy`、`key str`、`n_near int`（多声道数据；mono 即 P0）；
- **eval 无 2 s 截断/填充**：dataset 整段读（`avspeech_dataset.py:179-191`）；对齐由模型完成——`NeuroSCNetFusion` 把视频 `F` 帧 nearest 插值到音频 `T'` 步（`Falcon.py:257-258`），`STFTDecoder` 以 `istft(length=input_shape[-1])` 精确还原输入长度（`Falcon.py:1166,1182-1188`）；实测 `T=48513, F=75 → T'=380 → out T=48513`（≈5 个音频步/25fps 帧，640/128）；
- `use_near=False` 只作用于 train（`near=[]`，不读 near.json）；eval 仍是 `AVSpeechEvalDataset`、仍读 `_meta.json` 的 `n_near`（评测中丢弃）；`setup()` 构建 3 个 dataset → 只评测也要 train/valid 的 `mix.json`+`target.json`；near.json 不参与评测（实测删除全部 near.json 后 `setup()` 正常）。
- **dtype 补充（评审 04 时发现的重要细节）**：mouth 经 transform 管线后的 dtype **取决于 `.pt` 存储 dtype**——`uint8` 存储在 numpy `x - 0.0` 处提升为 **float64**（`docs/verification/scripts/verify_eval_flow_04.py:177` 的 fixture 即如此；§8.7 用真实权重 + uint8 `.pt` 复现亦得 float64），`float32` 存储则保持 **float32**（本文件 A5/D3 的 fixture）。真实 `.pt` 格式未知（Q2）；下游已统一：评测 `torch.from_numpy(mouth[None,None]).float()`（`audio_visual_test.py:230`）、训练 `mouth_in.type_as(mixtures)`（`av_litmodule_nass_v2.py:87`）→ FRCNN 输入始终 float32。§5.4 的 dtype 已按此条件化（R3）。

### 8.7 R3 真实权重验证：falcon.pt 格式 + 整句评测前向（2026-10-21）

> 资产：`/root/.cache/falcon_weights/falcon.pt`（47,156,343 B，sha256 `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c`；Lead 提供，非本仓库资产）。
> 脚本：`docs/verification/scripts/data_pipe/verify_realweight_eval.py`（原 `/tmp/dp-verify/verify_realweight_eval.py`）；日志：`docs/verification/runs/data_pipe/verify_realweight.log`。
> 命令：`PYTHONPATH=$PWD/docs/verification/stubs:$PWD/docs/verification/stubs/data_pipe:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/data_pipe/verify_realweight_eval.py`（CPU，~4 s，不联网/不训练）。

**[1] `falcon.pt` 顶层格式**（支持 `studio/falcon_model/PROVENANCE.md:36-40` 与 `README:107` 的说法）：
```
torch.load type : dict
top-level keys  : ['state_dict', 'architecture', 'epoch']
architecture    : FalconSpeakerContrastive
epoch           : 714
state_dict size : 662  (audio_model.*: 525, video_model.*: 137, 其他: 0)
```
→ `audio_visual_test.py:119-120` 的 `ckpt["state_dict"]` **与发布文件兼容，不会 KeyError**（这是对 04 文档 Q1/B5 的实证关闭，见 `docs/reviews/round1/04-losses-metrics-video-review-by-data-pipe.md` P1-1）；剩余阻断仍是 B2/B3（conf.yml 类名 / `${DATA_ROOT}`）。

**[2] 严格加载**：从 `configs/Falcon.yml` 构建 `FalconSpeakerContrastive`（audio）+`FRCNNVideoModel(pretrain=None)`（video），`load_state_dict(strict=True)` 均 OK；参数 537,697 + 11,186,688 = 11,724,385，与 README:100 一致。

**[3] 评测整句前向（复刻 `audio_visual_test.py:229-231`，真实 Dataset + 真实权重，2 声道 wav + uint8 `(F,96,96,1)` .pt）**：

| 样本 | T（采样点） | mouth F | Dataset item | Fusion 输入（audio ← video） | 输出 |
|---|---|---|---|---|---|
| `seg2s` | 32000 | 50 | `mix(32000,) f32`、`mouth(50,88,88) float64` | `audio (1,112,251,129) ← video (1,512,50)` | `est (1,1,32000) f32`，`len==T`，finite |
| `odd` | 48513 | 75 | `mix(48513,) f32`、`mouth(75,88,88) float64` | `audio (1,112,380,129) ← video (1,512,75)` | `est (1,1,48513) f32`，`len==T`，finite |

→ 结论：**真实权重下，评测整句语义成立**：T 任意、F 任意；视频帧在 `NeuroSCNetFusion` 内被 nearest 插值到音频时间步（`Falcon.py:257-258`），解码器 `istft(length=input_shape[-1])` 精确还原输入长度（`Falcon.py:1166,1182-1188`）；`T'/F ≈ 380/75 ≈ 5.07`（≈640/128 个音频步/视频帧）。同时实测 uint8 `.pt` → Dataset mouth **float64**（下游 `.float()` 转 f32），坐实 §5.4 的 dtype 条件化结论。

---

## 9. 变更日志与评审响应（R3）

### 9.1 R3 评审响应（逐条，对应 `docs/reviews/round1/05-data-pipeline-review-by-train-sys.md`）

> 评审锚点已核对：评审开始 `sha256=f1c90fa3…`（777 行）→ 评审结束 `fde1ed6a…`（797 行，作者追加 §8.6）；本 R3 修订基于其后 `dcd32572…`（799 行）版本，定稿 sha256 由复核人 `sha256sum` 计算（本文件不自引用哈希）。评审的 P0/P1/P2 全部采纳或部分采纳，无拒绝项。

| 评审编号 | 意见摘要 | 处理 | 落点 |
|---|---|---|---|
| **P0-1** | `test_loader` 并非“只被保存/从未消费/死路径”：`val_dataloader()` 返回 `[val_loader, test_loader]`，fit 每 epoch 遍历、每 10 epoch 算 `test_loss`；`trainer.test()` 未调用 | **采纳（已改写三处）**：§3.4 错误句替换为“第二个 validation loader”说明；§4.7 P2-8 改写；§7 Q14 改写。新增 §3.4 val 循环组成、§5.7 `val_dataloader`/训练期验证与评测共用 dataset 契约、断言 A64 | §3.4、§4.7 P2-8、§5.7、§7 Q14、§6 A64 |
| **P1-1** | `_meta.json` 严重度被低估：README 流程无生成者 → 全 fallback → K=0 → NASS 负项=0 / MDCL 关闭 / val 分桶全 `0near` | **采纳并升格为主结论**：新增 §1.4 六步证据链；§4.7 P1-6 升格；§5.1 把 meta 标为必需；§5.9 README 行改 ❌；Q6 结案（文档层）；新增断言 A65 | §1.4、§4.7 P1-6、§5.1、§5.9、§7 Q6、§6 A65 |
| **P1-2** | A18“1/3 通道 .pt 均可工作”是自建 torchvision stub 的行为，非真实依赖验证 | **采纳**：A18 降级为 `[代码]+[推断]` 并加条件列；§5.1 mouths 行注明“C=3 最可信，1 通道未验证”；§8.5 明示 stub 保真边界；§8.6 同步 | §6 A18 条件列、§5.1、§8.5、§8.6 |
| **P1-3** | “batch dtype=float32 权威结论”过度概括（uint8 存储→float64） | **采纳**：§5.4 改为条件化（uint8→float64；float32→float32）+ 下游 `type_as`/`.float()` 兜底；§1.1 图、§5.3、§5.8、§8.6 同步；断言 A19/A23 加条件列；§8.7 用真实权重 + uint8 `.pt` 实测 float64 | §5.4、§1.1、§5.3、§5.8、§6 A19/A23、§8.7 |
| **P1-4** | near wav 采样率未纳入契约/未校验（near.json 只存 frames，按 16k 解释） | **采纳**：§5.1 near 行标“必须 16 kHz/不重采样”；§5.2 注明 near.json 无 sample_rate；§5.8 增行；§4.7 新增 P1-7；§7 新增 Q16；断言 A66 | §5.1、§5.2、§5.8、§4.7 P1-7、§7 Q16、§6 A66 |
| P2-1 | README:134 是 `### Python API`，格式只在 :132 | **采纳**：§5.8/§5.9 引用改为 `README:132` | §5.8、§5.9 |
| P2-2 | `__len__` 引用 :171，实际 :173-174（:171 是 `self.length = orig_len`） | **采纳**：改为 `:171,173-174` | §2.2.3 |
| P2-3 | 评测数据构造引用 `:178-189`，实际 `:183-189` | **采纳**：§1.1 改为 `:183-189`（§3.5/§8.6 本已精确） | §1.1 |
| P2-4 | “README 只有下载 TODO”不准确：唯一 TODO 是 backbone 权重，数据/meta 无任何说明 | **采纳**：改为“README 完全未提 meta/ mouths 产生方式”；新增 P2-14 | §2.2.3 附近说明、§4.7 P2-14 |
| P2-5 | `requirements.txt:6-7` 中 torchvision 实际在 :7（:6 是 torchaudio） | **采纳**：改为 `requirements.txt:5,7,11,13-14,23-24` | §8.4 |
| P2-6 | 未提 `wav_path.replace(out_filename,"mouths")` 是全路径字符串替换风险；且不校验 `.pt` 存在 | **采纳**：新增 P2-12 | §4.7 P2-12 |
| P2-7 | 未提每样本 `torch.load(.pt)` 整段加载只为取 50 帧的 I/O 开销、eval 每条读 meta | **采纳**：新增 P2-13 | §4.7 P2-13 |
| P2-8 | 内部口径矛盾：§1.1 说 meta“只用于生成 near.json”，但 eval/val 也用它 | **采纳**：§1.1 目录图改为“用于 near.json + 评测/验证 n_near 统计” | §1.1 |
| P2-9 | 验证脚本/日志主体仍在 `/tmp/dp-verify/`，未按 VERIFY-ENV §4 落仓库 | **采纳**：补交到 `docs/verification/scripts/data_pipe/`、`runs/data_pipe/`、`stubs/data_pipe/`（含 README 与路径说明） | §8.5、§8.7、`docs/verification/**` |
| §8.6 附注 | 评审复核了 §8.6 的 5 处跨模块行号，全部准确 | 保留，并在 §8.6 沿用（新增 dtype/A18 边界说明）；§8.7 追加真实权重证据 | §8.6、§8.7 |
| C1/C2/C3 | C1：03 写 mouth 96 是 03 的错误（05 保持 88 正确）；C2/C3：val loader 组成 03 正确、05 错误 | C1 维持 05 结论；C2/C3 已按 P0-1 修正并补 `[cv,tt]`/`test_loss` 契约 | §3.4、§5.7 |

**对评审 §6 问题清单的回答**：
- **Q-A（P0-1）**：同意，三处已改；并明确区分“fit 的 val loop 会消费 test_loader（每 epoch 遍历/每 10 epoch 出 loss）”与“`trainer.test()` 从未调用”。
- **Q-B（P1-1）**：同意，已升格为主结论（§1.4 + §4.7 P1-6 + §5.1 + Q6 结案），并把 6 步后果链写全。
- **Q-C（P1-2/P1-3）**：同意，A18 已降级并加条件列；§5.4 已条件化。真实训练 `.pt` 的 dtype 仍无资产样例（Q2）；§8.7 用真实权重 + uint8 `.pt` 实测了“uint8→float64”路径。
- **Q-D（P1-4）**：near wav 16 kHz 无法由本仓库保证（无生成器）→ 已写成契约（§5.1/§5.2/§5.8）+ 风险（P1-7）+ Q16。
- **Q-E（P2-9）**：已补交脚本/日志/stubs 与运行说明到 `docs/verification/`。

**R3 额外证据（Lead 提供真实权重）**：`/root/.cache/falcon_weights/falcon.pt` 实载验证——顶层 `{state_dict, architecture, epoch}`、662 keys（525 audio+137 video）、strict 加载 OK；真实权重下 `T=32000/F=50`、`T=48513/F=75` 整句评测均输出 `len==T`、finite（§8.7/A68）。该证据同时关闭了 04 文档 Q1/B5 的“ckpt 格式未知”问题（详见 `docs/reviews/round1/04-losses-metrics-video-review-by-data-pipe.md` P1-1）。

**R3.1 小修（round2 复核响应，2026-10-21）**：对象 `docs/reviews/round2/05-data-pipeline-round2-review-by-layers-dsp.md`（结论：通过，P0=0/P1=2/P2=4）。仅做两处 P1 + 两处 P2，**不改任何结论**：

| 复核编号 | 问题 | 处理 | 落点 |
|---|---|---|---|
| **P1-1** | A18 的“降级”未落到断言表置信度列（表内仍 `[代码][验证]`，与 §8.5/§9.1 矛盾） | **已改**：A18 置信度列改 **`[代码]+[推断]`**；断言正文拆成“3 通道 [代码] / 1 通道仅 stub 下 [推断]”；§8.5 同步“R3/R3.1 已降级、表已同步”。三处一致 | §6 A18、§8.5、§9.1-P1-2 |
| **P1-2** | A60/§3.9/§8.4(1) 的“uint8 输入 max\|Δ\|=0.0”不可复现（实测 raw 2.574e-07 / cast 2.384e-07；0.0 仅在训练侧输入 float32 时成立） | **已改条件式**：§3.9、A60、§8.4(1) 统一为“uint8 `.pt` → float32 精度内等价（raw 2.574e-07、cast 2.384e-07）；float32 `.pt` → 逐位 0.0”；补充边界“`preprocess_mouth` 仅接受 `(50,96,96)`，F=75 抛 `ValueError`，A60 不覆盖变长 F” | §3.9、§6 A60（断言+前提列）、§8.4(1) |
| P2-3（review P2-1） | §8.4(1) 的训练侧输出形状/ dtype 与实际管线不符（实际 `(50,88,88)`，uint8 时 float64） | **已改**：写明 `get_preprocessing_pipelines()["val"]` 返回 `(50,88,88)`；`(1,1,·)`/f32 是手工包装 `torch.from_numpy(out)[None,None].float()`，并给出实际 dtype | §8.4(1) |
| P2-4（review P2-2） | §8.7 命令用 `/tmp` stub | **已改**：命令改为 `PYTHONPATH=$PWD/docs/verification/stubs:$PWD/docs/verification/stubs/data_pipe:$PWD`；§8.1/§8.6 的脚本/日志路径同步改为仓库内 `docs/verification/`（fixtures 仍在 `/tmp/dp-verify`，可重生成） | §8.1、§8.6、§8.7 |

### 9.2 变更日志（Changelog）

| 日期 / 轮次 | 改动 | 采纳的 review 意见 |
|---|---|---|
| 2026-10-21 / R1 | 初版：8 节模板 + 63 条断言 + 15 条未解疑问；完成磁盘→JSON→样本→collate→batch 全链路梳理、config 逐键对照、DataPreProcess 动态验证、Studio/FRCNN 交叉验证 | —（待 R2 评审） |
| 2026-10-21 / R2-prep | 新增 §8.6：用真实模型坐实 P0（mono→eval 0 维→RuntimeError，阻断 README 评测命令）+ 评测入口数据路径契约（不用 DataLoader/整句/无截断/模型内插值/use_near=False 行为）；新增 `docs/verification/data-pipe-supplement.md` | Lead 补充验证要求 |
| 2026-10-21 / R2 | 完成 `04-losses-metrics-video.md` 交叉评审（`docs/reviews/round1/04-losses-metrics-video-review-by-data-pipe.md`）；据评审发现补 §8.6 的 mouth dtype 依赖说明 | 评审 04 时自查发现 |
| 2026-10-21 / R3 | 采纳 train-sys 评审全部 P0/P1/P2：修正 `test_loader` 三处错误（P0-1）并补 val_dataloader=[cv,tt]/test_loss 契约；`_meta.json` 升格为主结论（§1.4 六步链）；A18 降级、dtype 条件化、near 采样率契约；新增 §8.7 真实权重验证（falcon.pt 格式 + 整句评测形状）；断言表增至 68 条并加“前提/条件”列；补交验证脚本/日志/stubs 到 `docs/verification/{scripts,runs,stubs}/data_pipe/`（§9.1 逐条响应） | train-sys R1 评审（P0×1、P1×4、P2×9） |
| 2026-10-21 / R3.1 | 按 round2 复核（P0=0/P1=2/P2=4）小修：A18 置信度列改 `[代码]+[推断]`（三处一致）；A60/§3.9/§8.4(1) 改为条件式等价（uint8→float32 精度内 2.574e-07；float32→逐位 0.0）并补 `preprocess_mouth` 仅支持 F=50 的边界；§8.4(1) 写明 val 管线实际输出 `(50,88,88)` 与手工包装；§8.1/§8.6/§8.7 命令改用仓库 stub 路径（不再依赖 `/tmp`）；sha256 更新并记录到 `runs/data_pipe/05-data-pipeline-r3.sha256` | `docs/reviews/round2/05-data-pipeline-round2-review-by-layers-dsp.md` |
