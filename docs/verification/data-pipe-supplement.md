# Data Pipeline 补充验证（R2-prep）：mono-eval P0 复现 + 评测入口数据路径契约

> 作者：`data-pipe`（应 Lead 要求，2026-10-21）。
> 关联文档：`docs/understanding/05-data-pipeline.md`（§4.7 P0、§5.3/§5.4/§3.5）。
> 目的：(1) 用真实模型复现并坐实 P0（mono wav → `AVSpeechEvalDataset` 退化 0 维 → 评测命令崩溃）；(2) 固化 `audio_visual_test.py` 的数据路径断言，供 R2 评审 `04-losses-metrics-video.md` 使用。
> 限制：无真实数据集/权重；用最小合成 fixture + 真实代码路径（真实 `look2hear` model，`pretrain=None`）。**未修改任何源码**。

---

## 1. 环境与复现方法

- 解释器：`/root/.venvs/falcon/bin/python`（3.10.18 / torch 2.8）+
  `PYTHONPATH=/tmp/dp-verify/stubs:<repo>/docs/verification/stubs:<repo>`。
  - `/tmp/dp-verify/stubs/` 是本次为 `torchvision.transforms/cv2/matplotlib/torchvision.models` 补的最小 stub（仓库 `docs/verification/stubs` 只有 timm/typeguard）。`torchvision.transforms.Grayscale` 按官方 luma 权重实现；transform.py 的 numpy 变换用的是仓库真实代码。
- 脚本：`/tmp/dp-verify/verify_p0_eval.py`；完整 stdout 日志 `/tmp/dp-verify/logs/verify_p0_eval.log`（脚本自写）与 `/tmp/dp-verify/p0_run.log`（本次运行 stdout）。
- 命令（在仓库根）：
  ```bash
  PYTHONPATH=/tmp/dp-verify/stubs:$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_p0_eval.py
  ```
- Fixtures（`/tmp/dp-verify/p0/`，脚本自建，均含 `mix.json`/`target.json`，`.pt` 为 `(F,96,96,1) float32`）：
  | fixture | 音频 | 音频长度 | mouth 帧数 |
  |---|---|---|---|
  | `mono/` | 1 声道 wav（按 README:67“mono 16 kHz”语义） | 40000 | 62（2.5 s@25fps） |
  | `stereo/` | 2 声道（双通道复制） | 40000 | 62 |
  | `stereo_odd/` | 2 声道 | 48513（非 640 整数倍） | 75（3.0 s@25fps） |
- 真实模型：`FalconSpeakerContrastive(sample_rate=16000, **configs/Falcon.yml audionet)`（537,697 参数）+ `FRCNNVideoModel(..., pretrain=None)`（11,186,688 参数），均 `eval()`。

---

## 2. P0 复现：mono wav → eval Dataset 退化 0 维 → 评测/验证崩溃

### 2.1 数据层：`AVSpeechEvalDataset` item 形状

实测（mono fixture）：

```
len(eval_mono) = 1
item types      : ['Tensor', 'Tensor', 'ndarray', 'str', 'int']
mixture.shape   : () torch.float32 | ndim = 0      <-- 退化为 0 维
source.shape    : () torch.float32
mouth.shape     : (62, 88, 88) float32
n_near          : 2
mix[None].shape : (1,)                             <-- 送模型时变成“长度为 1 的音频”
```

根因（2 行代码，无 ndim 守卫）：

- `look2hear/datas/avspeech_dataset.py:193` `source = torch.from_numpy(source).mean(-1)`
- `look2hear/datas/avspeech_dataset.py:194` `mixture = torch.from_numpy(mix_source).mean(-1)`

`sf.read` 对单声道 wav 返回 `(T,)`（1 维），`mean(-1)` 对 1 维张量是**对时间维求均值** → 0 维标量。对照：训练集 near 版有守卫 `avspeech_dataset_near.py:176-177`（`if mix_source.ndim > 1 ... else torch.from_numpy(...)`），所以训练路径 mono 可活，评测路径不行。同样的无守卫拷贝还存在于旧训练集 `avspeech_dataset.py:114-115` 与 `check_avspeech_dataset.py:121-122,201-202`（当前入口不经过，但属同一缺陷族）。

### 2.2 模型层：评测命令在第一个样本上崩溃

复刻 `audio_visual_test.py:229-231` 三行（真实视频/音频模型）：

| 步骤 | 形状 | 结果 |
|---|---|---|
| `mix, sources, mouth, key, _ = tensors_to_device(test_set[idx], cpu)` | mix `()`、mouth `(62,88,88)` | OK |
| `mouth_emb = video_model(torch.from_numpy(mouth[None,None]).float())` | `(1,1,62,88,88)` → `(1,512,62)` | OK |
| `est_sources = model(mix[None], mouth_emb)` | `mix[None] = (1,)` → `STFTEncoder.unsqueeze_to_2D`（`Falcon.py:1095-1097`）把 `(1,)` reshape 成 `(1,1)`，`torch.stft`（`Falcon.py:1110-1116`，`n_fft=256`, `center=True`）对长度 1 的信号 pad 128 失败 | **RuntimeError** |

实测异常原文：

```
RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension,
but got: padding (128, 128) at dimension 2 of input [1, 1, 1]
```

traceback 末三帧：`Falcon.py:1110 spec = torch.stft(...)` → `torch/functional.py:728 F.pad(...)` → `torch/nn/functional.py:5290 torch._C._nn.pad(...)`。

**该异常不会被吞掉**：`audio_visual_test.py:231` 的 `model(...)` 在 try 之外；try 从 `:233` 开始、只捕获 `ValueError`（`:237-242`，且仅 "all zeros"/"nan" 才 pass）。因此 RuntimeError 从 `process_worker` 抛出 → 单卡 `process_worker(0,1,config)`（`:284-285`）或 `mp.spawn` 直接失败。

**结论：若真实数据是单声道，README:165 的评测命令 `python audio_visual_test.py --conf_dir ... --gpus 0` 在第一个测试样本上必然崩溃。** 这不是理论推断，是真实模型前向的 [验证] 结果。

### 2.3 训练期验证路径同样受影响（重要推论）

`AVSpeechEvalDataset` 同时用于训练期 `val_dataloader`（`avspeech_dataset_near.py:286-290`、`309-318`）。用 `default_collate` 模拟 val batch（mono）：

```
val-batch types: ['Tensor','Tensor','Tensor','tuple','Tensor']
mixture.shape = (1,)  targets.shape = (1,)  mouth.shape = (1,62,88,88)  n_near.shape = (1,)
!! val path EXCEPTION: RuntimeError: ... padding (128,128) ... input [1,1,1]
```

推理：Falcon 若是用该仓库代码训练出来的，则**真实训练/评测数据必然是多声道**（否则第一次 `validation_step` 就崩，early stopping/ checkpoint 监控无法工作）；或评测/验证路径从未被真正跑过。这与 README:67 “RealSSA … all audio is mono 16 kHz” 的表述**存在矛盾**，需要数据侧确认（05 文档 Q1；本补充实测 stereo 路径正常：`(1,1,40000)`）。

### 2.4 修法建议（最小、与训练集一致）

1. **主修**：`look2hear/datas/avspeech_dataset.py:193-194` 改为与 `avspeech_dataset_near.py:176-177` 相同的守卫：
   ```python
   mixture = torch.from_numpy(mix_source).mean(-1) if mix_source.ndim > 1 else torch.from_numpy(mix_source)
   source  = torch.from_numpy(source).mean(-1)     if source.ndim > 1     else torch.from_numpy(source)
   ```
   实测该修法后：`fixed_mix.shape = (40000,)`、`est_sources.shape = (1,1,40000)`、输出长度 == 输入长度 → 评测可正常继续。
2. **可选系统化**：统一用 `sf.read(..., always_2d=True)` + `.mean(-1)`，或抽一个 `_to_mono()` helper，三处（eval / 旧 train / check 脚本）共用。
3. **防御性断言**：在 `AVSpeechEvalDataset.__getitem__` 返回前断言 `mixture.ndim == 1 and mixture.shape[-1] >= 1`（或在 DataModule 启动时打印首个样本形状），让数据格式错误早失败、且信息明确。
4. 文档层：README 数据准备章节明确“wav 通道数”要求（mono 或 2ch）；`_meta.json`/mouth `.pt` 同理。

---

## 3. 评测入口数据路径契约（供 04 评审）

### 3.1 `audio_visual_test.py` 确实不用 DataLoader

对 `audio_visual_test.py` 做 AST 审计（脚本 §8）：`DataLoader` / `train_dataloader` / `val_dataloader` / `test_dataloader` / `make_loader` **均不出现**；只出现 `setup`、`make_sets`、`tensors_to_device`。

代码路径（`[代码]`）：
- `:182` 仅有注释 “Initialize DataLoader / Dataset”；`:187-189` `constructor + setup() + _, _, test_set = datamodule.make_sets`；
- `:192-193` `indices = list(range(len(test_set))); local_indices = indices[rank::world_size]`；
- `:229` `mix, sources, mouth, key, _ = tensors_to_device(test_set[idx], device=device)` —— **逐条取 dataset item，无 collate、无 batch**；
- `:231` `est_sources = model(mix[None], mouth_emb)` —— 真实 batch 维由 `[None]` 手工制造。
- 因此 `AVSpeechDataModuleWithNear.val_dataloader/test_dataloader`（`avspeech_dataset_near.py:309-329`，硬编码 batch_size=1）在评测中**从未被调用**；`batch_size/num_workers/pin_memory/persistent_workers` 对评测无影响。

### 3.2 dataset 输出形状/类型（真实实测，多声道数据）

`test_set[idx]`（`AVSpeechEvalDataset`）= 5 元组：

| # | 内容 | 形状/类型 | 证据 |
|---|---|---|---|
| 0 | mixture | `(T,)` `torch.float32`（mono 时退化为 `()`，见 §2） | `avspeech_dataset.py:193-194` |
| 1 | source | `(T,)` `torch.float32` | 同上（:193） |
| 2 | mouth | `(F,88,88)` `numpy.float32`（val 管线，CenterCrop 88） | `avspeech_dataset.py:189-191`；`transform.py:161-167` |
| 3 | key | `str`（mix basename） | `avspeech_dataset.py:212` |
| 4 | n_near | `int`（`_meta.json` 中 type=="near" 数） | `avspeech_dataset.py:201-210` |

实测（stereo fixture）：`mix (40000,) f32`、`mouth (62,88,88) f32`、`n_near int`；`tensors_to_device(tuple)` 返回 `list[5]`（`look2hear/utils/torch_utils.py:19-40`）。`audio_visual_test.py:230` 再 `mouth[None,None]` → `(1,1,F,88,88)` → FRCNN → `(1,512,F)`（实测 F=62 → `(1,512,62)`）。

### 3.3 eval 下 mouth 与 audio 的时间对齐：整句、不截断/不填充，模型内部按帧数插值

- **Dataset 层无任何截断/填充/2 s 窗口**：`start=0, stop=None` 读整段（`avspeech_dataset.py:179-187`），嘴部整段帧（`:189-191`）。实测：48513 采样点、75 帧全部原样返回（`audio T = 48513 | mouth F = 75 | input wav frames = 48513`）。
- **模型层容忍任意帧数比**：`NeuroSCNetFusion.forward`（`look2hear/models/Falcon.py:248-262`）在 `v_feat.shape[-1] != T_a` 时 `F.interpolate(v_feat, size=T_a, mode="nearest")`（`:257-258`）。实测 fusion 调用：`audio (1,112,380,129), video (1,512,75)` → 75 帧最近邻插值到 380 个音频时间步。
- **输出长度严格等于输入**：`STFTDecoder.forward` 取 `length = input_shape[-1]`（`Falcon.py:1166`），`torch.istft(..., length=length)`（`:1182-1188`）→ 实测 `out.shape = (1,1,48513)`，`length == T: True`。
- **标称对应关系**：25 fps 每帧 640 采样点，STFT hop=128（`configs/Falcon.yml:28-29`）→ 每视频帧 ≈ 640/128 = 5 个音频时间步；实测 T'=380 / F=75 ≈ 5.07（因 `T' = 1 + floor(T/128)` 的取整项）。**评测侧没有显式的 2 s 截断/补齐，也没有对 mouth 帧数做校验**——若 `.pt` 帧率/时长与音频不一致，模型会静默 nearest 拉伸（质量风险而非报错）。
- 对 04 的含义：若 04 声称“评测按 2 s 窗口/50 帧”或“mouth 与 audio 被裁剪对齐”，与代码不符；评测是**整句变长**推理（T、F 均随样本变化）。

### 3.4 `make_sets` 与 `use_near=False` 的实际行为

复刻 `audio_visual_test.py:183-189`（`data_config` 剔除 `use_near/train_samples` 后强制 `use_near=False`）：

```
make_sets types: ['AVSpeechTrainDatasetWithNear', 'AVSpeechEvalDataset', 'AVSpeechEvalDataset']
data_train.use_near = False | #non-empty near lists = 0
data_val/data_test type = AVSpeechEvalDataset / AVSpeechEvalDataset   (use_near=False 不传播给 eval 数据集)
test_set[0] tuple len = 5 | mouth.shape = (62, 88, 88)
use_near=False + 删除全部 near.json: setup() OK (train near all empty: True)
```

要点：
1. `make_sets` 是 `@property`，返回 **3 个已构建的 dataset**（`avspeech_dataset_near.py:335-337`），评测只用第 3 个；`setup()` 同时构建 train/val/test，所以**即使只评测，也要求 train_dir/valid_dir 存在且 JSON 合法**（`avspeech_dataset_near.py:277-295`；实测把 valid_dir 指向不存在路径 → `FileNotFoundError: .../cv/mix.json`）。
2. `use_near=False` 只作用于 **train** dataset（不加载 near.json、`near=[]`）；eval 数据集没有 `use_near` 概念，仍会读 `_meta.json` 统计 `n_near`（`avspeech_dataset.py:201-210`），但该值在评测脚本中被 `_` 丢弃（`audio_visual_test.py:229`）。
3. 因此评测**不依赖 near.json**；但依赖 train/valid/test 三份 `mix.json`+`target.json`。

### 3.5 评测口径（供 04 交叉核对）

- 音频：整句 `(T,)`，不是 32000 窗口；SI-SDR/SDR 在整句上计算（`audio_visual_test.py:233-241`）。
- 视频：整句 `(F,88,88)`，F 由 `.pt` 决定；无 50 帧上限（50 帧是训练段/fps_len=50，见 05 §3.7）。
- batch=1（逐条），无 DataLoader、无 pin_memory/worker。

---

## 4. 结论清单

| 编号 | 结论 | 证据 | 等级 |
|---|---|---|---|
| S1 | mono wav 在 `AVSpeechEvalDataset` 中 mixture/source 退化为 0 维 | `avspeech_dataset.py:193-194`；实测 `()` | [代码][验证] |
| S2 | 退化后 `mix[None]` 形状 `(1,)`，真实音频模型在 `torch.stft` 处抛 RuntimeError | 实测；`Falcon.py:1095-1097,1110-1116` | [验证] |
| S3 | 该异常不被评测脚本捕获（model 调用在 try 之外，try 只吞 metrics 的 ValueError） | `audio_visual_test.py:231,233-242` | [代码] |
| S4 | 训练期 val 路径同样受影响（default_collate + 同一 Dataset） | 实测；`avspeech_dataset_near.py:286-290,309-318`；`av_litmodule.py:86-119` | [代码][验证] |
| S5 | 若真实数据为 mono，README 评测命令必然在第一个样本崩溃；能训练成功则反证真实数据是多声道 → README:67“mono”表述需核实 | S1-S4 | [推断] |
| S6 | 修法：在 eval 的 mean(-1) 前加 ndim 守卫（与 near.py:176-177 一致）；实测修复后 `(1,1,T)` 正常 | §2.4；实测 | [验证] |
| S7 | `audio_visual_test.py` 不用 DataLoader，逐条 `test_set[idx]`，整句推理 | `:189,192-193,229-231`；AST 审计无 DataLoader/dataloader/make_loader | [代码][验证] |
| S8 | eval dataset 输出 `(T,)`/`(T,)`/`(F,88,88)`/str/int（多声道数据） | 实测；`avspeech_dataset.py:176-212` | [验证] |
| S9 | eval 无 2 s 截断/补齐；视频帧数由模型 `F.interpolate(..., size=T_a, mode="nearest")` 对齐到音频时间步；输出长度==输入长度 | `Falcon.py:257-258,1166,1182-1188`；实测 F=75→T'=380→out T=48513 | [代码][验证] |
| S10 | `use_near=False` 只影响 train；eval 仍需 3 个 split 的 JSON；near.json 不参与评测 | `avspeech_dataset_near.py:277-295,335-337`；实测 | [代码][验证] |

---

## 5. 未能验证 / 边界

- 未使用真实 LRS2/LRS3/RealSSA 数据：真实 wav 通道数仍未知（05 Q1）→ S5 是“若 mono 则必然崩溃”的条件结论 + “能训练成功则应为多声道”的反证推断。
- 未运行完整 `audio_visual_test.py`（缺 `pystoi/pypesq/ptflops` 与 checkpoint）；但 P0 发生在模型前向、早于 metrics，且异常不被捕获，故阻断结论成立。
- `FRCNNVideoModel` 用 `pretrain=None` 构建（未加载 `frcnn_128_512.backbone.pth.tar`）；形状/前向契约与权重无关。
- 复现脚本与 fixture 均在 `/tmp/dp-verify/`（不随仓库提交）。
