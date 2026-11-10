> **2026-11-10 复核提示：** 历史评测更新：本地发布权重的训练语料身份尚待确认；论文差距及 epoch 变化涉及多个训练因素；RealSSA-Real full4 修正了 full3 的覆盖缺口，但帧选择和相位配对仍存在协议限制。阈值过滤保留 `loss > -30` 的结果；CSV 文件身份可用哈希核对。比较结果时同时核对样本和评测协议。详见[评测专题](../20261110-evaluation-and-apps.md)和[数据审计](../../verification/deepdive-20261110/data-audit-20261110/README.md)。

# 07 评测入口与 benchmark 数字审计

> W1-07 · 2026-11-08 · 范围：`audio_visual_test.py`(约 875L) 全文、`docs/verification/phase2/**`、`phase1/**` 评测相关、`/root/data/falcon_eval`、`docs/STATUS.md §3` 复现矩阵。
> 远端只读：`dev_a800_2gpus:/root/code/github_repos/Falcon-fork` @ `6c1d5b7`，工作区 `git status` 干净（改动前后均验证）。
> 本文所有 `path:line` 指远端仓库路径；CSV 行号为**含表头的 1-based 行号**。

---

## 0. 结论速览

1. **0 次切窗**：`audio_visual_test.py:479` 把整段 `mix` 一次送进模型，全文没有 `window/hop/chunk/pad`；「2 s 窗 / 0.5 s hop / 80 ms guard」只属于 Studio 运行时与 RealSSA-Real 长音频评测器（`:475-486`、`Falcon.py:1480-1499`、`Falcon.py:1189-1195`）。
2. **1 路输出**：`n_src: 1`（`configs/LRS2_2Mix.yml:11`），`SI-SNRi = SI-SDR(ŝ, s1) − SI-SDR(mix, s1)`，逐句算，PIT 在 S=1 时退化为恒等（`core.py:75-79,152-155`）。这里既不是 2 源分离口径，也没有「best-of-2 输出通道」这一步。
3. **5.3e-8 dB 的复现误差**：官方 `falcon.pt` 跑 5 条，`max|Δ|` 对 `P2D-eval-falcon-full.csv` 前 5 行为 5.3e-8 dB（SI-SNRi）、PESQ 2.3e-5；两次历史全量 run 之间 3.3e-7 dB。误差在 float32 内核噪声量级，评测本身确定性足够。
4. **9,169 与 9,166 两个分母**：RealSSA-Sim 的 SI-SNRi 用 9,169 条，SDRi 与 PESQ 只有 **9,166** 条（3 条 NaN），STATUS 把两个口径写成了同一个数。avg 行可精确定位：LRS2 类 `:3002`、RealSSA-Sim 类 `:9171`，两份逐句 CSV 都能复算。
5. **−1.6 ~ −2.3 dB 是协议伪影**：整段 vs 分窗，5 句实测 −1.648，300 句历史 −1.912/−2.318。2 s 片段上模拟流式时首窗有 1.42 s 零前缀，这个差值不能反过来解释论文 14.1 与本仓库 7.03 的差距。
6. **论文差距原因待核实**：官方权重在 LRS2-2Mix 为 7.029/7.713，自训 ep124 为 8.858/9.270；发布权重训练语料及论文权重身份尚未确认。
7. **+0.0040 OVRL 与一次确定性错误**：DNSMOS full3 → full4 的均值只涨 0.0040，但 full3 每条估计开头 21,440 样本（1.34 s）恒为 0，属输出确定性错误，4 个值作废（§4.4）。
8. **覆盖率断言：当时 0 处，2026-11-08 已补**：在 `6c1d5b7` 时点，`audio_visual_test.py` 全文 23 处 `raise`（按 `raise` 语句计）没有一条校验「跑了几条/共几条」，只能靠日志 `utterances=N` 与 `metrics_counts.json.n_utterances` 事后核对；`eval_realssa_real.py` 是在 `2062528` 才补上逐样本 `filled` 断言的（`:428-459`）。2026-11-08 已补三层覆盖率自检（rank 内「计划 vs 实际」、跨 rank 合计、合并 CSV 行数），并加 `--allow-partial`（`9a920ce`）；`raise` 现 28 处（`STATUS.md:141`，`grep -c 'raise '` 同值；其中 27 处为语句，另 1 处是文件头 docstring 的描述）。
9. **0 条失败句被剔除**：`drop_utt/drop_len` 是永远打印 0 的死代码（`:162-175`）；指标级 NaN 由 `nanmean` 静默跳过，只在 `n_*` 列留痕。这是 §0-4 那个 9,166 的成因。
10. **4 个 LRS2 数字都是 `limit=all`**：P2A(200/20)、P2B(300) 在报告里显式标注子集，没有子集被当全量（§3.5）。
11. **可对外引用的 5 组数**：7.774/8.479（RealSSA-Sim mean，带 ckpt 与通道声明）、7.977/8.452（ch0）、2.7941/3.3365（DNSMOS full4）、7.029/7.713（LRS2 跨语料基线）、8.858/9.270（LRS2 自训 ep124）。full3 的四个 DNSMOS 值不进入任何对外表述；7.03 与 14.1 只能并列成「不可比」，写成「复现」是错的；47.1/56.5 ms 缺原始测量文件，只作线索。

---

## 1. 事实基线

### 1.1 入口与参数表（`audio_visual_test.py:110-139`）

| 参数 | 默认 | 语义 | 代码行 |
|---|---|---|---|
| `--conf_dir` | `configs/Falcon.yml` | 训练配置 YAML，须含 `audionet`/`videonet`/`datamodule` | `:111-113`、`:211-215` |
| `--data_dir` | `""` | JSON 索引目录，**覆盖** `datamodule.data_config.test_dir` | `:114-117`、`:223-239` |
| `--ckpt_path` | `""` | `.ckpt` / `falcon.pt` / 裸 state_dict / `hf://org/repo/file` | `:118-121`、`:842-860` |
| `--gpus` | `"0"` | 逗号分隔物理卡号；`""`/`cpu` 强制 CPU | `:122-125`、`:378-385` |
| `--limit` | `0` | 只评前 N 条（0=全部） | `:126-132`、`:416-431` |

- 无 `--batch_size`、无 `--window/--hop/--chunk`、无 `--min-duration`、无 `--seed`、无 `--out-dir`。
- **结果落盘位置不受参数控制**：`<cwd>/Experiments/checkpoint/<exp_name>/results/`（`:446-448`、`:817-820`）。`exp_name` 来自配置（`LRS2_2Mix.yml:154` = `LRS2_2Mix`，`Falcon.yml:173` = `Falcon`）→ 同一 cwd 跑不同 ckpt **会互相覆盖**。历史上有 2 次直接写进了仓库树（`P2E-realssa-sim-eval-official.log` 的 results dir = `/root/code/github_repos/Falcon-fork/Experiments/checkpoint/Falcon/results`，`P2D-eval-epoch124-contrastive.log` 同理）；因 `.gitignore` 有 `results/`，`git status` 仍干净，但这是**产物与源码混放**的风险点。
- 数据/索引解析：
  - `_resolve_index_dir()` 依次尝试 `<path>/tt`、`<path>`，**必须同时存在 `mix.json` 与 `target.json`**（`:187-203`，判据在 `:194-197`）。→ 参考永远是**单一 `target`**，没有 `s1.json/s2.json` 这类 2 源索引。
  - 占位符 `_expand_env_vars()` 递归展开 `${VAR}`/`$VAR`（`:144-155`），未解析的仍会打印 warning（`:217-221`）；但 `test_dir` 里的占位符会直接报错（`:233-237`）。
  - `configs/Falcon.yml:161` 的 `test_dir: --` 是合法"未设置"哨兵（`:227-232`），所以用 `Falcon.yml` 必须给 `--data_dir`；`configs/LRS2_2Mix.yml:142` 则把 `test_dir` 硬编码成本机路径。

### 1.2 checkpoint 加载：四条路径 + 两处严格校验

```
ckpt_path 为空 → _find_best_ckpt(exp_dir)            # :449
  ├─ best_k_models.json 中 val_loss 最低且本地存在    # :256-268（不存在则 warning 后降级）
  ├─ last.ckpt                                     # :270-273
  └─ epoch=*.ckpt 按**数值** epoch 取最大            # :275-283（epoch=1000 > epoch=742）
ckpt_path 给定 → 直接用（不存在则 FileNotFoundError） # :450-451
```

- 反序列化：先 `torch.load(..., weights_only=True)`，失败才回落 `weights_only=False` 并打印原因（`:307-313`）。
- 结构校验：`_extract_state_dict()` 要求 state_dict 里**至少有一个** `audio_model.*` / `video_model.*` 键（`:291-304`）；纯音频 `best_model.pth`（serialize 格式）会被明确拒绝（`:299-303`）。
- 音频侧：按 `audionet_name` 从 `look2hear.models` 取类，`strict=True` 加载 `audio_model.*`（去掉前缀）（`:323-345`）；失败信息会提示"config 与 ckpt 架构不匹配"（`:339-344`）。
- 视频侧：**先 `pop("pretrain")` 再传 `pretrain=None`**，绝不让预训练权重覆盖 ckpt 里的 137 个 key（`:348-359`）；只有在 state_dict 无 `video_model.*` 时才回落到 `pretrain` 路径（`:360-368`）。
- 复现日志实证（`P2D-eval-falcon-full.log:16-17`）：`✓ audio_model loaded: 525 keys`、`✓ video_model loaded: 137 keys from falcon.pt`。
- 载入后 `model.eval()`、`video_model.eval()`、`torch.no_grad()`（`:370-371`、`:474`、`:488`）。

### 1.3 数据加载：`AVSpeechEvalDataset`（`avspeech_dataset.py:130-219`）

| 环节 | 行为 | 代码行 |
|---|---|---|
| 索引 | 只读 `mix.json` + `target.json`，`mix_infos[i]` 与 `target[i]` 一一对应 | `:147-170` |
| 长度 | `for i in range(len(mix_infos)): for src_inf in sources_infos:` —— 仅 1 个 source json，故 `len(self.mix)=len(mix_infos)`；`self.length = orig_len` | `:166-176` |
| 音频读取 | `sf.read(path, start=0, stop=None)` —— **整文件**，不信 `mix.json[i][1]` 里声明的样本数 | `:184-192` |
| 声道 | `mix.mean(-1)` / `source.mean(-1)`（多声道取均值，mono 保持 1-D） | `:200-201` |
| 归一化 | `normalize_audio=False`（`LRS2_2Mix.yml:146`、`Falcon.yml:165`）→ 不做 RMS 归一 | `:203-206` |
| mouth | `target.json[i][1]` 指向 `.pt`，`(F,96,96,3)→Grayscale→/255→CenterCrop(88)→(x−0.421)/0.165` | `:194-196`、`:143-145` |
| near 计数 | 从同名 `_meta.json` 数 `type=="near"` 分量，异常静默吞掉 | `:208-217` |
| 返回 | `(mixture, source, source_mouth, wav_basename, n_near)` | `:219` |

- **声明长度 vs 实读长度不一致的风险**：`mix.json` 里每条是 `[wav_path, 32000]`（我实测 3000/3000 条都是 32000），但 `__getitem__` 用 `stop=None` 读全长。若某条 wav 被替换成更长文件，评测不会报错也不会截断，会静默用全长（与 `metrics` 的 `istft(length=T_in)` 自洽，但与"2 s 协议"脱钩）。
- **潜在静默错位**：`self.length = orig_len` 而 `self.mix` 是按 `len(sources_infos)` 次追加的。当前 `sources_infos` 长度恒为 1 所以相等；一旦有人加第二个 source json，`len(dataset)` 会变成一半而前一半条目被截断，**不会报错**。
- RealSSA-Sim 的 tt 是 5.8 s 级（`mix.json[0]` 长度 92,800），同样整段读入。

### 1.4 逐句推理与切窗规则（核心问题的正面回答）

worker 主循环（`audio_visual_test.py:475-486`）：

```python
mix, sources, mouth, key, _ = tensors_to_device(dataset[idx], device=device)   # :476
mouth_in  = _mouth_to_tensor(mouth).float().to(device)                        # :477  [1,1,F,96,96]
mouth_emb = video_model(mouth_in)                                             # :478
est_sources = model(mix.unsqueeze(0), mouth_emb)                              # :479  [1,n_src,T]
metrics(mix=mix, clean=sources.unsqueeze(0), estimate=est_sources.squeeze(0), key=key)  # :480-483
```

**切窗规则 = 不切窗。** 依据四条：

1. 入口层：`:479` 把整段 `mix`（`[T]`）一次性 `unsqueeze(0)` 送进模型，循环体里**没有任何 `for start in range(0,T,hop)`**；全文无 `window`/`hop`/`chunk`/`pad` 字样（只有 `--limit` 与 progress bar）。
2. 模型层：`Falcon.forward(audio_mixture, mouth_embedding)`（`Falcon.py:1480-1499`）是 `encoder → audio_bottleneck → separator → mask_generator → decoder` 的单遍直通，**无分块、无状态缓存**；`separator` 也只按 `fusion_repeats / audio_repeats` 顺序叠块（`Falcon.py:1032-1059`）。
3. 时域边界：`torch.stft` **未传 `center=`** → PyTorch 默认 `center=True`，即两端 reflect-pad `win//2=128` 样本（`Falcon.py:1117-1123`，`win: 256 / hop_length: 128`，见 `configs/LRS2_2Mix.yml:23-31`）。**没有额外补零**。
4. 输出拼接：`STFTDecoder.forward(x, input_shape)` 里 `length = input_shape[-1]`（`Falcon.py:1173`），`torch.istft(..., length=length)`（`:1189-1195`）→ 输出长度**恒等于输入长度**，不裁不补。因此 `est` 与 `clean`、`mix` 天然逐样本对齐，SI-SDR 不需要任何对齐/裁剪。

对比三种真实存在的"窗口化"实现（**都不是** `audio_visual_test.py`）：

| 实现 | 窗/跳/guard | 拼接方式 | crossfade | 用途 |
|---|---|---|---|---|
| `studio/server.py::OutputSlicer` + `studio/web/app.js` | 2.0 s / 0.5 s / 80 ms | 取 `pred[endpoint−GUARD−HOP : endpoint−GUARD]` | 320 样本（20 ms） | 线上实时 demo |
| `tools/repro/phase2/compare_variants.py:149-178` | 同上（常量 `:58-66`：`WINDOW=32000, HOP=8000, GUARD=1280, FADE=320, LEFT_CONTEXT=22720`） | 同上 | 有，`:168-171` | LRS2 上"整段 vs 分窗"消融 |
| `tools/repro/phase2/eval_realssa_real.py:381-459` | 同上 | `pred[LEFT:LEFT+HOP]`，`abs_start=endpoint−(WINDOW−LEFT)` | **无**（`:404` 注释声明） | RealSSA-Real 长音频（整段前向会 OOM） |

### 1.5 PIT 与指标调用顺序（`allwrapper.py:98-145`）

单条 utterance 的调用链，**PIT 在指标内部、不在入口**：

```
metrics(mix, clean, estimate, key)                       # allwrapper.py:132
└─ _compute(mix, clean, estimate)                        # :98
   ├─ clean_src = _as_source_matrix(clean)   → [S=1, T]   # :99,  core.py:103-112
   ├─ est_src   = _as_source_matrix(estimate)→ [S=1, T]   # :100
   ├─ n_src     = clean_src.shape[0]         → 1         # :101  ← 用 clean 定 n_src，不是用 est
   ├─ si_sdr      = -PIT_SI_SDR(est,clean)               # :103, core.py:152-155
   ├─ si_sdr_mix  = -PIT_SI_SDR(mix_dup,clean)           # :104, core.py:132-141
   ├─ sisnr_i     = si_sdr - si_sdr_mix                  # :105  ← 逐句、同一个 mix
   ├─ sdr         = fast_bss_eval.sdr_pit_loss(f=512)    # :107, core.py:168-177
   ├─ sdr_mix/sdr_i                                      # :108-110
   └─ n_src==1 → pesq(clean,est,fs) / stoi(clean,est,fs) # :112-114（n_src>1 才记 nan，:115-119）
```

- `PIT_SI_SDR` 定义在 `core.py:75-79`：`PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=True), pit_from="pw_mtx", threshold_byloss=False)`。`threshold_byloss=False` 是必须的——否则 `pit_wrapper.py:227-229` 会保留 `loss > -30`，剔除更低 loss 的样本，让"做对了"变成 nan。
- **PIT 在本配置下是恒等操作**：`PITLossWrapper.forward(ests, targets)`（`pit_wrapper.py:168-237`）对 `pw_mtx` 先算 `[B,S,S]` 配对矩阵，再 `find_best_perm` 取最优排列后 `mean`。S=1 时矩阵是 1×1，唯一排列就是恒等 → `si_sdr()` 返回值 = 单路 SI-SDR。**不存在"从 2 个输出里挑最好的"**（那是 S=2 才有的行为；本仓库 `n_src=1`，见 §2.1）。
- `_prepare_pair()`（`core.py:115-129`）里 `ref.expand(..., est.shape[1], ...)`（`:119-120`）只在 S 不匹配时生效，本例 S 相等不触发。
- 数值口径：SI-SDR `zero_mean=True`（`core.py:76`）；SDR = `fast_bss_eval.sdr_pit_loss(filter_length=512)`（`core.py:69,174`）；PESQ/PSTOI 采样率 16 kHz（`core.py:70`），STOI `extended=False`（`core.py:215`）。
- 逐句写盘：`__call__` 里立刻 `writer.writerow(row)`（`allwrapper.py:136`），所以**崩溃前的行不会丢**（这也是历史 ep86 能"重跑补 CSV"的原因之一）。

### 1.6 CSV 列定义与聚合（`allwrapper.py:52-64`；`:702-808`）

| 列 | 定义 | 来源 |
|---|---|---|
| `snt_id` | `mix wav` 的 basename（含 SIR 信息，如 `..._0.53084_..._-0.53084.wav`） | `:135` |
| `sdr`, `sdr_i` | BSS-Eval-v4 SDR / 相对 mix 的 SDR 提升，正数越大越好 | `core.py:168-185` |
| `si-snr`, `si-snr_i` | 零均值 SI-SDR / 相对 mix 的提升 | `core.py:152-162` |
| `pesq`, `stoi` | `pesq(clean, est, 16000)`、`stoi(clean, est, 16000, extended=False)` | `core.py:196-217` |
| `si-snr_mix`, `sdr_mix` | **逐句** mix 基线（2026-10-30 追加，A4-4） | `:104,108` |
| `n_<metric>` | 该列**有限值**计数；逐句行留空，`avg`/`std` 行填真实计数（A4-3） | `:741-752` |
| `avg` 行 | 逐句算术均值，NaN 跳过 | `:747` |
| `std` 行 | **总体标准差 `ddof=0`**，NaN 跳过 | `:748`（`allwrapper.py:154-159` 的 `nanstd` 语义一致） |

- 合并：多 rank 分片 `metrics_rank{r}.csv` 按 rank 顺序 `pd.concat`（`:706-720`）→ **多卡时行序是 rank 交织后的顺序，不是全局 idx 顺序**（单卡无影响）。合并后删除分片（`:801-804`）。
- 恒等式校验（`:761-767`）：`|si-snr − si-snr_i − si-snr_mix|`，`max` 写入 `metrics_counts.json`。实测：
  - `P2D-eval-epoch99-cols-20261031.counts.json` → `"max": 5.329e-15`, `finite_pairs: 3000`
  - `P2D-eval-falcon-full-cols-20261031.counts.json` → `"max": 4.441e-15`, `finite_pairs: 3000`
  - 我自己的 5 句 run → 逐行 `0 / 1.776e-15 / 0 / 1.776e-15 / 0`，max `1.776e-15`
  → **恒等式真的成立**（这是构造性的：`si-snr_i ≡ si-snr − si-snr_mix` 由 `:105` 直接相减得到，恒等式只验证三列的算术关系，文件身份需用哈希核对）。
- `metrics_counts.json` 的 `n_utterances` 是**唯一的机器可读覆盖率凭据**（`:768-789`）。

---

## 2. 机制拆解：这个仓库报的 SI-SNRi 到底在测什么

### 2.1 `n_src = 1`：单输出目标说话人提取，不是 2 源分离

| 证据 | 内容 |
|---|---|
| `configs/Falcon.yml:12`、`configs/LRS2_2Mix.yml:11`、`configs/LRS2_2Mix_Paper_v2.yml:28` | `audionet_config.n_src: 1`（三份配置全为 1） |
| `configs/*.yml` 的 `datamodule.data_config.n_src`（如 `LRS2_2Mix.yml:143`） | 也是 `1` |
| `audio_visual_test.py:194-197` | 只认 `mix.json` + `target.json`（单参考），没有多源索引概念 |
| `/root/data/falcon_lrs2_2mix/tt/target.json[0]` | `['.../target/xxx.wav', '.../mouths/xxx.pt', 32000]` —— **一条 target + 一个 mouth** |
| `conversion_stats.json`（`/root/data/falcon_lrs2_2mix/`） | `"target": "s1"`、`"near_mode": "fallback"` |
| `P2A-report.md:5` | "3000 条 LRS2-2Mix tt，**target=s1**" |
| `P2A-report.md:23` | "`mix == s1+s2`（残差 3.05e-5 = PCM16 LSB）、**`target == s1`**、mouth == s1 的 npz（逐元素相等）" |
| `P2D-report.md:35-36` | "本复现固定 **target = s1**（`configs/LRS2_2Mix.yml` 的数据由 P2-C 按该约定生成）" |

所以：**输入 = 2 人混合语音 + 目标说话人（s1）的唇部 ROI；输出 = 1 路 s1 估计**。这就是 AVTSE（音视频目标说话人提取）的评测口径，不是"分离出 2 路再匹配"。

`Falcon.forward` 的输出形状印证：`separated_audios = self.decoder(emb, audio_mixture.shape)` → `view(batch, self.n_src, length)`（`Falcon.py:1490,1197`），`n_src=1` → `[1,1,T]`；入口 `est_sources.squeeze(0)` → `[1,T]`（`audio_visual_test.py:482`），`clean` → `[1,1,T]→[1,T]`。二者都是 1 路。

> 历史线索：`look2hear/metrics/splitwrapper.py:31-36` 还留着 `one_si-snr/one_si-snr_i` 与 `two_si-snr/two_si-snr_i` 的分裂式统计，说明更早版本确实区分"1 源/2 源"两种口径；**当前评测入口完全不用 `SPlitMetricsTracker`**（`audio_visual_test.py:91` 只 import `ALLMetricsTracker`）。

### 2.2 SI-SNRi 的确切数学（逐句、相对未处理 mix）

对第 i 条 utterance：

```
ŝ = f_θ(mix_i, mouth_i)                      # [T]，单路
SI-SDR(x, s) = 10log10( ‖αs‖² / ‖x − αs‖² ),  α = <x,s>/‖s‖²,  且 x、s 各自去均值
SI-SNR_i      = SI-SDR(ŝ, s1)                
SI-SNR_mix,i  = SI-SDR(mix, s1)              # 同一 mix、同一 s1、逐句
SI-SNRi_i     = SI-SNR_i − SI-SNR_mix,i      # allwrapper.py:105
报告值         = mean_i(SI-SNRi_i)           # nanmean，3000/9169 条
```

三点容易读错，逐条写清：
1. **基线是"逐句未处理 mix"，不是"整段/语料级平均 mix"**（`allwrapper.py:104` 每次调用用当条的 `mix`）。这与论文"relative to unprocessed mixture"（`PAPER_ALIGNMENT_20261030.md:145`）同义。
2. **SI-SDR 对 ŝ 的尺度不变**，但 mix 基线值取决于该条混合的 SIR。实测全 3000 条：`SI-SDR(mix, s1)` 均值 **4.9937**（`P2D-eval-falcon-full-cols-20261031.csv:3002` 的 `si-snr_mix` 列），与 `P2A-report.md:23` 的"SIR 均值 5.01 / 实测 4.97"、`P2F-report.md` 的"baseline 全 3000 条 mean 4.994、std 3.003、median 5.011"三方吻合。**7.029 + 4.994 = 12.023**（`si-snr`）—— 这就是 §1.6 恒等式的物理含义：绝对 SI-SDR ≈ 12 dB，其中 5 dB 来自混合本身。
3. **不存在"PIT 上界"红利**：target 固定 s1。`P2A-report.md:16` 实测"best-of-s1/s2（PIT 上界）**+0.02 dB**"，`P2F-report.md` §1 独立测得"2-target oracle 上界 **+0.000 dB**"（两处微小分歧见 §7-G6）。→ 用固定 s1 与用"任选一个说话人"评测，差别 ≤0.02 dB，**结果间差异不超过 0.02 dB**。

### 2.3 与他文 14.1 的口径同义性判定

| 口径轴 | 本仓库 | 常见 AV 分离论文（如 AV-TF-GridNet 系） | 是否同义 |
|---|---|---|---|
| 任务定义 | 单输出，target 固定 s1，唇部指定 | 同为 target-speaker / 指定说话人（LRS2-2Mix 惯例） | ✅ 同义（oracle 上界仅 +0.02 dB） |
| 参考信号 | `target = s1` 波形（HF `JusperLee/LRS2-2Mix` 发布版） | 同一发布数据集 | ✅ 同源 |
| 测试集条数 | **3,000**（`tt/mix.json` 实测 3000 条） | 同数据集同 split → 应同为 3,000 | ✅ 同源；⚠️ 但**排序/子集口径**未在其文声明 |
| 归一化 | `normalize_audio=False`，不重标定 | 多数论文同样不重标定 | ⚠️ 需其文声明；本仓库实测"target 按 mix 能量/RMS 缩放 ±0.00"（`P2A-report.md:16`），故不敏感 |
| SDR 实现 | `fast_bss_eval.sdr_pit_loss(filter_length=512)` | 论文极少声明滤波器长度 | ⚠️ **不可逐位对齐**：512→2048 差 **+0.257 dB**（`P2F-report.md` §1） |
| 推理协议 | 整段一次前向（2 s clip = 一段） | 离线评测通常也是整段 | ✅ 同义（若其文用流式 2 s 窗，则我方乐观 ~2 dB，见 §2.4） |
| ckpt | `falcon.pt` 训练来源未知 | 论文对应权重身份待核 | ⚠️ 无法确认训练来源及差距原因 |

**结论：SI-SNRi 的定义可对照；权重与完整评测协议仍需确认，差距原因尚未确定。**

### 2.4 分窗协议在 LRS2 上的真实含义（别把它当"论文复现失败"的解释）

`compare_variants.infer_windowed`（`:149-178`）在 2 s clip 上的行为：
- `LEFT_CONTEXT = 32000−1280−8000 = 22720`（1.42 s）→ 首窗输入 = `[zeros(22720) ‖ audio[0:9280]]`；
- 首块输出 = `pred[22720:30720]` → 对应 `audio[0:8000]`（0.5 s）；
- 随后每窗整窗推进，`chunk = pred[WINDOW−GUARD−HOP : WINDOW−GUARD]`，与 Studio 语义一致。

即：**在 2 s 孤立片段上模拟流式，模型第一条窗看到的是 71% 的静音**，输出质量必然塌。所以测得 −1.9 dB 是"Studio 在 2 s 片段上的真实表现"，**不能**解读为"论文若用流式就会掉 2 dB"（真实流式有前文）。`P2B-report.md:139` 自己也写了这条限制。

---

## 3. 关键数据/参数表

### 3.1 数字审计表（STATUS §3 逐行 → 原始证据）

图例：✅ 可信（有逐句证据且可复算） / ⚠️ 需修口径（数字对但表述/分母/引用文件名要改） / ❌ 无证据或不可比。

| # | STATUS §3 行 | 声明值 | 原始证据文件 | 行数 | 该数字所在行 | 判定 |
|---|---|---|---|---|---|---|
| 1 | L1 参数/MACs | 537,697（生效 532,129；扣对比头 523,113）+FRCNN 11,186,688 = 11,724,385；10.611 G | `docs/verification/scripts/verify_model_params.py`（非本任务范围） | — | 我的独立 run 也打印 `audio=537,697 video=11,186,688 total=11,724,385`（`compare_variants` stdout） | ✅（旁证）。口径：10.611 GMAC = ptflops + 音频侧 + **含 LSTM** + 不含视觉；视觉 88² 输入另算 **15.897 GMAC**，整窗合计 **26.51 GMAC、视觉 60.0%**（详见 W2-11 §2.4）。`flop_counter` 口径的 5.382 GMAC 是另一把尺子（它不给 `aten::lstm` 记账），不能与 10.611 并列 |
| 2 | L1 权重一致性 | strict 零 missing；max\|Δ\|=0 | `verify_real_weights.py` / `verify_r3_studio_weights.py` | — | 我只旁证了 `✓ audio_model loaded: 525 keys`（`P2D-eval-falcon-full.log:16`） | ✅（旁证，正主见 W1-05/06） |
| 3 | RealSSA-Sim SI-SNRi/SDRi（mean） | **7.774 / 8.479**（9,169 条） | `docs/verification/phase2/P2E-realssa-sim-metrics_combined.csv` + `P2E-realssa-sim-eval-official.log` | 9172 行（表头+9169+avg+std） | **`:9171`** = `avg,10.644734,8.479395,9.980200,7.774307,2.093770,0.817290` | ✅ 数字对；⚠️ "9,169 条"只对 SI-SNRi 成立，**SDRi 分母是 9,166**（见 §3.4） |
| 4 | RealSSA-Sim（ch0） | **7.977 / 8.452**（9,169 条） | `P2E-realssa-sim-ch0-metrics_combined-20261030.csv` + `P2E-realssa-sim-ch0-eval-20261030.log` | 9172 行 | **`:9171`** = `avg,10.111060,8.452396,9.319856,7.976709,2.004064,0.815009` | ✅ 数字与文件名均核过（STATUS 的 `P2E-realssa-sim-ch0-{eval-20261030.log,metrics_combined-20261030.csv}` 两个实物都在）；⚠️ 同样有 9,169→9,166 的分母问题 |
| 5 | RealSSA-Real DNSMOS | OVRL **2.7941** / P808 **3.3365**（full4） | `DNSMOS-realssa-real-mean-full4-20261031/summary.json` + `realssa_real_dnsmos_per_segment.csv` + `run_metadata.json` | summary.json 77 行；per-segment 55 行（表头+54） | `summary.json` 的 `metrics.ovrl_mos.mean` / `metrics.p808_mos.mean`；per-seg 逐段可复算 | ✅ 数字对；⚠️ 引用时要带「官方 `falcon.pt`，不是 RealSSA-trained ckpt」这句限定，否则会被读成同 ckpt 对照（STATUS 已写） |
| 6 | RealSSA-Real DNSMOS（full3，作废） | 2.7901 / 3.3285 | `DNSMOS-realssa-real-mean-full3/{summary.json,per_segment.csv}` | summary 65 行；per-seg 55 行 | 同结构 | ❌ 作废值，不进入任何对外表述（理由见 §4.4） |
| 7 | LRS2-2Mix 官方权重（跨语料） | **7.029 / 7.713**（tt 3,000） | `P2D-eval-falcon-full.csv` + `.log` | 3003 行 | **`:3002`** = `avg,12.813387,7.712925,12.022821,7.029101,2.195634,0.916350` | ✅ 可信。日志另证 `utterances=3000 / limit=all / ckpt=falcon.pt`（`.log:5` ckpt / `:7` limit=all / `:15` utterances=3000） |
| 8 | LRS2 自训 ep86 | 8.618 / 9.042（val 13.1182） | `P2D-eval-epoch86.csv`（2026-10-30 重跑）+ `P2D-eval-epoch86-20261030.log`；历史日志 `P2D-eval-epoch86.log` | 3003 行 | **`:3002`** = `avg,14.141969,9.041506,13.611639,8.617919,2.546623,0.940860` | ✅ 可信（**两份日志的 Final metrics 六位小数完全相同** → 确定性得证） |
| 9 | LRS2 自训 ep99（best val） | **8.726 / 9.122**（val 13.2014） | `P2D-eval-epoch99.csv` + `.log`；另有 `P2D-eval-epoch99-cols-20261031.csv` | 3003 行 | **`:3002`** = `avg,14.222311,9.121848,13.719868,8.726148,2.577971,0.943988` | ✅ 可信 |
| 10 | LRS2 自训 ep124（+near 微调） | **8.858 / 9.270**（val 13.3909） | `P2D-eval-epoch124-contrastive.csv` + `.log` | 3003 行 | **`:3002`** = `avg,14.370037,9.269574,13.851912,8.858192,2.576086,0.942796` | ✅ 可信（注意：该 run 的 results 写进了仓库树，见 §1.1） |
| 11 | LRS3-2Mix | 16.4 / 16.7 | **无** | — | — | ❌ 无证据（数据未发布，`PAPER_ALIGNMENT:224`） |
| 12 | VoxCeleb2-2Mix | 10.5 / 11.8 | **无** | — | — | ❌ 无证据（`PAPER_ALIGNMENT:225`） |
| 13 | 效率（3090 51.3 ms / P.835 RAM 125.3 MB / 显存 364.7 MB） | 未测（A800 47.1 / 56.5 ms） | `docs/understanding/deepdive-20261024/06-studio-backend.md:402-403`（**只有 md 表格**） | — | 该表格行 | ⚠️ 需修口径：**无原始测量 json/log**，且硬件不同（A800 ≠ 3090），不能并列 |

补充核对（P2D 报告正文与表的一致性）：
- `P2D-report.md:204-207` 的 4 行与本表 #7/#8/#9/#10 完全一致（逐句 avg 值六位小数相同）。
- `P2D-report.md` §8.3 曾把论文硬件写成 10 卡/global 40，已于 `8b9d903`／2026-11-09 校正为 8×3090、bs4/GPU（global 32）——与 `STATUS.md`（`8×RTX3090、bs4/GPU、global 32`）和 `PAPER_ALIGNMENT_20261030.md:22`（**8×3090 / bs4 / global 32**）一致 → 见 §6-C3。
- 跨文件一致性：`P2D-eval-falcon-full.csv` 与 `P2D-eval-falcon-full-cols-20261031.csv`（相隔 7 天的两次独立 run）逐行 `max|Δ|` = SDR **2.220e-07**、SI-SNR **2.308e-07**、SI-SNRi **2.308e-07**、PESQ **9.394e-05**、STOI **3.887e-08**；行序 3000/3000 完全相同。
- `P2F-report.md` §TL;DR 1 的第三方对拍结论："核心指标与 CSV 最大差 **1.8e-7 dB**（要求 ≤0.05）、CSV avg/std == 逐句重算精确成立、由 CSV 反推 baseline 与独立计算差 4.5e-8 dB" → 与我的测量同量级，**指标实现本身无问题**。

### 3.2 口径归一表（哪些差异动了数字，动多少）

| 口径轴 | 本项目 | 对照 | 量化影响 | 证据 |
|---|---|---|---|---|
| **整段 vs 分窗** | 整段一次前向 | Studio/RealSSA-Real 用 2 s/0.5 s/80 ms | **−1.912 / −1.730**（300 句，small，whole→windowed）；**−2.318 / −2.133**（big）；我 5 句实测 **−1.648 / −1.537** | `P2B-report.md:92-93`、§4.2 本报告 |
| **mean vs ch0（RealSSA-Sim）** | 4ch 均值 | 论文用 channel-1（代码 `--channel 0`） | SI-SNRi **−0.202**（7.774 vs 7.977）、SDRi **+0.027**（8.479 vs 8.452） | 两份 CSV 的 `:9171` 行 |
| **ckpt 训练语料** | `falcon.pt` = LRS3（ep714） | 论文 LRS2 列 = LRS2 训练 | 本机 LRS2 自训 ep124 = 8.858/9.270，**仍差 −5.24/−5.43**；跨语料 7.029/7.713 | `README.md:232`；`P2D-report.md:204-207` |
| **ckpt epoch** | ep86 / ep99 / ep124（单卡 100(+30) ep） | 论文 1000 ep / 8×3090 | ep86→ep99 **+0.108/+0.080**；ep99→ep124 **+0.132/+0.148**（同协议可加） | #8/#9/#10 |
| **tt 3,000 vs 9,169** | LRS2 tt=3000；RealSSA-Sim tt=9169 | 两个不同语料 | **不可横向比**；同语料内子集 300 条比 3000 条低 **−0.113/−0.221** | `mix.json` 长度；`P2B-report.md:74,81` |
| **s1 单向 vs 双向 6,000** | 评测用 3,000（只 target=s1） | 仓库另有 `falcon_lrs2_2mix_paper`（tt=**6000**，s1/s2 双向） | **从未评测过**，方向未知 → GAP | `/root/data/falcon_lrs2_2mix_paper/tt/mix.json` = 6000 |
| **SDR 滤波器长度** | 512 | 他文常不声明 | 512→256/128/64 = **−0.09/−0.21/−0.35**；512→2048 = **+0.257** | `P2A-report.md:16`、`P2F-report.md` §1 |
| **SI-SDR 去均值** | `zero_mean=True` | 旧口径 False | **−0.02**（P2A）/ **−0.0005**（P2F） | 同上 |
| **plain-SNR vs SI-SDRi** | 投影式 SI-SDR | 不做投影的 SNR | **+0.34**（更乐观） | `P2A-report.md:16` |
| **采样率 16k vs 8k** | 16k | 8k 变体 | ΔSI-SDR **+0.111** / ΔSDR **+0.211** | `P2F-report.md` §1 |
| **DNSMOS 版本** | full4（已修窗口起点） | full3（旧） | OVRL **+0.0040**、P808 **+0.0079**；**mix_* 逐位不变** | §4.4 |
| **DNSMOS 内部窗** | speechmos `INPUT_LENGTH=9.01s`、hop=1 s | 论文未声明 | 长音频被切成 ≥9 s 窗取均值；<9.01 s 的片段会退化 | `speechmos/dnsmos.py:12,65-73` |

### 3.3 不可比却并列的数字（需改写表述）

1. **RealSSA-Sim 与论文结果**：论文报告 7.7/8.2；发布 `falcon.pt` 在 RealSSA-Sim tt、四通道均值和整段前向下得到 **7.774/8.479**。论文使用 channel-1；发布 checkpoint 与论文表格权重的身份及训练语料仍待确认。两组数据的通道与权重身份应随结果一起说明。
2. **ch0 行 "✅ +0.28/+0.25"**：通道轴修好了（对齐论文 channel-1），但 **ckpt 轴仍不同** → 只能写「口径向论文靠拢一步」，不能写"复现"。
3. **LRS2 7.029/7.713 与 14.1/14.7 并列**：STATUS 已标注"跨语料基线，不是论文列"（正确）；但同列排版容易被读成"复现失败"，换成"复现"两字则是错的——两个数分别是 LRS3 训练权重与 LRS2 训练权重的结果。建议补一列"ckpt 训练集"。
4. **ep124 "距论文 −5.24/−5.43"**：这是**训练预算差**，不是协议差。同表里"官方权重 −7.07/−6.99"才是语料不可比项。两者混在同一"结论"列里会误导。
5. **A800 47.1/56.5 ms 与论文 3090 51.3 ms**：硬件不同，且本地两个数没有原始测量文件。并列会得出"我们更快"的结论，而这个结论没有依据。
6. **DNSMOS 2.7941 与论文 3.276**：STATUS 判 ❌ 正确；需常驻提示"该 3.276 属 RealSSA-trained ckpt，同 ckpt 的 VoxCeleb2 版是 2.878（`PAPER_ALIGNMENT:158`），与本机 2.7941 同量级"。

### 3.4 覆盖率与断言

| 项 | `audio_visual_test.py` | `eval_realssa_real.py` |
|---|---|---|
| 覆盖率断言 | **当时（`6c1d5b7`）无**：全文 23 处 `raise`（`:144,153,161,176,178,192,197,248,257,260,262,289,299,303,321,328,360,391,474,480,576,583,589`）没有一条校验"完成句数 == 总数"。**2026-11-08 已补**三层自检：各 rank 的 planned/produced、跨 rank 合计 vs `planned_total`、合并 CSV 行数 vs `planned_total`（`_check_coverage`，`:570-700`），任一处不一致即非 0 退出；`--allow-partial` 显式降级为 warning 并把 `verified` 记 false（`9a920ce`）。`raise` 现 28 处（`STATUS.md:141`） | **有**（`2062528` 新增）：逐样本 `filled[]` 位图，循环后 `if not filled.all(): raise RuntimeError("windowed inference coverage failure: …gaps…")`（`:428-459`） |
| 事后可核 | `[eval] test index: … \| utterances=N \| this rank=M`（`:435-436`）+ `metrics_counts.json.n_utterances`（`:768-789`） | `summary.json` 的 `segments_total/segments_ok/segments_error`（`:544-563`） |
| 失败行为 | 任一 `sf.read`/前向/指标异常 → 直接抛 → `traceback` + `sys.exit(1)`（`:871-875`）；**不会跳过继续** | 单段异常 → 记 `row["error"]` 继续（`:681-687`），`--fail-fast` 才中止（`:684-686`） |
| 指标级 NaN | `nanmean` 静默跳过，仅 `n_*` 留痕（`allwrapper.py:147-160`、`audio_visual_test.py:741-752`） | `score_array` 遇到 NaN/Inf 直接 `raise`（`eval_dnsmos.py:76-77`） |

**实测分母修正**（这是最实质的一处口径瑕疵）：

| 文件 | `si-snr_i` 有限值 | `sdr`/`sdr_i` 有限值 | `pesq` 有限值 | 后果 |
|---|---|---|---|---|
| `P2E-realssa-sim-metrics_combined.csv` | **9169/9169** | **9166/9169** | **9166/9169** | STATUS 写"9,169 条"对 SDRi/PESQ **不成立**；3 条 NaN 是 `00003_02_41188.wav`、`00012_02_84551.wav`、`50016_74250.wav`（两个通道变体**同为这 3 条** → 数据本身退化，非随机） |
| `P2E-realssa-sim-ch0-…-20261030.csv` | 9169/9169 | 9166/9169 | 9166/9169 | 同上 |
| `P2D-eval-falcon-full.csv` | 3000/3000 | 3000/3000 | 3000/3000 | 无 NaN（LRS2 全部 2 s，PESQ 够长） |
| `P2D-eval-epoch{86,99,124}.csv` | 3000/3000 | 3000/3000 | 3000/3000 | 无 NaN |

日志侧证据：`P2E-realssa-sim-eval-official.log:3-11` 明确打印了 **6 次** `SDR computation failed (…singular…)`（行 3,4,6,7,9,10）与 **3 次** `PESQ computation failed (b'No utterances detected')`（行 5,8,11）—— 所以"9,169 条全有效"这句话在证据层面就是自相矛盾的。

**失败句分布**（说明"均值被少数句拖累"的程度，也说明为什么不能加"剔除失败句"的美化）：

| 语料 / ckpt | SI-SNRi<0 占比 | SI-SNRi<−10 占比 | min | p1 | median |
|---|---|---|---|---|---|
| RealSSA-Sim（mean） | **0.414%**（38/9169） | 0.065% | −47.70 | 1.75 | 7.66 |
| RealSSA-Sim（ch0） | 0.338% | 0.065% | −44.47 | 1.95 | 7.77 |
| LRS2 官方 `falcon.pt` | **2.167%**（65/3000） | 0.067% | −13.10 | −1.52 | 7.31 |
| LRS2 自训 ep124 | 0.400% | 0.000% | −7.64 | 1.56 | 9.05 |

→ 没有任何剔除逻辑，均值里含着这些负值；这与论文口径一致（论文也未声明剔除），**但报告里若写"9,169 条全有效"就是错的**。

### 3.5 `--limit` 使用审计

| run | `--limit` | 实际条数（日志） | 报告里的标注 | 是否被当全量 |
|---|---|---|---|---|
| P2A `small-200` | 200 | 200（随机 seed0 子集） | `P2A-report.md:13` 明写"固定 200 条随机子集" | 否 |
| P2A `anchor-first20` | 20 | 20 | `P2A-report.md:43` 明写 anchor | 否 |
| P2B `b300` | 300 | 300 | `P2B-report.md:74-80` 表头写 `300`；同表另起一行 **3000** 标全量 | 否 |
| P2D 官方/ep86/ep99/ep124 | 未给 | **3000**（`limit=all`） | `P2D-report.md:196-199` | 否 |
| P2E RealSSA-Sim（两版） | 未给 | **9169**（`limit=all`） | `P2E-report.md:246` | 否 |
| RealSSA-Real DNSMOS | 未给 | **54/54 段**（`limit: 0`） | `run_metadata.json` 的 `"limit": 0` | 否 |

结论：**没有发现 `--limit` 子集被当全量的实例**。唯一需要提醒的是 `audio_visual_test.py:416-431` 的 `--limit` 在 **多卡时先截断再 `indices[rank::world_size]`**，所以 `world_size=2 --limit 5` 得到 3+2 条（总数仍是 5，不会翻倍），语义正确。

---

## 4. 实测验证

环境：`/root/miniforge3/envs/falcon/bin/python`（Py3.11.17 / torch 2.8.0+cu128），`CUDA_VISIBLE_DEVICES` 用物理卡 1，A800-SXM4-80GB，跑前后 GPU 空闲。**仓库零写入**：cwd 全部在 `/root/tmp/w107_eval`（`git status --porcelain` 前后均为空）。

### 4.1 5 样本复现：官方 `falcon.pt` vs `P2D-eval-falcon-full.csv`

命令（可直接复跑）：

```bash
REPO=/root/code/github_repos/Falcon-fork
ENV=/root/miniforge3/envs/falcon/bin/python
mkdir -p /root/tmp/w107_eval && cd /root/tmp/w107_eval
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/audio_visual_test.py \
  --conf_dir $REPO/configs/LRS2_2Mix.yml \
  --ckpt_path /root/.cache/falcon_weights/falcon.pt \
  --data_dir /root/data/falcon_lrs2_2mix \
  --gpus 1 --limit 5
```

关键 stdout：

```
[eval] test index  : /root/data/falcon_lrs2_2mix/tt
[eval] ckpt_path   : /root/.cache/falcon_weights/falcon.pt
🚀 [eval] 1 worker(s) | gpus=1 | limit=5
3000 3000
Drop 0 utts(0.00 h) from 3000 (shorter than None samples)
[eval] test index: /root/data/falcon_lrs2_2mix/tt | utterances=5 | this rank=5 | device=cuda:0
   ✓ audio_model loaded: 525 keys from falcon.pt
   ✓ video_model loaded: 137 keys from falcon.pt
Final metrics (per-utterance, all workers):
sdr           12.936967     sdr_i  7.567688
si-snr        11.823868     si-snr_i  6.522868
pesq           1.948288     stoi   0.935970
si-snr_mix     5.301000     sdr_mix 5.369279
Valid counts (finite values per column):
sdr=5/5, sdr_i=5/5, si-snr=5/5, si-snr_i=5/5, pesq=5/5, stoi=5/5, si-snr_mix=5/5, sdr_mix=5/5
```

逐行对拍 `P2D-eval-falcon-full.csv` 前 5 行（同一 `snt_id` 键连接，5/5 全部匹配）：

| 列 | max\|Δ\| | mean\|Δ\| |
|---|---|---|
| `sdr` / `sdr_i` | **7.451e-08** | 3.821e-08 |
| `si-snr` / `si-snr_i` | **5.317e-08** | 3.266e-08 |
| `pesq` | **2.265e-05** | 8.202e-06 |
| `stoi` | **1.789e-09** | 9.972e-10 |

→ **复现成功**，误差为 float32/内核级噪声（PESQ 的 2e-5 来自 `pypesq` 内部 FFT）。恒等式逐行：`0 / 1.776e-15 / 0 / 1.776e-15 / 0`（max 1.776e-15）。

### 4.2 整段 vs 分窗（同 5 句、同权重、同数据）

```bash
cd /root/tmp/w107_eval
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/tools/repro/phase2/compare_variants.py \
  --data-dir /root/data/falcon_lrs2_2mix --limit 5 \
  --variants small --protocols whole,windowed --visuals real \
  --device cuda --tag w107 --out-dir /root/tmp/w107_eval
```

| protocol | n | SI-SNRi | SDRi | PESQ | STOI | wall |
|---|---|---|---|---|---|---|
| whole（= `audio_visual_test.py` 路径） | 5 | **6.523** | **7.568** | 1.948 | 0.936 | 2.1 s |
| windowed（Studio 2 s/0.5 s/80 ms/320 fade） | 5 | **4.875** | **6.031** | 1.742 | 0.920 | 1.6 s |
| **Δ（win − whole）** |  | **−1.648** | **−1.537** | −0.206 | −0.016 |  |

两处交叉验证：
- `whole` 的 6.523/7.568 与 4.1 节独立入口的 6.522868/7.567688 一致到 **1e-9**（两个入口走同一条实现路径）；
- 5 句的 −1.648 与 300 句的 −1.912（`P2B-report.md:92`）同向同量级，说明**样本少不是问题**，协议差是稳定的。

### 4.3 确定性（两次独立全量 run 的逐行对拍）

| 对比 | 行序 | `sdr` | `si-snr_i` | `pesq` | `stoi` |
|---|---|---|---|---|---|
| falcon-full（10-24）vs falcon-full-cols（10-31） | 3000/3000 相同 | 2.220e-07 | **2.308e-07** | 9.394e-05 | 3.887e-08 |
| epoch99（10-24）vs epoch99-cols（10-31） | 3000/3000 相同 | 3.262e-07 | **3.307e-07** | 4.330e-04 | 1.773e-08 |
| epoch86（10-24 日志）vs epoch86（10-30 重跑日志） | — | 9.041506 == 9.041506 | **8.617919 == 8.617919** | — | — |

→ 同一 ckpt + 同一数据 + 同一脚本，跨 7 天/跨机器状态重跑，**六位小数完全相同或差 1e-7**。**"数字不可复现"不是本项目的风险；风险全在口径。**

### 4.4 DNSMOS 专项：full3 → full4（窗口起点 bug）

**bug 与修复（commit `2062528`，2026-10-30 18:04 UTC，`tools/repro/phase2/eval_realssa_real.py`）**

```diff
@@ def _extract_windowed(runtime, audio, mouth) ...
-    endpoint = LEFT + HOP                              # 首块绝对起点 = 0
+    filled = np.zeros(samples, dtype=bool)             # 覆盖率断言用
+    # Studio 语义：首窗输出绝对 [0, HOP)，因此 endpoint 初值 = HOP + GUARD（不是 LEFT + HOP）。
+    endpoint = HOP + GUARD                             # 9280; abs_start = endpoint-(HOP+GUARD) = 0
@@
-            abs_start = endpoint - (WINDOW - LEFT)
+            abs_start = endpoint - (WINDOW - LEFT)     # = endpoint-(HOP+GUARD)
             chunk = pred[LEFT:LEFT + HOP]
             lo2, hi2 = max(0, abs_start), min(samples, abs_start + HOP)
             if hi2 > lo2:
                 out[lo2:hi2] = chunk[lo2 - abs_start:hi2 - abs_start]
+                filled[lo2:hi2] = True
             endpoint += HOP
+    if not filled.all():
+        ... raise RuntimeError("windowed inference coverage failure: ...")
```

数学形式：`LEFT = WINDOW − GUARD − HOP = 32000−1280−8000 = 22720`，每窗写 `[endpoint−GUARD−HOP, endpoint−GUARD)`。
- 旧：`endpoint₀ = LEFT+HOP = 30720` → 首窗写 `[21440, 29440)`，**`[0, 21440)` 永远不被写 → 21,440 样本 = 1.34 s 静音前缀**。
- 新：`endpoint₀ = HOP+GUARD = 9280` → 首窗写 `[0, 8000)`，覆盖率 100%。
- 与 Studio 语义对齐的旁证（P2E 报告 §10.1 引用）：`studio/server.py:266`（`start=endpoint−GUARD−HOP, end=endpoint−GUARD`）、`studio/web/app.js:624`（首窗 `end = sessionEpoch+HOP+GUARD`）。

**两版数值对比（54 段全跑，54/54 成功，0 error）**

| 指标 | full3（作废） | full4（有效） | Δ | 逐段 max\|Δ\| |
|---|---|---|---|---|
| SIG | 3.2209 ± 0.3691 | **3.2226 ± 0.3703** | +0.0017 | 0.3348 |
| BAK | 3.6921 ± 0.2640 | **3.6981 ± 0.2598** | +0.0060 | 0.1545 |
| OVRL | 2.7901 ± 0.4082 | **2.7941 ± 0.4100** | **+0.0040** | **0.2472** |
| P808 | 3.3285 ± 0.4045 | **3.3365 ± 0.4066** | +0.0079 | 0.1249 |
| mix_SIG / BAK / OVRL / P808 | 2.4176 / 1.7908 / 1.7599 / 2.6765 | 同左（**逐位不变**） | +0.0000 | 0 |

我做的逐段 diff（`full3/per_segment.csv` ⨝ `full4/.../realssa_real_dnsmos_per_segment.csv`，54 段全匹配）：
- OVRL `|Δ|` 按段长分层：**0–30 s（n=4）mean 0.0820 / max 0.2472**；30–60 s（n=11）mean 0.0422 / max 0.1244；60–200 s（n=39）mean 0.0288 / max 0.0777。
- 最大 OVRL 变动段 = `outdoor/street_side-2/2-1`（29.68 s，静音前缀占 4.5%）。
- **mix_* 四列逐位相同**；该对照不能单独证明 DNSMOS 侧所有条件均无漂移（与 `P2E-report.md:281` 的声明一致）。

**为什么幅度这么小（可验证的解释）**：`speechmos/dnsmos.py:12` `INPUT_LENGTH = 9.01`、`:65-73` 以 **9.01 s 窗 + 1 s hop** 滑动取均值。1.34 s 静音只落在首窗，占该窗内容的 15%，而段长 20.6–153.4 s（`run_metadata.json` 合计 5,037.6 s / 54 段，均 93.3 s），首窗在 N 个窗里只占 `1/N`。→ 均值移动 +0.002~+0.008 MOS 完全可解释，且**短段影响更大**（实测就是 0–30 s 组 |Δ| 最大）。

**full3 作废的判定**：不是"幅度大所以要重跑"，而是"**输出确定性错误**"——每条估计开头 1.34 s 恒为 0，这是可被任何第三方测出来的 bug（P2E §10.2 用合成探针复现：旧初值未覆盖 21,440/64,000 样本，新初值 0 未覆盖）。因此 full3 的 4 个值不进入任何对外表述，`…full3/` 只作历史证据。

副产品（我核对到的实物清单）：
- `DNSMOS-realssa-real-mean-full4-20261031/`：`summary.json`(77L)、`run_metadata.json`(97L)、`realssa_real_dnsmos_per_segment.csv`(55L)、`P2E-realssa-real-full4.log`(117L)。
- 54 段来自 **40 个 recording**（部分 recording 切成 `-1/-2` 段）；分场景：big_indoor 10/1234.6 s、mid_indoor 10/1261.3 s、outdoor 14/1288.7 s、small_indoor 20/1253.0 s。
- `/root/data/falcon_eval/` 只有 `realssa_real_mean_full{,2,3}` 三个目录，**没有 full4**（full4 直接落在仓库 `docs/verification/phase2/` 下）→ 数据目录与证据目录不对称，见 §7-G7。

---

## 5. 坑与反直觉点

1. **"causal" 不是"流式"**：配置里 `causal: true`（`configs/LRS2_2Mix.yml:13,32,67,83`）只作用于层内（因果卷积/RNN），但 `AudioBlock.gateway` 与 `AudioBlock.projection` 被**硬编码 `causal=False`**（`Falcon.py:727,736`），且 `torch.stft` 是 `center=True`（两端各 pad 128）、`cLNhw` 是全窗 GN（`configs/LRS2_2Mix.yml:18,38`；实现 `look2hear/layers/normalizations.py:229` 的 `LayerNormalizationHW`）→ 模型**不是严格因果**，离线整段评测因此天然优于流式。`STATUS.md:50` 写的"AudioBlock 对称卷积（causal=False）"指的就是这两处，但要连着配置里的 `causal: true` 一起读，否则会误以为整网非因果。见 §7-G2。
2. **评测入口永远不会"切窗"，所以模型看到的分辨率与训练一致（LRS2 都是 2 s）**，但 RealSSA-Real 是 20.6–153.4 s：那里只能切窗（整段前向会 OOM），于是 DNSMOS 数字里混着窗口化损失，而 LRS2 的 SI-SNRi 里没有。**两个语料的数字不是同一种推理协议**（这正是 §3.3 第 3 条要点名的）。
3. **`drop_utt` 死代码**：`AVSpeechEvalDataset.__init__` 打印 `"Drop 0 utts(0.00 h) from 3000 (shorter than None samples)"`（`:171-175`），但 `drop_utt/drop_len` 从未被 `+=`，阈值是 `None`。→ **看起来"有过滤"，实际零过滤**；真正会掉句的唯一途径是 `sf.read` 抛异常（直接崩）。
4. **`self.length = orig_len` 的静默截断隐患**（`:166-176`）：见 §1.3。加第二个 source json 会让数据集"合法地"只剩一半句数，而日志会打印 `Drop 0 utts … from N`，看不出异常。
5. **`mix.json` 里的声明长度没人校验**（`:184-192` 用 `stop=None`）。RealSSA-Sim 的 `mix.json[0]` 声明 92,800、LRS2 声明 32,000 —— 一旦数据被替换，评测不会察觉。
6. **PESQ/STOI 在 `n_src>1` 时是 nan，但 `n_src` 取自 `clean` 而非 `estimate`**（`allwrapper.py:101`）。若将来跑 2 源模型而 `clean` 仍是单 target，PESQ 会被"合法地"算出来但比的是 `est_src[0]`（`:113`），而 SI-SDR 走的是 PIT —— **两者会测不同的通道**。当前 `n_src=1` 不触发，但这是下一个改数据的人必踩的坑。
7. **`--limit` + 多卡是"先截断后分片"**（`:416-431`）：语义正确但反直觉（`--limit 5 --gpus 0,1` → rank0 拿 3 条、rank1 拿 2 条）。
8. **结果目录不看参数、只看 cwd + `exp_name`**（`:446-448`）：换 ckpt 不换 cwd = 覆盖上一条 CSV。历史上 P2E/P2D-epoch124 就是直接写进仓库树（`.gitignore` 的 `results/` 救了 git status）。
9. **`_merge_results` 先删分片再返回**（`:801-804`）：`metrics_rank{r}.csv` 只在崩溃时留下，正常结束即消失 → 事后无法核对"逐句行是否被改动"，只能靠 `metrics_counts.json` + 恒等式。
10. **DNSMOS 的"整段"其实是 9.01 s 窗均值**（`speechmos/dnsmos.py:12,65-73`），且 `score_array` 在 `peak>1` 时会先做峰值归一（`eval_dnsmos.py:81-83`）。→ 与论文"DNSMOS=3.276"的自定义实现细节不可能逐位对齐。
11. **PESQ 在 3 条 RealSSA-Sim 句子上报 `No utterances detected`**（`P2E-realssa-sim-eval-official.log:5,8,11`），`SDR` 有 6 次 `singular matrix`、`PESQ` 有 3 次。这些 NaN 被 `nanmean` 静默吞掉，**只有 `n_*` 列能看出来**（而 STATUS 引用的 10-24 版 CSV 还没有 `n_*` 列）。
12. **恒等式检查与独立指标检查**：`si-snr_i ≡ si-snr − si-snr_mix` 是 `allwrapper.py:105` 直接相减得到的，恒等式核对了这三列的算术关系；CSV 文件身份可通过哈希核对。真正的外部校验是 `P2F` 的独立实现（1.8e-7 dB）。

---

## 6. 与既有文档的冲突/更正

| # | 旧说法（出处） | 实际（证据） | 影响 |
|---|---|---|---|
| C1 | `P2E-report.md:107`：「`infer_file.extract`（**整段推理**；模型输入按 2 s 窗口因果前向…）」 | `eval_realssa_real.py:461-464` 的 `separate()` **直接 return `_extract_windowed(...)`**；全脚本只有窗口化路径，整段前向已被 OOM 理由废弃（`:381-383` 注释） | 描述自相矛盾；会让人误以为 RealSSA-Real 是整段评测。应改为「窗口化推理（2 s/0.5 s/80 ms，**无 crossfade**）」 |
| C2 | `P2E-report.md:246`：「RealSSA-Sim test（9,169 条…**整段推理**）」 | 这一条是**对的**：`P2E-realssa-sim-eval-official.log:24` 显示 `utterances=9169`、走 `audio_visual_test.py`（整段，无窗） | 与 C1 形成"同报告内两句口径不一致"，需分语料分别声明 |
| C3 | `P2D-report.md` §8.3 曾把论文硬件写成 10 卡/global 40，已于 `8b9d903`／2026-11-09 校正为 8×3090、bs4/GPU（global 32） | `STATUS.md` 与 `PAPER_ALIGNMENT_20261030.md:22` 均为 **8×RTX3090、bs4/GPU、global 32**（论文 Table 16） | 该句已校正；训练预算归因统一按 8×3090/global 32 计，"还差 5.2 dB"的分母不再被夸大 |
| C4 | `STATUS.md:50`：「AudioBlock 对称卷积（`causal=False`）」 | 配置全链 `causal: true`（`LRS2_2Mix.yml:13,32,67,83`），只有 `gateway`/`projection` 两处硬编码 `causal=False`（`Falcon.py:727,736`）；`stft(center=True)` | 表述不完整 → 见 §5-1、§7-G2 |
| C5 | `STATUS.md §3` 用 `P2E-realssa-sim-ch0-{eval-20261030.log,metrics_combined-20261030.csv}` 花括号简写引用 | 两个实物都存在（`ls` 已核）；但花括号简写在 Markdown 里不可点击，且与同表其它行的裸文件名风格不一致 | 只是可读性/可点性问题，建议拆成两个完整文件名 |
| C6 | `STATUS.md §3`「9,169 条」同时用于 SI-SNRi 与 SDRi | SI-SNRi n=9169，**SDRi/PESQ n=9166**（3 条 NaN；日志共 9 条 warning：6 次 SDR + 3 次 PESQ） | 分母不同却写成一个数 → 需修口径 |
| C7 | `P2A-report.md:16`：「best-of-s1/s2（PIT 上界）**+0.02 dB**」 | `P2F-report.md` §1：「2-target oracle 上界 **+0.000 dB**」 | 两处量化不一致（0.02 vs 0.000）；结论方向一致（可忽略），但应统一到一个数并注明实现 |
| C8 | `STATUS.md:44`「A800 单窗 small 47.1 ms / big 56.5 ms」 | 唯一证据是 `deepdive-20261024/06-studio-backend.md:402-403` 的 md 表格，**无原始 json/log** | 属"文档级证据"，不能与论文 3090 数字并列 |
| C9 | `P2E-report.md:288,306`：「full3 作废」 | 与代码/数值一致（§4.4 复核通过），**但 `docs/STATUS.md §3` 仍并列引用 full3 的 2.7901/3.3285 作为"作废对照"** | 建议 STATUS 只保留 full4，作废值移到脚注，避免被二次引用 |

> 未发现冲突的既有结论（已被我独立复核）：`7.774/8.479`、`7.977/8.452`、`2.7941/3.3365`、`7.029/7.713`、`8.618/9.042`、`8.726/9.122`、`8.858/9.270`、`P2A` 的 `mix=s1+s2 / target=s1`、`P2F` 的 `baseline mean 4.994`、恒等式成立性（我独立复算 max 1.776e-15）。

---

## 7. GAP LIST

| # | 问题（可直接认领） | 为什么现在答不了 | 建议动作 |
|---|---|---|---|
| G1 | **LRS2-2Mix 论文列 14.1/14.7 无本地证据** | 论文按 benchmark 分别训练，LRS2-trained ckpt 未发布；本机最好 8.858/9.270（单卡 130 ep） | 8×3090 跑 `configs/LRS2_2Mix_Paper_v2.yml`（验收 tt 3000 SI-SNRi ≥13.6）；或向作者索取 LRS2 ckpt |
| G2 | **因果性/感受野与"窗口级因果"的确切边界**（STATUS 的 57 帧≈456 ms 只是一遍 AudioBlock 的几何值） | 需要逐层算 `receptive_field` + 实测遮挡实验；属 W1-05/06 范围，我只确认了 `gateway/projection: causal=False` 与 `stft(center=True)` | 用 `verify_causality.py` + 逐层 pad 分析复核三档数字：单遍几何 57 帧/456 ms、整链几何 113 帧/904 ms、as-shipped 依赖到窗尾（≤2000 ms）；详见 W2-12 |
| G3 | **双向 6,000 条 tt（`falcon_lrs2_2mix_paper`）从未评测** | 需要 GPU 全量跑（6000 条 ≈18 min） | 用 ep124 跑一遍，量化"s1 单向 3000 vs 双向 6000"的差 |
| G4 | **RealSSA-Real 窗口化损失未量化**（DNSMOS 里混了多少窗口化损失？） | 整段前向 20.6–153.4 s 会 OOM，无法做 whole 对照；且无 clean 参考 | 取最长/最短各 1 段，用 chunk 拼成"伪整段"（大显存或分段 overlap-add）对照 DNSMOS |
| G5 | **PESQ/STOI 的 `n_src` 语义坑**（`allwrapper.py:101,113`） | 当前 n_src=1 不触发；需构造 n_src=2 + 单 target 的合成用例才能证明"测错通道" | 写一个单测：est 两路、clean 一路，断言 PESQ 用的是哪一路 |
| G6 | **P2A(+0.02) vs P2F(+0.000) 的 oracle 上界分歧** | 两者实现不同（P2A 用 `gap_analysis.py` 自算，P2F 用独立 NumPy），我没复跑 | 对比两脚本在同一 50 条上的 run，定位差异 |
| G7 | **`/root/data/falcon_eval` 与证据目录不对称**（无 full4） | full4 产物直接写进仓库 `docs/verification/phase2/` | 把 full4 三件套复制一份到 `falcon_eval/realssa_real_mean_full4`，或反相在 STATUS 注明 |
| G8 | **`P2B-report.md:140` 仍在解释一个 0 字节文件**（`P2B-full-big-whole-real.csv`） | 我实测该文件**已不在** `docs/verification/phase2/`，且该目录下已无任何 0 字节 CSV | 文档陈述过期，删掉那条注解即可（不影响任何数字） |
| G9 | **LRS3-2Mix / VoxCeleb2-2Mix 完全无数据** | 数据集未发布（HF 仅 README） | 只能引用论文值并标注"不可复现"，不假装评测 |
| G10 | **3090 延迟/RAM/显存无法对标** | 本机只有 A800；A800 的 47.1/56.5 ms 又无原始测量文件 | 在 A800 上按论文协议（CUDA events、median of 25、bs1、2 s）重测并把 json 落到 `docs/verification/` |

---

## 8. 证据清单

### 8.1 源码（远端仓库，HEAD `6c1d5b7`）

| 文件 | 关键行 | 用途 |
|---|---|---|
| `audio_visual_test.py` | 33, 110-141, 144-240, 242-374, 378-399, 401-545, 550-568, 570-875 | 入口全流程（文件现 875 行；本列行号为现行口径） |
| `look2hear/datas/avspeech_dataset.py` | 130-219（147-150, 162-176, 184-192, 194-206, 208-219） | 数据加载/整段读取/死代码 |
| `look2hear/metrics/allwrapper.py` | 11-64（CSV schema）, 98-129, 132-145, 147-160 | 指标与 CSV |
| `look2hear/metrics/core.py` | 69-79, 103-141, 152-185, 196-217 | SI-SDR/SDR/PESQ/STOI 实现 |
| `look2hear/metrics/wrapper.py` / `splitwrapper.py` / `__init__.py` | 26-96 / 17-112 / 1-30 | 历史口径（`one_/two_`）与导出口 |
| `look2hear/losses/pit_wrapper.py` | 152-237（168-171, 204-237）, 227-229 | PIT 退化 + `threshold_byloss` |
| `look2hear/models/Falcon.py` | 693-780（727,736）, 978-1059, 1114-1128, 1170-1199, 1284-1347, 1480-1499 | 前向/STFT/ISTFT/causal 混用 |
| `configs/Falcon.yml` | 12, 14, 36, 161, 163-165, 173 | n_src=1 / test_dir 哨兵 |
| `configs/LRS2_2Mix.yml` | 11, 13, 23-31, 67, 83, 142-147, 154 | n_src=1 / STFT 参数 / normalize |
| `configs/LRS2_2Mix_Paper_v2.yml` | 28, 30, 161-165 | 论文配方版 n_src=1 |
| `tools/repro/phase2/compare_variants.py` | 58-66, 120-145, 149-178, 182-227, 230-240 | 整段/分窗消融 |
| `tools/repro/phase2/eval_realssa_real.py` | 381-459, 461-464, 621-623, 648-700 | 窗口化+覆盖率断言+DNSMOS 循环 |
| `tools/repro/phase2/eval_dnsmos.py` | 68-90 | `score_array` 峰值归一 |
| `tools/repro/phase2/eval_p2d.sh` | 1-45 | P2D 复跑脚本（`--limit` 可选） |
| `/root/miniforge3/envs/falcon/…/speechmos/dnsmos.py` | 12, 65-73 | DNSMOS 9.01 s 窗/1 s hop |

### 8.2 数据与索引（远端）

| 路径 | 事实 |
|---|---|
| `/root/data/falcon_lrs2_2mix/tt/{mix,target,mix.json,target.json,near.json}` | `mix.json` **3000** 条，每条 `[wav, 32000]`；`target.json[0] = [wav, mouth.pt, 32000]` |
| `/root/data/falcon_lrs2_2mix/conversion_stats.json` | `"target": "s1"`、`"near_mode": "fallback"`；tr 20000 / cv 5000 / tt 3000 |
| `/root/data/falcon_lrs2_2mix_paper/tt/mix.json` | **6000** 条（双向 target，从未用于评测） |
| `/root/data/falcon_realssa_sim/tt/mix.json`、`…_ch0/tt/mix.json` | 各 **9169** 条，首条 92,800 样本 |
| `/root/data/RealSSA-Real/` | 官方 inference view：54 段 / 40 recording；`metadata.jsonl` + `mouth/` + `raw/` |
| `/root/data/falcon_eval/realssa_real_mean_full{,_2,_3}/` | full4 **不在**此处（见 G7） |
| `/root/.cache/falcon_weights/falcon.pt` | keys `['state_dict','architecture','epoch']`；`architecture=FalconSpeakerContrastive`、`epoch=714`；`falcon-big.pt` `epoch=715`、`architecture=SpeakerContrastiveV2` |

### 8.3 证据文件（`docs/verification/phase2/`，md5 已核）

| 文件 | 行数 | md5 / 备注 |
|---|---|---|
| `P2D-eval-falcon-full.csv` | 3003（avg `:3002`, std `:3003`） | `ff9b9d74912d8111c1759066193b3754`（与 `P2D-report.md:241` 记载一致） |
| `P2D-eval-epoch99.csv` | 3003 | `5a7e307f2916df5990b45c40f9e099b6`（同上） |
| `P2D-eval-epoch86.csv` | 3003 | `978302f1d1fcb152050a4fa8eba79c74`（10-30 重跑恢复） |
| `P2D-eval-epoch124-contrastive.csv` | 3003 | `b1e018e95ae551c644568baa78d37f9e` |
| `P2D-eval-falcon-full-cols-20261031.csv` / `.counts.json` | 3003 / 1 | 恒等式 max 4.441e-15；n_utterances 3000 |
| `P2D-eval-epoch99-cols-20261031.csv` / `.counts.json` | 3003 / 1 | 恒等式 max 5.329e-15；n_utterances 3000 |
| `P2E-realssa-sim-metrics_combined.csv` | 9172（avg `:9171`） | `d53135e9de610122d55883b540fc6e76` |
| `P2E-realssa-sim-ch0-metrics_combined-20261030.csv` | 9172（avg `:9171`） | `043a37632d0b26caaea3a1024611a480` |
| `P2E-realssa-sim-eval-official.log` / `…ch0-eval-20261030.log` | — / — | 前者含 6 次 SDR `singular` + 3 次 PESQ `No utterances detected`（行 3-11） |
| `DNSMOS-realssa-real-mean-full3/{summary.json,per_segment.csv,run_metadata.json}` | 65 / 55 / — | **作废**，仅历史 |
| `DNSMOS-realssa-real-mean-full4-20261031/{summary.json,run_metadata.json,realssa_real_dnsmos_per_segment.csv,P2E-realssa-real-full4.log}` | 77 / 97 / 55 / 117 | OVRL 2.7941、P808 3.3365、54/54、`limit:0`、speechmos 0.0.1.1 + onnxruntime 1.31.0 |
| `P2B-b300-{small,big}-{whole,windowed}-real.csv`、`P2B-b300-*-{zero,shuffle}.csv`、`P2B-summary-b300*.json` | 各 302 / 55 KB | 整段 vs 分窗、视觉消融（旧口径，正数 SI-SDRi） |
| `P2A-report.md`（5, 13, 16, 18, 23, 43, 62-64, 96, 116-128） | — | `target=s1`、`mix=s1+s2`、变体量化、mouth 对齐 |
| `P2B-report.md`（74-81, 91-96, 138-142） | — | 300/3000 行、windowed 惩罚、0 字节空文件注解（已过期，见 G8） |
| `P2D-report.md`（35-36, 196-227） | — | 终态表、精确值、md5、复跑命令 |
| `P2E-report.md`（107, 245-251, 261-320） | — | DNSMOS 修复与两版对比（107 行有 C1 冲突） |
| `P2F-report.md`（§1 TL;DR 1-4） | — | 第三方独立对拍（1.8e-7 dB）、baseline 4.994、oracle +0.000 |
| `docs/STATUS.md`（§3 矩阵、§4 口径、§6 待办） | 141 行 | 审计对象 |
| `docs/understanding/PAPER_ALIGNMENT_20261030.md`（21-22, 24, 93-94, 145, 158, 222-233） | — | 论文协议（Table 16 = 8×3090/bs4/global 32 在 `:22`）/Table 3 对照 |
| `README.md`（43, 122, 232） | — | `falcon.pt` = `LRS3_complex_nonmoving`（ep714）、14.1/14.7 需 LRS2-trained ckpt |

### 8.4 本次实测产物（**不在仓库**，位于 `/root/tmp/w107_eval/`）

```
/root/tmp/w107_eval/Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv   # 5 句，17 列
/root/tmp/w107_eval/Experiments/checkpoint/LRS2_2Mix/results/metrics_counts.json    # n_utterances=5, 恒等式 max 1.776e-15
/root/tmp/w107_eval/P2B-w107-small-whole-real.csv                                   # 5 句，6.523/7.568
/root/tmp/w107_eval/P2B-w107-small-windowed-real.csv                                # 5 句，4.875/6.031
/root/tmp/w107_eval/P2B-summary-w107-small.json
```

复跑总命令（5 样本复现 + 整段/分窗对照，约 15 s，A800 单卡）：

```bash
REPO=/root/code/github_repos/Falcon-fork; ENV=/root/miniforge3/envs/falcon/bin/python
mkdir -p /root/tmp/w107_repro && cd /root/tmp/w107_repro
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/audio_visual_test.py \
  --conf_dir $REPO/configs/LRS2_2Mix.yml --ckpt_path /root/.cache/falcon_weights/falcon.pt \
  --data_dir /root/data/falcon_lrs2_2mix --gpus 1 --limit 5
PYTHONPATH=$REPO OMP_NUM_THREADS=4 $ENV $REPO/tools/repro/phase2/compare_variants.py \
  --data-dir /root/data/falcon_lrs2_2mix --limit 5 --variants small \
  --protocols whole,windowed --visuals real --device cuda --tag repro --out-dir /root/tmp/w107_repro
```

对拍脚本（把 5 行与我给的参照比）：

```bash
/root/miniforge3/envs/falcon/bin/python - <<'PY'
import pandas as pd
mine = pd.read_csv("/root/tmp/w107_repro/Experiments/checkpoint/LRS2_2Mix/results/metrics_combined.csv")
mine = mine[~mine.snt_id.isin(["avg","std"])]
ref  = pd.read_csv("/root/code/github_repos/Falcon-fork/docs/verification/phase2/P2D-eval-falcon-full.csv")
ref  = ref[~ref.snt_id.isin(["avg","std"])].head(5)
m = ref.merge(mine, on="snt_id", suffixes=("_ref","_mine"))
for c in ["sdr","sdr_i","si-snr","si-snr_i","pesq","stoi"]:
    print(c, "max|d|=", (m[c+"_mine"]-m[c+"_ref"]).abs().max())
PY
```

期望输出（我实测）：`sdr/sdr_i 7.451e-08`、`si-snr/si-snr_i 5.317e-08`、`pesq 2.265e-05`、`stoi 1.789e-09`。

---
