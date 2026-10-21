# 验证记录：层与 DSP（layers-dsp，R1）

> 对应文档：`docs/understanding/02-layers-dsp.md`
> 脚本：`docs/verification/scripts/verify_layers_dsp.py`（777 行，只读仓库、不改源码）
> 日志：`docs/verification/runs/layers-dsp-verify.log`
> 环境：`/root/.venvs/falcon/bin/python`（3.10.18 + torch 2.8.0+cu128，CPU）
> stub：`PYTHONPATH=$PWD/docs/verification/stubs:$PWD`（timm.DropPath / typeguard.check_argument_types）

## 1. 复现命令

```bash
cd /root/code/github_repos/Falcon-fork

# (1) 两份 stft.py 的同源证明
diff -u look2hear/layers/stft.py look2hear/utils/stft.py            # 15 行 / 2 hunk
diff -B  look2hear/layers/stft.py look2hear/utils/stft.py           # 仅 164d164（空白行）
diff    look2hear/layers/stft.py look2hear/utils/stft.py | grep -c '^<'   # 1
diff    look2hear/layers/stft.py look2hear/utils/stft.py | grep -c '^>'   # 1
for f in look2hear/layers/stft.py look2hear/utils/stft.py; do tr -d '[:space:]' < $f | md5sum; done
# -> 两个文件均为 3f7053135a33126c8d2a078b98fc2927（去空白后完全一致）

# (2) 动态验证（STFT 对拍 / 归一化语义 / 形状 / 因果性 / 参数）
PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
  /root/.venvs/falcon/bin/python docs/verification/scripts/verify_layers_dsp.py \
  | tee docs/verification/runs/layers-dsp-verify.log
```

## 2. 关键结果摘要

### 2.1 两份 stft.py：同源复制（仅 2 行纯空白差异）

| 项 | 值 |
|---|---|
| 行数 | 均 785 行 |
| `diff -u` 输出 | 15 行，2 个 hunk |
| 变更行 | `utils/stft.py` 第 1 行多一个空行；`layers/stft.py:164` 一行 4 空格空白被删 |
| 去空白 md5 | 两文件相同（`3f705313…`） |
| 结论 | **同一份代码的复制粘贴**，不是两份独立实现 |

### 2.2 STFT/iSTFT 与 torch 的等价性（float32）

| 对比 | shape | max abs diff |
|---|---|---|
| `forward_stft(librosa, hann, center=True)` vs `torch.stft` | `[2,129,251,2]` | `2.289e-05` |
| 同上 `center=False` | `[2,129,249,2]` | `2.289e-05` |
| `forward_stft(mode="torch")` vs `torch.stft` | `[2,129,251,2]` | `0.000e+00`（逐位） |
| `inverse_stft(librosa)` vs `torch.istft` | `[2,32000]` | `2.146e-06` |
| round-trip（librosa） | `[2,32000]` | `2.384e-06` |

`round_pow_of_two` 行为：`frame_len 200→fft 256→129 bins`；`256→256→129`；`400→512→257`；
帧数始终 `1+32000/128 = 251`。

### 2.3 归一化语义（与手算逐元素比对，全部 True）

| 检查 | 结果 |
|---|---|
| `GlobalLayerNorm == nn.GroupNorm(1,C)`，统计跨 (C,T,F) | ✅（`2.4e-07`） |
| `CumulateLN4D`：统计 = 跨 C 与 `t'≤t`，**逐频点 f 独立**；gain `[1,C,1,1]` | ✅ |
| `LayerNormalization4D((C,1))`：`dim=(1,)`，`γ=[1,C,1,1]` | ✅ |
| `LayerNormalization4D((C,F))`：`dim=(1,3)`，`γ=[1,C,1,F]` | ✅ |
| `cLN=CumulateLN`：3D 输入累积，gain `[1,C,1]` | ✅ |
| `cLNhw == CumulateLN4D(GroupNorm(1,C)(x))` | ✅ |
| `get("BatchNorm1d") is nn.BatchNorm1d` | ✅（非本文件包装类） |
| `get("BatchNorm") is normalizations.BatchNorm` | ✅ |
| `get(None) is nn.Identity` | ✅ |

### 2.4 形状与参数（`configs/Falcon.yml` 构建 `FalconSpeakerContrastive`）

```
[1,32000] → STFTEncoder (1,112,251,129) → audio_bottleneck (1,112,251,129)
          → Separator (1,112,251,129) → MaskGenerator (1,1,112,251,129) → STFTDecoder (1,1,32000)
参数：encoder 3,024 | audio_bottleneck 13,104 | separator 492,312 | mask_gen 12,657 | decoder 2,016
      audio 子模块合计 523,113 | video_bottleneck 0（Identity）
解码器帧账本：251 →(pad) 253 →(convT) 253 →(crop) 252 → istft(length=32000)
```

### 2.5 因果性（含反例）

| 探针 | 结果 |
|---|---|
| 扰动后半段音频，`STFTEncoder` 前 120 帧 | `0.000e+00`（严格因果） |
| `audio_bottleneck`（pre_norm=cLNhw）前 120 帧 | `4.7e-05`（非 0，全窗口统计泄漏） |
| `Separator` 前 120 帧 | `1.706`（量级与后段相同，强泄漏） |
| 最终波形 `[0:8000]` / `[8000:16000]` / `[16000:]` | `9.7e-04` / `1.65e-01` / `6.6e+00` |
| `STFTDecoder` 对帧 ≥120 扰动的响应 | `<15360` 采样严格 0，`[15360:16000)` 起 `6.3e-01`（iSTFT OLA 扩张） |
| `DualPathRNN(dim=4/dim=3)`、`MHSA2D(causal=True)` | 早期 t `0.000e+00`（时间因果） |
| `MHSA2D(causal=False)` | 早期 t `5.100`（非因果，作为对照） |
| `cLNhw` 单帧尖峰探针 | 早期 t `1.2e-03`（泄漏） |
| `torch.stft(center=True)` 脉冲 p=20000 | 受影响帧 `[156,157]`，符合 `[⌈(p−127)/128⌉, ⌊(p+128)/128⌋]` ⇒ 帧级 ≤8 ms lookahead |

### 2.6 导入链与依赖（`sys.modules`）

`import look2hear.models` 之后实际加载：`look2hear.layers.stft`、`look2hear.layers.stft_tfgn`、
`look2hear.utils.stft`、`look2hear.utils.complex_utils`、`look2hear.layers.enc_dec`、
`look2hear.layers.cnnlayers`、`look2hear.layers.rnnlayers`、第三方 `librosa`、`torch_complex`、`typeguard`。
⇒ 只依赖 `normalizations/activations` 的 Falcon 模型，实际被迫加载整个 layers 库与 librosa 栈。

### 2.7 其它确认

- `DualPathRNN` 默认 `rnn_type="Comba"` → `AttributeError: module 'torch.nn' has no attribute 'Comba'`（默认值不可用）；
  config 显式 `LSTM` 才能构造。
- `out_proj`：dim=4 `Linear(48→64, bias=False)`；dim=3 `Linear(72→64, bias=False)`。
- `Separator.__init__` 会 `pop` 传入字典的 `repeats`/`shared`（`Falcon.py:991-997`）⇒ `studio/falcon_runtime.py:149-150`
  用 `copy.deepcopy` 自我保护。
- `model.encoder.__module__ == "look2hear.models.Falcon"` ⇒ Falcon 不用 `layers/enc_dec.py`。
- Studio 侧 `falcon_model/layers/` 只有 `__init__.py`（docstring）、`normalizations.py`、`activations.py`；
  `normalizations.py` 与训练侧 diff 26 行（全为中文注释）、`activations.py` 0 行 ⇒ 无 librosa 依赖。

## 3. 失败与修正

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | `forward_stft` 对拍 `251 vs 129` 形状错误 | 脚本把 ref 多转置一次 | 修正比较方向 |
| 2 | `models.DualPathRNN` AttributeError | `models/__init__.py` 只导出 Falcon/FalconSpeakerContrastive | 改从 `models.Falcon` 导入 |
| 3 | cLNhw 泄漏只显示 `7e-07` | 替换同分布随机数时 GroupNorm 尺度不变性的巧合 | 改"放大 10×"与"单帧尖峰"两种强探针 |
| 4 | Separator 配置 pop 探针无效 | 复用了已被前面模型构造污染的配置字典 | 重新 `yaml.safe_load` |

## 4. 未执行（诚实声明）

- 未加载真实权重（仓库无 `falcon.pt`，`pretrain_zoo/` 不存在）⇒ 只验证结构与数值语义，不验证"训练好的模型行为"。
- 未在 Apple MPS 上实测 `CPU FFT bridge`（沙箱无 Apple GPU）⇒ 仅静态 diff 证据。
- 未运行训练/评测入口（缺 pypesq 等依赖）⇒ `layers` 死代码只做静态分析。
