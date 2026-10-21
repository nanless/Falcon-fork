# Lead 独立验证记录（R1 期间，供全员复核）

> **运行环境更新（2026-10-21）**：本文件与其它历史记录中的 `/root/.venvs/falcon/bin/python` 是 conda 环境建立前的
> 临时环境，已废弃。复跑这些脚本请使用项目专属 conda 环境：`/root/miniforge3/envs/falcon/bin/python`
> （见仓库根 `AGENTS.md` 与 `docs/understanding/VERIFY-ENV.md`）。历史日志保留原样以保持证据链。


> 目的：Lead 不只看队友的结论，独立用可执行验证给最关键的数字/声明“兜底”。所有结论可复跑。
> 环境与 stub 见 `docs/understanding/VERIFY-ENV.md`；脚本在 `docs/verification/scripts/`，日志在 `docs/verification/runs/`。
>
> 复跑命令（仓库根）：
> ```bash
> PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_model_params.py
> PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_causality.py
> ```

## V1. 模型构建 / 参数量（`runs/lead-model-params.log`，exit 0）

从 `configs/Falcon.yml`（`audionet_name: FalconSpeakerContrastive`）构建：

| 子模块 | 参数量 |
|---|---|
| encoder (STFTEncoder) | 3,024 |
| audio_bottleneck (ConvNormAct) | 13,104 |
| video_bottleneck (ConvNormAct, kernel=-1) | **0** |
| separator (AudioBlock+VideoBlock+GCI/TFRB) | 492,312 |
| mask_generator (MaskGenerator) | 12,657 |
| decoder (STFTDecoder) | 2,016 |
| spk_pool_proj (AttentionSpeakerPool) | 14,584 |
| **音频网络合计（全部可训练）** | **537,697** |
| 冻结视觉编码器 FRCNNVideoModel | 11,186,688 |
| **合计** | **11,724,385 ≈ README 的 11.72 M** |

结论：README 的“0.5 M trainable（论文口径）/ 11.72 M 总计”与代码一致 [验证]。
注意：537,697 是“模型全部参数”，其中并无需要冻结的项；FRCNN 在加载 pretrain 时由
`update_frcnn_parameter` 置 `requires_grad=False`（`look2hear/videomodels/frcnn_videomodel.py:112-124`）。

## V2. 形状契约（同上日志）

- 输入 `audio [B, L]`（16 kHz）、`mouth_emb [B, 512, Tv]`；训练模式返回 `(est[B,1,L], sep_emb[B,1,112,T',129])`，
  eval 模式只返回 `est[B,1,L]`（`Falcon_Contrastive.py:94-118`）。
- `T' = 1 + L/hop = 1 + L/128`：L=16000→126，32000→251，48000→376 [验证]。
- `F = win/2 + 1 = 129`（win=256，实 STFT）。
- `sep_emb` 通道 = encoder out_chan = 112。
- 前向 `est` 长度与输入严格相等（decoder `torch.istft(..., length=length)`）。

## V3. FRCNNVideoModel（同上日志）

- 输入 `[B,1,T,96,96]` → 输出 `[B,512,T]`：**时间维不降采样**（frontend3D 时间 stride=1、
  MaxPool3d 时间 kernel=1），所以 25 fps × 2 s = 50 帧输入 → 50 时间步输出，每帧 512 维。
- 参数 11,186,688；`pretrain` 加载走 `init_from → update_frcnn_parameter`（过滤 `tcn` key 并冻结全部参数）。
- 96×96 输入是“嘴部裁剪 + 灰度”的约定；README 说的“center-crop 88×88”在模型里**没有**出现
  （FRCNN 前端 Conv3d kernel (5,7,7)），应在 `falcon_runtime.preprocess_mouth` / 数据集中，见 06/05 文档核实。

## V4. 因果性探测（`runs/lead-causality.log`，exit 0）—— 重要 nuance

方法：把输入第 20000 个采样点 +1.0，观察输出从哪个采样点开始变化（严格因果应 ≥20000）。

| stage | 首变位置 | 说明 |
|---|---|---|
| encoder | 帧 156–159（≈19968–20352 采样） | 仅 `torch.stft(center=True)` 的 ±128 采样（±8 ms）分析窗前瞻，属正常 |
| audio_bottleneck | 帧 156 起，泄漏到窗口末尾 | 因果卷积（causal padding=2）只向后传播，符合预期 |
| **separator** | **帧 0 起（全窗口）** | **泄漏源**：内部多处 `cLNhw = LayerNormalizationHW`，其 `GlobalLayerNorm = nn.GroupNorm(1,C)` 在 `[B,C,T,F]` 上对 (C,T,F) 做**全窗口统计** → 任一点变化影响整窗 |
| mask_generator | 帧 0 起 | 继承了 separator 的全窗口依赖 |
| decoder | — | 因果 ConvTranspose + `istft` |

最终波形：改动 s=20000 时，输出从 **9612** 采样点就开始有 >1e-6 的差异（约 -649 ms 的“反向前瞻”），
越靠近改动点差异越大（改动点处 ~7.8e-2），[验证]。

**结论（建议全组统一表述）**：Falcon 的“causal”指**结构上无未来卷积/未来 RNN 步/未来 attention**
（causal padding、单向时间轴 LSTM、masked attention），且单次前向的因果卷积链严格单向；
但 **cLNhw 的全局归一化统计使模型在窗口级并非严格逐采样因果**——整窗统计会把未来信息带进过去。
这与 Studio 的做法自洽：整窗 2 s 重算 + 保留 80 ms 右上下文，而不是维护流式状态（README 也明确说
“windowed inference, not a native streaming model”）。论文的 latency/因果性论述应以“窗口级因果”理解。

> 相关证据：`look2hear/layers/normalizations.py:204-221`（LayerNormalizationHW）、
> `:134-143`（GlobalLayerNorm=GroupNorm(1)）、`:165-202`（CumulateLN4D 沿 H 累积）；
> `look2hear/models/Falcon.py:1080-1093`（encoder 因果 ConvNormAct）、`:1123-1192`（decoder 因果）。
> `separator` 中含 `LayerNormalizationHW` 的模块见日志尾部（projection/downsample_layers/concat_layers 的
> global_gate/local_embedding/global_embedding 等）。

## V5. 死代码 / 不一致（同上日志）

- `look2hear/models/Falcon.py:1480` 调用 `self._mdcl_loss(...)`，但类中只有 `_mdcl_loss_hard`
  （`:1410`）→ 若以训练模式、且 `near_audio is not None` 调用基类 `Falcon.forward`，必 `AttributeError` [验证]。
  实际 config 用 `FalconSpeakerContrastive`（覆写 forward，`:94`），故 paper 配置不触发；MDCL 工作流用
  `av_litmodule_mdcl.py:123` 直接调 `_mdcl_loss_hard`。
- `look2hear/models/__init__.py` 的 `__all__ = ["AVTIGER_causal"]` 与实际导出的 `Falcon`/`FalconSpeakerContrastive`
  不符（`get()` 不受影响，因为 `get()` 遍历 globals）。
- `mdcl_window` 用 `register_buffer(..., persistent=False)`（`Falcon.py:1339-1343`）→ 不写入 state_dict，
  与“权重文件只含 audio_model.* / video_model.*”的说明一致，但注意 `named_buffers()` 仍会列出它。

## V6. 待队友补充的验证（Lead 未做，交给 owner）

1. Studio `preprocess_mouth` 的 96→88 裁剪/归一化值域与训练数据的一致性（06/05）。
2. `falcon_model` vs `look2hear` 的逐文件 diff（06 + model-core R4 复核）。
3. `audio_visual_test.py` 的 best-ckpt 选择与 CSV 写入（04，可用假目录验证）。
4. 前后端 WS 字段/常量对照（07 ↔ 06）。
5. `audio_train.py` 的 exp_dir/callbacks 装配与 `configs/Falcon.yml` 的 `spk_contrast` 是否真的被传进 litmodule（03）。
6. 纯 JS 测试全部跑通（07/08）；`pytest studio/tests/test_device.py` 是否可在本环境跑（06/08）。
