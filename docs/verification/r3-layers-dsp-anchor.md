# 版本锚点：`docs/understanding/02-layers-dsp.md`（R3 交付版）

> 用途：被评审文档的 R3 版行数与 sha256 外置记录（文件自哈希无法内嵌于自身）。
> 本文件只读锚点；若文档再修订（R4/R5），请更新本文件的"当前版本"。

## 1. 版本记录

| 版本 | 行数 | sha256 | 说明 |
|---|---|---|---|
| R1 | 757 | `ae01042f95cef3e95961027b308b5e610a60f2b97b17dce0a7fad279a31bbd2f` | 被 `model-core`（R2）与 `train-sys`（R4）评审锚定 |
| **R3（交付，最终）** | **895** | `1b6def62519bdf19ffa1573f04b1180944f7ba27ccf9f7d827dd0d078492afdd` | 本文件记录；对应 Lead R3 指令 + 两份评审响应（02 文档 §9.1）。最终版另含与 01-R3 §4.6-⑨ 五协议表的口径对齐说明 |

## 2. 校验命令

```bash
cd /root/code/github_repos/Falcon-fork
wc -l docs/understanding/02-layers-dsp.md
sha256sum docs/understanding/02-layers-dsp.md
# 期望：895 行；1b6def62519bdf19ffa1573f04b1180944f7ba27ccf9f7d827dd0d078492afdd
```

## 3. R3 验证脚本与日志（本版新增）

| 脚本 | 行数 | 日志 | 行数 |
|---|---|---|---|
| `docs/verification/scripts/verify_r3_layers_revision.py` | 216 | `docs/verification/runs/r3-layers-revision.log` | 92 |
| `docs/verification/scripts/verify_r3_studio_weights.py` | 87 | `docs/verification/runs/r3-studio-weights.log` | 40 |

权重：`/root/.cache/falcon_weights/falcon.pt`（epoch 714，只读；`state_dict` 662 key = audio 525 + video 137）。

## 4. R3 关键修订摘要（详见文档 §9.1/§9.3）

1. **P0**：`STFTEncoder` 由"严格因果"改为 **conv 子层严格因果 + `torch.stft(center=True)` 的 ≤128 采样/8 ms
   帧级 lookahead**（A36′；§8-[V1]：`s≥16000 → 首变帧 125`）。
2. **P1**：§4.1 采样级 lookahead 由"≤256 采样/16 ms"改为 **encoder→decoder 串联 ≤127 采样**
   （解析 `= p mod 128`；实测 `p=20000 → 19969`，泄漏 31；§8-[V2]/A59）。
3. **P1**：A19 增"扰动协议表 + 两抵消机理"（CumulateLN4D 抵消纯尺度泄漏：GN-only ×10 `3.93` vs cLNhw ×10
   `1.44e-03`；common-mode：整段全通道 +5 → `7.15e-07` vs 整段单通道 +5 → `5.084`；A60）。
4. **P1**：A49 改逐级 **L0 ±1 / L1 ±4 / L2 ±8 / L3 ±16** + 单遍 **57 帧 = 7,296 采样 ≈ 456 ms**；
   剔除 `:823`（k=1 `residual_conv`）作为非因果证据。
5. **P1**：A57 证据行号 `66-68` → **`:84-85`**（`causal=False` 分支）。
6. **P2**：§2.1 计数 15/6 → **13/5**；§4.7 PL 归因改为 `utils/__init__.py:12 → lightning_utils.py:3` 顶层加载
   （保留"会拉入 PL"结论，不采纳 R2 的否定判断）；A52 encoder 桥行号 → `:1111-1122`；A03/A05 限定产品代码；
   nearest 插值改为"单调、因果"；补 VideoBlock(train) BN 非因果与 FRCNN ±2 帧=±80 ms（§8-[V4b][V4c]）。
7. **真实权重验证**：Studio 副本与训练副本 `strict=True` 均加载成功（525 key、537,697 参数），
   CPU 输出 **`max|Δ|=0.0`**（bit-identical）；Studio 副本 import 不加载 librosa/torch_complex（A63/§[W1][W2]）。
8. **统一表述**：采用与 `01-model-core.md` R3（975 行，sha `afe00188…db0058`）一致的
   **"窗口级因果（window-causal）"**，并与 01-R3 §4.6-⑨ 五协议表口径对齐（cLNhw 数字全部保留、标注协议）。
