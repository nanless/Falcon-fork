# 论文结果表：旧版 README 的作者结果

以下两张表转录自修订前的英文 README（基线 `6e5c1b5`），数值来自论文。比较对象、设备和统计范围见表题；项目评测结果见[复现状态](STATUS.md)，原文见 [Falcon 论文](https://arxiv.org/abs/2610.32749)。

Causal AVTSE methods (SI-SNRi / SDRi in dB; DNSMOS on the real-recorded split). **Bold** marks the best result per column. See [the current reproduction status](STATUS.md#已有评测结果) for this repository’s evaluation results.

| Method | RealSSA SI-SNRi | SDRi | LRS2-2Mix SI-SNRi | SDRi | LRS3-2Mix SI-SNRi | SDRi | VoxCeleb2-2Mix SI-SNRi | SDRi | Real-world DNSMOS |
|---|---|---|---|---|---|---|---|---|---|
| AV-ConvTasNet | 1.2 | 2.6 | 9.8 | 10.3 | 12.5 | 12.8 | 6.9 | 8.2 | 2.580 |
| AV-Sepformer | 5.5 | 6.3 | 10.6 | 11.1 | 13.4 | 13.7 | 6.9 | 8.3 | 2.813 |
| AV-TF-GridNet | 7.6 | 8.1 | 12.7 | 13.1 | 14.3 | 14.8 | 5.8 | 7.7 | 3.030 |
| RTFSNet-12 | 4.9 | 5.7 | 12.4 | 12.8 | 14.0 | 14.4 | 8.4 | 10.7 | 3.066 |
| CTCNet | 5.4 | 6.2 | 8.0 | 8.8 | 9.4 | 10.2 | 2.5 | 4.7 | 2.840 |
| AV-DPRNN | 5.4 | 6.2 | 10.2 | 10.7 | 11.9 | 12.3 | 7.8 | 9.0 | 2.901 |
| AV-Mossformer2 | 6.0 | 6.7 | 13.2 | 13.9 | 16.2 | 16.6 | 7.7 | 9.1 | 2.912 |
| Swift-Net-12 | 7.3 | 7.8 | 13.9 | 14.1 | 16.0 | 16.3 | 9.4 | 10.8 | **3.276** |
| Dolphin | 6.7 | 7.3 | 14.0 | 14.6 | 14.3 | 14.9 | 9.2 | 11.1 | 2.960 |
| **Falcon (ours)** | **7.7** | **8.2** | **14.1** | **14.7** | **16.4** | **16.7** | **10.5** | **11.8** | **3.276** |

Efficiency on a 2 s, 16 kHz clip with 50 mouth frames (RTX 3090 / Intel Xeon, batch size 1; trainable parameters only):

| Method | Params (M) | MACs (G) | CPU latency (ms) | GPU latency (ms) | RAM (MB) | GPU memory (MB) |
|---|---|---|---|---|---|---|
| AV-ConvTasNet | 11.0 | 15.0 | 674.4 | 13.3 | 100.0 | 241.8 |
| AV-Sepformer | 39.7 | 226.5 | 5143.0 | 53.9 | 178.0 | 369.7 |
| AV-TF-GridNet | 6.2 | 213.8 | 5646.3 | 567.8 | 136.3 | 1015.8 |
| RTFSNet-12 | 0.6 | 67.0 | 4149.3 | 164.7 | 457.9 | 411.7 |
| CTCNet | 6.8 | 150.1 | 3267.4 | 162.8 | 233.2 | 216.5 |
| AV-DPRNN | 1.3 | **4.5** | **271.0** | **5.8** | **73.0** | **192.4** |
| AV-Mossformer2 | 56.7 | 224.2 | 16670.7 | 114.9 | 330.9 | 539.8 |
| Swift-Net-12 | **0.5** | 45.3 | 4163.8 | 172.1 | 482.3 | 411.7 |
| Dolphin | 5.6 | 18.1 | 2411.0 | 87.9 | 560.2 | 390.3 |
| **Falcon (ours)** | **0.5** | 10.6 | 1361.2 | 51.3 | 125.3 | 364.7 |

计算量工具的算子覆盖不一定完整。当前项目实测与其边界见[模型说明](understanding/20261110-model-and-streaming.md)和[算子探针](verification/deepdive-20261110/falcon_ptflops_ops_probe_20261110.txt)。
