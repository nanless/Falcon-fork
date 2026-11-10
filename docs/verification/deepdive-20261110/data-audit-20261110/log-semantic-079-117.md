# 历史日志第 79–117 份语义复核

范围以本次检查前 `docs/verification/` 117 个 `.log` 的路径排序为准；每行回指原件并给出 SHA-256 的前 12 位。对主要正文、运行终态和关键条件做语义阅读；第 86 份一行 33,312 字符的参数 key 列表未逐名审阅。日志记录对应各次运行的代码和环境。

| 序号与日志 | SHA-256 前 12 位 | 终态、内容与边界 |
|---|---|---|
| 79 [`runs/lead-causality.log`](../../runs/lead-causality.log) | `5ac84391583c` | 单点扰动首变输出 9612，separator 自第 0 帧受影响；后半大量为归一化模块清单，末尾无 DONE；不代表端到端时延。 |
| 80 [`runs/lead-model-params.log`](../../runs/lead-model-params.log) | `3c90e110d69e` | 音频 537697、视觉 11186688；旧裸 Falcon 缺 _mdcl_loss 是当时结果，不能当现今未修复。 |
| 81 [`runs/lead-real-weights.log`](../../runs/lead-real-weights.log) | `34c6a3ee41e3` | small/big strict 加载，small 训练侧与 Studio 单个 CPU 输入相同；714/715 仅 epoch 元数据，不能推出训练语料。 |
| 82 [`runs/model-core-r3-verify.log`](../../runs/model-core-r3-verify.log) | `675d14e68876` | 权重 key/shape 与 strict 加载、concat[2] 未调用、分界位置 lookahead 56–58 帧；状态元素数不等于参数数。 |
| 83 [`runs/model-core-verify.log`](../../runs/model-core-verify.log) | `5452e064acc9` | 模型形状、模块调用和旧缺陷探针完成；旧裸 Falcon/导出问题不能直接套到当前代码；一次配置重用结论须区别当前修复。 |
| 84 [`runs/model-core-verify2.log`](../../runs/model-core-verify2.log) | `8759b04b1c24` | 全模型扰动输出后，调用 eval() 返回 None 导致视频探针 TypeError；脚本异常终止，未执行的探针不能当通过。 |
| 85 [`runs/model-core-verify3.log`](../../runs/model-core-verify3.log) | `f187710b0bb9` | 训练返回 tuple、旧 MDCL 的 K=1/K=3 与梯度小样本通过；这些不是当前 NASS 全部训练结果。 |
| 86 [`runs/model-core-verify4.log`](../../runs/model-core-verify4.log) | `1c3aeec73e3e` | 因果和视频探针后 strict load 参数不匹配异常终止；历史共享配置污染影响比较。末尾一行 33312 字符的逐键列表未逐名核验，已核查异常类型及前缀。 |
| 87 [`runs/r2-causality-conditions.log`](../../runs/r2-causality-conditions.log) | `da8a15c3b318` | 五种扰动及时频边界探针 DONE-R2C；GN 与后接累积归一化会部分抵消，不能将单次幅度推广。 |
| 88 [`runs/r2-review-03-pl-contract.log`](../../runs/r2-review-03-pl-contract.log) | `e6e114c5ca29` | CPU 小线性网络、两个验证 loader 的 Lightning 合约通过；CSVLogger experiment.log 和 GPU 数量错误是历史有意错误探针。 |
| 89 [`runs/r2-review-03.log`](../../runs/r2-review-03.log) | `50d534ac0f44` | 静态未定义名、旧 logger/依赖/10卡错误等检查 DONE；模型历史缺陷不能直接推定当前仍存在。 |
| 90 [`runs/r3-layers-revision.log`](../../runs/r3-layers-revision.log) | `e4cb31a6e6fa` | 层探针 DONE；57 帧是特定边界单遍几何结果；FRCNN 未来两帧，VideoBlock 结果为特定 batch。 |
| 91 [`runs/r3-macs.log`](../../runs/r3-macs.log) | `e291cea563d1` | 音频 10.611 GMAC、视觉 88² 15.90 GMAC；ptflops 覆盖有限，不能宣称是硬件完整 FLOP。 |
| 92 [`runs/r3-mono-eval-repro.log`](../../runs/r3-mono-eval-repro.log) | `7c253fc888c9` | 历史 mono 数据被降维后报错、ndim guard 后可运行，旧缺陷已有修复；对齐的样本数不等于声画物理同步。 |
| 93 [`runs/r3-studio-weights.log`](../../runs/r3-studio-weights.log) | `450e1b40878d` | 真实小模型训练/Studio 单例 CPU 输出相同；MPS FFT bridge 源码记录，不等于跨设备完全一致。 |
| 94 [`runs/r3-video.log`](../../runs/r3-video.log) | `73b3f14a4aa6` | FRCNN 输入尺寸及冻结探针完成；eval() 返回 None 是该类历史实现行为；替代骨干未全面验证。 |
| 95 [`runs/r3-weights.log`](../../runs/r3-weights.log) | `9adac3b56a35` | small/big checkpoint key 与 strict 加载通过；state_dict 元素数与模型参数数有别。 |
| 96 [`runs/r4-incremental-01-r3.log`](../../runs/r4-incremental-01-r3.log) | `67b5eb98ff62` | small/big 哈希和加载、concat[2] 未调用、small 单例两副本一致，big 是 runtime 对自身检查。 |
| 97 [`runs/r4-prep-studio-backend.log`](../../runs/r4-prep-studio-backend.log) | `64640f1a2896` | lead 参数脚本与因果脚本的摘录重跑；历史 _mdcl_loss 错误及模块清单不能直接当现在状态。 |
| 98 [`runs/r4-review-01-exec.log`](../../runs/r4-review-01-exec.log) | `90991824f883` | 参数/两遍调用/单例未来扰动与历史 n_src=2 失败；旧开关比较为当时配置，不代表现有开关所有情况。 |
| 99 [`runs/r4-review-08-d22.log`](../../runs/r4-review-08-d22.log) | `9efd1d3798e6` | 公开主机名 Host 测试 400，loopback 200；是历史 TestClient 条件下结果。 |
| 100 [`runs/r4-review-08-node.log`](../../runs/r4-review-08-node.log) | `bc735d1c249e` | 8 项 Node 检查 rc=0、21 项语法检查通过；不等于真实摄像头联调。 |
| 101 [`runs/r4-review-08-pytest-cache.log`](../../runs/r4-review-08-pytest-cache.log) | `7522523744cb` | 历史 pytest 35 passed/2 skipped；依赖已淘汰 venv 和缓存，不是当前全套测试数。 |
| 102 [`runs/r4-review-08-pytest-nocache.log`](../../runs/r4-review-08-pytest-nocache.log) | `5492106262dd` | 历史无缓存 pytest 33 passed/4 skipped；差异包含条件性跳过，不能解释为模型质量差异。 |
| 103 [`runs/r4-review-08-scripts.log`](../../runs/r4-review-08-scripts.log) | `559f7b5175eb` | 6 个 shell 脚本语法通过；两个历史 Python 环境检查失败，版本要求是旧快照；BlackHole 检查仅当时机器。 |
| 104 [`runs/review-08-by-studio-backend.log`](../../runs/review-08-by-studio-backend.log) | `e259cc381b18` | 8 Node 通过、带 stub pytest 33/4、公共 Host 400；头尾文档哈希一致仅证明被审稿没改。 |
| 105 [`runs/review-model-core-checks.log`](../../runs/review-model-core-checks.log) | `bcd62c51eb89` | 金字塔、调用和扰动探针 DONE；历史配置重用塌缩、VideoBlock B=1 探针需结合后来修复和批次串扰。 |
| 106 [`runs/review_layers/05-review-checks.log`](../../runs/review_layers/05-review-checks.log) | `060188da3dcb` | 权重严格加载与长短输入通过；val 管线 float64 与 Studio float32 有小差别；NASS K=0 正项仍在。 |
| 107 [`runs/studio-backend-verify.log`](../../runs/studio-backend-verify.log) | `ae81e12931c5` | 包含预设 simulated checkpoint failure 的错误路径；末尾 overall OK，主要验证 mock HTTP/WS、合成权重和输入协议，非真实浏览器会议。 |
| 108 [`runs/train-sys-config-mutation.log`](../../runs/train-sys-config-mutation.log) | `62e2f6777b57` | 记录构造后字典键发生变化，属于当时实验；当前调用是否复制配置应看现行实现。 |
| 109 [`runs/train-sys-ddp-bucket-hang.log`](../../runs/train-sys-ddp-bucket-hang.log) | `e0f09d01d19e` | 两个 CPU/Gloo rank 不同 bucket 后日志停止，无退出码或最终诊断；有风险信号，不证明当前真实 Falcon 多 GPU 必然死锁。 |
| 110 [`runs/train-sys-litmodules.log`](../../runs/train-sys-litmodules.log) | `e7e68522bc65` | 受限 batch 的 NASSv2 11 epoch 和 MDCL 3 epoch 完成；旧 CSVLogger 拟合 AttributeError 为历史版本；PIT tuple 失败是边界探针。 |
| 111 [`runs/train-sys-parser-roundtrip.log`](../../runs/train-sys-parser-roundtrip.log) | `d3b046c7beac` | 旧配置解析结果是 10 GPU 和 anneal 200 的快照，不是论文实际训练条件或当前机器配置。 |
| 112 [`runs/train-sys-r2-crosscheck.log`](../../runs/train-sys-r2-crosscheck.log) | `d9b20c2f911a` | FRCNN 形状与 NumPy dtype 双探针通过；文档哈希只标识当时输入。 |
| 113 [`runs/train-sys-r3-exec.log`](../../runs/train-sys-r3-exec.log) | `03218818d551` | 旧训练配置输出可丢失 repeats/shared，模拟重建变 49081 参数；日志 strict=False 的 dropped elems=0 与 unexpected 505 同时出现，不可将该统计当证明权重完整加载。 |
| 114 [`runs/train-sys-r4-causality-baseline.log`](../../runs/train-sys-r4-causality-baseline.log) | `5ac84391583c` | 与 lead-causality 相同的点扰动和模块清单，无 DONE；证据是限定条件的数值，不是完整端到端延迟。 |
| 115 [`runs/train-sys-r4-crosscheck.log`](../../runs/train-sys-r4-crosscheck.log) | `913dc22886a4` | encoder/decoder 两个采样位置、重复 CumulateLN 定义与参数核对；一位置出现早 31 采样，不能写成完整链严格因果。 |
| 116 [`runs/train-sys-r4-layers-causal.log`](../../runs/train-sys-r4-layers-causal.log) | `62564387de72` | 层因果、STFT 数值和帧数探针 DONE；用阈值把 4.768e-7 标为因果只说明该阈值下未检出，不证明数学上严格因果。 |
| 117 [`runs/train-sys-undefined-globals.log`](../../runs/train-sys-undefined-globals.log) | `af7462a20571` | AST 给出旧 comet_logger 等未定义名，须核对现行代码后判断是否仍可达。 |

其中第 84、86 份日志明确以异常终止；第 109 份停在两 rank 不同 bucket 的汇合之前，没有保存退出码。第 107 份的 checkpoint 失败为脚本有意模拟的分支。当前代码和真实 GPU 环境的运行情况需结合对应测试记录。
