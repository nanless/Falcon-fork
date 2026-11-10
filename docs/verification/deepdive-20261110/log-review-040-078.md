# 历史日志逐项审阅：第 40–78 份

范围：以 2026-11-10 扫描 `docs/verification` 下的 `*.log`，按 Python `Path` 字典序排列，共 117 份；本页为 1 起第 40–78 份，共 39 份。逐份检查配置、终态、数值与异常；长日志先按关键行及首尾阅读，后续逐份补读见[补读台账](data-audit-20261110/log-complete-review-040-078.tsv)。表中的完成状态、路径、软件版本和代码状态均来自当时日志；无退出码的记录写明最后观察到的内容。

| 序号 | 日志（相对 `docs/verification/`） | 记录与结论 |
| --- | --- | --- |
| 40 | [phase2/P2D-eval-falcon-full-cols-20261031.log](../phase2/P2D-eval-falcon-full-cols-20261031.log) | 官方 small 权重、LRS2 全量 3000/3000；SI-SNRi 7.029101、SDRi 7.712925，八列各 3000 有效；日志有 `exit=0`。 |
| 41 | [phase2/P2D-eval-falcon-full.log](../phase2/P2D-eval-falcon-full.log) | 同类全量 3000 条，SI-SNRi 7.029101、SDRi 7.712925，结果已合并；无显式退出码。 |
| 42 | [phase2/P2D-eval-falcon.pt.log](../phase2/P2D-eval-falcon.pt.log) | small 权重，限 300 条，SI-SNRi 6.916240、SDRi 7.491614；结果已合并，无显式退出码，不是全量结论。 |
| 43 | [phase2/P2D-near-train.log](../phase2/P2D-near-train.log) | 从上一轮 last.ckpt 恢复训练，跑至 max_epochs=130，epoch 124 最佳验证损失 -13.39090；记录 exit=0。此日志本身不能隔离损失项对提升的因果作用。 |
| 44 | [phase2/P2D-smoke-1gpu.log](../phase2/P2D-smoke-1gpu.log) | 单卡小样本冒烟，1 epoch/100 step，val loss -2.71445；exit=0，不代表正式模型质量。 |
| 45 | [phase2/P2D-smoke-bs8.log](../phase2/P2D-smoke-bs8.log) | 双卡、bs8 冒烟，rank 1 CUDA OOM；当时 GPU1 被多个进程占用，失败不能单独推导空闲设备上的容量极限。 |
| 46 | [phase2/P2D-smoke.log](../phase2/P2D-smoke.log) | 两卡 1 epoch/100 step 冒烟，val loss -1.94570，exit=0；不能与其他不同资源条件下的 OOM 混同。 |
| 47 | [phase2/P2D-train.log](../phase2/P2D-train.log) | 训练达到 max_epochs=100，epoch 99/step 250000，最佳验证损失 -13.20141；末尾 Finished Training 并保存模型，无显式 shell exit。 |
| 48 | [phase2/P2E-dnsmos-synthetic.log](../phase2/P2E-dnsmos-synthetic.log) | 4 个合成信号验证 speechmos 0.0.1.1 / onnxruntime 1.31.0 离线链路，标记 PASS；不是任何真实数据集的得分。 |
| 49 | [phase2/P2E-realssa-real-full.log](../phase2/P2E-realssa-real-full.log) | 54 条均报 `VIDEO_FPS` 未定义，虽然写出 CSV/summary 路径，但 exit=1，无有效整集结论。 |
| 50 | [phase2/P2E-realssa-real-full2.log](../phase2/P2E-realssa-real-full2.log) | 54 条均报 `torch` 未定义，虽然写出 CSV/summary 路径，但 exit=1。 |
| 51 | [phase2/P2E-realssa-real-full3.log](../phase2/P2E-realssa-real-full3.log) | 日志称 54 段完成、OVRL 2.7901、exit=0；后续审查发现窗口起点错误，**该组数值已作废**，参见现行状态，不因退出成功恢复有效性。 |
| 52 | [phase2/P2E-realssa-sim-ch0-convert.log](../phase2/P2E-realssa-sim-ch0-convert.log) | 19 个 tt shard、ch0，转换 9169 条，incomplete/errors 均为 0，near 为混音占位 9169 条；日志到 done，无显式退出码。日志说 K=0 时整个对比损失 inactive 有误，见下。 |
| 53 | [phase2/P2E-realssa-sim-ch0-eval-20261030.log](../phase2/P2E-realssa-sim-ch0-eval-20261030.log) | ch0、9169 条；SI-SNRi 7.976709、SDRi 8.452396、PESQ 2.004064。存在 3 条 SDR 奇异矩阵和 3 条 PESQ 无语音异常；部分 SDR 对输出与基线各出现一次报错。结果已合并，无显式退出码。 |
| 54 | [phase2/P2E-realssa-sim-ch0-eval.log](../phase2/P2E-realssa-sim-ch0-eval.log) | 历史记录明示被 SIGKILL 延期、逐句 CSV 为零字节并删除；不是成功评测；后续第 53 项是另一轮评测。 |
| 55 | [phase2/P2E-realssa-sim-convert.log](../phase2/P2E-realssa-sim-convert.log) | 19 个 tt shard、mean 通道，转换 9169 条，incomplete/errors 为 0，9169 near 占位；exit=0；同样需更正 K=0 的损失解释。 |
| 56 | [phase2/P2E-realssa-sim-eval-official.log](../phase2/P2E-realssa-sim-eval-official.log) | mean 通道 9169 条，SI-SNRi 7.774307、SDRi 8.479395、PESQ 2.093770；个别 SDR/PESQ 报错记 NaN；exit=0。 |
| 57 | [phase2/P2E-sim-pipeline.log](../phase2/P2E-sim-pipeline.log) | 两次 dry-run；当时一次 gated 检查为 403，即使命令显示 rc=0 也未下载、转换或评测。历史 `--proxy` 命令不能直接当成当前接口用法。 |
| 58 | [phase2/P2E-smoke-loader-dryrun.log](../phase2/P2E-smoke-loader-dryrun.log) | mock 目录 fallback loader 发现 2 条；明确没有加载模型、计算 DNSMOS 或写 CSV。 |
| 59 | [phase2/P2E-smoke-official-loader.log](../phase2/P2E-smoke-official-loader.log) | mock 格式 official loader 1 条、2 秒 GPU 推理，N=1 OVRL 2.3637；仅接口冒烟。 |
| 60 | [phase2/P2E-smoke-realssa-real.log](../phase2/P2E-smoke-realssa-real.log) | LRS2 风格模拟 RealSSA 2 条、2 秒，N=2 OVRL 2.246；日志明确不是 RealSSA 论文结果。 |
| 61 | [phase2/P2F-protocol-check.log](../phase2/P2F-protocol-check.log) | LRS2 集合 3000、目标/混音 5 条重建抽查、N=50 缓存对拍；同步探针自评 inconclusive。末段由训练配置推断 falcon.pt 属 LRS3 并写为确定事实，配置与权重的身份链未建立，不能单凭该段确定训练谱系。 |
| 62 | [phase2/QA-regression.log](../phase2/QA-regression.log) | 2026-10-21 分阶段 QA，早期有失败，v1 当时 111 passed/1 skipped、Studio 35 passed/2 skipped；当时训练仍在进行，不能替代之后的运行结果。 |
| 63 | [phase2/logs/P2B-big-b300.log](../phase2/logs/P2B-big-b300.log) | big，300 条真实视频；whole SI-SNRi 7.887，windowed 5.569，均有 RESULT；该特定窗口协议不能泛化为全部流式模式。 |
| 64 | [phase2/logs/P2B-big-full.log](../phase2/logs/P2B-big-full.log) | big 全量目标 3000，日志停在 100/3000，没有结果或完成标志；不可当成全量完成。 |
| 65 | [phase2/logs/P2B-big-visual-b300.log](../phase2/logs/P2B-big-visual-b300.log) | big、300 条零视觉，whole SI-SNRi 3.512，已输出 RESULT；是特定消融而非正常视频表现。 |
| 66 | [phase2/logs/P2B-small-b300.log](../phase2/logs/P2B-small-b300.log) | small、300 条真实视频，whole SI-SNRi 6.916、windowed 5.004，均输出 RESULT；记录音频 537697、视觉 11186688 参数。 |
| 67 | [phase2/logs/P2B-small-visual-b300.log](../phase2/logs/P2B-small-visual-b300.log) | small、300 条：零视频 SI-SNRi -8.173、打乱视频 -5.977；是该样本集和设置下的消融。 |
| 68 | [runs/04-eval-flow.log](../runs/04-eval-flow.log) | 旧版评测路径探针，DONE；记录当时模型配置 pop、副作用、单声道压成标量等问题，多项随后修复。不能把旧失败直接认定为当前缺陷。 |
| 69 | [runs/04-losses.log](../runs/04-losses.log) | 合成输入公式与损失探针，DONE；K=0 仍计算正样本项。历史 SISNRi 构造错误与旧指标口径须按当前代码状态判断。 |
| 70 | [runs/04-macs.log](../runs/04-macs.log) | 历史实测音频 537697、视频 11186688 参数；音频模型 2 秒 10.611 G MAC，视频前端 64/88/96 像素分别 7.77/15.90/17.48 G；DONE。10.6 G 不含视频编码器。 |
| 71 | [runs/04-metrics.log](../runs/04-metrics.log) | 旧版 tracker 的 SDR 列符号/定义及 PESQ 参数顺序问题，DONE；作为旧实现证据，不能直接套用随后修复的现行评测。 |
| 72 | [runs/04-video.log](../runs/04-video.log) | 视频前端不同帧数和尺寸的合成输入探针，输出时间步保持，DONE；记录当时 eval 链式返回和缺失预训练文件异常，不应假定当前仍存在。 |
| 73 | [runs/conda-env-verify.log](../runs/conda-env-verify.log) | 2026-10-21 环境快照：Python 3.11.17、torch 2.8.0+cu128、A800；Studio 当时 35 passed/2 skipped、Node 8 通过；不代表现环境或最新测试。 |
| 74 | [runs/data_pipe/verify_dataset.log](../runs/data_pipe/verify_dataset.log) | 人造数据集预处理/索引/边界探针，TOTAL FAILURES 0；包括当时单声道、短视频、8 kHz 对齐等异常被测试捕获。PASS 表示探针符合预期，并非这些异常没有发生。 |
| 75 | [runs/data_pipe/verify_p0_eval.log](../runs/data_pipe/verify_p0_eval.log) | 旧版单声道评测会压成标量并在 STFT 抛异常；立体声及加入维度保护的对照通过；DONE。该记录不是当前代码仍有该缺陷的证明。 |
| 76 | [runs/data_pipe/verify_preprocess.log](../runs/data_pipe/verify_preprocess.log) | 人造数据预处理 19 项探针，TOTAL FAILURES 0；包括空 split 崩溃、同数量不同 ID 时错配和未使用采样率参数等当时行为，PASS 代表捕获到预期现象。 |
| 77 | [runs/data_pipe/verify_realweight.log](../runs/data_pipe/verify_realweight.log) | small checkpoint epoch 714、525 音频与 137 视频键严格加载；2 秒及 48513 样本的测试各输出有限且等长，DONE；不证明所有长度或质量。 |
| 78 | [runs/layers-dsp-verify.log](../runs/layers-dsp-verify.log) | DSP、归一化及因果性历史探针，DONE；当时使用 timm/typeguard stub，参数 523113 不是实际环境完整模型口径；扰动揭示跨窗口统计和旧配置字典副作用。 |

## 与当前说明的交叉核对和边界

- [当前状态](../../STATUS.md) 与[数据和训练说明](../../understanding/20261110-data-and-training.md)将 mean 与 ch0 的结果、样本量以及 K=0 的正项保留分开；它们不能由旧日志的“loss stays inactive”反向覆盖。日志 52 和 55 所记 `K=0` 是负样本数，不意味着正样本相似度项必然为零。
- [验证目录说明](../README.md) 已标注 full3 窗口起点错误，因此第 51 项仅保留历史证据。第 49–50 项均退出失败，不能把已有输出路径当成功证据。
- [模型说明](../../understanding/20261110-model-and-streaming.md)解释全窗口归一化、真实参数与具体输入范围；旧版日志的 stub、未修复问题和当前代码不可混为一谈。
- 第 61 项权重训练谱系结论缺少独立身份链，现行状态按“未确认”处理。日志中显示某配置或 checkpoint 元数据，不足以证明两者源自同一次训练。
- 本页汇总历史日志；退出码、输入哈希和来源按原记录列出。训练、评测及逐项数据重算的结果见对应实验记录。
