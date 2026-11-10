# Phase 2 日志复核摘要（2026-11-10）

范围：`docs/verification/phase2` 递归 41 个 `.log`，共 2,986 行、275,571 字节。逐文件机器可读状态见 `status.json`（路径、SHA-256、字节和行数、条件、输出、异常和证据边界）。本次复核历史日志的正文与运行结果；两份转换日志的循环进度按重复记录处理。

- `P2B-big-full.log` 仅输出至 100/3000，没有结果或结束状态；已完成的 big/small 对比为 300 条。
- `P2A-small-200.log` 中多数变体 batch=4，而 audio_pad_3s 为 batch=2；已发现模型跨 batch 样本依赖，因此这组差异不只是时长或 padding 的干净控制实验。影响量未针对原始 200 条复测。
- `P2C-convert.log` 首次存在性检查 TypeError，之后更正为 28,000 条路径存在，200/每 split 抽样检查通过；逐样本目标与画面配对需要另外核查。
- RealSSA 两种 9,169 条转换均记录全部 near 为 fallback；日志称 K=0 时 NASS 整体 inactive 不准确，负项为零但正项仍可能生效。两次全量 Sim 评测各记录 6 次 SDR 计算失败和 3 次 PESQ 无语音，数值记 NaN，各指标有效数不同，日志未列有效数。早期 ch0 日志为 SIGKILL，不是完整结果。
- RealSSA-Real full/full2 分别 54/54 因 VIDEO_FPS/torch NameError 失败；full3 虽完成，但后续发现窗口起点问题被 full4 取代。full4 是本协议 54 条结果（OVRL 2.7941、P808 3.3365），论文对应指标的打分细节尚待核对。
- `P2D-smoke-bs8.log` 记录在 GPU 1 被其他进程占用时 OOM，独占 GPU 的容量及其他多卡条件待测；另有双卡和单卡冒烟成功记录。正式训练 0–99 与后续 100–129 记录均有结果，ep124 与 ep86 相隔 38 个 epoch，观察到的差别包含训练阶段变化。
- `P2F-protocol-check.log` 从配置和 checkpoint epoch 直接断言发布模型训练来源，证据不足；公开权重训练语料仍未确认。历史 `QA-regression.log` 记录当时的 111 passed/1 skipped。
