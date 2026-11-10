# Phase1 / Phase2 测试逐项阅读记录（2026-11-10）

两文件均已逐体通读；本轮没有执行。各项“依赖/跳过”是代码静态行为，非运行结果。通用副作用：导入可能生成缓存，pytest 自身可写缓存；明确的源码 py_compile 即使 PYTHONDONTWRITEBYTECODE 也写 .pyc，应另设 PYTHONPYCACHEPREFIX。数值指纹共用 session fixture，读配置和官方权重，固定 CPU 64 线程后恢复；CI=1/true 跳过带 _CI_SKIP 的测试，其他后端或硬件字节一致性不保证。训练模式前向不进行优化训练。

## test_phase1_regression.py：21 项

- :304 environment_and_weights：非 CI；Python 3.11、torch 2.8.0、固定权重存在、字节数及固定 SHA 与 baseline 对照；不前向；不证明其他权重。
- :319 parameter_counts_match_baseline：非 CI；fixture 加载配置、权重并运行单例音视频前向；核 525/137 键和音视频参数总量；仅 small 及单输入。
- :330 eval_forward_fingerprint_unchanged：非 CI，无法设置 64 线程则 skip；单例 eval 输出形状、SHA、sum 对 baseline；不测 batch 或物理对齐。
- :342 train_forward_fingerprint_unchanged：同样跳过条件；单例 train 模式估计及分离 embedding 的形状与 SHA；无 backward。
- :353 video_forward_fingerprint_unchanged：同样跳过条件；视频训练模式默认 BN、单例输出形状/SHA；不等于 eval 行为。
- :422 numeric_invariance_vs_main_worktree：非 CI、64 线程失败或基线 worktree 缺失 skip；tmp_path 写子进程脚本，在 /root/wt-base 或 /root/wt-main 重跑并比基线和四种 SHA/参数量；子进程可能写该 worktree 缓存；候选仅按文件存在选择，未用 git 验证其真实 HEAD。
- :469 import_completeness：仓库 cwd 子进程导入 12 个包与入口并断言成功标记；依赖完整环境，可能写导入缓存；不执行各入口。
- :482 studio_pytest_regression：仓库 cwd 嵌套运行 studio/tests，要求退出 0、passed≥35、skip 为 0 或 2、总数≥37；可能写仓库 pytest/导入缓存并执行 Studio 测试自身副作用；不保证真实浏览器或并发覆盖。
- :537 metrics_second_opinion_fast_bss_eval：metrics.core 缺失 importorskip；固定合成正弦/噪声，SDR 与同一 fast_bss_eval 后端再调用、SI-SDR 与 numpy 解析式对比，检查提升值及符号；不覆盖静音参考。
- :569 metrics_tracker_csv_sign_and_final_rows：allwrapper 导入异常直接 skip、core 缺失 skip；tmp_path 写单句 CSV，核均值/零标准差及 core 对拍；单句不测缺失值聚合。
- :637 diff_scope_only_allowed_files：冻结 git 区间 689555c..ac91800；要求有变更且路径符合历史白名单；只验历史路径、无当前工作区保护。
- :650 changed_python_files_compile：git 基线变更加未跟踪 Python 文件；py_compile 所有存在的文件，直接写对应源码目录缓存；不检验运行行为。
- :665 t1_audio_train_no_undefined_comet_logger：要求基线以来文件有修改；源码文本不得含旧 logger 写法、须含 CSVLogger 或 TensorBoardLogger；文本不是执行语义证明。
- :674 t3_metrics_core_available：要求对应目录有变化，模块缺失 importorskip；四指标可调用、正弦样例浮点有限；不覆盖静音/分布式。
- :689 t5_frcnn_eval_returns_self_and_mdcl_alias：两路径至少一处改变；仅对已改变路径检查 eval 返回 self / train 返回非 None 或 Falcon 有 _mdcl_loss；无视频数值比较。
- :708 t5_models_dunder_all_consistent：要求 models/__init__ 相对基线修改，逐项检查 __all__ 属性存在；__all__ 缺失则空表通过。
- :719 config_tracked_changes_are_comment_only：依赖完整历史，配置相对基线必须有变化；YAML 对象一致且 diff 行只含注释；只针对 Falcon.yml。
- :749 t1_gpu_spec_semantics：要求入口变化；实际解析字符串、列表、计数和错误输入；CUDA 可用时检查设备数及 DDPStrategy；tmp_path 创建 logger 路径；不启动多卡训练。
- :812 lrs2_mix_equals_s1_plus_s2：/root/tmp 指定两个 target/mix 资产缺失 skip；只取前三条，音频截到最短比较 mix 与已缩放 s1+s2、两源不同；不核视频或所有转换数据。
- :839 lrs2_eval_aggregate_matches_lead_evidence：缺两个历史 log skip；解析尾部指标与当时 20 条汇总容差及 target 差距；不重算样本、不能证明目标-嘴型对应。
- :855 lrs2_eval_end_to_end_smoke：旧真实数据或权重缺失 skip；CPU 子进程 cwd=tmp_path 跑两条评测并写结果 CSV，核两行有限和正向指标；不是全量，不覆盖批次对比。

## test_phase2_pipeline.py：19 项

- :170 official_weight_fingerprints_unchanged：非 CI，64 线程不生效 skip；同 Phase1 的单例真实权重与历史基线四条 SHA、参数数；不检验一般 batch 或跨设备。
- :192 scripts_compile_and_help：脚本全缺 skip；收集 phase2/*.py 与转换脚本，py_compile 写源码缓存，含 __main__ 的每个脚本在仓库 cwd 执行 --help；只有参数入口，若脚本 argparse 前执行操作则有实际副作用，不证明正常路径或失败码。
- :222 reports_have_evidence：所有报告不存在 skip；现存报告长度、标题及特定 docs/verification/phase2 下引用的文件存在；进行中报告可缺引用，已知被删除文件例外；未核内容真实性/hash，也不强制六份全存在。
- :257 lrs2_eval_end_to_end_smoke：转换数据和 fallback 均缺、或权重缺失时 skip；两条 CPU 评估在 tmp_path 写结果，检查行数、有限和正向指标；未测逐条原始音画关联。
- :316 eval_csv_aggregation_consistent：无非空候选 CSV skip；读入失败直接 continue；逐句至少2且 avg 存在才检查，有限列重新计算 mean/ddof0 std，非有限 avg/std 和无值列跳过；至少有一个结构合格文件，不保证全部 CSV 内容均检查。
- :353 lrs2_rebuild_dataset_counts：目录缺失 skip；统计存在分区的 mix wav 数量，与 20000/5000/3000 对照；某分区整个目录缺失会被 continue，只要求至少一个存在。
- :404 diff_scope_guard：git 历史区间 ac91800..2beec38 路径白名单；不核当前更改及未跟踪文件，路径前缀允许 configs/、tools/ 等范围宽于部分旧注释。
- :425 docs_verification_no_pid_or_empty_evidence：git tracked 列表非空，禁止 tracked .pid、存在的 tracked 文件 0 字节；删除未恢复的 tracked 文件不会被 empty 列表捕获，未跟踪产物不查。
- :444 p2b_summary_matches_csv：无 summary 或没有任何引用 CSV 可读则 skip；重算有引用且存在文件的样本数、列均值、两种 std；缺失 CSV 直接跳过，不要求所有组合齐全。
- :494 p2c_conversion_log_and_counts：缺日志或 tt/mix 索引 skip；三分区日志字符串、索引条数、各首条路径及日志声称 missing_mouth=0；不逐条检查目标/嘴型是否匹配，不能验证重入错配。
- :519 p2a_anchor_data_checks：缺历史 JSON skip；读取记录值检查 mix 重建、目标及 mouth 等性、格式与指标范围；不重读原始波形和 NPZ，不能证明未篡改或后续转换正确。
- :545 p2e_dnsmos_evidence：缺 synthetic JSON skip；检查 lock 包版本、JSON 内 ONNX 文件名/哈希格式/正字节和预测范围，若 smoke JSON 有则查错误数和均值；不重新散列模型文件/重新推理。
- :580 p2f_metric_implementation_matches_core：缺结果 JSON skip；检查记录的指标差异、逐项对拍、子集均值和部分 checkpoint 元数据；比较落盘值不重新算音频。
- :619 lrs2_config_invariants：缺配置 skip；核数据路径、采样率/段长/near、system、预训练文件名及与 Falcon 配置优化器等一致；不验证文件存在或运行采样。
- :651 p2d_smoke_evidence：缺日志 skip；日志须含 it/s 和训练/验证/测试字段；若报告存在，仅检索首个吞吐字符串；不重新训练或核真实 DDP collective。
- :669 p2b_paired_stats_and_report_consistency：报告或 b300 CSV 全缺 skip；读取存在 CSV，各 300 行、均值/std、报告含数值，按 snt_id 对六对中存在组合重算差和胜场，至少两对；仅集合配对，报告数字子串匹配不保证对应表格，不能证明广泛因果。
- :730 p2a_report_matches_200_json：报告或 JSON 缺失 skip；核 200 条汇总 n、表中均值 std/部分差值字符串及 target 差距；JSON 汇总非逐样本再计算。
- :777 p2d_training_curve_sane：训练 CSV 缺失、训练步少于10 skip；最新文件按 mtime，核步数有序、loss 有限、ep0 边界、val；near/beta 相关行存在才查，可无相关行直接通过，固定期望 near K=0 与配置历史绑定；只读 CSV，不训练。
- :827 p2d_eval_metrics_consistent：旧 epoch0 CSV 缺失 skip；至少5条、均值与 avg 一致，若报告存在则含四种两位数；不验证原始音频及 ckpt 身份。

## 问题覆盖映射

- 批次串扰/Space：两文件指纹 B=1，旧评测仅测汇总或两个样本正值；无同一窗口与不同伴随窗口/批量顺序对照。
- 口型帧索引、首尾补帧：目标 mouth 相等是读取历史 JSON 的断言，真实 smoke 的正指标不能证明时间正确；无逐帧时间戳对拍。
- causal stride2 裁剪：仅有官方 causal=False 相关基线指纹，无 paper_causal stride2 形状、数学期望或梯度测试。
- LRS2 重入复用口型：:494 核索引数量/日志声称与首条音频，:519 是已有 JSON；无同一输出目录先 s1 再 s2 的逐项视频核验。
- 物理 mix 波形：Phase1 :812 最多三条已缩放源进行加和检查；数据错配或增益异常均无法普查。独立排序的 source join 无覆盖。
- 分布式训练：GPU 解析仅验证构造策略，历史训练日志仅检查文本，不会触发不均衡 near bucket 的 all_gather。
- 损失边界：指标正弦非静音，未覆盖 silent SISNRi、MixIT 空分区和 mel_opt 异常。可测试历史数值不等于功能正确。
