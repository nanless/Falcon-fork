# tools/repro/tests 审阅进度

范围：16 个 test_*.py，共 4915 行。对全部 test 函数做了 AST 断言提取和索引；16 个文件均已逐体通读；Phase1/Phase2 两个大文件的逐测试记录另见 phase1-phase2-details.md。之前独立执行的四个文件共 29 passed；本页记录代码审阅结果。

|文件|阅读状态|关键断言和依赖|写入与待补检查|
|---|---|---|---|
|test_audio_train_entry.py|通读|假 Trainer 截断 fit，真实音频模型；设备、配置和日志装配|tmp_path 写 conf；无实际 DDP、训练或条件 all_gather 检测|
|test_audit_data.py|通读|微型假数据、near sentinel、形状/采样率及明确 bytes/SHA 检验|tmp_path 创建及删除；独立 verify_download 不覆盖下载恢复分支将超长文件当成功|
|test_eval_entry.py|通读|checkpoint 优先级、环境路径、state dict、rank CSV 合并|tmp_path 建 shard 并由合并删除；无多 rank 竞争、异常值和视频跨样本独立性|
|test_litmodule_logging.py|通读|三种 Lightning 模块及 dummy 模型 CSV logger 真实 CPU 短 fit|tmp_path 写日志、checkpoint 禁用；实际进行单设备短训练；DDP collective 待检查|
|test_lock_portable.py|通读|lock 格式、重复、cu128、与当前 pip freeze 集合版本一致|本地 subprocess 只读；新环境安装和二进制可用性待检查|
|test_lrs2_convert.py|通读|假音频、单独 s1/s2 target 与 mouth、near、dry-run、重复同参数、损坏 mix|tmp_path 输出；未覆盖同目录切换 target 且口型已存在，也未断言 total_seconds 正确|
|test_lrs2_paper_build.py|通读|双向 target/near 路径、复制模式、count gate、near 自引用|tmp_path 构造并有意改 JSON；未覆盖 process_lrs23 独立排序错配|
|test_metrics.py|通读|SI-SDR/SDR 解析值及 fast-bss-eval 对拍、CSV、常规 PIT/SISNRi|tmp_path CSV；不覆盖静音参考、MixIT 空分区、mel_opt AttributeError、自定义 EPS|
|test_model_rebuild.py|通读|配置不污染、525 state keys、序列化重建位相同、FRCNN 权重 strict|临时权重创建及 unlink；单样本重建一致；batch 依赖和 causal stride 裁剪待测|
|test_phase1_regression.py|通读（21 项；逐项详见 phase1-phase2-details.md）|权重哈希、形状与指纹、基线工作树、指标与历史日志|显式 py_compile 会写源码目录 __pycache__，并在仓库 cwd 嵌套执行 Studio 全套 pytest；部分依赖既有工作树和真实数据，欠缺资产可 skip；固定 batch=1 指纹不测批次串扰|
|test_phase2_data.py|通读|已转换数据系统抽样检查索引、mouth shape/name、s1 内容与混音和 near 路径|只读真实资产，缺失可 skip；抽样核对 mouth 形状和总帧数；抽帧时间待核对|
|test_phase2_launcher.py|通读|绝大多数为 shell 文本/YAML 字段；小 loss；monitor 同目录 version 顺序|tmp_path 伪日志/ckpt，seed 改进程随机状态；不运行 launcher；不验证累积步数公式或跨目录新旧 run|
|test_phase2_pipeline.py|通读（19 项；逐项详见 phase1-phase2-details.md）|历史指标/日志/报告一致性、权重指纹、真实 eval 限2条|eval 子进程 cwd=tmp_path；但 py_compile 直接在源码目录写 __pycache__，逐脚本 --help 也有潜在副作用；依赖真实资产、可跳过；相关性或报告一致不证明归因|
|test_realssa_convert.py|通读|合成 tar，channel 0/mean，长度取 audio/mouth 最短、缺项、external near|tmp_path 生成 tar 和转换；只比长度/形状，未校验真实帧时序索引或 Studio/RealSSA 首窗边界等价|
|test_synthetic_data.py|通读|合成数据生成、训练随机 item 维度、mono/stereo eval|tmp_path 生成数据；音画时间戳对齐及随机采样的 idx 统计性质待测|
|test_t5_backbone.py|通读|导出对象、权重 137 键及参数，frozen BN、MDCL alias 和一次 forward|CPU 创建模型；依赖缓存文件缺失则 skip；batch=1 不比较同一样本与其他样本共同推理|

优先补充检查：1. VideoBlock 批样本混合与 Space windows/批数差异；2. 时间索引和首尾视频填充比较而非形状断言；3. paper_causal stride=2 输出裁剪与几何，当前目录无直接覆盖；4. 同输出目录切换 LRS2 target 的一致性；5. process_lrs23 按键严格 join；6. 多 rank 非对称 near bucket 和 checkpoint 写入竞争；7. loss 退化输入及混合分区；8. launcher 全脚本失败传播。
