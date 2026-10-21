# 00 · 项目全景图（Lead 初版，v0，待各子系统文档校正）

> 本文件是 Lead 在 R1 前的快速侦察结论，作为全员导航。**请把它当作假设而非事实**：
> 各子系统 owner 在 R1 文档中要核实/纠正本文件中与你负责区域相关的内容，并在 R2 评审中提出修订。

## 1. 一句话

Falcon = 因果（causal）音视频目标说话人提取（AVTSE）模型 + 完整训练/评测工程 + 可实时运行的浏览器 Demo（Falcon Studio）。
论文 NeurIPS 2026，仓库含两条相互独立但共享模型定义的产品线。

## 2. 两条产品线与目录归属

| 产品线 | 范围 | 关键目录 | 运行方式 |
|---|---|---|---|
| 训练 & 评测 | 从数据构建到训练、评测、算力统计 | `look2hear/`、`audio_train.py`、`audio_visual_test.py`、`evaluated_mac_params.py`、`configs/`、`DataPreProcess/`、`Experiments/` | `pip install -r requirements.txt`; `python audio_train.py --conf_dir configs/Falcon.yml` |
| Falcon Studio | 浏览器采集 + 本地推理服务 + 会议输出 | `studio/`（`server.py`、`falcon_runtime.py`、`infer_file.py`、`falcon_model/`、`web/`、`scripts/`、`tests/`） | `cd studio && ./scripts/launch.sh` |

另有：`project_page/`（GitHub Pages 项目页，`.github/workflows/pages.yml` 发布）、`assets/`（README 图）、
`README.md` / `README_zh.md`、`LICENSE`（Apache-2.0）。

## 3. 顶层目录速查（含 LOC 量级，用于分配注意力）

```
Falcon/
├── audio_train.py            203  训练入口（DDP/PL，读 YAML，建 datamodule/model/video_model/optim/sched/callbacks）
├── audio_visual_test.py      329  评测入口（SI-SDR/SDR，单/多卡，best ckpt 发现，hf:// 权重）
├── evaluated_mac_params.py   151  参数量 / MACs / 延迟统计
├── configs/Falcon.yml        171  论文配置（模型超参 + 训练 + 数据 + exp_name=Falcon）
├── DataPreProcess/
│   ├── process_lrs23.py      121  生成 mix.json / target.json
│   └── generate_near_json.py 105  生成 near.json（对比学习负样本索引）
├── look2hear/               11952  训练侧代码（见下）
├── Experiments/checkpoint/Falcon/
│   ├── conf.yml              168  实验配置快照
│   └── best_k_models.json      6  最优 ckpt 元数据
├── assets/                         README 插图（png/jpg）
├── project_page/index.html   801  静态项目页（figures + 演示媒体引用 oss.cslikai.cn）
├── studio/                  35237  Studio（含 38MB vendor）
└── .github/workflows/pages.yml 35  GitHub Pages 部署（私有仓库不部署）
```

`look2hear/` 内部（LOC 为文件行数）：

```
look2hear/
├── models/      Falcon.py 1487（核心！AudioBlock/VideoBlock/Separator/MaskGenerator/STFT 编解码/Falcon/DualPathRNN/MHSA2D…）
│                Falcon_Contrastive.py 122（FalconSpeakerContrastive，加 speaker pooling/contrastive）
│                base_model.py 102
├── layers/      cnnlayers 884, rnnlayers 908, stft 785, enc_dec 461, normalizations 245, stft_tfgn 245, activations 67
├── losses/      matrix 382（PairwiseNegSDR 系）, pit_wrapper 324, mixit 309, nass_contrastive_loss 81, sisnri 41
├── metrics/     wrapper 84, allwrapper 120, splitwrapper 115
├── datas/       avspeech_dataset 327, avspeech_dataset_near 337, check_avspeech_dataset 347, transform 187
├── system/      av_litmodule 240, av_litmodule_nass_v2 185, av_litmodule_mdcl 173, optimizers 106, schedulers 111
├── utils/       nets_utils 502, parser_utils 151, stft 785（与 layers/stft.py 疑重复）, lightning_utils 102,
│                complex_utils 183, separator 69, torch_utils 42, get_layer_from_string 42, attn 28, inversible_interface 12
└── videomodels/ FRCNNVideoModel(122) + resnet(140)/resnet1D(139)/resnet_videomodel(125)/shufflenetv2(173)/wujian(182)/light(124)
```

`studio/` 内部：

```
studio/
├── server.py                 337  FastAPI + WebSocket；Engine、OutputSlicer、same_origin；GET /api/status、POST /api/load、/ws
├── falcon_runtime.py         259  常量（16k/25fps/2s/50帧）、VARIANTS、权重解析、strict load、preprocess_mouth、infer
├── infer_file.py             105  离线整段推理 CLI
├── falcon_model/            ~5900 推理版模型代码（Falcon.py 1496、SISNRRepel.py 1498、Speaker_Contrastive_v2.py 140 等）+ PROVENANCE.md
├── configs/Falcon.yml 171 / FalconBig.yml 166
├── web/                      ~3200 浏览器端：app.js 1165、mouth-tracker、capture-worklet、video-worker、playback-resampler、
│                                  spectrum-dsp/worker、audio-inspector、i18n*、index.html、style.css、vendor/mediapipe(38MB)
├── scripts/                  bootstrap.sh、launch.sh、doctor.sh、check_environment.py、install-blackhole.sh
├── tests/                    test_*.py（4 个）、check_*.mjs/cjs（5 个）、browser/、serve_browser.py
├── Start Falcon.command / Setup Meeting Audio.command（macOS Finder 启动器）
└── requirements.txt / requirements-lock.txt / requirements-dev.txt
```

## 4. 宏观数据流（待各 owner 细化）

### 4.1 训练流（推测，待 train-sys 验证）

```
LRS2/LRS3/VoxCeleb2 数据（tr/cv/tt：mix/*.wav + target/*.wav + mouth 帧）
  → DataPreProcess/{process_lrs23,generate_near_json}.py → mix.json/target.json/near.json
  → AVSpeechDataModuleWithNear（datas/avspeech_dataset_near.py, segment=2.0s, 16kHz）
  → AudioVisualLightningModuleNASSv2 / MDCL / 基础 litmodule（system/）
     ├─ audio model: FalconSpeakerContrastive ← configs/Falcon.yml audionet
     ├─ video model: FRCNNVideoModel（冻结 lip-reading backbone, pretrain_zoo/frcnn_128_512.backbone.pth.tar）
     └─ loss: PITLossWrapper(pairwise_neg_snr) + NASS speaker contrastive（β annealing）
  → PyTorch Lightning Trainer（DDP, AdamW 1e-3, ReduceLROnPlateau, EarlyStopping(30), grad clip 5）
  → Experiments/checkpoint/Falcon/{last.ckpt, best_k_models.json, conf.yml}
```

### 4.2 评测流（推测，待 loss-metric-eval 验证）

```
audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml [--ckpt_path hf://JusperLee/Falcon/falcon.pt]
  → _find_best_ckpt → _load_models_from_ckpt → process_worker（单/多卡）→ DataLoader(test_dir)
  → 前向 → SI-SDR/SDR（metrics/wrapper 等）→ Experiments/checkpoint/Falcon/results/metrics_combined.csv
```

### 4.3 Studio 实时流（推测，待 studio-backend / studio-frontend 验证）

```
浏览器: getUserMedia(audio/video) → AudioWorklet 重采样 16kHz → 累积 2s 窗口
        MediaPipe FaceLandmarker(worker) 追踪嘴部 → 96×96 灰度 crop（25fps）
        → WebSocket /ws 发送 {audio, frames} → server.py Engine.infer → FalconRuntime
        → 模型输出 → OutputSlicer 取最新 0.5s（保留 80ms 右上下文）→ 回传
        ← 前端 playback-resampler 重采样到设备率 + crossfade 拼接
        → 耳机监听 / 录音与导出 WAV / BlackHole 虚拟麦克风供 Zoom/腾讯会议
```

## 5. 关键数字（需子系统文档逐一验证）

- 模型规模：可训练 0.5M（论文口径，不含冻结 11.2M FRCNN）；含视觉编码器总计 11.72M（README 表）。
- 效率：2s@16k 输入 + 50 帧嘴部裁剪；MACs 10.6G；GPU 延迟 51.3ms（RTX 3090）；CPU 1361.2ms。
- Studio：窗口 2s、hop 0.5s、80ms 右上下文、首输出 ~0.58s 输入 + 1.42s 静音左上下文；M5 Pro 单窗口 ~0.12s。
- 结果：RealSSA SI-SNRi 7.7 / SDRi 8.2；LRS2-2Mix 14.1/14.7；LRS3-2Mix 16.4/16.7；VoxCeleb2-2Mix 10.5/11.8；Real DNSMOS 3.276。

## 6. 八个理解域与负责人（R1 分工）

| # | 域 | 负责人 | 文档 |
|---|---|---|---|
| 01 | 核心模型架构（Falcon.py / Falcon_Contrastive.py / base_model） | `model-core` | `01-model-core.md` |
| 02 | 层与 DSP（layers/、utils 中数值工具；含 stft 重复实现核对） | `layers-dsp` | `02-layers-dsp.md` |
| 03 | 训练系统（入口/配置/Lightning modules/optim/sched） | `train-sys` | `03-training-system.md` |
| 04 | 损失、指标、视觉模型、评测入口 | `loss-metric-eval` | `04-losses-metrics-video.md` |
| 05 | 数据管线（datas/、DataPreProcess/、transform） | `data-pipe` | `05-data-pipeline.md` |
| 06 | Studio 后端（server/runtime/infer_file/falcon_model/PROVENANCE） | `studio-backend` | `06-studio-backend.md` |
| 07 | Studio 前端（web/ 全部 + 浏览器端协议/追踪/播放/i18n） | `studio-frontend` | `07-studio-frontend.md` |
| 08 | 运维、测试、CI、文档与配置一致性 | `ops-tests-docs` | `08-ops-tests-docs.md` |

## 7. Lead 侦察阶段发现的“高风险待验证点”（R1 优先回答）

1. `look2hear/layers/stft.py` 与 `look2hear/utils/stft.py` 都是 785 行：是复制粘贴还是不同实现？哪份被实际 import？
2. `look2hear/models/__init__.py` 的 `__all__ = ["AVTIGER_causal"]` 与实际导出的 `Falcon`/`FalconSpeakerContrastive` 不一致，`get()` 是否受影响？
3. `configs/Falcon.yml` 中 `audio_params.layers.layer_1/layer_2/layer_3` 与 `AudioBlock` 构造的映射关系；`repeats: 2`、`upsampling_depth: 4` 的语义。
4. “因果性”到底如何实现（causal padding / 单向 RNN / 无未来帧），是否严格满足 streaming 因果？
5. Studio 的 `falcon_model/` 与 `look2hear/` 的模型代码差异（PROVENANCE 声称只去掉训练依赖 + 加 Metal FFT bridge）；`SISNRRepel.py` 是否只用于训练侧对比损失？
6. `audio_train.py` 默认配置路径指向不存在的 `NASSv2_lrs3_complex_nonmoving.yml`；实际训练用 `configs/Falcon.yml`，两处 `datamodule.data_config` 字段是否完全匹配 `AVSpeechDataModuleWithNear.__init__`？
7. 评测入口 `_find_best_ckpt` 的顺序（best_k_models.json → last.ckpt → epoch=*.ckpt）与 `metrics_combined.csv` 的写入位置/格式。
8. Studio 前端 `app.js` 的 1.42s 静音左上下文、80ms guard、OutputSlicer 的 endpoint 语义是否前后端一致（同一套采样数常量）。
9. `studio/server.py` 的 `same_origin` 安全限制到底覆盖哪些路径/方法；`POST /api/load` 与 `/ws` 的鉴权边界。
10. 测试覆盖真实程度：哪些 `tests/check_*.mjs` 是纯函数测试、哪些需要浏览器/MediaPipe；`test_*.py` 需要哪些依赖。

## 8. 环境限制（影响“完全理解”的验证边界）

- 无 torch 的系统 Python；无真实权重、无 `pretrain_zoo/`、无 LRS2/LRS3 数据 → **端到端训练/评测不可复现**，
  只能做“代码级 + 可运行的最小验证（JS 单测 / AST / 参数量与前向形状用外部 3.10+torch 环境）”。
- 结论中必须诚实区分 `[代码] / [推断] / [验证]`。
