# 03 视频前端与唇部 ROI 管线

> 范围：`look2hear/videomodels/*`、`look2hear/datas/transform.py`、`DataPreProcess/`、`tools/repro/lrs2_2mix_to_falcon.py`、`studio/`、`space/` 中的一切 video/mouth 预处理，外加 FRCNN 权重实物。
> 仓库 HEAD `6c1d5b7`（2026-10-30），只读 + CPU 实测，未写远端仓库。全部命令见 §8。
> 实测脚本落在远端 `/tmp/w103/probe1..9.py`（非仓库路径），Python 一律 `/root/miniforge3/envs/falcon/bin/python`。
> 论文 arXiv:2610.32749 已于 2026-10-23 公开。本报告 2026-11-08 经 Lead 复核，更正视觉/整窗 MACs 与占比一处。

---

## 0. 结论速览

1. **默认前端只有一个**：`FRCNNVideoModel(backbone_type=resnet, relu_type=prelu, width_mult=1.0, pretrain=/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar)`，由 `configs/LRS2_2Mix.yml:1-7` 指定、`audio_train.py:308-310` 用 `getattr(look2hear.videomodels, name)` 实例化。其余 6 个前端是死代码/试验品，默认路径一个都不走。
2. **模型输入不是 128×512**。真实链路是「离线存 96×96 灰度 → Dataset 内中心裁 88×88 → /255 → (x−0.421)/0.165」(`transform.py:147-169`)。512 是**输出通道数**，不是宽度；`128×512` 只是上游文件名，代码里没有任何 128×512 的空间路径（§4-A 反证：强行 `CenterCrop((128,512))` 得到 `(50,16,96)` 垃圾张量）。
3. **FRCNN 输出 `(B,512,T)`，T 不降采样**：Conv3d 时间 kernel=5/pad=2、MaxPool3d 时间 kernel=1 → `T'=T`（实测 T=1/25/50/120 全部原样输出，§4-E）。
4. **11,186,688 参数（11.187 M），加载后 100% 冻结**（`requires_grad=False` 由 `update_frcnn_parameter` 强制，`frcnn_videomodel.py:196-197`），并且**不进 optimizer**（`audio_train.py:314` 只喂 `model.parameters()`），是双重冻结。
5. **计算量大头在视频侧**：88×88×50 帧一次前向 **15.897 GMAC**（ptflops 同尺，本次复算 15.81 GMAC 差 0.5%）。
   音频侧同一把尺子是 **10.611 GMAC**，不是 5.38——后者来自 `torch.utils.flop_counter`，它不记 `aten::lstm`。
   整窗 **26.51 GMAC、视觉占 60.0%**。README 的「10.6 G MACs」只是音频侧。
   2026-11-08 Lead 复核更正：本节原写"整窗 21.19 GMAC / 视觉 74.6%"是拿 5.38 与 15.81 相加得来的混尺数，
   两把尺子不可相加；74.6% 更正为 **60.0%**。
6. **视觉时间感受野 ±2 帧（±80 ms），不是严格因果**：扰动第 0 帧只影响输出 0/1/2，扰动第 49 帧只影响 47/48/49（§4-J）。所谓 causal 是「窗口级」，视觉帧级会看未来 2 帧；Studio 的 80 ms guard 正好等于这个值。音频侧的 lookahead 是另一套数：单遍几何 57 帧 / 456 ms、整链 113 帧 / 904 ms、as-shipped 窗尾（见 12 §3）。
7. **时空对齐靠 `F.interpolate(mode="nearest")`**：视频 50 帧 → 音频 251 帧（`Falcon.py:258-260`），比例 5.02，每帧视频复述 5 或 6 个音频帧。注意音频 T 是 **251 不是 250**（STFT `center=True`），所以 50→251 不是整齐的 5×。
8. **离线 ROI 是预生成的**：MTCNN 检测 + `face_alignment` **68 点** → 相似变换对齐到平均脸（256×256，稳定点 33/36/39/42/45）→ 以嘴部 20 点（48–67）均值中心切 **96×96** → BGR2GRAY → `mouths/<id>.npz (50,96,96) uint8` → 本仓库转成 `(50,96,96,3) uint8` 的 `.pt`。**取帧上限 `int(video.fps*2)` 写死 2 秒**，这就是「每个 .pt 恰好 50 帧」的原因（`lrs2_rebuild/detectface.py:67`）。
9. **在线 ROI 是另一套**：MediaPipe Face Landmarker **478 点**（BlazeFace+FaceMesh-V2，`studio/web/vendor/mediapipe/NOTICE.md:7`），用 61/291 嘴角 + 234/454 脸宽算正方形框（比例常数 2.13），**直接 drawImage 到 96×96 并灰度化，无仿射对齐**（`mouth-tracker.js:46-52`、`video-worker.js:88-95`）。README 自己承认这是与训练分布不同的近似（`studio/README.md:101`）。
10. **三条坑**：`.pt` 存的是 3 通道复制（3× 磁盘浪费，`lrs2_2mix_to_falcon.py:224`）；Dataset 吐 **float64**（`transform.py:73` 的 numpy 除法），靠 `mouth_in.type_as(mixtures)` 在 litmodule 里救回来（`av_litmodule_nass_v2.py:87`）；`mouth_start = int(rand_start/16000*25)` 把采样率写死 16000（`avspeech_dataset_near.py:168`）。

---

## 1. 事实基线

### 1.1 前端清单与事实

| # | 类 | 文件:行 | 输入 | 输出 | 参数 | 冻结 | Falcon 默认 |
|---|---|---|---|---|---|---|---|
| 1 | `FRCNNVideoModel`(resnet/prelu) | `frcnn_videomodel.py:29-85` | `(B,1,T,H,W)` 灰度 z-score | `(B,512,T)` | **11,186,688** | 加载 pretrain 即冻结 | **✅ 唯一默认** |
| 2 | `FRCNNVideoModel`(resnet/relu) | 同上 | 同上 | `(B,512,T)` | 11,182,784 | 同上 | 否（配置不选） |
| 3 | `FRCNNVideoModel`(shufflenet) | `:44-52` | 同上 | `(B,1024,T)` | 1,258,860 | 同上 | 否（且 1024≠512，与 `pretrained_vout_chan` 冲突） |
| 4 | `LightVideomodel` | `light_videomodel.py:16-62` | 同上 | `(B,1024,T)`（w=2.0 时 2048） | 1,258,860（w1.0）/347,048（w0.5） | 同上 | 否 |
| 5 | `ResNetVideoModel` | `resnet_videomodel.py:22-68` | 同上 | `(B,512,T)` | 11,186,688 | 同上 | 否（FRCNN 的去分支简化版） |
| 6 | `WujianVideoModel` | `wujian_videomodel.py:142-164` | **`(B,T,H,W)`**（内部 `unsqueeze(1)`） | `(B,256,T)` | 11,318,465 | `update_wujian_parameter` 才冻结 | 否 |
| 7 | `ResNet` / `BasicBlock` | `resnet.py:73-140` | `(BT,3,H,W)` 2D | `(BT,512)` | 组件 | — | 被 1/5 复用 |
| 8 | `ShuffleNetV2` / `ResNet1D` | `shufflenetv2.py:109-173`、`resnet1D.py:73-139` | 2D / 1D | 分类 logits / `(B,512,T)` | 2,278,604 / — | — | 被 3/4 复用 |

要点：

- `FRCNNVideoModel.__init__` 里 `pretrain` 非空就 `self.init_from(pretrain)`（`frcnn_videomodel.py:72-74`），**只有走 pretrain 才会冻结**；`pretrain=None` 时实测 11,186,688 个参数全部 `requires_grad=True`（§4-Q）。所以「FRCNN 冻结」不是类属性，是加载路径的副作用。
- `train()` 重写只锁 BN 统计量、**并且有 `return self`**（`frcnn_videomodel.py:113-125`），实测 `train(True)` 后所有 `_BatchNorm.training == False`（§4-Q）。注意：这一条与我方既有文档的旧说法冲突，见 §6-C1。
- `WujianVideoModel` 的 3D 前端与 FRCNN 结构等价（Conv3d(1→64,(5,7,7)) + BN3d + PReLU + MaxPool3d），但 trunk 换成带 `fc`+`bnfc` 的 `wujian.ResNet`（`wujian_videomodel.py:52-53,94-95`），输出 256 维；`update_wujian_parameter` 里 `if "resnet" in k` 在 `k` 已被 `k.split(".")` 改成 list 之后执行（`:171-176`），子串语义失效，属于潜在加载 bug（当前无人调用，故只是隐患）。
- `ResNet` 的 `num_classes=1000` 形参**从未使用**，类里没有 `fc`（`resnet.py:73-140`）：`layerN → AdaptiveAvgPool2d(1) → flatten → 512`。这就是「去掉 LRW 分类头」的落点——不是 `nn.Identity()` 替换，而是根本没建。
- `resnet1D.py` / `shufflenetv2.py` 只被当作组件引出（`videomodels/__init__.py:4-6`），Falcon 推理路径不触达。

### 1.2 「128×512」到底是什么

三处硬证据表明它不是模型输入：

1. `configs/LRS2_2Mix.yml:6` 只把 `frcnn_128_512.backbone.pth.tar` 当文件名，`backbone_type: resnet`、无 `crop`/`resize` 参数；FRCNN 类也没有任何 crop/断言（`frcnn_videomodel.py:30-74`）。
2. `tools/repro/extract_video_backbone.py:17-20` 明确写：`configs/Falcon.yml` 期望该文件在 `pretrain_zoo/`，**上游 README 只有 TODO**，是本仓库从 `falcon.pt` 里抽出来的，因此文件名沿用上游/作者口径，不代表输入尺寸。
3. 实测把 `CenterCrop((128,512))` 塞进 transform 不报错，但输出 `(50,16,96)`——纯垃圾（§4-B）。若真是 128×512 输入，管线不可能静默产出这种形状。

权重里也只有 `(64,1,5,7,7)` 的 3D 卷积核（§4-C），空间尺寸完全由输入决定。**结论：`128` 的来源未定，进 GAP**（§7-G1）。

---

## 2. 机制拆解

### 2.1 FRCNN 逐层张量流（96×96 与 88×88 两条实测路径）

```
x                     (B, 1, T, H, W)        H,W ∈ {96, 88}（存储 96，裁后 88）
frontend3D[0] Conv3d(1→64, k=(5,7,7), s=(1,2,2), p=(2,3,3), bias=False)
                      (B, 64, T, H/2, W/2)   ← 时间不变、空间减半
frontend3D[1] BatchNorm3d(64)
frontend3D[2] PReLU(64)
frontend3D[3] MaxPool3d(k=(1,3,3), s=(1,2,2), p=(0,1,1))
                      (B, 64, T, H/4, W/4)   ← 96→24, 88→22（实测 §4-E）
threeD_to_2D_tensor   (B*T, 64, H/4, W/4)    transpose(1,2)+reshape（frcnn_videomodel.py:17-20）
trunk.layer1 (64, ×2, s1)   → (B*T, 64, 24, 24) / (…,22,22)
trunk.layer2 (128,×2, s2)   → (B*T,128, 12, 12) / (…,11,11)
trunk.layer3 (256,×2, s2)   → (B*T,256,  6,  6) / (…, 6, 6)
trunk.layer4 (512,×2, s2)   → (B*T,512,  3,  3) / (…, 3, 3)
trunk.avgpool AdaptiveAvgPool2d(1) → (B*T,512)  （resnet.py:95,138-139）
view(B,Tnew,512).transpose(1,2) → (B, 512, T)   （frcnn_videomodel.py:84-85）
```

- 时间维只有 `Conv3d` 的 kernel-5 参与，且 `stride=1, padding=2` ⇒ **T 不变**；`MaxPool3d` 时间 kernel=1、stride=1 ⇒ 也不降。故 `T' = T`。
- 时间感受野：kernel 5 / stride 1 / pad 2 ⇒ 每个输出帧看输入 `[t-2, t+2]`，即 **±2 帧 = ±80 ms @25fps**。实测与推导一致（§4-J）。
- 空间下采样共 4 次 ×2（Conv 1 次 + MaxPool 1 次 + layer2/3/4 各 1 次）= `/16`，从 88 → 5.5 → layer4 输入 6 → 输出 3（`ceil(6/2)`），所以 **88 与 96 都能算，但特征图不一样大**（3×3 都是 3×3 只在 88/96 这两个特定尺寸巧合成立：96/16=6→3，88/16=5.5→6→3）。
- `relu_type` 影响参数量：`prelu` 比 `relu` 多 11,186,688−11,182,784 = **3,904** 个 PReLU 通道参数（64×1 + 16×2×… 由 BasicBlock `nn.PReLU(num_parameters=planes)` 累计，`resnet.py:45-46`）。

### 2.2 权重结构与加载（strict 是真的 strict）

`/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar`（软链到 group-shared，44.83 MB）：

```
dict: {"model_state_dict": {137 tensors}, "meta": {...}}
meta = {"source_checkpoint": "falcon.pt", "source_architecture": "FalconSpeakerContrastive",
        "source_epoch": 714, "num_keys": 137, "note": "…visual encoder is frozen during training…"}
前缀：frontend3D.* = 7 个键，trunk.* = 130 个键；无 audio_model.*、无 tcn*
元素和：11,196,308（= 参数 11,186,688 + BN buffer 9,620）
```

- 加载链：`FRCNNVideoModel(pretrain=p)` → `init_from(p)`（`:87-111`）→ 校验文件存在 / 要求顶层是 `{"model_state_dict": …}` / 非空 → `update_frcnn_parameter`（`:157-198`）。
- `filter_frcnn_pretrained_dict`（`:143-154`）只放行 `("frontend3D.", "trunk.")` 前缀、并剔除任何含 `"tcn"` 的键（`:139-140`）。当前权重里 `tcn` 键为 0 个，所以过滤器是**为上游 LRW 检查点留的兼容位**，对本仓库权重是 no-op。
- **默认 `strict=True`**（`:157`），缺键/多键/形状不符一律 `RuntimeError`（`:184-193`），随后 `model.load_state_dict(filtered, strict=True)` 并 `requires_grad=False`（`:195-197`）。实测 5 种失败路径全中（§4-N）。
- 对照：`update_resnet_parameter`（`resnet_videomodel.py:112-125`）只剔除 `tcn` 后 `model_dict.update(update_dict)` 再整体 load —— **缺键不会报错**（缺的键保留随机初始化）；`update_light_parameter`（`:106-118`）连断言都没有；`update_wujian_parameter`（`:167-182`）缺键静默。**只有 FRCNN 这条是被加固过的（A2-2）**。
- 推理侧不复用 `pretrain`：`audio_visual_test.py:310-326` 优先从 ckpt 里取 `video_model.*` 严格加载，取不到才回退到 pretrain，两者都无则报错。Studio 同理（`studio/falcon_runtime.py:188-207`，`video_config["pretrain"]=None`）。
- `falcon.pt` 侧核对：662 键 = `audio_model.*` 525 + `video_model.*` **137**，`architecture=FalconSpeakerContrastive`、`epoch=714`；`falcon-big.pt` 也是 137 个视觉键、`trunk.layer4.1.bn2.weight` 同样是 `(512,)` ⇒ **big 的差异全在音频侧，视觉前端一模一样**（§4-C2）。这也是 `pretrained_vout_chan: 512` 在两个配置里都不变的原因。

### 2.3 离线 mouth ROI 管线（数据侧，非本仓库代码）

源目录 `/root/data/lrs2_2mix/lrs2_rebuild/`（脚本与产物都在数据盘，不在 git 里）：

```
raw video ──detectface.py──► faces/<name>.mp4 + landmark/<name>.npz
          ──crop_mouth_from_video.py──► mouths/<name>.npz (F,96,96) uint8
          ──tools/repro/lrs2_2mix_to_falcon.py──► falcon_lrs2_2mix/<split>/mouths/<id>.pt (F,96,96,3) uint8
```

**检测/对齐（`detectface.py`）**

| 步骤 | 证据 |
|---|---|
| MTCNN 检测人脸框（`keep_all=True`，取前 `number_of_speakers` 个） | `lrs2_rebuild/detectface.py:61,70-74` |
| 框扩成正方形、放大 1.5×（`face2head`） | `detectface.py:17-27,74` |
| 人脸裁出后 **resize 到 224×224** 再送关键点 | `detectface.py:83` |
| `face_alignment.FaceAlignment(LandmarksType._2D)` → **68 点** | `detectface.py:65,84` |
| 只处理**前 `int(video.fps*2)` 帧**（25fps ⇒ 50 帧） | `detectface.py:67` |
| 多人时用 IoU 跟踪分配 | `detectface.py:90-98` |

**裁剪（`crop_mouth_from_video.py`，源自 Pingchuan Ma 的 Lipreading_using_TCN）**

| 参数 | 值 | 证据 |
|---|---|---|
| crop 尺寸 | **96×96** | `crop_mouth_from_video.py:32-33` |
| 嘴部关键点 | **start=48, stop=68**（20 点） | `:34-35` |
| 平滑窗 | `window_margin=12` 帧滑动平均 | `:36,73-74` |
| 灰度 | `--convert-gray` 默认 True | `:38,180` |
| 平均脸 | `20words_mean_face.npy` | `:30,49` |
| 稳定点（估计相似变换） | `[33,36,39,42,45]`（鼻梁+双眼角） | `:50` |
| 对齐目标尺寸 | `STD_SIZE=(256,256)` | `:48` |
| 变换类型 | `skimage` `estimate_transform('similarity', …)`，`tf.warp(inverse_map=…)` | `lrs2_rebuild/transform.py:24-29` |
| 裁剪中心 | 20 个嘴部点的**均值**，`height=crop_h//2=48` | `transform.py:38-40,60-61` |
| 越界处理 | 贴边夹到 `[height, H-height]`，超 5 px 直接抛异常 | `transform.py:42-58` |
| 缺失帧关键点 | 线性插值 + 首尾复制 | `crop_mouth_from_video.py:122-143` |
| 灰度公式 | `cv2.cvtColor(..., COLOR_BGR2GRAY)`（BT.601） | `transform.py:65-66` |
| 落盘 | `save2npz(mouths/<name>.npz, data)` | `crop_mouth_from_video.py:180-181` |

**本仓库只做格式搬运**（`tools/repro/lrs2_2mix_to_falcon.py:212-225`）：

```python
mouth = np.load(npz)["data"]                  # (50,96,96) uint8，校验 shape[1:]==(96,96) 与 dtype
mouth_rgb = np.repeat(mouth[:, :, :, None], 3, axis=-1)   # (50,96,96,3) uint8 ← 3 通道复制
torch.save(torch.from_numpy(mouth_rgb), mouths/<id>.pt)
```

命名规则：`target.json` 的 mouth 路径 = target wav 名里 `<id>` 对应到 `mouths/<id>.pt`（`DataPreProcess/process_lrs23.py:25` 用字符串替换 `out_filename → "mouths"`、`.wav → .pt`）。所以 **mouth 是按 speaker-utterance 去重共享的**：tr 有 20000 条 mix 但只有 **14,773** 个 mouth 文件（`conversion_stats.json` 的 `mouth_files`），cv 5000→476，tt 3000→267（§4-H）。

### 2.4 在线 mouth ROI 管线（Studio / Space）

**Studio（实时，浏览器 Worker）**

| 环节 | 事实 | 证据 |
|---|---|---|
| 模型 | MediaPipe `@mediapipe/tasks-vision` 1.0.1，Face Landmarker float16 v1，BlazeFace + **FaceMesh-V2 478 点**，VIDEO 模式，numFaces=2，CPU WASM | `studio/web/vendor/mediapipe/NOTICE.md:3-9`、`video-worker.js:29-37` |
| 关键点选择 | 嘴角 `61`/`291`，脸宽 `234`/`454`，鼻尖 `1`，外唇轮廓 20 点用于画示意 | `mouth-tracker.js:2-3,41-53` |
| 中心 | 嘴角中点 `(L61+R291)/2` | `mouth-tracker.js:49` |
| 边长 | `faceWidth × scaleRatio`，初始 `scaleRatio = clamp(2.13 × mouthWidth/faceWidth, .45, 1.1)` | `mouth-tracker.js:46-52` |
| 2.13 的来源 | 平均脸里 96 px 框中嘴宽 45 px（96/45≈2.13） | `mouth-tracker.js:46-47` 注释 |
| 平滑 | 位置 EMA `1-exp(-Δt/65)`，尺度 EMA `1-exp(-Δt/220)` | `mouth-tracker.js:94-98` |
| 丢失处理 | `graceMs=500` 内滑行（coast），之后要 4 帧一致才重锁 | `mouth-tracker.js:5-11,101-108` |
| 裁剪 | `drawImage(src, cx*W-size/2, cy*H-size/2, size, size, 0,0, 96,96)` ⇒ **直接缩放到 96×96** | `video-worker.js:88-89`、`app.js:568` |
| 灰度 | `round(.299R+.587G+.114B)`（BT.601，作用在 sRGB 值上） | `video-worker.js:91-94`、`app.js:570-573` |
| 编码 | JPEG **quality 0.88** | `video-worker.js:96`、`app.js:575` |
| 节流 | `nextTimestamp += 40000` µs ⇒ **25 fps 上限** | `video-worker.js:59` |
| 窗口装配 | `frameWindow()`：50 个槽位，`targetTime = start + (n+.5)/25`，取**时间最近**的帧；偏差 >0.16 s 直接拒发；clip 之前用「该目标第一帧」当合成历史 | `app.js:577-596` |
| 服务端校验 | 必须 `len(frames)==50`、每帧 96×96、JPEG/PNG、≤40 KB，转 `L` | `studio/server.py:81-97` |
| 归一化 | `frames[:,4:92,4:92]/255 → (x−0.421)/0.165`，`(1,1,50,88,88)` | `studio/falcon_runtime.py:131-140` |

**Space（整段录音，事后切窗）**

- 同一套 96×96 灰度 crop + 时间戳随请求上传（`space/README.md:32`）。
- 常量：`WINDOW=32000, HOP=8000, GUARD=1280, FADE=320, FRAME=640, WINDOW_FRAMES=50`（`space/clip.py:22-25`）。
- 每个视频槽位取「离槽中心最近」的 crop，ties 取更早；clip 前用首帧、clip 后用末帧（`space/clip.py:47-57`）；窗口音频越界补零（`:37-44`）。
- 归一化与 Studio 逐位一致（`space/clip.py:77-80`）。
- 批次推理：`separate()` 每次最多 32 窗（`:83-98`）。

**离线 vs 在线的差异（用户可见的口径差）**

| 维度 | 离线（训练数据） | 在线（Studio/Space） |
|---|---|---|
| 关键点 | face_alignment 68 点 | MediaPipe 478 点 |
| 对齐 | 相似变换对齐到平均脸 256×256 | **无仿射对齐**，只按脸宽定正方形框 |
| 框尺度 | 固定 96 px（相对平均脸） | 随人脸宽变化（2.13×嘴宽） |
| 缩放 | `tf.warp` 到 256 再切 96 | 浏览器 `drawImage` 双线性缩到 96 |
| 灰度 | OpenCV `BGR2GRAY` (BT.601) | JS 手写 BT.601 |
| 压缩 | 无损 npz/uint8 | **JPEG q=0.88（有损）** |
| 时间基准 | 先检测定帧数（50），再切音频 | 音频为时钟，视频按最近时间戳吸附 |
| 文档自述 | — | `studio/README.md:101`：「live input can differ from training conditions」 |

### 2.5 时空对齐规则

**训练/评测（Dataset 层）**

```python
self.seg_len = int(segment * sample_rate)     # 2.0*16000 = 32000 样本
self.fps_len = int(segment * 25)              # 2.0*25   = 50 帧      (near:77-78)
rand_start = random.randint(0, len - seg_len) # 本数据集恒为 0（所有 wav 恰好 32000）
mouth_start = int((rand_start / 16000) * 25)  # floor(rand_start/640)   (near:168)
source_mouth = pipeline(load_pt(...))[mouth_start : 50 + mouth_start]   (near:169-174)
```

- 换算常数：`16000/25 = 640` 样本/帧。2 s = 32000 = 50×640，**整除、无漂移**。
- 帧数不足：**没有 pad**。切片短于 50 就返回短张量（Python 切片静默截断），随后 `collate_with_near` 的 `torch.stack` 会崩（`avspeech_dataset_near.py:40-41`）。当前不炸的唯一原因是**所有样本都被预处理成恰好 32000 样本**（`conversion_stats.json` 里三 split 的 `sample_lengths` 都只有 `{"32000": N}`，实测 mix.json 全长 32000，§4-H）。
- 帧数超出：靠切片上界 `fps_len+mouth_start` 裁掉。
- 评测集完全不切：`AVSpeechEvalDataset` 直接用整段 mouth（`avspeech_dataset.py:194-196`），因为评测样本也是 2 s。
- 两个写死：`/16000`（`:168`）与 `*25`（`:78`）不随 `sample_rate`/帧率变化；`sample_rate=8000` 时 `mouth_start` 会算成 2 倍位置，`seg_len` 也只有 16000，静默错位。

**模型层（fusion）**

```python
# Falcon.py:250-267 NeuroSCNetFusion.forward
v_feat = self.v_proj(video)                              # Conv1d(512→112, k=1, groups=16)
if v_feat.shape[-1] != T_a:  v_feat = F.interpolate(v_feat, size=T_a, mode="nearest")  # :259-260
Z_sel = audio + v_feat.unsqueeze(-1)                     # :263-266 广播到 F=129
```

- 实测音频编码器输出 **251 帧**（不是 250）：`win=256, hop=128, center=True` ⇒ `1 + 32000//128 = 251`（§4-F）。
- 视频 50 帧 → nearest 上采样到 251，比例 **5.02**：`floor(i*50/251)`，前 12 个音频帧对应视频帧 `[0,0,0,0,0,1,1,1,1,1,2,2]`（§4-F）。即每 5–6 个音频帧复用一个视频帧，**一个窗口内无累积漂移**（插值覆盖整段）。
- `v_proj` 的 `groups=math.gcd(512,112)=16`（`Falcon.py:244-246`）⇒ 512→112 的分组卷积，不是 Identity。**`nn.Identity()` 只出现在两处**：(a) `video_bottleneck` 因为 `video_bn_params.kernel_size=-1` 走 `nn.Identity()`（`Falcon.py:32,68-69,86-87` + `configs/LRS2_2Mix.yml:14-15`），实测 `video_bottleneck.conv/norm/act` 三个都是 `Identity()`；(b) `NeuroSCNetFusion.v_proj` 在 `ain==vin` 时才是 Identity（`:247-248`），本配置不触发。
- 视频侧 `VideoBlock` 的 `stride=2` 参数**从未使用**（`Falcon.py:900-976`：只有 kernel=1 卷积 + 单向 LSTM + 残差），所以 `video_net` 输出仍是 `(B,512,50)`（§4-F），50→251 的上采样完全由 fusion 的 `interpolate` 承担。

---

## 3. 关键数据/参数表

| 项 | 值 | 口径/证据 |
|---|---|---|
| FRCNN 参数 | **11,186,688**（11.187 M） | `parameters()` 求和，§4-D |
| FRCNN state_dict 元素 | 11,196,308（+9,620 BN buffer） | §4-C |
| state_dict 键数 | 137（frontend3D 7 + trunk 130） | §4-C |
| 权重文件大小 | 44.83 MB | `/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar` |
| `falcon.pt` | 662 键（525 audio + 137 video），epoch 714，47.2 MB | §4-C2 |
| `falcon-big.pt` | 视觉侧同样 137 键 / 512 通道，epoch 715 | §4-C2 |
| 存储 mouth | `(50,96,96,3)` uint8，1.35 MB/文件 | §4-A |
| 送模型 mouth | `(1,1,50,88,88)` float32 | §4-I |
| 归一化 | `/255` → `(x−0.421)/0.165` | `transform.py:152-158` |
| 视频输出 | `(B,512,T)`，T 不降采样 | §4-E |
| MACs（视觉，T=50，flop_counter） | 88²: 15.81 GMAC；96²: 17.38 GMAC | §4-J |
| MACs（视觉，T=50，ptflops 同尺） | 88²: **15.897 GMAC**（与音频 10.611 可相加） | 05 §3.2 |
| MACs（视觉，T=25） | 88²: 7.90 GMAC | §4-J |
| MACs（音频全链，ptflops 同尺） | **10.611 G**（含 LSTM、不含 FRCNN） | 01 §3.3、05 §3.2 |
| MACs（音频全链，flop_counter） | 5.38 G（不记 `aten::lstm`，不可与 10.611 并列） | §4-K |
| 整窗 MACs | **26.51 GMAC（视觉 15.897 + 音频 10.611，同尺），视觉占 60.0%** | 00 §4、11 §3 |
| 音频参数 | 537,697（0.538 M） | §4-K |
| 时间感受野 | ±2 帧 = **±80 ms** | §4-J |
| CPU 前向（4 线程） | T=50/88² **115 ms**；T=50/96² 145 ms；T=25/96² 80 ms | §4-E |
| CPU 端到端单窗 | **661 ms**（视觉 143 ms + 音频 518 ms） | §4-F |
| mouth 亮度（60 文件） | per-file mean 中位 **92.0**（0.361），归一化后中位 −0.36 | §4-O |
| 纯黑像素占比 | 中位 0.066，最大 0.176 | §4-O |
| tr/cv/tt mouth 文件数 | 14,773 / 476 / 267（去重共享） | `conversion_stats.json` |
| 样本长度 | 全部 32000（三 split 无一例外） | `conversion_stats.json` |

---

## 4. 实测验证

环境：`/root/miniforge3/envs/falcon/bin/python`（torch 2.8.0+cu128），`CUDA_VISIBLE_DEVICES=` 置空，`torch.set_num_threads(4)`，全部 `torch.no_grad()`。

### A. 真实 mouth 样本统计（`/tmp/w103/probe1.py`）

```
[A] mouth .pt sample stats (raw bytes on disk)
n_mouth_pt(tr): 14773
file: 5535415699068794046_00004.pt size_KB=1351.8
shape: (50, 96, 96, 3) dtype: torch.uint8
min/max/mean/std: 0 / 180 / 61.149 / 34.614
channels identical c0==c1: True c0==c2: True
per-frame mean first 5: [72.72, 65.2, 59.92, 47.17, 45.44]
nonzero pixel fraction: 0.8772
frame-count over 25 files/split: {('cv', 50): 25, ('tr', 50): 25, ('tt', 50): 25}
```

⇒ 三通道完全相同（`np.repeat` 复制的灰度），磁盘 3× 浪费；帧数恒 50。

### B. transform 输出（`probe1.py` 续）

```
[B] transform pipeline (as dataset does it)
val pipeline: Compose(
    Normalize(mean=0.0, std=255.0)
    CenterCrop
    Normalize(mean=0.421, std=0.165))
after Grayscale:       (50, 96, 96) uint8 min/max: 0 179
after val pipeline:    (50, 88, 88) float64 min=-2.5515 max=1.7028 mean=-1.0843 std=0.7550
train pipeline: Compose(Normalize(0,255) RandomCrop(88,88) HorizontalFlip(0.5) Normalize(0.421,0.165))
train out:             (50, 88, 88) float64 min=-2.5515 max=1.7028
crop: 96x96 -> (88,88) center, delta=+/-4
  Crop (88, 88)  -> (50, 88, 88)
  Crop (128, 512) -> (50, 16, 96)      ← 反证「128×512 不是输入尺寸」
  Crop (96, 96)  -> (50, 96, 96)
```

- 输出 dtype 是 **float64**（`Normalize` 里 `frames/255.0` 触发 numpy 提升），不是 float32。
- `train` 管线的 `RandomCrop(88)` 与 `val` 的 `CenterCrop(88)` 在 96×96 上等价，因为 `96−88=8`，`random.randint(0,8)`；`HorizontalFlip(0.5)` 会**原地写** `frames[index]`（`transform.py:142-143`），对 uint8 输入无副作用，但会污染复用同一 array 的调用方。

### C. 权重结构（`probe2.py`）

```
[C] FRCNN 权重文件结构
top-level keys: ['model_state_dict', 'meta']
state_dict entries: 137
top-level prefixes: {'trunk': 130, 'frontend3D': 7}
total tensors elements: 11196308 = 11.196 M
frontend3D.0.weight (64, 1, 5, 7, 7)
frontend3D.1.{weight,bias,running_mean,running_var,num_batches_tracked} (64,…)
frontend3D.2.weight (64,)
trunk.layer1.0.conv1.weight (64, 64, 3, 3)
trunk.layer4.1.conv1.weight (512, 512, 3, 3)
any non-visual keys?: []
any 'tcn' keys?: []
```

**C2. `falcon.pt` / `falcon-big.pt` 对照**

```
falcon.pt: 662 keys, prefixes {'audio_model': 525, 'video_model': 137}, video tcn keys: []
BIG entries: 137 elems: 11196308 = 11.196M, prefixes {'frontend3D','trunk'}
BIG trunk.layer4.1.bn2.weight: (512,)
```

### D. 加载 + 冻结（`probe2.py` 续）

```
[D] params total = 11186688 (11.187 M); trainable = 0
all requires_grad False: True
eval() returns self: True
trunk has classifier?: False
```

### E. CPU 前向与 T 不变性（`probe2.py` / `probe3.py`）

```
[E] in (1,1,25,96,96) -> frontend3D (1,64,25,24,24) -> out (1,512,25)   80.0 ms
    in (1,1,50,96,96) -> frontend3D (1,64,50,24,24) -> out (1,512,50)  144.9 ms
    in (1,1,50,88,88) -> frontend3D (1,64,50,22,22) -> out (1,512,50)  115.2 ms
    in (1,1, 1,88,88) -> frontend3D (1,64, 1,22,22) -> out (1,512, 1)   25.0 ms

[G] T=  1 -> (1, 512, 1)     T= 10 -> (1,512,10)     T= 49 -> (1,512,49)
    T= 50 -> (1, 512, 50)    T= 51 -> (1,512,51)     T=120 -> (1,512,120)
```

### F. 端到端（`configs/LRS2_2Mix.yml` + 真实权重，`probe3.py`）

```
[F] video model: FRCNNVideoModel | params=11.187M | trainable=0
audio model: FalconSpeakerContrastive | params=0.538M
video_bottleneck: Identity() | norm: Identity() | act: Identity()
separator.fusion_repeats=1 audio_repeats=1 video_repeat=1
mouth_in: (1, 1, 50, 88, 88) range -2.55..1.37
visual emb: (1, 512, 50)   forward 142.8 ms
encoder out: (1, 112, 251, 129)              ← 251，不是 250
audio_bottleneck: (1, 112, 251, 129)
video_bottleneck: (1, 512, 50)
separator.video_net out: (1, 512, 50)        ← stride=2 未生效
Falcon out: (1, 1, 32000)  full forward 660.9 ms
NeuroSCNetFusion( (v_proj): Conv1d(512, 112, kernel_size=(1,), stride=(1,), groups=16) )
fusion: video 50 -> audio 250 via nearest; fused: (1, 112, 250, 129)   ← 用假 T_a=250 做的对照
nearest map first 12: [0,0,0,0,0,1,1,1,1,1,2,2]  last: [49,49,49,49,49]
```

### G. 缺帧/超帧行为（`probe3.py`）

见 §E 表格：**没有任何 pad 或报错**，输出长度 = 输入长度。缺帧保护完全不在模型层。

### H. 数据集真实产出（`probe4.py`）

```
[H] tr: n_mix=20000 n_target=20000 sample_len_hist={32000: 20000} n_unique_mouth=14773
    cv: n_mix=5000  … {32000: 5000}  n_unique_mouth=476
    tt: n_mix=3000  … {32000: 3000}  n_unique_mouth=267
    target.json[0] = ['…/tt/target/6300370419826092098_00001_1.3679_….wav', '…/tt/mouths/6300370419826092098_00001.pt', 32000]

[I] len(ds)=3 | fps_len=50 | seg_len=32000
  item: mixture(32000,) float32 | source(32000,) | mouth(50, 88, 88) float64 | near=1 | 22.4 ms
    mouth min/max/mean/std = -2.552/2.677/-0.2102/0.9221
  item: … mouth(50, 88, 88) float64 … min/max/mean/std = -2.552/3.176/-1.4630/0.6844
  item: … mouth(50, 88, 88) float64 … min/max/mean/std = -2.552/1.964/-0.3922/1.1553
```

⇒ 训练侧单样本 mouth 读取+变换 **约 13–22 ms**（含 1.35 MB `torch.load`）。

### I2. `.pt` 大样本抽查（`probe7.py`）

```
[M] tr: {(50,96,96,3): 40}   cv: {(50,96,96,3): 40}   tt: {(50,96,96,3): 40}
[L] 原始 npz：n_npz=20392，shape=(50,96,96) uint8，min=0 max=246 mean=93.65（首样本）
    npz 帧数分布(前200): {50: 200}
```

### J. 计算量 / 时间感受野（`probe5.py`）

```
[J] T=50 H=96 -> out (1,512,50) | FLOPs=34.755 G (MACs=17.378 G)
    T=50 H=88 -> out (1,512,50) | FLOPs=31.616 G (MACs=15.808 G)
    T=25 H=88 -> out (1,512,25) | FLOPs=15.808 G (MACs=7.904 G)
    改动第0帧后受影响的输出帧: [0, 1, 2]
    改动第49帧后受影响输出帧: [47, 48, 49]
```

⇒ 时间感受野 ±2 帧（首/尾因边界只剩 3 帧），**非严格因果**。

### K. 视觉/音频计算占比（`probe6.py`）

> 本节输出是 `torch.utils.flop_counter` 口径：它不记 `aten::lstm`，所以音频只显示 5.38 GMAC。与本报告其他段落
> 相加时要用 ptflops 的 10.611 GMAC；两把尺子不可混算（2026-11-08 Lead 复核更正）。

```
[K] video FLOPs=31.62 G (MACs 15.81 G)
    audio+fuse+mask+decoder FLOPs=10.76 G (MACs 5.38 G)
    total FLOPs=42.38 G (MACs 21.19 G)   ←[复核注] 混尺，整窗应为 26.51 GMAC / 视觉 60.0%
    audio params=0.5377M  video params=11.187M
```

**2026-11-08 Lead 复核更正**：上面这段原先写"`ptflops` 报的 macs=10.611 G 实际是 FLOPs，README 应读作 10.6 GFLOPs"，该判断错误。ptflops 的 conv hook 每个乘加记 1，返回值就是 MACs；它与 `FlopCounterMode` 的差别不在 2 flop/MAC，而在**是否给 `aten::lstm` 记账**（5.382 + 5.23 ≈ 10.611 G，差值恰为 LSTM）。所以：
- 论文、README、`docs/STATUS.md:33` 的 10.6 G 是 ptflops 口径的 MACs，与论文同尺；
- 本节 5.38 GMAC / 10.76 GFLOPs 是 flop_counter 口径，单独看没问题，与 10.6 并列就错了；
- 视觉 15.897 G 与音频 10.611 G 同为 ptflops 口径，整窗 **26.51 GMAC、视觉 60.0%**；本节的 21.19 G / 74.6% 作废。

### L. `strict` 加载的 5 条失败路径（`probe7.py`）

```
[N] 正常加载: OK, trainable=0
    缺 1 key -> RuntimeError: … missing=1 unexpected=0 shape_mismatch=0 (model keys=137, checkpoint visual keys=136)
    filter 后键数: 137（混入 audio_model.encoder.conv.weight 后被过滤掉）
    非视觉 dict -> RuntimeError: … contains no visual-encoder keys (allowed prefixes: ('frontend3D.','trunk.'))
    文件不存在 -> FileNotFoundError
    缺 model_state_dict -> KeyError … must be a dict with a 'model_state_dict' entry
    shufflenet+resnet权重 -> RuntimeError: … missing=330 unexpected=130 shape_mismatch=6 (model keys=337, ckpt 137)
```

### M. 全前端参数量/形状矩阵（`probe9.py`）

```
[P] FRCNNVideoModel(resnet,prelu)      11,186,688  96^2->(1,512,50)  88^2->(1,512,50)
    FRCNNVideoModel(resnet,relu)       11,182,784  96^2->(1,512,50)  88^2->(1,512,50)
    FRCNNVideoModel(shufflenet,w1.0)     1,258,860  96^2->(1,1024,50) 88^2->(1,1024,50)
    LightVideomodel(w1.0)                1,258,860  96^2->(1,1024,50) 88^2->(1,1024,50)
    LightVideomodel(w0.5)                  347,048  96^2->(1,1024,50) 88^2->(1,1024,50)
    ResNetVideoModel(prelu)             11,186,688  96^2->(1,512,50)  88^2->(1,512,50)
    WujianVideoModel()                  11,318,465  (B,T,H,W)->(B,256,T)
    ShuffleNetV2(input=96,w1.0)          2,278,604  分类头，Falcon 不用
[Q] FRCNNVideoModel(pretrain=None): trainable=11,186,688 / 11,186,688   ← 不传 pretrain 不冻结
    FRCNNVideoModel(train) 后 BN 是否仍 eval: True
    WujianVideoModel(torch.zeros(2,50,96,96)) -> (2, 256, 50)
```

注：`ShuffleNetV2(input_size=96)` 的 `globalpool=AvgPool2d(3)`（`shufflenetv2.py:160`）在 88×88 输入下恰好也得到 3×3（22→11→6→3），所以 light/shufflenet 分支对 96/88 **都能跑**；这是 stride-2 卷积 `ceil` 行为的巧合，不是设计保证。

### N. mouth 亮度分布（60 文件，`probe8.py`）

```
[O] per-file mean: min=39.3 p25=74.2 med=92.0 p75=102.7 max=156.0  (整体均值 91.68)
    per-file std : min=22.8 med=38.9 max=76.4
    纯黑像素占比: med=0.066 max=0.176  (>0.2 的文件数 0/60)
    per-file max : med=210 max=255
    归一化后等价均值 (m/255-0.421)/0.165: med=-0.36  预期 LRW 中心=0.00
    单样本逐帧均值(每5帧): [67.5, 76.6, 82.0, 79.8, 80.0, 77.4, 67.5, 63.7, 67.7, 71.2]
```

⇒ LRS2-2Mix 的嘴部 crop 明显比 LRW 平均脸暗（0.36 vs 0.421），z-score 后分布在 −0.36 附近而非 0；**不要"修正"这组常数**，因为 FRCNN 是在 LRW 统计下预训练的，改常数反而破坏其特征分布。约 6.6% 纯黑像素来自 `cut_patch` 的贴边夹取，属正常。

---

## 5. 坑与反直觉点

1. **`128_512` 是名字骗人**。模型输入 88×88 灰度，输出 512 通道。任何"把视频 resize 到 128×512"的实现都会得到错误张量（§4-B）。`frcnn_videomodel.py:201-202` 的 `__main__` 也只是 `torch.randn(1,1,100,96,96)`，坐实 96×96 口径。
2. **`.pt` 是 3 通道复制**（`lrs2_2mix_to_falcon.py:224`），Dataset 再 `Grayscale(1)` 把它压回 1 通道（`avspeech_dataset_near.py:81`）。存储浪费 3×（1.35 MB vs 0.45 MB），14,773 个 tr 文件多占约 **13 GB**。这是为了让所有下游（含 `.npz`/Studio 单通道）共用一套形状约定。
3. **Dataset 输出 float64**（§4-B）。`Normalize.__call__` 做的是 numpy 除法（`transform.py:73`），`uint8/255.0` → float64；`collate` 后 `mouth` 是 double，**显存/带宽翻倍**，直到 `litmodule` 里 `mouth_in.type_as(mixtures)` 才降为 float32（`av_litmodule_nass_v2.py:87`）。单测/自定义训练脚本若绕过 litmodule 会直接 `RuntimeError: expected scalar type Double but found Float`。
4. **帧数不足没有保护**（`avspeech_dataset_near.py:169-174` 只切片）。数据集恰好全是 2 s 才不炸；换数据集（LRS3 原片 1–3 s 不等）会立刻在 `collate_with_near` 的 `torch.stack` 处崩。要复用得自己 pad/裁。
5. **写死的 16000 与 25**（`near:78,168`）。`sample_rate=8000` 时 mouth 起点算成 2 倍偏移，静默错位——这类 bug 不报错只掉点。
6. **`F.interpolate(..., mode="nearest")` 无 `align_corners` 问题，但有"半帧"语义**：视频第 k 帧覆盖音频 `[k*5.02, (k+1)*5.02)`，第 0 帧对应音频帧 0–4，**不是中心对齐**。仓里所有 `interpolate` 都是 `mode="nearest"`（`Falcon.py:260,322-327,895`），全库 **`align_corners` 零出现**，所以不存在 `align_corners=True/False` 的经典坑。
7. **`VideoBlock.stride=2` 是死参数**（`Falcon.py:900-976` 从未用 `self.stride`）。以为视频被降到 25 帧再融合的推断是错的：全窗口视频一直是 50 帧，50→251 全靠 fusion 插值。
8. **非因果的 80 ms**：FRCNN 帧级看 `t±2`。所有"causal"表述只能落在窗口级；Studio 的 `GUARD=1280`（80 ms，`space/clip.py:23`）正好是这个感受野，二者是同一设计意图的两面。
9. **RGB/BGR 有三套口径**：离线 `cv2.cvtColor(frame, COLOR_BGR2GRAY)` 作用在 **BGR** 帧上（`transform.py:65-66` + `crop_mouth_from_video.py:64` 注释 `## -- BGR`）；detectface 里又 `cv2.cvtColor(frame, COLOR_BGR2RGB)` 转给 PIL/MTCNN（`detectface.py:63`）；在线 JS 直接对 **sRGB** 做 BT.601 加权（`video-worker.js:92`）。因为最终都塌成单通道灰度，通道顺序只在"BGR2GRAY vs RGB2GRAY"这一步影响数值（差异约 R/B 权重互换），**但 `crop_mouth_from_video.py:180` 的 `sequence[...,::-1]` 分支（不转灰度时）会输出反序 RGB**。默认 `convert_gray=True` 掩盖了这个坑。
10. **0-255 vs 0-1 三次转换**：npz/`.pt` 是 0–255 uint8 → transform `Normalize(0,255)` 变 0–1 → `Normalize(.421,.165)` 变 z-score。Studio 的 `preprocess_mouth`（`falcon_runtime.py:138-139`）与 `space/clip.py:79-80` 手工重写了这两步，**三处要同步**，任一处改动就会静默偏移。实测三者算法等价（都是 `[:,4:92,4:92]/255` 再减均值除方差）。
11. **`infer_file.py` 的下限是 1 s 不是 2 s**：`extract()` 允许 `frames>=25`（`studio/infer_file.py:70-73`），此时音频只有 1 s、视频 25 帧，FRCNN 输出 `(1,512,25)`，fusion 插到 126 帧——**模型能吃，但分布外**（训练永远 50 帧）。文件模式与实时窗口模式的口径不同，对比结果时要标明。
12. **`FRCNNVideoModel.train()` 冻结 BN 但不冻结梯度**：`requires_grad=False` 是 `update_frcnn_parameter` 设的（`:196-197`），若有人 `pretrain=None` 建模型再手动 `train()`，BN 会保持 eval 但权重仍可训——"BN 冻结 ≠ 权重冻结"。
13. **`filter_frcnn_pretrained_dict` 的 `"tcn"` 子串过滤**（`:140,149`）作用在**全长 key** 上，如果将来某个视觉子模块名里带 tcn（如 `trunk.tcn_block.…`）会被静默丢掉，然后被 strict 检查抓成 `missing` 报错——报错信息不会告诉你"是被过滤器吃了"。
14. **JPEG q=0.88 是分布外噪声源**：训练数据是无损 uint8 npz，在线是 JPEG 压缩后的 96×96（`video-worker.js:96`）。块效应落在高频，而 FRCNN 第一层是 7×7 卷积，对块效应敏感度不低；这是 Studio 与论文指标不可直接对比的原因之一。

---

## 6. 与既有文档的冲突/更正

| # | 既有说法（位置） | 实际（证据） | 判定 |
|---|---|---|---|
| C1 | 「`FRCNNVideoModel.train()` 重写**没有 `return self`** → `FRCNNVideoModel(...).eval()` 返回 `None`」`docs/understanding/FINAL_PROJECT_UNDERSTANDING.md:168`、`:484`；`docs/verification/scripts/verify_video_04.py:26` 也在打印这条 | 当前 HEAD 的 `train()` 结尾有 `return self`（`frcnn_videomodel.py:113-125`），实测 `eval()` 返回 `FRCNNVideoModel` 实例（§4-D），`isinstance(m.eval(), FRCNNVideoModel) == True` | **旧说法已过时**（A2-2 已修）。`studio/falcon_runtime.py:217-218` 的注释「The original video model's train() override does not return self」同样是历史残留 |
| C2 | 「mouths `(F,96,96)` uint8/float？」`FINAL_PROJECT_UNDERSTANDING.md:297` | 实际 `(50,96,96,3)` **uint8**，第 4 维是灰度复制（`lrs2_2mix_to_falcon.py:224`，§4-A/C2） | **补全**：不是 3-D，也确定是 uint8 |
| C3 | 「mouths `[B,50,88,88] float32`」`FINAL_PROJECT_UNDERSTANDING.md:317` | Dataset 吐 **float64**（§4-B/I）；float32 是 litmodule `type_as` 之后的状态（`av_litmodule_nass_v2.py:87`） | **精度口径需分层写** |
| C4 | 「视觉 MACs 64²=7.77 G、88²=15.90 G、96²=17.48 G」`docs/reviews/round2/04-…:21,59,105,135` | 我用 `torch.utils.flop_counter`：88²/T=50 = **15.81 GMAC**、96² = 17.38 GMAC、88²/T=25 = 7.90 GMAC；ptflops 复算 88² = **15.897 GMAC**。方向与量级一致，差 0.5–1.7% | 已统一写「MACs」；2026-11-08 更正：ptflops 的 "macs" 就是 MACs，不是 FLOPs（原先这半句判断错了） |
| C5 | 「MACs 10.611 G / 0.538 M」（`docs/STATUS.md:33`，README 的 10.6 G） | 这是**音频侧 ptflops MACs**（含 LSTM、含 STFT/iSTFT、不含 FRCNN）；**含冻结视觉的整窗是 26.51 GMAC，视觉占 60.0%**（视觉 15.897 G 同为 ptflops 口径） | 已补一句：README 的 10.6 G 不代表推理总成本。原写的"42.38 GFLOPs / 21.19 GMAC / 74.6%"是 flop_counter 与 ptflops 混算，2026-11-08 作废 |
| C6 | 「Conv3d(5,7,7) 时间 kernel 5 + 无时间下采样，时间感受野 ±2 帧 = ±80 ms」`FINAL_PROJECT_UNDERSTANDING.md:186` | 实测一致（扰动传播范围 3 帧，首尾截断），且 T'=T 逐点验证（§4-E/J） | **一致，确认为真** |
| C7 | 「帧对齐 `mouth_start=floor(rand_start/16000*25)`，取 50 帧」`FINAL_PROJECT_UNDERSTANDING.md:321-322` | 一致（`avspeech_dataset_near.py:168-174`），但漏了「无 pad、16000 写死」两个风险 | **补风险** |
| C8 | 「96×96 → `[4:92,4:92]` 中心裁剪 → /255 → (x−0.421)/0.165，Studio 与训练 val 管线算法一致」`FINAL_PROJECT_UNDERSTANDING.md:379`；`falcon_runtime.py:131-140` | 实测一致（§4-B/F），且 `space/clip.py:77-80` 是第三份拷贝 | **一致，但三份手工拷贝是维护风险** |
| C9 | 「视频前端 11,186,688 / state_dict 11,196,308（+9,620 BN buffer）」`FINAL_PROJECT_UNDERSTANDING.md:137,144,530` | 逐位一致（§4-C/D） | **一致** |
| C10 | 「Studio 的 80 ms guard 恰好等于 FRCNN 的 ±80 ms 时间感受野」`:191` | guard 常量确为 1280 样本=80 ms（`space/clip.py:23`，Studio 同源），FRCNN 感受野 ±2 帧=80 ms | **一致** |

另注一处在范围内但属 studio 侧的旧注释：`studio/falcon_runtime.py:217-218` 仍按「旧 train() 无返回」写了兼容代码，属无害冗余（该处显式调用 `.eval()` 而不依赖返回值）。

---

## 7. GAP LIST（可被下一轮直接认领）

| ID | 问题 | 建议验证方式 |
|---|---|---|
| G1 | `frcnn_128_512` 里的 **128** 指什么？（候选：上游 LRW 的 128 帧训练窗？某版 128 维中间层？） | 查上游 `JusperLee/Falcon` git 历史 + HuggingFace 同名文件说明；`git log -S"128_512"` 在 fork 里追（当前只有 README/tools 引用） |
| G2 | `face_alignment` 的 68 点版本与 FRCNN 预训练用的 LRW landmark 版本是否同一（`face_alignment` v1.3 vs v1.4 差异） | 比对 `lrs2_rebuild/landmark/*.npz` 与 LRW 官方 landmark 的数值范围；或重跑一帧对比 `fa.get_landmarks` 版本 |
| G3 | 离线 `detectface.py` 的 **2 s 截断**（`video.fps*2`）是否对所有 clip 都成立；若源视频非 25 fps，npz 帧数会 ≠50 吗 | 统计 20392 个 npz 的 `fps` 元数据/帧数分布（我只抽查了 200 个全 50） |
| G4 | Studio JPEG q=0.88 对 FRCNN 特征的影响量化 | 同一 crop 存 PNG vs JPEG，比较 `video_model` 输出余弦相似度；CPU 即可 |
| G5 | `space` 整段模式与 Studio 实时模式在**同一段录音**上的输出是否逐位一致（文档声称一致） | 用 `space/tests/test_clip.py` 已有夹具，对比 `clip.separate` 与逐窗 `runtime.infer` 的 max|Δ| |
| G6 | 训练时 `RandomCrop(88)` 的随机性是否影响收敛（val 用 CenterCrop，train/val 输入分布不同） | 关掉 RandomCrop 做 A/B（需训练，超本轮范围） |
| G7 | `avspeech_dataset.py`（非 near 版，val/test 也用它的 `AVSpeechEvalDataset`）里 `source.mean(-1)` 无条件通道均值是否会吃掉单声道时间轴 | 已有文档列为 P0-3；本轮未复核该文件 193-194 行当前状态 |
| G8 | `update_wujian_parameter` 的 `k` 被 `split` 后 `if "resnet" in k` 失效，是否有历史 checkpoint 依赖它 | 查 `Experiments/checkpoint` 是否有 wujian 权重 |
| G9 | 音频 251 帧 ≠ 250，`F.interpolate` 到 251 后每个视频帧的实际覆盖是否在窗边界产生 1 帧错位 | 数值验证：构造线性 ramp 视频帧，看 fusion 输入的 251 个值分界点 |
| G10 | `WINDOW=32000` 但 STFT `center=True` 使边缘 1 帧是补零产生的对称延拓，视觉窗口没有对应延拓 | 比较首/末 5 个音频帧的 attention/贡献（需 hook 内部） |

---

## 8. 证据清单

### 读过的源码（`path:line`）

- `look2hear/videomodels/frcnn_videomodel.py:17-20,29-74,76-85,87-111,113-125,139-154,157-198`
- `look2hear/videomodels/resnet.py:6-9,12-26,29-70,73-140`
- `look2hear/videomodels/resnet1D.py:73-139`
- `look2hear/videomodels/resnet_videomodel.py:22-68,107-125`
- `look2hear/videomodels/light_videomodel.py:16-62,101-118`
- `look2hear/videomodels/shufflenetv2.py:109-173`
- `look2hear/videomodels/wujian_videomodel.py:99-139,142-182`
- `look2hear/videomodels/__init__.py:3-25`
- `look2hear/datas/transform.py:39-53,59-74,82-100,127-144,147-169`
- `look2hear/datas/avspeech_dataset_near.py:23-43,77-81,160-174,176-223`
- `look2hear/datas/avspeech_dataset.py:38-43,108-112,143-145,194-196`
- `look2hear/models/Falcon.py:11-92,241-267,900-976,978-1049,1284-1348,1480-1499`
- `look2hear/system/av_litmodule_nass_v2.py:74-90`
- `audio_train.py:308-310,358`；`audio_visual_test.py:310-335`；`evaluated_mac_params.py:75-96`
- `configs/LRS2_2Mix.yml:1-7,14-15`；`configs/Falcon.yml:2-8`
- `tools/repro/extract_video_backbone.py:1-35`
- `tools/repro/lrs2_2mix_to_falcon.py:199-230`
- `DataPreProcess/process_lrs23.py:8-32`
- `studio/web/mouth-tracker.js:1-53,72-108`
- `studio/web/video-worker.js:12,53-103`
- `studio/web/app.js:560-596,618-650`
- `studio/web/vendor/mediapipe/NOTICE.md:3-9`
- `studio/falcon_runtime.py:1-6,25-40,131-140,278-300`
- `studio/server.py:72-97`
- `studio/infer_file.py:47-82`
- `studio/README.md:101`
- `space/clip.py:22-29,37-57,77-98,105-148`
- `space/README.md:32`

### 数据侧脚本（不在 git，数据盘只读）

- `/root/data/lrs2_2mix/lrs2_rebuild/detectface.py:17-27,48,61-67,70-74,83-84,90-98,115-125`
- `/root/data/lrs2_2mix/lrs2_rebuild/crop_mouth_from_video.py:22-50,53-120,122-143,146-183`
- `/root/data/lrs2_2mix/lrs2_rebuild/transform.py:15-21,24-35,38-62,65-66`
- `/root/data/lrs2_2mix/conversion_stats.json`（`mouth_files` / `sample_lengths` / `validation`）

### 实测脚本与命令

远端脚本（均 `/tmp/w103/`，非仓库）：`probe1.py`（mouth 原始统计 + transform）、`probe2.py`（权重结构 + 加载/冻结 + 前向）、`probe3.py`（端到端形状 + fusion 对齐 + 缺帧）、`probe4.py`（混合 JSON + Dataset 真实样本）、`probe5.py`（MACs + 感受野）、`probe6.py`（视觉/音频计算占比）、`probe7.py`（npz 源 + .pt 抽查 + strict 5 路径）、`probe8.py`（60 文件亮度分布）、`probe9.py`（全前端参数矩阵 + 冻结矩阵）。

统一执行方式：

```bash
timeout 115 ssh -o BatchMode=yes dev_a800_2gpus \
  'cd /tmp/w103 && export CUDA_VISIBLE_DEVICES= && \
   PYTHONPATH=/root/code/github_repos/Falcon-fork \
   /root/miniforge3/envs/falcon/bin/python probeN.py 2>&1 | tail -40'
```

> 注意：**不要**把脚本放在 `/tmp/*.py` 根目录直接跑——远端 `/tmp/profile.py` 是别的任务留下的同名模块，会遮挡标准库 `profile`，导致 `import torchvision` 崩在 `cProfile → import profile`（本轮真实踩过，见 §4 开始前的失败尝试）。要放子目录（如 `/tmp/w103/`）。
