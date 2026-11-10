# 160 ms 音视频增量流式模型

这套实验路径将混合音频和目标嘴部视频分成 160 ms 的连续片段，使用每个会话自己的历史状态，逐块输出等长音频。每次只计算新到的数据，输出直接覆盖当前块；保护区为 0。模型、训练和推理入口与原版 Falcon 并行保留。

本路径使用新建的因果结构和独立检查点。原 Falcon 的 `falcon.pt` 对应另一套网络结构，不能直接装入本模型。随机检查点和少量训练步骤用于验证运行链路，语音提取效果需要完整训练后评测。

## 300 ms 验收口径

按用户指定口径，以 **160 ms 分块等待 + 同机 `StreamingSession.push` 处理时间**衡量。8 ms 子帧等待包含在 160 ms 分块中。浏览器、网络和物理设备耗时不纳入这一口径。

2026-11-11 修复了频谱掩码合成的循环混叠，并将模型检查点升级为 `falcon-streaming-v2`。2026-11-10 的 CPU、A800 和 WebSocket 性能记录对应 v1，作为历史记录保留；v2 的速度由该版本重新测量，不将 v1 数字用于 v2 验收。

## 1. 输入、输出和时间轴

| 项目 | 格式 |
| --- | --- |
| 采样率 | 16,000 Hz，单声道 |
| 一块音频 | 2,560 个 `float32` 采样点，160 ms |
| 嘴部视频 | 25 帧/秒，一块 4 帧 |
| 嘴部原始图像 | 每帧 96×96，灰度或 RGB，`uint8`；预处理后为归一化的 88×88 灰度图 |
| 输出 | 每块 2,560 个 `float32` 采样点，与输入块采样序号对齐 |
| 块间状态 | 前一段音频、视觉历史帧和音频卷积缓存；每个会话各自保存 |
| 输出保护区 | 0；每个输出块都完整交付 |

```text
音频：  [0～160 ms] [160～320 ms] [320～480 ms] ...
视频：  0  40  80 120 | 160 200 240 280 | 320 360 400 440 ms
输出：  [0～160 ms] [160～320 ms] [320～480 ms] ...
状态：  初始空状态 ──→ 更新状态 ──→ 更新状态 ──→ ...
```

视觉帧以**音频时钟上的采样点**标记采集时间，或比采集时间更晚的保守可用时间。第 `i` 张图服务于对应的 40 ms 区间，标记时间不晚于该区间起点。第 0 块的四个区间起点是 `0, 640, 1280, 1920` 采样点。选择图像时，从起点及之前已经采集的画面中取最新可用帧；还需要计入摄像头实际到达、跟踪和时钟同步的时间。`StreamingSession` 检查调用者提供的时间戳，摄像头采集、检测、跟踪和跨设备时钟换算由接入端完成。浏览器客户端使用嘴图处理完成时的音频时钟作为保守可用时间，而不是摄像头的真实拍摄时刻。

启动新会话时，可先取得目标人的有效嘴部画面，再为音视频建立共同的零点；画面实际晚于区间起点时，应如实保留时间戳并等待下一次可用对齐。

训练数据索引中的嘴部帧按照 25 fps 等间隔时间轴使用。训练裁剪起点固定在 640 个音频采样点的倍数，避免随机裁剪产生不足一帧的视频偏移；源数据本身的音视频同步关系以其数据准备结果为准。

## 2. 模型如何增量运行

代码位于 [`look2hear/models/FalconStreaming.py`](../look2hear/models/FalconStreaming.py)。每次调用 `forward_chunk(audio, video, state)` 接收一个完整块并返回 `(output, new_state)`；下一块传入返回的状态。`forward(audio, video)` 在一个训练序列内按顺序执行同一接口，梯度贯穿序列中的块。

音频模块以 128 个采样点为一个处理步长，即 **8 ms 子帧**。它保留上一子帧的 128 个采样点，与当前 128 点合成 256 点频域分析窗；因果卷积分别保留自己的历史。v2 将每个频谱掩码变换为时域响应，保留前 **129 个因果抽头**、后面补零，再通过 overlap-save 滤波；这样 128 个历史采样足以覆盖整个因果滤波器，避免 v1 的 256 点循环卷积折叠。单位掩码下重建样本与输入采样序号对齐。每个 8 ms 子帧的输出需要收齐该子帧；模型按子帧因果，公开接口一次接收和返回 160 ms。

视觉前端沿用 FRCNN 的空间主干与 5 帧时间卷积结构，时间维使用当前帧及缓存的 4 帧历史。每次只处理本块的新图像，视觉投影按 40 ms 分配到 5 个 8 ms 音频步。模块中的 BatchNorm 使用固定统计量，避免训练时使用未来帧或其他样本的统计量。音频特征与视觉条件相加后进入因果残差块，得到频域掩码和当前块的输出。

会话开始或目标说话人改变时重置状态。训练时一个样本内部的块连续，新的样本从空状态开始；跨样本复用旧状态会混入另一条录音的历史。

## 3. 环境和运行目录

下面的命令使用服务器已有环境，模型产物写入新的临时目录。`configs/FalconStreaming.yml` 是尺寸和默认训练参数的参考清单；这些命令由命令行参数控制，不读取该文件。

```bash
cd /root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
RUN_DIR="$(mktemp -d /root/tmp/falcon-streaming-example-XXXXXX)"
echo "$RUN_DIR"
```

后续命令沿用同一 shell 中的 `PY` 和 `RUN_DIR`。将 `cuda:0` 换成 `cpu` 可在 CPU 运行。

## 4. 创建随机检查点

```bash
"$PY" -m tools.streaming.init \
  --output "$RUN_DIR/random.pt" \
  --seed 17
```

检查点包含模型参数、随机种子、`steps=0`、`training_status=random_initialization` 和 `falcon-streaming-v2` 格式标记。模型状态另有 `_streaming_format_version=2` 标量标记；载入时同时检查格式与标记。v1 检查点使用不同的合成语义，不能直接当作 v2 恢复；请使用本命令重新初始化或重新训练。检查点文件使用独占创建方式；要重跑例子，换一个输出路径或重新创建运行目录。

## 5. 用真实索引训练

训练入口读取 Falcon 数据目录下的 `mix.json` 和 `target.json`。前者给出混音路径与长度；后者给出目标语音、嘴部 `.pt` 和长度。入口逐行检查混音与目标文件名，长度不足所选片段的样本不进入训练列表。音频文件应为 16 kHz 单声道，嘴部应为 25 fps 的 `uint8` 96×96 灰度或 RGB 帧。

以下示例在已有 LRS2-2Mix 训练集上从随机模型运行两步；每个样本由 2 个 160 ms 块组成。该命令用于检查真实数据读取和反向传播，正式训练可按显存、语音长度和实验计划调整 `--chunks`、`--steps`、`--batch-size`。

```bash
"$PY" -m tools.streaming.train \
  --index-dir /root/data/falcon_lrs2_2mix/tr \
  --chunks 2 --steps 2 --batch-size 1 \
  --device cuda:0 \
  --output "$RUN_DIR/real-smoke.pt"
```

损失函数为整段输出与目标语音的负 SI-SNR。一次模型 `forward` 会按块顺序执行，音频和视觉网络的参数都参与训练；目标波形与输出使用相同采样序号。优化器为 Adam，默认学习率 `0.0003`，梯度范数裁剪上限为 5。`--synthetic` 可以生成随机样本做接口烟测，检查点标记为 `synthetic_smoke`；真实数据优化标记为 `real_data_optimization`，标记本身不代表训练已收敛或分离质量达标。

从已有检查点继续增加一步：

```bash
"$PY" -m tools.streaming.train \
  --index-dir /root/data/falcon_lrs2_2mix/tr \
  --chunks 2 --steps 1 --batch-size 1 \
  --device cuda:0 \
  --resume "$RUN_DIR/real-smoke.pt" \
  --output "$RUN_DIR/real-resume.pt"
```

续训恢复模型、优化器及累计步数；省略 `--lr` 时沿用检查点学习率，显式提供 `--lr 0.0001` 等参数时覆盖已恢复优化器的学习率。启动日志打印实际学习率。当前入口没有保存随机数状态和数据迭代位置，所以不提供逐样本完全一致的断点续训。两次训练均使用新的输出路径；目标路径已存在时，训练开始前报错。

## 6. 用真实样本回放分块推理

下面从现有测试索引取第一条混音及对应嘴部文件，用随机权重演示完整 160 ms 回放。音频和视频应来自同一时间轴。

```bash
MIX="$("$PY" -c 'import json; print(json.load(open("/root/data/falcon_lrs2_2mix/tt/mix.json"))[0][0])')"
MOUTH="$("$PY" -c 'import json; print(json.load(open("/root/data/falcon_lrs2_2mix/tt/target.json"))[0][1])')"
"$PY" -m tools.streaming.infer \
  --checkpoint "$RUN_DIR/random.pt" \
  --audio "$MIX" --mouth "$MOUTH" \
  --output "$RUN_DIR/output.wav" --device cuda:0
```

脚本按顺序传递状态。最后不足 160 ms 时，音频补零、额外的嘴部位置重复最后一帧，运行完成后裁掉补齐部分；输出 WAV 的样本数与原始音频相同。每块的计时包括该次模型调用和同步，不包括输入读取、数据转入设备、摄像头采集、网络或播放；第一次调用包含预热开销。

## 7. 在 Python 中接入实时会话

`StreamingSession` 封装状态、输入形状和时间戳检查。以下例子用真实索引中的第一块演示接口，索引帧时间按 25 fps 合成，实际摄像头接入时由采集端提供映射到音频时钟的真实时间戳。

```bash
"$PY" - "$RUN_DIR/random.pt" <<'PY'
import json
import sys
import torch
from tools.streaming.common import load_mono
from tools.streaming.session import StreamingSession

root = '/root/data/falcon_lrs2_2mix/tt'
mix = json.load(open(root + '/mix.json'))[0][0]
mouth = json.load(open(root + '/target.json'))[0][1]
audio = load_mono(mix, start=0, frames=2560).numpy()
frames = torch.load(mouth, map_location='cpu', weights_only=True)[:4]
session = StreamingSession(sys.argv[1], device='cuda:0')
output = session.push(audio, frames, sequence=0, start_sample=0,
                      video_samples=[0, 640, 1280, 1920])
print('output:', output.shape, 'samples; status:', session.training_status)
session.reset()
PY
```

完整块依次使用 `sequence=1, 2, ...`、`start_sample=2560, 5120, ...`。`video_samples` 是 4 个图像采集时间或保守可用时间对应的音频采样序号，检查非递减、不得晚于相应区间起点；覆盖有效音频的图像还检查最大帧龄，默认上限为 2,560 个采样点，可在构造会话时设置 `max_frame_age_samples`。顺序、形状或时间戳错误的请求会抛出异常，状态保持在上一块。

最后不足 2,560 点时，将有效音频放在块首、其余位置补零，向 `push` 传入有效点数 `valid_samples`（1～2,559）。接口仍返回 2,560 点，只使用前 `valid_samples` 点。音频后缀必须为零；4 张图的时间仍须非递减且不得来自各自区间的未来，但最大帧龄只针对实际覆盖有效音频的 40 ms 区间检查。部分块之后会话进入结束状态，继续接收新流前须调用 `reset()` 或创建新会话；`valid_samples` 不传时默认为完整的 2,560 点。

多个会话可以通过 `StreamingSession.from_model(model)` 共享处于 `eval()` 状态的模型实例，同时各自持有状态；运行期间模型权重、设备和模式保持不变。同一个会话的调用由接入程序按顺序串行执行，共享模型的并发调度由服务管理。

## 8. 启动 WebSocket 模型服务

[服务入口](../tools/streaming/server.py)接收已对齐的音频块与嘴部画面，返回当前块的音频。服务绑定本机 `127.0.0.1`，客户端负责音频与摄像头采集、嘴部定位、时间戳映射和播放。启动命令如下：

```bash
"$PY" -m tools.streaming.server \
  --checkpoint "$RUN_DIR/random.pt" --device cuda:0 --port 8765
```

服务运行后，`GET http://127.0.0.1:8765/status` 返回模型和音频格式；WebSocket 地址为 `ws://127.0.0.1:8765/ws`。连接建立后先收到 `ready` 消息。每个连接独立保存流状态；最多 8 个连接共享同一模型，模型调用由服务串行调度，因此多会话的排队时间需要单独测量。

每个 `chunk` 请求包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `type` | 固定为 `chunk` |
| `sequence` | 从 0 起依次递增 |
| `start_sample` | 当前块在会话音频时钟中的首采样序号，完整块依次为 0、2560、5120…… |
| `valid_samples` | 可选，默认 2,560；末尾不足一块时为 1～2,559，后续音频采样必须补零 |
| `video_samples` | 4 个整数，表示嘴部图像在同一音频时钟上的采集或保守可用采样序号 |
| `audio_f32le` | 2,560 个小端 float32 音频采样的字节，使用 Base64 编码 |
| `mouth_gray_u8` | 4 张 96×96 灰度 `uint8` 图片按时间顺序连接，使用 Base64 编码 |
| `mouth_jpeg` | 4 张 96×96 JPEG 图片的 Base64 字符串数组，每张压缩图最多 16,000 字节 |

嘴部图像使用 `mouth_gray_u8` 或 `mouth_jpeg` 中的一种；服务将 JPEG 解码为灰度图。整条 JSON 消息上限为 100,000 字节。浏览器客户端使用 JPEG，压缩可减少传输数据；JPEG 是有损编码，对训练后模型分离质量的影响需要另行测量。

响应的 `type` 为 `result`，带有 `sequence`、`start_sample`、`end_sample`、`valid_samples`、`audio_f32le`、`guard_samples=0` 和 `timing_ms`。`end_sample = start_sample + valid_samples`；响应的音频始终有 2,560 点，客户端按 `valid_samples` 截取有效输出。部分末块后，继续使用该连接前须发送 `{"type":"reset"}` 或关闭连接；重置后序号和采样序号均从 0 重新开始。计时字段包含服务端解码、等待模型锁、会话处理和编码。输入不符合接口约定时，服务返回错误消息和预期的下一序号。

服务已挂载浏览器客户端，访问 `http://127.0.0.1:8765/` 会跳转至 `/stream-ui/`。页面提供摄像头与麦克风采集、MediaPipe 嘴部跟踪、模型输出监听和状态显示：点击“开始采集”并允许设备权限；勾选“播放模型输出”可监听处理结果。为避免扬声器声音回灌麦克风，监听时使用耳机。页面输出由当前加载的检查点决定，随机初始化状态适合观察链路。

服务默认仅监听其运行机器的本地地址。如果服务运行在远程服务器，可以在本机另开终端执行 `ssh -L 8765:127.0.0.1:8765 dev_a800_2gpus`，再在本机浏览器访问上述地址；音视频从浏览器所在设备采集。使用复用连接时，结束后可运行 `ssh -O cancel -L 8765:127.0.0.1:8765 dev_a800_2gpus` 关闭该端口转发。浏览器的时钟、缓冲和协议自动测试已执行，真实摄像头、麦克风与扬声器组成的完整设备链路仍待实测。服务接受本机无 Origin 的客户端和同源请求。启动时服务先用 3 个合成块预热，再丢弃预热状态；启动耗时随设备和模型版本变化。点击停止时，采集端以 31 个源采样的零输入冲刷下采样滤波器，再发送缓存中的有效音频；客户端将最后不足 160 ms 的块补零，传入 `valid_samples` 并在输出时裁回有效长度，最后等待已排队音频播放；尾块中不覆盖有效音频的画面位置复用已有画面。采集 worklet 检测到音频 render quantum 时间轴断裂时，会通知客户端停止已启动的会话。

服务保持运行时，可以在另一个终端执行下列命令，按 160 ms 周期发送 35 个合成块，结果保存在新的临时目录：

```bash
cd /root/code/github_repos/Falcon-fork
PY=/root/miniforge3/envs/falcon/bin/python
export PYTHONPATH="$PWD"
RUN_DIR="$(mktemp -d /root/tmp/falcon-streaming-ws-XXXXXX)"
echo "$RUN_DIR"
"$PY" -m tools.streaming.bench_ws \
  --url ws://127.0.0.1:8765/ws --warmup 0 --chunks 35 \
  --output "$RUN_DIR/ws-paced.json"
```

[2026-11-10 的 v1 版 35 块记录](verification/streaming-20261110-ws-paced.json)使用随机权重、A800、同机 TCP 回环和合成数据，首块也进入统计：客户端从编码到收到并解析响应的 P50 为 11.20 ms、P95 为 14.23 ms，最大 30.19 ms；服务端解码至编码完成的 P50 为 5.95 ms、P95 为 8.83 ms。35 块中没有一次客户端往返超过 160 ms。这一测试没有包括真实摄像头和麦克风采集、嘴部跟踪、跨机网络或扬声器播放。

## 9. 已完成的检查及下一步

2026-11-10 使用 v1 版、A800 和项目 `falcon` 环境，真实 LRS2 训练索引上的 2 个连续块、batch 1 完成两次优化，loss 分别为 -1.19757、-1.54127；从该检查点恢复后完成第 3 步，优化器状态已加载。数值用于确认训练链路，模型效果需使用训练后的检查点和目标测试集评估。

同日使用 v1 随机检查点回放一条 2 秒真实音视频：13 个块、末块补 1,280 个采样点，输出裁为原始 32,000 点；模型调用计时中位数 3.761 ms、P90 4.496 ms。该次数值是单条离线回放，不含真实采集、视频追踪、网络、排队和播放。

2026-11-11 使用 v2 在 CPU 上对真实 LRS2 样本完成两步短训和一步续训，训练回归覆盖学习率覆盖、已有输出拒绝及检查点版本验证。具体命令、结果和训练范围见 [v2 训练复核记录](verification/streaming-20261111-training-review.md)。

验证脚本见 [`test_falcon_streaming_contract.py`](../tools/repro/tests/test_falcon_streaming_contract.py)，覆盖逐块与整段顺序调用、8 ms 子帧和视频帧边界的未来扰动、重置、输出对齐、梯度以及会话时钟检查；[`test_falcon_streaming_data.py`](../tools/repro/tests/test_falcon_streaming_data.py)覆盖索引配对、短数据过滤和音视频切片对齐。本次 300 ms 验收以 160 ms 分块加同机会话处理为口径；真实目标语音质量、实际声学环境表现，以及设备采集到播放的延迟分布可在后续训练与测试中分别测量。

### 历史 v1：连续 500 块的会话性能测试

[历史 v1 性能记录](verification/streaming-20261110-session-benchmark.json)使用 A800、随机初始化检查点、单会话和已准备好的合成输入。预热 10 块后连续处理 500 块，即 80 秒音频、1,280,000 个输出样本；`session.push` 的耗时包含 CPU 图像裁剪归一化、数据传入 GPU、模型和输出回传：

| 指标 | 耗时 |
| --- | ---: |
| P50 | 4.944 ms/块 |
| P95 | 5.693 ms/块 |
| P99 | 5.981 ms/块 |
| 最大值 | 6.929 ms/块 |

采集期间 PyTorch 已分配显存首尾均为 48,343,040 字节。测试连续调用模型，并未按 160 ms 的真实时间节奏送入；按本次约定，将固定 160 ms 分块时长加到各项处理耗时用于模型侧验收。数据不包含采集、嘴部跟踪、网络和播放。以下命令复现同样的测试范围，具体耗时随设备和负载变化：

```bash
"$PY" -m tools.streaming.bench_session \
  --checkpoint "$RUN_DIR/random.pt" --device cuda:0 \
  --warmup 10 --chunks 500 --output "$RUN_DIR/session-benchmark.json"
```

### 历史 v1：CPU 单路分块速度

2026-11-10 在服务器 Intel Xeon Platinum 8336C 上以 v1 随机初始化检查点顺序测试 1、4、8 线程；每组预热 10 块并连续处理 100 块，输入为预先准备的合成音视频。下表为 `session.push` 的处理耗时，包含嘴图预处理和模型计算；160 ms 分块时间可另加。原始数据见 [CPU 性能记录](verification/streaming-20261110-cpu-benchmark.json)。

| CPU 线程 | P50 | P99 | 最大值 |
| --- | ---: | ---: | ---: |
| 1 | 68.181 ms | 77.149 ms | 77.171 ms |
| 4 | 19.383 ms | 21.678 ms | 22.004 ms |
| 8 | 14.286 ms | 15.655 ms | 15.756 ms |

线程数同时通过 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 设置；这些数据对应本服务器和该次单路、非实时节奏的测量。

### 2026-11-11 v2 CPU 复核

新版 129 tap 因果重建经过真实音视频扰动、训练及 1/4/8 线程 CPU 测试。4 线程按 160 ms 节奏处理 500 块时，调度起点到完成的 P99 为 30.166 ms；按约定加 160 ms 凑块时间为 190.166 ms。分位数、各线程数据、因果测试和可复现证据见 [本轮 CPU 复核](verification/streaming-20261111-review.md)。
