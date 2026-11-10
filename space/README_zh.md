# Falcon Studio · Hugging Face Space

[English](README.md) | **中文**

论文：[arXiv:2610.32749](https://arxiv.org/abs/2610.32749)（NeurIPS 2026）

运行在 ZeroGPU Space 上的 [Falcon Studio](https://github.com/JusperLee/Falcon/tree/main/studio) 界面。用摄像头和麦克风录一段，Falcon 会提取画面中嘴部对应的说话者的声音。

ZeroGPU 每次调用临时分配 GPU，并计入每日配额，不适合实时工作台每 0.5 秒调用一次的循环。所以这个 Space 先录制、再分离。实时分离、耳机监听和会议接入请在本地运行 Falcon Studio。

## 工作方式

1. 浏览器以 16 kHz 采集音频，并用 MediaPipe Face Landmarker 跟踪嘴部。完整摄像头画面不上传；录制结束后会发送嘴部裁剪图像。
2. 每段录音最长 60 秒。停止后，音频和 96×96 灰度嘴部画面（各带采集时间）一起发送到 Space。开头和结尾没有嘴部画面的部分会被裁掉。
3. 服务端按 2 秒窗口和 0.5 秒步长切分，逐个位置取最近的嘴部画面，裁出距窗尾 80 ms 的 0.5 秒输出，再以 20 ms 淡化拼接。一次请求内部默认最多 32 个窗口一批；录音前后补零，画面边界沿用首帧或末帧。窗口几何接近 Studio，但当前默认模型的批次维度实现会使一个窗口受前序窗口视频影响，同一条真实音视频在 batch=1 与 batch=4 时输出有差异；整体质量影响有待更多样本测量。
4. 页面显示 GPU 耗时，可以试听、与原始混合对比，并导出两者的 WAV。

跟踪基于位置而非身份。人脸暂时丢失时，页面可能沿用上一次取景；其他人进入该区域时应重新确认目标。Space 在画面间隔较长时仍选取最近帧；音画同步需要结合采集时间检查。源码仓库中的 `docs/understanding/20261110-evaluation-and-apps.md` 解释窗口、实验及风险，`docs/STATUS.md` 汇总复现状态。

Space 启动时加载两个模型：[`JusperLee/Falcon`](https://huggingface.co/JusperLee/Falcon)（论文模型）和 [`JusperLee/Falcon-Big`](https://huggingface.co/JusperLee/Falcon-Big)。

## 隐私与配额

音频和嘴部画面只用于本次分离，不写入磁盘；结果只保存在你的浏览器中。从 Space 页面打开时，分离计入你自己的 ZeroGPU 配额。未登录访客的配额最少，可能被拒绝（"ZeroGPU runs limit"）；登录 Hugging Face 后配额更多、排队优先级更高。代码注释记录过不同长度的 GPU 耗时示例，但仓库没有随该说明保存逐次原始计时；实际等待还包括上传和排队，取决于设备与负载。

## 开发

Space 运行在 ZeroGPU 运行时（`sdk_version: 6.29.1`）：**Python 3.12 + `torch==2.13.0`**，
见 `space/requirements.txt`。这与本地 Studio 环境**不是同一个环境**：仓库的 `falcon`
conda 环境是 Python 3.11 + `torch 2.8`（见 `AGENTS.md` §2）。不要把
`space/requirements.txt` 装进 `falcon`；`space/tests/test_static.py` 与
`space/tests/check_i18n.mjs` 只需要 Python 和 Node，`space/tests/test_clip.py` 在
`falcon` 环境运行，HTTP/SSE 测试才需要这里的 Space 依赖。

在 GitHub 仓库根目录、Python 3.12 下：

```bash
pip install -r space/requirements.txt gradio==6.29.1
python space/build.py --out /tmp/falcon-space    # 复制 studio/ 的运行时和网页文件，再加上 space/
(cd /tmp/falcon-space && python app.py)          # 停止后返回仓库根目录
python -m pytest space/tests                     # 在仓库根目录运行
# 发布会删除目标 Space 中不属于本次构建的已有文件，请确认后执行：
python space/build.py --push JusperLee/Falcon-Studio
```

`space/` 只放与 Falcon Studio 不同的部分：`app.py`（Gradio server 模式、`@spaces.GPU`）、`clip.py`（窗口划分与请求校验）、`web/index.html` 和 `web/app.js`（先录制、再分离）以及 `web/i18n-space.js`。其余文件由 `build.py` 从 `studio/` 复制。
