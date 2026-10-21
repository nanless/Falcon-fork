"""R1 verification for docs/understanding/04: evaluated_mac_params.py methodology and README numbers.
Replicates the script's ptflops call on CPU (evaluated_mac_params.py uses .cuda(); MACs are the same).
No pretrain weights / no data needed. Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_macs_04.py
"""
import copy, sys, warnings
warnings.filterwarnings("ignore")
import torch, yaml
sys.path.insert(0, ".")
import look2hear.models, look2hear.videomodels
from ptflops import get_model_complexity_info

cfg = yaml.safe_load(open("configs/Falcon.yml"))
video_cfg = copy.deepcopy(cfg["videonet"]["videonet_config"]); video_cfg.pop("pretrain", None)
video = getattr(look2hear.videomodels, "FRCNNVideoModel")(**video_cfg)
print("=" * 78)
print("[1] 参数量（与 README 表格 0.5M / 11.72M 对账）")
audio = getattr(look2hear.models, "FalconSpeakerContrastive")(
    sample_rate=16000, **copy.deepcopy(cfg["audionet"]["audionet_config"]))
n_audio = sum(p.numel() for p in audio.parameters())
n_video = sum(p.numel() for p in video.parameters())
print(f"  FalconSpeakerContrastive (audio) = {n_audio:,} = {n_audio/1e6:.3f} M  (全部 requires_grad)")
print(f"  FRCNNVideoModel                  = {n_video:,} = {n_video/1e6:.3f} M  (加载 pretrain 后冻结)")
print(f"  audio + video                    = {n_audio+n_video:,} = {(n_audio+n_video)/1e6:.2f} M  -> README '11.72 M'")

print("\n[2] 复刻 evaluated_mac_params.py 的 ptflops 调用（input_constructor 字典）")
video.eval()
with torch.no_grad():
    vemb = video(torch.randn(1, 1, 50, 64, 64))      # 脚本第 81 行：50 帧 64x64
print(f"  video_model(randn(1,1,50,64,64)) -> {tuple(vemb.shape)}  (脚本第 82 行)")
a = torch.randn(1, 32000)                             # 脚本第 80 行：2 s @16 kHz
def input_func(inp):
    return {"audio_mixture": inp[0], "mouth_embedding": inp[1]}
audio.eval()
macs, params = get_model_complexity_info(
    audio, (a, vemb), input_constructor=input_func, as_strings=False,
    print_per_layer_stat=False, verbose=False)
print(f"  ptflops(audio_model, input=(a,vemb)) -> MACs = {macs/1e9:.3f} G, params = {params/1e6:.3f} M")
print(f"  -> 与 README 'Params 0.5 M / MACs 10.6 G' 对齐：{(macs/1e9):.2f} G vs 10.6 G（README 取一位小数）")

print("\n[3] 视频编码器 MACs（脚本没有统计它；这里量化被排除的部分）")
for shape in ((1, 50, 64, 64), (1, 50, 88, 88), (1, 50, 96, 96)):   # (C,T,H,W)，ptflops 自动补 batch
    vm, vp = get_model_complexity_info(video, shape, as_strings=False,
                                       print_per_layer_stat=False, verbose=False)
    print(f"  FRCNNVideoModel {shape}: MACs={vm/1e9:.2f} G, params={vp/1e6:.2f} M")
print("  -> evaluated_mac_params.py 只对 audio_model 调 ptflops（第 88 行），故 10.6 G 不含视觉编码器；")
print("     且脚本喂给视觉编码器的是 64x64（第 81 行），而评测/训练实际是 88x88。")

print("\n[4] MACs 数量级复核（手算 audio_bottleneck 单层 vs ptflops 全模型）")
x = torch.randn(1, 112, 251, 129)
conv = audio.audio_bottleneck
with torch.no_grad():
    y = conv(x)
print(f"  audio_bottleneck: in {tuple(x.shape)} -> out {tuple(y.shape)}; 该层 MACs ≈ "
      f"{x.shape[1]*x.shape[2]*x.shape[3]*y.shape[1]*1/1e9:.2f} G（Conv2d 1x1）")
print("  说明：ptflops 的 10.6 G 是 '2 s 音频 + 50 帧视觉特征' 全模型一次前向的乘加数，",
      "数量级与帧数/通道数开方一致；不能与视觉编码器 MACs 直接相加得到论文口径。")
print("\nDONE")
