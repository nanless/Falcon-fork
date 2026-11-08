"""Falcon audio-side MACs / params on one 2 s window (ptflops).

什么被统计（口径，务必先读）
--------------------------
本脚本用 ``ptflops.get_model_complexity_info`` 统计 **audio model（FalconSpeakerContrastive）**
一次前向的乘加数；被统计的输入是 ``audio_mixture=(1, 32000)`` + ``mouth_embedding=(1, 512, 50)``：

* **ptflops 0.7.5 的 conv hook 每个乘加只记 1**：``conv_per_position_flops =
  prod(kernel) * in_channels * filters_per_channel``（``ptflops/pytorch_ops.py``），
  返回变量名叫 ``flops``，**实际计的是 MACs**；要报 FLOPs 请自行 ×2。
* **含 LSTM hook**：ptflops 的 ``rnn_flops_counter_hook`` 对 ``nn.LSTM/RNN/GRU`` 按
  ``num_layers × seq_length × batch × (双向?2:1)`` 计入 ``w_ih/w_hh`` 矩阵乘与激活项，
  所以 10.6 G 里**已经包含** TFRB 的两级 LSTM，不是只算了 conv/linear。
* **不含冻结的 FRCNN 视觉编码器**：ptflops 只对 ``audio_model`` 调用；FRCNN 在本脚本里
  只负责把 ``(1, 1, 50, 64, 64)`` 的嘴部裁剪压成 ``(1, 512, 50)`` 的 ``mouth_embedding``。

三个参考量（同一台机器、同一 ptflops 复算；证据 `docs/verification/runs/r3-macs.log`）
--------------------------------------------------------------------------------------
======================  ==========  ====================================================
量                      GMAC        说明
======================  ==========  ====================================================
音频模型（本脚本打印）  **10.611**  2 s 音频 + 50 帧视觉特征，一次前向；README 的 10.6 G 即此值
冻结 FRCNN（不计入）    **15.897**  50×88×88（训练/评测的真实裁剪）；64²=7.77、96²=17.48
整窗端到端              **26.51**   10.611 + 15.897
======================  ==========  ====================================================

⚠ 三者是**不同口径**，不可混用：论文/README 的 "10.6 G" 只是音频部分；本脚本喂给 FRCNN 的是
**64×64**（只为造一个形状正确的 embedding，64² 的 FRCNN 成本是 7.77 G），而训练/评测实际是
**88×88**（15.897 G）。所以「整窗 26.51 G」是本脚本**没有**打印、但可以按 10.611+15.897 得到的量。

设备
----
本脚本默认需要 **1 张 CUDA GPU**（``--device auto`` / ``--device cuda``）：仓库文档里的数值是在
CUDA 构建上测的，纯 CPU 环境不保证与文档一致，因此检测不到 GPU 时**直接以退出码 2 结束**并给出提示。
显式传 ``--device cpu`` 可以强制 CPU 运行（同一算子图，理论数值不变），但该路径**未与文档基准复核**，
脚本会额外打印一条 WARNING；见 ``--help``。

配置文件固定读取**脚本同目录下的** ``configs/Falcon.yml``（不再依赖 cwd）。视觉 backbone 的
``pretrain`` 若在仓库内缺失（``configs/Falcon.yml`` 写的是相对路径 ``pretrain_zoo/...``），脚本会退回
一个形状相同的零 ``mouth_embedding`` 并打印 WARNING —— 因为冻结 FRCNN 本来就不计入 MACs。
"""

import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import contextlib
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
import look2hear.losses
import look2hear.models
import look2hear.videomodels
import yaml
from rich import print
from ptflops import get_model_complexity_info

# # Set PyTorch thread count to 1
# torch.set_num_threads(1)

#: paper/README reference numbers, all measured with this same ptflops recipe
REF_AUDIO_GMAC = 10.611          # audio-only, counted below
REF_FRCNN_GMAC_88 = 15.897       # frozen FRCNN @ 50x88x88, NOT counted below
REF_FRCNN_GMAC_64 = 7.77         # frozen FRCNN @ 50x64x64 (this script's dummy crop)
REF_FRCNN_GMAC_96 = 17.48        # frozen FRCNN @ 50x96x96
REF_WHOLE_GMAC = 26.51           # 10.611 + 15.897

#: CLI contract: no GPU -> exit 2 (documented in --help, see module docstring)
NO_GPU_MSG = """evaluated_mac_params: this script needs 1 CUDA GPU.
    The documented numbers (audio %.3f GMAC / 0.538 M params, frozen FRCNN %.3f GMAC,
    whole 2 s window %.2f GMAC; evidence: docs/verification/runs/r3-macs.log) were
    measured on the CUDA build, so a pure-CPU run cannot be assumed to match them.
    Fix: run on a machine with a GPU (--device auto / --device cuda).
    Escape hatch: --device cpu forces a CPU run; the printed numbers are then
    UNVERIFIED against the docs (the script prints an extra WARNING).
    See --help for the full contract.""" % (REF_AUDIO_GMAC, REF_FRCNN_GMAC_88, REF_WHOLE_GMAC)


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


def resolve_device(requested):
    """Map ``--device {auto,cuda,cpu}`` to a ``torch.device``.

    ``auto``/``cuda``: requires a visible CUDA device, otherwise prints :data:`NO_GPU_MSG`
    to stderr and exits with code 2 (the documented contract).
    ``cpu``: explicit opt-in; allowed, with a WARNING that the numbers are unverified.
    """
    if requested == "cpu":
        print("evaluated_mac_params WARNING: --device cpu -> the same operator graph is "
              "counted, so the MACs should equal the documented ones, but this path has NOT "
              "been re-verified against docs/verification/runs/r3-macs.log.")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        # plain stderr write: no rich markup parsing / no 80-col re-wrapping of an error
        sys.stderr.write(NO_GPU_MSG + "\n")
        sys.stderr.flush()
        raise SystemExit(2)
    # the original script pinned GPU 0 (``with torch.cuda.device(0)`` + ``.cuda()``)
    return torch.device("cuda", 0)


def build_video_model(videonet_cfg):
    """Build the frozen FRCNN; return ``None`` when its backbone weights are unavailable.

    ``configs/Falcon.yml`` ships a *relative* ``pretrain_zoo/frcnn_128_512.backbone.pth.tar``
    that is not part of the repository, so the unmodified script used to die with
    ``FileNotFoundError`` before counting anything. FRCNN is excluded from the MACs below,
    so a zero embedding with the same output shape is a faithful stand-in.
    """
    try:
        return getattr(look2hear.videomodels, videonet_cfg["videonet_name"])(
            **videonet_cfg["videonet_config"],
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"evaluated_mac_params WARNING: cannot build the frozen FRCNN video model "
              f"({exc}); falling back to a zero mouth_embedding with the same output shape. "
              f"The audio-side MACs below are unaffected (ptflops only counts audio_model).")
        return None


parser = argparse.ArgumentParser(
    description=(
        "Count Falcon audio-side MACs (ptflops 0.7.5; 1 MAC = 2 FLOPs; LSTM included; "
        "frozen FRCNN excluded) on one 2 s window."
    ),
    epilog=(
        "device contract: --device auto (default) and --device cuda REQUIRE a visible CUDA "
        "GPU and exit with code 2 otherwise (see --device). --device cpu is an explicit "
        "opt-in that is allowed but prints a WARNING because its numbers are unverified "
        "against docs/verification/runs/r3-macs.log. Reference values: audio 10.611 GMAC, "
        "frozen FRCNN 15.897 GMAC @50x88x88, whole window 26.51 GMAC."
    ),
)
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)
parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cuda", "cpu"],
    help=(
        "auto (default) / cuda: use 1 CUDA GPU, exit with code 2 when none is visible "
        "(the documented numbers come from the CUDA build). cpu: force CPU and accept "
        "numbers that are not re-verified against the docs (prints a WARNING)."
    ),
)

# Resolve the config next to this file instead of relying on cwd (portability):
# running the script from anywhere used to fail with FileNotFoundError.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "Falcon.yml")
with open(CONFIG_PATH) as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)
arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)

device = resolve_device(plain_args.device)
print(f"evaluated_mac_params device = {device}")
print(f"evaluated_mac_params config = {CONFIG_PATH}")
# tmp = torch.cuda.memory_allocated(0)
audiomodel = getattr(look2hear.models, arg_dic["audionet"]["audionet_name"])(
   #sample_rate=arg_dic["datamodule"]["data_config"]["sample_rate"],
    **arg_dic["audionet"]["audionet_config"]
)
audiomodel.to(device)
# b = torch.cuda.memory_allocated(0)
# print((b - tmp) / 10 ** 6)
video_model = build_video_model(arg_dic["videonet"])
# Frozen FRCNN output is (B, pretrained_vout_chan, T_frames); the two constants below are the
# fallback shape used when the backbone weights are missing (see build_video_model).
VOUT_CHAN = int(arg_dic["audionet"]["audionet_config"].get("pretrained_vout_chan", 512))
V_FRAMES = 50

# loss_func = getattr(look2hear.losses, arg_dic["loss"]["train"]["loss_func"])(
#             getattr(look2hear.losses, arg_dic["loss"]["train"]["sdr_type"]),
#             **arg_dic["loss"]["train"]["config"],
#         )
# optimizer = torch.optim.Adam(audiomodel.parameters(), lr=1e-4)
# with torch.cuda.device(3):
#     a = torch.randn(1, 16000).cuda()
#     v = torch.randn(1, 1, 25, 96, 96).cuda()
#     audiomodel = audiomodel.cuda()
#     video_model = video_model.cuda()
#     v = video_model(v)
#     print(audiomodel)
#     print(audiomodel(a, v).shape)
# print(v1 == v2)
# import pdb; pdb.set_trace()
# start = time.perf_counter()
# for i in range(100):
#     audiomodel(a)
# end = time.perf_counter()
# print((end - start)/100.)
def input_func(input):
    return {"audio_mixture":input[0],"mouth_embedding":input[1]}
# torch.cuda.device(...) only exists for CUDA; use a null-context on CPU.
_dev_idx = 0 if device.index is None else device.index
_dev_ctx = torch.cuda.device(_dev_idx) if device.type == "cuda" else contextlib.nullcontext()
with _dev_ctx:
    if video_model is not None:
        video_model = video_model.to(device)
        video_model.eval()
    a = torch.randn(1, 32000).to(device)
    v = torch.randn(1, 1, V_FRAMES, 64, 64).to(device)
    if video_model is not None:
        v = video_model(v)
    else:
        v = torch.zeros(1, VOUT_CHAN, V_FRAMES, device=device)
    total_macs = 0
    total_params = 0
    # DPRNN
    model = audiomodel.to(device)
    model.eval()
    macs, params = get_model_complexity_info(
        model, 
        (a, v), 
        input_constructor=input_func, 
        #backend='aten',              # <--- add this line to use aten backend
        as_strings=False, 
        print_per_layer_stat=False, 
        verbose=False
    )
    # optimizer.zero_grad()
    # loss = loss_func(a.unsqueeze(0), model(a, v))
    # loss.backward()
    # optimizer.step()
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is None:
    #         print(name)
    # model = audiomodel.cuda()
    # macs, params = get_model_complexity_info(
    #     model, (1,16000), as_strings=False, print_per_layer_stat=True, verbose=False
    # )
    total_macs += macs
    total_params += params
    # model = nn.Conv1d(1, 512, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (1, 32000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs
    # total_params += params

    # model = nn.ConvTranspose1d(512, 1, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (512, 2000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs*2
    # total_params += params*2
    # print(model(a, v).shape)
    print(f"(audio-only) MACs (GMAC; 1 MAC = 2 FLOPs; LSTM included; frozen FRCNN EXCLUDED)"
          f" = {total_macs / 10.0 ** 9:.3f}")
    print(f"(audio-only) Params (M; audio network only, excludes the 11.187 M frozen FRCNN)"
          f" = {total_params / 10.0 ** 6:.3f}")
    print(f"(reference) audio {REF_AUDIO_GMAC:.3f} GMAC (this line above) | "
          f"frozen FRCNN {REF_FRCNN_GMAC_88:.3f} GMAC @50x88x88 "
          f"(64^2 = {REF_FRCNN_GMAC_64:.2f}, 96^2 = {REF_FRCNN_GMAC_96:.2f}; this script feeds "
          f"64^2 only to build the embedding) | whole 2 s window {REF_WHOLE_GMAC:.2f} GMAC "
          f"= {REF_AUDIO_GMAC:.3f} + {REF_FRCNN_GMAC_88:.3f} (three different accounting scopes, "
          f"do not mix them)")
    
    # model.cpu()
    # a = a.cpu()
    # v = v.cpu()
    # # Set up cuda Events for timing measurement (PyTorch recommended interface)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # # Initialize a timing container
    # timings = np.zeros((100, 1))

    # with torch.no_grad():
    #     for rep in tqdm(range(200)):
    #         if rep >= 100:
    #             starter.record()
    #         _ = model(a, v)
    #         if rep >= 100:
    #             ender.record()  # end timing
    #             torch.cuda.synchronize()  # wait for GPU tasks to complete
    #             timings[rep - 100] = starter.elapsed_time(ender)  # record time

    # avg = timings.sum()/100
    # print('\navg={}\n'.format(avg))

    # # Measure audio time
    # sample_rate = arg_dic["datamodule"]["data_config"]["sample_rate"]  # get sample rate from config
    # num_samples = a.shape[1]  # get number of audio samples

    # duration = num_samples / sample_rate  # compute duration (seconds)
    # print(f"Audio duration: {duration} seconds")
