"""Falcon 用到的归一化算子。

本文件中 Falcon 生产配方实际实例化的只有三个：`LayerNormalizationHW`（别名
`cLNhw`，30 个）、`CumulateLN4D`（30 个，作为 `cLNhw` 的第二段）、
`LayerNormalization4D`（60 个）；`gLN` / `cLN` / `bN` / `LN` 在 Falcon 前向里
一次都没有被构造，属 Conv-TasNet / Asteroid 遗留（详见 `02-layers-dsp.md`）。

关于 `cLNhw` 的三条硬事实（实测见 `tools/repro/experiments/ln_variant.py`）：

1. `cLNhw` == ``GlobalLayerNorm(C,T,F)`` ∘ ``CumulateLN4D``：前者是
   ``nn.GroupNorm(num_groups=1, C)``，即**逐样本**在 ``(C,T,F)`` 上取标量均值/方差；
   后者沿 ``T`` 累积、统计量只跨 ``C``（对 ``F`` 逐频点独立）。
2. 初始态下第一段近似**空操作**：``|cLNhw(x) - CumulateLN4D(x)|max = 3.576e-07``
   （全窗 GN 的归一化被后一段的归一化重新吸收）；把 GN 的 affine 随机化后差异升到
   ``1.232``，所以这段的净增量实质上只有 ``2*C`` 个仿射参数。
3. 因为第一段用了**全窗**统计，``cLNhw`` 使窗内**完全非因果**：窗内任一帧的输出都
   依赖窗内所有帧。实测（2 s 窗 / 251 帧 / `falcon.pt` epoch 714）：扰动尾部 1 帧，
   帧 0..120 的相对响应是 -40.2 dB（处处响应）；把 30 个 `cLNhw` 换成逐 (t,f) 的
   ``LayerNormalization4D((C,1))`` 后，同一段降到 -240 dB（数值零）。

注意：本文件只做文档说明，任何改动都不得触碰计算路径——`cLNhw` 的数值行为是
`falcon.pt` 复现基线的一部分。
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from typing import List
from torch.nn.modules.batchnorm import _BatchNorm
from collections.abc import Iterable

EPS = 1e-5

def norm(x, dims: List[int], EPS: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt(var2 + EPS)
    return value


def glob_norm(x, ESP: float = 1e-8):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return norm(x, dims, ESP)


class MLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)

    def forward(self, x, EPS: float = 1e-8):
        pass


class GlobalLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        value = glob_norm(x, EPS)
        return self.apply_gain_and_bias(value)


class ChannelLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumulateLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        batch, channels, time = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=1)
        cnt = torch.arange(
            start=channels, end=channels * (time + 1), step=channels, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = (cum_pow_sum / cnt) - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError(
                "expected 4D or 3D input (got {}D input)".format(input.dim())
            )


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


class CumulateLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(CumulateLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )

class GlobalLayerNorm(nn.Module):
    """全窗归一化：``nn.GroupNorm(num_groups=1, num_channels=C)``。

    输入 ``(B,C,T,F)`` 时统计量落在 ``(C,T,F)`` 全部轴上（每个样本一组标量
    mean/var），因此**任意一帧的输出都依赖整个窗口**——在时间维上非因果。
    等价实现与手算 ``mean over (C,T,F)`` 的差为 2.384e-07（实测）。

    Falcon 只在 ``cLNhw`` 内部以第一段的形式用到它；它自己的仿射参数是
    ``2*C`` 个（``GroupNorm`` 的 weight/bias）。
    """

    def __init__(self, num_channels: int = 1, eps: float = EPS):
        super(GlobalLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.num_channels, eps=self.eps)

    def forward(self, x: torch.Tensor):
        return self.norm(x)


class LayerNormalization4D(nn.Module):
    """可配置统计轴的 4D LayerNorm（Falcon 里 60 个实例，全在 attention/DPRNN 内部）。

    ``input_dimension`` 是 ``(C, F)`` 二元组，构造时据此决定统计轴：

    * ``F > 1`` → ``self.dim = (1, 3)``，即跨 ``C`` 与 ``F``、**逐帧**独立（因果，
      但同帧内跨频点耦合）；Falcon 的 ``MultiHeadSelfAttention2D`` 用这个。
    * ``F == 1`` → ``self.dim = (1,)``，即只跨 ``C``、逐 ``(t,f)`` 独立（因果）；
      ``DualPathRNN`` 用这个，本文件里 ``ln_variant.py`` 的 ``ln_c`` 变体也用它来
      替换 ``cLNhw``。

    ``gamma/beta`` 形状都是 ``(1, C, 1, F)``；``eps`` 加在 ``var`` 上再开方。
    """

    def __init__(self, input_dimension: Iterable, eps: float = EPS):
        super(LayerNormalization4D, self).__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]

        self.dim = (1, 3) if param_size[-1] > 1 else (1,)
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
    
class CumulateLN4D(nn.Module):
    """4D 累积层归一化：沿 ``H``（= 时间轴 ``T``）累积，统计量只跨 ``C``。

    对输入 ``(B,C,H,W)``，``step_sum = x.sum(dim=1)`` 先跨通道求和，再沿 ``H`` 做
    ``cumsum``；分母 ``arange(C, C*(H+1), C)`` 即"截至该步已见过的元素数"，因此

        mean_t = (1/(t*C)) * sum_{t'<=t} sum_c x[t',c]
        var_t  = E[x^2]_{t'<=t} - mean_t^2

    逐 ``(b,w)`` 独立，最后一个 ``H`` 步的输出只用到 ``t'<=t``——**时间上因果**，
    但记忆**无界**（输出是全部历史的函数）。``eps`` 加在 ``var`` 上再开方。

    实测：与手算（cum over H、stats over C）差 ``0.0``；把 ``hidden`` 侧输入换成
    逐 (t,f) 的 ``LayerNormalization4D((C,1))`` 后，2 s 窗上"扰动尾部→帧 0..120"
    的相对响应从 -40.2 dB 掉到 -240 dB（数值零），即因果性的唯一来源就是这一段。
    """

    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(CumulateLN4D, self).__init__()
        self.eps = eps
        if trainable:
            # gain and bias dimensions match input, controlling (C, W) respectively
            self.gain = nn.Parameter(torch.ones(1, dimension, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1, 1))
        else:
            self.gain = torch.ones(1, dimension, 1, 1, requires_grad=False)
            self.bias = torch.zeros(1, dimension, 1, 1, requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Channel, Height, Width)
        B, C, H, W = input.size()
        
        # Cumulative normalization along the Height dimension (H corresponds to time steps)
        step_sum = input.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        step_pow_sum = input.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        cum_sum = torch.cumsum(step_sum, dim=2)  # (B, 1, H, W)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=2)  # (B, 1, H, W)

        # Cumulative mean and variance for each H step
        entry_cnt = torch.arange(C, C * (H + 1), C, dtype=input.dtype, device=input.device)
        entry_cnt = entry_cnt.view(1, 1, -1, 1).expand_as(cum_sum)  # (B, 1, H, W)

        cum_mean = cum_sum / entry_cnt  # (B, 1, H, W)
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # (B, 1, H, W)
        cum_std = (cum_var + self.eps).sqrt()  # (B, 1, H, W)

        # Broadcast cumulative mean and std to match input shape
        cum_mean = cum_mean.expand(-1, C, -1, -1)
        cum_std = cum_std.expand(-1, C, -1, -1)

        # Normalize
        x = (input - cum_mean) / cum_std
        return x * self.gain + self.bias

class LayerNormalizationHW(nn.Module):
    """``cLNhw``：``GlobalLayerNorm(C,T,F)`` ∘ ``CumulateLN4D``（Falcon 的主力归一化）。

    输入 ``(B,C,H=T,W=F)``，两步串联：

    1. ``GlobalLayerNorm`` → ``nn.GroupNorm(1,C)``，逐样本在 ``(C,T,F)`` 上取标量
       均值/方差。**这一步就注定了窗内非因果**：任何一帧的统计量都含其它所有帧。
    2. ``CumulateLN4D`` → 沿 ``T`` 累积、统计量只跨 ``C``、对 ``F`` 逐频点独立，
       本身是因果的，但它作用在已混入未来信息的张量上，救不回来。

    两条容易踩的量化事实：

    * **初始态下第 1 步近似空操作**：``|cLNhw(x) - CumulateLN4D(x)|max = 3.576e-07``
      （全窗 GN 的归一化被第 2 步重新吸收）；把 GN 的 affine 随机化后该差升到
      ``1.232``。所以第 1 步的净增量实质上只有 ``2*C`` 个仿射参数。
    * **非因果性可测**：2 s 窗（251 帧）上扰动尾部一帧，``falcon.pt`` 的
      separator 特征在帧 0..120 的相对响应为 -40.2 dB（处处响应）；把全部 30 个
      ``cLNhw`` 换成逐 (t,f) 的 ``LayerNormalization4D((C,1))`` 后同一段为
      -240 dB（数值零）。复现脚本：``tools/repro/experiments/ln_variant.py``。

    只接受 4D 输入：3D 会在 ``CumulateLN4D`` 的 ``B,C,H,W = input.size()`` 处
    ``ValueError``。
    """

    def __init__(self, num_channels, eps=1e-8):
        super(LayerNormalizationHW, self).__init__()
        # GlobalLayerNorm for W dimension
        self.global_layer_norm = GlobalLayerNorm(num_channels=num_channels, eps=eps)
        # CumulateLN for H dimension
        self.cumulate_ln = CumulateLN4D(dimension=num_channels, eps=eps)

    def forward(self, x):
        # Input shape: (B, C, H, W)

        # GlobalLayerNorm on W dimension
        x = self.global_layer_norm(x)  # Normalizes over C and W dimensions

        # CumulateLN on H dimension
        x = self.cumulate_ln(x)  # (B, C, H, W) cumulative normalization on H
        
        return x

# Aliases.
gLN = GlobalLayerNorm
cLN = CumulateLN
LN = CumulativeLayerNorm
bN = BatchNorm
LN4d = LayerNormalization4D
cLNhw = LayerNormalizationHW

def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if hasattr(nn, identifier):
            cls = getattr(nn, identifier)
        else:
            cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
