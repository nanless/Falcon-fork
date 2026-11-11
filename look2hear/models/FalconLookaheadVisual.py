"""Original FRCNN visual features with two-frame streaming lookahead.

Each call consumes three current 25 fps frames and the next two frames; it
returns exactly the features for the three current frames. The caller owns the
80 ms acquisition delay, end-of-stream zero padding, and timestamp validation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel, threeD_to_2D_tensor


class FalconLookaheadVisual(nn.Module):
    """Frozen original visual backbone with an explicit two-frame history.

    ``frames`` has shape [B, 5, H, W]: three frames for this chunk followed by
    two real lookahead frames (or zeros only when the stream has ended).
    ``history`` is the final two frames of the preceding *output* chunk. Passing
    ``None`` uses the original FRCNN's zero padding at the beginning of a stream.
    The returned state never includes lookahead frames.
    """

    def __init__(self, backbone: FRCNNVideoModel):
        super().__init__()
        conv = backbone.frontend3D[0]
        if (not isinstance(conv, nn.Conv3d) or conv.kernel_size != (5, 7, 7)
                or conv.stride != (1, 2, 2) or conv.padding != (2, 3, 3)):
            raise ValueError("expected the original symmetric Falcon FRCNN frontend")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, frames: torch.Tensor, history: torch.Tensor | None = None):
        if frames.ndim != 4 or frames.size(1) != 5:
            raise ValueError("frames must have shape [B, 5, H, W]")
        expected = (frames.size(0), 2, frames.size(2), frames.size(3))
        if history is None:
            history = frames.new_zeros(expected)
        elif history.shape != expected or history.dtype != frames.dtype or history.device != frames.device:
            raise ValueError("history shape, dtype and device must match frames")
        context = torch.cat((history, frames), dim=1).unsqueeze(1)
        conv = self.backbone.frontend3D[0]
        x = F.conv3d(context, conv.weight, conv.bias, stride=conv.stride,
                     padding=(0, *conv.padding[1:]), dilation=conv.dilation,
                     groups=conv.groups)
        for layer in list(self.backbone.frontend3D.children())[1:]:
            x = layer(x)
        batch, _, count, _, _ = x.shape
        x = self.backbone.trunk(threeD_to_2D_tensor(x))
        features = x.reshape(batch, count, -1).transpose(1, 2)
        # Only emitted frames become history; the lookahead is submitted again
        # as current input in the next call.
        return features, frames[:, 1:3].clone()
