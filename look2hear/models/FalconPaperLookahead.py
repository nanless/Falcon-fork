# 120 ms streaming Falcon-style separator with 80 ms shared lookahead.
from dataclasses import dataclass

import torch
from torch import nn

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from look2hear.models.FalconPaperStreamPrototype import PaperStreamPrototype, PaperStreamState
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel


@dataclass
class FalconLookaheadState:
    visual: torch.Tensor | None = None
    audio: PaperStreamState | None = None


class FalconPaperLookahead(nn.Module):
    def __init__(self, visual_init, width=48, rnn_hidden=64, heads=4, history_frames=125):
        super().__init__()
        backbone = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu")
        backbone.init_from(visual_init)
        self.visual = FalconLookaheadVisual(backbone)
        self.core = PaperStreamPrototype(width=width, rnn_hidden=rnn_hidden,
                                         heads=heads, history_frames=history_frames)

    def forward(self, audio, video, state=None):
        if state is None:
            state = FalconLookaheadState()
        features, visual_state = self.visual(video, state.visual)
        output, audio_state = self.core(audio, features.transpose(1, 2), state.audio)
        return output, FalconLookaheadState(visual=visual_state, audio=audio_state)
