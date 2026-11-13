"""Single-target Falcon with a direct frequency-conditioned mask input."""

import torch
from torch import nn
from torch.nn import functional as F

from .Falcon_Contrastive import FalconSpeakerContrastive


class FrequencyVisualCondition(nn.Module):
    """Predict a gain and offset for each channel group and frequency band."""

    def __init__(self, visual_channels, audio_channels, *, groups=8, bands=8, hidden=32):
        super().__init__()
        if min(visual_channels, audio_channels, groups, bands, hidden) < 1 or audio_channels % groups:
            raise ValueError("audio channels must divide into positive equal-sized groups")
        self.groups = groups
        self.bands = bands
        self.projection = nn.Conv1d(visual_channels, hidden, 1)
        self.modulation = nn.Conv1d(hidden, 2 * groups * bands, 1)
        # Preserve the released waveform at initialization. The output
        # projection can still receive a gradient on the first update.
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features, video):
        batch, channels, frames, bins = features.shape
        modulation = self.modulation(F.silu(self.projection(video))).tanh()
        modulation = modulation.reshape(batch, 2 * self.groups, self.bands, -1)
        modulation = modulation.transpose(-1, -2)
        modulation = F.interpolate(modulation, size=(frames, bins), mode="nearest")
        gain, bias = modulation.chunk(2, dim=1)
        grouped = features.reshape(batch, self.groups, channels // self.groups, frames, bins)
        return (grouped + grouped * gain.unsqueeze(2) + bias.unsqueeze(2)).reshape_as(features)


class FalconRollingConditioned(FalconSpeakerContrastive):
    """Use the released separator and directly condition its mask by frequency.

    Forward always returns one waveform per target request. The caller supplies
    the same bounded rolling window and target mouth features as Falcon Small.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_src != 1:
            raise ValueError("rolling extraction requires one target output")
        self.frequency_condition = FrequencyVisualCondition(
            self.pretrained_vout_chan, self.audio_bn_params["out_chan"]
        )

    def forward(self, audio_mixture, mouth_embedding):
        mixture_embedding = self.encoder(audio_mixture)
        audio = self.audio_bottleneck(mixture_embedding)
        video = self.video_bottleneck(mouth_embedding)
        refined = self.separator(audio, video)
        conditioned = self.frequency_condition(refined, video)
        estimate_embedding = self.mask_generator(conditioned, mixture_embedding)
        return self.decoder(estimate_embedding, audio_mixture.shape)
