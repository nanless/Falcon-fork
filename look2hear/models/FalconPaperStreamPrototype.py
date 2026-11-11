"""Experimental 120ms/80ms stream core. Visual frontend is supplied separately.

One step takes 3200 samples (1920 committed + 1280 lookahead) and three
already-computed current-chunk visual features. Computing those features with
the original five-frame frontend requires two additional future video frames.
History excludes speculative RNN/attention state. Do not use as a trained model.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class PaperStreamState:
    audio_tail: torch.Tensor
    temporal_h: torch.Tensor | None = None
    temporal_c: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    boundary_spec: torch.Tensor | None = None
    temporal_conv: torch.Tensor | None = None


class FrameNorm(nn.Module):
    """Channel-only normalization: no future time or other sample statistics."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PaperStreamPrototype(nn.Module):
    chunk_samples = 1920
    lookahead_samples = 1280
    hop = 128
    window = 256
    commit_frames = 15

    def __init__(self, width=24, rnn_hidden=32, heads=4, history_frames=125,
                 visual_channels=512, temporal_kernel=5, temporal_dilation=16):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by attention heads')
        self.width, self.history_frames, self.visual_channels = width, history_frames, visual_channels
        if temporal_kernel < 1 or temporal_dilation < 1:
            raise ValueError("temporal convolution kernel and dilation must be positive")
        self.temporal_context = (temporal_kernel - 1) * temporal_dilation
        self.temporal_depthwise = nn.Conv1d(width, width, temporal_kernel, dilation=temporal_dilation, groups=width)
        self.temporal_pointwise = nn.Conv1d(width, width, 1)
        self.input_conv = nn.Conv2d(3, width, (1, 3), padding=(0, 1))
        self.norm0 = FrameNorm(width)
        self.down1 = nn.Conv2d(width, width, (1, 5), stride=(1, 2), padding=(0, 2), groups=width)
        self.mix1 = nn.Conv2d(width, width, 1)
        self.norm1 = FrameNorm(width)
        self.down2 = nn.Conv2d(width, width, (1, 5), stride=(1, 2), padding=(0, 2), groups=width)
        self.mix2 = nn.Conv2d(width, width, 1)
        self.norm2 = FrameNorm(width)
        self.down3 = nn.Conv2d(width, width, (1, 5), stride=(1, 2), padding=(0, 2), groups=width)
        self.mix3 = nn.Conv2d(width, width, 1)
        self.norm3 = FrameNorm(width)
        self.global2 = nn.Conv2d(width, width, 1)
        self.global1 = nn.Conv2d(width, width, 1)
        self.global0 = nn.Conv2d(width, width, 1)
        self.freq_rnn = nn.LSTM(width, rnn_hidden, batch_first=True, bidirectional=True)
        self.freq_out = nn.Linear(rnn_hidden * 2, width)
        self.time_rnn = nn.LSTM(width, rnn_hidden, batch_first=True)
        self.time_out = nn.Linear(rnn_hidden, width)
        self.visual = nn.Linear(visual_channels, width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.up2 = nn.Conv2d(width * 2, width, 1)
        self.up1 = nn.Conv2d(width * 2, width, 1)
        self.up0 = nn.Conv2d(width * 2, width, 1)
        self.decode = nn.Conv2d(width, width, (1, 3), padding=(0, 1))
        self.decode_act = nn.PReLU(width)
        self.mask = nn.Conv2d(width, 2, 1)
        nn.init.normal_(self.mask.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mask.bias)
        self.register_buffer('hann', torch.hann_window(self.window), persistent=False)

    def initial_state(self, batch, *, device=None, dtype=None):
        return PaperStreamState(torch.zeros(batch, self.hop, device=device, dtype=dtype))

    @staticmethod
    def _resize_freq(x, size):
        return F.interpolate(x, size=(x.shape[-2], size), mode='nearest')

    def forward(self, audio, video, state=None):
        if audio.ndim != 2 or audio.shape[1] != self.chunk_samples + self.lookahead_samples:
            raise ValueError('audio must be [batch, 3200] with 120ms committed + 80ms future')
        batch = audio.shape[0]
        if video.shape != (batch, 3, self.visual_channels):
            raise ValueError('video must contain the three aligned current-chunk features')
        if state is None:
            state = self.initial_state(batch, device=audio.device, dtype=audio.dtype)
        # Analysis frames centered on each hop, with the one-hop left tail.
        audio_frames = torch.cat((state.audio_tail, audio), 1).float().unfold(1, self.window, self.hop)
        window = self.hann.to(dtype=torch.float32, device=audio.device)
        mix = torch.fft.rfft(audio_frames * window, n=self.window, dim=-1)
        inp = torch.stack((mix.real, mix.imag, mix.abs()), dim=1)
        x0 = F.prelu(self.norm0(self.input_conv(inp)), inp.new_tensor(0.25))
        x1 = F.prelu(self.norm1(self.mix1(self.down1(x0))), inp.new_tensor(0.25))
        x2 = F.prelu(self.norm2(self.mix2(self.down2(x1))), inp.new_tensor(0.25))
        x3 = F.prelu(self.norm3(self.mix3(self.down3(x2))), inp.new_tensor(0.25))
        b, c, t, f = x3.shape
        freq = x3.permute(0, 2, 3, 1).reshape(b*t, f, c)
        freq, _ = self.freq_rnn(freq)
        x3 = x3 + self.freq_out(freq).reshape(b, t, f, c).permute(0, 3, 1, 2)
        # A temporal LSTM runs separately at each frequency. Capture state only
        # at frame 15; later lookahead frames are speculative and uncommitted.
        seq = x3.permute(0, 3, 2, 1).reshape(b*f, t, c)
        conv_input = seq.transpose(1, 2)
        conv_history = state.temporal_conv
        if conv_history is None:
            conv_history = conv_input.new_zeros(b*f, c, self.temporal_context)
        if conv_history.shape != (b*f, c, self.temporal_context):
            raise ValueError("invalid temporal convolution state")
        joined = torch.cat((conv_history, conv_input), dim=-1)
        conv_result = self.temporal_pointwise(self.temporal_depthwise(joined)).transpose(1, 2)
        seq = seq + conv_result
        joined_committed = torch.cat((conv_history, conv_input[:, :, :self.commit_frames]), dim=-1)
        conv_next = joined_committed[:, :, -self.temporal_context:] if self.temporal_context else joined_committed[:, :, :0]
        prev = None if state.temporal_h is None else (state.temporal_h, state.temporal_c)
        committed, saved = self.time_rnn(seq[:, :self.commit_frames], prev)
        future, _ = self.time_rnn(seq[:, self.commit_frames:], saved)
        temporal = self.time_out(torch.cat((committed, future), dim=1))
        temporal = temporal.reshape(b, f, t, c).permute(0, 3, 2, 1)
        x3 = x3 + temporal
        vis = self.visual(video).repeat_interleave(5, dim=1)
        # Lookahead audio is available; future visual feature values are held,
        # avoiding a further FRCNN lookahead beyond the 80ms input budget.
        vis = torch.cat((vis, vis[:, -1:].expand(-1, t-vis.shape[1], -1)), dim=1)
        tokens = x3.mean(dim=-1).transpose(1, 2) + vis
        memory = state.memory
        keys = tokens if memory is None else torch.cat((memory, tokens), dim=1)
        attended, _ = self.attn(tokens, keys, keys, need_weights=False)
        x3 = x3 + attended.transpose(1, 2).unsqueeze(-1)
        # Inject bottleneck context into all skip scales, then fuse top-down.
        g2 = x2 + self.global2(self._resize_freq(x3, x2.shape[-1]))
        g1 = x1 + self.global1(self._resize_freq(x3, x1.shape[-1]))
        g0 = x0 + self.global0(self._resize_freq(x3, x0.shape[-1]))
        u2 = self.up2(torch.cat((g2, self._resize_freq(x3, x2.shape[-1])), dim=1))
        u1 = self.up1(torch.cat((g1, self._resize_freq(u2, x1.shape[-1])), dim=1))
        u0 = self.up0(torch.cat((g0, self._resize_freq(u1, x0.shape[-1])), dim=1))
        m = self.mask(self.decode_act(self.decode(u0))).permute(0, 2, 3, 1)
        # Complex mask permits phase correction. Identity-biased initialization.
        # PyTorch complex does not accept bfloat16; FFT/OLA stay float32.
        mask = torch.complex(1.0 + m[..., 0].float(), m[..., 1].float())
        pred = mix * mask
        if state.boundary_spec is not None:
            pred = torch.cat((state.boundary_spec[:, None], pred[:, 1:]), dim=1)
        # Weighted OLA. Frame 0 starts at sample -128 relative to this chunk.
        time = torch.fft.irfft(pred, n=self.window, dim=-1) * window
        total = (t - 1) * self.hop + self.window
        wav = F.fold(time.transpose(1, 2), output_size=(1, total), kernel_size=(1, self.window), stride=(1, self.hop)).reshape(batch, total)
        weights = window.square().view(1, self.window, 1).expand(1, self.window, t)
        norm = F.fold(weights, output_size=(1, total), kernel_size=(1, self.window), stride=(1, self.hop)).reshape(1, total)
        out = (wav / norm.clamp_min(1e-8))[:, self.hop:self.hop+self.chunk_samples]
        new_memory = keys[:, :keys.shape[1]-t+self.commit_frames]
        if self.history_frames <= 0:
            new_memory = None
        else:
            new_memory = new_memory[:, -self.history_frames:]
        new_state = PaperStreamState(audio[:, self.chunk_samples-self.hop:self.chunk_samples],
                                     saved[0], saved[1], new_memory, pred[:, self.commit_frames], conv_next)
        return out, new_state
