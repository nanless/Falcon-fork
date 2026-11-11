"""Causal, stateful audiovisual Falcon-family streaming research model.

This is a new architecture, not the published Falcon checkpoint architecture.
Each public call processes one complete 120 ms block. Its 15 audio hops and
three visual frames may interact bidirectionally within the block; later blocks
are never inputs to the current call. Per-stream caches contain only history.
"""
from __future__ import annotations

from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel, threeD_to_2D_tensor

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1920
VIDEO_FRAMES = 3
FRAME_SAMPLES = 128
FFT_SIZE = 256
# A hop retains 128 input samples, hence at most 129 causal FIR taps are valid.
FIR_TAPS = FRAME_SAMPLES + 1


def causal_mask_filter(spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a per-hop adaptive causal FIR and return its last 128 samples.

    The predicted real mask is inverse-transformed and truncated to 129 causal
    taps. The retained half of the inverse FFT is then exact linear convolution.
    The mask is computed once per 8 ms hop, not once per sample.
    """
    impulse = torch.fft.irfft(mask, n=FFT_SIZE, dim=-1)
    causal_impulse = F.pad(impulse[..., :FIR_TAPS], (0, FFT_SIZE - FIR_TAPS))
    response = torch.fft.rfft(causal_impulse, n=FFT_SIZE, dim=-1)
    filtered = torch.fft.irfft(spec * response, n=FFT_SIZE, dim=-1)
    return filtered[..., FRAME_SAMPLES:]


class CausalResidualBlock(nn.Module):
    """Depthwise dilated temporal convolution with explicit left state."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.history = 2 * dilation
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(channels, channels, 3, dilation=dilation,
                                   groups=channels)
        self.in_proj = nn.Conv1d(channels, channels * 2, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None):
        # LayerNorm is channel-only at each time, never across time or batch.
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if state is None:
            state = z.new_zeros(z.size(0), z.size(1), self.history)
        if state.shape != (z.size(0), z.size(1), self.history):
            raise ValueError("audio block state shape does not match input")
        padded = torch.cat((state, z), dim=-1)
        y = self.depthwise(padded)
        a, b = self.in_proj(y).chunk(2, dim=1)
        return x + self.out_proj(torch.tanh(a) * torch.sigmoid(b)), padded[..., -self.history:]


class FalconStreaming(nn.Module):
    """120 ms block-causal audio-visual extraction with explicit stream state.

    Input audio: [B, 1920], floating point mono at 16 kHz.
    Input mouth: [B, 3, 88, 88], normalized (gray / 255 - .421) / .165.
    Each mouth frame must have a capture timestamp <= its 40 ms interval start.
    No audio-visual alignment or tracking is performed inside this model.

    forward_chunk returns (waveform [B,1920], state). State belongs to one
    continuous stream/batch; None resets it. Reusing a state across identities
    would leak context. Detach state between truncated-BPTT training segments.
    forward supports concatenated whole chunks and preserves gradients across them.
    """
    def __init__(self, channels: int = 320, dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)):
        super().__init__()
        # V4 uses 120 ms blocks; older checkpoints do not encode
        # the same architecture and must not silently load as this model.
        self.register_buffer("_streaming_format_version", torch.tensor(4, dtype=torch.int64))
        if channels < 1 or not dilations or any(d < 1 for d in dilations):
            raise ValueError("channels and dilations must be positive")
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.video_model = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu", pretrain=None)
        # The 3D kernel is unchanged from Falcon; unlike the original symmetric
        # temporal padding, four past frames are explicitly cached.
        self.video_model.frontend3D[0].padding = (0, 3, 3)
        self.video_proj = nn.Conv1d(512, channels, 1)
        self.audio_proj = nn.Conv1d(FFT_SIZE + 2, channels, 1)
        self.blocks = nn.ModuleList(CausalResidualBlock(channels, d) for d in self.dilations)
        # Fifteen audio tokens and three video tokens, exclusively from this
        # block. The causal stack supplies bounded history; attention supplies
        # all-to-all context within the current block.
        self.chunk_norm = nn.LayerNorm(channels)
        self.chunk_attention = nn.MultiheadAttention(channels, num_heads=1, dropout=0.0,
                                                     batch_first=True)
        self.chunk_positions = nn.Parameter(torch.zeros(1, CHUNK_SAMPLES // FRAME_SAMPLES + VIDEO_FRAMES,
                                                        channels))
        nn.init.normal_(self.chunk_positions, std=0.02)
        self.mask = nn.Conv1d(channels, FFT_SIZE // 2 + 1, 1)
        self._freeze_bn_stats()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate even under strict=False: changing synthesis semantics while
        # keeping identical parameter names is not a safe partial load.
        name = prefix + "_streaming_format_version"
        marker = state_dict.get(name)
        if (not isinstance(marker, torch.Tensor) or marker.ndim != 0
                or marker.dtype != torch.int64 or marker.item() != 4):
            error_msgs.append(f"{name}: expected scalar int64 model version 4")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _freeze_bn_stats(self):
        # Training BatchNorm over frames mixes future frames and samples.
        # Fixed running statistics are necessary in *both* train and eval mode.
        for module in self.video_model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_bn_stats()
        return self

    def _visual(self, video: torch.Tensor, old: torch.Tensor | None):
        if old is None:
            old = video.new_zeros(video.size(0), 4, 88, 88)
        if old.shape != (video.size(0), 4, 88, 88):
            raise ValueError("video state shape does not match input")
        frames = torch.cat((old, video), dim=1)
        x = self.video_model.frontend3D(frames.unsqueeze(1))
        batch, _, count, _, _ = x.shape
        x = self.video_model.trunk(threeD_to_2D_tensor(x))
        x = x.reshape(batch, count, -1).transpose(1, 2)
        return self.video_proj(x), frames[:, -4:]

    def forward_chunk(self, audio: torch.Tensor, video: torch.Tensor,
                      state: dict[str, Any] | None = None):
        if audio.ndim != 2 or audio.shape[1] != CHUNK_SAMPLES:
            raise ValueError("audio must have shape [B, 1920]")
        if video.ndim != 4 or video.shape != (audio.size(0), VIDEO_FRAMES, 88, 88):
            raise ValueError("video must have shape [B, 3, 88, 88]")
        if audio.dtype != torch.float32 or video.dtype != torch.float32:
            raise ValueError("audio and video must be float32")
        if audio.device != video.device or audio.dtype != video.dtype:
            raise ValueError("audio and video must share device and dtype")
        if state is None:
            state = {}
        else:
            if not isinstance(state, dict) or set(state) != {"audio", "video", "blocks"}:
                raise ValueError("stream state must contain exactly audio, video and blocks")
            audio_state, video_state, block_states = state["audio"], state["video"], state["blocks"]
            expected = [
                (audio_state, (audio.size(0), FRAME_SAMPLES)),
                (video_state, (audio.size(0), 4, 88, 88)),
            ]
            if not isinstance(block_states, (list, tuple)) or len(block_states) != len(self.blocks):
                raise ValueError("stream state has the wrong number of audio blocks")
            expected.extend((cached, (audio.size(0), self.channels, block.history))
                            for cached, block in zip(block_states, self.blocks))
            for cached, shape in expected:
                if (not isinstance(cached, torch.Tensor) or tuple(cached.shape) != shape
                        or cached.dtype != audio.dtype or cached.device != audio.device):
                    raise ValueError("stream state tensor shape, dtype or device does not match input")
        old_audio = state.get("audio")
        if old_audio is None:
            old_audio = audio.new_zeros(audio.size(0), FRAME_SAMPLES)
        if old_audio.shape != (audio.size(0), FRAME_SAMPLES):
            raise ValueError("audio state shape does not match input")
        # 128 past + 15x128 new samples -> fifteen causal 256-sample windows.
        # 129-tap causal FIR plus overlap-save is alias-free and gives an
        # identity when mask == 1, without shifting/dropping samples.
        all_audio = torch.cat((old_audio, audio), dim=-1)
        frames = all_audio.unfold(-1, FFT_SIZE, FRAME_SAMPLES)
        spec = torch.fft.rfft(frames.float(), n=FFT_SIZE, dim=-1)
        feats = torch.cat((spec.real, spec.imag), dim=-1).transpose(1, 2)
        x = self.audio_proj(feats)
        visual, video_state = self._visual(video, state.get("video"))
        # A video frame at each 40ms bin start conditions the five 8ms steps.
        x = x + visual.repeat_interleave(5, dim=-1)
        previous = state.get("blocks")
        if previous is None:
            previous = [None] * len(self.blocks)
        if len(previous) != len(self.blocks):
            raise ValueError("wrong number of audio block states")
        next_states = []
        for block, block_state in zip(self.blocks, previous):
            x, block_state = block(x, block_state)
            next_states.append(block_state)
        # The call has received the entire block; the attention never sees
        # tokens from subsequent blocks, including during training.
        joint = torch.cat((x.transpose(1, 2), visual.transpose(1, 2)), dim=1)
        joint = self.chunk_norm(joint + self.chunk_positions)
        attended, _ = self.chunk_attention(joint, joint, joint, need_weights=False)
        x = x + attended[:, :CHUNK_SAMPLES // FRAME_SAMPLES].transpose(1, 2)
        mask = torch.sigmoid(self.mask(x)).transpose(1, 2)
        reconstructed = causal_mask_filter(spec, mask)
        output = reconstructed.reshape(audio.size(0), CHUNK_SAMPLES)
        return output, {"audio": all_audio[:, -FRAME_SAMPLES:],
                        "video": video_state, "blocks": next_states}

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        if audio.ndim != 2 or audio.size(1) == 0 or audio.size(1) % CHUNK_SAMPLES:
            raise ValueError("audio length must be a positive multiple of 1920")
        chunks = audio.size(1) // CHUNK_SAMPLES
        if video.ndim != 4 or video.shape != (audio.size(0), chunks * VIDEO_FRAMES, 88, 88):
            raise ValueError("expected three 88x88 video frames per audio chunk")
        state = None
        outputs = []
        for index in range(chunks):
            y, state = self.forward_chunk(audio[:, index*CHUNK_SAMPLES:(index+1)*CHUNK_SAMPLES],
                                          video[:, index*VIDEO_FRAMES:(index+1)*VIDEO_FRAMES], state)
            outputs.append(y)
        return torch.cat(outputs, dim=-1)
