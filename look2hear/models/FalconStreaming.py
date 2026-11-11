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


class SubbandRefiner(nn.Module):
    """Estimate a signed gain correction from local time-frequency structure."""

    def __init__(self, channels: int):
        super().__init__()
        self.bins = FFT_SIZE // 2 + 1
        self.group_width = 8
        self.groups = (self.bins + self.group_width - 1) // self.group_width
        self.pad_bins = self.groups * self.group_width - self.bins
        width = 16
        self.frequency_positions = nn.Parameter(torch.zeros(1, width, 1, self.groups))
        nn.init.normal_(self.frequency_positions, std=0.02)
        self.spectrum_proj = nn.Linear(3 * self.group_width, width)
        self.condition_proj = nn.Conv1d(channels, width, 1)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.output_proj = nn.Conv2d(width, self.group_width, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def spectral_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        magnitude = spectrum.abs()
        compressed = magnitude.log1p()
        # The scale tends to one at silence and retains relative complex phase.
        scale = (1.0 + magnitude).rsqrt()
        features = torch.stack((compressed, spectrum.real * scale, spectrum.imag * scale), dim=-1)
        features = F.pad(features, (0, 0, 0, self.pad_bins))
        return features.reshape(spectrum.size(0), spectrum.size(1), self.groups, -1)

    def forward(self, spectrum: torch.Tensor, condition: torch.Tensor, *,
                features: torch.Tensor | None = None) -> torch.Tensor:
        if features is None:
            features = self.spectral_features(spectrum)
        local = self.spectrum_proj(features).permute(0, 3, 1, 2)
        local = local + self.condition_proj(condition).unsqueeze(-1) + self.frequency_positions
        # A symmetric convolution shares weights across both axes of this
        # received chunk; padding introduces no samples from other chunks.
        local = local + F.gelu(self.pointwise(F.gelu(self.depthwise(local))))
        gains = self.output_proj(local).permute(0, 2, 3, 1).flatten(2)
        return gains[..., :self.bins].transpose(1, 2)


class SubbandResidualBlock(nn.Module):
    """Mix neighbouring hops and bands within the received chunk."""

    def __init__(self, width: int, dilation: tuple[int, int], depth: int):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=dilation,
                                   dilation=dilation, groups=width)
        self.expand = nn.Conv2d(width, 2 * width, 1)
        self.mix = nn.Conv2d(width, width, 1)
        self.residual_scale = depth ** -0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value, gate = self.expand(self.depthwise(features)).chunk(2, dim=1)
        return features + self.residual_scale * self.mix(value * F.silu(gate))


class SubbandCapacity(nn.Module):
    """Deeper local estimator, additive to the original v8 subband path."""

    def __init__(self, channels: int, width: int, depth: int, *,
                 bins: int, group_width: int, groups: int):
        super().__init__()
        self.bins = bins
        self.frequency_positions = nn.Parameter(torch.empty(1, width, 1, groups))
        nn.init.normal_(self.frequency_positions, std=0.02)
        self.visual_band_gate = nn.Parameter(torch.zeros(1, width, 1, groups))
        self.spectrum_proj = nn.Linear(3 * group_width, width)
        self.condition_proj = nn.Conv1d(channels, width, 1)
        self.visual_proj = nn.Conv1d(channels, 2 * width, 1)
        dilations = ((1, 1), (2, 1), (4, 2), (1, 3))
        self.blocks = nn.Sequential(*(SubbandResidualBlock(width, dilations[i % len(dilations)], depth)
                                      for i in range(depth)))
        self.output_proj = nn.Conv2d(width, group_width, 1)
        # v8, including a trained subband estimator, is unchanged at warm start.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor,
                visual: torch.Tensor) -> torch.Tensor:
        local = self.spectrum_proj(features).permute(0, 3, 1, 2)
        local = local + self.condition_proj(condition).unsqueeze(-1) + self.frequency_positions
        visual_gain, visual_bias = self.visual_proj(visual).unsqueeze(-1).chunk(2, dim=1)
        band_gate = self.visual_band_gate.sigmoid()
        local = local * (1.0 + band_gate * visual_gain.tanh()) + band_gate * visual_bias
        gains = self.output_proj(self.blocks(local)).permute(0, 2, 3, 1).flatten(2)
        return gains[..., :self.bins].transpose(1, 2)


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
    def __init__(self, channels: int = 320, dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                 visual_mode: str = "causal", synthesis_mode: str = "nonnegative",
                 attention_history_chunks: int = 0, subband_refiner: bool = False,
                 subband_width: int = 16, subband_depth: int = 1):
        super().__init__()
        if visual_mode not in ("causal", "chunk_centered"):
            raise ValueError("visual_mode must be causal or chunk_centered")
        if synthesis_mode not in ("nonnegative", "signed_residual"):
            raise ValueError("synthesis_mode must be nonnegative or signed_residual")
        if synthesis_mode == "signed_residual" and visual_mode != "causal":
            raise ValueError("signed_residual requires causal visual mode")
        if type(attention_history_chunks) is not int or attention_history_chunks < 0:
            raise ValueError("attention_history_chunks must be a nonnegative integer")
        if attention_history_chunks and synthesis_mode != "signed_residual":
            raise ValueError("attention history requires signed_residual synthesis")
        if type(subband_refiner) is not bool:
            raise ValueError("subband_refiner must be a boolean")
        if subband_refiner and synthesis_mode != "signed_residual":
            raise ValueError("subband_refiner requires signed_residual synthesis")
        if type(subband_width) is not int or subband_width < 1:
            raise ValueError("subband_width must be a positive integer")
        if type(subband_depth) is not int or subband_depth < 1:
            raise ValueError("subband_depth must be a positive integer")
        self.subband_width = subband_width
        self.subband_depth = subband_depth
        self.has_subband_capacity = (subband_width, subband_depth) != (16, 1)
        if self.has_subband_capacity and not subband_refiner:
            raise ValueError("nondefault subband_width or subband_depth requires subband_refiner")
        self.subband_refiner = subband_refiner
        # Training-only memory option. The frontend output and stream state do
        # not change when checkpointing is enabled.
        self.checkpoint_visual = False
        self.checkpoint_chunks = False
        self.visual_mode = visual_mode
        self.synthesis_mode = synthesis_mode
        self.attention_history_chunks = attention_history_chunks
        # Changed model semantics need a new checkpoint version.
        if self.has_subband_capacity:
            self._expected_format_version = 9
        elif subband_refiner:
            self._expected_format_version = 8
        elif attention_history_chunks:
            self._expected_format_version = 7
        elif synthesis_mode == "signed_residual":
            self._expected_format_version = 6
        else:
            self._expected_format_version = 4 if visual_mode == "causal" else 5
        self.register_buffer("_streaming_format_version",
                             torch.tensor(self._expected_format_version, dtype=torch.int64))
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
        if attention_history_chunks:
            self.history_attention_gate = nn.Parameter(torch.zeros(()))
        self.mask = nn.Conv1d(channels, FFT_SIZE // 2 + 1, 1)
        if synthesis_mode == "signed_residual":
            # A signed real spectrum spans the full 129-tap causal FIR basis
            # after irfft and truncation. Zero residual preserves warm start.
            self.synthesis_residual = nn.Conv1d(channels, FFT_SIZE // 2 + 1, 1)
            nn.init.zeros_(self.synthesis_residual.weight)
            nn.init.zeros_(self.synthesis_residual.bias)
        if subband_refiner:
            self.subband = SubbandRefiner(channels)
        if self.has_subband_capacity:
            self.subband_capacity = SubbandCapacity(
                channels, subband_width, subband_depth, bins=self.subband.bins,
                group_width=self.subband.group_width, groups=self.subband.groups)
        self._freeze_bn_stats()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate even under strict=False: changing synthesis semantics while
        # keeping identical parameter names is not a safe partial load.
        name = prefix + "_streaming_format_version"
        marker = state_dict.get(name)
        expected_version = self._expected_format_version
        if (not isinstance(marker, torch.Tensor) or marker.ndim != 0
                or marker.dtype != torch.int64 or marker.item() != expected_version):
            error_msgs.append(f"{name}: expected scalar int64 model version {expected_version}")
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
        if self.visual_mode == "chunk_centered":
            # At most two frames of context from the past and all three frames
            # already received in this block. Repetition fills missing right
            # context; no visual sample from a future block is consumed.
            visual_input = torch.cat(
                (old[:, -2:], video, video[:, -1:].expand(-1, 2, -1, -1)), dim=1
            )
        else:
            visual_input = frames
        if self.checkpoint_visual and self.training and torch.is_grad_enabled():
            # Non-reentrant checkpointing preserves gradients for trainable
            # frontend parameters even when input frames do not require grad.
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self._visual_backbone, visual_input, use_reentrant=False)
        else:
            x = self._visual_backbone(visual_input)
        return self.video_proj(x), frames[:, -4:]

    def _visual_backbone(self, visual_input: torch.Tensor) -> torch.Tensor:
        x = self.video_model.frontend3D(visual_input.unsqueeze(1))
        batch, _, count, _, _ = x.shape
        x = self.video_model.trunk(threeD_to_2D_tensor(x))
        return x.reshape(batch, count, -1).transpose(1, 2)

    def _attention_with_history(self, joint: torch.Tensor, local: torch.Tensor,
                                cached: tuple[torch.Tensor, torch.Tensor] | None):
        attention = self.chunk_attention
        query, key, value = F.linear(
            joint, attention.in_proj_weight, attention.in_proj_bias
        ).chunk(3, dim=-1)
        if cached is None:
            keys, values = key.contiguous(), value.contiguous()
            long_attention = local
        else:
            old_key, old_value = cached
            if old_key.dtype != key.dtype:
                raise ValueError("attention cache dtype does not match projection dtype")
            keys = torch.cat((old_key, key), dim=1)
            values = torch.cat((old_value, value), dim=1)
            long_attention = F.scaled_dot_product_attention(
                query.unsqueeze(1), keys.unsqueeze(1), values.unsqueeze(1),
                dropout_p=0.0,
            ).squeeze(1)
            long_attention = attention.out_proj(long_attention)
        # The new branch has zero effect until its gate learns to open.
        result = local + self.history_attention_gate * (long_attention - local)
        limit = self.attention_history_chunks * joint.size(1)
        if keys.size(1) > limit:
            keys = keys[:, -limit:].contiguous()
            values = values[:, -limit:].contiguous()
        return result, (keys, values)

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
            fields = {"audio", "video", "blocks"}
            if self.attention_history_chunks:
                fields.add("attention")
            if not isinstance(state, dict) or set(state) != fields:
                raise ValueError("stream state must contain exactly " + ", ".join(sorted(fields)))
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
            if self.attention_history_chunks:
                cached = state["attention"]
                if not isinstance(cached, (list, tuple)) or len(cached) != 2:
                    raise ValueError("attention state must contain cached key and value")
                key, value = cached
                tokens = self.chunk_positions.size(1)
                if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
                        or key.ndim != 3 or key.shape != value.shape
                        or key.size(0) != audio.size(0) or key.size(2) != self.channels
                        or key.size(1) % tokens != 0
                        or not 1 <= key.size(1) // tokens <= self.attention_history_chunks
                        or key.device != audio.device or value.device != audio.device
                        or key.dtype != value.dtype or not key.is_floating_point()):
                    raise ValueError("invalid attention cache geometry, dtype or device")
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
        visual_hops = visual.repeat_interleave(5, dim=-1)
        x = x + visual_hops
        previous = state.get("blocks")
        if previous is None:
            previous = [None] * len(self.blocks)
        if len(previous) != len(self.blocks):
            raise ValueError("wrong number of audio block states")
        next_states = []
        for block, block_state in zip(self.blocks, previous):
            x, block_state = block(x, block_state)
            next_states.append(block_state)
        # Attend within the received block, optionally adding earlier K/V.
        # Neither path uses tokens from subsequent blocks during training.
        joint = torch.cat((x.transpose(1, 2), visual.transpose(1, 2)), dim=1)
        joint = self.chunk_norm(joint + self.chunk_positions)
        attended, _ = self.chunk_attention(joint, joint, joint, need_weights=False)
        next_attention = None
        if self.attention_history_chunks:
            attended, next_attention = self._attention_with_history(joint, attended, state.get("attention"))
        x = x + attended[:, :CHUNK_SAMPLES // FRAME_SAMPLES].transpose(1, 2)
        mask = torch.sigmoid(self.mask(x))
        if self.synthesis_mode == "signed_residual":
            mask = mask + self.synthesis_residual(x)
        if self.subband_refiner:
            features = self.subband.spectral_features(spec)
            mask = mask + self.subband(spec, x, features=features)
            if self.has_subband_capacity:
                mask = mask + self.subband_capacity(features, x, visual_hops)
        reconstructed = causal_mask_filter(spec, mask.transpose(1, 2))
        output = reconstructed.reshape(audio.size(0), CHUNK_SAMPLES)
        next_state = {"audio": all_audio[:, -FRAME_SAMPLES:],
                      "video": video_state, "blocks": next_states}
        if self.attention_history_chunks:
            next_state["attention"] = next_attention
        return output, next_state

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        if audio.ndim != 2 or audio.size(1) == 0 or audio.size(1) % CHUNK_SAMPLES:
            raise ValueError("audio length must be a positive multiple of 1920")
        chunks = audio.size(1) // CHUNK_SAMPLES
        if video.ndim != 4 or video.shape != (audio.size(0), chunks * VIDEO_FRAMES, 88, 88):
            raise ValueError("expected three 88x88 video frames per audio chunk")
        state = None
        outputs = []
        for index in range(chunks):
            audio_chunk = audio[:, index*CHUNK_SAMPLES:(index+1)*CHUNK_SAMPLES]
            video_chunk = video[:, index*VIDEO_FRAMES:(index+1)*VIDEO_FRAMES]
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                # Non-reentrant checkpoint supports nested stream state,
                # preserving backpropagation through previous chunks.
                from torch.utils.checkpoint import checkpoint
                y, state = checkpoint(self.forward_chunk, audio_chunk, video_chunk, state,
                                      use_reentrant=False)
            else:
                y, state = self.forward_chunk(audio_chunk, video_chunk, state)
            outputs.append(y)
        return torch.cat(outputs, dim=-1)
