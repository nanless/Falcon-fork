"""Phase-sensitive, level-normalized spectral error for student training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MultiResolutionSpectralLoss(nn.Module):
    """Average complex STFT error across resolutions and return one term per example.

    Excluded rows return exact zeros. The trainer can average the full batch to
    keep its DDP weighting unchanged. This loss adds no inference state.
    """

    def __init__(self, fft_sizes: Sequence[int] = (256, 512, 1024)):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        if not self.fft_sizes or any(type(n) is not int or n < 4 or n % 4 for n in self.fft_sizes):
            raise ValueError("STFT sizes must be positive multiples of four")
        for index, size in enumerate(self.fft_sizes):
            window = torch.hann_window(size, dtype=torch.float32)
            self.register_buffer(f"window_{index}", window / window.norm(), persistent=False)

    def forward(self, output: Tensor, target: Tensor, mixture: Tensor, speech_mask: Tensor) -> Tensor:
        if output.ndim != 2 or output.shape != target.shape or output.shape != mixture.shape:
            raise ValueError("spectral loss requires matching [batch, time] waveforms")
        if output.shape[1] < max(self.fft_sizes):
            raise ValueError("training waveform is shorter than the largest STFT window")
        if speech_mask.shape != (output.shape[0],) or speech_mask.dtype != torch.bool:
            raise ValueError("speech_mask must be boolean with one value per waveform")

        # A mixture-relative floor limits the influence of exceptionally quiet
        # targets in loud mixtures, and the absolute floor bounds the gain.
        clean = target.detach().float()
        mixture_level = mixture.detach().float().square().mean(dim=1).sqrt()
        clean_level = clean.square().mean(dim=1).sqrt()
        level = torch.maximum(clean_level, 0.05 * mixture_level).clamp_min(1e-3)
        error = (output.float() - clean) / level[:, None]
        error = torch.where(speech_mask[:, None], error, 0.0)

        # STFT is linear. Analyze the difference once per resolution, keeping
        # phase and scale information without a second forward FFT of the target.
        terms = []
        for index, size in enumerate(self.fft_sizes):
            spectrum = torch.stft(
                error, n_fft=size, hop_length=size // 4,
                window=getattr(self, f"window_{index}"), center=False,
                return_complex=True,
            )
            terms.append(spectrum.abs().mean(dim=(1, 2)))
        return torch.stack(terms).mean(dim=0)
