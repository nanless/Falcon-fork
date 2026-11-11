"""Training-only reconstruction on overlapping complete 120 ms chunk windows."""

from __future__ import annotations

import torch
from torch import Tensor, nn


WINDOW_SAMPLES = 7680
HOP_SAMPLES = 1920
MIN_TARGET_POWER = 1e-6  # Centered RMS of 0.001.
MIN_TARGET_TO_MIXTURE = 1e-2  # -20 dB.
MIN_ESTIMATE_POWER_RELATIVE_TO_TARGET = 1e-2
PROJECTION_RATIO_FLOOR = 1e-2
RESIDUAL_FRACTION_FLOOR = 1e-3


def _centered_windows(waveform: Tensor) -> Tensor:
    windows = waveform.float().unfold(1, WINDOW_SAMPLES, HOP_SAMPLES)
    return windows - windows.mean(dim=-1, keepdim=True)


class LocalReconstructionLoss(nn.Module):
    """Return one reliability-masked 480 ms reconstruction term per example.

    Each window covers four complete chunks. Weak target windows contribute zero,
    and all complete windows count in the denominator so batch weighting stays
    fixed under DDP. The loss adds no inference state or extra model inputs.
    """

    def forward(self, output: Tensor, target: Tensor, mixture: Tensor) -> Tensor:
        if output.ndim != 2 or output.shape != target.shape or output.shape != mixture.shape:
            raise ValueError("local loss requires matching [batch, time] waveforms")
        if output.shape[1] < WINDOW_SAMPLES:
            raise ValueError("local loss requires at least one complete 480 ms window")

        clean = _centered_windows(target.detach())
        clean_power = clean.square().mean(dim=-1)
        noisy = _centered_windows(mixture.detach())
        noisy_power = noisy.square().mean(dim=-1)
        reliable = (clean_power >= MIN_TARGET_POWER) & (
            clean_power >= MIN_TARGET_TO_MIXTURE * noisy_power
        )

        estimate = _centered_windows(output)
        estimate_power = estimate.square().mean(dim=-1)
        # An estimate near silence has no stable direction. A variance floor
        # prevents division by an arbitrarily small estimate power.
        min_estimate_power = (MIN_ESTIMATE_POWER_RELATIVE_TO_TARGET * clean_power).clamp_min(
            MIN_TARGET_POWER
        )
        covariance = (estimate * clean).mean(dim=-1)
        projection_fraction = covariance.square() / (
            clean_power.clamp_min(MIN_TARGET_POWER)
            * estimate_power.clamp_min(min_estimate_power)
        )
        projection_fraction = projection_fraction.clamp(min=0.0, max=1.0)
        residual_fraction = 1.0 - projection_fraction
        # A soft lower bound on each end of the ratio limits gradients near
        # orthogonal and nearly exact predictions. The global metric is untouched.
        ratio = projection_fraction / (residual_fraction + RESIDUAL_FRACTION_FLOOR)
        error = -10.0 * torch.log10(ratio + PROJECTION_RATIO_FLOOR)
        return torch.where(reliable, error, 0.0).mean(dim=1)
