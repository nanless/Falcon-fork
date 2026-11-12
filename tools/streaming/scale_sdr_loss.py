"""Stable scale-dependent target-to-error SDR and silent-target objective.

For a non-constant target s and estimate y, the per-example training loss is
    10 log10((sum((y-s)**2) + eps) / (sum(s**2) + eps)).
No scale alignment or mean subtraction is performed: this is direct waveform
signal-to-error ratio, also called negative SNR in separation libraries. It is
not BSS Eval SDR, SI-SDR or scale-dependent SDR that retains SI projection.
Constant targets use a finite log1p error relative to mixture power instead.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def _log_energy(wave: torch.Tensor, *, mean: bool) -> torch.Tensor:
    """Log sum or mean square without squaring unscaled float32 extremes."""
    scale = wave.abs().amax(dim=1).clamp_min(1e-20)
    normalized = wave / scale[:, None]
    power = normalized.square().sum(dim=1)
    if mean:
        power = power / wave.shape[1]
    return 2.0 * scale.log() + power.clamp_min(1e-30).log()


def scale_sdr_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    *,
    eps: float = 1e-8,
    silence_rms_threshold: float = 0.0,
    silence_weight: float = 1.0,
):
    """Return per-example losses and masks, differentiable in output.

    The SDR numerator and denominator both have additive eps in energy units,
    bounding perfect reconstruction while retaining gain sensitivity. Constant
    targets and threshold-selected near-constant targets use the stable waveform branch.
    The speech SDR itself does not subtract the mean or align scales.
    """
    if (output.ndim != 2 or output.shape != target.shape or output.shape != mixture.shape
            or output.shape[0] == 0 or output.shape[1] == 0):
        raise ValueError("output, target and mixture must be matching nonempty [batch,time] tensors")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if not math.isfinite(silence_rms_threshold) or silence_rms_threshold < 0:
        raise ValueError("silence_rms_threshold must be nonnegative and finite")
    if not math.isfinite(silence_weight) or silence_weight < 0:
        raise ValueError("silence_weight must be nonnegative and finite")
    output, target, mixture = output.float(), target.float(), mixture.float()
    with torch.no_grad():
        centered = target - target.mean(dim=1, keepdim=True)
        constant = target.amax(dim=1) == target.amin(dim=1)
        degenerate = constant
        silent = target.abs().amax(dim=1) == 0
        if silence_rms_threshold:
            threshold_log = 2.0 * math.log(silence_rms_threshold)
            degenerate = degenerate | (_log_energy(centered, mean=True) <= threshold_log)
            silent = silent | (_log_energy(target, mean=True) <= threshold_log)
        speech = ~degenerate
    values = output.new_zeros(output.shape[0])
    log_eps = math.log(eps)
    db = 10.0 / math.log(10.0)
    if speech.any():
        t = target[speech]
        error = output[speech] - t
        target_log = _log_energy(t, mean=False)
        error_log = _log_energy(error, mean=False)
        eps_t = target_log.new_full(target_log.shape, log_eps)
        target_smoothed = torch.logaddexp(target_log, eps_t)
        error_smoothed = torch.logaddexp(error_log, eps_t)
        losses = db * (error_smoothed - target_smoothed)
        values = values.index_copy(0, speech.nonzero(as_tuple=True)[0], losses)
    if degenerate.any():
        t = target[degenerate]
        error_log = _log_energy(output[degenerate] - t, mean=True)
        mix_log = _log_energy(mixture[degenerate], mean=True)
        floor = mix_log.new_full(mix_log.shape, math.log(1e-8))
        denominator_log = torch.maximum(mix_log, floor)
        losses = db * F.softplus(error_log - denominator_log) * silence_weight
        values = values.index_copy(0, degenerate.nonzero(as_tuple=True)[0], losses)
    return values, {"speech": speech, "degenerate": degenerate, "silent": silent}
