"""Stable speech and constant-target reconstruction objective.

Nonconstant targets retain negative SI-SDR. All targets use 10*log10(1+normalized squared waveform error), computed
without square overflow. The waveform term is full-weight for constant targets
and auxiliary for speech targets, which retain negative SI-SDR.
"""
import math
import torch
from torch.nn import functional as F
from tools.streaming.train_ddp import target_degeneracy_masks


def robust_reconstruction(output, target, mixture, *, silence_rms_threshold=0.0, silence_weight=1.0, waveform_weight=0.1, ratio_floor=1e-8):
    if output.ndim != 2 or output.shape != target.shape or target.shape != mixture.shape or not output.numel():
        raise ValueError("output, target, mixture must be nonempty matching [batch,time] tensors")
    if silence_rms_threshold < 0 or silence_weight < 0 or waveform_weight < 0 or ratio_floor <= 0:
        raise ValueError("threshold and silence weight must be nonnegative")
    output, target, mixture = output.float(), target.float(), mixture.float()
    degenerate, silent = target_degeneracy_masks(target, silence_rms_threshold)
    values = output.new_zeros(output.shape[0])
    speech_mask = ~degenerate
    if speech_mask.any():
        est = output[speech_mask]
        ref = target[speech_mask]
        est = est - est.mean(dim=1, keepdim=True)
        ref = ref - ref.mean(dim=1, keepdim=True)
        # Rescale each centered vector before squaring: SI-SDR is invariant to
        # these positive scales, while numerical overflow is avoided.
        est_scale = est.abs().amax(dim=1, keepdim=True).clamp_min(1e-20)
        ref_scale = ref.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-20)
        est, ref = est / est_scale, ref / ref_scale
        projection = ((est * ref).sum(dim=1, keepdim=True) / ref.square().sum(dim=1, keepdim=True).clamp_min(1e-20)) * ref
        proj_power = projection.square().sum(dim=1)
        noise_power = (est - projection).square().sum(dim=1)
        # The floor bounds both perfect projection and zero projection.
        ratio = proj_power / (noise_power + ratio_floor * proj_power + ratio_floor)
        speech = -10 * torch.log10(ratio + ratio_floor)
        values = values.index_copy(0, speech_mask.nonzero().reshape(-1), speech)
    if degenerate.any() or (waveform_weight and speech_mask.any()):
        residual = output - target
        scale = residual.abs().amax(dim=1).clamp_min(1e-20)
        unit_mse = (residual / scale[:, None]).square().mean(dim=1).clamp_min(1e-30)
        mix = mixture
        denominator = mix.square().mean(dim=1).clamp_min(1e-8)
        log_ratio = 2 * torch.log(scale) + torch.log(unit_mse) - torch.log(denominator)
        waveform = (10.0 / math.log(10.0)) * F.softplus(log_ratio)
        if degenerate.any():
            values = values.index_copy(0, degenerate.nonzero().reshape(-1), waveform[degenerate] * silence_weight)
        if waveform_weight and speech_mask.any():
            values = values.index_add(0, speech_mask.nonzero().reshape(-1), waveform[speech_mask] * waveform_weight)
    return values, {"speech": speech_mask, "degenerate": degenerate, "silent": silent}
