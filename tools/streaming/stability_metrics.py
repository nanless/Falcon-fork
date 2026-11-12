"""Aggregate sample-class losses and output amplitude without autograd.

All outputs represent sums/maxima, never averages of rank averages. Degenerate
includes silent, so these are overlapping diagnostic categories. Caller may
accumulate multiple local batches with merge and then call distributed_reduce
once; all ranks must participate in the same collectives.
"""
from __future__ import annotations
import torch
import torch.distributed as dist

CATEGORIES = ("speech", "degenerate", "silent")
SUM_KEYS = ("count",) + tuple(k for n in CATEGORIES for k in (n + "_count", n + "_loss_sum"))
MAX_KEYS = ("output_abs_max", "output_mix_rms_ratio_max") + tuple(n + "_loss_max" for n in CATEGORIES)


def empty(device):
    return (torch.zeros(len(SUM_KEYS), dtype=torch.float64, device=device),
            torch.full((len(MAX_KEYS),), float("-inf"), dtype=torch.float64, device=device))


def collect(loss, masks, output, mix):
    """Create detached GPU-local metrics for a batch with [B,T] audio."""
    if output.ndim != 2 or output.shape != mix.shape or loss.shape != (output.shape[0],):
        raise ValueError("expected loss [B] and matching output/mix [B,T]")
    if output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError("empty batch is invalid")
    if any(k not in masks or masks[k].shape != loss.shape for k in CATEGORIES):
        raise ValueError("class masks must contain speech, degenerate and silent [B]")
    with torch.no_grad():
        loss = loss.detach().double()
        output = output.detach().float()
        mix = mix.detach().float()
        result = empty(output.device)
        sums, maxima = result
        sums[0] = loss.numel()
        for i, category in enumerate(CATEGORIES):
            selected = masks[category].detach().to(device=output.device, dtype=torch.bool)
            sums[1 + i*2] = selected.sum()
            sums[2 + i*2] = loss.masked_fill(~selected, 0).sum()
            maxima[2+i] = loss.masked_fill(~selected, float("-inf")).max()
        # Amplitude-safe RMS: normalize before squaring to avoid overflow.
        out_scale = output.abs().amax(dim=1).clamp_min(1e-20)
        mix_scale = mix.abs().amax(dim=1).clamp_min(1e-20)
        out_rms = out_scale.double() * (output / out_scale[:, None]).square().mean(dim=1).double().sqrt()
        mix_rms = mix_scale.double() * (mix / mix_scale[:, None]).square().mean(dim=1).double().sqrt()
        maxima[0] = output.abs().max().double()
        maxima[1] = (out_rms / mix_rms.clamp_min(1e-4)).max()
        return result


def merge(left, right):
    if left[0].shape != right[0].shape or left[1].shape != right[1].shape:
        raise ValueError("incompatible metric shapes")
    return (left[0] + right[0], torch.maximum(left[1], right[1]))


def distributed_reduce(metrics):
    sums, maxima = (x.clone() for x in metrics)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    return sums, maxima


def summarize(metrics):
    sums, maxima = (x.detach().cpu().tolist() for x in metrics)
    result = {"count": int(sums[0]), "output_abs_max": maxima[0],
              "output_mix_rms_ratio_max": maxima[1]}
    for i, category in enumerate(CATEGORIES):
        count = int(sums[1+i*2])
        result[category + "_count"] = count
        result[category + "_loss_mean"] = sums[2+i*2] / count if count else None
        result[category + "_loss_max"] = maxima[2+i] if count else None
    return result
