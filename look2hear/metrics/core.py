"""Sign-consistent, definition-explicit audio separation metrics.

This module is the single source of truth for the metrics used by
``audio_visual_test.py`` and :class:`look2hear.metrics.ALLMetricsTracker`.
The original repository mixed several conventions (the old ``sdr`` column was
actually ``-SNR``, per-utterance rows and ``avg`` rows had opposite signs, and
PESQ was called with ``(estimate, clean)`` instead of ``(clean, estimate)``).
Everything here follows one convention: **scores are positive, higher is better**.

Definitions (all in dB unless noted)
------------------------------------
SI-SDR (scale-invariant signal-to-distortion ratio)::

    s_hat = estimate - mean(estimate)
    s     = clean    - mean(clean)
    alpha = <s_hat, s> / (||s||^2 + eps)
    SI-SDR = 10 * log10( ||alpha * s||^2 / (||s_hat - alpha * s||^2 + eps) )

with PIT (permutation invariant training) over the source axis when more than
one source is provided.

SI-SDRi = SI-SDR(estimate) - SI-SDR(mixture)

SDR (BSS-Eval v4 style, multi-tap filter, PIT over sources; fast_bss_eval)::

    SDR = -fast_bss_eval.sdr_pit_loss(estimate, clean, filter_length=512)

``fast_bss_eval`` returns the *negative* SDR (a loss), so it is negated here.
``filter_length=512`` matches the historical ``MetricsTracker`` call and
BSS-Eval's default filter length.

SDRi = SDR(estimate) - SDR(mixture)

PESQ: ``pesq(ref=clean, deg=estimate, fs=16000)`` (pypesq argument order).
STOI: ``stoi(x=clean, y=estimate, fs=16000, extended=False)``.

The helper functions accept ``[T]``, ``[S, T]`` or ``[1, S, T]`` tensors /
numpy arrays and always compute in float64 on the CPU for determinism.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

try:  # metric backends are optional so the module can be imported in minimal envs
    import fast_bss_eval
except Exception:  # pragma: no cover - environment dependent
    fast_bss_eval = None

try:
    from pypesq import pesq as _pesq
except Exception:  # pragma: no cover
    _pesq = None

try:
    from pystoi import stoi as _stoi
except Exception:  # pragma: no cover
    _stoi = None

from ..losses import PITLossWrapper, PairwiseNegSDR

logger = logging.getLogger(__name__)

EPS = 1e-8
SDR_FILTER_LENGTH = 512
PESQ_SAMPLE_RATE = 16000

#: PIT-wrapped negative SI-SDR. ``threshold_byloss=False`` is essential here:
#: the training wrapper's default drops samples with SDR > 30 dB, which would
#: corrupt the *metric* (a near-perfect estimate must score high, not become nan).
PIT_SI_SDR = PITLossWrapper(
    PairwiseNegSDR("sisdr", zero_mean=True),
    pit_from="pw_mtx",
    threshold_byloss=False,
)


def _to_float64_cpu(x) -> torch.Tensor:
    """Convert numpy / torch input to a detached float64 CPU tensor."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.detach().cpu().to(torch.float64)


def _as_batched_sources(x) -> torch.Tensor:
    """Return ``[1, S, T]`` float64 tensor from ``[T]``, ``[S, T]`` or ``[1, S, T]``."""
    x = _to_float64_cpu(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"expected [T], [S, T] or [B, S, T] audio, got shape {tuple(x.shape)}")
    return x


def _as_source_matrix(x) -> torch.Tensor:
    """Return ``[S, T]`` float64 CPU tensor (the tracker's canonical layout)."""
    x = _to_float64_cpu(x)
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"expected [T], [S, T] or [1, S, T] audio, got shape {tuple(x.shape)}")


def _prepare_pair(estimate, clean):
    """Broadcast estimate/clean to matching ``[1, S, T]`` float64 tensors."""
    est = _as_batched_sources(estimate)
    ref = _as_batched_sources(clean)
    if ref.shape[1] == 1 and est.shape[1] > 1:
        ref = ref.expand(-1, est.shape[1], -1)
    if est.shape[1] == 1 and ref.shape[1] > 1:
        est = est.expand(-1, ref.shape[1], -1)
    if ref.shape[0] == 1 and est.shape[0] > 1:
        ref = ref.expand(est.shape[0], -1, -1)
    if est.shape[0] == 1 and ref.shape[0] > 1:
        est = est.expand(ref.shape[0], -1, -1)
    if est.shape != ref.shape:
        raise ValueError(f"estimate/target shape mismatch: {tuple(est.shape)} vs {tuple(ref.shape)}")
    return est, ref


def _mixture_sources(mixture, n_src: int) -> torch.Tensor:
    """Return ``[n_src, T]`` float64 tensor; a mono mixture is duplicated per source."""
    mix = _as_source_matrix(mixture)
    if mix.shape[0] == 1 and n_src > 1:
        mix = mix.expand(n_src, -1)
    if mix.shape[0] != n_src:
        raise ValueError(
            f"mixture has {mix.shape[0]} channel(s), expected {n_src} (clean has {n_src} source(s))"
        )
    return mix


def _nan(msg: str, *args) -> float:
    logger.warning(msg, *args)
    return float("nan")


# ---------------------------------------------------------------------------
# SI-SDR / SI-SDRi
# ---------------------------------------------------------------------------
def si_sdr(estimate, clean) -> float:
    """Positive SI-SDR in dB (PIT over sources, zero-mean, higher is better)."""
    est, ref = _prepare_pair(estimate, clean)
    return -float(PIT_SI_SDR(est, ref))


def si_sdr_improvement(estimate, clean, mixture) -> float:
    """SI-SDRi = SI-SDR(estimate) - SI-SDR(mixture), positive dB."""
    est, ref = _prepare_pair(estimate, clean)
    mix = _mixture_sources(mixture, ref.shape[1])
    return -float(PIT_SI_SDR(est, ref)) - (-float(PIT_SI_SDR(mix.unsqueeze(0), ref)))


# ---------------------------------------------------------------------------
# SDR / SDRi (fast_bss_eval, filter_length=512)
# ---------------------------------------------------------------------------
def sdr(estimate, clean, filter_length: int = SDR_FILTER_LENGTH) -> float:
    """Positive SDR in dB (fast_bss_eval BSS-Eval-v4 style, PIT over sources)."""
    if fast_bss_eval is None:  # pragma: no cover - environment dependent
        return _nan("fast_bss_eval is not installed; SDR will be nan")
    est, ref = _prepare_pair(estimate, clean)
    try:
        loss = fast_bss_eval.sdr_pit_loss(est, ref, filter_length=filter_length)
        return -float(loss.mean())
    except Exception as exc:  # too-short / degenerate signals
        return _nan("SDR computation failed (%s); recording nan", exc)


def sdr_improvement(estimate, clean, mixture, filter_length: int = SDR_FILTER_LENGTH) -> float:
    """SDRi = SDR(estimate) - SDR(mixture), positive dB."""
    est_sdr = sdr(estimate, clean, filter_length=filter_length)
    mix = _mixture_sources(mixture, _as_source_matrix(clean).shape[0])
    mix_sdr = sdr(mix, clean, filter_length=filter_length)
    return est_sdr - mix_sdr


# ---------------------------------------------------------------------------
# PESQ / STOI (single-source only)
# ---------------------------------------------------------------------------
def _as_1d_numpy(x) -> np.ndarray:
    arr = np.asarray(_to_float64_cpu(x).numpy(), dtype=np.float64).reshape(-1)
    return np.ascontiguousarray(arr)


def pesq_metric(clean, estimate, sample_rate: int = PESQ_SAMPLE_RATE) -> float:
    """PESQ in MOS (higher is better); argument order is ``(ref=clean, deg=estimate)``."""
    if _pesq is None:  # pragma: no cover
        return _nan("pypesq/pesq backend is not installed; PESQ will be nan")
    ref = _as_1d_numpy(clean)
    deg = _as_1d_numpy(estimate)
    try:
        return float(_pesq(ref, deg, int(sample_rate)))
    except Exception as exc:  # e.g. pesq.BufferTooShortError for < 0.25 s
        return _nan("PESQ computation failed (%s); recording nan", exc)


def stoi_metric(clean, estimate, sample_rate: int = PESQ_SAMPLE_RATE) -> float:
    """STOI in [0, 1] (higher is better); argument order is ``(clean, estimate)``."""
    if _stoi is None:  # pragma: no cover
        return _nan("pystoi backend is not installed; STOI will be nan")
    ref = _as_1d_numpy(clean)
    deg = _as_1d_numpy(estimate)
    try:
        return float(_stoi(ref, deg, int(sample_rate), extended=False))
    except Exception as exc:
        return _nan("STOI computation failed (%s); recording nan", exc)


__all__ = [
    "SDR_FILTER_LENGTH",
    "PESQ_SAMPLE_RATE",
    "si_sdr",
    "si_sdr_improvement",
    "sdr",
    "sdr_improvement",
    "pesq_metric",
    "stoi_metric",
]
