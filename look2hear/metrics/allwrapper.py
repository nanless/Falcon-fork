"""Metrics tracker used by the evaluation entry point (``audio_visual_test.py``).

Fixes (Phase1/T3, see docs/REPRODUCTION_PLAN.md §6.2 F7/F8):

* the old ``sdr`` / ``sdr_i`` columns were actually ``-SNR`` / ``SNRi`` computed
  with ``PairwiseNegSDR("snr")``; they are now real SDR / SDRi computed with
  ``fast_bss_eval.sdr_pit_loss(..., filter_length=512)``;
* the old per-utterance rows and the ``avg`` / ``std`` rows had opposite signs
  (the row wrote ``sdr.item()`` while accumulation stored ``-sdr.item()``);
  now **all values are positive, higher-is-better, and row == aggregation**;
* PESQ was called as ``pesq(estimate, clean, fs)`` while ``pypesq`` is
  ``pesq(ref, deg, fs)``; the call is now ``pesq(clean, estimate, fs)``;
* SI-SDR uses the standard zero-mean definition through
  :mod:`look2hear.metrics.core` (the old tracker used ``zero_mean=False``).

A4-3 / A4-4 additions (2026-10-30, appended columns only; existing order kept)::

    snt_id, sdr, sdr_i, si-snr, si-snr_i[, si-snr_mix, sdr_mix], pesq, stoi,
    n_sdr, n_sdr_i, n_si-snr, n_si-snr_i, n_pesq, n_stoi, n_si-snr_mix, n_sdr_mix

* ``si-snr_mix`` / ``sdr_mix``: the mixture-baseline scores, so that
  ``si-snr_i == si-snr - si-snr_mix`` can be re-derived offline (A4-4).
* ``n_*``: number of finite values that contribute to the row's aggregate.
  Per-utterance rows leave them empty; ``avg`` / ``std`` rows carry the true
  valid counts (A4-3: never silently ``nanmean`` without knowing ``n``).

All values in the CSV are positive dB scores (higher is better), including the
per-utterance rows and the ``avg`` / ``std`` rows.
"""

import csv
import logging

import numpy as np
import torch

from . import core

logger = logging.getLogger(__name__)

#: Metric columns whose finite-value counts are tracked (A4-3).
METRIC_COLUMNS = [
    "sdr",
    "sdr_i",
    "si-snr",
    "si-snr_i",
    "pesq",
    "stoi",
    "si-snr_mix",
    "sdr_mix",
]
COUNT_COLUMNS = [f"n_{c}" for c in METRIC_COLUMNS]

#: Full CSV schema. Existing columns keep their original order/position; the
#: A4-3/A4-4 columns (mixture baselines + valid counts) are appended at the end.
CSV_COLUMNS = [
    "snt_id",
    "sdr",
    "sdr_i",
    "si-snr",
    "si-snr_i",
    "pesq",
    "stoi",
    "si-snr_mix",
    "sdr_mix",
    *COUNT_COLUMNS,
]


class ALLMetricsTracker:
    """Accumulate per-utterance separation metrics and write a CSV.

    ``__call__(mix, clean, estimate, key)`` accepts ``[T]`` / ``[S, T]`` (or
    ``[1, S, T]``) tensors or numpy arrays. All reported values are positive
    dB scores (higher is better); PESQ/STOI are only computed for ``n_src == 1``
    and are ``nan`` otherwise.
    """

    def __init__(self, save_file: str = "", sample_rate: int = core.PESQ_SAMPLE_RATE,
                 sdr_filter_length: int = core.SDR_FILTER_LENGTH):
        self.sample_rate = int(sample_rate)
        self.sdr_filter_length = int(sdr_filter_length)
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        self.all_pesqs = []
        self.all_stois = []
        self.all_sisnrs_mix = []
        self.all_sdrs_mix = []
        self._warned_multisrc = False

        self.results_csv = open(save_file, "w", newline="")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()

    # -- metric computation -------------------------------------------------
    def _compute(self, mix, clean, estimate):
        clean_src = core._as_source_matrix(clean)
        est_src = core._as_source_matrix(estimate)
        n_src = clean_src.shape[0]

        si_sdr = core.si_sdr(est_src, clean_src)
        si_sdr_mix = core.si_sdr(core._mixture_sources(mix, n_src), clean_src)
        sisnr_i = si_sdr - si_sdr_mix

        sdr = core.sdr(est_src, clean_src, filter_length=self.sdr_filter_length)
        sdr_mix = core.sdr(core._mixture_sources(mix, n_src), clean_src,
                           filter_length=self.sdr_filter_length)
        sdr_i = sdr - sdr_mix

        if n_src == 1:
            pesq = core.pesq_metric(clean_src[0], est_src[0], self.sample_rate)
            stoi = core.stoi_metric(clean_src[0], est_src[0], self.sample_rate)
        else:
            if not self._warned_multisrc:
                logger.warning("PESQ/STOI are only defined for n_src=1; recording nan for n_src=%d", n_src)
                self._warned_multisrc = True
            pesq = stoi = float("nan")
        return {
            "sdr": sdr,
            "sdr_i": sdr_i,
            "si-snr": si_sdr,
            "si-snr_i": sisnr_i,
            "si-snr_mix": si_sdr_mix,
            "sdr_mix": sdr_mix,
            "pesq": pesq,
            "stoi": stoi,
        }

    # -- public API ---------------------------------------------------------
    def __call__(self, mix, clean, estimate, key):
        metrics = self._compute(mix, clean, estimate)
        # Per-utterance rows carry the values only; counts are aggregate-row metadata.
        row = {"snt_id": str(key), **metrics, **{c: "" for c in COUNT_COLUMNS}}
        self.writer.writerow(row)
        self.all_sdrs.append(metrics["sdr"])
        self.all_sdrs_i.append(metrics["sdr_i"])
        self.all_sisnrs.append(metrics["si-snr"])
        self.all_sisnrs_i.append(metrics["si-snr_i"])
        self.all_pesqs.append(metrics["pesq"])
        self.all_stois.append(metrics["stoi"])
        self.all_sisnrs_mix.append(metrics["si-snr_mix"])
        self.all_sdrs_mix.append(metrics["sdr_mix"])
        return row

    @staticmethod
    def _mean(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    @staticmethod
    def _std(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanstd(arr))

    def _value_attr(self):
        return (
            ("sdr", "all_sdrs"), ("sdr_i", "all_sdrs_i"),
            ("si-snr", "all_sisnrs"), ("si-snr_i", "all_sisnrs_i"),
            ("pesq", "all_pesqs"), ("stoi", "all_stois"),
            ("si-snr_mix", "all_sisnrs_mix"), ("sdr_mix", "all_sdrs_mix"),
        )

    def get_mean(self):
        return {k: self._mean(getattr(self, attr)) for k, attr in self._value_attr()}

    def get_std(self):
        return {k: self._std(getattr(self, attr)) for k, attr in self._value_attr()}

    def get_counts(self):
        """Number of finite (non-nan) values accumulated per metric column (A4-3)."""
        counts = {}
        for k, attr in self._value_attr():
            arr = np.asarray(getattr(self, attr), dtype=np.float64)
            counts[f"n_{k}"] = int(np.count_nonzero(np.isfinite(arr)))
        return counts

    def final(self):
        avg = self.get_mean()
        std = self.get_std()
        counts = self.get_counts()
        # counts are identical for avg/std (same accumulation); expose on both rows.
        self.writer.writerow({"snt_id": "avg", **avg, **counts})
        self.writer.writerow({"snt_id": "std", **std, **counts})
        self.results_csv.close()

    # convenience for callers that forget to close explicitly
    def close(self):
        if not self.results_csv.closed:
            self.final()


__all__ = ["ALLMetricsTracker", "CSV_COLUMNS", "METRIC_COLUMNS", "COUNT_COLUMNS"]
