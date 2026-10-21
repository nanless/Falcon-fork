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

CSV schema is unchanged for downstream compatibility::

    snt_id, sdr, sdr_i, si-snr, si-snr_i, pesq, stoi        (+ avg/std rows)

All values in the CSV are positive dB scores (higher is better), including the
per-utterance rows and the ``avg`` / ``std`` rows.
"""

import csv
import logging

import numpy as np
import torch

from . import core

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"]


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
            "pesq": pesq,
            "stoi": stoi,
        }

    # -- public API ---------------------------------------------------------
    def __call__(self, mix, clean, estimate, key):
        metrics = self._compute(mix, clean, estimate)
        row = {"snt_id": str(key), **metrics}
        self.writer.writerow(row)
        self.all_sdrs.append(metrics["sdr"])
        self.all_sdrs_i.append(metrics["sdr_i"])
        self.all_sisnrs.append(metrics["si-snr"])
        self.all_sisnrs_i.append(metrics["si-snr_i"])
        self.all_pesqs.append(metrics["pesq"])
        self.all_stois.append(metrics["stoi"])
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

    def get_mean(self):
        return {k: self._mean(getattr(self, attr)) for k, attr in (
            ("sdr", "all_sdrs"), ("sdr_i", "all_sdrs_i"),
            ("si-snr", "all_sisnrs"), ("si-snr_i", "all_sisnrs_i"),
            ("pesq", "all_pesqs"), ("stoi", "all_stois"),
        )}

    def get_std(self):
        return {k: self._std(getattr(self, attr)) for k, attr in (
            ("sdr", "all_sdrs"), ("sdr_i", "all_sdrs_i"),
            ("si-snr", "all_sisnrs"), ("si-snr_i", "all_sisnrs_i"),
            ("pesq", "all_pesqs"), ("stoi", "all_stois"),
        )}

    def final(self):
        avg = self.get_mean()
        std = self.get_std()
        self.writer.writerow({"snt_id": "avg", **avg})
        self.writer.writerow({"snt_id": "std", **std})
        self.results_csv.close()

    # convenience for callers that forget to close explicitly
    def close(self):
        if not self.results_csv.closed:
            self.final()


__all__ = ["ALLMetricsTracker", "CSV_COLUMNS"]
