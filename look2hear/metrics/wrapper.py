"""Legacy metric tracker (not used by ``audio_visual_test.py``).

Kept for API compatibility. Phase1/T3 fixed its semantics to match
:mod:`look2hear.metrics.allwrapper` and :mod:`look2hear.metrics.core`:

* ``sdr`` / ``sdr_i`` are real SDR / SDRi (``fast_bss_eval``, filter length 512),
  not ``-SNR``;
* SI-SDR uses the standard zero-mean definition (the old code passed
  ``zero_mean=False``);
* every CSV value (per-utterance rows and ``avg`` / ``std``) is positive and
  higher-is-better, with row == aggregation consistency.

The unused ``torch_mir_eval`` import (a hard dependency for a dead code path)
has been dropped.
"""

import csv
import logging

import numpy as np

from . import core

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]


class MetricsTracker:
    def __init__(self, save_file: str = "", sample_rate: int = core.PESQ_SAMPLE_RATE,
                 sdr_filter_length: int = core.SDR_FILTER_LENGTH):
        self.sample_rate = int(sample_rate)
        self.sdr_filter_length = int(sdr_filter_length)
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        self.results_csv = open(save_file, "w", newline="")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()

    def __call__(self, mix, clean, estimate, key):
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

        row = {
            "snt_id": str(key),
            "sdr": sdr,
            "sdr_i": sdr_i,
            "si-snr": si_sdr,
            "si-snr_i": sisnr_i,
        }
        self.writer.writerow(row)
        self.all_sdrs.append(sdr)
        self.all_sdrs_i.append(sdr_i)
        self.all_sisnrs.append(si_sdr)
        self.all_sisnrs_i.append(sisnr_i)

    def update(self):
        return {
            "sdr_i": float(np.nanmean(self.all_sdrs_i)) if self.all_sdrs_i else float("nan"),
            "si-snr_i": float(np.nanmean(self.all_sisnrs_i)) if self.all_sisnrs_i else float("nan"),
        }

    def final(self):
        def _mean(values):
            arr = np.asarray(values, dtype=np.float64)
            return float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")

        def _std(values):
            arr = np.asarray(values, dtype=np.float64)
            return float(np.nanstd(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")

        avg = {
            "snt_id": "avg",
            "sdr": _mean(self.all_sdrs),
            "sdr_i": _mean(self.all_sdrs_i),
            "si-snr": _mean(self.all_sisnrs),
            "si-snr_i": _mean(self.all_sisnrs_i),
        }
        std = {
            "snt_id": "std",
            "sdr": _std(self.all_sdrs),
            "sdr_i": _std(self.all_sdrs_i),
            "si-snr": _std(self.all_sisnrs),
            "si-snr_i": _std(self.all_sisnrs_i),
        }
        self.writer.writerow(avg)
        self.writer.writerow(std)
        self.results_csv.close()


__all__ = ["MetricsTracker", "CSV_COLUMNS"]
