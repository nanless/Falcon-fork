"""Phase1/T3 unit tests for the corrected separation metrics.

Checks (docs/REPRODUCTION_PLAN.md §6.2 F7/F8):

* SI-SDR / SI-SDRi match analytic values for orthogonal-noise signals (±0.05 dB);
* SDR / SDRi match analytic values (fast_bss_eval, filter_length=512; ±0.05 dB
  for SDR on a long enough signal) and the improvement has the right sign;
* PESQ is called as ``pesq(clean, estimate, fs)`` and STOI as
  ``stoi(clean, estimate, fs)`` (argument-order regression test);
* the CSV emitted by ``ALLMetricsTracker`` has per-utterance rows and ``avg`` /
  ``std`` rows with the *same* positive (higher-is-better) convention.

Run with::

    cd /root/code/github_repos/Falcon-fork
    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_metrics.py -q
"""

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo>/tools/repro/tests/<file>
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from look2hear.metrics import ALLMetricsTracker, core  # noqa: E402
from look2hear.losses import PairwiseNegSDR  # noqa: E402

SR = 16000
TOL_DB = 0.05


# ---------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------
def _orthogonal_noise(rng, signal):
    """Zero-mean white noise orthogonal to ``signal`` so the analytic SNR/SDR is exact.

    (The metric uses zero-mean SI-SDR, so the noise is mean-removed before and
    after the projection; this keeps the analytic ratio exact to float64
    precision.)
    """
    noise = rng.standard_normal(signal.shape)
    noise -= noise.mean()
    noise -= (noise @ signal) / (signal @ signal) * signal
    noise -= noise.mean()
    return noise


def _db_ratio(num, den):
    return 10.0 * math.log10(num / den)


def _make_case(T, alpha, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(T)
    target -= target.mean()
    noise = _orthogonal_noise(rng, target)
    # normalize the noise energy to the target energy: SNR = -20*log10(alpha) exactly
    noise *= np.sqrt((target @ target) / (noise @ noise))
    estimate = target + alpha * noise
    return target, noise, estimate


def _expected_snr_db(target, noise, alpha):
    return _db_ratio(target @ target, (alpha ** 2) * (noise @ noise))


# ---------------------------------------------------------------------------
# SI-SDR / SI-SDRi
# ---------------------------------------------------------------------------
def test_si_sdr_and_improvement_match_analytic():
    T = 32000
    target, noise, estimate = _make_case(T, alpha=0.1, seed=1)   # 20 dB
    mix = target + noise                                          # 0 dB (same noise, unit weight)

    expected_est = _expected_snr_db(target, noise, 0.1)
    expected_mix = _expected_snr_db(target, noise, 1.0)
    assert expected_est == pytest.approx(20.0, abs=1e-9)
    assert expected_mix == pytest.approx(0.0, abs=1e-9)
    assert core.si_sdr(estimate, target) == pytest.approx(expected_est, abs=TOL_DB)
    assert core.si_sdr(mix, target) == pytest.approx(expected_mix, abs=TOL_DB)
    assert core.si_sdr_improvement(estimate, target, mix) == pytest.approx(
        expected_est - expected_mix, abs=TOL_DB
    )
    # a mixture used as the estimate has zero improvement
    assert core.si_sdr_improvement(mix, target, mix) == pytest.approx(0.0, abs=TOL_DB)


def test_si_sdr_matches_pairwise_neg_sisdr():
    target, _, estimate = _make_case(8000, alpha=0.3, seed=3)
    est_t = torch.from_numpy(estimate)[None, None]
    ref_t = torch.from_numpy(target)[None, None]
    reference = -float(PairwiseNegSDR("sisdr", zero_mean=True)(est_t, ref_t)[0, 0, 0])
    assert core.si_sdr(estimate, target) == pytest.approx(reference, abs=1e-9)


# ---------------------------------------------------------------------------
# SDR / SDRi (fast_bss_eval, filter_length=512)
# ---------------------------------------------------------------------------
def test_sdr_matches_analytic_value():
    # 10 s signal: the 512-tap filter bias relative to the analytic SDR is ~0.015 dB,
    # well inside the ±0.05 dB tolerance (2 s signals have a ~0.067 dB bias).
    T = 160000
    target, noise, estimate = _make_case(T, alpha=0.3, seed=4)
    expected = _expected_snr_db(target, noise, 0.3)
    assert core.sdr(estimate, target) == pytest.approx(expected, abs=TOL_DB)


def test_sdr_matches_fast_bss_eval_directly():
    import fast_bss_eval

    target, _, estimate = _make_case(32000, alpha=0.5, seed=5)
    reference = -float(
        fast_bss_eval.sdr_pit_loss(
            torch.from_numpy(estimate)[None], torch.from_numpy(target)[None], filter_length=512
        ).mean()
    )
    assert core.sdr(estimate, target) == pytest.approx(reference, abs=1e-6)


def test_sdr_improvement_sign():
    T = 160000
    target, noise, good = _make_case(T, alpha=0.2, seed=6)   # ~14 dB SDR
    _, _, bad = _make_case(T, alpha=2.0, seed=6)             # same noise, worse by 20 dB
    rng = np.random.default_rng(7)
    noise_mix = _orthogonal_noise(rng, target)
    mix = target + 1.0 * noise_mix                            # ~0 dB SDR

    expected_good = _expected_snr_db(target, noise, 0.2)
    expected_mix = _expected_snr_db(target, noise_mix, 1.0)
    good_i = core.sdr_improvement(good, target, mix)
    bad_i = core.sdr_improvement(bad, target, mix)
    assert good_i == pytest.approx(expected_good - expected_mix, abs=0.1)
    assert good_i > 0 > bad_i
    # the better estimate scores ~20 dB higher SDR than the worse one
    assert core.sdr(good, target) - core.sdr(bad, target) == pytest.approx(
        _expected_snr_db(target, noise, 0.2) - _expected_snr_db(target, noise, 2.0), abs=0.1
    )


# ---------------------------------------------------------------------------
# PESQ / STOI argument order
# ---------------------------------------------------------------------------
def test_pesq_and_stoi_argument_order(monkeypatch, tmp_path):
    calls = {"pesq": [], "stoi": []}

    def fake_pesq(ref, deg, fs=16000):
        calls["pesq"].append((np.asarray(ref).copy(), np.asarray(deg).copy(), int(fs)))
        return 3.25

    def fake_stoi(x, y, fs_sig=16000, extended=False):
        calls["stoi"].append((np.asarray(x).copy(), np.asarray(y).copy(), int(fs_sig), bool(extended)))
        return 0.91

    monkeypatch.setattr(core, "_pesq", fake_pesq)
    monkeypatch.setattr(core, "_stoi", fake_stoi)

    rng = np.random.default_rng(8)
    T = 16000
    clean = rng.standard_normal(T).astype(np.float32) * 0.1
    estimate = clean + 0.01 * rng.standard_normal(T).astype(np.float32)
    mix = clean + 0.1 * rng.standard_normal(T).astype(np.float32)

    path = tmp_path / "metrics.csv"
    tracker = ALLMetricsTracker(save_file=str(path))
    tracker(mix=mix, clean=clean, estimate=estimate, key="u1")
    tracker.final()

    (ref, deg, fs), = calls["pesq"]
    assert fs == SR
    np.testing.assert_allclose(ref, clean, atol=0)
    np.testing.assert_allclose(deg, estimate, atol=0)

    (x, y, fs_sig, extended), = calls["stoi"]
    assert fs_sig == SR and extended is False
    np.testing.assert_allclose(x, clean, atol=0)
    np.testing.assert_allclose(y, estimate, atol=0)


# ---------------------------------------------------------------------------
# CSV convention: per-utterance rows and avg/std rows agree
# ---------------------------------------------------------------------------
def test_tracker_csv_rows_and_aggregation_consistent(monkeypatch, tmp_path):
    # avoid running PESQ/STOI on synthetic noise; they are covered by the argument-order test
    monkeypatch.setattr(core, "_pesq", lambda ref, deg, fs=16000: 3.0)
    monkeypatch.setattr(core, "_stoi", lambda x, y, fs_sig=16000, extended=False: 0.9)

    T = 32000
    target, _, good = _make_case(T, alpha=0.1, seed=9)
    mix = target + 1.0 * _orthogonal_noise(np.random.default_rng(10), target)
    path = tmp_path / "metrics.csv"
    tracker = ALLMetricsTracker(save_file=str(path))
    tracker(mix=mix, clean=target, estimate=good, key="good")
    tracker(mix=mix, clean=target, estimate=mix, key="mix")
    tracker.final()

    rows = list(csv.DictReader(open(path)))
    assert [r["snt_id"] for r in rows] == ["good", "mix", "avg", "std"]
    good_row, mix_row, avg_row, std_row = rows

    # good estimate: positive, higher-is-better for all four dB columns
    for column in ("sdr", "sdr_i", "si-snr", "si-snr_i"):
        assert float(good_row[column]) > 0, column
    # mixture as estimate: zero improvement, same sign convention as the avg row
    assert abs(float(mix_row["sdr_i"])) < 1e-6
    assert abs(float(mix_row["si-snr_i"])) < 1e-6

    # avg exactly equals the mean of per-utterance rows (rows and aggregation agree)
    for column in ("sdr", "sdr_i", "si-snr", "si-snr_i"):
        expected = (float(good_row[column]) + float(mix_row[column])) / 2.0
        assert float(avg_row[column]) == pytest.approx(expected, abs=1e-9), column
        expected_std = float(np.std([float(good_row[column]), float(mix_row[column])], ddof=0))
        assert float(std_row[column]) == pytest.approx(expected_std, abs=1e-9), column

    # sign sanity: the good estimate's SI-SDRi is positive and clearly larger than the mix row's
    assert float(good_row["si-snr_i"]) > 10.0 > float(mix_row["si-snr_i"])
    assert float(good_row["sdr_i"]) > 0.0


def test_tracker_multi_source_does_not_crash_and_pesq_is_nan(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_pesq", lambda *a, **k: 0.0)
    monkeypatch.setattr(core, "_stoi", lambda *a, **k: 0.0)
    rng = np.random.default_rng(11)
    T = 16000
    clean = rng.standard_normal((2, T)).astype(np.float32)
    estimate = clean + 0.01 * rng.standard_normal((2, T)).astype(np.float32)
    mix = clean + rng.standard_normal((2, T)).astype(np.float32)
    path = tmp_path / "multisrc.csv"
    tracker = ALLMetricsTracker(save_file=str(path))
    row = tracker(mix=mix, clean=clean, estimate=estimate, key="m")
    tracker.final()
    assert math.isnan(row["pesq"]) and math.isnan(row["stoi"])
    assert np.isfinite(row["sdr"]) and np.isfinite(row["si-snr"])


# ---------------------------------------------------------------------------
# Loss-side regressions (W1-05)
# ---------------------------------------------------------------------------
# ``SISNRi`` used to die on construction: its ``__init__`` called
# ``super(Loss, self).__init__()`` with an undefined name ``Loss``, so the class was
# exported from ``look2hear.losses`` but unusable. It is also the only loss in the
# package that returns SI-SNR *improvement* rather than SI-SNR, hence worth pinning.
def test_sisnri_instantiable_and_forward():
    from torch import nn

    from look2hear.losses import SISNRi  # noqa: WPS433

    loss = SISNRi()  # used to raise NameError: name 'Loss' is not defined
    assert isinstance(loss, nn.Module)

    # Non-degenerate setup: two distinct references, mixture = their sum.
    t = torch.arange(1000, dtype=torch.float64) / 8000.0
    s1 = torch.sin(2 * math.pi * 440 * t)
    s2 = torch.sin(2 * math.pi * 900 * t) + 0.3 * torch.sin(2 * math.pi * 1234 * t)
    n_src, batch = 2, 2
    refs = torch.stack([torch.stack([s1, s2])] * batch)  # [n_src, batch, T]
    mix = (s1 + s2).unsqueeze(0).repeat(batch, 1)        # [batch, T]

    value = loss.compute_loss(mix, refs.clone(), refs)
    assert value.ndim == 0
    assert torch.isfinite(value)
    # ``__call__`` must dispatch to the same numbers (nn.Module -> forward alias).
    assert torch.allclose(loss(mix, refs.clone(), refs), value)

    # Improvement semantics: perfect estimates score far better (loss far more negative)
    # than feeding the mixture itself back, whose improvement is 0 dB by definition.
    as_mix = loss.compute_loss(mix, mix.unsqueeze(0).expand(n_src, -1, -1), refs)
    assert abs(float(as_mix)) < 1e-6
    assert float(value) < -50.0

    # ``sisnr`` is the per-utterance improvement, shape [batch, 1].
    per_utt = loss.sisnr(mix, refs[0], refs[0])
    assert tuple(per_utt.shape) == (batch, 1)
    assert float(per_utt.min()) > 50.0


# ``pit_from="pw_pt"`` used to hand a 2-D ``(batch, time)`` pair slice straight to the
# pairwise loss, which requires 3-D ``(batch, n_src, time)`` -> TypeError for every
# single-source call. It now promotes the rank, so 2-D and 3-D agree bit-for-bit and
# losses that legitimately take 2-D (``SingleSrcNegSDR``) keep working.
def test_pit_pw_pt_2d_does_not_raise():
    from look2hear.losses import PITLossWrapper, SingleSrcNegSDR, pairwise_neg_sisdr

    t = torch.arange(2000, dtype=torch.float64) / 8000.0
    s = torch.sin(2 * math.pi * 440 * t) + 0.3 * torch.sin(2 * math.pi * 1234 * t)
    gen = torch.Generator().manual_seed(11)
    batch = 3
    ref = s.unsqueeze(0).repeat(batch, 1) + 0.01 * torch.randn(
        batch, t.numel(), generator=gen, dtype=torch.float64
    )
    est = ref + 0.2 * torch.randn(batch, t.numel(), generator=gen, dtype=torch.float64)

    w_pt = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_pt", threshold_byloss=False)
    w_mtx = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False)

    value_2d = w_pt(est, ref)                      # used to raise TypeError
    value_3d = w_pt(est.unsqueeze(1), ref.unsqueeze(1))
    value_mtx = w_mtx(est.unsqueeze(1), ref.unsqueeze(1))

    for value in (value_2d, value_3d, value_mtx):
        assert value.ndim == 0 and torch.isfinite(value)

    # Single source => the only permutation is the identity, so all three must agree,
    # and they must equal the plain mean of the pairwise negative SI-SDR matrix.
    expected = pairwise_neg_sisdr(est.unsqueeze(1), ref.unsqueeze(1)).mean()
    assert float(value_2d) == pytest.approx(float(expected), abs=1e-9)
    assert float(value_3d) == pytest.approx(float(value_2d), abs=1e-9)
    assert float(value_mtx) == pytest.approx(float(value_2d), abs=1e-9)

    # A loss whose documented contract is 2-D must not be broken by the promotion.
    w_single = PITLossWrapper(SingleSrcNegSDR("sisdr"), pit_from="pw_pt", threshold_byloss=False)
    value_single = w_single(est, ref)
    assert torch.isfinite(value_single)
    assert float(value_single) == pytest.approx(float(value_2d), abs=1e-9)
