"""Regression tests for RoomReader onset alignment and retention scoring."""
import numpy as np

from tools.streaming.evaluate_roomreader import measure_reopen
from tools.streaming.select_roomreader_reopen import locate_onset


def test_find_acoustic_onset_near_official_token_start():
    signal = np.zeros(800 * 16)
    signal[400 * 16:] = .07
    assert locate_onset(signal, 600, 1020) == 1000
    assert locate_onset(np.zeros_like(signal), 600, 1020) is None


def test_reopen_projection_retention_uses_native_mix_gain():
    n = 800 * 16
    time = np.arange(n, dtype=np.float64) / 16000
    target = .1 * np.sin(2 * np.pi * 100 * time)
    interferer = .2 * np.cos(2 * np.pi * 200 * time)
    mixture = 1.5 * target + .8 * interferer
    observed = {"onset_ms": 400, "start_ms": 160, "end_ms": 640, "start_context_ms": 0,
                "mouth_valid_all_score_ticks": True}
    rows = list(measure_reopen(observed, {"mixture": mixture, "target_half": .75 * target},
                               np.stack((target, interferer), axis=1), 0))
    assert [r["group"] for r in rows] == ["reopen_pre_silent", "reopen_120", "reopen_240"]
    assert "target_projection_delta_db" not in rows[0]["metrics"]["target_half"]
    for row in rows[1:]:
        scores = row["metrics"]["target_half"]
        assert abs(scores["target_projection_delta_db"] + 6.020599913) < 1e-6
        assert scores["target_projection_below_minus6db"]
        assert scores["target_sisdri_db"] > 60


def test_reopen_export_and_word_timestamp(tmp_path):
    import soundfile as sf

    from tools.streaming.evaluate_roomreader import export_reopen_waveforms
    from tools.streaming.score_roomreader_reopen_words import matched_words

    clean = np.array([0.1, -0.1], dtype=np.float32)
    output = 0.5 * clean
    export_reopen_waveforms(tmp_path, {"id": 4}, {"paired_on": output}, clean)
    actual, rate = sf.read(tmp_path / "004" / "paired_on.wav", dtype="float32")
    assert rate == 16000
    np.testing.assert_array_equal(actual, output)
    expected = {"text": "Right!", "start_ms": 1000, "end_ms": 1130}
    assert matched_words(expected, [{"word": " right.", "start": 0.46, "end": 0.65}], 500)
    assert not matched_words(expected, [{"word": " right.", "start": 1.2, "end": 1.3}], 500)
    assert not matched_words(expected, [{"word": " bright.", "start": 0.46, "end": 0.65}], 500)
