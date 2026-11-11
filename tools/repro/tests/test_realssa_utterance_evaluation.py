from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch

from tools.streaming.evaluate_realssa_utterance import (
    FORMAT,
    decode_aligned,
    merge_reports,
    predict_utterance,
    summarize,
)


def _wave(samples, first_channel):
    f = io.BytesIO()
    wave = np.full((samples, 4), 0.75, dtype=np.float32)
    wave[:, 0] = first_channel
    sf.write(f, wave, 16000, format="WAV", subtype="FLOAT")
    return f.getvalue()


def test_full_utterance_alignment_keeps_channel_zero_and_over_two_seconds():
    mouths = io.BytesIO()
    torch.save(torch.full((57, 96, 96), 128, dtype=torch.uint8), mouths)
    x, s, v, geometry = decode_aligned(_wave(55 * 640 + 11, 0.1),
                                       _wave(56 * 640, 0.2), mouths.getvalue())
    assert x.shape == s.shape == (55 * 640,)
    torch.testing.assert_close(x, torch.full_like(x, 0.1))
    torch.testing.assert_close(s, torch.full_like(s, 0.2))
    assert v.shape == (55, 88, 88)
    torch.testing.assert_close(v, torch.full_like(v, (128 / 255 - 0.421) / 0.165))
    assert geometry == {"aligned_samples": 55 * 640, "aligned_frames": 55,
                        "mixture_trimmed_samples": 11, "target_trimmed_samples": 640,
                        "video_trimmed_frames": 2, "streaming_pad_samples": 1280,
                        "streaming_pad_frames": 2}


def test_predict_uses_aligned_clock_and_does_not_score_tail_padding():
    def baseline(audio, mouth):
        assert audio.shape == (1, 2560)
        assert mouth.shape == (1, 1, 4, 88, 88)
        return audio[:, None]

    def student(audio, mouth):
        assert audio.shape == (1, 3840)
        assert not torch.count_nonzero(audio[:, 2560:])
        assert mouth.shape == (1, 6, 88, 88)
        assert mouth[0, :, 0, 0].tolist() == [0, 1, 2, 3, 3, 3]
        return audio + 1

    audio = torch.arange(2560, dtype=torch.float32)
    mouth = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1).expand(-1, 88, 88)
    old, new = predict_utterance(baseline, lambda x: x, student, audio, mouth, "cpu")
    torch.testing.assert_close(old, audio)
    torch.testing.assert_close(new, audio + 1)
    with pytest.raises(ValueError, match="aligned full 40 ms intervals"):
        predict_utterance(baseline, lambda x: x, student, audio[:-1], mouth, "cpu")


def _shard(start, rows):
    return {"format": FORMAT, "identity": {"tar_index": {"sha256": "index-a"}, "index_count": 3},
            "protocol": {"chunk_samples": 1920}, "start": start, "end_exclusive": start + len(rows),
            "rows": rows}


def test_merge_requires_index_order_full_coverage_and_identical_protocol():
    index = {"samples": [{"id": name} for name in ("a", "b", "c")]}
    rows = [{"index": i, "id": row["id"], "status": "constant_target"} for i, row in enumerate(index["samples"])]
    a, b = _shard(0, rows[:2]), _shard(2, rows[2:])
    assert merge_reports(index, "index-a", [b, a])["rows"] == rows
    with pytest.raises(ValueError, match="incomplete"):
        merge_reports(index, "index-a", [a])
    with pytest.raises(ValueError, match="contiguous"):
        merge_reports(index, "index-a", [a, a])
    with pytest.raises(ValueError, match="protocols"):
        merge_reports(index, "index-a", [a, {**b, "protocol": {}}])
    with pytest.raises(ValueError, match="row and index"):
        merge_reports(index, "index-a", [a, {**b, "rows": [{**rows[-1], "id": "b"}]}])


def test_metrics_use_separate_paired_denominators():
    stats = summarize([
        {"status": "ok", "released_si_snri_db": 4.0, "streaming_si_snri_db": 3.0,
         "released_sdri_db": 5.0, "streaming_sdri_db": 6.0},
        {"status": "invalid_metric", "released_si_snri_db": 5.0, "streaming_si_snri_db": 7.0,
         "released_sdri_db": 7.0, "streaming_sdri_db": None},
        {"status": "constant_target"},
    ])
    assert stats["constant_targets"] == 1 and stats["invalid_metrics"] == 1
    assert stats["si_snri_db"]["valid_pairs"] == 2 and stats["sdri_db"]["valid_pairs"] == 1
    assert stats["si_snri_db"]["paired_delta_mean"] == 0.5
    assert stats["sdri_db"]["paired_delta_mean"] == 1.0
