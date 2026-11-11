"""Data-pairing and aligned crop regression tests for streaming training."""
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from tools.streaming.common import IndexedAudioVisualDataset


def make_data(root, wrong=False):
    (root / 'mix').mkdir()
    (root / 'target').mkdir()
    (root / 'mouths').mkdir()
    mix = root / 'mix' / 'sample.wav'
    target = root / 'target' / ('wrong.wav' if wrong else 'sample.wav')
    mouth = root / 'mouths' / 'sample.pt'
    # A 40ms bin has its index in both audio and video.  PCM float preserves it.
    audio = np.repeat(np.arange(12, dtype=np.float32) / 20, 640)
    sf.write(mix, audio, 16000, subtype='FLOAT')
    sf.write(target, audio, 16000, subtype='FLOAT')
    frames = torch.arange(12, dtype=torch.uint8)[:, None, None].expand(-1, 96, 96).contiguous()
    torch.save(frames, mouth)
    (root / 'mix.json').write_text(json.dumps([[str(mix), len(audio)]]))
    (root / 'target.json').write_text(json.dumps([[str(target), str(mouth), len(audio)]]))


def test_real_index_crop_preserves_audio_video_grid(tmp_path, monkeypatch):
    make_data(tmp_path)
    monkeypatch.setattr('tools.streaming.common.random.randint', lambda a, b: 3)
    mix, target, frames = IndexedAudioVisualDataset(tmp_path, 2)[0]
    assert mix.shape == target.shape == (5120,)
    assert frames.shape == (8, 88, 88)
    torch.testing.assert_close(mix, target)
    expected = (torch.arange(3, 11, dtype=torch.float32) / 255 - .421) / .165
    torch.testing.assert_close(frames[:, 0, 0], expected)
    torch.testing.assert_close(mix[::640], torch.arange(3, 11, dtype=torch.float32) / 20)


def test_wrong_paired_filename_rejected(tmp_path):
    make_data(tmp_path, wrong=True)
    with pytest.raises(ValueError, match='identity mismatch'):
        IndexedAudioVisualDataset(tmp_path, 2)


def test_inadequate_audio_length_rejected(tmp_path):
    make_data(tmp_path)
    with pytest.raises(ValueError, match='long enough'):
        IndexedAudioVisualDataset(tmp_path, 4)


@pytest.mark.parametrize("frame_count, expected_max_start", [(8, 0), (9, 1), (10, 2), (12, 4)])
def test_crop_start_respects_shorter_but_sufficient_video(tmp_path, monkeypatch, frame_count, expected_max_start):
    make_data(tmp_path)
    path = tmp_path / "mouths" / "sample.pt"
    torch.save(torch.load(path, weights_only=True)[:frame_count], path)
    # The common start is limited by both the audio and the available frames.
    def last_legal(a, b):
        assert (a, b) == (0, expected_max_start)
        return b
    monkeypatch.setattr("tools.streaming.common.random.randint", last_legal)
    mix, target, frames = IndexedAudioVisualDataset(tmp_path, 2)[0]
    torch.testing.assert_close(mix, target)
    expected = torch.arange(expected_max_start, expected_max_start + 8, dtype=torch.float32)
    torch.testing.assert_close(mix[::640], expected / 20)
    torch.testing.assert_close(frames[:, 0, 0], (expected / 255 - .421) / .165)


def test_video_shorter_than_training_segment_has_clear_error(tmp_path):
    make_data(tmp_path)
    path = tmp_path / "mouths" / "sample.pt"
    torch.save(torch.load(path, weights_only=True)[:7], path)
    dataset = IndexedAudioVisualDataset(tmp_path, 2)
    with pytest.raises(ValueError, match="mouth video is too short"):
        dataset[0]
