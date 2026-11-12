import numpy as np
import pytest
import torch

from tools.streaming.infer_paper_lookahead import SavedPaperModel
from tools.streaming.paper_live_inference import PaperLiveInference


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(31)
    config = {"width": 8, "rnn_hidden": 8, "heads": 2, "history_frames": 20,
              "mask_limit": 4.0}
    model = SavedPaperModel(config).eval()
    path = tmp_path / "random.pt"
    torch.save({"config": config, "model": model.state_dict()}, path)
    return path, model


def test_online_matches_checkpoint_inference_with_short_tail(checkpoint):
    path, reference = checkpoint
    live = PaperLiveInference(path, threads=2)
    rng = np.random.default_rng(52)
    audio = rng.normal(size=8 * 640 + 413).astype(np.float32)
    video = rng.normal(size=(9, 88, 88)).astype(np.float32)
    outputs = []
    emitted_at = []
    for index in range(len(video)):
        result = live.push(audio[index * 640:(index + 1) * 640], video[index], sequence=index)
        if result:
            emitted_at.append(index)
            outputs.extend(result)
    outputs.extend(live.finish())
    assert emitted_at == [4, 7]
    actual = np.concatenate(outputs)
    with torch.inference_mode():
        expected = reference(torch.from_numpy(audio)[None], torch.from_numpy(video)[None], len(audio))
    np.testing.assert_array_equal(actual, expected[0].numpy())
    live.reset()
    assert live.session.next_sequence == 0
    assert live.push(np.zeros(640, np.float32), np.zeros((88, 88), np.float32), sequence=0) == []


def test_processor_failure_retries_same_accepted_tick(checkpoint):
    path, _ = checkpoint
    live = PaperLiveInference(path, threads=2)
    original = live.session.processor
    calls = 0
    def fails_once(audio, video, state):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.full(1920, np.nan, np.float32), "bad"
        return original(audio, video, state)
    live.session.processor = fails_once
    frame = np.zeros((88, 88), np.float32)
    for i in range(4):
        assert live.push(np.zeros(640, np.float32), frame, sequence=i) == []
    with pytest.raises(ValueError, match="non-finite"):
        live.push(np.zeros(640, np.float32), frame, sequence=4)
    assert live.session.state is None and live.session.next_sequence == 5
    outputs = live.retry_ready()
    assert len(outputs) == 1 and outputs[0].shape == (1920,)
    assert np.isfinite(outputs[0]).all()


def test_input_validation_failure_does_not_consume_tick(checkpoint):
    path, _ = checkpoint
    live = PaperLiveInference(path, threads=2)
    with pytest.raises(ValueError, match="finite"):
        live.push(np.full(640, np.nan, np.float32), np.zeros((88, 88), np.float32), sequence=0)
    assert live.session.next_sequence == 0
    assert live.push(np.zeros(640, np.float32), np.zeros((88, 88), np.float32), sequence=0) == []
    assert live.session.next_sequence == 1
