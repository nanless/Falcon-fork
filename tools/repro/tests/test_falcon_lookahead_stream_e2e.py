"""Bounded-lookahead full-model sessions, cross-block state and final flush."""
from pathlib import Path

import numpy as np
import pytest
import torch

from look2hear.models.FalconPaperLookahead import FalconPaperLookahead
from tools.streaming.lookahead_session import LookaheadSession


@pytest.fixture
def full_model():
    path = Path("/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar")
    if not path.exists():
        pytest.skip("released visual backbone is not available")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(3)
    model = FalconPaperLookahead(str(path), width=8, rnn_hidden=8, heads=2).eval()
    yield model
    torch.set_num_threads(old_threads)


def _processor(model):
    def process(audio, video, state):
        with torch.inference_mode():
            result, state = model(torch.from_numpy(audio)[None],
                                  torch.from_numpy(video)[None], state)
        return result[0].numpy(), state
    return process


def _session_run(model, audio, video):
    session = LookaheadSession(_processor(model))
    pieces = []
    emitted = []
    for i in range(len(video)):
        chunks = session.push(audio[i * 640:(i + 1) * 640], video[i], sequence=i)
        if chunks:
            emitted.append(i)
            pieces.extend(chunks)
    pieces.extend(session.finish())
    return np.concatenate(pieces), emitted


def _direct_stream_reference(model, audio, video):
    # Build each model input from the full recording and zero-pad only at EOF.
    # This independent scheduler checks the session flush and state progression;
    # it does not assert equivalence to a separately implemented offline model.
    process = _processor(model)
    pieces, state = [], None
    for begin in range(0, len(audio), 1920):
        chunk = audio[begin:begin + 3200]
        chunk = np.pad(chunk, (0, 3200 - len(chunk)))
        frame_begin = begin // 640
        frames = video[frame_begin:frame_begin + 5]
        frames = np.pad(frames, ((0, 5 - len(frames)), (0, 0), (0, 0)))
        output, state = process(chunk, frames, state)
        pieces.append(output[:min(1920, len(audio) - begin)])
    return np.concatenate(pieces)


def test_live_session_future_bound_and_short_tail(full_model):
    rng = np.random.default_rng(7)
    audio = rng.normal(size=8 * 640 + 413).astype(np.float32)
    video = rng.normal(size=(9, 88, 88)).astype(np.float32)
    original, emitted = _session_run(full_model, audio, video)
    assert emitted == [4, 7]
    assert len(original) == len(audio)
    assert np.isfinite(original).all()
    reference = _direct_stream_reference(full_model, audio, video)
    np.testing.assert_allclose(original, reference, rtol=0, atol=0)
    # Block k ends at (k+1)*120 ms. Inputs beyond another 80 ms must not
    # influence that block, including when state from earlier blocks is used.
    for k in (0, 1):
        cutoff_tick = 3 * (k + 1) + 2
        modified_audio = audio.copy()
        modified_video = video.copy()
        modified_audio[cutoff_tick * 640:] *= -9
        modified_video[cutoff_tick:] *= -9
        modified, _ = _session_run(full_model, modified_audio, modified_video)
        np.testing.assert_array_equal(original[k * 1920:(k + 1) * 1920],
                                      modified[k * 1920:(k + 1) * 1920])


def test_identity_reconstruction_at_boundaries_and_short_tail(full_model):
    rng = np.random.default_rng(8)
    audio = rng.normal(size=8 * 640 + 413).astype(np.float32)
    video = rng.normal(size=(9, 88, 88)).astype(np.float32)
    with torch.no_grad():
        full_model.core.mask.weight.zero_()
        full_model.core.mask.bias.zero_()
    output, _ = _session_run(full_model, audio, video)
    np.testing.assert_allclose(output, audio, rtol=1e-5, atol=2e-6)
