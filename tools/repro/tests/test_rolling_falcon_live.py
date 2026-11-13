import numpy as np
import torch

from tools.streaming import rolling_falcon_live


def test_rolling_window_bounds_and_only_commits_current_ticks(monkeypatch):
    calls = []

    def audio_model(audio, features):
        calls.append((audio.clone(), features.clone()))
        return audio[:, None]

    class Visual:
        def __init__(self, backbone):
            pass

        def __call__(self, mouth, context):
            return mouth[:, :3, 0, 0].unsqueeze(1), mouth[:, 1:3].clone()

    monkeypatch.setattr(rolling_falcon_live, "load_baseline", lambda *args: (audio_model, None, 0))
    monkeypatch.setattr(rolling_falcon_live, "FalconLookaheadVisual", Visual)
    session = rolling_falcon_live.RollingFalconLiveInference("fixture", history_blocks=1)
    ticks = [np.full(640 if i < 11 else 407, i, dtype=np.float32) for i in range(12)]
    emitted = []
    for i, audio in enumerate(ticks):
        emitted.extend(session.push(audio, np.full((88, 88), i, dtype=np.float32), sequence=i))
    emitted.extend(session.finish())
    np.testing.assert_array_equal(np.concatenate(emitted), np.concatenate(ticks))
    assert [wave.shape[-1] for wave, _ in calls] == [3200, 5120, 5120, 5120]
    assert calls[0][1].flatten().tolist() == [0, 1, 2, 2, 2]
    assert calls[1][1].flatten().tolist() == [0, 1, 2, 3, 4, 5, 5, 5]
    assert calls[2][1].flatten().tolist() == [3, 4, 5, 6, 7, 8, 8, 8]
    assert torch.equal(session.session.state.visual.flatten(), torch.tensor([9, 10, 11]))
    session.reset()
    assert session.session.state is None and session.session.next_sequence == 0


def test_missing_target_ticks_preserve_output_clock_and_mute_only_missing_frames(monkeypatch):
    class Visual:
        def __init__(self, backbone):
            pass

        def __call__(self, mouth, context):
            return mouth[:, :3, 0, 0].unsqueeze(1), mouth[:, 1:3].clone()

    monkeypatch.setattr(rolling_falcon_live, "load_baseline",
                        lambda *args: (lambda audio, video: audio[:, None], None, 0))
    monkeypatch.setattr(rolling_falcon_live, "FalconLookaheadVisual", Visual)
    stream = rolling_falcon_live.RollingFalconLiveInference("fixture")
    frame = np.ones((88, 88), dtype=np.float32)
    audio = np.ones(640, dtype=np.float32)
    blocks = []
    for i in range(7):
        blocks.extend(stream.push(audio, frame if i in (1, 4) else None, sequence=i))
    blocks.extend(stream.finish())
    assert np.concatenate(blocks).reshape(-1, 640).sum(axis=1).tolist() == [0, 640, 0, 0, 640, 0, 0]
