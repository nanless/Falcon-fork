import numpy as np
import pytest

from tools.streaming.lookahead_session import LookaheadSession


def test_incremental_lookahead_and_state_commit():
    calls = []
    def processor(audio, video, state):
        calls.append((audio.copy(), video[:, 0, 0].copy(), state))
        return audio[:1920], (state or 0) + 3
    session = LookaheadSession(processor)
    outputs = []
    for i in range(8):
        outputs.extend(session.push(np.full(640, i, np.float32), np.full((88, 88), i), sequence=i))
        assert len(outputs) == (0 if i < 4 else 1 if i < 7 else 2)
    outputs.extend(session.finish())
    assert [len(x) for x in outputs] == [1920, 1920, 1280]
    assert np.array_equal(calls[0][1], [0, 1, 2, 3, 4])
    assert np.array_equal(calls[1][1], [3, 4, 5, 6, 7])
    assert [x[2] for x in calls] == [None, 3, 6]
    assert np.array_equal(np.concatenate(outputs), np.repeat(np.arange(8), 640))


def test_short_tail_padding_and_rejection():
    def processor(audio, video, state):
        return audio[:1920], state
    session = LookaheadSession(processor)
    session.push(np.ones(123), np.zeros((88, 88)), sequence=0)
    with pytest.raises(ValueError):
        session.push(np.ones(640), np.zeros((88, 88)), sequence=1)
    output = session.finish()
    assert len(output) == 1 and len(output[0]) == 123
    assert np.all(output[0] == 1)
    with pytest.raises(ValueError):
        session.finish()


def test_rejects_missing_tick_and_nonfinite_without_state_change():
    session = LookaheadSession(lambda a, v, s: (a[:1920], s))
    with pytest.raises(ValueError):
        session.push(np.zeros(640), np.zeros((88, 88)), sequence=1)
    with pytest.raises(ValueError):
        session.push(np.full(640, np.nan), np.zeros((88, 88)), sequence=0)
    assert session.next_sequence == 0
