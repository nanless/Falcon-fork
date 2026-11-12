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


def test_nonfinite_output_is_not_committed_and_accepted_tick_can_retry():
    calls = []
    def processor(audio, video, state):
        calls.append((audio.copy(), state))
        if len(calls) == 2:
            return np.full(1920, np.nan), 999
        return audio[:1920], (state or 0) + 1
    session = LookaheadSession(processor)
    for i in range(7):
        result = session.push(np.full(640, i, np.float32), np.full((88, 88), i), sequence=i)
    assert session.state == 1
    with pytest.raises(ValueError, match="non-finite"):
        session.push(np.full(640, 7, np.float32), np.full((88, 88), 7), sequence=7)
    assert session.state == 1 and session.next_sequence == 8
    assert len(session.audio) == 5
    with pytest.raises(RuntimeError, match="retry_ready"):
        session.push(np.zeros(640), np.zeros((88, 88)), sequence=8)
    with pytest.raises(RuntimeError, match="retry_ready"):
        session.finish()
    retry = session.retry_ready()
    assert len(retry) == 1 and session.state == 2
    np.testing.assert_array_equal(calls[1][0], calls[2][0])
    assert calls[1][1] == calls[2][1] == 1
    assert session.next_sequence == 8 and len(session.audio) == 2


def test_failed_finish_retains_earlier_tail_blocks_and_can_retry():
    calls = 0
    def processor(audio, video, state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("temporary processor failure")
        return audio[:1920], (state or 0) + 1
    session = LookaheadSession(processor)
    for i in range(4):
        assert not session.push(np.full(640, i, np.float32), np.zeros((88, 88)), sequence=i)
    with pytest.raises(RuntimeError, match="temporary"):
        session.finish()
    assert session.state == 1 and len(session._finish_outputs) == 1
    with pytest.raises(ValueError, match="after finish"):
        session.retry_ready()
    outputs = session.finish()
    assert [len(x) for x in outputs] == [1920, 640]
    np.testing.assert_array_equal(np.concatenate(outputs), np.repeat(np.arange(4), 640))
    assert session.state == 2
    with pytest.raises(ValueError, match="already ended"):
        session.finish()
