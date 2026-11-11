import pytest
import torch

from tools.streaming.lookahead_training import (
    CHUNK_SAMPLES,
    LOOKAHEAD_SAMPLES,
    SAMPLES_PER_FRAME,
    WINDOW_SAMPLES,
    lookahead_windows,
    run_lookahead_sequence,
)


def test_two_seconds_exact_alignment_and_terminal_padding():
    samples = 32000
    audio = torch.arange(samples, dtype=torch.float32)[None]
    target = -audio
    video = torch.arange(50, dtype=torch.float32)[None, :, None]
    windows = list(lookahead_windows(audio, video, target))
    assert len(windows) == 17
    assert all(w.audio.shape == (1, WINDOW_SAMPLES) for w in windows)
    assert all(w.video.shape == (1, 5, 1) for w in windows)
    first = windows[0]
    assert first.start_sample == 0 and first.valid_samples == CHUNK_SAMPLES
    assert first.video.flatten().tolist() == [0, 1, 2, 3, 4]
    torch.testing.assert_close(first.audio[0], audio[0, :WINDOW_SAMPLES])
    middle = windows[3]
    assert middle.start_sample == 9 * SAMPLES_PER_FRAME
    assert middle.video.flatten().tolist() == [9, 10, 11, 12, 13]
    tail = windows[-1]
    assert tail.start_sample == 30720 and tail.valid_samples == 1280
    assert tail.video.flatten().tolist() == [48, 49, 0, 0, 0]
    torch.testing.assert_close(tail.audio[:, :1280], audio[:, 30720:])
    assert torch.count_nonzero(tail.audio[:, 1280:]) == 0
    torch.testing.assert_close(tail.target[:, :1280], target[:, 30720:])
    assert torch.count_nonzero(tail.target[:, 1280:]) == 0


def test_overlap_and_differentiability_stop_at_80ms():
    audio = torch.arange(5120, dtype=torch.float32)[None].requires_grad_()
    video = torch.ones(1, 8, 1, requires_grad=True)
    def step(a, v, state):
        return a[:, :CHUNK_SAMPLES] + a[:, LOOKAHEAD_SAMPLES:WINDOW_SAMPLES] + v.sum((1, 2))[:, None], state
    output, _ = run_lookahead_sequence(audio, video, step)
    output[:, :CHUNK_SAMPLES].sum().backward()
    assert torch.all(audio.grad[:, CHUNK_SAMPLES:WINDOW_SAMPLES] != 0)
    assert torch.count_nonzero(audio.grad[:, WINDOW_SAMPLES:]) == 0
    assert torch.all(video.grad[:, :5] != 0)
    assert torch.count_nonzero(video.grad[:, 5:]) == 0


def test_history_keeps_gradient_and_only_current_chunk_advances_state():
    audio = torch.ones(1, 2 * CHUNK_SAMPLES, requires_grad=True)
    video = torch.zeros(1, 6, 1)
    def step(a, v, state):
        state = a.new_zeros(a.shape[0], 1) if state is None else state
        output = state.expand(-1, CHUNK_SAMPLES)
        return output, state + a[:, :CHUNK_SAMPLES].sum(dim=1, keepdim=True)
    output, state = run_lookahead_sequence(audio, video, step)
    assert state.item() == 2 * CHUNK_SAMPLES
    output[:, CHUNK_SAMPLES:].sum().backward()
    assert torch.all(audio.grad[:, :CHUNK_SAMPLES] == CHUNK_SAMPLES)
    assert torch.count_nonzero(audio.grad[:, CHUNK_SAMPLES:]) == 0


def test_sequence_tail_cropped_and_zero_padding_preserved():
    audio = torch.arange(2001, dtype=torch.float32)[None]
    video = torch.zeros(1, 4, 1)
    seen = []
    def step(a, v, state):
        seen.append((a.clone(), v.clone()))
        return a[:, :CHUNK_SAMPLES], state
    output, _ = run_lookahead_sequence(audio, video, step)
    torch.testing.assert_close(output, audio)
    assert len(seen) == 2
    assert torch.count_nonzero(seen[-1][0][:, 81:]) == 0


@pytest.mark.parametrize("audio,video,target", [
    (torch.zeros(1, 0), torch.zeros(1, 0, 1), None),
    (torch.zeros(1, 640), torch.zeros(1, 2, 1), None),
    (torch.zeros(1, 640), torch.zeros(2, 1, 1), None),
    (torch.zeros(1, 640), torch.zeros(1, 1, 1), torch.zeros(1, 641)),
])
def test_rejects_unaligned_input(audio, video, target):
    with pytest.raises(ValueError):
        list(lookahead_windows(audio, video, target))


def test_rejects_wrong_output_length():
    with pytest.raises(ValueError, match="step output"):
        run_lookahead_sequence(torch.zeros(1, 640), torch.zeros(1, 1, 1),
                               lambda a, v, s: (a[:, :100], s))
