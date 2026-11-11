import pytest
import torch

from look2hear.models.FalconStreaming import CHUNK_SAMPLES, VIDEO_FRAMES, FalconStreaming


@pytest.fixture
def attention_models():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(734)
        options = dict(channels=16, dilations=(1, 2), synthesis_mode="signed_residual")
        local = FalconStreaming(**options).eval()
        history = FalconStreaming(**options, attention_history_chunks=2).eval()
        state = local.state_dict()
        state["history_attention_gate"] = torch.zeros(())
        state["_streaming_format_version"] = torch.tensor(7)
        history.load_state_dict(state, strict=True)
        yield local, history
    finally:
        torch.set_num_threads(threads)


def sample(chunks):
    return torch.randn(1, CHUNK_SAMPLES * chunks), torch.randn(1, VIDEO_FRAMES * chunks, 88, 88)


@torch.no_grad()
def test_zero_history_gate_preserves_local_estimate(attention_models):
    local, history = attention_models
    audio, video = sample(3)
    torch.testing.assert_close(local(audio, video), history(audio, video), rtol=0, atol=0)
    assert local._streaming_format_version.item() == 6
    assert history._streaming_format_version.item() == 7
    with pytest.raises(RuntimeError, match="expected scalar int64 model version 7"):
        history.load_state_dict(local.state_dict(), strict=False)


@torch.no_grad()
def test_open_gate_uses_history_with_bounded_state(attention_models):
    _, model = attention_models
    model.history_attention_gate.fill_(1)
    state = None
    for block in range(4):
        audio, video = sample(1)
        previous = state
        output, state = model.forward_chunk(audio, video, state)
        for cached in state["attention"]:
            assert cached.shape == (1, 18 * min(block + 1, 2), 16)
        if previous is not None:
            for prior, cached in zip(previous["attention"], state["attention"]):
                torch.testing.assert_close(cached[:, :18], prior[:, -18:], rtol=0, atol=0)
            edited = dict(previous, attention=(previous["attention"][0],
                                               previous["attention"][1] + 2))
            changed, _ = model.forward_chunk(audio, video, edited)
            assert torch.max(torch.abs(output - changed)).item() > 1e-4
    invalid = dict(state, attention=(state["attention"][0][:, :-1], state["attention"][1]))
    with pytest.raises(ValueError, match="invalid attention cache"):
        model.forward_chunk(audio, video, invalid)


@torch.no_grad()
def test_open_gate_has_no_future_dependency(attention_models):
    _, model = attention_models
    model.history_attention_gate.fill_(1)
    audio, video = sample(4)
    clean = model(audio, video)
    shifted_audio, shifted_video = audio.clone(), video.clone()
    shifted_audio[:, 2 * CHUNK_SAMPLES:] *= -3
    shifted_video[:, 2 * VIDEO_FRAMES:] += 2
    for altered in (model(shifted_audio, video), model(audio, shifted_video)):
        torch.testing.assert_close(altered[:, :2 * CHUNK_SAMPLES],
                                   clean[:, :2 * CHUNK_SAMPLES], rtol=0, atol=0)


def test_invalid_attention_history_configuration():
    with pytest.raises(ValueError, match="attention history requires signed_residual"):
        FalconStreaming(attention_history_chunks=1)
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError, match="must be a nonnegative integer"):
            FalconStreaming(attention_history_chunks=value)
