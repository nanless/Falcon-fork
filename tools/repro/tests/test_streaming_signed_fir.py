import pytest
import torch

from look2hear.models.FalconStreaming import CHUNK_SAMPLES, VIDEO_FRAMES, FalconStreaming


@pytest.fixture
def signed_models():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(600)
        legacy = FalconStreaming(channels=16, dilations=(1, 2)).eval()
        signed = FalconStreaming(channels=16, dilations=(1, 2),
                                 synthesis_mode="signed_residual").eval()
        state = legacy.state_dict()
        for name in ("weight", "bias"):
            key = "synthesis_residual." + name
            state[key] = torch.zeros_like(signed.state_dict()[key])
        state["_streaming_format_version"] = torch.tensor(6)
        signed.load_state_dict(state, strict=True)
        yield legacy, signed
    finally:
        torch.set_num_threads(threads)


@torch.no_grad()
def test_zero_residual_matches_legacy_across_chunks(signed_models):
    legacy, signed = signed_models
    assert legacy._streaming_format_version.item() == 4
    assert signed._streaming_format_version.item() == 6
    audio = torch.randn(1, CHUNK_SAMPLES * 2)
    video = torch.randn(1, VIDEO_FRAMES * 2, 88, 88)
    torch.testing.assert_close(legacy(audio, video), signed(audio, video), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="expected scalar int64 model version 6"):
        signed.load_state_dict(legacy.state_dict(), strict=False)


@torch.no_grad()
def test_signed_residual_changes_waveform_without_future_dependency(signed_models):
    legacy, signed = signed_models
    audio = torch.randn(1, CHUNK_SAMPLES * 3)
    video = torch.randn(1, VIDEO_FRAMES * 3, 88, 88)
    signed.synthesis_residual.bias.fill_(-2.0)
    estimate = signed(audio, video)
    original = legacy(audio, video)
    # A constant response of negative two subtracts twice the input.
    torch.testing.assert_close(estimate, original - 2 * audio, rtol=1e-5, atol=3e-6)
    future_audio, future_video = audio.clone(), video.clone()
    future_audio[:, CHUNK_SAMPLES:] *= -3
    future_video[:, VIDEO_FRAMES:] += 2
    altered = signed(future_audio, future_video)
    torch.testing.assert_close(altered[:, :CHUNK_SAMPLES], estimate[:, :CHUNK_SAMPLES], rtol=0, atol=0)


def test_invalid_synthesis_and_visual_combinations():
    with pytest.raises(ValueError, match="signed_residual requires causal visual mode"):
        FalconStreaming(visual_mode="chunk_centered", synthesis_mode="signed_residual")
    with pytest.raises(ValueError, match="synthesis_mode must be"):
        FalconStreaming(synthesis_mode="invalid")
