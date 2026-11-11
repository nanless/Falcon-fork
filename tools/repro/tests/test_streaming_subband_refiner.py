import pytest
import torch

from look2hear.models.FalconStreaming import CHUNK_SAMPLES, VIDEO_FRAMES, FalconStreaming


@pytest.fixture(params=(0, 2))
def migrated_models(request):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(917)
        options = dict(channels=16, dilations=(1, 2), synthesis_mode="signed_residual",
                       attention_history_chunks=request.param)
        old = FalconStreaming(**options).eval()
        new = FalconStreaming(**options, subband_refiner=True).eval()
        migrated = new.state_dict()
        migrated.update(old.state_dict())
        migrated["_streaming_format_version"] = torch.tensor(8)
        new.load_state_dict(migrated, strict=True)
        yield old, new
    finally:
        torch.set_num_threads(threads)


def sample(chunks):
    return torch.randn(1, CHUNK_SAMPLES * chunks), torch.randn(1, VIDEO_FRAMES * chunks, 88, 88)


@torch.no_grad()
def test_zero_subband_output_preserves_migrated_waveform(migrated_models):
    old, new = migrated_models
    audio, video = sample(2)
    torch.testing.assert_close(new(audio, video), old(audio, video), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="expected scalar int64 model version 8"):
        new.load_state_dict(old.state_dict(), strict=False)


@torch.no_grad()
def test_active_subband_output_cannot_see_future_chunks(migrated_models):
    old, new = migrated_models
    new.subband.output_proj.weight.normal_(std=0.02)
    audio, video = sample(3)
    prediction = new(audio, video)
    assert (prediction - old(audio, video)).abs().max().item() > 1e-4
    altered_audio, altered_video = audio.clone(), video.clone()
    altered_audio[:, 2 * CHUNK_SAMPLES:] *= -3
    altered_video[:, 2 * VIDEO_FRAMES:] += 2
    for altered in (new(altered_audio, video), new(audio, altered_video)):
        torch.testing.assert_close(altered[:, :2 * CHUNK_SAMPLES],
                                   prediction[:, :2 * CHUNK_SAMPLES], rtol=0, atol=0)


def test_subband_head_updates_and_propagates_finite_gradients(migrated_models):
    _, model = migrated_models
    for parameter in model.video_model.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.subband.parameters(), lr=1e-3)
    audio, video = sample(1)
    for step in range(2):
        optimizer.zero_grad()
        output = model(audio, video)
        output.square().mean().backward()
        assert model.subband.output_proj.weight.grad.abs().max() > 0
        for name, parameter in model.subband.named_parameters():
            assert torch.isfinite(parameter.grad).all(), name
            if step == 1:
                assert parameter.grad.abs().max() > 0, name
        optimizer.step()


def test_subband_configuration_is_strict():
    with pytest.raises(ValueError, match="requires signed_residual"):
        FalconStreaming(subband_refiner=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        FalconStreaming(synthesis_mode="signed_residual", subband_refiner=1)
