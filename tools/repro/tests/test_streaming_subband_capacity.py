import pytest
import torch

from look2hear.models.FalconStreaming import CHUNK_SAMPLES, VIDEO_FRAMES, FalconStreaming


@pytest.fixture
def low_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model(**options):
    return FalconStreaming(channels=16, dilations=(1, 2), synthesis_mode="signed_residual",
                           subband_refiner=True, **options)


def inputs(chunks):
    return (torch.randn(1, chunks * CHUNK_SAMPLES),
            torch.randn(1, chunks * VIDEO_FRAMES, 88, 88))


@pytest.mark.parametrize("width,depth", ((64, 4), (96, 6)))
@torch.no_grad()
def test_migrate_trained_v8_at_both_v9_capacities(low_threads, width, depth):
    torch.manual_seed(112)
    old = model().eval()
    # Simulate a learned v8 branch, so migration exercises its contribution.
    old.subband.output_proj.weight.normal_(std=0.02)
    old.subband.output_proj.bias.normal_(std=0.02)
    new = model(subband_width=width, subband_depth=depth).eval()
    assert new.has_subband_capacity and not old.has_subband_capacity
    assert old._streaming_format_version.item() == 8
    assert new._streaming_format_version.item() == 9
    with pytest.raises(RuntimeError, match="model version 9"):
        new.load_state_dict(old.state_dict(), strict=False)

    migrated = new.state_dict()
    migrated.update(old.state_dict())
    migrated["_streaming_format_version"] = torch.tensor(9)
    new.load_state_dict(migrated, strict=True)
    audio, video = inputs(2)
    torch.testing.assert_close(new(audio, video), old(audio, video), atol=0, rtol=0)


@torch.no_grad()
def test_active_v9_is_chunkwise_causal_and_receives_visual_condition(low_threads):
    torch.manual_seed(113)
    net = model(subband_width=32, subband_depth=4).eval()
    net.subband_capacity.output_proj.weight.normal_(std=0.02)
    audio, video = inputs(3)
    result = net(audio, video)
    end = 2 * CHUNK_SAMPLES
    frames = 2 * VIDEO_FRAMES
    changed_audio, changed_video = audio.clone(), video.clone()
    changed_audio[:, end:] += torch.randn_like(changed_audio[:, end:])
    changed_video[:, frames:] += torch.randn_like(changed_video[:, frames:])
    for output in (net(changed_audio, video), net(audio, changed_video)):
        torch.testing.assert_close(output[:, :end], result[:, :end], atol=0, rtol=0)
        assert (output[:, end:] - result[:, end:]).abs().max().item() > 1e-5

    spectrum = torch.fft.rfft(torch.randn(1, 15, 256))
    features = net.subband.spectral_features(spectrum)
    condition = torch.randn(1, net.channels, 15)
    visual = torch.randn_like(condition)
    gains = net.subband_capacity(features, condition, visual)
    without_visual = net.subband_capacity(features, condition, torch.zeros_like(visual))
    assert (gains - without_visual).abs().max().item() > 1e-4


def test_v9_capacity_trains_after_zero_initialized_head(low_threads):
    torch.manual_seed(114)
    net = model(subband_width=32, subband_depth=4).train()
    net.requires_grad_(False)
    net.subband_capacity.requires_grad_(True)
    optimizer = torch.optim.AdamW(net.subband_capacity.parameters(), lr=1e-3)
    audio, video = inputs(1)
    for step in range(2):
        optimizer.zero_grad()
        net(audio, video).square().mean().backward()
        assert net.subband_capacity.output_proj.weight.grad.abs().max().item() > 0
        for name, parameter in net.subband_capacity.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step == 1:
                assert parameter.grad.abs().max().item() > 0, name
        optimizer.step()


def test_nondefault_capacity_requires_subband_and_positive_integral_geometry():
    with pytest.raises(ValueError, match="requires subband_refiner"):
        FalconStreaming(synthesis_mode="signed_residual", subband_width=64)
    with pytest.raises(ValueError, match="subband_width must be a positive integer"):
        model(subband_width=True)
    with pytest.raises(ValueError, match="subband_depth must be a positive integer"):
        model(subband_depth=0)
