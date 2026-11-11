"""State completeness and chunk-boundary causality regression checks."""
import pytest
import torch
from look2hear.models.FalconStreaming import FalconStreaming

@pytest.fixture(scope="module")
def setup_model():
    torch.set_num_threads(2)
    torch.manual_seed(231)
    m = FalconStreaming(channels=8, dilations=(1, 2)).eval()
    a = torch.randn(1, 1920)
    v = torch.randn(1, 3, 88, 88)
    with torch.no_grad():
        _, state = m.forward_chunk(a, v)
    return m, a, v, state


def test_incomplete_or_malformed_state_is_rejected(setup_model):
    model, audio, video, state = setup_model
    malformed = [
        {},
        {"audio": state["audio"]},
        {k: v for k, v in state.items() if k != "blocks"},
        {**state, "other": 1},
        {**state, "blocks": [None] * len(model.blocks)},
        {**state, "blocks": state["blocks"][:-1]},
        {**state, "audio": state["audio"].double()},
        {**state, "video": state["video"][:, :3]},
        {**state, "blocks": "invalid"},
    ]
    with torch.no_grad():
        for bad in malformed:
            with pytest.raises(ValueError, match="stream state"):
                model.forward_chunk(audio, video, bad)
        result, next_state = model.forward_chunk(audio, video, state)
        assert result.shape == (1, 1920) and set(next_state) == {"audio", "video", "blocks"}


def test_history_state_changes_current_output(setup_model):
    model, audio, video, state = setup_model
    with torch.no_grad():
        with_history, _ = model.forward_chunk(audio, video, state)
        without_history, _ = model.forward_chunk(audio, video, None)
    assert (with_history - without_history).abs().max() > 1e-6


def test_multichunk_training_backpropagates_through_cached_state(setup_model):
    model, _, _, _ = setup_model
    model.train()
    audio = torch.randn(1, 3840, requires_grad=True)
    video = torch.randn(1, 6, 88, 88, requires_grad=True)
    model.zero_grad(set_to_none=True)
    prediction = model(audio, video)
    prediction[:, 1920:].square().mean().backward()
    assert torch.isfinite(prediction).all()
    assert audio.grad is not None and audio.grad[:, :1920].abs().sum() > 0
    assert video.grad is not None and video.grad[:, :3].abs().sum() > 0
    assert model.audio_proj.weight.grad is not None and model.audio_proj.weight.grad.abs().sum() > 0
    assert model.video_model.frontend3D[0].weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.eval()


def test_adaptive_fir_matches_linear_convolution_and_gradients():
    from look2hear.models.FalconStreaming import causal_mask_filter
    torch.manual_seed(71)
    # Multiple time-varying masks, not just an identity or one constant filter.
    x = torch.randn(2, 7, 256, dtype=torch.float64, requires_grad=True)
    mask = torch.rand(2, 7, 129, dtype=torch.float64, requires_grad=True)
    spectrum = torch.fft.rfft(x, n=256, dim=-1)
    actual = causal_mask_filter(spectrum, mask)
    impulse = torch.fft.irfft(mask, n=256, dim=-1)[..., :129]
    # For output samples 128:256, all 129 past samples are in this window.
    windows = x.unfold(-1, 129, 1)
    reference = (windows * impulse.flip(-1).unsqueeze(-2)).sum(-1)
    torch.testing.assert_close(actual, reference, atol=2e-14, rtol=2e-13)
    weights = torch.randn_like(actual)
    got_grad = torch.autograd.grad((actual * weights).sum(), (x, mask), retain_graph=True)
    ref_grad = torch.autograd.grad((reference * weights).sum(), (x, mask))
    for actual_grad, ref in zip(got_grad, ref_grad):
        torch.testing.assert_close(actual_grad, ref, atol=3e-13, rtol=3e-12)


def test_adaptive_fir_identity_mask_has_no_shift():
    from look2hear.models.FalconStreaming import causal_mask_filter
    torch.manual_seed(32)
    x = torch.randn(3, 5, 256)
    out = causal_mask_filter(torch.fft.rfft(x, dim=-1), torch.ones(3, 5, 129))
    torch.testing.assert_close(out, x[..., 128:], rtol=1e-5, atol=1e-6)


def test_legacy_state_dict_cannot_be_silently_strict_loaded(setup_model):
    model, _, _, _ = setup_model
    assert model.state_dict()["_streaming_format_version"].item() == 4
    legacy = {k: v for k, v in model.state_dict().items() if k != "_streaming_format_version"}
    with pytest.raises(RuntimeError, match="_streaming_format_version"):
        model.load_state_dict(legacy, strict=True)


@pytest.mark.parametrize("bad", [torch.tensor(1), torch.tensor(2), torch.tensor(3.0), torch.tensor([3])])
@pytest.mark.parametrize("strict", [True, False])
def test_direct_model_load_rejects_incorrect_version(setup_model, bad, strict):
    model, _, _, _ = setup_model
    state = dict(model.state_dict())
    state["_streaming_format_version"] = bad
    with pytest.raises(RuntimeError, match="expected scalar int64 model version 4"):
        model.load_state_dict(state, strict=strict)
