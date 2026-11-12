import math

import pytest
import torch

from tools.streaming.scale_sdr_loss import scale_sdr_loss


def test_nonzero_speech_matches_direct_target_error_formula_and_gradient():
    torch.manual_seed(124)
    target = torch.randn(5, 128)
    output = torch.randn_like(target, requires_grad=True)
    mix = torch.randn_like(target)
    actual, masks = scale_sdr_loss(output, target, mix)
    expected = 10 / math.log(10) * (
        (output - target).square().sum(1).add(1e-8).log()
        - target.square().sum(1).add(1e-8).log()
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual.sum(), output, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), output)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert masks["speech"].all()


def test_gain_changes_the_loss_and_gradient_direction():
    target = torch.tensor([[1.0, -2.0, 0.5, -1.5]])
    mix = target * 2
    small = (target * 0.5).requires_grad_()
    large = (target * 10.0).requires_grad_()
    small_loss, _ = scale_sdr_loss(small, target, mix)
    large_loss, _ = scale_sdr_loss(large, target, mix)
    assert small_loss.item() == pytest.approx(-6.0206, abs=1e-3)
    assert large_loss.item() == pytest.approx(19.0849, abs=1e-3)
    grad_small = torch.autograd.grad(small_loss.sum(), small)[0]
    grad_large = torch.autograd.grad(large_loss.sum(), large)[0]
    assert (grad_small * target).sum() < 0
    assert (grad_large * target).sum() > 0


def test_constant_nonzero_target_uses_degenerate_branch():
    target = torch.full((1, 10), 2.0)
    output = torch.full_like(target, 4.0)
    values, masks = scale_sdr_loss(output, target, target)
    assert masks["degenerate"].item() and not masks["silent"].item()
    assert values.item() == pytest.approx(10 * math.log10(2), abs=1e-5)


def test_zero_silence_log_loss_and_gradient():
    target = torch.zeros(2, 8)
    mix = torch.ones_like(target)
    output = torch.stack((torch.ones(8), torch.zeros(8))).requires_grad_()
    values, masks = scale_sdr_loss(output, target, mix)
    assert masks["silent"].all() and masks["degenerate"].all()
    assert values[0].item() == pytest.approx(10 * math.log10(2), rel=1e-6)
    assert values[1].item() < 1e-20
    g = torch.autograd.grad(values.sum(), output)[0]
    assert torch.isfinite(g).all() and torch.count_nonzero(g[1]) == 0
    assert (g[0] > 0).all()


def test_silent_threshold_uses_raw_rms_and_exclusive_masks():
    target = torch.tensor([[0., 0.], [1e-7, -1e-7], [0.1, 0.1]])
    values, masks = scale_sdr_loss(torch.ones_like(target), target, torch.ones_like(target),
                                            silence_rms_threshold=1e-6)
    assert masks["silent"].tolist() == [True, True, False]
    assert masks["speech"].tolist() == [False, False, False]
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("gain", [0.0, 1e-12, 0.5, 1.0, 2.0, 1e12, 1e20])
def test_extreme_gain_and_perfect_reconstruction_are_finite(gain):
    t = torch.tensor([[0.2, -0.1, 0.3, -0.4]])
    o = (t * gain).requires_grad_()
    result, _ = scale_sdr_loss(o, t, torch.ones_like(t))
    grad = torch.autograd.grad(result.sum(), o)[0]
    assert torch.isfinite(result).all() and torch.isfinite(grad).all()
    if gain == 1.0:
        assert torch.count_nonzero(grad) == 0


def test_mixed_extreme_silence_and_speech_gradients():
    target = torch.tensor([[0., 0., 0., 0.], [1e-7, -1e-7, 1e-7, -1e-7]])
    output = torch.tensor([[1e20, -1e20, 1e20, -1e20], [.1, -.2, .1, -.2]], requires_grad=True)
    result, masks = scale_sdr_loss(output, target, torch.full_like(output, .01))
    grad = torch.autograd.grad(result.sum(), output)[0]
    assert masks["speech"].tolist() == [False, True]
    assert torch.isfinite(result).all() and torch.isfinite(grad).all()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        scale_sdr_loss(torch.ones(1, 2), torch.ones(1, 1), torch.ones(1, 2))
    with pytest.raises(ValueError):
        scale_sdr_loss(torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2), eps=0)
