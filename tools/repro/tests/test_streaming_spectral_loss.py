import pytest
import torch

from tools.streaming.spectral_loss import MultiResolutionSpectralLoss


def test_clean_match_and_phase_and_level_errors():
    loss = MultiResolutionSpectralLoss()
    target = torch.randn(3, 2048, generator=torch.Generator().manual_seed(21))
    output = torch.stack((target[0], -target[1], target[2] * 0.5))
    terms = loss(output, target, target, torch.ones(3, dtype=torch.bool))
    assert terms.shape == (3,)
    assert terms[0].item() == 0.0
    assert torch.all(terms[1:] > 0.05)


def test_scale_relative_above_floor_and_gradients_skip_degenerate_examples():
    loss = MultiResolutionSpectralLoss()
    target = torch.randn(2, 2048, generator=torch.Generator().manual_seed(19)) * 0.1
    target[1].zero_()
    output = (target + 0.02).requires_grad_()
    mask = torch.tensor([True, False])
    result = loss(output, target, target, mask)
    scaled = loss(output * 0.5, target * 0.5, target * 0.5, mask)
    torch.testing.assert_close(result, scaled)
    assert result[1].item() == 0.0
    result.mean().backward()
    assert torch.isfinite(output.grad).all()
    assert torch.count_nonzero(output.grad[0]) > 0
    assert torch.count_nonzero(output.grad[1]) == 0


def test_absolute_and_mixture_relative_floors_bound_quiet_target_errors():
    loss = MultiResolutionSpectralLoss()
    target = torch.full((2, 2048), 1e-8)
    output = torch.full_like(target, 0.1)
    mixture = torch.stack((torch.ones(2048), torch.zeros(2048)))
    terms = loss(output, target, mixture, torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(terms).all()
    torch.testing.assert_close(terms[1], 50 * terms[0])


def test_reject_non_boolean_mask():
    loss = MultiResolutionSpectralLoss()
    x = torch.zeros(1, 2048)
    with pytest.raises(ValueError, match="speech_mask must be boolean"):
        loss(x, x, x, torch.ones(1))


def test_soft_sisdr_floor_reduces_gradient_for_nearly_orthogonal_speech():
    from tools.streaming.train_ddp import reconstruction_components

    target = torch.tensor([[1., -1., 1., -1.], [0., 0., 0., 0.]])
    mixture = torch.ones_like(target)
    output = (torch.tensor([[1., 1., -1., -1.], [0.1, 0.2, 0.3, 0.4]])
              + 0.003 * target).requires_grad_()
    original, original_metric, _, _, _ = reconstruction_components(output, target, mixture)
    softened, metric, _, _, _ = reconstruction_components(
        output, target, mixture, speech_ratio_floor=0.001)
    original_gradient, = torch.autograd.grad(original.sum(), output, retain_graph=True)
    softened_gradient, = torch.autograd.grad(softened.sum(), output)
    torch.testing.assert_close(metric, original_metric, rtol=0, atol=0)
    torch.testing.assert_close(softened[1], original[1], rtol=0, atol=0)
    torch.testing.assert_close(softened_gradient[1], original_gradient[1], rtol=0, atol=0)
    assert 0 < softened_gradient[0].norm() < original_gradient[0].norm() / 50
    assert 29 < softened[0] < 30
