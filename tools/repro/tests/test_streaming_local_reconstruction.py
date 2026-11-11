import pytest
import torch

from tools.streaming.local_reconstruction import (
    HOP_SAMPLES,
    WINDOW_SAMPLES,
    LocalReconstructionLoss,
)


def _signals(length):
    target = torch.tensor([1.0, -1.0, 1.0, -1.0]).repeat(length // 4).view(1, -1)
    interference = torch.tensor([1.0, 1.0, -1.0, -1.0]).repeat(length // 4).view(1, -1)
    return target, interference


def test_local_loss_prefers_clean_and_returns_a_per_sample_term():
    target, interference = _signals(WINDOW_SAMPLES)
    clean = target.repeat(3, 1)
    mixture = clean + interference
    estimate = torch.cat([target, target + 0.2 * interference, interference])
    loss = LocalReconstructionLoss()(estimate, clean, mixture)
    assert loss.shape == (3,)
    assert loss[0] < loss[1] < loss[2]
    assert loss[0].item() == pytest.approx(-30, abs=1e-3)
    assert loss[2].item() == pytest.approx(20, abs=1e-3)


def test_local_loss_rejects_unreliable_windows_and_preserves_fixed_denominator():
    length = WINDOW_SAMPLES + HOP_SAMPLES
    target, interference = _signals(length)
    target[..., -HOP_SAMPLES:] = 0
    # The second window sees a strong interferer and crosses the relative
    # reliability floor. The first remains reliable.
    mixture = target + interference
    mixture[..., -HOP_SAMPLES:] *= 100
    estimate = interference.requires_grad_()
    loss = LocalReconstructionLoss()(estimate, target, mixture)
    assert loss.item() == pytest.approx(10, abs=1e-4)
    loss.sum().backward()
    assert torch.count_nonzero(estimate.grad[..., -HOP_SAMPLES:]).item() == 0


def test_local_loss_zeroes_silence_and_targets_below_absolute_rms_floor():
    target, interference = _signals(WINDOW_SAMPLES)
    clean = torch.cat([torch.zeros_like(target), 0.0005 * target]).requires_grad_()
    mixture = torch.cat([interference, 0.0005 * (target + interference)]).requires_grad_()
    estimate = interference.repeat(2, 1).requires_grad_()
    loss = LocalReconstructionLoss()(estimate, clean, mixture)
    assert torch.count_nonzero(loss).item() == 0
    loss.sum().backward()
    assert torch.count_nonzero(estimate.grad).item() == 0
    assert clean.grad is None and mixture.grad is None


def test_local_loss_is_finite_for_near_orthogonal_and_low_output_power():
    target, interference = _signals(WINDOW_SAMPLES)
    clean = 0.002 * target.repeat(3, 1)
    mixture = 0.01 * (target + interference).repeat(3, 1)
    estimate = torch.cat([
        0.0001 * (interference + 0.0001 * target),
        0.002 * (interference + 0.002 * target),
        torch.zeros_like(target),
    ]).requires_grad_()
    loss = LocalReconstructionLoss()(estimate, clean, mixture)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(estimate.grad).all()
    assert 0 < estimate.grad[:2].norm().item() < 30
    assert torch.count_nonzero(estimate.grad[2]).item() == 0


def test_local_loss_uses_only_complete_windows():
    target, interference = _signals(WINDOW_SAMPLES)
    baseline = LocalReconstructionLoss()(interference, target, target + interference)
    extra = torch.ones(1, HOP_SAMPLES - 1)
    padded = [torch.cat([x, extra], dim=1) for x in (interference, target, target + interference)]
    torch.testing.assert_close(LocalReconstructionLoss()(*padded), baseline, rtol=0, atol=0)
    with pytest.raises(ValueError, match="480 ms"):
        LocalReconstructionLoss()(*[x[:, :-1] for x in (interference, target, target + interference)])
