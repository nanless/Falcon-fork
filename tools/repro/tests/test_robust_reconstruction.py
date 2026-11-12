import math
import torch
from tools.streaming.robust_reconstruction import robust_reconstruction
from tools.streaming.train_ddp import reconstruction_components


def test_speech_matches_existing_loss_and_gradient():
    torch.manual_seed(31)
    o = torch.randn(3, 256, requires_grad=True)
    t = torch.randn(3, 256); m = torch.randn(3, 256)
    old = reconstruction_components(o, t, m)[0]
    new, masks = robust_reconstruction(o, t, m, waveform_weight=0.0)
    torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-4)
    old_g = torch.autograd.grad(old.sum(), o, retain_graph=True)[0]
    new_g = torch.autograd.grad(new.sum(), o)[0]
    torch.testing.assert_close(new_g, old_g, rtol=3e-4, atol=1e-4)
    assert masks["speech"].all()


def test_zero_constant_and_mixed_gradients_are_finite():
    t = torch.tensor([[0., 0., 0., 0.], [2., 2., 2., 2.], [1e-7, -1e-7, 1e-7, -1e-7]])
    o = torch.tensor([[1e12, -1e12, 1e12, -1e12], [1., 1., 1., 1.], [.1, -.2, .1, -.2]], requires_grad=True)
    mix = torch.full_like(o, 0.01)
    loss, masks = robust_reconstruction(o, t, mix)
    assert masks["degenerate"].tolist() == [True, True, False]
    assert masks["silent"].tolist() == [True, False, False]
    assert torch.isfinite(loss).all() and loss[0] < 500
    loss.mean().backward()
    assert torch.isfinite(o.grad).all()


def test_exact_zero_loss_and_gradient():
    o = torch.zeros(1, 10, requires_grad=True)
    loss, _ = robust_reconstruction(o, torch.zeros_like(o), torch.zeros_like(o))
    loss.sum().backward()
    assert loss.item() < 1e-20
    assert torch.isfinite(o.grad).all()
    assert torch.count_nonzero(o.grad) == 0


def test_constant_loss_matches_log1p_definition():
    o = torch.tensor([[1., -2., 3., -4.]], requires_grad=True)
    mix = torch.full_like(o, .1)
    loss, _ = robust_reconstruction(o, torch.zeros_like(o), mix)
    expected = 10 / math.log(10) * torch.log1p(o.square().mean(dim=1) / mix.square().mean(dim=1))
    torch.testing.assert_close(loss, expected)
    a = torch.autograd.grad(loss.sum(), o, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), o)[0]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_speech_gain_explosion_penalty_has_correct_finite_gradient():
    target = torch.tensor([[0.2, -0.1, 0.3, -0.4]])
    mix = torch.ones_like(target) * 0.1
    o = torch.tensor([[0.3, 0.4, -0.1, -0.2]]) .mul(1e12).requires_grad_()
    loss, masks = robust_reconstruction(o, target, mix, waveform_weight=0.1)
    assert masks["speech"].item() and torch.isfinite(loss).all()
    grad = torch.autograd.grad(loss.sum(), o)[0]
    assert torch.isfinite(grad).all()
    no_aux, _ = robust_reconstruction(o, target, mix, waveform_weight=0.0)
    aux_grad = grad - torch.autograd.grad(no_aux.sum(), o)[0]
    assert (aux_grad * o).sum() > 0
    reduced, _ = robust_reconstruction(o.detach() * 1e-6, target, mix, waveform_weight=0.1)
    assert loss.item() > reduced.item()


def test_silence_not_double_counted_by_waveform_weight():
    target = torch.zeros(1, 4); mix = torch.ones(1, 4); o = torch.ones(1, 4)
    a, _ = robust_reconstruction(o, target, mix, waveform_weight=0.0)
    b, _ = robust_reconstruction(o, target, mix, waveform_weight=100.0)
    torch.testing.assert_close(a, b)


def test_perfect_alignment_extreme_scales_is_stable():
    t = torch.tensor([[0.2, -0.1, 0.3, -0.4]])
    m = torch.ones_like(t) * 0.1
    for scale in (1e-12, 1.0, 1e12):
        o = (t * scale).requires_grad_()
        loss, _ = robust_reconstruction(o, t, m)
        grad = torch.autograd.grad(loss.sum(), o)[0]
        assert torch.isfinite(loss).all() and torch.isfinite(grad).all()
        if scale > 1e6:
            assert (grad * o).sum() > 0
