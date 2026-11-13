import pytest
import torch

from tools.streaming.exclude_realssa import replace_excluded_inputs, training_mean
from tools.streaming.scale_sdr_loss import scale_sdr_loss


@pytest.mark.parametrize("threshold", [0, 0.0002])
def test_loss_and_masks_equal_individually_cropped_rows(threshold):
    output = torch.tensor([[.3, -.2, .5, 2., 3.], [.04, -.02, 2., 3., 4.],
                           [.3, .1, .05, .4, .5], [0., 0., 9., 9., 9.]], requires_grad=True)
    target = torch.tensor([[.5, -.6, .3, 0., 0.], [.01, .01, 0., 0., 0.],
                           [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
    mix = torch.full_like(output, .25)
    lengths = torch.tensor([3, 2, 5, 2])
    fixed, masks = scale_sdr_loss(output, target, mix, valid_samples=lengths, silence_rms_threshold=threshold)
    expected = []
    for index, size in enumerate(lengths.tolist()):
        value, flags = scale_sdr_loss(output[index:index + 1, :size], target[index:index + 1, :size],
                                     mix[index:index + 1, :size], silence_rms_threshold=threshold)
        expected.append(value[0])
        assert all(masks[key][index] == flag[0] for key, flag in flags.items())
    torch.testing.assert_close(fixed, torch.stack(expected))
    fixed.sum().backward()
    mask = torch.arange(output.shape[1]) >= lengths[:, None]
    assert torch.count_nonzero(output.grad[mask]) == 0
    assert torch.isfinite(output.grad).all()


def test_full_length_preserves_original_and_invalid_lengths_fail():
    torch.manual_seed(42)
    tensors = [torch.randn(3, 5) for _ in range(3)]
    baseline, flags = scale_sdr_loss(*tensors)
    full, full_flags = scale_sdr_loss(*tensors, valid_samples=torch.full((3,), 5))
    assert torch.equal(baseline, full)
    assert all(torch.equal(flags[k], full_flags[k]) for k in flags)
    with pytest.raises(ValueError, match="outside"):
        scale_sdr_loss(*tensors, valid_samples=torch.tensor([1, 0, 5]))


def test_finite_placeholders_prevent_nonfinite_excluded_forward():
    excluded = torch.tensor([True, False])
    mixture = torch.tensor([[float("inf"), 1.], [2., 0.]])
    target = torch.tensor([[1., 0.], [1., 0.]])
    video = torch.ones(2, 5, 3, 3)
    weight = torch.nn.Parameter(torch.tensor(1.))
    old_loss = training_mean(scale_sdr_loss(mixture * weight, target, mixture)[0], excluded)
    old_loss.backward()
    assert torch.isfinite(old_loss) and torch.isnan(weight.grad)

    weight.grad = None
    clean_mix, clean_target, clean_video = replace_excluded_inputs(excluded, mixture, target, video)
    for before, after in zip((mixture, target, video), (clean_mix, clean_target, clean_video), strict=True):
        assert torch.equal(before[1], after[1])
        assert torch.count_nonzero(after[0]) == 0
    values, _ = scale_sdr_loss(clean_mix * weight, clean_target, clean_mix)
    loss = training_mean(values, excluded)
    loss.backward()
    assert torch.isfinite(weight.grad)
    torch.testing.assert_close(loss, values[1] / 2)
