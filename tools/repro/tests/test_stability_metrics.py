import pytest
import torch
from tools.streaming.stability_metrics import collect, merge, empty, summarize, distributed_reduce


def test_categories_losses_and_amplitudes():
    loss = torch.tensor([-4., 10., 20.], requires_grad=True)
    masks = {"speech": torch.tensor([1, 0, 0]), "degenerate": torch.tensor([0, 1, 1]), "silent": torch.tensor([0, 1, 0])}
    output = torch.tensor([[1., -1.], [0., 0.], [4., -4.]])
    mix = torch.ones_like(output)
    result = summarize(distributed_reduce(collect(loss, masks, output, mix)))
    assert result == {"count": 3, "output_abs_max": 4., "output_mix_rms_ratio_max": 4.,
      "speech_count": 1, "speech_loss_mean": -4., "speech_loss_max": -4.,
      "degenerate_count": 2, "degenerate_loss_mean": 15., "degenerate_loss_max": 20.,
      "silent_count": 1, "silent_loss_mean": 10., "silent_loss_max": 10.}


def test_merge_matches_whole_and_empty_categories():
    torch.manual_seed(2)
    loss = torch.randn(4)
    masks = {"speech": torch.ones(4, dtype=torch.bool), "degenerate": torch.zeros(4, dtype=torch.bool), "silent": torch.zeros(4, dtype=torch.bool)}
    output = torch.randn(4, 8); mix = torch.randn(4, 8)
    whole = collect(loss, masks, output, mix)
    first = collect(loss[:2], {k:v[:2] for k,v in masks.items()}, output[:2], mix[:2])
    second = collect(loss[2:], {k:v[2:] for k,v in masks.items()}, output[2:], mix[2:])
    merged = merge(merge(empty(output.device), first), second)
    torch.testing.assert_close(whole[0], merged[0]); torch.testing.assert_close(whole[1], merged[1])
    assert summarize(merged)["silent_loss_mean"] is None


def test_finite_rms_extreme_and_zero_mix():
    values = torch.tensor([1.]); masks = {"speech": torch.tensor([True]), "degenerate": torch.tensor([False]), "silent": torch.tensor([False])}
    result = summarize(collect(values, masks, torch.full((1, 4), 1e30), torch.zeros(1, 4)))
    assert 1e29 < result["output_abs_max"] < 1e31
    assert 1e33 < result["output_mix_rms_ratio_max"] < 1e35


def test_invalid_shapes():
    with pytest.raises(ValueError): collect(torch.zeros(2), {}, torch.zeros(1,2), torch.zeros(1,2))
