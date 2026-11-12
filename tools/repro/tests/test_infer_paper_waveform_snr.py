import math
import pytest
import torch

from tools.streaming.infer_paper_lookahead import waveform_snr_values


def test_scale_sensitive_snr_and_silent_target():
    target = torch.tensor([[1., -1.], [1., -1.], [0., 0.]])
    estimate = torch.tensor([[2., -2.], [0.5, -0.5], [9., -9.]])
    score = waveform_snr_values(estimate, target)
    assert score[0].item() == pytest.approx(0, abs=1e-8)
    assert score[1].item() == pytest.approx(10 * math.log10(4))
    assert torch.isnan(score[2])


def test_perfect_reconstruction_is_finite_and_shape_checked():
    target = torch.tensor([[1., -1.]])
    assert waveform_snr_values(target, target).item() == pytest.approx(120.)
    with pytest.raises(ValueError):
        waveform_snr_values(target, target[:, :1])
