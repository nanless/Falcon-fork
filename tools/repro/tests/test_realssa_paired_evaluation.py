"""Paired evaluation must use the same content and remove padded samples."""
import torch
import pytest
from torch import nn
from tools.streaming.evaluate_realssa import predict, validate_reference


class Visual(nn.Module):
    def forward(self, x):
        assert x.shape == (1, 1, 50, 88, 88)
        self.frames = x.clone()
        return x


class Baseline(nn.Module):
    def forward(self, audio, visual):
        self.audio = audio.clone()
        self.visual = visual.clone()
        return audio.unsqueeze(1)


class Stream(nn.Module):
    def forward(self, audio, video):
        assert audio.shape == (1, 32640)
        assert video.shape == (1, 51, 88, 88)
        assert torch.count_nonzero(audio[:, 32000:]) == 0
        torch.testing.assert_close(video[:, 50], video[:, 49])
        self.audio, self.video = audio.clone(), video.clone()
        return audio


def test_two_models_receive_same_valid_samples_and_frames():
    mix = torch.linspace(-1, 1, 32000)
    mouth = torch.arange(50).float().view(50, 1, 1).expand(-1, 88, 88)
    baseline, visual, stream = Baseline(), Visual(), Stream()
    old, new = predict(baseline, visual, stream, mix, mouth, "cpu")
    torch.testing.assert_close(old, mix)
    torch.testing.assert_close(new, mix)
    torch.testing.assert_close(baseline.audio, stream.audio[:, :32000])
    torch.testing.assert_close(visual.frames[:, 0], stream.video[:, :50])


@pytest.mark.parametrize("value", [0.0, 0.125])
def test_constant_target_is_rejected(value):
    with pytest.raises(ValueError, match="degenerate constant target"):
        validate_reference(torch.full((32000,), value))


def test_nonconstant_target_is_accepted():
    validate_reference(torch.tensor([0.0, 1.0]))
