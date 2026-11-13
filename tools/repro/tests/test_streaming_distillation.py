import pytest
import torch
from torch import nn

from look2hear.models.Falcon import VideoBlock
from tools.streaming.distillation import (
    FrozenFalconTeacher,
    distillation_components,
    predict_teacher,
)


def test_teacher_keeps_default_video_lstm_and_rejects_temporal_mode():
    default = VideoBlock(in_chan=4, hid_chan=6, norm_type="BatchNorm1d")
    lstm = default.rnn
    teacher = FrozenFalconTeacher(nn.Sequential(default), nn.Identity())
    assert default.rnn is lstm
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    temporal = VideoBlock(in_chan=4, hid_chan=6, norm_type="BatchNorm1d", video_lstm_batch_first=True)
    with pytest.raises(ValueError, match="independent single-step"):
        FrozenFalconTeacher(nn.Sequential(temporal), nn.Identity())
    with pytest.raises(ValueError, match="no VideoBlock"):
        FrozenFalconTeacher(nn.Identity(), nn.Identity())


class _FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = []
        self.grad_enabled = []

    def forward(self, mixture, video):
        self.sizes.append(mixture.shape[0])
        self.grad_enabled.append(torch.is_grad_enabled())
        return mixture * 2 + video[:, 0, 0, 0, None]


def test_predict_teacher_splits_aligned_examples_including_short_last_microbatch():
    teacher = _FakeTeacher().eval()
    mixture = torch.arange(30.0).reshape(5, 6).requires_grad_()
    video = torch.arange(5.0).reshape(5, 1, 1, 1)
    prediction = predict_teacher(teacher, mixture, video, microbatch_size=2)
    assert teacher.sizes == [2, 2, 1]
    assert teacher.grad_enabled == [False] * 3
    assert not prediction.requires_grad
    torch.testing.assert_close(prediction, mixture * 2 + video.reshape(5, 1))
    with pytest.raises(ValueError, match="eval"):
        predict_teacher(teacher.train(), mixture, video, microbatch_size=2)


def test_distillation_selects_only_reliable_speech_and_backpropagates_to_student():
    clean = torch.tensor([[1., -1., 1., -1.]] * 4 + [[0., 0., 0., 0.]])
    noise = torch.tensor([[1., 1., -1., -1.]] * 5)
    teacher = (clean + noise * torch.tensor([.25, 2., 1., .25, 1.])[:, None]).requires_grad_()
    mixture = clean + noise * torch.tensor([1., 1., 2., .25, 1.])[:, None]
    student = (clean + .5 * noise).requires_grad_()
    loss, selected, gain = distillation_components(student, teacher, clean, mixture,
                                                   min_teacher_sisdr_db=3.0)
    assert selected.tolist() == [True, False, False, False, False]
    assert gain[0] > 0 and gain[1] < 0 and gain[2] > 0
    assert loss[0] > 0 and torch.count_nonzero(loss[1:]) == 0
    loss.mean().backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad[0]) > 0
    assert torch.count_nonzero(student.grad[1:]) == 0


def test_distillation_identical_shapes_up_to_gain_and_offset_have_zero_loss():
    target = torch.tensor([[1., -1., 1., -1.]])
    noise = torch.tensor([[1., 1., -1., -1.]])
    teacher = target + .1 * noise
    student = 3.0 - 2.0 * teacher
    loss, selected, gain = distillation_components(student, teacher, target, target + noise)
    assert selected.tolist() == [True]
    torch.testing.assert_close(gain, torch.tensor([20.0]), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-10, rtol=0.0)


def test_distillation_all_rejected_returns_zero_finite_gradient():
    student = torch.zeros(3, 8, requires_grad=True)
    target = torch.ones_like(student)
    loss, selected, _ = distillation_components(student, target, target, target)
    assert not selected.any()
    loss.mean().backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
