"""Frozen released-Falcon targets for waveform distillation during training."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from look2hear.models.Falcon import VideoBlock
from tools.streaming.evaluate_realssa import load_baseline


class _IndependentVideoLSTM(nn.Module):
    """Batch the length-one sequences used by released Falcon at batch size one."""

    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        if lstm.batch_first:
            raise ValueError("expected released Falcon video LSTM layout")
        self.lstm = lstm

    def forward(self, value: Tensor):
        batch, frames, features = value.shape
        output, state = self.lstm(value.reshape(1, batch * frames, features))
        return output.reshape(batch, frames, -1), state


class FrozenFalconTeacher(nn.Module):
    def __init__(self, audio: nn.Module, visual: nn.Module):
        super().__init__()
        self.audio = audio
        self.visual = visual
        blocks = [layer for layer in audio.modules() if isinstance(layer, VideoBlock)]
        if not blocks:
            raise ValueError("released Falcon teacher has no VideoBlock")
        for block in blocks:
            block.rnn = _IndependentVideoLSTM(block.rnn)
        self.requires_grad_(False)
        self.eval()

    def forward(self, mixture: Tensor, video: Tensor) -> Tensor:
        return self.audio(mixture, self.visual(video.unsqueeze(1)))[:, 0]


def load_teacher(checkpoint_path, device) -> FrozenFalconTeacher:
    """Load the same released weights/configuration as the paired evaluator."""
    audio, visual, _ = load_baseline(checkpoint_path, device)
    return FrozenFalconTeacher(audio, visual)


@torch.no_grad()
def predict_teacher(teacher: FrozenFalconTeacher, mixture: Tensor, video: Tensor,
                    *, microbatch_size: int) -> Tensor:
    """Predict a target for each original, unpadded training crop."""
    if microbatch_size < 1 or mixture.ndim != 2 or video.ndim != 4:
        raise ValueError("expected positive microbatch size and [B,T]/[B,F,H,W] inputs")
    if not mixture.shape[0] or mixture.shape[0] != video.shape[0]:
        raise ValueError("teacher inputs must have matching nonempty batches")
    if teacher.training:
        raise ValueError("teacher must be in eval mode")
    outputs = [teacher(mix_chunk, video_chunk)
               for mix_chunk, video_chunk in zip(mixture.split(microbatch_size),
                                                video.split(microbatch_size))]
    result = torch.cat(outputs)
    if result.shape != mixture.shape or not torch.isfinite(result).all():
        raise RuntimeError("teacher returned invalid waveform")
    return result


def _center(value: Tensor) -> Tensor:
    value = value.float()
    return value - value.mean(dim=-1, keepdim=True)


def _align(estimate: Tensor, reference: Tensor, eps: float) -> Tensor:
    numerator = (estimate * reference).mean(dim=-1, keepdim=True)
    denominator = estimate.square().mean(dim=-1, keepdim=True).clamp_min(eps)
    return estimate * (numerator / denominator)


def _si_sdr_db(estimate: Tensor, reference: Tensor, eps: float) -> Tensor:
    numerator = (estimate * reference).mean(dim=-1, keepdim=True)
    denominator = reference.square().mean(dim=-1, keepdim=True).clamp_min(eps)
    projection = reference * (numerator / denominator)
    signal = projection.square().mean(dim=-1).clamp_min(eps)
    residual = (estimate - projection).square().mean(dim=-1).clamp_min(eps)
    return 10.0 * (signal.log10() - residual.log10())


def distillation_components(
    output: Tensor,
    teacher_output: Tensor,
    target: Tensor,
    mixture: Tensor,
    *,
    min_improvement_db: float = 0.0,
    min_teacher_sisdr_db: float = 0.0,
    silence_rms_threshold: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-sample loss, reliable-teacher mask and teacher SI-SDR gain.

    For reliable speech, align both waveforms to the clean reference by a
    scalar least-squares projection and normalize their squared difference
    by the clean reference energy. Rejected examples get a zero loss, so
    averaging over the entire batch keeps DDP's global sample weighting.
    """
    if output.ndim != 2 or output.shape[1] < 2 or not (
        output.shape == teacher_output.shape == target.shape == mixture.shape
    ):
        raise ValueError("distillation expects matching [B,T] waveforms")
    if not all(math.isfinite(value) for value in
               (min_improvement_db, min_teacher_sisdr_db, silence_rms_threshold)):
        raise ValueError("distillation thresholds must be finite")
    if silence_rms_threshold < 0:
        raise ValueError("silence RMS threshold must be nonnegative")
    eps = 1e-8
    with torch.no_grad():
        clean = _center(target)
        teacher = _center(teacher_output)
        mixture = _center(mixture)
        clean_power = clean.square().mean(dim=-1)
        is_speech = ((target.amax(dim=-1) != target.amin(dim=-1)) &
                     (clean_power > max(eps, silence_rms_threshold ** 2)))
        teacher_score = _si_sdr_db(teacher, clean, eps)
        teacher_gain = teacher_score - _si_sdr_db(mixture, clean, eps)
        selected = (is_speech & (teacher.square().mean(dim=-1) > eps) &
                    (teacher_score >= min_teacher_sisdr_db) &
                    (teacher_gain > min_improvement_db))
        teacher = _align(teacher, clean, eps)
    student = _align(_center(output), clean, eps)
    loss = (student - teacher).square().mean(dim=-1) / clean_power.clamp_min(eps)
    return torch.where(selected, loss, 0.0), selected, teacher_gain
