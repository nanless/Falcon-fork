"""Training windows for 120 ms output and 80 ms audio/video lookahead.

Timestamps use a common 16 kHz audio clock. Frame i starts at audio sample
640*i (25 fps). Windows start at 1920*k and cover 3200 audio samples and
five video frames. Past context lives exclusively in the model state: a step
must advance that state by the three *current* frames, never by its two preview
frames. The step implementation owns this contract and must be tested itself.

Zero padding is used only beyond the end of an utterance. This includes the
last incomplete output chunk and missing future context. It is not a prediction
of the unseen future. A stream must flush using the same padding convention.
"""
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 640
CHUNK_FRAMES = 3
LOOKAHEAD_FRAMES = 2
CHUNK_SAMPLES = CHUNK_FRAMES * SAMPLES_PER_FRAME
LOOKAHEAD_SAMPLES = LOOKAHEAD_FRAMES * SAMPLES_PER_FRAME
WINDOW_FRAMES = CHUNK_FRAMES + LOOKAHEAD_FRAMES
WINDOW_SAMPLES = CHUNK_SAMPLES + LOOKAHEAD_SAMPLES


@dataclass(frozen=True)
class LookaheadWindow:
    """A batch-aligned model input, output target, and its absolute clock."""

    start_sample: int
    valid_samples: int
    audio: torch.Tensor
    video: torch.Tensor
    target: torch.Tensor | None


def _slice_and_pad(x: torch.Tensor, start: int, length: int, dim: int) -> torch.Tensor:
    part = x.narrow(dim, min(start, x.shape[dim]), min(length, max(0, x.shape[dim] - start)))
    missing = length - part.shape[dim]
    if missing:
        shape = list(part.shape)
        shape[dim] = missing
        part = torch.cat((part, x.new_zeros(shape)), dim=dim)
    return part


def lookahead_windows(audio: torch.Tensor, video: torch.Tensor,
                      target: torch.Tensor | None = None, *, output_samples: int | None = None) -> Iterator[LookaheadWindow]:
    """Yield ordered aligned windows, preserving gradients and batch identity.

    audio: [batch, samples], video: [batch, frames, ...]; target, if supplied,
    is exactly aligned with the start of audio and has output_samples length. A final partial video
    frame counts as one frame. The caller may compute loss only on the returned
    valid target samples, not on the padded tail. Different sequence lengths in
    a batch require separate masks or separate calls.
    """
    if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError("audio must have nonempty shape [batch, samples]")
    if video.ndim < 3 or video.shape[0] != audio.shape[0]:
        raise ValueError("video must have shape [batch, frames, ...]")
    expected_frames = (audio.shape[1] + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    if video.shape[1] != expected_frames:
        raise ValueError(f"expected {expected_frames} aligned video frames, got {video.shape[1]}")
    if video.device != audio.device:
        raise ValueError("audio and video must be on the same device")
    if output_samples is None:
        output_samples = audio.shape[1] if target is None else target.shape[1]
    if type(output_samples) is not int or not 0 < output_samples <= audio.shape[1]:
        raise ValueError("output_samples must be a positive int no greater than input length")
    if target is not None and (target.shape != (audio.shape[0], output_samples) or target.device != audio.device):
        raise ValueError("target must match batch and output length and device")
    for start in range(0, output_samples, CHUNK_SAMPLES):
        valid = min(CHUNK_SAMPLES, output_samples - start)
        yield LookaheadWindow(
            start_sample=start,
            valid_samples=valid,
            audio=_slice_and_pad(audio, start, WINDOW_SAMPLES, 1),
            video=_slice_and_pad(video, start // SAMPLES_PER_FRAME, WINDOW_FRAMES, 1),
            target=None if target is None else _slice_and_pad(target, start, CHUNK_SAMPLES, 1),
        )


def run_lookahead_sequence(audio: torch.Tensor, video: torch.Tensor,
                           step: Callable[[torch.Tensor, torch.Tensor, Any], tuple[torch.Tensor, Any]],
                           initial_state: Any = None, *,
                           output_samples: int | None = None) -> tuple[torch.Tensor, Any]:
    """Run a pluggable streaming step and concatenate only valid output.

    step(window_audio, window_video, state) must return (output, new_state),
    with output shaped [batch, 1920]. The step must advance state by 120 ms,
    not by the entire 200 ms window. State is intentionally not detached:
    gradients flow across chunks as well as through overlapping lookahead.
    The model step is responsible for managing its states without in-place
    modification that breaks autograd. For training on independently cropped
    utterances, begin each example with its own empty state.
    """
    pieces = []
    state = initial_state
    for window in lookahead_windows(audio, video, output_samples=output_samples):
        output, state = step(window.audio, window.video, state)
        if output.shape != (audio.shape[0], CHUNK_SAMPLES):
            raise ValueError("step output must have shape [batch, 1920]")
        if output.device != audio.device:
            raise ValueError("step output must be on the audio device")
        pieces.append(output[:, :window.valid_samples])
    return torch.cat(pieces, dim=1), state
