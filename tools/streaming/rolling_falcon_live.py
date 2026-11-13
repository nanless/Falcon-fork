"""Single-target Falcon inference with bounded history and 80 ms input preview."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from tools.streaming.evaluate_realssa import load_baseline
from tools.streaming.lookahead_session import BLOCK_TICKS, LOOKAHEAD_TICKS, TICK_SAMPLES, LookaheadSession
from tools.streaming.paper_live_inference import PaperLiveInference

BLOCK_SAMPLES = BLOCK_TICKS * TICK_SAMPLES
PREVIEW_SAMPLES = LOOKAHEAD_TICKS * TICK_SAMPLES


@dataclass(frozen=True)
class RollingState:
    audio: torch.Tensor
    visual: torch.Tensor
    visual_context: torch.Tensor


class RollingFalconLiveInference(PaperLiveInference):
    """Recompute a fixed-length audio window and cache committed mouth features.

    Accepts the same aligned 40 ms ticks and returns the same 120 ms outputs as
    PaperLiveInference. The mouth input belongs to one fixed target track.
    The released audio model recomputes at most history_blocks+1 output blocks
    and two preview ticks. Speculative audio and held features never enter the
    next state's committed history.
    """

    def __init__(self, checkpoint, *, history_blocks=5, device="cpu", threads=None):
        if type(history_blocks) is not int or history_blocks < 0:
            raise ValueError("history_blocks must be a nonnegative integer")
        if threads is not None:
            if type(threads) is not int or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.audio_model, backbone, _ = load_baseline(Path(checkpoint), self.device)
        self.visual_model = FalconLookaheadVisual(backbone)
        self.history_blocks = history_blocks
        self.reset()

    @torch.inference_mode()
    def _process(self, audio, frames, state, present):
        chunk = torch.as_tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)
        mouth = torch.as_tensor(frames, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature, context = self.visual_model(mouth, None if state is None else state.visual_context)
        if state is not None:
            chunk = torch.cat((state.audio, chunk), dim=-1)
            feature = torch.cat((state.visual, feature), dim=-1)
        held = feature[..., -1:].expand(-1, -1, LOOKAHEAD_TICKS)
        prediction = self.audio_model(chunk, torch.cat((feature, held), dim=-1))
        if prediction.shape != (1, 1, chunk.shape[-1]):
            raise ValueError("released Falcon returned an unexpected waveform shape")
        committed_audio = chunk[..., :-PREVIEW_SAMPLES]
        audio_start = max(0, committed_audio.shape[-1] - self.history_blocks * BLOCK_SAMPLES)
        visual_start = max(0, feature.shape[-1] - self.history_blocks * BLOCK_TICKS)
        next_state = RollingState(committed_audio[..., audio_start:].clone(),
                                  feature[..., visual_start:].clone(), context)
        output = prediction[0, 0, -(BLOCK_SAMPLES + PREVIEW_SAMPLES):-PREVIEW_SAMPLES]
        output = output.float().cpu().numpy().copy()
        for i, available in enumerate(present[:BLOCK_TICKS]):
            if not available:
                output[i * TICK_SAMPLES:(i + 1) * TICK_SAMPLES] = 0.0
        return output, next_state

    def reset(self):
        self.session = LookaheadSession(self._process, accept_missing_video=True)

    def push(self, audio, mouth_frame, *, sequence):
        """Use None when the target track supplies no mouth frame on this tick."""
        return self.session.push(audio, mouth_frame, sequence=sequence)
