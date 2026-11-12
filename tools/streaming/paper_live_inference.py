"""Incremental checkpoint inference for synchronized audio and mouth frames.

The caller supplies one 40 ms 16 kHz mono audio tick and its already cropped,
grayscale, normalized 88x88 mouth frame at a time. Face tracking, capture and
network transport are outside this interface. One result becomes available
after the first five ticks and every third tick thereafter.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from tools.streaming.infer_paper_lookahead import SavedPaperModel, digest
from tools.streaming.lookahead_session import LookaheadSession


class PaperLiveInference:
    """Load a training checkpoint and process a single synchronized stream.

    A short final audio tick must be followed by finish(). All frames use the
    same 25 fps audio clock, and sequence starts at zero. Input validation
    errors occur before accepting the tick: fix input and resend its sequence.
    A processor error occurs after acceptance: use retry_ready() instead of
    resending. If finish fails, call finish again. Each result is valid audio.
    Setting threads changes PyTorch's process-wide CPU thread count.

    This class accepts trusted local training checkpoints. torch.load uses
    Python pickle; do not load checkpoints from untrusted sources.
    """

    def __init__(self, checkpoint, *, device="cpu", precision="fp32", threads=None):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if precision == "bf16" and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA BF16 is unsupported")
        if threads is not None:
            if type(threads) is not int or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("config"), dict) or "model" not in loaded:
            raise ValueError("not a paper lookahead training checkpoint")
        self.config = dict(loaded["config"])
        self.checkpoint_path = str(path.resolve())
        self.checkpoint_sha256 = digest(path)
        self.precision = precision
        self.model = SavedPaperModel(self.config)
        self.model.load_state_dict(loaded["model"], strict=True)
        self.model.to(self.device).eval()
        self.reset()

    def _process(self, audio, frames, state):
        audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)
        frame_tensor = torch.as_tensor(frames, dtype=torch.float32, device=self.device).unsqueeze(0)
        core_state, visual_state = (None, None) if state is None else state
        # Both model components return new states and do not mutate input state.
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
                features, visual_state = self.model.visual(frame_tensor, visual_state)
                output, core_state = self.model.core(audio_tensor, features.transpose(1, 2).contiguous(), core_state)
        return output[0].float().cpu().numpy().copy(), (core_state, visual_state)

    def reset(self):
        """Start a fresh stream, discarding any unconsumed buffered input."""
        self.session = LookaheadSession(self._process)

    def push(self, audio, mouth_frame, *, sequence):
        """Accept an aligned tick; validation failures do not consume sequence."""
        return self.session.push(audio, np.asarray(mouth_frame, dtype=np.float32), sequence=sequence)

    def retry_ready(self):
        """Retry a processor error after push accepted a tick."""
        return self.session.retry_ready()

    def finish(self):
        """Pad only the end of the stream and return the remaining valid audio."""
        return self.session.finish()
