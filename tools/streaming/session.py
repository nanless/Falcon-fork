"""Session-level 160ms streaming API; camera tracking and transport are external."""
import numpy as np
import torch
if __package__:
    from .common import FRAMES, SAMPLES, SAMPLES_PER_FRAME, mouth_tensor, read_checkpoint
else:
    from common import FRAMES, SAMPLES, SAMPLES_PER_FRAME, mouth_tensor, read_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming

class StreamingSession:
    """One speaker/session, own state. Times are integer samples on the audio clock.

    `push` accepts a complete 160ms mono audio block and 4 mouth crops. Each crop
    timestamp must be no later than its 40ms interval start. This conservative
    acquisition rule is stricter than block causality; the model may use every
    frame in the current 160ms block for any output within that block. A reset starts a new independent stream.
    A partial final chunk specifies valid_samples, pads audio with zeros, and
    closes the stream until reset. Video bins covering valid audio are
    timestamp-checked; other supplied crops are replaced with the last valid
    crop and their timestamps are ignored. An endpoint should select/track faces externally and map timestamps to this
    audio clock. Calls on one session must be serialized by the caller.
    """
    def __init__(self, checkpoint, device="cpu", max_frame_age_samples=2560):
        self.device = torch.device(device)
        self.max_frame_age_samples = int(max_frame_age_samples)
        if self.max_frame_age_samples < 0: raise ValueError("invalid maximum frame age")
        ckpt = read_checkpoint(checkpoint, self.device)
        self.model = FalconStreaming().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()
        self.training_status = ckpt.get("training_status", "legacy_checkpoint")
        self.reset()

    @classmethod
    def from_model(cls, model, *, training_status="unknown", max_frame_age_samples=2560):
        """Create independent state using one immutable shared eval-mode model.

        External code must serialize calls if its deployment requires it; do not
        change weights, train/eval mode, or device while sessions are active.
        """
        if model.training:
            raise ValueError("shared model must be in eval mode")
        obj = cls.__new__(cls)
        obj.model = model
        obj.device = next(model.parameters()).device
        obj.training_status = training_status
        obj.max_frame_age_samples = int(max_frame_age_samples)
        if obj.max_frame_age_samples < 0: raise ValueError("invalid maximum frame age")
        obj.reset()
        return obj

    def reset(self):
        self.state = None
        self.next_sequence = 0
        self.next_sample = 0
        self.last_video_sample = None
        self.finished = False

    def push(self, audio, mouth, *, sequence, start_sample, video_samples, valid_samples=SAMPLES):
        if self.finished:
            raise ValueError("partial final chunk already received; reset before continuing")
        if type(valid_samples) is not int or not 1 <= valid_samples <= SAMPLES:
            raise ValueError("valid_samples must be an integer between 1 and 2560")
        if type(sequence) is not int or sequence != self.next_sequence:
            raise ValueError(f"expected sequence {self.next_sequence}")
        if type(start_sample) is not int or start_sample != self.next_sample:
            raise ValueError(f"expected start_sample {self.next_sample}")
        if len(video_samples) != FRAMES or any(type(i) is not int for i in video_samples):
            raise ValueError("video_samples must contain four integer audio-clock sample timestamps")
        # Prevent padded tail frames from providing future video to attention.
        valid_frames = (valid_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
        if any(b < a for a, b in zip(video_samples[:valid_frames], video_samples[1:valid_frames])):
            raise ValueError("video timestamps must be nondecreasing")
        if self.last_video_sample is not None and video_samples[0] < self.last_video_sample:
            raise ValueError("video timestamp moved backwards")
        for i, timestamp in enumerate(video_samples[:valid_frames]):
            nominal = start_sample + i * SAMPLES_PER_FRAME
            if timestamp > nominal:
                raise ValueError("video frame is from the future for its audio interval")
            if i * SAMPLES_PER_FRAME < valid_samples and nominal - timestamp > self.max_frame_age_samples:
                raise ValueError("video frame is too old")
        x = np.asarray(audio)
        if x.dtype != np.float32 or x.shape != (SAMPLES,) or not np.isfinite(x).all():
            raise ValueError("audio must be finite float32 [2560]")
        if valid_samples < SAMPLES and np.any(x[valid_samples:] != 0):
            raise ValueError("audio after valid_samples must be zero")
        v = mouth_tensor(mouth)
        if v.shape != (FRAMES, 88, 88):
            raise ValueError("mouth must have four uint8 96x96 crops")
        if valid_frames < FRAMES:
            v = v.clone()
            v[valid_frames:] = v[valid_frames - 1]
        with torch.inference_mode():
            output, new_state = self.model.forward_chunk(torch.from_numpy(x.copy())[None].to(self.device),
                                                          v[None].to(self.device), self.state)
            result = output[0].float().cpu().numpy().copy()
        if not np.isfinite(result).all(): raise RuntimeError("non-finite model output")
        self.state = new_state
        self.next_sequence += 1
        self.next_sample += valid_samples
        self.last_video_sample = video_samples[(valid_samples - 1) // SAMPLES_PER_FRAME]
        self.finished = valid_samples < SAMPLES
        return result
