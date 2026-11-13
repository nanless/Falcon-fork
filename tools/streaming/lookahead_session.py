"""Incremental 40 ms ingestion for 120 ms output with 80 ms lookahead.

Audio and video must share the same 25 Hz clock. The processor sees five
consecutive ticks, but commits only the first three and returns its state at
that boundary. A stream ending mid-tick may supply a short final audio tick.

The processor must treat its input state as read-only and return a new state.
With accept_missing_video, None contributes a zero frame and the processor
also receives a list of presence flags for the five input ticks.
Once push has accepted a tick, an execution error leaves that tick queued;
call retry_ready() rather than pushing the same sequence again. If finish()
fails, call finish() again; previously produced tail blocks are held until
finish succeeds, so the caller receives them exactly once.
"""
from __future__ import annotations

import numpy as np

TICK_SAMPLES = 640
BLOCK_TICKS = 3
LOOKAHEAD_TICKS = 2


class LookaheadSession:
    def __init__(self, processor, *, accept_missing_video=False):
        self.processor = processor
        self.accept_missing_video = accept_missing_video
        self.audio = []
        self.video = []
        self.video_present = []
        self.valid = []
        self.state = None
        self.closed = False
        self.short_seen = False
        self.next_sequence = 0
        self._pending_error = False
        self._finish_started = False
        self._finish_outputs = []

    def push(self, audio, video, *, sequence):
        if self.closed or self._finish_started or self.short_seen:
            raise ValueError("stream already ended or a short final tick was submitted")
        if self._pending_error:
            raise RuntimeError("processor failed after accepting a tick; call retry_ready()")
        if type(sequence) is not int or sequence != self.next_sequence:
            raise ValueError("sequence must be the next consecutive 40 ms tick")
        audio = np.asarray(audio, dtype=np.float32)
        video_present = video is not None
        video = (np.zeros((88, 88), dtype=np.float32)
                 if self.accept_missing_video and not video_present else np.asarray(video))
        if audio.ndim != 1 or not 0 < audio.size <= TICK_SAMPLES:
            raise ValueError("each audio tick must contain 1 to 640 samples")
        if not video_present and not self.accept_missing_video:
            raise ValueError("each video tick must be an 88x88 mouth frame")
        if not np.isfinite(audio).all() or not np.isfinite(video).all():
            raise ValueError("audio and video must contain finite values")
        if video.shape != (88, 88):
            raise ValueError("each video tick must be an 88x88 mouth frame")
        count = audio.size
        self.short_seen = count < TICK_SAMPLES
        self.audio.append(np.pad(audio, (0, TICK_SAMPLES - count)))
        self.video.append(video.copy())
        self.video_present.append(video_present)
        self.valid.append(count)
        self.next_sequence += 1
        return self._ready()

    def _ready(self):
        outputs = []
        while len(self.audio) >= BLOCK_TICKS + LOOKAHEAD_TICKS and sum(self.valid[:BLOCK_TICKS]):
            sound = np.concatenate(self.audio[:BLOCK_TICKS + LOOKAHEAD_TICKS])
            frames = np.stack(self.video[:BLOCK_TICKS + LOOKAHEAD_TICKS])
            try:
                if self.accept_missing_video:
                    present = self.video_present[:BLOCK_TICKS + LOOKAHEAD_TICKS]
                    result, new_state = self.processor(sound, frames, self.state, present)
                else:
                    result, new_state = self.processor(sound, frames, self.state)
                result = np.asarray(result)
                if result.shape != (BLOCK_TICKS * TICK_SAMPLES,):
                    raise ValueError("processor must return exactly 1920 samples")
                if not np.isfinite(result).all():
                    raise ValueError("processor returned non-finite samples")
            except Exception:
                self._pending_error = True
                raise
            valid = sum(self.valid[:BLOCK_TICKS])
            outputs.append(result[:valid].copy())
            self.state = new_state
            del self.audio[:BLOCK_TICKS]
            del self.video[:BLOCK_TICKS]
            del self.video_present[:BLOCK_TICKS]
            del self.valid[:BLOCK_TICKS]
        return outputs

    def retry_ready(self):
        """Retry a failed push without resubmitting its already accepted tick."""
        if self.closed or self._finish_started:
            raise ValueError("after finish has started, retry by calling finish()")
        if not self._pending_error:
            raise ValueError("no failed processor call to retry")
        self._pending_error = False
        return self._ready()

    def finish(self):
        if self.closed:
            raise ValueError("stream already ended")
        if self._pending_error and not self._finish_started:
            raise RuntimeError("retry the failed push with retry_ready() before finish")
        self._finish_started = True
        self._pending_error = False
        if not self.audio:
            self.closed = True
            return self._finish_outputs
        padding_frame = np.zeros_like(self.video[-1])
        while sum(self.valid):
            while len(self.audio) < BLOCK_TICKS + LOOKAHEAD_TICKS:
                self.audio.append(np.zeros(TICK_SAMPLES, dtype=np.float32))
                self.video.append(padding_frame.copy())
                self.video_present.append(False)
                self.valid.append(0)
            self._finish_outputs.extend(self._ready())
        self.closed = True
        return self._finish_outputs
