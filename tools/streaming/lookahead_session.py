"""Incremental 40 ms ingestion for 120 ms output with 80 ms lookahead.

Audio and video must share the same 25 Hz clock. The processor sees five
consecutive ticks, but commits only the first three and returns its state at
that boundary. A stream ending mid-tick may supply a short final audio tick.
"""
from __future__ import annotations

import numpy as np

TICK_SAMPLES = 640
BLOCK_TICKS = 3
LOOKAHEAD_TICKS = 2


class LookaheadSession:
    def __init__(self, processor):
        self.processor = processor
        self.audio = []
        self.video = []
        self.valid = []
        self.state = None
        self.closed = False
        self.short_seen = False
        self.next_sequence = 0

    def push(self, audio, video, *, sequence):
        if self.closed or self.short_seen:
            raise ValueError("stream already ended or a short final tick was submitted")
        if type(sequence) is not int or sequence != self.next_sequence:
            raise ValueError("sequence must be the next consecutive 40 ms tick")
        audio = np.asarray(audio, dtype=np.float32)
        video = np.asarray(video)
        if audio.ndim != 1 or not 0 < audio.size <= TICK_SAMPLES:
            raise ValueError("each audio tick must contain 1 to 640 samples")
        if not np.isfinite(audio).all() or not np.isfinite(video).all():
            raise ValueError("audio and video must contain finite values")
        if video.shape != (88, 88):
            raise ValueError("each video tick must be an 88x88 mouth frame")
        count = audio.size
        self.short_seen = count < TICK_SAMPLES
        self.audio.append(np.pad(audio, (0, TICK_SAMPLES - count)))
        self.video.append(video.copy())
        self.valid.append(count)
        self.next_sequence += 1
        return self._ready()

    def _ready(self):
        outputs = []
        while len(self.audio) >= BLOCK_TICKS + LOOKAHEAD_TICKS and sum(self.valid[:BLOCK_TICKS]):
            sound = np.concatenate(self.audio[:BLOCK_TICKS + LOOKAHEAD_TICKS])
            frames = np.stack(self.video[:BLOCK_TICKS + LOOKAHEAD_TICKS])
            result, new_state = self.processor(sound, frames, self.state)
            result = np.asarray(result)
            if result.shape != (BLOCK_TICKS * TICK_SAMPLES,):
                raise ValueError("processor must return exactly 1920 samples")
            valid = sum(self.valid[:BLOCK_TICKS])
            outputs.append(result[:valid].copy())
            self.state = new_state
            del self.audio[:BLOCK_TICKS]
            del self.video[:BLOCK_TICKS]
            del self.valid[:BLOCK_TICKS]
        return outputs

    def finish(self):
        if self.closed:
            raise ValueError("stream already ended")
        self.closed = True
        outputs = []
        if not self.audio:
            return outputs
        padding_frame = np.zeros_like(self.video[-1])
        while sum(self.valid):
            while len(self.audio) < BLOCK_TICKS + LOOKAHEAD_TICKS:
                self.audio.append(np.zeros(TICK_SAMPLES, dtype=np.float32))
                self.video.append(padding_frame.copy())
                self.valid.append(0)
            outputs.extend(self._ready())
        return outputs
