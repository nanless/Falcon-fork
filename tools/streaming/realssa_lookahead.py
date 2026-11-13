"""Official RealSSA crop with real future context for 120ms/80ms training."""
import io
import random

import numpy as np
import soundfile as sf
import torch
from torchvision.transforms.functional import rgb_to_grayscale

from tools.streaming.realssa_tar import RealSSATarDataset, SEGMENT_FRAMES, SEGMENT_SAMPLES
from tools.streaming.lookahead_training import CHUNK_SAMPLES, LOOKAHEAD_SAMPLES, SAMPLES_PER_FRAME
from tools.streaming.common import SR

# The final 2s output block is partial: 16 full 120ms blocks and 80ms.
# Complete the block and its 80ms preview, hence 2.12s = 53 video frames.
INPUT_SAMPLES = ((SEGMENT_SAMPLES + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES) * CHUNK_SAMPLES + LOOKAHEAD_SAMPLES
INPUT_FRAMES = INPUT_SAMPLES // SAMPLES_PER_FRAME


class RealSSALookaheadDataset(RealSSATarDataset):
    """Return (2.12s mixture, 2s target, 53 frames), channel zero.

    Crop selection is the same as RealSSATarDataset's 2-second crop, not a
    mandatory 2.12s crop. Keep context up to the shortest of the mixture,
    target, and mouth video, rounded down to complete 40 ms frames. The
    returned tensors are zero-padded. With return_valid_samples, the fourth
    element is the count of real target samples before padding. All real video
    frames use one shared spatial crop and flip.
    """
    def __init__(self, *args, return_valid_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_valid_samples = return_valid_samples

    def __getitem__(self, index):
        record = self.samples[index]
        sample_id = record["id"]
        shard = self.shards[record["shard"]]["path"]
        payload = {}
        with open(shard, "rb") as stream:
            for kind in ("mix", "target", "mouth"):
                offset, size = record["fields"][kind]
                stream.seek(offset)
                payload[kind] = stream.read(size)
                if len(payload[kind]) != size:
                    raise ValueError(f"truncated {kind} for {sample_id} in {shard}")
        audio = []
        for kind in ("mix", "target"):
            wav, sr = sf.read(io.BytesIO(payload[kind]), dtype="float32", always_2d=True)
            if sr != SR or wav.shape[1] != 4 or not np.isfinite(wav).all():
                raise ValueError(f"invalid {kind} audio for {sample_id}: {sr} Hz, {wav.shape}")
            audio.append(wav[:, 0].copy())
        frames = torch.load(io.BytesIO(payload["mouth"]), map_location="cpu", weights_only=True)
        if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8 or frames.ndim not in (3, 4):
            raise ValueError(f"invalid mouth tensor for {sample_id}")
        available = min(len(audio[0]) // SAMPLES_PER_FRAME,
                        len(audio[1]) // SAMPLES_PER_FRAME, len(frames))
        if available < 1:
            raise ValueError(f"no aligned 40 ms frame for {sample_id}")
        last = max(0, available - SEGMENT_FRAMES)
        rng = random.Random((self.seed << 64) ^ (self.epoch << 32) ^ index)
        if self.crop == "start":
            start = 0
        elif self.crop == "center":
            start = last // 2
        else:
            start = rng.randint(0, last)
        count = min(INPUT_FRAMES, available - start)
        target_count = min(SEGMENT_FRAMES, count)
        begin = start * SAMPLES_PER_FRAME
        mix = torch.zeros(INPUT_SAMPLES, dtype=torch.float32)
        mix[:count*SAMPLES_PER_FRAME] = torch.from_numpy(audio[0][begin:begin+count*SAMPLES_PER_FRAME])
        target = torch.zeros(SEGMENT_SAMPLES, dtype=torch.float32)
        target[:target_count*SAMPLES_PER_FRAME] = torch.from_numpy(audio[1][begin:begin+target_count*SAMPLES_PER_FRAME])
        raw = frames[start:start+count]
        if raw.ndim == 4 and raw.shape[1:] == (96, 96, 3):
            gray = rgb_to_grayscale(raw.permute(0, 3, 1, 2)).squeeze(1)
        elif raw.ndim == 3 and raw.shape[1:] == (96, 96):
            gray = raw
        else:
            raise ValueError(f"invalid mouth dimensions for {sample_id}: {tuple(raw.shape)}")
        if self.augment:
            left, top = rng.randint(0, 8), rng.randint(0, 8)
            gray = gray[:, top:top+88, left:left+88]
            if rng.random() < .5:
                gray = gray.flip(-1)
        else:
            gray = gray[:, 4:92, 4:92]
        video = torch.zeros((INPUT_FRAMES, 88, 88), dtype=torch.float32)
        video[:count] = (gray.float() / 255.0 - .421) / .165
        if self.return_valid_samples:
            return mix, target, video, target_count * SAMPLES_PER_FRAME
        return mix, target, video
