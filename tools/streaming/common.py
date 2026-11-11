"""Shared input/checkpoint utilities for the 120 ms causal prototype."""
import json
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
FPS = 25
SAMPLES = 1920
FRAMES = 3
SAMPLES_PER_FRAME = 640
CHECKPOINT_FORMAT = "falcon-streaming-v4"


def mouth_tensor(frames):
    """Convert stored uint8 T,H,W,(RGB) crops to normalized T,88,88."""
    frames = torch.as_tensor(frames)
    if frames.dtype != torch.uint8:
        raise ValueError("mouth frames must be unnormalized uint8")
    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = frames.float()
        frames = frames[..., 0] * 0.2989 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114
    elif frames.ndim == 3:
        frames = frames.float()
    else:
        raise ValueError(f"expected [T,H,W,3] or [T,H,W], got {tuple(frames.shape)}")
    if frames.shape[1:] != (96, 96):
        raise ValueError(f"expected 96x96 stored mouth crops, got {tuple(frames.shape)}")
    return (frames[:, 4:92, 4:92] / 255.0 - 0.421) / 0.165


def load_mono(path, start=0, frames=-1):
    x, rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    if rate != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {rate}")
    if x.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, got {x.shape[1]} channels")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite audio")
    return torch.from_numpy(x[:, 0].copy())


class IndexedAudioVisualDataset(Dataset):
    """Use Falcon mix.json/target.json; crop on the 25fps (640-sample) grid."""
    def __init__(self, directory, chunks):
        directory = Path(directory)
        with (directory / "mix.json").open() as f:
            mixtures = json.load(f)
        with (directory / "target.json").open() as f:
            targets = json.load(f)
        if len(mixtures) != len(targets):
            raise ValueError("mix and target indexes have different lengths")
        for i, (mix, target) in enumerate(zip(mixtures, targets)):
            if len(mix) < 2 or len(target) < 3:
                raise ValueError(f"invalid index entry at row {i}")
            if Path(mix[0]).name != Path(target[0]).name:
                raise ValueError(f"mix/target identity mismatch at row {i}: {mix[0]} vs {target[0]}")
        self.samples = chunks * SAMPLES
        self.frame_count = chunks * FRAMES
        self.items = [(m, t) for m, t in zip(mixtures, targets)
                      if m[1] >= self.samples and t[2] >= self.samples]
        if not self.items:
            raise ValueError("no index entries are long enough for the selected chunk count")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        mix, target = self.items[index]
        # Limit the common 40 ms grid by both indexed audio lengths and the
        # actual number of stored mouth frames before sampling a start.
        raw = torch.load(target[1], map_location="cpu", weights_only=True)
        if len(raw) < self.frame_count:
            raise ValueError(f"mouth video is too short for the selected chunk count: {target[1]}")
        audio_starts = (min(int(mix[1]), int(target[2])) - self.samples) // SAMPLES_PER_FRAME
        video_starts = len(raw) - self.frame_count
        frame_start = random.randint(0, min(audio_starts, video_starts))
        start = frame_start * SAMPLES_PER_FRAME
        mixture = load_mono(mix[0], start, self.samples)
        source = load_mono(target[0], start, self.samples)
        video = mouth_tensor(raw[frame_start:frame_start + self.frame_count])
        if len(mixture) != self.samples or len(source) != self.samples or len(video) != self.frame_count:
            raise ValueError(f"incomplete aligned segment for {mix[0]}")
        return mixture, source, video


def new_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def read_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError("invalid FalconStreaming checkpoint")
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("incompatible FalconStreaming checkpoint: expected falcon-streaming-v4 (120 ms)")
    state = ckpt.get("state_dict")
    version = state.get("_streaming_format_version") if isinstance(state, dict) else None
    if not isinstance(version, torch.Tensor) or version.shape != () or version.dtype != torch.int64 or version.item() != 4:
        raise ValueError("incompatible FalconStreaming model state: expected scalar int64 _streaming_format_version=4")
    config = ckpt.get("model_config")
    if (not isinstance(config, dict) or set(config) != {"channels", "dilations"}
            or type(config["channels"]) is not int or config["channels"] < 1
            or not isinstance(config["dilations"], (tuple, list)) or not config["dilations"]
            or any(type(d) is not int or d < 1 for d in config["dilations"])):
        raise ValueError("invalid or missing streaming model_config")
    return ckpt
