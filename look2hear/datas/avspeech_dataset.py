import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import json
import warnings
from typing import Dict, Iterable, List, Iterator
from .transform import get_preprocessing_pipelines
from torchvision import transforms
import matplotlib.pyplot as plt

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Mouth / audio alignment contract
# ---------------------------------------------------------------------------
# The stored mouth clips were generated at MOUTH_REF_SR_DEFAULT (16 kHz): video
# frame k covers audio samples [k * 16000/25, (k+1) * 16000/25)
# == [640k, 640k + 640). That reference rate -- not ``sample_rate`` -- is what
# maps an audio sample offset onto a mouth frame. The two are deliberately
# independent because ``sample_rate`` only governs how the audio is read; wiring
# the mouth timeline to it silently shifts the video whenever the audio rate is
# not 16 kHz. ``mouth_ref_sr``/``mouth_fps`` expose the two historical constants
# (``/ 16000``, ``* 25``) with their default behaviour unchanged.
MOUTH_FPS_DEFAULT = 25.0
MOUTH_REF_SR_DEFAULT = 16000


def _warn_mouth_sample_rate(
    sample_rate,
    mouth_ref_sr=MOUTH_REF_SR_DEFAULT,
    mouth_fps=MOUTH_FPS_DEFAULT,
):
    """Warn when the audio rate differs from the rate the mouth crops were built for.

    The historical code hard-coded ``16000`` and ``25`` when converting an audio
    sample offset into a mouth-frame offset. The mouth timeline keeps using that
    reference rate (``mouth_ref_sr``/``mouth_fps``), so a different
    ``sample_rate`` -- which only controls audio decoding -- does not shift the
    video; it does mean the audio and video timelines come from different sample
    rates. Surface that once per dataset instance instead of per crop.
    """
    if int(sample_rate) != int(mouth_ref_sr):
        warnings.warn(
            "sample_rate={} != mouth_ref_sr={}: the stored mouth crops were built for "
            "{} Hz audio at {:g} fps (one video frame = {:g} audio samples), so the "
            "audio/video timeline stays aligned with mouth_ref_sr/mouth_fps rather "
            "than sample_rate/mouth_fps. sample_rate is only used for audio reading; "
            "verify this dataset before training.".format(
                sample_rate, mouth_ref_sr, mouth_ref_sr, mouth_fps,
                float(mouth_ref_sr) / float(mouth_fps),
            ),
            stacklevel=3,
        )


def _fit_mouth_frames(source_mouth, expected, pad_mouth, where):
    """Validate the mouth clip length; optionally pad it by repeating its last frame.

    The loader slices ``[mouth_start : mouth_start + fps_len]``, so a clip shorter
    than the audio window yields fewer frames than the model expects. Historically
    that silently produced a short clip which then broke ``torch.stack`` inside the
    default collate. Now the mismatch is explicit.

    Parameters
    ----------
    source_mouth : np.ndarray
        Clip of shape ``[T, H, W]`` returned by the lipreading pipeline.
    expected : int
        Frames required for the requested segment, ``int(segment * mouth_fps)``.
    pad_mouth : bool
        ``False`` (default) raises ``ValueError`` when frames are missing.
        ``True`` pads by repeating the last frame (opt-in; keeps default numeric
        behaviour unchanged for datasets that already have enough frames).
    where : str
        Short context string for the error message.
    """
    got = int(source_mouth.shape[0])
    if got == expected:
        return source_mouth
    if got > expected:
        # Unreachable through ``[mouth_start : mouth_start + fps_len]`` slicing; kept
        # so a future indexing change cannot silently return a longer clip.
        return source_mouth[:expected]
    if pad_mouth and got > 0:
        pad = expected - got
        return np.concatenate(
            [source_mouth, np.repeat(source_mouth[-1:], pad, axis=0)], axis=0
        )
    raise ValueError(
        "mouth clip has {} frame(s) but {} are required for this segment ({}). "
        "The clip is shorter than the audio window, so audio and video would drift. "
        "Pass pad_mouth=True to repeat the last frame, request a shorter segment, or "
        "fix the dataset (each video frame covers mouth_ref_sr/mouth_fps audio "
        "samples)."
        .format(got, expected, where)
    )


class AVSpeechTrainDataset(Dataset):
    """Training dataset that streams mixture/target audio + a pre-cropped mouth clip.

    Opt-in knobs, all defaulting to the historical behaviour so existing results
    stay bit-identical:

    ``respect_index`` (default ``False``)
        ``False`` - historical mode: ``__getitem__`` ignores ``idx`` and draws
        ``random.randint`` from the whole index on every call. Consequences: the
        DataLoader's ``shuffle=True`` is a no-op, sampling is *with replacement*,
        and ``__len__`` (``train_samples``) is decoupled from the number of
        samples on disk. Kept as the default because every recorded run used it.
        ``True`` - honour ``idx`` exactly, and ``__len__`` returns ``len(self.mix)``
        so an epoch covers each sample once and ``shuffle=True`` becomes meaningful.
        The audio crop offset stays random, so successive epochs still see
        different windows of the same utterance.

    ``mouth_fps`` (default ``25.0``)
        Frame rate of the stored mouth crops.

    ``mouth_ref_sr`` (default ``16000``)
        Audio sample rate the stored mouth crops were generated at: one video
        frame covers ``mouth_ref_sr / mouth_fps`` audio samples (640 by default).
        ``mouth_fps`` + ``mouth_ref_sr`` replace the historical hard-coded
        ``int((rand_start / 16000) * 25)`` in the mouth offset, so their defaults
        reproduce that expression exactly for *any* ``sample_rate``.
        ``sample_rate`` is used only to read/decode the audio and never moves the
        mouth timeline; when the two differ a single warning is emitted.

    ``pad_mouth`` (default ``False``)
        ``False`` - raise ``ValueError`` if the clip has fewer than
        ``int(segment * mouth_fps)`` frames (loud failure instead of a short tensor
        that breaks ``torch.stack``).
        ``True`` - pad by repeating the last frame.
    """
    def __init__(
        self,
        json_dir: str = None,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        samples: int = None,
        respect_index: bool = False,
        mouth_fps: float = MOUTH_FPS_DEFAULT,
        mouth_ref_sr: int = MOUTH_REF_SR_DEFAULT,
        pad_mouth: bool = False,
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["train"]
        self.transform = transforms.Grayscale(num_output_channels=1)
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)

        self.respect_index = respect_index
        self.mouth_fps = float(mouth_fps)
        self.mouth_ref_sr = int(mouth_ref_sr)
        self.pad_mouth = pad_mouth
        _warn_mouth_sample_rate(self.sample_rate, self.mouth_ref_sr, self.mouth_fps)

        self.seg_len = int(segment * sample_rate)
        self.fps_len = int(segment * self.mouth_fps)

        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["target"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []

        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        print(len(mix_infos), len(sources_infos[0]))

        for i in range(len(mix_infos) - 1, -1, -1):
            if mix_infos[i][1] < self.seg_len:
                drop_utt = drop_utt + 1
                drop_len = drop_len + mix_infos[i][1]
                del mix_infos[i]
                for src_inf in sources_infos:
                    del src_inf[i]
            else:
                for src_inf in sources_infos:
                    self.mix.append(mix_infos[i])
                    self.sources.append(src_inf[i])

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
            )
        )
        self.length = samples

    def __len__(self):
        if self.respect_index:
            return len(self.mix)
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.respect_index:
            # Opt-in: honour idx so shuffle=True actually shuffles and an
            # epoch covers every sample once (no replacement).
            mix_index = idx if idx >= 0 else idx + len(self.mix)
            if not 0 <= mix_index < len(self.mix):
                raise IndexError(
                    "respect_index=True: idx={} out of range for {} samples"
                    .format(idx, len(self.mix))
                )
        else:
            # Historical default: ignore idx, draw uniformly with replacement.
            mix_index = random.randint(0, len(self.mix) - 1)

        if self.test:
            rand_start = 0
            stop = None
        else:
            # Use mix_index instead of idx
            rand_start = random.randint(0, self.mix[mix_index][1] - self.seg_len)
            stop = rand_start + self.seg_len

        # Use mix_index for all data access
        mix_source, _ = sf.read(
            self.mix[mix_index][0], start=rand_start, stop=stop, dtype="float32"
        )
        source = sf.read(
            self.sources[mix_index][0], start=rand_start, stop=stop, dtype="float32"
        )[0]

        # Mouth frames follow the crop reference rate, never the audio rate:
        # this is the historical int((rand_start / 16000) * 25) when the
        # mouth_ref_sr/mouth_fps defaults are used, whatever sample_rate is.
        mouth_start = int((rand_start / self.mouth_ref_sr) * self.mouth_fps)

        source_mouth = self.lipreading_preprocessing_func(
            self.transform(torch.load(self.sources[mix_index][1], weights_only=True).permute(0,3,1,2)).squeeze(1).numpy()
        )[mouth_start : self.fps_len + mouth_start]
        source_mouth = _fit_mouth_frames(
            source_mouth, self.fps_len, self.pad_mouth,
            "respect_index={}, rand_start={}, mouth_start={}, "
            "sample_rate={}, mouth_ref_sr={}, mouth_fps={}".format(
                self.respect_index, rand_start, mouth_start,
                self.sample_rate, self.mouth_ref_sr, self.mouth_fps),
        )

        # F5 fix (docs/REPRODUCTION_PLAN.md §6.1): `sf.read` returns (samples,) for mono
        # wavs and (samples, channels) for multi-channel wavs. A bare ``mean(-1)`` on a
        # mono array averages over *time* and collapses the waveform to a 0-dim tensor,
        # which later makes ``torch.stft`` fail. Only average over the channel axis when
        # it exists; keep mono 1-D as-is (same guard as the training set, near.py:176-177).
        source = torch.from_numpy(source).mean(-1) if source.ndim > 1 else torch.from_numpy(source)
        mixture = torch.from_numpy(mix_source).mean(-1) if mix_source.ndim > 1 else torch.from_numpy(mix_source)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)

        return mixture, source, source_mouth, self.mix[mix_index][0].split("/")[-1]


class AVSpeechEvalDataset(Dataset):
    def __init__(
        self,
        json_dir: str = None,
        sample_rate: int = 8000,
        normalize_audio: bool = False,
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["val"]
        self.transform = transforms.Grayscale(num_output_channels=1)
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)

        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["target"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        print(len(mix_infos), len(sources_infos[0]))
        
        for i in range(len(mix_infos)):
            for src_inf in sources_infos:
                self.mix.append(mix_infos[i])
                self.sources.append(src_inf[i])

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 3600, orig_len, None
            )
        )
        self.length = orig_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        
        rand_start = 0
        stop = None

        mix_source, _ = sf.read(
            self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
        )
        source = sf.read(
            self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
        )[0]

        source_mouth = self.lipreading_preprocessing_func(
            self.transform(torch.load(self.sources[idx][1], weights_only=True).permute(0,3,1,2)).squeeze(1).numpy()
        )
        
        # F5 fix (docs/REPRODUCTION_PLAN.md §6.1): mono wav -> keep 1-D; multi-channel -> channel mean.
        # See the identical guard in AVSpeechTrainDataset.__getitem__ above.
        source = torch.from_numpy(source).mean(-1) if source.ndim > 1 else torch.from_numpy(source)
        mixture = torch.from_numpy(mix_source).mean(-1) if mix_source.ndim > 1 else torch.from_numpy(mix_source)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)

        # Read near-speaker count from _meta.json (same stem as mix wav)
        n_near = 0
        mix_path = self.mix[idx][0]
        meta_path = mix_path.replace(".wav", "_meta.json")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            n_near = sum(1 for c in meta.get("components", []) if c.get("type") == "near")
        except Exception:
            pass

        return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1], n_near

class AVSpeechDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_samples: int = 10000,
        respect_index: bool = False,
        mouth_fps: float = MOUTH_FPS_DEFAULT,
        mouth_ref_sr: int = MOUTH_REF_SR_DEFAULT,
        pad_mouth: bool = False,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_samples = train_samples
        self.respect_index = respect_index
        self.mouth_fps = mouth_fps
        self.mouth_ref_sr = mouth_ref_sr
        self.pad_mouth = pad_mouth

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = AVSpeechTrainDataset(
            json_dir=self.train_dir,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            samples=self.train_samples,
            respect_index=self.respect_index,
            mouth_fps=self.mouth_fps,
            mouth_ref_sr=self.mouth_ref_sr,
            pad_mouth=self.pad_mouth,
        )
        self.data_val = AVSpeechEvalDataset(
            json_dir=self.valid_dir,
            sample_rate=self.sample_rate,
            normalize_audio=self.normalize_audio,
        )
        self.data_test = AVSpeechEvalDataset(
            json_dir=self.test_dir,
            sample_rate=self.sample_rate,
            normalize_audio=self.normalize_audio,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            #batch_size=self.batch_size,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            #batch_size=self.batch_size,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test

# if __name__ == "__main__":
#     from tqdm import tqdm
#     val_set = AVSpeechTrainDataset(
#         "${DATA_ROOT}/av-tiger-lrs3/DataPreProcess/LRS3_multi/tr",
#         n_src=2,
#         sample_rate=16000,
#         # segment=conf["data"]["segment"],
#         segment=2.0,
#         normalize_audio=False
#     )
#     for idx in tqdm(range(len(val_set))):
#         mixture, sources, sources_mouths, _ = val_set[idx]
#         import pdb; pdb.set_trace()
