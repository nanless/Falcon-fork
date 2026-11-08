import os
import json
import random
import sys
import warnings

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .transform import get_preprocessing_pipelines
from .avspeech_dataset import (
    AVSpeechEvalDataset,
    MOUTH_FPS_DEFAULT,
    MOUTH_REF_SR_DEFAULT,
    _fit_mouth_frames,
    _warn_mouth_sample_rate,
)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def collate_with_near(batch):
    """
    Custom collate function for AVSpeechTrainDatasetWithNear.

    Each item is (mixture, source, mouth, filename, near_audios) where
    near_audios is a variable-length list of [T] tensors.

    Returns:
        mixtures:   [B, T]
        sources:    [B, T]
        mouths:     [B, fps_len, H, W]
        filenames:  list[str]
        near_audios: list[list[Tensor[T]]]  – per sample, per near speaker
    """
    mixtures, sources, mouths, filenames, near_list = zip(*batch)
    mixtures = torch.stack(mixtures)
    sources = torch.stack(sources)
    mouths = torch.stack([
        m if isinstance(m, torch.Tensor) else torch.from_numpy(m) for m in mouths
    ])
    return mixtures, sources, mouths, list(filenames), list(near_list)


class AVSpeechTrainDatasetWithNear(Dataset):
    """
    Training dataset that additionally loads near-field interferer audio for MDCL.

    near.json format (produced by generate_near_json.py):
        [
          [[near_path_1, n_samples_1], [near_path_2, n_samples_2], ...],  # for mix[0]
          ...
        ]

    If a near entry's path equals the mix path (fallback inserted by the
    generator), it is silently skipped so as not to penalise est~mix similarity.

    Opt-in knobs, all defaulting to the historical behaviour so existing results
    stay bit-identical:

    ``respect_index`` (default ``False``)
        ``False`` - ignore ``idx`` and draw ``random.randint`` from the whole
        index on every call. Consequences: the DataLoader's ``shuffle=True``
        is a no-op, sampling is *with replacement*, and ``__len__``
        (``train_samples``) is decoupled from the number of samples on disk.
        This is the default because every recorded run used it.
        ``True`` - honour ``idx`` exactly and let ``__len__`` return
        ``len(self.mix)``, so ``shuffle=True`` shuffles and an epoch covers
        each sample once. The audio crop offset stays random.

    ``mouth_fps`` (default ``25.0``)
        Frame rate of the stored mouth crops.

    ``mouth_ref_sr`` (default ``16000``)
        Audio sample rate the stored mouth crops were generated at: one video
        frame covers ``mouth_ref_sr / mouth_fps`` audio samples (640 by
        default). ``mouth_fps`` + ``mouth_ref_sr`` replace the hard-coded
        ``int((rand_start / 16000) * 25)`` in the mouth offset, so their
        defaults reproduce it exactly for *any* ``sample_rate``.
        ``sample_rate`` is used only to read/decode the audio and never moves
        the mouth timeline; when the two differ a single warning is emitted.

    ``pad_mouth`` (default ``False``)
        ``False`` - raise ``ValueError`` when the clip has fewer than
        ``int(segment * mouth_fps)`` frames (loud failure instead of a short
        tensor that breaks ``torch.stack``). ``True`` - repeat the last frame.
    """

    def __init__(
        self,
        json_dir: str,
        sample_rate: int = 16000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        samples: int = None,
        use_near: bool = True,
        respect_index: bool = False,
        mouth_fps: float = MOUTH_FPS_DEFAULT,
        mouth_ref_sr: int = MOUTH_REF_SR_DEFAULT,
        pad_mouth: bool = False,
    ):
        super().__init__()
        if json_dir is None:
            raise ValueError("json_dir must not be None")

        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.use_near = use_near
        self.respect_index = respect_index
        self.mouth_fps = float(mouth_fps)
        self.mouth_ref_sr = int(mouth_ref_sr)
        self.pad_mouth = pad_mouth
        _warn_mouth_sample_rate(self.sample_rate, self.mouth_ref_sr, self.mouth_fps)

        self.seg_len = int(segment * sample_rate)
        self.fps_len = int(segment * self.mouth_fps)

        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["train"]
        self.transform = transforms.Grayscale(num_output_channels=1)
        print("lipreading_preprocessing_func:", self.lipreading_preprocessing_func)

        # ── Load JSON files ──────────────────────────────────────────────────
        mix_json = os.path.join(json_dir, "mix.json")
        target_json = os.path.join(json_dir, "target.json")
        near_json = os.path.join(json_dir, "near.json")

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        with open(target_json, "r") as f:
            target_infos = json.load(f)

        near_infos = None
        if use_near and os.path.exists(near_json):
            with open(near_json, "r") as f:
                near_infos = json.load(f)
            print(f"[AVSpeechTrainDatasetWithNear] Loaded near.json with {len(near_infos)} entries.")
        elif use_near:
            print(f"[AVSpeechTrainDatasetWithNear] WARNING: near.json not found at {near_json}. "
                  "MDCL will be disabled for this split.")

        # ── Filter samples shorter than seg_len ──────────────────────────────
        self.mix: list = []
        self.sources: list = []
        self.near: list = []  # list of [[path, n_samples], ...] per sample

        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0

        for i in range(len(mix_infos) - 1, -1, -1):
            if mix_infos[i][1] < self.seg_len:
                drop_utt += 1
                drop_len += mix_infos[i][1]
                del mix_infos[i]
                del target_infos[i]
                if near_infos is not None:
                    del near_infos[i]
            else:
                self.mix.append(mix_infos[i])
                self.sources.append(target_infos[i])
                self.near.append(near_infos[i] if near_infos is not None else [])

        print(
            f"Drop {drop_utt} utts({drop_len / sample_rate / 3600:.2f} h) from {orig_len} "
            f"(shorter than {self.seg_len} samples)"
        )

        # ── Near-speaker sanity check (runs in main process, always visible) ─
        samples_with_near = sum(1 for n in self.near if len(n) > 0)
        print(f"[AVSpeechTrainDatasetWithNear] {samples_with_near}/{len(self.near)} "
              f"samples have near-speaker entries.")
        if use_near and samples_with_near > 0:
            # Verify we can actually read the first near file right now
            for entry in self.near:
                if entry:
                    test_path, test_n = entry[0]
                    try:
                        sf.read(test_path, start=0, stop=min(100, test_n), dtype="float32")
                        print(f"[AVSpeechTrainDatasetWithNear] Near file access OK: {test_path}")
                    except Exception as e:
                        print(f"[AVSpeechTrainDatasetWithNear] *** NEAR FILE ACCESS FAILED ***: "
                              f"{test_path}\n  Error: {e}\n"
                              f"  Check that  is mounted and accessible.")
                    break

        self.length = samples

    def __len__(self):
        if self.respect_index:
            return len(self.mix)
        return self.length

    def __getitem__(self, idx: int):
        EPS = 1e-8
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

        rand_start = random.randint(0, self.mix[mix_index][1] - self.seg_len)
        stop = rand_start + self.seg_len

        # ── Audio ────────────────────────────────────────────────────────────
        mix_source, _ = sf.read(
            self.mix[mix_index][0], start=rand_start, stop=stop, dtype="float32"
        )
        source, _ = sf.read(
            self.sources[mix_index][0], start=rand_start, stop=stop, dtype="float32"
        )

        # ── Mouth (visual) ───────────────────────────────────────────────────
        # Mouth frames follow the crop reference rate, never the audio rate:
        # this is the historical int((rand_start / 16000) * 25) when the
        # mouth_ref_sr/mouth_fps defaults are used, whatever sample_rate is.
        mouth_start = int((rand_start / self.mouth_ref_sr) * self.mouth_fps)
        source_mouth = self.lipreading_preprocessing_func(
            self.transform(
                torch.load(self.sources[mix_index][1], weights_only=True)
                .permute(0, 3, 1, 2)
            ).squeeze(1).numpy()
        )[mouth_start: self.fps_len + mouth_start]
        source_mouth = _fit_mouth_frames(
            source_mouth, self.fps_len, self.pad_mouth,
            "respect_index={}, rand_start={}, mouth_start={}, "
            "sample_rate={}, mouth_ref_sr={}, mouth_fps={}".format(
                self.respect_index, rand_start, mouth_start,
                self.sample_rate, self.mouth_ref_sr, self.mouth_fps),
        )

        mixture = torch.from_numpy(mix_source).mean(-1) if mix_source.ndim > 1 else torch.from_numpy(mix_source)
        source = torch.from_numpy(source).mean(-1) if source.ndim > 1 else torch.from_numpy(source)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=EPS, std=m_std)

        # ── Near-field interferer audio ──────────────────────────────────────
        near_audios = []
        if self.use_near:
            mix_path = self.mix[mix_index][0]
            for near_path, n_samples in self.near[mix_index]:
                # Skip fallback entries where generator substituted the mix path
                if near_path == mix_path:
                    continue
                # Skip near speakers whose file is shorter than one full segment
                if n_samples < self.seg_len:
                    continue
                # Clamp read offset: if the near file is a pre-segmented clip
                # (n_samples == seg_len or n_samples < stop), start from the
                # latest valid position instead of silently dropping the speaker.
                near_start = min(rand_start, n_samples - self.seg_len)
                near_stop = near_start + self.seg_len
                try:
                    near_wav, _ = sf.read(
                        near_path, start=near_start, stop=near_stop, dtype="float32"
                    )
                    near_t = (
                        torch.from_numpy(near_wav).mean(-1)
                        if near_wav.ndim > 1
                        else torch.from_numpy(near_wav)
                    )
                    if self.normalize_audio:
                        near_t = normalize_tensor_wav(near_t, eps=EPS, std=m_std)
                    near_audios.append(near_t)
                except Exception as e:
                    # warnings.warn is visible from DataLoader worker processes
                    # (unlike print, which may be buffered/suppressed in workers)
                    warnings.warn(
                        f"[AVSpeechTrainDatasetWithNear] could not read near file "
                        f"{near_path}: {e}",
                        stacklevel=2,
                    )
                    sys.stderr.flush()

        filename = self.mix[mix_index][0].split("/")[-1]
        return mixture, source, source_mouth, filename, near_audios


class AVSpeechDataModuleWithNear:
    """
    Data module with near-speaker support for MDCL training.

    Identical to the original AVSpeechDataModule except:
    - Accepts use_near (bool) parameter (matches config key in datamodule.data_config)
    - Uses AVSpeechTrainDatasetWithNear for training set
    - Uses collate_with_near so the train DataLoader yields near_audios
    - Val / test loaders are unchanged (MDCL is training-only)
    """

    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 1,
        sample_rate: int = 16000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        train_samples: int = 10000,
        use_near: bool = True,
        respect_index: bool = False,
        mouth_fps: float = MOUTH_FPS_DEFAULT,
        mouth_ref_sr: int = MOUTH_REF_SR_DEFAULT,
        pad_mouth: bool = False,
    ) -> None:
        super().__init__()
        if None in (train_dir, valid_dir, test_dir):
            raise ValueError("A JSON dir is None!")
        if n_src not in [1, 2]:
            raise ValueError(f"{n_src} is not in [1, 2]")

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
        self.use_near = use_near
        self.respect_index = respect_index
        self.mouth_fps = mouth_fps
        self.mouth_ref_sr = mouth_ref_sr
        self.pad_mouth = pad_mouth

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = AVSpeechTrainDatasetWithNear(
            json_dir=self.train_dir,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            samples=self.train_samples,
            use_near=self.use_near,
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
            collate_fn=collate_with_near,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
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
