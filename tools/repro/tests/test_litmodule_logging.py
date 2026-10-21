"""Phase-1 T2: regression tests for the 3 Lightning modules' logging API (F2).

Before the fix, `on_validation_epoch_end` called the Comet-only
``self.logger.experiment.log(dict)`` which raises ``AttributeError`` with
CSVLogger/TensorBoardLogger and with ``logger=None``.  These tests run a real
(CPU, dummy-model) Lightning `fit` with `CSVLogger` for all three modules, plus a
no-logger smoke test, and check that the original metric keys are still logged.

Run:
    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_litmodule_logging.py -q
"""
import glob
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from look2hear.system import (  # noqa: E402
    AudioVisualLightningModule,
    AudioVisualLightningModuleNASSv2,
)
from look2hear.system.av_litmodule_mdcl import AudioVisualLightningModuleWithMDCL  # noqa: E402
from look2hear.losses import PITLossWrapper, pairwise_neg_snr, pairwise_neg_sisdr  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Dummy models / data (mirrors the real interfaces but runs in ms on CPU)
# ──────────────────────────────────────────────────────────────────────────────
class _DummyVideo(nn.Module):
    def forward(self, x):
        return x.new_zeros(x.shape[0], 512, 8)


def _n_t(T):
    return max((T - 256) // 128 + 1, 1)


class _DummyPlain(nn.Module):
    """Mimics plain `Falcon`: single-tensor output + MDCL helpers."""

    mdcl_alpha = 0.15
    mdcl_margin = 0.0

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, mix, mouth_emb=None, **kw):
        return (mix.unsqueeze(1) - 0.05) * self.w

    def _mag_sim(self, est, ref):
        return F.cosine_similarity(est, ref, dim=-1) * 10.0

    def _mdcl_loss_hard(self, est_target, near_audio, beta=1.0):
        sims = torch.stack(
            [self._mag_sim(est_target, n.to(est_target.device)) for n in near_audio], 0
        )
        w = torch.softmax(beta * sims, dim=0)
        return (w * F.relu(sims - self.mdcl_margin)).sum(0).mean()


class _DummyContrastive(nn.Module):
    """Mimics `FalconSpeakerContrastive`: 2-tuple in training, tensor in eval."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, mix, mouth_emb=None):
        B, T = mix.shape
        est = (mix.unsqueeze(1) - 0.05) * self.w           # [B,1,T]
        emb = torch.ones(B, 1, 4, _n_t(T), 4, device=mix.device) * self.w
        if self.training:
            return est, emb
        return est

    def pool_spk_embed(self, feat):  # [B,N,T',F] -> [B,F]
        return F.normalize(feat.mean(dim=(1, 2)), dim=-1)

    def encode_stft(self, wav):
        B, T = wav.shape
        return torch.ones(B, 4, _n_t(T), 4, device=wav.device) * 0.7


class _TrainDS(Dataset):
    T = 512

    def __len__(self):
        return 8

    def __getitem__(self, i):
        near = [torch.randn(self.T) for _ in range(i % 3)]
        return torch.randn(self.T), torch.randn(self.T), torch.randn(8, 88, 88), f"f{i}.wav", near


class _ValDS(Dataset):
    T = 512

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.randn(self.T), torch.randn(self.T), torch.randn(8, 88, 88), f"v{i}.wav", i % 4


def _collate(batch):
    mix, tgt, mouth, fn, near = zip(*batch)
    return torch.stack(mix), torch.stack(tgt), torch.stack(mouth), list(fn), list(near)


def _build(cls, audio, logger, **extra):
    opt = torch.optim.AdamW(audio.parameters(), lr=1e-3)
    loss_func = {
        "train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx", threshold_byloss=True),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False),
    }
    system = cls(
        audio_model=audio,
        video_model=_DummyVideo(),
        loss_func=loss_func,
        optimizer=opt,
        train_loader=DataLoader(_TrainDS(), batch_size=2, collate_fn=_collate, num_workers=0),
        val_loader=DataLoader(_ValDS(), batch_size=1, num_workers=0),
        test_loader=DataLoader(_ValDS(), batch_size=1, num_workers=0),
        scheduler=None,
        config={"exp": {"exp_name": "t2"}, "training": {"gpus": [0]}},
        **extra,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        num_sanity_val_steps=0,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    return system, trainer


def _assert_epoch_metrics_logged(trainer):
    keys = set(trainer.logged_metrics.keys())
    assert "learning_rate" in keys, keys
    assert "val_pit_sisnr" in keys, keys
    assert any(k.startswith("val_pit_sisnr_") and k.endswith("near") for k in keys), keys
    assert "test_pit_sisnr" in keys, keys
    # the ordinary self.log keys must still be there too
    assert any(k.startswith("val_loss") for k in keys), keys


# ──────────────────────────────────────────────────────────────────────────────
# Tests: CSVLogger fit for each module + no-logger smoke
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cls,audio,extra",
    [
        (AudioVisualLightningModule, _DummyPlain(), {}),
        (AudioVisualLightningModuleNASSv2, _DummyContrastive(),
         dict(contrastive_weight=1.0, margin=0.5, beta_init=0.0, beta_final=1.0,
              beta_anneal_epochs=10)),
        (AudioVisualLightningModuleWithMDCL, _DummyPlain(),
         dict(beta_init=0.0, beta_final=1.5, beta_anneal_epochs=10)),
    ],
    ids=["base", "nass_v2", "mdcl"],
)
def test_fit_with_csvlogger(tmp_path, cls, audio, extra):
    logger = CSVLogger(save_dir=str(tmp_path), name="t2")
    system, trainer = _build(cls, audio, logger, **extra)
    trainer.fit(system)  # must not raise AttributeError

    _assert_epoch_metrics_logged(trainer)

    csv_files = glob.glob(os.path.join(str(tmp_path), "**", "metrics.csv"), recursive=True)
    assert csv_files, "CSVLogger produced no metrics.csv"
    header = open(csv_files[0]).readline()
    for expected in ("learning_rate", "val_pit_sisnr", "test_pit_sisnr"):
        assert expected in header, (expected, header)


def test_fit_without_logger_does_not_crash(tmp_path):
    # PL2: logger=False disables logging (logger=None would create a default logger)
    system, trainer = _build(AudioVisualLightningModuleNASSv2, _DummyContrastive(), False,
                             contrastive_weight=1.0, margin=0.5)
    assert trainer.logger is None
    trainer.fit(system)
    _assert_epoch_metrics_logged(trainer)


def test_no_comet_only_api_left_in_system_modules():
    import inspect

    from look2hear.system import av_litmodule, av_litmodule_mdcl, av_litmodule_nass_v2

    for mod in (av_litmodule, av_litmodule_mdcl, av_litmodule_nass_v2):
        src = inspect.getsource(mod)
        # only the explanatory comment may mention it; no executable call may remain
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "logger.experiment" not in stripped, f"{mod.__name__}:{lineno}: {stripped}"
