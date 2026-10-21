import sys, types, tempfile, traceback, os
# --- stub missing dep: torch_optimizer imports pytorch_ranger ---
pr = types.ModuleType("pytorch_ranger")
class _R:
    def __init__(self, *a, **k): pass
pr.Ranger = pr.RangerQH = pr.RangerVA = _R
sys.modules["pytorch_ranger"] = pr
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from look2hear.system import AudioVisualLightningModuleNASSv2, AudioVisualLightningModule
from look2hear.system.av_litmodule_mdcl import AudioVisualLightningModuleWithMDCL
from look2hear.losses import PITLossWrapper, pairwise_neg_snr, pairwise_neg_sisdr

TMP = tempfile.mkdtemp(prefix="litprobe_")

def _n_t(T):  # mimic STFT frame count
    return max((T - 256)//128 + 1, 1)

class DummyAudioContrastive(nn.Module):
    """Mimics FalconSpeakerContrastive: 2-tuple output in training mode."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))
    def forward(self, mix, mouth_emb=None):
        B, T = mix.shape
        est = (mix.unsqueeze(1) - 0.05) * self.w          # [B,1,T]
        emb = torch.ones(B, 1, 4, _n_t(T), 4, device=mix.device) * self.w  # [B,1,N,T',F]
        if self.training:          # mimic FalconSpeakerContrastive.forward branch
            return est, emb
        return est
    def pool_spk_embed(self, feat):  # [B,N,T',F] -> [B,F]
        return F.normalize(feat.mean(dim=(1, 2)), dim=-1)
    def encode_stft(self, wav):
        B, T = wav.shape
        return torch.ones(B, 4, _n_t(T), 4, device=wav.device) * 0.7
    def serialize(self):
        return {"model_name": "dummy"}

class DummyAudioPlain(DummyAudioContrastive):
    """Mimics plain Falcon: single-tensor output, MDCL helpers."""
    mdcl_alpha = 0.15
    mdcl_margin = 0.0
    def forward(self, mix, mouth_emb=None, **kw):
        B, T = mix.shape
        return (mix.unsqueeze(1) - 0.05) * self.w
    def _mag_sim(self, est, ref):
        return F.cosine_similarity(est, ref, dim=-1) * 10.0
    def _mdcl_loss_hard(self, est_target, near_audio, beta=1.0):
        sims = torch.stack([self._mag_sim(est_target, n.to(est_target.device)) for n in near_audio], 0)
        w = torch.softmax(beta * sims, dim=0)
        return (w * F.relu(sims - self.mdcl_margin)).sum(0).mean()

class DummyVideo(nn.Module):
    def forward(self, x):
        return x.new_zeros(x.shape[0], 512, 8)

LOOP = {"n": 0}

class TrainDS(Dataset):
    def __init__(self, T=512): self.T = T
    def __len__(self): return 16
    def __getitem__(self, i):
        L = self.T
        near = [torch.randn(L) for _ in range(i % 3)]  # 0,1,2 near speakers
        return torch.randn(L), torch.randn(L), torch.randn(8, 96, 96), f"f{i}.wav", near

def collate(batch):
    mix, tgt, mouth, fn, near = zip(*batch)
    return torch.stack(mix), torch.stack(tgt), torch.stack(mouth), list(fn), list(near)

class ValDS(Dataset):
    def __init__(self, T=512): self.T = T
    def __len__(self): return 6
    def __getitem__(self, i):
        L = self.T
        return torch.randn(L), torch.randn(L), torch.randn(8, 96, 96), f"v{i}.wav", (i % 3)

results = {}

def build_system(cls, audio, **extra):
    tr = DataLoader(TrainDS(), batch_size=2, shuffle=False, collate_fn=collate)
    va = DataLoader(ValDS(), batch_size=1)
    te = DataLoader(ValDS(), batch_size=1)
    opt = torch.optim.AdamW(audio.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    lf = {"train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx", threshold_byloss=True),
          "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False)}
    sysm = cls(audio_model=audio, video_model=DummyVideo(), loss_func=lf, optimizer=opt,
               train_loader=tr, val_loader=va, test_loader=te, scheduler=sched,
               config={"exp": {"exp_name": "t"}}, **extra)
    return sysm

# ============ 1) beta annealing formula ============
try:
    audio = DummyAudioContrastive()
    sysm = build_system(AudioVisualLightningModuleNASSv2, audio, contrastive_weight=1.0,
                        margin=0.5, alpha_pos=1.0, alpha_neg=1.0,
                        beta_init=0.0, beta_final=1.0, beta_anneal_epochs=200)
    vals = {}
    for e in (0, 1, 50, 100, 150, 200, 250):
        sysm._trainer = types.SimpleNamespace(current_epoch=e)
        vals[e] = round(sysm._get_beta(), 4)
    results["beta(epoch)"] = vals
    # anneal_epochs <= 0 branch
    sysm2 = build_system(AudioVisualLightningModuleNASSv2, DummyAudioContrastive(),
                         beta_init=0.0, beta_final=1.5, beta_anneal_epochs=0)
    sysm2._trainer = types.SimpleNamespace(current_epoch=0)
    results["beta(anneal<=0)"] = sysm2._get_beta()
except Exception:
    results["beta annealing"] = "FAILED\n" + traceback.format_exc()[-800:]

# ============ 2) configure_optimizers ============
try:
    opt_out = sysm.configure_optimizers()
    results["configure_optimizers"] = (type(opt_out).__name__, len(opt_out[0]),
                                       opt_out[1][0]["monitor"], type(opt_out[1][0]["scheduler"]).__name__)
except Exception:
    results["configure_optimizers"] = "FAILED\n" + traceback.format_exc()[-600:]

# ============ 3) NASSv2 full fit with Comet-like logger ============
class FakeCometExp:
    def __init__(self): self.logged = []
    def log(self, d, *a, **k): self.logged.append(d)
class FakeCometLogger(pl.loggers.Logger):
    def __init__(self):
        super().__init__(); self._exp = FakeCometExp()
    @property
    def name(self): return "fake"
    @property
    def version(self): return "0"
    @property
    def experiment(self): return self._exp
    def log_metrics(self, metrics, step=None, **kwargs): pass
    def log_hyperparams(self, params, *a, **k): pass
    def save(self): pass
    def finalize(self, s): pass

try:
    audio = DummyAudioContrastive()
    sysm = build_system(AudioVisualLightningModuleNASSv2, audio, contrastive_weight=1.0,
                        margin=0.5, beta_init=0.0, beta_final=1.0, beta_anneal_epochs=200)
    ck = ModelCheckpoint(TMP, filename="{epoch}", monitor="val_loss/dataloader_idx_0",
                         mode="min", save_top_k=5, verbose=True, save_last=True)
    tr = pl.Trainer(max_epochs=11, callbacks=[ck], default_root_dir=TMP, devices=1,
                    accelerator="cpu", limit_train_batches=1, limit_val_batches=2,
                    gradient_clip_val=3.0, logger=FakeCometLogger(), sync_batchnorm=True,
                    num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False)
    tr.fit(sysm)
    lm = tr.logged_metrics
    results["NASSv2 fit(11ep)"] = "OK"
    results["NASSv2 logged keys"] = sorted(lm.keys())
    results["NASSv2 comet logged keys"] = sorted({k for d in tr.logger.experiment.logged for k in d})
    results["NASSv2 ckpt best_k n"] = len(ck.best_k_models)
    results["NASSv2 best_k value type"] = type(next(iter(ck.best_k_models.values()))).__name__
    results["NASSv2 sample loss values"] = {k: round(float(lm[k]), 4) for k in
        ["train_loss_epoch", "train_pit_loss_epoch", "train_spk_contrast_loss_epoch"] if k in lm}
except Exception:
    results["NASSv2 fit(11ep)"] = "FAILED\n" + traceback.format_exc()[-1500:]

# ============ 4) same fit with CSVLogger (no comet API) ============
try:
    audio = DummyAudioContrastive()
    sysm = build_system(AudioVisualLightningModuleNASSv2, audio)
    tr = pl.Trainer(max_epochs=1, default_root_dir=TMP, devices=1, accelerator="cpu",
                    limit_train_batches=1, limit_val_batches=1, logger=CSVLogger(TMP + "/csv"),
                    num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False)
    tr.fit(sysm)
    results["NASSv2 fit(CSVLogger)"] = "OK (unexpected, comet call passed?)"
except Exception as e:
    results["NASSv2 fit(CSVLogger)"] = f"FAILED as predicted: {type(e).__name__}: {str(e)[:300]}\n" + traceback.format_exc()[-900:]

# ============ 5) MDCL full fit with plain model ============
try:
    audio = DummyAudioPlain()
    sysm = build_system(AudioVisualLightningModuleWithMDCL, audio,
                        beta_init=0.0, beta_final=1.5, beta_anneal_epochs=200)
    tr = pl.Trainer(max_epochs=3, default_root_dir=TMP + "/mdcl", devices=1, accelerator="cpu",
                    limit_train_batches=1, limit_val_batches=1, logger=FakeCometLogger(),
                    num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False)
    tr.fit(sysm)
    lm = tr.logged_metrics
    results["MDCL fit(3ep)"] = "OK"
    results["MDCL logged keys"] = sorted([k for k in lm.keys() if "mdcl" in k or "pit" in k or "beta" in k])
    results["MDCL values"] = {k: round(float(lm[k]), 5) for k in
        ["train_pit_loss_epoch", "train_mdcl_loss_epoch", "train_beta_epoch"] if k in lm}
except Exception:
    results["MDCL fit(3ep)"] = "FAILED\n" + traceback.format_exc()[-1200:]

# ============ 6) base litmodule + contrastive model (tuple to PIT) ============
try:
    lf = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
    est_tuple = (torch.randn(2, 1, 512), torch.randn(2, 1, 4, 2, 4))
    out = lf(est_tuple, torch.randn(2, 1, 512))
    results["PITW(tuple_ests)"] = f"OK unexpectedly: {out}"
except Exception as e:
    results["PITW(tuple_ests)"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

for k, v in results.items():
    print(f"### {k}: {v}")
