import sys

def main():
    import types
    pr = types.ModuleType("pytorch_ranger")
    class _R:
        def __init__(self, *a, **k): pass
    pr.Ranger = pr.RangerQH = pr.RangerVA = _R
    sys.modules["pytorch_ranger"] = pr
    sys.path.insert(0, "/root/code/github_repos/Falcon-fork")

    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import pytorch_lightning as pl
    from look2hear.system import AudioVisualLightningModuleNASSv2
    from look2hear.losses import PITLossWrapper, pairwise_neg_snr, pairwise_neg_sisdr

    class FakeCometLogger(pl.loggers.Logger):
        @property
        def name(self): return "fake"
        @property
        def version(self): return "0"
        @property
        def experiment(self):
            class E:
                def log(self, d, *a, **k): pass
            return E()
        def log_metrics(self, metrics, step=None, **kwargs): pass
        def log_hyperparams(self, params, *a, **k): pass
        def save(self): pass
        def finalize(self, status): pass

    class DummyAudio(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(0.5))
        def forward(self, mix, mouth_emb=None):
            B, T = mix.shape
            est = (mix.unsqueeze(1) - 0.05) * self.w
            emb = torch.ones(B, 1, 4, 3, 4, device=mix.device) * self.w
            if self.training:
                return est, emb
            return est
        def pool_spk_embed(self, feat):
            return F.normalize(feat.mean(dim=(1, 2)), dim=-1)
        def encode_stft(self, wav):
            B, T = wav.shape
            return torch.ones(B, 4, 3, 4, device=wav.device) * 0.7

    class DummyVideo(nn.Module):
        def forward(self, x):
            return x.new_zeros(x.shape[0], 512, 8)

    # val items: n_near = [0,1,5,1,1,1]; DistributedSampler -> rank0:[0,1], rank1:[3,4]
    # limit_val_batches=2 => rank0 buckets {0,1} (2 collectives), rank1 bucket {1} (1 collective)
    N_NEAR = [0, 1, 5, 1, 1, 1]

    class ValDS(Dataset):
        def __len__(self): return len(N_NEAR)
        def __getitem__(self, i):
            return torch.randn(512), torch.randn(512), torch.randn(8, 96, 96), f"v{i}.wav", N_NEAR[i]

    class TrainDS(Dataset):
        def __len__(self): return 8
        def __getitem__(self, i):
            return torch.randn(512), torch.randn(512), torch.randn(8, 96, 96), f"t{i}.wav", []

    def collate(b):
        m, t, mo, f, n = zip(*b)
        return torch.stack(m), torch.stack(t), torch.stack(mo), list(f), list(n)

    class DebugNASSv2(AudioVisualLightningModuleNASSv2):
        def validation_step(self, batch, batch_nb, dataloader_idx=0):
            if dataloader_idx == 0:
                print(f"[rank{self.global_rank}] val idx={batch_nb} n_near={int(batch[4])}", flush=True)
            return super().validation_step(batch, batch_nb, dataloader_idx)

        def on_validation_epoch_end(self):
            occ = {b: len(v) for b, v in self._val_bucket_outputs.items() if v}
            print(f"[rank{self.global_rank}] bucket occupancy BEFORE gather: {occ}", flush=True)
            return super().on_validation_epoch_end()

    audio = DummyAudio()
    bot = DataLoader(TrainDS(), batch_size=2, collate_fn=collate, num_workers=0)
    val = DataLoader(ValDS(), batch_size=1, num_workers=0)
    test = DataLoader(ValDS(), batch_size=1, num_workers=0)
    opt = torch.optim.AdamW(audio.parameters(), lr=1e-3)
    lf = {"train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx"),
          "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")}
    system = DebugNASSv2(audio_model=audio, video_model=DummyVideo(), loss_func=lf,
                         optimizer=opt, train_loader=bot, val_loader=val, test_loader=test,
                         scheduler=None, config={"exp": {"exp_name": "ddp2"}},
                         contrastive_weight=1.0, beta_init=0.0, beta_final=1.0,
                         beta_anneal_epochs=2)
    trainer = pl.Trainer(max_epochs=1, devices=2, accelerator="cpu", strategy="ddp",
                         limit_train_batches=1, limit_val_batches=2,
                         num_sanity_val_steps=0, enable_progress_bar=False,
                         enable_model_summary=False, logger=FakeCometLogger())
    print("[ddp2-probe] starting fit (asymmetric buckets expected)", flush=True)
    trainer.fit(system)
    print("[ddp2-probe] fit finished OK", flush=True)

if __name__ == "__main__":
    main()
