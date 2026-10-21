"""Toy 2-GPU DDP fit used by `review_training_stack_checks.py` §G.

Validates that the PL 2.x Trainer arguments produced by T1's
`resolve_accelerator_and_devices` (accelerator="gpu", devices=[0,1],
DDPStrategy(find_unused_parameters=True), sync_batchnorm=True) actually launch
a multi-process fit with a CSVLogger in this environment. Toy model only (~1 s).

Run: /root/miniforge3/envs/falcon/bin/python docs/verification/phase1/_ddp_smoke_toy.py
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset


class Toy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 2)
        self.bn = torch.nn.BatchNorm1d(4)

    def forward(self, x):
        return self.l(self.bn(x))

    def training_step(self, batch, idx):
        x, y = batch
        loss = torch.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        loss = torch.nn.functional.cross_entropy(self(x), y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-3)


def main() -> None:
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[0, 1],
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=True,
        limit_train_batches=0.5,
        gradient_clip_val=5.0,
        num_sanity_val_steps=0,
        logger=CSVLogger(save_dir="/tmp/phase1-review-ddp", name="toy"),
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir="/tmp/phase1-review-ddp",
    )
    trainer.fit(Toy(), DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4))
    print("DDP TOY FIT OK")


if __name__ == "__main__":
    main()
