import os, tempfile, traceback
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import CSVLogger

TMP = tempfile.mkdtemp(prefix="plprobe_")

class Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.scheduler = [ReduceLROnPlateau(self.optimizer, patience=15, factor=0.5)]
        self.val_hook_log = None
        self.comet_api = None

    def training_step(self, batch, idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.lin(x), y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, idx, dataloader_idx=0):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.lin(x), y)
        if dataloader_idx == 0:
            self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        else:
            self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        try:
            self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, prog_bar=True, sync_dist=True)
            self.val_hook_log = "self.log in on_validation_epoch_end: OK"
        except Exception as e:
            self.val_hook_log = f"self.log in on_validation_epoch_end FAILED: {type(e).__name__}: {e}"
        try:
            self.logger.experiment.log({"val_pit_sisnr": 1.0, "epoch": self.current_epoch})
            self.comet_api = "logger.experiment.log(dict) OK"
        except Exception as e:
            self.comet_api = f"logger.experiment.log(dict) FAILED: {type(e).__name__}: {e}"

    def configure_optimizers(self):
        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.randn(16, 4), torch.randn(16, 1)), batch_size=4)

    def val_dataloader(self):
        d = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 1)), batch_size=2)
        return [d, d]

results = {}

# --- Test 1: ModelCheckpoint / EarlyStopping kwargs ---
try:
    ckpt = ModelCheckpoint(TMP, filename="{epoch}", monitor="val_loss/dataloader_idx_0",
                           mode="min", save_top_k=5, verbose=True, save_last=True)
    results["ModelCheckpoint kwargs"] = "OK"
except Exception as e:
    results["ModelCheckpoint kwargs"] = f"FAILED: {type(e).__name__}: {e}"
try:
    es = EarlyStopping(monitor="val_loss/dataloader_idx_0", mode="min", patience=30, verbose=True)
    results["EarlyStopping kwargs"] = "OK"
except Exception as e:
    results["EarlyStopping kwargs"] = f"FAILED: {type(e).__name__}: {e}"

# --- Test 2: fit with CSVLogger (non-Comet) and 2 val dataloaders ---
try:
    model = Tiny()
    trainer = pl.Trainer(max_epochs=2, callbacks=[ckpt, es], default_root_dir=TMP,
                         devices=1, accelerator="cpu", limit_train_batches=2.0,
                         gradient_clip_val=5.0, logger=CSVLogger(TMP),
                         sync_batchnorm=True, num_sanity_val_steps=0, enable_progress_bar=False)
    trainer.fit(model)
    results["fit(cpu,CSVLogger,2 val loaders)"] = "OK"
    results["logged val keys"] = sorted([k for k in trainer.logged_metrics.keys()])
    results["val_hook_log"] = model.val_hook_log
    results["comet_api"] = model.comet_api
    results["best_model_path"] = ckpt.best_model_path
    results["best_k_models"] = {os.path.basename(k): (type(v).__name__, v) for k, v in ckpt.best_k_models.items()}
    try:
        results["v.item()"] = {os.path.basename(k): v.item() for k, v in ckpt.best_k_models.items()}
    except Exception as e:
        results["v.item()"] = f"FAILED: {type(e).__name__}: {e}"
    # --- Test 3: torch.load default (weights_only) on Lightning ckpt ---
    try:
        raw = torch.load(ckpt.best_model_path)
        results["torch.load(default)"] = f"OK, keys={sorted(raw.keys())[:8]}"
    except Exception as e:
        results["torch.load(default)"] = f"FAILED: {type(e).__name__}: {e}"
except Exception as e:
    results["fit(cpu,CSVLogger,2 val loaders)"] = f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"

# --- Test 4: Trainer construction mimicking audio_train.py (CPU env) ---
try:
    t = pl.Trainer(max_epochs=1000, devices=None, accelerator=None,
                   strategy=DDPStrategy(find_unused_parameters=True),
                   limit_train_batches=1.0, gradient_clip_val=5.0, logger=False,
                   sync_batchnorm=True, num_sanity_val_steps=0)
    results["Trainer(devices=None,accelerator=None,strategy=DDP)"] = f"constructed OK, strategy={type(t.strategy).__name__}, accelerator={type(t.accelerator).__name__}"
except Exception as e:
    results["Trainer(devices=None,accelerator=None,strategy=DDP)"] = f"FAILED: {type(e).__name__}: {e}"

# --- Test 5: Trainer with 10 devices on a 0-GPU box ---
try:
    t = pl.Trainer(max_epochs=10, devices=[0,1,2,3,4,5,6,7,8,9], accelerator="cuda",
                   strategy=DDPStrategy(find_unused_parameters=True), logger=False)
    results["Trainer(devices=[0..9],cuda)"] = "constructed OK"
except Exception as e:
    results["Trainer(devices=[0..9],cuda)"] = f"FAILED: {type(e).__name__}: {e}"

results["torch.cuda.is_available"] = torch.cuda.is_available()
results["pl.__version__"] = pl.__version__
results["torch.__version__"] = torch.__version__
for k, v in results.items():
    print(f"### {k}: {v}")
