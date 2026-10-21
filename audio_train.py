import os

import torch
torch.cuda.empty_cache()

import argparse
import copy
import json
import yaml
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import look2hear.models
import look2hear.losses
import look2hear.system
import look2hear.videomodels
import look2hear.utils
from look2hear.system import make_optimizer
from look2hear.utils import print_only

from look2hear.datas.avspeech_dataset_near import AVSpeechDataModuleWithNear
from look2hear.system.av_litmodule_nass_v2 import AudioVisualLightningModuleNASSv2

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/Falcon.yml",
    help=(
        "Path to YAML config file (default: configs/Falcon.yml). "
        "Legacy configs/NASSv2_lrs3_complex_nonmoving.yml is not shipped with the repo."
    ),
)
parser.add_argument(
    "--ckpt_path",
    default="",
    help="Optional checkpoint path to resume from (leave empty to train from scratch)",
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (PL 2.x compatible, testable without launching a real training run)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_gpus_spec(value):
    """Split `training.gpus` into a semantic kind + payload.

    Semantic contract (see ``docs/verification/phase1/T1-report.md`` §2/§5):
      * ``None`` / empty string         -> ("none", None)
      * list / tuple / range / "0,1"    -> ("indices", [gpu_id, ...])  (GPU **indices**)
      * scalar int ``n > 0``            -> ("count", n)                (number of GPUs)
      * scalar int ``0``                -> ("cpu", None)               (legacy PL1 "no GPU")
      * scalar int ``-1``               -> ("all", None)               (legacy PL1 "all GPUs")
      * other negative ints             -> ValueError (only ``-1`` means "all GPUs")

    The old implementation flattened everything into a list and then guessed
    "one element == number of GPUs", so ``gpus: [1]`` silently used GPU 0 and
    ``gpus: [7]`` asked for 2 GPUs.  A list is now *always* interpreted as
    explicit indices and a scalar *always* as a count.
    """
    if value is None or value is False:
        return "none", None
    if isinstance(value, bool):  # bool is an int subclass -> reject early
        raise ValueError(f"training.gpus must be int/list/str/None, got bool {value!r}")
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip() != ""]
        if not parts:
            return "none", None
        try:
            return "indices", [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(
                f"training.gpus={value!r} is not a comma-separated integer list"
            ) from exc
    if isinstance(value, (list, tuple, range)):
        return "indices", [int(v) for v in value]
    if isinstance(value, int):
        if value == 0:
            return "cpu", None
        if value == -1:
            return "all", None
        if value < -1:
            raise ValueError(
                f"training.gpus={value} is not supported: use -1 for all visible GPUs, "
                "0 for CPU, a positive GPU count, or a list of GPU indices"
            )
        return "count", value
    raise ValueError(
        f"training.gpus must be int/list/str/None, got {type(value)!r}"
    )


def resolve_accelerator_and_devices(training_cfg):
    """Resolve PL 2.x (`accelerator`, `devices`, `num_devices`, `strategy`) from config.

    Rules (see :func:`_normalize_gpus_spec`):
      * no CUDA                        -> ("cpu", 1, 1, "auto")
      * ``gpus`` missing               -> single GPU
      * ``gpus: <int N>``              -> N GPUs (clamped to the visible count)
      * ``gpus: [i, j, ...]``          -> explicit GPU indices (list passed to PL)
      * >1 device                      -> DDPStrategy(find_unused_parameters=True)
      * <=1 device                     -> "auto"

    Invalid configurations raise ``ValueError`` here (instead of a delayed
    ``MisconfigurationException`` from ``pl.Trainer``): empty index list,
    duplicated indices, or indices that do not exist on this machine.
    ``configs/Falcon.yml`` ships a 10-GPU index list; on machines with fewer
    visible GPUs it is clamped to the visible subset (with a warning).
    """
    training_cfg = training_cfg or {}
    kind, payload = _normalize_gpus_spec(training_cfg.get("gpus"))

    if not torch.cuda.is_available():
        return "cpu", 1, 1, "auto"
    visible = torch.cuda.device_count()
    if visible <= 0:  # defensive; is_available() should imply >=1
        return "cpu", 1, 1, "auto"

    if kind == "cpu":  # legacy PL1 gpus: 0 -> CPU
        print_only("[audio_train] WARNING: legacy training.gpus=0 -> running on CPU")
        return "cpu", 1, 1, "auto"
    if kind == "all":  # legacy PL1 gpus: -1 -> all visible GPUs
        devices = list(range(visible))
        num_devices = visible
    elif kind == "none":
        devices = [0] if visible > 1 else 1
        num_devices = 1
    elif kind == "count":
        num_devices = min(payload, visible)
        if num_devices != payload:
            print_only(
                f"[audio_train] WARNING: config requests {payload} GPU(s) but only "
                f"{visible} visible -> using {num_devices}"
            )
        devices = num_devices
    else:  # kind == "indices"
        indices = payload
        if len(indices) == 0:
            raise ValueError(
                "training.gpus is an empty list; provide GPU indices (e.g. [0, 1]), "
                "a GPU count (e.g. 2), or omit the key to use a single GPU"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"training.gpus={indices} contains duplicate GPU indices")
        valid = [idx for idx in indices if 0 <= idx < visible]
        if not valid:
            raise ValueError(
                f"training.gpus={indices} does not match any visible GPU "
                f"(visible: {list(range(visible))}); on this machine use e.g. "
                f"{list(range(min(len(indices), visible)))} or a GPU count"
            )
        if len(valid) != len(indices):
            print_only(
                f"[audio_train] WARNING: config requests GPU devices {indices} but only "
                f"{list(range(visible))} visible -> using {valid}"
            )
        devices = valid
        num_devices = len(valid)

    strategy = DDPStrategy(find_unused_parameters=True) if num_devices > 1 else "auto"
    return "gpu", devices, num_devices, strategy


def build_logger(config, exp_dir):
    """Build a Lightning logger from `config["training"]["logger"]`.

    Supported values (case-insensitive): "csv" (default), "tensorboard"/"tb",
    "comet"/"comet_ml", and "none"/None/False (no logger).  Missing optional
    dependencies raise a clear RuntimeError instead of a bare ImportError.
    """
    training_cfg = config.get("training", {}) or {}
    logger_type = training_cfg.get("logger", "csv")
    exp_name = config.get("exp", {}).get("exp_name", "exp")

    if logger_type is None or logger_type is False:
        return None
    if not isinstance(logger_type, str):
        raise ValueError(
            f"training.logger must be a string (csv/tensorboard/comet), got {type(logger_type)!r}"
        )
    kind = logger_type.strip().lower()

    if kind in ("", "none", "false", "null"):
        return None
    if kind in ("csv", "csvlogger"):
        # exp_dir already ends with <exp_name>; log to <exp_dir>/version_N instead of
        # <exp_dir>/<exp_name>/version_N (avoid duplicated experiment name in the path).
        return CSVLogger(save_dir=os.path.dirname(os.path.abspath(str(exp_dir))), name=exp_name)
    if kind in ("tensorboard", "tb", "tensorboardlogger"):
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "training.logger='tensorboard' but TensorBoardLogger is unavailable; "
                "install it with `pip install tensorboard`."
            ) from exc
        try:
            return TensorBoardLogger(save_dir=exp_dir, name=exp_name)
        except (ImportError, ModuleNotFoundError) as exc:  # tensorboard not installed
            raise RuntimeError(
                "training.logger='tensorboard' but the `tensorboard` package is missing; "
                "install it with `pip install tensorboard`."
            ) from exc
    if kind in ("comet", "comet_ml", "cometlogger"):
        try:
            from pytorch_lightning.loggers import CometLogger
            import comet_ml  # noqa: F401  (required by CometLogger at construction time)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "training.logger='comet' requires the optional `comet-ml` package; "
                "install it with `pip install comet-ml` and set COMET_API_KEY, "
                "or use training.logger: csv / tensorboard."
            ) from exc
        comet_kwargs = dict(
            save_dir=exp_dir,
            name=exp_name,
            project=training_cfg.get("comet_project", "falcon"),
        )
        api_key = os.environ.get("COMET_API_KEY")
        if api_key:
            comet_kwargs["api_key"] = api_key
        return CometLogger(**comet_kwargs)

    raise ValueError(
        f"Unsupported training.logger={logger_type!r}; expected one of "
        "{csv, tensorboard, comet, none}."
    )


def main(config, ckpt_path=""):
    # ── Experiment dir + safe config snapshot (BEFORE any model construction) ──
    # NOTE: `Separator.__init__` (look2hear/models/Falcon.py) pops `repeats`/`shared`
    # from the config dict, and `Falcon.__init__` injects `causal`.  Dumping a
    # deepcopy *before* models are built keeps `conf.yml` reusable for evaluation.
    exp_name = config.get("exp", {}).get("exp_name", "exp")
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config.setdefault("main_args", {})["exp_dir"] = exp_dir
    conf_path = os.path.join(exp_dir, "conf.yml")
    safe_config = copy.deepcopy(config)
    with open(conf_path, "w") as f:
        yaml.safe_dump(safe_config, f)
    print_only(f"[audio_train] config snapshot (deepcopy) -> {conf_path}")

    use_near = config["datamodule"]["data_config"].get("use_near", False)
    print_only(f"[audio_train_nass_v2] use_near = {use_near}")

    # ── DataModule ────────────────────────────────────────────────────────────
    print_only(f"Instantiating datamodule: AVSpeechDataModuleWithNear (use_near={use_near})")
    datamodule = AVSpeechDataModuleWithNear(**config["datamodule"]["data_config"])
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader

    # ── Audio model ───────────────────────────────────────────────────────────
    print_only(f"Instantiating AudioNet <{config['audionet']['audionet_name']}>")
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    # ── Video model ───────────────────────────────────────────────────────────
    video_model = getattr(look2hear.videomodels, config["videonet"]["videonet_name"])(
        **config["videonet"]["videonet_config"],
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    print_only(f"Instantiating Optimizer <{config['optimizer']['optim_name']}>")
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(f"Instantiating Scheduler <{config['scheduler']['sche_name']}>")
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
            optimizer=optimizer, **config["scheduler"]["sche_config"]
        )

    # ── Loss functions ────────────────────────────────────────────────────────
    print_only(
        f"Instantiating Loss  train=<{config['loss']['train']['sdr_type']}>  "
        f"val=<{config['loss']['val']['sdr_type']}>"
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    # ── Speaker contrastive params ────────────────────────────────────────────
    spk_cfg = config.get("spk_contrast", {})
    contrastive_weight = spk_cfg.get("loss_weight", 1.0)
    margin             = spk_cfg.get("margin", 0.0)
    alpha_pos          = spk_cfg.get("alpha_pos", 1.0)
    alpha_neg          = spk_cfg.get("alpha_neg", 1.0)
    beta_init          = spk_cfg.get("beta_init", 0.0)
    beta_final         = spk_cfg.get("beta_final", 1.5)
    beta_anneal_epochs = spk_cfg.get("beta_anneal_epochs", 100)

    print_only(
        f"[NASS v2] weight={contrastive_weight} margin={margin} "
        f"alpha_pos={alpha_pos} alpha_neg={alpha_neg} "
        f"beta={beta_init}→{beta_final} over {beta_anneal_epochs} epochs"
    )

    # ── Lightning system ──────────────────────────────────────────────────────
    system = AudioVisualLightningModuleNASSv2(
        audio_model=model,
        video_model=video_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
        contrastive_weight=contrastive_weight,
        margin=margin,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        beta_init=beta_init,
        beta_final=beta_final,
        beta_anneal_epochs=beta_anneal_epochs,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint = ModelCheckpoint(
        exp_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks = [checkpoint]
    if config["training"].get("early_stop"):
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))

    # ── Logger (F1: build from config instead of the undefined `comet_logger`) ─
    logger = build_logger(config, exp_dir)
    print_only(f"[audio_train] logger = {type(logger).__name__ if logger is not None else None}")

    # ── Trainer (F6: PL 2.x accelerator/devices/strategy API) ─────────────────
    accelerator, devices, num_devices, strategy = resolve_accelerator_and_devices(
        config.get("training", {})
    )
    print_only(
        f"[audio_train] accelerator={accelerator} devices={devices} "
        f"num_devices={num_devices} strategy={strategy if isinstance(strategy, str) else type(strategy).__name__}"
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        # PL2 treats logger=None as 'create a default logger'; use False to disable.
        logger=logger if logger is not None else False,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
    )

    # ── Optional resume from checkpoint (kept CLI behavior) ───────────────────
    if ckpt_path and os.path.exists(ckpt_path):
        print_only(f"Resuming from checkpoint: {ckpt_path}")
        trainer.fit(system, ckpt_path=ckpt_path)
    else:
        if ckpt_path:
            print_only(f"[audio_train] WARNING: ckpt_path not found ({ckpt_path}); training from scratch")
        trainer.fit(system)

    print_only("Finished Training")

    # ── Post-training artifacts ───────────────────────────────────────────────
    best_k = {
        k: (v.item() if hasattr(v, "item") else float(v))
        for k, v in checkpoint.best_k_models.items()
    }
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    if checkpoint.best_model_path and os.path.exists(checkpoint.best_model_path):
        state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
        system.load_state_dict(state_dict=state_dict["state_dict"])
        system.cpu()
        to_save = system.audio_model.serialize()
        torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
        print_only(f"[audio_train] saved best_model.pth -> {exp_dir}")
    else:
        print_only("[audio_train] WARNING: no best checkpoint found; skip best_model.pth export")


if __name__ == "__main__":
    from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic, ckpt_path=args.ckpt_path)
