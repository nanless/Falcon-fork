"""Save and load experimental weights without changing released Falcon loaders."""

from pathlib import Path

import torch

from look2hear.models.Falcon_Contrastive import FalconSpeakerContrastive
from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from look2hear.models.FalconRollingConditioned import FalconRollingConditioned
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel
from tools.streaming.evaluate_realssa import load_baseline
from tools.streaming.rolling_falcon_live import RollingFalconLiveInference

ROLLING_FORMAT_VERSION = 1
ARCHITECTURES = {
    "FalconSpeakerContrastive": FalconSpeakerContrastive,
    "FalconRollingConditioned": FalconRollingConditioned,
}


def make_frequency_conditioned(base, *, seed):
    """Copy the complete released model and add a zero-increment visual path."""
    if type(base) is not FalconSpeakerContrastive:
        raise TypeError("expected the published Falcon Small architecture")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = FalconRollingConditioned(**base.get_model_args())
    parameter = next(base.parameters())
    model.to(device=parameter.device, dtype=parameter.dtype)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    expected = {
        f"frequency_condition.{name}" for name in model.frequency_condition.state_dict()
    }
    if set(missing) != expected or unexpected:
        raise ValueError("frequency-conditioned model did not inherit the complete base")
    base_parameters = dict(base.named_parameters())
    for name, value in model.named_parameters():
        if name in base_parameters:
            value.requires_grad_(base_parameters[name].requires_grad)
    return model.train(base.training)


def _checked_history(history_blocks):
    if type(history_blocks) is not int or history_blocks < 0:
        raise ValueError("history_blocks must be a nonnegative integer")
    return history_blocks


def pack_rolling_checkpoint(audio_model, visual_model, *, history_blocks=5, **metadata):
    architecture = type(audio_model).__name__
    if ARCHITECTURES.get(architecture) is not type(audio_model):
        raise ValueError("unsupported rolling model architecture")
    backbone = visual_model.backbone if isinstance(visual_model, FalconLookaheadVisual) else visual_model
    if type(backbone) is not FRCNNVideoModel or backbone.backbone_type != "resnet":
        raise ValueError("expected the published Falcon Small FRCNN")
    payload = {
        "rolling_format_version": ROLLING_FORMAT_VERSION,
        "architecture": architecture,
        "audio_config": audio_model.get_model_args(),
        "history_blocks": _checked_history(history_blocks),
        "state_dict": {
            f"{prefix}.{name}": value.detach().cpu().clone()
            for prefix, module in (("audio_model", audio_model), ("video_model", backbone))
            for name, value in module.state_dict().items()
        },
    }
    if payload.keys() & metadata.keys():
        raise ValueError("metadata cannot override rolling checkpoint fields")
    return payload | metadata


def load_rolling_checkpoint(path, device):
    """Return the waveform model, raw frozen visual backbone, and saved metadata."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("invalid rolling checkpoint")
    if "rolling_format_version" not in payload and payload.get("architecture") == "FalconSpeakerContrastive":
        audio_model, backbone, _ = load_baseline(path, device)
        audio_model.spk_pool_proj.requires_grad_(False)
        backbone.requires_grad_(False)
        return audio_model, backbone, payload | {"history_blocks": 5}
    if payload.get("rolling_format_version") != ROLLING_FORMAT_VERSION:
        raise ValueError("unsupported rolling checkpoint format")
    cls = ARCHITECTURES.get(payload.get("architecture"))
    config = payload.get("audio_config")
    weights = payload.get("state_dict")
    if cls is None or not isinstance(config, dict) or not isinstance(weights, dict):
        raise ValueError("incomplete rolling checkpoint")
    _checked_history(payload.get("history_blocks"))
    if config.get("n_src") != 1 or config.get("sample_rate") != 16000:
        raise ValueError("rolling checkpoints must represent one 16 kHz target")
    audio_model = cls(**config)
    backbone = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu")
    if any(not name.startswith(("audio_model.", "video_model.")) for name in weights):
        raise ValueError("unexpected rolling checkpoint state prefix")
    for prefix, module in (("audio_model.", audio_model), ("video_model.", backbone)):
        module.load_state_dict(
            {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)},
            strict=True,
        )
    audio_model.spk_pool_proj.requires_grad_(False)
    backbone.requires_grad_(False)
    return audio_model.to(device).eval(), backbone.to(device).eval(), payload


class RollingExperimentLiveInference(RollingFalconLiveInference):
    """Use an experimental checkpoint with the existing 40 ms tick API."""

    def __init__(self, checkpoint, *, device="cpu", threads=None):
        if threads is not None:
            if type(threads) is not int or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.audio_model, backbone, payload = load_rolling_checkpoint(checkpoint, self.device)
        self.visual_model = FalconLookaheadVisual(backbone)
        self.history_blocks = payload["history_blocks"]
        self.reset()
