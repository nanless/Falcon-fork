
import sys

sys.path.append("../../")
import os
import torch.nn as nn
import torch
from .shufflenetv2 import ShuffleNetV2
from .resnet1D import ResNet1D, BasicBlock1D
from .resnet import ResNet, BasicBlock
from torch.nn.modules.batchnorm import _BatchNorm


# -- auxiliary functions


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack(
        [torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0
    )


class FRCNNVideoModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        backbone_type="resnet",
        relu_type="prelu",
        width_mult=1.0,
        pretrain=None,
    ):
        super(FRCNNVideoModel, self).__init__()
        self.backbone_type = backbone_type
        if self.backbone_type == "resnet":
            self.frontend_nout = 64
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        elif self.backbone_type == "shufflenet":
            assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
            shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
            self.trunk = nn.Sequential(
                shufflenet.features, shufflenet.conv_last, shufflenet.globalpool
            )
            self.frontend_nout = 24
            self.backend_out = 1024 if width_mult != 2.0 else 2048
            self.stage_out_channels = shufflenet.stage_out_channels[-1]

        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == "prelu"
            else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.pretrain = pretrain
        if pretrain:
            self.init_from(pretrain)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # output should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        if self.backbone_type == "shufflenet":
            x = x.view(-1, self.stage_out_channels)
        x = x.view(B, Tnew, x.size(1))
        return x.transpose(1, 2)

    def init_from(self, path):
        """Load the frozen FRCNN visual backbone from ``path`` (A2-2).

        Fails fast instead of silently leaving parts of the encoder randomly
        initialised: the path must exist, the payload must be a
        ``{"model_state_dict": {...}}`` mapping, and the key set must match the
        model exactly (see ``update_frcnn_parameter``).
        """
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"FRCNN pretrain file not found: {path!r}. "
                "Export it from the released Falcon checkpoint with "
                "tools/repro/extract_video_backbone.py, or pass an existing path."
            )
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise KeyError(
                f"FRCNN pretrain file {path!r} must be a dict with a 'model_state_dict' "
                f"entry; got keys {list(payload) if isinstance(payload, dict) else type(payload)}"
            )
        pretrained_dict = payload["model_state_dict"]
        if not isinstance(pretrained_dict, dict) or not pretrained_dict:
            raise ValueError(f"FRCNN pretrain file {path!r} has an empty/non-dict model_state_dict")
        update_frcnn_parameter(self, pretrained_dict)

    def train(self, mode=True):
        """Set training mode while keeping BatchNorm statistics frozen.

        Returns ``self`` so that the standard ``model.train()`` / ``model.eval()`` contract
        holds (``nn.Module.eval`` is implemented as ``return self.train(False)``; without this
        return value ``FRCNNVideoModel(...).eval()`` would evaluate to ``None``).
        """
        super().train(mode)
        if mode:  # freeze BN stats
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


# Visual-encoder keys are the only ones a FRCNN backbone checkpoint may carry.
# Anything else (e.g. legacy ``*tcn*`` decoder heads in older checkpoints) is not
# part of the frozen lip-reading frontend and is dropped explicitly.
FRCNN_KEY_PREFIXES = ("frontend3D.", "trunk.")
FRCNN_LEGACY_SKIP_SUBSTRINGS = ("tcn",)


def filter_frcnn_pretrained_dict(pretrained_dict):
    """Keep only visual-encoder tensors from a pretrained dict (A2-2)."""
    filtered = {}
    for k, v in pretrained_dict.items():
        if not torch.is_tensor(v):
            continue
        if any(s in k for s in FRCNN_LEGACY_SKIP_SUBSTRINGS):
            continue
        if not k.startswith(FRCNN_KEY_PREFIXES):
            continue
        filtered[k] = v
    return filtered


def update_frcnn_parameter(model, pretrained_dict, strict=True):
    """Load the frozen FRCNN visual backbone, raising on any mismatch (A2-2).

    The previous implementation merged whatever happened to match and silently
    left everything else randomly initialised, then froze the whole encoder
    (missing keys -> random frozen features, no warning). Now, after filtering to
    the allowed visual-encoder prefixes, any missing / unexpected / wrong-shaped
    key raises ``RuntimeError`` unless ``strict=False`` is passed explicitly.
    On success all parameters are frozen (``requires_grad=False``).
    """
    model_dict = model.state_dict()
    filtered = filter_frcnn_pretrained_dict(pretrained_dict)
    if not filtered:
        found = sorted({k.split(".")[0] for k in pretrained_dict})
        raise RuntimeError(
            "FRCNN pretrain dict contains no visual-encoder keys "
            f"(allowed prefixes: {FRCNN_KEY_PREFIXES}); found top-level prefixes: {found}. "
            "Check that the file is an FRCNN/falcon visual backbone rather than a full model."
        )

    missing = [k for k in model_dict if k not in filtered]
    unexpected = [k for k in filtered if k not in model_dict]
    shape_mismatch = [
        (k, tuple(filtered[k].shape), tuple(model_dict[k].shape))
        for k in filtered
        if k in model_dict and tuple(filtered[k].shape) != tuple(model_dict[k].shape)
    ]
    if strict and (missing or unexpected or shape_mismatch):
        raise RuntimeError(
            "FRCNN pretrained backbone does not match the model: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"shape_mismatch={len(shape_mismatch)} "
            f"(model keys={len(model_dict)}, checkpoint visual keys={len(filtered)}). "
            f"missing[:5]={missing[:5]} unexpected[:5]={unexpected[:5]} "
            f"shape_mismatch[:3]={shape_mismatch[:3]}. "
            "Re-export with tools/repro/extract_video_backbone.py or fix the checkpoint."
        )

    model.load_state_dict(filtered, strict=bool(strict))
    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == "__main__":
    frames = torch.randn(1, 1, 100, 96, 96)
