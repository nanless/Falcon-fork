"""Phase-1 T5 regression tests: visual-backbone export + three API fixes.

Run::

    /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_t5_backbone.py -q

Weight-dependent tests are skipped when the cached checkpoints/export are unavailable
(``/root/.cache/falcon_weights/...``); everything else runs anywhere.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
EXPORT = Path("/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar")
VIDEO_PARAMS = 11_186_688

sys.path.insert(0, str(REPO))


def _load_cfg():
    cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
    return cfg


def _load_extractor():
    path = REPO / "tools" / "repro" / "extract_video_backbone.py"
    spec = importlib.util.spec_from_file_location("extract_video_backbone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_models_package_exports():
    """`__all__` must list names that actually exist (regression: was AVTIGER_causal)."""
    import look2hear.models as models

    assert models.__all__ == ["Falcon", "FalconSpeakerContrastive", "get", "register_model"]
    ns: dict = {}
    exec("from look2hear.models import *", ns)  # noqa: S102
    for name in models.__all__:
        assert name in ns, f"star-import missing {name}"
    assert models.get("Falcon") is models.Falcon
    assert models.get("FalconSpeakerContrastive") is models.FalconSpeakerContrastive


def test_frcnn_train_eval_return_self():
    """Regression: `FRCNNVideoModel.train()` used to return None, so `.eval()` returned None."""
    from torch.nn.modules.batchnorm import _BatchNorm

    from look2hear.videomodels import FRCNNVideoModel

    vcfg = copy.deepcopy(_load_cfg()["videonet"]["videonet_config"])
    vcfg.pop("pretrain", None)
    model = FRCNNVideoModel(**vcfg)

    assert model.train(True) is model
    assert model.eval() is model
    # BN statistics stay frozen while the module is in training mode
    model.train(True)
    assert model.training is True
    assert {m.training for m in model.modules() if isinstance(m, _BatchNorm)} == {False}


def test_mdcl_alias_is_numerically_identical():
    """`_mdcl_loss` alias must delegate to `_mdcl_loss_hard` without changing numbers."""
    from look2hear.models.Falcon import Falcon

    assert Falcon._mdcl_loss is Falcon._mdcl_loss_hard

    cfg = copy.deepcopy(_load_cfg()["audionet"]["audionet_config"])
    bare_kwargs = {k: v for k, v in cfg.items() if k != "spk_embed_dim"}
    model = Falcon(sample_rate=16000, **bare_kwargs)

    torch.manual_seed(0)
    ref = torch.randn(2, 16000)
    est = ref + 0.05 * torch.randn(2, 16000)
    loss_alias = model._mdcl_loss(est, [ref])
    loss_hard = model._mdcl_loss_hard(est, [ref])
    assert torch.equal(loss_alias, loss_hard)
    assert float(loss_alias) > 0.0  # correlated signals exceed margin=0


def test_falcon_forward_accepts_near_audio():
    """Regression: `Falcon.forward(..., near_audio=...)` raised AttributeError before the alias."""
    from look2hear.models.Falcon import Falcon

    cfg = copy.deepcopy(_load_cfg()["audionet"]["audionet_config"])
    bare_kwargs = {k: v for k, v in cfg.items() if k != "spk_embed_dim"}
    model = Falcon(sample_rate=16000, **bare_kwargs)
    model.train()

    torch.manual_seed(0)
    est, mdcl = model(
        torch.randn(1, 32000), torch.randn(1, 512, 50), near_audio=[torch.randn(1, 32000)]
    )
    assert tuple(est.shape) == (1, 1, 32000)
    assert torch.isfinite(mdcl)


@pytest.mark.skipif(not CKPT.is_file(), reason=f"missing released checkpoint {CKPT}")
def test_extract_state_dict_from_released_checkpoint():
    """Extraction helper yields exactly the 137 `video_model.*` tensors, prefix stripped."""
    mod = _load_extractor()
    ckpt = mod.load_checkpoint(CKPT)
    assert ckpt.get("architecture") == "FalconSpeakerContrastive"
    video_sd = mod.extract_video_state_dict(ckpt)
    assert len(video_sd) == 137
    assert not any(k.startswith("video_model.") for k in video_sd)
    assert sum(v.numel() for v in video_sd.values()) == 11_196_308
    payload = mod.build_output(video_sd, CKPT, ckpt)
    assert set(payload["model_state_dict"]) == set(video_sd)
    assert payload["meta"]["num_keys"] == 137


@pytest.mark.skipif(not EXPORT.is_file(), reason=f"missing exported backbone {EXPORT}")
def test_exported_backbone_loads_into_frcnn():
    """`FRCNNVideoModel(pretrain=<export>)`: 11,186,688 params, all frozen, correct forward shape."""
    from look2hear.videomodels import FRCNNVideoModel

    vcfg = copy.deepcopy(_load_cfg()["videonet"]["videonet_config"])
    vcfg["pretrain"] = str(EXPORT)
    model = FRCNNVideoModel(**vcfg)
    assert model.eval() is model
    assert sum(p.numel() for p in model.parameters()) == VIDEO_PARAMS
    assert all(not p.requires_grad for p in model.parameters())

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(torch.randn(1, 1, 50, 88, 88))
    assert tuple(out.shape) == (1, 512, 50)
    assert torch.isfinite(out).all()
