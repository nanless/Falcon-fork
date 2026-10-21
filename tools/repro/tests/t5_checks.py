#!/usr/bin/env python
"""Phase-1 T5 verification checks (visual backbone export + small API fixes).

Run standalone (its stdout is saved as ``docs/verification/phase1/T5-backbone-check.log``)::

    PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/tests/t5_checks.py

or through ``pytest tools/repro/tests/test_t5_backbone.py -q`` (thin wrappers).

Covers:
1. imports + ``look2hear.models.__all__`` + ``models.get``;
2. the three API fixes (``FRCNNVideoModel.train`` returns self, ``Falcon._mdcl_loss`` alias,
   the alias really unblocks ``Falcon.forward(..., near_audio=...)``);
3. ``tools/repro/extract_video_backbone.py`` export (137 tensors, ``init_from``-compatible);
4. ``FRCNNVideoModel(pretrain=<exported>)``: 11,186,688 params, all ``requires_grad=False``,
   forward ``[1,1,50,88,88] -> [1,512,50]``, and bit-identical to a manual ``load_state_dict``.
"""
from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CKPT = Path("/root/.cache/falcon_weights/falcon.pt")
EXPORT = Path("/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar")
VFRAME_SHAPE = (1, 1, 50, 88, 88)
VOUT_SHAPE = (1, 512, 50)
VIDEO_PARAMS = 11_186_688


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()


def _cfg():
    cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
    return cfg


def check_imports() -> None:
    print("\n[1] imports / __all__ / models.get")
    import look2hear.models as models
    import look2hear.videomodels as videomodels

    print(f"  look2hear.models: {models.__file__}")
    print(f"  look2hear.videomodels: {videomodels.__file__}")
    print(f"  models.__all__ = {models.__all__}")
    ns: dict = {}
    exec("from look2hear.models import *", ns)  # noqa: S102 - explicit star-import contract
    star_names = {k for k in ns if not k.startswith("__")}
    print(f"  star-import names: {sorted(star_names)}")
    assert {"Falcon", "FalconSpeakerContrastive", "get", "register_model"} <= star_names
    assert models.get("Falcon") is models.Falcon
    assert models.get("falconspeakercontrastive") is models.FalconSpeakerContrastive
    assert callable(models.get)
    print("  OK")


def check_api_fixes() -> None:
    print("\n[2] API fixes")
    from look2hear.models.Falcon import Falcon
    from look2hear.videomodels import FRCNNVideoModel

    assert hasattr(Falcon, "_mdcl_loss"), "Falcon._mdcl_loss alias missing"
    assert Falcon._mdcl_loss is Falcon._mdcl_loss_hard
    print("  Falcon._mdcl_loss is Falcon._mdcl_loss_hard: True")

    vcfg = copy.deepcopy(_cfg()["videonet"]["videonet_config"])
    vcfg.pop("pretrain", None)
    vm = FRCNNVideoModel(**vcfg)
    assert vm.train(True) is vm
    assert vm.eval() is vm
    print("  FRCNNVideoModel.train(True)/eval() return self: True")

    # `.train()` freezes BN stats: every BatchNorm module must be in eval mode afterwards
    from torch.nn.modules.batchnorm import _BatchNorm

    vm.train(True)
    bn_modes = {m.training for m in vm.modules() if isinstance(m, _BatchNorm)}
    print(f"  BatchNorm modules training flags after train(True): {bn_modes}")
    assert bn_modes == {False}

    # the alias must make the historical MDCL path callable (bare Falcon, official weights)
    cfg = _cfg()["audionet"]["audionet_config"]
    bare_kwargs = {k: copy.deepcopy(v) for k, v in cfg.items() if k != "spk_embed_dim"}
    bare = Falcon(sample_rate=16000, **bare_kwargs)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    audio_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    missing, unexpected = bare.load_state_dict(audio_sd, strict=False)
    print(f"  bare Falcon weight load: missing={len(missing)} unexpected={len(unexpected)}")
    bare.train()
    torch.manual_seed(0)
    audio = torch.randn(1, 32000)
    mouth_emb = torch.randn(1, 512, 50)
    near = [torch.randn(1, 32000)]
    with torch.no_grad():
        est, mdcl = bare(audio, mouth_emb, near_audio=near)
    print(f"  Falcon.forward(near_audio=[tensor]) -> est {tuple(est.shape)}, "
          f"mdcl={float(mdcl):.6f} finite={torch.isfinite(mdcl).item()}")
    assert est.shape == (1, 1, 32000)
    assert torch.isfinite(mdcl)
    print(f"  NOTE: random inputs give mdcl={float(mdcl):.6f} (margin=0 clips to 0); "
          "checking non-trivial equivalence below")

    # alias must be numerically identical to the implementation, including a non-zero case
    torch.manual_seed(0)
    ref = torch.randn(2, 16000)
    est_like = (ref + 0.05 * torch.randn(2, 16000)).requires_grad_(False)
    loss_alias = bare._mdcl_loss(est_like, [ref])
    loss_hard = bare._mdcl_loss_hard(est_like, [ref])
    print(f"  alias loss={float(loss_alias):.6f} hard loss={float(loss_hard):.6f} "
          f"equal={torch.equal(loss_alias, loss_hard)}")
    assert torch.equal(loss_alias, loss_hard)
    assert float(loss_alias) > 0.0
    print("  OK")


def run_extract(force: bool = True, dst: Path = EXPORT) -> Path:
    cmd = [sys.executable, str(REPO / "tools" / "repro" / "extract_video_backbone.py"),
           "--variant", "small", "--dst", str(dst)]
    if force:
        cmd.append("--force")
    print("\n[3] extract_video_backbone.py")
    print("  $ " + " ".join(cmd))
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(out.stdout.rstrip())
    if out.returncode != 0:
        print(out.stderr.rstrip())
        raise RuntimeError(f"extractor failed (exit {out.returncode})")
    assert dst.is_file()
    payload = torch.load(dst, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload and len(payload["model_state_dict"]) == 137
    assert "meta" in payload
    print(f"  file size: {dst.stat().st_size / 1e6:.1f} MB; keys=137; meta OK")
    return dst


def check_backbone_load(export: Path = EXPORT) -> None:
    print("\n[4] FRCNNVideoModel(pretrain=<exported>) load + forward")
    from look2hear.videomodels import FRCNNVideoModel

    vcfg = copy.deepcopy(_cfg()["videonet"]["videonet_config"])
    vcfg["pretrain"] = str(export)
    vm = FRCNNVideoModel(**vcfg)
    vm.eval()
    n_params = sum(p.numel() for p in vm.parameters())
    n_trainable = sum(p.numel() for p in vm.parameters() if p.requires_grad)
    print(f"  params={n_params:,} trainable={n_trainable:,} "
          f"all_frozen={n_trainable == 0}")
    assert n_params == VIDEO_PARAMS
    assert n_trainable == 0

    torch.manual_seed(0)
    frames = torch.randn(*VFRAME_SHAPE)
    with torch.no_grad():
        out = vm(frames)
    print(f"  forward {VFRAME_SHAPE} -> {tuple(out.shape)} (finite={torch.isfinite(out).all().item()})")
    assert tuple(out.shape) == VOUT_SHAPE

    # cross-check: pretrain path == manual load_state_dict path (bit identical)
    vcfg2 = copy.deepcopy(vcfg)
    vcfg2.pop("pretrain")
    vm2 = FRCNNVideoModel(**vcfg2)
    payload = torch.load(export, map_location="cpu", weights_only=False)["model_state_dict"]
    vm2.load_state_dict(payload, strict=True)
    vm2.eval()
    with torch.no_grad():
        out2 = vm2(frames)
    print(f"  pretrain vs manual-load output sha equal: {_sha(out) == _sha(out2)}")
    assert _sha(out) == _sha(out2)
    print(f"  output sha256={_sha(out)}")
    print("  OK")


def check_big_variant() -> None:
    print("\n[5] --variant big export (big ships a different visual backbone)")
    big_dst = EXPORT.with_name("frcnn_128_512.backbone.big.pth.tar")
    cmd = [sys.executable, str(REPO / "tools" / "repro" / "extract_video_backbone.py"),
           "--variant", "big", "--dst", str(big_dst), "--force"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(out.stdout.rstrip())
    if out.returncode != 0:
        print(out.stderr.rstrip())
        raise RuntimeError(f"big-variant extractor failed (exit {out.returncode})")
    small = torch.load(EXPORT, map_location="cpu", weights_only=False)["model_state_dict"]
    big = torch.load(big_dst, map_location="cpu", weights_only=False)["model_state_dict"]
    same_keys = set(small) == set(big)
    same_values = all(torch.equal(small[k], big[k]) for k in small)
    n_diff = sum(0 if torch.equal(small[k], big[k]) else 1 for k in small)
    print(f"  small vs big video tensors: same_keys={same_keys} same_values={same_values} "
          f"({n_diff}/{len(small)} tensors differ)")
    assert same_keys
    # FINDING: falcon.pt and falcon-big.pt ship *different* visual-backbone weights
    # (same architecture / key set). Always extract the backbone from the matching variant.
    print("  FINDING: small/big releases have distinct visual-backbone values -> "
          "use the matching --variant when exporting")
    print("  OK")


def check_fingerprint_invariance() -> None:
    print("\n[6] pre-fix vs post-fix forward fingerprint (official weights, bit-identical)")
    import json

    pre_p = REPO / "docs" / "verification" / "phase1" / "T5-pre-fix-fingerprint.json"
    post_p = REPO / "docs" / "verification" / "phase1" / "T5-post-fix-fingerprint.json"
    if not (pre_p.is_file() and post_p.is_file()):
        print("  SKIP: fingerprint JSONs not found")
        return
    pre = json.loads(pre_p.read_text())
    post = json.loads(post_p.read_text())
    same = True
    for sec in ("eval_forward", "train_forward", "video_forward"):
        for key, val in pre[sec].items():
            new = post[sec][key]
            if key.endswith(("sha256", "shape")):
                ok = val == new
            else:
                ok = abs(float(val) - float(new)) == 0.0
            same &= ok
            print(f"  {sec}.{key}: {'SAME' if ok else 'DIFF'}")
    same &= pre["params"] == post["params"]
    print(f"  params equal: {pre['params'] == post['params']} ({pre['params']})")
    print(f"  eval sha matches Lead baseline: "
          f"{post['eval_forward']['sha256'] == 'ad0f20a0e67e8e750058bfea69946ca1ee5ff9d7ec6d3caa1f212cb0e8f40757'}")
    print(f"  train est sha matches Lead baseline: "
          f"{post['train_forward']['est_sha256'] == '5f01a59658471af0955cf6c18c09bb7c0788f844c9e0db334db36fa0a7e7545a'}")
    assert same, "numeric invariance violated!"
    print("  OK")


def main() -> int:
    print("=" * 78)
    print("T5 backbone/API checks")
    print("=" * 78)
    check_imports()
    check_api_fixes()
    export = run_extract()
    check_backbone_load(export)
    check_big_variant()
    check_fingerprint_invariance()
    print("\nALL T5 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
