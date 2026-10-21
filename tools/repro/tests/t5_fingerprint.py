#!/usr/bin/env python
"""T5 helper: capture a deterministic forward fingerprint of the released Falcon weights.

Used by Phase-1 T5 to prove that the three small API fixes
(``FRCNNVideoModel.train()`` returning ``self``, the ``Falcon._mdcl_loss`` alias and the
``look2hear.models.__all__`` fix) do **not** change any forward value.

Inputs are fully specified here so the JSON is reproducible:

* eval / train audio-visual forward: ``torch.manual_seed(0)`` then
  ``audio = randn(1, 32000)``, ``mouth_emb = randn(1, 512, 50)``;
* video encoder forward: ``torch.manual_seed(0)`` then ``frames = randn(1, 1, 50, 88, 88)``.

(The eval/train hashes match ``docs/verification/phase1/baseline-fingerprint.json`` exactly;
the baseline file did not record the video input convention, so the video entry here is
self-defined -- see ``docs/verification/phase1/T5-report.md``.)

Usage::

    PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python \
        tools/repro/tests/t5_fingerprint.py --out docs/verification/phase1/T5-pre-fix-fingerprint.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CKPT = Path("/root/.cache/falcon_weights/falcon.pt")


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()


def _load_state_dict(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("state_dict", ck)


def build_models(ckpt: Path):
    sys.path.insert(0, str(REPO))
    import look2hear.models as models  # noqa: E402
    from look2hear.videomodels import FRCNNVideoModel  # noqa: E402

    cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
    acfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
    vcfg = copy.deepcopy(cfg["videonet"]["videonet_config"])
    vcfg.pop("pretrain", None)  # weights are injected below, no file needed

    sd = _load_state_dict(ckpt)
    audio_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    video_sd = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}

    audio_model = models.FalconSpeakerContrastive(sample_rate=16000, **acfg)
    audio_model.load_state_dict(audio_sd, strict=True)
    video_model = FRCNNVideoModel(**vcfg)
    video_model.load_state_dict(video_sd, strict=True)
    return audio_model, video_model, len(audio_sd), len(video_sd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()

    audio_model, video_model, n_audio_keys, n_video_keys = build_models(args.ckpt)

    # --- eval forward (matches baseline-fingerprint.json) ---------------------
    audio_model.eval()
    torch.manual_seed(0)
    audio = torch.randn(1, 32000)
    mouth_emb = torch.randn(1, 512, 50)
    with torch.no_grad():
        est_eval = audio_model(audio, mouth_emb)

    # --- train forward (returns (est, sep_emb)) ------------------------------
    audio_model.train()
    torch.manual_seed(0)
    audio_t = torch.randn(1, 32000)
    mouth_t = torch.randn(1, 512, 50)
    with torch.no_grad():
        est_train, sep_train = audio_model(audio_t, mouth_t)

    # --- video encoder forward ----------------------------------------------
    video_model.eval()
    torch.manual_seed(0)
    frames = torch.randn(1, 1, 50, 88, 88)
    with torch.no_grad():
        video_out = video_model(frames)

    fp = {
        "env": {"python": platform.python_version(), "torch": torch.__version__},
        "weights": {"path": str(args.ckpt), "audio_keys": n_audio_keys, "video_keys": n_video_keys},
        "inputs": {
            "audio": "torch.manual_seed(0); torch.randn(1, 32000)",
            "mouth_emb": "torch.manual_seed(0); torch.randn(1, 512, 50)",
            "video_frames": "torch.manual_seed(0); torch.randn(1, 1, 50, 88, 88)",
        },
        "params": {
            "audio": sum(p.numel() for p in audio_model.parameters()),
            "video": sum(p.numel() for p in video_model.parameters()),
        },
        "eval_forward": {
            "shape": list(est_eval.shape), "sum": float(est_eval.sum()),
            "mean": float(est_eval.mean()), "sha256": _sha(est_eval),
        },
        "train_forward": {
            "est_shape": list(est_train.shape), "est_sum": float(est_train.sum()),
            "est_sha256": _sha(est_train),
            "sep_shape": list(sep_train.shape), "sep_sum": float(sep_train.sum()),
            "sep_sha256": _sha(sep_train),
        },
        "video_forward": {
            "shape": list(video_out.shape), "sum": float(video_out.sum()),
            "mean": float(video_out.mean()), "sha256": _sha(video_out),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fp, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(f"  params audio={fp['params']['audio']:,} video={fp['params']['video']:,}")
    print(f"  eval  sha256={fp['eval_forward']['sha256']} sum={fp['eval_forward']['sum']:.6f}")
    print(f"  train est_sha256={fp['train_forward']['est_sha256']}")
    print(f"  video sha256={fp['video_forward']['sha256']} sum={fp['video_forward']['sum']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
