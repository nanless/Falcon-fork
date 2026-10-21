"""R2-prep supplement: reproduce P0 (mono wav -> eval Dataset collapse) with real models,
and verify the audio_visual_test data-path contract.

Run from repo root:
  PYTHONPATH=/tmp/dp-verify/stubs:$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_p0_eval.py
Writes nothing into the repo.
"""
import ast
import json
import os
import shutil
import sys
import traceback

import numpy as np
import soundfile as sf
import torch
import yaml

REPO = "/root/code/github_repos/Falcon-fork"
sys.path.insert(0, REPO)
_DP_STUB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "docs", "verification", "stubs", "data_pipe"))
sys.path.insert(0, _DP_STUB)
sys.path.insert(0, "/tmp/dp-verify/stubs")  # optional fallback

from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset
from look2hear.datas.avspeech_dataset_near import AVSpeechDataModuleWithNear
from look2hear.utils.torch_utils import tensors_to_device

SR = 16000
ROOT = "/tmp/dp-verify"
LOG = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.append(line)


def section(t):
    say("\n" + "=" * 78)
    say("== " + t)
    say("=" * 78)


def make_split(root, stems, mono=True, frames=62, length=40000, n_near_meta=0):
    """Create a minimal {mix,target,mouths} + JSON index split (one split dir)."""
    shutil.rmtree(root, ignore_errors=True)
    for sub in ["mix", "target", "mouths"]:
        os.makedirs(os.path.join(root, sub))
    rs = np.random.RandomState(0)
    mix_json, tgt_json = [], []
    for i, s in enumerate(stems):
        x = (rs.randn(length) * 0.3).astype(np.float32)
        y = (rs.randn(length) * 0.1).astype(np.float32)
        mixp = os.path.join(root, "mix", s + ".wav")
        tgtp = os.path.join(root, "target", s + ".wav")
        mop = os.path.join(root, "mouths", s + ".pt")
        if mono:
            sf.write(mixp, x, SR)
            sf.write(tgtp, y, SR)
        else:
            sf.write(mixp, np.stack([x, x], axis=1), SR)   # 2-channel (dup) like implicit requirement
            sf.write(tgtp, np.stack([y, y], axis=1), SR)
        torch.save(torch.rand(frames, 96, 96, 1) * 255.0, mop)
        n = sf.info(mixp).frames
        mix_json.append([mixp, n])
        tgt_json.append([tgtp, mop, n])
        if n_near_meta:
            meta = {"components": [{"type": "near", "path": f"/nonexistent_{k}.wav"} for k in range(n_near_meta)]
                    + [{"type": "far", "path": "/nonexistent_far.wav"}]}
            json.dump(meta, open(os.path.join(root, "mix", s + "_meta.json"), "w"))
    json.dump(mix_json, open(os.path.join(root, "mix.json"), "w"))
    json.dump(tgt_json, open(os.path.join(root, "target.json"), "w"))


section("0. build minimal fixtures")
P0MONO = os.path.join(ROOT, "p0/mono")
P0STEREO = os.path.join(ROOT, "p0/stereo")
P0ODD = os.path.join(ROOT, "p0/stereo_odd")
make_split(P0MONO, ["u0"], mono=True, frames=62, length=40000, n_near_meta=2)
make_split(P0STEREO, ["u0"], mono=False, frames=62, length=40000)
# odd audio length: 3.032 s (= 48513 samples, not a multiple of 640) with 75 mouth frames
make_split(P0ODD, ["u0"], mono=False, frames=75, length=48513)
say("mono fixture  :", P0MONO)
say("stereo fixture:", P0STEREO)
say("odd  fixture  :", P0ODD)

section("1. fake a mono-layout dir: AVSpeechEvalDataset item shapes")
ev_mono = AVSpeechEvalDataset(json_dir=P0MONO, sample_rate=16000, normalize_audio=False)
it_mono = ev_mono[0]
say("len(eval_mono) =", len(ev_mono))
say("item types      :", [type(x).__name__ for x in it_mono])
say("mixture.shape   :", tuple(it_mono[0].shape), it_mono[0].dtype, "| ndim =", it_mono[0].ndim)
say("source.shape    :", tuple(it_mono[1].shape), it_mono[1].dtype)
say("mouth.shape     :", tuple(it_mono[2].shape), it_mono[2].dtype)
say("n_near          :", it_mono[4])
say("mix[None].shape :", tuple(it_mono[0][None].shape))
ev_st = AVSpeechEvalDataset(json_dir=P0STEREO, sample_rate=16000, normalize_audio=False)
it_st = ev_st[0]
say("stereo: mixture.shape =", tuple(it_st[0].shape), it_st[0].dtype,
    "| mouth.shape =", tuple(it_st[2].shape))

section("2. build the real models from configs/Falcon.yml (pretrain=None)")


def build_models():
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "Falcon.yml")))
    import look2hear.models as models
    import look2hear.videomodels as vmodels
    a_cfg = dict(cfg["audionet"]["audionet_config"])
    model = models.FalconSpeakerContrastive(
        sample_rate=cfg["datamodule"]["data_config"]["sample_rate"], **a_cfg)
    v_cfg = dict(cfg["videonet"]["videonet_config"])
    v_cfg.pop("pretrain", None)
    video = vmodels.FRCNNVideoModel(**v_cfg, pretrain=None)
    model.eval()
    video.eval()
    return model, video


model, video = build_models()
say("audio params:", sum(p.numel() for p in model.parameters()),
    "| video params:", sum(p.numel() for p in video.parameters()))

section("3. replicate audio_visual_test.py:229-231 for the MONO fixture")
mix, sources, mouth, key, n_near = tensors_to_device(it_mono, device=torch.device("cpu"))
mouth_in = torch.from_numpy(mouth[None, None]).float().to("cpu")
say("key =", key, "| n_near =", n_near)
say("mouth_in.shape =", tuple(mouth_in.shape))
mouth_emb = video(mouth_in)
say("mouth_emb.shape =", tuple(mouth_emb.shape))
say("mix[None].shape =", tuple(mix[None].shape), "  (audio_visual_test.py:231 passes this as [B,T])")
try:
    est = model(mix[None], mouth_emb)
    say("!! NO CRASH: est_sources.shape =", tuple(est.shape), "-> SILENT DEGENERATION")
except Exception as e:  # noqa: BLE001
    tb = traceback.format_exc().strip().splitlines()
    say("!! EXCEPTION:", type(e).__name__ + ":", e)
    say("!! traceback tail:")
    for ln in tb[-6:]:
        say("   " + ln)

section("4. same lines for the STEREO fixture (implicit multi-channel requirement)")
mix_s, sources_s, mouth_s, key_s, n_s = tensors_to_device(it_st, device=torch.device("cpu"))
mouth_emb_s = video(torch.from_numpy(mouth_s[None, None]).float())
with torch.no_grad():
    est_s = model(mix_s[None], mouth_emb_s)
say("stereo: mix.shape =", tuple(mix_s.shape), "| mix[None].shape =", tuple(mix_s[None].shape))
say("stereo: mouth_emb.shape =", tuple(mouth_emb_s.shape))
say("stereo: est_sources.shape =", tuple(est_s.shape), est_s.dtype,
    "| output length == input length:", est_s.shape[-1] == mix_s.shape[-1])

section("5. manual fix demo: apply ndim guard (near.py:176-177 style) -> eval works on mono")
raw_mix = sf.read(os.path.join(P0MONO, "mix", "u0.wav"), dtype="float32")[0]
raw_src = sf.read(os.path.join(P0MONO, "target", "u0.wav"), dtype="float32")[0]
fixed_mix = torch.from_numpy(raw_mix).mean(-1) if raw_mix.ndim > 1 else torch.from_numpy(raw_mix)
fixed_src = torch.from_numpy(raw_src).mean(-1) if raw_src.ndim > 1 else torch.from_numpy(raw_src)
say("after guard: fixed_mix.shape =", tuple(fixed_mix.shape), fixed_mix.dtype)
with torch.no_grad():
    est_fixed = model(fixed_mix[None], mouth_emb)
say("after guard: est_sources.shape =", tuple(est_fixed.shape),
    "| length == T:", est_fixed.shape[-1] == fixed_mix.shape[-1])

section("6. training-time validation path also affected (default_collate of 0-dim item)")
from torch.utils.data import DataLoader


def collate_probe(batch):
    return torch.utils.data.dataloader.default_collate(batch)


dl = DataLoader(ev_mono, batch_size=1, shuffle=False, collate_fn=collate_probe)
b = next(iter(dl))
say("val-batch types:", [type(x).__name__ for x in b])
say("val-batch mixture.shape =", tuple(b[0].shape), "| targets.shape =", tuple(b[1].shape),
    "| mouth.shape =", tuple(b[2].shape), "| n_near.shape =", tuple(b[4].shape))
try:
    mouth_b = b[2].unsqueeze(1) if b[2].ndim == 4 else b[2]   # av_litmodule.py:53-58
    emb = video(mouth_b.float())
    out = model(b[0], emb)
    say("!! NO CRASH: val est.shape =", tuple(out.shape))
except Exception as e:  # noqa: BLE001
    say("!! val path EXCEPTION:", type(e).__name__ + ":", e)

section("7. eval-path alignment: whole utterance, no trunc/pad; video frames nearest-interpolated")
it_odd = AVSpeechEvalDataset(json_dir=P0ODD, sample_rate=16000, normalize_audio=False)[0]
say("odd fixture: audio T =", it_odd[0].shape[-1], "| mouth F =", it_odd[2].shape[0],
    "| input wav frames =", sf.info(os.path.join(P0ODD, "mix", "u0.wav")).frames)
# instrument NeuroSCNetFusion to capture (audio T_a, video T_v) at fusion time
from look2hear.models.Falcon import NeuroSCNetFusion  # noqa: E402
orig_fwd = NeuroSCNetFusion.forward
seen = []


def probe(self, audio, video_feat):
    seen.append((tuple(audio.shape), tuple(video_feat.shape)))
    return orig_fwd(self, audio, video_feat)


NeuroSCNetFusion.forward = probe
try:
    mix_o, src_o, mouth_o, key_o, n_o = tensors_to_device(it_odd, device=torch.device("cpu"))
    with torch.no_grad():
        est_o = model(mix_o[None], video(torch.from_numpy(mouth_o[None, None]).float()))
    say("fusion calls (audio.shape, video.shape):")
    for s in seen:
        say("   ", s)
    say("odd: out.shape =", tuple(est_o.shape), "| length == T:", est_o.shape[-1] == mix_o.shape[-1])
    t_audio = seen[0][0][2] if seen else None
    say("=> 25 fps frames F =", mouth_o.shape[0], "; audio STFT time steps T' =", t_audio,
        "; nominal ratio T'/F =", None if not t_audio else round(t_audio / mouth_o.shape[0], 2),
        "(640 samples/frame / hop 128 => ~5 audio steps per video frame)")
finally:
    NeuroSCNetFusion.forward = orig_fwd

section("8. audio_visual_test.py data-path static contract (AST audit)")
src = open(os.path.join(REPO, "audio_visual_test.py")).read()
tree = ast.parse(src)
attr_names, call_names = set(), set()
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute):
        attr_names.add(node.attr)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            call_names.add(f.id)
        elif isinstance(f, ast.Attribute):
            call_names.add(f.attr)
for probe_name in ["DataLoader", "train_dataloader", "val_dataloader", "test_dataloader",
                   "make_loader", "make_sets", "setup", "tensors_to_device"]:
    say(f"  audio_visual_test.py contains {probe_name:16s}:",
        probe_name in call_names or probe_name in attr_names)
say("  (only make_sets/setup/tensors_to_device appear; no DataLoader/dataloaders => eval is per-sample)")

section("9. make_sets + use_near=False actual behaviour (replicate audio_visual_test.py:183-189)")
cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "Falcon.yml")))
data_cfg = {k: v for k, v in cfg["datamodule"]["data_config"].items()
            if k not in ("use_near", "train_samples")}
data_cfg.update({"train_dir": os.path.join(ROOT, "index/tr"),
                 "valid_dir": os.path.join(ROOT, "index/cv"),
                 "test_dir": os.path.join(ROOT, "index/tt")})
dm = AVSpeechDataModuleWithNear(**data_cfg, use_near=False)
dm.setup()
_, _, test_set = dm.make_sets
say("make_sets types:", [type(d).__name__ for d in dm.make_sets])
say("data_train.use_near =", dm.data_train.use_near,
    "| #non-empty near lists =", sum(1 for n in dm.data_train.near if n))
say("data_val/data_test type =", type(dm.data_val).__name__, "/", type(dm.data_test).__name__,
    "(use_near=False does NOT propagate to eval datasets)")
say("data_val[0] n_near (still parsed from _meta.json) =", dm.data_val[0][4])
say("test_set[0] tuple len =", len(test_set[0]), "| mouth.shape =", tuple(test_set[0][2].shape))
# use_near=False must not require near.json for the *train* split either
copy_idx = os.path.join(ROOT, "index_nonear_tr")
shutil.rmtree(copy_idx, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, "index"), copy_idx)
for split in ["tr", "cv", "tt"]:
    p = os.path.join(copy_idx, split, "near.json")
    if os.path.exists(p):
        os.remove(p)
data_cfg2 = dict(data_cfg)
data_cfg2.update({"train_dir": os.path.join(copy_idx, "tr"),
                  "valid_dir": os.path.join(copy_idx, "cv"),
                  "test_dir": os.path.join(copy_idx, "tt")})
dm2 = AVSpeechDataModuleWithNear(**data_cfg2, use_near=False)
try:
    dm2.setup()
    say("use_near=False + near.json deleted for all splits: setup() OK (train near all empty:",
        all(n == [] for n in dm2.data_train.near), ")")
except Exception as e:  # noqa: BLE001
    say("use_near=False + near.json deleted: setup() FAILED ->", type(e).__name__, e)

say("\n== DONE")
open(os.path.join(ROOT, "logs", "verify_p0_eval.log"), "w").write("\n".join(LOG) + "\n")
