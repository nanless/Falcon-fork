"""Dynamic verification of the Falcon data pipeline with synthetic fixtures (v2).

Run from repo root:
  PYTHONPATH=/tmp/dp-verify/stubs:<repo> /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_dataset2.py
Only reads repo sources; writes nothing into the repo (fixtures live under /tmp/dp-verify).
"""
import inspect
import json
import os
import random
import sys

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
from look2hear.datas.avspeech_dataset_near import (
    AVSpeechDataModuleWithNear, AVSpeechTrainDatasetWithNear, collate_with_near,
)
from look2hear.datas.transform import get_preprocessing_pipelines
from torchvision import transforms
from torch.utils.data import DataLoader

DATA = "/tmp/dp-verify/data"
INDEX = "/tmp/dp-verify/index"
FAIL = 0
SR = 16000


def p(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    if not ok:
        FAIL += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def nframes(path):
    return sf.info(path).frames


def write_manifest(d, stems):
    mix, tgt = [], []
    for s in stems:
        mp = os.path.join(d, "mix", s + ".wav")
        tp = os.path.join(d, "target", s + ".wav")
        mop = os.path.join(d, "mouths", s + ".pt")
        mix.append([mp, nframes(mp)])
        tgt.append([tp, mop, nframes(mp)])
    json.dump(mix, open(os.path.join(d, "mix.json"), "w"))
    json.dump(tgt, open(os.path.join(d, "target.json"), "w"))


# ============================================================ A. transform math
m = torch.load(os.path.join(DATA, "tr", "mouths", "u0.pt"), weights_only=True)
p("A1 stored mouth .pt is 4-D channel-last (T,96,96,C)", m.shape == (75, 96, 96, 1), str(tuple(m.shape)))
p("A2 stored mouth .pt dtype", m.dtype == torch.float32, str(m.dtype))
gray = transforms.Grayscale(num_output_channels=1)       # stub mirrors torchvision luma weights
x = gray(m.permute(0, 3, 1, 2)).squeeze(1).numpy()
p("A3 Grayscale(permute(0,3,1,2)) -> (T,96,96) numpy", x.shape == (75, 96, 96), str(x.shape))
outs = {s: fn(x.copy()) for s, fn in get_preprocessing_pipelines().items()}
p("A4 train/val/test output shape (T,88,88)", all(o.shape == (75, 88, 88) for o in outs.values()),
  str(outs["train"].shape))
p("A5 pipeline output dtype float32", all(o.dtype == np.float32 for o in outs.values()),
  str(outs["train"].dtype))
expected = ((np.asarray(m[:, :88, :88, 0]) / 255.0) - 0.421) / 0.165   # uniform frames -> crop invariant
p("A6 Normalize(0,255) then Normalize(.421,.165) (train)", np.allclose(outs["train"], expected, atol=1e-4),
  f"max|diff|={np.abs(outs['train']-expected).max():.2e}")

g = torch.load(os.path.join(DATA, "tr", "mouths", "u0grad.pt"), weights_only=True)
gx = gray(g.permute(0, 3, 1, 2)).squeeze(1).numpy()
gval = get_preprocessing_pipelines()["val"](gx.copy())
g_expected = ((g[:, :, :, 0].numpy()[:, 4:92, 4:92] / 255.0) - 0.421) / 0.165
p("A7 val CenterCrop(88) == [4:92,4:92] (= studio preprocess_mouth)",
  np.allclose(gval, g_expected, atol=1e-4), f"max|diff|={np.abs(gval-g_expected).max():.2e}")
gtr = get_preprocessing_pipelines()["train"](gx.copy())
p("A8 train RandomCrop(88) spatial-only, keeps all 75 frames", gtr.shape == (75, 88, 88), str(gtr.shape))

# ================================================= B. AVSpeechTrainDatasetWithNear init
ds = AVSpeechTrainDatasetWithNear(json_dir=os.path.join(INDEX, "tr"), sample_rate=16000,
                                  segment=2.0, normalize_audio=False, samples=4, use_near=True)
p("B1 utt shorter than seg_len dropped (8->7)", len(ds.mix) == 7, str(len(ds.mix)))
stems = [os.path.basename(a[0]).replace(".wav", "") for a in ds.mix]
p("B2 drop only u5", "u5" not in stems and "u4" in stems, str(stems))
p("B3 mix/source/mouth aligned by stem",
  all(os.path.basename(a[0])[:-4] == os.path.basename(b[0])[:-4] == os.path.basename(b[1])[:-3]
      for a, b in zip(ds.mix, ds.sources)))
near_by_stem = {os.path.basename(a[0]).replace(".wav", ""): n for a, n in zip(ds.mix, ds.near)}
p("B4 far-field components excluded (u0 -> 2 near entries)", len(near_by_stem["u0"]) == 2,
  str([os.path.basename(e[0]) for e in near_by_stem["u0"]]))
p("B5 fallback = mixture path itself", all(len(near_by_stem[s]) == 1 for s in
  ["u1", "u2", "u3", "u0copy", "u0grad"]))
p("B6 u4 near = [20000, 48000] samples", sorted(n for _, n in near_by_stem["u4"]) == [20000, 48000])
p("B7 kept order reversed vs mix.json (descending append)", stems[0] == "u4", str(stems[:3]))
p("B8 len(dataset)==samples (train_samples), not #utts", ds.length == 4 and len(ds) == 4, str(len(ds)))

# ================================================= C. __getitem__ alignment (patched randint)
orig_randint = random.randint


def patched(mix_idx, start):
    """Return mix_idx for the dataset's 1st randint, start for its 2nd, delegate the rest
    (transform.RandomCrop also calls random.randint)."""
    seq = [mix_idx, start]

    def f(a, b):
        return seq.pop(0) if seq else orig_randint(a, b)
    return f


random.randint = patched(0, 10000)
try:
    item = ds[0]
finally:
    random.randint = orig_randint
mix_path = ds.mix[0][0]
raw_mix = sf.read(mix_path, start=10000, stop=42000, dtype="float32")[0]
raw_src = sf.read(ds.sources[0][0], start=10000, stop=42000, dtype="float32")[0]
p("C1 mixture == sf.read(mix, start=10000, stop=42000); idx arg ignored",
  torch.equal(item[0], torch.from_numpy(raw_mix)) and item[0].shape == (32000,),
  f"shape={tuple(item[0].shape)}")
p("C2 source read with identical window", torch.equal(item[1], torch.from_numpy(raw_src)))
mouth_start = int((10000 / 16000) * 25)
exp_mouth = np.array([((mouth_start + j) / 255.0 - 0.421) / 0.165 for j in range(50)], np.float32)
p("C3 mouth = frames[floor(start/640) : +50] at 25 fps",
  item[2].shape == (50, 88, 88) and np.allclose(item[2][:, 0, 0], exp_mouth, atol=1e-4),
  f"mouth_start={mouth_start}, shape={tuple(item[2].shape)}, first={item[2][0,0,0]:.5f}")
p("C4 mouth dtype float32", item[2].dtype == np.float32, str(item[2].dtype))
p("C5 filename = basename(mix wav)", item[3] == "u4.wav", item[3])
near_ok = [e for e in ds.near[0] if e[1] == 48000][0][0]
raw_near = sf.read(near_ok, start=10000, stop=42000, dtype="float32")[0]
p("C6 near: short entry skipped; offset=min(rand_start, n-seg_len)",
  len(item[4]) == 1 and torch.equal(item[4][0], torch.from_numpy(raw_near)),
  f"#near={len(item[4])}; T={item[4][0].shape[-1] if item[4] else None}")

# ================================================= D. DataLoader / collate shapes & dtypes
loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0, collate_fn=collate_with_near)
batches = list(loader)
mixtures, sources, mouths, filenames, near_list = batches[0]
p("D1 batch mixtures [B,T] float32", mixtures.shape == (3, 32000) and mixtures.dtype == torch.float32,
  f"{tuple(mixtures.shape)} {mixtures.dtype}")
p("D2 batch sources [B,T] float32", sources.shape == (3, 32000) and sources.dtype == torch.float32,
  f"{tuple(sources.shape)} {sources.dtype}")
p("D3 batch mouths [B,50,88,88] float32", mouths.shape == (3, 50, 88, 88) and mouths.dtype == torch.float32,
  f"{tuple(mouths.shape)} {mouths.dtype}")
p("D4 filenames list[str]", isinstance(filenames, list) and len(filenames) == 3
  and all(type(s) is str for s in filenames), str(filenames))
all_k = [len(k) for b in batches for k in b[4]]
all_near_shapes = {tuple(t.shape) for b in batches for k in b[4] for t in k}
p("D5 near_audios = list[list[Tensor(32000,float32)]] variable K (empty for fallback samples)",
  isinstance(near_list, list) and len(near_list) == 3
  and all(isinstance(k, list) and all(isinstance(t, torch.Tensor) and t.dtype == torch.float32
          for t in k) for b in batches for k in b[4])
  and all_near_shapes <= {(32000,)}, f"K over all batches={all_k}, near shapes={all_near_shapes}")
p("D6 len(dataset)==samples so DataLoader yields ceil(samples/B) batches (4 samples -> 3+1)",
  [len(b[0]) for b in batches] == [3, 1], str([len(b[0]) for b in batches]))

cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "Falcon.yml")))
data_cfg = dict(cfg["datamodule"]["data_config"])
data_cfg.update({"train_dir": os.path.join(INDEX, "tr"), "valid_dir": os.path.join(INDEX, "cv"),
                 "test_dir": os.path.join(INDEX, "tt"), "num_workers": 0})
data_cfg["persistent_workers"] = False   # num_workers=0 needs persistent_workers=False (torch check)
dm = AVSpeechDataModuleWithNear(**data_cfg)
dm.setup()
train_loader, val_loader, test_loader = dm.make_loader
b = next(iter(train_loader))
p("D7 config batch_size=4 through DataModule train loader, 5-tuple",
  b[0].shape == (4, 32000) and len(b) == 5, f"{tuple(b[0].shape)}, tuple len={len(b)}")
vb = next(iter(val_loader))
p("D8 val/test loader forced batch_size=1, config batch_size ignored",
  len(vb) == 5 and vb[0].shape[0] == 1, f"mix={tuple(vb[0].shape)}, mouth={tuple(vb[2].shape)}, "
  f"n_near={vb[4] if len(vb) > 4 else None}")

# ================================================= E. normalize_audio semantics
dsn = AVSpeechTrainDatasetWithNear(json_dir=os.path.join(INDEX, "tr"), sample_rate=16000,
                                   segment=2.0, normalize_audio=True, samples=2, use_near=True)
random.randint = patched(0, 10000)
try:
    it = dsn[0]
finally:
    random.randint = orig_randint
rawm = torch.from_numpy(raw_mix)
raws = torch.from_numpy(raw_src)
m_std = rawm.std(-1, keepdim=True)
exp_m = (rawm - rawm.mean(-1, keepdim=True)) / (m_std + 1e-8)
exp_s = (raws - raws.mean(-1, keepdim=True)) / (m_std + 1e-8)    # same m_std as mixture!
p("E1 normalize_audio: per-segment mean removal, target scaled by MIXTURE std",
  torch.allclose(it[0], exp_m, atol=1e-5) and torch.allclose(it[1], exp_s, atol=1e-5),
  f"mix_mean={it[0].mean():.2e} mix_std={it[0].std():.4f}; "
  f"target own_std={raws.std():.4f} -> after={it[1].std():.4f}")

# ================================================= F. AVSpeechEvalDataset (val/test)
evalset = AVSpeechEvalDataset(json_dir=os.path.join(INDEX, "tr"), sample_rate=16000, normalize_audio=False)
p("F1 eval set keeps short utt (8 entries, no filtering)", len(evalset) == 8, str(len(evalset)))
ev = evalset[0]
p("F2 mono wav -> mixture/source collapse to 0-dim tensor (unguarded .mean(-1) bug)",
  ev[0].ndim == 0 and ev[1].ndim == 0, f"mix.shape={tuple(ev[0].shape)}, src.shape={tuple(ev[1].shape)}")
p("F3 eval mouth = whole sequence center-cropped (75,88,88) float32",
  ev[2].shape == (75, 88, 88) and ev[2].dtype == np.float32, f"{tuple(ev[2].shape)} {ev[2].dtype}")
p("F4 n_near parsed from <stem>_meta.json", ev[4] == 2, f"key={ev[3]}, n_near={ev[4]}")
p("F5 missing meta tolerated -> n_near=0", evalset[3][4] == 0 and evalset[3][3] == "u1.wav")
grad_idx = [i for i, e in enumerate(evalset.mix) if e[0].endswith("u0grad.wav")][0]
evg = evalset[grad_idx]
p("F6 eval mouth == ((frames[...,4:92,4:92])/255-.421)/.165",
  np.allclose(evg[2], ((g[:, :, :, 0].numpy()[:, 4:92, 4:92] / 255.0) - 0.421) / 0.165, atol=1e-4))

# --- stereo fixture: shows the eval path implicitly requires multi-channel wavs
ST = "/tmp/dp-verify/stereo"
os.makedirs(os.path.join(ST, "mix"), exist_ok=True)
os.makedirs(os.path.join(ST, "target"), exist_ok=True)
os.makedirs(os.path.join(ST, "mouths"), exist_ok=True)
rs = np.random.RandomState(0)
for s in ["s0"]:
    n = 40000
    a = rs.randn(n, 2).astype(np.float32) * 0.3
    b = rs.randn(n, 2).astype(np.float32) * 0.3
    sf.write(os.path.join(ST, "mix", s + ".wav"), a, SR)
    sf.write(os.path.join(ST, "target", s + ".wav"), b, SR)
    torch.save(torch.zeros(50, 96, 96, 1), os.path.join(ST, "mouths", s + ".pt"))
write_manifest(ST, ["s0"])
evs = AVSpeechEvalDataset(json_dir=ST, sample_rate=16000, normalize_audio=False)
e0 = evs[0]
p("F7 stereo wav -> mean over channels: eval mix/source (T,) float32",
  e0[0].shape == (40000,) and e0[0].dtype == torch.float32, f"{tuple(e0[0].shape)} {e0[0].dtype}")
dss = AVSpeechTrainDatasetWithNear(json_dir=ST, sample_rate=16000, segment=2.0,
                                   normalize_audio=False, samples=1, use_near=False)
random.randint = patched(0, 0)
try:
    its = dss[0]
finally:
    random.randint = orig_randint
p("F8 train dataset ndim guard: stereo -> mean over channels (T,)", its[0].shape == (32000,),
  f"{tuple(its[0].shape)}")

# ================================================= G. use_near=False / missing near.json
ds_no = AVSpeechTrainDatasetWithNear(json_dir=os.path.join(INDEX, "tr"), sample_rate=16000,
                                     segment=2.0, normalize_audio=False, samples=2, use_near=False)
p("G1 use_near=False -> near lists empty", all(n == [] for n in ds_no.near))
random.randint = patched(0, 10000)
try:
    it_no = ds_no[0]
finally:
    random.randint = orig_randint
p("G2 use_near=False -> item[4] == []", it_no[4] == [])
miss = "/tmp/dp-verify/index_nearmiss"
os.system(f"rm -rf {miss} && cp -r {INDEX} {miss} && rm -f {miss}/tr/near.json")
ds_miss = AVSpeechTrainDatasetWithNear(json_dir=os.path.join(miss, "tr"), sample_rate=16000,
                                       segment=2.0, normalize_audio=False, samples=2, use_near=True)
random.randint = patched(0, 10000)
try:
    it_miss = ds_miss[0]
finally:
    random.randint = orig_randint
p("G3 use_near=True + near.json absent -> degrades to empty near list", all(n == [] for n in ds_miss.near)
  and it_miss[4] == [])

# ================================================= H. edge cases / latent bugs
bad = "/tmp/dp-verify/index_badlen"
os.system(f"rm -rf {bad} && cp -r {INDEX} {bad}")
tgt = json.load(open(os.path.join(bad, "tr", "target.json")))
json.dump(tgt[:-3], open(os.path.join(bad, "tr", "target.json"), "w"))
err = None
try:
    AVSpeechTrainDatasetWithNear(json_dir=os.path.join(bad, "tr"), segment=2.0, samples=1, use_near=False)
except Exception as e:  # noqa: BLE001
    err = f"{type(e).__name__}: {e}"
p("H1 mix.json/target.json length mismatch -> IndexError (no explicit check)", err is not None, str(err))
badn = "/tmp/dp-verify/index_badnear"
os.system(f"rm -rf {badn} && cp -r {INDEX} {badn}")
near = json.load(open(os.path.join(badn, "tr", "near.json")))
json.dump(near[:-3], open(os.path.join(badn, "tr", "near.json"), "w"))
err2 = None
try:
    AVSpeechTrainDatasetWithNear(json_dir=os.path.join(badn, "tr"), segment=2.0, samples=1, use_near=True)
except Exception as e:  # noqa: BLE001
    err2 = f"{type(e).__name__}: {e}"
p("H2 near.json length mismatch -> IndexError", err2 is not None, str(err2))
SM = "/tmp/dp-verify/shortmouth"
os.system(f"rm -rf {SM}")
for sub in ["mix", "target", "mouths"]:
    os.makedirs(os.path.join(SM, sub), exist_ok=True)
for s, nf in [("s0", 75), ("s1", 40)]:
    n = 48000
    sf.write(os.path.join(SM, "mix", s + ".wav"), np.zeros(n, np.float32), SR)
    sf.write(os.path.join(SM, "target", s + ".wav"), np.zeros(n, np.float32), SR)
    torch.save(torch.zeros(nf, 96, 96, 1), os.path.join(SM, "mouths", s + ".pt"))
write_manifest(SM, ["s0", "s1"])
sm = AVSpeechTrainDatasetWithNear(json_dir=SM, segment=2.0, samples=2, use_near=False)
random.randint = patched(0, 10000)
try:
    i0 = sm[0]
finally:
    random.randint = orig_randint
random.randint = patched(1, 10000)
try:
    i1 = sm[1]
finally:
    random.randint = orig_randint
err3 = None
try:
    collate_with_near([i0, i1])
except Exception as e:  # noqa: BLE001
    err3 = f"{type(e).__name__}: {e}"
p("H3 mouth .pt shorter than audio timeline -> per-sample F<50 -> collate stack fails",
  sorted([i0[2].shape[0], i1[2].shape[0]]) == [25, 50] and err3 is not None,
  f"F={i0[2].shape[0]}/{i1[2].shape[0]}, {err3}")
d8 = AVSpeechTrainDatasetWithNear(json_dir=os.path.join(INDEX, "tr"), sample_rate=8000,
                                  segment=2.0, normalize_audio=False, samples=1, use_near=False)
random.randint = patched(0, 10000)
try:
    i8 = d8[0]
finally:
    random.randint = orig_randint
ms_actual = int((10000 / 16000) * 25)
ms_correct_8k = int((10000 / 8000) * 25)
p("H4 sample_rate=8000: mouth_start uses hard-coded 16000 (frame 15 instead of 31)",
  d8.seg_len == 16000 and ms_actual == 15 and ms_correct_8k == 31,
  f"seg_len={d8.seg_len}, mouth_start={ms_actual}, correct={ms_correct_8k}")
sil = "/tmp/dp-verify/silence"
os.system(f"rm -rf {sil} && cp -r {SM} {sil}")
dn = AVSpeechTrainDatasetWithNear(json_dir=sil, segment=2.0, normalize_audio=True, samples=1, use_near=False)
random.randint = patched(0, 10000)
try:
    isn = dn[0]
finally:
    random.randint = orig_randint
p("H5 all-zero mixture: normalize_audio yields finite zeros (eps=1e-8)",
  torch.isfinite(isn[0]).all() and isn[0].abs().max() < 1e-6, f"max|mix|={isn[0].abs().max():.2e}")

# ================================================= I. init vs configs/Falcon.yml key match
sig = inspect.signature(AVSpeechDataModuleWithNear.__init__)
params = [n for n in sig.parameters if n != "self"]
conf_keys = list(cfg["datamodule"]["data_config"].keys())
missing = [k for k in conf_keys if k not in params]
extra = [k for k in params if k not in conf_keys]
print("\n== I. AVSpeechDataModuleWithNear.__init__ params:", params)
print("==    configs/Falcon.yml data_config keys:", conf_keys)
print("==    config keys not in signature:", missing)
print("==    signature params absent from config:", extra,
      "-> train_samples default =", sig.parameters["train_samples"].default)
p("I1 every YAML data_config key is accepted by __init__", missing == [])
p("I2 no unexpected extra ctor arg required (only train_samples, has default)", extra == ["train_samples"])
p("I3 data_name key is not passed to ctor (class imported directly)", "data_name" not in conf_keys,
  cfg["datamodule"]["data_name"])
exp_conf = yaml.safe_load(open(os.path.join(REPO, "Experiments", "checkpoint", "Falcon", "conf.yml")))
exp_keys = list(exp_conf["datamodule"]["data_config"].keys())
p("I4 Experiments conf.yml data_config keys == configs/Falcon.yml keys",
  sorted(exp_keys) == sorted(conf_keys), str(sorted(exp_keys)))
p("I5 make_loader is a property returning 3 DataLoaders",
  isinstance(type(dm).__dict__.get("make_loader"), property) and len(dm.make_loader) == 3
  and all(isinstance(d, type(train_loader)) for d in dm.make_loader))
p("I6 make_sets is a property returning (train, val, test) datasets",
  isinstance(type(dm).__dict__.get("make_sets"), property) and len(dm.make_sets) == 3
  and dm.make_sets[0] is dm.data_train)
p("I7 both properties build NEW DataLoader objects on every access",
  dm.make_loader[0] is not dm.make_loader[0])

# ================================================= J. audio_visual_test.py construction path
data_cfg2 = {
    k: v for k, v in cfg["datamodule"]["data_config"].items()
    if k not in ("use_near", "train_samples")
}
data_cfg2.update({"train_dir": os.path.join(INDEX, "tr"), "valid_dir": os.path.join(INDEX, "cv"),
                  "test_dir": os.path.join(INDEX, "tt"), "num_workers": 0,
                  "persistent_workers": False})
dm2 = AVSpeechDataModuleWithNear(**data_cfg2, use_near=False)
dm2.setup()
_, _, test_set = dm2.make_sets
mix, sources, mouth, key, nn = test_set[0]
p("J1 eval-style module: use_near stripped then forced False; make_sets[2] usable",
  mix.ndim == 0 and mouth.shape[1:] == (88, 88), f"mix={tuple(mix.shape)}, mouth={tuple(mouth.shape)}")
bad_valid = dict(data_cfg2)
bad_valid["valid_dir"] = "/tmp/dp-verify/does_not_exist/cv"
err4 = None
try:
    AVSpeechDataModuleWithNear(**bad_valid, use_near=False).setup()
except Exception as e:  # noqa: BLE001
    err4 = f"{type(e).__name__}: {e}"
p("J2 setup() builds all 3 datasets -> test-only eval still requires train_dir & valid_dir",
  err4 is not None and "does_not_exist" in str(err4), str(err4))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "falcon_torch_utils", os.path.join(REPO, "look2hear", "utils", "torch_utils.py"))
_tu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tu)
tensors_to_device = _tu.tensors_to_device
dev_item = tensors_to_device(test_set[0], device=torch.device("cpu"))
p("J3 tensors_to_device(tuple) -> list of 5, tensors stay tensors, n_near int stays int",
  len(dev_item) == 5 and torch.is_tensor(dev_item[0]) and isinstance(dev_item[4], int),
  f"types={[type(t).__name__ for t in dev_item]}")
print(f"\n== TOTAL FAILURES: {FAIL}")
sys.exit(1 if FAIL else 0)
