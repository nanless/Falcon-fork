"""R4 incremental re-review of 01-model-core R3 (sha afe00188...) — independent verification.

Checks, using the now-locally-cached real weights (/root/.cache/falcon_weights/, Lead-provided; no network):
  [W] sha256 + falcon.pt / falcon-big.pt structure: keys, architecture, epoch,
      state_dict count/keys/shapes vs models, strict load, params vs state_dict numel
  [S] Studio FalconRuntime local-path load of the REAL weights (strict, weights_only=True) and
      real-weight CPU forward bit-identical vs the training-code model (closes PROVENANCE claim)
  [H] AudioBlock hook counts / concat[2] idle params / 15 state_dict keys / effective params
  [F] A18 first-changed-frame formula f_first = floor((s-win/2)/hop)+1 at s=16000/20000/16128
  [A21] InjectionMultiSum pulse response: as-built (0,250), de-cLNhw [98,102], causal=True+de-norm [100,104]
  [D] studio vs look2hear Falcon.py diff stat (net +9)

Run (repo root):
  PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python docs/verification/scripts/r4_incremental_01_r3.py
"""
import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))
sys.path.insert(0, str(REPO / "studio"))

import look2hear.models as models  # noqa: E402
from look2hear.layers.normalizations import LayerNormalizationHW  # noqa: E402
from look2hear.models.Falcon import InjectionMultiSum  # noqa: E402
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel  # noqa: E402

W_SMALL = Path("/root/.cache/falcon_weights/falcon.pt")
W_BIG = Path("/root/.cache/falcon_weights/falcon-big.pt")
cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
acfg = cfg["audionet"]["audionet_config"]
vcfg = copy.deepcopy(cfg["videonet"]["videonet_config"]); vcfg["pretrain"] = None
bigcfg = yaml.safe_load((REPO / "studio" / "configs" / "FalconBig.yml").read_text())["audionet"]["audionet_config"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numel(state):
    return sum(v.numel() for v in state.values() if torch.is_tensor(v))


print("=== [W0] weight files ===")
for p in (W_SMALL, W_BIG):
    print(f"  {p}: {p.stat().st_size:,} B sha256={sha256(p)}")

print("\n=== [W1] falcon.pt structure ===")
ck = torch.load(W_SMALL, map_location="cpu", weights_only=False)
print("  top-level keys:", sorted(ck.keys()))
print("  architecture:", ck.get("architecture"), "| epoch:", ck.get("epoch"))
sd = ck["state_dict"]
audio_sd = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
video_sd = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
print(f"  state_dict={len(sd)} = audio {len(audio_sd)} + video {len(video_sd)}"
      f" | mdcl_window in state_dict: {'mdcl_window' in sd}")
print(f"  state_dict numel: audio={numel(audio_sd):,} video={numel(video_sd):,} total={numel(audio_sd)+numel(video_sd):,}")

audio = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg))
video = FRCNNVideoModel(**vcfg)
print(f"  model params: audio={sum(p.numel() for p in audio.parameters()):,}"
      f" video={sum(p.numel() for p in video.parameters()):,}"
      f" total={sum(p.numel() for p in audio.parameters())+sum(p.numel() for p in video.parameters()):,}")

for name, model in (("audio", audio), ("video", video)):
    msd = model.state_dict()
    ref = audio_sd if name == "audio" else video_sd
    same_keys = set(msd) == set(ref)
    shape_bad = [k for k in msd if k in ref and msd[k].shape != ref[k].shape]
    print(f"  {name}: keys_equal={same_keys} missing={sorted(set(msd)-set(ref))[:3]} extra={sorted(set(ref)-set(msd))[:3]}"
          f" shape_mismatch={len(shape_bad)}")
    model.load_state_dict(ref, strict=True)
print(f"  strict load OK; rebuilt params: audio={sum(p.numel() for p in audio.parameters()):,}"
      f" video={sum(p.numel() for p in video.parameters()):,}")

print("\n=== [W4] falcon-big.pt structure ===")
cb = torch.load(W_BIG, map_location="cpu", weights_only=False)
sdb = cb["state_dict"]
a_big = {k[len("audio_model."):]: v for k, v in sdb.items() if k.startswith("audio_model.")}
v_big = {k[len("video_model."):]: v for k, v in sdb.items() if k.startswith("video_model.")}
print(f"  keys={sorted(cb.keys())} architecture={cb.get('architecture')} epoch={cb.get('epoch')}")
print(f"  state_dict={len(sdb)} = audio {len(a_big)} + video {len(v_big)} | audio numel={numel(a_big):,}")
from look2hear.models.Falcon import Falcon  # noqa: E402
big_cls = __import__("studio.falcon_model.models.Speaker_Contrastive_v2", fromlist=["SpeakerContrastiveV2"]).SpeakerContrastiveV2
big = big_cls(sample_rate=16000, **copy.deepcopy(bigcfg))
msd = big.state_dict()
print(f"  big audio keys_equal={set(msd)==set(a_big)} shape_mismatch={sum(1 for k in msd if k in a_big and msd[k].shape!=a_big[k].shape)}"
      f" params={sum(p.numel() for p in big.parameters()):,}")
big.load_state_dict(a_big, strict=True)
print("  big audio strict load OK")

print("\n=== [S] Studio FalconRuntime local-path load of REAL weights ===")
from falcon_runtime import FalconRuntime  # noqa: E402
rt = FalconRuntime("small", W_SMALL).load("cpu")
st = rt.status()
print(f"  small ready={st['ready']} device={st['device']} audio_keys={st['audio_weight_keys']}"
      f" video_keys={st['video_weight_keys']} params={st['parameters']:,} warmup_ms={st['warmup_ms']}")
print(f"  checkpoint reported: {st['checkpoint']}")

# real-weight bit-identical forward: Studio runtime vs training-code model (CPU, eval)
rng = np.random.default_rng(20261021)
audio_in = (rng.standard_normal(32000) * 0.05).astype(np.float32)
frames_in = rng.integers(0, 256, (50, 96, 96), dtype=np.uint8)
out_studio = rt.infer(audio_in, frames_in)

train_audio = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg))
train_video = FRCNNVideoModel(**vcfg)
train_audio.load_state_dict(audio_sd, strict=True)
train_video.load_state_dict(video_sd, strict=True)
train_audio.eval(); train_video.eval()
frames_t = torch.from_numpy(frames_in[:, 4:92, 4:92].astype(np.float32) / 255.0)
frames_t = ((frames_t - 0.421) / 0.165)[None, None]
with torch.inference_mode():
    emb = train_video(frames_t)
    ref = train_audio(torch.from_numpy(audio_in).unsqueeze(0), emb)[0, 0].numpy()
print(f"  REAL-WEIGHT CPU forward: studio vs training  max|d|={np.abs(out_studio - ref).max():.3e}"
      f" bit-identical={np.array_equal(out_studio, ref)}")

rt_big = FalconRuntime("big", W_BIG).load("cpu")
stb = rt_big.status()
print(f"  big ready={stb['ready']} audio_keys={stb['audio_weight_keys']} video_keys={stb['video_weight_keys']}"
      f" params={stb['parameters']:,} warmup_ms={stb['warmup_ms']}")
out_big = rt_big.infer(audio_in, frames_in)
print(f"  big infer -> shape={out_big.shape} finite={bool(np.isfinite(out_big).all())}")

print("\n=== [H] AudioBlock hooks / concat[2] idle ===")
m = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg)).eval()
ab = m.separator.audio_net
counts = {}
hs = []
for i, layer in enumerate(ab.concat_layers):
    hs.append(layer.register_forward_hook(lambda mod, inp, out, i=i: counts.__setitem__(f"concat[{i}]", counts.get(f"concat[{i}]", 0) + 1)))
for i, layer in enumerate(ab.fusion_layers):
    hs.append(layer.register_forward_hook(lambda mod, inp, out, i=i: counts.__setitem__(f"fusion[{i}]", counts.get(f"fusion[{i}]", 0) + 1)))
with torch.inference_mode():
    m(torch.randn(1, 32000), torch.randn(1, 512, 50))
for h in hs:
    h.remove()
print("  hook counts:", dict(sorted(counts.items())))
concat2_keys = [k for k in m.state_dict() if "concat_layers.2" in k]
idle = sum(p.numel() for p in ab.concat_layers[2].parameters())
total = sum(p.numel() for p in m.parameters())
print(f"  concat[2] params={idle:,} state_dict_keys={len(concat2_keys)} total={total:,}"
      f" effective={total - idle:,} idle_ratio={idle / total:.4%}")

print("\n=== [F] A18 first-changed-frame formula ===")
enc = m.encoder.eval()
base = torch.randn(1, 32000)
for s in (16000, 20000, 16128):
    mod = base.clone()
    mod[..., s:] = torch.randn(1, 32000 - s)
    with torch.inference_mode():
        d = (enc(base) - enc(mod)).abs().amax(dim=(0, 1, 3))
    nz = (d > 0).nonzero().flatten()
    first = int(nz[0]) if len(nz) else None
    formula = (s - 128) // 128 + 1
    print(f"  s={s}: first_changed_frame={first} floor((s-128)/128)+1={formula} match={first == formula}")

print("\n=== [A21] InjectionMultiSum pulse response ===")
def pulse_span(layer):
    x = torch.zeros(1, 64, 251, 129)
    x[0, :, 100, :] = 1.0
    with torch.inference_mode():
        y = layer(x, x)
    d = (y - layer(torch.zeros_like(x), torch.zeros_like(x))).abs().amax(dim=(0, 1, 3))
    idx = (d > 1e-12).nonzero().flatten()
    return (int(idx[0]), int(idx[-1])) if len(idx) else None

def strip_norms(layer):
    for name, mod in list(layer.named_modules()):
        if isinstance(mod, LayerNormalizationHW):
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], torch.nn.Identity())
    layer.eval()
    return layer

as_built = InjectionMultiSum(in_chan=64, kernel_size=5, norm_type="cLNhw", is2d=True, causal=False).eval()
print("  A21a as-built pulse span:", pulse_span(as_built))
de_norm = strip_norms(InjectionMultiSum(in_chan=64, kernel_size=5, norm_type="cLNhw", is2d=True, causal=False))
print("  A21b de-cLNhw pulse span:", pulse_span(de_norm))
causal_norm = strip_norms(InjectionMultiSum(in_chan=64, kernel_size=5, norm_type="cLNhw", is2d=True, causal=True))
print("  A21c causal=True+de-norm pulse span:", pulse_span(causal_norm))

print("\n=== [D] studio vs look2hear Falcon.py diff stat ===")
diff = subprocess.run(["diff", "-u", str(REPO / "look2hear/models/Falcon.py"),
                       str(REPO / "studio/falcon_model/models/Falcon.py")],
                      capture_output=True, text=True).stdout.splitlines()
added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
print(f"  added={added} removed={removed} net(studio - look2hear)={added - removed}")

print("\nOK: incremental R3 verification finished")
