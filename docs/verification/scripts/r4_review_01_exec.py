"""R4 execution-level re-verification for `docs/understanding/01-model-core.md` (reviewer: studio-backend).

Independent of 01's own §8 scripts. Covers the claims a reviewer must check:
  - parameter counts (module-level, 0.5M/11.72M claims)
  - AudioBlock call counts (concat_layers[2] unused? fusion/concat/process layers)
  - causality: STFT natural lookahead (encoder frames), full-model leakage,
    cLNhw-removed structural lookahead, top-level `causal` switch inertness
  - forward shape chain, dead MDCL path

Run (repo root):
  PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python docs/verification/scripts/r4_review_01_exec.py
"""
import copy
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

import look2hear.models as models  # noqa: E402
from look2hear.models.Falcon import Falcon  # noqa: E402
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel  # noqa: E402
from look2hear.layers.normalizations import LayerNormalizationHW  # noqa: E402

torch.manual_seed(0)
cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
acfg = cfg["audionet"]["audionet_config"]
vcfg = copy.deepcopy(cfg["videonet"]["videonet_config"])
vcfg["pretrain"] = None

print("=== 1. parameter counts ===")
audio = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg))
video = FRCNNVideoModel(**vcfg)
n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
print(f"audio total={n(audio):,} video={n(video):,} total={n(audio)+n(video):,}")
for name in ("encoder", "audio_bottleneck", "separator", "mask_generator", "decoder", "spk_pool_proj"):
    print(f"  {name:18s} {n(getattr(audio, name)):>10,}")
print(f"  separator.audio_net {n(audio.separator.audio_net):>10,}")
print(f"  separator.video_net {n(audio.separator.video_net):>10,}")
print(f"  separator.caf       {n(audio.separator.caf):>10,}")

print("\n=== 2. AudioBlock call counts (one Separator.forward) ===")
counts = {}

def hook(name):
    def fn(module, inputs, output):
        counts[name] = counts.get(name, 0) + 1
    return fn

handles = []
ab = audio.separator.audio_net
for i, layer in enumerate(ab.concat_layers):
    handles.append(layer.register_forward_hook(hook(f"concat[{i}]")))
for i, layer in enumerate(ab.fusion_layers):
    handles.append(layer.register_forward_hook(hook(f"fusion[{i}]")))
for i, layer in enumerate(ab.process_f_layers):
    handles.append(layer.register_forward_hook(hook(f"process_f[{i}]")))
handles.append(audio.separator.register_forward_hook(hook("separator")))
for i, layer in enumerate(ab.downsample_layers):
    handles.append(layer.register_forward_hook(hook(f"downsample[{i}]")))
audio.eval()
with torch.inference_mode():
    wav = torch.randn(1, 32000)
    mouth_emb = torch.randn(1, 512, 50)
    audio(wav, mouth_emb)
for h in handles:
    h.remove()
print("  ", dict(sorted(counts.items())))

print("\n=== 3. causality probes ===")
enc = audio.encoder.eval()
audio_in = torch.randn(1, 32000)
mod = audio_in.clone()
mod[..., 16000:] = torch.randn(1, 16000)
with torch.inference_mode():
    f1, f2 = enc(audio_in), enc(mod)          # [1,112,251,129]
d = (f1 - f2).abs().amax(dim=(0, 1, 3))       # per-frame max delta
changed = (d > 0).nonzero().flatten().tolist()
print(f"  encoder frames 121-126 delta={[float(d[i]) for i in range(121, 127)]}")
print(f"  encoder first changed frame={changed[0] if changed else None} (expected 125 = ceil((16000-128)/128))")

full = audio.eval()
with torch.inference_mode():
    o1 = full(audio_in, mouth_emb)[0, 0]
    o2 = full(mod, mouth_emb)[0, 0]
od = (o1 - o2).abs()
for start in range(0, 32000, 4000):
    seg = od[start:start + 4000]
    first = int((seg > 0).nonzero()[0]) + start if (seg > 0).any() else None
    print(f"  output[{start:5d}:{start+4000:5d}] max|d|={float(seg.max()):.3e} first_nonzero={first}")

print("\n=== 3b. cLNhw removed -> structural lookahead ===")
model2 = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg))
for name, module in list(model2.named_modules()):
    if isinstance(module, LayerNormalizationHW):
        parts = name.split(".")
        parent = model2
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], torch.nn.Identity())
model2.eval()

def prefix_lookahead(model, sample=20000):
    a = torch.randn(1, 32000)
    b = a.clone()
    b[..., sample:] = torch.randn(1, 32000 - sample)
    with torch.inference_mode():
        y1 = model(a, torch.zeros(1, 512, 50))[0, 0]
        y2 = model(b, torch.zeros(1, 512, 50))[0, 0]
    diff = (y1 - y2).abs()
    nz = (diff > 0).nonzero()
    return int(nz[0]) if len(nz) else None, float(diff.max())

first_full, mx_full = prefix_lookahead(full)
first_wo, mx_wo = prefix_lookahead(model2)
print(f"  as-built (with cLNhw): first changed output sample={first_full} max|d|={mx_full:.3e}")
print(f"  cLNhw -> Identity    : first changed output sample={first_wo} max|d|={mx_wo:.3e}"
      f"  (lookahead={None if first_wo is None else 20000-first_wo} samples)")

print("\n=== 3b2. AudioBlock-only structural lookahead (01 claims 57 frames) ===")
ab2 = models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(acfg)).separator.audio_net
for name, module in list(ab2.named_modules()):
    if isinstance(module, LayerNormalizationHW):
        parts = name.split(".")
        parent = ab2
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], torch.nn.Identity())
ab2.eval()
with torch.inference_mode():
    x = torch.randn(1, 112, 251, 129)
    y = x.clone()
    y[:, :, 125:] = torch.randn(1, 112, 126, 129)
    o1, o2 = ab2(x), ab2(y)
    d = (o1 - o2).abs()
    nz = (d > 0).nonzero()
    first = int(nz[0][2]) if len(nz) else None
    print(f"  AudioBlock(cLNhw->Identity): first changed frame={first} (perturbed frame>=125)"
          f" -> lookahead={None if first is None else 125-first} frames"
          f" max|d|={float(d.max()):.3e}")

print("\n=== 3c. top-level `causal` switch inertness ===")
cfg_on, cfg_off = copy.deepcopy(acfg), copy.deepcopy(acfg)
cfg_on["causal"], cfg_off["causal"] = True, False
m_on = models.FalconSpeakerContrastive(sample_rate=16000, **cfg_on).eval()
m_off = models.FalconSpeakerContrastive(sample_rate=16000, **cfg_off).eval()
m_off.load_state_dict(m_on.state_dict())
with torch.inference_mode():
    yon = m_on(wav, mouth_emb)
    yoff = m_off(wav, mouth_emb)
print(f"  causal=True vs False: state_dict keys equal={set(m_on.state_dict()) == set(m_off.state_dict())}"
      f" params {n(m_on):,}/{n(m_off):,} bit-identical={torch.equal(yon, yoff)}"
      f" max|d|={float((yon-yoff).abs().max()):.3e}")

print("\n=== 4. dead MDCL path / constants ===")
print(f"  Falcon._mdcl_loss_hard={hasattr(Falcon, '_mdcl_loss_hard')} Falcon._mdcl_loss={hasattr(Falcon, '_mdcl_loss')}")
try:
    base = Falcon(sample_rate=16000, **copy.deepcopy({k: v for k, v in acfg.items() if k != 'spk_embed_dim'}))
    base.train()
    base(torch.randn(1, 32000), torch.randn(1, 512, 50), near_audio=[torch.randn(1, 32000)])
    print("  Falcon(train, near_audio) -> NO ERROR (unexpected)")
except AttributeError as exc:
    print(f"  Falcon(train, near_audio) -> AttributeError: {exc}")

print("\n=== 5. forward shape chain (hooks) ===")
shapes = {}
def rec(name):
    def fn(m, inputs, output):
        if name not in shapes:
            shapes[name] = tuple(output.shape)
        # must return None so the hook does not replace the module output
    return fn
targets = {
    "encoder": audio.encoder, "audio_bottleneck": audio.audio_bottleneck,
    "video_bottleneck": audio.video_bottleneck, "separator": audio.separator,
    "mask_generator": audio.mask_generator, "decoder": audio.decoder,
    "spk_pool_proj": audio.spk_pool_proj,
}
hs = [m.register_forward_hook(rec(k)) for k, m in targets.items()]
emb = torch.randn(1, 512, 50)
with torch.inference_mode():
    audio(wav, emb)
for h in hs:
    h.remove()
for k in ("encoder", "audio_bottleneck", "video_bottleneck", "separator",
          "mask_generator", "decoder", "spk_pool_proj"):
    print(f"  {k:18s} {shapes.get(k)}")

print("\nOK: r4 review execution checks finished")
