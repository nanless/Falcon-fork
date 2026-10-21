"""R3 真实权重检查（Lead 已下载 /root/.cache/falcon_weights/{falcon.pt,falcon-big.pt}）：
- falcon.pt 顶层结构/键/前缀数量；与本地 look2hear 模型 state_dict 的 key 集合与形状逐项对比，strict=True 回环加载；
- 走 audio_visual_test._load_models_from_ckpt 的真实函数路径（验证 Q1/B5 是否关闭）；
- falcon-big.pt 与 studio SpeakerContrastiveV2 的 key/形状对比与 strict 加载。
不训练、不联网、不改源码。Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r3_weights_04.py
"""
import copy, importlib.machinery, importlib.util, os, sys, types, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import torch
import yaml

WEIGHTS = "/root/.cache/falcon_weights"
print("=" * 78)

# ---------- stubs（audio_visual_test 的 import 链需要） ----------
def make_pkg(name, is_pkg=True):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_pkg)
    if is_pkg:
        mod.__path__ = []
    sys.modules[name] = mod
    return mod
for name, sub, attrs in [
    ("torch_mir_eval", "separation", {"bss_eval_sources": lambda *a, **k: (None,)}),
    ("torchvision", "transforms", {}),
    ("matplotlib", None, {}),
    ("cv2", None, {}),
    ("pystoi", None, {"stoi": lambda x, y, fs, extended=False: 0.9}),
    ("pypesq", None, {"pesq": lambda ref, deg, fs=16000, normalize=False: 3.0}),
]:
    mod = make_pkg(name); target = mod
    if sub:
        submod = make_pkg(f"{name}.{sub}", is_pkg=False); setattr(mod, sub, submod); target = submod
    for k, v in attrs.items():
        setattr(target, k, v)
sys.modules["matplotlib"].pyplot = make_pkg("matplotlib.pyplot", is_pkg=False)
fbe = types.ModuleType("fast_bss_eval")
fbe.sdr_pit_loss = lambda est, ref, **k: torch.zeros(est.shape[-2])
sys.modules["fast_bss_eval"] = fbe
sys.modules["torchvision"].transforms.Grayscale = lambda num_output_channels=1: (lambda x: x)
# pytorch_lightning 只需满足 look2hear/utils/lightning_utils.py 的 import
pl = make_pkg("pytorch_lightning"); pl_ut = make_pkg("pytorch_lightning.utilities"); pl.utilities = pl_ut
pl_ut.rank_zero_only = lambda fn: fn
pl_cb = make_pkg("pytorch_lightning.callbacks"); pl.callbacks = pl_cb
pl_prog = make_pkg("pytorch_lightning.callbacks.progress"); pl_cb.progress = pl_prog
pl_rp = make_pkg("pytorch_lightning.callbacks.progress.rich_progress", is_pkg=False)
class _DummyRichProgressBar:
    def __init__(self, *a, **k): pass
pl_rp.RichProgressBar = _DummyRichProgressBar; pl_prog.rich_progress = pl_rp

import look2hear.models, look2hear.videomodels
conf = yaml.safe_load(open("configs/Falcon.yml"))

def spec_info(sd):
    return len(sd), {k: tuple(v.shape) for k, v in sd.items()}

def compare_state(name, ref_sd, loaded_sd, prefix=""):
    ref = {k: v for k, v in ref_sd.items() if k.startswith(prefix)} if prefix else ref_sd
    lk = set(loaded_sd); rk = set(ref)
    only_l = sorted(lk - rk); only_r = sorted(rk - lk)
    shape_bad = [(k, tuple(ref[k].shape), tuple(loaded_sd[k].shape))
                 for k in lk & rk if tuple(ref[k].shape) != tuple(loaded_sd[k].shape)]
    print(f"  {name}: 参考 {len(rk)} 键 / ckpt {len(lk)} 键 | 仅 ckpt={len(only_l)} 仅参考={len(only_r)} 形状不符={len(shape_bad)}")
    for label, lst in (("only-in-ckpt", only_l[:3]), ("only-in-model", only_r[:3]), ("shape-bad", shape_bad[:2])):
        if lst: print(f"    {label}: {lst}")
    return not (only_l or only_r or shape_bad)

# ---------- 1) falcon.pt ----------
print("[1] falcon.pt（小模型 FalconSpeakerContrastive）")
ck = torch.load(os.path.join(WEIGHTS, "falcon.pt"), map_location="cpu", weights_only=False)
print(f"  top-level type={type(ck).__name__}; keys={sorted(ck.keys()) if isinstance(ck, dict) else None}")
print(f"  含 'state_dict'? {isinstance(ck, dict) and 'state_dict' in ck}")
for k in ck:
    if k != "state_dict":
        print(f"  meta[{k}] = {ck[k]!r}")
sd = ck["state_dict"]
audio = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
video = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
print(f"  state_dict 键数={len(sd)}; audio_model.*={len(audio)}, video_model.*={len(video)}, 其他前缀={len(sd)-len(audio)-len(video)}")
print(f"  audio 参数量={sum(v.numel() for v in audio.values()):,}; video 参数量={sum(v.numel() for v in video.values()):,}")
am = look2hear.models.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(conf["audionet"]["audionet_config"]))
vcfg = copy.deepcopy(conf["videonet"]["videonet_config"]); vcfg.pop("pretrain", None)
vm = look2hear.videomodels.FRCNNVideoModel(**vcfg)
ok_a = compare_state("audio_model", am.state_dict(), audio)
ok_v = compare_state("video_model", vm.state_dict(), video)
am.load_state_dict(audio, strict=True); vm.load_state_dict(video, strict=True)
print(f"  strict=True 回环加载: audio OK / video OK; 本地模型参数量 audio={sum(p.numel() for p in am.parameters()):,}, video={sum(p.numel() for p in vm.parameters()):,}")

# ---------- 2) 走 audio_visual_test._load_models_from_ckpt 真实函数 ----------
print("\n[2] audio_visual_test._load_models_from_ckpt(falcon.pt, configs/Falcon.yml) 真实函数路径")
spec = importlib.util.spec_from_file_location("avt", "audio_visual_test.py")
avt = importlib.util.module_from_spec(spec); spec.loader.exec_module(avt)
m2, v2 = avt._load_models_from_ckpt(os.path.join(WEIGHTS, "falcon.pt"),
                                    {"train_conf": conf}, "cpu")
print("  返回模型类型:", type(m2).__name__, "/", type(v2).__name__)
print(f"  参数量: {sum(p.numel() for p in m2.parameters()):,} / {sum(p.numel() for p in v2.parameters()):,}")

# ---------- 3) falcon-big.pt vs studio SpeakerContrastiveV2 ----------
print("\n[3] falcon-big.pt（Big 模型 SpeakerContrastiveV2，studio 代码）")
ckb = torch.load(os.path.join(WEIGHTS, "falcon-big.pt"), map_location="cpu", weights_only=False)
print(f"  top-level type={type(ckb).__name__}; keys={sorted(ckb.keys())}")
for k in ckb:
    if k != "state_dict":
        print(f"  meta[{k}] = {ckb[k]!r}")
sdb = ckb["state_dict"]
ab = {k[len("audio_model."):]: v for k, v in sdb.items() if k.startswith("audio_model.")}
vb = {k[len("video_model."):]: v for k, v in sdb.items() if k.startswith("video_model.")}
print(f"  state_dict 键数={len(sdb)}; audio_model.*={len(ab)}, video_model.*={len(vb)}, 其他={len(sdb)-len(ab)-len(vb)}")
print(f"  audio 参数量={sum(v.numel() for v in ab.values()):,}; video 参数量={sum(v.numel() for v in vb.values()):,}")
sys.path.insert(0, "studio")
bigconf = yaml.safe_load(open("studio/configs/FalconBig.yml"))
try:
    from falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2
    big = SpeakerContrastiveV2(sample_rate=16000, **copy.deepcopy(bigconf["audionet"]["audionet_config"]))
    ok_ab = compare_state("big audio_model", big.state_dict(), ab)
    bvcfg = copy.deepcopy(bigconf["videonet"]["videonet_config"]); bvcfg.pop("pretrain", None)
    bvm = look2hear.videomodels.FRCNNVideoModel(**bvcfg) if bvcfg.get("backbone_type") == "resnet" else None
    ok_vb = compare_state("big video_model", bvm.state_dict(), vb) if bvm is not None else None
    big.load_state_dict(ab, strict=True)
    if bvm is not None: bvm.load_state_dict(vb, strict=True)
    print(f"  strict=True 加载: audio OK / video {'OK' if bvm is not None else '未建'}; "
          f"本地 big audio 参数量={sum(p.numel() for p in big.parameters()):,}")
    print(f"  big 顶层键={'', }")
except Exception as e:
    import traceback
    print(f"  构建/对比 failed: {type(e).__name__}: {e}")
    traceback.print_exc(limit=3)
print("\nDONE")
