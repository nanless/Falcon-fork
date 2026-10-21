"""R1 verification for docs/understanding/04: audio_visual_test.py evaluation flow.
No real data / no real weights / no GPU: static + function-level + fabricated tiny ckpt/dataset.
Third-party metric packages and torchvision/cv2/matplotlib are stubbed (they are not installed).
Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_eval_flow_04.py
"""
import copy, csv, inspect, json, os, sys, tempfile, types, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
sys.path.insert(0, ".")

# ---------- stubs for missing third-party packages (with valid __spec__ for importlib) ----------
import importlib.machinery
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
    mod = make_pkg(name)
    target = mod
    if sub:
        submod = make_pkg(f"{name}.{sub}", is_pkg=False)
        setattr(mod, sub, submod)
        target = submod
    for k, v in attrs.items():
        setattr(target, k, v)
# pytorch_lightning 只为 look2hear/utils/lightning_utils.py 的两个 import 而存在（本脚本不训练）
pl = make_pkg("pytorch_lightning")
pl_ut = make_pkg("pytorch_lightning.utilities"); pl.utilities = pl_ut
pl_ut.rank_zero_only = lambda fn: fn
pl_cb = make_pkg("pytorch_lightning.callbacks"); pl.callbacks = pl_cb
pl_prog = make_pkg("pytorch_lightning.callbacks.progress"); pl_cb.progress = pl_prog
pl_rp = make_pkg("pytorch_lightning.callbacks.progress.rich_progress", is_pkg=False)
class _DummyRichProgressBar:  # noqa: D401 - 仅用于 import
    def __init__(self, *a, **k): pass
pl_rp.RichProgressBar = _DummyRichProgressBar
pl_prog.rich_progress = pl_rp
sys.modules["matplotlib"].pyplot = make_pkg("matplotlib.pyplot", is_pkg=False)
fbe = types.ModuleType("fast_bss_eval")
fbe.sdr_pit_loss = lambda est, ref, **k: torch.zeros(est.shape[-2])
sys.modules["fast_bss_eval"] = fbe
sys.modules["torchvision"].transforms.Grayscale = lambda num_output_channels=1: (lambda x: x)

import yaml
import importlib.util
spec = importlib.util.spec_from_file_location("avt", "audio_visual_test.py")
avt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avt)
conf_falcon = yaml.safe_load(open("configs/Falcon.yml"))
conf_snap = yaml.safe_load(open("Experiments/checkpoint/Falcon/conf.yml"))
print("=" * 78)

print("[1] _find_best_ckpt 顺序验证")
print("  1a) 仓库自带的 Experiments/checkpoint/Falcon（只有 conf.yml + best_k_models.json，${ROOT} 路径失效）:")
try:
    p = avt._find_best_ckpt("Experiments/checkpoint/Falcon")
    print("      ->", p)
except FileNotFoundError as e:
    print("      -> FileNotFoundError:", str(e)[:110])
print("      best_k_models.json 内容路径:", list(json.load(open("Experiments/checkpoint/Falcon/best_k_models.json")).keys()))
print("      os.path.exists(第一条) =", os.path.exists("${ROOT}/Falcon/epoch=714.ckpt"))
with tempfile.TemporaryDirectory() as td:
    def touch(name):
        open(os.path.join(td, name), "w").close()
    abs_paths = {os.path.join(td, n): sc for n, sc in (("a.ckpt", -9.98), ("b.ckpt", -9.90), ("c.ckpt", -10.01))}
    json.dump(abs_paths, open(os.path.join(td, "best_k_models.json"), "w"))
    for n in ("a.ckpt", "b.ckpt", "c.ckpt", "last.ckpt", "epoch=742.ckpt", "epoch=1000.ckpt"):
        touch(n)
    print("  1b) best_k json 中三条都存在 -> 选 min(val_loss)（-10.01 的 c.ckpt）:", os.path.basename(avt._find_best_ckpt(td)))
    os.remove(os.path.join(td, "best_k_models.json"))
    print("  1c) 无 best_k json -> 选 last.ckpt:", os.path.basename(avt._find_best_ckpt(td)))
    os.remove(os.path.join(td, "last.ckpt"))
    got = os.path.basename(avt._find_best_ckpt(td))
    print(f"  1d) 只有 epoch*.ckpt -> 返回 {got}；但字符串排序结果 = {sorted(['epoch=742.ckpt','epoch=1000.ckpt'])[-1]}"
          "（>=4 位 epoch 时会选错）")

print("\n[2] conf.yml 的 audionet_name 与 look2hear.models 不匹配")
import look2hear.models, look2hear.videomodels
print("  Experiments/checkpoint/Falcon/conf.yml audionet_name =", conf_snap["audionet"]["audionet_name"])
print("  configs/Falcon.yml               audionet_name =", conf_falcon["audionet"]["audionet_name"])
for name in (conf_snap["audionet"]["audionet_name"], conf_falcon["audionet"]["audionet_name"]):
    print(f"  getattr(look2hear.models, {name!r}) ->",
          "OK" if hasattr(look2hear.models, name) else "AttributeError（类不存在）")
print("  video net: getattr(look2hear.videomodels, 'FRCNNVideoModel') ->", hasattr(look2hear.videomodels, "FRCNNVideoModel"))

print("\n[3] 构建音频模型是否会污染 config（Separator 的 dict.pop）")
aud_cfg = copy.deepcopy(conf_falcon["audionet"]["audionet_config"])
model = getattr(look2hear.models, "FalconSpeakerContrastive")(sample_rate=16000, **aud_cfg)
n_audio = sum(p.numel() for p in model.parameters())
print(f"  FalconSpeakerContrastive 参数 = {n_audio:,}")
print(f"  构建后 config['audio_params'] 仍有 'repeats'? {'repeats' in aud_cfg['audio_params']}; "
      f"仍有 'shared'? {'shared' in aud_cfg['audio_params']}; 'video_params' 仍有 'repeats'? {'repeats' in aud_cfg['video_params']}")
# 模拟“被 pop 污染后的 conf.yml”再重建，观察退化
poll_cfg = copy.deepcopy(conf_falcon["audionet"]["audionet_config"])
poll_cfg["audio_params"].pop("repeats", None); poll_cfg["audio_params"].pop("shared", None)
poll_cfg["video_params"].pop("repeats", None); poll_cfg["video_params"].pop("shared", None)
m_poll = getattr(look2hear.models, "FalconSpeakerContrastive")(sample_rate=16000, **poll_cfg)
print(f"  用“缺 repeats/shared 的 conf”重建 -> 参数 {sum(p.numel() for p in m_poll.parameters()):,} "
      f"(远小于 {n_audio:,}，Separator 层数为 0)；此时 ckpt 里大部分权重会被 strict=False 静默丢弃")

print("\n[4] _load_models_from_ckpt：用伪造 ckpt 做函数级回环")
video_cfg = copy.deepcopy(conf_falcon["videonet"]["videonet_config"])
video_cfg.pop("pretrain", None)
video = getattr(look2hear.videomodels, "FRCNNVideoModel")(**video_cfg)
sd = {f"audio_model.{k}": v for k, v in model.state_dict().items()}
sd.update({f"video_model.{k}": v for k, v in video.state_dict().items()})
print(f"  ckpt keys: audio_model.*={sum(k.startswith('audio_model.') for k in sd)}, "
      f"video_model.*={sum(k.startswith('video_model.') for k in sd)}")
cfg_for_loader = {"train_conf": conf_falcon}
with tempfile.TemporaryDirectory() as td:
    ck = os.path.join(td, "fake.ckpt")
    torch.save({"state_dict": sd, "epoch": 1}, ck)
    m2, v2 = avt._load_models_from_ckpt(ck, cfg_for_loader, "cpu")
    same = all(torch.allclose(a, b) for a, b in zip(model.state_dict().values(), m2.state_dict().values()))
    print(f"  Lightning 风格 {{'state_dict': ...}} -> 加载成功, 参数与源模型一致={same}")
    print(f"  torch.load(weights_only=False) 见 audio_visual_test.py:119（Studio 用 weights_only=True）")
    # 裸 state_dict（无 "state_dict" 键）-> _load_models_from_ckpt 是否会失败？
    ck2 = os.path.join(td, "bare.pt")
    torch.save(sd, ck2)
    try:
        avt._load_models_from_ckpt(ck2, cfg_for_loader, "cpu")
        print("  裸 state_dict 文件 -> 也能加载")
    except Exception as e:
        print(f"  裸 state_dict 文件 -> {type(e).__name__}: {e} (falcon.pt 若无 state_dict 包装则 HF 评测命令失败)")

print("\n[5] 用带 SpeakerContrastiveV2 的 shipped conf.yml 走 _load_models_from_ckpt")
dummy_ck = os.path.join(tempfile.gettempdir(), "04_dummy.ckpt")
torch.save({"state_dict": {}}, dummy_ck)
try:
    avt._load_models_from_ckpt(dummy_ck, {"train_conf": conf_snap}, "cpu")
except Exception as e:
    print(f"  -> {type(e).__name__}: {e}（torch.load 成功后失败于模型类名 getattr）")

print("\n[6] 评测 DataModule 参数兼容性（process_worker 的 data_cfg 过滤后）")
from look2hear.datas.avspeech_dataset_near import AVSpeechDataModuleWithNear
sig = set(inspect.signature(AVSpeechDataModuleWithNear.__init__).parameters) - {"self"}
for label, c in (("configs/Falcon.yml", conf_falcon), ("shipped conf.yml", conf_snap)):
    dc = dict(c["datamodule"]["data_config"])
    filt = {k: v for k, v in dc.items() if k not in ("use_near", "train_samples")}
    invalid = [k for k in filt if k not in sig]
    missing = [k for k in ("train_dir", "valid_dir", "test_dir") if k not in filt]
    print(f"  {label}: 过滤后非法键={invalid or '无'}, 缺必需键={missing or '无'}, data_name={c['datamodule'].get('data_name')}")
print("  shipped conf.yml 的数据路径仍为未展开占位符:", conf_snap["datamodule"]["data_config"]["test_dir"])

print("\n[7] hf:// 解析与结果 CSV 路径")
ck = "hf://JusperLee/Falcon/falcon.pt"
repo_id, _, filename = ck[len("hf://"):].rpartition("/")
print(f"  {ck!r} -> repo_id={repo_id!r}, filename={filename!r} (audio_visual_test.py:326)")
exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", conf_falcon["exp"]["exp_name"])
print("  exp_dir =", exp_dir)
print("  每卡 CSV =", os.path.join(exp_dir, "results", "metrics_rank{rank}.csv"))
print("  合并 CSV =", os.path.join(exp_dir, "results", "metrics_combined.csv"))

print("\n[8] 评估数据集：伪造 1 s 样本验证 96x96 -> 88x88 CenterCrop 发生在 dataset 预处理（不在 FRCNN 内）")
import soundfile as sf
from look2hear.datas.avspeech_dataset import AVSpeechEvalDataset
with tempfile.TemporaryDirectory() as td:
    sr, L = 16000, 16000
    mixp, tgtp = os.path.join(td, "mix.wav"), os.path.join(td, "tgt.wav")
    # 立体声：AVSpeechEvalDataset 对音频无条件 .mean(-1)，单声道会被压成 0 维（见下方专项探针）
    m2 = (0.1 * np.random.randn(L)).astype(np.float32)
    t2 = (0.05 * np.random.randn(L)).astype(np.float32)
    sf.write(mixp, np.stack([m2, m2], axis=1), sr)
    sf.write(tgtp, np.stack([t2, t2], axis=1), sr)
    T = 25
    frames = (np.random.rand(T, 96, 96, 1) * 255).astype(np.uint8)
    mouthp = os.path.join(td, "mouth.pt")
    torch.save(torch.from_numpy(frames), mouthp)
    json.dump([[mixp, L]], open(os.path.join(td, "mix.json"), "w"))
    json.dump([[tgtp, mouthp]], open(os.path.join(td, "target.json"), "w"))
    ds = AVSpeechEvalDataset(json_dir=td, sample_rate=sr)
    mix, src, mouth, key, n_near = ds[0]
    print(f"  __getitem__ -> mix{tuple(mix.shape)}, src{tuple(src.shape)}, mouth{tuple(mouth.shape)} {mouth.dtype}, key={key}, n_near={n_near}")
    exp_mouth = ((frames[..., 0].astype(np.float32) - 0.0) / 255.0)[:, 4:92, 4:92]
    exp_mouth = (exp_mouth - 0.421) / 0.165
    print(f"  mouth == CenterCrop(88) after Normalize/255 与均值方差归一化: max|diff|={np.abs(mouth - exp_mouth).max():.3e}")
    print(f"  评估使用整段音频（{mix.shape[-1]} samples = {mix.shape[-1]/sr:.2f}s），不是 2 s 训练片段；嘴部 {T} 帧 = 25 fps")
    # 单声道探针：AVSpeechEvalDataset 无条件 mean(-1)
    monod = os.path.join(td, "mono"); os.makedirs(monod, exist_ok=True)
    sf.write(os.path.join(monod, "m.wav"), m2, sr)
    json.dump([[os.path.join(monod, "m.wav"), L]], open(os.path.join(monod, "mix.json"), "w"))
    json.dump([[os.path.join(monod, "m.wav"), mouthp]], open(os.path.join(monod, "target.json"), "w"))
    dsm = AVSpeechEvalDataset(json_dir=monod, sample_rate=sr)
    mm, ss, mo, kk, nn = dsm[0]
    print(f"  单声道 wav 探针: mix.shape={tuple(mm.shape)}, src.shape={tuple(ss.shape)} "
          f"-> 被 .mean(-1) 压成 0 维（训练集 avspeech_dataset_near.py:186 有 ndim>1 保护，评测集没有；依赖真实数据是多声道）")

    print("\n[9] 复刻 process_worker 单样本前向（真实模型 + 伪造数据 + stub 指标）")
    from look2hear.utils import tensors_to_device
    from look2hear.metrics import ALLMetricsTracker
    vmodel = getattr(look2hear.videomodels, "FRCNNVideoModel")(**video_cfg)
    print(f"  FRCNNVideoModel().eval() 的返回值 = {vmodel.eval()!r}（train() 未 return self -> 链式 .eval() 会得到 None）")
    vmodel = vmodel
    amodel = getattr(look2hear.models, "FalconSpeakerContrastive")(sample_rate=16000, **copy.deepcopy(conf_falcon["audionet"]["audionet_config"])).eval()
    dev = torch.device("cpu")
    mix, sources, mouth, key, _ = tensors_to_device(ds[0], device=dev)
    t0 = __import__("time").time()
    with torch.no_grad():
        mouth_emb = vmodel(torch.from_numpy(mouth[None, None]).float().to(dev))
        est_sources = amodel(mix[None], mouth_emb)
    print(f"  mouth_emb {tuple(mouth_emb.shape)}; est_sources {tuple(est_sources.shape)}; 前向耗时 {__import__('time').time()-t0:.2f}s (CPU)")
    met = ALLMetricsTracker(save_file=os.path.join(tempfile.gettempdir(), "04_eval_metrics.csv"))
    met(mix=mix, clean=sources.unsqueeze(0), estimate=est_sources.squeeze(0), key=key)
    met.final()
    rows = list(csv.DictReader(open(os.path.join(tempfile.gettempdir(), "04_eval_metrics.csv"))))
    print("  metrics CSV 列/首行:", list(rows[0].keys()), "|", {k: (round(float(v), 3) if k != "snt_id" else v) for k, v in rows[0].items()})
print("\nDONE")
