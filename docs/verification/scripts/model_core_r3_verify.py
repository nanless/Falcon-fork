"""model-core R3 验证：真实权重（falcon.pt / falcon-big.pt）+ concat[2] 闲置 + Studio 副本 diff + 探针分界敏感性。

运行：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_r3_verify.py

禁区：只 torch.load 读取，不训练、不改源码、不联网。
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "verification" / "stubs"))

CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
WEIGHTS = Path("/root/.cache/falcon_weights")

print("=" * 78)
print("[W1] 真实权重 falcon.pt 顶层结构")
print("=" * 78)
ck = torch.load(WEIGHTS / "falcon.pt", map_location="cpu", weights_only=False)
print("  top-level type:", type(ck).__name__)
print("  top-level keys:", list(ck.keys())[:8], "..." if len(ck) > 8 else "")
sd = ck.get("state_dict", ck)
print("  是否含 'state_dict' 包裹:", "state_dict" in ck,
      "| state_dict 条目数:", len(sd))
print("  非 state_dict 的字段:", {k: v for k, v in ck.items() if k != "state_dict"})
audio = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
video = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
other = [k for k in sd if not (k.startswith("audio_model.") or k.startswith("video_model."))]
print(f"  audio_model.* = {len(audio)} keys | video_model.* = {len(video)} keys | 其它 = {len(other)} {other[:3]}")
print("  mdcl_window 是否在 audio key 中:", any("mdcl_window" in k for k in audio))
print("  audio 参数（按权重形状求和）:", f"{sum(v.numel() for v in audio.values() if hasattr(v, 'numel')):,}")
print("  video 参数（按权重形状求和）:", f"{sum(v.numel() for v in video.values() if hasattr(v, 'numel')):,}")

print()
print("=" * 78)
print("[W2] 与 `FalconSpeakerContrastive(configs/Falcon.yml)` 的 key/shape 全等校验")
print("=" * 78)
import look2hear.models as models  # noqa: E402
m = getattr(models, CFG["audionet"]["audionet_name"])(
    sample_rate=16000, **copy.deepcopy(ACONF)).eval()
mkeys = set(m.state_dict().keys())
print("  model.state_dict() keys:", len(mkeys))
print("  key 集合相等:", mkeys == set(audio.keys()))
only_model = sorted(mkeys - set(audio))
only_ckpt = sorted(set(audio) - mkeys)
print("  仅模型有:", only_model[:5], "..." if len(only_model) > 5 else "")
print("  仅 ckpt 有:", only_ckpt[:5], "..." if len(only_ckpt) > 5 else "")
shape_bad = [(k, tuple(m.state_dict()[k].shape), tuple(audio[k].shape))
             for k in mkeys & set(audio) if tuple(m.state_dict()[k].shape) != tuple(audio[k].shape)]
print("  shape 不一致条目数:", len(shape_bad), shape_bad[:3])
missing, unexpected = m.load_state_dict(audio, strict=False)
print("  strict=False 载入 -> missing:", len(missing), "| unexpected:", len(unexpected))
m.load_state_dict(audio, strict=True)
print("  strict=True 载入 -> OK（与 Studio 的加载口径一致）")
print("  state_dict 载入后参数总量:", f"{sum(p.numel() for p in m.parameters()):,}")

print()
print("=" * 78)
print("[W3] 视频权重 FRCNNVideoModel 校验")
print("=" * 78)
from look2hear.videomodels import FRCNNVideoModel  # noqa: E402
vm = FRCNNVideoModel(pretrain=None)
vkeys = set(vm.state_dict().keys())
print("  video keys:", len(video), "| FRCNN keys:", len(vkeys),
      "| 集合相等:", vkeys == set(video))
bad = [k for k in vkeys & set(video) if tuple(vm.state_dict()[k].shape) != tuple(video[k].shape)]
print("  shape 不一致:", len(bad))
vm.load_state_dict(video, strict=True)
print("  video strict=True OK；参数:", f"{sum(p.numel() for p in vm.parameters()):,}")
print(f"  audio+video 合计 = {sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in vm.parameters()):,}")

print()
print("=" * 78)
print("[W4] falcon-big.pt 顶层结构（仅结构核对，模型类属 06 域）")
print("=" * 78)
ckb = torch.load(WEIGHTS / "falcon-big.pt", map_location="cpu", weights_only=False)
print("  top-level type:", type(ckb).__name__, "| keys:", list(ckb.keys())[:8])
sdb = ckb.get("state_dict", ckb)
a_b = {k[len("audio_model."):]: v for k, v in sdb.items() if k.startswith("audio_model.")}
v_b = {k[len("video_model."):]: v for k, v in sdb.items() if k.startswith("video_model.")}
print(f"  audio_model.* = {len(a_b)} keys | video_model.* = {len(v_b)} keys")
print("  audio 参数:", f"{sum(v.numel() for v in a_b.values() if hasattr(v, 'numel')):,}",
      "| video 参数:", f"{sum(v.numel() for v in v_b.values() if hasattr(v, 'numel')):,}")
print("  非 state_dict 字段:", {k: v for k, v in ckb.items() if k != "state_dict"})
print("  audio key 前缀样例:", list(a_b)[:3])

print()
print("=" * 78)
print("[W5] Studio 副本 vs 训练侧 Falcon.py：结构化 diff 统计")
print("=" * 78)
d = subprocess.run(["diff", "studio/falcon_model/models/Falcon.py", "look2hear/models/Falcon.py"],
                   cwd=REPO, capture_output=True, text=True).stdout.splitlines()
add = [ln[2:] for ln in d if ln.startswith("> ")]
rem = [ln[2:] for ln in d if ln.startswith("< ")]
print(f"  diff 总行数 {len(d)} | 新增 {len(add)} | 删除 {len(rem)} | 净 {len(add) - len(rem):+d}")
code_add = [x for x in add if x.strip() and not x.strip().startswith("#")]
code_rem = [x for x in rem if x.strip() and not x.strip().startswith("#")]
print(f"  去掉注释/空白后：新增 {len(code_add)} 行、删除 {len(code_rem)} 行")
print("  新增代码行样例:", code_add[:6])
print("  删除代码行样例:", code_rem[:4])

print()
print("=" * 78)
print("[W6] concat_layers[2] 闲置：hook 计数 + 参数")
print("=" * 78)
ab = m.separator.audio_net
cc = {i: 0 for i in range(4)}
fk = {i: 0 for i in range(4)}
hk = [ab.concat_layers[i].register_forward_hook(
    (lambda i: lambda *a: cc.__setitem__(i, cc[i] + 1))(i)) for i in range(4)]
hk += [ab.fusion_layers[i].register_forward_hook(
    (lambda i: lambda *a: fk.__setitem__(i, fk[i] + 1))(i)) for i in range(4)]
with torch.no_grad():
    m(torch.randn(1, 32000), torch.randn(1, 512, 50))
for h in hk:
    h.remove()
print("  concat_layers 调用次数（一次 model forward，AudioBlock×2）:", cc)
print("  fusion_layers 调用次数:", fk)
idle = sum(p.numel() for p in ab.concat_layers[2].parameters())
print(f"  concat_layers[2] 参数 = {idle:,} | 占 audio_model 537,697 的 {idle / 537697 * 100:.2f}%")
print(f"  有效参与前向的参数 = {537697 - idle:,}")
print("  concat_layers[2] 的 state_dict key 数:",
      len([k for k in m.state_dict() if "concat_layers.2." in k]))

print()
print("=" * 78)
print("[W7] AudioBlock lookahead 的分界敏感性（A20 的 ±1 帧量化）")
print("=" * 78)
from look2hear.models.Falcon import AudioBlock  # noqa: E402
lay = copy.deepcopy(ACONF["audio_params"]["layers"])
ab_nn = AudioBlock(in_chan=112, hid_chan=64, kernel_size=5, stride=2, norm_type=None,
                   act_type="PReLU", upsampling_depth=4, layers=lay, is2d=True,
                   causal=True, n_freqs=129).eval()
x = torch.randn(1, 112, 251, 129)
for split in (100, 125, 150):
    x2 = x.clone()
    x2[..., split:, :] = torch.randn(1, 112, 251 - split, 129)
    with torch.no_grad():
        y1, y2 = ab_nn(x), ab_nn(x2)
    prof = (y1 - y2).abs().amax(dim=(0, 1, 3))
    ch = (prof[:split] > 1e-12).nonzero().flatten()
    first = ch.min().item() if ch.numel() else None
    print(f"  split={split}: 前缀最左受影响帧 = {first}, lookahead = "
          f"{split - first if first is not None else 0} 帧 = {(split - first) * 128 if first is not None else 0} 样本")

print()
print("=" * 78)
print("[W8] encoder 首变帧公式复核（P2-3）：f_first = floor((s - win/2)/hop) + 1")
print("=" * 78)
enc = m.encoder
for s in (16000, 20000, 16128):
    w1 = torch.zeros(1, 32000)
    w2 = w1.clone()
    w2[..., s:] = 1.0
    with torch.no_grad():
        e1, e2 = enc(w1), enc(w2)
    prof = (e1 - e2).abs().amax(dim=(0, 1, 3))
    nz = (prof > 1e-12).nonzero().flatten()
    f = (s - 128) // 128 + 1
    print(f"  s={s}: 首变帧={nz.min().item()} 末变帧={nz.max().item()} | 公式 f_first={f} "
          f"({'一致' if f == nz.min().item() else '不一致'})")
print()
print("DONE-R3")
