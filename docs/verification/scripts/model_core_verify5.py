"""model-core 补充验证 5：参数量手算复核 + causal 开关是否改变行为 + config 污染（dict.pop）复现。

运行：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify5.py
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
from look2hear.videomodels import FRCNNVideoModel  # noqa: E402

torch.manual_seed(0)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
NAME = CFG["audionet"]["audionet_name"]


def build(**over):
    return getattr(models, NAME)(sample_rate=16000, **{**copy.deepcopy(ACONF), **over})


def n(mo):
    return sum(p.numel() for p in mo.parameters())


print("=" * 78)
print("[R] 参数量手算复核（每一项都是可复算公式）")
print("=" * 78)
m = build().eval()
sep, ab = m.separator, m.separator.audio_net
rows = [
    ("encoder.conv 3*112*3*3", 3 * 112 * 3 * 3, n(m.encoder.conv)),
    ("audio_bottleneck cLNhw(112)+conv112->112k1", 224 + 224 + 112 * 112 + 112,
     n(m.audio_bottleneck)),
    ("ab.gateway dwconv112k1", 112 + 112 + 1, n(ab.gateway)),
    ("ab.projection 112*64+64+cLNhw(64)+PReLU", 112 * 64 + 64 + 256 + 1, n(ab.projection)),
    ("ab.down[0] dwconv64k3+cLNhw", 64 * 9 + 64 + 256, n(ab.downsample_layers[0])),
    ("ab.down[1] dwconv64k5+cLNhw", 64 * 25 + 64 + 256, n(ab.downsample_layers[1])),
    ("proc_f[0] biLSTM64->24 + out_proj48->64 + LN", 2 * (4 * 24 * 64 + 4 * 24 * 24 + 2 * 96)
     + 48 * 64 + 128, n(ab.process_f_layers[0])),
    ("proc_t[0] LSTM64->72 + out_proj72->64 + LN", 4 * 72 * 64 + 4 * 72 * 72 + 2 * 288
     + 72 * 64 + 128, n(ab.process_t_layers[0])),
    ("proc_att[0] 4*(2*(64*4+4+1+2*4*129)+(64*16+16+1+2*16*129))"
     "+(64*64+64+1+2*64*129)",
     4 * (2 * (64 * 4 + 4 + 1 + 2 * 4 * 129) + (64 * 16 + 16 + 1 + 2 * 16 * 129))
     + (64 * 64 + 64 + 1 + 2 * 64 * 129), n(ab.process_att_layers[0])),
    ("fusion[0] 3*(64*25+cLNhw(64))", 3 * (64 * 25 + 256), n(ab.fusion_layers[0])),
    ("concat[0] 3*(64*25+cLNhw(64))", 3 * (64 * 25 + 256), n(ab.concat_layers[0])),
    ("ab.residual_conv 64*112+112", 64 * 112 + 112, n(ab.residual_conv)),
    ("video_net.gateway dwconv512k1", 512 + 512 + 1, n(sep.video_net.gateway)),
    ("video_net.projection 512*32+32+BN1d(32)+PReLU", 512 * 32 + 32 + 64 + 1,
     n(sep.video_net.projection)),
    ("video_net.rnn LSTM32->32", 4 * 32 * 32 + 4 * 32 * 32 + 2 * 128, n(sep.video_net.rnn)),
    ("video_net.residual_conv 32*512+512", 32 * 512 + 512, n(sep.video_net.residual_conv)),
    ("caf.v_proj 512*112/16+112", 512 * 112 // 16 + 112, n(sep.caf.v_proj)),
    ("mask_generator PReLU+conv112->112k1", 1 + 112 * 112 + 112, n(m.mask_generator)),
    ("decoder ConvTranspose2d 112*2*3*3", 112 * 2 * 9, n(m.decoder)),
    ("spk_pool_proj (112*28+28+28)+(112*64+64)+(64*64+64)",
     112 * 28 + 28 + 28 + 112 * 64 + 64 + 64 * 64 + 64, n(m.spk_pool_proj)),
]
tot_manual = 0
allok = True
for name, manual, actual in rows:
    flag = "OK" if manual == actual else "MISMATCH"
    allok &= manual == actual
    tot_manual += manual
    print(f"  {flag:9s} {name:60s} 手算={manual:>7,} 实测={actual:>7,}")
print(f"  合计 手算={tot_manual:,} 实测={n(m):,} 全部一致={allok and tot_manual == n(m)}")
frcnn = FRCNNVideoModel(pretrain=None)
print(f"  FRCNNVideoModel 参数量={n(frcnn):,}  audio+video={n(m) + n(frcnn):,}")
print(f"  README 口径: 可训练 {n(m) / 1e6:.3f}M（四舍五入到 1 位 = 0.5M）; "
      f"含冻结视觉 {n(m) + n(frcnn):,} = {(n(m) + n(frcnn)) / 1e6:.2f}M")

print()
print("=" * 78)
print("[S] 顶层 causal 开关是否改变任何行为？（每次用全新 deepcopy，避免 dict 污染）")
print("=" * 78)
mt = build(causal=True)
mf = build(causal=False)
print("  params:", n(mt), "vs", n(mf))
print("  state_dict keys 相同:", list(mt.state_dict().keys()) == list(mf.state_dict().keys()))
mf.load_state_dict(mt.state_dict())
mt.eval(); mf.eval()
a = torch.randn(1, 32000)
v = torch.randn(1, 512, 50)
with torch.no_grad():
    o1, o2 = mt(a, v), mf(a, v)
print("  相同权重、仅顶层 causal 不同 -> 输出 bit-identical:", torch.equal(o1, o2))
print("  检查: 模型里所有 nn.Module 的 causal 属性取值 =",
      sorted({getattr(x, "causal") for _, x in mt.named_modules() if hasattr(x, "causal")}))

print()
print("=" * 78)
print("[T] config dict 被 Separator dict.pop 污染 -> 同一 config 二次构建得到残缺模型")
print("=" * 78)
shared = copy.deepcopy(ACONF)                    # 一份共享 dict（浅层引用会被 pop）
a1 = getattr(models, NAME)(sample_rate=16000, **shared)
print("  第 1 次构建 params =", f"{n(a1):,}",
      "| audio_params 剩余 keys =", sorted(a1.audio_params.keys())[:4], "...")
a2 = getattr(models, NAME)(sample_rate=16000, **shared)
print("  第 2 次构建 params =", f"{n(a2):,}", " <- 缺失 repeats/shared 后 AudioBlock/VideoBlock 未建")
print("  a2.separator.audio_net type =", type(a2.separator.audio_net).__name__,
      "len =", len(a2.separator.audio_net) if hasattr(a2.separator.audio_net, "__len__") else "n/a")
print("  a2.separator.video_net type =", type(a2.separator.video_net).__name__,
      "| a2.separator.caf type =", type(a2.separator.caf).__name__)
try:
    a2.eval()
    with torch.no_grad():
        a2(torch.randn(1, 32000), torch.randn(1, 512, 50))
    print("  a2 前向: 未抛错（静默产出错误结果）")
except Exception as ex:
    print("  a2 前向:", type(ex).__name__, str(ex)[:120])
print("  依赖方防御: audio_visual_test.py:133 / falcon_runtime.py:149 均先 deepcopy")
print()
print("DONE5")
