"""R1 verification for docs/understanding/04: videomodels (FRCNN + variants).
Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_video_04.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import torch
from look2hear.videomodels import (FRCNNVideoModel, ResNetVideoModel, LightVideomodel,
                                   WujianVideoModel, ShuffleNetV2, update_frcnn_parameter)

def nparams(m): return sum(p.numel() for p in m.parameters())

print("=" * 78)
print("[1] FRCNNVideoModel(resnet, prelu, w=1.0): 参数量/形状/时间步")
cfg = dict(backbone_type="resnet", relu_type="prelu", width_mult=1.0)
m = FRCNNVideoModel(pretrain=None, **cfg)
print(f"  total params = {nparams(m):,} = {nparams(m)/1e6:.4f} M; requires_grad=True params = "
      f"{sum(p.requires_grad for p in m.parameters())} 个张量")
for shape in [(1, 1, 50, 96, 96), (1, 1, 50, 88, 88), (1, 1, 25, 88, 88), (1, 1, 1, 88, 88), (1, 1, 50, 64, 64)]:
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(*shape))
    print(f"  in {shape} -> out {tuple(y.shape)}   (T_out == T_in: {y.shape[-1] == shape[2]})")
print(f"  backend_out attr = {m.backend_out}; frontend_nout = {m.frontend_nout}")
print(f"  FRCNNVideoModel().eval() returns: {FRCNNVideoModel(pretrain=None, **cfg).eval()!r} (train() 无 return)")

print("\n[2] 冻结与 pretrain 路径")
sd = {k: v.clone() for k, v in m.state_dict().items()}
sd["tcn.dummy"] = torch.zeros(1)          # 模拟 pretrain 里多出来的 tcn.* key
update_frcnn_parameter(m, sd)
print(f"  update_frcnn_parameter 后 all requires_grad=False: {all(not p.requires_grad for p in m.parameters())}")
try:
    FRCNNVideoModel(pretrain="pretrain_zoo/frcnn_128_512.backbone.pth.tar", **cfg)
except FileNotFoundError as e:
    print(f"  pretrain 文件缺失 -> FileNotFoundError: {str(e)[:70]}")
m2 = FRCNNVideoModel(pretrain=None, **cfg); m2.train()
print(f"  train() 模式下 BatchNorm 仍处于 eval（冻结统计量）: "
      f"{all(not mod.training for mod in m2.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm))}")

print("\n[3] 其它 videomodel 变体（均不能被 configs/*.yml 选中）")
variants = {}
ms = FRCNNVideoModel(pretrain=None, backbone_type="shufflenet", relu_type="prelu", width_mult=1.0); ms.eval()
with torch.no_grad(): ys = ms(torch.zeros(1, 1, 50, 88, 88))
variants["FRCNNVideoModel(backbone=shufflenet,w=1.0)"] = (nparams(ms), tuple(ys.shape))
rv = ResNetVideoModel(relu_type="prelu", pretrain=None); rv.eval()
with torch.no_grad(): yr = rv(torch.zeros(1, 1, 50, 88, 88))
variants["ResNetVideoModel"] = (nparams(rv), tuple(yr.shape))
lv = LightVideomodel(relu_type="prelu", width_mult=1.0, pretrain=None); lv.eval()
with torch.no_grad(): yl = lv(torch.zeros(1, 1, 50, 88, 88))
variants["LightVideomodel(shufflenet)"] = (nparams(lv), tuple(yl.shape))
wv = WujianVideoModel(); wv.eval()
with torch.no_grad(): yw = wv(torch.zeros(1, 50, 88, 88))
variants["WujianVideoModel"] = (nparams(wv), tuple(yw.shape))
for k, (p, s) in variants.items():
    print(f"  {k:38s} params={p/1e6:7.3f} M  out={s}")
print(f"  ShuffleNetV2(standalone,input_size=96,w=1.0) params={nparams(ShuffleNetV2(96,96,1.0))/1e6:.2f} M")
# shufflenet 的 globalpool 固定 AvgPool2d(3)，输入 64x64 会崩
try:
    with torch.no_grad(): ms(torch.zeros(1, 1, 10, 64, 64))
except RuntimeError as e:
    print(f"  shufflenet 分支对 64x64 输入 -> RuntimeError: {str(e)[:90]}")

print("\n[4] 88x88 裁剪不在 FRCNN 内部：96x96 与 88x88 都能直接前向，模型无 crop/断言")
m.eval()
outs = []
with torch.no_grad():
    for hw in (96, 88):
        outs.append(m(torch.zeros(1, 1, 4, hw, hw)))
print(f"  96x96 -> {tuple(outs[0].shape)}, 88x88 -> {tuple(outs[1].shape)}（都在 dataset 预处理里裁成 88）")
print("\nDONE")
