import sys, traceback, torch
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel
m = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu", width_mult=1.0, pretrain=None)
print("frontend3D:", type(m.frontend3D).__name__, "| trunk:", type(m.trunk).__name__)
print("backend_out:", m.backend_out, "| attrs:", [a for a in dir(m) if not a.startswith('_')][:15])
x = torch.randn(2, 1, 50, 88, 88)
try:
    with torch.no_grad():
        y = m(x)
    print("OK", tuple(y.shape))
except Exception:
    traceback.print_exc()
