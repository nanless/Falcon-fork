"""Probe: import chain of look2hear.system / look2hear.utils under an installed
pytorch_lightning 2.6.6.  pytorch_ranger is stubbed because the sandbox venv
installed torch-optimizer with --no-deps."""
import sys, types, traceback

pr = types.ModuleType("pytorch_ranger")
class _R:
    def __init__(self, *a, **k): pass
pr.Ranger = pr.RangerQH = pr.RangerVA = _R
sys.modules["pytorch_ranger"] = pr
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")

for name in ["torch", "pytorch_lightning"]:
    try:
        m = __import__(name)
        print(f"[OK] import {name} {m.__version__}")
    except Exception as e:
        print(f"[FAIL] import {name}: {type(e).__name__}: {e}")

for mod in ["look2hear.utils", "look2hear.system", "look2hear.losses",
            "look2hear.system.av_litmodule_nass_v2", "look2hear.system.av_litmodule_mdcl",
            "look2hear.system.optimizers", "look2hear.system.schedulers"]:
    try:
        m = __import__(mod, fromlist=["*"])
        print(f"[OK] import {mod}")
    except Exception as e:
        print(f"[FAIL] import {mod}: {type(e).__name__}: {str(e)[:160]}")

try:
    import look2hear.models as M
    print("[OK] import look2hear.models; attrs:", [x for x in ("Falcon", "FalconSpeakerContrastive", "SpeakerContrastiveV2") if hasattr(M, x)])
except Exception as e:
    print(f"[FAIL] import look2hear.models: {type(e).__name__}: {str(e)[:160]}")

# torch_optimizer import (fails without pytorch_ranger)
import importlib
sys.modules.pop("pytorch_ranger", None)
try:
    import torch_optimizer
    print("[OK] import torch_optimizer (without stub)")
except Exception as e:
    print(f"[FAIL] import torch_optimizer without pytorch_ranger stub: {type(e).__name__}: {e}")

# torch.cuda facts
import torch
print("torch.cuda.is_available():", torch.cuda.is_available(), "| device_count:", torch.cuda.device_count())
