"""R3 执行级验证 B: 最小复现 audio_train.py:82-84 的 conf.yml dump（monkeypatch Trainer/comet_logger），
确认当前代码 dump 出的 conf.yml 会缺 audio_params.repeats/shared、多出 causal —— 即“污染后落盘键集”。"""
import os, sys, argparse, yaml, traceback

REPO = "/root/code/github_repos/Falcon-fork"
sys.path.insert(0, REPO)

# ---- 载入真实入口（依赖 stub：timm/typeguard/torchvision/cv2/matplotlib/pytorch_ranger 由 PYTHONPATH 提供）
import audio_train
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

# ---- monkeypatch: comet_logger 未定义是已知 P0；Trainer 换成 dummy，使 main() 能跑到 dump(:84)
audio_train.comet_logger = None

class DummyTrainer:
    last_kwargs = None
    def __init__(self, **kwargs):
        DummyTrainer.last_kwargs = kwargs
    def fit(self, system, **kw):
        print(f"[B] DummyTrainer.fit called (ckpt_path={kw.get('ckpt_path')!r})")

audio_train.pl.Trainer = DummyTrainer

# ---- 构造与 __main__ 等价的 arg_dic（真 YAML → parser round-trip）
def_conf = yaml.safe_load(open(os.path.join(REPO, "configs/Falcon.yml")))
def_conf["datamodule"]["data_config"].update(
    train_dir="/tmp/dp-verify/index/tr", valid_dir="/tmp/dp-verify/index/cv",
    test_dir="/tmp/dp-verify/index/tt", batch_size=2, num_workers=0,
    persistent_workers=False)
def_conf["videonet"]["videonet_config"]["pretrain"] = None       # 沙箱无 backbone
def_conf["training"]["gpus"] = [0]
def_conf["exp"]["exp_name"] = "repro_conf_dump"
parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir", default="configs/Falcon.yml")
parser.add_argument("--ckpt_path", default="")
parser = prepare_parser_from_dict(def_conf, parser=parser)
arg_dic, _ = parse_args_as_dict(parser, return_plain_args=True)

# ---- 在 /tmp 下跑（exp_dir = cwd/Experiments/checkpoint/<name>，不污染仓库）
os.makedirs("/tmp/repro_run", exist_ok=True)
os.chdir("/tmp/repro_run")
try:
    audio_train.main(arg_dic)
except Exception as e:
    print(f"[B] main() 末尾预期失败（best_model_path='' → torch.load）: {type(e).__name__}: {str(e)[:120]}")

conf_path = "/tmp/repro_run/Experiments/checkpoint/repro_conf_dump/conf.yml"
print(f"[B] dumped exists: {os.path.exists(conf_path)} -> {conf_path}")
dumped = yaml.safe_load(open(conf_path))
ap = dumped["audionet"]["audionet_config"]["audio_params"]
vp = dumped["audionet"]["audionet_config"]["video_params"]
print(f"[B] dumped audio_params: causal={'causal' in ap}, repeats={'repeats' in ap}, shared={'shared' in ap}; keys={sorted(ap)}")
print(f"[B] dumped video_params: causal={'causal' in vp}, repeats={'repeats' in vp}, shared={'shared' in vp}")
print(f"[B] dumped top-level keys: {sorted(dumped.keys())}")
print(f"[B] dumped main_args: {dumped.get('main_args')}")
print(f"[B] dumped options: {dumped.get('options')}")
print(f"[B] Trainer kwargs intercepted: {sorted(DummyTrainer.last_kwargs.keys())}")
for k in ("devices", "accelerator", "gradient_clip_val", "limit_train_batches",
          "sync_batchnorm", "num_sanity_val_steps", "max_epochs", "logger"):
    print(f"[B]   Trainer.{k} = {DummyTrainer.last_kwargs.get(k)!r}")
