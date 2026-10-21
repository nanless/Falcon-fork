"""R2 独立复核脚本：评审 docs/understanding/03-training-system.md 的关键断言。
只读源码 + 轻量动态检查（不训练、不下载）。Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_r2_review_03.py
"""
import ast, json, os, symtable, sys, types, warnings
warnings.filterwarnings("ignore")
import torch

REPO = os.getcwd()
print("=" * 78)

# ---------- 1) symtable: audio_train.py / lightning_utils.py 未定义全局名 ----------
BUILTINS = set(dir(__builtins__)) | {"__name__", "__file__", "__doc__", "self", "cls"}
def undefined_globals(path):
    src = open(path).read()
    st = symtable.symtable(src, path, "exec")
    defined = {s.get_name() for s in st.get_symbols() if s.is_assigned() or s.is_imported()}
    bad = []
    def walk(tab, prefix=""):
        for child in tab.get_children():
            for s in child.get_symbols():
                n = s.get_name()
                if s.is_global() and not s.is_assigned() and n not in defined and n not in BUILTINS:
                    bad.append((child.get_name(), n))
            walk(child, prefix)
    walk(st)
    return sorted(set(bad))
for f in ("audio_train.py", "look2hear/utils/lightning_utils.py"):
    print(f"[1] {f} 未定义全局名: {undefined_globals(f)}")

# ---------- 2) Comet 专属 API：PL 内置 logger 的 experiment 是否有 .log(dict) ----------
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
csv = CSVLogger(os.path.join("/tmp", "r2_csv"), name="x")
print(f"[2] CSVLogger.experiment 类型 = {type(csv.experiment).__name__}; hasattr(experiment,'log') = "
      f"{hasattr(csv.experiment, 'log')}")
try:
    csv.experiment.log({"a": 1})
except AttributeError as e:
    print(f"    csv.experiment.log(dict) -> AttributeError: {e}")
try:
    TensorBoardLogger(os.path.join("/tmp", "r2_tb"))
    print("    TensorBoardLogger 可创建（意外）")
except Exception as e:
    print(f"    TensorBoardLogger 创建失败: {type(e).__name__}: {str(e)[:80]}")

# ---------- 3) PL 2.6.6 契约：accelerator=None / devices=[0..9] ----------
def try_trainer(**kw):
    try:
        pl.Trainer(**kw)
        return "OK (no error)"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:110]}"
print("[3] Trainer(accelerator=None) ->", try_trainer(accelerator=None, devices=None))
print("    Trainer(devices=[0..9], accelerator='cuda') ->",
      try_trainer(devices=list(range(10)), accelerator="cuda", logger=False))
print(f"    torch.cuda.is_available()={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")

# ---------- 4) parser round-trip：main_args / options / 三级 dict ----------
import importlib.util
from argparse import ArgumentParser
spec = importlib.util.spec_from_file_location("parser_utils", "look2hear/utils/parser_utils.py")
pu = importlib.util.module_from_spec(spec); spec.loader.exec_module(pu)
import yaml
def_conf = yaml.safe_load(open("configs/Falcon.yml"))
p = pu.prepare_parser_from_dict(def_conf, parser=ArgumentParser())
arg_dic, plain = pu.parse_args_as_dict(p, return_plain_args=True)
print("[4] round-trip top-level keys =", sorted(arg_dic.keys()))
print(f"    main_args == {{}} ? {arg_dic['main_args'] == {}}; options keys = {sorted(arg_dic['options'].keys())}")
print(f"    loss.train is dict ? {isinstance(arg_dic['loss']['train'], dict)}; "
      f"training.early_stop is dict ? {isinstance(arg_dic['training']['early_stop'], dict)}")

# ---------- 5) best_k_models.json：格式 vs audio_train.py 的 indent=0 ----------
raw = open("Experiments/checkpoint/Falcon/best_k_models.json").read()
lines = raw.splitlines()
lead = [len(l) - len(l.lstrip(" ")) for l in lines[1:-1]]
d = json.loads(raw)
sim = json.dumps(d, indent=0)
print(f"[5] shipped best_k_models: {len(lines)} 行(无尾换行={not raw.endswith(chr(10))}), "
      f"条目={len(d)}, 条目行前导空格={set(lead)}")
print(f"    json.dumps(..., indent=0) 的条目行前导空格 = 0, 是否与 shipped 同格式 = {sim.strip()==raw.strip()}")
print(f"    值 min = {min(d.values())} @ {os.path.basename(min(d, key=d.get))}; key 是否含 ${{ROOT}}: {all('${ROOT}' in k for k in d)}")
print(f"    代码 audio_train.py:186 = json.dump(best_k, f, indent=0) —— shipped 为 indent=2 → 不是当前代码产出")

# ---------- 6) Falcon.py：forward 调用的 _mdcl_loss 是否存在 ----------
tree = ast.parse(open("look2hear/models/Falcon.py").read())
falcon_cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Falcon"][0]
methods = {n.name for n in falcon_cls.body if isinstance(n, ast.FunctionDef)}
print(f"[6] Falcon 类方法含 _mdcl_loss_hard = {'_mdcl_loss_hard' in methods}; "
      f"含 _mdcl_loss = {'_mdcl_loss' in methods}")
print(f"    方法清单中含 mdcl 的: {[m for m in methods if 'mdcl' in m.lower()]}")

# ---------- 7) audionet_config 叶子键数（03 说 33 个超参） ----------
def leaves(x):
    if isinstance(x, dict):
        return sum(leaves(v) for v in x.values())
    return 1
aud = json.loads(json.dumps(yaml.safe_load(open("configs/Falcon.yml"))["audionet"]["audionet_config"]))
print(f"[7] audionet_config 顶层键={len(aud)} 叶子键={leaves(aud)}（+sample_rate 后 kwargs={len(aud)+1}）")

# ---------- 8) torch_optimizer / pytorch_ranger（03 A45） ----------
try:
    import torch_optimizer
    print("[8] import torch_optimizer -> OK")
except Exception as e:
    print(f"[8] import torch_optimizer -> {type(e).__name__}: {e}")
try:
    import importlib.metadata as md
    print("    torch-optimizer 依赖:", md.requires("torch-optimizer"))
except Exception as e:
    print("    metadata err", e)

# ---------- 9) torch 2.8 default weights_only 加载微型 Lightning 风格 ckpt ----------
pth = "/tmp/r2_mini.ckpt"
torch.save({"state_dict": {"w": torch.zeros(2)}, "epoch": 1,
            "pytorch-lightning_version": "2.6.6", "training_config": {"a": {"b": 1}}}, pth)
try:
    ck = torch.load(pth)
    print(f"[9] torch.load(默认) 微型 Lightning 风格 ckpt -> OK keys={sorted(ck.keys())}")
except Exception as e:
    print(f"[9] torch.load(默认) -> {type(e).__name__}: {str(e)[:100]}")

# ---------- 10) mouth 形状证据（03 §3.3/§5.2/§5.3 写 96×96） ----------
print("[10] mouth 形状静态证据: dataset 用 get_preprocessing_pipelines()[val] = Normalize→CenterCrop(88)→Normalize")
print("     transform.py:147-168；avspeech_dataset.py:138,189-192 -> 输出 [Tv,88,88]；")
print("     train DS avspeech_dataset_near.py:166-174 同样经 RandomCrop(88) -> [Tv,88,88]")
print("     （R1 已动态验证：伪造 1 s 样本 mouth=(25,88,88)，日志 runs/04-eval-flow.log §[8]）")
print("DONE")
