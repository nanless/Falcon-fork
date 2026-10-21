import sys, copy, yaml, torch
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")
import look2hear.models as M

REPO = "/root/code/github_repos/Falcon-fork"
full_cfg = yaml.safe_load(open(f"{REPO}/configs/Falcon.yml"))["audionet"]["audionet_config"]
dumped_cfg = yaml.safe_load(open("/tmp/repro_run/Experiments/checkpoint/repro_conf_dump/conf.yml"))["audionet"]["audionet_config"]

full = M.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(full_cfg))
bad = M.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(dumped_cfg))
pf = sum(p.numel() for p in full.parameters()); pb = sum(p.numel() for p in bad.parameters())
print(f"[C] full params={pf:,} | rebuilt-from-current-code-dump params={pb:,} (ratio {pb/pf:.1%})")
sd = full.state_dict()
missing, unexpected = bad.load_state_dict(sd, strict=False)
tot = sum(v.numel() for v in sd.values()); miss = sum(sd[k].numel() for k in missing)
print(f"[C] strict=False load: missing keys={len(missing)}, unexpected={len(unexpected)}, "
      f"dropped elems={miss:,} ({miss/tot:.1%})")
print(f"[C] missing examples: {missing[:3]}")
# eval 用 path: bad.eval() 后 FalconSpeakerContrastive.forward 返回单张量；能否静默跑通？
bad.eval()
x = torch.randn(1, 32000)
mouth_emb = torch.randn(1, 512, 50)
try:
    with torch.no_grad():
        y = bad(x, mouth_emb)
    print(f"[C] polluted model forward eval OK -> {tuple(y.shape)} (silent wrong-result risk)")
except Exception as e:
    print(f"[C] polluted model forward eval FAILED: {type(e).__name__}: {str(e)[:150]}")
# 对照：完整模型同输入
full.eval()
with torch.no_grad():
    y2 = full(x, mouth_emb)
print(f"[C] full model forward eval OK -> {tuple(y2.shape)}")
