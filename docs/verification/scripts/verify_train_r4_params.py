import sys, yaml, copy
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")
sys.path.insert(0, "/root/code/github_repos/Falcon-fork/docs/verification/stubs")
import look2hear.models as M
cfg = yaml.safe_load(open('/root/code/github_repos/Falcon-fork/configs/Falcon.yml'))
m = M.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(cfg['audionet']['audionet_config']))
attrs = ['encoder', 'audio_bottleneck', 'separator', 'mask_generator', 'decoder']
tot = 0
for a in attrs:
    n = sum(p.numel() for p in getattr(m, a).parameters())
    tot += n
    print(f"{a}: {n:,}")
print(f"audio path total: {tot:,}")
print(f"spk_pool_proj: {sum(p.numel() for p in m.spk_pool_proj.parameters()):,}")
print(f"total: {sum(p.numel() for p in m.parameters()):,}")
