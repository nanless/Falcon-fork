import sys, torch, yaml
sys.path.insert(0, "/root/code/github_repos/Falcon-fork")
sys.path.insert(0, "/root/code/github_repos/Falcon-fork/docs/verification/stubs")
from look2hear.models.Falcon import STFTEncoder, STFTDecoder
cfg = yaml.safe_load(open("/root/code/github_repos/Falcon-fork/configs/Falcon.yml"))
enc_cfg = dict(cfg["audionet"]["audionet_config"]["enc_dec_params"])
torch.manual_seed(0)
enc = STFTEncoder(**enc_cfg).eval(); dec = STFTDecoder(**enc_cfg, in_chan=112, n_src=1).eval()
wav = torch.randn(1, 32000)
with torch.no_grad():
    d0 = dec(enc(wav), torch.Size([1,1,32000]))
    for s in (16000, 20000):
        w2 = wav.clone(); w2[0, s] += 5.0
        d1 = dec(enc(w2), torch.Size([1,1,32000]))
        diff = (d0 - d1).abs().amax(dim=[0,1])
        nz = torch.nonzero(diff > 1e-7).flatten()
        first = int(nz[0]) if nz.numel() else -1
        print(f"perturb s={s}: first changed output sample={first} (delta={first-s if first>=0 else 'NA'}), max|d|={float(diff.max()):.3e}")
