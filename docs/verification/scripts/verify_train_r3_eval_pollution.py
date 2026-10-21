"""R3 执行级验证 A: 建模污染 config 后重建模型 → 参数量下降 + strict=False 静默丢权重。
证据链：audio_train.py:55-58 以 **audionet_config 建模 → Separator.__init__ pop repeats/shared
(Falcon.py:991-996) / Falcon.__init__ 写入 causal (Falcon.py:1314-1315) → 若把被污染后的 conf.yml
再用 audio_visual_test._load_models_from_ckpt 的方式重建模型，会发生什么。"""
import sys, copy, yaml, torch

sys.path.insert(0, "/root/code/github_repos/Falcon-fork")
import look2hear.models as M

full_cfg = yaml.safe_load(open("/root/code/github_repos/Falcon-fork/configs/Falcon.yml"))["audionet"]["audionet_config"]
conf_cfg = yaml.safe_load(open("/root/code/github_repos/Falcon-fork/Experiments/checkpoint/Falcon/conf.yml"))["audionet"]["audionet_config"]

full = M.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(full_cfg))
polluted = M.FalconSpeakerContrastive(sample_rate=16000, **copy.deepcopy(conf_cfg))

pf = sum(p.numel() for p in full.parameters())
pp = sum(p.numel() for p in polluted.parameters())
print(f"[A] full(Falcon.yml) params      = {pf:,}")
print(f"[A] polluted(conf.yml) params    = {pp:,}  (ratio {pp/pf:.1%})")

sd = full.state_dict()
missing, unexpected = polluted.load_state_dict(sd, strict=False)
total_elems = sum(v.numel() for v in sd.values())
missing_elems = sum(sd[k].numel() for k in missing)
print(f"[A] state_dict keys={len(sd)}, missing={len(missing)}, unexpected={len(unexpected)}")
print(f"[A] silently dropped tensor elems = {missing_elems:,} ({missing_elems/total_elems:.1%} of all)")
print(f"[A] first 6 missing keys: {missing[:6]}")
try:
    polluted.load_state_dict(sd, strict=True)
    print("[A] strict=True: no error (unexpected)")
except Exception as e:
    print(f"[A] strict=True raises: {type(e).__name__}: {str(e)[:150]}")

# conf.yml 的 audionet_config 实际键（证明其中没有 repeats/shared，但有 causal）
for name, cfg in (("Falcon.yml", full_cfg), ("conf.yml", conf_cfg)):
    ap, vp = cfg["audio_params"], cfg["video_params"]
    print(f"[A] {name}: audio_params has causal={'causal' in ap}, repeats={'repeats' in ap}, "
          f"shared={'shared' in ap} | video_params has causal={'causal' in vp}, "
          f"repeats={'repeats' in vp}, shared={'shared' in vp}")
