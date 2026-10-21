"""model-core 补充验证 3：返回类型契约 + MDCL 非零损失/梯度/β 语义（用相关信号使 sim>margin）。

运行：
    PYTHONPATH=$PWD/docs/verification/stubs:$PWD \
      /root/.venvs/falcon/bin/python docs/verification/scripts/model_core_verify3.py
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

torch.manual_seed(0)
CFG = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
ACONF = CFG["audionet"]["audionet_config"]
m = getattr(models, CFG["audionet"]["audionet_name"])(
    sample_rate=16000, **copy.deepcopy(ACONF))

print("=" * 78)
print("[L] 返回类型契约（train=tuple / eval=Tensor）")
print("=" * 78)
audio = torch.randn(1, 32000)
mouth = torch.randn(1, ACONF["pretrained_vout_chan"], 50)
m.train()
with torch.no_grad():
    out_train = m(audio, mouth)
m.eval()
with torch.no_grad():
    out_eval = m(audio, mouth)
print("  train ->", type(out_train).__name__,
      [tuple(t.shape) for t in out_train] if isinstance(out_train, tuple) else tuple(out_train.shape))
print("  eval  ->", type(out_eval).__name__, tuple(out_eval.shape))
m.train()
try:
    m(audio, mouth, near_audio=audio)
    print("  train + near_audio -> 未抛错")
except TypeError as ex:
    print("  train + near_audio -> TypeError:", ex)

print()
print("=" * 78)
print("[M] MDCL 非零损失 / 梯度 / β 语义（est 与 near 相关，使 mag_sim > margin=0）")
print("=" * 78)
ref = torch.randn(2, 16000)
est = (ref + 0.05 * torch.randn(2, 16000)).clone().requires_grad_(True)
near = [ref + 0.05 * torch.randn(2, 16000) for _ in range(3)]
sims = torch.stack([m._mag_sim(est.detach(), it) for it in near], 0)
print("  sim_scores(dB) [K,B] =", [[round(v, 2) for v in row] for row in sims.tolist()])
L1 = m._mdcl_loss_hard(est, near[:1], beta=1.0)
man1 = torch.relu(m._mag_sim(est, near[0]) - m.mdcl_margin).mean()
print("  K=1 L =", f"{L1.item():.6f}", "| 手算:", f"{man1.item():.6f}",
      "| allclose:", torch.allclose(L1, man1))
for beta in (0.0, 1.0, 20.0):
    L = m._mdcl_loss_hard(est, near, beta=beta)
    w = torch.softmax(beta * sims, 0)
    man = (w * torch.relu(sims - m.mdcl_margin)).sum(0).mean()
    print(f"  K=3 beta={beta:<5} L={L.item():.6f} 手算={man.item():.6f} "
          f"allclose={torch.allclose(L, man)} w[:,0]={[round(v, 3) for v in w[:, 0].tolist()]}")
L = m._mdcl_loss_hard(est, near, beta=1.0)
L.requires_grad_(True)
L.backward()
print("  |grad| =", f"{est.grad.abs().sum().item():.4e}", "(>0 说明梯度可回传到估计波形)")
# margin 生效性
m.mdcl_margin = 500.0
print("  margin=500 时 loss =", m._mdcl_loss_hard(est.detach(), near, beta=1.0).item(),
      "(relu 截断 -> 0)")
m.mdcl_margin = -500.0
print("  margin=-500 时 loss =", f"{m._mdcl_loss_hard(est.detach(), near, beta=1.0).item():.4f}",
      "(无截断 -> 加权平均相似度)")
m.mdcl_margin = 0.0

print()
print("=" * 78)
print("[N] 训练态返回 tuple 与 NASSv2 litmodule 的契约一致性（静态）")
print("=" * 78)
print("  av_litmodule_nass_v2.py:90 -> est_sources, sep_audio_embs = self.audio_model(...)")
print("  仅在 self.audio_model.training == True 时成立（Falcon_Contrastive.py:116-118）")
print("  av_litmodule_mdcl.py:91    -> est_sources = self.audio_model(mixtures, mouth_emb)")
print("  在 train 模式下会拿到 tuple -> 与 FalconSpeakerContrastive 不兼容（仅兼容裸 Falcon）")
print()
print("DONE3")
