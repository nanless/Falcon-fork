"""R1 verification for docs/understanding/04: loss functions (pairwise/single/multi neg-SDR,
PITLossWrapper pit_from variants + threshold_byloss, MixITLossWrapper, NASSContrastiveLoss, SISNRi).
Pure-tensor checks, no data / no weights. Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_losses_04.py
"""
import itertools, math, sys, warnings
warnings.filterwarnings("ignore")
import torch
sys.path.insert(0, ".")
import look2hear.losses as L
from look2hear.losses.matrix import PairwiseNegSDR, SingleSrcNegSDR, MultiSrcNegSDR
from look2hear.losses.pit_wrapper import PITLossWrapper, new_PITLossWrapper
from look2hear.losses.mixit import MixITLossWrapper
from look2hear.losses.nass_contrastive_loss import NASSContrastiveLoss
from look2hear.losses.sisnri import SISNRi

torch.manual_seed(0)
EPS = 1e-8
def rel(a, b): return float((a - b).abs().max())

def ref_pairwise(ests, targets, kind, zm=True):
    if zm:
        targets = targets - targets.mean(-1, keepdim=True)
        ests = ests - ests.mean(-1, keepdim=True)
    B, S, T = targets.shape
    out = torch.zeros(B, S, S)
    for b in range(B):
        for i in range(S):
            e = ests[b, i]
            for j in range(S):
                s = targets[b, j]
                if kind in ("sisdr", "sdsdr"):
                    dot = (e * s).sum()
                    en = (s * s).sum() + EPS
                    proj = dot * s / en
                else:
                    proj = s
                enoise = e - s if kind in ("sdsdr", "snr") else e - proj
                val = (proj ** 2).sum() / ((enoise ** 2).sum() + EPS)
                out[b, i, j] = -10 * torch.log10(val + EPS)
    return out

def ref_single(e, s, kind, zm=True):
    if zm:
        s = s - s.mean(-1, keepdim=True); e = e - e.mean(-1, keepdim=True)
    if kind in ("sisdr", "sdsdr"):
        dot = (e * s).sum(-1, keepdim=True); en = (s * s).sum(-1, keepdim=True) + EPS
        proj = dot * s / en
    else:
        proj = s
    enoise = e - s if kind in ("sdsdr", "snr") else e - proj
    val = (proj ** 2).sum(-1) / ((enoise ** 2).sum(-1) + EPS)
    return -(10 * torch.log10(val + EPS))

print("=" * 78)
print("[1] PairwiseNegSDR: 三种 sdr_type 与手算公式对拍 (zero_mean=True)")
B, S, T = 2, 2, 50
targets = torch.randn(B, S, T); ests = torch.randn(B, S, T)
for kind in ("sisdr", "sdsdr", "snr"):
    got = PairwiseNegSDR(kind)(ests, targets)
    exp = ref_pairwise(ests, targets, kind, zm=True)
    print(f"  sdr_type={kind:6s} out{tuple(got.shape)} max|diff|={rel(got, exp):.3e} "
          f"sample loss[0,0,0]={got[0,0,0]:.4f}")

print("\n[2] zero_mean 语义：加直流偏置后 zero_mean=True 不变、False 改变")
bias = 3.0 * torch.ones(B, S, 1)
t2, e2 = targets + bias, ests - 0.5 * bias
a1 = PairwiseNegSDR("sisdr", zero_mean=True)(e2, t2)
a0 = PairwiseNegSDR("sisdr", zero_mean=True)(ests, targets)
b1 = PairwiseNegSDR("sisdr", zero_mean=False)(e2, t2)
b0 = PairwiseNegSDR("sisdr", zero_mean=False)(ests, targets)
print(f"  zero_mean=True : max|f(x+bias)-f(x)| = {rel(a1,a0):.3e}  (应≈0)")
print(f"  zero_mean=False: max|f(x+bias)-f(x)| = {rel(b1,b0):.3f}   (应>0)")
# 手算对照 zero_mean=False
exp_nz = ref_pairwise(e2, t2, "sisdr", zm=False)
print(f"  zero_mean=False 对拍 max|diff| = {rel(PairwiseNegSDR('sisdr', zero_mean=False)(e2,t2), exp_nz):.3e}")

print("\n[3] SNR 与 SDR 的关键差别：SNR 不尺度不变（即 allwrapper 里的 'sdr' 列其实是 SNR）")
e_s = torch.randn(1, 1, T); s_s = torch.randn(1, 1, T)
snr1 = -PairwiseNegSDR("snr")(e_s, s_s)[0, 0, 0].item()
snr2 = -PairwiseNegSDR("snr")(2.0 * e_s, s_s)[0, 0, 0].item()
sisdr1 = -PairwiseNegSDR("sisdr")(e_s, s_s)[0, 0, 0].item()
sisdr2 = -PairwiseNegSDR("sisdr")(2.0 * e_s, s_s)[0, 0, 0].item()
print(f"  snr   : est -> 2*est : {snr1:.3f} -> {snr2:.3f} dB (变化 {snr2-snr1:+.3f})")
print(f"  sisdr : est -> 2*est : {sisdr1:.3f} -> {sisdr2:.3f} dB (变化 {sisdr2-sisdr1:+.3f}, 应≈0)")

print("\n[4] SingleSrcNegSDR: reduction none/mean、sum 断言、公式对拍")
e1d = torch.randn(B, T); s1d = torch.randn(B, T)
got_none = SingleSrcNegSDR("sisdr", reduction="none")(e1d, s1d)
got_mean = SingleSrcNegSDR("sisdr", reduction="mean")(e1d, s1d)
exp_single = ref_single(e1d, s1d, "sisdr")
print(f"  reduction='none' max|diff|={rel(got_none, exp_single):.3e} shape={tuple(got_none.shape)}")
print(f"  reduction='mean' == mean(none): {abs(got_mean.item()-got_none.mean().item()):.3e}")
try:
    SingleSrcNegSDR("sisdr", reduction="sum"); print("  reduction='sum': 未报错 (意外)")
except AssertionError as ex: print(f"  reduction='sum' -> AssertionError({ex!r})  [代码 matrix.py:239]")

print("\n[5] MultiSrcNegSDR: 对每源独立算 -SDR 后取源均值（不做 PIT）")
m = MultiSrcNegSDR("sisdr")(ests, targets)
per_src = torch.stack([ref_single(ests[:, i], targets[:, i], "sisdr") for i in range(S)], 1)  # [B,S]
exp_m = per_src.mean(-1)   # ref_single 返回的已是 loss(-SDR)
print(f"  max|diff|={rel(m, exp_m):.3e} out{tuple(m.shape)}")

print("\n[6] PITLossWrapper pit_from 语义与手算对拍")
pairs = list(itertools.permutations(range(S)))
pw = ref_pairwise(ests, targets, "sisdr")           # [B,S,S] est x tgt
manual_best = []
for b in range(B):
    cand = [torch.stack([pw[b, i, p[i]] for i in range(S)]).mean() for p in pairs]
    manual_best.append(torch.stack(cand).min())
manual_pit = torch.stack(manual_best).mean()
w = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx", threshold_byloss=False)
print(f"  pw_mtx  : {w(ests, targets).item():.6f} vs 手算 {manual_pit.item():.6f}  diff={abs(w(ests,targets)-manual_pit):.2e}")
w_pt = PITLossWrapper(SingleSrcNegSDR("sisdr", reduction="none"), pit_from="pw_pt", threshold_byloss=False)
print(f"  pw_pt   : {w_pt(ests, targets).item():.6f} (应等于 pw_mtx) diff={abs(w_pt(ests,targets)-manual_pit):.2e}")
# perm_avg: 逐排列把 est 通道重排后喂给 multisrc 损失再取 min
ms = MultiSrcNegSDR("sisdr")
cand = torch.stack([ms(ests[:, list(p)], targets) for p in pairs], 1)  # [B, n_perm]
manual_pa = cand.min(1).values.mean()
w_pa = PITLossWrapper(ms, pit_from="perm_avg", threshold_byloss=False)
print(f"  perm_avg: {w_pa(ests, targets).item():.6f} vs 手算 {manual_pa.item():.6f} diff={abs(w_pa(ests,targets)-manual_pa):.2e}")
# pw_mtx_broadcast: PIT(pw_mtx) + 0.5 * 2nd-loss(reordered)
w_bc = PITLossWrapper((PairwiseNegSDR("sisdr"), MultiSrcNegSDR("sisdr")),
                      pit_from="pw_mtx_broadcast", threshold_byloss=False)
best_perm = [min(pairs, key=lambda p: torch.stack([pw[b, i, p[i]] for i in range(S)]).mean()) for b in range(B)]
reordered = torch.stack([ests[b][list(best_perm[b])] for b in range(B)])
exp_bc = manual_pit + 0.5 * ms(reordered, targets).mean()
print(f"  pw_mtx_broadcast: {w_bc(ests, targets).item():.6f} vs 手算 {exp_bc.item():.6f} diff={abs(w_bc(ests,targets)-exp_bc):.2e}")
# return_ests 正确性
_, re_ests = w(ests, targets, return_ests=True)
print(f"  return_ests: shape={tuple(re_ests.shape)} 与最佳排列重排一致={torch.allclose(re_ests, reordered)}")

print("\n[7] threshold_byloss 真实行为（matrix.py:227-230 在 PIT wrapper 内）")
B3, S3 = 3, 2
t3 = torch.randn(B3, S3, 200)
e3 = t3.clone()
e3[0] += torch.randn_like(e3[0]) * 0.5          # 普通样本 (SDR 低)
e3[1] += torch.randn_like(e3[1]) * 0.5          # 普通样本
e3[2] = t3[2] + 1e-5 * torch.randn_like(e3[2])  # 近乎完美 -> SDR > 30 dB
pw3 = ref_pairwise(e3, t3, "sisdr")
best3 = torch.stack([torch.stack([torch.stack([pw3[b, i, p[i]] for i in range(S3)]).mean() for p in pairs]).min() for b in range(B3)])
print(f"  min_loss per sample = {[round(x,3) for x in best3.tolist()]}  (> -30 的样本数 = {(best3>-30).sum().item()})")
w_thr = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx", threshold_byloss=True)
w_not = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx", threshold_byloss=False)
print(f"  threshold_byloss=True : {w_thr(e3,t3).item():.6f}   (= 只对 min_loss>-30 的样本求均值 {best3[best3>-30].mean().item():.6f})")
print(f"  threshold_byloss=False: {w_not(e3,t3).item():.6f}   (= 全部样本均值 {best3.mean().item():.6f})")
# 全部 >30dB 的极端情况
e_allgood = t3 + 1e-5 * torch.randn_like(t3)
out_all = w_thr(e_allgood, t3).item()
print(f"  全部样本 min_loss<-30 时 threshold=True -> {out_all} (过滤后为空则退回全体均值, 不报 nan)")

print("\n[8] multidecoder 变体与 new_PITLossWrapper")
ests_list = [ests + 0.1 * torch.randn_like(ests), ests + 0.3 * torch.randn_like(ests)]
w_mk = PITLossWrapper((PairwiseNegSDR("sisdr"), MultiSrcNegSDR("sisdr")), pit_from="pw_mtx_multidecoder_keepmtx", threshold_byloss=False)
print(f"  keepmtx: {w_mk(ests_list, targets).item():.6f} (matrix 用最后 block 选排列, 再对每 block 重排求 multisrc 均值)")
w_mb = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx_multidecoder_batchmin", threshold_byloss=False)
print(f"  batchmin: {w_mb(ests_list, targets).item():.6f} (两 block 在 batch 维拼接后做普通 PIT)")
# new_PITLossWrapper 的 pw_mtx 分支调用 loss_func(ests, ests_bn, targets, epoch)（4 实参）
try:
    new_PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx")(ests, targets, targets)
except TypeError as ex:
    print(f"  new_PITLossWrapper + PairwiseNegSDR -> TypeError: {ex} (签名不兼容)")
def staged_loss(ests, ests_bn, targets, epoch=None):
    return PairwiseNegSDR("sisdr")(ests, targets)   # 模拟 MultistageLoss 风格的 4 参损失
raw = new_PITLossWrapper(staged_loss, pit_from="pw_mtx")(ests, None, targets)
print(f"  new_PITLossWrapper(pw_mtx)+4参loss 返回原始矩阵 shape={tuple(raw.shape)} "
      f"与 pairwise 一致={torch.allclose(raw, PairwiseNegSDR('sisdr')(ests,targets))} -> 未做 PIT 归约（原样返回）")

print("\n[9] MixITLossWrapper: 2 混合 / 4 估计，最佳划分 + 手算对拍")
Bm, ns, nm = 2, 4, 2
tm = torch.randn(Bm, nm, 120)
em = torch.randn(Bm, ns, 120)
mw = MixITLossWrapper(MultiSrcNegSDR("sisdr"), generalized=True)
got_mixit = mw(em, tm).item()
def partitions(lst):
    out = []
    for k in range(len(lst) + 1):
        for c in itertools.combinations(lst, k):
            if c and len(c) != len(lst):
                out.append([list(c), [x for x in lst if x not in c]])
    return out
parts = partitions(list(range(ns)))
cand = []
for part in parts:
    mixes = torch.stack([em[:, idx, :].sum(1) for idx in part], 1)
    cand.append(MultiSrcNegSDR("sisdr")(mixes, tm))   # [B]
exp_mixit = torch.stack(cand, 1).min(1).values.mean()
loss_v, est_v = mw(em, tm, return_est=True)
print(f"  generalized=True : {got_mixit:.6f} vs 手算 {exp_mixit.item():.6f} diff={abs(got_mixit-exp_mixit.item()):.2e}; "
      f"return_est shape={tuple(est_v.shape)} n_parts={len(parts)}")
mw2 = MixITLossWrapper(MultiSrcNegSDR("sisdr"), generalized=False)
print(f"  generalized=False: {mw2(em, tm).item():.6f} (要求 nsrc%nmix==0, 划分为等大小组)")

print("\n[10] NASSContrastiveLoss: 公式/β 温度/margin/alpha 逐项对拍")
def ref_nass(q, p, negs, beta, margin, ap, an):
    tot = 0.0; n = 0
    for b in range(q.shape[0]):
        sim_p = torch.nn.functional.cosine_similarity(q[b][None], p[b][None]).item()
        lp = ap * (1.0 - sim_p)
        ng = negs[b]
        if ng is not None and ng.size(0) > 0:
            sn = torch.nn.functional.cosine_similarity(q[b][None].expand(ng.size(0), -1), ng, dim=1)
            if ng.size(0) == 1:
                ln = an * torch.relu(sn[0] - margin).item()
            else:
                wgt = torch.softmax(beta * sn, dim=0)
                ln = an * (wgt * torch.relu(sn - margin)).sum().item()
        else:
            ln = 0.0
        tot += lp + ln; n += 1
    return tot / n
q = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
p = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
def make_negs(qb, sims):
    out = []
    for t in sims:
        r = torch.randn_like(qb); r = r - (r @ qb) * qb; r = r / r.norm()
        out.append(t * qb + (1 - t * t) ** 0.5 * r)
    return torch.stack(out)
negs = [make_negs(q[0], [0.9]), None, make_negs(q[2], [0.95, 0.7, 0.6, 0.55])]  # 含 hard negatives
print("  crafted neg cos:", [None if n is None else [round(float(q[b] @ v), 2) for v in n] for b, n in enumerate(negs)])
crit = NASSContrastiveLoss(margin=0.5, alpha_pos=1.0, alpha_neg=1.0)
for beta in (0.0, 1.0, 5.0):
    g = crit(q, p, negs, beta).item(); r = ref_nass(q, p, negs, beta, 0.5, 1.0, 1.0)
    print(f"  beta={beta:3.1f}: loss={g:.6f} ref={r:.6f} diff={abs(g-r):.2e}")
c_am = NASSContrastiveLoss(margin=1.5, alpha_pos=2.0, alpha_neg=0.5)  # margin>所有 cos -> relu 全 0
print(f"  margin=1.5 (>max cos) -> loss={c_am(q,p,negs,1.0).item():.4f} (只剩 alpha_pos*(1-cos_pos))")
# 梯度仅经 query 有定义；pos/neg 若 requires_grad 也有梯度（"detached" 是调用方的责任）
qg = q.clone().requires_grad_(True); pg = p.clone().requires_grad_(True)
crit(qg, pg, negs, 1.0).backward()
print(f"  梯度: dL/dq 非零={qg.grad.abs().sum().item()>0}; dL/dp 非零={pg.grad.abs().sum().item()>0} "
      f"(loss 内部未 detach, 由 system 层 no_grad 保证)")
print(f"  全 batch 无负样本时 loss 仍算正样本项: {crit(q, p, [None, None, None], 1.0).item():.4f}")

print("\n[11] SISNRi 类：__init__ 存在 NameError bug（super(Loss, self)）")
try:
    SISNRi(); print("  实例化成功 (意外)")
except Exception as ex:
    print(f"  SISNRi() -> {type(ex).__name__}: {ex}   [sisnri.py:6]")
o = SISNRi.__new__(SISNRi)   # 绕过 __init__ 验证数学
mix = torch.randn(2, 400); est = torch.randn(2, 400); ref = torch.randn(2, 400)
imp = o.sisnr(mix, est, ref)
def ref_sisnr(mix, est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True); ref = ref - ref.mean(-1, keepdim=True); mix = mix - mix.mean(-1, keepdim=True)
    ep = (est * ref).sum(-1, keepdim=True) * ref / (ref * ref).sum(-1, keepdim=True)
    ev = est - ep
    mp = (mix * ref).sum(-1, keepdim=True) * ref / (ref * ref).sum(-1, keepdim=True)
    mv = mix - mp
    return 10*torch.log10((ep**2).sum(-1,keepdim=True)+eps) - 10*torch.log10((ev**2).sum(-1,keepdim=True)+eps) \
           - (10*torch.log10((mp**2).sum(-1,keepdim=True)+eps) - 10*torch.log10((mv**2).sum(-1,keepdim=True)+eps))
print(f"  sisnr() 手算对拍 max|diff|={rel(imp, ref_sisnr(mix,est,ref)):.3e} (返回 SI-SNR 改善量)")
print(f"  ____init____ 未定义变量: 见源码 sisnri.py:6 (super(Loss, self))")

print("\n[12] 配置到损失对象：按 audio_train.py:91-97 的方式从 configs/Falcon.yml 实例化")
import yaml
cfg = yaml.safe_load(open("configs/Falcon.yml"))["loss"]
for split in ("train", "val"):
    c = cfg[split]
    obj = getattr(L, c["loss_func"])(getattr(L, c["sdr_type"]), **c["config"])
    print(f"  {split}: {c['loss_func']}({c['sdr_type']}) pit_from={obj.pit_from} "
          f"threshold_byloss={obj.threshold_byloss} loss_func={type(obj.loss_func).__name__}/{obj.loss_func.sdr_type}")
print("\nDONE")
