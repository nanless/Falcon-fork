"""R1 verification for docs/understanding/04: metrics trackers (wrapper/allwrapper/splitwrapper)
and the metric path actually used by audio_visual_test.py.
Third-party metric packages are stubbed (they are not installed); stub signatures follow the real
packages (fast_bss_eval.sdr_pit_loss(est, ref) returns negative SDR; pypesq.pesq(ref, deg, fs);
pystoi.stoi(clean, deg, fs)). Run from repo root:
PYTHONPATH=$PWD/docs/verification/stubs:$PWD /root/.venvs/falcon/bin/python docs/verification/scripts/verify_metrics_04.py
"""
import csv, sys, types, warnings
warnings.filterwarnings("ignore")
import torch
sys.path.insert(0, ".")

# ---- stub third-party metric packages (signatures faithful to upstream sources, see doc §8) ----
mir = types.ModuleType("torch_mir_eval"); msep = types.ModuleType("torch_mir_eval.separation")
msep.bss_eval_sources = lambda ref, est: (None, None, None, None)
mir.separation = msep; sys.modules["torch_mir_eval"] = mir; sys.modules["torch_mir_eval.separation"] = msep

fbe = types.ModuleType("fast_bss_eval")
def sdr_pit_loss(est, ref, **kw):
    e = est - est.mean(-1, keepdim=True); r = ref - ref.mean(-1, keepdim=True)
    proj = (e * r).sum(-1, keepdim=True) * r / ((r * r).sum(-1, keepdim=True) + 1e-8)
    val = (proj ** 2).sum(-1) / (((e - proj) ** 2).sum(-1) + 1e-8)
    return -10 * torch.log10(val + 1e-8)          # negative SDR (loss), est first, ref second
fbe.sdr_pit_loss = sdr_pit_loss; sys.modules["fast_bss_eval"] = fbe

PESQ_CALLS, STOI_CALLS = [], []
pyp = types.ModuleType("pypesq")
def pesq(ref, deg, fs=16000, normalize=False):
    PESQ_CALLS.append((ref, deg, fs)); return 3.14
pyp.pesq = pesq; sys.modules["pypesq"] = pyp
pys = types.ModuleType("pystoi")
def stoi(x, y, fs_sig, extended=False):
    STOI_CALLS.append((x, y, fs_sig, extended)); return 0.9
pys.stoi = stoi; sys.modules["pystoi"] = pys

from look2hear.metrics import ALLMetricsTracker, MetricsTracker, SPlitMetricsTracker
from look2hear.losses import PairwiseNegSDR
torch.manual_seed(0)

def sign(x): return "neg" if x < 0 else "pos"
def read_csv(p):
    with open(p) as f: return list(csv.DictReader(f))

T = 800
mix = torch.randn(T) * 0.3
clean = torch.randn(1, T) * 0.05
est = clean + 0.1 * torch.randn(1, T)     # SNR 约 +20 dB 的“正常”估计
TMP = "/tmp/04_metrics"

print("=" * 78)
print("[1] ALLMetricsTracker（audio_visual_test.py 实际使用的类）")
a = ALLMetricsTracker(save_file=TMP + "_all.csv")
a(mix=mix, clean=clean, estimate=est, key="utt_001")
mean = a.get_mean(); a.final()
rows = read_csv(TMP + "_all.csv")
print("  CSV columns:", list(rows[0].keys()))
print("  per-utt row :", {k: (round(float(v), 3) if k != "snt_id" else v) for k, v in rows[0].items()})
print("  avg row     :", {k: (round(float(v), 3) if k != "snt_id" else v) for k, v in rows[1].items()})
print("  std row     :", {k: (round(float(v), 3) if k != "snt_id" else v) for k, v in rows[2].items()})
print(f"  >>> per-utt 'sdr'={float(rows[0]['sdr']):+.3f} ({sign(float(rows[0]['sdr']))}), 'avg' 行 sdr={float(rows[1]['sdr']):+.3f} ({sign(float(rows[1]['sdr']))})"
      " -> 同一列符号相反")
print(f"  >>> get_mean() = {{'sdr_i': {mean['sdr_i']:.3f}, 'si-snr_i': {mean['si-snr_i']:.3f}}}（内部已取负，故为正）")
# 该列的 sdr 到底是什么？与 pairwise_neg_snr 直接对比
snr_loss = PairwiseNegSDR("snr")(est.unsqueeze(0), clean.unsqueeze(0)).item()
print(f"  per-utt 'sdr' == PairwiseNegSDR('snr') 的 loss 值 ({float(rows[0]['sdr']):.6f} vs {snr_loss:.6f}) "
      f"-> 名为 SDR，实为 -SNR；avg 行再取负 -> 是 SNR")
# 尺度不变性探针：真 SDRi 应对 est 整体缩放不变，这里会变 -> 证明该列不是 SDR 体系
a_s = ALLMetricsTracker(save_file=TMP + "_all_scale.csv")
a_s(mix=mix, clean=clean, estimate=2.0 * est, key="scale")
a_s.final(); rs = read_csv(TMP + "_all_scale.csv")[0]
print(f"  est -> 2*est : per-utt 'sdr' {float(rows[0]['sdr']):+.3f} -> {float(rs['sdr']):+.3f} dB, "
      f"'sdr_i' {float(rows[0]['sdr_i']):+.3f} -> {float(rs['sdr_i']):+.3f} dB (真 SDR/SDRi 应尺度不变)")
# PESQ/STOI 参数顺序
print(f"  PESQ 调用参数顺序: 第1个实参 == estimate ? {PESQ_CALLS[0][0] is not None and torch.allclose(torch.from_numpy(PESQ_CALLS[0][0]), est.squeeze(0))}"
      f" （pypesq 合同: pesq(ref, deg, fs) 第1个应是 clean -> 传反了）")
print(f"  STOI 调用参数顺序: 第1个实参 == clean ? {torch.allclose(torch.from_numpy(STOI_CALLS[0][0]), clean.squeeze(0))} "
      f"（pystoi 合同: stoi(clean, deg, fs) -> 正确）")

print("\n[2] MetricsTracker（wrapper.py；有 fast_bss_eval 的真 SDR，但无人调用）")
b = MetricsTracker(save_file=TMP + "_w.csv")
mix2 = torch.randn(T); clean2 = torch.randn(2, T); est2 = clean2 + 0.2 * torch.randn(2, T)
b(mix=mix2, clean=clean2, estimate=est2, key="utt_002")
print("  update() =", {k: round(float(v), 3) for k, v in b.update().items()})
b.final()
rows = read_csv(TMP + "_w.csv")
print("  CSV columns:", list(rows[0].keys()))
print("  per-utt row :", {k: (round(float(v), 3) if k != "snt_id" else v) for k, v in rows[0].items()})
print(f"  >>> 该 tracker 的 'sdr' = -sdr_pit_loss (fast_bss_eval 返回负 SDR) -> per-utt 为 {sign(float(rows[0]['sdr']))} 值 (正常)")

print("\n[3] 两个 tracker 的 SI-SDR 口径不同：wrapper.py 显式 zero_mean=False（含直流分量）")
mix3 = torch.randn(T); clean3 = torch.randn(1, T); est3 = clean3 + 0.2 * torch.randn(1, T)
clean3_dc = clean3 + 5.0; est3_dc = est3 + 5.0; mix3_dc = mix3 + 5.0
a2 = ALLMetricsTracker(save_file=TMP + "_all2.csv"); a2(mix=mix3_dc, clean=clean3_dc, estimate=est3_dc, key="dc"); mean2 = a2.get_mean(); a2.final()
b2 = MetricsTracker(save_file=TMP + "_w2.csv"); b2(mix=mix3_dc, clean=clean3_dc, estimate=est3_dc, key="dc"); b2.final()
r_all = read_csv(TMP + "_all2.csv")[0]; r_w = read_csv(TMP + "_w2.csv")[0]
print(f"  加 +5 直流后 si-snr: ALLMetricsTracker(zero_mean=True)={float(r_all['si-snr']):.3f} dB, "
      f"MetricsTracker(zero_mean=False)={float(r_w['si-snr']):.3f} dB")
print(f"  同输入换尺度 si-snr(ALL): x1={float(r_all['si-snr']):.3f} dB")

print("\n[4] SPlitMetricsTracker（splitwrapper.py，要求 n_src=3：2 目标 + 1 干扰，未在评测入口使用）")
c = SPlitMetricsTracker(save_file=TMP + "_s.csv")
mix4 = torch.randn(T); clean4 = torch.randn(3, T); est4 = clean4 + 0.3 * torch.randn(3, T)
c(mix=mix4, clean=clean4, estimate=est4, key="utt_003"); c.final()
rows = read_csv(TMP + "_s.csv")
print("  CSV columns:", list(rows[0].keys()))
print("  per-utt row :", {k: (round(float(v), 3) if k != "snt_id" else v) for k, v in rows[0].items()})
print(f"  final 行数 = {len(rows)} (只有 avg，无 std)")
print("\nDONE")
