import hashlib, importlib.util
from pathlib import Path
import numpy as np
from scipy.signal import correlate, correlation_lags
p = Path("/root/code/github_repos/Falcon-fork/tools/repro/phase2/protocol_check.py")
spec = importlib.util.spec_from_file_location("falcon_protocol_check", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r = np.zeros(64); r[15] = 1.0
e = np.zeros(64); e[20] = 1.0
lag, score0, score_script = m._best_lag(r, e, max_lag=10)
corrected = np.concatenate([e[lag:], np.zeros(lag)])
print("protocol_sha256", hashlib.sha256(p.read_bytes()).hexdigest())
print("ref_peak", int(np.argmax(r)), "estimate_peak", int(np.argmax(e)), "lag", lag)
print("script_shifted_peak", int(np.argmax(np.r_[np.zeros(lag), e[:-lag]])), "correct_shifted_peak", int(np.argmax(corrected)))
print("si_sdr_before", score0, "script_after", score_script, "correct_after", m.my_si_sdr(r, corrected))
assert lag == 5 and np.argmax(corrected) == 15
