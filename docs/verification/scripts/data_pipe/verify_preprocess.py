"""Verify DataPreProcess scripts + check_avspeech_dataset.py with subprocess runs.

Run from repo root:
  PYTHONPATH=/tmp/dp-verify/stubs:<repo> /root/.venvs/falcon/bin/python /tmp/dp-verify/verify_preprocess.py
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = "/root/code/github_repos/Falcon-fork"
PY = "/root/.venvs/falcon/bin/python"
ROOT = "/tmp/dp-verify"
ENV = dict(os.environ, PYTHONPATH=":".join([os.path.join(REPO, "docs", "verification", "stubs", "data_pipe"), REPO]))
FAIL = 0


def p(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    if not ok:
        FAIL += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def run(args, expect_ok=True):
    r = subprocess.run([PY] + args, cwd=REPO, env=ENV, capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print("STDERR:", r.stderr[-800:])
    return r


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- 1. reentrancy
r = run(["DataPreProcess/process_lrs23.py", "--in_audio_dir", f"{ROOT}/data", "--out_dir", f"{ROOT}/index"])
p("1 process_lrs23.py exits 0", r.returncode == 0, r.returncode)
hashes1 = {f: sha(f) for f in glob.glob(f"{ROOT}/index/*/*.json")}
r2 = run(["DataPreProcess/process_lrs23.py", "--in_audio_dir", f"{ROOT}/data", "--out_dir", f"{ROOT}/index"])
hashes2 = {f: sha(f) for f in glob.glob(f"{ROOT}/index/*/*.json")}
p("2 re-run is idempotent (byte-identical JSONs, overwrite in place)", hashes1 == hashes2,
  f"{len(hashes1)} files")
mix = json.load(open(f"{ROOT}/index/tr/mix.json"))
tgt = json.load(open(f"{ROOT}/index/tr/target.json"))
near = json.load(open(f"{ROOT}/index/tr/near.json"))
p("3 mix.json schema [[abs_wav_path, n_samples], ...]", len(mix[0]) == 2 and len(mix) == 8
  and os.path.isabs(mix[0][0]), str(mix[0]))
p("4 target.json schema [[abs_wav_path, abs_mouth_pt_path, n_samples], ...]",
  len(tgt[0]) == 3 and tgt[0][1].endswith(".pt") and "/mouths/" in tgt[0][1], str(tgt[0]))
p("5 near.json schema [ [[near_path, n_samples], ...], ... ] aligned to mix.json",
  len(near) == len(mix) and all(isinstance(e, list) and len(e) >= 1 for e in near),
  f"len={len(near)}; per-sample K={[len(e) for e in near]}")
p("6 only type=='near' components indexed (far-field excluded)",
  [os.path.basename(x[0]) for x in near[0]] == ["u0_n1.wav", "u0_n2.wav"], str(near[0]))
fb = [i for i, (e, m) in enumerate(zip(near, mix)) if len(e) == 1 and e[0][0] == m[0]]
p("7 missing meta / missing file / only-far -> fallback entry == [mix_path, mix_n]",
  fb == [1, 2, 3, 4, 5, 7], f"fallback indices={fb} (u0copy,u0grad,u1,u2,u3,u5)")
p("8 target mouth path = in_audio_dir/tr/mouths/<stem>.pt (string replace target->mouths)",
  tgt[0][1] == f"{ROOT}/data/tr/target/u0.wav".replace("target", "mouths").replace(".wav", ".pt"))

# ---------------------------------------------------------------- 2. missing input dir
r = run(["DataPreProcess/process_lrs23.py", "--in_audio_dir", f"{ROOT}/no_such_dir",
         "--out_dir", f"{ROOT}/index_x"], expect_ok=False)
p("9 missing input dir -> uncaught FileNotFoundError (non-zero exit)",
  r.returncode != 0 and "FileNotFoundError" in r.stderr, r.stderr.strip().splitlines()[-1][:120])

# ---------------------------------------------------------------- 3. empty split crash
empty = f"{ROOT}/empty_data"
shutil.rmtree(empty, ignore_errors=True)
for split in ["tr", "cv", "tt"]:
    os.makedirs(f"{empty}/{split}/mix")
    os.makedirs(f"{empty}/{split}/target")
r = run(["DataPreProcess/process_lrs23.py", "--in_audio_dir", empty, "--out_dir", f"{ROOT}/index_empty"],
        expect_ok=False)
written = os.path.exists(f"{ROOT}/index_empty/tr/mix.json") and os.path.exists(f"{ROOT}/index_empty/tr/near.json")
p("10 empty split dirs: JSONs written as [] but script crashes on min([]) in stats print",
  r.returncode != 0 and written and "min() arg is an empty sequence" in r.stderr,
  f"rc={r.returncode}, mix.json exists={os.path.exists(f'{ROOT}/index_empty/tr/mix.json')}, "
  f"near.json exists={os.path.exists(f'{ROOT}/index_empty/tr/near.json')}")

# ---------------------------------------------------------------- 4. standalone generate_near_json
INDEX2 = f"{ROOT}/index2"
shutil.rmtree(INDEX2, ignore_errors=True)
run(["DataPreProcess/process_lrs23.py", "--in_audio_dir", f"{ROOT}/data", "--out_dir", INDEX2])
base_near = {f: sha(f) for f in glob.glob(f"{INDEX2}/*/near.json")}
r = run(["DataPreProcess/generate_near_json.py", "--data_root", f"{ROOT}/data", "--out_root", INDEX2,
         "--sample_rate", "16000"])
p("11 standalone generate_near_json exits 0", r.returncode == 0, r.returncode)
p("12 standalone output identical to process_lrs23-embedded generation",
  base_near == {f: sha(f) for f in glob.glob(f"{INDEX2}/*/near.json")})
r = run(["DataPreProcess/generate_near_json.py", "--data_root", f"{ROOT}/data", "--out_root", INDEX2,
         "--sample_rate", "8000"])
p("13 --sample_rate is accepted but unused (output unchanged)", r.returncode == 0
  and base_near == {f: sha(f) for f in glob.glob(f"{INDEX2}/*/near.json")})
INDEX3 = f"{ROOT}/index3"
shutil.rmtree(INDEX3, ignore_errors=True)
os.makedirs(f"{INDEX3}/tr")
shutil.copy(f"{INDEX2}/tr/mix.json", f"{INDEX3}/tr/mix.json")
r = run(["DataPreProcess/generate_near_json.py", "--data_root", f"{ROOT}/data", "--out_root", INDEX3])
p("14 missing <split>/mix.json -> prints 'Skipping <split>' and continues",
  "Skipping cv" in r.stdout and "Skipping tt" in r.stdout and os.path.exists(f"{INDEX3}/tr/near.json"))

# ---------------------------------------------------------------- 5. silent misalignment
mis = f"{ROOT}/data_mismatch"
shutil.rmtree(mis, ignore_errors=True)
import numpy as np
import soundfile as sf

for split in ["tr", "cv", "tt"]:
    for sub in ["mix", "target", "mouths"]:
        os.makedirs(f"{mis}/{split}/{sub}")
# tr: mix has a.wav + c.wav; target has b.wav + c.wav -> same count, different sorted names
for sub, names in [("mix", ["a", "c"]), ("target", ["b", "c"])]:
    for name in names:
        n = 48000
        sf.write(f"{mis}/tr/{sub}/{name}.wav", np.zeros(n, np.float32), 16000)
for name in ["b", "c"]:
    import torch
    torch.save(torch.zeros(75, 96, 96, 1), f"{mis}/tr/mouths/{name}.pt")
for split in ["cv", "tt"]:
    for name in ["x"]:
        sf.write(f"{mis}/{split}/mix/{name}.wav", np.zeros(40000, np.float32), 16000)
        sf.write(f"{mis}/{split}/target/{name}.wav", np.zeros(40000, np.float32), 16000)
        import torch
        torch.save(torch.zeros(62, 96, 96, 1), f"{mis}/{split}/mouths/{name}.pt")
r = run(["DataPreProcess/process_lrs23.py", "--in_audio_dir", mis, "--out_dir", f"{ROOT}/index_mismatch"])
m = json.load(open(f"{ROOT}/index_mismatch/tr/mix.json"))
t = json.load(open(f"{ROOT}/index_mismatch/tr/target.json"))
p("15 dir-name mismatch with equal file count -> separate sorted orders -> silent index misalignment "
  "(no check that mix[i] and target[i] share the stem)",
  len(m) == len(t) == 2
  and os.path.basename(m[0][0]).replace(".wav", "") != os.path.basename(t[0][0]).replace(".wav", ""),
  f"mix[0]={os.path.basename(m[0][0])}, target[0]={os.path.basename(t[0][0])}, rc={r.returncode}")
from look2hear.datas.avspeech_dataset_near import AVSpeechTrainDatasetWithNear  # noqa: E402
sys.path.insert(0, REPO)
_DP_STUB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "docs", "verification", "stubs", "data_pipe"))
sys.path.insert(0, _DP_STUB)
sys.path.insert(0, "/tmp/dp-verify/stubs")  # optional fallback
ds = AVSpeechTrainDatasetWithNear(json_dir=f"{ROOT}/index_mismatch/tr", sample_rate=16000,
                                  segment=2.0, samples=1, use_near=True)
stems_m = [os.path.basename(a[0])[:-4] for a in ds.mix]
stems_s = [os.path.basename(b[0])[:-4] for b in ds.sources]
p("16 dataset loads misaligned index without any error (pairing silently wrong)",
  stems_m != stems_s, f"mix={stems_m}, target={stems_s}")

# ---------------------------------------------------------------- 6. check script
r = run(["look2hear/datas/check_avspeech_dataset.py"], expect_ok=False)
p("17 check_avspeech_dataset.py is not runnable as-is: hard-coded '${DATA_ROOT}/...' path",
  r.returncode != 0 and ("FileNotFoundError" in r.stderr or "No such file" in r.stderr),
  (r.stderr.strip().splitlines() or [""])[-1][:160])

# ---------------------------------------------------------------- 7. CLI defaults
r = run(["DataPreProcess/process_lrs23.py", "--help"])
p("18 process_lrs23.py --help lists --in_audio_dir/--out_dir (defaults '--' NOT shown by argparse)",
  "--in_audio_dir" in r.stdout and "--out_dir" in r.stdout and "default" not in r.stdout,
  [ln.strip() for ln in r.stdout.splitlines() if "--" in ln][:4])
r = run(["DataPreProcess/generate_near_json.py", "--help"])
p("19 generate_near_json.py --help lists --data_root/--out_root/--sample_rate",
  all(k in r.stdout for k in ["--data_root", "--out_root", "--sample_rate"]))
print(f"\n== TOTAL FAILURES: {FAIL}")
sys.exit(1 if FAIL else 0)
