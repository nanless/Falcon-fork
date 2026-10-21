"""Build a synthetic LRS2/LRS3-layout data tree + near-source wavs for data-pipeline verification.

Layout (matches the conventions hard-coded in DataPreProcess/process_lrs23.py):
    data/{tr,cv,tt}/{mix,target,mouths}/...
    data/tr/near/*.wav                      (near-field interferer dry sources, referenced by meta)
    data/tr/mix/<stem>_meta.json            (RealSSA-style per-mixture meta: {"components": [...]})
"""
import json
import os
import shutil

import numpy as np
import soundfile as sf
import torch

SR = 16000
ROOT = "/tmp/dp-verify"
DATA = os.path.join(ROOT, "data")


def wav(path, n, kind="noise", seed=0, gain=0.5):
    rs = np.random.RandomState(seed)
    if kind == "noise":
        x = rs.randn(n).astype(np.float32) * gain
    elif kind == "ramp":
        x = (np.arange(n, dtype=np.float32) / n) * gain
    else:
        raise ValueError(kind)
    sf.write(path, x, SR)
    return n


def mouth(path, n_frames, value_fn=None, channels=1, hw=96):
    if value_fn is None:
        value_fn = lambda f: float(f)  # frame f is filled with value f (easy alignment checks)
    frames = np.stack([
        np.full((hw, hw), value_fn(f), dtype=np.float32) for f in range(n_frames)
    ])  # (T, 96, 96)
    if channels == 1:
        t = torch.from_numpy(frames[:, :, :, None])  # (T, 96, 96, 1) channel-last, like dataset expects
    else:
        t = torch.from_numpy(np.stack([frames, frames * 0.5, frames * 0.25], -1).astype(np.float32))
    torch.save(t, path)


def meta(path, mix_stem, near_files):
    comps = []
    for p, typ in near_files:
        comps.append({"type": typ, "path": p})
    with open(path, "w") as f:
        json.dump({"mixture": mix_stem, "components": comps}, f, indent=2)


def main():
    if os.path.isdir(DATA):
        shutil.rmtree(DATA)
    for split in ["tr", "cv", "tt"]:
        for sub in ["mix", "target", "mouths"]:
            os.makedirs(os.path.join(DATA, split, sub), exist_ok=True)
    os.makedirs(os.path.join(DATA, "tr", "near"), exist_ok=True)

    # ---- near-field interferers referenced by the tr meta files ----
    n1 = os.path.join(DATA, "tr", "near", "u0_n1.wav")
    n2 = os.path.join(DATA, "tr", "near", "u0_n2.wav")
    far = os.path.join(DATA, "tr", "near", "u0_far.wav")
    short = os.path.join(DATA, "tr", "near", "u4_short.wav")
    ok = os.path.join(DATA, "tr", "near", "u4_ok.wav")
    wav(n1, 48000, "ramp", seed=1)             # 3.0 s
    wav(n2, 48000, "ramp", seed=2)
    wav(far, 48000, "noise", seed=3)
    wav(short, 20000, "ramp", seed=4)          # < 32000 -> must be skipped by the dataset
    wav(ok, 48000, "ramp", seed=5)

    # ---- tr: u0..u5 (u5 is shorter than the 2 s segment -> must be dropped) ----
    tr_specs = {"u0": 3.0, "u1": 3.0, "u2": 3.0, "u3": 3.0, "u4": 3.0, "u5": 1.0}
    for i, (stem, dur) in enumerate(tr_specs.items()):
        n = int(dur * SR)
        wav(os.path.join(DATA, "tr", "mix", stem + ".wav"), n, "noise", seed=100 + i)
        wav(os.path.join(DATA, "tr", "target", stem + ".wav"), n, "noise", seed=200 + i)
        mouth(os.path.join(DATA, "tr", "mouths", stem + ".pt"), int(dur * 25))

    meta(os.path.join(DATA, "tr", "mix", "u0_meta.json"), "u0",
         [(n1, "near"), (n2, "near"), (far, "far")])
    # u1: no meta at all -> generate_near_json falls back to the mixture itself
    meta(os.path.join(DATA, "tr", "mix", "u2_meta.json"), "u2", [(far, "far")])
    meta(os.path.join(DATA, "tr", "mix", "u3_meta.json"), "u3",
         [(os.path.join(DATA, "tr", "near", "u3_missing.wav"), "near")])
    meta(os.path.join(DATA, "tr", "mix", "u4_meta.json"), "u4", [(short, "near"), (ok, "near")])
    # u5: no meta

    # ---- cv / tt: 2.5 s utterances, no meta (fallback) ----
    for split, d in [("cv", 2.5), ("tt", 2.5)]:
        for i in range(2):
            stem = f"{split}{i}"
            n = int(d * SR)
            wav(os.path.join(DATA, split, "mix", stem + ".wav"), n, "noise", seed=300 + i)
            wav(os.path.join(DATA, split, "target", stem + ".wav"), n, "noise", seed=400 + i)
            mouth(os.path.join(DATA, split, "mouths", stem + ".pt"), int(d * 25))

    # ---- special fixtures ----
    # (a) 3-channel mouth tensor, to check the dataset is channel-last/3-channel agnostic
    wav(os.path.join(DATA, "tr", "mix", "u0copy.wav"), 48000, "noise", seed=7)
    wav(os.path.join(DATA, "tr", "target", "u0copy.wav"), 48000, "noise", seed=8)
    mouth(os.path.join(DATA, "tr", "mouths", "u0copy.pt"), 75, channels=3)
    # (b) gradient frames, to verify the val CenterCrop(88,88) == [4:92, 4:92]
    grad = np.stack([
        np.arange(96)[:, None] * 100.0 + np.arange(96)[None, :] for _ in range(75)
    ]).astype(np.float32)
    torch.save(torch.from_numpy(grad[:, :, :, None]), os.path.join(DATA, "tr", "mouths", "u0grad.pt"))
    sf.write(os.path.join(DATA, "tr", "mix", "u0grad.wav"), np.zeros(48000, np.float32), SR)
    sf.write(os.path.join(DATA, "tr", "target", "u0grad.wav"), np.zeros(48000, np.float32), SR)

    print("fake data written under", DATA)
    for split in ["tr", "cv", "tt"]:
        for sub in ["mix", "target", "mouths"]:
            print(split, sub, len(os.listdir(os.path.join(DATA, split, sub))),
                  "files;", "first:", sorted(os.listdir(os.path.join(DATA, split, sub)))[:3])


if __name__ == "__main__":
    main()
