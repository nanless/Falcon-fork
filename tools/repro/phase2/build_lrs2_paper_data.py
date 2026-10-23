#!/usr/bin/env python3
"""构建 LRS2-2Mix “论文配方” 训练数据：双向 target（s1/s2 随机对调）+ near 负样本。

动机：论文 LRS2-2Mix 列（SI-SNRi 14.1）需要：
- 双向 target：每条 mixture 同时构造 (target=s1, mouth=s1, near=s2) 与 (target=s2, mouth=s2, near=s1)，
  训练时数据量翻倍，迫使模型按“嘴唇”而不是“谁更响”（LRS2-2Mix 中 s1 恒更响）选目标；
- near 负样本：用另一说话人（干扰源）作为 NASS 对比损失的负样本（该数据集无 near-field 源，这是可用替代，
  报告中必须声明）。

输入：`--src /root/data/lrs2_2mix/lrs2_rebuild`（HF 发布包解压目录）
输出：`--out`（默认 /root/data/falcon_lrs2_2mix_paper）：
  {tr,cv,tt}/mix.json|target.json|near.json   # 每个 split 2N 条
  {tr,cv,tt}/mouths/<speaker_id>.pt            # uint8 (50,96,96,3)，按 utterance id 去重
  conversion_stats.json
音频不复制：JSON 里直接引用原始 `audio/wav16k/min/<split>/{mix,s1,s2}/*.wav`
（如需自包含数据集加 --copy-wavs）。

用法：
  python build_lrs2_paper_data.py                      # 全量
  python build_lrs2_paper_data.py --limit 200 --jobs 8 # 小样本验证
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SRC = Path("/root/data/lrs2_2mix/lrs2_rebuild")
OUT = Path("/root/data/falcon_lrs2_2mix_paper")


def parse_mix_name(stem: str) -> tuple[str, str]:
    """<id1>_<utt1>_<snr1>_<id2>_<utt2>_<snr2> -> (id1, id2)."""
    t = stem.split("_")
    if len(t) != 6:
        raise ValueError(f"unexpected mixture name: {stem}")
    return f"{t[0]}_{t[1]}", f"{t[3]}_{t[4]}"


def mouth_to_pt(src_npz: Path, dst_pt: Path) -> bool:
    if dst_pt.exists() and dst_pt.stat().st_size > 0:
        return True
    data = np.load(src_npz)["data"]
    if data.ndim != 3:
        return False
    t = torch.from_numpy(np.stack([data] * 3, axis=-1))  # (T,96,96,3) uint8
    tmp = dst_pt.with_suffix(".pt.tmp")
    torch.save(t, tmp)
    tmp.replace(dst_pt)
    return True


def build_split(src: Path, out: Path, split: str, limit: int, jobs: int, copy_wavs: bool) -> dict:
    wav_root = src / "audio" / "wav16k" / "min" / split
    mixes = sorted((wav_root / "mix").glob("*.wav"))
    if limit:
        mixes = mixes[:limit]
    (out / split / "mouths").mkdir(parents=True, exist_ok=True)
    mouth_src = src / "mouths"
    tasks = []
    entries = []
    for wav in mixes:
        stem = wav.stem
        try:
            id1, id2 = parse_mix_name(stem)
        except ValueError as e:
            print(f"[skip] {e}")
            continue
        n = sf.info(str(wav)).frames
        s1 = wav_root / "s1" / f"{stem}.wav"
        s2 = wav_root / "s2" / f"{stem}.wav"
        if not s1.exists() or not s2.exists():
            print(f"[skip] missing source for {stem}")
            continue
        mp = out / split / "mouths" / f"{id1}.pt"
        mp2 = out / split / "mouths" / f"{id2}.pt"
        tasks.append((mouth_src / f"{id1}.npz", mp))
        tasks.append((mouth_src / f"{id2}.npz", mp2))
        if copy_wavs:
            mix_dst = out / split / "mix" / wav.name
            tgt_dst = out / split / "target"
            near_dst = out / split / "near"
            for d in (mix_dst.parent, tgt_dst, near_dst):
                d.mkdir(parents=True, exist_ok=True)
            if not mix_dst.exists():
                shutil.copyfile(wav, mix_dst)
            for src_wav, dstdir in ((s1, tgt_dst), (s2, tgt_dst), (s1, near_dst), (s2, near_dst)):
                d = dstdir / src_wav.name
                if not d.exists():
                    shutil.copyfile(src_wav, d)
            mix_path, s1p, s2p = str(mix_dst), str(tgt_dst / f"{stem}.wav"), str(near_dst / f"{stem}.wav")
        else:
            mix_path, s1p, s2p = str(wav), str(s1), str(s2)
        # A: target=s1, mouth=s1, near=s2
        entries.append({"mix": [mix_path, n], "target": [s1p, str(mp), n], "near": [[s2p, n]]})
        # B: target=s2, mouth=s2, near=s1
        entries.append({"mix": [mix_path, n], "target": [s2p, str(mp2), n], "near": [[s1p, n]]})

    # 去重：同一说话人 id 会出现在很多 mixture 里，重复任务会并发写同一个 .pt（竞态）
    uniq = {}
    for sp, dp in tasks:
        uniq[str(dp)] = (sp, dp)
    tasks = list(uniq.values())
    fails = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(mouth_to_pt, s, d) for s, d in tasks]
        for fut in as_completed(futs):
            try:
                if not fut.result():
                    fails.append("bad-mouth")
            except Exception as e:  # noqa: BLE001
                fails.append(f"{type(e).__name__}:{e}")
    entries = [e for e in entries if Path(e["target"][1]).exists()]
    mix_json = [e["mix"] for e in entries]
    tgt_json = [e["target"] for e in entries]
    near_json = [e["near"] for e in entries]
    for name, obj in (("mix.json", mix_json), ("target.json", tgt_json), ("near.json", near_json)):
        json.dump(obj, open(out / split / name, "w"), indent=0)
    stats = {"split": split, "mixtures": len(mixes), "entries": len(entries),
             "unique_mouths": len({t[1] for t in tasks}), "mouth_fails": len(fails),
             "elapsed_s": None}
    print(f"[{split}] mixtures={len(mixes)} entries={len(entries)} mouths={stats['unique_mouths']} fails={len(fails)}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--splits", default="tr,cv,tt")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--copy-wavs", action="store_true")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"src not found: {src}")
    t0 = time.time()
    stats = []
    for split in args.splits.split(","):
        t = time.time()
        st = build_split(src, out, split.strip(), args.limit, args.jobs, args.copy_wavs)
        st["elapsed_s"] = round(time.time() - t, 1)
        stats.append(st)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"src": str(src), "out": str(out), "copy_wavs": args.copy_wavs,
               "splits": stats, "elapsed_s": round(time.time() - t0, 1),
               "note": "bidirectional target(s1/s2); near=other speaker; audio references raw src unless --copy-wavs"}
    json.dump(summary, open(out / "conversion_stats.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
