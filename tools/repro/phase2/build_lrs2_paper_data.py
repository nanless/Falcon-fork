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
（如需自包含数据集加 --copy-wavs；复制布局见下）。

``--copy-wavs`` 布局（review A5-1 修复）::

    <out>/<split>/mix/<stem>.wav
    <out>/<split>/target/<stem>.s1.wav       # target=s1 条目
    <out>/<split>/target/<stem>.s2.wav       # target=s2 条目
    <out>/<split>/near/<stem>.s1.wav         # 作 target=s2 时的负样本（s1 波形）
    <out>/<split>/near/<stem>.s2.wav         # 作 target=s1 时的负样本（s2 波形）

旧实现把 s1/s2 都拷成 `target/<stem>.wav`，第二次复制被 `if not exists()` 跳过，导致 target 与 near
指向同一文件、target 与 mouth 说话人不一致（静默错配）。现在 target/near 文件名带 `.s1`/`.s2` 后缀，
二者永不相等；构建后运行闸门 ``validate_build``（计数 / target≠near / mouth 说话人==target 说话人 /
near 非自引用覆盖 100%），任何不符打印细节并以 exit 2 失败。
"""
from __future__ import annotations

import argparse
import json
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
# 全量论文配方（无 --limit）的期望条数：每条 mixture 双向两条
PAPER_ENTRIES = {"tr": 40000, "cv": 10000, "tt": 6000}


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
            tgt_dir = out / split / "target"
            near_dir = out / split / "near"
            tgt_s1 = tgt_dir / f"{stem}.s1.wav"
            tgt_s2 = tgt_dir / f"{stem}.s2.wav"
            near_s1 = near_dir / f"{stem}.s1.wav"
            near_s2 = near_dir / f"{stem}.s2.wav"
            for d in (mix_dst.parent, tgt_dir, near_dir):
                d.mkdir(parents=True, exist_ok=True)
            # 每次都在独立文件上判断/写入：s1/s2 不再竞争同一个 target/<stem>.wav
            for src_wav, dst in ((wav, mix_dst), (s1, tgt_s1), (s2, tgt_s2),
                                 (s1, near_s1), (s2, near_s2)):
                if not dst.exists():
                    shutil.copyfile(src_wav, dst)
            mix_path = str(mix_dst)
            s1p, s2p = str(tgt_s1), str(tgt_s2)
            near_of_s1, near_of_s2 = str(near_s2), str(near_s1)  # target=s1 -> near=s2；反之 near=s1
        else:
            mix_path, s1p, s2p = str(wav), str(s1), str(s2)
            near_of_s1, near_of_s2 = str(s2), str(s1)
        # A: target=s1, mouth=s1, near=s2
        entries.append({"mix": [mix_path, n], "target": [s1p, str(mp), n],
                        "near": [[near_of_s1, n]]})
        # B: target=s2, mouth=s2, near=s1
        entries.append({"mix": [mix_path, n], "target": [s2p, str(mp2), n],
                        "near": [[near_of_s2, n]]})

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


# ---------------------------------------------------------------------------
# 构建后闸门（review A5-1）
# ---------------------------------------------------------------------------
def expected_entries(split: str, mixtures: int, limit: int, overrides: dict[str, int]) -> int:
    """每个 split 期望条目数：显式 override > --limit 推导 > 全量论文配方 2N（tr 40000/cv 10000/tt 6000）。"""
    if split in overrides:
        return overrides[split]
    if limit:
        return 2 * mixtures
    return PAPER_ENTRIES.get(split, 2 * mixtures)


def _target_speaker_id(stem: str, target_path: Path, copy_wavs: bool) -> tuple[str | None, str | None]:
    """返回 (target 说话人 id, 问题描述)；id 来自 stem recipe 与 target 文件名/目录的交叉检查。"""
    id1, id2 = parse_mix_name(stem)
    name = Path(target_path).name
    if name.endswith(".s1.wav"):
        return id1, None
    if name.endswith(".s2.wav"):
        return id2, None
    if not copy_wavs:
        parent = Path(target_path).parent.name
        if parent == "s1":
            return id1, None
        if parent == "s2":
            return id2, None
    return None, f"cannot infer target speaker from {target_path}"


def validate_build(out: Path, splits: list[str], expected: dict[str, int],
                   copy_wavs: bool, max_details: int = 10) -> list[str]:
    """逐 split 校验产物；返回问题列表（空 = 通过）。"""
    problems: list[str] = []
    for split in splits:
        sdir = out / split
        try:
            mix_json = json.loads((sdir / "mix.json").read_text())
            target_json = json.loads((sdir / "target.json").read_text())
            near_json = json.loads((sdir / "near.json").read_text())
        except FileNotFoundError as exc:
            problems.append(f"[{split}] missing JSON: {exc}")
            continue
        want = expected.get(split)
        if not (len(mix_json) == len(target_json) == len(near_json)):
            problems.append(f"[{split}] JSON lengths differ: mix={len(mix_json)} "
                            f"target={len(target_json)} near={len(near_json)}")
            continue
        if want is not None and len(mix_json) != want:
            problems.append(f"[{split}] entry count {len(mix_json)} != expected {want}")
        split_problems: list[str] = []
        non_self = 0
        for idx, (mix_entry, tgt_entry, near_entry) in enumerate(
                zip(mix_json, target_json, near_json)):
            mix_path = Path(mix_entry[0])
            tgt_path = Path(tgt_entry[0])
            mouth_path = Path(tgt_entry[1])
            stem = mix_path.stem
            tag = f"[{split}:{idx}:{stem[:28]}]"
            if not mix_path.is_file():
                split_problems.append(f"{tag} missing mix {mix_path}")
            if not tgt_path.is_file():
                split_problems.append(f"{tag} missing target {tgt_path}")
            if not mouth_path.is_file():
                split_problems.append(f"{tag} missing mouth {mouth_path}")
            if len(near_entry) != 1:
                split_problems.append(f"{tag} near K={len(near_entry)} (expected 1)")
            else:
                near_path = Path(near_entry[0][0])
                n_near = near_entry[0][1]
                if near_path.resolve() == tgt_path.resolve():
                    split_problems.append(f"{tag} near == target (self-reference): {near_path}")
                else:
                    non_self += 1
                if not near_path.is_file():
                    split_problems.append(f"{tag} missing near {near_path}")
                if n_near != mix_entry[1]:
                    split_problems.append(f"{tag} near n_samples {n_near} != mix {mix_entry[1]}")
            target_id, why = _target_speaker_id(stem, tgt_path, copy_wavs)
            if why:
                split_problems.append(f"{tag} {why}")
            elif target_id is not None and mouth_path.stem != target_id:
                split_problems.append(f"{tag} mouth id {mouth_path.stem} != target id {target_id}")
        total = len(mix_json)
        coverage = (100.0 * non_self / total) if total else 0.0
        if total and non_self != total:
            split_problems.append(f"[{split}] non-self near coverage {non_self}/{total} "
                                  f"({coverage:.2f}%) != 100%")
        if split_problems:
            problems.extend(split_problems[:max_details])
            if len(split_problems) > max_details:
                problems.append(f"[{split}] ... {len(split_problems) - max_details} more problem(s)")
        else:
            print(f"[gate] {split}: OK entries={total} K=1 non-self coverage={coverage:.2f}%")
    return problems


def parse_expect_entries(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--expect-entries 需为 split=count 形式，得到 {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError:
            raise SystemExit(f"--expect-entries 计数不是整数：{item!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--splits", default="tr,cv,tt")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--copy-wavs", action="store_true")
    ap.add_argument("--expect-entries", default=None,
                    help="覆盖闸门期望条数，如 'tr=4,cv=2'（小样本验证用）")
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"src not found: {src}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    overrides = parse_expect_entries(args.expect_entries)
    t0 = time.time()
    stats = []
    for split in splits:
        t = time.time()
        st = build_split(src, out, split, args.limit, args.jobs, args.copy_wavs)
        st["elapsed_s"] = round(time.time() - t, 1)
        stats.append(st)
    out.mkdir(parents=True, exist_ok=True)
    expected = {st["split"]: expected_entries(st["split"], st["mixtures"], args.limit, overrides)
                for st in stats}
    print(f"== gate: expected entries {expected}")
    problems = validate_build(out, splits, expected, args.copy_wavs)
    summary = {"src": str(src), "out": str(out), "copy_wavs": args.copy_wavs,
               "splits": stats, "elapsed_s": round(time.time() - t0, 1),
               "expected_entries": expected, "gate": {"ok": not problems, "problems": problems},
               "note": "bidirectional target(s1/s2); near=other speaker; "
                       "copy-wavs uses target/<stem>.s1|s2.wav + near/<stem>.s1|s2.wav"}
    json.dump(summary, open(out / "conversion_stats.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    if problems:
        print("== gate FAILED (exit 2) ==")
        for p in problems:
            print("  -", p)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
