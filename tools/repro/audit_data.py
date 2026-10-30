#!/usr/bin/env python3
"""审计 Falcon folder-format 数据（mix/target/near + mouths）是否满足训练契约。

用途：数据转换（LRS2-2Mix / RealSSA-Sim）之后、训练之前的一道只读门禁。逐 split 读
``mix.json`` / ``target.json`` / ``near.json``，统计 K 分布与"自引用哨兵"（near 路径 == mix
路径，Dataset 会静默跳过 → 对比损失失效）比例；抽样（默认 200 条/split，``--full`` 全量）
校验音频存在/可读/16 kHz、target 与 mix 等长、mouths ``.pt`` 形状 ``[T,96,96,3]`` 且
``T == n_samples/640``、near wav 存在/可读/16 kHz。

退出码：
  0  全部通过（无缺失、fallback == 0、抽样文件契约全过）
  1  有任何问题（JSON 缺失/长度不一致/文件缺失不可读/near fallback > 0/mouth 形状不符等）

用法：
  python tools/repro/audit_data.py /root/data/falcon_lrs2_2mix
  python tools/repro/audit_data.py /root/data/falcon_lrs2_2mix_paper --full --json /tmp/audit.json
  python tools/repro/audit_data.py /root/data/falcon_realssa_sim --sample 50   # 预期 exit 1（near 全 fallback）
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import soundfile as sf
import torch

SR = 16000
FRAME = SR // 25  # 640 samples per 25 fps video frame
MOUTH_TAIL = (96, 96, 3)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def audit_split(split_dir: Path, split: str, sample: int, full: bool, seed: int,
                max_details: int = 10) -> dict:
    """审计一个 split；返回统计 + problems 列表。"""
    res: dict = {"split": split, "entries": 0, "k_hist": {}, "empty": 0, "fallback": 0,
                 "coverage_pct": 0.0, "sampled": 0, "problems": [], "missing_json": False}
    problems: list[str] = res["problems"]
    try:
        mix_json = load_json(split_dir / "mix.json")
        target_json = load_json(split_dir / "target.json")
        near_json = load_json(split_dir / "near.json")
    except FileNotFoundError as exc:
        res["missing_json"] = True
        problems.append(f"[{split}] missing JSON: {exc}")
        return res

    lengths = (len(mix_json), len(target_json), len(near_json))
    if len(set(lengths)) != 1:
        problems.append(f"[{split}] JSON lengths differ: mix={lengths[0]} target={lengths[1]} "
                        f"near={lengths[2]}")
        return res
    n = lengths[0]
    res["entries"] = n
    if n == 0:
        problems.append(f"[{split}] empty split (0 entries)")

    non_self = 0
    k_hist: collections.Counter = collections.Counter()
    for i in range(n):
        mix_path = mix_json[i][0]
        near_entry = near_json[i]
        k_hist[len(near_entry)] += 1
        if not near_entry:
            res["empty"] += 1
            continue
        is_self = any(Path(p).resolve() == Path(mix_path).resolve() for p, _ in near_entry)
        if is_self:
            res["fallback"] += 1
        else:
            non_self += 1
    res["k_hist"] = {str(k): v for k, v in sorted(k_hist.items())}
    res["coverage_pct"] = round(100.0 * non_self / n, 2) if n else 0.0
    if res["fallback"]:
        problems.append(f"[{split}] near fallback (self-reference sentinel) {res['fallback']}/{n} "
                        f"({100.0 * res['fallback'] / n:.2f}%) -> K=0 for those samples")
    if res["empty"]:
        problems.append(f"[{split}] near empty list for {res['empty']}/{n} samples")

    # ── 文件级抽样校验 ─────────────────────────────────────────────────────
    indices = list(range(n)) if full else (
        random.Random(seed).sample(range(n), min(sample, n)) if n else [])
    res["sampled"] = len(indices)
    file_problems: list[str] = []
    for i in indices:
        mix_path, mix_n = mix_json[i]
        tgt_path, mouth_path, tgt_n = target_json[i]
        tag = f"[{split}:{i}]"
        try:
            mix_info = sf.info(mix_path)
        except Exception as exc:  # noqa: BLE001
            file_problems.append(f"{tag} unreadable mix {mix_path}: {type(exc).__name__}: {exc}")
            continue
        if not Path(mix_path).is_file():
            file_problems.append(f"{tag} missing mix {mix_path}")
        if mix_info.samplerate != SR:
            file_problems.append(f"{tag} mix samplerate {mix_info.samplerate} != {SR}")
        if mix_info.frames != mix_n:
            file_problems.append(f"{tag} mix frames {mix_info.frames} != JSON {mix_n}")
        try:
            tgt_info = sf.info(tgt_path)
            if not Path(tgt_path).is_file():
                file_problems.append(f"{tag} missing target {tgt_path}")
            if tgt_info.samplerate != SR:
                file_problems.append(f"{tag} target samplerate {tgt_info.samplerate} != {SR}")
            if tgt_info.frames != mix_info.frames:
                file_problems.append(f"{tag} target frames {tgt_info.frames} != mix {mix_info.frames}")
        except Exception as exc:  # noqa: BLE001
            file_problems.append(f"{tag} unreadable target {tgt_path}: {type(exc).__name__}: {exc}")
        try:
            mouths = torch.load(mouth_path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(mouths):
                file_problems.append(f"{tag} mouths not a tensor: {mouth_path}")
            elif mouths.ndim != 4 or tuple(mouths.shape[1:]) != MOUTH_TAIL:
                file_problems.append(f"{tag} mouths shape {tuple(mouths.shape)} "
                                     f"(expected [T,96,96,3]) {mouth_path}")
            elif mix_info.frames % FRAME or mouths.shape[0] != mix_info.frames // FRAME:
                file_problems.append(f"{tag} mouth frames {mouths.shape[0]} != "
                                     f"n_samples/640 {mix_info.frames // FRAME}")
        except Exception as exc:  # noqa: BLE001
            file_problems.append(f"{tag} unreadable mouths {mouth_path}: {type(exc).__name__}: {exc}")
        for near_path, near_n in near_json[i]:
            if Path(near_path).resolve() == Path(mix_path).resolve():
                continue  # fallback sentinel already counted above
            try:
                near_info = sf.info(near_path)
                if not Path(near_path).is_file():
                    file_problems.append(f"{tag} missing near {near_path}")
                if near_info.samplerate != SR:
                    file_problems.append(f"{tag} near samplerate {near_info.samplerate} != {SR}")
                if near_info.frames != near_n:
                    file_problems.append(f"{tag} near frames {near_info.frames} != JSON {near_n}")
            except Exception as exc:  # noqa: BLE001
                file_problems.append(f"{tag} unreadable near {near_path}: {type(exc).__name__}: {exc}")
    if file_problems:
        problems.extend(file_problems[:max_details])
        if len(file_problems) > max_details:
            problems.append(f"[{split}] ... {len(file_problems) - max_details} more file problem(s)")
    return res


def fmt_k(res: dict) -> str:
    hist = res["k_hist"]
    if not hist:
        k = "n/a"
    else:
        ks = sorted(int(k) for k in hist)
        total = sum(hist.values())
        mean = sum(int(k) * v for k, v in hist.items()) / total
        k = f"min{ks[0]} max{ks[-1]} mean{mean:.2f} hist={hist}"
    return k


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="审计 Falcon folder-format 数据（K 分布 / fallback / wav / mouths）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("data_root", help="数据集根目录（含 tr/cv/tt）")
    ap.add_argument("--splits", default="tr,cv,tt")
    ap.add_argument("--sample", type=int, default=200, help="每 split 抽样条数（文件级校验）")
    ap.add_argument("--full", action="store_true", help="全量校验（不抽样）")
    ap.add_argument("--seed", type=int, default=0, help="抽样随机种子")
    ap.add_argument("--json", default=None, help="把审计结果写成 JSON")
    args = ap.parse_args(argv)

    root = Path(args.data_root).expanduser()
    if not root.is_dir():
        print(f"[audit] data_root not found: {root}")
        return 1
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    results = []
    failed = False
    for split in splits:
        res = audit_split(root / split, split, args.sample, args.full, args.seed)
        results.append(res)
        print(f"[{split}] entries={res['entries']} K={fmt_k(res)} "
              f"fallback={res['fallback']} coverage={res['coverage_pct']:.2f}% "
              f"sampled={res['sampled']}{' (full)' if args.full else ''}")
        if res["problems"]:
            failed = True
            for p in res["problems"]:
                print("   -", p)
    total_entries = sum(r["entries"] for r in results)
    total_fallback = sum(r["fallback"] for r in results)
    summary = {"data_root": str(root), "splits": {r["split"]: r for r in results},
               "ok": not failed and total_fallback == 0, "total_entries": total_entries,
               "total_fallback": total_fallback}
    print(f"[audit] total_entries={total_entries} total_fallback={total_fallback} "
          f"status={'OK' if summary['ok'] else 'FAILED'}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[audit] wrote {args.json}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
