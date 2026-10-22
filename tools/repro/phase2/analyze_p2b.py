#!/usr/bin/env python
"""Phase-2 B analysis: aggregate ``compare_variants.py`` outputs.

Reads the summary JSONs / per-utterance CSVs under ``--dir`` and prints

1. a metrics table (variant x protocol x visual);
2. paired comparisons on identical utterances (mean delta, std, win-rate), which is what
   answers "is big better than small?", "does windowing hurt?", "does the visual branch help?".

Usage::

    PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python tools/repro/phase2/analyze_p2b.py \
        --dir docs/verification/phase2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRIC = "si-snr_i"


def _read_csv(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["snt_id"] in ("avg", "std", ""):
                continue
            try:
                out[row["snt_id"]] = {k: float(v) for k, v in row.items() if k != "snt_id"}
            except ValueError:
                continue
    return out


def _pair(a: dict, b: dict, metric=METRIC):
    common = sorted(set(a) & set(b))
    diffs = [b[k][metric] - a[k][metric] for k in common]
    if not diffs:
        return None
    n = len(diffs)
    mean = sum(diffs) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in diffs) / n) if n > 1 else 0.0
    wins = sum(d > 0 for d in diffs)
    return {"n": n, "mean_delta": mean, "std": std, "wins": wins, "win_rate": wins / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("docs/verification/phase2"))
    ap.add_argument("--small-full-csv", type=Path,
                    default=Path("Experiments/checkpoint/Falcon/results/metrics_combined.csv"),
                    help="Phase-1 full tt (3000) metrics CSV of the small model, used for pairing "
                         "against the big full run without re-running small")
    args = ap.parse_args(argv)

    # Build the run list from per-utterance CSVs (robust when summary JSONs are overwritten by
    # concurrently running jobs); fall back to summary JSONs when no CSV is present.
    runs = []
    for csv_path in sorted(args.dir.glob("P2B-*.csv")):
        parts = csv_path.stem.split("-")           # P2B, <tag>, <variant>, <protocol>, <visual>
        if len(parts) != 5 or parts[0] != "P2B":
            continue
        _, tag, variant, protocol, visual = parts
        data = _read_csv(csv_path)
        if not data:
            continue
        n = len(data)
        vals = {m: [v[m] for v in data.values()] for m in ("si-snr", "si-snr_i", "sdr", "sdr_i", "pesq", "stoi")}
        mean = {m: sum(v) / len(v) for m, v in vals.items()}
        std = {m: math.sqrt(sum((x - mean[m]) ** 2 for x in v) / len(v)) for m, v in vals.items()}
        runs.append({
            "variant": variant, "protocol": protocol, "visual": visual, "n": n, "tag": tag,
            "si_snr": mean["si-snr"], "si_snr_i": mean["si-snr_i"],
            "sdr": mean["sdr"], "sdr_i": mean["sdr_i"], "pesq": mean["pesq"], "stoi": mean["stoi"],
            "si_snr_i_std": std["si-snr_i"], "sdr_i_std": std["sdr_i"], "csv": csv_path.name,
        })
    if not runs:  # fallback
        for p in sorted(args.dir.glob("P2B-summary-*.json")):
            runs.extend(json.loads(p.read_text()))
    # keep the largest n per (variant, protocol, visual)
    best: dict = {}
    for r in runs:
        key = (r["variant"], r["protocol"], r["visual"])
        if key not in best or r["n"] > best[key]["n"]:
            best[key] = r
    runs = list(best.values())

    print("## Metrics summary\n")
    print("| variant | protocol | visual | n | SI-SNR | SI-SNRi | SDR | SDRi | PESQ | STOI |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda x: (x["variant"], x["protocol"], x["visual"], -x["n"])):
        print(f"| {r['variant']} | {r['protocol']} | {r['visual']} | {r['n']} | "
              f"{r['si_snr']:.3f} | {r['si_snr_i']:.3f} | {r['sdr']:.3f} | {r['sdr_i']:.3f} | "
              f"{r['pesq']:.3f} | {r['stoi']:.3f} |")

    # pair definitions: (label, baseline row, candidate row)
    def find(variant, protocol, visual, n=None):
        cands = [r for r in runs if r["variant"] == variant and r["protocol"] == protocol
                 and r["visual"] == visual and (n is None or r["n"] == n)]
        return max(cands, key=lambda r: r["n"]) if cands else None

    pairs = [
        ("big vs small (whole, real)", find("small", "whole", "real"), find("big", "whole", "real")),
        ("small windowed vs whole", find("small", "whole", "real"), find("small", "windowed", "real")),
        ("big windowed vs whole", find("big", "whole", "real"), find("big", "windowed", "real")),
        ("small zero-mouth vs real (whole)", find("small", "whole", "real"), find("small", "whole", "zero")),
        ("small shuffled-mouth vs real (whole)", find("small", "whole", "real"), find("small", "whole", "shuffle")),
        ("big zero-mouth vs real (whole)", find("big", "whole", "real"), find("big", "whole", "zero")),
    ]
    print("\n## Paired comparisons (same utterances; delta = candidate - baseline, dB)\n")
    print("| comparison | n | dSI-SNRi mean | std | wins | win rate |")
    print("|---|---|---|---|---|---|")
    for label, base, cand in pairs:
        if not base or not cand:
            continue
        a = _read_csv(args.dir / base["csv"])
        b = _read_csv(args.dir / cand["csv"])
        pair = _pair(a, b)
        if not pair:
            continue
        print(f"| {label} | {pair['n']} | {pair['mean_delta']:+.3f} | {pair['std']:.3f} | "
              f"{pair['wins']}/{pair['n']} | {pair['win_rate'] * 100:.1f}% |")

    # full-scale pairing: big-full (3000) vs the Phase-1 small-full CSV (3000), matched by snt_id
    big_full = find("big", "whole", "real")
    if big_full and args.small_full_csv.is_file() and big_full["n"] >= 1000:
        base = _read_csv(args.small_full_csv)
        cand = _read_csv(args.dir / big_full["csv"])
        pair = _pair(base, cand)
        if pair:
            print(f"| big-full vs small-full (whole, real; Phase-1 CSV) | {pair['n']} | "
                  f"{pair['mean_delta']:+.3f} | {pair['std']:.3f} | "
                  f"{pair['wins']}/{pair['n']} | {pair['win_rate'] * 100:.1f}% |")
        # absolute numbers side by side
        def _mean(d, key):
            vals = [v[key] for v in d.values()]
            return sum(vals) / len(vals)
        print(f"\nFull-scale absolute means: small-full SI-SNRi={_mean(base, 'si-snr_i'):.3f} "
              f"SDRi={_mean(base, 'sdr_i'):.3f} | big-full SI-SNRi={_mean(cand, 'si_snr_i') if False else _mean(cand, 'si-snr_i'):.3f} "
              f"SDRi={_mean(cand, 'sdr_i'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
