"""Paired factorial contrasts for the four rolling Falcon short-training arms."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean

ARMS = ("base_natural", "base_augmented", "conditioned_natural", "conditioned_augmented")
CONTRASTS = {
    "structure_under_natural": (-1, 0, 1, 0),
    "structure_under_augmented": (0, -1, 0, 1),
    "data_under_base": (-1, 1, 0, 0),
    "data_under_conditioned": (0, 0, -1, 1),
    "structure_average": (-0.5, -0.5, 0.5, 0.5),
    "data_average": (-0.5, 0.5, -0.5, 0.5),
    "interaction": (1, -1, -1, 1),
}
METRICS = ("si_sdri_db", "snr_db", "selection_margin_db", "target_preferred", "leakage_db",
           "both_polarities_correct")
REAL_METRICS = ("target_si_sdr_db", "target_sisdri_db", "target_to_loudest_interferer_db",
                "target_minus_loudest_other_sisdr_db", "output_to_mixture_db")


def arm_values(rows, metric):
    values = [row.get(metric) for row in rows]
    if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        return values
    if all(value is None for value in values):
        return None
    raise ValueError(f"unpaired or invalid values for {metric}")


def indexed(rows, keys):
    result = {tuple(row.get(key) for key in keys): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate scoring keys: {keys}")
    return result


def paired_report(cases, draws=2000):
    """Aggregate in original units, resampling entire mixtures or real sessions."""
    groups = defaultdict(list)
    for group, values in cases:
        groups[group].append(values)
    if not groups:
        raise ValueError("no paired values")
    clusters = list(groups.values())
    rows = [row for cluster in clusters for row in cluster]
    rng = random.Random(20261113)
    indexes = range(len(clusters))
    samples = [rng.choices(indexes, k=len(clusters)) for _ in range(draws)] if len(clusters) > 1 else []
    sizes = [len(cluster) for cluster in clusters]
    sample_sizes = [sum(sizes[i] for i in sample) for sample in samples]
    totals = [tuple(map(sum, zip(*cluster))) for cluster in clusters]

    def effects(coefficients):
        cluster_effects = [sum(value * coefficient for value, coefficient in zip(total, coefficients, strict=True))
                           for total in totals]
        simulated = sorted(sum(cluster_effects[i] for i in sample) / size
                           for sample, size in zip(samples, sample_sizes, strict=True))
        interval = [simulated[int(.025*(draws-1))], simulated[int(.975*(draws-1))]] if simulated else None
        return {"delta": sum(cluster_effects) / len(rows), "paired_cluster_bootstrap_95": interval}

    return {"rows": len(rows), "clusters": len(clusters),
            "arm_means": dict(zip(ARMS, map(mean, zip(*rows)), strict=True)),
            "arm_max": dict(zip(ARMS, map(max, zip(*rows)), strict=True)),
            "contrasts": {name: effects(coeffs) for name, coeffs in CONTRASTS.items()}}


def controlled(root, draws):
    panels = {arm: json.loads((root / f"{arm}.json").read_text())["rows"] for arm in ARMS}
    if any(set(panel) != set(panels[ARMS[0]]) for panel in panels.values()):
        raise ValueError("controlled arms have different scoring groups")
    output = {}
    for kind in panels[ARMS[0]]:
        rows = [indexed(panels[arm][kind], ("row", "direction")) for arm in ARMS]
        if any(set(arm) != set(rows[0]) for arm in rows):
            raise ValueError(f"{kind} has different sample IDs across arms")
        matched = [(key, [arm[key] for arm in rows]) for key in sorted(rows[0])]
        if any(len({row.get("start_block") for row in arms}) != 1 for _, arms in matched):
            raise ValueError(f"{kind} has different score windows across arms")
        output[kind] = {}
        for metric in METRICS:
            cases = [(key[0], values) for key, arms in matched if (values := arm_values(arms, metric)) is not None]
            if cases:
                if len(cases) != len(matched):
                    raise ValueError(f"{kind}/{metric} differs in coverage")
                output[kind][metric] = paired_report(cases, draws)
        by_pair = defaultdict(list)
        for (row, direction), arms in matched:
            if direction is not None and (flags := arm_values(arms, "target_preferred")) is not None:
                by_pair[row].append(flags)
        if by_pair:
            if any(len(directions) != 2 for directions in by_pair.values()):
                raise ValueError(f"{kind} has an incomplete two-direction pair")
            output[kind]["both_directions_correct"] = paired_report(
                [(row, [all(values) for values in zip(*directions)]) for row, directions in by_pair.items()], draws)
    return output


def real(path, draws):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    indexed(rows, ("group", "session", "id"))
    output = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        scored = [row for row in selected if row.get("metrics")]
        result = {"attempted": len(selected), "scored": len(scored), "metrics": {}}
        for metric in REAL_METRICS:
            cases = [(row["session"], values) for row in scored
                     if (values := arm_values([row["metrics"][arm] for arm in ARMS], metric)) is not None]
            if cases:
                if len(cases) != len(scored):
                    raise ValueError(f"{group}/{metric} differs in coverage")
                result["metrics"][metric] = paired_report(cases, draws)
                if metric in ("target_to_loudest_interferer_db", "target_minus_loudest_other_sisdr_db"):
                    result["metrics"]["target_preferred"] = paired_report(
                        [(session, [v > 0 for v in values]) for session, values in cases], draws)
        output[group] = result
    return output


def misp(path, draws):
    with path.open() as stream:
        models = json.load(stream)["models"]
    panels = [indexed(models[arm]["rows"], ("group", "start_ms", "target")) for arm in ARMS]
    if any(set(panel) != set(panels[0]) for panel in panels):
        raise ValueError("MISP arms have different scored sample IDs")
    result = {}
    for group in sorted({key[0] for key in panels[0]}):
        keys = sorted(key for key in panels[0] if key[0] == group)
        result[group] = {}
        metrics = ("output_to_mix_db", "difference_to_mix_db", "difference_to_output_db")
        for metric in metrics:
            # Alternate mouths exist on every triple and only selected silent clips.
            cases = [(0, values) for key in keys
                     if (values := arm_values([panel[key] for panel in panels], metric)) is not None]
            if cases:
                if (metric == "output_to_mix_db" or group == "triple") and len(cases) != len(keys):
                    raise ValueError(f"MISP {group}/{metric} differs in coverage")
                result[group][metric] = paired_report(cases, draws)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controlled_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--roomreader", type=Path)
    parser.add_argument("--easycom", type=Path)
    parser.add_argument("--misp", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    result = {"controlled": controlled(args.controlled_dir, args.bootstrap)}
    for name, path in (("roomreader", args.roomreader), ("easycom", args.easycom)):
        if path:
            result[name] = real(path, args.bootstrap)
    if args.misp:
        result["misp"] = misp(args.misp, args.bootstrap)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
