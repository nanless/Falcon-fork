"""Verify and combine disjoint RealSSA paired-evaluation shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = ("falcon_si_sdri_db", "streaming_si_sdri_db")
PROTOCOL_KEYS = (
    "scope", "split", "index_sha256", "baseline_sha256", "baseline_epoch",
    "streaming_sha256", "streaming_steps", "streaming_training_status",
    "streaming_model_config", "preprocessing", "metric", "short_input_padding",
    "code_hash_scope", "code_sha256", "torch_version", "device", "threads",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def merge(index_path, shard_paths):
    """Return a full, exactly covered result; raise if any evidence is inconsistent."""
    index_path = Path(index_path)
    shard_paths = [Path(path) for path in shard_paths]
    if not shard_paths:
        raise ValueError("at least one evaluation shard is required")
    index_hash = digest(index_path)
    shard_hashes = [digest(path) for path in shard_paths]
    with index_path.open(encoding="utf8") as stream:
        index = json.load(stream)
    samples = index.get("samples")
    if (index.get("format") != "realssa-tar-index-v1" or index.get("split") != "tt"
            or not isinstance(samples, list) or not samples
            or type(index.get("sample_count")) is not int or index["sample_count"] != len(samples)
            or index.get("incomplete_count") != 0 or index.get("incomplete") != []):
        raise ValueError("expected a complete, nonempty RealSSA tt index")
    ids = [record.get("id") if isinstance(record, dict) else None for record in samples]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("index contains invalid or duplicate sample IDs")

    by_index = {}
    protocol = None
    shard_records = []
    for path, input_hash in zip(shard_paths, shard_hashes):
        with path.open(encoding="utf8") as stream:
            shard = json.load(stream)
        if not isinstance(shard, dict) or any(key not in shard for key in PROTOCOL_KEYS):
            raise ValueError(f"missing protocol metadata: {path}")
        current = {key: shard[key] for key in PROTOCOL_KEYS}
        if current["index_sha256"] != index_hash or current["split"] != index["split"]:
            raise ValueError(f"index identity mismatch: {path}")
        if protocol is None:
            protocol = current
        elif current != protocol:
            changed = [key for key in PROTOCOL_KEYS if current[key] != protocol[key]]
            raise ValueError(f"inconsistent evaluation protocol {changed}: {path}")
        start, end, rows = shard.get("start"), shard.get("end_exclusive"), shard.get("rows")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(samples)
                or not isinstance(rows, list) or len(rows) != end - start):
            raise ValueError(f"invalid shard range or row count: {path}")
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid evaluation row: {path}")
            row_index = row.get("index")
            if type(row_index) is not int or not start <= row_index < end:
                raise ValueError(f"row index outside declared shard: {path}")
            if row_index in seen or row_index in by_index:
                raise ValueError(f"duplicate sample index {row_index}: {path}")
            if row.get("id") != ids[row_index]:
                raise ValueError(f"sample ID mismatch for index {row_index}: {path}")
            if row.get("status") != "ok" or not all(finite_number(row.get(key)) for key in METRICS):
                raise ValueError(f"failed or nonfinite paired evaluation at index {row_index}: {path}")
            seen.add(row_index)
            by_index[row_index] = row
        if seen != set(range(start, end)):
            raise ValueError(f"shard does not exactly cover its declared range: {path}")
        summary = shard.get("summary")
        if not isinstance(summary, dict) or any(summary.get(k) != v for k, v in
                (("attempted", len(rows)), ("valid_pairs", len(rows)), ("failed", 0))):
            raise ValueError(f"shard summary counts disagree with rows: {path}")
        for key in METRICS:
            expected = math.fsum(row[key] for row in rows) / len(rows)
            actual = summary.get(key + "_mean")
            if not finite_number(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f"shard mean disagrees with rows ({key}): {path}")
        expected_delta = summary[METRICS[1] + "_mean"] - summary[METRICS[0] + "_mean"]
        delta = summary.get("streaming_minus_falcon_db")
        if not finite_number(delta) or not math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"shard paired difference disagrees with means: {path}")
        shard_records.append({"path": str(path.resolve()), "sha256": input_hash, "start": start, "end_exclusive": end})
    missing = sorted(set(range(len(samples))) - by_index.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} sample indices; first: {missing[:10]}")
    if digest(index_path) != index_hash or any(digest(path) != h for path, h in zip(shard_paths, shard_hashes)):
        raise RuntimeError("index or shard changed during merge")
    rows = [by_index[i] for i in range(len(samples))]
    summary = {"attempted": len(rows), "valid_pairs": len(rows), "failed": 0}
    for key in METRICS:
        summary[key + "_mean"] = math.fsum(row[key] for row in rows) / len(rows)
    summary["streaming_minus_falcon_db"] = summary[METRICS[1] + "_mean"] - summary[METRICS[0] + "_mean"]
    return {"format": "realssa-paired-evaluation-merged-v1", **protocol,
            "index": str(index_path.resolve()), "start": 0, "end_exclusive": len(rows),
            "shards": sorted(shard_records, key=lambda item: item["start"]), "summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tar-index", required=True, help="complete official tt index")
    p.add_argument("--output", required=True, help="new merged JSON path")
    p.add_argument("shards", nargs="+", help="all disjoint evaluate_realssa JSON outputs")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    result = merge(args.tar_index, args.shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
