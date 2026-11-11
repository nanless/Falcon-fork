"""Complete-coverage and provenance checks for paired evaluation aggregation."""
import json
import subprocess
import sys

import pytest

from tools.streaming.merge_realssa_evaluations import METRICS, PROTOCOL_KEYS, digest, merge


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")
    return path


@pytest.fixture
def artifacts(tmp_path):
    index = write_json(tmp_path / "tt.json", {
        "format": "realssa-tar-index-v1", "split": "tt", "sample_count": 5,
        "incomplete_count": 0, "incomplete": [],
        "samples": [{"id": f"id-{i}"} for i in range(5)],
    })
    meta = {key: "value" for key in PROTOCOL_KEYS}
    meta.update(split="tt", index_sha256=digest(index), baseline_epoch=714, streaming_steps=91,
                streaming_model_config={"channels": 320, "dilations": [1, 2]},
                code_sha256={"model.py": "abc"}, threads=4)

    def shard(start, end):
        rows = [{"id": f"id-{i}", "index": i, "status": "ok", METRICS[0]: float(i),
                 METRICS[1]: float(2 * i + 1)} for i in range(start, end)]
        old = sum(row[METRICS[0]] for row in rows) / len(rows)
        new = sum(row[METRICS[1]] for row in rows) / len(rows)
        return {**meta, "start": start, "end_exclusive": end, "rows": rows,
                "summary": {"attempted": len(rows), "valid_pairs": len(rows), "failed": 0,
                            METRICS[0] + "_mean": old, METRICS[1] + "_mean": new,
                            "streaming_minus_falcon_db": new - old}}
    return index, shard


def test_complete_unequal_out_of_order_shards_weight_by_sample(artifacts, tmp_path):
    index, shard = artifacts
    a = write_json(tmp_path / "a.json", shard(0, 2))
    b = write_json(tmp_path / "b.json", shard(2, 5))
    result = merge(index, [b, a])
    assert [row["index"] for row in result["rows"]] == list(range(5))
    assert result["summary"]["attempted"] == 5
    assert result["summary"][METRICS[0] + "_mean"] == 2.0
    assert result["summary"][METRICS[1] + "_mean"] == 5.0
    assert result["summary"]["streaming_minus_falcon_db"] == 3.0
    assert [s["start"] for s in result["shards"]] == [0, 2]


def test_reject_missing_or_duplicate_shard(artifacts, tmp_path):
    index, shard = artifacts
    a = write_json(tmp_path / "a.json", shard(0, 2))
    with pytest.raises(ValueError, match="missing 3"):
        merge(index, [a])
    with pytest.raises(ValueError, match="duplicate sample index"):
        merge(index, [a, a])


def test_reject_wrong_id_and_duplicate_rows(artifacts, tmp_path):
    index, shard = artifacts
    data = shard(0, 5)
    data["rows"][2]["id"] = "incorrect"
    path = write_json(tmp_path / "bad.json", data)
    with pytest.raises(ValueError, match="sample ID mismatch"):
        merge(index, [path])
    data = shard(0, 5)
    data["rows"][3] = data["rows"][2].copy()
    write_json(path, data)
    with pytest.raises(ValueError, match="duplicate sample index"):
        merge(index, [path])


@pytest.mark.parametrize("changed", ["streaming_sha256", "baseline_sha256", "code_sha256", "device", "index_sha256"])
def test_reject_different_provenance(artifacts, tmp_path, changed):
    index, shard = artifacts
    a, data = shard(0, 2), shard(2, 5)
    data[changed] = "wrong"
    one = write_json(tmp_path / "a.json", a)
    two = write_json(tmp_path / "b.json", data)
    with pytest.raises(ValueError, match="identity mismatch|inconsistent evaluation protocol"):
        merge(index, [one, two])


def test_reject_failed_nonfinite_or_bad_summary(artifacts, tmp_path):
    index, shard = artifacts
    path = tmp_path / "bad.json"
    data = shard(0, 5)
    data["rows"][1]["status"] = "error"
    write_json(path, data)
    with pytest.raises(ValueError, match="failed or nonfinite"):
        merge(index, [path])
    data = shard(0, 5)
    data["rows"][1][METRICS[0]] = float("nan")
    write_json(path, data)
    with pytest.raises(ValueError, match="failed or nonfinite"):
        merge(index, [path])
    data = shard(0, 5)
    data["summary"][METRICS[0] + "_mean"] = 123
    write_json(path, data)
    with pytest.raises(ValueError, match="mean disagrees"):
        merge(index, [path])


def test_cli_refuses_overwrite(artifacts, tmp_path):
    index, shard = artifacts
    source = write_json(tmp_path / "shard.json", shard(0, 5))
    output = tmp_path / "out.json"
    command = [sys.executable, "-m", "tools.streaming.merge_realssa_evaluations",
               "--tar-index", str(index), "--output", str(output), str(source)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    prior = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0 and "FileExistsError" in second.stderr
    assert output.read_bytes() == prior
