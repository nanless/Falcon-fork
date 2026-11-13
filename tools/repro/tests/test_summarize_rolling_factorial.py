import json
import pytest

from tools.streaming.summarize_rolling_factorial import ARMS, controlled, paired_report


def test_factorial_effects_use_paired_differences_and_keep_interaction():
    result = paired_report([("session1", (1, 4, 2, 8)), ("session2", (2, 5, 3, 9))], 100)
    assert result["arm_means"]["conditioned_augmented"] == 8.5
    assert result["contrasts"]["structure_under_natural"]["delta"] == 1
    assert result["contrasts"]["structure_under_augmented"]["delta"] == 4
    assert result["contrasts"]["data_under_base"]["delta"] == 3
    assert result["contrasts"]["data_under_conditioned"]["delta"] == 6
    assert result["contrasts"]["interaction"] == {"delta": 3, "paired_cluster_bootstrap_95": [3, 3]}


def test_controlled_counts_both_requests_and_rejects_unmatched_rows(tmp_path):
    base_rows = [{"row": 0, "direction": d, "start_block": 5, "target_preferred": d == 0} for d in (0, 1)]
    for arm in ARMS:
        rows = [dict(row, target_preferred=True) for row in base_rows] if arm.endswith("augmented") else base_rows
        (tmp_path / f"{arm}.json").write_text(json.dumps({"rows": {"natural": rows}}))
    result = controlled(tmp_path, 100)["natural"]
    assert result["both_directions_correct"]["arm_means"]["base_natural"] == 0
    assert result["both_directions_correct"]["arm_means"]["base_augmented"] == 1
    (tmp_path / "base_natural.json").write_text(json.dumps({"rows": {"natural": base_rows[:1]}}))
    with pytest.raises(ValueError, match="different sample IDs"):
        controlled(tmp_path, 100)


def test_silence_factorial_uses_named_contrasts_and_retains_tail():
    from tools.streaming.summarize_rolling_factorial import SILENCE_ARMS, SILENCE_CONTRASTS

    result = paired_report([("one", (-20, -17, -24, -25)), ("two", (-3, -2, -10, -11))], 100,
                           arms=SILENCE_ARMS, contrasts=SILENCE_CONTRASTS)
    assert result["arm_max"]["level_off_silence_on"] == -10
    assert result["contrasts"]["level_under_silence_off"]["delta"] == 2
    assert result["contrasts"]["silence_under_level_on"]["delta"] == -8.5
    assert result["contrasts"]["interaction"]["delta"] == -3
