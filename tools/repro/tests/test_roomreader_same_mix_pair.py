import json

import numpy as np
import pytest

from tools.streaming.evaluate_roomreader_same_mix_pair import score_request, summarize


def test_swap_has_to_change_reference_preference_and_beat_all_active_sources():
    rng = np.random.default_rng(42)
    stems = rng.normal(size=(2000, 3)) * [0.1, 0.4, 0.2]
    names = ["weak", "strong", "other"]
    mixture = stems.sum(axis=1)

    def measure(output, target):
        return score_request(output, mixture, stems, names, names, target)

    weak = measure(stems[:, 0], "weak")
    strong = measure(stems[:, 1], "strong")
    louder_wrong = measure(stems[:, 1], "weak")
    third_wrong = measure(stems[:, 2] + .5 * stems[:, 0], "weak")
    assert weak["target_preferred"] and strong["target_preferred"]
    assert weak["fit_target_preferred"] and strong["fit_target_preferred"]
    assert json.loads(json.dumps(weak, allow_nan=False)) == weak
    assert not louder_wrong["target_preferred"] and not third_wrong["target_preferred"]
    assert third_wrong["preferred_active_reference"] == "other"
    assert weak["target_projection_delta_db"] == pytest.approx(0, abs=1e-8)

    row = {"session": "test", "group": "weak_three_plus", "active": names,
           "targets": names[:2], "start_ms": 0, "end_ms": 400,
           "metrics": {"paired_off": [louder_wrong, strong], "paired_on": [weak, strong]}}
    result = summarize([row])["weak_all"]
    assert result["models"]["paired_off"]["both_outputs_prefer_same"] == 1
    assert result["models"]["paired_off"]["both_targets_preferred"] == 0
    assert result["models"]["paired_on"]["both_targets_preferred"] == 1
    assert result["paired_on_minus_off"]["both_targets_preferred"]["wins"] == 1
