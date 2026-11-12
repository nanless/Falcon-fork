import math
import pytest
from tools.streaming.robust_schedule import RobustWarmupPlateau


def test_best_tracked_during_warmup_and_severe_drop_stays_scaled():
    s = RobustWarmupPlateau(.01, 1000, patience=5, severe_delta=10)
    assert s.observe(-3.3)["improved"]
    assert not s.observe(-3.2)["improved"]
    assert s.best == -3.3
    assert s.learning_rate(399) == pytest.approx(.004)
    event = s.observe(1e6)
    assert event["severe"] and event["reduced"] and s.scale == .5
    assert s.learning_rate(400) == pytest.approx(.002005)
    assert s.learning_rate(1000) == pytest.approx(.005)


def test_plateau_decreases_scale_and_state_restores_exactly():
    s = RobustWarmupPlateau(.01, 1000, patience=1, severe_delta=100)
    s.observe(-3.0); s.observe(-2.9)
    assert s.observe(-2.8)["reduced"]
    saved = s.state_dict()
    restored = RobustWarmupPlateau(.01, 1000, patience=1, severe_delta=100)
    restored.load_state_dict(saved)
    for step, val in [(350,-3.2),(400,-3.1),(450,-3.0)]:
        assert restored.learning_rate(step) == s.learning_rate(step)
        assert restored.observe(val) == s.observe(val)
    assert restored.state_dict() == s.state_dict()


def test_config_and_invalid_value_checks():
    s = RobustWarmupPlateau(.01, 100)
    with pytest.raises(ValueError): s.observe(float("nan"))
    with pytest.raises(ValueError): s.learning_rate(-1)
    with pytest.raises(ValueError): RobustWarmupPlateau(.02, 100).load_state_dict(s.state_dict())
    assert math.isinf(s.best)


def test_bad_gradient_reduces_scale_and_nan_is_failure():
    import torch
    from tools.streaming.robust_schedule import global_gradient_norm
    s = RobustWarmupPlateau(.01, 1000)
    normal = global_gradient_norm(torch.tensor(119.0))
    assert normal.item() < 1000
    bad = global_gradient_norm(torch.tensor(float("nan")))
    assert torch.isinf(bad)
    assert s.reduce_after_bad_step()
    assert s.learning_rate(400) == pytest.approx(.002005)
    restored = RobustWarmupPlateau(.01, 1000)
    restored.load_state_dict(s.state_dict())
    assert restored.learning_rate(400) == s.learning_rate(400)


def test_gradient_gate_uses_maximum_from_other_rank(monkeypatch):
    import torch
    from tools.streaming import robust_schedule as module
    calls = []
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    def all_reduce(value, op):
        calls.append(op)
        value.fill_(7182.0)
    monkeypatch.setattr(module.dist, "all_reduce", all_reduce)
    result = module.global_gradient_norm(torch.tensor(15.0))
    assert result.item() > 1000 and calls == [module.dist.ReduceOp.MAX]


def test_cumulative_small_improvements_reset_stale_against_significant_best():
    schedule = RobustWarmupPlateau(.01, 1000, patience=5, min_delta=.005)
    stale = 0
    for value in [-3.0, -3.001, -3.002, -3.003, -3.004, -3.006]:
        previous_significant_best = schedule.plateau_best
        schedule.observe(value)
        stale = 0 if value < previous_significant_best - schedule.min_delta else stale + 1
    assert stale == 0
    assert schedule.plateau_best == pytest.approx(-3.006)
