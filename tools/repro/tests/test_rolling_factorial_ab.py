import pytest
import torch

from tools.streaming import rolling_factorial_ab as factorial
from tools.streaming import rolling_init_ab as init


def paired_record():
    time = torch.arange(init.TOTAL).float()
    targets = torch.stack((.04*torch.sin(time/11), .08*torch.sin(time/13)))
    features = torch.arange(51).float().view(1,1,51).expand(2,1,51).clone()
    features[1] += 100
    return {"mix": targets.sum(0), "targets": targets, "features": features}


@pytest.mark.parametrize("level", [-12, -6, 0, 6, 12])
def test_gain_control_preserves_same_mixture_for_both_mouths(level):
    pair = paired_record()
    mix, refs = factorial.shared_level(pair, level)
    achieved = init.score_ratio(refs[0].square().mean(), refs[1].square().mean())
    assert achieved == pytest.approx(level, abs=2e-5)
    torch.testing.assert_close(mix, refs.sum(0))
    slot = {"kind":"counterlevel","s0_minus_s1_db":level,"row":0,"block":5}
    audio, lips, target, cropped_mix = factorial.pack([pair],[slot],True,"cpu")
    torch.testing.assert_close(audio[0],audio[1],rtol=0,atol=0)
    assert not torch.equal(lips[0],lips[1])
    torch.testing.assert_close(target.sum(0),cropped_mix[0])


def test_quiet_replacement_changes_only_target_and_shared_current_audio():
    pair = paired_record()
    slot = {"kind":"visible_quiet","quiet_direction":1,"row":0,"block":5}
    natural = factorial.pack([pair],[slot],False,"cpu")
    changed = factorial.pack([pair],[slot],True,"cpu")
    assert torch.equal(natural[1],changed[1])
    assert torch.equal(changed[0][0],changed[0][1])
    assert torch.equal(changed[2][0],natural[2][0])
    assert changed[2][1].count_nonzero() == 0
    history = init.HISTORY*init.BLOCK_SAMPLES
    assert torch.equal(changed[0][:,:history], natural[0][:,:history])
    assert torch.equal(changed[0][:,-init.PREVIEW_SAMPLES:], natural[0][:,-init.PREVIEW_SAMPLES:])


def test_schedule_uses_all_visible_quiet_windows_and_preserves_two_directions():
    data = {"train_schedule":[[[0,0,"natural",0,6],[0,1,"natural",0,6]]*4 for _ in range(96)]}
    quiet = [{"row":i,"block":8,"direction":i%2} for i in range(11)]
    schedule = factorial.schedule_from_frozen(data,quiet,96)
    selected = [row for batch in schedule for row in batch if row["kind"]=="visible_quiet"]
    assert len(selected)==24
    assert {r["row"] for r in selected}==set(range(11))
    assert sum(row["kind"]=="counterlevel" for batch in schedule for row in batch)==192
    assert sum(row["kind"]=="natural" for batch in schedule for row in batch)==168
