import numpy as np
import pytest
import torch

from tools.streaming import rolling_init_ab as init, rolling_silence_ab as ab
from tools.streaming.evaluate_realssa import sha


def test_only_quiet_row_changes_gradient_across_silence_switch():
    # Last two rows are the identical real silent and re-opened requests.
    amplitude = torch.nn.Parameter(torch.full((4, 1), .2))
    n = init.BLOCK_SAMPLES
    wave = torch.randn(4, n+init.PREVIEW_SAMPLES)
    features = torch.zeros(4, 1, 5)
    reference = torch.randn(4, n) * .05
    mixture = torch.randn(4, n) * .1
    def model(audio, video):
        return amplitude[:len(audio), :, None].expand(-1, 1, audio.shape[-1])
    tensors = wave, features, reference, mixture
    mixed = tuple(v[:2] for v in tensors)
    real = tuple(v[2:] for v in tensors)
    gradients = []
    for enabled in (False, True):
        loss, _, quiet, active = ab.compute_losses(model, mixed, real, enabled)
        assert torch.isfinite(quiet) and torch.isfinite(active)
        gradients.append(torch.autograd.grad(loss, amplitude)[0])
    off, on = gradients
    assert off[2].item() == 0 and on[2].item() > 0
    torch.testing.assert_close(off[[0,1,3]],on[[0,1,3]])


def test_real_episode_checks_visibility_and_annotation(tmp_path):
    ticks = 23
    path = tmp_path/"episode.npz"
    arrays = {"audio": np.ones(ticks*640, np.float32), "target": np.zeros(ticks*640,np.float32),
              "mouth": np.zeros((ticks,96,96),np.uint8), "mouth_valid": np.ones(ticks,np.bool_),
              "silent": np.zeros(ticks,np.bool_), "active": np.zeros(ticks,np.bool_)}
    arrays["silent"][15:18] = True
    arrays["active"][18:21] = True
    np.savez(path, **arrays)
    row = {"id":"ep","recording_id":"r","speaker_id":"s","npz":"episode.npz",
           "sha256":sha(path),"silent_blocks":[5],"active_blocks":[6]}
    manifest = tmp_path/"manifest.json"
    init.save(manifest, {"episodes":[row]})
    ep, = ab.load_episodes(manifest)
    assert ep["silent"] == [5] and ep["active"] == [6]
    arrays["mouth_valid"][19] = False
    np.savez(path,**arrays)
    manifest.unlink()
    row["sha256"] = sha(path)
    init.save(manifest,{"episodes":[row]})
    with pytest.raises(ValueError,match="real silent blocks"):
        ab.load_episodes(manifest)


def test_budget_and_repeated_schedule_keep_identical_real_windows():
    data = {"train_schedule":[[[0,0,"natural",0,6],[0,1,"natural",0,6]]*4 for _ in range(96)]}
    episodes = [{"recording_id":"r1","silent":[5,6],"active":[7,8]},{"recording_id":"r2","silent":[9],"active":[10]}]
    schedule = ab.make_schedule(data,episodes,96)
    assert [len(step["lrs2"])*2 + (2 if step["real"] else 0) for step in schedule] == [8]*96
    assert sum(step["real"] is not None for step in schedule) == 24
    assert schedule[3]["real"] == {"episode":0,"silent_block":6,"active_block":7}
    assert schedule[11]["real"] == {"episode":0,"silent_block":6,"active_block":7}



def test_real_slots_round_robin_recordings():
    data = {"train_schedule":[[[0,0,"natural",0,6],[0,1,"natural",0,6]]*4 for _ in range(12)]}
    episodes = [{"recording_id":sid,"silent":[5],"active":[7]} for sid in ("r1","r1","r2","r2","r3")]
    schedule = ab.make_schedule(data,episodes,12)
    assert [r["real"]["episode"] for r in schedule if r["real"]] == [0,2,4]
