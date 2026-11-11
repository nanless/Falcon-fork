import pytest
import torch

from tools.repro.tests.test_realssa_tar import make_shard
from tools.streaming.lookahead_training import lookahead_windows, run_lookahead_sequence, CHUNK_SAMPLES
from tools.streaming.realssa_lookahead import RealSSALookaheadDataset, INPUT_SAMPLES, INPUT_FRAMES
from tools.streaming.realssa_tar import RealSSATarDataset, build_index


def test_real_future_and_real_eof_zero_padding(tmp_path):
    root = tmp_path / "src"
    make_shard(root / "tr" / "one.tar", [("a", 60, None), ("b", 51, None)])
    path = tmp_path / "train.json"
    build_index(root, "tr", path)
    ds = RealSSALookaheadDataset(path, crop="start")
    original = RealSSATarDataset(path, crop="start")
    mix, target, video = ds[0]
    old_mix, old_target, old_video = original[0]
    assert INPUT_SAMPLES == 33920 and INPUT_FRAMES == 53
    assert mix.shape == (33920,) and target.shape == (32000,) and video.shape == (53, 88, 88)
    torch.testing.assert_close(mix[:32000], old_mix)
    torch.testing.assert_close(target, old_target)
    torch.testing.assert_close(video[:50], old_video)
    assert mix[32000].item() == pytest.approx(32000 / 100000)
    assert video[52, 0, 0].item() == pytest.approx((51 / 255 - .421) / .165, abs=2e-5)
    windows = list(lookahead_windows(mix[None], video[None], target[None]))
    last = windows[-1]
    assert len(windows) == 17 and last.start_sample == 30720 and last.valid_samples == 1280
    torch.testing.assert_close(last.audio[0], mix[30720:33920])
    torch.testing.assert_close(last.video[0], video[48:53])
    assert last.audio[0, -1].item() == pytest.approx(33919 / 100000)
    assert last.video[0, -1, 0, 0].item() == video[52, 0, 0].item()
    short_mix, _, short_video = ds[1]
    assert short_mix[50*640].item() == pytest.approx(32000 / 100000)
    assert torch.count_nonzero(short_mix[51*640:]) == 0
    assert torch.count_nonzero(short_video[51:]) == 0
    assert short_video[50, 0, 0].item() != 0


def test_output_length_is_independent_from_full_preview():
    audio = torch.arange(33920, dtype=torch.float32)[None]
    video = torch.zeros(1, 53, 1)
    observed = []
    def step(a, v, state):
        observed.append(a)
        return a[:, :CHUNK_SAMPLES], state
    output, _ = run_lookahead_sequence(audio, video, step, output_samples=32000)
    assert output.shape == (1, 32000) and len(observed) == 17
    torch.testing.assert_close(output, audio[:, :32000])
    assert observed[-1][0, -1].item() == 33919
