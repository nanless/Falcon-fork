import pytest
import torch

from tools.streaming import rolling_init_ab as ab


def test_rolling_crop_uses_bounded_audio_history_and_only_holds_features():
    mixture = torch.arange(ab.TOTAL).float()
    pair = {"features": torch.arange(51).float().view(1, 1, 51).expand(2, 1, 51)}
    audio, mouth = ab.window(pair, mixture, 1, 7)
    assert audio.shape == (12800,)
    assert audio[0].item() == 3840
    assert audio[-1].item() == 16639
    assert mouth.flatten().tolist() == list(range(6, 24)) + [23, 23]
    def identity_model(sound, feature):
        assert feature.shape == (1, 1, 20)
        return sound[:, None]
    torch.testing.assert_close(ab.forward(identity_model, audio[None], mouth[None])[0],
                               mixture[7 * 1920:8 * 1920])


def test_fixture_controls_relative_power_and_never_uses_third_parent_from_same_pair():
    time = torch.arange(ab.TOTAL).float()
    pair = {"mix": torch.sin(time / 10) * .02,
            "targets": torch.stack([torch.sin(time / 10) * .02, torch.cos(time / 14) * .05])}
    rows = [{"parent_video_ids": ["one", "two"]}, {"parent_video_ids": ["three", "four"]}]
    third = ab.third_sources(rows)
    assert [idx for idx, _ in third] == [1, 0]
    for kind, n in (("weak6", 1), ("weak12", 1), ("single", 0), ("two", 2), ("mute_target", 1)):
        mix, target, near = ab.fixture([pair, pair], third, 0, 0, kind,
                                       {"weak6": -6, "weak12": -12}.get(kind, 0))
        assert len(near) == n
        torch.testing.assert_close(mix, target + sum(near))
        if kind.startswith("weak"):
            achieved = ab.score_ratio(target.square().mean(), near[0].square().mean())
            assert achieved == pytest.approx(-float(kind[4:]), abs=1e-5)
        if kind == "mute_target":
            assert target.count_nonzero() == 0


def test_schedule_presents_both_directions_with_fixed_recipe():
    schedule = ab.training_schedule([{"span": 6}] * 12, 12, 4)
    assert schedule == ab.training_schedule([{"span": 6}] * 12, 12, 4)
    assert set(row[2] for step in schedule for row in step) == {"scaled", "two", "single", "mute_target", "natural"}
    for step in schedule:
        assert len({row[0] for row in step}) == 4
        for a, b in zip(step[::2], step[1::2], strict=True):
            assert a[0] == b[0] and (a[1], b[1]) == (0, 1)
            assert a[2] == b[2] and a[4] == b[4]
            assert a[3] == -b[3]
