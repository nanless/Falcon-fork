from pathlib import Path

import numpy as np
import pytest
import torch

from tools.streaming.evaluate_realssa import load_baseline
from tools.streaming.rolling_model_checkpoint import (
    RollingExperimentLiveInference,
    load_rolling_checkpoint,
    make_frequency_conditioned,
    pack_rolling_checkpoint,
)

RELEASE = Path("/root/.cache/falcon_weights/falcon.pt")


@pytest.fixture(scope="module")
def released():
    if not RELEASE.is_file():
        pytest.skip("released Falcon Small checkpoint not on this machine")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield load_baseline(RELEASE, "cpu")
    finally:
        torch.set_num_threads(old_threads)


def test_warm_start_is_bit_exact_and_new_path_can_learn(released):
    base, _, _ = released
    base.spk_pool_proj.requires_grad_(False)
    state = torch.random.get_rng_state()
    model = make_frequency_conditioned(base, seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 20640
    for key, value in base.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    for name, value in base.named_parameters():
        assert dict(model.named_parameters())[name].requires_grad == value.requires_grad

    audio = torch.randn(2, 3200) * .03
    video = torch.randn(2, 512, 5)
    with torch.no_grad():
        expected = base(audio, video)
    actual = model(audio, video)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    mod = model.frequency_condition.modulation
    assert mod.weight.grad is not None
    assert torch.isfinite(mod.weight.grad).all()
    assert mod.weight.grad.abs().max() > 0


@torch.no_grad()
def test_active_condition_has_frequency_selectivity_and_no_batch_crosstalk(released):
    base, _, _ = released
    model = make_frequency_conditioned(base, seed=42)
    path = model.frequency_condition
    path.modulation.weight[0, 0, 0] = 1
    features = torch.ones(1, 112, 5, 129)
    conditioned = path(features, torch.ones(1, 512, 5))
    assert (conditioned[:, :14, :, :17] != 1).all()
    assert (conditioned[:, 14:] == 1).all()
    assert (conditioned[..., 17:] == 1).all()

    audio = torch.randn(2, 3200) * .03
    video = torch.randn(2, 512, 5)
    reference = model(audio, video)[0]
    video[1].neg_()
    torch.testing.assert_close(model(audio, video)[0], reference, rtol=0, atol=0)


@torch.no_grad()
def test_checkpoint_round_trip_preserves_architecture_and_history(released, tmp_path):
    base, backbone, _ = released
    audio = torch.randn(1, 3200) * .03
    video = torch.randn(1, 512, 5)
    conditioned = make_frequency_conditioned(base, seed=42)
    conditioned.frequency_condition.modulation.weight.normal_(std=.02)
    for model in (base, conditioned):
        path = tmp_path / f"{type(model).__name__}.pt"
        torch.save(pack_rolling_checkpoint(model, backbone, history_blocks=3, steps=24), path)
        loaded, visual, payload = load_rolling_checkpoint(path, "cpu")
        assert type(loaded) is type(model)
        assert payload["history_blocks"] == 3
        assert payload["steps"] == 24
        assert all(not parameter.requires_grad for parameter in visual.parameters())
        torch.testing.assert_close(loaded(audio, video), model(audio, video), rtol=0, atol=0)

    payload["state_dict"].pop("audio_model.frequency_condition.modulation.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="frequency_condition.modulation.weight"):
        load_rolling_checkpoint(path, "cpu")


@torch.no_grad()
def test_active_adapter_respects_shared_preview_boundary(released, tmp_path):
    base, backbone, _ = released
    model = make_frequency_conditioned(base, seed=42)
    model.frequency_condition.modulation.weight.normal_(std=.02)
    checkpoint = tmp_path / "conditioned.pt"
    torch.save(pack_rolling_checkpoint(model, backbone, history_blocks=5), checkpoint)
    live = RollingExperimentLiveInference(checkpoint)
    generator = np.random.default_rng(42)
    audio = generator.normal(0, .03, size=(11, 640)).astype(np.float32)
    video = generator.normal(0, 1, size=(11, 88, 88)).astype(np.float32)

    def run(sound, mouth):
        live.reset()
        outputs = []
        ticks = []
        for i in range(len(sound)):
            if result := live.push(sound[i], mouth[i], sequence=i):
                outputs.extend(result)
                ticks.append(i)
        outputs.extend(live.finish())
        assert ticks == [4, 7, 10]
        assert sum(len(chunk) for chunk in outputs) == sound.size
        return outputs

    original = run(audio, video)
    for start in (5, 8):
        future_audio, future_video = audio.copy(), video.copy()
        future_audio[start:] *= -1
        future_video[start:] *= -1
        changed = run(future_audio, future_video)
        for block in range((start - 2) // 3):
            np.testing.assert_array_equal(original[block], changed[block])
        assert not np.array_equal(original[(start - 2) // 3], changed[(start - 2) // 3])
