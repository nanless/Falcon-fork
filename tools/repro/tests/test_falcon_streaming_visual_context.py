"""Chunk-centered visual-context geometry, causality and version contracts."""
from pathlib import Path

import pytest
import torch

from look2hear.models.FalconStreaming import FalconStreaming, CHUNK_SAMPLES, VIDEO_FRAMES
from tools.streaming.common import (
    CHECKPOINT_FORMAT, CENTERED_CHECKPOINT_FORMAT,
    checkpoint_format_for_visual_mode, read_checkpoint,
)
from tools.streaming.session import StreamingSession


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    return FalconStreaming(channels=8, dilations=(1, 2), visual_mode="chunk_centered").eval()


def test_frontend_uses_only_current_chunk_and_two_prior_frames(model):
    observed = []
    hook = model.video_model.frontend3D.register_forward_pre_hook(
        lambda mod, args: observed.append(args[0].detach().clone())
    )
    old = torch.stack([torch.full((88, 88), x) for x in (-4., -3., -2., -1.)])[None]
    video = torch.stack([torch.full((88, 88), x) for x in (10., 11., 12.)])[None]
    with torch.inference_mode():
        features, state = model._visual(video, old)
    hook.remove()
    assert features.shape == (1, 8, 3)
    assert observed[0].shape == (1, 1, 7, 88, 88)
    assert observed[0][0, 0, :, 0, 0].tolist() == [-2., -1., 10., 11., 12., 12., 12.]
    assert state[0, :, 0, 0].tolist() == [-1., 10., 11., 12.]


def test_chunk_centered_future_block_independence_and_stream_equivalence(model):
    gen = torch.Generator().manual_seed(543)
    audio = torch.randn(1, CHUNK_SAMPLES * 2, generator=gen)
    video = torch.randn(1, VIDEO_FRAMES * 2, 88, 88, generator=gen)
    with torch.inference_mode():
        original = model(audio, video)
        one, state = model.forward_chunk(audio[:, :CHUNK_SAMPLES], video[:, :VIDEO_FRAMES])
        two, _ = model.forward_chunk(audio[:, CHUNK_SAMPLES:], video[:, VIDEO_FRAMES:], state)
        changed_audio, changed_video = audio.clone(), video.clone()
        changed_audio[:, CHUNK_SAMPLES:] += 5
        changed_video[:, VIDEO_FRAMES:] -= 3
        audio_future_changed = model(changed_audio, video)
        video_future_changed = model(audio, changed_video)
    torch.testing.assert_close(original, torch.cat([one, two], dim=1), rtol=0, atol=0)
    torch.testing.assert_close(original[:, :CHUNK_SAMPLES], audio_future_changed[:, :CHUNK_SAMPLES], rtol=0, atol=0)
    torch.testing.assert_close(original[:, :CHUNK_SAMPLES], video_future_changed[:, :CHUNK_SAMPLES], rtol=0, atol=0)


def test_centered_frontend_first_frame_uses_all_current_video(model):
    # Future inside the same block is allowed; perturbing the third frame must
    # influence the first visual feature under this five-frame frontend.
    torch.manual_seed(321)
    video = torch.randn(1, VIDEO_FRAMES, 88, 88)
    changed = video.clone()
    changed[:, 2] += torch.randn_like(changed[:, 2]) * 3
    with torch.inference_mode():
        original, _ = model._visual(video, None)
        alternate, _ = model._visual(changed, None)
    assert (original[:, :, 0] - alternate[:, :, 0]).abs().max().item() > 1e-8


def test_legacy_causal_path_is_identical_to_original_frontend():
    torch.set_num_threads(2)
    torch.manual_seed(41)
    model = FalconStreaming(channels=8, dilations=(1,), visual_mode="causal").eval()
    old, video = torch.randn(1, 4, 88, 88), torch.randn(1, 3, 88, 88)
    from look2hear.videomodels.frcnn_videomodel import threeD_to_2D_tensor
    with torch.inference_mode():
        actual, state = model._visual(video, old)
        frames = torch.cat((old, video), 1)
        features = model.video_model.frontend3D(frames.unsqueeze(1))
        b, _, count, _, _ = features.shape
        expected = model.video_proj(model.video_model.trunk(threeD_to_2D_tensor(features)).reshape(b, count, -1).transpose(1, 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, frames[:, -4:], rtol=0, atol=0)
    assert model.state_dict()["_streaming_format_version"].item() == 4


def test_new_checkpoint_reconstructs_centered_session_and_forbids_cross_version(model, tmp_path):
    target = tmp_path / "centered.pt"
    config = {"channels": 8, "dilations": [1, 2], "visual_mode": "chunk_centered"}
    torch.save({"format": CENTERED_CHECKPOINT_FORMAT, "model_config": config,
                "state_dict": model.state_dict()}, target)
    assert read_checkpoint(target, "cpu")["model_config"] == config
    session = StreamingSession(target)
    assert session.model.visual_mode == "chunk_centered"
    assert session.model.state_dict()["_streaming_format_version"].item() == 5
    legacy = FalconStreaming(channels=8, dilations=(1, 2)).eval()
    with pytest.raises(RuntimeError, match="model version 5"):
        model.load_state_dict(legacy.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="model version 4"):
        legacy.load_state_dict(model.state_dict(), strict=False)


def test_checkpoint_format_and_configuration_cannot_disagree(model, tmp_path):
    target = tmp_path / "wrong.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "model_config":
                {"channels": 8, "dilations": [1, 2], "visual_mode": "chunk_centered"},
                "state_dict": {**model.state_dict(), "_streaming_format_version": torch.tensor(4)}}, target)
    with pytest.raises(ValueError, match="model_config"):
        read_checkpoint(target, "cpu")
    assert checkpoint_format_for_visual_mode("causal") == CHECKPOINT_FORMAT
    assert checkpoint_format_for_visual_mode("chunk_centered") == CENTERED_CHECKPOINT_FORMAT
    with pytest.raises(ValueError, match="visual_mode"):
        checkpoint_format_for_visual_mode("not_a_mode")


def test_centered_v5_server_status_ready_and_chunk(model):
    import base64
    import numpy as np
    from fastapi.testclient import TestClient
    from tools.streaming.server import create_app
    with TestClient(create_app(model, "random_untrained"), base_url="http://127.0.0.1:8765") as client:
        status = client.get("/status").json()
        assert status["format"] == "falcon-streaming-v5"
        with client.websocket_connect("ws://127.0.0.1:8765/ws") as ws:
            ready = ws.receive_json()
            assert ready["format"] == "falcon-streaming-v5"
            audio = np.zeros(CHUNK_SAMPLES, dtype="<f4")
            video = np.zeros((VIDEO_FRAMES, 96, 96), dtype=np.uint8)
            ws.send_json({"type": "chunk", "sequence": 0, "start_sample": 0,
                          "video_samples": [0, 640, 1280],
                          "audio_f32le": base64.b64encode(audio.tobytes()).decode("ascii"),
                          "mouth_gray_u8": base64.b64encode(video.tobytes()).decode("ascii")})
            result = ws.receive_json()
            assert result["type"] == "result"
            assert result["end_sample"] == CHUNK_SAMPLES
            assert len(base64.b64decode(result["audio_f32le"])) == CHUNK_SAMPLES * 4


@pytest.mark.parametrize("valid_samples", [1, 640, 641, 1280, 1281, 1919])
def test_centered_partial_tail_masks_invalid_video_and_rejects_nonzero_audio(model, valid_samples):
    import numpy as np
    rng = np.random.default_rng(20261112 + valid_samples)
    samples = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    samples[:valid_samples] = rng.standard_normal(valid_samples).astype(np.float32) * .01
    invalid_audio = samples.copy()
    invalid_audio[valid_samples] = .25
    mouth = rng.integers(0, 256, (VIDEO_FRAMES, 96, 96), dtype=np.uint8)
    altered = mouth.copy()
    valid_frames = (valid_samples + 639) // 640
    altered[valid_frames:] = 255 - altered[valid_frames:]
    normal_times = [0, 640, 1280]
    altered_times = [normal_times[i] if i < valid_frames else 999999 for i in range(VIDEO_FRAMES)]
    one = StreamingSession.from_model(model)
    two = StreamingSession.from_model(model)
    with pytest.raises(ValueError, match="after valid_samples must be zero"):
        one.push(invalid_audio, mouth, sequence=0, start_sample=0,
                 video_samples=normal_times, valid_samples=valid_samples)
    assert one.next_sequence == 0 and one.state is None
    result1 = one.push(samples, mouth, sequence=0, start_sample=0,
                       video_samples=normal_times, valid_samples=valid_samples)
    result2 = two.push(samples, altered, sequence=0, start_sample=0,
                       video_samples=altered_times, valid_samples=valid_samples)
    np.testing.assert_array_equal(result1, result2)
    assert one.finished and two.finished
