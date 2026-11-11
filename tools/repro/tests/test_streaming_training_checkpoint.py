"""Checkpointed training must retain visual and cross-chunk gradients."""
import copy

import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from look2hear.models.FalconStreaming import FalconStreaming, CHUNK_SAMPLES, VIDEO_FRAMES


@pytest.mark.parametrize("visual_checkpoint,chunk_checkpoint", [(True, False), (False, True), (True, True)])
def test_training_checkpoint_matches_uncheckpointed_visual_and_history_gradients(
    visual_checkpoint, chunk_checkpoint
):
    torch.set_num_threads(2)
    torch.manual_seed(214)
    baseline = FalconStreaming(channels=32, dilations=(1, 2), synthesis_mode="signed_residual",
                               subband_refiner=True, subband_width=24, subband_depth=2).train()
    candidate = copy.deepcopy(baseline)
    candidate.checkpoint_visual = visual_checkpoint
    candidate.checkpoint_chunks = chunk_checkpoint
    assert all(not layer.training for layer in candidate.video_model.modules() if isinstance(layer, _BatchNorm))

    audio = torch.randn(1, 2 * CHUNK_SAMPLES) * 0.1
    video = torch.randn(1, 2 * VIDEO_FRAMES, 88, 88)
    sample = torch.randn(1, CHUNK_SAMPLES)

    def run(model):
        source = audio.clone().requires_grad_()
        prediction = model(source, video)
        # The objective reads only the second output chunk. Its derivative to
        # the first input chunk proves that historical state retains gradients.
        (prediction[:, CHUNK_SAMPLES:] * sample).mean().backward()
        names = [
            "video_model.frontend3D.0.weight",
            "audio_proj.weight",
            "blocks.0.depthwise.weight",
            "subband_capacity.output_proj.weight",
        ]
        params = dict(model.named_parameters())
        return prediction.detach(), source.grad.detach(), {n: params[n].grad.detach() for n in names}

    expected_output, expected_audio_grad, expected_grads = run(baseline)
    output, audio_grad, grads = run(candidate)
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(audio_grad, expected_audio_grad, atol=1e-6, rtol=1e-4)
    assert torch.isfinite(audio_grad).all()
    assert audio_grad[:, :CHUNK_SAMPLES].abs().max() > 0
    for name in expected_grads:
        assert torch.isfinite(grads[name]).all(), name
        torch.testing.assert_close(grads[name], expected_grads[name], atol=1e-6, rtol=1e-4)
    assert grads["video_model.frontend3D.0.weight"].abs().max() > 0
