"""Streaming original-FRCNN alignment and lookahead tests."""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from look2hear.models.FalconLookaheadVisual import FalconLookaheadVisual
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def build(pretrained=False):
    backbone = FRCNNVideoModel(backbone_type="resnet", relu_type="prelu")
    if pretrained:
        path = Path("/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar")
        if not path.exists():
            pytest.skip("released visual weights are not available")
        backbone.init_from(path)
    return backbone.eval()


@pytest.mark.parametrize("pretrained", [False, True])
def test_exact_original_features_including_stream_boundaries(pretrained):
    torch.manual_seed(42)
    backbone = build(pretrained)
    stream = FalconLookaheadVisual(backbone).eval()
    # Nine frames ensure two internal chunk boundaries and a final flush.
    video = torch.randn(1, 1, 9, 88, 88)
    padded = F.pad(video[:, 0], (0, 0, 0, 0, 0, 2))
    with torch.inference_mode():
        expected = backbone(video)
        state = None
        outputs = []
        for start in range(0, 9, 3):
            output, state = stream(padded[:, start:start + 5], state)
            outputs.append(output)
            torch.testing.assert_close(state, video[:, 0, start + 1:start + 3], rtol=0, atol=0)
        actual = torch.cat(outputs, dim=-1)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_future_boundaries_and_frozen_backbone():
    torch.manual_seed(42)
    backbone = build()
    stream = FalconLookaheadVisual(backbone).train()
    assert not backbone.training
    assert all(not p.requires_grad for p in backbone.parameters())
    video = torch.randn(1, 1, 9, 88, 88)
    changed = video.clone()
    changed[:, :, 5:] += 3
    with torch.inference_mode():
        whole = backbone(video)
        modified = backbone(changed)
        first, state = stream(video[:, 0, :5])
    torch.testing.assert_close(first, whole[:, :, :3], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(whole[:, :, :3], modified[:, :, :3], rtol=0, atol=0)
    assert not torch.allclose(whole[:, :, 3], modified[:, :, 3])
    torch.testing.assert_close(state, video[:, 0, 1:3], rtol=0, atol=0)


def test_bad_shapes_rejected():
    stream = FalconLookaheadVisual(build())
    with pytest.raises(ValueError):
        stream(torch.zeros(1, 3, 88, 88))
    with pytest.raises(ValueError):
        stream(torch.zeros(1, 5, 88, 88), torch.zeros(1, 4, 88, 88))
