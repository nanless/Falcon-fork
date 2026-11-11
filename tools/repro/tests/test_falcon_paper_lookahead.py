from pathlib import Path

import pytest
import torch

from look2hear.models.FalconPaperLookahead import FalconPaperLookahead


def test_visual_checkpoint_and_exact_per_chunk_future_bound():
    path = Path("/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar")
    if not path.exists():
        pytest.skip("released visual weights are not available")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        torch.manual_seed(42)
        model = FalconPaperLookahead(str(path), width=8, rnn_hidden=8, heads=2).eval()
        assert all(not p.requires_grad for p in model.visual.parameters())
        audio = torch.randn(1, 5120)
        video = torch.randn(1, 8, 88, 88)
        with torch.inference_mode():
            y, s = model(audio[:, :3200], video[:, :5])
            # Remaining samples and video are beyond the first output boundary.
            audio[:, 3200:] *= -50
            video[:, 5:] *= -50
            z, q = model(audio[:, :3200], video[:, :5])
            next_output, _ = model(audio[:, 1920:5120], video[:, 3:8], s)
        torch.testing.assert_close(y, z, rtol=0, atol=0)
        torch.testing.assert_close(s.visual, q.visual, rtol=0, atol=0)
        assert y.shape == next_output.shape == (1, 1920)
        assert torch.isfinite(y).all() and torch.isfinite(next_output).all()
    finally:
        torch.set_num_threads(old_threads)
