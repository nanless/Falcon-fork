from pathlib import Path
import copy

import pytest
import torch

from tools.streaming.train_paper_lookahead import PaperLookaheadTrainingModel


VISUAL = "/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar"


@pytest.mark.skipif(not Path(VISUAL).exists(), reason="pretrained visual asset unavailable")
def test_activation_recompute_matches_outputs_and_core_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = {"visual_weights": VISUAL, "width": 8, "rnn_hidden": 8,
              "heads": 2, "history_frames": 20, "checkpoint_chunks": False}
    normal = PaperLookaheadTrainingModel(config).train()
    recomputed = copy.deepcopy(normal).train()
    recomputed.checkpoint_chunks = True
    audio = torch.randn(1, 3840)
    video = torch.randn(1, 6, 88, 88)
    first = normal(audio, video)
    second = recomputed(audio, video)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    first.square().mean().backward()
    second.square().mean().backward()
    paired = zip(normal.core.named_parameters(), recomputed.core.named_parameters())
    count = 0
    for (name, p), (other_name, q) in paired:
        assert name == other_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)
            count += 1
    assert count > 0
    assert all(not p.requires_grad and p.grad is None for p in normal.visual.parameters())
    assert all(not p.requires_grad and p.grad is None for p in recomputed.visual.parameters())
