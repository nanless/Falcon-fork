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
              "heads": 2, "history_frames": 20, "checkpoint_chunks": False, "mask_limit": 4.0}
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


@pytest.mark.skipif(not Path(VISUAL).exists(), reason="pretrained visual asset unavailable")
def test_full_batch_activation_slicing_matches_all_gradients():
    from tools.streaming.robust_reconstruction import robust_reconstruction
    torch.set_num_threads(2)
    torch.manual_seed(92)
    config = {"visual_weights": VISUAL, "width": 8, "rnn_hidden": 8,
              "heads": 2, "history_frames": 20, "checkpoint_chunks": True,
              "mask_limit": 4.0, "activation_batch_chunk": 0}
    whole = PaperLookaheadTrainingModel(config).train()
    sliced = copy.deepcopy(whole).train()
    sliced.activation_batch_chunk = 2
    audio = torch.randn(4, 3840)
    target = torch.randn(4, 3840)
    video = torch.randn(4, 6, 88, 88)
    a = whole(audio, video)
    b = sliced(audio, video)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    la = robust_reconstruction(a, target, audio, waveform_weight=0.1)[0].mean()
    lb = robust_reconstruction(b, target, audio, waveform_weight=0.1)[0].mean()
    la.backward()
    lb.backward()
    for (name, x), (name2, y) in zip(whole.core.named_parameters(), sliced.core.named_parameters()):
        assert name == name2 and (x.grad is None) == (y.grad is None), name
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, rtol=3e-3, atol=5e-5, msg=name)
