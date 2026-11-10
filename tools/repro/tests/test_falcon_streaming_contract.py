"""Independent stream contract checks; random weights do not test separation quality."""
import torch
import pytest
from look2hear.models.FalconStreaming import FalconStreaming, CHUNK_SAMPLES

@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    torch.manual_seed(441)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return FalconStreaming(channels=8, dilations=(1, 2)).to(device).eval()

@pytest.fixture(scope="module")
def inputs(model):
    torch.manual_seed(903)
    dev = next(model.parameters()).device
    return (torch.randn(1, 3 * CHUNK_SAMPLES, device=dev),
            torch.randn(1, 12, 88, 88, device=dev))

@torch.no_grad()
def test_chunk_matches_sequence_and_reset(model, inputs):
    audio, video = inputs
    whole = model(audio, video)
    state = None
    pieces = []
    for i in range(3):
        out, state = model.forward_chunk(audio[:,i*2560:(i+1)*2560], video[:,i*4:(i+1)*4], state)
        pieces.append(out)
    torch.testing.assert_close(whole, torch.cat(pieces, 1), rtol=0, atol=0)
    reset, _ = model.forward_chunk(audio[:,:2560], video[:,:4], None)
    torch.testing.assert_close(reset, pieces[0], rtol=0, atol=0)
    assert whole.shape == audio.shape and torch.isfinite(whole).all()

@torch.no_grad()
def test_future_audio_video_do_not_change_prior_hops(model, inputs):
    audio, video = inputs
    baseline = model(audio, video)
    # Frame / hop aligned perturbations test both intra-chunk and later chunks.
    for cut in (128, 896, 1280, 2432, 2560, 5120):
        changed = audio.clone(); changed[:,cut:] += 2 * torch.randn_like(changed[:,cut:])
        actual = model(changed, video)
        torch.testing.assert_close(actual[:,:cut], baseline[:,:cut], rtol=0, atol=2e-6)
        assert (actual[:,cut:] - baseline[:,cut:]).abs().max() > 1e-5
    for frame in (2, 4, 8):
        changed = video.clone(); changed[:,frame:] += 4 * torch.randn_like(changed[:,frame:])
        actual = model(audio, changed)
        cutoff = frame * 640
        torch.testing.assert_close(actual[:,:cutoff], baseline[:,:cutoff], rtol=0, atol=2e-6)
        assert (actual[:,cutoff:] - baseline[:,cutoff:]).abs().max() > 1e-7

@torch.no_grad()
def test_identity_mask_alignment_at_boundaries(model, inputs):
    audio, video = inputs
    weight, bias = model.mask.weight.clone(), model.mask.bias.clone()
    try:
        model.mask.weight.zero_(); model.mask.bias.fill_(30)
        out = model(audio, video)
        torch.testing.assert_close(out, audio, rtol=1e-5, atol=2e-6)
    finally:
        model.mask.weight.copy_(weight); model.mask.bias.copy_(bias)

def test_audio_and_visual_gradient_reach_trainable_parameters(model, inputs):
    model.train()
    audio, video = (x.detach().clone().requires_grad_() for x in inputs)
    out = model(audio[:,:2560], video[:,:4])
    loss = (out - audio[:,:2560].detach()*0.3).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert video.grad is not None and video.grad.abs().sum() > 0
    assert model.video_proj.weight.grad is not None and model.video_proj.weight.grad.abs().sum() > 0
    assert model.audio_proj.weight.grad is not None and model.audio_proj.weight.grad.abs().sum() > 0
    assert model.video_model.frontend3D[0].weight.grad is not None
    model.zero_grad(set_to_none=True); model.eval()

def test_session_clock_rejection_reset_and_recovery(tmp_path):
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming"))
    from session import StreamingSession
    torch.manual_seed(77)
    checkpoint = tmp_path / "random.pt"
    fresh = FalconStreaming()
    torch.save({"format": "falcon-streaming-v2", "state_dict": fresh.state_dict(), "training_status": "random_untrained"}, checkpoint)
    del fresh
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = StreamingSession(checkpoint, device=device, max_frame_age_samples=640)
    x = np.random.default_rng(31).normal(size=2560).astype(np.float32)
    mouth = np.random.default_rng(32).integers(0, 256, size=(4,96,96), dtype=np.uint8)
    original = session.push(x, mouth, sequence=0, start_sample=0, video_samples=[0,640,1280,1920])
    assert original.shape == (2560,) and np.isfinite(original).all() and session.training_status == "random_untrained"
    for bad in (
        dict(sequence=0,start_sample=2560,video_samples=[2560,3200,3840,4480]),
        dict(sequence=1,start_sample=2559,video_samples=[2560,3200,3840,4480]),
        dict(sequence=1,start_sample=2560,video_samples=[2561,3200,3840,4480]),
        dict(sequence=1,start_sample=2560,video_samples=[1920,1920,3840,4480]),
    ):
        with pytest.raises(ValueError): session.push(x, mouth, **bad)
        assert session.next_sequence == 1 and session.next_sample == 2560
    session.push(x, mouth, sequence=1,start_sample=2560,video_samples=[2560,3200,3840,4480])
    session.reset()
    reset = session.push(x, mouth, sequence=0, start_sample=0, video_samples=[0,640,1280,1920])
    np.testing.assert_allclose(reset, original, rtol=0, atol=0)

@torch.no_grad()
def test_train_mode_preserves_future_and_batch_independence(model, inputs):
    model.train()
    assert all(not m.training for m in model.video_model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    audio, video = inputs
    baseline = model(audio[:,:2560], video[:,:4])
    changed = audio[:,:2560].clone(); changed[:,1280:] += 4 * torch.randn_like(changed[:,1280:])
    torch.testing.assert_close(model(changed,video[:,:4])[:,:1280], baseline[:,:1280], rtol=0, atol=3e-6)
    other_a = torch.randn_like(audio[:,:2560]); other_v = torch.randn_like(video[:,:4])
    batch_a = torch.cat([other_a, audio[:,:2560]], 0)
    batch_v = torch.cat([other_v, video[:,:4]], 0)
    paired = model(batch_a, batch_v)[1:2]
    batch_a[0] = torch.randn_like(batch_a[0]) * 3
    batch_v[0] = torch.randn_like(batch_v[0]) * 3
    paired_changed = model(batch_a, batch_v)[1:2]
    torch.testing.assert_close(paired_changed, paired, rtol=0, atol=0)
    model.eval()
