"""Independent stream contract checks; random weights do not test separation quality."""
import torch
import pytest
from look2hear.models.FalconStreaming import FalconStreaming, CHUNK_SAMPLES, VIDEO_FRAMES

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
            torch.randn(1, 3 * VIDEO_FRAMES, 88, 88, device=dev))

@torch.no_grad()
def test_chunk_matches_sequence_and_reset(model, inputs):
    audio, video = inputs
    whole = model(audio, video)
    state = None
    pieces = []
    for i in range(3):
        out, state = model.forward_chunk(audio[:,i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES], video[:,i*VIDEO_FRAMES:(i+1)*VIDEO_FRAMES], state)
        pieces.append(out)
    torch.testing.assert_close(whole, torch.cat(pieces, 1), rtol=0, atol=0)
    reset, _ = model.forward_chunk(audio[:,:CHUNK_SAMPLES], video[:,:VIDEO_FRAMES], None)
    torch.testing.assert_close(reset, pieces[0], rtol=0, atol=0)
    assert whole.shape == audio.shape and torch.isfinite(whole).all()

@pytest.mark.parametrize("mode", ["eval", "train"])
@torch.no_grad()
def test_future_chunks_cannot_change_prior_audio(model, inputs, mode):
    model.train(mode == "train")
    audio, video = inputs
    baseline = model(audio, video)
    for block in (1, 2):
        cut = block * CHUNK_SAMPLES
        changed_audio = audio.clone()
        changed_audio[:, cut:] += 2 * torch.randn_like(changed_audio[:, cut:])
        changed_video = video.clone()
        changed_video[:, block*VIDEO_FRAMES:] += 4 * torch.randn_like(changed_video[:, block*VIDEO_FRAMES:])
        for new_audio, new_video in ((changed_audio, video), (audio, changed_video), (changed_audio, changed_video)):
            actual = model(new_audio, new_video)
            torch.testing.assert_close(actual[:, :cut], baseline[:, :cut], rtol=0, atol=2e-6)
    model.eval()

def test_attention_routes_later_audio_and_video_to_earlier_output():
    # Force uniform attention with identity value/output projections. Injecting
    # into the attention input checks actual model wiring, not random weights.
    torch.manual_seed(82)
    m = FalconStreaming(channels=8, dilations=(1, 2)).eval()
    with torch.no_grad():
        attn = m.chunk_attention
        attn.in_proj_weight.zero_(); attn.in_proj_bias.zero_()
        attn.in_proj_weight[16:24].copy_(torch.eye(8))
        attn.out_proj.weight.copy_(torch.eye(8)); attn.out_proj.bias.zero_()
        m.mask.weight.zero_(); m.mask.weight[:, 0, 0].fill_(2)
        m.mask.bias.zero_()
    audio = torch.linspace(-1, 1, CHUNK_SAMPLES).unsqueeze(0)
    video = torch.zeros(1, VIDEO_FRAMES, 88, 88)
    with torch.no_grad():
        base, _ = m.forward_chunk(audio, video)
        for token in (14, 17):  # final audio token and final video token
            def inject(module, args):
                changed = args[0].clone()
                changed[:, token, 0] += 12
                return changed, changed, changed
            handle = attn.register_forward_pre_hook(inject)
            try:
                out, _ = m.forward_chunk(audio, video)
            finally:
                handle.remove()
            assert (out[:, :128] - base[:, :128]).abs().max() > 1e-5


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
    out = model(audio[:,:CHUNK_SAMPLES], video[:,:VIDEO_FRAMES])
    loss = (out - audio[:,:CHUNK_SAMPLES].detach()*0.3).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert video.grad is not None and video.grad.abs().sum() > 0
    assert model.video_proj.weight.grad is not None and model.video_proj.weight.grad.abs().sum() > 0
    assert model.audio_proj.weight.grad is not None and model.audio_proj.weight.grad.abs().sum() > 0
    grad = model.chunk_attention.in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
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
    torch.save({"format": "falcon-streaming-v4", "model_config": {"channels": fresh.channels, "dilations": list(fresh.dilations)}, "state_dict": fresh.state_dict(), "training_status": "random_untrained"}, checkpoint)
    del fresh
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = StreamingSession(checkpoint, device=device, max_frame_age_samples=640)
    x = np.random.default_rng(31).normal(size=CHUNK_SAMPLES).astype(np.float32)
    mouth = np.random.default_rng(32).integers(0, 256, size=(VIDEO_FRAMES,96,96), dtype=np.uint8)
    original = session.push(x, mouth, sequence=0, start_sample=0, video_samples=[0,640,1280])
    assert original.shape == (CHUNK_SAMPLES,) and np.isfinite(original).all() and session.training_status == "random_untrained"
    for bad in (
        dict(sequence=0,start_sample=CHUNK_SAMPLES,video_samples=[1920,2560,3200]),
        dict(sequence=1,start_sample=1919,video_samples=[1920,2560,3200]),
        dict(sequence=1,start_sample=CHUNK_SAMPLES,video_samples=[1921,2560,3200]),
        dict(sequence=1,start_sample=CHUNK_SAMPLES,video_samples=[1280,1280,3200]),
    ):
        with pytest.raises(ValueError): session.push(x, mouth, **bad)
        assert session.next_sequence == 1 and session.next_sample == CHUNK_SAMPLES
    session.push(x, mouth, sequence=1,start_sample=CHUNK_SAMPLES,video_samples=[1920,2560,3200])
    session.reset()
    reset = session.push(x, mouth, sequence=0, start_sample=0, video_samples=[0,640,1280])
    np.testing.assert_allclose(reset, original, rtol=0, atol=0)

@torch.no_grad()
def test_train_mode_preserves_future_and_batch_independence(model, inputs):
    model.train()
    assert all(not m.training for m in model.video_model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    audio, video = inputs
    baseline = model(audio[:,:CHUNK_SAMPLES], video[:,:VIDEO_FRAMES])
    other_a = torch.randn_like(audio[:,:CHUNK_SAMPLES]); other_v = torch.randn_like(video[:,:VIDEO_FRAMES])
    batch_a = torch.cat([other_a, audio[:,:CHUNK_SAMPLES]], 0)
    batch_v = torch.cat([other_v, video[:,:VIDEO_FRAMES]], 0)
    paired = model(batch_a, batch_v)[1:2]
    batch_a[0] = torch.randn_like(batch_a[0]) * 3
    batch_v[0] = torch.randn_like(batch_v[0]) * 3
    paired_changed = model(batch_a, batch_v)[1:2]
    torch.testing.assert_close(paired_changed, paired, rtol=0, atol=0)
    model.eval()
