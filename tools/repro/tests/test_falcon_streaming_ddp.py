import torch
import pytest
from tools.streaming.train_ddp import pad_two_seconds, neg_si_snr, SEGMENT_SAMPLES


def test_two_second_training_pad_and_loss_ignores_invalid_tail():
    audio = torch.linspace(-1, 1, 32000).reshape(1, -1)
    video = torch.arange(50, dtype=torch.float32).reshape(1, 50, 1, 1).expand(-1, -1, 88, 88)
    padded, frames = pad_two_seconds(audio, video)
    assert padded.shape == (1, 32640) and frames.shape == (1, 51, 88, 88)
    torch.testing.assert_close(padded[:, :32000], audio)
    assert torch.count_nonzero(padded[:, 32000:]) == 0
    assert frames[0, 49, 0, 0].item() == frames[0, 50, 0, 0].item() == 49
    y = padded.clone().requires_grad_()
    loss = neg_si_snr(y[:, :SEGMENT_SAMPLES], audio)
    loss.backward()
    assert torch.count_nonzero(y.grad[:, SEGMENT_SAMPLES:]) == 0


def test_short_audio_or_video_rejected():
    with pytest.raises(ValueError, match="2 seconds"):
        pad_two_seconds(torch.zeros(1, 31999), torch.zeros(1, 50, 88, 88))
    with pytest.raises(ValueError, match="50 aligned"):
        pad_two_seconds(torch.zeros(1, 32000), torch.zeros(1, 49, 88, 88))


@pytest.mark.parametrize("scale", [0.0, 1e-7, 1.0])
def test_reconstruction_loss_and_gradient_match_official_pairwise(scale):
    from look2hear.losses import pairwise_neg_sisdr
    gen = torch.Generator().manual_seed(219)
    target = torch.randn(2, 32000, generator=gen) * scale
    prediction = (torch.randn(2, 32000, generator=gen) * .1).requires_grad_()
    actual = neg_si_snr(prediction, target)
    expected = pairwise_neg_sisdr(prediction[:, None], target[:, None]).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_actual = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(grad_actual, grad_expected, rtol=0, atol=0)


def test_checkpoint_publish_is_complete_and_never_replaces(tmp_path):
    from tools.streaming.train_ddp import atomic_checkpoint
    path = tmp_path / "weights.pt"
    atomic_checkpoint(path, {"step": 3})
    assert torch.load(path, weights_only=True)["step"] == 3
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"step": 4})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_positive_contrastive_uses_valid_audio_and_detaches_reference():
    from look2hear.models.FalconStreaming import FalconStreaming
    from tools.streaming.train_ddp import TrainingWrapper
    torch.manual_seed(311)
    wrapped = TrainingWrapper(FalconStreaming(channels=8, dilations=(1,)), True)
    output = torch.randn(2, SEGMENT_SAMPLES, requires_grad=True)
    target = torch.randn(2, SEGMENT_SAMPLES, requires_grad=True)
    assert wrapped.encode_waveform(output).shape == (2, 8, 250, 1)
    loss = wrapped.positive_loss(output, target)
    loss.backward()
    assert output.grad is not None and torch.count_nonzero(output.grad) > 0
    assert target.grad is None
    assert wrapped.speaker_pool.proj[0].weight.grad is not None
    assert torch.count_nonzero(wrapped.speaker_pool.proj[0].weight.grad) > 0
    assert wrapped.model.audio_proj.weight.grad is not None
    with pytest.raises(ValueError, match="exactly the valid"):
        wrapped.encode_waveform(torch.zeros(1, 33280))


def test_resume_protocol_rejects_changed_data_batch_seed_or_optimizer():
    from tools.streaming.train_ddp import validate_resume_protocol
    saved = {"train": {"kind": "tar", "sha256": "a"}, "validation": {"kind": "tar", "sha256": "b"},
             "world_size": 2, "batch_per_rank": 4, "accumulation": 4,
             "seed": 42, "weight_decay": .01, "optimizer": "AdamW"}
    validate_resume_protocol(saved, dict(saved))
    for key, value in [("train", {"kind": "tar", "sha256": "changed"}),
                       ("validation", None), ("batch_per_rank", 2),
                       ("accumulation", 2), ("world_size", 1), ("seed", 43),
                       ("weight_decay", 0.0), ("optimizer", "Adam")]:
        altered = dict(saved); altered[key] = value
        with pytest.raises(ValueError, match="resume protocol mismatch"):
            validate_resume_protocol(saved, altered)
    with pytest.raises(ValueError, match="lacks strict protocol"):
        validate_resume_protocol(None, saved)


def test_tar_split_and_identity_guards(monkeypatch):
    from types import SimpleNamespace
    import tools.streaming.train_ddp as trainer
    tr = SimpleNamespace(split="tr", samples=[{"id": "same"}])
    cv = SimpleNamespace(split="cv", samples=[{"id": "same"}])
    monkeypatch.setattr(trainer, "_collision_target_evidence",
                        lambda ds, item: (ds.split + "_source", ds.split + "_sha"))
    audit = trainer.validate_tar_splits(tr, cv)
    assert audit["matching_bare_ids"] == 1
    assert audit["same_target_source"] == audit["same_target_wav"] == 0
    with pytest.raises(ValueError, match="split=tr"):
        trainer.validate_tar_splits(SimpleNamespace(split="tt"), cv)
    with pytest.raises(ValueError, match="split=cv"):
        trainer.validate_tar_splits(tr, SimpleNamespace(split="tt"))
    monkeypatch.setattr(trainer, "_collision_target_evidence",
                        lambda ds, item: ("shared_source", ds.split + "_sha"))
    with pytest.raises(ValueError, match="same-ID target conflict"):
        trainer.validate_tar_splits(tr, cv)
    monkeypatch.setattr(trainer, "_collision_target_evidence",
                        lambda ds, item: (ds.split + "_source", "shared_sha"))
    with pytest.raises(ValueError, match="same-ID target conflict"):
        trainer.validate_tar_splits(tr, cv)

def test_partial_validation_does_not_advance_scheduler_or_best():
    from tools.streaming.train_ddp import advance_validation
    class Scheduler:
        def __init__(self): self.calls = []
        def step(self, loss): self.calls.append(loss)
    scheduler = Scheduler()
    assert advance_validation(scheduler, -10.0, -5.0, 4, True) == (-5.0, 4, False)
    assert scheduler.calls == []
    assert advance_validation(scheduler, -10.0, -5.0, 4, False) == (-10.0, 0, True)
    assert scheduler.calls == [-10.0]
    assert advance_validation(scheduler, -9.0, -10.0, 0, False) == (-10.0, 1, False)


def test_warmup_and_resume_use_absolute_update_count():
    from tools.streaming.train_ddp import warmup_learning_rate
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=.001)
    warmup_learning_rate(optimizer, .001, 0, 100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.00001)
    warmup_learning_rate(optimizer, .001, 49, 100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.0005)
    warmup_learning_rate(optimizer, .001, 99, 100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.001)
    optimizer.param_groups[0]["lr"] = .00025
    warmup_learning_rate(optimizer, .001, 101, 100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.00025)


def test_early_stop_min_delta_only_counts_meaningful_improvements():
    from tools.streaming.train_ddp import advance_validation
    class Scheduler:
        def __init__(self): self.calls = []
        def step(self, x): self.calls.append(x)
    scheduler = Scheduler()
    assert advance_validation(scheduler, -1.005, -1.0, 0, False, .01) == (-1.0, 1, False)
    assert advance_validation(scheduler, -1.02, -1.0, 1, False, .01) == (-1.02, 0, True)
    assert advance_validation(scheduler, -9.0, -1.02, 0, True, .01) == (-1.02, 0, False)
    assert scheduler.calls == [-1.005, -1.02]
