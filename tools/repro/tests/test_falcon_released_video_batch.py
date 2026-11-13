import pytest
import torch

from look2hear.models.Falcon import VideoBlock
from studio.falcon_model.models.Falcon import VideoBlock as StudioSmallVideoBlock
from studio.falcon_model.models.SISNRRepel import VideoBlock as StudioBigVideoBlock


def build_block(block_type, temporal=False):
    return block_type(
        in_chan=8,
        hid_chan=5,
        norm_type="BatchNorm1d",
        video_lstm_batch_first=temporal,
    ).eval()


@pytest.mark.parametrize("block_type", [VideoBlock, StudioSmallVideoBlock, StudioBigVideoBlock])
def test_released_video_block_batch_preserves_single_item_semantics(block_type):
    torch.manual_seed(42)
    block = build_block(block_type)
    video = torch.randn(4, 8, 13)
    order = torch.tensor([2, 0, 3, 1])
    replacement = torch.cat([video[:1] * 3, video[1:]], dim=0)
    with torch.inference_mode():
        singles = torch.cat([block(row[None]) for row in video])
        torch.testing.assert_close(block(video), singles, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(block(video[order])[order.argsort()], singles, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(block(replacement)[1:], singles[1:], atol=1e-6, rtol=1e-6)
        sample = video[:1]
        residual = block.gateway(sample)
        encoded = block.projection(residual).transpose(1, 2).contiguous()
        legacy_output, _ = block.rnn(encoded)
        expected = block.residual_conv(legacy_output.transpose(1, 2).contiguous()) + residual
        assert torch.equal(block(sample), expected)


@pytest.mark.parametrize("block_type", [VideoBlock, StudioSmallVideoBlock, StudioBigVideoBlock])
def test_temporal_video_recurrence_is_explicit_and_strictly_loads_same_weights(block_type):
    torch.manual_seed(43)
    independent = build_block(block_type)
    temporal = build_block(block_type, temporal=True)
    assert list(temporal.state_dict()) == list(independent.state_dict())
    temporal.load_state_dict(independent.state_dict(), strict=True)
    video = torch.randn(3, 8, 9)
    with torch.inference_mode():
        residual = temporal.gateway(video)
        encoded = temporal.projection(residual).transpose(1, 2).contiguous()
        expected_rnn, _ = temporal.rnn(encoded)
        expected = temporal.residual_conv(expected_rnn.transpose(1, 2).contiguous()) + residual
        result = temporal(video)
        assert torch.equal(result, expected)
        assert not torch.allclose(result, independent(video))
