"""Regression tests for the RoomReader frozen-input integrity checks."""
import numpy as np
import pytest

from tools.streaming.evaluate_roomreader import array_sha256, check_clip


def test_frozen_roomreader_rejects_changed_guide_or_geometry():
    mixture = np.zeros(1280, dtype=np.float32)
    stems = np.zeros((1280, 3), dtype=np.float32)
    mouths = np.zeros((2, 88, 88), dtype=np.float32)
    present = np.array([True, False])
    observed = {
        "id": 0,
        "start_context_ms": 0,
        "end_context_ms": 80,
        "speaker_order": ["P1", "P2", "P3"],
        "mixture_crop_sha256": array_sha256(mixture),
        "stems_sha256": array_sha256(stems),
        "mouth_track_sha256": array_sha256(mouths),
        "present_sha256": array_sha256(present),
    }
    check_clip(observed, mixture, stems, mouths, present)
    with pytest.raises(ValueError, match="hash differs"):
        check_clip(observed, mixture, stems, mouths, ~present)
    with pytest.raises(ValueError, match="geometry differs"):
        check_clip(observed, mixture, stems[:, :2], mouths, present)
