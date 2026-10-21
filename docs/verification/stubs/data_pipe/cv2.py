"""Minimal cv2 stub: only cvtColor(RGB2GRAY) and flip are referenced in transform.py."""
import numpy as np
COLOR_RGB2GRAY = 6


def cvtColor(img, code):
    assert code == COLOR_RGB2GRAY
    return np.asarray(img, dtype=np.float32) * 0 + img[..., 0]  # unused in dataset path


def flip(img, axis):
    return np.flip(img, axis=axis).copy()
