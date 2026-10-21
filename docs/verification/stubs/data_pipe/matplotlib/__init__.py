"""Minimal matplotlib stub: repo datas import pyplot but never plot in the data path.
torchmetrics/plot.py (pulled in by pytorch_lightning) probes matplotlib; we provide the
minimum surface it touches at import time."""
__version__ = "0.0.0-stub"

from . import colors, axes, figure, pyplot  # noqa: E402,F401
