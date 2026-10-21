"""Minimal local stub of `timm` for offline static/dynamic verification.

The real project only imports `timm.models.layers.DropPath` (Falcon.py:6 and the
Studio copy). The real timm package requires torchvision, which is not installed
in this sandbox; DropPath is a trivial module, so we reproduce it faithfully.
"""
