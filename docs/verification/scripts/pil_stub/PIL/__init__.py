"""Minimal offline stand-in for Pillow used ONLY by studio-backend verification.

It supports exactly the API surface server.py touches:
    from PIL import Image, UnidentifiedImageError
    Image.new("L", (96, 96), fill).save(fp, format="JPEG")
    with Image.open(fp) as im: im.size, im.format, im.convert("L")
Images are stored as a tiny self-describing marker, NOT real JPEG/PNG bytes.
"""
from . import Image
from .Image import UnidentifiedImageError

__version__ = "0.0.0-stub"
