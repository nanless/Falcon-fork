"""Stub Pillow.Image (see PIL/__init__.py). No real image codec."""
import numpy as np

MAGIC = b"FALCONPILSTUB1"


class UnidentifiedImageError(Exception):
    pass


class _StubImage:
    def __init__(self, fmt, size, fill):
        self.format = fmt
        self.size = size          # (width, height) like Pillow
        self._fill = fill

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def convert(self, mode):
        width, height = self.size
        return np.full((height, width), self._fill, dtype=np.uint8)

    def save(self, fp, format=None, **kw):
        width, height = self.size
        fmt = (format or self.format or "PNG").encode("ascii")
        fp.write(MAGIC + bytes([len(fmt)]) + fmt + bytes([width & 0xFF, width >> 8,
                                                           height & 0xFF, height >> 8,
                                                           int(self._fill) & 0xFF]))


def new(mode, size, color=0):
    if mode != "L":
        raise ValueError("stub PIL only supports mode 'L'")
    return _StubImage("PNG", tuple(size), int(color))


def open(fp):
    data = fp.read() if hasattr(fp, "read") else fp
    if not isinstance(data, (bytes, bytearray)) or not data.startswith(MAGIC):
        raise UnidentifiedImageError("stub PIL cannot identify image file")
    try:
        n = data[len(MAGIC)]
        fmt = data[len(MAGIC) + 1: len(MAGIC) + 1 + n].decode("ascii")
        off = len(MAGIC) + 1 + n
        width = data[off] | (data[off + 1] << 8)
        height = data[off + 2] | (data[off + 3] << 8)
        fill = data[off + 4]
    except Exception as exc:  # noqa: BLE001
        raise UnidentifiedImageError("corrupt stub image") from exc
    if fmt not in ("JPEG", "PNG"):
        raise UnidentifiedImageError("unsupported stub format")
    return _StubImage(fmt, (width, height), fill)
