from contextlib import contextmanager

from .axes import Axes
from .figure import Figure


class _Style:
    @contextmanager
    def context(self, *a, **k):
        yield


style = _Style()


def figure(*a, **k):
    return Figure()


def subplots(*a, **k):
    return Figure(), Axes()


def imshow(*a, **k):
    raise NotImplementedError("stub")


def savefig(*a, **k):
    raise NotImplementedError("stub")


def show(*a, **k):
    return None


def subplot(*a, **k):
    return Axes()


def title(*a, **k):
    return None


def close(*a, **k):
    return None
