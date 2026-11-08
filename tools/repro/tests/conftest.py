"""Shared fixtures for ``tools/repro/tests``.

`pl.Trainer()` installs a default ``ModelCheckpoint`` callback that writes
``<cwd>/checkpoints/epoch=*-step=*-vNN.ckpt`` unless the Trainer is told
otherwise.  ``pytest`` runs with the repo root as cwd, and ``.gitignore``
ignores ``*.ckpt``, so this litter is invisible to ``git status`` -- 87 stale
``epoch=0-step=2-vNN.ckpt`` stubs had piled up in the repo root before this
guard existed.

The guard below makes the litter visible: if a test lets a Trainer write new
``checkpoints/*.ckpt`` files into the repo root, *that test* fails with a hint
on how to fix it.
"""
import glob
import os

import pytest

_REPO_ROOT = os.path.dirname(  # tools/repro/tests -> repo root
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
_CKPT_GLOB = os.path.join(_REPO_ROOT, "checkpoints", "*.ckpt")


def _repo_root_ckpts():
    return set(glob.glob(_CKPT_GLOB))


@pytest.fixture(autouse=True)
def guard_repo_root_checkpoints():
    """Fail the offending test if a Trainer writes `checkpoints/*.ckpt` at repo root."""
    before = _repo_root_ckpts()
    yield
    leaked = sorted(_repo_root_ckpts() - before)
    if leaked:
        pytest.fail(
            "test wrote Lightning checkpoints into the repo root: "
            + ", ".join(os.path.relpath(p, _REPO_ROOT) for p in leaked)
            + "\nThese are hidden from `git status` because .gitignore ignores *.ckpt."
            "\nFix: pass `enable_checkpointing=False` and "
            "`default_root_dir=str(tmp_path)` to `pl.Trainer(...)`"
            " (see test_litmodule_logging.py::_build), then delete the leaked files.",
            pytrace=False,
        )
