"""Stub for `typeguard.check_argument_types` (removed in typeguard>=3).

Only `look2hear/layers/stft_tfgn.py:8` imports it, and only to assert argument
types. The installed real typeguard (via shared site-packages) no longer exports
it, so this stub keeps the module importable for offline verification.
"""


def check_argument_types(*args, **kwargs):  # pragma: no cover - no-op
    return True
