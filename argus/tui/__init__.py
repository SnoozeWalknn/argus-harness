"""Keyboard-driven terminal UI (Textual), a thin layer over the headless core.

Needs the ``tui`` extra: ``pip install 'argus-harness[tui]'``.
"""

from __future__ import annotations


def available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True
