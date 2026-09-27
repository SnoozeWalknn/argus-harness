"""TUI themes: Textual's built-ins plus Omarchy-style palettes, following Omarchy's theme.

On an Omarchy system ``~/.config/omarchy/current/theme`` points at the active
theme; with ``tui.theme = "auto"`` argus uses the matching palette, so it looks
like the rest of the desktop. The last theme picked with ctrl+t is remembered in
``$XDG_STATE_HOME/argus/tui.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from textual.theme import Theme

EXTRA = [
    Theme(
        name="matte-black", primary="#e68e0d", secondary="#8a8a8d", accent="#f59e0b",
        foreground="#bebebe", background="#121212", surface="#1a1a1a", panel="#262626",
        success="#8a9a5b", warning="#e68e0d", error="#d35f5f", dark=True,
    ),
    Theme(
        name="everforest", primary="#a7c080", secondary="#7fbbb3", accent="#dbbc7f",
        foreground="#d3c6aa", background="#2d353b", surface="#343f44", panel="#3d484d",
        success="#a7c080", warning="#dbbc7f", error="#e67e80", dark=True,
    ),
    Theme(
        name="kanagawa", primary="#7e9cd8", secondary="#957fb8", accent="#ffa066",
        foreground="#dcd7ba", background="#1f1f28", surface="#2a2a37", panel="#363646",
        success="#98bb6c", warning="#e6c384", error="#e82424", dark=True,
    ),
    Theme(
        name="osaka-jade", primary="#2dd5b7", secondary="#509475", accent="#e5c736",
        foreground="#c1c497", background="#111c18", surface="#1a2a24", panel="#23372f",
        success="#549e6a", warning="#e5c736", error="#ff5345", dark=True,
    ),
    Theme(
        name="ristretto", primary="#f38d70", secondary="#adda78", accent="#f9cc6c",
        foreground="#e6d9db", background="#2c2525", surface="#362c2c", panel="#403838",
        success="#adda78", warning="#f9cc6c", error="#fd6883", dark=True,
    ),
]  # fmt: skip

# The order ctrl+t cycles through.
ORDER = [
    "tokyo-night", "catppuccin-mocha", "gruvbox", "nord", "everforest", "kanagawa",
    "rose-pine", "matte-black", "osaka-jade", "ristretto", "dracula", "flexoki",
    "catppuccin-latte", "solarized-light",
]  # fmt: skip

# Omarchy theme directory name -> theme here.
OMARCHY = {
    "tokyo-night": "tokyo-night", "catppuccin": "catppuccin-mocha",
    "catppuccin-latte": "catppuccin-latte", "gruvbox": "gruvbox", "nord": "nord",
    "everforest": "everforest", "kanagawa": "kanagawa", "rose-pine": "rose-pine",
    "matte-black": "matte-black", "osaka-jade": "osaka-jade", "ristretto": "ristretto",
    "flexoki-light": "flexoki",
}  # fmt: skip
DEFAULT = "tokyo-night"


def omarchy_theme() -> str | None:
    link = Path(os.path.expanduser("~/.config/omarchy/current/theme"))
    try:
        name = (link.resolve() if link.is_symlink() else link).name
    except OSError:
        return None
    return OMARCHY.get(name)


def state_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "argus" / "tui.json"


def saved_theme() -> str | None:
    try:
        return json.loads(state_path().read_text()).get("theme")
    except (OSError, ValueError):
        return None


def save_theme(name: str) -> None:
    p = state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(p.read_text()) if p.exists() else {}
        data["theme"] = name
        p.write_text(json.dumps(data))
    except (OSError, ValueError):
        pass


def initial(configured: str, available: set[str]) -> str:
    """The configured theme, else the last one picked, else Omarchy's, else the default."""
    for name in (
        configured if configured != "auto" else None,
        saved_theme(),
        omarchy_theme(),
        DEFAULT,
    ):
        if name and name in available:
            return name
    return DEFAULT
