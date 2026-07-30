"""The canonical text shell for Kilix Voice screens.

Kilix TUI's VirtualBox manager established the visual contract used here:
identity and application name on row zero, numbered tabs on row one, one
divider on row two, status on row three, content below it, and a quiet footer
on the last row.  This small stdlib-only copy keeps the independently
installable voice tools consistent without adding a package dependency.
"""

from __future__ import annotations

import curses
from typing import NamedTuple, Sequence


class Body(NamedTuple):
    """The rectangle left for application-owned content."""

    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height


_ATTRS: dict[str, int] | None = None
_PAIR_BASE = 16


def _attrs() -> dict[str, int]:
    """Resolve the shared blue/red/white/grey Tango roles."""
    global _ATTRS
    if _ATTRS is not None:
        return _ATTRS
    try:
        if not curses.has_colors():
            raise curses.error("no colours")
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        pairs = (
            ("title", curses.COLOR_WHITE, background),
            ("accent", curses.COLOR_BLUE, background),
            ("alert", curses.COLOR_RED, background),
            ("muted", curses.COLOR_WHITE, background),
            ("selected", curses.COLOR_WHITE, curses.COLOR_BLUE),
            ("danger", curses.COLOR_WHITE, curses.COLOR_RED),
        )
        resolved: dict[str, int] = {}
        for index, (role, foreground, pair_background) in enumerate(pairs):
            curses.init_pair(
                _PAIR_BASE + index, foreground, pair_background)
            resolved[role] = curses.color_pair(_PAIR_BASE + index)
        resolved["title"] |= curses.A_BOLD
        resolved["muted"] |= curses.A_DIM
        resolved["selected"] |= curses.A_BOLD
        resolved["danger"] |= curses.A_BOLD
        _ATTRS = resolved
    except curses.error:
        _ATTRS = {
            "title": curses.A_BOLD,
            "accent": curses.A_NORMAL,
            "alert": curses.A_BOLD,
            "muted": curses.A_DIM,
            "selected": curses.A_REVERSE | curses.A_BOLD,
            "danger": curses.A_REVERSE | curses.A_BOLD,
        }
    return _ATTRS


def attr(role: str) -> int:
    """Return one of the canonical shell's semantic attributes."""
    return _attrs().get(role, curses.A_NORMAL)


def reset() -> None:
    """Forget cached terminal attributes between curses sessions or tests."""
    global _ATTRS
    _ATTRS = None


def put(screen, row: int, column: int, value: object, role: str = "") -> None:
    """Write one clipped, resize-safe string."""
    try:
        height, width = screen.getmaxyx()
    except Exception:
        height, width = 24, 80
    if not (0 <= row < height) or width <= 0:
        return
    column = max(0, column)
    if column >= width:
        return
    text = str(value)[:max(0, width - column - 1)]
    if not text:
        return
    style = attr(role) if role else curses.A_NORMAL
    try:
        screen.addnstr(row, column, text, len(text), style)
    except AttributeError:
        try:
            screen.addstr(row, column, text, style)
        except (curses.error, UnicodeEncodeError):
            pass
    except (curses.error, UnicodeEncodeError):
        pass


def draw(
    screen,
    *,
    title: str,
    sections: Sequence[str] = ("Overview",),
    active: int = 0,
    summary: str = "",
    footer: str = "",
    summary_role: str = "muted",
) -> Body:
    """Draw the canonical four-row Kilix frame and return its body."""
    try:
        height, width = screen.getmaxyx()
    except Exception:
        height, width = 24, 80
    if height <= 0 or width <= 0:
        return Body(0, 0, 0, 0)

    left = 1 if width > 2 else 0
    inner_width = max(0, width - (2 if width > 2 else 1))
    put(screen, 0, left, "KILIX TUI"[:inner_width], "title")

    strap = str(title)
    strap_column = width - len(strap) - 1
    if strap and strap_column > left + len("KILIX TUI"):
        put(screen, 0, strap_column, strap, "muted")

    column = left
    for index, label in enumerate(tuple(sections) or ("Overview",)):
        marker = "▶" if index == active else " "
        text = f"{marker}{index + 1} {label} "
        if column + len(text) >= width:
            break
        put(
            screen,
            1,
            column,
            text,
            "selected" if index == active else "muted",
        )
        column += len(text)

    put(screen, 2, 0, "─" * max(0, width - 1), "muted")
    put(screen, 3, left, str(summary)[:inner_width], summary_role)
    put(screen, height - 1, left, str(footer)[:inner_width], "muted")
    return Body(4, left, max(0, height - 5), inner_width)
