"""kilix-voice — read-aloud and dictation for Kilix.

Importing this package must stay free of side effects: no audio device, no
microphone, no subprocess, no network.  It exposes the release version and
nothing else; every module is imported explicitly by its consumer.
"""

from __future__ import annotations

import pathlib


def _read_version() -> str:
    """Return the repo VERSION string, or a placeholder when it is absent.

    The version lives in one file that packaging, the TUIs and the daemon all
    read, so an installed tree without VERSION degrades instead of failing to
    import.
    """
    try:
        raw = (pathlib.Path(__file__).resolve().parent.parent / "VERSION").read_text(
            encoding="utf-8")
    except OSError:
        return "0.0.0"
    return raw.strip() or "0.0.0"


__version__ = _read_version()

__all__ = ["__version__"]
