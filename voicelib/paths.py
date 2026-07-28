"""Filesystem layout for kilix-voice. Frozen contract — see DESIGN.md.

Every path is derived from the Kilix environment, so a session that relocates
its storage relocates the voice sockets, models and library with it.  Nothing
here writes outside the Kilix-owned tree, and nothing creates a socket: the
control socket is the daemon's, and the per-pane dictation sockets belong to
the kitty fork.
"""

from __future__ import annotations

import os
import re
import stat

SETTINGS_BASENAME = "settings.conf"
CONTROL_SOCKET_NAME = "control.sock"
VOICE_LEAF = "voice"

DIR_MODE = 0o700
SOCKET_MODE = 0o600

# Pane ids and catalog ids are turned into filenames under the session and data
# directories, so they are restricted to a leaf-safe alphabet: no separator, no
# "..", nothing that could climb out of the directory it names.
_PANE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CATALOG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class PathError(RuntimeError):
    """A required directory is missing, foreign-owned, or not private."""


def _expand(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value))


def gpu_terminal_home() -> str:
    """Return the root of the writable state shared by the whole stack."""
    return _expand(os.environ.get("GPU_TERMINAL_HOME")
                   or os.path.join(os.path.expanduser("~"), ".local", "gpu_terminal"))


def storage_home() -> str:
    """Return the root of Kilix-owned writable files."""
    return _expand(os.environ.get("KILIX_STORAGE_HOME")
                   or os.path.join(gpu_terminal_home(), "kilix"))


def settings_file() -> str:
    """Return the shared KEY=value settings document Kilix's SDK writes."""
    override = os.environ.get("GPU_TERMINAL_SETTINGS_FILE")
    if override:
        return _expand(override)
    return os.path.join(gpu_terminal_home(), SETTINGS_BASENAME)


def _owned_dir(env_name: str, leaf: str) -> str:
    return _expand(os.environ.get(env_name) or os.path.join(storage_home(), leaf))


def session_home() -> str:
    """Return the Kilix session directory (cleared between sessions)."""
    return _owned_dir("KILIX_SESSION_HOME", "session")


def data_home() -> str:
    """Return the Kilix data directory (durable across sessions)."""
    return _owned_dir("KILIX_DATA_HOME", "data")


def session_dir() -> str:
    """Return the voice session directory: the only place we may bind."""
    return os.path.join(session_home(), VOICE_LEAF)


def data_dir() -> str:
    """Return the voice data directory: models and the vosk library."""
    return os.path.join(data_home(), VOICE_LEAF)


def control_socket() -> str:
    """Return the daemon's control socket path."""
    return os.path.join(session_dir(), CONTROL_SOCKET_NAME)


def dictate_socket(pane_id: str | int) -> str:
    """Return the dictation socket path for a kitty pane.

    The socket itself is created by the kitty fork; this only names it so both
    sides agree, and so a request naming a pane cannot name a path.
    """
    token = str(pane_id)
    if not _PANE_ID.match(token):
        raise PathError(
            f"invalid kitty pane id {pane_id!r}: expected 1-64 characters from "
            "[A-Za-z0-9_-]. Pass the pane id kitty reported, for example 3.")
    return os.path.join(session_dir(), f"dictate-{token}.sock")


def lib_dir() -> str:
    """Return the directory holding the currently selected native library."""
    return os.path.join(data_dir(), "lib", "current")


def libvosk_path() -> str:
    """Return the expected libvosk.so path (may not exist)."""
    return os.path.join(lib_dir(), "libvosk.so")


def models_dir() -> str:
    """Return the directory holding downloaded speech models."""
    return os.path.join(data_dir(), "models")


def model_dir(catalog_id: str) -> str:
    """Return the directory for one catalog model id (may not exist)."""
    token = str(catalog_id)
    if not _CATALOG_ID.match(token):
        raise PathError(
            f"invalid model id {catalog_id!r}: expected 1-64 characters from "
            "[A-Za-z0-9._-] starting with a letter or digit. Use a catalog id "
            "such as small-en-us.")
    return os.path.join(models_dir(), token)


def ensure_private_dir(path: str, mode: int = DIR_MODE) -> str:
    """Create ``path`` with ``mode`` if absent, and prove it is private.

    An existing directory is verified rather than repaired: sockets carrying
    recognised speech live here, so a directory owned by somebody else, or one
    another user can enter, is a fault to report — not something to silently
    chmod out from under whoever made it that way.
    """
    target = _expand(str(path))
    parent = os.path.dirname(target)
    try:
        if parent and parent != target:
            os.makedirs(parent, mode=DIR_MODE, exist_ok=True)
    except OSError as error:
        raise PathError(
            f"cannot create {parent}: {error}. Create it yourself, or point "
            "KILIX_SESSION_HOME / KILIX_DATA_HOME at a writable directory."
        ) from error
    created = True
    try:
        os.mkdir(target, mode)
    except FileExistsError:
        created = False
    except OSError as error:
        raise PathError(
            f"cannot create {target}: {error}. Check the permissions on "
            f"{parent} and free space on that filesystem.") from error

    # O_NOFOLLOW rejects a symlink in the final position, which mkdir(exist_ok)
    # would have accepted; O_DIRECTORY rejects a regular file left in its place.
    # Inspecting the opened descriptor, rather than the name, means the checks
    # below describe the object we would actually use.
    try:
        handle = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise PathError(
            f"{target} is not a usable private directory ({error}). It must be "
            "a real directory, not a symlink or a file — remove or rename what "
            "is there and let kilix-voice recreate it.") from error
    try:
        info = os.fstat(handle)
        if created:
            # The process umask would otherwise widen a directory we just made.
            os.fchmod(handle, mode)
            return target
        if info.st_uid != os.geteuid():
            raise PathError(
                f"{target} is owned by uid {info.st_uid}, not by you "
                f"(uid {os.geteuid()}). Voice sockets carry your dictation: "
                "remove that directory or run as its owner.")
        actual = stat.S_IMODE(info.st_mode)
        if actual != mode:
            raise PathError(
                f"{target} has mode {actual:04o}, expected {mode:04o}. Run: "
                f"chmod {mode:o} {target}")
    finally:
        os.close(handle)
    return target
