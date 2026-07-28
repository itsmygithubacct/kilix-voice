"""Who owns the audio device, and who owns the session. See DESIGN.md.

Two kinds of exclusivity live here, because both are policy rather than
mechanism and neither belongs inside an engine.

The **session lock** makes exactly one process the owner of a voice session.
Two daemons sharing one session directory would fight over ``control.sock``
and over the microphone, so the second one must refuse to start rather than
quietly win a race.  The lock is an ``O_EXCL`` file holding the owner's pid: a
daemon that was killed leaves it behind, so an existing lock whose owner is
gone is taken over instead of blocking the session until somebody deletes a
file by hand.

The **half-duplex policy** is the rule that the microphone never hears the
speakers.  Opening the microphone cancels speech that is in flight — that is
barge-in, and it is why ``begin_listen`` runs the cancel hook before it marks
the session as listening.  The reverse is refused rather than pre-empted:
starting a read-aloud during dictation would speak over the person the machine
is transcribing.

The Arbiter holds no audio objects.  Cancellation is a callable the daemon
supplies, so this module stays testable with nothing installed and the policy
stays readable in one place.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

from . import paths

LOCK_BASENAME = "voiced.lock"

# A lock file is created empty and written a moment later, so one that is both
# unreadable and this new belongs to a daemon still starting up rather than to
# a crash.  Below the grace period an unparsable lock counts as held.
STALE_GRACE_S = 5.0

# Bounded, because each attempt only retries after removing a lock we proved
# was stale; a lock that keeps reappearing is another daemon, not a race.
_TAKEOVER_ATTEMPTS = 3

_MAX_LOCK_BYTES = 4096


class ArbiterError(RuntimeError):
    """The session is owned elsewhere, or a turn broke the half-duplex rule."""


def _start_ticks(pid: int) -> str:
    """Return a process's start time from /proc, or "" where unavailable.

    Pids are recycled.  Pairing the pid with the moment its process started is
    what stops a reused pid from making a session look permanently occupied,
    and costs one read of a file that is already in memory.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read(_MAX_LOCK_BYTES)
    except OSError:
        return ""
    # The comm field is parenthesised and may itself contain spaces and
    # brackets, so the fields after it are counted from the last ')'.  proc(5)
    # numbers starttime as field 22, which is index 19 of that remainder.
    fields = raw.rpartition(b")")[2].split()
    return fields[19].decode("ascii", "replace") if len(fields) > 19 else ""


def _process_alive(pid: int, start: str) -> bool:
    """Return whether the process that took a lock is still running."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, but owned by another user: not our daemon, and not ours to
        # displace either.
        return True
    except OSError:
        return True  # unable to tell; the safe reading is "still running"
    current = _start_ticks(pid)
    if start and current and current != start:
        return False
    return True


def _read_lock(path: str) -> tuple[int | None, str, float]:
    """Return (pid, start time, age in seconds) for an existing lock file."""
    try:
        info = os.stat(path)
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read(_MAX_LOCK_BYTES)
    except FileNotFoundError:
        return None, "", float("inf")
    except OSError as error:
        raise ArbiterError(
            f"cannot read the session lock {path}: {error}. Check the "
            "permissions on that file, or delete it if no kilix-voiced is "
            "running.") from error
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, raw = line.partition("=")
        if sep:
            fields[key.strip()] = raw.strip()
    try:
        pid: int | None = int(fields.get("pid", ""))
    except ValueError:
        pid = None
    return pid, fields.get("start", ""), max(0.0, time.time() - info.st_mtime)


class Arbiter:
    """Session ownership and the half-duplex speak/listen policy.

    Turn ids are the daemon's own, not the caller's: ``end_speech`` and
    ``end_listen`` ignore an id that is no longer current, so a worker thread
    finishing after its turn was pre-empted cannot clear the state of the turn
    that replaced it.
    """

    def __init__(self, session_dir: str | None = None, *,
                 cancel_speech: Callable[[], None] | None = None) -> None:
        self._session_dir = (
            os.path.abspath(os.path.expanduser(str(session_dir)))
            if session_dir else paths.session_dir())
        self._lock_path = os.path.join(self._session_dir, LOCK_BASENAME)
        self._cancel_speech = cancel_speech
        self._lock = threading.RLock()
        self._fd: int | None = None
        self._ino: int | None = None
        self._speaking: str | None = None
        self._listening: str | None = None

    @property
    def session_dir(self) -> str:
        """Directory the lock and the control socket live in."""
        return self._session_dir

    @property
    def lock_path(self) -> str:
        """Path of the single-owner lock file."""
        return self._lock_path

    @property
    def held(self) -> bool:
        """True while this process owns the session."""
        return self._fd is not None

    @property
    def speaking(self) -> bool:
        """True while a read-aloud turn owns the speakers."""
        with self._lock:
            return self._speaking is not None

    @property
    def listening(self) -> bool:
        """True while a dictation turn owns the microphone."""
        with self._lock:
            return self._listening is not None

    @property
    def speech_turn(self) -> str | None:
        """Id of the read-aloud turn in flight, or None."""
        with self._lock:
            return self._speaking

    @property
    def listen_turn(self) -> str | None:
        """Id of the dictation turn in flight, or None."""
        with self._lock:
            return self._listening

    # -- session ownership --------------------------------------------------

    def acquire_session(self) -> None:
        """Become the one owner of this voice session, or explain who is."""
        if self._fd is not None:
            raise ArbiterError(
                "acquire_session() was called twice on the same Arbiter. One "
                "Arbiter owns one session; call release() before acquiring "
                "again.")
        try:
            paths.ensure_private_dir(self._session_dir)
        except paths.PathError as error:
            raise ArbiterError(str(error)) from error
        for _ in range(_TAKEOVER_ATTEMPTS):
            try:
                handle = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600)
            except FileExistsError:
                self._take_over_stale_lock()
                continue
            except OSError as error:
                raise ArbiterError(
                    f"cannot create the session lock {self._lock_path}: "
                    f"{error}. Check that the directory is writable, or point "
                    "KILIX_SESSION_HOME at one that is.") from error
            self._claim(handle)
            return
        raise ArbiterError(
            f"could not take the session lock {self._lock_path} after "
            f"{_TAKEOVER_ATTEMPTS} attempts: it keeps being recreated. "
            "Another kilix-voiced is starting repeatedly — stop it, then "
            "start one daemon.")

    def release(self) -> None:
        """Give up session ownership. Idempotent."""
        handle, self._fd = self._fd, None
        ino, self._ino = self._ino, None
        if handle is None:
            return
        try:
            # Only unlink a lock that is still the one we created: a successor
            # that took over a lock we failed to clean up must keep its own.
            if ino is not None and os.stat(self._lock_path).st_ino == ino:
                os.unlink(self._lock_path)
        except OSError:
            pass
        try:
            os.close(handle)
        except OSError:
            pass

    def _claim(self, handle: int) -> None:
        """Record ownership in a lock file we have just created."""
        try:
            # The creation mode is masked by the process umask, which can only
            # narrow it; fchmod pins the file at exactly 0600 either way.
            os.fchmod(handle, 0o600)
            pid = os.getpid()
            payload = f"pid={pid}\nstart={_start_ticks(pid)}\n"
            os.write(handle, payload.encode("utf-8"))
            self._ino = os.fstat(handle).st_ino
        except OSError as error:
            os.close(handle)
            try:
                os.unlink(self._lock_path)
            except OSError:
                pass
            raise ArbiterError(
                f"cannot write the session lock {self._lock_path}: {error}. "
                "Check free space and the permissions on that directory."
            ) from error
        self._fd = handle

    def _take_over_stale_lock(self) -> None:
        """Remove a lock whose owner is gone, or say who still holds it."""
        pid, start, age = _read_lock(self._lock_path)
        if pid is None:
            if age < STALE_GRACE_S:
                raise ArbiterError(
                    f"another kilix-voiced is starting in this session: "
                    f"{self._lock_path} was created moments ago but names no "
                    "process yet. Wait a second and retry, or delete that file "
                    "if no daemon is running.")
        elif _process_alive(pid, start):
            raise ArbiterError(
                f"kilix-voiced is already running for this session as pid "
                f"{pid} ({self._lock_path}). Talk to the running daemon over "
                f"{os.path.join(self._session_dir, paths.CONTROL_SOCKET_NAME)}"
                f", or stop it first: kill {pid}.")
        try:
            os.unlink(self._lock_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ArbiterError(
                f"cannot remove the stale session lock {self._lock_path}: "
                f"{error}. Delete it by hand, then start kilix-voiced again."
            ) from error

    # -- half-duplex policy -------------------------------------------------

    def begin_speech(self, turn_id: str) -> None:
        """Claim the speakers for ``turn_id``, or refuse with the reason."""
        with self._lock:
            if self._listening is not None:
                raise ArbiterError(
                    "cannot read aloud while dictation is running: "
                    "kilix-voice is half-duplex, so the microphone never hears "
                    "the speakers. Send stop-dictation first, or wait for the "
                    "turn to end.")
            if self._speaking is not None and self._speaking != turn_id:
                raise ArbiterError(
                    "a read-aloud turn is already in flight. Send stop-speech "
                    "first; one turn owns the speakers at a time.")
            self._speaking = turn_id

    def end_speech(self, turn_id: str) -> None:
        """Release the speakers if ``turn_id`` is still the current turn."""
        with self._lock:
            if self._speaking == turn_id:
                self._speaking = None

    def begin_listen(self, turn_id: str) -> None:
        """Claim the microphone for ``turn_id``, cancelling speech first."""
        with self._lock:
            if self._listening is not None and self._listening != turn_id:
                raise ArbiterError(
                    "dictation is already running. Send stop-dictation first; "
                    "one turn owns the microphone at a time.")
            if self._speaking is not None:
                # Barge-in. Cancelling before the state changes means a failure
                # to silence the speakers leaves the session exactly as it was,
                # rather than opening the microphone into a room the machine is
                # still talking into.
                if self._cancel_speech is not None:
                    try:
                        self._cancel_speech()
                    except Exception as error:  # the hook is the daemon's
                        raise ArbiterError(
                            f"cannot stop read-aloud before opening the "
                            f"microphone: {error}. Send stop-speech, then "
                            "start dictation again.") from error
                self._speaking = None
            self._listening = turn_id

    def end_listen(self, turn_id: str) -> None:
        """Release the microphone if ``turn_id`` is still the current turn."""
        with self._lock:
            if self._listening == turn_id:
                self._listening = None
