"""Bounded microphone ownership with authenticated complete-cleanup proof.

No constructor opens audio. Callbacks must be bounded and return actual bools;
unknown lock state refuses capture. The shared namespace is a microphone domain,
separate from model/accelerator scheduling. A grant is no hardware qualification.
"""
from __future__ import annotations

import array
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import select
import socket
import struct
import subprocess
import threading
import time
from typing import Callable
import uuid

from . import device_leases

MAX_SECONDS = 120.0
CLEANUP_MARKER = b"KILIX_MIC_REAPED_V1"


class CaptureError(RuntimeError):
    """Stable capture refusal; no audio or transcript in error text."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ReceiverLifetime:
    """Bounded peer check, including datagram receivers which retain their path.

    The caller supplies a valid unchanged partial/status heartbeat, never final
    text or a key. An unresponsive receiver stops capture instead of buffering
    further audio. This borrows the socket and changes neither ownership nor mode.
    """

    def __init__(self, receiver: socket.socket, heartbeat: Callable[[], bytes]):
        self.receiver, self.heartbeat = receiver, heartbeat
        self._next = 0.0
        self._gone = False

    def __call__(self) -> bool:
        if self._gone:
            return True
        try:
            poller = select.poll()
            poller.register(self.receiver.fileno(), select.POLLHUP | select.POLLERR | select.POLLNVAL)
            if poller.poll(0):
                self._gone = True
            elif time.monotonic() >= self._next:
                data = self.heartbeat()
                if type(data) is not bytes or not 0 < len(data) <= 65536:
                    raise ValueError("Invalid microphone heartbeat")
                if self.receiver.send(data, socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL) != len(data):
                    raise OSError("Short microphone heartbeat")
                self._next = time.monotonic() + 0.25
        except (OSError, ValueError):
            self._gone = True
        return self._gone


class LogindLockState:
    """Bounded actual logind adapter: True=locked/inactive, None=unknown.

    Select an explicit login session or XDG_SESSION_ID, otherwise this user's
    logind Display session. Verify its User, Active and LockedHint properties.
    An unavailable manager, foreign session or malformed answer is never False.
    Cached answers live at most 100 ms; each subprocess is bounded to 150 ms.
    """

    def __init__(self, session: str | None = None):
        self.session = session if session is not None else os.environ.get("XDG_SESSION_ID")
        self._until = 0.0
        self._value: bool | None = None

    @staticmethod
    def _query(*arguments: str) -> str:
        result = subprocess.run(["/usr/bin/loginctl", *arguments, "--no-pager"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=0.15, check=True, text=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "SYSTEMD_PAGER": "cat"})
        if len(result.stdout) > 4096:
            raise ValueError("Oversized lock-state answer")
        return result.stdout.strip()

    def __call__(self) -> bool | None:
        now = time.monotonic()
        if now < self._until:
            return self._value
        value = None
        try:
            session = self.session or self._query("show-user", str(os.geteuid()), "-p", "Display", "--value")
            if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", session, re.ASCII):
                raise ValueError("Invalid session identity")
            answer = self._query("show-session", session, "-p", "User", "-p", "Active", "-p", "LockedHint")
            rows = [line.split("=", 1) for line in answer.splitlines()]
            if len(rows) != 3 or any(len(row) != 2 for row in rows):
                raise ValueError("Invalid lock-state answer")
            values = dict(rows)
            if (set(values) != {"User", "Active", "LockedHint"}
                    or values["User"] != str(os.geteuid())
                    or values["Active"] not in ("yes", "no")
                    or values["LockedHint"] not in ("yes", "no")):
                raise ValueError("Unknown session state")
            value = values["Active"] != "yes" or values["LockedHint"] == "yes"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        self._value, self._until = value, time.monotonic() + 0.1
        return value


@dataclass(frozen=True)
class CaptureControl:
    deadline: float
    cancelled: Callable[[], bool] | None = None
    disconnected: Callable[[], bool] | None = None
    locked: Callable[[], bool | None] | None = None
    indicator: Callable[[str], None] | None = None
    namespace: str | None = None

    def __post_init__(self):
        if (type(self.deadline) not in (float, int) or self.deadline < 0
                or self.deadline > time.monotonic() + MAX_SECONDS
                or not math.isfinite(self.deadline)
                or any(callback is not None and not callable(callback) for callback in
                       (self.cancelled, self.disconnected, self.locked, self.indicator))):
            raise CaptureError("invalid-control", "Capture needs bounded controls and a deadline within 120 seconds")
        if self.locked is None:
            object.__setattr__(self, "locked", LogindLockState())

    def check(self):
        if time.monotonic() >= self.deadline:
            raise CaptureError("deadline", "Microphone capture deadline expired")
        for code, callback in (("cancelled", self.cancelled), ("disconnected", self.disconnected), ("locked", self.locked)):
            if callback is None:
                continue
            try:
                value = callback()
            except Exception as error:
                raise CaptureError("control-unavailable", "Microphone control authority failed") from error
            if type(value) is not bool:
                raise CaptureError("control-unavailable", "Microphone control state is unknown")
            if value:
                raise CaptureError(code, "Microphone capture stopped: " + code)
        if time.monotonic() >= self.deadline:
            raise CaptureError("deadline", "Microphone capture deadline expired")


def cleanup_proven(channel: socket.socket, pid: int) -> bool:
    """Require one supervisor-credentialed record and true EOF; close all rights."""
    credentials = struct.pack("3i", pid, os.geteuid(), os.getegid())
    try:
        channel.settimeout(0.25)
        records = []
        for _ in range(2):
            data, ancillary, flags, _address = channel.recvmsg(64, socket.CMSG_SPACE(256), socket.MSG_CMSG_CLOEXEC)
            for level, kind, value in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors = array.array("i")
                    descriptors.frombytes(value[:len(value) - len(value) % descriptors.itemsize])
                    for fd in descriptors:
                        os.close(fd)
            # Linux echoes this requested receive option in msg_flags.
            records.append((data, ancillary, flags & ~socket.MSG_CMSG_CLOEXEC))
        return records == [(CLEANUP_MARKER, [(socket.SOL_SOCKET, socket.SCM_CREDENTIALS, credentials)], 0), (b"", [], 0)]
    except (OSError, ValueError):
        return False


class CaptureOwner:
    """Own one dedicated supervisor, its guard and proof. stop() is idempotent."""

    def __init__(self, control: CaptureControl):
        self.control = control
        self.phase = "inactive"
        self.cleanup_complete = True
        self.error: CaptureError | None = None
        self.process: subprocess.Popen | None = None
        self._lease = None
        self._channel: socket.socket | None = None
        self._writer = -1
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._done = threading.Event()

    def _phase(self, phase: str):
        self.phase = phase
        if self.control.indicator is not None:
            try:
                self.control.indicator(phase)
            except Exception as error:
                raise CaptureError("indicator-unavailable", "Microphone indicator failed") from error

    def start(self, command: list[str], stderr) -> subprocess.Popen:
        if self.process is not None or not self.cleanup_complete:
            raise CaptureError("unavailable", "Microphone owner cannot be reused")
        self.control.check()

        def queued(_status):
            raise CaptureError("busy", "Microphone is already in use")

        sender = None
        reader = -1
        try:
            self._lease = device_leases.acquire(job_id=uuid.uuid4().hex, workload="stt-job", device="microphone",
                deadline=self.control.deadline, cancelled=self.control.cancelled, disconnected=self.control.disconnected,
                progress=queued, namespace=self.control.namespace or f"/run/user/{os.geteuid()}/kilix-microphone-leases-v1")
            self.control.check()
            self._phase("recording")
            self._channel, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
            self._channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            reader, self._writer = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
            guard = self._lease.guard_fd
            self.control.check()
            self.process = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "-S",
                str(Path(__file__).with_name("microphone_supervisor.py")), str(guard), str(reader),
                str(sender.fileno()), str(self.control.deadline), "--", *command],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, bufsize=0,
                pass_fds=(guard, reader, sender.fileno()), start_new_session=True,
                env={key: value for key, value in os.environ.items()
                     if not key.startswith(("LD_", "PYTHON"))})
            self.cleanup_complete = False
            self._watcher = threading.Thread(target=self._watch, name="kilix-microphone-owner", daemon=True)
            self._watcher.start()
            return self.process
        except BaseException:
            if self.process is not None:
                self._stop.set()
                if self._watcher is None or self._watcher.ident is None:
                    # Thread allocation failed after Popen. Close this sender
                    # before synchronously reading the supervisor's true EOF.
                    if sender is not None:
                        sender.close()
                        sender = None
                    self._watch()
                else:
                    self.stop()
                if self.process.stdout is not None:
                    self.process.stdout.close()
            else:
                self._finish(True)
            raise
        finally:
            if reader >= 0:
                os.close(reader)
            if sender is not None:
                sender.close()

    def _watch(self):
        try:
            while self.process.poll() is None:
                if self._stop.is_set():
                    break
                try:
                    self.control.check()
                    os.write(self._writer, b".")
                except (CaptureError, OSError) as error:
                    self.error = error if isinstance(error, CaptureError) else CaptureError("control-unavailable", "Microphone control pipe failed")
                    break
                self._stop.wait(0.05)
            try:
                self._phase("stopping")
            except CaptureError as error:
                self.error = error
            if self._writer >= 0:
                os.close(self._writer)
                self._writer = -1
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            proven = self.process.poll() is not None and cleanup_proven(self._channel, self.process.pid)
            self._finish(proven)
        except BaseException:
            # Unexpected watcher failure never converts lack of proof to release.
            self._finish(False)
        finally:
            self._done.set()

    def _finish(self, proven: bool):
        if self._writer >= 0:
            os.close(self._writer)
            self._writer = -1
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        if self._lease is not None:
            try:
                self._lease.release(cleanup_complete=proven)
            except device_leases.LeaseError:
                proven = False
            finally:
                self._lease = None
        self.cleanup_complete = proven
        if not proven:
            self.error = CaptureError("unavailable", "Microphone cleanup is unproven; ownership remains unavailable")
        try:
            self._phase("inactive" if proven else "unavailable")
        except CaptureError as error:
            self.error = error
            self.phase = "unavailable"

    def stop(self) -> bool:
        if self.process is None:
            return self.cleanup_complete
        self._stop.set()
        if (self._watcher is not None and self._watcher.ident is not None
                and self._watcher is not threading.current_thread()):
            self._watcher.join(timeout=4.0)
        if not self._done.is_set():
            self.error = CaptureError("unavailable", "Microphone cleanup is unproven; ownership remains unavailable")
            try:
                self._phase("unavailable")
            except CaptureError:
                pass
            return False
        return self.cleanup_complete

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)
