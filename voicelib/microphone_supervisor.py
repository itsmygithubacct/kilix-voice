"""Single-threaded recorder owner. Executed directly with isolated Python."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import select
import resource
import signal
import socket
import subprocess
import sys
import time

CLEANUP_MARKER = b"KILIX_MIC_REAPED_V1"
stopping = False


def stop(_signal, _frame):
    global stopping
    stopping = True


def reap_descendants():
    """Only this dedicated process reaps its own children and adopted descendants."""
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        for value in children_file.read_text().split():
            try:
                os.kill(int(value), signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                child, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if child == 0:
                break
        time.sleep(0.005)


def main():
    # Fixed private argv prefix; the remaining argv is an ordinary recorder,
    # executed without a shell. Completion and control never reach that child.
    if len(sys.argv) < 7 or sys.argv[5] != "--":
        return 125
    guard, control, completion_fd = map(int, sys.argv[1:4])
    deadline = float(sys.argv[4])
    if (len({guard, control, completion_fd}) != 3
            or min(guard, control, completion_fd) < 3
            or not math.isfinite(deadline)
            or deadline > time.monotonic() + 120):
        return 125
    completion = socket.socket(fileno=completion_fd)
    if completion.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET:
        return 125
    for fd in (guard, control, completion_fd):
        os.set_inheritable(fd, False)
    parent = os.getppid()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    if (libc.prctl(36, 1, 0, 0, 0) != 0
            or libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0
            or libc.prctl(38, 1, 0, 0, 0) != 0):
        return 125
    if parent == 1 or os.getppid() != parent:
        return 125
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for kind, limit in ((resource.RLIMIT_FSIZE, 1024 * 1024), (resource.RLIMIT_NOFILE, 128)):
        _soft, hard = resource.getrlimit(kind)
        bound = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(kind, (bound, bound))
    poller = select.poll()
    poller.register(control, select.POLLIN | select.POLLHUP | select.POLLERR)
    child = None
    try:
        if stopping or time.monotonic() >= deadline:
            return 130
        if poller.poll(0):
            initial = os.read(control, 256)
            if not initial or initial != b"." * len(initial):
                return 130
        child = subprocess.Popen(sys.argv[6:], stdin=subprocess.DEVNULL,
                                 pass_fds=(guard,), start_new_session=True)
        # A private heartbeat makes a stalled/dead caller stop even when its
        # pipe is inherited accidentally. Its bytes are control, never audio.
        heartbeat = time.monotonic()
        while child.poll() is None and not stopping:
            now = time.monotonic()
            if now >= deadline or now - heartbeat >= 1.0:
                return 130
            events = poller.poll(20)
            if events:
                data = os.read(control, 256)
                if not data or data != b"." * len(data):
                    return 130
                heartbeat = time.monotonic()
        return 130 if stopping else child.returncode
    finally:
        # Even a successful recorder can leave an escaped process holding audio.
        reap_descendants()
        try:
            completion.send(CLEANUP_MARKER)
        except OSError:
            pass
        finally:
            completion.close()


if __name__ == "__main__":
    raise SystemExit(main())
