"""Teardown of children launched with start_new_session=True."""
import os
import signal
import subprocess
import threading
import time


# Only serializes the signal and its completion marker, never a process wait.
_group_lock = threading.Lock()


def require_waitable_children() -> None:
    """Establish exclusive reaping before creating an owned group leader."""
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        # An inherited SIG_IGN auto-reaps on Linux; a custom handler may
        # reap behind the owner too. Neither can retain a zombie's identity.
        raise OSError("process-group ownership requires the default SIGCHLD disposition")


def poll_exit(process: subprocess.Popen) -> int | None:
    """Observe exit without reaping the leader or releasing its group ID.

    Owners check require_waitable_children before launch and use this instead
    of Popen.poll/wait until kill_group succeeds. They never install a reaper.
    The unreaped leader (including a zombie) reserves the numeric identity.
    Only the teardown owner reaps, after the last possible group signal.
    """
    if process.returncode is not None:
        return process.returncode
    try:
        info = os.waitid(os.P_PID, process.pid,
                         os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        # Teardown may have reaped concurrently, after sealing group signals.
        return process.returncode
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def wait_exit(process: subprocess.Popen, timeout: float) -> int:
    """Bounded observation for a capture reader; leave reaping to stop()."""
    deadline = time.monotonic() + timeout
    while True:
        code = poll_exit(process)
        if code is not None:
            return code
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(.01, left))


def kill_group(process: subprocess.Popen) -> None:
    """Send the owned group's final signal once, before the leader is reaped.

    Only call for a child we started in its own session: its original PID is
    the group ID. Owners retain that leader with poll_exit/wait_exit, even
    after exit, so descendants holding pipes cannot outlive group teardown.
    SIGKILL bounds teardown even for stopped or SIGTERM-ignoring descendants.
    After success, seal off every later signal before allowing the owner to
    wait(). This also makes concurrent request_stop/stop and cancel/close
    safe. An exception before completion leaves ownership available to retry.

    This is group cleanup, not containment: setsid/setpgid can escape it.
    """
    with _group_lock:
        if getattr(process, "_voice_group_signalled", False):
            return
        if process.returncode is not None:
            # Defensive refusal for an object reaped outside our ownership
            # protocol. Its integer alone can no longer authorize a signal.
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # no live member remains
        process._voice_group_signalled = True
