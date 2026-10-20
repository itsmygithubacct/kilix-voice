"""Teardown of children launched with start_new_session=True."""
import os
import signal
import subprocess


def kill_group(process: subprocess.Popen) -> None:
    """Kill the owned session's process group, even after its leader exits.

    Only call for a child we started in its own session: its original PID is
    the group ID. Do not poll or look up its current pgid first; a reaped
    leader can still have descendants holding the provider or capture pipes.
    SIGKILL bounds teardown even for stopped or SIGTERM-ignoring descendants.
    The caller retains ownership of pipe closure and immediate-child reaping.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # the whole group has already exited
