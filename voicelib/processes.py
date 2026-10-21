"""Teardown of children launched with start_new_session=True."""
from .lifecycle import Resource, closing

import contextlib
import errno
import logging
import json
from pathlib import Path
import secrets
import select
import socket
import sys
import os
import signal
import subprocess
import threading
import time


# Only serializes the signal and its completion marker, never a process wait.
_group_lock = threading.RLock()
PIDFD_SIGNAL_PROCESS_GROUP = 1 << 2
_pidfd_available = hasattr(os, 'pidfd_open')
_pidfd_group_supported = None if hasattr(signal, 'pidfd_send_signal') else False


def _close_pidfd(process):
    fd = getattr(process, '_voice_pidfd', None)
    if fd is not None:
        os.close(fd)
        process._voice_pidfd = None


def _outside_reap(process):
    if not getattr(process, '_voice_outside_reap', False):
        process._voice_outside_reap = True
        logging.getLogger(__name__).warning(
            'process group %s: outside reap; cleanup unproven', process.pid)
    _close_pidfd(process)


def _waitid(process):
    fd = getattr(process, '_voice_pidfd', None)
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    if fd is not None and hasattr(os, 'P_PIDFD'):
        try:
            return os.waitid(os.P_PIDFD, fd, flags)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
            # Linux 5.3 has pidfd_open but not waitid(P_PIDFD).
    return os.waitid(os.P_PID, process.pid, flags)


def _pidfd_pid(fd):
    for line in Path(f'/proc/self/fdinfo/{fd}').read_text().splitlines():
        if line.startswith('Pid:'):
            return int(line.split()[1])
    raise OSError('pidfd identity unavailable')


def _verify_spawn_identity(process, nonce):
    """The trusted launcher is still gated; it cannot normally exit or exec."""
    fd = process._voice_pidfd
    if fd is not None and _pidfd_pid(fd) != process.pid:
        return False
    status = Path(f'/proc/{process.pid}/status').read_text()
    parent = next(line for line in status.splitlines() if line.startswith('PPid:'))
    if int(parent.split()[1]) != os.getpid():
        return False
    if _member_stat(process.pid)[1] != process.pid:
        return False
    environ = Path(f'/proc/{process.pid}/environ').read_bytes().split(b'\0')
    if ('_KILIX_GROUP_NONCE=' + nonce).encode() not in environ:
        return False
    if _waitid(process) is not None:
        return False  # the gated launcher died before admission
    if fd is not None:
        # On Linux 5.3 waitid(P_PIDFD) falls back to P_PID. Poll and fdinfo
        # still refer to the original process, even if an outside reaper ran.
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        if poller.poll(0) or _pidfd_pid(fd) != process.pid:
            return False
    return True


def _identity_unproven(process):
    process._voice_identity_unproven = True
    logging.getLogger(__name__).warning(
        'process group %s: acquisition identity unproven; cleanup unproven', process.pid)
    _close_pidfd(process)


def _member_stat(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return fields[0], int(fields[2]), fields[19]  # state, pgrp, starttime


def _scan_group_members(group):
    """Open stable handles, then validate membership again after each open."""
    members = []
    try:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            fd = None
            try:
                before = _member_stat(pid)
                if before[1] != group or before[0] in ('Z', 'X'):
                    continue
                fd = os.pidfd_open(pid)
                after = _member_stat(pid)
                if after[1:] != before[1:] or _pidfd_pid(fd) != pid:
                    continue
                members.append(fd)
                fd = None
            except (FileNotFoundError, ProcessLookupError):
                pass  # member exited during enumeration
            finally:
                if fd is not None:
                    os.close(fd)
        return members
    except BaseException:
        for fd in members:
            os.close(fd)
        raise


def _check_leader(process):
    # Never replace the stable leader check with a check of its number alone.
    fd = process._voice_pidfd
    if _pidfd_pid(fd) != process.pid:
        raise ChildProcessError('outside reap')
    _waitid(process)
    if _pidfd_pid(fd) != process.pid:
        raise ChildProcessError('outside reap')


def _kill_members(process):
    """Older pidfd kernels: bounded rescans, with no numeric signal at all."""
    deadline = time.monotonic() + 1.0
    while True:
        _check_leader(process)
        members = _scan_group_members(process.pid)
        try:
            # The unreaped leader pins the group number over the entire scan.
            # If that proof was lost, none of the collected handles is used.
            _check_leader(process)
            if not members:
                return
            for fd in members:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
                except ProcessLookupError:
                    pass  # this exact member already exited
        finally:
            for fd in members:
                os.close(fd)
        if time.monotonic() >= deadline:
            raise TimeoutError('process group members remain; cleanup unproven')
        time.sleep(.002)


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
    with _group_lock:
        if getattr(process, '_voice_outside_reap', False) or getattr(process, '_voice_identity_unproven', False):
            return None
        if process.returncode is not None:
            return process.returncode
        try:
            info = _waitid(process)
        except ChildProcessError:
            _outside_reap(process)
            return None
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
        if getattr(process, '_voice_identity_unproven', False):
            raise ChildProcessError('acquisition identity unproven')
        if getattr(process, '_voice_outside_reap', False):
            raise ChildProcessError('outside reap; process group cleanup is unproven')
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
    global _pidfd_group_supported
    with _group_lock:
        if getattr(process, "_voice_group_signalled", False):
            return
        if getattr(process, '_voice_outside_reap', False) or getattr(process, '_voice_identity_unproven', False):
            return
        if process.returncode is not None:
            # Defensive refusal for an object reaped outside our ownership
            # protocol. Its integer alone can no longer authorize a signal.
            return
        fd = getattr(process, '_voice_pidfd', None)
        try:
            if fd is not None and _pidfd_group_supported is not False:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL, None,
                                             PIDFD_SIGNAL_PROCESS_GROUP)
                except OSError as error:
                    if error.errno != errno.EINVAL:
                        raise
                    _pidfd_group_supported = False
                else:
                    _pidfd_group_supported = True
                    process._voice_group_signalled = True
                    return
            if fd is not None:
                try:
                    _check_leader(process)
                except ChildProcessError:
                    _outside_reap(process)
                    return
            if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
                raise OSError('changed SIGCHLD disposition; group cleanup unproven')
            if fd is not None:
                try:
                    _kill_members(process)
                except ChildProcessError:
                    _outside_reap(process)
                    return
            else:
                # Only kernels without pidfd_open use numeric signals. This
                # path still requires exclusive reaping through the syscall.
                try:
                    _waitid(process)
                except ChildProcessError:
                    _outside_reap(process)
                    return
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # ESRCH alone is not proof: the leader may have been reaped by
            # somebody else while descendants still own devices or pipes.
            if poll_exit(process) is None:
                _outside_reap(process)
                return
        process._voice_group_signalled = True


class GroupPopen(subprocess.Popen):
    """A pipe client whose leader reserves its group until the final signal.

    communicate() also calls wait(), so the ordering must live on the process
    itself. Observing a normal exit never permits surviving pipe holders to
    escape the group's final signal. Only this object's owner may reap it.
    """

    def __init__(self, *args, **kwargs):
        global _pidfd_available
        require_waitable_children()
        self._voice_pidfd = None
        self._voice_identity_unproven = False
        kwargs['start_new_session'] = True
        if not _pidfd_available:
            super().__init__(*args, **kwargs)
            return
        # A gated exec avoids the uncheckable nonce of an already dead child.
        # No preexec_fn is used: that would be unsafe in this threaded daemon.
        command = args[0] if args else kwargs.pop('args')
        argv = [os.fsdecode(command)] if isinstance(command, (str, bytes, os.PathLike)) else [os.fsdecode(x) for x in command]
        executable = kwargs.pop('executable', None)
        if kwargs.pop('shell', False):
            argv = [os.fsdecode(executable or '/bin/sh'), '-c', *argv]
            executable = argv[0]
        executable = os.fsdecode(executable) if executable is not None else argv[0]
        nonce = secrets.token_hex(32)
        env = kwargs.get('env')
        env = {os.fsdecode(k): os.fsdecode(v) for k, v in (os.environ if env is None else env).items()}
        target_env = env.copy()
        env['_KILIX_GROUP_NONCE'] = nonce
        kwargs['env'] = env
        parent, child = socket.socketpair()
        try:
            kwargs['pass_fds'] = (*kwargs.get('pass_fds', ()), child.fileno())
            launch = [sys.executable, '-I', '-B', '-S', str(Path(__file__).with_name('process_launch.py')), str(child.fileno())]
            super().__init__(launch, *args[1:], **kwargs)
            self.args = command
            child.close()
            if parent.recv(1) != b'R':
                _identity_unproven(self)
                return
            try:
                self._voice_pidfd = os.pidfd_open(self.pid)
            except OSError as error:
                if error.errno == errno.ENOSYS:
                    _pidfd_available = False
                else:
                    _identity_unproven(self)
                    return
            else:
                os.set_inheritable(self._voice_pidfd, False)
            try:
                verified = _verify_spawn_identity(self, nonce)
            except (OSError, ValueError, StopIteration):
                verified = False
            if not verified:
                _identity_unproven(self)
                return
            parent.sendall((json.dumps(dict(argv=argv, executable=executable, env=target_env,
                restore_signals=kwargs.get('restore_signals', True))) + '\n').encode())
            # EOF is exec success. On failure the launcher reports errno before
            # exiting; the verified owner still retires it before raising.
            failure = parent.recv(32)
            if failure:
                kill_group(self)
                self.wait(timeout=2)
                for stream in (self.stdin, self.stdout, self.stderr):
                    if stream is not None:
                        stream.close()
                code = int(failure)
                raise OSError(code, os.strerror(code), executable)
        finally:
            parent.close()
            child.close()

    def poll(self):
        return poll_exit(self)

    def _internal_poll(self, *args, **kwargs):
        # Popen.__del__ and subprocess's global cleanup must not reap before
        # the owner seals group signalling. Explicit wait is the only reaper.
        return self.returncode

    def _try_wait(self, flags):
        # Popen's default ECHILD handling fabricates returncode=0. Never let
        # that become resource/lease proof after an outside reap.
        with _group_lock:
            try:
                pid, status = os.waitpid(self.pid, flags)
            except ChildProcessError:
                _outside_reap(self)
                raise ChildProcessError('outside reap; process group cleanup is unproven') from None
            if pid == self.pid:
                self._handle_exitstatus(status)
                _close_pidfd(self)
            return pid, status

    def wait(self, timeout=None):
        if getattr(self, '_voice_identity_unproven', False):
            raise ChildProcessError('acquisition identity unproven; process group cleanup is unproven')
        if getattr(self, '_voice_outside_reap', False):
            raise ChildProcessError('outside reap; process group cleanup is unproven')
        # No owner wait is unbounded, even for a caller without its own cap.
        timeout = 300 if timeout is None else timeout
        deadline = time.monotonic() + timeout
        if not getattr(self, '_voice_group_signalled', False):
            wait_exit(self, timeout)
            kill_group(self)
            if not getattr(self, '_voice_group_signalled', False):
                raise ChildProcessError('outside reap; process group cleanup is unproven')
        return super().wait(timeout=max(0, deadline - time.monotonic()))

    def kill(self):
        kill_group(self)


class SynthesisOwner(Resource):
    """Keep a synthesizer's child through IO and every failed close attempt."""

    resource_kinds = ("piper-status",)
    lifecycle_error = OSError

    def __init__(self):
        self._process = None
        self._process_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._cancelled = threading.Event()

    def cancel(self):
        with self._process_lock:
            self._cancelled.set()
            process = self._process
        if process is not None:
            process.kill()

    def _finish_process(self, process):
        process.kill()
        process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        with self._process_lock:
            if self._process is process:
                self._process = None

    def _on_close_requested(self):
        self._cancelled.set()
        process = self._process
        if process is not None and (self.lifecycle.starting or self.lifecycle.users):
            kill_group(process)

    def _dispose_complete(self):
        return self._process is None

    @closing
    def close(self):
        self.cancel()
        # A close blocked on IO remains registered. Daemon escalation uses
        # _process directly and never needs either of these locks.
        with self._io_lock:
            process = self._process
            if process is not None:
                self._finish_process(process)

    def _running_process(self, *args, **kwargs):
        return _running_process(self, *args, **kwargs)


@contextlib.contextmanager
def _running_process(owner, *args, **kwargs):
    life = owner.lifecycle
    try:
        life.enter(owner)
    except OSError:
        yield None
        return
    locked = False
    process = None
    try:
        locked = owner._io_lock.acquire(timeout=2)
        if not locked:
            raise TimeoutError("synthesis IO is still in use")
        with owner._process_lock:
            if not owner._cancelled.is_set():
                if owner._process is not None:
                    raise OSError('previous synthesis cleanup is unproven')
                try:
                    process = owner._publish("_process", lambda: GroupPopen(*args, **kwargs))
                except OSError as error:
                    if not getattr(error, 'resource_closed', False):
                        raise
                    # A late published child stays owned until the final
                    # close below. The caller still sees cancellation.
        try:
            yield process
        finally:
            if process is not None:
                owner._finish_process(process)
    finally:
        if locked:
            owner._io_lock.release()
        life.leave()
        if life.requested:
            owner.close()


def run_owned(command, *, timeout, owner=None, **kwargs):
    """A bounded provider inspection with the same child ownership as synth."""
    owner = owner if owner is not None else SynthesisOwner()
    kwargs.pop('capture_output', None)
    kwargs.pop('check', None)
    with owner._running_process(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, **kwargs) as process:
        if process is None:
            raise OSError('provider inspection was cancelled')
        out, err = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, out, err)
