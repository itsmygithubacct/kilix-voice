"""Process identity and cleanup proof, without namespaces or PID reuse."""
import errno
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest import mock

from voicelib import processes


class IdentityTests(unittest.TestCase):
    def test_outside_reap_never_authorizes_numeric_signal(self):
        for fallback, disposition in ((f, d) for f in (False, True) for d in ('waitpid', 'ignore')):
            with self.subTest(fallback=fallback, disposition=disposition), \
                    mock.patch.object(processes, '_pidfd_group_supported', False if fallback else None):
                # Keep the child alive until the disposition is installed.
                # SIG_IGN does not reap a zombie that predates the change.
                child = processes.GroupPopen(['/bin/cat'], stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL)
                previous = signal.getsignal(signal.SIGCHLD)
                try:
                    if disposition == 'ignore':
                        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                        child.stdin.close()
                        deadline = time.monotonic() + 15
                        while Path(f'/proc/{child.pid}').exists():
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(.005)
                    else:
                        child.stdin.close()
                        os.waitpid(child.pid, 0)
                    with mock.patch.object(processes.os, 'killpg') as numeric:
                        error = None
                        try:
                            processes.kill_group(child)
                        except OSError as caught:
                            error = caught
                        self.assertIsNone(error, 'outside reap was not classified before signalling')
                        numeric.assert_not_called()
                    self.assertTrue(getattr(child, '_voice_outside_reap', False),
                                    'outside reap was not recorded')
                    self.assertFalse(getattr(child, '_voice_group_signalled', False),
                                     'missing process was counted as group cleanup proof')
                    with self.assertRaisesRegex(ChildProcessError, 'outside reap'):
                        child.wait(timeout=.1)
                finally:
                    signal.signal(signal.SIGCHLD, previous)
                    child.stdin.close()
                    # Only this already exited child; no signal by stale PID.
                    try:
                        subprocess.Popen.wait(child, timeout=2)
                    except ChildProcessError:
                        pass  # already reaped by this test
                    fd = getattr(child, '_voice_pidfd', None)
                    if fd is not None:
                        os.close(fd)
                        child._voice_pidfd = None

    def test_group_signal_uses_pidfd_and_closes_it_after_reap(self):
        child = processes.GroupPopen(['/bin/sleep', '30'])
        try:
            fd = getattr(child, '_voice_pidfd', None)
            self.assertIsInstance(fd, int, 'group leader has no stable identity')
            self.assertFalse(os.get_inheritable(fd), 'pidfd can leak across exec')
            code = ('import os,sys\n'
                    'try: os.fstat(int(sys.argv[1]))\n'
                    'except OSError: sys.exit(0)\n'
                    'sys.exit(1)\n')
            inherited = subprocess.run([sys.executable, '-c', code, str(fd)], close_fds=False)
            self.assertEqual(inherited.returncode, 0, 'exec inherited the pidfd')
            with mock.patch.object(signal, 'pidfd_send_signal', wraps=signal.pidfd_send_signal) as send, \
                    mock.patch.object(os, 'killpg', wraps=os.killpg) as numeric:
                processes.kill_group(child)
                send.assert_called_once_with(fd, signal.SIGKILL, None, 1 << 2)
                numeric.assert_not_called()
                try:
                    os.fstat(fd)  # retained while final reap has not happened
                except OSError as error:
                    self.fail(f'pidfd closed before final reap: {error}')
                child.wait(timeout=2)
                processes.kill_group(child)
                self.assertEqual(send.call_count, 1)
            with self.assertRaises(OSError) as closed:
                os.fstat(fd)
            self.assertEqual(closed.exception.errno, errno.EBADF)
        finally:
            processes.kill_group(child)
            child.wait(timeout=2)

    def test_pidfd_signal_reaches_background_group_members(self):
        child = processes.GroupPopen(['/bin/sh', '-c', 'sleep 30 & echo $!; wait'], stdout=subprocess.PIPE)
        background_fd = None
        try:
            background = int(child.stdout.readline())
            background_fd = os.pidfd_open(background)
            processes.kill_group(child)
            child.wait(timeout=2)
            deadline = time.monotonic() + 2
            while True:
                try:
                    state = Path(f'/proc/{background}/stat').read_text().rsplit(') ', 1)[1].split()[0]
                except FileNotFoundError:
                    break
                if state == 'Z':
                    break
                self.assertLess(time.monotonic(), deadline, 'background group member survived')
                time.sleep(.005)
        finally:
            # Keep a stable fixture-only handle even if a mutant reaps the
            # leader before terminating its background member.
            if background_fd is not None:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(background_fd, signal.SIGKILL)
                os.close(background_fd)
            processes.kill_group(child)
            child.wait(timeout=2)
            child.stdout.close()

    def test_fallback_checks_pidfd_immediately_before_numeric_signal(self):
        # Historical test name retained for mutation-catalogue references.
        # With a pidfd this backend must now use member descriptors only.
        with mock.patch.object(processes, '_pidfd_group_supported', False):
            child = processes.GroupPopen(['/bin/sleep', '30'])
            try:
                order = []
                check, scan, send = processes._check_leader, processes._scan_group_members, signal.pidfd_send_signal
                def checked(p):
                    order.append('check')
                    return check(p)
                def scanned(p):
                    order.append('scan')
                    return scan(p)
                def signalled(*args):
                    order.append('signal')
                    return send(*args)
                with mock.patch.object(processes, '_check_leader', side_effect=checked), \
                        mock.patch.object(processes, '_scan_group_members', side_effect=scanned), \
                        mock.patch.object(signal, 'pidfd_send_signal', side_effect=signalled), \
                        mock.patch.object(os, 'killpg') as numeric:
                    processes.kill_group(child)
                    numeric.assert_not_called()
                self.assertIn('signal', order)
                for index, event in enumerate(order):
                    if event == 'scan':
                        self.assertEqual(order[index-1:index+2], ['check', 'scan', 'check'])
            finally:
                processes.kill_group(child)
                child.wait(timeout=2)

    def test_fallback_refuses_sigchld_ignore_while_child_is_alive(self):
        with mock.patch.object(processes, '_pidfd_group_supported', False):
            child = processes.GroupPopen(['/bin/sleep', '30'])
            try:
                with mock.patch.object(signal, 'getsignal', return_value=signal.SIG_IGN), \
                        mock.patch.object(os, 'killpg') as numeric:
                    with self.assertRaisesRegex(OSError, 'SIGCHLD'):
                        processes.kill_group(child)
                    numeric.assert_not_called()
                    self.assertFalse(getattr(child, '_voice_group_signalled', False))
            finally:
                # Contain a mutant that seals a mocked (undelivered) signal.
                signal.pidfd_send_signal(child._voice_pidfd, signal.SIGKILL)
                processes.kill_group(child)
                child.wait(timeout=2)

    def test_unsupported_group_flag_is_detected_once(self):
        original = signal.pidfd_send_signal
        def old_kernel(fd, sig, info=None, flags=0):
            if flags == processes.PIDFD_SIGNAL_PROCESS_GROUP:
                raise OSError(errno.EINVAL, 'old kernel')
            return original(fd, sig, info, flags)
        with mock.patch.object(processes, '_pidfd_group_supported', None), \
                mock.patch.object(signal, 'pidfd_send_signal', side_effect=old_kernel) as send, \
                mock.patch.object(os, 'killpg', wraps=os.killpg) as numeric:
            for _ in range(2):
                child = processes.GroupPopen(['/bin/sleep', '30'])
                processes.kill_group(child)
                child.wait(timeout=2)
            self.assertEqual(sum(call.args[3] == 1 << 2 for call in send.call_args_list), 1)
            self.assertGreaterEqual(sum(call.args[3] == 0 for call in send.call_args_list), 2)
            numeric.assert_not_called()

    def test_no_pidfd_uses_waitable_child_fallback(self):
        with mock.patch.object(processes, '_pidfd_available', False):
            child = processes.GroupPopen(['/bin/sleep', '30'])
            try:
                with mock.patch.object(os, 'waitid', wraps=os.waitid) as check:
                    processes.kill_group(child)
                check.assert_called_once_with(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            finally:
                child.wait(timeout=2)

    def test_outside_reap_retains_resource_and_grant_and_shutdown_logs_it(self):
        from tests.test_cleanup_recovery import Boundaries
        from voicelib import audio
        daemon, _, _ = Boundaries.daemon(self)
        daemon._speech = daemon._dictation = daemon._player = daemon._socket = None
        daemon._socket_ino = None
        daemon._wake_w = -1
        capture = audio.MicCapture({'audio': {'capture_cmd': ['/bin/true']}})
        capture.start()
        child, reader = capture._process, capture._reader
        os.waitpid(child.pid, 0)
        grant = mock.Mock()
        key, state = daemon._register_cleanup('outside', 'capture', capture, grant, 'stop')
        try:
            daemon._shutdown()
            self.assertIn(key, daemon._cleanup_owners)
            self.assertIs(capture._process, child)
            self.assertFalse(capture.cleanup_complete)
            grant.reaped.assert_not_called()
            self.assertIn('outside reap', str(daemon._warn.call_args_list))
        finally:
            reader.join(3)
            child.stdout.close()
            if getattr(capture, '_stderr', None) is not None:
                capture._stderr.close()

    def test_readme_explains_escaped_descendants(self):
        readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
        section = readme.split('## Recovering incomplete voice cleanup', 1)[1].split('### Accelerator quarantine', 1)[0]
        for term in ('setsid()', 'setpgid()', 'abort', 'shutdown'):
            self.assertIn(term, section)


if __name__ == '__main__':
    unittest.main()
