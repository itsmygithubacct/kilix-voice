"""Failed constructors keep ownership until their launcher can be disposed."""
import contextlib
import errno
import os
import signal
import subprocess
import time
import unittest
from unittest import mock

from voicelib import processes


class FailedLaunch(unittest.TestCase):
    def check_cleanup(self, failure):
        observed = {}
        verify = processes._verify_spawn_identity
        retry_allowed = False

        def identity(child, nonce):
            result = verify(child, nonce)
            observed.update(child=child, fd=child._voice_pidfd)
            return result

        def scan(group):
            child = observed['child']
            self.assertEqual(group, child.pid)
            if not retry_allowed:
                if failure == 'scan':
                    return []  # no proof, even though this launcher exited
                raise OSError(errno.EIO, 'fixture unreadable procfs')
            # This fixture owns one trusted, nonforking failed launcher.
            return processes._MemberScan([os.dup(child._voice_pidfd)])

        caught = None
        try:
            with mock.patch.object(processes, '_pidfd_group_supported', False), \
                    mock.patch.object(processes, '_verify_spawn_identity', side_effect=identity), \
                    mock.patch.object(processes, '_scan_group_members', side_effect=scan):
                try:
                    processes.GroupPopen(['/nonexistent-failed-launch-fixture'],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                except Exception as error:
                    caught = (type(error), getattr(error, 'errno', None))
                child, fd = observed['child'], observed['fd']
                self.assertFalse(getattr(child, '_voice_group_signalled', False))
                self.assertIsNone(child.returncode)
                self.assertIsNotNone(child._voice_pidfd)
                self.assertIn(child, getattr(getattr(processes, '_failed_launches', None), 'pending', ()),
                              'failed constructor has no explicit cleanup owner')
                retry_allowed = True
                deadline = time.monotonic() + 5
                while (child._voice_pidfd is not None or not all(stream.closed for stream in
                       (child.stdin, child.stdout, child.stderr))) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(caught, (FileNotFoundError, errno.ENOENT),
                                 'cleanup replaced the original exec failure')
                self.assertIsNotNone(child.returncode, 'no owner retried launcher cleanup')
                self.assertIsNone(child._voice_pidfd)
                with self.assertRaises(OSError):
                    os.fstat(fd)
                with self.assertRaises(ChildProcessError):
                    os.waitpid(child.pid, os.WNOHANG)
                self.assertTrue(all(stream.closed for stream in
                                    (child.stdin, child.stdout, child.stderr)))
                self.assertNotIn(child, subprocess._active)
        finally:
            # Contain only this fixture's unreaped child if an assertion fails.
            child = observed.get('child')
            if child is not None and child.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child.pid, signal.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(child.pid, 0)
                processes._close_pidfd(child)
                child.returncode = 127
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()

    def test_inconclusive_scan_preserves_error_and_retries_cleanup(self):
        self.check_cleanup('scan')

    def test_scan_error_preserves_error_and_retries_cleanup(self):
        self.check_cleanup('error')
