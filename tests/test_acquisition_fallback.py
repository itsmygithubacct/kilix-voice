"""Identity refusal and signal routing; no PID reuse, tracing or namespaces."""
import contextlib
import errno
import os
from pathlib import Path
import signal
import subprocess
import unittest
from unittest import mock
from voicelib import processes
from tests.test_process_identity import finish_group


class Acquisition(unittest.TestCase):
    def contain(self, child):
        # Every PID here belongs to a child this test just started and has
        # never reaped. No identity reuse or external reaper runs in a fixture.
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(child.pid, 0)
        fd = getattr(child, '_voice_pidfd', None)
        if fd is not None:
            os.close(fd)
            child._voice_pidfd = None
        child.returncode = -signal.SIGKILL

    def test_wrong_live_pidfd_is_refused_at_acquisition(self):
        other = subprocess.Popen(['/bin/sleep', '30'])
        original = os.pidfd_open
        try:
            with mock.patch.object(os, 'pidfd_open', side_effect=lambda *_: original(other.pid)):
                child = processes.GroupPopen(['/bin/sleep', '30'])
            try:
                with mock.patch.object(signal, 'pidfd_send_signal') as send, mock.patch.object(os, 'killpg') as numeric:
                    processes.kill_group(child)
                    self.assertTrue(getattr(child, '_voice_identity_unproven', False), 'wrong pidfd accepted at acquisition')
                    send.assert_not_called()
                    numeric.assert_not_called()
                    self.assertFalse(getattr(child, '_voice_group_signalled', False))
            finally:
                self.contain(child)
        finally:
            other.kill(); other.wait(timeout=3)

    def test_pidfd_fallback_never_signals_by_number(self):
        with mock.patch.object(processes, '_pidfd_group_supported', False):
            child = processes.GroupPopen(['/bin/sleep', '30'])
            try:
                with mock.patch.object(os, 'killpg') as numeric:
                    finish_group(self, child)
                    numeric.assert_not_called()
            finally:
                self.contain(child)

    def test_wrong_parent_is_refused_at_acquisition(self):
        read = Path.read_text
        def status(path, *args, **kwargs):
            text = read(path, *args, **kwargs)
            if path.name == 'status':
                text = '\n'.join('PPid:\t0' if line.startswith('PPid:') else line for line in text.splitlines())
            return text
        with mock.patch.object(Path, 'read_text', status):
            child = processes.GroupPopen(['/bin/sleep', '30'])
        try:
            with mock.patch.object(signal, 'pidfd_send_signal') as send, mock.patch.object(os, 'killpg') as numeric:
                processes.kill_group(child)
                self.assertTrue(child._voice_identity_unproven, 'wrong PPid was accepted')
                send.assert_not_called()
                numeric.assert_not_called()
        finally:
            self.contain(child)

    def test_missing_nonce_is_refused(self):
        with mock.patch.object(Path, 'read_bytes', return_value=b'UNRELATED=1\0'):
            child = processes.GroupPopen(['/bin/sleep', '30'])
        try:
            self.assertTrue(child._voice_identity_unproven)
            with mock.patch.object(os, 'killpg') as numeric:
                processes.kill_group(child)
                numeric.assert_not_called()
        finally:
            self.contain(child)

    def test_transient_pidfd_failure_does_not_select_numeric_fallback(self):
        with mock.patch.object(os, 'pidfd_open', side_effect=OSError(errno.EMFILE, 'fixture')):
            child = processes.GroupPopen(['/bin/sleep', '30'])
        try:
            with mock.patch.object(os, 'killpg') as numeric:
                processes.kill_group(child)
                self.assertTrue(child._voice_identity_unproven)
                numeric.assert_not_called()
        finally:
            self.contain(child)

    def test_fallback_rechecks_leader_after_scan_before_any_signal(self):
        with mock.patch.object(processes, '_pidfd_group_supported', False):
            child = processes.GroupPopen(['/bin/sleep', '30'])
            member = os.dup(child._voice_pidfd)
            scanned = False
            check = processes._check_leader
            def checked(p):
                if scanned:
                    raise ChildProcessError('fixture outside reap after scan')
                return check(p)
            def scan(group):
                nonlocal scanned
                scanned = True
                return [member]
            try:
                with mock.patch.object(processes, '_check_leader', side_effect=checked), \
                        mock.patch.object(processes, '_scan_group_members', side_effect=scan), \
                        mock.patch.object(signal, 'pidfd_send_signal') as send, \
                        mock.patch.object(os, 'killpg') as numeric:
                    processes.kill_group(child)
                    send.assert_not_called()
                    numeric.assert_not_called()
                    self.assertTrue(child._voice_outside_reap)
                    self.assertFalse(getattr(child, '_voice_group_signalled', False))
                with self.assertRaises(OSError):
                    os.fstat(member)
            finally:
                with contextlib.suppress(OSError):
                    os.close(member)
                self.contain(child)

    def test_fallback_revalidates_membership_after_open(self):
        child = processes.GroupPopen(['/bin/sleep', '30'])
        member = os.dup(child._voice_pidfd)
        try:
            with mock.patch.object(Path, 'iterdir', return_value=iter([Path('/proc/123')])) , \
                    mock.patch.object(processes, '_member_stat', side_effect=[('S', child.pid, '1'), ('S', child.pid+1, '2')]), \
                    mock.patch.object(os, 'pidfd_open', return_value=member):
                self.assertEqual(processes._scan_group_members(child.pid), [])
            with self.assertRaises(OSError):
                os.fstat(member)
        finally:
            with contextlib.suppress(OSError):
                os.close(member)
            self.contain(child)

    def test_gate_preserves_fast_exit_exec_errors_and_environment(self):
        for _ in range(8):
            child = processes.GroupPopen(['/bin/true'])
            self.assertEqual(child.wait(timeout=3), 0)
        with self.assertRaises(FileNotFoundError):
            processes.GroupPopen(['/nonexistent-fixture-command'])
        child = processes.GroupPopen(['/usr/bin/env'], stdout=subprocess.PIPE, env={'FIXTURE': 'yes'})
        output, _ = child.communicate(timeout=3)
        self.assertEqual(output.strip(), b'FIXTURE=yes')

    def test_post_signal_outside_wait_never_proves_cleanup_or_lease(self):
        from tests.test_cleanup_recovery import Boundaries
        from voicelib import audio
        daemon, _, _ = Boundaries.daemon(self)
        capture = audio.MicCapture({'audio': {'capture_cmd': ['/bin/sleep', '30']}})
        # No reader observes the child between our external wait and wait().
        # Otherwise poll_exit could detect ECHILD first and mask _try_wait.
        child = processes.GroupPopen(['/bin/sleep', '30'])
        capture._process = child
        grant = mock.Mock()
        key, state = daemon._register_cleanup('final-wait', 'capture', capture, grant, 'stop')
        try:
            processes.kill_group(child)
            os.waitpid(child.pid, 0)
            self.assertFalse(getattr(child, '_voice_outside_reap', False))
            with self.assertRaisesRegex(ChildProcessError, 'outside reap'):
                child.wait(timeout=.5)
            self.assertIsNone(child.returncode)
            daemon._retry_dictation_cleanup(.2)
            self.assertIn(key, daemon._cleanup_owners)
            self.assertIs(capture._process, child)
            self.assertFalse(capture.cleanup_complete)
            grant.reaped.assert_not_called()
        finally:
            fd = getattr(child, '_voice_pidfd', None)
            if fd is not None:
                os.close(fd); child._voice_pidfd = None


if __name__ == '__main__':
    unittest.main()
