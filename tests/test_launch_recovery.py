"""Protocol failures and retry scheduling preserve failed-launch ownership."""
import contextlib
import errno
import os
import signal
import socket
import subprocess
import threading
import time
import unittest
from unittest import mock

from voicelib import processes
from tests.test_terminal_outcomes import voiced
from tests import test_cleanup_recovery as recovery


class Recovery(unittest.TestCase):

    def test_launch_calls_failed_launch_retry(self):
        with mock.patch.object(processes, 'retry_failed_launches') as retry, mock.patch.object(processes, 'require_waitable_children', side_effect=ValueError('no spawn fixture')):
            with self.assertRaises(ValueError):
                processes.GroupPopen(['/bin/true'])
            retry.assert_called_once_with()

    def test_idle_hook_restores_one_worker_after_construction_failure(self):
        owner = processes._FailedLaunches()
        child = mock.Mock(pid=123)
        with mock.patch.object(processes, '_failed_launches', owner), mock.patch.object(processes, '_dispose_failed_launch') as dispose:
            with mock.patch.object(threading, 'Thread', side_effect=RuntimeError('fixture cannot construct thread')):
                owner.retain(child)
            self.assertIn(child, owner.pending)
            self.assertIsNone(owner.worker)
            d = recovery.Boundaries.daemon(self)[0]
            d._idle_seconds = 0
            d._owned_expire = mock.Mock()
            d._check_idle()
            worker = owner.worker
            if worker:
                worker.join(3)
            self.assertFalse(owner.pending)
            dispose.assert_called_once_with(child)

    def fixture(self, stack):
        owner = processes._FailedLaunches()
        observed = {}
        verify = processes._verify_spawn_identity

        def identity(child, nonce):
            observed.update(child=child, fd=child._voice_pidfd)
            return verify(child, nonce)

        def scan(group):
            child = observed['child']
            self.assertEqual(group, child.pid)
            # Only this fixture's trusted, nonforking invalid-exec launcher.
            return processes._MemberScan([os.dup(child._voice_pidfd)])

        stack.enter_context(mock.patch.object(processes, '_failed_launches', owner))
        stack.enter_context(mock.patch.object(processes, '_verify_spawn_identity', side_effect=identity))
        stack.enter_context(mock.patch.object(processes, '_pidfd_group_supported', False))
        stack.enter_context(mock.patch.object(processes, '_scan_group_members', side_effect=scan))
        return owner, observed

    def contain(self, owner, observed):
        # Never let a failing assertion leave this fixture's child or worker.
        worker = owner.worker
        if worker is not None:
            worker.join(5)
        child = observed.get('child')
        if child is not None:
            if child.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child.pid, signal.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(child.pid, 0)
                processes._close_pidfd(child)
                child.returncode = 127
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
            with owner.condition:
                owner.pending.discard(child)
                owner.condition.notify_all()
            if child in subprocess._active:
                subprocess._active.remove(child)
        if worker is not None:
            worker.join(5)

    def assert_disposed(self, owner, observed):
        child = observed['child']
        deadline = time.monotonic() + 5
        with owner.condition:
            worker = owner.worker
        if worker is not None:
            worker.join(max(0, deadline-time.monotonic()))
        self.assertIsNotNone(child.returncode, 'failed launcher was never reaped')
        self.assertIsNone(child._voice_pidfd)
        self.assertTrue(all(s.closed for s in (child.stdin, child.stdout, child.stderr)))
        self.assertNotIn(child, owner.pending)
        self.assertNotIn(child, subprocess._active)
        with self.assertRaises(OSError):
            os.fstat(observed['fd'])
        with self.assertRaises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)

    def test_protocol_exceptions_after_admission_keep_original_and_retry(self):
        for point in ('send', 'receive', 'encode', 'close', 'receive-and-close'):
            with self.subTest(point=point), contextlib.ExitStack() as stack:
                owner, observed = self.fixture(stack)
                release = threading.Event()
                dispose = processes._dispose_failed_launch
                pair = socket.socketpair
                encode = processes.marshal.dumps
                error = ConnectionResetError(errno.ECONNRESET, 'fixture protocol failure')

                class Parent:
                    def __init__(self, endpoint):
                        self.endpoint = endpoint
                    def recv(self, size):
                        if point in ('receive', 'receive-and-close') and size == 65536:
                            raise error
                        return self.endpoint.recv(size)
                    def sendall(self, message):
                        if point == 'send':
                            raise error
                        return self.endpoint.sendall(message)
                    def close(self):
                        self.endpoint.close()
                        if point == 'close':
                            raise error
                        if point == 'receive-and-close':
                            raise OSError('secondary channel close failure')

                def sockets():
                    parent, child = pair()
                    return Parent(parent), child

                def encoding(request):
                    if point == 'encode':
                        raise error
                    return encode(request)

                def disposing(child):
                    if not release.is_set():
                        raise OSError('fixture disposal temporarily inconclusive')
                    dispose(child)

                stack.enter_context(mock.patch.object(processes.socket, 'socketpair', side_effect=sockets))
                stack.enter_context(mock.patch.object(processes.marshal, 'dumps', side_effect=encoding))
                stack.enter_context(mock.patch.object(processes, '_dispose_failed_launch', side_effect=disposing))
                try:
                    caught = None
                    try:
                        command = '/bin/true' if point == 'close' else '/nonexistent-launch-recovery-fixture'
                        processes.GroupPopen([command],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as failure:
                        caught = failure
                    self.assertIs(caught, error)
                    self.assertIn(observed['child'], owner.pending,
                                  'post-admission failure escaped the managed owner')
                    release.set()
                    self.assert_disposed(owner, observed)
                finally:
                    release.set()
                    self.contain(owner, observed)

    def test_thread_start_failure_preserves_exec_error_and_idle_tick_retries(self):
        with contextlib.ExitStack() as stack:
            owner, observed = self.fixture(stack)
            try:
                with mock.patch.object(processes, '_dispose_failed_launch', side_effect=OSError('inconclusive')), \
                        mock.patch.object(threading.Thread, 'start', side_effect=RuntimeError("can't start new thread")), \
                        self.assertLogs('voicelib.processes', 'WARNING') as logs:
                    caught = None
                    try:
                        processes.GroupPopen(['/nonexistent-launch-recovery-fixture'],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as failure:
                        caught = failure
                    self.assertIsInstance(caught, FileNotFoundError,
                                          'scheduler failure replaced the exec error')
                    self.assertEqual(caught.errno, errno.ENOENT)
                    self.assertIn(observed['child'], owner.pending)
                    self.assertIsNone(owner.worker)
                self.assertTrue(any('could not start' in entry for entry in logs.output), logs.output)
                self.assertTrue(any('retained' in entry for entry in logs.output), logs.output)
                daemon = object.__new__(voiced.Daemon)
                daemon._owned_expire = mock.Mock()
                daemon._idle_seconds = 0
                daemon._check_idle()  # ordinary periodic work; no later retain needed
                self.assert_disposed(owner, observed)
            finally:
                self.contain(owner, observed)

    def test_protocol_failure_can_dispose_immediately_without_retaining_handles(self):
        with contextlib.ExitStack() as stack:
            owner, observed = self.fixture(stack)
            pair = socket.socketpair
            error = ConnectionResetError(errno.ECONNRESET, 'fixture reply reset')

            class Parent:
                def __init__(self, endpoint):
                    self.endpoint = endpoint
                def recv(self, size):
                    if size == 65536:
                        raise error
                    return self.endpoint.recv(size)
                def sendall(self, message):
                    return self.endpoint.sendall(message)
                def close(self):
                    self.endpoint.close()

            def sockets():
                parent, child = pair()
                return Parent(parent), child

            stack.enter_context(mock.patch.object(processes.socket, 'socketpair', side_effect=sockets))
            try:
                with self.assertRaises(ConnectionResetError) as caught:
                    processes.GroupPopen(['/nonexistent-launch-recovery-fixture'],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertIs(caught.exception, error)
                self.assertIsNone(owner.worker)
                self.assert_disposed(owner, observed)
            finally:
                self.contain(owner, observed)

    def test_concurrent_handoffs_use_one_daemon_worker_and_release_disposed_handles(self):
        owner = processes._FailedLaunches()
        release, entered = threading.Event(), threading.Event()
        children = [mock.Mock(pid=i) for i in range(32)]
        calls = []

        def dispose(child):
            calls.append(child)
            entered.set()
            release.wait(5)

        with mock.patch.object(processes, '_dispose_failed_launch', side_effect=dispose):
            threads = [threading.Thread(target=owner.retain, args=(child,)) for child in children]
            worker = None
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(5)
                    self.assertFalse(thread.is_alive())
                self.assertTrue(entered.wait(5))
                worker = owner.worker
                self.assertTrue(worker.daemon)
                self.assertEqual(owner.pending, set(children))
                self.assertEqual([t for t in threading.enumerate() if t.name == 'failed-launch-cleanup'], [worker])
                release.set()
                worker.join(5)
                self.assertEqual(len(calls), len(children))
                self.assertFalse(owner.pending)
                self.assertIsNone(owner.worker)
            finally:
                release.set()
                if worker is not None:
                    # A planted no-removal defect must not strand a worker.
                    worker.join(5)
                    if worker.is_alive():
                        with owner.condition:
                            owner.pending.clear()
                            owner.condition.notify_all()
                        worker.join(5)


if __name__ == '__main__':
    unittest.main()
