"""Actual capture ownership, IPC and lifecycle; all audio is synthetic."""
import array
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from voicelib import audio, microphone
from tests.test_tui_shell import load_script

PEER = str(Path(__file__).with_name('microphone_peer.py'))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.namespace = str(self.root / 'leases')
        self.owners = []
        self.fd_before = len(os.listdir('/proc/self/fd'))

    def tearDown(self):
        for owner, stderr in self.owners:
            owner.stop()
            if owner.process is not None and owner.process.stdout is not None:
                owner.process.stdout.close()
            stderr.close()
        self.temp.cleanup()
        self.assertEqual(len(os.listdir('/proc/self/fd')), self.fd_before)

    def control(self, **kwargs):
        return microphone.CaptureControl(**{'deadline': time.monotonic() + 8,
            'locked': lambda: False, 'namespace': self.namespace, **kwargs})

    def start(self, **controls):
        marker = self.root / ('recorder-' + str(len(self.owners)) + '.json')
        owner = microphone.CaptureOwner(self.control(**controls))
        stderr = tempfile.TemporaryFile()
        self.owners.append((owner, stderr))
        owner.start([sys.executable, '-I', PEER, 'recorder', str(marker), 'escape'], stderr)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                return owner, json.loads(marker.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.01)
        self.fail('Recorder did not start')

    def test_actual_pcm_and_normal_reap(self):
        phases = []
        owner, tree = self.start(indicator=phases.append)
        self.assertEqual(os.read(owner.process.stdout.fileno(), 640), b'\x21\x00' * 320)
        self.assertTrue(any(link.endswith('/accelerator.lock') for _, link in tree['fds']))
        self.assertFalse(any(link.startswith('socket:') for _, link in tree['fds']))
        self.assertEqual(tree['limits']['CORE'], [0, 0])
        self.assertEqual(tree['limits']['FSIZE'], [1024 * 1024, 1024 * 1024])
        self.assertLessEqual(tree['limits']['NOFILE'][1], 128)
        self.assertEqual(tree['loader_overrides'], [])
        self.assertTrue(owner.stop())
        self.assertEqual(phases, ['recording', 'stopping', 'inactive'])
        self.assertFalse(Path(f"/proc/{tree['pid']}").exists())
        self.assertFalse(Path(f"/proc/{tree['escaped']}").exists())

    def test_busy_immediate_and_stale_stop_preserves_successor(self):
        owner, _ = self.start()
        queued = microphone.CaptureOwner(self.control())
        with tempfile.TemporaryFile() as stderr:
            began = time.monotonic()
            with self.assertRaises(microphone.CaptureError) as caught:
                queued.start(['/bin/true'], stderr)
            self.assertEqual(caught.exception.code, 'busy')
            self.assertLess(time.monotonic() - began, 0.3)
        self.assertTrue(owner.stop())
        second, _ = self.start()
        self.assertTrue(owner.stop())
        self.assertIsNone(second.process.poll())
        with tempfile.TemporaryFile() as stderr:
            with self.assertRaises(microphone.CaptureError):
                microphone.CaptureOwner(self.control()).start(['/bin/true'], stderr)

    def test_cancellation_disconnect_and_lock_reap_escaped_children(self):
        for field in ('cancelled', 'disconnected', 'locked'):
            with self.subTest(field=field):
                trigger = threading.Event()
                owner, tree = self.start(**{field: trigger.is_set})
                trigger.set()
                self.assertTrue(owner.wait(3))
                self.assertTrue(owner.cleanup_complete)
                self.assertEqual(owner.error.code, field)
                self.assertFalse(Path(f"/proc/{tree['pid']}").exists())
                self.assertFalse(Path(f"/proc/{tree['escaped']}").exists())

    def test_deadline_reaps_without_stop_call(self):
        owner, tree = self.start(deadline=time.monotonic() + 0.6)
        self.assertTrue(owner.wait(3))
        self.assertTrue(owner.cleanup_complete)
        self.assertFalse(Path(f"/proc/{tree['pid']}").exists())
        self.assertFalse(Path(f"/proc/{tree['escaped']}").exists())

    def test_unresponsive_embedding_thread_loses_heartbeat(self):
        delay = threading.Event()
        calls = [0]
        def locked():
            calls[0] += 1
            if calls[0] >= 4:
                delay.wait(2)
            return False
        owner, _ = self.start(locked=locked)
        self.assertIsNotNone(owner.process.wait(timeout=2))
        delay.set()
        self.assertTrue(owner.wait(3))
        self.assertTrue(owner.cleanup_complete)

    def test_indicator_failure_refuses_before_recorder(self):
        def indicator(_phase):
            raise RuntimeError('fixture')
        capture = microphone.CaptureOwner(self.control(indicator=indicator))
        with tempfile.TemporaryFile() as stderr:
            with self.assertRaises(microphone.CaptureError):
                capture.start(['/bin/true'], stderr)
        self.assertIsNone(capture.process)
        self.assertEqual(capture.phase, 'unavailable')
        owner, _ = self.start()
        self.assertTrue(owner.stop())

    def test_unknown_locked_and_bad_controls_do_not_allocate(self):
        for value in (None, True, 'no'):
            owner = microphone.CaptureOwner(self.control(locked=lambda v=value: v))
            with tempfile.TemporaryFile() as stderr, self.assertRaises(microphone.CaptureError):
                owner.start(['/bin/true'], stderr)
            self.assertIsNone(owner.process)
        self.assertFalse(Path(self.namespace).exists())
        for value in (True, float('nan'), float('inf'), time.monotonic() + 121,
                      10**10000, -(10**10000)):
            with self.assertRaises(microphone.CaptureError):
                self.control(deadline=value)

    def test_real_owner_death_and_supervisor_loss_remain_quarantined(self):
        for mode in ('loss', 'loss-audio', 'parent-death'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                result = subprocess.run([sys.executable, '-I', PEER, mode, root],
                    capture_output=True, text=True, timeout=15, check=True)
                report = json.loads(result.stdout)
                self.assertTrue(report['persistent_unavailable'])
                self.assertTrue(report['owned_absent'])
                self.assertTrue(report['unrelated_preserved'])

    def test_mic_capture_frames_and_idempotent_proven_stop(self):
        capture = audio.MicCapture({'audio': {'capture_cmd': [sys.executable, '-I', '-c',
            'import os;os.write(1,b"x"*1281)']}}, control=self.control())
        try:
            capture.start()
            self.assertEqual(capture.read(timeout=2), b'x' * 640)
            self.assertEqual(capture.read(timeout=2), b'x' * 640)
            self.assertTrue(capture.stop())
            self.assertTrue(capture.stop())
            self.assertEqual(capture.phase, 'inactive')
            self.assertIsNone(capture.error)
        finally:
            capture.stop()

    def test_monitor_and_reader_thread_failures_cleanup_started_helper(self):
        original = threading.Thread.start
        for failed_name in ('kilix-microphone-owner', 'kilix-voice-mic'):
            with self.subTest(thread=failed_name):
                def start(thread):
                    if thread.name == failed_name:
                        raise RuntimeError('Synthetic thread allocation failure')
                    return original(thread)
                capture = audio.MicCapture({'audio': {'capture_cmd': ['/bin/sleep', '20']}}, control=self.control())
                with mock.patch.object(threading.Thread, 'start', start), self.assertRaises(audio.AudioError):
                    capture.start()
                self.assertTrue(capture.stop())
                self.assertTrue(capture.cleanup_complete)
                self.assertIsNotNone(capture._owner.process.poll())
                owner, _ = self.start()
                self.assertTrue(owner.stop())

    def test_reader_construction_and_start_interruptions_reap_before_return(self):
        real_thread, real_start = threading.Thread, threading.Thread.start
        real_file = tempfile.TemporaryFile
        for boundary in ('construct', 'start', 'started'):
            for error_type in (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit):
                with self.subTest(boundary=boundary, error=error_type.__name__):
                    marker = self.root / f'{boundary}-{error_type.__name__}.json'
                    diagnostics, trees = [], []
                    capture = audio.MicCapture({'audio': {'capture_cmd': [sys.executable,
                        '-I', PEER, 'recorder', str(marker), 'escape']}}, control=self.control())

                    def diagnostic(*args, **kwargs):
                        stream = real_file(*args, **kwargs)
                        diagnostics.append(stream)
                        return stream

                    def refuse():
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            try:
                                trees.append(json.loads(marker.read_text()))
                                break
                            except (FileNotFoundError, json.JSONDecodeError):
                                time.sleep(.01)
                        self.assertTrue(trees, 'The actual recorder must precede the fault')
                        raise error_type('Synthetic reader allocation interruption')

                    def construct(*args, **kwargs):
                        if kwargs.get('name') == 'kilix-voice-mic':
                            refuse()
                        return real_thread(*args, **kwargs)

                    def start(thread):
                        if thread.name != 'kilix-voice-mic':
                            return real_start(thread)
                        if boundary == 'started':
                            real_start(thread)
                        refuse()

                    patch = (mock.patch.object(threading, 'Thread', construct) if boundary == 'construct'
                             else mock.patch.object(threading.Thread, 'start', start))
                    try:
                        with patch, mock.patch.object(tempfile, 'TemporaryFile', diagnostic):
                            expected = audio.AudioError if issubclass(error_type, Exception) else error_type
                            with self.assertRaises(expected):
                                capture.start()
                        self.assertTrue(capture.cleanup_complete)
                        self.assertEqual(capture.phase, 'inactive')
                        self.assertIsNotNone(capture._owner.process.poll())
                        self.assertTrue(all(stream.closed for stream in diagnostics))
                        self.assertTrue(all(not Path(f'/proc/{pid}').exists()
                                            for pid in (trees[0]['pid'], trees[0]['escaped'])))
                        self.assertIsNone(capture._reader)
                    finally:
                        capture.stop()


class CompletionTests(unittest.TestCase):
    def test_exact_credentials_marker_and_true_eof(self):
        before = len(os.listdir('/proc/self/fd'))
        for tail, valid in (([], True), ([b''], False), ([b'', b'garbage'], False), ([b'garbage'], False)):
            with self.subTest(tail=tail):
                receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                receiver.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
                sender.send(microphone.CLEANUP_MARKER)
                for data in tail:
                    sender.send(data)
                sender.close()
                self.assertEqual(microphone.cleanup_proven(receiver, os.getpid()), valid)
                receiver.close()
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    def test_forged_identity_truncation_and_rights_refuse_without_fd_leak(self):
        before = len(os.listdir('/proc/self/fd'))
        for mode in ('wrong-pid', 'wrong-marker', 'truncated', 'rights'):
            with self.subTest(mode=mode):
                receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                receiver.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
                if mode == 'rights':
                    with tempfile.TemporaryFile() as f:
                        sender.sendmsg([microphone.CLEANUP_MARKER], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [f.fileno()]))])
                else:
                    sender.send(b'x' * 100 if mode == 'truncated' else b'bad' if mode == 'wrong-marker' else microphone.CLEANUP_MARKER)
                sender.close()
                self.assertFalse(microphone.cleanup_proven(receiver, os.getpid() + (mode == 'wrong-pid')))
                receiver.close()
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)


class AuthorityTests(unittest.TestCase):
    def test_lock_answers_unknown_foreign_inactive_and_locked(self):
        uid = str(os.geteuid())
        for answer, expected in ((f'User={uid}\nActive=yes\nLockedHint=no', False),
                (f'User={uid}\nActive=no\nLockedHint=no', True),
                (f'User={uid}\nActive=yes\nLockedHint=yes', True),
                ('User=99999\nActive=yes\nLockedHint=no', None),
                (f'User={uid}\nActive=yes\nLockedHint=unknown', None), ('', None)):
            with self.subTest(answer=answer), mock.patch.object(microphone.LogindLockState, '_query', return_value=answer):
                self.assertIs(microphone.LogindLockState('2')(), expected)
        with mock.patch.object(microphone.LogindLockState, '_query', side_effect=subprocess.TimeoutExpired('fixture', 0.15)):
            self.assertIsNone(microphone.LogindLockState('2')())

    def test_actual_socket_disconnect_and_backpressure(self):
        for kind in (socket.SOCK_DGRAM, socket.SOCK_SEQPACKET):
            left, right = socket.socketpair(socket.AF_UNIX, kind)
            try:
                lifetime = microphone.ReceiverLifetime(left, lambda: b'valid partial')
                self.assertFalse(lifetime())
                self.assertEqual(right.recv(64), b'valid partial')
                right.close()
                lifetime._next = 0
                self.assertTrue(lifetime())
            finally:
                left.close(); right.close()
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            left.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            lifetime = microphone.ReceiverLifetime(left, lambda: b'x' * 1024)
            for _ in range(100):
                lifetime._next = 0
                if lifetime():
                    break
            self.assertTrue(lifetime())
        finally:
            left.close(); right.close()


