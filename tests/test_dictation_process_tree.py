"""Track real provider/capture descendants through every dictation teardown.

The test is a Linux subreaper so dead grandchildren can be reaped even in a
container whose PID 1 does not reap. Assertions only waitpid(WNOHANG); killing
leaked fixture PIDs belongs to cleanup, after the assertion has failed.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

from tests import test_daemon, test_dictation_liveness as live
from voicelib import audio


FORK = r'''
import json, os, signal, time
def fork_descendants(owner):
    if os.fork():
        return
    # Keep inherited provider/capture descriptors, and resist polite teardown.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    depth = 1 if os.fork() else 2
    row = dict(descendant=owner, depth=depth, pid=os.getpid(), pgid=os.getpgrp())
    fd = os.open(os.environ["TEST_PROCESSES"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    os.write(fd, (json.dumps(row) + "\n").encode())
    os.close(fd)
    time.sleep(60)
    os._exit(0)
'''


class _TreeFixture:
    engine_name = "whisper"
    _stop_daemon = live.DictationLivenessTests._stop_daemon
    _log_tail = live.DictationLivenessTests._log_tail
    _wait_until_serving = live.DictationLivenessTests._wait_until_serving
    _send = live.DictationLivenessTests._send
    entries = live.DictationLivenessTests.entries
    receiver = live.DictationLivenessTests.receiver
    terminal = live.DictationLivenessTests.terminal
    wait_for = live.DictationLivenessTests.wait_for

    def setUp(self):
        libc = ctypes.CDLL(None, use_errno=True)
        prior = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(prior), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        self.addCleanup(libc.prctl, 36, prior.value, 0, 0, 0)
        # Whistle has a fixed native provider command. Substitute only that
        # command in a private launcher; use the real WhistleStt construction,
        # session, pipes, held weights, cancellation, close and daemon paths.
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "daemon.py"
            root = Path(test_daemon.DAEMON).parent
            launcher.write_text(
                "import os, runpy, sys\n"
                f"sys.path.insert(0, {str(root)!r})\n"
                "from voicelib import stt\n"
                "stt.WhistleStt._command = lambda self: "
                "[sys.executable, os.environ['TEST_TREE_PROVIDER']]\n"
                f"runpy.run_path({str(test_daemon.DAEMON)!r}, run_name='__main__')\n")
            with mock.patch.object(test_daemon, "DAEMON", launcher):
                test_daemon.DaemonTestCase.setUp(self)
        self.addCleanup(self.clean_descendants)

    def _environment(self):
        env = live.DictationLivenessTests._environment(self)
        root = Path(self.root)
        provider = root / "provider"
        source = provider.read_text()
        source = source.replace('log({"provider": True})',
                                FORK + '\nfork_descendants("provider")\nlog({"provider": True})')
        if "decode_error" in self._testMethodName:
            source = source.replace('{"text": "synthetic words"}', '{"error": "decode refused"}')
        provider.write_text(source)
        if "provider_exit" not in self._testMethodName:
            capture = root / "capture.py"
            capture.write_text(FORK + '\nfork_descendants("capture")\n' + capture.read_text())
        if self.engine_name == "whistle":
            (root / "model/whistle.cact").write_bytes(b"synthetic weights")
            library = root / "library.so"
            library.write_bytes(b"synthetic provider command substitutes this library")
            cfg = json.loads(self.config.read_text())
            cfg["stt"]["engine"] = "whistle"
            self.config.write_text(json.dumps(cfg))
            env["KILIX_VOICE_WHISTLE_LIBRARY"] = str(library)
        env["TEST_TREE_PROVIDER"] = str(provider)
        return env

    def descendants_gone(self):
        for row in self.entries("descendant"):
            try:
                os.waitpid(row["pid"], os.WNOHANG)
            except ChildProcessError:
                pass
        return all(not Path(f"/proc/{row['pid']}").exists()
                   for row in self.entries("descendant"))

    def clean_descendants(self):
        for row in self.entries("descendant"):
            try:
                os.kill(row["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3
        while not self.descendants_gone() and time.monotonic() < deadline:
            time.sleep(.005)

    def tearDown(self):
        self.clean_descendants()
        live.DictationLivenessTests.tearDown(self)

    def start_tree(self):
        receiver = self.receiver()
        job, pids = live.DictationLivenessTests.start_turn(self, receiver)
        count = 2 if "provider_exit" in self._testMethodName else 4
        self.wait_for(lambda: len(self.entries("descendant")) == count,
                      2, "fixture descendants did not start")
        self.assertEqual({row["depth"] for row in self.entries("descendant")}, {1, 2})
        return receiver, job, pids

    def assert_tree_stopped(self, started, job, pids, outcome):
        live.DictationLivenessTests.assert_stopped(self, started, job, pids, outcome)
        self.wait_for(self.descendants_gone, max(.001, started + 2 - time.monotonic()),
                      "descendants survived the 2000 ms teardown limit: "
                      + repr(self.entries("descendant")))
        print(f"TREE {self.engine_name} {self._testMethodName}: "
              f"{self.entries('descendant')} elapsed_ms={(time.monotonic()-started)*1000:.1f} "
              "survivors=0", flush=True)

    def test_provider_death_during_silent_capture(self):
        receiver, job, pids = self.start_tree()
        started = time.monotonic()
        os.kill(pids["provider"], signal.SIGKILL)
        self.assert_tree_stopped(started, job, pids, "failed")
        self.assertEqual(self.terminal(receiver).get("code"), "unavailable")

    def test_provider_exit_descendants_during_silent_capture(self):
        receiver, job, pids = self.start_tree()
        started = time.monotonic()
        os.kill(pids["provider"], signal.SIGUSR1)
        self.assert_tree_stopped(started, job, pids, "failed")
        self.assertEqual(self.terminal(receiver).get("code"), "unavailable")

    def test_receiver_loss_during_silent_capture(self):
        receiver, job, pids = self.start_tree()
        started = time.monotonic()
        os.unlink(receiver.getsockname())
        self.assert_tree_stopped(started, job, pids, "cancelled")
        self.assertEqual(self.terminal(receiver).get("code"), "cancelled")

    def test_normal_finish(self):
        receiver, job, pids = self.start_tree()
        started = time.monotonic()
        self.assertTrue(self._send({"op": "stop-dictation"})["stopped"])
        self.assert_tree_stopped(started, job, pids, "completed")
        self.assertEqual(self.terminal(receiver).get("final"), "synthetic words")

    def test_abort_during_silent_capture(self):
        receiver, job, pids = self.start_tree()
        started = time.monotonic()
        self.assertTrue(self._send({"op": "stop-dictation", "mode": "abort"})["stopped"])
        self.assert_tree_stopped(started, job, pids, "cancelled")
        self.assertEqual(self.terminal(receiver).get("code"), "cancelled")
        self.assertFalse(self.entries("decode"))

    def test_private_groups_and_daemon_interrupt_during_silent_capture(self):
        _receiver, _job, pids = self.start_tree()
        for pid in pids.values():
            self.assertEqual(os.getpgid(pid), pid)
            self.assertEqual(os.getsid(pid), pid)
            self.assertNotEqual(os.getpgid(self.daemon.pid), pid)
        for row in self.entries("descendant"):
            self.assertEqual(row["pgid"], pids[row["descendant"]])
        started = time.monotonic()
        self.daemon.send_signal(signal.SIGINT)
        self.wait_for(lambda: self.daemon.poll() is not None, 2, "daemon ignored SIGINT")
        self.wait_for(self.descendants_gone, max(.001, started + 2 - time.monotonic()),
                      "daemon interrupt left descendants")
        self.assertTrue(all(not Path(f"/proc/{pid}").exists() for pid in pids.values()))

    def test_receiver_loss_during_decode(self):
        receiver, job, pids = self.start_tree()
        self.assertTrue(self._send({"op": "stop-dictation"})["stopped"])
        self.wait_for(lambda: bool(self.entries("decode")), 3, "decode did not start")
        started = time.monotonic()
        os.unlink(receiver.getsockname())
        self.assert_tree_stopped(started, job, pids, "cancelled")
        self.assertEqual(self.terminal(receiver).get("code"), "cancelled")

    def test_decode_error(self):
        receiver, job, pids = self.start_tree()
        started = time.monotonic()
        self.assertTrue(self._send({"op": "stop-dictation"})["stopped"])
        self.assert_tree_stopped(started, job, pids, "failed")
        message = self.terminal(receiver)
        self.assertIn("error", message)
        self.assertNotIn("final", message)

    def test_capture_group_after_leader_exit(self):
        # A capture leader may exit while descendants retain its pipe. Stop
        # still owns the original group and must unblock and join the reader.
        capture = audio.MicCapture({"audio": {"capture_cmd": [
            sys.executable, "-c", FORK + '\nfork_descendants("capture")\ntime.sleep(60)']}})
        with mock.patch.dict(os.environ, {"TEST_PROCESSES": str(self.process_log)}):
            capture.start()
        self.addCleanup(capture.stop)
        self.wait_for(lambda: len(self.entries("descendant")) == 2, 2, "no capture descendants")
        process = capture._process
        process.kill()
        process.wait(timeout=1)
        started = time.monotonic()
        # The old stop can block closing a pipe whose reader holds its lock.
        # A thread keeps a failing pre-fix run bounded; cleanup releases it.
        import threading
        worker = threading.Thread(target=capture.stop, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 3)
        self.addCleanup(self.clean_descendants)
        self.wait_for(self.descendants_gone, 1, "capture descendants survived leader exit")
        worker.join(1)
        self.assertFalse(worker.is_alive(), "capture stop left its reader blocked")
        self.assertLess(time.monotonic() - started, 2)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux process groups and subreaping")
class WhisperProcessTreeTests(_TreeFixture, unittest.TestCase):
    pass


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux process groups and subreaping")
class WhistleProcessTreeTests(_TreeFixture, unittest.TestCase):
    engine_name = "whistle"
