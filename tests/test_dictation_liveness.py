"""N-STOP on real daemon sockets and child processes, using synthetic audio.

The capture command and recogniser are selected through the daemon's normal
configuration. No device, downloaded model, or user store is used.
"""
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests import test_daemon
from tests.test_whisper_engine import FILES
from tests.test_terminal_outcomes import _worker_daemon, voiced
from voicelib import stt


PROVIDER = r"""#!{python}
import json, os, signal, sys, time
def log(event):
    with open(os.environ["TEST_PROCESSES"], "a") as out:
        out.write(json.dumps(dict(event, pid=os.getpid())) + "\n")
signal.signal(signal.SIGUSR1, lambda *_: sys.exit(0))
def close_output(*_):
    if os.environ.get("TEST_CLOSE_INPUT"):
        log({{"input_closed": True}})
        os.close(0)
        time.sleep(60)  # keep the process and output alive after losing input
        return
    log({{"output_closed": True}})
    os.close(1)
signal.signal(signal.SIGUSR2, close_output)
log({{"provider": True}})
if os.environ.get("TEST_SLOW_MODEL"):
    time.sleep(1)
print('{{"ready": true}}', flush=True)
while True:
    header = sys.stdin.buffer.readline()
    if not header:
        break
    size = json.loads(header)["pcm_bytes"]
    pcm = sys.stdin.buffer.read(size)
    log({{"decode": True}})
    while os.path.exists(os.environ["TEST_HANG_DECODE"]):
        time.sleep(.02)
    print('{{"text": "synthetic words"}}', flush=True)
"""

CAPTURE = r"""
import json, os, sys, time
with open(os.environ["TEST_PROCESSES"], "a") as out:
    out.write(json.dumps({"capture": True, "pid": os.getpid()}) + "\n")
mode = open(os.environ["TEST_CAPTURE_FILE"]).read()
while True:
    if mode != "stall":
        frame = b"\0" * 640 if mode == "silent" else b"\xff\x3f\x01\xc0" * 160
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()
    time.sleep(.02)
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux daemon protocol")
class DictationLivenessTests(unittest.TestCase):
    setUp = test_daemon.DaemonTestCase.setUp
    _stop_daemon = test_daemon.DaemonTestCase._stop_daemon
    _log_tail = test_daemon.DaemonTestCase._log_tail
    _wait_until_serving = test_daemon.DaemonTestCase._wait_until_serving
    _send = test_daemon.DaemonTestCase._send

    def _environment(self):
        env = test_daemon.DaemonTestCase._environment(self)
        root = Path(self.root)
        self.process_log = root / "processes.jsonl"
        provider = root / "provider"
        provider.write_text(PROVIDER.format(python=sys.executable))
        provider.chmod(0o700)
        capture = root / "capture.py"
        capture.write_text(CAPTURE)
        model = root / "model"
        model.mkdir()
        for name, data in FILES.items():
            (model / name).write_bytes(data)
        self.config = root / "config.json"
        self.config.write_text(json.dumps({
            "audio": {"capture_cmd": [sys.executable, str(capture)]},
            "stt": {"engine": "whisper", "model_path": str(model), "max_seconds": 30},
            "vad": {"silence_ms": 1500},
        }))
        mode = ("stall" if "without_frames" in self._testMethodName else
                "silent" if "silent" in self._testMethodName else "speech")
        self.capture_mode = root / "capture-mode"
        self.capture_mode.write_text(mode)
        self.hang_decode = root / "hang-decode"
        if "during_decode" in self._testMethodName:
            self.hang_decode.touch()
        env.update(KILIX_VOICE_CONFIG=str(self.config),
                   KILIX_VOICE_WHISPER=str(provider),
                   KILIX_VOICE_REQUIRE_CONSENT="0",
                   TEST_PROCESSES=str(self.process_log), TEST_CAPTURE_FILE=str(self.capture_mode),
                   TEST_HANG_DECODE=str(self.hang_decode))
        if "slow_model" in self._testMethodName:
            env["TEST_SLOW_MODEL"] = "1"
        if "closed_input" in self._testMethodName:
            env["TEST_CLOSE_INPUT"] = "1"
        self.turn_number = 0
        return env

    def entries(self, key):
        if not self.process_log.exists():
            return []
        return [row for line in self.process_log.read_text().splitlines()
                if (row := json.loads(line)).get(key)]

    def tearDown(self):
        # Kill only fixture children we recorded. This also contains the
        # deliberately broken pre-fix and mutation runs.
        for key in ("provider", "capture"):
            for row in self.entries(key):
                try:
                    pid = row["pid"]
                    status = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
                    if int(status[1]) == self.daemon.pid:
                        os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, FileNotFoundError):
                    pass
        self._stop_daemon()

    def wait_for(self, predicate, seconds, message):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return time.monotonic()
            time.sleep(.005)
        self.fail(message + "\n" + self._log_tail())

    def receiver(self, name=None, kind=socket.SOCK_DGRAM):
        endpoint = socket.socket(socket.AF_UNIX, kind)
        endpoint.bind(str(Path(self.session_dir) / (name or f"r{self.turn_number}.sock")))
        endpoint.settimeout(3)
        self.addCleanup(endpoint.close)
        return endpoint

    def start_turn(self, receiver):
        self.turn_number += 1
        result = self._send({"op": "dictate", "sock": receiver.getsockname()})
        self.assertTrue(result["ok"], result)
        self.wait_for(lambda: len(self.entries("provider")) >= self.turn_number
                      and len(self.entries("capture")) >= self.turn_number,
                      3, "fixture children did not start")
        pids = {key: self.entries(key)[-1]["pid"] for key in ("capture", "provider")}
        time.sleep(.06)  # allow at least one frame, when frames are enabled
        return result["turn"], pids

    def terminal(self, receiver):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            message = json.loads(receiver.recv(65536))
            if "final" in message or "error" in message:
                return message
        self.fail("no terminal datagram")

    def assert_stopped(self, started, job, pids, outcome):
        # Reaped, not merely a zombie; the daemon remains running.
        mic = self.wait_for(lambda: not Path(f"/proc/{pids['capture']}").exists(),
                            max(.001, started + 1 - time.monotonic()),
                            "capture survived the 1000 ms N-STOP limit")
        tree = self.wait_for(
            lambda: all(not Path(f"/proc/{pid}").exists() for pid in pids.values()),
            max(.001, started + 2 - time.monotonic()), "turn children survived 2000 ms")
        lease = self.wait_for(
            lambda: not self._send({"op": "status"})["status"]["listening"],
            max(.001, started + 2 - time.monotonic()), "microphone lease survived 2000 ms")
        record = self._send({"op": "status", "job": job})["status"]["job"]
        self.assertEqual(record["outcome"], outcome, record)
        print(f"N-STOP {self._testMethodName}: mic={(mic-started)*1000:.1f} ms "
              f"tree={(tree-started)*1000:.1f} ms lease={(lease-started)*1000:.1f} ms "
              "survivors=0", flush=True)

    def assert_healthy_again(self, started):
        self.capture_mode.write_text("speech")
        self.hang_decode.unlink(missing_ok=True)
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        self.assertTrue(self._send({"op": "stop-dictation"})["stopped"])
        self.assertEqual(self.terminal(receiver)["final"], "synthetic words")
        self.wait_for(lambda: not self._send({"op": "status"})["status"]["listening"],
                      2, "healthy turn did not release its lease")
        self.assertTrue(all(not Path(f"/proc/{pid}").exists() for pid in pids.values()))
        elapsed = time.monotonic() - started
        self.assertLessEqual(elapsed, 10)
        print(f"N-STOP recovery: healthy={elapsed:.3f} s survivors=0", flush=True)

    def provider_death(self, sig=signal.SIGKILL):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        os.kill(pids["provider"], sig)
        self.assert_stopped(started, job, pids, "failed")
        message = self.terminal(receiver)
        self.assertIn("error", message)
        self.assertNotIn("final", message)
        self.assertEqual(message["code"], "unavailable")
        self.assertIn("provider exited", message["error"])
        self.assert_healthy_again(started)

    def test_provider_death_during_speech(self):
        self.provider_death()

    def test_provider_death_during_silent_capture(self):
        self.provider_death()

    def test_provider_death_without_frames(self):
        self.provider_death()

    def test_provider_clean_exit_during_recording(self):
        self.provider_death(signal.SIGUSR1)

    def closed_output(self, *, decoding=False):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        if decoding:
            self.assertTrue(self._send({"op": "stop-dictation"})["stopped"])
            self.wait_for(lambda: bool(self.entries("decode")), 3, "decode did not start")
        started = time.monotonic()
        os.kill(pids["provider"], signal.SIGUSR2)
        self.wait_for(lambda: bool(self.entries("output_closed")), .5, "output did not close")
        self.assert_stopped(started, job, pids, "failed")
        message = self.terminal(receiver)
        self.assertIn("error", message)
        self.assertNotIn("final", message)
        self.assertEqual(message["code"], "unavailable")
        self.assert_healthy_again(started)

    def test_closed_pipe_during_silent_capture(self):
        self.closed_output()

    def test_closed_pipe_without_frames(self):
        self.closed_output()

    def test_closed_pipe_during_decode(self):
        self.closed_output(decoding=True)

    def test_closed_input_during_silent_capture(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        os.kill(pids["provider"], signal.SIGUSR2)
        self.wait_for(lambda: bool(self.entries("input_closed")), .5, "input did not close")
        self.assert_stopped(started, job, pids, "failed")
        message = self.terminal(receiver)
        self.assertIn("error", message)
        self.assertNotIn("final", message)
        self.assertEqual(message["code"], "unavailable")
        self.assert_healthy_again(started)

    def test_slow_model_during_silent_capture(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        time.sleep(.5)
        self.assertTrue(self._send({"op": "status"})["status"]["listening"])
        self._send({"op": "stop-dictation"})
        self.assertEqual(self.terminal(receiver)["final"], "synthetic words")
        self.wait_for(lambda: not self._send({"op": "status"})["status"]["listening"],
                      2, "slow healthy turn did not release its lease")
        self.assertTrue(all(not Path(f"/proc/{pid}").exists() for pid in pids.values()))

    @unittest.skipIf(os.geteuid() == 0, "root bypasses directory search permissions")
    def test_directory_search_revoked_during_silent_capture(self):
        directory = Path(self.session_dir) / "nested"
        directory.mkdir(mode=0o700)
        self.addCleanup(directory.chmod, 0o700)
        receiver = self.receiver("nested/r.sock")
        job, pids = self.start_turn(receiver)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.connect(receiver.getsockname())
            directory.chmod(0)
            sender.send(b'{"direct":true}')
            self.assertEqual(receiver.recv(100), b'{"direct":true}')
        time.sleep(.4)
        self.assertTrue(self._send({"op": "status"})["status"]["listening"],
                        "pathname EACCES cancelled a reachable receiver")
        self._send({"op": "stop-dictation"})
        self.assertEqual(self.terminal(receiver)["final"], "synthetic words")

    def test_receiver_unlinked_during_silent_capture(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        os.unlink(receiver.getsockname())
        self.assert_stopped(started, job, pids, "cancelled")
        message = self.terminal(receiver)  # connected socket still reaches it
        self.assertEqual(message["code"], "cancelled")
        self.assertNotIn("final", message)
        self.assert_healthy_again(started)

    def test_receiver_closed_leaving_stale_path_during_silent_capture(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        path = receiver.getsockname()
        started = time.monotonic()
        receiver.close()
        self.assertTrue(os.path.exists(path))
        self.assert_stopped(started, job, pids, "cancelled")
        self.assert_healthy_again(started)

    def replaced_receiver_parent(self, *, loop=False):
        directory = Path(self.session_dir) / "nested"
        directory.mkdir()
        receiver = self.receiver("nested/r.sock")
        job, pids = self.start_turn(receiver)
        directory.rename(directory.with_name("moved"))
        if loop:
            directory.symlink_to(directory.name)
        else:
            directory.write_text("not a directory")
        receiver.close()
        started = time.monotonic()
        with self.assertRaises(OSError) as caught:
            os.lstat(directory / "r.sock")
        self.assertEqual(caught.exception.errno, errno.ELOOP if loop else errno.ENOTDIR)
        self.assert_stopped(started, job, pids, "cancelled")
        self.assert_healthy_again(started)

    def test_parent_replaced_file_during_silent_capture(self):
        self.replaced_receiver_parent()

    def test_symlink_loop_parent_during_silent_capture(self):
        self.replaced_receiver_parent(loop=True)

    def test_receiver_replaced_at_same_path(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        path = receiver.getsockname()
        started = time.monotonic()
        os.unlink(path)
        replacement = self.receiver(Path(path).name)
        self.assert_stopped(started, job, pids, "cancelled")
        replacement.settimeout(.1)
        with self.assertRaises(socket.timeout):
            replacement.recv(65536)
        self.assertNotIn("final", self.terminal(receiver))
        self.assert_healthy_again(started)

    def test_receiver_unlinked_before_explicit_abort_is_cancelled(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        os.unlink(receiver.getsockname())
        time.sleep(.25)  # give the liveness check the first chance to notice
        self.assertTrue(self._send({"op": "stop-dictation", "mode": "abort"})["ok"])
        self.assert_stopped(started, job, pids, "cancelled")
        self.assertEqual(self.terminal(receiver)["code"], "cancelled")
        self.assert_healthy_again(started)

    def test_receiver_loss_during_decode(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        self.assertTrue(self._send({"op": "stop-dictation"})["stopped"])
        self.wait_for(lambda: bool(self.entries("decode")), 3, "decode did not start")
        started = time.monotonic()
        os.unlink(receiver.getsockname())
        self.assert_stopped(started, job, pids, "cancelled")
        self.assertEqual(self.terminal(receiver)["code"], "cancelled")
        self.assert_healthy_again(started)

    def test_receiver_loss_without_frames(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        receiver.close()
        self.assert_stopped(started, job, pids, "cancelled")
        self.assert_healthy_again(started)

    def test_seqpacket_receiver_hangup_during_silent_capture(self):
        receiver = self.receiver(kind=socket.SOCK_SEQPACKET)
        receiver.listen(1)
        job, pids = self.start_turn(receiver)
        peer, _ = receiver.accept()
        started = time.monotonic()
        peer.close()
        self.assert_stopped(started, job, pids, "cancelled")
        self.assert_healthy_again(started)

    def test_caller_sigkill_during_silent_capture(self):
        path = str(Path(self.session_dir) / "client.sock")
        client = subprocess.Popen([sys.executable, "-c",
            "import socket,sys,time; s=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);"
            "s.bind(sys.argv[1]); time.sleep(60)", path])
        def stop_client():
            if client.poll() is None:
                client.kill()
            client.wait(timeout=3)
        self.addCleanup(stop_client)
        self.wait_for(lambda: os.path.exists(path), 3, "caller did not bind")
        class Receiver:
            def getsockname(self):
                return path
        job, pids = self.start_turn(Receiver())
        started = time.monotonic()
        client.kill()
        client.wait(timeout=1)
        self.assert_stopped(started, job, pids, "cancelled")
        self.assert_healthy_again(started)

    def test_live_silent_receiver_is_not_cancelled_or_sent_heartbeats(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        time.sleep(.7)
        self.assertTrue(self._send({"op": "status"})["status"]["listening"])
        receiver.settimeout(.05)
        with self.assertRaises(socket.timeout):
            receiver.recv(65536)
        receiver.settimeout(3)
        self._send({"op": "stop-dictation"})
        self.assertEqual(self.terminal(receiver)["final"], "synthetic words")

    def finish_scoped_turn(self, receiver, job, pids, **fields):
        started = time.monotonic()
        reply = self._send(dict(op="stop-dictation", sock=receiver.getsockname(), **fields))
        self.assertEqual((reply.get("ok"), reply.get("stopped"), reply.get("mode")),
                         (True, True, "finish"), reply)
        message = self.terminal(receiver)
        self.assertEqual(message.get("final"), "synthetic words", message)
        self.assertNotIn("error", message)
        self.assert_stopped(started, job, pids, "completed")

    def test_scoped_finish_uses_the_existing_finish_path(self):
        for fields in ({}, {"mode": "finish"}):
            with self.subTest(fields=fields):
                receiver = self.receiver()
                job, pids = self.start_turn(receiver)
                self.finish_scoped_turn(receiver, job, pids, **fields)
        self.assertEqual(len(self.entries("decode")), 2)

    def test_scoped_abort_uses_the_existing_abort_path(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        reply = self._send(dict(op="stop-dictation", mode="abort",
                                sock=receiver.getsockname()))
        self.assertEqual((reply.get("ok"), reply.get("stopped"), reply.get("mode")),
                         (True, True, "abort"), reply)
        self.assert_stopped(started, job, pids, "cancelled")
        message = self.terminal(receiver)
        self.assertEqual(message.get("code"), "cancelled", message)
        self.assertNotIn("final", message)
        self.assertEqual(self.entries("decode"), [])

    def test_scoped_abort_during_decode_suppresses_final(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        path = receiver.getsockname()
        self.assertTrue(self._send(dict(op="stop-dictation", sock=path))["stopped"])
        self.wait_for(lambda: bool(self.entries("decode")), 3, "decode did not start")
        started = time.monotonic()
        reply = self._send(dict(op="stop-dictation", sock=path, mode="abort"))
        self.assertEqual((reply.get("ok"), reply.get("stopped"), reply.get("mode")),
                         (True, True, "abort"), reply)
        # Existing abort suppresses the final when decode returns. Release the
        # fixture so this checks that delivery boundary, not a decode timeout.
        self.hang_decode.unlink()
        self.assert_stopped(started, job, pids, "cancelled")
        message = self.terminal(receiver)
        self.assertEqual(message.get("code"), "cancelled", message)
        self.assertNotIn("final", message)

    def late_scoped_stop(self, **fields):
        earlier = self.receiver()
        old_job, old_pids = self.start_turn(earlier)
        self.finish_scoped_turn(earlier, old_job, old_pids)
        os.unlink(earlier.getsockname())
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        reply = self._send(dict(op="stop-dictation", sock=earlier.getsockname(),
                                id="late", **fields))
        self.assertEqual(reply, dict(ok=True, id="late", stopped=False,
                                     mode=fields.get("mode", "finish")))
        # Give both the worker and its receiver watcher a chance to run.
        receiver.settimeout(.25)
        with self.assertRaises(socket.timeout):
            receiver.recv(65536)
        receiver.settimeout(3)
        self.assertTrue(self._send({"op": "status"})["status"]["listening"])
        self.assertTrue(all(Path(f"/proc/{pid}").exists() for pid in pids.values()))
        self.finish_scoped_turn(receiver, job, pids)

    def test_scoped_late_finish_leaves_successor_recording(self):
        for fields in ({}, {"mode": "finish"}):
            with self.subTest(fields=fields):
                self.late_scoped_stop(**fields)

    def test_scoped_late_abort_leaves_successor_recording(self):
        self.late_scoped_stop(mode="abort")

    def test_scoped_stop_distinguishes_same_basename_receivers(self):
        for name in ("caller-a", "caller-b"):
            (Path(self.session_dir) / name).mkdir()
        earlier = self.receiver("caller-a/r.sock")
        receiver = self.receiver("caller-b/r.sock")
        job, pids = self.start_turn(receiver)
        for mode in ("finish", "abort"):
            with self.subTest(mode=mode):
                reply = self._send(dict(op="stop-dictation", mode=mode,
                                        sock=earlier.getsockname()))
                self.assertEqual(reply, dict(ok=True, id="", stopped=False, mode=mode))
                time.sleep(.15)
                self.assertTrue(self._send({"op": "status"})["status"]["listening"])
                self.assertTrue(all(Path(f"/proc/{pid}").exists() for pid in pids.values()))
        self.finish_scoped_turn(receiver, job, pids)

    def test_scoped_stop_without_current_turn_is_noop(self):
        path = str(Path(self.session_dir) / "removed.sock")
        for fields in ({}, {"mode": "finish"}, {"mode": "abort"}):
            with self.subTest(fields=fields):
                reply = self._send(dict(op="stop-dictation", sock=path, **fields))
                self.assertEqual(reply, dict(ok=True, id="", stopped=False,
                                             mode=fields.get("mode", "finish")))

    def test_unscoped_stop_still_stops_the_current_turn(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        started = time.monotonic()
        reply = self._send({"op": "stop-dictation"})
        self.assertEqual((reply.get("ok"), reply.get("stopped"), reply.get("mode")),
                         (True, True, "finish"), reply)
        self.assertEqual(self.terminal(receiver).get("final"), "synthetic words")
        self.assert_stopped(started, job, pids, "completed")

    def test_scoped_abort_after_unlink_stays_cancelled_with_watcher(self):
        for delay in (0, .25):
            with self.subTest(delay=delay):
                receiver = self.receiver()
                job, pids = self.start_turn(receiver)
                started = time.monotonic()
                os.unlink(receiver.getsockname())
                time.sleep(delay)
                reply = self._send(dict(op="stop-dictation", mode="abort",
                                        sock=receiver.getsockname()))
                self.assertTrue(reply["ok"], reply)
                self.assertEqual(reply["mode"], "abort")
                self.assert_stopped(started, job, pids, "cancelled")
                message = self.terminal(receiver)
                self.assertEqual(message.get("code"), "cancelled", message)
                self.assertNotIn("final", message)
        self.assertEqual(self.entries("decode"), [])

    def test_scoped_stop_normalises_the_dictate_receiver(self):
        receiver = self.receiver()
        alias = str(Path(self.session_dir) / "alias.sock")
        os.symlink(receiver.getsockname(), alias)

        class Alias:
            def getsockname(self):
                return alias

        # Dictate records the resolved receiver; stop receives the alias again.
        job, pids = self.start_turn(Alias())
        started = time.monotonic()
        reply = self._send(dict(op="stop-dictation", sock=alias))
        self.assertEqual((reply.get("ok"), reply.get("stopped")), (True, True), reply)
        self.assertEqual(self.terminal(receiver).get("final"), "synthetic words")
        self.assert_stopped(started, job, pids, "completed")

    def test_malformed_scoped_stop_is_refused_without_stopping(self):
        receiver = self.receiver()
        job, pids = self.start_turn(receiver)
        for raw in (None, "", False, [], str(Path(self.root) / "outside.sock"),
                    str(Path(self.session_dir) / ("x" * 4096))):
            for mode in ("finish", "abort"):
                with self.subTest(raw=raw, mode=mode):
                    reply = self._send(dict(op="stop-dictation", sock=raw, mode=mode))
                    self.assertFalse(reply["ok"], reply)
                    self.assertEqual(reply.get("code"), "malformed")
                    self.assertTrue(self._send({"op": "status"})["status"]["listening"])
        self.finish_scoped_turn(receiver, job, pids)


class DictationLivenessBoundaryTests(unittest.TestCase):
    def receiver_turn(self, directory):
        path = str(Path(directory) / "receiver.sock")
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(receiver.close)
        receiver.bind(path)
        receiver.settimeout(.1)
        sender = voiced._connect_dictation(path)
        self.addCleanup(sender.close)
        return receiver, voiced._DictationTurn("boundary", sender, receiver_path=path)

    def test_unlinked_receiver_cannot_get_final_before_watcher_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver, turn = self.receiver_turn(directory)
            sent = []
            daemon = _worker_daemon(sent)
            os.unlink(turn.receiver_path)
            with self.assertRaises(voiced.Cancelled):
                daemon._deliver_terminal(turn, {"final": "must be discarded", "segment": turn.id})
            self.assertEqual(sent, [])
            self.assertIsNone(turn.outcome)

    def test_whistle_dead_child_fails_health(self):
        engine = object.__new__(stt.WhistleStt)
        engine._closed = False
        engine._lifecycle = threading.Lock()
        with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
            engine._process = process
            process.wait(timeout=3)
            with self.assertRaises(stt.SttError):
                engine.check_health()

    def test_broken_provider_input_fails_without_waiting_for_exit(self):
        process = subprocess.Popen(
            [sys.executable, "-c",
             "import os,time; os.close(0); print('closed', flush=True); time.sleep(60)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            self.assertEqual(process.stdout.readline(), b"closed\n")
            engine = object.__new__(stt.WhisperStt)
            started = time.monotonic()
            with self.assertRaisesRegex(stt.SttError, "being sent audio"):
                engine._write_all(process, b"audio", started + 3)
            self.assertLess(time.monotonic() - started, .5)
            self.assertIsNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)

    def test_output_eof_interrupts_blocked_provider_input(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import os,signal,time; "
             "signal.signal(signal.SIGUSR2, lambda *_: os.close(1)); "
             "print('ready', flush=True); time.sleep(60)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        engine = object.__new__(stt.WhisperStt)
        engine._closed = False
        engine._lifecycle = threading.Lock()
        engine._process = process
        errors = []
        def write():
            try:
                engine._write_all(process, b"\0" * (1 << 20), time.monotonic() + 10)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=write)
        try:
            self.assertEqual(process.stdout.readline(), b"ready\n")
            os.set_blocking(process.stdin.fileno(), False)
            worker.start()
            time.sleep(.1)
            self.assertTrue(worker.is_alive(), "fixture input pipe did not fill")
            process.send_signal(signal.SIGUSR2)
            worker.join(1)
            self.assertFalse(worker.is_alive(), "output EOF left input delivery blocked")
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], stt.SttError)
        finally:
            if process.poll() is None:
                process.kill()
            if worker.ident is not None:
                worker.join(3)
            process.communicate(timeout=3)

    def test_emfile_does_not_cancel_reachable_receiver(self):
        import resource
        with tempfile.TemporaryDirectory() as directory:
            receiver, turn = self.receiver_turn(directory)
            self.assertTrue(turn.receiver_alive())
            limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            descriptors = []
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, limit[0]), limit[1]))
                while True:
                    try:
                        descriptors.append(os.open("/dev/null", os.O_RDONLY))
                    except OSError as error:
                        self.assertEqual(error.errno, errno.EMFILE)
                        break
                alive = turn.receiver_alive()
                turn.receiver.send(b"existing receiver still reachable")
                self.assertEqual(receiver.recv(100), b"existing receiver still reachable")
                self.assertTrue(alive, "EMFILE cancelled a reachable receiver")
            finally:
                for fd in descriptors:
                    os.close(fd)
                resource.setrlimit(resource.RLIMIT_NOFILE, limit)

    def test_inconclusive_receiver_probes_preserve_delivery(self):
        errors = (errno.EACCES, errno.EPERM, errno.EMFILE, errno.ENFILE,
                  errno.ENOBUFS, errno.ENOMEM, errno.EINTR, errno.EAGAIN)
        with tempfile.TemporaryDirectory() as directory:
            receiver, turn = self.receiver_turn(directory)
            for code in errors:
                for operation in ("stat", "socket", "connect"):
                    with self.subTest(errno=code, operation=operation):
                        error = OSError(code, os.strerror(code))
                        target, name = {
                            "stat": (voiced.os, "lstat"),
                            "socket": (voiced.socket, "socket"),
                            "connect": (socket.socket, "connect"),
                        }[operation]
                        with mock.patch.object(target, name, side_effect=error):
                            self.assertTrue(turn.receiver_alive())
                            sent = []
                            delivery = voiced._DictationTurn(
                                "delivery", turn.receiver, receiver_path=turn.receiver_path)
                            self.assertTrue(delivery.receiver_alive())
                            self.assertTrue(_worker_daemon(sent)._deliver_terminal(
                                delivery, {"final": "healthy words"}))
                            self.assertEqual(sent, [{"final": "healthy words"}])
                        turn.receiver.send(b"still reachable")
                        self.assertEqual(receiver.recv(100), b"still reachable")

    def test_definite_receiver_probe_failures_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver, turn = self.receiver_turn(directory)
            for code in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTDIR,
                         errno.ELOOP, errno.ENAMETOOLONG, errno.ENOTSOCK,
                         errno.EPROTOTYPE):
                for target, name in ((voiced.os, "lstat"), (voiced.socket, "socket"),
                                     (socket.socket, "connect")):
                    with self.subTest(errno=code, operation=name):
                        with mock.patch.object(target, name, side_effect=OSError(code, os.strerror(code))):
                            self.assertFalse(turn.receiver_alive())
                            if name == "lstat":
                                initial = voiced._DictationTurn(
                                    "initial", turn.receiver, receiver_path=turn.receiver_path)
                        if name == "lstat":
                            self.assertFalse(initial.receiver_alive(),
                                             "initial definite loss must remain recorded")
            os.unlink(turn.receiver_path)
            Path(turn.receiver_path).write_text("not a socket")
            self.assertFalse(turn.receiver_alive())

    def test_unknown_receiver_probe_errors_preserve_delivery(self):
        # Filesystem I/O and stale mount errors need not mean the connected
        # peer died. An unrecognized future errno follows the same policy.
        with tempfile.TemporaryDirectory() as directory:
            receiver, turn = self.receiver_turn(directory)
            for code in (errno.EIO, errno.ESTALE, errno.EINVAL, 123456):
                for target, name in ((voiced.os, "lstat"), (voiced.socket, "socket"),
                                     (socket.socket, "connect")):
                    with self.subTest(errno=code, operation=name):
                        with mock.patch.object(target, name, side_effect=OSError(code, "probe failed")):
                            self.assertTrue(turn.receiver_alive())
                        turn.receiver.send(b"still reachable")
                        self.assertEqual(receiver.recv(100), b"still reachable")

    def test_seqpacket_hangup_is_seen_even_when_path_is_inaccessible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "receiver.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as receiver:
                receiver.bind(path)
                receiver.listen(1)
                with voiced._connect_dictation(path) as sender:
                    peer, _ = receiver.accept()
                    self.addCleanup(peer.close)
                    turn = voiced._DictationTurn("boundary", sender, receiver_path=path)
                    with mock.patch.object(voiced.os, "lstat",
                                           side_effect=PermissionError(errno.EACCES, "denied")):
                        self.assertTrue(turn.receiver_alive())
                        peer.close()
                        self.assertFalse(turn.receiver_alive())

    def test_connected_datagram_refusal_explains_receiver_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "receiver.sock")
            peer_path = str(Path(directory) / "peer.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver, \
                    socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as peer:
                receiver.bind(path)
                peer.bind(peer_path)
                receiver.connect(peer_path)
                receiver.settimeout(.1)
                peer.sendto(b"healthy chosen peer", path)
                self.assertEqual(receiver.recv(100), b"healthy chosen peer")
                with self.assertRaises(voiced.DaemonError) as caught:
                    voiced._connect_dictation(path)
                message = str(caught.exception)
                self.assertIn("unconnected datagram", message)
                self.assertNotIn("probably gone away", message)

    def run_turn(self, *, health_error=False, send_fails=False):
        sent = []
        daemon = _worker_daemon(sent)
        daemon._stopping = threading.Event()
        daemon._cfg = {"stt": {"engine": "null", "max_seconds": 30}}
        turn = voiced._DictationTurn("listen-boundary", mock.Mock())
        capture = mock.Mock(rate=16000, overruns=0, frame_bytes=640, error=None)
        reads = []
        def read(**kwargs):
            if reads:
                turn.stop.set()
            reads.append(True)
            return b"\0" * 640
        capture.read.side_effect = read
        engine = mock.Mock(supports_partials=True)
        engine.feed.return_value = "unfinished words"
        engine.end_utterance.return_value = "must not be delivered"
        if health_error:
            engine.check_health.side_effect = stt.SttError("provider exited")
        if send_fails:
            daemon._send = lambda receiver, msg: sent.append(msg) or "partial" not in msg
        grant = mock.Mock()
        order = []
        capture.stop.side_effect = lambda: order.append("capture stopped")
        engine.close.side_effect = lambda: order.append("engine closed")
        grant.reaped.side_effect = lambda: order.append("lease reaped")
        daemon._dictate = lambda t: daemon._record_turn(t, stt.resolve_stt(daemon._cfg), grant)
        with mock.patch.object(voiced.audio, "MicCapture", return_value=capture), \
                mock.patch.object(voiced.stt_lib, "make_stt", return_value=engine):
            daemon._run_dictation(turn)
        return turn, sent, engine, order

    def test_failed_partial_send_aborts_without_decoding_or_final(self):
        turn, sent, engine, order = self.run_turn(send_fails=True)
        self.assertEqual(turn.outcome.outcome, "cancelled")
        self.assertEqual(sent[-1]["code"], "cancelled")
        self.assertFalse(any("final" in message for message in sent))
        engine.end_utterance.assert_not_called()
        self.assertEqual(order[-3:], ["capture stopped", "engine closed", "lease reaped"])

    def test_provider_failure_reaps_capture_and_engine_before_lease(self):
        turn, sent, engine, order = self.run_turn(health_error=True)
        self.assertEqual(turn.outcome.outcome, "failed")
        self.assertEqual(sent[-1]["code"], "unavailable")
        self.assertFalse(any("final" in message for message in sent))
        engine.end_utterance.assert_not_called()
        self.assertEqual(order, ["capture stopped", "engine closed", "lease reaped"])
