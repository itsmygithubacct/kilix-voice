"""N-STOP on real daemon sockets and child processes, using synthetic audio.

The capture command and recogniser are selected through the daemon's normal
configuration. No device, downloaded model, or user store is used.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
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
log({{"provider": True}})
print('{{"ready": true}}', flush=True)
while True:
    header = sys.stdin.buffer.readline()
    if not header:
        break
    size = json.loads(header)["pcm_bytes"]
    pcm = sys.stdin.buffer.read(size)
    log({{"decode": True}})
    if os.path.exists(os.environ["TEST_HANG_DECODE"]):
        time.sleep(60)
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


class DictationLivenessBoundaryTests(unittest.TestCase):
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
