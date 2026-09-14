"""Exactly one terminal outcome for every job, joinable by its id.

A real Daemon, built by its constructor, driven through Daemon._dispatch: a
real audio.Player feeding a fake sink process, real SEQPACKET receiver
sockets, and scripted synthesiser and recogniser engines -- the shape the
V12, V14 and V15 vectors use -- so what is observed is the daemon's own job
lifecycle, not a helper's.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_terminal", importlib.machinery.SourceFileLoader(
        "kilix_voiced_terminal", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import consent, jobs, protocol, stt as stt_lib  # noqa: E402

BIG = 262144        # stalls behind the sink's hold file
SMALL = 16000
TWO = "First sentence here. Second sentence here."
FOUR = ("First sentence here. Second sentence here. "
        "Third sentence here. Fourth sentence here.")

SINK = r'''
import os, sys, time
d = sys.argv[1]
if os.path.exists(os.path.join(d, "fail")):
    sys.exit(3)
log = open(os.path.join(d, "sink-%d.log" % os.getpid()), "ab", buffering=0)
inp = sys.stdin.buffer
total = 0
hold = os.path.join(d, "hold")
while True:
    if total >= 65536:
        while os.path.exists(hold):
            time.sleep(0.005)
    chunk = inp.read1(4096)
    if not chunk:
        break
    total += len(chunk)
    log.write(b"%d\n" % len(chunk))
log.write(b"eof\n")
'''

RECORDER = r'''
import sys, time
out = sys.stdout.buffer
while True:
    out.write(b"\x00" * 640)
    out.flush()
    time.sleep(0.02)
'''


def wait_until(predicate, timeout: float, step: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


class Receiver:
    """A real SEQPACKET listener standing in for a subscriber."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.bind(path)
        self.listener.listen(2)
        self.listener.settimeout(0.05)
        self.msgs: list[dict] = []
        self.eof = False
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        end = time.monotonic() + 60
        conn = None
        while not self.closed and time.monotonic() < end:
            try:
                conn, _ = self.listener.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                return
        if conn is None:
            return
        conn.settimeout(0.05)
        try:
            while not self.closed:
                try:
                    data = conn.recv(1 << 20)
                except socket.timeout:
                    continue
                if not data:
                    self.eof = True
                    return
                for line in data.decode("utf-8").splitlines():
                    if line.strip():
                        self.msgs.append(json.loads(line))
        except OSError:
            return
        finally:
            conn.close()

    def close(self) -> None:
        self.closed = True
        self.thread.join(timeout=2)
        self.listener.close()


class ScriptedTts:
    name = "scripted"
    model = "espeak"
    voice = "en-us"
    rate = 170
    seed = 0

    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.calls = 0

    def synth(self, text, *, budget=None):
        size = self.sizes[min(self.calls, len(self.sizes) - 1)]
        self.calls += 1
        return b"\x00" * size, 22050

    def cancel(self) -> None:
        pass


class ScriptedStt:
    name = "scripted"
    supports_partials = True

    def __init__(self) -> None:
        self.feeds = 0

    def start_utterance(self) -> None:
        pass

    def feed(self, frame):
        self.feeds += 1
        return f"word{self.feeds}"

    def end_utterance(self) -> str:
        return "hello world"

    def close(self) -> None:
        pass


class _LiveTurnsFixture(unittest.TestCase):
    MAX_SECONDS = 2

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="kv-term-", dir="/tmp")
        self.addCleanup(shutil.rmtree, self.root, True)
        for name, source in (("sink.py", SINK), ("rec.py", RECORDER)):
            with open(os.path.join(self.root, name), "w", encoding="utf-8") as handle:
                handle.write(source)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("KILIX_", "GPU_TERMINAL_"))}
        env.update(HOME=self.root,
                   GPU_TERMINAL_HOME=os.path.join(self.root, "gt"),
                   GPU_TERMINAL_SETTINGS_FILE=os.path.join(self.root, "settings.conf"),
                   KILIX_SESSION_HOME=os.path.join(self.root, "session"),
                   KILIX_DATA_HOME=os.path.join(self.root, "data"))
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tts_sizes = [SMALL]
        for target, factory in ((voiced.tts_lib, "make_tts"), (voiced.stt_lib, "make_stt")):
            fake = ((lambda *a, **k: ScriptedTts(self.tts_sizes)) if factory == "make_tts"
                    else (lambda *a, **k: ScriptedStt()))
            patch = mock.patch.object(target, factory, fake)
            patch.start()
            self.addCleanup(patch.stop)
        config = os.path.join(self.root, "voiced.json")
        with open(config, "w", encoding="utf-8") as handle:
            json.dump({"audio": {"play_cmd": [sys.executable, "-B",
                                              os.path.join(self.root, "sink.py"),
                                              self.root, "{rate}"],
                                 "capture_cmd": [sys.executable, "-B",
                                                 os.path.join(self.root, "rec.py")]},
                       "stt": {"engine": "null", "max_seconds": self.MAX_SECONDS},
                       "daemon": {"idle_seconds": 0}}, handle)
        self.daemon = voiced.Daemon(config_path=config, idle_seconds=0)
        os.makedirs(self.daemon._session_dir, mode=0o700, exist_ok=True)
        self.receivers: list[Receiver] = []
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        d = self.daemon
        d._stopping.set()
        for name in ("hold", "fail"):
            self.flag(name, False)
        with d._lock:
            speech, dictation = d._speech, d._dictation
        for turn, token in ((speech, "cancel"), (dictation, "stop")):
            if turn is not None:
                getattr(turn, token).set()
                if turn.thread is not None:
                    turn.thread.join(timeout=5)
        if d._player is not None:
            d._player.close()
        for receiver in self.receivers:
            receiver.close()

    def call(self, **request) -> dict:
        return self.daemon._dispatch(protocol.encode(request))

    def status(self, **fields) -> dict:
        return self.call(op="status", **fields)

    def settled(self, job: str, timeout: float = 20) -> dict:
        def record():
            return (self.status(job=job).get("status") or {}).get("job") or {}
        self.assertTrue(wait_until(lambda: record().get("state") == "settled", timeout),
                        f"{job} never settled: {record()}")
        return record()

    def receiver(self) -> Receiver:
        path = os.path.join(self.daemon._session_dir, f"r{len(self.receivers)}.sock")
        receiver = Receiver(path)
        self.receivers.append(receiver)
        return receiver

    def flag(self, name: str, on: bool) -> None:
        path = os.path.join(self.root, name)
        if on:
            open(path, "w").close()
        elif os.path.exists(path):
            os.unlink(path)

    def sink_stalled(self) -> bool:
        for name in os.listdir(self.root):
            if name.startswith("sink-"):
                with open(os.path.join(self.root, name), "rb") as handle:
                    total = sum(int(x) for x in handle.read().split() if x.isdigit())
                if total >= 65536:
                    return True
        return False

    def idle(self) -> bool:
        return self.daemon._speech is None and not self.daemon._arbiter.speaking

    def grant_consent(self) -> None:
        resolved = stt_lib.resolve_stt(self.daemon._cfg)
        consent.grant("dictation", consent.capture_digest(
            resolved.model_id, resolved.engine,
            consent.payload_digest_at(resolved.model_dir, resolved.engine)),
            model_id=resolved.model_id)


class SpeechOutcomes(_LiveTurnsFixture):

    def test_a_completed_speak_without_a_subscriber_settles_completed(self) -> None:
        reply = self.call(op="speak", id="a", text=TWO)
        self.assertTrue(reply["ok"], reply)
        record = self.settled(reply["turn"])
        self.assertEqual((record["kind"], record["outcome"]), ("speech", "completed"))
        self.assertNotIn("code", record)
        self.assertIn(reply["turn"], [r["job"] for r in self.status()["status"]["jobs"]])

    def test_stop_speech_mid_turn_settles_cancelled_and_reports_no_failure(self) -> None:
        self.tts_sizes = [BIG]
        self.flag("hold", True)
        serial = self.status()["status"]["speech_error_serial"]
        job = self.call(op="speak", text=FOUR)["turn"]
        self.assertTrue(wait_until(self.sink_stalled, 15))
        self.assertIs(self.call(op="stop-speech")["stopped"], True)
        self.flag("hold", False)
        record = self.settled(job)
        self.assertEqual((record["outcome"], record["code"]), ("cancelled", "cancelled"))
        self.assertEqual(self.status()["status"]["speech_error_serial"], serial)

    def test_barge_in_by_dictation_settles_the_speech_cancelled(self) -> None:
        self.grant_consent()
        self.tts_sizes = [BIG]
        self.flag("hold", True)
        job = self.call(op="speak", text=FOUR)["turn"]
        self.assertTrue(wait_until(self.sink_stalled, 15))
        receiver = self.receiver()
        dictation = self.call(op="dictate", sock=receiver.path)
        self.assertTrue(dictation["ok"], dictation)
        self.flag("hold", False)
        self.assertEqual(self.settled(job)["outcome"], "cancelled")
        self.assertTrue(wait_until(lambda: receiver.eof, 20))

    def test_a_sink_that_fails_settles_failed_and_status_names_the_turn(self) -> None:
        self.flag("fail", True)
        serial = self.status()["status"]["speech_error_serial"]
        job = self.call(op="speak", text="Only one secret sentence here.")["turn"]
        record = self.settled(job)
        self.assertEqual((record["outcome"], record["code"]), ("failed", "unavailable"))
        status = self.status()["status"]
        self.assertEqual(status["speech_error_serial"], serial + 1)
        self.assertEqual(status["speech_error_turn"], job)
        self.assertNotIn("secret", json.dumps(status))       # never the spoken text

    def test_a_budget_spent_during_playback_settles_deadline(self) -> None:
        self.tts_sizes = [BIG]
        self.flag("hold", True)
        job = self.call(op="speak", text=FOUR, deadline_ms=400)["turn"]
        record = self.settled(job)
        self.assertEqual((record["outcome"], record["code"]), ("deadline", "deadline"))

    def test_a_worker_that_settles_inside_start_is_not_left_running(self) -> None:
        # The job ends before the handler gets back from _start. Recording it
        # as begun after that must not put a settled job back to running.
        self.daemon._start = lambda thread, undo: thread.run()
        job = self.call(op="speak", text="Only one sentence here.")["turn"]
        self.assertEqual(self.status(job=job)["status"]["job"]["state"], "settled")

    def test_status_for_an_unknown_job_is_not_found_and_a_bad_id_is_malformed(self) -> None:
        self.assertEqual(self.status(job="speak-999")["code"], protocol.ERR_NOT_FOUND)
        self.assertEqual(self.status(job="rm -rf")["code"], protocol.ERR_MALFORMED)


class TerminalMessages(_LiveTurnsFixture):

    def stream(self, **request):
        receiver = self.receiver()
        reply = self.call(op="speak", chunk_sock=receiver.path, **request)
        self.assertTrue(reply["ok"], reply)
        return receiver, reply

    def assert_one_terminal_last(self, receiver: Receiver, outcome: str) -> dict:
        self.assertTrue(wait_until(lambda: receiver.eof, 20), receiver.msgs)
        positions = [i for i, m in enumerate(receiver.msgs) if m.get("terminal") is True]
        self.assertEqual(positions, [len(receiver.msgs) - 1], receiver.msgs)
        self.assertTrue(all("sequence" in m for m in receiver.msgs[:-1]))
        last = receiver.msgs[-1]
        self.assertEqual(last["outcome"], outcome, last)
        return last

    def test_a_1_2_subscriber_gets_one_terminal_after_the_last_descriptor(self) -> None:
        receiver, reply = self.stream(text=TWO, v="1.2")
        last = self.assert_one_terminal_last(receiver, "completed")
        self.assertEqual([m["sequence"] for m in receiver.msgs[:-1]], [0, 1])
        self.assertEqual((last["job"], last["kind"], last["chunks"]),
                         (reply["turn"], "speech", 2))
        self.assertEqual(self.settled(reply["turn"])["outcome"], "completed")

    def test_a_cancelled_1_2_stream_ends_with_a_cancelled_terminal(self) -> None:
        self.tts_sizes = [BIG]
        self.flag("hold", True)
        receiver, reply = self.stream(text=FOUR, v="1.2")
        self.assertTrue(wait_until(lambda: receiver.msgs and self.sink_stalled(), 15))
        self.call(op="stop-speech")
        self.flag("hold", False)
        last = self.assert_one_terminal_last(receiver, "cancelled")
        self.assertEqual(last["code"], protocol.ERR_CANCELLED)
        self.assertEqual(self.settled(reply["turn"])["outcome"], "cancelled")

    def test_a_failed_1_2_stream_ends_with_a_failed_terminal(self) -> None:
        self.flag("fail", True)
        receiver, reply = self.stream(text="Only one sentence here.", v="1.2")
        last = self.assert_one_terminal_last(receiver, "failed")
        self.assertEqual(last["code"], protocol.ERR_UNAVAILABLE)

    def test_a_subscriber_that_declared_no_1_2_gets_no_terminal(self) -> None:
        for version in (None, "1", "1.1"):
            with self.subTest(v=version):
                fields = {} if version is None else {"v": version}
                receiver, reply = self.stream(text=TWO, **fields)
                self.assertTrue(wait_until(lambda: receiver.eof, 20))
                self.assertEqual([m.get("final") for m in receiver.msgs], [False, True])
                self.assertFalse(any("terminal" in m for m in receiver.msgs))
                self.settled(reply["turn"])
                self.assertTrue(wait_until(self.idle, 5))


class SettleOnce(unittest.TestCase):

    def test_settle_attempted_from_many_threads_records_one_outcome(self) -> None:
        ledger = jobs.JobLedger()
        turn = voiced._SpeechTurn("speak-1", ["a"], mock.Mock())
        turn.on_settle = ledger.record
        barrier = threading.Barrier(8)
        results = []

        def attempt(index: int) -> None:
            barrier.wait()
            results.append(turn.settle(jobs.JobOutcome(
                "speak-1", "speech", "failed", protocol.ERR_UNAVAILABLE,
                message=f"attempt {index}")))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertEqual(results.count(True), 1)
        self.assertEqual(ledger.get("speak-1")["error"], turn.outcome.message)

    def test_a_turn_without_a_ledger_still_settles_once(self) -> None:
        turn = voiced._SpeechTurn("speak-2", ["a"], mock.Mock())
        first = jobs.JobOutcome("speak-2", "speech", "completed", None)
        second = jobs.JobOutcome("speak-2", "speech", "cancelled", protocol.ERR_CANCELLED)
        self.assertTrue(turn.settle(first))
        self.assertFalse(turn.settle(second))
        self.assertIs(turn.outcome, first)


if __name__ == "__main__":
    unittest.main()
