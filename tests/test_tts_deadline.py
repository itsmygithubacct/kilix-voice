"""A probe or synthesis failure is a deadline if and only if the budget cut it.

R6 finding 1: the Piper probe attribution coded any provider failure under a
5 s budget as `deadline`, discarding "not installed, run X". The fix is typed:
TtsDeadlineExceeded is raised only where the kill fired under the caller's
budget, or before anything ran on a spent one. Every provider here is a
throwaway shell script reached through the trusted KILIX_PIPER_TTS override or
the tts.cmd seam; nothing needs Piper or espeak installed.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_tts_deadline", importlib.machinery.SourceFileLoader(
        "kilix_voiced_tts_deadline", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol, tts as tts_lib  # noqa: E402


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _script(directory, name, body):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\n" + body + "\n")
    os.chmod(path, stat.S_IRWXU)
    return path


class _ProviderFixture(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.marker = os.path.join(self.dir, "provider-ran")

    def speak(self, provider, deadline_ms):
        engine = tts_lib.PiperTts(rate=170)
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.dir
        d._cfg = {}
        d._refresh_config = lambda: None
        d._touch = lambda: None
        d._speech_chunks = lambda text: []
        with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}), \
             mock.patch.object(voiced.tts_lib, "make_tts", lambda *a, **k: engine):
            started = time.monotonic()
            reply = voiced.Daemon._dispatch(d, protocol.encode(
                {"op": "speak", "text": "hi", "deadline_ms": deadline_ms}))
            return reply, time.monotonic() - started


class PiperProbeAttributionTestCase(_ProviderFixture):

    def test_a_provider_that_fails_at_once_keeps_its_message(self) -> None:
        provider = _script(self.dir, "piper", "echo 'provider is broken' >&2; exit 3")
        reply, _elapsed = self.speak(provider, 3000)
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)
        self.assertIn("exited 3", reply["error"])

    def test_a_provider_hung_past_a_300_ms_budget_is_a_deadline(self) -> None:
        provider = _script(self.dir, "piper", "exec sleep 8")
        reply, elapsed = self.speak(provider, 300)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        self.assertLess(elapsed, 1.5)

    def test_the_probes_own_ceiling_firing_is_not_a_deadline(self) -> None:
        # The budget is a minute; the 0.3 s that expires is the probe's own.
        provider = _script(self.dir, "piper", "exec sleep 8")
        with mock.patch.object(tts_lib, "PIPER_STATUS_TIMEOUT_S", 0.3):
            reply, elapsed = self.speak(provider, 60_000)
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)
        self.assertIn("did not answer", reply["error"])
        self.assertLess(elapsed, 1.5)

    def test_a_genuine_failure_inspected_after_the_deadline_is_not_a_deadline(self) -> None:
        # Deterministic form of the over-attribution a clock re-read makes: the
        # provider exits 3 on its own, and by the time anyone looks the clock
        # is past the deadline. Nothing cut it, so it is not a deadline.
        clock = _Clock()

        def run(argv, **kwargs):
            clock.t += 1.0
            return subprocess.CompletedProcess(argv, 3, "", "provider is broken\n")

        with mock.patch.object(voiced.time, "monotonic", clock), \
             mock.patch.object(tts_lib, "piper_binary", lambda: "/fixed/piper"), \
             mock.patch.object(tts_lib.subprocess, "run", run):
            reply, _elapsed = self.speak("/fixed/piper", 300)
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)
        self.assertIn("exited 3", reply["error"])

    def test_a_spent_budget_starts_no_lookup_and_no_process(self) -> None:
        def no_lookup():
            raise AssertionError("the PATH lookup ran for a spent budget")

        with mock.patch.object(tts_lib, "piper_binary", no_lookup):
            with self.assertRaises(tts_lib.TtsDeadlineExceeded):
                tts_lib.piper_probe(budget=0.0)
        provider = _script(self.dir, "piper", f"touch '{self.marker}'; exit 99")
        with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}):
            with self.assertRaises(tts_lib.TtsDeadlineExceeded):
                tts_lib.piper_probe(budget=0.0)
        self.assertFalse(os.path.exists(self.marker))

    def test_piper_status_still_returns_a_tuple_for_a_budget_cut(self) -> None:
        provider = _script(self.dir, "piper", "exec sleep 8")
        with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}):
            available, detail = tts_lib.piper_status(budget=0.2)
        self.assertFalse(available)
        self.assertIn("deadline", detail)


class SynthesisAttributionTestCase(_ProviderFixture):

    def test_bounded_refuses_a_spent_budget_as_a_deadline(self) -> None:
        with self.assertRaises(tts_lib.TtsDeadlineExceeded):
            tts_lib._bounded(210.0, 0.0)

    def test_piper_synthesis_cut_by_the_budget_is_a_deadline(self) -> None:
        provider = _script(self.dir, "piper", "exec sleep 8")
        with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}):
            with self.assertRaises(tts_lib.TtsDeadlineExceeded):
                tts_lib.PiperTts(rate=170).synth("hello", budget=0.3)

    def test_piper_synthesis_cut_by_its_own_ceiling_is_not_a_deadline(self) -> None:
        provider = _script(self.dir, "piper", "exec sleep 8")
        with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}), \
             mock.patch.object(tts_lib, "PIPER_SYNTH_TIMEOUT_S", 0.3):
            with self.assertRaises(tts_lib.TtsError) as caught:
                tts_lib.PiperTts(rate=170).synth("hello", budget=60.0)
        self.assertNotIsInstance(caught.exception, tts_lib.TtsDeadlineExceeded)

    def test_espeak_synthesis_cut_by_the_budget_is_a_deadline(self) -> None:
        engine = tts_lib.EspeakTts(
            {"tts": {"cmd": [_script(self.dir, "espeak", "exec sleep 8")]}},
            voice="en-us", rate=170)
        with self.assertRaises(tts_lib.TtsDeadlineExceeded):
            engine.synth("hello", budget=0.3)

    def test_espeak_synthesis_cut_by_its_own_ceiling_is_not_a_deadline(self) -> None:
        engine = tts_lib.EspeakTts(
            {"tts": {"cmd": [_script(self.dir, "espeak", "exec sleep 8")]}},
            voice="en-us", rate=170)
        with mock.patch.object(tts_lib, "SYNTH_TIMEOUT_BASE_S", 0.3):
            with self.assertRaises(tts_lib.TtsError) as caught:
                engine.synth("hello", budget=60.0)
        self.assertNotIsInstance(caught.exception, tts_lib.TtsDeadlineExceeded)

    def test_a_budget_cut_does_not_mark_mbrola_broken(self) -> None:
        script = _script(
            self.dir, "espeak",
            'case "$1" in mb-*) exec sleep 8;; esac\n'
            f"touch '{self.marker}'")
        engine = tts_lib.EspeakTts({"tts": {"cmd": [script, "{voice}"]}},
                                   voice="us1", rate=170, mbrola=True)
        with self.assertRaises(tts_lib.TtsDeadlineExceeded):
            engine.synth("hello", budget=0.3)
        self.assertTrue(engine._mbrola_ok, "a deadline marked mbrola broken")
        self.assertFalse(os.path.exists(self.marker),
                         "a fallback process ran with no budget left")

    def test_a_synthesis_deadline_is_reported_as_the_deadline(self) -> None:
        reports = []
        engine = mock.Mock(voice="en-us", model="m", rate=170, effective_model=None)
        engine.synth.side_effect = tts_lib.TtsDeadlineExceeded(
            "espeak-ng did not finish within 0.3s for 5 characters of text")
        turn = voiced._SpeechTurn("speak-1", ["one"], engine)
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._speech = turn
        d._warn = lambda *a, **k: None
        d._touch = lambda: None
        d._arbiter = mock.Mock()
        d._get_player = lambda: mock.Mock(error="", playing=False)
        d._report_speech_failure = lambda t, message: reports.append(message)
        voiced.Daemon._run_speech(d, turn)
        self.assertEqual(reports,
                         ["read-aloud stopped: the request deadline elapsed."])


class StatusDeadlineTestCase(_ProviderFixture):
    """R6 finding 4: status honours the deadline_ms it accepts."""

    def status(self, deadline_ms=None, *, engine="piper", refresh=lambda: None,
               provider=None):
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.dir
        d._socket_path = os.path.join(self.dir, "control.sock")
        d._cfg = {}
        d._refresh_config = refresh
        d._touch = lambda: None
        d._lock = threading.RLock()
        d._started_at = d._last_activity = time.monotonic()
        d._idle_seconds = 300
        d._speech_error, d._speech_error_serial = "", 0
        d._arbiter = mock.Mock(speaking=False, listening=False)
        message = {"op": "status"}
        if deadline_ms is not None:
            message["deadline_ms"] = deadline_ms
        env = {tts_lib.PIPER_ENV_COMMAND: provider} if provider else {}
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(voiced.settings, "tts_engine",
                               lambda path=None: engine):
            started = time.monotonic()
            reply = voiced.Daemon._dispatch(d, protocol.encode(message))
            return d, reply, time.monotonic() - started

    def test_a_spent_status_budget_starts_no_provider(self) -> None:
        provider = _script(self.dir, "piper", f"touch '{self.marker}'; exit 99")
        _d, reply, _elapsed = self.status(
            1, provider=provider, refresh=lambda: time.sleep(0.005))
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        self.assertFalse(os.path.exists(self.marker))

    def test_a_hung_provider_cannot_hold_status_past_its_budget(self) -> None:
        provider = _script(self.dir, "piper", "exec sleep 8")
        _d, reply, elapsed = self.status(150, provider=provider)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        self.assertLess(elapsed, 1.0)

    def test_a_budget_cut_probe_ends_the_status_work_there(self) -> None:
        # The typed refusal is redundant in OUTCOME with the boundary check
        # below it -- a budget that cut the probe has necessarily elapsed, so
        # both say `deadline` -- but not in EFFECT. Reporting the cut as
        # `available: false` instead went on to inspect dictation, capture
        # and playback for a caller that had already given up: a mutation
        # doing exactly that survived every other test here.
        provider = _script(self.dir, "piper", "exec sleep 8")
        with mock.patch.object(voiced.Daemon, "_stt_status",
                               mock.Mock(return_value={})) as stt_status:
            _d, reply, _elapsed = self.status(150, provider=provider)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        stt_status.assert_not_called()

    def test_a_status_that_finishes_late_is_not_answered_ok(self) -> None:
        # The delivery boundary: nothing here spawns a process, the report is
        # simply slow, and the caller has given up by the time it is ready.
        with mock.patch.object(voiced.Daemon, "_stt_status",
                               lambda self: time.sleep(0.2) or {}):
            _d, reply, _elapsed = self.status(50, engine="off")
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)

    def test_a_status_without_a_deadline_is_unchanged(self) -> None:   # control
        provider = _script(self.dir, "piper", "echo broken >&2; exit 3")
        _d, reply, _elapsed = self.status(provider=provider)
        self.assertTrue(reply["ok"], reply)
        self.assertFalse(reply["status"]["tts"]["available"])
        self.assertIn("exited 3", reply["status"]["tts"]["detail"])

    def test_a_generous_budget_still_answers_ok(self) -> None:       # control
        provider = _script(self.dir, "piper", "echo broken >&2; exit 3")
        _d, reply, _elapsed = self.status(60_000, provider=provider)
        self.assertTrue(reply["ok"], reply)
        self.assertFalse(reply["status"]["tts"]["available"])


if __name__ == "__main__":
    unittest.main()
