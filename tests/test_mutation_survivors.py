"""Tests for the ten mutants that survived the wave-1 verification battery.

Each class names its survivor. The mutant is a real change to the behaviour
the class describes, and the class asserts that behaviour's effect. None was
genuinely equivalent: three looked unreachable or redundant from the wire, but
each changes what a caller can observe on some path, and that path is the one
the test drives.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_survivors", importlib.machinery.SourceFileLoader(
        "kilix_voiced_survivors", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import (  # noqa: E402
    audio, cancel, daemon_config, models, protocol, stt as stt_lib,
    tts as tts_lib)


def _script(folder: str, name: str, body: str) -> str:
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\n" + body + "\n")
    os.chmod(path, 0o755)
    return path


class SttFamilyCodeTestCase(unittest.TestCase):
    """Survivor: SttError.code changed to `internal`.

    The existing agreement test only checks that the worker and the control
    path give the same code as each other, so a family-wide change moved both
    together and passed. This pins the code itself, on a real recogniser
    failure.
    """

    def test_a_missing_recogniser_library_reaches_the_caller_as_unavailable(self) -> None:
        sent = []
        with tempfile.TemporaryDirectory() as model_dir:
            d = object.__new__(voiced.Daemon)
            d._cfg = {"stt": {"engine": "vosk", "model_path": model_dir,
                              "lib_path": os.path.join(model_dir, "no-libvosk.so"),
                              "max_seconds": 1}}
            d._stopping = threading.Event()
            d._warn = d._debug = lambda *a, **k: None
            d._send = lambda receiver, msg: sent.append(msg) or True
            d._clear_dictation = lambda turn: None
            d._touch = lambda: None
            d._require_capture_consent = lambda resolved=None: None
            with mock.patch.object(voiced.audio, "MicCapture",
                                   lambda cfg: mock.Mock(rate=16000)):
                voiced.Daemon._run_dictation(
                    d, voiced._DictationTurn("listen-1", mock.Mock()))
        self.assertEqual(len(sent), 1, sent)
        self.assertIn("libvosk", sent[0]["error"])       # the real SttError
        self.assertEqual(sent[0]["code"], protocol.ERR_UNAVAILABLE, sent)

    def test_a_recogniser_failure_refused_by_a_handler_is_unavailable(self) -> None:
        d = object.__new__(voiced.Daemon)
        d._session_dir = "/tmp"
        d._touch = lambda: None
        d._op_status = mock.Mock(side_effect=stt_lib.SttError("no recogniser"))
        reply = voiced.Daemon._dispatch(d, protocol.encode({"op": "status"}))
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)


class PiperRateRefusalTestCase(unittest.TestCase):
    """Survivor: PiperTts's rate refusal demoted from TtsUnsupported to TtsError.

    The wire and the settings file both restrict rates to Piper's five today,
    so the daemon never reached this refusal. It is the engine's own guard for
    when the two vocabularies drift apart, and a caller reaching it asked for
    something the engine cannot do, which is `unsupported`, not a missing
    provider.
    """

    def test_the_engine_refuses_a_rate_it_cannot_use_as_unsupported(self) -> None:
        with self.assertRaises(tts_lib.TtsUnsupported) as caught:
            tts_lib.PiperTts(rate=171)
        self.assertEqual(caught.exception.code, protocol.ERR_UNSUPPORTED)

    def test_a_wider_wire_vocabulary_still_answers_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as session:
            d = object.__new__(voiced.Daemon)
            d._session_dir = os.path.realpath(session)
            d._cfg = {}
            d._refresh_config = lambda: None
            d._touch = lambda: None
            with mock.patch.object(protocol, "TTS_RATE_CHOICES",
                                   protocol.TTS_RATE_CHOICES + (171,)):
                reply = voiced.Daemon._dispatch(d, protocol.encode(
                    {"op": "speak", "text": "hello",
                     "model": models.PIPER_KRISTIN_MODEL, "rate": 171}))
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_UNSUPPORTED, reply)


class BudgetCutBoundaryTestCase(unittest.TestCase):
    """Survivor: _budget_cut's `budget <= cap` loosened to `budget < cap`.

    With a budget exactly equal to the probe's own ceiling, both bounds expire
    at once, and the caller's budget is spent when the probe is killed. That
    is a deadline. Under the mutant it read as the provider not answering.
    """

    def probe(self, budget: float):
        with tempfile.TemporaryDirectory() as tmp:
            provider = _script(tmp, "piper", "exec sleep 5")
            with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}), \
                 mock.patch.object(tts_lib, "PIPER_STATUS_TIMEOUT_S", 0.3):
                return tts_lib.piper_probe(budget=budget)

    def test_a_budget_equal_to_the_ceiling_that_runs_out_is_a_deadline(self) -> None:
        with self.assertRaises(tts_lib.TtsDeadlineExceeded):
            self.probe(0.3)

    def test_a_budget_above_the_ceiling_is_the_probes_own_timeout(self) -> None:  # control
        available, detail = self.probe(0.6)
        self.assertFalse(available)
        self.assertIn("did not answer within 0.3 s", detail)


class CleanEnvRunnerTestCase(unittest.TestCase):
    """Survivor: tests/cleanenv.sh exiting 0 whatever its command did.

    This runner is the gate every commit passes through, and nothing tested
    it: a runner that always exits 0 reports every suite green.
    """

    RUNNER = os.path.join(ROOT, "tests", "cleanenv.sh")

    def run_runner(self, *command: str, env: dict | None = None):
        return subprocess.run([self.RUNNER, *command], capture_output=True,
                              text=True, timeout=60, env=env,
                              stdin=subprocess.DEVNULL)

    def test_the_commands_exit_status_is_the_runners(self) -> None:
        for status in (0, 1, 7):
            with self.subTest(status=status):
                self.assertEqual(
                    self.run_runner("/bin/sh", "-c", f"exit {status}").returncode,
                    status)

    def test_the_command_runs_scrubbed_and_its_tree_is_removed(self) -> None:
        env = dict(os.environ, KILIX_LEAK_PROBE="1", GPU_TERMINAL_LEAK_PROBE="1",
                   PATH="/usr/bin:/bin")
        proc = self.run_runner("/bin/sh", "-c", 'printf "%s\\n" "$HOME"; env',
                               env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        home = proc.stdout.splitlines()[0]
        self.assertNotIn("LEAK_PROBE", proc.stdout)
        self.assertIn("PATH=/usr/bin:/bin", proc.stdout.splitlines())
        self.assertTrue(home.endswith("/home"), home)
        self.assertFalse(os.path.exists(os.path.dirname(home)),
                         "the runner left its temporary tree behind")


class PathEchoBoundTestCase(unittest.TestCase):
    """Survivor: _PATH_ECHO_CHARS raised from 512 to 1 MiB.

    PATH_MAX and the 4096-character prose cap still bound the reply, which is
    why the mutant looked redundant. But the cap then cuts the refusal short,
    and what it cuts is the end: the explanation of where dictation sockets
    come from. The bound on each quoted path is what keeps that remedy in.
    """

    def test_a_long_outside_path_is_refused_with_its_remedy_intact(self) -> None:
        with tempfile.TemporaryDirectory() as session:
            raw = "/" + "a" * 4000                  # absolute, under PATH_MAX
            with self.assertRaises(protocol.ProtocolError) as caught:
                protocol.validate_request({"op": "dictate", "sock": raw},
                                          os.path.realpath(session))
        message = str(caught.exception)
        self.assertLess(len(message), 2048)
        reply = protocol.reply_error(message, protocol.ERR_MALFORMED)
        self.assertNotIn("[truncated]", reply["error"])
        self.assertIn("created by the kitty fork", reply["error"])


class SessionDirectoryResolutionTestCase(unittest.TestCase):
    """Survivor: _validated_socket's `except (ValueError, OSError)` narrowed.

    Reachable: realpath of a relative session directory calls getcwd, which
    raises FileNotFoundError once the process's working directory has been
    removed. Under the mutant that escaped validate_request as a
    non-ProtocolError, which the daemon codes `internal`.
    """

    def test_an_unresolvable_session_directory_is_a_protocol_error(self) -> None:
        previous = os.getcwd()
        gone = tempfile.mkdtemp(prefix="kv-gone-")
        os.chdir(gone)
        os.rmdir(gone)
        try:
            with self.assertRaises(protocol.ProtocolError) as caught:
                protocol.validate_request(
                    {"op": "dictate", "sock": "/tmp/kv-none/dictate-1.sock"},
                    "relative-session")
        finally:
            os.chdir(previous)
        self.assertIn("session directory", str(caught.exception))


class OverSizeStrFrameTestCase(unittest.TestCase):
    """Survivor: decode's character-count pre-check loosened eightfold.

    Allocation stays bounded without it, which is why it looked redundant.
    But the size refusal then no longer precedes interpretation. An
    over-size str frame holding a lone surrogate came back as a malformed
    ProtocolError instead of MessageTooLarge, so it was coded malformed, not
    too-large.
    """

    def test_an_over_size_str_frame_is_too_large_whatever_it_holds(self) -> None:
        for tail in ("x", "é", "\ud800"):
            with self.subTest(tail=ascii(tail)):
                with self.assertRaises(protocol.MessageTooLarge) as caught:
                    protocol.decode("x" * protocol.MAX_REQUEST_BYTES + tail)
                self.assertEqual(caught.exception.code, protocol.ERR_TOO_LARGE)


class CancellationClearTestCase(unittest.TestCase):
    """Survivor: Cancellation.clear() leaving the event set.

    The token is documented as threading.Event-compatible, and clear() is
    part of that surface whether or not the daemon calls it today. A clear
    that forgot only the timestamp left a token that says "stopped" with no
    time for the stop.
    """

    def test_clear_withdraws_the_stop(self) -> None:
        token = cancel.Cancellation()
        token.set()
        token.clear()
        self.assertFalse(token.is_set())
        self.assertFalse(token.done())
        self.assertFalse(token.wait(0))
        self.assertFalse(token.wait_done(0.01))
        self.assertIsNone(token.reason())
        token.check()                                   # must not raise


class ConfigMergeDepthTestCase(unittest.TestCase):
    """Survivor: daemon_config.merge made shallow.

    Grant and gate share effective_config, so they agreed either way, and
    that agreement was all anything tested. A config file that set one key
    of a section discarded the rest of it, e.g. dictation's engine and
    ceiling under a model_path override.
    """

    def test_one_key_of_a_section_keeps_its_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = os.path.join(tmp, "settings.conf")
            with open(settings_file, "w", encoding="utf-8") as handle:
                handle.write("KILIX_VOICE_STT_MAX_SECONDS=60\n")
            config = os.path.join(tmp, "voiced.json")
            with open(config, "w", encoding="utf-8") as handle:
                handle.write('{"stt": {"model_path": "/models/mine"},'
                             ' "audio": {"play_cmd": ["cat"]}}')
            with mock.patch.dict(os.environ,
                                 {"GPU_TERMINAL_SETTINGS_FILE": settings_file}):
                cfg = daemon_config.effective_config(
                    daemon_config.load_overrides(config))
        self.assertEqual(cfg["stt"]["model_path"], "/models/mine")
        self.assertEqual(cfg["stt"]["max_seconds"], 60)
        self.assertIn("engine", cfg["stt"])
        self.assertEqual(cfg["audio"]["play_cmd"], ["cat"])
        self.assertEqual(cfg["audio"]["rate"], audio.DEFAULT_RATE)


class SttToolConfigErrorTestCase(unittest.TestCase):
    """Survivor: kilix-stt main's except tuple without ConfigError.

    A KILIX_VOICE_CONFIG that cannot be read then ended the grant command
    with a Python traceback instead of the one line saying what to fix.
    """

    def test_an_unreadable_daemon_config_is_one_line_and_exit_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "LANG": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1",
                   "GPU_TERMINAL_SETTINGS_FILE": os.path.join(tmp, "settings.conf"),
                   "KILIX_DATA_HOME": os.path.join(tmp, "data"),
                   "KILIX_VOICE_CONFIG": os.path.join(tmp, "missing", "voiced.json")}
            proc = subprocess.run(
                [sys.executable, "-B", os.path.join(ROOT, "kilix-stt"),
                 "--grant-consent"], env=env, capture_output=True, text=True,
                timeout=60, stdin=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(proc.stderr.startswith("kilix-stt: cannot read the daemon config"),
                        proc.stderr)
        self.assertEqual(len(proc.stderr.strip().splitlines()), 1, proc.stderr)


if __name__ == "__main__":
    unittest.main()
