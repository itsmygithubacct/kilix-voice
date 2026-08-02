"""kilix-voiced as a black box: a real process, a real socket, no audio.

The daemon is started in a temporary session with PATH pointing at an empty
directory — so there is no recorder, no sink and no synthesiser to find — and
with KILIX_DATA_HOME pointing at an empty tree, so there is no vosk library and
no model either.  That is the machine DESIGN.md cares about most: every one of
those is optional, and a voice session must still start, answer, and say what
is missing rather than refuse to run.

Nothing here plays or records anything.  The only ops exercised are the ones
that touch no device: ``status``, refusals, and a shutdown.  The dictation
tests are refusals that never reach the microphone, and the containment check
they exercise is the one that stops a request from redirecting everything the
microphone hears to a path of the caller's choosing.

Requests go over the wire as plain JSON rather than through
``voicelib.protocol``, so the framing the daemon actually accepts is what is
being tested rather than a shared helper agreeing with itself.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest

DAEMON = pathlib.Path(__file__).resolve().parent.parent / "kilix-voiced"

# Generous: a loaded machine may take a while to fork an interpreter, and a
# test that flakes on timing is worse than one that takes a second longer.
STARTUP_TIMEOUT_S = 30.0
REPLY_TIMEOUT_S = 15.0
EXIT_TIMEOUT_S = 20.0
POLL_S = 0.05

# Long enough that no test can race the idle exit, short enough that a daemon
# somehow orphaned by a killed test run goes away on its own.
IDLE_SECONDS = "120"

MAX_REPLY_BYTES = 1 << 20


@unittest.skipUnless(sys.platform.startswith("linux"),
                     "kilix-voiced needs SO_PEERCRED and AF_UNIX/SOCK_SEQPACKET")
class DaemonTestCase(unittest.TestCase):
    """One daemon per test, in its own session directory."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="kilix-voiced-test-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.nowhere = os.path.join(self.root, "empty-path")
        self.settings = os.path.join(self.root, "settings.conf")
        self.session_dir = os.path.join(self.root, "session", "voice")
        self.data_dir = os.path.join(self.root, "data", "voice")
        self.control = os.path.join(self.session_dir, "control.sock")
        os.mkdir(self.nowhere)
        # Written explicitly rather than relying on the defaults, so the status
        # assertions below describe one known configuration.
        pathlib.Path(self.settings).write_text(
            "KILIX_VOICE_TTS_ENGINE=espeak\n"
            "KILIX_VOICE_STT_ENGINE=vosk\n"
            "KILIX_VOICE_STT_MODEL=small-en-us\n", encoding="utf-8")
        self.log = open(os.path.join(self.root, "daemon.log"), "w+b")
        self.addCleanup(self.log.close)
        self.daemon = subprocess.Popen(
            [sys.executable, str(DAEMON), "--idle-seconds", IDLE_SECONDS,
             "--verbose"],
            cwd=self.root, env=self._environment(), stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=self.log)
        # unittest skips tearDown when setUp fails, so the process is also
        # registered here: no daemon may outlive its test either way.
        self.addCleanup(self._stop_daemon)
        self._wait_until_serving()

    def tearDown(self) -> None:
        self._stop_daemon()

    def _stop_daemon(self) -> None:
        """Stop the daemon if it is still running. Idempotent."""
        if self.daemon.poll() is None:
            self.daemon.terminate()
            try:
                self.daemon.wait(timeout=EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.daemon.kill()
                self.daemon.wait()

    # -- fixture ------------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        """Return a child environment with no audio and no Kilix state."""
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("KILIX_", "GPU_TERMINAL_"))}
        env.update(
            # shutil.which() falls back to a built-in system path when PATH is
            # empty, so "nothing installed" has to be an empty directory rather
            # than an empty string.
            PATH=self.nowhere,
            HOME=self.root,
            GPU_TERMINAL_HOME=os.path.join(self.root, "gpu_terminal"),
            GPU_TERMINAL_SETTINGS_FILE=self.settings,
            KILIX_SESSION_HOME=os.path.dirname(self.session_dir),
            KILIX_DATA_HOME=os.path.dirname(self.data_dir),
        )
        return env

    def _log_tail(self) -> str:
        """Return the daemon's stderr, for a failure message worth reading."""
        self.log.flush()
        position = self.log.tell()
        self.log.seek(0)
        try:
            return self.log.read().decode("utf-8", "replace").strip()
        finally:
            self.log.seek(position)

    def _wait_until_serving(self) -> None:
        """Block until the control socket answers, or fail with the log."""
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            code = self.daemon.poll()
            if code is not None:
                self.fail(f"kilix-voiced exited with status {code} before it "
                          f"served anything:\n{self._log_tail()}")
            try:
                self._send({"op": "status"})
            except OSError:
                time.sleep(POLL_S)
                continue
            return
        self.fail(f"kilix-voiced did not answer on {self.control} within "
                  f"{STARTUP_TIMEOUT_S:.0f}s:\n{self._log_tail()}")

    def _send(self, message: dict) -> dict:
        """Send one request on its own connection and return the reply."""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(REPLY_TIMEOUT_S)
        with client:
            client.connect(self.control)
            client.send(json.dumps(message).encode("utf-8") + b"\n")
            raw = client.recv(MAX_REPLY_BYTES)
        if not raw:
            self.fail(f"the daemon hung up without a reply to {message!r}:\n"
                      f"{self._log_tail()}")
        return json.loads(raw.decode("utf-8"))

    def request(self, message: dict) -> dict:
        """Send a request, failing the test with the log if the socket dies."""
        try:
            return self._send(message)
        except OSError as error:
            self.fail(f"cannot talk to kilix-voiced on {self.control} "
                      f"({error}):\n{self._log_tail()}")

    # -- status -------------------------------------------------------------

    def test_status_answers_with_a_session_description(self) -> None:
        reply = self.request({"op": "status", "id": "42"})
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(reply["id"], "42")
        status = reply["status"]
        self.assertEqual(status["pid"], self.daemon.pid)
        self.assertEqual(status["session"], self.session_dir)
        self.assertEqual(status["socket"], self.control)
        self.assertFalse(status["speaking"])
        self.assertFalse(status["listening"])
        self.assertEqual(status["speech_error"], "")
        self.assertEqual(status["speech_error_serial"], 0)

    def test_status_reports_every_missing_dependency(self) -> None:
        status = self.request({"op": "status"})["status"]
        for part in ("tts", "stt", "capture", "playback"):
            with self.subTest(part=part):
                self.assertFalse(status[part]["available"],
                                 f"{part} claims to be available with nothing "
                                 "installed")
                # "Unavailable" is only useful if it says what to install.
                self.assertTrue(status[part]["detail"].strip())
        self.assertIn("espeak", status["tts"]["detail"])
        self.assertIn("parec", status["capture"]["detail"])
        self.assertIn("pacat", status["playback"]["detail"])

    def test_async_playback_failure_remains_visible_in_status(self) -> None:
        """A detached daemon must not hide a sink failure on stderr only."""
        synthesiser = pathlib.Path(self.nowhere) / "espeak-ng"
        synthesiser.write_text(
            f"#!{sys.executable}\n"
            "import io, sys, wave\n"
            "output = io.BytesIO()\n"
            "with wave.open(output, 'wb') as wav:\n"
            "    wav.setnchannels(1)\n"
            "    wav.setsampwidth(2)\n"
            "    wav.setframerate(22050)\n"
            "    wav.writeframes(b'\\x01\\x00' * 4000)\n"
            "sys.stdout.buffer.write(output.getvalue())\n",
            encoding="utf-8",
        )
        synthesiser.chmod(0o755)
        sink = pathlib.Path(self.nowhere) / "pacat"
        sink.write_text(f"#!{sys.executable}\nraise SystemExit(23)\n",
                        encoding="utf-8")
        sink.chmod(0o755)

        accepted = self.request({"op": "speak", "text": "test phrase"})
        self.assertTrue(accepted["ok"], accepted)

        deadline = time.monotonic() + REPLY_TIMEOUT_S
        status = {}
        while time.monotonic() < deadline:
            status = self.request({"op": "status"})["status"]
            if status["speech_error_serial"]:
                break
            time.sleep(POLL_S)
        self.assertEqual(status["speech_error_serial"], 1, status)
        self.assertIn("playback command", status["speech_error"])
        self.assertIn("status 23", status["speech_error"])
        self.assertFalse(status["speaking"], status)

        # A later request clears the stale message before its worker runs; an
        # old failure must not be reported as the result of a new click.
        accepted = self.request({"op": "speak", "text": "another phrase"})
        self.assertTrue(accepted["ok"], accepted)
        status = self.request({"op": "status"})["status"]
        self.assertEqual(status["speech_error"], "", status)
        self.assertEqual(status["speech_error_serial"], 1, status)

    def test_status_names_the_missing_library_and_model(self) -> None:
        status = self.request({"op": "status"})["status"]
        library = os.path.join(self.data_dir, "lib", "current", "libvosk.so")
        model = os.path.join(self.data_dir, "models", "small-en-us")
        self.assertEqual(status["stt"]["engine"], "vosk")
        self.assertEqual(status["stt"]["library"], library)
        self.assertEqual(status["stt"]["model_path"], model)
        self.assertIn(library, status["stt"]["detail"])
        self.assertIn(model, status["stt"]["detail"])

    # -- privacy of the session ---------------------------------------------

    def test_the_session_directory_and_socket_are_private(self) -> None:
        session = os.stat(self.session_dir)
        control = os.stat(self.control)
        self.assertTrue(stat.S_ISSOCK(control.st_mode))
        self.assertEqual(stat.S_IMODE(session.st_mode), 0o700,
                         "the session directory carries dictation sockets and "
                         "must be enterable by nobody else")
        self.assertEqual(stat.S_IMODE(control.st_mode), 0o600)

    # -- dictation containment ----------------------------------------------

    def _listener(self, path: str) -> socket.socket:
        """Bind a receiver the daemon must never be talked into using."""
        endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(endpoint.close)
        endpoint.bind(path)
        endpoint.listen(1)
        endpoint.settimeout(0.5)
        return endpoint

    def assert_refused_outside_the_session(self, sock_path: str) -> None:
        """A dictate naming ``sock_path`` is refused, and nothing connects."""
        listener = self._listener(os.path.join(self.root, "outside.sock"))
        reply = self.request({"op": "dictate", "sock": sock_path})
        self.assertFalse(reply["ok"], reply)
        self.assertIn("session directory", reply["error"])
        with self.assertRaises(socket.timeout):
            listener.accept()
        self.assertFalse(self.request({"op": "status"})["status"]["listening"])

    def test_dictate_to_a_socket_outside_the_session_is_refused(self) -> None:
        self.assert_refused_outside_the_session(
            os.path.join(self.root, "outside.sock"))

    def test_dictate_through_a_symlink_out_of_the_session_is_refused(self) -> None:
        # The link lives inside the session directory, so only resolving it
        # before the containment test catches this one.
        link = os.path.join(self.session_dir, "dictate-9.sock")
        os.symlink(os.path.join(self.root, "outside.sock"), link)
        self.assert_refused_outside_the_session(link)

    def test_dictate_without_a_socket_is_refused(self) -> None:
        reply = self.request({"op": "dictate"})
        self.assertFalse(reply["ok"], reply)
        self.assertIn("sock", reply["error"])

    # -- degrading rather than dying ----------------------------------------

    def test_speak_without_a_synthesiser_is_answered_not_fatal(self) -> None:
        reply = self.request({"op": "speak", "text": "hello there"})
        self.assertFalse(reply["ok"], reply)
        self.assertIn("espeak", reply["error"])
        self.assertTrue(self.request({"op": "status"})["ok"],
                        "a failed speak must not take the daemon down")

    def test_an_unknown_op_is_refused_and_the_daemon_survives(self) -> None:
        reply = self.request({"op": "recite-poetry"})
        self.assertFalse(reply["ok"], reply)
        self.assertIn("status", reply["error"])   # lists what it does accept
        self.assertTrue(self.request({"op": "status"})["ok"])

    # -- shutdown -----------------------------------------------------------

    def test_sigterm_removes_the_control_socket(self) -> None:
        self.daemon.send_signal(signal.SIGTERM)
        self.assertEqual(self.daemon.wait(timeout=EXIT_TIMEOUT_S), 0,
                         self._log_tail())
        self.assertFalse(os.path.lexists(self.control),
                         "a killed daemon's socket would make the next one "
                         "look like a session that is already served")
        self.assertFalse(os.path.lexists(
            os.path.join(self.session_dir, "voiced.lock")))


if __name__ == "__main__":
    unittest.main()
