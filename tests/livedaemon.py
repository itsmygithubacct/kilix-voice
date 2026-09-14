"""A real kilix-voiced process on a real control socket, for black-box tests.

Not a test module (the name does not start with ``test``): test modules
subclass LiveDaemonTestCase and add their own methods. It follows
tests/test_daemon.py's fixture -- an empty PATH directory, a temporary session
and data tree, no KILIX_* or GPU_TERMINAL_* variable inherited -- and lets a
subclass choose the shared settings the daemon reads.

Requests go over the wire as bytes rather than through voicelib.protocol, so
what is tested is the framing the daemon accepts, not a helper agreeing with
itself.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

DAEMON = pathlib.Path(__file__).resolve().parent.parent / "kilix-voiced"

STARTUP_TIMEOUT_S = 30.0
REPLY_TIMEOUT_S = 15.0
EXIT_TIMEOUT_S = 20.0
POLL_S = 0.05
IDLE_SECONDS = "120"
MAX_REPLY_BYTES = 1 << 20


@unittest.skipUnless(sys.platform.startswith("linux"),
                     "kilix-voiced needs SO_PEERCRED and AF_UNIX/SOCK_SEQPACKET")
class LiveDaemonTestCase(unittest.TestCase):
    """One daemon per test, in its own short temporary session directory."""

    SETTINGS = ("KILIX_VOICE_TTS_ENGINE=espeak\n"
                "KILIX_VOICE_STT_ENGINE=vosk\n"
                "KILIX_VOICE_STT_MODEL=small-en-us\n")

    def setUp(self) -> None:
        # Short: every socket path below must fit AF_UNIX's 108 bytes.
        self.root = tempfile.mkdtemp(prefix="kv-live-", dir="/tmp")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.nowhere = os.path.join(self.root, "empty-path")
        self.settings = os.path.join(self.root, "settings.conf")
        self.session_dir = os.path.join(self.root, "session", "voice")
        self.data_dir = os.path.join(self.root, "data", "voice")
        self.control = os.path.join(self.session_dir, "control.sock")
        os.mkdir(self.nowhere)
        pathlib.Path(self.settings).write_text(self.SETTINGS, encoding="utf-8")
        self.log = open(os.path.join(self.root, "daemon.log"), "w+b")
        self.addCleanup(self.log.close)
        self.daemon = subprocess.Popen(
            [sys.executable, "-B", str(DAEMON), "--idle-seconds", IDLE_SECONDS,
             "--verbose"],
            cwd=self.root, env=self._environment(), stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=self.log)
        self.addCleanup(self._stop_daemon)
        self._wait_until_serving()

    def _stop_daemon(self) -> None:
        if self.daemon.poll() is None:
            self.daemon.terminate()
            try:
                self.daemon.wait(timeout=EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.daemon.kill()
                self.daemon.wait()

    def _environment(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("KILIX_", "GPU_TERMINAL_"))}
        env.update(
            PATH=self.nowhere,
            HOME=self.root,
            PYTHONDONTWRITEBYTECODE="1",
            GPU_TERMINAL_HOME=os.path.join(self.root, "gpu_terminal"),
            GPU_TERMINAL_SETTINGS_FILE=self.settings,
            KILIX_SESSION_HOME=os.path.dirname(self.session_dir),
            KILIX_DATA_HOME=os.path.dirname(self.data_dir),
        )
        return env

    def log_tail(self) -> str:
        self.log.flush()
        position = self.log.tell()
        self.log.seek(0)
        try:
            return self.log.read().decode("utf-8", "replace").strip()[-4000:]
        finally:
            self.log.seek(position)

    def _wait_until_serving(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            code = self.daemon.poll()
            if code is not None:
                self.fail(f"kilix-voiced exited with status {code}:\n"
                          f"{self.log_tail()}")
            try:
                self.exchange(b'{"op":"status"}\n')
            except OSError:
                time.sleep(POLL_S)
                continue
            return
        self.fail(f"kilix-voiced did not answer within {STARTUP_TIMEOUT_S:.0f}s:"
                  f"\n{self.log_tail()}")

    def exchange(self, payload: bytes) -> bytes:
        """Send one record on its own connection; return the first reply record."""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(REPLY_TIMEOUT_S)
        with client:
            client.connect(self.control)
            client.send(payload)
            return client.recv(MAX_REPLY_BYTES)

    def request(self, message: dict) -> dict:
        """Send ``message`` as ASCII JSON (non-ASCII as \\u escapes); decode the reply."""
        try:
            raw = self.exchange(json.dumps(message).encode("ascii") + b"\n")
        except OSError as error:
            self.fail(f"cannot talk to kilix-voiced ({error}):\n{self.log_tail()}")
        if not raw:
            self.fail(f"the daemon hung up without a reply to {message!r}:\n"
                      f"{self.log_tail()}")
        self.assertLessEqual(len(raw), 65535, self.log_tail())
        return json.loads(raw.decode("utf-8"))

    def assert_still_serving(self) -> None:
        self.assertIsNone(self.daemon.poll(), self.log_tail())
        self.assertTrue(self.request({"op": "status"})["ok"])
