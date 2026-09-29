"""Consent binds the VibeVoice bytes that load, across a whole real turn.

Seat 1 (0.2.2 RC3) found that consent hashed the GGUF paths while asr_infer
opened them only when the turn ended, so a same-path swap during recording
loaded bytes nobody consented to. This drives the real daemon through the
consent gate, swaps the model after the microphone opens, and requires the
runtime to have loaded the consented bytes. The fake asr_infer prints the
SHA-256 of what it loaded as the transcript.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

from tests import test_daemon as daemon_tests
from voicelib import models, stt

FAKE_ASR = r"""#!{python}
import hashlib, sys
args = sys.argv[1:]
data = open(args[args.index("--lm-model") + 1], "rb").read()
print(hashlib.sha256(data).hexdigest())
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class VibeVoiceConsentBindingTests(unittest.TestCase):
    setUp = daemon_tests.DaemonTestCase.setUp
    tearDown = daemon_tests.DaemonTestCase.tearDown
    _stop_daemon = daemon_tests.DaemonTestCase._stop_daemon
    _log_tail = daemon_tests.DaemonTestCase._log_tail
    _wait_until_serving = daemon_tests.DaemonTestCase._wait_until_serving
    _send = daemon_tests.DaemonTestCase._send

    def _environment(self):
        env = daemon_tests.DaemonTestCase._environment(self)
        root = Path(self.root)
        self.model = Path(self.data_dir) / "models" / models.VIBEVOICE_MODEL
        self.model.mkdir(parents=True)
        (self.model / stt.VIBEASR_VAE).write_bytes(b"vae A")
        (self.model / stt.VIBEASR_LM).write_bytes(b"lm A")
        runtime = root / "asr_infer"
        runtime.write_text(FAKE_ASR.format(python=sys.executable))
        runtime.chmod(0o755)
        mic = root / "mic.py"
        # Speech-level frames, then silence long enough to end the turn.
        mic.write_text(
            "import sys,time\n"
            "for i in range(400):\n"
            " frame=(b'\\xff\\x3f\\x01\\xc0'*160) if 25 <= i < 60 else b'\\0'*640\n"
            " sys.stdout.buffer.write(frame);sys.stdout.buffer.flush();time.sleep(.02)\n")
        config = root / "config.json"
        config.write_text(json.dumps({
            "audio": {"capture_cmd": [sys.executable, str(mic)]},
            "stt": {"engine": "vibevoice", "max_seconds": 30},
            "vad": {"silence_ms": 300}}))
        env.update(KILIX_VOICE_CONFIG=str(config), KILIX_VOICE_VIBEASR=str(runtime))
        granted = subprocess.run(
            [sys.executable, str(Path(daemon_tests.DAEMON).parent / "kilix-stt"),
             "--grant-consent"], env=env, capture_output=True, text=True)
        assert granted.returncode == 0, granted.stderr
        return env

    def test_a_swap_after_the_microphone_opens_still_loads_the_consented_model(self):
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(str(Path(self.session_dir) / "mic.sock"))
        receiver.settimeout(20)
        self.addCleanup(receiver.close)
        owner = "a" * 32
        reply = self._send({"op": "owned-dictate", "owner": owner, "hold": False,
                            "silence_ms": 300, "sock": receiver.getsockname()})
        self.assertTrue(reply["ok"], reply)
        # "listening" is sent after the consent gate and the engine are done.
        self.assertEqual(json.loads(receiver.recv(65536)), {"listening": True})
        swap = self.model / (stt.VIBEASR_LM + ".new")
        swap.write_bytes(b"lm B")
        os.replace(swap, self.model / stt.VIBEASR_LM)
        final = None
        deadline = time.monotonic() + 20
        while final is None and time.monotonic() < deadline:
            self._send({"op": "owned-dictation-status", "owner": owner})
            message = json.loads(receiver.recv(65536))
            if "final" in message or "code" in message:
                final = message
        self.assertIsNotNone(final, self._log_tail())
        self.assertEqual(final.get("final"), sha(b"lm A"), final)



from tests.test_terminal_outcomes import _LiveTurnsFixture, voiced, wait_until
from unittest import mock


class ConsentDigestReachesTheEngineTests(_LiveTurnsFixture):
    """The digest the gate granted is the one the engine is built with."""

    def test_the_turn_carries_the_granted_payload_to_the_engine(self):
        self.grant_consent()
        seen = []
        factory = voiced.stt_lib.make_stt

        def recording(*args, **kwargs):
            seen.append(kwargs.get("resolved"))
            return factory(*args, **kwargs)

        with mock.patch.object(voiced.stt_lib, "make_stt", recording):
            receiver = self.receiver()
            reply = self.call(op="dictate", sock=receiver.path)
            self.assertTrue(reply["ok"], reply)
            self.assertTrue(wait_until(lambda: receiver.eof, 20))
        (resolved,) = seen
        expected = voiced.stt_lib.consent_identity(resolved).payload_digest
        self.assertIsNotNone(resolved.consented_payload)
        self.assertEqual(resolved.consented_payload, expected)

if __name__ == "__main__":
    unittest.main()
