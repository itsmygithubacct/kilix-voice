"""Every refusal the consent gate makes reaches the caller as `denied`.

R6 finding 6: when consent.json could not be read or did not verify,
consent.granted raised ConsentError -- a ValueError -- and the dictation
worker's broad arm coded it `unavailable`, the "no microphone" code R4 finding
3 existed to separate from consent. These drive the REAL gate through the real
worker against a private data directory; the microphone must stay unopened in
every case.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_consent_gate", importlib.machinery.SourceFileLoader(
        "kilix_voiced_consent_gate", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import consent, protocol  # noqa: E402


class BrokenConsentRecordTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"KILIX_DATA_HOME": self.tmp.name,
                                           "KILIX_VOICE_REQUIRE_CONSENT": "1"})
        env.start()
        self.addCleanup(env.stop)
        os.makedirs(os.path.dirname(consent.consent_path()), exist_ok=True)

    def write_record(self, text: str) -> str:
        path = consent.consent_path()
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def dictate(self):
        opened, sent = [], []
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent",
                          "max_seconds": 120}, "vad": {"silence_ms": 900}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        with mock.patch.object(voiced.audio, "MicCapture",
                               lambda cfg: opened.append("mic")):
            voiced.Daemon._run_dictation(
                d, voiced._DictationTurn("listen-1", mock.Mock()))
        self.assertEqual(opened, [], "the microphone was opened")
        self.assertEqual(len(sent), 1, sent)
        return sent[0]

    def test_a_record_with_the_wrong_schema_is_denied(self) -> None:
        self.write_record(json.dumps({"schema": "something/else", "grants": {}}))
        datagram = self.dictate()
        self.assertEqual(datagram["code"], protocol.ERR_DENIED, datagram)
        self.assertIn("grant consent again", datagram["error"])

    def test_a_grant_missing_its_model_revision_is_denied(self) -> None:
        self.write_record(json.dumps({"schema": consent.CONSENT_SCHEMA, "grants": {
            "dictation": {"digest": "0" * 64, "granted_utc": "2026-09-13T00:00:00+00:00",
                          "allowed_use": "x", "output_identity": "y",
                          "model_id": "small-en-us"}}}))
        datagram = self.dictate()
        self.assertEqual(datagram["code"], protocol.ERR_DENIED, datagram)
        self.assertIn("model_revision", datagram["error"])

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file anyway")
    def test_an_unreadable_record_is_denied(self) -> None:
        path = self.write_record(json.dumps({"schema": consent.CONSENT_SCHEMA}))
        os.chmod(path, 0)
        self.addCleanup(os.chmod, path, 0o600)
        datagram = self.dictate()
        self.assertEqual(datagram["code"], protocol.ERR_DENIED, datagram)
        self.assertIn("unreadable", datagram["error"])

    def test_no_grant_at_all_is_still_the_plain_refusal(self) -> None:   # control
        datagram = self.dictate()
        self.assertEqual(datagram["code"], protocol.ERR_DENIED, datagram)
        self.assertIn("no recorded consent", datagram["error"])


if __name__ == "__main__":
    unittest.main()
