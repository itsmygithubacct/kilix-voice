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


def _private_environment(root: str, **store: str) -> dict[str, str]:
    """Return this process's environment with every store root inside ``root``.

    The store resolvers honour KILIX_DATA_HOME, KILIX_STORAGE_HOME,
    KILIX_SESSION_HOME and GPU_TERMINAL_HOME ahead of HOME, and a desktop
    session exports them. These tests once set HOME and KILIX_STORAGE_HOME only,
    so KILIX_DATA_HOME still named the user's real store, and the grants and
    fixture model files they record were written into it. Every stack variable
    is dropped and each root named explicitly, so this module stays safe even
    when it runs without the tests package's guard, as a script or under
    discover without -t.
    """
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("KILIX", "GPU_TERMINAL_", "PLEB_", "XDG_"))}
    env.update(HOME=root,
               GPU_TERMINAL_HOME=os.path.join(root, "gpu_terminal"),
               KILIX_STORAGE_HOME=os.path.join(root, "storage"),
               KILIX_DATA_HOME=os.path.join(root, "storage", "data"),
               KILIX_SESSION_HOME=os.path.join(root, "storage", "session"))
    env.update(store)
    return env


class BrokenConsentRecordTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, _private_environment(
            self.tmp.name, KILIX_DATA_HOME=self.tmp.name,
            KILIX_VOICE_REQUIRE_CONSENT="1"), clear=True)
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


def _load_tool(name, filename):
    spec = importlib.util.spec_from_loader(
        name, importlib.machinery.SourceFileLoader(name, os.path.join(ROOT, filename)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GrantMatchesTheGateTestCase(unittest.TestCase):
    """R6 finding 7: the grant command records what the gate will check.

    Under KILIX_VOICE_MODEL_PATH, or a daemon config file naming
    stt.model_path, kilix-stt --grant-consent hashed the catalogue directory
    and the gate hashed the directory the recogniser opens, so no grant could
    ever satisfy the gate. Each arm below runs the real grant, then the real
    gate against a configuration the daemon built for itself.
    """

    def setUp(self) -> None:
        import argparse
        from voicelib import daemon_config, models, settings, stt as stt_lib
        self.argparse, self.daemon_config = argparse, daemon_config
        self.models, self.stt_lib = models, stt_lib
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        settings_file = os.path.join(root, "settings")
        with open(settings_file, "w") as handle:
            handle.write(f"{settings.KEY_STT_ENGINE}=vosk\n")
        env = _private_environment(root, GPU_TERMINAL_SETTINGS_FILE=settings_file,
                                   KILIX_VOICE_REQUIRE_CONSENT="1")
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (stt_lib.ENV_MODEL, daemon_config.ENV_CONFIG):
            os.environ.pop(name, None)
        self.tool = _load_tool("kilix_stt_consent_gate", "kilix-stt")
        self.target = os.path.join(root, "my-model")
        self.install(self.target, b"the model the user has")

    def install(self, directory, body):
        for rel in self.models.REQUIRED_FILES["vosk"]:
            path = os.path.join(directory, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(body)

    def grant(self, **args):
        args.setdefault("revoke_consent", False)
        self.tool._consent_command(self.argparse.Namespace(**args))

    def daemon(self, config_path=None):
        """A daemon whose configuration comes from its own real builder."""
        d = object.__new__(voiced.Daemon)
        d._overrides = (self.daemon_config.load_overrides(config_path)
                        if config_path else {})
        d._cfg = {}
        voiced.Daemon._refresh_config(d)
        return d

    def gate(self, d):
        resolved = self.stt_lib.resolve_stt(d._cfg)
        self.assertEqual(resolved.model_dir, self.target,
                         "precondition: the daemon opens the override directory")
        voiced.Daemon._require_capture_consent(d, resolved)

    def config_file(self):
        path = os.path.join(self.tmp.name, "daemon.json")
        with open(path, "w") as handle:
            json.dump({"stt": {"model_path": self.target}}, handle)
        return path

    def test_a_grant_under_the_model_path_variable_satisfies_the_gate(self) -> None:
        os.environ[self.stt_lib.ENV_MODEL] = self.target
        self.grant()
        self.gate(self.daemon())

    def test_a_grant_under_the_daemon_config_variable_satisfies_the_gate(self) -> None:
        path = self.config_file()
        os.environ[self.daemon_config.ENV_CONFIG] = path
        self.grant()
        self.gate(self.daemon(path))

    def test_a_grant_given_the_config_argument_satisfies_the_gate(self) -> None:
        path = self.config_file()
        self.grant(config=path)
        self.gate(self.daemon(path))

    def test_a_granted_payload_that_changes_is_refused(self) -> None:   # S03
        os.environ[self.stt_lib.ENV_MODEL] = self.target
        self.grant()
        d = self.daemon()
        self.gate(d)
        self.install(self.target, b"the model the user has, but edited!")
        with self.assertRaises(voiced.ConsentDenied):
            self.gate(d)

    def test_the_refusal_names_the_directory_and_both_overrides(self) -> None:
        os.environ[self.stt_lib.ENV_MODEL] = self.target
        with self.assertRaises(voiced.ConsentDenied) as caught:
            self.gate(self.daemon())
        message = str(caught.exception)
        self.assertIn(self.target, message)
        self.assertIn(self.stt_lib.ENV_MODEL, message)
        self.assertIn(self.daemon_config.ENV_CONFIG, message)

    def test_revoking_needs_no_resolvable_configuration(self) -> None:
        os.environ[self.daemon_config.ENV_CONFIG] = os.path.join(
            self.tmp.name, "missing.json")
        self.grant(revoke_consent=True)                      # must not raise

    def test_an_unreadable_daemon_config_is_still_a_daemon_error(self) -> None:
        with self.assertRaises(voiced.DaemonError):
            voiced.Daemon(config_path=os.path.join(self.tmp.name, "missing.json"))


if __name__ == "__main__":
    unittest.main()
