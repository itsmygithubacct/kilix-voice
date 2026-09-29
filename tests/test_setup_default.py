"""kilix stt --setup-default and the Whisper install hand-offs, offline.

The setup is what Kilix 95's first-boot dictation offer runs in a terminal:
the sizer's default for this computer, its licence and install, microphone
consent, then the setting. Every external step is a stand-in here.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from tests.test_sizing import response
from tests.test_stt_tool import tool
from voicelib import consent, licensing, models, settings, sizing, stt

WHISPER = tool.MODEL_BY_ID[models.WHISPER_MODEL]
SMALL = tool.MODEL_BY_ID["small-en-us"]


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        environment = mock.patch.dict(os.environ, {
            "HOME": str(self.root),
            "GPU_TERMINAL_HOME": str(self.root),
            "GPU_TERMINAL_SETTINGS_FILE": str(self.root / "settings.conf"),
            "KILIX_DATA_HOME": str(self.root / "data"),
            "KILIX_STATE_DIRECTORY": str(self.root / "state"),
            "XDG_STATE_HOME": str(self.root / "xdg-state"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        for name in (stt.ENV_MODEL, "KILIX_VOICE_CONFIG"):
            os.environ.pop(name, None)
        self.calls: list[list[str]] = []
        launcher = mock.patch.object(tool, "kilix_launcher", return_value="/kilix")
        launcher.start()
        self.addCleanup(launcher.stop)

    def install_files(self, spec) -> None:
        target = pathlib.Path(tool.paths.model_dir(spec.catalog_id))
        for relative in models.REQUIRED_FILES[spec.engine]:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            (target / relative).write_bytes(b"fixture " + relative.encode())

    def fake_call(self, installs=None, status=0):
        """subprocess.call stand-in: records argv, 'installs' a model."""
        def call(argv):
            self.calls.append(list(argv))
            if installs is not None and argv[1:3] == ["models", "install"]:
                self.install_files(installs)
            return status
        return mock.patch.object(tool.subprocess, "call", side_effect=call)


class WhisperHandOffTests(Fixture):
    def test_whisper_installs_through_content_then_its_runtime(self) -> None:
        self.assertEqual(tool.model_install_argv(WHISPER),
                         ["/kilix", "models", "install", "faster-whisper-small-en"])
        self.assertEqual(tool.runtime_install_argv(WHISPER), ["/kilix", "voice", "whisper"])
        self.assertIsNone(tool.runtime_install_argv(SMALL))
        self.assertEqual(licensing.content_asset_id(WHISPER.catalog_id), "faster-whisper-small-en")

    def test_the_content_flow_is_the_licence_gate_for_whisper(self) -> None:
        # No receipt anywhere: the Vosk hand-off refuses, the content flow
        # (which shows the licence itself) is run.
        with mock.patch.object(licensing, "require_covering_receipt",
                               side_effect=licensing.LicenseRefused("small-en-us", "none")):
            with self.fake_call(installs=WHISPER):
                self.assertEqual(tool.install_model(WHISPER), 0)
            with self.assertRaises(licensing.LicenseRefused):
                tool.install_model(SMALL)
        self.assertEqual(self.calls, [["/kilix", "models", "install", "faster-whisper-small-en"],
                                      ["/kilix", "voice", "whisper"]])

    def test_installed_whisper_weights_are_not_fetched_again(self) -> None:
        self.install_files(WHISPER)
        with self.fake_call():
            self.assertEqual(tool.install_model(WHISPER), 0)
        self.assertEqual(self.calls, [["/kilix", "voice", "whisper"]])

    def test_a_failed_model_install_skips_the_runtime(self) -> None:
        with self.fake_call(status=5):
            self.assertEqual(tool.install_model(WHISPER), 5)
        self.assertEqual(len(self.calls), 1)

    def test_diagnosis_names_the_whisper_runtime(self) -> None:
        values = {key: settings.value(key) for key in settings.SPEC}
        values[settings.KEY_STT_ENGINE] = "whisper"
        values[settings.KEY_STT_MODEL] = WHISPER.catalog_id
        report = tool.diagnose(values)
        self.assertEqual(report.library, stt.whisper_binary())
        self.assertFalse(report.library_ok)
        self.assertIn("kilix stt --install whisper-small-en", report.engine_note)


class SetupDefaultTests(Fixture):
    def setup(self, answers, *, default=WHISPER.catalog_id, sizer_error=None,
              installs=WHISPER, status=0):
        result = self.root / "result.json"
        size = (mock.patch.object(tool, "_size_models", side_effect=sizer_error)
                if sizer_error else
                mock.patch.object(tool, "_size_models",
                                  return_value={"task": "stt", "defaults": {"stt": default}}))
        def receipt(catalog_id, **_options):
            # Only kilix-content's licence screen writes a receipt.
            if not any(argv[1:3] == ["models", "install"] for argv in self.calls):
                raise licensing.LicenseRefused(catalog_id, "none")
        pending = list(answers)
        def answer(_prompt):
            if not pending:
                raise EOFError
            return pending.pop(0)
        out, err = io.StringIO(), io.StringIO()
        with size, mock.patch.object(licensing, "require_covering_receipt", side_effect=receipt), \
                self.fake_call(installs=installs, status=status), \
                mock.patch("builtins.input", side_effect=answer), \
                redirect_stdout(out), redirect_stderr(err):
            code = tool.main(["--setup-default", "--result", str(result)])
        self.output = out.getvalue() + err.getvalue()
        return code, json.loads(result.read_text())

    def test_yes_to_both_installs_consents_and_sets_the_default(self) -> None:
        code, result = self.setup(["", "y"])
        self.assertEqual((code, result["status"], result["model"]), (0, "ready", WHISPER.catalog_id))
        self.assertEqual(self.calls, [["/kilix", "models", "install", "faster-whisper-small-en"],
                                      ["/kilix", "voice", "whisper"]])
        self.assertEqual((settings.stt_engine(), settings.stt_model()), ("whisper", WHISPER.catalog_id))
        identity = stt.consent_identity(stt.resolve_stt({}))
        self.assertTrue(consent.granted("dictation", identity.digest))
        self.assertIn("Ctrl+Shift+D", self.output)
        self.assertIn("third of the", self.output)

    def test_no_at_the_start_changes_nothing(self) -> None:
        code, result = self.setup(["n"])
        self.assertEqual((code, result["status"]), (1, "declined"))
        self.assertEqual(self.calls, [])
        self.assertEqual(settings.stt_engine(), "vosk")

    def test_no_to_the_microphone_leaves_the_default_and_no_consent(self) -> None:
        code, result = self.setup(["y", "n"])
        self.assertEqual((code, result["status"]), (1, "declined"))
        self.assertIn("microphone", result["detail"])
        self.assertEqual((settings.stt_engine(), settings.stt_model()), ("vosk", "small-en-us"))
        identity = stt.consent_identity(stt.resolve_stt({"stt": {"engine": "whisper"}}))
        self.assertFalse(consent.granted("dictation", identity.digest))

    def test_end_of_input_is_a_no(self) -> None:
        code, result = self.setup([])
        self.assertEqual((code, result["status"]), (1, "declined"))

    def test_a_declined_licence_is_declined(self) -> None:
        code, result = self.setup(["y"], installs=None, status=4)
        self.assertEqual((code, result["status"]), (1, "declined"))
        self.assertEqual(settings.stt_engine(), "vosk")

    def test_an_installer_that_leaves_no_model_has_failed(self) -> None:
        code, result = self.setup(["y"], installs=None)
        self.assertEqual((code, result["status"]), (2, "failed"))
        self.assertIn("is not at", result["detail"])

    def test_without_the_sizer_the_compact_model_is_offered(self) -> None:
        code, result = self.setup(["y", "y"], sizer_error=sizing.SizerError("absent"),
                                  installs=SMALL)
        self.assertEqual((code, result["model"]), (0, "small-en-us"))
        # The Vosk path needs a receipt, so its licence screen comes first.
        self.assertEqual(self.calls[0], ["/kilix", "models", "install", "vosk-model-small-en-us-0.15"])
        self.assertIn("could not be asked", self.output)
        self.assertEqual(settings.stt_engine(), "vosk")

    def test_a_sizer_with_no_default_offers_the_compact_model(self) -> None:
        self.assertEqual(tool.setup_default_model.__name__, "setup_default_model")
        with mock.patch.object(tool, "_size_models", return_value={"task": "stt", "defaults": {"stt": None}}):
            self.assertEqual(tool.setup_default_model()[0], "small-en-us")
        with mock.patch.object(tool, "_size_models", return_value={"task": "stt"}):
            self.assertEqual(tool.setup_default_model()[0], "small-en-us")

    def test_result_requires_setup_and_setup_is_standalone(self) -> None:
        for argv in (["--result", "x"], ["--setup-default", "--models"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    tool.main(argv)
                self.assertEqual(raised.exception.code, 2)


class SizerDefaultsTests(unittest.TestCase):
    def report(self, task="stt"):
        request = sizing.request_document(task, {})
        report = response(request, task)
        report["defaults"] = {"stt": models.WHISPER_MODEL} if task == "stt" else {}
        report["cpu"] = {"logical_cpus": 12, "flags": ["avx2"]}
        return request, report

    def test_whisper_is_in_the_request_and_a_fitting_default_is_accepted(self) -> None:
        request, report = self.report()
        self.assertIn(models.WHISPER_MODEL, [row["id"] for row in request["models"]])
        sizing.validate_report(report, request, "stt")
        self.assertEqual(sizing.default_model(report), models.WHISPER_MODEL)
        self.assertIn("Default for this hardware: whisper-small-en", sizing.summary(report))
        request, report = self.report("tts")
        sizing.validate_report(report, request, "tts")
        self.assertIsNone(sizing.default_model(report))

    def test_a_provider_without_defaults_names_none(self) -> None:
        request, report = self.report()
        del report["defaults"], report["cpu"]
        sizing.validate_report(report, request, "stt")
        self.assertIsNone(sizing.default_model(report))

    def test_invalid_defaults_and_cpu_are_rejected(self) -> None:
        request, baseline = self.report()
        mutations = [lambda r: r.update(defaults=None), lambda r: r.update(defaults={}),
                     lambda r: r.update(defaults={"stt": "outside"}),
                     lambda r: r.update(defaults={"stt": 3}),
                     lambda r: r.update(defaults={"stt": None, "tts": None}),
                     lambda r: r.update(cpu=[]), lambda r: r.update(cpu={"logical_cpus": 0, "flags": None}),
                     lambda r: r.update(cpu={"logical_cpus": True, "flags": None}),
                     lambda r: r.update(cpu={"logical_cpus": 4, "flags": "avx2"}),
                     lambda r: r.update(cpu={"logical_cpus": 4}),
                     lambda r: r["shortlists"]["stt"].remove(models.WHISPER_MODEL)]
        for index, mutation in enumerate(mutations):
            report = deepcopy(baseline)
            mutation(report)
            if index == len(mutations) - 1:
                # The default must still fit: take it off the shortlist too.
                for row in report["candidates"]:
                    if row["id"] == models.WHISPER_MODEL:
                        row.update(verdict="unknown", inference=None)
                report["provisional_candidates"]["stt"] = report["shortlists"]["stt"][0]
            with self.subTest(index=index), self.assertRaises(sizing.SizerError):
                sizing.validate_report(report, request, "stt")


if __name__ == "__main__":
    unittest.main()
