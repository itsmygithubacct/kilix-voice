"""Provider delegation, hostile responses and read-only voice actions."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from voicelib import sizing, settings
from tests.test_stt_tool import load_stt_tool
from tests.test_tts_tool import load_tool


def response(request, task):
    rows = []
    for model in request["models"]:
        fits = model["runtime_supported"]
        rows.append({**model, "verdict": "estimated-fit" if fits else "unsupported",
                     "qualification_eligible": False, "installation": {"verdict": "unknown"},
                     "inference": {"verdict": "estimated-fit", "resources": {
                         "ram": {"required_bytes": 1024, "budget_bytes": 2048, "status": "estimated-fit"}}} if fits else None})
    ids = [row["id"] for row in rows if row["verdict"] == "estimated-fit"]
    return {"schema": sizing.RESPONSE_SCHEMA, "task": task, "resource_source": "live",
            "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "selected_model": None, "qualification_eligible": False, "candidates": rows,
            "shortlists": {task: ids}, "provisional_candidates": {task: ids[0] if ids else None}}


class SizerClientTests(unittest.TestCase):
    def test_request_uses_shared_catalog_and_relocated_data_path(self):
        request = sizing.request_document("stt", {"small-en-us": True})
        expected = response(request, "stt")
        with mock.patch.object(sizing, "provider_executable", return_value="/provider"), \
             mock.patch.object(sizing, "_run", return_value=json.dumps(expected).encode()) as run, \
             mock.patch.dict(os.environ, {"KILIX_DATA_HOME": "/test-data"}):
            self.assertEqual(sizing.recommend("stt", {"small-en-us": True}), expected)
        argv, body = run.call_args.args
        self.assertEqual(argv, ["/provider", "recommend", "voice", "--task", "stt", "--catalog", "-",
                                "--data-root", "/test-data/voice", "--json"])
        self.assertEqual(json.loads(body), request)
        self.assertFalse(next(row for row in request["models"] if row["id"] == "vibevoice-asr-bitnet")["runtime_supported"])

    def test_wrong_schema_binding_model_and_promotion_are_rejected(self):
        request = sizing.request_document("stt", {})
        baseline = response(request, "stt")
        mutations = [lambda r: r.update(schema="future"), lambda r: r.update(request_sha256="0" * 64),
                     lambda r: r.update(qualification_eligible=True), lambda r: r.update(selected_model="small-en-us"),
                     lambda r: r["candidates"][0].update(id="outside"), lambda r: r["candidates"].pop(),
                     lambda r: r["shortlists"]["stt"].append("vibevoice-asr-bitnet"),
                     lambda r: r["candidates"][0].update(inference=None),
                     lambda r: r["candidates"][0].pop("inference"),
                     lambda r: r["candidates"][0]["inference"]["resources"]["ram"].update(budget_bytes=0)]
        for mutation in mutations:
            report = deepcopy(baseline); mutation(report)
            with self.assertRaises(sizing.SizerError): sizing.validate_report(report, request, "stt")

    def test_empty_shortlist_is_valid_and_unknown_is_not_a_default(self):
        request = sizing.request_document("tts", {})
        report = response(request, "tts")
        for row in report["candidates"]:
            row.update(verdict="unknown", inference=None)
        report["shortlists"] = {"tts": []}; report["provisional_candidates"] = {"tts": None}
        sizing.validate_report(report, request, "tts")
        self.assertIn("No candidate", sizing.summary(report))

    def test_strict_json_and_explicit_missing_executable(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'\xff'):
            with self.assertRaises(sizing.SizerError): sizing._parse(raw)
        with mock.patch.dict(os.environ, {"PLEBIAN_MODEL_SIZER": "/nonexistent/sizer"}):
            self.assertEqual(sizing.provider_executable(), "/nonexistent/sizer")
            with self.assertRaises(sizing.SizerError): sizing.recommend("tts", {})

    def test_subprocess_output_timeout_and_exit_are_bounded(self):
        programs = ["import sys; sys.stdout.write('x'*20000)", "import time; time.sleep(10)", "raise SystemExit(3)"]
        for program in programs:
            with self.subTest(program=program), mock.patch.object(sizing, "TIMEOUT_SECONDS", 0.1), \
                 mock.patch.object(sizing, "MAX_REPLY_BYTES", 10000), self.assertRaises(sizing.SizerError):
                sizing._run([sys.executable, "-c", program], b"{}")


class VoiceSizingActionsTests(unittest.TestCase):
    def setUp(self):
        self.stt = load_stt_tool()
        self.tts = load_tool()

    def test_cli_recommendations_never_save_install_or_speak(self):
        for task, tool in (("stt", self.stt), ("tts", self.tts)):
            report = response(sizing.request_document(task, {}), task)
            output = StringIO()
            with mock.patch.object(tool, "_size_models", return_value=report), \
                 mock.patch.object(settings, "update") as save, redirect_stdout(output):
                self.assertEqual(tool.main(["--recommend", "--json"]), 0)
            save.assert_not_called()
            self.assertEqual(json.loads(output.getvalue()), report)

    def test_sizing_cannot_be_combined_with_mutations(self):
        for tool, extra in ((self.stt, ["--default", "small-en-us"]), (self.stt, ["--install", "small-en-us"]),
                            (self.tts, ["--speak", "hello"]), (self.tts, ["--set", "engine=piper"])):
            with redirect_stderr(StringIO()), mock.patch.object(tool, "_size_models") as size, self.assertRaises(SystemExit) as error:
                tool.main(["--recommend", *extra])
            self.assertEqual(error.exception.code, 2)
            size.assert_not_called()

    def test_missing_provider_has_nonzero_exit_and_no_heuristic_fallback(self):
        for tool in (self.stt, self.tts):
            output = StringIO()
            with mock.patch.object(tool, "_size_models", side_effect=sizing.SizerError("unavailable")), \
                 redirect_stdout(output), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                tool.main(["--recommend", "--json"])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.getvalue(), "")

    def test_dictation_tui_sizing_does_not_select_or_write_settings(self):
        tool = self.stt
        ui = tool.Ui.__new__(tool.Ui)
        ui._section = tool.SECTION_MODELS
        ui._selected = [0] * len(tool.SECTIONS)
        ui._discard_armed = False
        ui._sizing_report = None
        report = response(sizing.request_document("stt", {}), "stt")
        with mock.patch.object(tool, "_size_models", return_value=report), \
             mock.patch.object(settings, "update") as save, mock.patch.object(ui, "_use_model") as choose:
            ui._handle(ord("n"))
        self.assertEqual(ui._sizing_report, report)
        self.assertIn("Smallest resource candidate", ui._message)
        save.assert_not_called(); choose.assert_not_called()

    def test_read_aloud_tui_requests_sizing_without_changing_values(self):
        tool = self.tts
        screen = mock.Mock(); screen.getch.side_effect = [ord("n"), ord("q")]
        report = response(sizing.request_document("tts", {}), "tts")
        with mock.patch.object(tool.curses, "curs_set"), mock.patch.object(tool, "discover_voices", return_value=()), \
             mock.patch.object(tool, "discover_sinks", return_value=()), mock.patch.object(tool, "probe"), \
             mock.patch.object(tool, "_draw") as draw, mock.patch.object(tool, "_size_models", return_value=report), \
             mock.patch.object(settings, "update") as save:
            self.assertEqual(tool._run_tui(screen), 0)
        self.assertIn("Smallest resource candidate", draw.call_args.args[-1])
        save.assert_not_called()

    def test_installed_runtime_can_delegate_without_source_or_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory); prefix = temp / "prefix"
            subprocess.run(["make", "install", f"PREFIX={prefix}"], cwd=root, check=True, capture_output=True)
            provider = temp / "provider"
            provider.write_text(f"#!{sys.executable}\nimport json,sys\nrequest=json.load(sys.stdin)\nprint(json.dumps(request))\n")
            provider.chmod(0o700)
            env = {"PATH": os.environ.get("PATH", ""), "PLEBIAN_MODEL_SIZER": str(provider),
                   "GPU_TERMINAL_HOME": str(temp / "data"), "PYTHONDONTWRITEBYTECODE": "1"}
            # The fake returns a request instead of a response: proving that
            # the installed module executes and validates the provider reply.
            result = subprocess.run([str(prefix / "bin/kilix-stt"), "--recommend"], cwd=temp,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("incompatible or mismatched", result.stderr)
            self.assertFalse((temp / "data").exists())


if __name__ == "__main__":
    unittest.main()
