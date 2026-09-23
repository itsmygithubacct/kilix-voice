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


if __name__ == "__main__":
    unittest.main()
