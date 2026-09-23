"""Read-only client of plebian-model-sizer; all fit arithmetic stays there."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time

from . import models, paths

REQUEST_SCHEMA = "kilix.voice.sizing-request/v1"
RESPONSE_SCHEMA = "plebian.models.voice-sizing/v1-development"
MAX_REPLY_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 15


class SizerError(RuntimeError):
    """The shared provider is missing, failed, or returned an invalid report."""


def request_document(task: str, installed: dict[str, bool | None]) -> dict:
    if task not in ("tts", "stt"):
        raise SizerError("Unsupported speech sizing task.")
    catalog = models.TTS_MODELS if task == "tts" else models.MODELS
    return {"schema": REQUEST_SCHEMA, "models": [
        {"id": spec.catalog_id, "task": task, "backend": "cpu",
         "installed": installed.get(spec.catalog_id), "runtime_supported": spec.runtime_supported}
        for spec in catalog]}


def provider_executable() -> str:
    override = os.environ.get("PLEBIAN_MODEL_SIZER")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    installed = shutil.which("plebian-model-sizer")
    if installed:
        return installed
    root = Path(__file__).resolve().parents[1]
    # A checkout can use its sibling provider. Installed runtimes require an
    # installed provider or an explicit executable, never ambient PYTHONPATH.
    if (root / "Makefile").is_file():
        source = root.parents[1] / "kilix-system-monitor/components/plebian-model-sizer/plebian-model-sizer"
        if source.is_file() and os.access(source, os.X_OK):
            return str(source)
    raise SizerError("Model sizing unavailable. Install plebian-model-sizer 0.2.0+ or set PLEBIAN_MODEL_SIZER.")


def _run(argv: list[str], payload: bytes) -> bytes:
    """Bound stdout and wall time, including descendants holding a pipe open."""
    if len(payload) > 4096:
        raise SizerError("Speech sizing request exceeds its limit.")
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True, bufsize=0)
    except OSError as error:
        raise SizerError("Unable to start the configured model sizer.") from error
    deadline = time.monotonic() + TIMEOUT_SECONDS
    output = bytearray()
    try:
        process.stdin.write(payload)
        process.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SizerError("Model sizing timed out.")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(chunk)
                    if len(output) > MAX_REPLY_BYTES:
                        raise SizerError("Model sizer reply exceeds its limit.")
        if process.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
            raise SizerError("Model sizer failed; check that the installed provider supports speech sizing.")
        return bytes(output)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SizerError("Model sizing failed or timed out.") from error
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        process.stdout.close()
        process.stdin.close()


def _parse(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("non-finite number")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise SizerError("Model sizer returned invalid JSON.") from error


def validate_report(report: dict, request: dict, task: str) -> None:
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (not isinstance(report, dict) or report.get("schema") != RESPONSE_SCHEMA
            or report.get("request_sha256") != digest or report.get("task") != task
            or report.get("resource_source") != "live" or report.get("selected_model", "missing") is not None
            or report.get("qualification_eligible") is not False):
        raise SizerError("Model sizer returned an incompatible or mismatched report.")
    expected = {row["id"]: row for row in request["models"]}
    rows = report.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise SizerError("Model sizer returned an incomplete catalog.")
    seen = set()
    fitting = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] not in expected or row["id"] in seen:
            raise SizerError("Model sizer returned an unknown or duplicate model.")
        seen.add(row["id"])
        if any(key not in row or type(row.get(key)) is not type(value) or row.get(key) != value for key, value in expected[row["id"]].items()):
            raise SizerError("Model sizer changed the requested model metadata.")
        verdict = row.get("verdict")
        if verdict not in ("estimated-fit", "does-not-fit", "unknown", "unsupported") or row.get("qualification_eligible") is not False:
            raise SizerError("Model sizer returned an invalid fit verdict.")
        if verdict == "estimated-fit":
            if row["runtime_supported"] is not True:
                raise SizerError("Model sizer recommended an unsupported speech runtime.")
            fitting.add(row["id"])
        installation = row.get("installation")
        if not isinstance(installation, dict) or installation.get("verdict") not in ("estimated-fit", "does-not-fit", "unknown"):
            raise SizerError("Model sizer returned an invalid installation verdict.")
        inference = row.get("inference")
        if "inference" not in row:
            raise SizerError("Model sizer omitted its resource assessment.")
        if verdict == "estimated-fit" and (not isinstance(inference, dict) or inference.get("verdict") != "estimated-fit"):
            raise SizerError("Model sizer returned a fit without a resource assessment.")
        if inference is not None:
            if not isinstance(inference, dict) or not isinstance(inference.get("resources"), dict):
                raise SizerError("Model sizer returned invalid resource figures.")
            for resource in inference["resources"].values():
                if not isinstance(resource, dict) or type(resource.get("required_bytes")) is not int or resource["required_bytes"] < 0:
                    raise SizerError("Model sizer returned invalid resource figures.")
            if verdict == "estimated-fit":
                required = {"ram", "vram"} if row["backend"] == "cuda" else {"ram"}
                if not required <= inference["resources"].keys():
                    raise SizerError("Model sizer omitted a required memory budget.")
                for resource in inference["resources"].values():
                    available = resource.get("budget_bytes")
                    if (type(available) is not int or available < resource["required_bytes"]
                            or resource.get("status") != "estimated-fit"):
                        raise SizerError("Model sizer returned an inconsistent fit budget.")
    lists, provisional = report.get("shortlists"), report.get("provisional_candidates")
    if not isinstance(lists, dict) or set(lists) != {task} or not isinstance(provisional, dict) or set(provisional) != {task}:
        raise SizerError("Model sizer returned an invalid shortlist.")
    shortlist = lists[task]
    if (not isinstance(shortlist, list) or any(not isinstance(name, str) for name in shortlist)
            or len(set(shortlist)) != len(shortlist) or set(shortlist) != fitting
            or provisional[task] != (shortlist[0] if shortlist else None)):
        raise SizerError("Model sizer returned an inconsistent recommendation.")


def recommend(task: str, installed: dict[str, bool | None]) -> dict:
    request = request_document(task, installed)
    return recommend_request(request, task)


def recommend_request(request: dict, task: str) -> dict:
    """Assess an explicit consumer catalog through the same bounded protocol."""
    argv = [provider_executable(), "recommend", "voice", "--task", task,
            "--catalog", "-", "--data-root", paths.data_dir(), "--json"]
    report = _parse(_run(argv, json.dumps(request).encode()))
    validate_report(report, request, task)
    return report


def summary(report: dict) -> str:
    candidate = report["provisional_candidates"][report["task"]]
    return (f"Smallest resource candidate: {candidate}. Selection remains manual." if candidate else
            "No candidate has a usable resource estimate. See --recommend --json for details.")


def print_report(report: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, sort_keys=True))
        return
    print("Speech resource estimates; local runtime identity and quality remain unverified.")
    for row in report["candidates"]:
        resource = (row["inference"] or {}).get("resources", {}).get("ram", {})
        amount = resource.get("required_bytes")
        memory = f"{amount / 1024**2:.1f} MiB" if amount is not None else "unknown"
        installed = {True: "yes", False: "no", None: "unknown"}[row["installed"]]
        print(f"{row['id']}: {row['verdict']}; RAM {memory}; installed {installed}; installation {row['installation']['verdict']}")
    print(summary(report))
