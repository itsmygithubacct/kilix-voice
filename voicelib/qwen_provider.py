"""Explicit Qwen named voices through the separate local provider executable.

The inference environment is pinned to Python 3.12 while Voice runs under the
system Python. A bounded subprocess client keeps those environments separate;
the provider owns the authenticated socket, model snapshot, and job teardown.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import threading
import time

from . import models, protocol, util
from .tts import (SynthesisProvenance, TtsDeadlineExceeded, TtsError,
                  TtsUnsupported, _as_text, _bounded, _budget_cut)

SAMPLE_RATE = 24000
MAX_DURATION_MS = 60000
MAX_WAV_BYTES = 44 + MAX_DURATION_MS * 48
PROBE_TIMEOUT_S = 5.0
SYNTH_TIMEOUT_S = 300.0
REFUSAL = re.compile(rb"KILIX_QWEN_TTS_REFUSAL \[([A-Z_]+)\] speech request failed\s*")


class QwenBusy(TtsError):
    code = protocol.ERR_BUSY


def provider_binary() -> str | None:
    return util.which(os.environ.get("KILIX_QWEN_TTS", "kilix-qwen-tts"))


class QwenProviderTts:
    """Use only the 0.6B CustomVoice model, never an implicit fallback."""

    name = models.TTS_ENGINE_QWEN
    model = models.QWEN_CUSTOMVOICE_MODEL
    rate = 0  # The provider has no words-per-minute control.
    last_provenance: SynthesisProvenance | None = None

    def __init__(self, *, voice: str | None = None, rate: int | None = None):
        if rate is not None:
            raise TtsUnsupported("Qwen does not support --rate; omit it for this model")
        self.voice = "Vivian" if voice is None else voice
        if (not isinstance(self.voice, str) or not 1 <= len(self.voice) <= 32
                or not self.voice.isascii() or not all(
                    char.isalnum() or char in "._:+-" for char in self.voice)):
            raise TtsUnsupported("Qwen voice must be a bounded provider voice ID")
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def _run(self, arguments: list[str], payload: bytes, *, cap: float,
             budget: float | None, maximum: int) -> tuple[bytes, bytes, int]:
        timeout = _bounded(cap, budget)
        binary = provider_binary()
        if binary is None:
            raise TtsError("kilix-qwen-tts is not installed. Install its pinned "
                           "client and start a receipt-backed provider service.")
        if self._cancelled.is_set():
            return b"", b"", -9
        deadline = time.monotonic() + timeout
        try:
            process = subprocess.Popen([binary, *arguments], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as error:
            raise TtsError(f"cannot start kilix-qwen-tts: {error}") from error
        with self._lock:
            self._process = process
            if self._cancelled.is_set():
                process.kill()
        buffers = (bytearray(), bytearray())
        try:
            with selectors.DefaultSelector() as ready:
                for channel in (process.stdin, process.stdout, process.stderr):
                    os.set_blocking(channel.fileno(), False)
                ready.register(process.stdout, selectors.EVENT_READ, 0)
                ready.register(process.stderr, selectors.EVENT_READ, 1)
                if payload:
                    ready.register(process.stdin, selectors.EVENT_WRITE, 2)
                else:
                    process.stdin.close()
                pending = memoryview(payload)
                while ready.get_map():
                    if self._cancelled.is_set():
                        return b"", b"", -9
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        failure = TtsDeadlineExceeded if _budget_cut(cap, budget) else TtsError
                        raise failure("Qwen client exceeded the speech deadline")
                    for key, _ in ready.select(min(0.05, remaining)):
                        channel, index = key.fileobj, key.data
                        try:
                            if index == 2:
                                pending = pending[os.write(channel.fileno(), pending[:4096]):]
                                if not pending:
                                    ready.unregister(channel)
                                    channel.close()
                                continue
                            block = os.read(channel.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            ready.unregister(channel)
                            channel.close()
                            continue
                        if not block:
                            ready.unregister(channel)
                            continue
                        buffers[index].extend(block)
                        if len(buffers[index]) > (maximum if index == 0 else 65536):
                            raise TtsError("Qwen client returned more data than permitted")
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                failure = TtsDeadlineExceeded if _budget_cut(cap, budget) else TtsError
                raise failure("Qwen client exceeded the speech deadline") from error
            return bytes(buffers[0]), bytes(buffers[1]), process.returncode
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            for channel in (process.stdin, process.stdout, process.stderr):
                channel.close()
            with self._lock:
                if self._process is process:
                    self._process = None

    def check_available(self, *, budget: float | None = None) -> None:
        out, _err, code = self._run(["models"], b"", cap=PROBE_TIMEOUT_S,
                                    budget=budget, maximum=65536)
        if code:
            raise TtsError("Qwen provider is unavailable; start its local service")
        try:
            records = json.loads(out).get("models")
            selected = [row for row in records if isinstance(row, dict)
                        and row.get("id") == self.model] if isinstance(records, list) else []
        except (ValueError, UnicodeError, AttributeError):
            selected = []
        if (len(selected) != 1 or selected[0].get("installed") is not True
                or selected[0].get("asset_authority") != "kilix-content"
                or selected[0].get("capabilities") != ["named_voice"]):
            raise TtsError(f"Qwen model {self.model} is not a receipt-backed "
                           "named-voice installation")

    def synth(self, text: str, *, budget: float | None = None) -> tuple[bytes, int]:
        clean = _as_text(text).strip()
        if not clean or self._cancelled.is_set():
            return b"", SAMPLE_RATE
        try:
            encoded = clean.encode("utf-8", "strict")
        except UnicodeError as error:
            raise TtsUnsupported("Qwen speech text must be valid UTF-8") from error
        if len(encoded) > 16384:
            raise TtsUnsupported("Qwen accepts at most 16384 UTF-8 bytes per clip")
        timeout = _bounded(SYNTH_TIMEOUT_S, budget)
        out, err, code = self._run([
            "synthesize", "--wav-stdout", "--require-installed-asset",
            "--model-id", self.model, "--voice-id", self.voice,
            "--language", "en", "--seed", "0", "--timeout", str(timeout),
            "--max-duration-ms", str(MAX_DURATION_MS),
        ], encoded, cap=SYNTH_TIMEOUT_S, budget=budget, maximum=MAX_WAV_BYTES)
        if self._cancelled.is_set():
            return b"", SAMPLE_RATE
        if code:
            match = REFUSAL.fullmatch(err)
            refusal = match[1].decode() if match else "PROVIDER_UNAVAILABLE"
            if refusal == "DEADLINE_EXCEEDED" and _budget_cut(SYNTH_TIMEOUT_S, budget):
                raise TtsDeadlineExceeded("Qwen speech deadline elapsed")
            if refusal == "BUSY":
                raise QwenBusy("Qwen provider is busy with another synthesis")
            if refusal == "UNSUPPORTED_CAPABILITY":
                raise TtsUnsupported("Qwen provider does not support this model or voice")
            raise TtsError(f"Qwen provider refused synthesis ({refusal})")
        try:
            metadata = json.loads(err)
            if (not isinstance(metadata, dict) or metadata.get("model_id") != self.model
                    or type(metadata.get("seed")) is not int or metadata["seed"] != 0
                    or metadata.get("audio") != {"byte_length": len(out),
                                                "sha256": hashlib.sha256(out).hexdigest()}):
                raise ValueError("result does not match the selected model or audio")
            pcm, sample_rate = util.parse_wav_bytes(out, strict=True)
            if not pcm or sample_rate != SAMPLE_RATE:
                raise ValueError("expected nonempty 24 kHz mono PCM16")
        except (ValueError, UnicodeError) as error:
            raise TtsError(f"Qwen provider returned invalid speech: {error}") from error
        self.last_provenance = SynthesisProvenance(
            self.model, self.voice, seed=0, seed_consumed=True,
            reproducible=False, rate_wpm=0)
        return pcm, sample_rate

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    close = cancel
