"""Explicit Qwen named-voice adapter for the local, receipt-backed provider."""

from __future__ import annotations

import threading
import time
import uuid

from . import models, protocol, util
from .tts import (SynthesisProvenance, TtsDeadlineExceeded, TtsError,
                  TtsUnsupported, _as_text, _bounded)


class QwenBusy(TtsError):
    code = protocol.ERR_BUSY


def _provider():
    try:
        from kilix_qwen_tts.protocol import ProtocolError
        from kilix_qwen_tts.service import client_request, request_value, runtime_directory
    except ImportError as error:
        raise TtsError(
            "kilix-qwen-tts client is not installed in the Voice runtime. "
            "Install the pinned Qwen provider and start its receipt-backed "
            "CPU service before choosing this model.") from error
    return ProtocolError, client_request, request_value, runtime_directory


def _translate(error: Exception) -> TtsError:
    code = getattr(error, "code", "")
    if code == "DEADLINE_EXCEEDED":
        return TtsDeadlineExceeded("Qwen provider exceeded the request deadline")
    if code == "BUSY":
        return QwenBusy("Qwen provider is busy with another synthesis")
    if code == "UNSUPPORTED_CAPABILITY":
        return TtsUnsupported("Qwen provider does not support this model or voice")
    return TtsError(f"Qwen provider unavailable ({code or type(error).__name__}): {error}")


class QwenProviderTts:
    """Use only the 0.6B CustomVoice model, never an implicit provider fallback."""

    name = models.TTS_ENGINE_QWEN
    model = models.QWEN_CUSTOMVOICE_MODEL
    rate = 0  # No words-per-minute control in the candidate provider contract.
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

    def check_available(self, *, budget: float | None = None) -> None:
        timeout = _bounded(5.0, budget)
        ProtocolError, client_request, request_value, runtime_directory = _provider()
        try:
            response = client_request(runtime_directory(),
                                      request_value("models", timeout=timeout))
        except (ProtocolError, OSError) as error:
            raise _translate(error) from error
        rows = response.get("models") if isinstance(response, dict) else None
        selected = next((row for row in rows if isinstance(row, dict)
                         and row.get("id") == self.model), None) if isinstance(rows, list) else None
        if selected is None:
            raise TtsError(f"Qwen model {self.model} is not served by the local provider")
        if (selected.get("asset_authority") != "kilix-content"
                or selected.get("installed") is not True
                or selected.get("capabilities") != ["named_voice"]):
            raise TtsError("Qwen model is not a receipt-backed named-voice installation")

    def synth(self, text: str, *, budget: float | None = None) -> tuple[bytes, int]:
        clean = _as_text(text).strip()
        if not clean:
            return b"", 24000
        timeout = _bounded(300.0, budget)
        started = time.monotonic()
        if self._cancelled.is_set():
            return b"", 24000
        # Recheck at each chunk: a stale acknowledgement cannot authorize a
        # later clip after the provider's installed asset has been removed.
        self.check_available(budget=timeout)
        remaining = _bounded(timeout, timeout - (time.monotonic() - started))
        ProtocolError, client_request, request_value, runtime_directory = _provider()
        args = {"task": "synthesize", "text": clean, "model_id": self.model,
                "language": "en", "seed": 0, "max_duration_ms": 60000,
                "mode": "named_voice", "voice_id": self.voice,
                "output": {"sample_format": "s16le", "sample_rate_hz": 24000,
                           "channels": 1}}
        try:
            result, wav = client_request(
                runtime_directory(), request_value("submit", job_id=uuid.uuid4().hex,
                                                   args=args, timeout=remaining),
                cancelled=self._cancelled.is_set)
        except ProtocolError as error:
            if error.code == "CANCELED" and self._cancelled.is_set():
                return b"", 24000
            raise _translate(error) from error
        except OSError as error:
            raise _translate(error) from error
        if result.get("model_id") != self.model or result.get("seed") != 0:
            raise TtsError("Qwen provider returned the wrong model or seed")
        try:
            pcm, sample_rate = util.parse_wav_bytes(wav, strict=True)
        except ValueError as error:
            raise TtsError(f"Qwen provider returned invalid WAV audio: {error}") from error
        if sample_rate != 24000:
            raise TtsError("Qwen provider returned audio at an unexpected sample rate")
        self.last_provenance = SynthesisProvenance(
            self.model, self.voice, seed=0, seed_consumed=True,
            reproducible=False, rate_wpm=0)
        return pcm, sample_rate

    def cancel(self) -> None:
        self._cancelled.set()

    close = cancel
