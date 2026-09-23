"""Offline contract tests for Voice's explicit Qwen provider adapter."""

from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

from voicelib import models, protocol, tts
from voicelib.qwen_provider import QwenProviderTts


class ProviderError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def wave(pcm=b"\x01\x00" * 2400):
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", len(pcm) + 36,
                       b"WAVE", b"fmt ", 16, 1, 1, 24000, 48000, 2, 16,
                       b"data", len(pcm)) + pcm


class QwenAdapterTests(unittest.TestCase):
    def provider(self, *, authority="kilix-content", failure=None):
        calls = []

        def request_value(operation, **kwargs):
            return {"operation": operation, **kwargs}

        def client_request(_directory, request, **kwargs):
            calls.append((request, kwargs))
            if request["operation"] == "models":
                return {"models": [{"id": models.QWEN_CUSTOMVOICE_MODEL,
                                    "installed": True,
                                    "capabilities": ["named_voice"],
                                    "asset_authority": authority}]}
            if failure:
                raise ProviderError(failure)
            return {"model_id": models.QWEN_CUSTOMVOICE_MODEL, "seed": 0}, wave()

        return calls, patch("voicelib.qwen_provider._provider", return_value=(
            ProviderError, client_request, request_value, lambda: "/runtime"))

    def test_explicit_selection_and_bound_audio(self):
        self.assertEqual(tts.make_tts(model=models.QWEN_CUSTOMVOICE_MODEL).model,
                         models.QWEN_CUSTOMVOICE_MODEL)
        calls, provider = self.provider()
        with provider:
            engine = QwenProviderTts()
            pcm, rate = engine.synth("Hello", budget=2)
        self.assertEqual((len(pcm), rate), (4800, 24000))
        self.assertEqual(calls[-1][0]["args"]["model_id"], engine.model)
        self.assertEqual(calls[-1][0]["args"]["voice_id"], "Vivian")
        self.assertEqual(engine.last_provenance.model, engine.model)
        self.assertEqual(engine.last_provenance.seed, 0)

    def test_unreceipted_runtime_is_refused_before_submit(self):
        calls, provider = self.provider(authority="local-stage")
        with provider, self.assertRaises(tts.TtsError) as caught:
            QwenProviderTts().synth("Hello")
        self.assertIn("receipt-backed", str(caught.exception))
        self.assertEqual([request["operation"] for request, _ in calls], ["models"])

    def test_unsupported_rate_and_voice(self):
        with self.assertRaises(tts.TtsUnsupported):
            QwenProviderTts(rate=170)
        with self.assertRaises(tts.TtsUnsupported):
            QwenProviderTts(voice="/tmp/voice")

    def test_busy_and_deadline_codes(self):
        for provider_code, expected in (("BUSY", protocol.ERR_BUSY),
                                        ("DEADLINE_EXCEEDED", protocol.ERR_DEADLINE)):
            with self.subTest(provider_code=provider_code):
                _, provider = self.provider(failure=provider_code)
                with provider, self.assertRaises(tts.TtsError) as caught:
                    QwenProviderTts().synth("Hello")
                self.assertEqual(caught.exception.code, expected)

    def test_cancelled_before_synthesis_does_not_submit(self):
        calls, provider = self.provider()
        with provider:
            engine = QwenProviderTts()
            engine.cancel()
            self.assertEqual(engine.synth("Hello"), (b"", 24000))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
