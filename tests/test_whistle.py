"""Whistle protocol, long-turn boundaries, local-only loading and consent."""
import ctypes
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from voicelib import consent, models, stt, whistle_provider as provider
from tests.test_stt_tool import load_stt_tool


class ProtocolTests(unittest.TestCase):
    def serve(self, data, transcribe=None):
        out = io.BytesIO()
        with mock.patch.object(provider, "Whistle") as model:
            model.return_value.transcribe.side_effect = transcribe or (lambda pcm: "words" if pcm else "")
            code = provider.serve(SimpleNamespace(model="local", library="local.so", language="en"),
                                  io.BytesIO(data), out)
            calls = model.return_value.transcribe.call_args_list
        return code, [json.loads(line) for line in out.getvalue().splitlines()], calls

    def test_multiple_requests_remain_framed_after_an_empty_turn(self):
        code, rows, calls = self.serve(b'{"pcm_bytes": 0}\n{"pcm_bytes": 4}\n\0\0\1\0')
        self.assertEqual(code, 0)
        self.assertEqual(rows[1:], [{"text": ""}, {"text": "words"}])
        self.assertEqual([c.args[0] for c in calls], [b"", b"\0\0\1\0"])

    def test_bad_headers_and_truncated_pcm_stop_without_decoding(self):
        for data in (b'[]\n', b'{"pcm_bytes": true}\n', b'{"pcm_bytes": 3}\n',
                     b'{"pcm_bytes": -2}\n', b'{"pcm_bytes": 4}\n\0\0',
                     b'{"pcm_bytes": 19200002}\n', b'{"pcm_bytes": 0, "x": 1}\n',
                     b'x' * 257, b'{"pcm_bytes": 0}'):
            with self.subTest(data=data[:30]):
                code, rows, calls = self.serve(data)
                self.assertEqual(code, 2)
                self.assertIn("error", rows[-1])
                self.assertFalse(calls)

    def test_decode_error_does_not_desynchronize_the_next_request(self):
        code, rows, calls = self.serve(b'{"pcm_bytes": 0}\n' * 2,
                                      [provider.ProviderError("bad decode"), "next"])
        self.assertEqual(code, 0)
        self.assertEqual(rows[1:], [{"error": "bad decode"}, {"text": "next"}])

    def test_missing_library_is_refused_without_any_download(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(provider.ProviderError, "local Whistle library"):
                provider.Whistle("not-a-model-name", "/does-not-exist/libneedle3.so")


class NativeBoundaryTests(unittest.TestCase):
    def engine(self):
        engine = object.__new__(provider.Whistle)
        engine.language = b"en"
        engine.buffer = ctypes.create_string_buffer(provider.REPLY_BYTES)
        engine.lib = mock.Mock()
        def result(text):
            engine.buffer.value = json.dumps({"text": text}).encode()
            return 1
        engine.lib.needle_transcribe.side_effect = lambda *args: result("short")
        engine.lib.needle_stream_transcribe_process.side_effect = lambda *args: result("part")
        engine.lib.needle_stream_transcribe_stop.side_effect = lambda *args: result("tail")
        return engine

    def test_thirty_seconds_is_one_shot_and_one_sample_more_is_streamed(self):
        engine = self.engine()
        self.assertEqual(engine.transcribe(b'\0' * 32000 * 30), "short")
        engine.lib.needle_transcribe.assert_called_once()
        self.assertEqual(engine.transcribe(b'\0' * (32000 * 30 + 2)), " ".join(["part"] * 31 + ["tail"]))
        calls = engine.lib.needle_stream_transcribe_process.call_args_list
        self.assertEqual(sum(c.args[1] for c in calls), 480001)
        self.assertEqual(calls[-1].args[1], 1)
        engine.lib.needle_stream_transcribe_stop.assert_called_once()

    def test_stream_failure_resets_native_state_before_the_next_turn(self):
        engine = self.engine()
        engine.lib.needle_stream_transcribe_process.return_value = -1
        engine.lib.needle_stream_transcribe_process.side_effect = None
        engine.lib.needle_last_error.return_value = b"decode failed"
        with self.assertRaisesRegex(provider.ProviderError, "decode failed"):
            engine.transcribe(b'\0' * 32000 * 31)
        engine.lib.needle_stream_transcribe_stop.assert_called_once()
        self.assertEqual(engine.transcribe(b'\0' * 32000), "short")

    def test_empty_and_invalid_pcm_never_enter_the_native_runtime(self):
        engine = self.engine()
        self.assertEqual(engine.transcribe(b""), "")
        with self.assertRaises(provider.ProviderError):
            engine.transcribe(b"x")
        engine.lib.needle_transcribe.assert_not_called()


class IntegrationTests(unittest.TestCase):
    def test_engine_only_selection_resolves_the_whistle_content_model(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
                "KILIX_DATA_HOME": tmp, "KILIX_VOICE_MODEL_PATH": ""}):
            target = stt.resolve_stt({"stt": {"engine": "whistle", "model": "small-en-us"}})
            self.assertEqual((target.engine, target.model_id), ("whistle", "whistle"))
            self.assertEqual(target.model_dir, str(Path(tmp) / "desktop-apps/assets/whistle"))

    def test_only_the_held_whistle_weights_are_bound_and_inherited(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
                "KILIX_DATA_HOME": tmp, "XDG_RUNTIME_DIR": tmp}):
            root = Path(tmp)
            model = root / 'model'; model.mkdir()
            weights = model / 'whistle.cact'; weights.write_bytes(b'weights A')
            library = root / 'libneedle3.so'; library.write_bytes(b'library')
            digest = consent.payload_digest_at(str(model), "whistle")
            with mock.patch.dict(os.environ, {"KILIX_VOICE_WHISTLE_LIBRARY": str(library)}), \
                    mock.patch.object(stt.WhistleStt, '_start'):
                engine = stt.WhistleStt(model_path=str(model), consented_payload=digest)
                try:
                    self.assertEqual(set(engine._held), {'whistle.cact'})
                    engine._links = None
                    command = engine._command()
                    self.assertIn(str(library), command)
                    self.assertIn('en', command)
                    self.assertTrue(command[1].endswith('whistle_provider.py'))
                    weights.write_bytes(b'weights B')
                    with self.assertRaises(stt.SttError):
                        engine._require_unchanged('during decode')
                finally:
                    engine.close()
                with self.assertRaises(stt.SttError):
                    stt.WhistleStt(model_path=str(model), consented_payload=digest)

    def test_install_is_separate_from_selection_and_routes_through_content(self):
        tool = load_stt_tool()
        spec = models.MODEL_BY_ID['whistle']
        with mock.patch.object(tool, 'kilix_launcher', return_value='/kilix'):
            self.assertEqual(tool.model_install_argv(spec), ['/kilix', 'models', 'install', 'whistle'])
            self.assertEqual(tool.runtime_install_argv(spec), ['/kilix', 'voice', 'whistle'])
