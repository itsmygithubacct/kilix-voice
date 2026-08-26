"""Offline coverage for kilix-tts speech text and model selection.

No daemon, synthesizer, or audio device is opened.  The executable is loaded as
a module and its one control exchange is replaced with a recorder, so these
tests cover the exact request the CLI would put on the private socket.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from voicelib import models, protocol


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    loader = importlib.machinery.SourceFileLoader(
        "kilix_tts_cli_test", str(ROOT / "kilix-tts"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise AssertionError("could not load kilix-tts")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


class TtsToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()

    @staticmethod
    def accepted(*, chunks: int = 1, model: str = "espeak",
                 voice: str = "en-us", rate: int = 170) -> dict:
        return {"ok": True, "id": "kilix-tts", "chunks": chunks,
                "model": model, "voice": voice, "rate": rate}

    def test_cli_speaks_arbitrary_text_with_explicit_selection(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
                self.tool, "control",
                return_value=self.accepted(chunks=2, model="mbrola",
                                           voice="us1", rate=200)) as control, \
                contextlib.redirect_stdout(output):
            result = self.tool.main([
                "--speak", "Kilix can say this exact text.",
                "--model", "mbrola", "--voice", "us1", "--rate", "200",
            ])

        self.assertEqual(result, 0)
        control.assert_called_once_with({
            "op": "speak", "id": "kilix-tts",
            "text": "Kilix can say this exact text.",
            "model": "mbrola", "voice": "us1", "rate": 200,
        })
        self.assertIn("Speaking 2 clips with mbrola/us1 at 200 wpm",
                      output.getvalue())

    def test_dash_reads_utf8_text_from_stdin_without_rewriting_it(self) -> None:
        text = "Hello, café.\nSecond line."
        output = io.StringIO()
        with mock.patch.object(self.tool.sys, "stdin", io.StringIO(text)), \
                mock.patch.object(
                    self.tool, "control",
                    return_value=self.accepted(voice="en-gb")) as control, \
                contextlib.redirect_stdout(output):
            result = self.tool.main([
                "--speak", "-", "--model", "espeak", "--voice", "en-gb",
            ])

        self.assertEqual(result, 0)
        self.assertEqual(control.call_args.args[0]["text"], text)
        self.assertIn("espeak/en-gb", output.getvalue())

    def test_empty_speech_is_refused_before_control(self) -> None:
        with mock.patch.object(self.tool, "control") as control:
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool.speak_text(" \n\t ")
        control.assert_not_called()
        self.assertIn("nothing to read aloud", str(caught.exception))

    def test_invalid_selectors_are_refused_before_control(self) -> None:
        cases = (
            {"model": "../../model"},
            {"voice": "en-us; shutdown"},
            {"rate": 999},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), \
                    mock.patch.object(self.tool, "control") as control:
                with self.assertRaises(self.tool.VoiceToolError):
                    self.tool.speak_text("hello", **arguments)
                control.assert_not_called()

    def test_unconfirmed_override_stops_the_possibly_wrong_voice(self) -> None:
        replies = [
            {"ok": True, "chunks": 1},
            {"ok": True, "stopped": True},
        ]
        with mock.patch.object(self.tool, "control", side_effect=replies) as control:
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool.speak_text("hello", model="mbrola", voice="us1")

        self.assertEqual(control.call_count, 2)
        self.assertEqual(control.call_args_list[1].args[0]["op"],
                         protocol.OP_STOP_SPEECH)
        self.assertIn("matching kilix-tts and kilix-voiced",
                      str(caught.exception))

    def test_lost_speak_ack_gets_a_compensating_stop(self) -> None:
        uncertain = self.tool.VoiceToolError(
            "reply was lost", request_may_be_active=True)
        with mock.patch.object(
                self.tool, "control",
                side_effect=[uncertain, {"ok": True, "stopped": True}]
        ) as control:
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool.speak_text("hello")

        self.assertEqual(control.call_count, 2)
        self.assertEqual(control.call_args_list[1].args[0]["op"],
                         protocol.OP_STOP_SPEECH)
        self.assertIn("stop request was confirmed", str(caught.exception))

    def test_stdin_is_strict_utf8_and_bounded(self) -> None:
        class Input:
            def __init__(self, payload: bytes) -> None:
                self.buffer = io.BytesIO(payload)

        with mock.patch.object(self.tool.sys, "stdin", Input(b"\xff")):
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool._stdin_speech_text()
        self.assertIn("valid UTF-8", str(caught.exception))

        oversized = b"x" * (protocol.MAX_REQUEST_BYTES + 1)
        with mock.patch.object(self.tool.sys, "stdin", Input(oversized)):
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool._stdin_speech_text()
        self.assertIn("larger than", str(caught.exception))

    def test_encoded_packet_limit_is_checked_before_opening_a_socket(self) -> None:
        with mock.patch.object(self.tool.socket, "socket") as socket_factory:
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool.control({"op": "speak", "text": "x" *
                                   protocol.MAX_REQUEST_BYTES})
        socket_factory.assert_not_called()
        self.assertIn("control-packet limit", str(caught.exception))

    def test_speech_options_require_the_speech_action(self) -> None:
        for arguments in (["--model", "espeak"], ["--voice", "en-us"],
                          ["--rate", "170"], ["--output", "speech.wav"]):
            with self.subTest(arguments=arguments), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self.tool.main(list(arguments))
                self.assertEqual(caught.exception.code, 2)

    def test_cli_saves_private_wav_without_contacting_daemon(self) -> None:
        pcm = b"\x01\x00\xff\xff" * 100
        rendered = self.tool.tts_lib.RenderedSpeech(
            pcm, 16000, 2, "espeak", "en-gb", 200)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "paragraph.wav"
            with mock.patch.object(
                    self.tool.tts_lib, "render_text",
                    return_value=rendered) as render, \
                    mock.patch.object(self.tool, "control") as control, \
                    contextlib.redirect_stdout(output):
                result = self.tool.main([
                    "--speak", "This is an export test.",
                    "--model", "espeak", "--voice", "en-gb",
                    "--rate", "200", "--output", str(target),
                ])

            self.assertEqual(result, 0)
            control.assert_not_called()
            render.assert_called_once_with(
                "This is an export test.", model="espeak", voice="en-gb",
                rate=200, max_chars=mock.ANY)
            parsed, sample_rate = self.tool.util.parse_wav_bytes(
                target.read_bytes())
            self.assertEqual((parsed, sample_rate), (pcm, 16000))
            self.assertEqual(os.stat(target).st_mode & 0o777, 0o600)
            self.assertIn("Saved 2 clips with espeak/en-gb", output.getvalue())
            self.assertIn("as WAV", output.getvalue())

    def test_save_alias_encodes_mp3_after_building_a_wav(self) -> None:
        rendered = self.tool.tts_lib.RenderedSpeech(
            b"\x01\x00" * 100, 22050, 1, "espeak", "en-us", 170)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "paragraph.mp3"
            with mock.patch.object(
                    self.tool.tts_lib, "render_text",
                    return_value=rendered), \
                    mock.patch.object(
                        self.tool, "_encode_mp3",
                        return_value=(b"ID3-test", "test-encoder")) as encode, \
                    contextlib.redirect_stdout(output):
                self.assertEqual(self.tool.main([
                    "--speak", "MP3 export test.", "--save", str(target),
                ]), 0)

            self.assertEqual(target.read_bytes(), b"ID3-test")
            wav = encode.call_args.args[0]
            self.assertEqual(wav[:4], b"RIFF")

    def test_export_rejects_unknown_suffix_and_existing_file(self) -> None:
        with mock.patch.object(self.tool.tts_lib, "render_text") as render:
            with self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool.save_text("hello", "speech.ogg")
        render.assert_not_called()
        self.assertIn(".wav or .mp3", str(caught.exception))

        rendered = self.tool.tts_lib.RenderedSpeech(
            b"\x01\x00", 16000, 1, "espeak", "en-us", 170)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exists.wav"
            target.write_bytes(b"keep me")
            with mock.patch.object(
                    self.tool.tts_lib, "render_text",
                    return_value=rendered), \
                    self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool.save_text("hello", str(target))
            self.assertEqual(target.read_bytes(), b"keep me")
        self.assertIn("refusing to overwrite", str(caught.exception))

    def test_mp3_without_an_encoder_has_an_actionable_error(self) -> None:
        with mock.patch.object(self.tool.util, "which", return_value=None), \
                self.assertRaises(self.tool.VoiceToolError) as caught:
            self.tool._encode_mp3(b"RIFF")
        self.assertIn("Install ffmpeg", str(caught.exception))

    def test_failed_output_write_leaves_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "partial.wav"
            with mock.patch.object(
                    self.tool.os, "fsync", side_effect=OSError("disk full")), \
                    self.assertRaises(self.tool.VoiceToolError) as caught:
                self.tool._write_new_output(str(target), b"audio")
            self.assertFalse(target.exists())
        self.assertIn("No partial audio file was kept", str(caught.exception))

    def test_models_lists_only_registered_families(self) -> None:
        output = io.StringIO()
        with mock.patch.object(self.tool.tts_lib, "espeak_binary",
                               return_value="/usr/bin/espeak-ng"), \
                mock.patch.object(self.tool.util, "which",
                                  return_value="/usr/bin/mbrola"), \
                mock.patch.object(self.tool, "discover_voices",
                                  return_value=("en-gb", "en-us")), \
                mock.patch.object(self.tool.settings, "tts_engine",
                                  return_value="espeak"), \
                mock.patch.object(self.tool.tts_lib, "piper_status",
                                  return_value=(False, "not installed")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(self.tool.main(["--models"]), 0)

        shown = output.getvalue()
        for spec in models.TTS_MODELS:
            self.assertIn(f"model={spec.catalog_id}", shown)
        self.assertNotIn("command=", shown)
        self.assertNotIn("url=", shown)

    def test_install_delegates_only_to_the_fixed_piper_provider(self) -> None:
        completed = self.tool.subprocess.CompletedProcess(
            ["provider"], 0, "Installed Kristin\n", "")
        output = io.StringIO()
        with mock.patch.object(
                self.tool.tts_lib, "piper_binary",
                return_value="/fixed/kilix-piper-tts"), \
                mock.patch.object(
                    self.tool.subprocess, "run",
                    return_value=completed) as run, \
                contextlib.redirect_stdout(output):
            self.assertEqual(self.tool.main([
                "--install", models.PIPER_KRISTIN_MODEL]), 0)
        self.assertEqual(run.call_args.args[0], [
            "/fixed/kilix-piper-tts", "install",
            models.PIPER_KRISTIN_MODEL])
        self.assertIn("Installed Kristin", output.getvalue())


if __name__ == "__main__":
    unittest.main()
