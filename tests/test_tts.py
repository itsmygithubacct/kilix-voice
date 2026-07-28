"""Offline tests for voicelib.tts: conditioning, chunking, engine selection.

Nothing here speaks.  The only "engine" that runs is a fake injected through
the ``tts.cmd`` seam with ``subprocess.Popen`` replaced, so the suite passes
with no espeak-ng, no mbrola voice, no audio server and no sound — which is
also the guarantee DESIGN.md makes about this module.
"""

from __future__ import annotations

import base64
import os
import struct
import tempfile
import unittest
from unittest import mock

from voicelib import settings, tts, util

# What kitty actually writes for an image: an APC introducer, key=value pairs,
# ';', base64 payload, string terminator — chunked at 4096 encoded bytes.
_GRAPHICS_CHUNK = 4096

# Big enough that a conditioner which merely trimmed the header would leave
# kilobytes of base64 to be read out loud, and small enough to stay readable.
_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 40


def kitty_graphics(payload: bytes) -> str:
    """Return the APC sequence kitty emits to put ``payload`` on screen."""
    encoded = base64.b64encode(payload).decode("ascii")
    pieces = [encoded[at:at + _GRAPHICS_CHUNK]
              for at in range(0, len(encoded), _GRAPHICS_CHUNK)] or [""]
    parts = []
    for index, piece in enumerate(pieces):
        more = int(index < len(pieces) - 1)
        header = (f"a=T,f=100,s=64,v=64,m={more}" if index == 0
                  else f"m={more}")
        parts.append(f"\x1b_G{header};{piece}\x1b\\")
    return "".join(parts)


class ConditionEscapes(unittest.TestCase):
    """Escape sequences and their parameters never reach the synthesiser."""

    def test_csi_sgr_is_removed_with_its_parameters(self) -> None:
        text = "\x1b[1;32mgreen\x1b[0m and \x1b[38;2;255;0;0mred\x1b[m"
        self.assertEqual(tts.condition_text(text, max_chars=None),
                         "green and red")

    def test_csi_cursor_control_is_removed(self) -> None:
        text = "\x1b[2J\x1b[H\x1b[3;10Hprompt\x1b[K"
        self.assertEqual(tts.condition_text(text, max_chars=None), "prompt")

    def test_osc_title_terminated_by_bel_is_removed(self) -> None:
        text = "\x1b]0;user@host: ~/src\x07$ ls"
        self.assertEqual(tts.condition_text(text, max_chars=None), "$ ls")

    def test_osc_hyperlink_terminated_by_st_keeps_only_the_label(self) -> None:
        text = "\x1b]8;;https://example.invalid/x\x1b\\label\x1b]8;;\x1b\\"
        self.assertEqual(tts.condition_text(text, max_chars=None), "label")

    def test_dcs_sixel_payload_is_removed(self) -> None:
        text = "before \x1bPq#0;2;0;0;0#1~~~~\x1b\\ after"
        self.assertEqual(tts.condition_text(text, max_chars=None),
                         "before after")

    def test_eight_bit_c1_introducers_are_removed(self) -> None:
        # A capture decoded as latin-1 upstream carries C1 bytes, not ESC.
        self.assertEqual(
            tts.condition_text("\x9b31mred\x9b0m", max_chars=None), "red")
        self.assertEqual(tts.condition_text("\x9d0;title\x9cbody",
                                            max_chars=None), "body")

    def test_two_character_and_charset_escapes_are_removed(self) -> None:
        text = "\x1b7saved\x1b8\x1b(Bascii\x1bc"
        self.assertEqual(tts.condition_text(text, max_chars=None),
                         "savedascii")

    def test_control_characters_and_del_are_removed(self) -> None:
        self.assertEqual(tts.condition_text("a\x00b\x7fc\x01", max_chars=None),
                         "abc")

    def test_tab_becomes_a_space_and_carriage_return_a_break(self) -> None:
        self.assertEqual(tts.condition_text("a\tb\r\nc", max_chars=None),
                         "a b\nc")


class ConditionGraphics(unittest.TestCase):
    """A kitty graphics payload is dropped whole, never read out loud."""

    def test_large_chunked_image_is_dropped_entirely(self) -> None:
        image = kitty_graphics(_IMAGE_BYTES)
        self.assertEqual(image.count("\x1b_G"), 4)  # a genuinely chunked image
        result = tts.condition_text(f"before {image} after", max_chars=None)
        self.assertEqual(result, "before after")

    def test_no_fragment_of_the_base64_survives(self) -> None:
        encoded = base64.b64encode(_IMAGE_BYTES).decode("ascii")
        result = tts.condition_text(kitty_graphics(_IMAGE_BYTES),
                                    max_chars=None)
        self.assertEqual(result, "")
        # Belt and braces: no run of the payload can have leaked into the text.
        for at in range(0, len(encoded) - 48, 512):
            self.assertNotIn(encoded[at:at + 48], result)

    def test_unterminated_payload_is_dropped_to_the_end(self) -> None:
        # A capture cut mid-image leaves a header and megabytes of base64 with
        # no string terminator; losing the tail beats speaking it.
        encoded = base64.b64encode(_IMAGE_BYTES).decode("ascii")
        result = tts.condition_text(f"visible \x1b_Ga=T,f=100;{encoded}",
                                    max_chars=None)
        self.assertEqual(result, "visible")

    def test_image_between_lines_leaves_the_text_around_it(self) -> None:
        text = f"line one\n{kitty_graphics(b'small')}\nline two"
        self.assertEqual(tts.condition_text(text, max_chars=None),
                         "line one\nline two")


class ConditionLayout(unittest.TestCase):
    """Box drawing and whitespace are decoration, not speech."""

    def test_run_of_three_or_more_box_characters_is_removed(self) -> None:
        self.assertEqual(tts.condition_text("a───b", max_chars=None), "ab")
        self.assertEqual(tts.condition_text("═" * 80, max_chars=None), "")

    def test_shorter_runs_survive(self) -> None:
        # Two in a row can still be meaningful, and so can three *different*
        # box characters — only a repeated one is a rule.
        self.assertEqual(tts.condition_text("a ── b", max_chars=None),
                         "a ── b")
        self.assertEqual(tts.condition_text("─│┌", max_chars=None), "─│┌")

    def test_a_framed_box_keeps_its_contents(self) -> None:
        text = "┌" + "─" * 10 + "┐\n│ hello    │\n└" + "─" * 10 + "┘"
        result = tts.condition_text(text, max_chars=None)
        self.assertIn("hello", result)
        self.assertNotIn("───", result)

    def test_whitespace_runs_collapse(self) -> None:
        self.assertEqual(tts.condition_text("a   \t  b", max_chars=None),
                         "a b")

    def test_blank_lines_collapse_and_the_edges_are_trimmed(self) -> None:
        text = "\n\n   hello   \n\n\n   world  \n\n"
        self.assertEqual(tts.condition_text(text, max_chars=None),
                         "hello\nworld")

    def test_a_screen_of_pure_decoration_keeps_only_its_corners(self) -> None:
        text = "\x1b[0m┌" + "─" * 40 + "┐\n│" + " " * 40 + "│\n"
        self.assertEqual(tts.condition_text(text, max_chars=None), "┌┐\n│ │")


class ConditionBudget(unittest.TestCase):
    """The character budget is applied to what would actually be spoken."""

    def test_truncation_appends_the_note(self) -> None:
        result = tts.condition_text("a" * 100, max_chars=10)
        self.assertEqual(result, "aaaaaaaaaa …truncated")
        self.assertEqual(tts.TRUNCATION_NOTE, " …truncated")

    def test_the_cut_is_rstripped_before_the_note(self) -> None:
        self.assertEqual(tts.condition_text("aaaa bbbb cccc", max_chars=5),
                         "aaaa …truncated")

    def test_text_exactly_at_the_limit_is_not_marked(self) -> None:
        self.assertEqual(tts.condition_text("abc", max_chars=3), "abc")
        self.assertNotIn(tts.TRUNCATION_NOTE,
                         tts.condition_text("abc", max_chars=4))

    def test_none_means_unlimited(self) -> None:
        text = "word " * 5000
        result = tts.condition_text(text, max_chars=None)
        self.assertEqual(len(result), len("word " * 5000) - 1)
        self.assertNotIn(tts.TRUNCATION_NOTE, result)

    def test_zero_budget_yields_nothing(self) -> None:
        self.assertEqual(tts.condition_text("anything", max_chars=0), "")

    def test_the_budget_counts_speech_not_decoration(self) -> None:
        # 200 characters of escape and rule, then eight of text: the budget is
        # spent on the text, so nothing is truncated.
        text = "\x1b[1;31m" + "─" * 200 + "\x1b[0m" + "greeting"
        self.assertEqual(tts.condition_text(text, max_chars=20), "greeting")


class ConditionAdversarial(unittest.TestCase):
    """Whatever a pane was showing, conditioning returns a string."""

    CASES = (
        "", "\x1b", "\x1b[", "\x1b[38;2;", "\x1b]", "\x1b_", "\x1b\\",
        "\x1b]0;unterminated title", "\x1b_Gf=100;dGVzdA", "\x1b\x1b\x1b[m",
        "\x9b", "\x9d", "\x1bP", "\x00\x00\x00", "\x7f\x7f", "\ud800lone",
        "\udce9 surrogateescape", "text\x1b", "\x1b[999999999m",
        "\n" * 1000, "\t" * 1000, "─" * 5000,
    )

    def test_never_raises(self) -> None:
        for case in self.CASES:
            for budget in (None, 0, 1, 40, 4000):
                with self.subTest(text=repr(case)[:32], budget=budget):
                    result = tts.condition_text(case, max_chars=budget)
                    self.assertIsInstance(result, str)

    def test_a_very_long_line_is_handled(self) -> None:
        line = ("\x1b[32m" + "x" * 4000 + "\x1b[0m") * 50
        result = tts.condition_text(line, max_chars=4000)
        self.assertTrue(result.startswith("x"))
        self.assertTrue(result.endswith(tts.TRUNCATION_NOTE))
        self.assertEqual(len(result),
                         4000 + len(tts.TRUNCATION_NOTE))

    def test_non_text_input_is_coerced_rather_than_rejected(self) -> None:
        self.assertEqual(tts.condition_text(None, max_chars=None), "")
        self.assertEqual(tts.condition_text(b"bytes\x1b[0m", max_chars=None),
                         "bytes")

    def test_lone_surrogates_are_removed(self) -> None:
        # A str carrying surrogateescape bytes cannot be encoded for the
        # engine's stdin, so it must not survive conditioning.
        result = tts.condition_text("caf\udce9 au lait", max_chars=None)
        self.assertEqual(result, "caf au lait")
        result.encode("utf-8")


class Chunking(unittest.TestCase):
    """SentenceChunker hands out clips as soon as they are complete."""

    def test_sentence_boundaries(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed("Hello world. Second one! Third?"),
                         ["Hello world.", "Second one!"])
        self.assertEqual(chunker.flush(), "Third?")

    def test_a_sentence_split_across_feeds(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed("Hello wo"), [])
        self.assertEqual(chunker.feed("rld. Next.\n"),
                         ["Hello world.", "Next."])
        self.assertEqual(chunker.flush(), "")

    def test_lines_are_clips_even_without_punctuation(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed("total 92\ndrwxr-xr-x 3 user\n"),
                         ["total 92", "drwxr-xr-x 3 user"])

    def test_abbreviations_and_decimals_do_not_split(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed("Dr. Smith arrived. Version 3.5 ships."),
                         ["Dr. Smith arrived."])
        self.assertEqual(chunker.flush(), "Version 3.5 ships.")

    def test_a_closing_quote_stays_with_its_sentence(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed('He said "Stop." Then silence.'),
                         ['He said "Stop."'])

    def test_flush_returns_the_trailing_partial_once(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed("no terminator here"), [])
        self.assertEqual(chunker.flush(), "no terminator here")
        self.assertEqual(chunker.flush(), "")

    def test_flush_of_whitespace_only_is_empty(self) -> None:
        chunker = tts.SentenceChunker()
        self.assertEqual(chunker.feed("   \n   \n"), [])
        self.assertEqual(chunker.flush(), "")

    def test_unpunctuated_text_is_cut_rather_than_buffered(self) -> None:
        # The failure this guards against is a hang, so it is bounded twice:
        # every clip is short enough to start playing, and what is left over is
        # smaller than one clip rather than growing with the input.
        chunker = tts.SentenceChunker()
        pieces = chunker.feed("x" * 10000)
        self.assertTrue(pieces)
        for piece in pieces:
            self.assertLessEqual(len(piece), tts.MAX_CHUNK_CHARS)
        self.assertLess(len(chunker.flush()), tts.MAX_CHUNK_CHARS)

    def test_a_long_unpunctuated_line_breaks_on_a_space(self) -> None:
        chunker = tts.SentenceChunker()
        pieces = chunker.feed("word " * 200)
        self.assertTrue(pieces)
        for piece in pieces:
            self.assertLessEqual(len(piece), tts.MAX_CHUNK_CHARS)
            self.assertTrue(piece.startswith("word"))
            self.assertTrue(piece.endswith("word"))

    def test_conditioned_output_feeds_straight_into_the_chunker(self) -> None:
        screen = ("\x1b[1;34m$\x1b[0m ls\n" + "─" * 30 + "\n"
                  + kitty_graphics(b"thumbnail") + "\nDone. Next.")
        chunker = tts.SentenceChunker()
        pieces = chunker.feed(tts.condition_text(screen, max_chars=None))
        self.assertEqual(pieces, ["$ ls", "Done."])
        self.assertEqual(chunker.flush(), "Next.")


class SynthCommand(unittest.TestCase):
    """build_synth_cmd inspects PATH and the config; it launches nothing."""

    def test_espeak_ng_is_preferred_over_espeak(self) -> None:
        with mock.patch.object(util, "which",
                               lambda name: f"/usr/bin/{name}"):
            self.assertEqual(tts.espeak_binary(), "/usr/bin/espeak-ng")
            self.assertEqual(
                tts.build_synth_cmd(None, voice="en-us", rate=170),
                ["/usr/bin/espeak-ng", "-b", "1", "-v", "en-us", "-s", "170",
                 "--stdout"])

    def test_plain_espeak_is_accepted(self) -> None:
        with mock.patch.object(
                util, "which",
                lambda name: "/usr/bin/espeak" if name == "espeak" else None):
            self.assertEqual(tts.espeak_binary(), "/usr/bin/espeak")

    def test_missing_synthesiser_is_actionable(self) -> None:
        with mock.patch.object(util, "which", lambda name: None):
            self.assertIsNone(tts.espeak_binary())
            with self.assertRaises(tts.TtsError) as caught:
                tts.build_synth_cmd({}, voice="en-us", rate=170)
        message = str(caught.exception)
        self.assertIn("espeak-ng", message)
        self.assertIn("apt install", message)
        self.assertIn(settings.KEY_TTS_ENGINE, message)

    def test_override_substitutes_voice_and_rate(self) -> None:
        cfg = {"tts": {"cmd": ["fake-engine", "-v", "{voice}",
                               "-s", "{rate}"]}}
        with mock.patch.object(util, "which", lambda name: None):
            self.assertEqual(
                tts.build_synth_cmd(cfg, voice="mb-en1", rate=200),
                ["fake-engine", "-v", "mb-en1", "-s", "200"])

    def test_override_must_be_a_list_of_strings(self) -> None:
        for bad in ("espeak-ng --stdout", ["espeak-ng", 7], 5):
            with self.subTest(override=bad):
                with self.assertRaises(tts.TtsError) as caught:
                    tts.build_synth_cmd({"tts": {"cmd": bad}}, voice="en-us",
                                        rate=170)
                self.assertIn("tts.cmd", str(caught.exception))

    def test_an_empty_override_means_auto_detection(self) -> None:
        # An empty list reads as "no override" rather than as an error, so the
        # failure a caller sees is the one about the missing engine.
        with mock.patch.object(util, "which", lambda name: None):
            with self.assertRaises(tts.TtsError) as caught:
                tts.build_synth_cmd({"tts": {"cmd": []}}, voice="en-us",
                                    rate=170)
        self.assertIn("neither espeak-ng nor espeak", str(caught.exception))


class _FakeProcess:
    """A Popen stand-in: nothing is executed, no device is written to."""

    def __init__(self, out: bytes, err: bytes, code: int) -> None:
        self._out = out
        self._err = err
        self.returncode = code
        self.stdin_text = b""

    def communicate(self, data: bytes | None = None,
                    timeout: float | None = None) -> tuple[bytes, bytes]:
        self.stdin_text = data or b""
        return self._out, self._err


class _FakeEngine:
    """Records the argv of every synthesis; fails the voices it is told to."""

    def __init__(self, *, fail: tuple[str, ...] = (),
                 pcm: bytes = b"", rate: int = 16000) -> None:
        self.fail = fail
        self.wav = util.write_wav(pcm, rate)
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_kwargs: object) -> _FakeProcess:
        self.calls.append(list(argv))
        voice = argv[argv.index("-v") + 1]
        if voice in self.fail:
            return _FakeProcess(b"", b"mbrowrap error: no voice mb-en-us", 1)
        return _FakeProcess(self.wav, b"", 0)

    @property
    def voices(self) -> list[str]:
        return [argv[argv.index("-v") + 1] for argv in self.calls]


class Synthesis(unittest.TestCase):
    """EspeakTts, exercised through the command seam with a fake engine."""

    CFG = {"tts": {"cmd": ["fake-engine", "-v", "{voice}", "-s", "{rate}",
                           "--stdout"]}}
    PCM = struct.pack("<4h", 0, 1000, 0, -1000) * 50

    def engine(self, **kwargs: object) -> _FakeEngine:
        fake = _FakeEngine(pcm=self.PCM, rate=16000, **kwargs)
        patcher = mock.patch.object(tts.subprocess, "Popen", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def test_synth_returns_pcm_and_the_rate_from_the_wav_header(self) -> None:
        fake = self.engine()
        engine = tts.EspeakTts(self.CFG, voice="en-us", rate=170)
        pcm, rate = engine.synth("hi")
        self.assertEqual(pcm, self.PCM)
        self.assertEqual(rate, 16000)
        self.assertEqual(fake.voices, ["en-us"])

    def test_empty_text_is_an_empty_clip_not_a_failure(self) -> None:
        fake = self.engine()
        engine = tts.EspeakTts(self.CFG, voice="en-us", rate=170)
        self.assertEqual(engine.synth("   \n  "),
                         (b"", tts.ESPEAK_SAMPLE_RATE))
        self.assertEqual(fake.calls, [])

    def test_mbrola_falls_back_to_plain_espeak_and_remembers(self) -> None:
        fake = self.engine(fail=("mb-en-us",))
        engine = tts.EspeakTts(self.CFG, voice="en-us", rate=170, mbrola=True)
        pcm, _rate = engine.synth("first")
        self.assertEqual(pcm, self.PCM)
        self.assertEqual(fake.voices, ["mb-en-us", "en-us"])
        self.assertIn("mbrola", engine.mbrola_error)
        # The rest of the page must not pay for the doomed process again.
        engine.synth("second")
        self.assertEqual(fake.voices, ["mb-en-us", "en-us", "en-us"])

    def test_a_failing_engine_raises_with_the_voice_to_check(self) -> None:
        self.engine(fail=("en-us",))
        engine = tts.EspeakTts(self.CFG, voice="en-us", rate=170)
        with self.assertRaises(tts.TtsError) as caught:
            engine.synth("hello")
        message = str(caught.exception)
        self.assertIn("--voices", message)
        self.assertIn("mbrowrap error", message)

    def test_output_that_is_not_a_wav_raises(self) -> None:
        fake = self.engine()
        fake.wav = b"not a wav at all"
        engine = tts.EspeakTts(self.CFG, voice="en-us", rate=170)
        with self.assertRaises(tts.TtsError) as caught:
            engine.synth("hello")
        self.assertIn("no usable audio", str(caught.exception))

    def test_an_invalid_voice_is_refused_before_anything_runs(self) -> None:
        for bad in ("en-us; rm -rf /", "", "-" * 40, "en us"):
            with self.subTest(voice=bad):
                with self.assertRaises(tts.TtsError) as caught:
                    tts.EspeakTts(self.CFG, voice=bad, rate=170)
                self.assertIn("A-Za-z0-9_+-", str(caught.exception))


class EngineSelection(unittest.TestCase):
    """make_tts dispatches on the shared settings file and probes nothing."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, "settings.conf")
        patcher = mock.patch.dict(os.environ,
                                  {"GPU_TERMINAL_SETTINGS_FILE": self.path})
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, engine: str) -> None:
        with open(self.path, "w", encoding="utf-8") as stream:
            stream.write(f"{settings.KEY_TTS_ENGINE}={engine}\n")

    def test_off_selects_the_silent_engine(self) -> None:
        self.write("off")
        engine = tts.make_tts()
        self.assertIsInstance(engine, tts.NullTts)
        self.assertEqual(engine.name, "null")
        self.assertEqual(engine.synth("anything"),
                         (b"", tts.ESPEAK_SAMPLE_RATE))

    def test_espeak_and_mbrola_select_the_espeak_engine(self) -> None:
        for setting, mbrola in (("espeak", False), ("mbrola", True)):
            with self.subTest(engine=setting):
                self.write(setting)
                engine = tts.make_tts()
                self.assertIsInstance(engine, tts.EspeakTts)
                self.assertEqual(engine.mbrola, mbrola)

    def test_an_unknown_engine_falls_back_to_the_default(self) -> None:
        self.write("festival")
        self.assertIsInstance(tts.make_tts(), tts.EspeakTts)

    def test_construction_does_not_require_a_synthesiser(self) -> None:
        # A missing espeak-ng must degrade the read when it is asked for, not
        # stop a TUI or the daemon from starting.
        self.write("espeak")
        with mock.patch.object(util, "which", lambda name: None):
            engine = tts.make_tts()
            with self.assertRaises(tts.TtsError):
                engine.synth("hello")


if __name__ == "__main__":
    unittest.main()
