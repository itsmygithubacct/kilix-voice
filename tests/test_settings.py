"""Offline tests for voicelib.settings.

The settings document is shared with the rest of the GPU Terminal stack, so the
two properties under test are: a value we do not recognise reads back as our
default (never coerced, never guessed), and a write leaves everything we do not
own — foreign keys, comments, layout — exactly as it found it.  Every test uses
a temporary file; the developer's real settings.conf is never opened.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import unittest

from voicelib import settings

# The settings vocabulary as DESIGN.md states it: key -> (default, choices).
# Kept as a literal so a drift between the table and SPEC fails a test rather
# than silently changing what a TUI offers.
DESIGN_TABLE: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "KILIX_CHROME_SPEAK": ("1", None),
    "KILIX_CHROME_DICTATE": ("1", None),
    "KILIX_VOICE_TTS_ENGINE": ("espeak", ("espeak", "mbrola", "off")),
    "KILIX_VOICE_TTS_VOICE": ("en-us", None),
    "KILIX_VOICE_TTS_RATE": ("170", ("120", "150", "170", "200", "240")),
    "KILIX_VOICE_TTS_EXTENT": ("screen", ("screen", "scrollback", "selection")),
    "KILIX_VOICE_TTS_MAX_CHARS": ("4000", ("1000", "4000", "16000", "unlimited")),
    "KILIX_VOICE_STT_ENGINE": ("vosk", ("vosk", "vibevoice", "off")),
    "KILIX_VOICE_STT_MODEL": ("small-en-us", ("small-en-us", "lgraph-en-us",
                                              "vibevoice-asr-bitnet")),
    "KILIX_VOICE_STT_SUBMIT": ("never", ("never", "confirm")),
    "KILIX_VOICE_STT_MAX_SECONDS": ("30", ("15", "30", "60", "120")),
    "KILIX_VOICE_STT_SILENCE_MS": ("900", ("500", "900", "1500")),
    "KILIX_VOICE_STT_PUNCTUATION": ("1", None),
    "KILIX_VOICE_DEVICE_IN": ("default", None),
    "KILIX_VOICE_DEVICE_OUT": ("default", None),
    "KILIX_VOICE_HISTORY": ("off", ("off", "on")),
}

# Legal values for the keys that have an alphabet instead of a vocabulary.
FREEFORM_SAMPLES: dict[str, tuple[str, ...]] = {
    "KILIX_VOICE_TTS_VOICE": ("en-us", "en-gb", "en_us+f3", "A" * 32),
    "KILIX_VOICE_DEVICE_IN": ("default",
                              "alsa_input.pci-0000_00_1f.3.analog-stereo"),
    "KILIX_VOICE_DEVICE_OUT": ("default",
                               "alsa_output.usb-Generic_USB_Audio-00.iec958"),
}

BOOL_SAMPLES = (("1", "1"), ("0", "0"), (True, "1"), (False, "0"),
                ("yes", "1"), ("off", "0"), ("TRUE", "1"), ("disabled", "0"))


class SettingsFileTestCase(unittest.TestCase):
    """Fixture owning one temporary settings document."""

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="kilix-voice-settings-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "settings.conf")

    def write(self, text: str) -> None:
        with open(self.path, "w", encoding="utf-8") as stream:
            stream.write(text)

    def read(self) -> str:
        with open(self.path, encoding="utf-8") as stream:
            return stream.read()

    def mode(self) -> int:
        return stat.S_IMODE(os.stat(self.path).st_mode)


class SpecTestCase(unittest.TestCase):

    def test_spec_matches_the_design_table(self) -> None:
        self.assertEqual(settings.SPEC, DESIGN_TABLE)

    def test_defaults_are_the_documented_defaults(self) -> None:
        expected = {key: default for key, (default, _) in DESIGN_TABLE.items()}
        self.assertEqual(settings.defaults(), expected)

    def test_defaults_returns_a_fresh_mapping(self) -> None:
        first = settings.defaults()
        first["KILIX_VOICE_TTS_RATE"] = "999"
        self.assertEqual(settings.defaults()["KILIX_VOICE_TTS_RATE"], "170")

    def test_submit_has_no_always_value(self) -> None:
        # Load-bearing safety rule: dictation never presses Enter for you.
        _default, choices = settings.SPEC[settings.KEY_STT_SUBMIT]
        self.assertNotIn("always", choices)

    def test_bool_keys_have_no_vocabulary(self) -> None:
        for key in settings.BOOL_KEYS:
            with self.subTest(key=key):
                self.assertIsNone(settings.SPEC[key][1])

    def test_every_default_is_a_legal_value(self) -> None:
        for key, (default, choices) in settings.SPEC.items():
            with self.subTest(key=key):
                if choices is not None:
                    self.assertIn(default, choices)


class TruthyTestCase(unittest.TestCase):

    def test_documented_false_words(self) -> None:
        for raw in ("", "0", "no", "false", "off", "disabled",
                    "NO", "False", "OFF", "Disabled", "  off  "):
            with self.subTest(raw=raw):
                self.assertFalse(settings.truthy(raw))

    def test_everything_else_is_true(self) -> None:
        for raw in ("1", "yes", "true", "on", "enabled", "2", "maybe", True, 1):
            with self.subTest(raw=raw):
                self.assertTrue(settings.truthy(raw))


class ParseTestCase(SettingsFileTestCase):

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        values = settings.parse_text(
            "# a comment\n\n   \nKILIX_VOICE_TTS_RATE=200\n# KILIX_X=1\n")
        self.assertEqual(values, {"KILIX_VOICE_TTS_RATE": "200"})

    def test_values_are_stripped(self) -> None:
        values = settings.parse_text("KILIX_VOICE_TTS_RATE=  200  \n")
        self.assertEqual(values["KILIX_VOICE_TTS_RATE"], "200")

    def test_last_assignment_wins(self) -> None:
        values = settings.parse_text(
            "KILIX_VOICE_TTS_RATE=120\nKILIX_VOICE_TTS_RATE=240\n")
        self.assertEqual(values["KILIX_VOICE_TTS_RATE"], "240")

    def test_values_are_literal_not_shell(self) -> None:
        # The document is parsed, never executed: no quoting, no expansion, and
        # no inline comment stripping.
        values = settings.parse_text('KILIX_VOICE_DEVICE_IN=$HOME # note\n')
        self.assertEqual(values["KILIX_VOICE_DEVICE_IN"], "$HOME # note")

    def test_malformed_lines_are_skipped(self) -> None:
        values = settings.parse_text("no equals here\n=orphan\n9BAD=1\nOK=2\n")
        self.assertEqual(values, {"OK": "2"})

    def test_load_of_a_missing_file_is_the_defaults(self) -> None:
        self.assertEqual(settings.load(self.path), settings.defaults())

    def test_load_of_an_unreadable_path_is_the_defaults(self) -> None:
        self.assertEqual(settings.load(self.dir), settings.defaults())

    def test_load_keeps_foreign_keys(self) -> None:
        self.write("KILIX_THEME=dark\nKILIX_VOICE_TTS_RATE=240\n")
        values = settings.load(self.path)
        self.assertEqual(values["KILIX_THEME"], "dark")
        self.assertEqual(values["KILIX_VOICE_TTS_RATE"], "240")
        self.assertEqual(values["KILIX_VOICE_STT_SUBMIT"], "never")

    def test_load_returns_raw_values(self) -> None:
        # Validation belongs to value(); load() is the unfiltered snapshot.
        self.write("KILIX_VOICE_TTS_RATE=999\n")
        self.assertEqual(settings.load(self.path)["KILIX_VOICE_TTS_RATE"], "999")

    def test_undecodable_bytes_do_not_break_the_read(self) -> None:
        with open(self.path, "wb") as stream:
            stream.write(b"KILIX_VOICE_TTS_RATE=240\nKILIX_JUNK=\xff\xfe\n")
        self.assertEqual(settings.value(settings.KEY_TTS_RATE, self.path), "240")


class ValueTestCase(SettingsFileTestCase):

    def test_missing_file_yields_every_default(self) -> None:
        for key, (default, _choices) in settings.SPEC.items():
            with self.subTest(key=key):
                self.assertEqual(settings.value(key, self.path), default)

    def test_recognised_values_are_returned(self) -> None:
        self.write("KILIX_VOICE_TTS_ENGINE=mbrola\n"
                   "KILIX_VOICE_TTS_RATE=240\n"
                   "KILIX_VOICE_STT_SUBMIT=confirm\n")
        self.assertEqual(settings.tts_engine(self.path), "mbrola")
        self.assertEqual(settings.tts_rate(self.path), 240)
        self.assertEqual(settings.stt_submit(self.path), "confirm")

    def test_choices_match_case_insensitively(self) -> None:
        self.write("KILIX_VOICE_TTS_EXTENT=ScrollBack\n"
                   "KILIX_VOICE_STT_ENGINE=VOSK\n")
        self.assertEqual(settings.tts_extent(self.path), "scrollback")
        self.assertEqual(settings.stt_engine(self.path), "vosk")

    def test_unrecognised_value_falls_back_to_the_default(self) -> None:
        cases = {
            # A near miss is never rounded to the nearest legal preset.
            "KILIX_VOICE_TTS_RATE": ("175", "170"),
            "KILIX_VOICE_STT_MAX_SECONDS": ("45", "30"),
            "KILIX_VOICE_STT_SILENCE_MS": ("1000", "900"),
            "KILIX_VOICE_TTS_MAX_CHARS": ("0", "4000"),
            # A value from a newer or different stack is not guessed at.
            "KILIX_VOICE_TTS_ENGINE": ("festival", "espeak"),
            "KILIX_VOICE_STT_ENGINE": ("whisper", "vosk"),
            "KILIX_VOICE_STT_MODEL": ("large-en-us", "small-en-us"),
            "KILIX_VOICE_TTS_EXTENT": ("everything", "screen"),
            "KILIX_VOICE_HISTORY": ("yes", "off"),
            # An empty assignment is not a value.
            "KILIX_VOICE_STT_SUBMIT": ("", "never"),
        }
        for key, (written, expected) in cases.items():
            with self.subTest(key=key):
                self.write(f"{key}={written}\n")
                self.assertEqual(settings.value(key, self.path), expected)

    def test_submit_never_becomes_always(self) -> None:
        # The one setting where a fallback is a safety property, not a nicety.
        for written in ("always", "auto", "1", "enter", "yes"):
            with self.subTest(written=written):
                self.write(f"KILIX_VOICE_STT_SUBMIT={written}\n")
                self.assertEqual(settings.stt_submit(self.path), "never")

    def test_history_is_off_unless_it_says_on(self) -> None:
        # "yes" is truthy but is not a member of the vocabulary, so it must
        # fall back to "off" rather than switching recording on.
        for written, expected in (("on", True), ("off", False), ("yes", False),
                                  ("true", False), ("1", False)):
            with self.subTest(written=written):
                self.write(f"KILIX_VOICE_HISTORY={written}\n")
                self.assertIs(settings.history_enabled(self.path), expected)

    def test_freeform_values_are_held_to_an_alphabet(self) -> None:
        cases = {
            "KILIX_VOICE_TTS_VOICE": ("en us", "en-us"),
            "KILIX_VOICE_DEVICE_IN": ("$(rm -rf ~)", "default"),
            "KILIX_VOICE_DEVICE_OUT": ("sink; reboot", "default"),
        }
        for key, (written, expected) in cases.items():
            with self.subTest(key=key):
                self.write(f"{key}={written}\n")
                self.assertEqual(settings.value(key, self.path), expected)

    def test_freeform_length_limits(self) -> None:
        self.write("KILIX_VOICE_TTS_VOICE=" + "a" * 33 + "\n")
        self.assertEqual(settings.tts_voice(self.path), "en-us")
        self.write("KILIX_VOICE_DEVICE_IN=" + "a" * 129 + "\n")
        self.assertEqual(settings.device_in(self.path), "default")

    def test_junk_never_escapes_a_vocabulary(self) -> None:
        junk = ("", " ", "-1", "None", "null", "always", "/etc/passwd",
                "a" * 200, "1;reboot", "ON\n")
        for key, (default, choices) in settings.SPEC.items():
            if choices is None:
                continue
            for written in junk:
                with self.subTest(key=key, written=written):
                    self.write(f"{key}={written}\n")
                    got = settings.value(key, self.path)
                    self.assertIn(got, choices)
                    if written.strip().lower() not in choices:
                        self.assertEqual(got, default)

    def test_booleans_read_the_shared_false_words(self) -> None:
        for key in sorted(settings.BOOL_KEYS):
            for written, expected in (("1", True), ("0", False), ("", False),
                                      ("no", False), ("false", False),
                                      ("off", False), ("disabled", False),
                                      ("yes", True), ("true", True),
                                      ("on", True)):
                with self.subTest(key=key, written=written):
                    self.write(f"{key}={written}\n")
                    self.assertIs(settings.enabled(key, self.path), expected)

    def test_unknown_key_is_a_settings_error(self) -> None:
        for key in ("KILIX_VOICE_NOT_A_KEY", "", "kilix_voice_tts_rate"):
            with self.subTest(key=key):
                with self.assertRaises(settings.SettingsError) as caught:
                    settings.value(key, self.path)
                self.assertIn("unknown kilix-voice setting",
                              str(caught.exception))

    def test_reading_never_rewrites_the_file(self) -> None:
        original = "KILIX_VOICE_TTS_RATE=999\n# hand written\n"
        self.write(original)
        for key in settings.SPEC:
            settings.value(key, self.path)
        self.assertEqual(self.read(), original)


class AccessorTestCase(SettingsFileTestCase):

    def test_numeric_accessors_return_ints(self) -> None:
        self.write("KILIX_VOICE_TTS_RATE=120\n"
                   "KILIX_VOICE_STT_MAX_SECONDS=120\n"
                   "KILIX_VOICE_STT_SILENCE_MS=1500\n"
                   "KILIX_VOICE_TTS_MAX_CHARS=16000\n")
        self.assertEqual(settings.tts_rate(self.path), 120)
        self.assertEqual(settings.stt_max_seconds(self.path), 120)
        self.assertEqual(settings.stt_silence_ms(self.path), 1500)
        self.assertEqual(settings.tts_max_chars(self.path), 16000)

    def test_unlimited_max_chars_is_none(self) -> None:
        self.write("KILIX_VOICE_TTS_MAX_CHARS=unlimited\n")
        self.assertIsNone(settings.tts_max_chars(self.path))

    def test_numeric_accessors_survive_junk(self) -> None:
        self.write("KILIX_VOICE_TTS_RATE=fast\n"
                   "KILIX_VOICE_STT_MAX_SECONDS=forever\n"
                   "KILIX_VOICE_STT_SILENCE_MS=-1\n"
                   "KILIX_VOICE_TTS_MAX_CHARS=all\n")
        self.assertEqual(settings.tts_rate(self.path), 170)
        self.assertEqual(settings.stt_max_seconds(self.path), 30)
        self.assertEqual(settings.stt_silence_ms(self.path), 900)
        self.assertEqual(settings.tts_max_chars(self.path), 4000)

    def test_chrome_toggles_default_on(self) -> None:
        self.assertTrue(settings.enabled(settings.KEY_SPEAK, self.path))
        self.assertTrue(settings.enabled(settings.KEY_DICTATE, self.path))
        self.assertTrue(settings.enabled(settings.KEY_STT_PUNCTUATION,
                                         self.path))

    def test_history_defaults_off(self) -> None:
        self.assertFalse(settings.history_enabled(self.path))


class UpdateTestCase(SettingsFileTestCase):

    def test_creates_the_document_when_absent(self) -> None:
        returned = settings.update({settings.KEY_TTS_RATE: "240"}, self.path)
        self.assertEqual(returned, self.path)
        text = self.read()
        self.assertIn(settings.SETTINGS_HEADER, text)
        self.assertIn(settings.SETTINGS_MARKER, text)
        self.assertIn("KILIX_VOICE_TTS_RATE=240", text)
        self.assertEqual(settings.tts_rate(self.path), 240)

    def test_creates_a_missing_directory(self) -> None:
        nested = os.path.join(self.dir, "deeper", "settings.conf")
        settings.update({settings.KEY_TTS_RATE: "150"}, nested)
        self.assertTrue(os.path.isfile(nested))
        self.assertEqual(stat.S_IMODE(os.stat(nested).st_mode), 0o600)

    def test_written_file_is_private(self) -> None:
        previous = os.umask(0o000)
        self.addCleanup(os.umask, previous)
        settings.update({settings.KEY_TTS_RATE: "150"}, self.path)
        self.assertEqual(self.mode(), 0o600)

    def test_tightens_a_world_readable_file(self) -> None:
        self.write("KILIX_VOICE_TTS_RATE=120\n")
        os.chmod(self.path, 0o644)
        settings.update({settings.KEY_TTS_RATE: "150"}, self.path)
        self.assertEqual(self.mode(), 0o600)

    def test_preserves_foreign_keys_and_comments(self) -> None:
        original = (
            "# GPU Terminal shared settings (KEY=value; not shell code).\n"
            "# Managed by the Kilix SDK. Hand edits are preserved.\n"
            "\n"
            "# -- Kilix chrome --\n"
            "KILIX_THEME=dark\n"
            "KILIX_FONT_SIZE=13\n"
            "KILIX_MOTD=hello # not a comment, part of the value\n"
            "\n"
            "# -- Kilix voice --\n"
            "KILIX_VOICE_TTS_RATE=120\n"
            "\n"
            "# trailing note\n")
        self.write(original)
        settings.update({settings.KEY_TTS_RATE: "200"}, self.path)
        expected = original.replace("KILIX_VOICE_TTS_RATE=120",
                                    "KILIX_VOICE_TTS_RATE=200")
        self.assertEqual(self.read(), expected)

    def test_preserves_blank_lines_around_an_edited_key(self) -> None:
        # Regression: a whitespace-swallowing match would eat the blank line
        # separating our section from the one above it.
        original = ("# header\n\nKILIX_THEME=dark\n\nKILIX_VOICE_TTS_RATE=120\n"
                    "\n# tail\n")
        self.write(original)
        settings.update({settings.KEY_TTS_RATE: "240"}, self.path)
        self.assertEqual(self.read(),
                         original.replace("RATE=120", "RATE=240"))

    def test_appends_new_keys_under_one_marker(self) -> None:
        self.write("# GPU Terminal shared settings.\nKILIX_THEME=dark\n")
        settings.update({settings.KEY_TTS_RATE: "150"}, self.path)
        settings.update({settings.KEY_STT_SUBMIT: "confirm"}, self.path)
        text = self.read()
        self.assertEqual(text.count(settings.SETTINGS_MARKER), 1)
        self.assertIn("KILIX_THEME=dark", text)
        self.assertEqual(settings.tts_rate(self.path), 150)
        self.assertEqual(settings.stt_submit(self.path), "confirm")

    def test_edits_the_effective_assignment_of_a_duplicated_key(self) -> None:
        self.write("KILIX_VOICE_TTS_RATE=120\nKILIX_THEME=dark\n"
                   "KILIX_VOICE_TTS_RATE=150\n")
        settings.update({settings.KEY_TTS_RATE: "240"}, self.path)
        self.assertEqual(self.read(),
                         "KILIX_VOICE_TTS_RATE=120\nKILIX_THEME=dark\n"
                         "KILIX_VOICE_TTS_RATE=240\n")
        self.assertEqual(settings.tts_rate(self.path), 240)

    def test_applies_several_changes_at_once(self) -> None:
        settings.update({settings.KEY_TTS_ENGINE: "mbrola",
                         settings.KEY_TTS_VOICE: "en-gb",
                         settings.KEY_SPEAK: False}, self.path)
        self.assertEqual(settings.tts_engine(self.path), "mbrola")
        self.assertEqual(settings.tts_voice(self.path), "en-gb")
        self.assertFalse(settings.enabled(settings.KEY_SPEAK, self.path))

    def test_every_documented_key_round_trips(self) -> None:
        for key, (_default, choices) in settings.SPEC.items():
            if key in settings.BOOL_KEYS:
                samples = BOOL_SAMPLES
            elif choices is not None:
                samples = tuple((choice, choice) for choice in choices)
                samples += ((choices[0].upper(), choices[0]),)
            else:
                samples = tuple((raw, raw) for raw in FREEFORM_SAMPLES[key])
            for written, expected in samples:
                with self.subTest(key=key, written=written):
                    settings.update({key: written}, self.path)
                    self.assertEqual(settings.value(key, self.path), expected)

    def test_the_whole_vocabulary_survives_one_write(self) -> None:
        changes = {key: (choices[-1] if choices else FREEFORM_SAMPLES[key][-1])
                   for key, (_default, choices) in settings.SPEC.items()
                   if key not in settings.BOOL_KEYS}
        changes.update({key: False for key in settings.BOOL_KEYS})
        settings.update(changes, self.path)
        for key, written in changes.items():
            with self.subTest(key=key):
                expected = "0" if key in settings.BOOL_KEYS else written
                self.assertEqual(settings.value(key, self.path), expected)

    def test_invalid_value_is_refused_before_anything_is_written(self) -> None:
        self.write("KILIX_VOICE_TTS_RATE=120\n")
        cases = {
            settings.KEY_TTS_RATE: "175",
            settings.KEY_STT_SUBMIT: "always",
            settings.KEY_TTS_ENGINE: "festival",
            settings.KEY_TTS_VOICE: "en us",
            settings.KEY_DEVICE_IN: "$(reboot)",
            settings.KEY_HISTORY: "yes",
        }
        for key, written in cases.items():
            with self.subTest(key=key):
                with self.assertRaises(settings.SettingsError) as caught:
                    settings.update({key: written}, self.path)
                self.assertIn(key, str(caught.exception))
                self.assertEqual(self.read(), "KILIX_VOICE_TTS_RATE=120\n")

    def test_a_rejected_change_blocks_the_whole_batch(self) -> None:
        with self.assertRaises(settings.SettingsError):
            settings.update({settings.KEY_TTS_RATE: "150",
                             settings.KEY_STT_SUBMIT: "always"}, self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_unknown_key_is_refused(self) -> None:
        with self.assertRaises(settings.SettingsError) as caught:
            settings.update({"KILIX_VOICE_NOT_A_KEY": "1"}, self.path)
        self.assertIn("unknown kilix-voice setting", str(caught.exception))
        self.assertFalse(os.path.exists(self.path))

    def test_refuses_to_write_through_a_symlink(self) -> None:
        real = os.path.join(self.dir, "real.conf")
        with open(real, "w", encoding="utf-8") as stream:
            stream.write("KILIX_VOICE_TTS_RATE=120\n")
        os.symlink(real, self.path)
        with self.assertRaises(settings.SettingsError) as caught:
            settings.update({settings.KEY_TTS_RATE: "240"}, self.path)
        self.assertIn("symlink", str(caught.exception))
        self.assertTrue(os.path.islink(self.path))
        with open(real, encoding="utf-8") as stream:
            self.assertEqual(stream.read(), "KILIX_VOICE_TTS_RATE=120\n")

    def test_leaves_no_temporary_files_behind(self) -> None:
        settings.update({settings.KEY_TTS_RATE: "150"}, self.path)
        settings.update({settings.KEY_TTS_RATE: "200"}, self.path)
        self.assertEqual(os.listdir(self.dir), ["settings.conf"])

    def test_an_unwritable_directory_is_reported(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        readonly = os.path.join(self.dir, "readonly")
        os.mkdir(readonly, 0o500)
        self.addCleanup(os.chmod, readonly, 0o700)
        with self.assertRaises(settings.SettingsError) as caught:
            settings.update({settings.KEY_TTS_RATE: "150"},
                            os.path.join(readonly, "settings.conf"))
        self.assertIn("cannot write", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
