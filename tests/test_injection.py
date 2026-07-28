"""The injection sanitiser, unit-tested.

`test_daemon.py` drives kilix-voiced as a black box, which cannot reach this
function: recognised text only appears when an STT engine produced some, and
the suite deliberately runs with no engine. So the one transformation standing
between a speech recogniser and a shell was covered by inspection alone. It is
covered here instead.

The guarantee under test is DESIGN.md safety rule 2 and 3: recognised text is
delivered with no trailing newline and no control characters. A regression here
does not crash anything — it silently types control bytes into a PTY.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parent.parent


def _load_daemon():
    """Import kilix-voiced, which has no .py suffix, as a module."""
    path = REPO / "kilix-voiced"
    loader = importlib.machinery.SourceFileLoader("kilix_voiced_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class CleanForInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = staticmethod(_load_daemon().clean_for_injection)

    def test_a_trailing_newline_is_removed(self):
        # The single most important case: a newline reaching the PTY is a
        # command the user never chose to run.
        self.assertEqual(self.clean("ls -la\n"), "ls -la")
        self.assertEqual(self.clean("ls -la\r\n"), "ls -la")
        self.assertEqual(self.clean("ls -la\n\n\n"), "ls -la")

    def test_an_embedded_newline_becomes_a_space(self):
        self.assertEqual(self.clean("first\nsecond"), "first second")
        self.assertEqual(self.clean("first\r\nsecond"), "first second")

    def test_no_output_ever_contains_a_newline(self):
        for text in ("a\nb", "\n", "\r\n\r\n", "x\n" * 50, "  "):
            with self.subTest(text=text):
                self.assertNotIn("\n", self.clean(text))
                self.assertNotIn("\r", self.clean(text))

    def test_c0_controls_are_stripped(self):
        for code in range(0x00, 0x20):
            char = chr(code)
            with self.subTest(code=code):
                self.assertNotIn(char, self.clean(f"a{char}b"))

    def test_del_and_c1_controls_are_stripped(self):
        for code in [0x7F] + list(range(0x80, 0xA0)):
            char = chr(code)
            with self.subTest(code=code):
                self.assertNotIn(char, self.clean(f"a{char}b"))

    def test_escape_introducers_are_defanged(self):
        # The ESC itself must go. What remains is inert literal text: harmless
        # to type, and visible to the user rather than silently interpreted.
        cleaned = self.clean("echo \x1b[31mred")
        self.assertNotIn("\x1b", cleaned)
        self.assertEqual(cleaned, "echo [31mred")

        for introducer in ("\x1b]0;title\x07", "\x1b_payload\x1b\\", "\x9b31m"):
            with self.subTest(introducer=introducer):
                out = self.clean(f"x{introducer}y")
                self.assertNotIn("\x1b", out)
                self.assertNotIn("\x9b", out)

    def test_lone_surrogates_are_stripped(self):
        # A recogniser should never emit these, but they raise
        # UnicodeEncodeError at the PTY boundary if one ever does.
        cleaned = self.clean("\ud800lone\udfff")
        self.assertEqual(cleaned, "lone")
        cleaned.encode("utf-8")  # must not raise

    def test_surrounding_whitespace_is_trimmed_and_runs_collapse(self):
        self.assertEqual(self.clean("  padded  "), "padded")
        self.assertEqual(self.clean("a \t  b"), "a b")

    def test_ordinary_text_survives_intact(self):
        # The sanitiser must not damage what people actually dictate.
        for text in ("git commit --amend", "echo 'hello world'",
                     "cd ~/projects && ls", "print(x[0], y['k'])",
                     "café naïve — em dash", "grep -rn 'a|b' ."):
            with self.subTest(text=text):
                self.assertEqual(self.clean(text), text)

    def test_empty_and_whitespace_only_input(self):
        self.assertEqual(self.clean(""), "")
        self.assertEqual(self.clean("   \n\t "), "")

    def test_is_idempotent(self):
        for text in ("a\nb", "echo \x1b[31mred", "  x  ", "plain"):
            with self.subTest(text=text):
                once = self.clean(text)
                self.assertEqual(self.clean(once), once)

    def test_non_string_input_does_not_raise(self):
        # The value arrives from parsed JSON, so it is not guaranteed to be str.
        self.assertEqual(self.clean(None), "None")
        self.assertEqual(self.clean(12), "12")

    def test_the_sanitiser_is_not_the_identity_function(self):
        # A guard against the sanitiser being reduced to `return text` during a
        # refactor: every assertion above would still pass for well-formed
        # input, so pin the transformation itself.
        hostile = "run\nthis\x00now\x1b[2J\x7f"
        self.assertNotEqual(self.clean(hostile), hostile)
        self.assertEqual(self.clean(hostile), "run this now [2J")


if __name__ == "__main__":
    unittest.main()
