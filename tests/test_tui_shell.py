"""The launchable voice screens share the VirtualBox-manager text shell."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import sys
import unittest

from voicelib import settings, tui_shell


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Screen:
    def __init__(self, height: int = 24, width: int = 100) -> None:
        self.height = height
        self.width = width
        self.lines = [" " * width for _ in range(height)]

    def getmaxyx(self) -> tuple[int, int]:
        return self.height, self.width

    def erase(self) -> None:
        self.lines = [" " * self.width for _ in range(self.height)]

    def refresh(self) -> None:
        pass

    def addnstr(
        self, row: int, column: int, text: str, count: int, attr: int = 0,
    ) -> None:
        del attr
        clipped = text[:count][:max(0, self.width - column)]
        line = self.lines[row]
        self.lines[row] = (
            line[:column] + clipped + line[column + len(clipped):]
        )

    def text(self) -> str:
        return "\n".join(line.rstrip() for line in self.lines)


def load_script(filename: str, module_name: str):
    loader = importlib.machinery.SourceFileLoader(
        module_name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise AssertionError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


class VoiceShellTests(unittest.TestCase):
    def setUp(self) -> None:
        tui_shell.reset()

    def assert_shell(self, screen: Screen, strap: str) -> None:
        lines = screen.text().splitlines()
        self.assertIn("KILIX TUI", lines[0])
        self.assertIn(strap, lines[0])
        self.assertIn("▶1 ", lines[1])
        self.assertTrue(lines[2].startswith("─"))
        self.assertTrue(lines[3].strip())
        self.assertNotIn(" // ", screen.text())

    def test_read_aloud_screen_uses_canonical_shell(self) -> None:
        module = load_script("kilix-tts", "kilix_tts_tui_test")
        values = {
            field.key: module._default_value(field)
            for field in module.FIELDS
        }
        status = module.Status(
            True, "running", True, "espeak ready", True, "default sink")
        screen = Screen()

        module._draw(screen, values, 0, status, "", 0, 0, "")

        self.assert_shell(screen, "Kilix Voice · Read Aloud")

    def test_suspended_source_is_explained_over_the_level_meter(self) -> None:
        # A suspended source is the steady state of an input nobody records
        # from, and the truth behind a meter that opens and stays at silence.
        # The screen must say so instead of showing an unexplained flat bar.
        module = load_script("kilix-stt", "kilix_stt_suspended_test")
        ui = self._dictation_ui(module)
        ui._section = module.SECTION_MICROPHONE
        ui._pulse = module.PulseState(
            (module.Source(
                "alsa_input.pci-0000_00_1f.3.analog-stereo", False,
                "SUSPENDED"),),
            "alsa_input.pci-0000_00_1f.3.analog-stereo", "")

        ui._draw()

        text = ui._screen.text()
        self.assertIn("is suspended", text)
        self.assertIn("asleep because nothing", text)

    def test_pactl_source_states_are_kept_not_discarded(self) -> None:
        module = load_script("kilix-stt", "kilix_stt_states_test")
        listing = (
            "0\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\t"
            "module-alsa-card.c\ts16le 2ch 48000Hz\tIDLE\n"
            "1\talsa_input.pci-0000_00_1f.3.analog-stereo\t"
            "module-alsa-card.c\ts16le 2ch 48000Hz\tSUSPENDED\n"
        )

        def fake_run_tool(argv):
            if argv[:2] == ["pactl", "list"]:
                return listing
            return "Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo\n"

        original = module._run_tool
        module._run_tool = fake_run_tool
        try:
            state = module.pulse_state()
        finally:
            module._run_tool = original

        # Real inputs still come first, and each source carries the server's
        # word for it so the screen can explain a silent meter.
        self.assertEqual(
            [source.state for source in state.sources],
            ["SUSPENDED", "IDLE"])
        self.assertEqual(state.sources[0].monitor, False)
        self.assertEqual(state.sources[1].monitor, True)

    def _dictation_ui(self, module):
        ui = object.__new__(module.Ui)
        ui._screen = Screen()
        ui._glyphs = module.UNICODE_GLYPHS
        ui._section = module.SECTION_DICTATION
        ui._values = {
            control.key: (
                "1" if settings.truthy(settings.SPEC[control.key][0]) else "0"
            ) if control.key in settings.BOOL_KEYS else str(
                settings.SPEC[control.key][0]
            )
            for control in module.CONTROLS
        }
        ui._original = dict(ui._values)
        ui._selected = [0] * len(module.SECTIONS)
        ui._message = ""
        ui._discard_armed = False
        ui._pulse = module.PulseState((), "", "")
        ui._daemon = None
        ui._diagnostics = module.Diagnostics(
            True, "parec", True, "libvosk", True, "model", "")
        ui._mic = None
        return ui

    def test_dictation_screen_uses_canonical_shell(self) -> None:
        module = load_script("kilix-stt", "kilix_stt_tui_test")
        ui = object.__new__(module.Ui)
        ui._screen = Screen()
        ui._glyphs = module.UNICODE_GLYPHS
        ui._section = module.SECTION_DICTATION
        ui._values = {
            control.key: (
                "1" if settings.truthy(settings.SPEC[control.key][0]) else "0"
            ) if control.key in settings.BOOL_KEYS else str(
                settings.SPEC[control.key][0]
            )
            for control in module.CONTROLS
        }
        ui._original = dict(ui._values)
        ui._selected = [0] * len(module.SECTIONS)
        ui._message = ""
        ui._discard_armed = False
        ui._pulse = module.PulseState((), "", "")
        ui._daemon = None
        ui._diagnostics = module.Diagnostics(
            True, "parec", True, "libvosk", True, "model", "")
        ui._mic = None

        ui._draw()

        self.assert_shell(ui._screen, "Kilix Voice · Dictation")


if __name__ == "__main__":
    unittest.main()
