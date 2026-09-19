"""Bounded parsing checks for kilix-voiced's optional JSON config.

The override file is read by voicelib.daemon_config.load_overrides, which
kilix-voiced and ``kilix-stt --grant-consent`` share, so the bounds live there
and both commands get them. kilix-voiced reports the ConfigError as a
DaemonError.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_daemon_tool():
    loader = importlib.machinery.SourceFileLoader(
        "kilix_voiced_config_test", str(ROOT / "kilix-voiced"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise AssertionError("could not load kilix-voiced")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


tool = load_daemon_tool()

from voicelib import daemon_config  # noqa: E402


def _document_of_size(size: int) -> bytes:
    """A valid JSON object of exactly ``size`` bytes."""
    shell = b'{"pad": ""}'
    return shell[:-2] + b"x" * (size - len(shell)) + shell[-2:]


def _nested(levels: int) -> str:
    """An object whose deepest container sits ``levels`` levels down."""
    return '{"daemon":' + "[" * (levels - 1) + "]" * (levels - 1) + "}"


class ConfigParsingTests(unittest.TestCase):
    def write(self, root: str, name: str, data: bytes) -> str:
        path = pathlib.Path(root) / name
        path.write_bytes(data)
        return str(path)

    def test_oversized_config_is_refused_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "oversized.json", _document_of_size(
                daemon_config.MAX_CONFIG_BYTES + 1))
            with mock.patch.object(daemon_config.json, "loads") as loads:
                with self.assertRaisesRegex(daemon_config.ConfigError,
                                            "larger than"):
                    daemon_config.load_overrides(path)
            loads.assert_not_called()

    def test_the_read_itself_is_bounded(self) -> None:
        # Refusing after an unbounded read() would still say "larger than",
        # having first taken the whole file (or all of /dev/zero) into memory.
        sizes: list[int] = []
        real_open = open

        def spy_open(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            real_read = handle.read

            def read(size=-1):
                sizes.append(size)
                return real_read(size)

            handle.read = read
            return handle

        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "oversized.json", _document_of_size(
                daemon_config.MAX_CONFIG_BYTES + 1))
            with mock.patch.object(daemon_config, "open", spy_open,
                                   create=True):
                with self.assertRaises(daemon_config.ConfigError):
                    daemon_config.load_overrides(path)
        self.assertEqual(sizes, [daemon_config.MAX_CONFIG_BYTES + 1])

    def test_config_at_the_size_bound_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "bound.json", _document_of_size(
                daemon_config.MAX_CONFIG_BYTES))
            self.assertIn("pad", daemon_config.load_overrides(path))

    def test_excessive_nesting_is_a_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "nested.json", _nested(
                daemon_config.MAX_CONFIG_DEPTH + 1).encode())
            with self.assertRaisesRegex(daemon_config.ConfigError,
                                        "nested more than"):
                daemon_config.load_overrides(path)

    def test_nesting_at_the_depth_bound_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "deep.json", _nested(
                daemon_config.MAX_CONFIG_DEPTH).encode())
            self.assertIn("daemon", daemon_config.load_overrides(path))

    def test_nesting_past_the_parsers_recursion_limit_is_a_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "recursion.json", b"[" * 200000)
            with self.assertRaisesRegex(daemon_config.ConfigError,
                                        "not valid JSON"):
                daemon_config.load_overrides(path)

    def test_bytes_that_are_not_utf8_are_a_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "latin1.json", b'{"daemon": "\xff"}')
            with self.assertRaisesRegex(daemon_config.ConfigError,
                                        "not valid JSON"):
                daemon_config.load_overrides(path)

    def test_small_object_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "valid.json",
                              b'{"daemon": {"idle_seconds": 12}}')
            self.assertEqual(daemon_config.load_overrides(path),
                             {"daemon": {"idle_seconds": 12}})

    def test_the_daemon_reports_an_oversized_config_as_a_daemon_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, "oversized.json", _document_of_size(
                daemon_config.MAX_CONFIG_BYTES + 1))
            with self.assertRaisesRegex(tool.DaemonError, "larger than"):
                tool.Daemon(config_path=path)

    def test_grant_consent_reports_an_oversized_config_in_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.write(tmp, "voiced.json", _document_of_size(
                daemon_config.MAX_CONFIG_BYTES + 1))
            env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "LANG": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1",
                   "GPU_TERMINAL_SETTINGS_FILE": os.path.join(tmp, "settings.conf"),
                   "KILIX_DATA_HOME": os.path.join(tmp, "data"),
                   "KILIX_VOICE_CONFIG": config}
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "kilix-stt"), "--grant-consent"],
                env=env, capture_output=True, text=True, timeout=60,
                stdin=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("larger than", proc.stderr)
        self.assertEqual(len(proc.stderr.strip().splitlines()), 1, proc.stderr)


class IdleTimeoutTests(unittest.TestCase):
    def test_idle_timeout_is_finite_and_non_negative(self) -> None:
        self.assertEqual(tool._idle_timeout(0), 0.0)
        self.assertEqual(tool._idle_timeout("12.5"), 12.5)
        for raw in ("forever", None, -1, float("nan"), float("inf")):
            with self.subTest(raw=raw), self.assertRaises(tool.DaemonError):
                tool._idle_timeout(raw)

    def test_a_non_finite_idle_timeout_in_the_config_stops_startup(self) -> None:
        # json accepts the NaN and Infinity literals. Either one would leave
        # the idle check comparing against a value no elapsed time can reach.
        for literal in ("NaN", "Infinity", "-1", '"forever"'):
            with self.subTest(literal=literal), \
                    tempfile.TemporaryDirectory() as root:
                path = pathlib.Path(root) / "voiced.json"
                path.write_text('{"daemon": {"idle_seconds": %s}}' % literal,
                                encoding="utf-8")
                self.assertIsInstance(
                    json.loads(path.read_text(encoding="utf-8")), dict)
                with self.assertRaisesRegex(tool.DaemonError,
                                            "daemon.idle_seconds"):
                    tool.Daemon(config_path=str(path))


if __name__ == "__main__":
    unittest.main()
