"""Bounded parsing checks for kilix-voiced's optional JSON config."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import sys
import tempfile
import unittest


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


class ConfigParsingTests(unittest.TestCase):
    def test_oversized_config_is_refused_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "oversized.json"
            path.write_bytes(b" " * (tool.MAX_CONFIG_BYTES + 1))
            with self.assertRaisesRegex(tool.DaemonError, "larger than"):
                tool._load_overrides(str(path))

    def test_excessive_nesting_is_a_daemon_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "nested.json"
            nested = '{"daemon":' + "[" * 100 + "]" * 100 + "}"
            path.write_text(nested, encoding="utf-8")
            with self.assertRaisesRegex(tool.DaemonError, "nested more than"):
                tool._load_overrides(str(path))

    def test_small_object_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "valid.json"
            path.write_text('{"daemon": {"idle_seconds": 12}}',
                            encoding="utf-8")
            self.assertEqual(tool._load_overrides(str(path)),
                             {"daemon": {"idle_seconds": 12}})

    def test_idle_timeout_is_finite_and_non_negative(self) -> None:
        self.assertEqual(tool._idle_timeout(0), 0.0)
        self.assertEqual(tool._idle_timeout("12.5"), 12.5)
        for raw in ("forever", -1, float("nan"), float("inf")):
            with self.subTest(raw=raw), self.assertRaises(tool.DaemonError):
                tool._idle_timeout(raw)


if __name__ == "__main__":
    unittest.main()
