from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class InstalledRuntimeTests(unittest.TestCase):
    def test_make_install_produces_self_contained_executable_tools(self):
        with tempfile.TemporaryDirectory(prefix="kilix-voice-install-") as raw:
            temp = pathlib.Path(raw)
            prefix = temp / "prefix"
            subprocess.run(
                ["make", "install", f"PREFIX={prefix}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            package = prefix / "lib" / "kilix-voice" / "voicelib"
            self.assertTrue((package / "__init__.py").is_file())
            self.assertEqual(
                (prefix / "lib" / "kilix-voice" / "VERSION").read_text(),
                "0.1.6\n",
            )

            env = {
                "HOME": str(temp / "home"),
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C.UTF-8",
            }
            for tool in ("kilix-tts", "kilix-stt", "kilix-voiced"):
                with self.subTest(tool=tool):
                    executable = prefix / "bin" / tool
                    self.assertTrue(executable.is_file())
                    self.assertTrue(os.access(executable, os.X_OK))
                    result = subprocess.run(
                        [executable, "--version"],
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.stdout.strip(), f"{tool} 0.1.6")

            subprocess.run(
                ["make", "uninstall", f"PREFIX={prefix}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            for tool in ("kilix-tts", "kilix-stt", "kilix-voiced"):
                self.assertFalse((prefix / "bin" / tool).exists())
            self.assertFalse((prefix / "lib" / "kilix-voice").exists())

    def test_make_uninstall_refuses_a_modified_command(self):
        with tempfile.TemporaryDirectory(prefix="kilix-voice-install-") as raw:
            prefix = pathlib.Path(raw) / "prefix"
            subprocess.run(
                ["make", "install", f"PREFIX={prefix}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            changed = prefix / "bin" / "kilix-stt"
            changed.write_text("foreign replacement\n", encoding="utf-8")

            result = subprocess.run(
                ["make", "uninstall", f"PREFIX={prefix}"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("modified or foreign file", result.stderr)
            self.assertTrue((prefix / "bin" / "kilix-tts").exists())
            self.assertEqual(changed.read_text(encoding="utf-8"),
                             "foreign replacement\n")


if __name__ == "__main__":
    unittest.main()
