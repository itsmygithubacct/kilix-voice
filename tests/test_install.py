from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
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


SOURCES = ("Makefile", "VERSION", "kilix-tts", "kilix-stt", "kilix-voiced")
EDIT = "# an uncommitted operator edit\n"


def snapshot(root: pathlib.Path) -> dict[str, str]:
    """Every entry under root: a symlink's target, a file's sha256, or dir."""
    entries = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            entries[name] = "link " + os.readlink(path)
        elif path.is_file():
            entries[name] = "file " + hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            entries[name] = "dir"
    return entries


def copy_checkout(dest: pathlib.Path) -> pathlib.Path:
    """A checkout's install inputs, with an uncommitted edit in one module."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        shutil.copy2(ROOT / name, dest / name)
    shutil.copytree(ROOT / "voicelib", dest / "voicelib",
                    ignore=shutil.ignore_patterns("__pycache__"))
    with open(dest / "voicelib" / "util.py", "a", encoding="utf-8") as handle:
        handle.write(EDIT)
    return dest


def make(checkout: pathlib.Path, *args: str,
         env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "-s", *args], cwd=checkout, env=env,
                          check=False, capture_output=True, text=True)


class UninstallKeepsTheCheckoutTests(unittest.TestCase):
    """make uninstall never removes the files it compares against.

    Its refusal compares each target with the checkout's file. When the
    target IS the checkout's file -- the checkout sits at
    PREFIX/lib/kilix-voice, or that path links to it -- the comparison
    passes trivially, and the removal that follows deleted the checkout's
    VERSION and every voicelib module, uncommitted edits included.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(prefix="kilix-voice-uninstall-")
        self.addCleanup(self._temp.cleanup)
        self.base = pathlib.Path(self._temp.name)
        self.prefix = self.base / "prefix"
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "lib").mkdir()

    def test_a_checkout_that_lives_at_the_install_path_keeps_its_sources(self):
        checkout = copy_checkout(self.prefix / "lib" / "kilix-voice")
        before = snapshot(checkout)
        result = make(checkout, "uninstall", f"PREFIX={self.prefix}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("is this checkout or lies inside it", result.stderr)
        self.assertEqual(snapshot(checkout), before)
        self.assertIn(EDIT, (checkout / "voicelib" / "util.py").read_text())

    def test_a_lib_dir_that_links_to_the_checkout_keeps_its_sources(self):
        checkout = copy_checkout(self.base / "dev-checkout")
        (self.prefix / "lib" / "kilix-voice").symlink_to(checkout)
        before = snapshot(checkout)
        result = make(checkout, "uninstall", f"PREFIX={self.prefix}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("is this checkout or lies inside it", result.stderr)
        self.assertEqual(snapshot(checkout), before)
        self.assertTrue((self.prefix / "lib" / "kilix-voice").is_symlink())

    def test_a_prefix_inside_the_checkout_is_refused(self):
        checkout = copy_checkout(self.base / "dev-checkout")
        stage = checkout / "stage"
        self.assertEqual(make(checkout, "install", f"PREFIX={stage}").returncode, 0)
        before = snapshot(checkout)
        result = make(checkout, "uninstall", f"PREFIX={stage}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("is this checkout or lies inside it", result.stderr)
        self.assertEqual(snapshot(checkout), before)

    def test_a_target_that_is_its_own_source_is_refused(self):
        # A hard link is the same file under another path; so is a bind
        # mount, which no path resolution sees through.
        checkout = copy_checkout(self.base / "dev-checkout")
        self.assertEqual(
            make(checkout, "install", f"PREFIX={self.prefix}").returncode, 0)
        installed = self.prefix / "lib" / "kilix-voice" / "voicelib" / "util.py"
        installed.unlink()
        os.link(checkout / "voicelib" / "util.py", installed)
        before = snapshot(self.prefix), snapshot(checkout)
        result = make(checkout, "uninstall", f"PREFIX={self.prefix}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("is this checkout's own voicelib/util.py", result.stderr)
        self.assertEqual((snapshot(self.prefix), snapshot(checkout)), before)


if __name__ == "__main__":
    unittest.main()
