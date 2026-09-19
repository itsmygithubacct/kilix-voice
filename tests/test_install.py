from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
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


class UninstallRemovesOnlyItsOwnFilesTests(unittest.TestCase):
    """make uninstall removes what install copied, and its bytecode.

    Beside the install, an operator's own files survive, including one left
    in voicelib/__pycache__, which uninstall used to remove whole.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(prefix="kilix-voice-uninstall-")
        self.addCleanup(self._temp.cleanup)
        self.prefix = pathlib.Path(self._temp.name) / "prefix"

    def test_a_modified_voicelib_module_is_refused_and_nothing_removed(self):
        self.assertEqual(make(ROOT, "install", f"PREFIX={self.prefix}").returncode, 0)
        module = self.prefix / "lib" / "kilix-voice" / "voicelib" / "util.py"
        with open(module, "a", encoding="utf-8") as handle:
            handle.write(EDIT)
        before = snapshot(self.prefix)
        result = make(ROOT, "uninstall", f"PREFIX={self.prefix}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(self.prefix), before)
        self.assertIn(f"modified or foreign file: {module}", result.stderr)

    def test_a_version_from_another_release_is_refused_and_nothing_removed(self):
        self.assertEqual(make(ROOT, "install", f"PREFIX={self.prefix}").returncode, 0)
        version = self.prefix / "lib" / "kilix-voice" / "VERSION"
        version.write_text("0.0.0\n", encoding="utf-8")
        before = snapshot(self.prefix)
        result = make(ROOT, "uninstall", f"PREFIX={self.prefix}")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(self.prefix), before)
        self.assertIn(f"modified or foreign file: {version}", result.stderr)

    def test_bytecode_is_removed_only_for_installed_modules(self):
        self.assertEqual(make(ROOT, "install", f"PREFIX={self.prefix}").returncode, 0)
        cache = self.prefix / "lib" / "kilix-voice" / "voicelib" / "__pycache__"
        cache.mkdir()
        tag = sys.implementation.cache_tag
        ours = [cache / f"util.{tag}.pyc", cache / f"__init__.{tag}.opt-1.pyc"]
        for path in ours:
            path.write_bytes(b"bytecode")
        planted = cache / "operator-notes.txt"
        planted.write_text("keep me\n", encoding="utf-8")
        result = make(ROOT, "uninstall", f"PREFIX={self.prefix}")
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in ours:
            self.assertFalse(path.exists(), path)
        self.assertEqual(planted.read_text(encoding="utf-8"), "keep me\n")
        self.assertEqual(
            sorted(str(p.relative_to(self.prefix))
                   for p in self.prefix.rglob("*")),
            ["bin", "lib", "lib/kilix-voice", "lib/kilix-voice/voicelib",
             "lib/kilix-voice/voicelib/__pycache__",
             "lib/kilix-voice/voicelib/__pycache__/operator-notes.txt"])

    def test_a_linked_bytecode_directory_is_left_alone(self):
        self.assertEqual(make(ROOT, "install", f"PREFIX={self.prefix}").returncode, 0)
        elsewhere = pathlib.Path(self._temp.name) / "shared-cache"
        elsewhere.mkdir()
        cached = elsewhere / f"util.{sys.implementation.cache_tag}.pyc"
        cached.write_bytes(b"another program's bytecode")
        cache = self.prefix / "lib" / "kilix-voice" / "voicelib" / "__pycache__"
        cache.symlink_to(elsewhere)
        result = make(ROOT, "uninstall", f"PREFIX={self.prefix}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(cached.read_bytes(), b"another program's bytecode")
        self.assertTrue(cache.is_symlink())


class UninstallAtTheDefaultPrefixTests(unittest.TestCase):
    """At PREFIX=$HOME/.local, kilix's managed entrypoints are not ours.

    kilix's install-kilix-voice.sh runs make install into a generation in
    its store and points ~/.local/bin/kilix-{tts,stt,voiced} at it with
    symlinks. Those bytes match a checkout of the same release, so a
    comparison alone removed them. install never creates a symlink.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(prefix="kilix-voice-home-")
        self.addCleanup(self._temp.cleanup)
        self.home = pathlib.Path(self._temp.name) / "home"
        self.home.mkdir()
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": str(self.home), "LANG": "C.UTF-8"}
        # Before anything is removed: the default prefix is this scratch
        # HOME's, never the invoking user's.
        shown = subprocess.run(
            ["make", "-s", "--no-print-directory",
             "--eval", "show-prefix: ; @echo $(PREFIX)", "show-prefix"],
            cwd=ROOT, env=self.env, check=True, capture_output=True,
            text=True).stdout.strip()
        self.assertEqual(shown, f"{self.home}/.local")
        self.bin = self.home / ".local" / "bin"
        self.bin.mkdir(parents=True)
        store = self.home / ".local" / "gpu_terminal" / "kilix" / "data" / "voice"
        generation = store / "runtime" / "generations" / "kilix-voice-gen1"
        self.generation = generation
        subprocess.run(["make", "-s", "install", f"PREFIX={generation}"],
                       cwd=ROOT, env=self.env, check=True, capture_output=True)
        (store / "runtime" / "current").symlink_to("generations/kilix-voice-gen1")
        model = store / "models" / "vosk-model-small-en-us-0.15-test"
        (model / "conf").mkdir(parents=True)
        (model / "conf" / "model.conf").write_text("model\n", encoding="utf-8")
        (store / "consent.json").write_text("{}\n", encoding="utf-8")
        (store / "consent.lock").write_bytes(b"")
        for tool in ("kilix-tts", "kilix-stt", "kilix-voiced"):
            (self.bin / tool).symlink_to(store / "runtime" / "current" / "bin" / tool)
        # Other components' commands beside them.
        (self.bin / "kilix").symlink_to(self.home / "elsewhere" / "kilix")
        (self.bin / "other-tool").write_text("#!/bin/sh\n", encoding="utf-8")

    def test_managed_entrypoints_and_the_store_are_left_alone(self):
        before = snapshot(self.home)
        result = make(ROOT, "uninstall", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(self.home), before)
        for tool in ("kilix-tts", "kilix-stt", "kilix-voiced"):
            self.assertIn(f"leaving {self.bin / tool}: a symlink", result.stderr)

    def test_entrypoints_of_another_release_neither_block_nor_go(self):
        # kilix may manage another release than this checkout. Its commands
        # then differ from ours, and are still not ours to judge or remove.
        with open(self.generation / "bin" / "kilix-stt", "a",
                  encoding="utf-8") as handle:
            handle.write("# another release\n")
        before = snapshot(self.home)
        result = make(ROOT, "uninstall", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(self.home), before)

    def test_a_make_install_beside_them_is_removed_and_they_stay(self):
        # An earlier make install left its package under ~/.local/lib; kilix
        # has since replaced the commands with its symlinks.
        lib = self.home / ".local" / "lib" / "kilix-voice"
        self.assertEqual(
            make(ROOT, "install", f"PREFIX={self.home / 'staging'}",
                 env=self.env).returncode, 0)
        shutil.move(self.home / "staging" / "lib" / "kilix-voice", lib)
        shutil.rmtree(self.home / "staging")
        before = snapshot(self.home)
        result = make(ROOT, "uninstall", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        after = snapshot(self.home)
        removed = sorted(set(before) - set(after))
        self.assertTrue(removed)
        self.assertTrue(all(name.startswith(".local/lib/kilix-voice")
                            for name in removed), removed)
        self.assertFalse(lib.exists())
        self.assertEqual(after, {name: value for name, value in before.items()
                                 if name not in removed})


if __name__ == "__main__":
    unittest.main()
