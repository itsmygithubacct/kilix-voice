"""ISO-01: no test reads or writes the invoking user's real Kilix store.

A desktop session exports KILIX_DATA_HOME and KILIX_STORAGE_HOME naming the
user's store. Two test modules installed fixture model files and consent
grants through those resolvers and overwrote a real installed dictation model
and its consent record. These tests stand a SENTINEL store in for the real
one. Its roots are exported the way the session exports them. Tests that
install models and grant consent are run against it, and every file and
directory in it must be unchanged afterwards: bytes, size, mode, mtime and
ctime, and no entry added or removed.

A comparison that cannot fail proves nothing, so a control runs the incident's
own test, in every form that never imports the tests package, in a copy of the
checkout whose voicelib guard does nothing, and requires that the same
comparison and audit see the store change there.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIXES = ("KILIX", "GPU_TERMINAL_", "PLEB_", "XDG_")
SENTINEL_MODEL = b"sentinel: the dictation model the owner installed\n"
# A valid, empty record, as a real one is: a write that replaces it is then
# the grant the incident recorded, not a refusal to read a broken file.
SENTINEL_CONSENT = b'{"schema": "kilix.voice.consent/v1", "grants": {}}\n'
# The modules that recorded grants and model files into the real store.
INSTALLING_TESTS = (
    "tests.test_consent_gate",
    "tests.test_r6_attack.ATTACK_GrantCannotSatisfyTheGateUnderAModelOverride",
    "tests.test_r6_attack.ATTACK_BrokenConsentRecordCodedUnavailable",
)


def seed_sentinel(root: str) -> dict[str, str]:
    """Build a store shaped like a live one; return the session's variables for it."""
    storage = os.path.join(root, "gpu_terminal", "kilix")
    data = os.path.join(storage, "data")
    voice = os.path.join(data, "voice")
    for model in ("small-en-us", "vosk-model-small-en-us-0.15-sentinel"):
        for rel in ("conf/model.conf", "am/final.mdl"):
            path = os.path.join(voice, "models", model, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(SENTINEL_MODEL)
    with open(os.path.join(voice, "consent.json"), "wb") as handle:
        handle.write(SENTINEL_CONSENT)
    open(os.path.join(voice, "consent.lock"), "wb").close()
    session = os.path.join(storage, "session")
    os.makedirs(os.path.join(session, "voice"))
    settings_file = os.path.join(root, "gpu_terminal", "settings.conf")
    with open(settings_file, "w", encoding="utf-8") as handle:
        handle.write("KILIX_VOICE_STT_ENGINE=vosk\nKILIX_VOICE_STT_MODEL=small-en-us\n")
    env = {
        "HOME": os.path.join(root, "home"),
        "GPU_TERMINAL_HOME": os.path.join(root, "gpu_terminal"),
        "GPU_TERMINAL_SETTINGS_FILE": settings_file,
        "KILIX_STORAGE_HOME": storage,
        "KILIX_DATA_HOME": data,
        "KILIX_SESSION_HOME": session,
    }
    for variable, leaf in (("XDG_DATA_HOME", "share"), ("XDG_CONFIG_HOME", "config"),
                           ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache")):
        env[variable] = os.path.join(root, "home", ".xdg", leaf)
        os.makedirs(env[variable])
    return env


def snapshot(root: str) -> list[tuple]:
    """Every entry under ``root``, with what a write to it would change."""
    entries = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in [""] + sorted(filenames):
            path = os.path.join(directory, name) if name else directory
            info = os.lstat(path)
            digest = None
            if stat.S_ISREG(info.st_mode):
                with open(path, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
            entries.append((os.path.relpath(path, root), info.st_mode, info.st_size,
                            info.st_mtime_ns, info.st_ctime_ns, digest))
    return entries


def _remove(root: str) -> None:
    # A test may leave a mode-000 file behind; make everything removable first.
    for directory, _dirs, files in os.walk(root):
        os.chmod(directory, 0o700)
        for name in files:
            try:
                os.chmod(os.path.join(directory, name), 0o600, follow_symlinks=False)
            except OSError:
                pass
    shutil.rmtree(root, ignore_errors=True)


class PackageGuardTestCase(unittest.TestCase):

    def test_every_store_root_resolves_inside_the_private_tree(self) -> None:
        if not __name__.startswith("tests."):
            self.skipTest("run through the tests package, as the suite runners do")
        import tests
        from voicelib import consent, paths
        private = os.path.realpath(tests.ISOLATION_ROOT) + os.sep
        leftover = sorted(name for name in os.environ
                          if name.startswith(PREFIXES[:3]))
        self.assertEqual(leftover, [], "a stack variable reached the tests")
        resolved = {
            "HOME": os.path.expanduser("~"),
            "gpu_terminal_home": paths.gpu_terminal_home(),
            "storage_home": paths.storage_home(),
            "data_home": paths.data_home(),
            "session_home": paths.session_home(),
            "settings_file": paths.settings_file(),
            "consent_path": consent.consent_path(),
        }
        resolved.update({name: value for name, value in os.environ.items()
                         if name.startswith("XDG_")})
        for name, value in resolved.items():
            with self.subTest(name=name):
                self.assertTrue(os.path.realpath(value).startswith(private),
                                f"{name} resolves outside the private tree")


    def test_every_stack_variable_is_taken_out_whatever_the_runner_exported(self) -> None:
        # Wave-3a survivor I1: the scrubbed runner exports no stack variable,
        # so a prefix dropped from the guard went unseen there. This child is
        # handed one of each, all naming a tree outside its own.
        import json
        outside = tempfile.mkdtemp(prefix="kv-exported-")
        self.addCleanup(shutil.rmtree, outside, True)
        exported = {
            "GPU_TERMINAL_HOME": os.path.join(outside, "gpu_terminal"),
            "GPU_TERMINAL_SETTINGS_FILE": os.path.join(outside, "settings.conf"),
            "KILIX_STORAGE_HOME": os.path.join(outside, "storage"),
            "KILIX_DATA_HOME": os.path.join(outside, "storage", "data"),
            "PLEB_DATA_HOME": os.path.join(outside, "pleb"),
            "XDG_DATA_HOME": os.path.join(outside, "share"),
        }
        probe = ("import json, os, tests\n"
                 "from voicelib import paths\n"
                 "print(json.dumps({'left': sorted(n for n in os.environ if n.startswith("
                 "('KILIX', 'GPU_TERMINAL_', 'PLEB_')) or os.environ[n].startswith("
                 + repr(outside) + ")),\n"
                 "  'root': tests.ISOLATION_ROOT, 'settings': paths.settings_file(),\n"
                 "  'home': paths.gpu_terminal_home(), 'data': paths.data_home()}))\n")
        child = subprocess.run(
            [sys.executable, "-B", "-c", probe], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **exported),
            capture_output=True, text=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        seen = json.loads(child.stdout)
        self.assertEqual(seen["left"], [], "a stack variable reached the tests")
        private = os.path.realpath(seen["root"]) + os.sep
        for name in ("settings", "home", "data"):
            with self.subTest(root=name):
                self.assertTrue(os.path.realpath(seen[name]).startswith(private), seen)

    def test_the_private_tree_is_removed_when_the_process_exits(self) -> None:
        child = subprocess.run(
            [sys.executable, "-B", "-c",
             "import os, tests; os.makedirs(os.path.join(os.environ['HOME'], 'x'));"
             "print(tests.ISOLATION_ROOT)"],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        root = child.stdout.strip()
        self.assertTrue(root and os.path.isabs(root), child.stdout)
        self.assertFalse(os.path.exists(root), "the private tree outlived its process")


class SentinelStoreTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="kv-sentinel-")
        self.addCleanup(_remove, self.root)
        self.session_env = seed_sentinel(self.root)
        self.before = snapshot(self.root)

    def run_unittest(self, *argv: str) -> subprocess.CompletedProcess:
        """Run unittest as a user in that session would: its store is exported."""
        env = {name: value for name, value in os.environ.items()
               if not name.startswith(PREFIXES)}
        env.update(self.session_env)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([sys.executable, "-B", "-m", "unittest", *argv],
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=300)

    def changed(self) -> list[str]:
        after = {entry[0]: entry for entry in snapshot(self.root)}
        before = {entry[0]: entry for entry in self.before}
        return sorted(name for name in set(after) | set(before)
                      if after.get(name) != before.get(name))

    def test_installing_tests_leave_an_exported_store_untouched(self) -> None:
        proc = self.run_unittest(*INSTALLING_TESTS)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertRegex(proc.stderr, r"(?m)^OK")
        self.assertEqual(self.changed(), [], "a test wrote into the exported store")

    def test_the_consent_module_protects_itself_without_the_package(self) -> None:
        # discover without -t imports test_consent_gate as a top-level module,
        # so the package guard never runs; the module's own isolation must hold.
        proc = self.run_unittest("discover", "-s", "tests", "-p", "test_consent_gate.py")
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertRegex(proc.stderr, r"(?m)^Ran 11 tests")
        self.assertEqual(self.changed(), [], "a test wrote into the exported store")


# ISO-01R: the incident's own module, tests/test_r6_attack.py, started in each
# way that never imports the tests package. Its bytes are a verbatim review
# artefact, so it is protected from outside, by voicelib/_test_isolation.py.
R6_WRITERS = ("ATTACK_GrantCannotSatisfyTheGateUnderAModelOverride",
              "ATTACK_BrokenConsentRecordCodedUnavailable")
UNPACKAGED_FORMS = (
    ("script from the checkout", ".", ("tests/test_r6_attack.py",) + R6_WRITERS),
    ("script from tests/", "tests", ("test_r6_attack.py",) + R6_WRITERS),
    ("discover -s tests without -t", ".",
     ("-m", "unittest", "discover", "-s", "tests", "-p", "test_r6_attack.py",
      "-k", R6_WRITERS[0], "-k", R6_WRITERS[1])),
    ("unittest from inside tests/", "tests",
     ("-m", "unittest") + tuple(f"test_r6_attack.{name}" for name in R6_WRITERS)),
)
# Loaded by every child through PYTHONPATH: records each open, mkdir, rename,
# replace, remove and chmod naming the sentinel, reads included, so a test that
# only READS the exported store is seen too.
AUDIT_HOOK = '''
import os, sys
_root, _log = os.environ.get("KV_ISO_SENTINEL"), os.environ.get("KV_ISO_AUDIT")
if _root and _log:
    def _audit(event, args, _root=_root, _log=_log):
        if event in ("open", "os.mkdir", "os.rename", "os.replace", "os.remove", "os.chmod"):
            path = args[0] if args else None
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if isinstance(path, str) and os.path.realpath(path).startswith(_root):
                with open(_log, "a") as handle:
                    handle.write(event + " " + path + "\\n")
    sys.addaudithook(_audit)
'''


class UnpackagedFormsTestCase(unittest.TestCase):
    """Every way of starting a test file leaves an exported store untouched."""

    def sentinel(self) -> tuple[str, dict[str, str], list[tuple]]:
        root = os.path.realpath(tempfile.mkdtemp(prefix="kv-sentinel-"))
        self.addCleanup(_remove, root)
        env = seed_sentinel(root)
        return root, env, snapshot(root)

    def run_form(self, tree: str, cwd: str, argv: tuple[str, ...]):
        """Run one form in ``tree`` against a fresh sentinel; return (proc, changed, audited)."""
        root, session_env, before = self.sentinel()
        hooks = tempfile.mkdtemp(prefix="kv-audit-")
        self.addCleanup(shutil.rmtree, hooks, True)
        with open(os.path.join(hooks, "sitecustomize.py"), "w", encoding="utf-8") as handle:
            handle.write(AUDIT_HOOK)
        log = os.path.join(hooks, "audit.log")
        env = {name: value for name, value in os.environ.items()
               if not name.startswith(PREFIXES)}
        env.update(session_env)
        env.update(PYTHONDONTWRITEBYTECODE="1", KV_ISO_SENTINEL=root, KV_ISO_AUDIT=log,
                   PYTHONPATH=os.pathsep.join(
                       part for part in (hooks, os.environ.get("PYTHONPATH")) if part))
        proc = subprocess.run([sys.executable, "-B", *argv], cwd=os.path.join(tree, cwd),
                              env=env, capture_output=True, text=True, timeout=300)
        after = {entry[0]: entry for entry in snapshot(root)}
        earlier = {entry[0]: entry for entry in before}
        changed = sorted(name for name in set(after) | set(earlier)
                         if after.get(name) != earlier.get(name))
        audited = []
        if os.path.exists(log):
            with open(log, encoding="utf-8") as handle:
                audited = handle.read().splitlines()
        return proc, changed, audited

    def test_the_incident_module_run_unpackaged_neither_reads_nor_writes_the_store(self) -> None:
        for label, cwd, argv in UNPACKAGED_FORMS:
            with self.subTest(form=label):
                proc, changed, audited = self.run_form(ROOT, cwd, argv)
                self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
                self.assertRegex(proc.stderr, r"(?m)^Ran 3 tests")
                self.assertRegex(proc.stderr, r"(?m)^OK")
                self.assertEqual(changed, [], "a test wrote into the exported store")
                self.assertEqual(audited, [], "a test opened the exported store")

    def test_control_without_the_guard_the_unpackaged_forms_are_seen_writing(self) -> None:
        # The same forms in a copy of the checkout whose guard does nothing.
        # If the comparison and the audit could not see the incident's writes
        # there, their silence above would mean nothing.
        tree = tempfile.mkdtemp(prefix="kv-unguarded-")
        self.addCleanup(shutil.rmtree, tree, True)
        ignore = shutil.ignore_patterns("__pycache__")
        for name in ("voicelib", "tests"):
            shutil.copytree(os.path.join(ROOT, name), os.path.join(tree, name), ignore=ignore)
        # Every executable the incident's tests load: a copy missing one fails
        # those tests before they write, and would pass for the wrong reason.
        for name in ("kilix-voiced", "kilix-stt", "kilix-tts", "VERSION"):
            shutil.copy2(os.path.join(ROOT, name), os.path.join(tree, name))
        with open(os.path.join(tree, "voicelib", "_test_isolation.py"), "w",
                  encoding="utf-8") as handle:
            handle.write("def guard():\n    pass\n")
        voice = os.path.join("gpu_terminal", "kilix", "data", "voice")
        for label, cwd, argv in UNPACKAGED_FORMS:
            with self.subTest(form=label):
                proc, changed, audited = self.run_form(tree, cwd, argv)
                self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
                self.assertRegex(proc.stderr, r"(?m)^Ran 3 tests")
                self.assertIn(os.path.join(voice, "models", "small-en-us", "conf",
                                           "model.conf"), changed)
                self.assertIn(os.path.join(voice, "consent.json"), changed)
                self.assertNotEqual(audited, [])

    def test_an_installed_runtime_is_never_isolated(self) -> None:
        # No tests/ beside voicelib: importing it changes nothing in the
        # environment, even for a process whose main file sits in a tests/
        # directory of its own.
        tree = tempfile.mkdtemp(prefix="kv-installed-")
        self.addCleanup(shutil.rmtree, tree, True)
        shutil.copytree(os.path.join(ROOT, "voicelib"), os.path.join(tree, "lib", "voicelib"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        os.mkdir(os.path.join(tree, "tests"))
        script = os.path.join(tree, "tests", "probe.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("import os, sys\nsys.path.insert(0, sys.argv[1])\nimport voicelib\n"
                         "print(os.environ.get('KILIX_DATA_HOME'), 'tests' in sys.modules)\n")
        env = dict(os.environ, KILIX_DATA_HOME="/nonexistent/kilix-data",
                   PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", script, os.path.join(tree, "lib")],
                              env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split(), ["/nonexistent/kilix-data", "False"])


if __name__ == "__main__":
    unittest.main()
