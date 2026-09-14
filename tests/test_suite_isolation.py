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
own test the one way the package guard does not cover, and requires that the
same comparison sees the store change.
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

    def test_control_the_unguarded_incident_test_is_seen_writing(self) -> None:
        # The incident's own test, run where nothing isolates it: it installs
        # fixture model files and a grant through the exported data root. If
        # the comparison cannot see that, its passes above mean nothing.
        self.run_unittest("discover", "-s", "tests", "-p", "test_r6_attack.py",
                          "-k", "test_control_catalogue_model_grant_is_accepted")
        changed = self.changed()
        model_conf = os.path.join("gpu_terminal", "kilix", "data", "voice", "models",
                                  "small-en-us", "conf", "model.conf")
        self.assertIn(model_conf, changed)
        self.assertIn(os.path.join("gpu_terminal", "kilix", "data", "voice",
                                   "consent.json"), changed)


if __name__ == "__main__":
    unittest.main()
