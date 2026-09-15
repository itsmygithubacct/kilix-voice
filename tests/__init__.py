"""The kilix-voice test package: no test can reach the invoking user's store.

Importing this package runs before any test module is imported, under every
supported runner: ``make test``, ``make test-clean``, ``python -m unittest
discover -s tests -t .`` and ``python -m unittest tests.<module>``. It takes
every KILIX*, GPU_TERMINAL_*, PLEB_* and XDG_* variable out of this process's
environment, and points HOME and the XDG tree at a fresh per-process temporary
directory that is removed when the process exits.

Why: a desktop session exports KILIX_DATA_HOME and KILIX_STORAGE_HOME naming
the user's real Kilix store, and the path resolvers honour them ahead of HOME.
A test that replaced HOME, or even KILIX_STORAGE_HOME, and then installed
fixture model files or recorded a consent grant wrote them into that real
store, over an installed dictation model and its consent record. Fixing each
test is necessary but not sufficient, because the next test would need the
same care. Here one rule covers every test.

What remains is the environment tests/cleanenv.sh builds, a private HOME and
XDG tree with no stack variable, so the suite behaves the same under either
runner. tests/test_suite_isolation.py checks the guard against a sentinel store.
Some invocations never import this package: a test file run as a script,
``discover -s tests`` without ``-t .``, and ``python -m unittest test_<name>``
from inside tests/. For those, voicelib loads this file when it is first
imported (voicelib/_test_isolation.py), before any test can reach a store.
Modules that write a store also protect themselves (see
tests/test_consent_gate.py).
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

SCRUBBED_PREFIXES = ("KILIX", "GPU_TERMINAL_", "PLEB_", "XDG_")
_XDG_LEAVES = (
    ("XDG_DATA_HOME", "data"),
    ("XDG_CONFIG_HOME", "config"),
    ("XDG_STATE_HOME", "state"),
    ("XDG_CACHE_HOME", "cache"),
    ("XDG_RUNTIME_DIR", "runtime"),
)


def _remove_tree(root: str, owner: int) -> None:
    # A forked child that exits normally runs atexit too, and must not remove
    # the tree its parent is still using.
    if os.getpid() == owner:
        shutil.rmtree(root, ignore_errors=True)


def _isolate() -> str:
    root = tempfile.mkdtemp(prefix="kilix-voice-tests-")
    atexit.register(_remove_tree, root, os.getpid())
    for name in [name for name in os.environ if name.startswith(SCRUBBED_PREFIXES)]:
        del os.environ[name]
    home = os.path.join(root, "home")
    os.mkdir(home, 0o700)
    os.environ["HOME"] = home
    for variable, leaf in _XDG_LEAVES:
        path = os.path.join(root, leaf)
        os.mkdir(path, 0o700)
        os.environ[variable] = path
    return root


# The private tree every test in this process resolves its store under.
ISOLATION_ROOT = _isolate()
