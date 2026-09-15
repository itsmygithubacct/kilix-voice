"""Isolate a test run of this checkout, however its test file was started.

tests/__init__.py takes every stack variable out of the environment and points
HOME and the XDG tree at a private temporary directory, so that no test can
reach the invoking user's Kilix store. It runs only when the tests PACKAGE is
imported, and four ways of starting a test never import it: a test file run
as a script, from the checkout or from tests/; ``python -m unittest discover
-s tests`` without ``-t .``; and ``python -m unittest test_<name>`` from inside
tests/. In a session that exports KILIX_DATA_HOME, tests/test_r6_attack.py run
any of those ways wrote fixture model files and a consent grant into the
user's real store. That file is a verbatim review artefact and stays as it is.

Every test module imports voicelib, directly or through kilix-voiced, before
it can touch a store. So voicelib runs guard() when it is first imported: if a
module already loaded lives in this checkout's tests/ directory -- the script
being run, or the test module being imported -- tests/__init__.py is loaded
before anything else in voicelib is. The rule itself stays in that one file;
this module only makes sure it runs.

An installed runtime has no tests/ directory beside its voicelib, and no
production entry point is loaded from one, so there guard() does nothing.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_PACKAGE = "tests"
# The name the guard is loaded under when some other project's module already
# holds "tests" in this process.
_FALLBACK = "_kilix_voice_tests"


def _tests_directory() -> str | None:
    """This checkout's tests/ directory, or None when there is none."""
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    directory = os.path.join(root, _PACKAGE)
    if os.path.isfile(os.path.join(directory, "__init__.py")):
        return directory
    return None


def _loaded_from(directory: str) -> bool:
    """True when a module already in this process was loaded from ``directory``."""
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not isinstance(path, str):
            continue
        parent = os.path.dirname(os.path.abspath(path))
        if os.path.basename(parent) == _PACKAGE and os.path.realpath(parent) == directory:
            return True
    return False


def guard() -> None:
    """Run tests/__init__.py first, when this process is running a test from it."""
    directory = _tests_directory()
    if directory is None or not _loaded_from(directory):
        return
    init = os.path.join(directory, "__init__.py")
    name = _PACKAGE
    loaded = sys.modules.get(_PACKAGE)
    if loaded is not None:
        if os.path.realpath(getattr(loaded, "__file__", None) or "") == init:
            return                      # imported as a package: already isolated
        name = _FALLBACK
        if name in sys.modules:
            return
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[directory])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
