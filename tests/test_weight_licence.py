"""V-ACC: model weights are fetched only after a covering licence receipt.

OS-V-VERIFY F2: the only speech-model install action the operating-system image
advertises (``kilix stt --install <model> --default <model>``) reached the
pinned installer with no acceptance of any kind, on a provisioning path with
nobody there to accept anything. OD-S and OD-BB require the opposite.

Every test here runs the real command as a subprocess against a **spy
installer**: a stand-in ``kilix`` (and ``kilix-piper-tts``) that records its
argv and writes under the model store exactly as a real fetch would. "Fetches
nothing" is therefore observed -- the spy did not run and the store did not
change -- rather than read off the source. The control that the spy would have
seen a fetch is :meth:`CoveringReceiptTests.test_a_covering_receipt_proceeds`,
and the planted regression that removes the gate is
:class:`PlantedRegressionTests`; both must keep working or the no-fetch
assertions below prove nothing.

No test here fetches anything real: the sandbox has no network route in the X3
runner, the spy is a shell script, and no URL appears anywhere in this file.

The authority's source must be on the import path: ``make test`` and ``make
test-clean`` put it there from LICENSE_SRC. When it is not, these tests FAIL
naming LICENSE_SRC; they never skip, because weights fetched without a licence
is exactly what they exist to rule out.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest

try:
    import kilix_license as LICENCE
except ImportError:
    LICENCE = None

from voicelib import licensing, models, paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MISSING = (
    "kilix_license is not importable. Run the suite with the kilix-license "
    "source on PYTHONPATH: make test and make test-clean set it from "
    "LICENSE_SRC (default ../../kilix-modules/kilix-license/src).")

# Any well-formed sha256 will do for the fixture receipts: this checkout never
# computes a manifest, release or catalogue digest, and never asserts on one.
FIXTURE_MANIFEST = "a" * 64
FIXTURE_RELEASE = "b" * 64
FIXTURE_CATALOGUE = "c" * 64

VOSK_MODEL = "small-en-us"
OTHER_VOSK_MODEL = "lgraph-en-us"
PIPER_MODEL = models.PIPER_KRISTIN_MODEL

SPY_TEMPLATE = """#!/bin/sh
# Spy installer. A real one fetches weights and publishes them under the model
# store; this records that it was reached and writes there, so a test can see
# either. It is never reached when the licence gate refuses.
printf '%s\\n' "$*" >> {log}
mkdir -p {models}/fetched-by-{name}
printf 'WEIGHTS' > {models}/fetched-by-{name}/final.mdl
exit 0
"""


def authority_source() -> str:
    """Return the directory holding the importable kilix_license package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(LICENCE.__file__)))


class _WeightsFixture(unittest.TestCase):
    """A private store, a spy installer, and no receipt unless a test mints one."""

    def setUp(self) -> None:
        if LICENCE is None:
            self.fail(MISSING)
        self.root = tempfile.mkdtemp(prefix="v-acc-licence-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        self.store = os.path.join(self.root, "store")
        self.kilix_home = os.path.join(self.store, "kilix")
        self.data = os.path.join(self.kilix_home, "data")
        self.models_dir = os.path.join(self.data, paths.VOICE_LEAF, "models")
        os.makedirs(self.models_dir, mode=0o700)
        self.receipts = os.path.join(self.store, licensing.RECEIPTS_LEAF)

        self.home = os.path.join(self.root, "home")
        self.tmp = os.path.join(self.root, "tmp")
        self.spy_dir = os.path.join(self.root, "spy")
        # A directory with no launcher and no provider in it, for the tests
        # that need the gate to be reached on a machine where neither exists.
        self.no_bin = os.path.join(self.root, "no-bin")
        for directory in (self.home, self.tmp, self.spy_dir, self.no_bin):
            os.makedirs(directory, mode=0o700)
        self.spy_log = os.path.join(self.root, "spy.log")
        for name in ("kilix", "kilix-piper-tts"):
            self._write_spy(name)

    # -- fixture plumbing ------------------------------------------------

    def _write_spy(self, name: str) -> None:
        path = os.path.join(self.spy_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(SPY_TEMPLATE.format(
                log=self.spy_log, models=self.models_dir, name=name))
        os.chmod(path, 0o700 | stat.S_IXUSR)

    def _child_env(self, *, authority: bool = True,
                   stub_authority: str | None = None,
                   launcher: bool = True) -> dict[str, str]:
        """The environment one command runs in: private store, spy installer.

        ``launcher=False`` takes the fetcher away entirely: no ``kilix`` and no
        ``kilix-piper-tts`` on ``PATH``, and ``KILIX_HOME`` pointing at an
        empty directory. That is the machine on which the gate's position is
        observable -- see :class:`LicenceOrderingTests`.
        """
        import_path = [ROOT]
        if stub_authority is not None:
            import_path.insert(0, stub_authority)
        elif authority:
            import_path.append(authority_source())
        fetcher_dir = self.spy_dir if launcher else self.no_bin
        return {
            "PATH": os.pathsep.join((fetcher_dir, "/usr/bin", "/bin")),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(import_path),
            "HOME": self.home,
            "TMPDIR": self.tmp,
            "GPU_TERMINAL_HOME": self.store,
            "KILIX_STORAGE_HOME": self.kilix_home,
            "KILIX_DATA_HOME": self.data,
            "KILIX_SESSION_HOME": os.path.join(self.kilix_home, "session"),
            # kilix-stt resolves the launcher here before PATH.
            "KILIX_HOME": fetcher_dir,
        }

    def run_tool(self, tool: str, *args: str, **kwargs) -> subprocess.CompletedProcess:
        return self.run_script(os.path.join(ROOT, tool), *args, **kwargs)

    # The one *install* gate in each command, as written there, paired with
    # what is left when it is removed. Used both to plant its removal
    # (PlantedRegressionTests) and to show that the two orderings are
    # distinguishable (LicenceOrderingTests).
    #
    # Each anchor carries enough context to match exactly once. Both commands
    # also call the gate from their --check-licence probe, which must not be
    # what gets planted: kilix-stt's probe passes a different argument, so one
    # line is already unique there, and kilix-tts's passes the same one, so its
    # anchor carries the statement that follows the install gate. The
    # count assertion below is the guard that keeps these honest.
    GATE_STT = ("    licensing.require_covering_receipt(spec.catalog_id)\n", "")
    GATE_TTS = ("    licensing.require_covering_receipt(catalog_id)\n"
                "    binary = tts_lib.piper_binary()\n",
                "    binary = tts_lib.piper_binary()\n")

    def plant(self, tool: str, gate: tuple[str, str]) -> str:
        """Copy one command with its licence gate removed."""
        anchor, remainder = gate
        with open(os.path.join(ROOT, tool), encoding="utf-8") as handle:
            source = handle.read()
        self.assertEqual(source.count(anchor), 1,
                         f"{tool} no longer holds exactly one install gate "
                         f"matching {anchor!r}")
        planted = os.path.join(self.root, f"regression-{tool}")
        with open(planted, "w", encoding="utf-8") as handle:
            handle.write(source.replace(anchor, remainder))
        return planted

    def run_script(self, script: str, *args: str,
                   **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, script, *args],
            env=self._child_env(**kwargs), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120, check=False)

    # -- receipts --------------------------------------------------------

    def record_for(self, catalog_id: str):
        records = LICENCE.RecordIndex(LICENCE.load_determined_records())
        return records.by_id(catalog_id)

    def mint_receipt(self, catalog_id: str, *, manifest: str = FIXTURE_MANIFEST):
        """Write the receipt the authority itself would write on acceptance."""
        record = self.record_for(catalog_id)
        typed = (LICENCE.typed_agreement_line(record)
                 if record.expected_decision == "accept" else None)
        agreement = LICENCE.capture_agreement(record, typed)
        receipt = LICENCE.receipt_from_agreement(
            record, agreement, manifest_digest=manifest,
            release_digest=FIXTURE_RELEASE, catalogue_digest=FIXTURE_CATALOGUE)
        LICENCE.ReceiptStore(self.receipts).write(receipt)
        return receipt

    def tamper_receipt(self, catalog_id: str, field: str, value: str) -> str:
        """Write a receipt filed under the right record with one field changed.

        The file is named exactly as the authority names a covering receipt, so
        the store finds it. Only its bytes are wrong, which is what a
        record-scoped lookup has to refuse rather than accept on the strength
        of its name.

        "Tampered", not "forged": what these prove is that a **corrupted or
        non-covering** receipt refuses. A well-formed receipt is accepted by
        design, because that is what an acceptance produces and receipts carry
        no subject, timestamp or signature to check. See voicelib/licensing.py
        on why a receipt is therefore never shipped, and on whose job it is to
        make a receipt bind more.
        """
        record = self.record_for(catalog_id)
        receipt = self.mint_receipt(catalog_id)
        payload = receipt.to_jsonable()
        payload[field] = value
        path = LICENCE.ReceiptStore(self.receipts).path_for(
            record.digest, FIXTURE_MANIFEST)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        return str(path)

    # -- observation -----------------------------------------------------

    def store_snapshot(self) -> dict[str, str]:
        """Every file under the private data home, by content."""
        found: dict[str, str] = {}
        for directory, _, names in os.walk(self.data):
            for name in names:
                full = os.path.join(directory, name)
                with open(full, "rb") as handle:
                    found[os.path.relpath(full, self.data)] = hashlib.sha256(
                        handle.read()).hexdigest()
        return found

    def assert_installer_never_ran(self, before: dict[str, str]) -> None:
        """The spy did not run and nothing appeared under the model store."""
        self.assertFalse(
            os.path.exists(self.spy_log),
            f"the installer ran: {self._spy_log_text()}")
        self.assertEqual(before, self.store_snapshot(),
                         "something was written under the model store")

    def assert_refusal(self, result: subprocess.CompletedProcess,
                       tool: str, catalog_id: str) -> None:
        self.assertEqual(result.returncode, licensing.LICENCE_REFUSED_EXIT,
                         f"stdout={result.stdout!r} stderr={result.stderr!r}")
        self.assertIn(f"{tool}: refusing to fetch the {catalog_id} model "
                      "weights", result.stderr)
        self.assertIn(licensing.accept_command(catalog_id), result.stderr)
        self.assertEqual(result.stdout, "")

    def _spy_log_text(self) -> str:
        try:
            with open(self.spy_log, encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return "<no log>"


class NoReceiptTests(_WeightsFixture):
    """With no receipt: refuse, fetch nothing, write nothing, exit non-zero."""

    def test_the_advertised_install_refuses_and_runs_no_installer(self) -> None:
        """The exact row the OS image runs: kilix stt --install M --default M."""
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL,
                               "--default", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)
        # The refusal is before everything, so --default did not take effect
        # either: a refused install leaves no half-applied state behind.
        self.assertFalse(
            os.path.exists(os.path.join(self.store, "settings.conf")),
            "a refused install still wrote the shared settings document")

    def test_the_refusal_names_the_command_that_obtains_consent(self) -> None:
        """And it names kilix-content's asset id, not this catalog's id.

        kilix-content files the Vosk weights under their upstream ids, so a
        refusal that echoed this catalog's id would send the user to an asset
        that does not exist.
        """
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assertIn("Run: kilix models install vosk-model-small-en-us-0.15",
                      result.stderr)
        self.assertIn("shown and accepted", result.stderr)

    def test_each_model_names_its_own_content_asset(self) -> None:
        for tool, catalog_id, asset_id in (
                ("kilix-stt", "small-en-us", "vosk-model-small-en-us-0.15"),
                ("kilix-stt", "lgraph-en-us", "vosk-model-en-us-0.22-lgraph"),
                ("kilix-stt", "vibevoice-asr-bitnet", "vibevoice-asr-bitnet"),
                ("kilix-tts", PIPER_MODEL, PIPER_MODEL)):
            with self.subTest(model=catalog_id):
                result = self.run_tool(tool, "--install", catalog_id)
                self.assertIn(f"Run: kilix models install {asset_id}",
                              result.stderr)

    def test_every_catalog_model_refuses(self) -> None:
        for catalog_id in models.MODEL_IDS:
            with self.subTest(model=catalog_id):
                before = self.store_snapshot()
                result = self.run_tool("kilix-stt", "--install", catalog_id)
                self.assert_refusal(result, "kilix-stt", catalog_id)
                self.assert_installer_never_ran(before)

    def test_the_piper_voice_refuses(self) -> None:
        before = self.store_snapshot()
        result = self.run_tool("kilix-tts", "--install", PIPER_MODEL)
        self.assert_refusal(result, "kilix-tts", PIPER_MODEL)
        self.assert_installer_never_ran(before)

    def test_an_absent_authority_refuses_rather_than_fetching(self) -> None:
        """A machine that cannot check a licence must not fetch weights."""
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL,
                               authority=False)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn("kilix-license authority is not installed", result.stderr)
        self.assert_installer_never_ran(before)

    def test_an_authority_without_the_api_refuses(self) -> None:
        stub = os.path.join(self.root, "stub")
        os.makedirs(stub, mode=0o700)
        with open(os.path.join(stub, "kilix_license.py"), "w",
                  encoding="utf-8") as handle:
            handle.write('"""Not the authority."""\n')
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL,
                               stub_authority=stub)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn("does not expose the licence authority's API",
                      result.stderr)
        self.assert_installer_never_ran(before)


class BrokenAuthorityTests(_WeightsFixture):
    """An authority that cannot answer refuses cleanly, not by crashing.

    Exit 3 carries "a human must accept a licence"; exit 1 carries "the
    installer broke". An authority that raised while being imported used to
    escape as an uncaught traceback and exit 1 (V-ACC-VERIFY F4) -- it failed
    closed, so nothing was fetched, but it inverted the one distinction the
    status exists to carry, and inside the curses screen it would have taken
    the terminal down instead of printing a message.
    """

    def stub(self, name: str, source: str) -> str:
        directory = os.path.join(self.root, name)
        os.makedirs(directory, mode=0o700)
        with open(os.path.join(directory, "kilix_license.py"), "w",
                  encoding="utf-8") as handle:
            handle.write(source)
        return directory

    def test_an_authority_that_raises_while_importing_refuses(self) -> None:
        stub = self.stub("raiser", "raise RuntimeError('authority is broken')\n")
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL,
                               stub_authority=stub)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn("could not be loaded", result.stderr)
        self.assertIn("RuntimeError", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assert_installer_never_ran(before)

    def test_an_authority_that_cannot_read_its_records_refuses(self) -> None:
        """Same class of fault, one step later: the API is there, the data is not."""
        stub = self.stub("recordless", (
            "class AssetRef: pass\n"
            "class CoverageRefused(Exception): pass\n"
            "class ReceiptStore:\n"
            "    def __init__(self, root): self.root = root\n"
            "class RecordIndex:\n"
            "    def __init__(self, records): pass\n"
            "def covers(*a, **k): return True\n"
            "def load_determined_records():\n"
            "    raise OSError('the records directory is gone')\n"
            "def require(*a, **k): raise CoverageRefused('receipt')\n"))
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL,
                               stub_authority=stub)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn("could not read its licence records", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assert_installer_never_ran(before)

    def test_the_tui_survives_a_broken_authority(self) -> None:
        """The TUI catches LicenseRefused; an escaping error would kill curses.

        Checked without a terminal by exercising the same handler the screen
        uses: the refusal must be a LicenseRefused, which is what
        ``Ui._install_model`` catches.
        """
        stub = self.stub("raiser2", "raise RuntimeError('authority is broken')\n")
        script = (
            "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "from voicelib import licensing\n"
            "try:\n"
            "    licensing.require_covering_receipt(%r)\n"
            "except licensing.LicenseRefused as error:\n"
            "    print('LicenseRefused')\n"
            % (stub, ROOT, VOSK_MODEL))
        path = os.path.join(self.root, "handler.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
        result = self.run_script(path, stub_authority=stub)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "LicenseRefused")


class TheGateNeverWritesTests(_WeightsFixture):
    """Verifying a receipt must not change the filesystem. Ever.

    kilix-voice reads receipts and never produces them, so asking the question
    must leave no trace. The authority's ``ReceiptStore.__init__`` is a
    writer's constructor -- it creates the directory and chmods it to 0700 --
    so simply constructing it to read one mutated the store: a refusal silently
    re-moded an existing store, and a store the user could not read was
    *repaired* and the install then proceeded on the strength of that repair
    (V-ACC-VERIFY F3). The gate now opens the store without that constructor.
    """

    def receipts_mode(self) -> int:
        return stat.S_IMODE(os.stat(self.receipts).st_mode)

    def make_receipts_dir(self, mode: int) -> None:
        os.makedirs(self.receipts, exist_ok=True)
        os.chmod(self.receipts, mode)
        self.addCleanup(self._restore_mode)

    def _restore_mode(self) -> None:
        try:
            os.chmod(self.receipts, 0o700)
        except OSError:
            pass

    def test_the_refusal_does_not_create_the_store_it_names(self) -> None:
        self.assertFalse(os.path.exists(self.receipts))
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn(self.receipts, result.stderr)
        self.assertFalse(os.path.exists(self.receipts),
                         "the refusal created the receipt store it named")

    def test_the_refusing_path_leaves_an_existing_store_alone(self) -> None:
        self.make_receipts_dir(0o755)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertEqual(self.receipts_mode(), 0o755,
                         "the refusing path changed the receipt store's mode")
        self.assertEqual(sorted(os.listdir(self.receipts)), [],
                         "the refusing path wrote into the receipt store")
        self.assert_installer_never_ran(before)

    def test_the_covered_path_leaves_the_store_alone_too(self) -> None:
        """Not just the refusal: verifying never writes, whatever the answer."""
        self.mint_receipt(VOSK_MODEL)
        filed = sorted(os.listdir(self.receipts))
        os.chmod(self.receipts, 0o755)
        self.addCleanup(self._restore_mode)
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.receipts_mode(), 0o755,
                         "the covered path changed the receipt store's mode")
        self.assertEqual(sorted(os.listdir(self.receipts)), filed,
                         "the covered path wrote into the receipt store")

    def test_a_store_it_cannot_read_refuses_instead_of_repairing_it(self) -> None:
        """A store at 0000 is unreadable, so the answer is no.

        It used to be chmod'ed to 0700 by the constructor, after which the
        receipt inside became visible and the install proceeded -- an install
        let through by a permission change this command performed.
        """
        self.mint_receipt(VOSK_MODEL)
        os.chmod(self.receipts, 0o000)
        self.addCleanup(self._restore_mode)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertEqual(self.receipts_mode(), 0o000,
                         "the gate repaired a store it could not read")
        self.assert_installer_never_ran(before)

    def test_the_control_the_same_store_readable_does_cover(self) -> None:
        """Without this, the test above could be refusing for any other reason."""
        self.mint_receipt(VOSK_MODEL)
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._spy_log_text().strip(),
                         f"voice install --model {VOSK_MODEL}")


class NonCoveringReceiptTests(_WeightsFixture):
    """A corrupted or non-covering receipt is no receipt at all."""

    def test_a_receipt_for_another_model_does_not_cover(self) -> None:
        self.mint_receipt(OTHER_VOSK_MODEL)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    def test_a_tampered_licensor_is_refused(self) -> None:
        self.tamper_receipt(VOSK_MODEL, "licensor", "Somebody Else Ltd.")
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn("does not cover this licence record", result.stderr)
        self.assert_installer_never_ran(before)

    def test_a_tampered_licence_text_digest_is_refused(self) -> None:
        self.tamper_receipt(VOSK_MODEL, "licence_text_digest", "d" * 64)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    def test_a_tampered_record_digest_is_refused(self) -> None:
        self.tamper_receipt(VOSK_MODEL, "record_digest", "e" * 64)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    # -- the decision class ----------------------------------------------
    #
    # `decision` is the field that separates "the user typed the agreement
    # line" (accept) from "the licence was only displayed" (record). Two of
    # the four voice models are accept and one is record, so a gate that
    # tolerated the difference would take an informational receipt for a
    # licence that requires agreement -- and OD-S says the user "gives an
    # explicit acceptance before download". The other bound fields were
    # already pinned above; this one was not, and a mutant that swallowed
    # exactly this refusal survived the whole suite (V-ACC-VERIFY F1).

    def test_a_receipt_downgraded_from_accept_to_record_is_refused(self) -> None:
        """An agreement-class licence is not covered by a display-class receipt."""
        record = self.record_for(VOSK_MODEL)
        self.assertEqual(record.expected_decision, "accept")
        self.tamper_receipt(VOSK_MODEL, "decision", "record")
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        # The authority names the field it refused, and the gate passes that
        # through, so the refusal says which class was wrong.
        self.assertIn("decision", result.stderr)
        self.assert_installer_never_ran(before)

    def test_a_receipt_upgraded_from_record_to_accept_is_refused(self) -> None:
        """And the other direction: the decision must match the record's class."""
        record = self.record_for(PIPER_MODEL)
        self.assertEqual(record.expected_decision, "record")
        self.tamper_receipt(PIPER_MODEL, "decision", "accept")
        before = self.store_snapshot()
        result = self.run_tool("kilix-tts", "--install", PIPER_MODEL)
        self.assert_refusal(result, "kilix-tts", PIPER_MODEL)
        self.assertIn("decision", result.stderr)
        self.assert_installer_never_ran(before)

    def test_only_the_decision_class_of_the_receipt_was_changed(self) -> None:
        """The control: the same receipt, untouched, does cover.

        Without this the two tests above could be refusing for some other
        reason -- a broken fixture refuses just as convincingly as a gate.
        """
        self.mint_receipt(VOSK_MODEL)
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._spy_log_text().strip(),
                         f"voice install --model {VOSK_MODEL}")

    def test_a_receipt_that_is_not_a_receipt_is_refused(self) -> None:
        record = self.record_for(VOSK_MODEL)
        os.makedirs(self.receipts, mode=0o700, exist_ok=True)
        path = LICENCE.ReceiptStore(self.receipts).path_for(
            record.digest, FIXTURE_MANIFEST)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not json at all\n")
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)


class LicenceOrderingTests(_WeightsFixture):
    """The gate runs BEFORE the fetcher is located, on all three hand-offs.

    This tree hands the terminal to a fetcher three times -- ``kilix voice
    install --model ID`` and ``kilix bonsai pull ID`` from kilix-stt, and
    ``kilix-piper-tts install ID`` from kilix-tts -- and both the code and the
    design say the receipt is checked before that fetcher is even located.
    Nothing pinned it, and a mutant that moved the call below the lookup
    survived the whole suite (V-ACC-VERIFY F2).

    It is observable, which is why it matters: on a machine with no launcher
    and no receipt, the gate first gives exit 3, "a human must accept a
    licence"; the lookup first gives exit 1, "the installer broke". Exit 3
    exists to carry exactly that distinction to an unattended caller, so on a
    bare image the advertised catalog row would report the wrong cause.
    """

    def assert_no_fetcher_is_reachable(self) -> None:
        """Fail loudly rather than pass for the wrong reason.

        If a real ``kilix`` or ``kilix-piper-tts`` were on the system PATH,
        these tests would be measuring a machine that *has* a fetcher, and
        would prove nothing about the ordering.
        """
        path = self._child_env(launcher=False)["PATH"]
        for name in ("kilix", "kilix-piper-tts"):
            self.assertIsNone(
                shutil.which(name, path=path),
                f"{name} is reachable on {path}; this test needs a machine "
                "with no fetcher installed")

    def test_the_vosk_hand_off_refuses_before_the_launcher_is_located(self) -> None:
        """Hand-off 1: kilix voice install --model ID."""
        self.assert_no_fetcher_is_reachable()
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL,
                               launcher=False)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertNotIn("launcher could not be found", result.stderr)
        self.assert_installer_never_ran(before)

    def test_the_vibevoice_hand_off_refuses_before_the_launcher_is_located(self) -> None:
        """Hand-off 2: kilix bonsai pull ID, the other branch of the same lookup."""
        self.assert_no_fetcher_is_reachable()
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", "vibevoice-asr-bitnet",
                               launcher=False)
        self.assert_refusal(result, "kilix-stt", "vibevoice-asr-bitnet")
        self.assertNotIn("launcher could not be found", result.stderr)
        self.assert_installer_never_ran(before)

    def test_the_piper_hand_off_refuses_before_the_provider_is_located(self) -> None:
        """Hand-off 3: kilix-piper-tts install ID."""
        self.assert_no_fetcher_is_reachable()
        before = self.store_snapshot()
        result = self.run_tool("kilix-tts", "--install", PIPER_MODEL,
                               launcher=False)
        self.assert_refusal(result, "kilix-tts", PIPER_MODEL)
        self.assertNotIn("is not installed", result.stderr)
        self.assert_installer_never_ran(before)

    def test_without_the_gate_the_same_machine_reports_an_installer_fault(self) -> None:
        """The control: the two orders really are distinguishable here.

        With the gate removed, the same commands on the same fetcher-less
        machine give the installer's own status and message instead. If this
        did not hold, the three assertions above would be passing for some
        reason other than the gate's position.
        """
        planted = self.plant("kilix-stt", self.GATE_STT)
        result = self.run_script(planted, "--install", VOSK_MODEL,
                                 launcher=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("launcher could not be found", result.stderr)

        planted = self.plant("kilix-tts", self.GATE_TTS)
        result = self.run_script(planted, "--install", PIPER_MODEL,
                                 launcher=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("kilix-piper-tts is not installed", result.stderr)


class RefusalNamesRunnableCommandsTests(_WeightsFixture):
    """Every command the refusal names must be one that actually runs.

    The refusal carries two lines: the acceptance route (`kilix models install
    <content asset id>`), which is the cure, and the probe (`kilix-stt
    --check-licence <model>` / `kilix-tts --check-licence <model>`), which
    re-checks without fetching. The acceptance route is not wired yet -- that
    is the `kilix` repository's item, recorded in README.md -- but the probe
    is in this tree, so nothing stops it being tested, and a probe line naming
    a command the tool rejects would be the same defect in a new place.
    """

    def _second_line(self, catalog_id: str, tool: str) -> str:
        result = self.run_tool(tool, "--install", catalog_id)
        self.assertEqual(result.returncode, licensing.LICENCE_REFUSED_EXIT)
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        self.assertTrue(lines[-1].startswith("Re-check with: "),
                        f"last refusal line was {lines[-1]!r}")
        return lines[-1].split("Re-check with: ", 1)[1].strip()

    def test_the_probe_line_is_a_command_this_tree_accepts(self) -> None:
        for catalog_id, tool in ((VOSK_MODEL, "kilix-stt"),
                                 (OTHER_VOSK_MODEL, "kilix-stt"),
                                 ("vibevoice-asr-bitnet", "kilix-stt"),
                                 (PIPER_MODEL, "kilix-tts")):
            with self.subTest(model=catalog_id):
                named = self._second_line(catalog_id, tool)
                self.assertEqual(
                    named, licensing.check_command(catalog_id))
                probe_tool, *probe_args = named.split()
                before = self.store_snapshot()
                result = self.run_tool(probe_tool, *probe_args)
                # 3 is the licence answer. 2 would mean the tool rejected its
                # own advice, which is the defect this test exists to stop.
                self.assertEqual(result.returncode,
                                 licensing.LICENCE_REFUSED_EXIT,
                                 f"{named} -> exit {result.returncode}: "
                                 f"{result.stderr!r}")
                self.assert_installer_never_ran(before)

    def test_the_probe_line_answers_zero_once_a_receipt_covers(self) -> None:
        """And it is the same answer the install would give, both ways round."""
        for catalog_id, tool in ((VOSK_MODEL, "kilix-stt"),
                                 (PIPER_MODEL, "kilix-tts")):
            with self.subTest(model=catalog_id):
                named = licensing.check_command(catalog_id)
                probe_tool, *probe_args = named.split()
                self.mint_receipt(catalog_id)
                before = self.store_snapshot()
                result = self.run_tool(probe_tool, *probe_args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(catalog_id, result.stdout)
                self.assert_installer_never_ran(before)

    def test_the_refusal_still_names_the_acceptance_route_first(self) -> None:
        """The probe is an addition, not a replacement: both lines are there."""
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assertIn("Run: kilix models install vosk-model-small-en-us-0.15",
                      result.stderr)
        self.assertIn("Re-check with: kilix-stt --check-licence small-en-us",
                      result.stderr)
        self.assertLess(
            result.stderr.index("Run: kilix models install"),
            result.stderr.index("Re-check with:"),
            "the cure must be named before the diagnostic")


class CoveringReceiptTests(_WeightsFixture):
    """With a covering receipt, the install proceeds exactly as before."""

    def test_a_covering_receipt_proceeds(self) -> None:
        self.mint_receipt(VOSK_MODEL)
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        # The control for every "fetches nothing" assertion above: the spy is
        # reachable, it records the pinned installer's own argv, and its write
        # under the model store is visible to store_snapshot().
        self.assertEqual(self._spy_log_text().strip(),
                         f"voice install --model {VOSK_MODEL}")
        self.assertIn(os.path.join("voice", "models", "fetched-by-kilix",
                                   "final.mdl"),
                      self.store_snapshot())

    def test_the_vibevoice_weights_still_go_to_the_pinned_store(self) -> None:
        self.mint_receipt("vibevoice-asr-bitnet")
        result = self.run_tool("kilix-stt", "--install", "vibevoice-asr-bitnet")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._spy_log_text().strip(),
                         "bonsai pull vibevoice-asr-bitnet")

    def test_a_covering_receipt_lets_the_piper_voice_install(self) -> None:
        self.mint_receipt(PIPER_MODEL)
        result = self.run_tool("kilix-tts", "--install", PIPER_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._spy_log_text().strip(),
                         f"install {PIPER_MODEL}")


class LibraryIsNotWeightsTests(_WeightsFixture):
    """The Vosk library is Apache-2.0 code. It is not gated."""

    def test_the_catalog_and_settings_surfaces_need_no_receipt(self) -> None:
        for tool, args in (("kilix-stt", ("--models",)),
                           ("kilix-stt", ("--models", "--json")),
                           ("kilix-stt", ("--print",)),
                           ("kilix-tts", ("--models",))):
            with self.subTest(tool=tool, args=args):
                result = self.run_tool(tool, *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("refusing to fetch", result.stderr)

    def test_those_surfaces_work_without_the_authority_too(self) -> None:
        """No receipt, no authority: everything but a weight fetch still runs."""
        for tool, args in (("kilix-stt", ("--models",)),
                           ("kilix-stt", ("--print",)),
                           ("kilix-tts", ("--models",))):
            with self.subTest(tool=tool, args=args):
                result = self.run_tool(tool, *args, authority=False)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_resolving_and_reporting_the_library_needs_no_receipt(self) -> None:
        """The library path is reported with no receipt anywhere in sight."""
        library = os.path.join(self.data, paths.VOICE_LEAF, "lib", "current")
        os.makedirs(library, mode=0o700)
        with open(os.path.join(library, "libvosk.so"), "wb") as handle:
            handle.write(b"\x7fELF fixture")
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--print", authority=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("libvosk.so", result.stdout)
        self.assertEqual(before, self.store_snapshot())

    def test_an_os_supplied_synthesis_family_is_not_a_weight_fetch(self) -> None:
        """espeak/mbrola are distribution packages, so --install never takes them.

        They are rejected by the argument parser, before the licence gate and
        before any provider, and the refusal says so: nothing about a licence,
        because there are no weights here to license.
        """
        for catalog_id in ("espeak", "mbrola"):
            with self.subTest(model=catalog_id):
                before = self.store_snapshot()
                result = self.run_tool("kilix-tts", "--install", catalog_id)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("invalid choice", result.stderr)
                self.assertNotIn("refusing to fetch", result.stderr)
                self.assert_installer_never_ran(before)

    def test_only_the_piper_weights_are_installable_by_kilix_tts(self) -> None:
        """The one synthesis id --install accepts is the one that is weights."""
        result = self.run_tool("kilix-tts", "--install", "not-a-model")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(f"choose from {PIPER_MODEL}", result.stderr)


class CheckLicenceProbeTests(_WeightsFixture):
    """`kilix-stt --check-licence MODEL`: the answer, with no fetch either way.

    A fetcher outside this tree cannot import voicelib, but it can run one
    command and stop on its status. Kilix's own install-kilix-voice.sh, which
    pleb reaches as `kilix voice install`, is that fetcher.
    """

    def test_no_receipt_answers_three_and_fetches_nothing(self) -> None:
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--check-licence", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    def test_a_covering_receipt_answers_zero_and_still_fetches_nothing(self) -> None:
        self.mint_receipt(VOSK_MODEL)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--check-licence", VOSK_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(VOSK_MODEL, result.stdout)
        self.assertIn(licensing.RECEIPTS_LEAF, result.stdout)
        # Answering is not installing: the probe is safe before every fetch.
        self.assert_installer_never_ran(before)

    def test_an_absent_authority_answers_three(self) -> None:
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--check-licence", VOSK_MODEL,
                               authority=False)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    def test_it_answers_for_every_catalog_model(self) -> None:
        for catalog_id in models.MODEL_IDS:
            with self.subTest(model=catalog_id):
                result = self.run_tool("kilix-stt", "--check-licence",
                                       catalog_id)
                self.assertEqual(result.returncode,
                                 licensing.LICENCE_REFUSED_EXIT, result.stderr)
                self.mint_receipt(catalog_id)
                result = self.run_tool("kilix-stt", "--check-licence",
                                       catalog_id)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_an_unknown_model_is_rejected_by_the_parser(self) -> None:
        result = self.run_tool("kilix-stt", "--check-licence", "not-a-model")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("invalid choice", result.stderr)


class PlantedRegressionTests(_WeightsFixture):
    """Restoring the unconditional fetch must fail the assertions above."""

    def test_a_kilix_stt_without_the_gate_fetches_and_is_caught(self) -> None:
        planted = self.plant("kilix-stt", self.GATE_STT)
        before = self.store_snapshot()
        result = self.run_script(planted, "--install", VOSK_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Every assertion NoReceiptTests makes must now fail. If any of them
        # still held, they would be proving nothing about the real command.
        with self.assertRaises(AssertionError):
            self.assert_installer_never_ran(before)
        with self.assertRaises(AssertionError):
            self.assert_refusal(result, "kilix-stt", VOSK_MODEL)

    def test_a_kilix_tts_without_the_gate_fetches_and_is_caught(self) -> None:
        planted = self.plant("kilix-tts", self.GATE_TTS)
        before = self.store_snapshot()
        result = self.run_script(planted, "--install", PIPER_MODEL)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(AssertionError):
            self.assert_installer_never_ran(before)
        with self.assertRaises(AssertionError):
            self.assert_refusal(result, "kilix-tts", PIPER_MODEL)


class GateUnitTests(unittest.TestCase):
    """The parts of the gate that need no subprocess."""

    def test_the_refusal_status_is_three_and_is_not_success(self) -> None:
        """Three constants the rest of the suite is measured against.

        This was called ``test_importing_the_gate_touches_no_store``, which it
        never tested (V-ACC-VERIFY F5); the import claim is now tested, for
        real, by :class:`ImportIsInertTests`. Keeping the literal 3 here
        matters: every subprocess assertion elsewhere compares against
        ``licensing.LICENCE_REFUSED_EXIT``, so a change to that constant would
        move them all together and only this test would notice.
        """
        self.assertEqual(licensing.RECEIPTS_LEAF, "license-receipts")
        self.assertEqual(licensing.LICENCE_REFUSED_EXIT, 3)
        self.assertNotEqual(licensing.LICENCE_REFUSED_EXIT, 0)

    def test_the_receipt_store_follows_the_stack_root(self) -> None:
        root = tempfile.mkdtemp(prefix="v-acc-root-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        previous = os.environ.get("GPU_TERMINAL_HOME")
        os.environ["GPU_TERMINAL_HOME"] = root
        self.addCleanup(_restore, "GPU_TERMINAL_HOME", previous)
        self.assertEqual(licensing.receipt_store_root(),
                         os.path.join(root, licensing.RECEIPTS_LEAF))

    def test_the_receipt_store_can_be_relocated(self) -> None:
        previous = os.environ.get(licensing.ENV_RECEIPTS)
        os.environ[licensing.ENV_RECEIPTS] = "/elsewhere/receipts"
        self.addCleanup(_restore, licensing.ENV_RECEIPTS, previous)
        self.assertEqual(licensing.receipt_store_root(), "/elsewhere/receipts")

    def test_the_refusal_names_the_model_and_the_command(self) -> None:
        error = licensing.LicenseRefused(VOSK_MODEL, "no licence receipt")
        self.assertIn(VOSK_MODEL, str(error))
        self.assertIn("kilix models install vosk-model-small-en-us-0.15",
                      str(error))
        self.assertEqual(error.catalog_id, VOSK_MODEL)
        self.assertIsInstance(licensing.AuthorityUnavailable(VOSK_MODEL, "x"),
                              licensing.LicenseRefused)

    def test_every_catalog_model_maps_to_a_content_asset(self) -> None:
        """No model may fall back to echoing this catalog's id by accident."""
        for catalog_id in models.MODEL_IDS + (PIPER_MODEL,):
            with self.subTest(model=catalog_id):
                self.assertIn(catalog_id, licensing.CONTENT_ASSET_ID)
                self.assertTrue(licensing.content_asset_id(catalog_id))

    def test_an_unmapped_id_falls_back_to_itself(self) -> None:
        self.assertEqual(licensing.content_asset_id("something-new"),
                         "something-new")


class ImportIsInertTests(unittest.TestCase):
    """voicelib/licensing.py: "Importing this module performs no ... work".

    Asserted in the module docstring and, until now, by nothing (V-ACC-VERIFY
    F5). It matters because every command in this tree imports the module at
    start-up, including on paths that must never touch the receipt store, and
    because a gate that created its own store at import would answer its own
    question.
    """

    PROBE = """
import json, os, sys
root = sys.argv[1]
os.environ["GPU_TERMINAL_HOME"] = os.path.join(root, "gt")
os.environ["KILIX_STORAGE_HOME"] = os.path.join(root, "kilix")
os.environ.pop("KILIX_VOICE_LICENSE_RECEIPTS", None)


def walk():
    found = []
    for base, dirs, files in os.walk(root):
        for name in dirs + files:
            found.append(os.path.relpath(os.path.join(base, name), root))
    return sorted(found)


before = walk()
from voicelib import licensing  # noqa: E402
print(json.dumps({
    "created": [p for p in walk() if p not in before],
    "authority_imported": "kilix_license" in sys.modules,
    "receipt_root": licensing.receipt_store_root(),
}))
"""

    def test_importing_the_gate_creates_nothing_and_asks_nobody(self) -> None:
        root = tempfile.mkdtemp(prefix="v-acc-import-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        probe = os.path.join(root, "probe.py")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write(self.PROBE)
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                ([ROOT, authority_source()] if LICENCE is not None else [ROOT])),
            "HOME": root,
            "TMPDIR": root,
        }
        result = subprocess.run(
            [sys.executable, probe, root], env=env, capture_output=True,
            text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)
        # probe.py itself is the only thing under root before the import, and
        # the import must add nothing -- no store, no receipts directory, no
        # stack home.
        self.assertEqual(seen["created"], [],
                         f"importing the gate created {seen['created']}")
        # And it asked the authority nothing: kilix_license is importable in
        # this process (it is on PYTHONPATH) but was never imported.
        self.assertFalse(seen["authority_imported"],
                         "importing the gate imported the licence authority")
        # The store root it would read is under the fresh stack home, and it
        # does not exist, which is what makes "created nothing" meaningful.
        self.assertTrue(seen["receipt_root"].startswith(root))
        self.assertFalse(os.path.exists(seen["receipt_root"]))

    def test_the_module_runs_no_call_at_import_time(self) -> None:
        """A structural control for the behavioural test above.

        The probe can only see what an import *did* in that one environment.
        This says the module has no top-level statement that could do
        anything: every top-level node is an import, a constant assignment, a
        function or a class.
        """
        source = os.path.join(ROOT, "voicelib", "licensing.py")
        with open(source, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef,
                   ast.AsyncFunctionDef, ast.ClassDef, ast.Expr, ast.Assign,
                   ast.AnnAssign)
        for node in tree.body:
            self.assertIsInstance(node, allowed)
            if isinstance(node, ast.Expr):
                # docstrings only
                self.assertIsInstance(node.value, ast.Constant)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for inner in ast.walk(node.value) if node.value else ():
                    self.assertNotIsInstance(
                        inner, ast.Call,
                        "a module-level assignment calls something")


class TuiLicenceTests(_WeightsFixture):
    """The TUI install key is gated too, and it is gated on purpose.

    ``kilix-stt``'s Models tab installs on ``i``. Nothing in this file
    exercised it; the one test that caught a TUI-only bypass was a mock-based
    catalog test about something else, so the coverage was incidental and
    would evaporate with that test (V-ACC-VERIFY F12). These drive the real
    curses screen through a pty and press the real key.
    """

    KEYS = (b"3", b"i")          # Models section, then install
    SETTLE = 0.6
    BUDGET = 25.0

    def drive_tui(self, *, until) -> str:
        """Run the real kilix-stt TUI, press 3 then i, and return what it drew.

        ``until`` is called with the bytes drawn so far and stops the run when
        it returns True, so neither arm waits out the whole budget.
        """
        import pty
        import select

        env = dict(self._child_env())
        env.update(TERM="xterm", LINES="40", COLUMNS="120")
        pid, fd = pty.fork()
        if pid == 0:                                    # pragma: no cover
            try:
                os.execve(sys.executable,
                          [sys.executable, os.path.join(ROOT, "kilix-stt")],
                          env)
            finally:
                os._exit(127)
        drawn = b""
        sent = False
        deadline = time.monotonic() + self.BUDGET
        try:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([fd], [], [], 0.4)
                if ready:
                    try:
                        chunk = os.read(fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    drawn += chunk
                if not sent and b"Models" in drawn and len(drawn) > 400:
                    time.sleep(self.SETTLE)
                    for key in self.KEYS:
                        os.write(fd, key)
                        time.sleep(self.SETTLE)
                    sent = True
                if sent and until(drawn):
                    break
            self.assertTrue(sent, "the TUI never drew its Models tab")
            try:
                os.write(fd, b"\nq")
            except OSError:
                pass
            time.sleep(0.3)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        return drawn.decode("utf-8", "replace")

    def test_the_tui_install_key_refuses_and_starts_no_fetcher(self) -> None:
        before = self.store_snapshot()
        drawn = self.drive_tui(until=lambda seen: b"refusing to fetch" in seen)
        flat = " ".join(drawn.split())
        self.assertIn("refusing to fetch", flat,
                      f"the TUI drew: {flat[-300:]!r}")
        self.assert_installer_never_ran(before)

    def test_the_same_key_with_a_covering_receipt_does_fetch(self) -> None:
        """The control: without it the refusal above could be a broken keypress."""
        self.mint_receipt(VOSK_MODEL)
        self.drive_tui(until=lambda _: os.path.exists(self.spy_log))
        self.assertTrue(os.path.exists(self.spy_log),
                        "the TUI key did not reach the installer even with a "
                        "covering receipt; the refusal test proves nothing")
        self.assertEqual(self._spy_log_text().strip(),
                         f"voice install --model {VOSK_MODEL}")


def _restore(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
