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

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
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
        for directory in (self.home, self.tmp, self.spy_dir):
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
                   stub_authority: str | None = None) -> dict[str, str]:
        """The environment one command runs in: private store, spy installer."""
        import_path = [ROOT]
        if stub_authority is not None:
            import_path.insert(0, stub_authority)
        elif authority:
            import_path.append(authority_source())
        return {
            "PATH": os.pathsep.join((self.spy_dir, "/usr/bin", "/bin")),
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
            "KILIX_HOME": self.spy_dir,
        }

    def run_tool(self, tool: str, *args: str, **kwargs) -> subprocess.CompletedProcess:
        return self.run_script(os.path.join(ROOT, tool), *args, **kwargs)

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

    def forge_receipt(self, catalog_id: str, field: str, value: str) -> str:
        """Write a receipt filed under the right record with one field changed.

        The file is named exactly as the authority names a covering receipt, so
        the store finds it. Only its bytes are wrong, which is the forgery a
        record-scoped lookup has to refuse rather than accept on the strength
        of its name.
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


class NonCoveringReceiptTests(_WeightsFixture):
    """A receipt that does not cover this model is no receipt at all."""

    def test_a_receipt_for_another_model_does_not_cover(self) -> None:
        self.mint_receipt(OTHER_VOSK_MODEL)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    def test_a_forged_licensor_is_refused(self) -> None:
        self.forge_receipt(VOSK_MODEL, "licensor", "Somebody Else Ltd.")
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assertIn("does not cover this licence record", result.stderr)
        self.assert_installer_never_ran(before)

    def test_a_forged_licence_text_digest_is_refused(self) -> None:
        self.forge_receipt(VOSK_MODEL, "licence_text_digest", "d" * 64)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

    def test_a_forged_record_digest_is_refused(self) -> None:
        self.forge_receipt(VOSK_MODEL, "record_digest", "e" * 64)
        before = self.store_snapshot()
        result = self.run_tool("kilix-stt", "--install", VOSK_MODEL)
        self.assert_refusal(result, "kilix-stt", VOSK_MODEL)
        self.assert_installer_never_ran(before)

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

    GATE_STT = "    licensing.require_covering_receipt(spec.catalog_id)\n"
    GATE_TTS = "    licensing.require_covering_receipt(catalog_id)\n"

    def plant(self, tool: str, gate: str) -> str:
        """Copy one command with its licence gate removed."""
        with open(os.path.join(ROOT, tool), encoding="utf-8") as handle:
            source = handle.read()
        self.assertEqual(source.count(gate), 1,
                         f"{tool} no longer holds exactly one licence gate")
        planted = os.path.join(self.root, f"regression-{tool}")
        with open(planted, "w", encoding="utf-8") as handle:
            handle.write(source.replace(gate, ""))
        return planted

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

    def test_importing_the_gate_touches_no_store(self) -> None:
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


def _restore(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
