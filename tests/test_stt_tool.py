"""Offline coverage for kilix-stt's model catalog and lazy-install CLI."""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from voicelib import models, settings, stt


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_stt_tool():
    loader = importlib.machinery.SourceFileLoader(
        "kilix_stt_model_test", str(ROOT / "kilix-stt"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise AssertionError("could not load kilix-stt")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


tool = load_stt_tool()


class ModelCatalogTests(unittest.TestCase):

    def test_catalog_exposes_every_shared_model(self) -> None:
        self.assertEqual(
            tuple(tool.MODEL_BY_ID),
            settings.SPEC[settings.KEY_STT_MODEL][1],
        )
        self.assertEqual(
            tool.MODEL_BY_ID["vibevoice-asr-bitnet"].size,
            1705771590,
        )
        self.assertIs(tool.MODELS, models.MODELS)
        self.assertEqual(models.CATALOG_SCHEMA, "kilix.speech.models/v1")

    def test_vosk_models_use_the_pinned_voice_installer(self) -> None:
        with mock.patch.object(tool, "kilix_launcher", return_value="/kilix"):
            for catalog_id in ("small-en-us", "lgraph-en-us"):
                with self.subTest(model=catalog_id):
                    self.assertEqual(
                        tool.model_install_argv(tool.MODEL_BY_ID[catalog_id]),
                        ["/kilix", "voice", "install", "--model", catalog_id],
                    )

    def test_vibevoice_uses_bonsais_shared_weight_store(self) -> None:
        with mock.patch.object(tool, "kilix_launcher", return_value="/kilix"):
            self.assertEqual(
                tool.model_install_argv(
                    tool.MODEL_BY_ID["vibevoice-asr-bitnet"]),
                ["/kilix", "bonsai", "pull", "vibevoice-asr-bitnet"],
            )

    def test_models_output_is_a_download_free_install_menu(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {"KILIX_DATA_HOME": os.path.join(root, "data")},
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                tool._print_models()
        shown = output.getvalue()
        for catalog_id in tool.MODEL_BY_ID:
            self.assertIn(f"model={catalog_id}", shown)
            self.assertIn(f"install=kilix stt --install {catalog_id}", shown)
        self.assertIn(
            "model=vibevoice-asr-bitnet engine=vibevoice", shown)
        self.assertIn("runtime_supported=no", shown)

    def test_json_catalog_is_versioned_complete_and_download_free(self) -> None:
        # GPU_TERMINAL_SETTINGS_FILE as well as KILIX_DATA_HOME: a live Kilix
        # session exports it, and the catalog's default_model is read through
        # it. Without this the assertion below tests whichever model the
        # operator happens to have selected, and fails on any machine where
        # that is not the default. Every other test in this file already
        # points it at a sandbox.
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {"KILIX_DATA_HOME": os.path.join(root, "data"),
             "GPU_TERMINAL_SETTINGS_FILE": os.path.join(root, "settings.conf")},
        ), mock.patch.object(tool, "install_model") as install:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(tool.main(["--models", "--json"]), 0)

        install.assert_not_called()
        document = json.loads(output.getvalue())
        self.assertEqual(document["schema"], "kilix.speech.models/v1")
        self.assertEqual(document["default_model"], "small-en-us")
        records = {record["id"]: record for record in document["models"]}
        self.assertEqual(tuple(records), tuple(tool.MODEL_BY_ID))
        for catalog_id, spec in tool.MODEL_BY_ID.items():
            with self.subTest(model=catalog_id):
                record = records[catalog_id]
                self.assertEqual(record["engine"], spec.engine)
                self.assertEqual(record["download_bytes"], spec.size)
                self.assertEqual(
                    record["runtime_supported"], spec.runtime_supported)
                self.assertEqual(record["install_and_default_argv"], [
                    "kilix", "stt", "--install", catalog_id,
                    "--default", catalog_id,
                ])

    def test_json_requires_the_catalog_action(self) -> None:
        with self.assertRaises(SystemExit) as caught, mock.patch(
                "sys.stderr", new=io.StringIO()):
            tool.main(["--json"])
        self.assertEqual(caught.exception.code, 2)

    def test_tui_install_repairs_vosk_even_when_model_directory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {"KILIX_DATA_HOME": os.path.join(root, "data")},
        ):
            model = tool.paths.model_dir("small-en-us")
            for relative in tool.MODEL_REQUIRED_FILES[stt.ENGINE_VOSK]:
                target = os.path.join(model, relative)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                pathlib.Path(target).write_bytes(b"fixture")
            ui = object.__new__(tool.Ui)
            ui._selected = [0, 0, 0]
            ui._screen = object()
            ui._message = ""
            ui._rescan = mock.Mock()
            with mock.patch.object(tool, "install_model", return_value=0) as install:
                ui._install_model()

        install.assert_called_once_with(
            tool.MODEL_BY_ID["small-en-us"], ui._screen)
        self.assertIn("Installed or verified", ui._message)


class ModelCliTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings_path = os.path.join(
            self.temporary.name, "settings.conf")
        self.environment = mock.patch.dict(os.environ, {
            "HOME": self.temporary.name,
            "GPU_TERMINAL_HOME": self.temporary.name,
            "GPU_TERMINAL_SETTINGS_FILE": self.settings_path,
            "KILIX_DATA_HOME": os.path.join(self.temporary.name, "data"),
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop(stt.ENV_MODEL, None)

    def test_install_and_default_can_be_one_explicit_action(self) -> None:
        spec = tool.MODEL_BY_ID["lgraph-en-us"]
        with mock.patch.object(tool, "install_model", return_value=0) as install:
            self.assertEqual(tool.main([
                "--install", spec.catalog_id,
                "--default", spec.catalog_id,
            ]), 0)
        install.assert_called_once_with(spec)
        self.assertEqual(settings.stt_model(), spec.catalog_id)
        self.assertEqual(settings.stt_engine(), spec.engine)

    def test_each_default_also_selects_its_matching_engine(self) -> None:
        for spec in tool.MODELS:
            with self.subTest(model=spec.catalog_id):
                self.assertEqual(tool.main(["--default", spec.catalog_id]), 0)
                self.assertEqual(settings.stt_model(), spec.catalog_id)
                self.assertEqual(settings.stt_engine(), spec.engine)

    def test_failed_install_does_not_change_the_default(self) -> None:
        with mock.patch.object(tool, "install_model", return_value=19):
            self.assertEqual(tool.main([
                "--install", "lgraph-en-us",
                "--default", "lgraph-en-us",
            ]), 19)
        self.assertEqual(settings.stt_model(), "small-en-us")
        self.assertEqual(settings.stt_engine(), "vosk")


if __name__ == "__main__":
    unittest.main()
