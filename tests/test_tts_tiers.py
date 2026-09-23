"""Read-only tier selection and refusal before engine startup."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from voicelib import tts_tiers as tiers
from tests.test_tts_tool import load_tool


def provider(request, task):
    return {"candidates": [{**row, "verdict": "estimated-fit", "budget": {"gpu_index": 0}}
                            for row in request["models"]], "notes": []}


class TierTests(unittest.TestCase):
    def availability(self):
        return {key: (True, True, "fixture missing") for key in tiers.TIER_IDS}

    def report(self, available=None, run=provider):
        with patch.object(tiers, "availability", return_value=available or self.availability()), \
                patch.object(tiers.sizing, "recommend_request", side_effect=run):
            return tiers.report()

    def test_all_seven_tiers_are_explicitly_selectable(self):
        result = self.report()
        self.assertEqual([row["tier"] for row in result["candidates"]], list(tiers.TIER_IDS))
        for row in result["candidates"]:
            args = SimpleNamespace(tier=row["tier"], voice=None)
            tiers.select(args, result)
            self.assertTrue(row["selectable"])
            if args.tier == "qwen-gpu":
                self.assertEqual((args.device, args.attention), ("cuda:0", "flash_attention_2"))
            elif args.tier == "qwen-base-gpu":
                self.assertEqual((args.device, args.attention, args.synthetic_reference),
                                 ("cuda:0", "flash_attention_2", True))
                self.assertEqual(args.qwen_model_dir,
                                 str(tiers.qwen_directory(tiers.QWEN_BASE_ID)))
            elif args.tier == "qwen-cpu":
                self.assertEqual((args.device, args.attention), ("cpu", "sdpa"))
            elif args.tier == "pocket-cpu":
                self.assertEqual((args.pocket_model_dir, args.voice),
                                 (str(tiers.pocket_directory()), "Alba"))
            elif args.tier == "small":
                self.assertEqual((args.model, args.voice), ("mbrola", "us1"))

    def test_unknown_or_insufficient_memory_never_selectable(self):
        for verdict in ("unknown", "does-not-fit", "unsupported"):
            def run(request, task):
                result = provider(request, task)
                for row in result["candidates"]:
                    row["verdict"] = verdict
                return result
            result = self.report(run=run)
            self.assertTrue(all(not row["selectable"] for row in result["candidates"]))
            with self.assertRaises(tiers.sizing.SizerError):
                tiers.select(SimpleNamespace(tier="neural"), result)

    def test_missing_models_or_runtime_not_selectable_even_if_memory_fits(self):
        for installed, runtime in ((False, True), (True, False), (False, False)):
            result = self.report({key: (installed, runtime, "missing") for key in tiers.TIER_IDS})
            self.assertTrue(all(not row["selectable"] for row in result["candidates"]))

    def test_pocket_fit_visible_before_runtime_and_first_use_requires_it(self):
        available = self.availability()
        available["pocket-cpu"] = (False, False, "runtime missing")
        def inspect(request, task):
            pocket = next(row for row in request["models"]
                          if row["id"] == "audition-pocket-tts-english-python-alba-cpu")
            self.assertTrue(pocket["runtime_supported"])
            return provider(request, task)
        row = next(row for row in self.report(available, inspect)["candidates"]
                   if row["tier"] == "pocket-cpu")
        self.assertFalse(row["selectable"] or row["installable"])
        available["pocket-cpu"] = (False, True, "weights missing")
        row = next(row for row in self.report(available)["candidates"]
                   if row["tier"] == "pocket-cpu")
        self.assertTrue(row["installable"])

    def test_pocket_tier_rejects_other_voice_before_install(self):
        result = self.report()
        for overrides in ({"voice": "Ryan"}, {"rate": 150}, {"language": "English"}):
            args = SimpleNamespace(tier="pocket-cpu", voice=None, rate=None, language=None)
            vars(args).update(overrides)
            with self.assertRaises(tiers.sizing.SizerError):
                tiers.select(args, result)

    def test_pocket_weight_probe_checks_all_pinned_sizes_without_hashing(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(tiers, "pocket_directory", return_value=Path(temp)), \
                patch.object(tiers.pocket, "PINNED", {"model.safetensors": (3, "digest")}):
            self.assertFalse(tiers.pocket_installed())
            (Path(temp) / "model.safetensors").write_bytes(b"abc")
            self.assertTrue(tiers.pocket_installed())
            (Path(temp) / "model.safetensors").write_bytes(b"ab")
            self.assertFalse(tiers.pocket_installed())

    def test_cpu_fit_remains_visible_before_lazy_runtime_install(self):
        available = self.availability()
        available["qwen-cpu"] = (False, False, "runtime missing")
        result = self.report(available)
        row = next(row for row in result["candidates"] if row["tier"] == "qwen-cpu")
        self.assertEqual(row["verdict"], "estimated-fit")
        self.assertFalse(row["selectable"] or row["installable"])

    def test_gpu_fit_remains_visible_before_lazy_runtime_install(self):
        available = self.availability()
        available["qwen-gpu"] = (False, False, "runtime missing")
        def inspect_request(request, task):
            gpu = next(row for row in request["models"] if row["backend"] == "cuda")
            self.assertTrue(gpu["runtime_supported"])
            return provider(request, task)
        result = self.report(available, inspect_request)
        row = next(row for row in result["candidates"] if row["tier"] == "qwen-gpu")
        self.assertEqual(row["verdict"], "estimated-fit")
        self.assertFalse(row["selectable"] or row["installable"])

    def test_base_uses_measured_profile_and_requires_synthetic_reference(self):
        def inspect_request(request, task):
            base = next(row for row in request["models"]
                        if row["id"] == tiers.QWEN_BASE_ID + "-cuda")
            self.assertEqual((base["backend"], base["runtime_supported"]),
                             ("cuda", True))
            return provider(request, task)
        result = self.report(run=inspect_request)
        row = next(row for row in result["candidates"] if row["tier"] == "qwen-base-gpu")
        self.assertEqual(row["verdict"], "estimated-fit")
        with self.assertRaises(tiers.sizing.SizerError):
            tiers.select(SimpleNamespace(tier="qwen-base-gpu", voice="Ryan"), result)

    def test_missing_piper_model_can_be_selected_for_first_use(self):
        available = self.availability()
        available["neural"] = (False, True, "model missing")
        result = self.report(available)
        piper = next(row for row in result["candidates"] if row["tier"] == "neural")
        self.assertTrue(piper["installable"])
        args = SimpleNamespace(tier="neural", model=None, voice=None)
        tiers.select(args, result)
        self.assertEqual(args.model, "piper-en-us-kristin-medium")
        for unavailable in ((False, False), (True, False)):
            available["neural"] = (*unavailable, "unavailable")
            self.assertFalse(next(row for row in self.report(available)["candidates"]
                                  if row["tier"] == "neural")["installable"])

    def test_missing_qwen_weights_can_be_selected_only_with_ready_runtime(self):
        available = self.availability()
        available["qwen-cpu"] = (False, True, "weights missing")
        result = self.report(available)
        row = next(row for row in result["candidates"] if row["tier"] == "qwen-cpu")
        self.assertTrue(row["installable"])
        args = SimpleNamespace(tier="qwen-cpu", voice=None)
        tiers.select(args, result)
        self.assertEqual(args.qwen_model_dir, str(tiers.qwen_directory()))
        available["qwen-cpu"] = (False, False, "runtime missing")
        row = next(row for row in self.report(available)["candidates"]
                   if row["tier"] == "qwen-cpu")
        self.assertFalse(row["installable"])

    def test_other_gpu_budget_not_used_for_gpu_zero(self):
        def run(request, task):
            result = provider(request, task)
            for row in result["candidates"]:
                row["budget"]["gpu_index"] = 1
            return result
        result = self.report(run=run)
        self.assertFalse(result["candidates"][-1]["selectable"])
        self.assertTrue(result["candidates"][0]["selectable"])

    def test_qwen_probe_uses_current_python_and_fails_closed(self):
        with patch.object(tiers.importlib.util, "find_spec", return_value=object()), \
                patch.object(tiers.sizing, "_run", return_value=b"[true, true]") as run, \
                patch.dict(tiers.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual(tiers.qwen_runtime(), (True, True))
            self.assertEqual(run.call_args.args[0][0], tiers.sys.executable)
            with patch.dict(tiers.os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"}):
                self.assertEqual(tiers.qwen_runtime(), (True, False))
            run.return_value = b"not json"
            self.assertEqual(tiers.qwen_runtime(), (False, False))
            run.return_value = b"[1, 1]"
            self.assertEqual(tiers.qwen_runtime(), (False, False))
            run.side_effect = tiers.sizing.SizerError("timeout")
            self.assertEqual(tiers.qwen_runtime(), (False, False))

    def test_qwen_missing_dependencies_does_not_spawn(self):
        with patch.object(tiers.importlib.util, "find_spec", return_value=None), \
                patch.object(tiers.sizing, "_run") as run:
            self.assertEqual(tiers.qwen_runtime(), (False, False))
            run.assert_not_called()

    def test_local_weights_require_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tiers, "qwen_directory", return_value=Path(temp)):
            root = Path(temp)
            self.assertFalse(tiers.qwen_installed())
            (root / "config.json").write_text(json.dumps({"tts_model_type": "custom_voice"}))
            (root / "model.safetensors").touch()
            self.assertFalse(tiers.qwen_installed())
            (root / "speech_tokenizer").mkdir()
            (root / "speech_tokenizer/model.safetensors").touch()
            self.assertTrue(tiers.qwen_installed())
            (root / "config.json").write_text("[]")
            self.assertFalse(tiers.qwen_installed())

    def test_base_weights_require_base_config(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(tiers, "qwen_directory", return_value=Path(temp)):
            root = Path(temp)
            (root / "config.json").write_text(json.dumps({"tts_model_type": "base"}))
            (root / "model.safetensors").touch()
            (root / "speech_tokenizer").mkdir()
            (root / "speech_tokenizer/model.safetensors").touch()
            self.assertTrue(tiers.qwen_installed(tiers.QWEN_BASE_ID))
            self.assertFalse(tiers.qwen_installed())

    def test_list_does_not_start_session_or_change_settings(self):
        tool = load_tool()
        with patch.object(tiers, "report", return_value=self.report()), \
                patch.object(tool.settings, "update") as save, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(tool.main(["--tiers", "--json"]), 0)
            self.assertEqual(len(json.loads(output.getvalue())["candidates"]), 7)
            save.assert_not_called()

    def test_conflicting_cli_flags_refused_before_probing(self):
        tool = load_tool()
        for argv in (["--tier", "neural"], ["--tiers", "--speak", "hello"],
                     ["--json"], ["--interactive", "--tier", "neural", "--model", "espeak"],
                     ["--interactive", "--tier", "qwen-cpu", "--download-qwen", tiers.QWEN_ID]):
            with patch.object(tiers, "report") as report, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    tool.main(argv)
                self.assertEqual(caught.exception.code, 2)
                report.assert_not_called()

    def test_cli_maps_selected_tier_into_session(self):
        from voicelib import interactive
        tool = load_tool()
        with patch.object(tiers, "report", return_value=self.report()), \
                patch.object(tool.sys.stdin, "isatty", return_value=True), \
                patch.object(tool.sys.stdout, "isatty", return_value=True), \
                patch.object(interactive, "run", return_value=0) as run:
            self.assertEqual(tool.main(["--interactive", "--tier", "qwen-gpu", "--voice", "Ryan"]), 0)
            args = run.call_args.args[0]
            self.assertEqual((args.device, args.attention, args.voice), ("cuda:0", "flash_attention_2", "Ryan"))

    def test_cli_installs_piper_only_after_fit_and_rechecks(self):
        from voicelib import interactive, qwen_setup
        tool = load_tool()
        before_availability = self.availability()
        before_availability["neural"] = (False, True, "model missing")
        before = self.report(before_availability)
        after = self.report()
        with patch.object(tiers, "report", side_effect=[before, after]), \
                patch.object(qwen_setup, "install", return_value="/content/model") as first_use, \
                patch.object(tool, "_install_model", return_value="installed") as provider_install, \
                patch.object(tool.sys.stdin, "isatty", return_value=True), \
                patch.object(tool.sys.stdout, "isatty", return_value=True), \
                patch.object(interactive, "run", return_value=0) as session:
            self.assertEqual(tool.main(["--interactive", "--tier", "neural"]), 0)
        first_use.assert_called_once_with(qwen_setup.PIPER_ID)
        provider_install.assert_called_once_with(qwen_setup.PIPER_ID)
        session.assert_called_once()

    def test_cli_installs_pocket_only_after_fit_and_rechecks(self):
        from voicelib import interactive, qwen_setup
        tool = load_tool()
        before_availability = self.availability()
        before_availability["pocket-cpu"] = (False, True, "weights missing")
        before = self.report(before_availability)
        after = self.report()
        with patch.object(tiers, "report", side_effect=[before, after]), \
                patch.object(qwen_setup, "install", return_value="/content/model") as first_use, \
                patch.object(tool.sys.stdin, "isatty", return_value=True), \
                patch.object(tool.sys.stdout, "isatty", return_value=True), \
                patch.object(interactive, "run", return_value=0) as session:
            self.assertEqual(tool.main(["--interactive", "--tier", "pocket-cpu"]), 0)
        first_use.assert_called_once_with(qwen_setup.POCKET_ID)
        self.assertEqual(session.call_args.args[0].voice, "Alba")

    def test_pocket_tier_decline_does_not_start_session(self):
        from voicelib import interactive, qwen_setup
        tool = load_tool()
        available = self.availability()
        available["pocket-cpu"] = (False, True, "weights missing")
        with patch.object(tiers, "report", return_value=self.report(available)), \
                patch.object(qwen_setup, "install", side_effect=qwen_setup.Declined("Quit")), \
                patch.object(tool.sys.stdin, "isatty", return_value=True), \
                patch.object(tool.sys.stdout, "isatty", return_value=True), \
                patch.object(interactive, "run") as session:
            self.assertEqual(tool.main(["--interactive", "--tier", "pocket-cpu"]), 0)
        session.assert_not_called()

    def test_cli_installs_qwen_weights_only_after_fit_and_rechecks(self):
        from voicelib import interactive, qwen_setup
        tool = load_tool()
        before_availability = self.availability()
        before_availability["qwen-cpu"] = (False, True, "weights missing")
        before = self.report(before_availability)
        after = self.report()
        with patch.object(tiers, "report", side_effect=[before, after]), \
                patch.object(qwen_setup, "install", return_value="/content/model") as first_use, \
                patch.object(tool, "_install_model") as provider_install, \
                patch.object(tool.sys.stdin, "isatty", return_value=True), \
                patch.object(tool.sys.stdout, "isatty", return_value=True), \
                patch.object(interactive, "run", return_value=0) as session:
            self.assertEqual(tool.main(["--interactive", "--tier", "qwen-cpu"]), 0)
        first_use.assert_called_once_with(tiers.QWEN_ID)
        provider_install.assert_not_called()
        self.assertEqual(session.call_args.args[0].device, "cpu")

    def test_cli_installs_base_weights_after_fit_and_selects_synthetic_reference(self):
        from voicelib import interactive, qwen_setup
        tool = load_tool()
        before_availability = self.availability()
        before_availability["qwen-base-gpu"] = (False, True, "weights missing")
        before = self.report(before_availability)
        after = self.report()
        with patch.object(tiers, "report", side_effect=[before, after]), \
                patch.object(qwen_setup, "install", return_value="/content/model") as first_use, \
                patch.object(tool.sys.stdin, "isatty", return_value=True), \
                patch.object(tool.sys.stdout, "isatty", return_value=True), \
                patch.object(interactive, "run", return_value=0) as session:
            self.assertEqual(tool.main(["--interactive", "--tier", "qwen-base-gpu"]), 0)
        first_use.assert_called_once_with(tiers.QWEN_BASE_ID)
        args = session.call_args.args[0]
        self.assertTrue(args.synthetic_reference)
        self.assertEqual(args.qwen_model_dir, str(tiers.qwen_directory(tiers.QWEN_BASE_ID)))

    def test_insufficient_memory_refuses_before_piper_install(self):
        from voicelib import qwen_setup
        tool = load_tool()
        available = self.availability()
        available["neural"] = (False, True, "model missing")
        result = self.report(available)
        next(row for row in result["candidates"] if row["tier"] == "neural")["verdict"] = "does-not-fit"
        with patch.object(tiers, "report", return_value=result), \
                patch.object(qwen_setup, "install") as first_use, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                tool.main(["--interactive", "--tier", "neural"])
        self.assertEqual(caught.exception.code, 1)
        first_use.assert_not_called()


if __name__ == "__main__":
    unittest.main()
