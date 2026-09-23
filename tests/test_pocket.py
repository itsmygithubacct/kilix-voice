"""Pocket audition contract without fetching or loading model weights."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from voicelib import interactive, pocket, qwen_setup, tts
from tests.test_interactive import options
from tests.test_tts_tool import load_tool


class PocketTests(unittest.TestCase):
    def test_pin_verification_and_tamper(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "embeddings").mkdir()
            data = {"model.safetensors": b"model", "tokenizer.model": b"tokenizer",
                    "embeddings/alba.safetensors": b"alba"}
            pins = {name: (len(value), hashlib.sha256(value).hexdigest())
                    for name, value in data.items()}
            for name, value in data.items():
                (root / name).write_bytes(value)
            with mock.patch.object(pocket, "PINNED", pins):
                self.assertEqual(set(pocket.verify(root)), set(pins))
                (root / "embeddings/alba.safetensors").write_bytes(b"evil")
                with self.assertRaisesRegex(tts.TtsError, "checksum mismatch"):
                    pocket.verify(root)

    def test_offline_session_dispatch_and_voice(self):
        engine = mock.Mock(name="pocket-engine")
        engine.name, engine.voice = "Pocket", "Alba"
        player = mock.Mock(error=None)
        messages = []
        factory = mock.Mock(return_value=engine)
        self.assertEqual(interactive.run(options(pocket_model_dir="/fixture", voice="Alba"),
                                         read=lambda _: "/quit", emit=messages.append,
                                         engine_factory=factory, player_factory=lambda _: player), 0)
        factory.assert_called_once_with("/fixture", voice="Alba", threads=4, seed=0)
        self.assertTrue(any("Pocket" in line for line in messages))

    def test_rejects_other_voice_before_weight_load(self):
        with self.assertRaisesRegex(tts.TtsError, "only the licensed Alba"):
            pocket.ResidentPocket("/missing", voice="other")

    def test_interactive_recovers_from_pocket_generation_failure(self):
        engine = mock.Mock(name="pocket-engine")
        engine.name, engine.voice = "Pocket", "Alba"
        engine.synth.side_effect = [RuntimeError("worker failed"), (b"\x00\x00" * 240, 24000)]
        player = mock.Mock(error=None)
        player.wait.return_value = True
        messages = []
        self.assertEqual(interactive.run(options(pocket_model_dir="/fixture"),
                                         read=mock.Mock(side_effect=["first", "second", "/quit"]),
                                         emit=messages.append,
                                         engine_factory=mock.Mock(return_value=engine),
                                         player_factory=lambda _: player), 0)
        self.assertIn("Error: worker failed", messages)
        engine.synth.assert_has_calls([mock.call("first"), mock.call("second")])
        player.play.assert_called_once()
        engine.close.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("pocket_tts"), "Pocket runtime not installed")
    def test_local_config_and_python_api(self):
        import yaml
        from pocket_tts import TTSModel
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "embeddings").mkdir()
            for name in pocket.PINNED:
                (root / name).write_bytes(b"fixture")
            model = mock.Mock(sample_rate=24000)
            seen = {}
            def load(*, config):
                seen.update(yaml.safe_load(Path(config).read_text()))
                return model
            with mock.patch.object(pocket, "verify", return_value={
                    name: root / name for name in pocket.PINNED}), \
                    mock.patch.object(TTSModel, "load_model", side_effect=load):
                engine = pocket.ResidentPocket(root)
            self.assertEqual(seen["weights_path"], str(root / "model.safetensors"))
            self.assertEqual(seen["flow_lm"]["lookup_table"]["tokenizer_path"],
                             str(root / "tokenizer.model"))
            model.get_state_for_audio_prompt.assert_called_once_with(
                root / "embeddings/alba.safetensors")
            self.assertEqual(engine.voices(), ["Alba"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch runtime not installed")
    def test_generation_state_can_be_updated_by_pocket_worker_thread(self):
        import numpy as np
        import torch

        def generate_audio(_state, _text):
            # Pocket creates state on the caller thread, then updates it from
            # an autoregressive worker. Inference-mode tensors reject that.
            state = torch.zeros(1)
            errors = []

            def update_state():
                try:
                    state.add_(1)
                except RuntimeError as error:
                    errors.append(error)

            worker = threading.Thread(target=update_state)
            worker.start()
            worker.join()
            if errors:
                raise errors[0]
            return state.repeat(240)

        engine = object.__new__(pocket.ResidentPocket)
        engine.np, engine.torch, engine.seed = np, torch, 0
        engine.model = mock.Mock(sample_rate=24000, generate_audio=generate_audio)
        engine.state = object()
        pcm, rate = engine.synth("hi")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(pcm), 480)

    def test_cli_requires_interactive_and_exclusive_engine(self):
        tool = load_tool()
        for args in (["--pocket-model-dir", "/fixture"],
                     ["--interactive", "--pocket-model-dir", "/fixture",
                      "--qwen-model-dir", "/other"],
                     ["--interactive", "--pocket-model-dir", "/fixture",
                      "--device", "cuda:0"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                tool.main(args)

    def test_first_use_download_starts_pocket_session_after_agreement(self):
        tool = load_tool()
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(qwen_setup, "install", return_value="/admitted/model") as install, \
                mock.patch.object(interactive, "run", return_value=0) as session:
            self.assertEqual(tool.main(["--interactive", "--download-pocket"]), 0)
        install.assert_called_once_with(qwen_setup.POCKET_ID)
        self.assertEqual(session.call_args.args[0].pocket_model_dir, "/admitted/model")

    def test_declining_pocket_terms_never_starts_session(self):
        tool = load_tool()
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(qwen_setup, "install", side_effect=qwen_setup.Declined("Quit")), \
                mock.patch.object(interactive, "run") as session:
            self.assertEqual(tool.main(["--interactive", "--download-pocket"]), 0)
        session.assert_not_called()

    def test_real_content_first_use_decline_and_accept_in_isolated_store(self):
        try:
            from kilix_content import default_catalog
            from kilix_content.install import Installer
            from kilix_license import ReceiptStore
            default_catalog().require_asset(qwen_setup.POCKET_ID)
        except (ImportError, KeyError, ValueError):
            self.skipTest("the Pocket CPU Content candidate is not installed")
        for key, accepts in (("q", False), (" ", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as scratch:
                store = ReceiptStore(Path(scratch) / "receipts")
                output = io.StringIO()
                output.isatty = lambda: True
                def fetch(*_args, **_kwargs):
                    self.assertEqual(len(list(store.root.glob("*.json"))), 1)
                    return ()
                with mock.patch.object(qwen_setup.sys.stdin, "isatty", return_value=True), \
                        mock.patch("voicelib.paths.gpu_terminal_home", return_value=scratch), \
                        mock.patch.object(ReceiptStore, "shared", return_value=store), \
                        mock.patch.object(Installer, "ensure_upstream_asset", side_effect=fetch) as acquire:
                    if accepts:
                        self.assertTrue(qwen_setup.install(qwen_setup.POCKET_ID,
                            read_key=lambda _: key, output=output).endswith("/model"))
                        acquire.assert_called_once()
                    else:
                        with self.assertRaises(qwen_setup.Declined):
                            qwen_setup.install(qwen_setup.POCKET_ID,
                                read_key=lambda _: key, output=output)
                        acquire.assert_not_called()
                        self.assertFalse(list(store.root.glob("*.json")))
                self.assertIn("Alba MacKenna", output.getvalue())
                self.assertIn("binding:pocket-prohibited-use", output.getvalue())


if __name__ == "__main__":
    unittest.main()
