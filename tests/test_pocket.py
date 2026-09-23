"""Pocket audition contract without fetching or loading model weights."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from voicelib import interactive, pocket, tts
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

    def test_cli_requires_interactive_and_exclusive_engine(self):
        tool = load_tool()
        for args in (["--pocket-model-dir", "/fixture"],
                     ["--interactive", "--pocket-model-dir", "/fixture",
                      "--qwen-model-dir", "/other"],
                     ["--interactive", "--pocket-model-dir", "/fixture",
                      "--device", "cuda:0"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                tool.main(args)


if __name__ == "__main__":
    unittest.main()
