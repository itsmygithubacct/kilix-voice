"""Offline session coverage: no weights, audio servers, or live stores."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from voicelib import interactive, qwen_setup
from tests.test_tts_tool import load_tool


def options(**overrides):
    values = dict(qwen_model_dir=None, model="espeak", voice=None, rate=None,
                  language=None, device="cpu", threads=4, seed=0,
                  synthetic_reference=False, speak=None, attention="sdpa")
    return SimpleNamespace(**(values | overrides))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.engine = mock.Mock(name="engine")
        self.engine.name, self.engine.voice = "fixture", "test"
        self.engine.synth.return_value = (b"\x01\x00" * 240, 24000)
        self.player = mock.Mock(error=None)
        self.player.wait.return_value = True
        self.messages = []

    def run_session(self, lines, **overrides):
        return interactive.run(options(**overrides), read=mock.Mock(side_effect=lines),
                               emit=self.messages.append,
                               engine_factory=mock.Mock(return_value=self.engine),
                               player_factory=mock.Mock(return_value=self.player))

    def test_warm_session_and_replay(self):
        self.assertEqual(self.run_session(["Hello", "/repeat", "World", "/quit"]), 0)
        self.assertEqual(self.engine.synth.call_count, 2)
        self.assertEqual(self.player.play.call_count, 3)
        self.engine.close.assert_called_once()
        self.player.close.assert_called_once()

    def test_initial_prompt_and_eof(self):
        self.run_session([EOFError()], speak="Welcome")
        self.engine.synth.assert_called_once_with("Welcome")

    def test_interrupt_does_not_end_session(self):
        self.run_session([KeyboardInterrupt(), "Hello", "/quit"])
        self.player.stop.assert_called_once()
        self.engine.synth.assert_called_once_with("Hello")

    def test_synthesis_interrupt_keeps_last_good_clip(self):
        good = self.engine.synth.return_value
        self.engine.synth.side_effect = [good, KeyboardInterrupt()]
        self.run_session(["One", "Two", "/repeat", "/quit"])
        self.assertEqual(self.player.play.call_count, 2)

    def test_missing_clip_and_unknown_commands_are_recoverable(self):
        self.run_session(["/repeat", "/save missing.wav", "/invalid", "/quit"])
        self.assertEqual(sum(m.startswith("Error:") for m in self.messages), 3)
        self.engine.synth.assert_not_called()

    def test_text_limit_counts_utf8_bytes(self):
        self.run_session(["é" * 8193, "/quit"])
        self.engine.synth.assert_not_called()

    def test_qwen_selectors(self):
        self.run_session(["/voice Aiden", "/language Japanese", "/quit"],
                         qwen_model_dir="/fixture")
        self.engine.set_voice.assert_called_once_with("Aiden")
        self.engine.set_language.assert_called_once_with("Japanese")

    def test_flash_attention_is_forwarded_to_resident_engine(self):
        factory = mock.Mock(return_value=self.engine)
        interactive.run(options(qwen_model_dir="/fixture", device="cuda:0",
                                attention="flash_attention_2"),
                        read=lambda _: "/quit", emit=self.messages.append,
                        engine_factory=factory, player_factory=lambda _: self.player)
        self.assertEqual(factory.call_args.kwargs["attention"], "flash_attention_2")
        self.assertEqual(factory.call_args.kwargs["device"], "cuda:0")

    def test_cpu_flash_attention_refused_before_model_or_imports(self):
        with self.assertRaisesRegex(interactive.tts.TtsError, "requires --device"):
            interactive.ResidentQwen("/missing", attention="flash_attention_2")

    def test_flash_attention_loads_cuda_bfloat16_without_fallback(self):
        torch = mock.Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_capability.return_value = (8, 6)
        upstream = mock.Mock()
        upstream.from_pretrained.return_value.get_supported_speakers.return_value = ["ryan"]
        upstream.from_pretrained.return_value.get_supported_languages.return_value = ["english"]
        modules = {"torch": torch, "numpy": mock.Mock(), "flash_attn": mock.Mock(),
                   "qwen_tts": SimpleNamespace(Qwen3TTSModel=upstream)}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("sys.modules", modules), \
                mock.patch.dict(interactive.os.environ):
            (Path(tmp) / "config.json").write_text('{"tts_model_type":"custom_voice"}')
            engine = interactive.ResidentQwen(tmp, device="cuda:0", attention="flash_attention_2")
            kwargs = upstream.from_pretrained.call_args.kwargs
            self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")
            self.assertEqual(kwargs["device_map"], "cuda:0")
            self.assertIs(kwargs["dtype"], torch.bfloat16)
            self.assertTrue(kwargs["local_files_only"])
            engine.close()
            upstream.reset_mock()
            with mock.patch.dict("sys.modules", {"flash_attn": None}):
                with self.assertRaisesRegex(interactive.tts.TtsError, "could not load"):
                    interactive.ResidentQwen(tmp, device="cuda:0", attention="flash_attention_2")
            upstream.from_pretrained.assert_not_called()
            torch.cuda.get_device_capability.return_value = (7, 5)
            with self.assertRaisesRegex(interactive.tts.TtsError, "Ampere"):
                interactive.ResidentQwen(tmp, device="cuda:0", attention="flash_attention_2")
            upstream.from_pretrained.assert_not_called()

    def test_save_exclusive_and_valid_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            interactive.save_clip(path, self.engine.synth.return_value)
            self.assertEqual(path.read_bytes()[:4], b"RIFF")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                interactive.save_clip(path, self.engine.synth.return_value)
            self.assertEqual(before, path.read_bytes())

    def test_save_symlink_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "link.wav"
            target = Path(tmp) / "target.wav"
            path.symlink_to(target)
            with self.assertRaises(FileExistsError):
                interactive.save_clip(path, self.engine.synth.return_value)
            self.assertFalse(target.exists())

    def test_qwen_bad_kind_refused_before_dependency_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text('{"tts_model_type":"voice_design"}')
            with self.assertRaisesRegex(interactive.tts.TtsError, "supports Qwen"):
                interactive.ResidentQwen(tmp)

    def test_base_requires_explicit_synthetic_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text('{"tts_model_type":"base"}')
            with self.assertRaisesRegex(interactive.tts.TtsError, "synthetic-reference"):
                interactive.ResidentQwen(tmp)


class SessionCliTests(unittest.TestCase):
    def test_download_continues_into_interactive_in_same_process(self):
        tool = load_tool()
        events = []
        def download(model):
            events.append("download")
            return "/installed/model"
        def session(args):
            events.append("session")
            self.assertEqual(args.qwen_model_dir, "/installed/model")
            self.assertEqual(args.voice, "Ryan")
            return 0
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(qwen_setup, "install", side_effect=download), \
                mock.patch.object(interactive, "run", side_effect=session), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            # redirect_stdout replaces the stream, so supply its tty property too.
            output.isatty = lambda: True
            self.assertEqual(tool.main(["--interactive", "--download-qwen",
                                        "qwen3-tts-0.6b-customvoice", "--voice", "Ryan"]), 0)
        self.assertEqual(events, ["download", "session"])

    def test_quit_during_setup_never_starts_session(self):
        tool = load_tool()
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(qwen_setup, "install", side_effect=qwen_setup.Declined("Quit")), \
                mock.patch.object(interactive, "run") as session:
            self.assertEqual(tool.main(["--interactive", "--download-qwen",
                                        "qwen3-tts-0.6b-customvoice"]), 0)
        session.assert_not_called()

    def test_invalid_interactive_download_fetches_nothing(self):
        tool = load_tool()
        for extra in (["--download-qwen", "all"],
                      ["--download-qwen", "qwen3-tts-1.7b-voicedesign"],
                      ["--download-qwen", "qwen3-tts-0.6b-base"],
                      ["--download-qwen", "qwen3-tts-0.6b-customvoice", "--threads", "0"],
                      ["--download-qwen", "qwen3-tts-0.6b-customvoice", "--attention", "flash_attention_2"],
                      ["--download-qwen", "qwen3-tts-0.6b-customvoice", "--qwen-model-dir", "/tmp/model"]):
            with self.subTest(extra=extra), mock.patch.object(qwen_setup, "install") as acquire, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main(["--interactive", *extra])
            acquire.assert_not_called()

    def test_interrupt_reaps_conventional_synthesizers(self):
        for model in ("espeak", "piper-en-us-kristin-medium"):
            with self.subTest(model=model):
                process = mock.Mock()
                process.communicate.side_effect = [KeyboardInterrupt(), (b"", b"")]
                with mock.patch("voicelib.tts.util.which", return_value="/fixture/synth"), \
                        mock.patch("voicelib.tts.subprocess.Popen", return_value=process):
                    engine = interactive.tts.make_tts(model=model)
                    with self.assertRaises(KeyboardInterrupt):
                        engine.synth("Hello")
                    process.kill.assert_called_once()
                    self.assertEqual(process.communicate.call_count, 2)
                    engine.close()

    def test_interactive_dispatch(self):
        tool = load_tool()
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(interactive, "run", return_value=0) as run:
            self.assertEqual(tool.main(["--interactive", "--model", "espeak"]), 0)
        self.assertEqual(run.call_args.args[0].model, "espeak")

    def test_conflicts_refused(self):
        tool = load_tool()
        for args in (["--interactive", "--status"],
                     ["--qwen-model-dir", "/fixture"],
                     ["--interactive", "--output", "out.wav"],
                     ["--interactive", "--speak", "-"],
                     ["--interactive", "--qwen-model-dir", "/fixture", "--rate", "170"],
                     ["--interactive", "--qwen-model-dir", "/fixture", "--threads", "0"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    tool.main(args)
                self.assertEqual(error.exception.code, 2)

    def test_no_terminal_refused(self):
        tool = load_tool()
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=False), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(["--interactive"])
