import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from voicelib import document_reader as D, settings, qwen_setup
from tests.test_tts_tool import load_tool


class DocumentTests(unittest.TestCase):
    def test_long_document_is_not_truncated_to_pane_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'long.txt'
            text = 'A sentence worth reading. ' * 12000 + 'The final sentence.'
            path.write_text(text)
            chunks = D.load_document(path)
            self.assertEqual(chunks[-1], 'The final sentence.')
            self.assertGreater(len(chunks), 12000)
            self.assertLessEqual(max(map(len, chunks)), 600)

    def test_markdown_prose_and_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'book.md'
            path.write_text('\ufeff---\ntitle: Hidden\n---\n# Hello\nRead **this** [link](https://example.test).\n```sh\necho secret-code\n```\nGoodbye.')
            text = ' '.join(D.load_document(path))
            self.assertEqual(text, 'Hello Read this link. Goodbye.')

    def test_reject_empty_binary_unsupported_large_and_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'empty.txt'
            path.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'no readable'):
                D.load_document(path)
            path.write_bytes(b'\xff')
            with self.assertRaisesRegex(ValueError, 'UTF-8'):
                D.load_document(path)
            path.write_bytes(b'x'*16)
            with mock.patch.object(D, 'MAX_TEXT', 8), self.assertRaisesRegex(ValueError, 'limit'):
                D.load_document(path)
            path.unlink(); os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, 'regular'):
                D.load_document(path)
            with self.assertRaisesRegex(ValueError, 'Choose'):
                D.load_document(Path(directory)/'x.exe')

    def test_pdf_missing_dependency(self):
        with mock.patch.object(D.subprocess, 'Popen', side_effect=FileNotFoundError), self.assertRaisesRegex(ValueError, 'poppler-utils'):
            D.pdf_text(Path('/tmp/book.pdf'))

    def test_pdf_output_is_bounded_and_failed_extractor_is_reaped(self):
        original = subprocess.Popen
        processes = []
        def spawn(*args, **kwargs):
            p = original([sys.executable, '-c', 'import sys; sys.stdout.write("x"*100)'], **kwargs)
            processes.append(p)
            return p
        with mock.patch.object(D.subprocess, 'Popen', side_effect=spawn), mock.patch.object(D, 'MAX_TEXT', 8), self.assertRaisesRegex(ValueError, 'limit'):
            D.pdf_text(Path('/tmp/book.pdf'))
        self.assertIsNotNone(processes[0].poll())

    def test_reader_process_is_silent_until_play_and_exits_on_owner_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'book.txt'; path.write_text('Hello.')
            result = subprocess.run([sys.executable, 'kilix-tts', '--reader-session', str(path)], input=b'', capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(b'"state": "ready"', result.stdout)
            self.assertNotIn(b'"state": "playing"', result.stdout)


class PlaybackTests(unittest.TestCase):
    def make_reader(self, synth=None):
        player = mock.Mock(playing=False, error=None)
        engine = mock.Mock()
        engine.synth.side_effect = synth or (lambda text, **kw: (text.encode(), 22050))
        events = []
        return D.Reader(['First.', 'Second.'], events.append, player=player, factory=lambda: engine), player, engine, events

    def finish_synthesis(self, reader):
        reader.tick()
        self.assertTrue(reader.job.done.wait(2))
        reader.player.playing = True
        reader.tick()

    def test_pause_cancels_stale_audio_and_resume_repeats_passage(self):
        gate = threading.Event()
        reader, player, engine, events = self.make_reader(lambda text, **kw: (gate.wait(2) and b'pcm', 22050))
        reader.command('p'); reader.tick()
        reader.command('a')
        gate.set(); self.assertTrue(reader.job.done.wait(2)); reader.tick()
        player.play.assert_not_called()
        self.assertEqual(reader.state, 'paused')
        reader.command('p'); self.finish_synthesis(reader)
        self.assertEqual(engine.synth.call_args_list[0].args, engine.synth.call_args_list[1].args)
        reader.close(); player.close.assert_called_once()

    def test_navigation_stop_completion_and_error(self):
        reader, player, engine, events = self.make_reader()
        reader.command('n'); self.assertEqual(reader.index, 1)
        reader.command('b'); self.assertEqual(reader.index, 0)
        reader.command('p'); self.finish_synthesis(reader)
        player.playing = False; reader.tick()
        self.assertEqual(reader.index, 1)
        self.assertTrue(reader.job.done.wait(2)); player.playing=True; reader.tick()
        player.playing=False; reader.tick()
        self.assertEqual(reader.state, 'finished'); self.assertFalse(reader.active)
        reader.command('s'); self.assertEqual(reader.index, 0)
        reader.close()
        reader, player, engine, events = self.make_reader()
        engine.synth.side_effect=RuntimeError('missing voice')
        reader.command('p'); reader.tick(); self.assertTrue(reader.job.done.wait(2)); reader.tick()
        self.assertEqual(events[-1]['error'], 'missing voice')
        reader.close()

    def test_off_and_piper_receipt_gate(self):
        with mock.patch.object(settings, 'tts_engine', return_value='off'), self.assertRaisesRegex(ValueError, 'off'):
            D.reader_engine()
        with mock.patch.object(settings, 'tts_engine', return_value=D.models.TTS_ENGINE_PIPER), mock.patch.object(D.licensing, 'require_covering_receipt', side_effect=RuntimeError('receipt missing')), mock.patch.object(D.tts, 'make_tts') as factory, self.assertRaisesRegex(RuntimeError, 'receipt'):
            D.reader_engine()
        factory.assert_not_called()


class OptInTests(unittest.TestCase):
    def test_preference_changes_only_after_successful_explicit_setup(self):
        tool = load_tool()
        with mock.patch.object(sys.stdin, 'isatty', return_value=True), mock.patch.object(sys.stdout, 'isatty', return_value=True), mock.patch.object(qwen_setup, 'install') as install, mock.patch.object(tool, '_install_model') as model, mock.patch.object(settings, 'update') as update, contextlib.redirect_stdout(io.StringIO()):
            # redirect_stdout needs its own tty marker for the setup gate.
            with mock.patch.object(sys.stdout, 'isatty', return_value=True):
                self.assertEqual(tool.main(['--enable-kristin']), 0)
                update.assert_called_once_with({settings.KEY_TTS_ENGINE: D.models.TTS_ENGINE_PIPER, settings.KEY_TTS_VOICE:'en_US-kristin-medium'})
                update.reset_mock(); install.side_effect=qwen_setup.Declined('No')
                with self.assertRaises(SystemExit) as raised:
                    tool.main(['--enable-kristin'])
                self.assertEqual(raised.exception.code, 3)
                update.assert_not_called()

    def test_optin_requires_terminal_and_cannot_mix_with_speech(self):
        tool = load_tool()
        with mock.patch.object(sys.stdin, 'isatty', return_value=False), mock.patch.object(qwen_setup, 'install') as install, contextlib.redirect_stderr(io.StringIO()):
            for arguments in [['--enable-kristin'], ['--enable-kristin','--speak','hello'], ['--reader-session','book.txt','--model','espeak']]:
                with self.assertRaises(SystemExit): tool.main(arguments)
            install.assert_not_called()
