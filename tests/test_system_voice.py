"""Startup speech lifecycle without downloading models or playing sound."""
import contextlib
import io
import unittest
from unittest import mock
from voicelib import system_voice, licensing
from tests.test_tts_tool import load_tool


class StartupVoiceTests(unittest.TestCase):
    def run_voice(self, greeting='hello', loaded=True):
        engine = mock.Mock()
        engine.synth.return_value = (b'\0\0', 22050)
        player = mock.Mock(error=None)
        wait = mock.Mock(side_effect=[False, True])
        with mock.patch.object(licensing, 'require_covering_receipt') as receipt, contextlib.redirect_stdout(io.StringIO()):
            system_voice.run(greeting, wait=wait, loaded=lambda: loaded,
                engine_factory=lambda **kw: engine, player_factory=lambda cfg: player)
        receipt.assert_called_once()
        player.close.assert_called_once()
        return engine, player

    def test_greet_once_and_keep_loaded_provider_alive(self):
        engine, player = self.run_voice()
        engine.synth.assert_called_once_with('hello')
        player.play.assert_called_once_with(b'\0\0', 22050)

    def test_empty_greeting_preloads_without_playback(self):
        engine, player = self.run_voice('')
        engine.synth.assert_called_once_with('hello')
        player.play.assert_not_called()

    def test_provider_restart_recovers_without_repeating_greeting(self):
        engine, player = self.run_voice('Welcome back', loaded=False)
        self.assertEqual(engine.synth.call_args_list, [mock.call('Welcome back'), mock.call('hello')])
        player.play.assert_called_once()

    def test_refusal_precedes_engine_creation(self):
        factory=mock.Mock()
        with mock.patch.object(licensing, 'require_covering_receipt', side_effect=RuntimeError('missing receipt')):
            with self.assertRaisesRegex(RuntimeError, 'missing receipt'):
                system_voice.run(engine_factory=factory)
        factory.assert_not_called()

    def test_invalid_greeting_precedes_receipt_and_engine(self):
        for text in ('x'*1025, 'a\0b'):
            with self.assertRaises(ValueError):system_voice.validate_greeting(text)

    def test_cli_rejects_unrelated_options(self):
        tool=load_tool()
        for argv in (['--system-voice','--tier','neural'], ['--prepare-system-voice','--speak','hello'], ['--system-voice','--prepare-system-voice']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                tool.main(argv)
            self.assertEqual(error.exception.code,2)

    def test_cli_keeps_greeting_literal(self):
        tool=load_tool()
        with mock.patch.object(system_voice,'run',return_value=0) as run:
            self.assertEqual(tool.main(['--system-voice','--speak','hello; $(false)']),0)
        run.assert_called_once_with('hello; $(false)')

    def test_prepare_uses_visible_notice_before_install(self):
        tool=load_tool()
        from voicelib import qwen_setup
        events=[]
        with mock.patch.object(tool.sys.stdin,'isatty',return_value=True), mock.patch.object(tool.sys.stdout,'isatty',return_value=True), mock.patch.object(qwen_setup,'install',side_effect=lambda model: events.append('notice')), mock.patch.object(tool,'_install_model',side_effect=lambda model: events.append('install')), contextlib.redirect_stdout(io.StringIO()) as output:
            # redirect_stdout replaces the patched TTY: explicitly identify the
            # test stream as a terminal while keeping all installer calls fake.
            output.isatty=lambda:True
            self.assertEqual(tool.main(['--prepare-system-voice']),0)
        self.assertEqual(events,['notice','install'])

    def test_prepare_decline_does_not_install_or_claim_success(self):
        tool=load_tool()
        from voicelib import qwen_setup
        with mock.patch.object(tool.sys.stdin,'isatty',return_value=True), mock.patch.object(tool.sys.stdout,'isatty',return_value=True), mock.patch.object(qwen_setup,'install',side_effect=qwen_setup.Declined('declined')), mock.patch.object(tool,'_install_model') as install, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            tool.main(['--prepare-system-voice'])
        self.assertEqual(result.exception.code,3)
        install.assert_not_called()
