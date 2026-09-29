"""Shutdown speech and logind lifetime without shutting down the test machine."""
import contextlib
import io
import os
import unittest
from unittest import mock
from voicelib import shutdown, system_voice


class ShutdownTests(unittest.TestCase):
    def test_delay_released_after_signal_and_reacquired_after_cancel(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, write_fd)
        manager = mock.Mock()
        manager.Inhibit.return_value.take.side_effect = [read_fd, os.dup(read_fd)]
        bus, context = mock.Mock(), mock.Mock()
        context.pending.return_value = False
        watch = shutdown.ShutdownWatch(bus, manager, context)
        self.assertFalse(watch.poll())
        watch.prepare(True)
        self.assertTrue(watch.poll())
        self.assertFalse(watch.poll())
        watch.release()
        with self.assertRaises(OSError):os.fstat(read_fd)
        watch.prepare(False)
        self.assertIsNotNone(watch.fd)
        watch.close()
        self.assertEqual(manager.Inhibit.call_count, 2)
        self.assertEqual(manager.Inhibit.call_args.args[-1], 'delay')
        self.assertEqual(bus.add_signal_receiver.call_args.kwargs['bus_name'], 'org.freedesktop.login1')
        bus.add_signal_receiver.return_value.remove.assert_called_once()
        bus.close.assert_called_once()

    def test_denied_inhibitor_closes_bus(self):
        manager, bus = mock.Mock(), mock.Mock()
        manager.Inhibit.side_effect = RuntimeError('denied')
        with self.assertRaisesRegex(RuntimeError, 'denied'):
            shutdown.ShutdownWatch(bus, manager, mock.Mock())
        bus.close.assert_called_once()
        bus.add_signal_receiver.return_value.remove.assert_called_once()

    def run_voice(self, events, watch=None, player_error=None):
        engine = mock.Mock()
        engine.synth.side_effect = lambda text: (text.encode(), 22050)
        player = mock.Mock(error=player_error)
        with mock.patch.object(system_voice.licensing, 'require_covering_receipt'), contextlib.redirect_stdout(io.StringIO()):
            system_voice.run('', wait=mock.Mock(side_effect=events),
                engine_factory=lambda **kw: engine, player_factory=lambda cfg: player,
                monitor_factory=None, shutdown_factory=lambda: watch, clock=lambda: 0)
        return engine, player

    def test_menu_and_logind_signal_play_once_from_cache(self):
        watch = mock.Mock()
        watch.poll.side_effect = [False, True]
        engine, player = self.run_voice(['goodbye', False, True], watch)
        player.play.assert_called_once_with(b'Goodbye', 22050)
        self.assertEqual(engine.synth.call_args_list, [mock.call('hello'), mock.call('Goodbye')])
        watch.release.assert_called_once()
        watch.close.assert_called_once()

    def test_external_shutdown_plays_then_releases_delay(self):
        watch = mock.Mock()
        watch.poll.return_value = True
        order = []
        player = mock.Mock(error=None)
        player.wait.side_effect = lambda _: order.append('finished') or True
        watch.release.side_effect = lambda: order.append('released')
        with mock.patch.object(system_voice.audio, 'Player', return_value=player):
            # The run default is bound at definition, so inject explicitly.
            with mock.patch.object(system_voice.licensing, 'require_covering_receipt'), contextlib.redirect_stdout(io.StringIO()):
                system_voice.run('', wait=mock.Mock(side_effect=[False, True]),
                    engine_factory=lambda **kw: mock.Mock(synth=lambda text: (text.encode(), 22050)),
                    player_factory=lambda cfg: player, monitor_factory=None,
                    shutdown_factory=lambda: watch, clock=lambda: 0)
        self.assertEqual(order, ['finished', 'released'])
        player.play.assert_called_once_with(b'Goodbye', 22050)

    def test_owner_eof_does_not_say_goodbye(self):
        _, player = self.run_voice([True])
        player.play.assert_not_called()

    def test_audio_failure_releases_shutdown_delay(self):
        watch = mock.Mock()
        watch.poll.return_value = True
        with self.assertRaisesRegex(system_voice.tts.TtsError, 'unavailable'):
            self.run_voice([False], watch, player_error='unavailable')
        watch.release.assert_called_once()
        watch.close.assert_called_once()

    def test_pipe_command_is_distinct_from_eof(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd) as stream, mock.patch.object(system_voice.sys, 'stdin', stream):
            os.write(write_fd, b'g')
            self.assertEqual(system_voice.wait_for_owner(.01), 'goodbye')
            self.assertFalse(system_voice.wait_for_owner(.01))
            os.close(write_fd)
            self.assertIs(system_voice.wait_for_owner(.01), True)


if __name__ == '__main__':unittest.main()
