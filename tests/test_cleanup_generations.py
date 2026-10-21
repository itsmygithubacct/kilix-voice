"""Stale cancellation must not resurrect a successfully disposed owner."""
import threading
import unittest
from unittest import mock

from tests import test_cleanup_recovery as recovery
from tests.test_terminal_outcomes import voiced


class Generations(unittest.TestCase):
    def daemon(self):
        daemon, _, _ = recovery.Boundaries.daemon(self)
        daemon._dictation = daemon._player = daemon._socket = None
        daemon._socket_ino = None
        daemon._wake_w = -1
        daemon._speech_error = ''
        daemon._speech_error_serial = 0
        return daemon

    def test_speech_finishes_between_cancel_registration_and_preparation(self):
        daemon = self.daemon()
        player = mock.Mock(error=None)
        engine = mock.Mock()
        turn = voiced._SpeechTurn('speech-race', ['fixture'], engine)
        daemon._speech = turn
        daemon._get_player = lambda: player
        entered, registered, finished = (threading.Event() for _ in range(3))
        errors = []
        register = daemon._register_cleanup

        def registering(*args, **kwargs):
            result = register(*args, **kwargs)
            if threading.current_thread().name == 'fixture-cancel' and args[1] == 'synthesiser':
                registered.set()
                if not finished.wait(5):
                    raise AssertionError('speech finalization did not finish')
            return result

        def synth(*_):
            entered.set()
            if not registered.wait(5):
                raise AssertionError('cancel did not register')
            return None

        def speech():
            try:
                daemon._run_speech(turn)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        def cancel():
            try:
                daemon._cancel_speech()
            except BaseException as error:
                errors.append(error)

        daemon._register_cleanup = registering
        daemon._synth = synth
        worker = threading.Thread(target=speech, daemon=True)
        stopper = threading.Thread(target=cancel, name='fixture-cancel', daemon=True)
        try:
            worker.start()
            self.assertTrue(entered.wait(5))
            stopper.start()
            worker.join(6)
            stopper.join(6)
        finally:
            registered.set()
            finished.set()
            worker.join(6)
            if stopper.ident is not None:
                stopper.join(6)
        self.assertFalse(worker.is_alive())
        self.assertFalse(stopper.is_alive())
        self.assertEqual(errors, [], 'concurrent cancel corrupted the registry')
        self.assertEqual(daemon._retained_dictation_cleanup(), [])
        daemon._retry_dictation_cleanup(0)
        daemon._shutdown()
        self.assertFalse(daemon._failed_dictation_cleanup)
        self.assertFalse(daemon._dictation_cleanup_state)
        self.assertFalse(daemon._cleanup_owners)
        engine.close.assert_called_once()
        self.assertEqual(turn.outcome.outcome, 'cancelled')

    def test_stale_preparation_and_attempt_leave_successor_generation_untouched(self):
        daemon = self.daemon()
        old, new = mock.Mock(), mock.Mock()
        key, state = daemon._register_cleanup('playback', 'player', old)
        daemon._prepare_cleanup(key, old, state)
        self.assertIsNone(daemon._attempt_dictation_cleanup(key, old, state))
        _, successor = daemon._register_cleanup('playback', 'player', new)
        daemon._prepare_cleanup(key, new, successor)
        daemon._prepare_cleanup(key, old, state)
        self.assertIs(daemon._failed_dictation_cleanup[key], new)
        self.assertIsNone(daemon._attempt_dictation_cleanup(key, old, state))
        self.assertIs(daemon._dictation_cleanup_state[key], successor)
        self.assertIs(daemon._cleanup_owners[key], new)
        self.assertIs(daemon._failed_dictation_cleanup[key], new)
        old.close.assert_called_once()
        daemon._retry_dictation_cleanup(2)
        new.close.assert_called_once()
        self.assertFalse(daemon._cleanup_owners)


if __name__ == '__main__':
    unittest.main()
