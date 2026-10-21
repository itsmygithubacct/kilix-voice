"""Stale cancellation must not resurrect a successfully disposed owner."""
import json
import time
import threading
import unittest
from unittest import mock

from tests import test_cleanup_recovery as recovery
from tests.test_terminal_outcomes import voiced
from voicelib import audio, processes


class Generations(unittest.TestCase):

    def test_registration_keeps_each_resources_state_and_retry_ownership(self):
        daemon = self.daemon()
        old, new, old_grant, new_grant = (mock.Mock() for _ in range(4))
        key, state = daemon._register_cleanup('playback', 'player', old, old_grant)
        daemon._prepare_cleanup(key, old, state)
        state.closed = state.running = True
        next_key, successor = daemon._register_cleanup('playback', 'player', new, new_grant)
        self.assertIsNot(successor, state)
        self.assertFalse(successor.closed)
        self.assertFalse(successor.running)
        self.assertIs(successor.grant, new_grant)
        self.assertIs(state.grant, old_grant)
        self.assertIs(daemon._cleanup_owners[key], old)
        self.assertIs(daemon._cleanup_owners[next_key], new)
        self.assertIs(daemon._failed_dictation_cleanup[key], old)
        self.assertEqual(daemon._register_cleanup('playback', 'player', new),
                         (next_key, successor))
        self.assertEqual(daemon._register_cleanup('playback', 'player', old),
                         (key, state))
        state.running = False
        daemon._prepare_cleanup(next_key, new, successor)
        daemon._retry_dictation_cleanup(2)
        old.close.assert_not_called()
        new.close.assert_called_once_with()
        old_grant.reaped.assert_called_once_with()
        new_grant.reaped.assert_called_once_with()
        self.assertFalse(daemon._cleanup_owners)

    def test_first_retry_worker_can_enter_before_second_thread_start_returns(self):
        daemon = self.daemon()
        entered, release = threading.Event(), threading.Event()
        owners = [mock.Mock(), mock.Mock()]
        for i, owner in enumerate(owners):
            key, state = daemon._register_cleanup(str(i), 'provider', owner)
            daemon._prepare_cleanup(key, owner, state)
        def close_first():
            entered.set()
            release.wait(5)
        owners[0].close.side_effect = close_first
        start = threading.Thread.start
        observations = []
        workers = []
        def starting(worker):
            workers.append(worker)
            if len(workers) == 2:
                observations.append(entered.wait(2))
            return start(worker)
        try:
            with mock.patch.object(threading.Thread, 'start', starting):
                daemon._retry_dictation_cleanup(0)
            self.assertEqual(observations, [True],
                             'registry lock blocked the first cleanup during the second start')
        finally:
            release.set()
            for worker in workers:
                worker.join(5)
            daemon._retry_dictation_cleanup(2)

    def test_late_speech_finalization_cannot_share_successor_state_and_erase_shutdown_owner(self):
        d = self.daemon()
        old = mock.Mock(error=None)
        old._process = None
        new = audio.Player({'audio': {'play_cmd': ['/bin/cat']}})
        new.play(b'\x00\x00', 16000)
        self.assertTrue(new.wait(5))
        real_close = new.close
        new.close = mock.Mock(wraps=real_close)
        engine = mock.Mock()
        turn = voiced._SpeechTurn('old-turn', ['fixture'], engine)
        turn.deadline = time.monotonic() - 1
        d._speech = turn
        d._synth = lambda *_: None
        stale_ready, allow_stale, old_close_entered, release_old = (threading.Event() for _ in range(4))
        errors = []
        count = [0]
        main = threading.current_thread()

        def close_old():
            count[0] += 1
            if count[0] == 2:
                old_close_entered.set()
                if not release_old.wait(8):
                    raise AssertionError('old close not released')
        old.close.side_effect = close_old
        register = d._register_cleanup
        retry = d._retry_dictation_cleanup

        def registering(*args, **kwargs):
            if threading.current_thread().name == 'old-finalizer' and args[0:2] == ('playback', 'player') and (args[2] is old):
                stale_ready.set()
                if not allow_stale.wait(8):
                    raise AssertionError('stale finalization not released')
            return register(*args, **kwargs)

        def speech():
            try:
                d._run_speech(turn)
            except BaseException as e:
                errors.append((type(e).__name__, str(e)))
        d._register_cleanup = registering
        with mock.patch.object(voiced.audio, 'Player', return_value=old):
            self.assertIs(d._get_player(), old)
        worker = threading.Thread(target=speech, name='old-finalizer', daemon=True)
        worker.start()
        observed = {}
        try:
            self.assertTrue(stale_ready.wait(5))
            self.assertIsNone(d._speech)
            d._cancel_speech()
            self.assertIsNone(d._player)
            successor_turn = voiced._SpeechTurn('successor-turn', ['fixture'], mock.Mock())
            d._speech = successor_turn
            with mock.patch.object(voiced.audio, 'Player', return_value=new):
                d._run_speech(successor_turn)
            self.assertIsNone(d._speech)
            self.assertIs(d._player, new)
            successor = d._dictation_cleanup_state['playback', 'player']
            allow_stale.set()
            self.assertTrue(old_close_entered.wait(5))
            observed['stale_registration_reuses_successor_state'] = d._dictation_cleanup_state['playback', 'player'] is successor
            observed['registry_points_to_old_before_shutdown'] = d._cleanup_owners['playback', 'player'] is old
            released = [False]

            def shutdown_retry(timeout):
                result = retry(timeout)
                if threading.current_thread() is main and (not released[0]):
                    released[0] = True
                    release_old.set()
                    worker.join(5)
                    self.assertFalse(worker.is_alive())
                    observed['successor_state_marked_closed_by_old'] = successor.closed
                return result
            d._retry_dictation_cleanup = shutdown_retry
            with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', 0.05), mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', 0.2):
                d._shutdown()
            observed.update(errors=errors, old_close_calls=old.close.call_count, new_close_calls=new.close.call_count, owners=len(d._cleanup_owners or {}), retained=d._retained_dictation_cleanup(), player_cleared=d._player is None, new_cleanup_complete=new.cleanup_complete, new_worker_alive=new._worker is not None and new._worker.is_alive(), warnings=[c.args[0] for c in d._warn.call_args_list])
            print('SUCCESSOR_SHUTDOWN', json.dumps(observed), flush=True)
            self.assertEqual(errors, [])
            self.assertEqual(new.close.call_count, 1, 'shutdown discarded the successor without closing it')
        finally:
            allow_stale.set()
            release_old.set()
            worker.join(5)
            real_close()


    def test_completion_of_displaced_attempt_cannot_publish_or_remove_successor(self):
        for failing in (False, True):
            with self.subTest(failing=failing):
                d = self.daemon()
                old, new = (mock.Mock(), mock.Mock())
                key, state = d._register_cleanup('owned', 'provider', old)

                def replace():
                    with d._cleanup_lock:
                        d._cleanup_owners.pop(key)
                        d._dictation_cleanup_state.pop(key)
                    _, successor = d._register_cleanup('owned', 'provider', new)
                    d._prepare_cleanup(key, new, successor)
                    if failing:
                        raise OSError('old close failed after displacement')
                old.close.side_effect = replace
                d._attempt_dictation_cleanup(key, old, state)
                self.assertIs(d._cleanup_owners.get(key), new)
                self.assertIs(d._failed_dictation_cleanup.get(key), new)
                self.assertIsNot(d._dictation_cleanup_state.get(key), state)
                new.close.assert_not_called()

    def test_stale_attempt_does_not_acknowledge_old_grant_again(self):
        d = self.daemon()
        old, new, grant = (mock.Mock(), mock.Mock(), mock.Mock())
        key, state = d._register_cleanup('old', 'provider', old, grant)
        d._prepare_cleanup(key, old, state)
        d._attempt_dictation_cleanup(key, old, state)
        _, successor = d._register_cleanup('old', 'provider', new)
        d._prepare_cleanup(key, new, successor)
        d._attempt_dictation_cleanup(key, old, state)
        grant.reaped.assert_called_once_with()
        self.assertIs(d._cleanup_owners[key], new)

    def test_stale_initial_claim_does_not_mark_retired_state_running(self):
        d = self.daemon()
        old, new = (mock.Mock(), mock.Mock())
        prepare = d._prepare_cleanup
        seen = []

        def preparing(key, owner, state):
            prepare(key, owner, state)
            seen.append(state)
            d._attempt_dictation_cleanup(key, owner, state)
            _, nxt = d._register_cleanup(*key, new)
            prepare(key, new, nxt)
        d._prepare_cleanup = preparing
        d._close_dictation_resources(type('Turn', (), {'id': 'retired'})(), None, old)
        self.assertFalse(seen[0].running)
        self.assertIs(d._cleanup_owners['retired', 'provider'], new)

    def test_cleanup_pass_calls_failed_launch_retry(self):
        d = self.daemon()
        with mock.patch.object(processes, 'retry_failed_launches') as retry:
            d._retry_dictation_cleanup(0)
            retry.assert_called_once_with()

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
