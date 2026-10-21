"""Every logical slot/resource pair retains its own cleanup and retry state."""
import threading
import unittest
from unittest import mock

from tests import test_cleanup_recovery as recovery
from tests.test_terminal_outcomes import voiced


def daemon_fixture(case):
    daemon, _, _ = recovery.Boundaries.daemon(case)
    daemon._speech = daemon._dictation = daemon._player = daemon._socket = None
    daemon._socket_ino = None
    daemon._wake_w = -1
    daemon._speech_error = ''
    daemon._speech_error_serial = 0
    daemon._turns = 0
    daemon._refresh_config = lambda: None
    daemon._arbiter.speaking = daemon._arbiter.listening = False
    daemon._started_at = daemon._last_activity = 0
    daemon._idle_seconds = 60
    daemon._session_dir = daemon._socket_path = 'fixture'
    daemon._stt_status = lambda: {'available': False}
    daemon._tool_status = lambda _: {'available': False}
    return daemon


class Aliases(unittest.TestCase):
    def test_three_resources_and_each_logical_slot_keep_distinct_states(self):
        daemon = daemon_fixture(self)
        owners = [mock.Mock() for _ in range(3)]
        registrations = [daemon._register_cleanup('playback', 'player', owner)
                         for owner in owners]
        other_turn = daemon._register_cleanup('other', 'player', owners[0])
        other_kind = daemon._register_cleanup('playback', 'provider', owners[0])
        all_registrations = registrations + [other_turn, other_kind]
        self.assertEqual(len({key for key, _ in all_registrations}), 5)
        self.assertEqual(len({id(state) for _, state in all_registrations}), 5)
        self.assertEqual(len(daemon._cleanup_owners), 5)
        for _ in range(3):
            for owner, registration in zip(owners, registrations):
                self.assertEqual(daemon._register_cleanup('playback', 'player', owner),
                                 registration)
                self.assertIs(daemon._cleanup_owners[registration[0]], owner)
            self.assertEqual(daemon._register_cleanup('other', 'player', owners[0]),
                             other_turn)
            self.assertEqual(daemon._register_cleanup('playback', 'provider', owners[0]),
                             other_kind)

    def test_status_reports_all_failed_aliases_without_retrying_or_evicting(self):
        daemon = daemon_fixture(self)
        owners = [mock.Mock() for _ in range(4)]
        expected = set()
        for owner in owners:
            owner.close.side_effect = InterruptedError('fixture teardown refusal')
            key, state = daemon._register_cleanup('playback', 'player', owner)
            daemon._prepare_cleanup(key, owner, state)
            expected.add(key)
        daemon._retry_dictation_cleanup(2)
        calls = [owner.close.call_count for owner in owners]
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='off'), \
                mock.patch.object(voiced.settings, 'enabled', return_value=False):
            for _ in range(3):
                reply = daemon._op_status({'id': 'status'})
                self.assertTrue(reply['ok'])
                status = reply['status']
                for field in ('retained_cleanup', 'resource_owners'):
                    self.assertEqual({(row['turn'], row['kind']) for row in status[field]},
                                     expected)
                self.assertEqual([owner.close.call_count for owner in owners], calls)
                self.assertEqual(set(daemon._cleanup_owners), expected)
        for owner in owners:
            owner.close.side_effect = None
        daemon._retry_dictation_cleanup(2)
        self.assertFalse(daemon._cleanup_owners)

    def test_shutdown_closes_every_active_alias_including_current_player(self):
        daemon = daemon_fixture(self)
        owners = [mock.Mock(_process=None) for _ in range(4)]
        for owner in owners:
            daemon._register_cleanup('playback', 'player', owner)
        daemon._player = owners[-1]
        daemon._shutdown()
        daemon._shutdown()
        for owner in owners:
            owner.close.assert_called_once_with()
        self.assertIsNone(daemon._player)
        self.assertFalse(daemon._cleanup_owners)
        self.assertFalse(daemon._dictation_cleanup_state)
        self.assertFalse(daemon._retained_dictation_cleanup())

    def test_failed_start_releases_and_notifies_reservation_then_retries_owner(self):
        daemon = daemon_fixture(self)
        owners = [mock.Mock(), mock.Mock()]
        entries = []
        for owner in owners:
            key, state = daemon._register_cleanup('playback', 'player', owner)
            daemon._prepare_cleanup(key, owner, state)
            entries.append((key, state))
        start = threading.Thread.start
        starts = []

        def starting(worker):
            starts.append(worker)
            if len(starts) == 1:
                raise RuntimeError('fixture thread start refusal')
            return start(worker)

        with mock.patch.object(threading.Thread, 'start', starting):
            daemon._retry_dictation_cleanup(2)
        key, state = entries[0]
        self.assertFalse(state.running, 'a thread that never started still owns the reservation')
        self.assertTrue(state.done.is_set(), 'waiters were not notified of the refused start')
        self.assertIs(daemon._cleanup_owners.get(key), owners[0])
        self.assertIs(daemon._failed_dictation_cleanup.get(key), owners[0])
        owners[0].close.assert_not_called()
        owners[1].close.assert_called_once_with()
        self.assertEqual(len(daemon._retained_dictation_cleanup()), 1)
        daemon._retry_dictation_cleanup(2)
        for owner in owners:
            owner.close.assert_called_once_with()
        self.assertFalse(daemon._cleanup_owners)

    def test_successful_old_alias_cleanup_preserves_current_player(self):
        daemon = daemon_fixture(self)
        old, current = mock.Mock(), mock.Mock()
        old_key, old_state = daemon._register_cleanup('playback', 'player', old)
        new_key, new_state = daemon._register_cleanup('playback', 'player', current)
        daemon._player = current
        daemon._prepare_cleanup(old_key, old, old_state)
        daemon._retry_dictation_cleanup(2)
        old.close.assert_called_once_with()
        current.close.assert_not_called()
        self.assertIs(daemon._player, current)
        self.assertIs(daemon._cleanup_owners.get(new_key), current)
        self.assertIs(daemon._dictation_cleanup_state.get(new_key), new_state)
        self.assertFalse(new_state.closed)
        self.assertFalse(new_state.running)
        daemon._prepare_cleanup(new_key, current, new_state)
        daemon._retry_dictation_cleanup(2)
        current.close.assert_called_once_with()
        self.assertIsNone(daemon._player)


if __name__ == '__main__':
    unittest.main()
