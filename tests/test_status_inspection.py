"""Status remains answerable without accumulating unproven provider probes."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

from tests.test_cleanup_aliases import daemon_fixture
from tests.test_terminal_outcomes import voiced
from voicelib import processes


class Inspection(unittest.TestCase):
    def test_status_probes_and_reports_the_configured_female02(self):
        daemon = daemon_fixture(self)
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'), \
                mock.patch.object(voiced.settings, 'tts_voice', return_value='kilix-female-02'), \
                mock.patch.object(voiced.tts_lib, 'piper_probe', return_value=(True, 'ready')) as probe:
            status = daemon._tts_status()
            self.assertEqual((status['model'], status['voice']), ('kilix-female-02', 'kilix-female-02'))
            self.assertTrue(status['available'])
            self.assertEqual(probe.call_args.kwargs['model'], 'kilix-female-02')
            daemon._tts_inspecting = True
            blocked = daemon._tts_status()
            self.assertFalse(blocked['available'])
            self.assertEqual(blocked['voice'], 'kilix-female-02')
            self.assertEqual(probe.call_count, 1)

    def test_failed_probe_blocks_repeated_status_and_recovers_after_cleanup(self):
        daemon = daemon_fixture(self)
        owners = []
        factory = processes.SynthesisOwner
        before = len(os.listdir('/proc/self/fd'))

        def create():
            owner = factory()
            owners.append(owner)
            return owner

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'probe'
            payload = {'model': voiced.models.PIPER_KRISTIN_MODEL,
                       'voice': voiced.tts_lib.PIPER_VOICE, 'installed': True}
            binary.write_text('#!' + sys.executable + '\nprint(' + repr(json.dumps(payload)) + ')\n')
            binary.chmod(0o700)
            try:
                with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'), \
                        mock.patch.object(voiced.settings, 'enabled', return_value=False), \
                        mock.patch.object(voiced.tts_lib, 'piper_binary', return_value=str(binary)), \
                        mock.patch.object(processes, 'SynthesisOwner', side_effect=create):
                    with mock.patch.object(processes, 'kill_group',
                                           side_effect=InterruptedError('fixture pre-signal refusal')):
                        first = daemon._op_status({'id': 'first'})
                        self.assertTrue(first['ok'])
                        self.assertFalse(first['status']['tts']['available'])
                        expected = dict(daemon._cleanup_owners)
                        self.assertEqual(len(expected), 1)
                        for i in range(12):
                            reply = daemon._op_status({'id': str(i)})
                            self.assertTrue(reply['ok'], 'blocked inspection must still answer status')
                            status = reply['status']
                            self.assertFalse(status['tts']['available'])
                            self.assertEqual(len(owners), 1, 'status launched another unproven probe')
                            self.assertEqual(daemon._cleanup_owners, expected)
                            for field in ('retained_cleanup', 'resource_owners'):
                                self.assertEqual({(r['turn'], r['kind']) for r in status[field]},
                                                 set(expected))
                        self.assertIn('inspection', status['tts']['detail'])
                        self.assertIn('cleanup', status['tts']['detail'])
                        self.assertLessEqual(len(os.listdir('/proc/self/fd')), before + 1)
                    daemon._retry_dictation_cleanup(3)
                    self.assertFalse(daemon._cleanup_owners)
                    self.assertIsNone(owners[0]._process)
                    for _ in range(3):
                        self.assertTrue(daemon._tts_status()['available'])
                        self.assertFalse(daemon._cleanup_owners)
                    self.assertEqual(len(owners), 4, 'proven cleanup must reopen inspection admission')
            finally:
                for owner in owners:
                    owner.close()
                daemon._retry_dictation_cleanup(3)
        self.assertFalse(daemon._cleanup_owners)
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    def test_concurrent_status_does_not_start_a_second_inspection(self):
        daemon = daemon_fixture(self)
        entered, release = threading.Event(), threading.Event()
        results, errors = [], []

        def probe(**kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError('fixture inspection was not released')
            return True, 'fixture provider available'

        def inspect():
            try:
                results.append(daemon._tts_status())
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=inspect, daemon=True)
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'), \
                mock.patch.object(voiced.tts_lib, 'piper_probe', side_effect=probe) as probing:
            try:
                worker.start()
                self.assertTrue(entered.wait(5))
                # The second call must answer while the first is still paused.
                probing.side_effect = lambda **_: (True, 'duplicate inspection')
                reply = daemon._tts_status()
                self.assertFalse(reply['available'])
                self.assertEqual(probing.call_count, 1)
                self.assertEqual(len(daemon._resource_owners()), 1)
            finally:
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]['available'])
            self.assertFalse(daemon._cleanup_owners)
            self.assertTrue(daemon._tts_status()['available'])
            self.assertEqual(probing.call_count, 2)

    def test_probe_error_releases_admission_and_preserves_deadline(self):
        daemon = daemon_fixture(self)
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'), \
                mock.patch.object(voiced.tts_lib, 'piper_probe',
                                  side_effect=voiced.tts_lib.TtsDeadlineExceeded('fixture deadline')):
            with self.assertRaises(voiced.tts_lib.TtsDeadlineExceeded):
                daemon._tts_status()
        self.assertFalse(daemon._cleanup_owners)
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'), \
                mock.patch.object(voiced.tts_lib, 'piper_probe', return_value=(True, 'ready')):
            self.assertTrue(daemon._tts_status()['available'])

    def test_blocked_status_keeps_every_preexisting_owner_and_honors_deadline(self):
        daemon = daemon_fixture(self)
        owners = [mock.Mock() for _ in range(4)]
        expected = {}
        for owner in owners:
            key, state = daemon._register_cleanup('earlier-probe', 'synthesiser', owner)
            daemon._prepare_cleanup(key, owner, state)
            expected[key] = owner
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'), \
                mock.patch.object(voiced.settings, 'enabled', return_value=False), \
                mock.patch.object(voiced.tts_lib, 'piper_probe', return_value=(True, 'ready')) as probe:
            for _ in range(3):
                reply = daemon._op_status({'id': 'blocked'})
                self.assertTrue(reply['ok'])
                self.assertFalse(reply['status']['tts']['available'])
                for field in ('resource_owners', 'retained_cleanup'):
                    self.assertEqual({(row['turn'], row['kind']) for row in reply['status'][field]},
                                     set(expected))
                self.assertEqual(daemon._cleanup_owners, expected)
                self.assertEqual(daemon._failed_dictation_cleanup, expected)
                for owner in owners:
                    owner.close.assert_not_called()
            with self.assertRaises(voiced.tts_lib.TtsDeadlineExceeded):
                daemon._tts_status(deadline=0)
            probe.assert_not_called()
        daemon._retry_dictation_cleanup(2)
        self.assertFalse(daemon._cleanup_owners)

    def test_owner_construction_failure_does_not_leave_inspection_reserved(self):
        daemon = daemon_fixture(self)
        with mock.patch.object(voiced.settings, 'tts_engine', return_value='piper'):
            with mock.patch.object(processes, 'SynthesisOwner', side_effect=RuntimeError('fixture')):
                with self.assertRaisesRegex(RuntimeError, 'fixture'):
                    daemon._tts_status()
            with mock.patch.object(voiced.tts_lib, 'piper_probe', return_value=(True, 'ready')):
                self.assertTrue(daemon._tts_status()['available'])
        self.assertFalse(daemon._cleanup_owners)


if __name__ == '__main__':
    unittest.main()
