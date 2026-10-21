"""Initial cleanup ownership, atomic settlement and lease acknowledgement regressions."""
import json
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from tests.test_cleanup_recovery import Boundaries
from tests.test_accelerator import _LeaseFixture
from tests.test_terminal_outcomes import voiced, _worker_daemon, _LiveTurnsFixture, wait_until
from voicelib import accelerator, audio, jobs, stt, processes

class Boundaries8(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_stop_accepted_at_return_from_last_cleanup_check_keeps_cause(self):
        for cause in ('cancelled', 'deadline'):
            for failing in ('capture', 'provider', 'both', 'neither'):
                with self.subTest(cause=cause, failing=failing):
                    d, sent, ledger = self.daemon()
                    now = [0.0]
                    turn = voiced._DictationTurn('last-check', mock.Mock(), deadline=1, clock=lambda: now[0])
                    turn.on_settle = ledger.record
                    d._dictation = turn
                    capture, engine = (mock.Mock(), mock.Mock())
                    accepted = []
                    if failing in ('capture', 'both'):
                        capture.stop.side_effect = InterruptedError('capture')
                    if failing in ('provider', 'both'):
                        engine.close.side_effect = InterruptedError('provider')
                    d._record = mock.Mock(side_effect=stt.SttError('primary decoder error'))
                    real = d._check_delivery

                    def checking(t, where):
                        real(t, where)
                        if where == 'while resources shut down':
                            if cause == 'cancelled':
                                accepted.append(d._op_stop_dictation({'id': 'a', 'mode': 'abort'}))
                            now[0] = 2
                    d._check_delivery = checking
                    d._dictate = lambda t: d._record_turn(t, SimpleNamespace(), None)
                    with mock.patch.object(voiced.audio, 'MicCapture', return_value=capture), mock.patch.object(voiced.stt_lib, 'make_stt', return_value=engine):
                        d._run_dictation(turn)
                    row = ledger.get(turn.id)
                    print('LAST_CLEANUP_CHECK', json.dumps(dict(cause=cause, failing=failing, accepted=accepted, outcome=row['outcome'], code=row['code'], terminals=len(sent))), flush=True)
                    self.assertEqual((row['outcome'], row['code']), (cause, cause))
                    self.assertEqual(len(sent), 1)
                    self.assertEqual(sent[0]['code'], cause)
                    self.assertNotIn('final', sent[0])
                    if cause == 'cancelled':
                        self.assertEqual(len(accepted), 1)
                        self.assertTrue(accepted[0]['stopped'])
                        self.assertTrue(accepted[0]['quiesced'])

    def test_initial_cleanup_is_reported_at_shutdown_join_timeout(self):
        d, sent, ledger = self.daemon()
        d._speech = d._player = d._socket = None
        d._socket_ino = None
        d._wake_w = -1
        cfg = {'audio': {'capture_cmd': [sys.executable, '-c', 'import time;time.sleep(60)']}}
        capture = audio.MicCapture(cfg)
        engine = mock.Mock()
        entered = threading.Event()
        release = threading.Event()
        stop = capture.stop
        turn = voiced._DictationTurn('initial-stuck', mock.Mock())
        turn.on_settle = ledger.record
        d._dictation = turn
        child = []

        def record(*_):
            child.append(capture._process)
            self.assertEqual({r['kind'] for r in d._resource_owners()}, {'capture', 'provider'})
            self.assertTrue(d._op_stop_dictation({'id': 'a', 'mode': 'abort'})['stopped'])
            turn.stop.check()

        def blocked():
            entered.set()
            release.wait(15)
            stop()
        capture.stop = blocked
        d._record = record
        d._dictate = lambda t: d._record_turn(t, SimpleNamespace(), None)
        try:
            with mock.patch.object(voiced.audio, 'MicCapture', return_value=capture), mock.patch.object(voiced.stt_lib, 'make_stt', return_value=engine):
                turn.thread = threading.Thread(target=d._run_dictation, args=(turn,), daemon=True)
                turn.thread.start()
                self.assertTrue(entered.wait(3))
                before = d._retained_dictation_cleanup()
                start = time.monotonic()
                d._shutdown()
                elapsed = time.monotonic() - start
                warnings = [c.args[0] for c in d._warn.call_args_list]
                live = processes.poll_exit(child[0]) is None
                print('INITIAL_CLEANUP_SHUTDOWN', json.dumps(dict(seconds=elapsed, recorder_live=live, retained=before, states=list(map(list, d._dictation_cleanup_state)), warnings=warnings, worker_alive=turn.thread.is_alive(), session_release=d._arbiter.release.call_count)), flush=True)
                self.assertTrue(any(('shutdown' in w and 'unproven' in w and (turn.id in w) for w in warnings)), 'shutdown silently releases the session with initial cleanup and its recorder still live')
                self.assertTrue(any((r['kind'] == 'capture' and r.get('state') == 'in_flight' for r in before)), before)
                self.assertTrue(wait_until(lambda: processes.poll_exit(child[0]) is not None, .5), 'shutdown did not signal initial cleanup')
                self.assertLess(elapsed, 6.5)
                self.assertTrue(d._retained_dictation_cleanup())
        finally:
            release.set()
            if turn.thread:
                turn.thread.join(5)
            stop()

    def test_retry_waits_are_shared_and_threads_do_not_accumulate(self):
        d, _, _ = self.daemon()
        release = threading.Event()
        calls = [0, 0]
        owners = []
        for i in range(2):
            owner = mock.Mock()
            owner.stop.side_effect = InterruptedError('first')
            owners.append(owner)
            d._close_dictation_resources(SimpleNamespace(id=str(i)), owner, None)

        def blocking(i):
            calls[i] += 1
            release.wait(8)
        for i, owner in enumerate(owners):
            owner.stop.side_effect = lambda i=i: blocking(i)
        try:
            durations = []
            for _ in range(8):
                start = time.monotonic()
                d._retry_dictation_cleanup(0.1)
                durations.append(time.monotonic() - start)
                self.assertEqual(len(d._retained_dictation_cleanup()), 2)
            threads = [t for t in threading.enumerate() if t.name.startswith('kilix-voice-cleanup-')]
            print('RETRY_BOUNDS', json.dumps(dict(calls=calls, threads=len(threads), durations=durations)), flush=True)
            self.assertEqual(calls, [1, 1])
            self.assertEqual(len(threads), 2)
            self.assertLess(max(durations), 0.17)
        finally:
            release.set()
            d._retry_dictation_cleanup(1)
        self.assertFalse(d._failed_dictation_cleanup)

class LeaseAcknowledgement(_LeaseFixture):

    def test_quarantine_refuses_acknowledgement_even_if_descriptor_is_open(self):
        # Quarantine is authoritative even if a provider's release failed
        # before closing its descriptor. Never turn a later no-op into proof.
        grant = accelerator.ExecutionGrant(None, lambda error: error)
        grant._lease = mock.Mock(_fd=71)
        grant.acknowledgement = 'quarantined'
        with self.assertRaises(accelerator.AcceleratorRefused):
            grant.reaped()
        grant._lease.release.assert_not_called()
        self.assertFalse(grant._reaped)

    def test_failed_acknowledgement_cannot_be_silently_acknowledged_by_retry(self):
        d = _worker_daemon([], jobs.JobLedger())
        d._warn = mock.Mock()
        owner = mock.Mock()
        turn = SimpleNamespace(id='failed-ack')
        with self.execution() as grant:
            with mock.patch.object(self.lease._Registry, 'save', side_effect=OSError('fixture write failure')):
                first = d._close_dictation_resources(turn, None, owner, grant)
        before_fd = grant._lease._fd
        retry_error = None
        try:
            grant.reaped()
        except accelerator.AcceleratorRefused as error:
            retry_error = error
        self.assertIsNotNone(retry_error, 'a closed release cannot acknowledge cleanup')
        self.assertFalse(grant._reaped)
        d._retry_dictation_cleanup(0.2)
        try:
            refused = None
            try:
                with self.execution(job_id='next', deadline_monotonic=time.monotonic() + 0.2) as nxt:
                    nxt.reaped()
            except accelerator.AcceleratorRefused as error:
                refused = str(error)
            print('ACK_FAILURE_RETRY', json.dumps(dict(first_error=type(first).__name__, fd_after_error=before_fd, reaped=grant._reaped, retained=len(d._failed_dictation_cleanup or {}), next_refused=refused, close_calls=owner.close.call_count)), flush=True)
            self.assertFalse(grant._reaped, 'a no-op on an already closed lease is treated as an acknowledgement')
            self.assertEqual(grant.acknowledgement, 'quarantined')
            self.assertEqual(owner.close.call_count, 1)
            for _ in range(3):
                d._retry_dictation_cleanup(0.05)
            rows = d._retained_dictation_cleanup()
            self.assertEqual(rows[0]['state'], 'quarantined')
            self.assertIn('namespace and its anchor', rows[0]['recovery'])
            d._lock = threading.RLock()
            d._stopping = threading.Event()
            d._speech = d._dictation = d._player = d._socket = None
            d._socket_ino = None
            d._wake_w = -1
            d._arbiter = mock.Mock()
            d._shutdown()
            self.assertIn('acknowledgement failed', str(d._warn.call_args_list))
            self.assertTrue(d._retained_dictation_cleanup())
            self.assertIn('quarantined', refused)
        finally:
            grant._lease.release(cleanup_complete=False)

class Privacy(_LiveTurnsFixture):

    def test_raw_status_omits_capture_and_playback_command_arguments(self):
        self.daemon._overrides['audio']['capture_cmd'] = ['/fixture/recorder', 'private-capture-argument']
        self.daemon._overrides['audio']['play_cmd'] = ['/fixture/player', 'private-playback-argument']
        payload = self.status()['status']
        print('STATUS_ARGUMENTS', json.dumps({k: payload[k] for k in ['retained_cleanup', 'capture', 'playback']}), flush=True)
        self.assertNotIn('private-capture-argument', json.dumps(payload))
        self.assertNotIn('private-playback-argument', json.dumps(payload))

class SpeechOwnership(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_blocked_synthesiser_close_remains_visible_and_shutdown_escalates(self):
        import subprocess
        from pathlib import Path
        d, _, _ = self.daemon()
        d._speech = d._player = d._socket = None
        d._socket_ino = None
        d._wake_w = -1
        entered, release = (threading.Event(), threading.Event())
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
        engine = mock.Mock()
        engine._process = child
        engine.synth.return_value = (b'\x00\x00', 16000)

        def close():
            entered.set()
            release.wait(10)
            processes.kill_group(child)
            child.wait(2)
        engine.close.side_effect = close
        player = mock.Mock()
        player.error = None
        d._get_player = lambda: player
        d._synth = lambda *_: None
        d._speech_error = ''
        d._speech_error_serial = 0
        turn = voiced._SpeechTurn('blocked-synth', ['fixture'], engine)
        d._speech = turn
        turn.thread = threading.Thread(target=d._run_speech, args=(turn,), daemon=True)
        try:
            with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', 0.1), mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', 0.1):
                turn.thread.start()
                self.assertTrue(entered.wait(1), 'speech never tears down its synthesiser')
                rows = d._retained_dictation_cleanup()
                self.assertTrue(any((r['kind'] == 'synthesiser' for r in rows)), rows)
                started = time.monotonic()
                d._shutdown()
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertTrue(wait_until(lambda: processes.poll_exit(child) is not None, .5))
                self.assertEqual(engine.close.call_count, 1, 'duplicate first cleanup')
                warnings = str(d._warn.call_args_list)
                self.assertIn('blocked-synth', warnings)
                self.assertIn('cleanup unproven', warnings)
        finally:
            release.set()
            turn.thread.join(3)
            processes.kill_group(child)
            child.wait(2)

    def test_blocked_player_close_cannot_block_shutdown_or_lose_owner(self):
        import subprocess
        d, _, _ = self.daemon()
        d._speech = d._dictation = d._socket = None
        d._socket_ino = None
        d._wake_w = -1
        entered, release = (threading.Event(), threading.Event())
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
        player = mock.Mock()
        player._process = child

        def close():
            entered.set()
            release.wait(10)
            processes.kill_group(child)
            child.wait(2)
        player.close.side_effect = close
        d._player = player
        worker = threading.Thread(target=d._shutdown, daemon=True)
        try:
            with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', 0.1), mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', 0.1):
                worker.start()
                self.assertTrue(entered.wait(1))
                worker.join(0.6)
                self.assertFalse(worker.is_alive(), 'shutdown blocks inside the first player.close')
                self.assertTrue(wait_until(lambda: processes.poll_exit(child) is not None, .5))
                self.assertEqual(player.close.call_count, 1)
                self.assertTrue(any((r['kind'] == 'player' for r in d._retained_dictation_cleanup())))
                self.assertIn('cleanup unproven', str(d._warn.call_args_list))
        finally:
            release.set()
            worker.join(3)
            processes.kill_group(child)
            child.wait(2)

class TwoOwnerCalls(_LiveTurnsFixture):

    def test_retry_admission_and_shutdown_share_budget_and_keep_one_worker_per_owner(self):
        d = self.daemon
        release = threading.Event()
        calls = [[], []]
        owners = []
        for i in range(2):
            owner = mock.Mock()
            owner.stop.side_effect = InterruptedError('first attempt')
            owners.append(owner)
            d._close_dictation_resources(SimpleNamespace(id=str(i)), owner, None)

        def blocked(index):
            calls[index].append(threading.get_ident())
            release.wait(10)
        for i, owner in enumerate(owners):
            owner.stop.side_effect = lambda i=i: blocked(i)
        try:
            with mock.patch.object(voiced, 'CLEANUP_RETRY_WAIT_S', 0.15), mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', 0.15):
                for _ in range(3):
                    for action in ('retry', 'admission', 'shutdown'):
                        with self.subTest(action=action):
                            start = time.monotonic()
                            if action == 'retry':
                                d._retry_dictation_cleanup(0.15)
                            elif action == 'admission':
                                result = self.call(op='dictate', sock=self.receiver().path)
                                self.assertFalse(result['ok'])
                                self.assertEqual(result['code'], 'busy')
                            else:
                                d._shutdown()
                            self.assertLess(time.monotonic() - start, 0.26, 'each owner received a fresh wait budget')
                            self.assertEqual(list(map(len, calls)), [1, 1])
                            workers = [t for t in threading.enumerate() if t.ident in {n for row in calls for n in row}]
                            self.assertEqual(len(workers), 2)
                            self.assertEqual(len(d._retained_dictation_cleanup()), 2)
        finally:
            release.set()
            d._retry_dictation_cleanup(1)

class RealSpeechChildren(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_each_synthesiser_retains_first_blocked_teardown_for_shutdown(self):
        import contextlib
        import os
        import tempfile
        from pathlib import Path
        from voicelib import tts, qwen_provider
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / 'synth'
            fixture.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(60)\n')
            fixture.chmod(448)
            for kind in ('espeak', 'piper', 'qwen'):
                with self.subTest(kind=kind), contextlib.ExitStack() as stack:
                    d, _, _ = self.daemon()
                    d._speech = d._dictation = d._player = d._socket = None
                    d._socket_ino = None
                    d._wake_w = -1
                    d._speech_error = ''
                    d._speech_error_serial = 0
                    if kind == 'espeak':
                        engine = tts.EspeakTts({'tts': {'cmd': [str(fixture)]}}, voice='en-us', rate=170)
                    elif kind == 'piper':
                        engine = tts.PiperTts(rate=170)
                        stack.enter_context(mock.patch.object(tts, 'piper_binary', return_value=str(fixture)))
                    else:
                        engine = qwen_provider.QwenProviderTts()
                        stack.enter_context(mock.patch.object(qwen_provider, 'provider_binary', return_value=str(fixture)))
                    entered, release = (threading.Event(), threading.Event())
                    captured = []
                    finish = engine._finish_process

                    def blocked(process):
                        captured.append(process)
                        entered.set()
                        release.wait(10)
                        finish(process)
                    engine._finish_process = blocked
                    player = mock.Mock()
                    player.error = None
                    d._get_player = lambda: player
                    turn = voiced._SpeechTurn('real-' + kind, ['fixture'], engine, deadline=0.05, clock=lambda: 0.0)
                    d._speech = turn
                    turn.thread = threading.Thread(target=d._run_speech, args=(turn,), daemon=True)
                    try:
                        turn.thread.start()
                        self.assertTrue(entered.wait(15), 'first cleanup not reached')
                        child = captured[0]
                        self.assertEqual(os.getpgid(child.pid), child.pid)
                        self.assertIs(engine._process, child)
                        self.assertIsNone(child.returncode, 'leader reaped before final group signal')
                        self.assertTrue(any((r['kind'] == 'synthesiser' for r in d._resource_owners())))
                        with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', 0.1), mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', 0.1):
                            start = time.monotonic()
                            d._shutdown()
                            self.assertLess(time.monotonic() - start, 0.5)
                        self.assertTrue(wait_until(lambda: processes.poll_exit(child) is not None, .5))
                        self.assertTrue(child._voice_group_signalled)
                        self.assertIn('cleanup unproven', str(d._warn.call_args_list))
                        self.assertTrue(d._retained_dictation_cleanup())
                    finally:
                        release.set()
                        turn.thread.join(3)
                        engine.close()
                        d._retry_dictation_cleanup(1)
                    self.assertFalse(turn.thread.is_alive())
                    self.assertEqual(d._retained_dictation_cleanup(), [])
                    self.assertFalse(Path('/proc', str(child.pid)).exists())

    def test_real_player_group_is_retained_while_first_close_blocks(self):
        import os
        from pathlib import Path
        d, _, _ = self.daemon()
        d._speech = d._dictation = d._socket = None
        d._socket_ino = None
        d._wake_w = -1
        player = audio.Player({'audio': {'play_cmd': [sys.executable, '-c', 'import sys,time; sys.stdin.buffer.read(); time.sleep(60)']}})
        d._player = player
        d._player = None
        with mock.patch.object(voiced.audio, 'Player', return_value=player):
            self.assertIs(d._get_player(), player)
        entered, release = (threading.Event(), threading.Event())
        close_hook = player._on_close_requested

        def blocked():
            entered.set()
            release.wait(15)
            close_hook()
        player._on_close_requested = blocked
        try:
            player.play(b'\x00\x00', 16000)
            end = time.monotonic() + 2
            while player._process is None and time.monotonic() < end:
                time.sleep(0.005)
            child = player._process
            self.assertIsNotNone(child)
            self.assertEqual(os.getpgid(child.pid), child.pid)
            # Establish the blocked first close before measuring shutdown.
            # Under load its worker need not start within shutdown's budget.
            key, state = d._register_cleanup('playback', 'player', player)
            d._prepare_cleanup(key, player, state)
            d._retry_dictation_cleanup(0)
            self.assertTrue(entered.wait(15), 'first player cleanup not reached')
            with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', 0.1), mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', 0.1):
                start = time.monotonic()
                d._shutdown()
                self.assertLess(time.monotonic() - start, 0.5)
            self.assertTrue(entered.is_set())
            self.assertTrue(wait_until(lambda: processes.poll_exit(child) is not None, .5))
            self.assertTrue(d._retained_dictation_cleanup())
            self.assertIn('cleanup unproven', str(d._warn.call_args_list))
        finally:
            release.set()
            player._on_close_requested = close_hook
            d._retry_dictation_cleanup(2)
            player.close()
            d._retry_dictation_cleanup(2)
        self.assertFalse(Path('/proc', str(child.pid)).exists())
        self.assertEqual(d._retained_dictation_cleanup(), [])

class RetryableAcknowledgement(_LeaseFixture):

    def test_acknowledgement_deadline_with_open_descriptor_can_retry(self):
        d = _worker_daemon([], jobs.JobLedger())
        d._warn = mock.Mock()
        owner = mock.Mock()
        with self.execution() as grant:
            fd = grant._lease._fd
            with mock.patch.object(self.lease, '_Registry', side_effect=self.lease.LeaseError('deadline', 'busy registry')):
                error = d._close_dictation_resources(SimpleNamespace(id='ack-timeout'), None, owner, grant)
            self.assertIsNotNone(error)
            self.assertFalse(grant._reaped)
            self.assertEqual(grant._lease._fd, fd)
            self.assertEqual(grant.acknowledgement, 'retryable')
            self.assertEqual(len(d._retained_dictation_cleanup()), 1)
            d._retry_dictation_cleanup(1)
            self.assertTrue(grant._reaped)
            self.assertEqual(grant.acknowledgement, 'acknowledged')
            self.assertEqual(d._retained_dictation_cleanup(), [])
            owner.close.assert_called_once()
        with self.execution(job_id='successor') as successor:
            successor.reaped()

class ConstructorOwnership(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_provider_is_discoverable_during_constructor_cleanup(self):
        import subprocess
        d, _, _ = self.daemon()
        d._speech = d._player = d._socket = None
        d._socket_ino = None
        d._wake_w = -1
        entered, release = threading.Event(), threading.Event()
        children = []
        calls = []
        original_close = stt.WhisperStt.close
        def starting(engine):
            engine._process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                                                start_new_session=True)
            children.append(engine._process)
            raise OSError('construction failed after launch')
        def closing(engine):
            calls.append(threading.get_ident())
            entered.set()
            release.wait(10)
            original_close(engine)
        turn = voiced._DictationTurn('construction', mock.Mock())
        d._dictation = turn
        resolved = SimpleNamespace(engine=stt.ENGINE_WHISPER, model_dir='/fixture', consented_payload=None)
        d._dictate = lambda t: d._record_turn(t, resolved, None)
        turn.thread = threading.Thread(target=d._run_dictation, args=(turn,), daemon=True)
        with mock.patch.object(stt.WhisperStt, '_missing', return_value=[]), \
                mock.patch.object(stt, '_resolve_model', return_value='/fixture'), \
                mock.patch.object(stt.WhisperStt, '_binary_path', return_value='/fixture'), \
                mock.patch.object(stt.WhisperStt, '_hold'), \
                mock.patch.object(stt.WhisperStt, '_start', starting), \
                mock.patch.object(stt.WhisperStt, 'close', closing), \
                mock.patch.object(voiced.audio, 'MicCapture', return_value=mock.Mock(rate=16000)):
            try:
                turn.thread.start()
                self.assertTrue(entered.wait(2))
                owned = d._cleanup_owners or {}
                self.assertIn((turn.id, 'provider'), owned,
                              'provider hidden until its constructor returns')
                with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', .1), \
                        mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', .1):
                    d._shutdown()
                self.assertEqual(len(calls), 1, 'shutdown starts a second close during constructor cleanup')
                self.assertTrue(wait_until(lambda: processes.poll_exit(children[0]) is not None, .5))
                self.assertIn('provider cleanup unproven', str(d._warn.call_args_list))
            finally:
                release.set()
                turn.thread.join(3)
                for child in children:
                    processes.kill_group(child)
                    child.wait(2)

class AtomicSettlement(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_abort_accepted_at_settlement_lock_boundary_wins_error(self):
        d, sent, ledger = self.daemon()
        turn = voiced._DictationTurn('settle-boundary', mock.Mock())
        turn.on_settle = ledger.record
        d._dictation = turn
        entered, release = threading.Event(), threading.Event()
        lock = threading.Lock()
        class BoundaryLock:
            def __enter__(self):
                if threading.current_thread().name == 'settlement':
                    entered.set()
                    release.wait(3)
                lock.acquire()
            def __exit__(self, *_):
                lock.release()
        turn._settle_lock = BoundaryLock()
        worker = threading.Thread(name='settlement', target=d._deliver_terminal,
                                  args=(turn, {'error': 'decoder failed', 'code': 'unavailable'}),
                                  daemon=True)
        try:
            worker.start()
            self.assertTrue(entered.wait(2))
            reply = d._op_stop_dictation({'id': 'abort', 'mode': 'abort'})
            self.assertTrue(reply['stopped'])
            release.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual([m['code'] for m in sent], ['cancelled'])
            self.assertEqual(ledger.get(turn.id)['outcome'], 'cancelled')
        finally:
            release.set()
            worker.join(3)


class DirectCleanupCauses(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_record_turn_preserves_stop_at_its_own_cleanup_boundary(self):
        for established in (False, True):
            for cause in ('cancelled', 'deadline'):
                for failing in ('capture', 'provider', 'both', 'neither'):
                    with self.subTest(established=established, cause=cause, failing=failing):
                        d, _, _ = self.daemon()
                        now = [0.0]
                        turn = voiced._DictationTurn('direct', mock.Mock(), deadline=1, clock=lambda: now[0])
                        d._dictation = turn
                        capture, engine = mock.Mock(), mock.Mock()
                        def stop_cause():
                            if cause == 'cancelled':
                                self.assertTrue(d._op_stop_dictation({'id': 'a', 'mode': 'abort'})['stopped'])
                            now[0] = 2
                        def record(*_):
                            if established:
                                stop_cause()
                                turn.stop.check()
                            raise stt.SttError('primary decoder failure')
                        def close_capture():
                            if not established:
                                stop_cause()
                            if failing in ('capture', 'both'):
                                raise InterruptedError('capture')
                        capture.stop.side_effect = close_capture
                        if failing in ('provider', 'both'):
                            engine.close.side_effect = InterruptedError('provider')
                        d._record = record
                        actual = None
                        with mock.patch.object(voiced.audio, 'MicCapture', return_value=capture), \
                                mock.patch.object(voiced.stt_lib, 'make_stt', return_value=engine):
                            try:
                                d._record_turn(turn, SimpleNamespace(), None)
                            except Exception as error:
                                actual = error
                        expected = voiced.Cancelled if cause == 'cancelled' else voiced.DeadlineExceeded
                        self.assertIsInstance(actual, expected)
                        self.assertIsNone(turn.outcome)


class GroupIdentity(unittest.TestCase):
    def test_synthesis_poll_does_not_release_group_identity_before_final_signal(self):
        import os
        child = processes.GroupPopen([sys.executable, '-c', 'pass'])
        try:
            end = time.monotonic() + 2
            while child.poll() is None and time.monotonic() < end:
                time.sleep(.005)
            self.assertEqual(child.poll(), 0)
            self.assertIsNone(child.returncode, 'poll reaped the group leader')
            observed = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            self.assertIsNotNone(observed)
            child.wait(timeout=1)
            self.assertTrue(child._voice_group_signalled)
            self.assertEqual(child.returncode, 0)
        finally:
            processes.kill_group(child)
            child.wait(timeout=2)

class StopSpeechOwnership(unittest.TestCase):
    daemon = Boundaries.daemon

    def test_cancel_registers_blocked_teardown_and_returns_with_one_worker(self):
        import subprocess
        d, _, _ = self.daemon()
        d._dictation = d._player = d._socket = None
        d._socket_ino = None
        d._wake_w = -1
        entered, release = threading.Event(), threading.Event()
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                                 start_new_session=True)
        calls = []
        def close():
            calls.append(threading.get_ident())
            entered.set()
            release.wait(10)
            processes.kill_group(child)
            child.wait(2)
        engine = SimpleNamespace(_process=child, close=close, cancel=close)
        turn = voiced._SpeechTurn('cancel-blocked', ['fixture'], engine)
        d._speech = turn
        d._register_cleanup(turn.id, 'synthesiser', engine)
        worker = threading.Thread(target=d._cancel_speech, daemon=True)
        try:
            worker.start()
            self.assertTrue(entered.wait(2))
            worker.join(.6)
            self.assertFalse(worker.is_alive(), 'cancel blocks before publishing its cleanup')
            self.assertEqual(d._retained_dictation_cleanup()[0]['state'], 'in_flight')
            d._cancel_speech()
            with mock.patch.object(voiced, 'JOIN_TIMEOUT_S', .1), \
                    mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', .1):
                d._shutdown()
            self.assertEqual(len(calls), 1)
            self.assertTrue(wait_until(lambda: processes.poll_exit(child) is not None, .5))
        finally:
            release.set()
            worker.join(3)
            d._retry_dictation_cleanup(1)
            processes.kill_group(child)
            child.wait(2)

class BusyProviderAcknowledgement(_LeaseFixture):
    def test_close_handoff_to_decode_is_not_reap_proof(self):
        from tests.test_process_cleanup import CleanupTests
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(provider=engine_type.__name__):
                engine, child = CleanupTests.provider(self, engine_type)
                engine._busy = True
                d = _worker_daemon([], jobs.JobLedger())
                d._warn = mock.Mock()
                with self.execution() as grant:
                    error = d._close_dictation_resources(SimpleNamespace(id='busy-decode'), None, engine, grant)
                    self.assertIsNotNone(error, 'close handed reaping to decode; it did not prove teardown')
                    self.assertFalse(grant._reaped)
                    self.assertGreaterEqual(grant._lease._fd, 0)
                    self.assertIsNone(child.returncode)
                    self.assertTrue(d._retained_dictation_cleanup())
                    engine._busy = False
                    d._retry_dictation_cleanup(1)
                    self.assertIsNotNone(child.returncode)
                    self.assertTrue(grant._reaped)
                    self.assertEqual(d._retained_dictation_cleanup(), [])
