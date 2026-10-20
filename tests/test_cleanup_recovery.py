"""Cleanup recovery and signal boundaries; synthetic owners and private leases only."""
import contextlib
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from tests.test_stt_tool import tool
from tests.test_terminal_outcomes import voiced, _worker_daemon, _LiveTurnsFixture, wait_until
from voicelib import audio, jobs, stt, processes, accelerator, resources

class Boundaries(unittest.TestCase):

    def daemon(self):
        sent = []
        ledger = jobs.JobLedger()
        d = _worker_daemon(sent, ledger)
        d._cfg = {}
        d._lock = threading.RLock()
        d._stopping = threading.Event()
        d._warn = mock.Mock()
        d._arbiter = mock.Mock()
        d._dictation = None
        d._clear_dictation = lambda turn: voiced.Daemon._clear_dictation(d, turn)
        return (d, sent, ledger)

    def test_retained_capture_is_retried_before_successor_or_shutdown(self):
        d, sent, ledger = self.daemon()
        cfg = {'audio': {'capture_cmd': [sys.executable, '-c', 'import time; time.sleep(60)']}}
        capture = audio.MicCapture(cfg)
        engine = mock.Mock()
        engine.start_utterance.return_value = None
        turn = voiced._DictationTurn('leaked-capture', mock.Mock())
        turn.on_settle = ledger.record
        d._dictation = turn
        child = []
        accepted = []

        def record(*_):
            child.append(capture._process)
            accepted.append(d._op_stop_dictation({'id': 'abort', 'mode': 'abort'})['stopped'])
            turn.stop.check()
        d._record = record
        d._dictate = lambda t: d._record_turn(t, SimpleNamespace(), None)
        try:
            with mock.patch.object(voiced.audio, 'MicCapture', return_value=capture), mock.patch.object(voiced.stt_lib, 'make_stt', return_value=engine), mock.patch.object(audio, 'kill_group', side_effect=InterruptedError('one transient pre-signal failure')):
                d._run_dictation(turn)
            first = dict(accepted=accepted, outcome=ledger.get(turn.id)['outcome'], active_turn=d._dictation, recorder_live=processes.poll_exit(child[0]) is None, registry=[list(k) for k in d._failed_dictation_cleanup], warnings=[c.args[0] for c in d._warn.call_args_list])
            print('AFTER_FAILED_CAPTURE', json.dumps(first), flush=True)
            second = voiced._DictationTurn('successor', mock.Mock())
            second.on_settle = ledger.record
            d._dictation = second
            successor_capture = audio.MicCapture(cfg)
            next_engine = mock.Mock()
            next_engine.end_utterance.return_value = 'successor words'
            d._record = lambda *_: True
            with mock.patch.object(voiced.audio, 'MicCapture', return_value=successor_capture), mock.patch.object(voiced.stt_lib, 'make_stt', return_value=next_engine):
                d._run_dictation(second)
            before_shutdown = processes.poll_exit(child[0]) is None
            d._speech = d._dictation = d._player = d._socket = None
            d._socket_ino = None
            d._wake_w = -1
            start = time.monotonic()
            d._shutdown()
            elapsed = time.monotonic() - start
            after = dict(successor=ledger.get(second.id)['outcome'], old_recorder_live_before_shutdown=before_shutdown, old_recorder_live_after_shutdown=processes.poll_exit(child[0]) is None, shutdown_seconds=elapsed, registry_size=len(d._failed_dictation_cleanup), stop_after_shutdown=d._op_stop_dictation({'id': 'again', 'mode': 'abort'}))
            print('AFTER_SUCCESSOR_AND_SHUTDOWN', json.dumps(after), flush=True)
            self.assertEqual(first['outcome'], 'cancelled')
            self.assertEqual(after['successor'], 'completed')
            self.assertLess(elapsed, 1)
            self.assertFalse(after['old_recorder_live_after_shutdown'], 'recoverable recorder failure is never retried, including shutdown')
        finally:
            capture.stop()
            if child:
                self.assertIsNotNone(child[0].returncode)

    def test_registry_retains_all_owners_without_eviction(self):
        d, _, _ = self.daemon()
        owners = []
        for i in range(1025):
            c = mock.Mock()
            c.stop.side_effect = InterruptedError('transient')
            owners.append(c)
            d._close_dictation_resources(SimpleNamespace(id=f'failed-{i}'), c, None)
        print('REGISTRY_SIZE', len(d._failed_dictation_cleanup), 'failed_owners', len(owners), flush=True)
        self.assertEqual(len(d._failed_dictation_cleanup), len(owners))
        for i, c in enumerate(owners):
            self.assertIs(d._failed_dictation_cleanup[f'failed-{i}', 'capture'], c)

    def test_real_grant_remains_quarantined_after_successful_resource_retry(self):
        d, _, _ = self.daemon()
        turn = voiced._DictationTurn('quarantine', mock.Mock())
        engine = mock.Mock()
        engine.close.side_effect = InterruptedError('transient')
        with tempfile.TemporaryDirectory() as tmp:
            namespace = str(Path(tmp) / 'leases')
            opts = dict(device_class=resources.DEVICE_CUDA, task='transcribe', job_id=turn.id, deadline_monotonic=time.monotonic() + 2, token=turn.stop, device_label='gpu0', namespace=namespace)
            with accelerator.execution(**opts) as grant:
                self.assertIsNotNone(d._close_dictation_resources(turn, None, engine))
            engine.close.side_effect = None
            self.assertIsNone(d._close_dictation_resources(turn, None, engine))
            self.assertEqual(d._failed_dictation_cleanup, {})
            grant.reaped()
            refused = None
            try:
                with accelerator.execution(**dict(opts, job_id='next', deadline_monotonic=time.monotonic() + 0.2)) as next_grant:
                    next_grant.reaped()
            except accelerator.AcceleratorRefused as error:
                refused = str(error)
            print('QUARANTINE_AFTER_SUCCESSFUL_RETRY', json.dumps(dict(refused=refused, registry=d._failed_dictation_cleanup)), flush=True)
            self.assertIsNone(refused, 'successful resource teardown has no route to clear its prior quarantine')

    def test_stop_during_cleanup_after_provider_error(self):
        for cause in ['cancelled', 'deadline']:
            for failing in ['capture', 'provider', 'both', 'neither']:
                with self.subTest(cause=cause, failing=failing):
                    d, sent, ledger = self.daemon()
                    now = [0.0]
                    turn = voiced._DictationTurn('late-stop', mock.Mock(), deadline=1, clock=lambda: now[0])
                    turn.on_settle = ledger.record
                    d._dictation = turn
                    capture = mock.Mock()
                    engine = mock.Mock()
                    accepted = []

                    def recording(*_):
                        raise stt.SttError('primary decoder error')

                    def stopping():
                        if cause == 'cancelled':
                            accepted.append(d._op_stop_dictation({'id': 'abort', 'mode': 'abort'})['stopped'])
                        now[0] = 2
                        if failing in ['capture', 'both']:
                            raise InterruptedError('capture cleanup')
                    capture.stop.side_effect = stopping
                    if failing in ['provider', 'both']:
                        engine.close.side_effect = InterruptedError('provider cleanup')
                    d._record = recording
                    d._dictate = lambda t: d._record_turn(t, SimpleNamespace(), None)
                    with mock.patch.object(voiced.audio, 'MicCapture', return_value=capture), mock.patch.object(voiced.stt_lib, 'make_stt', return_value=engine):
                        d._run_dictation(turn)
                    row = ledger.get(turn.id)
                    print('LATE_STOP_AFTER_PRIMARY_ERROR', json.dumps(dict(cause=cause, failing=failing, accepted=accepted, outcome=row['outcome'], code=row['code'])), flush=True)
                    self.assertEqual((row['outcome'], row['code']), (cause, cause))
                    self.assertEqual(len(sent), 1)
                    self.assertEqual(sent[0]['code'], cause)
                    self.assertNotIn('final', sent[0])
                    if cause == 'cancelled':
                        self.assertIs(accepted[0], True)

class Masks(unittest.TestCase):

    def ui(self):
        ui = object.__new__(tool.Ui)
        ui._mic = SimpleNamespace(poll=lambda: None, stop=lambda: None, running=True)
        ui._screen = mock.Mock()
        ui._screen.getch.return_value = ord('q')
        ui._draw = lambda: None
        ui._handle = lambda _: 0
        return ui

    def test_inherited_masks_and_every_restoration_signal_boundary(self):
        nums = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
        old_handlers = {n: signal.getsignal(n) for n in nums}
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        original = signal.signal
        total = 0
        try:
            for blocked_tuple in itertools.chain.from_iterable((itertools.combinations(nums, r) for r in range(4))):
                blocked = set(blocked_tuple) | {signal.SIGUSR1}
                for injection in nums:
                    for point in ['before-block', 'after-block', *[f'{when}-{int(n)}' for n in nums for when in ['before', 'after']], 'before-unmask', 'after-unmask']:
                        delivered = []
                        injected = []
                        phase = ['run']

                        def handler(n, _):
                            delivered.append((n, phase[0]))
                        for n in nums:
                            original(n, handler)
                        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)

                        def send():
                            if not injected:
                                injected.append(True)
                                signal.raise_signal(injection)
                        realmask = signal.pthread_sigmask

                        def masking(how, mask):
                            if how == signal.SIG_BLOCK:
                                phase[0] = 'restoring'
                                if point == 'before-block':
                                    send()
                                result = realmask(how, mask)
                                if point == 'after-block':
                                    send()
                                return result
                            if point == 'before-unmask':
                                send()
                            phase[0] = 'restored'
                            result = realmask(how, mask)
                            if point == 'after-unmask':
                                send()
                            return result

                        def setting(n, h):
                            if h is handler and point == f'before-{int(n)}':
                                send()
                            result = original(n, h)
                            if h is handler and point == f'after-{int(n)}':
                                send()
                            return result
                        try:
                            with mock.patch.object(signal, 'signal', side_effect=setting), mock.patch.object(signal, 'pthread_sigmask', side_effect=masking):
                                self.ui().run()
                            current = realmask(signal.SIG_BLOCK, [])
                            self.assertEqual(current, blocked, (blocked, injection, point, 'mask changed'))
                            self.assertEqual({n: signal.getsignal(n) for n in nums}, {n: handler for n in nums})
                            if injection in blocked:
                                self.assertEqual(delivered, [])
                                self.assertIn(injection, signal.sigpending())
                            elif point == 'before-block':
                                self.assertEqual(delivered, [])
                            else:
                                self.assertEqual(delivered, [(injection, 'restored')])
                            total += 1
                        finally:
                            for n in nums:
                                original(n, handler)
                            realmask(signal.SIG_SETMASK, old_mask)
            print('MASK_MATRIX', total, 'cases; all 8 inherited subsets, 3 signals, 10 restoration points', flush=True)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            for n, h in old_handlers.items():
                original(n, h)

    def test_mask_restored_if_handler_restoration_raises(self):
        nums = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
        saved = {n: signal.getsignal(n) for n in nums}
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        original = signal.signal
        try:
            for failing in nums:
                inherited = {signal.SIGHUP, signal.SIGUSR1}
                signal.pthread_sigmask(signal.SIG_SETMASK, inherited)

                def setting(n, h):
                    if n == failing and h is saved[n]:
                        raise RuntimeError('restoration failure')
                    return original(n, h)
                with mock.patch.object(signal, 'signal', side_effect=setting):
                    with self.assertRaisesRegex(RuntimeError, 'restoration failure'):
                        self.ui().run()
                self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, []), inherited)
                print('MASK_AFTER_RESTORATION_ERROR', signal.Signals(failing).name, 'exact', flush=True)
                for n, h in saved.items():
                    original(n, h)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            for n, h in saved.items():
                original(n, h)

class InterruptedMasks(unittest.TestCase):
    ui = Masks.ui

    def test_pending_interrupt_unmask_restores_exact_mask_and_handlers(self):
        nums = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
        original = signal.signal
        saved = {n: signal.getsignal(n) for n in nums}
        oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        total = 0
        try:
            for boundary in nums:
                inherited = {signal.SIGHUP, signal.SIGUSR1}
                signal.pthread_sigmask(signal.SIG_SETMASK, inherited)
                original(signal.SIGINT, signal.default_int_handler)
                before = {n: signal.getsignal(n) for n in nums}
                injected = []

                def setting(n, h):
                    if n == boundary and h is before[n] and (not injected):
                        injected.append(True)
                        signal.raise_signal(signal.SIGINT)
                    return original(n, h)
                with mock.patch.object(signal, 'signal', side_effect=setting):
                    with self.assertRaises(KeyboardInterrupt):
                        self.ui().run()
                self.assertEqual({n: signal.getsignal(n) for n in nums}, before)
                self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, []), inherited)
                total += 1
            print('PENDING_INTERRUPT_UNMASK', total, 'restoration points; exact caller mask including preblocked SIGHUP', flush=True)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
            for n, h in saved.items():
                original(n, h)
