"""Every daemon resource runs the same lifecycle schedules, using real owners.

Only native Vosk entry points and external executables are substituted. All
starts, publications, use guards, closes, daemon registry operations, signals
and reaps run production code. Gates are events, not sleeps or scheduler luck.
The table is deliberately independent of the production kind registry.
"""
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from tests import test_cleanup_recovery as recovery
from tests.test_terminal_outcomes import voiced, wait_until
from voicelib import audio, lifecycle, models, processes, qwen_provider, stt, tts

SETUP = 15
FRAME = b'\x01\x00' * 320
# Field occurrence numbers distinguish each held model descriptor.
MATRIX = {
    'capture': (audio.MicCapture, ('_stderr', '_process', '_reader')),
    'vosk': (stt.VoskStt, ('_model', '_rec')),
    'whisper': (stt.WhisperStt, ('_held',) * len(models.REQUIRED_FILES['whisper']) + ('_links', '_process')),
    'whistle': (stt.WhistleStt, ('_held',) * len(models.REQUIRED_FILES['whistle']) + ('_links', '_process')),
    'vibevoice': (stt.VibeVoiceStt, ('_held',) * len(models.REQUIRED_FILES['vibevoice']) + ('_wav', '_process')),
    'null-stt': (stt.NullStt, ()),
    'espeak': (tts.EspeakTts, ('_process',)),
    'mbrola': (tts.EspeakTts, ('_process',)),
    'piper': (tts.PiperTts, ('_process',)),
    'qwen': (qwen_provider.QwenProviderTts, ('_process',)),
    'piper-status': (processes.SynthesisOwner, ('_process',)),
    'null-tts': (tts.NullTts, ()),
    'player': (audio.Player, ('_worker', '_sink_stderr', '_process')),
    'sink': (audio.Player, ('_worker', '_sink_stderr', '_process')),
}
CONSTRUCTORS = {'vosk', 'whisper', 'whistle', 'vibevoice'}
NO_ALLOCATION = {'null-stt', 'null-tts'}
NO_CHILD = NO_ALLOCATION | {'vosk'}
# These are executable skip reasons, also consumed by the evidence report.
NOT_APPLICABLE = {
    'null-stt': 'No allocating startup or constructor, native handle, child, or handoff; only Python turn state.',
    'null-tts': 'No allocating startup or constructor, native handle, child, or handoff; returns an empty clip.',
    'vosk': 'No child/reap handoff; active native calls are covered by the use-count scenario.',
    'constructor': 'Constructor only validates/configures Python state; allocation failure is exercised at every later startup publication.',
}


class Case:
    def __init__(self, test, kind):
        self.test, self.kind = test, kind
        self.cls, self.points = MATRIX[kind]
        self.stack = contextlib.ExitStack()
        self.owner = None
        self.children = []
        self.errors = []
        self.threads = []
        self.gates = []
        self.d, _, _ = recovery.Boundaries.daemon(test)
        self.d._speech = self.d._dictation = self.d._player = self.d._socket = None
        self.d._socket_ino = None
        self.d._wake_w = -1
        self.category = ('capture' if kind == 'capture' else
                         'player' if kind in ('player', 'sink') else
                         'provider' if kind in CONSTRUCTORS | {'null-stt'} else 'synthesiser')
        self.key = ('matrix-' + kind, self.category)

    def patch(self, *args, **kwargs):
        return self.stack.enter_context(mock.patch.object(*args, **kwargs))

    def __enter__(self):
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(mock.patch.dict(os.environ, {
            'KILIX_DATA_HOME': str(self.root/'data'),
            'KILIX_SESSION_HOME': str(self.root/'session'),
        }))
        self.stack.enter_context(mock.patch.object(lifecycle, 'CLOSE_WAIT_S', .03))
        self.stack.enter_context(mock.patch.object(voiced, 'JOIN_TIMEOUT_S', .03))
        self.stack.enter_context(mock.patch.object(voiced, 'CLEANUP_SHUTDOWN_WAIT_S', .03))
        self.binary = self.root/'fixture-child'
        self.binary.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(60)\n')
        self.binary.chmod(0o700)
        self.command = [str(self.binary)]
        if self.kind in CONSTRUCTORS:
            self.model = self.root/'model'
            self.model.mkdir()
            self.patch(stt, '_resolve_model', return_value=str(self.model))
            if self.kind == 'vosk':
                self.frees = []
                self.allocations = []
                self.lib = SimpleNamespace(
                    vosk_set_log_level=lambda *_: None,
                    vosk_model_new=lambda *_: (self.allocations.append(('model',456)),456)[1],
                    vosk_recognizer_new=lambda *_: (self.allocations.append(('recognizer',123)),123)[1],
                    vosk_model_free=lambda p: self.frees.append(('model', p)),
                    vosk_recognizer_free=lambda p: self.frees.append(('recognizer', p)),
                    vosk_recognizer_accept_waveform=lambda *_: 0,
                    vosk_recognizer_partial_result=lambda *_: b'{"partial":"fixture"}',
                    vosk_recognizer_final_result=lambda *_: b'{"text":"fixture"}')
                self.patch(stt, '_resolve_library', return_value='/inert-library')
                self.patch(stt, '_load_library', return_value=self.lib)
            else:
                for name in models.REQUIRED_FILES[self.cls.name]:
                    (self.model/name).write_bytes(b'fixture weights')
                if self.kind == 'vibevoice':
                    self.patch(stt, 'vibevoice_missing', return_value=[])
                    self.patch(stt, 'vibeasr_binary', return_value=str(self.binary))
                else:
                    self.patch(self.cls, '_missing', return_value=[])
                    self.patch(self.cls, '_binary_path', return_value=str(self.binary))
                    self.patch(self.cls, '_command', return_value=self.command)
        self.patch(tts, 'piper_binary', return_value=str(self.binary))
        self.patch(qwen_provider, 'provider_binary', return_value=str(self.binary))
        self.patch(tts, 'resolve_mbrola_voice', return_value='mb-us1')
        publish = lifecycle.Resource._publish
        def tracked(owner, field, allocate):
            def allocating():
                value = allocate()
                if field == '_process':
                    self.children.append(value)
                return value
            return publish(owner, field, allocating)
        self.patch(lifecycle.Resource, '_publish', tracked)
        return self

    def register(self, owner, callback=None):
        self.owner = owner
        if callback:
            callback(owner)
        else:
            self.d._register_cleanup(*self.key, owner, method=owner.cleanup_method)

    def create(self, callback=None):
        if self.kind in CONSTRUCTORS:
            owner = self.cls(model_path=str(self.model), on_owner=lambda r: self.register(r, callback))
        elif self.kind == 'capture':
            owner = self.cls({'audio': {'capture_cmd': self.command}})
        elif self.kind in ('player', 'sink'):
            owner = self.cls({'audio': {'play_cmd': self.command}})
        elif self.kind in ('espeak', 'mbrola'):
            owner = self.cls({'tts': {'cmd': self.command}}, voice='en-us', rate=170,
                             mbrola=self.kind == 'mbrola', mbrola_fallback=False)
        elif self.kind == 'piper':
            owner = self.cls(rate=170)
        else:
            owner = self.cls()
        if self.owner is None:
            self.register(owner, callback)
        return owner

    def activate(self, budget=.2):
        r = self.owner
        if self.kind == 'capture':
            r.start()
        elif self.kind in ('player', 'sink'):
            r.play(FRAME, 16000)
        elif self.kind == 'piper-status':
            processes.run_owned(self.command, timeout=budget, owner=r)
        elif self.kind in ('espeak', 'mbrola', 'piper', 'qwen', 'null-tts'):
            r.synth('fixture', budget=budget)
        elif self.kind not in ('whisper', 'whistle', 'vosk', 'null-stt'):
            r._decode(FRAME)

    def run_start(self):
        self.create()
        self.activate()

    def run_use(self, native='feed'):
        r = self.owner
        if self.kind == 'capture':
            r.read(timeout=.2)
        elif self.kind in ('vosk', 'null-stt'):
            r.start_utterance()
            if native == 'end':
                r.end_utterance()
            else:
                r.feed(FRAME)
        elif self.kind in ('whisper', 'whistle', 'vibevoice'):
            r._decode(FRAME)
        else:
            self.activate()

    def worker(self, fn):
        def run():
            try:
                fn()
            except BaseException as error:
                self.errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        self.threads.append(worker)
        worker.start()
        return worker

    def gate(self):
        entered, release = threading.Event(), threading.Event()
        self.gates.append(release)
        def block():
            entered.set()
            if not release.wait(SETUP * 2):
                raise AssertionError('fixture gate was never released')
        return entered, release, block

    def close_registered(self):
        r = self.owner
        key, state = self.d._register_cleanup(*self.key, r, method=r.cleanup_method)
        self.d._prepare_cleanup(key, r, state)
        self.d._retry_dictation_cleanup(.15)

    def drain(self, contain=False):
        for release in self.gates:
            release.set()
        for thread in self.threads:
            if contain:
                deadline = time.monotonic() + SETUP
                while thread.is_alive() and time.monotonic() < deadline:
                    # A deliberately broken starter may publish after the
                    # failed assertion. Contain only these recorded children.
                    for child in self.children:
                        processes.kill_group(child)
                    thread.join(.01)
            else:
                thread.join(SETUP)
            self.test.assertFalse(thread.is_alive(), 'fixture worker did not finish')
        if self.owner:
            self.test.assertTrue(wait_until(
                lambda: not self.owner.lifecycle.starting and not self.owner.lifecycle.users
                and not self.owner.lifecycle.disposing, SETUP), 'resource operation did not finish')
            child = getattr(self.owner, '_process', None)
            if child is not None:
                self.children.append(child)
            getattr(self.owner, self.owner.cleanup_method)()
            self.d._retry_dictation_cleanup(2)
        try:
            if self.kind == 'vosk':
                self.test.assertEqual(sorted(self.frees), sorted(self.allocations), 'native handle dropped without free')
            for child in self.children:
                self.test.assertIsInstance(child, processes.GroupPopen, 'child bypassed the group owner')
                self.test.assertIsNotNone(child.returncode, 'close dropped an unreaped child')
                self.test.assertTrue(getattr(child, '_voice_group_signalled', False), 'reaped before final group signal')
            self.test.assertTrue(all(isinstance(e, (OSError, RuntimeError)) for e in self.errors), self.errors)
        finally:
            for child in self.children:
                processes.kill_group(child)
                child.wait(timeout=3)
        self.test.assertFalse(self.d._resource_owners(), 'proven cleanup was not retired')

    def __exit__(self, *exc):
        if exc[0] is not None:
            for child in self.children:
                processes.kill_group(child)
        try:
            self.drain(contain=exc[0] is not None)
        finally:
            self.stack.close()


class Conformance(unittest.TestCase):
    def test_registry_is_exhaustive(self):
        self.assertEqual(set(MATRIX), set(lifecycle.RESOURCE_KINDS),
                         'every registered resource kind must have a matrix row')
        for kind, (cls, _) in MATRIX.items():
            self.assertIs(lifecycle.RESOURCE_KINDS[kind], cls)
            self.assertTrue(issubclass(cls, lifecycle.Resource))

    def test_synthesis_cancel_during_allocator_keeps_cancelled_outcome(self):
        for kind in ('espeak', 'mbrola', 'piper', 'qwen'):
            with self.subTest(kind=kind), Case(self, kind) as c:
                engine = c.create()
                c.d._speech_error = ''
                c.d._speech_error_serial = 0
                c.d._get_player = lambda: mock.Mock(error=None)
                turn = voiced._SpeechTurn(c.key[0], ['fixture'], engine)
                c.d._speech = turn
                publish = lifecycle.Resource._publish
                def cancel_allocation(owner, field, allocate):
                    def allocating():
                        child = allocate()
                        turn.cancel.set()
                        owner.close()
                        return child
                    return publish(owner, field, allocating if field == '_process' else allocate)
                c.patch(lifecycle.Resource, '_publish', cancel_allocation)
                c.d._run_speech(turn)
                self.assertEqual(turn.outcome.outcome, 'cancelled',
                                 'allocation cancellation was recoded as synthesis failure')

    def publication(self, kind):
        if kind in NO_ALLOCATION:
            self.skipTest(NOT_APPLICABLE[kind])
        for index, field in enumerate(MATRIX[kind][1]):
            for phase in ('before', 'allocating', 'after'):
                with self.subTest(field=field, occurrence=index, phase=phase), Case(self, kind) as c:
                    entered, release, block = c.gate()
                    publish = lifecycle.Resource._publish
                    observed = []
                    resumed = []
                    def gated(owner, attr, allocate):
                        occurrence = len(observed)
                        observed.append(attr)
                        hit = occurrence == index
                        if hit and phase == 'before':
                            block()
                        factory = allocate
                        if hit and phase == 'allocating':
                            def factory():
                                block()
                                return allocate()
                        result = publish(owner, attr, factory)
                        if hit and phase == 'allocating' and owner.lifecycle.requested:
                            resumed.append(attr)
                            # The assertion below records the illegal handoff;
                            # contain this late fixture child by its own group.
                            if attr == '_process':
                                processes.kill_group(result)
                        if hit and phase == 'after':
                            block()
                        return result
                    c.patch(lifecycle.Resource, '_publish', gated)
                    c.worker(c.run_start)
                    self.assertTrue(entered.wait(SETUP), (kind, index, observed))
                    r = c.owner
                    self.assertIsNotNone(r)
                    self.assertGreater(r.lifecycle.starting + r.lifecycle.users, 0, 'startup is not counted')
                    c.d._shutdown()
                    self.assertFalse(r.cleanup_complete, 'close proved a starter that can still allocate')
                    self.assertIn(c.key, c.d._cleanup_owners)
                    self.assertIn('cleanup unproven', str(c.d._warn.call_args_list))
                    release.set()
                    c.drain()
                    self.assertEqual(resumed, [], "cancelled allocation returned usable work")
                    self.assertTrue(r.cleanup_complete)
                    self.assertEqual(observed[:index+1], list(c.points[:index+1]))

    def active(self, kind):
        for native in (('feed', 'end') if kind == 'vosk' else ('feed',)):
            with self.subTest(call=native), Case(self, kind) as c:
                r = c.create()
                if kind == 'capture':
                    r.start()
                entered, release, block = c.gate()
                if kind == 'capture':
                    original = r._queue.get
                    c.patch(r._queue, 'get', side_effect=lambda *a, **k: (block(), original(*a, **k))[1])
                elif kind == 'vosk':
                    target = 'vosk_recognizer_accept_waveform' if native == 'feed' else 'vosk_recognizer_final_result'
                    original = getattr(c.lib, target)
                    c.patch(c.lib, target, side_effect=lambda *a: (block(), original(*a))[1])
                elif kind == 'null-stt':
                    original = stt._frame_bytes
                    c.patch(stt, '_frame_bytes', side_effect=lambda *a: (block(), original(*a))[1])
                elif kind == 'null-tts':
                    original = tts.SynthesisProvenance
                    c.patch(tts, 'SynthesisProvenance', side_effect=lambda *a, **k: (block(), original(*a, **k))[1])
                elif kind in ('whisper', 'whistle'):
                    c.patch(r, '_read_reply', side_effect=lambda *a: (block(), {'ready': True, 'text': ''})[1])
                elif kind == 'qwen':
                    original = r._exchange
                    c.patch(r, '_exchange', side_effect=lambda *a: (block(), original(*a))[1])
                else:
                    target = 'wait' if kind in ('player', 'sink') else 'communicate'
                    original = getattr(processes.GroupPopen, target)
                    def gated(child, *a, **k):
                        if child not in c.children:
                            c.children.append(child)
                        block()
                        return original(child, *a, **k)
                    c.patch(processes.GroupPopen, target, gated)
                c.worker(lambda: c.run_use(native))
                self.assertTrue(entered.wait(SETUP), c.errors)
                child = getattr(r, '_process', None)
                held = dict(getattr(r, '_held', {}))
                self.assertGreater(r.lifecycle.users, 0, 'active use is not counted')
                c.close_registered()
                self.assertEqual(getattr(r, '_held', {}), held, 'freed model files still in use')
                self.assertFalse(r.cleanup_complete, 'a busy handoff was counted as proof')
                self.assertIn(c.key, c.d._cleanup_owners)
                if child:
                    self.assertIs(r._process, child, 'lost escalation handle before reaping')
                    self.assertIsNone(child.returncode, 'active owner has not reaped')
                if kind == 'vosk':
                    self.assertEqual(c.frees, [], 'native handle freed during active call')
                began = time.monotonic()
                c.d._shutdown()
                self.assertLess(time.monotonic()-began, .5)
                self.assertIn('cleanup unproven', str(c.d._warn.call_args_list))
                self.assertFalse(r.cleanup_complete)
                release.set()
                c.drain()
                self.assertTrue(r.cleanup_complete)

    def blocked_close(self, kind):
        with Case(self, kind) as c:
            r = c.create()
            if kind == 'capture':
                r.start()
            elif kind not in NO_CHILD | {'whisper', 'whistle'}:
                c.worker(lambda: c.activate(budget=60))
                self.assertTrue(wait_until(lambda: getattr(r, '_process', None) is not None, SETUP), c.errors)
            child = getattr(r, '_process', None)
            entered, release, block = c.gate()
            calls = []
            original = r._on_close_requested
            def close_hook():
                calls.append(threading.get_ident())
                block()
                original()
            c.patch(r, '_on_close_requested', close_hook)
            c.worker(lambda: getattr(r, r.cleanup_method)())
            self.assertTrue(entered.wait(SETUP))
            if child is not None:
                self.assertIs(r._process, child, "first close lost its escalation handle")
            for _ in range(3):
                c.worker(lambda: getattr(r, r.cleanup_method)())
            c.d._shutdown()
            self.assertEqual(len(calls), 1, 'more than one disposal is in flight')
            self.assertIn(c.key, c.d._cleanup_owners)
            self.assertIn('cleanup unproven', str(c.d._warn.call_args_list))
            if child is not None:
                self.assertIsInstance(child, processes.GroupPopen)
                self.assertTrue(wait_until(lambda: processes.poll_exit(child) is not None, .5))
                self.assertTrue(child._voice_group_signalled, 'shutdown did not seal the owned group')
            release.set()
            c.drain()
            self.assertTrue(r.cleanup_complete)

    def constructor(self, kind):
        if kind not in CONSTRUCTORS:
            self.skipTest(NOT_APPLICABLE.get(kind, NOT_APPLICABLE['constructor']))
        # Fail at every constructor publication. Gate its own close before
        # entering the shared helper to witness duplicate public close calls.
        points = MATRIX[kind][1]
        if kind == 'vibevoice':
            points = points[:-2]  # WAV and child start at decode, after construction
        for index in range(len(points)):
            with self.subTest(publication=index), Case(self, kind) as c:
                entered, release, block = c.gate()
                calls = []
                publish = lifecycle.Resource._publish
                count = 0
                def fail(owner, field, allocate):
                    nonlocal count
                    value = publish(owner, field, allocate)
                    count += 1
                    if count == index+1:
                        raise OSError('constructor failed after publication')
                    return value
                close = c.cls.close
                def closing(owner):
                    calls.append(threading.get_ident())
                    if len(calls) == 1:
                        block()
                    return close(owner)
                c.patch(lifecycle.Resource, '_publish', fail)
                c.patch(c.cls, 'close', closing)
                c.patch(voiced.stt_lib, 'make_stt', side_effect=lambda *a, **kw: c.create(kw.get('on_owner')))
                c.patch(voiced.audio, 'MicCapture', return_value=mock.Mock(rate=16000))
                turn = voiced._DictationTurn(c.key[0], mock.Mock())
                c.d._dictation = turn
                c.d._dictate = lambda t: c.d._record_turn(t, SimpleNamespace(), None)
                turn.thread = c.worker(lambda: c.d._run_dictation(turn))
                self.assertTrue(entered.wait(SETUP), c.errors)
                c.d._shutdown()
                self.assertEqual(len(calls), 1, 'shutdown entered a constructor cleanup already in flight')
                self.assertIn(c.key, c.d._cleanup_owners)
                self.assertIn('cleanup unproven', str(c.d._warn.call_args_list))
                release.set()
                c.drain()
                self.assertTrue(c.owner.cleanup_complete)


# unittest reports every kind x scenario cell by name, including explicit N/A.
for _kind in MATRIX:
    for _scenario in ('publication', 'active', 'blocked_close', 'constructor'):
        def _test(self, kind=_kind, scenario=_scenario):
            return getattr(self, scenario)(kind)
        setattr(Conformance, 'test_' + _kind.replace('-', '_') + '_' + _scenario, _test)
