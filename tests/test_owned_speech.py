"""Owner isolation and heartbeat expiry use the existing job cancellation path."""
import threading
import time
import unittest
from types import SimpleNamespace
from voicelib.owned import OwnedTurns
from voicelib import protocol

class Harness(OwnedTurns):
    def __init__(self):
        self._lock=threading.RLock()
        self._speech=self._dictation=None
        self._arbiter=SimpleNamespace(listening=False,speaking=False)
        self._player=SimpleNamespace(playing=False)
        self.starts=0
    def _op_speak(self, request):
        self.starts+=1
        self._speech=SimpleNamespace(id=f'speak-{self.starts}',outcome=None)
        self._arbiter.speaking=True
        return {'ok':True}
    def _cancel_speech(self):
        self._speech.outcome=SimpleNamespace(outcome='cancelled',message='')
        self._speech=None
        self._arbiter.speaking=False

class OwnedSpeechTests(unittest.TestCase):
    def request(self, **changes):
        request=dict(op='owned-speak',owner='a'*32,utterance='b'*32,text='Hello',id=1)
        request.update(changes)
        return protocol.validate_request(request,"/tmp")
    def test_retry_is_idempotent_and_different_text_refused(self):
        d=Harness(); first=d._op_owned(self.request())
        self.assertTrue(first['ok'])
        from tests.test_control_codes import voiced
        self.assertTrue(voiced._guard_reply(first)['ok'])
        self.assertIn('playing',first)
        self.assertEqual(d._op_owned(self.request())['turn'],first['turn'])
        self.assertEqual(d.starts,1)
        self.assertFalse(d._op_owned(self.request(text='Different'))['ok'])
    def test_other_owner_cannot_inspect_or_stop(self):
        d=Harness(); turn=d._op_owned(self.request())['turn']
        self.assertEqual(d._op_owned(self.request(op='owned-status',owner='c'*32))['state'],'unknown')
        self.assertFalse(d._op_owned(self.request(op='owned-stop',owner='c'*32,turn=turn))['stopped'])
        self.assertFalse(d._op_owned(self.request(owner='c'*32))['ok'])
        self.assertIsNotNone(d._speech)
    def test_lease_cancels_and_terminal_status_survives(self):
        d=Harness();d._op_owned(self.request())
        d._speech.lease_until=time.monotonic()-1
        d._owned_expire()
        self.assertIsNone(d._speech)
        self.assertEqual(d._op_owned(self.request(op='owned-status'))['state'],'cancelled')
    def test_heartbeat_renews_and_wrong_turn_does_not_cancel(self):
        d=Harness();d._op_owned(self.request())
        d._speech.lease_until=0
        d._op_owned(self.request(op='owned-status'))
        d._owned_expire()
        self.assertIsNotNone(d._speech)
        self.assertFalse(d._op_owned(self.request(op='owned-stop',turn='speak-999'))['stopped'])
        self.assertTrue(d._op_owned(self.request(op='owned-stop',turn=d._speech.id))['stopped'])
    def test_tokens_and_limits(self):
        for changes in ({'owner':'bad'},{'utterance':'bad'},{'text':'x'*4097}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(protocol.ProtocolError): self.request(**changes)

class RecommendTests(unittest.TestCase):
    def test_recommend_is_read_only_and_preserves_tier_json(self):
        import contextlib
        import io
        from unittest.mock import patch
        from tests.test_tts_tool import load_tool
        tool=load_tool()
        with patch.object(tool.sizing,'recommend',return_value={'fixture':True}) as recommend, \
             patch.object(tool.sizing,'print_report') as output, \
             patch.object(tool.tts_lib,'piper_status',return_value=(True,'')), \
             patch.object(tool.tts_lib,'espeak_binary',return_value='/fixture/espeak'):
            self.assertEqual(tool.main(['--recommend','--json']),0)
            self.assertEqual(recommend.call_args.args[0],'tts')
            self.assertTrue(recommend.call_args.args[1]['piper-en-us-kristin-medium'])
            output.assert_called_once_with({'fixture':True},as_json=True)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main(['--recommend','--speak','hello'])
