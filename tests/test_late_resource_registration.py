"""Constructor pre-registration work resumes after daemon shutdown.

Only the tree's synthetic child is launched. No PID reuse or signal/reap
race is manufactured: gates control resource construction and registration.
"""
import json,threading,time,unittest
from types import SimpleNamespace
from unittest import mock
from tests.test_lifecycle_matrix import Case, SETUP
from tests.test_terminal_outcomes import voiced,wait_until
from voicelib import audio,processes,stt
from voicelib.lifecycle import RESOURCE_KINDS
from tests import test_cleanup_recovery as recovery


class LateAdmissionContract(unittest.TestCase):
 def test_every_registered_resource_kind_refuses_allocation_while_stopping(self):
  daemon,_,_=recovery.Boundaries.daemon(self)
  daemon._stopping=threading.Event()
  daemon._stopping.set()
  for kind,resource_type in RESOURCE_KINDS.items():
   with self.subTest(kind=kind):
    resource=object.__new__(resource_type)
    key,state=daemon._register_cleanup('late-registration',kind,resource)
    self.assertTrue(resource.lifecycle.requested)
    self.assertTrue(state.pending)
    self.assertIs(daemon._failed_dictation_cleanup[key],resource)
    allocate=mock.Mock()
    with self.assertRaises(resource.lifecycle_error):
     resource._publish('_fixture',allocate)
    allocate.assert_not_called()

class LateRegistration:
 def test_shutdown_refuses_provider_registered_after_shutdown_pass(self):
  with Case(self,self.kind) as c:
   pre,proceed,preblock=c.gate()
   launched,finish,postblock=c.gate()
   def resolve_model(*args,**kwargs):preblock();return str(c.model)
   original=c.cls._start
   def start(owner):
    original(owner)
    postblock()
   c.patch(stt,'_resolve_model',side_effect=resolve_model)
   c.patch(c.cls,'_start',start)
   c.patch(voiced.stt_lib,'make_stt',side_effect=lambda *a,**kw:c.create(kw.get('on_owner')))
   reader_entered,reader_release,reader_block=c.gate()
   original_read=audio.MicCapture._read_stream
   def reader(owner,*args):reader_block();return original_read(owner,*args)
   c.patch(audio.MicCapture,'_read_stream',reader)
   capture=audio.MicCapture({'audio':{'capture_cmd':c.command}})
   c.patch(voiced.audio,'MicCapture',return_value=capture)
   turn=voiced._DictationTurn(c.key[0],mock.Mock());c.d._dictation=turn
   turn.owner='fixture-owner';turn.owned_finish=threading.Event()
   c.d._dictate=lambda t:c.d._record_turn(t,SimpleNamespace(),None)
   turn.thread=c.worker(lambda:c.d._run_dictation(turn))
   self.assertTrue(pre.wait(SETUP),c.errors)
   self.assertTrue(reader_entered.wait(SETUP),c.errors)
   before=[]
   def release_session():
    # _shutdown has finished escalation, retries and retained-owner logging.
    # Filesystem/session release can yield while a pre-owner constructor returns.
    before.extend(c.d._resource_owners())
    proceed.set()
    self.assertTrue(wait_until(lambda:launched.is_set() or not turn.thread.is_alive(),SETUP),c.errors)
   c.d._arbiter.release.side_effect=release_session
   began=time.monotonic()
   c.d._shutdown()
   shutdown_seconds=time.monotonic()-began
   self.assertTrue(before)
   child=getattr(c.owner,"_process",None)
   observation=dict(kind=c.kind,shutdown_returned=True,shutdown_seconds=shutdown_seconds,owners_at_final_cleanup_pass=before,
      owners_at_shutdown_return=c.d._resource_owners(),
      child_pid=getattr(child,"pid",None),child_live=child is not None and processes.poll_exit(child) is None,
      group_signalled=getattr(child,'_voice_group_signalled',False),
      resource_close_requested=c.owner.lifecycle.requested,
      constructor_active=turn.thread.is_alive(),capture_complete=capture.cleanup_complete,warnings=str(c.d._warn.call_args_list))
   print('LATE_REGISTRATION',json.dumps(observation),flush=True)
   try:
    self.assertFalse(observation['child_live'],
      'provider allocated a live child after shutdown finished its only escalation pass')
   finally:
    reader_release.set()
    finish.set()

class WhisperLateRegistration(LateRegistration, unittest.TestCase):
 kind = 'whisper'

class WhistleLateRegistration(LateRegistration, unittest.TestCase):
 kind = 'whistle'

if __name__=='__main__':unittest.main(verbosity=2)
