"""One lifecycle conformance matrix and the round-nine requirement schedules."""

import json,threading,unittest
from types import SimpleNamespace
from unittest import mock
from tests import test_cleanup_recovery as bp
from tests.test_terminal_outcomes import voiced
from voicelib import stt
class VoskActive(unittest.TestCase):
 def test_shutdown_must_not_free_recognizer_during_feed(self):
  d,_,_=bp.Boundaries.daemon(self)
  d._speech=d._player=d._socket=None;d._socket_ino=None;d._wake_w=-1
  entered=threading.Event();release=threading.Event();frees=[]
  engine=object.__new__(stt.VoskStt)
  engine._open=True;engine._closed=False;engine._rec=123;engine._model=456
  def accept(*_):
   entered.set();release.wait(5);raise stt.SttError('fixture native call ended')
  engine._lib=SimpleNamespace(vosk_recognizer_accept_waveform=accept,vosk_recognizer_free=lambda p:frees.append(('recognizer',p,release.is_set())),vosk_model_free=lambda p:frees.append(('model',p,release.is_set())))
  turn=voiced._DictationTurn('vosk-active-feed',mock.Mock());d._dictation=turn
  d._register_cleanup(turn.id,'provider',engine)
  def feed():
   try:engine.feed(b'\x01\x00'*320)
   except stt.SttError:pass
  turn.thread=threading.Thread(target=feed,daemon=True)
  try:
   turn.thread.start();self.assertTrue(entered.wait(1))
   with mock.patch.object(voiced,'JOIN_TIMEOUT_S',.1),mock.patch.object(voiced,'CLEANUP_SHUTDOWN_WAIT_S',.1):d._shutdown()
   print('VOSK_ACTIVE_CLOSE',json.dumps(dict(frees=frees,feed_alive=turn.thread.is_alive(),owners=d._resource_owners(),warnings=[c.args[0] for c in d._warn.call_args_list])),flush=True)
   self.assertEqual(frees,[],'shutdown frees native handles before their active feed returns')
  finally:release.set();turn.thread.join(2)

import json,os,sys,threading,time,unittest
from types import SimpleNamespace
from unittest import mock
from tests import test_cleanup_recovery as bp
from tests import test_stt_binding as fixture
from tests.test_terminal_outcomes import voiced
from voicelib import processes
class VibeOwnership(unittest.TestCase):
 setUp=fixture.VibeVoiceEngineTestCase.setUp
 engine=fixture.VibeVoiceEngineTestCase.engine
 def test_first_blocked_kill_loses_shutdown_escalation_handle(self):
  d,_,_=bp.Boundaries.daemon(self)
  d._speech=d._player=d._socket=None;d._socket_ino=None;d._wake_w=-1
  engine=self.engine();turn=voiced._DictationTurn('vibe-first-kill',mock.Mock());d._dictation=turn
  entered=threading.Event();release=threading.Event();errors=[]
  def decode():
   try:engine._decode(fixture.FRAME)
   except Exception as e:errors.append(type(e).__name__)
  with mock.patch.dict(os.environ,FAKE_SLEEP='60'):
   turn.thread=threading.Thread(target=decode,daemon=True);turn.thread.start()
   deadline=time.monotonic()+3
   while engine._process is None and time.monotonic()<deadline:time.sleep(.005)
   child=engine._process;self.assertIsNotNone(child)
   kill=child.kill
   def blocked_kill():entered.set();release.wait(15);kill()
   child.kill=blocked_kill
   closer=threading.Thread(target=d._close_dictation_resources,args=(turn,None,engine),daemon=True)
   try:
    closer.start();self.assertTrue(entered.wait(3));self.assertIs(engine._process,child, "child must remain discoverable through teardown")
    with mock.patch.object(voiced,'JOIN_TIMEOUT_S',.1),mock.patch.object(voiced,'CLEANUP_SHUTDOWN_WAIT_S',.1):
     start=time.monotonic();d._shutdown();duration=time.monotonic()-start
    live=processes.poll_exit(child) is None
    print('VIBE_FIRST_KILL',json.dumps(dict(seconds=duration,child_live=live,resource_process=engine._process,retained=d._retained_dictation_cleanup(),warnings=[c.args[0] for c in d._warn.call_args_list],decode_alive=turn.thread.is_alive(),cleanup_alive=closer.is_alive())),flush=True)
    self.assertFalse(live,'VibeVoice drops its only discoverable process before first kill; shutdown cannot escalate it')
   finally:
    release.set();closer.join(3);turn.thread.join(3);kill();child.wait(3)
 def test_close_is_not_proof_while_decode_has_not_reaped(self):
  d,_,_=bp.Boundaries.daemon(self)
  engine=self.engine();turn=SimpleNamespace(id='vibe-busy')
  entered=threading.Event();release=threading.Event();errors=[]
  # Hold the decoder immediately before its already-owned child communicates.
  original=fixture.stt.subprocess.Popen.communicate
  def communicate(child,*a,**kw):
   entered.set();release.wait(10);return original(child,*a,**kw)
  def decode():
   try:engine._decode(fixture.FRAME)
   except Exception as e:errors.append(type(e).__name__)
  with mock.patch.dict(os.environ,FAKE_SLEEP='60'),mock.patch.object(fixture.stt.subprocess.Popen,'communicate',communicate):
   worker=threading.Thread(target=decode,daemon=True);worker.start();self.assertTrue(entered.wait(3));child=engine._process
   try:
    error=d._close_dictation_resources(turn,None,engine)
    print('VIBE_FALSE_PROOF',json.dumps(dict(error=repr(error),retained=d._retained_dictation_cleanup(),complete=engine.cleanup_complete,decoder_alive=worker.is_alive(),child_returncode=child.returncode)),flush=True)
    self.assertTrue(d._retained_dictation_cleanup(),'busy decoder handoff was counted as complete before reap')
   finally:
    release.set();worker.join(3);child.kill();child.wait(3)

import json,sys,threading,time,unittest
from types import SimpleNamespace
from unittest import mock
from tests import test_cleanup_recovery as recovery
from tests.test_terminal_outcomes import voiced
from voicelib import audio,processes
class StartupRace(unittest.TestCase):
 def test_capture_cannot_start_after_shutdown_has_removed_its_owner(self):
  d,_,ledger=recovery.Boundaries.daemon(self)
  d._speech=d._player=d._socket=None;d._socket_ino=None;d._wake_w=-1
  capture=audio.MicCapture({'audio':{'capture_cmd':[sys.executable,'-c','import time;time.sleep(60)']}})
  turn=voiced._DictationTurn('startup-race',mock.Mock());turn.on_settle=ledger.record;d._dictation=turn
  entered=threading.Event();launch=threading.Event();recorded=threading.Event();finish=threading.Event()
  build=audio.build_capture_cmd
  def gated(cfg):
   entered.set();self.assertTrue(launch.wait(15));return build(cfg)
  def record(*args):
   recorded.set();self.assertTrue(finish.wait(15));turn.stop.check()
  d._record=record;d._dictate=lambda t:d._record_turn(t,SimpleNamespace(),None)
  engine=mock.Mock()
  turn.thread=threading.Thread(target=d._run_dictation,args=(turn,),daemon=True)
  shutdown=threading.Thread(target=d._shutdown,daemon=True)
  try:
   with mock.patch.object(voiced.audio,'MicCapture',return_value=capture),mock.patch.object(voiced.stt_lib,'make_stt',return_value=engine),mock.patch.object(audio,'build_capture_cmd',side_effect=gated):
    turn.thread.start();self.assertTrue(entered.wait(3))
    before=d._resource_owners();shutdown.start()
    shutdown.join(8);self.assertFalse(shutdown.is_alive())
    removed=d._resource_owners()
    self.assertTrue(any(r["kind"]=="capture" for r in removed), "startup is still able to allocate; ownership must remain")
    self.assertIn("cleanup unproven",str(d._warn.call_args_list))
    launch.set();turn.thread.join(4);child=capture._process
    live=child is not None and processes.poll_exit(child) is None
    print('STARTUP_RACE',json.dumps(dict(before=before,after_prelaunch_cleanup=removed,after_shutdown=d._resource_owners(),recorder_pid=getattr(child,"pid",None),recorder_live=live,worker_alive=turn.thread.is_alive(),warnings=[c.args[0] for c in d._warn.call_args_list],arbiter_released=d._arbiter.release.call_count)),flush=True)
    self.assertFalse(live,'shutdown proved an empty recorder before start returned, removed it, then missed its live child')
  finally:
   launch.set();finish.set();turn.thread.join(5);shutdown.join(8);capture.stop();d._retry_dictation_cleanup(1)
