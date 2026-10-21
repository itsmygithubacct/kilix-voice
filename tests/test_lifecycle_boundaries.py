"""Deterministic helper-level requirements using inert resources and child tokens."""
import threading,unittest
from unittest import mock
from voicelib import lifecycle,processes

class HelperWitness(unittest.TestCase):
 def test_registration_check_and_insertion_share_shutdown_lock(self):
  from tests.test_lifecycle_matrix import Case
  from voicelib import audio
  with Case(self, 'capture') as c:
   original = c.d._cleanup_lock
   class Interleave:
    fired = False
    def __enter__(self):
     original.acquire()
    def __exit__(self, *exc):
     original.release()
     if not self.fired:
      self.fired = True
      c.d._shutdown()
   c.d._cleanup_lock = Interleave()
   c.owner = audio.MicCapture()
   c.d._register_cleanup(*c.key, c.owner, method='stop')
   self.assertTrue(c.owner.lifecycle.requested,
                   'shutdown slipped between the stopping check and registry insertion')

 def test_late_publication_signals_before_handoff_unwinds(self):
  owner=lifecycle.Resource();token=object();seen=[]
  def allocate():
   owner.request_close()
   return token
  def signal(value):seen.append((value is token,owner._process is token,owner.lifecycle.starting))
  with mock.patch.object(processes,'kill_group',side_effect=signal):
   with self.assertRaisesRegex(RuntimeError,'closed during allocation'):
    owner._publish('_process',allocate)
  self.assertEqual(seen,[(True,True,1)],'late child was not signalled while publication still owned it')
  self.assertIs(owner._process,token)

 def test_timeout_cannot_skip_native_disposal_after_use_returns(self):
  class Inert(lifecycle.Resource):
   def __init__(self):self.freed=0
   @lifecycle.using
   def use(self):
    entered.set();release.wait(3)
   @lifecycle.closing
   def close(self):self.freed+=1
  entered=threading.Event();release=threading.Event();owner=Inert()
  worker=threading.Thread(target=owner.use);worker.start()
  try:
   self.assertTrue(entered.wait(2))
   with mock.patch.object(lifecycle,'CLOSE_WAIT_S',.01):owner.close()
   self.assertFalse(owner.cleanup_complete)
  finally:release.set();worker.join(3)
  owner.close()
  self.assertEqual(owner.freed,1,'deadline proof bypassed the eventual disposer')
  self.assertTrue(owner.cleanup_complete)

if __name__=='__main__':unittest.main(verbosity=2)
