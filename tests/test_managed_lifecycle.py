"""Independent requirements; synthetic capture, private leases, no real devices."""
import json, os, sys, tempfile, threading, time, unittest
from pathlib import Path
from unittest import mock
from voicelib import audio, lifecycle, microphone
from voicelib.managed_audio import ManagedMicCapture
from tests.test_lifecycle_matrix import MATRIX
from tests import test_cleanup_recovery as recovery
class ManagedRequirements(unittest.TestCase):
 def make(self,root):
  return audio.MicCapture({'audio':{'capture_cmd':[sys.executable,'-I','-c','import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(20)',str(Path(root)/'recorder.pid')]}},control=microphone.CaptureControl(time.monotonic()+15,locked=lambda:False,namespace=str(Path(root)/'leases')))
 def test_managed_kind_has_independent_matrix_row(self):
  print('REGISTRY',json.dumps({k:v.__name__ for k,v in lifecycle.RESOURCE_KINDS.items()}), 'MATRIX',[(k,c.__name__) for k,(c,_) in MATRIX.items()],flush=True)
  self.assertIn(ManagedMicCapture,lifecycle.RESOURCE_KINDS.values())
  self.assertIn(ManagedMicCapture,[c for c,_ in MATRIX.values()])
 def test_stopping_registry_refuses_managed_start(self):
  with tempfile.TemporaryDirectory() as root:
   capture=self.make(root)
   daemon,_,_=recovery.Boundaries.daemon(self)
   daemon._stopping=threading.Event();daemon._stopping.set()
   key,state=daemon._register_cleanup('managed-late','capture',capture,method='stop')
   print('REGISTERED',capture.lifecycle.requested,'cleanup_complete',capture.cleanup_complete,'pending',state.pending,flush=True)
   try:
    with self.assertRaises(audio.AudioError):
     capture.start()
     deadline=time.monotonic()+3
     while not (Path(root)/'recorder.pid').exists():
      self.assertLess(time.monotonic(),deadline);time.sleep(.01)
   finally:
    process=capture._process
    print('AFTER_START',{'requested':capture.lifecycle.requested,'pid':getattr(process,'pid',None),'live':process is not None and process.poll() is None,'recorder':(Path(root)/'recorder.pid').read_text() if (Path(root)/'recorder.pid').exists() else None},flush=True)
    self.assertTrue(capture.stop())
 def test_stop_during_start_cannot_prove_cleanup_or_drop_late_child(self):
  with tempfile.TemporaryDirectory() as root:
   capture=self.make(root);entered=threading.Event();release=threading.Event();errors=[]
   original=microphone.CaptureOwner.start
   def delayed(owner,*args):
    entered.set()
    if not release.wait(5):raise RuntimeError('setup gate timed out')
    return original(owner,*args)
   def start():
    try:capture.start()
    except BaseException as error:errors.append(repr(error))
   with mock.patch.object(microphone.CaptureOwner,'start',delayed):
    worker=threading.Thread(target=start)
    worker.start()
    try:
     self.assertTrue(entered.wait(3))
     result=capture.stop()
     print('CLOSE_WHILE_STARTING',dict(result=result,proof=capture.cleanup_complete,starting=capture.lifecycle.starting),flush=True)
     release.set();worker.join(5);self.assertFalse(worker.is_alive())
     process=capture._process
     print('AFTER_RELEASE',dict(pid=getattr(process,'pid',None),live=process is not None and process.poll() is None,errors=errors),flush=True)
     self.assertFalse(result,'close cannot prove cleanup while managed allocation is active')
     self.assertTrue(process is None or process.poll() is not None,'late child must be refused or signalled')
    finally:
     release.set();worker.join(6);capture.stop()


"""Inherited stop API on explicit managed capture; synthetic child only.

No namespace, tracing, subreaper setup, or PID reuse. A pidfd contains the
exact synthetic recorder if product cancellation loses its supervisor.
"""
import json,os,select,signal,sys,tempfile,time,unittest
from pathlib import Path
from voicelib import audio,microphone
class RequestStop(unittest.TestCase):
 def test_request_stop_retires_managed_recorder(self):
  with tempfile.TemporaryDirectory() as root:
   marker=Path(root)/'recorder.pid';fd=None
   code='import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(15)'
   capture=audio.MicCapture({'audio':{'capture_cmd':[sys.executable,'-I','-c',code,str(marker)]}},control=microphone.CaptureControl(time.monotonic()+10,locked=lambda:False,namespace=str(Path(root)/'leases')))
   try:
    capture.start();deadline=time.monotonic()+3
    while not marker.exists():
     self.assertLess(time.monotonic(),deadline);time.sleep(.01)
    recorder=int(marker.read_text());fd=os.pidfd_open(recorder)
    capture.request_stop()
    self.assertTrue(capture._owner.wait(4),'supervisor watcher did not finish')
    result=capture.stop()
    poller=select.poll();poller.register(fd,select.POLLIN)
    stopped=bool(poller.poll(100))
    print('MANAGED_REQUEST_STOP',json.dumps(dict(supervisor=capture._owner.process.pid,supervisor_status=capture._owner.process.returncode,recorder=recorder,recorder_exited=stopped,stop_result=result,phase=capture.phase,error=str(capture.error))),flush=True)
    self.assertTrue(stopped,'inherited request_stop killed supervisor but left synthetic recorder alive')
   finally:
    capture.stop()
    if fd is not None:
     try:signal.pidfd_send_signal(fd,signal.SIGKILL)
     except ProcessLookupError:pass
     poller=select.poll();poller.register(fd,select.POLLIN)
     self.assertTrue(poller.poll(3000),'synthetic recorder did not exit during containment')
     os.close(fd)
if __name__=='__main__':unittest.main(verbosity=2)
