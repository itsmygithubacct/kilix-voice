"""A finite owned fixture forks after procfs enumeration, before stat reads.
No PID reuse, tracing, namespace, or unrelated process is involved.
"""
import contextlib,os,select,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
from voicelib import processes
from tests import test_acquisition_fallback

class ScanRequirement(unittest.TestCase):
 def test_fork_during_scan_cannot_be_sealed_as_cleanup(self):
  with tempfile.TemporaryDirectory() as root:
   marker=Path(root)/'member.pid'
   readfd,writefd=os.pipe()
   code='''import os,sys,time,pathlib
os.read(int(sys.argv[1]),1)
pid=os.fork()
if pid==0:
 p=pathlib.Path(sys.argv[2]);t=p.with_suffix('.ready');t.write_text(str(os.getpid()));t.replace(p)
 time.sleep(30)
 os._exit(0)
os._exit(0)
'''
   leader=processes.GroupPopen([sys.executable,'-I','-c',code,str(readfd),str(marker)],pass_fds=(readfd,))
   os.close(readfd);memberfd=None
   original=Path.iterdir;ran=False
   def enumerate_then_fork(path):
    nonlocal ran,memberfd
    snapshot=list(original(path))
    if str(path)=='/proc' and not ran:
     ran=True;os.write(writefd,b'!')
     deadline=time.monotonic()+3
     while not marker.exists() or processes.poll_exit(leader) is None:
      if time.monotonic()>deadline:raise RuntimeError('fixture did not publish and exit')
      time.sleep(.005)
     memberfd=os.pidfd_open(int(marker.read_text()))
    return iter(snapshot)
   try:
    with mock.patch.object(processes,'_pidfd_group_supported',False),mock.patch.object(Path,'iterdir',enumerate_then_fork):
     try:processes.kill_group(leader)
     except TimeoutError:
      self.assertFalse(getattr(leader,'_voice_group_signalled',False))
      self.assertIsNone(leader.returncode)
      self.assertIsNotNone(leader._voice_pidfd)
      return  # unrelated process churn can prevent a complete scan
    poller=select.poll();poller.register(memberfd,select.POLLIN)
    alive=not bool(poller.poll(0))
    sealed=getattr(leader,'_voice_group_signalled',False)
    code=leader.wait(timeout=2)
    print('SCAN_RESULT',dict(leader=leader.pid,member=int(marker.read_text()),sealed=sealed,member_alive=alive,wait_returncode=code,pidfd_after_wait=leader._voice_pidfd),flush=True)
    self.assertFalse(sealed and alive,'fallback certified cleanup while member forked during scan is alive')
   finally:
    os.close(writefd)
    if memberfd is not None:
     with contextlib.suppress(ProcessLookupError):signal.pidfd_send_signal(memberfd,signal.SIGKILL)
     os.close(memberfd)
    with contextlib.suppress(ProcessLookupError):os.kill(leader.pid,signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):os.waitpid(leader.pid,0)
    if leader._voice_pidfd is not None:os.close(leader._voice_pidfd);leader._voice_pidfd=None
    leader.returncode=0
class ScanProof(unittest.TestCase):
 def test_birth_during_enumeration_prevents_completion_proof(self):
  child=processes.GroupPopen(['/bin/sleep','30'])
  read=Path.read_text;count=iter(range(10000));clock=iter(range(10000))
  def contents(path,*args,**kwargs):
   if str(path)=='/proc/stat':return 'processes '+str(next(count))+'\n'
   return read(path,*args,**kwargs)
  try:
   signal.pidfd_send_signal(child._voice_pidfd,signal.SIGKILL)
   processes.wait_exit(child,2)
   with mock.patch.object(processes,'_pidfd_group_supported',False), \
        mock.patch.object(Path,'read_text',contents), \
        mock.patch.object(processes.time,'monotonic',side_effect=lambda:next(clock)/10):
    with self.assertRaises(TimeoutError):processes.kill_group(child)
    self.assertFalse(getattr(child,'_voice_group_signalled',False))
    self.assertIsNotNone(child._voice_pidfd)
  finally:test_acquisition_fallback.Acquisition.contain(self,child)

 def test_empty_snapshot_keeps_process_and_lease_owned(self):
  from tests.test_cleanup_recovery import Boundaries
  from voicelib import audio
  daemon,_,_=Boundaries.daemon(self)
  capture=audio.MicCapture({'audio':{'capture_cmd':['/bin/sleep','30']}})
  child=processes.GroupPopen(['/bin/sleep','30']);capture._process=child
  grant=mock.Mock()
  key,_=daemon._register_cleanup('scan','capture',capture,grant,'stop')
  clock=iter(range(10000))
  try:
   with mock.patch.object(processes,'_pidfd_group_supported',False), \
        mock.patch.object(processes,'_scan_group_members',return_value=[]), \
        mock.patch.object(processes.time,'monotonic',side_effect=lambda:next(clock)):
    with self.assertRaises(TimeoutError):processes.kill_group(child)
    self.assertFalse(getattr(child,'_voice_group_signalled',False))
    self.assertIsNotNone(child._voice_pidfd)
    self.assertIsNone(child.returncode)
   with mock.patch.object(processes,'kill_group',side_effect=TimeoutError('unproven scan')):
    daemon._retry_dictation_cleanup(.2)
   self.assertIn(key,daemon._cleanup_owners)
   self.assertIs(capture._process,child)
   self.assertFalse(capture.cleanup_complete)
   grant.reaped.assert_not_called()
  finally:test_acquisition_fallback.Acquisition.contain(self,child)

 def test_zombie_stat_does_not_prove_all_threads_exited(self):
  child=processes.GroupPopen(['/bin/sleep','30'])
  fds=[]
  try:
   with mock.patch.object(Path,'iterdir',return_value=iter([Path('/proc')/str(child.pid)])), \
        mock.patch.object(processes,'_member_stat',return_value=('Z',child.pid,'1')):
    fds=processes._scan_group_members(child.pid)
   self.assertEqual(len(fds),1,'a zombie group leader can still have live threads')
  finally:
   for fd in fds:os.close(fd)
   test_acquisition_fallback.Acquisition.contain(self,child)

 def test_member_created_after_first_scan_requires_another_pass(self):
  with tempfile.TemporaryDirectory() as root:
   marker=Path(root)/'child';r,w=os.pipe()
   code='''import os,sys,time,pathlib
os.read(int(sys.argv[1]),1)
if os.fork()==0:
 p=pathlib.Path(sys.argv[2]);t=p.with_suffix('.ready');t.write_text(str(os.getpid()));t.replace(p)
time.sleep(30)
'''
   child=processes.GroupPopen([sys.executable,'-I','-c',code,str(r),str(marker)],pass_fds=(r,))
   os.close(r);memberfd=None;sent=False;send=signal.pidfd_send_signal
   def fork_then_signal(fd,sig,*args):
    nonlocal sent,memberfd
    if not sent:
     sent=True;os.write(w,b'!');deadline=time.monotonic()+3
     while not marker.exists():
      if time.monotonic()>deadline:raise RuntimeError('fixture did not fork')
      time.sleep(.002)
     memberfd=os.pidfd_open(int(marker.read_text()))
    return send(fd,sig,*args)
   try:
    with mock.patch.object(processes,'_pidfd_group_supported',False), \
         mock.patch.object(signal,'pidfd_send_signal',side_effect=fork_then_signal):
     try:processes.kill_group(child)
     except TimeoutError:pass
    poller=select.poll();poller.register(memberfd,select.POLLIN)
    self.assertFalse(getattr(child,'_voice_group_signalled',False) and not poller.poll(0),
                     'one signalling pass certified a live newly forked member')
   finally:
    os.close(w)
    if memberfd is not None:
     with contextlib.suppress(ProcessLookupError):send(memberfd,signal.SIGKILL)
     os.close(memberfd)
    test_acquisition_fallback.Acquisition.contain(self,child)

if __name__=='__main__':unittest.main(verbosity=2)
