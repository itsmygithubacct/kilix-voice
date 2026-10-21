import contextlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from voicelib import processes
class Arguments(unittest.TestCase):
 def check(self,command,**kw):
  expected=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
  child=processes.GroupPopen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
  out,err=child.communicate(timeout=4)
  print('ARGV',repr(command),'kwargs',repr(kw),'expected',repr(expected.stdout),'observed',repr(out),flush=True)
  self.assertEqual((child.returncode,out,err),(expected.returncode,expected.stdout,expected.stderr))
 def test_argv_env_cwd_pipes_shell_executable(self):
  with tempfile.TemporaryDirectory() as root:
   self.check(['/usr/bin/env'],env={'EXACT':'value'})
   self.check(['/bin/pwd'],cwd=root)
   self.check(['/bin/echo','one space','quote"','unicode-é'])
   self.check(['echo','path lookup'],env={'PATH':'/not-a-directory:/bin'})
   self.check(['different-argv0','hello'],executable='/bin/echo')
   self.check('printf "%s" "$VALUE"',shell=True,env={'VALUE':'shell works'})
   self.check(['printf "%s/%s" "$0" "$1"','first','second'],shell=True,executable='/bin/sh')
   child=processes.GroupPopen(['/bin/cat'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   self.assertEqual(child.communicate(b'pipe payload',timeout=4),(b'pipe payload',b''))
 def test_pass_fds_and_handshake_closes_on_exec(self):
  with tempfile.TemporaryFile() as file:
   fd=file.fileno()
   code='import os,sys;print(os.fstat(int(sys.argv[1])).st_ino);print(sorted(os.readlink("/proc/self/fd/"+n) for n in os.listdir("/proc/self/fd") if n not in ("0","1","2") and os.path.exists("/proc/self/fd/"+n)))'
   self.check([sys.executable,'-I','-c',code,str(fd)],pass_fds=(fd,))
 def test_close_fds_false_is_explicitly_refused(self):
  with tempfile.TemporaryFile() as file:
   fd=file.fileno();os.set_inheritable(fd,True)
   code='import os,sys;print(os.path.exists("/proc/self/fd/"+sys.argv[1]))'
   child=None
   try:
    with self.assertRaisesRegex(ValueError, 'close_fds=False'):
     child=processes.GroupPopen([sys.executable,'-I','-c',code,str(fd)],close_fds=False)
   finally:
    if child is not None:child.wait(timeout=3)
 def test_exec_errors(self):
  with tempfile.TemporaryDirectory() as root:
   denied=Path(root)/'denied';denied.write_text('#!/bin/sh\nexit 0\n');denied.chmod(0o600)
   for command in (['/nonexistent-command-rv11'],[str(denied)],[root]):
    with self.subTest(command=command):
     with self.assertRaises(OSError) as a:subprocess.Popen(command)
     with self.assertRaises(type(a.exception)) as b:processes.GroupPopen(command)
     self.assertEqual(a.exception.errno,b.exception.errno)
   with self.assertRaises(PermissionError):
    processes.GroupPopen(['denied'],env={'PATH':str(root)+':/nonexistent'})
if __name__=='__main__':unittest.main(verbosity=2)
