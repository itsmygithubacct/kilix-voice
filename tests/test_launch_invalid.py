import subprocess,unittest
from unittest import mock
from voicelib import processes
from voicelib.processes import GroupPopen
class Invalid(unittest.TestCase):
 def test_embedded_nul_preserves_synchronous_error(self):
  with self.assertRaises(ValueError):subprocess.Popen(['/bin/echo','a\0b'])
  p=None
  try:
   with self.assertRaises(ValueError):p=GroupPopen(['/bin/echo','a\0b'])
  finally:
   if p is not None:p.wait(timeout=3)

 def test_launcher_value_error_is_reported_even_after_exec_marker(self):
  encode=processes.marshal.dumps
  def invalid(request):
   request['argv']=['/bin/true','a\0b']
   return encode(request)
  p=None
  try:
   with mock.patch.object(processes.marshal,'dumps',side_effect=invalid):
    with self.assertRaises(ValueError):p=GroupPopen(['/bin/true'])
  finally:
   if p is not None:p.wait(timeout=3)

 def test_launcher_failure_before_exec_marker_is_reported(self):
  encode=processes.marshal.dumps
  def invalid(request):
   del request['restore_signals']
   return encode(request)
  p=None
  try:
   with mock.patch.object(processes.marshal,'dumps',side_effect=invalid):
    with self.assertRaisesRegex(RuntimeError,'launcher failed'):p=GroupPopen(['/bin/true'])
  finally:
   if p is not None:p.wait(timeout=3)

if __name__=='__main__':unittest.main(verbosity=2)
