"""Real owned microphone protocol, isolated synthetic capture only."""
import json
import os
from pathlib import Path
import socket
import sys
import subprocess
import time

import unittest
from tests import test_daemon as daemon_tests


class OwnedMicrophoneTests(unittest.TestCase):
    setUp = daemon_tests.DaemonTestCase.setUp
    tearDown = daemon_tests.DaemonTestCase.tearDown
    _stop_daemon = daemon_tests.DaemonTestCase._stop_daemon
    _log_tail = daemon_tests.DaemonTestCase._log_tail
    _wait_until_serving = daemon_tests.DaemonTestCase._wait_until_serving
    _send = daemon_tests.DaemonTestCase._send

    def _environment(self):
        env = daemon_tests.DaemonTestCase._environment(self)
        script = Path(self.root)/'capture.py'
        script.write_text('import os,sys,time\n' +
                          f'open({str(Path(self.root)/"capture.pid")!r},"w").write(str(os.getpid()))\n' +
                          'while True:\n sys.stdout.buffer.write(b"\\0"*640);sys.stdout.buffer.flush();time.sleep(.02)\n')
        if self._testMethodName == 'test_on_waits_three_seconds_after_speech':
            script.write_text('import sys,time\n' +
                'for i in range(240):\n' +
                ' frame=(b"\\xff\\x3f\\x01\\xc0"*160) if 10 <= i < 25 else b"\\0"*640\n' +
                ' sys.stdout.buffer.write(frame);sys.stdout.buffer.flush();time.sleep(.02)\n')
        config = Path(self.root)/'config.json' 
        config.write_text(json.dumps({'audio':{'capture_cmd':[sys.executable,str(script)]},
                                      'stt':{'engine':'null','max_seconds':30},'vad':{'silence_ms':200}}))
        env['KILIX_VOICE_CONFIG']=str(config)
        subprocess.run([sys.executable,str(Path(daemon_tests.DAEMON).parent/'kilix-stt'),'--grant-consent'],env=env,check=True,capture_output=True)
        return env

    def receiver(self, name='mic.sock'):
        endpoint=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
        endpoint.bind(str(Path(self.session_dir)/name));endpoint.settimeout(5)
        self.addCleanup(endpoint.close)
        return endpoint

    def start(self, receiver, owner='a'*32, hold=True):
        result=self._send({'op':'owned-dictate','owner':owner,'hold':hold,'sock':receiver.getsockname()})
        self.assertTrue(result['ok'],result)
        self.assertEqual(json.loads(receiver.recv(65536)),{'listening':True})
        return owner

    def test_on_waits_three_seconds_after_speech(self):
        receiver=self.receiver();owner='a'*32
        result=self._send({'op':'owned-dictate','owner':owner,'hold':False,
                           'silence_ms':3000,'sock':receiver.getsockname()})
        self.assertTrue(result['ok'],result)
        started=time.monotonic()
        receiver.settimeout(.15)
        final=None
        while time.monotonic()-started < 5:
            self._send({'op':'owned-dictation-status','owner':owner})
            try: message=json.loads(receiver.recv(65536))
            except socket.timeout: continue
            if 'final' in message:
                final=message;break
        self.assertIsNotNone(final)
        self.assertGreaterEqual(time.monotonic()-started,3.3)
        self.assertLess(time.monotonic()-started,4.5)

    def test_release_flushes_and_only_owner_can_stop(self):
        receiver=self.receiver();owner=self.start(receiver)
        self.assertFalse(self._send({'op':'owned-stop-dictation','owner':'b'*32})['stopped'])
        self.assertTrue(self._send({'op':'owned-dictation-status','owner':owner})['active'])
        time.sleep(.1)
        started=time.monotonic()
        self.assertTrue(self._send({'op':'owned-stop-dictation','owner':owner})['stopped'])
        self.assertIn('final',json.loads(receiver.recv(65536)))
        self.assertLess(time.monotonic()-started,1)
        self.assertFalse(self._send({'op':'status'})['status']['listening'])

    def test_lost_client_lease_closes_capture(self):
        receiver=self.receiver();self.start(receiver)
        time.sleep(3.8)
        self.assertFalse(self._send({'op':'status'})['status']['listening'])
        self.assertEqual(json.loads(receiver.recv(65536))['code'],'cancelled')

    def test_invalid_owner_and_hold_cannot_open_microphone(self):
        receiver=self.receiver()
        for fields in ({'owner':'bad'}, {'owner':'a'*32,'hold':'yes'}, {'owner':'a'*32,'silence_ms':True}, {'owner':'a'*32,'silence_ms':0}):
            self.assertFalse(self._send({'op':'owned-dictate','sock':receiver.getsockname(),**fields})['ok'])
        self.assertFalse((Path(self.root)/'capture.pid').exists())

    def test_another_recording_is_not_replaced(self):
        receiver=self.receiver();owner=self.start(receiver)
        other=self.receiver('other.sock')
        response=self._send({'op':'owned-dictate','owner':'b'*32,'sock':other.getsockname(),'hold':True})
        self.assertFalse(response['ok'])
        self.assertTrue(self._send({'op':'owned-dictation-status','owner':owner})['active'])
        self._send({'op':'owned-stop-dictation','owner':owner})
