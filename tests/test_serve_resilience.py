"""No request may end the daemon's serve loop, and every reply fits a client.

Four same-user requests below MAX_REQUEST_BYTES made _accept_one raise --
refusals that quoted the request back into a reply too large to encode, and
nesting past the recursion limit -- and three more raised out of _dispatch.
Nothing above _accept_one caught them, so one request ended the voice session.
The subprocess test in test_daemon.py drives the whole class end to end; these
pin each guard on its own, so no layer is tested only through another.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_serve", importlib.machinery.SourceFileLoader(
        "kilix_voiced_serve", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol  # noqa: E402


class _ListeningFixture(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "control.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(self.listener.close)
        self.listener.bind(self.path)
        self.listener.listen(4)
        self.listener.setblocking(False)
        self.warnings = []

    def daemon(self):
        d = object.__new__(voiced.Daemon)
        d._socket = self.listener
        d._stopping = threading.Event()
        d._warn = lambda message: self.warnings.append(message)
        d._debug = lambda *a, **k: None
        return d

    def client(self, payload=None):
        c = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(c.close)
        c.settimeout(5)
        c.connect(self.path)
        if payload is not None:
            c.send(payload)
        return c


class ServeLoopGuardTestCase(_ListeningFixture):

    def test_a_connection_that_raises_does_not_end_the_serve_loop(self) -> None:
        d = self.daemon()
        # Reads the request as the real _answer does: a record left unread
        # when the connection closes resets the peer before it reads the reply.
        d._answer = lambda conn: (conn.recv(1 << 16), protocol.reply_ok("served"))[1]
        d._check_idle = lambda: None
        d._wake_r, d._wake_w = os.pipe()
        self.addCleanup(os.close, d._wake_r)
        self.addCleanup(os.close, d._wake_w)
        calls = []

        def flaky_accept():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("a failure no narrower guard caught")
            return voiced.Daemon._accept_one(d)

        d._accept_one = flaky_accept
        server = threading.Thread(target=voiced.Daemon._serve, args=(d,),
                                  daemon=True)
        server.start()
        try:
            c = self.client(b'{"op":"status"}\n')
            reply = protocol.decode(c.recv(1 << 16))
        finally:
            d._stopping.set()
            os.write(d._wake_w, b"\0")
            server.join(10)
        self.assertTrue(reply["ok"], reply)
        self.assertGreaterEqual(len(calls), 2)
        self.assertFalse(server.is_alive())


class AcceptOneGuardTestCase(_ListeningFixture):

    def test_an_answer_that_raises_is_contained_and_the_peer_released(self) -> None:
        d = self.daemon()

        def broken(conn):
            raise RuntimeError("secret request text")

        d._answer = broken
        c = self.client()
        voiced.Daemon._accept_one(d)            # must not raise
        self.assertEqual(c.recv(1 << 16), b"")   # the connection was closed
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("RuntimeError", self.warnings[0])
        self.assertNotIn("secret", self.warnings[0])

    def test_a_reply_too_large_to_encode_is_still_answered(self) -> None:
        d = self.daemon()
        d._answer = lambda conn: protocol.reply_ok("", status={"x": "y" * 300_000})
        c = self.client()
        voiced.Daemon._accept_one(d)
        frame = c.recv(1 << 20)
        self.assertLessEqual(len(frame), protocol.MAX_REPLY_BYTES)
        reply = protocol.decode(frame)
        self.assertEqual(reply["code"], protocol.ERR_INTERNAL)


class DispatchNetTestCase(unittest.TestCase):

    def test_a_non_protocol_failure_reading_a_request_is_answered(self) -> None:
        d = object.__new__(voiced.Daemon)
        d._session_dir = "/tmp"
        warnings = []
        d._warn = warnings.append
        with mock.patch.object(voiced.protocol, "validate_request",
                               mock.Mock(side_effect=KeyError("op"))):
            reply = voiced.Daemon._dispatch(d, b'{"op":"status"}\n')
        self.assertIs(reply["ok"], False)
        self.assertEqual(reply["code"], protocol.ERR_INTERNAL)
        self.assertEqual(warnings, ["could not read a request (KeyError)"])


class DatagramBoundTestCase(unittest.TestCase):

    def _daemon(self):
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        return d

    def test_an_over_size_transcript_is_refused_not_sent(self) -> None:
        receiver = mock.Mock()
        d = self._daemon()
        huge = protocol.dictation_final("word " * 14_000, "listen-1")
        d._dictate = lambda turn: voiced.Daemon._send(d, turn.receiver, huge)
        voiced.Daemon._run_dictation(d, voiced._DictationTurn("listen-1", receiver))
        self.assertEqual(receiver.send.call_count, 1)
        frame = receiver.send.call_args.args[0]
        self.assertLessEqual(len(frame), protocol.MAX_REPLY_BYTES)
        datagram = protocol.decode(frame)
        self.assertNotIn("final", datagram)
        self.assertEqual(datagram["code"], protocol.ERR_TOO_LARGE)

    def test_a_chunk_descriptor_that_cannot_be_sent_does_not_stop_the_audio(self) -> None:
        d = self._daemon()
        d._send = mock.Mock(side_effect=protocol.MessageTooLarge("too big"))
        engine = mock.Mock(voice="en-us", model="m", rate=170, seed=None,
                           effective_model=None)
        turn = voiced._SpeechTurn("speak-1", ["one"], engine)
        turn.receiver = mock.Mock()
        voiced.Daemon._publish_chunk(d, turn, b"\x00\x00", 22050)   # must not raise
        d._send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
