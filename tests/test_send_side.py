"""SEND-01: no frame leaves kilix-voiced with a code outside the closed set, or none.

The closed vocabulary was enforced only by the constructors that build a
refusal. A reply or datagram a handler or worker built by hand -- an unknown
code, no code, no ok -- crossed the wire exactly as built. The guard now sits
at the final wire boundary, _transmit, for the control connection and for
receiver sockets alike, and a terminal is guarded before its job settles, so
the ledger holds the code that crossed.

Every test here reads what a peer receives on a real AF_UNIX socket.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_send_side", importlib.machinery.SourceFileLoader(
        "kilix_voiced_send_side", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import jobs, protocol  # noqa: E402

PLANTED = "a message built by hand, bypassing the constructors"
_INT = struct.calcsize("i")

# What a handler hands the send path instead of a constructor's refusal.
UNCODED_REFUSALS = (
    ("code 'bogus'", {"ok": False, "error": PLANTED, "code": "bogus"}),
    ("no code", {"ok": False, "error": PLANTED}),
    ("code 'failed', a job outcome", {"ok": False, "error": PLANTED, "code": "failed"}),
    ("code None", {"ok": False, "error": PLANTED, "code": None}),
    ("code 'Internal'", {"ok": False, "error": PLANTED, "code": "Internal"}),
    ("code 'internal ' with a space", {"ok": False, "error": PLANTED, "code": "internal "}),
    ("code 5", {"ok": False, "error": PLANTED, "code": 5}),
    ("no ok key", {"error": PLANTED, "code": "bogus"}),
    ("ok true carrying an error", {"ok": True, "error": PLANTED}),
    ("prose that is not text", {"ok": False, "error": 5, "code": "bogus"}),
    ("not a mapping at all", "a bare string"),
)

# What a dictation worker hands _deliver_terminal instead of a coded terminal.
UNCODED_ERRORS = (
    ("code 'bogus'", {"error": PLANTED, "code": "bogus"}),
    ("no code", {"error": PLANTED}),
    ("code 'failed'", {"error": PLANTED, "code": "failed"}),
    ("the constructor's code-less form", protocol.dictation_error(PLANTED)),
    ("a final that also carries an error", {"final": "words", "error": PLANTED}),
    ("neither a transcript nor an error", {"partial": "words"}),
    ("an error that will not encode", {"error": "name \udcff", "code": "bogus",
                                       "detail": object()}),
    ("prose far past any frame", {"error": "x" * 300_000}),
)


def _received(ancillary) -> list[int]:
    return [fd for level, kind, blob in ancillary
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS
            for fd in struct.unpack(f"{len(blob) // _INT}i", blob)]


class ControlWireTestCase(unittest.TestCase):
    """What a client receives from the real _accept_one."""

    def setUp(self) -> None:
        root = tempfile.mkdtemp(prefix="kv-send-", dir="/tmp")
        self.addCleanup(shutil.rmtree, root, True)
        self.path = os.path.join(root, "control.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(self.listener.close)
        self.listener.bind(self.path)
        self.listener.listen(4)
        self.listener.setblocking(False)

    def exchange(self, reply) -> tuple[list[bytes], list[int]]:
        """One request answered with ``reply``: every frame and descriptor received."""
        d = object.__new__(voiced.Daemon)
        d._socket = self.listener
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        # Reads the request as the real _answer does, then answers with ``reply``.
        d._answer = lambda conn: (conn.recv(1 << 16), reply)[1]
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(5)
        with client:
            client.connect(self.path)
            client.send(b'{"op":"status"}\n')
            voiced.Daemon._accept_one(d)
            frames, fds = [], []
            while True:
                data, ancillary, _flags, _ = client.recvmsg(1 << 16, socket.CMSG_SPACE(4 * _INT))
                fds.extend(_received(ancillary))
                if not data:
                    break
                frames.append(data)
        for fd in fds:
            os.close(fd)
        return frames, fds

    def test_a_refusal_without_a_closed_code_goes_out_internal(self) -> None:
        for label, reply in UNCODED_REFUSALS:
            with self.subTest(reply=label):
                frames, fds = self.exchange(reply)
                self.assertEqual(len(frames), 1, frames)
                sent = protocol.decode(frames[0])
                self.assertIs(sent["ok"], False, sent)
                self.assertEqual(sent["code"], protocol.ERR_INTERNAL, sent)
                self.assertIsInstance(sent["error"], str)
                self.assertEqual(fds, [])

    def test_a_recoded_refusal_keeps_its_prose_and_its_other_fields(self) -> None:
        frames, _fds = self.exchange({"ok": False, "id": "r-1", "error": PLANTED,
                                      "code": "bogus", "protocol": {"major": 1}})
        self.assertEqual(protocol.decode(frames[0]),
                         {"ok": False, "error": PLANTED, "code": protocol.ERR_INTERNAL,
                          "id": "r-1", "protocol": {"major": 1}})

    def test_a_well_formed_refusal_or_success_crosses_unchanged(self) -> None:   # control
        for reply in ({"ok": False, "error": PLANTED, "code": protocol.ERR_BUSY},
                      protocol.reply_error("x", protocol.ERR_DENIED),
                      protocol.reply_ok("s-1", stopped=True)):
            with self.subTest(reply=reply):
                frames, _fds = self.exchange(reply)
                self.assertEqual(frames, [protocol.encode_reply(reply)])

    def test_a_refusal_never_carries_a_descriptor(self) -> None:
        for code, expected in (("bogus", protocol.ERR_INTERNAL),
                               (protocol.ERR_BUSY, protocol.ERR_BUSY)):
            with self.subTest(code=code):
                reader, writer = os.pipe()
                os.close(writer)
                frames, fds = self.exchange(voiced._Outbound(
                    {"ok": False, "error": PLANTED, "code": code}, fds=(reader,)))
                self.assertEqual(protocol.decode(frames[0])["code"], expected)
                self.assertEqual(fds, [])
                with self.assertRaises(OSError):       # the daemon closed its copy
                    os.fstat(reader)

    def test_a_success_keeps_its_descriptor(self) -> None:                       # control
        fd = os.memfd_create("kv-send-side")
        os.write(fd, b"audio")
        frames, fds = self.exchange(voiced._Outbound(
            protocol.reply_ok("i-1", audio_fd=0), fds=(fd,)))
        self.assertIs(protocol.decode(frames[0])["ok"], True)
        self.assertEqual(len(fds), 1)


class FinalBoundaryTestCase(unittest.TestCase):
    """The guard is where a frame leaves, not where it was built."""

    def pair(self) -> tuple[socket.socket, socket.socket]:
        daemon_end, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(daemon_end.close)
        self.addCleanup(peer.close)
        peer.settimeout(5)
        return daemon_end, peer

    def test_bytes_handed_straight_to_transmit_are_guarded(self) -> None:
        table = (
            ("reply, code bogus", voiced._CHANNEL_REPLY,
             b'{"ok":false,"error":"x","code":"bogus"}\n',
             {"ok": False, "error": "x", "code": protocol.ERR_INTERNAL}),
            ("reply, not JSON", voiced._CHANNEL_REPLY, b"not json\n",
             {"ok": False, "code": protocol.ERR_INTERNAL}),
            ("datagram, no code", voiced._CHANNEL_MESSAGE,
             b'{"error":"x","segment":"listen-1"}\n',
             {"error": "x", "code": protocol.ERR_INTERNAL, "segment": "listen-1"}),
            ("completed terminal carrying prose", voiced._CHANNEL_MESSAGE,
             b'{"terminal":true,"job":"speak-1","kind":"speech","outcome":"completed",'
             b'"chunks":1,"error":"read-aloud finished"}\n',
             {"terminal": True, "job": "speak-1", "outcome": "failed",
              "code": protocol.ERR_INTERNAL, "error": "read-aloud finished"}),
            ("cancelled terminal with no code", voiced._CHANNEL_MESSAGE,
             b'{"terminal":true,"job":"speak-2","kind":"speech","outcome":"cancelled",'
             b'"chunks":0,"error":"stopped"}\n',
             {"outcome": "failed", "code": protocol.ERR_INTERNAL}),
        )
        for label, channel, frame, expected in table:
            with self.subTest(frame=label):
                daemon_end, peer = self.pair()
                voiced._transmit(daemon_end, frame, channel=channel)
                sent = json.loads(peer.recv(1 << 16))
                self.assertEqual({key: sent.get(key) for key in expected}, expected, sent)

    def test_a_well_formed_frame_crosses_byte_for_byte(self) -> None:            # control
        table = (
            (voiced._CHANNEL_REPLY, protocol.reply_error("x", protocol.ERR_BUSY)),
            (voiced._CHANNEL_REPLY, protocol.reply_ok("s", status={"jobs": []})),
            (voiced._CHANNEL_MESSAGE, protocol.dictation_partial("word", "listen-1")),
            (voiced._CHANNEL_MESSAGE, protocol.dictation_error(
                "x", protocol.ERR_DEADLINE, "listen-1")),
            (voiced._CHANNEL_MESSAGE, {"terminal": True, "job": "speak-1", "kind": "speech",
                                       "outcome": "completed", "chunks": 0}),
            (voiced._CHANNEL_MESSAGE, {"terminal": True, "job": "speak-1", "kind": "speech",
                                       "outcome": "failed", "chunks": 0,
                                       "code": protocol.ERR_UNAVAILABLE, "error": "x"}),
        )
        for channel, message in table:
            with self.subTest(message=message):
                daemon_end, peer = self.pair()
                frame = protocol.encode(message, limit=protocol.MAX_REPLY_BYTES)
                voiced._transmit(daemon_end, frame, channel=channel)
                self.assertEqual(peer.recv(1 << 16), frame)


class DictationWireTestCase(unittest.TestCase):
    """What a dictation receiver gets, and what the ledger settles, for one terminal."""

    def deliver(self, message) -> tuple[bool, list[dict], dict]:
        daemon_end, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(daemon_end.close)
        self.addCleanup(peer.close)
        ledger = jobs.JobLedger()
        d = object.__new__(voiced.Daemon)
        d._warn = d._debug = lambda *a, **k: None
        d._jobs = ledger
        turn = voiced._DictationTurn("listen-3", daemon_end)
        turn.on_settle = ledger.record
        delivered = voiced.Daemon._deliver_terminal(d, turn, message)
        peer.setblocking(False)
        datagrams = []
        while True:
            try:
                datagrams.append(json.loads(peer.recv(1 << 16)))
            except BlockingIOError:
                break
        return delivered, datagrams, ledger.get("listen-3")

    def test_an_error_without_a_closed_code_goes_out_and_settles_internal(self) -> None:
        for label, message in UNCODED_ERRORS:
            with self.subTest(message=label):
                delivered, datagrams, record = self.deliver(message)
                self.assertIs(delivered, True)
                self.assertEqual(len(datagrams), 1, datagrams)
                sent = datagrams[0]
                self.assertNotIn("final", sent)
                self.assertEqual(sent["code"], protocol.ERR_INTERNAL, sent)
                self.assertIsInstance(sent["error"], str)
                self.assertLessEqual(len(protocol.encode(sent)), protocol.MAX_REPLY_BYTES)
                self.assertEqual(sent["segment"], "listen-3")
                self.assertEqual((record["outcome"], record["code"], record["delivered"]),
                                 ("failed", protocol.ERR_INTERNAL, True))
                if label == "prose far past any frame":
                    # Cut to fit, not replaced: the reason still reaches a human.
                    self.assertTrue(sent["error"].startswith("x" * 1000), sent["error"][:80])

    def test_send_itself_never_raises_for_an_error_that_will_not_encode(self) -> None:
        daemon_end, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(daemon_end.close)
        self.addCleanup(peer.close)
        peer.settimeout(5)
        d = object.__new__(voiced.Daemon)
        d._warn = d._debug = lambda *a, **k: None
        for message in ({"code": "bogus", "detail": object()},
                        {"error": PLANTED, "detail": object(), "segment": "listen-9"}):
            with self.subTest(message=sorted(message)):
                self.assertIs(voiced.Daemon._send(d, daemon_end, message), True)
                sent = json.loads(peer.recv(1 << 16))
                self.assertEqual(sent["code"], protocol.ERR_INTERNAL, sent)
                self.assertIsInstance(sent["error"], str)

    def test_a_coded_error_keeps_its_code_on_the_wire_and_in_the_ledger(self) -> None:  # control
        for code, outcome in ((protocol.ERR_BUSY, "failed"),
                              (protocol.ERR_CANCELLED, "cancelled"),
                              (protocol.ERR_DEADLINE, "deadline")):
            with self.subTest(code=code):
                _delivered, datagrams, record = self.deliver(
                    protocol.dictation_error(PLANTED, code))
                self.assertEqual([(m["code"], m["error"]) for m in datagrams], [(code, PLANTED)])
                self.assertEqual((record["outcome"], record["code"]), (outcome, code))

    def test_a_transcript_crosses_unchanged_and_settles_completed(self) -> None:  # control
        _delivered, datagrams, record = self.deliver(
            protocol.dictation_final("the words", "listen-3"))
        self.assertEqual(datagrams, [{"final": "the words", "segment": "listen-3",
                                      "stable": True}])
        self.assertEqual(record["outcome"], "completed")


if __name__ == "__main__":
    unittest.main()
