"""AUD-01: descriptors on the control socket and on outbound datagrams.

The control listener takes SCM_RIGHTS descriptors with a request and hands
them to dispatch. It keeps none past the reply, and serves an op that takes
none as if none had been sent. A reply can carry a descriptor, and the daemon
keeps no copy of it. Outbound sends can attach a descriptor, and can decline
to wait on a receiver that is not reading. Everything runs on real AF_UNIX
sockets.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_descriptors", importlib.machinery.SourceFileLoader(
        "kilix_voiced_descriptors", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import audiofd, protocol  # noqa: E402

_INT = struct.calcsize("i")


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def _rights(fds) -> list:
    return [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{len(fds)}i", *fds))]


def _received(ancillary) -> list[int]:
    return [fd for level, kind, blob in ancillary
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS
            for fd in struct.unpack(f"{len(blob) // _INT}i", blob)]


class ControlDescriptors(unittest.TestCase):
    """A real SEQPACKET listener served by the real Daemon._accept_one."""

    def setUp(self) -> None:
        self.session = os.path.realpath(tempfile.mkdtemp(prefix="kv-fd-", dir="/tmp"))
        self.addCleanup(shutil.rmtree, self.session, True)
        self.path = os.path.join(self.session, "control.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(self.listener.close)
        self.listener.bind(self.path)
        self.listener.listen(4)
        self.listener.setblocking(False)
        self.warnings = []
        d = object.__new__(voiced.Daemon)
        d._socket = self.listener
        d._session_dir = self.session
        d._stopping = threading.Event()
        d._lock = threading.RLock()
        d._speech = None
        d._player = None
        d._arbiter = mock.Mock()
        d._touch = lambda: None
        d._warn = self.warnings.append
        d._debug = lambda *a, **k: None
        self.daemon = d

    def connect(self) -> socket.socket:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(client.close)
        client.settimeout(5)
        client.connect(self.path)
        return client

    def exchange(self, request: dict, fds=()) -> dict:
        client = self.connect()
        frame = protocol.encode(request)
        if fds:
            client.sendmsg([frame], _rights(fds))
        else:
            client.send(frame)
        voiced.Daemon._accept_one(self.daemon)
        reply = protocol.decode(client.recv(1 << 16))
        client.close()
        return reply

    def test_descriptor_on_stop_speech_is_ignored_and_closed(self) -> None:
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, write_end)
        before = _open_descriptors()
        reply = self.exchange({"op": "stop-speech", "id": "s"}, [read_end])
        os.close(read_end)
        self.assertIs(reply["ok"], True, reply)
        self.assertEqual(reply["id"], "s")
        # Nobody holds the read end any more -- not this test, not the daemon.
        with self.assertRaises(BrokenPipeError):
            os.write(write_end, b"x")
        self.assertEqual(_open_descriptors(), before - 1)
        self.assertEqual(self.warnings, [])

    def test_five_descriptors_are_all_closed(self) -> None:
        pipes = [os.pipe() for _ in range(5)]
        for _read, write in pipes:
            self.addCleanup(os.close, write)
        before = _open_descriptors()
        reply = self.exchange({"op": "stop-speech"}, [read for read, _ in pipes])
        for read, _write in pipes:
            os.close(read)
        self.assertIs(reply["ok"], True, reply)
        for _read, write in pipes:
            with self.assertRaises(BrokenPipeError):
                os.write(write, b"x")
        self.assertEqual(_open_descriptors(), before - 5)

    def test_dispatch_is_handed_the_descriptors_that_arrived(self) -> None:
        # Closing alone proves nothing about receiving: a plain recv makes the
        # kernel close them too. The descriptors must reach dispatch, usable
        # and close-on-exec, and an over-limit batch must say it was cut.
        seen = []

        def dispatch(raw, fds=(), truncated=False):
            seen.append({"n": len(fds), "truncated": truncated,
                         "inheritable": [os.get_inheritable(fd) for fd in fds],
                         "bytes": [os.read(fd, 16) for fd in fds]})
            return protocol.reply_ok("seen")

        self.daemon._dispatch = dispatch
        pipes = [os.pipe() for _ in range(5)]
        for read, write in pipes:
            os.write(write, b"ping")
            os.close(write)
        self.exchange({"op": "stop-speech"}, [pipes[0][0]])
        self.exchange({"op": "stop-speech"}, [read for read, _ in pipes])
        for read, _write in pipes:
            os.close(read)
        self.assertEqual(seen[0], {"n": 1, "truncated": False,
                                   "inheritable": [False], "bytes": [b"ping"]})
        self.assertEqual(seen[1]["n"], audiofd.MAX_INBOUND_FDS)
        self.assertIs(seen[1]["truncated"], True)
        self.assertEqual(set(seen[1]["inheritable"]), {False})

    def test_a_reply_descriptor_reaches_the_peer_and_no_copy_is_kept(self) -> None:
        memfd = audiofd.sealed_readonly(b"reply audio")

        def answer(conn):
            conn.recv(1 << 16)
            return voiced._Outbound(protocol.reply_ok("r", audio_fd=0), fds=(memfd,))

        self.daemon._answer = answer
        client = self.connect()
        client.send(protocol.encode({"op": "status"}))
        voiced.Daemon._accept_one(self.daemon)
        # Checked before this process receives anything that could reuse the
        # number: the daemon's own copy is gone once the reply is sent.
        with self.assertRaises(OSError):
            os.fstat(memfd)
        data, ancillary, _flags, _ = client.recvmsg(
            1 << 16, socket.CMSG_SPACE(4 * _INT))
        fds = _received(ancillary)
        for fd in fds:
            self.addCleanup(os.close, fd)
        self.assertEqual(protocol.decode(data), {"ok": True, "id": "r", "audio_fd": 0})
        self.assertEqual(len(fds), 1)
        self.assertEqual(os.pread(fds[0], 64, 0), b"reply audio")

    def test_a_request_without_descriptors_is_served_as_before(self) -> None:  # control
        before = _open_descriptors()
        reply = self.exchange({"op": "status", "v": "9"})     # refused, still answered
        self.assertEqual(reply["code"], protocol.ERR_UNSUPPORTED, reply)
        self.assertIs(self.exchange({"op": "stop-speech"})["ok"], True)
        self.assertEqual(_open_descriptors(), before)


class Outbound(unittest.TestCase):
    """Daemon._send with descriptors and without waiting."""

    SLOW_S = 0.05

    def daemon(self):
        d = object.__new__(voiced.Daemon)
        self.warnings = []
        d._warn = self.warnings.append
        return d

    def pair(self, kind=socket.SOCK_SEQPACKET):
        a, b = socket.socketpair(socket.AF_UNIX, kind)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return a, b

    def test_fd_reaches_recvmsg_and_plain_recv_keeps_json(self) -> None:
        for kind in (socket.SOCK_SEQPACKET, socket.SOCK_DGRAM):
            with self.subTest(kind=kind.name):
                sender, receiver = self.pair(kind)
                memfd = audiofd.sealed_readonly(b"sealed clip bytes")
                self.addCleanup(os.close, memfd)
                d = self.daemon()
                message = {"sequence": 0, "final": False}
                self.assertTrue(voiced.Daemon._send(
                    d, sender, voiced._Outbound(message, fds=(memfd,))))
                data, ancillary, _flags, _ = receiver.recvmsg(
                    1 << 16, socket.CMSG_SPACE(4 * _INT))
                received = _received(ancillary)
                for fd in received:
                    self.addCleanup(os.close, fd)
                self.assertEqual(len(received), 1)
                self.assertEqual(os.pread(received[0], 64, 0), b"sealed clip bytes")
                self.assertEqual(protocol.decode(data), message)
                # A receiver that reads with plain recv, as V12/V14/V15 do,
                # still gets the JSON intact; the kernel drops the descriptor.
                self.assertTrue(voiced.Daemon._send(
                    d, sender, voiced._Outbound(message, fds=(memfd,))))
                self.assertEqual(protocol.decode(receiver.recv(1 << 16)), message)

    def fill(self, droppable: bool):
        """Send until the receiver's queue refuses; every send must be quick."""
        sender, receiver = self.pair()
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
        sender.settimeout(1.0)                 # as _connect_dictation sets it
        d = self.daemon()
        results = []
        for index in range(2000):
            started = time.monotonic()
            results.append(voiced.Daemon._send(d, sender, voiced._Outbound(
                {"partial": "word " * 20, "n": index},
                nonblocking=True, droppable=droppable)))
            elapsed = time.monotonic() - started
            if elapsed > self.SLOW_S:
                self.fail(f"send {index} waited {elapsed:.3f} s on a full receiver")
            if results[-1] is False:
                break
        self.assertEqual(sender.gettimeout(), 1.0,
                         "the socket's own timeout was not restored")
        return results, receiver

    def test_nonblocking_full_receiver_returns_false_fast(self) -> None:
        results, _receiver = self.fill(droppable=False)
        self.assertIs(results[-1], False, "the queue never filled")

    def test_droppable_full_receiver_returns_true(self) -> None:
        results, receiver = self.fill(droppable=True)
        self.assertNotIn(False, results)
        receiver.setblocking(False)
        delivered = 0
        try:
            while receiver.recv(1 << 16):
                delivered += 1
        except BlockingIOError:
            pass
        self.assertLess(delivered, len(results), "nothing was dropped, so nothing was tested")

    def test_a_gone_receiver_is_false_whether_or_not_it_waits(self) -> None:
        sender, receiver = self.pair()
        receiver.close()
        d = self.daemon()
        self.assertFalse(voiced.Daemon._send(d, sender, {"partial": "x"}))
        self.assertFalse(voiced.Daemon._send(
            d, sender, voiced._Outbound({"partial": "x"}, droppable=True)))

    def test_an_over_size_message_still_raises(self) -> None:
        # _run_dictation turns this into a too-large datagram; swallowing it
        # here would leave the caller with no terminal at all.
        sender, _receiver = self.pair()
        with self.assertRaises(protocol.MessageTooLarge):
            voiced.Daemon._send(self.daemon(), sender, {"final": "x" * 70_000})


if __name__ == "__main__":
    unittest.main()
