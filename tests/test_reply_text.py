"""UTF8-S: a success reply stays encodable whatever names the daemon was given.

A status quotes values the daemon took from its environment and configuration:
a model or library path, the session directory, a capture command, failure
prose kept in the job ledger. A Linux name is bytes, and one that is not UTF-8
decodes to lone surrogates, which no UTF-8 frame can carry. Such a status used
to go out as `internal`, "this is a bug", in place of the answer. Every string
in a reply is now escaped as refusal prose already was: each lone surrogate
written as its backslash escape.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import socket
import struct
import unittest

from tests.livedaemon import LiveDaemonTestCase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_reply_text", importlib.machinery.SourceFileLoader(
        "kilix_voiced_reply_text", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol, stt as stt_lib  # noqa: E402

_INT = struct.calcsize("i")


class NonUtf8EnvironmentStatusTestCase(LiveDaemonTestCase):
    """A real kilix-voiced whose model and library paths are exported as non-UTF-8 bytes."""

    def _environment(self) -> dict:
        env = super()._environment()
        self.model_path = os.path.join(os.fsencode(self.root), b"model-\xff")
        self.library_path = os.path.join(os.fsencode(self.root), b"lib-\xfe.so")
        env[stt_lib.ENV_MODEL] = self.model_path
        env[stt_lib.ENV_LIBRARY] = self.library_path
        return env

    def test_status_answers_with_the_names_escaped(self) -> None:
        reply = self.request({"op": "status"})
        self.assertIs(reply["ok"], True, reply)
        stt = reply["status"]["stt"]
        escaped_model = os.fsdecode(self.model_path).encode("utf-8", "backslashreplace").decode()
        escaped_library = os.fsdecode(self.library_path).encode(
            "utf-8", "backslashreplace").decode()
        self.assertEqual(stt["model_path"], escaped_model)
        self.assertTrue(stt["model_path"].endswith("model-\\udcff"), stt)
        self.assertEqual(stt["library"], escaped_library)
        self.assertIn("model-\\udcff", stt["detail"])
        self.assertIs(stt["available"], False)
        self.assertEqual(reply["status"]["session"], os.path.dirname(self.control))
        self.assert_still_serving()


class ReplyFrameTestCase(unittest.TestCase):
    """What the frame for a success reply carries, read back from the bytes."""

    def test_every_string_in_a_success_reply_is_escaped_keys_included(self) -> None:
        reply = protocol.reply_ok("s-1", status={
            "session": "/run/voice-\udcff",
            "jobs": [{"job": "speak-1", "error": "cannot run '/opt/\udcfe/synth'"}],
            "capture": {"detail": "arecord -D hw\udcff"},
            "k\udcff": ["a\udcfd", 1, None]})
        frame, fds = voiced._reply_frame(reply)
        self.assertEqual(fds, ())
        sent = protocol.decode(frame)
        self.assertIs(sent["ok"], True, sent)
        status = sent["status"]
        self.assertEqual(status["session"], "/run/voice-\\udcff")
        self.assertEqual(status["jobs"][0]["error"], "cannot run '/opt/\\udcfe/synth'")
        self.assertEqual(status["capture"]["detail"], "arecord -D hw\\udcff")
        self.assertEqual(status["k\\udcff"], ["a\\udcfd", 1, None])

    def test_a_success_reply_keeps_its_descriptor_when_escaped(self) -> None:
        fd = os.memfd_create("kv-reply-text")
        self.addCleanup(os.close, fd)
        reply = voiced._Outbound(protocol.reply_ok("i-1", job="ingest-1", note="x\udcff",
                                                   audio_fd=0), fds=(fd,))
        frame, fds = voiced._reply_frame(reply)
        self.assertEqual(fds, (fd,))
        self.assertEqual(protocol.decode(frame)["note"], "x\\udcff")

    def test_text_that_encodes_crosses_byte_for_byte(self) -> None:          # control
        reply = protocol.reply_ok("s-2", status={"session": "/run/voice-é", "n": [1, "ü"]})
        frame, _fds = voiced._reply_frame(reply)
        self.assertEqual(frame, protocol.encode_reply(reply))

    def test_the_escaped_reply_crosses_the_wire(self) -> None:
        server, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(server.close)
        self.addCleanup(peer.close)
        frame, fds = voiced._reply_frame(protocol.reply_ok("w", path="/x/\udcff"))
        voiced._transmit(server, frame, fds, channel=voiced._CHANNEL_REPLY)
        self.assertEqual(json.loads(peer.recv(1 << 16)),
                         {"ok": True, "id": "w", "path": "/x/\\udcff"})


if __name__ == "__main__":
    unittest.main()
