"""A dictation stop may name its receiver, even after that path is removed."""
import os
import socket
import threading
from unittest import mock

from tests import test_protocol
from tests.test_terminal_outcomes import voiced
from voicelib import protocol


class ScopedStopProtocolTests(test_protocol.SessionTestCase):
    def stop(self, path, **fields):
        return protocol.validate_request(
            dict(op="stop-dictation", sock=path, **fields), self.session)

    def test_scope_is_preserved_with_each_mode(self):
        path = self.sock("receiver.sock")
        for fields in ({}, {"mode": "finish"}, {"mode": "abort"}):
            with self.subTest(fields=fields):
                self.assertEqual(self.stop(path, **fields),
                                 dict(op="stop-dictation", id="", sock=path, **fields))

    def test_scope_normalises_like_dictate_including_symlinks(self):
        path = self.sock("receiver.sock")
        os.symlink(path, self.sock("alias.sock"))
        os.symlink(self.session, os.path.join(self.root, "session-link"))
        for raw in (self.sock(".", "receiver.sock"),
                    self.sock("missing", "..", "receiver.sock"),
                    self.session + "//receiver.sock", self.sock("alias.sock"),
                    os.path.join(self.root, "session-link", "receiver.sock")):
            with self.subTest(raw=raw):
                self.assertEqual(self.dictate(raw)["sock"], path)
                self.assertEqual(self.stop(raw).get("sock"), path)

    def test_missing_and_unlinked_receiver_paths_are_accepted(self):
        path = self.sock("receiver.sock")
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.stop(path).get("sock"), path)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
            receiver.bind(path)
            os.unlink(path)
            self.assertEqual(self.stop(path, mode="abort").get("sock"), path)

    def test_malformed_scope_is_refused(self):
        os.symlink(self.outside, self.sock("escape"))
        for raw in (None, "", 3, True, [], {}, "relative.sock",
                    self.sock("nul\0.sock"), self.sock("\ud800"),
                    self.sock("a" * 4096), self.sock("é" * 4096),
                    self.session, self.session + "/.", self.session + "-other/r.sock",
                    self.sock("..", "outside.sock"),
                    os.path.join(self.outside, "receiver.sock"),
                    self.sock("escape", "receiver.sock")):
            for mode in ("finish", "abort"):
                with self.subTest(raw=raw, mode=mode):
                    with self.assertRaises(protocol.ProtocolError) as caught:
                        self.stop(raw, mode=mode)
                    # The request handler maps the default code to malformed.
                    self.assertIn(caught.exception.code, (None, protocol.ERR_MALFORMED))

    def test_mode_validation_still_applies_with_scope(self):
        for mode in ("cancel", "", None, True, 3, [], {}):
            with self.subTest(mode=mode):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self.stop(self.sock("receiver.sock"), mode=mode)
                self.assertIn(caught.exception.code, (None, protocol.ERR_MALFORMED))

    def test_unscoped_request_shape_is_unchanged(self):
        for fields in ({}, {"mode": "finish"}, {"mode": "abort"}):
            with self.subTest(fields=fields):
                self.assertEqual(protocol.validate_request(
                    dict(op="stop-dictation", **fields), self.session),
                    dict(op="stop-dictation", id="", **fields))

    def test_unlinked_scope_aborts_before_the_watcher_runs(self):
        # Make the stop win this ordering deterministically. The process
        # tests separately leave the watcher running across unlink and stop.
        path = self.sock("receiver.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
            receiver.bind(path)
            sender = voiced._connect_dictation(path)
            self.addCleanup(sender.close)
            turn = voiced._DictationTurn("listen-1", sender, receiver_path=path)
            daemon = object.__new__(voiced.Daemon)
            daemon._lock = threading.RLock()
            daemon._dictation = turn
            os.unlink(path)
            reply = daemon._op_stop_dictation(self.stop(path, mode="abort"))
            self.assertEqual(reply, dict(ok=True, id="", stopped=True,
                                         quiesced=True, mode="abort"))
            self.assertTrue(turn.abort)
            self.assertTrue(turn.stop.is_set())

    def test_mismatched_scope_never_waits_for_or_changes_the_current_turn(self):
        turn = voiced._DictationTurn("listen-2", mock.Mock(),
                                    receiver_path=self.sock("b.sock"))
        turn.progress_lock = mock.Mock()
        daemon = object.__new__(voiced.Daemon)
        daemon._lock = threading.RLock()
        daemon._dictation = turn
        for mode in ("finish", "abort"):
            with self.subTest(mode=mode):
                reply = daemon._op_stop_dictation(
                    self.stop(self.sock("a.sock"), mode=mode))
                self.assertEqual(reply, dict(ok=True, id="", stopped=False, mode=mode))
                turn.progress_lock.acquire.assert_not_called()
                self.assertFalse(turn.stop.is_set())
                self.assertFalse(turn.abort)
