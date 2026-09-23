"""First-use wiring, always with a temporary receipt and content store."""
import io
import os
from pathlib import Path
import select
import subprocess
import sys
import termios
import tempfile
import time
import unittest
from unittest import mock

from voicelib import qwen_setup


class QwenSetupTests(unittest.TestCase):
    def test_unknown_model_rejected_before_import_or_fetch(self):
        with self.assertRaisesRegex(ValueError, "not in the first-use catalog"):
            qwen_setup.install("invented-model")

    def test_nonterminal_cannot_accept(self):
        with mock.patch.object(qwen_setup.sys.stdin, "isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "terminal"):
                qwen_setup.install(qwen_setup.MODEL_IDS[0])

    def test_decline_never_writes_receipt_or_fetches(self):
        for key in ("q", "Q", "", "\x03", "\x04"):
            with self.subTest(key=key):
                self.exercise(accept=False, key=key)

    def test_acceptance_goes_through_authority_then_fetch(self):
        for key in (" ", "\n", "x"):
            with self.subTest(key=key):
                self.exercise(accept=True, key=key)

    def exercise(self, *, accept, key):
        from kilix_license import ReceiptStore
        from kilix_content.install import Installer
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts")
            output = io.StringIO()
            output.isatty = lambda: True
            def fetch(*args, **kwargs):
                # The real first-use authority must have written a receipt
                # before the mocked network operation can be reached.
                self.assertEqual(len(list(store.root.glob("*.json"))), 1)
                return ()
            with mock.patch.object(qwen_setup.sys.stdin, "isatty", return_value=True), \
                    mock.patch("voicelib.paths.gpu_terminal_home", return_value=tmp), \
                    mock.patch.object(ReceiptStore, "shared", return_value=store), \
                    mock.patch.object(Installer, "ensure_upstream_asset", side_effect=fetch) as acquire:
                if accept:
                    result = qwen_setup.install(qwen_setup.MODEL_IDS[0],
                                                read_key=lambda output: key, output=output)
                    self.assertTrue(result.endswith("qwen3-tts-0.6b-customvoice/model"))
                    acquire.assert_called_once()
                else:
                    with self.assertRaises(qwen_setup.Declined):
                        qwen_setup.install(qwen_setup.MODEL_IDS[0],
                                           read_key=lambda output: key, output=output)
                    acquire.assert_not_called()
                    self.assertFalse(list(store.root.glob("*.json")))
            self.assertIn("Apache License", output.getvalue())
            self.assertIn("Continuing accepts", output.getvalue())

    def test_real_terminal_single_key_and_stale_input(self):
        for key in (b" ", b"q", b"\n"):
            with self.subTest(key=key):
                master, slave = os.openpty()
                previous = termios.tcgetattr(slave)
                process = None
                try:
                    # A prior line's pending Enter must not accept the notice.
                    os.write(master, b"\n")
                    code = ("from voicelib.qwen_setup import read_continue_key; import sys; "
                            "print('KEY=' + repr(read_continue_key(sys.stdout)), flush=True)")
                    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
                    process = subprocess.Popen([sys.executable, "-u", "-c", code],
                                               stdin=slave, stdout=slave, stderr=slave, env=env)
                    data = b""
                    deadline = time.monotonic() + 5
                    while b"any key to continue:" not in data:
                        self.assertLess(time.monotonic(), deadline)
                        if select.select([master], [], [], 0.1)[0]:
                            data += os.read(master, 4096)
                    self.assertIsNone(process.poll())
                    os.write(master, key)  # No trailing Enter for q or space.
                    self.assertEqual(process.wait(timeout=5), 0)
                    while select.select([master], [], [], 0.1)[0]:
                        data += os.read(master, 4096)
                    self.assertIn(("KEY=" + repr(key.decode())).encode(), data)
                    self.assertEqual(termios.tcgetattr(slave), previous)
                finally:
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait()
                    os.close(master)
                    os.close(slave)

    def test_terminal_restored_after_interrupt(self):
        with mock.patch.object(qwen_setup.sys.stdin, "fileno", return_value=9), \
                mock.patch.object(qwen_setup.termios, "tcgetattr", return_value=[1, 2]) as attrs, \
                mock.patch.object(qwen_setup.tty, "setcbreak"), \
                mock.patch.object(qwen_setup.termios, "tcsetattr") as restore, \
                mock.patch.object(qwen_setup.os, "read", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                qwen_setup.read_continue_key(io.StringIO())
            restore.assert_called_once_with(9, termios.TCSAFLUSH, attrs.return_value)
