"""Explicit TTS acquisition through Content's pinned, first-use flow.

Only catalogued models determined by the licence authority are offered. No
receipts are shipped or manufactured here, and no caller can supply an
agreement as a CLI flag.
"""
from pathlib import Path
import os
import sys
import termios
import time
import tty

MODEL_IDS = ("qwen3-tts-0.6b-customvoice", "qwen3-tts-0.6b-base",
             "qwen3-tts-1.7b-voicedesign")
PIPER_ID = "piper-en-us-kristin-medium"
POCKET_ID = "pocket-tts-english-python-alba"


class Declined(ValueError):
    """The person at the terminal quit without accepting."""


def read_continue_key(output):
    """Read a fresh key without Enter; always restore the terminal mode."""
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        # Discard stale input before displaying the decision prompt: a newline
        # left by a previous command must not accept a new licence.
        tty.setcbreak(fd, termios.TCSAFLUSH)
        output.write("\nPress q to quit or any key to continue: ")
        output.flush()
        key = os.read(fd, 1)
        return key.decode("latin-1")
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, previous)
        output.write("\n")
        output.flush()


def install(model_id, *, read_key=None, output=None):
    if model_id not in (*MODEL_IDS, PIPER_ID, POCKET_ID):
        raise ValueError("TTS model is not in the first-use catalog")
    output = output or sys.stdout
    if not sys.stdin.isatty() or not output.isatty():
        raise ValueError("TTS installation needs a terminal for first-use licence acceptance")
    try:
        from kilix_license import (ReceiptStore, load_determined_records,
                                   load_determined_texts, typed_agreement_line)
        from kilix_content.first_use import (install_with_agreement,
                                             license_record_for, needs_agreement,
                                             present_asset)
        from kilix_content.install import Installer
        from kilix_content.model import Catalog
        from kilix_content.receipt import catalog_bytes, _verify_frozen_schema
    except ImportError as error:
        raise RuntimeError("install the 0.2.2 kilix-content and kilix-license "
                           "packages in this Python environment first") from error
    _verify_frozen_schema()
    catalog = Catalog.loads(catalog_bytes().decode("utf-8"))
    try:
        spec = next(item for item in catalog.assets if item.asset_id == model_id)
    except StopIteration as error:
        raise RuntimeError(f"{model_id} is missing from the pinned Content catalog") from error
    records = load_determined_records()
    # The authority alone names the shared receipt store. Installed weights
    # stay in a separate audition store and do not select a system default.
    store = ReceiptStore.shared()
    from . import paths
    root = Path(paths.gpu_terminal_home()) / "tts-auditions"
    texts = load_determined_texts(root / "licence-texts")
    installer = Installer(str(root / "content"))
    deadline = time.monotonic() + 3600
    if needs_agreement(spec, records=records, store=store):
        record = license_record_for(spec, records)
        payload = present_asset(spec, record, texts, receipts=store, records=records)
        output.write(payload.decode("utf-8"))
        output.flush()
        output.write("\nContinuing accepts the licence shown above and starts the download.\n")
        output.flush()
        answer = (read_key or read_continue_key)(output)
        if answer in ("", "q", "Q", "\x03", "\x04"):
            raise Declined("Quit; no model was downloaded and no receipt was written.")
        # The 0.2.2 authority accepts a canonical agreement string. Adapt the
        # user's explicit keypress to that API only after the notice and key
        # event; we do not claim that the user typed the canonical sentence.
        install_with_agreement(spec, installer=installer, store=store,
                               records=records, texts=texts,
                               typed_text=typed_agreement_line(record),
                               report=lambda message: print(message, file=output, flush=True),
                               deadline=deadline)
    else:
        installer.ensure_upstream_asset(
            spec, store=store, records=records, notices=texts,
            report=lambda message: print(message, file=output, flush=True),
            deadline=deadline)
    return str(Path(installer.asset_destination(spec)) / "model")
