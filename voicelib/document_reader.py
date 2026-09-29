"""Bounded document extraction and desktop-owned, cancellable playback.

The private stdin protocol is one byte per command: p play/resume, a pause,
s stop/reset, n next, b previous, q quit. EOF always stops this reader only.
JSON lines on stdout report state and the current passage. No daemon required.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import threading
import time

from . import audio, licensing, models, settings, tts

MAX_FILE = 16 * 1024 * 1024
MAX_TEXT = 8 * 1024 * 1024


def markdown_text(text):
    """A deliberately small prose reader, not a Markdown renderer."""
    text = re.sub(r'\A---\s*\n.*?\n---\s*\n', '', text, flags=re.S)
    lines, fence = [], None
    for line in text.splitlines():
        mark = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if mark:
            token = mark[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    text = '\n'.join(lines)
    text = re.sub(r'!\[([^\]]*)\]\([^\n)]*\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\n)]*\)', r'\1', text)
    text = re.sub(r'<[^>\n]*>', '', text)
    text = re.sub(r'(?m)^\s{0,3}(?:#{1,6}\s+|>\s*|[-+*]\s+)', '', text)
    text = re.sub(r'[*_`~]', '', text)
    return html.unescape(text)


def pdf_text(path):
    try:
        process = subprocess.Popen(['pdftotext', '-enc', 'UTF-8', str(path), '-'],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError as error:
        raise ValueError('PDF reading needs pdftotext (the poppler-utils package).') from error
    data = bytearray()
    deadline = time.monotonic() + 30
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('PDF extraction timed out after 30 seconds.')
            ready, _, _ = select.select([process.stdout], [], [], min(.1, remaining))
            if not ready:
                continue
            block = os.read(process.stdout.fileno(), 65536)
            if not block:
                break
            data.extend(block)
            if len(data) > MAX_TEXT:
                raise ValueError('Extracted PDF text exceeds the 8 MiB reader limit.')
        if process.wait(timeout=max(.1, deadline-time.monotonic())):
            raise ValueError('Cannot extract this PDF. It may be encrypted or damaged.')
        text = data.decode('utf-8')
        if not text.strip():
            raise ValueError('This PDF has no text. Scanned pages need OCR before reading.')
        return text
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def load_document(filename):
    path = Path(filename).expanduser().resolve()
    if path.suffix.lower() not in {'.txt', '.md', '.markdown', '.pdf'}:
        raise ValueError('Choose a TXT, Markdown (.md), or PDF document.')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Choose a regular document file.')
        if info.st_size > MAX_FILE:
            raise ValueError('Document exceeds the 16 MiB reader limit.')
        if path.suffix.lower() == '.pdf':
            text = pdf_text(path)
        else:
            data = stream.read(MAX_TEXT + 1)
            if len(data) > MAX_TEXT:
                raise ValueError('Text exceeds the 8 MiB reader limit.')
            try:
                text = data.decode('utf-8-sig')
            except UnicodeDecodeError as error:
                raise ValueError('Save this document as UTF-8 text before reading.') from error
            if path.suffix.lower() in {'.md', '.markdown'}:
                text = markdown_text(text)
    chunks = tts.speech_chunks(text, max_chars=None)
    if not chunks:
        raise ValueError('This document has no readable text.')
    return chunks


def reader_engine():
    if settings.tts_engine() == 'off':
        raise ValueError('Read-aloud is off. Choose a voice in Read Aloud Settings or opt in to Kristin.')
    if settings.tts_engine() == models.TTS_ENGINE_PIPER:
        licensing.require_covering_receipt(models.PIPER_KRISTIN_MODEL)
    return tts.make_tts()


class Synthesis:
    def __init__(self, text, factory):
        self.lock = threading.Lock()
        self.cancelled = False
        self.engine = None
        self.clip = self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(text, factory), daemon=True)
        self.thread.start()

    def run(self, text, factory):
        try:
            engine = factory()
            with self.lock:
                self.engine = engine
                cancelled = self.cancelled
            if not cancelled:
                self.clip = engine.synth(text, budget=60)
        except Exception as error:
            self.error = str(error)
        finally:
            try:
                if self.engine is not None:
                    self.engine.close()
            finally:
                self.done.set()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            engine = self.engine
        if engine is not None:
            engine.cancel()


class Reader:
    def __init__(self, chunks, emit, *, player=None, factory=reader_engine):
        self.chunks, self.emit = chunks, emit
        self.player = player if player is not None else audio.Player({})
        self.factory = factory
        self.index = 0
        self.state = 'ready'
        self.job = None
        self.active = False
        self.playing = False
        self.report()

    def report(self, error=None):
        self.emit({'state': self.state, 'index': self.index + 1,
                   'total': len(self.chunks), 'text': self.chunks[self.index],
                   'engine': settings.tts_engine(), 'error': error})

    def command(self, command):
        if command == 'p':
            if not self.active:
                self.active = True
                self.state = 'loading'
                self.report()
            return
        if command not in 'asnb':
            return
        self.active = False
        self.playing = False
        if self.job:
            self.job.cancel()
        self.player.stop()
        if command == 's':
            self.index = 0
        elif command == 'n':
            self.index = min(self.index + 1, len(self.chunks) - 1)
        elif command == 'b':
            self.index = max(0, self.index - 1)
        self.state = 'paused' if command == 'a' else 'ready'
        self.report()

    def tick(self):
        if self.job and self.job.done.is_set():
            job, self.job = self.job, None
            if self.active and not job.cancelled:
                if job.error:
                    self.active = False
                    self.state = 'error'
                    self.report(job.error)
                else:
                    self.player.play(*job.clip)
                    self.playing = True
                    self.state = 'playing'
                    self.report()
        if self.playing and not self.player.playing:
            self.playing = False
            if self.player.error:
                self.active = False
                self.state = 'error'
                self.report(self.player.error)
            elif self.index + 1 == len(self.chunks):
                self.active = False
                self.index = 0
                self.state = 'finished'
                self.report()
            else:
                self.index += 1
        if self.active and not self.playing and self.job is None:
            self.state = 'loading'
            self.report()
            self.job = Synthesis(self.chunks[self.index], self.factory)

    def close(self):
        if self.job:
            self.job.cancel()
        self.player.close()
        if self.job:
            self.job.thread.join(timeout=2)


def run(filename):
    def emit(event):
        print(json.dumps(event), flush=True)
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    reader = None
    try:
        emit({'state': 'opening'})
        reader = Reader(load_document(filename), emit)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], .03)
            if ready:
                commands = os.read(sys.stdin.fileno(), 1024)
                if not commands or b'q' in commands:
                    break
                for command in commands.decode('ascii', errors='ignore'):
                    reader.command(command)
            reader.tick()
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    except Exception as error:
        emit({'state': 'error', 'error': str(error)})
        return 1
    finally:
        if reader:
            reader.close()
        signal.signal(signal.SIGTERM, previous)
