"""Desktop-owned Piper session: greet once, then keep its shared provider warm."""
from __future__ import annotations

import json
import select
import signal
import subprocess
import sys

from . import audio, licensing, models, tts


def validate_greeting(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > 1024 or '\0' in text:
        raise ValueError('Startup greeting must be at most 1024 UTF-8 bytes.')
    return text.strip()


def provider_loaded():
    binary = tts.piper_binary()
    if not binary:
        raise tts.TtsError('Piper is not installed. Enable system voice in Settings first.')
    result = subprocess.run([binary, 'status', '--json'], capture_output=True,
                            text=True, timeout=10, check=True)
    return bool(json.loads(result.stdout).get('loaded'))


def wait_for_owner(seconds):
    """The desktop owns stdin; closing it releases this session on desktop exit."""
    ready, _, _ = select.select([sys.stdin], [], [], seconds)
    return bool(ready and not sys.stdin.read(1))


def run(greeting='hello', *, wait=wait_for_owner, loaded=provider_loaded,
        engine_factory=tts.PiperTts, player_factory=audio.Player):
    greeting = validate_greeting(greeting)
    licensing.require_covering_receipt(models.PIPER_KRISTIN_MODEL)
    def stop(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    engine = player = None
    try:
        engine = engine_factory(rate=170)
        player = player_factory({})
        # An empty greeting still loads the model, without making a sound.
        clip = engine.synth(greeting or 'hello')
        if greeting:
            player.play(*clip)
            if not player.wait(60):
                raise tts.TtsError('Startup greeting playback timed out.')
            if player.error:
                raise tts.TtsError(player.error)
        print('System voice ready: Piper Kristin', flush=True)
        while not wait(60):
            # Status refreshes the provider's idle timer. Recover a restarted
            # provider silently; the greeting belongs to desktop startup only.
            if not loaded():
                engine.synth('hello')
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous)
        if player is not None:
            player.close()
        if hasattr(engine, 'close'):
            engine.close()
    return 0
