"""Desktop-owned Piper greeting, cached health warnings, and provider keepalive."""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time

from . import audio, licensing, models, shutdown, tts


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
    if not ready:
        return False
    command = os.read(sys.stdin.fileno(), 1)
    return 'goodbye' if command == b'g' else not command


def health_monitor():
    try:
        from kilix_sdk.health import HealthMonitor
    except ImportError as error:
        raise tts.TtsError('System alerts require the matching Kilix host; use kilix tts --system-voice.') from error
    return HealthMonitor()


def play_clip(player, clip):
    player.play(*clip)
    if not player.wait(60):
        raise tts.TtsError('System voice playback timed out.')
    if player.error:
        raise tts.TtsError(player.error)


def run(greeting='hello', *, wait=wait_for_owner, loaded=provider_loaded,
        engine_factory=tts.PiperTts, player_factory=audio.Player,
        monitor_factory=health_monitor, shutdown_factory=shutdown.connect,
        clock=time.monotonic):
    greeting = validate_greeting(greeting)
    licensing.require_covering_receipt(models.PIPER_KRISTIN_MODEL)
    def stop(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    engine = player = monitor = shutdown_watch = None
    try:
        engine = engine_factory(rate=170)
        player = player_factory({})
        # An empty greeting still loads the model, without making a sound.
        clip = engine.synth(greeting or 'hello')
        if greeting:
            play_clip(player, clip)
        monitor = monitor_factory() if monitor_factory is not None else None
        # Prepare fixed warnings once; incident playback needs no synthesis.
        warning_clips = {phrase: engine.synth(phrase) for phrase in monitor.phrases} if monitor else {}
        goodbye_clip = engine.synth('Goodbye')
        shutdown_watch = shutdown_factory() if shutdown_factory else None
        print('System voice ready: Piper Kristin', flush=True)
        next_keepalive = clock()+60
        next_health = clock()
        goodbye_at = -float('inf')
        while True:
            event = wait(.25 if shutdown_watch else 2 if monitor else 60)
            if event is True:  # Owner EOF is cleanup, not machine shutdown.
                break
            shutting_down = shutdown_watch.poll() if shutdown_watch else False
            if event == 'goodbye' or shutting_down:
                try:
                    # A menu request followed by logind's signal is one goodbye.
                    if clock()-goodbye_at >= 10:
                        play_clip(player, goodbye_clip)
                        goodbye_at = clock()
                    print('System voice goodbye complete', flush=True)
                finally:
                    if shutting_down:
                        shutdown_watch.release()
                continue
            if monitor and (shutdown_watch is None or clock() >= next_health):
                for phrase in monitor.poll():
                    if phrase in warning_clips:
                        play_clip(player, warning_clips[phrase])
                next_health = clock()+2
            if clock() >= next_keepalive or (monitor is None and shutdown_watch is None):
                # Recovery stays silent. Alerts use cached PCM even if the
                # shared provider has restarted in the meantime.
                if not loaded():
                    engine.synth('hello')
                next_keepalive = clock()+60
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous)
        if shutdown_watch is not None:
            shutdown_watch.close()
        if monitor is not None:
            monitor.close()
        if player is not None:
            player.close()
        if hasattr(engine, 'close'):
            engine.close()
    return 0
