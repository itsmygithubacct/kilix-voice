"""Per-conversation PCM gain; unrelated to the system-announcement cap."""
from array import array
import sys
import threading


class PlaybackVolume:
    """A mutable gain attached to one owned turn, never the shared mixer."""

    def __init__(self, percent=100):
        self._lock = threading.Lock()
        self.set(percent)

    def set(self, percent):
        if type(percent) is not int or not 0 <= percent <= 100:
            raise ValueError("Playback volume must be an integer from 0 to 100")
        with self._lock:
            self._percent = percent

    @property
    def percent(self):
        with self._lock:
            return self._percent

    def apply(self, pcm):
        percent = self.percent
        if percent == 100:
            return pcm
        if percent == 0:
            return bytes(len(pcm))
        samples = array('h')
        samples.frombytes(pcm)
        if sys.byteorder != 'little':
            samples.byteswap()
        for index, sample in enumerate(samples):
            samples[index] = int(sample * percent / 100)
        if sys.byteorder != 'little':
            samples.byteswap()
        return samples.tobytes()
