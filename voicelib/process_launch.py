"""Hold a new group leader until its parent has verified its identity.

Executed directly with an isolated interpreter, before the requested program.
The private socket closes on exec; an errno reply preserves exec failures.
"""
import json
import os
import signal
import socket
import sys


def main():
    channel = socket.socket(fileno=int(sys.argv[1]))
    os.set_inheritable(channel.fileno(), False)
    channel.sendall(b"R")
    data = bytearray()
    while not data.endswith(b'\n'):
        part = channel.recv(65536)
        if not part:
            return 125  # parent refused identity or died before admission
        data.extend(part)
    request = json.loads(data)
    os.environ.pop('_KILIX_GROUP_NONCE', None)
    if request['restore_signals']:
        for name in ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ'):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_DFL)
    try:
        os.execvpe(request['executable'], request['argv'], request['env'])
    except OSError as error:
        channel.sendall(str(error.errno).encode('ascii'))
        return 127


if __name__ == '__main__':
    sys.exit(main())
