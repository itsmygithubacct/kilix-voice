"""Hold a new group leader until its parent has verified its identity.

Only built-in/frozen modules are needed on the isolated interpreter's hot
path. The private descriptor closes on exec. Marshal is used only between
this trusted launcher and its parent, running the same interpreter.
"""
import marshal
import posix as os
import _signal as signal
import sys


def read_exact(fd, count):
    data = bytearray()
    while len(data) < count:
        part = os.read(fd, count - len(data))
        if not part:
            raise EOFError('parent closed launch channel')
        data.extend(part)
    return data


def write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def main():
    channel = int(sys.argv[1])
    try:
        os.set_inheritable(channel, False)
        write_all(channel, b'R')
        size = int.from_bytes(read_exact(channel, 8), 'big')
        request = marshal.loads(read_exact(channel, size))
        if request['restore_signals']:
            for name in ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ'):
                if hasattr(signal, name):
                    signal.signal(getattr(signal, name), signal.SIG_DFL)
        write_all(channel, b'X')
        saved_error = last_error = None
        for executable in request['executables']:
            try:
                os.execve(executable, request['argv'], request['env'])
            except OSError as error:
                last_error = error
                if not isinstance(error, (FileNotFoundError, NotADirectoryError)) and saved_error is None:
                    saved_error = error
        raise saved_error or last_error
    except BaseException as error:
        failure = marshal.dumps((type(error).__name__, getattr(error, 'errno', None), str(error)))
        try:
            write_all(channel, b'F' + failure)
        except OSError:
            pass  # parent refused identity or died before admission
        return 127


if __name__ == '__main__':
    sys.exit(main())
