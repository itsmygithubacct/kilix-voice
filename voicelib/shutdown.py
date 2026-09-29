"""Logind shutdown notification with a short, bounded delay inhibitor.

The system owns the delay limit. Releasing the FD after cached speech lets
shutdown proceed; abrupt power loss and forced shutdown cannot be announced.
"""
import os
import sys


class ShutdownWatch:
    def __init__(self, bus, manager, context):
        self.bus, self.manager, self.context = bus, manager, context
        self.fd = None
        self.requested = False
        self.match = bus.add_signal_receiver(
            self.prepare, signal_name='PrepareForShutdown',
            dbus_interface='org.freedesktop.login1.Manager',
            bus_name='org.freedesktop.login1', path='/org/freedesktop/login1')
        try:
            self.acquire()
        except Exception:
            self.close()
            raise

    def acquire(self):
        if self.fd is None:
            self.fd = self.manager.Inhibit(
                'shutdown', 'Kilix system voice', 'Say Goodbye', 'delay',
                timeout=1).take()

    def prepare(self, active):
        self.requested = bool(active)
        if not active:  # A canceled shutdown must leave the next one covered.
            try:
                self.acquire()
            except Exception as error:
                print(f'System voice shutdown delay unavailable: {error}', file=sys.stderr)

    def poll(self):
        for _ in range(32):
            if not self.context.pending():
                break
            self.context.iteration(False)
        requested, self.requested = self.requested, False
        return requested

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def close(self):
        self.release()
        self.match.remove()
        self.bus.close()


def connect():
    """Optional on generic hosts; Plebian-OS supplies these system bindings."""
    try:
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop
        from gi.repository import GLib
        DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus(private=True)
        try:
            manager = dbus.Interface(bus.get_object(
                'org.freedesktop.login1', '/org/freedesktop/login1'),
                'org.freedesktop.login1.Manager')
            return ShutdownWatch(bus, manager, GLib.MainContext.default())
        except Exception:
            bus.close()
            raise
    except Exception as error:
        print(f'System voice OS shutdown notification unavailable: {error}', file=sys.stderr)
        return None
