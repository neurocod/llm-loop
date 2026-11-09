"""Isolate real signal transition tests from pytest's own signal state."""
import logging
import signal
import socket
import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_loop import ctrlc


def main(case):
    before = signal.getsignal(signal.SIGINT)
    old_fd = signal.set_wakeup_fd(-1)
    sockets = []
    pair = socket.socketpair

    def tracked_pair():
        result = pair()
        sockets.extend(result)
        return result

    def capture():
        try:
            with ctrlc.captured() as interrupt:
                yield interrupt
        except SystemExit as exc:
            assert exc.code == 130

    import contextlib
    capture = contextlib.contextmanager(capture)
    try:
        with patch.object(socket, 'socketpair', tracked_pair):
            if case == 'activation':
                start = threading.Thread.start
                def injected_start(thread):
                    if thread.name == 'ctrlc-wakeup':
                        signal.raise_signal(signal.SIGINT)
                    return start(thread)
                with patch.object(threading.Thread, 'start', injected_start):
                    with capture() as interrupt:
                        assert interrupt.presses == 1, interrupt.presses
                assert interrupt.presses == 1, interrupt.presses
            elif case in ('shutdown_signals', 'interrupted_join'):
                close = ctrlc._SignalWakeup.close
                join = threading.Thread.join
                injected = []
                def injected_close(wakeup):
                    result = close(wakeup)
                    if case == 'shutdown_signals':
                        # Also cover close returning with both sockets closed,
                        # while captured still has its SIGINT callback installed.
                        signal.raise_signal(signal.SIGINT)
                    return result
                def injected_join(thread, *args, **kwargs):
                    if thread.name == 'ctrlc-wakeup' and not injected:
                        injected.append(True)
                        if case == 'interrupted_join':
                            raise KeyboardInterrupt('injected join interruption')
                        signal.raise_signal(signal.SIGINT)
                    return join(thread, *args, **kwargs)
                with patch.object(ctrlc._SignalWakeup, 'close', injected_close), \
                        patch.object(threading.Thread, 'join', injected_join):
                    try:
                        with capture() as interrupt:
                            pass
                    except KeyboardInterrupt:
                        assert case == 'interrupted_join'
                if case == 'shutdown_signals':
                    assert interrupt.presses == 2, interrupt.presses
            elif case in ('foreign_owner', 'replacement'):
                reader, writer = pair()
                reader.settimeout(1)  # A ready signal byte; no asynchronous wait.
                writer.setblocking(False)
                try:
                    if case == 'foreign_owner':
                        signal.set_wakeup_fd(writer.fileno())
                    with ctrlc.captured():
                        if case == 'replacement':
                            signal.set_wakeup_fd(writer.fileno())
                        else:
                            previous_break = signal.signal(signal.SIGBREAK, lambda *_: None)
                            try:
                                signal.raise_signal(signal.SIGBREAK)
                                assert reader.recv(1) == bytes([signal.SIGBREAK])
                            finally:
                                signal.signal(signal.SIGBREAK, previous_break)
                    assert signal.set_wakeup_fd(-1) == writer.fileno()
                finally:
                    signal.set_wakeup_fd(-1)
                    reader.close()
                    writer.close()
            elif case in ('socket_failure', 'fd_failure'):
                target = (patch.object(socket, 'socketpair', side_effect=OSError('no sockets'))
                          if case == 'socket_failure' else
                          patch.object(signal, 'set_wakeup_fd', side_effect=ValueError('no fd')))
                with target, patch.object(logging.Logger, 'warning') as warning:
                    with capture() as interrupt:
                        signal.raise_signal(signal.SIGINT)
                        assert interrupt.presses == 1
                    assert warning.call_count == 1
            else:
                raise AssertionError(case)
        assert ctrlc._active is None, 'capture leaked its active Interrupt'
        assert signal.getsignal(signal.SIGINT) is before
        assert signal.set_wakeup_fd(-1) == -1, 'capture leaked its wakeup fd'
        assert all(sock.fileno() == -1 for sock in sockets), 'capture leaked sockets'
        assert not any(thread.name == 'ctrlc-wakeup' for thread in threading.enumerate())
    finally:
        signal.signal(signal.SIGINT, before)
        signal.set_wakeup_fd(old_fd)


if __name__ == '__main__':
    main(sys.argv[1])
