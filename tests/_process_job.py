"""Windows probe-tree lifetime, independent of its parent's lifetime."""
import ctypes
import threading
import time
from ctypes import wintypes as w


class Job:
    def __init__(self):
        self._lock = threading.Lock()
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', w.DWORD), ('minimum', ctypes.c_size_t),
                        ('maximum', ctypes.c_size_t), ('active', w.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', w.DWORD),
                        ('scheduling', w.DWORD)]
        class Limits(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', ctypes.c_uint64 * 6),
                        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                        ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.api.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.api.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int,
                                                       ctypes.c_void_p, w.DWORD,
                                                       ctypes.c_void_p]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close(time.monotonic())
            raise error

    def assign(self, proc):
        with self._lock:
            if self.handle is None:
                raise OSError('probe job already closed')
            if not self.api.AssignProcessToJobObject(self.handle, int(proc._handle)):
                raise ctypes.WinError(ctypes.get_last_error())

    def close(self, deadline):
        # Swap under the lock; only the winner owns the native handle.
        with self._lock:
            handle, self.handle = self.handle, None
        if handle:
            try:
                self._end_members(handle, deadline)
            finally:
                self.api.CloseHandle(handle)

    def _end_members(self, handle, deadline):
        """Wait for job membership to drain before releasing its last handle.

        Kill-on-close requests termination; it does not join descendants.
        Keep the job queryable until Windows reports no active processes.
        """
        class Accounting(ctypes.Structure):
            _fields_ = [('times', ctypes.c_int64 * 4), ('faults', w.DWORD),
                        ('total', w.DWORD), ('active', w.DWORD), ('terminated', w.DWORD)]
        if not self.api.TerminateJobObject(handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        while True:
            accounting = Accounting()
            if not self.api.QueryInformationJobObject(
                    handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.active == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('probe job still has active processes after termination')
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))
