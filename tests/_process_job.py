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
        self.api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.api.OpenProcess.restype = w.HANDLE
        self.api.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
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
            self.close()
            raise error

    def assign(self, proc):
        with self._lock:
            if self.handle is None:
                raise OSError('probe job already closed')
            if not self.api.AssignProcessToJobObject(self.handle, int(proc._handle)):
                raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        # Transfer sole ownership under the lock before calling Win32. Closing
        # outside it is safe: no other caller can obtain this handle again,
        # and assign holds the same lock while using it. A concurrent close
        # may return before CloseHandle finishes; run_probe joins its watchdog
        # before main-path cleanup needs the tree to have ended.
        with self._lock:
            handle = self.handle
            self.handle = None
        if handle:
            members = []
            try:
                members = self._member_handles(handle)
                self._end_members(handle)
                for member in members:
                    if self.api.WaitForSingleObject(member, 5000) != 0:
                        raise TimeoutError('probe process did not finish exiting')
            finally:
                self.api.CloseHandle(handle)
                for member in members:
                    self.api.CloseHandle(member)

    def _member_handles(self, handle):
        # Job accounting can reach zero before process handles are signaled.
        # Retain identities before termination so cleanup can join them too.
        capacity = 16
        while True:
            class Pids(ctypes.Structure):
                _fields_ = [('assigned', w.DWORD), ('count', w.DWORD),
                            ('pids', ctypes.c_size_t * capacity)]
            pids = Pids()
            if self.api.QueryInformationJobObject(handle, 3, ctypes.byref(pids),
                                                 ctypes.sizeof(pids), None):
                break
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity *= 2
        members = []
        try:
            for pid in pids.pids[:pids.count]:
                member = self.api.OpenProcess(0x100000, False, pid)
                if member:
                    members.append(member)
                elif ctypes.get_last_error() != 87:  # Already exited.
                    raise ctypes.WinError(ctypes.get_last_error())
            return members
        except BaseException:
            for member in members:
                self.api.CloseHandle(member)
            raise

    def _end_members(self, handle):
        """Wait for job membership to drain before releasing its last handle.

        Kill-on-close requests termination; it does not join descendants.
        Keep the job queryable until Windows reports no active processes.
        """
        class Accounting(ctypes.Structure):
            _fields_ = [('times', ctypes.c_int64 * 4), ('faults', w.DWORD),
                        ('total', w.DWORD), ('active', w.DWORD), ('terminated', w.DWORD)]
        if not self.api.TerminateJobObject(handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        # Completed cleanup measured 0.023 / 0.020 s beyond the 5 s timeout
        # (live/exited parent), Windows 2026-11-09. Allow 5 s for termination,
        # far above twice either measured overhead.
        deadline = time.monotonic() + 5
        while True:
            accounting = Accounting()
            if not self.api.QueryInformationJobObject(
                    handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.active == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('probe job still has active processes after termination')
            time.sleep(0.01)
