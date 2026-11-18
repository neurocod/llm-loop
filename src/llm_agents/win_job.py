"""Windows job-object primitives shared by every job in this package:
`child_job.ChildJob`, `cores.Job` and `git.live_workers.WorkerJob`.

The one step all three need the same way is getting a child INTO the job:
`start` (or `adopt`, for a caller that runs its own `Popen`) creates it
SUSPENDED, assigns it, and only then resumes it, so not even a grandchild
spawned in the child's first instruction can begin outside the job. A
`Popen` followed by an assignment leaves that window open, and what is
spawned in it outlives the job. A child that cannot be put in the job is
killed and reaped before the error is raised: it never runs uncontained.

Windows only: every function here needs `ctypes.wintypes`, imported inside
them so that importing this module stays harmless elsewhere (before Python
3.12 `ctypes.wintypes` raises on import off Windows).
"""
from __future__ import annotations

import subprocess
from typing import Optional, Sequence

CREATE_SUSPENDED = 0x00000004
KILL_ON_JOB_CLOSE = 0x00002000
BREAKAWAY_OK = 0x00000800
_EXTENDED_LIMIT_INFORMATION = 9


def kernel32():
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                            wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32


def set_limits(handle: int, flags: int) -> None:
    """Replace the job's limit flags with `flags` (the other limits stay
    zero: none of the jobs here sets one). OSError when Windows refuses."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic),
                    ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = Extended()
    info.BasicLimitInformation.LimitFlags = flags
    if not kernel32().SetInformationJobObject(handle, _EXTENDED_LIMIT_INFORMATION,
                                              ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())


def create(name: Optional[str], flags: int) -> int:
    """A new job handle (named `name`, or anonymous for None) with `flags`;
    OSError when Windows will not make one. The handle is not inheritable:
    a child holding it would keep a kill-on-close job alive."""
    import ctypes

    k32 = kernel32()
    handle = k32.CreateJobObjectW(None, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        set_limits(handle, flags)
    except OSError:
        k32.CloseHandle(handle)
        raise
    return handle


def adopt(handle: int, child) -> None:
    """Assign a `Popen` started with CREATE_SUSPENDED to the job and resume
    it. ANY way out before the resume succeeded — a refusal, a Ctrl+C, any
    BaseException — kills and reaps the child, then re-raises: the caller
    has no Popen yet to clean up (`start` has not returned), and a suspended
    child left behind, one in its own process group least of all, would
    outlive everyone."""
    import ctypes

    try:
        process = int(child._handle)
        if not kernel32().AssignProcessToJobObject(handle, process):
            raise ctypes.WinError(ctypes.get_last_error())
        # NtResumeProcess: the one call that resumes a whole process from its
        # handle (Popen keeps no thread handle to ResumeThread).
        if ctypes.WinDLL("ntdll").NtResumeProcess(ctypes.c_void_p(process)) != 0:
            raise OSError("cannot resume the child after putting it in the job")
    except BaseException:
        try:
            child.kill()
        except OSError:
            pass  # it has exited already
        child.wait()
        raise


def start(handle: int, argv: Sequence[str], popen=None, **options):
    """`popen(argv, **options)` started suspended, in the job, then resumed
    (`adopt`); the caller's own `creationflags` are kept. OSError from the
    start itself or from `adopt`. `popen` None: `subprocess.Popen`, looked
    up at the call, so a test patching it there sees the start."""
    options["creationflags"] = options.get("creationflags", 0) | CREATE_SUSPENDED
    child = (popen or subprocess.Popen)(argv, **options)
    adopt(handle, child)
    return child
