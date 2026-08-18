"""
Example wrapper: the parallel counterpart of runFileList.py.

Same per-file work, but with N worker threads running the selected provider
concurrently
(`.main_parallel()`) instead of one file at a time. It subclasses runFileList.py's
driver (the same list path, target naming, prompt and model) and only relabels
the entry point; the list file is the single source of truth, guarded by one
lock, so the run stays idempotent.

CLI mirrors the family; the additions are `-j/--jobs N`, and `--max-runs N` caps
the *total* number of files processed this run, not iterations. The worker count
is the first of these that is set: `-j/--jobs`, the driver's `jobs` attribute
(below), the engine's default of 10.

The status line gives such a run one row per job — no wiring needed here; pass
`--no-statusline` for the plain scrolling output instead.

Copy this into your host project root next to runFileList.py and run
`python runFileListParallel.py -j 8`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "llm-loop", "src"))

from runFileList import LIST_FILE_REL, FileListDriver


class FileListParallelDriver(FileListDriver):
    # app_name (the mirror log's name) and prog (the --help name) are left unset
    # on purpose: both are derived from this wrapper's filename, so this entry
    # point gets its own log without repeating the name here.
    description = (f"Process the files listed in {LIST_FILE_REL} with N "
                   "concurrent LLM workers.")
    # A default worth pinning per task: long iterations are bounded by the shared
    # session budget rather than by CPU. An explicit -j/--jobs always wins.
    jobs = 4


if __name__ == "__main__":
    FileListParallelDriver.main_parallel()
