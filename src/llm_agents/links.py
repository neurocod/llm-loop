"""Whether a path, as spelled, is a symlink or a Windows directory junction.

`os.path.isjunction` (and `Path.is_junction`) exist only from Python 3.12;
the package supports 3.9, where a missing attribute must not read as "no
junction": the checks that call this refuse a link along a path, so a
silent False there would accept the very junction they exist to refuse.
Stdlib only, so a broker started as `python -I <file>` may import it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Union


def is_junction(path: Union[str, Path]) -> bool:
    """Whether `path` itself (not followed) is a directory junction; False
    off Windows, where there are none, and for a path that does not exist."""
    if os.name != "nt":
        return False
    try:
        return os.lstat(path).st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    except OSError:
        return False


def is_link(path: Union[str, Path]) -> bool:
    """Whether `path` itself is a symlink or a junction."""
    return os.path.islink(path) or is_junction(path)
