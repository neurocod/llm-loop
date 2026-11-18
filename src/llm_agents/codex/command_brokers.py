"""Starting a broker of the bridge's own checkout: `python -I <its file> ...`.

A broker (`llm_agents.git.rebase`, `llm_agents.git.commit_broker`) is the
one way a sandboxed worker runs git outside the sandbox. An escalated
command runs with the user's full rights and the judge can only read it, not
rewrite it, so what it grants is a file the worker cannot edit — the
bridge's OWN copy of the broker, outside every writable root — started so
that nothing the worker controls runs before it: spelled literally
(`broker_script`), under `-I`, by a `python` the PATH cannot swap
(`trusted_python`), with no link along the file's path
(`broker_launch_refusal`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..links import is_link
from .command_arguments import CommandContext
from .command_paths import within


@dataclass(frozen=True)
class Broker:
    """One broker the python shape knows: its file as the bridge imported it
    (resolved), how refusals name it, and the judge of its arguments."""
    path: Path
    what: str
    judge: Callable[[list[str], Path, CommandContext], str]


def _spelled(path: str) -> str:
    """Letter case and (on Windows) `/` folded — and nothing else: no
    `normpath`, which drops `..` lexically, while on POSIX the kernel walks
    `j/..` through whatever `j` links to (review 2026-11-18:
    `/srv/main-wt1/j/../../main/<broker>` with `j` a symlink normalised to
    the broker and opened a worker file)."""
    return os.path.normcase(path.replace("/", os.sep))


def _dotted(path: str) -> bool:
    """Whether a path holds a `.` or `..` segment or a redundant separator
    (an empty segment past the root)."""
    return any(part in ("", ".", "..")
               for part in path.replace("/", os.sep).split(os.sep)[1:])


def broker_script(word: str, broker: Broker) -> bool:
    """Whether a `python <word>` names `broker` — by its LITERAL spelling:
    the one absolute canonical path of the bridge's file (letter case and
    separators folded, nothing resolved). A spelling that only RESOLVES to
    it (a junction or symlink in the worker's tree, `..` through one) is not
    the broker: the worker could swap the link between the grant and Python
    opening the file, and run its own code with the user's rights (review
    2026-11-18). Such a word is judged as any other script, unless it is a
    dotted spelling of the broker (`broker_spelling_refusal`)."""
    return (os.path.isabs(word) and not _dotted(word)
            and _spelled(word) == _spelled(str(broker.path)))


def broker_spelling_refusal(word: str, broker: Broker) -> str:
    """Refuse outright a script word with `.`/`..` segments or a redundant
    separator that LEXICALLY reads as the broker: it is no spelling of the
    broker the judge accepts, and what it opens depends on the links along
    it, not on the text."""
    if _dotted(word) and _spelled(os.path.normpath(word)) == _spelled(str(broker.path)):
        return (f"script {word}: spell {broker.what} exactly as {broker.path} "
                f"(no `.`/`..` segments, no doubled separators)")
    return ""


def _linked(path: Path) -> Optional[Path]:
    """The first of `path` and its parents, AS SPELLED (nothing resolved),
    that is a symlink or junction, or None."""
    for folder in (path, *path.parents):
        if is_link(folder):
            return folder
    return None


def _dotted_entry(folder: str) -> bool:
    """Whether a PATH entry holds a `.`/`..` segment or STARTS with a doubled
    separator (a UNC share, POSIX's implementation-defined `//`). A doubled
    separator inside the entry is one separator to both kernels and names
    no link, so it is not refused: Windows runners ship `C:\\\\ghcup\\bin`
    on their PATH (CI 2026-11-18)."""
    parts = folder.rstrip("\\/").replace("/", os.sep).split(os.sep)
    return (any(part in (".", "..") for part in parts)
            or (len(parts) > 1 and parts[0] == "" and parts[1] == ""))


def _swappable_link(path: Path, depth: int = 0) -> Optional[Path]:
    """The first link or junction along `path` AS SPELLED, from the root
    down, that this user could repoint, or None.

    A link is as fixed as the folder holding it: escalated with the user's
    rights, the worker can repoint only what the user may write. So a link
    in a folder `os.access` says this user may not write — `/sbin ->
    usr/sbin` under a root-owned `/`, on every usr-merged Linux — stays, and
    its target is judged the same way (a `..` in it refuses: what it names
    depends on links again). On Windows `os.access` reads no ACL, so every
    junction there counts as swappable: the rule of the review of 2432
    unchanged."""
    for folder in (*reversed(path.parents), path):
        if not is_link(folder):
            continue
        if os.name == "nt" or depth > 8 or os.access(folder.parent, os.W_OK):
            return folder
        try:
            target = Path(os.readlink(folder))
        except OSError:
            return folder
        if not target.is_absolute():
            target = folder.parent / target
        if ".." in target.parts or _swappable_link(target, depth + 1) is not None:
            return folder
    return None


def _program_names(name: str) -> tuple[str, ...]:
    """The file names a shell's lookup of `name` may open in one PATH folder:
    on Windows each PATHEXT extension and PowerShell's `.ps1`, on POSIX the
    name itself."""
    if os.name != "nt":
        return (name,)
    extensions = [ext.strip() for ext in
                  os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")]
    return tuple(name + ext for ext in dict.fromkeys(
        [ext.lower() for ext in extensions if ext] + [".ps1"]))


def trusted_python(name: str, roots: tuple[Path, ...],
                   path: Optional[str] = None) -> tuple[Optional[Path], str]:
    """The interpreter the escalated PowerShell starts for the word `name`,
    or None and why it is not pinned.

    The worker types a NAME (`argv_refusal`: programs are named, not given
    by path), which PowerShell looks up on the PATH the bridge hands every
    escalated command. Pinned like `gitlib.trusted_git`, with one difference:
    the bridge resolves git and runs it by its absolute path, while here the
    shell resolves the word, after the grant — so not only the first hit
    counts. Refused: any relative PATH entry (the cwd in disguise) and any
    entry inside a writable root, wherever it stands (a later one still
    serves `python.cmd`, `python3`, or a name the first folder lacks after
    the worker plants it); on Windows a first folder holding anything but a
    lone `<name>.exe` (a `.cmd`/`.bat`/`.ps1` beside it may win the lookup
    and is a script anyone with the folder may rewrite)."""
    found: Optional[Path] = None
    for raw in (path if path is not None else os.environ.get("PATH", "")).split(os.pathsep):
        folder = raw.strip().strip('"')
        if not folder:
            continue
        if not os.path.isabs(folder):
            return None, (f"PATH entry {raw!r} is relative: the shell would look "
                          f"`{name}` up in the working directory")
        # Judged as SPELLED, not only resolved: the shell walks this text
        # after the grant, so a junction along it (`C:\slot\python-link` ->
        # the real Python) resolves clean at the judge and is the worker's
        # to swap before PowerShell looks `name` up (review of 2432).
        if _dotted_entry(folder):
            return None, (f"PATH entry {folder} has `.`/`..` segments or starts with "
                          f"a doubled separator: what it names depends on the links "
                          f"along it")
        link = _swappable_link(Path(folder))
        if link is not None:
            return None, (f"PATH entry {folder} passes through the link or junction "
                          f"{link}: it may be swapped after the grant")
        if within(Path(folder), roots) or within(Path(folder).resolve(), roots):
            return None, (f"PATH entry {folder} lies in your writable roots: a "
                          f"`{name}` planted there would run")
        if found is not None:
            continue
        hits = [Path(folder) / file for file in _program_names(name)
                if (Path(folder) / file).is_file()]
        if os.name != "nt":
            hits = [hit for hit in hits if os.access(hit, os.X_OK)]
        if not hits:
            continue
        if os.name == "nt" and [hit.name.lower() for hit in hits] != [f"{name}.exe"]:
            return None, (f"`{name}` on PATH is {', '.join(hit.name for hit in hits)} "
                          f"in {folder}: only a lone {name}.exe is pinned")
        found = hits[0].resolve()
    if found is None:
        return None, f"no `{name}` on the bridge's PATH"
    if within(found, roots):
        return None, f"the `{name}` on PATH, {found}, lies in your writable roots"
    return found, ""


def broker_launch_refusal(program: str, switches: list[str], ctx: CommandContext,
                          broker: Broker) -> str:
    """Why `<program> <switches> <broker>` may not start the broker, or "":
    `program`, as typed (`CommandContext.program_word`), is `python`/`python3`
    that `trusted_python` pins (not `py`: the launcher picks an interpreter by
    its own rules — registry, py.ini, PY_PYTHON — after the grant; not
    `python.cmd`/`.bat`/`.ps1`, a script the shell finds wherever it lies, nor
    `python.exe`, which `trusted_python` does not judge); `-I` is required (isolated mode: no
    PYTHON* variables, no user site, no script folder from the cwd — a
    `sitecustomize.py` on an inherited PYTHONPATH runs before any check of
    the broker's); and no folder along the broker's path may be a link or
    junction or lie in a writable root: its literal spelling must name the
    file it names now and after the grant."""
    if program.lower() not in ("python", "python3"):
        return (f"{broker.what} starts as `python -I {broker.path} ...`, not "
                f"through `{program}`")
    _, why = trusted_python(program.lower(), ctx.roots)
    if why:
        return f"{broker.what}: {why}"
    if "-I" not in switches:
        return (f"{broker.what} starts only isolated: `python -I {broker.path} "
                f"...` (an inherited PYTHONPATH could plant code before it runs)")
    link = _linked(broker.path)
    if link is not None:
        return f"{broker.what}'s path holds a link or junction at {link}"
    for folder in (broker.path, *broker.path.parents):
        if within(folder, ctx.roots):
            return f"{broker.what} {broker.path} lies in your writable roots"
    return ""
