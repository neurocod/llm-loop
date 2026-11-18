"""The judge of the rebase broker (`llm_agents.git.rebase`) for a worker in a
pool slot.

The worker never gets `git` itself outside the sandbox: an escalated command
runs with the user's full rights, and the judge can only read a command, not
rewrite it — it cannot turn hooks off, pin the executable or force the
configuration of a `git` the worker typed (the broker's docstring lists
what each of those would let through). What it grants instead is `python -I
<BROKER> <op> ...`, where BROKER is the broker file of the bridge's OWN
checkout, spelled literally (`broker_script`): outside every writable root,
so the code that runs is not the worker's to edit. The broker takes its facts from the environment the
bridge hands every escalated command (`Grant`, set up by
`sandbox.escalation_judge`), not from the command line, and checks the
request again (`rebase.refusal`) right before git runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..git import rebase as broker
from ..git.slots import pool_slot
from .command_arguments import CommandContext
from .command_paths import protected_part, within

# The broker file the worker names, as the bridge imported it.
BROKER = Path(broker.__file__).resolve()


def _rebase_broker(args: list[str], cwd: Path, ctx: CommandContext) -> str:
    """`python <BROKER> <op> ...` from a worker: only in a pool slot
    (`pool_slot`), on a host naming its main branch, with the bridge's grant
    in place (`CommandContext.rebase`), the broker outside the worker's
    roots, and the request one `rebase.refusal` passes; the paths of
    `add`/`rm` also outside the host's `protected_parts`."""
    if not pool_slot(ctx.workspace):
        return ("the rebase broker: a worker rebases only its own pool slot "
                f"(`<repo>-wtN`, {ctx.policy.pool_script})")
    if ctx.rebase is None:
        return f"the rebase broker: {ctx.rebase_refusal or 'not granted here'}"
    if within(BROKER, ctx.roots):
        return (f"the rebase broker {BROKER} lies in your writable roots: the "
                f"bridge must run from a checkout the worker cannot edit")
    if not args:
        return f"the rebase broker with no operation ({', '.join(broker.OPERATIONS)})"
    op, rest = args[0], args[1:]
    why = broker.refusal(ctx.rebase, op, rest)
    if why:
        return why
    if op in ("add", "rm"):
        for raw in rest:
            protected = protected_part(Path(raw.replace("\\", "/")),
                                       ctx.policy.protected_parts)
            if protected:
                return f"rebase broker {op} {raw}: under {protected}/, which agents never write"
    return ""


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


def broker_script(word: str) -> bool:
    """Whether a `python <word>` names the broker — by its LITERAL spelling:
    the one absolute canonical path of the bridge's file (`BROKER`, letter
    case and separators folded, nothing resolved). A spelling that only
    RESOLVES to it (a junction or symlink in the worker's tree, `..` through
    one) is not the broker: the worker could swap the link between the grant
    and Python opening the file, and run its own code with the user's rights
    (review 2026-11-18). Such a word is judged as any other script, unless it
    is a dotted spelling of the broker (`broker_spelling_refusal`)."""
    return (os.path.isabs(word) and not _dotted(word)
            and _spelled(word) == _spelled(str(BROKER)))


def broker_spelling_refusal(word: str) -> str:
    """Refuse outright a script word with `.`/`..` segments or a redundant
    separator that LEXICALLY reads as the broker: it is no spelling of the
    broker the judge accepts, and what it opens depends on the links along
    it, not on the text."""
    if _dotted(word) and _spelled(os.path.normpath(word)) == _spelled(str(BROKER)):
        return (f"script {word}: spell the rebase broker exactly as {BROKER} "
                f"(no `.`/`..` segments, no doubled separators)")
    return ""


def broker_launch_refusal(switches: list[str], ctx: CommandContext) -> str:
    """Why `python <switches> <BROKER>` may not start the broker, or "":
    `-I` is required (isolated mode: no PYTHON* variables, no user site, no
    script folder from the cwd — a `sitecustomize.py` on an inherited
    PYTHONPATH runs before any check of the broker's), and no folder along
    BROKER's path may be a link or junction or lie in a writable root: its
    literal spelling must name the file it names now and after the grant."""
    if "-I" not in switches:
        return (f"the rebase broker starts only isolated: `python -I {BROKER} "
                f"<op>` (an inherited PYTHONPATH could plant code before it runs)")
    for folder in (BROKER, *BROKER.parents):
        if folder.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(folder):
            return f"the rebase broker's path holds a link or junction at {folder}"
        if within(folder, ctx.roots):
            return f"the rebase broker {BROKER} lies in your writable roots"
    return ""
