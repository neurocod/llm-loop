"""The judge of the rebase broker (`llm_agents.git.rebase`) for a worker in a
pool slot.

The worker never gets `git` itself outside the sandbox: an escalated command
runs with the user's full rights, and the judge can only read a command, not
rewrite it — it cannot turn hooks off, pin the executable or force the
configuration of a `git` the worker typed (the broker's docstring lists
what each of those would let through). What it grants instead is `python
<BROKER> <op> ...`, where BROKER is the broker file of the bridge's OWN
checkout: outside every writable root, so the code that runs is not the
worker's to edit. The broker takes its facts from the environment the
bridge hands every escalated command (`Grant`, set up by
`sandbox.escalation_judge`), not from the command line, and checks the
request again (`rebase.refusal`) right before git runs.
"""

from __future__ import annotations

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


def broker_script(word: str, cwd: Path) -> bool:
    """Whether a `python <word>` names the bridge's broker file."""
    try:
        return (cwd / word).resolve() == BROKER
    except (OSError, ValueError):
        return False
