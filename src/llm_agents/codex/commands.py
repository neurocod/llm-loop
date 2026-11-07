"""Reading a command an agent asks to run outside the sandbox.

Codex's shell wrapper (`split_shell`), the words PowerShell or cmd would hand
the program (`command_shell._words`), the paths those words name (`path_refusal`) — and, on
top of them, the worker's ALLOWLIST (`worker_command_refusal`, the host's
`WorkerCommands.shapes`). The file-change judge in sandbox.py shares the
path half. The facade imports policies; no policy imports sandbox.py.

Trust model (orchestrator decision 2026-11-03, after the third review):
the judge protects a COOPERATIVE worker from ordinary mistakes — git
mutation, writes or deletes outside its roots or into the host's
`protected_parts`, and whatever the host's own judges add (an online
workload without `--allow-online-probes`, a write into a directory the host
shares). It does not defend against a worker crafting an evasion. Accepted
residuals the judge does not read: npm's `.npmrc` and environment
configuration (`script-shell` and the like), `PYTEST_ADDOPTS` and ini files,
the contents of response files (for the programs whose `@` is not refused
outright), side effects inside node/tsx script bodies, the git hooks a
slot's commit runs (`core.hooksPath` is the tree's own `.githooks/`, as
every listed script is the tree's own file), and what the host names as its
own residuals beside its table.

Operator's decision 2026-11-03 (after both reviews of the denylist that
preceded this): a worker's command leaves the sandbox only when every
part of it is a KNOWN SHAPE, and anything the reader does not understand is
a "no". The denylist read the command for forbidden words and granted the
rest, and the reviews granted themselves `Set-Content`/`Copy-Item` into the
main tree, `& "Remove-Item"`, `-LiteralPath:C:/outside`, `@('a','C:\\x')`,
`git -C "a b" commit`, `-c alias.ci=commit`, `python -m tools.git.commit` —
a list of what is forbidden is never finished; a list of what is allowed is.
The cost of a wrong "no" is a step the agent does inside the sandbox or asks
the orchestrator for; a wrong "yes" runs with the user's full rights.
"""

from .command_arguments import (
    CommandContext,
)
from .command_python import (
    Entrypoint,
)
from .command_node import (
    NpmScript,
    npm_script,
)
from .command_paths import (
    PathRefused,
    path_refusal,
    protected_part,
    resolve_path,
    within,
)
from .command_dispatch import (
    Shape,
    argv_refusal,
    shapes_text,
    worker_command_refusal,
)
from .command_shell import (
    Token,
    Unparsed,
    Word,
    split_shell,
    unwrap_shell,
)
from .command_commit import (
    pool_slot,
)

__all__ = ['CommandContext', 'Entrypoint', 'NpmScript', 'PathRefused', 'Shape', 'Token', 'Unparsed', 'Word', 'argv_refusal', 'npm_script', 'path_refusal', 'pool_slot', 'protected_part', 'resolve_path', 'shapes_text', 'split_shell', 'unwrap_shell', 'within', 'worker_command_refusal']
