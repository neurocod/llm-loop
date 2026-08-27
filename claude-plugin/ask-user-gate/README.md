# ask-user-gate

A `PreToolUse` hook for Claude Code that refuses shell commands which would
stop the session on a permission prompt, and names the replacement for each one
instead of parking the run on a dialog nobody is watching.

It refuses seven shapes, all of which defeat the allow-list rather than merely
missing it: a chain that also moves the working directory, backgrounding `&`,
a command over the analyser's 10 000-character limit, a heredoc or PowerShell
here-string (a file written by the command body), `sed -i`, a quote inside
unquoted `{ }`, and waiting by the clock (`sleep`, usually inside a `while`).
Why each one is on the list — and why the list is deliberately short — is in the
module docstring of `hooks/ask_user_gate.py`.

Escape hatch: put `allowAskUser` anywhere in the command (a trailing
`# allowAskUser` is a comment in both shells) and it passes through unchanged.

## Install

```
/plugin marketplace add neurocod/llm-loop
/plugin install ask-user-gate@neurocod
```

Nothing else is configured: `hooks/hooks.json` names the script through
`${CLAUDE_PLUGIN_ROOT}`, so one checkout guards every project on the machine
and no absolute path is written down anywhere. Requires `python` on PATH.

## What ships in `bin/`

`replace_in_file.py` and `try_patch.py` — the two scripts the refusals send the
caller to. They travel with the gate on purpose: advice naming a path that does
not exist on this machine costs the reader a search that ends in nothing. A
repository that keeps its own copies under `tools/` is told the short path it
already knows instead (`tool_path()` in the gate decides).

Both are useful on their own. `replace_in_file.py` is a checked stand-in for
`sed -i`: it refuses to write unless the number of matches is the one you named,
so a wrong pattern fails instead of silently editing nothing and exiting 0.
`try_patch.py` mutates a file, runs a command and restores the file from a
`finally` — with `--expect-fail` for the usual case, proving that a test really
does fail without its fix.

## Self-test

```
python hooks/ask_user_gate.py --self-test
python bin/try_patch.py --selftest
```

The first covers the scanner, the wiring (every tool the scanner handles must
also be in the matcher of `hooks/hooks.json` — a gate that guards its logic but
not its wiring has a hole exactly the shape of the one that happened), and both
branches of the path resolver.
