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
and no absolute path is written down anywhere.

Requires an interpreter named exactly `python` on PATH -- if a machine has only
`python3`, make `python` resolve to it. A wrapper picking among `python3`,
`python` and `py` was considered and declined (2026-08-27): the hook runs before
EVERY shell command, so the probe such a wrapper needs -- `command -v` alone
finds the Microsoft Store stub on Windows -- would put a second interpreter
start on that path, roughly doubling the cost of the gate to save one symlink.

## Update

An installed plugin is a snapshot, not a live checkout: it is copied into
`~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/`, and neither it nor
the marketplace clone follows the repository on its own. A push reaches a
machine only through

```
claude plugin marketplace update neurocod
claude plugin update ask-user-gate
```

and a restart, which the CLI asks for because Claude Code snapshots its hooks
at startup.

**Maintainers: bump `version` in `.claude-plugin/plugin.json` in the same
commit as any change to this plugin.** That version is what the cache directory
is named after, so a push that leaves it alone risks being a no-op on every
machine that already installed — silently, and the thing left running is a
security-adjacent gate. `claude plugin tag` cuts the matching git tag and
checks that the manifest and the marketplace entry still agree.

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

## The C++ port, for machines that call it a lot

`cpp/ask_user_gate.cpp` is the same gate compiled. It exists because the hook
runs before EVERY shell call, and measured on the author's machine that call
costs **69.8 ms as Python and 6.7 ms as the binary** (median of 60 hook-mode
runs each, identical payload) — at ~100 000 tool calls a month, about two hours
of wall clock spent starting an interpreter.

```
python cpp/build.py            # Release; writes hooks/ask_user_gate.exe
python cpp/parity_check.py     # the two halves must answer identically
```

The build needs CMake and a C++23 compiler (Visual Studio 2022 or newer brings
both). The binary lands in `hooks/`, next to `hooks.json`, and that is load-
bearing rather than tidy: `HERE` is the binary's own directory, and the
`../bin/<tool>` paths the refusals name — and the wiring self-test — resolve
from it. Then point the hook at the `.exe` instead of `python …/ask_user_gate.py`
and nothing else changes.

`hooks/hooks.json` keeps naming the script, because a plugin installed from the
marketplace is a snapshot that cannot build anything and may not be on Windows.
Using the binary is therefore a local opt-in: name it in the `settings.json`
that wires the gate for your checkout.

**`hooks/ask_user_gate.py` stays the reference.** It carries the rationale for
every rule and is the version to read and to change first; the port must then
follow. `cpp/parity_check.py` is what makes "must" a mechanism rather than a
hope — it runs the reference's own `SELF_TEST_CASES` plus a corpus aimed at the
seams (offsets, refusal text, quoting, the shlex fallback, multi-byte lengths)
through both halves and diffs verdict, exit code and rendered text. A rule
changed on one side alone fails it.

## Self-test

```
python hooks/ask_user_gate.py --self-test
hooks/ask_user_gate.exe --self-test
python bin/try_patch.py --selftest
```

The first two cover the scanner, the wiring (every tool the scanner handles must
also be in the matcher of `hooks/hooks.json` — a gate that guards its logic but
not its wiring has a hole exactly the shape of the one that happened), and both
branches of the path resolver.

A self-test alone is not enough to trust the port, and that is measured: a
deliberately broken redirect check passed `--self-test` 66/66 on both halves and
was caught only by `parity_check.py`, because the self-test asserts the verdict
and the bug was in the refusal text.
