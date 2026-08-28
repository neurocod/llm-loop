#!/usr/bin/env python3
"""PreToolUse gate: refuse shell commands that would stop for a human prompt.

CLAUDE.md lists the shapes that make the permission analyser give up and ask a
human no matter what the allow-rules say -- a command longer than the parser's
limit, a file written by the command body, `sed -i`, and chains, of which this
refuses the ones that also move the working directory (see CD_COMMAND for why
the rest are let through). To those it adds a quote inside unquoted braces,
which the analyser reads as expansion obfuscation -- observed in a refusal, not
guessed, and waiting by the clock (`sleep`, usually inside a `while` loop),
which no allow-rule matches and which the tools make unnecessary anyway.
Prose only reminds; this gate is the mechanism. It runs before every call that
carries a shell command -- Bash, PowerShell and Monitor, which has its own
`command` field and runs it in the same local shell -- and a command that
matches one of those shapes is denied with the reason
and the repo's replacement for it, so the agent rewrites the call instead of
parking the session on a dialog nobody is watching.

The gate is deliberately narrow. It does NOT try to decide whether a command
matches the allow-list -- that is the permission system's job, and guessing at
it would deny far more than it saved. It only refuses commands that are known
to defeat the allow-list entirely.

Escape hatch: put `allowAskUser` anywhere in the command (a trailing
`# allowAskUser` is a comment in both shells) and the gate passes it straight
through. The prompt will then be asked -- that is the point of the marker: it
says "I know, and I want this command anyway".

It ships as a Claude Code PLUGIN rather than as a file inside one repository,
because what it encodes -- one product's permission analyser -- is the same on
every machine and in every project, while a path under `.claude/` is neither.
The wiring is the hooks.json next to this file, and it names this script through
${CLAUDE_PLUGIN_ROOT}, so installing the plugin is the whole install: no
absolute path is written down anywhere, and one checkout guards every project
on the machine. A missing script blocks every shell call until it is fixed
(loudly, with python's own "can't open file"), which is the failure mode to
prefer over a gate that silently stops guarding.

The two scripts its refusals send the caller to travel WITH it, in ../bin, for
the same reason: advice naming a path that does not exist on this machine costs
the reader a search that ends in nothing. See tool_path() for which of the two
copies gets named.

This file is the REFERENCE, and there is a second implementation of it:
../cpp/ask_user_gate.cpp, compiled, for machines where the interpreter start
(~70 ms against the binary's ~7 ms, before every shell call) is worth removing.
A rule changed here has to reach it too, and ../cpp/parity_check.py is the gate
that says so -- it runs SELF_TEST_CASES below plus a corpus aimed at the seams
through both halves and diffs verdict, exit code and rendered text. Note what
that implies for the shape of a change: text and offsets are part of the
contract, not decoration.

Standalone use (same verdict, exit 1 when denied):
  python hooks/ask_user_gate.py --check "git add -A && git commit -m x"
  python hooks/ask_user_gate.py --check "Get-Item a; Get-Item b" --shell powershell
  python hooks/ask_user_gate.py --check-file cmd.txt   # multi-line commands
  python hooks/ask_user_gate.py --self-test
"""

import argparse
import json
import os
import re
import shlex
import string
import sys
import tempfile
from dataclasses import dataclass

SELF = os.path.abspath(__file__)
HERE = os.path.dirname(SELF)

# The scripts the refusals below send the caller to. They ship in ../bin, so a
# machine that has this gate always has them too; see tool_path().
SHIPPED_TOOLS = ("replace_in_file.py", "try_patch.py")

# What a `tools/<name>.py` in a checkout must contain to be recognised as a
# stand-in for one of those rather than an unrelated script of the same name.
# The stand-ins name the plugin in their docstring and say there that this
# string is read from here -- reword one of them and refusals silently go back
# to naming the long path.
TOOL_MARKER = "ask-user-gate"

# The shell's working directory, from the hook payload, once main() has read
# it. None in standalone use, where the process's own directory is the same
# thing. See caller_dir().
_CALLER_CWD: "str | None" = None

# The marker that turns the gate off for one command, matched case-insensitively.
# ASCII-insensitively: str.lower() folds U+212A KELVIN SIGN to "k", so
# `allowasKuser` disarmed the gate here and not in the C++ port. The marker
# is an ASCII token and the escape hatch should be spellable exactly one way.
MARKER = "allowaskuser"
ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

# Hard limit of the command analyser inside Claude Code (MAX_COMMAND_LENGTH,
# measured -- see CLAUDE.md). Above it the parse aborts, nothing can be scanned,
# and the decision is forced down to "ask the human" whatever the allow-rules say.
MAX_COMMAND_LENGTH = 10000

BASH_TOOLS = {"Bash"}
POWERSHELL_TOOLS = {"PowerShell"}
# Monitor takes a `command` too, and runs it in the same local shell (its own
# stop message calls the task `local_bash`), so the analyser judges it exactly
# like a Bash call -- but under a tool name of its own, which is how the shape
# this gate exists to stop walked straight past it into a dialog nobody was
# watching (2026-08-18, a subagent polling `while true; do ... sleep 10; done`
# for a background run to end). Allow-rules are written `Bash(...)` and do not
# carry over to it either, so a Monitor command is MORE likely to ask, not less.
MONITOR_TOOLS = {"Monitor"}

# A redirection target that is not absolute and not a device. On Windows the
# analyser refuses a chained command whose write target is relative: under Git
# Bash the final working directory of a chain cannot be determined statically,
# so it cannot check the target for a Cygwin-emulated symlink and drops to
# "ask the human". Only used to explain the chain finding better.
#
# re.ASCII on this pattern and the three below, because `\s`, `\d` and `\b` are
# Unicode-aware by default on a str and a SHELL is not. Without it `cd\xa0webgame
# && ls` matched CD_COMMAND -- but a non-breaking space is not a word separator
# to bash, so that text is one word `cd\xa0webgame` and no directory change at
# all, and the same held for `sleep ٣٠` with Arabic-Indic digits. The
# port's matchers are ASCII (see cpp/ask_user_gate.cpp); this is the half that
# was wrong, so this is the half that moved.
RELATIVE_REDIRECT = re.compile(r">>?\s*(?!/|[A-Za-z]:[\\/]|&)([^\s|;&<>()]+)",
                               re.ASCII)

# A command word that moves the working directory, at the start of the command
# or right after a separator. This is what makes a chain unanalysable: the
# directory the rest of the chain runs in stops being a static fact, so the
# analyser cannot resolve any relative path in it and asks a human instead.
#
# A chain WITHOUT one of these is currently let through, on the user's reading
# (2026-08-15) that the dialogs they saw were all cd-compounds rather than
# chains as such. That is a relaxation of the CLAUDE.md rule as ENFORCEMENT
# only -- the rule itself stands, since one command per call also stays
# retryable and lets the harness attribute a failure. If the prompts come back
# for a plain `a; b`, widen this to any separator again: the change is the
# `cwd_changing` guard at the end of scan_shell_syntax, nothing else.
CD_COMMAND = re.compile(
    r"(?:^|[;&|(){}\n])\s*(?:cd|pushd|popd|chdir|[Ss]et-[Ll]ocation|sl)(?:\s|$)",
    re.ASCII)

# Waiting BY THE CLOCK -- `sleep` as a command word with a duration after it,
# and its PowerShell twin. Refused for two independent reasons, either of which
# is enough: the allow-rules match neither `sleep` nor the `while` loop it
# usually hides in, so the call parks the session on a dialog; and the wait is
# unnecessary, because the tools have three ways to wait that do not burn
# wall-clock (SLEEP_FIX names them).
#
# The loop keywords are anchors on purpose. The shape actually observed was
# `i=0; while [ $i -lt 55 ]; do sleep 30; i=$((i+1)); done`, where the sleep
# follows `do` rather than a separator -- and that loop is also how a foreground
# `sleep`, which the Bash tool blocks on its own, gets smuggled past the block.
# Requiring a DURATION after the word is what keeps `ls sleep.txt` and
# `npm run x -- --sleep 5` out of the net. In PowerShell `sleep` is an alias of
# Start-Sleep, so the first pattern is applied to both shells.
SLEEP_COMMAND = re.compile(
    r"(?:^|[;&|(){}\n]|\b(?:do|then|else)\s)\s*sleep\b\s*[\d$]", re.ASCII)
START_SLEEP_COMMAND = re.compile(
    r"(?:^|[;&|(){}\n]|\b(?:do|then|else)\s)\s*start-sleep\b",
    re.IGNORECASE | re.ASCII)

# Printed in place of the dialog, so the agent picks a replacement instead of
# asking what to do. Ordered by how often each one is the right answer.
SLEEP_FIX = (
    "wait with a tool, not with the clock. Three replacements, in order:\n"
    "       (1) run the long command in the FOREGROUND with the tool's "
    "`timeout` parameter (up to 600000 ms) -- the call blocks for you, so "
    "there is nothing left to sleep for;\n"
    "       (2) wait for a CONDITION with the Monitor tool (it is deferred: "
    "run ToolSearch(\"select:Monitor\") first, then give it an until-loop);\n"
    "       (3) start the work with the tool's `run_in_background` parameter "
    "and let the completion notification reach you.\n"
    "       A SUBAGENT is not woken by its own background job -- only the main "
    "loop is -- so for a subagent (3) means no wakeup at all: run it in the "
    "foreground instead. A wait longer than the 10-minute foreground cap is "
    "the orchestrator's work, not the subagent's: report and hand it back.\n"
    "       If you truly must idle, one long call costs one prompt and a loop "
    "of short ones costs a prompt per iteration.")

# The same wait, written inside a Monitor command. Monitor documents a poll
# loop as its shape for "one event per OCCURRENCE, indefinitely" -- but this
# refusal is about the other case, waiting for a condition that happens ONCE,
# which its own description sends elsewhere too. Worth stating separately
# because the generic advice above ends in "use Monitor", and that is exactly
# what the author of this command did.
MONITOR_SLEEP_FIX = (
    "a one-shot wait (`break` when the thing is done) is not what Monitor is "
    "for -- its own description sends that case to a foreground call or to "
    "Bash `run_in_background`.\n"
    "       (1) run the work itself in the FOREGROUND with the tool's "
    "`timeout` parameter (up to 600000 ms) instead of starting it detached and "
    "watching for its death;\n"
    "       (2) if you are a SUBAGENT, this is the only option: background "
    "events do not re-invoke you, so a Monitor armed by a subagent notifies "
    "nobody and only burns its timeout. A wait longer than the foreground cap "
    "is the orchestrator's work -- report and hand it back.\n"
    "       Note also that allow-rules are written `Bash(...)` and do not "
    "cover Monitor, so its command asks the human even when the same text "
    "would have been allowed as a Bash call.")


def is_windows() -> bool:
    return os.name == "nt"


def caller_dir() -> str:
    """The directory a relative path in a refusal will be resolved against.

    The hook payload carries the shell's working directory, and that -- not the
    project root -- is what the agent's next command will start from. They are
    usually the same; when they are not (this repository declares working
    directories outside the project), answering the question about the wrong
    one is how a refusal ends up naming a path the reader cannot open.
    """
    return _CALLER_CWD or os.getcwd()


def is_our_copy(path: str) -> bool:
    """Is that OUR script, or something else that happens to share its name?

    `tools/try_patch.py` is not a reserved name, and a checkout that has one of
    its own would be sent to it by an existence check alone -- with a confident
    sentence describing flags it does not have, which is worse than the long
    path it replaced. The stand-ins in this repository name the plugin in their
    docstring, so a cheap read settles it; anything else falls back to the
    shipped copy, which costs a longer path and nothing else.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return TOOL_MARKER in handle.read(4096)
    except OSError:
        return False


def shipped_path(name: str) -> str:
    """Where one of SHIPPED_TOOLS sits on THIS machine, absolute.

    The layout `hooks/<this file>` beside `bin/<tool>` is what makes the pair
    one shipping unit, so the `os.pardir` hop is a fact about the plugin rather
    than about any checkout.

    One spelling on purpose. The check with teeth is check_paths() asking the
    FILESYSTEM whether the answer names a real file -- measured: breaking the
    `os.pardir` hop fails the self-test 64/66 -- and its comparisons then pin
    which BRANCH tool_path() took, not how the path is spelled.
    """
    return os.path.normpath(os.path.join(HERE, os.pardir, "bin", name))


def tool_path(name: str, base: "str | None" = None) -> str:
    """How to spell one of SHIPPED_TOOLS so that THIS machine can run it.

    A refusal that names a path which does not exist here is worse than no
    advice at all: the reader spends the search before concluding the same
    thing. There are two spellings and the right one depends on where the
    command was issued.

    A checkout that keeps its own stand-ins under `tools/` (this gate grew up
    in one, and its CLAUDE.md, its docs and its source comments all say
    `tools/try_patch.py`) gets the short, already-familiar path -- but only
    when that path resolves from where the command will actually run, because
    that is the spelling being handed out. Anywhere else -- a work checkout
    that never heard of this convention -- gets the absolute path to the copy
    that shipped with the plugin, which is the only one certain to be there.
    """
    if base is None:
        base = caller_dir()
    if is_our_copy(os.path.join(base, "tools", name)):
        return f"tools/{name}"
    return shipped_path(name)


@dataclass(frozen=True)
class Finding:
    """One reason to refuse, plus what to do instead."""
    reason: str
    fix: str
    # Chain findings survive only when the command also moves the working
    # directory; everything else is refused on its own.
    chain: bool = False


def scan_shell_syntax(command: str, shell: str, windows: bool,
                      sleep_fix: str = SLEEP_FIX) -> list[Finding]:
    """Walk the command outside quotes, reporting separators and heredocs.

    Quote tracking is what makes this usable: `python -c 'a; b'` keeps its
    semicolon inside a string and is not a chain, while `a; b` is. The escape
    character differs per shell -- backslash in sh, backtick in PowerShell,
    where a trailing backslash in a Windows path would otherwise eat the
    closing quote and desynchronise the whole scan.
    """
    escape = "\\" if shell == "bash" else "`"
    findings: list[Finding] = []
    seen: set[str] = set()

    def report(token: str, index: int, reason: str, fix: str,
               chain: bool = False) -> None:
        if token in seen:  # one line per kind of violation, not per occurrence
            return
        seen.add(token)
        findings.append(Finding(f"{reason} (at offset {index})", fix, chain))

    chain_fix = ("this chain also moves the working directory, so where the "
                 "rest of it runs stops being a static fact and no relative "
                 "path in it can be resolved. Name the directory with the "
                 "tool's own flag (--cwd / --prefix / -C) or an absolute path "
                 "instead of `cd X && ...`, and keep one command per call.")
    if windows and RELATIVE_REDIRECT.search(command):
        chain_fix += (" On Windows this one is unappealable: the write target "
                      "is relative, and under Git Bash the working directory a "
                      "chain ends in cannot be determined statically, so the "
                      "analyser cannot check that target and refuses to "
                      "delegate the decision at all.")
    heredoc_fix = (f"write files with the Write tool, edit them with Edit, and "
                   f"do scripted replacements with "
                   f"{tool_path('replace_in_file.py')}. If the content must be "
                   f"produced by a program, let the script write the file and "
                   f"keep the command a call to that script.")

    brace_fix = ("keep the payload out of the command line: write it to a file "
                 "with the Write tool and let the command be a call to that "
                 "file, or quote the whole argument so the braces sit inside a "
                 "string.")

    i, n = 0, len(command)
    quote = None
    braces = 0  # depth of `{ ... }` groups seen OUTSIDE quotes
    plain: list[str] = []  # the command with quoted runs blanked out
    while i < n:
        c = command[i]
        if quote is not None:
            if c == escape and quote == '"' and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        # From here on we are at top level, so `plain` gets this character --
        # a quoted run collapses to one space, which keeps words apart without
        # letting a string body look like a command word.
        plain.append(" " if c in "'\"" else c)
        if c == escape and i + 1 < n:
            i += 2  # escaped char, including a line continuation
            continue
        if c in "'\"":
            if braces > 0:
                # The analyser reads a quote inside unquoted braces as
                # expansion obfuscation and asks a human. `awk '{print "x"}'`
                # is fine (the braces are inside the quotes, not the other way
                # round); a heredoc body or a bare ${VAR:-"x"} is not.
                report("{", i, "quote character inside unquoted `{ ... }` -- "
                               "read as expansion obfuscation", brace_fix)
            quote = c
            i += 1
            continue
        if c == "{":
            braces += 1
            i += 1
            continue
        if c == "}":
            braces = max(0, braces - 1)
            i += 1
            continue
        if c == "#" and (i == 0 or command[i - 1] in " \t\n"):
            nl = command.find("\n", i)
            if nl < 0:
                break
            i = nl  # the newline itself is still a top-level separator
            continue

        two = command[i:i + 2]
        if two in ("&&", "||"):
            report(two, i, f"shell chain operator `{two}` in a chain that "
                           f"changes directory", chain_fix, chain=True)
            i += 2
            continue
        if two == "<<":
            if command[i:i + 3] == "<<<":
                i += 3  # here-string: one line, not a file body
                continue
            report("<<", i, "heredoc (`<<`) -- a file written by the command body",
                   heredoc_fix)
            i += 2
            continue
        if two in ("@'", '@"') and shell == "powershell":
            report("@'", i, "PowerShell here-string -- a file written by the "
                            "command body", heredoc_fix)
            i += 2
            continue
        if c == ";":
            report(";", i, "command separator `;` in a chain that changes "
                           "directory", chain_fix, chain=True)
            i += 1
            continue
        if c == "\n":
            report("\n", i, "newline outside quotes -- a second command on its "
                            "own line, in a chain that changes directory",
                   chain_fix, chain=True)
            i += 1
            continue
        if c == "&":
            # `2>&1`, `&>file`, `|&` are redirections, not backgrounding.
            prev = command[i - 1] if i else ""
            if prev not in "><|" and command[i + 1:i + 2] != ">":
                report("&", i, "backgrounding `&`",
                       "use the tool's run_in_background parameter instead of "
                       "detaching with `&`.")
            i += 1
            continue
        i += 1

    plain_command = "".join(plain)
    # Offsets are deliberately absent here: `plain` drops quoted runs rather
    # than blanking them, so its indices are not the command's.
    if SLEEP_COMMAND.search(plain_command) or START_SLEEP_COMMAND.search(
            plain_command):
        findings.append(Finding("waiting by the clock (`sleep`)", sleep_fix))
    if not CD_COMMAND.search(plain_command):
        # A chain of allow-listed commands appears to be matched fine; it is
        # the directory change that defeats the analyser. See CD_COMMAND.
        findings = [finding for finding in findings if not finding.chain]
    return findings


def scan_sed_in_place(command: str) -> list[Finding]:
    """`sed -i` -- banned by CLAUDE.md and not on any allow-list here."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if token != "sed" and not token.endswith("/sed"):
            continue
        for arg in tokens[index + 1:]:
            if arg in ("|", ";", "&&", "||"):
                break
            if arg == "--in-place" or arg.startswith("--in-place="):
                pass
            elif not (len(arg) > 1 and arg[0] == "-" and arg[1] != "-"
                      and "i" in arg.split(".")[0].lstrip("-")):
                continue
            return [Finding(
                "`sed -i`",
                f"use {tool_path('replace_in_file.py')} (it refuses to write "
                f"when the match count is not the one you named, so a wrong "
                f"pattern cannot pass as a successful edit), or "
                f"{tool_path('try_patch.py')} when the edit is temporary and "
                f"must be undone after a test run.")]
    return []


def scan(command: str, shell: str, windows: "bool | None" = None,
         tool: str = "Bash") -> list[Finding]:
    """All reasons this command would stop for a human, or an empty list."""
    if MARKER in command.translate(ASCII_LOWER):
        return []
    if windows is None:
        windows = is_windows()
    sleep_fix = MONITOR_SLEEP_FIX if tool in MONITOR_TOOLS else SLEEP_FIX
    findings: list[Finding] = []
    if len(command) > MAX_COMMAND_LENGTH:
        findings.append(Finding(
            f"the command is {len(command)} characters, over the analyser's "
            f"{MAX_COMMAND_LENGTH}-character limit",
            "above that limit the command cannot be parsed at all and the "
            "decision is forced to 'ask the human' whatever the allow-rules "
            "say. Move the payload into a file written with the Write tool and "
            "keep the command a short call to it."))
    findings.extend(scan_shell_syntax(command, shell, windows, sleep_fix))
    if shell == "bash":
        findings.extend(scan_sed_in_place(command))
    return findings


def render(findings: list[Finding]) -> str:
    """The refusal the agent reads, in place of the dialog the user would."""
    # Resolved, not hardcoded: the same script is a plugin on one machine and a
    # loose hook on another, so a literal path lies on half of them -- and the
    # bare basename, while greppable, does not tell a reader who has never seen
    # this plugin where the thing refusing their command lives.
    lines = [f"Blocked by {SELF}: this command would stop "
             f"the session on a permission prompt, so it was not run.", ""]
    said: set[str] = set()  # several findings usually share one remedy
    for finding in findings:
        lines.append(f"  - {finding.reason}")
        if finding.fix not in said:
            said.add(finding.fix)
            lines.append(f"    -> {finding.fix}")
    lines.append("")
    lines.append("If you want this exact command anyway and accept that the "
                 "user will be asked, add the marker `allowAskUser` to it "
                 "(e.g. append ` # allowAskUser`) and it will be passed "
                 "through unchanged.")
    return "\n".join(lines)


def shell_for_tool(tool_name: str) -> "str | None":
    if tool_name in BASH_TOOLS or tool_name in MONITOR_TOOLS:
        return "bash"
    if tool_name in POWERSHELL_TOOLS:
        return "powershell"
    return None


SELF_TEST_CASES = [
    # (command, shell, should_be_denied[, tool])
    ("git status", "bash", False),
    ("npm run build --prefix webgame", "bash", False),
    ("git log --oneline -20 | head -5", "bash", False),
    ("python -c 'print(1); print(2)'", "bash", False),
    ("grep -rn 'a && b' src", "bash", False),
    ("node -e \"console.log(1)\" 2>&1", "bash", False),
    # A plain chain is let through; a chain that moves the cwd is not.
    ("git add -A && git commit -m x", "bash", False),
    ("ls; pwd", "bash", False),
    ("rg --version; rg --files | head -2", "bash", False),
    ("cd webgame && npx vitest run", "bash", True),
    ("cd webgame; npx vitest run", "bash", True),
    ("pushd webgame && npm run build", "bash", True),
    ("echo 'cd webgame && x'", "bash", False),
    ("git log --format=%cd; git status", "bash", False),
    ("Set-Location webgame; npx vitest run", "powershell", True),
    ("npm run dev &", "bash", True),
    ("cat > f.txt <<'EOF'\nbody\nEOF", "bash", True),
    ("ls a <<< 'x'", "bash", False),
    ("sed -i 's/a/b/' file.ts", "bash", True),
    ("sed -i.bak 's/a/b/' file.ts", "bash", True),
    ("sed -n '1,5p' file.ts", "bash", False),
    ("echo 'sed -i is banned'", "bash", False),
    ("x" * (MAX_COMMAND_LENGTH + 1), "bash", True),
    # Waiting by the clock, including the loop that smuggles a foreground
    # `sleep` past the Bash tool's own block on it.
    ("sleep 30", "bash", True),
    ("i=0; while [ $i -lt 55 ]; do sleep 30; i=$((i+1)); done", "bash", True),
    ("while true; do sleep $DELAY; done", "bash", True),
    ("npm run dev --prefix webgame; sleep 2", "bash", True),
    ("Start-Sleep -Seconds 30", "powershell", True),
    ("sleep 5", "powershell", True),
    # ... and the words that only look like it.
    ("ls sleep.txt", "bash", False),
    ("npm run probe --prefix webgame -- --sleep 5", "bash", False),
    ("grep -n 'sleep 30' scripts/probe.sh", "bash", False),
    ("python -c \"import time; time.sleep(1)\"", "bash", False),
    ("awk '{print \"x\"}' file", "bash", False),
    ("find . -name '*.tmp' -exec rm {} +", "bash", False),
    ("echo ${VAR:-\"x\"}", "bash", True),
    ("git add -A && git commit -m x  # allowAskUser", "bash", False),
    ("Get-ChildItem C:\\game\\1339", "powershell", False),
    ("Get-Item a; Get-Item b", "powershell", False),
    ("git commit -F @'\nmsg\n'@", "powershell", True),
    # Monitor carries a shell command under a tool name of its own. The first
    # case is the one that walked past this gate into a dialog (2026-08-18).
    ("while true; do if ! tasklist //FI \"IMAGENAME eq node.exe\" | grep -qi "
     "node.exe; then echo done; break; fi; sleep 10; done", "bash", True,
     "Monitor"),
    ("tail -f webgame/.devlogs/latest.log | grep --line-buffered ERROR",
     "bash", False, "Monitor"),
]


def check_wiring() -> "tuple[list[str], int]":
    """Every tool this script handles must also be in the hook's matcher.

    The failure this pins is the one that happened: the scanner refused the
    command in every standalone check, and the session still stopped on a
    dialog, because the tool that carried it (Monitor) was not in the matcher
    of the PreToolUse entry -- so the hook never ran. A gate that guards its own
    logic but not its wiring is a gate with a hole exactly this shape.

    EVERY file that could be running it is checked, not the first one found,
    and that is the whole point. There are two ways to be wired -- the
    `hooks.json` shipped next to this script (a plugin), and a `settings.json`
    naming it by path (a loose hook, which is how a checkout runs it before the
    plugin is installed anywhere) -- and checking only one of them recreates
    the hole: on a machine where the LIVE wiring is the settings.json and the
    inert one is the plugin's own hooks.json, a matcher edit to the file that
    actually runs would pass, silently, which is this repository's
    verification-that-cannot-fail failure exactly. Measured, not reasoned: with
    only the next-to-me candidate, dropping Monitor from `.claude/settings.json`
    still printed a clean self-test.

    An entry counts as ours by the STEM, not the file name, because there are
    two implementations: a settings.json may name `ask_user_gate.exe` (the C++
    port) while this script is the one running the check. Matching the full name
    would silently stop looking at the live wiring the moment a checkout switched
    to the binary -- reopening the hole above from the other side.
    """
    problems: list[str] = []
    handled = sorted(BASH_TOOLS | POWERSHELL_TOOLS | MONITOR_TOOLS)
    checks = len(handled) + 1
    for tool in handled:
        if shell_for_tool(tool) is None:
            problems.append(f"shell_for_tool({tool!r}) is None")
    if shell_for_tool("Read") is not None:
        problems.append("shell_for_tool routes a tool that carries no command")

    stem = os.path.splitext(os.path.basename(SELF))[0]
    candidates = list(dict.fromkeys([
        os.path.join(HERE, "hooks.json"),
        os.path.join(HERE, "settings.json"),
        os.path.join(caller_dir(), ".claude", "settings.json"),
        os.path.join(os.path.expanduser("~"), ".claude", "settings.json"),
    ]))
    wired = 0
    for config in candidates:
        if not os.path.isfile(config):
            continue
        checks += 1
        try:
            with open(config, encoding="utf-8") as handle:
                entries = json.load(handle).get("hooks", {}).get("PreToolUse",
                                                                 [])
        except (OSError, ValueError) as error:
            problems.append(f"cannot read {config}: {error}")
            continue
        ours = [entry for entry in entries
                if any(stem in hook.get("command", "")
                       for hook in entry.get("hooks", []))]
        wired += len(ours)
        for entry in ours:
            matcher = entry.get("matcher", "")
            checks += len(handled)
            for tool in handled:
                if not re.search(rf"(?:^|\|){re.escape(tool)}(?:\||$)",
                                 matcher):
                    problems.append(f"{tool} is handled here but missing from "
                                    f"the matcher {matcher!r} in {config}")
    if not wired:
        problems.append(f"nothing runs this gate: no PreToolUse entry names "
                        f"{stem} in any of {', '.join(candidates)}")
    return problems, checks


def check_paths() -> "tuple[list[str], int]":
    """Both spellings tool_path() can produce must land on a real file.

    Without this the resolver is untestable in the only way that matters: its
    output is prose inside a refusal, which nothing compiles and nobody diffs,
    so a rename in ../bin or a wrong `os.pardir` would surface as an agent
    searching for a file that is not there -- exactly the cost the resolver
    exists to remove. Here it surfaces as a failing self-test instead.

    The away-from-a-repo branch is checked against the filesystem (the shipped
    copy must exist); the in-a-repo branch is checked against a directory built
    for the purpose, because whether the machine running the test happens to
    sit in such a repository is not something a test may depend on.

    The last case is the one that earns the rest: `base=None`, the only spelling
    the hook ever uses. Passing `base` explicitly everywhere leaves the default
    -- where the directory actually comes from -- untested, and a typo there
    degrades every refusal to the long path with nothing to say so.
    """
    problems: list[str] = []
    checks = 0
    try:
        project = tempfile.TemporaryDirectory()
    except OSError as error:  # a locked-down machine is where this ships
        return [f"cannot create a temporary directory to check the paths in: "
                f"{error}"], 1
    with project as root:
        os.mkdir(os.path.join(root, "tools"))

        def plant(name: str, ours: bool) -> str:
            """Give the fake checkout a `tools/<name>`, ours or a stranger's.

            The two bodies differ only by the marker, which is the whole
            question is_our_copy() answers -- writing them apart is how a test
            ends up planting a marked file and asserting the unmarked answer.
            """
            path = os.path.join(root, "tools", name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(f"# a stand-in, marked {TOOL_MARKER}\n" if ours
                             else "# someone else's script of the same name\n")
            return path

        for name in SHIPPED_TOOLS:
            checks += 4
            shipped = shipped_path(name)
            if not os.path.isfile(shipped):
                problems.append(f"{name} is named in a refusal but is not "
                                f"shipped at {shipped}")
            away = tool_path(name, base=root)
            if away != shipped:
                problems.append(f"{name}: outside a repository the refusal "
                                f"names {away!r}, not the shipped copy")
            local = plant(name, ours=True)
            near = tool_path(name, base=root)
            if near != f"tools/{name}":
                problems.append(f"{name}: a checkout with its own stand-in is "
                                f"told {near!r}, not the short path")
            plant(name, ours=False)
            stranger = tool_path(name, base=root)
            if stranger != shipped:
                problems.append(f"{name}: an unrelated tools/{name} is handed "
                                f"out as {stranger!r} instead of being ignored")
            os.remove(local)

        # The default base, which is the only one the hook uses.
        checks += 2
        global _CALLER_CWD
        restore, _CALLER_CWD = _CALLER_CWD, root
        try:
            name = SHIPPED_TOOLS[0]  # the branch is the same for either name
            if tool_path(name) != shipped_path(name):
                problems.append(f"{name}: with no base given the resolver does "
                                f"not fall back to the shipped copy")
            local = plant(name, ours=True)
            if tool_path(name) != f"tools/{name}":
                problems.append(f"{name}: with no base given the resolver does "
                                f"not read the caller's directory")
            # ... and standalone, where no payload said where the caller is.
            checks += 1
            here = os.getcwd()
            _CALLER_CWD = None
            try:
                os.chdir(root)
                if tool_path(name) != f"tools/{name}":
                    problems.append(f"{name}: with no payload the resolver "
                                    f"does not fall back to the process's own "
                                    f"directory")
            finally:
                os.chdir(here)
            os.remove(local)
        finally:
            _CALLER_CWD = restore
    return problems, checks


def self_test() -> int:
    failures = 0
    checks = len(SELF_TEST_CASES)
    for label, (problems, count) in (("wiring", check_wiring()),
                                     ("paths", check_paths())):
        checks += count
        for problem in problems:
            failures += 1
            print(f"FAIL [{label}] {problem}", file=sys.stderr)
    for case in SELF_TEST_CASES:
        command, shell, expected = case[0], case[1], case[2]
        tool = case[3] if len(case) > 3 else "Bash"
        denied = bool(scan(command, shell, windows=True, tool=tool))
        if denied != expected:
            failures += 1
            label = "denied" if denied else "allowed"
            shown = command if len(command) < 60 else command[:57] + "..."
            print(f"FAIL [{shell}/{tool}] {shown!r}: {label}, expected "
                  f"{'denied' if expected else 'allowed'}", file=sys.stderr)
    print(f"{checks - failures}/{checks} checks pass")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse shell commands that would stop for a human "
                    "permission prompt. Reads a PreToolUse hook payload on "
                    "stdin unless --check/--self-test is given.")
    parser.add_argument("--check", metavar="COMMAND",
                        help="scan one command and print the verdict")
    parser.add_argument("--check-file", metavar="PATH",
                        help="scan the command stored in a file (for the "
                             "multi-line ones an argument cannot carry)")
    parser.add_argument("--shell", choices=("bash", "powershell"),
                        default="bash", help="shell to assume (default: bash)")
    parser.add_argument("--tool", default="Bash",
                        help="tool the command came from; only Monitor differs "
                             "(its remedy is not the same one) (default: Bash)")
    parser.add_argument("--platform", choices=("auto", "windows", "posix"),
                        default="auto",
                        help="host the command would run on; only the Git Bash "
                             "note depends on it (default: auto)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in scanner cases")
    options = parser.parse_args()

    if options.self_test:
        return self_test()

    windows = (is_windows() if options.platform == "auto"
               else options.platform == "windows")

    command = options.check
    if options.check_file is not None:
        with open(options.check_file, encoding="utf-8") as handle:
            command = handle.read()
    if command is not None:
        findings = scan(command, options.shell, windows, options.tool)
        if not findings:
            print("allowed")
            return 0
        print(render(findings))
        return 1

    # Hook mode. Anything unexpected here must fail OPEN: a broken gate that
    # blocked every command would be worse than the prompts it prevents.
    global _CALLER_CWD
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get("tool_name", "")
        shell = shell_for_tool(tool_name)
        if shell is None:
            return 0
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd:
            _CALLER_CWD = cwd
        # Monitor's other form is `ws` (a socket, no shell); missing `command`
        # then means there is nothing to scan, not that something went wrong.
        command = payload.get("tool_input", {}).get("command", "")
        if not isinstance(command, str):
            return 0
        findings = scan(command, shell, tool=tool_name)
    except Exception:
        return 0

    if not findings:
        return 0
    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": render(findings),
    }}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
