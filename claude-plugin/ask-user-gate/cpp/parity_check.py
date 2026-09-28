#!/usr/bin/env python3
"""Gate: the C++ port and ask_user_gate.py must answer identically.

Two implementations of one rule set drift silently, and this one is
security-adjacent: the half that gets it wrong is the half that stops refusing.
So the port is not pinned by its own copy of the cases -- that pins nothing, the
two lists being equal by construction -- but by running the REFERENCE and the
BINARY over one corpus and diffing verdict, exit code and refusal text.

The corpus is ask_user_gate.SELF_TEST_CASES (so a case added there reaches the
port for free) plus EXTRA_CASES below, which exist because the shared list only
pins the yes/no. Text, offsets, quoting and the shlex fallback are where a
hand-written matcher diverges from a regex first, and none of them changes a
verdict until it is far too late.

  python cpp/parity_check.py                 # both halves, whole corpus
  python cpp/parity_check.py --exe PATH      # a binary built somewhere else
  python cpp/parity_check.py --verbose       # print every case
  python cpp/parity_check.py --jobs 1        # one launch at a time
  PARITY_JOBS=2 python cpp/parity_check.py   # the same, for a caller (pytest)
                                             # that cannot pass --jobs

Exit 1 on any difference. Requires the binary, built with cpp/build.py since
the source last changed (exit 2 otherwise, see newer_sources).
"""

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.normpath(os.path.join(HERE, os.pardir))
HOOKS = os.path.join(PLUGIN, "hooks")
sys.path.insert(0, HOOKS)

import ask_user_gate as reference  # noqa: E402  (needs the path above)

# Cases the shared list does not cover, chosen for the seams between a regex and
# a hand-written matcher rather than for coverage of the rules themselves.
EXTRA_CASES = [
    # Offsets, and the "one line per kind" dedupe that decides which offset wins.
    ("a; b; c", "bash", "Bash"),
    ("cd x; a; b && c", "bash", "Bash"),
    ("cd x && a && b", "bash", "Bash"),
    # The Windows note on the chain fix -- a relative redirect target vs the
    # spellings the lookahead excludes.
    ("cd x && echo hi > out.txt", "bash", "Bash"),
    ("cd x && echo hi > /tmp/out.txt", "bash", "Bash"),
    ("cd x && echo hi > C:/tmp/out.txt", "bash", "Bash"),
    ("cd x && echo hi >> out.txt", "bash", "Bash"),
    ("cd x && node app.js 2>&1", "bash", "Bash"),
    ("cd x && node app.js &> out", "bash", "Bash"),
    # The cd word list: prefixes, near-misses and the one that also spells sleep.
    ("cdx foo; bar", "bash", "Bash"),
    ("chdir x; bar", "bash", "Bash"),
    ("sl x; bar", "powershell", "PowerShell"),
    ("sleep_ok x; bar", "bash", "Bash"),
    ("set-location x; bar", "powershell", "PowerShell"),
    ("Set-location x; bar", "powershell", "PowerShell"),
    ("popd; ls", "bash", "Bash"),
    ("echo %cd%; ls", "bash", "Bash"),
    # Sleep anchors: the keyword alternative, its word boundary, the argument.
    ("then sleep 5", "bash", "Bash"),
    ("else sleep 5", "bash", "Bash"),
    ("undo sleep 5", "bash", "Bash"),
    ("if x; then sleep 1; fi", "bash", "Bash"),
    ("sleep", "bash", "Bash"),
    ("sleep abc", "bash", "Bash"),
    ("sleep  7", "bash", "Bash"),
    ("sleeper 5", "bash", "Bash"),
    ("do sleepy 5", "bash", "Bash"),
    ("START-SLEEP -Seconds 1", "powershell", "PowerShell"),
    ("dostart-sleep 1", "powershell", "PowerShell"),
    ("(sleep 5)", "bash", "Bash"),
    # Monitor gets a different remedy for the same finding.
    ("while true; do sleep 5; done", "bash", "Monitor"),
    ("sleep 5", "bash", "Monitor"),
    # Quote tracking, escapes and the shell-dependent escape character.
    ("echo \"a; b\"", "bash", "Bash"),
    ("echo \\\"a; b", "bash", "Bash"),
    ("echo 'unterminated; b", "bash", "Bash"),
    ("Get-Item 'C:\\game\\'; Get-Item b", "powershell", "PowerShell"),
    ("Get-Item \"C:\\game\\\"; Get-Item b", "powershell", "PowerShell"),
    ("echo \"C:\\game\\\"; cd x", "bash", "Bash"),
    ("echo hi # cd x && ls", "bash", "Bash"),
    ("echo hi # cd x\ncd y && ls", "bash", "Bash"),
    ("cd x&&ls", "bash", "Bash"),
    # Braces: the finding, and the two shapes that must NOT trip it.
    ("awk '{print $1}' f", "bash", "Bash"),
    ("echo ${VAR:-'x'}", "bash", "Bash"),
    ("echo }{ \"x\"", "bash", "Bash"),
    ("find . -exec rm {} \\;", "bash", "Bash"),
    # Heredocs and here-strings on both shells.
    ("python <<EOF\nprint(1)\nEOF", "bash", "Bash"),
    ("cat <<<'x'", "bash", "Bash"),
    ("cat <<-EOF\nx\nEOF", "bash", "Bash"),
    ("$x = @\"\nline\n\"@", "powershell", "PowerShell"),
    ("$x = @'\nline\n'@", "powershell", "PowerShell"),
    # sed: the flag forms, the ones that only look like them, and the shlex
    # fallback an unbalanced quote forces.
    ("sed --in-place 's/a/b/' f", "bash", "Bash"),
    ("sed --in-place=.bak 's/a/b/' f", "bash", "Bash"),
    ("sed -n -i 's/a/b/' f", "bash", "Bash"),
    ("sed -ni 's/a/b/' f", "bash", "Bash"),
    ("/usr/bin/sed -i s/a/b/ f", "bash", "Bash"),
    ("mysed -i s/a/b/ f", "bash", "Bash"),
    ("sed --posix 's/a/b/' f", "bash", "Bash"),
    ("sed 's/a/b/' f | sed -i 's/c/d/' g", "bash", "Bash"),
    ("sed -i 's/a/b/ f", "bash", "Bash"),
    ("sed -i 's/a/b/' f", "powershell", "PowerShell"),
    # tail -f on a task's output: the word boundary, the flag forms, the path
    # test, the simple-command end, quoting, and a mixed refusal whose header
    # must stay the prompt one.
    ("/usr/bin/tail -f /t/tasks/a.output", "bash", "Bash"),
    ("mytail -f /t/tasks/a.output", "bash", "Bash"),
    ("timeout 400 tail -f /t/tasks/a.output | grep -m1 x", "bash", "Bash"),
    ("tail -qf /t/tasks/a.output", "bash", "Bash"),
    ("tail --follow=name /t/tasks/a.output", "bash", "Bash"),
    ("tail -f -n +1 /t/tasks/a.output", "bash", "Bash"),
    ("tail -f /t/tasks/.output", "bash", "Bash"),
    ("tail -f tasks/a.output", "bash", "Bash"),
    ("tail -f /t/tasks/a.output.bak", "bash", "Bash"),
    ("tail -n 5 /t/tasks/a.output | tail -f", "bash", "Bash"),
    ("tail -f /t/x.log; cat /t/tasks/a.output", "bash", "Bash"),
    ("tail -f '/t/my tasks/tasks/a b.output'", "bash", "Bash"),
    ("tail\t-f /t/tasks/a.output", "bash", "Bash"),
    ("tail -f /t/tasks/a.output", "powershell", "PowerShell"),
    ("cd x && tail -f /t/tasks/a.output", "bash", "Bash"),
    ("echo яя; tail -f /t/tasks/a.output", "bash", "Monitor"),
    # Backgrounding vs the redirections that share the character.
    ("npm run dev & echo started", "bash", "Bash"),
    ("cmd |& tee log", "bash", "Bash"),
    # The escape hatch, in the spellings it will actually be typed in.
    ("cd webgame && npx vitest run # allowAskUser", "bash", "Bash"),
    ("cd webgame && npx vitest run # ALLOWASKUSER", "bash", "Bash"),
    # Length: on the limit, over it, and over it in multi-byte characters, where
    # a byte count and a code-point count disagree.
    ("x" * reference.MAX_COMMAND_LENGTH, "bash", "Bash"),
    ("x" * (reference.MAX_COMMAND_LENGTH + 1), "bash", "Bash"),
    ("\u044f" * (reference.MAX_COMMAND_LENGTH - 1), "bash", "Bash"),
    ("\u044f" * (reference.MAX_COMMAND_LENGTH + 1), "bash", "Bash"),
    ("echo \u044f\u044f\u044f; cd x", "bash", "Bash"),
    # A long whitespace run where each pattern repeats `\s`. The port's regex
    # engine recurses once per character of a greedy repeat, and a blown stack
    # is a dead hook, not a verdict: the redirect one exited 0xC00000FD.
    ("cd x && echo >" + " " * 200000 + "out.txt", "bash", "Bash"),
    (";" + " " * 200000 + "cd x && ls", "bash", "Bash"),
    ("do" + " " * 200000 + "sleep" + " " * 200000 + "5", "bash", "Bash"),
    (";" + " " * 200000 + "Start-Sleep 5", "powershell", "PowerShell"),
    # A LEADING `&` is PowerShell's call operator, not backgrounding, and the
    # reference exempts it through `"" in "><|"` being True in Python.
    ("& \"C:\\Program Files\\App\\app.exe\" arg", "powershell", "PowerShell"),
    ("& echo hi", "bash", "Bash"),
    ("&& echo hi", "bash", "Bash"),
    ("cd x; & app.exe", "powershell", "PowerShell"),
    # Unicode where a shell sees none: NBSP is not a word separator, Arabic-Indic
    # digits are not a duration, and the escape hatch is an ASCII token.
    ("x; cd\u00a0webgame && ls", "bash", "Bash"),
    ("x;\u001ccd webgame && ls", "bash", "Bash"),
    ("sleep \u0663\u0660", "bash", "Bash"),
    ("sleep\u00a05", "bash", "Bash"),
    ("cd x && ls # allowas\u212Auser", "bash", "Bash"),
    # re.IGNORECASE covers a whole pattern, so the loop keyword before
    # Start-Sleep is case-insensitive and the one before `sleep` is not.
    ("Get-Job; THEN Start-Sleep -Seconds 5", "powershell", "PowerShell"),
    ("x; DO start-sleep 5", "powershell", "PowerShell"),
    ("x; DO sleep 5", "bash", "Bash"),
    ("x; do sleep 5", "bash", "Bash"),
    # shlex.whitespace is ' \t\r\n' exactly, and a backslash inside double
    # quotes escapes only `"` and `\`.
    ("sed\v-i 's/a/b/' f", "bash", "Bash"),
    ("sed\f-i 's/a/b/' f", "bash", "Bash"),
    ("sed \"-\\'i\" f", "bash", "Bash"),
    ("sed \"-\\i\" f", "bash", "Bash"),
    # Nothing at all.
    ("", "bash", "Bash"),
    ("   ", "bash", "Bash"),
]

# Commands written to the --check-file with CRLF. The reference opens that file
# in TEXT mode, so every offset after a CRLF differs from a byte-wise read --
# invisible while the corpus was written with newline="" for both halves.
CRLF_CASES = [
    ("cd x\nls\ncd y && ls", "bash", "Bash"),
    ("cat > f.txt <<'EOF'\nbody\nEOF", "bash", "Bash"),
    ("git commit -F @'\nmsg\n'@", "powershell", "PowerShell"),
]

# What argv_verdict makes of the output worth pinning; see ARGV_CASES.
ALLOWED = "allowed\n"
HELP = "<help>"
SELF_TEST = "<self-test>"
# sed -i is refused under bash and not under PowerShell, so this command tells
# which shell an abbreviated --shell actually set.
SED = "sed -i s/a/b/ f"
# Refused on every host, with the Git Bash note on Windows only, so its text
# tells which platform an abbreviated --platform actually set.
CHAIN = "cd x && echo hi > out.txt"


def denied(command: str, shell: str = "bash", tool: str = "Bash",
           windows: bool = True) -> str:
    """The refusal the reference prints for `command` under these options.

    Computed in process, not copied: the refusal's wording is the corpus's to
    pin (above), and what an ARGV_CASES row pins is which shell, tool and
    platform its argv delivered to scan() -- the remedy for Monitor, the Git
    Bash note, which of two repeated --check values won. It also keeps a crash
    out of a row that expects a refusal: a traceback exits 1 as well.
    """
    findings = reference.scan(command, shell, windows, tool)
    if not findings:
        raise ValueError(f"{command!r} is allowed under {shell}/{tool}; "
                         f"an exit-1 row needs a command that is refused")
    return reference.render(findings) + "\n"


# The command line itself: the corpus above goes in through --check-file, so it
# never sees how a VALUE is found. These are the shapes an operator's shell
# delivers -- above all Windows PowerShell 5.1, which drops an empty argument,
# so `--check ""` arrives as a bare --check before the next flag or at the end.
# The contract (ask_user_gate.py, at the --check add_argument): `--check=` is
# the empty command, a bare --check is a usage error (exit 2), and a token
# argparse reads as a flag is never taken as a value.
#
# Rows are (argv, exit code, text), and the table has two readers. main() below
# runs each argv through BOTH halves and compares what argv_verdict returns --
# only the code on a usage error and on the help, which argparse and the port
# word differently. tests/test_ask_user_gate_parity.py holds the REFERENCE alone
# to the code and, where it is not None, to the text: that half needs no
# binary, so it is the one every checkout and every CI interpreter runs. Text is
# None for a usage error
# only (always "<usage error>"), and a verdict always names its text -- the
# pytest reader refuses a row that does not: a 0 because exit 0 with no output
# is also what hook mode returns on the empty stdin argv_verdict gives it, a 1
# because a traceback exits 1 too. A refusal's text is denied(), compared
# through normalise() as the halves are.
ARGV_CASES = [
    (["--check="], 0, ALLOWED),
    (["--check=", "--shell", "powershell"], 0, ALLOWED),
    (["--shell=powershell", "--check="], 0, ALLOWED),
    (["--check", ""], 0, ALLOWED),            # a shell that keeps the token
    (["--check"], 2, None),                   # `--check ""` at the end
    (["--check", "--shell", "powershell"], 2, None),  # ... before a flag
    (["--check", "--platform=windows"], 2, None),     # ... one written with `=`
    (["--check", "--sh", "powershell"], 2, None),     # ... an abbreviation
    (["--check", "-x"], 2, None),             # dash-led, no space: a flag
    # The gate's own negative-number rule (_Parser.NEGATIVE_NUMBER), whatever
    # this interpreter's argparse would say: a negative number is a value ...
    (["--check", "-1"], 0, ALLOWED),
    (["--check", "-.5"], 0, ALLOWED),
    (["--check", "-2.5"], 0, ALLOWED),
    # ... and these are not negative numbers: flags, so usage errors.
    (["--check", "-1e5"], 2, None),
    (["--check", "-1abc"], 2, None),
    (["--check", "-1.2.3"], 2, None),
    (["--check", "-.5x"], 2, None),
    (["--check", "-1_000"], 2, None),
    (["--check", "-5\n"], 2, None),           # the old argparse `$` took it
    (["--check", "-٥"], 2, None),        # a Unicode digit, ARABIC-INDIC 5
    (["--check", "-= x"], 2, None),           # `-` before `=`: every option
    # A `-=` token and an ambiguous prefix are flags, refused on arrival.
    # 3.9-3.11 argparse read `-= x` as a value and refused an ambiguous prefix
    # before running anything, --help included; _Parser._get_option_tuples pins
    # the newer reading, which the port shares. 3.12.14+ argparse gives every
    # one of these natively, so there the rows guard nothing of the override:
    # they bite on 3.9-3.11 only.
    (["--check", "-==x y"], 2, None),
    (["--tool", "-= x", "--check", "ls"], 2, None),   # after any value flag
    # Glued: a `-=` VALUE, the control the other way -- an over-broad refusal
    # would turn it into a usage error.
    (["--check=-= x"], 0, ALLOWED),
    # Refused where parsing REACHES it, so a help before it still prints and
    # one after it never runs.
    (["--help", "-= x"], 0, HELP),
    (["-= x", "--help"], 2, None),
    (["--help", "--c"], 0, HELP),             # ... an ambiguous prefix as well
    # dash-led with a space: a value
    (["--check", "-n 1; cd y && ls"], 1, denied("-n 1; cd y && ls")),
    (["--tool", "--check", "cd x && ls"], 2, None),
    (["--tool", "--check=cd x && ls"], 2, None),  # a flag, space or not
    (["--check", "--sh=a b"], 2, None),       # an abbreviation, space or not
    (["--check", "-h x"], 2, None),           # -h glued to anything is -h
    (["--check", "cd x && ls", "--tool", "-h"], 2, None),
    # Not a usage error in the argparse sense, but exit 2 all the same: exit 1
    # is "denied", and a file that cannot be read was not judged.
    (["--check-file", "no-such-file-7c1f0e.txt"], 2, None),
    # No command at all: hook mode, on the empty stdin argv_verdict gives it.
    (["--tool=Bash"], 0, ""),
    # An unambiguous prefix is its option, with the value space- or =-joined;
    # the verdict row shows WHICH option it set, not just that it parsed.
    (["--sh", "powershell", "--check", SED], 0, ALLOWED),
    (["--sh=powershell", "--check", SED], 0, ALLOWED),
    (["--check", SED], 1, denied(SED)),       # the control: bash refuses it
    # Monitor's remedy for a sleep is not Bash's (see `--tool=--` below).
    (["--t=Monitor", "--check", "sleep 5"], 1, denied("sleep 5", tool="Monitor")),
    (["--pl", "posix", "--check", CHAIN], 1, denied(CHAIN, windows=False)),
    (["--check", CHAIN], 1, denied(CHAIN)),   # the control: the Windows note
    (["--plat=bogus", "--check", "ls"], 2, None),
    (["--check-f", os.devnull], 0, ALLOWED),
    (["--check-fi=" + os.devnull], 0, ALLOWED),
    (["--check", "cd x && ls", "--self"], 0, SELF_TEST),
    # A prefix of several options is ambiguous, with `=` or without.
    (["--che", "ls"], 2, None),
    (["--che=ls"], 2, None),
    (["--s=bash"], 2, None),
    # A flag that takes no value refuses one given with `=`, the empty one too.
    (["--help="], 2, None),
    (["--self-test="], 2, None),
    (["--sh="], 2, None),                     # a value option: '' is no shell
    (["--tool=", "--check", "ls"], 0, ALLOWED),   # ... but it is a tool name
    # -h glued to more h's is that many -h; to `-` or `=` it is refused by every
    # Python. (To a letter it depends on the version: no row, see the port.)
    (["-hh"], 0, HELP),
    (["--zz", "-hhh"], 0, HELP),
    (["-h-", "--help"], 2, None),
    (["-hh=x"], 2, None),
    # Unrecognised tokens are reported after the parse, so a --help anywhere
    # still prints; every other refusal stops where parsing reaches it.
    (["-x", "--help"], 0, HELP),
    (["--zz=1", "--help"], 0, HELP),
    (["foo", "--help"], 0, HELP),
    (["-1", "--help"], 0, HELP),
    (["-", "--help"], 0, HELP),
    (["", "--help"], 0, HELP),
    (["--help", "foo"], 0, HELP),
    (["--shell", "zsh", "--help"], 2, None),
    (["--help", "--shell", "zsh"], 0, HELP),
    (["--zz", "--check", "ls"], 2, None),     # ... and a verdict never prints
    (["--check", "ls", "foo"], 2, None),
    (["--self-test", "--zz"], 2, None),
    # `--` ends the options, and nothing here takes what follows it.
    (["--"], 2, None),
    (["--help", "--"], 0, HELP),
    (["--check", "ls", "--", "--help"], 2, None),
    (["--", "--check", "ls"], 2, None),
    (["--check", "--", "ls"], 2, None),
    (["--=x", "--help"], 2, None),            # `--` before `=`: every option
    # ... but glued to an option it is that option's value: the command `--`, a
    # path, a tool name, no choice. 3.9/3.10 argparse gave [] instead -- a
    # traceback (exit 1: "denied") or a pass; _Parser._get_values pins 3.11+.
    (["--check=--"], 0, ALLOWED),
    (["--check-file=--"], 2, None),           # no file of that name
    (["--tool=--", "--check", "sleep 5"], 1, denied("sleep 5", tool="--")),
    (["--shell=--", "--check", "ls"], 2, None),
    (["--platform=--", "--check", "ls"], 2, None),
    # Values that only look like flags, and repeats: the last one wins.
    (["--check", "--zz x"], 0, ALLOWED),
    (["--check", "--sh x"], 0, ALLOWED),      # no `=`: the space decides
    (["--check", "-"], 0, ALLOWED),
    (["--check=cd x && ls", "--check=ls"], 0, ALLOWED),
    (["--check=ls", "--check=cd x && ls"], 1, denied("cd x && ls")),
]


def reference_options() -> "dict[str, bool]":
    """Every option string of the reference's CLI -> whether it takes a value.

    The port's kOptions is a hand copy of exactly this; the pytest suite
    compares the two without a binary, and _option_rows below makes the
    comparison behavioural where there is one.
    """
    return {option: action.nargs != 0
            for action in reference.build_parser()._actions
            for option in action.option_strings}


def _option_rows() -> list:
    """Two ARGV_CASES rows per option string, and two per shortest unambiguous
    prefix of each long one: `[x, "--help"]` and `[x=v, "--help"]`.

    An option nobody knows is reported after the parse, so --help wins both
    rows. A value option fails the first (a flag is no value) and prints the
    help on the second; a flag prints the help on the first and refuses the
    `=` of the second. So every real spelling answers unlike an unknown one,
    and one missing from the port's kOptions makes a DIFF -- generated from
    the reference's own parser, so an option added there needs no row by hand.
    """
    options = reference_options()
    actions = reference.build_parser()._option_string_actions
    rows = []
    for option, takes_value in options.items():
        spellings = [option]
        for end in range(3, len(option)) if option.startswith("--") else ():
            if [name for name in options if name.startswith(option[:end])] == [option]:
                spellings.append(option[:end])
                break
        choices = actions[option].choices
        value = choices[0] if choices else "x"
        for spelling in spellings:
            rows.append(([spelling, "--help"],) + ((2, None) if takes_value else (0, HELP)))
            rows.append(([f"{spelling}={value}", "--help"],)
                        + ((0, HELP) if takes_value else (2, None)))
    return rows


ARGV_CASES += _option_rows()

# Hook mode: the path that actually runs. The first group is ordinary traffic;
# the rest is what a JSON reader has to survive without taking the session with
# it. The contract is fail OPEN -- a payload neither half understands must leave
# the call alone -- so a non-zero exit is a failure here even when the two agree.
def _payload(**fields) -> bytes:
    return json.dumps(fields).encode()


DENY = {"tool_name": "Bash", "tool_input": {"command": "cd x && ls"}}

HOOK_CASES = [
    ("ordinary allow", _payload(tool_name="Bash", cwd=os.getcwd(),
                                tool_input={"command": "git status"})),
    ("ordinary deny", _payload(**DENY)),
    ("powershell deny", _payload(tool_name="PowerShell",
                                 tool_input={"command": "Set-Location x; ls"})),
    ("monitor deny", _payload(tool_name="Monitor",
                              tool_input={"command": "while true; do sleep 5; done"})),
    ("unhandled tool", _payload(tool_name="Read", tool_input={"file_path": "x"})),
    ("monitor without a command", _payload(tool_name="Monitor",
                                           tool_input={"ws": "x"})),
    ("cwd elsewhere", _payload(tool_name="Bash", cwd=PLUGIN,
                               tool_input={"command": "sed -i s/a/b/ f"})),
    ("non-ascii command", _payload(tool_name="Bash",
                                   tool_input={"command": "cd \u044f && ls"})),
    ("empty stdin", b""),
    ("not json", b"{not json"),
    ("truncated", b'{"tool_name": "Bash"'),
    ("top-level array", b"[1,2,3]"),
    ("top-level string", b'"hello"'),
    ("nesting 100", b"[" * 100 + b"]" * 100),
    ("nesting 10k", b"[" * 10000 + b"]" * 10000),
    ("nesting 200k unclosed", b"[" * 200000),
    ("nested objects 50k", b'{"a":' * 50000 + b"1" + b"}" * 50000),
    ("lone high surrogate", b'{"tool_name":"\\ud800","tool_input":{"command":"cd x && ls"}}'),
    ("lone low surrogate", b'{"tool_name":"\\udc00","tool_input":{"command":"cd x && ls"}}'),
    ("bad \\u escape", b'{"tool_name":"\\uZZZZ"}'),
    ("unknown escape", b'{"tool_name":"\\q"}'),
    ("unterminated string", b'{"tool_name":"Bash'),
    ("NUL in string", b'{"tool_name":"Ba\x00sh","tool_input":{"command":"cd x && ls"}}'),
    ("raw newline in string", b'{"tool_name":"Ba\nsh"}'),
    # json.load builds a dict, so a repeated key keeps the LAST value.
    ("duplicate keys", b'{"tool_name":"Read","tool_name":"Bash",'
                       b'"tool_input":{"command":"cd x && ls"}}'),
    ("command not a string", b'{"tool_name":"Bash","tool_input":{"command":123}}'),
    ("tool_input not an object", b'{"tool_name":"Bash","tool_input":"x"}'),
    ("cwd not a string", b'{"tool_name":"Bash","cwd":42,'
                         b'"tool_input":{"command":"cd x && ls"}}'),
    ("huge number", b'{"tool_name":"Bash","n":1e99999,'
                    b'"tool_input":{"command":"cd x && ls"}}'),
    ("trailing garbage", json.dumps(DENY).encode() + b" trailing"),
    ("bom", b"\xef\xbb\xbf" + json.dumps(DENY).encode()),
    # JSON whitespace is ' \t\n\r' and nothing else. A reader that also skips
    # `\v`/`\f` parses a payload json.load rejects, so one half denies while the
    # other fails open -- and no --check case can see it, because this is the
    # reader, not the scanner.
    ("vertical tab as json whitespace",
     b'{\x0b"tool_name":"Bash","tool_input":{"command":"cd x && ls"}}'),
    ("form feed as json whitespace",
     b'{\x0c"tool_name":"Bash","tool_input":{"command":"cd x && ls"}}'),
    # json.load is strict about control characters in strings and about the
    # number grammar, yet takes NaN and +-Infinity. The port's first reader got
    # all seven of these backwards: a strtod-shaped number scan took `+1`, `01`
    # and `1.`, and a raw tab went into the string.
    ("raw tab in command",
     b'{"tool_name":"Bash","tool_input":{"command":"cd x\t&& ls"}}'),
    ("raw tab in another value",
     b'{"tool_name":"Bash","x":"a\tb","tool_input":{"command":"cd x && ls"}}'),
    ("NaN", b'{"tool_name":"Bash","x":NaN,"tool_input":{"command":"cd x && ls"}}'),
    ("-Infinity", b'{"tool_name":"Bash","x":-Infinity,'
                  b'"tool_input":{"command":"cd x && ls"}}'),
    ("leading zero", b'{"tool_name":"Bash","x":01,"tool_input":{"command":"cd x && ls"}}'),
    ("trailing dot", b'{"tool_name":"Bash","x":1.,"tool_input":{"command":"cd x && ls"}}'),
    ("plus sign", b'{"tool_name":"Bash","x":+1,"tool_input":{"command":"cd x && ls"}}'),
    ("full number grammar", b'{"tool_name":"Bash","x":[-0.5e+3,0,10E-2,-0],'
                            b'"tool_input":{"command":"cd x && ls"}}'),
    # An integer literal becomes a Python int, and int() refuses more digits
    # than sys.get_int_max_str_digits() (4300 by default); a float does not.
    ("int at the digit cap", b'{"tool_name":"Bash","x":' + b"1" * 4300
                             + b',"tool_input":{"command":"cd x && ls"}}'),
    ("int over the digit cap", b'{"tool_name":"Bash","x":-' + b"1" * 4301
                               + b',"tool_input":{"command":"cd x && ls"}}'),
    ("long float", b'{"tool_name":"Bash","x":' + b"1" * 5000
                   + b'.5,"tool_input":{"command":"cd x && ls"}}'),
]


# Nearly all of this script's time is ~315 REFERENCE launches: the port starts
# in 4-35 ms, an interpreter in 0.05 s in one phase of the author's machine and
# 0.4-1.3 s in another (measured 2026-09-29; `python -I -S -c pass` the same,
# stdin irrelevant), so one launch at a time the run took 215 s and 238 s in a
# slow phase -- and 506-772 s on 2026-09-28 in a slot while a second tree ran
# the same suite, which was read as a hang. Launches are independent, so they
# share a pool. 8 and not more: in a slow phase 8 workers gave 0.59 s a launch
# against 1.35 s for one and 0.42 s for 16 (32 launches each), the whole run
# 101 s against 215-238 s, and the machine is usually running another suite.
# Capped by the CPU count, and overridden by --jobs or, for a caller that runs
# this script without choosing its argv (the pytest parity case), by the
# PARITY_JOBS environment variable: two slots each running the suite would
# otherwise start 16 interpreters at once. See resolve_jobs.
DEFAULT_JOBS = 8
JOBS_ENV = "PARITY_JOBS"

# Per launch, only to turn a hang into a failure that names its argv instead
# of an endless test. Slowest single launch, measured 2026-09-29: 6.6 s in one
# run's pool of 8 (3.3 s one at a time); with two runs of 8 at once -- what a
# second suite on the same machine looks like -- 7.0 s and 1.2 s in a fast
# phase, 12.1 s and 15.0 s in a slow one (those two runs took 120 s and 108 s).
LAUNCH_TIMEOUT = 120

# The "exit code" of a launch that outlived LAUNCH_TIMEOUT. Not an int: every
# int is some process's real return code (-1 is SIGHUP on POSIX), and a hang
# must never compare equal to one.
TIMED_OUT = "timeout"


def resolve_jobs(cli: "int | None", environ=os.environ) -> int:
    """Launches at a time: --jobs, else $PARITY_JOBS, else DEFAULT_JOBS capped
    by the CPU count. A value that is not a whole number >= 1 is an error
    (ValueError), never a quiet fall back to the default."""
    if cli is not None:
        source, value = "--jobs", cli
    elif JOBS_ENV in environ:
        raw = environ[JOBS_ENV]
        if not re.fullmatch(r"[0-9]+", raw):
            raise ValueError(f"{JOBS_ENV}={raw!r} is not a whole number")
        source, value = JOBS_ENV, int(raw)
    else:
        return min(DEFAULT_JOBS, os.cpu_count() or 1)
    if value < 1:
        raise ValueError(f"{source} must be at least 1, not {value}")
    return value


class _Launches:
    """The children in flight, so that a run stopped early -- an exception in
    a worker, Ctrl+C -- kills them rather than waiting each one out, and starts
    no new one after abort()."""

    def __init__(self):
        self._lock = threading.Lock()
        self._live = set()
        self._aborted = False

    def start(self, argv: "list[str]", stdin) -> subprocess.Popen:
        if self._aborted:
            raise RuntimeError("the parity run was aborted")
        process = subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        with self._lock:
            self._live.add(process)
            aborted = self._aborted
        # abort() ran between the check above and the add: it did not see this
        # child, so it is killed here.
        if aborted:
            process.kill()
        return process

    def finish(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._live.discard(process)

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            live = list(self._live)
        for process in live:
            process.kill()

    def reopen(self) -> None:
        """Only once no worker is left: the next main() starts afresh."""
        with self._lock:
            self._aborted = False


_LAUNCHES = _Launches()


def launch(argv: "list[str]", payload: "bytes | None" = None
           ) -> "subprocess.CompletedProcess | None":
    """One run of either half, None when it outlived LAUNCH_TIMEOUT.

    stdin is the payload or closed, never inherited: an argv that one half
    reads as "no command" puts it in hook mode, and an inherited stdin (a
    console, or the pipe of whoever started this script) would keep it waiting
    for a payload that never comes.

    The payload goes in through a temporary FILE, not a pipe, so the timeout
    covers the whole launch. communicate(input=...) sets its deadline and then
    writes the input; on Windows through 3.13 (3.13.7 read 2026-09-29; 3.14
    moved it to a thread) that write blocks this thread, so a child that never
    reads a payload larger than the pipe buffer (HOOK_CASES has 200 KB ones)
    held the launch forever, timeout or not. The price: hook mode reads a file here and a pipe in a real
    session, which both halves' readers (json.load, fread) do not tell apart.
    """
    stdin_file = None
    try:
        if payload is None:
            stdin = subprocess.DEVNULL
        else:
            stdin = stdin_file = tempfile.TemporaryFile()
            stdin_file.write(payload)
            stdin_file.seek(0)
        process = _LAUNCHES.start(argv, stdin)
    finally:
        if stdin_file is not None:
            stdin_file.close()  # the child holds its own handle
    try:
        with process:
            try:
                stdout, stderr = process.communicate(timeout=LAUNCH_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return None
    finally:
        _LAUNCHES.finish(process)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def timed_out(argv: "list[str]") -> "tuple[str, str]":
    """The verdict of a launch that never finished; it names the argv, so the
    report says which half hung on what rather than just that one did."""
    return TIMED_OUT, f"<no answer in {LAUNCH_TIMEOUT} s from {ascii(argv)}>"


def default_exe() -> str:
    name = "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate"
    return os.path.join(HOOKS, name)


def newer_sources(exe: str) -> "list[str]":
    """The port's sources edited after `exe` was built: a stale binary.

    The binary is gitignored and built by hand, so an edit to the .cpp that
    nobody rebuilt leaves the OLD port answering here -- and the comparison
    then pins yesterday's code while reading as today's. By mtime: a checkout
    that changes the source touches it too, which is right, since the binary
    no longer matches what is checked out. CMakeLists.txt is left out on
    purpose: a change there need not relink, and would read as stale forever.
    """
    built = os.path.getmtime(exe)
    sources = [os.path.join(HERE, "ask_user_gate.cpp")]
    for folder, _, names in os.walk(os.path.join(HERE, "third_party")):
        sources += [os.path.join(folder, name) for name in names]
    return [path for path in sources if os.path.getmtime(path) > built]


def check_verdict(argv: "list[str]", command: str, shell: str, tool: str,
                  scratch: str, newline: str = "") -> "tuple[int | str, str]":
    """One command through one gate's CLI, via --check-file.

    Both halves go through their COMMAND LINE, the reference included. Calling
    reference.scan() in-process instead was faster and blind by exactly the
    width of the CLI: file reading, newline translation and print()'s own CRLF
    never got compared, and a real offset bug lived in that gap.

    --check-file and not --check: the corpus carries newlines and
    10 000-character commands, and an argv is the wrong place for either. A
    file per call, because calls run concurrently (see DEFAULT_JOBS).
    """
    descriptor, path = tempfile.mkstemp(suffix=".txt", dir=scratch)
    with open(descriptor, "w", encoding="utf-8", newline=newline) as stream:
        stream.write(command)
    full = argv + ["--check-file", path, "--shell", shell, "--tool", tool,
                   "--platform", "windows"]
    result = launch(full)
    if result is None:
        return timed_out(full)
    if result.stderr:
        return result.returncode, ("<stderr> "
                                   + result.stderr.decode("utf-8", "replace"))
    return result.returncode, result.stdout.decode("utf-8").replace("\r\n", "\n")


def argv_verdict(argv: "list[str]", arguments: "list[str]") -> "tuple[int | str, str]":
    """One ARGV_CASES argv through one gate's CLI, host fixed to Windows.

    `--platform windows` goes FIRST so that a case can end on a bare --check.
    A usage error (exit 2) and the help keep only their code, see ARGV_CASES,
    and so does a passing self-test: the two halves count different checks.
    """
    full = argv + ["--platform", "windows"] + arguments
    result = launch(full)
    if result is None:
        return timed_out(full)
    if result.returncode == 2:
        return 2, "<usage error>"
    if result.stderr:
        return result.returncode, ("<stderr> "
                                   + result.stderr.decode("utf-8", "replace"))
    text = result.stdout.decode("utf-8").replace("\r\n", "\n")
    if result.returncode == 0 and text.startswith("usage:"):
        return 0, HELP
    if result.returncode == 0 and re.fullmatch(r"\d+/\d+ checks pass\n", text):
        return 0, SELF_TEST
    return result.returncode, text


def hook_verdict(argv: "list[str]", payload: bytes) -> "tuple[int | str, str]":
    """One payload through one gate's HOOK mode -- the path that runs 100k times
    a month, and the only one that exercises the JSON reader, the tool_name
    routing, `cwd` and the escaping of the reason into the envelope."""
    result = launch(argv, payload)
    if result is None:
        return timed_out(argv)
    body = result.stdout.decode("utf-8", "replace").strip()
    if not body:
        return result.returncode, "<no verdict>"
    try:
        decoded = json.loads(body)
    except ValueError:
        return result.returncode, "<unparseable> " + body
    inner = decoded.get("hookSpecificOutput", {})
    # The reason is compared decoded, not as bytes: json.dump defaults to
    # ensure_ascii and the port emits raw UTF-8, which is the same value spelled
    # two ways and only visible at all under a non-ASCII install path.
    return result.returncode, "\n".join([
        str(inner.get("hookEventName")), str(inner.get("permissionDecision")),
        str(inner.get("permissionDecisionReason"))])


def normalise(text: str) -> str:
    """Erase the one difference that is by design: which copy did the refusing.

    Keyed on the sentence, not on a path spelling handed in from outside: the
    binary prints GetModuleFileNameW's idea of its own path, so matching a
    caller-supplied --exe made every case a DIFF whenever the two disagreed on
    case or on being relative. Only the path is erased, never the sentence --
    a port that dropped the rest of that line should still fail here.
    """
    # One marker per header render() can print.
    markers = (": this command would stop the session",
               ": this command would outlive the job")
    lines = text.strip("\n").split("\n")
    for index, line in enumerate(lines):
        for marker in markers:
            if line.startswith("Blocked by ") and marker in line:
                lines[index] = "Blocked by <gate>" + line[line.index(marker):]
                break
    return "\n".join(lines)


@contextlib.contextmanager
def launch_pool(jobs: int):
    """A ThreadPoolExecutor that an exception LEAVES rather than drains.

    main() queues every launch up front, and a plain `with` executor waits for
    all of them on the way out: a decode error in the first case surfaced only
    after all ~630 launches had run, and Ctrl+C the same. Here any exception --
    from a worker, via result(), or KeyboardInterrupt -- cancels what is still
    queued, kills the children in flight and reaps them, then propagates. On
    Windows a console Ctrl+C reaches those children as well, but not the ones
    a worker starts after it; the kill covers both.
    """
    pool = concurrent.futures.ThreadPoolExecutor(jobs)
    try:
        yield pool
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        _LAUNCHES.abort()
        pool.shutdown(wait=True)
        _LAUNCHES.reopen()
        raise
    pool.shutdown(wait=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", default=default_exe(),
                        help="the built gate (default: next to hooks.json)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jobs", type=int, default=None,
                        help=f"launches at a time (default: ${JOBS_ENV}, else "
                             f"{DEFAULT_JOBS} capped by the CPU count)")
    options = parser.parse_args()
    try:
        jobs = resolve_jobs(options.jobs)
    except ValueError as error:
        parser.error(str(error))

    if not os.path.isfile(options.exe):
        print(f"{options.exe} is not there -- build it with "
              f"`python cpp/build.py`", file=sys.stderr)
        return 2
    # Only the binary built from THIS checkout: one given with --exe may come
    # from anywhere, and its mtime says nothing about these sources.
    stale = (newer_sources(options.exe)
             if os.path.normcase(os.path.abspath(options.exe))
             == os.path.normcase(default_exe()) else [])
    if stale:
        print(f"{options.exe} is older than {', '.join(stale)} -- a stale "
              f"binary; rebuild it with `python cpp/build.py`", file=sys.stderr)
        return 2

    reference_argv = [sys.executable, os.path.join(HOOKS, "ask_user_gate.py")]
    gate_argv = [options.exe]
    corpus = [(case[0], case[1], case[3] if len(case) > 3 else "Bash")
              for case in reference.SELF_TEST_CASES] + EXTRA_CASES

    failures = 0
    compared = 0
    # Every launch is queued up front and read back in corpus order, so the
    # report is the one a sequential run prints. The pool is left before the
    # scratch directory the --check-file cases write into.
    with tempfile.TemporaryDirectory() as scratch, launch_pool(jobs) as pool:
        def both(verdict, *args):
            return (pool.submit(verdict, reference_argv, *args),
                    pool.submit(verdict, gate_argv, *args))

        check_runs = [(command, shell, tool, newline,
                       both(check_verdict, command, shell, tool, scratch, newline))
                      for newline, cases in (("", corpus), ("\r\n", CRLF_CASES))
                      for command, shell, tool in cases]
        argv_runs = [(arguments, both(argv_verdict, arguments))
                     for arguments, _, _ in ARGV_CASES]
        hook_runs = [(label, both(hook_verdict, payload))
                     for label, payload in HOOK_CASES]
        self_test_runs = [(label, command, pool.submit(launch, command))
                          for label, command in (
                              ("python", reference_argv + ["--self-test"]),
                              ("c++", gate_argv + ["--self-test"]))]

        for command, shell, tool, newline, (py_run, cpp_run) in check_runs:
            compared += 1
            py_code, py_text = py_run.result()
            cpp_code, cpp_text = cpp_run.result()
            shown = command if len(command) < 50 else command[:47] + "..."
            shown = shown.replace("\n", "\\n").replace("\v", "\\v")
            tag = f"{shell}/{tool}" + ("/crlf" if newline else "")
            if (py_code == cpp_code
                    and normalise(py_text) == normalise(cpp_text)):
                if options.verbose:
                    print(f"ok   [{tag}] {shown!r} -> "
                          f"{'denied' if py_code else 'allowed'}")
                continue
            failures += 1
            print(f"DIFF [{tag}] {shown!r}", file=sys.stderr)
            print(f"  python (exit {py_code}):\n{normalise(py_text)}",
                  file=sys.stderr)
            print(f"  c++    (exit {cpp_code}):\n{normalise(cpp_text)}",
                  file=sys.stderr)

        for arguments, (py_run, cpp_run) in argv_runs:
            compared += 1
            py_code, py_text = py_run.result()
            cpp_code, cpp_text = cpp_run.result()
            if (py_code == cpp_code
                    and normalise(py_text) == normalise(cpp_text)):
                # ascii(), not repr(): a case carries a non-ASCII digit, and a
                # cp1252 console raises on it mid-report.
                if options.verbose:
                    print(f"ok   [argv] {ascii(arguments)} -> exit {py_code}")
                continue
            failures += 1
            print(f"DIFF [argv] {ascii(arguments)}\n"
                  f"  python (exit {py_code}):\n{normalise(py_text)}\n"
                  f"  c++    (exit {cpp_code}):\n{normalise(cpp_text)}",
                  file=sys.stderr)

        for label, (py_run, cpp_run) in hook_runs:
            compared += 1
            py_code, py_text = py_run.result()
            cpp_code, cpp_text = cpp_run.result()
            # Fail OPEN is the contract, so a crash or a hang is a failure even
            # when both halves manage it: `catch (...)` does not see a Windows
            # stack overflow, and a dead hook returns no verdict at all.
            crashed = [f"{name} {text}" if code == TIMED_OUT
                       else f"{name} exited {code}"
                       for name, code, text in (("python", py_code, py_text),
                                                ("c++", cpp_code, cpp_text))
                       if code != 0]
            if crashed:
                failures += 1
                print(f"CRASH [hook] {label}: {', '.join(crashed)}",
                      file=sys.stderr)
                continue
            if normalise(py_text) == normalise(cpp_text):
                if options.verbose:
                    print(f"ok   [hook] {label} -> {py_text.splitlines()[1]}"
                          if py_text != "<no verdict>" else
                          f"ok   [hook] {label} -> pass-through")
                continue
            failures += 1
            print(f"DIFF [hook] {label}\n  python:\n{normalise(py_text)}\n"
                  f"  c++:\n{normalise(cpp_text)}", file=sys.stderr)

        # Both self-tests too: parity says the two agree, not that either is
        # right. Two implementations can agree on a wrong answer, and the
        # wiring and path checks live only in the self-tests -- nothing in the
        # corpus reaches them.
        broken = 0
        for label, command, run in self_test_runs:
            result = run.result()
            if result is None:
                text = timed_out(command)[1]
            else:
                text = (result.stdout + result.stderr).decode("utf-8",
                                                              "replace").strip()
            print(f"{label} --self-test: {text}")
            broken += 1 if result is None or result.returncode else 0

    print(f"{compared - failures}/{compared} parity cases agree")
    return 1 if (failures or broken) else 0


if __name__ == "__main__":
    sys.exit(main())
