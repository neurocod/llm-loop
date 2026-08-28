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

Exit 1 on any difference. Requires the binary; build it with cpp/build.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile

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
    # Nothing at all.
    ("", "bash", "Bash"),
    ("   ", "bash", "Bash"),
]


def default_exe() -> str:
    name = "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate"
    return os.path.join(HOOKS, name)


def python_verdict(command: str, shell: str, tool: str) -> "tuple[int, str]":
    findings = reference.scan(command, shell, windows=True, tool=tool)
    if not findings:
        return 0, "allowed"
    return 1, reference.render(findings)


def cpp_verdict(exe: str, command: str, shell: str, tool: str,
                scratch: str) -> "tuple[int, str]":
    # --check-file, not --check: the corpus carries newlines and 10 000-character
    # commands, and an argv is the wrong place for either.
    path = os.path.join(scratch, "command.txt")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(command)
    result = subprocess.run(
        [exe, "--check-file", path, "--shell", shell, "--tool", tool,
         "--platform", "windows"],
        capture_output=True)
    if result.stderr:
        return result.returncode, ("<stderr> "
                                   + result.stderr.decode("utf-8", "replace"))
    return result.returncode, result.stdout.decode("utf-8").replace("\r\n", "\n")


def normalise(text: str, exe: str) -> str:
    """Erase the one difference that is by design: which copy did the refusing.

    Only the path is erased, not the whole line -- the rest of that sentence is
    text like any other, and a port that dropped it should fail here.
    """
    lines = text.strip("\n").split("\n")
    if lines and lines[0].startswith(f"Blocked by {exe}:"):
        lines[0] = "Blocked by <gate>:" + lines[0][len(f"Blocked by {exe}:"):]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", default=default_exe(),
                        help="the built gate (default: next to hooks.json)")
    parser.add_argument("--verbose", action="store_true")
    options = parser.parse_args()

    if not os.path.isfile(options.exe):
        print(f"{options.exe} is not there -- build it with "
              f"`python cpp/build.py`", file=sys.stderr)
        return 2

    corpus = [(case[0], case[1], case[3] if len(case) > 3 else "Bash")
              for case in reference.SELF_TEST_CASES] + EXTRA_CASES

    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        for command, shell, tool in corpus:
            py_code, py_text = python_verdict(command, shell, tool)
            cpp_code, cpp_text = cpp_verdict(options.exe, command, shell, tool,
                                             scratch)
            shown = command if len(command) < 50 else command[:47] + "..."
            shown = shown.replace("\n", "\\n")
            same = (py_code == cpp_code
                    and normalise(py_text, reference.SELF)
                    == normalise(cpp_text, options.exe))
            if same:
                if options.verbose:
                    print(f"ok   [{shell}/{tool}] {shown!r} -> "
                          f"{'denied' if py_code else 'allowed'}")
                continue
            failures += 1
            print(f"DIFF [{shell}/{tool}] {shown!r}", file=sys.stderr)
            print(f"  python (exit {py_code}):\n"
                  f"{normalise(py_text, reference.SELF)}", file=sys.stderr)
            print(f"  c++    (exit {cpp_code}):\n"
                  f"{normalise(cpp_text, options.exe)}", file=sys.stderr)

    # Both self-tests too: parity says the two agree, not that either is right.
    # Two implementations can agree on a wrong answer, and the wiring and path
    # checks live only in the self-tests -- nothing in the corpus reaches them.
    broken = 0
    for label, command in (("python", [sys.executable,
                                       os.path.join(HOOKS, "ask_user_gate.py"),
                                       "--self-test"]),
                           ("c++", [options.exe, "--self-test"])):
        result = subprocess.run(command, capture_output=True)
        text = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
        print(f"{label} --self-test: {text}")
        broken += 1 if result.returncode else 0

    print(f"{len(corpus) - failures}/{len(corpus)} parity cases agree")
    return 1 if (failures or broken) else 0


if __name__ == "__main__":
    sys.exit(main())
