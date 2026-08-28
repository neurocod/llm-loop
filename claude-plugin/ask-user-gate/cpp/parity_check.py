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
import json
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
]


def default_exe() -> str:
    name = "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate"
    return os.path.join(HOOKS, name)


def check_verdict(argv: "list[str]", command: str, shell: str, tool: str,
                  scratch: str, newline: str = "") -> "tuple[int, str]":
    """One command through one gate's CLI, via --check-file.

    Both halves go through their COMMAND LINE, the reference included. Calling
    reference.scan() in-process instead was faster and blind by exactly the
    width of the CLI: file reading, newline translation and print()'s own CRLF
    never got compared, and a real offset bug lived in that gap.

    --check-file and not --check: the corpus carries newlines and
    10 000-character commands, and an argv is the wrong place for either.
    """
    path = os.path.join(scratch, "command.txt")
    with open(path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write(command)
    result = subprocess.run(
        argv + ["--check-file", path, "--shell", shell, "--tool", tool,
                "--platform", "windows"],
        capture_output=True)
    if result.stderr:
        return result.returncode, ("<stderr> "
                                   + result.stderr.decode("utf-8", "replace"))
    return result.returncode, result.stdout.decode("utf-8").replace("\r\n", "\n")


def hook_verdict(argv: "list[str]", payload: bytes) -> "tuple[int, str]":
    """One payload through one gate's HOOK mode -- the path that runs 100k times
    a month, and the only one that exercises the JSON reader, the tool_name
    routing, `cwd` and the escaping of the reason into the envelope."""
    try:
        result = subprocess.run(argv, input=payload, capture_output=True,
                                timeout=60)
    except subprocess.TimeoutExpired:
        return -1, "<timeout>"
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
    marker = ": this command would stop the session"
    lines = text.strip("\n").split("\n")
    for index, line in enumerate(lines):
        if line.startswith("Blocked by ") and marker in line:
            lines[index] = "Blocked by <gate>" + line[line.index(marker):]
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

    reference_argv = [sys.executable, os.path.join(HOOKS, "ask_user_gate.py")]
    gate_argv = [options.exe]
    corpus = [(case[0], case[1], case[3] if len(case) > 3 else "Bash")
              for case in reference.SELF_TEST_CASES] + EXTRA_CASES

    failures = 0
    compared = 0
    with tempfile.TemporaryDirectory() as scratch:
        for newline, cases in (("", corpus), ("\r\n", CRLF_CASES)):
            for command, shell, tool in cases:
                compared += 1
                py_code, py_text = check_verdict(reference_argv, command, shell,
                                                 tool, scratch, newline)
                cpp_code, cpp_text = check_verdict(gate_argv, command, shell,
                                                   tool, scratch, newline)
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

        for label, payload in HOOK_CASES:
            compared += 1
            py_code, py_text = hook_verdict(reference_argv, payload)
            cpp_code, cpp_text = hook_verdict(gate_argv, payload)
            # Fail OPEN is the contract, so a crash or a hang is a failure even
            # when both halves manage it: `catch (...)` does not see a Windows
            # stack overflow, and a dead hook returns no verdict at all.
            crashed = [name for name, code in (("python", py_code),
                                               ("c++", cpp_code)) if code != 0]
            if crashed:
                failures += 1
                print(f"CRASH [hook] {label}: {', '.join(crashed)} exited "
                      f"non-zero (python {py_code}, c++ {cpp_code})",
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

    print(f"{compared - failures}/{compared} parity cases agree")
    return 1 if (failures or broken) else 0


if __name__ == "__main__":
    sys.exit(main())
