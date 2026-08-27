#!/usr/bin/env python3
"""Mutate a file, run a command, put the file back -- in ONE shell call.

The thing this replaces is a chain: sed -i the constant, run the test, sed -i
it back. That chain has three problems, and only the first is about typing.

 1. It is a chain, so the permission layer cannot match an allow-rule against
    it and the run parks on a dialog until a human notices. One call to one
    script is one rule.
 2. The restore is just the next link. If the command crashes, hangs and is
    killed, or the run is interrupted, the restore never happens and the file
    is left mutated -- silently, because the shell reports the failure of the
    command, not the state of your tree. Here the restore is a `finally` and
    runs on exceptions and Ctrl-C alike, and the restored bytes are compared
    against the original.
 3. `sed -i` with a pattern that matches nothing edits nothing and exits 0, so
    the test runs against UNMUTATED code and passes. That reads as "the pin
    holds" when nothing was ever pinned. replace_in_file refuses to write
    unless the match count is what you said, so a wrong pattern fails here
    instead of lying there.

--expect-fail is the point of the usual case: verifying that a pin actually
fails without its fix. It inverts the exit code, so a test that passes with
the fix removed -- a pin that pins nothing -- comes back as a failure of THIS
command rather than a line of output somebody has to notice.

Runs from anywhere. Use --cwd for the command's directory instead of `cd X &&`
(on Windows a cd-compound also defeats the permission classifier).

DO NOT PIPE THIS INTO `Select-Object -First N`. That cmdlet stops the upstream
process the moment it has its N objects, and stopping it is stopping THIS one:
the `finally` never runs and the file is left mutated -- the exact silent
half-state the script exists to prevent (paid for once, 2026-08-20). The
restore contract only covers deaths this process survives long enough to
handle. `-Last N` reads the stream to the end and is safe; so is redirecting to
a file and reading that. Same caution for any consumer that closes the pipe
early (`| head` under a shell that turns SIGPIPE into a kill).

Usage:
  python tools/try_patch.py --file webgame/src/sim/player.ts \\
      --old 'const RISING_SPEED = 1e-4;' --new 'const RISING_SPEED = 0;' \\
      --cwd webgame --expect-fail \\
      -- npx vitest run test/player.rising.test.ts

  # several edits in one go: repeat --file/--old/--new as a triple
  python tools/try_patch.py --file a.ts --old X --new Y \\
                            --file b.ts --old P --new Q -- npm test

  # several edits of the SAME file are fine too: each is applied on top of the
  # previous one, and the file is restored to its pre-run bytes exactly once
  python tools/try_patch.py --file a.ts --old X --new Y \\
                            --file a.ts --old P --new Q -- npm test

  python tools/try_patch.py --selftest   # pins the restore contract, no repo
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replace_in_file import (  # noqa: E402  (needs the path line above)
    EditError, apply_replacement, changed_lines, parse_count, read_text,
    write_text,
)


class Triple(argparse.Action):
    """Collect --file/--old/--new into ordered edit triples.

    argparse has no notion of "these three flags belong together", so each is
    appended to its own list and zipped at the end; a caller who passes an
    unequal number of them gets told, rather than getting the wrong pairing.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        key = option_string.lstrip("-")
        getattr(namespace, "_order").setdefault(key, []).append(values)


def collect_edits(namespace) -> list[tuple[Path, str, str]]:
    order = namespace._order
    files = order.get("file", [])
    olds = order.get("old", [])
    news = order.get("new", [])
    if not files:
        raise EditError("no --file given")
    if not (len(files) == len(olds) == len(news)):
        raise EditError(
            f"--file/--old/--new must come in triples "
            f"(got {len(files)} file, {len(olds)} old, {len(news)} new)")
    return [(Path(f), o, n) for f, o, n in zip(files, olds, news)]


@dataclass
class Touched:
    """Everything needed to put ONE file back, however many edits hit it.

    `original` is the file as it was before this run and is snapshotted the
    FIRST time a path is edited -- never reassigned. That "never" is the whole
    class: with several --file/--old/--new triples aimed at the SAME file, each
    later edit necessarily reads a text that already carries the earlier
    mutations, so keeping a per-EDIT "before" makes the last restore write a
    half-mutated text over the good one. This is not hypothetical -- it shipped:
    three edits of one file printed three "restored" lines and left two of the
    three mutations in the working tree, silently disabling two guards. Hence
    one entry per file, keyed by the resolved path so `a.ts` and `./a.ts` are
    the same file, and hence the selftest below.

    `expected` is what we last wrote, i.e. what the file should still hold when
    the command is done; a mismatch means the command rewrote the file itself
    and is reported before those changes are discarded.
    """

    display: Path   # the path as the caller spelled it, for messages
    original: str
    expected: str


def main() -> int:
    # Looked for before argparse, and only ahead of the bare `--`, so that a
    # command containing the word (`-- grep --selftest`) is still a command.
    argv = sys.argv[1:]
    head = argv[:argv.index("--")] if "--" in argv else argv
    if "--selftest" in head:
        return selftest()

    parser = argparse.ArgumentParser(
        description="Apply edits, run a command, always restore the files.")
    parser.add_argument("--file", action=Triple, metavar="FILE",
                        help="file to edit (repeatable, with --old/--new)")
    parser.add_argument("--old", action=Triple, metavar="TEXT")
    parser.add_argument("--new", action=Triple, metavar="TEXT")
    parser.add_argument("--regex", action="store_true",
                        help="treat every --old as a regex (MULTILINE)")
    parser.add_argument("--count", type=parse_count, default=1, metavar="N",
                        help="occurrences required per edit, or 'any'")
    parser.add_argument("--cwd", type=Path, default=None, metavar="DIR",
                        help="run the command here (instead of `cd DIR &&`)")
    parser.add_argument("--expect-fail", action="store_true",
                        help="succeed only if the command FAILS (pin check)")
    parser.add_argument("--keep", action="store_true",
                        help="leave the edits in place (debugging this script)")
    parser.add_argument("--selftest", action="store_true",
                        help="run this script's own restore tests and exit")
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD",
                        help="the command to run, after a bare --")
    parser.set_defaults(_order={})
    options = parser.parse_args()

    command = options.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given; put it after a bare --")

    try:
        edits = collect_edits(options)
    except EditError as exc:
        print(f"try_patch: {exc}", file=sys.stderr)
        return 2

    # Apply every edit before running anything: a half-applied mutation would
    # test a state that neither branch of the comparison describes.
    # Keyed by resolved path, so repeated edits of one file share one snapshot
    # of its pre-run bytes; insertion-ordered, so restore reports files in the
    # order the caller named them.
    touched: dict[Path, Touched] = {}
    try:
        for path, old, new in edits:
            # `before` is the file as it is NOW -- already carrying any earlier
            # edit of the same file, which is what makes stacked edits compose.
            # It is the restore snapshot only when this path is new.
            before, crlf = read_text(path)
            after, hits = apply_replacement(before, old, new, options.regex,
                                            options.count, crlf)
            write_text(path, after)
            invalidate_bytecode(path)
            key = path.resolve()
            entry = touched.get(key)
            if entry is None:
                touched[key] = Touched(path, before, after)
            else:
                entry.expected = after
            print(f"try_patch: {path}: {hits} occurrence(s) mutated")
            for line in changed_lines(before, after, limit=4):
                print(line)
    except EditError as exc:
        print(f"try_patch: {exc}", file=sys.stderr)
        restore(touched)
        return 2
    except BaseException:
        # Ctrl-C between two edits of a multi-edit run would otherwise leave the
        # earlier ones applied: the `finally` below only covers the command.
        restore(touched)
        raise

    status = 1
    started = False
    try:
        print(f"try_patch: running {' '.join(command)}", flush=True)
        status = subprocess.call([resolve_program(command[0]), *command[1:]],
                                 cwd=options.cwd, shell=False)
        started = True
    except KeyboardInterrupt:
        print("try_patch: interrupted", file=sys.stderr)
        status = 130
    except OSError as exc:
        print(f"try_patch: cannot run the command: {exc}", file=sys.stderr)
        status = 127
    finally:
        if options.keep:
            print("try_patch: --keep, files left mutated")
        elif not restore(touched):
            # A failed restore outranks whatever the command reported: the
            # tree is wrong, and that is the thing the caller must act on.
            return 3

    if options.expect_fail:
        # A command that never STARTED is not a pin that fired. Windows makes
        # this the likely outcome rather than an exotic one: `npx`, `npm` and
        # friends are .cmd shims, and shell=False cannot launch them, so
        # `--expect-fail -- npx vitest run` used to report "failed as expected"
        # having run no test at all. That reads as a green mutation table --
        # every row of it -- which is worse than no table, and it has already
        # been believed once (2026-08-18, slice U2). Say it instead.
        if not started:
            print("try_patch: the command never RAN, so it proves nothing about "
                  "the pin. On Windows a .cmd shim (npx, npm, tsc) needs its "
                  "real name -- npx.cmd -- because the command is spawned "
                  "without a shell.", file=sys.stderr)
            return 127
        if status == 0:
            print("try_patch: the command PASSED without the fix -- the pin "
                  "does not pin anything", file=sys.stderr)
            return 1
        print(f"try_patch: the command failed as expected (exit {status})")
        return 0
    return status


def resolve_program(name: str) -> str:
    """Turn a bare command name into something CreateProcess can launch.

    The command is spawned WITHOUT a shell (a shell would re-parse the caller's
    quoting), and on Windows that means the name is not resolved through
    PATHEXT: `npx`, `npm`, `tsx`, `vitest` are .cmd shims, so the obvious
    spelling raised WinError 2 and -- before the guard above -- read as a pin
    that fired. Resolving here lets the natural spelling work instead of
    demanding `npx.cmd` from every caller.

    Two traps this walks around, both measured rather than assumed:

    - since 3.12 `shutil.which` can answer with an EXTENSIONLESS file
      (cpython#109590), and `node_modules/.bin` is full of exactly those --
      bash shims CreateProcess cannot start. So a PATHEXT match is preferred
      over which()'s first answer, and only then does its answer stand;
    - once the name resolves to a .cmd, CreateProcess implicitly starts
      cmd.exe, whose escaping rules are not the ones Python quoted for
      (CVE-2024-24576, "BatBadBut"). Harmless here because the arguments are
      the caller's own command line -- but do NOT grow this into passing file
      contents or anything else untrusted through it.

    Anything unresolvable is handed back untouched: the OSError that follows is
    a better message than a guess, and non-Windows needs none of this.
    """
    if os.name != "nt" or os.path.dirname(name):
        return name
    exts = [ext for ext in os.environ.get("PATHEXT", "").split(os.pathsep) if ext]
    for ext in exts:
        found = shutil.which(name + ext)
        if found:
            return found
    return shutil.which(name) or name


def invalidate_bytecode(path: Path) -> None:
    """Drop the cached bytecode of a Python file we just rewrote.

    CPython validates a `.pyc` against the source's mtime IN WHOLE SECONDS plus
    its size. A mutation of the same length -- `= 1` for `= 2`, the commonest
    shape there is -- applied and undone inside one second leaves both fields
    unchanged, so the interpreter happily reuses bytecode for the other text.
    It goes wrong in both directions, and both are silent:

    - the `.pyc` written DURING the mutated run outlives the restore, and the
      NEXT run imports the mutant. Measured: a restored tree failed
      `test_textwidth` on two width pins, and a full run reported 17 failures
      where the mutation had caused 2;
    - the `.pyc` compiled BEFORE the run can survive the mutation, so the
      command tests unmutated code and `--expect-fail` reports "the pin does
      not pin anything" about a pin that was never challenged -- the very lie
      this script exists to prevent.

    Deleting the cache entry is what makes the outcome independent of the
    clock. It is done after mutating and again after restoring, because either
    write can be the one that collides.
    """
    if path.suffix != ".py":
        return
    stem = path.stem
    caches = [*(path.parent / "__pycache__").glob(f"{stem}.*.pyc"),
              *path.parent.glob(f"{stem}.pyc")]
    for cached in caches:
        try:
            cached.unlink()
        except OSError:
            # A held or read-only .pyc is not worth failing the run over: the
            # command below still gets the source we wrote, and the stale entry
            # is the next run's problem, which this line at least keeps rare.
            pass


def restore(touched: "dict[Path, Touched]") -> bool:
    """Put every touched file back. Returns False if any file is left wrong.

    One pass per FILE, never per edit: writing a file more than once here is
    how the old version undid its own restore (see Touched). The write is
    followed by a read-back compared byte for byte, because a write that lands
    on a read-only or externally-held file is exactly the case where a silent
    "restored" line would be a lie.

    A file that no longer holds the mutation we wrote was edited by the command
    itself -- a formatter, a codegen step, a test that rewrites a snapshot. The
    restore below discards that work, which is the right default (the caller
    asked for a temporary mutation, not a commit), but doing it silently would
    lose an edit somebody may have wanted. So it is named.
    """
    ok = True
    for entry in touched.values():
        path = entry.display
        try:
            current, _ = read_text(path)
            if current != entry.expected:
                print(f"try_patch: {path} was changed by the command; "
                      "those changes are being discarded with the mutation",
                      file=sys.stderr)
            write_text(path, entry.original)
            invalidate_bytecode(path)
            check, _ = read_text(path)
            if check != entry.original:
                ok = False
                print(f"try_patch: RESTORE FAILED for {path}", file=sys.stderr)
            else:
                print(f"try_patch: restored {path}")
        except EditError as exc:
            ok = False
            print(f"try_patch: RESTORE FAILED for {path}: {exc}",
                  file=sys.stderr)
    return ok


# --------------------------------------------------------------------------
# selftest
#
# There is no test runner under tools/, and this script must keep working when
# the repo it edits does not build, so its regression lives in the script:
# `python tools/try_patch.py --selftest`, no arguments, no fixtures, nothing
# touched outside a temp directory.
#
# Every case runs THIS FILE as a subprocess. That is deliberate: the defect
# these pin -- several edits of one file restoring the file to a half-mutated
# state -- lived in the seam between collecting edits and restoring them, and
# an in-process call to a helper would have stepped straight over it. What is
# asserted is the only thing a caller can see: the exit code, and the bytes on
# disk afterwards.
# --------------------------------------------------------------------------

GUARDS = b"guardA = true;\nguardB = true;\nguardC = true;\n"

# `pass` and a non-zero exit, spelled with this interpreter so the selftest
# needs nothing on PATH.
CMD_OK = ["--", sys.executable, "-c", "pass"]
CMD_FAIL = ["--", sys.executable, "-c", "raise SystemExit(7)"]
CMD_COUNT_FALSE = [
    "--", sys.executable, "-c",
    "import pathlib, sys;"
    "sys.exit(0 if pathlib.Path('victim.cpp').read_text().count('false') == 3"
    " else 9)",
]
CMD_REWRITE = [
    "--", sys.executable, "-c",
    "import pathlib; pathlib.Path('victim.cpp').write_text('the command '"
    "'was here\\n')",
]


class SelftestFailure(Exception):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelftestFailure(message)


def _run(work: Path, *args: str,
         env: "dict[str, str] | None" = None) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), *args],
                          cwd=work, capture_output=True, text=True, env=env)


def _victim(work: Path, body: bytes = GUARDS) -> Path:
    path = work / "victim.cpp"
    path.write_bytes(body)
    return path


def _expect_bytes(path: Path, expected: bytes, result) -> None:
    actual = path.read_bytes()
    _expect(actual == expected,
            f"file not restored: {actual!r} != {expected!r}\n"
            f"--- exit {result.returncode}\n{result.stdout}{result.stderr}")


def _flip(name: str) -> list[str]:
    return ["--file", "victim.cpp",
            "--old", f"{name} = true;", "--new", f"{name} = false;"]


def _case_stacked_edits_of_one_file(work: Path) -> None:
    """The incident: three edits of one file left two of them behind."""
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *_flip("guardC"),
                  *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_stacked_edits_all_reach_the_command(work: Path) -> None:
    """...and the restore must not be bought by dropping the later edits."""
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *_flip("guardC"),
                  *CMD_COUNT_FALSE)
    _expect(result.returncode == 0,
            f"the command did not see all 3 mutations (exit "
            f"{result.returncode})\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_same_file_spelled_two_ways(work: Path) -> None:
    victim = _victim(work)
    result = _run(work,
                  "--file", "victim.cpp",
                  "--old", "guardA = true;", "--new", "guardA = false;",
                  "--file", str(victim.resolve()),
                  "--old", "guardB = true;", "--new", "guardB = false;",
                  *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_failing_command_restores_and_reports(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *CMD_FAIL)
    _expect(result.returncode == 7,
            f"command exit not passed through: {result.returncode}")
    _expect_bytes(victim, GUARDS, result)


def _case_expect_fail_accepts_a_failing_command(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, "--expect-fail", *_flip("guardA"), *_flip("guardB"),
                  *CMD_FAIL)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_expect_fail_rejects_a_passing_command(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, "--expect-fail", *_flip("guardA"), *_flip("guardB"),
                  *CMD_OK)
    _expect(result.returncode == 1,
            f"a pin that pins nothing must fail: exit {result.returncode}")
    _expect_bytes(victim, GUARDS, result)


def _case_command_rewriting_the_file_is_reported(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *CMD_REWRITE)
    _expect("was changed by the command" in result.stderr,
            f"silent discard of the command's own edit\n{result.stderr}")
    _expect(result.stderr.count("was changed by the command") == 1,
            f"one file, one report expected\n{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_missing_pattern_rolls_the_earlier_edits_back(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"),
                  "--file", "victim.cpp",
                  "--old", "guardZ = true;", "--new", "guardZ = false;",
                  *CMD_OK)
    _expect(result.returncode == 2,
            f"a pattern that matches nothing must be loud: {result.returncode}")
    _expect("try_patch: running" not in result.stdout,
            "the command ran on a half-applied mutation")
    _expect_bytes(victim, GUARDS, result)


def _case_crlf_survives_stacked_edits(work: Path) -> None:
    body = GUARDS.replace(b"\n", b"\r\n")
    victim = _victim(work, body)
    result = _run(work, *_flip("guardA"), *_flip("guardC"), *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, body, result)


def _case_unrunnable_command_restores(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"),
                  "--", "no-such-binary-6f2a1c", "--please")
    _expect(result.returncode == 127,
            f"exit {result.returncode}: {result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_expect_fail_rejects_an_unrunnable_command(work: Path) -> None:
    """The mutation table's own trap: a command that never ran is not a pin.

    Windows makes it the likely spelling error rather than an exotic one --
    npx/npm/tsc are .cmd shims and cannot be spawned without a shell -- so
    accepting "did not start" as "failed as expected" turns a whole table of
    mutations green without running one test.
    """
    victim = _victim(work)
    result = _run(work, "--expect-fail", *_flip("guardA"),
                  "--", "no-such-binary-6f2a1c", "--please")
    _expect(result.returncode == 127,
            f"an unrunnable command must not pass as a fired pin: exit "
            f"{result.returncode}\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_pathext_shim_runs_by_its_bare_name(work: Path) -> None:
    """`-- npx vitest run` must work, and must not pick the bash shim.

    Windows only, because only there is the problem: the command is spawned
    without a shell, so a bare `npx` is unresolvable, and `node_modules/.bin`
    additionally offers an extensionless bash shim that CreateProcess cannot
    start. The fixture is that pair -- `tool.cmd` beside `tool` -- and the exit
    code says which one ran.
    """
    if os.name != "nt":
        return
    binaries = work / "bin"
    binaries.mkdir(exist_ok=True)
    (binaries / "trypatchtool.cmd").write_text("@exit /b 3\n", encoding="ascii")
    (binaries / "trypatchtool").write_text("#!/bin/sh\nexit 4\n", encoding="ascii")
    env = dict(os.environ, PATH=f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}")
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), "--", "trypatchtool", env=env)
    _expect(result.returncode == 3,
            f"the .cmd shim did not run by its bare name: exit "
            f"{result.returncode}\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_a_python_edit_leaves_no_stale_bytecode(work: Path) -> None:
    """A same-length mutation must not survive in `__pycache__`.

    The fixture is the collision itself: a module compiled before the run, a
    mutation that changes no byte count, and a whole run inside one second — the
    two fields CPython validates a `.pyc` against. What is asserted is the
    mechanism rather than the race: no cache entry for the edited file outlives
    the run, so no later interpreter can be handed the mutant (nor, on the way
    in, the pre-mutation text). See `invalidate_bytecode`.
    """
    victim = work / "victim.py"
    victim.write_bytes(b"VALUE = 1\n")
    reader = ("import sys; sys.path.insert(0, '.'); import victim; "
              "sys.exit(0 if victim.VALUE == {} else 9)")
    warm = subprocess.run([sys.executable, "-c", reader.format(1)],
                          cwd=work, capture_output=True, text=True)
    _expect(warm.returncode == 0, f"fixture broken: {warm.stderr}")
    cache = work / "__pycache__"
    _expect(any(cache.glob("victim.*.pyc")), "fixture broken: nothing cached")

    result = _run(work, "--file", "victim.py",
                  "--old", "VALUE = 1", "--new", "VALUE = 2",
                  "--", sys.executable, "-c", reader.format(2))

    _expect(result.returncode == 0,
            f"the command did not see the mutation (exit {result.returncode})"
            f"\n{result.stdout}{result.stderr}")
    _expect(not any(cache.glob("victim.*.pyc")),
            "bytecode for the mutated source outlived the restore; the next "
            "run would import the mutant")
    _expect_bytes(victim, b"VALUE = 1\n", result)
    after = subprocess.run([sys.executable, "-c", reader.format(1)],
                           cwd=work, capture_output=True, text=True)
    _expect(after.returncode == 0,
            f"a later run still saw the mutant (exit {after.returncode})"
            f"\n{after.stdout}{after.stderr}")


def _case_keep_leaves_every_mutation(work: Path) -> None:
    """--keep is the escape hatch; it must keep ALL edits, not the last one."""
    victim = _victim(work)
    result = _run(work, "--keep", *_flip("guardA"), *_flip("guardB"), *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, b"guardA = false;\nguardB = false;\nguardC = true;\n",
                  result)


SELFTEST_CASES = (
    _case_stacked_edits_of_one_file,
    _case_stacked_edits_all_reach_the_command,
    _case_same_file_spelled_two_ways,
    _case_failing_command_restores_and_reports,
    _case_expect_fail_accepts_a_failing_command,
    _case_expect_fail_rejects_a_passing_command,
    _case_expect_fail_rejects_an_unrunnable_command,
    _case_pathext_shim_runs_by_its_bare_name,
    _case_command_rewriting_the_file_is_reported,
    _case_missing_pattern_rolls_the_earlier_edits_back,
    _case_crlf_survives_stacked_edits,
    _case_a_python_edit_leaves_no_stale_bytecode,
    _case_unrunnable_command_restores,
    _case_keep_leaves_every_mutation,
)


def selftest() -> int:
    failed = 0
    with tempfile.TemporaryDirectory(prefix="try_patch_selftest_") as tmp:
        for case in SELFTEST_CASES:
            name = case.__name__[len("_case_"):]
            work = Path(tmp) / name
            work.mkdir()
            try:
                case(work)
            except SelftestFailure as exc:
                failed += 1
                print(f"FAIL {name}\n     {exc}")
            else:
                print(f"ok   {name}")
    total = len(SELFTEST_CASES)
    if failed:
        print(f"try_patch selftest: {failed}/{total} FAILED", file=sys.stderr)
        return 1
    print(f"try_patch selftest: {total} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
