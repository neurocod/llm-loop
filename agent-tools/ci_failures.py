#!/usr/bin/env python3
r"""Which jobs of a GitHub Actions run failed, and their pytest failures, once.

    python ci_failures.py                       # latest run for HEAD of the repo in cwd
    python ci_failures.py --run 42537657856 --repo neurocod/llm-loop
    python ci_failures.py --sha 0ca0abdb --workflow tests --wait
    python ci_failures.py --json --raw-dir ci-logs

Prints one line per job (`<job name>: <conclusion> [@ <failed step>]`), then
every pytest failure of the failed jobs ONCE, with the jobs it hit: a matrix of
5 red jobs that share one broken test is one block, not five. Blocks are told
apart by their text after `normalize` takes the per-job noise out (runner
paths, temp dirs, interpreter paths, addresses, clock times, durations); the
block shown is the first job's exact text. A failed job with no pytest blocks
(an install step failed, a crash) keeps its last `--tail` lines instead, and
the report says so. Every failed job's whole log goes to `--raw-dir` (a fresh
temp dir by default), next to `report.txt`, the report without any caps.

Why: an agent fixing a red CI otherwise asks the human to copy the failed
output into the chat, or reads `gh run view --log-failed` whole -- 1633 lines
for the 5 red jobs of run 42537657856, where the distinct failures are 3.

Needs the GitHub CLI (`gh`), logged in. It refuses to guess: no run for the
commit, a run still going, a missing or logged-out gh and any other gh error
each exit with their own code (EXIT_*), never 0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# Exit codes. 0 and 1 answer the question (green / red, report printed); every
# other code means there is no answer yet, so a caller can tell "the run is
# red" from "this tool could not look".
EXIT_GREEN = 0
EXIT_RED = 1
# argparse's own code, also for input that cannot be derived (no --repo and no
# GitHub `origin`, no --sha/--run and no git HEAD).
EXIT_USAGE = 2
# No such run, repository, or no run for the commit (not pushed yet?).
EXIT_NOT_FOUND = 3
# The run has not completed: no --wait, or --wait ran out of --timeout.
EXIT_RUNNING = 4
# gh is not installed, or `gh auth status` fails.
EXIT_NO_GH = 5
# Any other gh failure: network, API error, unreadable output.
EXIT_GH_ERROR = 6

# Where the GitHub CLI's Windows installer puts gh; a shell started before the
# install has it outside PATH.
WINDOWS_GH = r"C:\Program Files\GitHub CLI\gh.exe"
INSTALL_HINT = "install the GitHub CLI from https://cli.github.com/ and run `gh auth login`"

# `--wait`'s default ceiling. The 8 runs of llm-loop's `tests` workflow before
# 2026-11-09 took 9.3-15.4 min from creation to the last update (15.4 min:
# run 42478828885, 10.9 min: run 42537657856), so 30 min is twice the slowest
# measured and still ends a hung run within one coffee.
WAIT_TIMEOUT_S = 1800
# `--wait`'s status poll. One `gh run view --json status` is one API request;
# 30 s keeps a 30-minute wait at 60 of the 5000 requests an hour a token has.
POLL_S = 30
# One gh call. `--log-failed` of the 12 red jobs of run 42525390878 took 1.5 s
# (2026-11-09); gh falls back to one request per job when the zip lacks job
# names, unmeasured here, so the ceiling is generous rather than tight.
GH_TIMEOUT_S = 120
GIT_TIMEOUT_S = 30

# A failed job without pytest blocks keeps this many lines, ending at its last
# `##[error]` line (the runner's "Process completed with exit code N.").
TAIL_LINES = 40
# The printed report's cap; the full report is always in `report.txt`.
MAX_LINES = 250
# A printed block longer than BLOCK_HEAD + BLOCK_TAIL lines keeps its head (the
# test's source) and its tail (the `E` lines and the location), which is where
# pytest puts the cause; the file report keeps the whole block.
BLOCK_HEAD = 15
BLOCK_TAIL = 45
# A summary line can carry a whole repr; the headline is cut at this width.
HEADLINE_CHARS = 240

# Job conclusions that are not a failure. Everything else a completed job can
# end with (failure, timed_out, cancelled, startup_failure, ...) is one.
PASSING = frozenset({"success", "skipped", "neutral"})
FAILED_STEP = frozenset({"failure", "timed_out", "cancelled"})


class Failure(Exception):
    """A reason this tool cannot answer, with the exit code that names it."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- processes


def run_cmd(argv: list, timeout: float, cwd: str | None = None) -> subprocess.CompletedProcess | None:
    """`subprocess.run` with the script's text settings; None when the
    executable cannot be started or the call times out (the caller names it)."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired):
        return None


def find_gh(explicit: str | None) -> str | None:
    """`--gh`, else gh on PATH, else the Windows installer's path; None if none exists."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    found = shutil.which("gh")
    if found:
        return found
    return WINDOWS_GH if os.path.isfile(WINDOWS_GH) else None


# gh's stderr for a run id or repository that does not exist (live, gh 2.x,
# 2026-11-09): "failed to get run: HTTP 404: Not Found (https://api...)".
NOT_FOUND_RE = re.compile(r"HTTP 404|could not find|not found", re.IGNORECASE)


class Gh:
    def __init__(self, path: str, repo: str):
        self.path, self.repo = path, repo

    def call(self, args: list) -> str:
        argv = [self.path, *args, "--repo", self.repo]
        proc = run_cmd(argv, GH_TIMEOUT_S)
        if proc is None:
            raise Failure(EXIT_GH_ERROR, f"gh did not answer within {GH_TIMEOUT_S} s: {' '.join(args)}")
        if proc.returncode != 0:
            said = (proc.stderr or proc.stdout or "").strip()
            code = EXIT_NOT_FOUND if NOT_FOUND_RE.search(said) else EXIT_GH_ERROR
            raise Failure(code, f"gh {' '.join(args)} failed (exit {proc.returncode}): {said}")
        return proc.stdout

    def json(self, args: list):
        out = self.call(args)
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise Failure(EXIT_GH_ERROR, f"gh {' '.join(args)} returned invalid JSON: {exc}") from exc


def check_auth(gh_path: str) -> None:
    """`gh auth status` must pass before any API call: logged out, gh answers
    a private repository with the same 404 as a missing run."""
    proc = run_cmd([gh_path, "auth", "status"], GH_TIMEOUT_S)
    if proc is None:
        raise Failure(EXIT_NO_GH, f"cannot run {gh_path}; {INSTALL_HINT}")
    if proc.returncode != 0:
        said = (proc.stderr or proc.stdout or "").strip()
        raise Failure(EXIT_NO_GH, f"gh auth status failed: {said}\n{INSTALL_HINT}")


def git(args: list, cwd: str | None) -> str | None:
    """One git call's stripped stdout, or None when it fails."""
    proc = run_cmd(["git", *args], GIT_TIMEOUT_S, cwd=cwd)
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


REMOTE_RE = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")


def repo_from_remote(url: str) -> str | None:
    """`OWNER/NAME` of an https or ssh GitHub remote URL, None for anything else."""
    m = REMOTE_RE.search(url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def derive_repo(cwd: str | None) -> str:
    url = git(["remote", "get-url", "origin"], cwd)
    if not url:
        raise Failure(EXIT_USAGE, "no --repo given and the cwd has no git remote `origin`")
    repo = repo_from_remote(url)
    if not repo:
        raise Failure(EXIT_USAGE, f"no --repo given and origin is not a GitHub URL: {url}")
    return repo


def derive_sha(sha: str | None, cwd: str | None) -> str:
    """The full sha of `--sha` (expanded through git when it resolves) or of HEAD."""
    if sha:
        if re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            return sha.lower()
        return git(["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"], cwd) or sha.lower()
    head = git(["rev-parse", "HEAD"], cwd)
    if not head:
        raise Failure(EXIT_USAGE, "no --run/--sha given and the cwd is not a git repository")
    return head


RUN_FIELDS = "databaseId,headSha,status,conclusion,workflowName,createdAt"


def select_run(gh: Gh, sha: str, workflow: str | None) -> tuple[int, list]:
    """(the newest run for `sha`, the newest run of every OTHER workflow for it).

    `gh run list --commit` filters by the full sha only, so a short one that
    git could not expand is matched by prefix among the latest 100 runs."""
    args = ["run", "list", "--json", RUN_FIELDS]
    if workflow:
        args += ["--workflow", workflow]
    if len(sha) == 40:
        runs = gh.json(args + ["--commit", sha, "--limit", "20"])
    else:
        runs = [r for r in gh.json(args + ["--limit", "100"])
                if str(r.get("headSha", "")).startswith(sha)]
    if not runs:
        what = f" of workflow {workflow!r}" if workflow else ""
        raise Failure(EXIT_NOT_FOUND, f"no run{what} for commit {sha} in {gh.repo} (not pushed yet?)")
    runs.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
    chosen = runs[0]
    others, seen = [], {chosen.get("workflowName")}
    for r in runs[1:]:
        if r.get("workflowName") not in seen:
            seen.add(r.get("workflowName"))
            others.append(r)
    return int(chosen["databaseId"]), others


VIEW_FIELDS = "databaseId,workflowName,headSha,status,conclusion,url,displayTitle,attempt,jobs"


def fetch_run(gh: Gh, run_id: int) -> dict:
    return gh.json(["run", "view", str(run_id), "--json", VIEW_FIELDS])


def wait_run(gh: Gh, run: dict, timeout: float) -> dict:
    """Poll until `run` completes or `timeout` passes; the last view either way."""
    deadline = time.monotonic() + timeout
    status = run.get("status")
    print(f"waiting for run {run.get('databaseId')} ({status}) up to {timeout:g} s", file=sys.stderr, flush=True)
    while run.get("status") != "completed":
        left = deadline - time.monotonic()
        if left <= 0:
            break
        time.sleep(min(POLL_S, left))
        run = fetch_run(gh, run["databaseId"])
        if run.get("status") != status:
            status = run.get("status")
            print(f"  run is {status}", file=sys.stderr, flush=True)
    return run


# ---------------------------------------------------------------- the log


# `gh run view --log-failed` prefixes every line with `<job>\t<step>\t`. The
# step is "UNKNOWN STEP" on every line of the live runs (gh says why in its
# help: the zip does not always map lines to steps), so the failed step comes
# from the jobs JSON instead. The text starts with the runner's ISO timestamp;
# a job's first line carries a BOM before it.
LINE_RE = re.compile(r"^([^\t]*)\t([^\t]*)\t(.*)$")
TS_RE = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ?")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
NEWLINE_RE = re.compile(r"\r\n|\r|\n")


def split_log(text: str) -> dict:
    """{job name: [its raw lines, timestamps kept]}, in log order. A line
    without the prefix continues the previous job (a stray CR mid-line)."""
    jobs: dict = {}
    current = None
    for line in NEWLINE_RE.split(text.lstrip("\ufeff")):
        m = LINE_RE.match(line)
        if m:
            current = m.group(1).lstrip("\ufeff")
            jobs.setdefault(current, []).append(m.group(3))
        elif current is not None and line:
            jobs[current].append(line)
    return jobs


def clean(line: str) -> str:
    """A raw log line without its timestamp and colour codes."""
    return ANSI_RE.sub("", TS_RE.sub("", line, count=1)).rstrip()


BANNER_RE = re.compile(r"^=+ (.+?) =+$")
BLOCK_RE = re.compile(r"^_+ (.+?) _+$")
ENTRY_RE = re.compile(r"^(FAILED|ERROR) (\S.*)$")
OUTCOME_RE = re.compile(
    r"^=*\s*(?:\d+ (?:failed|passed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b"
    r"|no tests ran\b).*\bin \d+(?:\.\d+)?s\b")


def parse_pytest(lines: list) -> tuple[list, list, list]:
    """(blocks, summary entries, outcome lines) of cleaned log lines.

    A block is (title, lines) of a `= FAILURES =` or `= ERRORS =` section,
    split at pytest's `___ title ___` headers (not at `_ _ _ _`, its separator
    between chained tracebacks). An entry is a `short test summary info` line
    with its continuation lines: a multi-line message goes on unprefixed.
    A section ends at the next `===` banner, the `N failed, ... in Xs` line
    (`-q` prints it without banners) or a runner `##[` line. Several pytest
    runs in one job simply add up."""
    blocks, entries, outcomes = [], [], []
    section = None
    for line in lines:
        if OUTCOME_RE.match(line):
            outcomes.append(line.strip("= "))
            section = None
            continue
        banner = BANNER_RE.match(line)
        if banner:
            name = banner.group(1).strip()
            section = ("blocks" if name in ("FAILURES", "ERRORS")
                       else "summary" if name == "short test summary info" else None)
            continue
        if line.startswith("##["):
            section = None
            continue
        if section == "blocks":
            head = BLOCK_RE.match(line)
            if head and not re.fullmatch(r"[_ ]+", head.group(1)):
                blocks.append((head.group(1), []))
            elif blocks:
                blocks[-1][1].append(line)
        elif section == "summary":
            if ENTRY_RE.match(line):
                entries.append([line])
            elif entries:
                entries[-1].append(line)
    for _, body in blocks:
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
    return blocks, entries, outcomes


def entry_nodeid(entry_line: str) -> str:
    """`FAILED tests/x.py::T::t[a] - msg` -> `tests/x.py::T::t[a]`."""
    rest = ENTRY_RE.match(entry_line).group(2)
    return rest.split(" - ", 1)[0].strip()


def block_matches(title: str, nodeid: str) -> bool:
    """Does a block titled `title` belong to the summary entry of `nodeid`?
    Titles drop the file and join a class to its test with a dot; error
    blocks say `ERROR at setup of t` or `ERROR collecting tests/x.py`."""
    test = ".".join(nodeid.split("::")[1:])
    if test and (title == test or title.endswith(" of " + test)):
        return True
    return title == f"ERROR collecting {nodeid}"


def tail_before_error(lines: list, n: int) -> list:
    """The last `n` lines up to the last `##[error]` line (the job's own end;
    the post-job cleanup after it is noise), or the last `n` overall."""
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("##[error]"):
            end = i + 1
            break
    kept = [ln for ln in lines[:end] if not ln.startswith("##[endgroup]")]
    return [ln[len("##[group]"):] if ln.startswith("##[group]") else ln for ln in kept][-n:]


def failure_units(lines: list, tail: int) -> tuple[list, list]:
    """(units, outcome lines) of one failed job's cleaned lines.

    A unit is {"kind", "headline", "block"}: a summary entry with its block,
    a block without an entry, an entry without a block -- or, when the job has
    no pytest failures at all, one "tail" unit of its last lines."""
    blocks, entries, outcomes = parse_pytest(lines)
    units, used = [], set()
    for entry in entries:
        nodeid = entry_nodeid(entry[0])
        index = next((i for i, (title, _) in enumerate(blocks)
                      if i not in used and block_matches(title, nodeid)), None)
        kind = ENTRY_RE.match(entry[0]).group(1)
        if index is None:
            units.append({"kind": "pytest", "headline": entry[0], "block": entry[1:]})
        else:
            used.add(index)
            title, body = blocks[index]
            units.append({"kind": "pytest", "headline": f"{kind} {nodeid}", "block": body,
                          "title": title})
    for i, (title, body) in enumerate(blocks):
        if i not in used:
            units.append({"kind": "pytest", "headline": title, "block": body, "title": title})
    if not units:
        units.append({"kind": "tail", "headline": "no pytest failures in this job's log; its last lines",
                      "block": tail_before_error(lines, tail)})
    return units, outcomes


# What `normalize` replaces, in order. Each rule takes out a difference that
# one failure shows between matrix jobs (all seen in llm-loop's runs of
# 2026-11-08 unless said otherwise):
NOISE = [
    # The checkout: /Users/runner/work/R/R/ (macOS), /home/runner/work/R/R/
    # (Linux), D:\a\R\R\ (Windows; backslashes are slashes by now).
    (re.compile(r"(?:/Users/runner/work|/home/runner/work|\b[A-Za-z]:/a)/[^/\s'\"]+/[^/\s'\"]+/"), "<ws>/"),
    # The interpreter's stdlib: /Library/Frameworks/Python.framework/Versions/
    # 3.12/lib/python3.12/, /opt/hostedtoolcache/Python/3.12.16/x64/lib/
    # python3.12/, C:\hostedtoolcache\windows\Python\3.12.10\x64\Lib\.
    (re.compile(r"[^\s'\"(]*/lib/python3\.\d+/"), "<py>/"),
    (re.compile(r"[^\s'\"(]*hostedtoolcache/[^\s'\"]*?/Lib/"), "<py>/"),
    # Stdlib line numbers move between patch releases.
    (re.compile(r'(<py>/[^"\s]*", line )\d+'), r"\1N"),
    # pytest's tmp_path base: /private/var/folders/../T/pytest-of-runner/pytest-0,
    # /tmp/pytest-of-runner/pytest-0, C:/Users/runneradmin/AppData/Local/Temp/
    # pytest-of-runneradmin/pytest-0.
    (re.compile(r"[^\s'\"(]*pytest-of-[^/\s'\"]+/pytest-\d+"), "<tmp>"),
    (re.compile(r"(?:/private)?/var/folders/[^\s'\"]*?/T/|(?<![\w.])/tmp/|\b[A-Za-z]:/Users/[^/\s]+/AppData/Local/Temp/"), "<tmp>/"),
    (re.compile(r"\btmp[a-z0-9_]{6,}"), "tmp*"),
    (re.compile(r"\b(?:Posix|Windows)Path\("), "Path("),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "0x?"),
    (re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?Z?"), "<time>"),
    (re.compile(r"\b\d\d?:\d\d(?::\d\d)?\b"), "<hh:mm>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds)\b"), "<dur>"),
    (re.compile(r"\bpid[ =:]*\d+", re.IGNORECASE), "pid ?"),
]
# Python 3.11+ underlines the failing expression with ^/~ lines; 3.9 and 3.10
# print none, so they are left out of the comparison.
CARETS_RE = re.compile(r"^\s*(?:E\s+)?[\^~][\s\^~]*$")
# A traceback frame line, plain or inside pytest's `E` lines.
FRAME_RE = re.compile(r'^(E?\s*)File "[^"]*", line \d+')


def drop_frame_sources(lines: list) -> list:
    """`lines` without the source lines under each `File "...", line N` frame.

    Interpreters print a frame's source differently: 3.13 shows every line of
    a multi-line call and the source of a `-c` script, 3.12 only the first
    line and nothing for `<string>` (one red test of run 42537657856 printed
    both ways). The frames themselves and the exception line are the same, so
    the key keeps those: a frame's source is what is indented deeper than its
    `File` line, up to the next frame or the first line that is not."""
    out, frame_indent = [], None
    for line in lines:
        frame = FRAME_RE.match(line)
        if frame:
            frame_indent = len(frame.group(1))
            out.append(line)
            continue
        if frame_indent is not None:
            indent = len(re.match(r"^E?\s*", line).group(0))
            if indent > frame_indent and line.strip() not in ("", "E"):
                continue
            frame_indent = None
        out.append(line)
    return out


def normalize(text: str) -> str:
    """The comparison key of a failure: its text without per-job noise (NOISE,
    CARETS_RE, drop_frame_sources) and without trailing blanks. Never shown --
    the report shows the first job's exact text."""
    out = []
    lines = [ln for ln in text.replace("\\", "/").split("\n") if not CARETS_RE.match(ln)]
    for line in drop_frame_sources(lines):
        for pattern, repl in NOISE:
            line = pattern.sub(repl, line)
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def group_failures(per_job: list) -> list:
    """[(job, units)] -> groups of identical units: {"headline", "block",
    "title", "kind", "jobs", "shownFrom"}, the most widespread first, ties in
    order of first appearance."""
    groups: dict = {}
    for job, units in per_job:
        for unit in units:
            key = normalize(unit["headline"] + "\n" + "\n".join(unit["block"]))
            group = groups.get(key)
            if group is None:
                groups[key] = group = {**unit, "jobs": [], "shownFrom": job}
            if job not in group["jobs"]:
                group["jobs"].append(job)
    ordered = list(groups.values())
    ordered.sort(key=lambda g: -len(g["jobs"]))  # stable: first appearance breaks ties
    return ordered


def short_names(names: list) -> dict:
    """{name: label}: `pytest (ubuntu-latest, 3.12)` -> `ubuntu-latest 3.12`
    when every name is `<same prefix> (<matrix values>)`; else the names."""
    parts = [re.match(r"^(.*) \((.*)\)$", n) for n in names]
    if names and all(parts) and len({p.group(1) for p in parts}) == 1:
        return {n: p.group(2).replace(", ", " ") for n, p in zip(names, parts)}
    return {n: n for n in names}


# ---------------------------------------------------------------- the report


def failed_step(job: dict) -> str | None:
    return next((s.get("name") for s in job.get("steps") or []
                 if s.get("conclusion") in FAILED_STEP), None)


def is_failed(job: dict) -> bool:
    return job.get("status") == "completed" and job.get("conclusion") not in PASSING


def job_line(job: dict, outcomes: list | None = None) -> str:
    state = job.get("conclusion") or job.get("status") or "?"
    line = f"{job.get('name')}: {state}"
    step = failed_step(job)
    if step:
        line += f" @ {step}"
    if outcomes:
        line += f" -- {'; '.join(outcomes)}"
    return line


def run_header(run: dict) -> list:
    sha = str(run.get("headSha", ""))[:8]
    state = run.get("status") if run.get("status") != "completed" else run.get("conclusion")
    title = run.get("displayTitle")
    return [f"run {run.get('databaseId')} {run.get('workflowName')!r} {sha}: {state}"
            + (f" ({title})" if title else ""),
            str(run.get("url", ""))]


def render_group(index: int, group: dict, labels: dict, cap: bool, report_path: str) -> list:
    jobs = ", ".join(labels.get(j, j) for j in group["jobs"])
    headline = group["headline"]
    if len(headline) > HEADLINE_CHARS:
        headline = headline[:HEADLINE_CHARS] + "..."
    out = [f"[{index}] {headline} -- {jobs}"]
    block = group["block"]
    if len(group["jobs"]) > 1:
        out.append(f"    (text from {labels.get(group['shownFrom'], group['shownFrom'])})")
    if cap and len(block) > BLOCK_HEAD + BLOCK_TAIL:
        omitted = len(block) - BLOCK_HEAD - BLOCK_TAIL
        block = (block[:BLOCK_HEAD] + [f"... {omitted} lines omitted, full block in {report_path}"]
                 + block[-BLOCK_TAIL:])
    out.extend("    " + ln for ln in block)
    return out


def render(run: dict, outcomes: dict, groups: list, others: list, raw: dict,
           raw_dir: str, report_path: str, cap: bool) -> list:
    out = run_header(run)
    for r in others:
        out.append(f"also for this commit: run {r.get('databaseId')} {r.get('workflowName')!r}: "
                   f"{r.get('conclusion') or r.get('status')}")
    for job in run.get("jobs") or []:
        out.append(job_line(job, outcomes.get(job.get("name"))))
    failed = [j.get("name") for j in run.get("jobs") or [] if is_failed(j)]
    if groups:
        labels = short_names(failed)
        out.append("")
        out.append(f"{len(groups)} distinct failure(s) in {len(failed)} failed job(s):")
        for i, group in enumerate(groups, 1):
            out.append("")
            out.extend(render_group(i, group, labels, cap, report_path))
    if raw:
        out.append("")
        out.append(f"raw logs in {raw_dir}: " + ", ".join(os.path.basename(p) for p in raw.values()))
    return out


def safe_name(name: str, taken: set) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "job"
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}_{n}"
    taken.add(candidate)
    return candidate + ".log"


def analyse(run: dict, log_text: str, raw_dir: str, tail: int) -> tuple[dict, list, dict]:
    """(pytest outcome lines by job, failure groups, raw log path by job) of
    the failed jobs of `run`, writing each one's whole log into `raw_dir`."""
    by_job = split_log(log_text)
    outcomes, per_job, raw, taken = {}, [], {}, set()
    for job in run.get("jobs") or []:
        name = job.get("name")
        if not is_failed(job):
            continue
        lines = by_job.get(name)
        if lines is None:
            per_job.append((name, [{"kind": "missing", "headline": "no log lines for this job in "
                                    "`gh run view --log-failed`", "block": []}]))
            continue
        path = os.path.join(raw_dir, safe_name(name, taken))
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        raw[name] = path
        units, outs = failure_units([clean(ln) for ln in lines], tail)
        outcomes[name] = outs
        per_job.append((name, units))
    return outcomes, group_failures(per_job), raw


def emit(lines: list, max_lines: int, report_path: str) -> None:
    if len(lines) > max_lines:
        shown = lines[:max(max_lines - 1, 0)]
        shown.append(f"... report cut at {len(shown)} of {len(lines)} lines; full report: {report_path}")
        lines = shown
    else:
        lines = lines + [f"full report: {report_path}"]
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--run", type=int, default=None, metavar="ID", help="the run's database id")
    pick.add_argument("--sha", default=None,
                      help="the newest run for this commit (default: git HEAD of the cwd)")
    ap.add_argument("--repo", default=None, metavar="OWNER/NAME",
                    help="default: from `git remote get-url origin` of the cwd")
    ap.add_argument("--workflow", default=None, metavar="NAME", help="only runs of this workflow")
    ap.add_argument("--wait", action="store_true", help="wait for an unfinished run to complete")
    ap.add_argument("--timeout", type=float, default=WAIT_TIMEOUT_S,
                    help=f"--wait's ceiling in seconds (default {WAIT_TIMEOUT_S}, `WAIT_TIMEOUT_S`)")
    ap.add_argument("--raw-dir", default=None, metavar="DIR",
                    help="where the failed jobs' whole logs and report.txt go (default: a new temp dir)")
    ap.add_argument("--max-lines", type=int, default=MAX_LINES, metavar="N",
                    help=f"cap of the printed report (default {MAX_LINES}); report.txt has it all")
    ap.add_argument("--tail", type=int, default=TAIL_LINES, metavar="N",
                    help=f"lines kept of a failed job without pytest failures (default {TAIL_LINES})")
    ap.add_argument("--json", action="store_true", help="print jobs and grouped failures as JSON")
    ap.add_argument("--gh", default=None, metavar="PATH", help="the gh executable (default: PATH, then "
                    + WINDOWS_GH + ")")
    return ap


def answer(args: argparse.Namespace) -> int:
    gh_path = find_gh(args.gh)
    if gh_path is None:
        where = args.gh or f"PATH or {WINDOWS_GH}"
        raise Failure(EXIT_NO_GH, f"gh not found in {where}; {INSTALL_HINT}")
    cwd = os.getcwd()
    repo = args.repo or derive_repo(cwd)
    check_auth(gh_path)
    gh = Gh(gh_path, repo)
    others = []
    if args.run is not None:
        run_id = args.run
    else:
        run_id, others = select_run(gh, derive_sha(args.sha, cwd), args.workflow)
    run = fetch_run(gh, run_id)
    if run.get("status") != "completed" and args.wait:
        run = wait_run(gh, run, args.timeout)
    if run.get("status") != "completed":
        lines = run_header(run) + [job_line(j) for j in run.get("jobs") or []]
        lines.append(f"run is {run.get('status')}"
                     + (f" after waiting {args.timeout:g} s" if args.wait else "; pass --wait to wait"))
        if args.json:
            print(json.dumps({"run": run_summary(run), "jobs": [job_json(j, {}, {}) for j in run.get("jobs") or []],
                              "failures": [], "exit": EXIT_RUNNING}))
        else:
            print("\n".join(lines))
        return EXIT_RUNNING
    green = run.get("conclusion") == "success"
    raw_dir = os.path.abspath(args.raw_dir or tempfile.mkdtemp(prefix=f"ci_failures_{run_id}_"))
    os.makedirs(raw_dir, exist_ok=True)
    report_path = os.path.join(raw_dir, "report.txt")
    outcomes, groups, raw = {}, [], {}
    if any(is_failed(j) for j in run.get("jobs") or []):
        outcomes, groups, raw = analyse(run, gh.call(["run", "view", str(run_id), "--log-failed"]),
                                        raw_dir, args.tail)
    full = render(run, outcomes, groups, others, raw, raw_dir, report_path, cap=False)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(full) + "\n")
    code = EXIT_GREEN if green else EXIT_RED
    if args.json:
        print(json.dumps({"run": run_summary(run),
                          "jobs": [job_json(j, outcomes, raw) for j in run.get("jobs") or []],
                          "failures": [{k: g.get(k) for k in ("kind", "headline", "jobs", "shownFrom", "block")}
                                       for g in groups],
                          "others": others, "rawDir": raw_dir, "report": report_path, "exit": code}))
    else:
        emit(render(run, outcomes, groups, others, raw, raw_dir, report_path, cap=True),
             args.max_lines, report_path)
    return code


def run_summary(run: dict) -> dict:
    return {k: run.get(k) for k in ("databaseId", "workflowName", "headSha", "status", "conclusion",
                                    "url", "displayTitle", "attempt")}


def job_json(job: dict, outcomes: dict, raw: dict) -> dict:
    name = job.get("name")
    return {"name": name, "status": job.get("status"), "conclusion": job.get("conclusion"),
            "failedStep": failed_step(job), "outcome": outcomes.get(name, []), "rawLog": raw.get(name)}


def main(argv: list | None = None) -> int:
    # Logs carry any character; a cp1252 console must not end the report.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return answer(args)
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
