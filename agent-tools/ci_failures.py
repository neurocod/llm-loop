#!/usr/bin/env python3
r"""Which jobs of a GitHub Actions run failed, and their pytest failures, once.

    python ci_failures.py                       # the runs for HEAD of the repo in cwd
    python ci_failures.py --run 42537657856 --repo neurocod/llm-loop
    python ci_failures.py --sha 0ca0abdb --workflow tests --wait
    python ci_failures.py --json --raw-dir ci-logs

Prints one line per job (`<job name>: <conclusion> [@ <failed steps>]`), an
index of the distinct failures, then every pytest failure of the failed jobs
ONCE, with the jobs it hit: a matrix of 5 red jobs that share one broken test
is one block, not five. Blocks are told apart by their node id and their text
after `normalize` takes out the per-job noise (runner and interpreter paths,
temp dirs, object addresses, pytest's own timing totals); the block shown is
the first job's exact text. A failed job with no pytest blocks (an install
step failed, a crash, a cancelled job) keeps its last `--tail` lines instead,
and so does a job whose log goes on failing after pytest's outcome line (a
later step, a timeout). Every failed job's whole log goes to `--raw-dir` (a
fresh temp dir named by the run ids by default), next to `report.txt`, the
report without any caps.

Without `--run`, every workflow that ran for the commit (or every one named
`--workflow`) is judged: the newest run of each (workflow id, event) pair,
out of ALL the commit's runs -- a listing that cannot be shown complete exits
EXIT_INCOMPLETE. A run still going shows the failures of its jobs that
already failed (from their own logs) and exits EXIT_RUNNING. `--timeout` is
one deadline from the start: every gh call, the `--wait` polls and the log
downloads included, ends by it.

The printed report is the head -- every run's jobs, its index of distinct
failures and the paths of the saved logs, never cut -- then the failure
blocks, which `--max-lines` cuts.

`--json` prints one object (`"schema"` changes with any change of this shape):
    schema   1
    exit     the exit code below
    rawDir   the directory of the saved logs; report  report.txt in it
    runs     one object per run judged:
      run       databaseId, workflowName, event, headSha, status,
                conclusion, url, displayTitle, attempt (gh's names)
      exit      this run's own code; note  why it is not green, or null
      jobs      [{name, status, conclusion, failedSteps: [step names],
                  outcome: [pytest outcome lines], rawLog: path or null}]
      failures  [{kind: pytest | tail | missing, headline, jobs: [names],
                  shownFrom: job name, block: [lines], savedLog: path or null}]

Why: an agent fixing a red CI otherwise asks the human to copy the failed
output into the chat, or reads `gh run view --log-failed` whole -- 1633 lines
for the 5 red jobs of run 42537657856, where the distinct failures are 3.

Needs the GitHub CLI (`gh`), logged in. It refuses to guess: green (0) needs a
completed successful run whose job list is present, finished and free of
failures; anything short of that exits with its own code (EXIT_*), never 0.
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
import traceback

# Exit codes. 0 and 1 answer the question (green / red, report printed); every
# other code means there is no answer yet, so a caller can tell "the run is
# red" from "this tool could not look".
EXIT_GREEN = 0
EXIT_RED = 1
# argparse's own code, also for input that cannot be derived (no --repo and no
# GitHub `origin`, no --sha/--run and no git HEAD, a --sha too short or
# matching several commits).
EXIT_USAGE = 2
# No such run, repository, or no run for the commit (not pushed yet?).
EXIT_NOT_FOUND = 3
# A run has not completed: no --wait, or --wait ran out of --timeout. The jobs
# that already failed are reported all the same.
EXIT_RUNNING = 4
# The schema of --json (see the docstring); bumped by any change of its shape.
JSON_SCHEMA = 1
# gh is not installed, `gh auth status` fails, or its login cannot read the
# repository.
EXIT_NO_GH = 5
# Any other gh failure: network, API error, unreadable output.
EXIT_GH_ERROR = 6
# The run completed but gives no verdict: it claims success without a job
# list, with a job that has no conclusion or with every job skipped, or it
# concluded neutral/skipped/action_required/stale. Also: the commit's runs
# could not all be listed, so a red one may be missing.
EXIT_INCOMPLETE = 7
# A defect of this tool (an unexpected exception; the traceback is on stderr).
# Not 1: a crash must never read as "the run is red". 70 is sysexits' EX_SOFTWARE.
EXIT_INTERNAL = 70

# Where the GitHub CLI's Windows installer puts gh; a shell started before the
# install has it outside PATH.
WINDOWS_GH = r"C:\Program Files\GitHub CLI\gh.exe"
INSTALL_HINT = "install the GitHub CLI from https://cli.github.com/ and run `gh auth login`"
GH_HOST = "github.com"

# `--timeout`'s default: the deadline of the whole answer, `--wait` included. The 8 runs of llm-loop's `tests` workflow before
# 2026-11-09 took 9.3-15.4 min from creation to the last update (15.4 min:
# run 42478828885, 10.9 min: run 42537657856), so 30 min is twice the slowest
# measured and still ends a hung run within one coffee.
WAIT_TIMEOUT_S = 1800
# `--wait`'s status poll. One `gh run view --json status` is one API request;
# 30 s keeps a 30-minute wait at 60 of the 5000 requests an hour a token has.
POLL_S = 30
# One gh call. `--log-failed` of the 12 red jobs of run 42525390878 took 1.5 s
# (2026-11-09); gh falls back to one request per job when the zip lacks job
# names, unmeasured here, so the ceiling is generous rather than tight. A call
# is also bounded by what is left of --timeout.
GH_TIMEOUT_S = 120
GIT_TIMEOUT_S = 30
# The most bytes of one log this tool reads. The whole `--log-failed` of the 12
# red jobs of run 42525390878 was 430 KB and one job's full log 29 KB
# (2026-11-09), so 32 MiB is ~75 times the largest measured. The parse holds a
# few copies of the text as Python strings (5-10x its bytes), so the limit
# keeps a runaway log (a test printing in a loop) at a few hundred MB instead
# of whatever gh streams. A log over it is not parsed: the report names the
# file it was saved to.
MAX_LOG_BYTES = 32 * 1024 * 1024

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
# A summary line can carry a whole repr; the printed headline is cut at this
# width (report.txt keeps it whole).
HEADLINE_CHARS = 240

# Job conclusions that are not a failure. Everything else a completed job can
# end with (failure, timed_out, cancelled, startup_failure, action_required,
# ...) is one.
PASSING = frozenset({"success", "skipped", "neutral"})
# Run conclusions that are red whatever the jobs say.
RED_CONCLUSIONS = frozenset({"failure", "cancelled", "timed_out", "startup_failure"})
FAILED_STEP = frozenset({"failure", "timed_out", "cancelled"})


class Failure(Exception):
    """A reason this tool cannot answer, with the exit code that names it."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class GhTimeout(Failure):
    """A gh call that did not answer within its timeout."""

    def __init__(self, message: str):
        super().__init__(EXIT_GH_ERROR, message)


class DeadlineHit(GhTimeout):
    """`--timeout` ran out before or during a gh call."""


def bounded(deadline: float, own: float, what: str) -> tuple[float, bool]:
    """(the timeout of one gh call, is it the deadline's) -- the call's own
    timeout or what is left before the monotonic `deadline`, the smaller.
    Raises DeadlineHit when nothing is left."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise DeadlineHit(f"--timeout ran out before gh {what}")
    return min(own, left), left < own


class CmdError(Exception):
    """A command that could not be started (OSError) or did not end in time."""

    def __init__(self, timed_out: bool, message: str):
        super().__init__(message)
        self.timed_out = timed_out


# ---------------------------------------------------------------- processes


def run_cmd(argv: list, timeout: float, cwd: str | None = None, stdout=None) -> subprocess.CompletedProcess:
    """`subprocess.run` with the script's text settings, or with stdout into the
    binary file `stdout` (stderr then comes back as bytes). Raises CmdError
    saying whether the executable could not start or the call timed out."""
    try:
        if stdout is None:
            return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout, cwd=cwd)
        return subprocess.run(argv, stdout=stdout, stderr=subprocess.PIPE, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as exc:
        raise CmdError(True, f"{argv[0]} did not answer within {timeout:g} s") from exc
    except OSError as exc:
        raise CmdError(False, f"cannot start {argv[0]}: {exc}") from exc


def as_text(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data or ""


def find_gh(explicit: str | None) -> str | None:
    """`--gh` (a path, or a name looked up on PATH), else gh on PATH, else the
    Windows installer's path; None if none exists."""
    if explicit:
        return explicit if os.path.isfile(explicit) else shutil.which(explicit)
    found = shutil.which("gh")
    if found:
        return found
    return WINDOWS_GH if os.path.isfile(WINDOWS_GH) else None


# gh's stderr for a run id or repository that does not exist (live, gh 2.x,
# 2026-11-09): "failed to get run: HTTP 404: Not Found (https://api...)". The
# API answers a private repository the token cannot read with the same 404.
NOT_FOUND_RE = re.compile(r"HTTP 404|could not find|not found|Could not resolve to a Repository", re.IGNORECASE)
# What gh says when the login is valid but may not read the repository.
NO_ACCESS_RE = re.compile(r"HTTP 403|Resource not accessible|SAML", re.IGNORECASE)
RATE_LIMIT_RE = re.compile(r"rate limit", re.IGNORECASE)


def gh_failure(what: str, returncode: int, said: str, repo: str) -> Failure:
    message = f"gh {what} failed (exit {returncode}): {said}"
    if RATE_LIMIT_RE.search(said):
        return Failure(EXIT_GH_ERROR, message)
    if NO_ACCESS_RE.search(said):
        return Failure(EXIT_NO_GH, f"{message}\nthe gh login cannot read {repo}: "
                                   "log in with an account that can (`gh auth login`) or check the token's scopes")
    if NOT_FOUND_RE.search(said):
        return Failure(EXIT_NOT_FOUND, f"{message}\n(a private repository the gh login cannot read "
                                       "is also answered with 404)")
    return Failure(EXIT_GH_ERROR, message)


class Gh:
    """gh for one repository; no call outlives the monotonic `deadline`."""

    def __init__(self, path: str, repo: str, deadline: float):
        self.path, self.repo, self.deadline = path, repo, deadline

    def _run(self, argv: list, what: str, timeout: float, stdout=None) -> subprocess.CompletedProcess:
        bound, by_deadline = bounded(self.deadline, timeout, what)
        try:
            proc = run_cmd(argv, bound, stdout=stdout)
        except CmdError as exc:
            if exc.timed_out and by_deadline:
                raise DeadlineHit(f"--timeout ran out during gh {what} ({bound:.0f} s were left)") from exc
            if exc.timed_out:
                raise GhTimeout(f"gh did not answer within {bound:g} s: {what}") from exc
            raise Failure(EXIT_GH_ERROR, f"{exc} (gh {what})") from exc
        if proc.returncode != 0:
            said = (as_text(proc.stderr) or as_text(proc.stdout)).strip()
            raise gh_failure(what, proc.returncode, said, self.repo)
        return proc

    def _run_to_file(self, argv: list, what: str, path: str, timeout: float) -> int:
        """`_run` with gh's stdout written to `path` (never held in memory); its size."""
        with open(path, "wb") as fh:
            self._run(argv, what, timeout, stdout=fh)
        return os.path.getsize(path)

    def call(self, args: list, timeout: float = GH_TIMEOUT_S) -> str:
        return as_text(self._run([self.path, *args, "--repo", self.repo], " ".join(args), timeout).stdout)

    def json(self, args: list, timeout: float = GH_TIMEOUT_S):
        out = self.call(args, timeout)
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise Failure(EXIT_GH_ERROR, f"gh {' '.join(args)} returned invalid JSON: {exc}") from exc

    def api_json(self, endpoint: str, jq: str):
        """`gh api endpoint --jq jq`, parsed (`gh api` takes no --repo: the
        endpoint names it)."""
        what = f"api {endpoint}"
        out = as_text(self._run([self.path, "api", endpoint, "--jq", jq], what, GH_TIMEOUT_S).stdout)
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise Failure(EXIT_GH_ERROR, f"gh {what} returned invalid JSON: {exc}") from exc

    def call_to_file(self, args: list, path: str, timeout: float = GH_TIMEOUT_S) -> int:
        return self._run_to_file([self.path, *args, "--repo", self.repo], " ".join(args), path, timeout)

    def job_log_to_file(self, job_id, path: str, timeout: float = GH_TIMEOUT_S) -> int:
        """One job's whole log into `path`; its size. Through the REST API,
        because `gh run view --job ID --log` refuses while the run is still in
        progress (gh 2.103.3, run 42546225202: "run ... is still in progress;
        logs will be available when it is complete") although the job's own log
        is complete. `gh api` refuses a body with escape sequences unless told;
        a gh older than that flag is retried without it."""
        endpoint = f"repos/{self.repo}/actions/jobs/{job_id}/logs"
        flag = "--allow-escape-sequences"
        try:
            return self._run_to_file([self.path, "api", endpoint, flag], f"api {endpoint}", path, timeout)
        except Failure as exc:
            if "unknown flag" not in str(exc):
                raise
        return self._run_to_file([self.path, "api", endpoint], f"api {endpoint}", path, timeout)


def check_auth(gh_path: str, deadline: float) -> None:
    """`gh auth status` must pass before any API call: logged out, gh answers
    a private repository with the same 404 as a missing run."""
    bound, by_deadline = bounded(deadline, GH_TIMEOUT_S, "auth status")
    try:
        proc = run_cmd([gh_path, "auth", "status", "--hostname", GH_HOST], bound)
    except CmdError as exc:
        if exc.timed_out and by_deadline:
            raise DeadlineHit(f"--timeout ran out during gh auth status ({bound:.0f} s were left)") from exc
        if exc.timed_out:
            raise Failure(EXIT_GH_ERROR, f"gh auth status: {exc}") from exc
        raise Failure(EXIT_NO_GH, f"{exc}; {INSTALL_HINT}") from exc
    if proc.returncode != 0:
        said = (proc.stderr or proc.stdout or "").strip()
        raise Failure(EXIT_NO_GH, f"gh auth status failed: {said}\n{INSTALL_HINT}")


def git(args: list, cwd: str | None) -> str | None:
    """One git call's stripped stdout, or None when it fails."""
    try:
        proc = run_cmd(["git", *args], GIT_TIMEOUT_S, cwd=cwd)
    except CmdError:
        return None
    if proc.returncode != 0:
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


# The shortest --sha prefix accepted: git's own default abbreviation. Shorter
# ones match several commits of a busy repository.
MIN_SHA_CHARS = 7
FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")
HEX_RE = re.compile(r"[0-9a-fA-F]+")


def derive_sha(sha: str | None, cwd: str | None) -> str:
    """The full sha of `--sha` (expanded through git when it resolves), a hex
    prefix of at least MIN_SHA_CHARS that git does not know, or HEAD's."""
    if sha:
        sha = sha.strip()
        if FULL_SHA_RE.fullmatch(sha):
            return sha.lower()
        if HEX_RE.fullmatch(sha) and len(sha) < MIN_SHA_CHARS:
            raise Failure(EXIT_USAGE, f"--sha {sha} is shorter than {MIN_SHA_CHARS} hex characters")
        full = git(["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"], cwd)
        if full and FULL_SHA_RE.fullmatch(full):
            return full.lower()
        if HEX_RE.fullmatch(sha):
            return sha.lower()
        raise Failure(EXIT_USAGE, f"--sha {sha} is neither a commit git knows here nor a hex prefix")
    head = git(["rev-parse", "HEAD"], cwd)
    if not head:
        raise Failure(EXIT_USAGE, "no --run/--sha given and the cwd is not a git repository")
    return head


def resolve_commit(gh: Gh, prefix: str) -> str:
    """The full sha of a hex prefix git could not expand, from the repository
    itself (`GET repos/R/commits/<prefix>`), so no listing window can hide a
    second commit with that prefix. GitHub answers an unknown and an
    ambiguous prefix alike: 422 "No commit found for SHA"."""
    try:
        # `{sha}`, not `.sha`: gh's --jq prints a bare string unquoted, not as JSON.
        data = gh.api_json(f"repos/{gh.repo}/commits/{prefix}", "{sha}")
    except Failure as exc:
        if "No commit found" in str(exc) or "HTTP 422" in str(exc):
            raise Failure(EXIT_NOT_FOUND, f"--sha {prefix}: no commit, or more than one, starts with it "
                                          f"in {gh.repo} (GitHub does not tell which); give the full sha") from exc
        raise
    full = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(full, str) or not FULL_SHA_RE.fullmatch(full):
        raise Failure(EXIT_GH_ERROR, f"gh api repos/{gh.repo}/commits/{prefix} gave no sha: {full!r}")
    return full.lower()


# The REST API's list of a commit's runs: 100 a page at most, and at most
# 1000 runs for a `head_sha` query in all (GitHub's docs of "List workflow
# runs for a repository": "up to 1,000 results for each search when using
# ... head_sha"). A commit with more cannot be judged whole.
RUNS_PAGE = 100
RUNS_MAX = 1000
RUNS_JQ = ("{total_count, workflow_runs: [.workflow_runs[] | {id, workflow_id, name, path, event, "
           "head_sha, created_at}]}")


def runs_for_commit(gh: Gh, sha: str) -> list:
    """Every run of the commit `sha` (40 hex), paged until the count the API
    states is reached. Raises EXIT_INCOMPLETE when it is not: over RUNS_MAX,
    or a page that came back short (runs created while paging shift them)."""
    runs, total, page = {}, None, 1
    while True:
        data = gh.api_json(f"repos/{gh.repo}/actions/runs?head_sha={sha}&per_page={RUNS_PAGE}&page={page}",
                           RUNS_JQ)
        batch = data.get("workflow_runs") if isinstance(data, dict) else None
        total = data.get("total_count") if isinstance(data, dict) else None
        if not isinstance(batch, list) or not isinstance(total, int):
            raise Failure(EXIT_GH_ERROR, f"the run listing of {sha} is not the API's shape: {str(data)[:200]}")
        for r in batch:
            runs[r.get("id")] = r
        if len(runs) >= total or not batch or page * RUNS_PAGE >= RUNS_MAX:
            break
        page += 1
    if len(runs) < total:
        raise Failure(EXIT_INCOMPLETE, f"commit {sha} has {total} runs and only {len(runs)} could be listed "
                                       f"(the API serves {RUNS_MAX} at most): a red one may be missing; "
                                       "judge one with --run")
    return list(runs.values())


def workflow_matches(run: dict, workflow: str) -> bool:
    """`--workflow` as gh takes it: the name, the file (`tests.yml` or its
    path) or the workflow id."""
    path = str(run.get("path") or "")
    return workflow in (run.get("name"), path, path.rsplit("/", 1)[-1], str(run.get("workflow_id")))


def select_runs(gh: Gh, sha: str, workflow: str | None) -> list:
    """The ids of the runs to judge for `sha`: the newest run of every
    (workflow id, event) pair among ALL the commit's runs (of `workflow` only,
    when given), newest first. By id, not name: two workflow files may share a
    display name, and the newest run of one must not hide the other's."""
    full = sha if FULL_SHA_RE.fullmatch(sha) else resolve_commit(gh, sha)
    runs = runs_for_commit(gh, full)
    if workflow:
        runs = [r for r in runs if workflow_matches(r, workflow)]
    if not runs:
        what = f" of workflow {workflow!r}" if workflow else ""
        raise Failure(EXIT_NOT_FOUND, f"no run{what} for commit {full} in {gh.repo} (not pushed yet?)")
    runs.sort(key=lambda r: (str(r.get("created_at") or ""), int(r.get("id") or 0)), reverse=True)
    chosen, seen = [], set()
    for r in runs:
        key = (r.get("workflow_id"), r.get("event"))
        if key not in seen:
            seen.add(key)
            chosen.append(int(r["id"]))
    return chosen


VIEW_FIELDS = "databaseId,workflowName,event,headSha,status,conclusion,url,displayTitle,attempt,jobs"


def fetch_run(gh: Gh, run_id: int) -> dict:
    return gh.json(["run", "view", str(run_id), "--json", VIEW_FIELDS])


def wait_run(gh: Gh, run: dict) -> dict:
    """Poll until `run` completes or gh's deadline passes; the last view
    either way. A poll the deadline cuts short ends the wait."""
    status = run.get("status")
    left = gh.deadline - time.monotonic()
    print(f"waiting for run {run.get('databaseId')} ({status}) up to {max(left, 0):.0f} s",
          file=sys.stderr, flush=True)
    while run.get("status") != "completed":
        left = gh.deadline - time.monotonic()
        if left <= 0:
            break
        time.sleep(min(POLL_S, left))
        try:
            run = fetch_run(gh, run["databaseId"])
        except DeadlineHit:
            break
        if run.get("status") != status:
            status = run.get("status")
            print(f"  run {run.get('databaseId')} is {status}", file=sys.stderr, flush=True)
    return run


# ---------------------------------------------------------------- the log: pytest's output


# `gh run view --log-failed` prefixes every line with `<job>\t<step>\t`. The
# step is "UNKNOWN STEP" on every line of the live runs (gh says why in its
# help: the zip does not always map lines to steps), so the failed steps come
# from the jobs JSON instead. The text starts with the runner's ISO timestamp;
# a job's first line carries a BOM before it. A job's own log (the API's) has
# the same lines without the prefix.
LINE_RE = re.compile(r"^([^\t]*)\t([^\t]*)\t(.*)$")
TS_RE = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ?")
# CSI sequences, and OSC ones (hyperlinks, titles) ended by BEL or by ST (ESC \).
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
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


def split_job_log(text: str) -> list:
    """The raw lines of one job's own log (no job/step prefix)."""
    lines = NEWLINE_RE.split(text.lstrip("\ufeff"))
    while lines and not lines[-1]:
        lines.pop()
    return lines


def clean(line: str) -> str:
    """A raw log line without its timestamp and colour codes."""
    return ANSI_RE.sub("", TS_RE.sub("", line, count=1)).rstrip()


BANNER_RE = re.compile(r"^=+ (.+?) =+$")
BLOCK_RE = re.compile(r"^_+ (.+?) _+$")
ENTRY_RE = re.compile(r"^(FAILED|ERROR) (\S.*)$")
OUTCOME_RE = re.compile(
    r"^=*\s*(?:\d+ (?:failed|passed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b"
    r"|no tests ran\b).*\bin \d+(?:\.\d+)?s\b")
# pytest's own section banners that may follow its FAILURES/ERRORS blocks. A
# block is a test's whole report, captured output included, and a test that
# runs pytest in a child prints banners and outcome lines of its own: inside
# the blocks only these names (and the final outcome banner) end a section.
SECTION_BANNERS = frozenset({"FAILURES", "ERRORS", "warnings summary", "short test summary info",
                             "test session starts", "PASSES", "rerun test summary info"})
SLOWEST_RE = re.compile(r"^slowest (?:\d+ )?durations$")
# `!!!!! stopping after 1 failures !!!!!`, `!!! Interrupted: ... !!!`.
STOP_RE = re.compile(r"^!{3,} .* !{3,}$")


# pytest's header of a block's captured output: `---- Captured stdout call ----`
# (stdout/stderr/log, setup/call/teardown). What follows is the test's output
# verbatim, unindented, so a child pytest's banners look like the outer run's.
CAPTURED_RE = re.compile(r"^-+ Captured .+ -+$")
SESSION_START = "test session starts"


def child_session_end(lines: list, start: int) -> int | None:
    """The index of the outcome line closing the child pytest session whose
    `test session starts` banner is `lines[start]`, or None.

    pytest opens every session (outer or child) with that banner and closes
    it with its outcome line (`=== 1 failed, 1 passed in 0.09s ===`), and the
    outer session's banner comes once, before any FAILURES: inside a block's
    captured output it can only be a child's. The child's lines up to its
    outcome (its FAILURES, its blocks, its summary) are the block's, not
    sections of the outer run. Nested children count. Not a close: the last
    outcome line before the step's end (`##[`) -- that one is the outer run's
    own, so a child that died before its outcome closes nothing: its banner
    is a line of the block, and whatever banners it printed after it end the
    block as before. A `-q` child prints no session banner and is not
    recognised."""
    stop = next((j for j in range(start, len(lines)) if lines[j].startswith("##[")), len(lines))
    last_outcome = max((j for j in range(start, stop) if OUTCOME_RE.match(lines[j])), default=None)
    depth = 0
    for j in range(start, stop):
        banner = BANNER_RE.match(lines[j])
        if banner and banner.group(1).strip() == SESSION_START:
            depth += 1
        elif OUTCOME_RE.match(lines[j]):
            depth -= 1
            if depth == 0:
                return j if j != last_outcome else None
    return None


def drop_trailing_blanks(lines: list) -> None:
    while lines and not lines[-1].strip():
        lines.pop()


def is_section_banner(line: str, name: str) -> bool:
    return name in SECTION_BANNERS or bool(SLOWEST_RE.match(name)) or bool(OUTCOME_RE.match(line))


def parse_pytest(lines: list) -> tuple[list, list, list, int]:
    """(blocks, summary entries, outcome lines, index of pytest's last line) of
    cleaned log lines; the index is -1 when there is no pytest output.

    A block is (title, lines) of a `= FAILURES =` or `= ERRORS =` section,
    split at pytest's `___ title ___` headers (not at `_ _ _ _`, its separator
    between chained tracebacks). Inside the blocks a section ends only at a
    SECTION_BANNERS banner, the final outcome banner or a runner `##[` line;
    `-q` prints the outcome line without a banner, so an outcome-looking line
    counts only when it is the last line before the `##[` line (else it is a
    test's captured output). A child pytest session in a block's captured
    output stays in the block whole (`child_session_end`). An entry is a
    `short test summary info` line with its continuation lines: a multi-line
    message goes on unprefixed. Several pytest runs in one job simply add up."""
    blocks, entries, outcomes = [], [], []
    section, end, pending = None, -1, None
    captured, skip_to = False, -1

    def close_blocks():
        # The unbannered `-q` outcome line is the last of the last block.
        if pending is not None and blocks:
            body = blocks[-1][1]
            drop_trailing_blanks(body)
            if body and body[-1] == lines[pending]:
                body.pop()
                outcomes.append(lines[pending].strip("= "))

    for i, line in enumerate(lines):
        if i <= skip_to:
            continue
        banner = BANNER_RE.match(line)
        name = banner.group(1).strip() if banner else ""
        if section == "blocks":
            if line.startswith("##["):
                close_blocks()
                pending, section = None, None
                continue
            if captured and blocks and name == SESSION_START:
                # Never the outer run's banner here; without its close, just a line.
                close = child_session_end(lines, i)
                close = i if close is None else close
                blocks[-1][1].extend(lines[i:close + 1])
                skip_to = end = close
                continue
            if not (banner and is_section_banner(line, name)):
                head = BLOCK_RE.match(line)
                if head and not re.fullmatch(r"[_ ]+", head.group(1)):
                    blocks.append((head.group(1), []))
                    captured = False
                elif blocks:
                    captured = captured or bool(CAPTURED_RE.match(line))
                    blocks[-1][1].append(line)
                    if OUTCOME_RE.match(line):
                        pending = i
                end = i
                continue
            pending = None
        if OUTCOME_RE.match(line):
            outcomes.append(line.strip("= "))
            section, end = None, i
            continue
        if banner:
            section = ("blocks" if name in ("FAILURES", "ERRORS")
                       else "summary" if name == "short test summary info" else None)
            if is_section_banner(line, name):
                end = i
            continue
        if STOP_RE.match(line):
            section, end = None, i
            continue
        if line.startswith("##["):
            section = None
            continue
        if section == "summary":
            if ENTRY_RE.match(line):
                entries.append([line])
            elif entries:
                entries[-1].append(line)
            end = i
    if section == "blocks":
        close_blocks()
    for _, body in blocks:
        drop_trailing_blanks(body)
        while body and not body[0].strip():
            body.pop(0)
    return blocks, entries, outcomes, end


def entry_parts(entry_line: str) -> tuple[str, str] | None:
    """`FAILED tests/x.py::T::t[a] - msg` -> (`FAILED`, `tests/x.py::T::t[a]`);
    None for a line that is not a summary entry."""
    m = ENTRY_RE.match(entry_line)
    return (m.group(1), m.group(2).split(" - ", 1)[0].strip()) if m else None


def block_matches(title: str, nodeid: str, kind: str) -> bool:
    """Does a block titled `title` belong to the `kind` (FAILED/ERROR) summary
    entry of `nodeid`? Titles drop the file and join a class to its test with
    a dot. A FAILED entry owns the block titled with the test only; an ERROR
    entry owns `ERROR at <phase> of <test>` or `ERROR collecting <path>` -- a
    test failing in its call and erroring in teardown has one of each, and
    pytest prints the ERRORS section first."""
    test = ".".join(nodeid.split("::")[1:])
    if kind == "FAILED":
        return bool(test) and title == test
    if test:
        return re.fullmatch(r"ERROR at \w+ of " + re.escape(test), title) is not None
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


# The runner's end of a failed step, which follows pytest's outcome line.
EXIT_LINE_RE = re.compile(r"^##\[error\]Process completed with exit code \d+\.$")


def fails_after_pytest(lines: list, end: int) -> bool:
    """Does the log go on failing after pytest's last line? True when, up to
    its last `##[error]`, something other than the step's own exit-code line
    follows: a second failed step, a crash, a timeout message."""
    after = lines[end + 1:]
    last = max((i for i, ln in enumerate(after) if ln.startswith("##[error]")), default=None)
    if last is None:
        return False
    after = after[:last + 1]
    errors = [ln for ln in after if ln.startswith("##[error]")]
    other = [ln for ln in after if ln.strip() and not ln.startswith("##[")]
    return len(errors) > 1 or bool(other) or not EXIT_LINE_RE.match(errors[0])


def failure_units(lines: list, tail: int, failed_steps: int = 1, source: str = "log") -> tuple[list, list]:
    """(units, outcome lines) of one failed job's cleaned lines.

    A unit is {"kind", "headline", "block"}: a summary entry with its block,
    a block without an entry, an entry without a block -- or, when the job has
    no pytest failures at all, one "tail" unit of its last lines. A job whose
    log goes on failing after pytest, or with more than one failed step, gets
    a tail unit of what follows pytest as well."""
    blocks, entries, outcomes, end = parse_pytest(lines)
    units, used = [], set()
    for entry in entries:
        kind, nodeid = entry_parts(entry[0])
        index = next((i for i, (title, _) in enumerate(blocks)
                      if i not in used and block_matches(title, nodeid, kind)), None)
        if index is None:
            units.append({"kind": "pytest", "headline": entry[0], "block": entry[1:]})
        else:
            used.add(index)
            units.append({"kind": "pytest", "headline": f"{kind} {nodeid}", "block": blocks[index][1]})
    for i, (title, body) in enumerate(blocks):
        if i not in used:
            units.append({"kind": "pytest", "headline": title, "block": body})
    if not units:
        units.append({"kind": "tail", "headline": f"no pytest failures in this job's {source}; its last lines",
                      "block": tail_before_error(lines, tail)})
    elif failed_steps > 1 or fails_after_pytest(lines, end):
        why = f"{failed_steps} failed steps" if failed_steps > 1 else "an error after pytest's outcome"
        units.append({"kind": "tail", "headline": f"the job's {source} after pytest ({why}); its last lines",
                      "block": tail_before_error(lines[end + 1:], tail) or tail_before_error(lines, tail)})
    return units, outcomes


# ---------------------------------------------------------------- the dedup across jobs


# What `normalize` replaces, in order. Only diagnostic metadata that one
# failure shows differently between matrix jobs (all seen in llm-loop's runs of
# 2026-11-08 unless said otherwise): paths of the checkout, the interpreter and
# the temp dirs, object addresses. Never values: `t[5s]` and `t[10s]`, `0x20`
# and `0x30`, `< 2 s` and `< 9 s` are different failures, and a key that
# merged them would lose one.
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
    # tempfile's names (`tmp` + 8 characters) as a path component.
    (re.compile(r"(?<=[/'\"])tmp[a-z0-9_]{8}(?![A-Za-z0-9_])"), "tmp*"),
    (re.compile(r"\b(?:Posix|Windows)Path\("), "Path("),
    # The address in a default repr: `<Foo object at 0x1059b1f70>`.
    (re.compile(r"(\bat )0x[0-9a-fA-F]+(?=>)"), r"\g<1>0x?"),
]
# pytest's own timing total in an outcome line (a child pytest's, captured).
TOTALS_RE = re.compile(r"\bin \d+(?:\.\d+)?s(?: \(\d+:\d\d:\d\d\))?")
# pytest's `name = repr` lines of a traceback entry's arguments (its
# ReprFuncArgs: `a = 1, b = [2]`, a long repr alone on its line, then a blank
# line), printed first in the entry: at the block's start or after the `_ _ _`
# entry separator or a chained exception's message. A parametrised value is in
# the node id, compared exactly; what is left can be a fixture whose repr
# carries the wall clock: run 42525390878 split one red over 3 jobs only by
# `network_clock = [' . RUNNING since 2026-11-08 21:28:59 (... since 21:28 ...)']`.
# Clock times are taken out of such argument values only, and only of a
# container or object repr (`[...]`, `(...)`, `{...}`, `<...>`, `Name(...)`):
# a plain literal (`at = '12:30'`) is a value under test, and so is a line of
# captured output (it comes after the traceback, not at an entry's head), an
# `E` line or a headline.
ARG_LINE_RE = re.compile(r"^[A-Za-z_]\w* = ")
ARG_NEXT_RE = re.compile(r", [A-Za-z_]\w* = ")
ENTRY_SEP_RE = re.compile(r"^_(?: _)+$")
CHAIN_RE = re.compile(r"^(?:The above exception was the direct cause of the following exception:"
                      r"|During handling of the above exception, another exception occurred:)$")
OBJECT_REPR_RE = re.compile(r"^(?:[\[({<]|[A-Za-z_][\w.]*\()")
CLOCK_RES = [(re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?Z?"), "<time>"),
             (re.compile(r"\b\d\d?:\d\d(?::\d\d)?\b"), "<hh:mm>")]


def split_args(line: str) -> list | None:
    """[[name, value]] of a ReprFuncArgs line, split at `, name = ` outside
    quotes and brackets only; None for a line that is not one."""
    if not ARG_LINE_RE.match(line):
        return None
    parts, start, depth, quote, k = [], 0, 0, None, 0
    while k < len(line):
        c = line[k]
        if quote:
            if c == "\\":
                k += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth = max(depth - 1, 0)
        elif depth == 0 and ARG_NEXT_RE.match(line, k):
            parts.append(line[start:k])
            start = k + 2
        k += 1
    parts.append(line[start:])
    return [p.split(" = ", 1) for p in parts]


def without_clock_times(args: list) -> str:
    """An argument line again, clock times out of its object reprs."""
    out = []
    for name, value in args:
        if OBJECT_REPR_RE.match(value):
            for pattern, repl in CLOCK_RES:
                value = pattern.sub(repl, value)
        out.append(f"{name} = {value}")
    return ", ".join(out)
# Windows paths, made comparable to POSIX ones: drive-letter paths (also
# inside a repr, where the backslashes are doubled), pytest's location line
# `tests\x.py:74: Error` and a traceback frame's file.
DRIVE_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s'\"]*")
LOCATION_RE = re.compile(r"^[\w.\-\\/]+\.py(?=:\d+: )")
FRAME_FILE_RE = re.compile(r'(File ")([^"]*)(")')
# Python 3.11+ underlines the failing expression with ^/~ lines; 3.9 and 3.10
# print none, so they are left out of the comparison.
CARETS_RE = re.compile(r"^\s*(?:E\s+)?[\^~][\s\^~]*$")
# A traceback frame line, plain or inside pytest's `E` lines.
FRAME_RE = re.compile(r'^(E?\s*)File "[^"]*", line \d+')


def posix_paths(line: str) -> str:
    line = DRIVE_PATH_RE.sub(lambda m: re.sub(r"\\+", "/", m.group(0)), line)
    line = LOCATION_RE.sub(lambda m: m.group(0).replace("\\", "/"), line)
    return FRAME_FILE_RE.sub(lambda m: m.group(1) + re.sub(r"\\+", "/", m.group(2)) + m.group(3), line)


def drop_frame_sources(lines: list) -> list:
    """`lines` without the source lines under each `File "...", line N` frame.

    Interpreters print a frame's source differently: 3.13 shows every line of
    a multi-line call and the source of a `-c` script, 3.12 only the first
    line and nothing for `<string>` (one red test of run 42537657856 printed
    both ways). The frames themselves and the exception line are the same, so
    the key keeps those: a frame's source is what is indented deeper than its
    `File` line with the same `E` prefix or none, up to the next frame or the
    first line that is not. A plain frame's following `E` lines are pytest's
    verdict, not its source, and stay."""
    out, frame_indent, frame_e = [], None, False
    for line in lines:
        frame = FRAME_RE.match(line)
        if frame:
            frame_indent, frame_e = len(frame.group(1)), frame.group(1).startswith("E")
            out.append(line)
            continue
        if frame_indent is not None:
            lead = re.match(r"^(E?)\s*", line)
            same_kind = (lead.group(1) == "E") == frame_e
            if same_kind and len(lead.group(0)) > frame_indent and line.strip() not in ("", "E"):
                continue
            frame_indent = None
        out.append(line)
    return out


def normalize(text: str) -> str:
    """The comparison key of a failure's text, without per-job noise. Never
    shown -- the report shows the first job's exact text."""
    out, entry_head = [], True  # a block starts with its first entry's arguments
    lines = [posix_paths(ln) for ln in text.split("\n") if not CARETS_RE.match(ln)]
    for line in drop_frame_sources(lines):
        for pattern, repl in NOISE:
            line = pattern.sub(repl, line)
        if OUTCOME_RE.match(re.sub(r"^E\s+", "", line)):
            line = TOTALS_RE.sub("in <dur>", line)
        args = split_args(line) if entry_head else None
        if args:
            line = without_clock_times(args)
        elif ENTRY_SEP_RE.match(line) or CHAIN_RE.match(line):
            entry_head = True
        elif line.strip():
            entry_head = False
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def group_key(unit: dict) -> str:
    """A unit's identity: its kind and node id (or block title) exactly, the
    rest of its headline and its block through `normalize`."""
    head, rest = unit["headline"], ""
    parts = entry_parts(head)
    if unit["kind"] == "pytest" and parts:
        ident = " ".join(parts)
        head, rest = ident, head[len(ident):]
    # Apart: the block's first line is its first entry's head (`normalize`);
    # the rest (` - message`) starts with a space, so it is never an argument line.
    # A saved log is per job: two jobs' unparsed logs are two groups, each with its path.
    return "\0".join([unit["kind"], head, normalize(rest), normalize("\n".join(unit["block"])),
                      unit.get("savedLog") or ""])


def group_failures(per_job: list) -> list:
    """[(job, units)] -> groups of identical units: {"headline", "block",
    "kind", "jobs", "shownFrom"}, the most widespread first, ties in order of
    first appearance."""
    groups: dict = {}
    for job, units in per_job:
        for unit in units:
            key = group_key(unit)
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


# ---------------------------------------------------------------- the verdict


def failed_steps(job: dict) -> list:
    return [s.get("name") for s in job.get("steps") or [] if s.get("conclusion") in FAILED_STEP]


def is_failed(job: dict) -> bool:
    return (job.get("status") == "completed" and bool(job.get("conclusion"))
            and job.get("conclusion") not in PASSING)


def verdict(run: dict) -> tuple[int, str | None]:
    """(exit code, why there is no green) of one run's view. Green needs all of
    it: completed, concluded success, a job list, every job finished without a
    failure and at least one job that ran."""
    if run.get("status") != "completed":
        return EXIT_RUNNING, f"run is {run.get('status') or 'in an unknown state'}"
    jobs = run.get("jobs") or []
    conclusion = run.get("conclusion") or ""
    if any(is_failed(j) for j in jobs) or conclusion in RED_CONCLUSIONS:
        if not any(is_failed(j) for j in jobs):
            return EXIT_RED, f"run concluded {conclusion} but lists no failed job: nothing to show from the logs"
        return EXIT_RED, None
    if not jobs:
        return EXIT_INCOMPLETE, f"run concluded {conclusion or 'nothing'} but lists no jobs: no verdict"
    unfinished = [j.get("name") for j in jobs if j.get("status") != "completed" or not j.get("conclusion")]
    if unfinished:
        return EXIT_INCOMPLETE, f"run is completed but these jobs have no conclusion: {', '.join(map(str, unfinished))}"
    if conclusion != "success":
        return EXIT_INCOMPLETE, f"run concluded {conclusion or 'nothing'}: neither success nor failure"
    if not any(j.get("conclusion") == "success" for j in jobs):
        return EXIT_INCOMPLETE, "no job of the run succeeded (all skipped or neutral): nothing was tested"
    return EXIT_GREEN, None


def combined_code(codes: list) -> int:
    """Red if any run is red, else running if any runs, else no verdict if any
    has none, else green."""
    for code in (EXIT_RED, EXIT_RUNNING, EXIT_INCOMPLETE):
        if code in codes:
            return code
    return EXIT_GREEN


# ---------------------------------------------------------------- the report


def job_line(job: dict, outcomes: list | None = None) -> str:
    state = job.get("conclusion") or job.get("status") or "?"
    line = f"{job.get('name')}: {state}"
    steps = failed_steps(job)
    if steps:
        line += f" @ {', '.join(map(str, steps))}"
    if outcomes:
        line += f" -- {'; '.join(outcomes)}"
    return line


def run_title(run: dict) -> str:
    sha = str(run.get("headSha", ""))[:8]
    state = run.get("status") if run.get("status") != "completed" else run.get("conclusion")
    event = f" [{run['event']}]" if run.get("event") else ""
    title = run.get("displayTitle")
    return (f"run {run.get('databaseId')} {run.get('workflowName')!r}{event} {sha}: {state}"
            + (f" ({title})" if title else ""))


def cut_headline(headline: str, cap: bool) -> str:
    if cap and len(headline) > HEADLINE_CHARS:
        return headline[:HEADLINE_CHARS] + "..."
    return headline


def group_jobs(group: dict, labels: dict) -> str:
    return ", ".join(labels.get(j, j) for j in group["jobs"])


def failed_labels(run: dict) -> tuple[list, dict]:
    failed = [j.get("name") for j in run.get("jobs") or [] if is_failed(j)]
    return failed, short_names(failed)


def saved_log_line(group: dict, indent: str) -> list:
    """A path the reader acts on, on its own line: never inside a headline,
    which the print cuts at HEADLINE_CHARS."""
    return [f"{indent}saved log: {group['savedLog']}"] if group.get("savedLog") else []


def render_group(index: int, group: dict, labels: dict, cap: bool, report_path: str) -> list:
    out = [f"[{index}] {cut_headline(group['headline'], cap)} -- {group_jobs(group, labels)}"]
    out.extend(saved_log_line(group, "    "))
    block = group["block"]
    if len(group["jobs"]) > 1:
        out.append(f"    (text from {labels.get(group['shownFrom'], group['shownFrom'])})")
    if cap and len(block) > BLOCK_HEAD + BLOCK_TAIL:
        omitted = len(block) - BLOCK_HEAD - BLOCK_TAIL
        block = (block[:BLOCK_HEAD] + [f"... {omitted} lines omitted, full block in {report_path}"]
                 + block[-BLOCK_TAIL:])
    out.extend("    " + ln for ln in block)
    return out


def render_head(rep: dict, raw_dir: str, cap: bool) -> list:
    """One run's jobs, its note, the index of its distinct failures and its
    saved logs -- the part of the report the print never cuts."""
    run = rep["run"]
    out = [run_title(run), str(run.get("url", ""))]
    for job in run.get("jobs") or []:
        out.append(job_line(job, rep["outcomes"].get(job.get("name"))))
    if rep["note"]:
        out.append(rep["note"])
    groups = rep["groups"]
    if groups:
        failed, labels = failed_labels(run)
        out.append("")
        out.append(f"{len(groups)} distinct failure(s) in {len(failed)} failed job(s):")
        for i, group in enumerate(groups, 1):
            out.append(f"  [{i}] {cut_headline(group['headline'], cap)} -- {group_jobs(group, labels)}")
            out.extend(saved_log_line(group, "      "))
    if rep["raw"]:
        out.append("raw logs: " + ", ".join(os.path.basename(p) for p in rep["raw"].values()))
        out.append(f"  in {raw_dir}")
    return out


def render_blocks(rep: dict, report_path: str, cap: bool, titled: bool) -> list:
    """One run's failure blocks, numbered as in its index."""
    groups = rep["groups"]
    if not groups:
        return []
    _, labels = failed_labels(rep["run"])
    out = [f"-- the failures of run {rep['run'].get('databaseId')}"] if titled else []
    for i, group in enumerate(groups, 1):
        if out:
            out.append("")
        out.extend(render_group(i, group, labels, cap, report_path))
    return out


def render(reports: list, raw_dir: str, report_path: str, cap: bool) -> tuple[list, list]:
    """(head, detail): every run's head first, then every run's blocks, so
    that a long first run cannot push the second run's index out of a cut
    print; `emit` cuts the detail only."""
    head = []
    if len(reports) > 1:
        sha = str(reports[0]["run"].get("headSha", ""))[:8]
        head.append(f"{len(reports)} runs for commit {sha}:")
        head.extend("  " + run_title(rep["run"]) for rep in reports)
    for rep in reports:
        if head:
            head.append("")
        head.extend(render_head(rep, raw_dir, cap))
    detail = []
    for rep in reports:
        blocks = render_blocks(rep, report_path, cap, titled=len(reports) > 1)
        if blocks:
            detail.append("")
            detail.extend(blocks)
    return head, detail


def run_json(rep: dict) -> dict:
    run = rep["run"]
    return {"run": {k: run.get(k) for k in ("databaseId", "workflowName", "event", "headSha", "status",
                                            "conclusion", "url", "displayTitle", "attempt")},
            "exit": rep["code"], "note": rep["note"],
            "jobs": [job_json(j, rep["outcomes"], rep["raw"]) for j in run.get("jobs") or []],
            "failures": [{k: g.get(k) for k in ("kind", "headline", "jobs", "shownFrom", "block", "savedLog")}
                         for g in rep["groups"]]}


def job_json(job: dict, outcomes: dict, raw: dict) -> dict:
    name = job.get("name")
    return {"name": name, "status": job.get("status"), "conclusion": job.get("conclusion"),
            "failedSteps": failed_steps(job), "outcome": outcomes.get(name, []), "rawLog": raw.get(name)}


# ---------------------------------------------------------------- reading the failed jobs' logs


def safe_name(name: str, taken: set) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "job"
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}_{n}"
    taken.add(candidate)
    return candidate + ".log"


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def over_limit(what: str, size: int, path: str) -> list:
    """The unit of a log too big to parse; its path goes on its own line."""
    return missing(f"{what} is {size} bytes, over the {MAX_LOG_BYTES}-byte limit "
                   "(MAX_LOG_BYTES): not parsed; the whole log is saved", path)


def missing(headline: str, saved_log: str | None = None) -> list:
    return [{"kind": "missing", "headline": headline, "block": [], "savedLog": saved_log}]


def analyse(gh: Gh, run: dict, raw_dir: str, tail: int, prefix: str, taken: set) -> tuple[dict, list, dict]:
    """(pytest outcome lines by job, failure groups, raw log path by job) of
    the failed jobs of `run`, writing each one's whole log into `raw_dir`.

    A completed run's logs come from `--log-failed`; a job missing there (a
    cancelled or timed-out job often is) and every job of a run still in
    progress is read from its own log instead."""
    failed = [j for j in run.get("jobs") or [] if is_failed(j)]
    outcomes, per_job, raw = {}, [], {}
    if not failed:
        return outcomes, [], raw
    run_id = run.get("databaseId")
    by_job, combined = {}, None
    if run.get("status") == "completed":
        path = os.path.join(raw_dir, f"{prefix}log-failed.txt")
        try:
            size = gh.call_to_file(["run", "view", str(run_id), "--log-failed"], path)
        except DeadlineHit as exc:
            # The verdict is known; the failures' text is not -- said per job.
            combined = missing(f"`gh run view --log-failed` not read: {exc}")
        else:
            if size > MAX_LOG_BYTES:
                combined = over_limit("`gh run view --log-failed`", size, path)
            else:
                by_job = split_log(read_text(path))
                os.remove(path)
    for job in failed:
        name = job.get("name")
        if combined:
            per_job.append((name, combined))
            continue
        path = os.path.join(raw_dir, safe_name(prefix + str(name), taken))
        lines, source = by_job.get(name), "log"
        if lines:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(lines) + "\n")
        else:
            job_id = job.get("databaseId")
            where = "no lines in `gh run view --log-failed`" if run.get("status") == "completed" else "run in progress"
            if job_id is None:
                per_job.append((name, missing(f"{where} and no job id to read the job's own log")))
                continue
            try:
                size = gh.job_log_to_file(job_id, path)
            except Failure as exc:
                per_job.append((name, missing(f"{where}; its own log could not be read: {exc}")))
                continue
            if size > MAX_LOG_BYTES:
                raw[name] = path
                per_job.append((name, over_limit("the job's own log", size, path)))
                continue
            lines, source = split_job_log(read_text(path)), "own log"
        raw[name] = path
        units, outs = failure_units([clean(ln) for ln in lines], tail, len(failed_steps(job)), source)
        outcomes[name] = outs
        per_job.append((name, units))
    return outcomes, group_failures(per_job), raw


# ---------------------------------------------------------------- the command line


def emit(head: list, detail: list, max_lines: int, report_path: str) -> None:
    """Print the head whole and as much of the detail as `max_lines` leaves;
    the last line names the full report, on its own."""
    room = max_lines - len(head) - 1
    if len(detail) > room:
        shown = detail[:max(room - 1, 0)]
        detail = shown + [f"... failure blocks cut at {len(shown)} of {len(detail)} lines (--max-lines {max_lines})"]
    print("\n".join(head + detail + [f"full report: {report_path}"]))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--run", type=int, default=None, metavar="ID", help="the run's database id")
    pick.add_argument("--sha", default=None,
                      help="the newest runs for this commit (default: git HEAD of the cwd); "
                           f"a hex prefix needs {MIN_SHA_CHARS}+ characters")
    ap.add_argument("--repo", default=None, metavar="OWNER/NAME",
                    help="default: from `git remote get-url origin` of the cwd")
    ap.add_argument("--workflow", default=None, metavar="NAME",
                    help="only this workflow (name, file or id): the newest run of each event "
                         "(default: of every workflow and event)")
    ap.add_argument("--wait", action="store_true", help="wait for unfinished runs to complete")
    ap.add_argument("--timeout", type=float, default=WAIT_TIMEOUT_S,
                    help="seconds for the whole answer, every gh call and --wait included "
                         f"(default {WAIT_TIMEOUT_S}, `WAIT_TIMEOUT_S`)")
    ap.add_argument("--raw-dir", default=None, metavar="DIR",
                    help="where the failed jobs' whole logs and report.txt go (default: a new temp dir "
                         "named by the run ids, kept)")
    ap.add_argument("--max-lines", type=int, default=MAX_LINES, metavar="N",
                    help=f"cap of the printed report (default {MAX_LINES}); it cuts the failure blocks "
                         "only, never the jobs and the index; report.txt has it all")
    ap.add_argument("--tail", type=int, default=TAIL_LINES, metavar="N",
                    help=f"lines kept of a failed job without pytest failures (default {TAIL_LINES})")
    ap.add_argument("--json", action="store_true", help="print jobs and grouped failures as JSON")
    ap.add_argument("--gh", default=None, metavar="PATH", help="the gh executable (default: PATH, then "
                    + WINDOWS_GH + ")")
    return ap


def answer(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    gh_path = find_gh(args.gh)
    if gh_path is None:
        where = args.gh or f"PATH or {WINDOWS_GH}"
        raise Failure(EXIT_NO_GH, f"gh not found in {where}; {INSTALL_HINT}")
    cwd = os.getcwd()
    repo = args.repo or derive_repo(cwd)
    check_auth(gh_path, deadline)
    gh = Gh(gh_path, repo, deadline)
    if args.run is not None:
        run_ids = [args.run]
    else:
        run_ids = select_runs(gh, derive_sha(args.sha, cwd), args.workflow)
    runs = [fetch_run(gh, run_id) for run_id in run_ids]
    if args.wait:
        runs = [wait_run(gh, r) if r.get("status") != "completed" else r for r in runs]
    ids = "_".join(str(i) for i in run_ids[:4])
    raw_dir = os.path.abspath(args.raw_dir or tempfile.mkdtemp(prefix=f"ci_failures_{ids}_"))
    os.makedirs(raw_dir, exist_ok=True)
    report_path = os.path.join(raw_dir, "report.txt")
    reports, taken = [], set()
    for run in runs:
        code, note = verdict(run)
        if code == EXIT_RUNNING:
            note += f" after waiting {args.timeout:g} s" if args.wait else "; pass --wait to wait"
        prefix = f"{run.get('databaseId')}_" if len(runs) > 1 else ""
        outcomes, groups, raw = analyse(gh, run, raw_dir, args.tail, prefix, taken)
        reports.append({"run": run, "code": code, "note": note, "outcomes": outcomes,
                        "groups": groups, "raw": raw})
    code = combined_code([rep["code"] for rep in reports])
    head, detail = render(reports, raw_dir, report_path, cap=False)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(head + detail) + "\n")
    if args.json:
        print(json.dumps({"schema": JSON_SCHEMA, "exit": code, "rawDir": raw_dir, "report": report_path,
                          "runs": [run_json(rep) for rep in reports]}))
    else:
        head, detail = render(reports, raw_dir, report_path, cap=True)
        emit(head, detail, args.max_lines, report_path)
    return code


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
    except Exception:  # a defect here must not read as a red run
        traceback.print_exc()
        print(f"error: internal error in ci_failures.py (exit {EXIT_INTERNAL}); the run was not judged",
              file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
